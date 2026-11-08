"""TEACH-WRITTEN: clrs Net @6d5728b9 (same modules, same names, same arithmetic) with per-step access to its latents.

TNet differs from clrs._src.nets.Net in four ways, none of which changes the numbers when unused:
  on_step(i, lat)   called inside every step with node_fts, edge_fts, graph_fts, the next hidden and the triplet edge
                    latent; what it returns is stacked over steps and handed back as the third result
  force(i), inject  optional one-step state forcing (dev knob): for instances whose coin is up, this step starts from the true hints
                    and from inject(i, hidden, coin), the teacher's hidden state in the student's normalized coordinates
  hint_cut          from this step on, encoded hints are left out (dev test: is the hidden state Markov?)
  probe             detached hint decoders on the latents, used only to read hint accuracy: no gradient reaches the model
Scan steps are rematerialised because heapsort does not fit 24 GB otherwise (WSP-GD-GAP).
"""
import functools

import haiku as hk
import jax
import jax.numpy as jnp

from clrs._src import decoders, encoders, nets as N, probing, specs

_Type = specs.Type
_Stage = specs.Stage


class TNet(N.Net):
    def __init__(self, *args, on_step=None, inject=None, force=None, hint_cut=None, probe=False, **kw):
        super().__init__(*args, **kw)
        self.on_step, self.inject, self.force, self.hint_cut, self.probe = on_step, inject, force, hint_cut, probe
        self._i = 0

    def _msg_passing_step(self, mp_state, i, **kw):
        self._i = i
        new, acc = super()._msg_passing_step(mp_state, i, **kw)
        return new.replace(lstm_state=None), acc.replace(lstm_state=new.lstm_state)

    def _one_step_pred(self, inputs, hints, hidden, batch_size, nb_nodes, lstm_state, spec, encs, decs, repred):
        node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
        edge_fts = jnp.zeros((batch_size, nb_nodes, nb_nodes, self.hidden_dim))
        graph_fts = jnp.zeros((batch_size, self.hidden_dim))
        adj_mat = jnp.repeat(jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)
        coin = None
        if self.force is not None:
            coin = self.force(self._i)
            hints = [self._true_hint(dp, coin, nb_nodes) for dp in hints]
        trajectories = [inputs]
        if self.encode_hints:
            trajectories.append(hints)
        for traj_i, trajectory in enumerate(trajectories):
            keep = 1.0
            if traj_i == 1 and self.hint_cut is not None:
                keep = (self._i < self.hint_cut).astype(jnp.float32) if hasattr(self._i, "astype") else float(self._i < self.hint_cut)
            for dp in trajectory:
                dp = encoders.preprocess(dp, nb_nodes)
                assert dp.type_ != _Type.SOFT_POINTER
                adj_mat = encoders.accum_adj_mat(dp, adj_mat)
                encoder = encs[dp.name]
                e0, n0, g0 = edge_fts, node_fts, graph_fts
                edge_fts = e0 + keep * (encoders.accum_edge_fts(encoder, dp, e0) - e0)
                node_fts = n0 + keep * (encoders.accum_node_fts(encoder, dp, n0) - n0)
                graph_fts = g0 + keep * (encoders.accum_graph_fts(encoder, dp, g0) - g0)
        if self.inject is not None:
            hidden = self.inject(self._i, hidden, coin)
        nxt_hidden = hidden
        for _ in range(self.nb_msg_passing_steps):
            nxt_hidden, nxt_edge = self.processor(node_fts, edge_fts, graph_fts, adj_mat, nxt_hidden,
                                                  batch_size=batch_size, nb_nodes=nb_nodes)
        if not repred:
            nxt_hidden = hk.dropout(hk.next_rng_key(), self._dropout_prob, nxt_hidden)
        h_t = jnp.concatenate([node_fts, hidden, nxt_hidden], axis=-1)
        e_t = jnp.concatenate([edge_fts, nxt_edge], axis=-1) if nxt_edge is not None else edge_fts
        hint_preds, output_preds = decoders.decode_fts(
            decoders=decs, spec=spec, h_t=h_t, adj_mat=adj_mat, edge_fts=e_t, graph_fts=graph_fts,
            inf_bias=self.processor.inf_bias, inf_bias_edge=self.processor.inf_bias_edge, repred=repred)
        extra = {}
        if self.on_step is not None:
            extra["step"] = self.on_step(self._i, dict(node=node_fts, edge=edge_fts, graph=graph_fts, hidden=nxt_hidden,
                                                       tri=nxt_edge))
        if self.probe:
            sg = jax.lax.stop_gradient
            pspec = {k: v for k, v in spec.items() if k in self._probe_decs}
            probe_preds, _ = decoders.decode_fts(
                decoders=self._probe_decs, spec=pspec, h_t=sg(h_t), adj_mat=sg(adj_mat), edge_fts=sg(e_t),
                graph_fts=sg(graph_fts), inf_bias=self.processor.inf_bias, inf_bias_edge=self.processor.inf_bias_edge,
                repred=repred)
            extra["probe"] = probe_preds
        return nxt_hidden, output_preds, hint_preds, extra

    def _true_hint(self, dp, coin, nb_nodes):
        true = jnp.asarray(self._hints[dp.name])[self._i]
        if dp.type_ == _Type.SOFT_POINTER:
            true = hk.one_hot(true, nb_nodes)
        return probing.DataPoint(name=dp.name, location=dp.location, type_=dp.type_,
                                 data=jnp.where(N._expand_to(coin, true), true, dp.data))

    def __call__(self, features_list, repred, algorithm_index, return_hints, return_all_outputs):
        """clrs Net.__call__, one algorithm; returns (output_preds, hint_preds, extras stacked over steps)."""
        features = features_list[0]
        self.encoders, self.decoders = self._construct_encoders_decoders()
        self.processor = self.processor_factory(self.hidden_dim)
        self.lstm = None
        spec = self.spec[0]
        if self.probe:
            self._probe_decs = {name: decoders.construct_decoders(loc, t, hidden_dim=self.hidden_dim,
                                                                  nb_dims=self.nb_dims[0][name], name=f"probe_{name}")
                                for name, (stage, loc, t) in spec.items() if stage == _Stage.HINT}
        inputs, hints, lengths = features.inputs, features.hints, features.lengths
        self._hints = {h.name: h.data for h in hints}
        batch_size, nb_nodes = N._data_dimensions(features)
        nb_mp_steps = max(1, hints[0].data.shape[0] - 1)
        hiddens = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
        mp_state = N._MessagePassingScanState(hint_preds=None, output_preds=None, hiddens=hiddens, lstm_state=None)
        common = dict(hints=hints, repred=repred, inputs=inputs, batch_size=batch_size, nb_nodes=nb_nodes,
                      lengths=lengths, spec=spec, encs=self.encoders[0], decs=self.decoders[0],
                      return_hints=return_hints, return_all_outputs=return_all_outputs)
        mp_state, lean = self._msg_passing_step(mp_state, i=0, first_step=True, **common)
        scan_fn = functools.partial(self._msg_passing_step, first_step=False, **common)
        output_mp_state, accum = hk.scan(hk.remat(scan_fn), mp_state, jnp.arange(nb_mp_steps - 1) + 1,
                                         length=nb_mp_steps - 1)
        accum = jax.tree_util.tree_map(lambda init, tail: jnp.concatenate([init[None], tail], axis=0), lean, accum)
        hint_preds = None
        if accum.hint_preds:
            d = accum.hint_preds
            hint_preds = [dict(zip(d, x)) for x in zip(*d.values())]
        return output_mp_state.output_preds, hint_preds, accum.lstm_state
