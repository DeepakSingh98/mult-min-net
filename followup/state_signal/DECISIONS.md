# Lane decisions log (for the morning)

Times are UTC on 2026-11-07. The user is asleep, so these decisions are mine, made within the protocol and the ceiling.

- 06:40: lane registered at a $25 ceiling. Protocol committed before any measurement (988e21fec8).
- 07:30: coordinator additions taken as an amendment (denominators, output-gain control, ALTA), before any Experiment 2 result.
- 08:40: Experiment 2 grid cut for cost (500 steps, batch 32, 5 lr, 2 tau, 1 dev draw). Only the rate had been read. 91 dev runs at the old settings were discarded unread.
- 09:00: on `two` dev, the positive control fails in every frozen arm, because Adam's equal step breaks the other tight weights. The frozen arms still run. A post hoc oracle arm (`dmg`, only damaged entries move) was added and labelled.
- 09:10: Experiment 3 resumed after two Spot preemptions and a GPU out-of-memory on matched arms with 4 processes. Rerun with 2 processes.
- 09:15: `ntt` n = 32 register cell cut. It would have run past the job limit, and it isn't needed for any read; registers at n = 2 and 8 are complete.
- 09:20: Experiment 4 added at the user's request, protocol frozen before running.
- 09:25: ceiling raised to $100 by the user ("make sure budget doesnt stop anything, its ok if we go to $100 on any lane"). PENDING and `cloud.py` updated. Plan for the room: after the cut Experiment 2 runs finish, rerun the scored `anylen` arms at the protocol's original 1,500 steps and batch 64, with the lr and tau already chosen, as a labelled sensitivity check on the cut. No new scope.
- 09:40: the 1,500-step sensitivity check applies only to the `dmg` arms on `anylen`, 192 runs. The frozen arms are dead by step 100 at every lr on dev, so more steps cannot change them.
- 12:00: post hoc lr sweep on anylen z1 oracle arm (std vs sur at every lr, same 12 draws), to check that the one surrogate difference is not an lr-choice artefact.
