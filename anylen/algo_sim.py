"""Integer-level simulation of the systolic multiplication schedule (no network): validates timing before weights."""
import random, sys

def run(a, b, extra=0):
    A = [int(c) for c in str(a)[::-1]]; B = [int(c) for c in str(b)[::-1]]
    n, m = len(A), len(B)
    toks = ['BOS'] + A + ['*'] + B + ['EOS']
    C = [int(c) for c in str(a * b).zfill(n + m)[::-1]]
    toks += C
    T = len(toks)
    d = [t if isinstance(t, int) else 0 for t in toks]
    S = d[:]; Tl = [1 if t == 'BOS' else 0 for t in toks]
    Lsep = [1 if t == '*' else 0 for t in toks]; G = [1 if t == 'EOS' else 0 for t in toks]
    isEOS = G[:]; isStar = Lsep[:]
    Zu = [0]*T; Zc = [0]*T; ZH = [0]*T; Yu = [0]*T; Yc = [0]*T; YH = [0]*T
    Fz = [0]*T; Ans = [None]*T
    pred = lambda R: [R[0]] + R[:-1]
    K = 2 * T + extra
    for t in range(1, K + 1):
        S, Tl = pred(S), pred(Tl)
        Lsep = [max(x, y) for x, y in zip(Lsep, pred(Lsep))]
        G = [max(x, y) for x, y in zip(G, pred(G))]
        Fb = pred(isStar)
        nZu, nZc, nZH = pred(Yu), pred(Yc), pred(YH)
        Yu, Yc, YH = Zu, Zc, ZH
        Zu, Zc, ZH = nZu, nZc, nZH
        # MLP1
        P = [S[p] * d[p] * (Lsep[p] and not G[p]) for p in range(T)]
        YH = [max(YH[p], Fb[p] and Tl[p]) for p in range(T)]
        seed = [ZH[p] and isEOS[p] for p in range(T)]
        Fin = pred(Fz)
        # MLP2
        for p in range(T):
            V = Yu[p] + Zc[p] + P[p]
            assert V <= 99
            Yu[p], Yc[p] = V % 10, V // 10
            Zc[p] = 0
            edge = max(seed[p], Fin[p]) - Fz[p]
            assert edge in (0, 1)
            if edge:
                assert Ans[p] is None
                Ans[p] = V % 10
                Fz[p] = 1
    Q0 = n + m + 2
    got = [Ans[Q0 + k] for k in range(n + m)]
    return got == C, got, C

random.seed(1)
bad = 0
cases = [(a, b) for a in range(0, 1000) for b in range(0, 1000) if random.random() < 0.002]
cases += [(10**k - 1, 10**j - 1) for k in range(1, 9) for j in range(1, 9)]
cases += [(random.randrange(10**(k-1), 10**k), random.randrange(10**(j-1), 10**j)) for k in range(1, 25) for j in (1, 3, k, 20)]
for a, b in cases:
    ok, got, C = run(a, b)
    if not ok:
        bad += 1
        if bad < 5: print('BAD', a, b, got, C)
print('cases', len(cases), 'bad', bad)
