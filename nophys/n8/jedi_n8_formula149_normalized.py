"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.
  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1
             avg_k = the term's average size on the entire training set (term / avg is 1 on average), so share_k is the
             fraction of the neuron's average input that comes from that if-statement (sign: pushes it up / down).
  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1
             h_j = neuron j (after max(0, .) and the network's rounding), avg_j = its average on the training jets.

How much each neuron matters (share of all class scores, averaged over the training jets):
  neuron 13:  13.2%   (on for 90% of jets)
  neuron  9:  11.9%   (on for 71% of jets)
  neuron  7:  10.3%   (on for 61% of jets)
  neuron  5:   8.9%   (on for 79% of jets)
  neuron  6:   8.1%   (on for 35% of jets)
  neuron  2:   8.0%   (on for 96% of jets)
  neuron 10:   7.7%   (on for 71% of jets)
  neuron  3:   7.5%   (on for 24% of jets)
  neuron 11:   6.1%   (on for 80% of jets)
  neuron 14:   4.4%   (on for 27% of jets)
  neuron  0:   4.0%   (on for 43% of jets)
  neuron  4:   3.3%   (on for 51% of jets)
  neuron  1:   3.2%   (on for 61% of jets)
  neuron 15:   2.0%   (on for 28% of jets)
  neuron  8:   1.2%   (on for 34% of jets)
  neuron 12:   0.3%   (on for 5% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.4% (the network: 65.8%); same class as the network for 86.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.centroid_offset        distance of the pT centroid from the jet axis
"""
import math
from types import SimpleNamespace

CLASSES = ['g', 'q', 'W', 'Z', 't']
W = [[-0.15625, 0.0, 0.34375, 0.0, 0.015625], [0.390625, 0.0, -0.03125, 0.125, 0.0], [0.4296875, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.5, -0.5625, 0.0625], [-0.03125, -0.09375, 0.0, 0.078125, 0.125], [-0.1875, 0.046875, 0.0, 0.015625, -0.25], [0.109375, 0.125, -0.3125, -0.375, 0.0], [0.0, 0.0, 0.21875, 0.46875, 0.0], [0.0, 0.0625, -0.25, 0.0, 0.1875], [0.171875, 0.25390625, -0.03125, -0.03125, 0.0], [0.0, -0.125, 0.0, 0.0, 0.375], [0.0, 0.0, 0.375, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -0.5], [0.0, 0.0, 0.0703125, 0.0546875, -0.40625], [0.0, 0.0, -0.75, 0.375, 0.0], [0.0, 0.0625, -0.6875, -0.15625, 0.0]]
B = [-0.4375, 0.03125, -0.125, -0.09375, 1.34375]
INT_BITS = [3, 5, 4, 4, 5, 5, 4, 4, 3, 4, 5, 4, 5, 4, 3, 3]
FRAC_BITS = [3, 3, 4, 3, 2, 3, 3, 3, 3, 2, 4, 4, 3, 3, 4, 3]


def quantities(pt, eta, phi):
    pt, eta, phi = [float(x) for x in pt], [float(x) for x in eta], [float(x) for x in phi]
    n = len(pt)
    P = range(n)
    real = [i for i in P if pt[i] > 0]
    tot = sum(pt)
    z = [x / tot for x in pt]
    zs = sorted(z, reverse=True)
    dr = [math.hypot(eta[i], phi[i]) for i in P]

    def dist2(i, j):
        return (eta[i] - eta[j]) ** 2 + (phi[i] - phi[j]) ** 2

    def mass_of(k):
        E = sum(pt[i] * math.cosh(eta[i]) for i in range(k))
        px = sum(pt[i] * math.cos(phi[i]) for i in range(k))
        py = sum(pt[i] * math.sin(phi[i]) for i in range(k))
        pz = sum(pt[i] * math.sinh(eta[i]) for i in range(k))
        return math.sqrt(max(E * E - px * px - py * py - pz * pz, 0.0))

    def pair_mass(i, j):
        return math.sqrt(max(2 * pt[i] * pt[j] * (math.cosh(eta[i] - eta[j]) - math.cos(phi[i] - phi[j])), 0.0))

    def tau(k):
        axes = [(eta[max(P, key=lambda i: pt[i])], phi[max(P, key=lambda i: pt[i])])]
        for _ in range(1, k):
            far = max(P, key=lambda i: pt[i] * min(math.hypot(eta[i] - a, phi[i] - b) for a, b in axes))
            axes.append((eta[far], phi[far]))
        for _ in range(6):
            nearest = [min(range(k), key=lambda j: math.hypot(eta[i] - axes[j][0], phi[i] - axes[j][1])) for i in P]
            for j in range(k):
                w = sum(pt[i] for i in P if nearest[i] == j)
                if w > 0:
                    axes[j] = (sum(pt[i] * eta[i] for i in P if nearest[i] == j) / w, sum(pt[i] * phi[i] for i in P if nearest[i] == j) / w)
        return sum(z[i] * min(math.hypot(eta[i] - a, phi[i] - b) for a, b in axes) for i in P) / 0.8

    ta = sum(z[i] * eta[i] ** 2 for i in P)
    tb = sum(z[i] * eta[i] * phi[i] for i in P)
    tc = sum(z[i] * phi[i] ** 2 for i in P)
    disc = math.sqrt(max((ta - tc) ** 2 / 4 + tb ** 2, 0.0))
    lam1, lam2 = (ta + tc) / 2 + disc, max((ta + tc) / 2 - disc, 0.0)

    hard = sorted(P, key=lambda i: -pt[i])[:24]
    R = {(i, j): math.sqrt(dist2(i, j)) for i in hard for j in hard}
    e2 = sum(z[i] * z[j] * R[i, j] for i in hard for j in hard if i < j)
    e3 = sum(z[i] * z[j] * z[k] * R[i, j] * R[i, k] * R[j, k] for i in hard for j in hard for k in hard if i < j < k)

    _memo = {}

    def m4(v):
        return math.sqrt(max(v[3] ** 2 - v[0] ** 2 - v[1] ** 2 - v[2] ** 2, 0.0))

    def ecf(name):
        if 'ecf' not in _memo:
            h3, h4 = min(n, 24), min(n, 12)
            Rm = {(i, j): math.sqrt(dist2(i, j)) for i in range(h3) for j in range(h3)}
            o = dict(e2=0.0, e2b2=0.0, e3=0.0, e3b2=0.0, g31=0.0, g32=0.0, e4=0.0, g41=0.0, g42=0.0)
            for i in range(h3):
                for j in range(i + 1, h3):
                    w = z[i] * z[j]
                    o['e2'] += w * Rm[i, j]
                    o['e2b2'] += w * Rm[i, j] ** 2
                    for k in range(j + 1, h3):
                        a, b, c = Rm[i, j], Rm[i, k], Rm[j, k]
                        w3 = z[i] * z[j] * z[k]
                        s = sorted((a, b, c))
                        o['e3'] += w3 * a * b * c
                        o['e3b2'] += w3 * (a * b * c) ** 2
                        o['g31'] += w3 * s[0]
                        o['g32'] += w3 * s[0] * s[1]
            for i in range(h4):
                for j in range(i + 1, h4):
                    for k in range(j + 1, h4):
                        for l in range(k + 1, h4):
                            d = (Rm[i, j], Rm[i, k], Rm[i, l], Rm[j, k], Rm[j, l], Rm[k, l])
                            w4 = z[i] * z[j] * z[k] * z[l]
                            s = sorted(d)
                            o['e4'] += w4 * d[0] * d[1] * d[2] * d[3] * d[4] * d[5]
                            o['g41'] += w4 * s[0]
                            o['g42'] += w4 * s[0] * s[1]
            _memo['ecf'] = o
        return _memo['ecf'][name]

    def kaxes(k):
        if ('ax', k) not in _memo:
            hard = max(P, key=lambda i: pt[i])
            axes = [(eta[hard], phi[hard])]
            for _ in range(1, k):
                far = max(P, key=lambda i: min(math.hypot(eta[i] - a, phi[i] - b) for a, b in axes) * pt[i])
                axes.append((eta[far], phi[far]))
            for _ in range(6):
                near = [min(range(k), key=lambda j: math.hypot(eta[i] - axes[j][0], phi[i] - axes[j][1])) for i in P]
                for j in range(k):
                    w = sum(pt[i] for i in P if near[i] == j)
                    if w > 0:
                        axes[j] = (sum(pt[i] * eta[i] for i in P if near[i] == j) / w, sum(pt[i] * phi[i] for i in P if near[i] == j) / w)
            dist = [[math.hypot(eta[i] - a, phi[i] - b) for a, b in axes] for i in P]
            _memo['ax', k] = (axes, dist)
        return _memo['ax', k]

    def tau_n(k, beta=1):
        axes, dist = kaxes(k)
        return sum(z[i] * min(dist[i]) ** beta for i in P) / 0.8 ** beta

    def subjets(k):
        if ('sj', k) not in _memo:
            axes, dist = kaxes(k)
            S, V = [0.0] * k, [[0.0] * 4 for _ in range(k)]
            for i in real:
                j = min(range(k), key=lambda a: dist[i][a])
                S[j] += z[i]
                for c, x in enumerate((pt[i] * math.cos(phi[i]), pt[i] * math.sin(phi[i]), pt[i] * math.sinh(eta[i]), pt[i] * math.cosh(eta[i]))):
                    V[j][c] += x
            order = sorted(range(k), key=lambda j: -S[j])
            pairs = [(p, q) for p in range(k) for q in range(p + 1, k)]
            _memo['sj', k] = dict(z=[S[j] for j in order], mass=[m4(V[j]) for j in order],
                                  dr=[math.hypot(axes[order[p]][0] - axes[order[q]][0], axes[order[p]][1] - axes[order[q]][1]) for p, q in pairs],
                                  mpair=[m4([V[order[p]][c] + V[order[q]][c] for c in range(4)]) for p, q in pairs])
        return _memo['sj', k]

    def softdrop(what):
        if 'sd' not in _memo:
            node = {i: (pt[i] * math.cos(phi[i]), pt[i] * math.sin(phi[i]), pt[i] * math.sinh(eta[i]), pt[i] * math.cosh(eta[i])) for i in range(min(n, 20)) if pt[i] > 0}
            live, kids, new = list(node), {}, 1000

            def dR2(u, v):
                ya, pa = 0.5 * math.log(max(u[3] + u[2], 1e-300) / max(u[3] - u[2], 1e-300)), math.atan2(u[1], u[0])
                yb, pb = 0.5 * math.log(max(v[3] + v[2], 1e-300) / max(v[3] - v[2], 1e-300)), math.atan2(v[1], v[0])
                return (ya - yb) ** 2 + ((pa - pb + math.pi) % (2 * math.pi) - math.pi) ** 2
            while len(live) > 1:                                  # Cambridge/Aachen: merge the closest pair
                best = None
                for a in range(len(live)):
                    for b in range(a + 1, len(live)):
                        d = dR2(node[live[a]], node[live[b]])
                        if best is None or d < best[0]:
                            best = (d, a, b)
                _, a, b = best
                node[new] = tuple(x + y for x, y in zip(node[live[a]], node[live[b]]))
                kids[new] = (live[a], live[b])
                live[a] = new
                live.pop(b)
                new += 1
            cur, removed, res = live[0], 0, (0.0, 0.0, 0.0)
            while cur in kids:                                    # soft drop, beta = 0, z_cut = 0.1
                c1, c2 = kids[cur]
                p1, p2 = math.hypot(node[c1][0], node[c1][1]), math.hypot(node[c2][0], node[c2][1])
                zz = min(p1, p2) / max(p1 + p2, 1e-300)
                if zz > 0.1:
                    res = (m4(node[cur]), zz, math.sqrt(dR2(node[c1], node[c2])))
                    break
                cur = c1 if p1 >= p2 else c2
                removed += 1
            _memo['sd'] = dict(mass=res[0], zg=res[1], rg=res[2], removed=float(removed))
        return _memo['sd'][what]

    def softp(s, what):
        j = len(real) - s
        if j < 15:
            return 0.0
        return dict(pt=pt[j], z=z[j], abseta=abs(eta[j]), absphi=abs(phi[j]), dr=dr[j], dr0=math.sqrt(dist2(0, j)))[what]

    def ncum(f):
        c = 0.0
        for k, i in enumerate(real):
            c += z[i]
            if c >= f:
                return float(k + 1)
        return float(len(real))

    return SimpleNamespace(
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top2=mass_of(2),
        max_dr=max(dr[i] for i in real),
        pt_6=pt[6],
        pt_7=pt[7],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sj2_dr=subjets(2)["dr"][0],
        sum_pt_top5=sum(pt[:5]),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 14.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.78194 * (0.0001278587
        - 0.2489848 * max(0.0, 0.0044 - Q.width) / 0.001579604   # -24.9%  width < 0.0044
        + 0.2303089 * max(0.0, 0.0088 - Q.width) / 0.004545275   # +23.0%  width < 0.0088
        + 0.1318563 * max(0.0, 0.026 - Q.e2) / 0.00818946   # +13.2%  e2 < 0.026
        - 0.08940701 * max(0.0, 56.0 - Q.mass) / 21.14574   # -8.9%  mass < 56
        - 0.06718103 * max(0.0, 0.0017 - Q.C2_b2) / 0.0009832334   # -6.7%  C2_b2 < 0.0017
        - 0.05196502 * max(0.0, Q.sj3_dr_max - 0.23) / 0.02603877   # -5.2%  sj3_dr_max > 0.23
        + 0.04763956 * max(0.0, 23.0 - Q.mass) / 4.856586   # +4.8%  mass < 23
        + 0.04621549 * max(0.0, 0.12 - Q.planar_flow) / 0.03733085   # +4.6%  planar_flow < 0.12
        + 0.03891211 * max(0.0, 0.013 - Q.girth2) * max(0.0, 1.0 - Q.D2) / 0.0009667167   # +3.9%  girth2 < 0.013 and D2 < 1
        - 0.01761847 * max(0.0, 0.015 - Q.girth2) * max(0.0, Q.centroid_offset - 0.019) / 2.411436e-05   # -1.8%  girth2 < 0.015 and centroid_offset > 0.019
        - 0.01715388 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.88 - Q.D2) / 8.049764e-05   # -1.7%  lam1 < 0.0065 and D2 < 0.88
        - 0.0127575 * max(0.0, Q.sum_pt - 910.0) / 11.22503   # -1.3%  sum_pt > 910
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 9.948;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.948083 * (0.008845925
        - 0.3438687 * max(0.0, 0.0074 - Q.width) / 0.003455388   # -34.4%  width < 0.0074
        + 0.2504245 * max(0.0, 0.0065 - Q.mass_over_sum_pt_sq) / 0.003030711   # +25.0%  mass_over_sum_pt_sq < 0.0065
        + 0.19798 * max(0.0, Q.log_sum_pt - 6.4) / 0.1930903   # +19.8%  log_sum_pt > 6.4
        - 0.1026596 * max(0.0, 0.061 - Q.z_7) / 0.0138009   # -10.3%  z_7 < 0.061
        + 0.07010674 * max(0.0, Q.pt_7 - 34.0) / 4.744406   # +7.0%  pt_7 > 34
        - 0.03496041 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 0.017 - Q.tau2) / 0.03974732   # -3.5%  pt_7 > 34 and tau2 < 0.017
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 2.442;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.441756 * (0.3391003
        + 0.5415704 * max(0.0, 0.006 - Q.lam1) / 0.002547943   # +54.2%  lam1 < 0.006
        + 0.140572 * max(0.0, 6.5 - Q.log_sum_pt) / 0.08351396   # +14.1%  log_sum_pt < 6.5
        - 0.112254 * max(0.0, 0.033 - Q.z_7) / 0.002175372   # -11.2%  z_7 < 0.033
        + 0.1027016 * max(0.0, 780.0 - Q.sum_pt) / 106.7115   # +10.3%  sum_pt < 780
        - 0.05187734 * max(0.0, 0.0077 - Q.girth) / 0.0002341438   # -5.2%  girth < 0.0077
        + 0.05102476 * max(0.0, 0.00033 - Q.girth2) * max(0.0, 35.0 - Q.mass_top2) / 0.001427148   # +5.1%  girth2 < 0.00033 and mass_top2 < 35
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 8.649;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.649203 * (-0.6139294
        + 0.4065138 * max(0.0, Q.girth - 0.036) / 0.03005146   # +40.7%  girth > 0.036
        - 0.1993476 * max(0.0, Q.girth2 - 0.0093) / 0.002095016   # -19.9%  girth2 > 0.0093
        + 0.09344297 * max(0.0, Q.sj2_dr - 0.2) / 0.02322434   # +9.3%  sj2_dr > 0.2
        + 0.07914285 * max(0.0, Q.girth - 0.033) * max(0.0, Q.log_sum_pt - 6.2) / 0.007522226   # +7.9%  girth > 0.033 and log_sum_pt > 6.2
        + 0.07900616 * max(0.0, Q.tau1 - 0.11) / 0.00760958   # +7.9%  tau1 > 0.11
        - 0.06637246 * max(0.0, Q.mass_over_sum_pt - 0.062) * max(0.0, 0.22 - Q.sj2_dr) / 0.0002028512   # -6.6%  mass_over_sum_pt > 0.062 and sj2_dr < 0.22
        + 0.04726441 * max(0.0, Q.centroid_offset - 0.011) / 0.008791386   # +4.7%  centroid_offset > 0.011
        - 0.02890975 * max(0.0, Q.mass - 66.0) / 3.008981   # -2.9%  mass > 66
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 12.85;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.85292 * (-0.1128148
        + 0.4512359 * max(0.0, 0.009 - Q.C2_b2) / 0.007258699   # +45.1%  C2_b2 < 0.009
        - 0.3511822 * max(0.0, 0.00066 - Q.lam2) / 0.0004965587   # -35.1%  lam2 < 0.00066
        + 0.1249975 * max(0.0, 0.21 - Q.N2) / 0.0446273   # +12.5%  N2 < 0.21
        - 0.0454612 * max(0.0, Q.mass_over_sum_pt - 0.091) / 0.008183604   # -4.5%  mass_over_sum_pt > 0.091
        - 0.02712327 * max(0.0, 68.0 - Q.mass) * max(0.0, 0.96 - Q.D2) / 1.734395   # -2.7%  mass < 68 and D2 < 0.96
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.13;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.13231 * (0.1964013
        + 0.1669234 * max(0.0, Q.log_sum_pt - 6.2) / 0.3575729   # +16.7%  log_sum_pt > 6.2
        - 0.1293751 * max(0.0, Q.pt_7 - 22.0) / 13.40357   # -12.9%  pt_7 > 22
        + 0.1212644 * max(0.0, 0.0016 - Q.girth2) * max(0.0, 0.023 - Q.centroid_offset) / 6.53558e-06   # +12.1%  girth2 < 0.0016 and centroid_offset < 0.023
        + 0.1161564 * max(0.0, 0.0055 - Q.e2_sq) / 0.002368074   # +11.6%  e2_sq < 0.0055
        - 0.1157292 * max(0.0, 0.083 - Q.z_7) * max(0.0, 0.034 - Q.centroid_offset) / 0.0006662526   # -11.6%  z_7 < 0.083 and centroid_offset < 0.034
        - 0.1143975 * max(0.0, 0.022 - Q.zdr_0) / 0.009659263   # -11.4%  zdr_0 < 0.022
        - 0.07857646 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt) / 0.004257547   # -7.9%  LHA < 0.22 and log_sum_pt < 6.8
        + 0.07003057 * max(0.0, 0.003 - Q.girth2_top3) * max(0.0, 0.0066 - Q.tau3) / 3.461325e-06   # +7.0%  girth2_top3 < 0.003 and tau3 < 0.0066
        - 0.04252541 * max(0.0, Q.log_sum_pt - 6.7) / 0.03560998   # -4.3%  log_sum_pt > 6.7
        + 0.02868662 * max(0.0, 0.03 - Q.z_7) / 0.001614788   # +2.9%  z_7 < 0.03
        - 0.01633497 * max(0.0, Q.log_sum_pt - 6.9) / 0.003849095   # -1.6%  log_sum_pt > 6.9
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 24.82;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.82495 * (0.1538774
        - 0.3291391 * max(0.0, 0.0087 - Q.girth2) / 0.004464952   # -32.9%  girth2 < 0.0087
        + 0.1465847 * max(0.0, 0.11 - Q.tau1) / 0.05423188   # +14.7%  tau1 < 0.11
        + 0.09355697 * max(0.0, 0.18 - Q.sj3_dr_max) / 0.05451989   # +9.4%  sj3_dr_max < 0.18
        - 0.0877015 * max(0.0, 0.0012 - Q.lam2) / 0.0009763163   # -8.8%  lam2 < 0.0012
        - 0.07629079 * max(0.0, 0.14 - Q.max_dr) / 0.0444581   # -7.6%  max_dr < 0.14
        + 0.06488473 * max(0.0, Q.centroid_offset - 0.0082) / 0.01045948   # +6.5%  centroid_offset > 0.0082
        + 0.0485076 * max(0.0, 0.0031 - Q.e2_sq) / 0.00110477   # +4.9%  e2_sq < 0.0031
        + 0.03961793 * max(0.0, 40.0 - Q.pt_6) * max(0.0, 6.8 - Q.log_sum_pt) / 0.9548671   # +4.0%  pt_6 < 40 and log_sum_pt < 6.8
        - 0.0310204 * max(0.0, Q.centroid_offset - 0.019) / 0.005310896   # -3.1%  centroid_offset > 0.019
        - 0.03023828 * max(0.0, 41.0 - Q.pt_6) * max(0.0, Q.z_7 - 0.024) / 0.06584769   # -3.0%  pt_6 < 41 and z_7 > 0.024
        + 0.02083021 * max(0.0, Q.sj3_dr_max - 0.18) / 0.04273628   # +2.1%  sj3_dr_max > 0.18
        - 0.02051834 * max(0.0, Q.sj3_pair_mass_min - 3.2) / 4.107796   # -2.1%  sj3_pair_mass_min > 3.2
        - 0.01110946 * max(0.0, Q.sj3_dr_min - 0.14) / 0.007053496   # -1.1%  sj3_dr_min > 0.14
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 22.15;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.15405 * (0.3480177
        - 0.2155154 * max(0.0, Q.width - 0.0056) / 0.003080348   # -21.6%  width > 0.0056
        - 0.1427495 * max(0.0, 0.0081 - Q.lam1) / 0.00408062   # -14.3%  lam1 < 0.0081
        + 0.1234179 * max(0.0, Q.girth2 - 0.013) / 0.001477949   # +12.3%  girth2 > 0.013
        + 0.1042069 * max(0.0, Q.mass_over_sum_pt - 0.075) / 0.01254677   # +10.4%  mass_over_sum_pt > 0.075
        - 0.07978941 * max(0.0, Q.mass_over_sum_pt - 0.091) / 0.008183604   # -8.0%  mass_over_sum_pt > 0.091
        - 0.07928187 * max(0.0, 8.1e-05 - Q.e3) / 5.593676e-05   # -7.9%  e3 < 8.1e-05
        + 0.0520047 * max(0.0, Q.mass - 37.0) / 13.78128   # +5.2%  mass > 37
        - 0.04624889 * max(0.0, 0.023 - Q.centroid_offset) / 0.009947573   # -4.6%  centroid_offset < 0.023
        - 0.0403437 * max(0.0, Q.e2 - 0.05) / 0.00341136   # -4.0%  e2 > 0.05
        + 0.03481167 * max(0.0, 0.15 - Q.sj2_dr) / 0.04406969   # +3.5%  sj2_dr < 0.15
        - 0.03223131 * max(0.0, 0.00097 - Q.girth2) / 0.0002069722   # -3.2%  girth2 < 0.00097
        + 0.03096476 * max(0.0, 0.051 - Q.tau1) / 0.01569782   # +3.1%  tau1 < 0.051
        - 0.01843393 * max(0.0, Q.mass - 77.0) / 1.512542   # -1.8%  mass > 77
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 2.482;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.482175 * (-0.1438255
        + 0.5914871 * max(0.0, 0.0057 - Q.girth2) * max(0.0, 0.027 - Q.centroid_offset) / 3.823372e-05   # +59.1%  girth2 < 0.0057 and centroid_offset < 0.027
        - 0.4085129 * max(0.0, 0.19 - Q.LHA) / 0.02278653   # -40.9%  LHA < 0.19
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 17.49;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.49073 * (-0.04322289
        + 0.2105029 * max(0.0, 0.0038 - Q.girth2) / 0.001291877   # +21.1%  girth2 < 0.0038
        - 0.1940961 * max(0.0, 0.062 - Q.mass_over_sum_pt) / 0.01907238   # -19.4%  mass_over_sum_pt < 0.062
        - 0.1587525 * max(0.0, 0.14 - Q.sj3_dr_max) / 0.03615491   # -15.9%  sj3_dr_max < 0.14
        + 0.1273346 * max(0.0, 0.2 - Q.sj3_dr_max) / 0.06668192   # +12.7%  sj3_dr_max < 0.2
        + 0.09737116 * max(0.0, 58.0 - Q.mass) * max(0.0, 0.022 - Q.centroid_offset) / 0.2504549   # +9.7%  mass < 58 and centroid_offset < 0.022
        + 0.06986601 * max(0.0, 0.11 - Q.max_dr) / 0.0275227   # +7.0%  max_dr < 0.11
        - 0.04036777 * max(0.0, Q.girth - 0.061) / 0.01634403   # -4.0%  girth > 0.061
        + 0.03956947 * max(0.0, 52.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt) / 4.272216   # +4.0%  mass < 52 and log_sum_pt < 6.8
        + 0.03382869 * max(0.0, Q.e2 - 0.042) / 0.004889989   # +3.4%  e2 > 0.042
        + 0.01498333 * max(0.0, 0.05 - Q.z_7) / 0.007730664   # +1.5%  z_7 < 0.05
        + 0.01332743 * max(0.0, Q.lam2 - 0.0013) / 0.0002958205   # +1.3%  lam2 > 0.0013
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 6.071;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.071366 * (0.1540016
        + 0.1968919 * max(0.0, 77.0 - Q.mass) / 38.19178   # +19.7%  mass < 77
        + 0.1397375 * max(0.0, Q.tau1 - 0.059) / 0.0241022   # +14.0%  tau1 > 0.059
        - 0.1355437 * max(0.0, 0.0042 - Q.lam1) / 0.001521137   # -13.6%  lam1 < 0.0042
        + 0.1040398 * max(0.0, Q.girth2 - 0.0073) / 0.002526654   # +10.4%  girth2 > 0.0073
        - 0.09992052 * max(0.0, 0.0015 - Q.lam1) / 0.0003913897   # -10.0%  lam1 < 0.0015
        - 0.08873845 * max(0.0, Q.LHA - 0.3) / 0.01897055   # -8.9%  LHA > 0.3
        - 0.08441792 * max(0.0, Q.e3 - 7.5e-08) / 7.504129e-05   # -8.4%  e3 > 7.5e-08
        + 0.05622745 * max(0.0, Q.sj3_pair_mass_min - 11.0) / 1.950728   # +5.6%  sj3_pair_mass_min > 11
        + 0.03595689 * max(0.0, Q.lam2 - 0.00019) / 0.0004473513   # +3.6%  lam2 > 0.00019
        + 0.03402945 * max(0.0, Q.C2_b2 - 0.0075) / 0.001861309   # +3.4%  C2_b2 > 0.0075
        - 0.02449651 * max(0.0, Q.lam1 - 0.0071) * max(0.0, 0.38 - Q.D2_b2) / 0.0003029069   # -2.4%  lam1 > 0.0071 and D2_b2 < 0.38
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 16.98;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.9771 * (0.01278192
        + 0.2317462 * max(0.0, 0.0094 - Q.width) / 0.005031174   # +23.2%  width < 0.0094
        - 0.1731811 * max(0.0, 0.17 - Q.sj3_dr_max) / 0.04924812   # -17.3%  sj3_dr_max < 0.17
        + 0.1471681 * max(0.0, 0.26 - Q.sj3_dr_max) / 0.1110439   # +14.7%  sj3_dr_max < 0.26
        - 0.1179124 * max(0.0, 0.072 - Q.girth) / 0.02521171   # -11.8%  girth < 0.072
        - 0.06921558 * max(0.0, Q.centroid_offset - 0.0094) / 0.009711403   # -6.9%  centroid_offset > 0.0094
        + 0.05355439 * max(0.0, Q.z_7 - 0.019) / 0.03354975   # +5.4%  z_7 > 0.019
        + 0.05343417 * max(0.0, 0.15 - Q.max_dr) / 0.05096389   # +5.3%  max_dr < 0.15
        - 0.04422606 * max(0.0, 0.04 - Q.centroid_offset) * max(0.0, 900.0 - Q.sum_pt) / 3.850411   # -4.4%  centroid_offset < 0.04 and sum_pt < 900
        - 0.03459423 * max(0.0, 0.021 - Q.girth) / 0.002706496   # -3.5%  girth < 0.021
        + 0.02095581 * max(0.0, 15.0 - Q.mass) / 2.251702   # +2.1%  mass < 15
        - 0.0160648 * max(0.0, Q.centroid_offset - 0.05) * max(0.0, 2.9 - Q.D2) / 0.001398635   # -1.6%  centroid_offset > 0.05 and D2 < 2.9
        + 0.01576494 * max(0.0, Q.sj2_dr - 0.27) / 0.008286158   # +1.6%  sj2_dr > 0.27
        + 0.01131189 * max(0.0, Q.centroid_offset - 0.047) * max(0.0, 0.7 - Q.tau32) / 0.0002319361   # +1.1%  centroid_offset > 0.047 and tau32 < 0.7
        + 0.01087038 * max(0.0, 0.0091 - Q.centroid_offset) / 0.001809289   # +1.1%  centroid_offset < 0.0091
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.1566;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.1565548 * (-7.537299
        + 0.6030123 * max(0.0, Q.girth2 - 0.02) / 0.0006510652   # +60.3%  girth2 > 0.02
        + 0.3969877 * max(0.0, Q.mass - 91.0) / 0.5919078   # +39.7%  mass > 91
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 8.479;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.479059 * (-0.06852176
        + 0.401962 * max(0.0, 0.15 - Q.girth) / 0.09211511   # +40.2%  girth < 0.15
        + 0.1810581 * max(0.0, 0.016 - Q.lam1) / 0.01081128   # +18.1%  lam1 < 0.016
        - 0.1196511 * max(0.0, 0.15 - Q.girth) * max(0.0, 39.0 - Q.pt_7) / 0.7094604   # -12.0%  girth < 0.15 and pt_7 < 39
        + 0.1091631 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 40.0 - Q.pt_7) / 771.3339   # +10.9%  sum_pt_top5 > 660 and pt_7 < 40
        - 0.1065095 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt) / 0.0200689   # -10.7%  girth < 0.15 and log_sum_pt < 6.8
        - 0.0401511 * max(0.0, 0.028 - Q.z_7) / 0.001299403   # -4.0%  z_7 < 0.028
        + 0.03397197 * max(0.0, 0.15 - Q.girth) * max(0.0, Q.tau2 - 0.0083) / 0.0002924369   # +3.4%  girth < 0.15 and tau2 > 0.0083
        - 0.007533166 * max(0.0, Q.sum_pt - 990.0) * max(0.0, 62.0 - Q.pt_6) / 101.3876   # -0.8%  sum_pt > 990 and pt_6 < 62
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 35.7;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.70251 * (-0.09243047
        - 0.1724928 * max(0.0, Q.width - 0.0068) / 0.002665986   # -17.2%  width > 0.0068
        + 0.139403 * max(0.0, Q.girth - 0.027) / 0.03606549   # +13.9%  girth > 0.027
        - 0.09737115 * max(0.0, Q.girth - 0.087) / 0.007900897   # -9.7%  girth > 0.087
        + 0.08946641 * max(0.0, Q.mass_over_sum_pt - 0.08) / 0.01086454   # +8.9%  mass_over_sum_pt > 0.08
        + 0.08660846 * max(0.0, Q.sj2_dr - 0.16) / 0.03884597   # +8.7%  sj2_dr > 0.16
        - 0.08133938 * max(0.0, Q.e2 - 0.017) / 0.01569741   # -8.1%  e2 > 0.017
        + 0.06605457 * max(0.0, Q.girth - 0.041) / 0.02695216   # +6.6%  girth > 0.041
        - 0.04563059 * max(0.0, Q.sj2_dr - 0.062) / 0.09994642   # -4.6%  sj2_dr > 0.062
        + 0.04557671 * max(0.0, Q.girth - 0.081) / 0.009245472   # +4.6%  girth > 0.081
        + 0.03124471 * max(0.0, Q.e2 - 0.05) / 0.00341136   # +3.1%  e2 > 0.05
        - 0.02888346 * max(0.0, 76.0 - Q.sd_mass) / 43.32825   # -2.9%  sd_mass < 76
        - 0.02335281 * max(0.0, Q.sj2_dr - 0.2) / 0.02322434   # -2.3%  sj2_dr > 0.2
        - 0.02206889 * max(0.0, Q.centroid_offset - 0.05) / 0.0008015409   # -2.2%  centroid_offset > 0.05
        - 0.01632981 * max(0.0, Q.z_dr_0p05_0p1 - 0.75) / 0.02295335   # -1.6%  z_dr_0p05_0p1 > 0.75
        + 0.0162506 * max(0.0, 50.0 - Q.sd_mass) / 22.66356   # +1.6%  sd_mass < 50
        + 0.01438483 * max(0.0, Q.z_dr_0p05_0p1 - 0.75) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02079249   # +1.4%  z_dr_0p05_0p1 > 0.75 and n_dr_0p2_0p4 < 1
        - 0.008819888 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.0075 - Q.lam1) / 6.162273e-05   # -0.9%  planar_flow < 0.11 and lam1 < 0.0075
        - 0.008250444 * max(0.0, Q.centroid_offset - 0.027) / 0.003226304   # -0.8%  centroid_offset > 0.027
        + 0.006471457 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 0.006 - Q.width) / 2.593123e-05   # +0.6%  planar_flow < 0.1 and width < 0.006
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 14.11;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.11027 * (0.003415953
        - 0.3599805 * max(0.0, 0.0074 - Q.girth2) / 0.003455388   # -36.0%  girth2 < 0.0074
        + 0.1942034 * max(0.0, 0.043 - Q.e2) / 0.01902959   # +19.4%  e2 < 0.043
        + 0.1599681 * max(0.0, 0.013 - Q.width) / 0.008032715   # +16.0%  width < 0.013
        - 0.1082601 * max(0.0, 0.018 - Q.centroid_offset) / 0.006472794   # -10.8%  centroid_offset < 0.018
        - 0.07081784 * max(0.0, 0.23 - Q.N2) / 0.05460429   # -7.1%  N2 < 0.23
        + 0.04500084 * max(0.0, 0.23 - Q.N2) * max(0.0, Q.sum_pt_top5 - 410.0) / 10.32478   # +4.5%  N2 < 0.23 and sum_pt_top5 > 410
        + 0.0411407 * max(0.0, 0.15 - Q.mass_over_sum_pt) * max(0.0, 0.71 - Q.D2) / 0.005004363   # +4.1%  mass_over_sum_pt < 0.15 and D2 < 0.71
        - 0.0206285 * max(0.0, 0.0083 - Q.girth2) * max(0.0, 0.72 - Q.D2) / 0.0001243905   # -2.1%  girth2 < 0.0083 and D2 < 0.72
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.033583193277311, 0.7834785714285715, 2.491916386554622, 0.8832241596638656, 1.3214159663865546, 2.3665857142857143, 1.1787644957983194, 2.0036764705882355, 0.31435903361344536, 3.2615302521008402, 2.0530627100840335, 2.162036869747899, 0.07157331932773109, 3.30866281512605, 0.5227341386554621, 0.28887920168067227]
T = [2.7128209624474793, 1.5046201582195378, 3.442629955028887, 2.6402499622505253, 3.0369464351365547]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +39%, n9 +21%, n5 -16%, n1 +11%, n0 -6%, n6 +5% ...
            + 0.3946981 * h[2] / H_AVG[2]
            + 0.2066393 * h[9] / H_AVG[9]
            - 0.1635695 * h[5] / H_AVG[5]
            + 0.1128148 * h[1] / H_AVG[1]
            - 0.05953116 * h[0] / H_AVG[0]
            + 0.0475252 * h[6] / H_AVG[6]
            - 0.01522189 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +55%, n10 -17%, n6 +10%, n4 -8%, n5 +7%, n8 +1% ...
            + 0.5503867 * h[9] / H_AVG[9]
            - 0.1705632 * h[10] / H_AVG[10]
            + 0.09792874 * h[6] / H_AVG[6]
            - 0.0823349 * h[4] / H_AVG[4]
            + 0.07372871 * h[5] / H_AVG[5]
            + 0.01305807 * h[8] / H_AVG[8]
            + 0.01199967 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -13%, n7 +13%, n14 -11%, n6 -11%, n0 +10% ...
            + 0.2355071 * h[11] / H_AVG[11]
            - 0.1282775 * h[3] / H_AVG[3]
            + 0.1273167 * h[7] / H_AVG[7]
            - 0.1138811 * h[14] / H_AVG[14]
            - 0.1070007 * h[6] / H_AVG[6]
            + 0.1032043 * h[0] / H_AVG[0]
            + 0.06757635 * h[13] / H_AVG[13]
            - 0.05768975 * h[15] / H_AVG[15]
            - 0.02960609 * h[9] / H_AVG[9]
            - 0.02282841 * h[8] / H_AVG[8]
            - 0.007111919 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +36%, n3 -19%, n6 -17%, n14 +7%, n13 +7%, n4 +4% ...
            + 0.3557327 * h[7] / H_AVG[7]
            - 0.1881691 * h[3] / H_AVG[3]
            - 0.1674223 * h[6] / H_AVG[6]
            + 0.07424498 * h[14] / H_AVG[14]
            + 0.06853234 * h[13] / H_AVG[13]
            + 0.0391007 * h[4] / H_AVG[4]
            - 0.03860347 * h[9] / H_AVG[9]
            + 0.03709301 * h[1] / H_AVG[1]
            - 0.01709587 * h[15] / H_AVG[15]
            + 0.01400545 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -44%, n10 +25%, n5 -19%, n4 +5%, n8 +2%, n3 +2% ...
            - 0.4425973 * h[13] / H_AVG[13]
            + 0.2535107 * h[10] / H_AVG[10]
            - 0.1948162 * h[5] / H_AVG[5]
            + 0.05438917 * h[4] / H_AVG[4]
            + 0.01940842 * h[8] / H_AVG[8]
            + 0.01817665 * h[3] / H_AVG[3]
            - 0.01178376 * h[12] / H_AVG[12]
            + 0.005317755 * h[0] / H_AVG[0]
        ),
    ]]


def classify(pt, eta, phi):
    s = logits(jet_layer_4(quantities(pt, eta, phi)))
    m = max(s)
    e = [math.exp(x - m) for x in s]
    p = [x / sum(e) for x in e]
    return CLASSES[s.index(m)], s, p


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:8] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:8] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:8] + [0.0] * 0
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
