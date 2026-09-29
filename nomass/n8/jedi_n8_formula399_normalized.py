"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.1%   (on for 89% of jets)
  neuron  9:  11.7%   (on for 70% of jets)
  neuron  7:  10.1%   (on for 62% of jets)
  neuron  3:   8.6%   (on for 29% of jets)
  neuron 10:   8.2%   (on for 88% of jets)
  neuron  2:   7.2%   (on for 88% of jets)
  neuron  6:   7.1%   (on for 50% of jets)
  neuron  5:   7.1%   (on for 71% of jets)
  neuron 11:   6.5%   (on for 73% of jets)
  neuron 14:   4.7%   (on for 34% of jets)
  neuron  0:   4.1%   (on for 44% of jets)
  neuron  1:   3.3%   (on for 60% of jets)
  neuron  4:   3.1%   (on for 48% of jets)
  neuron 15:   2.7%   (on for 34% of jets)
  neuron  8:   1.0%   (on for 36% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.3% (the network: 65.8%); same class as the network for 86.7% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        max_dr=max(dr[i] for i in real),
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_6=z[6],
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sj2_dr=subjets(2)["dr"][0],
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 40.68;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 40.67861 * (-0.3269532
        + 0.4184535 * max(0.0, 0.019 - Q.sum_z_dr2) / 0.01329852   # +41.8%  sum_z_dr2 < 0.019
        - 0.1138733 * max(0.0, 0.00734 - Q.sum_zz_dr2) / 0.003647409   # -11.4%  sum_zz_dr2 < 0.00734
        + 0.08166742 * max(0.0, 0.0129 - Q.sum_z_dr2) / 0.007947649   # +8.2%  sum_z_dr2 < 0.0129
        - 0.04582362 * max(0.0, Q.pt_7 - 20.3) / 14.91233   # -4.6%  pt_7 > 20.3
        + 0.04085552 * max(0.0, Q.sum_zz_dr2 - 0.00729) / 0.0022071   # +4.1%  sum_zz_dr2 > 0.00729
        + 0.04026495 * max(0.0, Q.z_7 - 0.0362) / 0.01887007   # +4.0%  z_7 > 0.0362
        - 0.03299916 * max(0.0, 6.71 - Q.log_sum_pt) / 0.1997559   # -3.3%  log_sum_pt < 6.71
        - 0.02610717 * max(0.0, 0.0154 - Q.sum_z_dr2_top3) / 0.01102807   # -2.6%  sum_z_dr2_top3 < 0.0154
        - 0.02594421 * max(0.0, 0.00569 - Q.sum_z_dr2) / 0.002289315   # -2.6%  sum_z_dr2 < 0.00569
        - 0.02211507 * max(0.0, Q.sum_z_dr2 - 0.0184) / 0.0008032235   # -2.2%  sum_z_dr2 > 0.0184
        + 0.02153196 * max(0.0, 0.00131 - Q.lam2) / 0.001076032   # +2.2%  lam2 < 0.00131
        - 0.02127359 * max(0.0, 0.175 - Q.sj3_dr_max) / 0.05181918   # -2.1%  sj3_dr_max < 0.175
        + 0.01921514 * max(0.0, Q.LHA - 0.253) / 0.03850468   # +1.9%  LHA > 0.253
        - 0.01660476 * max(0.0, 0.24 - Q.LHA) / 0.04248167   # -1.7%  LHA < 0.24
        + 0.01524689 * max(0.0, 0.0435 - Q.max_dr) / 0.004398741   # +1.5%  max_dr < 0.0435
        - 0.01461793 * max(0.0, 0.219 - Q.sj2_dr) / 0.08706256   # -1.5%  sj2_dr < 0.219
        + 0.01186421 * max(0.0, Q.lam1 - 0.00872) / 0.001774336   # +1.2%  lam1 > 0.00872
        - 0.009954929 * max(0.0, 0.00113 - Q.C2_b2) / 0.0005640009   # -1.0%  C2_b2 < 0.00113
        + 0.007328539 * max(0.0, 0.117 - Q.planar_flow) * max(0.0, 1.26 - Q.D2_b2) / 0.03836741   # +0.7%  planar_flow < 0.117 and D2_b2 < 1.26
        - 0.007081848 * max(0.0, 0.0379 - Q.z_7) / 0.00333812   # -0.7%  z_7 < 0.0379
        - 0.005116372 * max(0.0, 714.0 - Q.sum_pt) * max(0.0, 1.07 - Q.D2_b2) / 34.74572   # -0.5%  sum_pt < 714 and D2_b2 < 1.07
        + 0.002059877 * max(0.0, 0.0234 - Q.z_7) / 0.0007350257   # +0.2%  z_7 < 0.0234
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 13.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.41102 * (-0.05756459
        - 0.1155659 * max(0.0, 0.101 - Q.sum_z_dr) / 0.04783507   # -11.6%  sum_z_dr < 0.101
        - 0.09390012 * max(0.0, 0.00722 - Q.sum_z_dr2) / 0.003322682   # -9.4%  sum_z_dr2 < 0.00722
        + 0.07502423 * max(0.0, 0.0723 - Q.sum_z_dr) / 0.02540787   # +7.5%  sum_z_dr < 0.0723
        + 0.07497819 * Q.pt_5 / 47.65565   # +7.5%  pt_5
        - 0.05993647 * max(0.0, 0.0663 - Q.z_7) / 0.01743621   # -6.0%  z_7 < 0.0663
        + 0.05725655 * max(0.0, Q.sum_pt - 626.0) / 123.2534   # +5.7%  sum_pt > 626
        + 0.05526133 * max(0.0, 0.101 - Q.tau1) / 0.04690575   # +5.5%  tau1 < 0.101
        + 0.05501661 * Q.z_6 / 0.0599861   # +5.5%  z_6
        + 0.05231924 * max(0.0, 0.066 - Q.z_7) * max(0.0, 0.0158 - Q.sum_z_dr2_top2) / 0.0002300506   # +5.2%  z_7 < 0.066 and sum_z_dr2_top2 < 0.0158
        - 0.04805226 * max(0.0, 0.0096 - Q.lam1) * max(0.0, Q.pt_5 - 23.4) / 0.1328722   # -4.8%  lam1 < 0.0096 and pt_5 > 23.4
        + 0.04648141 * max(0.0, 6.56 - Q.log_sum_pt) / 0.1099406   # +4.6%  log_sum_pt < 6.56
        + 0.03906846 * max(0.0, 0.164 - Q.sj2_dr) / 0.05086874   # +3.9%  sj2_dr < 0.164
        - 0.02695269 * max(0.0, 560.0 - Q.sum_pt_top5) / 56.30265   # -2.7%  sum_pt_top5 < 560
        - 0.02612776 * max(0.0, 0.184 - Q.max_dr) / 0.07551725   # -2.6%  max_dr < 0.184
        + 0.02343958 * max(0.0, 0.00317 - Q.lam2) * max(0.0, 0.279 - Q.tau21_b2) / 0.0004452531   # +2.3%  lam2 < 0.00317 and tau21_b2 < 0.279
        + 0.01769933 * max(0.0, Q.log_sum_pt - 6.37) * max(0.0, 0.927 - Q.z_dr_0_0p05) / 0.05358151   # +1.8%  log_sum_pt > 6.37 and z_dr_0_0p05 < 0.927
        + 0.0164623 * max(0.0, 0.00932 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0117) / 2.266697e-05   # +1.6%  lam1 < 0.00932 and centroid_offset > 0.0117
        - 0.01646033 * max(0.0, Q.log_sum_pt - 6.62) * max(0.0, 0.00763 - Q.sum_z_dr2_top3) / 0.0003956091   # -1.6%  log_sum_pt > 6.62 and sum_z_dr2_top3 < 0.00763
        + 0.01547419 * max(0.0, Q.pt_7 - 35.0) / 4.243858   # +1.5%  pt_7 > 35
        - 0.01537921 * max(0.0, 474.0 - Q.sum_pt_top5) / 24.37954   # -1.5%  sum_pt_top5 < 474
        - 0.0128868 * max(0.0, Q.pt_7 - 33.3) * max(0.0, 4.05e-05 - Q.e3) / 0.0001234465   # -1.3%  pt_7 > 33.3 and e3 < 4.05e-05
        - 0.01181354 * max(0.0, 0.0106 - Q.lam1_plus_lam2) * max(0.0, 0.251 - Q.planar_flow) / 0.0005127238   # -1.2%  lam1_plus_lam2 < 0.0106 and planar_flow < 0.251
        - 0.01174395 * max(0.0, 0.00037 - Q.lam1) / 5.565315e-05   # -1.2%  lam1 < 0.00037
        + 0.009888532 * max(0.0, Q.pt_7 - 41.7) / 1.839325   # +1.0%  pt_7 > 41.7
        - 0.004969918 * max(0.0, 0.0573 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.0393) / 0.00178213   # -0.5%  z_7 < 0.0573 and z_dr_0p05_0p1 > 0.0393
        - 0.004848845 * max(0.0, 0.0126 - Q.centroid_offset) / 0.003369325   # -0.5%  centroid_offset < 0.0126
        + 0.004614399 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0292 - Q.tau21_b2) / 0.0004872741   # +0.5%  log_sum_pt > 6.6 and tau21_b2 < 0.0292
        - 0.003759186 * max(0.0, 0.0467 - Q.z_7) * max(0.0, 0.3 - Q.D2_b2) / 0.0003908103   # -0.4%  z_7 < 0.0467 and D2_b2 < 0.3
        - 0.002760553 * max(0.0, Q.pt_7 - 53.4) / 0.3335301   # -0.3%  pt_7 > 53.4
        - 0.001858154 * max(0.0, Q.e3 - 5.9e-05) / 5.370634e-05   # -0.2%  e3 > 5.9e-05
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 17.33;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.3347 * (-0.1315281
        + 0.2258617 * Q.pt_7 / 34.64819   # +22.6%  pt_7
        + 0.1494358 * max(0.0, 0.00875 - Q.lam1_plus_lam2) / 0.004505086   # +14.9%  lam1_plus_lam2 < 0.00875
        + 0.1325828 * max(0.0, 6.84 - Q.log_sum_pt) / 0.3052168   # +13.3%  log_sum_pt < 6.84
        - 0.1266024 * max(0.0, 0.00785 - Q.sum_zz_dr2) / 0.004041647   # -12.7%  sum_zz_dr2 < 0.00785
        - 0.08028993 * max(0.0, Q.z_7 - 0.0262) / 0.02702528   # -8.0%  z_7 > 0.0262
        - 0.07292688 * max(0.0, Q.LHA - 0.109) / 0.1374093   # -7.3%  LHA > 0.109
        - 0.05625366 * max(0.0, 6.77 - Q.log_sum_pt) * max(0.0, 1.06 - Q.D2_b2) / 0.1263135   # -5.6%  log_sum_pt < 6.77 and D2_b2 < 1.06
        + 0.05264207 * max(0.0, 0.00707 - Q.lam1) * max(0.0, 0.272 - Q.max_dr) / 0.0006660835   # +5.3%  lam1 < 0.00707 and max_dr < 0.272
        + 0.04296298 * max(0.0, 6.64 - Q.log_sum_pt) * max(0.0, 1.08 - Q.D2_b2) / 0.07906056   # +4.3%  log_sum_pt < 6.64 and D2_b2 < 1.08
        - 0.01413339 * max(0.0, 0.0084 - Q.lam1) * max(0.0, 60.6 - Q.pt_6) / 0.0917596   # -1.4%  lam1 < 0.0084 and pt_6 < 60.6
        - 0.01290688 * max(0.0, 0.000796 - Q.sum_zz_dr2) / 0.0002052632   # -1.3%  sum_zz_dr2 < 0.000796
        + 0.009728823 * max(0.0, Q.sum_pt - 827.0) / 27.82941   # +1.0%  sum_pt > 827
        + 0.008770803 * max(0.0, Q.log_sum_pt - 6.9) / 0.003849095   # +0.9%  log_sum_pt > 6.9
        - 0.007424129 * max(0.0, Q.sum_pt - 988.0) / 4.42251   # -0.7%  sum_pt > 988
        - 0.004397752 * max(0.0, 0.00817 - Q.sum_z_dr) / 0.0002833967   # -0.4%  sum_z_dr < 0.00817
        + 0.001859364 * max(0.0, Q.sum_pt_top5 - 800.0) * max(0.0, 1.85 - Q.D2_b2) / 6.991654   # +0.2%  sum_pt_top5 > 800 and D2_b2 < 1.85
        - 0.001220702 * max(0.0, Q.log_sum_pt - 6.92) * max(0.0, 2.49 - Q.D2_b2) / 0.002918689   # -0.1%  log_sum_pt > 6.92 and D2_b2 < 2.49
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 17.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.18019 * (-0.01420241
        - 0.248289 * max(0.0, 0.00778 - Q.sum_z_dr2) / 0.003741799   # -24.8%  sum_z_dr2 < 0.00778
        - 0.1596752 * max(0.0, 0.251 - Q.sj3_dr_max) / 0.103911   # -16.0%  sj3_dr_max < 0.251
        + 0.1347617 * max(0.0, 0.00733 - Q.lam1) / 0.003486793   # +13.5%  lam1 < 0.00733
        + 0.1140476 * max(0.0, 0.189 - Q.sj3_dr_max) / 0.05973653   # +11.4%  sj3_dr_max < 0.189
        + 0.08169059 * max(0.0, 0.112 - Q.tau1) / 0.05591472   # +8.2%  tau1 < 0.112
        + 0.07773058 * max(0.0, Q.sum_zz_dr2 - 0.00702) / 0.002275001   # +7.8%  sum_zz_dr2 > 0.00702
        - 0.04449177 * max(0.0, Q.sum_zz_dr2 - 0.0111) / 0.001516621   # -4.4%  sum_zz_dr2 > 0.0111
        + 0.03683502 * max(0.0, 0.319 - Q.tau21_b2) / 0.1797819   # +3.7%  tau21_b2 < 0.319
        + 0.03434996 * max(0.0, Q.LHA - 0.316) / 0.01453544   # +3.4%  LHA > 0.316
        - 0.020399 * max(0.0, Q.log_sum_pt - 6.43) / 0.1717935   # -2.0%  log_sum_pt > 6.43
        - 0.01911122 * max(0.0, Q.sum_z_dr - 0.0992) / 0.005811227   # -1.9%  sum_z_dr > 0.0992
        + 0.009023171 * max(0.0, Q.sj2_dr - 0.0981) * max(0.0, Q.eccentricity - 0.968) / 0.001040401   # +0.9%  sj2_dr > 0.0981 and eccentricity > 0.968
        + 0.007931394 * max(0.0, 617.0 - Q.sum_pt) / 30.41581   # +0.8%  sum_pt < 617
        - 0.005861959 * max(0.0, Q.lam1_plus_lam2 - 0.00442) * max(0.0, Q.pt_6 - 32.1) / 0.03024311   # -0.6%  lam1_plus_lam2 > 0.00442 and pt_6 > 32.1
        - 0.004352107 * max(0.0, Q.sj2_dr - 0.112) * max(0.0, 0.00265 - Q.sum_z_dr2_top2) / 2.932157e-05   # -0.4%  sj2_dr > 0.112 and sum_z_dr2_top2 < 0.00265
        + 0.001449794 * max(0.0, Q.log_sum_pt - 6.89) / 0.0043469   # +0.1%  log_sum_pt > 6.89
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 40.46;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 40.45814 * (-0.1450882
        + 0.1360996 * max(0.0, 0.0159 - Q.sum_zz_dr2) / 0.01090364   # +13.6%  sum_zz_dr2 < 0.0159
        + 0.1275962 * max(0.0, 0.0002 - Q.e3) / 0.0001613221   # +12.8%  e3 < 0.0002
        + 0.1236408 * max(0.0, 0.0247 - Q.C2_b2) / 0.02184399   # +12.4%  C2_b2 < 0.0247
        - 0.09918906 * max(0.0, 0.0179 - Q.lam1_plus_lam2) / 0.01230983   # -9.9%  lam1_plus_lam2 < 0.0179
        - 0.06462944 * max(0.0, 0.00864 - Q.sum_z_dr2) / 0.00441687   # -6.5%  sum_z_dr2 < 0.00864
        - 0.06295445 * max(0.0, 0.00791 - Q.lam1_plus_lam2) / 0.003841659   # -6.3%  lam1_plus_lam2 < 0.00791
        + 0.05492914 * max(0.0, 0.416 - Q.LHA) / 0.1749867   # +5.5%  LHA < 0.416
        + 0.0548623 * max(0.0, Q.e2 - 0.0187) / 0.01469951   # +5.5%  e2 > 0.0187
        - 0.05479303 * max(0.0, 0.00125 - Q.lam2) / 0.001021578   # -5.5%  lam2 < 0.00125
        - 0.03557004 * max(0.0, 0.35 - Q.sj3_dr_max) * max(0.0, 6.81 - Q.log_sum_pt) / 0.04360902   # -3.6%  sj3_dr_max < 0.35 and log_sum_pt < 6.81
        - 0.03492541 * Q.sum_zz_dr2 / 0.00588757   # -3.5%  sum_zz_dr2
        - 0.02281484 * max(0.0, 0.000183 - Q.e3) * max(0.0, 841.0 - Q.sum_pt) / 0.01781942   # -2.3%  e3 < 0.000183 and sum_pt < 841
        + 0.01909532 * max(0.0, 0.00739 - Q.sum_z_dr2) * max(0.0, 6.7 - Q.log_sum_pt) / 0.0004364752   # +1.9%  sum_z_dr2 < 0.00739 and log_sum_pt < 6.7
        + 0.01625481 * max(0.0, Q.max_dr - 0.126) / 0.0337251   # +1.6%  max_dr > 0.126
        + 0.01292917 * max(0.0, 0.238 - Q.N2) / 0.05884028   # +1.3%  N2 < 0.238
        - 0.01125869 * max(0.0, 0.000288 - Q.lam2) / 0.0001844153   # -1.1%  lam2 < 0.000288
        - 0.009245202 * max(0.0, 0.239 - Q.N2) * max(0.0, 55.4 - Q.pt_7) / 1.15803   # -0.9%  N2 < 0.239 and pt_7 < 55.4
        - 0.00904177 * max(0.0, Q.sj2_dr - 0.205) / 0.02177459   # -0.9%  sj2_dr > 0.205
        + 0.00793492 * max(0.0, 0.00563 - Q.sum_z_dr2_top2) * max(0.0, Q.centroid_offset - 0.00501) / 2.3433e-05   # +0.8%  sum_z_dr2_top2 < 0.00563 and centroid_offset > 0.00501
        + 0.007276355 * max(0.0, 580.0 - Q.sum_pt_top3) / 147.1939   # +0.7%  sum_pt_top3 < 580
        - 0.005551292 * max(0.0, Q.sj2_dr - 0.236) * max(0.0, 1.61 - Q.D2_b2) / 0.01395   # -0.6%  sj2_dr > 0.236 and D2_b2 < 1.61
        + 0.004134599 * max(0.0, 0.00202 - Q.sum_z_dr2) / 0.0005594588   # +0.4%  sum_z_dr2 < 0.00202
        - 0.003496265 * max(0.0, Q.sj3_dr_min - 0.0771) / 0.0146431   # -0.3%  sj3_dr_min > 0.0771
        + 0.003430225 * max(0.0, Q.n_dr_0p1_0p2 - 1.16) / 0.8360273   # +0.3%  n_dr_0p1_0p2 > 1.16
        - 0.003331452 * max(0.0, Q.e3 - 0.000198) / 3.662619e-05   # -0.3%  e3 > 0.000198
        - 0.003000229 * max(0.0, Q.sum_pt_top3 - 585.0) / 30.04547   # -0.3%  sum_pt_top3 > 585
        + 0.00296722 * max(0.0, Q.log_sum_pt - 6.76) / 0.0203817   # +0.3%  log_sum_pt > 6.76
        - 0.001667137 * max(0.0, 0.000149 - Q.lam1) / 1.338279e-05   # -0.2%  lam1 < 0.000149
        - 0.001519478 * max(0.0, 0.0159 - Q.sum_z_dr) / 0.001536882   # -0.2%  sum_z_dr < 0.0159
        - 0.001465485 * max(0.0, 0.018 - Q.centroid_offset) / 0.006472794   # -0.1%  centroid_offset < 0.018
        + 0.001102035 * max(0.0, -0.0236 - Q.mean_eta) / 0.000906225   # +0.1%  mean_eta < -0.0236
        - 0.0009431342 * max(0.0, 9e-05 - Q.sum_zz_dr2) / 8.153302e-06   # -0.1%  sum_zz_dr2 < 9e-05
        + 0.0009220741 * max(0.0, Q.sj3_dr_max - 0.38) * max(0.0, 1.69 - Q.D2_b2) / 0.001541546   # +0.1%  sj3_dr_max > 0.38 and D2_b2 < 1.69
        + 0.0007357115 * max(0.0, 0.0333 - Q.max_dr) / 0.002272177   # +0.1%  max_dr < 0.0333
        + 0.0006931231 * max(0.0, 0.271 - Q.N2) * max(0.0, -0.0195 - Q.mean_phi) / 8.987971e-05   # +0.1%  N2 < 0.271 and mean_phi < -0.0195
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 7.006;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.005982 * (-0.2455045
        + 0.2430046 * max(0.0, Q.z_7 - 0.0273) / 0.02607176   # +24.3%  z_7 > 0.0273
        + 0.219921 * max(0.0, 47.2 - Q.pt_7) / 13.39794   # +22.0%  pt_7 < 47.2
        - 0.1204886 * max(0.0, 730.0 - Q.sum_pt) / 77.44415   # -12.0%  sum_pt < 730
        + 0.1098153 * max(0.0, 0.00278 - Q.sum_z_dr2) * max(0.0, 0.019 - Q.centroid_offset) / 9.545464e-06   # +11.0%  sum_z_dr2 < 0.00278 and centroid_offset < 0.019
        - 0.09369565 * max(0.0, 0.0338 - Q.centroid_offset) / 0.01875514   # -9.4%  centroid_offset < 0.0338
        + 0.08047243 * max(0.0, 0.0356 - Q.e2) * max(0.0, 0.0714 - Q.z_7) / 0.0003809381   # +8.0%  e2 < 0.0356 and z_7 < 0.0714
        + 0.04779661 * max(0.0, Q.log_sum_pt - 6.59) / 0.07805645   # +4.8%  log_sum_pt > 6.59
        + 0.02604005 * max(0.0, 0.000994 - Q.sum_z_dr2) * max(0.0, 2.46 - Q.n_dr_0p2_0p4) / 0.0005257526   # +2.6%  sum_z_dr2 < 0.000994 and n_dr_0p2_0p4 < 2.46
        - 0.02281071 * max(0.0, 0.173 - Q.LHA) * max(0.0, 6.87 - Q.log_sum_pt) / 0.002520685   # -2.3%  LHA < 0.173 and log_sum_pt < 6.87
        - 0.01409169 * max(0.0, Q.pt_7 - 45.0) / 1.162852   # -1.4%  pt_7 > 45
        + 0.01345855 * max(0.0, 28.3 - Q.pt_6) / 1.098955   # +1.3%  pt_6 < 28.3
        - 0.008404679 * max(0.0, 0.0516 - Q.e2) * max(0.0, Q.log_sum_pt - 6.87) / 0.0002336628   # -0.8%  e2 < 0.0516 and log_sum_pt > 6.87
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 24.65;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.64591 * (0.02219435
        + 0.1911942 * Q.pt_7 / 34.64819   # +19.1%  pt_7
        - 0.08072831 * max(0.0, Q.log_sum_pt - 6.35) / 0.2308147   # -8.1%  log_sum_pt > 6.35
        - 0.07583875 * max(0.0, 0.00951 - Q.sum_z_dr2) / 0.005120863   # -7.6%  sum_z_dr2 < 0.00951
        - 0.07394236 * max(0.0, Q.sj3_dr_max - 0.0711) * max(0.0, Q.eccentricity - 0.864) / 0.009491545   # -7.4%  sj3_dr_max > 0.0711 and eccentricity > 0.864
        - 0.07152181 * max(0.0, Q.z_7 - 0.0397) / 0.01632148   # -7.2%  z_7 > 0.0397
        - 0.06527323 * max(0.0, 0.0117 - Q.lam1_plus_lam2) / 0.00693413   # -6.5%  lam1_plus_lam2 < 0.0117
        + 0.05571925 * max(0.0, 0.00558 - Q.sum_zz_dr2) / 0.002417696   # +5.6%  sum_zz_dr2 < 0.00558
        + 0.05487599 * max(0.0, 0.0979 - Q.tau1) / 0.0444891   # +5.5%  tau1 < 0.0979
        - 0.04786832 * max(0.0, 0.00597 - Q.lam1_plus_lam2) / 0.002462961   # -4.8%  lam1_plus_lam2 < 0.00597
        - 0.04533305 * max(0.0, 0.191 - Q.max_dr) / 0.08096191   # -4.5%  max_dr < 0.191
        + 0.04481344 * max(0.0, 0.00668 - Q.C2_b2) / 0.005185296   # +4.5%  C2_b2 < 0.00668
        + 0.03778445 * max(0.0, Q.sj3_dr_max - 0.167) * max(0.0, Q.eccentricity - 0.855) / 0.004066516   # +3.8%  sj3_dr_max > 0.167 and eccentricity > 0.855
        + 0.02191444 * max(0.0, 0.0405 - Q.z_7) / 0.004091678   # +2.2%  z_7 < 0.0405
        + 0.02112292 * max(0.0, Q.LHA - 0.234) * max(0.0, Q.eccentricity - 0.873) / 0.003718526   # +2.1%  LHA > 0.234 and eccentricity > 0.873
        + 0.01754552 * max(0.0, 6.34 - Q.log_sum_pt) / 0.03573764   # +1.8%  log_sum_pt < 6.34
        + 0.01469115 * max(0.0, Q.log_sum_pt - 6.67) / 0.04531624   # +1.5%  log_sum_pt > 6.67
        + 0.01430117 * max(0.0, 0.136 - Q.sj2_dr) / 0.0380632   # +1.4%  sj2_dr < 0.136
        + 0.01365094 * max(0.0, Q.centroid_offset - 0.0126) / 0.007953663   # +1.4%  centroid_offset > 0.0126
        - 0.01332557 * max(0.0, Q.lam1_plus_lam2 - 0.00613) / 0.002880885   # -1.3%  lam1_plus_lam2 > 0.00613
        - 0.01248386 * max(0.0, Q.sum_pt_top5 - 676.0) / 40.64414   # -1.2%  sum_pt_top5 > 676
        - 0.01102375 * max(0.0, Q.centroid_offset - 0.00913) * max(0.0, 1050.0 - Q.sum_pt) / 4.432141   # -1.1%  centroid_offset > 0.00913 and sum_pt < 1050
        + 0.006364218 * max(0.0, 31.8 - Q.pt_6) / 1.800826   # +0.6%  pt_6 < 31.8
        + 0.004924222 * max(0.0, Q.centroid_offset - 0.00872) * max(0.0, 0.0119 - Q.mean_phi2) / 7.055927e-05   # +0.5%  centroid_offset > 0.00872 and mean_phi2 < 0.0119
        - 0.002377296 * max(0.0, Q.C2_b2 - 0.00696) / 0.001921004   # -0.2%  C2_b2 > 0.00696
        + 0.001381819 * max(0.0, Q.sum_pt - 967.0) / 5.629121   # +0.1%  sum_pt > 967
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 67.43;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 67.43266 * (-0.02001997
        + 0.3052536 * max(0.0, 0.0187 - Q.sum_z_dr2) / 0.01302789   # +30.5%  sum_z_dr2 < 0.0187
        - 0.1340508 * max(0.0, 0.0168 - Q.sum_zz_dr2) / 0.01170907   # -13.4%  sum_zz_dr2 < 0.0168
        - 0.07575727 * max(0.0, 0.00759 - Q.lam1_plus_lam2) / 0.003597545   # -7.6%  lam1_plus_lam2 < 0.00759
        - 0.07529348 * max(0.0, Q.sum_z_dr - 0.0171) / 0.04339521   # -7.5%  sum_z_dr > 0.0171
        + 0.07129508 * max(0.0, Q.tau1 - 0.0139) / 0.05109051   # +7.1%  tau1 > 0.0139
        - 0.05311187 * max(0.0, 0.088 - Q.sum_z_dr) / 0.03699871   # -5.3%  sum_z_dr < 0.088
        + 0.04205447 * max(0.0, 0.0959 - Q.tau1) / 0.04296734   # +4.2%  tau1 < 0.0959
        - 0.03447053 * max(0.0, 0.183 - Q.sj2_dr) / 0.06182019   # -3.4%  sj2_dr < 0.183
        - 0.03243492 * max(0.0, 0.00605 - Q.lam1_plus_lam2) / 0.002513992   # -3.2%  lam1_plus_lam2 < 0.00605
        + 0.03170153 * max(0.0, 0.158 - Q.sj2_dr) / 0.04782368   # +3.2%  sj2_dr < 0.158
        + 0.02906127 * max(0.0, 0.0057 - Q.sum_zz_dr2) / 0.00249323   # +2.9%  sum_zz_dr2 < 0.0057
        - 0.02667864 * max(0.0, 0.000186 - Q.e3) / 0.0001486786   # -2.7%  e3 < 0.000186
        - 0.02221428 * max(0.0, Q.lam1_plus_lam2 - 0.0042) / 0.003726289   # -2.2%  lam1_plus_lam2 > 0.0042
        - 0.0214914 * max(0.0, Q.e2 - 0.0073) / 0.02215936   # -2.1%  e2 > 0.0073
        + 0.01186705 * max(0.0, Q.LHA - 0.248) / 0.04103726   # +1.2%  LHA > 0.248
        + 0.006441604 * max(0.0, Q.sum_z_dr2 - 0.00287) * max(0.0, 0.225 - Q.planar_flow) / 0.0004300737   # +0.6%  sum_z_dr2 > 0.00287 and planar_flow < 0.225
        + 0.004976257 * max(0.0, Q.e3 - 8.88e-05) / 4.891577e-05   # +0.5%  e3 > 8.88e-05
        + 0.00351342 * max(0.0, 0.196 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.48) / 0.01081823   # +0.4%  planar_flow < 0.196 and log_sum_pt > 6.48
        - 0.00334003 * max(0.0, 0.0136 - Q.sum_zz_dr2) * max(0.0, 46.4 - Q.pt_7) / 0.1173058   # -0.3%  sum_zz_dr2 < 0.0136 and pt_7 < 46.4
        - 0.002150785 * max(0.0, Q.lam2 - 0.00064) / 0.0003681044   # -0.2%  lam2 > 0.00064
        - 0.002107186 * max(0.0, Q.sj3_dr_max - 0.276) / 0.01623922   # -0.2%  sj3_dr_max > 0.276
        + 0.001743064 * max(0.0, 0.00725 - Q.lam1_plus_lam2) * max(0.0, -0.00506 - Q.mean_phi) / 6.640644e-06   # +0.2%  lam1_plus_lam2 < 0.00725 and mean_phi < -0.00506
        - 0.001572216 * max(0.0, 0.214 - Q.sj3_dr_max) * max(0.0, Q.eccentricity - 0.944) / 0.000869006   # -0.2%  sj3_dr_max < 0.214 and eccentricity > 0.944
        + 0.001538462 * max(0.0, Q.sum_z_dr2 - 0.0128) * max(0.0, Q.eccentricity - 0.952) / 1.872609e-05   # +0.2%  sum_z_dr2 > 0.0128 and eccentricity > 0.952
        + 0.001501764 * max(0.0, 0.00635 - Q.lam1_plus_lam2) * max(0.0, -0.00704 - Q.mean_eta) / 4.20199e-06   # +0.2%  lam1_plus_lam2 < 0.00635 and mean_eta < -0.00704
        + 0.001298379 * max(0.0, 0.00655 - Q.lam1_plus_lam2) * max(0.0, Q.mean_eta - 0.00792) / 4.034707e-06   # +0.1%  lam1_plus_lam2 < 0.00655 and mean_eta > 0.00792
        + 0.001103322 * max(0.0, 0.0067 - Q.lam1_plus_lam2) * max(0.0, Q.mean_phi - 0.00994) / 3.248906e-06   # +0.1%  lam1_plus_lam2 < 0.0067 and mean_phi > 0.00994
        - 0.001096895 * max(0.0, 0.225 - Q.planar_flow) * max(0.0, 0.0414 - Q.z_7) / 0.0003456382   # -0.1%  planar_flow < 0.225 and z_7 < 0.0414
        - 0.0005345902 * max(0.0, 0.0736 - Q.tau1) * max(0.0, Q.z_dr_0p05_0p1 - 0.489) / 0.0003612108   # -0.1%  tau1 < 0.0736 and z_dr_0p05_0p1 > 0.489
        + 0.0003458357 * max(0.0, 735.0 - Q.sum_pt) / 80.13958   # +0.0%  sum_pt < 735
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 18.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.77815 * (-0.07828248
        + 0.1859923 * max(0.0, 0.00745 - Q.sum_z_dr2) / 0.003492591   # +18.6%  sum_z_dr2 < 0.00745
        + 0.1366093 * max(0.0, 0.00435 - Q.lam1_plus_lam2) / 0.001554708   # +13.7%  lam1_plus_lam2 < 0.00435
        - 0.07517754 * max(0.0, 0.00734 - Q.lam1) / 0.003494294   # -7.5%  lam1 < 0.00734
        - 0.07069391 * max(0.0, 0.0553 - Q.sum_z_dr) * max(0.0, 0.00561 - Q.lam1_plus_lam2) / 7.808827e-05   # -7.1%  sum_z_dr < 0.0553 and lam1_plus_lam2 < 0.00561
        - 0.06699361 * max(0.0, 0.00715 - Q.sum_zz_dr2) / 0.003504222   # -6.7%  sum_zz_dr2 < 0.00715
        - 0.0506366 * max(0.0, 0.00426 - Q.lam1) / 0.001551161   # -5.1%  lam1 < 0.00426
        + 0.03594167 * max(0.0, 0.346 - Q.sj3_dr_max) / 0.1844038   # +3.6%  sj3_dr_max < 0.346
        + 0.03306521 * max(0.0, Q.sum_zz_dr2 - 9.35e-05) / 0.005802835   # +3.3%  sum_zz_dr2 > 9.35e-05
        + 0.02596924 * max(0.0, Q.sum_pt_top5 - 690.0) / 36.39211   # +2.6%  sum_pt_top5 > 690
        - 0.02541112 * max(0.0, Q.log_sum_pt - 6.7) / 0.03560998   # -2.5%  log_sum_pt > 6.7
        - 0.02354621 * max(0.0, 0.138 - Q.sj3_dr_max) / 0.03537233   # -2.4%  sj3_dr_max < 0.138
        - 0.01897602 * max(0.0, 0.0362 - Q.C2) / 0.01535924   # -1.9%  C2 < 0.0362
        - 0.01868474 * max(0.0, Q.sum_pt_top5 - 691.0) * max(0.0, 0.00966 - Q.sum_z_dr2_top3) / 0.3050998   # -1.9%  sum_pt_top5 > 691 and sum_z_dr2_top3 < 0.00966
        - 0.01864005 * max(0.0, 0.0011 - Q.lam2) / 0.0008861408   # -1.9%  lam2 < 0.0011
        + 0.01832412 * max(0.0, 0.178 - Q.max_dr) / 0.07094701   # +1.8%  max_dr < 0.178
        + 0.01674966 * max(0.0, 0.000188 - Q.lam2) / 0.0001069822   # +1.7%  lam2 < 0.000188
        - 0.01523694 * max(0.0, Q.sum_z_dr - 0.0618) / 0.01598444   # -1.5%  sum_z_dr > 0.0618
        + 0.01510819 * max(0.0, 0.00652 - Q.sum_z_dr2) * max(0.0, 0.0245 - Q.centroid_offset) / 3.984604e-05   # +1.5%  sum_z_dr2 < 0.00652 and centroid_offset < 0.0245
        - 0.0131021 * max(0.0, 0.00732 - Q.sum_z_dr2_top3) / 0.004080152   # -1.3%  sum_z_dr2_top3 < 0.00732
        + 0.01288317 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.00632 - Q.lam1_plus_lam2) / 0.0001591592   # +1.3%  log_sum_pt > 6.7 and lam1_plus_lam2 < 0.00632
        - 0.01188345 * max(0.0, 0.0219 - Q.sum_z_dr) / 0.002936174   # -1.2%  sum_z_dr < 0.0219
        - 0.01106928 * max(0.0, 0.00507 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.00798) / 9.713108e-06   # -1.1%  sum_z_dr2 < 0.00507 and centroid_offset > 0.00798
        + 0.00912844 * max(0.0, 0.0993 - Q.sj3_dr_max) / 0.02134685   # +0.9%  sj3_dr_max < 0.0993
        - 0.008770517 * max(0.0, 0.00544 - Q.sum_z_dr2) * max(0.0, 44.2 - Q.pt_7) / 0.02643564   # -0.9%  sum_z_dr2 < 0.00544 and pt_7 < 44.2
        + 0.007964716 * max(0.0, 0.000518 - Q.lam1) / 9.064401e-05   # +0.8%  lam1 < 0.000518
        - 0.007556525 * max(0.0, 0.0202 - Q.C2) / 0.004793835   # -0.8%  C2 < 0.0202
        - 0.006750041 * max(0.0, 0.161 - Q.sj2_dr) / 0.04932034   # -0.7%  sj2_dr < 0.161
        + 0.005971246 * max(0.0, 5.94e-05 - Q.lam2) / 1.981077e-05   # +0.6%  lam2 < 5.94e-05
        - 0.005803821 * max(0.0, 0.00277 - Q.mean_phi) / 0.006986219   # -0.6%  mean_phi < 0.00277
        + 0.005330211 * max(0.0, 0.214 - Q.N2) / 0.04655418   # +0.5%  N2 < 0.214
        + 0.005110423 * max(0.0, 47.4 - Q.pt_7) / 13.57345   # +0.5%  pt_7 < 47.4
        - 0.004657236 * max(0.0, 0.0486 - Q.sj3_dr_min) / 0.02449699   # -0.5%  sj3_dr_min < 0.0486
        - 0.004576858 * max(0.0, Q.lam1_plus_lam2 - 0.0184) / 0.0008032235   # -0.5%  lam1_plus_lam2 > 0.0184
        + 0.003042961 * max(0.0, 0.198 - Q.sj3_dr_max) / 0.06537891   # +0.3%  sj3_dr_max < 0.198
        - 0.003027031 * max(0.0, 0.038 - Q.planar_flow) / 0.005572749   # -0.3%  planar_flow < 0.038
        + 0.002974805 * max(0.0, 0.118 - Q.N2) / 0.0101015   # +0.3%  N2 < 0.118
        + 0.002674125 * max(0.0, 0.000334 - Q.sum_z_dr2) / 4.328889e-05   # +0.3%  sum_z_dr2 < 0.000334
        + 0.0025466 * max(0.0, Q.sum_z_dr - 0.124) / 0.002686542   # +0.3%  sum_z_dr > 0.124
        + 0.002336311 * max(0.0, -0.011 - Q.mean_phi) / 0.002161162   # +0.2%  mean_phi < -0.011
        - 0.002156613 * max(0.0, 0.00491 - Q.sum_z_dr2) * max(0.0, 6.55 - Q.log_sum_pt) / 0.0001014967   # -0.2%  sum_z_dr2 < 0.00491 and log_sum_pt < 6.55
        + 0.001946193 * max(0.0, Q.centroid_offset - 0.0343) / 0.002076471   # +0.2%  centroid_offset > 0.0343
        - 0.001430068 * max(0.0, 29.9 - Q.pt_6) / 1.384228   # -0.1%  pt_6 < 29.9
        + 0.001428242 * max(0.0, Q.mean_phi - -0.000447) / 0.005622586   # +0.1%  mean_phi > -0.000447
        + 0.001213581 * max(0.0, Q.sum_zz_dr2 - 0.0164) / 0.0008378238   # +0.1%  sum_zz_dr2 > 0.0164
        + 0.001037514 * max(0.0, 0.00814 - Q.centroid_offset) / 0.001453925   # +0.1%  centroid_offset < 0.00814
        + 0.001011599 * max(0.0, Q.mean_eta - 0.00151) / 0.0048583   # +0.1%  mean_eta > 0.00151
        + 0.0005129412 * max(0.0, 0.166 - Q.sj3_dr_max) * max(0.0, Q.lam1_plus_lam2 - 0.00264) / 4.24321e-06   # +0.1%  sj3_dr_max < 0.166 and lam1_plus_lam2 > 0.00264
        - 0.0003470081 * max(0.0, Q.sum_z_dr2_top2 - 0.0146) / 0.001075276   # -0.0%  sum_z_dr2_top2 > 0.0146
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 12.79;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.79175 * (-0.2650145
        + 0.4852441 * max(0.0, 0.00712 - Q.sum_z_dr2) / 0.003249802   # +48.5%  sum_z_dr2 < 0.00712
        - 0.115169 * max(0.0, 0.0903 - Q.tau1) / 0.03887105   # -11.5%  tau1 < 0.0903
        + 0.109723 * max(0.0, 6.86 - Q.log_sum_pt) / 0.3233984   # +11.0%  log_sum_pt < 6.86
        + 0.09493024 * max(0.0, 0.0197 - Q.centroid_offset) * max(0.0, 0.0643 - Q.C2) / 0.0003489437   # +9.5%  centroid_offset < 0.0197 and C2 < 0.0643
        - 0.04471682 * max(0.0, 0.122 - Q.sj3_dr_max) / 0.02933366   # -4.5%  sj3_dr_max < 0.122
        + 0.03976168 * max(0.0, Q.lam2 - 0.000442) / 0.0003973606   # +4.0%  lam2 > 0.000442
        - 0.02236547 * max(0.0, 6.83 - Q.log_sum_pt) * max(0.0, Q.planar_flow - 0.0176) / 0.08058975   # -2.2%  log_sum_pt < 6.83 and planar_flow > 0.0176
        - 0.02024553 * max(0.0, 0.0181 - Q.centroid_offset) * max(0.0, 0.336 - Q.D2_b2) / 0.0006556349   # -2.0%  centroid_offset < 0.0181 and D2_b2 < 0.336
        - 0.01610303 * max(0.0, Q.lam2 - 0.000494) * max(0.0, Q.planar_flow - 0.122) / 0.0002343413   # -1.6%  lam2 > 0.000494 and planar_flow > 0.122
        - 0.01415462 * max(0.0, 0.0677 - Q.sum_z_dr) * max(0.0, -0.00128 - Q.mean_phi) / 5.711746e-05   # -1.4%  sum_z_dr < 0.0677 and mean_phi < -0.00128
        - 0.01226166 * max(0.0, 0.0062 - Q.lam1_plus_lam2) * max(0.0, -0.00284 - Q.mean_eta) / 6.53534e-06   # -1.2%  lam1_plus_lam2 < 0.0062 and mean_eta < -0.00284
        - 0.01130704 * max(0.0, 0.00608 - Q.lam1_plus_lam2) * max(0.0, Q.mean_eta - 0.00407) / 5.457992e-06   # -1.1%  lam1_plus_lam2 < 0.00608 and mean_eta > 0.00407
        - 0.009043182 * max(0.0, 0.07 - Q.sum_z_dr) * max(0.0, Q.mean_phi - 0.00476) / 3.881817e-05   # -0.9%  sum_z_dr < 0.07 and mean_phi > 0.00476
        + 0.004974769 * max(0.0, Q.sj3_dr_max - 0.309) / 0.01102877   # +0.5%  sj3_dr_max > 0.309
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 11.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.67048 * (1.011098
        - 0.6781303 * max(0.0, 0.0161 - Q.lam1) / 0.01090097   # -67.8%  lam1 < 0.0161
        - 0.1203989 * max(0.0, Q.lam1 - 0.00417) / 0.003252576   # -12.0%  lam1 > 0.00417
        + 0.04420634 * max(0.0, Q.lam1 - 0.0159) / 0.0007380674   # +4.4%  lam1 > 0.0159
        - 0.04260959 * max(0.0, Q.LHA - 0.209) / 0.06326646   # -4.3%  LHA > 0.209
        - 0.02764748 * max(0.0, Q.sj2_dr - 0.166) * max(0.0, 0.71 - Q.planar_flow) / 0.01772853   # -2.8%  sj2_dr > 0.166 and planar_flow < 0.71
        + 0.02325487 * max(0.0, Q.lam2 - 0.000281) / 0.0004267224   # +2.3%  lam2 > 0.000281
        - 0.01733855 * max(0.0, Q.LHA - 0.312) * max(0.0, 0.503 - Q.planar_flow) / 0.00379642   # -1.7%  LHA > 0.312 and planar_flow < 0.503
        + 0.01220601 * max(0.0, Q.C2 - 0.0499) / 0.004998246   # +1.2%  C2 > 0.0499
        - 0.01027002 * max(0.0, Q.lam2 - 4.09e-05) * max(0.0, 6.69 - Q.log_sum_pt) / 0.0002041841   # -1.0%  lam2 > 4.09e-05 and log_sum_pt < 6.69
        - 0.008693253 * max(0.0, 46.8 - Q.pt_6) * max(0.0, 6.55 - Q.log_sum_pt) / 0.9393928   # -0.9%  pt_6 < 46.8 and log_sum_pt < 6.55
        - 0.008309236 * max(0.0, 0.0645 - Q.z_6) / 0.01159961   # -0.8%  z_6 < 0.0645
        - 0.00693549 * max(0.0, Q.e3 - -7.29e-05) * max(0.0, 32.6 - Q.pt_7) / 0.000402689   # -0.7%  e3 > -7.29e-05 and pt_7 < 32.6
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 31.55;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.55 * (0.01099841
        + 0.2864295 * max(0.0, 0.00897 - Q.lam1_plus_lam2) / 0.004682306   # +28.6%  lam1_plus_lam2 < 0.00897
        - 0.2589747 * max(0.0, 0.00841 - Q.sum_zz_dr2) / 0.004489369   # -25.9%  sum_zz_dr2 < 0.00841
        + 0.1224639 * max(0.0, 0.013 - Q.lam1_plus_lam2) / 0.008032715   # +12.2%  lam1_plus_lam2 < 0.013
        - 0.07627478 * max(0.0, 0.077 - Q.sum_z_dr) / 0.02861438   # -7.6%  sum_z_dr < 0.077
        + 0.05422719 * max(0.0, 0.00126 - Q.lam2) / 0.001030643   # +5.4%  lam2 < 0.00126
        - 0.04295 * max(0.0, 0.00944 - Q.C2_b2) / 0.007655778   # -4.3%  C2_b2 < 0.00944
        - 0.03779839 * max(0.0, 0.00457 - Q.lam1_plus_lam2) / 0.001665558   # -3.8%  lam1_plus_lam2 < 0.00457
        + 0.02558086 * max(0.0, 0.274 - Q.planar_flow) / 0.1220993   # +2.6%  planar_flow < 0.274
        + 0.02076324 * max(0.0, 0.0546 - Q.tau1) / 0.01746881   # +2.1%  tau1 < 0.0546
        - 0.02019515 * max(0.0, Q.sj3_dr_max - 0.174) / 0.04551121   # -2.0%  sj3_dr_max > 0.174
        - 0.01517543 * max(0.0, 0.0165 - Q.centroid_offset) * max(0.0, 0.0801 - Q.sj3_dr_min) / 0.000325704   # -1.5%  centroid_offset < 0.0165 and sj3_dr_min < 0.0801
        - 0.01448189 * max(0.0, 40.2 - Q.pt_7) / 7.79699   # -1.4%  pt_7 < 40.2
        - 0.01107854 * max(0.0, 0.286 - Q.planar_flow) * max(0.0, 861.0 - Q.sum_pt) / 19.6364   # -1.1%  planar_flow < 0.286 and sum_pt < 861
        + 0.007123601 * max(0.0, 0.0166 - Q.centroid_offset) * max(0.0, 0.0549 - Q.tau21_b2) / 8.779282e-05   # +0.7%  centroid_offset < 0.0166 and tau21_b2 < 0.0549
        + 0.004046124 * max(0.0, Q.log_sum_pt - 6.46) * max(0.0, 0.0543 - Q.z_7) / 0.002989583   # +0.4%  log_sum_pt > 6.46 and z_7 < 0.0543
        - 0.002436765 * max(0.0, 0.265 - Q.planar_flow) * max(0.0, 6.99e-06 - Q.e3) / 9.932809e-08   # -0.2%  planar_flow < 0.265 and e3 < 6.99e-06
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 6.547;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.546625 * (-0.4429764
        - 0.2180988 * max(0.0, 0.00406 - Q.lam1_plus_lam2) / 0.001413674   # -21.8%  lam1_plus_lam2 < 0.00406
        + 0.1168133 * max(0.0, 0.000272 - Q.e3) / 0.0002269236   # +11.7%  e3 < 0.000272
        - 0.1046281 * max(0.0, 0.0132 - Q.sum_z_dr2) / 0.008203127   # -10.5%  sum_z_dr2 < 0.0132
        + 0.09239156 * max(0.0, Q.pt_7 - 17.5) / 17.4813   # +9.2%  pt_7 > 17.5
        - 0.07881525 * max(0.0, Q.tau1 - 0.0357) / 0.03659389   # -7.9%  tau1 > 0.0357
        - 0.07221124 * max(0.0, 0.0573 - Q.e2) / 0.03110131   # -7.2%  e2 < 0.0573
        + 0.06843718 * max(0.0, Q.sum_zz_dr2 - 0.00587) / 0.002620073   # +6.8%  sum_zz_dr2 > 0.00587
        + 0.05129976 * max(0.0, Q.centroid_offset - 0.00287) / 0.01447587   # +5.1%  centroid_offset > 0.00287
        + 0.03976232 * max(0.0, Q.log_sum_pt - 6.7) / 0.03560998   # +4.0%  log_sum_pt > 6.7
        + 0.02416312 * max(0.0, Q.lam1_plus_lam2 - 0.0179) / 0.0008550643   # +2.4%  lam1_plus_lam2 > 0.0179
        - 0.01762164 * max(0.0, Q.lam1 - 0.00624) / 0.002383518   # -1.8%  lam1 > 0.00624
        + 0.01669548 * max(0.0, Q.n_dr_0p2_0p4 - 0.699) / 0.211002   # +1.7%  n_dr_0p2_0p4 > 0.699
        - 0.0115985 * max(0.0, Q.sj2_dr - 0.144) / 0.0474569   # -1.2%  sj2_dr > 0.144
        + 0.01003566 * max(0.0, Q.centroid_offset - 0.0231) / 0.00410623   # +1.0%  centroid_offset > 0.0231
        - 0.009609639 * max(0.0, Q.sum_z_dr2 - 0.0207) * max(0.0, 816.0 - Q.sum_pt) / 0.1772132   # -1.0%  sum_z_dr2 > 0.0207 and sum_pt < 816
        - 0.007691134 * max(0.0, 3.64 - Q.n_dr_0p1_0p2) / 2.517548   # -0.8%  n_dr_0p1_0p2 < 3.64
        + 0.007405712 * max(0.0, Q.sum_zz_dr2 - 0.0194) / 0.0005611391   # +0.7%  sum_zz_dr2 > 0.0194
        - 0.006013071 * max(0.0, Q.ptdr0_4 - 14.9) * max(0.0, 967.0 - Q.sum_pt) / 59.73493   # -0.6%  ptdr0_4 > 14.9 and sum_pt < 967
        - 0.005962257 * max(0.0, 545.0 - Q.sum_pt) / 13.45954   # -0.6%  sum_pt < 545
        - 0.005953156 * max(0.0, Q.z_dr_0p2_0p4 - 0.13) / 0.01123143   # -0.6%  z_dr_0p2_0p4 > 0.13
        - 0.005488149 * max(0.0, Q.sum_z_dr2 - 0.0178) * max(0.0, 52.1 - Q.pt_7) / 0.01522409   # -0.5%  sum_z_dr2 > 0.0178 and pt_7 < 52.1
        - 0.004787251 * max(0.0, Q.sum_z_dr2 - 0.0162) * max(0.0, 0.684 - Q.planar_flow) / 0.0003447782   # -0.5%  sum_z_dr2 > 0.0162 and planar_flow < 0.684
        + 0.004356777 * max(0.0, Q.sj3_dr_max - 0.291) / 0.01371259   # +0.4%  sj3_dr_max > 0.291
        + 0.003920967 * max(0.0, Q.ptdr0_4 - 14.6) * max(0.0, 820.0 - Q.sum_pt) / 35.70111   # +0.4%  ptdr0_4 > 14.6 and sum_pt < 820
        + 0.003686062 * max(0.0, Q.lam1_plus_lam2 - 0.0103) * max(0.0, Q.log_sum_pt - 6.45) / 4.914717e-05   # +0.4%  lam1_plus_lam2 > 0.0103 and log_sum_pt > 6.45
        + 0.003302984 * max(0.0, Q.sum_z_dr - 0.146) / 0.001019971   # +0.3%  sum_z_dr > 0.146
        + 0.003080041 * max(0.0, Q.sj2_dr - 0.266) / 0.008882763   # +0.3%  sj2_dr > 0.266
        - 0.002452262 * max(0.0, -0.0164 - Q.mean_phi) / 0.001420712   # -0.2%  mean_phi < -0.0164
        + 0.001905406 * max(0.0, Q.ptdr0_4 - 17.1) / 0.1113748   # +0.2%  ptdr0_4 > 17.1
        + 0.0009630262 * max(0.0, Q.centroid_offset - 0.0554) / 0.0005679794   # +0.1%  centroid_offset > 0.0554
        - 0.0006044293 * max(0.0, Q.e2 - 0.0722) * max(0.0, Q.sum_pt - 606.0) / 0.01576483   # -0.1%  e2 > 0.0722 and sum_pt > 606
        - 0.0002457909 * max(0.0, Q.sum_z_dr2_top2 - 0.0289) * max(0.0, Q.sum_pt - 573.0) / 0.00502844   # -0.0%  sum_z_dr2_top2 > 0.0289 and sum_pt > 573
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 16.13;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.13095 * (0.3645166
        + 0.2546832 * max(0.0, 0.0156 - Q.lam1) / 0.01045364   # +25.5%  lam1 < 0.0156
        - 0.1164273 * max(0.0, Q.LHA - 0.18) / 0.08237211   # -11.6%  LHA > 0.18
        - 0.1056094 * max(0.0, Q.pt_6 - 23.4) / 17.40123   # -10.6%  pt_6 > 23.4
        + 0.09005233 * max(0.0, Q.z_6 - 0.0438) / 0.01955088   # +9.0%  z_6 > 0.0438
        - 0.08739838 * max(0.0, 0.111 - Q.tau1) / 0.05507106   # -8.7%  tau1 < 0.111
        - 0.08367857 * max(0.0, 0.134 - Q.sum_z_dr) * max(0.0, 6.85 - Q.log_sum_pt) / 0.01947785   # -8.4%  sum_z_dr < 0.134 and log_sum_pt < 6.85
        - 0.05084053 * max(0.0, 7.24e-05 - Q.e3) * max(0.0, 0.0374 - Q.centroid_offset) / 1.197235e-06   # -5.1%  e3 < 7.24e-05 and centroid_offset < 0.0374
        - 0.04356057 * max(0.0, 0.146 - Q.sum_z_dr) * max(0.0, 42.9 - Q.pt_7) / 0.9319278   # -4.4%  sum_z_dr < 0.146 and pt_7 < 42.9
        + 0.03164615 * max(0.0, Q.sum_pt_top5 - 599.0) * max(0.0, 39.6 - Q.pt_7) / 1046.071   # +3.2%  sum_pt_top5 > 599 and pt_7 < 39.6
        - 0.03031578 * max(0.0, 707.0 - Q.sum_pt) / 65.72882   # -3.0%  sum_pt < 707
        - 0.02655652 * max(0.0, 0.0446 - Q.z_6) / 0.003569849   # -2.7%  z_6 < 0.0446
        + 0.02349021 * max(0.0, Q.log_sum_pt - 6.56) / 0.09287243   # +2.3%  log_sum_pt > 6.56
        - 0.01858202 * max(0.0, 0.000269 - Q.lam2) / 0.0001693479   # -1.9%  lam2 < 0.000269
        - 0.01596879 * max(0.0, 0.0208 - Q.lam1) * max(0.0, 26.6 - Q.pt_7) / 0.02809071   # -1.6%  lam1 < 0.0208 and pt_7 < 26.6
        + 0.01529777 * max(0.0, 0.0254 - Q.sum_z_dr) / 0.003892235   # +1.5%  sum_z_dr < 0.0254
        - 0.003542306 * max(0.0, Q.lam1 - 0.017) / 0.0006293038   # -0.4%  lam1 > 0.017
        - 0.001545764 * max(0.0, Q.sum_pt - 1010.0) / 3.497147   # -0.2%  sum_pt > 1010
        + 0.0008044333 * max(0.0, Q.sum_pt - 1020.0) * max(0.0, Q.z_6 - 0.0217) / 0.04915257   # +0.1%  sum_pt > 1020 and z_6 > 0.0217
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 65.85;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 65.84717 * (0.06423966
        + 0.2034391 * max(0.0, 0.0184 - Q.sum_z_dr2) / 0.01275799   # +20.3%  sum_z_dr2 < 0.0184
        - 0.1447834 * max(0.0, 0.0169 - Q.sum_zz_dr2) / 0.01179898   # -14.5%  sum_zz_dr2 < 0.0169
        + 0.0942928 * max(0.0, 0.148 - Q.planar_flow) / 0.05089274   # +9.4%  planar_flow < 0.148
        - 0.08557568 * max(0.0, Q.eccentricity - 0.96) / 0.01398242   # -8.6%  eccentricity > 0.96
        - 0.06455439 * max(0.0, Q.sum_zz_dr2 - 0.00209) / 0.004474447   # -6.5%  sum_zz_dr2 > 0.00209
        - 0.0600263 * max(0.0, Q.tau1 - 0.0134) / 0.05146565   # -6.0%  tau1 > 0.0134
        - 0.0540377 * max(0.0, 0.00731 - Q.sum_z_dr2) / 0.00338879   # -5.4%  sum_z_dr2 < 0.00731
        + 0.03703492 * max(0.0, Q.tau1 - 0.0352) / 0.03689327   # +3.7%  tau1 > 0.0352
        - 0.03104858 * max(0.0, 0.0818 - Q.sum_z_dr) / 0.03214562   # -3.1%  sum_z_dr < 0.0818
        + 0.0271189 * max(0.0, 0.152 - Q.sj2_dr) / 0.04497993   # +2.7%  sj2_dr < 0.152
        - 0.02300291 * max(0.0, 0.221 - Q.sj2_dr) / 0.08857756   # -2.3%  sj2_dr < 0.221
        + 0.02091017 * max(0.0, Q.sum_zz_dr2 - 0.00612) / 0.002535682   # +2.1%  sum_zz_dr2 > 0.00612
        - 0.02059902 * max(0.0, 0.211 - Q.sj3_dr_max) / 0.07411951   # -2.1%  sj3_dr_max < 0.211
        - 0.01798015 * max(0.0, Q.sum_z_dr - 0.0891) / 0.007493306   # -1.8%  sum_z_dr > 0.0891
        + 0.01455453 * max(0.0, Q.z_7 - 0.0245) / 0.02852306   # +1.5%  z_7 > 0.0245
        + 0.01442031 * max(0.0, Q.LHA - 0.313) / 0.01526587   # +1.4%  LHA > 0.313
        + 0.01187359 * max(0.0, Q.sum_zz_dr2 - 0.002) * max(0.0, Q.log_sum_pt - 6.24) / 0.0008353018   # +1.2%  sum_zz_dr2 > 0.002 and log_sum_pt > 6.24
        + 0.01185742 * max(0.0, Q.lam1_plus_lam2 - 0.0129) / 0.001492883   # +1.2%  lam1_plus_lam2 > 0.0129
        - 0.01078124 * max(0.0, 0.0043 - Q.lam1_plus_lam2) / 0.001529987   # -1.1%  lam1_plus_lam2 < 0.0043
        + 0.009601458 * max(0.0, 0.0254 - Q.e2) / 0.007883153   # +1.0%  e2 < 0.0254
        + 0.007685682 * max(0.0, Q.e2 - 0.0417) / 0.004961573   # +0.8%  e2 > 0.0417
        - 0.007653537 * max(0.0, Q.pt_7 - 26.8) / 9.437524   # -0.8%  pt_7 > 26.8
        - 0.005830105 * max(0.0, 0.113 - Q.planar_flow) * max(0.0, 0.0163 - Q.lam1) / 0.0003226016   # -0.6%  planar_flow < 0.113 and lam1 < 0.0163
        - 0.005748817 * max(0.0, Q.sum_z_dr2 - 0.00797) * max(0.0, Q.log_sum_pt - 6.25) / 0.0002540559   # -0.6%  sum_z_dr2 > 0.00797 and log_sum_pt > 6.25
        - 0.003197281 * max(0.0, Q.z_dr_0p05_0p1 - 0.746) / 0.0235758   # -0.3%  z_dr_0p05_0p1 > 0.746
        + 0.002658226 * max(0.0, Q.z_dr_0p05_0p1 - 0.736) * max(0.0, 0.973 - Q.n_dr_0p2_0p4) / 0.02207272   # +0.3%  z_dr_0p05_0p1 > 0.736 and n_dr_0p2_0p4 < 0.973
        - 0.002560696 * max(0.0, 0.000414 - Q.lam1) / 6.560879e-05   # -0.3%  lam1 < 0.000414
        - 0.002322228 * max(0.0, Q.centroid_offset - 0.0282) / 0.002998278   # -0.2%  centroid_offset > 0.0282
        - 0.001294753 * max(0.0, Q.log_sum_pt - 6.8) / 0.01319749   # -0.1%  log_sum_pt > 6.8
        - 0.001294691 * max(0.0, 0.115 - Q.planar_flow) * max(0.0, 0.0182 - Q.centroid_offset) / 0.0002261319   # -0.1%  planar_flow < 0.115 and centroid_offset < 0.0182
        + 0.001138577 * max(0.0, 0.00645 - Q.mean_phi) / 0.009562767   # +0.1%  mean_phi < 0.00645
        - 0.0008628194 * max(0.0, 513.0 - Q.sum_pt) / 8.673927   # -0.1%  sum_pt < 513
        + 0.0002600187 * max(0.0, Q.D2_b2 - 1.21) / 0.750943   # +0.0%  D2_b2 > 1.21
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 33.3;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.30019 * (-0.1585577
        - 0.1457408 * max(0.0, 0.113 - Q.tau1) / 0.05676251   # -14.6%  tau1 < 0.113
        + 0.1361389 * max(0.0, 0.0181 - Q.sum_z_dr2) / 0.01248884   # +13.6%  sum_z_dr2 < 0.0181
        + 0.1302836 * max(0.0, 0.104 - Q.tau1) / 0.04930076   # +13.0%  tau1 < 0.104
        - 0.127747 * max(0.0, 0.0076 - Q.sum_z_dr2) / 0.003605085   # -12.8%  sum_z_dr2 < 0.0076
        + 0.05586921 * max(0.0, 0.0498 - Q.e2) / 0.02460919   # +5.6%  e2 < 0.0498
        + 0.04500789 * max(0.0, 0.00735 - Q.lam1) / 0.003501802   # +4.5%  lam1 < 0.00735
        - 0.04139166 * max(0.0, Q.sum_z_dr - 0.0839) / 0.00856118   # -4.1%  sum_z_dr > 0.0839
        + 0.03901695 * max(0.0, 0.00343 - Q.lam2) / 0.00305711   # +3.9%  lam2 < 0.00343
        + 0.03320448 * max(0.0, 0.00644 - Q.sum_zz_dr2) / 0.00298842   # +3.3%  sum_zz_dr2 < 0.00644
        - 0.03188586 * max(0.0, 0.236 - Q.sj3_dr_max) / 0.09233087   # -3.2%  sj3_dr_max < 0.236
        + 0.02835354 * max(0.0, Q.sum_zz_dr2 - 0.00448) / 0.003189791   # +2.8%  sum_zz_dr2 > 0.00448
        - 0.02418679 * max(0.0, 0.00494 - Q.lam1_plus_lam2) / 0.001860103   # -2.4%  lam1_plus_lam2 < 0.00494
        + 0.02378965 * max(0.0, Q.LHA - 0.318) / 0.01407104   # +2.4%  LHA > 0.318
        + 0.02149764 * max(0.0, 0.178 - Q.sj3_dr_max) / 0.05342354   # +2.1%  sj3_dr_max < 0.178
        + 0.01858247 * max(0.0, Q.tau1 - 0.0509) / 0.02812726   # +1.9%  tau1 > 0.0509
        + 0.01336893 * max(0.0, Q.LHA - 0.243) / 0.04364588   # +1.3%  LHA > 0.243
        - 0.01299261 * max(0.0, 0.000163 - Q.e3) / 0.0001280048   # -1.3%  e3 < 0.000163
        + 0.00859948 * max(0.0, 0.206 - Q.N2) * max(0.0, Q.LHA - 0.275) / 0.001847511   # +0.9%  N2 < 0.206 and LHA > 0.275
        - 0.007667144 * max(0.0, 0.228 - Q.N2) * max(0.0, 0.189 - Q.sj2_dr) / 0.0008865185   # -0.8%  N2 < 0.228 and sj2_dr < 0.189
        + 0.006943944 * max(0.0, 0.215 - Q.planar_flow) * max(0.0, Q.sum_pt_top5 - 434.0) / 15.41564   # +0.7%  planar_flow < 0.215 and sum_pt_top5 > 434
        - 0.005807221 * max(0.0, 0.0213 - Q.centroid_offset) / 0.00871088   # -0.6%  centroid_offset < 0.0213
        - 0.00578491 * max(0.0, 5.52e-05 - Q.lam2) / 1.75126e-05   # -0.6%  lam2 < 5.52e-05
        - 0.005585736 * max(0.0, 0.252 - Q.N2) * max(0.0, Q.sum_z_dr2 - 0.00799) / 0.0001706477   # -0.6%  N2 < 0.252 and sum_z_dr2 > 0.00799
        - 0.004564087 * max(0.0, Q.z_dr_0p05_0p1 - 0.719) / 0.0280415   # -0.5%  z_dr_0p05_0p1 > 0.719
        + 0.003944314 * max(0.0, Q.z_dr_0p05_0p1 - 0.702) * max(0.0, 0.0509 - Q.z_dr_0p2_0p4) / 0.001415371   # +0.4%  z_dr_0p05_0p1 > 0.702 and z_dr_0p2_0p4 < 0.0509
        + 0.003704005 * max(0.0, 0.23 - Q.N2) * max(0.0, 0.161 - Q.sj2_dr) / 0.0003099097   # +0.4%  N2 < 0.23 and sj2_dr < 0.161
        - 0.003343136 * max(0.0, 0.186 - Q.N2) * max(0.0, Q.LHA - 0.331) / 0.0004840306   # -0.3%  N2 < 0.186 and LHA > 0.331
        - 0.002408565 * max(0.0, Q.sum_zz_dr2 - 0.0195) / 0.0005531426   # -0.2%  sum_zz_dr2 > 0.0195
        - 0.002127567 * max(0.0, 757.0 - Q.sum_pt) / 92.61226   # -0.2%  sum_pt < 757
        + 0.002099256 * max(0.0, 0.0314 - Q.planar_flow) / 0.003862188   # +0.2%  planar_flow < 0.0314
        - 0.001909082 * max(0.0, 0.00164 - Q.sum_z_dr2_top3) * max(0.0, Q.eccentricity - 0.954) / 3.194612e-06   # -0.2%  sum_z_dr2_top3 < 0.00164 and eccentricity > 0.954
        + 0.001524334 * max(0.0, Q.log_sum_pt - 6.82) / 0.01042312   # +0.2%  log_sum_pt > 6.82
        + 0.001395981 * max(0.0, 0.00889 - Q.mean_phi) / 0.01144986   # +0.1%  mean_phi < 0.00889
        - 0.001224035 * max(0.0, Q.sj3_dr_max - 0.326) / 0.008822636   # -0.1%  sj3_dr_max > 0.326
        - 0.001052074 * max(0.0, Q.sum_pt_top3 - 676.0) / 13.95787   # -0.1%  sum_pt_top3 > 676
        - 0.000566808 * max(0.0, 0.0133 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.0491) / 9.830632e-07   # -0.1%  sum_z_dr2 < 0.0133 and centroid_offset > 0.0491
        - 0.000346316 * max(0.0, 0.227 - Q.N2) * max(0.0, 24.2 - Q.pt_7) / 0.02972265   # -0.0%  N2 < 0.227 and pt_7 < 24.2
        - 0.0003329377 * max(0.0, Q.lam2 - 0.000251) / 0.0004330816   # -0.0%  lam2 > 0.000251
        + 1.11633e-05 * max(0.0, 0.0244 - Q.sum_z_dr) / 0.003609127   # +0.0%  sum_z_dr < 0.0244
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9932109243697479, 0.7529630252100841, 2.08085, 0.9514886554621849, 1.1843382352941176, 1.7674128151260504, 0.9630113445378151, 1.8225926470588236, 0.24051302521008402, 2.985120588235294, 2.0277047268907564, 2.140271638655462, 0.1000563025210084, 3.2970409663865548, 0.518253781512605, 0.3727357142857143]
T = [2.3302280626313028, 1.3639875147715335, 3.3731083360688023, 2.4911216238839287, 2.859818953518908]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +38%, n9 +22%, n5 -14%, n1 +13%, n0 -7%, n6 +5% ...
            + 0.3837029 * h[2] / H_AVG[2]
            + 0.2201791 * h[9] / H_AVG[9]
            - 0.1422135 * h[5] / H_AVG[5]
            + 0.1262221 * h[1] / H_AVG[1]
            - 0.06659829 * h[0] / H_AVG[0]
            + 0.04520131 * h[6] / H_AVG[6]
            - 0.01588281 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -19%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5556801 * h[9] / H_AVG[9]
            - 0.1858251 * h[10] / H_AVG[10]
            + 0.08825331 * h[6] / H_AVG[6]
            - 0.08140229 * h[4] / H_AVG[4]
            + 0.06073917 * h[5] / H_AVG[5]
            + 0.01707932 * h[15] / H_AVG[15]
            + 0.01102068 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +12%, n14 -12%, n0 +10%, n6 -9% ...
            + 0.2379413 * h[11] / H_AVG[11]
            - 0.1410403 * h[3] / H_AVG[3]
            + 0.1181973 * h[7] / H_AVG[7]
            - 0.1152321 * h[14] / H_AVG[14]
            + 0.1012171 * h[0] / H_AVG[0]
            - 0.08921772 * h[6] / H_AVG[6]
            - 0.07597023 * h[15] / H_AVG[15]
            + 0.06872687 * h[13] / H_AVG[13]
            - 0.02765551 * h[9] / H_AVG[9]
            - 0.01782577 * h[8] / H_AVG[8]
            - 0.00697579 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +34%, n3 -21%, n6 -14%, n14 +8%, n13 +7%, n1 +4% ...
            + 0.3429541 * h[7] / H_AVG[7]
            - 0.2148479 * h[3] / H_AVG[3]
            - 0.1449665 * h[6] / H_AVG[6]
            + 0.07801513 * h[14] / H_AVG[14]
            + 0.07237982 * h[13] / H_AVG[13]
            + 0.03778233 * h[1] / H_AVG[1]
            - 0.03744699 * h[9] / H_AVG[9]
            + 0.03714248 * h[4] / H_AVG[4]
            - 0.02337901 * h[15] / H_AVG[15]
            + 0.0110857 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +27%, n5 -15%, n4 +5%, n3 +2%, n12 -2% ...
            - 0.4683593 * h[13] / H_AVG[13]
            + 0.2658872 * h[10] / H_AVG[10]
            - 0.1545039 * h[5] / H_AVG[5]
            + 0.05176631 * h[4] / H_AVG[4]
            + 0.02079434 * h[3] / H_AVG[3]
            - 0.01749347 * h[12] / H_AVG[12]
            + 0.0157689 * h[8] / H_AVG[8]
            + 0.00542654 * h[0] / H_AVG[0]
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
