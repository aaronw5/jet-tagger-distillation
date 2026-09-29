"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.0%   (on for 89% of jets)
  neuron  9:  11.8%   (on for 69% of jets)
  neuron  7:  10.3%   (on for 61% of jets)
  neuron  3:   8.8%   (on for 25% of jets)
  neuron  5:   7.8%   (on for 70% of jets)
  neuron 10:   7.7%   (on for 68% of jets)
  neuron  2:   7.5%   (on for 93% of jets)
  neuron  6:   7.1%   (on for 29% of jets)
  neuron 11:   5.7%   (on for 73% of jets)
  neuron  0:   4.4%   (on for 43% of jets)
  neuron 14:   4.3%   (on for 25% of jets)
  neuron  1:   3.3%   (on for 55% of jets)
  neuron 15:   2.9%   (on for 27% of jets)
  neuron  4:   2.9%   (on for 49% of jets)
  neuron  8:   1.1%   (on for 28% of jets)
  neuron 12:   0.2%   (on for 4% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 87.0% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        z_top3_slots=sum(pt[:3]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 19.31;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.31474 * (-0.0916601
        + 0.1248448 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +12.5%  lam1 < 0.012
        + 0.1134433 * max(0.0, Q.sj3_dr_max - 0.1426152) / 0.06279029   # +11.3%  sj3_dr_max > 0.1426
        - 0.08984733 * max(0.0, 0.01713288 - Q.tau2) / 0.008065871   # -9.0%  tau2 < 0.01713
        + 0.07145609 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +7.1%  e3 < 0.0005117
        - 0.06554783 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -6.6%  lam1 < 0.004184
        - 0.05884671 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # -5.9%  sj3_dr_max > 0.169
        + 0.05850305 * max(0.0, Q.LHA - 0.1546891) / 0.1006902   # +5.9%  LHA > 0.1547
        - 0.05813644 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -5.8%  centroid_offset > 0.00679
        - 0.04926721 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # -4.9%  sj3_dr_max > 0.1986
        - 0.04833987 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -4.8%  LHA > 0.3033
        - 0.04727893 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.0005116989 - Q.e3) / 0.03717154   # -4.7%  sum_pt < 763.8 and e3 < 0.0005117
        + 0.03161724 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +3.2%  planar_flow < 0.1484
        - 0.02573024 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -2.6%  lam1 < 0.005954
        + 0.02533134 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +2.5%  tau1 < 0.05357
        + 0.02344277 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, Q.centroid_offset - 0.01096064) / 1.576925   # +2.3%  sum_pt < 763.8 and centroid_offset > 0.01096
        + 0.02128971 * max(0.0, 0.04505724 - Q.planar_flow) * max(0.0, 0.05932655 - Q.tau21_b2) / 0.0003618271   # +2.1%  planar_flow < 0.04506 and tau21_b2 < 0.05933
        + 0.0159446 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, Q.z_7 - 0.01685855) / 5.053524   # +1.6%  sum_pt < 763.8 and z_7 > 0.01686
        + 0.01416041 * max(0.0, 0.004183811 - Q.lam1) * max(0.0, 763.825 - Q.sum_pt) / 0.08579672   # +1.4%  lam1 < 0.004184 and sum_pt < 763.8
        - 0.01308796 * max(0.0, Q.log_sum_pt - 6.766778) / 0.01900249   # -1.3%  log_sum_pt > 6.767
        + 0.0129281 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 0.0887688 - Q.sj3_z3) / 9.661655e-07   # +1.3%  e3 < 8.148e-05 and sj3_z3 < 0.08877
        - 0.01140041 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # -1.1%  log_sum_pt > 6.843
        + 0.008339115 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +0.8%  sum_pt_top5 > 791.1
        + 0.006168758 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 0.1230713 - Q.tau21_b2) / 0.4912655   # +0.6%  sum_pt_top5 > 791.1 and tau21_b2 < 0.1231
        - 0.00255318 * max(0.0, 0.004183811 - Q.lam1) * max(0.0, 0.7459513 - Q.D2) / 6.576264e-06   # -0.3%  lam1 < 0.004184 and D2 < 0.746
        - 0.002494628 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.04019753 - Q.tau21_b2) / 5.609094e-05   # -0.2%  log_sum_pt > 6.843 and tau21_b2 < 0.0402
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 16.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.57381 * (0.08020343
        - 0.08173764 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -8.2%  lam1 < 0.008376
        - 0.07444724 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.006756161 - Q.girth2_top3) / 0.0004453888   # -7.4%  log_sum_pt > 6.573 and girth2_top3 < 0.006756
        + 0.07121992 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +7.1%  log_sum_pt > 6.378
        + 0.06334599 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # +6.3%  tau1 < 0.1028
        - 0.06086766 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.001130645 - Q.lam2) / 4.482383e-06   # -6.1%  lam1 < 0.008376 and lam2 < 0.001131
        + 0.05870237 * max(0.0, Q.log_sum_pt - 6.572938) / 0.08631512   # +5.9%  log_sum_pt > 6.573
        - 0.05851454 * max(0.0, 0.06473447 - Q.z_7) / 0.01631636   # -5.9%  z_7 < 0.06473
        - 0.05782524 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 0.1550922 - Q.dr_max_012) / 0.0002929109   # -5.8%  lam1 < 0.005433 and dr_max_012 < 0.1551
        + 0.04498256 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +4.5%  sj2_dr < 0.1592
        + 0.03943849 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, 0.007929074 - Q.girth2_top3) / 9.283431e-05   # +3.9%  z_7 < 0.06473 and girth2_top3 < 0.007929
        + 0.03694676 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 1.340118e-05 - Q.e3) / 1.490664e-06   # +3.7%  log_sum_pt > 6.378 and e3 < 1.34e-05
        + 0.03601341 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +3.6%  pt_7 > 34.53
        - 0.03575621 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, 0.005180665 - Q.girth2_top2) / 5.709888e-05   # -3.6%  z_7 < 0.06473 and girth2_top2 < 0.005181
        + 0.03422594 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # +3.4%  lam1 < 0.005433
        - 0.03174517 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # -3.2%  max_dr < 0.1118
        + 0.02686989 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.1206357 - Q.dr_max_012) / 0.007392713   # +2.7%  log_sum_pt > 6.573 and dr_max_012 < 0.1206
        - 0.02680913 * max(0.0, 0.04737278 - Q.z_6) / 0.004344387   # -2.7%  z_6 < 0.04737
        - 0.02473873 * max(0.0, Q.sj3_dr_min - 0.02913153) / 0.02609727   # -2.5%  sj3_dr_min > 0.02913
        - 0.02015554 * max(0.0, 1.340118e-05 - Q.e3) / 5.081834e-06   # -2.0%  e3 < 1.34e-05
        - 0.01751829 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # -1.8%  pt_7 > 34.53 and tau2 < 0.01713
        + 0.01747083 * max(0.0, 0.04737278 - Q.z_6) * max(0.0, 0.004007842 - Q.girth2_top2) / 1.318326e-05   # +1.7%  z_6 < 0.04737 and girth2_top2 < 0.004008
        + 0.01678806 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +1.7%  lam1 < 0.002464
        - 0.01520283 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.00233503   # -1.5%  lam1 < 0.012 and n_pt_above_50 > 6
        - 0.0121977 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, Q.tau21_b2 - 0.04019753) / 0.0006219781   # -1.2%  lam1 < 0.005433 and tau21_b2 > 0.0402
        + 0.01140654 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # +1.1%  sum_pt_top5 > 840
        + 0.01096585 * max(0.0, Q.n_pt_above_50 - 6.0) * max(0.0, 8.147744e-05 - Q.e3) / 1.821055e-05   # +1.1%  n_pt_above_50 > 6 and e3 < 8.148e-05
        - 0.007909512 * max(0.0, Q.sum_pt_top5 - 839.9547) * max(0.0, 3.892127e-05 - Q.e3) / 0.0003012957   # -0.8%  sum_pt_top5 > 840 and e3 < 3.892e-05
        - 0.006197953 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.0095303 - Q.girth2_top2) / 0.03873367   # -0.6%  sum_pt > 988.4 and girth2_top2 < 0.00953
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 17.26;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.26365 * (0.2908103
        - 0.1886451 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # -18.9%  z_7 > 0.02321
        - 0.122881 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -12.3%  pt_7 < 53.44
        - 0.09860183 * Q.pt_7 / 34.64819   # -9.9%  pt_7
        + 0.09534104 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # +9.5%  girth < 0.1019
        + 0.07401032 * max(0.0, Q.pt_7 - 20.125) / 15.07003   # +7.4%  pt_7 > 20.12
        - 0.06282489 * max(0.0, 0.1222545 - Q.M2) / 0.06341994   # -6.3%  M2 < 0.1223
        + 0.06117392 * max(0.0, 6.701242 - Q.log_sum_pt) / 0.1935491   # +6.1%  log_sum_pt < 6.701
        + 0.06087142 * max(0.0, Q.z_7 - 0.02807091) / 0.02541111   # +6.1%  z_7 > 0.02807
        + 0.04084387 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # +4.1%  log_sum_pt < 6.606
        + 0.03267613 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.2507612 - Q.max_dr) / 0.00062784   # +3.3%  lam1 < 0.00733 and max_dr < 0.2508
        - 0.02304749 * max(0.0, 752.1 - Q.sum_pt_top5) / 179.9248   # -2.3%  sum_pt_top5 < 752.1
        + 0.02191301 * max(0.0, 0.0586137 - Q.z_7) / 0.01231218   # +2.2%  z_7 < 0.05861
        - 0.02160743 * max(0.0, 0.0586137 - Q.z_7) * max(0.0, 0.07897949 - Q.absphi_0) / 0.0007310394   # -2.2%  z_7 < 0.05861 and absphi_0 < 0.07898
        + 0.01676565 * max(0.0, 752.1 - Q.sum_pt_top5) * max(0.0, Q.D2_b2 - 0.1830092) / 113.9873   # +1.7%  sum_pt_top5 < 752.1 and D2_b2 > 0.183
        - 0.01365615 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -1.4%  sj2_dr < 0.1295
        - 0.01346814 * max(0.0, Q.log_sum_pt - 6.080494) * max(0.0, 8.836697e-05 - Q.mean_phi2) / 6.819601e-06   # -1.3%  log_sum_pt > 6.08 and mean_phi2 < 8.837e-05
        + 0.01266699 * max(0.0, 8.836697e-05 - Q.mean_phi2) / 1.02303e-05   # +1.3%  mean_phi2 < 8.837e-05
        - 0.01242558 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, 0.8986489 - Q.planar_flow) / 0.005489272   # -1.2%  tau1 < 0.0437 and planar_flow < 0.8986
        - 0.01115026 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, Q.D2_b2 - 0.01618699) / 0.1028513   # -1.1%  log_sum_pt < 6.606 and D2_b2 > 0.01619
        - 0.007913342 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # -0.8%  girth < 0.007674
        + 0.004469812 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.1056549 - Q.absphi_0) / 0.0003841548   # +0.4%  log_sum_pt > 6.896 and absphi_0 < 0.1057
        - 0.003046628 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.3%  sum_pt > 988.4
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 8.681;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.681277 * (0.03827999
        - 0.296355 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # -29.6%  lam2 < 0.003408
        + 0.1164807 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +11.6%  sj2_dr > 0.1683
        - 0.08772954 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # -8.8%  girth > 0.1019
        + 0.08175648 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.09482124 - Q.C2) / 0.001379963   # +8.2%  sj2_dr > 0.1683 and C2 < 0.09482
        + 0.07453358 * max(0.0, Q.sj3_dr_max - 0.213399) / 0.03061279   # +7.5%  sj3_dr_max > 0.2134
        - 0.05554225 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.log_sum_pt - 6.080494) / 0.01252901   # -5.6%  sj2_dr > 0.1683 and log_sum_pt > 6.08
        + 0.05155961 * max(0.0, Q.girth - 0.06663269) / 0.01393206   # +5.2%  girth > 0.06663
        + 0.04288259 * max(0.0, Q.girth - 0.06663269) * max(0.0, Q.eccentricity - 0.7117266) / 0.0025938   # +4.3%  girth > 0.06663 and eccentricity > 0.7117
        + 0.03514793 * max(0.0, Q.girth - 0.1245537) / 0.002632136   # +3.5%  girth > 0.1246
        - 0.0328931 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -3.3%  lam1 > 0.012
        + 0.03087703 * max(0.0, Q.LHA - 0.2931906) / 0.02125042   # +3.1%  LHA > 0.2932
        - 0.02594948 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.n_dr_0_0p05 - 2.0) / 0.04103234   # -2.6%  sj2_dr > 0.1683 and n_dr_0_0p05 > 2
        + 0.01832085 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +1.8%  tau1 > 0.1136
        - 0.01516853 * max(0.0, Q.max_dr - 0.2507612) * max(0.0, 0.6080732 - Q.z_dr_0_0p05) / 0.001567661   # -1.5%  max_dr > 0.2508 and z_dr_0_0p05 < 0.6081
        - 0.01435584 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # -1.4%  sd_rg > 0.2788
        - 0.0117486 * max(0.0, Q.sd_rg - 0.1876504) / 0.01921774   # -1.2%  sd_rg > 0.1877
        - 0.008698919 * max(0.0, Q.girth - 0.1245537) * max(0.0, Q.eccentricity - 0.7792127) / 0.0002736432   # -0.9%  girth > 0.1246 and eccentricity > 0.7792
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 30.14;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 30.13966 * (0.03091663
        + 0.2146073 * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.166579   # +21.5%  sj3_dr_min < 0.209
        + 0.1334343 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +13.3%  e3 < 0.0001869
        - 0.1313761 * max(0.0, 0.2089872 - Q.sj3_dr_min) * max(0.0, 0.003408389 - Q.lam2) / 0.000543125   # -13.1%  sj3_dr_min < 0.209 and lam2 < 0.003408
        - 0.08309645 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -8.3%  lam2 < 0.001131
        - 0.06743575 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -6.7%  lam2 < 0.0005373
        - 0.04953572 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -5.0%  lam1 < 0.006507
        - 0.04360885 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -4.4%  lam1 < 0.008376
        + 0.04115885 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +4.1%  N2 < 0.2233
        - 0.03563957 * max(0.0, 0.33555 - Q.tau21) / 0.105517   # -3.6%  tau21 < 0.3356
        + 0.03194221 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +3.2%  lam1 < 0.002464
        - 0.02756617 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -2.8%  lam1 < 0.00733
        - 0.01830873 * max(0.0, 763.825 - Q.sum_pt) / 96.68366   # -1.8%  sum_pt < 763.8
        + 0.0159306 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +1.6%  sj3_dr_max > 0.169
        - 0.01502459 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, Q.log_sum_pt - 6.267538) / 0.004076983   # -1.5%  sj3_dr_max > 0.2337 and log_sum_pt > 6.268
        - 0.01476822 * max(0.0, Q.C2 - 0.01867771) / 0.01385506   # -1.5%  C2 > 0.01868
        - 0.01330686 * max(0.0, Q.sj2_dr - 0.2179769) * max(0.0, Q.lam2 - 0.001130645) / 2.486958e-05   # -1.3%  sj2_dr > 0.218 and lam2 > 0.001131
        + 0.01311493 * max(0.0, 0.08082334 - Q.dr_0) / 0.0377444   # +1.3%  dr_0 < 0.08082
        + 0.01307299 * max(0.0, 0.08082334 - Q.dr_0) * max(0.0, Q.centroid_offset - 0.009480685) / 0.000183564   # +1.3%  dr_0 < 0.08082 and centroid_offset > 0.009481
        - 0.01226767 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.2%  sj3_dr_max > 0.2337
        + 0.009279137 * max(0.0, Q.max_dr - 0.121681) * max(0.0, 0.009032972 - Q.C2_b2) / 0.0001556695   # +0.9%  max_dr > 0.1217 and C2_b2 < 0.009033
        + 0.007361272 * max(0.0, Q.log_sum_pt - 6.327379) / 0.2487223   # +0.7%  log_sum_pt > 6.327
        - 0.005623961 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.06108421 - Q.dr_1) / 1.247729   # -0.6%  sum_pt < 763.8 and dr_1 < 0.06108
        + 0.002539654 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, 0.05966518 - Q.sj2_zsoft) / 6.432356e-05   # +0.3%  sj3_dr_max > 0.2337 and sj2_zsoft < 0.05967
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 12.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.06787 * (0.08090274
        + 0.1453963 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +14.5%  e3 < 0.0001869
        - 0.1248715 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # -12.5%  dr_0 < 0.06413
        - 0.09535176 * max(0.0, 0.03117077 - Q.centroid_offset) / 0.01649145   # -9.5%  centroid_offset < 0.03117
        + 0.06977368 * max(0.0, 0.0008722282 - Q.lam1) / 0.0001882079   # +7.0%  lam1 < 0.0008722
        - 0.06368287 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -6.4%  sum_pt < 739.5
        - 0.06154727 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.09543973 - Q.z_6) / 0.001514041   # -6.2%  LHA < 0.2161 and z_6 < 0.09544
        + 0.05843301 * max(0.0, 0.03556091 - Q.e2) * max(0.0, 0.0753896 - Q.z_7) / 0.0004290247   # +5.8%  e2 < 0.03556 and z_7 < 0.07539
        + 0.05549132 * max(0.0, 0.002464291 - Q.lam1) * max(0.0, 0.01837778 - Q.centroid_offset) / 8.036566e-06   # +5.5%  lam1 < 0.002464 and centroid_offset < 0.01838
        + 0.04042643 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +4.0%  lam1 < 0.002464
        + 0.03338074 * max(0.0, 0.02054282 - Q.girth) * max(0.0, Q.sum_pt_top5 - 716.8828) / 0.2273844   # +3.3%  girth < 0.02054 and sum_pt_top5 > 716.9
        + 0.03306421 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +3.3%  lam1 < 0.00484
        + 0.03052798 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +3.1%  z_7 < 0.02807
        + 0.02783068 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +2.8%  LHA < 0.2161
        - 0.02755825 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -2.8%  LHA < 0.2161 and log_sum_pt < 6.804
        - 0.02691955 * max(0.0, 0.0262518 - Q.tau1) / 0.005366404   # -2.7%  tau1 < 0.02625
        - 0.02666746 * max(0.0, Q.log_sum_pt - 6.804164) * max(0.0, 5.334511e-05 - Q.e3) / 6.116454e-07   # -2.7%  log_sum_pt > 6.804 and e3 < 5.335e-05
        - 0.02343484 * max(0.0, Q.n_pt_above_50 - 6.0) / 0.2884353   # -2.3%  n_pt_above_50 > 6
        + 0.01931118 * max(0.0, 739.5 - Q.sum_pt) * max(0.0, 0.008675069 - Q.mean_eta2) / 0.3937849   # +1.9%  sum_pt < 739.5 and mean_eta2 < 0.008675
        - 0.01785818 * max(0.0, 0.03556091 - Q.e2) * max(0.0, Q.log_sum_pt - 6.896095) / 0.0001063848   # -1.8%  e2 < 0.03556 and log_sum_pt > 6.896
        + 0.01090661 * max(0.0, 7.12483e-05 - Q.lam1) / 3.380156e-06   # +1.1%  lam1 < 7.125e-05
        - 0.007566183 * max(0.0, 7.12483e-05 - Q.lam1) * max(0.0, 73.75 - Q.pt_5) / 8.367553e-05   # -0.8%  lam1 < 7.125e-05 and pt_5 < 73.75
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 26.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.67395 * (0.03846165
        - 0.1061464 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -10.6%  lam2 < 0.001131
        - 0.0959963 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # -9.6%  max_dr < 0.1773
        - 0.07872651 * max(0.0, Q.sj3_dr_max - 0.03464708) / 0.1367546   # -7.9%  sj3_dr_max > 0.03465
        + 0.06867875 * max(0.0, Q.LHA - 0.251526) / 0.03924325   # +6.9%  LHA > 0.2515
        - 0.06396546 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -6.4%  girth > 0.08724
        - 0.05932203 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -5.9%  pt_6 < 39.75
        + 0.04744693 * max(0.0, 39.75 - Q.pt_6) * max(0.0, 711.875 - Q.sum_pt_top3) / 684.1746   # +4.7%  pt_6 < 39.75 and sum_pt_top3 < 711.9
        + 0.04397814 * max(0.0, 0.04355037 - Q.z_6) / 0.003302426   # +4.4%  z_6 < 0.04355
        + 0.04359506 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # +4.4%  C2 < 0.03579
        + 0.03387155 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, Q.sj2_zsoft - 0.05966518) / 0.005829511   # +3.4%  sj2_dr > 0.1592 and sj2_zsoft > 0.05967
        + 0.03281649 * max(0.0, Q.sj3_dr_max - 0.1789613) * max(0.0, 1.59265 - Q.D2_b2) / 0.03741161   # +3.3%  sj3_dr_max > 0.179 and D2_b2 < 1.593
        - 0.03229459 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # -3.2%  girth > 0.1019
        + 0.03202893 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.001460719   # +3.2%  centroid_offset > 0.008092 and sj3_dr_min < 0.209
        + 0.02829809 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +2.8%  sj3_dr_max > 0.179
        + 0.02632282 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +2.6%  sj2_dr > 0.1683
        - 0.02416368 * max(0.0, 39.75 - Q.pt_6) * max(0.0, 0.7861046 - Q.z_top3_slots) / 0.3107099   # -2.4%  pt_6 < 39.75 and z_top3_slots < 0.7861
        - 0.02413471 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -2.4%  sj2_dr > 0.1592
        + 0.02393531 * max(0.0, Q.LHA - 0.3255822) / 0.01245304   # +2.4%  LHA > 0.3256
        - 0.02309507 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -2.3%  centroid_offset < 0.01838
        - 0.01936883 * max(0.0, 6.423044 - Q.log_sum_pt) / 0.05674989   # -1.9%  log_sum_pt < 6.423
        + 0.01855845 * max(0.0, 6.423044 - Q.log_sum_pt) * max(0.0, 38.53125 - Q.pt_7) / 0.3227003   # +1.9%  log_sum_pt < 6.423 and pt_7 < 38.53
        + 0.01465166 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.01426135 - Q.mean_phi2) / 9.496308e-05   # +1.5%  centroid_offset > 0.008092 and mean_phi2 < 0.01426
        + 0.01160966 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_dr_0p05_0p1 - 0.0) / 0.01198963   # +1.2%  centroid_offset < 0.01838 and n_dr_0p05_0p1 > 0
        - 0.007679962 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 5.0 - Q.n_dr_0_0p05) / 0.01797539   # -0.8%  centroid_offset > 0.02077 and n_dr_0_0p05 < 5
        + 0.007400339 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.N2 - 0.198361) / 0.0004989479   # +0.7%  centroid_offset > 0.008092 and N2 > 0.1984
        - 0.007362715 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.1115136 - Q.planar_flow) / 0.0002206557   # -0.7%  centroid_offset < 0.01838 and planar_flow < 0.1115
        + 0.00714065 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 0.002094451   # +0.7%  lam2 < 0.003408 and n_dr_0p1_0p2 > 1
        - 0.007103262 * max(0.0, 31.90625 - Q.pt_6) / 1.826854   # -0.7%  pt_6 < 31.91
        - 0.006191274 * max(0.0, Q.girth - 0.08723651) * max(0.0, 0.380911 - Q.D2_b2) / 0.001021187   # -0.6%  girth > 0.08724 and D2_b2 < 0.3809
        + 0.004116442 * max(0.0, Q.girth - 0.1484084) / 0.0008957407   # +0.4%  girth > 0.1484
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 35.63;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.62722 * (0.03880829
        + 0.07501456 * max(0.0, 0.08865369 - Q.tau1) / 0.03771145   # +7.5%  tau1 < 0.08865
        - 0.0716991 * max(0.0, Q.LHA - 0.3467135) / 0.008898884   # -7.2%  LHA > 0.3467
        + 0.06384623 * max(0.0, Q.girth - 0.0479157) * max(0.0, 2.357246 - Q.D2) / 0.03555018   # +6.4%  girth > 0.04792 and D2 < 2.357
        - 0.05541941 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # -5.5%  lam1 > 0.01643
        - 0.04974468 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -5.0%  e3 < 8.148e-05
        + 0.04660586 * max(0.0, Q.LHA - 0.1329373) / 0.1175814   # +4.7%  LHA > 0.1329
        + 0.04602927 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # +4.6%  girth > 0.1019
        + 0.04525514 * max(0.0, 0.1594012 - Q.sj3_dr_max) / 0.04422261   # +4.5%  sj3_dr_max < 0.1594
        - 0.04307947 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # -4.3%  lam1 < 0.00484
        - 0.0423919 * max(0.0, Q.girth - 0.0479157) / 0.02294968   # -4.2%  girth > 0.04792
        + 0.0421333 * max(0.0, 0.02685622 - Q.centroid_offset) / 0.01292674   # +4.2%  centroid_offset < 0.02686
        - 0.04212589 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 2.843757 - Q.D2) / 0.004089948   # -4.2%  lam1 > 0.00733 and D2 < 2.844
        + 0.03879126 * max(0.0, Q.lam1 - 0.01643375) * max(0.0, 3.885568 - Q.D2) / 0.002114372   # +3.9%  lam1 > 0.01643 and D2 < 3.886
        - 0.03610978 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # -3.6%  lam1 < 0.003377
        - 0.03591785 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # -3.6%  sj3_dr_max < 0.2134
        + 0.0320632 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # +3.2%  lam1 > 0.00733
        + 0.02497357 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # +2.5%  lam1 > 0.012
        - 0.02343343 * max(0.0, Q.lam1 - 0.002464291) * max(0.0, Q.D2 - 0.2568137) / 0.002599874   # -2.3%  lam1 > 0.002464 and D2 > 0.2568
        + 0.02139028 * max(0.0, Q.tau1 - 0.06345984) / 0.02203215   # +2.1%  tau1 > 0.06346
        - 0.02134481 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.lam1 - 0.008375572) / 0.0001274849   # -2.1%  planar_flow < 0.195 and lam1 > 0.008376
        - 0.02029052 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 43.5 - Q.pt_7) / 0.0006101229   # -2.0%  e3 < 8.148e-05 and pt_7 < 43.5
        - 0.01823121 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.001397205   # -1.8%  z_dr_0p1_0p2 < 0.1586 and centroid_offset < 0.02686
        - 0.0169824 * max(0.0, Q.girth - 0.08065885) * max(0.0, 2.843757 - Q.D2) / 0.01853389   # -1.7%  girth > 0.08066 and D2 < 2.844
        + 0.01542001 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) / 0.09532154   # +1.5%  z_dr_0p1_0p2 < 0.1586
        - 0.01526107 * max(0.0, 0.02685622 - Q.centroid_offset) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.01150938   # -1.5%  centroid_offset < 0.02686 and n_dr_0p2_0p4 < 1
        + 0.0144949 * max(0.0, 0.2346184 - Q.LHA) / 0.04003744   # +1.4%  LHA < 0.2346
        + 0.01170522 * max(0.0, Q.girth - 0.1245537) / 0.002632136   # +1.2%  girth > 0.1246
        + 0.009528972 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.46415) / 0.01159499   # +1.0%  planar_flow < 0.195 and log_sum_pt > 6.464
        - 0.007111157 * max(0.0, 0.0006570502 - Q.girth2_top5) / 0.0001395474   # -0.7%  girth2_top5 < 0.0006571
        + 0.006594788 * max(0.0, Q.lam1 - 0.002464291) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0003749324   # +0.7%  lam1 > 0.002464 and planar_flow < 0.195
        - 0.004399011 * max(0.0, Q.lam1 - 0.002464291) * max(0.0, 0.1879486 - Q.sj3_dr_max) / 1.34726e-05   # -0.4%  lam1 > 0.002464 and sj3_dr_max < 0.1879
        - 0.002611764 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # -0.3%  tau1 > 0.1136
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 21.91;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.91108 * (-0.06481372
        + 0.137648 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, 0.0153634 - Q.girth2_top3) / 0.0003596312   # +13.8%  sj3_dr_max < 0.107 and girth2_top3 < 0.01536
        - 0.1252006 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # -12.5%  sj3_dr_max < 0.107
        - 0.1104763 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, 0.0153634 - Q.girth2_top3) / 0.0005547224   # -11.0%  sj3_dr_max < 0.1426 and girth2_top3 < 0.01536
        - 0.06063441 * max(0.0, 0.1767241 - Q.LHA) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.00104409   # -6.1%  LHA < 0.1767 and z_dr_0p2_0p4 < 0.05644
        - 0.05236582 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -5.2%  centroid_offset > 0.00679
        + 0.05148551 * max(0.0, 0.0262518 - Q.tau1) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0001164763   # +5.1%  tau1 < 0.02625 and centroid_offset < 0.03117
        + 0.04709621 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.02355416 - Q.centroid_offset) / 4.507403e-05   # +4.7%  lam1 < 0.00733 and centroid_offset < 0.02355
        + 0.0454029 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.0001209282   # +4.5%  lam1 < 0.005433 and z_dr_0p2_0p4 < 0.05644
        - 0.0418247 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.01627885) / 7.113e-05   # -4.2%  sj3_dr_max < 0.107 and centroid_offset > 0.01628
        + 0.03614772 * max(0.0, 9.651826e-05 - Q.lam2) / 4.237938e-05   # +3.6%  lam2 < 9.652e-05
        + 0.03494615 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +3.5%  sj3_dr_max < 0.1426
        + 0.03473325 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 0.2752975   # +3.5%  sj3_dr_max < 0.1986 and n_dr_0p05_0p1 < 5
        - 0.02681697 * max(0.0, 0.03360421 - Q.girth) / 0.006496997   # -2.7%  girth < 0.0336
        - 0.02633714 * max(0.0, 0.251526 - Q.LHA) * max(0.0, Q.z_7 - 0.01685855) / 0.001243859   # -2.6%  LHA < 0.2515 and z_7 > 0.01686
        + 0.02207853 * max(0.0, 0.1767241 - Q.LHA) / 0.01861764   # +2.2%  LHA < 0.1767
        - 0.02176296 * max(0.0, 0.03360421 - Q.girth) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0001637288   # -2.2%  girth < 0.0336 and centroid_offset < 0.03117
        + 0.02133204 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 0.03117077 - Q.centroid_offset) / 4.498827e-05   # +2.1%  lam1 < 0.005433 and centroid_offset < 0.03117
        - 0.020008 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -2.0%  log_sum_pt > 6.701
        + 0.01984304 * max(0.0, 0.1594012 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.003343241) / 0.00037024   # +2.0%  sj3_dr_max < 0.1594 and centroid_offset > 0.003343
        + 0.01867428 * max(0.0, 0.03360421 - Q.girth) * max(0.0, Q.pt_7 - 15.55391) / 0.1111381   # +1.9%  girth < 0.0336 and pt_7 > 15.55
        - 0.01777605 * max(0.0, 0.019014 - Q.tau1) / 0.003025453   # -1.8%  tau1 < 0.01901
        + 0.01481983 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +1.5%  sj3_dr_max < 0.1986
        - 0.01258965 * max(0.0, Q.z_dr_0_0p05 - 0.9008535) / 0.03206403   # -1.3%  z_dr_0_0p05 > 0.9009
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 16.66;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.66177 * (-0.3754955
        + 0.1926153 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 0.001130645 - Q.lam2) / 2.659012e-06   # +19.3%  lam1 < 0.005954 and lam2 < 0.001131
        + 0.1219276 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, 0.00733008 - Q.lam1) / 4.507403e-05   # +12.2%  centroid_offset < 0.02355 and lam1 < 0.00733
        + 0.10935 * max(0.0, 0.03117077 - Q.centroid_offset) / 0.01649145   # +10.9%  centroid_offset < 0.03117
        + 0.09563275 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +9.6%  sum_pt < 988.4
        - 0.07543537 * max(0.0, 0.05464922 - Q.girth) / 0.01532978   # -7.5%  girth < 0.05465
        + 0.06222706 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +6.2%  lam2 < 0.0001948
        + 0.05558059 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # +5.6%  lam1 < 0.003377
        - 0.05474258 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -5.5%  sj3_dr_max < 0.1426
        + 0.04689741 * max(0.0, 813.4156 - Q.sum_pt) / 129.0758   # +4.7%  sum_pt < 813.4
        - 0.03681414 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 0.08543881 - Q.sj3_dr_min) / 0.0001709388   # -3.7%  lam1 < 0.005954 and sj3_dr_min < 0.08544
        - 0.02913247 * max(0.0, 1.960701e-05 - Q.e3) / 8.488392e-06   # -2.9%  e3 < 1.961e-05
        + 0.0251507 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +2.5%  sj3_dr_max < 0.2134
        - 0.02329324 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -2.3%  sj2_dr < 0.1295
        + 0.02072045 * max(0.0, Q.lam2 - 0.000537286) / 0.0003825703   # +2.1%  lam2 > 0.0005373
        + 0.0189736 * max(0.0, 0.1027585 - Q.max_dr) / 0.02411847   # +1.9%  max_dr < 0.1028
        + 0.01187398 * max(0.0, 0.02689598 - Q.girth) / 0.004330137   # +1.2%  girth < 0.0269
        + 0.01065179 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +1.1%  sj3_dr_min > 0.1278
        - 0.008980878 * max(0.0, Q.lam2 - 0.000537286) * max(0.0, 0.9979797 - Q.eccentricity) / 0.000146617   # -0.9%  lam2 > 0.0005373 and eccentricity < 0.998
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 13.22;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.22369 * (0.2837321
        + 0.2162162 * Q.lam1 / 0.00591636   # +21.6%  lam1
        - 0.1646232 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # -16.5%  girth < 0.1484
        + 0.08652399 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # +8.7%  sj2_dr > 0.1295
        - 0.07368664 * max(0.0, Q.LHA - 0.1967397) / 0.07109091   # -7.4%  LHA > 0.1967
        - 0.07011467 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -7.0%  sj2_dr > 0.1592
        - 0.06349203 * Q.e2 / 0.02863215   # -6.3%  e2
        - 0.06117928 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # -6.1%  lam1 < 0.003377
        - 0.04227473 * max(0.0, Q.lam1 - 0.00483998) / 0.002931468   # -4.2%  lam1 > 0.00484
        + 0.04121304 * max(0.0, Q.lam2 - 0.0001947983) / 0.0004461515   # +4.1%  lam2 > 0.0001948
        - 0.04100465 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.6947818 - Q.planar_flow) / 0.0166918   # -4.1%  sj2_dr > 0.1683 and planar_flow < 0.6948
        + 0.03188699 * max(0.0, Q.tau1 - 0.04369778) / 0.031989   # +3.2%  tau1 > 0.0437
        + 0.03146954 * max(0.0, 0.002915531 - Q.girth2_top3) / 0.001178811   # +3.1%  girth2_top3 < 0.002916
        - 0.02037475 * max(0.0, Q.LHA - 0.3127275) * max(0.0, 0.6947818 - Q.planar_flow) / 0.005887138   # -2.0%  LHA > 0.3127 and planar_flow < 0.6948
        - 0.0146572 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.eccentricity - 0.4829602) / 8.513077e-05   # -1.5%  lam2 > 0.0001948 and eccentricity > 0.483
        - 0.01238573 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.planar_flow - 0.06126205) / 0.0002819968   # -1.2%  lam2 > 0.0001948 and planar_flow > 0.06126
        - 0.01131859 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -1.1%  lam2 > 0.003408
        + 0.01126754 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +1.1%  sj3_dr_min > 0.1278
        + 0.006311287 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # +0.6%  e3 > 8.148e-05
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 19.13;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.13354 * (-0.02793602
        + 0.1951432 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +19.5%  lam1 < 0.012
        - 0.1119302 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # -11.2%  sj3_dr_max > 0.179
        + 0.09494066 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # +9.5%  sj2_dr > 0.1295
        - 0.07688881 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -7.7%  girth < 0.0717
        + 0.06608572 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 0.001130645 - Q.lam2) / 1.136697e-05   # +6.6%  lam1 < 0.01643 and lam2 < 0.001131
        - 0.06540464 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # -6.5%  lam1 < 0.01643
        + 0.06475995 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +6.5%  planar_flow < 0.2534
        - 0.06330317 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, 0.02685622 - Q.centroid_offset) / 2.007754e-05   # -6.3%  lam1 < 0.003377 and centroid_offset < 0.02686
        + 0.05513095 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +5.5%  centroid_offset < 0.01838
        + 0.03898788 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +3.9%  lam1 < 0.008376
        - 0.03842776 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.003652679   # -3.8%  planar_flow < 0.2534 and centroid_offset < 0.0499
        - 0.03792434 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -3.8%  sj2_dr > 0.1592
        - 0.03608385 * max(0.0, 45.75 - Q.pt_7) / 12.14632   # -3.6%  pt_7 < 45.75
        - 0.02147416 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.centroid_offset - 0.01627885) / 2.281746e-05   # -2.1%  lam1 < 0.012 and centroid_offset > 0.01628
        + 0.02060643 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # +2.1%  sj2_dr > 0.1779
        + 0.00940329 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # +0.9%  sj3_dr_max > 0.2623
        + 0.003504984 * max(0.0, Q.sj2_dr - 0.2414351) * max(0.0, 739.5 - Q.sum_pt) / 2.049171   # +0.4%  sj2_dr > 0.2414 and sum_pt < 739.5
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 1.102;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.102321 * (-1.601817
        + 0.3154869 * max(0.0, Q.girth - 0.1245537) / 0.002632136   # +31.5%  girth > 0.1246
        - 0.2375737 * max(0.0, Q.sd_rg - 0.324646) * max(0.0, 6.896095 - Q.log_sum_pt) / 0.001091861   # -23.8%  sd_rg > 0.3246 and log_sum_pt < 6.896
        + 0.1391284 * max(0.0, Q.sd_rg - 0.324646) / 0.001748965   # +13.9%  sd_rg > 0.3246
        + 0.1243131 * max(0.0, Q.sd_rg - 0.324646) * max(0.0, 6.701242 - Q.log_sum_pt) / 0.0007568975   # +12.4%  sd_rg > 0.3246 and log_sum_pt < 6.701
        - 0.1163064 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -11.6%  e2 > 0.06344
        - 0.06719156 * max(0.0, Q.girth - 0.1245537) * max(0.0, 62.25 - Q.pt_6) / 0.06207836   # -6.7%  girth > 0.1246 and pt_6 < 62.25
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 19.46;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.45793 * (0.0419033
        + 0.3540618 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # +35.4%  girth < 0.1484
        + 0.1524817 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +15.2%  lam1 < 0.01643
        - 0.1021647 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -10.2%  girth < 0.1484 and log_sum_pt < 6.804
        - 0.09692079 * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.09151624   # -9.7%  sj3_dr_min < 0.1278
        - 0.07990516 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.03585464 - Q.tau2) / 0.002456165   # -8.0%  girth < 0.1484 and tau2 < 0.03585
        - 0.03996804 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -4.0%  tau1 < 0.1136
        + 0.03536042 * max(0.0, Q.z_7 - 0.04624032) / 0.01204993   # +3.5%  z_7 > 0.04624
        - 0.03054695 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -3.1%  centroid_offset < 0.01838
        - 0.02621008 * max(0.0, Q.pt_7 - 31.85938) / 5.945918   # -2.6%  pt_7 > 31.86
        + 0.02330142 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 40.04062 - Q.pt_7) / 652.1541   # +2.3%  sum_pt_top5 > 687.4 and pt_7 < 40.04
        - 0.01755443 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 25.57812 - Q.pt_7) / 0.01870322   # -1.8%  lam1 < 0.01643 and pt_7 < 25.58
        - 0.01742688 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -1.7%  girth < 0.1484 and pt_7 < 38.53
        - 0.01537153 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.5%  sj3_dr_max > 0.2337
        - 0.0087261 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -0.9%  log_sum_pt > 6.67
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 41.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 41.29297 * (-0.1051723
        - 0.1510194 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # -15.1%  lam2 < 0.003408
        - 0.09369597 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -9.4%  lam1 < 0.006507
        + 0.08975623 * max(0.0, Q.girth - 0.02689598) / 0.03613842   # +9.0%  girth > 0.0269
        + 0.07534108 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +7.5%  tau1 < 0.09539
        - 0.06565422 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -6.6%  girth > 0.08724
        + 0.05646415 * max(0.0, 0.1416226 - Q.tau1) / 0.08190733   # +5.6%  tau1 < 0.1416
        + 0.04637972 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +4.6%  e2 < 0.04111
        + 0.04213589 * max(0.0, Q.sd_rg - 0.1572969) / 0.02905066   # +4.2%  sd_rg > 0.1573
        + 0.03384545 * max(0.0, 0.2330919 - Q.sd_rg) / 0.1256101   # +3.4%  sd_rg < 0.2331
        + 0.0328911 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.LHA - 0.1767241) / 0.0002170323   # +3.3%  lam2 < 0.003408 and LHA > 0.1767
        + 0.0327168 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +3.3%  lam1 < 0.012
        - 0.02773975 * max(0.0, Q.sd_rg - 0.1876504) / 0.01921774   # -2.8%  sd_rg > 0.1877
        - 0.02625111 * max(0.0, Q.sd_rg - 0.1449048) / 0.03437807   # -2.6%  sd_rg > 0.1449
        - 0.0261119 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -2.6%  planar_flow < 0.1115 and lam1 < 0.00733
        - 0.0233189 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -2.3%  centroid_offset > 0.0499
        + 0.02182059 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.sd_rg - 0.1572969) / 6.238901e-05   # +2.2%  lam2 < 0.003408 and sd_rg > 0.1573
        + 0.02180501 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.1644126   # +2.2%  z_dr_0p05_0p1 > 0.1639 and n_dr_0p2_0p4 < 1
        - 0.02171533 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -2.2%  e3 < 8.148e-05
        + 0.02029241 * max(0.0, Q.girth - 0.08065885) / 0.009330634   # +2.0%  girth > 0.08066
        + 0.01714876 * max(0.0, Q.girth - 0.06663269) / 0.01393206   # +1.7%  girth > 0.06663
        - 0.01399072 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -1.4%  LHA > 0.3033
        + 0.01193539 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # +1.2%  planar_flow < 0.1115 and lam1 < 0.01643
        + 0.009610181 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00595415 - Q.lam1) / 3.161575e-05   # +1.0%  planar_flow < 0.1115 and lam1 < 0.005954
        - 0.009084684 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) / 0.1950558   # -0.9%  z_dr_0p05_0p1 > 0.1639
        - 0.008203565 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 0.20552 - Q.z_dr_0p2_0p4) / 0.03609844   # -0.8%  z_dr_0p05_0p1 > 0.1639 and z_dr_0p2_0p4 < 0.2055
        - 0.007026998 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02685622) / 1.516228e-05   # -0.7%  lam1 < 0.01643 and centroid_offset > 0.02686
        + 0.005515013 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.2832687 - Q.sj2_zsoft) / 0.002541531   # +0.6%  planar_flow < 0.1115 and sj2_zsoft < 0.2833
        - 0.004763247 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 6.670067 - Q.log_sum_pt) / 0.03946963   # -0.5%  z_dr_0p05_0p1 > 0.1639 and log_sum_pt < 6.67
        - 0.003766478 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.sd_rg - 0.2037854) / 2.928331e-05   # -0.4%  lam2 < 0.003408 and sd_rg > 0.2038
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 27.54;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.54389 * (-0.06753234
        - 0.1133502 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -11.3%  girth < 0.0717
        + 0.1026287 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +10.3%  lam2 < 0.003408
        - 0.09432438 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -9.4%  lam1 < 0.00733
        + 0.07888906 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +7.9%  tau1 < 0.05357
        - 0.06590258 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -6.6%  lam1 < 0.006507
        - 0.04992551 * max(0.0, Q.tau1 - 0.1416226) / 0.003662393   # -5.0%  tau1 > 0.1416
        + 0.04907509 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +4.9%  lam1 < 0.005954
        + 0.0443606 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +4.4%  lam1 < 0.01643
        + 0.04376087 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # +4.4%  tau1 > 0.1028
        - 0.03676113 * max(0.0, 0.03360421 - Q.girth) / 0.006496997   # -3.7%  girth < 0.0336
        - 0.03665621 * max(0.0, Q.tau1 - 0.08865369) / 0.01243546   # -3.7%  tau1 > 0.08865
        + 0.0317648 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.2931906) / 0.001536651   # +3.2%  N2 < 0.2233 and LHA > 0.2932
        + 0.03038196 * max(0.0, 0.007639643 - Q.girth2_top2) / 0.004543329   # +3.0%  girth2_top2 < 0.00764
        + 0.02812842 * max(0.0, Q.eccentricity - 0.9458207) * max(0.0, Q.sum_pt_top5 - 367.5938) / 5.10997   # +2.8%  eccentricity > 0.9458 and sum_pt_top5 > 367.6
        + 0.02702999 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # +2.7%  girth2_top3 < 0.002152
        - 0.02463641 * max(0.0, Q.eccentricity - 0.9458207) / 0.02147681   # -2.5%  eccentricity > 0.9458
        - 0.02254289 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3855647) / 0.0002330351   # -2.3%  N2 < 0.2233 and LHA > 0.3856
        + 0.0202514 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +2.0%  tau1 > 0.1136
        - 0.01867641 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3255822) / 0.0007869707   # -1.9%  N2 < 0.2233 and LHA > 0.3256
        - 0.01858062 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 0.0002261574   # -1.9%  lam1 < 0.008376 and D2 < 0.8757
        - 0.01852595 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.lam1 - 0.008375572) / 0.0001169809   # -1.9%  N2 < 0.2233 and lam1 > 0.008376
        + 0.01206309 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # +1.2%  lam1 < 0.006507 and D2 < 0.8757
        - 0.01195242 * max(0.0, Q.tau1 - 0.09538712) / 0.01057268   # -1.2%  tau1 > 0.09539
        - 0.009941744 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1872617 - Q.sj2_dr) / 0.0007896743   # -1.0%  N2 < 0.2233 and sj2_dr < 0.1873
        - 0.007582336 * max(0.0, 0.002151568 - Q.girth2_top3) * max(0.0, 0.1950135 - Q.planar_flow) / 2.579988e-05   # -0.8%  girth2_top3 < 0.002152 and planar_flow < 0.195
        - 0.002307226 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.4235925) / 7.321856e-05   # -0.2%  N2 < 0.2233 and LHA > 0.4236
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.140925, 0.8069840336134454, 2.3144644957983194, 1.0396338235294118, 1.1743319327731092, 2.062102100840336, 1.0198518907563026, 1.9755355042016807, 0.29719726890756304, 3.211995798319328, 2.029491071428571, 2.026585294117647, 0.05906953781512605, 3.4893867647058823, 0.510160819327731, 0.42644915966386554]
T = [2.57494422761292, 1.44869623490021, 3.543875828847164, 2.6672524799763657, 3.009002865677521]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +39%, n9 +21%, n5 -15%, n1 +12%, n0 -7%, n6 +4% ...
            + 0.3862206 * h[2] / H_AVG[2]
            + 0.2143976 * h[9] / H_AVG[9]
            - 0.1501563 * h[5] / H_AVG[5]
            + 0.1224213 * h[1] / H_AVG[1]
            - 0.06923239 * h[0] / H_AVG[0]
            + 0.04331989 * h[6] / H_AVG[6]
            - 0.01425191 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -18%, n6 +9%, n4 -8%, n5 +7%, n15 +2% ...
            + 0.5629516 * h[9] / H_AVG[9]
            - 0.1751136 * h[10] / H_AVG[10]
            + 0.08799739 * h[6] / H_AVG[6]
            - 0.07599496 * h[4] / H_AVG[4]
            + 0.06672278 * h[5] / H_AVG[5]
            + 0.01839797 * h[15] / H_AVG[15]
            + 0.01282176 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +21%, n3 -15%, n7 +12%, n0 +11%, n14 -11%, n6 -9% ...
            + 0.2144459 * h[11] / H_AVG[11]
            - 0.1466803 * h[3] / H_AVG[3]
            + 0.1219423 * h[7] / H_AVG[7]
            + 0.1106678 * h[0] / H_AVG[0]
            - 0.1079667 * h[14] / H_AVG[14]
            - 0.08993084 * h[6] / H_AVG[6]
            - 0.0827297 * h[15] / H_AVG[15]
            + 0.06923141 * h[13] / H_AVG[13]
            - 0.02832347 * h[9] / H_AVG[9]
            - 0.02096555 * h[8] / H_AVG[8]
            - 0.007116009 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -22%, n6 -14%, n14 +7%, n13 +7%, n1 +4% ...
            + 0.3471858 * h[7] / H_AVG[7]
            - 0.2192496 * h[3] / H_AVG[3]
            - 0.1433852 * h[6] / H_AVG[6]
            + 0.07172561 * h[14] / H_AVG[14]
            + 0.07154397 * h[13] / H_AVG[13]
            + 0.03781907 * h[1] / H_AVG[1]
            - 0.03763231 * h[9] / H_AVG[9]
            + 0.0343967 * h[4] / H_AVG[4]
            - 0.02498177 * h[15] / H_AVG[15]
            + 0.01207998 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +25%, n5 -17%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4711074 * h[13] / H_AVG[13]
            + 0.2529274 * h[10] / H_AVG[10]
            - 0.1713277 * h[5] / H_AVG[5]
            + 0.0487841 * h[4] / H_AVG[4]
            + 0.02159423 * h[3] / H_AVG[3]
            + 0.01851925 * h[8] / H_AVG[8]
            - 0.009815467 * h[12] / H_AVG[12]
            + 0.005924538 * h[0] / H_AVG[0]
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
