"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  13.8%   (on for 89% of jets)
  neuron  9:  11.8%   (on for 71% of jets)
  neuron  7:  10.4%   (on for 62% of jets)
  neuron  3:   8.8%   (on for 25% of jets)
  neuron  5:   7.8%   (on for 72% of jets)
  neuron 10:   7.6%   (on for 68% of jets)
  neuron  2:   7.6%   (on for 92% of jets)
  neuron  6:   7.1%   (on for 30% of jets)
  neuron 11:   5.7%   (on for 72% of jets)
  neuron  0:   4.5%   (on for 44% of jets)
  neuron 14:   4.3%   (on for 25% of jets)
  neuron  1:   3.5%   (on for 57% of jets)
  neuron  4:   2.9%   (on for 52% of jets)
  neuron 15:   2.7%   (on for 28% of jets)
  neuron  8:   1.2%   (on for 32% of jets)
  neuron 12:   0.2%   (on for 5% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.3% (the network: 65.8%); same class as the network for 86.5% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
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
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
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
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        max_dr=max(dr[i] for i in real),
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
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 10.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.64073 * (0.2838152
        - 0.1481117 * max(0.0, 0.017 - Q.tau2) / 0.007959681   # -14.8%  tau2 < 0.017
        + 0.1312279 * max(0.0, Q.sj3_dr_max - 0.13) / 0.07052328   # +13.1%  sj3_dr_max > 0.13
        - 0.1300961 * max(0.0, Q.sj3_dr_max - 0.19) / 0.03856038   # -13.0%  sj3_dr_max > 0.19
        - 0.1139761 * max(0.0, Q.LHA - 0.31) / 0.01604218   # -11.4%  LHA > 0.31
        - 0.1015144 * max(0.0, 0.0041 - Q.lam1) / 0.001471644   # -10.2%  lam1 < 0.0041
        - 0.0862151 * max(0.0, Q.centroid_offset - 0.0061) / 0.01189872   # -8.6%  centroid_offset > 0.0061
        - 0.05756349 * max(0.0, 760.0 - Q.sum_pt) * max(0.0, 0.00052 - Q.e3) / 0.03689865   # -5.8%  sum_pt < 760 and e3 < 0.00052
        + 0.05379009 * max(0.0, 0.14 - Q.planar_flow) / 0.04691522   # +5.4%  planar_flow < 0.14
        + 0.03856041 * max(0.0, 0.045 - Q.planar_flow) * max(0.0, 0.061 - Q.tau21_b2) / 0.0003730099   # +3.9%  planar_flow < 0.045 and tau21_b2 < 0.061
        + 0.03363224 * max(0.0, 700.0 - Q.sum_pt) * max(0.0, Q.centroid_offset - 0.012) / 1.061933   # +3.4%  sum_pt < 700 and centroid_offset > 0.012
        - 0.03212328 * max(0.0, Q.log_sum_pt - 6.8) / 0.01319749   # -3.2%  log_sum_pt > 6.8
        + 0.02804548 * max(0.0, 6.6e-05 - Q.e3) * max(0.0, 0.087 - Q.sj3_z3) / 7.173661e-07   # +2.8%  e3 < 6.6e-05 and sj3_z3 < 0.087
        + 0.02144802 * max(0.0, 0.0042 - Q.lam1) * max(0.0, 730.0 - Q.sum_pt) / 0.06615146   # +2.1%  lam1 < 0.0042 and sum_pt < 730
        + 0.01246396 * max(0.0, Q.sum_pt_top5 - 800.0) * max(0.0, 0.14 - Q.tau21_b2) / 0.5242118   # +1.2%  sum_pt_top5 > 800 and tau21_b2 < 0.14
        - 0.005788095 * max(0.0, Q.log_sum_pt - 6.8) * max(0.0, 0.048 - Q.tau21_b2) / 0.0001296622   # -0.6%  log_sum_pt > 6.8 and tau21_b2 < 0.048
        - 0.005443581 * max(0.0, 0.0042 - Q.lam1) * max(0.0, 0.78 - Q.D2) / 8.204485e-06   # -0.5%  lam1 < 0.0042 and D2 < 0.78
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 8.749;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.749179 * (0.1588721
        - 0.2695603 * max(0.0, 0.0079 - Q.lam1) * max(0.0, 0.0015 - Q.lam2) / 5.536224e-06   # -27.0%  lam1 < 0.0079 and lam2 < 0.0015
        + 0.1509557 * max(0.0, Q.log_sum_pt - 6.4) / 0.1930903   # +15.1%  log_sum_pt > 6.4
        - 0.1192147 * max(0.0, 0.065 - Q.z_7) / 0.01650365   # -11.9%  z_7 < 0.065
        + 0.09857236 * max(0.0, 0.1 - Q.tau1) / 0.0461191   # +9.9%  tau1 < 0.1
        + 0.08727834 * max(0.0, Q.log_sum_pt - 6.6) / 0.0734244   # +8.7%  log_sum_pt > 6.6
        - 0.06925617 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0075 - Q.girth2_top3) / 0.0004359242   # -6.9%  log_sum_pt > 6.6 and girth2_top3 < 0.0075
        + 0.06615678 * max(0.0, Q.pt_7 - 34.0) / 4.744406   # +6.6%  pt_7 > 34
        - 0.05028423 * max(0.0, Q.sj3_dr_min - 0.028) / 0.02650276   # -5.0%  sj3_dr_min > 0.028
        + 0.04027035 * max(0.0, Q.log_sum_pt - 6.3) * max(0.0, 1.3e-05 - Q.e3) / 1.779457e-06   # +4.0%  log_sum_pt > 6.3 and e3 < 1.3e-05
        - 0.02872105 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 0.015 - Q.tau2) / 0.03188904   # -2.9%  pt_7 > 34 and tau2 < 0.015
        - 0.01114463 * max(0.0, 0.0088 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.1) / 0.001390961   # -1.1%  lam1 < 0.0088 and n_pt_above_50 > 6.1
        - 0.008585416 * max(0.0, Q.sum_pt - 990.0) * max(0.0, 0.0074 - Q.girth2_top2) / 0.02900206   # -0.9%  sum_pt > 990 and girth2_top2 < 0.0074
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 10.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.23144 * (0.1407427
        - 0.2699686 * max(0.0, 55.0 - Q.pt_7) / 20.61318   # -27.0%  pt_7 < 55
        + 0.1974665 * max(0.0, 0.1 - Q.girth) / 0.04698525   # +19.7%  girth < 0.1
        + 0.1701941 * max(0.0, 0.081 - Q.z_7) / 0.02956417   # +17.0%  z_7 < 0.081
        + 0.1224392 * max(0.0, 6.6 - Q.log_sum_pt) / 0.1304926   # +12.2%  log_sum_pt < 6.6
        - 0.1014196 * max(0.0, 0.12 - Q.M2) / 0.06140051   # -10.1%  M2 < 0.12
        + 0.02930614 * max(0.0, 750.0 - Q.sum_pt_top5) * max(0.0, Q.D2_b2 - 0.18) / 113.1486   # +2.9%  sum_pt_top5 < 750 and D2_b2 > 0.18
        - 0.02311247 * max(0.0, Q.log_sum_pt - 6.1) * max(0.0, 8.9e-05 - Q.mean_phi2) / 6.698972e-06   # -2.3%  log_sum_pt > 6.1 and mean_phi2 < 8.9e-05
        - 0.02290369 * max(0.0, 0.043 - Q.tau1) * max(0.0, 0.89 - Q.planar_flow) / 0.005242453   # -2.3%  tau1 < 0.043 and planar_flow < 0.89
        + 0.02073478 * max(0.0, 8.7e-05 - Q.mean_phi2) / 9.959933e-06   # +2.1%  mean_phi2 < 8.7e-05
        - 0.0205152 * max(0.0, 6.6 - Q.log_sum_pt) * max(0.0, Q.D2_b2 - -0.04) / 0.107641   # -2.1%  log_sum_pt < 6.6 and D2_b2 > -0.04
        - 0.01415896 * max(0.0, 0.0078 - Q.girth) / 0.0002442942   # -1.4%  girth < 0.0078
        - 0.005782417 * max(0.0, Q.z_7 - 0.079) / 0.0009405791   # -0.6%  z_7 > 0.079
        + 0.001998366 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.1 - Q.absphi_0) / 0.0003447918   # +0.2%  log_sum_pt > 6.9 and absphi_0 < 0.1
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 7.411;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.410939 * (-0.05640311
        - 0.2730925 * max(0.0, 0.0032 - Q.lam2) / 0.00283853   # -27.3%  lam2 < 0.0032
        + 0.1544374 * max(0.0, Q.girth - 0.07) / 0.01263274   # +15.4%  girth > 0.07
        + 0.1499128 * max(0.0, Q.sj2_dr - 0.16) / 0.03884597   # +15.0%  sj2_dr > 0.16
        + 0.1027818 * max(0.0, Q.sj2_dr - 0.16) * max(0.0, 0.096 - Q.C2) / 0.001627585   # +10.3%  sj2_dr > 0.16 and C2 < 0.096
        - 0.0945932 * max(0.0, Q.girth - 0.099) / 0.005841871   # -9.5%  girth > 0.099
        + 0.0813497 * max(0.0, Q.sj3_dr_max - 0.22) / 0.02870846   # +8.1%  sj3_dr_max > 0.22
        - 0.05910513 * max(0.0, Q.sj2_dr - 0.17) * max(0.0, Q.log_sum_pt - 6.1) / 0.01161869   # -5.9%  sj2_dr > 0.17 and log_sum_pt > 6.1
        - 0.03397058 * max(0.0, Q.sj2_dr - 0.16) * max(0.0, Q.n_dr_0_0p05 - 1.6) / 0.05201528   # -3.4%  sj2_dr > 0.16 and n_dr_0_0p05 > 1.6
        - 0.03106935 * max(0.0, Q.sd_rg - 0.27) / 0.006027567   # -3.1%  sd_rg > 0.27
        - 0.01968746 * max(0.0, Q.max_dr - 0.25) * max(0.0, 0.66 - Q.z_dr_0_0p05) / 0.001751532   # -2.0%  max_dr > 0.25 and z_dr_0_0p05 < 0.66
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 31.36;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.3596 * (-0.03026186
        + 0.3531461 * max(0.0, 0.21 - Q.sj3_dr_min) / 0.1675419   # +35.3%  sj3_dr_min < 0.21
        - 0.2084709 * max(0.0, 0.2 - Q.sj3_dr_min) * max(0.0, 0.0034 - Q.lam2) / 0.0005147687   # -20.8%  sj3_dr_min < 0.2 and lam2 < 0.0034
        + 0.1414673 * max(0.0, 0.0002 - Q.e3) / 0.0001613221   # +14.1%  e3 < 0.0002
        - 0.1261191 * max(0.0, 0.0087 - Q.lam1) / 0.00456176   # -12.6%  lam1 < 0.0087
        - 0.09642815 * max(0.0, 0.00074 - Q.lam2) / 0.0005662825   # -9.6%  lam2 < 0.00074
        - 0.03501845 * max(0.0, 840.0 - Q.sum_pt) / 148.4006   # -3.5%  sum_pt < 840
        + 0.01861676 * max(0.0, 0.075 - Q.dr_0) * max(0.0, Q.centroid_offset - 0.0011) / 0.000333608   # +1.9%  dr_0 < 0.075 and centroid_offset > 0.0011
        - 0.01306267 * max(0.0, Q.sj2_dr - 0.2) * max(0.0, Q.lam2 - 0.0012) / 2.947052e-05   # -1.3%  sj2_dr > 0.2 and lam2 > 0.0012
        - 0.007670584 * max(0.0, Q.sj3_dr_max - 0.23) * max(0.0, Q.log_sum_pt - 6.3) / 0.003735193   # -0.8%  sj3_dr_max > 0.23 and log_sum_pt > 6.3
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 9.917;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.916731 * (0.06907518
        + 0.1791509 * max(0.0, 0.00018 - Q.e3) / 0.0001432735   # +17.9%  e3 < 0.00018
        - 0.1265378 * max(0.0, 0.064 - Q.dr_0) / 0.02509683   # -12.7%  dr_0 < 0.064
        - 0.1000126 * max(0.0, 0.029 - Q.centroid_offset) / 0.01467157   # -10.0%  centroid_offset < 0.029
        + 0.09329114 * max(0.0, 0.0024 - Q.lam1) * max(0.0, 0.019 - Q.centroid_offset) / 8.187107e-06   # +9.3%  lam1 < 0.0024 and centroid_offset < 0.019
        - 0.07423797 * max(0.0, 0.22 - Q.LHA) * max(0.0, 0.09 - Q.z_6) / 0.001404958   # -7.4%  LHA < 0.22 and z_6 < 0.09
        + 0.07417844 * max(0.0, 0.00085 - Q.lam1) / 0.0001816315   # +7.4%  lam1 < 0.00085
        + 0.0665521 * max(0.0, 0.038 - Q.e2) * max(0.0, 0.073 - Q.z_7) / 0.0004429391   # +6.7%  e2 < 0.038 and z_7 < 0.073
        + 0.06254431 * max(0.0, 0.0029 - Q.lam1) / 0.0009271077   # +6.3%  lam1 < 0.0029
        - 0.04103918 * max(0.0, 740.0 - Q.sum_pt) / 82.88687   # -4.1%  sum_pt < 740
        + 0.03708187 * max(0.0, 0.028 - Q.z_7) / 0.001299403   # +3.7%  z_7 < 0.028
        + 0.03656314 * max(0.0, 0.019 - Q.girth) * max(0.0, Q.sum_pt_top5 - 700.0) / 0.223819   # +3.7%  girth < 0.019 and sum_pt_top5 > 700
        - 0.03212464 * max(0.0, 0.21 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt) / 0.003599677   # -3.2%  LHA < 0.21 and log_sum_pt < 6.8
        - 0.02827044 * max(0.0, Q.log_sum_pt - 6.8) * max(0.0, 5.7e-05 - Q.e3) / 6.888216e-07   # -2.8%  log_sum_pt > 6.8 and e3 < 5.7e-05
        - 0.02739875 * max(0.0, Q.n_pt_above_50 - 6.0) / 0.2884353   # -2.7%  n_pt_above_50 > 6
        - 0.02101669 * max(0.0, 0.036 - Q.e2) * max(0.0, Q.log_sum_pt - 6.9) / 0.0001031767   # -2.1%  e2 < 0.036 and log_sum_pt > 6.9
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 18.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.29178 * (-0.07599043
        - 0.1584144 * max(0.0, 0.0011 - Q.lam2) / 0.0008861408   # -15.8%  lam2 < 0.0011
        - 0.1073531 * max(0.0, 0.18 - Q.max_dr) / 0.07246046   # -10.7%  max_dr < 0.18
        + 0.09997316 * max(0.0, Q.LHA - 0.25) / 0.04001503   # +10.0%  LHA > 0.25
        - 0.09075605 * max(0.0, Q.girth - 0.092) / 0.006975165   # -9.1%  girth > 0.092
        - 0.07892805 * max(0.0, 39.0 - Q.pt_6) / 4.348597   # -7.9%  pt_6 < 39
        + 0.0659337 * max(0.0, 0.036 - Q.C2) / 0.01520863   # +6.6%  C2 < 0.036
        + 0.05787947 * max(0.0, 39.0 - Q.pt_6) * max(0.0, 720.0 - Q.sum_pt_top3) / 630.1895   # +5.8%  pt_6 < 39 and sum_pt_top3 < 720
        + 0.05455161 * max(0.0, Q.centroid_offset - 0.0067) * max(0.0, 0.22 - Q.sj3_dr_min) / 0.001720424   # +5.5%  centroid_offset > 0.0067 and sj3_dr_min < 0.22
        + 0.05134589 * max(0.0, 0.044 - Q.z_6) / 0.0034153   # +5.1%  z_6 < 0.044
        + 0.04893924 * max(0.0, Q.sj3_dr_max - 0.18) * max(0.0, 1.8 - Q.D2_b2) / 0.04409781   # +4.9%  sj3_dr_max > 0.18 and D2_b2 < 1.8
        + 0.04112105 * max(0.0, Q.sj2_dr - 0.16) * max(0.0, Q.sj2_zsoft - 0.052) / 0.006017417   # +4.1%  sj2_dr > 0.16 and sj2_zsoft > 0.052
        + 0.03118854 * max(0.0, Q.centroid_offset - 0.0056) * max(0.0, 0.017 - Q.mean_phi2) / 0.0001437012   # +3.1%  centroid_offset > 0.0056 and mean_phi2 < 0.017
        - 0.03065842 * max(0.0, 40.0 - Q.pt_6) * max(0.0, 0.79 - Q.z_top3_slots) / 0.3358066   # -3.1%  pt_6 < 40 and z_top3_slots < 0.79
        + 0.02640976 * max(0.0, 6.4 - Q.log_sum_pt) * max(0.0, 39.0 - Q.pt_7) / 0.3116655   # +2.6%  log_sum_pt < 6.4 and pt_7 < 39
        - 0.02615997 * max(0.0, 6.4 - Q.log_sum_pt) / 0.05015852   # -2.6%  log_sum_pt < 6.4
        - 0.01708782 * max(0.0, Q.centroid_offset - 0.018) * max(0.0, 5.8 - Q.n_dr_0_0p05) / 0.02441926   # -1.7%  centroid_offset > 0.018 and n_dr_0_0p05 < 5.8
        + 0.01329982 * max(0.0, Q.centroid_offset - 0.01) * max(0.0, Q.N2 - 0.19) / 0.0004846163   # +1.3%  centroid_offset > 0.01 and N2 > 0.19
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 20.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.01318 * (0.1159236
        + 0.1166338 * max(0.0, 0.085 - Q.tau1) / 0.03520684   # +11.7%  tau1 < 0.085
        - 0.1144704 * max(0.0, Q.LHA - 0.34) / 0.009917393   # -11.4%  LHA > 0.34
        - 0.09622624 * max(0.0, 0.0049 - Q.lam1) / 0.001888033   # -9.6%  lam1 < 0.0049
        + 0.08723832 * max(0.0, 0.029 - Q.centroid_offset) / 0.01467157   # +8.7%  centroid_offset < 0.029
        + 0.07367952 * max(0.0, Q.girth - 0.047) * max(0.0, 2.2 - Q.D2) / 0.03269539   # +7.4%  girth > 0.047 and D2 < 2.2
        - 0.06907053 * max(0.0, 0.0032 - Q.lam1) / 0.001055207   # -6.9%  lam1 < 0.0032
        - 0.06665116 * max(0.0, 7.6e-05 - Q.e3) / 5.170163e-05   # -6.7%  e3 < 7.6e-05
        + 0.05998479 * max(0.0, Q.girth - 0.1) / 0.005689511   # +6.0%  girth > 0.1
        + 0.04156504 * max(0.0, 0.15 - Q.sj3_dr_max) / 0.04018593   # +4.2%  sj3_dr_max < 0.15
        - 0.03959422 * max(0.0, 0.16 - Q.z_dr_0p1_0p2) * max(0.0, 0.028 - Q.centroid_offset) / 0.001506476   # -4.0%  z_dr_0p1_0p2 < 0.16 and centroid_offset < 0.028
        - 0.03767826 * max(0.0, 0.17 - Q.planar_flow) * max(0.0, Q.lam1 - 0.0081) / 0.0001077231   # -3.8%  planar_flow < 0.17 and lam1 > 0.0081
        + 0.03687124 * max(0.0, 0.16 - Q.z_dr_0p1_0p2) / 0.09633301   # +3.7%  z_dr_0p1_0p2 < 0.16
        - 0.03644827 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 0.31) / 0.002391626   # -3.6%  lam1 > 0.0025 and D2 > 0.31
        - 0.0343514 * max(0.0, 8.2e-05 - Q.e3) * max(0.0, 43.0 - Q.pt_7) / 0.000592656   # -3.4%  e3 < 8.2e-05 and pt_7 < 43
        + 0.02939865 * max(0.0, Q.tau1 - 0.06) / 0.02362894   # +2.9%  tau1 > 0.06
        - 0.02840426 * max(0.0, 0.028 - Q.centroid_offset) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.01230432   # -2.8%  centroid_offset < 0.028 and n_dr_0p2_0p4 < 1
        + 0.02001432 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.4) / 0.01371748   # +2.0%  planar_flow < 0.18 and log_sum_pt > 6.4
        - 0.01171955 * max(0.0, Q.lam1 - 0.0022) * max(0.0, 0.19 - Q.sj3_dr_max) / 1.663444e-05   # -1.2%  lam1 > 0.0022 and sj3_dr_max < 0.19
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 7.897;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.897112 * (-0.1544869
        - 0.1955581 * max(0.0, Q.centroid_offset - 0.006) / 0.01197166   # -19.6%  centroid_offset > 0.006
        + 0.1476111 * max(0.0, 0.0057 - Q.lam1) * max(0.0, 0.053 - Q.z_dr_0p2_0p4) / 0.0001215538   # +14.8%  lam1 < 0.0057 and z_dr_0p2_0p4 < 0.053
        + 0.1225529 * max(0.0, 0.0072 - Q.lam1) * max(0.0, 0.024 - Q.centroid_offset) / 4.522496e-05   # +12.3%  lam1 < 0.0072 and centroid_offset < 0.024
        - 0.1079493 * max(0.0, 0.035 - Q.girth) / 0.006987602   # -10.8%  girth < 0.035
        - 0.09973424 * max(0.0, 0.14 - Q.sj3_dr_max) * max(0.0, 0.016 - Q.girth2_top3) / 0.0005625803   # -10.0%  sj3_dr_max < 0.14 and girth2_top3 < 0.016
        - 0.09913838 * max(0.0, 0.11 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.016) / 7.601038e-05   # -9.9%  sj3_dr_max < 0.11 and centroid_offset > 0.016
        + 0.09689394 * max(0.0, 9.6e-05 - Q.lam2) / 4.204299e-05   # +9.7%  lam2 < 9.6e-05
        + 0.08984368 * max(0.0, 0.2 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.0011) / 0.0007024808   # +9.0%  sj3_dr_max < 0.2 and centroid_offset > 0.0011
        - 0.04071845 * max(0.0, Q.log_sum_pt - 6.7) / 0.03560998   # -4.1%  log_sum_pt > 6.7
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 12.75;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.74754 * (-0.458912
        + 0.228982 * max(0.0, 0.0057 - Q.lam1) * max(0.0, 0.0013 - Q.lam2) / 2.890057e-06   # +22.9%  lam1 < 0.0057 and lam2 < 0.0013
        + 0.1972543 * max(0.0, 0.024 - Q.centroid_offset) * max(0.0, 0.0072 - Q.lam1) / 4.522496e-05   # +19.7%  centroid_offset < 0.024 and lam1 < 0.0072
        + 0.1939783 * max(0.0, 930.0 - Q.sum_pt) / 222.77   # +19.4%  sum_pt < 930
        + 0.1320137 * max(0.0, 0.034 - Q.centroid_offset) / 0.01892969   # +13.2%  centroid_offset < 0.034
        - 0.1129357 * max(0.0, 0.067 - Q.girth) / 0.02208057   # -11.3%  girth < 0.067
        + 0.06450336 * max(0.0, 0.00019 - Q.lam2) / 0.0001084775   # +6.5%  lam2 < 0.00019
        - 0.04637881 * max(0.0, 0.13 - Q.sj3_dr_max) / 0.03230688   # -4.6%  sj3_dr_max < 0.13
        + 0.02395384 * max(0.0, Q.sj3_dr_min - 0.12) / 0.009007451   # +2.4%  sj3_dr_min > 0.12
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 7.796;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.796335 * (0.9106843
        - 0.546861 * max(0.0, 0.017 - Q.lam1) / 0.01171294   # -54.7%  lam1 < 0.017
        - 0.1356581 * max(0.0, 0.0036 - Q.lam1) / 0.001234115   # -13.6%  lam1 < 0.0036
        + 0.07850853 * max(0.0, 0.0029 - Q.girth2_top3) / 0.001170323   # +7.9%  girth2_top3 < 0.0029
        - 0.07291845 * max(0.0, Q.sj2_dr - 0.18) * max(0.0, 0.74 - Q.planar_flow) / 0.01532336   # -7.3%  sj2_dr > 0.18 and planar_flow < 0.74
        - 0.0444834 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.69 - Q.planar_flow) / 0.00615999   # -4.4%  LHA > 0.31 and planar_flow < 0.69
        + 0.03696311 * max(0.0, Q.lam2 - 0.00028) / 0.0004269286   # +3.7%  lam2 > 0.00028
        + 0.02704049 * max(0.0, Q.lam1 - 0.017) / 0.0006293038   # +2.7%  lam1 > 0.017
        + 0.02102726 * max(0.0, Q.sj3_dr_min - 0.12) / 0.009007451   # +2.1%  sj3_dr_min > 0.12
        - 0.01873986 * max(0.0, Q.lam2 - 0.0034) / 0.0001574377   # -1.9%  lam2 > 0.0034
        - 0.01779978 * max(0.0, Q.lam2 - 0.00025) * max(0.0, Q.eccentricity - 0.47) / 8.513685e-05   # -1.8%  lam2 > 0.00025 and eccentricity > 0.47
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 12.6;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.60046 * (-0.07920343
        + 0.330819 * max(0.0, 0.012 - Q.lam1) / 0.00731311   # +33.1%  lam1 < 0.012
        - 0.1698866 * max(0.0, Q.sj3_dr_max - 0.17) / 0.04746452   # -17.0%  sj3_dr_max > 0.17
        + 0.1220446 * max(0.0, Q.sj2_dr - 0.12) / 0.06175976   # +12.2%  sj2_dr > 0.12
        - 0.08558404 * max(0.0, 0.071 - Q.girth) / 0.02456489   # -8.6%  girth < 0.071
        - 0.07940969 * max(0.0, 0.0035 - Q.lam1) * max(0.0, 0.027 - Q.centroid_offset) / 2.110968e-05   # -7.9%  lam1 < 0.0035 and centroid_offset < 0.027
        + 0.06667286 * max(0.0, 0.31 - Q.planar_flow) / 0.1448464   # +6.7%  planar_flow < 0.31
        + 0.06504159 * max(0.0, 0.019 - Q.centroid_offset) / 0.007126558   # +6.5%  centroid_offset < 0.019
        - 0.0566941 * max(0.0, 46.0 - Q.pt_7) / 12.35938   # -5.7%  pt_7 < 46
        - 0.02384761 * max(0.0, 0.01 - Q.lam1) * max(0.0, Q.centroid_offset - 0.016) / 1.697689e-05   # -2.4%  lam1 < 0.01 and centroid_offset > 0.016
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.6459;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.6459479 * (-3.467772
        + 0.4702137 * max(0.0, Q.girth - 0.12) / 0.003099322   # +47.0%  girth > 0.12
        + 0.2092076 * max(0.0, Q.sd_rg - 0.31) / 0.002578954   # +20.9%  sd_rg > 0.31
        - 0.1654605 * max(0.0, Q.e2 - 0.065) / 0.00161205   # -16.5%  e2 > 0.065
        - 0.1551182 * max(0.0, Q.sd_rg - 0.31) * max(0.0, 6.9 - Q.log_sum_pt) / 0.001603173   # -15.5%  sd_rg > 0.31 and log_sum_pt < 6.9
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 17.38;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.38006 * (0.03377433
        + 0.3476829 * max(0.0, 0.15 - Q.girth) / 0.09211511   # +34.8%  girth < 0.15
        + 0.159245 * max(0.0, 0.016 - Q.lam1) / 0.01081128   # +15.9%  lam1 < 0.016
        - 0.1102618 * max(0.0, 0.13 - Q.sj3_dr_min) / 0.09348085   # -11.0%  sj3_dr_min < 0.13
        - 0.0905945 * max(0.0, 0.15 - Q.girth) * max(0.0, 0.036 - Q.tau2) / 0.002507226   # -9.1%  girth < 0.15 and tau2 < 0.036
        - 0.07954487 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt) / 0.01355387   # -8.0%  girth < 0.15 and log_sum_pt < 6.7
        + 0.04524928 * max(0.0, Q.z_7 - 0.045) / 0.01280839   # +4.5%  z_7 > 0.045
        - 0.04181269 * max(0.0, 0.11 - Q.tau1) / 0.05423188   # -4.2%  tau1 < 0.11
        - 0.03344389 * max(0.0, 0.018 - Q.centroid_offset) / 0.006472794   # -3.3%  centroid_offset < 0.018
        + 0.02611327 * max(0.0, Q.sum_pt_top5 - 690.0) * max(0.0, 39.0 - Q.pt_7) / 607.5639   # +2.6%  sum_pt_top5 > 690 and pt_7 < 39
        - 0.02609391 * max(0.0, 0.016 - Q.lam1) * max(0.0, 26.0 - Q.pt_7) / 0.01921668   # -2.6%  lam1 < 0.016 and pt_7 < 26
        - 0.02182966 * max(0.0, Q.pt_7 - 35.0) / 4.243858   # -2.2%  pt_7 > 35
        - 0.01812819 * max(0.0, Q.sj3_dr_max - 0.23) / 0.02603877   # -1.8%  sj3_dr_max > 0.23
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 34.65;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.64849 * (-0.1238149
        - 0.1599571 * max(0.0, 0.0034 - Q.lam2) / 0.003028564   # -16.0%  lam2 < 0.0034
        + 0.1342756 * max(0.0, Q.girth - 0.027) / 0.03606549   # +13.4%  girth > 0.027
        - 0.095771 * max(0.0, 0.0065 - Q.lam1) / 0.002885496   # -9.6%  lam1 < 0.0065
        + 0.08944324 * max(0.0, 0.096 - Q.tau1) / 0.04304269   # +8.9%  tau1 < 0.096
        - 0.07292016 * max(0.0, Q.girth - 0.088) / 0.007702969   # -7.3%  girth > 0.088
        + 0.06566889 * max(0.0, 0.041 - Q.e2) / 0.01750252   # +6.6%  e2 < 0.041
        + 0.05107896 * max(0.0, 0.14 - Q.tau1) / 0.08044585   # +5.1%  tau1 < 0.14
        + 0.04745199 * max(0.0, 0.24 - Q.sd_rg) / 0.1315312   # +4.7%  sd_rg < 0.24
        - 0.03912457 * max(0.0, Q.sd_rg - 0.19) / 0.01864659   # -3.9%  sd_rg > 0.19
        - 0.03046036 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.0073 - Q.lam1) / 5.704895e-05   # -3.0%  planar_flow < 0.11 and lam1 < 0.0073
        - 0.0277602 * max(0.0, Q.centroid_offset - 0.05) / 0.0008015409   # -2.8%  centroid_offset > 0.05
        - 0.02676837 * max(0.0, 7.9e-05 - Q.e3) / 5.42388e-05   # -2.7%  e3 < 7.9e-05
        + 0.02470824 * max(0.0, 0.0034 - Q.lam2) * max(0.0, Q.LHA - 0.18) / 0.0002098292   # +2.5%  lam2 < 0.0034 and LHA > 0.18
        + 0.02449542 * max(0.0, 0.0034 - Q.lam2) * max(0.0, Q.sd_rg - 0.16) / 5.935171e-05   # +2.4%  lam2 < 0.0034 and sd_rg > 0.16
        + 0.02362597 * max(0.0, Q.z_dr_0p05_0p1 - 0.16) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.1657094   # +2.4%  z_dr_0p05_0p1 > 0.16 and n_dr_0p2_0p4 < 1
        + 0.02269341 * max(0.0, Q.sd_rg - 0.16) / 0.02798193   # +2.3%  sd_rg > 0.16
        - 0.01773157 * max(0.0, Q.z_dr_0p05_0p1 - 0.17) / 0.1925931   # -1.8%  z_dr_0p05_0p1 > 0.17
        + 0.01447786 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.016 - Q.lam1) / 0.0003003809   # +1.4%  planar_flow < 0.11 and lam1 < 0.016
        + 0.01132665 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.0059 - Q.lam1) / 2.995813e-05   # +1.1%  planar_flow < 0.11 and lam1 < 0.0059
        - 0.008371103 * max(0.0, 0.016 - Q.lam1) * max(0.0, Q.centroid_offset - 0.027) / 1.414859e-05   # -0.8%  lam1 < 0.016 and centroid_offset > 0.027
        + 0.005951491 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.28 - Q.sj2_zsoft) / 0.002426002   # +0.6%  planar_flow < 0.11 and sj2_zsoft < 0.28
        - 0.005937818 * max(0.0, Q.z_dr_0p05_0p1 - 0.16) * max(0.0, 6.7 - Q.log_sum_pt) / 0.04462829   # -0.6%  z_dr_0p05_0p1 > 0.16 and log_sum_pt < 6.7
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 16.21;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.20619 * (-0.1116857
        + 0.1991112 * max(0.0, 0.0033 - Q.lam2) / 0.002933486   # +19.9%  lam2 < 0.0033
        - 0.1499759 * max(0.0, 0.0071 - Q.lam1) / 0.003315878   # -15.0%  lam1 < 0.0071
        - 0.1468783 * max(0.0, 0.071 - Q.girth) / 0.02456489   # -14.7%  girth < 0.071
        + 0.1214031 * max(0.0, 0.053 - Q.tau1) / 0.01667358   # +12.1%  tau1 < 0.053
        - 0.06364048 * max(0.0, 0.033 - Q.girth) / 0.006288841   # -6.4%  girth < 0.033
        + 0.05914267 * max(0.0, 0.0081 - Q.girth2_top2) / 0.004915269   # +5.9%  girth2_top2 < 0.0081
        - 0.04549798 * max(0.0, Q.eccentricity - 0.94) / 0.02474326   # -4.5%  eccentricity > 0.94
        + 0.04367371 * max(0.0, Q.eccentricity - 0.94) * max(0.0, Q.sum_pt_top5 - 370.0) / 5.801512   # +4.4%  eccentricity > 0.94 and sum_pt_top5 > 370
        - 0.04189773 * max(0.0, 0.22 - Q.N2) * max(0.0, Q.lam1 - 0.0081) / 0.0001170694   # -4.2%  N2 < 0.22 and lam1 > 0.0081
        + 0.03661062 * max(0.0, 0.21 - Q.N2) * max(0.0, Q.LHA - 0.28) / 0.001755381   # +3.7%  N2 < 0.21 and LHA > 0.28
        - 0.0279523 * max(0.0, 0.0087 - Q.lam1) * max(0.0, 0.86 - Q.D2) / 0.0002461958   # -2.8%  lam1 < 0.0087 and D2 < 0.86
        - 0.0195065 * max(0.0, 0.23 - Q.N2) * max(0.0, 0.19 - Q.sj2_dr) / 0.0009380599   # -2.0%  N2 < 0.23 and sj2_dr < 0.19
        + 0.01733978 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.85 - Q.D2) / 7.205433e-05   # +1.7%  lam1 < 0.0065 and D2 < 0.85
        - 0.01517842 * max(0.0, 0.16 - Q.N2) * max(0.0, Q.LHA - 0.38) / 0.0001308428   # -1.5%  N2 < 0.16 and LHA > 0.38
        - 0.007007178 * max(0.0, Q.tau1 - 0.14) / 0.003823558   # -0.7%  tau1 > 0.14
        - 0.005184095 * max(0.0, 0.0022 - Q.girth2_top3) * max(0.0, 0.14 - Q.planar_flow) / 1.500258e-05   # -0.5%  girth2_top3 < 0.0022 and planar_flow < 0.14
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.1601823529411766, 0.8530756302521009, 2.3418398109243697, 1.0343781512605041, 1.1630995798319328, 2.0605481092436975, 1.0277861344537815, 2.010894957983193, 0.3224834033613445, 3.1959810924369747, 2.0146871848739494, 2.028122268907563, 0.05940567226890756, 3.444470588235294, 0.505656512605042, 0.39958256302521006]
T = [2.6051934455422794, 1.4425466402967435, 3.540912657563025, 2.6798620568539917, 2.988293303571428]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +39%, n9 +21%, n5 -15%, n1 +13%, n0 -7%, n6 +4% ...
            + 0.3862513 * h[2] / H_AVG[2]
            + 0.2108516 * h[9] / H_AVG[9]
            - 0.148301 * h[5] / H_AVG[5]
            + 0.1279109 * h[1] / H_AVG[1]
            - 0.06958351 * h[0] / H_AVG[0]
            + 0.04315 * h[6] / H_AVG[6]
            - 0.01395169 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -17%, n6 +9%, n4 -8%, n5 +7%, n15 +2% ...
            + 0.5625326 * h[9] / H_AVG[9]
            - 0.1745773 * h[10] / H_AVG[10]
            + 0.08906004 * h[6] / H_AVG[6]
            - 0.07558895 * h[4] / H_AVG[4]
            + 0.06695672 * h[5] / H_AVG[5]
            + 0.01731238 * h[15] / H_AVG[15]
            + 0.01397197 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +21%, n3 -15%, n7 +12%, n0 +11%, n14 -11%, n6 -9% ...
            + 0.2147881 * h[11] / H_AVG[11]
            - 0.146061 * h[3] / H_AVG[3]
            + 0.1242288 * h[7] / H_AVG[7]
            + 0.1126299 * h[0] / H_AVG[0]
            - 0.107103 * h[14] / H_AVG[14]
            - 0.09070632 * h[6] / H_AVG[6]
            - 0.07758254 * h[15] / H_AVG[15]
            + 0.06839743 * h[13] / H_AVG[13]
            - 0.02820584 * h[9] / H_AVG[9]
            - 0.02276838 * h[8] / H_AVG[8]
            - 0.007528741 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -22%, n6 -14%, n14 +7%, n13 +7%, n1 +4% ...
            + 0.3517371 * h[7] / H_AVG[7]
            - 0.2171148 * h[3] / H_AVG[3]
            - 0.1438208 * h[6] / H_AVG[6]
            + 0.07075782 * h[14] / H_AVG[14]
            + 0.07029074 * h[13] / H_AVG[13]
            + 0.03979102 * h[1] / H_AVG[1]
            - 0.03726849 * h[9] / H_AVG[9]
            + 0.0339074 * h[4] / H_AVG[4]
            - 0.02329776 * h[15] / H_AVG[15]
            + 0.01201408 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +25%, n5 -17%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.468266 * h[13] / H_AVG[13]
            + 0.2528225 * h[10] / H_AVG[10]
            - 0.172385 * h[5] / H_AVG[5]
            + 0.04865234 * h[4] / H_AVG[4]
            + 0.02163397 * h[3] / H_AVG[3]
            + 0.02023417 * h[8] / H_AVG[8]
            - 0.009939733 * h[12] / H_AVG[12]
            + 0.006066288 * h[0] / H_AVG[0]
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
