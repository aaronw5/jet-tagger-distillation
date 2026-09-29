"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.1%   (on for 94% of jets)
  neuron  9:  12.2%   (on for 68% of jets)
  neuron  7:  10.2%   (on for 63% of jets)
  neuron 10:   8.3%   (on for 74% of jets)
  neuron  5:   8.1%   (on for 88% of jets)
  neuron  6:   7.9%   (on for 35% of jets)
  neuron  2:   7.2%   (on for 92% of jets)
  neuron  3:   7.2%   (on for 21% of jets)
  neuron 11:   6.0%   (on for 79% of jets)
  neuron 14:   4.0%   (on for 26% of jets)
  neuron  0:   3.9%   (on for 41% of jets)
  neuron  1:   3.6%   (on for 61% of jets)
  neuron  4:   3.3%   (on for 61% of jets)
  neuron 15:   2.6%   (on for 29% of jets)
  neuron  8:   1.1%   (on for 27% of jets)
  neuron 12:   0.3%   (on for 5% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 87.2% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
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
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        max_dr=max(dr[i] for i in real),
        n_for_90pct=ncum(0.9),
        pt_6=pt[6],
        pt_7=pt[7],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr23=subjets(3)["dr"][2],
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 19.6;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.59899 * (-0.0007296294
        + 0.2504668 * max(0.0, 0.0088 - Q.lam1_plus_lam2) / 0.004545275   # +25.0%  lam1_plus_lam2 < 0.0088
        - 0.1789236 * max(0.0, 0.0044 - Q.lam1_plus_lam2) / 0.001579604   # -17.9%  lam1_plus_lam2 < 0.0044
        - 0.1147774 * max(0.0, 0.088 - Q.sum_z_dr) / 0.03699871   # -11.5%  sum_z_dr < 0.088
        + 0.07721055 * max(0.0, 0.025 - Q.e2) / 0.007681467   # +7.7%  e2 < 0.025
        - 0.06991631 * max(0.0, 0.0016 - Q.C2_b2) / 0.0009074762   # -7.0%  C2_b2 < 0.0016
        + 0.04316359 * max(0.0, 0.15 - Q.planar_flow) / 0.05189956   # +4.3%  planar_flow < 0.15
        - 0.04283312 * max(0.0, 56.0 - Q.mass) / 21.14574   # -4.3%  mass < 56
        - 0.03693445 * max(0.0, Q.sj3_dr_max - 0.23) / 0.02603877   # -3.7%  sj3_dr_max > 0.23
        - 0.03143288 * max(0.0, Q.log_sum_pt - 6.7) / 0.03560998   # -3.1%  log_sum_pt > 6.7
        + 0.03026642 * max(0.0, Q.sum_pt_top5 - 690.0) / 36.39211   # +3.0%  sum_pt_top5 > 690
        + 0.02781313 * max(0.0, Q.n_dr_0_0p05 - 4.0) / 1.61275   # +2.8%  n_dr_0_0p05 > 4
        + 0.02750792 * max(0.0, 18.0 - Q.mass) / 3.171338   # +2.8%  mass < 18
        + 0.02244279 * max(0.0, 0.013 - Q.sum_z_dr2) * max(0.0, 1.0 - Q.D2) / 0.0009667167   # +2.2%  sum_z_dr2 < 0.013 and D2 < 1
        - 0.0134029 * max(0.0, 0.013 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.018) / 2.068373e-05   # -1.3%  sum_z_dr2 < 0.013 and centroid_offset > 0.018
        - 0.01211816 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.87 - Q.D2) / 7.761561e-05   # -1.2%  lam1 < 0.0065 and D2 < 0.87
        - 0.01154301 * max(0.0, Q.sum_pt - 900.0) / 12.63862   # -1.2%  sum_pt > 900
        + 0.009246954 * max(0.0, 7.1e-05 - Q.lam2) * max(0.0, 0.28 - Q.D2_b2) / 2.40041e-06   # +0.9%  lam2 < 7.1e-05 and D2_b2 < 0.28
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 9.436;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.435779 * (0.01041779
        + 0.2036556 * max(0.0, Q.log_sum_pt - 6.3) / 0.2710366   # +20.4%  log_sum_pt > 6.3
        + 0.1832275 * max(0.0, 0.0064 - Q.mass_over_sum_pt_sq) / 0.002960435   # +18.3%  mass_over_sum_pt_sq < 0.0064
        - 0.1604082 * max(0.0, 0.0088 - Q.sum_z_dr2) / 0.004545275   # -16.0%  sum_z_dr2 < 0.0088
        - 0.1458278 * max(0.0, 0.076 - Q.sum_z_dr) / 0.02791073   # -14.6%  sum_z_dr < 0.076
        - 0.1114822 * max(0.0, 0.06 - Q.z_7) / 0.01316548   # -11.1%  z_7 < 0.06
        + 0.06285127 * max(0.0, Q.pt_7 - 34.0) / 4.744406   # +6.3%  pt_7 > 34
        + 0.04575515 * max(0.0, Q.log_sum_pt - 6.6) / 0.0734244   # +4.6%  log_sum_pt > 6.6
        - 0.03453219 * max(0.0, Q.pt_7 - 33.0) * max(0.0, 0.017 - Q.tau2) / 0.04385439   # -3.5%  pt_7 > 33 and tau2 < 0.017
        + 0.02733944 * max(0.0, Q.sj3_dr_max - 0.19) / 0.03856038   # +2.7%  sj3_dr_max > 0.19
        - 0.02492062 * max(0.0, Q.sj3_dr_min - 0.041) / 0.02239481   # -2.5%  sj3_dr_min > 0.041
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 5.872;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.871657 * (0.299745
        + 0.1862984 * max(0.0, Q.N2 - 0.12) / 0.1115067   # +18.6%  N2 > 0.12
        - 0.1789442 * max(0.0, Q.LHA - 0.13) / 0.1199428   # -17.9%  LHA > 0.13
        - 0.1691686 * max(0.0, 53.0 - Q.pt_7) / 18.70622   # -16.9%  pt_7 < 53
        + 0.1353891 * max(0.0, 0.006 - Q.lam1) / 0.002547943   # +13.5%  lam1 < 0.006
        + 0.11867 * max(0.0, 810.0 - Q.sum_pt) / 126.689   # +11.9%  sum_pt < 810
        - 0.08593711 * max(0.0, 36.0 - Q.mass) / 10.01177   # -8.6%  mass < 36
        + 0.05641766 * max(0.0, 53.0 - Q.pt_7) * max(0.0, 1.1 - Q.D2_b2) / 8.881103   # +5.6%  pt_7 < 53 and D2_b2 < 1.1
        + 0.03939836 * max(0.0, 6.5 - Q.log_sum_pt) / 0.08351396   # +3.9%  log_sum_pt < 6.5
        - 0.02977655 * max(0.0, Q.sum_pt_top5 - 750.0) / 21.69202   # -3.0%  sum_pt_top5 > 750
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 13.72;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.72083 * (-0.3833588
        - 0.2642443 * max(0.0, Q.sum_z_dr2 - 0.0089) / 0.002171048   # -26.4%  sum_z_dr2 > 0.0089
        + 0.1864939 * max(0.0, Q.sum_z_dr - 0.042) / 0.02635273   # +18.6%  sum_z_dr > 0.042
        - 0.1003886 * max(0.0, 0.0044 - Q.sum_z_dr2) / 0.001579604   # -10.0%  sum_z_dr2 < 0.0044
        + 0.08239097 * max(0.0, Q.sj2_dr - 0.18) / 0.029986   # +8.2%  sj2_dr > 0.18
        + 0.07991168 * max(0.0, Q.lam1 - 0.0081) / 0.00189698   # +8.0%  lam1 > 0.0081
        + 0.07392123 * max(0.0, Q.sum_z_dr - 0.038) * max(0.0, Q.log_sum_pt - 6.1) / 0.008975755   # +7.4%  sum_z_dr > 0.038 and log_sum_pt > 6.1
        + 0.07115453 * max(0.0, Q.sum_z_dr - 0.078) / 0.01003391   # +7.1%  sum_z_dr > 0.078
        - 0.04725886 * max(0.0, Q.mass_over_sum_pt - 0.066) * max(0.0, 0.22 - Q.sj2_dr) / 0.0001515025   # -4.7%  mass_over_sum_pt > 0.066 and sj2_dr < 0.22
        + 0.04006494 * max(0.0, Q.e2 - 0.064) / 0.001707218   # +4.0%  e2 > 0.064
        + 0.03247414 * max(0.0, Q.centroid_offset - 0.01) * max(0.0, 8.4 - Q.n_pt_above_50) / 0.03536286   # +3.2%  centroid_offset > 0.01 and n_pt_above_50 < 8.4
        - 0.02014604 * max(0.0, Q.mass - 65.0) / 3.19931   # -2.0%  mass > 65
        + 0.001550835 * max(0.0, Q.lam1 - 0.016) * max(0.0, 6.0 - Q.n_for_90pct) / 5.674331e-06   # +0.2%  lam1 > 0.016 and n_for_90pct < 6
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 20.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.28966 * (0.1715159
        - 0.213225 * max(0.0, 0.00054 - Q.lam2) / 0.0003932965   # -21.3%  lam2 < 0.00054
        + 0.1238324 * max(0.0, 0.21 - Q.N2) / 0.0446273   # +12.4%  N2 < 0.21
        - 0.1115501 * max(0.0, 0.056 - Q.centroid_offset) / 0.03936196   # -11.2%  centroid_offset < 0.056
        + 0.1019608 * max(0.0, 0.0082 - Q.sum_z_dr2_top2) / 0.004996981   # +10.2%  sum_z_dr2_top2 < 0.0082
        - 0.09005424 * max(0.0, 0.22 - Q.sj3_dr_max) / 0.08049206   # -9.0%  sj3_dr_max < 0.22
        + 0.08947728 * max(0.0, 0.0017 - Q.lam1_plus_lam2) / 0.0004449665   # +8.9%  lam1_plus_lam2 < 0.0017
        + 0.07780357 * max(0.0, Q.sd_mass - 43.0) / 9.509686   # +7.8%  sd_mass > 43
        - 0.0552079 * max(0.0, Q.mass_over_sum_pt - 0.086) / 0.009257435   # -5.5%  mass_over_sum_pt > 0.086
        + 0.03026485 * max(0.0, Q.max_dr - 0.15) / 0.02466119   # +3.0%  max_dr > 0.15
        - 0.03011858 * max(0.0, 0.22 - Q.N2) * max(0.0, 53.0 - Q.pt_7) / 0.8487442   # -3.0%  N2 < 0.22 and pt_7 < 53
        - 0.02341547 * max(0.0, 0.0093 - Q.sum_z_dr2_top2) * max(0.0, Q.C2_b2 - 0.00095) / 7.725071e-06   # -2.3%  sum_z_dr2_top2 < 0.0093 and C2_b2 > 0.00095
        - 0.02121345 * max(0.0, Q.sd_mass - 43.0) * max(0.0, 0.29 - Q.sd_zg) / 0.4455628   # -2.1%  sd_mass > 43 and sd_zg < 0.29
        - 0.017715 * max(0.0, Q.mass - 76.0) / 1.611799   # -1.8%  mass > 76
        - 0.01416145 * max(0.0, Q.lam1_plus_lam2 - 0.005) * max(0.0, 0.11 - Q.sj3_z3) / 4.845379e-05   # -1.4%  lam1_plus_lam2 > 0.005 and sj3_z3 < 0.11
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 6.083;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.08325 * (0.01060288
        + 0.2049473 * max(0.0, 0.0017 - Q.sum_z_dr2) * max(0.0, 0.03 - Q.centroid_offset) / 1.00544e-05   # +20.5%  sum_z_dr2 < 0.0017 and centroid_offset < 0.03
        + 0.1724501 * max(0.0, 0.0055 - Q.sum_zz_dr2) / 0.002368074   # +17.2%  sum_zz_dr2 < 0.0055
        - 0.1301777 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt) / 0.004257547   # -13.0%  LHA < 0.22 and log_sum_pt < 6.8
        - 0.1146196 * max(0.0, 0.075 - Q.z_7) * max(0.0, 0.025 - Q.centroid_offset) / 0.0003320285   # -11.5%  z_7 < 0.075 and centroid_offset < 0.025
        + 0.1077432 * max(0.0, 0.057 - Q.z_7) / 0.01135925   # +10.8%  z_7 < 0.057
        + 0.07989402 * max(0.0, Q.sum_pt_top5 - 350.0) * max(0.0, Q.sj2_dr - 0.098) / 15.23559   # +8.0%  sum_pt_top5 > 350 and sj2_dr > 0.098
        - 0.07492833 * max(0.0, Q.log_sum_pt - 6.7) / 0.03560998   # -7.5%  log_sum_pt > 6.7
        + 0.05249105 * max(0.0, 0.029 - Q.z_7) / 0.001451437   # +5.2%  z_7 < 0.029
        + 0.03535121 * max(0.0, 0.0074 - Q.sum_z_dr) / 0.0002048098   # +3.5%  sum_z_dr < 0.0074
        - 0.02739749 * max(0.0, Q.log_sum_pt - 6.9) / 0.003849095   # -2.7%  log_sum_pt > 6.9
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 25.52;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.52138 * (0.118724
        - 0.2834182 * max(0.0, 0.0087 - Q.sum_z_dr2) / 0.004464952   # -28.3%  sum_z_dr2 < 0.0087
        + 0.1515096 * max(0.0, 0.11 - Q.tau1) / 0.05423188   # +15.2%  tau1 < 0.11
        + 0.09826722 * max(0.0, 0.18 - Q.sj3_dr_max) / 0.05451989   # +9.8%  sj3_dr_max < 0.18
        - 0.0868385 * max(0.0, 0.0012 - Q.lam2) / 0.0009763163   # -8.7%  lam2 < 0.0012
        - 0.07194438 * max(0.0, 0.14 - Q.max_dr) / 0.0444581   # -7.2%  max_dr < 0.14
        + 0.06262223 * max(0.0, Q.centroid_offset - 0.0079) / 0.0106547   # +6.3%  centroid_offset > 0.0079
        + 0.05182267 * max(0.0, 0.0032 - Q.sum_zz_dr2) / 0.001150075   # +5.2%  sum_zz_dr2 < 0.0032
        + 0.04388786 * max(0.0, 41.0 - Q.pt_6) * max(0.0, 6.8 - Q.log_sum_pt) / 1.098116   # +4.4%  pt_6 < 41 and log_sum_pt < 6.8
        - 0.04247331 * max(0.0, 64.0 - Q.mass) * max(0.0, 0.052 - Q.z_dr_0p2_0p4) / 1.358368   # -4.2%  mass < 64 and z_dr_0p2_0p4 < 0.052
        - 0.03252468 * max(0.0, 41.0 - Q.pt_6) * max(0.0, Q.z_7 - 0.023) / 0.06917289   # -3.3%  pt_6 < 41 and z_7 > 0.023
        - 0.02850915 * max(0.0, Q.centroid_offset - 0.019) / 0.005310896   # -2.9%  centroid_offset > 0.019
        + 0.01537218 * max(0.0, Q.sj3_dr_max - 0.18) / 0.04273628   # +1.5%  sj3_dr_max > 0.18
        - 0.01462202 * max(0.0, Q.sj3_pair_mass_min - 4.0) / 3.694793   # -1.5%  sj3_pair_mass_min > 4
        + 0.008339019 * max(0.0, 0.0028 - Q.lam2) * max(0.0, 3.1 - Q.n_dr_0_0p05) / 0.002216909   # +0.8%  lam2 < 0.0028 and n_dr_0_0p05 < 3.1
        - 0.007849078 * max(0.0, Q.sj3_dr_min - 0.14) / 0.007053496   # -0.8%  sj3_dr_min > 0.14
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 25.72;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.7207 * (0.269044
        - 0.1592827 * max(0.0, Q.lam1_plus_lam2 - 0.0056) / 0.003080348   # -15.9%  lam1_plus_lam2 > 0.0056
        + 0.1212768 * max(0.0, 0.16 - Q.sj2_dr) / 0.04881572   # +12.1%  sj2_dr < 0.16
        - 0.1160928 * max(0.0, 0.19 - Q.sj2_dr) / 0.0663553   # -11.6%  sj2_dr < 0.19
        - 0.1136905 * max(0.0, 0.086 - Q.sum_z_dr) / 0.03540191   # -11.4%  sum_z_dr < 0.086
        + 0.09538602 * max(0.0, Q.sum_z_dr2 - 0.013) / 0.001477949   # +9.5%  sum_z_dr2 > 0.013
        - 0.08307645 * max(0.0, 8.1e-05 - Q.e3) / 5.593676e-05   # -8.3%  e3 < 8.1e-05
        + 0.07756149 * max(0.0, Q.mass_over_sum_pt - 0.075) / 0.01254677   # +7.8%  mass_over_sum_pt > 0.075
        - 0.06299803 * max(0.0, Q.mass_over_sum_pt - 0.091) / 0.008183604   # -6.3%  mass_over_sum_pt > 0.091
        + 0.04171785 * max(0.0, 0.059 - Q.tau1) / 0.01972449   # +4.2%  tau1 < 0.059
        + 0.02992364 * max(0.0, Q.mass - 36.0) / 14.33253   # +3.0%  mass > 36
        - 0.02427146 * max(0.0, Q.e2 - 0.05) / 0.00341136   # -2.4%  e2 > 0.05
        - 0.02283749 * max(0.0, 0.001 - Q.sum_z_dr2) / 0.0002159545   # -2.3%  sum_z_dr2 < 0.001
        + 0.02248234 * max(0.0, Q.sum_z_dr2 - 0.0032) * max(0.0, 0.21 - Q.planar_flow) / 0.0003683195   # +2.2%  sum_z_dr2 > 0.0032 and planar_flow < 0.21
        - 0.01687671 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.lam1_plus_lam2 - 0.008) / 0.0001736323   # -1.7%  planar_flow < 0.23 and lam1_plus_lam2 > 0.008
        - 0.01252576 * max(0.0, Q.mass - 77.0) / 1.512542   # -1.3%  mass > 77
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 5.895;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.895221 * (-0.1155173
        + 0.3225721 * max(0.0, 0.0063 - Q.sum_z_dr2) * max(0.0, 0.025 - Q.centroid_offset) / 3.912827e-05   # +32.3%  sum_z_dr2 < 0.0063 and centroid_offset < 0.025
        - 0.3081676 * max(0.0, 0.21 - Q.LHA) / 0.0298802   # -30.8%  LHA < 0.21
        + 0.2135101 * max(0.0, 0.058 - Q.tau1) * max(0.0, 0.0032 - Q.lam1_plus_lam2) / 4.679141e-05   # +21.4%  tau1 < 0.058 and lam1_plus_lam2 < 0.0032
        - 0.1244001 * max(0.0, 0.13 - Q.sj3_dr_max) / 0.03230688   # -12.4%  sj3_dr_max < 0.13
        - 0.03135011 * max(0.0, Q.log_sum_pt - 6.7) / 0.03560998   # -3.1%  log_sum_pt > 6.7
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 19.27;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.27492 * (0.004741914
        - 0.138993 * max(0.0, 0.14 - Q.sj3_dr_max) / 0.03615491   # -13.9%  sj3_dr_max < 0.14
        - 0.1082962 * max(0.0, 55.0 - Q.mass) / 20.46471   # -10.8%  mass < 55
        + 0.09863619 * max(0.0, 0.0059 - Q.sum_z_dr2) / 0.002418835   # +9.9%  sum_z_dr2 < 0.0059
        + 0.09612692 * max(0.0, 54.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset) / 0.3057489   # +9.6%  mass < 54 and centroid_offset < 0.027
        + 0.08279414 * max(0.0, 0.0035 - Q.sum_z_dr2) / 0.001156413   # +8.3%  sum_z_dr2 < 0.0035
        + 0.07967695 * max(0.0, 0.11 - Q.max_dr) / 0.0275227   # +8.0%  max_dr < 0.11
        - 0.07633488 * max(0.0, Q.log_sum_pt - 6.4) / 0.1930903   # -7.6%  log_sum_pt > 6.4
        + 0.07542643 * max(0.0, 0.21 - Q.sj3_dr_max) / 0.07342616   # +7.5%  sj3_dr_max < 0.21
        + 0.06396368 * max(0.0, 0.019 - Q.centroid_offset) / 0.007126558   # +6.4%  centroid_offset < 0.019
        - 0.04622349 * max(0.0, Q.sum_z_dr - 0.069) / 0.01300663   # -4.6%  sum_z_dr > 0.069
        - 0.03559213 * max(0.0, 0.028 - Q.C3) / 0.008459128   # -3.6%  C3 < 0.028
        - 0.02405024 * max(0.0, 0.02 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.5) / 0.00707735   # -2.4%  centroid_offset < 0.02 and n_for_90pct > 5.5
        + 0.021278 * max(0.0, 570.0 - Q.sum_pt) / 18.30945   # +2.1%  sum_pt < 570
        + 0.02016495 * max(0.0, Q.e2 - 0.047) / 0.003886777   # +2.0%  e2 > 0.047
        + 0.01651399 * max(0.0, 0.0086 - Q.sum_z_dr2) * max(0.0, 0.33 - Q.planar_flow) / 0.0005076647   # +1.7%  sum_z_dr2 < 0.0086 and planar_flow < 0.33
        + 0.01400369 * max(0.0, Q.lam2 - 0.00083) / 0.0003442857   # +1.4%  lam2 > 0.00083
        + 0.001925055 * max(0.0, Q.log_sum_pt - 6.9) / 0.003849095   # +0.2%  log_sum_pt > 6.9
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 9.078;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.078439 * (0.03745137
        + 0.1523046 * max(0.0, 8e-05 - Q.e3) / 5.508718e-05   # +15.2%  e3 < 8e-05
        - 0.1337284 * max(0.0, 0.0056 - Q.lam1) / 0.002294981   # -13.4%  lam1 < 0.0056
        + 0.1202732 * max(0.0, 75.0 - Q.mass) / 36.39643   # +12.0%  mass < 75
        + 0.09269878 * max(0.0, Q.sum_z_dr2 - 0.0075) / 0.002475177   # +9.3%  sum_z_dr2 > 0.0075
        - 0.08277504 * max(0.0, 0.0015 - Q.lam1) / 0.0003913897   # -8.3%  lam1 < 0.0015
        + 0.07730407 * max(0.0, Q.tau1 - 0.057) / 0.0250643   # +7.7%  tau1 > 0.057
        - 0.06937561 * max(0.0, Q.LHA - 0.3) / 0.01897055   # -6.9%  LHA > 0.3
        - 0.05418368 * max(0.0, 46.0 - Q.pt_7) / 12.35938   # -5.4%  pt_7 < 46
        + 0.03988993 * max(0.0, 0.022 - Q.zdr_0) * max(0.0, 0.24 - Q.z_dr_0p05_0p1) / 0.001775188   # +4.0%  zdr_0 < 0.022 and z_dr_0p05_0p1 < 0.24
        + 0.03983099 * max(0.0, Q.lam2 - 0.00034) / 0.0004151587   # +4.0%  lam2 > 0.00034
        - 0.03149775 * max(0.0, Q.e3 - 7.5e-05) / 5.097154e-05   # -3.1%  e3 > 7.5e-05
        + 0.02364349 * max(0.0, Q.C2_b2 - 0.0093) / 0.001676922   # +2.4%  C2_b2 > 0.0093
        + 0.02296994 * max(0.0, 45.0 - Q.pt_7) * max(0.0, 0.98 - Q.D2) / 1.616521   # +2.3%  pt_7 < 45 and D2 < 0.98
        + 0.02170236 * max(0.0, Q.sj3_pair_mass_min - 11.0) / 1.950728   # +2.2%  sj3_pair_mass_min > 11
        - 0.01762767 * max(0.0, Q.lam1 - 0.0074) * max(0.0, 0.38 - Q.D2_b2) / 0.000288866   # -1.8%  lam1 > 0.0074 and D2_b2 < 0.38
        - 0.01115087 * max(0.0, Q.lam2 - 0.0034) / 0.0001574377   # -1.1%  lam2 > 0.0034
        - 0.009043594 * max(0.0, 8.1e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.18) / 5.822817e-07   # -0.9%  e3 < 8.1e-05 and sj3_dr23 > 0.18
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 16.62;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.62306 * (0.05023143
        + 0.214285 * max(0.0, 0.0094 - Q.lam1_plus_lam2) / 0.005031174   # +21.4%  lam1_plus_lam2 < 0.0094
        - 0.1611676 * max(0.0, 0.17 - Q.sj3_dr_max) / 0.04924812   # -16.1%  sj3_dr_max < 0.17
        + 0.119574 * max(0.0, 0.26 - Q.sj3_dr_max) / 0.1110439   # +12.0%  sj3_dr_max < 0.26
        - 0.1076837 * max(0.0, 0.072 - Q.sum_z_dr) / 0.02521171   # -10.8%  sum_z_dr < 0.072
        - 0.08724221 * max(0.0, Q.centroid_offset - 0.01) / 0.009356337   # -8.7%  centroid_offset > 0.01
        + 0.07407041 * max(0.0, Q.z_7 - 0.018) / 0.03448955   # +7.4%  z_7 > 0.018
        - 0.05967806 * max(0.0, 0.041 - Q.centroid_offset) * max(0.0, 900.0 - Q.sum_pt) / 4.016323   # -6.0%  centroid_offset < 0.041 and sum_pt < 900
        + 0.04598783 * max(0.0, 0.15 - Q.max_dr) / 0.05096389   # +4.6%  max_dr < 0.15
        - 0.04282054 * max(0.0, 0.021 - Q.sum_z_dr) / 0.002706496   # -4.3%  sum_z_dr < 0.021
        + 0.02791414 * max(0.0, 16.0 - Q.mass) / 2.549551   # +2.8%  mass < 16
        + 0.01790067 * max(0.0, 0.01 - Q.centroid_offset) / 0.002171999   # +1.8%  centroid_offset < 0.01
        - 0.01683892 * max(0.0, Q.centroid_offset - 0.05) * max(0.0, 2.8 - Q.D2) / 0.001320351   # -1.7%  centroid_offset > 0.05 and D2 < 2.8
        + 0.01307307 * max(0.0, Q.centroid_offset - 0.047) * max(0.0, 0.64 - Q.tau32) / 0.00018734   # +1.3%  centroid_offset > 0.047 and tau32 < 0.64
        + 0.01176398 * max(0.0, Q.sj2_dr - 0.27) / 0.008286158   # +1.2%  sj2_dr > 0.27
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.4577;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.4576847 * (-2.993327
        + 0.5638872 * max(0.0, Q.sum_z_dr2 - 0.019) / 0.0007437538   # +56.4%  sum_z_dr2 > 0.019
        - 0.3128646 * max(0.0, Q.e2 - 0.063) / 0.001805717   # -31.3%  e2 > 0.063
        + 0.1232482 * max(0.0, Q.mass - 91.0) / 0.5919078   # +12.3%  mass > 91
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 14.14;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.14143 * (0.002658854
        + 0.3426283 * max(0.0, 0.15 - Q.sum_z_dr) / 0.09211511   # +34.3%  sum_z_dr < 0.15
        + 0.1873052 * max(0.0, 0.016 - Q.lam1) / 0.01081128   # +18.7%  lam1 < 0.016
        - 0.1505576 * max(0.0, 0.08 - Q.e2) / 0.05192925   # -15.1%  e2 < 0.08
        - 0.07109973 * max(0.0, 0.15 - Q.sum_z_dr) * max(0.0, 6.8 - Q.log_sum_pt) / 0.0200689   # -7.1%  sum_z_dr < 0.15 and log_sum_pt < 6.8
        - 0.06798362 * max(0.0, 0.15 - Q.sum_z_dr) * max(0.0, 38.0 - Q.pt_7) / 0.6495849   # -6.8%  sum_z_dr < 0.15 and pt_7 < 38
        + 0.06319156 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 41.0 - Q.pt_7) / 812.381   # +6.3%  sum_pt_top5 > 660 and pt_7 < 41
        - 0.05610608 * max(0.0, 6.1e-05 - Q.e3) * max(0.0, 0.038 - Q.centroid_offset) / 9.930165e-07   # -5.6%  e3 < 6.1e-05 and centroid_offset < 0.038
        + 0.02852972 * max(0.0, 5.8e-05 - Q.e3) * max(0.0, Q.D3 - 0.16) / 4.482789e-05   # +2.9%  e3 < 5.8e-05 and D3 > 0.16
        - 0.02811719 * max(0.0, 0.028 - Q.z_7) / 0.001299403   # -2.8%  z_7 < 0.028
        - 0.004480962 * max(0.0, Q.sum_pt - 990.0) * max(0.0, 62.0 - Q.pt_6) / 101.3876   # -0.4%  sum_pt > 990 and pt_6 < 62
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 35.1;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.10039 * (-0.049857
        - 0.2108461 * max(0.0, Q.sum_z_dr2 - 0.0075) / 0.002475177   # -21.1%  sum_z_dr2 > 0.0075
        + 0.111997 * max(0.0, Q.sum_z_dr - 0.027) / 0.03606549   # +11.2%  sum_z_dr > 0.027
        - 0.09749278 * max(0.0, Q.e2 - 0.017) / 0.01569741   # -9.7%  e2 > 0.017
        + 0.08850444 * max(0.0, Q.mass_over_sum_pt - 0.085) / 0.009500123   # +8.9%  mass_over_sum_pt > 0.085
        - 0.06320985 * max(0.0, 75.0 - Q.sd_mass) / 42.42237   # -6.3%  sd_mass < 75
        - 0.05500195 * max(0.0, Q.sum_z_dr - 0.089) / 0.007512022   # -5.5%  sum_z_dr > 0.089
        + 0.04880082 * max(0.0, Q.sum_z_dr - 0.042) / 0.02635273   # +4.9%  sum_z_dr > 0.042
        + 0.04548581 * max(0.0, Q.sj2_dr - 0.16) / 0.03884597   # +4.5%  sj2_dr > 0.16
        + 0.03719107 * max(0.0, 50.0 - Q.sd_mass) / 22.66356   # +3.7%  sd_mass < 50
        + 0.03625137 * max(0.0, Q.e2 - 0.05) / 0.00341136   # +3.6%  e2 > 0.05
        + 0.03404368 * max(0.0, Q.lam1_plus_lam2 - 0.014) / 0.001333646   # +3.4%  lam1_plus_lam2 > 0.014
        + 0.03241687 * max(0.0, Q.lam1_plus_lam2 - 0.0033) / 0.00421424   # +3.2%  lam1_plus_lam2 > 0.0033
        - 0.0253476 * max(0.0, Q.centroid_offset - 0.05) / 0.0008015409   # -2.5%  centroid_offset > 0.05
        - 0.02228242 * max(0.0, 5.1e-05 - Q.e3) / 3.116023e-05   # -2.2%  e3 < 5.1e-05
        + 0.01922538 * max(0.0, Q.e2 - 0.042) / 0.004889989   # +1.9%  e2 > 0.042
        - 0.0155089 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.0072 - Q.lam1) / 5.48206e-05   # -1.6%  planar_flow < 0.11 and lam1 < 0.0072
        - 0.0132331 * max(0.0, Q.sj2_dr - 0.2) / 0.02322434   # -1.3%  sj2_dr > 0.2
        + 0.0111251 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.016 - Q.lam1) / 0.0003003809   # +1.1%  planar_flow < 0.11 and lam1 < 0.016
        - 0.01045787 * max(0.0, Q.centroid_offset - 0.026) / 0.00343061   # -1.0%  centroid_offset > 0.026
        - 0.008964204 * max(0.0, Q.sum_z_dr - 0.094) / 0.006638123   # -0.9%  sum_z_dr > 0.094
        + 0.008510977 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.0061 - Q.lam1_plus_lam2) / 3.272055e-05   # +0.9%  planar_flow < 0.11 and lam1_plus_lam2 < 0.0061
        - 0.004102682 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 0.018 - Q.centroid_offset) / 0.0001818254   # -0.4%  planar_flow < 0.1 and centroid_offset < 0.018
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 23.97;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.96792 * (-0.004213966
        + 0.2076738 * max(0.0, 0.014 - Q.lam1_plus_lam2) / 0.008888412   # +20.8%  lam1_plus_lam2 < 0.014
        - 0.1919172 * max(0.0, 0.1 - Q.sum_z_dr) / 0.04698525   # -19.2%  sum_z_dr < 0.1
        - 0.1713495 * max(0.0, 0.008 - Q.sum_z_dr2) / 0.003911325   # -17.1%  sum_z_dr2 < 0.008
        + 0.1419995 * max(0.0, 0.04 - Q.e2) / 0.01676567   # +14.2%  e2 < 0.04
        - 0.137728 * max(0.0, 0.2 - Q.sj2_dr) / 0.07319409   # -13.8%  sj2_dr < 0.2
        + 0.09878047 * max(0.0, 0.16 - Q.sj2_dr) / 0.04881572   # +9.9%  sj2_dr < 0.16
        - 0.02917706 * max(0.0, 0.0005 - Q.sum_z_dr2_top2) / 0.0001067654   # -2.9%  sum_z_dr2_top2 < 0.0005
        + 0.01263197 * max(0.0, 0.15 - Q.mass_over_sum_pt) * max(0.0, 0.73 - Q.D2) / 0.005368121   # +1.3%  mass_over_sum_pt < 0.15 and D2 < 0.73
        - 0.008742528 * max(0.0, 0.0061 - Q.lam1_plus_lam2) * max(0.0, 0.72 - Q.D2) / 2.823992e-05   # -0.9%  lam1_plus_lam2 < 0.0061 and D2 < 0.72
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0124420168067227, 0.8800220588235295, 2.2291656512605043, 0.8428869747899159, 1.3386302521008404, 2.135642857142857, 1.1354243697478992, 1.963429411764706, 0.2977911764705882, 3.3096563025210086, 2.1961405462184875, 2.1361579831932773, 0.08245546218487394, 3.510571218487395, 0.4728797268907563, 0.37977100840336137]
T = [2.5950967461594017, 1.5247405166754202, 3.4228146188944333, 2.6002985047925415, 3.1165251772584033]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +37%, n9 +22%, n5 -15%, n1 +13%, n0 -6%, n6 +5% ...
            + 0.3690978 * h[2] / H_AVG[2]
            + 0.2192008 * h[9] / H_AVG[9]
            - 0.1543037 * h[5] / H_AVG[5]
            + 0.1324647 * h[1] / H_AVG[1]
            - 0.06095883 * h[0] / H_AVG[0]
            + 0.04785449 * h[6] / H_AVG[6]
            - 0.01611971 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +55%, n10 -18%, n6 +9%, n4 -8%, n5 +7%, n15 +2% ...
            + 0.551138 * h[9] / H_AVG[9]
            - 0.1800422 * h[10] / H_AVG[10]
            + 0.09308341 * h[6] / H_AVG[6]
            - 0.08230685 * h[4] / H_AVG[4]
            + 0.06565593 * h[5] / H_AVG[5]
            + 0.01556703 * h[15] / H_AVG[15]
            + 0.01220663 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +23%, n7 +13%, n3 -12%, n6 -10%, n14 -10%, n0 +10% ...
            + 0.2340352 * h[11] / H_AVG[11]
            + 0.1254816 * h[7] / H_AVG[7]
            - 0.1231278 * h[3] / H_AVG[3]
            - 0.1036633 * h[6] / H_AVG[6]
            - 0.1036164 * h[14] / H_AVG[14]
            + 0.1016786 * h[0] / H_AVG[0]
            - 0.07628008 * h[15] / H_AVG[15]
            + 0.07211522 * h[13] / H_AVG[13]
            - 0.03021687 * h[9] / H_AVG[9]
            - 0.02175046 * h[8] / H_AVG[8]
            - 0.008034525 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -18%, n6 -16%, n13 +7%, n14 +7%, n1 +4% ...
            + 0.353943 * h[7] / H_AVG[7]
            - 0.1823344 * h[3] / H_AVG[3]
            - 0.1637443 * h[6] / H_AVG[6]
            + 0.07383166 * h[13] / H_AVG[13]
            + 0.06819598 * h[14] / H_AVG[14]
            + 0.0423039 * h[1] / H_AVG[1]
            + 0.04021865 * h[4] / H_AVG[4]
            - 0.03977496 * h[9] / H_AVG[9]
            - 0.02282016 * h[15] / H_AVG[15]
            + 0.01283292 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -46%, n10 +26%, n5 -17%, n4 +5%, n8 +2%, n3 +2% ...
            - 0.4576153 * h[13] / H_AVG[13]
            + 0.2642535 * h[10] / H_AVG[10]
            - 0.171316 * h[5] / H_AVG[5]
            + 0.05369082 * h[4] / H_AVG[4]
            + 0.01791606 * h[8] / H_AVG[8]
            + 0.01690358 * h[3] / H_AVG[3]
            - 0.01322875 * h[12] / H_AVG[12]
            + 0.005075976 * h[0] / H_AVG[0]
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
