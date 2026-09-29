"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.3%   (on for 86% of jets)
  neuron  9:  12.0%   (on for 68% of jets)
  neuron  7:  10.4%   (on for 60% of jets)
  neuron  3:   8.7%   (on for 25% of jets)
  neuron 10:   7.9%   (on for 76% of jets)
  neuron  6:   7.6%   (on for 32% of jets)
  neuron  5:   7.2%   (on for 70% of jets)
  neuron  2:   7.2%   (on for 89% of jets)
  neuron 11:   6.3%   (on for 73% of jets)
  neuron 14:   3.9%   (on for 29% of jets)
  neuron  0:   3.7%   (on for 41% of jets)
  neuron  1:   3.5%   (on for 59% of jets)
  neuron  4:   3.3%   (on for 44% of jets)
  neuron 15:   2.6%   (on for 25% of jets)
  neuron  8:   1.1%   (on for 34% of jets)
  neuron 12:   0.3%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.7% (the network: 65.8%); same class as the network for 90.3% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z1                 pT share of subjet 1 of 3 (by pT)
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_nremoved            number of branches removed by soft drop
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_4                  ΔR between particle 4 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.phi_0                  Δφ of particle 0
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
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau4                   N-subjettiness τ4 (β=1)
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
        z_top5=sum(zs[:5]),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        pt_1=pt[1],
        pt_2=pt[2],
        pt_3=pt[3],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z1=subjets(3)["z"][0],
        z_top3_slots=sum(pt[:3]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_7=z[7] * dr[7],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_nremoved=softdrop("removed"),
        abseta_0=abs(eta[0]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_4=math.sqrt(dist2(1, 4)) if pt[4] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        eta_0=eta[0],
        phi_0=phi[0],
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
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 18.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.78443 * (-0.08152191
        + 0.1146143 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +11.5%  e3 < 0.0005117
        + 0.1066703 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +10.7%  lam1 < 0.012
        + 0.06894026 * max(0.0, Q.sj3_dr_max - 0.1426152) / 0.06279029   # +6.9%  sj3_dr_max > 0.1426
        - 0.05994547 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -6.0%  lam1 < 0.004184
        - 0.05315827 * max(0.0, 0.01713288 - Q.tau2) / 0.008065871   # -5.3%  tau2 < 0.01713
        - 0.04503922 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -4.5%  centroid_offset > 0.00679
        - 0.04492949 * max(0.0, 0.08065885 - Q.girth) / 0.03128522   # -4.5%  girth < 0.08066
        - 0.04295671 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # -4.3%  sj3_dr_max > 0.169
        + 0.03951496 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +4.0%  e3 < 8.148e-05
        - 0.03507151 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -3.5%  LHA > 0.3033
        - 0.03445508 * max(0.0, 763.825 - Q.sum_pt) / 96.68366   # -3.4%  sum_pt < 763.8
        - 0.03337385 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # -3.3%  sj3_dr_max > 0.1986
        + 0.03090059 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +3.1%  tau1 < 0.05357
        - 0.03038027 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.002399898   # -3.0%  lam1 < 0.005954 and n_dr_0p2_0p4 < 1
        + 0.02945375 * max(0.0, Q.LHA - 0.1546891) / 0.1006902   # +2.9%  LHA > 0.1547
        + 0.02309116 * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.8069227   # +2.3%  n_dr_0p2_0p4 < 1
        - 0.01799938 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -1.8%  lam1 < 0.005954
        + 0.01578352 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +1.6%  sum_pt_top5 > 791.1
        + 0.0155827 * max(0.0, 0.02045966 - Q.e2) / 0.005531925   # +1.6%  e2 < 0.02046
        + 0.0151223 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, Q.centroid_offset - 0.01096064) / 1.576925   # +1.5%  sum_pt < 763.8 and centroid_offset > 0.01096
        - 0.01497761 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # -1.5%  sj3_dr_max < 0.3456
        - 0.01439357 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.0005116989 - Q.e3) / 0.03717154   # -1.4%  sum_pt < 763.8 and e3 < 0.0005117
        - 0.01431742 * max(0.0, 0.007929074 - Q.girth2_top3) / 0.004560262   # -1.4%  girth2_top3 < 0.007929
        + 0.01214123 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, Q.z_7 - 0.01685855) / 5.053524   # +1.2%  sum_pt < 763.8 and z_7 > 0.01686
        - 0.0109717 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # -1.1%  log_sum_pt > 6.843
        + 0.01088173 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +1.1%  e2 > 0.04111
        + 0.01002192 * max(0.0, 0.004183811 - Q.lam1) * max(0.0, 763.825 - Q.sum_pt) / 0.08579672   # +1.0%  lam1 < 0.004184 and sum_pt < 763.8
        - 0.009600797 * max(0.0, Q.log_sum_pt - 6.766778) / 0.01900249   # -1.0%  log_sum_pt > 6.767
        + 0.009052274 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +0.9%  planar_flow < 0.1484
        + 0.006206682 * max(0.0, Q.sj3_dr_max - 0.1426152) * max(0.0, Q.D2 - 0.2568137) / 0.08032253   # +0.6%  sj3_dr_max > 0.1426 and D2 > 0.2568
        + 0.005631937 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.001732318   # +0.6%  planar_flow < 0.1484 and centroid_offset < 0.0499
        + 0.005032531 * max(0.0, 0.04505724 - Q.planar_flow) * max(0.0, 0.05932655 - Q.tau21_b2) / 0.0003618271   # +0.5%  planar_flow < 0.04506 and tau21_b2 < 0.05933
        + 0.004344193 * max(0.0, Q.log_sum_pt - 6.766778) * max(0.0, 0.04510668 - Q.z_dr_0p1_0p2) / 0.0006838589   # +0.4%  log_sum_pt > 6.767 and z_dr_0p1_0p2 < 0.04511
        - 0.003697028 * max(0.0, 27.57812 - Q.pt_6) / 0.9875435   # -0.4%  pt_6 < 27.58
        - 0.003145465 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.380911 - Q.D2_b2) / 10.72478   # -0.3%  sum_pt < 763.8 and D2_b2 < 0.3809
        - 0.002224063 * max(0.0, 0.004183811 - Q.lam1) * max(0.0, 0.7459513 - Q.D2) / 6.576264e-06   # -0.2%  lam1 < 0.004184 and D2 < 0.746
        - 0.001857622 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 0.04037476 - Q.eta_0) / 0.5936549   # -0.2%  sum_pt_top5 > 791.1 and eta_0 < 0.04037
        + 0.001747934 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, Q.C3 - 0.04857476) / 0.0001233564   # +0.2%  planar_flow < 0.1484 and C3 > 0.04857
        + 0.001516067 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, 0.06396627 - Q.dr_7) / 0.0002655991   # +0.2%  sj3_dr_max > 0.169 and dr_7 < 0.06397
        + 0.001255182 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.03843804 - Q.dr_2) / 0.0001824973   # +0.1%  log_sum_pt > 6.843 and dr_2 < 0.03844
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 19.28;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.28449 * (0.1249629
        - 0.09770469 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # -9.8%  girth < 0.1019
        + 0.06896972 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +6.9%  lam1 < 0.012
        - 0.068569 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.001130645 - Q.lam2) / 4.482383e-06   # -6.9%  lam1 < 0.008376 and lam2 < 0.001131
        + 0.06807662 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +6.8%  log_sum_pt > 6.378
        - 0.06719996 * max(0.0, 0.06473447 - Q.z_7) / 0.01631636   # -6.7%  z_7 < 0.06473
        - 0.04975947 * max(0.0, 0.08670959 - Q.z_7) / 0.03486957   # -5.0%  z_7 < 0.08671
        - 0.04492693 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # -4.5%  sj3_dr_max < 0.3456
        + 0.03743064 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, 0.0005116989 - Q.e3) / 7.903391e-06   # +3.7%  z_7 < 0.06473 and e3 < 0.0005117
        + 0.03667812 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # +3.7%  tau1 < 0.1028
        - 0.03406142 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.006756161 - Q.girth2_top3) / 0.0004453888   # -3.4%  log_sum_pt > 6.573 and girth2_top3 < 0.006756
        + 0.03094021 * max(0.0, Q.log_sum_pt - 6.572938) / 0.08631512   # +3.1%  log_sum_pt > 6.573
        - 0.028002 * max(0.0, 0.04737278 - Q.z_6) / 0.004344387   # -2.8%  z_6 < 0.04737
        + 0.02721176 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +2.7%  lam2 < 0.001131
        + 0.02126317 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +2.1%  tau1 < 0.07284
        - 0.01890103 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.dr_0 - 0.005517012) / 0.0001856778   # -1.9%  lam1 < 0.012 and dr_0 > 0.005517
        + 0.01864364 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +1.9%  lam1 < 0.002464
        + 0.0173211 * max(0.0, 0.04737278 - Q.z_6) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0003931974   # +1.7%  z_6 < 0.04737 and z_dr_0p2_0p4 < 0.101
        + 0.01726696 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, 0.007929074 - Q.girth2_top3) / 9.283431e-05   # +1.7%  z_7 < 0.06473 and girth2_top3 < 0.007929
        - 0.01662522 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # -1.7%  max_dr < 0.1118
        + 0.01634371 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.1206357 - Q.dr_max_012) / 0.007392713   # +1.6%  log_sum_pt > 6.573 and dr_max_012 < 0.1206
        - 0.01608689 * max(0.0, Q.z_7 - 0.04939969) / 0.01023002   # -1.6%  z_7 > 0.0494
        - 0.01518255 * max(0.0, 1.340118e-05 - Q.e3) / 5.081834e-06   # -1.5%  e3 < 1.34e-05
        + 0.01488698 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 1.340118e-05 - Q.e3) / 1.490664e-06   # +1.5%  log_sum_pt > 6.378 and e3 < 1.34e-05
        - 0.01387169 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 0.1550922 - Q.dr_max_012) / 0.0002929109   # -1.4%  lam1 < 0.005433 and dr_max_012 < 0.1551
        - 0.01361043 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -1.4%  lam1 < 0.008376
        - 0.01329231 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -1.3%  zdr_0 < 0.02118
        + 0.01272356 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +1.3%  sj2_dr < 0.1592
        + 0.01069096 * max(0.0, 0.169029 - Q.sj3_dr_max) / 0.04876366   # +1.1%  sj3_dr_max < 0.169
        - 0.01041654 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.1147987 - Q.dr_2) / 0.01592659   # -1.0%  log_sum_pt > 6.378 and dr_2 < 0.1148
        - 0.009798418 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.004007842 - Q.girth2_top2) / 0.0005390217   # -1.0%  log_sum_pt > 6.378 and girth2_top2 < 0.004008
        + 0.008433591 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +0.8%  pt_7 > 34.53
        - 0.007840795 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.00233503   # -0.8%  lam1 < 0.012 and n_pt_above_50 > 6
        + 0.007509793 * max(0.0, Q.pt_7 - 41.65625) / 1.850204   # +0.8%  pt_7 > 41.66
        - 0.006884334 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, 0.716559 - Q.D2_b2) / 0.004098393   # -0.7%  z_7 < 0.06473 and D2_b2 < 0.7166
        + 0.006018892 * max(0.0, Q.n_pt_above_50 - 6.0) * max(0.0, 8.147744e-05 - Q.e3) / 1.821055e-05   # +0.6%  n_pt_above_50 > 6 and e3 < 8.148e-05
        + 0.005876847 * max(0.0, Q.LHA - 0.2160559) / 0.05893777   # +0.6%  LHA > 0.2161
        + 0.005703018 * max(0.0, Q.sum_pt - 763.825) / 48.90223   # +0.6%  sum_pt > 763.8
        - 0.005670573 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # -0.6%  pt_7 > 34.53 and tau2 < 0.01713
        + 0.004450016 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # +0.4%  sum_pt_top5 > 840
        + 0.004355634 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # +0.4%  lam1 < 0.008376 and centroid_offset > 0.02077
        - 0.003241544 * max(0.0, Q.sj3_dr_min - 0.02913153) / 0.02609727   # -0.3%  sj3_dr_min > 0.02913
        + 0.003140714 * max(0.0, 0.1591713 - Q.sj2_dr) * max(0.0, 0.716559 - Q.D2_b2) / 0.002739304   # +0.3%  sj2_dr < 0.1592 and D2_b2 < 0.7166
        + 0.003040619 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 5.0 - Q.n_dr_0_0p05) / 8.652865   # +0.3%  pt_7 > 34.53 and n_dr_0_0p05 < 5
        - 0.002903378 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.0095303 - Q.girth2_top2) / 0.03873367   # -0.3%  sum_pt > 988.4 and girth2_top2 < 0.00953
        - 0.002723401 * max(0.0, Q.sum_pt_top5 - 839.9547) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 15.08352   # -0.3%  sum_pt_top5 > 840 and n_dr_0p2_0p4 < 2
        - 0.002086194 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.2%  pt_7 > 53.44
        - 0.002071705 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.2%  sum_pt > 988.4
        + 0.001593252 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.01265416 - Q.zdr_7) / 0.05068001   # +0.2%  sum_pt > 988.4 and zdr_7 < 0.01265
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 20.04;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.03794 * (0.3311503
        - 0.1659391 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # -16.6%  z_7 > 0.02321
        - 0.1158495 * max(0.0, Q.log_sum_pt - 6.080494) / 0.4685351   # -11.6%  log_sum_pt > 6.08
        + 0.1007388 * max(0.0, Q.pt_7 - 20.125) / 15.07003   # +10.1%  pt_7 > 20.12
        - 0.0754484 * max(0.0, Q.LHA - 0.111565) / 0.1352185   # -7.5%  LHA > 0.1116
        - 0.05200348 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -5.2%  pt_7 < 53.44
        - 0.0493649 * Q.pt_7 / 34.64819   # -4.9%  pt_7
        + 0.04913563 * max(0.0, Q.z_7 - 0.02807091) / 0.02541111   # +4.9%  z_7 > 0.02807
        + 0.03255464 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # +3.3%  girth < 0.1019
        + 0.03067186 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.2507612 - Q.max_dr) / 0.00062784   # +3.1%  lam1 < 0.00733 and max_dr < 0.2508
        + 0.03036346 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +3.0%  lam1 < 0.012
        - 0.02857223 * max(0.0, Q.log_sum_pt - 6.080494) * max(0.0, 8.836697e-05 - Q.mean_phi2) / 6.819601e-06   # -2.9%  log_sum_pt > 6.08 and mean_phi2 < 8.837e-05
        + 0.02807685 * max(0.0, 0.0586137 - Q.z_7) / 0.01231218   # +2.8%  z_7 < 0.05861
        + 0.02731057 * max(0.0, 8.836697e-05 - Q.mean_phi2) / 1.02303e-05   # +2.7%  mean_phi2 < 8.837e-05
        - 0.02313277 * max(0.0, 0.04447357 - Q.e2) / 0.02019334   # -2.3%  e2 < 0.04447
        + 0.02295273 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # +2.3%  log_sum_pt < 6.606
        + 0.015709 * max(0.0, 0.004032342 - Q.C2_b2) / 0.002883542   # +1.6%  C2_b2 < 0.004032
        - 0.01356784 * max(0.0, 0.0423228 - Q.C2) / 0.02012665   # -1.4%  C2 < 0.04232
        - 0.01354105 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 62.25 - Q.pt_6) / 0.07984241   # -1.4%  lam1 < 0.00733 and pt_6 < 62.25
        - 0.01306064 * max(0.0, 0.0586137 - Q.z_7) * max(0.0, 0.1057739 - Q.abseta_0) / 0.001054575   # -1.3%  z_7 < 0.05861 and abseta_0 < 0.1058
        + 0.01191565 * max(0.0, 641.7188 - Q.sum_pt) / 38.47522   # +1.2%  sum_pt < 641.7
        - 0.01111056 * max(0.0, 752.1 - Q.sum_pt_top5) / 179.9248   # -1.1%  sum_pt_top5 < 752.1
        + 0.01095768 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.1056549 - Q.absphi_0) / 0.0003841548   # +1.1%  log_sum_pt > 6.896 and absphi_0 < 0.1057
        - 0.01023459 * max(0.0, Q.z_7 - 0.02320757) * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.002539696   # -1.0%  z_7 > 0.02321 and sj3_dr_min < 0.1278
        - 0.009001505 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.1056549 - Q.absphi_0) / 0.419244   # -0.9%  sum_pt > 988.4 and absphi_0 < 0.1057
        + 0.0088326 * max(0.0, 6.701242 - Q.log_sum_pt) / 0.1935491   # +0.9%  log_sum_pt < 6.701
        - 0.008379756 * max(0.0, 0.0586137 - Q.z_7) * max(0.0, 0.07897949 - Q.absphi_0) / 0.0007310394   # -0.8%  z_7 < 0.05861 and absphi_0 < 0.07898
        - 0.008138997 * max(0.0, Q.log_sum_pt - 6.080494) * max(0.0, 0.0001947983 - Q.lam2) / 6.2639e-05   # -0.8%  log_sum_pt > 6.08 and lam2 < 0.0001948
        - 0.00596413 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, Q.D2_b2 - 0.01618699) / 0.1028513   # -0.6%  log_sum_pt < 6.606 and D2_b2 > 0.01619
        + 0.004382431 * max(0.0, 6.701242 - Q.log_sum_pt) * max(0.0, Q.D2_b2 - 0.380911) / 0.1008885   # +0.4%  log_sum_pt < 6.701 and D2_b2 > 0.3809
        + 0.0041391 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.4%  sum_pt > 988.4
        - 0.003623617 * max(0.0, 23.21641 - Q.pt_7) / 0.9212932   # -0.4%  pt_7 < 23.22
        + 0.003593185 * max(0.0, Q.sum_pt - 868.5094) * max(0.0, 4.721224 - Q.D2_b2) / 44.90046   # +0.4%  sum_pt > 868.5 and D2_b2 < 4.721
        - 0.002167632 * max(0.0, 6.701242 - Q.log_sum_pt) * max(0.0, 8.836697e-05 - Q.mean_phi2) / 5.122736e-07   # -0.2%  log_sum_pt < 6.701 and mean_phi2 < 8.837e-05
        - 0.002140029 * max(0.0, 0.005576073 - Q.centroid_offset) * max(0.0, 0.03607145 - Q.dr_7) / 9.086096e-06   # -0.2%  centroid_offset < 0.005576 and dr_7 < 0.03607
        - 0.002049979 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.2%  log_sum_pt > 6.896
        + 0.001663194 * max(0.0, Q.sum_pt - 937.0312) * max(0.0, 8.836697e-05 - Q.mean_phi2) / 0.0002643409   # +0.2%  sum_pt > 937 and mean_phi2 < 8.837e-05
        - 0.001656581 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 4.721224 - Q.D2_b2) / 0.009941678   # -0.2%  log_sum_pt > 6.896 and D2_b2 < 4.721
        + 0.001460309 * max(0.0, 324.5922 - Q.sum_pt_top5) / 1.968832   # +0.1%  sum_pt_top5 < 324.6
        - 0.0005950874 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.004071886   # -0.1%  log_sum_pt > 6.896 and n_pt_above_50 > 5
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 9.514;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.514151 * (0.05903212
        - 0.3228114 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # -32.3%  lam2 < 0.003408
        + 0.07796501 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +7.8%  sj2_dr > 0.1683
        - 0.07450361 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # -7.5%  girth > 0.1019
        + 0.05979672 * max(0.0, Q.girth - 0.06663269) / 0.01393206   # +6.0%  girth > 0.06663
        + 0.05774896 * max(0.0, Q.LHA - 0.2931906) / 0.02125042   # +5.8%  LHA > 0.2932
        + 0.04819964 * max(0.0, Q.sj3_dr_max - 0.213399) / 0.03061279   # +4.8%  sj3_dr_max > 0.2134
        - 0.0430443 * max(0.0, Q.LHA - 0.3255822) / 0.01245304   # -4.3%  LHA > 0.3256
        + 0.03391427 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.eccentricity - 0.9458207) / 0.0008867862   # +3.4%  sj2_dr > 0.1683 and eccentricity > 0.9458
        - 0.02927604 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.log_sum_pt - 6.080494) / 0.01252901   # -2.9%  sj2_dr > 0.1683 and log_sum_pt > 6.08
        + 0.02856674 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.09482124 - Q.C2) / 0.001379963   # +2.9%  sj2_dr > 0.1683 and C2 < 0.09482
        - 0.02842022 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -2.8%  lam1 > 0.012
        + 0.02753427 * max(0.0, Q.girth - 0.1245537) / 0.002632136   # +2.8%  girth > 0.1246
        + 0.02310124 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +2.3%  tau1 > 0.1136
        - 0.02079946 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.n_dr_0_0p05 - 2.0) / 0.04103234   # -2.1%  sj2_dr > 0.1683 and n_dr_0_0p05 > 2
        + 0.02001771 * max(0.0, Q.girth - 0.06663269) * max(0.0, Q.eccentricity - 0.7117266) / 0.0025938   # +2.0%  girth > 0.06663 and eccentricity > 0.7117
        + 0.01635044 * max(0.0, Q.sd_rg - 0.1876504) * max(0.0, 0.1515405 - Q.z_dr_0_0p05) / 0.002283966   # +1.6%  sd_rg > 0.1877 and z_dr_0_0p05 < 0.1515
        + 0.0155376 * max(0.0, 0.2654572 - Q.N2) / 0.07451069   # +1.6%  N2 < 0.2655
        - 0.01491719 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # -1.5%  sd_rg > 0.2788
        + 0.01169771 * max(0.0, -0.006779839 - Q.mean_eta) / 0.003173922   # +1.2%  mean_eta < -0.00678
        - 0.00997352 * max(0.0, Q.sj3_dr_max - 0.2623172) * max(0.0, Q.n_dr_0p05_0p1 - 4.0) / 0.004621037   # -1.0%  sj3_dr_max > 0.2623 and n_dr_0p05_0p1 > 4
        + 0.008911091 * max(0.0, Q.sj3_dr_max - 0.213399) * max(0.0, Q.n_dr_0p05_0p1 - 4.0) / 0.007898646   # +0.9%  sj3_dr_max > 0.2134 and n_dr_0p05_0p1 > 4
        - 0.007264412 * max(0.0, Q.girth - 0.06663269) * max(0.0, Q.n_pt_above_50 - 3.0) / 0.0246836   # -0.7%  girth > 0.06663 and n_pt_above_50 > 3
        + 0.004792713 * max(0.0, Q.sd_rg - 0.1876504) * max(0.0, 0.000537286 - Q.lam2) / 3.14287e-06   # +0.5%  sd_rg > 0.1877 and lam2 < 0.0005373
        - 0.004564222 * max(0.0, Q.max_dr - 0.2507612) / 0.004590338   # -0.5%  max_dr > 0.2508
        - 0.003691981 * max(0.0, -0.006779839 - Q.mean_eta) * max(0.0, -0.03967285 - Q.eta_0) / 9.048956e-05   # -0.4%  mean_eta < -0.00678 and eta_0 < -0.03967
        - 0.003584461 * max(0.0, Q.sj3_dr23 - 0.3661115) / 0.003532972   # -0.4%  sj3_dr23 > 0.3661
        - 0.003015031 * max(0.0, Q.max_dr - 0.2507612) * max(0.0, 0.6080732 - Q.z_dr_0_0p05) / 0.001567661   # -0.3%  max_dr > 0.2508 and z_dr_0_0p05 < 0.6081
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 36.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 36.59346 * (0.01801235
        + 0.1323163 * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.166579   # +13.2%  sj3_dr_min < 0.209
        + 0.1071844 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +10.7%  e3 < 0.0001869
        - 0.1061378 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -10.6%  lam1 < 0.006507
        - 0.1015111 * max(0.0, 0.2089872 - Q.sj3_dr_min) * max(0.0, 0.003408389 - Q.lam2) / 0.000543125   # -10.2%  sj3_dr_min < 0.209 and lam2 < 0.003408
        - 0.06798876 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -6.8%  lam1 < 0.008376
        + 0.04345138 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, Q.eccentricity - 0.7117266) / 0.0004354758   # +4.3%  lam1 < 0.006507 and eccentricity > 0.7117
        - 0.04082318 * max(0.0, Q.eccentricity - 0.7117266) / 0.1947312   # -4.1%  eccentricity > 0.7117
        + 0.04043365 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +4.0%  N2 < 0.2233
        + 0.03494511 * max(0.0, Q.log_sum_pt - 6.327379) / 0.2487223   # +3.5%  log_sum_pt > 6.327
        + 0.03454023 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +3.5%  sj3_dr_max > 0.169
        - 0.03410433 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -3.4%  lam2 < 0.0005373
        + 0.02836111 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # +2.8%  LHA < 0.3127
        - 0.02541941 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -2.5%  sj3_dr_max < 0.1426
        - 0.02002986 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -2.0%  lam2 < 0.001131
        - 0.01632836 * max(0.0, 0.04435703 - Q.M2) / 0.008629948   # -1.6%  M2 < 0.04436
        - 0.01333482 * max(0.0, 763.825 - Q.sum_pt) / 96.68366   # -1.3%  sum_pt < 763.8
        - 0.01248041 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.2%  sj3_dr_max > 0.2337
        - 0.01230625 * max(0.0, Q.sj2_dr - 0.2001708) / 0.02317349   # -1.2%  sj2_dr > 0.2002
        + 0.01208786 * max(0.0, 0.08082334 - Q.dr_0) * max(0.0, Q.centroid_offset - 0.009480685) / 0.000183564   # +1.2%  dr_0 < 0.08082 and centroid_offset > 0.009481
        - 0.01063193 * max(0.0, 0.01705377 - Q.zdr_0) / 0.006142531   # -1.1%  zdr_0 < 0.01705
        + 0.009793866 * max(0.0, Q.max_dr - 0.121681) * max(0.0, 0.009032972 - Q.C2_b2) / 0.0001556695   # +1.0%  max_dr > 0.1217 and C2_b2 < 0.009033
        - 0.009232715 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, Q.log_sum_pt - 6.267538) / 0.004076983   # -0.9%  sj3_dr_max > 0.2337 and log_sum_pt > 6.268
        - 0.008842382 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, 0.009799324 - Q.zdr_2) / 0.001337409   # -0.9%  log_sum_pt > 6.327 and zdr_2 < 0.009799
        - 0.007418386 * max(0.0, 25.57812 - Q.pt_7) / 1.327389   # -0.7%  pt_7 < 25.58
        - 0.007221355 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.2832687 - Q.sj2_zsoft) / 0.002686709   # -0.7%  N2 < 0.2233 and sj2_zsoft < 0.2833
        - 0.006751168 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, 7.998907 - Q.ptdr0_6) / 1.442602   # -0.7%  log_sum_pt > 6.327 and ptdr0_6 < 7.999
        + 0.005983837 * max(0.0, Q.max_dr - 0.121681) / 0.03562206   # +0.6%  max_dr > 0.1217
        - 0.005780012 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1626038 - Q.abseta_7) / 0.005313035   # -0.6%  N2 < 0.2233 and abseta_7 < 0.1626
        - 0.00571858 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.107953 - Q.M3) / 3.326739   # -0.6%  sum_pt < 763.8 and M3 < 0.108
        - 0.003575249 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7) / 0.8992052   # -0.4%  N2 < 0.2233 and pt_7 < 53.44
        - 0.003439682 * max(0.0, 0.1329373 - Q.LHA) / 0.007753684   # -0.3%  LHA < 0.1329
        + 0.003235176 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, Q.absphi_7 - 0.001937771) / 3.668163e-05   # +0.3%  lam2 < 0.001131 and absphi_7 > 0.001938
        - 0.002912529 * max(0.0, Q.sj2_dr - 0.2179769) * max(0.0, 0.5327104 - Q.D2_b2) / 0.003818074   # -0.3%  sj2_dr > 0.218 and D2_b2 < 0.5327
        - 0.002659532 * max(0.0, Q.max_dr - 0.121681) * max(0.0, 0.07861328 - Q.abseta_0) / 0.001374529   # -0.3%  max_dr > 0.1217 and abseta_0 < 0.07861
        - 0.002514888 * max(0.0, Q.C2 - 0.01867771) / 0.01385506   # -0.3%  C2 > 0.01868
        - 0.002409502 * max(0.0, Q.sj2_dr - 0.2179769) * max(0.0, Q.lam2 - 0.001130645) / 2.486958e-05   # -0.2%  sj2_dr > 0.218 and lam2 > 0.001131
        + 0.002247871 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.n_dr_0p1_0p2 - 2.0) / 0.0465202   # +0.2%  N2 < 0.2233 and n_dr_0p1_0p2 > 2
        + 0.002034155 * max(0.0, Q.eccentricity - 0.7117266) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.05767584   # +0.2%  eccentricity > 0.7117 and n_pt_above_50 > 6
        + 0.001972627 * max(0.0, Q.sj2_dr - 0.2179769) / 0.01834468   # +0.2%  sj2_dr > 0.218
        - 0.00170738 * max(0.0, 0.08082334 - Q.dr_0) * max(0.0, Q.centroid_offset - 0.02355416) / 4.26964e-05   # -0.2%  dr_0 < 0.08082 and centroid_offset > 0.02355
        + 0.001569254 * max(0.0, 25.57812 - Q.pt_7) * max(0.0, Q.eccentricity - 0.9458207) / 0.01996572   # +0.2%  pt_7 < 25.58 and eccentricity > 0.9458
        + 0.001338934 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, 0.05966518 - Q.sj2_zsoft) / 6.432356e-05   # +0.1%  sj3_dr_max > 0.2337 and sj2_zsoft < 0.05967
        + 0.001278295 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, Q.lam2 - 0.000537286) / 5.527649e-08   # +0.1%  lam1 < 0.00733 and lam2 > 0.0005373
        - 0.001048607 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, Q.dr0_7 - 0.09118326) / 1.143817e-05   # -0.1%  lam2 < 0.0005373 and dr0_7 > 0.09118
        - 0.0009644166 * max(0.0, Q.max_dr - 0.121681) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.004992694   # -0.1%  max_dr > 0.1217 and n_pt_above_50 > 6
        - 0.0009226071 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -0.1%  lam1 > 0.012
        - 0.0007265729 * max(0.0, 0.003206041 - Q.tau21_b2) / 4.532753e-05   # -0.1%  tau21_b2 < 0.003206
        - 0.000651835 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, Q.dr1_4 - 0.1883455) / 0.001466951   # -0.1%  log_sum_pt > 6.327 and dr1_4 > 0.1883
        - 0.0006177813 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.lam2 - 0.001130645) / 2.416387e-06   # -0.1%  N2 < 0.2233 and lam2 > 0.001131
        + 0.0005601099 * max(0.0, 0.2089872 - Q.sj3_dr_min) * max(0.0, -0.02594505 - Q.mean_phi) / 8.519257e-05   # +0.1%  sj3_dr_min < 0.209 and mean_phi < -0.02595
        - 0.0004545033 * max(0.0, Q.sj3_dr_max - 0.3456459) * max(0.0, Q.sj3_z1 - 0.7947645) / 7.825074e-05   # -0.0%  sj3_dr_max > 0.3456 and sj3_z1 > 0.7948
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 13.9;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.9029 * (0.1045912
        - 0.09110917 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -9.1%  sum_pt < 739.5
        + 0.07947485 * max(0.0, 579.875 - Q.sum_pt_top5) / 65.87113   # +7.9%  sum_pt_top5 < 579.9
        + 0.06909213 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +6.9%  LHA < 0.2161
        - 0.05462771 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # -5.5%  centroid_offset < 0.03776
        - 0.05081697 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # -5.1%  dr_0 < 0.06413
        + 0.04797094 * max(0.0, 0.03556091 - Q.e2) * max(0.0, 0.0003061234 - Q.lam2) / 3.480488e-06   # +4.8%  e2 < 0.03556 and lam2 < 0.0003061
        + 0.04538653 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +4.5%  e3 < 0.0001869
        - 0.04458648 * max(0.0, 0.03556091 - Q.e2) / 0.01371972   # -4.5%  e2 < 0.03556
        + 0.0407723 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +4.1%  lam1 < 0.002464
        - 0.03940471 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.09543973 - Q.z_6) / 0.001514041   # -3.9%  LHA < 0.2161 and z_6 < 0.09544
        + 0.03441003 * max(0.0, 0.002464291 - Q.lam1) * max(0.0, 0.01837778 - Q.centroid_offset) / 8.036566e-06   # +3.4%  lam1 < 0.002464 and centroid_offset < 0.01838
        - 0.03384539 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -3.4%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.03322727 * max(0.0, 0.06081235 - Q.z_6) * max(0.0, 0.0005116989 - Q.e3) / 4.735652e-06   # +3.3%  z_6 < 0.06081 and e3 < 0.0005117
        + 0.03197927 * max(0.0, 0.0008722282 - Q.lam1) / 0.0001882079   # +3.2%  lam1 < 0.0008722
        - 0.03094695 * max(0.0, 0.06081235 - Q.z_6) / 0.009654642   # -3.1%  z_6 < 0.06081
        - 0.02591918 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -2.6%  sj3_dr_max < 0.3012
        - 0.02375459 * max(0.0, 0.8233866 - Q.z_top5) / 0.02979882   # -2.4%  z_top5 < 0.8234
        + 0.0229639 * max(0.0, 0.03556091 - Q.e2) * max(0.0, 0.0753896 - Q.z_7) / 0.0004290247   # +2.3%  e2 < 0.03556 and z_7 < 0.07539
        + 0.01617604 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +1.6%  lam1 < 0.00484
        + 0.01561017 * max(0.0, 0.06081235 - Q.z_6) * max(0.0, 28.39396 - Q.pt1_dr01) / 0.2343338   # +1.6%  z_6 < 0.06081 and pt1_dr01 < 28.39
        + 0.01516901 * max(0.0, 0.02054282 - Q.girth) * max(0.0, Q.sum_pt_top5 - 716.8828) / 0.2273844   # +1.5%  girth < 0.02054 and sum_pt_top5 > 716.9
        - 0.01462024 * max(0.0, 0.0262518 - Q.tau1) / 0.005366404   # -1.5%  tau1 < 0.02625
        - 0.01286292 * max(0.0, Q.pt_7 - 37.15625) / 3.295689   # -1.3%  pt_7 > 37.16
        - 0.01186329 * max(0.0, 0.02054282 - Q.girth) / 0.002592388   # -1.2%  girth < 0.02054
        - 0.01137764 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # -1.1%  sj3_dr_max < 0.107
        + 0.01064171 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +1.1%  z_7 < 0.02807
        - 0.009554303 * max(0.0, Q.log_sum_pt - 6.804164) / 0.01257682   # -1.0%  log_sum_pt > 6.804
        - 0.008790033 * max(0.0, Q.log_sum_pt - 6.804164) * max(0.0, 5.334511e-05 - Q.e3) / 6.116454e-07   # -0.9%  log_sum_pt > 6.804 and e3 < 5.335e-05
        - 0.008130827 * max(0.0, 0.266913 - Q.LHA) / 0.05602629   # -0.8%  LHA < 0.2669
        + 0.008043378 * max(0.0, 7.12483e-05 - Q.lam1) / 3.380156e-06   # +0.8%  lam1 < 7.125e-05
        + 0.007375422 * max(0.0, 0.03629544 - Q.z_7) / 0.002921513   # +0.7%  z_7 < 0.0363
        - 0.005965963 * max(0.0, Q.n_pt_above_50 - 6.0) / 0.2884353   # -0.6%  n_pt_above_50 > 6
        + 0.005757328 * max(0.0, 0.03556091 - Q.e2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.001328529   # +0.6%  e2 < 0.03556 and planar_flow < 0.3221
        + 0.005414685 * max(0.0, 0.03556091 - Q.e2) * max(0.0, Q.zdr_0 - 0.004918231) / 3.97457e-05   # +0.5%  e2 < 0.03556 and zdr_0 > 0.004918
        - 0.00434662 * max(0.0, Q.z_top5_slots - 0.908903) / 0.002542271   # -0.4%  z_top5_slots > 0.9089
        - 0.003976925 * max(0.0, 20.125 - Q.pt_7) / 0.5468342   # -0.4%  pt_7 < 20.12
        + 0.003750047 * max(0.0, 0.02886576 - Q.z_6) / 0.0008381782   # +0.4%  z_6 < 0.02887
        - 0.003588103 * max(0.0, 0.02054282 - Q.girth) * max(0.0, 0.09607851 - Q.M3) / 4.66701e-05   # -0.4%  girth < 0.02054 and M3 < 0.09608
        - 0.00346128 * max(0.0, 7.12483e-05 - Q.lam1) * max(0.0, 73.75 - Q.pt_5) / 8.367553e-05   # -0.3%  lam1 < 7.125e-05 and pt_5 < 73.75
        + 0.003340835 * max(0.0, Q.log_sum_pt - 6.804164) * max(0.0, 0.01251955 - Q.mean_phi) / 0.0001601797   # +0.3%  log_sum_pt > 6.804 and mean_phi < 0.01252
        - 0.003072318 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.3%  log_sum_pt > 6.896
        + 0.002558725 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # +0.3%  pt_5 < 24.58
        + 0.002159113 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # +0.2%  z_6 < 0.0216
        + 0.001267722 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, Q.mean_phi - -0.01275329) / 5.207735e-05   # +0.1%  log_sum_pt > 6.896 and mean_phi > -0.01275
        - 0.0008369827 * max(0.0, 0.02886576 - Q.z_6) * max(0.0, Q.phi_0 - 0.008850098) / 1.965886e-06   # -0.1%  z_6 < 0.02887 and phi_0 > 0.00885
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 24.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.67163 * (0.01352274
        - 0.08586595 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # -8.6%  max_dr < 0.1773
        - 0.08448977 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -8.4%  lam2 < 0.001131
        - 0.07555739 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -7.6%  girth > 0.08724
        - 0.07187223 * max(0.0, Q.sj3_dr_max - 0.03464708) / 0.1367546   # -7.2%  sj3_dr_max > 0.03465
        + 0.06139521 * max(0.0, Q.LHA - 0.251526) / 0.03924325   # +6.1%  LHA > 0.2515
        + 0.06027507 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # +6.0%  C2 < 0.03579
        + 0.04167432 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +4.2%  sj3_dr_max > 0.179
        + 0.04140857 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +4.1%  lam2 < 0.003408
        - 0.02932006 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -2.9%  centroid_offset < 0.01838
        + 0.02837256 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, Q.sj2_zsoft - 0.05966518) / 0.005829511   # +2.8%  sj2_dr > 0.1592 and sj2_zsoft > 0.05967
        - 0.02619191 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -2.6%  pt_6 < 39.75
        - 0.02437653 * max(0.0, Q.psi_0p2 - 0.8990266) / 0.08559522   # -2.4%  psi_0p2 > 0.899
        - 0.02434907 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -2.4%  sj2_dr > 0.1592
        + 0.02350262 * max(0.0, 39.75 - Q.pt_6) * max(0.0, 711.875 - Q.sum_pt_top3) / 684.1746   # +2.4%  pt_6 < 39.75 and sum_pt_top3 < 711.9
        + 0.02348896 * max(0.0, Q.LHA - 0.3255822) / 0.01245304   # +2.3%  LHA > 0.3256
        + 0.02340061 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +2.3%  sj2_dr > 0.1683
        + 0.02245876 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.001460719   # +2.2%  centroid_offset > 0.008092 and sj3_dr_min < 0.209
        + 0.0152635 * max(0.0, Q.sj3_dr_max - 0.1789613) * max(0.0, 1.59265 - Q.D2_b2) / 0.03741161   # +1.5%  sj3_dr_max > 0.179 and D2_b2 < 1.593
        + 0.0147498 * max(0.0, 33.21875 - Q.pt_7) / 3.73296   # +1.5%  pt_7 < 33.22
        - 0.01404138 * max(0.0, 5.794419e-05 - Q.lam2) / 1.900658e-05   # -1.4%  lam2 < 5.794e-05
        - 0.01357485 * max(0.0, 0.03978449 - Q.z_7) / 0.003874427   # -1.4%  z_7 < 0.03978
        + 0.01298592 * max(0.0, 0.04355037 - Q.z_6) / 0.003302426   # +1.3%  z_6 < 0.04355
        + 0.01152524 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.01426135 - Q.mean_phi2) / 9.496308e-05   # +1.2%  centroid_offset > 0.008092 and mean_phi2 < 0.01426
        + 0.01125263 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_dr_0p05_0p1 - 0.0) / 0.01198963   # +1.1%  centroid_offset < 0.01838 and n_dr_0p05_0p1 > 0
        + 0.01068312 * max(0.0, Q.sum_pt - 840.0195) / 24.43934   # +1.1%  sum_pt > 840
        + 0.009757361 * max(0.0, 0.08082334 - Q.dr_0) / 0.0377444   # +1.0%  dr_0 < 0.08082
        - 0.00923555 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # -0.9%  girth > 0.1019
        - 0.009091686 * max(0.0, 39.75 - Q.pt_6) * max(0.0, 4.721224 - Q.D2_b2) / 13.45505   # -0.9%  pt_6 < 39.75 and D2_b2 < 4.721
        - 0.008094102 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.8%  z_6 < 0.0216
        - 0.0080932 * max(0.0, Q.sj3_dr_max - 0.03464708) * max(0.0, 0.01488897 - Q.dr_min_012) / 0.001063317   # -0.8%  sj3_dr_max > 0.03465 and dr_min_012 < 0.01489
        - 0.007848006 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # -0.8%  tau1 > 0.05357
        + 0.006636286 * max(0.0, 19.46875 - Q.pt_6) / 0.2515892   # +0.7%  pt_6 < 19.47
        + 0.00650831 * max(0.0, 7.330536e-06 - Q.e3) / 2.382852e-06   # +0.7%  e3 < 7.331e-06
        + 0.006478749 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.09482124 - Q.C2) / 0.001613212   # +0.6%  sj2_dr > 0.1592 and C2 < 0.09482
        + 0.006042016 * max(0.0, Q.girth - 0.1484084) / 0.0008957407   # +0.6%  girth > 0.1484
        - 0.005884188 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.1115136 - Q.planar_flow) / 0.0002206557   # -0.6%  centroid_offset < 0.01838 and planar_flow < 0.1115
        + 0.005383946 * max(0.0, 31.90625 - Q.pt_6) / 1.826854   # +0.5%  pt_6 < 31.91
        + 0.005327723 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 0.002094451   # +0.5%  lam2 < 0.003408 and n_dr_0p1_0p2 > 1
        - 0.00506792 * max(0.0, Q.centroid_offset - 0.02685622) / 0.003254849   # -0.5%  centroid_offset > 0.02686
        - 0.004486164 * max(0.0, Q.tau1 - 0.1748444) / 0.001246453   # -0.4%  tau1 > 0.1748
        + 0.004464327 * max(0.0, 6.423044 - Q.log_sum_pt) / 0.05674989   # +0.4%  log_sum_pt < 6.423
        - 0.004098842 * max(0.0, 39.75 - Q.pt_6) * max(0.0, 0.7861046 - Q.z_top3_slots) / 0.3107099   # -0.4%  pt_6 < 39.75 and z_top3_slots < 0.7861
        + 0.004037771 * max(0.0, Q.centroid_offset - 0.02076709) / 0.004751537   # +0.4%  centroid_offset > 0.02077
        - 0.003616185 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 5.0 - Q.n_dr_0_0p05) / 0.01797539   # -0.4%  centroid_offset > 0.02077 and n_dr_0_0p05 < 5
        + 0.003527724 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.N2 - 0.198361) / 0.0004989479   # +0.4%  centroid_offset > 0.008092 and N2 > 0.1984
        - 0.003111709 * max(0.0, Q.girth - 0.08723651) * max(0.0, 0.380911 - Q.D2_b2) / 0.001021187   # -0.3%  girth > 0.08724 and D2_b2 < 0.3809
        + 0.002979322 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, Q.n_dr_0p1_0p2 - 0.0) / 0.01432845   # +0.3%  centroid_offset > 0.02077 and n_dr_0p1_0p2 > 0
        - 0.002505009 * max(0.0, Q.sum_pt - 840.0195) * max(0.0, Q.D2_b2 - 0.716559) / 84.38587   # -0.3%  sum_pt > 840 and D2_b2 > 0.7166
        + 0.002387525 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +0.2%  centroid_offset > 0.008092
        + 0.002194972 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, Q.pt_2 - 69.875) / 0.03159951   # +0.2%  centroid_offset > 0.02686 and pt_2 > 69.88
        - 0.001956828 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.D3 - 0.8272948) / 0.005997404   # -0.2%  centroid_offset > 0.008092 and D3 > 0.8273
        + 0.001933418 * max(0.0, 24.42188 - Q.pt_6) / 0.6046572   # +0.2%  pt_6 < 24.42
        + 0.001748396 * max(0.0, 6.423044 - Q.log_sum_pt) * max(0.0, 38.53125 - Q.pt_7) / 0.3227003   # +0.2%  log_sum_pt < 6.423 and pt_7 < 38.53
        + 0.001634854 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, 0.01498227 - Q.zdr_2) / 1.069935e-05   # +0.2%  centroid_offset > 0.02686 and zdr_2 < 0.01498
        - 0.001340951 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, Q.mean_eta - 0.02644207) / 7.861676e-05   # -0.1%  sj2_dr > 0.1592 and mean_eta > 0.02644
        + 0.001292363 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, Q.C3 - 0.04165477) / 2.138046e-06   # +0.1%  lam2 < 0.001131 and C3 > 0.04165
        - 0.001158582 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, 0.380911 - Q.D2_b2) / 0.0003886793   # -0.1%  centroid_offset > 0.02686 and D2_b2 < 0.3809
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 39.77;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 39.76762 * (0.05570303
        - 0.06115593 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # -6.1%  lam1 > 0.01643
        - 0.05888346 * max(0.0, Q.LHA - 0.3467135) / 0.008898884   # -5.9%  LHA > 0.3467
        - 0.05671785 * max(0.0, 0.4235925 - Q.LHA) / 0.1821673   # -5.7%  LHA < 0.4236
        + 0.05421964 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # +5.4%  girth > 0.1019
        - 0.04966088 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -5.0%  girth > 0.08724
        + 0.04597251 * max(0.0, 0.08865369 - Q.tau1) / 0.03771145   # +4.6%  tau1 < 0.08865
        + 0.04596306 * max(0.0, Q.girth - 0.0479157) * max(0.0, 2.357246 - Q.D2) / 0.03555018   # +4.6%  girth > 0.04792 and D2 < 2.357
        + 0.04284475 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +4.3%  lam1 < 0.008376
        - 0.04073304 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # -4.1%  lam1 < 0.00484
        + 0.03933406 * max(0.0, Q.lam1 - 0.002464291) / 0.004201699   # +3.9%  lam1 > 0.002464
        + 0.03667044 * max(0.0, 0.1594012 - Q.sj3_dr_max) / 0.04422261   # +3.7%  sj3_dr_max < 0.1594
        + 0.03087457 * max(0.0, Q.lam1 - 0.01643375) * max(0.0, 3.885568 - Q.D2) / 0.002114372   # +3.1%  lam1 > 0.01643 and D2 < 3.886
        - 0.03062047 * max(0.0, Q.girth - 0.0479157) / 0.02294968   # -3.1%  girth > 0.04792
        + 0.02327792 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # +2.3%  tau1 > 0.1028
        + 0.02318635 * max(0.0, Q.lam1 - 0.002464291) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0003749324   # +2.3%  lam1 > 0.002464 and planar_flow < 0.195
        - 0.02257635 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # -2.3%  lam1 > 0.00733
        - 0.02017325 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.lam1 - 0.008375572) / 0.0001274849   # -2.0%  planar_flow < 0.195 and lam1 > 0.008376
        - 0.01969017 * max(0.0, Q.lam1 - 0.002464291) * max(0.0, Q.D2 - 0.2568137) / 0.002599874   # -2.0%  lam1 > 0.002464 and D2 > 0.2568
        - 0.01792553 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # -1.8%  lam1 < 0.003377
        - 0.0167186 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -1.7%  e3 < 8.148e-05
        - 0.0154939 * max(0.0, Q.girth - 0.08065885) * max(0.0, 2.843757 - Q.D2) / 0.01853389   # -1.5%  girth > 0.08066 and D2 < 2.844
        + 0.01540425 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # +1.5%  lam1 > 0.012
        - 0.01382573 * max(0.0, 0.02685622 - Q.centroid_offset) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.01150938   # -1.4%  centroid_offset < 0.02686 and n_dr_0p2_0p4 < 1
        - 0.01356006 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # -1.4%  sj3_dr_max < 0.2134
        - 0.01335763 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # -1.3%  lam2 < 0.003408
        - 0.01262215 * max(0.0, 0.0931108 - Q.max_dr) / 0.02003526   # -1.3%  max_dr < 0.09311
        + 0.0125616 * max(0.0, Q.LHA - 0.1329373) / 0.1175814   # +1.3%  LHA > 0.1329
        + 0.01138197 * max(0.0, Q.sj3_dr_max - 0.07264452) / 0.1084421   # +1.1%  sj3_dr_max > 0.07264
        + 0.01094095 * max(0.0, Q.N2 - 0.1333619) / 0.1019708   # +1.1%  N2 > 0.1334
        + 0.01070678 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 90.625 - Q.pt_4) / 0.08206501   # +1.1%  lam1 > 0.00733 and pt_4 < 90.62
        + 0.009893129 * max(0.0, Q.tau1 - 0.06345984) / 0.02203215   # +1.0%  tau1 > 0.06346
        - 0.009831689 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 2.843757 - Q.D2) / 0.004089948   # -1.0%  lam1 > 0.00733 and D2 < 2.844
        - 0.009093536 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 0.2507612 - Q.max_dr) / 0.008139731   # -0.9%  planar_flow < 0.195 and max_dr < 0.2508
        + 0.008837685 * max(0.0, 0.08050702 - Q.max_dr) / 0.01536431   # +0.9%  max_dr < 0.08051
        + 0.008822895 * max(0.0, 0.04990367 - Q.centroid_offset) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.06026392   # +0.9%  centroid_offset < 0.0499 and n_dr_0p2_0p4 < 2
        + 0.008443358 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.46415) / 0.01159499   # +0.8%  planar_flow < 0.195 and log_sum_pt > 6.464
        - 0.008199375 * max(0.0, 0.03360421 - Q.girth) / 0.006496997   # -0.8%  girth < 0.0336
        + 0.008198734 * max(0.0, 1.050302e-05 - Q.e3) / 3.717236e-06   # +0.8%  e3 < 1.05e-05
        + 0.007100373 * max(0.0, 0.02685622 - Q.centroid_offset) / 0.01292674   # +0.7%  centroid_offset < 0.02686
        - 0.007035339 * max(0.0, 1.432482 - Q.D2) / 0.4017356   # -0.7%  D2 < 1.432
        - 0.006881123 * max(0.0, 0.06108601 - Q.girth) / 0.01868685   # -0.7%  girth < 0.06109
        - 0.006880936 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # -0.7%  lam2 < 0.0003061
        - 0.005106303 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 43.5 - Q.pt_7) / 0.0006101229   # -0.5%  e3 < 8.148e-05 and pt_7 < 43.5
        + 0.004684505 * max(0.0, 0.04990367 - Q.centroid_offset) / 0.03352573   # +0.5%  centroid_offset < 0.0499
        + 0.004638475 * max(0.0, 0.1950135 - Q.planar_flow) / 0.07573236   # +0.5%  planar_flow < 0.195
        - 0.00400277 * max(0.0, 0.04656688 - Q.max_dr) / 0.005138022   # -0.4%  max_dr < 0.04657
        - 0.003869312 * max(0.0, Q.lam1 - 0.002464291) * max(0.0, 0.1879486 - Q.sj3_dr_max) / 1.34726e-05   # -0.4%  lam1 > 0.002464 and sj3_dr_max < 0.1879
        - 0.003527572 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 56.53125 - Q.pt_6) / 1.191827   # -0.4%  planar_flow < 0.195 and pt_6 < 56.53
        + 0.002613038 * max(0.0, 4.0 - Q.n_dr_0_0p05) / 1.462824   # +0.3%  n_dr_0_0p05 < 4
        - 0.001695321 * max(0.0, 0.04990367 - Q.centroid_offset) * max(0.0, Q.tau4 - 0.002042374) / 4.300788e-05   # -0.2%  centroid_offset < 0.0499 and tau4 > 0.002042
        + 0.001289368 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, 1.432482 - Q.D2) / 0.003360448   # +0.1%  sj3_dr_max < 0.1426 and D2 < 1.432
        - 0.001197328 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 29.04219 - Q.pt_7) / 0.1265631   # -0.1%  planar_flow < 0.195 and pt_7 < 29.04
        + 0.0007585269 * max(0.0, Q.girth - 0.08065885) * max(0.0, 0.09384951 - Q.sj2_zsoft) / 5.444145e-06   # +0.1%  girth > 0.08066 and sj2_zsoft < 0.09385
        + 0.0003454601 * max(0.0, Q.girth - 0.08065885) * max(0.0, Q.D2_b2 - 1.911889) / 0.0002280963   # +0.0%  girth > 0.08066 and D2_b2 > 1.912
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 19.82;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.82428 * (-0.03994558
        - 0.1229794 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # -12.3%  sj3_dr_max < 0.107
        + 0.07865139 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, 0.0153634 - Q.girth2_top3) / 0.0003596312   # +7.9%  sj3_dr_max < 0.107 and girth2_top3 < 0.01536
        - 0.06914244 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, 0.0153634 - Q.girth2_top3) / 0.0005547224   # -6.9%  sj3_dr_max < 0.1426 and girth2_top3 < 0.01536
        + 0.05436434 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +5.4%  lam2 < 0.0001948
        + 0.049045 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +4.9%  sj3_dr_max < 0.1986
        - 0.04763377 * max(0.0, 0.1767241 - Q.LHA) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.00104409   # -4.8%  LHA < 0.1767 and z_dr_0p2_0p4 < 0.05644
        + 0.04753185 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, 0.001864148 - Q.girth2_top2) / 3.719196e-05   # +4.8%  sj3_dr_max < 0.107 and girth2_top2 < 0.001864
        - 0.03863333 * max(0.0, 0.1594012 - Q.sj3_dr_max) * max(0.0, 0.001864148 - Q.girth2_top2) / 6.497216e-05   # -3.9%  sj3_dr_max < 0.1594 and girth2_top2 < 0.001864
        + 0.0378836 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +3.8%  sj3_dr_max < 0.1426
        - 0.03680061 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -3.7%  centroid_offset > 0.00679
        - 0.03664951 * max(0.0, 0.01713288 - Q.tau2) / 0.008065871   # -3.7%  tau2 < 0.01713
        + 0.03431466 * max(0.0, 0.0262518 - Q.tau1) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0001164763   # +3.4%  tau1 < 0.02625 and centroid_offset < 0.03117
        + 0.03107249 * max(0.0, 0.1767241 - Q.LHA) / 0.01861764   # +3.1%  LHA < 0.1767
        + 0.02541684 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.0001209282   # +2.5%  lam1 < 0.005433 and z_dr_0p2_0p4 < 0.05644
        - 0.02356343 * max(0.0, 0.0262518 - Q.tau1) / 0.005366404   # -2.4%  tau1 < 0.02625
        - 0.0191959 * max(0.0, 0.03360421 - Q.girth) / 0.006496997   # -1.9%  girth < 0.0336
        + 0.01616749 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 0.03117077 - Q.centroid_offset) / 4.498827e-05   # +1.6%  lam1 < 0.005433 and centroid_offset < 0.03117
        + 0.01537599 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # +1.5%  lam1 < 0.003377
        + 0.01535535 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.02355416 - Q.centroid_offset) / 4.507403e-05   # +1.5%  lam1 < 0.00733 and centroid_offset < 0.02355
        + 0.01380932 * max(0.0, 0.03360421 - Q.girth) * max(0.0, Q.pt_7 - 15.55391) / 0.1111381   # +1.4%  girth < 0.0336 and pt_7 > 15.55
        - 0.01379468 * max(0.0, 0.03360421 - Q.girth) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0001637288   # -1.4%  girth < 0.0336 and centroid_offset < 0.03117
        - 0.01303493 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.001056655 - Q.girth2_top2) / 2.034239e-06   # -1.3%  lam1 < 0.00733 and girth2_top2 < 0.001057
        - 0.01231453 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.01627885) / 7.113e-05   # -1.2%  sj3_dr_max < 0.107 and centroid_offset > 0.01628
        - 0.0119695 * max(0.0, 0.251526 - Q.LHA) * max(0.0, Q.z_7 - 0.01685855) / 0.001243859   # -1.2%  LHA < 0.2515 and z_7 > 0.01686
        - 0.009730476 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -1.0%  log_sum_pt > 6.701
        + 0.008584298 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 5.351077 - Q.pt1_dr01) / 0.008800757   # +0.9%  lam1 < 0.005433 and pt1_dr01 < 5.351
        - 0.008238744 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 38.53125 - Q.pt_7) / 0.01790589   # -0.8%  lam1 < 0.005433 and pt_7 < 38.53
        + 0.008210854 * max(0.0, 0.1594012 - Q.sj3_dr_max) / 0.04422261   # +0.8%  sj3_dr_max < 0.1594
        + 0.007485349 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 0.2752975   # +0.7%  sj3_dr_max < 0.1986 and n_dr_0p05_0p1 < 5
        + 0.007469669 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.00327978 - Q.girth2_top5) / 7.935857e-05   # +0.7%  log_sum_pt > 6.701 and girth2_top5 < 0.00328
        + 0.007352885 * max(0.0, 0.04355037 - Q.z_6) / 0.003302426   # +0.7%  z_6 < 0.04355
        - 0.00670977 * max(0.0, 0.251526 - Q.LHA) / 0.04800428   # -0.7%  LHA < 0.2515
        - 0.006615932 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01002369 - Q.girth2_top3) / 0.0003076784   # -0.7%  log_sum_pt > 6.701 and girth2_top3 < 0.01002
        + 0.006292587 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.002915531 - Q.girth2_top3) / 7.495703e-05   # +0.6%  log_sum_pt > 6.701 and girth2_top3 < 0.002916
        + 0.00621366 * max(0.0, 0.03360421 - Q.girth) * max(0.0, Q.psi_0p1 - 0.9761279) / 0.0001475117   # +0.6%  girth < 0.0336 and psi_0p1 > 0.9761
        - 0.006099732 * max(0.0, 0.02054282 - Q.girth) / 0.002592388   # -0.6%  girth < 0.02054
        - 0.006065467 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.00633598 - Q.girth2_top5) / 0.0001712067   # -0.6%  log_sum_pt > 6.701 and girth2_top5 < 0.006336
        + 0.006012811 * max(0.0, 0.1594012 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.003343241) / 0.00037024   # +0.6%  sj3_dr_max < 0.1594 and centroid_offset > 0.003343
        + 0.005400113 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.5925897 - Q.planar_flow) / 0.0009146444   # +0.5%  lam1 < 0.00733 and planar_flow < 0.5926
        + 0.005297658 * max(0.0, 0.03459477 - Q.tau1) * max(0.0, 0.001056655 - Q.girth2_top2) / 6.77448e-06   # +0.5%  tau1 < 0.03459 and girth2_top2 < 0.001057
        + 0.004974807 * max(0.0, 0.0002758826 - Q.lam1) / 3.586469e-05   # +0.5%  lam1 < 0.0002759
        + 0.003873388 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +0.4%  lam1 < 0.00733
        - 0.003225032 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 6.502799 - Q.log_sum_pt) / 0.0001002502   # -0.3%  lam1 < 0.005433 and log_sum_pt < 6.503
        - 0.002737585 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, Q.centroid_offset - 0.01627885) / 1.955234e-06   # -0.3%  lam1 < 0.003377 and centroid_offset > 0.01628
        - 0.002699327 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, Q.girth2_top5 - 0.002270363) / 2.386126e-05   # -0.3%  sj3_dr_max < 0.1986 and girth2_top5 > 0.00227
        - 0.002484535 * max(0.0, 0.019014 - Q.tau1) * max(0.0, Q.pt_7 - 20.125) / 0.03986084   # -0.2%  tau1 < 0.01901 and pt_7 > 20.12
        - 0.002143859 * max(0.0, 0.01357518 - Q.tau1) * max(0.0, 0.001056655 - Q.girth2_top2) / 1.347821e-06   # -0.2%  tau1 < 0.01358 and girth2_top2 < 0.001057
        - 0.0009556895 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.centroid_offset - 0.006789738) / 9.294835e-05   # -0.1%  log_sum_pt > 6.701 and centroid_offset > 0.00679
        - 0.0004253723 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, 22.38578 - Q.pt1_dr01) / 0.780888   # -0.0%  sj3_dr_max < 0.1426 and pt1_dr01 < 22.39
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 24.66;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.66304 * (-0.2450863
        + 0.1446205 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 0.001130645 - Q.lam2) / 2.659012e-06   # +14.5%  lam1 < 0.005954 and lam2 < 0.001131
        + 0.09101373 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +9.1%  sum_pt < 988.4
        + 0.05101912 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0002544934   # +5.1%  tau1 < 0.0437 and centroid_offset < 0.03117
        - 0.04100042 * max(0.0, 0.04081947 - Q.girth) / 0.009176315   # -4.1%  girth < 0.04082
        - 0.03993808 * max(0.0, 813.4156 - Q.sum_pt) / 129.0758   # -4.0%  sum_pt < 813.4
        + 0.03920928 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +3.9%  sj3_dr_max < 0.2134
        - 0.03845263 * max(0.0, 0.05464922 - Q.girth) / 0.01532978   # -3.8%  girth < 0.05465
        + 0.03789462 * max(0.0, 0.03117077 - Q.centroid_offset) / 0.01649145   # +3.8%  centroid_offset < 0.03117
        + 0.03771969 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # +3.8%  lam1 < 0.003377
        + 0.03664151 * max(0.0, 0.01627885 - Q.centroid_offset) * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.0005664198   # +3.7%  centroid_offset < 0.01628 and sj3_dr_min < 0.1278
        - 0.03624467 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -3.6%  sj3_dr_max < 0.1426
        - 0.03393276 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 0.08543881 - Q.sj3_dr_min) / 0.0001709388   # -3.4%  lam1 < 0.005954 and sj3_dr_min < 0.08544
        + 0.03337776 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +3.3%  lam1 < 0.005954
        + 0.03004608 * max(0.0, 813.4156 - Q.sum_pt) * max(0.0, 3.0374e-08 - Q.e4) / 2.845153e-06   # +3.0%  sum_pt < 813.4 and e4 < 3.037e-08
        + 0.02632423 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # +2.6%  C2_b2 < 0.009033
        - 0.01993512 * max(0.0, 1.960701e-05 - Q.e3) / 8.488392e-06   # -2.0%  e3 < 1.961e-05
        + 0.01755682 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +1.8%  lam2 < 0.0001948
        + 0.01688498 * max(0.0, 0.01627885 - Q.centroid_offset) / 0.005402048   # +1.7%  centroid_offset < 0.01628
        + 0.01687499 * max(0.0, 0.02355416 - Q.centroid_offset) / 0.0103615   # +1.7%  centroid_offset < 0.02355
        - 0.01628912 * max(0.0, 988.4078 - Q.sum_pt) * max(0.0, 0.01627885 - Q.centroid_offset) / 1.063436   # -1.6%  sum_pt < 988.4 and centroid_offset < 0.01628
        - 0.01613026 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -1.6%  sj2_dr < 0.1295
        + 0.01541143 * max(0.0, 6.638339 - Q.log_sum_pt) / 0.1524936   # +1.5%  log_sum_pt < 6.638
        - 0.0152828 * max(0.0, 0.02689598 - Q.girth) / 0.004330137   # -1.5%  girth < 0.0269
        + 0.01483976 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, 0.00733008 - Q.lam1) / 4.507403e-05   # +1.5%  centroid_offset < 0.02355 and lam1 < 0.00733
        - 0.01385337 * max(0.0, 0.5925897 - Q.planar_flow) / 0.3505892   # -1.4%  planar_flow < 0.5926
        + 0.01139883 * max(0.0, Q.lam2 - 0.000537286) / 0.0003825703   # +1.1%  lam2 > 0.0005373
        + 0.01045219 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +1.0%  e3 > 0.0001869
        + 0.009144665 * max(0.0, 0.1027585 - Q.max_dr) / 0.02411847   # +0.9%  max_dr < 0.1028
        + 0.009080312 * max(0.0, 0.5925897 - Q.planar_flow) * max(0.0, 4.721224 - Q.D2_b2) / 1.452457   # +0.9%  planar_flow < 0.5926 and D2_b2 < 4.721
        + 0.006662252 * max(0.0, 0.1778793 - Q.sj2_dr) / 0.05866968   # +0.7%  sj2_dr < 0.1779
        - 0.006431104 * max(0.0, 0.01627885 - Q.centroid_offset) * max(0.0, 0.1830092 - Q.D2_b2) / 0.0002309693   # -0.6%  centroid_offset < 0.01628 and D2_b2 < 0.183
        + 0.005426485 * max(0.0, 988.4078 - Q.sum_pt) * max(0.0, 0.05753583 - Q.z_5) / 0.5190201   # +0.5%  sum_pt < 988.4 and z_5 < 0.05754
        - 0.005001127 * max(0.0, 0.0285317 - Q.e2) / 0.009532881   # -0.5%  e2 < 0.02853
        + 0.004075005 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # +0.4%  lam1 > 0.01643
        - 0.004049092 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, -0.006779839 - Q.mean_eta) / 4.043269e-06   # -0.4%  lam1 < 0.005954 and mean_eta < -0.00678
        - 0.003938181 * max(0.0, 0.01627885 - Q.centroid_offset) * max(0.0, 1.808379 - Q.N3) / 0.00218053   # -0.4%  centroid_offset < 0.01628 and N3 < 1.808
        - 0.003846052 * max(0.0, 0.02227783 - Q.absphi_1) / 0.006894148   # -0.4%  absphi_1 < 0.02228
        + 0.003540202 * max(0.0, 0.0001413912 - Q.lam1) / 1.223988e-05   # +0.4%  lam1 < 0.0001414
        - 0.003535169 * max(0.0, 0.05464922 - Q.girth) * max(0.0, 0.0006973656 - Q.mean_phi) / 4.723126e-05   # -0.4%  girth < 0.05465 and mean_phi < 0.0006974
        - 0.003304195 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.mean_eta - 0.009367547) / 3.008598e-06   # -0.3%  lam1 < 0.005954 and mean_eta > 0.009368
        + 0.003170178 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.3%  n_dr_0p2_0p4 > 1
        - 0.003150394 * max(0.0, Q.e3 - 0.0005116989) / 1.444524e-05   # -0.3%  e3 > 0.0005117
        - 0.003111985 * max(0.0, Q.lam2 - 0.000537286) * max(0.0, 0.9979797 - Q.eccentricity) / 0.000146617   # -0.3%  lam2 > 0.0005373 and eccentricity < 0.998
        - 0.002945438 * max(0.0, 813.4156 - Q.sum_pt) * max(0.0, Q.absphi_0 - 0.02548218) / 3.440201   # -0.3%  sum_pt < 813.4 and absphi_0 > 0.02548
        - 0.002688551 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.mean_phi - 0.01251955) / 2.019727e-06   # -0.3%  lam1 < 0.005954 and mean_phi > 0.01252
        - 0.00260926 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 1.911889 - Q.D2_b2) / 0.1770159   # -0.3%  n_dr_0p2_0p4 > 1 and D2_b2 < 1.912
        - 0.002490497 * max(0.0, Q.pt1_dr01 - 22.38578) / 0.8276113   # -0.2%  pt1_dr01 > 22.39
        - 0.002053221 * max(0.0, 0.004918231 - Q.zdr_0) * max(0.0, 182.125 - Q.pt_1) / 0.02327674   # -0.2%  zdr_0 < 0.004918 and pt_1 < 182.1
        - 0.002044012 * max(0.0, 0.002316125 - Q.centroid_offset) / 9.872932e-05   # -0.2%  centroid_offset < 0.002316
        - 0.001538285 * max(0.0, 4.192959e-06 - Q.e3) * max(0.0, Q.n_dr_0p05_0p1 - 3.0) / 6.634517e-08   # -0.2%  e3 < 4.193e-06 and n_dr_0p05_0p1 > 3
        + 0.001529833 * max(0.0, 813.4156 - Q.sum_pt) * max(0.0, 0.0095303 - Q.girth2_top2) / 0.5861327   # +0.2%  sum_pt < 813.4 and girth2_top2 < 0.00953
        - 0.001122487 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, -0.009352575 - Q.mean_phi) / 2.9369e-06   # -0.1%  lam1 < 0.005954 and mean_phi < -0.009353
        + 0.000741995 * max(0.0, 31.125 - Q.pt_4) / 0.3140792   # +0.1%  pt_4 < 31.12
        + 0.0004252114 * max(0.0, 6.638339 - Q.log_sum_pt) * max(0.0, 39.3125 - Q.pt_3) / 0.05315412   # +0.0%  log_sum_pt < 6.638 and pt_3 < 39.31
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 13.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.63503 * (0.2684947
        + 0.1992622 * Q.lam1 / 0.00591636   # +19.9%  lam1
        - 0.1177381 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # -11.8%  girth < 0.1484
        - 0.07624055 * max(0.0, Q.LHA - 0.1967397) / 0.07109091   # -7.6%  LHA > 0.1967
        + 0.07418652 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # +7.4%  sj2_dr > 0.1295
        - 0.0657234 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -6.6%  sj2_dr > 0.1592
        - 0.05299586 * Q.e2 / 0.02863215   # -5.3%  e2
        + 0.04807981 * max(0.0, Q.lam2 - 0.0001947983) / 0.0004461515   # +4.8%  lam2 > 0.0001948
        + 0.03416525 * max(0.0, Q.tau1 - 0.04369778) / 0.031989   # +3.4%  tau1 > 0.0437
        - 0.03098813 * max(0.0, 0.06810151 - Q.z_7) / 0.01877013   # -3.1%  z_7 < 0.0681
        - 0.02916208 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.6947818 - Q.planar_flow) / 0.0166918   # -2.9%  sj2_dr > 0.1683 and planar_flow < 0.6948
        - 0.02630408 * max(0.0, Q.lam1 - 0.00483998) / 0.002931468   # -2.6%  lam1 > 0.00484
        - 0.02134957 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, 0.2229947 - Q.dr_7) / 0.0002186438   # -2.1%  lam1 < 0.003377 and dr_7 < 0.223
        - 0.0198321 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # -2.0%  lam1 < 0.003377
        - 0.01965308 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.planar_flow - 0.06126205) / 0.0002819968   # -2.0%  lam2 > 0.0001948 and planar_flow > 0.06126
        - 0.01897277 * max(0.0, Q.lam1 - 0.0008722282) / 0.00523234   # -1.9%  lam1 > 0.0008722
        + 0.01724849 * max(0.0, 0.002915531 - Q.girth2_top3) / 0.001178811   # +1.7%  girth2_top3 < 0.002916
        - 0.01379911 * max(0.0, Q.lam1 - 0.00483998) * max(0.0, 0.5042684 - Q.sj3_pairmin_over_m) / 0.0008624329   # -1.4%  lam1 > 0.00484 and sj3_pairmin_over_m < 0.5043
        + 0.0137893 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +1.4%  sj2_dr > 0.1683
        - 0.01320047 * max(0.0, Q.pt_7 - 20.125) * max(0.0, 0.2669656 - Q.D2_b2) / 1.268076   # -1.3%  pt_7 > 20.12 and D2_b2 < 0.267
        - 0.01294175 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -1.3%  log_sum_pt > 6.67
        + 0.0123811 * max(0.0, 0.2669656 - Q.D2_b2) / 0.07454565   # +1.2%  D2_b2 < 0.267
        - 0.01005431 * max(0.0, Q.LHA - 0.3127275) * max(0.0, 0.6947818 - Q.planar_flow) / 0.005887138   # -1.0%  LHA > 0.3127 and planar_flow < 0.6948
        - 0.009105222 * max(0.0, Q.LHA - 0.3127275) / 0.01533444   # -0.9%  LHA > 0.3127
        - 0.008629611 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, 6.538559 - Q.log_sum_pt) / 0.0001231869   # -0.9%  lam2 > 0.0001948 and log_sum_pt < 6.539
        + 0.00807905 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.8%  n_dr_0p2_0p4 > 1
        - 0.007929772 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.eccentricity - 0.4829602) / 8.513077e-05   # -0.8%  lam2 > 0.0001948 and eccentricity > 0.483
        + 0.006462404 * max(0.0, Q.centroid_offset - 0.02355416) / 0.003991673   # +0.6%  centroid_offset > 0.02355
        - 0.006401867 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 4.721224 - Q.D2_b2) / 0.5862342   # -0.6%  n_dr_0p2_0p4 > 1 and D2_b2 < 4.721
        - 0.006024254 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -0.6%  lam2 > 0.003408
        + 0.004273911 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # +0.4%  e3 > 8.148e-05
        + 0.004132252 * max(0.0, Q.lam2 - 0.003408389) * max(0.0, 111.75 - Q.pt_2) / 0.006488771   # +0.4%  lam2 > 0.003408 and pt_2 < 111.8
        - 0.003673134 * max(0.0, 48.03125 - Q.pt_6) * max(0.0, 6.46415 - Q.log_sum_pt) / 0.7570358   # -0.4%  pt_6 < 48.03 and log_sum_pt < 6.464
        + 0.003256393 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +0.3%  sj2_dr > 0.3004
        + 0.002049751 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.2%  sum_pt > 988.4
        + 0.001914258 * max(0.0, Q.lam2 - 0.003408389) * max(0.0, Q.pt2_over_pt0 - 0.1852611) / 6.009766e-05   # +0.2%  lam2 > 0.003408 and pt2_over_pt0 > 0.1853
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 20.77;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.76753 * (0.005415296
        + 0.1683 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +16.8%  lam1 < 0.012
        + 0.07488302 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 0.001130645 - Q.lam2) / 1.136697e-05   # +7.5%  lam1 < 0.01643 and lam2 < 0.001131
        + 0.07056534 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +7.1%  planar_flow < 0.2534
        - 0.06161234 * max(0.0, 0.04447357 - Q.e2) / 0.02019334   # -6.2%  e2 < 0.04447
        - 0.05210182 * max(0.0, 0.06108601 - Q.girth) / 0.01868685   # -5.2%  girth < 0.06109
        - 0.05019604 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -5.0%  girth < 0.0717
        + 0.04991672 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +5.0%  tau1 < 0.07284
        - 0.04369827 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # -4.4%  lam1 < 0.01643
        + 0.03930537 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # +3.9%  sj2_dr > 0.1295
        + 0.03797434 * max(0.0, 0.05028464 - Q.e2) / 0.02502152   # +3.8%  e2 < 0.05028
        - 0.03656711 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, 0.02685622 - Q.centroid_offset) / 2.007754e-05   # -3.7%  lam1 < 0.003377 and centroid_offset < 0.02686
        - 0.03478098 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # -3.5%  sj3_dr_max > 0.179
        + 0.02839705 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +2.8%  centroid_offset < 0.01838
        + 0.02719192 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +2.7%  lam1 < 0.005954
        - 0.02397088 * max(0.0, 45.75 - Q.pt_7) / 12.14632   # -2.4%  pt_7 < 45.75
        - 0.02051241 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -2.1%  sj2_dr > 0.1592
        - 0.01867479 * max(0.0, 7.330536e-06 - Q.e3) / 2.382852e-06   # -1.9%  e3 < 7.331e-06
        - 0.0177069 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.centroid_offset - 0.01627885) / 2.281746e-05   # -1.8%  lam1 < 0.012 and centroid_offset > 0.01628
        - 0.0137129 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.003652679   # -1.4%  planar_flow < 0.2534 and centroid_offset < 0.0499
        - 0.01297513 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 0.716559 - Q.D2_b2) / 0.0002593059   # -1.3%  lam1 < 0.005954 and D2_b2 < 0.7166
        + 0.01246179 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.716559 - Q.D2_b2) / 0.0007061152   # +1.2%  lam1 < 0.008376 and D2_b2 < 0.7166
        - 0.01117219 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -1.1%  planar_flow < 0.2534 and sum_pt < 840
        - 0.009880323 * max(0.0, Q.centroid_offset - 0.02355416) / 0.003991673   # -1.0%  centroid_offset > 0.02355
        - 0.009193664 * max(0.0, Q.sj3_dr_max - 0.1426152) / 0.06279029   # -0.9%  sj3_dr_max > 0.1426
        - 0.009145407 * max(0.0, 0.1357675 - Q.tau21) / 0.01683052   # -0.9%  tau21 < 0.1358
        - 0.009093925 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 3.916009 - Q.D3) / 0.2321659   # -0.9%  planar_flow < 0.2534 and D3 < 3.916
        - 0.009020566 * max(0.0, 0.002127561 - Q.mean_phi2) / 0.0009769232   # -0.9%  mean_phi2 < 0.002128
        - 0.008486553 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 937.0312 - Q.sum_pt) / 1.062777   # -0.8%  centroid_offset < 0.01838 and sum_pt < 937
        + 0.007120913 * max(0.0, 0.01133547 - Q.tau2) / 0.003675653   # +0.7%  tau2 < 0.01134
        + 0.006131086 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.6%  sj2_dr > 0.2688
        + 0.005931749 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # +0.6%  sj3_dr_max > 0.2623
        - 0.005605542 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, 0.2534037 - Q.planar_flow) / 4.073218e-05   # -0.6%  lam1 < 0.003377 and planar_flow < 0.2534
        + 0.002561478 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.centroid_offset - 0.01837778) / 3.28605e-05   # +0.3%  lam1 < 0.01643 and centroid_offset > 0.01838
        - 0.00242762 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, Q.C3 - 0.04473419) / 6.06784e-07   # -0.2%  lam2 < 0.0005373 and C3 > 0.04473
        + 0.00229265 * max(0.0, 7.330536e-06 - Q.e3) * max(0.0, 0.716559 - Q.D2_b2) / 1.072852e-07   # +0.2%  e3 < 7.331e-06 and D2_b2 < 0.7166
        + 0.002148937 * max(0.0, 0.06108601 - Q.girth) * max(0.0, Q.dr1_7 - 0.2025074) / 5.186613e-05   # +0.2%  girth < 0.06109 and dr1_7 > 0.2025
        - 0.002138617 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.dr1_7 - 0.2411619) / 1.071397e-05   # -0.2%  lam1 < 0.012 and dr1_7 > 0.2412
        - 0.001010786 * max(0.0, 0.01133547 - Q.tau2) * max(0.0, Q.pt1_dr01 - 22.38578) / 0.002203675   # -0.1%  tau2 < 0.01134 and pt1_dr01 > 22.39
        + 0.0006781324 * max(0.0, Q.sj2_dr - 0.2414351) * max(0.0, 739.5 - Q.sum_pt) / 2.049171   # +0.1%  sj2_dr > 0.2414 and sum_pt < 739.5
        - 0.0004547787 * max(0.0, 0.004918231 - Q.zdr_0) * max(0.0, Q.dr0_7 - 0.09118326) / 2.337194e-06   # -0.0%  zdr_0 < 0.004918 and dr0_7 > 0.09118
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 1.153;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.15303 * (-0.5977546
        + 0.2824766 * max(0.0, Q.girth - 0.1245537) / 0.002632136   # +28.2%  girth > 0.1246
        - 0.1624792 * max(0.0, Q.sd_rg - 0.324646) * max(0.0, 6.896095 - Q.log_sum_pt) / 0.001091861   # -16.2%  sd_rg > 0.3246 and log_sum_pt < 6.896
        - 0.1251185 * max(0.0, Q.LHA - 0.3855647) / 0.004130434   # -12.5%  LHA > 0.3856
        + 0.1067799 * max(0.0, Q.sd_rg - 0.324646) * max(0.0, 6.701242 - Q.log_sum_pt) / 0.0007568975   # +10.7%  sd_rg > 0.3246 and log_sum_pt < 6.701
        + 0.07255245 * max(0.0, Q.sd_rg - 0.324646) / 0.001748965   # +7.3%  sd_rg > 0.3246
        - 0.0682048 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -6.8%  e2 > 0.06344
        - 0.06089898 * max(0.0, Q.girth - 0.1245537) * max(0.0, 62.25 - Q.pt_6) / 0.06207836   # -6.1%  girth > 0.1246 and pt_6 < 62.25
        + 0.02693592 * max(0.0, Q.girth - 0.1484084) / 0.0008957407   # +2.7%  girth > 0.1484
        + 0.02599629 * max(0.0, Q.girth - 0.1245537) * max(0.0, Q.C2_b2 - 0.009032972) / 2.778481e-05   # +2.6%  girth > 0.1246 and C2_b2 > 0.009033
        + 0.02413717 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # +2.4%  centroid_offset > 0.03776
        + 0.01783521 * max(0.0, Q.n_dr_0p2_0p4 - 2.0) / 0.05970756   # +1.8%  n_dr_0p2_0p4 > 2
        - 0.01270298 * max(0.0, Q.girth - 0.1484084) * max(0.0, 588.5859 - Q.sum_pt) / 0.08773343   # -1.3%  girth > 0.1484 and sum_pt < 588.6
        + 0.01016221 * max(0.0, Q.girth - 0.1245537) * max(0.0, Q.log_sum_pt - 6.502799) / 3.140724e-05   # +1.0%  girth > 0.1246 and log_sum_pt > 6.503
        - 0.003719762 * max(0.0, Q.girth - 0.1484084) * max(0.0, Q.log_sum_pt - 6.502799) / 8.799927e-06   # -0.4%  girth > 0.1484 and log_sum_pt > 6.503
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 22.27;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.26605 * (0.04908941
        + 0.3296281 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # +33.0%  girth < 0.1484
        + 0.1138952 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +11.4%  lam1 < 0.01643
        - 0.07056854 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -7.1%  girth < 0.1484 and log_sum_pt < 6.804
        - 0.0609851 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # -6.1%  C2_b2 < 0.009033
        - 0.05762383 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -5.8%  tau1 < 0.1136
        - 0.04822215 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -4.8%  e2 < 0.08001
        - 0.04549752 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 1.399607e-06   # -4.5%  e3 < 8.148e-05 and centroid_offset < 0.03776
        - 0.03155256 * max(0.0, Q.pt_7 - 31.85938) / 5.945918   # -3.2%  pt_7 > 31.86
        + 0.03116978 * max(0.0, Q.z_7 - 0.04624032) / 0.01204993   # +3.1%  z_7 > 0.04624
        - 0.02347361 * max(0.0, 48.71875 - Q.pt_7) / 14.7465   # -2.3%  pt_7 < 48.72
        + 0.01993275 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +2.0%  e3 < 8.148e-05
        + 0.01800823 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 40.04062 - Q.pt_7) / 652.1541   # +1.8%  sum_pt_top5 > 687.4 and pt_7 < 40.04
        - 0.01585779 * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.09151624   # -1.6%  sj3_dr_min < 0.1278
        - 0.01452167 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -1.5%  log_sum_pt > 6.67
        - 0.01398268 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.4%  sj3_dr_max > 0.2337
        - 0.01213503 * max(0.0, 50.25 - Q.pt_6) / 11.66482   # -1.2%  pt_6 < 50.25
        - 0.01129365 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -1.1%  girth < 0.1484 and pt_7 < 38.53
        - 0.01046782 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.0003061234 - Q.lam2) / 2.082734e-05   # -1.0%  girth < 0.1484 and lam2 < 0.0003061
        - 0.01041757 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 25.57812 - Q.pt_7) / 0.01870322   # -1.0%  lam1 < 0.01643 and pt_7 < 25.58
        - 0.009561893 * max(0.0, Q.z_dr_0_0p05 - 0.6080732) / 0.1760045   # -1.0%  z_dr_0_0p05 > 0.6081
        + 0.009416588 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.9%  sum_pt_top5 > 687.4
        + 0.006248847 * max(0.0, 0.08000524 - Q.e2) * max(0.0, 60.03125 - Q.pt_4) / 0.4322666   # +0.6%  e2 < 0.08001 and pt_4 < 60.03
        + 0.00516361 * max(0.0, Q.pt_7 - 31.85938) * max(0.0, 0.1598486 - Q.max_dr) / 0.4085666   # +0.5%  pt_7 > 31.86 and max_dr < 0.1598
        + 0.005080028 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +0.5%  sum_pt_top5 > 791.1
        - 0.005002451 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # -0.5%  log_sum_pt < 6.464
        - 0.002635289 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # -0.3%  pt_5 < 24.58
        - 0.00240869 * max(0.0, Q.z_top5 - 0.8770155) / 0.00720125   # -0.2%  z_top5 > 0.877
        + 0.002402178 * max(0.0, Q.z_7 - 0.04624032) * max(0.0, Q.max_dr - 0.08050702) / 0.0007498341   # +0.2%  z_7 > 0.04624 and max_dr > 0.08051
        - 0.002397071 * max(0.0, 27.57812 - Q.pt_6) / 0.9875435   # -0.2%  pt_6 < 27.58
        + 0.002369538 * max(0.0, Q.pt_7 - 31.85938) * max(0.0, 0.01228369 - Q.zdr_1) / 0.03012054   # +0.2%  pt_7 > 31.86 and zdr_1 < 0.01228
        - 0.00236795 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.2%  sum_pt > 988.4
        - 0.002123621 * max(0.0, Q.z_7 - 0.0586137) * max(0.0, Q.lam2 - 4.64158e-05) / 5.481848e-06   # -0.2%  z_7 > 0.05861 and lam2 > 4.642e-05
        + 0.0009620887 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # +0.1%  sum_pt_top5 > 840
        - 0.0009336781 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, Q.C2_b2 - 0.0001052765) / 4.647133e-05   # -0.1%  log_sum_pt > 6.67 and C2_b2 > 0.0001053
        + 0.0008219265 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.z_6 - 0.03932388) / 0.02599753   # +0.1%  sum_pt > 988.4 and z_6 > 0.03932
        - 0.000699277 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.sj3_pairmin_over_m - 0.2195798) / 0.4564276   # -0.1%  sum_pt > 988.4 and sj3_pairmin_over_m > 0.2196
        - 0.0001717378 * max(0.0, 6.46415 - Q.log_sum_pt) * max(0.0, 31.125 - Q.pt_4) / 0.01308589   # -0.0%  log_sum_pt < 6.464 and pt_4 < 31.12
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 26.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.06644 * (-0.191834
        + 0.1312227 * max(0.0, Q.girth - 0.02689598) / 0.03613842   # +13.1%  girth > 0.0269
        + 0.09673247 * max(0.0, 0.1667615 - Q.sd_rg) / 0.07405603   # +9.7%  sd_rg < 0.1668
        - 0.07723775 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -7.7%  lam1 < 0.006507
        - 0.06364194 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -6.4%  girth > 0.08724
        + 0.06313745 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +6.3%  lam1 < 0.012
        - 0.05421809 * max(0.0, 0.1773029 - Q.sd_rg) / 0.08115541   # -5.4%  sd_rg < 0.1773
        + 0.05270412 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +5.3%  tau1 < 0.09539
        + 0.04835838 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +4.8%  e2 < 0.04111
        - 0.04137122 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 1.0 - Q.psi_0p3) / 0.0007331701   # -4.1%  z_dr_0p05_0p1 > 0.1639 and psi_0p3 < 1
        - 0.03938896 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -3.9%  centroid_offset > 0.0499
        + 0.03535598 * max(0.0, 0.1416226 - Q.tau1) / 0.08190733   # +3.5%  tau1 < 0.1416
        + 0.0226167 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.sd_rg - 0.1572969) / 6.238901e-05   # +2.3%  lam2 < 0.003408 and sd_rg > 0.1573
        - 0.02079594 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -2.1%  e3 < 8.148e-05
        - 0.02030716 * max(0.0, Q.sd_rg - 0.1876504) / 0.01921774   # -2.0%  sd_rg > 0.1877
        - 0.01848894 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.158545   # -1.8%  n_dr_0p05_0p1 < 5
        - 0.01845018 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -1.8%  lam1 < 0.00733
        + 0.01645076 * max(0.0, Q.girth - 0.06663269) / 0.01393206   # +1.6%  girth > 0.06663
        - 0.0154723 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -1.5%  planar_flow < 0.1115 and lam1 < 0.00733
        - 0.01432237 * max(0.0, 0.06545715 - Q.sd_rg) / 0.02200467   # -1.4%  sd_rg < 0.06546
        - 0.01398992 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 0.20552 - Q.z_dr_0p2_0p4) / 0.03609844   # -1.4%  z_dr_0p05_0p1 > 0.1639 and z_dr_0p2_0p4 < 0.2055
        + 0.01277876 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.1644126   # +1.3%  z_dr_0p05_0p1 > 0.1639 and n_dr_0p2_0p4 < 1
        - 0.01245133 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -1.2%  tau1 < 0.1136
        + 0.0115233 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # +1.2%  planar_flow < 0.1115 and lam1 < 0.01643
        + 0.01103832 * max(0.0, Q.sd_rg - 0.1572969) / 0.02905066   # +1.1%  sd_rg > 0.1573
        + 0.008796788 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 0.04510668 - Q.z_dr_0p1_0p2) / 0.01057962   # +0.9%  z_dr_0p05_0p1 < 0.5883 and z_dr_0p1_0p2 < 0.04511
        - 0.008495305 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.LHA - 0.1767241) / 0.0002170323   # -0.8%  lam2 < 0.003408 and LHA > 0.1767
        - 0.007675318 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # -0.8%  girth2_top3 < 0.002152
        - 0.007553785 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02685622) / 1.516228e-05   # -0.8%  lam1 < 0.01643 and centroid_offset > 0.02686
        + 0.00698046 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +0.7%  LHA > 0.3033
        - 0.006394413 * max(0.0, 0.01289969 - Q.e2) / 0.002527207   # -0.6%  e2 < 0.0129
        + 0.005309537 * max(0.0, Q.sd_rg - 0.1449048) / 0.03437807   # +0.5%  sd_rg > 0.1449
        - 0.004881798 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 1.0 - Q.sd_nremoved) / 0.02680832   # -0.5%  planar_flow < 0.1115 and sd_nremoved < 1
        - 0.004668662 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 6.670067 - Q.log_sum_pt) / 0.03946963   # -0.5%  z_dr_0p05_0p1 > 0.1639 and log_sum_pt < 6.67
        - 0.004560496 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.sd_rg - 0.2037854) / 2.928331e-05   # -0.5%  lam2 < 0.003408 and sd_rg > 0.2038
        + 0.003555397 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.2832687 - Q.sj2_zsoft) / 0.002541531   # +0.4%  planar_flow < 0.1115 and sj2_zsoft < 0.2833
        + 0.003508633 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00595415 - Q.lam1) / 3.161575e-05   # +0.4%  planar_flow < 0.1115 and lam1 < 0.005954
        + 0.00341447 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.centroid_offset - 0.02685622) / 6.831551e-06   # +0.3%  lam2 < 0.003408 and centroid_offset > 0.02686
        - 0.003150373 * max(0.0, 0.2330919 - Q.sd_rg) / 0.1256101   # -0.3%  sd_rg < 0.2331
        - 0.00299453 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.z_dr_0p2_0p4 - 0.0) / 5.139675e-05   # -0.3%  lam2 < 0.003408 and z_dr_0p2_0p4 > 0
        + 0.0018969 * max(0.0, 0.1116471 - Q.sd_rg) / 0.04385005   # +0.2%  sd_rg < 0.1116
        - 0.00171637 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01627885 - Q.centroid_offset) / 0.0001743838   # -0.2%  planar_flow < 0.1115 and centroid_offset < 0.01628
        + 0.00128676 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.1974628) / 1.23895e-05   # +0.1%  lam1 < 0.006507 and sj3_dr23 > 0.1975
        + 0.0005688641 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 0.415246 - Q.D2) / 0.0001335382   # +0.1%  lam1 < 0.01643 and D2 < 0.4152
        + 0.0005241144 * max(0.0, 0.04110972 - Q.e2) * max(0.0, -0.02665591 - Q.mean_eta) / 5.627002e-06   # +0.1%  e2 < 0.04111 and mean_eta < -0.02666
        + 1.19835e-05 * max(0.0, 0.1136369 - Q.tau1) * max(0.0, 0.415246 - Q.D2) / 0.0002136968   # +0.0%  tau1 < 0.1136 and D2 < 0.4152
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 28.71;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.71283 * (-0.01432593
        - 0.1135599 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -11.4%  girth < 0.0717
        + 0.08077067 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +8.1%  lam1 < 0.01643
        - 0.07301542 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -7.3%  lam1 < 0.00733
        - 0.07221757 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -7.2%  girth < 0.08724
        - 0.06296667 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # -6.3%  LHA < 0.3127
        + 0.05324187 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +5.3%  e2 < 0.04111
        + 0.0357167 * max(0.0, 0.007639643 - Q.girth2_top2) / 0.004543329   # +3.6%  girth2_top2 < 0.00764
        + 0.03390969 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # +3.4%  girth2_top3 < 0.002152
        + 0.03328134 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # +3.3%  lam1 < 0.005433
        - 0.03250436 * max(0.0, Q.tau1 - 0.09538712) / 0.01057268   # -3.3%  tau1 > 0.09539
        - 0.03188048 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -3.2%  lam1 < 0.008376
        + 0.02819957 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +2.8%  tau1 < 0.05357
        + 0.02699338 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # +2.7%  tau1 > 0.1028
        + 0.02627338 * max(0.0, 0.032347 - Q.e2) / 0.01171807   # +2.6%  e2 < 0.03235
        + 0.02482308 * max(0.0, Q.eccentricity - 0.9458207) * max(0.0, Q.sum_pt_top5 - 367.5938) / 5.10997   # +2.5%  eccentricity > 0.9458 and sum_pt_top5 > 367.6
        + 0.02178432 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +2.2%  lam1 < 0.002464
        + 0.01808048 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +1.8%  lam1 < 0.012
        - 0.01510083 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.lam1 - 0.008375572) / 0.0001169809   # -1.5%  N2 < 0.2233 and lam1 > 0.008376
        - 0.01358164 * max(0.0, 7.330536e-06 - Q.e3) / 2.382852e-06   # -1.4%  e3 < 7.331e-06
        + 0.01332508 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.2931906) / 0.001536651   # +1.3%  N2 < 0.2233 and LHA > 0.2932
        - 0.0129201 * max(0.0, 0.03360421 - Q.girth) / 0.006496997   # -1.3%  girth < 0.0336
        - 0.01282478 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 0.01849752 - Q.zdr_1) / 3.216563e-05   # -1.3%  lam2 < 0.003408 and zdr_1 < 0.0185
        + 0.01279234 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +1.3%  LHA < 0.3033
        + 0.01088204 * max(0.0, Q.sj3_dr13 - 0.181053) / 0.02525513   # +1.1%  sj3_dr13 > 0.1811
        - 0.01003328 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1872617 - Q.sj2_dr) / 0.0007896743   # -1.0%  N2 < 0.2233 and sj2_dr < 0.1873
        - 0.009877719 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 0.0002261574   # -1.0%  lam1 < 0.008376 and D2 < 0.8757
        - 0.009530477 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.4235925) / 7.321856e-05   # -1.0%  N2 < 0.2233 and LHA > 0.4236
        - 0.008934726 * max(0.0, Q.eccentricity - 0.9458207) / 0.02147681   # -0.9%  eccentricity > 0.9458
        - 0.008698802 * max(0.0, Q.tau1 - 0.1416226) / 0.003662393   # -0.9%  tau1 > 0.1416
        + 0.008090964 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.1974628) / 2.285306e-05   # +0.8%  lam1 < 0.008376 and sj3_dr23 > 0.1975
        - 0.00788838 * max(0.0, 0.002151568 - Q.girth2_top3) * max(0.0, 0.1950135 - Q.planar_flow) / 2.579988e-05   # -0.8%  girth2_top3 < 0.002152 and planar_flow < 0.195
        - 0.007821452 * max(0.0, Q.sj3_dr13 - 0.249546) / 0.01061611   # -0.8%  sj3_dr13 > 0.2495
        + 0.007707517 * max(0.0, Q.tau1 - 0.1027642) * max(0.0, 0.5327104 - Q.D2_b2) / 0.001862765   # +0.8%  tau1 > 0.1028 and D2_b2 < 0.5327
        - 0.006138985 * max(0.0, Q.eccentricity - 0.9458207) * max(0.0, Q.mean_phi - -0.01753483) / 0.0004008253   # -0.6%  eccentricity > 0.9458 and mean_phi > -0.01753
        + 0.005266552 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # +0.5%  lam1 < 0.006507 and D2 < 0.8757
        + 0.005163464 * max(0.0, 0.009233892 - Q.zdr_0) / 0.002026168   # +0.5%  zdr_0 < 0.009234
        - 0.004964431 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.centroid_offset - 0.03776099) / 5.89929e-06   # -0.5%  lam1 < 0.01643 and centroid_offset > 0.03776
        - 0.004363124 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.1974628) / 9.808733e-06   # -0.4%  lam1 < 0.005954 and sj3_dr23 > 0.1975
        - 0.004100159 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3855647) / 0.0002330351   # -0.4%  N2 < 0.2233 and LHA > 0.3856
        + 0.003917027 * max(0.0, Q.tau1 - 0.1416226) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0006707461   # +0.4%  tau1 > 0.1416 and D2_b2 < 0.5327
        - 0.003868681 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3255822) / 0.0007869707   # -0.4%  N2 < 0.2233 and LHA > 0.3256
        - 0.003800149 * max(0.0, 0.2213841 - Q.D3) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 0.0005276367   # -0.4%  D3 < 0.2214 and n_dr_0p1_0p2 > 1
        - 0.003043066 * max(0.0, 0.1329373 - Q.LHA) / 0.007753684   # -0.3%  LHA < 0.1329
        + 0.002778334 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +0.3%  tau1 > 0.1136
        + 0.002556484 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 0.001116597   # +0.3%  lam1 < 0.01643 and D2 < 0.8757
        - 0.002275606 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.z_dr_0p2_0p4 - 0.20552) / 0.000298346   # -0.2%  N2 < 0.2233 and z_dr_0p2_0p4 > 0.2055
        + 0.002176977 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1591713 - Q.sj2_dr) / 0.0002571871   # +0.2%  N2 < 0.2233 and sj2_dr < 0.1592
        + 0.002119595 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, Q.centroid_offset - 0.03776099) / 7.468637e-07   # +0.2%  lam1 < 0.00733 and centroid_offset > 0.03776
        - 0.001013701 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.C2_b2 - 0.009032972) / 3.381092e-06   # -0.1%  lam1 < 0.01643 and C2_b2 > 0.009033
        + 0.0008807878 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, 0.2213841 - Q.D3) / 1.704383e-05   # +0.1%  girth2_top2 < 0.00764 and D3 < 0.2214
        - 0.0008778018 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, Q.sj3_dr23 - 0.1797097) / 0.0003211663   # -0.1%  z_dr_0p05_0p1 > 0.7509 and sj3_dr23 > 0.1797
        - 0.0008334482 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.02320757 - Q.z_7) / 1.527561e-05   # -0.1%  N2 < 0.2233 and z_7 < 0.02321
        - 0.0006326165 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2213841 - Q.D3) / 2.970205e-05   # -0.1%  zdr_0 < 0.02118 and D3 < 0.2214
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9005867647058824, 0.7963481092436975, 2.0618375, 0.9535363445378151, 1.2308752100840337, 1.7825397058823529, 1.0183390756302522, 1.8695502100840335, 0.28036428571428573, 3.0465319327731093, 1.9621139705882353, 2.0702853991596637, 0.08640231092436974, 3.3312453781512605, 0.4264518907563025, 0.36139138655462183]
T = [2.3454305204503676, 1.3851509585084034, 3.282601726628151, 2.5119192981880247, 2.8580426503413867]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +38%, n9 +22%, n5 -14%, n1 +13%, n0 -6%, n6 +5% ...
            + 0.3777327 * h[2] / H_AVG[2]
            + 0.2232523 * h[9] / H_AVG[9]
            - 0.142501 * h[5] / H_AVG[5]
            + 0.1326296 * h[1] / H_AVG[1]
            - 0.0599961 * h[0] / H_AVG[0]
            + 0.04748844 * h[6] / H_AVG[6]
            - 0.01639991 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -18%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5584471 * h[9] / H_AVG[9]
            - 0.1770668 * h[10] / H_AVG[10]
            + 0.09189784 * h[6] / H_AVG[6]
            - 0.08330829 * h[4] / H_AVG[4]
            + 0.06032306 * h[5] / H_AVG[5]
            + 0.0163065 * h[15] / H_AVG[15]
            + 0.01265044 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -15%, n7 +12%, n14 -10%, n6 -10%, n0 +9% ...
            + 0.2365066 * h[11] / H_AVG[11]
            - 0.1452409 * h[3] / H_AVG[3]
            + 0.1245854 * h[7] / H_AVG[7]
            - 0.09743458 * h[14] / H_AVG[14]
            - 0.09694474 * h[6] / H_AVG[6]
            + 0.09430833 * h[0] / H_AVG[0]
            - 0.07568892 * h[15] / H_AVG[15]
            + 0.07135443 * h[13] / H_AVG[13]
            - 0.02900264 * h[9] / H_AVG[9]
            - 0.02135229 * h[8] / H_AVG[8]
            - 0.007581145 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -21%, n6 -15%, n13 +7%, n14 +6%, n1 +4% ...
            + 0.3488773 * h[7] / H_AVG[7]
            - 0.2135276 * h[3] / H_AVG[3]
            - 0.152026 * h[6] / H_AVG[6]
            + 0.07252521 * h[13] / H_AVG[13]
            + 0.06366425 * h[14] / H_AVG[14]
            + 0.03962847 * h[1] / H_AVG[1]
            + 0.03828233 * h[4] / H_AVG[4]
            - 0.03790095 * h[9] / H_AVG[9]
            - 0.02247978 * h[15] / H_AVG[15]
            + 0.01108801 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +26%, n5 -16%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4735123 * h[13] / H_AVG[13]
            + 0.2574464 * h[10] / H_AVG[10]
            - 0.1559231 * h[5] / H_AVG[5]
            + 0.05383384 * h[4] / H_AVG[4]
            + 0.02085204 * h[3] / H_AVG[3]
            + 0.01839311 * h[8] / H_AVG[8]
            - 0.01511564 * h[12] / H_AVG[12]
            + 0.004923533 * h[0] / H_AVG[0]
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
