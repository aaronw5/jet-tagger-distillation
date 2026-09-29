"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.
  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1
             avg_k = the term's average size on the entire training set (term / avg is 1 on average), so share_k is the
             fraction of the neuron's average input that comes from that if-statement (sign: pushes it up / down).
  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1
             h_j = neuron j (after max(0, .) and the network's rounding), avg_j = its average on the training jets.

How much each neuron matters (share of all class scores, averaged over the training jets):
  neuron  8:  12.6%   (on for 53% of jets)
  neuron  1:  12.6%   (on for 95% of jets)
  neuron  5:  12.0%   (on for 78% of jets)
  neuron  4:  10.1%   (on for 66% of jets)
  neuron 13:   7.5%   (on for 86% of jets)
  neuron  0:   7.5%   (on for 63% of jets)
  neuron  9:   7.3%   (on for 60% of jets)
  neuron 10:   6.4%   (on for 77% of jets)
  neuron 12:   4.9%   (on for 59% of jets)
  neuron  6:   4.0%   (on for 55% of jets)
  neuron  7:   3.8%   (on for 29% of jets)
  neuron  3:   3.5%   (on for 56% of jets)
  neuron 11:   3.4%   (on for 70% of jets)
  neuron 15:   2.2%   (on for 54% of jets)
  neuron 14:   1.7%   (on for 30% of jets)
  neuron  2:   0.6%   (on for 42% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.4% (the network: 81.1%); same class as the network for 94.1% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_12         mass of particles 0 and 12 [GeV]
  Q.pair_mass_0_3          mass of particles 0 and 3 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.ptdr0_13               pT13 · ΔR(0, 13) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_pt               pT [GeV] of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.z_1                    pT of particle 1 / total pT
  Q.z_11                   pT of particle 11 / total pT
  Q.z_14                   pT of particle 14 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_nremoved            number of branches removed by soft drop
  Q.absphi_13              |Δφ| of particle 13
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft10_dr0             ΔR between the hardest and the 10. softest real particle (0 if among the 15 hardest)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.soft10_dr              ΔR from the jet axis of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.eta_0                  Δη of particle 0
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
"""
import math
from types import SimpleNamespace

CLASSES = ['g', 'q', 'W', 'Z', 't']
W = [[0.0, 0.0, 0.75, -1.375, 0.125], [0.625, -0.1875, 0.0, 0.0, 0.0], [0.0, 0.125, 0.0, -0.375, 0.0], [-0.75, 0.0, 0.4375, 0.4375, 0.0], [-0.875, -1.0625, 0.34375, 0.0, 0.1875], [-0.3125, 0.0, 0.578125, 0.6875, -0.46875], [0.15625, 0.21875, 0.0625, -0.96875, 0.0], [0.0, 0.0, -0.625, 0.90625, -0.28125], [0.03125, 0.015625, -0.875, -0.9375, 0.2109375], [0.515625, 0.5625, -0.21875, 0.0, 0.0], [-0.015625, -0.015625, 0.0, 0.0, 0.984375], [0.0, -0.25, 0.59375, 0.0, 0.0], [0.234375, 0.34375, -0.40625, 0.3125, -0.375], [0.0, 0.0, 0.0, 0.0, -0.90625], [0.0, 0.0, -1.375, 0.0, 0.0], [0.0, 0.0, -0.1875, 0.5625, -0.375]]
B = [-1.078125, 1.359375, 0.09375, 0.984375, 0.78125]
INT_BITS = [2, 4, 3, 2, 2, 2, 3, 3, 4, 3, 3, 3, 3, 2, 2, 3]
FRAC_BITS = [5, 5, 3, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 5, 5, 4]


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
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_12=pair_mass(0, 12),
        pair_mass_0_3=pair_mass(0, 3),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top2=mass_of(2),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_0=pt[0],
        pt_11=pt[11],
        ptdr0_13=pt[13] * math.sqrt(dist2(0, 13)) if pt[13] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft2_pt=softp(2, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        z_1=z[1],
        z_11=z[11],
        z_14=z[14],
        z_6=z[6],
        z_9=z[9],
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft7_z=softp(7, 'z'),
        soft8_z=softp(8, 'z'),
        z_top10_slots=sum(pt[:10]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        zdr_5=z[5] * dr[5],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sd_nremoved=softdrop("removed"),
        absphi_13=abs(phi[13]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft10_dr0=softp(10, 'dr0'),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        soft10_dr=softp(10, 'dr'),
        eta_0=eta[0],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
    )


def neuron_0(Q):
    # scale S = 13.15;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.1533 * (0.1022228
        - 0.1993411 * max(0.0, Q.mass - 78.26182) / 21.33658   # -19.9%  mass > 78.26
        - 0.1636795 * max(0.0, Q.mass - 92.85979) / 14.48273   # -16.4%  mass > 92.86
        - 0.06299867 * max(0.0, Q.mass - 74.25181) / 24.07648   # -6.3%  mass > 74.25
        + 0.05939495 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 0.002241068   # +5.9%  mass_over_sum_pt_sq < 0.007873
        - 0.05469153 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -5.5%  mass < 101
        - 0.05402029 * max(0.0, Q.mass - 91.03469) / 15.0694   # -5.4%  mass > 91.03
        + 0.04758499 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq) / 0.001689525   # +4.8%  mass_over_sum_pt_sq < 0.006939
        - 0.04420897 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -4.4%  e2_sq < 0.006167
        + 0.03044838 * max(0.0, Q.mass - 87.36377) / 16.57741   # +3.0%  mass > 87.36
        + 0.02601292 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +2.6%  sum_pt_top50 < 1157
        + 0.02388526 * max(0.0, Q.mass_top50 - 71.79516) / 24.22501   # +2.4%  mass_top50 > 71.8
        + 0.02243873 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # +2.2%  mass_top50 > 82.04
        - 0.02139156 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -2.1%  girth2_top30 < 0.006364
        - 0.01795821 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -1.8%  tau1 < 0.07057
        + 0.01764474 * max(0.0, 80.24626 - Q.mass_top30) / 14.57104   # +1.8%  mass_top30 < 80.25
        + 0.01744812 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +1.7%  n_dr_0p2_0p4 < 15
        - 0.01683564 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # -1.7%  sum_pt_top40 < 1070
        - 0.01542365 * max(0.0, 0.007538019 - Q.girth2_top20) / 0.002740841   # -1.5%  girth2_top20 < 0.007538
        - 0.01536082 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # -1.5%  z_dr_0p2_0p4 < 0.09123
        - 0.01197128 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -1.2%  sum_pt < 1013
        + 0.01134643 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +1.1%  log_sum_pt < 7.017
        + 0.01077404 * max(0.0, 0.005312783 - Q.girth2_top20) / 0.001437214   # +1.1%  girth2_top20 < 0.005313
        + 0.01010262 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +1.0%  psi_0p3 > 0.9956
        - 0.00902647 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # -0.9%  girth2_top20 < 0.006374
        + 0.008638462 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957) / 1.852228   # +0.9%  log_sum_pt < 6.989 and sum_pt_top50 > 959.1
        - 0.007031999 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -0.7%  lam1 < 0.005914
        + 0.006436582 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +0.6%  mass_top40 < 80.89
        - 0.006281916 * max(0.0, 0.9906378 - Q.z_top50_slots) / 0.004783315   # -0.6%  z_top50_slots < 0.9906
        + 0.002961781 * max(0.0, 846.1934 - Q.sum_pt_top20) / 22.83716   # +0.3%  sum_pt_top20 < 846.2
        + 0.002543253 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.0223982 - Q.C3) / 0.2900897   # +0.3%  sum_pt < 1013 and C3 < 0.0224
        - 0.00211709 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3) / 0.2515325   # -0.2%  sum_pt < 1013 and M3 < 0.03457
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 15.35;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.35147 * (0.02642515
        + 0.1234061 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # +12.3%  log_sum_pt > 6.894
        - 0.1066526 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -10.7%  log_sum_pt < 7.139
        + 0.08022377 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +8.0%  n_particles > 38
        + 0.07042385 * max(0.0, 0.1751567 - Q.tau1) / 0.09101058   # +7.0%  tau1 < 0.1752
        - 0.0503855 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -5.0%  sum_pt_top50 > 959.1
        - 0.04752867 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -4.8%  mass < 101
        + 0.04477601 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +4.5%  log_sum_pt > 6.91
        + 0.04383007 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.2734   # +4.4%  sum_pt_top2 < 689.2
        - 0.03885811 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt) / 15.05788   # -3.9%  n_particles > 38 and soft1_pt < 2.275
        - 0.03550645 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -3.6%  log_sum_pt > 6.959
        + 0.02867395 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # +2.9%  sum_pt_top40 < 1070
        + 0.02845002 * max(0.0, 1.521582 - Q.soft1_pt) / 0.9527349   # +2.8%  soft1_pt < 1.522
        + 0.02278566 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +2.3%  mass < 89.74
        - 0.01898452 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2) / 190.5117   # -1.9%  sj3_mass1 < 32.5 and sj3_mass2 < 18.68
        - 0.0178097 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -1.8%  log_sum_pt > 6.989
        - 0.01609191 * max(0.0, 47.88842 - Q.mass_top20) / 6.040308   # -1.6%  mass_top20 < 47.89
        + 0.01596926 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +1.6%  sum_pt < 1017
        - 0.01461879 * max(0.0, 31.35938 - Q.pt_9) / 6.538304   # -1.5%  pt_9 < 31.36
        - 0.01393389 * max(0.0, 30.26161 - Q.sj2_mass1) / 7.997784   # -1.4%  sj2_mass1 < 30.26
        - 0.01247778 * max(0.0, Q.z_top30_slots - 0.9341838) / 0.03406818   # -1.2%  z_top30_slots > 0.9342
        + 0.01167963 * max(0.0, 80.24626 - Q.mass_top30) / 14.57104   # +1.2%  mass_top30 < 80.25
        - 0.01161626 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -1.2%  n_dr_0p2_0p4 < 7
        + 0.01136322 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2) / 0.0008500795   # +1.1%  z_top30_slots > 0.9342 and C2 < 0.0728
        + 0.01063628 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 39.23115   # +1.1%  mass_top20 < 47.89 and n_real_top40 > 29
        - 0.01028667 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # -1.0%  M3 < 0.03188
        + 0.00831266 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1) / 1.005598   # +0.8%  n_particles > 38 and dr_1 < 0.1612
        - 0.007996416 * max(0.0, Q.zdr_0 - 0.001901263) / 0.007974889   # -0.8%  zdr_0 > 0.001901
        + 0.007980156 * max(0.0, Q.z_top30_slots - 0.9564984) / 0.01947087   # +0.8%  z_top30_slots > 0.9565
        - 0.007211715 * max(0.0, Q.z_dr_0_0p05 - 0.8459004) / 0.02484119   # -0.7%  z_dr_0_0p05 > 0.8459
        + 0.007161393 * max(0.0, Q.mass_top10 - 31.33272) / 22.02168   # +0.7%  mass_top10 > 31.33
        - 0.007114635 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.7%  psi_0p3 > 0.998
        + 0.0067033 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.7%  lam1 < 0.004673
        + 0.006554057 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0) / 0.570275   # +0.7%  n_particles > 38 and dr_0 < 0.112
        + 0.006552412 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.01395978 - Q.zdr_2) / 0.09391572   # +0.7%  n_particles > 38 and zdr_2 < 0.01396
        + 0.005973245 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793) / 0.1941151   # +0.6%  z_top30_slots > 0.9342 and max_pair_mass > 13.05
        + 0.005702241 * max(0.0, 0.003270031 - Q.girth2_top15) / 0.0007969965   # +0.6%  girth2_top15 < 0.00327
        + 0.004597424 * max(0.0, 0.0005522528 - Q.girth2_top3) / 0.000118253   # +0.5%  girth2_top3 < 0.0005523
        + 0.004182526 * max(0.0, 10.0 - Q.n_dr_0_0p05) / 2.509415   # +0.4%  n_dr_0_0p05 < 10
        + 0.003780654 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 15.47191 - Q.pair_mass_0_12) / 61.85839   # +0.4%  pt_9 < 31.36 and pair_mass_0_12 < 15.47
        + 0.003700117 * max(0.0, Q.sj2_zsoft - 0.2912328) / 0.04122672   # +0.4%  sj2_zsoft > 0.2912
        + 0.003589722 * max(0.0, 0.0007894752 - Q.girth2_top15) / 9.062109e-05   # +0.4%  girth2_top15 < 0.0007895
        + 0.003453674 * max(0.0, 0.08822608 - Q.D3) / 0.02694065   # +0.3%  D3 < 0.08823
        - 0.003256392 * max(0.0, 0.03187688 - Q.M3) * max(0.0, Q.M2 - 0.05568888) / 0.0001450223   # -0.3%  M3 < 0.03188 and M2 > 0.05569
        + 0.002111876 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874) / 0.05244257   # +0.2%  z_top30_slots > 0.9342 and ptdr0_3 > 7.408
        + 0.001826848 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, 30.0 - Q.n_real_top30) / 3.822398   # +0.2%  n_dr_0p2_0p4 < 7 and n_real_top30 < 30
        + 0.001792063 * max(0.0, 0.003270031 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 6.08612e-07   # +0.2%  girth2_top15 < 0.00327 and psi_0p3 > 0.9974
        + 0.00178823 * max(0.0, Q.sum_pt_top30 - 1191.938) / 6.938323   # +0.2%  sum_pt_top30 > 1192
        + 0.001689513 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, Q.ptdr0_4 - 4.470953) / 0.03202477   # +0.2%  z_top30_slots > 0.9565 and ptdr0_4 > 4.471
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 6.329;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.329214 * (-0.0003708532
        - 0.1490996 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -14.9%  mass < 92.86
        + 0.1019303 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15) / 0.0009227558   # +10.2%  log_sum_pt > 6.903 and girth2_top15 < 0.02147
        + 0.09854401 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # +9.9%  sum_pt > 1017
        + 0.09488105 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +9.5%  mass < 121.4
        - 0.07336948 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # -7.3%  sum_pt < 1261
        + 0.07121448 * max(0.0, 91.03469 - Q.mass) / 16.36287   # +7.1%  mass < 91.03
        - 0.06537789 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15) / 1.045047   # -6.5%  sum_pt_top50 > 988.5 and girth2_top15 < 0.02147
        - 0.05848176 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -5.8%  sum_pt > 1053
        + 0.04366105 * max(0.0, 143.7876 - Q.mass) / 57.99337   # +4.4%  mass < 143.8
        - 0.04109896 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # -4.1%  girth2_top50 < 0.008125
        - 0.03100009 * max(0.0, 787.6281 - Q.sum_pt_top3) / 328.0371   # -3.1%  sum_pt_top3 < 787.6
        + 0.03069736 * max(0.0, 0.006088416 - Q.girth2_top30) / 0.001533892   # +3.1%  girth2_top30 < 0.006088
        + 0.02439293 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +2.4%  sum_pt_top40 < 1053
        + 0.0209983 * max(0.0, Q.sum_pt_top50 - 1156.659) / 13.71464   # +2.1%  sum_pt_top50 > 1157
        - 0.01611952 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -1.6%  log_sum_pt > 7.063
        - 0.01589617 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # -1.6%  sum_pt > 1116
        + 0.01527895 * max(0.0, 1041.263 - Q.sum_pt_top40) / 49.16087   # +1.5%  sum_pt_top40 < 1041
        - 0.01273653 * max(0.0, 82.66587 - Q.mass_top30) / 15.96638   # -1.3%  mass_top30 < 82.67
        - 0.01249969 * max(0.0, 51.0 - Q.n_particles) / 9.158477   # -1.2%  n_particles < 51
        - 0.01197746 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003463934   # -1.2%  log_sum_pt > 6.903 and psi_0p3 > 0.9299
        - 0.01074447 * max(0.0, 996.8867 - Q.sum_pt_top30) / 42.94346   # -1.1%  sum_pt_top30 < 996.9
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 3.117;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.117044 * (-0.09221258
        - 0.1457955 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # -14.6%  girth2 < 0.009615
        + 0.1069087 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +10.7%  girth2_top40 < 0.008841
        + 0.08865119 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # +8.9%  girth2_top50 < 0.008125
        - 0.08366173 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # -8.4%  girth2_top40 < 0.00626
        + 0.06980532 * max(0.0, 0.006026828 - Q.girth2_top40) / 0.001351778   # +7.0%  girth2_top40 < 0.006027
        + 0.05837659 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +5.8%  n_dr_0p2_0p4 < 5
        + 0.04819772 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +4.8%  lam2 < 0.0006155
        + 0.04452577 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +4.5%  n_particles < 46
        - 0.03956516 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -4.0%  tau21 < 0.3862
        - 0.03772324 * max(0.0, 79.21004 - Q.mass_top50) / 10.65754   # -3.8%  mass_top50 < 79.21
        + 0.03513698 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) / 2.171439   # +3.5%  n_dr_0p1_0p2 < 10
        + 0.03111838 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # +3.1%  n_dr_0p2_0p4 < 8
        + 0.03060037 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732) / 1477.074   # +3.1%  n_particles < 46 and sum_pt_top30 > 800.7
        + 0.02649527 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1) / 0.1325294   # +2.6%  n_dr_0p2_0p4 < 5 and psi_0p1 < 0.9538
        - 0.02592103 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349) / 64.17204   # -2.6%  n_particles < 46 and mass_top15 > 57.87
        + 0.02297379 * max(0.0, 87.36377 - Q.mass) / 14.19996   # +2.3%  mass < 87.36
        + 0.02019327 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421) / 0.000406438   # +2.0%  D2 < 2.41 and psi_0p3 > 0.9985
        - 0.01868463 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.006794973 - Q.zdr_4) / 0.003693981   # -1.9%  n_dr_0p2_0p4 < 5 and zdr_4 < 0.006795
        - 0.01797516 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0) / 3.186294   # -1.8%  n_dr_0p2_0p4 < 5 and n_dr_0p1_0p2 > 9
        - 0.01788403 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5) / 0.004484684   # -1.8%  n_dr_0p2_0p4 < 5 and zdr_5 < 0.007099
        - 0.01613151 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243) / 0.0001457263   # -1.6%  tau21 < 0.3862 and lam1 > 0.007671
        - 0.01367468 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -1.4%  mass < 53.87
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 10.4;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.39736 * (0.06650051
        + 0.09634525 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.07479306   # +9.6%  mass < 101 and lam2 < 0.003688
        - 0.09592942 * max(0.0, 87.36377 - Q.mass) / 14.19996   # -9.6%  mass < 87.36
        - 0.08557032 * max(0.0, 94.64253 - Q.mass_top40) / 21.02021   # -8.6%  mass_top40 < 94.64
        + 0.08305833 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +8.3%  mass < 121.4
        + 0.08105692 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # +8.1%  mass_top40 < 83.33
        - 0.07254013 * max(0.0, 79.65241 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.03386155   # -7.3%  mass < 79.65 and lam2 < 0.003688
        - 0.05154059 * max(0.0, 74.25181 - Q.mass) / 8.587064   # -5.2%  mass < 74.25
        - 0.0425556 * max(0.0, Q.n_particles - 22.0) / 23.9717   # -4.3%  n_particles > 22
        - 0.04198626 * max(0.0, 121.737 - Q.mass_top30) / 46.42587   # -4.2%  mass_top30 < 121.7
        + 0.04171444 * max(0.0, 5.378975 - Q.D2) / 2.781156   # +4.2%  D2 < 5.379
        - 0.0416791 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 6.916121 - Q.D2) / 32.02927   # -4.2%  mass_top40 < 83.33 and D2 < 6.916
        - 0.03084401 * max(0.0, 0.004855289 - Q.girth2_top15) / 0.001418414   # -3.1%  girth2_top15 < 0.004855
        + 0.02253212 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +2.3%  mass < 101
        + 0.02080927 * max(0.0, 69.02716 - Q.mass_top15) / 18.23029   # +2.1%  mass_top15 < 69.03
        + 0.01991419 * max(0.0, Q.n_dr_0_0p05 - 5.0) / 8.271047   # +2.0%  n_dr_0_0p05 > 5
        + 0.01847648 * max(0.0, 79.65241 - Q.mass) / 10.37103   # +1.8%  mass < 79.65
        + 0.01748526 * max(0.0, Q.e2_sq - 0.01396296) / 0.002220692   # +1.7%  e2_sq > 0.01396
        + 0.01642698 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +1.6%  n_dr_0p2_0p4 < 15
        - 0.0149052 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # -1.5%  girth2_top10 < 0.007679
        - 0.01397576 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # -1.4%  lam2 < 0.001776
        - 0.01344497 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # -1.3%  mass_top40 < 67.73
        - 0.01267369 * max(0.0, 1042.609 - Q.sum_pt) / 35.65948   # -1.3%  sum_pt < 1043
        + 0.01021077 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +1.0%  psi_0p3 > 0.9974
        - 0.007640208 * max(0.0, 1.0 - Q.sd_nremoved) / 0.5731092   # -0.8%  sd_nremoved < 1
        + 0.006133966 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, 1053.047 - Q.sum_pt_top40) / 2417.732   # +0.6%  mass_top30 < 121.7 and sum_pt_top40 < 1053
        - 0.00598854 * max(0.0, Q.girth2_top15 - 0.007887677) / 0.002755219   # -0.6%  girth2_top15 > 0.007888
        + 0.005912701 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +0.6%  mass_top30 < 60.44
        + 0.005411651 * max(0.0, 0.004855289 - Q.girth2_top15) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 0.007590625   # +0.5%  girth2_top15 < 0.004855 and n_dr_0p05_0p1 > 2
        - 0.005321991 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # -0.5%  sj2_dr > 0.2232
        - 0.004927304 * max(0.0, Q.psi_0p1 - 0.8747961) / 0.03440015   # -0.5%  psi_0p1 > 0.8748
        + 0.004539823 * max(0.0, Q.sj3_pair_mass_min - 32.51366) / 5.88267   # +0.5%  sj3_pair_mass_min > 32.51
        - 0.003139714 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.z_top50_slots) / 0.03204382   # -0.3%  n_dr_0p2_0p4 < 15 and z_top50_slots < 1
        - 0.002721003 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.zdr_0 - 0.006864207) / 0.008666302   # -0.3%  mass < 87.36 and zdr_0 > 0.006864
        - 0.002588053 * max(0.0, Q.mass_top10 - 76.9886) / 2.774301   # -0.3%  mass_top10 > 76.99
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 13.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.24374 * (0.1030761
        + 0.1725425 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +17.3%  sum_pt_top50 > 934.2
        - 0.08147507 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -8.1%  log_sum_pt > 6.91
        + 0.07426353 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +7.4%  sum_pt > 907.9
        - 0.07121746 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # -7.1%  sum_pt > 986.1
        - 0.07022586 * max(0.0, 150.0144 - Q.mass_top40) / 66.86198   # -7.0%  mass_top40 < 150
        + 0.04875104 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +4.9%  n_particles < 64
        - 0.03542514 * max(0.0, Q.z_top30_slots - 0.9048492) / 0.05621641   # -3.5%  z_top30_slots > 0.9048
        - 0.03409203 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # -3.4%  tau1 < 0.1507
        - 0.03386085 * max(0.0, Q.mass_over_sum_pt - 0.09046749) / 0.01421039   # -3.4%  mass_over_sum_pt > 0.09047
        - 0.03316803 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -3.3%  log_sum_pt > 6.92
        + 0.03194091 * max(0.0, Q.sd_mass - 69.65633) / 11.73272   # +3.2%  sd_mass > 69.66
        - 0.02456898 * max(0.0, Q.sd_mass - 83.30647) / 6.989667   # -2.5%  sd_mass > 83.31
        + 0.02232373 * max(0.0, Q.n_pt_above_1 - 28.0) / 14.80415   # +2.2%  n_pt_above_1 > 28
        + 0.02156596 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # +2.2%  e2 < 0.04083
        - 0.01944452 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -1.9%  max_dr > 0.2405
        + 0.01743759 * max(0.0, Q.log_sum_pt - 6.935549) / 0.0371538   # +1.7%  log_sum_pt > 6.936
        - 0.01732486 * max(0.0, 787.6281 - Q.sum_pt_top3) / 328.0371   # -1.7%  sum_pt_top3 < 787.6
        + 0.01640507 * max(0.0, Q.sum_pt_top30 - 933.1875) / 79.88556   # +1.6%  sum_pt_top30 > 933.2
        - 0.01497504 * max(0.0, 0.03700182 - Q.z_dr_0p2_0p4) / 0.0183096   # -1.5%  z_dr_0p2_0p4 < 0.037
        - 0.01355688 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -1.4%  sum_pt_top40 > 1025
        + 0.01345894 * max(0.0, 23.45965 - Q.sj3_mass1) / 9.283354   # +1.3%  sj3_mass1 < 23.46
        - 0.01122788 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2) / 10.81389   # -1.1%  n_particles < 64 and D2 < 2.179
        - 0.01117761 * max(0.0, 0.5100475 - Q.tau21) / 0.1343519   # -1.1%  tau21 < 0.51
        + 0.01085884 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +1.1%  n_dr_0p2_0p4 < 11
        + 0.01079503 * max(0.0, 0.01375115 - Q.z_11) / 0.0009450582   # +1.1%  z_11 < 0.01375
        - 0.01046272 * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 9.82015   # -1.0%  n_dr_0p1_0p2 < 21
        - 0.009894841 * max(0.0, 14.14062 - Q.pt_11) / 0.9740783   # -1.0%  pt_11 < 14.14
        + 0.008907352 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # +0.9%  log_sum_pt > 6.989
        - 0.00831636 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -0.8%  mass_over_sum_pt_sq > 0.0292
        + 0.008125846 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +0.8%  mass_over_sum_pt > 0.1709
        + 0.008091312 * max(0.0, 64.48544 - Q.mass) / 5.935866   # +0.8%  mass < 64.49
        - 0.006964528 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -0.7%  mass_top50 > 157.5
        + 0.005636194 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +0.6%  sum_pt > 907.9 and e4 < 5.851e-08
        + 0.003993688 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2) / 0.001543279   # +0.4%  psi_0p3 > 0.9974 and D2 < 3.345
        - 0.003609819 * max(0.0, 0.707925 - Q.psi_0p1) / 0.07386333   # -0.4%  psi_0p1 < 0.7079
        - 0.003497317 * max(0.0, Q.girth2_top50 - 0.01951641) / 0.001208668   # -0.3%  girth2_top50 > 0.01952
        + 0.00338184 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # +0.3%  mass_top10 > 71.78
        - 0.003252633 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.1958008 - Q.absphi_13) / 0.007588689   # -0.3%  log_sum_pt > 6.91 and absphi_13 < 0.1958
        - 0.001900258 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.2%  sum_pt_top10 > 943.7
        + 0.001020668 * max(0.0, Q.max_dr - 0.4357228) / 0.007711928   # +0.1%  max_dr > 0.4357
        + 0.0008612552 * max(0.0, Q.mass - 172.4888) / 0.7446233   # +0.1%  mass > 172.5
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 16.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.07073 * (0.02235016
        + 0.2148808 * max(0.0, 0.02580859 - Q.e2_sq) / 0.01698103   # +21.5%  e2_sq < 0.02581
        - 0.1099922 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -11.0%  mass < 101
        - 0.09996188 * max(0.0, 121.3913 - Q.mass) / 39.46288   # -10.0%  mass < 121.4
        - 0.08443258 * max(0.0, 0.02412652 - Q.girth2_top30) / 0.01613825   # -8.4%  girth2_top30 < 0.02413
        - 0.07949379 * max(0.0, 0.02550569 - Q.girth2_top50) / 0.01683664   # -7.9%  girth2_top50 < 0.02551
        + 0.0634999 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +6.3%  mass < 92.86
        - 0.05998832 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # -6.0%  mass_over_sum_pt < 0.09795
        + 0.03332102 * max(0.0, 172.4888 - Q.mass) / 83.49222   # +3.3%  mass < 172.5
        + 0.03206716 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +3.2%  mass < 89.74
        - 0.02870371 * max(0.0, 0.01649354 - Q.lam1) / 0.009604223   # -2.9%  lam1 < 0.01649
        + 0.02308072 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # +2.3%  girth2_top30 < 0.008376
        + 0.02233177 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +2.2%  mass < 82.85
        + 0.0214383 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +2.1%  lam1 < 0.007259
        + 0.02135132 * max(0.0, 76.60223 - Q.sj3_pair_mass_min) / 50.46144   # +2.1%  sj3_pair_mass_min < 76.6
        + 0.01805389 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +1.8%  e3 < 0.0003372
        + 0.01690079 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # +1.7%  mass_top50 < 71.8
        + 0.01319409 * max(0.0, 0.003687605 - Q.lam2) / 0.002601874   # +1.3%  lam2 < 0.003688
        - 0.01153054 * max(0.0, 2.178951 - Q.D2) / 0.4648731   # -1.2%  D2 < 2.179
        + 0.005843381 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +0.6%  tau1 < 0.06311
        - 0.005669213 * max(0.0, 0.07952881 - Q.eta_0) * max(0.0, Q.n_dr_0_0p05 - 5.0) / 0.6601925   # -0.6%  eta_0 < 0.07953 and n_dr_0_0p05 > 5
        + 0.004778806 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.5%  e2 > 0.04755
        - 0.003535186 * max(0.0, 0.001425993 - Q.girth2_top2) / 0.0004557017   # -0.4%  girth2_top2 < 0.001426
        - 0.003426961 * max(0.0, 89.74183 - Q.mass) * max(0.0, 1129.275 - Q.sum_pt_top20) / 2511.831   # -0.3%  mass < 89.74 and sum_pt_top20 < 1129
        - 0.003106574 * max(0.0, Q.e2 - 0.02793599) * max(0.0, 6.06065 - Q.ptdr0_13) / 0.02788669   # -0.3%  e2 > 0.02794 and ptdr0_13 < 6.061
        - 0.002830879 * max(0.0, Q.sj3_mass1 - 19.30204) / 2.042969   # -0.3%  sj3_mass1 > 19.3
        + 0.002813558 * max(0.0, 0.00287991 - Q.girth2_top20) / 0.000559956   # +0.3%  girth2_top20 < 0.00288
        - 0.00274588 * max(0.0, Q.sj3_dr_min - 0.1204829) / 0.02211875   # -0.3%  sj3_dr_min > 0.1205
        + 0.002700087 * max(0.0, 121.3913 - Q.mass) * max(0.0, 1003.544 - Q.sum_pt_top50) / 688.1559   # +0.3%  mass < 121.4 and sum_pt_top50 < 1004
        - 0.002467629 * max(0.0, Q.soft10_dr - 0.2075213) / 0.02066177   # -0.2%  soft10_dr > 0.2075
        + 0.001881409 * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) * max(0.0, Q.soft10_dr0 - 0.1909669) / 0.004902464   # +0.2%  z_dr_0p1_0p2 < 0.2865 and soft10_dr0 > 0.191
        - 0.001510199 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.z_14 - 0.01634243) / 5.402196e-06   # -0.2%  e2 > 0.04755 and z_14 > 0.01634
        + 0.000919417 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) / 0.293679   # +0.1%  n_dr_0p1_0p2 > 33
        - 0.0007909305 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) * max(0.0, 9.0 - Q.n_dr_0_0p05) / 2.271573   # -0.1%  n_dr_0p1_0p2 > 33 and n_dr_0_0p05 < 9
        + 0.0007570506 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.psi_0p3 - 0.9896594) / 5.449059e-06   # +0.1%  e2 > 0.04755 and psi_0p3 > 0.9897
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 33.89;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.88632 * (-0.006565295
        - 0.119819 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -12.0%  mass < 92.86
        + 0.08881271 * max(0.0, 91.03469 - Q.mass) / 16.36287   # +8.9%  mass < 91.03
        + 0.08079158 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.7612129   # +8.1%  mass < 101 and psi_0p3 > 0.9638
        - 0.07701617 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5305504   # -7.7%  mass < 91.03 and psi_0p3 > 0.9638
        + 0.05283812 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # +5.3%  LHA < 0.3098
        + 0.04766427 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.02554674   # +4.8%  mass < 101 and psi_0p3 > 0.9974
        - 0.04334972 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.01682193   # -4.3%  mass < 91.03 and psi_0p3 > 0.9974
        - 0.03947259 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -3.9%  tau1 < 0.1073
        - 0.03870354 * max(0.0, 0.003418057 - Q.girth2_top50) / 0.0004594629   # -3.9%  girth2_top50 < 0.003418
        - 0.03495038 * max(0.0, 73.35236 - Q.mass_top20) / 16.12893   # -3.5%  mass_top20 < 73.35
        - 0.03478969 * max(0.0, 0.008190222 - Q.width) / 0.002454223   # -3.5%  width < 0.00819
        + 0.03397694 * max(0.0, 70.42121 - Q.mass_top20) / 14.59536   # +3.4%  mass_top20 < 70.42
        + 0.02815817 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +2.8%  mass < 121.4
        - 0.02619558 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -2.6%  mass < 101
        + 0.0260708 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # +2.6%  girth2 < 0.007877
        - 0.02345456 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -2.3%  mass < 82.85
        - 0.02250203 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -2.3%  lam1 < 0.008242
        + 0.02200463 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +2.2%  mass_over_sum_pt < 0.1182
        - 0.01911924 * max(0.0, 0.2845608 - Q.LHA) / 0.05172777   # -1.9%  LHA < 0.2846
        - 0.01513843 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # -1.5%  LHA < 0.3332
        + 0.01426732 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # +1.4%  mass_over_sum_pt_sq < 0.009595
        + 0.01371108 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # +1.4%  tau1 < 0.08787
        - 0.01347304 * max(0.0, 78.26182 - Q.mass) / 9.857173   # -1.3%  mass < 78.26
        + 0.01161769 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +1.2%  tau1 < 0.07709
        - 0.01043873 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # -1.0%  LHA < 0.2601
        + 0.009859892 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # +1.0%  lam1 < 0.00619
        - 0.007917174 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.8%  psi_0p3 > 0.998
        + 0.005381308 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +0.5%  tau21_b2 < 0.2352
        - 0.004895734 * max(0.0, Q.z_top20_slots - 0.9102775) / 0.02689878   # -0.5%  z_top20_slots > 0.9103
        + 0.004596961 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.5%  n_dr_0p2_0p4 < 6
        - 0.004512042 * max(0.0, 0.006403325 - Q.girth2) / 0.001406912   # -0.5%  girth2 < 0.006403
        + 0.004091283 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.01175018   # +0.4%  mass < 82.85 and psi_0p3 > 0.9974
        - 0.003006106 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt) / 11.66143   # -0.3%  tau21_b2 < 0.2352 and sum_pt < 1261
        + 0.002643969 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p3 - 0.9973959) / 4.396153e-05   # +0.3%  mass_over_sum_pt < 0.1182 and psi_0p3 > 0.9974
        - 0.002183554 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3) / 6.520272e-05   # -0.2%  n_dr_0p2_0p4 < 6 and e3 < 7.876e-05
        + 0.002066699 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.9973959 - Q.psi_0p3) / 0.03659629   # +0.2%  mass < 91.03 and psi_0p3 < 0.9974
        + 0.001903069 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.sd_mass - 76.29481) / 33.79048   # +0.2%  mass < 121.4 and sd_mass > 76.29
        - 0.0018207 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -0.2%  tau21 < 0.3862
        - 0.001787109 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.sd_mass - 55.96662) / 29.57136   # -0.2%  mass < 91.03 and sd_mass > 55.97
        + 0.001472738 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 11.65938 - Q.sj3_mass3) / 0.4349059   # +0.1%  tau21_b2 < 0.2352 and sj3_mass3 < 11.66
        - 0.001013001 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.orientation_deg - -9.840088) / 0.01845442   # -0.1%  psi_0p3 > 0.998 and orientation_deg > -9.84
        - 0.0008652314 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.pair_mass_0_3 - 6.624378) / 0.003511398   # -0.1%  psi_0p3 > 0.998 and pair_mass_0_3 > 6.624
        + 0.0008306987 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.zdr_3 - 0.00396157) / 9.828894e-07   # +0.1%  psi_0p3 > 0.998 and zdr_3 > 0.003962
        + 0.0008167362 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9299135) / 1.084471   # +0.1%  mass < 91.03 and psi_0p3 > 0.9299
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 18.21;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.20938 * (-0.04715511
        - 0.07693402 * max(0.0, Q.e2_sq - 0.009606007) / 0.003182984   # -7.7%  e2_sq > 0.009606
        + 0.07192301 * max(0.0, 0.007887677 - Q.girth2_top15) / 0.00326329   # +7.2%  girth2_top15 < 0.007888
        - 0.06736546 * max(0.0, Q.mass - 101.0497) / 12.31084   # -6.7%  mass > 101
        + 0.0644377 * max(0.0, Q.mass - 64.48544) / 31.19164   # +6.4%  mass > 64.49
        - 0.05236202 * max(0.0, Q.mass_top50 - 80.35535) / 18.59691   # -5.2%  mass_top50 > 80.36
        + 0.04928351 * max(0.0, Q.mass - 80.78464) / 19.8061   # +4.9%  mass > 80.78
        + 0.04915961 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # +4.9%  mass_over_sum_pt > 0.07697
        - 0.04762459 * max(0.0, 0.01563836 - Q.girth2_top15) / 0.009593305   # -4.8%  girth2_top15 < 0.01564
        + 0.04475409 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +4.5%  e3 < 0.0003372
        + 0.04346713 * max(0.0, Q.e2_sq - 0.007872294) / 0.00365949   # +4.3%  e2_sq > 0.007872
        - 0.04222858 * max(0.0, 0.02737453 - Q.girth2_top20) / 0.01974268   # -4.2%  girth2_top20 < 0.02737
        - 0.04195956 * max(0.0, Q.girth2_top40 - 0.006026828) / 0.00421794   # -4.2%  girth2_top40 > 0.006027
        + 0.04016649 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266) / 0.00762696   # +4.0%  mass_over_sum_pt_sq > 0.001785
        - 0.03396617 * max(0.0, Q.sum_pt_top40 - 858.8262) / 166.7848   # -3.4%  sum_pt_top40 > 858.8
        + 0.02980637 * max(0.0, Q.mass_top50 - 97.93004) / 11.91764   # +3.0%  mass_top50 > 97.93
        + 0.02776608 * max(0.0, Q.girth2_top15 - 0.001319197) / 0.006270373   # +2.8%  girth2_top15 > 0.001319
        + 0.02767425 * max(0.0, Q.girth2_top40 - 0.008031986) / 0.003375597   # +2.8%  girth2_top40 > 0.008032
        - 0.0195468 * max(0.0, Q.tau1 - 0.06310829) / 0.03403397   # -2.0%  tau1 > 0.06311
        + 0.01927026 * max(0.0, Q.girth2_top40 - 0.005196966) / 0.004725809   # +1.9%  girth2_top40 > 0.005197
        + 0.01263309 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4) / 0.08975043   # +1.3%  z_dr_0p2_0p4 < 0.1292
        - 0.01258998 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -1.3%  n_dr_0p2_0p4 < 15
        + 0.0113794 * max(0.0, 6.935549 - Q.log_sum_pt) / 0.02809622   # +1.1%  log_sum_pt < 6.936
        - 0.01009137 * max(0.0, 0.005383629 - Q.girth2_top15) / 0.001666876   # -1.0%  girth2_top15 < 0.005384
        + 0.008640184 * max(0.0, 1003.544 - Q.sum_pt_top50) / 20.02299   # +0.9%  sum_pt_top50 < 1004
        - 0.008226024 * max(0.0, 1032.405 - Q.sum_pt_top40) / 43.16371   # -0.8%  sum_pt_top40 < 1032
        + 0.007989045 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) / 3.639534   # +0.8%  n_dr_0p2_0p4 > 8
        - 0.007979228 * max(0.0, Q.mass - 121.3913) / 7.81281   # -0.8%  mass > 121.4
        + 0.007568767 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # +0.8%  log_sum_pt > 6.959
        - 0.007005175 * max(0.0, Q.mass_top20 - 90.4505) / 6.038147   # -0.7%  mass_top20 > 90.45
        - 0.006605526 * max(0.0, 0.007887677 - Q.girth2_top15) * max(0.0, 0.6133424 - Q.tau21_b2) / 0.0006475533   # -0.7%  girth2_top15 < 0.007888 and tau21_b2 < 0.6133
        - 0.004873931 * max(0.0, 0.007856958 - Q.girth2_top30) / 0.002601721   # -0.5%  girth2_top30 < 0.007857
        - 0.004501782 * max(0.0, 1032.405 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477) / 0.1123331   # -0.5%  sum_pt_top40 < 1032 and psi_0p3 > 0.9924
        + 0.004451078 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.4%  sj2_dr > 0.2232
        + 0.00413034 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +0.4%  LHA > 0.3332
        + 0.004057954 * max(0.0, Q.mass - 89.74183) / 15.55572   # +0.4%  mass > 89.74
        + 0.003700203 * max(0.0, Q.mass_top40 - 111.2487) / 7.565029   # +0.4%  mass_top40 > 111.2
        + 0.003605827 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.4%  e2 > 0.04755
        + 0.003144683 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435) / 1.470253   # +0.3%  log_sum_pt > 6.959 and sj3_pair_mass_max > 28.35
        + 0.00311876 * max(0.0, Q.n_pt_above_1 - 54.0) / 1.837408   # +0.3%  n_pt_above_1 > 54
        - 0.002517025 * max(0.0, Q.mass - 143.7876) / 3.946979   # -0.3%  mass > 143.8
        - 0.002350212 * max(0.0, 1003.544 - Q.sum_pt_top50) * max(0.0, 21.0 - Q.n_dr_0p05_0p1) / 202.9375   # -0.2%  sum_pt_top50 < 1004 and n_dr_0p05_0p1 < 21
        + 0.002232963 * max(0.0, Q.n_dr_0p1_0p2 - 19.0) / 1.732133   # +0.2%  n_dr_0p1_0p2 > 19
        + 0.002026778 * max(0.0, 1001.523 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477) / 0.05538252   # +0.2%  sum_pt_top40 < 1002 and psi_0p3 > 0.9924
        + 0.001838558 * max(0.0, Q.C2 - 0.07996447) / 0.01049526   # +0.2%  C2 > 0.07996
        - 0.001537948 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.2%  log_sum_pt < 6.811
        - 0.001508463 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) * max(0.0, 26.76006 - Q.sj3_mass1) / 27.22019   # -0.2%  n_dr_0p2_0p4 > 8 and sj3_mass1 < 26.76
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 6.599;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.598507 * (-0.02624213
        - 0.1662109 * max(0.0, 152.6883 - Q.mass_top30) / 74.18247   # -16.6%  mass_top30 < 152.7
        + 0.1153374 * max(0.0, 121.737 - Q.mass_top30) / 46.42587   # +11.5%  mass_top30 < 121.7
        + 0.09215405 * max(0.0, 87.36377 - Q.mass) / 14.19996   # +9.2%  mass < 87.36
        + 0.07120066 * max(0.0, 79.65241 - Q.mass) / 10.37103   # +7.1%  mass < 79.65
        - 0.06083428 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -6.1%  mass < 64.49
        + 0.05867563 * max(0.0, Q.girth - 0.08589404) / 0.01080971   # +5.9%  girth > 0.08589
        + 0.05841521 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # +5.8%  girth2_top40 < 0.00626
        + 0.05319771 * max(0.0, 0.2091025 - Q.LHA) / 0.02095782   # +5.3%  LHA < 0.2091
        - 0.04621471 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -4.6%  tau1 < 0.05445
        - 0.04185727 * max(0.0, Q.mass - 143.7876) / 3.946979   # -4.2%  mass > 143.8
        - 0.03159773 * max(0.0, Q.z_top40_slots - 0.9574183) / 0.02831517   # -3.2%  z_top40_slots > 0.9574
        + 0.02899955 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # +2.9%  girth2_top20 < 0.01083
        + 0.02459207 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +2.5%  lam1 > 0.01174
        + 0.02109012 * max(0.0, 0.2187642 - Q.z_dr_0p1_0p2) / 0.1086627   # +2.1%  z_dr_0p1_0p2 < 0.2188
        - 0.01938649 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # -1.9%  lam1 < 0.004673
        - 0.01719523 * max(0.0, 89.6788 - Q.mass_top40) / 17.47899   # -1.7%  mass_top40 < 89.68
        + 0.01642494 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 934.2416 - Q.sum_pt_top50) / 121.4929   # +1.6%  mass_top40 < 80.89 and sum_pt_top50 < 934.2
        - 0.01435283 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt) / 447.38   # -1.4%  mass_top40 < 80.89 and sum_pt < 1035
        - 0.01410925 * max(0.0, Q.girth2_top30 - 0.02412652) / 0.0004997242   # -1.4%  girth2_top30 > 0.02413
        + 0.009517129 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +1.0%  mass_top40 < 80.89
        + 0.00919639 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.n_particles - 38.0) / 73.85286   # +0.9%  mass < 87.36 and n_particles > 38
        + 0.007064797 * max(0.0, Q.mass - 172.4888) / 0.7446233   # +0.7%  mass > 172.5
        + 0.005444882 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, Q.n_dr_0p2_0p4 - 10.0) / 39.79287   # +0.5%  mass_top30 < 121.7 and n_dr_0p2_0p4 > 10
        + 0.004676774 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # +0.5%  sum_pt_top50 < 959.1
        - 0.004002365 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, 20.0 - Q.n_pt_above_10) / 29.14649   # -0.4%  sum_pt_top40 < 956.2 and n_pt_above_10 < 20
        - 0.003585565 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.894531) / 2.293712   # -0.4%  sum_pt_top40 < 956.2 and soft5_pt > 2.895
        + 0.00269368 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258) / 5.834847   # +0.3%  sum_pt_top40 < 956.2 and soft4_pt > 1.789
        + 0.001076916 * max(0.0, Q.lam1 - 0.01174405) * max(0.0, Q.soft5_z - 0.002346473) / 4.122063e-07   # +0.1%  lam1 > 0.01174 and soft5_z > 0.002346
        + 0.0008954943 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft5_pt - 2.894531) / 1.196279   # +0.1%  sum_pt_top50 < 959.1 and soft5_pt > 2.895
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 19.2;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.20085 * (-0.205666
        + 0.1987602 * Q.sum_pt / 1043.796   # +19.9%  sum_pt
        - 0.09710344 * max(0.0, Q.mass - 64.48544) / 31.19164   # -9.7%  mass > 64.49
        + 0.08699256 * max(0.0, Q.mass - 82.85409) / 18.72167   # +8.7%  mass > 82.85
        + 0.07564391 * max(0.0, 132.4278 - Q.mass_top40) / 51.50818   # +7.6%  mass_top40 < 132.4
        - 0.05453694 * max(0.0, 80.3008 - Q.sj2_mass1) / 50.41572   # -5.5%  sj2_mass1 < 80.3
        + 0.05227408 * max(0.0, 0.01807679 - Q.girth2_top30) / 0.01083719   # +5.2%  girth2_top30 < 0.01808
        - 0.04112536 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -4.1%  mass_top40 < 83.33
        - 0.04111192 * max(0.0, 0.06524004 - Q.e2) / 0.03476195   # -4.1%  e2 < 0.06524
        + 0.03725581 * max(0.0, 87.27603 - Q.mass_top40) / 15.9837   # +3.7%  mass_top40 < 87.28
        + 0.03593019 * max(0.0, Q.mass_top40 - 67.72643) / 24.79585   # +3.6%  mass_top40 > 67.73
        - 0.02580309 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -2.6%  girth < 0.1207
        + 0.02431001 * max(0.0, 839.9547 - Q.sum_pt_top5) / 255.1112   # +2.4%  sum_pt_top5 < 840
        - 0.02222641 * max(0.0, Q.log_sum_pt - 6.903423) / 0.05662275   # -2.2%  log_sum_pt > 6.903
        + 0.01979936 * max(0.0, 0.6133424 - Q.tau21_b2) / 0.3099701   # +2.0%  tau21_b2 < 0.6133
        + 0.01839282 * max(0.0, 3.814159 - Q.D2) / 1.519965   # +1.8%  D2 < 3.814
        - 0.01663487 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -1.7%  tau1 < 0.05445
        - 0.01643413 * max(0.0, Q.mass - 143.7876) / 3.946979   # -1.6%  mass > 143.8
        + 0.01288306 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +1.3%  mass < 74.25
        + 0.01275944 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.3%  z_dr_0p2_0p4 < 0.09123
        + 0.01055194 * max(0.0, Q.mass - 64.48544) * max(0.0, 2.537109 - Q.soft2_pt) / 47.07428   # +1.1%  mass > 64.49 and soft2_pt < 2.537
        - 0.009792001 * max(0.0, 71.781 - Q.mass_top10) / 28.28521   # -1.0%  mass_top10 < 71.78
        - 0.009574988 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -1.0%  n_dr_0p2_0p4 < 11
        + 0.00907154 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +0.9%  dr_0 < 0.06413
        - 0.008964874 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # -0.9%  mass_over_sum_pt > 0.1709
        + 0.007683175 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # +0.8%  mass_over_sum_pt_sq > 0.0292
        + 0.007127648 * max(0.0, Q.mass_top5 - 14.54404) / 16.7137   # +0.7%  mass_top5 > 14.54
        + 0.00702924 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, 464.75 - Q.sum_pt_top2) / 5683.5   # +0.7%  sj2_mass1 < 80.3 and sum_pt_top2 < 464.8
        + 0.005894609 * max(0.0, 0.06940472 - Q.dr_1) / 0.0280884   # +0.6%  dr_1 < 0.0694
        - 0.005581933 * max(0.0, 3.814159 - Q.D2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.04343614   # -0.6%  D2 < 3.814 and sj2_dr > 0.1938
        - 0.004880653 * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0004356739   # -0.5%  psi_0p3 > 0.9985
        - 0.004749277 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # -0.5%  girth2_top30 < 0.005402
        - 0.004337181 * max(0.0, 75.26407 - Q.mass_top15) / 22.29039   # -0.4%  mass_top15 < 75.26
        - 0.003788235 * max(0.0, Q.z_dr_0_0p05 - 0.7674734) / 0.05221088   # -0.4%  z_dr_0_0p05 > 0.7675
        - 0.003179533 * max(0.0, Q.mass - 162.8363) / 1.509832   # -0.3%  mass > 162.8
        + 0.002299343 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.0830523 - Q.dr_4) / 0.002292504   # +0.2%  log_sum_pt > 6.903 and dr_4 < 0.08305
        + 0.00171885 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.91979) / 145.9595   # +0.2%  sj2_mass1 < 80.3 and sj2_mass2 > 11.92
        + 0.001220725 * max(0.0, 111.2487 - Q.mass_top40) * max(0.0, Q.pt1_dr01 - 12.6865) / 28.78446   # +0.1%  mass_top40 < 111.2 and pt1_dr01 > 12.69
        - 0.00109792 * max(0.0, Q.mass_top40 - 163.2541) / 0.6399318   # -0.1%  mass_top40 > 163.3
        + 0.0008843807 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # +0.1%  mass_top50 > 157.5
        + 0.0005943685 * max(0.0, Q.mass_top40 - 163.2541) * max(0.0, Q.soft4_z - 0.001186042) / 0.0003797437   # +0.1%  mass_top40 > 163.3 and soft4_z > 0.001186
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 8.223;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.223297 * (-0.03381096
        + 0.1416569 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +14.2%  mass < 92.86
        - 0.07933641 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # -7.9%  girth < 0.07374
        - 0.07380023 * max(0.0, 85.8667 - Q.mass_top50) / 13.95835   # -7.4%  mass_top50 < 85.87
        + 0.0699163 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # +7.0%  mass_over_sum_pt_sq < 0.009595
        + 0.06941408 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +6.9%  e2_sq < 0.008184
        - 0.06883779 * max(0.0, 0.006929741 - Q.girth2_top30) / 0.002004687   # -6.9%  girth2_top30 < 0.00693
        + 0.06551749 * max(0.0, 0.02793599 - Q.e2) / 0.005715277   # +6.6%  e2 < 0.02794
        + 0.06526458 * max(0.0, 0.2845608 - Q.LHA) / 0.05172777   # +6.5%  LHA < 0.2846
        - 0.06504808 * max(0.0, 79.65241 - Q.mass) / 10.37103   # -6.5%  mass < 79.65
        - 0.05438186 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -5.4%  lam1 < 0.006717
        - 0.0424223 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # -4.2%  e2 < 0.04083
        + 0.03930229 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +3.9%  n_dr_0p2_0p4 < 10
        + 0.01810929 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +1.8%  mass_top30 < 60.44
        + 0.0178964 * max(0.0, 0.003213724 - Q.girth2_top10) / 0.0009423838   # +1.8%  girth2_top10 < 0.003214
        - 0.01762489 * max(0.0, Q.z_top10_slots - 0.7271951) / 0.0653343   # -1.8%  z_top10_slots > 0.7272
        - 0.01635526 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2) / 0.005195774   # -1.6%  n_dr_0p2_0p4 < 10 and girth2 < 0.005532
        + 0.01414322 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +1.4%  n_dr_0p2_0p4 < 6
        - 0.01211226 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # -1.2%  e2 < 0.02516
        + 0.01158863 * max(0.0, 15.0 - Q.n_dr_0p1_0p2) / 5.139044   # +1.2%  n_dr_0p1_0p2 < 15
        + 0.009730084 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 0.05112769 - Q.C2_b2) / 0.0003814199   # +1.0%  z_top30_slots > 0.9735 and C2_b2 < 0.05113
        + 0.009604162 * max(0.0, 0.1548383 - Q.z_dr_0p1_0p2) / 0.06710802   # +1.0%  z_dr_0p1_0p2 < 0.1548
        - 0.00847848 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0) / 30.16986   # -0.8%  n_dr_0p2_0p4 < 10 and n_particles > 34
        + 0.007622321 * max(0.0, 0.05048381 - Q.girth) / 0.008533025   # +0.8%  girth < 0.05048
        + 0.007541894 * max(0.0, Q.psi_0p2 - 0.9935324) / 0.001465572   # +0.8%  psi_0p2 > 0.9935
        + 0.00575779 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.mass_top10 - 38.52415) / 111.7968   # +0.6%  mass < 101 and mass_top10 > 38.52
        - 0.004788218 * max(0.0, 79.65241 - Q.mass) * max(0.0, 3.814159 - Q.D2) / 4.810976   # -0.5%  mass < 79.65 and D2 < 3.814
        + 0.00374877 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 43.23581   # +0.4%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 3.477;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.476833 * (-0.05735687
        + 0.2339659 * max(0.0, 80.78464 - Q.mass) / 10.84952   # +23.4%  mass < 80.78
        - 0.2040309 * max(0.0, 94.64253 - Q.mass_top40) / 21.02021   # -20.4%  mass_top40 < 94.64
        + 0.1761206 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +17.6%  mass < 82.85
        + 0.0559316 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # +5.6%  z_dr_0p2_0p4 < 0.06849
        - 0.05325749 * max(0.0, 0.00363788 - Q.e2_sq) / 0.0004963098   # -5.3%  e2_sq < 0.003638
        - 0.05088923 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -5.1%  mass < 53.87
        + 0.04996393 * max(0.0, 87.36377 - Q.mass) / 14.19996   # +5.0%  mass < 87.36
        + 0.03703351 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # +3.7%  n_dr_0p2_0p4 < 7
        - 0.03195092 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3) / 0.0414109   # -3.2%  mass < 87.36 and psi_0p3 < 0.999
        - 0.01728053 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.6147674   # -1.7%  mass < 87.36 and z_dr_0p1_0p2 < 0.06473
        - 0.01197141 * max(0.0, Q.z_dr_0_0p05 - 0.9084912) / 0.009227347   # -1.2%  z_dr_0_0p05 > 0.9085
        - 0.01122678 * max(0.0, 77.93668 - Q.mass_top40) / 11.12028   # -1.1%  mass_top40 < 77.94
        - 0.01069763 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -1.1%  sum_pt < 972
        + 0.009272021 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.3036026 - Q.planar_flow) / 0.0008856905   # +0.9%  z_dr_0p1_0p2 > 0.4479 and planar_flow < 0.3036
        - 0.008854795 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.001776308 - Q.lam2) / 5.387726e-06   # -0.9%  z_dr_0p1_0p2 > 0.4479 and lam2 < 0.001776
        - 0.008633374 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.lam2 - 0.0003497174) / 0.002699158   # -0.9%  mass < 87.36 and lam2 > 0.0003497
        + 0.008162502 * max(0.0, 77.93668 - Q.mass_top40) * max(0.0, Q.lam2 - 0.000404306) / 0.002110211   # +0.8%  mass_top40 < 77.94 and lam2 > 0.0004043
        + 0.006226051 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) / 0.02385119   # +0.6%  z_dr_0p1_0p2 > 0.4479
        + 0.004680198 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 435.25 - Q.pt_0) / 490.7786   # +0.5%  sj3_pair_mass_max > 122.7 and pt_0 < 435.2
        + 0.003474027 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1) / 9.827377   # +0.3%  sd_mass > 133.3 and sj3_mass1 < 32.5
        - 0.003208067 * max(0.0, Q.sum_pt - 1260.541) / 7.655483   # -0.3%  sum_pt > 1261
        - 0.003168466 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.soft1_pt - 0.8144531) / 0.3040698   # -0.3%  n_dr_0p2_0p4 < 7 and soft1_pt > 0.8145
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 6.239;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.239475 * (0.1897459
        - 0.1125481 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -11.3%  sum_pt < 1013
        - 0.1087568 * max(0.0, Q.mass - 143.7876) / 3.946979   # -10.9%  mass > 143.8
        + 0.08960563 * max(0.0, 0.02497133 - Q.girth2_top40) / 0.01655385   # +9.0%  girth2_top40 < 0.02497
        - 0.0722386 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -7.2%  sum_pt < 1085
        + 0.06993488 * max(0.0, Q.mass - 64.48544) / 31.19164   # +7.0%  mass > 64.49
        + 0.05443756 * max(0.0, Q.mass_top50 - 138.8977) / 3.930268   # +5.4%  mass_top50 > 138.9
        + 0.05286041 * max(0.0, Q.mass - 78.26182) / 21.33658   # +5.3%  mass > 78.26
        + 0.05017601 * max(0.0, 1008.935 - Q.sum_pt_top50) / 22.0436   # +5.0%  sum_pt_top50 < 1009
        - 0.04483699 * max(0.0, Q.mass_top50 - 97.93004) / 11.91764   # -4.5%  mass_top50 > 97.93
        + 0.04418446 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +4.4%  sum_pt_top40 < 1053
        - 0.03765745 * max(0.0, Q.mass_over_sum_pt - 0.06030419) / 0.03186977   # -3.8%  mass_over_sum_pt > 0.0603
        + 0.03280284 * max(0.0, Q.psi_0p3 - 0.9777125) / 0.01571542   # +3.3%  psi_0p3 > 0.9777
        - 0.03228147 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -3.2%  mass_top50 > 157.5
        + 0.0305858 * max(0.0, Q.mass_top40 - 150.0144) / 1.666872   # +3.1%  mass_top40 > 150
        + 0.01613932 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.5260785 - Q.D3) / 23.2296   # +1.6%  sum_pt_top40 < 1053 and D3 < 0.5261
        - 0.01589565 * max(0.0, 14.0 - Q.n_for_90pct) / 1.135089   # -1.6%  n_for_90pct < 14
        - 0.01581495 * max(0.0, 889.8383 - Q.sum_pt_top20) / 35.81237   # -1.6%  sum_pt_top20 < 889.8
        - 0.01551244 * max(0.0, 19.0 - Q.n_dr_0p1_0p2) / 8.166005   # -1.6%  n_dr_0p1_0p2 < 19
        - 0.01076882 * max(0.0, Q.mass - 172.4888) * max(0.0, 41.4375 - Q.pt_9) / 7.126564   # -1.1%  mass > 172.5 and pt_9 < 41.44
        + 0.01054563 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # +1.1%  tau1 < 0.08787
        - 0.0101258 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -1.0%  log_sum_pt < 6.811
        + 0.009594486 * max(0.0, Q.mass - 172.4888) / 0.7446233   # +1.0%  mass > 172.5
        - 0.00898656 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # -0.9%  mass_top10 > 71.78
        + 0.00853023 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +0.9%  mass_over_sum_pt > 0.1709
        + 0.007785166 * max(0.0, Q.mass_top10 - 60.5496) / 6.712581   # +0.8%  mass_top10 > 60.55
        + 0.007573794 * max(0.0, Q.mass_top5 - 37.76455) / 5.869288   # +0.8%  mass_top5 > 37.76
        + 0.007413252 * max(0.0, 6.879399 - Q.log_sum_pt) / 0.01083309   # +0.7%  log_sum_pt < 6.879
        + 0.005617799 * max(0.0, Q.mass_top50 - 97.93004) * max(0.0, 2.873047 - Q.soft3_pt) / 19.23325   # +0.6%  mass_top50 > 97.93 and soft3_pt < 2.873
        - 0.004251707 * max(0.0, Q.mass_top50 - 168.9698) / 0.6651775   # -0.4%  mass_top50 > 169
        + 0.003661094 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.01833434 - Q.z_9) / 0.01801529   # +0.4%  sum_pt < 1013 and z_9 < 0.01833
        - 0.003428604 * max(0.0, Q.sum_pt - 1260.541) / 7.655483   # -0.3%  sum_pt > 1261
        + 0.003287327 * max(0.0, 14.0 - Q.n_for_90pct) * max(0.0, 3.345339 - Q.D2) / 0.7823248   # +0.3%  n_for_90pct < 14 and D2 < 3.345
        + 0.002160407 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # +0.2%  log_sum_pt > 7.139
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 19.48;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.47701 * (0.00771485
        - 0.1225003 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) / 0.0178873   # -12.3%  mass_over_sum_pt < 0.09047
        - 0.1033709 * max(0.0, 91.03469 - Q.mass) / 16.36287   # -10.3%  mass < 91.03
        + 0.09333457 * max(0.0, 0.01986381 - Q.mass_over_sum_pt_sq) / 0.0117775   # +9.3%  mass_over_sum_pt_sq < 0.01986
        + 0.07406779 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +7.4%  mass_over_sum_pt < 0.1182
        - 0.0697301 * max(0.0, 79.65241 - Q.mass) / 10.37103   # -7.0%  mass < 79.65
        - 0.06108825 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -6.1%  mass < 82.85
        + 0.05266998 * max(0.0, 0.08873143 - Q.mass_over_sum_pt) / 0.01671255   # +5.3%  mass_over_sum_pt < 0.08873
        + 0.04345348 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +4.3%  mass_top40 < 67.73
        - 0.04305263 * max(0.0, 0.01292642 - Q.girth2_top40) / 0.006292908   # -4.3%  girth2_top40 < 0.01293
        - 0.03565359 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -3.6%  girth2_top20 < 0.01083
        + 0.03342428 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +3.3%  mass_over_sum_pt < 0.09795
        + 0.02743994 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.002405315 - Q.soft8_z) / 0.0166304   # +2.7%  mass < 91.03 and soft8_z < 0.002405
        - 0.02397946 * max(0.0, 0.01897915 - Q.girth2_top40) / 0.01130444   # -2.4%  girth2_top40 < 0.01898
        + 0.01932881 * max(0.0, 0.03680582 - Q.e2) / 0.01052402   # +1.9%  e2 < 0.03681
        - 0.01735312 * max(0.0, Q.n_pt_above_10 - 11.0) / 8.944079   # -1.7%  n_pt_above_10 > 11
        + 0.01634425 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 1.059188 - Q.N3) / 0.002548597   # +1.6%  psi_0p3 > 0.9924 and N3 < 1.059
        - 0.01492725 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -1.5%  mass_top40 < 83.33
        + 0.01292072 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +1.3%  tau21_b2 < 0.3425
        - 0.01170542 * max(0.0, 0.1008497 - Q.tau1) / 0.02962944   # -1.2%  tau1 < 0.1008
        - 0.01031371 * max(0.0, 0.001592178 - Q.girth2_top3) / 0.000513961   # -1.0%  girth2_top3 < 0.001592
        + 0.01015457 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +1.0%  mass < 121.4
        - 0.009847311 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # -1.0%  psi_0p3 > 0.9924
        - 0.009537254 * max(0.0, 19.0 - Q.n_dr_0p1_0p2) / 8.166005   # -1.0%  n_dr_0p1_0p2 < 19
        - 0.007904081 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.0458688 - Q.dr_3) / 4.832428e-05   # -0.8%  psi_0p3 > 0.9924 and dr_3 < 0.04587
        + 0.00750172 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # +0.8%  girth2_top20 < 0.006374
        + 0.007113458 * max(0.0, 0.03948167 - Q.dr_3) / 0.0095916   # +0.7%  dr_3 < 0.03948
        - 0.006145429 * max(0.0, 0.4226723 - Q.N2) / 0.1194948   # -0.6%  N2 < 0.4227
        - 0.006022918 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2738063) / 0.009655725   # -0.6%  N2 < 0.4227 and max_dr > 0.2738
        - 0.005022925 * max(0.0, 0.04142826 - Q.tau2) / 0.01568306   # -0.5%  tau2 < 0.04143
        + 0.005019437 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 4.250195 - Q.soft6_pt) / 0.009807099   # +0.5%  psi_0p3 > 0.9924 and soft6_pt < 4.25
        - 0.004941852 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.zdr_3 - 0.0003281655) / 0.08597332   # -0.5%  mass < 121.4 and zdr_3 > 0.0003282
        - 0.004868183 * max(0.0, 0.001595561 - Q.soft7_z) / 0.0003533115   # -0.5%  soft7_z < 0.001596
        + 0.004549468 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +0.5%  e2 < 0.0303
        + 0.004309288 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, Q.pt_3 - 60.59375) / 0.06871383   # +0.4%  psi_0p3 > 0.9924 and pt_3 > 60.59
        + 0.00357552 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.n_for_50pct - 5.0) / 4.765654   # +0.4%  mass < 82.85 and n_for_50pct > 5
        - 0.003420966 * max(0.0, 0.01256572 - Q.e2) / 0.0008030352   # -0.3%  e2 < 0.01257
        + 0.00332508 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.05799644 - Q.z_6) / 0.001993324   # +0.3%  tau21_b2 < 0.3425 and z_6 < 0.058
        - 0.003088164 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.02048524 - Q.lam1) / 0.001304679   # -0.3%  tau21_b2 < 0.3425 and lam1 < 0.02049
        - 0.003001676 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.1231747 - Q.z_1) / 0.00154957   # -0.3%  tau21_b2 < 0.3425 and z_1 < 0.1232
        - 0.001938482 * max(0.0, 0.01083435 - Q.girth2_top20) * max(0.0, Q.mass_top3 - 16.89912) / 0.01539405   # -0.2%  girth2_top20 < 0.01083 and mass_top3 > 16.9
        - 0.001579912 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.34119 - Q.sj2_zsoft) / 0.009756021   # -0.2%  tau21_b2 < 0.3425 and sj2_zsoft < 0.3412
        - 0.000473782 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.mass_top2 - 36.76827) / 5.658434   # -0.0%  mass < 121.4 and mass_top2 > 36.77
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 5.26;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.259905 * (-0.1128166
        + 0.1212085 * max(0.0, 79.18312 - Q.sd_mass) / 30.46368   # +12.1%  sd_mass < 79.18
        + 0.115636 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +11.6%  log_sum_pt < 7.017
        + 0.09981863 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +10.0%  girth2_top5 < 0.00833
        - 0.06241943 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -6.2%  psi_0p3 > 0.9897
        + 0.05796932 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +5.8%  z_dr_0p1_0p2 < 0.1203
        - 0.05049625 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -5.0%  tau1 < 0.07057
        - 0.04863046 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # -4.9%  mass_over_sum_pt > 0.1709
        - 0.04788849 * max(0.0, 40.97891 - Q.sd_mass) / 11.60739   # -4.8%  sd_mass < 40.98
        - 0.03758289 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -3.8%  sum_pt < 1002
        + 0.03416259 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +3.4%  lam2 < 0.001776
        - 0.03184049 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) / 0.3666529   # -3.2%  z_dr_0_0p05 < 0.8459
        - 0.0310785 * max(0.0, 0.002270363 - Q.girth2_top5) / 0.0007634099   # -3.1%  girth2_top5 < 0.00227
        - 0.02373247 * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 3.233647   # -2.4%  n_dr_0p2_0p4 > 9
        - 0.02364301 * max(0.0, 1017.778 - Q.sum_pt_top20) / 107.0228   # -2.4%  sum_pt_top20 < 1018
        + 0.02269326 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # +2.3%  girth2_top10 < 0.007679
        - 0.02052916 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 76.29481 - Q.sd_mass) / 2.433638   # -2.1%  z_dr_0p1_0p2 < 0.1203 and sd_mass < 76.29
        - 0.01652686 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # -1.7%  mass_top50 < 71.8
        - 0.01638885 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -1.6%  log_sum_pt < 6.903
        - 0.01635367 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583) / 0.0009330603   # -1.6%  girth2_top5 < 0.00833 and sj3_pairmin_over_m > 0.09541
        + 0.01295968 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2) / 0.0004272674   # +1.3%  sj2_dr > 0.2232 and C2_b2 < 0.04009
        - 0.01289442 * max(0.0, Q.psi_0p1 - 0.9371031) / 0.01088804   # -1.3%  psi_0p1 > 0.9371
        + 0.01270042 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +1.3%  lam1 < 0.004673
        + 0.0126201 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +1.3%  sj2_dr > 0.2232
        + 0.01239936 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) / 3.82104   # +1.2%  n_dr_0p1_0p2 < 13
        - 0.01188063 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677) / 0.04548726   # -1.2%  girth2_top5 < 0.00833 and sj3_mass1 > 5.113
        + 0.01132911 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +1.1%  sum_pt < 986.1
        + 0.01129225 * max(0.0, 52.15154 - Q.mass_top50) / 3.332261   # +1.1%  mass_top50 < 52.15
        + 0.007015474 * max(0.0, 0.00287991 - Q.girth2_top20) / 0.000559956   # +0.7%  girth2_top20 < 0.00288
        - 0.006866541 * max(0.0, 1018.832 - Q.sum_pt_top30) / 55.61816   # -0.7%  sum_pt_top30 < 1019
        + 0.005289834 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt) / 2.722316   # +0.5%  z_dr_0_0p05 < 0.8459 and sum_pt < 949.9
        + 0.004153429 * max(0.0, 869.693 - Q.sum_pt_top20) / 29.2727   # +0.4%  sum_pt_top20 < 869.7
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6138134978991596, 2.868420430672269, 0.22444642857142857, 0.40298529411764705, 0.7582651260504202, 1.0831048319327732, 0.520673424369748, 0.3841704831932773, 1.128927731092437, 1.0409306722689076, 1.170698949579832, 0.7378827731092437, 0.5442510504201681, 1.541438550420168, 0.23496938025210085, 0.3623517857142857]
T = [3.8961690881039917, 2.4784508042279407, 4.061897541360294, 4.133929569327732, 3.962099458377101]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +46%, n4 -17%, n9 +14%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.4601347 * h[1] / H_AVG[1]
            - 0.1702909 * h[4] / H_AVG[4]
            + 0.1377584 * h[9] / H_AVG[9]
            - 0.08687258 * h[5] / H_AVG[5]
            - 0.07757337 * h[3] / H_AVG[3]
            + 0.03273955 * h[12] / H_AVG[12]
            + 0.02088082 * h[6] / H_AVG[6]
            + 0.00905479 * h[8] / H_AVG[8]
            - 0.004694912 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +24%, n1 -22%, n12 +8%, n11 -7%, n6 +5% ...
            - 0.3250646 * h[4] / H_AVG[4]
            + 0.2362458 * h[9] / H_AVG[9]
            - 0.217002 * h[1] / H_AVG[1]
            + 0.07548518 * h[12] / H_AVG[12]
            - 0.07442984 * h[11] / H_AVG[11]
            + 0.04595504 * h[6] / H_AVG[6]
            + 0.0113199 * h[2] / H_AVG[2]
            - 0.007380486 * h[10] / H_AVG[10]
            + 0.007117146 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +15%, n0 +11%, n11 +11%, n14 -8%, n4 +6% ...
            - 0.2431897 * h[8] / H_AVG[8]
            + 0.154157 * h[5] / H_AVG[5]
            + 0.1133362 * h[0] / H_AVG[0]
            + 0.1078604 * h[11] / H_AVG[11]
            - 0.07953989 * h[14] / H_AVG[14]
            + 0.06417041 * h[4] / H_AVG[4]
            - 0.05911192 * h[7] / H_AVG[7]
            - 0.05605843 * h[9] / H_AVG[9]
            - 0.05443318 * h[12] / H_AVG[12]
            + 0.04340485 * h[3] / H_AVG[3]
            - 0.01672641 * h[15] / H_AVG[15]
            + 0.008011548 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -20%, n5 +18%, n6 -12%, n7 +8%, n15 +5% ...
            - 0.2560203 * h[8] / H_AVG[8]
            - 0.2041625 * h[0] / H_AVG[0]
            + 0.1801275 * h[5] / H_AVG[5]
            - 0.1220152 * h[6] / H_AVG[6]
            + 0.08421878 * h[7] / H_AVG[7]
            + 0.04930487 * h[15] / H_AVG[15]
            + 0.04264854 * h[3] / H_AVG[3]
            + 0.04114208 * h[12] / H_AVG[12]
            - 0.02036015 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -35%, n10 +29%, n5 -13%, n8 +6%, n12 -5%, n4 +4% ...
            - 0.3525728 * h[13] / H_AVG[13]
            + 0.2908576 * h[10] / H_AVG[10]
            - 0.1281405 * h[5] / H_AVG[5]
            + 0.06010278 * h[8] / H_AVG[8]
            - 0.05151162 * h[12] / H_AVG[12]
            + 0.03588368 * h[4] / H_AVG[4]
            - 0.03429543 * h[15] / H_AVG[15]
            - 0.02727038 * h[7] / H_AVG[7]
            + 0.01936516 * h[0] / H_AVG[0]
        ),
    ]]


def classify(pt, eta, phi):
    s = logits(jet_layer_4(quantities(pt, eta, phi)))
    m = max(s)
    e = [math.exp(x - m) for x in s]
    p = [x / sum(e) for x in e]
    return CLASSES[s.index(m)], s, p


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:64] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:64] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:64] + [0.0] * 56
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
