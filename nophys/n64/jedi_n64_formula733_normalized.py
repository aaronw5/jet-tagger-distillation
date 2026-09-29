"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  1:  12.6%   (on for 95% of jets)
  neuron  8:  12.2%   (on for 49% of jets)
  neuron  5:  12.0%   (on for 80% of jets)
  neuron  4:  10.1%   (on for 65% of jets)
  neuron  9:   7.8%   (on for 77% of jets)
  neuron 13:   7.5%   (on for 86% of jets)
  neuron  0:   7.4%   (on for 63% of jets)
  neuron 10:   6.5%   (on for 77% of jets)
  neuron 12:   4.6%   (on for 44% of jets)
  neuron  6:   3.9%   (on for 51% of jets)
  neuron  3:   3.5%   (on for 56% of jets)
  neuron 11:   3.5%   (on for 74% of jets)
  neuron  7:   3.5%   (on for 26% of jets)
  neuron 15:   2.3%   (on for 54% of jets)
  neuron 14:   2.0%   (on for 33% of jets)
  neuron  2:   0.6%   (on for 43% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.4% (the network: 81.1%); same class as the network for 94.2% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
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
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_12         mass of particles 0 and 12 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
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
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.ptdr0_13               pT13 · ΔR(0, 13) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
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
  Q.z_7                    pT of particle 7 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_z                pT share of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.absphi_0               |Δφ| of particle 0
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.soft10_dr              ΔR from the jet axis of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_dr               ΔR from the jet axis of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
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
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
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
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau4                   N-subjettiness τ4 (β=1)
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
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_12=pair_mass(0, 12),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
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
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_particles=len(real),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_0=pt[0],
        pt_11=pt[11],
        ptdr0_13=pt[13] * math.sqrt(dist2(0, 13)) if pt[13] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
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
        z_7=z[7],
        z_9=z[9],
        soft1_z=softp(1, 'z'),
        soft3_z=softp(3, 'z'),
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft8_z=softp(8, 'z'),
        sj3_z3=subjets(3)["z"][2],
        z_top10_slots=sum(pt[:10]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_mass=softdrop("mass"),
        absphi_0=abs(phi[0]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        soft10_dr=softp(10, 'dr'),
        soft3_dr=softp(3, 'dr'),
        eta_0=eta[0],
        eta_1=eta[1],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
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
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
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
        tau3=tau_n(3),
        tau4=tau_n(4),
    )


def neuron_0(Q):
    # scale S = 15.27;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.26989 * (0.08704956
        - 0.1921325 * max(0.0, Q.mass - 78.26182) / 21.33658   # -19.2%  mass > 78.26
        - 0.1483187 * max(0.0, Q.mass - 92.85979) / 14.48273   # -14.8%  mass > 92.86
        + 0.08626715 * max(0.0, Q.mass - 87.36377) / 16.57741   # +8.6%  mass > 87.36
        - 0.0533676 * max(0.0, Q.mass - 74.25181) / 24.07648   # -5.3%  mass > 74.25
        - 0.04336044 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -4.3%  mass < 101
        + 0.04155473 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq) / 0.001689525   # +4.2%  mass_over_sum_pt_sq < 0.006939
        + 0.04068585 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 0.002241068   # +4.1%  mass_over_sum_pt_sq < 0.007873
        - 0.03062007 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -3.1%  e2_sq < 0.006167
        - 0.02910385 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # -2.9%  log_sum_pt < 6.989
        - 0.02731982 * max(0.0, Q.mass - 91.03469) / 15.0694   # -2.7%  mass > 91.03
        + 0.02354177 * max(0.0, Q.mass_top50 - 71.79516) / 24.22501   # +2.4%  mass_top50 > 71.8
        - 0.02236613 * max(0.0, Q.mass_top50 - 92.16545) / 13.44564   # -2.2%  mass_top50 > 92.17
        + 0.02208229 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +2.2%  sum_pt_top50 < 1157
        + 0.02018399 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +2.0%  mass_top40 < 80.89
        + 0.01862861 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +1.9%  log_sum_pt < 7.017
        - 0.0184271 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -1.8%  mass_top40 < 83.33
        + 0.01757964 * max(0.0, 6.97212 - Q.log_sum_pt) / 0.05249646   # +1.8%  log_sum_pt < 6.972
        - 0.01711376 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -1.7%  tau1 < 0.07057
        - 0.01632605 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -1.6%  girth2_top30 < 0.006364
        + 0.01608551 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +1.6%  n_dr_0p2_0p4 < 15
        - 0.01251836 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # -1.3%  sum_pt_top40 < 1070
        + 0.01057953 * max(0.0, 76.41544 - Q.mass_top30) / 12.67596   # +1.1%  mass_top30 < 76.42
        - 0.01013473 * max(0.0, 0.007538019 - Q.girth2_top20) / 0.002740841   # -1.0%  girth2_top20 < 0.007538
        - 0.00891495 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # -0.9%  z_dr_0p2_0p4 < 0.09123
        + 0.008748879 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957) / 1.852228   # +0.9%  log_sum_pt < 6.989 and sum_pt_top50 > 959.1
        - 0.007259024 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -0.7%  lam1 < 0.005914
        - 0.007082656 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -0.7%  sum_pt < 1013
        + 0.006348682 * max(0.0, Q.sj2_dr - 0.1411617) / 0.06721719   # +0.6%  sj2_dr > 0.1412
        + 0.005421662 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +0.5%  psi_0p3 > 0.9956
        + 0.005412314 * max(0.0, 0.007856958 - Q.girth2_top30) / 0.002601721   # +0.5%  girth2_top30 < 0.007857
        - 0.005237621 * max(0.0, 0.9906378 - Q.z_top50_slots) / 0.004783315   # -0.5%  z_top50_slots < 0.9906
        - 0.005116502 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # -0.5%  girth2_top20 < 0.006374
        - 0.004314595 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -0.4%  mass_top50 > 82.04
        + 0.004139201 * max(0.0, 950.9324 - Q.sum_pt_top30) / 25.09781   # +0.4%  sum_pt_top30 < 950.9
        + 0.003724593 * max(0.0, 0.005312783 - Q.girth2_top20) / 0.001437214   # +0.4%  girth2_top20 < 0.005313
        - 0.003124068 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # -0.3%  e2 > 0.05557
        - 0.002756091 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -0.3%  sum_pt < 972
        - 0.002265349 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # -0.2%  sj2_dr > 0.2232
        + 0.001062366 * max(0.0, Q.mass - 78.26182) * max(0.0, 0.06909411 - Q.sj2_zsoft) / 0.005641476   # +0.1%  mass > 78.26 and sj2_zsoft < 0.06909
        - 0.0007732352 * max(0.0, Q.mass_top50 - 71.79516) * max(0.0, 0.06909411 - Q.sj2_zsoft) / 0.007761882   # -0.1%  mass_top50 > 71.8 and sj2_zsoft < 0.06909
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 19.08;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.08223 * (0.1297869
        + 0.09006583 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # +9.0%  log_sum_pt > 6.894
        + 0.07941291 * max(0.0, 1.521582 - Q.soft1_pt) / 0.9527349   # +7.9%  soft1_pt < 1.522
        - 0.07628471 * max(0.0, 2.275391 - Q.soft1_pt) / 1.652278   # -7.6%  soft1_pt < 2.275
        - 0.0731133 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -7.3%  log_sum_pt < 7.139
        + 0.04885249 * max(0.0, 0.1751567 - Q.tau1) / 0.09101058   # +4.9%  tau1 < 0.1752
        + 0.04420259 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +4.4%  log_sum_pt > 6.91
        - 0.03835251 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -3.8%  mass < 101
        - 0.03763503 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -3.8%  sum_pt_top50 > 959.1
        - 0.03501374 * max(0.0, Q.z_top50_slots - 0.9586536) / 0.03427112   # -3.5%  z_top50_slots > 0.9587
        - 0.03245864 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -3.2%  log_sum_pt > 6.959
        + 0.03057855 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.2734   # +3.1%  sum_pt_top2 < 689.2
        + 0.03026513 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +3.0%  n_particles > 38
        + 0.02405046 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # +2.4%  sum_pt_top40 < 1070
        + 0.02355715 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +2.4%  mass < 89.74
        - 0.02045153 * max(0.0, 1.091797 - Q.soft1_pt) / 0.5750351   # -2.0%  soft1_pt < 1.092
        + 0.01612505 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +1.6%  girth2_top5 < 0.00833
        - 0.0155465 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2) / 190.5117   # -1.6%  sj3_mass1 < 32.5 and sj3_mass2 < 18.68
        - 0.015204 * max(0.0, 47.88842 - Q.mass_top20) / 6.040308   # -1.5%  mass_top20 < 47.89
        - 0.01296078 * max(0.0, 50.3522 - Q.m012) / 35.90052   # -1.3%  m012 < 50.35
        + 0.01063731 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +1.1%  sum_pt < 1017
        - 0.01047448 * max(0.0, 121.3913 - Q.mass) / 39.46288   # -1.0%  mass < 121.4
        - 0.01031516 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt) / 15.05788   # -1.0%  n_particles > 38 and soft1_pt < 2.275
        - 0.009948537 * max(0.0, Q.z_top30_slots - 0.9341838) / 0.03406818   # -1.0%  z_top30_slots > 0.9342
        - 0.009858682 * max(0.0, 30.26161 - Q.sj2_mass1) / 7.997784   # -1.0%  sj2_mass1 < 30.26
        + 0.009073629 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # +0.9%  tau2 < 0.0795
        + 0.008622415 * max(0.0, 80.24626 - Q.mass_top30) / 14.57104   # +0.9%  mass_top30 < 80.25
        - 0.008275338 * max(0.0, Q.zdr_0 - 0.001901263) / 0.007974889   # -0.8%  zdr_0 > 0.001901
        + 0.008118997 * max(0.0, 0.00100099 - Q.girth2_top5) / 0.0002508867   # +0.8%  girth2_top5 < 0.001001
        - 0.007957954 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # -0.8%  log_sum_pt > 7.017
        - 0.007910989 * max(0.0, 68.43422 - Q.mass_top5) / 42.12422   # -0.8%  mass_top5 < 68.43
        - 0.007752432 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -0.8%  n_dr_0p2_0p4 < 7
        + 0.007744136 * max(0.0, 0.03209934 - Q.tau3) / 0.01307604   # +0.8%  tau3 < 0.0321
        - 0.00773206 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.8%  psi_0p3 > 0.998
        + 0.007688945 * max(0.0, Q.z_top30_slots - 0.9564984) / 0.01947087   # +0.8%  z_top30_slots > 0.9565
        + 0.00751 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 39.23115   # +0.8%  mass_top20 < 47.89 and n_real_top40 > 29
        + 0.007159534 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2) / 0.0008500795   # +0.7%  z_top30_slots > 0.9342 and C2 < 0.0728
        + 0.007051859 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1) / 1.005598   # +0.7%  n_particles > 38 and dr_1 < 0.1612
        - 0.006926862 * max(0.0, 31.35938 - Q.pt_9) / 6.538304   # -0.7%  pt_9 < 31.36
        + 0.006382172 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.01395978 - Q.zdr_2) / 0.09391572   # +0.6%  n_particles > 38 and zdr_2 < 0.01396
        - 0.005700192 * max(0.0, 33.78125 - Q.pt_11) / 12.89848   # -0.6%  pt_11 < 33.78
        - 0.00564514 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # -0.6%  sj3_mass1 < 32.5
        - 0.00513801 * max(0.0, Q.z_dr_0_0p05 - 0.8459004) / 0.02484119   # -0.5%  z_dr_0_0p05 > 0.8459
        + 0.004434337 * max(0.0, 0.003270031 - Q.girth2_top15) / 0.0007969965   # +0.4%  girth2_top15 < 0.00327
        + 0.004416917 * max(0.0, 10.0 - Q.n_dr_0_0p05) / 2.509415   # +0.4%  n_dr_0_0p05 < 10
        - 0.004393046 * max(0.0, 14.54404 - Q.mass_top5) / 4.336972   # -0.4%  mass_top5 < 14.54
        - 0.004360402 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # -0.4%  M3 < 0.03188
        + 0.004200265 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793) / 0.1941151   # +0.4%  z_top30_slots > 0.9342 and max_pair_mass > 13.05
        + 0.004034267 * max(0.0, 0.001823079 - Q.girth2_top20) / 0.000269929   # +0.4%  girth2_top20 < 0.001823
        - 0.003963166 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.4%  n_dr_0p1_0p2 < 8
        + 0.003435479 * max(0.0, 0.0007894752 - Q.girth2_top15) / 9.062109e-05   # +0.3%  girth2_top15 < 0.0007895
        + 0.003356286 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.3%  lam1 < 0.004673
        - 0.003259467 * max(0.0, 0.03107249 - Q.z_7) / 0.003035085   # -0.3%  z_7 < 0.03107
        + 0.003154725 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0) / 0.570275   # +0.3%  n_particles > 38 and dr_0 < 0.112
        - 0.003096798 * max(0.0, Q.sum_pt_top3 - 353.0625) / 132.9972   # -0.3%  sum_pt_top3 > 353.1
        - 0.003062151 * max(0.0, 0.03187688 - Q.M3) * max(0.0, Q.M2 - 0.05568888) / 0.0001450223   # -0.3%  M3 < 0.03188 and M2 > 0.05569
        - 0.002964353 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, 7.880676 - Q.sj3_mass3) / 0.1008495   # -0.3%  z_top30_slots > 0.9565 and sj3_mass3 < 7.881
        + 0.002929658 * max(0.0, 0.1751567 - Q.tau1) * max(0.0, Q.eta_0 - -0.008410263) / 0.001385655   # +0.3%  tau1 < 0.1752 and eta_0 > -0.00841
        - 0.002704561 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.932092   # -0.3%  n_dr_0p2_0p4 < 13
        + 0.002435309 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 15.47191 - Q.pair_mass_0_12) / 61.85839   # +0.2%  pt_9 < 31.36 and pair_mass_0_12 < 15.47
        - 0.00208432 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -0.2%  log_sum_pt > 6.989
        + 0.001991416 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874) / 0.05244257   # +0.2%  z_top30_slots > 0.9342 and ptdr0_3 > 7.408
        + 0.001812232 * max(0.0, Q.mass_top10 - 31.33272) / 22.02168   # +0.2%  mass_top10 > 31.33
        + 0.001567014 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, Q.ptdr0_4 - 4.470953) / 0.03202477   # +0.2%  z_top30_slots > 0.9565 and ptdr0_4 > 4.471
        + 0.001392675 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, 30.0 - Q.n_real_top30) / 3.822398   # +0.1%  n_dr_0p2_0p4 < 7 and n_real_top30 < 30
        + 0.001373449 * max(0.0, Q.sum_pt_top30 - 1191.938) / 6.938323   # +0.1%  sum_pt_top30 > 1192
        + 0.001170623 * max(0.0, 0.003270031 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 6.08612e-07   # +0.1%  girth2_top15 < 0.00327 and psi_0p3 > 0.9974
        + 0.0006513504 * max(0.0, Q.sj3_dr_max - 0.3483732) / 0.01384956   # +0.1%  sj3_dr_max > 0.3484
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 5.493;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.492888 * (0.008690287
        - 0.1669741 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -16.7%  mass < 92.86
        + 0.1105411 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15) / 0.0009227558   # +11.1%  log_sum_pt > 6.903 and girth2_top15 < 0.02147
        + 0.09583857 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # +9.6%  sum_pt > 1017
        + 0.0846832 * max(0.0, 91.03469 - Q.mass) / 16.36287   # +8.5%  mass < 91.03
        + 0.07344509 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +7.3%  mass < 121.4
        - 0.07277461 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15) / 1.045047   # -7.3%  sum_pt_top50 > 988.5 and girth2_top15 < 0.02147
        - 0.06884998 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # -6.9%  sum_pt < 1261
        + 0.05405096 * max(0.0, 143.7876 - Q.mass) / 57.99337   # +5.4%  mass < 143.8
        - 0.04363725 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -4.4%  sum_pt > 1053
        - 0.02631961 * max(0.0, 132.4278 - Q.mass_top40) / 51.50818   # -2.6%  mass_top40 < 132.4
        + 0.02504401 * max(0.0, 1041.263 - Q.sum_pt_top40) / 49.16087   # +2.5%  sum_pt_top40 < 1041
        + 0.02261566 * max(0.0, Q.sum_pt_top50 - 1156.659) / 13.71464   # +2.3%  sum_pt_top50 > 1157
        - 0.02137086 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -2.1%  log_sum_pt > 7.063
        - 0.02094622 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003463934   # -2.1%  log_sum_pt > 6.903 and psi_0p3 > 0.9299
        - 0.01990357 * max(0.0, 1003.329 - Q.sum_pt_top15) / 147.1939   # -2.0%  sum_pt_top15 < 1003
        + 0.01947821 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +1.9%  sum_pt_top40 < 1053
        - 0.01549518 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # -1.5%  sum_pt > 1116
        + 0.01528035 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # +1.5%  mass_top40 < 83.33
        - 0.01444539 * max(0.0, 51.0 - Q.n_particles) / 9.158477   # -1.4%  n_particles < 51
        - 0.01368572 * max(0.0, 996.8867 - Q.sum_pt_top30) / 42.94346   # -1.4%  sum_pt_top30 < 996.9
        - 0.007378364 * max(0.0, 78.26182 - Q.mass) / 9.857173   # -0.7%  mass < 78.26
        - 0.003936146 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.mass_top40 - 77.93668) / 1.109603   # -0.4%  log_sum_pt > 6.903 and mass_top40 > 77.94
        - 0.002412294 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -0.2%  lam1 < 0.006717
        + 0.0008935758 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.girth2_top30 - 0.008376291) / 1.929368e-05   # +0.1%  log_sum_pt > 7.063 and girth2_top30 > 0.008376
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 3.628;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.628127 * (-0.01283838
        - 0.1631314 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # -16.3%  girth2 < 0.009615
        + 0.08068683 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +8.1%  girth2_top40 < 0.008841
        + 0.06181273 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +6.2%  n_particles < 46
        + 0.04939451 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # +4.9%  girth2_top50 < 0.008125
        - 0.04739902 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -4.7%  tau21 < 0.3862
        + 0.047208 * max(0.0, 0.006026828 - Q.girth2_top40) / 0.001351778   # +4.7%  girth2_top40 < 0.006027
        + 0.04539407 * max(0.0, 0.08665515 - Q.mass_over_sum_pt) / 0.01542259   # +4.5%  mass_over_sum_pt < 0.08666
        - 0.04042334 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # -4.0%  girth2_top40 < 0.00626
        - 0.03767133 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # -3.8%  mass_top40 < 80.89
        - 0.03526376 * max(0.0, 0.02197187 - Q.tau3) / 0.00529808   # -3.5%  tau3 < 0.02197
        + 0.03288365 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +3.3%  tau4 < 0.01627
        + 0.03225449 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +3.2%  n_dr_0p2_0p4 < 5
        + 0.03147551 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +3.1%  lam2 < 0.0006155
        + 0.03059335 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +3.1%  mass < 92.86
        + 0.03022685 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1) / 0.1325294   # +3.0%  n_dr_0p2_0p4 < 5 and psi_0p1 < 0.9538
        + 0.0293169 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741) / 0.0491674   # +2.9%  n_dr_0p2_0p4 < 8 and z_top50_slots > 0.9787
        - 0.02551734 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, 18.0 - Q.n_dr_0_0p05) / 19.28245   # -2.6%  n_dr_0p2_0p4 < 8 and n_dr_0_0p05 < 18
        - 0.02383202 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243) / 0.0001457263   # -2.4%  tau21 < 0.3862 and lam1 > 0.007671
        + 0.01830275 * max(0.0, 1.36316 - Q.D2_b2) / 0.3773161   # +1.8%  D2_b2 < 1.363
        - 0.01685933 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349) / 64.17204   # -1.7%  n_particles < 46 and mass_top15 > 57.87
        - 0.01618576 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -1.6%  mass < 53.87
        - 0.01572588 * max(0.0, 79.21004 - Q.mass_top50) / 10.65754   # -1.6%  mass_top50 < 79.21
        + 0.01421366 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421) / 0.000406438   # +1.4%  D2 < 2.41 and psi_0p3 > 0.9985
        + 0.01286895 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.9087063) / 0.1465289   # +1.3%  n_dr_0p1_0p2 < 10 and psi_0p2 > 0.9087
        - 0.01244384 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.006794973 - Q.zdr_4) / 0.003693981   # -1.2%  n_dr_0p2_0p4 < 5 and zdr_4 < 0.006795
        + 0.01208358 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p05_0p1 - 0.5106729) / 0.121515   # +1.2%  n_dr_0p2_0p4 < 5 and z_dr_0p05_0p1 > 0.5107
        - 0.01188775 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0) / 3.186294   # -1.2%  n_dr_0p2_0p4 < 5 and n_dr_0p1_0p2 > 9
        + 0.01107631 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732) / 1477.074   # +1.1%  n_particles < 46 and sum_pt_top30 > 800.7
        - 0.007024047 * max(0.0, Q.psi_0p3 - 0.9995915) / 8.716828e-05   # -0.7%  psi_0p3 > 0.9996
        - 0.005679506 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 988.4375 - Q.sum_pt_top30) / 17.46027   # -0.6%  n_dr_0p2_0p4 < 5 and sum_pt_top30 < 988.4
        - 0.001163541 * max(0.0, 0.2738063 - Q.max_dr) / 0.009268927   # -0.1%  max_dr < 0.2738
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 12.48;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.47908 * (0.02414297
        - 0.07473183 * max(0.0, 87.36377 - Q.mass) / 14.19996   # -7.5%  mass < 87.36
        + 0.07379177 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +7.4%  mass < 101
        - 0.06922349 * max(0.0, 79.65241 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.03386155   # -6.9%  mass < 79.65 and lam2 < 0.003688
        - 0.06576866 * max(0.0, 94.64253 - Q.mass_top40) / 21.02021   # -6.6%  mass_top40 < 94.64
        + 0.06537744 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.07479306   # +6.5%  mass < 101 and lam2 < 0.003688
        - 0.05162252 * max(0.0, Q.n_particles - 22.0) / 23.9717   # -5.2%  n_particles > 22
        + 0.04476196 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # +4.5%  mass_top40 < 83.33
        - 0.04281201 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # -4.3%  girth2_top10 < 0.007679
        - 0.042059 * max(0.0, 74.25181 - Q.mass) / 8.587064   # -4.2%  mass < 74.25
        + 0.03941284 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +3.9%  mass < 121.4
        + 0.03595004 * max(0.0, 0.008956554 - Q.girth2_top10) / 0.004473389   # +3.6%  girth2_top10 < 0.008957
        - 0.03202176 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 6.916121 - Q.D2) / 32.02927   # -3.2%  mass_top40 < 83.33 and D2 < 6.916
        - 0.03141024 * max(0.0, 121.737 - Q.mass_top30) / 46.42587   # -3.1%  mass_top30 < 121.7
        + 0.02848474 * max(0.0, 79.65241 - Q.mass) / 10.37103   # +2.8%  mass < 79.65
        + 0.0263106 * max(0.0, 6.916121 - Q.D2) / 4.129327   # +2.6%  D2 < 6.916
        - 0.01922015 * max(0.0, 0.004855289 - Q.girth2_top15) / 0.001418414   # -1.9%  girth2_top15 < 0.004855
        + 0.01689158 * max(0.0, Q.n_dr_0_0p05 - 5.0) / 8.271047   # +1.7%  n_dr_0_0p05 > 5
        + 0.01660506 * max(0.0, Q.e2_sq - 0.01396296) / 0.002220692   # +1.7%  e2_sq > 0.01396
        + 0.01453741 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt) / 36.47388   # +1.5%  n_particles > 22 and soft1_pt < 2.275
        + 0.01425732 * max(0.0, 0.008184368 - Q.mass_over_sum_pt_sq) / 0.002451608   # +1.4%  mass_over_sum_pt_sq < 0.008184
        + 0.01396225 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +1.4%  n_dr_0p2_0p4 < 26
        - 0.01320831 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -1.3%  tau1 < 0.07057
        - 0.01316094 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # -1.3%  mass_top40 < 67.73
        - 0.01110362 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -1.1%  mass < 92.86
        + 0.0110944 * max(0.0, 69.02716 - Q.mass_top15) / 18.23029   # +1.1%  mass_top15 < 69.03
        - 0.009990394 * max(0.0, 0.03577037 - Q.girth) / 0.004182456   # -1.0%  girth < 0.03577
        - 0.009670075 * max(0.0, Q.sj2_dr - 0.1825048) / 0.04110551   # -1.0%  sj2_dr > 0.1825
        - 0.009644127 * max(0.0, Q.e3 - 6.567534e-05) / 6.326067e-05   # -1.0%  e3 > 6.568e-05
        + 0.009233529 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.9%  psi_0p3 > 0.9974
        + 0.008099963 * max(0.0, Q.sj2_dr - 0.1512157) / 0.05988199   # +0.8%  sj2_dr > 0.1512
        + 0.008068794 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, 1053.047 - Q.sum_pt_top40) / 2417.732   # +0.8%  mass_top30 < 121.7 and sum_pt_top40 < 1053
        - 0.007843766 * max(0.0, 1042.609 - Q.sum_pt) / 35.65948   # -0.8%  sum_pt < 1043
        + 0.007587815 * max(0.0, Q.mass - 143.7876) / 3.946979   # +0.8%  mass > 143.8
        + 0.007495362 * max(0.0, Q.e2 - 0.03680582) / 0.004603798   # +0.7%  e2 > 0.03681
        - 0.005924595 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # -0.6%  LHA > 0.372
        - 0.005619626 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # -0.6%  lam2 < 0.001776
        + 0.005209391 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +0.5%  e2 < 0.02516
        - 0.00473772 * max(0.0, Q.mass_top20 - 103.6749) / 3.795511   # -0.5%  mass_top20 > 103.7
        + 0.004463734 * max(0.0, Q.lam2 - 0.0004662705) / 0.001028518   # +0.4%  lam2 > 0.0004663
        + 0.004428393 * max(0.0, Q.e3 - 0.0001841806) / 4.035159e-05   # +0.4%  e3 > 0.0001842
        - 0.004376055 * max(0.0, Q.mass_top50 - 138.8977) / 3.930268   # -0.4%  mass_top50 > 138.9
        + 0.003976596 * max(0.0, 48.0 - Q.n_pt_above_1) / 9.959713   # +0.4%  n_pt_above_1 < 48
        - 0.003501642 * max(0.0, Q.girth2_top15 - 0.007887677) / 0.002755219   # -0.4%  girth2_top15 > 0.007888
        + 0.002991189 * max(0.0, Q.sj3_pair_mass_min - 32.51366) / 5.88267   # +0.3%  sj3_pair_mass_min > 32.51
        - 0.002768421 * max(0.0, 0.005809485 - Q.girth2_top30) / 0.001402136   # -0.3%  girth2_top30 < 0.005809
        + 0.002697077 * max(0.0, 0.004855289 - Q.girth2_top15) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 0.007590625   # +0.3%  girth2_top15 < 0.004855 and n_dr_0p05_0p1 > 2
        - 0.00242908 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.zdr_0 - 0.006864207) / 0.008666302   # -0.2%  mass < 87.36 and zdr_0 > 0.006864
        + 0.001462705 * max(0.0, 0.2982 - Q.max_dr) / 0.01350087   # +0.1%  max_dr < 0.2982
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 16.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.22803 * (0.05782476
        + 0.08157886 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +8.2%  sum_pt_top50 > 934.2
        - 0.06751121 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # -6.8%  sum_pt > 986.1
        + 0.04604847 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # +4.6%  sum_pt_top50 > 959.1
        - 0.04515028 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -4.5%  log_sum_pt > 6.91
        + 0.04071298 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +4.1%  sum_pt > 907.9
        - 0.03771979 * max(0.0, Q.mass_over_sum_pt - 0.08873143) / 0.01477171   # -3.8%  mass_over_sum_pt > 0.08873
        - 0.03531885 * max(0.0, Q.mass - 162.8363) / 1.509832   # -3.5%  mass > 162.8
        - 0.03509626 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -3.5%  log_sum_pt > 6.92
        + 0.03313911 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +3.3%  sum_pt > 907.9 and e4 < 5.851e-08
        + 0.03244239 * max(0.0, Q.mass - 87.36377) / 16.57741   # +3.2%  mass > 87.36
        - 0.03165005 * max(0.0, 150.0144 - Q.mass_top40) / 66.86198   # -3.2%  mass_top40 < 150
        + 0.0315406 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +3.2%  n_particles < 64
        - 0.02722149 * max(0.0, Q.mass - 91.03469) / 15.0694   # -2.7%  mass > 91.03
        + 0.02705941 * max(0.0, Q.mass_over_sum_pt - 0.09046749) / 0.01421039   # +2.7%  mass_over_sum_pt > 0.09047
        - 0.02635986 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -2.6%  max_dr > 0.2405
        + 0.02264031 * max(0.0, Q.sd_mass - 69.65633) / 11.73272   # +2.3%  sd_mass > 69.66
        - 0.02226178 * max(0.0, Q.sum_pt_top40 - 994.2695) * max(0.0, 5.8505e-08 - Q.e4) / 2.919126e-06   # -2.2%  sum_pt_top40 > 994.3 and e4 < 5.851e-08
        - 0.02187159 * max(0.0, Q.z_top30_slots - 0.9048492) / 0.05621641   # -2.2%  z_top30_slots > 0.9048
        - 0.01706788 * max(0.0, Q.sd_mass - 83.30647) / 6.989667   # -1.7%  sd_mass > 83.31
        - 0.01657582 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # -1.7%  tau1 < 0.1507
        + 0.01516002 * max(0.0, Q.soft3_pt - 0.3395996) / 0.709118   # +1.5%  soft3_pt > 0.3396
        + 0.01405932 * max(0.0, 0.002741632 - Q.soft3_z) / 0.001786174   # +1.4%  soft3_z < 0.002742
        - 0.01317859 * max(0.0, Q.girth2 - 0.004756928) / 0.005348122   # -1.3%  girth2 > 0.004757
        + 0.01292552 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # +1.3%  e2 < 0.04083
        + 0.01289759 * max(0.0, Q.n_pt_above_1 - 28.0) / 14.80415   # +1.3%  n_pt_above_1 > 28
        - 0.01261891 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2) / 10.81389   # -1.3%  n_particles < 64 and D2 < 2.179
        - 0.01246529 * max(0.0, 0.03700182 - Q.z_dr_0p2_0p4) / 0.0183096   # -1.2%  z_dr_0p2_0p4 < 0.037
        + 0.0124354 * max(0.0, 23.45965 - Q.sj3_mass1) / 9.283354   # +1.2%  sj3_mass1 < 23.46
        + 0.01214738 * max(0.0, Q.sum_pt_top40 - 994.2695) / 51.72055   # +1.2%  sum_pt_top40 > 994.3
        + 0.01151714 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +1.2%  n_dr_0p2_0p4 < 11
        + 0.01120735 * max(0.0, Q.log_sum_pt - 6.935549) / 0.0371538   # +1.1%  log_sum_pt > 6.936
        - 0.01103606 * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 9.82015   # -1.1%  n_dr_0p1_0p2 < 21
        + 0.009496786 * max(0.0, Q.sum_pt_top30 - 933.1875) / 79.88556   # +0.9%  sum_pt_top30 > 933.2
        - 0.008846049 * max(0.0, 0.5100475 - Q.tau21) / 0.1343519   # -0.9%  tau21 < 0.51
        - 0.008620399 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # -0.9%  psi_0p3 > 0.9974
        + 0.007993653 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # +0.8%  log_sum_pt > 6.989
        - 0.007944481 * max(0.0, Q.soft3_z - 0.0008504382) / 0.0003473759   # -0.8%  soft3_z > 0.0008504
        + 0.007550362 * max(0.0, Q.sum_pt_top10 - 822.975) / 44.35907   # +0.8%  sum_pt_top10 > 823
        - 0.007306743 * max(0.0, 787.6281 - Q.sum_pt_top3) / 328.0371   # -0.7%  sum_pt_top3 < 787.6
        - 0.006229901 * max(0.0, Q.n_real_top50 - 22.0) / 19.58685   # -0.6%  n_real_top50 > 22
        - 0.006205635 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -0.6%  sum_pt_top40 > 1025
        + 0.006067051 * max(0.0, 1.976207 - Q.D2) / 0.367701   # +0.6%  D2 < 1.976
        - 0.006021787 * max(0.0, 0.007856958 - Q.girth2_top30) / 0.002601721   # -0.6%  girth2_top30 < 0.007857
        + 0.005344419 * max(0.0, 0.01375115 - Q.z_11) / 0.0009450582   # +0.5%  z_11 < 0.01375
        - 0.005227633 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -0.5%  mass_over_sum_pt_sq > 0.0292
        + 0.005132555 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2) / 0.001543279   # +0.5%  psi_0p3 > 0.9974 and D2 < 3.345
        + 0.005123355 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +0.5%  mass_over_sum_pt > 0.1709
        - 0.004932803 * max(0.0, Q.z_top10_slots - 0.7887522) / 0.03614094   # -0.5%  z_top10_slots > 0.7888
        - 0.004597906 * max(0.0, 14.14062 - Q.pt_11) / 0.9740783   # -0.5%  pt_11 < 14.14
        - 0.004362758 * max(0.0, Q.sum_pt_top15 - 935.1043) / 29.65775   # -0.4%  sum_pt_top15 > 935.1
        + 0.004042762 * max(0.0, 64.48544 - Q.mass) / 5.935866   # +0.4%  mass < 64.49
        - 0.003992732 * max(0.0, 0.707925 - Q.psi_0p1) / 0.07386333   # -0.4%  psi_0p1 < 0.7079
        + 0.003815685 * max(0.0, Q.max_dr - 0.2404747) * max(0.0, 13.23436 - Q.sj3_mass2) / 0.6134585   # +0.4%  max_dr > 0.2405 and sj3_mass2 < 13.23
        + 0.003665088 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # +0.4%  mass_top50 > 157.5
        + 0.002547523 * max(0.0, Q.sum_pt_top20 - 969.7891) / 31.86784   # +0.3%  sum_pt_top20 > 969.8
        - 0.002276829 * max(0.0, Q.girth2_top50 - 0.01951641) / 0.001208668   # -0.2%  girth2_top50 > 0.01952
        - 0.002127884 * max(0.0, Q.mass - 172.4888) * max(0.0, 0.161871 - Q.soft3_dr) / 0.01314483   # -0.2%  mass > 172.5 and soft3_dr < 0.1619
        - 0.001962867 * max(0.0, Q.mass_top40 - 111.2487) / 7.565029   # -0.2%  mass_top40 > 111.2
        + 0.001898386 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # +0.2%  mass_top10 > 71.78
        - 0.001716672 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.2%  sum_pt_top10 > 943.7
        + 0.001537687 * max(0.0, Q.z_dr_0_0p05 - 0.9084912) / 0.009227347   # +0.2%  z_dr_0_0p05 > 0.9085
        - 0.001428303 * max(0.0, Q.mass - 172.4888) / 0.7446233   # -0.1%  mass > 172.5
        + 0.001367731 * max(0.0, Q.max_dr - 0.4357228) / 0.007711928   # +0.1%  max_dr > 0.4357
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 17.74;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.74271 * (-0.02585767
        + 0.1795987 * max(0.0, 0.02580859 - Q.e2_sq) / 0.01698103   # +18.0%  e2_sq < 0.02581
        - 0.1112224 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -11.1%  mass < 101
        - 0.0808889 * max(0.0, 0.02550569 - Q.girth2_top50) / 0.01683664   # -8.1%  girth2_top50 < 0.02551
        - 0.0718819 * max(0.0, 0.02412652 - Q.girth2_top30) / 0.01613825   # -7.2%  girth2_top30 < 0.02413
        - 0.07146044 * max(0.0, 121.3913 - Q.mass) / 39.46288   # -7.1%  mass < 121.4
        + 0.06821129 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +6.8%  mass < 92.86
        - 0.05022296 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # -5.0%  mass_over_sum_pt < 0.09795
        + 0.04370836 * max(0.0, 172.4888 - Q.mass) / 83.49222   # +4.4%  mass < 172.5
        + 0.03333085 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +3.3%  mass < 89.74
        - 0.02821523 * max(0.0, Q.e2 - 0.02793599) / 0.008664879   # -2.8%  e2 > 0.02794
        + 0.02127862 * max(0.0, Q.tau1 - 0.05444509) / 0.03990678   # +2.1%  tau1 > 0.05445
        + 0.01928488 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # +1.9%  girth2_top30 < 0.008376
        + 0.01766759 * max(0.0, 65.20727 - Q.sj2_mass1) / 36.50254   # +1.8%  sj2_mass1 < 65.21
        + 0.01595238 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +1.6%  lam1 < 0.007259
        + 0.01504883 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +1.5%  mass < 82.85
        + 0.01468688 * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) / 0.159722   # +1.5%  z_dr_0p1_0p2 < 0.2865
        - 0.01466034 * max(0.0, 51.2028 - Q.sj2_mass1) / 24.19897   # -1.5%  sj2_mass1 < 51.2
        + 0.013902 * max(0.0, Q.e2 - 0.02793599) * max(0.0, Q.psi_0p3 - 0.9896594) / 4.018702e-05   # +1.4%  e2 > 0.02794 and psi_0p3 > 0.9897
        + 0.01321648 * max(0.0, 0.003687605 - Q.lam2) / 0.002601874   # +1.3%  lam2 < 0.003688
        - 0.01279157 * max(0.0, 0.01649354 - Q.lam1) / 0.009604223   # -1.3%  lam1 < 0.01649
        + 0.01230107 * max(0.0, 76.60223 - Q.sj3_pair_mass_min) / 50.46144   # +1.2%  sj3_pair_mass_min < 76.6
        + 0.01196423 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # +1.2%  mass_top50 < 71.8
        + 0.008121551 * max(0.0, Q.mass_top40 - 94.64253) / 11.19703   # +0.8%  mass_top40 > 94.64
        + 0.007765356 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.8%  e2 > 0.04755
        - 0.007679238 * max(0.0, Q.mass - 143.7876) / 3.946979   # -0.8%  mass > 143.8
        - 0.006430724 * max(0.0, Q.psi_0p1 - 0.7703036) / 0.09218239   # -0.6%  psi_0p1 > 0.7703
        - 0.005810216 * max(0.0, 89.74183 - Q.mass) * max(0.0, 1129.275 - Q.sum_pt_top20) / 2511.831   # -0.6%  mass < 89.74 and sum_pt_top20 < 1129
        + 0.005412823 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +0.5%  psi_0p3 > 0.9924
        + 0.005024954 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +0.5%  tau1 < 0.06311
        - 0.004983556 * max(0.0, 2.178951 - Q.D2) / 0.4648731   # -0.5%  D2 < 2.179
        + 0.004682877 * max(0.0, 65.20727 - Q.sj2_mass1) * max(0.0, Q.eccentricity - 0.3346888) / 16.90037   # +0.5%  sj2_mass1 < 65.21 and eccentricity > 0.3347
        + 0.003048992 * max(0.0, Q.n_dr_0p1_0p2 - 17.0) / 2.16319   # +0.3%  n_dr_0p1_0p2 > 17
        + 0.002991491 * max(0.0, 0.00287991 - Q.girth2_top20) / 0.000559956   # +0.3%  girth2_top20 < 0.00288
        + 0.002735685 * max(0.0, 121.3913 - Q.mass) * max(0.0, 1003.544 - Q.sum_pt_top50) / 688.1559   # +0.3%  mass < 121.4 and sum_pt_top50 < 1004
        - 0.002520597 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.psi_0p3 - 0.9896594) / 5.449059e-06   # -0.3%  e2 > 0.04755 and psi_0p3 > 0.9897
        - 0.002457626 * max(0.0, 1.409617 - Q.D2) / 0.1519846   # -0.2%  D2 < 1.41
        - 0.002323745 * max(0.0, Q.sj3_mass1 - 19.30204) / 2.042969   # -0.2%  sj3_mass1 > 19.3
        - 0.001784315 * max(0.0, Q.e2 - 0.02793599) * max(0.0, 6.06065 - Q.ptdr0_13) / 0.02788669   # -0.2%  e2 > 0.02794 and ptdr0_13 < 6.061
        + 0.001529339 * max(0.0, 0.06310829 - Q.tau1) * max(0.0, Q.lam2 - 0.0002104765) / 2.6798e-06   # +0.2%  tau1 < 0.06311 and lam2 > 0.0002105
        + 0.001469754 * max(0.0, Q.mass_top10 - 99.06678) / 0.7458098   # +0.1%  mass_top10 > 99.07
        - 0.001085654 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.z_14 - 0.01634243) / 5.402196e-06   # -0.1%  e2 > 0.04755 and z_14 > 0.01634
        - 0.0005016408 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) * max(0.0, 9.0 - Q.n_dr_0_0p05) / 2.271573   # -0.1%  n_dr_0p1_0p2 > 33 and n_dr_0_0p05 < 9
        - 0.0001439875 * max(0.0, Q.soft10_dr - 0.2075213) / 0.02066177   # -0.0%  soft10_dr > 0.2075
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 41.92;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 41.91536 * (0.1951136
        - 0.1981304 * Q.log_sum_pt / 6.944607   # -19.8%  log_sum_pt
        - 0.09794668 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -9.8%  mass < 92.86
        + 0.07012795 * max(0.0, 91.03469 - Q.mass) / 16.36287   # +7.0%  mass < 91.03
        + 0.05615617 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.7612129   # +5.6%  mass < 101 and psi_0p3 > 0.9638
        - 0.05508859 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5305504   # -5.5%  mass < 91.03 and psi_0p3 > 0.9638
        + 0.03889987 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # +3.9%  LHA < 0.3098
        - 0.03319988 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.01682193   # -3.3%  mass < 91.03 and psi_0p3 > 0.9974
        + 0.03297105 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.02554674   # +3.3%  mass < 101 and psi_0p3 > 0.9974
        - 0.02922265 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -2.9%  tau1 < 0.1073
        + 0.02452198 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +2.5%  mass < 121.4
        - 0.02416097 * max(0.0, 0.003418057 - Q.girth2_top50) / 0.0004594629   # -2.4%  girth2_top50 < 0.003418
        - 0.02384871 * max(0.0, 0.008190222 - Q.width) / 0.002454223   # -2.4%  width < 0.00819
        - 0.02320196 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # -2.3%  girth2_top30 < 0.008376
        + 0.02291831 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # +2.3%  mass_over_sum_pt_sq < 0.009595
        - 0.02238193 * max(0.0, 73.35236 - Q.mass_top20) / 16.12893   # -2.2%  mass_top20 < 73.35
        + 0.01989 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +2.0%  mass < 78.26
        - 0.01838912 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -1.8%  mass < 101
        + 0.01728508 * max(0.0, 70.42121 - Q.mass_top20) / 14.59536   # +1.7%  mass_top20 < 70.42
        - 0.0154864 * max(0.0, 0.2845608 - Q.LHA) / 0.05172777   # -1.5%  LHA < 0.2846
        - 0.01294251 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # -1.3%  LHA < 0.3332
        + 0.01249369 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +1.2%  mass_over_sum_pt < 0.1182
        - 0.01151401 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -1.2%  lam1 < 0.008242
        + 0.01141772 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +1.1%  tau1 < 0.07709
        + 0.01137657 * max(0.0, 0.006929741 - Q.girth2_top30) / 0.002004687   # +1.1%  girth2_top30 < 0.00693
        + 0.009346503 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # +0.9%  girth2_top30 < 0.01216
        + 0.009255256 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # +0.9%  tau1 < 0.08787
        + 0.008527106 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +0.9%  tau21_b2 < 0.2352
        - 0.008262924 * max(0.0, 1245.697 - Q.sum_pt_top50) / 217.4702   # -0.8%  sum_pt_top50 < 1246
        + 0.007599842 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # +0.8%  girth2 < 0.007877
        + 0.006803312 * max(0.0, 90.4505 - Q.mass_top20) / 27.98123   # +0.7%  mass_top20 < 90.45
        - 0.005468221 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.5%  psi_0p3 > 0.998
        - 0.005433449 * max(0.0, 0.05936026 - Q.tau2) / 0.0302199   # -0.5%  tau2 < 0.05936
        - 0.00511785 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # -0.5%  LHA < 0.2601
        + 0.004591675 * max(0.0, 0.007877041 - Q.girth2) * max(0.0, 0.1134943 - Q.M2) / 8.436081e-05   # +0.5%  girth2 < 0.007877 and M2 < 0.1135
        + 0.004351137 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.01175018   # +0.4%  mass < 82.85 and psi_0p3 > 0.9974
        + 0.003710561 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # +0.4%  lam1 < 0.00619
        + 0.003531467 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9299135) / 1.084471   # +0.4%  mass < 91.03 and psi_0p3 > 0.9299
        - 0.003445357 * max(0.0, 43.0 - Q.n_pt_above_1) / 6.916252   # -0.3%  n_pt_above_1 < 43
        - 0.003096929 * max(0.0, Q.z_top20_slots - 0.9102775) / 0.02689878   # -0.3%  z_top20_slots > 0.9103
        + 0.002574468 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 2.275391 - Q.soft1_pt) / 2.585031   # +0.3%  n_dr_0p2_0p4 < 6 and soft1_pt < 2.275
        - 0.002366819 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3) / 6.520272e-05   # -0.2%  n_dr_0p2_0p4 < 6 and e3 < 7.876e-05
        - 0.002162659 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt) / 11.66143   # -0.2%  tau21_b2 < 0.2352 and sum_pt < 1261
        + 0.00209169 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p3 - 0.9973959) / 4.396153e-05   # +0.2%  mass_over_sum_pt < 0.1182 and psi_0p3 > 0.9974
        + 0.001874305 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +0.2%  mass < 82.85
        - 0.001543408 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003362293   # -0.2%  tau21_b2 < 0.2352 and psi_0p3 > 0.9299
        + 0.00152525 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.sd_mass - 76.29481) / 33.79048   # +0.2%  mass < 121.4 and sd_mass > 76.29
        - 0.001480247 * max(0.0, 0.006403325 - Q.girth2) / 0.001406912   # -0.1%  girth2 < 0.006403
        + 0.001455208 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 11.65938 - Q.sj3_mass3) / 0.4349059   # +0.1%  tau21_b2 < 0.2352 and sj3_mass3 < 11.66
        - 0.00136818 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.sd_mass - 55.96662) / 29.57136   # -0.1%  mass < 91.03 and sd_mass > 55.97
        + 0.001287828 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.1%  n_dr_0p2_0p4 < 6
        - 0.001121389 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.001696484   # -0.1%  tau21_b2 < 0.2352 and sj2_dr > 0.1938
        - 0.001105419 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.n_real_top50 - 34.0) / 0.287919   # -0.1%  tau21_b2 < 0.2352 and n_real_top50 > 34
        - 0.001079246 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 5.106465e-05   # -0.1%  tau21_b2 < 0.2352 and mass_over_sum_pt_sq < 0.007873
        - 0.001019641 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.1825048 - Q.sj2_dr) / 0.06647299   # -0.1%  n_dr_0p2_0p4 < 6 and sj2_dr < 0.1825
        + 0.0009669794 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.08816102   # +0.1%  tau21_b2 < 0.2352 and n_pt_above_50 < 7
        + 0.0008758658 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.9973959 - Q.psi_0p3) / 0.03659629   # +0.1%  mass < 91.03 and psi_0p3 < 0.9974
        + 0.0008508786 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.pt_4 - 60.03125) / 11.52427   # +0.1%  n_dr_0p2_0p4 < 6 and pt_4 > 60.03
        - 0.0006403007 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.3825328 - Q.pt2_over_pt0) / 0.002626916   # -0.1%  tau21_b2 < 0.2352 and pt2_over_pt0 < 0.3825
        - 0.0003842154 * max(0.0, 91.03469 - Q.mass) * max(0.0, 6.0 - Q.n_dr_0_0p05) / 1.610656   # -0.0%  mass < 91.03 and n_dr_0_0p05 < 6
        + 0.0001122369 * max(0.0, 73.35236 - Q.mass_top20) * max(0.0, 0.002181998 - Q.soft1_z) / 0.02705102   # +0.0%  mass_top20 < 73.35 and soft1_z < 0.002182
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 19.8;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.80456 * (-0.0321618
        + 0.08413819 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266) / 0.00762696   # +8.4%  mass_over_sum_pt_sq > 0.001785
        - 0.06833096 * max(0.0, Q.e2_sq - 0.009606007) / 0.003182984   # -6.8%  e2_sq > 0.009606
        - 0.05795818 * max(0.0, Q.mass - 101.0497) / 12.31084   # -5.8%  mass > 101
        - 0.0557156 * max(0.0, Q.girth2_top40 - 0.006026828) / 0.00421794   # -5.6%  girth2_top40 > 0.006027
        + 0.05493807 * max(0.0, Q.mass - 80.78464) / 19.8061   # +5.5%  mass > 80.78
        + 0.04905711 * max(0.0, Q.mass - 64.48544) / 31.19164   # +4.9%  mass > 64.49
        - 0.04885096 * max(0.0, 0.0289594 - Q.girth2_top50) / 0.02002571   # -4.9%  girth2_top50 < 0.02896
        + 0.0450338 * max(0.0, 0.007887677 - Q.girth2_top15) / 0.00326329   # +4.5%  girth2_top15 < 0.007888
        + 0.04333184 * max(0.0, Q.e2_sq - 0.007872294) / 0.00365949   # +4.3%  e2_sq > 0.007872
        - 0.04306268 * max(0.0, Q.mass_top50 - 80.35535) / 18.59691   # -4.3%  mass_top50 > 80.36
        + 0.03983931 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +4.0%  e3 < 0.0003372
        - 0.03758818 * max(0.0, 0.01563836 - Q.girth2_top15) / 0.009593305   # -3.8%  girth2_top15 < 0.01564
        - 0.03674457 * max(0.0, Q.mass_top50 - 43.66601) / 46.00141   # -3.7%  mass_top50 > 43.67
        + 0.0342652 * max(0.0, Q.girth2_top40 - 0.005196966) / 0.004725809   # +3.4%  girth2_top40 > 0.005197
        + 0.02582037 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # +2.6%  mass_over_sum_pt > 0.07697
        + 0.02542873 * max(0.0, Q.mass_top50 - 97.93004) / 11.91764   # +2.5%  mass_top50 > 97.93
        + 0.02148221 * max(0.0, Q.girth2_top40 - 0.008031986) / 0.003375597   # +2.1%  girth2_top40 > 0.008032
        - 0.01925684 * max(0.0, Q.tau1 - 0.06310829) / 0.03403397   # -1.9%  tau1 > 0.06311
        + 0.01468764 * max(0.0, 0.007856958 - Q.girth2_top30) / 0.002601721   # +1.5%  girth2_top30 < 0.007857
        + 0.01431309 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4) / 0.08975043   # +1.4%  z_dr_0p2_0p4 < 0.1292
        - 0.01377259 * max(0.0, 0.02737453 - Q.girth2_top20) / 0.01974268   # -1.4%  girth2_top20 < 0.02737
        - 0.01323752 * max(0.0, 80.24626 - Q.mass_top30) / 14.57104   # -1.3%  mass_top30 < 80.25
        + 0.01257971 * max(0.0, 6.935549 - Q.log_sum_pt) / 0.02809622   # +1.3%  log_sum_pt < 6.936
        - 0.0121482 * max(0.0, Q.mass - 121.3913) / 7.81281   # -1.2%  mass > 121.4
        + 0.01174045 * max(0.0, Q.girth2_top15 - 0.001319197) / 0.006270373   # +1.2%  girth2_top15 > 0.001319
        - 0.009704086 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -1.0%  n_dr_0p2_0p4 < 15
        - 0.009477258 * max(0.0, Q.sum_pt_top40 - 858.8262) / 166.7848   # -0.9%  sum_pt_top40 > 858.8
        + 0.009012737 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # +0.9%  log_sum_pt > 6.959
        - 0.008088756 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -0.8%  mass_over_sum_pt > 0.1182
        + 0.007505082 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) / 3.639534   # +0.8%  n_dr_0p2_0p4 > 8
        + 0.007016282 * max(0.0, Q.mass - 89.74183) / 15.55572   # +0.7%  mass > 89.74
        - 0.006747178 * max(0.0, 0.007887677 - Q.girth2_top15) * max(0.0, 0.6133424 - Q.tau21_b2) / 0.0006475533   # -0.7%  girth2_top15 < 0.007888 and tau21_b2 < 0.6133
        + 0.006329751 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +0.6%  LHA > 0.3332
        - 0.006300762 * max(0.0, Q.sum_pt_top50 - 1038.855) / 34.95069   # -0.6%  sum_pt_top50 > 1039
        + 0.005905013 * max(0.0, 1003.544 - Q.sum_pt_top50) / 20.02299   # +0.6%  sum_pt_top50 < 1004
        + 0.004261398 * max(0.0, Q.lam1 - 0.008241985) / 0.002595658   # +0.4%  lam1 > 0.008242
        - 0.003956704 * max(0.0, 1032.405 - Q.sum_pt_top40) / 43.16371   # -0.4%  sum_pt_top40 < 1032
        + 0.003735207 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.4%  e2 > 0.04755
        + 0.003726971 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.4%  sj2_dr > 0.2232
        - 0.003353833 * max(0.0, 1032.405 - Q.sum_pt_top40) * max(0.0, 50.3522 - Q.mass_top3) / 1525.305   # -0.3%  sum_pt_top40 < 1032 and mass_top3 < 50.35
        - 0.003191322 * max(0.0, 0.005383629 - Q.girth2_top15) / 0.001666876   # -0.3%  girth2_top15 < 0.005384
        + 0.003051313 * max(0.0, 818.9605 - Q.sum_pt_top15) / 38.21852   # +0.3%  sum_pt_top15 < 819
        + 0.00285528 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435) / 1.470253   # +0.3%  log_sum_pt > 6.959 and sj3_pair_mass_max > 28.35
        + 0.00256657 * max(0.0, Q.n_pt_above_1 - 54.0) / 1.837408   # +0.3%  n_pt_above_1 > 54
        + 0.001830792 * max(0.0, 0.006630167 - Q.girth2_top40) / 0.001656929   # +0.2%  girth2_top40 < 0.00663
        - 0.001629155 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) * max(0.0, Q.zdr_0 - 0.005911134) / 0.02907477   # -0.2%  n_dr_0p2_0p4 > 8 and zdr_0 > 0.005911
        + 0.001536082 * max(0.0, Q.C2 - 0.07996447) / 0.01049526   # +0.2%  C2 > 0.07996
        - 0.001235963 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.1%  log_sum_pt < 6.811
        - 0.00117237 * max(0.0, 1032.405 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477) / 0.1123331   # -0.1%  sum_pt_top40 < 1032 and psi_0p3 > 0.9924
        - 0.001083263 * max(0.0, Q.mass_top20 - 90.4505) / 6.038147   # -0.1%  mass_top20 > 90.45
        + 0.001072957 * max(0.0, Q.n_dr_0p1_0p2 - 19.0) / 1.732133   # +0.1%  n_dr_0p1_0p2 > 19
        - 0.0003319287 * max(0.0, 907.9372 - Q.sum_pt) / 3.961002   # -0.0%  sum_pt < 907.9
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 11.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.77512 * (0.02422554
        + 0.2116201 * max(0.0, 132.4278 - Q.mass_top40) / 51.50818   # +21.2%  mass_top40 < 132.4
        - 0.195348 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -19.5%  mass_top40 < 163.3
        - 0.0865052 * max(0.0, 121.737 - Q.mass_top30) / 46.42587   # -8.7%  mass_top30 < 121.7
        + 0.04467511 * max(0.0, 87.36377 - Q.mass) / 14.19996   # +4.5%  mass < 87.36
        - 0.03505115 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -3.5%  mass < 64.49
        + 0.03126853 * max(0.0, 79.65241 - Q.mass) / 10.37103   # +3.1%  mass < 79.65
        + 0.02680812 * max(0.0, 0.2091025 - Q.LHA) / 0.02095782   # +2.7%  LHA < 0.2091
        + 0.02520827 * max(0.0, Q.girth - 0.08589404) / 0.01080971   # +2.5%  girth > 0.08589
        + 0.02359049 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # +2.4%  girth2_top30 < 0.01216
        + 0.02292233 * max(0.0, 152.6883 - Q.mass_top30) / 74.18247   # +2.3%  mass_top30 < 152.7
        - 0.02161197 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -2.2%  tau1 < 0.05445
        - 0.01949931 * max(0.0, Q.mass - 143.7876) / 3.946979   # -1.9%  mass > 143.8
        - 0.01677997 * max(0.0, 89.6788 - Q.mass_top40) / 17.47899   # -1.7%  mass_top40 < 89.68
        + 0.01601926 * max(0.0, 73.33139 - Q.mass_top30) / 11.37701   # +1.6%  mass_top30 < 73.33
        - 0.01537511 * max(0.0, Q.girth - 0.08589404) * max(0.0, 0.002036949 - Q.soft5_z) / 7.737606e-06   # -1.5%  girth > 0.08589 and soft5_z < 0.002037
        + 0.01454498 * max(0.0, Q.girth - 0.09749958) * max(0.0, 0.002036949 - Q.soft5_z) / 5.873801e-06   # +1.5%  girth > 0.0975 and soft5_z < 0.002037
        - 0.01390451 * max(0.0, 0.05048381 - Q.girth) / 0.008533025   # -1.4%  girth < 0.05048
        - 0.01311679 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -1.3%  psi_0p3 > 0.9943
        + 0.01206069 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # +1.2%  girth2_top40 < 0.00626
        + 0.01147542 * max(0.0, 0.00634935 - Q.girth2_top50) / 0.001425305   # +1.1%  girth2_top50 < 0.006349
        + 0.01113443 * max(0.0, 956.2133 - Q.sum_pt_top40) / 14.00451   # +1.1%  sum_pt_top40 < 956.2
        + 0.0110276 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +1.1%  n_dr_0p2_0p4 < 26
        + 0.01049141 * max(0.0, 0.2187642 - Q.z_dr_0p1_0p2) / 0.1086627   # +1.0%  z_dr_0p1_0p2 < 0.2188
        - 0.009236988 * max(0.0, Q.mass - 162.8363) / 1.509832   # -0.9%  mass > 162.8
        + 0.008767096 * max(0.0, 0.003636595 - Q.mass_over_sum_pt_sq) / 0.0004958684   # +0.9%  mass_over_sum_pt_sq < 0.003637
        + 0.008109162 * max(0.0, Q.mass - 101.0497) / 12.31084   # +0.8%  mass > 101
        + 0.007632281 * max(0.0, Q.mass - 172.4888) / 0.7446233   # +0.8%  mass > 172.5
        - 0.007004092 * max(0.0, Q.girth2_top30 - 0.02412652) / 0.0004997242   # -0.7%  girth2_top30 > 0.02413
        - 0.006739082 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -0.7%  girth2_top20 < 0.01083
        + 0.006209735 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +0.6%  mass_top40 < 80.89
        - 0.005923874 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt) / 447.38   # -0.6%  mass_top40 < 80.89 and sum_pt < 1035
        + 0.005835766 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +0.6%  lam1 > 0.01174
        - 0.005549327 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft5_z - 0.0005078411) / 0.0203614   # -0.6%  sum_pt_top40 < 956.2 and soft5_z > 0.0005078
        - 0.00451849 * max(0.0, 886.3438 - Q.sum_pt_top30) / 11.17185   # -0.5%  sum_pt_top30 < 886.3
        + 0.004411891 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.n_particles - 38.0) / 73.85286   # +0.4%  mass < 87.36 and n_particles > 38
        - 0.004230393 * max(0.0, 0.002582316 - Q.girth2_top30) / 0.0003518773   # -0.4%  girth2_top30 < 0.002582
        + 0.003579233 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # +0.4%  sum_pt_top50 < 959.1
        + 0.003519087 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft5_z - 0.0009919684) / 0.008783167   # +0.4%  sum_pt_top50 < 959.1 and soft5_z > 0.000992
        - 0.002866161 * max(0.0, Q.mass - 121.3913) / 7.81281   # -0.3%  mass > 121.4
        + 0.002724794 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 934.2416 - Q.sum_pt_top50) / 121.4929   # +0.3%  mass_top40 < 80.89 and sum_pt_top50 < 934.2
        + 0.002680285 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, Q.n_dr_0p2_0p4 - 10.0) / 39.79287   # +0.3%  mass_top30 < 121.7 and n_dr_0p2_0p4 > 10
        + 0.002402123 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258) / 5.834847   # +0.2%  sum_pt_top40 < 956.2 and soft4_pt > 1.789
        + 0.001796817 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # +0.2%  e3 > 0.0003372
        - 0.001511234 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, 20.0 - Q.n_pt_above_10) / 29.14649   # -0.2%  sum_pt_top40 < 956.2 and n_pt_above_10 < 20
        - 0.001210348 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft4_pt - 1.375) / 4.279282   # -0.1%  sum_pt_top50 < 959.1 and soft4_pt > 1.375
        + 0.001203809 * max(0.0, 0.00217213 - Q.girth2_top40) / 0.0002075259   # +0.1%  girth2_top40 < 0.002172
        - 0.001079744 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.894531) / 2.293712   # -0.1%  sum_pt_top40 < 956.2 and soft5_pt > 2.895
        + 0.0008059579 * max(0.0, Q.mass - 143.7876) * max(0.0, Q.z_dr_0p05_0p1 - 0.4026646) / 0.1395143   # +0.1%  mass > 143.8 and z_dr_0p05_0p1 > 0.4027
        + 0.0004134785 * max(0.0, Q.mass_over_sum_pt - 0.1708801) * max(0.0, Q.soft4_pt - 1.789258) / 0.0003712571   # +0.0%  mass_over_sum_pt > 0.1709 and soft4_pt > 1.789
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 23.37;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.37482 * (-0.09077196
        + 0.1025966 * Q.sum_pt / 1043.796   # +10.3%  sum_pt
        - 0.09193428 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -9.2%  girth < 0.1207
        + 0.08706827 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # +8.7%  tau1 < 0.1507
        - 0.06877061 * max(0.0, Q.mass - 64.48544) / 31.19164   # -6.9%  mass > 64.49
        + 0.05465059 * max(0.0, Q.mass - 80.78464) / 19.8061   # +5.5%  mass > 80.78
        + 0.04325834 * max(0.0, Q.mass - 89.74183) / 15.55572   # +4.3%  mass > 89.74
        - 0.04306887 * max(0.0, 80.3008 - Q.sj2_mass1) / 50.41572   # -4.3%  sj2_mass1 < 80.3
        - 0.03713779 * max(0.0, 0.06524004 - Q.e2) / 0.03476195   # -3.7%  e2 < 0.06524
        + 0.0353108 * max(0.0, 0.01807679 - Q.girth2_top30) / 0.01083719   # +3.5%  girth2_top30 < 0.01808
        + 0.0343715 * max(0.0, 132.4278 - Q.mass_top40) / 51.50818   # +3.4%  mass_top40 < 132.4
        - 0.01983536 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -2.0%  tau1 < 0.05445
        + 0.01863669 * max(0.0, 0.07852883 - Q.mass_over_sum_pt) / 0.01107516   # +1.9%  mass_over_sum_pt < 0.07853
        - 0.01831491 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -1.8%  mass_top40 < 83.33
        - 0.01813034 * max(0.0, Q.mass - 143.7876) / 3.946979   # -1.8%  mass > 143.8
        + 0.01727827 * max(0.0, 111.2487 - Q.mass_top40) / 33.99436   # +1.7%  mass_top40 < 111.2
        + 0.01712096 * max(0.0, 0.6133424 - Q.tau21_b2) / 0.3099701   # +1.7%  tau21_b2 < 0.6133
        - 0.01671221 * max(0.0, Q.log_sum_pt - 6.903423) / 0.05662275   # -1.7%  log_sum_pt > 6.903
        + 0.0165988 * max(0.0, 839.9547 - Q.sum_pt_top5) / 255.1112   # +1.7%  sum_pt_top5 < 840
        + 0.01608547 * max(0.0, 87.27603 - Q.mass_top40) / 15.9837   # +1.6%  mass_top40 < 87.28
        + 0.01571373 * max(0.0, Q.mass_top40 - 67.72643) / 24.79585   # +1.6%  mass_top40 > 67.73
        + 0.01570675 * max(0.0, 3.814159 - Q.D2) / 1.519965   # +1.6%  D2 < 3.814
        - 0.01472812 * max(0.0, Q.mass - 101.0497) / 12.31084   # -1.5%  mass > 101
        + 0.0144736 * max(0.0, 0.009962397 - Q.girth2_top15) / 0.004894699   # +1.4%  girth2_top15 < 0.009962
        + 0.01433156 * max(0.0, Q.pt_entropy - 2.07371) / 0.8049786   # +1.4%  pt_entropy > 2.074
        - 0.01406912 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # -1.4%  e2 < 0.04083
        - 0.0110288 * max(0.0, Q.mass - 82.85409) / 18.72167   # -1.1%  mass > 82.85
        - 0.009964547 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # -1.0%  psi_0p3 > 0.9956
        - 0.009158496 * max(0.0, Q.lam1 - 0.001260456) / 0.006719699   # -0.9%  lam1 > 0.00126
        - 0.00799158 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # -0.8%  mass_over_sum_pt > 0.1709
        + 0.007975859 * max(0.0, Q.mass_top50 - 138.8977) / 3.930268   # +0.8%  mass_top50 > 138.9
        - 0.007936849 * max(0.0, 71.781 - Q.mass_top10) / 28.28521   # -0.8%  mass_top10 < 71.78
        + 0.007397618 * max(0.0, Q.mass - 64.48544) * max(0.0, 2.537109 - Q.soft2_pt) / 47.07428   # +0.7%  mass > 64.49 and soft2_pt < 2.537
        - 0.007144395 * max(0.0, 75.26407 - Q.mass_top15) / 22.29039   # -0.7%  mass_top15 < 75.26
        + 0.007143443 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # +0.7%  mass_over_sum_pt_sq > 0.0292
        + 0.006656635 * max(0.0, Q.mass_top5 - 14.54404) / 16.7137   # +0.7%  mass_top5 > 14.54
        - 0.006259122 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -0.6%  n_dr_0p2_0p4 < 11
        + 0.005570593 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +0.6%  dr_0 < 0.06413
        - 0.005379246 * max(0.0, 0.005312783 - Q.girth2_top20) / 0.001437214   # -0.5%  girth2_top20 < 0.005313
        - 0.00524404 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # -0.5%  girth2_top30 < 0.005402
        - 0.005052619 * max(0.0, 3.814159 - Q.D2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.04343614   # -0.5%  D2 < 3.814 and sj2_dr > 0.1938
        + 0.005027184 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, 464.75 - Q.sum_pt_top2) / 5683.5   # +0.5%  sj2_mass1 < 80.3 and sum_pt_top2 < 464.8
        - 0.004964844 * max(0.0, Q.M2 - 0.04577853) / 0.02232023   # -0.5%  M2 > 0.04578
        + 0.004922369 * max(0.0, 0.06940472 - Q.dr_1) / 0.0280884   # +0.5%  dr_1 < 0.0694
        + 0.004886359 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +0.5%  z_dr_0p2_0p4 < 0.09123
        - 0.00449188 * max(0.0, Q.mass - 162.8363) / 1.509832   # -0.4%  mass > 162.8
        + 0.003538226 * max(0.0, 0.01541561 - Q.e2) / 0.001436351   # +0.4%  e2 < 0.01542
        + 0.002288009 * max(0.0, 0.03577037 - Q.girth) / 0.004182456   # +0.2%  girth < 0.03577
        - 0.002136635 * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0004356739   # -0.2%  psi_0p3 > 0.9985
        + 0.002068638 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.0830523 - Q.dr_4) / 0.002292504   # +0.2%  log_sum_pt > 6.903 and dr_4 < 0.08305
        - 0.002029548 * max(0.0, Q.z_dr_0_0p05 - 0.7674734) / 0.05221088   # -0.2%  z_dr_0_0p05 > 0.7675
        + 0.001864184 * max(0.0, Q.mass - 82.85409) * max(0.0, Q.psi_0p3 - 0.9853273) / 0.08864606   # +0.2%  mass > 82.85 and psi_0p3 > 0.9853
        + 0.001523711 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.91979) / 145.9595   # +0.2%  sj2_mass1 < 80.3 and sj2_mass2 > 11.92
        + 0.001177867 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +0.1%  mass < 74.25
        + 0.001015476 * max(0.0, 111.2487 - Q.mass_top40) * max(0.0, Q.pt1_dr01 - 12.6865) / 28.78446   # +0.1%  mass_top40 < 111.2 and pt1_dr01 > 12.69
        + 0.0009003054 * max(0.0, Q.mass - 172.4888) / 0.7446233   # +0.1%  mass > 172.5
        - 0.000869499 * max(0.0, Q.mass_top40 - 163.2541) / 0.6399318   # -0.1%  mass_top40 > 163.3
        + 0.000487903 * max(0.0, Q.mass_top40 - 163.2541) * max(0.0, Q.soft4_z - 0.001186042) / 0.0003797437   # +0.0%  mass_top40 > 163.3 and soft4_z > 0.001186
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 13.58;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.58334 * (-0.03292285
        - 0.1207737 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # -12.1%  LHA < 0.3098
        + 0.1040826 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +10.4%  mass < 92.86
        + 0.0574964 * max(0.0, 0.3203321 - Q.LHA) / 0.07499257   # +5.7%  LHA < 0.3203
        + 0.05016536 * max(0.0, 0.2941033 - Q.LHA) / 0.05718399   # +5.0%  LHA < 0.2941
        - 0.04830436 * max(0.0, 85.8667 - Q.mass_top50) / 13.95835   # -4.8%  mass_top50 < 85.87
        + 0.04150399 * max(0.0, 97.93004 - Q.mass_top50) / 22.03533   # +4.2%  mass_top50 < 97.93
        - 0.03422139 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # -3.4%  girth < 0.07374
        - 0.03385041 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -3.4%  mass < 101
        + 0.03332929 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +3.3%  e2_sq < 0.008184
        + 0.03121092 * max(0.0, 0.2845608 - Q.LHA) / 0.05172777   # +3.1%  LHA < 0.2846
        + 0.02996844 * max(0.0, 0.004752876 - Q.girth2_top10) / 0.001623418   # +3.0%  girth2_top10 < 0.004753
        + 0.0299116 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # +3.0%  tau1 < 0.1073
        + 0.02976187 * max(0.0, 60.5496 - Q.mass_top10) / 20.08884   # +3.0%  mass_top10 < 60.55
        - 0.02909425 * max(0.0, 79.65241 - Q.mass) / 10.37103   # -2.9%  mass < 79.65
        + 0.02360605 * max(0.0, 0.02793599 - Q.e2) / 0.005715277   # +2.4%  e2 < 0.02794
        + 0.02353362 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +2.4%  n_dr_0p2_0p4 < 10
        - 0.0231118 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # -2.3%  tau1 < 0.08787
        - 0.01860483 * max(0.0, 0.006929741 - Q.girth2_top30) / 0.002004687   # -1.9%  girth2_top30 < 0.00693
        + 0.01855739 * max(0.0, Q.psi_0p3 - 0.9853273) / 0.009408723   # +1.9%  psi_0p3 > 0.9853
        - 0.01813085 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -1.8%  lam1 < 0.006717
        - 0.01647825 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # -1.6%  e2 < 0.04083
        - 0.01604889 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # -1.6%  girth2_top10 < 0.007679
        - 0.01581645 * max(0.0, 38.52415 - Q.mass_top10) / 8.8473   # -1.6%  mass_top10 < 38.52
        - 0.01529891 * max(0.0, 86.25221 - Q.mass_top30) / 18.20727   # -1.5%  mass_top30 < 86.25
        - 0.01434421 * max(0.0, 0.007639643 - Q.girth2_top2) / 0.004543329   # -1.4%  girth2_top2 < 0.00764
        - 0.01183531 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2) / 0.005195774   # -1.2%  n_dr_0p2_0p4 < 10 and girth2 < 0.005532
        + 0.01153538 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.mass_top10 - 38.52415) / 111.7968   # +1.2%  mass < 101 and mass_top10 > 38.52
        + 0.01060927 * max(0.0, 15.0 - Q.n_dr_0p1_0p2) / 5.139044   # +1.1%  n_dr_0p1_0p2 < 15
        - 0.01019464 * max(0.0, Q.z_top10_slots - 0.7271951) / 0.0653343   # -1.0%  z_top10_slots > 0.7272
        + 0.009700641 * max(0.0, 0.05048381 - Q.girth) / 0.008533025   # +1.0%  girth < 0.05048
        + 0.009233587 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +0.9%  mass_top30 < 60.44
        - 0.007646699 * max(0.0, 4.204324e-05 - Q.e3) / 1.197776e-05   # -0.8%  e3 < 4.204e-05
        + 0.007299561 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.7%  n_dr_0p2_0p4 < 6
        - 0.006615758 * max(0.0, Q.z_top30_slots - 0.9734886) / 0.01006171   # -0.7%  z_top30_slots > 0.9735
        + 0.006343078 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 0.05112769 - Q.C2_b2) / 0.0003814199   # +0.6%  z_top30_slots > 0.9735 and C2_b2 < 0.05113
        - 0.006240081 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # -0.6%  z_dr_0p2_0p4 < 0.06849
        + 0.005832488 * max(0.0, 0.144845 - Q.tau21_b2) / 0.01786964   # +0.6%  tau21_b2 < 0.1448
        + 0.005452645 * max(0.0, 0.1548383 - Q.z_dr_0p1_0p2) / 0.06710802   # +0.5%  z_dr_0p1_0p2 < 0.1548
        - 0.005113253 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0) / 30.16986   # -0.5%  n_dr_0p2_0p4 < 10 and n_particles > 34
        - 0.002235403 * max(0.0, 79.65241 - Q.mass) * max(0.0, 3.814159 - Q.D2) / 4.810976   # -0.2%  mass < 79.65 and D2 < 3.814
        - 0.001976193 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -0.2%  mass < 64.49
        + 0.001930006 * max(0.0, 0.003213724 - Q.girth2_top10) / 0.0009423838   # +0.2%  girth2_top10 < 0.003214
        - 0.001525227 * max(0.0, 79.65241 - Q.mass) * max(0.0, Q.z_11 - 0.01375115) / 0.05008222   # -0.2%  mass < 79.65 and z_11 > 0.01375
        - 0.001474945 * max(0.0, 0.09573228 - Q.z_dr_0_0p05) / 0.02128126   # -0.1%  z_dr_0_0p05 < 0.09573
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 2.911;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.910811 * (-0.1548695
        + 0.2380301 * max(0.0, 80.78464 - Q.mass) / 10.84952   # +23.8%  mass < 80.78
        + 0.1751966 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +17.5%  mass < 82.85
        + 0.08177978 * max(0.0, 87.36377 - Q.mass) / 14.19996   # +8.2%  mass < 87.36
        - 0.07356494 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -7.4%  mass < 53.87
        + 0.07172339 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # +7.2%  z_dr_0p2_0p4 < 0.06849
        - 0.06135452 * max(0.0, 94.64253 - Q.mass_top40) / 21.02021   # -6.1%  mass_top40 < 94.64
        - 0.04892882 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3) / 0.0414109   # -4.9%  mass < 87.36 and psi_0p3 < 0.999
        - 0.04660706 * max(0.0, 0.00363788 - Q.e2_sq) / 0.0004963098   # -4.7%  e2_sq < 0.003638
        - 0.04514679 * max(0.0, 77.93668 - Q.mass_top40) / 11.12028   # -4.5%  mass_top40 < 77.94
        - 0.03814164 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.6147674   # -3.8%  mass < 87.36 and z_dr_0p1_0p2 < 0.06473
        - 0.01875204 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -1.9%  sum_pt < 972
        - 0.01792362 * max(0.0, 0.06895248 - Q.mass_over_sum_pt) / 0.007729512   # -1.8%  mass_over_sum_pt < 0.06895
        + 0.01474231 * max(0.0, 0.0004005745 - Q.girth2_top5) / 6.9335e-05   # +1.5%  girth2_top5 < 0.0004006
        + 0.01374657 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) / 0.02385119   # +1.4%  z_dr_0p1_0p2 > 0.4479
        + 0.008584481 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.3036026 - Q.planar_flow) / 0.0008856905   # +0.9%  z_dr_0p1_0p2 > 0.4479 and planar_flow < 0.3036
        - 0.007689184 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.001776308 - Q.lam2) / 5.387726e-06   # -0.8%  z_dr_0p1_0p2 > 0.4479 and lam2 < 0.001776
        + 0.006839747 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.4026646) / 0.2505715   # +0.7%  mass < 87.36 and z_dr_0p05_0p1 > 0.4027
        + 0.00624105 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 435.25 - Q.pt_0) / 490.7786   # +0.6%  sj3_pair_mass_max > 122.7 and pt_0 < 435.2
        - 0.005846598 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.2502892 - Q.sj3_z3) / 0.002463794   # -0.6%  z_dr_0p1_0p2 > 0.4479 and sj3_z3 < 0.2503
        - 0.00575842 * max(0.0, Q.sum_pt - 1260.541) / 7.655483   # -0.6%  sum_pt > 1261
        - 0.005079622 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 0.05112769 - Q.C2_b2) / 0.2840034   # -0.5%  sum_pt < 972 and C2_b2 < 0.05113
        + 0.004575686 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1) / 9.827377   # +0.5%  sd_mass > 133.3 and sj3_mass1 < 32.5
        + 0.003746971 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 65.20727 - Q.sj2_mass1) / 34.79473   # +0.4%  sj3_pair_mass_max > 122.7 and sj2_mass1 < 65.21
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 6.067;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.067134 * (0.195578
        - 0.1092932 * max(0.0, Q.mass - 143.7876) / 3.946979   # -10.9%  mass > 143.8
        - 0.0994216 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -9.9%  sum_pt < 1013
        - 0.08441683 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -8.4%  sum_pt < 1085
        + 0.07866546 * max(0.0, 0.02497133 - Q.girth2_top40) / 0.01655385   # +7.9%  girth2_top40 < 0.02497
        + 0.07663359 * max(0.0, Q.mass - 64.48544) / 31.19164   # +7.7%  mass > 64.49
        - 0.06011345 * max(0.0, Q.mass_top50 - 97.93004) / 11.91764   # -6.0%  mass_top50 > 97.93
        + 0.05811668 * max(0.0, Q.mass_top50 - 138.8977) / 3.930268   # +5.8%  mass_top50 > 138.9
        + 0.05561421 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +5.6%  sum_pt_top40 < 1053
        + 0.04820021 * max(0.0, Q.mass - 78.26182) / 21.33658   # +4.8%  mass > 78.26
        + 0.04117574 * max(0.0, Q.psi_0p3 - 0.9777125) / 0.01571542   # +4.1%  psi_0p3 > 0.9777
        + 0.03816046 * max(0.0, 1008.935 - Q.sum_pt_top50) / 22.0436   # +3.8%  sum_pt_top50 < 1009
        - 0.03458598 * max(0.0, Q.mass_over_sum_pt - 0.06030419) / 0.03186977   # -3.5%  mass_over_sum_pt > 0.0603
        - 0.02481169 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -2.5%  mass_top50 > 157.5
        - 0.02052721 * max(0.0, 889.8383 - Q.sum_pt_top20) / 35.81237   # -2.1%  sum_pt_top20 < 889.8
        + 0.01947261 * max(0.0, Q.mass_top40 - 150.0144) / 1.666872   # +1.9%  mass_top40 > 150
        - 0.01816563 * max(0.0, 3.654944 - Q.pt_entropy) / 0.8125705   # -1.8%  pt_entropy < 3.655
        + 0.0164645 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.5260785 - Q.D3) / 23.2296   # +1.6%  sum_pt_top40 < 1053 and D3 < 0.5261
        - 0.01312934 * max(0.0, 14.0 - Q.n_for_90pct) / 1.135089   # -1.3%  n_for_90pct < 14
        + 0.01019976 * max(0.0, Q.mass_over_sum_pt - 0.07435617) / 0.02188714   # +1.0%  mass_over_sum_pt > 0.07436
        + 0.008889528 * max(0.0, Q.mass_top50 - 97.93004) * max(0.0, 2.873047 - Q.soft3_pt) / 19.23325   # +0.9%  mass_top50 > 97.93 and soft3_pt < 2.873
        + 0.008677786 * max(0.0, Q.mass_top5 - 37.76455) / 5.869288   # +0.9%  mass_top5 > 37.76
        + 0.008119933 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # +0.8%  sum_pt_top50 < 959.1
        + 0.008000699 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # +0.8%  tau1 < 0.08787
        + 0.007812306 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # +0.8%  log_sum_pt > 7.139
        - 0.007528582 * max(0.0, Q.sum_pt - 1260.541) / 7.655483   # -0.8%  sum_pt > 1261
        + 0.006781247 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +0.7%  mass_over_sum_pt > 0.1709
        - 0.005845315 * max(0.0, 6.879399 - Q.log_sum_pt) / 0.01083309   # -0.6%  log_sum_pt < 6.879
        + 0.005146199 * max(0.0, Q.mass_top10 - 60.5496) / 6.712581   # +0.5%  mass_top10 > 60.55
        - 0.004990391 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # -0.5%  mass_top10 > 71.78
        + 0.004194602 * max(0.0, 14.0 - Q.n_for_90pct) * max(0.0, 3.345339 - Q.D2) / 0.7823248   # +0.4%  n_for_90pct < 14 and D2 < 3.345
        + 0.004149103 * max(0.0, Q.lam1 - 0.01649354) / 0.001003295   # +0.4%  lam1 > 0.01649
        - 0.003672397 * max(0.0, Q.mass - 172.4888) * max(0.0, 41.4375 - Q.pt_9) / 7.126564   # -0.4%  mass > 172.5 and pt_9 < 41.44
        + 0.003577732 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.01833434 - Q.z_9) / 0.01801529   # +0.4%  sum_pt < 1013 and z_9 < 0.01833
        - 0.00229684 * max(0.0, Q.mass_top50 - 168.9698) / 0.6651775   # -0.2%  mass_top50 > 169
        - 0.001356297 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.00317537 - Q.C3) / 7.466262e-06   # -0.1%  log_sum_pt > 7.139 and C3 < 0.003175
        + 0.001013196 * max(0.0, Q.mass - 172.4888) * max(0.0, 0.002357824 - Q.soft6_z) / 0.0004574479   # +0.1%  mass > 172.5 and soft6_z < 0.002358
        - 0.0006428102 * max(0.0, Q.mass - 172.4888) / 0.7446233   # -0.1%  mass > 172.5
        + 0.0001368581 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, -0.0892334 - Q.eta_1) / 1.48343e-05   # +0.0%  log_sum_pt < 6.811 and eta_1 < -0.08923
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 17.56;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.55618 * (0.03332285
        - 0.127964 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) / 0.0178873   # -12.8%  mass_over_sum_pt < 0.09047
        + 0.09869233 * max(0.0, 0.01986381 - Q.mass_over_sum_pt_sq) / 0.0117775   # +9.9%  mass_over_sum_pt_sq < 0.01986
        - 0.09424702 * max(0.0, 91.03469 - Q.mass) / 16.36287   # -9.4%  mass < 91.03
        + 0.07907186 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +7.9%  mass_over_sum_pt < 0.1182
        - 0.07094912 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -7.1%  mass < 82.85
        + 0.05074645 * max(0.0, 0.08873143 - Q.mass_over_sum_pt) / 0.01671255   # +5.1%  mass_over_sum_pt < 0.08873
        - 0.04782616 * max(0.0, 0.01292642 - Q.girth2_top40) / 0.006292908   # -4.8%  girth2_top40 < 0.01293
        - 0.04009428 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -4.0%  sum_pt_top50 > 959.1
        - 0.0399469 * max(0.0, 0.01897915 - Q.girth2_top40) / 0.01130444   # -4.0%  girth2_top40 < 0.01898
        + 0.03405111 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +3.4%  mass_over_sum_pt < 0.09795
        - 0.0295006 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -3.0%  girth2_top20 < 0.01083
        - 0.02472122 * max(0.0, 0.008956554 - Q.girth2_top10) / 0.004473389   # -2.5%  girth2_top10 < 0.008957
        + 0.02328159 * max(0.0, Q.log_sum_pt - 6.935549) / 0.0371538   # +2.3%  log_sum_pt > 6.936
        + 0.02295299 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # +2.3%  girth2_top10 < 0.007679
        + 0.02185776 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +2.2%  mass < 121.4
        - 0.01601196 * max(0.0, Q.n_pt_above_10 - 11.0) / 8.944079   # -1.6%  n_pt_above_10 > 11
        + 0.01419957 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +1.4%  mass_top40 < 67.73
        + 0.01412579 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +1.4%  tau21_b2 < 0.3425
        + 0.01340966 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 1.059188 - Q.N3) / 0.002548597   # +1.3%  psi_0p3 > 0.9924 and N3 < 1.059
        + 0.0122445 * max(0.0, 0.03680582 - Q.e2) / 0.01052402   # +1.2%  e2 < 0.03681
        - 0.01172824 * max(0.0, 19.0 - Q.n_dr_0p1_0p2) / 8.166005   # -1.2%  n_dr_0p1_0p2 < 19
        - 0.01154075 * max(0.0, Q.e2 - 0.02210818) / 0.01219843   # -1.2%  e2 > 0.02211
        - 0.01031295 * max(0.0, 0.04142826 - Q.tau2) / 0.01568306   # -1.0%  tau2 < 0.04143
        + 0.009730683 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +1.0%  e2 < 0.0303
        - 0.009182838 * max(0.0, 7.876005e-05 - Q.e3) / 3.594075e-05   # -0.9%  e3 < 7.876e-05
        - 0.007967677 * max(0.0, 0.001592178 - Q.girth2_top3) / 0.000513961   # -0.8%  girth2_top3 < 0.001592
        - 0.007004101 * max(0.0, 0.1008497 - Q.tau1) / 0.02962944   # -0.7%  tau1 < 0.1008
        - 0.006179835 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2738063) / 0.009655725   # -0.6%  N2 < 0.4227 and max_dr > 0.2738
        - 0.00615254 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.0458688 - Q.dr_3) / 4.832428e-05   # -0.6%  psi_0p3 > 0.9924 and dr_3 < 0.04587
        + 0.005728496 * max(0.0, 0.03948167 - Q.dr_3) / 0.0095916   # +0.6%  dr_3 < 0.03948
        + 0.004445583 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # +0.4%  girth2_top20 < 0.006374
        + 0.004116765 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.n_for_50pct - 5.0) / 4.765654   # +0.4%  mass < 82.85 and n_for_50pct > 5
        + 0.004012004 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.002405315 - Q.soft8_z) / 0.0166304   # +0.4%  mass < 91.03 and soft8_z < 0.002405
        - 0.003721908 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -0.4%  tau1 < 0.07057
        + 0.003691678 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.01691783 - Q.C2_b2) / 0.001347445   # +0.4%  tau21_b2 < 0.3425 and C2_b2 < 0.01692
        + 0.003620649 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, Q.pt_3 - 60.59375) / 0.06871383   # +0.4%  psi_0p3 > 0.9924 and pt_3 > 60.59
        - 0.003233563 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -9.840088) / 3.025948   # -0.3%  tau21_b2 < 0.3425 and orientation_deg > -9.84
        - 0.002845983 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.zdr_3 - 0.0003281655) / 0.08597332   # -0.3%  mass < 121.4 and zdr_3 > 0.0003282
        + 0.002410694 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 4.250195 - Q.soft6_pt) / 0.009807099   # +0.2%  psi_0p3 > 0.9924 and soft6_pt < 4.25
        - 0.002217719 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.1231747 - Q.z_1) / 0.00154957   # -0.2%  tau21_b2 < 0.3425 and z_1 < 0.1232
        - 0.001719819 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -0.2%  mass_top50 > 157.5
        - 0.001296769 * max(0.0, 0.01256572 - Q.e2) / 0.0008030352   # -0.1%  e2 < 0.01257
        - 0.001243928 * max(0.0, Q.sum_pt_top40 - 906.6023) / 122.3672   # -0.1%  sum_pt_top40 > 906.6
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 10.11;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.10608 * (0.0001144733
        + 0.08743151 * max(0.0, 0.08068193 - Q.girth) / 0.02368455   # +8.7%  girth < 0.08068
        + 0.06623815 * max(0.0, 0.140939 - Q.mass_over_sum_pt) / 0.05796036   # +6.6%  mass_over_sum_pt < 0.1409
        + 0.06229099 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +6.2%  log_sum_pt < 7.017
        - 0.06040414 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # -6.0%  LHA < 0.3024
        + 0.04247104 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +4.2%  girth2_top5 < 0.00833
        - 0.03754884 * max(0.0, Q.lam1 - 0.01649354) / 0.001003295   # -3.8%  lam1 > 0.01649
        + 0.03616223 * max(0.0, 80.24626 - Q.mass_top30) / 14.57104   # +3.6%  mass_top30 < 80.25
        - 0.03576933 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # -3.6%  e3 < 0.0005178
        + 0.03428168 * max(0.0, 79.18312 - Q.sd_mass) / 30.46368   # +3.4%  sd_mass < 79.18
        - 0.03221612 * max(0.0, Q.psi_0p2 - 0.8706159) / 0.08984765   # -3.2%  psi_0p2 > 0.8706
        - 0.03133791 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -3.1%  tau1 < 0.07057
        - 0.0308002 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # -3.1%  lam1 < 0.01174
        - 0.02843663 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # -2.8%  e2 > 0.0303
        + 0.02562697 * max(0.0, Q.lam1 - 0.006189818) / 0.003311577   # +2.6%  lam1 > 0.00619
        + 0.02296284 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +2.3%  z_dr_0p1_0p2 < 0.1203
        - 0.02189357 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -2.2%  girth2_top30 < 0.01216
        - 0.02183498 * max(0.0, 1017.778 - Q.sum_pt_top20) / 107.0228   # -2.2%  sum_pt_top20 < 1018
        - 0.02000214 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -2.0%  psi_0p3 > 0.9897
        - 0.01996278 * max(0.0, 0.08329945 - Q.mass_over_sum_pt) / 0.0135093   # -2.0%  mass_over_sum_pt < 0.0833
        - 0.01915456 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) / 0.3666529   # -1.9%  z_dr_0_0p05 < 0.8459
        - 0.01884317 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -1.9%  sum_pt < 1002
        + 0.01835634 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # +1.8%  girth2_top10 < 0.007679
        + 0.01752639 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) / 3.82104   # +1.8%  n_dr_0p1_0p2 < 13
        + 0.01436528 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +1.4%  lam2 < 0.001776
        - 0.01420873 * max(0.0, 40.97891 - Q.sd_mass) / 11.60739   # -1.4%  sd_mass < 40.98
        + 0.01291637 * max(0.0, Q.sj3_dr_max - 0.2442262) / 0.04751597   # +1.3%  sj3_dr_max > 0.2442
        - 0.0126606 * max(0.0, 0.07031778 - Q.girth) / 0.01719521   # -1.3%  girth < 0.07032
        - 0.01188416 * max(0.0, 0.002270363 - Q.girth2_top5) / 0.0007634099   # -1.2%  girth2_top5 < 0.00227
        + 0.01161228 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 1260.541 - Q.sum_pt) / 87.05552   # +1.2%  z_dr_0_0p05 < 0.8459 and sum_pt < 1261
        - 0.01142528 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -1.1%  log_sum_pt < 6.903
        + 0.01099401 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +1.1%  sj2_dr > 0.2232
        - 0.01065088 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # -1.1%  mass_top50 < 71.8
        - 0.009365773 * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 3.233647   # -0.9%  n_dr_0p2_0p4 > 9
        - 0.008727795 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583) / 0.0009330603   # -0.9%  girth2_top5 < 0.00833 and sj3_pairmin_over_m > 0.09541
        - 0.008427041 * max(0.0, 1018.832 - Q.sum_pt_top30) / 55.61816   # -0.8%  sum_pt_top30 < 1019
        + 0.008375484 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +0.8%  sum_pt < 986.1
        - 0.007821684 * max(0.0, 68.28697 - Q.mass_top30) / 9.510711   # -0.8%  mass_top30 < 68.29
        - 0.007765564 * max(0.0, Q.sj3_dr_max - 0.2841485) / 0.0316155   # -0.8%  sj3_dr_max > 0.2841
        - 0.007053105 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677) / 0.04548726   # -0.7%  girth2_top5 < 0.00833 and sj3_mass1 > 5.113
        + 0.006666268 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.7%  lam1 < 0.004673
        + 0.006627838 * max(0.0, 0.2091025 - Q.LHA) / 0.02095782   # +0.7%  LHA < 0.2091
        + 0.005097103 * max(0.0, 869.693 - Q.sum_pt_top20) / 29.2727   # +0.5%  sum_pt_top20 < 869.7
        - 0.004560999 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.02970886 - Q.absphi_0) / 0.06277088   # -0.5%  n_dr_0p1_0p2 < 13 and absphi_0 < 0.02971
        - 0.003901309 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -0.4%  psi_0p1 > 0.8976
        - 0.00389424 * max(0.0, Q.sj3_dr_max - 0.3483732) / 0.01384956   # -0.4%  sj3_dr_max > 0.3484
        - 0.003732568 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0) / 0.2039042   # -0.4%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p2_0p4 > 5
        - 0.001900334 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.n_dr_0p05_0p1 - 8.0) / 14.10129   # -0.2%  n_dr_0p1_0p2 < 13 and n_dr_0p05_0p1 > 8
        + 0.001406266 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt) / 2.722316   # +0.1%  z_dr_0_0p05 < 0.8459 and sum_pt < 949.9
        - 0.001240477 * max(0.0, Q.sj2_dr - 0.1666442) / 0.04983259   # -0.1%  sj2_dr > 0.1666
        - 0.0008908397 * max(0.0, 0.2982 - Q.max_dr) / 0.01350087   # -0.1%  max_dr < 0.2982
        - 0.0002752004 * max(0.0, 52.15154 - Q.mass_top50) / 3.332261   # -0.0%  mass_top50 < 52.15
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6083603991596639, 2.8572145483193276, 0.21974075630252102, 0.39818387605042016, 0.7562030462184874, 1.0868807773109244, 0.5164110294117648, 0.3549723739495798, 1.0883213235294118, 1.1072518907563025, 1.181958718487395, 0.7659356092436975, 0.5069609243697479, 1.5199653361344538, 0.2645715336134454, 0.3852855042016807]
T = [3.9086380005908614, 2.5036800059086137, 4.064153458180146, 4.057750837053572, 3.9322638548778883]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +46%, n4 -17%, n9 +15%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.456875 * h[1] / H_AVG[1]
            - 0.169286 * h[4] / H_AVG[4]
            + 0.146068 * h[9] / H_AVG[9]
            - 0.08689734 * h[5] / H_AVG[5]
            - 0.0764046 * h[3] / H_AVG[3]
            + 0.03039907 * h[12] / H_AVG[12]
            + 0.02064382 * h[6] / H_AVG[6]
            + 0.008701251 * h[8] / H_AVG[8]
            - 0.004724946 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -32%, n9 +25%, n1 -21%, n11 -8%, n12 +7%, n6 +5% ...
            - 0.3209139 * h[4] / H_AVG[4]
            + 0.2487655 * h[9] / H_AVG[9]
            - 0.2139761 * h[1] / H_AVG[1]
            - 0.07648098 * h[11] / H_AVG[11]
            + 0.06960467 * h[12] / H_AVG[12]
            + 0.04511955 * h[6] / H_AVG[6]
            + 0.01097089 * h[2] / H_AVG[2]
            - 0.007376384 * h[10] / H_AVG[10]
            + 0.00679201 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -23%, n5 +15%, n0 +11%, n11 +11%, n14 -9%, n4 +6% ...
            - 0.2343123 * h[8] / H_AVG[8]
            + 0.1546086 * h[5] / H_AVG[5]
            + 0.112267 * h[0] / H_AVG[0]
            + 0.1118989 * h[11] / H_AVG[11]
            - 0.08951086 * h[14] / H_AVG[14]
            + 0.06396038 * h[4] / H_AVG[4]
            - 0.059597 * h[9] / H_AVG[9]
            - 0.05458892 * h[7] / H_AVG[7]
            - 0.05067547 * h[12] / H_AVG[12]
            + 0.04286389 * h[3] / H_AVG[3]
            - 0.01777517 * h[15] / H_AVG[15]
            + 0.007941553 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -25%, n0 -21%, n5 +18%, n6 -12%, n7 +8%, n15 +5% ...
            - 0.251445 * h[8] / H_AVG[8]
            - 0.2061476 * h[0] / H_AVG[0]
            + 0.1841489 * h[5] / H_AVG[5]
            - 0.1232883 * h[6] / H_AVG[6]
            + 0.07927882 * h[7] / H_AVG[7]
            + 0.05340966 * h[15] / H_AVG[15]
            + 0.04293153 * h[3] / H_AVG[3]
            + 0.03904264 * h[12] / H_AVG[12]
            - 0.0203075 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -35%, n10 +30%, n5 -13%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3502991 * h[13] / H_AVG[13]
            + 0.2958831 * h[10] / H_AVG[10]
            - 0.1295629 * h[5] / H_AVG[5]
            + 0.05838056 * h[8] / H_AVG[8]
            - 0.04834628 * h[12] / H_AVG[12]
            - 0.03674272 * h[15] / H_AVG[15]
            + 0.03605762 * h[4] / H_AVG[4]
            - 0.02538893 * h[7] / H_AVG[7]
            + 0.01933875 * h[0] / H_AVG[0]
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
