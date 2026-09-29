"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  12.8%   (on for 51% of jets)
  neuron  1:  11.9%   (on for 93% of jets)
  neuron  5:  11.3%   (on for 78% of jets)
  neuron  4:  10.1%   (on for 66% of jets)
  neuron  0:   7.7%   (on for 59% of jets)
  neuron 13:   7.4%   (on for 87% of jets)
  neuron  9:   6.8%   (on for 65% of jets)
  neuron 10:   6.8%   (on for 78% of jets)
  neuron 12:   4.9%   (on for 56% of jets)
  neuron  7:   4.5%   (on for 29% of jets)
  neuron  6:   4.0%   (on for 60% of jets)
  neuron 11:   3.4%   (on for 69% of jets)
  neuron  3:   3.3%   (on for 52% of jets)
  neuron 15:   2.6%   (on for 65% of jets)
  neuron 14:   1.9%   (on for 32% of jets)
  neuron  2:   0.6%   (on for 39% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.2% (the network: 81.1%); same class as the network for 93.7% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
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
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_13         mass of particles 0 and 13 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_10                  pT of particle 10 [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_pt              pT [GeV] of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_pt               pT [GeV] of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.z_2                    pT of particle 2 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_4                    pT of particle 4 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_11                 pT share × ΔR of particle 11 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_13              |Δφ| of particle 13
  Q.absphi_4               |Δφ| of particle 4
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft10_dr0             ΔR between the hardest and the 10. softest real particle (0 if among the 15 hardest)
  Q.soft3_dr0              ΔR between the hardest and the 3. softest real particle (0 if among the 15 hardest)
  Q.dr1_12                 ΔR between particle 12 and the 2nd-hardest particle
  Q.dr1_5                  ΔR between particle 5 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_9                   ΔR of particle 9 from the jet axis
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
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
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
  Q.pt_dispersion          √(Σ pTᵢ²) / Σ pTᵢ
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
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_13=pair_mass(0, 13),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_0=pt[0],
        pt_10=pt[10],
        pt_11=pt[11],
        pt_2=pt[2],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_6=pt[6],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft10_pt=softp(10, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft7_pt=softp(7, 'pt'),
        soft9_pt=softp(9, 'pt'),
        z_11=z[11],
        z_2=z[2],
        z_3=z[3],
        z_4=z[4],
        z_6=z[6],
        z_7=z[7],
        z_9=z[9],
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft7_z=softp(7, 'z'),
        soft8_z=softp(8, 'z'),
        sj3_z3=subjets(3)["z"][2],
        z_top15_slots=sum(pt[:15]) / tot,
        z_top2_slots=sum(pt[:2]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_11=z[11] * dr[11],
        zdr_2=z[2] * dr[2],
        zdr_4=z[4] * dr[4],
        zdr_5=z[5] * dr[5],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        absphi_0=abs(phi[0]),
        absphi_13=abs(phi[13]),
        absphi_4=abs(phi[4]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft10_dr0=softp(10, 'dr0'),
        soft3_dr0=softp(3, 'dr0'),
        dr1_12=math.sqrt(dist2(1, 12)) if pt[12] > 0 else 0.0,
        dr1_5=math.sqrt(dist2(1, 5)) if pt[5] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_9=dr[9] if pt[9] > 0 else 0.0,
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
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
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
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 14.12;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.12206 * (0.08472197
        - 0.1698387 * max(0.0, Q.mass - 78.26182) / 21.33658   # -17.0%  mass > 78.26
        - 0.09025319 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -9.0%  mass_top50 > 82.04
        - 0.06889107 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -6.9%  mass < 101
        + 0.06603784 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 0.002241068   # +6.6%  mass_over_sum_pt_sq < 0.007873
        + 0.06382778 * max(0.0, Q.mass - 91.19) / 15.01545   # +6.4%  mass > 91.19
        + 0.04263327 * max(0.0, Q.mass - 89.74183) / 15.55572   # +4.3%  mass > 89.74
        + 0.04242608 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq) / 0.001689525   # +4.2%  mass_over_sum_pt_sq < 0.006939
        + 0.04200932 * max(0.0, Q.mass - 92.85979) / 14.48273   # +4.2%  mass > 92.86
        - 0.04165943 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -4.2%  e2_sq < 0.006167
        - 0.04049816 * max(0.0, Q.mass - 74.25181) / 24.07648   # -4.0%  mass > 74.25
        + 0.03726327 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +3.7%  log_sum_pt < 7.017
        + 0.03032139 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +3.0%  n_dr_0p2_0p4 < 15
        - 0.02556977 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -2.6%  girth2_top30 < 0.006364
        + 0.02509813 * max(0.0, Q.mass_top50 - 71.79516) / 24.22501   # +2.5%  mass_top50 > 71.8
        - 0.02479039 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # -2.5%  z_dr_0p2_0p4 < 0.09123
        - 0.01965285 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # -2.0%  girth2_top20 < 0.006374
        - 0.01710291 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -1.7%  tau1 < 0.07057
        + 0.01652471 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # +1.7%  e3 < 0.0005178
        + 0.01431123 * max(0.0, 80.4 - Q.mass_top30) / 14.65577   # +1.4%  mass_top30 < 80.4
        - 0.01294224 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # -1.3%  sum_pt_top40 < 1070
        - 0.01253915 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -1.3%  sum_pt < 1013
        + 0.01196122 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +1.2%  sum_pt_top50 < 1157
        + 0.008871277 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +0.9%  psi_0p3 > 0.9956
        - 0.006717725 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -0.7%  lam1 < 0.005914
        + 0.006335526 * max(0.0, 0.004754444 - Q.mass_over_sum_pt_sq) / 0.0007997283   # +0.6%  mass_over_sum_pt_sq < 0.004754
        + 0.006130091 * max(0.0, 0.007538019 - Q.girth2_top20) / 0.002740841   # +0.6%  girth2_top20 < 0.007538
        + 0.005766042 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +0.6%  e2 < 0.01879
        - 0.005315539 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -0.5%  n_dr_0p2_0p4 < 11
        - 0.004956258 * max(0.0, 0.9906378 - Q.z_top50_slots) / 0.004783315   # -0.5%  z_top50_slots < 0.9906
        + 0.00481225 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957) / 1.852228   # +0.5%  log_sum_pt < 6.989 and sum_pt_top50 > 959.1
        - 0.004481544 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # -0.4%  mass_top40 < 80.89
        + 0.003662803 * max(0.0, 0.005312783 - Q.girth2_top20) / 0.001437214   # +0.4%  girth2_top20 < 0.005313
        - 0.003660808 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # -0.4%  sj2_dr > 0.2232
        - 0.003315405 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -0.3%  sum_pt < 972
        + 0.0032285 * max(0.0, 0.007856958 - Q.girth2_top30) / 0.002601721   # +0.3%  girth2_top30 < 0.007857
        - 0.002928567 * max(0.0, 0.05602756 - Q.C2) / 0.009882647   # -0.3%  C2 < 0.05603
        + 0.002873975 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +0.3%  lam2 < 0.0006155
        + 0.002813267 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) * max(0.0, 34.0625 - Q.pt_10) / 0.02980095   # +0.3%  mass_over_sum_pt_sq < 0.007873 and pt_10 < 34.06
        + 0.002396881 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) * max(0.0, 118.5 - Q.pt_2) / 1.543055   # +0.2%  z_dr_0p2_0p4 < 0.09123 and pt_2 < 118.5
        - 0.001746807 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # -0.2%  e2 > 0.05557
        + 0.001563005 * max(0.0, 846.1934 - Q.sum_pt_top20) / 22.83716   # +0.2%  sum_pt_top20 < 846.2
        + 0.001219456 * max(0.0, 53.87362 - Q.mass) / 3.56547   # +0.1%  mass < 53.87
        + 0.0005816747 * max(0.0, 858.8262 - Q.sum_pt_top40) / 3.625959   # +0.1%  sum_pt_top40 < 858.8
        + 0.0004705117 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 0.05554124 - Q.z_4) / 0.06569978   # +0.0%  sum_pt < 972 and z_4 < 0.05554
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 20.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.63845 * (0.06522236
        + 0.07869826 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # +7.9%  log_sum_pt > 6.894
        - 0.06050497 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -6.1%  log_sum_pt < 7.139
        + 0.05980263 * max(0.0, 0.1953848 - Q.tau1) / 0.1097382   # +6.0%  tau1 < 0.1954
        + 0.04993799 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +5.0%  log_sum_pt > 6.91
        - 0.04563688 * max(0.0, Q.z_top50_slots - 0.9586536) / 0.03427112   # -4.6%  z_top50_slots > 0.9587
        - 0.04211163 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -4.2%  sum_pt_top50 > 959.1
        - 0.04125962 * max(0.0, 2.275391 - Q.soft1_pt) / 1.652278   # -4.1%  soft1_pt < 2.275
        - 0.04117075 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # -4.1%  mass_over_sum_pt_sq < 0.009595
        - 0.03885553 * max(0.0, 120.6 - Q.mass) / 38.8279   # -3.9%  mass < 120.6
        + 0.03862903 * max(0.0, Q.pt_entropy - 2.07371) / 0.8049786   # +3.9%  pt_entropy > 2.074
        + 0.03351065 * max(0.0, 1.521582 - Q.soft1_pt) / 0.9527349   # +3.4%  soft1_pt < 1.522
        + 0.0257185 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # +2.6%  sum_pt_top40 < 1070
        + 0.02259869 * max(0.0, 0.008031986 - Q.girth2_top40) / 0.002514593   # +2.3%  girth2_top40 < 0.008032
        + 0.02209753 * max(0.0, 0.04516808 - Q.tau3) / 0.02499032   # +2.2%  tau3 < 0.04517
        - 0.02200209 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -2.2%  log_sum_pt > 6.959
        + 0.01947347 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.2734   # +1.9%  sum_pt_top2 < 689.2
        + 0.01777946 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +1.8%  n_particles > 38
        - 0.01761983 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # -1.8%  sj3_mass1 < 32.5
        + 0.0158051 * max(0.0, 0.02412652 - Q.girth2_top30) / 0.01613825   # +1.6%  girth2_top30 < 0.02413
        - 0.01241878 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2) / 190.5117   # -1.2%  sj3_mass1 < 32.5 and sj3_mass2 < 18.68
        - 0.01211107 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -1.2%  log_sum_pt > 6.989
        - 0.01175179 * max(0.0, 47.88842 - Q.mass_top20) / 6.040308   # -1.2%  mass_top20 < 47.89
        - 0.0116934 * max(0.0, Q.n_for_90pct - 11.0) / 10.33888   # -1.2%  n_for_90pct > 11
        - 0.01168658 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt) / 15.05788   # -1.2%  n_particles > 38 and soft1_pt < 2.275
        - 0.01013876 * max(0.0, Q.z_top20_slots - 0.8965411) * max(0.0, 0.03843804 - Q.dr_2) / 0.0005290849   # -1.0%  z_top20_slots > 0.8965 and dr_2 < 0.03844
        + 0.01000591 * max(0.0, Q.z_top20_slots - 0.8965411) / 0.0341307   # +1.0%  z_top20_slots > 0.8965
        + 0.00990029 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.0%  z_dr_0p2_0p4 < 0.09123
        + 0.009034474 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +0.9%  sum_pt < 1017
        + 0.008786308 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.9%  lam1 < 0.004673
        - 0.008293953 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.932092   # -0.8%  n_dr_0p2_0p4 < 13
        + 0.008253478 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2) / 0.0008500795   # +0.8%  z_top30_slots > 0.9342 and C2 < 0.0728
        - 0.008243982 * max(0.0, 31.35938 - Q.pt_9) / 6.538304   # -0.8%  pt_9 < 31.36
        - 0.007750873 * max(0.0, 30.26161 - Q.sj2_mass1) / 7.997784   # -0.8%  sj2_mass1 < 30.26
        + 0.007657761 * max(0.0, 80.35535 - Q.mass_top50) / 11.1399   # +0.8%  mass_top50 < 80.36
        + 0.007186601 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1) / 1.005598   # +0.7%  n_particles > 38 and dr_1 < 0.1612
        + 0.006932783 * max(0.0, 0.007538019 - Q.girth2_top20) / 0.002740841   # +0.7%  girth2_top20 < 0.007538
        + 0.006711302 * max(0.0, 120.6 - Q.mass) * max(0.0, 0.03843804 - Q.dr_2) / 0.6524502   # +0.7%  mass < 120.6 and dr_2 < 0.03844
        + 0.005954032 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # +0.6%  LHA < 0.2601
        - 0.005818859 * max(0.0, 6.160019 - Q.sj3_mass3) / 2.534872   # -0.6%  sj3_mass3 < 6.16
        + 0.005679528 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 39.23115   # +0.6%  mass_top20 < 47.89 and n_real_top40 > 29
        + 0.005506468 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0) / 0.570275   # +0.6%  n_particles > 38 and dr_0 < 0.112
        - 0.005269483 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -0.5%  n_dr_0p2_0p4 < 7
        - 0.005217773 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # -0.5%  M3 < 0.03188
        - 0.005039067 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # -0.5%  e2 < 0.02516
        - 0.004922864 * max(0.0, Q.z_top30_slots - 0.9341838) / 0.03406818   # -0.5%  z_top30_slots > 0.9342
        + 0.004908581 * max(0.0, 12.0 - Q.n_dr_0_0p05) / 3.466761   # +0.5%  n_dr_0_0p05 < 12
        - 0.004641867 * max(0.0, Q.pt_entropy - 2.07371) * max(0.0, 0.3797 - Q.soft3_dr) / 0.1539306   # -0.5%  pt_entropy > 2.074 and soft3_dr < 0.3797
        + 0.004250906 * max(0.0, 0.001594761 - Q.soft5_z) / 0.0005119901   # +0.4%  soft5_z < 0.001595
        - 0.004211661 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.4%  psi_0p3 > 0.998
        - 0.004191115 * max(0.0, Q.sum_pt_top5 - 430.75) / 176.7518   # -0.4%  sum_pt_top5 > 430.8
        - 0.003587917 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, Q.soft3_dr0 - 0.02918107) / 57.76427   # -0.4%  sum_pt_top2 < 689.2 and soft3_dr0 > 0.02918
        - 0.003523399 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -0.4%  z_dr_0_0p05 > 0.8789
        + 0.003221873 * max(0.0, 0.001823079 - Q.girth2_top20) / 0.000269929   # +0.3%  girth2_top20 < 0.001823
        + 0.003034164 * max(0.0, 0.0006570502 - Q.girth2_top5) / 0.0001395474   # +0.3%  girth2_top5 < 0.0006571
        - 0.002932142 * max(0.0, 7.139296 - Q.log_sum_pt) * max(0.0, 5.351077 - Q.pt1_dr01) / 0.6416883   # -0.3%  log_sum_pt < 7.139 and pt1_dr01 < 5.351
        + 0.002927804 * max(0.0, Q.psi_0p1 - 0.8747961) / 0.03440015   # +0.3%  psi_0p1 > 0.8748
        - 0.002918972 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.3%  log_sum_pt > 7.063
        + 0.002864507 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793) / 0.1941151   # +0.3%  z_top30_slots > 0.9342 and max_pair_mass > 13.05
        + 0.002807991 * max(0.0, 0.1416054 - Q.D3) / 0.05511785   # +0.3%  D3 < 0.1416
        + 0.002733386 * max(0.0, 0.007538019 - Q.girth2_top20) * max(0.0, Q.eta_1 - -0.04302979) / 0.000120198   # +0.3%  girth2_top20 < 0.007538 and eta_1 > -0.04303
        - 0.00262716 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.3%  n_dr_0p1_0p2 < 8
        + 0.002622178 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, Q.sj3_mass2 - 2.12465) / 2353.378   # +0.3%  sum_pt_top2 < 689.2 and sj3_mass2 > 2.125
        + 0.002449429 * max(0.0, 1.788105 - Q.D2) / 0.2865857   # +0.2%  D2 < 1.788
        + 0.002439107 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 11.29505 - Q.pair_mass_0_13) / 38.90723   # +0.2%  pt_9 < 31.36 and pair_mass_0_13 < 11.3
        + 0.002405315 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 0.3241858 - Q.dr1_12) / 1.371943   # +0.2%  pt_9 < 31.36 and dr1_12 < 0.3242
        - 0.001953464 * max(0.0, 0.03187688 - Q.M3) * max(0.0, Q.M2 - 0.05568888) / 0.0001450223   # -0.2%  M3 < 0.03188 and M2 > 0.05569
        - 0.001911088 * Q.absphi_0 / 0.03295052   # -0.2%  absphi_0
        + 0.00191035 * max(0.0, 0.08974974 - Q.sj3_z3) / 0.02798989   # +0.2%  sj3_z3 < 0.08975
        + 0.001882373 * max(0.0, 0.0007894752 - Q.girth2_top15) / 9.062109e-05   # +0.2%  girth2_top15 < 0.0007895
        - 0.001613893 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 0.02980347 - Q.eta_0) / 5.83951   # -0.2%  sum_pt_top5 > 430.8 and eta_0 < 0.0298
        + 0.001605473 * max(0.0, Q.sum_pt_top30 - 1191.938) / 6.938323   # +0.2%  sum_pt_top30 > 1192
        - 0.001598245 * max(0.0, 0.003468228 - Q.C2_b2) / 0.0005968599   # -0.2%  C2_b2 < 0.003468
        + 0.001587881 * max(0.0, Q.mass_top10 - 56.92192) / 8.071231   # +0.2%  mass_top10 > 56.92
        + 0.00156619 * max(0.0, 30.26161 - Q.sj2_mass1) * max(0.0, 0.2366434 - Q.soft10_dr0) / 1.029958   # +0.2%  sj2_mass1 < 30.26 and soft10_dr0 < 0.2366
        + 0.001553941 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.2313408 - Q.sj3_dr12) / 0.7781115   # +0.2%  n_particles > 38 and sj3_dr12 < 0.2313
        + 0.001545219 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1546554 - Q.dr_9) / 0.7931187   # +0.2%  n_particles > 38 and dr_9 < 0.1547
        + 0.001483987 * max(0.0, Q.sj3_dr_max - 0.3276439) / 0.01855333   # +0.1%  sj3_dr_max > 0.3276
        + 0.001455321 * max(0.0, Q.n_particles - 38.0) * max(0.0, 57.87349 - Q.mass_top15) / 116.4836   # +0.1%  n_particles > 38 and mass_top15 < 57.87
        + 0.001237923 * max(0.0, 42.41192 - Q.mass_top30) / 2.481216   # +0.1%  mass_top30 < 42.41
        + 0.001215837 * max(0.0, 0.003270031 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 6.08612e-07   # +0.1%  girth2_top15 < 0.00327 and psi_0p3 > 0.9974
        + 0.0009418605 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874) / 0.05244257   # +0.1%  z_top30_slots > 0.9342 and ptdr0_3 > 7.408
        + 0.0009306357 * max(0.0, 0.03187688 - Q.M3) * max(0.0, 1.0 - Q.psi_0p3) / 0.000198055   # +0.1%  M3 < 0.03188 and psi_0p3 < 1
        - 0.0008445113 * max(0.0, 0.02274107 - Q.z_7) / 0.0009068898   # -0.1%  z_7 < 0.02274
        + 0.0006924118 * max(0.0, 0.009068077 - Q.tau3) / 0.0001836134   # +0.1%  tau3 < 0.009068
        + 0.0006589185 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_4 - 6.984554) / 0.03731215   # +0.1%  z_top30_slots > 0.9342 and ptdr0_4 > 6.985
        + 0.000620444 * max(0.0, 0.3628388 - Q.psi_0p1) / 0.02326895   # +0.1%  psi_0p1 < 0.3628
        - 0.0004066083 * max(0.0, Q.N2 - 0.4678622) / 0.001556028   # -0.0%  N2 > 0.4679
        + 0.0003051715 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # +0.0%  e2 > 0.06524
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 7.998;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.997778 * (0.05107193
        - 0.1438279 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -14.4%  mass < 92.86
        - 0.08903391 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -8.9%  sum_pt < 972
        - 0.08335114 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 4.657698e-07   # -8.3%  sum_pt < 972 and e4 < 5.851e-08
        - 0.07989476 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # -8.0%  sum_pt < 1261
        + 0.0741015 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15) / 0.0009227558   # +7.4%  log_sum_pt > 6.903 and girth2_top15 < 0.02147
        - 0.06182201 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15) / 1.045047   # -6.2%  sum_pt_top50 > 988.5 and girth2_top15 < 0.02147
        + 0.05646027 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +5.6%  mass_over_sum_pt < 0.09795
        + 0.05545496 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # +5.5%  sum_pt > 1017
        + 0.04120768 * max(0.0, 91.19 - Q.mass) / 16.46423   # +4.1%  mass < 91.19
        - 0.04111677 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -4.1%  log_sum_pt > 6.959
        + 0.03592368 * max(0.0, 1048.098 - Q.sum_pt_top50) / 44.36839   # +3.6%  sum_pt_top50 < 1048
        + 0.02559569 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # +2.6%  mass_top50 < 92.17
        + 0.02413983 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +2.4%  lam1 < 0.01174
        - 0.02384098 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # -2.4%  sum_pt_top50 < 1079
        - 0.01681948 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003463934   # -1.7%  log_sum_pt > 6.903 and psi_0p3 > 0.9299
        + 0.01655056 * max(0.0, Q.log_sum_pt - 6.903423) / 0.05662275   # +1.7%  log_sum_pt > 6.903
        - 0.01597703 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -1.6%  girth2_top30 < 0.006364
        + 0.01501994 * max(0.0, Q.z_top20_slots - 0.7516206) / 0.1429779   # +1.5%  z_top20_slots > 0.7516
        + 0.01449459 * max(0.0, 82.66587 - Q.mass_top30) / 15.96638   # +1.4%  mass_top30 < 82.67
        - 0.0133354 * max(0.0, 51.0 - Q.n_particles) / 9.158477   # -1.3%  n_particles < 51
        + 0.01323366 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +1.3%  sum_pt_top40 > 1070
        - 0.01274383 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # -1.3%  sum_pt > 1116
        - 0.01117437 * max(0.0, 996.8867 - Q.sum_pt_top30) / 42.94346   # -1.1%  sum_pt_top30 < 996.9
        + 0.009352547 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +0.9%  mass < 78.26
        + 0.008476019 * max(0.0, 1041.263 - Q.sum_pt_top40) / 49.16087   # +0.8%  sum_pt_top40 < 1041
        - 0.00609563 * max(0.0, Q.sum_pt_top40 - 984.7009) / 58.19535   # -0.6%  sum_pt_top40 > 984.7
        - 0.005568722 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 0.397021 - Q.max_dr) / 11.85447   # -0.6%  sum_pt < 1261 and max_dr < 0.397
        - 0.003422221 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.mass_top40 - 77.93668) / 1.109603   # -0.3%  log_sum_pt > 6.903 and mass_top40 > 77.94
        + 0.001600155 * max(0.0, Q.sum_pt_top20 - 1129.275) / 6.499168   # +0.2%  sum_pt_top20 > 1129
        + 0.0003647384 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.girth2_top30 - 0.008376291) / 1.929368e-05   # +0.0%  log_sum_pt > 7.063 and girth2_top30 > 0.008376
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 3.329;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.328595 * (-0.05100772
        - 0.1137803 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # -11.4%  girth2 < 0.009615
        - 0.08377391 * max(0.0, 80.4 - Q.mass_top40) / 12.19371   # -8.4%  mass_top40 < 80.4
        + 0.07551342 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +7.6%  mass < 92.86
        + 0.06902143 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +6.9%  girth2_top40 < 0.008841
        + 0.05725868 * max(0.0, 0.08665515 - Q.mass_over_sum_pt) / 0.01542259   # +5.7%  mass_over_sum_pt < 0.08666
        - 0.05681754 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -5.7%  girth2_top30 < 0.006364
        + 0.05513621 * max(0.0, 2.410481 - Q.D2) * max(0.0, 5.142486 - Q.D2_b2) / 2.680544   # +5.5%  D2 < 2.41 and D2_b2 < 5.142
        - 0.0496043 * max(0.0, 2.410481 - Q.D2) / 0.587301   # -5.0%  D2 < 2.41
        + 0.04777695 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741) / 0.0491674   # +4.8%  n_dr_0p2_0p4 < 8 and z_top50_slots > 0.9787
        + 0.0446852 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # +4.5%  girth2_top50 < 0.008125
        + 0.03558827 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +3.6%  n_particles < 46
        - 0.03362591 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -3.4%  tau21 < 0.3862
        + 0.03244845 * max(0.0, 0.006026828 - Q.girth2_top40) / 0.001351778   # +3.2%  girth2_top40 < 0.006027
        + 0.02564975 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +2.6%  lam2 < 0.0006155
        + 0.02536634 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.psi_0p3 - 0.9924477) / 0.007646232   # +2.5%  n_dr_0p2_0p4 < 5 and psi_0p3 > 0.9924
        - 0.01786991 * max(0.0, 79.21004 - Q.mass_top50) / 10.65754   # -1.8%  mass_top50 < 79.21
        - 0.01735509 * max(0.0, 0.019523 - Q.z_dr_0p2_0p4) / 0.007664401   # -1.7%  z_dr_0p2_0p4 < 0.01952
        - 0.01700503 * max(0.0, Q.z_top30_slots - 0.9734886) / 0.01006171   # -1.7%  z_top30_slots > 0.9735
        + 0.01617825 * max(0.0, 27.56535 - Q.sj2_mass1) / 6.309897   # +1.6%  sj2_mass1 < 27.57
        + 0.01295581 * max(0.0, 86.4 - Q.mass) / 13.67632   # +1.3%  mass < 86.4
        + 0.01286063 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732) / 1477.074   # +1.3%  n_particles < 46 and sum_pt_top30 > 800.7
        + 0.01251602 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.9313699) / 0.1032909   # +1.3%  n_dr_0p1_0p2 < 10 and psi_0p2 > 0.9314
        - 0.01086731 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5) / 0.004484684   # -1.1%  n_dr_0p2_0p4 < 5 and zdr_5 < 0.007099
        + 0.01082964 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.eccentricity - 0.7333655) / 0.1710102   # +1.1%  n_dr_0p2_0p4 < 5 and eccentricity > 0.7334
        - 0.0104891 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243) / 0.0001457263   # -1.0%  tau21 < 0.3862 and lam1 > 0.007671
        + 0.009252439 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1) / 0.1325294   # +0.9%  n_dr_0p2_0p4 < 5 and psi_0p1 < 0.9538
        + 0.008885111 * max(0.0, 0.001260456 - Q.lam1) / 8.753718e-05   # +0.9%  lam1 < 0.00126
        - 0.008510032 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -0.9%  mass < 53.87
        - 0.008466468 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0) / 3.186294   # -0.8%  n_dr_0p2_0p4 < 5 and n_dr_0p1_0p2 > 9
        + 0.006097514 * max(0.0, 0.2738063 - Q.max_dr) / 0.009268927   # +0.6%  max_dr < 0.2738
        - 0.00535576 * max(0.0, 40.2 - Q.mass) / 1.359187   # -0.5%  mass < 40.2
        + 0.004546448 * max(0.0, Q.psi_0p2 - 0.9985434) / 0.00014398   # +0.5%  psi_0p2 > 0.9985
        - 0.003912737 * max(0.0, 1.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.n_dr_0p4_up) / 0.05515294   # -0.4%  n_dr_0p2_0p4 < 1 and n_dr_0p4_up < 1
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 10.71;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.70919 * (0.05107344
        - 0.05776409 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -5.8%  mass_top40 < 163.3
        + 0.05639491 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +5.6%  mass < 101
        - 0.05509911 * max(0.0, 86.4 - Q.mass) / 13.67632   # -5.5%  mass < 86.4
        - 0.04947751 * max(0.0, 78.26182 - Q.mass) / 9.857173   # -4.9%  mass < 78.26
        + 0.04918759 * max(0.0, Q.e2_sq - 0.009606007) / 0.003182984   # +4.9%  e2_sq > 0.009606
        - 0.04717364 * max(0.0, 91.19 - Q.mass_top15) / 34.88803   # -4.7%  mass_top15 < 91.19
        - 0.04512077 * max(0.0, Q.n_particles - 22.0) / 23.9717   # -4.5%  n_particles > 22
        + 0.0419351 * max(0.0, 143.7876 - Q.mass) / 57.99337   # +4.2%  mass < 143.8
        + 0.04137637 * max(0.0, 120.6 - Q.mass) / 38.8279   # +4.1%  mass < 120.6
        + 0.03555447 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +3.6%  n_dr_0p2_0p4 < 26
        + 0.03097046 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +3.1%  n_dr_0p2_0p4 < 15
        - 0.03018014 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -3.0%  mass < 92.86
        - 0.02844258 * max(0.0, 80.78464 - Q.mass) / 10.84952   # -2.8%  mass < 80.78
        - 0.02476986 * max(0.0, Q.girth2_top15 - 0.00727763) / 0.002923446   # -2.5%  girth2_top15 > 0.007278
        - 0.02404889 * max(0.0, 91.69753 - Q.mass_top30) / 22.0127   # -2.4%  mass_top30 < 91.7
        - 0.02389885 * max(0.0, Q.lam1 - 0.008241985) / 0.002595658   # -2.4%  lam1 > 0.008242
        + 0.02193263 * max(0.0, 0.07031778 - Q.girth) / 0.01719521   # +2.2%  girth < 0.07032
        - 0.02180584 * max(0.0, 0.05660088 - Q.girth) / 0.01080382   # -2.2%  girth < 0.0566
        + 0.01872537 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.005180665 - Q.girth2_top2) / 0.07860143   # +1.9%  mass < 92.86 and girth2_top2 < 0.005181
        + 0.01645501 * max(0.0, 57.87349 - Q.mass_top15) / 12.46085   # +1.6%  mass_top15 < 57.87
        - 0.0152082 * max(0.0, 62.55 - Q.mass_top40) / 6.231505   # -1.5%  mass_top40 < 62.55
        + 0.01432788 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt) / 36.47388   # +1.4%  n_particles > 22 and soft1_pt < 2.275
        - 0.01392538 * max(0.0, 0.004855289 - Q.girth2_top15) / 0.001418414   # -1.4%  girth2_top15 < 0.004855
        + 0.01334927 * max(0.0, 101.0497 - Q.mass) * max(0.0, 3.852812 - Q.D2_b2) / 25.01551   # +1.3%  mass < 101 and D2_b2 < 3.853
        + 0.01332896 * max(0.0, 69.02716 - Q.mass_top15) / 18.23029   # +1.3%  mass_top15 < 69.03
        + 0.01104769 * max(0.0, Q.girth2_top15 - 0.01563836) / 0.001334556   # +1.1%  girth2_top15 > 0.01564
        + 0.01079622 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +1.1%  sj2_dr > 0.2232
        + 0.01067652 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +1.1%  psi_0p3 > 0.9974
        + 0.01030025 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +1.0%  mass < 74.25
        + 0.00956625 * max(0.0, 0.06030419 - Q.mass_over_sum_pt) / 0.005383371   # +1.0%  mass_over_sum_pt < 0.0603
        + 0.008934184 * max(0.0, Q.n_pt_above_10 - 11.0) / 8.944079   # +0.9%  n_pt_above_10 > 11
        + 0.008830581 * max(0.0, 0.9909875 - Q.psi_0p1) / 0.2210377   # +0.9%  psi_0p1 < 0.991
        + 0.008790843 * max(0.0, Q.e2 - 0.02793599) / 0.008664879   # +0.9%  e2 > 0.02794
        + 0.008452861 * max(0.0, 26.0 - Q.n_dr_0p1_0p2) / 14.20854   # +0.8%  n_dr_0p1_0p2 < 26
        - 0.008123659 * max(0.0, Q.sj2_dr - 0.2412757) / 0.01833576   # -0.8%  sj2_dr > 0.2413
        - 0.00793869 * max(0.0, Q.girth - 0.1207452) / 0.004285542   # -0.8%  girth > 0.1207
        - 0.006996976 * max(0.0, Q.psi_0p2 - 0.9734513) / 0.0116938   # -0.7%  psi_0p2 > 0.9735
        + 0.006889211 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +0.7%  mass_top40 < 67.73
        - 0.006322761 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.zdr_0 - 0.0008718296) / 0.0994482   # -0.6%  mass < 101 and zdr_0 > 0.0008718
        - 0.00610363 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.36624 - Q.D2_b2) / 35.56485   # -0.6%  n_dr_0p2_0p4 < 15 and D2_b2 < 7.366
        + 0.005731487 * max(0.0, Q.sum_pt - 995.6769) / 62.24161   # +0.6%  sum_pt > 995.7
        + 0.005658294 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +0.6%  lam1 > 0.01174
        - 0.005399155 * max(0.0, Q.log_sum_pt - 6.97212) / 0.02498281   # -0.5%  log_sum_pt > 6.972
        - 0.004976763 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.max_dr - 0.1939977) / 1.031549   # -0.5%  n_dr_0p2_0p4 < 15 and max_dr > 0.194
        - 0.004494727 * max(0.0, 75.26407 - Q.mass_top15) / 22.29039   # -0.4%  mass_top15 < 75.26
        + 0.004288586 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.n_dr_0_0p05 - 10.0) / 143.1065   # +0.4%  n_particles > 22 and n_dr_0_0p05 > 10
        - 0.004253697 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # -0.4%  e2 > 0.04755
        - 0.003935945 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 20.40995 - Q.D2_b2) / 0.3948355   # -0.4%  sj2_dr > 0.2232 and D2_b2 < 20.41
        + 0.003609696 * max(0.0, 0.1900123 - Q.D3) / 0.08663362   # +0.4%  D3 < 0.19
        - 0.003593919 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 3.014827 - Q.D2_b2) / 4.336836   # -0.4%  mass_top40 < 83.33 and D2_b2 < 3.015
        - 0.003539958 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # -0.4%  C2 > 0.06656
        + 0.003515815 * max(0.0, Q.sj3_pair_mass_min - 32.51366) / 5.88267   # +0.4%  sj3_pair_mass_min > 32.51
        - 0.003239555 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 13.0 - Q.n_dr_0_0p05) / 0.003836025   # -0.3%  psi_0p3 > 0.9974 and n_dr_0_0p05 < 13
        - 0.003047019 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 56.19962   # -0.3%  n_dr_0p2_0p4 < 26 and n_dr_0p1_0p2 > 11
        - 0.002582654 * max(0.0, Q.girth2_top5 - 0.007164202) / 0.002334697   # -0.3%  girth2_top5 > 0.007164
        - 0.002555604 * max(0.0, Q.girth2_top15 - 0.00727763) * max(0.0, 33.78125 - Q.pt_11) / 0.03009459   # -0.3%  girth2_top15 > 0.007278 and pt_11 < 33.78
        + 0.002412464 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, Q.zdr_0 - 0.001901263) / 0.02085507   # +0.2%  mass_top40 < 83.33 and zdr_0 > 0.001901
        - 0.002162835 * max(0.0, 80.78464 - Q.mass) * max(0.0, 3.852812 - Q.D2_b2) / 3.983646   # -0.2%  mass < 80.78 and D2_b2 < 3.853
        - 0.001996704 * max(0.0, Q.sd_rg - 0.2042612) / 0.02046744   # -0.2%  sd_rg > 0.2043
        - 0.001967019 * max(0.0, Q.lam2 - 0.001776308) / 0.0005876074   # -0.2%  lam2 > 0.001776
        - 0.0017981 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.z_top50_slots) / 0.03204382   # -0.2%  n_dr_0p2_0p4 < 15 and z_top50_slots < 1
        + 0.001595613 * max(0.0, Q.e2_sq - 0.01396296) / 0.002220692   # +0.2%  e2_sq > 0.01396
        - 0.001479695 * max(0.0, 120.6 - Q.mass) * max(0.0, 0.9943058 - Q.psi_0p3) / 0.08236936   # -0.1%  mass < 120.6 and psi_0p3 < 0.9943
        + 0.001321401 * max(0.0, Q.e3 - 0.0001841806) / 4.035159e-05   # +0.1%  e3 > 0.0001842
        + 0.0006200439 * max(0.0, Q.girth2_top5 - 0.0168592) / 0.0008909511   # +0.1%  girth2_top5 > 0.01686
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 12.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.58601 * (0.04631481
        + 0.1076951 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +10.8%  sum_pt > 907.9
        + 0.1058605 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +10.6%  sum_pt_top50 > 934.2
        - 0.08349182 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # -8.3%  sum_pt > 986.1
        - 0.06132946 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -6.1%  log_sum_pt > 6.91
        - 0.04907424 * max(0.0, 150.0144 - Q.mass_top40) / 66.86198   # -4.9%  mass_top40 < 150
        + 0.03845672 * max(0.0, Q.girth2_top30 - 0.006363916) / 0.003802703   # +3.8%  girth2_top30 > 0.006364
        + 0.03838741 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # +3.8%  sum_pt_top50 > 959.1
        + 0.03819821 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +3.8%  n_particles < 64
        - 0.03729789 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -3.7%  log_sum_pt > 6.92
        + 0.03331637 * max(0.0, Q.mass - 74.25181) / 24.07648   # +3.3%  mass > 74.25
        - 0.03185358 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # -3.2%  mass_over_sum_pt > 0.07697
        - 0.03043082 * max(0.0, Q.girth2_top30 - 0.007463985) / 0.00336109   # -3.0%  girth2_top30 > 0.007464
        - 0.02261842 * max(0.0, 787.6281 - Q.sum_pt_top3) / 328.0371   # -2.3%  sum_pt_top3 < 787.6
        - 0.02116587 * max(0.0, Q.mass - 91.19) / 15.01545   # -2.1%  mass > 91.19
        + 0.01973457 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # +2.0%  e2 < 0.03876
        - 0.01746764 * max(0.0, Q.z_top30_slots - 0.9048492) / 0.05621641   # -1.7%  z_top30_slots > 0.9048
        + 0.01292442 * max(0.0, Q.n_pt_above_1 - 28.0) / 14.80415   # +1.3%  n_pt_above_1 > 28
        - 0.01194557 * max(0.0, 0.01807679 - Q.girth2_top30) / 0.01083719   # -1.2%  girth2_top30 < 0.01808
        + 0.0112246 * max(0.0, Q.sd_mass - 69.65633) / 11.73272   # +1.1%  sd_mass > 69.66
        - 0.0109275 * max(0.0, Q.z_top50_slots - 0.9906378) / 0.006446962   # -1.1%  z_top50_slots > 0.9906
        - 0.01084907 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2) / 10.81389   # -1.1%  n_particles < 64 and D2 < 2.179
        - 0.01066865 * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 9.82015   # -1.1%  n_dr_0p1_0p2 < 21
        + 0.01060188 * max(0.0, Q.sum_pt_top30 - 933.1875) / 79.88556   # +1.1%  sum_pt_top30 > 933.2
        - 0.009786605 * max(0.0, 0.5494307 - Q.tau21) / 0.1591168   # -1.0%  tau21 < 0.5494
        + 0.00961781 * max(0.0, Q.psi_0p3 - 0.9638082) / 0.02792824   # +1.0%  psi_0p3 > 0.9638
        + 0.009567893 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +1.0%  mass_over_sum_pt > 0.1709
        - 0.009262693 * max(0.0, Q.mass - 172.8) / 0.7291439   # -0.9%  mass > 172.8
        + 0.009006609 * max(0.0, Q.log_sum_pt - 6.949481) / 0.03162473   # +0.9%  log_sum_pt > 6.949
        + 0.008996839 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +0.9%  sum_pt > 907.9 and e4 < 5.851e-08
        + 0.00830263 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # +0.8%  log_sum_pt > 6.989
        - 0.007777796 * max(0.0, 0.1937447 - Q.z_dr_0p2_0p4) / 0.146372   # -0.8%  z_dr_0p2_0p4 < 0.1937
        + 0.007074175 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +0.7%  n_dr_0p2_0p4 < 11
        - 0.006583244 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -0.7%  mass_over_sum_pt_sq > 0.0292
        + 0.006331495 * max(0.0, Q.mass - 172.4888) / 0.7446233   # +0.6%  mass > 172.5
        - 0.00631732 * max(0.0, Q.sd_mass - 86.4) / 6.275027   # -0.6%  sd_mass > 86.4
        + 0.006009415 * max(0.0, 0.01375115 - Q.z_11) / 0.0009450582   # +0.6%  z_11 < 0.01375
        - 0.005743829 * max(0.0, Q.mass_top40 - 120.6) / 5.823744   # -0.6%  mass_top40 > 120.6
        - 0.005641159 * max(0.0, Q.girth2_top50 - 0.01951641) / 0.001208668   # -0.6%  girth2_top50 > 0.01952
        + 0.005584718 * max(0.0, 1.976207 - Q.D2) / 0.367701   # +0.6%  D2 < 1.976
        + 0.005244012 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2) / 0.001543279   # +0.5%  psi_0p3 > 0.9974 and D2 < 3.345
        - 0.004448282 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -0.4%  max_dr > 0.2405
        - 0.004386111 * max(0.0, 0.01541561 - Q.e2) / 0.001436351   # -0.4%  e2 < 0.01542
        + 0.004163001 * max(0.0, 0.03787151 - Q.M3) / 0.01370579   # +0.4%  M3 < 0.03787
        - 0.004060295 * max(0.0, 14.14062 - Q.pt_11) / 0.9740783   # -0.4%  pt_11 < 14.14
        - 0.003839817 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -0.4%  sum_pt_top40 > 1025
        - 0.003463541 * max(0.0, Q.sum_pt_top40 - 994.2695) / 51.72055   # -0.3%  sum_pt_top40 > 994.3
        - 0.003402427 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -0.3%  mass_top50 > 157.5
        + 0.0033156 * max(0.0, 64.48544 - Q.mass) / 5.935866   # +0.3%  mass < 64.49
        + 0.002971141 * max(0.0, 23.31647 - Q.sj2_mass1) / 3.96589   # +0.3%  sj2_mass1 < 23.32
        - 0.002784567 * max(0.0, Q.z_top20_slots - 0.9102775) / 0.02689878   # -0.3%  z_top20_slots > 0.9103
        + 0.002670409 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # +0.3%  n_dr_0p1_0p2 < 8
        - 0.002494287 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.1958008 - Q.absphi_13) / 0.007588689   # -0.2%  log_sum_pt > 6.91 and absphi_13 < 0.1958
        + 0.002426303 * max(0.0, Q.z_dr_0_0p05 - 0.9084912) / 0.009227347   # +0.2%  z_dr_0_0p05 > 0.9085
        - 0.002234213 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # -0.2%  sum_pt_top40 > 1070
        - 0.001936565 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.2535773 - Q.dr_11) / 0.00936088   # -0.2%  log_sum_pt > 6.91 and dr_11 < 0.2536
        - 0.001927796 * max(0.0, Q.sum_pt_top40 - 994.2695) * max(0.0, 0.3241858 - Q.dr1_12) / 12.50457   # -0.2%  sum_pt_top40 > 994.3 and dr1_12 < 0.3242
        - 0.001624782 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.2%  sum_pt_top10 > 943.7
        - 0.001474019 * max(0.0, 0.6635952 - Q.psi_0p1) / 0.06403963   # -0.1%  psi_0p1 < 0.6636
        + 0.001446371 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # +0.1%  mass_top10 > 71.78
        + 0.001410998 * max(0.0, Q.girth2 - 0.004756928) / 0.005348122   # +0.1%  girth2 > 0.004757
        - 0.001150925 * max(0.0, Q.n_dr_0p2_0p4 - 21.0) / 0.6469597   # -0.1%  n_dr_0p2_0p4 > 21
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 17.87;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.87297 * (-0.004348233
        + 0.1437221 * max(0.0, 172.8 - Q.mass_top50) / 85.49388   # +14.4%  mass_top50 < 172.8
        - 0.09376804 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -9.4%  mass < 101
        - 0.07739072 * max(0.0, 168.9698 - Q.mass_top50) / 81.82258   # -7.7%  mass_top50 < 169
        + 0.07334363 * max(0.0, Q.mass_over_sum_pt - 0.02682209) / 0.06022589   # +7.3%  mass_over_sum_pt > 0.02682
        - 0.07265135 * max(0.0, 120.6 - Q.mass) / 38.8279   # -7.3%  mass < 120.6
        + 0.06323085 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +6.3%  mass < 92.86
        - 0.04990884 * max(0.0, 0.01807679 - Q.girth2_top30) / 0.01083719   # -5.0%  girth2_top30 < 0.01808
        - 0.03601085 * max(0.0, Q.mass_over_sum_pt - 0.09795415) / 0.01222897   # -3.6%  mass_over_sum_pt > 0.09795
        + 0.02908806 * max(0.0, 0.06524004 - Q.e2) / 0.03476195   # +2.9%  e2 < 0.06524
        + 0.02582799 * max(0.0, 86.4 - Q.mass) / 13.67632   # +2.6%  mass < 86.4
        - 0.02266592 * max(0.0, Q.girth - 0.02085222) / 0.04959222   # -2.3%  girth > 0.02085
        + 0.02181911 * max(0.0, 0.003687605 - Q.lam2) / 0.002601874   # +2.2%  lam2 < 0.003688
        + 0.02099696 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +2.1%  lam1 < 0.007259
        - 0.02071375 * max(0.0, 0.02048524 - Q.lam1) / 0.01309746   # -2.1%  lam1 < 0.02049
        + 0.02040176 * max(0.0, Q.tau1 - 0.05444509) / 0.03990678   # +2.0%  tau1 > 0.05445
        - 0.01740789 * max(0.0, 1129.275 - Q.sum_pt_top20) / 207.3581   # -1.7%  sum_pt_top20 < 1129
        + 0.01290305 * max(0.0, Q.girth2_top40 - 0.008031986) / 0.003375597   # +1.3%  girth2_top40 > 0.008032
        + 0.01206584 * max(0.0, 91.19 - Q.mass_top15) / 34.88803   # +1.2%  mass_top15 < 91.19
        + 0.01165921 * max(0.0, Q.girth2_top20 - 0.008031209) / 0.002906852   # +1.2%  girth2_top20 > 0.008031
        + 0.01161926 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # +1.2%  mass_top50 < 71.8
        - 0.01060076 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -1.1%  mass_over_sum_pt > 0.1182
        - 0.01034449 * max(0.0, Q.e2 - 0.02793599) / 0.008664879   # -1.0%  e2 > 0.02794
        + 0.01006874 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # +1.0%  sum_pt < 1261
        + 0.009010695 * max(0.0, Q.girth2_top30 - 0.008376291) / 0.003092781   # +0.9%  girth2_top30 > 0.008376
        - 0.008708352 * max(0.0, 0.004573744 - Q.girth2_top50) / 0.0007739713   # -0.9%  girth2_top50 < 0.004574
        + 0.007138693 * max(0.0, 0.003811746 - Q.lam1) / 0.0006796118   # +0.7%  lam1 < 0.003812
        - 0.007132262 * max(0.0, Q.mass_over_sum_pt - 0.02682209) * max(0.0, 0.7108211 - Q.z_dr_0p05_0p1) / 0.02513932   # -0.7%  mass_over_sum_pt > 0.02682 and z_dr_0p05_0p1 < 0.7108
        - 0.006849002 * max(0.0, 0.1011005 - Q.M2) / 0.03683604   # -0.7%  M2 < 0.1011
        - 0.005978397 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2) / 0.04240098   # -0.6%  mass_over_sum_pt > 0.1182 and D2_b2 < 7.366
        + 0.005720008 * max(0.0, 0.00375223 - Q.girth2_top30) / 0.0006714094   # +0.6%  girth2_top30 < 0.003752
        - 0.005454565 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # -0.5%  log_sum_pt < 7.017
        - 0.005208884 * max(0.0, 0.08329945 - Q.mass_over_sum_pt) / 0.0135093   # -0.5%  mass_over_sum_pt < 0.0833
        + 0.004968035 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +0.5%  tau1 < 0.06311
        + 0.004947241 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +0.5%  LHA > 0.372
        - 0.004842459 * max(0.0, 0.003418057 - Q.girth2_top50) / 0.0004594629   # -0.5%  girth2_top50 < 0.003418
        - 0.004638964 * max(0.0, Q.sj3_pair_mass_min - 29.00832) / 6.839984   # -0.5%  sj3_pair_mass_min > 29.01
        - 0.004620323 * max(0.0, 0.001592178 - Q.girth2_top3) / 0.000513961   # -0.5%  girth2_top3 < 0.001592
        + 0.004310794 * max(0.0, 0.03700182 - Q.z_dr_0p2_0p4) / 0.0183096   # +0.4%  z_dr_0p2_0p4 < 0.037
        + 0.003997365 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt) / 683.4113   # +0.4%  mass < 120.6 and sum_pt < 1008
        + 0.003491793 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +0.3%  z_dr_0p1_0p2 < 0.1203
        - 0.003472313 * max(0.0, 91.19 - Q.mass_top15) * max(0.0, 0.04008677 - Q.C2_b2) / 0.9324124   # -0.3%  mass_top15 < 91.19 and C2_b2 < 0.04009
        + 0.003364584 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.3%  e2 > 0.04755
        - 0.003324741 * max(0.0, Q.girth2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529) / 8.566262e-05   # -0.3%  girth2_top20 > 0.008031 and C2_b2 > 0.0008188
        + 0.002667284 * max(0.0, Q.e2 - 0.02793599) * max(0.0, 0.4357228 - Q.max_dr) / 0.0007248268   # +0.3%  e2 > 0.02794 and max_dr < 0.4357
        - 0.002389811 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, Q.dr_max_012 - 0.004415714) / 1.536542e-05   # -0.2%  e3 < 0.0003372 and dr_max_012 > 0.004416
        + 0.001940116 * max(0.0, 0.04704395 - Q.C2) / 0.005827731   # +0.2%  C2 < 0.04704
        - 0.001926401 * max(0.0, 1.409617 - Q.D2) / 0.1519846   # -0.2%  D2 < 1.41
        + 0.001891556 * max(0.0, 0.002197765 - Q.girth2_top15) / 0.0004509993   # +0.2%  girth2_top15 < 0.002198
        + 0.001744188 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832) / 8.191828e-05   # +0.2%  mass_over_sum_pt > 0.1182 and C2_b2 > 0.02705
        - 0.001475274 * max(0.0, Q.e2 - 0.02793599) * max(0.0, 0.01718455 - Q.zdr_1) / 5.467259e-05   # -0.1%  e2 > 0.02794 and zdr_1 < 0.01718
        - 0.001374158 * max(0.0, 30.35994 - Q.mass_top15) / 3.169072   # -0.1%  mass_top15 < 30.36
        - 0.001306278 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722) / 0.001787472   # -0.1%  girth2_top30 > 0.008376 and D2_b2 > 1.677
        - 0.001157636 * max(0.0, Q.sj3_mass1 - 21.11128) / 1.638992   # -0.1%  sj3_mass1 > 21.11
        - 0.001127949 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.1%  log_sum_pt < 6.811
        - 0.00108732 * max(0.0, Q.sj3_dr_min - 0.1204829) / 0.02211875   # -0.1%  sj3_dr_min > 0.1205
        - 0.000934947 * max(0.0, Q.girth2_top30 - 0.02809026) / 0.0001993235   # -0.1%  girth2_top30 > 0.02809
        + 0.0007213255 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707) / 2.369749e-05   # +0.1%  e2 > 0.05557 and zdr_0 > 0.00137
        + 0.0007189818 * max(0.0, 0.9300465 - Q.z_top40_slots) / 0.002595351   # +0.1%  z_top40_slots < 0.93
        + 0.0006573103 * max(0.0, Q.sj3_pair_mass_min - 29.00832) * max(0.0, Q.psi_0p3 - 0.9896594) / 0.01664863   # +0.1%  sj3_pair_mass_min > 29.01 and psi_0p3 > 0.9897
        - 0.0005231722 * max(0.0, Q.mass_over_sum_pt - 0.09795415) * max(0.0, 0.001160626 - Q.soft8_z) / 8.555108e-07   # -0.1%  mass_over_sum_pt > 0.09795 and soft8_z < 0.001161
        - 0.000488954 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.orientation_deg - 26.6454) / 0.03454067   # -0.0%  girth2_top30 > 0.008376 and orientation_deg > 26.65
        + 0.0004786626 * max(0.0, Q.e2 - 0.05557149) * max(0.0, 1.67722 - Q.D2_b2) / 0.000547635   # +0.0%  e2 > 0.05557 and D2_b2 < 1.677
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 43.05;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 43.05188 * (-0.01928119
        - 0.11776 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.3075308   # -11.8%  mass < 91.19 and psi_0p3 > 0.9777
        + 0.1169556 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5337743   # +11.7%  mass < 91.19 and psi_0p3 > 0.9638
        - 0.0887329 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5700006   # -8.9%  mass < 92.86 and psi_0p3 > 0.9638
        + 0.06599173 * max(0.0, 120.6 - Q.mass) / 38.8279   # +6.6%  mass < 120.6
        + 0.05598757 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.4385899   # +5.6%  mass < 101 and psi_0p3 > 0.9777
        - 0.05355248 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -5.4%  mass < 101
        - 0.04233245 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -4.2%  mass < 82.85
        - 0.03694611 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -3.7%  tau1 < 0.1073
        - 0.03068 * max(0.0, 0.008190222 - Q.girth2) / 0.002454223   # -3.1%  girth2 < 0.00819
        + 0.03052204 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # +3.1%  girth2 < 0.009615
        + 0.02743915 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +2.7%  mass_over_sum_pt < 0.1182
        + 0.02702053 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.2231625   # +2.7%  mass < 82.85 and psi_0p3 > 0.9777
        - 0.02542136 * max(0.0, 86.4 - Q.sd_mass) / 35.84633   # -2.5%  sd_mass < 86.4
        + 0.02308366 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.01745475   # +2.3%  mass < 101 and psi_0p3 > 0.998
        + 0.02001886 * max(0.0, 69.65633 - Q.sd_mass) / 24.56034   # +2.0%  sd_mass < 69.66
        - 0.01771959 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.01143143   # -1.8%  mass < 91.19 and psi_0p3 > 0.998
        + 0.01630822 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +1.6%  mass < 92.86
        - 0.01517984 * max(0.0, 91.19 - Q.mass) / 16.46423   # -1.5%  mass < 91.19
        - 0.01510891 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -1.5%  lam1 < 0.008242
        - 0.01457579 * max(0.0, Q.sd_mass - 98.05743) / 4.276978   # -1.5%  sd_mass > 98.06
        - 0.01337132 * max(0.0, 78.26182 - Q.mass) / 9.857173   # -1.3%  mass < 78.26
        - 0.01324259 * max(0.0, 0.1881908 - Q.sd_rg) / 0.07278694   # -1.3%  sd_rg < 0.1882
        + 0.01204473 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +1.2%  tau21_b2 < 0.2352
        + 0.01195345 * max(0.0, 0.09591084 - Q.tau1) / 0.02629125   # +1.2%  tau1 < 0.09591
        + 0.009608837 * max(0.0, 0.1596365 - Q.sd_rg) / 0.05492863   # +1.0%  sd_rg < 0.1596
        - 0.008191503 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.8%  psi_0p3 > 0.998
        - 0.008154684 * max(0.0, 0.00363788 - Q.e2_sq) / 0.0004963098   # -0.8%  e2_sq < 0.003638
        + 0.006776107 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # +0.7%  girth2 < 0.007877
        + 0.006168752 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # +0.6%  lam1 < 0.00619
        + 0.00537749 * max(0.0, 9.0 - Q.n_dr_0p2_0p4) / 3.177318   # +0.5%  n_dr_0p2_0p4 < 9
        + 0.005352433 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +0.5%  tau1 < 0.07709
        + 0.005166894 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +0.5%  lam1 < 0.01174
        - 0.00422517 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # -0.4%  psi_0p2 > 0.9314
        - 0.004117423 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt) / 11.66143   # -0.4%  tau21_b2 < 0.2352 and sum_pt < 1261
        + 0.00361707 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.z_top50_slots - 0.978741) / 1.779326e-05   # +0.4%  psi_0p3 > 0.9974 and z_top50_slots > 0.9787
        + 0.003234914 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.3%  psi_0p3 > 0.9974
        - 0.003128221 * max(0.0, 18.0 - Q.n_dr_0_0p05) / 7.289224   # -0.3%  n_dr_0_0p05 < 18
        - 0.002921068 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003362293   # -0.3%  tau21_b2 < 0.2352 and psi_0p3 > 0.9299
        - 0.002794368 * max(0.0, 0.006403325 - Q.girth2) / 0.001406912   # -0.3%  girth2 < 0.006403
        + 0.002562913 * max(0.0, 0.2639816 - Q.sd_rg) / 0.1327654   # +0.3%  sd_rg < 0.264
        + 0.002524466 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # +0.3%  tau1 < 0.08787
        + 0.002352455 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.2%  n_dr_0p2_0p4 < 6
        + 0.002159837 * max(0.0, Q.mass_top20 - 85.79457) / 7.06893   # +0.2%  mass_top20 > 85.79
        - 0.002091813 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3) / 6.520272e-05   # -0.2%  n_dr_0p2_0p4 < 6 and e3 < 7.876e-05
        - 0.002013918 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -0.2%  tau21 < 0.3862
        + 0.001922499 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.02947847   # +0.2%  mass < 82.85 and psi_0p3 < 0.9985
        - 0.001782615 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.4947602 - Q.z_dr_0_0p05) / 0.572942   # -0.2%  mass < 92.86 and z_dr_0_0p05 < 0.4948
        + 0.001769921 * max(0.0, 0.006403325 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.9980008) / 9.11248e-07   # +0.2%  girth2 < 0.006403 and psi_0p3 > 0.998
        + 0.001671879 * max(0.0, Q.mass_top20 - 73.35236) / 11.28399   # +0.2%  mass_top20 > 73.35
        - 0.00119243 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 5.106465e-05   # -0.1%  tau21_b2 < 0.2352 and mass_over_sum_pt_sq < 0.007873
        + 0.001079326 * max(0.0, 0.09573228 - Q.z_dr_0_0p05) / 0.02128126   # +0.1%  z_dr_0_0p05 < 0.09573
        - 0.001057732 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.001696484   # -0.1%  tau21_b2 < 0.2352 and sj2_dr > 0.1938
        + 0.001023664 * max(0.0, 0.2982 - Q.max_dr) / 0.01350087   # +0.1%  max_dr < 0.2982
        + 0.0008892289 * max(0.0, 0.000404306 - Q.lam2) / 5.964336e-05   # +0.1%  lam2 < 0.0004043
        - 0.0007727219 * max(0.0, Q.psi_0p3 - 0.9995915) / 8.716828e-05   # -0.1%  psi_0p3 > 0.9996
        - 0.0007399412 * max(0.0, Q.sd_mass - 45.595) / 24.70213   # -0.1%  sd_mass > 45.59
        - 0.0005413826 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -9.840088) / 1.453315   # -0.1%  tau21_b2 < 0.2352 and orientation_deg > -9.84
        - 0.000474534 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.1063277 - Q.z_2) / 0.0009585458   # -0.0%  tau21_b2 < 0.2352 and z_2 < 0.1063
        - 0.0003990365 * max(0.0, 0.008241985 - Q.lam1) * max(0.0, Q.zdr_0 - 0.01564747) / 7.458275e-07   # -0.0%  lam1 < 0.008242 and zdr_0 > 0.01565
        - 0.0001938179 * max(0.0, 0.1939977 - Q.max_dr) / 0.001716714   # -0.0%  max_dr < 0.194
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 14.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.66993 * (0.1279942
        - 0.09638333 * max(0.0, 0.009614971 - Q.width) / 0.003502545   # -9.6%  width < 0.009615
        + 0.05287809 * max(0.0, Q.mass - 74.25181) / 24.07648   # +5.3%  mass > 74.25
        + 0.05172686 * max(0.0, Q.mass - 64.48544) / 31.19164   # +5.2%  mass > 64.49
        + 0.04543432 * max(0.0, Q.mass - 87.36377) / 16.57741   # +4.5%  mass > 87.36
        - 0.04274131 * max(0.0, Q.mass - 101.0497) / 12.31084   # -4.3%  mass > 101
        - 0.04097375 * max(0.0, Q.z_top50_slots - 0.9704436) / 0.02331682   # -4.1%  z_top50_slots > 0.9704
        - 0.03948036 * max(0.0, Q.n_for_90pct - 7.0) / 13.93571   # -3.9%  n_for_90pct > 7
        + 0.0393146 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt) / 2.336698   # +3.9%  mass_over_sum_pt > 0.07697 and sum_pt < 1116
        + 0.03710554 * max(0.0, 1028.184 - Q.sum_pt) / 26.95739   # +3.7%  sum_pt < 1028
        - 0.0367466 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -3.7%  mass_top50 > 82.04
        - 0.03529263 * max(0.0, 39.0 - Q.n_for_90pct) / 18.29776   # -3.5%  n_for_90pct < 39
        + 0.03084818 * max(0.0, 0.007463985 - Q.girth2_top30) / 0.00233708   # +3.1%  girth2_top30 < 0.007464
        + 0.03005968 * max(0.0, Q.girth2_top20 - 0.008031209) / 0.002906852   # +3.0%  girth2_top20 > 0.008031
        - 0.02986821 * max(0.0, 0.02146578 - Q.girth2_top15) / 0.01470441   # -3.0%  girth2_top15 < 0.02147
        - 0.02609304 * max(0.0, Q.girth2_top20 - 0.006043209) / 0.003593885   # -2.6%  girth2_top20 > 0.006043
        - 0.025757 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # -2.6%  mass_over_sum_pt > 0.07697
        - 0.02557138 * max(0.0, 6.930088 - Q.log_sum_pt) / 0.02522479   # -2.6%  log_sum_pt < 6.93
        - 0.02539524 * max(0.0, Q.mass - 125.1) / 7.098976   # -2.5%  mass > 125.1
        + 0.0245236 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # +2.5%  girth2 < 0.007877
        - 0.02424768 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt) / 0.0005328922   # -2.4%  girth2_top40 > 0.005197 and log_sum_pt < 7.017
        + 0.02366421 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +2.4%  lam1 < 0.007671
        + 0.02180436 * max(0.0, Q.girth2_top40 - 0.005196966) / 0.004725809   # +2.2%  girth2_top40 > 0.005197
        + 0.01808587 * max(0.0, Q.mass_top50 - 117.0487) / 7.705579   # +1.8%  mass_top50 > 117
        + 0.0163489 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +1.6%  sum_pt < 1017
        - 0.01251485 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -1.3%  n_dr_0p2_0p4 < 15
        - 0.01210577 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # -1.2%  sum_pt_top40 < 1025
        - 0.01129682 * max(0.0, Q.e3 - 5.13841e-05) / 6.821576e-05   # -1.1%  e3 > 5.138e-05
        + 0.01023221 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # +1.0%  tau2 < 0.0795
        - 0.009468062 * max(0.0, Q.sum_pt_top50 - 889.8503) / 149.655   # -0.9%  sum_pt_top50 > 889.9
        + 0.009343752 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +0.9%  girth2_top40 < 0.008841
        - 0.009096494 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -0.9%  mass_over_sum_pt > 0.1182
        - 0.009068353 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # -0.9%  lam1 < 0.01174
        + 0.008163589 * max(0.0, Q.n_dr_0p2_0p4 - 3.0) / 6.482203   # +0.8%  n_dr_0p2_0p4 > 3
        + 0.00743565 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +0.7%  n_particles > 38
        + 0.00736603 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.6773544   # +0.7%  sum_pt < 1028 and dr_max_012 > 0.1828
        - 0.006868278 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.5693287   # -0.7%  sum_pt < 1017 and dr_max_012 > 0.1828
        + 0.006129732 * max(0.0, Q.sum_pt_top30 - 978.0762) / 49.23409   # +0.6%  sum_pt_top30 > 978.1
        + 0.00598725 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # +0.6%  C2 > 0.06656
        - 0.005372211 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -0.5%  psi_0p3 > 0.9943
        + 0.005001715 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.5%  e2 > 0.04755
        + 0.003694521 * max(0.0, 0.8316924 - Q.z_top15_slots) / 0.04702674   # +0.4%  z_top15_slots < 0.8317
        + 0.003497208 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.3%  sj2_dr > 0.2232
        + 0.002278504 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.03720487   # +0.2%  z_dr_0p1_0p2 > 0.334
        + 0.002196207 * max(0.0, 1001.523 - Q.sum_pt_top40) / 26.72886   # +0.2%  sum_pt_top40 < 1002
        + 0.00204542 * max(0.0, Q.n_dr_0p1_0p2 - 19.0) / 1.732133   # +0.2%  n_dr_0p1_0p2 > 19
        - 0.001901938 * max(0.0, Q.mass - 64.48544) * max(0.0, Q.zdr_0 - 0.0008718296) / 0.4297651   # -0.2%  mass > 64.49 and zdr_0 > 0.0008718
        + 0.001742407 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # +0.2%  e3 > 0.0003372
        - 0.001575871 * max(0.0, Q.mass_top20 - 119.2969) / 1.8402   # -0.2%  mass_top20 > 119.3
        - 0.001532679 * max(0.0, 0.3628388 - Q.psi_0p1) / 0.02326895   # -0.2%  psi_0p1 < 0.3628
        - 0.001297216 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, Q.sj2_dr - 0.1512157) / 6.78729e-05   # -0.1%  girth2_top30 < 0.007464 and sj2_dr > 0.1512
        - 0.000820388 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # -0.1%  log_sum_pt > 7.139
        + 0.0006626517 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # +0.1%  e2 > 0.06524
        - 0.0004871437 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, 13.0 - Q.n_for_90pct) / 15.66746   # -0.0%  sum_pt < 1017 and n_for_90pct < 13
        + 0.0004723425 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.09686657   # +0.0%  n_dr_0p2_0p4 < 15 and z_dr_0p1_0p2 > 0.334
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 44.58;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 44.57839 * (-0.04033253
        - 0.2536628 * max(0.0, 160.8 - Q.mass) / 72.78344   # -25.4%  mass < 160.8
        + 0.2224685 * max(0.0, 162.8363 - Q.mass) / 74.60496   # +22.2%  mass < 162.8
        + 0.17639 * max(0.0, 172.8 - Q.mass) / 83.78792   # +17.6%  mass < 172.8
        - 0.1402996 * max(0.0, 143.7876 - Q.mass) / 57.99337   # -14.0%  mass < 143.8
        + 0.0544108 * max(0.0, 138.8977 - Q.mass_top50) / 55.01559   # +5.4%  mass_top50 < 138.9
        - 0.02572354 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -2.6%  mass_top40 < 163.3
        + 0.02520192 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +2.5%  mass < 92.86
        - 0.01676968 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # -1.7%  mass_top50 < 92.17
        + 0.01536285 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +1.5%  mass < 82.85
        + 0.005515097 * max(0.0, Q.girth - 0.09749958) / 0.008315822   # +0.6%  girth > 0.0975
        - 0.005340556 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -0.5%  log_sum_pt < 6.903
        - 0.005200428 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -0.5%  mass < 64.49
        + 0.005145533 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +0.5%  sum_pt < 986.1
        - 0.004833966 * max(0.0, Q.z_top40_slots - 0.9574183) / 0.02831517   # -0.5%  z_top40_slots > 0.9574
        - 0.004805407 * max(0.0, Q.girth2_top30 - 0.0008564881) / 0.007661497   # -0.5%  girth2_top30 > 0.0008565
        + 0.003776524 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # +0.4%  girth2_top40 < 0.00626
        + 0.003262984 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +0.3%  mass_top40 < 80.89
        - 0.003173394 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -0.3%  tau1 < 0.05445
        - 0.002913818 * max(0.0, Q.girth - 0.1402186) / 0.001871547   # -0.3%  girth > 0.1402
        + 0.002845325 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.3%  lam1 < 0.004673
        + 0.002683862 * max(0.0, 956.2133 - Q.sum_pt_top40) / 14.00451   # +0.3%  sum_pt_top40 < 956.2
        + 0.002558672 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +0.3%  lam1 > 0.01174
        + 0.002511961 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +0.3%  LHA > 0.4042
        + 0.002432677 * max(0.0, 73.33139 - Q.mass_top30) / 11.37701   # +0.2%  mass_top30 < 73.33
        + 0.002348496 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # +0.2%  psi_0p3 > 0.9943
        - 0.001967682 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt) / 447.38   # -0.2%  mass_top40 < 80.89 and sum_pt < 1035
        - 0.00120237 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.z_top50_slots - 0.9995789) / 0.005945365   # -0.1%  mass < 92.86 and z_top50_slots > 0.9996
        - 0.001077479 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -0.1%  z_dr_0_0p05 > 0.8789
        - 0.001017985 * max(0.0, 0.00217213 - Q.girth2_top40) / 0.0002075259   # -0.1%  girth2_top40 < 0.002172
        + 0.0006896655 * max(0.0, 125.1 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 42.49581   # +0.1%  mass_top40 < 125.1 and n_dr_0p2_0p4 > 9
        - 0.0006758145 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # -0.1%  mass_over_sum_pt > 0.1709
        - 0.0006481324 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258) / 5.834847   # -0.1%  sum_pt_top40 < 956.2 and soft4_pt > 1.789
        + 0.00057059 * max(0.0, Q.D2 - 2.178951) / 1.450771   # +0.1%  D2 > 2.179
        + 0.0005421358 * max(0.0, Q.z_top5 - 0.7963975) / 0.006402524   # +0.1%  z_top5 > 0.7964
        + 0.000517464 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40) / 104.3982   # +0.1%  mass_top40 < 80.89 and sum_pt_top40 < 906.6
        + 0.0005084318 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # +0.1%  e3 > 0.0003372
        - 0.0003028699 * max(0.0, Q.n_for_90pct - 35.0) / 0.4575429   # -0.0%  n_for_90pct > 35
        - 0.0002607702 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, 0.001776308 - Q.lam2) / 0.005060516   # -0.0%  sum_pt_top40 < 956.2 and lam2 < 0.001776
        + 0.0001497131 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047) / 1.704178   # +0.0%  sum_pt_top40 < 956.2 and soft3_pt > 2.873
        - 0.0001277259 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.psi_0p1 - 0.9184255) / 0.111142   # -0.0%  sum_pt_top40 < 956.2 and psi_0p1 > 0.9184
        + 0.0001027861 * max(0.0, 986.0565 - Q.sum_pt) * max(0.0, Q.soft5_z - 0.002036949) / 0.003199714   # +0.0%  sum_pt < 986.1 and soft5_z > 0.002037
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 17.95;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.94866 * (0.05153058
        - 0.1316864 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -13.2%  girth < 0.1207
        + 0.1025765 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # +10.3%  tau1 < 0.1507
        + 0.0777495 * max(0.0, Q.mass - 78.26182) / 21.33658   # +7.8%  mass > 78.26
        - 0.07709877 * max(0.0, Q.mass - 64.48544) / 31.19164   # -7.7%  mass > 64.49
        + 0.04919446 * max(0.0, 0.01655983 - Q.girth2_top20) / 0.01003486   # +4.9%  girth2_top20 < 0.01656
        - 0.04784271 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -4.8%  mass < 101
        + 0.04382608 * max(0.0, Q.e2 - 0.01256572) / 0.01912291   # +4.4%  e2 > 0.01257
        + 0.04062145 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +4.1%  mass < 89.74
        - 0.03079414 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -3.1%  e2 < 0.04359
        - 0.02656996 * max(0.0, 65.20727 - Q.sj2_mass1) / 36.50254   # -2.7%  sj2_mass1 < 65.21
        + 0.02449059 * max(0.0, 86.4 - Q.mass) / 13.67632   # +2.4%  mass < 86.4
        - 0.02147721 * max(0.0, Q.mass - 143.7876) / 3.946979   # -2.1%  mass > 143.8
        - 0.01785684 * max(0.0, Q.mass_top30 - 78.53034) / 14.31024   # -1.8%  mass_top30 > 78.53
        - 0.01621657 * max(0.0, 0.01976735 - Q.girth2_top10) / 0.01371247   # -1.6%  girth2_top10 < 0.01977
        + 0.01580155 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +1.6%  mass_top50 > 136.8
        + 0.01464986 * max(0.0, 0.04828819 - Q.tau2) / 0.02103241   # +1.5%  tau2 < 0.04829
        + 0.01450444 * max(0.0, 0.0001086251 - Q.e3) / 5.924919e-05   # +1.5%  e3 < 0.0001086
        + 0.01316761 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.3%  z_dr_0p2_0p4 < 0.09123
        - 0.01172616 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -1.2%  n_dr_0p2_0p4 < 11
        + 0.01092123 * max(0.0, 935.1043 - Q.sum_pt_top15) / 95.709   # +1.1%  sum_pt_top15 < 935.1
        - 0.01048824 * max(0.0, 0.04466492 - Q.tau1) / 0.005217805   # -1.0%  tau1 < 0.04466
        - 0.01007046 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, 0.4357228 - Q.max_dr) / 0.001259784   # -1.0%  girth2_top10 < 0.01977 and max_dr < 0.4357
        - 0.009780977 * max(0.0, 72.18744 - Q.mass_top15) / 20.20734   # -1.0%  mass_top15 < 72.19
        + 0.009524144 * max(0.0, 59.40777 - Q.mass_top5) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1) / 22.61156   # +1.0%  mass_top5 < 59.41 and z_dr_0p05_0p1 < 0.8509
        + 0.009049994 * max(0.0, Q.mass_top5 - 22.18342) / 12.67741   # +0.9%  mass_top5 > 22.18
        - 0.008947709 * max(0.0, Q.mass - 162.8363) / 1.509832   # -0.9%  mass > 162.8
        - 0.008425956 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.8%  psi_0p3 > 0.998
        + 0.008395434 * max(0.0, Q.mass_top30 - 89.17293) / 10.17054   # +0.8%  mass_top30 > 89.17
        - 0.00832345 * max(0.0, Q.z_dr_0_0p05 - 0.7674734) / 0.05221088   # -0.8%  z_dr_0_0p05 > 0.7675
        - 0.007840686 * max(0.0, 0.005312783 - Q.girth2_top20) / 0.001437214   # -0.8%  girth2_top20 < 0.005313
        - 0.007310318 * max(0.0, Q.pt_dispersion - 0.27462) / 0.06481184   # -0.7%  pt_dispersion > 0.2746
        + 0.007118246 * max(0.0, 0.008824206 - Q.zdr_1) / 0.00383206   # +0.7%  zdr_1 < 0.008824
        - 0.006341316 * max(0.0, 59.40777 - Q.mass_top5) / 33.74882   # -0.6%  mass_top5 < 59.41
        - 0.006264241 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # -0.6%  girth2_top30 < 0.005402
        + 0.006157631 * max(0.0, 0.5760704 - Q.z_top2_slots) / 0.2302819   # +0.6%  z_top2_slots < 0.5761
        + 0.006034243 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +0.6%  dr_0 < 0.06413
        + 0.005606036 * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0004356739   # +0.6%  psi_0p3 > 0.9985
        + 0.00557495 * max(0.0, 0.02078982 - Q.zdr_0) / 0.01170192   # +0.6%  zdr_0 < 0.02079
        + 0.005252145 * max(0.0, Q.n_particles - 41.0) / 8.920024   # +0.5%  n_particles > 41
        + 0.005026378 * max(0.0, 2.975532 - Q.D2) / 0.9290697   # +0.5%  D2 < 2.976
        - 0.004923073 * max(0.0, Q.mass_over_sum_pt - 0.1606361) / 0.001344467   # -0.5%  mass_over_sum_pt > 0.1606
        + 0.00473513 * max(0.0, 0.007043525 - Q.zdr_2) / 0.003147169   # +0.5%  zdr_2 < 0.007044
        - 0.003890654 * max(0.0, 0.006756161 - Q.girth2_top3) / 0.003650633   # -0.4%  girth2_top3 < 0.006756
        + 0.00327204 * max(0.0, 0.01541561 - Q.e2) / 0.001436351   # +0.3%  e2 < 0.01542
        - 0.003127402 * max(0.0, Q.mass_top50 - 136.785) * max(0.0, Q.soft5_z - 0.001594761) / 0.001639485   # -0.3%  mass_top50 > 136.8 and soft5_z > 0.001595
        - 0.002941793 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # -0.3%  sum_pt_top50 < 959.1
        - 0.002898005 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.9985421) / 6.34706e-06   # -0.3%  girth2_top10 < 0.01977 and psi_0p3 > 0.9985
        + 0.002724213 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897) / 0.001013644   # +0.3%  mass > 162.8 and soft5_z > 0.001435
        + 0.00238033 * max(0.0, Q.mass_top50 - 160.8) / 1.246461   # +0.2%  mass_top50 > 160.8
        - 0.00225604 * max(0.0, Q.e2 - 0.03263075) / 0.006288927   # -0.2%  e2 > 0.03263
        - 0.002169799 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.2%  sum_pt_top10 > 943.7
        + 0.002021523 * max(0.0, Q.girth2_top40 - 0.02497133) / 0.0004755075   # +0.2%  girth2_top40 > 0.02497
        + 0.001896247 * max(0.0, 65.20727 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 7.769597) / 168.6958   # +0.2%  sj2_mass1 < 65.21 and sj2_mass2 > 7.77
        + 0.001880077 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, Q.psi_0p3 - 0.9985421) / 4.280363e-07   # +0.2%  girth2_top30 < 0.005402 and psi_0p3 > 0.9985
        + 0.001744594 * max(0.0, 2.975532 - Q.D2) * max(0.0, 0.08728027 - Q.absphi_4) / 0.03660338   # +0.2%  D2 < 2.976 and absphi_4 < 0.08728
        - 0.001601833 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # -0.2%  mass_top10 > 71.78
        + 0.001593582 * max(0.0, Q.sj3_dr_min - 0.1204829) / 0.02211875   # +0.2%  sj3_dr_min > 0.1205
        - 0.001427382 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # -0.1%  sum_pt < 986.1
        + 0.001346586 * max(0.0, Q.mass - 78.26182) * max(0.0, Q.psi_0p3 - 0.9853273) / 0.1102552   # +0.1%  mass > 78.26 and psi_0p3 > 0.9853
        - 0.001238856 * max(0.0, 2.975532 - Q.D2) * max(0.0, Q.sj2_dr - 0.2070855) / 0.0198522   # -0.1%  D2 < 2.976 and sj2_dr > 0.2071
        - 0.001114762 * max(0.0, 0.002575211 - Q.girth2) / 0.0002582872   # -0.1%  girth2 < 0.002575
        + 0.00103834 * max(0.0, Q.pt_dispersion - 0.27462) * max(0.0, 30.0 - Q.n_real_top30) / 0.1960387   # +0.1%  pt_dispersion > 0.2746 and n_real_top30 < 30
        + 0.0009587397 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, 631.275 - Q.sum_pt_top5) / 0.06578377   # +0.1%  girth2_top30 < 0.005402 and sum_pt_top5 < 631.3
        + 0.0009342812 * max(0.0, Q.mass_over_sum_pt - 0.1606361) * max(0.0, 0.09696199 - Q.z_3) / 5.287414e-05   # +0.1%  mass_over_sum_pt > 0.1606 and z_3 < 0.09696
        + 0.0008569874 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.pt1_dr01 - 5.351077) / 32.20836   # +0.1%  mass < 101 and pt1_dr01 > 5.351
        - 0.0008146428 * max(0.0, 986.0565 - Q.sum_pt) * max(0.0, Q.dr_3 - 0.01436663) / 0.7268239   # -0.1%  sum_pt < 986.1 and dr_3 > 0.01437
        + 0.0007754784 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, 0.2384186 - Q.dr_9) / 1.423778   # +0.1%  sum_pt_top50 < 959.1 and dr_9 < 0.2384
        + 0.0007549135 * max(0.0, Q.sum_pt_top10 - 943.6922) * max(0.0, 0.6469679 - Q.z_dr_0p05_0p1) / 6.713086   # +0.1%  sum_pt_top10 > 943.7 and z_dr_0p05_0p1 < 0.647
        - 0.0007149738 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109) / 0.0004735247   # -0.1%  mass_top50 > 160.8 and soft4_z > 0.001721
        + 0.0007148648 * max(0.0, 0.2404747 - Q.max_dr) / 0.005136976   # +0.1%  max_dr < 0.2405
        - 0.0005240609 * max(0.0, Q.mass_top50 - 172.8) / 0.5062389   # -0.1%  mass_top50 > 172.8
        + 0.0003955463 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, 0.2277069 - Q.dr1_5) / 1.438963   # +0.0%  sum_pt_top50 < 959.1 and dr1_5 < 0.2277
        - 2.859975e-05 * max(0.0, Q.sum_pt_top10 - 943.6922) * max(0.0, -0.07794189 - Q.eta_0) / 0.002015064   # -0.0%  sum_pt_top10 > 943.7 and eta_0 < -0.07794
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 9.932;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.931833 * (-0.04419154
        + 0.1303533 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +13.0%  mass < 92.86
        + 0.08129995 * max(0.0, 0.08589404 - Q.girth) / 0.02745918   # +8.1%  girth < 0.08589
        - 0.07074146 * max(0.0, 0.076787 - Q.girth) / 0.02105267   # -7.1%  girth < 0.07679
        - 0.06352787 * max(0.0, 80.35535 - Q.mass_top50) / 11.1399   # -6.4%  mass_top50 < 80.36
        + 0.06182892 * max(0.0, 97.93004 - Q.mass_top50) / 22.03533   # +6.2%  mass_top50 < 97.93
        + 0.03726343 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +3.7%  e2_sq < 0.008184
        - 0.03410878 * max(0.0, 0.00625621 - Q.girth2_top10) / 0.002474789   # -3.4%  girth2_top10 < 0.006256
        - 0.03207605 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # -3.2%  e2 < 0.03876
        + 0.02881549 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 43.23581   # +2.9%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21
        - 0.02608862 * max(0.0, 80.4 - Q.mass) / 10.6806   # -2.6%  mass < 80.4
        - 0.02597247 * max(0.0, 86.4 - Q.mass_top30) / 18.30352   # -2.6%  mass_top30 < 86.4
        - 0.02584668 * max(0.0, 0.006929741 - Q.girth2_top30) / 0.002004687   # -2.6%  girth2_top30 < 0.00693
        + 0.02350239 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +2.4%  e2 < 0.02516
        - 0.02195524 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -2.2%  mass_top40 < 83.33
        - 0.02091963 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -2.1%  lam1 < 0.006717
        + 0.02088306 * max(0.0, 0.00406126 - Q.girth2_top10) / 0.001298509   # +2.1%  girth2_top10 < 0.004061
        + 0.02081334 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +2.1%  e2 < 0.0303
        + 0.01942109 * max(0.0, 80.35535 - Q.mass_top50) * max(0.0, 0.003932029 - Q.zdr_4) / 0.03420305   # +1.9%  mass_top50 < 80.36 and zdr_4 < 0.003932
        - 0.01875966 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.004495205 - Q.zdr_4) / 0.05697234   # -1.9%  mass < 92.86 and zdr_4 < 0.004495
        - 0.01775246 * max(0.0, 0.07999061 - Q.mass_over_sum_pt) / 0.01176803   # -1.8%  mass_over_sum_pt < 0.07999
        + 0.01770024 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # +1.8%  girth2_top30 < 0.005402
        + 0.01586959 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # +1.6%  psi_0p2 > 0.9314
        - 0.0154941 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2) / 0.005195774   # -1.5%  n_dr_0p2_0p4 < 10 and girth2 < 0.005532
        + 0.01425867 * max(0.0, Q.D2 - 2.178951) / 1.450771   # +1.4%  D2 > 2.179
        + 0.01296041 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +1.3%  mass_top30 < 60.44
        + 0.01133529 * max(0.0, 50.0 - Q.n_real_top50) / 8.569617   # +1.1%  n_real_top50 < 50
        + 0.008993512 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +0.9%  mass_top40 < 67.73
        - 0.00856619 * max(0.0, 0.008031209 - Q.girth2_top20) / 0.00309849   # -0.9%  girth2_top20 < 0.008031
        - 0.008462161 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -0.8%  mass < 64.49
        - 0.008225036 * max(0.0, Q.D2 - 3.814159) / 0.870654   # -0.8%  D2 > 3.814
        + 0.007233011 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # +0.7%  girth < 0.07374
        + 0.007014284 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.z_9 - 0.01620892) / 0.2014824   # +0.7%  mass < 101 and z_9 > 0.01621
        + 0.00694538 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.7%  n_dr_0p2_0p4 < 6
        + 0.00687896 * max(0.0, 15.0 - Q.n_dr_0p1_0p2) / 5.139044   # +0.7%  n_dr_0p1_0p2 < 15
        - 0.006866004 * max(0.0, Q.z_top5_slots - 0.534626) / 0.08390497   # -0.7%  z_top5_slots > 0.5346
        - 0.006748819 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_real_top40 - 26.0) / 33.17834   # -0.7%  n_dr_0p2_0p4 < 10 and n_real_top40 > 26
        - 0.006269585 * max(0.0, 3.793233e-05 - Q.e3) / 1.004564e-05   # -0.6%  e3 < 3.793e-05
        - 0.005614402 * max(0.0, 0.06030419 - Q.mass_over_sum_pt) / 0.005383371   # -0.6%  mass_over_sum_pt < 0.0603
        + 0.004931616 * max(0.0, 0.00130722 - Q.girth2_top10) / 0.0002794102   # +0.5%  girth2_top10 < 0.001307
        - 0.004683412 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.5%  n_dr_0p1_0p2 < 8
        - 0.004578125 * max(0.0, 101.0497 - Q.mass) * max(0.0, 891.875 - Q.sum_pt_top10) / 2090.674   # -0.5%  mass < 101 and sum_pt_top10 < 891.9
        - 0.004495844 * max(0.0, 80.4 - Q.mass) * max(0.0, Q.z_9 - 0.01833434) / 0.06571201   # -0.4%  mass < 80.4 and z_9 > 0.01833
        + 0.004116327 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) * max(0.0, Q.z_dr_0p1_0p2 - 0.1203437) / 6.825269e-05   # +0.4%  mass_over_sum_pt_sq < 0.009595 and z_dr_0p1_0p2 > 0.1203
        + 0.004108125 * max(0.0, Q.psi_0p2 - 0.9935324) / 0.001465572   # +0.4%  psi_0p2 > 0.9935
        - 0.003540687 * max(0.0, 0.005911134 - Q.zdr_0) / 0.001247991   # -0.4%  zdr_0 < 0.005911
        + 0.003357944 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # +0.3%  psi_0p3 > 0.998
        + 0.002878288 * max(0.0, Q.z_top40_slots - 0.996191) * max(0.0, 0.006580753 - Q.C2_b2) / 4.241011e-06   # +0.3%  z_top40_slots > 0.9962 and C2_b2 < 0.006581
        - 0.002066561 * max(0.0, 1.916992 - Q.soft10_pt) * max(0.0, Q.sj3_mass1 - 5.112677) / 3.344471   # -0.2%  soft10_pt < 1.917 and sj3_mass1 > 5.113
        - 0.001333936 * max(0.0, 19.89956 - Q.sj2_mass1) / 2.423382   # -0.1%  sj2_mass1 < 19.9
        + 0.001332703 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.pt1_dr01 - 12.6865) / 16.11867   # +0.1%  mass < 101 and pt1_dr01 > 12.69
        - 0.001210841 * max(0.0, Q.D2 - 5.378975) / 0.5670303   # -0.1%  D2 > 5.379
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 2.915;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.914976 * (-0.003287068
        + 0.3196223 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +32.0%  mass < 82.85
        + 0.12963 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.04417088   # +13.0%  mass < 86.4 and lam2 < 0.003688
        + 0.09219414 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +9.2%  mass < 74.25
        - 0.08899522 * max(0.0, 62.55 - Q.mass) / 5.464058   # -8.9%  mass < 62.55
        - 0.06512012 * max(0.0, 0.06895248 - Q.mass_over_sum_pt) / 0.007729512   # -6.5%  mass_over_sum_pt < 0.06895
        - 0.06432053 * max(0.0, 74.78616 - Q.mass_top40) / 9.958349   # -6.4%  mass_top40 < 74.79
        - 0.05324491 * max(0.0, 0.006142802 - Q.girth2_top15) / 0.002080651   # -5.3%  girth2_top15 < 0.006143
        - 0.04010457 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.5993597   # -4.0%  mass < 86.4 and z_dr_0p1_0p2 < 0.06473
        - 0.03448383 * max(0.0, 86.4 - Q.mass) / 13.67632   # -3.4%  mass < 86.4
        - 0.01866122 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.03580539   # -1.9%  mass < 86.4 and psi_0p3 < 0.9985
        - 0.01733287 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, 334.5 - Q.pt_0) / 1668.723   # -1.7%  mass_top40 < 89.68 and pt_0 < 334.5
        + 0.01253197 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, Q.n_dr_0p05_0p1 - 1.0) / 124.4066   # +1.3%  mass_top40 < 89.68 and n_dr_0p05_0p1 > 1
        + 0.01072953 * max(0.0, 0.0007431905 - Q.girth2_top10) / 0.0001243965   # +1.1%  girth2_top10 < 0.0007432
        + 0.008760657 * max(0.0, Q.sd_mass - 125.1) / 1.231011   # +0.9%  sd_mass > 125.1
        - 0.008564294 * max(0.0, Q.sj3_pair_mass_max - 120.6) / 2.130243   # -0.9%  sj3_pair_mass_max > 120.6
        - 0.007839807 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -0.8%  mass < 53.87
        + 0.005979308 * max(0.0, 0.0007431905 - Q.girth2_top10) * max(0.0, 1069.671 - Q.sum_pt_top40) / 0.006568426   # +0.6%  girth2_top10 < 0.0007432 and sum_pt_top40 < 1070
        + 0.005814842 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min) / 54.09437   # +0.6%  sj3_pair_mass_max > 120.6 and sj3_pair_mass_min < 76.6
        - 0.00492048 * max(0.0, 0.0007431905 - Q.girth2_top10) * max(0.0, 36.0 - Q.n_real_top40) / 0.0006803313   # -0.5%  girth2_top10 < 0.0007432 and n_real_top40 < 36
        - 0.003751529 * max(0.0, Q.sd_mass - 125.1) * max(0.0, 0.4199841 - Q.sd_zg) / 0.1704664   # -0.4%  sd_mass > 125.1 and sd_zg < 0.42
        + 0.003739079 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926) / 0.5295314   # +0.4%  sj3_pair_mass_max > 120.6 and z_dr_0p1_0p2 > 0.2865
        - 0.003658796 * max(0.0, 6.856375 - Q.log_sum_pt) / 0.008053459   # -0.4%  log_sum_pt < 6.856
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 9.647;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.646797 * (0.2256971
        - 0.1699135 * max(0.0, 0.02580396 - Q.mass_over_sum_pt_sq) / 0.01697662   # -17.0%  mass_over_sum_pt_sq < 0.0258
        + 0.06295064 * Q.n_particles / 45.81523   # +6.3%  n_particles
        + 0.06216014 * max(0.0, Q.mass - 74.25181) / 24.07648   # +6.2%  mass > 74.25
        + 0.04562881 * max(0.0, Q.mass_top50 - 27.46086) / 60.61506   # +4.6%  mass_top50 > 27.46
        - 0.04248238 * max(0.0, Q.mass - 160.8) / 1.724663   # -4.2%  mass > 160.8
        - 0.03948384 * max(0.0, Q.mass - 101.0497) / 12.31084   # -3.9%  mass > 101
        - 0.03699634 * max(0.0, 1115.723 - Q.sum_pt) / 92.38491   # -3.7%  sum_pt < 1116
        + 0.03584998 * max(0.0, 0.02550569 - Q.girth2_top50) / 0.01683664   # +3.6%  girth2_top50 < 0.02551
        + 0.03543312 * max(0.0, 0.02550569 - Q.girth2_top50) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.0005189214   # +3.5%  girth2_top50 < 0.02551 and psi_0p3 > 0.9638
        - 0.03482593 * max(0.0, 6.910131 - Q.log_sum_pt) / 0.017275   # -3.5%  log_sum_pt < 6.91
        - 0.03446308 * max(0.0, Q.mass - 143.7876) / 3.946979   # -3.4%  mass > 143.8
        + 0.02772468 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +2.8%  sum_pt_top40 < 1053
        - 0.02461142 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -2.5%  sum_pt < 1085
        + 0.02340823 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +2.3%  mass_top50 > 136.8
        + 0.02098277 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +2.1%  mass_over_sum_pt > 0.1709
        - 0.01965547 * max(0.0, Q.mass - 74.25181) * max(0.0, 1028.184 - Q.sum_pt) / 733.8204   # -2.0%  mass > 74.25 and sum_pt < 1028
        + 0.01863817 * max(0.0, 0.08286256 - Q.tau1) / 0.0188894   # +1.9%  tau1 < 0.08286
        - 0.018267 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -1.8%  mass_over_sum_pt_sq > 0.0292
        + 0.01700881 * max(0.0, Q.mass - 162.8363) / 1.509832   # +1.7%  mass > 162.8
        - 0.01475138 * max(0.0, Q.mass - 136.785) / 5.043396   # -1.5%  mass > 136.8
        + 0.01446346 * max(0.0, 0.007820315 - Q.girth2_top50) / 0.002264874   # +1.4%  girth2_top50 < 0.00782
        + 0.01363422 * max(0.0, 997.0189 - Q.sum_pt_top50) / 17.90852   # +1.4%  sum_pt_top50 < 997
        - 0.01354911 * max(0.0, 966.0633 - Q.sum_pt_top30) / 29.99923   # -1.4%  sum_pt_top30 < 966.1
        + 0.0134333 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # +1.3%  sum_pt_top40 < 1007
        - 0.01072575 * max(0.0, Q.mass - 136.785) * max(0.0, Q.D2 - 0.8942376) / 6.48176   # -1.1%  mass > 136.8 and D2 > 0.8942
        - 0.009835843 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -1.0%  log_sum_pt < 6.811
        - 0.008790711 * max(0.0, Q.n_pt_above_1 - 34.0) / 10.44526   # -0.9%  n_pt_above_1 > 34
        - 0.008023995 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.8310045 - Q.tau21_b2) / 33.42141   # -0.8%  sum_pt < 1085 and tau21_b2 < 0.831
        + 0.007595782 * max(0.0, Q.mass - 160.8) * max(0.0, Q.D2 - 0.8942376) / 2.121004   # +0.8%  mass > 160.8 and D2 > 0.8942
        - 0.007497132 * max(0.0, Q.mass_top50 - 136.785) * max(0.0, Q.soft6_z - 0.0008188601) / 0.003727182   # -0.7%  mass_top50 > 136.8 and soft6_z > 0.0008189
        + 0.006938409 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft6_z - 0.0005748372) / 0.002073713   # +0.7%  mass > 162.8 and soft6_z > 0.0005748
        - 0.006685595 * max(0.0, 0.08871546 - Q.z_dr_0p1_0p2) / 0.03070029   # -0.7%  z_dr_0p1_0p2 < 0.08872
        + 0.005921038 * max(0.0, Q.mass - 136.785) * max(0.0, 1028.184 - Q.sum_pt) / 132.4986   # +0.6%  mass > 136.8 and sum_pt < 1028
        - 0.005752271 * max(0.0, 16.0 - Q.n_for_90pct) / 1.776607   # -0.6%  n_for_90pct < 16
        + 0.005660958 * max(0.0, Q.mass - 172.8) / 0.7291439   # +0.6%  mass > 172.8
        - 0.005468906 * max(0.0, 1034.834 - Q.sum_pt) * max(0.0, 2.975532 - Q.D2) / 24.6298   # -0.5%  sum_pt < 1035 and D2 < 2.976
        + 0.004823498 * max(0.0, Q.mass_top5 - 33.71058) / 7.394508   # +0.5%  mass_top5 > 33.71
        - 0.004819997 * max(0.0, Q.mass_top20 - 91.19) / 5.893841   # -0.5%  mass_top20 > 91.19
        - 0.004777432 * max(0.0, 2.752054 - Q.pt_entropy) / 0.1846134   # -0.5%  pt_entropy < 2.752
        + 0.004486355 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 858.8262 - Q.sum_pt_top40) / 1219.241   # +0.4%  sum_pt < 1085 and sum_pt_top40 < 858.8
        - 0.004310047 * max(0.0, Q.sum_pt_top20 - 956.5062) / 37.46564   # -0.4%  sum_pt_top20 > 956.5
        - 0.00420222 * max(0.0, 76.9886 - Q.mass_top10) / 32.58955   # -0.4%  mass_top10 < 76.99
        + 0.003886994 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.1289397 - Q.dr_4) / 4.4378   # +0.4%  sum_pt < 1085 and dr_4 < 0.1289
        - 0.003547199 * max(0.0, 18.32812 - Q.pt_11) / 2.095384   # -0.4%  pt_11 < 18.33
        + 0.00299989 * max(0.0, 16.0 - Q.n_for_90pct) * max(0.0, 9.676985 - Q.D2) / 8.10578   # +0.3%  n_for_90pct < 16 and D2 < 9.677
        + 0.002859318 * max(0.0, Q.sum_pt_top30 - 1111.245) / 12.62573   # +0.3%  sum_pt_top30 > 1111
        + 0.002831186 * max(0.0, Q.mass_top30 - 138.3818) / 1.758216   # +0.3%  mass_top30 > 138.4
        + 0.002763761 * max(0.0, 16.0 - Q.n_for_90pct) * max(0.0, 3.814159 - Q.D2) / 1.817419   # +0.3%  n_for_90pct < 16 and D2 < 3.814
        + 0.002680698 * max(0.0, Q.psi_0p1 - 0.9371031) / 0.01088804   # +0.3%  psi_0p1 > 0.9371
        - 0.002656333 * max(0.0, Q.mass_top50 - 92.16545) * max(0.0, 0.001923089 - Q.soft7_z) / 0.006887627   # -0.3%  mass_top50 > 92.17 and soft7_z < 0.001923
        + 0.002523477 * max(0.0, Q.n_pt_above_1 - 58.0) / 0.9616151   # +0.3%  n_pt_above_1 > 58
        + 0.002167395 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +0.2%  lam1 > 0.01174
        - 0.002045552 * max(0.0, Q.tau1 - 0.1751567) / 0.002322569   # -0.2%  tau1 > 0.1752
        - 0.001970368 * max(0.0, Q.mass - 172.8) * max(0.0, Q.D2 - 0.8942376) / 0.9099067   # -0.2%  mass > 172.8 and D2 > 0.8942
        - 0.001865391 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # -0.2%  log_sum_pt > 7.139
        - 0.001281468 * max(0.0, Q.mass - 172.8) * max(0.0, Q.soft5_z - 0.0004140594) / 0.001251279   # -0.1%  mass > 172.8 and soft5_z > 0.0004141
        + 0.001241073 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.1312677 - Q.dr_13) / 0.0002562188   # +0.1%  log_sum_pt < 6.811 and dr_13 < 0.1313
        + 0.001227367 * max(0.0, Q.mass_top50 - 168.9698) / 0.6651775   # +0.1%  mass_top50 > 169
        - 0.001149334 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.005041702 - Q.zdr_11) / 1.572361e-05   # -0.1%  log_sum_pt < 6.811 and zdr_11 < 0.005042
        + 0.001108107 * max(0.0, Q.mass - 172.8) * max(0.0, 56.53125 - Q.pt_6) / 9.768822   # +0.1%  mass > 172.8 and pt_6 < 56.53
        + 0.001011594 * max(0.0, Q.mass - 172.8) * max(0.0, Q.soft6_z - 0.0004464147) / 0.001308464   # +0.1%  mass > 172.8 and soft6_z > 0.0004464
        - 0.0008556556 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.00317537 - Q.C3) / 7.466262e-06   # -0.1%  log_sum_pt > 7.139 and C3 < 0.003175
        + 0.0006961515 * max(0.0, 1007.788 - Q.sum_pt) * max(0.0, Q.sum_pt_top3 - 512.4375) / 295.3022   # +0.1%  sum_pt < 1008 and sum_pt_top3 > 512.4
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 25.04;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.04269 * (-0.02082033
        + 0.09340214 * max(0.0, 0.140939 - Q.mass_over_sum_pt) / 0.05796036   # +9.3%  mass_over_sum_pt < 0.1409
        + 0.05620367 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +5.6%  mass_over_sum_pt < 0.1182
        + 0.04883737 * max(0.0, 125.1 - Q.mass) / 42.45775   # +4.9%  mass < 125.1
        - 0.04516427 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) / 0.0178873   # -4.5%  mass_over_sum_pt < 0.09047
        - 0.04232954 * max(0.0, 91.19 - Q.mass) / 16.46423   # -4.2%  mass < 91.19
        - 0.03943492 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -3.9%  mass < 82.85
        - 0.03901754 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -3.9%  girth2_top30 < 0.01216
        - 0.03824019 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -3.8%  lam1 < 0.008242
        + 0.03737499 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +3.7%  mass_over_sum_pt < 0.09795
        + 0.0339105 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +3.4%  lam1 < 0.007671
        - 0.02989985 * max(0.0, 97.93004 - Q.mass_top50) / 22.03533   # -3.0%  mass_top50 < 97.93
        - 0.02901948 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # -2.9%  sd_rg < 0.3017
        + 0.02732133 * max(0.0, 80.4 - Q.mass) / 10.6806   # +2.7%  mass < 80.4
        - 0.02698005 * max(0.0, 89.74183 - Q.mass) / 15.55633   # -2.7%  mass < 89.74
        - 0.0248636 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p2 - 0.9087063) / 0.00142529   # -2.5%  mass_over_sum_pt < 0.09047 and psi_0p2 > 0.9087
        - 0.02312975 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -2.3%  tau1 < 0.1073
        - 0.02094935 * max(0.0, 0.01897915 - Q.girth2_top40) / 0.01130444   # -2.1%  girth2_top40 < 0.01898
        + 0.01996696 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +2.0%  tau21_b2 < 0.3425
        - 0.01813419 * max(0.0, 117.0487 - Q.mass_top50) / 36.94196   # -1.8%  mass_top50 < 117
        + 0.01809822 * max(0.0, 77.37641 - Q.mass_top50) / 9.980855   # +1.8%  mass_top50 < 77.38
        + 0.01750771 * max(0.0, 85.8667 - Q.mass_top50) / 13.95835   # +1.8%  mass_top50 < 85.87
        - 0.01652607 * max(0.0, 78.26182 - Q.mass) / 9.857173   # -1.7%  mass < 78.26
        + 0.01492294 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # +1.5%  e2 < 0.04359
        + 0.01392947 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +1.4%  psi_0p3 > 0.9924
        - 0.01262291 * max(0.0, 0.076787 - Q.girth) / 0.02105267   # -1.3%  girth < 0.07679
        - 0.01179833 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # -1.2%  tau2 < 0.0795
        - 0.01110768 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.02048524 - Q.lam1) / 0.001304679   # -1.1%  tau21_b2 < 0.3425 and lam1 < 0.02049
        + 0.01052478 * max(0.0, 82.66587 - Q.mass_top30) / 15.96638   # +1.1%  mass_top30 < 82.67
        + 0.01028373 * max(0.0, 0.1778185 - Q.sd_rg) / 0.06576479   # +1.0%  sd_rg < 0.1778
        + 0.01003064 * max(0.0, 89.6788 - Q.mass_top40) / 17.47899   # +1.0%  mass_top40 < 89.68
        + 0.01000006 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # +1.0%  girth2_top5 < 0.007164
        - 0.009860811 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -1.0%  girth2_top20 < 0.01083
        + 0.008827235 * max(0.0, 6.941997 - Q.log_sum_pt) / 0.03180972   # +0.9%  log_sum_pt < 6.942
        + 0.008801961 * max(0.0, 0.08873143 - Q.mass_over_sum_pt) / 0.01671255   # +0.9%  mass_over_sum_pt < 0.08873
        + 0.007966179 * max(0.0, 69.65633 - Q.sd_mass) / 24.56034   # +0.8%  sd_mass < 69.66
        - 0.007863361 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # -0.8%  log_sum_pt < 6.989
        - 0.006804704 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # -0.7%  mass_top50 < 71.8
        + 0.006653468 * max(0.0, 0.076787 - Q.girth) * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.0007421723   # +0.7%  girth < 0.07679 and z_dr_0p2_0p4 < 0.0518
        - 0.006521249 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # -0.7%  n_dr_0p2_0p4 < 18
        - 0.006151729 * max(0.0, 3.080078 - Q.soft9_pt) / 1.143104   # -0.6%  soft9_pt < 3.08
        - 0.005520723 * max(0.0, 0.0001086251 - Q.e3) / 5.924919e-05   # -0.6%  e3 < 0.0001086
        + 0.005281168 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +0.5%  e2 < 0.0303
        + 0.005207701 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 3.445312 - Q.soft9_pt) / 12.48859   # +0.5%  n_dr_0p2_0p4 < 18 and soft9_pt < 3.445
        + 0.004961844 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # +0.5%  sum_pt_top40 < 1025
        - 0.004760908 * max(0.0, 0.001595561 - Q.soft7_z) / 0.0003533115   # -0.5%  soft7_z < 0.001596
        - 0.004584783 * max(0.0, 976.277 - Q.sum_pt_top50) / 12.87298   # -0.5%  sum_pt_top50 < 976.3
        + 0.004440363 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # +0.4%  girth2_top20 < 0.006374
        - 0.00401396 * max(0.0, 0.9734886 - Q.z_top30_slots) / 0.03219848   # -0.4%  z_top30_slots < 0.9735
        + 0.003649839 * max(0.0, 1.650391 - Q.soft7_pt) / 0.3647391   # +0.4%  soft7_pt < 1.65
        - 0.003506277 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1881908 - Q.sd_rg) / 0.0002726472   # -0.4%  psi_0p3 > 0.9924 and sd_rg < 0.1882
        - 0.003374778 * max(0.0, 0.4226723 - Q.N2) / 0.1194948   # -0.3%  N2 < 0.4227
        + 0.003239688 * max(0.0, 37.45803 - Q.sj2_mass1) / 13.07509   # +0.3%  sj2_mass1 < 37.46
        + 0.002789828 * max(0.0, 0.07852883 - Q.mass_over_sum_pt) / 0.01107516   # +0.3%  mass_over_sum_pt < 0.07853
        - 0.002694583 * max(0.0, 0.007164202 - Q.girth2_top5) * max(0.0, Q.n_pt_above_10 - 13.0) / 0.02038471   # -0.3%  girth2_top5 < 0.007164 and n_pt_above_10 > 13
        - 0.002599245 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # -0.3%  lam1 < 0.007259
        - 0.002557127 * max(0.0, 0.002582316 - Q.girth2_top30) / 0.0003518773   # -0.3%  girth2_top30 < 0.002582
        - 0.002326714 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # -0.2%  tau1 < 0.06311
        - 0.001709691 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.007709916 - Q.girth2_top40) / 0.0001062984   # -0.2%  tau21_b2 < 0.3425 and girth2_top40 < 0.00771
        - 0.001607508 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, Q.n_for_50pct - 3.0) / 0.1980908   # -0.2%  tau21_b2 < 0.3425 and n_for_50pct > 3
        + 0.00158816 * max(0.0, Q.psi_0p2 - 0.9804031) * max(0.0, Q.pt2_over_pt0 - 0.1852611) / 0.002296628   # +0.2%  psi_0p2 > 0.9804 and pt2_over_pt0 > 0.1853
        + 0.0014936 * max(0.0, Q.n_pt_above_1 - 58.0) / 0.9616151   # +0.1%  n_pt_above_1 > 58
        - 0.001466767 * max(0.0, 74.25181 - Q.mass) / 8.587064   # -0.1%  mass < 74.25
        + 0.001429034 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03787151 - Q.M3) / 0.1118325   # +0.1%  n_dr_0p2_0p4 < 18 and M3 < 0.03787
        - 0.001266536 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2404747) / 0.01288721   # -0.1%  N2 < 0.4227 and max_dr > 0.2405
        + 0.0008172229 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.006118006 - Q.girth2_top50) / 2.65509e-05   # +0.1%  tau21_b2 < 0.3425 and girth2_top50 < 0.006118
        - 0.0007235268 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, Q.sd_rg - 0.2280025) / 1.846945e-05   # -0.1%  psi_0p3 > 0.9924 and sd_rg > 0.228
        - 0.0007143352 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 32.0 - Q.n_real_top40) / 0.1595466   # -0.1%  tau21_b2 < 0.3425 and n_real_top40 < 32
        - 0.0006911684 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.n_dr_0p05_0p1 - 14.0) / 15.57557   # -0.1%  mass < 91.19 and n_dr_0p05_0p1 > 14
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 9.258;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.258341 * (0.04196497
        - 0.09277263 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # -9.3%  sd_rg < 0.3017
        + 0.07114102 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +7.1%  log_sum_pt < 7.017
        + 0.05112618 * max(0.0, 0.08589404 - Q.girth) / 0.02745918   # +5.1%  girth < 0.08589
        + 0.05098942 * max(0.0, 0.2042612 - Q.sd_rg) / 0.08448097   # +5.1%  sd_rg < 0.2043
        + 0.04897965 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +4.9%  girth2_top5 < 0.00833
        + 0.04136986 * max(0.0, 79.18312 - Q.sd_mass) / 30.46368   # +4.1%  sd_mass < 79.18
        - 0.03727016 * max(0.0, 0.2941033 - Q.LHA) / 0.05718399   # -3.7%  LHA < 0.2941
        + 0.03517188 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # +3.5%  girth2_top10 < 0.007679
        - 0.03485038 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # -3.5%  e2 > 0.0303
        - 0.03315142 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # -3.3%  lam1 < 0.01174
        + 0.02995494 * max(0.0, Q.mass_top30 - 80.24626) / 13.49216   # +3.0%  mass_top30 > 80.25
        - 0.02331109 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.3%  tau1 < 0.07057
        + 0.02293668 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +2.3%  lam2 < 0.001776
        - 0.02126271 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -2.1%  psi_0p3 > 0.9897
        - 0.02030763 * max(0.0, 0.008956554 - Q.girth2_top10) / 0.004473389   # -2.0%  girth2_top10 < 0.008957
        - 0.01962327 * max(0.0, 1017.778 - Q.sum_pt_top20) / 107.0228   # -2.0%  sum_pt_top20 < 1018
        - 0.01843149 * max(0.0, 45.595 - Q.sd_mass) / 13.46843   # -1.8%  sd_mass < 45.59
        - 0.01802545 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # -1.8%  mass_top50 < 71.8
        + 0.01797857 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +1.8%  z_dr_0p1_0p2 < 0.1203
        - 0.0172396 * max(0.0, Q.mass_top50 - 97.93004) / 11.91764   # -1.7%  mass_top50 > 97.93
        + 0.0159019 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +1.6%  lam1 < 0.004673
        - 0.01516139 * max(0.0, Q.sj2_dr - 0.1666442) / 0.04983259   # -1.5%  sj2_dr > 0.1666
        - 0.01487256 * max(0.0, 0.006399858 - Q.e2_sq) / 0.001405963   # -1.5%  e2_sq < 0.0064
        + 0.01415929 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) / 3.82104   # +1.4%  n_dr_0p1_0p2 < 13
        - 0.01386347 * max(0.0, 0.4226723 - Q.N2) / 0.1194948   # -1.4%  N2 < 0.4227
        + 0.01384317 * max(0.0, 42.0 - Q.n_pt_above_5) / 14.5829   # +1.4%  n_pt_above_5 < 42
        - 0.0137563 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -1.4%  sum_pt < 1002
        + 0.01343253 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +1.3%  sj2_dr > 0.2232
        - 0.01304604 * max(0.0, 0.002270363 - Q.girth2_top5) / 0.0007634099   # -1.3%  girth2_top5 < 0.00227
        + 0.01208201 * max(0.0, 89.17293 - Q.mass_top30) / 20.1761   # +1.2%  mass_top30 < 89.17
        - 0.01126745 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.n_real_top40 - 22.0) / 0.06106188   # -1.1%  girth2_top5 < 0.00833 and n_real_top40 > 22
        - 0.011037 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -1.1%  psi_0p1 > 0.8976
        - 0.01101247 * max(0.0, 1018.698 - Q.sum_pt_top40) / 34.89083   # -1.1%  sum_pt_top40 < 1019
        - 0.009982489 * max(0.0, 0.001056655 - Q.girth2_top2) / 0.0003002514   # -1.0%  girth2_top2 < 0.001057
        - 0.009297316 * max(0.0, 1.976207 - Q.D2) / 0.367701   # -0.9%  D2 < 1.976
        + 0.007738681 * max(0.0, 52.15154 - Q.mass_top50) / 3.332261   # +0.8%  mass_top50 < 52.15
        - 0.007386168 * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 3.233647   # -0.7%  n_dr_0p2_0p4 > 9
        + 0.00708923 * max(0.0, 0.0005124533 - Q.girth2_top2) / 0.0001104529   # +0.7%  girth2_top2 < 0.0005125
        + 0.006257426 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) / 0.3666529   # +0.6%  z_dr_0_0p05 < 0.8459
        - 0.005329401 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677) / 0.04548726   # -0.5%  girth2_top5 < 0.00833 and sj3_mass1 > 5.113
        - 0.00517101 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.02970886 - Q.absphi_0) / 0.06277088   # -0.5%  n_dr_0p1_0p2 < 13 and absphi_0 < 0.02971
        + 0.005002489 * max(0.0, 1.976207 - Q.D2) * max(0.0, 0.2535773 - Q.dr_11) / 0.05913718   # +0.5%  D2 < 1.976 and dr_11 < 0.2536
        - 0.004915974 * max(0.0, Q.psi_0p1 - 0.9371031) / 0.01088804   # -0.5%  psi_0p1 > 0.9371
        - 0.004755842 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583) / 0.0009330603   # -0.5%  girth2_top5 < 0.00833 and sj3_pairmin_over_m > 0.09541
        - 0.004072432 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -0.4%  log_sum_pt < 6.903
        + 0.004039944 * max(0.0, 0.2416266 - Q.sj2_zsoft) / 0.06095505   # +0.4%  sj2_zsoft < 0.2416
        + 0.004018242 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2) / 0.0004272674   # +0.4%  sj2_dr > 0.2232 and C2_b2 < 0.04009
        - 0.003910101 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 14.45919 - Q.sj2_mass2) / 0.03398137   # -0.4%  psi_0p3 > 0.9897 and sj2_mass2 < 14.46
        + 0.003505882 * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.01931881   # +0.4%  z_dr_0p1_0p2 < 0.06473
        + 0.003416375 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, 0.04770182 - Q.z_6) / 5.265762e-05   # +0.3%  girth2_top5 < 0.00833 and z_6 < 0.0477
        + 0.003274547 * max(0.0, Q.e3 - 0.0001841806) / 4.035159e-05   # +0.3%  e3 > 0.0001842
        - 0.003104383 * max(0.0, Q.lam1 - 0.01649354) / 0.001003295   # -0.3%  lam1 > 0.01649
        + 0.002869501 * max(0.0, 933.1875 - Q.sum_pt_top30) / 20.26547   # +0.3%  sum_pt_top30 < 933.2
        - 0.002675152 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0) / 0.2039042   # -0.3%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p2_0p4 > 5
        + 0.002562708 * max(0.0, 0.00287991 - Q.girth2_top20) * max(0.0, 0.4357228 - Q.max_dr) / 4.319324e-05   # +0.3%  girth2_top20 < 0.00288 and max_dr < 0.4357
        + 0.002542072 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 0.777038 - Q.planar_flow) / 0.0106194   # +0.3%  z_dr_0p1_0p2 < 0.1203 and planar_flow < 0.777
        - 0.002302833 * max(0.0, 0.2982 - Q.max_dr) / 0.01350087   # -0.2%  max_dr < 0.2982
        - 0.001748799 * max(0.0, Q.sj2_dr - 0.2971569) / 0.005899669   # -0.2%  sj2_dr > 0.2972
        + 0.001608684 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt) / 2.722316   # +0.2%  z_dr_0_0p05 < 0.8459 and sum_pt < 949.9
        - 0.001341432 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, 0.948102 - Q.psi_0p2) / 3.786811e-05   # -0.1%  girth2_top5 < 0.00833 and psi_0p2 < 0.9481
        + 0.0007512822 * max(0.0, 0.00287991 - Q.girth2_top20) * max(0.0, Q.sum_pt - 1167.447) / 0.01532317   # +0.1%  girth2_top20 < 0.00288 and sum_pt > 1167
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6464058298319327, 2.758532037815126, 0.2117, 0.3808162289915966, 0.7664076680672269, 1.0329884453781513, 0.5344111344537815, 0.46567384453781513, 1.1620199579831934, 0.9874432773109244, 1.250378886554622, 0.7608730042016807, 0.5481350840336134, 1.5268417016806723, 0.2657419642857143, 0.43204495798319326]
T = [3.7800819278492646, 2.4466681016938026, 4.19012993697479, 4.288421093749999, 4.066909378282563]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +46%, n4 -18%, n9 +13%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.4560966 * h[1] / H_AVG[1]
            - 0.1774053 * h[4] / H_AVG[4]
            + 0.134693 * h[9] / H_AVG[9]
            - 0.08539733 * h[5] / H_AVG[5]
            - 0.07555714 * h[3] / H_AVG[3]
            + 0.03398581 * h[12] / H_AVG[12]
            + 0.02208993 * h[6] / H_AVG[6]
            + 0.009606438 * h[8] / H_AVG[8]
            - 0.005168451 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +23%, n1 -21%, n11 -8%, n12 +8%, n6 +5% ...
            - 0.3328233 * h[4] / H_AVG[4]
            + 0.2270177 * h[9] / H_AVG[9]
            - 0.2113996 * h[1] / H_AVG[1]
            - 0.07774583 * h[11] / H_AVG[11]
            + 0.07701144 * h[12] / H_AVG[12]
            + 0.04778026 * h[6] / H_AVG[6]
            + 0.01081573 * h[2] / H_AVG[2]
            - 0.007985215 * h[10] / H_AVG[10]
            + 0.007420934 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +14%, n0 +12%, n11 +11%, n14 -9%, n7 -7% ...
            - 0.2426577 * h[8] / H_AVG[8]
            + 0.1425246 * h[5] / H_AVG[5]
            + 0.1157015 * h[0] / H_AVG[0]
            + 0.1078173 * h[11] / H_AVG[11]
            - 0.08720379 * h[14] / H_AVG[14]
            - 0.06945993 * h[7] / H_AVG[7]
            + 0.06287457 * h[4] / H_AVG[4]
            - 0.05314391 * h[12] / H_AVG[12]
            - 0.05155048 * h[9] / H_AVG[9]
            + 0.0397618 * h[3] / H_AVG[3]
            - 0.01933315 * h[15] / H_AVG[15]
            + 0.007971279 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -25%, n0 -21%, n5 +17%, n6 -12%, n7 +10%, n15 +6% ...
            - 0.2540314 * h[8] / H_AVG[8]
            - 0.2072576 * h[0] / H_AVG[0]
            + 0.165604 * h[5] / H_AVG[5]
            - 0.1207229 * h[6] / H_AVG[6]
            + 0.09840846 * h[7] / H_AVG[7]
            + 0.05667011 * h[15] / H_AVG[15]
            + 0.03994296 * h[12] / H_AVG[12]
            + 0.03885045 * h[3] / H_AVG[3]
            - 0.01851206 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +30%, n5 -12%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3402339 * h[13] / H_AVG[13]
            + 0.3026479 * h[10] / H_AVG[10]
            - 0.1190617 * h[5] / H_AVG[5]
            + 0.06027024 * h[8] / H_AVG[8]
            - 0.05054223 * h[12] / H_AVG[12]
            - 0.03983783 * h[15] / H_AVG[15]
            + 0.03533431 * h[4] / H_AVG[4]
            - 0.032204 * h[7] / H_AVG[7]
            + 0.01986785 * h[0] / H_AVG[0]
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
