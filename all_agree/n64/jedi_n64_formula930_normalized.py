"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; all observables, tuned for agreement), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  14.3%   (on for 54% of jets)
  neuron  4:  11.7%   (on for 76% of jets)
  neuron  5:   9.6%   (on for 78% of jets)
  neuron  1:   9.6%   (on for 75% of jets)
  neuron  9:   9.4%   (on for 91% of jets)
  neuron  0:   8.0%   (on for 64% of jets)
  neuron 13:   7.2%   (on for 86% of jets)
  neuron 12:   5.4%   (on for 48% of jets)
  neuron 10:   4.8%   (on for 63% of jets)
  neuron  6:   4.7%   (on for 70% of jets)
  neuron  7:   4.7%   (on for 84% of jets)
  neuron  3:   4.4%   (on for 68% of jets)
  neuron 11:   2.4%   (on for 66% of jets)
  neuron 14:   1.9%   (on for 53% of jets)
  neuron 15:   1.7%   (on for 61% of jets)
  neuron  2:   0.2%   (on for 24% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.4% (the network: 81.1%); same class as the network for 92.5% of jets.

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
  Q.pair_mass_0_10         mass of particles 0 and 10 [GeV]
  Q.pair_mass_0_12         mass of particles 0 and 12 [GeV]
  Q.pair_mass_0_13         mass of particles 0 and 13 [GeV]
  Q.pair_mass_0_2          mass of particles 0 and 2 [GeV]
  Q.pair_mass_0_4          mass of particles 0 and 4 [GeV]
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
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top15           number of real particles among the 15 hardest
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_10                  pT of particle 10 [GeV]
  Q.ptdr0_10               pT10 · ΔR(0, 10) [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_14                  pT of particle 14 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_pt               pT [GeV] of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_10                   pT of particle 10 / total pT
  Q.z_11                   pT of particle 11 / total pT
  Q.z_2                    pT of particle 2 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_4                    pT of particle 4 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_8                    pT of particle 8 / total pT
  Q.soft10_z               pT share of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_z                pT share of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.sj3_z2                 pT share of subjet 2 of 3 (by pT)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_11                 pT share × ΔR of particle 11 (its part of the girth)
  Q.zdr_14                 pT share × ΔR of particle 14 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.pt1_over_pt0           pT1 / pT0
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_4               |Δη| of particle 4
  Q.soft10_abseta          |Δη| of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_12              |Δφ| of particle 12
  Q.absphi_13              |Δφ| of particle 13
  Q.absphi_2               |Δφ| of particle 2
  Q.absphi_4               |Δφ| of particle 4
  Q.absphi_7               |Δφ| of particle 7
  Q.absphi_9               |Δφ| of particle 9
  Q.soft4_absphi           |Δφ| of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft5_dr0              ΔR between the hardest and the 5. softest real particle (0 if among the 15 hardest)
  Q.dr0_10                 ΔR between particle 10 and the hardest particle
  Q.dr0_3                  ΔR between particle 3 and the hardest particle
  Q.dr0_8                  ΔR between particle 8 and the hardest particle
  Q.dr1_12                 ΔR between particle 12 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_12                  ΔR of particle 12 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr_8                   ΔR of particle 8 from the jet axis
  Q.dr_9                   ΔR of particle 9 from the jet axis
  Q.soft10_dr              ΔR from the jet axis of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_dr               ΔR from the jet axis of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_dr               ΔR from the jet axis of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_dr               ΔR from the jet axis of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_dr               ΔR from the jet axis of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_dr               ΔR from the jet axis of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_dr               ΔR from the jet axis of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.dr01                   ΔR between particles 0 and 1
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_2                  Δη of particle 2
  Q.eta_3                  Δη of particle 3
  Q.eta_4                  Δη of particle 4
  Q.eta_5                  Δη of particle 5
  Q.eta_9                  Δη of particle 9
  Q.phi_0                  Δφ of particle 0
  Q.phi_10                 Δφ of particle 10
  Q.phi_13                 Δφ of particle 13
  Q.phi_14                 Δφ of particle 14
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
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.z_dr_0p4_up            pT share of the particles with 0.4 ≤ ΔR < 10
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_phi               pT-weighted mean Δφ
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
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.tau43                  N-subjettiness τ4/τ3
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
        pair_mass_0_10=pair_mass(0, 10),
        pair_mass_0_12=pair_mass(0, 12),
        pair_mass_0_13=pair_mass(0, 13),
        pair_mass_0_2=pair_mass(0, 2),
        pair_mass_0_4=pair_mass(0, 4),
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
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        n_real_top15=sum(1 for x in pt[:15] if x > 0),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_0=pt[0],
        pt_1=pt[1],
        pt_10=pt[10],
        ptdr0_10=pt[10] * math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        pt_11=pt[11],
        pt_14=pt[14],
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft2_pt=softp(2, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft5_pt=softp(5, 'pt'),
        z_10=z[10],
        z_11=z[11],
        z_2=z[2],
        z_3=z[3],
        z_4=z[4],
        z_6=z[6],
        z_7=z[7],
        z_8=z[8],
        soft10_z=softp(10, 'z'),
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft9_z=softp(9, 'z'),
        sj3_z2=subjets(3)["z"][1],
        sj3_z3=subjets(3)["z"][2],
        z_top15_slots=sum(pt[:15]) / tot,
        z_top2_slots=sum(pt[:2]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_11=z[11] * dr[11],
        zdr_14=z[14] * dr[14],
        zdr_5=z[5] * dr[5],
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        abseta_4=abs(eta[4]),
        soft10_abseta=softp(10, 'abseta'),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_12=abs(phi[12]),
        absphi_13=abs(phi[13]),
        absphi_2=abs(phi[2]),
        absphi_4=abs(phi[4]),
        absphi_7=abs(phi[7]),
        absphi_9=abs(phi[9]),
        soft4_absphi=softp(4, 'absphi'),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft5_dr0=softp(5, 'dr0'),
        dr0_10=math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        dr0_3=math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        dr0_8=math.sqrt(dist2(0, 8)) if pt[8] > 0 else 0.0,
        dr1_12=math.sqrt(dist2(1, 12)) if pt[12] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_12=dr[12] if pt[12] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr_8=dr[8] if pt[8] > 0 else 0.0,
        dr_9=dr[9] if pt[9] > 0 else 0.0,
        soft10_dr=softp(10, 'dr'),
        soft2_dr=softp(2, 'dr'),
        soft3_dr=softp(3, 'dr'),
        soft4_dr=softp(4, 'dr'),
        soft5_dr=softp(5, 'dr'),
        soft7_dr=softp(7, 'dr'),
        soft8_dr=softp(8, 'dr'),
        dr01=math.sqrt(dist2(0, 1)),
        eta_0=eta[0],
        eta_1=eta[1],
        eta_2=eta[2],
        eta_3=eta[3],
        eta_4=eta[4],
        eta_5=eta[5],
        eta_9=eta[9],
        phi_0=phi[0],
        phi_10=phi[10],
        phi_13=phi[13],
        phi_14=phi[14],
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
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        z_dr_0p4_up=sum(z[i] for i in real if 0.4 <= dr[i] < 10),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
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
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    # scale S = 15.77;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.77072 * (0.0777004
        - 0.1759948 * max(0.0, Q.mass - 78.26182) / 21.33658   # -17.6%  mass > 78.26
        + 0.1008428 * max(0.0, Q.mass - 91.19) / 15.01545   # +10.1%  mass > 91.19
        + 0.09310115 * max(0.0, Q.mass - 89.74183) / 15.55572   # +9.3%  mass > 89.74
        - 0.06974268 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -7.0%  mass_top50 > 82.04
        + 0.05246361 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 0.002241068   # +5.2%  mass_over_sum_pt_sq < 0.007873
        + 0.04899722 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq) / 0.001689525   # +4.9%  mass_over_sum_pt_sq < 0.006939
        - 0.04648354 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -4.6%  mass < 101
        - 0.04273227 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -4.3%  e2_sq < 0.006167
        + 0.03163705 * max(0.0, Q.mass_top50 - 71.79516) / 24.22501   # +3.2%  mass_top50 > 71.8
        - 0.02920047 * max(0.0, Q.mass - 74.25181) / 24.07648   # -2.9%  mass > 74.25
        - 0.02162906 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -2.2%  girth2_top30 < 0.006364
        - 0.02105229 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # -2.1%  z_dr_0p2_0p4 < 0.09123
        - 0.02041065 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # -2.0%  girth2_top20 < 0.006374
        - 0.02006969 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.0%  tau1 < 0.07057
        + 0.01894839 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +1.9%  sum_pt_top50 < 1157
        + 0.01566999 * max(0.0, 0.007856958 - Q.girth2_top30) / 0.002601721   # +1.6%  girth2_top30 < 0.007857
        - 0.01536292 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # -1.5%  sum_pt_top40 < 1070
        - 0.01333506 * max(0.0, Q.mass - 92.85979) / 14.48273   # -1.3%  mass > 92.86
        + 0.01140937 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957) / 1.852228   # +1.1%  log_sum_pt < 6.989 and sum_pt_top50 > 959.1
        + 0.01138513 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4) / 0.08975043   # +1.1%  z_dr_0p2_0p4 < 0.1292
        + 0.01137293 * max(0.0, 0.004754444 - Q.mass_over_sum_pt_sq) / 0.0007997283   # +1.1%  mass_over_sum_pt_sq < 0.004754
        + 0.01029453 * max(0.0, 0.007538019 - Q.girth2_top20) / 0.002740841   # +1.0%  girth2_top20 < 0.007538
        + 0.008461115 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +0.8%  log_sum_pt < 7.017
        - 0.008177308 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -0.8%  lam1 < 0.005914
        - 0.007762569 * max(0.0, 0.002788182 - Q.soft5_z) / 0.001489217   # -0.8%  soft5_z < 0.002788
        + 0.007733466 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +0.8%  n_dr_0p2_0p4 < 15
        + 0.007636443 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # +0.8%  e3 < 0.0005178
        + 0.006846789 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +0.7%  psi_0p3 > 0.9956
        + 0.006783122 * max(0.0, 80.4 - Q.mass_top30) / 14.65577   # +0.7%  mass_top30 < 80.4
        + 0.00658842 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # +0.7%  log_sum_pt < 6.989
        + 0.006575552 * max(0.0, 0.005312783 - Q.girth2_top20) / 0.001437214   # +0.7%  girth2_top20 < 0.005313
        - 0.004969986 * max(0.0, 0.9906378 - Q.z_top50_slots) / 0.004783315   # -0.5%  z_top50_slots < 0.9906
        - 0.004911196 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # -0.5%  mass_top40 < 80.89
        + 0.004193916 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.02980347 - Q.eta_0) / 0.7431884   # +0.4%  mass < 101 and eta_0 < 0.0298
        + 0.003977323 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +0.4%  n_dr_0p2_0p4 < 11
        - 0.003804526 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -0.4%  sum_pt < 1013
        + 0.003734324 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +0.4%  e2 < 0.01879
        + 0.003078051 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +0.3%  lam2 < 0.0006155
        + 0.002378537 * max(0.0, 0.03843804 - Q.dr_2) / 0.0095582   # +0.2%  dr_2 < 0.03844
        - 0.002354935 * max(0.0, Q.psi_0p1 - 0.9184255) / 0.01693566   # -0.2%  psi_0p1 > 0.9184
        + 0.002269202 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) * max(0.0, 34.0625 - Q.pt_10) / 0.02980095   # +0.2%  mass_over_sum_pt_sq < 0.007873 and pt_10 < 34.06
        + 0.002264938 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) * max(0.0, 118.5 - Q.pt_2) / 1.543055   # +0.2%  z_dr_0p2_0p4 < 0.09123 and pt_2 < 118.5
        - 0.001918746 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.0223982 - Q.C3) / 0.2900897   # -0.2%  sum_pt < 1013 and C3 < 0.0224
        - 0.001638126 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -0.2%  sum_pt < 972
        - 0.001310055 * max(0.0, 0.05602756 - Q.C2) / 0.009882647   # -0.1%  C2 < 0.05603
        - 0.001129318 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, Q.pt_4 - 65.1875) / 0.009594495   # -0.1%  psi_0p3 > 0.9956 and pt_4 > 65.19
        - 0.001015744 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, Q.dr_max_012 - 0.06112084) / 0.0001025737   # -0.1%  psi_0p3 > 0.9956 and dr_max_012 > 0.06112
        + 0.00101361 * max(0.0, Q.z_dr_0p05_0p1 - 0.7108211) / 0.01402576   # +0.1%  z_dr_0p05_0p1 > 0.7108
        + 0.0007496375 * max(0.0, 7.017258 - Q.log_sum_pt) * max(0.0, 0.199317 - Q.soft4_dr) / 0.004832394   # +0.1%  log_sum_pt < 7.017 and soft4_dr < 0.1993
        + 0.0007361553 * max(0.0, 53.87362 - Q.mass) / 3.56547   # +0.1%  mass < 53.87
        + 0.0006430639 * max(0.0, 846.1934 - Q.sum_pt_top20) / 22.83716   # +0.1%  sum_pt_top20 < 846.2
        - 0.0006324357 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.eta_1 - 0.0001021922) / 0.1848682   # -0.1%  mass < 101 and eta_1 > 0.0001022
        + 0.0005584878 * max(0.0, 0.005913555 - Q.lam1) * max(0.0, 0.03230249 - Q.z_7) / 8.285101e-06   # +0.1%  lam1 < 0.005914 and z_7 < 0.0323
        - 0.0005111265 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3) / 0.2515325   # -0.1%  sum_pt < 1013 and M3 < 0.03457
        + 0.0004976963 * max(0.0, 0.05602756 - Q.C2) * max(0.0, Q.phi_10 - 0.0647583) / 3.977301e-05   # +0.0%  C2 < 0.05603 and phi_10 > 0.06476
        + 0.0004300554 * max(0.0, 0.002788182 - Q.soft5_z) * max(0.0, 30.0 - Q.n_real_top30) / 0.001485805   # +0.0%  soft5_z < 0.002788 and n_real_top30 < 30
        - 0.0002549724 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 0.05554124 - Q.z_4) / 0.06569978   # -0.0%  sum_pt < 972 and z_4 < 0.05554
        + 0.00016188 * max(0.0, Q.psi_0p1 - 0.9184255) * max(0.0, -0.02580643 - Q.eta_4) / 4.137911e-05   # +0.0%  psi_0p1 > 0.9184 and eta_4 < -0.02581
        - 0.0001579 * max(0.0, 858.8262 - Q.sum_pt_top40) / 3.625959   # -0.0%  sum_pt_top40 < 858.8
        - 3.582232e-06 * max(0.0, 0.007856958 - Q.girth2_top30) * max(0.0, Q.ptdr0_4 - 8.939062) / 0.0003207945   # -0.0%  girth2_top30 < 0.007857 and ptdr0_4 > 8.939
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 16.56;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.55807 * (0.03660404
        - 0.1118208 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -11.2%  log_sum_pt < 7.139
        + 0.1056728 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # +10.6%  log_sum_pt > 6.894
        + 0.07191987 * max(0.0, 0.02412652 - Q.girth2_top30) / 0.01613825   # +7.2%  girth2_top30 < 0.02413
        + 0.06069332 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +6.1%  n_particles > 38
        + 0.05989565 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +6.0%  log_sum_pt > 6.91
        - 0.04841357 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -4.8%  sum_pt_top50 > 959.1
        - 0.04407919 * max(0.0, 120.6 - Q.mass) / 38.8279   # -4.4%  mass < 120.6
        + 0.03797171 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.2734   # +3.8%  sum_pt_top2 < 689.2
        + 0.03584417 * max(0.0, Q.pt_entropy - 2.07371) / 0.8049786   # +3.6%  pt_entropy > 2.074
        + 0.03421394 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # +3.4%  sum_pt_top40 < 1070
        - 0.02871249 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # -2.9%  sj3_mass1 < 32.5
        - 0.02606746 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2) / 190.5117   # -2.6%  sj3_mass1 < 32.5 and sj3_mass2 < 18.68
        - 0.02507566 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt) / 15.05788   # -2.5%  n_particles > 38 and soft1_pt < 2.275
        - 0.02092713 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -2.1%  log_sum_pt > 6.989
        - 0.01723396 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -1.7%  log_sum_pt > 6.959
        - 0.0170126 * max(0.0, 31.35938 - Q.pt_9) / 6.538304   # -1.7%  pt_9 < 31.36
        + 0.01660077 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +1.7%  sum_pt < 1017
        - 0.01630465 * max(0.0, 30.26161 - Q.sj2_mass1) / 7.997784   # -1.6%  sj2_mass1 < 30.26
        - 0.016075 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -1.6%  n_dr_0p2_0p4 < 7
        + 0.01482708 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +1.5%  lam1 < 0.004673
        + 0.01442974 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1) / 1.005598   # +1.4%  n_particles > 38 and dr_1 < 0.1612
        + 0.01306405 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0) / 0.570275   # +1.3%  n_particles > 38 and dr_0 < 0.112
        + 0.01151037 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 39.23115   # +1.2%  mass_top20 < 47.89 and n_real_top40 > 29
        - 0.01082152 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -1.1%  psi_0p3 > 0.998
        + 0.01082038 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2) / 0.0008500795   # +1.1%  z_top30_slots > 0.9342 and C2 < 0.0728
        + 0.01052772 * max(0.0, 0.1416054 - Q.D3) / 0.05511785   # +1.1%  D3 < 0.1416
        + 0.008712559 * max(0.0, 0.0005522528 - Q.girth2_top3) / 0.000118253   # +0.9%  girth2_top3 < 0.0005523
        + 0.008595489 * max(0.0, 0.003270031 - Q.girth2_top15) / 0.0007969965   # +0.9%  girth2_top15 < 0.00327
        - 0.008412367 * max(0.0, 7.139296 - Q.log_sum_pt) * max(0.0, 5.351077 - Q.pt1_dr01) / 0.6416883   # -0.8%  log_sum_pt < 7.139 and pt1_dr01 < 5.351
        - 0.00757918 * max(0.0, Q.sum_pt_top5 - 430.75) / 176.7518   # -0.8%  sum_pt_top5 > 430.8
        + 0.006482904 * max(0.0, 0.001594761 - Q.soft5_z) / 0.0005119901   # +0.6%  soft5_z < 0.001595
        + 0.006120441 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.2313408 - Q.sj3_dr12) / 0.7781115   # +0.6%  n_particles > 38 and sj3_dr12 < 0.2313
        - 0.005666428 * max(0.0, 47.88842 - Q.mass_top20) / 6.040308   # -0.6%  mass_top20 < 47.89
        - 0.005399442 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # -0.5%  M3 < 0.03188
        + 0.005394017 * max(0.0, Q.n_particles - 38.0) * max(0.0, 57.87349 - Q.mass_top15) / 116.4836   # +0.5%  n_particles > 38 and mass_top15 < 57.87
        - 0.005362525 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.5%  n_dr_0p1_0p2 < 8
        + 0.005304282 * max(0.0, Q.z_top20_slots - 0.8965411) / 0.0341307   # +0.5%  z_top20_slots > 0.8965
        + 0.00501011 * max(0.0, 42.41192 - Q.mass_top30) / 2.481216   # +0.5%  mass_top30 < 42.41
        - 0.004928341 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -0.5%  z_dr_0_0p05 > 0.8789
        + 0.004524287 * max(0.0, Q.mass_top10 - 56.92192) / 8.071231   # +0.5%  mass_top10 > 56.92
        + 0.003738653 * max(0.0, Q.z_top30_slots - 0.9341838) / 0.03406818   # +0.4%  z_top30_slots > 0.9342
        + 0.003407785 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793) / 0.1941151   # +0.3%  z_top30_slots > 0.9342 and max_pair_mass > 13.05
        + 0.003229965 * max(0.0, 12.0 - Q.n_dr_0_0p05) / 3.466761   # +0.3%  n_dr_0_0p05 < 12
        - 0.003124536 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 0.02980347 - Q.eta_0) / 5.83951   # -0.3%  sum_pt_top5 > 430.8 and eta_0 < 0.0298
        + 0.003118896 * max(0.0, Q.sum_pt_top30 - 1191.938) / 6.938323   # +0.3%  sum_pt_top30 > 1192
        + 0.002540584 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874) / 0.05244257   # +0.3%  z_top30_slots > 0.9342 and ptdr0_3 > 7.408
        + 0.002090119 * max(0.0, Q.sj3_dr_max - 0.3276439) / 0.01855333   # +0.2%  sj3_dr_max > 0.3276
        - 0.001542395 * max(0.0, 0.03187688 - Q.M3) * max(0.0, Q.M2 - 0.05568888) / 0.0001450223   # -0.2%  M3 < 0.03188 and M2 > 0.05569
        + 0.001192982 * max(0.0, 0.03187688 - Q.M3) * max(0.0, 1.0 - Q.psi_0p3) / 0.000198055   # +0.1%  M3 < 0.03188 and psi_0p3 < 1
        + 0.001191388 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 11.29505 - Q.pair_mass_0_13) / 38.90723   # +0.1%  pt_9 < 31.36 and pair_mass_0_13 < 11.3
        + 0.0009763894 * max(0.0, 0.003270031 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 6.08612e-07   # +0.1%  girth2_top15 < 0.00327 and psi_0p3 > 0.9974
        + 0.0009549448 * max(0.0, 1.521582 - Q.soft1_pt) / 0.9527349   # +0.1%  soft1_pt < 1.522
        - 0.0008893133 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 0.9624339 - Q.tau43) / 39.08134   # -0.1%  sum_pt_top2 < 689.2 and tau43 < 0.9624
        - 0.0008221703 * max(0.0, 0.009068077 - Q.tau3) / 0.0001836134   # -0.1%  tau3 < 0.009068
        - 0.0007515982 * Q.absphi_0 / 0.03295052   # -0.1%  absphi_0
        - 0.0006767251 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.002968979 - Q.soft10_abseta) / 0.000988223   # -0.1%  n_dr_0p2_0p4 < 7 and soft10_abseta < 0.002969
        + 0.000650693 * max(0.0, 0.02412652 - Q.girth2_top30) * max(0.0, Q.zdr_14 - 0.0025721) / 1.011019e-06   # +0.1%  girth2_top30 < 0.02413 and zdr_14 > 0.002572
        - 0.0006491097 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 0.3241858 - Q.dr1_12) / 1.371943   # -0.1%  pt_9 < 31.36 and dr1_12 < 0.3242
        + 0.0002269221 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_4 - 6.984554) / 0.03731215   # +0.0%  z_top30_slots > 0.9342 and ptdr0_4 > 6.985
        + 0.0001967824 * max(0.0, Q.N2 - 0.4678622) / 0.001556028   # +0.0%  N2 > 0.4679
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 6.629;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.629063 * (0.06203312
        - 0.1692067 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -16.9%  mass < 92.86
        - 0.09092447 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # -9.1%  sum_pt < 1261
        + 0.07866815 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15) / 0.0009227558   # +7.9%  log_sum_pt > 6.903 and girth2_top15 < 0.02147
        + 0.07116097 * max(0.0, 91.19 - Q.mass) / 16.46423   # +7.1%  mass < 91.19
        + 0.06717273 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # +6.7%  sum_pt > 1017
        - 0.06363255 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15) / 1.045047   # -6.4%  sum_pt_top50 > 988.5 and girth2_top15 < 0.02147
        + 0.06231425 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +6.2%  mass_over_sum_pt < 0.09795
        + 0.04971537 * max(0.0, 1048.098 - Q.sum_pt_top50) / 44.36839   # +5.0%  sum_pt_top50 < 1048
        - 0.04029898 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # -4.0%  sum_pt_top50 < 1079
        - 0.0347422 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -3.5%  log_sum_pt > 6.959
        - 0.02465905 * max(0.0, 1018.698 - Q.sum_pt_top40) / 34.89083   # -2.5%  sum_pt_top40 < 1019
        - 0.02206146 * max(0.0, 996.8867 - Q.sum_pt_top30) / 42.94346   # -2.2%  sum_pt_top30 < 996.9
        - 0.02199394 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003463934   # -2.2%  log_sum_pt > 6.903 and psi_0p3 > 0.9299
        + 0.02194953 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # +2.2%  mass_top50 < 92.17
        - 0.01892888 * max(0.0, Q.sum_pt_top40 - 984.7009) / 58.19535   # -1.9%  sum_pt_top40 > 984.7
        + 0.01840801 * max(0.0, 1041.263 - Q.sum_pt_top40) / 49.16087   # +1.8%  sum_pt_top40 < 1041
        + 0.01577012 * max(0.0, 82.66587 - Q.mass_top30) / 15.96638   # +1.6%  mass_top30 < 82.67
        - 0.01374793 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -1.4%  girth2_top30 < 0.006364
        - 0.01319652 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # -1.3%  sum_pt > 1116
        + 0.01317846 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +1.3%  sum_pt_top40 > 1070
        + 0.01275007 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +1.3%  lam1 < 0.01174
        + 0.01197235 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +1.2%  mass < 78.26
        + 0.009395346 * max(0.0, 966.0633 - Q.sum_pt_top30) / 29.99923   # +0.9%  sum_pt_top30 < 966.1
        - 0.008166127 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # -0.8%  psi_0p3 > 0.9974
        - 0.006500936 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 0.397021 - Q.max_dr) / 11.85447   # -0.7%  sum_pt < 1261 and max_dr < 0.397
        - 0.00464003 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.mass_top40 - 77.93668) / 1.109603   # -0.5%  log_sum_pt > 6.903 and mass_top40 > 77.94
        + 0.004119903 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 30.0 - Q.n_real_top30) / 223.0591   # +0.4%  sum_pt < 1261 and n_real_top30 < 30
        + 0.003758171 * max(0.0, Q.log_sum_pt - 6.903423) / 0.05662275   # +0.4%  log_sum_pt > 6.903
        - 0.003374181 * max(0.0, 51.0 - Q.n_particles) / 9.158477   # -0.3%  n_particles < 51
        + 0.002759166 * max(0.0, Q.sj3_pair_mass_min - 18.02031) / 11.34956   # +0.3%  sj3_pair_mass_min > 18.02
        - 0.002731384 * max(0.0, Q.sum_pt_top50 - 988.4554) / 62.81258   # -0.3%  sum_pt_top50 > 988.5
        - 0.002666594 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 1.433504e-05 - Q.e3) / 1.867416e-07   # -0.3%  log_sum_pt > 6.903 and e3 < 1.434e-05
        - 0.002631449 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.3%  log_sum_pt > 7.063
        + 0.002190341 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 4.657698e-07   # +0.2%  sum_pt < 972 and e4 < 5.851e-08
        + 0.002012239 * max(0.0, Q.sum_pt - 1017.435) * max(0.0, Q.C2 - 0.02207346) / 1.810321   # +0.2%  sum_pt > 1017 and C2 > 0.02207
        + 0.001582346 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.01585274 - Q.z_10) / 5.217933e-05   # +0.2%  log_sum_pt > 6.903 and z_10 < 0.01585
        + 0.001063899 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.soft5_z - 0.0005078411) / 4.845388e-05   # +0.1%  log_sum_pt > 6.903 and soft5_z > 0.0005078
        - 0.0008064751 * max(0.0, 0.004763596 - Q.girth2_top30) / 0.0009958273   # -0.1%  girth2_top30 < 0.004764
        + 0.000745977 * max(0.0, Q.sum_pt_top20 - 1129.275) / 6.499168   # +0.1%  sum_pt_top20 > 1129
        - 0.000683988 * max(0.0, 56.53125 - Q.pt_6) / 17.03571   # -0.1%  pt_6 < 56.53
        - 0.0005675275 * max(0.0, Q.z_top20_slots - 0.7516206) * max(0.0, Q.sj3_mass1 - 6.519856) / 0.9623658   # -0.1%  z_top20_slots > 0.7516 and sj3_mass1 > 6.52
        + 0.0005030185 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, 0.01898316 - Q.z_10) / 1.422215e-05   # +0.1%  log_sum_pt > 7.063 and z_10 < 0.01898
        + 0.0004944025 * max(0.0, Q.sum_pt_top50 - 1156.659) / 13.71464   # +0.0%  sum_pt_top50 > 1157
        + 0.0004941987 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # +0.0%  sum_pt < 972
        + 0.00039848 * max(0.0, 1048.098 - Q.sum_pt_top50) * max(0.0, 0.3742561 - Q.max_dr) / 1.246515   # +0.0%  sum_pt_top50 < 1048 and max_dr < 0.3743
        + 0.0003928325 * max(0.0, 996.8867 - Q.sum_pt_top30) * max(0.0, 0.03275811 - Q.soft7_dr) / 0.03631807   # +0.0%  sum_pt_top30 < 996.9 and soft7_dr < 0.03276
        + 0.000255958 * max(0.0, Q.z_top20_slots - 0.7516206) / 0.1429779   # +0.0%  z_top20_slots > 0.7516
        - 0.0002214415 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.08734131) / 0.003497383   # -0.0%  sum_pt_top20 > 1129 and eta_1 > 0.08734
        + 0.000184438 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.girth2_top30 - 0.008376291) / 1.929368e-05   # +0.0%  log_sum_pt > 7.063 and girth2_top30 > 0.008376
        - 9.005786e-05 * max(0.0, Q.sum_pt - 1115.723) * max(0.0, Q.mean_phi - 0.0004404746) / 0.001864981   # -0.0%  sum_pt > 1116 and mean_phi > 0.0004405
        + 6.25695e-05 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, 1.0 - Q.n_dr_0_0p05) / 0.1096012   # +0.0%  sum_pt_top20 > 1129 and n_dr_0_0p05 < 1
        + 5.38285e-05 * max(0.0, Q.sum_pt - 1115.723) * max(0.0, -0.1135254 - Q.eta_5) / 0.03156393   # +0.0%  sum_pt > 1116 and eta_5 < -0.1135
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 3.425;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.424814 * (-0.01692251
        - 0.177409 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # -17.7%  girth2 < 0.009615
        + 0.1184578 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +11.8%  girth2_top40 < 0.008841
        + 0.07209436 * max(0.0, 0.08665515 - Q.mass_over_sum_pt) / 0.01542259   # +7.2%  mass_over_sum_pt < 0.08666
        + 0.07169648 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +7.2%  mass < 92.86
        + 0.04826584 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +4.8%  n_particles < 46
        - 0.04751205 * max(0.0, 80.4 - Q.mass_top40) / 12.19371   # -4.8%  mass_top40 < 80.4
        + 0.03809853 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741) / 0.0491674   # +3.8%  n_dr_0p2_0p4 < 8 and z_top50_slots > 0.9787
        + 0.02926431 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +2.9%  n_dr_0p2_0p4 < 5
        + 0.02908292 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732) / 1477.074   # +2.9%  n_particles < 46 and sum_pt_top30 > 800.7
        + 0.02839025 * max(0.0, 27.56535 - Q.sj2_mass1) / 6.309897   # +2.8%  sj2_mass1 < 27.57
        - 0.02615377 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -2.6%  tau21 < 0.3862
        + 0.02527206 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) / 2.171439   # +2.5%  n_dr_0p1_0p2 < 10
        - 0.02232704 * max(0.0, 0.006026828 - Q.girth2_top40) / 0.001351778   # -2.2%  girth2_top40 < 0.006027
        - 0.01637436 * max(0.0, 0.003235754 - Q.zdr_0) / 0.0003842694   # -1.6%  zdr_0 < 0.003236
        + 0.0154729 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # +1.5%  psi_0p3 > 0.998
        + 0.01419306 * max(0.0, 0.009614971 - Q.girth2) * max(0.0, 0.02980347 - Q.eta_0) / 0.0001109857   # +1.4%  girth2 < 0.009615 and eta_0 < 0.0298
        - 0.01407743 * max(0.0, 79.21004 - Q.mass_top50) / 10.65754   # -1.4%  mass_top50 < 79.21
        - 0.01386159 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349) / 64.17204   # -1.4%  n_particles < 46 and mass_top15 > 57.87
        + 0.01091281 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +1.1%  tau4 < 0.01627
        + 0.01053936 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +1.1%  lam2 < 0.0006155
        + 0.01050936 * max(0.0, 27.56535 - Q.sj2_mass1) * max(0.0, 0.1455078 - Q.phi_13) / 0.9329567   # +1.1%  sj2_mass1 < 27.57 and phi_13 < 0.1455
        + 0.01002229 * max(0.0, Q.psi_0p2 - 0.9985434) / 0.00014398   # +1.0%  psi_0p2 > 0.9985
        - 0.01000954 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -1.0%  mass < 53.87
        + 0.008528977 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.002788182 - Q.soft5_z) / 0.00316606   # +0.9%  n_dr_0p2_0p4 < 8 and soft5_z < 0.002788
        - 0.008383882 * max(0.0, 86.4 - Q.mass) / 13.67632   # -0.8%  mass < 86.4
        - 0.008163112 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0) / 3.186294   # -0.8%  n_dr_0p2_0p4 < 5 and n_dr_0p1_0p2 > 9
        + 0.007997952 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, Q.n_dr_0_0p05 - 9.0) / 0.03881192   # +0.8%  z_top30_slots > 0.9735 and n_dr_0_0p05 > 9
        + 0.007933782 * max(0.0, 11.0 - Q.n_for_90pct) / 0.4757244   # +0.8%  n_for_90pct < 11
        - 0.007071832 * max(0.0, Q.psi_0p3 - 0.9995915) / 8.716828e-05   # -0.7%  psi_0p3 > 0.9996
        + 0.006948321 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.eccentricity - 0.7333655) / 0.1710102   # +0.7%  n_dr_0p2_0p4 < 5 and eccentricity > 0.7334
        - 0.006472017 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03447112 - Q.z_10) / 0.01282533   # -0.6%  n_dr_0p2_0p4 < 5 and z_10 < 0.03447
        - 0.006123276 * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.0585395   # -0.6%  n_dr_0p2_0p4 < 1
        + 0.006049676 * max(0.0, 0.0006154841 - Q.lam2) * max(0.0, 0.1309949 - Q.soft10_dr) / 9.534011e-06   # +0.6%  lam2 < 0.0006155 and soft10_dr < 0.131
        - 0.005496062 * max(0.0, 79.21004 - Q.mass_top50) * max(0.0, Q.mass_top3 - 8.921413) / 4.657205   # -0.5%  mass_top50 < 79.21 and mass_top3 > 8.921
        + 0.005486215 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 0.1269531 - Q.eta_9) / 0.001302695   # +0.5%  z_top30_slots > 0.9735 and eta_9 < 0.127
        + 0.005316882 * max(0.0, 0.0006154841 - Q.lam2) * max(0.0, 1.0 - Q.n_dr_0p4_up) / 0.0001383002   # +0.5%  lam2 < 0.0006155 and n_dr_0p4_up < 1
        - 0.005035806 * max(0.0, Q.z_top30_slots - 0.9734886) / 0.01006171   # -0.5%  z_top30_slots > 0.9735
        - 0.004162164 * max(0.0, 0.001260456 - Q.lam1) / 8.753718e-05   # -0.4%  lam1 < 0.00126
        - 0.004078084 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.eta_1 - -0.04302979) / 3.206471e-05   # -0.4%  psi_0p3 > 0.998 and eta_1 > -0.04303
        - 0.004044174 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243) / 0.0001457263   # -0.4%  tau21 < 0.3862 and lam1 > 0.007671
        + 0.003956548 * max(0.0, 0.0006154841 - Q.lam2) * max(0.0, -9.840088 - Q.orientation_deg) / 0.002640902   # +0.4%  lam2 < 0.0006155 and orientation_deg < -9.84
        + 0.003624553 * max(0.0, 0.2738063 - Q.max_dr) / 0.009268927   # +0.4%  max_dr < 0.2738
        - 0.003208067 * max(0.0, 2.410481 - Q.D2) / 0.587301   # -0.3%  D2 < 2.41
        + 0.002725061 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # +0.3%  n_dr_0p2_0p4 < 8
        + 0.002708593 * max(0.0, 2.410481 - Q.D2) * max(0.0, 0.136982 - Q.soft5_dr) / 0.01829419   # +0.3%  D2 < 2.41 and soft5_dr < 0.137
        + 0.002575535 * max(0.0, 0.008840538 - Q.girth2_top40) * max(0.0, Q.ptdr0_6 - 5.648151) / 0.0006378931   # +0.3%  girth2_top40 < 0.008841 and ptdr0_6 > 5.648
        - 0.002057645 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5) / 0.004484684   # -0.2%  n_dr_0p2_0p4 < 5 and zdr_5 < 0.007099
        + 0.001941927 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03329468 - Q.eta_3) / 0.04612469   # +0.2%  n_dr_0p2_0p4 < 5 and eta_3 < 0.03329
        + 0.001593376 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1) / 0.1325294   # +0.2%  n_dr_0p2_0p4 < 5 and psi_0p1 < 0.9538
        + 0.001488335 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.max_dr - 0.4357228) / 0.004818361   # +0.1%  D2 < 2.41 and max_dr > 0.4357
        + 0.001385895 * max(0.0, 0.04748396 - Q.z_dr_0p1_0p2) / 0.01200196   # +0.1%  z_dr_0p1_0p2 < 0.04748
        + 0.001330581 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421) / 0.000406438   # +0.1%  D2 < 2.41 and psi_0p3 > 0.9985
        + 0.001309617 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 988.4375 - Q.sum_pt_top30) / 17.46027   # +0.1%  n_dr_0p2_0p4 < 5 and sum_pt_top30 < 988.4
        + 0.0005844381 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, 0.254323 - Q.sd_zg) / 2.332676e-05   # +0.1%  psi_0p3 > 0.998 and sd_zg < 0.2543
        + 0.0005429435 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, 0.004962409 - Q.soft10_z) / 1.385853e-06   # +0.1%  psi_0p3 > 0.998 and soft10_z < 0.004962
        + 0.0005138539 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.soft4_dr - 0.08815263) / 3.958279e-05   # +0.1%  psi_0p3 > 0.998 and soft4_dr > 0.08815
        + 0.0004849861 * max(0.0, Q.psi_0p2 - 0.9985434) * max(0.0, 0.05004883 - Q.absphi_4) / 2.395711e-06   # +0.0%  psi_0p2 > 0.9985 and absphi_4 < 0.05005
        + 0.0004194628 * max(0.0, Q.psi_0p2 - 0.9985434) * max(0.0, 0.08575439 - Q.abseta_4) / 6.302321e-06   # +0.0%  psi_0p2 > 0.9985 and abseta_4 < 0.08575
        - 0.0001434788 * max(0.0, 0.2738063 - Q.max_dr) * max(0.0, Q.phi_14 - 0.102478) / 1.293714e-05   # -0.0%  max_dr < 0.2738 and phi_14 > 0.1025
        - 0.0001158085 * max(0.0, 0.0006154841 - Q.lam2) * max(0.0, Q.soft8_dr - 0.04226709) / 1.028625e-05   # -0.0%  lam2 < 0.0006155 and soft8_dr > 0.04227
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 10.43;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.43078 * (0.1018032
        + 0.09833732 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +9.8%  mass < 101
        - 0.07493704 * max(0.0, 78.26182 - Q.mass) / 9.857173   # -7.5%  mass < 78.26
        + 0.05957946 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +6.0%  mass < 74.25
        + 0.05572467 * max(0.0, 143.7876 - Q.mass) / 57.99337   # +5.6%  mass < 143.8
        - 0.05210196 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -5.2%  mass_top40 < 163.3
        - 0.04603929 * max(0.0, 80.78464 - Q.mass) / 10.84952   # -4.6%  mass < 80.78
        + 0.04491732 * max(0.0, Q.e2_sq - 0.009606007) / 0.003182984   # +4.5%  e2_sq > 0.009606
        - 0.04140978 * max(0.0, Q.n_particles - 22.0) / 23.9717   # -4.1%  n_particles > 22
        + 0.04069356 * max(0.0, 120.6 - Q.mass) / 38.8279   # +4.1%  mass < 120.6
        - 0.03993056 * max(0.0, 86.4 - Q.mass) / 13.67632   # -4.0%  mass < 86.4
        + 0.03949772 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +3.9%  n_dr_0p2_0p4 < 26
        - 0.03244096 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -3.2%  mass_top40 < 83.33
        - 0.03087441 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -3.1%  mass < 92.86
        - 0.02922569 * max(0.0, Q.lam1 - 0.008241985) / 0.002595658   # -2.9%  lam1 > 0.008242
        - 0.02768424 * max(0.0, 91.19 - Q.mass_top15) / 34.88803   # -2.8%  mass_top15 < 91.19
        - 0.02694468 * max(0.0, 91.69753 - Q.mass_top30) / 22.0127   # -2.7%  mass_top30 < 91.7
        + 0.02368089 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt) / 36.47388   # +2.4%  n_particles > 22 and soft1_pt < 2.275
        - 0.02049829 * max(0.0, 0.004855289 - Q.girth2_top15) / 0.001418414   # -2.0%  girth2_top15 < 0.004855
        - 0.01866782 * max(0.0, Q.girth2_top15 - 0.00727763) / 0.002923446   # -1.9%  girth2_top15 > 0.007278
        + 0.01840293 * max(0.0, 57.87349 - Q.mass_top15) / 12.46085   # +1.8%  mass_top15 < 57.87
        + 0.01393036 * max(0.0, 120.6 - Q.mass_top30) / 45.45988   # +1.4%  mass_top30 < 120.6
        + 0.01326769 * max(0.0, Q.e2_sq - 0.01396296) / 0.002220692   # +1.3%  e2_sq > 0.01396
        + 0.01326416 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +1.3%  mass_top40 < 67.73
        + 0.01070544 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +1.1%  psi_0p3 > 0.9974
        + 0.009897317 * max(0.0, 0.9909875 - Q.psi_0p1) / 0.2210377   # +1.0%  psi_0p1 < 0.991
        + 0.009102944 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +0.9%  n_dr_0p2_0p4 < 15
        + 0.008725026 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.n_dr_0_0p05 - 10.0) / 143.1065   # +0.9%  n_particles > 22 and n_dr_0_0p05 > 10
        + 0.008125413 * max(0.0, Q.girth2_top15 - 0.01563836) / 0.001334556   # +0.8%  girth2_top15 > 0.01564
        + 0.006861526 * max(0.0, Q.sj3_pair_mass_min - 32.51366) / 5.88267   # +0.7%  sj3_pair_mass_min > 32.51
        - 0.00587533 * max(0.0, Q.girth - 0.1207452) / 0.004285542   # -0.6%  girth > 0.1207
        - 0.005427101 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 13.0 - Q.n_dr_0_0p05) / 0.003836025   # -0.5%  psi_0p3 > 0.9974 and n_dr_0_0p05 < 13
        + 0.005381301 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +0.5%  mass_top30 < 60.44
        + 0.005149713 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sj3_dr13 - 0.1348442) / 0.3099271   # +0.5%  n_dr_0p2_0p4 < 15 and sj3_dr13 > 0.1348
        + 0.004800362 * max(0.0, 69.02716 - Q.mass_top15) / 18.23029   # +0.5%  mass_top15 < 69.03
        - 0.004272649 * max(0.0, Q.log_sum_pt - 6.97212) / 0.02498281   # -0.4%  log_sum_pt > 6.972
        - 0.00367101 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # -0.4%  sj2_dr > 0.2232
        - 0.003618316 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, Q.zdr_0 - 0.001901263) / 0.02085507   # -0.4%  mass_top40 < 83.33 and zdr_0 > 0.001901
        - 0.003550919 * max(0.0, 62.55 - Q.mass_top40) / 6.231505   # -0.4%  mass_top40 < 62.55
        - 0.003398673 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.max_dr - 0.1939977) / 1.031549   # -0.3%  n_dr_0p2_0p4 < 15 and max_dr > 0.194
        - 0.003379695 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.z_top50_slots) / 0.03204382   # -0.3%  n_dr_0p2_0p4 < 15 and z_top50_slots < 1
        + 0.003289914 * max(0.0, Q.e3 - 0.0001841806) / 4.035159e-05   # +0.3%  e3 > 0.0001842
        - 0.003182388 * max(0.0, Q.n_particles - 22.0) * max(0.0, 33.21875 - Q.pt_7) / 60.66924   # -0.3%  n_particles > 22 and pt_7 < 33.22
        - 0.003121305 * max(0.0, Q.lam2 - 0.001776308) / 0.0005876074   # -0.3%  lam2 > 0.001776
        - 0.003035696 * max(0.0, 21.52288 - Q.sj2_mass1) / 3.113774   # -0.3%  sj2_mass1 < 21.52
        + 0.003031189 * max(0.0, 0.3036026 - Q.planar_flow) / 0.04665298   # +0.3%  planar_flow < 0.3036
        + 0.002995634 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.605781   # +0.3%  n_particles > 22 and psi_0p3 > 0.9638
        - 0.002765935 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 56.19962   # -0.3%  n_dr_0p2_0p4 < 26 and n_dr_0p1_0p2 > 11
        - 0.002450941 * max(0.0, 0.9909875 - Q.psi_0p1) * max(0.0, Q.pt_1 - 99.0) / 4.936238   # -0.2%  psi_0p1 < 0.991 and pt_1 > 99
        - 0.002450379 * max(0.0, Q.sd_rg - 0.2042612) / 0.02046744   # -0.2%  sd_rg > 0.2043
        - 0.001958709 * max(0.0, Q.girth2_top15 - 0.00727763) * max(0.0, 33.78125 - Q.pt_11) / 0.03009459   # -0.2%  girth2_top15 > 0.007278 and pt_11 < 33.78
        - 0.001350005 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 0.4953696 - Q.D2_b2) / 0.08717725   # -0.1%  mass_top40 < 83.33 and D2_b2 < 0.4954
        - 0.001307716 * max(0.0, 75.26407 - Q.mass_top15) / 22.29039   # -0.1%  mass_top15 < 75.26
        - 0.001288882 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.005911134 - Q.zdr_0) / 9.63153e-07   # -0.1%  psi_0p3 > 0.9974 and zdr_0 < 0.005911
        - 0.001198358 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # -0.1%  e2 > 0.06524
        - 0.0009992911 * max(0.0, 0.3036026 - Q.planar_flow) * max(0.0, Q.n_dr_0p1_0p2 - 13.0) / 0.1025731   # -0.1%  planar_flow < 0.3036 and n_dr_0p1_0p2 > 13
        - 0.0007468332 * max(0.0, Q.pair_mass_0_12 - 15.47191) / 0.4335287   # -0.1%  pair_mass_0_12 > 15.47
        - 0.0002926348 * max(0.0, 120.6 - Q.mass) * max(0.0, 0.9943058 - Q.psi_0p3) / 0.08236936   # -0.0%  mass < 120.6 and psi_0p3 < 0.9943
        + 0.0002213765 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.sj3_mass1 - 32.50209) / 17.4117   # +0.0%  n_particles > 22 and sj3_mass1 > 32.5
        - 0.0001857843 * max(0.0, 0.3036026 - Q.planar_flow) * max(0.0, Q.z_dr_0p4_up - 0.0) / 5.226105e-06   # -0.0%  planar_flow < 0.3036 and z_dr_0p4_up > 0
        + 0.0001334779 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.zdr_0 - 0.0008718296) / 0.0994482   # +0.0%  mass < 101 and zdr_0 > 0.0008718
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 9.559;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.55889 * (0.05281121
        + 0.1892459 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +18.9%  sum_pt_top50 > 934.2
        - 0.08281517 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -8.3%  log_sum_pt > 6.91
        - 0.06209825 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # -6.2%  sum_pt > 986.1
        - 0.05482202 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -5.5%  log_sum_pt > 6.92
        + 0.05250575 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +5.3%  sum_pt > 907.9
        + 0.03840861 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +3.8%  n_particles < 64
        + 0.03101681 * max(0.0, Q.sum_pt_top30 - 933.1875) / 79.88556   # +3.1%  sum_pt_top30 > 933.2
        - 0.02943963 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -2.9%  sum_pt_top40 > 1025
        - 0.02624992 * max(0.0, Q.mass - 91.19) / 15.01545   # -2.6%  mass > 91.19
        - 0.02558447 * max(0.0, 150.0144 - Q.mass_top40) / 66.86198   # -2.6%  mass_top40 < 150
        - 0.02521079 * max(0.0, Q.mass_over_sum_pt - 0.09046749) / 0.01421039   # -2.5%  mass_over_sum_pt > 0.09047
        + 0.023387 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +2.3%  sum_pt > 907.9 and e4 < 5.851e-08
        + 0.02308237 * max(0.0, Q.mass - 74.25181) / 24.07648   # +2.3%  mass > 74.25
        - 0.02222543 * max(0.0, Q.z_top30_slots - 0.9048492) / 0.05621641   # -2.2%  z_top30_slots > 0.9048
        + 0.02203485 * max(0.0, Q.log_sum_pt - 6.935549) / 0.0371538   # +2.2%  log_sum_pt > 6.936
        - 0.01826592 * max(0.0, 787.6281 - Q.sum_pt_top3) / 328.0371   # -1.8%  sum_pt_top3 < 787.6
        + 0.01638851 * max(0.0, 0.03787151 - Q.M3) / 0.01370579   # +1.6%  M3 < 0.03787
        + 0.0142927 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # +1.4%  log_sum_pt > 6.989
        + 0.01353045 * max(0.0, Q.sd_mass - 69.65633) / 11.73272   # +1.4%  sd_mass > 69.66
        - 0.01269896 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2) / 10.81389   # -1.3%  n_particles < 64 and D2 < 2.179
        - 0.01260521 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -1.3%  max_dr > 0.2405
        + 0.01228533 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +1.2%  n_dr_0p2_0p4 < 11
        + 0.01165531 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +1.2%  mass_over_sum_pt > 0.1709
        - 0.01127738 * max(0.0, 0.1937447 - Q.z_dr_0p2_0p4) / 0.146372   # -1.1%  z_dr_0p2_0p4 < 0.1937
        + 0.01119499 * max(0.0, Q.n_pt_above_1 - 28.0) / 14.80415   # +1.1%  n_pt_above_1 > 28
        + 0.01093822 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # +1.1%  e2 < 0.03876
        - 0.01052541 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -1.1%  mass_over_sum_pt_sq > 0.0292
        - 0.009117139 * max(0.0, Q.sd_mass - 86.4) / 6.275027   # -0.9%  sd_mass > 86.4
        - 0.008899787 * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 9.82015   # -0.9%  n_dr_0p1_0p2 < 21
        + 0.008199584 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2) / 0.001543279   # +0.8%  psi_0p3 > 0.9974 and D2 < 3.345
        - 0.007925377 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.2535773 - Q.dr_11) / 0.00936088   # -0.8%  log_sum_pt > 6.91 and dr_11 < 0.2536
        - 0.00757718 * max(0.0, Q.mass_top40 - 120.6) / 5.823744   # -0.8%  mass_top40 > 120.6
        - 0.006489039 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.1958008 - Q.absphi_13) / 0.007588689   # -0.6%  log_sum_pt > 6.91 and absphi_13 < 0.1958
        + 0.006298939 * max(0.0, Q.girth2 - 0.004756928) / 0.005348122   # +0.6%  girth2 > 0.004757
        - 0.005982891 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.1269782 - Q.dr_12) / 0.003311185   # -0.6%  log_sum_pt > 6.91 and dr_12 < 0.127
        + 0.005693776 * max(0.0, 23.31647 - Q.sj2_mass1) / 3.96589   # +0.6%  sj2_mass1 < 23.32
        + 0.005634779 * max(0.0, Q.girth2_top30 - 0.006363916) / 0.003802703   # +0.6%  girth2_top30 > 0.006364
        + 0.005489667 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.5%  psi_0p3 > 0.9974
        + 0.005442978 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # +0.5%  n_dr_0p1_0p2 < 8
        + 0.005441035 * max(0.0, 0.01375115 - Q.z_11) / 0.0009450582   # +0.5%  z_11 < 0.01375
        - 0.004857878 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -0.5%  mass_top50 > 157.5
        - 0.004525678 * max(0.0, Q.z_top30_slots - 0.9048492) * max(0.0, Q.sj3_mass1 - 10.87918) / 0.1762583   # -0.5%  z_top30_slots > 0.9048 and sj3_mass1 > 10.88
        - 0.004509384 * max(0.0, Q.girth2_top50 - 0.01951641) / 0.001208668   # -0.5%  girth2_top50 > 0.01952
        - 0.004319216 * max(0.0, 0.0004816552 - Q.girth2_top15) / 3.66329e-05   # -0.4%  girth2_top15 < 0.0004817
        + 0.003750497 * max(0.0, 1.976207 - Q.D2) / 0.367701   # +0.4%  D2 < 1.976
        - 0.003660287 * max(0.0, 0.5494307 - Q.tau21) / 0.1591168   # -0.4%  tau21 < 0.5494
        + 0.003566701 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 0.03260972 - Q.z_8) / 0.1432208   # +0.4%  n_particles < 64 and z_8 < 0.03261
        - 0.003461048 * max(0.0, 14.14062 - Q.pt_11) / 0.9740783   # -0.3%  pt_11 < 14.14
        + 0.003005761 * max(0.0, Q.mass - 172.8) / 0.7291439   # +0.3%  mass > 172.8
        + 0.002205745 * max(0.0, Q.sum_pt_top20 - 1129.275) / 6.499168   # +0.2%  sum_pt_top20 > 1129
        + 0.002188375 * max(0.0, Q.z_dr_0_0p05 - 0.9084912) / 0.009227347   # +0.2%  z_dr_0_0p05 > 0.9085
        + 0.001774485 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, Q.dr0_10 - 0.06373449) / 0.002273253   # +0.2%  log_sum_pt > 6.91 and dr0_10 > 0.06373
        - 0.001522574 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.2%  sum_pt_top10 > 943.7
        - 0.001345571 * max(0.0, 1.20817 - Q.ptdr0_10) / 0.4204752   # -0.1%  ptdr0_10 < 1.208
        + 0.001198006 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # +0.1%  mass_top10 > 71.78
        + 0.0006700304 * max(0.0, 0.1937447 - Q.z_dr_0p2_0p4) * max(0.0, Q.absphi_9 - 0.03289795) / 0.003158116   # +0.1%  z_dr_0p2_0p4 < 0.1937 and absphi_9 > 0.0329
        - 0.0005744533 * max(0.0, 0.03787151 - Q.M3) * max(0.0, Q.soft2_dr - 0.1573586) / 0.0009567484   # -0.1%  M3 < 0.03787 and soft2_dr > 0.1574
        - 0.0004031349 * max(0.0, Q.n_pt_above_10 - 31.0) / 0.1555479   # -0.0%  n_pt_above_10 > 31
        + 0.0003308854 * max(0.0, Q.max_dr - 0.4357228) / 0.007711928   # +0.0%  max_dr > 0.4357
        + 0.0001528208 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.pair_mass_0_10 - 1.569656) / 31.11071   # +0.0%  sum_pt_top20 > 1129 and pair_mass_0_10 > 1.57
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 9.184;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.184063 * (0.1318348
        - 0.1892035 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -18.9%  mass < 101
        + 0.1417868 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +14.2%  mass < 92.86
        - 0.130229 * max(0.0, 120.6 - Q.mass) / 38.8279   # -13.0%  mass < 120.6
        + 0.06395603 * max(0.0, 172.8 - Q.mass_top50) / 85.49388   # +6.4%  mass_top50 < 172.8
        + 0.04350267 * max(0.0, 86.4 - Q.mass) / 13.67632   # +4.4%  mass < 86.4
        - 0.03619417 * max(0.0, 0.01807679 - Q.girth2_top30) / 0.01083719   # -3.6%  girth2_top30 < 0.01808
        + 0.03162336 * max(0.0, Q.girth2_top30 - 0.008376291) / 0.003092781   # +3.2%  girth2_top30 > 0.008376
        + 0.02602958 * max(0.0, 91.19 - Q.mass_top15) / 34.88803   # +2.6%  mass_top15 < 91.19
        + 0.02390527 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # +2.4%  mass_top50 < 71.8
        - 0.02195541 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2) / 0.04240098   # -2.2%  mass_over_sum_pt > 0.1182 and D2_b2 < 7.366
        + 0.02168852 * max(0.0, 0.003687605 - Q.lam2) / 0.002601874   # +2.2%  lam2 < 0.003688
        - 0.02035592 * max(0.0, Q.mass_over_sum_pt - 0.09795415) / 0.01222897   # -2.0%  mass_over_sum_pt > 0.09795
        + 0.01911233 * max(0.0, 0.003811746 - Q.lam1) / 0.0006796118   # +1.9%  lam1 < 0.003812
        - 0.01880908 * max(0.0, 0.004573744 - Q.girth2_top50) / 0.0007739713   # -1.9%  girth2_top50 < 0.004574
        + 0.01877743 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +1.9%  e3 < 0.0003372
        + 0.01834935 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +1.8%  tau1 < 0.06311
        - 0.016072 * max(0.0, Q.e2 - 0.02793599) / 0.008664879   # -1.6%  e2 > 0.02794
        + 0.01601497 * max(0.0, Q.girth2_top20 - 0.008031209) / 0.002906852   # +1.6%  girth2_top20 > 0.008031
        + 0.01275303 * max(0.0, 0.00375223 - Q.girth2_top30) / 0.0006714094   # +1.3%  girth2_top30 < 0.003752
        - 0.01263677 * max(0.0, 0.1011005 - Q.M2) / 0.03683604   # -1.3%  M2 < 0.1011
        - 0.01038689 * max(0.0, Q.girth2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529) / 8.566262e-05   # -1.0%  girth2_top20 > 0.008031 and C2_b2 > 0.0008188
        - 0.00830595 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # -0.8%  log_sum_pt < 7.017
        - 0.007129443 * max(0.0, Q.sj3_pair_mass_min - 29.00832) / 6.839984   # -0.7%  sj3_pair_mass_min > 29.01
        - 0.006521838 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, Q.dr_max_012 - 0.004415714) / 1.536542e-05   # -0.7%  e3 < 0.0003372 and dr_max_012 > 0.004416
        - 0.006357195 * max(0.0, Q.pt_14 - 10.78125) / 5.796532   # -0.6%  pt_14 > 10.78
        - 0.006266764 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722) / 0.001787472   # -0.6%  girth2_top30 > 0.008376 and D2_b2 > 1.677
        + 0.006144086 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt) / 683.4113   # +0.6%  mass < 120.6 and sum_pt < 1008
        + 0.005998708 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) / 1.539217   # +0.6%  n_dr_0p2_0p4 > 15
        + 0.005716844 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +0.6%  z_dr_0p1_0p2 < 0.1203
        - 0.004931375 * max(0.0, 91.19 - Q.mass_top15) * max(0.0, 0.04008677 - Q.C2_b2) / 0.9324124   # -0.5%  mass_top15 < 91.19 and C2_b2 < 0.04009
        - 0.004685593 * max(0.0, Q.sj3_dr_min - 0.1204829) / 0.02211875   # -0.5%  sj3_dr_min > 0.1205
        - 0.003896798 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -0.4%  mass_over_sum_pt > 0.1182
        - 0.00377162 * max(0.0, 0.00242543 - Q.girth2_top50) / 0.0002374304   # -0.4%  girth2_top50 < 0.002425
        + 0.003602076 * max(0.0, 0.9300465 - Q.z_top40_slots) / 0.002595351   # +0.4%  z_top40_slots < 0.93
        - 0.003275661 * max(0.0, Q.n_dr_0_0p05 - 9.0) / 5.593776   # -0.3%  n_dr_0_0p05 > 9
        - 0.002685877 * max(0.0, 0.001592178 - Q.girth2_top3) / 0.000513961   # -0.3%  girth2_top3 < 0.001592
        - 0.002440549 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.2%  log_sum_pt < 6.811
        + 0.002384404 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832) / 8.191828e-05   # +0.2%  mass_over_sum_pt > 0.1182 and C2_b2 > 0.02705
        + 0.002258336 * max(0.0, Q.e2 - 0.05557149) * max(0.0, 1.67722 - Q.D2_b2) / 0.000547635   # +0.2%  e2 > 0.05557 and D2_b2 < 1.677
        + 0.002115049 * max(0.0, Q.sj3_dr_min - 0.1204829) * max(0.0, 33.78125 - Q.pt_11) / 0.2695668   # +0.2%  sj3_dr_min > 0.1205 and pt_11 < 33.78
        + 0.002074762 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +0.2%  LHA > 0.372
        + 0.002032302 * max(0.0, 0.002197765 - Q.girth2_top15) / 0.0004509993   # +0.2%  girth2_top15 < 0.002198
        + 0.001820231 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.z_dr_0p05_0p1 - 0.6469679) / 1.118691e-05   # +0.2%  girth2_top30 > 0.008376 and z_dr_0p05_0p1 > 0.647
        + 0.001807157 * max(0.0, Q.sj3_pair_mass_min - 29.00832) * max(0.0, Q.psi_0p3 - 0.9896594) / 0.01664863   # +0.2%  sj3_pair_mass_min > 29.01 and psi_0p3 > 0.9897
        - 0.00151685 * max(0.0, Q.girth2_top30 - 0.02809026) / 0.0001993235   # -0.2%  girth2_top30 > 0.02809
        + 0.001436467 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # +0.1%  e2 > 0.05557
        - 0.001338369 * max(0.0, Q.sj3_pair_mass_min - 76.60223) / 0.3297656   # -0.1%  sj3_pair_mass_min > 76.6
        + 0.001274576 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, 43.0 - Q.n_real_top50) / 0.0007262366   # +0.1%  girth2_top30 > 0.008376 and n_real_top50 < 43
        + 0.0009184176 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707) / 2.369749e-05   # +0.1%  e2 > 0.05557 and zdr_0 > 0.00137
        + 0.0006364401 * max(0.0, Q.sj3_dr_min - 0.1204829) * max(0.0, 7.863702 - Q.sj3_mass2) / 0.02256917   # +0.1%  sj3_dr_min > 0.1205 and sj3_mass2 < 7.864
        + 0.0006301775 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.z_2 - 0.120204) / 8.801305e-07   # +0.1%  e2 > 0.05557 and z_2 > 0.1202
        - 0.0005906465 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, Q.max_pair_mass - 33.3761) / 3.25809   # -0.1%  n_dr_0p2_0p4 > 15 and max_pair_mass > 33.38
        + 0.0005633109 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) / 0.293679   # +0.1%  n_dr_0p1_0p2 > 33
        - 0.0004362434 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.orientation_deg - 26.6454) / 0.03454067   # -0.0%  girth2_top30 > 0.008376 and orientation_deg > 26.65
        - 0.0003887255 * max(0.0, Q.sj3_mass1 - 21.11128) / 1.638992   # -0.0%  sj3_mass1 > 21.11
        + 0.0003313473 * max(0.0, 19.46875 - Q.pt_6) / 0.2515892   # +0.0%  pt_6 < 19.47
        - 0.0001621644 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) * max(0.0, Q.ptdr0_10 - 6.130737) / 0.3325374   # -0.0%  n_dr_0p1_0p2 > 33 and ptdr0_10 > 6.131
        - 8.786395e-05 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) * max(0.0, 0.12854 - Q.absphi_2) / 0.01260196   # -0.0%  n_dr_0p1_0p2 > 33 and absphi_2 < 0.1285
        - 6.392037e-05 * max(0.0, 0.1011005 - Q.M2) * max(0.0, -0.07794189 - Q.eta_0) / 6.491626e-05   # -0.0%  M2 < 0.1011 and eta_0 < -0.07794
        - 5.976497e-05 * max(0.0, 0.04704395 - Q.C2) / 0.005827731   # -0.0%  C2 < 0.04704
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 38.88;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 38.88161 * (0.003552366
        + 0.136492 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5337743   # +13.6%  mass < 91.19 and psi_0p3 > 0.9638
        - 0.1347632 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.3075308   # -13.5%  mass < 91.19 and psi_0p3 > 0.9777
        - 0.1281248 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5700006   # -12.8%  mass < 92.86 and psi_0p3 > 0.9638
        + 0.08861592 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.4385899   # +8.9%  mass < 101 and psi_0p3 > 0.9777
        + 0.05060468 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.2231625   # +5.1%  mass < 82.85 and psi_0p3 > 0.9777
        - 0.04004362 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -4.0%  mass < 101
        - 0.0386211 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.01143143   # -3.9%  mass < 91.19 and psi_0p3 > 0.998
        - 0.03485378 * max(0.0, 0.008190222 - Q.girth2) / 0.002454223   # -3.5%  girth2 < 0.00819
        + 0.03043141 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # +3.0%  girth2 < 0.009615
        + 0.02971228 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.01745475   # +3.0%  mass < 101 and psi_0p3 > 0.998
        + 0.02926419 * max(0.0, 120.6 - Q.mass) / 38.8279   # +2.9%  mass < 120.6
        - 0.02827336 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -2.8%  tau1 < 0.1073
        - 0.0276136 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -2.8%  mass < 82.85
        + 0.01966202 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +2.0%  mass_over_sum_pt < 0.1182
        + 0.01847727 * max(0.0, 0.09591084 - Q.tau1) / 0.02629125   # +1.8%  tau1 < 0.09591
        - 0.01744652 * max(0.0, 86.4 - Q.sd_mass) / 35.84633   # -1.7%  sd_mass < 86.4
        - 0.01574551 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -1.6%  lam1 < 0.008242
        + 0.01431398 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +1.4%  mass < 78.26
        + 0.01295382 * max(0.0, 69.65633 - Q.sd_mass) / 24.56034   # +1.3%  sd_mass < 69.66
        + 0.01226602 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.007828415   # +1.2%  mass < 82.85 and psi_0p3 > 0.998
        + 0.01153708 * max(0.0, 91.19 - Q.mass) / 16.46423   # +1.2%  mass < 91.19
        + 0.009346187 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # +0.9%  girth2 < 0.007877
        + 0.009025828 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +0.9%  tau21_b2 < 0.2352
        + 0.008974446 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # +0.9%  lam1 < 0.00619
        + 0.00564041 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.6%  n_dr_0p2_0p4 < 6
        + 0.00534889 * max(0.0, Q.mass_top20 - 73.35236) / 11.28399   # +0.5%  mass_top20 > 73.35
        - 0.004629737 * max(0.0, Q.sd_mass - 45.595) / 24.70213   # -0.5%  sd_mass > 45.59
        + 0.004044726 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # +0.4%  tau1 < 0.08787
        + 0.003649335 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +0.4%  mass < 92.86
        - 0.003573811 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.4%  psi_0p3 > 0.998
        - 0.003384516 * max(0.0, Q.mass_top20 - 85.79457) / 7.06893   # -0.3%  mass_top20 > 85.79
        - 0.002775113 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt) / 11.66143   # -0.3%  tau21_b2 < 0.2352 and sum_pt < 1261
        - 0.002555077 * max(0.0, 0.006403325 - Q.girth2) / 0.001406912   # -0.3%  girth2 < 0.006403
        - 0.002272226 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003362293   # -0.2%  tau21_b2 < 0.2352 and psi_0p3 > 0.9299
        + 0.00218565 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.2%  psi_0p3 > 0.9974
        - 0.002061247 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3) / 6.520272e-05   # -0.2%  n_dr_0p2_0p4 < 6 and e3 < 7.876e-05
        - 0.001400068 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.4947602 - Q.z_dr_0_0p05) / 0.572942   # -0.1%  mass < 92.86 and z_dr_0_0p05 < 0.4948
        - 0.001045351 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -0.1%  tau21 < 0.3862
        - 0.00104257 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 5.106465e-05   # -0.1%  tau21_b2 < 0.2352 and mass_over_sum_pt_sq < 0.007873
        + 0.0009912551 * max(0.0, Q.sd_mass - 98.05743) / 4.276978   # +0.1%  sd_mass > 98.06
        + 0.0008567181 * max(0.0, 0.006403325 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.9980008) / 9.11248e-07   # +0.1%  girth2 < 0.006403 and psi_0p3 > 0.998
        - 0.0006945192 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -9.840088) / 1.453315   # -0.1%  tau21_b2 < 0.2352 and orientation_deg > -9.84
        + 0.0006775907 * max(0.0, 0.000404306 - Q.lam2) / 5.964336e-05   # +0.1%  lam2 < 0.0004043
        - 0.0005157874 * max(0.0, 91.19 - Q.mass) * max(0.0, 6.0 - Q.n_dr_0_0p05) / 1.647199   # -0.1%  mass < 91.19 and n_dr_0_0p05 < 6
        + 0.0004718135 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.n_real_top50 - 34.0) / 0.287919   # +0.0%  tau21_b2 < 0.2352 and n_real_top50 > 34
        - 0.0003921991 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.1063277 - Q.z_2) / 0.0009585458   # -0.0%  tau21_b2 < 0.2352 and z_2 < 0.1063
        - 0.0003447342 * max(0.0, 0.008241985 - Q.lam1) * max(0.0, Q.zdr_0 - 0.01564747) / 7.458275e-07   # -0.0%  lam1 < 0.008242 and zdr_0 > 0.01565
        - 0.0003389548 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.max_dr - 0.3618493) / 0.02505788   # -0.0%  n_dr_0p2_0p4 < 6 and max_dr > 0.3618
        + 0.0002878755 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.ptdr0_2 - 11.87288) / 2.425576   # +0.0%  n_dr_0p2_0p4 < 6 and ptdr0_2 > 11.87
        - 0.000271265 * max(0.0, 18.0 - Q.n_dr_0_0p05) / 7.289224   # -0.0%  n_dr_0_0p05 < 18
        + 0.0002495926 * max(0.0, Q.z_dr_0p05_0p1 - 0.6469679) / 0.02199777   # +0.0%  z_dr_0p05_0p1 > 0.647
        + 0.0002343426 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.1825048 - Q.sj2_dr) / 0.06647299   # +0.0%  n_dr_0p2_0p4 < 6 and sj2_dr < 0.1825
        + 0.0001931203 * max(0.0, 0.09573228 - Q.z_dr_0_0p05) / 0.02128126   # +0.0%  z_dr_0_0p05 < 0.09573
        - 0.0001869159 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.001696484   # -0.0%  tau21_b2 < 0.2352 and sj2_dr > 0.1938
        + 0.0001801161 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.n_dr_0p1_0p2 - 14.0) / 0.001240426   # +0.0%  psi_0p3 > 0.998 and n_dr_0p1_0p2 > 14
        - 0.0001388857 * max(0.0, 0.00363788 - Q.e2_sq) / 0.0004963098   # -0.0%  e2_sq < 0.003638
        + 0.0001113374 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.02947847   # +0.0%  mass < 82.85 and psi_0p3 < 0.9985
        - 3.322103e-05 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.eta_1 - 0.0001021922) / 0.02682082   # -0.0%  n_dr_0p2_0p4 < 6 and eta_1 > 0.0001022
        + 2.992071e-05 * max(0.0, 0.09573228 - Q.z_dr_0_0p05) * max(0.0, 5.541989 - Q.sj3_mass3) / 0.03328332   # +0.0%  z_dr_0_0p05 < 0.09573 and sj3_mass3 < 5.542
        + 2.853694e-05 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 15.0) / 1.394886   # +0.0%  n_dr_0p2_0p4 < 6 and n_dr_0p1_0p2 > 15
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 16.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.66831 * (0.1332321
        - 0.1217235 * max(0.0, 0.009614971 - Q.width) / 0.003502545   # -12.2%  width < 0.009615
        + 0.05326072 * max(0.0, Q.mass - 87.36377) / 16.57741   # +5.3%  mass > 87.36
        - 0.0471364 * max(0.0, Q.n_for_90pct - 7.0) / 13.93571   # -4.7%  n_for_90pct > 7
        + 0.04425195 * max(0.0, Q.mass - 64.48544) / 31.19164   # +4.4%  mass > 64.49
        - 0.04397583 * max(0.0, 39.0 - Q.n_for_90pct) / 18.29776   # -4.4%  n_for_90pct < 39
        + 0.04233758 * max(0.0, Q.mass - 74.25181) / 24.07648   # +4.2%  mass > 74.25
        + 0.04001972 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # +4.0%  girth2 < 0.007877
        + 0.03953649 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt) / 2.336698   # +4.0%  mass_over_sum_pt > 0.07697 and sum_pt < 1116
        - 0.03588039 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -3.6%  mass_top50 > 82.04
        + 0.03470275 * max(0.0, Q.girth2_top20 - 0.008031209) / 0.002906852   # +3.5%  girth2_top20 > 0.008031
        - 0.03306412 * max(0.0, Q.mass - 101.0497) / 12.31084   # -3.3%  mass > 101
        + 0.02823042 * max(0.0, 1028.184 - Q.sum_pt) / 26.95739   # +2.8%  sum_pt < 1028
        - 0.02762243 * max(0.0, Q.girth2_top20 - 0.006043209) / 0.003593885   # -2.8%  girth2_top20 > 0.006043
        - 0.02411664 * max(0.0, 0.02146578 - Q.girth2_top15) / 0.01470441   # -2.4%  girth2_top15 < 0.02147
        + 0.02408394 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +2.4%  girth2_top40 < 0.008841
        + 0.023681 * max(0.0, 0.007463985 - Q.girth2_top30) / 0.00233708   # +2.4%  girth2_top30 < 0.007464
        - 0.02239669 * max(0.0, Q.mass - 125.1) / 7.098976   # -2.2%  mass > 125.1
        - 0.02197003 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt) / 0.0005328922   # -2.2%  girth2_top40 > 0.005197 and log_sum_pt < 7.017
        + 0.02044665 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +2.0%  sum_pt < 1017
        - 0.01991527 * max(0.0, 6.930088 - Q.log_sum_pt) / 0.02522479   # -2.0%  log_sum_pt < 6.93
        + 0.0187322 * max(0.0, Q.girth2_top40 - 0.005196966) / 0.004725809   # +1.9%  girth2_top40 > 0.005197
        - 0.01851656 * max(0.0, Q.z_top50_slots - 0.9704436) / 0.02331682   # -1.9%  z_top50_slots > 0.9704
        + 0.01786082 * max(0.0, Q.mass_top50 - 117.0487) / 7.705579   # +1.8%  mass_top50 > 117
        - 0.01671266 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # -1.7%  sum_pt_top40 < 1025
        - 0.01611588 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -1.6%  n_dr_0p2_0p4 < 15
        - 0.0155055 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # -1.6%  mass_over_sum_pt > 0.07697
        + 0.01521966 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +1.5%  lam1 < 0.007671
        + 0.01497286 * max(0.0, 1001.523 - Q.sum_pt_top40) / 26.72886   # +1.5%  sum_pt_top40 < 1002
        - 0.01151123 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # -1.2%  lam1 < 0.01174
        + 0.0106472 * max(0.0, Q.n_dr_0p2_0p4 - 3.0) / 6.482203   # +1.1%  n_dr_0p2_0p4 > 3
        - 0.008618953 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -0.9%  psi_0p3 > 0.9943
        + 0.008419134 * max(0.0, 9.0 - Q.n_dr_0p2_0p4) / 3.177318   # +0.8%  n_dr_0p2_0p4 < 9
        - 0.007931082 * max(0.0, Q.e3 - 5.13841e-05) / 6.821576e-05   # -0.8%  e3 > 5.138e-05
        - 0.007750406 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -0.8%  mass_over_sum_pt > 0.1182
        + 0.007235155 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.6773544   # +0.7%  sum_pt < 1028 and dr_max_012 > 0.1828
        - 0.006582922 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.5693287   # -0.7%  sum_pt < 1017 and dr_max_012 > 0.1828
        + 0.005716455 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # +0.6%  C2 > 0.06656
        + 0.00479204 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # +0.5%  tau2 < 0.0795
        + 0.00428249 * max(0.0, 1.219727 - Q.soft5_pt) / 0.2575   # +0.4%  soft5_pt < 1.22
        + 0.004275912 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.4%  e2 > 0.04755
        + 0.003934283 * max(0.0, 0.0795038 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.38202) / 0.1361121   # +0.4%  tau2 < 0.0795 and sj3_mass1 > 13.38
        + 0.003541685 * max(0.0, Q.n_particles - 51.0) / 3.973704   # +0.4%  n_particles > 51
        - 0.002697552 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, Q.sj2_dr - 0.1512157) / 6.78729e-05   # -0.3%  girth2_top30 < 0.007464 and sj2_dr > 0.1512
        + 0.00259604 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.3%  sj2_dr > 0.2232
        + 0.0024797 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.03720487   # +0.2%  z_dr_0p1_0p2 > 0.334
        + 0.002304082 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # +0.2%  e3 > 0.0003372
        - 0.00188952 * max(0.0, Q.mass_top20 - 119.2969) / 1.8402   # -0.2%  mass_top20 > 119.3
        + 0.001707264 * max(0.0, Q.eta_0 - -0.01382675) / 0.02505234   # +0.2%  eta_0 > -0.01383
        - 0.001589175 * max(0.0, 0.3628388 - Q.psi_0p1) / 0.02326895   # -0.2%  psi_0p1 < 0.3628
        + 0.001555123 * max(0.0, 6.930088 - Q.log_sum_pt) * max(0.0, 0.05595395 - Q.dr01) / 0.0008519083   # +0.2%  log_sum_pt < 6.93 and dr01 < 0.05595
        + 0.001425668 * max(0.0, Q.n_dr_0p1_0p2 - 19.0) / 1.732133   # +0.1%  n_dr_0p1_0p2 > 19
        - 0.001267143 * max(0.0, 0.8316924 - Q.z_top15_slots) / 0.04702674   # -0.1%  z_top15_slots < 0.8317
        - 0.0008420374 * max(0.0, Q.n_particles - 51.0) * max(0.0, 1.091797 - Q.soft1_pt) / 1.364428   # -0.1%  n_particles > 51 and soft1_pt < 1.092
        + 0.0006385575 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.09686657   # +0.1%  n_dr_0p2_0p4 < 15 and z_dr_0p1_0p2 > 0.334
        - 0.0006024717 * max(0.0, Q.mass - 64.48544) * max(0.0, Q.zdr_0 - 0.0008718296) / 0.4297651   # -0.1%  mass > 64.49 and zdr_0 > 0.0008718
        - 0.0004605566 * max(0.0, Q.mass_top20 - 119.2969) * max(0.0, Q.absphi_1 - 0.07141113) / 0.06373091   # -0.0%  mass_top20 > 119.3 and absphi_1 > 0.07141
        - 0.0002883346 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # -0.0%  log_sum_pt > 7.139
        + 0.0002161892 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, 13.0 - Q.n_for_90pct) / 15.66746   # +0.0%  sum_pt < 1017 and n_for_90pct < 13
        + 0.0001280523 * max(0.0, Q.girth2_top20 - 0.006043209) * max(0.0, 0.08173536 - Q.sj3_z2) / 1.545259e-07   # +0.0%  girth2_top20 > 0.006043 and sj3_z2 < 0.08174
        + 8.494354e-05 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # +0.0%  e2 > 0.06524
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 43.82;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 43.81865 * (-0.02125861
        - 0.2562291 * max(0.0, 160.8 - Q.mass) / 72.78344   # -25.6%  mass < 160.8
        + 0.2244055 * max(0.0, 162.8363 - Q.mass) / 74.60496   # +22.4%  mass < 162.8
        + 0.1569187 * max(0.0, 172.8 - Q.mass) / 83.78792   # +15.7%  mass < 172.8
        - 0.1246837 * max(0.0, 143.7876 - Q.mass) / 57.99337   # -12.5%  mass < 143.8
        + 0.06315117 * max(0.0, 138.8977 - Q.mass_top50) / 55.01559   # +6.3%  mass_top50 < 138.9
        - 0.04247595 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -4.2%  mass_top40 < 163.3
        + 0.02761639 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +2.8%  mass < 92.86
        - 0.01737588 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # -1.7%  mass_top50 < 92.17
        + 0.01526961 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +1.5%  mass < 82.85
        + 0.006938521 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +0.7%  mass_top40 < 80.89
        + 0.005465643 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # +0.5%  girth2_top40 < 0.00626
        - 0.004894911 * max(0.0, Q.girth2_top30 - 0.0008564881) / 0.007661497   # -0.5%  girth2_top30 > 0.0008565
        + 0.004748165 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +0.5%  sum_pt < 986.1
        + 0.004613606 * max(0.0, 125.1 - Q.mass_top40) / 45.33615   # +0.5%  mass_top40 < 125.1
        - 0.004360319 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -0.4%  mass < 64.49
        - 0.004103251 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -0.4%  log_sum_pt < 6.903
        + 0.003722168 * max(0.0, 73.33139 - Q.mass_top30) / 11.37701   # +0.4%  mass_top30 < 73.33
        + 0.00371946 * max(0.0, Q.girth - 0.09749958) / 0.008315822   # +0.4%  girth > 0.0975
        + 0.003665117 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.4%  lam1 < 0.004673
        + 0.002824062 * max(0.0, 956.2133 - Q.sum_pt_top40) / 14.00451   # +0.3%  sum_pt_top40 < 956.2
        - 0.002695088 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -0.3%  tau1 < 0.05445
        + 0.002463782 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # +0.2%  psi_0p3 > 0.9943
        - 0.002453656 * max(0.0, Q.girth - 0.1402186) / 0.001871547   # -0.2%  girth > 0.1402
        - 0.002178044 * max(0.0, Q.z_top40_slots - 0.9574183) / 0.02831517   # -0.2%  z_top40_slots > 0.9574
        - 0.00180157 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt) / 447.38   # -0.2%  mass_top40 < 80.89 and sum_pt < 1035
        + 0.001422204 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +0.1%  LHA > 0.4042
        + 0.001350765 * max(0.0, 125.1 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 42.49581   # +0.1%  mass_top40 < 125.1 and n_dr_0p2_0p4 > 9
        + 0.001267922 * max(0.0, Q.D2 - 2.178951) / 1.450771   # +0.1%  D2 > 2.179
        + 0.001137346 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +0.1%  lam1 > 0.01174
        + 0.0009026584 * max(0.0, Q.z_top5 - 0.7963975) / 0.006402524   # +0.1%  z_top5 > 0.7964
        - 0.0008405447 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, 0.001776308 - Q.lam2) / 0.005060516   # -0.1%  sum_pt_top40 < 956.2 and lam2 < 0.001776
        - 0.0007022754 * max(0.0, 0.00217213 - Q.girth2_top40) / 0.0002075259   # -0.1%  girth2_top40 < 0.002172
        - 0.0006854518 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.z_top50_slots - 0.9995789) / 0.005945365   # -0.1%  mass < 92.86 and z_top50_slots > 0.9996
        - 0.0005574609 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258) / 5.834847   # -0.1%  sum_pt_top40 < 956.2 and soft4_pt > 1.789
        + 0.0005364078 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40) / 104.3982   # +0.1%  mass_top40 < 80.89 and sum_pt_top40 < 906.6
        - 0.0005070496 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -0.1%  z_dr_0_0p05 > 0.8789
        + 0.0004972133 * max(0.0, Q.n_for_90pct - 35.0) / 0.4575429   # +0.0%  n_for_90pct > 35
        + 0.0003251923 * max(0.0, 986.0565 - Q.sum_pt) * max(0.0, Q.soft5_z - 0.002036949) / 0.003199714   # +0.0%  sum_pt < 986.1 and soft5_z > 0.002037
        - 0.000320494 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) / 0.006411021   # -0.0%  z_dr_0p1_0p2 > 0.6882
        - 5.831573e-05 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # -0.0%  mass_over_sum_pt > 0.1709
        + 3.296332e-05 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # +0.0%  e3 > 0.0003372
        + 2.69876e-05 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, 154.25 - Q.pt_2) / 0.4838258   # +0.0%  z_dr_0p1_0p2 > 0.6882 and pt_2 < 154.2
        - 2.432065e-05 * max(0.0, Q.sj3_dr13 - 0.1654269) / 0.05535154   # -0.0%  sj3_dr13 > 0.1654
        + 2.071375e-05 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047) / 1.704178   # +0.0%  sum_pt_top40 < 956.2 and soft3_pt > 2.873
        - 1.025594e-05 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.psi_0p1 - 0.9184255) / 0.111142   # -0.0%  sum_pt_top40 < 956.2 and psi_0p1 > 0.9184
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 12.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.58629 * (0.0109899
        + 0.09944249 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +9.9%  mass < 89.74
        + 0.09895121 * max(0.0, Q.mass - 78.26182) / 21.33658   # +9.9%  mass > 78.26
        - 0.08092285 * max(0.0, Q.mass - 64.48544) / 31.19164   # -8.1%  mass > 64.49
        + 0.07883034 * max(0.0, Q.e2 - 0.01256572) / 0.01912291   # +7.9%  e2 > 0.01257
        - 0.06521341 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -6.5%  mass < 101
        + 0.05110589 * max(0.0, 0.01655983 - Q.girth2_top20) / 0.01003486   # +5.1%  girth2_top20 < 0.01656
        - 0.0397368 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -4.0%  e2 < 0.04359
        - 0.03544576 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -3.5%  girth < 0.1207
        - 0.02832644 * max(0.0, Q.mass - 143.7876) / 3.946979   # -2.8%  mass > 143.8
        + 0.02792006 * max(0.0, 0.5760704 - Q.z_top2_slots) / 0.2302819   # +2.8%  z_top2_slots < 0.5761
        + 0.02333982 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +2.3%  mass_top50 > 136.8
        + 0.02238483 * max(0.0, 935.1043 - Q.sum_pt_top15) / 95.709   # +2.2%  sum_pt_top15 < 935.1
        + 0.02059816 * max(0.0, 2.975532 - Q.D2) / 0.9290697   # +2.1%  D2 < 2.976
        - 0.02048434 * max(0.0, 65.20727 - Q.sj2_mass1) / 36.50254   # -2.0%  sj2_mass1 < 65.21
        - 0.01960368 * max(0.0, 59.40777 - Q.mass_top5) / 33.74882   # -2.0%  mass_top5 < 59.41
        + 0.01843895 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.8%  z_dr_0p2_0p4 < 0.09123
        - 0.01823268 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -1.8%  n_dr_0p2_0p4 < 11
        + 0.01703756 * max(0.0, 0.0001086251 - Q.e3) / 5.924919e-05   # +1.7%  e3 < 0.0001086
        - 0.01673464 * max(0.0, 72.18744 - Q.mass_top15) / 20.20734   # -1.7%  mass_top15 < 72.19
        - 0.01506709 * max(0.0, 0.01976735 - Q.girth2_top10) / 0.01371247   # -1.5%  girth2_top10 < 0.01977
        - 0.01451106 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, 0.4357228 - Q.max_dr) / 0.001259784   # -1.5%  girth2_top10 < 0.01977 and max_dr < 0.4357
        - 0.01420813 * max(0.0, 86.4 - Q.mass) / 13.67632   # -1.4%  mass < 86.4
        - 0.01381564 * max(0.0, Q.mass - 162.8363) / 1.509832   # -1.4%  mass > 162.8
        - 0.0135815 * max(0.0, Q.z_dr_0_0p05 - 0.7674734) / 0.05221088   # -1.4%  z_dr_0_0p05 > 0.7675
        - 0.01320978 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.9985421) / 6.34706e-06   # -1.3%  girth2_top10 < 0.01977 and psi_0p3 > 0.9985
        - 0.01172974 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # -1.2%  girth2_top30 < 0.005402
        + 0.01098259 * max(0.0, 59.40777 - Q.mass_top5) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1) / 22.61156   # +1.1%  mass_top5 < 59.41 and z_dr_0p05_0p1 < 0.8509
        + 0.0106666 * max(0.0, 0.5760704 - Q.z_top2_slots) * max(0.0, 0.002970691 - Q.soft9_z) / 0.0002253038   # +1.1%  z_top2_slots < 0.5761 and soft9_z < 0.002971
        - 0.0101878 * max(0.0, Q.mass_top30 - 78.53034) / 14.31024   # -1.0%  mass_top30 > 78.53
        + 0.008622851 * max(0.0, Q.mass_top5 - 22.18342) / 12.67741   # +0.9%  mass_top5 > 22.18
        + 0.00790482 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +0.8%  dr_0 < 0.06413
        - 0.005604428 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.6%  sum_pt_top10 > 943.7
        - 0.005489958 * max(0.0, Q.mass_over_sum_pt - 0.1606361) / 0.001344467   # -0.5%  mass_over_sum_pt > 0.1606
        + 0.005435455 * max(0.0, Q.sj3_dr_min - 0.1204829) / 0.02211875   # +0.5%  sj3_dr_min > 0.1205
        + 0.005422299 * max(0.0, Q.n_particles - 41.0) / 8.920024   # +0.5%  n_particles > 41
        - 0.005397264 * max(0.0, 0.002575211 - Q.girth2) / 0.0002582872   # -0.5%  girth2 < 0.002575
        - 0.004759975 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # -0.5%  sum_pt_top50 < 959.1
        - 0.004166991 * max(0.0, Q.mass_top50 - 136.785) * max(0.0, Q.soft5_z - 0.001594761) / 0.001639485   # -0.4%  mass_top50 > 136.8 and soft5_z > 0.001595
        + 0.004135145 * max(0.0, 0.008824206 - Q.zdr_1) / 0.00383206   # +0.4%  zdr_1 < 0.008824
        + 0.003965496 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.pt1_dr01 - 5.351077) / 32.20836   # +0.4%  mass < 101 and pt1_dr01 > 5.351
        - 0.00372798 * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0004356739   # -0.4%  psi_0p3 > 0.9985
        + 0.003594539 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897) / 0.001013644   # +0.4%  mass > 162.8 and soft5_z > 0.001435
        - 0.003304504 * max(0.0, 2.975532 - Q.D2) * max(0.0, Q.sj2_dr - 0.2070855) / 0.0198522   # -0.3%  D2 < 2.976 and sj2_dr > 0.2071
        - 0.002212954 * max(0.0, 0.09021906 - Q.z_2nd) / 0.004040099   # -0.2%  z_2nd < 0.09022
        + 0.002204642 * max(0.0, Q.mass_over_sum_pt - 0.1606361) * max(0.0, 0.09696199 - Q.z_3) / 5.287414e-05   # +0.2%  mass_over_sum_pt > 0.1606 and z_3 < 0.09696
        + 0.002178554 * max(0.0, Q.mass_top50 - 172.8) / 0.5062389   # +0.2%  mass_top50 > 172.8
        - 0.001694057 * max(0.0, 750.7313 - Q.sum_pt_top20) / 7.067235   # -0.2%  sum_pt_top20 < 750.7
        + 0.001648371 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, Q.psi_0p3 - 0.9985421) / 4.280363e-07   # +0.2%  girth2_top30 < 0.005402 and psi_0p3 > 0.9985
        - 0.00154531 * max(0.0, Q.e2 - 0.03263075) / 0.006288927   # -0.2%  e2 > 0.03263
        + 0.001102284 * max(0.0, Q.mass_top50 - 160.8) / 1.246461   # +0.1%  mass_top50 > 160.8
        - 0.001054794 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, 53.4375 - Q.pt_7) / 18.94242   # -0.1%  mass_top50 > 160.8 and pt_7 < 53.44
        + 0.001026844 * max(0.0, Q.sum_pt_top10 - 943.6922) * max(0.0, 0.6469679 - Q.z_dr_0p05_0p1) / 6.713086   # +0.1%  sum_pt_top10 > 943.7 and z_dr_0p05_0p1 < 0.647
        - 0.0009905028 * max(0.0, Q.e2 - 0.03263075) * max(0.0, Q.soft2_pt - 1.413232) / 0.001032157   # -0.1%  e2 > 0.03263 and soft2_pt > 1.413
        - 0.0006560311 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109) / 0.0004735247   # -0.1%  mass_top50 > 160.8 and soft4_z > 0.001721
        + 0.0003879393 * max(0.0, Q.psi_0p3 - 0.9985421) * max(0.0, Q.soft5_z - 0.001434897) / 3.081682e-07   # +0.0%  psi_0p3 > 0.9985 and soft5_z > 0.001435
        - 0.0003118918 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, 0.8476928 - Q.tau32) / 0.001347663   # -0.0%  girth2_top10 < 0.01977 and tau32 < 0.8477
        - 0.0002632723 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.absphi_4 - 0.004672432) / 0.4741157   # -0.0%  sum_pt_top50 < 959.1 and absphi_4 > 0.004672
        + 0.0002399666 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, 0.2748443 - Q.dr0_3) / 1.93979   # +0.0%  sum_pt_top50 < 959.1 and dr0_3 < 0.2748
        - 0.000125318 * max(0.0, 2.975532 - Q.D2) * max(0.0, Q.dr_12 - 0.1269782) / 0.01318085   # -0.0%  D2 < 2.976 and dr_12 > 0.127
        + 0.0001157253 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, 631.275 - Q.sum_pt_top5) / 0.06578377   # +0.0%  girth2_top30 < 0.005402 and sum_pt_top5 < 631.3
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 7.976;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.976189 * (-0.009147784
        + 0.1634623 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +16.3%  mass < 92.86
        - 0.09416144 * max(0.0, 0.076787 - Q.girth) / 0.02105267   # -9.4%  girth < 0.07679
        + 0.08615992 * max(0.0, 0.08589404 - Q.girth) / 0.02745918   # +8.6%  girth < 0.08589
        - 0.08227903 * max(0.0, 80.4 - Q.mass) / 10.6806   # -8.2%  mass < 80.4
        + 0.04728019 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +4.7%  e2_sq < 0.008184
        - 0.04338146 * max(0.0, 0.006929741 - Q.girth2_top30) / 0.002004687   # -4.3%  girth2_top30 < 0.00693
        - 0.03754731 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -3.8%  mass_top40 < 83.33
        + 0.0294674 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 43.23581   # +2.9%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21
        - 0.02822131 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # -2.8%  e2 < 0.03876
        + 0.02738401 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +2.7%  e2 < 0.02516
        - 0.02583468 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2) / 0.005195774   # -2.6%  n_dr_0p2_0p4 < 10 and girth2 < 0.005532
        - 0.02489183 * max(0.0, 80.35535 - Q.mass_top50) / 11.1399   # -2.5%  mass_top50 < 80.36
        - 0.0232915 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -2.3%  lam1 < 0.006717
        - 0.02279129 * max(0.0, 0.00625621 - Q.girth2_top10) / 0.002474789   # -2.3%  girth2_top10 < 0.006256
        + 0.01910317 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +1.9%  e2 < 0.0303
        + 0.01831017 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +1.8%  n_dr_0p2_0p4 < 10
        + 0.0172515 * max(0.0, 0.00406126 - Q.girth2_top10) / 0.001298509   # +1.7%  girth2_top10 < 0.004061
        + 0.01616883 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +1.6%  mass_top40 < 67.73
        + 0.01458344 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +1.5%  mass < 101
        + 0.01452055 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # +1.5%  girth2_top30 < 0.005402
        - 0.0143445 * max(0.0, 0.06030419 - Q.mass_over_sum_pt) / 0.005383371   # -1.4%  mass_over_sum_pt < 0.0603
        - 0.01321596 * max(0.0, 3.793233e-05 - Q.e3) / 1.004564e-05   # -1.3%  e3 < 3.793e-05
        - 0.01046683 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_real_top40 - 26.0) / 33.17834   # -1.0%  n_dr_0p2_0p4 < 10 and n_real_top40 > 26
        + 0.01035856 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.pt_11 - 6.632422) / 55.73806   # +1.0%  n_dr_0p2_0p4 < 10 and pt_11 > 6.632
        + 0.01024106 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.3563114 - Q.planar_flow) / 0.8671033   # +1.0%  mass < 101 and planar_flow < 0.3563
        + 0.009125233 * max(0.0, Q.D2 - 2.178951) / 1.450771   # +0.9%  D2 > 2.179
        + 0.008015965 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # +0.8%  psi_0p3 > 0.998
        + 0.007949076 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) * max(0.0, Q.z_dr_0p1_0p2 - 0.1203437) / 6.825269e-05   # +0.8%  mass_over_sum_pt_sq < 0.009595 and z_dr_0p1_0p2 > 0.1203
        + 0.00655138 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # +0.7%  girth < 0.07374
        - 0.00592587 * max(0.0, 19.89956 - Q.sj2_mass1) / 2.423382   # -0.6%  sj2_mass1 < 19.9
        + 0.005865445 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +0.6%  mass_top30 < 60.44
        + 0.005719547 * max(0.0, 0.00130722 - Q.girth2_top10) / 0.0002794102   # +0.6%  girth2_top10 < 0.001307
        - 0.005263319 * max(0.0, Q.D2 - 5.378975) / 0.5670303   # -0.5%  D2 > 5.379
        - 0.005169128 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # -0.5%  mass_over_sum_pt_sq < 0.009595
        + 0.004380511 * max(0.0, Q.psi_0p2 - 0.9935324) / 0.001465572   # +0.4%  psi_0p2 > 0.9935
        - 0.004369438 * max(0.0, Q.C2_b2 - 0.01330402) / 0.00729146   # -0.4%  C2_b2 > 0.0133
        + 0.004183466 * max(0.0, 0.009962397 - Q.girth2_top15) / 0.004894699   # +0.4%  girth2_top15 < 0.009962
        + 0.003945662 * max(0.0, Q.z_top40_slots - 0.996191) / 0.001738831   # +0.4%  z_top40_slots > 0.9962
        + 0.003622403 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +0.4%  psi_0p3 > 0.9897
        + 0.003552733 * max(0.0, Q.z_top40_slots - 0.996191) * max(0.0, 0.006580753 - Q.C2_b2) / 4.241011e-06   # +0.4%  z_top40_slots > 0.9962 and C2_b2 < 0.006581
        - 0.003303863 * max(0.0, Q.psi_0p1 - 0.9770626) / 0.001598392   # -0.3%  psi_0p1 > 0.9771
        + 0.002843909 * max(0.0, Q.psi_0p2 - 0.9313699) * max(0.0, 0.08052407 - Q.dr_13) / 0.001040005   # +0.3%  psi_0p2 > 0.9314 and dr_13 < 0.08052
        - 0.002446089 * max(0.0, 101.0497 - Q.mass) * max(0.0, 891.875 - Q.sum_pt_top10) / 2090.674   # -0.2%  mass < 101 and sum_pt_top10 < 891.9
        - 0.002265998 * max(0.0, Q.psi_0p2 - 0.9313699) * max(0.0, 0.3897349 - Q.soft5_dr0) / 0.009147053   # -0.2%  psi_0p2 > 0.9314 and soft5_dr0 < 0.3897
        + 0.0020051 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.pt1_dr01 - 12.6865) / 16.11867   # +0.2%  mass < 101 and pt1_dr01 > 12.69
        + 0.00143728 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.pair_mass_0_12 - 7.607175) / 4.784519   # +0.1%  n_dr_0p2_0p4 < 10 and pair_mass_0_12 > 7.607
        + 0.001349909 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.absphi_7 - 0.05749512) / 0.03655045   # +0.1%  n_dr_0p2_0p4 < 10 and absphi_7 > 0.0575
        - 0.001246515 * max(0.0, 80.4 - Q.mass) * max(0.0, 0.8128995 - Q.pt1_over_pt0) / 2.534616   # -0.1%  mass < 80.4 and pt1_over_pt0 < 0.8129
        + 0.001218112 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0402832 - Q.phi_0) / 0.166757   # +0.1%  n_dr_0p2_0p4 < 10 and phi_0 < 0.04028
        + 0.00105768 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0) / 30.16986   # +0.1%  n_dr_0p2_0p4 < 10 and n_particles > 34
        - 0.0006359046 * max(0.0, Q.psi_0p2 - 0.9935324) * max(0.0, Q.dr_11 - 0.05788126) / 3.477997e-05   # -0.1%  psi_0p2 > 0.9935 and dr_11 > 0.05788
        + 0.0003514106 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.ptdr0_2 - 11.87288) / 8.86429   # +0.0%  mass < 101 and ptdr0_2 > 11.87
        - 0.0003132388 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.pair_mass_0_13 - 8.669189) / 2.917386   # -0.0%  n_dr_0p2_0p4 < 10 and pair_mass_0_13 > 8.669
        - 0.0002956819 * max(0.0, 0.08589404 - Q.girth) * max(0.0, Q.soft3_dr - 0.161871) / 0.001840993   # -0.0%  girth < 0.08589 and soft3_dr > 0.1619
        - 0.0002508278 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.dr_8 - 0.05067946) / 0.09994598   # -0.0%  n_dr_0p2_0p4 < 10 and dr_8 > 0.05068
        + 0.0002231604 * max(0.0, 19.89956 - Q.sj2_mass1) * max(0.0, 0.002968979 - Q.soft10_abseta) / 0.001635055   # +0.0%  sj2_mass1 < 19.9 and soft10_abseta < 0.002969
        + 0.0002230219 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # +0.0%  psi_0p2 > 0.9314
        - 8.644163e-05 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_6 - 0.02949408) / 0.04399604   # -0.0%  n_dr_0p2_0p4 < 10 and z_6 > 0.02949
        + 8.127959e-05 * max(0.0, 19.89956 - Q.sj2_mass1) * max(0.0, 0.1147522 - Q.dr_9) / 0.1209699   # +0.0%  sj2_mass1 < 19.9 and dr_9 < 0.1148
        - 7.214565e-06 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.pair_mass_0_10 - 7.661644) / 6.353801   # -0.0%  n_dr_0p2_0p4 < 10 and pair_mass_0_10 > 7.662
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 3.494;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.494335 * (-0.07328361
        + 0.2428504 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +24.3%  mass < 82.85
        + 0.1267881 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.04417088   # +12.7%  mass < 86.4 and lam2 < 0.003688
        + 0.1054907 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +10.5%  mass < 74.25
        - 0.06088798 * max(0.0, 86.4 - Q.mass) / 13.67632   # -6.1%  mass < 86.4
        - 0.05635555 * max(0.0, 62.55 - Q.mass) / 5.464058   # -5.6%  mass < 62.55
        - 0.04572278 * max(0.0, 0.06895248 - Q.mass_over_sum_pt) / 0.007729512   # -4.6%  mass_over_sum_pt < 0.06895
        - 0.04436295 * max(0.0, 74.78616 - Q.mass_top40) / 9.958349   # -4.4%  mass_top40 < 74.79
        - 0.03747236 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.5993597   # -3.7%  mass < 86.4 and z_dr_0p1_0p2 < 0.06473
        + 0.02807923 * max(0.0, 89.6788 - Q.mass_top40) / 17.47899   # +2.8%  mass_top40 < 89.68
        - 0.021922 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.03580539   # -2.2%  mass < 86.4 and psi_0p3 < 0.9985
        - 0.02156942 * max(0.0, 1.339844 - Q.soft5_pt) / 0.3268008   # -2.2%  soft5_pt < 1.34
        + 0.02074431 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, Q.n_dr_0p05_0p1 - 1.0) / 124.4066   # +2.1%  mass_top40 < 89.68 and n_dr_0p05_0p1 > 1
        - 0.01699418 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # -1.7%  psi_0p3 > 0.9974
        + 0.01278549 * max(0.0, 0.002752094 - Q.lam1) / 0.0003912817   # +1.3%  lam1 < 0.002752
        + 0.01162378 * max(0.0, Q.sd_mass - 125.1) / 1.231011   # +1.2%  sd_mass > 125.1
        - 0.01093004 * max(0.0, 0.006142802 - Q.girth2_top15) * max(0.0, Q.z_10 - 0.01143749) / 2.042492e-05   # -1.1%  girth2_top15 < 0.006143 and z_10 > 0.01144
        + 0.01037865 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.5100475 - Q.tau21) / 0.4909635   # +1.0%  mass < 86.4 and tau21 < 0.51
        + 0.009908697 * max(0.0, Q.psi_0p1 - 0.9538343) / 0.006310723   # +1.0%  psi_0p1 > 0.9538
        - 0.009227304 * max(0.0, 0.1464158 - Q.soft4_dr) / 0.02966654   # -0.9%  soft4_dr < 0.1464
        + 0.0075396 * max(0.0, Q.psi_0p1 - 0.9770626) / 0.001598392   # +0.8%  psi_0p1 > 0.9771
        + 0.007483608 * max(0.0, 0.006142802 - Q.girth2_top15) * max(0.0, 48.28125 - Q.pt_5) / 0.01386018   # +0.7%  girth2_top15 < 0.006143 and pt_5 < 48.28
        + 0.007182064 * max(0.0, 0.0007431905 - Q.girth2_top10) * max(0.0, 1069.671 - Q.sum_pt_top40) / 0.006568426   # +0.7%  girth2_top10 < 0.0007432 and sum_pt_top40 < 1070
        + 0.006689649 * max(0.0, 0.0007431905 - Q.girth2_top10) / 0.0001243965   # +0.7%  girth2_top10 < 0.0007432
        + 0.006228097 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min) / 54.09437   # +0.6%  sj3_pair_mass_max > 120.6 and sj3_pair_mass_min < 76.6
        - 0.00591083 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, 334.5 - Q.pt_0) / 1668.723   # -0.6%  mass_top40 < 89.68 and pt_0 < 334.5
        - 0.005809352 * max(0.0, 6.856375 - Q.log_sum_pt) / 0.008053459   # -0.6%  log_sum_pt < 6.856
        + 0.005160791 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926) / 0.5295314   # +0.5%  sj3_pair_mass_max > 120.6 and z_dr_0p1_0p2 > 0.2865
        + 0.00503615 * max(0.0, Q.mass - 160.8) / 1.724663   # +0.5%  mass > 160.8
        - 0.004907133 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.985099 - Q.z_top50_slots) / 0.008644046   # -0.5%  mass < 86.4 and z_top50_slots < 0.9851
        - 0.004863443 * max(0.0, 6.856375 - Q.log_sum_pt) * max(0.0, 0.002396991 - Q.lam2) / 7.289393e-06   # -0.5%  log_sum_pt < 6.856 and lam2 < 0.002397
        + 0.004787685 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 32.50209 - Q.sj3_mass1) / 22.26646   # +0.5%  sj3_pair_mass_max > 120.6 and sj3_mass1 < 32.5
        - 0.003895975 * max(0.0, 0.006142802 - Q.girth2_top15) / 0.002080651   # -0.4%  girth2_top15 < 0.006143
        - 0.003567086 * max(0.0, Q.sd_mass - 125.1) * max(0.0, Q.lam2 - 0.0001679609) / 0.005601117   # -0.4%  sd_mass > 125.1 and lam2 > 0.000168
        + 0.002844586 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) / 0.006411021   # +0.3%  z_dr_0p1_0p2 > 0.6882
        - 0.002721965 * max(0.0, Q.sd_mass - 125.1) * max(0.0, 0.4199841 - Q.sd_zg) / 0.1704664   # -0.3%  sd_mass > 125.1 and sd_zg < 0.42
        - 0.002304401 * max(0.0, 0.006142802 - Q.girth2_top15) * max(0.0, Q.sum_pt_top40 - 1225.842) / 0.02866211   # -0.2%  girth2_top15 < 0.006143 and sum_pt_top40 > 1226
        + 0.002256139 * max(0.0, Q.z_dr_0_0p05 - 0.9351131) / 0.004598373   # +0.2%  z_dr_0_0p05 > 0.9351
        - 0.002208153 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, 0.2502892 - Q.sj3_z3) / 0.0007552095   # -0.2%  z_dr_0p1_0p2 > 0.6882 and sj3_z3 < 0.2503
        - 0.001914354 * max(0.0, Q.sj3_pair_mass_max - 120.6) / 2.130243   # -0.2%  sj3_pair_mass_max > 120.6
        + 0.00181314 * max(0.0, Q.sd_mass - 125.1) * max(0.0, Q.girth2_top3 - 0.01002369) / 0.01693538   # +0.2%  sd_mass > 125.1 and girth2_top3 > 0.01002
        + 0.00162079 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 0.00104408 - Q.soft9_z) / 7.375548e-05   # +0.2%  sj3_pair_mass_max > 120.6 and soft9_z < 0.001044
        + 0.001590201 * max(0.0, 24.20196 - Q.mass_top15) / 1.791326   # +0.2%  mass_top15 < 24.2
        + 0.001467417 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, Q.n_dr_0p4_up - 0.0) / 0.001442994   # +0.1%  z_dr_0p1_0p2 > 0.6882 and n_dr_0p4_up > 0
        + 0.001378509 * max(0.0, 0.05077291 - Q.mass_over_sum_pt) / 0.003255623   # +0.1%  mass_over_sum_pt < 0.05077
        + 0.000992372 * max(0.0, Q.z_dr_0_0p05 - 0.9351131) * max(0.0, 15.0 - Q.n_real_top15) / 0.0004988097   # +0.1%  z_dr_0_0p05 > 0.9351 and n_real_top15 < 15
        + 0.000779374 * max(0.0, 86.4 - Q.mass) * max(0.0, Q.zdr_0 - 0.007860787) / 0.006171938   # +0.1%  mass < 86.4 and zdr_0 > 0.007861
        + 0.000744467 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.1873652 - Q.sj3_z2) / 1.419703e-05   # +0.1%  psi_0p3 > 0.9974 and sj3_z2 < 0.1874
        - 0.0005711139 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.orientation_deg - 35.6254) / 17.12462   # -0.1%  sj3_pair_mass_max > 120.6 and orientation_deg > 35.63
        - 0.0005631485 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, -62.72059 - Q.orientation_deg) / 4.424567   # -0.1%  sj3_pair_mass_max > 120.6 and orientation_deg < -62.72
        + 0.0003710797 * max(0.0, 0.0007431905 - Q.girth2_top10) * max(0.0, 36.0 - Q.n_real_top40) / 0.0006803313   # +0.0%  girth2_top10 < 0.0007432 and n_real_top40 < 36
        + 0.0002920945 * max(0.0, 74.25181 - Q.mass) * max(0.0, Q.dr_max_012 - 0.1206357) / 0.0008878091   # +0.0%  mass < 74.25 and dr_max_012 > 0.1206
        - 0.0002513107 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, Q.soft4_absphi - 0.170166) / 0.0001623923   # -0.0%  z_dr_0p1_0p2 > 0.6882 and soft4_absphi > 0.1702
        - 0.0001600229 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -0.0%  mass < 53.87
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 8.898;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.898207 * (0.27296
        - 0.09446603 * max(0.0, 0.02580396 - Q.mass_over_sum_pt_sq) / 0.01697662   # -9.4%  mass_over_sum_pt_sq < 0.0258
        + 0.08361211 * max(0.0, 0.02550569 - Q.girth2_top50) / 0.01683664   # +8.4%  girth2_top50 < 0.02551
        - 0.08102993 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -8.1%  sum_pt < 1085
        + 0.07942272 * max(0.0, Q.mass - 74.25181) / 24.07648   # +7.9%  mass > 74.25
        - 0.04762919 * max(0.0, Q.mass - 136.785) / 5.043396   # -4.8%  mass > 136.8
        - 0.04719299 * max(0.0, Q.mass - 143.7876) / 3.946979   # -4.7%  mass > 143.8
        + 0.04617369 * Q.n_particles / 45.81523   # +4.6%  n_particles
        - 0.04256369 * max(0.0, Q.mass_top50 - 92.16545) / 13.44564   # -4.3%  mass_top50 > 92.17
        + 0.0385121 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +3.9%  mass_top50 > 136.8
        - 0.03119332 * max(0.0, 160.8 - Q.mass_top40) / 76.75884   # -3.1%  mass_top40 < 160.8
        + 0.03065689 * max(0.0, Q.psi_0p3 - 0.9638082) / 0.02792824   # +3.1%  psi_0p3 > 0.9638
        - 0.02973703 * max(0.0, Q.mass - 160.8) / 1.724663   # -3.0%  mass > 160.8
        - 0.02641539 * max(0.0, 6.910131 - Q.log_sum_pt) / 0.017275   # -2.6%  log_sum_pt < 6.91
        + 0.02492902 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +2.5%  sum_pt_top40 < 1053
        - 0.02205212 * max(0.0, 1007.788 - Q.sum_pt) / 17.7122   # -2.2%  sum_pt < 1008
        - 0.02000001 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.8310045 - Q.tau21_b2) / 33.42141   # -2.0%  sum_pt < 1085 and tau21_b2 < 0.831
        + 0.01988756 * max(0.0, Q.mass - 162.8363) / 1.509832   # +2.0%  mass > 162.8
        + 0.01967039 * max(0.0, 0.007820315 - Q.girth2_top50) / 0.002264874   # +2.0%  girth2_top50 < 0.00782
        + 0.01555241 * max(0.0, Q.psi_0p3 - 0.9638082) * max(0.0, 0.0223982 - Q.C3) / 0.0004215996   # +1.6%  psi_0p3 > 0.9638 and C3 < 0.0224
        + 0.01534409 * max(0.0, 0.08286256 - Q.tau1) / 0.0188894   # +1.5%  tau1 < 0.08286
        + 0.0142699 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +1.4%  mass_over_sum_pt > 0.1709
        - 0.01367686 * max(0.0, Q.mass - 74.25181) * max(0.0, 1028.184 - Q.sum_pt) / 733.8204   # -1.4%  mass > 74.25 and sum_pt < 1028
        + 0.01332167 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # +1.3%  log_sum_pt < 6.811
        - 0.01228773 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -1.2%  mass_over_sum_pt_sq > 0.0292
        - 0.01192912 * max(0.0, 966.0633 - Q.sum_pt_top30) / 29.99923   # -1.2%  sum_pt_top30 < 966.1
        + 0.01095966 * max(0.0, Q.mass - 172.8) / 0.7291439   # +1.1%  mass > 172.8
        + 0.01034801 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # +1.0%  sum_pt_top40 < 1007
        + 0.009841314 * max(0.0, 997.0189 - Q.sum_pt_top50) / 17.90852   # +1.0%  sum_pt_top50 < 997
        + 0.006677063 * max(0.0, Q.sum_pt_top30 - 1111.245) / 12.62573   # +0.7%  sum_pt_top30 > 1111
        + 0.006654314 * max(0.0, Q.mass - 136.785) * max(0.0, 1028.184 - Q.sum_pt) / 132.4986   # +0.7%  mass > 136.8 and sum_pt < 1028
        - 0.006463986 * max(0.0, 16.0 - Q.n_for_90pct) / 1.776607   # -0.6%  n_for_90pct < 16
        + 0.00636913 * max(0.0, Q.psi_0p3 - 0.9638082) * max(0.0, 0.4357228 - Q.max_dr) / 0.002832914   # +0.6%  psi_0p3 > 0.9638 and max_dr < 0.4357
        + 0.006220111 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.2179035 - Q.sj2_zsoft) / 2.389974   # +0.6%  sum_pt_top40 < 1053 and sj2_zsoft < 0.2179
        + 0.006024656 * max(0.0, 16.0 - Q.n_for_90pct) * max(0.0, 3.814159 - Q.D2) / 1.817419   # +0.6%  n_for_90pct < 16 and D2 < 3.814
        + 0.005817642 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +0.6%  pt_6 < 41.22
        + 0.005173753 * max(0.0, Q.psi_0p3 - 0.9638082) * max(0.0, Q.absphi_12 - 0.01163483) / 0.001186382   # +0.5%  psi_0p3 > 0.9638 and absphi_12 > 0.01163
        + 0.003881249 * max(0.0, 0.08871546 - Q.z_dr_0p1_0p2) / 0.03070029   # +0.4%  z_dr_0p1_0p2 < 0.08872
        - 0.003694568 * max(0.0, Q.mass_top50 - 168.9698) / 0.6651775   # -0.4%  mass_top50 > 169
        + 0.003343291 * max(0.0, Q.mass_top5 - 33.71058) / 7.394508   # +0.3%  mass_top5 > 33.71
        - 0.003107274 * max(0.0, Q.tau1 - 0.1751567) / 0.002322569   # -0.3%  tau1 > 0.1752
        - 0.002615988 * max(0.0, Q.mass_top10 - 56.92192) / 8.071231   # -0.3%  mass_top10 > 56.92
        + 0.002543251 * max(0.0, 76.9886 - Q.mass_top10) / 32.58955   # +0.3%  mass_top10 < 76.99
        - 0.00244049 * max(0.0, Q.psi_0p3 - 0.9638082) * max(0.0, Q.sj3_dr13 - 0.2465017) / 0.0003007569   # -0.2%  psi_0p3 > 0.9638 and sj3_dr13 > 0.2465
        - 0.00196272 * max(0.0, Q.C2 - 0.0977156) / 0.006150774   # -0.2%  C2 > 0.09772
        + 0.001933318 * max(0.0, 18.32812 - Q.pt_11) / 2.095384   # +0.2%  pt_11 < 18.33
        + 0.001927858 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # +0.2%  n_dr_0p1_0p2 < 17
        - 0.001766524 * max(0.0, Q.mass - 172.8) * max(0.0, 56.53125 - Q.pt_6) / 9.768822   # -0.2%  mass > 172.8 and pt_6 < 56.53
        - 0.001734093 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.1023813 - Q.dr_13) / 2.159889   # -0.2%  sum_pt < 1085 and dr_13 < 0.1024
        + 0.001015653 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.005041702 - Q.zdr_11) / 1.572361e-05   # +0.1%  log_sum_pt < 6.811 and zdr_11 < 0.005042
        + 0.000942981 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 858.8262 - Q.sum_pt_top40) / 1219.241   # +0.1%  sum_pt < 1085 and sum_pt_top40 < 858.8
        + 0.0008142698 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.1312677 - Q.dr_13) / 0.0002562188   # +0.1%  log_sum_pt < 6.811 and dr_13 < 0.1313
        - 0.0006401523 * max(0.0, Q.sum_pt_top20 - 956.5062) / 37.46564   # -0.1%  sum_pt_top20 > 956.5
        + 0.0005819563 * max(0.0, Q.mass - 172.8) * max(0.0, 0.04568661 - Q.z_6) / 0.006788705   # +0.1%  mass > 172.8 and z_6 < 0.04569
        + 0.0005608186 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.sd_zg - 0.4462823) / 1.289737e-05   # +0.1%  log_sum_pt > 7.139 and sd_zg > 0.4463
        - 0.000494303 * max(0.0, Q.C2 - 0.0977156) * max(0.0, 53.9394 - Q.orientation_deg) / 0.3542801   # -0.0%  C2 > 0.09772 and orientation_deg < 53.94
        + 0.0004717758 * max(0.0, 2.752054 - Q.pt_entropy) / 0.1846134   # +0.0%  pt_entropy < 2.752
        + 0.0004263577 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.05398073 - Q.dr_12) / 9.985673e-05   # +0.0%  log_sum_pt > 7.139 and dr_12 < 0.05398
        - 0.0003912218 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.00317537 - Q.C3) / 7.466262e-06   # -0.0%  log_sum_pt > 7.139 and C3 < 0.003175
        + 0.0003372443 * max(0.0, 1007.788 - Q.sum_pt) * max(0.0, Q.sum_pt_top3 - 512.4375) / 295.3022   # +0.0%  sum_pt < 1008 and sum_pt_top3 > 512.4
        + 0.0002970359 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # +0.0%  log_sum_pt > 7.139
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 12.71;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.71218 * (-0.009680158
        - 0.1333475 * max(0.0, 91.19 - Q.mass) / 16.46423   # -13.3%  mass < 91.19
        - 0.1293451 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) / 0.0178873   # -12.9%  mass_over_sum_pt < 0.09047
        + 0.1046949 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +10.5%  mass_over_sum_pt < 0.1182
        + 0.0600987 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +6.0%  mass_over_sum_pt < 0.09795
        - 0.05900473 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -5.9%  girth2_top30 < 0.01216
        + 0.05170566 * max(0.0, 80.4 - Q.mass) / 10.6806   # +5.2%  mass < 80.4
        + 0.04941027 * max(0.0, 125.1 - Q.mass) / 42.45775   # +4.9%  mass < 125.1
        + 0.04871175 * max(0.0, 0.08873143 - Q.mass_over_sum_pt) / 0.01671255   # +4.9%  mass_over_sum_pt < 0.08873
        - 0.04387649 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p2 - 0.9087063) / 0.00142529   # -4.4%  mass_over_sum_pt < 0.09047 and psi_0p2 > 0.9087
        + 0.03629714 * max(0.0, 0.140939 - Q.mass_over_sum_pt) / 0.05796036   # +3.6%  mass_over_sum_pt < 0.1409
        - 0.02440333 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -2.4%  girth2_top20 < 0.01083
        - 0.02048005 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -2.0%  tau1 < 0.1073
        - 0.01687285 * max(0.0, 0.076787 - Q.girth) / 0.02105267   # -1.7%  girth < 0.07679
        + 0.01640765 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +1.6%  psi_0p3 > 0.9924
        - 0.01638123 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # -1.6%  sd_rg < 0.3017
        + 0.01481606 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # +1.5%  girth2_top5 < 0.007164
        + 0.01457029 * max(0.0, 0.1778185 - Q.sd_rg) / 0.06576479   # +1.5%  sd_rg < 0.1778
        + 0.01253312 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # +1.3%  girth2_top20 < 0.006374
        - 0.01219885 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1881908 - Q.sd_rg) / 0.0002726472   # -1.2%  psi_0p3 > 0.9924 and sd_rg < 0.1882
        + 0.01158009 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +1.2%  e2 < 0.0303
        + 0.01028152 * max(0.0, 82.66587 - Q.mass_top30) / 15.96638   # +1.0%  mass_top30 < 82.67
        + 0.01015408 * max(0.0, 6.941997 - Q.log_sum_pt) / 0.03180972   # +1.0%  log_sum_pt < 6.942
        + 0.01012723 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +1.0%  tau1 < 0.06311
        + 0.009523844 * max(0.0, 0.076787 - Q.girth) * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.0007421723   # +1.0%  girth < 0.07679 and z_dr_0p2_0p4 < 0.0518
        + 0.008363003 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +0.8%  tau21_b2 < 0.3425
        - 0.008187582 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.007709916 - Q.girth2_top40) / 0.0001062984   # -0.8%  tau21_b2 < 0.3425 and girth2_top40 < 0.00771
        + 0.005580652 * max(0.0, 0.07852883 - Q.mass_over_sum_pt) / 0.01107516   # +0.6%  mass_over_sum_pt < 0.07853
        - 0.005329751 * max(0.0, 0.02210818 - Q.e2) / 0.00342102   # -0.5%  e2 < 0.02211
        + 0.00527027 * max(0.0, 69.65633 - Q.sd_mass) / 24.56034   # +0.5%  sd_mass < 69.66
        + 0.005199895 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # +0.5%  sum_pt_top40 < 1025
        - 0.004599206 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # -0.5%  log_sum_pt < 6.989
        + 0.003958528 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +0.4%  n_dr_0p2_0p4 < 18
        - 0.003643218 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2404747) / 0.01288721   # -0.4%  N2 < 0.4227 and max_dr > 0.2405
        + 0.00361072 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # +0.4%  sj3_mass1 < 32.5
        - 0.00316631 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -0.3%  mass < 82.85
        - 0.003147802 * max(0.0, 0.007164202 - Q.girth2_top5) * max(0.0, Q.n_pt_above_10 - 13.0) / 0.02038471   # -0.3%  girth2_top5 < 0.007164 and n_pt_above_10 > 13
        - 0.002583892 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1231747 - Q.z_2nd) / 6.106735e-05   # -0.3%  psi_0p3 > 0.9924 and z_2nd < 0.1232
        - 0.002309277 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 32.0 - Q.n_real_top40) / 0.1595466   # -0.2%  tau21_b2 < 0.3425 and n_real_top40 < 32
        + 0.00154645 * max(0.0, 0.001163277 - Q.lam2) / 0.0004869716   # +0.2%  lam2 < 0.001163
        - 0.001481952 * max(0.0, 0.01215787 - Q.girth2_top30) * max(0.0, Q.pair_mass_0_2 - 22.41128) / 0.003197931   # -0.1%  girth2_top30 < 0.01216 and pair_mass_0_2 > 22.41
        + 0.001442671 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03787151 - Q.M3) / 0.1118325   # +0.1%  n_dr_0p2_0p4 < 18 and M3 < 0.03787
        - 0.001370483 * max(0.0, 906.6023 - Q.sum_pt_top40) / 6.984569   # -0.1%  sum_pt_top40 < 906.6
        + 0.001338517 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.1943707   # +0.1%  tau21_b2 < 0.3425 and n_pt_above_50 < 7
        - 0.00126892 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.n_dr_0p05_0p1 - 14.0) / 15.57557   # -0.1%  mass < 91.19 and n_dr_0p05_0p1 > 14
        - 0.001185508 * max(0.0, 976.277 - Q.sum_pt_top50) / 12.87298   # -0.1%  sum_pt_top50 < 976.3
        + 0.00102094 * max(0.0, 0.4226723 - Q.N2) * max(0.0, 0.2053949 - Q.dr_12) / 0.0131481   # +0.1%  N2 < 0.4227 and dr_12 < 0.2054
        + 0.000994209 * max(0.0, 0.4226723 - Q.N2) / 0.1194948   # +0.1%  N2 < 0.4227
        + 0.0009776559 * max(0.0, 0.001592178 - Q.girth2_top3) / 0.000513961   # +0.1%  girth2_top3 < 0.001592
        - 0.0009412373 * max(0.0, 125.1 - Q.mass) * max(0.0, Q.mass_top2 - 28.78966) / 17.57849   # -0.1%  mass < 125.1 and mass_top2 > 28.79
        - 0.0008759764 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, Q.sj3_mass2 - 5.762132) / 0.2507301   # -0.1%  tau21_b2 < 0.3425 and sj3_mass2 > 5.762
        + 0.000815423 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.006118006 - Q.girth2_top50) / 2.65509e-05   # +0.1%  tau21_b2 < 0.3425 and girth2_top50 < 0.006118
        + 0.0006353783 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.02048524 - Q.lam1) / 0.001304679   # +0.1%  tau21_b2 < 0.3425 and lam1 < 0.02049
        + 0.0005426015 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, Q.n_for_50pct - 3.0) / 0.1980908   # +0.1%  tau21_b2 < 0.3425 and n_for_50pct > 3
        + 0.0005005127 * max(0.0, 0.007164202 - Q.girth2_top5) * max(0.0, Q.sj2_mass2 - 14.45919) / 0.003408014   # +0.1%  girth2_top5 < 0.007164 and sj2_mass2 > 14.46
        + 0.0004470406 * max(0.0, 0.007164202 - Q.girth2_top5) * max(0.0, Q.pair_mass_0_4 - 8.587823) / 0.003768116   # +0.0%  girth2_top5 < 0.007164 and pair_mass_0_4 > 8.588
        - 0.0004088596 * max(0.0, 0.007164202 - Q.girth2_top5) * max(0.0, Q.zdr_5 - 0.008949744) / 1.839629e-07   # -0.0%  girth2_top5 < 0.007164 and zdr_5 > 0.00895
        + 0.0003119707 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4) / 0.08975043   # +0.0%  z_dr_0p2_0p4 < 0.1292
        + 6.877275e-05 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, Q.sd_rg - 0.2280025) / 1.846945e-05   # +0.0%  psi_0p3 > 0.9924 and sd_rg > 0.228
        + 4.560679e-05 * max(0.0, Q.pt_10 - 28.15625) / 1.573077   # +0.0%  pt_10 > 28.16
        + 6.72617e-06 * max(0.0, Q.n_pt_above_1 - 58.0) / 0.9616151   # +0.0%  n_pt_above_1 > 58
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 5.027;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.027327 * (-0.03055075
        + 0.1070913 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +10.7%  log_sum_pt < 7.017
        + 0.07849058 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +7.8%  girth2_top5 < 0.00833
        + 0.06381139 * max(0.0, 79.18312 - Q.sd_mass) / 30.46368   # +6.4%  sd_mass < 79.18
        - 0.04939076 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -4.9%  tau1 < 0.07057
        + 0.04813378 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +4.8%  z_dr_0p1_0p2 < 0.1203
        + 0.04294149 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # +4.3%  girth2_top10 < 0.007679
        - 0.03160588 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 1260.541 - Q.sum_pt) / 87.05552   # -3.2%  z_dr_0_0p05 < 0.8459 and sum_pt < 1261
        - 0.02912541 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -2.9%  psi_0p1 > 0.8976
        - 0.02897997 * max(0.0, 1018.698 - Q.sum_pt_top40) / 34.89083   # -2.9%  sum_pt_top40 < 1019
        - 0.02811727 * max(0.0, 45.595 - Q.sd_mass) / 13.46843   # -2.8%  sd_mass < 45.59
        - 0.02608208 * max(0.0, 1.976207 - Q.D2) / 0.367701   # -2.6%  D2 < 1.976
        - 0.02561953 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -2.6%  psi_0p3 > 0.9897
        + 0.02534445 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) / 0.3666529   # +2.5%  z_dr_0_0p05 < 0.8459
        + 0.02320274 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +2.3%  lam2 < 0.001776
        - 0.01975981 * max(0.0, 0.001056655 - Q.girth2_top2) / 0.0003002514   # -2.0%  girth2_top2 < 0.001057
        - 0.01946835 * max(0.0, 0.002270363 - Q.girth2_top5) / 0.0007634099   # -1.9%  girth2_top5 < 0.00227
        - 0.01788745 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -1.8%  sum_pt < 1002
        + 0.01786875 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) / 3.82104   # +1.8%  n_dr_0p1_0p2 < 13
        - 0.0170051 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -1.7%  log_sum_pt < 6.903
        - 0.01626577 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # -1.6%  mass_top50 < 71.8
        + 0.01595611 * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.01931881   # +1.6%  z_dr_0p1_0p2 < 0.06473
        + 0.01595163 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +1.6%  sj2_dr > 0.2232
        - 0.01518396 * max(0.0, Q.psi_0p1 - 0.9371031) / 0.01088804   # -1.5%  psi_0p1 > 0.9371
        - 0.01472386 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 14.45919 - Q.sj2_mass2) / 0.03398137   # -1.5%  psi_0p3 > 0.9897 and sj2_mass2 < 14.46
        + 0.01381771 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +1.4%  lam1 < 0.004673
        + 0.01314882 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2) / 0.0004272674   # +1.3%  sj2_dr > 0.2232 and C2_b2 < 0.04009
        - 0.01287796 * max(0.0, 1017.778 - Q.sum_pt_top20) / 107.0228   # -1.3%  sum_pt_top20 < 1018
        - 0.01254893 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583) / 0.0009330603   # -1.3%  girth2_top5 < 0.00833 and sj3_pairmin_over_m > 0.09541
        - 0.01169001 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.n_real_top40 - 22.0) / 0.06106188   # -1.2%  girth2_top5 < 0.00833 and n_real_top40 > 22
        - 0.01159412 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # -1.2%  sum_pt < 986.1
        + 0.011501 * max(0.0, 933.1875 - Q.sum_pt_top30) / 20.26547   # +1.2%  sum_pt_top30 < 933.2
        - 0.0110303 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.02970886 - Q.absphi_0) / 0.06277088   # -1.1%  n_dr_0p1_0p2 < 13 and absphi_0 < 0.02971
        - 0.01077799 * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 3.233647   # -1.1%  n_dr_0p2_0p4 > 9
        + 0.009705688 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, 0.04770182 - Q.z_6) / 5.265762e-05   # +1.0%  girth2_top5 < 0.00833 and z_6 < 0.0477
        - 0.009501683 * max(0.0, Q.lam1 - 0.01649354) / 0.001003295   # -1.0%  lam1 > 0.01649
        + 0.009036524 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 137.5 - Q.pt_2) / 0.2617409   # +0.9%  psi_0p3 > 0.9897 and pt_2 < 137.5
        + 0.008389779 * max(0.0, 35.30013 - Q.sj3_pair_mass_max) / 1.577545   # +0.8%  sj3_pair_mass_max < 35.3
        + 0.00815469 * max(0.0, 0.0005124533 - Q.girth2_top2) / 0.0001104529   # +0.8%  girth2_top2 < 0.0005125
        - 0.007965559 * max(0.0, 0.002288114 - Q.z_dr_0p2_0p4) / 0.0002954373   # -0.8%  z_dr_0p2_0p4 < 0.002288
        + 0.007912777 * max(0.0, 935.8189 - Q.sum_pt_top40) / 10.51928   # +0.8%  sum_pt_top40 < 935.8
        + 0.007541905 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 0.777038 - Q.planar_flow) / 0.0106194   # +0.8%  z_dr_0p1_0p2 < 0.1203 and planar_flow < 0.777
        - 0.006620574 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0) / 0.2039042   # -0.7%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p2_0p4 > 5
        - 0.006615235 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.n_dr_0p05_0p1 - 8.0) / 14.10129   # -0.7%  n_dr_0p1_0p2 < 13 and n_dr_0p05_0p1 > 8
        - 0.003867767 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677) / 0.04548726   # -0.4%  girth2_top5 < 0.00833 and sj3_mass1 > 5.113
        + 0.003735734 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt) / 2.722316   # +0.4%  z_dr_0_0p05 < 0.8459 and sum_pt < 949.9
        - 0.003490049 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 76.29481 - Q.sd_mass) / 2.433638   # -0.3%  z_dr_0p1_0p2 < 0.1203 and sd_mass < 76.29
        + 0.003265609 * max(0.0, 0.00287991 - Q.girth2_top20) / 0.000559956   # +0.3%  girth2_top20 < 0.00288
        + 0.003032696 * max(0.0, 0.00287991 - Q.girth2_top20) * max(0.0, 0.4357228 - Q.max_dr) / 4.319324e-05   # +0.3%  girth2_top20 < 0.00288 and max_dr < 0.4357
        - 0.002800577 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 58.0 - Q.n_particles) / 0.8649544   # -0.3%  z_dr_0p1_0p2 < 0.1203 and n_particles < 58
        - 0.002505975 * max(0.0, Q.sj2_dr - 0.2971569) / 0.005899669   # -0.3%  sj2_dr > 0.2972
        - 0.002376046 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, 0.948102 - Q.psi_0p2) / 3.786811e-05   # -0.2%  girth2_top5 < 0.00833 and psi_0p2 < 0.9481
        + 0.001722824 * max(0.0, Q.dr_7 - 0.2229947) / 0.002728807   # +0.2%  dr_7 > 0.223
        - 0.001340854 * max(0.0, 79.18312 - Q.sd_mass) * max(0.0, Q.eta_2 - -0.03302612) / 1.045361   # -0.1%  sd_mass < 79.18 and eta_2 > -0.03303
        + 0.001150134 * max(0.0, 45.595 - Q.sd_mass) * max(0.0, -0.009371567 - Q.eta_1) / 0.02469491   # +0.1%  sd_mass < 45.59 and eta_1 < -0.009372
        + 0.000556502 * max(0.0, 0.001776308 - Q.lam2) * max(0.0, Q.dr_12 - 0.2053949) / 4.456629e-06   # +0.1%  lam2 < 0.001776 and dr_12 > 0.2054
        - 0.0005462848 * max(0.0, 0.00287991 - Q.girth2_top20) * max(0.0, Q.sum_pt - 1167.447) / 0.01532317   # -0.1%  girth2_top20 < 0.00288 and sum_pt > 1167
        - 0.0005318925 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.0009102912 - Q.soft5_z) / 2.772596e-06   # -0.1%  sj2_dr > 0.2232 and soft5_z < 0.0009103
        + 0.0005060624 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.dr0_8 - 0.2441824) / 1.193182e-05   # +0.1%  girth2_top5 < 0.00833 and dr0_8 > 0.2442
        + 9.040995e-05 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p4_up - 0.0) / 0.006297153   # +0.0%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p4_up > 0
        - 4.191194e-05 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # -0.0%  mass_over_sum_pt > 0.1709
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9689016281512605, 3.2148966386554623, 0.09698949579831932, 0.7326774159663866, 1.2917269957983193, 1.2761654411764707, 0.9184373949579832, 0.7093543067226891, 1.883441806722689, 1.9681808823529412, 1.274841806722689, 0.7717379201680672, 0.8757718487394958, 2.166137920168067, 0.3801191176470588, 0.41738487394957985]
T = [5.530266379989495, 3.8387159007352945, 6.223261973148634, 6.373293864889706, 5.261227135307248]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +36%, n4 -20%, n9 +18%, n3 -10%, n5 -7%, n12 +4% ...
            + 0.3633298 * h[1] / H_AVG[1]
            - 0.2043773 * h[4] / H_AVG[4]
            + 0.1835071 * h[9] / H_AVG[9]
            - 0.09936376 * h[3] / H_AVG[3]
            - 0.07211257 * h[5] / H_AVG[5]
            + 0.03711558 * h[12] / H_AVG[12]
            + 0.02594917 * h[6] / H_AVG[6]
            + 0.01064281 * h[8] / H_AVG[8]
            - 0.003601889 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -36%, n9 +29%, n1 -16%, n12 +8%, n6 +5%, n11 -5% ...
            - 0.357531 * h[4] / H_AVG[4]
            + 0.2884042 * h[9] / H_AVG[9]
            - 0.1570299 * h[1] / H_AVG[1]
            + 0.07842377 * h[12] / H_AVG[12]
            + 0.05233734 * h[6] / H_AVG[6]
            - 0.05026016 * h[11] / H_AVG[11]
            + 0.007666308 * h[8] / H_AVG[8]
            - 0.00518908 * h[10] / H_AVG[10]
            + 0.003158266 * h[2] / H_AVG[2]
        ),
        0.09375 + T[2] * (   # class W: n8 -26%, n5 +12%, n0 +12%, n14 -8%, n11 +7%, n4 +7% ...
            - 0.2648148 * h[8] / H_AVG[8]
            + 0.1185525 * h[5] / H_AVG[5]
            + 0.1167677 * h[0] / H_AVG[0]
            - 0.0839855 * h[14] / H_AVG[14]
            + 0.0736301 * h[11] / H_AVG[11]
            + 0.07135023 * h[4] / H_AVG[4]
            - 0.0712402 * h[7] / H_AVG[7]
            - 0.0691823 * h[9] / H_AVG[9]
            - 0.05716975 * h[12] / H_AVG[12]
            + 0.05150777 * h[3] / H_AVG[3]
            - 0.01257534 * h[15] / H_AVG[15]
            + 0.009223834 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -28%, n0 -21%, n6 -14%, n5 +14%, n7 +10%, n3 +5% ...
            - 0.2770509 * h[8] / H_AVG[8]
            - 0.2090347 * h[0] / H_AVG[0]
            - 0.1396038 * h[6] / H_AVG[6]
            + 0.1376625 * h[5] / H_AVG[5]
            + 0.1008666 * h[7] / H_AVG[7]
            + 0.05029524 * h[3] / H_AVG[3]
            + 0.04294148 * h[12] / H_AVG[12]
            + 0.03683794 * h[15] / H_AVG[15]
            - 0.005706792 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -37%, n10 +24%, n5 -11%, n8 +8%, n12 -6%, n4 +5% ...
            - 0.3731187 * h[13] / H_AVG[13]
            + 0.2385228 * h[10] / H_AVG[10]
            - 0.1137002 * h[5] / H_AVG[5]
            + 0.07551252 * h[8] / H_AVG[8]
            - 0.06242164 * h[12] / H_AVG[12]
            + 0.04603466 * h[4] / H_AVG[4]
            - 0.03792003 * h[7] / H_AVG[7]
            - 0.02974959 * h[15] / H_AVG[15]
            + 0.02301986 * h[0] / H_AVG[0]
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
