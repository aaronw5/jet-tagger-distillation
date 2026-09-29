"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; all observables, tuned for agreement), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.9%   (on for 53% of jets)
  neuron  4:  12.4%   (on for 74% of jets)
  neuron  5:  10.8%   (on for 83% of jets)
  neuron  9:   9.8%   (on for 91% of jets)
  neuron  1:   9.6%   (on for 71% of jets)
  neuron  0:   7.4%   (on for 65% of jets)
  neuron 13:   6.2%   (on for 83% of jets)
  neuron 12:   5.4%   (on for 47% of jets)
  neuron  6:   5.0%   (on for 72% of jets)
  neuron  7:   4.9%   (on for 79% of jets)
  neuron 10:   4.4%   (on for 60% of jets)
  neuron  3:   4.1%   (on for 66% of jets)
  neuron 11:   2.6%   (on for 71% of jets)
  neuron 14:   2.1%   (on for 55% of jets)
  neuron 15:   1.2%   (on for 45% of jets)
  neuron  2:   0.1%   (on for 25% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 79.6% (the network: 81.1%); same class as the network for 91.7% of jets.

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
  Q.pair_mass_0_5          mass of particles 0 and 5 [GeV]
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
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
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
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.ptdr0_7                pT7 · ΔR(0, 7) [GeV]
  Q.ptdr0_8                pT8 · ΔR(0, 8) [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_pt              pT [GeV] of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_pt               pT [GeV] of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_pt               pT [GeV] of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_10                   pT of particle 10 / total pT
  Q.z_11                   pT of particle 11 / total pT
  Q.z_2                    pT of particle 2 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_8                    pT of particle 8 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft10_z               pT share of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_z                pT share of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_z                pT share of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.sj3_z1                 pT share of subjet 1 of 3 (by pT)
  Q.sj3_z2                 pT share of subjet 2 of 3 (by pT)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_10                 pT share × ΔR of particle 10 (its part of the sum_z_dr)
  Q.zdr_11                 pT share × ΔR of particle 11 (its part of the sum_z_dr)
  Q.zdr_12                 pT share × ΔR of particle 12 (its part of the sum_z_dr)
  Q.zdr_13                 pT share × ΔR of particle 13 (its part of the sum_z_dr)
  Q.zdr_14                 pT share × ΔR of particle 14 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.pt1_over_pt0           pT1 / pT0
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_11              |Δη| of particle 11
  Q.abseta_14              |Δη| of particle 14
  Q.abseta_3               |Δη| of particle 3
  Q.abseta_4               |Δη| of particle 4
  Q.soft1_abseta           |Δη| of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_abseta          |Δη| of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_abseta           |Δη| of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_abseta           |Δη| of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_12              |Δφ| of particle 12
  Q.absphi_13              |Δφ| of particle 13
  Q.absphi_2               |Δφ| of particle 2
  Q.absphi_4               |Δφ| of particle 4
  Q.absphi_5               |Δφ| of particle 5
  Q.absphi_7               |Δφ| of particle 7
  Q.absphi_9               |Δφ| of particle 9
  Q.soft3_absphi           |Δφ| of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_absphi           |Δφ| of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_absphi           |Δφ| of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_absphi           |Δφ| of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft10_dr0             ΔR between the hardest and the 10. softest real particle (0 if among the 15 hardest)
  Q.soft3_dr0              ΔR between the hardest and the 3. softest real particle (0 if among the 15 hardest)
  Q.soft4_dr0              ΔR between the hardest and the 4. softest real particle (0 if among the 15 hardest)
  Q.soft5_dr0              ΔR between the hardest and the 5. softest real particle (0 if among the 15 hardest)
  Q.soft6_dr0              ΔR between the hardest and the 6. softest real particle (0 if among the 15 hardest)
  Q.soft7_dr0              ΔR between the hardest and the 7. softest real particle (0 if among the 15 hardest)
  Q.soft8_dr0              ΔR between the hardest and the 8. softest real particle (0 if among the 15 hardest)
  Q.dr0_10                 ΔR between particle 10 and the hardest particle
  Q.dr0_12                 ΔR between particle 12 and the hardest particle
  Q.dr0_14                 ΔR between particle 14 and the hardest particle
  Q.dr0_3                  ΔR between particle 3 and the hardest particle
  Q.dr0_8                  ΔR between particle 8 and the hardest particle
  Q.dr1_11                 ΔR between particle 11 and the 2nd-hardest particle
  Q.dr1_12                 ΔR between particle 12 and the 2nd-hardest particle
  Q.dr1_13                 ΔR between particle 13 and the 2nd-hardest particle
  Q.dr1_3                  ΔR between particle 3 and the 2nd-hardest particle
  Q.dr1_5                  ΔR between particle 5 and the 2nd-hardest particle
  Q.dr1_8                  ΔR between particle 8 and the 2nd-hardest particle
  Q.dr1_9                  ΔR between particle 9 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_10                  ΔR of particle 10 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_12                  ΔR of particle 12 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr_8                   ΔR of particle 8 from the jet axis
  Q.dr_9                   ΔR of particle 9 from the jet axis
  Q.soft1_dr               ΔR from the jet axis of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_dr              ΔR from the jet axis of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_dr               ΔR from the jet axis of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_dr               ΔR from the jet axis of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_dr               ΔR from the jet axis of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_dr               ΔR from the jet axis of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_dr               ΔR from the jet axis of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_dr               ΔR from the jet axis of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_dr               ΔR from the jet axis of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.dr01                   ΔR between particles 0 and 1
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_11                 Δη of particle 11
  Q.eta_12                 Δη of particle 12
  Q.eta_13                 Δη of particle 13
  Q.eta_2                  Δη of particle 2
  Q.eta_3                  Δη of particle 3
  Q.eta_4                  Δη of particle 4
  Q.eta_5                  Δη of particle 5
  Q.eta_8                  Δη of particle 8
  Q.eta_9                  Δη of particle 9
  Q.phi_0                  Δφ of particle 0
  Q.phi_10                 Δφ of particle 10
  Q.phi_11                 Δφ of particle 11
  Q.phi_13                 Δφ of particle 13
  Q.phi_14                 Δφ of particle 14
  Q.phi_2                  Δφ of particle 2
  Q.phi_5                  Δφ of particle 5
  Q.phi_6                  Δφ of particle 6
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.sum_z_dr2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.sum_z_dr2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.z_dr_0p4_up            pT share of the particles with 0.4 ≤ ΔR < 10
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_phi               pT-weighted mean Δφ
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.tau43                  N-subjettiness τ4/τ3
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
        pair_mass_0_10=pair_mass(0, 10),
        pair_mass_0_12=pair_mass(0, 12),
        pair_mass_0_13=pair_mass(0, 13),
        pair_mass_0_2=pair_mass(0, 2),
        pair_mass_0_4=pair_mass(0, 4),
        pair_mass_0_5=pair_mass(0, 5),
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
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
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
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        ptdr0_7=pt[7] * math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        ptdr0_8=pt[8] * math.sqrt(dist2(0, 8)) if pt[8] > 0 else 0.0,
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft10_pt=softp(10, 'pt'),
        soft2_pt=softp(2, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft7_pt=softp(7, 'pt'),
        soft9_pt=softp(9, 'pt'),
        z_10=z[10],
        z_11=z[11],
        z_2=z[2],
        z_3=z[3],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        z_8=z[8],
        z_9=z[9],
        soft10_z=softp(10, 'z'),
        soft3_z=softp(3, 'z'),
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft7_z=softp(7, 'z'),
        soft8_z=softp(8, 'z'),
        soft9_z=softp(9, 'z'),
        sj3_z1=subjets(3)["z"][0],
        sj3_z2=subjets(3)["z"][1],
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
        zdr_10=z[10] * dr[10],
        zdr_11=z[11] * dr[11],
        zdr_12=z[12] * dr[12],
        zdr_13=z[13] * dr[13],
        zdr_14=z[14] * dr[14],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        abseta_0=abs(eta[0]),
        abseta_11=abs(eta[11]),
        abseta_14=abs(eta[14]),
        abseta_3=abs(eta[3]),
        abseta_4=abs(eta[4]),
        soft1_abseta=softp(1, 'abseta'),
        soft10_abseta=softp(10, 'abseta'),
        soft2_abseta=softp(2, 'abseta'),
        soft9_abseta=softp(9, 'abseta'),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_12=abs(phi[12]),
        absphi_13=abs(phi[13]),
        absphi_2=abs(phi[2]),
        absphi_4=abs(phi[4]),
        absphi_5=abs(phi[5]),
        absphi_7=abs(phi[7]),
        absphi_9=abs(phi[9]),
        soft3_absphi=softp(3, 'absphi'),
        soft4_absphi=softp(4, 'absphi'),
        soft5_absphi=softp(5, 'absphi'),
        soft9_absphi=softp(9, 'absphi'),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft10_dr0=softp(10, 'dr0'),
        soft3_dr0=softp(3, 'dr0'),
        soft4_dr0=softp(4, 'dr0'),
        soft5_dr0=softp(5, 'dr0'),
        soft6_dr0=softp(6, 'dr0'),
        soft7_dr0=softp(7, 'dr0'),
        soft8_dr0=softp(8, 'dr0'),
        dr0_10=math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        dr0_12=math.sqrt(dist2(0, 12)) if pt[12] > 0 else 0.0,
        dr0_14=math.sqrt(dist2(0, 14)) if pt[14] > 0 else 0.0,
        dr0_3=math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        dr0_8=math.sqrt(dist2(0, 8)) if pt[8] > 0 else 0.0,
        dr1_11=math.sqrt(dist2(1, 11)) if pt[11] > 0 else 0.0,
        dr1_12=math.sqrt(dist2(1, 12)) if pt[12] > 0 else 0.0,
        dr1_13=math.sqrt(dist2(1, 13)) if pt[13] > 0 else 0.0,
        dr1_3=math.sqrt(dist2(1, 3)) if pt[3] > 0 else 0.0,
        dr1_5=math.sqrt(dist2(1, 5)) if pt[5] > 0 else 0.0,
        dr1_8=math.sqrt(dist2(1, 8)) if pt[8] > 0 else 0.0,
        dr1_9=math.sqrt(dist2(1, 9)) if pt[9] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_10=dr[10] if pt[10] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_12=dr[12] if pt[12] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr_8=dr[8] if pt[8] > 0 else 0.0,
        dr_9=dr[9] if pt[9] > 0 else 0.0,
        soft1_dr=softp(1, 'dr'),
        soft10_dr=softp(10, 'dr'),
        soft2_dr=softp(2, 'dr'),
        soft3_dr=softp(3, 'dr'),
        soft4_dr=softp(4, 'dr'),
        soft5_dr=softp(5, 'dr'),
        soft7_dr=softp(7, 'dr'),
        soft8_dr=softp(8, 'dr'),
        soft9_dr=softp(9, 'dr'),
        dr01=math.sqrt(dist2(0, 1)),
        eta_0=eta[0],
        eta_1=eta[1],
        eta_11=eta[11],
        eta_12=eta[12],
        eta_13=eta[13],
        eta_2=eta[2],
        eta_3=eta[3],
        eta_4=eta[4],
        eta_5=eta[5],
        eta_8=eta[8],
        eta_9=eta[9],
        phi_0=phi[0],
        phi_10=phi[10],
        phi_11=phi[11],
        phi_13=phi[13],
        phi_14=phi[14],
        phi_2=phi[2],
        phi_5=phi[5],
        phi_6=phi[6],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        sum_z_dr2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        sum_z_dr2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        z_dr_0p4_up=sum(z[i] for i in real if 0.4 <= dr[i] < 10),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 15.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.57308 * (0.07708857
        - 0.179898 * max(0.0, Q.mass - 78.26182) / 21.33658   # -18.0%  mass > 78.26
        + 0.1041738 * max(0.0, Q.mass - 91.19) / 15.01545   # +10.4%  mass > 91.19
        + 0.09254465 * max(0.0, Q.mass - 89.74183) / 15.55572   # +9.3%  mass > 89.74
        - 0.06598467 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -6.6%  mass_top50 > 82.04
        + 0.05233804 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 0.002241068   # +5.2%  mass_over_sum_pt_sq < 0.007873
        + 0.04890913 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq) / 0.001689525   # +4.9%  mass_over_sum_pt_sq < 0.006939
        - 0.04727455 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -4.7%  mass < 101
        - 0.04197135 * max(0.0, 0.00616708 - Q.sum_zz_dr2) / 0.001295555   # -4.2%  sum_zz_dr2 < 0.006167
        + 0.03132074 * max(0.0, Q.mass_top50 - 71.79516) / 24.22501   # +3.1%  mass_top50 > 71.8
        - 0.02678909 * max(0.0, Q.mass - 74.25181) / 24.07648   # -2.7%  mass > 74.25
        - 0.02221551 * max(0.0, 0.006363916 - Q.sum_z_dr2_top30) / 0.001678624   # -2.2%  sum_z_dr2_top30 < 0.006364
        - 0.02051574 * max(0.0, 0.006374178 - Q.sum_z_dr2_top20) / 0.001987751   # -2.1%  sum_z_dr2_top20 < 0.006374
        - 0.02038338 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.0%  tau1 < 0.07057
        + 0.0180239 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +1.8%  sum_pt_top50 < 1157
        - 0.01714217 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # -1.7%  z_dr_0p2_0p4 < 0.09123
        + 0.01709511 * max(0.0, 0.007856958 - Q.sum_z_dr2_top30) / 0.002601721   # +1.7%  sum_z_dr2_top30 < 0.007857
        - 0.0157018 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # -1.6%  sum_pt_top40 < 1070
        - 0.01292538 * max(0.0, Q.mass - 92.85979) / 14.48273   # -1.3%  mass > 92.86
        + 0.01094405 * max(0.0, 0.004754444 - Q.mass_over_sum_pt_sq) / 0.0007997283   # +1.1%  mass_over_sum_pt_sq < 0.004754
        + 0.01087984 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957) / 1.852228   # +1.1%  log_sum_pt < 6.989 and sum_pt_top50 > 959.1
        + 0.009668338 * max(0.0, 0.007538019 - Q.sum_z_dr2_top20) / 0.002740841   # +1.0%  sum_z_dr2_top20 < 0.007538
        - 0.009178105 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -0.9%  lam1 < 0.005914
        + 0.008266311 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # +0.8%  e3 < 0.0005178
        + 0.007815107 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +0.8%  log_sum_pt < 7.017
        + 0.006834592 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4) / 0.08975043   # +0.7%  z_dr_0p2_0p4 < 0.1292
        + 0.006679628 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +0.7%  n_dr_0p2_0p4 < 15
        + 0.006488792 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +0.6%  psi_0p3 > 0.9956
        + 0.006485246 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # +0.6%  log_sum_pt < 6.989
        - 0.005802559 * max(0.0, 0.002788182 - Q.soft5_z) / 0.001489217   # -0.6%  soft5_z < 0.002788
        + 0.00575241 * max(0.0, 80.4 - Q.mass_top30) / 14.65577   # +0.6%  mass_top30 < 80.4
        + 0.005571763 * max(0.0, 0.005312783 - Q.sum_z_dr2_top20) / 0.001437214   # +0.6%  sum_z_dr2_top20 < 0.005313
        - 0.005318687 * max(0.0, 0.9906378 - Q.z_top50_slots) / 0.004783315   # -0.5%  z_top50_slots < 0.9906
        - 0.005115541 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # -0.5%  mass_top40 < 80.89
        - 0.004343945 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -0.4%  sum_pt < 1013
        + 0.004029385 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +0.4%  e2 < 0.01879
        + 0.003931481 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +0.4%  n_dr_0p2_0p4 < 11
        + 0.003669246 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.02980347 - Q.eta_0) / 0.7431884   # +0.4%  mass < 101 and eta_0 < 0.0298
        - 0.003071294 * max(0.0, 2.744141 - Q.soft10_pt) / 0.8181707   # -0.3%  soft10_pt < 2.744
        + 0.003018115 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +0.3%  lam2 < 0.0006155
        + 0.002623623 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) * max(0.0, 34.0625 - Q.pt_10) / 0.02980095   # +0.3%  mass_over_sum_pt_sq < 0.007873 and pt_10 < 34.06
        + 0.002480279 * max(0.0, 86.4 - Q.sd_mass) / 35.84633   # +0.2%  sd_mass < 86.4
        + 0.00242976 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) * max(0.0, 118.5 - Q.pt_2) / 1.543055   # +0.2%  z_dr_0p2_0p4 < 0.09123 and pt_2 < 118.5
        + 0.002315369 * max(0.0, 1156.659 - Q.sum_pt_top50) * max(0.0, 0.2702923 - Q.soft9_dr) / 17.58105   # +0.2%  sum_pt_top50 < 1157 and soft9_dr < 0.2703
        + 0.002313255 * max(0.0, 0.03843804 - Q.dr_2) / 0.0095582   # +0.2%  dr_2 < 0.03844
        + 0.002303534 * max(0.0, 1156.659 - Q.sum_pt_top50) * max(0.0, 0.1846742 - Q.soft10_dr) / 9.291397   # +0.2%  sum_pt_top50 < 1157 and soft10_dr < 0.1847
        - 0.002116751 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # -0.2%  sj2_dr > 0.2232
        - 0.002108778 * max(0.0, Q.psi_0p1 - 0.9184255) / 0.01693566   # -0.2%  psi_0p1 > 0.9184
        - 0.001468314 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.0223982 - Q.C3) / 0.2900897   # -0.1%  sum_pt < 1013 and C3 < 0.0224
        - 0.001362417 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -0.1%  sum_pt < 972
        - 0.001115056 * max(0.0, 0.05602756 - Q.C2) / 0.009882647   # -0.1%  C2 < 0.05603
        - 0.0009975233 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, Q.pt_4 - 65.1875) / 0.009594495   # -0.1%  psi_0p3 > 0.9956 and pt_4 > 65.19
        + 0.000902265 * max(0.0, Q.z_dr_0p05_0p1 - 0.7108211) / 0.01402576   # +0.1%  z_dr_0p05_0p1 > 0.7108
        - 0.0008579634 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, Q.dr_max_012 - 0.06112084) / 0.0001025737   # -0.1%  psi_0p3 > 0.9956 and dr_max_012 > 0.06112
        + 0.0006859526 * max(0.0, 846.1934 - Q.sum_pt_top20) / 22.83716   # +0.1%  sum_pt_top20 < 846.2
        - 0.0006715157 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 0.2400617 - Q.soft9_dr) / 0.822337   # -0.1%  sum_pt < 972 and soft9_dr < 0.2401
        + 0.0006642884 * max(0.0, 7.017258 - Q.log_sum_pt) * max(0.0, 0.199317 - Q.soft4_dr) / 0.004832394   # +0.1%  log_sum_pt < 7.017 and soft4_dr < 0.1993
        + 0.0006496254 * max(0.0, 0.002788182 - Q.soft5_z) * max(0.0, 30.0 - Q.n_real_top30) / 0.001485805   # +0.1%  soft5_z < 0.002788 and n_real_top30 < 30
        + 0.0005530142 * max(0.0, 0.005913555 - Q.lam1) * max(0.0, 0.03230249 - Q.z_7) / 8.285101e-06   # +0.1%  lam1 < 0.005914 and z_7 < 0.0323
        - 0.0005426676 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.eta_1 - 0.0001021922) / 0.1848682   # -0.1%  mass < 101 and eta_1 > 0.0001022
        + 0.0005355967 * max(0.0, 53.87362 - Q.mass) / 3.56547   # +0.1%  mass < 53.87
        - 0.0004905883 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3) / 0.2515325   # -0.0%  sum_pt < 1013 and M3 < 0.03457
        + 0.0004636616 * max(0.0, 0.05602756 - Q.C2) * max(0.0, Q.phi_10 - 0.0647583) / 3.977301e-05   # +0.0%  C2 < 0.05603 and phi_10 > 0.06476
        - 0.0002530644 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # -0.0%  e2 > 0.05557
        + 0.0002020787 * max(0.0, 86.4 - Q.sd_mass) * max(0.0, Q.soft5_dr - 0.1693667) / 1.884029   # +0.0%  sd_mass < 86.4 and soft5_dr > 0.1694
        + 0.000182794 * max(0.0, Q.psi_0p1 - 0.9184255) * max(0.0, -0.02580643 - Q.eta_4) / 4.137911e-05   # +0.0%  psi_0p1 > 0.9184 and eta_4 < -0.02581
        + 0.0001438841 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, Q.mass_top3 - 23.66386) / 31.48208   # +0.0%  sum_pt < 972 and mass_top3 > 23.66
        + 0.0001419761 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, Q.dr_9 - 0.04680031) / 1.099591   # +0.0%  sum_pt < 1013 and dr_9 > 0.0468
        + 0.0001357773 * max(0.0, Q.M2 - 0.05568888) / 0.01563617   # +0.0%  M2 > 0.05569
        - 7.330282e-05 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 0.05554124 - Q.z_4) / 0.06569978   # -0.0%  sum_pt < 972 and z_4 < 0.05554
        + 5.756375e-05 * max(0.0, 0.007856958 - Q.sum_z_dr2_top30) * max(0.0, Q.ptdr0_4 - 8.939062) / 0.0003207945   # +0.0%  sum_z_dr2_top30 < 0.007857 and ptdr0_4 > 8.939
        + 5.360864e-05 * max(0.0, 858.8262 - Q.sum_pt_top40) / 3.625959   # +0.0%  sum_pt_top40 < 858.8
        - 3.275565e-05 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4) * max(0.0, Q.pair_mass_0_5 - 18.80588) / 0.04566649   # -0.0%  z_dr_0p2_0p4 < 0.1292 and pair_mass_0_5 > 18.81
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 23.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.06891 * (0.06250791
        + 0.07171728 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # +7.2%  log_sum_pt > 6.894
        - 0.06185505 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -6.2%  log_sum_pt < 7.139
        + 0.05991165 * max(0.0, 0.1953848 - Q.tau1) / 0.1097382   # +6.0%  tau1 < 0.1954
        - 0.04997142 * max(0.0, Q.z_top50_slots - 0.9586536) / 0.03427112   # -5.0%  z_top50_slots > 0.9587
        + 0.04815811 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +4.8%  log_sum_pt > 6.91
        - 0.04342973 * max(0.0, 2.275391 - Q.soft1_pt) / 1.652278   # -4.3%  soft1_pt < 2.275
        + 0.04274053 * max(0.0, Q.pt_entropy - 2.07371) / 0.8049786   # +4.3%  pt_entropy > 2.074
        - 0.03722805 * max(0.0, 120.6 - Q.mass) / 38.8279   # -3.7%  mass < 120.6
        - 0.03517441 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # -3.5%  mass_over_sum_pt_sq < 0.009595
        - 0.03092918 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -3.1%  sum_pt_top50 > 959.1
        - 0.02575472 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # -2.6%  sj3_mass1 < 32.5
        + 0.02216541 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # +2.2%  sum_pt_top40 < 1070
        + 0.02195004 * max(0.0, 1.521582 - Q.soft1_pt) / 0.9527349   # +2.2%  soft1_pt < 1.522
        + 0.02194337 * max(0.0, 0.008031986 - Q.sum_z_dr2_top40) / 0.002514593   # +2.2%  sum_z_dr2_top40 < 0.008032
        + 0.02151054 * max(0.0, 0.02412652 - Q.sum_z_dr2_top30) / 0.01613825   # +2.2%  sum_z_dr2_top30 < 0.02413
        + 0.02047093 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +2.0%  n_particles > 38
        + 0.0204102 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.2734   # +2.0%  sum_pt_top2 < 689.2
        - 0.01809188 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2) / 190.5117   # -1.8%  sj3_mass1 < 32.5 and sj3_mass2 < 18.68
        - 0.01552429 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -1.6%  log_sum_pt > 6.959
        - 0.0115576 * max(0.0, 30.26161 - Q.sj2_mass1) / 7.997784   # -1.2%  sj2_mass1 < 30.26
        + 0.01112674 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.1%  z_dr_0p2_0p4 < 0.09123
        - 0.0108071 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.932092   # -1.1%  n_dr_0p2_0p4 < 13
        - 0.01045957 * max(0.0, Q.z_top20_slots - 0.8965411) * max(0.0, 0.03843804 - Q.dr_2) / 0.0005290849   # -1.0%  z_top20_slots > 0.8965 and dr_2 < 0.03844
        + 0.01045045 * max(0.0, 0.04516808 - Q.tau3) / 0.02499032   # +1.0%  tau3 < 0.04517
        + 0.01012907 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +1.0%  sum_pt < 1017
        + 0.009932903 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1) / 1.005598   # +1.0%  n_particles > 38 and dr_1 < 0.1612
        + 0.009820475 * max(0.0, 0.007538019 - Q.sum_z_dr2_top20) / 0.002740841   # +1.0%  sum_z_dr2_top20 < 0.007538
        - 0.009684966 * max(0.0, 31.35938 - Q.pt_9) / 6.538304   # -1.0%  pt_9 < 31.36
        - 0.009210494 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -0.9%  log_sum_pt > 6.989
        - 0.008845657 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -0.9%  n_dr_0p2_0p4 < 7
        - 0.008790494 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt) / 15.05788   # -0.9%  n_particles > 38 and soft1_pt < 2.275
        - 0.00865744 * max(0.0, 47.88842 - Q.mass_top20) / 6.040308   # -0.9%  mass_top20 < 47.89
        + 0.00812705 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0) / 0.570275   # +0.8%  n_particles > 38 and dr_0 < 0.112
        + 0.007251173 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.7%  lam1 < 0.004673
        - 0.007219283 * max(0.0, Q.z_top30_slots - 0.9341838) / 0.03406818   # -0.7%  z_top30_slots > 0.9342
        + 0.007101177 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 39.23115   # +0.7%  mass_top20 < 47.89 and n_real_top40 > 29
        - 0.0066055 * max(0.0, 6.160019 - Q.sj3_mass3) / 2.534872   # -0.7%  sj3_mass3 < 6.16
        - 0.00646319 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.6%  psi_0p3 > 0.998
        + 0.006356654 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2) / 0.0008500795   # +0.6%  z_top30_slots > 0.9342 and C2 < 0.0728
        - 0.006354049 * max(0.0, 7.139296 - Q.log_sum_pt) * max(0.0, 5.351077 - Q.pt1_dr01) / 0.6416883   # -0.6%  log_sum_pt < 7.139 and pt1_dr01 < 5.351
        + 0.006312014 * max(0.0, 120.6 - Q.mass) * max(0.0, 0.03843804 - Q.dr_2) / 0.6524502   # +0.6%  mass < 120.6 and dr_2 < 0.03844
        + 0.006001955 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # +0.6%  LHA < 0.2601
        + 0.005876116 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, Q.sj3_mass2 - 2.12465) / 2353.378   # +0.6%  sum_pt_top2 < 689.2 and sj3_mass2 > 2.125
        - 0.005864159 * max(0.0, 1.521582 - Q.soft1_pt) * max(0.0, 0.1307373 - Q.soft10_abseta) / 0.06026128   # -0.6%  soft1_pt < 1.522 and soft10_abseta < 0.1307
        + 0.005737673 * max(0.0, 0.1416054 - Q.D3) / 0.05511785   # +0.6%  D3 < 0.1416
        + 0.005729725 * max(0.0, Q.z_top20_slots - 0.8965411) / 0.0341307   # +0.6%  z_top20_slots > 0.8965
        + 0.005504209 * max(0.0, 0.007538019 - Q.sum_z_dr2_top20) * max(0.0, Q.eta_1 - -0.04302979) / 0.000120198   # +0.6%  sum_z_dr2_top20 < 0.007538 and eta_1 > -0.04303
        + 0.005246106 * max(0.0, 0.001823079 - Q.sum_z_dr2_top20) / 0.000269929   # +0.5%  sum_z_dr2_top20 < 0.001823
        - 0.005054227 * max(0.0, Q.n_for_90pct - 11.0) / 10.33888   # -0.5%  n_for_90pct > 11
        + 0.004810516 * max(0.0, 0.0006570502 - Q.sum_z_dr2_top5) / 0.0001395474   # +0.5%  sum_z_dr2_top5 < 0.0006571
        + 0.004793916 * max(0.0, 0.001594761 - Q.soft5_z) / 0.0005119901   # +0.5%  soft5_z < 0.001595
        + 0.004641588 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1546554 - Q.dr_9) / 0.7931187   # +0.5%  n_particles > 38 and dr_9 < 0.1547
        - 0.00462207 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.5%  n_dr_0p1_0p2 < 8
        + 0.004602075 * max(0.0, 0.08974974 - Q.sj3_z3) / 0.02798989   # +0.5%  sj3_z3 < 0.08975
        + 0.00425386 * max(0.0, 80.35535 - Q.mass_top50) / 11.1399   # +0.4%  mass_top50 < 80.36
        - 0.004237886 * max(0.0, Q.sum_pt_top5 - 430.75) / 176.7518   # -0.4%  sum_pt_top5 > 430.8
        + 0.004208487 * max(0.0, 12.0 - Q.n_dr_0_0p05) / 3.466761   # +0.4%  n_dr_0_0p05 < 12
        + 0.003938461 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.2313408 - Q.sj3_dr12) / 0.7781115   # +0.4%  n_particles > 38 and sj3_dr12 < 0.2313
        + 0.003517598 * max(0.0, 0.0007894752 - Q.sum_z_dr2_top15) / 9.062109e-05   # +0.4%  sum_z_dr2_top15 < 0.0007895
        + 0.003498057 * max(0.0, Q.n_particles - 38.0) * max(0.0, 57.87349 - Q.mass_top15) / 116.4836   # +0.3%  n_particles > 38 and mass_top15 < 57.87
        + 0.0034972 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793) / 0.1941151   # +0.3%  z_top30_slots > 0.9342 and max_pair_mass > 13.05
        - 0.003434986 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -0.3%  z_dr_0_0p05 > 0.8789
        + 0.00314729 * max(0.0, 0.003270031 - Q.sum_z_dr2_top15) / 0.0007969965   # +0.3%  sum_z_dr2_top15 < 0.00327
        + 0.002803683 * max(0.0, Q.mass_top10 - 56.92192) / 8.071231   # +0.3%  mass_top10 > 56.92
        - 0.00275917 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # -0.3%  M3 < 0.03188
        + 0.002506705 * max(0.0, Q.sum_pt_top30 - 1191.938) / 6.938323   # +0.3%  sum_pt_top30 > 1192
        + 0.001969506 * max(0.0, 42.41192 - Q.mass_top30) / 2.481216   # +0.2%  mass_top30 < 42.41
        - 0.001910083 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # -0.2%  e2 < 0.02516
        - 0.001897789 * max(0.0, 0.02274107 - Q.z_7) / 0.0009068898   # -0.2%  z_7 < 0.02274
        + 0.001870183 * max(0.0, Q.psi_0p1 - 0.8747961) / 0.03440015   # +0.2%  psi_0p1 > 0.8748
        + 0.001855378 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874) / 0.05244257   # +0.2%  z_top30_slots > 0.9342 and ptdr0_3 > 7.408
        + 0.00155994 * max(0.0, 0.0005522528 - Q.sum_z_dr2_top3) / 0.000118253   # +0.2%  sum_z_dr2_top3 < 0.0005523
        - 0.001523423 * max(0.0, 12.0 - Q.n_dr_0_0p05) * max(0.0, 35.6254 - Q.orientation_deg) / 151.9785   # -0.2%  n_dr_0_0p05 < 12 and orientation_deg < 35.63
        + 0.001507494 * max(0.0, 0.3628388 - Q.psi_0p1) / 0.02326895   # +0.2%  psi_0p1 < 0.3628
        + 0.001356982 * max(0.0, Q.sj3_dr_max - 0.3276439) / 0.01855333   # +0.1%  sj3_dr_max > 0.3276
        - 0.001126296 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.1%  log_sum_pt > 7.063
        - 0.001122556 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, Q.soft3_dr0 - 0.02918107) / 57.76427   # -0.1%  sum_pt_top2 < 689.2 and soft3_dr0 > 0.02918
        - 0.001117952 * max(0.0, 30.26161 - Q.sj2_mass1) * max(0.0, 0.2366434 - Q.soft10_dr0) / 1.029958   # -0.1%  sj2_mass1 < 30.26 and soft10_dr0 < 0.2366
        - 0.0009865166 * max(0.0, 0.03187688 - Q.M3) * max(0.0, Q.M2 - 0.05568888) / 0.0001450223   # -0.1%  M3 < 0.03188 and M2 > 0.05569
        - 0.0009504459 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, Q.dr_2 - 0.08525808) / 0.07283314   # -0.1%  pt_9 < 31.36 and dr_2 > 0.08526
        - 0.0009232636 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 0.02980347 - Q.eta_0) / 5.83951   # -0.1%  sum_pt_top5 > 430.8 and eta_0 < 0.0298
        - 0.0009011772 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 0.3241858 - Q.dr1_12) / 1.371943   # -0.1%  pt_9 < 31.36 and dr1_12 < 0.3242
        + 0.0008532076 * max(0.0, 0.003468228 - Q.C2_b2) / 0.0005968599   # +0.1%  C2_b2 < 0.003468
        + 0.0007308714 * max(0.0, 0.03187688 - Q.M3) * max(0.0, 1.0 - Q.psi_0p3) / 0.000198055   # +0.1%  M3 < 0.03188 and psi_0p3 < 1
        + 0.0006242609 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_4 - 6.984554) / 0.03731215   # +0.1%  z_top30_slots > 0.9342 and ptdr0_4 > 6.985
        - 0.000622051 * max(0.0, 0.009068077 - Q.tau3) / 0.0001836134   # -0.1%  tau3 < 0.009068
        - 0.0006208154 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.002968979 - Q.soft10_abseta) / 0.000988223   # -0.1%  n_dr_0p2_0p4 < 7 and soft10_abseta < 0.002969
        - 0.0005278095 * Q.absphi_0 / 0.03295052   # -0.1%  absphi_0
        + 0.0004763408 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 11.29505 - Q.pair_mass_0_13) / 38.90723   # +0.0%  pt_9 < 31.36 and pair_mass_0_13 < 11.3
        + 0.0004676754 * max(0.0, 0.02412652 - Q.sum_z_dr2_top30) * max(0.0, Q.zdr_14 - 0.0025721) / 1.011019e-06   # +0.0%  sum_z_dr2_top30 < 0.02413 and zdr_14 > 0.002572
        - 0.0004475715 * max(0.0, 1.788105 - Q.D2) / 0.2865857   # -0.0%  D2 < 1.788
        + 0.0003545935 * max(0.0, 0.003270031 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 6.08612e-07   # +0.0%  sum_z_dr2_top15 < 0.00327 and psi_0p3 > 0.9974
        + 0.0002649841 * max(0.0, Q.N2 - 0.4678622) / 0.001556028   # +0.0%  N2 > 0.4679
        + 0.0002304214 * max(0.0, 85.41209 - Q.mass_top10) / 39.98104   # +0.0%  mass_top10 < 85.41
        - 0.0002118805 * max(0.0, Q.pt_entropy - 2.07371) * max(0.0, 0.3797 - Q.soft3_dr) / 0.1539306   # -0.0%  pt_entropy > 2.074 and soft3_dr < 0.3797
        - 0.000211644 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 0.9624339 - Q.tau43) / 39.08134   # -0.0%  sum_pt_top2 < 689.2 and tau43 < 0.9624
        + 0.0001628104 * max(0.0, Q.M2 - 0.1011005) / 0.001541556   # +0.0%  M2 > 0.1011
        - 0.0001265096 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # -0.0%  e2 > 0.06524
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 6.638;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.638163 * (0.06302112
        - 0.1690982 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -16.9%  mass < 92.86
        - 0.08772723 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # -8.8%  sum_pt < 1261
        + 0.07811099 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.sum_z_dr2_top15) / 0.0009227558   # +7.8%  log_sum_pt > 6.903 and sum_z_dr2_top15 < 0.02147
        + 0.07106661 * max(0.0, 91.19 - Q.mass) / 16.46423   # +7.1%  mass < 91.19
        + 0.06657291 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # +6.7%  sum_pt > 1017
        - 0.06398293 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.sum_z_dr2_top15) / 1.045047   # -6.4%  sum_pt_top50 > 988.5 and sum_z_dr2_top15 < 0.02147
        + 0.06247897 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +6.2%  mass_over_sum_pt < 0.09795
        + 0.04941797 * max(0.0, 1048.098 - Q.sum_pt_top50) / 44.36839   # +4.9%  sum_pt_top50 < 1048
        - 0.04018447 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # -4.0%  sum_pt_top50 < 1079
        - 0.03541399 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -3.5%  log_sum_pt > 6.959
        - 0.02453793 * max(0.0, 1018.698 - Q.sum_pt_top40) / 34.89083   # -2.5%  sum_pt_top40 < 1019
        - 0.02193411 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003463934   # -2.2%  log_sum_pt > 6.903 and psi_0p3 > 0.9299
        - 0.02172398 * max(0.0, 996.8867 - Q.sum_pt_top30) / 42.94346   # -2.2%  sum_pt_top30 < 996.9
        + 0.0215415 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # +2.2%  mass_top50 < 92.17
        - 0.01894797 * max(0.0, Q.sum_pt_top40 - 984.7009) / 58.19535   # -1.9%  sum_pt_top40 > 984.7
        + 0.01847129 * max(0.0, 1041.263 - Q.sum_pt_top40) / 49.16087   # +1.8%  sum_pt_top40 < 1041
        + 0.0151611 * max(0.0, 82.66587 - Q.mass_top30) / 15.96638   # +1.5%  mass_top30 < 82.67
        - 0.01484624 * max(0.0, 0.006363916 - Q.sum_z_dr2_top30) / 0.001678624   # -1.5%  sum_z_dr2_top30 < 0.006364
        - 0.01422721 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # -1.4%  sum_pt > 1116
        + 0.01408108 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +1.4%  lam1 < 0.01174
        + 0.01258864 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +1.3%  sum_pt_top40 > 1070
        + 0.0120632 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +1.2%  mass < 78.26
        + 0.009617104 * max(0.0, 966.0633 - Q.sum_pt_top30) / 29.99923   # +1.0%  sum_pt_top30 < 966.1
        - 0.006139676 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # -0.6%  psi_0p3 > 0.9974
        - 0.005525336 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 0.397021 - Q.max_dr) / 11.85447   # -0.6%  sum_pt < 1261 and max_dr < 0.397
        - 0.004412228 * max(0.0, 51.0 - Q.n_particles) / 9.158477   # -0.4%  n_particles < 51
        - 0.004217707 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.mass_top40 - 77.93668) / 1.109603   # -0.4%  log_sum_pt > 6.903 and mass_top40 > 77.94
        + 0.004023717 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 30.0 - Q.n_real_top30) / 223.0591   # +0.4%  sum_pt < 1261 and n_real_top30 < 30
        - 0.003744786 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.4%  log_sum_pt > 7.063
        + 0.003565102 * max(0.0, Q.log_sum_pt - 6.903423) / 0.05662275   # +0.4%  log_sum_pt > 6.903
        + 0.003240167 * max(0.0, Q.z_top20_slots - 0.7516206) / 0.1429779   # +0.3%  z_top20_slots > 0.7516
        + 0.003086293 * max(0.0, Q.sj3_pair_mass_min - 18.02031) / 11.34956   # +0.3%  sj3_pair_mass_min > 18.02
        - 0.003058263 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 1.433504e-05 - Q.e3) / 1.867416e-07   # -0.3%  log_sum_pt > 6.903 and e3 < 1.434e-05
        - 0.002920253 * max(0.0, Q.sum_pt_top50 - 988.4554) / 62.81258   # -0.3%  sum_pt_top50 > 988.5
        + 0.002060257 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.01585274 - Q.z_10) / 5.217933e-05   # +0.2%  log_sum_pt > 6.903 and z_10 < 0.01585
        + 0.001533927 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 4.657698e-07   # +0.2%  sum_pt < 972 and e4 < 5.851e-08
        + 0.001521525 * max(0.0, Q.sum_pt - 1017.435) * max(0.0, Q.C2 - 0.02207346) / 1.810321   # +0.2%  sum_pt > 1017 and C2 > 0.02207
        - 0.001483422 * max(0.0, 0.004763596 - Q.sum_z_dr2_top30) / 0.0009958273   # -0.1%  sum_z_dr2_top30 < 0.004764
        + 0.001017644 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, 0.01898316 - Q.z_10) / 1.422215e-05   # +0.1%  log_sum_pt > 7.063 and z_10 < 0.01898
        + 0.0008295331 * max(0.0, Q.sum_pt_top20 - 1129.275) / 6.499168   # +0.1%  sum_pt_top20 > 1129
        + 0.0005311681 * max(0.0, 1048.098 - Q.sum_pt_top50) * max(0.0, 0.3742561 - Q.max_dr) / 1.246515   # +0.1%  sum_pt_top50 < 1048 and max_dr < 0.3743
        + 0.000524195 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.soft5_z - 0.0005078411) / 4.845388e-05   # +0.1%  log_sum_pt > 6.903 and soft5_z > 0.0005078
        + 0.0005018998 * max(0.0, Q.z_top20_slots - 0.7516206) * max(0.0, Q.sj3_mass1 - 6.519856) / 0.9623658   # +0.1%  z_top20_slots > 0.7516 and sj3_mass1 > 6.52
        - 0.0004313957 * max(0.0, Q.sum_pt_top50 - 1156.659) / 13.71464   # -0.0%  sum_pt_top50 > 1157
        + 0.0003956369 * max(0.0, 996.8867 - Q.sum_pt_top30) * max(0.0, 0.03275811 - Q.soft7_dr) / 0.03631807   # +0.0%  sum_pt_top30 < 996.9 and soft7_dr < 0.03276
        - 0.0002946706 * max(0.0, 56.53125 - Q.pt_6) / 17.03571   # -0.0%  pt_6 < 56.53
        - 0.0002521083 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -0.0%  sum_pt < 972
        - 0.0002483293 * max(0.0, Q.sum_pt - 1115.723) * max(0.0, Q.mean_phi - 0.0004404746) / 0.001864981   # -0.0%  sum_pt > 1116 and mean_phi > 0.0004405
        + 0.0002457187 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.sum_z_dr2_top30 - 0.008376291) / 1.929368e-05   # +0.0%  log_sum_pt > 7.063 and sum_z_dr2_top30 > 0.008376
        - 0.0002099429 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.08734131) / 0.003497383   # -0.0%  sum_pt_top20 > 1129 and eta_1 > 0.08734
        + 0.0001157855 * max(0.0, Q.sum_pt - 1115.723) * max(0.0, -0.1135254 - Q.eta_5) / 0.03156393   # +0.0%  sum_pt > 1116 and eta_5 < -0.1135
        + 7.368454e-05 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, 1.0 - Q.n_dr_0_0p05) / 0.1096012   # +0.0%  sum_pt_top20 > 1129 and n_dr_0_0p05 < 1
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 3.969;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.969051 * (-0.01063561
        - 0.1408424 * max(0.0, 0.009614971 - Q.sum_z_dr2) / 0.003502545   # -14.1%  sum_z_dr2 < 0.009615
        + 0.07992801 * max(0.0, 0.008840538 - Q.sum_z_dr2_top40) / 0.003108622   # +8.0%  sum_z_dr2_top40 < 0.008841
        + 0.04954864 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +5.0%  n_particles < 46
        + 0.04857891 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +4.9%  mass < 92.86
        + 0.04003731 * max(0.0, 0.008124776 - Q.sum_z_dr2_top50) / 0.002470567   # +4.0%  sum_z_dr2_top50 < 0.008125
        + 0.03996725 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741) / 0.0491674   # +4.0%  n_dr_0p2_0p4 < 8 and z_top50_slots > 0.9787
        - 0.03751139 * max(0.0, 80.4 - Q.mass_top40) / 12.19371   # -3.8%  mass_top40 < 80.4
        - 0.02936292 * max(0.0, 2.410481 - Q.D2) / 0.587301   # -2.9%  D2 < 2.41
        + 0.02422402 * max(0.0, 0.009614971 - Q.sum_z_dr2) * max(0.0, 23.45965 - Q.sj3_mass1) / 0.03718818   # +2.4%  sum_z_dr2 < 0.009615 and sj3_mass1 < 23.46
        + 0.02362738 * max(0.0, 0.009614971 - Q.sum_z_dr2) * max(0.0, 11.91094 - Q.sj3_mass2) / 0.01964083   # +2.4%  sum_z_dr2 < 0.009615 and sj3_mass2 < 11.91
        - 0.02246907 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -2.2%  tau21 < 0.3862
        + 0.02124288 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732) / 1477.074   # +2.1%  n_particles < 46 and sum_pt_top30 > 800.7
        + 0.02077464 * max(0.0, 0.08665515 - Q.mass_over_sum_pt) / 0.01542259   # +2.1%  mass_over_sum_pt < 0.08666
        - 0.02008432 * max(0.0, 0.006363916 - Q.sum_z_dr2_top30) / 0.001678624   # -2.0%  sum_z_dr2_top30 < 0.006364
        + 0.01955477 * max(0.0, 2.410481 - Q.D2) * max(0.0, 5.142486 - Q.D2_b2) / 2.680544   # +2.0%  D2 < 2.41 and D2_b2 < 5.142
        + 0.0188263 * max(0.0, 27.56535 - Q.sj2_mass1) / 6.309897   # +1.9%  sj2_mass1 < 27.57
        + 0.01853443 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.psi_0p3 - 0.9924477) / 0.007646232   # +1.9%  n_dr_0p2_0p4 < 5 and psi_0p3 > 0.9924
        - 0.01759718 * max(0.0, 79.21004 - Q.mass_top50) / 10.65754   # -1.8%  mass_top50 < 79.21
        - 0.01450069 * max(0.0, 0.019523 - Q.z_dr_0p2_0p4) / 0.007664401   # -1.5%  z_dr_0p2_0p4 < 0.01952
        - 0.01358165 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, 18.0 - Q.n_dr_0_0p05) / 19.28245   # -1.4%  n_dr_0p2_0p4 < 8 and n_dr_0_0p05 < 18
        + 0.01200121 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.9313699) / 0.1032909   # +1.2%  n_dr_0p1_0p2 < 10 and psi_0p2 > 0.9314
        + 0.01189139 * max(0.0, 0.009614971 - Q.sum_z_dr2) * max(0.0, 0.02980347 - Q.eta_0) / 0.0001109857   # +1.2%  sum_z_dr2 < 0.009615 and eta_0 < 0.0298
        + 0.01173717 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) / 2.171439   # +1.2%  n_dr_0p1_0p2 < 10
        - 0.01069146 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349) / 64.17204   # -1.1%  n_particles < 46 and mass_top15 > 57.87
        + 0.01025961 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +1.0%  n_dr_0p2_0p4 < 5
        + 0.009633944 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # +1.0%  psi_0p3 > 0.998
        + 0.009538533 * max(0.0, 86.4 - Q.mass) / 13.67632   # +1.0%  mass < 86.4
        + 0.009512421 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +1.0%  lam2 < 0.0006155
        + 0.008358341 * max(0.0, Q.psi_0p2 - 0.9985434) / 0.00014398   # +0.8%  psi_0p2 > 0.9985
        + 0.00788247 * max(0.0, 27.56535 - Q.sj2_mass1) * max(0.0, 0.06334002 - Q.dr_10) / 0.09789943   # +0.8%  sj2_mass1 < 27.57 and dr_10 < 0.06334
        - 0.007868575 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -0.8%  mass < 53.87
        + 0.007822226 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, Q.n_dr_0_0p05 - 9.0) / 0.03881192   # +0.8%  z_top30_slots > 0.9735 and n_dr_0_0p05 > 9
        - 0.00770453 * max(0.0, 0.0001140644 - Q.sum_z_dr2_top5) / 1.008187e-05   # -0.8%  sum_z_dr2_top5 < 0.0001141
        - 0.007342009 * max(0.0, 0.003235754 - Q.zdr_0) / 0.0003842694   # -0.7%  zdr_0 < 0.003236
        - 0.007104943 * max(0.0, Q.psi_0p3 - 0.9995915) / 8.716828e-05   # -0.7%  psi_0p3 > 0.9996
        + 0.006751216 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +0.7%  tau4 < 0.01627
        + 0.006328137 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # +0.6%  n_dr_0p2_0p4 < 8
        - 0.005745632 * max(0.0, 0.02374759 - Q.tau3) / 0.006500707   # -0.6%  tau3 < 0.02375
        + 0.005728515 * max(0.0, 0.009614971 - Q.sum_z_dr2) * max(0.0, 0.002076054 - Q.phi_0) / 3.173958e-05   # +0.6%  sum_z_dr2 < 0.009615 and phi_0 < 0.002076
        - 0.005683581 * max(0.0, 0.003235754 - Q.zdr_0) * max(0.0, 0.331314 - Q.sj3_z2) / 5.366568e-05   # -0.6%  zdr_0 < 0.003236 and sj3_z2 < 0.3313
        - 0.005620659 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03447112 - Q.z_10) / 0.01282533   # -0.6%  n_dr_0p2_0p4 < 5 and z_10 < 0.03447
        + 0.005414563 * max(0.0, 11.0 - Q.n_for_90pct) / 0.4757244   # +0.5%  n_for_90pct < 11
        - 0.005017602 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0) / 3.186294   # -0.5%  n_dr_0p2_0p4 < 5 and n_dr_0p1_0p2 > 9
        - 0.004854205 * max(0.0, Q.z_top30_slots - 0.9734886) / 0.01006171   # -0.5%  z_top30_slots > 0.9735
        + 0.00481242 * max(0.0, 27.56535 - Q.sj2_mass1) * max(0.0, 0.1455078 - Q.phi_13) / 0.9329567   # +0.5%  sj2_mass1 < 27.57 and phi_13 < 0.1455
        - 0.004578624 * max(0.0, 0.008840538 - Q.sum_z_dr2_top40) * max(0.0, Q.soft5_pt - 1.652344) / 0.0008509678   # -0.5%  sum_z_dr2_top40 < 0.008841 and soft5_pt > 1.652
        + 0.004426254 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.eccentricity - 0.7333655) / 0.1710102   # +0.4%  n_dr_0p2_0p4 < 5 and eccentricity > 0.7334
        + 0.004393558 * max(0.0, 0.0006154841 - Q.lam2) * max(0.0, 1.0 - Q.n_dr_0p4_up) / 0.0001383002   # +0.4%  lam2 < 0.0006155 and n_dr_0p4_up < 1
        - 0.004022183 * max(0.0, 79.21004 - Q.mass_top50) * max(0.0, Q.mass_top3 - 8.921413) / 4.657205   # -0.4%  mass_top50 < 79.21 and mass_top3 > 8.921
        + 0.003962312 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.phi_5 - -0.03793335) / 0.09242684   # +0.4%  n_dr_0p1_0p2 < 10 and phi_5 > -0.03793
        + 0.003948178 * max(0.0, 0.2738063 - Q.max_dr) / 0.009268927   # +0.4%  max_dr < 0.2738
        - 0.003628204 * max(0.0, 0.002582316 - Q.sum_z_dr2_top30) / 0.0003518773   # -0.4%  sum_z_dr2_top30 < 0.002582
        - 0.003566366 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.eta_1 - -0.04302979) / 3.206471e-05   # -0.4%  psi_0p3 > 0.998 and eta_1 > -0.04303
        - 0.003542755 * max(0.0, 1.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.n_dr_0p4_up) / 0.05515294   # -0.4%  n_dr_0p2_0p4 < 1 and n_dr_0p4_up < 1
        + 0.003514572 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1) / 0.1325294   # +0.4%  n_dr_0p2_0p4 < 5 and psi_0p1 < 0.9538
        - 0.003462575 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, 0.002210911 - Q.soft10_z) / 2.961306e-05   # -0.3%  tau21 < 0.3862 and soft10_z < 0.002211
        + 0.003224766 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.002788182 - Q.soft5_z) / 0.00316606   # +0.3%  n_dr_0p2_0p4 < 8 and soft5_z < 0.002788
        + 0.003034259 * max(0.0, 9.256377 - Q.sj3_mass1) / 0.6187012   # +0.3%  sj3_mass1 < 9.256
        - 0.003010681 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, Q.z_dr_0p05_0p1 - 0.2992503) / 0.001171475   # -0.3%  z_top30_slots > 0.9735 and z_dr_0p05_0p1 > 0.2993
        - 0.002990391 * max(0.0, 46.0 - Q.n_particles) * max(0.0, 43.66338 - Q.D2_b2) / 195.6286   # -0.3%  n_particles < 46 and D2_b2 < 43.66
        + 0.002977481 * max(0.0, 0.0006154841 - Q.lam2) * max(0.0, -9.840088 - Q.orientation_deg) / 0.002640902   # +0.3%  lam2 < 0.0006155 and orientation_deg < -9.84
        - 0.002934454 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243) / 0.0001457263   # -0.3%  tau21 < 0.3862 and lam1 > 0.007671
        + 0.002723478 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 0.1269531 - Q.eta_9) / 0.001302695   # +0.3%  z_top30_slots > 0.9735 and eta_9 < 0.127
        + 0.002525058 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421) / 0.000406438   # +0.3%  D2 < 2.41 and psi_0p3 > 0.9985
        - 0.002381644 * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.0585395   # -0.2%  n_dr_0p2_0p4 < 1
        - 0.00229232 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5) / 0.004484684   # -0.2%  n_dr_0p2_0p4 < 5 and zdr_5 < 0.007099
        - 0.002186694 * max(0.0, 0.02227783 - Q.absphi_1) / 0.006894148   # -0.2%  absphi_1 < 0.02228
        + 0.002162625 * max(0.0, 0.02374759 - Q.tau3) * max(0.0, Q.sj3_dr13 - 0.2870549) / 4.105435e-05   # +0.2%  tau3 < 0.02375 and sj3_dr13 > 0.2871
        + 0.002060213 * max(0.0, 0.0006154841 - Q.lam2) * max(0.0, 0.1309949 - Q.soft10_dr) / 9.534011e-06   # +0.2%  lam2 < 0.0006155 and soft10_dr < 0.131
        - 0.001745027 * max(0.0, Q.psi_0p3 - 0.9995915) * max(0.0, Q.mass_top2 - 16.30802) / 0.0003241301   # -0.2%  psi_0p3 > 0.9996 and mass_top2 > 16.31
        + 0.001674733 * max(0.0, 2.410481 - Q.D2) * max(0.0, 0.136982 - Q.soft5_dr) / 0.01829419   # +0.2%  D2 < 2.41 and soft5_dr < 0.137
        - 0.001564544 * max(0.0, 79.21004 - Q.mass_top50) * max(0.0, Q.ptdr0_2 - 7.740999) / 0.4958331   # -0.2%  mass_top50 < 79.21 and ptdr0_2 > 7.741
        + 0.001512346 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_pt_top40 - 1095.686) / 42.63504   # +0.2%  n_dr_0p2_0p4 < 8 and sum_pt_top40 > 1096
        - 0.001329616 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, 0.004577637 - Q.soft9_abseta) / 3.67766e-07   # -0.1%  psi_0p3 > 0.998 and soft9_abseta < 0.004578
        + 0.001327458 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03329468 - Q.eta_3) / 0.04612469   # +0.1%  n_dr_0p2_0p4 < 5 and eta_3 < 0.03329
        + 0.001320536 * max(0.0, 0.006395224 - Q.tau4) / 6.609081e-05   # +0.1%  tau4 < 0.006395
        + 0.001219908 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.max_dr - 0.4357228) / 0.004818361   # +0.1%  D2 < 2.41 and max_dr > 0.4357
        + 0.001200298 * max(0.0, 0.04748396 - Q.z_dr_0p1_0p2) / 0.01200196   # +0.1%  z_dr_0p1_0p2 < 0.04748
        - 0.001156733 * max(0.0, 0.0036273 - Q.soft5_absphi) / 0.000134249   # -0.1%  soft5_absphi < 0.003627
        + 0.001108061 * max(0.0, 0.008840538 - Q.sum_z_dr2_top40) * max(0.0, Q.ptdr0_6 - 5.648151) / 0.0006378931   # +0.1%  sum_z_dr2_top40 < 0.008841 and ptdr0_6 > 5.648
        + 0.001073569 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 988.4375 - Q.sum_pt_top30) / 17.46027   # +0.1%  n_dr_0p2_0p4 < 5 and sum_pt_top30 < 988.4
        - 0.001073323 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, 0.254323 - Q.sd_zg) / 2.332676e-05   # -0.1%  psi_0p3 > 0.998 and sd_zg < 0.2543
        + 0.0009557412 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.1225688 - Q.D3) / 0.02400118   # +0.1%  n_dr_0p2_0p4 < 5 and D3 < 0.1226
        - 0.0009390219 * max(0.0, Q.psi_0p3 - 0.9995915) * max(0.0, Q.eta_13 - 0.07098389) / 4.893698e-07   # -0.1%  psi_0p3 > 0.9996 and eta_13 > 0.07098
        - 0.0008560223 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, 0.004962409 - Q.soft10_z) / 1.385853e-06   # -0.1%  psi_0p3 > 0.998 and soft10_z < 0.004962
        - 0.0007577245 * max(0.0, 40.2 - Q.mass) / 1.359187   # -0.1%  mass < 40.2
        - 0.0007178449 * max(0.0, 27.56535 - Q.sj2_mass1) * max(0.0, Q.sj3_mass3 - 0.8123034) / 14.63582   # -0.1%  sj2_mass1 < 27.57 and sj3_mass3 > 0.8123
        + 0.0006937873 * max(0.0, 0.001260456 - Q.lam1) * max(0.0, 1001.523 - Q.sum_pt_top40) / 0.001041465   # +0.1%  lam1 < 0.00126 and sum_pt_top40 < 1002
        + 0.0006378075 * max(0.0, 0.009614971 - Q.sum_z_dr2) * max(0.0, Q.pair_mass_0_2 - 32.99439) / 0.0002801011   # +0.1%  sum_z_dr2 < 0.009615 and pair_mass_0_2 > 32.99
        + 0.0004994683 * max(0.0, 0.001260456 - Q.lam1) / 8.753718e-05   # +0.0%  lam1 < 0.00126
        + 0.0004938504 * max(0.0, Q.psi_0p2 - 0.9985434) * max(0.0, 0.05004883 - Q.absphi_4) / 2.395711e-06   # +0.0%  psi_0p2 > 0.9985 and absphi_4 < 0.05005
        + 0.0004447191 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, -0.05975342 - Q.eta_8) / 6.890271e-05   # +0.0%  z_top30_slots > 0.9735 and eta_8 < -0.05975
        + 0.0004072204 * max(0.0, Q.psi_0p2 - 0.9985434) * max(0.0, 0.08575439 - Q.abseta_4) / 6.302321e-06   # +0.0%  psi_0p2 > 0.9985 and abseta_4 < 0.08575
        + 0.0003841273 * max(0.0, 0.0006154841 - Q.lam2) * max(0.0, Q.soft8_dr - 0.04226709) / 1.028625e-05   # +0.0%  lam2 < 0.0006155 and soft8_dr > 0.04227
        - 0.0003066901 * max(0.0, 0.0006154841 - Q.lam2) * max(0.0, 525.5 - Q.pt_0) / 0.03566195   # -0.0%  lam2 < 0.0006155 and pt_0 < 525.5
        - 0.0002770566 * max(0.0, 0.2738063 - Q.max_dr) * max(0.0, Q.phi_14 - 0.102478) / 1.293714e-05   # -0.0%  max_dr < 0.2738 and phi_14 > 0.1025
        - 0.0002460354 * max(0.0, 0.006026828 - Q.sum_z_dr2_top40) / 0.001351778   # -0.0%  sum_z_dr2_top40 < 0.006027
        + 0.0002370281 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, 0.0144043 - Q.soft3_absphi) / 5.318291e-05   # +0.0%  tau21 < 0.3862 and soft3_absphi < 0.0144
        - 0.0001781545 * max(0.0, 0.2738063 - Q.max_dr) * max(0.0, Q.eta_11 - 0.1364746) / 3.092542e-06   # -0.0%  max_dr < 0.2738 and eta_11 > 0.1365
        - 1.24077e-05 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.soft4_dr - 0.08815263) / 3.958279e-05   # -0.0%  psi_0p3 > 0.998 and soft4_dr > 0.08815
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 11.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.2934 * (0.03330697
        - 0.0716333 * max(0.0, 78.26182 - Q.mass) / 9.857173   # -7.2%  mass < 78.26
        + 0.06732063 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +6.7%  mass < 101
        + 0.04857476 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +4.9%  mass < 74.25
        + 0.04490971 * max(0.0, 143.7876 - Q.mass) / 57.99337   # +4.5%  mass < 143.8
        - 0.04384851 * max(0.0, 86.4 - Q.mass) / 13.67632   # -4.4%  mass < 86.4
        + 0.04138073 * max(0.0, 120.6 - Q.mass) / 38.8279   # +4.1%  mass < 120.6
        - 0.03972772 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -4.0%  mass_top40 < 163.3
        + 0.03884202 * max(0.0, Q.sum_zz_dr2 - 0.009606007) / 0.003182984   # +3.9%  sum_zz_dr2 > 0.009606
        - 0.03779896 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -3.8%  mass < 92.86
        + 0.03511164 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +3.5%  n_dr_0p2_0p4 < 26
        - 0.02766224 * max(0.0, 91.19 - Q.mass_top15) / 34.88803   # -2.8%  mass_top15 < 91.19
        - 0.02281697 * max(0.0, Q.lam1 - 0.008241985) / 0.002595658   # -2.3%  lam1 > 0.008242
        - 0.02190422 * max(0.0, 91.69753 - Q.mass_top30) / 22.0127   # -2.2%  mass_top30 < 91.7
        + 0.02038725 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt) / 36.47388   # +2.0%  n_particles > 22 and soft1_pt < 2.275
        - 0.01996259 * max(0.0, 0.05660088 - Q.sum_z_dr) / 0.01080382   # -2.0%  sum_z_dr < 0.0566
        - 0.01884359 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -1.9%  mass_top40 < 83.33
        - 0.0188372 * max(0.0, Q.n_particles - 22.0) / 23.9717   # -1.9%  n_particles > 22
        - 0.0178884 * max(0.0, 0.004855289 - Q.sum_z_dr2_top15) / 0.001418414   # -1.8%  sum_z_dr2_top15 < 0.004855
        + 0.0177912 * max(0.0, 101.0497 - Q.mass) * max(0.0, 3.852812 - Q.D2_b2) / 25.01551   # +1.8%  mass < 101 and D2_b2 < 3.853
        - 0.01629713 * max(0.0, Q.sum_z_dr2_top15 - 0.00727763) / 0.002923446   # -1.6%  sum_z_dr2_top15 > 0.007278
        + 0.01493382 * max(0.0, 0.07031778 - Q.sum_z_dr) / 0.01719521   # +1.5%  sum_z_dr < 0.07032
        + 0.01492367 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.005180665 - Q.sum_z_dr2_top2) / 0.07860143   # +1.5%  mass < 92.86 and sum_z_dr2_top2 < 0.005181
        + 0.01428728 * max(0.0, 58.0 - Q.n_pt_above_1) / 17.32351   # +1.4%  n_pt_above_1 < 58
        + 0.01226826 * max(0.0, 26.0 - Q.n_dr_0p1_0p2) / 14.20854   # +1.2%  n_dr_0p1_0p2 < 26
        + 0.01209189 * max(0.0, 57.87349 - Q.mass_top15) / 12.46085   # +1.2%  mass_top15 < 57.87
        + 0.01070505 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +1.1%  psi_0p3 > 0.9974
        - 0.01057667 * max(0.0, 80.78464 - Q.mass) / 10.84952   # -1.1%  mass < 80.78
        + 0.01028161 * max(0.0, 0.9909875 - Q.psi_0p1) / 0.2210377   # +1.0%  psi_0p1 < 0.991
        + 0.009572647 * max(0.0, Q.n_pt_above_10 - 11.0) / 8.944079   # +1.0%  n_pt_above_10 > 11
        + 0.009316753 * max(0.0, 69.02716 - Q.mass_top15) / 18.23029   # +0.9%  mass_top15 < 69.03
        + 0.009289259 * max(0.0, 120.6 - Q.mass_top30) / 45.45988   # +0.9%  mass_top30 < 120.6
        + 0.008787749 * max(0.0, Q.e2 - 0.02793599) / 0.008664879   # +0.9%  e2 > 0.02794
        - 0.008522301 * max(0.0, Q.psi_0p2 - 0.9734513) / 0.0116938   # -0.9%  psi_0p2 > 0.9735
        + 0.008487091 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.36624 - Q.D2_b2) / 35.56485   # +0.8%  n_dr_0p2_0p4 < 15 and D2_b2 < 7.366
        + 0.008016911 * max(0.0, 0.1900123 - Q.D3) / 0.08663362   # +0.8%  D3 < 0.19
        + 0.007656715 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +0.8%  n_dr_0p2_0p4 < 15
        + 0.007647387 * max(0.0, Q.sum_z_dr2_top15 - 0.01563836) / 0.001334556   # +0.8%  sum_z_dr2_top15 > 0.01564
        + 0.007360327 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.n_dr_0_0p05 - 10.0) / 143.1065   # +0.7%  n_particles > 22 and n_dr_0_0p05 > 10
        - 0.007094337 * max(0.0, 80.78464 - Q.mass) * max(0.0, 3.852812 - Q.D2_b2) / 3.983646   # -0.7%  mass < 80.78 and D2_b2 < 3.853
        - 0.006452296 * max(0.0, Q.sj2_dr - 0.2412757) / 0.01833576   # -0.6%  sj2_dr > 0.2413
        - 0.006358288 * max(0.0, 0.06030419 - Q.mass_over_sum_pt) / 0.005383371   # -0.6%  mass_over_sum_pt < 0.0603
        + 0.005893795 * max(0.0, Q.sj3_pair_mass_min - 32.51366) / 5.88267   # +0.6%  sj3_pair_mass_min > 32.51
        - 0.005683751 * max(0.0, Q.log_sum_pt - 6.97212) / 0.02498281   # -0.6%  log_sum_pt > 6.972
        - 0.005677329 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 3.014827 - Q.D2_b2) / 4.336836   # -0.6%  mass_top40 < 83.33 and D2_b2 < 3.015
        + 0.005652228 * max(0.0, Q.sum_zz_dr2 - 0.01396296) / 0.002220692   # +0.6%  sum_zz_dr2 > 0.01396
        - 0.005395247 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -0.5%  sum_z_dr > 0.1207
        + 0.005319557 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.5%  sj2_dr > 0.2232
        - 0.004721476 * max(0.0, Q.n_particles - 22.0) * max(0.0, 3.852812 - Q.D2_b2) / 43.35921   # -0.5%  n_particles > 22 and D2_b2 < 3.853
        + 0.004432538 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +0.4%  lam1 > 0.01174
        + 0.004417474 * max(0.0, 0.03577037 - Q.sum_z_dr) / 0.004182456   # +0.4%  sum_z_dr < 0.03577
        + 0.004416361 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +0.4%  mass_top30 < 60.44
        - 0.004299652 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 13.0 - Q.n_dr_0_0p05) / 0.003836025   # -0.4%  psi_0p3 > 0.9974 and n_dr_0_0p05 < 13
        + 0.003869494 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +0.4%  mass_top40 < 67.73
        - 0.003779955 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 20.40995 - Q.D2_b2) / 0.3948355   # -0.4%  sj2_dr > 0.2232 and D2_b2 < 20.41
        - 0.003775966 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.max_dr - 0.1939977) / 1.031549   # -0.4%  n_dr_0p2_0p4 < 15 and max_dr > 0.194
        + 0.003668093 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sj3_dr13 - 0.1348442) / 0.3099271   # +0.4%  n_dr_0p2_0p4 < 15 and sj3_dr13 > 0.1348
        + 0.003462812 * max(0.0, Q.e3 - 0.0001841806) / 4.035159e-05   # +0.3%  e3 > 0.0001842
        - 0.003411782 * max(0.0, 0.004855289 - Q.sum_z_dr2_top15) * max(0.0, 0.1645959 - Q.D3) / 0.0001692448   # -0.3%  sum_z_dr2_top15 < 0.004855 and D3 < 0.1646
        - 0.003354439 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.z_top50_slots) / 0.03204382   # -0.3%  n_dr_0p2_0p4 < 15 and z_top50_slots < 1
        - 0.002554598 * max(0.0, Q.sum_z_dr2_top5 - 0.007164202) / 0.002334697   # -0.3%  sum_z_dr2_top5 > 0.007164
        - 0.00252543 * max(0.0, Q.sd_rg - 0.2042612) / 0.02046744   # -0.3%  sd_rg > 0.2043
        + 0.002380757 * max(0.0, 0.002227078 - Q.zdr_12) / 0.0009859068   # +0.2%  zdr_12 < 0.002227
        - 0.002286456 * max(0.0, Q.n_particles - 22.0) * max(0.0, 33.21875 - Q.pt_7) / 60.66924   # -0.2%  n_particles > 22 and pt_7 < 33.22
        - 0.002177405 * max(0.0, 0.9909875 - Q.psi_0p1) * max(0.0, Q.pt_1 - 99.0) / 4.936238   # -0.2%  psi_0p1 < 0.991 and pt_1 > 99
        - 0.001953232 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 56.19962   # -0.2%  n_dr_0p2_0p4 < 26 and n_dr_0p1_0p2 > 11
        - 0.001944511 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # -0.2%  e2 > 0.04755
        + 0.001853546 * max(0.0, 91.19 - Q.mass_top15) * max(0.0, Q.mass_top2 - 3.19816) / 74.73878   # +0.2%  mass_top15 < 91.19 and mass_top2 > 3.198
        + 0.001849128 * max(0.0, 0.3036026 - Q.planar_flow) / 0.04665298   # +0.2%  planar_flow < 0.3036
        - 0.001822487 * max(0.0, 143.7876 - Q.mass) * max(0.0, Q.C2_b2 - 0.002289486) / 0.6546793   # -0.2%  mass < 143.8 and C2_b2 > 0.002289
        - 0.001775385 * max(0.0, Q.sum_z_dr2_top15 - 0.00727763) * max(0.0, 33.78125 - Q.pt_11) / 0.03009459   # -0.2%  sum_z_dr2_top15 > 0.007278 and pt_11 < 33.78
        - 0.001757403 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # -0.2%  C2 > 0.06656
        + 0.001569599 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.6616141 - Q.tau32) / 0.002828546   # +0.2%  sj2_dr > 0.2232 and tau32 < 0.6616
        + 0.001480592 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.605781   # +0.1%  n_particles > 22 and psi_0p3 > 0.9638
        - 0.001417209 * max(0.0, 62.55 - Q.mass_top40) / 6.231505   # -0.1%  mass_top40 < 62.55
        - 0.001328679 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, Q.zdr_0 - 0.001901263) / 0.02085507   # -0.1%  mass_top40 < 83.33 and zdr_0 > 0.001901
        - 0.001295252 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # -0.1%  e2 > 0.06524
        - 0.00123072 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # -0.1%  M3 < 0.03188
        + 0.001178364 * max(0.0, Q.sum_z_dr2_top5 - 0.0168592) / 0.0008909511   # +0.1%  sum_z_dr2_top5 > 0.01686
        - 0.00102165 * max(0.0, 21.52288 - Q.sj2_mass1) / 3.113774   # -0.1%  sj2_mass1 < 21.52
        - 0.0009562881 * max(0.0, 21.52288 - Q.sj2_mass1) * max(0.0, Q.pair_mass_0_13 - 8.669189) / 3.234208   # -0.1%  sj2_mass1 < 21.52 and pair_mass_0_13 > 8.669
        - 0.0009334823 * max(0.0, Q.log_sum_pt - 6.97212) * max(0.0, Q.abseta_11 - 0.02558899) / 0.0006298046   # -0.1%  log_sum_pt > 6.972 and abseta_11 > 0.02559
        - 0.0007743887 * max(0.0, 0.03187688 - Q.M3) * max(0.0, Q.soft6_dr0 - 0.06490217) / 0.0012584   # -0.1%  M3 < 0.03188 and soft6_dr0 > 0.0649
        - 0.0007399964 * max(0.0, Q.lam2 - 0.001776308) / 0.0005876074   # -0.1%  lam2 > 0.001776
        + 0.0005267749 * max(0.0, Q.tau2 - 0.05936026) / 0.004947723   # +0.1%  tau2 > 0.05936
        - 0.0005029487 * max(0.0, Q.pair_mass_0_12 - 15.47191) / 0.4335287   # -0.1%  pair_mass_0_12 > 15.47
        - 0.0004462401 * max(0.0, 0.3036026 - Q.planar_flow) * max(0.0, Q.n_dr_0p1_0p2 - 13.0) / 0.1025731   # -0.0%  planar_flow < 0.3036 and n_dr_0p1_0p2 > 13
        - 0.0004188411 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.zdr_0 - 0.0008718296) / 0.0994482   # -0.0%  mass < 101 and zdr_0 > 0.0008718
        - 0.0003663432 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.005911134 - Q.zdr_0) / 9.63153e-07   # -0.0%  psi_0p3 > 0.9974 and zdr_0 < 0.005911
        - 0.0002606317 * max(0.0, Q.sum_pt - 995.6769) / 62.24161   # -0.0%  sum_pt > 995.7
        + 0.0002319009 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.sj3_mass1 - 32.50209) / 17.4117   # +0.0%  n_particles > 22 and sj3_mass1 > 32.5
        - 0.0002267908 * max(0.0, 0.3036026 - Q.planar_flow) * max(0.0, Q.z_dr_0p4_up - 0.0) / 5.226105e-06   # -0.0%  planar_flow < 0.3036 and z_dr_0p4_up > 0
        + 0.0001811376 * max(0.0, 62.55 - Q.mass_top40) * max(0.0, 3.014827 - Q.D2_b2) / 0.824937   # +0.0%  mass_top40 < 62.55 and D2_b2 < 3.015
        - 0.0001809909 * max(0.0, 80.78464 - Q.mass) * max(0.0, 0.4953696 - Q.D2_b2) / 0.03149441   # -0.0%  mass < 80.78 and D2_b2 < 0.4954
        + 0.0001291953 * max(0.0, 75.26407 - Q.mass_top15) / 22.29039   # +0.0%  mass_top15 < 75.26
        - 0.000127905 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 0.4953696 - Q.D2_b2) / 0.08717725   # -0.0%  mass_top40 < 83.33 and D2_b2 < 0.4954
        + 0.0001204289 * max(0.0, Q.sum_z_dr - 0.1207452) * max(0.0, Q.abseta_3 - 0.005970001) / 0.0004035593   # +0.0%  sum_z_dr > 0.1207 and abseta_3 > 0.00597
        - 3.663693e-05 * max(0.0, 80.78464 - Q.mass) * max(0.0, Q.mean_eta - 0.00249209) / 2.872187e-05   # -0.0%  mass < 80.78 and mean_eta > 0.002492
        - 3.565558e-05 * max(0.0, 120.6 - Q.mass) * max(0.0, 0.9943058 - Q.psi_0p3) / 0.08236936   # -0.0%  mass < 120.6 and psi_0p3 < 0.9943
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 12.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.77557 * (0.05800398
        + 0.09609651 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +9.6%  sum_pt_top50 > 934.2
        + 0.06323103 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +6.3%  sum_pt > 907.9
        - 0.05827811 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -5.8%  log_sum_pt > 6.91
        - 0.05750626 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # -5.8%  sum_pt > 986.1
        + 0.04912474 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # +4.9%  sum_pt_top50 > 959.1
        - 0.03371587 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # -3.4%  mass_over_sum_pt > 0.07697
        - 0.03256702 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -3.3%  log_sum_pt > 6.92
        + 0.0309356 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +3.1%  n_particles < 64
        - 0.03055536 * max(0.0, Q.sum_z_dr2_top30 - 0.007463985) / 0.00336109   # -3.1%  sum_z_dr2_top30 > 0.007464
        + 0.03001129 * max(0.0, Q.sum_z_dr2_top30 - 0.006363916) / 0.003802703   # +3.0%  sum_z_dr2_top30 > 0.006364
        + 0.02765889 * max(0.0, Q.sum_z_dr2 - 0.004756928) / 0.005348122   # +2.8%  sum_z_dr2 > 0.004757
        + 0.02665794 * max(0.0, Q.sum_pt_top30 - 933.1875) / 79.88556   # +2.7%  sum_pt_top30 > 933.2
        - 0.02625912 * max(0.0, 150.0144 - Q.mass_top40) / 66.86198   # -2.6%  mass_top40 < 150
        + 0.02382682 * max(0.0, Q.mass - 172.4888) / 0.7446233   # +2.4%  mass > 172.5
        - 0.02143744 * max(0.0, Q.mass - 172.8) / 0.7291439   # -2.1%  mass > 172.8
        - 0.01988928 * max(0.0, Q.mass - 91.19) / 15.01545   # -2.0%  mass > 91.19
        - 0.01787821 * max(0.0, Q.z_top30_slots - 0.9048492) / 0.05621641   # -1.8%  z_top30_slots > 0.9048
        - 0.01673342 * max(0.0, 0.01807679 - Q.sum_z_dr2_top30) / 0.01083719   # -1.7%  sum_z_dr2_top30 < 0.01808
        + 0.01614296 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +1.6%  sum_pt > 907.9 and e4 < 5.851e-08
        - 0.01575895 * max(0.0, Q.sum_pt_top40 - 994.2695) / 51.72055   # -1.6%  sum_pt_top40 > 994.3
        + 0.01411236 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # +1.4%  e2 < 0.03876
        + 0.01329124 * max(0.0, Q.psi_0p3 - 0.9638082) / 0.02792824   # +1.3%  psi_0p3 > 0.9638
        - 0.01321106 * max(0.0, 787.6281 - Q.sum_pt_top3) / 328.0371   # -1.3%  sum_pt_top3 < 787.6
        + 0.01184616 * max(0.0, Q.mass - 74.25181) / 24.07648   # +1.2%  mass > 74.25
        + 0.01134626 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # +1.1%  log_sum_pt > 6.989
        + 0.01103664 * max(0.0, 0.03787151 - Q.M3) / 0.01370579   # +1.1%  M3 < 0.03787
        - 0.01001709 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -1.0%  max_dr > 0.2405
        + 0.009610099 * max(0.0, Q.sd_mass - 69.65633) / 11.73272   # +1.0%  sd_mass > 69.66
        - 0.00907373 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2) / 10.81389   # -0.9%  n_particles < 64 and D2 < 2.179
        - 0.008647608 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -0.9%  sum_pt_top40 > 1025
        + 0.008427973 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +0.8%  mass_over_sum_pt > 0.1709
        - 0.007731093 * max(0.0, Q.mass_over_sum_pt - 0.09046749) / 0.01421039   # -0.8%  mass_over_sum_pt > 0.09047
        - 0.007687266 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -0.8%  mass_over_sum_pt_sq > 0.0292
        + 0.007509137 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +0.8%  n_dr_0p2_0p4 < 11
        - 0.006717283 * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 9.82015   # -0.7%  n_dr_0p1_0p2 < 21
        - 0.006370107 * max(0.0, Q.sd_mass - 86.4) / 6.275027   # -0.6%  sd_mass > 86.4
        + 0.006030784 * max(0.0, Q.sj2_dr - 0.1411617) / 0.06721719   # +0.6%  sj2_dr > 0.1412
        + 0.006003392 * max(0.0, Q.n_pt_above_1 - 28.0) / 14.80415   # +0.6%  n_pt_above_1 > 28
        + 0.005818461 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2) / 0.001543279   # +0.6%  psi_0p3 > 0.9974 and D2 < 3.345
        + 0.005778597 * max(0.0, Q.log_sum_pt - 6.949481) / 0.03162473   # +0.6%  log_sum_pt > 6.949
        - 0.005363122 * max(0.0, 0.01083435 - Q.sum_z_dr2_top20) / 0.00529825   # -0.5%  sum_z_dr2_top20 < 0.01083
        - 0.004963784 * max(0.0, Q.mass_top40 - 120.6) / 5.823744   # -0.5%  mass_top40 > 120.6
        - 0.004961761 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.2535773 - Q.dr_11) / 0.00936088   # -0.5%  log_sum_pt > 6.91 and dr_11 < 0.2536
        - 0.004869531 * max(0.0, Q.z_top20_slots - 0.9102775) / 0.02689878   # -0.5%  z_top20_slots > 0.9103
        - 0.004587251 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -0.5%  mass_top50 > 157.5
        + 0.004523192 * Q.soft4_dr0 / 0.1956066   # +0.5%  soft4_dr0
        + 0.004482981 * max(0.0, 0.01375115 - Q.z_11) / 0.0009450582   # +0.4%  z_11 < 0.01375
        + 0.00415042 * max(0.0, Q.n_pt_above_1 - 28.0) * max(0.0, 0.942303 - Q.tau43) / 1.657469   # +0.4%  n_pt_above_1 > 28 and tau43 < 0.9423
        - 0.004130916 * max(0.0, 0.04828819 - Q.tau2) / 0.02103241   # -0.4%  tau2 < 0.04829
        + 0.00396257 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # +0.4%  n_dr_0p1_0p2 < 8
        - 0.003861945 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, Q.soft4_dr - 0.06076665) / 16.9581   # -0.4%  sum_pt > 907.9 and soft4_dr > 0.06077
        - 0.003755436 * max(0.0, 0.01541561 - Q.e2) / 0.001436351   # -0.4%  e2 < 0.01542
        - 0.003619869 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.1269782 - Q.dr_12) / 0.003311185   # -0.4%  log_sum_pt > 6.91 and dr_12 < 0.127
        - 0.003570098 * max(0.0, 14.14062 - Q.pt_11) / 0.9740783   # -0.4%  pt_11 < 14.14
        + 0.003367003 * max(0.0, 1.976207 - Q.D2) / 0.367701   # +0.3%  D2 < 1.976
        + 0.003359318 * max(0.0, 23.31647 - Q.sj2_mass1) / 3.96589   # +0.3%  sj2_mass1 < 23.32
        - 0.003350283 * max(0.0, Q.z_top30_slots - 0.9048492) * max(0.0, Q.sj3_mass1 - 10.87918) / 0.1762583   # -0.3%  z_top30_slots > 0.9048 and sj3_mass1 > 10.88
        - 0.003347791 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # -0.3%  sum_pt_top40 > 1070
        - 0.003268146 * max(0.0, 0.1937447 - Q.z_dr_0p2_0p4) / 0.146372   # -0.3%  z_dr_0p2_0p4 < 0.1937
        - 0.003217624 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.1958008 - Q.absphi_13) / 0.007588689   # -0.3%  log_sum_pt > 6.91 and absphi_13 < 0.1958
        - 0.003171437 * max(0.0, Q.sum_pt_top40 - 994.2695) * max(0.0, 0.3241858 - Q.dr1_12) / 12.50457   # -0.3%  sum_pt_top40 > 994.3 and dr1_12 < 0.3242
        + 0.002931429 * max(0.0, 19.34375 - Q.pt_14) / 5.007666   # +0.3%  pt_14 < 19.34
        + 0.002694362 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.3%  psi_0p3 > 0.9974
        - 0.002569563 * max(0.0, 0.6635952 - Q.psi_0p1) / 0.06403963   # -0.3%  psi_0p1 < 0.6636
        - 0.002535312 * max(0.0, 0.5494307 - Q.tau21) / 0.1591168   # -0.3%  tau21 < 0.5494
        + 0.002183687 * max(0.0, 0.3437357 - Q.max_dr) / 0.02558865   # +0.2%  max_dr < 0.3437
        + 0.002161231 * max(0.0, 0.005788041 - Q.sum_z_dr2_top15) / 0.001877672   # +0.2%  sum_z_dr2_top15 < 0.005788
        + 0.002057099 * max(0.0, 64.48544 - Q.mass) / 5.935866   # +0.2%  mass < 64.49
        + 0.001995666 * max(0.0, Q.z_dr_0_0p05 - 0.9084912) / 0.009227347   # +0.2%  z_dr_0_0p05 > 0.9085
        - 0.00197123 * max(0.0, 0.0004816552 - Q.sum_z_dr2_top15) / 3.66329e-05   # -0.2%  sum_z_dr2_top15 < 0.0004817
        - 0.001962117 * max(0.0, Q.sum_z_dr2_top50 - 0.01951641) / 0.001208668   # -0.2%  sum_z_dr2_top50 > 0.01952
        + 0.00194326 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 0.03260972 - Q.z_8) / 0.1432208   # +0.2%  n_particles < 64 and z_8 < 0.03261
        + 0.001903989 * max(0.0, Q.sum_pt_top20 - 1129.275) / 6.499168   # +0.2%  sum_pt_top20 > 1129
        - 0.001629962 * max(0.0, 1.692473 - Q.ptdr0_8) / 0.6936217   # -0.2%  ptdr0_8 < 1.692
        + 0.001225404 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, Q.dr0_10 - 0.06373449) / 0.002273253   # +0.1%  log_sum_pt > 6.91 and dr0_10 > 0.06373
        - 0.00103528 * max(0.0, 1.20817 - Q.ptdr0_10) / 0.4204752   # -0.1%  ptdr0_10 < 1.208
        - 0.0009107484 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.1%  sum_pt_top10 > 943.7
        + 0.0008655493 * max(0.0, Q.sum_pt_top40 - 1069.671) * max(0.0, 0.07895359 - Q.dr1_12) / 0.8620997   # +0.1%  sum_pt_top40 > 1070 and dr1_12 < 0.07895
        + 0.0008396347 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # +0.1%  mass_top10 > 71.78
        + 0.000820924 * max(0.0, Q.log_sum_pt - 6.935549) / 0.0371538   # +0.1%  log_sum_pt > 6.936
        + 0.0008052384 * max(0.0, 0.02946315 - Q.z_5) / 0.0006616775   # +0.1%  z_5 < 0.02946
        + 0.0007808877 * max(0.0, Q.log_sum_pt - 6.98945) * max(0.0, Q.soft4_dr - 0.02734277) / 0.003230455   # +0.1%  log_sum_pt > 6.989 and soft4_dr > 0.02734
        - 0.0007702614 * max(0.0, Q.psi_0p3 - 0.9638082) * max(0.0, Q.ptdr0_8 - 4.453578) / 0.01742855   # -0.1%  psi_0p3 > 0.9638 and ptdr0_8 > 4.454
        - 0.000755549 * max(0.0, Q.n_dr_0p2_0p4 - 21.0) / 0.6469597   # -0.1%  n_dr_0p2_0p4 > 21
        + 0.0005650105 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.orientation_deg - -0.8247956) / 102.6756   # +0.1%  n_dr_0p2_0p4 < 11 and orientation_deg > -0.8248
        - 0.0005553245 * max(0.0, 0.1937447 - Q.z_dr_0p2_0p4) * max(0.0, Q.absphi_9 - 0.03289795) / 0.003158116   # -0.1%  z_dr_0p2_0p4 < 0.1937 and absphi_9 > 0.0329
        - 0.0005315765 * max(0.0, Q.mass_top10 - 71.781) * max(0.0, 114.75 - Q.pt_3) / 166.4082   # -0.1%  mass_top10 > 71.78 and pt_3 < 114.8
        + 0.0003982981 * max(0.0, Q.max_dr - 0.4357228) / 0.007711928   # +0.0%  max_dr > 0.4357
        - 0.0003628739 * max(0.0, 0.03787151 - Q.M3) * max(0.0, Q.soft3_dr - 0.1968156) / 0.0005985639   # -0.0%  M3 < 0.03787 and soft3_dr > 0.1968
        - 0.0003479581 * max(0.0, Q.mass_top10 - 71.781) * max(0.0, Q.tau43 - 0.7207148) / 0.4279737   # -0.0%  mass_top10 > 71.78 and tau43 > 0.7207
        - 0.0003280165 * max(0.0, Q.n_pt_above_10 - 31.0) / 0.1555479   # -0.0%  n_pt_above_10 > 31
        - 0.0002909031 * max(0.0, 14.14062 - Q.pt_11) * max(0.0, 15.0 - Q.n_real_top15) / 0.1378737   # -0.0%  pt_11 < 14.14 and n_real_top15 < 15
        + 0.000199242 * max(0.0, Q.z_top50_slots - 0.9906378) / 0.006446962   # +0.0%  z_top50_slots > 0.9906
        - 0.0001837517 * max(0.0, Q.z_top50_slots - 0.9906378) * max(0.0, 0.2696957 - Q.dr1_13) / 0.001113422   # -0.0%  z_top50_slots > 0.9906 and dr1_13 < 0.2697
        + 0.0001532184 * max(0.0, Q.mass_over_sum_pt - 0.09046749) * max(0.0, 0.008031476 - Q.soft7_dr0) / 1.24889e-07   # +0.0%  mass_over_sum_pt > 0.09047 and soft7_dr0 < 0.008031
        + 0.0001474827 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.pair_mass_0_10 - 1.569656) / 31.11071   # +0.0%  sum_pt_top20 > 1129 and pair_mass_0_10 > 1.57
        + 0.0001124388 * max(0.0, 0.5494307 - Q.tau21) * max(0.0, Q.soft1_dr - 0.2690904) / 0.004430241   # +0.0%  tau21 < 0.5494 and soft1_dr > 0.2691
        - 5.988128e-05 * max(0.0, 0.03787151 - Q.M3) * max(0.0, Q.soft2_dr - 0.1573586) / 0.0009567484   # -0.0%  M3 < 0.03787 and soft2_dr > 0.1574
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 16.12;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.11914 * (-0.01076667
        + 0.1482425 * max(0.0, 172.8 - Q.mass_top50) / 85.49388   # +14.8%  mass_top50 < 172.8
        - 0.09953561 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -10.0%  mass < 101
        - 0.0871642 * max(0.0, 168.9698 - Q.mass_top50) / 81.82258   # -8.7%  mass_top50 < 169
        + 0.0760305 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +7.6%  mass < 92.86
        - 0.07180827 * max(0.0, 120.6 - Q.mass) / 38.8279   # -7.2%  mass < 120.6
        + 0.06189297 * max(0.0, Q.mass_over_sum_pt - 0.02682209) / 0.06022589   # +6.2%  mass_over_sum_pt > 0.02682
        + 0.02946062 * max(0.0, 0.06524004 - Q.e2) / 0.03476195   # +2.9%  e2 < 0.06524
        - 0.02884902 * max(0.0, Q.mass_over_sum_pt - 0.09795415) / 0.01222897   # -2.9%  mass_over_sum_pt > 0.09795
        - 0.0269552 * max(0.0, Q.sum_z_dr - 0.02085222) / 0.04959222   # -2.7%  sum_z_dr > 0.02085
        - 0.02563249 * max(0.0, 0.01807679 - Q.sum_z_dr2_top30) / 0.01083719   # -2.6%  sum_z_dr2_top30 < 0.01808
        + 0.02381292 * max(0.0, 86.4 - Q.mass) / 13.67632   # +2.4%  mass < 86.4
        + 0.02369703 * max(0.0, Q.tau1 - 0.05444509) / 0.03990678   # +2.4%  tau1 > 0.05445
        + 0.02104589 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +2.1%  lam1 < 0.007259
        + 0.02012436 * max(0.0, 0.003687605 - Q.lam2) / 0.002601874   # +2.0%  lam2 < 0.003688
        + 0.01641056 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # +1.6%  mass_top50 < 71.8
        - 0.01630678 * max(0.0, 1129.275 - Q.sum_pt_top20) / 207.3581   # -1.6%  sum_pt_top20 < 1129
        - 0.01610917 * max(0.0, 0.02048524 - Q.lam1) / 0.01309746   # -1.6%  lam1 < 0.02049
        + 0.01366495 * max(0.0, Q.sum_z_dr2_top40 - 0.008031986) / 0.003375597   # +1.4%  sum_z_dr2_top40 > 0.008032
        + 0.0128059 * max(0.0, 91.19 - Q.mass_top15) / 34.88803   # +1.3%  mass_top15 < 91.19
        - 0.01007516 * max(0.0, Q.e2 - 0.02793599) / 0.008664879   # -1.0%  e2 > 0.02794
        + 0.009727432 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # +1.0%  sum_pt < 1261
        + 0.009128783 * max(0.0, Q.sum_z_dr2_top20 - 0.008031209) / 0.002906852   # +0.9%  sum_z_dr2_top20 > 0.008031
        - 0.008939915 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2) / 0.04240098   # -0.9%  mass_over_sum_pt > 0.1182 and D2_b2 < 7.366
        + 0.008781515 * max(0.0, Q.sum_z_dr2_top30 - 0.008376291) / 0.003092781   # +0.9%  sum_z_dr2_top30 > 0.008376
        + 0.008281596 * max(0.0, 0.003811746 - Q.lam1) / 0.0006796118   # +0.8%  lam1 < 0.003812
        - 0.00738276 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -0.7%  mass_over_sum_pt > 0.1182
        - 0.006726347 * max(0.0, 0.004573744 - Q.sum_z_dr2_top50) / 0.0007739713   # -0.7%  sum_z_dr2_top50 < 0.004574
        + 0.006311171 * max(0.0, 0.00375223 - Q.sum_z_dr2_top30) / 0.0006714094   # +0.6%  sum_z_dr2_top30 < 0.003752
        - 0.00626646 * max(0.0, 0.1011005 - Q.M2) / 0.03683604   # -0.6%  M2 < 0.1011
        - 0.005271518 * max(0.0, Q.sum_z_dr2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529) / 8.566262e-05   # -0.5%  sum_z_dr2_top20 > 0.008031 and C2_b2 > 0.0008188
        - 0.004985307 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # -0.5%  log_sum_pt < 7.017
        - 0.004976592 * max(0.0, 0.08329945 - Q.mass_over_sum_pt) / 0.0135093   # -0.5%  mass_over_sum_pt < 0.0833
        - 0.004835484 * max(0.0, Q.mass_over_sum_pt - 0.02682209) * max(0.0, 0.7108211 - Q.z_dr_0p05_0p1) / 0.02513932   # -0.5%  mass_over_sum_pt > 0.02682 and z_dr_0p05_0p1 < 0.7108
        + 0.004254015 * max(0.0, 0.03700182 - Q.z_dr_0p2_0p4) / 0.0183096   # +0.4%  z_dr_0p2_0p4 < 0.037
        - 0.004118956 * max(0.0, 1.409617 - Q.D2) / 0.1519846   # -0.4%  D2 < 1.41
        + 0.004076223 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +0.4%  tau1 < 0.06311
        - 0.003744131 * max(0.0, Q.sj3_pair_mass_min - 29.00832) / 6.839984   # -0.4%  sj3_pair_mass_min > 29.01
        + 0.00351533 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt) / 683.4113   # +0.4%  mass < 120.6 and sum_pt < 1008
        + 0.003352158 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +0.3%  z_dr_0p1_0p2 < 0.1203
        - 0.003221239 * max(0.0, 0.003418057 - Q.sum_z_dr2_top50) / 0.0004594629   # -0.3%  sum_z_dr2_top50 < 0.003418
        - 0.003211509 * max(0.0, 91.19 - Q.mass_top15) * max(0.0, 0.04008677 - Q.C2_b2) / 0.9324124   # -0.3%  mass_top15 < 91.19 and C2_b2 < 0.04009
        + 0.003102066 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) / 1.539217   # +0.3%  n_dr_0p2_0p4 > 15
        + 0.003056558 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.3%  e2 > 0.04755
        - 0.00293857 * max(0.0, Q.sum_z_dr2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722) / 0.001787472   # -0.3%  sum_z_dr2_top30 > 0.008376 and D2_b2 > 1.677
        + 0.002698777 * max(0.0, 0.002197765 - Q.sum_z_dr2_top15) / 0.0004509993   # +0.3%  sum_z_dr2_top15 < 0.002198
        - 0.002403531 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, Q.dr_max_012 - 0.004415714) / 1.536542e-05   # -0.2%  e3 < 0.0003372 and dr_max_012 > 0.004416
        - 0.002217315 * max(0.0, Q.sj3_dr_min - 0.1204829) / 0.02211875   # -0.2%  sj3_dr_min > 0.1205
        + 0.001746487 * max(0.0, 0.9300465 - Q.z_top40_slots) / 0.002595351   # +0.2%  z_top40_slots < 0.93
        + 0.001583766 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +0.2%  e3 < 0.0003372
        + 0.001373162 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +0.1%  LHA > 0.372
        - 0.001368614 * max(0.0, 0.001592178 - Q.sum_z_dr2_top3) / 0.000513961   # -0.1%  sum_z_dr2_top3 < 0.001592
        - 0.001355791 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.1%  log_sum_pt < 6.811
        + 0.001314959 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832) / 8.191828e-05   # +0.1%  mass_over_sum_pt > 0.1182 and C2_b2 > 0.02705
        - 0.001281693 * max(0.0, Q.e2 - 0.02793599) * max(0.0, 0.01718455 - Q.zdr_1) / 5.467259e-05   # -0.1%  e2 > 0.02794 and zdr_1 < 0.01718
        - 0.001274338 * max(0.0, 30.35994 - Q.mass_top15) / 3.169072   # -0.1%  mass_top15 < 30.36
        + 0.001240328 * max(0.0, Q.e2 - 0.05557149) * max(0.0, 1.67722 - Q.D2_b2) / 0.000547635   # +0.1%  e2 > 0.05557 and D2_b2 < 1.677
        - 0.001183572 * max(0.0, Q.n_dr_0_0p05 - 9.0) / 5.593776   # -0.1%  n_dr_0_0p05 > 9
        + 0.001122397 * max(0.0, Q.sj3_dr_min - 0.1204829) * max(0.0, 33.78125 - Q.pt_11) / 0.2695668   # +0.1%  sj3_dr_min > 0.1205 and pt_11 < 33.78
        - 0.0009826021 * max(0.0, Q.pt_14 - 10.78125) / 5.796532   # -0.1%  pt_14 > 10.78
        + 0.0009577199 * max(0.0, Q.sum_z_dr2_top30 - 0.008376291) * max(0.0, Q.z_dr_0p05_0p1 - 0.6469679) / 1.118691e-05   # +0.1%  sum_z_dr2_top30 > 0.008376 and z_dr_0p05_0p1 > 0.647
        - 0.0009576476 * max(0.0, 7.017258 - Q.log_sum_pt) * max(0.0, Q.soft10_dr0 - 0.1732831) / 0.004149624   # -0.1%  log_sum_pt < 7.017 and soft10_dr0 > 0.1733
        + 0.0009432149 * max(0.0, Q.sj2_dr - 0.2595052) / 0.01337254   # +0.1%  sj2_dr > 0.2595
        + 0.0009224448 * max(0.0, Q.e2 - 0.02793599) * max(0.0, 0.4357228 - Q.max_dr) / 0.0007248268   # +0.1%  e2 > 0.02794 and max_dr < 0.4357
        + 0.0009019072 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, Q.soft5_dr0 - 0.3255096) / 0.03360383   # +0.1%  n_dr_0p2_0p4 > 15 and soft5_dr0 > 0.3255
        + 0.0009012829 * max(0.0, 0.007259287 - Q.lam1) * max(0.0, Q.n_dr_0p4_up - 0.0) / 0.0002748582   # +0.1%  lam1 < 0.007259 and n_dr_0p4_up > 0
        - 0.0008392955 * max(0.0, Q.sum_z_dr2_top30 - 0.02809026) / 0.0001993235   # -0.1%  sum_z_dr2_top30 > 0.02809
        + 0.0007137704 * max(0.0, Q.mass_over_sum_pt - 0.02682209) * max(0.0, Q.eta_12 - 0.09442749) / 0.0005875947   # +0.1%  mass_over_sum_pt > 0.02682 and eta_12 > 0.09443
        + 0.0007058788 * max(0.0, 0.04704395 - Q.C2) / 0.005827731   # +0.1%  C2 < 0.04704
        - 0.0006881446 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # -0.1%  e2 > 0.05557
        + 0.0006575024 * max(0.0, Q.sj3_pair_mass_min - 29.00832) * max(0.0, Q.psi_0p3 - 0.9896594) / 0.01664863   # +0.1%  sj3_pair_mass_min > 29.01 and psi_0p3 > 0.9897
        + 0.0006443235 * max(0.0, Q.pt_14 - 10.78125) * max(0.0, Q.sj3_mass3 - 0.4118081) / 29.43323   # +0.1%  pt_14 > 10.78 and sj3_mass3 > 0.4118
        + 0.0005984546 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707) / 2.369749e-05   # +0.1%  e2 > 0.05557 and zdr_0 > 0.00137
        + 0.0005729056 * max(0.0, Q.sum_z_dr2_top30 - 0.008376291) * max(0.0, 43.0 - Q.n_real_top50) / 0.0007262366   # +0.1%  sum_z_dr2_top30 > 0.008376 and n_real_top50 < 43
        + 0.0005716805 * max(0.0, Q.sj3_mass2 - 11.91094) / 0.99081   # +0.1%  sj3_mass2 > 11.91
        - 0.0005663902 * max(0.0, 1.409617 - Q.D2) * max(0.0, Q.soft9_dr - 0.2400617) / 0.001081664   # -0.1%  D2 < 1.41 and soft9_dr > 0.2401
        + 0.0005159184 * max(0.0, 120.6 - Q.mass) * max(0.0, -0.003213499 - Q.mean_eta) / 0.0005246177   # +0.1%  mass < 120.6 and mean_eta < -0.003213
        + 0.0004183884 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) / 0.293679   # +0.0%  n_dr_0p1_0p2 > 33
        - 0.0004174887 * max(0.0, Q.sj3_mass1 - 21.11128) / 1.638992   # -0.0%  sj3_mass1 > 21.11
        - 0.0003978018 * max(0.0, Q.sj3_pair_mass_min - 76.60223) / 0.3297656   # -0.0%  sj3_pair_mass_min > 76.6
        + 0.000386039 * max(0.0, Q.sj3_dr_min - 0.1204829) * max(0.0, 7.863702 - Q.sj3_mass2) / 0.02256917   # +0.0%  sj3_dr_min > 0.1205 and sj3_mass2 < 7.864
        + 0.0003416719 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.z_2 - 0.120204) / 8.801305e-07   # +0.0%  e2 > 0.05557 and z_2 > 0.1202
        - 0.0002874724 * max(0.0, Q.sum_z_dr2_top30 - 0.008376291) * max(0.0, Q.orientation_deg - 26.6454) / 0.03454067   # -0.0%  sum_z_dr2_top30 > 0.008376 and orientation_deg > 26.65
        - 0.0002862618 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, Q.max_pair_mass - 33.3761) / 3.25809   # -0.0%  n_dr_0p2_0p4 > 15 and max_pair_mass > 33.38
        - 0.0002475765 * max(0.0, -0.1135254 - Q.eta_5) / 0.002643976   # -0.0%  eta_5 < -0.1135
        + 0.0002080285 * max(0.0, Q.e2 - 0.04755309) * max(0.0, 0.4118081 - Q.sj3_mass3) / 3.78812e-06   # +0.0%  e2 > 0.04755 and sj3_mass3 < 0.4118
        - 0.0001789487 * max(0.0, Q.sum_pt_top10 - 975.0641) / 8.216129   # -0.0%  sum_pt_top10 > 975.1
        - 0.0001731883 * max(0.0, 0.00242543 - Q.sum_z_dr2_top50) / 0.0002374304   # -0.0%  sum_z_dr2_top50 < 0.002425
        + 0.0001633778 * max(0.0, 19.46875 - Q.pt_6) / 0.2515892   # +0.0%  pt_6 < 19.47
        - 0.0001337087 * max(0.0, Q.LHA - 0.3719813) * max(0.0, Q.z_dr_0p05_0p1 - 0.6469679) / 2.665919e-06   # -0.0%  LHA > 0.372 and z_dr_0p05_0p1 > 0.647
        - 0.0001195527 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) * max(0.0, Q.ptdr0_10 - 6.130737) / 0.3325374   # -0.0%  n_dr_0p1_0p2 > 33 and ptdr0_10 > 6.131
        - 0.0001191913 * max(0.0, Q.n_dr_0p05_0p1 - 21.0) / 0.7932437   # -0.0%  n_dr_0p05_0p1 > 21
        + 0.0001017989 * max(0.0, 0.04704395 - Q.C2) * max(0.0, Q.dr1_8 - 0.2050905) / 9.23615e-06   # +0.0%  C2 < 0.04704 and dr1_8 > 0.2051
        - 9.428983e-05 * max(0.0, Q.sj3_pair_mass_min - 29.00832) * max(0.0, Q.sj3_z1 - 0.7389287) / 0.01191426   # -0.0%  sj3_pair_mass_min > 29.01 and sj3_z1 > 0.7389
        - 8.308965e-05 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) * max(0.0, 0.03967285 - Q.soft9_absphi) / 0.001136098   # -0.0%  n_dr_0p1_0p2 > 33 and soft9_absphi < 0.03967
        - 7.844585e-05 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.sj3_z1 - 0.7707617) / 2.638064e-07   # -0.0%  e2 > 0.04755 and sj3_z1 > 0.7708
        - 4.770395e-05 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) * max(0.0, 0.12854 - Q.absphi_2) / 0.01260196   # -0.0%  n_dr_0p1_0p2 > 33 and absphi_2 < 0.1285
        + 4.003191e-05 * max(0.0, 0.1011005 - Q.M2) * max(0.0, -0.07794189 - Q.eta_0) / 6.491626e-05   # +0.0%  M2 < 0.1011 and eta_0 < -0.07794
        - 3.332586e-05 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.soft3_dr0 - 0.1966723) / 0.0006797808   # -0.0%  mass_over_sum_pt > 0.1182 and soft3_dr0 > 0.1967
        - 7.05019e-06 * max(0.0, Q.mass_over_sum_pt - 0.09795415) * max(0.0, 0.001160626 - Q.soft8_z) / 8.555108e-07   # -0.0%  mass_over_sum_pt > 0.09795 and soft8_z < 0.001161
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 39.74;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 39.73608 * (0.002604149
        + 0.128959 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5337743   # +12.9%  mass < 91.19 and psi_0p3 > 0.9638
        - 0.1277606 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.3075308   # -12.8%  mass < 91.19 and psi_0p3 > 0.9777
        - 0.1206614 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5700006   # -12.1%  mass < 92.86 and psi_0p3 > 0.9638
        + 0.08408095 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.4385899   # +8.4%  mass < 101 and psi_0p3 > 0.9777
        + 0.04829102 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.2231625   # +4.8%  mass < 82.85 and psi_0p3 > 0.9777
        - 0.03579404 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -3.6%  mass < 101
        - 0.03565645 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.01143143   # -3.6%  mass < 91.19 and psi_0p3 > 0.998
        + 0.03367123 * max(0.0, 0.009614971 - Q.sum_z_dr2) / 0.003502545   # +3.4%  sum_z_dr2 < 0.009615
        - 0.03154263 * max(0.0, 0.008190222 - Q.sum_z_dr2) / 0.002454223   # -3.2%  sum_z_dr2 < 0.00819
        - 0.02894344 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -2.9%  tau1 < 0.1073
        + 0.02693314 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.01745475   # +2.7%  mass < 101 and psi_0p3 > 0.998
        + 0.02654424 * max(0.0, 120.6 - Q.mass) / 38.8279   # +2.7%  mass < 120.6
        - 0.02551062 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -2.6%  mass < 82.85
        - 0.02128761 * max(0.0, 86.4 - Q.sd_mass) / 35.84633   # -2.1%  sd_mass < 86.4
        + 0.01928719 * max(0.0, 0.09591084 - Q.tau1) / 0.02629125   # +1.9%  tau1 < 0.09591
        - 0.01882343 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -1.9%  lam1 < 0.008242
        - 0.01639526 * max(0.0, 0.1881908 - Q.sd_rg) / 0.07278694   # -1.6%  sd_rg < 0.1882
        + 0.01532421 * max(0.0, 69.65633 - Q.sd_mass) / 24.56034   # +1.5%  sd_mass < 69.66
        + 0.01402409 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +1.4%  mass < 78.26
        + 0.01204644 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +1.2%  mass_over_sum_pt < 0.1182
        + 0.01195808 * max(0.0, 0.2639816 - Q.sd_rg) / 0.1327654   # +1.2%  sd_rg < 0.264
        + 0.01137282 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.007828415   # +1.1%  mass < 82.85 and psi_0p3 > 0.998
        + 0.009221604 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +0.9%  lam1 < 0.01174
        + 0.008867839 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # +0.9%  lam1 < 0.00619
        + 0.008675731 * max(0.0, 91.19 - Q.mass) / 16.46423   # +0.9%  mass < 91.19
        + 0.00845521 * max(0.0, 0.1596365 - Q.sd_rg) / 0.05492863   # +0.8%  sd_rg < 0.1596
        + 0.008117746 * max(0.0, 0.007877041 - Q.sum_z_dr2) / 0.002242297   # +0.8%  sum_z_dr2 < 0.007877
        + 0.007806111 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +0.8%  tau21_b2 < 0.2352
        - 0.00427396 * max(0.0, Q.sd_mass - 45.595) / 24.70213   # -0.4%  sd_mass > 45.59
        + 0.004189941 * max(0.0, Q.mass_top20 - 73.35236) / 11.28399   # +0.4%  mass_top20 > 73.35
        + 0.003468328 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.3%  n_dr_0p2_0p4 < 6
        + 0.003425474 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +0.3%  tau1 < 0.07709
        - 0.002775249 * max(0.0, Q.mass_top20 - 85.79457) / 7.06893   # -0.3%  mass_top20 > 85.79
        - 0.002632288 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.3%  psi_0p3 > 0.998
        + 0.002590488 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +0.3%  mass < 92.86
        - 0.002516316 * max(0.0, 0.006403325 - Q.sum_z_dr2) / 0.001406912   # -0.3%  sum_z_dr2 < 0.006403
        - 0.002287555 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt) / 11.66143   # -0.2%  tau21_b2 < 0.2352 and sum_pt < 1261
        - 0.002220404 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003362293   # -0.2%  tau21_b2 < 0.2352 and psi_0p3 > 0.9299
        - 0.00208695 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3) / 6.520272e-05   # -0.2%  n_dr_0p2_0p4 < 6 and e3 < 7.876e-05
        - 0.002084013 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # -0.2%  psi_0p2 > 0.9314
        + 0.002035867 * max(0.0, 9.0 - Q.n_dr_0p2_0p4) / 3.177318   # +0.2%  n_dr_0p2_0p4 < 9
        - 0.001789697 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # -0.2%  tau1 < 0.08787
        + 0.001451283 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.z_top50_slots - 0.978741) / 1.779326e-05   # +0.1%  psi_0p3 > 0.9974 and z_top50_slots > 0.9787
        - 0.001249532 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.4947602 - Q.z_dr_0_0p05) / 0.572942   # -0.1%  mass < 92.86 and z_dr_0_0p05 < 0.4948
        - 0.001016833 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 5.106465e-05   # -0.1%  tau21_b2 < 0.2352 and mass_over_sum_pt_sq < 0.007873
        + 0.000948542 * max(0.0, 0.006403325 - Q.sum_z_dr2) * max(0.0, Q.psi_0p3 - 0.9980008) / 9.11248e-07   # +0.1%  sum_z_dr2 < 0.006403 and psi_0p3 > 0.998
        + 0.0009179099 * max(0.0, Q.sd_mass - 98.05743) / 4.276978   # +0.1%  sd_mass > 98.06
        - 0.0008776347 * max(0.0, 18.0 - Q.n_dr_0_0p05) / 7.289224   # -0.1%  n_dr_0_0p05 < 18
        + 0.0008488755 * max(0.0, 0.2982 - Q.max_dr) / 0.01350087   # +0.1%  max_dr < 0.2982
        + 0.0007655741 * max(0.0, Q.psi_0p2 - 0.9313699) * max(0.0, 6.0 - Q.n_pt_above_50) / 0.0408466   # +0.1%  psi_0p2 > 0.9314 and n_pt_above_50 < 6
        - 0.0006694936 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -0.1%  tau21 < 0.3862
        + 0.0006102055 * max(0.0, 0.000404306 - Q.lam2) / 5.964336e-05   # +0.1%  lam2 < 0.0004043
        - 0.0005542565 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -9.840088) / 1.453315   # -0.1%  tau21_b2 < 0.2352 and orientation_deg > -9.84
        - 0.0004643769 * max(0.0, Q.psi_0p3 - 0.9995915) / 8.716828e-05   # -0.0%  psi_0p3 > 0.9996
        + 0.0004512754 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.0%  psi_0p3 > 0.9974
        - 0.0004173575 * max(0.0, 0.008241985 - Q.lam1) * max(0.0, Q.zdr_0 - 0.01564747) / 7.458275e-07   # -0.0%  lam1 < 0.008242 and zdr_0 > 0.01565
        - 0.00040675 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, 28.39396 - Q.pt1_dr01) / 1.35449   # -0.0%  tau21 < 0.3862 and pt1_dr01 < 28.39
        + 0.0004031603 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.004060732 - Q.soft6_z) / 0.003068066   # +0.0%  n_dr_0p2_0p4 < 6 and soft6_z < 0.004061
        + 0.0003951725 * max(0.0, 0.09573228 - Q.z_dr_0_0p05) / 0.02128126   # +0.0%  z_dr_0_0p05 < 0.09573
        - 0.000388787 * max(0.0, 91.19 - Q.mass) * max(0.0, 6.0 - Q.n_dr_0_0p05) / 1.647199   # -0.0%  mass < 91.19 and n_dr_0_0p05 < 6
        - 0.0003502151 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.1063277 - Q.z_2) / 0.0009585458   # -0.0%  tau21_b2 < 0.2352 and z_2 < 0.1063
        + 0.0003430654 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.n_real_top50 - 34.0) / 0.287919   # +0.0%  tau21_b2 < 0.2352 and n_real_top50 > 34
        + 0.0003124805 * max(0.0, Q.z_dr_0p05_0p1 - 0.6469679) / 0.02199777   # +0.0%  z_dr_0p05_0p1 > 0.647
        + 0.0002178139 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.ptdr0_2 - 11.87288) / 2.425576   # +0.0%  n_dr_0p2_0p4 < 6 and ptdr0_2 > 11.87
        + 0.0002117327 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.1825048 - Q.sj2_dr) / 0.06647299   # +0.0%  n_dr_0p2_0p4 < 6 and sj2_dr < 0.1825
        + 0.0001860324 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.n_dr_0p1_0p2 - 14.0) / 0.001240426   # +0.0%  psi_0p3 > 0.998 and n_dr_0p1_0p2 > 14
        - 0.0001856228 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.max_dr - 0.3618493) / 0.02505788   # -0.0%  n_dr_0p2_0p4 < 6 and max_dr > 0.3618
        - 0.0001585364 * max(0.0, 0.1939977 - Q.max_dr) / 0.001716714   # -0.0%  max_dr < 0.194
        + 0.0001484314 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, -0.03771973 - Q.eta_12) / 0.01514749   # +0.0%  n_dr_0p2_0p4 < 6 and eta_12 < -0.03772
        - 0.0001076168 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.001696484   # -0.0%  tau21_b2 < 0.2352 and sj2_dr > 0.1938
        + 9.022968e-05 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.ptdr0_4 - 10.58726) / 0.6314498   # +0.0%  n_dr_0p2_0p4 < 6 and ptdr0_4 > 10.59
        - 7.938068e-05 * max(0.0, 0.00363788 - Q.sum_zz_dr2) / 0.0004963098   # -0.0%  sum_zz_dr2 < 0.003638
        + 7.753977e-05 * max(0.0, 0.09573228 - Q.z_dr_0_0p05) * max(0.0, 5.541989 - Q.sj3_mass3) / 0.03328332   # +0.0%  z_dr_0_0p05 < 0.09573 and sj3_mass3 < 5.542
        - 7.378157e-05 * max(0.0, 0.1578973 - Q.tau21) / 0.003122095   # -0.0%  tau21 < 0.1579
        - 5.420305e-05 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.eta_1 - 0.0001021922) / 0.02682082   # -0.0%  n_dr_0p2_0p4 < 6 and eta_1 > 0.0001022
        + 4.466672e-05 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, 0.009867229 - Q.z_9) / 3.934916e-06   # +0.0%  tau21 < 0.3862 and z_9 < 0.009867
        + 4.440963e-05 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, -0.07593384 - Q.phi_6) / 0.003845976   # +0.0%  n_dr_0p2_0p4 < 6 and phi_6 < -0.07593
        + 3.023284e-05 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 15.0) / 1.394886   # +0.0%  n_dr_0p2_0p4 < 6 and n_dr_0p1_0p2 > 15
        - 2.632721e-05 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.zdr_3 - 0.00396157) / 0.0001179414   # -0.0%  tau21_b2 < 0.2352 and zdr_3 > 0.003962
        - 1.359839e-05 * max(0.0, 0.00363788 - Q.sum_zz_dr2) * max(0.0, Q.ptdr0_4 - 12.44629) / 1.91781e-07   # -0.0%  sum_zz_dr2 < 0.003638 and ptdr0_4 > 12.45
        - 1.348472e-05 * max(0.0, Q.mass_top20 - 73.35236) * max(0.0, 10.10938 - Q.pt_9) / 0.1983943   # -0.0%  mass_top20 > 73.35 and pt_9 < 10.11
        - 3.546893e-06 * max(0.0, 0.07708632 - Q.tau1) * max(0.0, Q.ptdr0_8 - 4.453578) / 0.0007351971   # -0.0%  tau1 < 0.07709 and ptdr0_8 > 4.454
        - 1.316898e-06 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.02947847   # -0.0%  mass < 82.85 and psi_0p3 < 0.9985
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 16.54;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.53724 * (0.1239471
        - 0.1200416 * max(0.0, 0.009614971 - Q.lam1_plus_lam2) / 0.003502545   # -12.0%  lam1_plus_lam2 < 0.009615
        + 0.05599788 * max(0.0, Q.mass - 87.36377) / 16.57741   # +5.6%  mass > 87.36
        + 0.04593386 * max(0.0, Q.mass - 74.25181) / 24.07648   # +4.6%  mass > 74.25
        + 0.04057812 * max(0.0, Q.mass - 64.48544) / 31.19164   # +4.1%  mass > 64.49
        + 0.03998197 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt) / 2.336698   # +4.0%  mass_over_sum_pt > 0.07697 and sum_pt < 1116
        - 0.03911873 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -3.9%  mass_top50 > 82.04
        + 0.0373389 * max(0.0, 0.007877041 - Q.sum_z_dr2) / 0.002242297   # +3.7%  sum_z_dr2 < 0.007877
        + 0.03451788 * max(0.0, Q.sum_z_dr2_top20 - 0.008031209) / 0.002906852   # +3.5%  sum_z_dr2_top20 > 0.008031
        - 0.03342292 * max(0.0, Q.n_for_90pct - 7.0) / 13.93571   # -3.3%  n_for_90pct > 7
        - 0.03305129 * max(0.0, Q.mass - 101.0497) / 12.31084   # -3.3%  mass > 101
        + 0.03093618 * max(0.0, 1028.184 - Q.sum_pt) / 26.95739   # +3.1%  sum_pt < 1028
        - 0.02908881 * max(0.0, 39.0 - Q.n_for_90pct) / 18.29776   # -2.9%  n_for_90pct < 39
        - 0.02721874 * max(0.0, Q.sum_z_dr2_top20 - 0.006043209) / 0.003593885   # -2.7%  sum_z_dr2_top20 > 0.006043
        - 0.02328753 * max(0.0, Q.z_top50_slots - 0.9704436) / 0.02331682   # -2.3%  z_top50_slots > 0.9704
        - 0.02319321 * max(0.0, 0.02146578 - Q.sum_z_dr2_top15) / 0.01470441   # -2.3%  sum_z_dr2_top15 < 0.02147
        + 0.02318499 * max(0.0, 0.008840538 - Q.sum_z_dr2_top40) / 0.003108622   # +2.3%  sum_z_dr2_top40 < 0.008841
        + 0.02314173 * max(0.0, 0.007463985 - Q.sum_z_dr2_top30) / 0.00233708   # +2.3%  sum_z_dr2_top30 < 0.007464
        - 0.02194938 * max(0.0, Q.sum_z_dr2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt) / 0.0005328922   # -2.2%  sum_z_dr2_top40 > 0.005197 and log_sum_pt < 7.017
        - 0.02149095 * max(0.0, Q.mass - 125.1) / 7.098976   # -2.1%  mass > 125.1
        - 0.02096403 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # -2.1%  sum_pt_top40 < 1025
        + 0.02033189 * max(0.0, Q.sum_z_dr2_top40 - 0.005196966) / 0.004725809   # +2.0%  sum_z_dr2_top40 > 0.005197
        - 0.02019337 * max(0.0, Q.sum_pt_top50 - 889.8503) / 149.655   # -2.0%  sum_pt_top50 > 889.9
        + 0.01757879 * max(0.0, Q.mass_top50 - 117.0487) / 7.705579   # +1.8%  mass_top50 > 117
        + 0.01688811 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +1.7%  sum_pt < 1017
        - 0.01642916 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # -1.6%  mass_over_sum_pt > 0.07697
        - 0.01625949 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -1.6%  n_dr_0p2_0p4 < 15
        - 0.01499397 * max(0.0, 6.930088 - Q.log_sum_pt) / 0.02522479   # -1.5%  log_sum_pt < 6.93
        + 0.01446748 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +1.4%  lam1 < 0.007671
        + 0.01171019 * max(0.0, 1001.523 - Q.sum_pt_top40) / 26.72886   # +1.2%  sum_pt_top40 < 1002
        - 0.01135239 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # -1.1%  lam1 < 0.01174
        + 0.01060661 * max(0.0, Q.n_dr_0p2_0p4 - 3.0) / 6.482203   # +1.1%  n_dr_0p2_0p4 > 3
        - 0.008515434 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -0.9%  mass_over_sum_pt > 0.1182
        - 0.008094605 * max(0.0, Q.e3 - 5.13841e-05) / 6.821576e-05   # -0.8%  e3 > 5.138e-05
        - 0.007984185 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -0.8%  psi_0p3 > 0.9943
        + 0.0078011 * max(0.0, 9.0 - Q.n_dr_0p2_0p4) / 3.177318   # +0.8%  n_dr_0p2_0p4 < 9
        + 0.007223721 * max(0.0, Q.sum_pt_top30 - 978.0762) / 49.23409   # +0.7%  sum_pt_top30 > 978.1
        + 0.006864738 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.6773544   # +0.7%  sum_pt < 1028 and dr_max_012 > 0.1828
        - 0.006581656 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.5693287   # -0.7%  sum_pt < 1017 and dr_max_012 > 0.1828
        + 0.005907638 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # +0.6%  C2 > 0.06656
        + 0.005582334 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # +0.6%  tau2 < 0.0795
        + 0.004263438 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.4%  e2 > 0.04755
        + 0.004017582 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +0.4%  n_particles > 38
        + 0.003827467 * max(0.0, 1.219727 - Q.soft5_pt) / 0.2575   # +0.4%  soft5_pt < 1.22
        + 0.003765927 * max(0.0, 0.0795038 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.38202) / 0.1361121   # +0.4%  tau2 < 0.0795 and sj3_mass1 > 13.38
        + 0.003324779 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.3%  sj2_dr > 0.2232
        - 0.00290187 * max(0.0, 0.007463985 - Q.sum_z_dr2_top30) * max(0.0, Q.sj2_dr - 0.1512157) / 6.78729e-05   # -0.3%  sum_z_dr2_top30 < 0.007464 and sj2_dr > 0.1512
        + 0.002724418 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.03720487   # +0.3%  z_dr_0p1_0p2 > 0.334
        + 0.002059224 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # +0.2%  e3 > 0.0003372
        + 0.00168845 * max(0.0, 6.930088 - Q.log_sum_pt) * max(0.0, 0.05595395 - Q.dr01) / 0.0008519083   # +0.2%  log_sum_pt < 6.93 and dr01 < 0.05595
        - 0.001653457 * max(0.0, Q.mass_top20 - 119.2969) / 1.8402   # -0.2%  mass_top20 > 119.3
        - 0.001430782 * max(0.0, 0.3628388 - Q.psi_0p1) / 0.02326895   # -0.1%  psi_0p1 < 0.3628
        + 0.001407439 * max(0.0, Q.eta_0 - -0.01382675) / 0.02505234   # +0.1%  eta_0 > -0.01383
        + 0.001204702 * max(0.0, Q.n_dr_0p1_0p2 - 19.0) / 1.732133   # +0.1%  n_dr_0p1_0p2 > 19
        + 0.0009338474 * max(0.0, Q.n_particles - 51.0) / 3.973704   # +0.1%  n_particles > 51
        + 0.0008438993 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.09686657   # +0.1%  n_dr_0p2_0p4 < 15 and z_dr_0p1_0p2 > 0.334
        - 0.0007469495 * max(0.0, Q.n_particles - 51.0) * max(0.0, 1.091797 - Q.soft1_pt) / 1.364428   # -0.1%  n_particles > 51 and soft1_pt < 1.092
        - 0.0005950785 * max(0.0, Q.mass - 74.25181) * max(0.0, 5.253859 - Q.sj3_mass2) / 3.184986   # -0.1%  mass > 74.25 and sj3_mass2 < 5.254
        - 0.0005292258 * max(0.0, Q.mass - 64.48544) * max(0.0, Q.zdr_0 - 0.0008718296) / 0.4297651   # -0.1%  mass > 64.49 and zdr_0 > 0.0008718
        - 0.0004809778 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr1_9 - 0.1651809) / 0.6680865   # -0.0%  sum_pt < 1017 and dr1_9 > 0.1652
        + 0.0004205532 * max(0.0, Q.sum_pt_top50 - 889.8503) * max(0.0, -0.1315918 - Q.eta_9) / 0.3539655   # +0.0%  sum_pt_top50 > 889.9 and eta_9 < -0.1316
        - 0.0004133562 * max(0.0, Q.mass_top20 - 119.2969) * max(0.0, Q.absphi_1 - 0.07141113) / 0.06373091   # -0.0%  mass_top20 > 119.3 and absphi_1 > 0.07141
        + 0.0003313741 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, 13.0 - Q.n_for_90pct) / 15.66746   # +0.0%  sum_pt < 1017 and n_for_90pct < 13
        - 0.0001915358 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # -0.0%  log_sum_pt > 7.139
        - 0.0001682013 * max(0.0, 0.8316924 - Q.z_top15_slots) / 0.04702674   # -0.0%  z_top15_slots < 0.8317
        + 0.0001471395 * max(0.0, Q.sum_z_dr2_top20 - 0.006043209) * max(0.0, 0.08173536 - Q.sj3_z2) / 1.545259e-07   # +0.0%  sum_z_dr2_top20 > 0.006043 and sj3_z2 < 0.08174
        - 7.814335e-05 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, Q.mean_phi - 2.298159e-05) / 4.904976e-06   # -0.0%  sj2_dr > 0.2232 and mean_phi > 2.298e-05
        + 4.462999e-05 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # +0.0%  e2 > 0.06524
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 43.8;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 43.8027 * (-0.02070938
        - 0.2559244 * max(0.0, 160.8 - Q.mass) / 72.78344   # -25.6%  mass < 160.8
        + 0.2248985 * max(0.0, 162.8363 - Q.mass) / 74.60496   # +22.5%  mass < 162.8
        + 0.157458 * max(0.0, 172.8 - Q.mass) / 83.78792   # +15.7%  mass < 172.8
        - 0.1244479 * max(0.0, 143.7876 - Q.mass) / 57.99337   # -12.4%  mass < 143.8
        + 0.06342718 * max(0.0, 138.8977 - Q.mass_top50) / 55.01559   # +6.3%  mass_top50 < 138.9
        - 0.04204939 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -4.2%  mass_top40 < 163.3
        + 0.02766914 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +2.8%  mass < 92.86
        - 0.0173498 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # -1.7%  mass_top50 < 92.17
        + 0.0152884 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +1.5%  mass < 82.85
        + 0.006941762 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +0.7%  mass_top40 < 80.89
        + 0.005464737 * max(0.0, 0.006259772 - Q.sum_z_dr2_top40) / 0.001461823   # +0.5%  sum_z_dr2_top40 < 0.00626
        + 0.00479337 * max(0.0, 125.1 - Q.mass_top40) / 45.33615   # +0.5%  mass_top40 < 125.1
        + 0.00476658 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +0.5%  sum_pt < 986.1
        - 0.004681903 * max(0.0, Q.sum_z_dr2_top30 - 0.0008564881) / 0.007661497   # -0.5%  sum_z_dr2_top30 > 0.0008565
        - 0.004370416 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -0.4%  mass < 64.49
        - 0.004133852 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -0.4%  log_sum_pt < 6.903
        + 0.003719271 * max(0.0, 73.33139 - Q.mass_top30) / 11.37701   # +0.4%  mass_top30 < 73.33
        + 0.003670837 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # +0.4%  sum_z_dr > 0.0975
        + 0.003664448 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.4%  lam1 < 0.004673
        - 0.002700729 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -0.3%  tau1 < 0.05445
        + 0.002666862 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # +0.3%  psi_0p3 > 0.9943
        + 0.00265503 * max(0.0, 956.2133 - Q.sum_pt_top40) / 14.00451   # +0.3%  sum_pt_top40 < 956.2
        - 0.002475152 * max(0.0, Q.sum_z_dr - 0.1402186) / 0.001871547   # -0.2%  sum_z_dr > 0.1402
        - 0.001792994 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt) / 447.38   # -0.2%  mass_top40 < 80.89 and sum_pt < 1035
        - 0.001494634 * max(0.0, Q.z_top40_slots - 0.9574183) / 0.02831517   # -0.1%  z_top40_slots > 0.9574
        + 0.001413368 * max(0.0, 125.1 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 42.49581   # +0.1%  mass_top40 < 125.1 and n_dr_0p2_0p4 > 9
        + 0.001379114 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +0.1%  LHA > 0.4042
        + 0.001254879 * max(0.0, Q.D2 - 2.178951) / 1.450771   # +0.1%  D2 > 2.179
        + 0.001073638 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +0.1%  lam1 > 0.01174
        + 0.0008928178 * max(0.0, Q.z_top5 - 0.7963975) / 0.006402524   # +0.1%  z_top5 > 0.7964
        - 0.0008651431 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, 0.001776308 - Q.lam2) / 0.005060516   # -0.1%  sum_pt_top40 < 956.2 and lam2 < 0.001776
        - 0.0007142127 * max(0.0, 0.00217213 - Q.sum_z_dr2_top40) / 0.0002075259   # -0.1%  sum_z_dr2_top40 < 0.002172
        - 0.00064473 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.z_top50_slots - 0.9995789) / 0.005945365   # -0.1%  mass < 92.86 and z_top50_slots > 0.9996
        - 0.0006151887 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258) / 5.834847   # -0.1%  sum_pt_top40 < 956.2 and soft4_pt > 1.789
        + 0.0005958293 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40) / 104.3982   # +0.1%  mass_top40 < 80.89 and sum_pt_top40 < 906.6
        - 0.0004976588 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -0.0%  z_dr_0_0p05 > 0.8789
        + 0.0003370301 * max(0.0, Q.n_for_90pct - 35.0) / 0.4575429   # +0.0%  n_for_90pct > 35
        - 0.0003339639 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) / 0.006411021   # -0.0%  z_dr_0p1_0p2 > 0.6882
        + 0.0003198063 * max(0.0, Q.sj3_dr13 - 0.1654269) / 0.05535154   # +0.0%  sj3_dr13 > 0.1654
        + 0.0003065462 * max(0.0, 986.0565 - Q.sum_pt) * max(0.0, Q.soft5_z - 0.002036949) / 0.003199714   # +0.0%  sum_pt < 986.1 and soft5_z > 0.002037
        + 0.0001154491 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # +0.0%  e3 > 0.0003372
        - 5.221277e-05 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # -0.0%  mass_over_sum_pt > 0.1709
        - 3.612729e-05 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.psi_0p1 - 0.9184255) / 0.111142   # -0.0%  sum_pt_top40 < 956.2 and psi_0p1 > 0.9184
        + 3.361394e-05 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047) / 1.704178   # +0.0%  sum_pt_top40 < 956.2 and soft3_pt > 2.873
        + 1.332517e-05 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, 154.25 - Q.pt_2) / 0.4838258   # +0.0%  z_dr_0p1_0p2 > 0.6882 and pt_2 < 154.2
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 17.69;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.68956 * (0.0502683
        - 0.1339273 * max(0.0, 0.1207452 - Q.sum_z_dr) / 0.05578614   # -13.4%  sum_z_dr < 0.1207
        + 0.1018581 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # +10.2%  tau1 < 0.1507
        + 0.07617559 * max(0.0, Q.mass - 78.26182) / 21.33658   # +7.6%  mass > 78.26
        - 0.06880034 * max(0.0, Q.mass - 64.48544) / 31.19164   # -6.9%  mass > 64.49
        + 0.04745019 * max(0.0, Q.e2 - 0.01256572) / 0.01912291   # +4.7%  e2 > 0.01257
        + 0.04677217 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +4.7%  mass < 89.74
        - 0.04353052 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -4.4%  mass < 101
        + 0.03872902 * max(0.0, 0.01655983 - Q.sum_z_dr2_top20) / 0.01003486   # +3.9%  sum_z_dr2_top20 < 0.01656
        - 0.03550546 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -3.6%  e2 < 0.04359
        - 0.02668173 * max(0.0, 65.20727 - Q.sj2_mass1) / 36.50254   # -2.7%  sj2_mass1 < 65.21
        - 0.01825719 * max(0.0, Q.mass - 143.7876) / 3.946979   # -1.8%  mass > 143.8
        - 0.01784122 * max(0.0, Q.mass_top30 - 78.53034) / 14.31024   # -1.8%  mass_top30 > 78.53
        + 0.01471084 * max(0.0, 935.1043 - Q.sum_pt_top15) / 95.709   # +1.5%  sum_pt_top15 < 935.1
        + 0.01428814 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +1.4%  mass_top50 > 136.8
        - 0.01411962 * max(0.0, 0.01976735 - Q.sum_z_dr2_top10) / 0.01371247   # -1.4%  sum_z_dr2_top10 < 0.01977
        + 0.01340817 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.3%  z_dr_0p2_0p4 < 0.09123
        - 0.01283345 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -1.3%  n_dr_0p2_0p4 < 11
        - 0.01237874 * max(0.0, 59.40777 - Q.mass_top5) / 33.74882   # -1.2%  mass_top5 < 59.41
        + 0.01191828 * max(0.0, 0.0001086251 - Q.e3) / 5.924919e-05   # +1.2%  e3 < 0.0001086
        - 0.01124238 * max(0.0, 0.01976735 - Q.sum_z_dr2_top10) * max(0.0, 0.4357228 - Q.max_dr) / 0.001259784   # -1.1%  sum_z_dr2_top10 < 0.01977 and max_dr < 0.4357
        + 0.01102245 * max(0.0, 86.4 - Q.mass) / 13.67632   # +1.1%  mass < 86.4
        + 0.01007009 * max(0.0, 0.04828819 - Q.tau2) / 0.02103241   # +1.0%  tau2 < 0.04829
        + 0.009926801 * max(0.0, Q.mass_top30 - 89.17293) / 10.17054   # +1.0%  mass_top30 > 89.17
        - 0.009243834 * max(0.0, Q.mass - 162.8363) / 1.509832   # -0.9%  mass > 162.8
        - 0.009237339 * max(0.0, 0.005312783 - Q.sum_z_dr2_top20) / 0.001437214   # -0.9%  sum_z_dr2_top20 < 0.005313
        - 0.009195039 * max(0.0, 0.04466492 - Q.tau1) / 0.005217805   # -0.9%  tau1 < 0.04466
        + 0.008715977 * max(0.0, 2.975532 - Q.D2) / 0.9290697   # +0.9%  D2 < 2.976
        + 0.008692055 * max(0.0, 0.02078982 - Q.zdr_0) / 0.01170192   # +0.9%  zdr_0 < 0.02079
        - 0.008619643 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.9%  psi_0p3 > 0.998
        - 0.00860758 * max(0.0, 0.006756161 - Q.sum_z_dr2_top3) / 0.003650633   # -0.9%  sum_z_dr2_top3 < 0.006756
        + 0.008246791 * max(0.0, Q.mass_top5 - 22.18342) / 12.67741   # +0.8%  mass_top5 > 22.18
        + 0.00806522 * max(0.0, 59.40777 - Q.mass_top5) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1) / 22.61156   # +0.8%  mass_top5 < 59.41 and z_dr_0p05_0p1 < 0.8509
        - 0.007766704 * max(0.0, Q.z_dr_0_0p05 - 0.7674734) / 0.05221088   # -0.8%  z_dr_0_0p05 > 0.7675
        - 0.007383571 * max(0.0, 72.18744 - Q.mass_top15) / 20.20734   # -0.7%  mass_top15 < 72.19
        - 0.007092343 * max(0.0, Q.pt_dispersion - 0.27462) / 0.06481184   # -0.7%  pt_dispersion > 0.2746
        + 0.006933232 * max(0.0, 0.5760704 - Q.z_top2_slots) * max(0.0, 0.002970691 - Q.soft9_z) / 0.0002253038   # +0.7%  z_top2_slots < 0.5761 and soft9_z < 0.002971
        + 0.006828748 * max(0.0, 0.5760704 - Q.z_top2_slots) / 0.2302819   # +0.7%  z_top2_slots < 0.5761
        + 0.006778658 * max(0.0, 0.008824206 - Q.zdr_1) / 0.00383206   # +0.7%  zdr_1 < 0.008824
        - 0.006253378 * max(0.0, Q.mass_over_sum_pt - 0.1606361) / 0.001344467   # -0.6%  mass_over_sum_pt > 0.1606
        - 0.005675403 * max(0.0, 0.01976735 - Q.sum_z_dr2_top10) * max(0.0, Q.psi_0p3 - 0.9985421) / 6.34706e-06   # -0.6%  sum_z_dr2_top10 < 0.01977 and psi_0p3 > 0.9985
        + 0.005163013 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +0.5%  dr_0 < 0.06413
        - 0.005148437 * max(0.0, 0.005402331 - Q.sum_z_dr2_top30) / 0.001231293   # -0.5%  sum_z_dr2_top30 < 0.005402
        - 0.003503672 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.4%  sum_pt_top10 > 943.7
        + 0.003368177 * max(0.0, 2.975532 - Q.D2) * max(0.0, 0.08728027 - Q.absphi_4) / 0.03660338   # +0.3%  D2 < 2.976 and absphi_4 < 0.08728
        + 0.003153114 * max(0.0, Q.sj3_dr_min - 0.1204829) / 0.02211875   # +0.3%  sj3_dr_min > 0.1205
        - 0.003145806 * max(0.0, Q.mass_top50 - 136.785) * max(0.0, Q.soft5_z - 0.001594761) / 0.001639485   # -0.3%  mass_top50 > 136.8 and soft5_z > 0.001595
        + 0.002882135 * max(0.0, 0.007043525 - Q.zdr_2) / 0.003147169   # +0.3%  zdr_2 < 0.007044
        + 0.002778916 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897) / 0.001013644   # +0.3%  mass > 162.8 and soft5_z > 0.001435
        + 0.002634461 * max(0.0, Q.sum_z_dr2_top40 - 0.02497133) / 0.0004755075   # +0.3%  sum_z_dr2_top40 > 0.02497
        - 0.00242571 * max(0.0, Q.M2 - 0.06621477) / 0.0101271   # -0.2%  M2 > 0.06621
        + 0.002395032 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.pt1_dr01 - 5.351077) / 32.20836   # +0.2%  mass < 101 and pt1_dr01 > 5.351
        - 0.002211699 * max(0.0, 0.002575211 - Q.sum_z_dr2) / 0.0002582872   # -0.2%  sum_z_dr2 < 0.002575
        - 0.002143717 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # -0.2%  mass_top10 > 71.78
        + 0.002007388 * max(0.0, Q.n_particles - 41.0) / 8.920024   # +0.2%  n_particles > 41
        - 0.001968751 * max(0.0, Q.mass - 78.26182) * max(0.0, 0.0006154841 - Q.lam2) / 0.0007406418   # -0.2%  mass > 78.26 and lam2 < 0.0006155
        + 0.001903232 * max(0.0, Q.mass_top5 - 22.18342) * max(0.0, Q.lam2 - 0.0009731947) / 0.02103127   # +0.2%  mass_top5 > 22.18 and lam2 > 0.0009732
        + 0.001836528 * max(0.0, Q.mass_top50 - 172.8) / 0.5062389   # +0.2%  mass_top50 > 172.8
        + 0.001789219 * max(0.0, Q.e2 - 0.03263075) / 0.006288927   # +0.2%  e2 > 0.03263
        - 0.001706458 * max(0.0, 2.975532 - Q.D2) * max(0.0, Q.sj2_dr - 0.2070855) / 0.0198522   # -0.2%  D2 < 2.976 and sj2_dr > 0.2071
        + 0.001528852 * max(0.0, 0.01541561 - Q.e2) / 0.001436351   # +0.2%  e2 < 0.01542
        + 0.001516293 * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0004356739   # +0.2%  psi_0p3 > 0.9985
        - 0.00141488 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # -0.1%  sum_pt_top50 < 959.1
        - 0.00129191 * max(0.0, 750.7313 - Q.sum_pt_top20) / 7.067235   # -0.1%  sum_pt_top20 < 750.7
        + 0.001246845 * max(0.0, 2.975532 - Q.D2) * max(0.0, Q.soft2_dr - 0.2302212) / 0.03178642   # +0.1%  D2 < 2.976 and soft2_dr > 0.2302
        + 0.001224582 * max(0.0, 65.20727 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 7.769597) / 168.6958   # +0.1%  sj2_mass1 < 65.21 and sj2_mass2 > 7.77
        + 0.00119167 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, 0.2384186 - Q.dr_9) / 1.423778   # +0.1%  sum_pt_top50 < 959.1 and dr_9 < 0.2384
        - 0.001078502 * max(0.0, 0.09021906 - Q.z_2nd) / 0.004040099   # -0.1%  z_2nd < 0.09022
        + 0.001002067 * max(0.0, Q.mass_over_sum_pt - 0.1606361) * max(0.0, 0.09696199 - Q.z_3) / 5.287414e-05   # +0.1%  mass_over_sum_pt > 0.1606 and z_3 < 0.09696
        - 0.0008352902 * max(0.0, 0.01976735 - Q.sum_z_dr2_top10) * max(0.0, 0.8476928 - Q.tau32) / 0.001347663   # -0.1%  sum_z_dr2_top10 < 0.01977 and tau32 < 0.8477
        + 0.0008347024 * max(0.0, 0.005402331 - Q.sum_z_dr2_top30) * max(0.0, Q.psi_0p3 - 0.9985421) / 4.280363e-07   # +0.1%  sum_z_dr2_top30 < 0.005402 and psi_0p3 > 0.9985
        - 0.0008057437 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, 53.4375 - Q.pt_7) / 18.94242   # -0.1%  mass_top50 > 160.8 and pt_7 < 53.44
        - 0.0007885413 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # -0.1%  sum_pt < 986.1
        - 0.0006551942 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, 0.2748443 - Q.dr0_3) / 1.93979   # -0.1%  sum_pt_top50 < 959.1 and dr0_3 < 0.2748
        + 0.0006231192 * max(0.0, Q.psi_0p3 - 0.9985421) * max(0.0, Q.soft5_z - 0.001434897) / 3.081682e-07   # +0.1%  psi_0p3 > 0.9985 and soft5_z > 0.001435
        + 0.0005874846 * max(0.0, 59.40777 - Q.mass_top5) * max(0.0, Q.zdr_5 - 0.005134657) / 0.01177167   # +0.1%  mass_top5 < 59.41 and zdr_5 > 0.005135
        + 0.0005279687 * max(0.0, Q.e2 - 0.03263075) * max(0.0, 0.842732 - Q.tau43) / 0.0003536832   # +0.1%  e2 > 0.03263 and tau43 < 0.8427
        - 0.0005178767 * max(0.0, Q.e2 - 0.03263075) * max(0.0, Q.soft2_pt - 1.413232) / 0.001032157   # -0.1%  e2 > 0.03263 and soft2_pt > 1.413
        + 0.0004965191 * max(0.0, Q.sum_pt_top10 - 943.6922) * max(0.0, 0.6469679 - Q.z_dr_0p05_0p1) / 6.713086   # +0.0%  sum_pt_top10 > 943.7 and z_dr_0p05_0p1 < 0.647
        - 0.0004353015 * max(0.0, 986.0565 - Q.sum_pt) * max(0.0, Q.dr_3 - 0.01436663) / 0.7268239   # -0.0%  sum_pt < 986.1 and dr_3 > 0.01437
        - 0.0004327472 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109) / 0.0004735247   # -0.0%  mass_top50 > 160.8 and soft4_z > 0.001721
        + 0.0004231807 * max(0.0, 0.5760704 - Q.z_top2_slots) * max(0.0, Q.orientation_deg - -44.87344) / 11.62503   # +0.0%  z_top2_slots < 0.5761 and orientation_deg > -44.87
        + 0.0004118782 * max(0.0, 0.005402331 - Q.sum_z_dr2_top30) * max(0.0, 631.275 - Q.sum_pt_top5) / 0.06578377   # +0.0%  sum_z_dr2_top30 < 0.005402 and sum_pt_top5 < 631.3
        + 0.000353167 * max(0.0, 0.01541561 - Q.e2) * max(0.0, 0.7904923 - Q.pt2_over_pt0) / 0.000518365   # +0.0%  e2 < 0.01542 and pt2_over_pt0 < 0.7905
        + 0.0003178942 * max(0.0, Q.mass_top50 - 160.8) / 1.246461   # +0.0%  mass_top50 > 160.8
        - 0.0002829889 * max(0.0, Q.mass_top50 - 172.8) * max(0.0, Q.M2 - 0.04577853) / 0.009136797   # -0.0%  mass_top50 > 172.8 and M2 > 0.04578
        - 0.0002734513 * max(0.0, 0.5760704 - Q.z_top2_slots) * max(0.0, Q.absphi_13 - 0.1468506) / 0.001227084   # -0.0%  z_top2_slots < 0.5761 and absphi_13 > 0.1469
        - 0.0002524721 * max(0.0, Q.M2 - 0.06621477) * max(0.0, Q.sj3_dr13 - 0.1348442) / 0.000622409   # -0.0%  M2 > 0.06621 and sj3_dr13 > 0.1348
        - 0.0002078683 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.absphi_4 - 0.004672432) / 0.4741157   # -0.0%  sum_pt_top50 < 959.1 and absphi_4 > 0.004672
        - 0.0002006229 * max(0.0, 0.5760704 - Q.z_top2_slots) * max(0.0, Q.dr0_12 - 0.2665639) / 0.00166507   # -0.0%  z_top2_slots < 0.5761 and dr0_12 > 0.2666
        + 0.0001952476 * max(0.0, 0.2404747 - Q.max_dr) / 0.005136976   # +0.0%  max_dr < 0.2405
        + 0.0001939081 * max(0.0, Q.pt_dispersion - 0.27462) * max(0.0, 30.0 - Q.n_real_top30) / 0.1960387   # +0.0%  pt_dispersion > 0.2746 and n_real_top30 < 30
        + 0.0001675075 * max(0.0, Q.mass_over_sum_pt - 0.1606361) * max(0.0, Q.sj3_mass2 - 4.756459) / 0.01414387   # +0.0%  mass_over_sum_pt > 0.1606 and sj3_mass2 > 4.756
        - 0.0001536103 * max(0.0, 2.975532 - Q.D2) * max(0.0, Q.dr_12 - 0.1269782) / 0.01318085   # -0.0%  D2 < 2.976 and dr_12 > 0.127
        + 0.0001482645 * max(0.0, Q.mass_top50 - 136.785) * max(0.0, 0.1938641 - Q.ptdr0_7) / 0.01379562   # +0.0%  mass_top50 > 136.8 and ptdr0_7 < 0.1939
        + 0.0001416403 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, 0.2277069 - Q.dr1_5) / 1.438963   # +0.0%  sum_pt_top50 < 959.1 and dr1_5 < 0.2277
        - 0.0001188399 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, 0.01034918 - Q.C3) / 0.001987701   # -0.0%  mass_top50 > 160.8 and C3 < 0.01035
        - 8.429421e-05 * max(0.0, Q.mass_top50 - 172.8) * max(0.0, Q.soft2_abseta - 0.3344727) / 0.001133844   # -0.0%  mass_top50 > 172.8 and soft2_abseta > 0.3345
        - 3.699596e-05 * max(0.0, Q.mass_top50 - 172.8) * max(0.0, Q.tau43 - 0.9187998) / 0.001835161   # -0.0%  mass_top50 > 172.8 and tau43 > 0.9188
        - 2.875685e-05 * max(0.0, Q.mass - 78.26182) * max(0.0, Q.psi_0p3 - 0.9853273) / 0.1102552   # -0.0%  mass > 78.26 and psi_0p3 > 0.9853
        - 2.252373e-05 * max(0.0, Q.sum_pt_top10 - 943.6922) * max(0.0, -0.07794189 - Q.eta_0) / 0.002015064   # -0.0%  sum_pt_top10 > 943.7 and eta_0 < -0.07794
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 9.9;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.90038 * (-0.007872457
        + 0.130739 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +13.1%  mass < 92.86
        - 0.07350305 * max(0.0, 0.076787 - Q.sum_z_dr) / 0.02105267   # -7.4%  sum_z_dr < 0.07679
        + 0.06639872 * max(0.0, 0.08589404 - Q.sum_z_dr) / 0.02745918   # +6.6%  sum_z_dr < 0.08589
        - 0.06114973 * max(0.0, 80.35535 - Q.mass_top50) / 11.1399   # -6.1%  mass_top50 < 80.36
        + 0.05073029 * max(0.0, 97.93004 - Q.mass_top50) / 22.03533   # +5.1%  mass_top50 < 97.93
        + 0.04686517 * max(0.0, 0.00818374 - Q.sum_zz_dr2) / 0.002451404   # +4.7%  sum_zz_dr2 < 0.008184
        - 0.02857466 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -2.9%  mass_top40 < 83.33
        - 0.02477301 * max(0.0, 0.006929741 - Q.sum_z_dr2_top30) / 0.002004687   # -2.5%  sum_z_dr2_top30 < 0.00693
        - 0.02369167 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.004495205 - Q.zdr_4) / 0.05697234   # -2.4%  mass < 92.86 and zdr_4 < 0.004495
        - 0.02164006 * max(0.0, 80.4 - Q.mass) / 10.6806   # -2.2%  mass < 80.4
        + 0.02114657 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 43.23581   # +2.1%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21
        + 0.02096792 * max(0.0, 80.35535 - Q.mass_top50) * max(0.0, 0.003932029 - Q.zdr_4) / 0.03420305   # +2.1%  mass_top50 < 80.36 and zdr_4 < 0.003932
        - 0.02096507 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -2.1%  mass < 101
        + 0.02065587 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +2.1%  e2 < 0.02516
        - 0.02026675 * max(0.0, 86.4 - Q.mass_top30) / 18.30352   # -2.0%  mass_top30 < 86.4
        - 0.01918276 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.sum_z_dr2) / 0.005195774   # -1.9%  n_dr_0p2_0p4 < 10 and sum_z_dr2 < 0.005532
        - 0.01902423 * max(0.0, 0.07999061 - Q.mass_over_sum_pt) / 0.01176803   # -1.9%  mass_over_sum_pt < 0.07999
        - 0.01656464 * max(0.0, 0.00625621 - Q.sum_z_dr2_top10) / 0.002474789   # -1.7%  sum_z_dr2_top10 < 0.006256
        - 0.01640199 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # -1.6%  e2 < 0.03876
        - 0.01625047 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -1.6%  lam1 < 0.006717
        + 0.01607017 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +1.6%  e2 < 0.0303
        + 0.01495083 * max(0.0, 0.005402331 - Q.sum_z_dr2_top30) / 0.001231293   # +1.5%  sum_z_dr2_top30 < 0.005402
        + 0.01275979 * max(0.0, 0.00406126 - Q.sum_z_dr2_top10) / 0.001298509   # +1.3%  sum_z_dr2_top10 < 0.004061
        - 0.01105686 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -1.1%  mass < 64.49
        + 0.01041191 * max(0.0, Q.D2 - 2.178951) / 1.450771   # +1.0%  D2 > 2.179
        - 0.009979319 * max(0.0, 0.008031209 - Q.sum_z_dr2_top20) / 0.00309849   # -1.0%  sum_z_dr2_top20 < 0.008031
        + 0.009610456 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +1.0%  mass_top40 < 67.73
        - 0.009187032 * max(0.0, 3.793233e-05 - Q.e3) / 1.004564e-05   # -0.9%  e3 < 3.793e-05
        - 0.008836911 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # -0.9%  mass_over_sum_pt_sq < 0.009595
        + 0.008792625 * max(0.0, 0.07374472 - Q.sum_z_dr) / 0.01915593   # +0.9%  sum_z_dr < 0.07374
        + 0.008335507 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +0.8%  mass_top30 < 60.44
        + 0.007187979 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.3563114 - Q.planar_flow) / 0.8671033   # +0.7%  mass < 101 and planar_flow < 0.3563
        + 0.007106466 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.pt_11 - 6.632422) / 55.73806   # +0.7%  n_dr_0p2_0p4 < 10 and pt_11 > 6.632
        + 0.007004754 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # +0.7%  psi_0p3 > 0.998
        + 0.006533209 * max(0.0, 15.0 - Q.n_dr_0p1_0p2) / 5.139044   # +0.7%  n_dr_0p1_0p2 < 15
        + 0.006530613 * max(0.0, 50.0 - Q.n_real_top50) / 8.569617   # +0.7%  n_real_top50 < 50
        + 0.006116539 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) * max(0.0, 0.02980347 - Q.eta_0) / 0.0001554673   # +0.6%  sum_z_dr2_top15 < 0.009962 and eta_0 < 0.0298
        - 0.006101966 * max(0.0, Q.D2 - 3.814159) / 0.870654   # -0.6%  D2 > 3.814
        + 0.006060532 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) * max(0.0, Q.z_dr_0p1_0p2 - 0.1203437) / 6.825269e-05   # +0.6%  mass_over_sum_pt_sq < 0.009595 and z_dr_0p1_0p2 > 0.1203
        + 0.005992105 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.z_9 - 0.01620892) / 0.2014824   # +0.6%  mass < 101 and z_9 > 0.01621
        - 0.005989571 * max(0.0, 80.4 - Q.mass) * max(0.0, Q.z_9 - 0.01833434) / 0.06571201   # -0.6%  mass < 80.4 and z_9 > 0.01833
        + 0.005815163 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # +0.6%  sum_z_dr2_top15 < 0.009962
        - 0.004885665 * max(0.0, 19.89956 - Q.sj2_mass1) / 2.423382   # -0.5%  sj2_mass1 < 19.9
        + 0.004707863 * max(0.0, 0.00130722 - Q.sum_z_dr2_top10) / 0.0002794102   # +0.5%  sum_z_dr2_top10 < 0.001307
        - 0.004681397 * max(0.0, 0.06030419 - Q.mass_over_sum_pt) / 0.005383371   # -0.5%  mass_over_sum_pt < 0.0603
        - 0.004185992 * max(0.0, Q.z_top5_slots - 0.534626) / 0.08390497   # -0.4%  z_top5_slots > 0.5346
        + 0.00391943 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, Q.max_dr - 0.2982) / 0.0003195731   # +0.4%  psi_0p3 > 0.9897 and max_dr > 0.2982
        - 0.003840853 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_real_top40 - 26.0) / 33.17834   # -0.4%  n_dr_0p2_0p4 < 10 and n_real_top40 > 26
        + 0.003341571 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) / 0.0003002514   # +0.3%  sum_z_dr2_top2 < 0.001057
        + 0.003266128 * max(0.0, Q.psi_0p2 - 0.9935324) / 0.001465572   # +0.3%  psi_0p2 > 0.9935
        - 0.003192767 * max(0.0, Q.C2_b2 - 0.01330402) / 0.00729146   # -0.3%  C2_b2 > 0.0133
        + 0.003152602 * max(0.0, Q.z_top40_slots - 0.996191) * max(0.0, 0.006580753 - Q.C2_b2) / 4.241011e-06   # +0.3%  z_top40_slots > 0.9962 and C2_b2 < 0.006581
        + 0.00309776 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.3%  n_dr_0p2_0p4 < 6
        + 0.00307305 * max(0.0, Q.psi_0p2 - 0.9313699) * max(0.0, 0.08052407 - Q.dr_13) / 0.001040005   # +0.3%  psi_0p2 > 0.9314 and dr_13 < 0.08052
        - 0.002909722 * max(0.0, Q.z_top40_slots - 0.996191) / 0.001738831   # -0.3%  z_top40_slots > 0.9962
        - 0.002857249 * max(0.0, 101.0497 - Q.mass) * max(0.0, 891.875 - Q.sum_pt_top10) / 2090.674   # -0.3%  mass < 101 and sum_pt_top10 < 891.9
        + 0.002756598 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +0.3%  n_dr_0p2_0p4 < 10
        - 0.002169864 * max(0.0, 64.48544 - Q.mass) * max(0.0, Q.pt_5 - 35.5) / 84.14794   # -0.2%  mass < 64.49 and pt_5 > 35.5
        - 0.002076899 * max(0.0, 0.076787 - Q.sum_z_dr) * max(0.0, Q.n_dr_0p4_up - 0.0) / 0.002691802   # -0.2%  sum_z_dr < 0.07679 and n_dr_0p4_up > 0
        + 0.002054932 * max(0.0, 0.03029714 - Q.e2) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583) / 0.001648545   # +0.2%  e2 < 0.0303 and sj3_pairmin_over_m > 0.09541
        - 0.002008445 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.2%  n_dr_0p1_0p2 < 8
        + 0.001803779 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0) / 30.16986   # +0.2%  n_dr_0p2_0p4 < 10 and n_particles > 34
        + 0.00161949 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.pair_mass_0_12 - 7.607175) / 4.784519   # +0.2%  n_dr_0p2_0p4 < 10 and pair_mass_0_12 > 7.607
        - 0.001609014 * max(0.0, 80.4 - Q.mass) * max(0.0, 0.8128995 - Q.pt1_over_pt0) / 2.534616   # -0.2%  mass < 80.4 and pt1_over_pt0 < 0.8129
        - 0.001572629 * max(0.0, 0.03875945 - Q.e2) * max(0.0, Q.soft1_abseta - 0.008917999) / 0.001603813   # -0.2%  e2 < 0.03876 and soft1_abseta > 0.008918
        - 0.001497784 * max(0.0, Q.psi_0p1 - 0.9770626) / 0.001598392   # -0.1%  psi_0p1 > 0.9771
        + 0.001418482 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.pt1_dr01 - 12.6865) / 16.11867   # +0.1%  mass < 101 and pt1_dr01 > 12.69
        - 0.001366324 * max(0.0, Q.D2 - 5.378975) / 0.5670303   # -0.1%  D2 > 5.379
        + 0.001200325 * max(0.0, 1.916992 - Q.soft10_pt) * max(0.0, Q.max_pair_mass - 18.09798) / 1.450926   # +0.1%  soft10_pt < 1.917 and max_pair_mass > 18.1
        - 0.001188604 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -0.1%  psi_0p3 > 0.9897
        + 0.001066387 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0402832 - Q.phi_0) / 0.166757   # +0.1%  n_dr_0p2_0p4 < 10 and phi_0 < 0.04028
        + 0.001004377 * max(0.0, Q.z_top40_slots - 0.996191) * max(0.0, 0.1711407 - Q.dr_12) / 0.0001648233   # +0.1%  z_top40_slots > 0.9962 and dr_12 < 0.1711
        - 0.001003942 * max(0.0, 0.005911134 - Q.zdr_0) / 0.001247991   # -0.1%  zdr_0 < 0.005911
        + 0.0009472653 * max(0.0, 1.916992 - Q.soft10_pt) * max(0.0, Q.sj3_mass1 - 5.112677) / 3.344471   # +0.1%  soft10_pt < 1.917 and sj3_mass1 > 5.113
        + 0.0009119057 * max(0.0, Q.z_top30_slots - 0.9909379) * max(0.0, Q.z_5 - 0.03252317) / 3.911495e-05   # +0.1%  z_top30_slots > 0.9909 and z_5 > 0.03252
        + 0.000840995 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.04641724 - Q.soft4_absphi) / 5.248227e-05   # +0.1%  psi_0p3 > 0.9897 and soft4_absphi < 0.04642
        + 0.0008391561 * max(0.0, Q.psi_0p2 - 0.9935324) * max(0.0, 0.1206146 - Q.abseta_14) / 0.0001099624   # +0.1%  psi_0p2 > 0.9935 and abseta_14 < 0.1206
        + 0.0007206626 * max(0.0, 1.916992 - Q.soft10_pt) / 0.3586715   # +0.1%  soft10_pt < 1.917
        + 0.000689878 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.zdr_7 - 0.004742028) / 0.0003820655   # +0.1%  n_dr_0p2_0p4 < 10 and zdr_7 > 0.004742
        + 0.0006700777 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.absphi_7 - 0.05749512) / 0.03655045   # +0.1%  n_dr_0p2_0p4 < 10 and absphi_7 > 0.0575
        - 0.0006342402 * max(0.0, Q.psi_0p2 - 0.9935324) * max(0.0, -0.01510849 - Q.eta_1) / 1.78772e-05   # -0.1%  psi_0p2 > 0.9935 and eta_1 < -0.01511
        - 0.0005283553 * max(0.0, 0.00818374 - Q.sum_zz_dr2) * max(0.0, Q.soft2_abseta - 0.1358643) / 0.0001061411   # -0.1%  sum_zz_dr2 < 0.008184 and soft2_abseta > 0.1359
        - 0.0004977996 * max(0.0, Q.psi_0p2 - 0.9313699) * max(0.0, 0.3897349 - Q.soft5_dr0) / 0.009147053   # -0.0%  psi_0p2 > 0.9314 and soft5_dr0 < 0.3897
        - 0.0004491643 * max(0.0, 0.08589404 - Q.sum_z_dr) * max(0.0, Q.soft3_dr - 0.161871) / 0.001840993   # -0.0%  sum_z_dr < 0.08589 and soft3_dr > 0.1619
        + 0.0003963929 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.pair_mass_0_10 - 7.661644) / 6.353801   # +0.0%  n_dr_0p2_0p4 < 10 and pair_mass_0_10 > 7.662
        + 0.0003905873 * max(0.0, Q.psi_0p2 - 0.9313699) * max(0.0, 10.81508 - Q.sj3_mass2) / 0.1687066   # +0.0%  psi_0p2 > 0.9314 and sj3_mass2 < 10.82
        + 0.0003772761 * max(0.0, 19.89956 - Q.sj2_mass1) * max(0.0, 0.002968979 - Q.soft10_abseta) / 0.001635055   # +0.0%  sj2_mass1 < 19.9 and soft10_abseta < 0.002969
        + 0.0003757843 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.ptdr0_2 - 11.87288) / 8.86429   # +0.0%  mass < 101 and ptdr0_2 > 11.87
        + 0.0003742838 * max(0.0, Q.psi_0p2 - 0.9935324) * max(0.0, Q.dr_11 - 0.05788126) / 3.477997e-05   # +0.0%  psi_0p2 > 0.9935 and dr_11 > 0.05788
        + 0.0003063669 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.pair_mass_0_13 - 8.669189) / 2.917386   # +0.0%  n_dr_0p2_0p4 < 10 and pair_mass_0_13 > 8.669
        + 0.0002780753 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.zdr_3 - 0.00781909) / 0.001157779   # +0.0%  n_dr_0p2_0p4 < 10 and zdr_3 > 0.007819
        + 0.0002482268 * max(0.0, Q.psi_0p2 - 0.9313699) * max(0.0, 0.2943504 - Q.soft10_dr0) / 0.00701521   # +0.0%  psi_0p2 > 0.9314 and soft10_dr0 < 0.2944
        + 0.0002256512 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.M2 - 0.06621477) / 0.03985969   # +0.0%  n_dr_0p2_0p4 < 10 and M2 > 0.06621
        + 0.000212786 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # +0.0%  psi_0p2 > 0.9314
        - 0.0001806796 * max(0.0, Q.z_top30_slots - 0.9909379) / 0.002533046   # -0.0%  z_top30_slots > 0.9909
        + 0.0001604124 * max(0.0, Q.z_top40_slots - 0.996191) * max(0.0, Q.dr_10 - 0.06334002) / 5.578992e-05   # +0.0%  z_top40_slots > 0.9962 and dr_10 > 0.06334
        + 0.0001462795 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_6 - 0.02949408) / 0.04399604   # +0.0%  n_dr_0p2_0p4 < 10 and z_6 > 0.02949
        + 0.000104196 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.dr_8 - 0.05067946) / 0.09994598   # +0.0%  n_dr_0p2_0p4 < 10 and dr_8 > 0.05068
        + 2.80891e-05 * max(0.0, 19.89956 - Q.sj2_mass1) * max(0.0, Q.phi_10 - -0.1351318) / 0.3308324   # +0.0%  sj2_mass1 < 19.9 and phi_10 > -0.1351
        + 1.351257e-05 * max(0.0, 19.89956 - Q.sj2_mass1) * max(0.0, 0.1147522 - Q.dr_9) / 0.1209699   # +0.0%  sj2_mass1 < 19.9 and dr_9 < 0.1148
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 3.458;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.458282 * (-0.07331993
        + 0.2460515 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +24.6%  mass < 82.85
        + 0.1287513 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.04417088   # +12.9%  mass < 86.4 and lam2 < 0.003688
        + 0.106951 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +10.7%  mass < 74.25
        - 0.06063963 * max(0.0, 86.4 - Q.mass) / 13.67632   # -6.1%  mass < 86.4
        - 0.05672062 * max(0.0, 62.55 - Q.mass) / 5.464058   # -5.7%  mass < 62.55
        - 0.04602287 * max(0.0, 0.06895248 - Q.mass_over_sum_pt) / 0.007729512   # -4.6%  mass_over_sum_pt < 0.06895
        - 0.04451409 * max(0.0, 74.78616 - Q.mass_top40) / 9.958349   # -4.5%  mass_top40 < 74.79
        - 0.03721341 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.5993597   # -3.7%  mass < 86.4 and z_dr_0p1_0p2 < 0.06473
        + 0.02946064 * max(0.0, 89.6788 - Q.mass_top40) / 17.47899   # +2.9%  mass_top40 < 89.68
        - 0.02141874 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.03580539   # -2.1%  mass < 86.4 and psi_0p3 < 0.9985
        - 0.01946355 * max(0.0, 1.339844 - Q.soft5_pt) / 0.3268008   # -1.9%  soft5_pt < 1.34
        + 0.01907881 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, Q.n_dr_0p05_0p1 - 1.0) / 124.4066   # +1.9%  mass_top40 < 89.68 and n_dr_0p05_0p1 > 1
        - 0.01866463 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # -1.9%  psi_0p3 > 0.9974
        + 0.01307813 * max(0.0, 0.002752094 - Q.lam1) / 0.0003912817   # +1.3%  lam1 < 0.002752
        + 0.01096291 * max(0.0, Q.sd_mass - 125.1) / 1.231011   # +1.1%  sd_mass > 125.1
        + 0.01016259 * max(0.0, Q.psi_0p1 - 0.9538343) / 0.006310723   # +1.0%  psi_0p1 > 0.9538
        - 0.008653699 * max(0.0, 0.1464158 - Q.soft4_dr) / 0.02966654   # -0.9%  soft4_dr < 0.1464
        - 0.008555577 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) * max(0.0, Q.z_10 - 0.01143749) / 2.042492e-05   # -0.9%  sum_z_dr2_top15 < 0.006143 and z_10 > 0.01144
        + 0.008387705 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) * max(0.0, 48.28125 - Q.pt_5) / 0.01386018   # +0.8%  sum_z_dr2_top15 < 0.006143 and pt_5 < 48.28
        + 0.007892117 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.5100475 - Q.tau21) / 0.4909635   # +0.8%  mass < 86.4 and tau21 < 0.51
        + 0.007692982 * max(0.0, Q.psi_0p1 - 0.9770626) / 0.001598392   # +0.8%  psi_0p1 > 0.9771
        + 0.007520415 * max(0.0, 0.0007431905 - Q.sum_z_dr2_top10) * max(0.0, 1069.671 - Q.sum_pt_top40) / 0.006568426   # +0.8%  sum_z_dr2_top10 < 0.0007432 and sum_pt_top40 < 1070
        + 0.007462054 * max(0.0, 0.0007431905 - Q.sum_z_dr2_top10) / 0.0001243965   # +0.7%  sum_z_dr2_top10 < 0.0007432
        - 0.00544146 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, 334.5 - Q.pt_0) / 1668.723   # -0.5%  mass_top40 < 89.68 and pt_0 < 334.5
        + 0.005271342 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min) / 54.09437   # +0.5%  sj3_pair_mass_max > 120.6 and sj3_pair_mass_min < 76.6
        - 0.005124313 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.985099 - Q.z_top50_slots) / 0.008644046   # -0.5%  mass < 86.4 and z_top50_slots < 0.9851
        - 0.005078816 * max(0.0, 6.856375 - Q.log_sum_pt) / 0.008053459   # -0.5%  log_sum_pt < 6.856
        - 0.004823522 * max(0.0, 6.856375 - Q.log_sum_pt) * max(0.0, 0.002396991 - Q.lam2) / 7.289393e-06   # -0.5%  log_sum_pt < 6.856 and lam2 < 0.002397
        + 0.004416164 * max(0.0, Q.mass - 160.8) / 1.724663   # +0.4%  mass > 160.8
        + 0.00430926 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926) / 0.5295314   # +0.4%  sj3_pair_mass_max > 120.6 and z_dr_0p1_0p2 > 0.2865
        + 0.003830603 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 32.50209 - Q.sj3_mass1) / 22.26646   # +0.4%  sj3_pair_mass_max > 120.6 and sj3_mass1 < 32.5
        - 0.003473652 * max(0.0, Q.sd_mass - 125.1) * max(0.0, 0.4199841 - Q.sd_zg) / 0.1704664   # -0.3%  sd_mass > 125.1 and sd_zg < 0.42
        - 0.003298609 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) * max(0.0, Q.sum_pt_top40 - 1225.842) / 0.02866211   # -0.3%  sum_z_dr2_top15 < 0.006143 and sum_pt_top40 > 1226
        - 0.003242476 * max(0.0, Q.sd_mass - 125.1) * max(0.0, Q.lam2 - 0.0001679609) / 0.005601117   # -0.3%  sd_mass > 125.1 and lam2 > 0.000168
        + 0.002835169 * max(0.0, 86.4 - Q.mass) * max(0.0, Q.zdr_0 - 0.007860787) / 0.006171938   # +0.3%  mass < 86.4 and zdr_0 > 0.007861
        - 0.002798528 * max(0.0, Q.sj3_pair_mass_max - 120.6) / 2.130243   # -0.3%  sj3_pair_mass_max > 120.6
        - 0.002683384 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, 0.2502892 - Q.sj3_z3) / 0.0007552095   # -0.3%  z_dr_0p1_0p2 > 0.6882 and sj3_z3 < 0.2503
        + 0.002564449 * max(0.0, Q.z_dr_0_0p05 - 0.9351131) / 0.004598373   # +0.3%  z_dr_0_0p05 > 0.9351
        + 0.002543762 * max(0.0, 24.20196 - Q.mass_top15) / 1.791326   # +0.3%  mass_top15 < 24.2
        + 0.001948646 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) / 0.006411021   # +0.2%  z_dr_0p1_0p2 > 0.6882
        + 0.001614179 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, Q.n_dr_0p4_up - 0.0) / 0.001442994   # +0.2%  z_dr_0p1_0p2 > 0.6882 and n_dr_0p4_up > 0
        + 0.001600658 * max(0.0, Q.sd_mass - 125.1) * max(0.0, Q.sum_z_dr2_top3 - 0.01002369) / 0.01693538   # +0.2%  sd_mass > 125.1 and sum_z_dr2_top3 > 0.01002
        + 0.001519651 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 0.00104408 - Q.soft9_z) / 7.375548e-05   # +0.2%  sj3_pair_mass_max > 120.6 and soft9_z < 0.001044
        + 0.001498723 * max(0.0, 0.05077291 - Q.mass_over_sum_pt) / 0.003255623   # +0.1%  mass_over_sum_pt < 0.05077
        - 0.001210927 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) / 0.002080651   # -0.1%  sum_z_dr2_top15 < 0.006143
        + 0.0009380932 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.1873652 - Q.sj3_z2) / 1.419703e-05   # +0.1%  psi_0p3 > 0.9974 and sj3_z2 < 0.1874
        + 0.0008200889 * max(0.0, Q.z_dr_0_0p05 - 0.9351131) * max(0.0, 15.0 - Q.n_real_top15) / 0.0004988097   # +0.1%  z_dr_0_0p05 > 0.9351 and n_real_top15 < 15
        - 0.0006640799 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.orientation_deg - 35.6254) / 17.12462   # -0.1%  sj3_pair_mass_max > 120.6 and orientation_deg > 35.63
        + 0.0003333724 * max(0.0, 0.0007431905 - Q.sum_z_dr2_top10) * max(0.0, 36.0 - Q.n_real_top40) / 0.0006803313   # +0.0%  sum_z_dr2_top10 < 0.0007432 and n_real_top40 < 36
        - 0.0003139849 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, -62.72059 - Q.orientation_deg) / 4.424567   # -0.0%  sj3_pair_mass_max > 120.6 and orientation_deg < -62.72
        + 0.0002715951 * max(0.0, 74.25181 - Q.mass) * max(0.0, Q.dr_max_012 - 0.1206357) / 0.0008878091   # +0.0%  mass < 74.25 and dr_max_012 > 0.1206
        - 0.000203919 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, Q.soft4_absphi - 0.170166) / 0.0001623923   # -0.0%  z_dr_0p1_0p2 > 0.6882 and soft4_absphi > 0.1702
        - 7.584005e-06 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -0.0%  mass < 53.87
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 10.34;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.33829 * (0.210964
        - 0.1078381 * max(0.0, 0.02580396 - Q.mass_over_sum_pt_sq) / 0.01697662   # -10.8%  mass_over_sum_pt_sq < 0.0258
        + 0.06882504 * max(0.0, 0.02550569 - Q.sum_z_dr2_top50) / 0.01683664   # +6.9%  sum_z_dr2_top50 < 0.02551
        + 0.05388531 * Q.n_particles / 45.81523   # +5.4%  n_particles
        + 0.05372241 * max(0.0, Q.mass - 74.25181) / 24.07648   # +5.4%  mass > 74.25
        - 0.05066007 * max(0.0, 1115.723 - Q.sum_pt) / 92.38491   # -5.1%  sum_pt < 1116
        + 0.04393388 * max(0.0, 0.02550569 - Q.sum_z_dr2_top50) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.0005189214   # +4.4%  sum_z_dr2_top50 < 0.02551 and psi_0p3 > 0.9638
        - 0.03636819 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -3.6%  sum_pt < 1085
        + 0.03582143 * max(0.0, Q.mass_top50 - 27.46086) / 60.61506   # +3.6%  mass_top50 > 27.46
        - 0.03423527 * max(0.0, Q.mass - 160.8) / 1.724663   # -3.4%  mass > 160.8
        - 0.03221777 * max(0.0, Q.mass - 143.7876) / 3.946979   # -3.2%  mass > 143.8
        + 0.03125837 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +3.1%  sum_pt_top40 < 1053
        - 0.02813003 * max(0.0, Q.mass - 101.0497) / 12.31084   # -2.8%  mass > 101
        - 0.02596564 * max(0.0, 6.910131 - Q.log_sum_pt) / 0.017275   # -2.6%  log_sum_pt < 6.91
        - 0.02405784 * max(0.0, Q.mass - 136.785) / 5.043396   # -2.4%  mass > 136.8
        + 0.01885378 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +1.9%  mass_top50 > 136.8
        - 0.01809435 * max(0.0, 1034.834 - Q.sum_pt) / 30.78807   # -1.8%  sum_pt < 1035
        + 0.01686414 * max(0.0, 0.08286256 - Q.tau1) / 0.0188894   # +1.7%  tau1 < 0.08286
        + 0.01466361 * max(0.0, Q.psi_0p3 - 0.9638082) / 0.02792824   # +1.5%  psi_0p3 > 0.9638
        - 0.01433593 * max(0.0, 966.0633 - Q.sum_pt_top30) / 29.99923   # -1.4%  sum_pt_top30 < 966.1
        + 0.01392984 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +1.4%  mass_over_sum_pt > 0.1709
        + 0.01390729 * max(0.0, Q.psi_0p3 - 0.9638082) * max(0.0, 0.0223982 - Q.C3) / 0.0004215996   # +1.4%  psi_0p3 > 0.9638 and C3 < 0.0224
        - 0.01363989 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.8310045 - Q.tau21_b2) / 33.42141   # -1.4%  sum_pt < 1085 and tau21_b2 < 0.831
        - 0.01249091 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -1.2%  mass_over_sum_pt_sq > 0.0292
        + 0.01200555 * max(0.0, Q.mass - 162.8363) / 1.509832   # +1.2%  mass > 162.8
        - 0.01154003 * max(0.0, Q.mass - 74.25181) * max(0.0, 1028.184 - Q.sum_pt) / 733.8204   # -1.2%  mass > 74.25 and sum_pt < 1028
        - 0.01046637 * max(0.0, Q.mass_top10 - 23.38894) / 27.40926   # -1.0%  mass_top10 > 23.39
        + 0.01018697 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # +1.0%  log_sum_pt < 6.811
        + 0.00942312 * max(0.0, Q.mass - 172.8) / 0.7291439   # +0.9%  mass > 172.8
        + 0.008926192 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # +0.9%  sum_pt_top40 < 1007
        - 0.008832776 * max(0.0, 1034.834 - Q.sum_pt) * max(0.0, 2.975532 - Q.D2) / 24.6298   # -0.9%  sum_pt < 1035 and D2 < 2.976
        - 0.008563464 * max(0.0, Q.mass - 136.785) * max(0.0, Q.D2 - 0.8942376) / 6.48176   # -0.9%  mass > 136.8 and D2 > 0.8942
        + 0.007277082 * max(0.0, Q.n_pt_above_1 - 58.0) / 0.9616151   # +0.7%  n_pt_above_1 > 58
        - 0.006814857 * max(0.0, 16.0 - Q.n_for_90pct) / 1.776607   # -0.7%  n_for_90pct < 16
        + 0.00654047 * max(0.0, 160.8 - Q.mass_top40) / 76.75884   # +0.7%  mass_top40 < 160.8
        - 0.006381258 * max(0.0, 1007.788 - Q.sum_pt) / 17.7122   # -0.6%  sum_pt < 1008
        + 0.006204464 * max(0.0, 0.007820315 - Q.sum_z_dr2_top50) / 0.002264874   # +0.6%  sum_z_dr2_top50 < 0.00782
        + 0.006182686 * max(0.0, Q.mass - 136.785) * max(0.0, 1028.184 - Q.sum_pt) / 132.4986   # +0.6%  mass > 136.8 and sum_pt < 1028
        + 0.006111754 * max(0.0, Q.mass - 160.8) * max(0.0, Q.D2 - 0.8942376) / 2.121004   # +0.6%  mass > 160.8 and D2 > 0.8942
        + 0.005810195 * max(0.0, Q.psi_0p3 - 0.9638082) * max(0.0, 0.4357228 - Q.max_dr) / 0.002832914   # +0.6%  psi_0p3 > 0.9638 and max_dr < 0.4357
        - 0.005669322 * max(0.0, Q.mass_top50 - 136.785) * max(0.0, Q.soft6_z - 0.0008188601) / 0.003727182   # -0.6%  mass_top50 > 136.8 and soft6_z > 0.0008189
        + 0.005488986 * max(0.0, Q.sum_pt_top30 - 1111.245) / 12.62573   # +0.5%  sum_pt_top30 > 1111
        - 0.005439586 * max(0.0, Q.mass_top20 - 91.19) / 5.893841   # -0.5%  mass_top20 > 91.19
        - 0.005126822 * max(0.0, Q.sum_z_dr2_top50 - 0.01951641) / 0.001208668   # -0.5%  sum_z_dr2_top50 > 0.01952
        - 0.004591478 * max(0.0, 76.9886 - Q.mass_top10) / 32.58955   # -0.5%  mass_top10 < 76.99
        + 0.004572112 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft6_z - 0.0005748372) / 0.002073713   # +0.5%  mass > 162.8 and soft6_z > 0.0005748
        + 0.004311276 * max(0.0, 997.0189 - Q.sum_pt_top50) / 17.90852   # +0.4%  sum_pt_top50 < 997
        + 0.004254842 * max(0.0, Q.psi_0p3 - 0.9638082) * max(0.0, Q.absphi_12 - 0.01163483) / 0.001186382   # +0.4%  psi_0p3 > 0.9638 and absphi_12 > 0.01163
        - 0.003976998 * max(0.0, Q.mass_top50 - 92.16545) * max(0.0, 0.001923089 - Q.soft7_z) / 0.006887627   # -0.4%  mass_top50 > 92.17 and soft7_z < 0.001923
        + 0.003782434 * max(0.0, 16.0 - Q.n_for_90pct) * max(0.0, 9.676985 - Q.D2) / 8.10578   # +0.4%  n_for_90pct < 16 and D2 < 9.677
        + 0.003551028 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # +0.4%  n_dr_0p1_0p2 < 17
        + 0.003407345 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.2179035 - Q.sj2_zsoft) / 2.389974   # +0.3%  sum_pt_top40 < 1053 and sj2_zsoft < 0.2179
        + 0.003370351 * max(0.0, Q.mass_top5 - 33.71058) / 7.394508   # +0.3%  mass_top5 > 33.71
        + 0.003270575 * max(0.0, 16.0 - Q.n_for_90pct) * max(0.0, 3.814159 - Q.D2) / 1.817419   # +0.3%  n_for_90pct < 16 and D2 < 3.814
        + 0.002908618 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +0.3%  pt_6 < 41.22
        + 0.002513137 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.3%  psi_0p3 > 0.9974
        - 0.002422631 * max(0.0, Q.mass - 172.8) * max(0.0, Q.D2 - 0.8942376) / 0.9099067   # -0.2%  mass > 172.8 and D2 > 0.8942
        - 0.002276604 * max(0.0, Q.mass - 172.8) * max(0.0, Q.soft6_z - 0.0004464147) / 0.001308464   # -0.2%  mass > 172.8 and soft6_z > 0.0004464
        - 0.002149416 * max(0.0, Q.mass_top50 - 92.16545) / 13.44564   # -0.2%  mass_top50 > 92.17
        - 0.001950757 * max(0.0, Q.sum_pt_top20 - 956.5062) / 37.46564   # -0.2%  sum_pt_top20 > 956.5
        - 0.001836803 * max(0.0, Q.C2 - 0.0977156) / 0.006150774   # -0.2%  C2 > 0.09772
        - 0.001780399 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.1023813 - Q.dr_13) / 2.159889   # -0.2%  sum_pt < 1085 and dr_13 < 0.1024
        - 0.001767043 * max(0.0, Q.psi_0p3 - 0.9638082) * max(0.0, Q.sj3_dr13 - 0.2465017) / 0.0003007569   # -0.2%  psi_0p3 > 0.9638 and sj3_dr13 > 0.2465
        + 0.00159803 * max(0.0, Q.psi_0p1 - 0.9371031) / 0.01088804   # +0.2%  psi_0p1 > 0.9371
        + 0.001567588 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.005041702 - Q.zdr_11) / 1.572361e-05   # +0.2%  log_sum_pt < 6.811 and zdr_11 < 0.005042
        - 0.001522764 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.1289397 - Q.dr_4) / 4.4378   # -0.2%  sum_pt < 1085 and dr_4 < 0.1289
        + 0.00147431 * max(0.0, Q.mass_top30 - 138.3818) / 1.758216   # +0.1%  mass_top30 > 138.4
        - 0.0014627 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.1617791 - Q.dr0_3) / 5.794926   # -0.1%  sum_pt_top40 < 1053 and dr0_3 < 0.1618
        + 0.001456727 * max(0.0, 1034.834 - Q.sum_pt) * max(0.0, 16.64062 - Q.pt_9) / 29.01849   # +0.1%  sum_pt < 1035 and pt_9 < 16.64
        - 0.001275925 * max(0.0, Q.mass - 172.8) * max(0.0, 56.53125 - Q.pt_6) / 9.768822   # -0.1%  mass > 172.8 and pt_6 < 56.53
        + 0.001262959 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 858.8262 - Q.sum_pt_top40) / 1219.241   # +0.1%  sum_pt < 1085 and sum_pt_top40 < 858.8
        - 0.001234496 * max(0.0, Q.mass_top50 - 168.9698) / 0.6651775   # -0.1%  mass_top50 > 169
        + 0.001228058 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.1312677 - Q.dr_13) / 0.0002562188   # +0.1%  log_sum_pt < 6.811 and dr_13 < 0.1313
        + 0.001220922 * max(0.0, 0.08871546 - Q.z_dr_0p1_0p2) / 0.03070029   # +0.1%  z_dr_0p1_0p2 < 0.08872
        + 0.0008813892 * max(0.0, Q.mass_top10 - 56.92192) / 8.071231   # +0.1%  mass_top10 > 56.92
        - 0.0008704299 * max(0.0, 18.32812 - Q.pt_11) / 2.095384   # -0.1%  pt_11 < 18.33
        + 0.0007938839 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.phi_11 - -0.1395264) / 0.0007721772   # +0.1%  log_sum_pt > 7.139 and phi_11 > -0.1395
        - 0.0007267092 * max(0.0, Q.C2 - 0.0977156) * max(0.0, 53.9394 - Q.orientation_deg) / 0.3542801   # -0.1%  C2 > 0.09772 and orientation_deg < 53.94
        - 0.0006955794 * max(0.0, 1007.788 - Q.sum_pt) * max(0.0, Q.sum_pt_top3 - 512.4375) / 295.3022   # -0.1%  sum_pt < 1008 and sum_pt_top3 > 512.4
        - 0.0006470489 * max(0.0, Q.tau1 - 0.1751567) / 0.002322569   # -0.1%  tau1 > 0.1752
        - 0.0005830163 * max(0.0, Q.mass_top30 - 138.3818) * max(0.0, 0.1502914 - Q.dr_8) / 0.04041135   # -0.1%  mass_top30 > 138.4 and dr_8 < 0.1503
        + 0.0005717903 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 2.680253 - Q.D2) / 0.001923611   # +0.1%  log_sum_pt < 6.811 and D2 < 2.68
        + 0.0005025751 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.sd_zg - 0.4462823) / 1.289737e-05   # +0.1%  log_sum_pt > 7.139 and sd_zg > 0.4463
        - 0.0004961604 * max(0.0, 2.752054 - Q.pt_entropy) / 0.1846134   # -0.0%  pt_entropy < 2.752
        + 0.0004158875 * max(0.0, 18.32812 - Q.pt_11) * max(0.0, 0.2327893 - Q.dr0_14) / 0.2441059   # +0.0%  pt_11 < 18.33 and dr0_14 < 0.2328
        + 0.0002924358 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # +0.0%  log_sum_pt > 7.139
        - 0.0002858911 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.04047238 - Q.dr_9) / 6.697833e-05   # -0.0%  log_sum_pt > 7.139 and dr_9 < 0.04047
        + 0.0002519088 * max(0.0, Q.mass - 172.8) * max(0.0, 0.04568661 - Q.z_6) / 0.006788705   # +0.0%  mass > 172.8 and z_6 < 0.04569
        + 0.000238221 * max(0.0, Q.psi_0p3 - 0.9638082) * max(0.0, Q.abseta_0 - 0.02487183) / 0.0004113988   # +0.0%  psi_0p3 > 0.9638 and abseta_0 > 0.02487
        + 0.0002328002 * max(0.0, Q.sum_pt_top30 - 1111.245) * max(0.0, 0.01190756 - Q.mratio_min_012) / 0.001736523   # +0.0%  sum_pt_top30 > 1111 and mratio_min_012 < 0.01191
        + 0.000218579 * max(0.0, Q.mass - 172.8) * max(0.0, Q.soft5_z - 0.0004140594) / 0.001251279   # +0.0%  mass > 172.8 and soft5_z > 0.0004141
        + 0.0002087586 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.05398073 - Q.dr_12) / 9.985673e-05   # +0.0%  log_sum_pt > 7.139 and dr_12 < 0.05398
        + 0.0001871446 * max(0.0, Q.mass - 136.785) * max(0.0, 0.02291864 - Q.z_9) / 0.002659285   # +0.0%  mass > 136.8 and z_9 < 0.02292
        - 9.211767e-05 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # -0.0%  lam1 > 0.01174
        - 5.983264e-05 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.00317537 - Q.C3) / 7.466262e-06   # -0.0%  log_sum_pt > 7.139 and C3 < 0.003175
        - 5.847171e-05 * max(0.0, Q.n_pt_above_1 - 34.0) / 10.44526   # -0.0%  n_pt_above_1 > 34
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 18.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.63768 * (-0.001917026
        + 0.07791483 * max(0.0, 0.140939 - Q.mass_over_sum_pt) / 0.05796036   # +7.8%  mass_over_sum_pt < 0.1409
        - 0.07603514 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) / 0.0178873   # -7.6%  mass_over_sum_pt < 0.09047
        - 0.06236251 * max(0.0, 89.74183 - Q.mass) / 15.55633   # -6.2%  mass < 89.74
        + 0.06110121 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +6.1%  mass_over_sum_pt < 0.1182
        - 0.05348393 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -5.3%  lam1 < 0.008242
        + 0.05237318 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +5.2%  mass_over_sum_pt < 0.09795
        - 0.04791906 * max(0.0, 91.19 - Q.mass) / 16.46423   # -4.8%  mass < 91.19
        + 0.0449162 * max(0.0, 125.1 - Q.mass) / 42.45775   # +4.5%  mass < 125.1
        + 0.04275349 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +4.3%  lam1 < 0.007671
        - 0.03784543 * max(0.0, 0.01897915 - Q.sum_z_dr2_top40) / 0.01130444   # -3.8%  sum_z_dr2_top40 < 0.01898
        - 0.03212416 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p2 - 0.9087063) / 0.00142529   # -3.2%  mass_over_sum_pt < 0.09047 and psi_0p2 > 0.9087
        + 0.0307275 * max(0.0, 80.4 - Q.mass) / 10.6806   # +3.1%  mass < 80.4
        + 0.02542589 * max(0.0, 85.8667 - Q.mass_top50) / 13.95835   # +2.5%  mass_top50 < 85.87
        - 0.02415021 * max(0.0, 0.01215787 - Q.sum_z_dr2_top30) / 0.005935412   # -2.4%  sum_z_dr2_top30 < 0.01216
        - 0.01981535 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -2.0%  tau1 < 0.1073
        + 0.01709856 * max(0.0, 0.1778185 - Q.sd_rg) / 0.06576479   # +1.7%  sd_rg < 0.1778
        - 0.01630455 * max(0.0, 117.0487 - Q.mass_top50) / 36.94196   # -1.6%  mass_top50 < 117
        - 0.01508832 * max(0.0, 0.01083435 - Q.sum_z_dr2_top20) / 0.00529825   # -1.5%  sum_z_dr2_top20 < 0.01083
        + 0.0132109 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # +1.3%  e2 < 0.04359
        - 0.01083503 * max(0.0, 97.93004 - Q.mass_top50) / 22.03533   # -1.1%  mass_top50 < 97.93
        - 0.01060784 * max(0.0, 0.2042612 - Q.sd_rg) / 0.08448097   # -1.1%  sd_rg < 0.2043
        + 0.01057264 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +1.1%  psi_0p3 > 0.9924
        - 0.01051638 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # -1.1%  mass_top50 < 71.8
        - 0.01006405 * max(0.0, 0.076787 - Q.sum_z_dr) / 0.02105267   # -1.0%  sum_z_dr < 0.07679
        + 0.009696346 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) / 0.003638607   # +1.0%  sum_z_dr2_top5 < 0.007164
        - 0.00882972 * max(0.0, 0.0001086251 - Q.e3) / 5.924919e-05   # -0.9%  e3 < 0.0001086
        + 0.008432611 * max(0.0, 0.08873143 - Q.mass_over_sum_pt) / 0.01671255   # +0.8%  mass_over_sum_pt < 0.08873
        + 0.007791922 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +0.8%  mass < 74.25
        - 0.007555297 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # -0.8%  tau2 < 0.0795
        + 0.007530581 * max(0.0, 0.076787 - Q.sum_z_dr) * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.0007421723   # +0.8%  sum_z_dr < 0.07679 and z_dr_0p2_0p4 < 0.0518
        - 0.007289913 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1881908 - Q.sd_rg) / 0.0002726472   # -0.7%  psi_0p3 > 0.9924 and sd_rg < 0.1882
        + 0.006985608 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +0.7%  e2 < 0.0303
        + 0.006958741 * max(0.0, 0.006374178 - Q.sum_z_dr2_top20) / 0.001987751   # +0.7%  sum_z_dr2_top20 < 0.006374
        + 0.006635726 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +0.7%  tau1 < 0.06311
        + 0.006536315 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +0.7%  mass < 78.26
        + 0.00640629 * max(0.0, 6.941997 - Q.log_sum_pt) / 0.03180972   # +0.6%  log_sum_pt < 6.942
        - 0.006061007 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -0.6%  mass < 82.85
        + 0.005936059 * max(0.0, 0.07852883 - Q.mass_over_sum_pt) / 0.01107516   # +0.6%  mass_over_sum_pt < 0.07853
        + 0.00574007 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +0.6%  tau21_b2 < 0.3425
        - 0.005485414 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # -0.5%  sd_rg < 0.3017
        - 0.005426924 * max(0.0, 77.37641 - Q.mass_top50) / 9.980855   # -0.5%  mass_top50 < 77.38
        + 0.005331312 * max(0.0, 1.650391 - Q.soft7_pt) / 0.3647391   # +0.5%  soft7_pt < 1.65
        - 0.004974863 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.007709916 - Q.sum_z_dr2_top40) / 0.0001062984   # -0.5%  tau21_b2 < 0.3425 and sum_z_dr2_top40 < 0.00771
        - 0.004411621 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # -0.4%  log_sum_pt < 6.989
        + 0.004340137 * max(0.0, 89.6788 - Q.mass_top40) / 17.47899   # +0.4%  mass_top40 < 89.68
        + 0.004158594 * max(0.0, 82.66587 - Q.mass_top30) / 15.96638   # +0.4%  mass_top30 < 82.67
        + 0.004154553 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # +0.4%  sum_pt_top40 < 1025
        - 0.003352127 * max(0.0, 0.02210818 - Q.e2) / 0.00342102   # -0.3%  e2 < 0.02211
        + 0.002803623 * max(0.0, 0.4226723 - Q.N2) / 0.1194948   # +0.3%  N2 < 0.4227
        - 0.002752259 * max(0.0, 0.001595561 - Q.soft7_z) / 0.0003533115   # -0.3%  soft7_z < 0.001596
        - 0.002563206 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2404747) / 0.01288721   # -0.3%  N2 < 0.4227 and max_dr > 0.2405
        + 0.002330892 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 3.445312 - Q.soft9_pt) / 12.48859   # +0.2%  n_dr_0p2_0p4 < 18 and soft9_pt < 3.445
        + 0.002022535 * max(0.0, 69.65633 - Q.sd_mass) / 24.56034   # +0.2%  sd_mass < 69.66
        - 0.001755017 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) * max(0.0, Q.n_pt_above_10 - 13.0) / 0.02038471   # -0.2%  sum_z_dr2_top5 < 0.007164 and n_pt_above_10 > 13
        - 0.001662752 * max(0.0, 0.9734886 - Q.z_top30_slots) / 0.03219848   # -0.2%  z_top30_slots < 0.9735
        - 0.001652125 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1231747 - Q.z_2nd) / 6.106735e-05   # -0.2%  psi_0p3 > 0.9924 and z_2nd < 0.1232
        + 0.001609997 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +0.2%  lam1 < 0.007259
        - 0.001580619 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # -0.2%  n_dr_0p2_0p4 < 18
        + 0.001514 * max(0.0, 0.0795038 - Q.tau2) * max(0.0, 0.1178619 - Q.absphi_1) / 0.004334208   # +0.2%  tau2 < 0.0795 and absphi_1 < 0.1179
        - 0.001436629 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 32.0 - Q.n_real_top40) / 0.1595466   # -0.1%  tau21_b2 < 0.3425 and n_real_top40 < 32
        - 0.001425435 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -18.47737) / 3.570094   # -0.1%  tau21_b2 < 0.3425 and orientation_deg > -18.48
        + 0.001402638 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.06195068 - Q.absphi_5) / 0.002511795   # +0.1%  tau21_b2 < 0.3425 and absphi_5 < 0.06195
        + 0.00137036 * max(0.0, 0.001163277 - Q.lam2) / 0.0004869716   # +0.1%  lam2 < 0.001163
        + 0.001355768 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # +0.1%  sj3_mass1 < 32.5
        + 0.001293233 * max(0.0, 0.1722488 - Q.tau21_b2) / 0.02681502   # +0.1%  tau21_b2 < 0.1722
        - 0.001203366 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4) / 0.08975043   # -0.1%  z_dr_0p2_0p4 < 0.1292
        + 0.001101456 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.sj3_dr_min - 0.05940798) / 0.005051091   # +0.1%  N2 < 0.4227 and sj3_dr_min > 0.05941
        + 0.001006162 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.02048524 - Q.lam1) / 0.001304679   # +0.1%  tau21_b2 < 0.3425 and lam1 < 0.02049
        + 0.000997768 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03787151 - Q.M3) / 0.1118325   # +0.1%  n_dr_0p2_0p4 < 18 and M3 < 0.03787
        - 0.0009557443 * max(0.0, 0.01215787 - Q.sum_z_dr2_top30) * max(0.0, Q.pair_mass_0_2 - 22.41128) / 0.003197931   # -0.1%  sum_z_dr2_top30 < 0.01216 and pair_mass_0_2 > 22.41
        - 0.0009182444 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.n_dr_0p05_0p1 - 14.0) / 15.57557   # -0.1%  mass < 91.19 and n_dr_0p05_0p1 > 14
        - 0.0008927814 * max(0.0, 976.277 - Q.sum_pt_top50) / 12.87298   # -0.1%  sum_pt_top50 < 976.3
        + 0.0008113236 * max(0.0, Q.psi_0p2 - 0.9804031) * max(0.0, Q.pt2_over_pt0 - 0.1852611) / 0.002296628   # +0.1%  psi_0p2 > 0.9804 and pt2_over_pt0 > 0.1853
        - 0.0007868005 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, Q.sj3_mass2 - 5.762132) / 0.2507301   # -0.1%  tau21_b2 < 0.3425 and sj3_mass2 > 5.762
        - 0.0007721082 * max(0.0, 906.6023 - Q.sum_pt_top40) / 6.984569   # -0.1%  sum_pt_top40 < 906.6
        + 0.0007362306 * max(0.0, 37.45803 - Q.sj2_mass1) / 13.07509   # +0.1%  sj2_mass1 < 37.46
        + 0.0006973589 * max(0.0, Q.n_pt_above_1 - 58.0) / 0.9616151   # +0.1%  n_pt_above_1 > 58
        - 0.0006522447 * max(0.0, 125.1 - Q.mass) * max(0.0, Q.mass_top2 - 28.78966) / 17.57849   # -0.1%  mass < 125.1 and mass_top2 > 28.79
        + 0.0005850066 * max(0.0, 0.4226723 - Q.N2) * max(0.0, 0.2053949 - Q.dr_12) / 0.0131481   # +0.1%  N2 < 0.4227 and dr_12 < 0.2054
        + 0.0004791726 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.1943707   # +0.0%  tau21_b2 < 0.3425 and n_pt_above_50 < 7
        + 0.000383014 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) * max(0.0, Q.sj2_mass2 - 14.45919) / 0.003408014   # +0.0%  sum_z_dr2_top5 < 0.007164 and sj2_mass2 > 14.46
        - 0.0003350352 * max(0.0, 3.080078 - Q.soft9_pt) / 1.143104   # -0.0%  soft9_pt < 3.08
        + 0.0003267348 * max(0.0, 0.01256572 - Q.e2) / 0.0008030352   # +0.0%  e2 < 0.01257
        + 0.0002856416 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.3682013 - Q.sj3_dr13) / 0.02227903   # +0.0%  tau21_b2 < 0.3425 and sj3_dr13 < 0.3682
        + 0.0002672197 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.006118006 - Q.sum_z_dr2_top50) / 2.65509e-05   # +0.0%  tau21_b2 < 0.3425 and sum_z_dr2_top50 < 0.006118
        - 0.0002569909 * max(0.0, Q.psi_0p1 - 0.9538343) / 0.006310723   # -0.0%  psi_0p1 > 0.9538
        + 0.0002325621 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) * max(0.0, Q.pair_mass_0_4 - 8.587823) / 0.003768116   # +0.0%  sum_z_dr2_top5 < 0.007164 and pair_mass_0_4 > 8.588
        - 0.0002226176 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) * max(0.0, Q.zdr_5 - 0.008949744) / 1.839629e-07   # -0.0%  sum_z_dr2_top5 < 0.007164 and zdr_5 > 0.00895
        + 0.0002139941 * max(0.0, Q.pt_10 - 28.15625) / 1.573077   # +0.0%  pt_10 > 28.16
        + 0.0002084185 * max(0.0, 37.45803 - Q.sj2_mass1) * max(0.0, -0.05432129 - Q.phi_6) / 0.07853312   # +0.0%  sj2_mass1 < 37.46 and phi_6 < -0.05432
        + 0.0001730701 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, Q.n_for_50pct - 3.0) / 0.1980908   # +0.0%  tau21_b2 < 0.3425 and n_for_50pct > 3
        - 0.0001399757 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4) * max(0.0, Q.eta_11 - -0.04888916) / 0.005127364   # -0.0%  z_dr_0p2_0p4 < 0.1292 and eta_11 > -0.04889
        + 0.0001258441 * max(0.0, 0.001592178 - Q.sum_z_dr2_top3) / 0.000513961   # +0.0%  sum_z_dr2_top3 < 0.001592
        + 0.0001118996 * max(0.0, Q.psi_0p2 - 0.9804031) * max(0.0, -0.0966217 - Q.phi_2) / 5.246579e-06   # +0.0%  psi_0p2 > 0.9804 and phi_2 < -0.09662
        + 0.0001093182 * max(0.0, 0.002582316 - Q.sum_z_dr2_top30) / 0.0003518773   # +0.0%  sum_z_dr2_top30 < 0.002582
        - 9.860487e-05 * max(0.0, 117.0487 - Q.mass_top50) * max(0.0, Q.zdr_6 - 0.006336433) / 0.001577598   # -0.0%  mass_top50 < 117 and zdr_6 > 0.006336
        - 5.65751e-05 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.abseta_14 - 0.2005615) / 0.0002634049   # -0.0%  N2 < 0.4227 and abseta_14 > 0.2006
        + 3.167526e-05 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, Q.sd_rg - 0.2280025) / 1.846945e-05   # +0.0%  psi_0p3 > 0.9924 and sd_rg > 0.228
        + 2.581622e-05 * max(0.0, Q.psi_0p2 - 0.9804031) / 0.007668897   # +0.0%  psi_0p2 > 0.9804
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 8.878;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.877668 * (0.04643252
        - 0.0755115 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # -7.6%  sd_rg < 0.3017
        + 0.06035724 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +6.0%  log_sum_pt < 7.017
        + 0.056993 * max(0.0, 0.007678544 - Q.sum_z_dr2_top10) / 0.00347358   # +5.7%  sum_z_dr2_top10 < 0.007679
        + 0.04837781 * max(0.0, 0.08589404 - Q.sum_z_dr) / 0.02745918   # +4.8%  sum_z_dr < 0.08589
        - 0.04341288 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # -4.3%  lam1 < 0.01174
        - 0.04164023 * max(0.0, 0.008956554 - Q.sum_z_dr2_top10) / 0.004473389   # -4.2%  sum_z_dr2_top10 < 0.008957
        + 0.04118822 * max(0.0, 0.2042612 - Q.sd_rg) / 0.08448097   # +4.1%  sd_rg < 0.2043
        + 0.03829952 * max(0.0, 79.18312 - Q.sd_mass) / 30.46368   # +3.8%  sd_mass < 79.18
        + 0.03671466 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # +3.7%  sum_z_dr2_top5 < 0.00833
        + 0.02772561 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +2.8%  z_dr_0p1_0p2 < 0.1203
        - 0.02456131 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.5%  tau1 < 0.07057
        - 0.02445576 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # -2.4%  e2 > 0.0303
        - 0.02318067 * max(0.0, 0.2941033 - Q.LHA) / 0.05718399   # -2.3%  LHA < 0.2941
        - 0.01978074 * max(0.0, 45.595 - Q.sd_mass) / 13.46843   # -2.0%  sd_mass < 45.59
        + 0.01772922 * max(0.0, Q.mass_top30 - 80.24626) / 13.49216   # +1.8%  mass_top30 > 80.25
        - 0.01736464 * max(0.0, 1018.698 - Q.sum_pt_top40) / 34.89083   # -1.7%  sum_pt_top40 < 1019
        - 0.01647675 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -1.6%  psi_0p1 > 0.8976
        - 0.01596409 * max(0.0, 0.006399858 - Q.sum_zz_dr2) / 0.001405963   # -1.6%  sum_zz_dr2 < 0.0064
        - 0.01510521 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # -1.5%  mass_top50 < 71.8
        - 0.01499771 * max(0.0, Q.mass_top50 - 97.93004) / 11.91764   # -1.5%  mass_top50 > 97.93
        + 0.01439637 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +1.4%  lam1 < 0.004673
        - 0.01353891 * max(0.0, 0.4226723 - Q.N2) / 0.1194948   # -1.4%  N2 < 0.4227
        - 0.01283729 * max(0.0, 1017.778 - Q.sum_pt_top20) / 107.0228   # -1.3%  sum_pt_top20 < 1018
        + 0.01159481 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) / 3.82104   # +1.2%  n_dr_0p1_0p2 < 13
        + 0.01124054 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) / 0.3666529   # +1.1%  z_dr_0_0p05 < 0.8459
        - 0.01113406 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) / 0.0003002514   # -1.1%  sum_z_dr2_top2 < 0.001057
        - 0.01108544 * max(0.0, 1.976207 - Q.D2) / 0.367701   # -1.1%  D2 < 1.976
        - 0.009938632 * max(0.0, 0.002270363 - Q.sum_z_dr2_top5) / 0.0007634099   # -1.0%  sum_z_dr2_top5 < 0.00227
        + 0.009777761 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +1.0%  lam2 < 0.001776
        + 0.009316801 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.9%  sj2_dr > 0.2232
        - 0.008599226 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 14.45919 - Q.sj2_mass2) / 0.03398137   # -0.9%  psi_0p3 > 0.9897 and sj2_mass2 < 14.46
        + 0.008285081 * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.01931881   # +0.8%  z_dr_0p1_0p2 < 0.06473
        - 0.008279126 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -0.8%  psi_0p3 > 0.9897
        - 0.007821582 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -0.8%  log_sum_pt < 6.903
        + 0.007815722 * max(0.0, Q.psi_0p2 - 0.8706159) / 0.08984765   # +0.8%  psi_0p2 > 0.8706
        - 0.007725244 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -0.8%  sum_pt < 1002
        - 0.007152813 * max(0.0, Q.psi_0p1 - 0.9371031) / 0.01088804   # -0.7%  psi_0p1 > 0.9371
        + 0.006959167 * max(0.0, 89.17293 - Q.mass_top30) / 20.1761   # +0.7%  mass_top30 < 89.17
        - 0.00677844 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, Q.n_real_top40 - 22.0) / 0.06106188   # -0.7%  sum_z_dr2_top5 < 0.00833 and n_real_top40 > 22
        - 0.00665492 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 1260.541 - Q.sum_pt) / 87.05552   # -0.7%  z_dr_0_0p05 < 0.8459 and sum_pt < 1261
        - 0.006625666 * max(0.0, Q.sj2_dr - 0.1666442) / 0.04983259   # -0.7%  sj2_dr > 0.1666
        + 0.006574846 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2) / 0.0004272674   # +0.7%  sj2_dr > 0.2232 and C2_b2 < 0.04009
        + 0.006494482 * max(0.0, 42.0 - Q.n_pt_above_5) / 14.5829   # +0.6%  n_pt_above_5 < 42
        + 0.005669347 * max(0.0, 933.1875 - Q.sum_pt_top30) / 20.26547   # +0.6%  sum_pt_top30 < 933.2
        - 0.005536237 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.02970886 - Q.absphi_0) / 0.06277088   # -0.6%  n_dr_0p1_0p2 < 13 and absphi_0 < 0.02971
        + 0.005528297 * max(0.0, 52.15154 - Q.mass_top50) / 3.332261   # +0.6%  mass_top50 < 52.15
        - 0.005415382 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583) / 0.0009330603   # -0.5%  sum_z_dr2_top5 < 0.00833 and sj3_pairmin_over_m > 0.09541
        - 0.005184906 * max(0.0, 22.0 - Q.n_pt_above_10) / 3.900906   # -0.5%  n_pt_above_10 < 22
        + 0.004937163 * max(0.0, 0.2416266 - Q.sj2_zsoft) / 0.06095505   # +0.5%  sj2_zsoft < 0.2416
        + 0.004784734 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 0.777038 - Q.planar_flow) / 0.0106194   # +0.5%  z_dr_0p1_0p2 < 0.1203 and planar_flow < 0.777
        - 0.004628919 * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 3.233647   # -0.5%  n_dr_0p2_0p4 > 9
        - 0.004475408 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0) / 0.2039042   # -0.4%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p2_0p4 > 5
        + 0.004334345 * max(0.0, 935.8189 - Q.sum_pt_top40) / 10.51928   # +0.4%  sum_pt_top40 < 935.8
        + 0.004023204 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt) / 2.722316   # +0.4%  z_dr_0_0p05 < 0.8459 and sum_pt < 949.9
        - 0.003841212 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # -0.4%  sum_pt < 986.1
        + 0.003709785 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, 0.04770182 - Q.z_6) / 5.265762e-05   # +0.4%  sum_z_dr2_top5 < 0.00833 and z_6 < 0.0477
        - 0.003701352 * max(0.0, 0.002288114 - Q.z_dr_0p2_0p4) / 0.0002954373   # -0.4%  z_dr_0p2_0p4 < 0.002288
        + 0.00345432 * max(0.0, 0.0005124533 - Q.sum_z_dr2_top2) / 0.0001104529   # +0.3%  sum_z_dr2_top2 < 0.0005125
        + 0.003385219 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 137.5 - Q.pt_2) / 0.2617409   # +0.3%  psi_0p3 > 0.9897 and pt_2 < 137.5
        - 0.003257815 * max(0.0, 0.2982 - Q.max_dr) / 0.01350087   # -0.3%  max_dr < 0.2982
        - 0.003158153 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.002741632 - Q.soft3_z) / 1.078038e-05   # -0.3%  psi_0p3 > 0.9897 and soft3_z < 0.002742
        - 0.002927156 * max(0.0, Q.lam1 - 0.01649354) / 0.001003295   # -0.3%  lam1 > 0.01649
        - 0.00288309 * max(0.0, 0.05268713 - Q.dr_3) / 0.01614963   # -0.3%  dr_3 < 0.05269
        - 0.002858668 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, Q.sj3_mass1 - 5.112677) / 0.04548726   # -0.3%  sum_z_dr2_top5 < 0.00833 and sj3_mass1 > 5.113
        + 0.002820227 * max(0.0, 0.001776308 - Q.lam2) * max(0.0, Q.soft8_dr0 - 0.0373368) / 0.0001035277   # +0.3%  lam2 < 0.001776 and soft8_dr0 > 0.03734
        - 0.00258308 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 58.0 - Q.n_particles) / 0.8649544   # -0.3%  z_dr_0p1_0p2 < 0.1203 and n_particles < 58
        + 0.00227001 * max(0.0, 0.001776308 - Q.lam2) * max(0.0, Q.ptdr0_10 - 3.719859) / 0.0003099944   # +0.2%  lam2 < 0.001776 and ptdr0_10 > 3.72
        - 0.002208868 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.n_dr_0p05_0p1 - 8.0) / 14.10129   # -0.2%  n_dr_0p1_0p2 < 13 and n_dr_0p05_0p1 > 8
        + 0.002098848 * max(0.0, 0.00287991 - Q.sum_z_dr2_top20) * max(0.0, 0.4357228 - Q.max_dr) / 4.319324e-05   # +0.2%  sum_z_dr2_top20 < 0.00288 and max_dr < 0.4357
        - 0.002001328 * max(0.0, 79.18312 - Q.sd_mass) * max(0.0, Q.pt_10 - 8.304297) / 401.3254   # -0.2%  sd_mass < 79.18 and pt_10 > 8.304
        + 0.001975632 * max(0.0, 35.30013 - Q.sj3_pair_mass_max) / 1.577545   # +0.2%  sj3_pair_mass_max < 35.3
        - 0.001895437 * max(0.0, Q.sd_rg - 0.1690338) / 0.03150941   # -0.2%  sd_rg > 0.169
        - 0.001859818 * max(0.0, Q.e2 - 0.03029714) * max(0.0, 0.3166869 - Q.sj3_z2) / 0.0002537782   # -0.2%  e2 > 0.0303 and sj3_z2 < 0.3167
        + 0.001855485 * max(0.0, Q.dr_13 - 0.2103034) / 0.006604759   # +0.2%  dr_13 > 0.2103
        + 0.001712446 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, Q.ptdr0_8 - 3.519576) / 0.005024459   # +0.2%  psi_0p3 > 0.9897 and ptdr0_8 > 3.52
        - 0.001522824 * max(0.0, Q.sj2_dr - 0.2971569) / 0.005899669   # -0.2%  sj2_dr > 0.2972
        - 0.001379105 * max(0.0, 1.976207 - Q.D2) * max(0.0, 0.2535773 - Q.dr_11) / 0.05913718   # -0.1%  D2 < 1.976 and dr_11 < 0.2536
        - 0.00122156 * max(0.0, 79.18312 - Q.sd_mass) * max(0.0, Q.eta_2 - -0.03302612) / 1.045361   # -0.1%  sd_mass < 79.18 and eta_2 > -0.03303
        + 0.001127805 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, Q.dr1_11 - 0.2621232) / 1.423011e-05   # +0.1%  sum_z_dr2_top5 < 0.00833 and dr1_11 > 0.2621
        - 0.001104999 * max(0.0, Q.psi_0p2 - 0.8706159) * max(0.0, 0.002782871 - Q.zdr_13) / 0.0001527524   # -0.1%  psi_0p2 > 0.8706 and zdr_13 < 0.002783
        + 0.001095221 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +0.1%  mass_over_sum_pt > 0.1709
        - 0.0009947996 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 76.29481 - Q.sd_mass) / 2.433638   # -0.1%  z_dr_0p1_0p2 < 0.1203 and sd_mass < 76.29
        - 0.0009913264 * max(0.0, Q.n_dr_0p2_0p4 - 26.0) / 0.2669462   # -0.1%  n_dr_0p2_0p4 > 26
        + 0.0009881732 * max(0.0, 7.017258 - Q.log_sum_pt) * max(0.0, 13.23436 - Q.sj3_mass2) / 0.4918041   # +0.1%  log_sum_pt < 7.017 and sj3_mass2 < 13.23
        - 0.0009648946 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.02334595 - Q.absphi_9) / 9.860126e-05   # -0.1%  sj2_dr > 0.2232 and absphi_9 < 0.02335
        + 0.0008494424 * max(0.0, Q.e3 - 0.0001841806) / 4.035159e-05   # +0.1%  e3 > 0.0001842
        - 0.0008315704 * max(0.0, 7.017258 - Q.log_sum_pt) * max(0.0, Q.n_dr_0p05_0p1 - 6.0) / 0.5506689   # -0.1%  log_sum_pt < 7.017 and n_dr_0p05_0p1 > 6
        - 0.0007819341 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, 0.948102 - Q.psi_0p2) / 3.786811e-05   # -0.1%  sum_z_dr2_top5 < 0.00833 and psi_0p2 < 0.9481
        - 0.0007631765 * max(0.0, 0.001776308 - Q.lam2) * max(0.0, Q.dr1_3 - 0.1637906) / 6.05992e-06   # -0.1%  lam2 < 0.001776 and dr1_3 > 0.1638
        - 0.0007453334 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) * max(0.0, Q.dr1_11 - 0.1710892) / 3.049596e-06   # -0.1%  sum_z_dr2_top2 < 0.001057 and dr1_11 > 0.1711
        + 0.0005546204 * max(0.0, 45.595 - Q.sd_mass) * max(0.0, -0.009371567 - Q.eta_1) / 0.02469491   # +0.1%  sd_mass < 45.59 and eta_1 < -0.009372
        + 0.0005488468 * max(0.0, 22.0 - Q.n_pt_above_10) * max(0.0, 0.3493565 - Q.soft1_dr) / 0.5156613   # +0.1%  n_pt_above_10 < 22 and soft1_dr < 0.3494
        + 0.0004283386 * max(0.0, Q.dr_7 - 0.2229947) / 0.002728807   # +0.0%  dr_7 > 0.223
        + 0.0003565585 * max(0.0, 0.00287991 - Q.sum_z_dr2_top20) / 0.000559956   # +0.0%  sum_z_dr2_top20 < 0.00288
        - 0.0003115182 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.0009102912 - Q.soft5_z) / 2.772596e-06   # -0.0%  sj2_dr > 0.2232 and soft5_z < 0.0009103
        + 0.0002936419 * max(0.0, Q.dr_7 - 0.2229947) * max(0.0, 0.002337294 - Q.zdr_10) / 1.884273e-06   # +0.0%  dr_7 > 0.223 and zdr_10 < 0.002337
        + 0.0002771517 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p4_up - 0.0) / 0.006297153   # +0.0%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p4_up > 0
        + 0.0002406425 * max(0.0, 0.001776308 - Q.lam2) * max(0.0, Q.dr_12 - 0.2053949) / 4.456629e-06   # +0.0%  lam2 < 0.001776 and dr_12 > 0.2054
        + 0.0001649756 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, Q.dr0_8 - 0.2441824) / 1.193182e-05   # +0.0%  sum_z_dr2_top5 < 0.00833 and dr0_8 > 0.2442
        + 2.194285e-05 * max(0.0, 0.00287991 - Q.sum_z_dr2_top20) * max(0.0, Q.sum_pt - 1167.447) / 0.01532317   # +0.0%  sum_z_dr2_top20 < 0.00288 and sum_pt > 1167
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9272126575630252, 3.3151360819327733, 0.08260252100840336, 0.7111046743697479, 1.4021620798319328, 1.4739653361344538, 0.9939029411764706, 0.7609654411764706, 1.8860506302521007, 2.109816491596639, 1.2175996848739497, 0.8558701680672269, 0.9005058823529412, 1.9265321953781513, 0.4327871323529412, 0.3051577731092437]
T = [5.8249861508665965, 4.097909560464811, 6.516489023109244, 6.504053019957984, 5.078206921284139]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +36%, n4 -21%, n9 +19%, n3 -9%, n5 -8%, n12 +4% ...
            + 0.3557021 * h[1] / H_AVG[1]
            - 0.2106257 * h[4] / H_AVG[4]
            + 0.18676 * h[9] / H_AVG[9]
            - 0.09155876 * h[3] / H_AVG[3]
            - 0.07907558 * h[5] / H_AVG[5]
            + 0.03623289 * h[12] / H_AVG[12]
            + 0.02666055 * h[6] / H_AVG[6]
            + 0.01011832 * h[8] / H_AVG[8]
            - 0.003266101 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -36%, n9 +29%, n1 -15%, n12 +8%, n6 +5%, n11 -5% ...
            - 0.3635505 * h[4] / H_AVG[4]
            + 0.2896042 * h[9] / H_AVG[9]
            - 0.1516842 * h[1] / H_AVG[1]
            + 0.07553825 * h[12] / H_AVG[12]
            + 0.05305541 * h[6] / H_AVG[6]
            - 0.05221383 * h[11] / H_AVG[11]
            + 0.00719136 * h[8] / H_AVG[8]
            - 0.00464261 * h[10] / H_AVG[10]
            + 0.002519654 * h[2] / H_AVG[2]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +13%, n0 +11%, n14 -9%, n11 +8%, n4 +7% ...
            - 0.253249 * h[8] / H_AVG[8]
            + 0.1307662 * h[5] / H_AVG[5]
            + 0.1067154 * h[0] / H_AVG[0]
            - 0.09131947 * h[14] / H_AVG[14]
            + 0.07798262 * h[11] / H_AVG[11]
            + 0.07396517 * h[4] / H_AVG[4]
            - 0.07298461 * h[7] / H_AVG[7]
            - 0.07082378 * h[9] / H_AVG[9]
            - 0.05613921 * h[12] / H_AVG[12]
            + 0.0477417 * h[3] / H_AVG[3]
            + 0.009532577 * h[6] / H_AVG[6]
            - 0.008780354 * h[15] / H_AVG[15]
        ),
        0.984375 + T[3] * (   # class Z: n8 -27%, n0 -20%, n5 +16%, n6 -15%, n7 +11%, n3 +5% ...
            - 0.271857 * h[8] / H_AVG[8]
            - 0.1960189 * h[0] / H_AVG[0]
            + 0.155803 * h[5] / H_AVG[5]
            - 0.1480375 * h[6] / H_AVG[6]
            + 0.10603 * h[7] / H_AVG[7]
            + 0.04783299 * h[3] / H_AVG[3]
            + 0.04326657 * h[12] / H_AVG[12]
            + 0.02639143 * h[15] / H_AVG[15]
            - 0.00476256 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +24%, n5 -14%, n8 +8%, n12 -7%, n4 +5% ...
            - 0.3438064 * h[13] / H_AVG[13]
            + 0.2360232 * h[10] / H_AVG[10]
            - 0.1360561 * h[5] / H_AVG[5]
            + 0.07834238 * h[8] / H_AVG[8]
            - 0.06649782 * h[12] / H_AVG[12]
            + 0.0517713 * h[4] / H_AVG[4]
            - 0.0421451 * h[7] / H_AVG[7]
            + 0.02282333 * h[0] / H_AVG[0]
            - 0.02253436 * h[15] / H_AVG[15]
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
