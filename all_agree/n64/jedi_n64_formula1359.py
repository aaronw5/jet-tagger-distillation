"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; all observables, tuned for agreement), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_10                 pT share × ΔR of particle 10 (its part of the girth)
  Q.zdr_11                 pT share × ΔR of particle 11 (its part of the girth)
  Q.zdr_12                 pT share × ΔR of particle 12 (its part of the girth)
  Q.zdr_13                 pT share × ΔR of particle 13 (its part of the girth)
  Q.zdr_14                 pT share × ΔR of particle 14 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
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
  Q.mean_eta               pT-weighted mean Δη
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
        mean_eta=sum(z[i] * eta[i] for i in P),
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
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = 1.200506
    if Q.mass < 53.87362:
        z += 0.02883047 * Q.mass - 3.023672
    if 53.87362 <= Q.mass < 74.25181:
        z += 0.03116982 * Q.mass - 3.149702
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.01384218 * Q.mass - 1.863093
    if 78.26182 <= Q.mass < 89.74183:
        z += -0.1174612 * Q.mass + 8.412951
    if 89.74183 <= Q.mass < 91.19:
        z += -0.02481328 * Q.mass + 0.09855383
    if 91.19 <= Q.mass < 92.85979:
        z += 0.08322921 * Q.mass - 9.75384
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.06933072 * Q.mass - 8.46323
    if Q.mass >= 101.0497:
        z += 0.0381609 * Q.mass - 5.313529
    if Q.girth2_top20 < 0.005312783:
        z += 45.42341 * Q.girth2_top20 - 0.2896823
    if 0.005312783 <= Q.girth2_top20 < 0.006374178:
        z += 105.7968 * Q.girth2_top20 - 0.6104331
    if 0.006374178 <= Q.girth2_top20 < 0.007538019:
        z += -54.93417 * Q.girth2_top20 + 0.4140948
    if Q.sum_pt < 972.0419:
        z += 0.005674473 * Q.sum_pt - 5.656463
    if 972.0419 <= Q.sum_pt < 1012.673:
        z += 0.003461335 * Q.sum_pt - 3.5052
    if Q.psi_0p3 >= 0.9956185:
        z += 52.14975 * Q.psi_0p3 - 51.92125
    if Q.log_sum_pt < 6.98945:
        z += -2.897859 * Q.log_sum_pt + 20.29247
    if 6.98945 <= Q.log_sum_pt < 7.017258:
        z += -1.36731 * Q.log_sum_pt + 9.594769
    if Q.mass_top30 < 80.4:
        z += -0.006112454 * Q.mass_top30 + 0.4914413
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.02754481 * Q.n_dr_0p2_0p4 + 0.3585984
    if 11.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += -0.01390137 * Q.n_dr_0p2_0p4 + 0.2085206
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += 3.397709 * Q.z_dr_0p2_0p4 - 0.2649414
    if 0.09122568 <= Q.z_dr_0p2_0p4 < 0.1291856:
        z += -1.185907 * Q.z_dr_0p2_0p4 + 0.1532021
    if Q.girth2_top30 < 0.006363916:
        z += 103.7738 * Q.girth2_top30 - 0.5076306
    if 0.006363916 <= Q.girth2_top30 < 0.007856958:
        z += -102.3259 * Q.girth2_top30 + 0.8039705
    if Q.lam1 < 0.005913555:
        z += 97.63578 * Q.lam1 - 0.5773745
    if Q.mass_over_sum_pt_sq < 0.004754444:
        z += -1027.624 * Q.mass_over_sum_pt_sq + 7.004823
    if 0.004754444 <= Q.mass_over_sum_pt_sq < 0.006938798:
        z += -814.5111 * Q.mass_over_sum_pt_sq + 5.991589
    if 0.006938798 <= Q.mass_over_sum_pt_sq < 0.007873266:
        z += -363.6946 * Q.mass_over_sum_pt_sq + 2.863465
    if Q.tau1 < 0.0705748:
        z += 23.61185 * Q.tau1 - 1.666402
    if Q.e2_sq < 0.00616708:
        z += 504.5122 * Q.e2_sq - 3.111367
    if Q.e2 < 0.01879315:
        z += -26.64192 * Q.e2 + 0.5006856
    if Q.e2 >= 0.05557149:
        z += -3.519588 * Q.e2 + 0.1955888
    if 71.79516 <= Q.mass_top50 < 82.04491:
        z += 0.02013458 * Q.mass_top50 - 1.445565
    if Q.mass_top50 >= 82.04491:
        z += -0.03788638 * Q.mass_top50 + 3.31476
    if Q.z_top50_slots < 0.9906378:
        z += 17.3161 * Q.z_top50_slots - 17.15398
    if Q.soft5_z < 0.002788182:
        z += 60.67867 * Q.soft5_z - 0.1691832
    if Q.psi_0p1 >= 0.9184255:
        z += -1.939114 * Q.psi_0p1 + 1.780931
    if Q.z_dr_0p05_0p1 >= 0.7108211:
        z += 1.001803 * Q.z_dr_0p05_0p1 - 0.7121027
    if Q.sum_pt_top40 < 858.8262:
        z += 0.003228626 * Q.sum_pt_top40 - 3.502114
    if 858.8262 <= Q.sum_pt_top40 < 1069.671:
        z += 0.003458869 * Q.sum_pt_top40 - 3.699853
    if Q.sum_pt_top20 < 846.1934:
        z += -0.0004677637 * Q.sum_pt_top20 + 0.3958186
    if Q.mass_top40 < 80.89043:
        z += 0.006406981 * Q.mass_top40 - 0.5182634
    if Q.e3 < 0.0005178279:
        z += -303.8286 * Q.e3 + 0.1573309
    if Q.C2 < 0.05602756:
        z += 1.757106 * Q.C2 - 0.09844637
    if Q.lam2 < 0.0006154841:
        z += -313.4458 * Q.lam2 + 0.1929209
    if Q.sum_pt_top50 < 1156.659:
        z += -0.002084149 * Q.sum_pt_top50 + 2.410651
    if Q.dr_2 < 0.03843804:
        z += -3.768963 * Q.dr_2 + 0.1448716
    if Q.soft10_pt < 2.744141:
        z += 0.05845908 * Q.soft10_pt - 0.1604199
    if Q.sj2_dr >= 0.2232169:
        z += -1.364845 * Q.sj2_dr + 0.3046565
    if Q.sd_mass < 86.4:
        z += -0.001077533 * Q.sd_mass + 0.09309883
    if Q.M2 >= 0.05568888:
        z += 0.1352294 * Q.M2 - 0.007530775
    if Q.sum_pt < 1012.673 and Q.M3 < 0.03457336:
        z += -0.0303737 * (1012.673 - Q.sum_pt) * (0.03457336 - Q.M3)
    if Q.lam1 < 0.005913555 and Q.z_7 < 0.03230249:
        z += 1039.473 * (0.005913555 - Q.lam1) * (0.03230249 - Q.z_7)
    if Q.log_sum_pt < 6.98945 and Q.sum_pt_top50 > 959.0957:
        z += 0.09147502 * (6.98945 - Q.log_sum_pt) * (Q.sum_pt_top50 - 959.0957)
    if Q.z_dr_0p2_0p4 < 0.09122568 and Q.pt_2 < 118.5:
        z += 0.02452203 * (0.09122568 - Q.z_dr_0p2_0p4) * (118.5 - Q.pt_2)
    if Q.soft5_z < 0.002788182 and Q.n_real_top30 < 30.0:
        z += 6.808879 * (0.002788182 - Q.soft5_z) * (30.0 - Q.n_real_top30)
    if Q.mass < 101.0497 and Q.eta_1 > 0.0001021922:
        z += -0.04571367 * (101.0497 - Q.mass) * (Q.eta_1 - 0.0001021922)
    if Q.psi_0p3 > 0.9956185 and Q.dr_max_012 > 0.06112084:
        z += -130.2588 * (Q.psi_0p3 - 0.9956185) * (Q.dr_max_012 - 0.06112084)
    if Q.girth2_top30 < 0.007856958 and Q.ptdr0_4 > 8.939062:
        z += 2.794452 * (0.007856958 - Q.girth2_top30) * (Q.ptdr0_4 - 8.939062)
    if Q.log_sum_pt < 7.017258 and Q.soft4_dr < 0.199317:
        z += 2.140764 * (7.017258 - Q.log_sum_pt) * (0.199317 - Q.soft4_dr)
    if Q.mass_over_sum_pt_sq < 0.007873266 and Q.pt_10 < 34.0625:
        z += 1.371026 * (0.007873266 - Q.mass_over_sum_pt_sq) * (34.0625 - Q.pt_10)
    if Q.sum_pt < 1012.673 and Q.C3 < 0.0223982:
        z += -0.0788245 * (1012.673 - Q.sum_pt) * (0.0223982 - Q.C3)
    if Q.sum_pt < 972.0419 and Q.z_4 < 0.05554124:
        z += -0.01737526 * (972.0419 - Q.sum_pt) * (0.05554124 - Q.z_4)
    if Q.psi_0p3 > 0.9956185 and Q.pt_4 > 65.1875:
        z += -1.619107 * (Q.psi_0p3 - 0.9956185) * (Q.pt_4 - 65.1875)
    if Q.C2 < 0.05602756 and Q.phi_10 > 0.0647583:
        z += 181.5462 * (0.05602756 - Q.C2) * (Q.phi_10 - 0.0647583)
    if Q.mass < 101.0497 and Q.eta_0 < 0.02980347:
        z += 0.07688691 * (101.0497 - Q.mass) * (0.02980347 - Q.eta_0)
    if Q.psi_0p1 > 0.9184255 and Q.eta_4 < -0.02580643:
        z += 68.79475 * (Q.psi_0p1 - 0.9184255) * (-0.02580643 - Q.eta_4)
    if Q.sum_pt_top50 < 1156.659 and Q.soft10_dr < 0.1846742:
        z += 0.003860896 * (1156.659 - Q.sum_pt_top50) * (0.1846742 - Q.soft10_dr)
    if Q.sum_pt < 972.0419 and Q.mass_top3 > 23.66386:
        z += 7.11744e-05 * (972.0419 - Q.sum_pt) * (Q.mass_top3 - 23.66386)
    if Q.sum_pt < 972.0419 and Q.soft9_dr < 0.2400617:
        z += -0.01271689 * (972.0419 - Q.sum_pt) * (0.2400617 - Q.soft9_dr)
    if Q.sum_pt_top50 < 1156.659 and Q.soft9_dr < 0.2702923:
        z += 0.002050926 * (1156.659 - Q.sum_pt_top50) * (0.2702923 - Q.soft9_dr)
    if Q.sum_pt < 1012.673 and Q.dr_9 > 0.04680031:
        z += 0.002010752 * (1012.673 - Q.sum_pt) * (Q.dr_9 - 0.04680031)
    if Q.z_dr_0p2_0p4 < 0.1291856 and Q.pair_mass_0_5 > 18.80588:
        z += -0.01117025 * (0.1291856 - Q.z_dr_0p2_0p4) * (Q.pair_mass_0_5 - 18.80588)
    if Q.sd_mass < 86.4 and Q.soft5_dr > 0.1693667:
        z += 0.00167035 * (86.4 - Q.sd_mass) * (Q.soft5_dr - 0.1693667)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.441989
    if Q.n_particles >= 38.0:
        z += 0.04372904 * Q.n_particles - 1.661703
    if Q.log_sum_pt < 6.893714:
        z += 7.129588 * Q.log_sum_pt - 50.90024
    if 6.893714 <= Q.log_sum_pt < 6.910131:
        z += 32.91002 * Q.log_sum_pt - 228.6232
    if 6.910131 <= Q.log_sum_pt < 6.959294:
        z += 54.37739 * Q.log_sum_pt - 376.9655
    if 6.959294 <= Q.log_sum_pt < 6.98945:
        z += 41.79383 * Q.log_sum_pt - 289.3928
    if 6.98945 <= Q.log_sum_pt < 7.062574:
        z += 31.74424 * Q.log_sum_pt - 219.1517
    if 7.062574 <= Q.log_sum_pt < 7.139296:
        z += 29.35589 * Q.log_sum_pt - 202.2838
    if Q.log_sum_pt >= 7.139296:
        z += 22.2263 * Q.log_sum_pt - 151.3836
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.008244866 * Q.sum_pt_top50 + 7.907615
    if Q.psi_0p3 >= 0.9980008:
        z += -223.709 * Q.psi_0p3 + 223.2618
    if Q.sum_pt_top2 < 689.25:
        z += -0.00145648 * Q.sum_pt_top2 + 1.003879
    if Q.z_top30_slots >= 0.9341838:
        z += -4.888462 * Q.z_top30_slots + 4.566722
    if Q.mass_top20 < 47.88842:
        z += 0.03306416 * Q.mass_top20 - 1.58339
    if Q.sj3_mass1 < 32.50209:
        z += 0.03391061 * Q.sj3_mass1 - 1.102166
    if Q.sum_pt_top40 < 1069.671:
        z += -0.007232909 * Q.sum_pt_top40 + 7.736834
    if Q.girth2_top15 < 0.0007894752:
        z += -986.5531 * Q.girth2_top15 + 1.004832
    if 0.0007894752 <= Q.girth2_top15 < 0.003270031:
        z += -91.0977 * Q.girth2_top15 + 0.2978923
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.1422228 * Q.n_dr_0p2_0p4 - 1.247722
    if 7.0 <= Q.n_dr_0p2_0p4 < 13.0:
        z += 0.04202699 * Q.n_dr_0p2_0p4 - 0.5463509
    if Q.girth2_top3 < 0.0005522528:
        z += -304.3148 * Q.girth2_top3 + 0.1680587
    if Q.M3 < 0.03187688:
        z += 6.940784 * Q.M3 - 0.2212506
    if Q.sj2_mass1 < 30.26161:
        z += 0.0333369 * Q.sj2_mass1 - 1.008828
    if Q.soft5_z < 0.001594761:
        z += -216.0011 * Q.soft5_z + 0.3444702
    if Q.pt_9 < 31.35938:
        z += 0.03417119 * Q.pt_9 - 1.071587
    if Q.lam1 < 0.004673423:
        z += -175.0279 * Q.lam1 + 0.8179794
    if Q.mass < 120.6:
        z += 0.02211839 * Q.mass - 2.667478
    if Q.girth2_top30 < 0.02412652:
        z += -30.74836 * Q.girth2_top30 + 0.741851
    if Q.N2 >= 0.4678622:
        z += 3.928525 * Q.N2 - 1.838009
    if Q.tau3 < 0.009068077:
        z += 68.5066 * Q.tau3 - 0.272968
    if 0.009068077 <= Q.tau3 < 0.04516808:
        z += -9.646957 * Q.tau3 + 0.4357345
    if Q.D3 < 0.1416054:
        z += -2.401434 * Q.D3 + 0.3400561
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -5.004112 * Q.z_dr_0_0p05 + 4.398144
    if Q.mass_top30 < 42.41192:
        z += -0.01831133 * Q.mass_top30 + 0.7766186
    if Q.n_dr_0_0p05 < 12.0:
        z += -0.02800459 * Q.n_dr_0_0p05 + 0.336055
    if Q.mass_top10 < 56.92192:
        z += -0.0001329523 * Q.mass_top10 + 0.01135573
    if 56.92192 <= Q.mass_top10 < 85.41209:
        z += 0.007880436 * Q.mass_top10 - 0.4447817
    if Q.mass_top10 >= 85.41209:
        z += 0.008013388 * Q.mass_top10 - 0.4561375
    z += -0.3695236 * Q.absphi_0
    if Q.sj3_dr_max >= 0.3276439:
        z += 1.68725 * Q.sj3_dr_max - 0.552817
    if Q.soft1_pt < 1.521582:
        z += 0.07487679 * Q.soft1_pt - 0.5710113
    if 1.521582 <= Q.soft1_pt < 2.275391:
        z += 0.606361 * Q.soft1_pt - 1.379708
    if Q.sum_pt < 1017.435:
        z += -0.01083465 * Q.sum_pt + 11.02355
    if Q.sum_pt_top30 >= 1191.938:
        z += 0.008334429 * Q.sum_pt_top30 - 9.934122
    if Q.z_top20_slots >= 0.8965411:
        z += 3.872717 * Q.z_top20_slots - 3.47205
    if Q.sum_pt_top5 >= 430.75:
        z += -0.0005531112 * Q.sum_pt_top5 + 0.2382526
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.08073564 * Q.n_dr_0p1_0p2 - 0.6458851
    if Q.pt_entropy >= 2.07371:
        z += 1.224849 * Q.pt_entropy - 2.539982
    if Q.girth2_top20 < 0.001823079:
        z += -531.0037 * Q.girth2_top20 + 1.440437
    if 0.001823079 <= Q.girth2_top20 < 0.007538019:
        z += -82.65627 * Q.girth2_top20 + 0.6230645
    if Q.psi_0p1 < 0.3628388:
        z += -1.494534 * Q.psi_0p1 + 0.5422748
    if Q.psi_0p1 >= 0.8747961:
        z += 1.254154 * Q.psi_0p1 - 1.097129
    if Q.mass_top50 < 80.35535:
        z += -0.008809049 * Q.mass_top50 + 0.7078542
    if Q.LHA < 0.2601462:
        z += -3.489932 * Q.LHA + 0.9078925
    if Q.sj3_mass3 < 6.160019:
        z += 0.06011416 * Q.sj3_mass3 - 0.3703044
    if Q.sj3_z3 < 0.08974974:
        z += -3.792972 * Q.sj3_z3 + 0.3404183
    if Q.M2 >= 0.1011005:
        z += 2.436408 * Q.M2 - 0.2463219
    if Q.e2 < 0.02515919:
        z += 9.684255 * Q.e2 - 0.243648
    if Q.e2 >= 0.06524004:
        z += -7.161653 * Q.e2 + 0.4672265
    if Q.girth2_top5 < 0.0006570502:
        z += -795.2379 * Q.girth2_top5 + 0.5225112
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += 232.5491 * Q.mass_over_sum_pt_sq - 2.231312
    if Q.girth2_top40 < 0.008031986:
        z += -201.3088 * Q.girth2_top40 + 1.61691
    if Q.z_top50_slots >= 0.9586536:
        z += -33.63726 * Q.z_top50_slots + 32.24648
    if Q.n_for_90pct >= 11.0:
        z += -0.01127738 * Q.n_for_90pct + 0.1240512
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -4.407204 * Q.z_dr_0p2_0p4 + 0.4020502
    if Q.tau1 < 0.1953848:
        z += -12.59449 * Q.tau1 + 2.460773
    if Q.C2_b2 < 0.003468228:
        z += -32.97687 * Q.C2_b2 + 0.1143713
    if Q.D2 < 1.788105:
        z += 0.03602758 * Q.D2 - 0.06442109
    if Q.z_7 < 0.02274107:
        z += 48.2748 * Q.z_7 - 1.097821
    if Q.n_particles > 38.0 and Q.mass_top15 < 57.87349:
        z += 0.0006927706 * (Q.n_particles - 38.0) * (57.87349 - Q.mass_top15)
    if Q.z_top30_slots > 0.9341838 and Q.max_pair_mass > 13.04793:
        z += 0.4156123 * (Q.z_top30_slots - 0.9341838) * (Q.max_pair_mass - 13.04793)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.1119555:
        z += 0.3287576 * (Q.n_particles - 38.0) * (0.1119555 - Q.dr_0)
    if Q.mass_top20 < 47.88842 and Q.n_real_top40 > 29.0:
        z += 0.004175672 * (47.88842 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 38.0 and Q.soft1_pt < 2.275391:
        z += -0.01346717 * (Q.n_particles - 38.0) * (2.275391 - Q.soft1_pt)
    if Q.z_top30_slots > 0.9341838 and Q.C2 < 0.07279889:
        z += 172.5028 * (Q.z_top30_slots - 0.9341838) * (0.07279889 - Q.C2)
    if Q.sj3_mass1 < 32.50209 and Q.sj3_mass2 < 18.68222:
        z += -0.002190732 * (32.50209 - Q.sj3_mass1) * (18.68222 - Q.sj3_mass2)
    if Q.girth2_top15 < 0.003270031 and Q.psi_0p3 > 0.9973959:
        z += 13440.56 * (0.003270031 - Q.girth2_top15) * (Q.psi_0p3 - 0.9973959)
    if Q.M3 < 0.03187688 and Q.M2 > 0.05568888:
        z += -156.9267 * (0.03187688 - Q.M3) * (Q.M2 - 0.05568888)
    if Q.n_particles > 38.0 and Q.dr_1 < 0.1611545:
        z += 0.2278657 * (Q.n_particles - 38.0) * (0.1611545 - Q.dr_1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 > 7.407874:
        z += 0.8161604 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_3 - 7.407874)
    if Q.M3 < 0.03187688 and Q.psi_0p3 < 1.0:
        z += 85.12991 * (0.03187688 - Q.M3) * (1.0 - Q.psi_0p3)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_4 > 6.984554:
        z += 0.3859606 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_4 - 6.984554)
    if Q.sum_pt_top2 < 689.25 and Q.tau43 < 0.9624339:
        z += -0.0001249291 * (689.25 - Q.sum_pt_top2) * (0.9624339 - Q.tau43)
    if Q.pt_9 < 31.35938 and Q.dr1_12 < 0.3241858:
        z += -0.01515309 * (31.35938 - Q.pt_9) * (0.3241858 - Q.dr1_12)
    if Q.log_sum_pt < 7.139296 and Q.pt1_dr01 < 5.351077:
        z += -0.2284302 * (7.139296 - Q.log_sum_pt) * (5.351077 - Q.pt1_dr01)
    if Q.pt_9 < 31.35938 and Q.pair_mass_0_13 < 11.29505:
        z += 0.0002824324 * (31.35938 - Q.pt_9) * (11.29505 - Q.pair_mass_0_13)
    if Q.n_particles > 38.0 and Q.sj3_dr12 < 0.2313408:
        z += 0.1167648 * (Q.n_particles - 38.0) * (0.2313408 - Q.sj3_dr12)
    if Q.sum_pt_top5 > 430.75 and Q.eta_0 < 0.02980347:
        z += -0.003647342 * (Q.sum_pt_top5 - 430.75) * (0.02980347 - Q.eta_0)
    if Q.girth2_top30 < 0.02412652 and Q.zdr_14 > 0.0025721:
        z += 10671.18 * (0.02412652 - Q.girth2_top30) * (Q.zdr_14 - 0.0025721)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.soft10_abseta < 0.002968979:
        z += -14.49221 * (7.0 - Q.n_dr_0p2_0p4) * (0.002968979 - Q.soft10_abseta)
    if Q.z_top20_slots > 0.8965411 and Q.dr_2 < 0.03843804:
        z += -456.0535 * (Q.z_top20_slots - 0.8965411) * (0.03843804 - Q.dr_2)
    if Q.mass < 120.6 and Q.dr_2 < 0.03843804:
        z += 0.2231761 * (120.6 - Q.mass) * (0.03843804 - Q.dr_2)
    if Q.girth2_top20 < 0.007538019 and Q.eta_1 > -0.04302979:
        z += 1056.391 * (0.007538019 - Q.girth2_top20) * (Q.eta_1 - -0.04302979)
    if Q.sum_pt_top2 < 689.25 and Q.sj3_mass2 > 2.12465:
        z += 5.760043e-05 * (689.25 - Q.sum_pt_top2) * (Q.sj3_mass2 - 2.12465)
    if Q.pt_9 < 31.35938 and Q.dr_2 > 0.08525808:
        z += -0.3010409 * (31.35938 - Q.pt_9) * (Q.dr_2 - 0.08525808)
    if Q.soft1_pt < 1.521582 and Q.soft10_abseta < 0.1307373:
        z += -2.244887 * (1.521582 - Q.soft1_pt) * (0.1307373 - Q.soft10_abseta)
    if Q.sj2_mass1 < 30.26161 and Q.soft10_dr0 < 0.2366434:
        z += -0.0250398 * (30.26161 - Q.sj2_mass1) * (0.2366434 - Q.soft10_dr0)
    if Q.n_particles > 38.0 and Q.dr_9 < 0.1546554:
        z += 0.1350068 * (Q.n_particles - 38.0) * (0.1546554 - Q.dr_9)
    if Q.pt_entropy > 2.07371 and Q.soft3_dr < 0.3797:
        z += -0.03175362 * (Q.pt_entropy - 2.07371) * (0.3797 - Q.soft3_dr)
    if Q.sum_pt_top2 < 689.25 and Q.soft3_dr0 > 0.02918107:
        z += -0.0004483071 * (689.25 - Q.sum_pt_top2) * (Q.soft3_dr0 - 0.02918107)
    if Q.n_dr_0_0p05 < 12.0 and Q.orientation_deg < 35.6254:
        z += -0.0002312413 * (12.0 - Q.n_dr_0_0p05) * (35.6254 - Q.orientation_deg)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.4183445
    if 6.903423 <= Q.log_sum_pt < 6.959294:
        z += 0.4179544 * Q.log_sum_pt - 2.885316
    if 6.959294 <= Q.log_sum_pt < 7.062574:
        z += -7.842184 * Q.log_sum_pt + 54.59941
    if Q.log_sum_pt >= 7.062574:
        z += -10.12722 * Q.log_sum_pt + 70.73762
    if Q.sum_pt < 972.0419:
        z += 0.002769692 * Q.sum_pt - 3.440948
    if 972.0419 <= Q.sum_pt < 1017.435:
        z += 0.002595126 * Q.sum_pt - 3.271263
    if 1017.435 <= Q.sum_pt < 1115.723:
        z += 0.0118157 * Q.sum_pt - 12.6526
    if 1115.723 <= Q.sum_pt < 1260.541:
        z += 0.007199295 * Q.sum_pt - 7.501965
    if Q.sum_pt >= 1260.541:
        z += 0.004604169 * Q.sum_pt - 4.230703
    if Q.n_particles < 51.0:
        z += 0.003198031 * Q.n_particles - 0.1630996
    if Q.sum_pt_top20 >= 1129.275:
        z += 0.000847274 * Q.sum_pt_top20 - 0.9568053
    if Q.sum_pt_top50 < 988.4554:
        z += -0.003457934 * Q.sum_pt_top50 + 3.502657
    if 988.4554 <= Q.sum_pt_top50 < 1048.098:
        z += -0.003766553 * Q.sum_pt_top50 + 3.807712
    if 1048.098 <= Q.sum_pt_top50 < 1078.994:
        z += 0.003627101 * Q.sum_pt_top50 - 3.941563
    if 1078.994 <= Q.sum_pt_top50 < 1156.659:
        z += -0.0003086184 * Q.sum_pt_top50 + 0.3050556
    if Q.sum_pt_top50 >= 1156.659:
        z += -0.0005174227 * Q.sum_pt_top50 + 0.546571
    if Q.z_top20_slots >= 0.7516206:
        z += 0.1504341 * Q.z_top20_slots - 0.1130694
    if Q.mass < 78.26182:
        z += 0.02699686 * Q.mass - 2.673358
    if 78.26182 <= Q.mass < 91.19:
        z += 0.03512063 * Q.mass - 3.30914
    if 91.19 <= Q.mass < 92.85979:
        z += 0.06377377 * Q.mass - 5.922019
    if Q.mass_top30 < 82.66587:
        z += -0.006303362 * Q.mass_top30 + 0.5210729
    if Q.mass_over_sum_pt < 0.09795415:
        z += -17.72983 * Q.mass_over_sum_pt + 1.73671
    if Q.sum_pt_top40 < 984.7009:
        z += 0.002174302 * Q.sum_pt_top40 - 2.158675
    if 984.7009 <= Q.sum_pt_top40 < 1018.698:
        z += 1.296564e-05 * Q.sum_pt_top40 - 0.03040532
    if 1018.698 <= Q.sum_pt_top40 < 1041.263:
        z += -0.004655503 * Q.sum_pt_top40 + 4.725354
    if 1041.263 <= Q.sum_pt_top40 < 1069.671:
        z += -0.002161336 * Q.sum_pt_top40 + 2.128269
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.00147053 * Q.sum_pt_top40 - 1.756633
    if Q.sum_pt_top30 < 966.0633:
        z += 0.001230023 * Q.sum_pt_top30 - 1.291787
    if 966.0633 <= Q.sum_pt_top30 < 996.8867:
        z += 0.003358074 * Q.sum_pt_top30 - 3.347619
    if Q.girth2_top30 < 0.004763596:
        z += 68.59831 * Q.girth2_top30 - 0.4207291
    if 0.004763596 <= Q.girth2_top30 < 0.006363916:
        z += 58.70985 * Q.girth2_top30 - 0.3736245
    if Q.psi_0p3 >= 0.9973959:
        z += -42.69642 * Q.psi_0p3 + 42.58524
    if Q.lam1 < 0.01174405:
        z += -16.45945 * Q.lam1 + 0.1933006
    if Q.sj3_pair_mass_min >= 18.02031:
        z += 0.00180512 * Q.sj3_pair_mass_min - 0.03252882
    if Q.pt_6 < 56.53125:
        z += 0.0001148218 * Q.pt_6 - 0.006491022
    if Q.mass_top50 < 92.16545:
        z += -0.008034058 * Q.mass_top50 + 0.7404626
    if Q.log_sum_pt > 6.903423 and Q.mass_top40 > 77.93668:
        z += -0.0252323 * (Q.log_sum_pt - 6.903423) * (Q.mass_top40 - 77.93668)
    if Q.log_sum_pt > 6.903423 and Q.girth2_top15 < 0.02146578:
        z += 561.9185 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.girth2_top15)
    if Q.log_sum_pt > 6.903423 and Q.e3 < 1.433504e-05:
        z += -108713.0 * (Q.log_sum_pt - 6.903423) * (1.433504e-05 - Q.e3)
    if Q.log_sum_pt > 7.062574 and Q.girth2_top30 > 0.008376291:
        z += 84.54169 * (Q.log_sum_pt - 7.062574) * (Q.girth2_top30 - 0.008376291)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p3 > 0.9299135:
        z += -42.03377 * (Q.log_sum_pt - 6.903423) * (Q.psi_0p3 - 0.9299135)
    if Q.sum_pt > 1017.435 and Q.C2 > 0.02207346:
        z += 0.005579194 * (Q.sum_pt - 1017.435) * (Q.C2 - 0.02207346)
    if Q.log_sum_pt > 6.903423 and Q.soft5_z > 0.0005078411:
        z += 71.81453 * (Q.log_sum_pt - 6.903423) * (Q.soft5_z - 0.0005078411)
    if Q.sum_pt_top50 > 988.4554 and Q.girth2_top15 < 0.02146578:
        z += -0.406421 * (Q.sum_pt_top50 - 988.4554) * (0.02146578 - Q.girth2_top15)
    if Q.sum_pt < 972.0419 and Q.e4 < 5.8505e-08:
        z += 21861.57 * (972.0419 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.sum_pt_top20 > 1129.275 and Q.n_dr_0_0p05 < 1.0:
        z += 0.004462818 * (Q.sum_pt_top20 - 1129.275) * (1.0 - Q.n_dr_0_0p05)
    if Q.sum_pt > 1115.723 and Q.mean_phi > 0.0004404746:
        z += -0.8838965 * (Q.sum_pt - 1115.723) * (Q.mean_phi - 0.0004404746)
    if Q.sum_pt > 1115.723 and Q.eta_5 < -0.1135254:
        z += 0.02435068 * (Q.sum_pt - 1115.723) * (-0.1135254 - Q.eta_5)
    if Q.sum_pt < 1260.541 and Q.n_real_top30 < 30.0:
        z += 0.0001197444 * (1260.541 - Q.sum_pt) * (30.0 - Q.n_real_top30)
    if Q.log_sum_pt > 7.062574 and Q.z_10 < 0.01898316:
        z += 474.9835 * (Q.log_sum_pt - 7.062574) * (0.01898316 - Q.z_10)
    if Q.log_sum_pt > 6.903423 and Q.z_10 < 0.01585274:
        z += 262.1023 * (Q.log_sum_pt - 6.903423) * (0.01585274 - Q.z_10)
    if Q.z_top20_slots > 0.7516206 and Q.sj3_mass1 > 6.519856:
        z += 0.003461982 * (Q.z_top20_slots - 0.7516206) * (Q.sj3_mass1 - 6.519856)
    if Q.sum_pt_top50 < 1048.098 and Q.max_dr < 0.3742561:
        z += 0.002828671 * (1048.098 - Q.sum_pt_top50) * (0.3742561 - Q.max_dr)
    if Q.sum_pt < 1260.541 and Q.max_dr < 0.397021:
        z += -0.00309403 * (1260.541 - Q.sum_pt) * (0.397021 - Q.max_dr)
    if Q.sum_pt_top30 < 996.8867 and Q.soft7_dr < 0.03275811:
        z += 0.07231392 * (996.8867 - Q.sum_pt_top30) * (0.03275811 - Q.soft7_dr)
    if Q.sum_pt_top20 > 1129.275 and Q.eta_1 > 0.08734131:
        z += -0.3984794 * (Q.sum_pt_top20 - 1129.275) * (Q.eta_1 - 0.08734131)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.04221327
    if Q.lam2 < 0.0006154841:
        z += -251.785 * Q.lam2 + 0.1549697
    if Q.n_dr_0p2_0p4 < 1.0:
        z += 0.1148735 * Q.n_dr_0p2_0p4 + 0.1007155
    if 1.0 <= Q.n_dr_0p2_0p4 < 5.0:
        z += -0.04660495 * Q.n_dr_0p2_0p4 + 0.262194
    if 5.0 <= Q.n_dr_0p2_0p4 < 8.0:
        z += -0.009723077 * Q.n_dr_0p2_0p4 + 0.07778461
    if Q.n_particles < 46.0:
        z += -0.03076359 * Q.n_particles + 1.415125
    if Q.psi_0p2 >= 0.9985434:
        z += 230.4117 * Q.psi_0p2 - 230.0761
    if Q.tau21 < 0.3861957:
        z += 1.291864 * Q.tau21 - 0.4989125
    if Q.D2 < 2.410481:
        z += 0.1984382 * Q.D2 - 0.4783315
    if 0.9980008 <= Q.psi_0p3 < 0.9995915:
        z += 57.37204 * Q.psi_0p3 - 57.25734
    if Q.psi_0p3 >= 0.9995915:
        z += -266.1388 * Q.psi_0p3 + 266.1214
    if Q.mass_over_sum_pt < 0.08665515:
        z += -5.346416 * Q.mass_over_sum_pt + 0.4632945
    if Q.z_top30_slots >= 0.9734886:
        z += -1.914842 * Q.z_top30_slots + 1.864077
    if Q.girth2_top40 < 0.006026828:
        z += -101.3287 * Q.girth2_top40 + 0.8978332
    if 0.006026828 <= Q.girth2_top40 < 0.008840538:
        z += -102.0511 * Q.girth2_top40 + 0.902187
    if Q.zdr_0 < 0.003235754:
        z += 75.83432 * Q.zdr_0 - 0.2453812
    if Q.mass_top50 < 79.21004:
        z += 0.006553492 * Q.mass_top50 - 0.5191023
    if Q.mass < 40.2:
        z += -0.002750721 * Q.mass + 0.6955576
    if 40.2 <= Q.mass < 53.87362:
        z += -0.004963402 * Q.mass + 0.7845074
    if 53.87362 <= Q.mass < 86.4:
        z += -0.01372263 * Q.mass + 1.256399
    if 86.4 <= Q.mass < 92.85979:
        z += -0.01095442 * Q.mass + 1.017226
    if Q.girth2 < 0.009614971:
        z += 159.6013 * Q.girth2 - 1.534562
    if Q.n_dr_0p1_0p2 < 10.0:
        z += -0.02145373 * Q.n_dr_0p1_0p2 + 0.2145373
    if Q.n_for_90pct < 11.0:
        z += -0.04517465 * Q.n_for_90pct + 0.4969211
    if Q.sj2_mass1 < 27.56535:
        z += -0.01184212 * Q.sj2_mass1 + 0.3264322
    if Q.mass_top40 < 80.4:
        z += 0.01220995 * Q.mass_top40 - 0.98168
    if Q.lam1 < 0.001260456:
        z += -22.64655 * Q.lam1 + 0.02854497
    if Q.tau4 < 0.006395224:
        z += -88.34758 * Q.tau4 + 0.6542991
    if 0.006395224 <= Q.tau4 < 0.01626937:
        z += -9.043462 * Q.tau4 + 0.1471314
    if Q.z_dr_0p1_0p2 < 0.04748396:
        z += -0.3969387 * Q.z_dr_0p1_0p2 + 0.01884822
    if Q.max_dr < 0.2738063:
        z += -1.690651 * Q.max_dr + 0.4629109
    if Q.girth2_top30 < 0.002582316:
        z += 88.41357 * Q.girth2_top30 - 0.4078951
    if 0.002582316 <= Q.girth2_top30 < 0.006363916:
        z += 47.48872 * Q.girth2_top30 - 0.3022142
    if Q.z_dr_0p2_0p4 < 0.019523:
        z += 7.509263 * Q.z_dr_0p2_0p4 - 0.1466033
    if Q.girth2_top5 < 0.0001140644:
        z += 3033.136 * Q.girth2_top5 - 0.3459728
    if Q.tau3 < 0.02374759:
        z += 3.508035 * Q.tau3 - 0.08330739
    if Q.soft5_absphi < 0.0036273:
        z += 34.19864 * Q.soft5_absphi - 0.1240487
    if Q.absphi_1 < 0.02227783:
        z += 1.258908 * Q.absphi_1 - 0.02804575
    if Q.sj3_mass1 < 9.256377:
        z += -0.01946518 * Q.sj3_mass1 + 0.1801771
    if Q.girth2_top50 < 0.008124776:
        z += -64.32131 * Q.girth2_top50 + 0.5225963
    if Q.n_dr_0p2_0p4 < 5.0 and Q.sum_pt_top30 < 988.4375:
        z += 0.0002440427 * (5.0 - Q.n_dr_0p2_0p4) * (988.4375 - Q.sum_pt_top30)
    if Q.n_particles < 46.0 and Q.mass_top15 > 57.87349:
        z += -0.0006612688 * (46.0 - Q.n_particles) * (Q.mass_top15 - 57.87349)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_dr_0p1_0p2 > 9.0:
        z += -0.006250246 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 9.0)
    if Q.n_particles < 46.0 and Q.sum_pt_top30 > 800.732:
        z += 5.708181e-05 * (46.0 - Q.n_particles) * (Q.sum_pt_top30 - 800.732)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.z_10 < 0.03447112:
        z += -1.739424 * (5.0 - Q.n_dr_0p2_0p4) * (0.03447112 - Q.z_10)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.eccentricity > 0.7333655:
        z += 0.1027309 * (5.0 - Q.n_dr_0p2_0p4) * (Q.eccentricity - 0.7333655)
    if Q.psi_0p3 > 0.9980008 and Q.soft10_z < 0.004962409:
        z += -2451.628 * (Q.psi_0p3 - 0.9980008) * (0.004962409 - Q.soft10_z)
    if Q.D2 < 2.410481 and Q.psi_0p3 > 0.9985421:
        z += 24.65833 * (2.410481 - Q.D2) * (Q.psi_0p3 - 0.9985421)
    if Q.lam2 < 0.0006154841 and Q.orientation_deg < -9.840088:
        z += 4.4749 * (0.0006154841 - Q.lam2) * (-9.840088 - Q.orientation_deg)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.zdr_5 < 0.007099471:
        z += -2.028758 * (5.0 - Q.n_dr_0p2_0p4) * (0.007099471 - Q.zdr_5)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.psi_0p1 < 0.9538343:
        z += 0.105256 * (5.0 - Q.n_dr_0p2_0p4) * (0.9538343 - Q.psi_0p1)
    if Q.tau21 < 0.3861957 and Q.lam1 > 0.007671243:
        z += -79.92378 * (0.3861957 - Q.tau21) * (Q.lam1 - 0.007671243)
    if Q.z_top30_slots > 0.9734886 and Q.eta_9 < 0.1269531:
        z += 8.297894 * (Q.z_top30_slots - 0.9734886) * (0.1269531 - Q.eta_9)
    if Q.D2 < 2.410481 and Q.max_dr > 0.4357228:
        z += 1.004881 * (2.410481 - Q.D2) * (Q.max_dr - 0.4357228)
    if Q.D2 < 2.410481 and Q.soft5_dr < 0.136982:
        z += 0.3633449 * (2.410481 - Q.D2) * (0.136982 - Q.soft5_dr)
    if Q.z_top30_slots > 0.9734886 and Q.n_dr_0_0p05 > 9.0:
        z += 0.7999299 * (Q.z_top30_slots - 0.9734886) * (Q.n_dr_0_0p05 - 9.0)
    if Q.psi_0p3 > 0.9980008 and Q.soft4_dr > 0.08815263:
        z += -1.244147 * (Q.psi_0p3 - 0.9980008) * (Q.soft4_dr - 0.08815263)
    if Q.lam2 < 0.0006154841 and Q.soft10_dr < 0.1309949:
        z += 857.676 * (0.0006154841 - Q.lam2) * (0.1309949 - Q.soft10_dr)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.z_top50_slots > 0.978741:
        z += 3.226367 * (8.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.978741)
    if Q.lam2 < 0.0006154841 and Q.n_dr_0p4_up < 1.0:
        z += 126.0899 * (0.0006154841 - Q.lam2) * (1.0 - Q.n_dr_0p4_up)
    if Q.girth2 < 0.009614971 and Q.eta_0 < 0.02980347:
        z += 425.2579 * (0.009614971 - Q.girth2) * (0.02980347 - Q.eta_0)
    if Q.mass_top50 < 79.21004 and Q.mass_top3 > 8.921413:
        z += -0.003427861 * (79.21004 - Q.mass_top50) * (Q.mass_top3 - 8.921413)
    if Q.girth2_top40 < 0.008840538 and Q.ptdr0_6 > 5.648151:
        z += 6.894493 * (0.008840538 - Q.girth2_top40) * (Q.ptdr0_6 - 5.648151)
    if Q.sj2_mass1 < 27.56535 and Q.phi_13 < 0.1455078:
        z += 0.02047334 * (27.56535 - Q.sj2_mass1) * (0.1455078 - Q.phi_13)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.eta_3 < 0.03329468:
        z += 0.1142284 * (5.0 - Q.n_dr_0p2_0p4) * (0.03329468 - Q.eta_3)
    if Q.psi_0p3 > 0.9980008 and Q.eta_1 > -0.04302979:
        z += -441.4539 * (Q.psi_0p3 - 0.9980008) * (Q.eta_1 - -0.04302979)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.soft5_z < 0.002788182:
        z += 4.042647 * (8.0 - Q.n_dr_0p2_0p4) * (0.002788182 - Q.soft5_z)
    if Q.psi_0p2 > 0.9985434 and Q.absphi_4 < 0.05004883:
        z += 818.1777 * (Q.psi_0p2 - 0.9985434) * (0.05004883 - Q.absphi_4)
    if Q.psi_0p3 > 0.9980008 and Q.sd_zg < 0.254323:
        z += -182.626 * (Q.psi_0p3 - 0.9980008) * (0.254323 - Q.sd_zg)
    if Q.lam2 < 0.0006154841 and Q.soft8_dr > 0.04226709:
        z += 148.2194 * (0.0006154841 - Q.lam2) * (Q.soft8_dr - 0.04226709)
    if Q.max_dr < 0.2738063 and Q.phi_14 > 0.102478:
        z += -84.99958 * (0.2738063 - Q.max_dr) * (Q.phi_14 - 0.102478)
    if Q.psi_0p2 > 0.9985434 and Q.abseta_4 < 0.08575439:
        z += 256.4577 * (Q.psi_0p2 - 0.9985434) * (0.08575439 - Q.abseta_4)
    if Q.sj2_mass1 < 27.56535 and Q.dr_10 < 0.06334002:
        z += 0.3195721 * (27.56535 - Q.sj2_mass1) * (0.06334002 - Q.dr_10)
    if Q.n_dr_0p2_0p4 < 1.0 and Q.n_dr_0p4_up < 1.0:
        z += -0.2549524 * (1.0 - Q.n_dr_0p2_0p4) * (1.0 - Q.n_dr_0p4_up)
    if Q.z_top30_slots > 0.9734886 and Q.eta_8 < -0.05975342:
        z += 25.61747 * (Q.z_top30_slots - 0.9734886) * (-0.05975342 - Q.eta_8)
    if Q.psi_0p3 > 0.9980008 and Q.soft9_abseta < 0.004577637:
        z += -14349.65 * (Q.psi_0p3 - 0.9980008) * (0.004577637 - Q.soft9_abseta)
    if Q.girth2_top40 < 0.008840538 and Q.soft5_pt > 1.652344:
        z += -21.35544 * (0.008840538 - Q.girth2_top40) * (Q.soft5_pt - 1.652344)
    if Q.girth2 < 0.009614971 and Q.pair_mass_0_2 > 32.99439:
        z += 9.037773 * (0.009614971 - Q.girth2) * (Q.pair_mass_0_2 - 32.99439)
    if Q.girth2 < 0.009614971 and Q.phi_0 < 0.002076054:
        z += 716.3538 * (0.009614971 - Q.girth2) * (0.002076054 - Q.phi_0)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.D3 < 0.1225688:
        z += 0.15805 * (5.0 - Q.n_dr_0p2_0p4) * (0.1225688 - Q.D3)
    if Q.n_dr_0p1_0p2 < 10.0 and Q.phi_5 > -0.03793335:
        z += 0.1701521 * (10.0 - Q.n_dr_0p1_0p2) * (Q.phi_5 - -0.03793335)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.n_dr_0_0p05 < 18.0:
        z += -0.002795614 * (8.0 - Q.n_dr_0p2_0p4) * (18.0 - Q.n_dr_0_0p05)
    if Q.z_top30_slots > 0.9734886 and Q.z_dr_0p05_0p1 > 0.2992503:
        z += -10.20043 * (Q.z_top30_slots - 0.9734886) * (Q.z_dr_0p05_0p1 - 0.2992503)
    if Q.tau21 < 0.3861957 and Q.soft10_z < 0.002210911:
        z += -464.0904 * (0.3861957 - Q.tau21) * (0.002210911 - Q.soft10_z)
    if Q.max_dr < 0.2738063 and Q.eta_11 > 0.1364746:
        z += -228.6483 * (0.2738063 - Q.max_dr) * (Q.eta_11 - 0.1364746)
    if Q.lam2 < 0.0006154841 and Q.pt_0 < 525.5:
        z += -0.03413355 * (0.0006154841 - Q.lam2) * (525.5 - Q.pt_0)
    if Q.psi_0p3 > 0.9995915 and Q.mass_top2 > 16.30802:
        z += -21.36828 * (Q.psi_0p3 - 0.9995915) * (Q.mass_top2 - 16.30802)
    if Q.D2 < 2.410481 and Q.D2_b2 < 5.142486:
        z += 0.02895452 * (2.410481 - Q.D2) * (5.142486 - Q.D2_b2)
    if Q.girth2 < 0.009614971 and Q.sj3_mass1 < 23.45965:
        z += 2.585401 * (0.009614971 - Q.girth2) * (23.45965 - Q.sj3_mass1)
    if Q.girth2 < 0.009614971 and Q.sj3_mass2 < 11.91094:
        z += 4.774659 * (0.009614971 - Q.girth2) * (11.91094 - Q.sj3_mass2)
    if Q.n_particles < 46.0 and Q.D2_b2 < 43.66338:
        z += -6.067118e-05 * (46.0 - Q.n_particles) * (43.66338 - Q.D2_b2)
    if Q.tau21 < 0.3861957 and Q.soft3_absphi < 0.0144043:
        z += 17.68945 * (0.3861957 - Q.tau21) * (0.0144043 - Q.soft3_absphi)
    if Q.sj2_mass1 < 27.56535 and Q.sj3_mass3 > 0.8123034:
        z += -0.0001946706 * (27.56535 - Q.sj2_mass1) * (Q.sj3_mass3 - 0.8123034)
    if Q.mass_top50 < 79.21004 and Q.ptdr0_2 > 7.740999:
        z += -0.01252388 * (79.21004 - Q.mass_top50) * (Q.ptdr0_2 - 7.740999)
    if Q.zdr_0 < 0.003235754 and Q.sj3_z2 < 0.331314:
        z += -420.3511 * (0.003235754 - Q.zdr_0) * (0.331314 - Q.sj3_z2)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.psi_0p3 > 0.9924477:
        z += 9.620961 * (5.0 - Q.n_dr_0p2_0p4) * (Q.psi_0p3 - 0.9924477)
    if Q.tau3 < 0.02374759 and Q.sj3_dr13 > 0.2870549:
        z += 209.0782 * (0.02374759 - Q.tau3) * (Q.sj3_dr13 - 0.2870549)
    if Q.psi_0p3 > 0.9995915 and Q.eta_13 > 0.07098389:
        z += -7615.97 * (Q.psi_0p3 - 0.9995915) * (Q.eta_13 - 0.07098389)
    if Q.lam1 < 0.001260456 and Q.sum_pt_top40 < 1001.523:
        z += 2.644042 * (0.001260456 - Q.lam1) * (1001.523 - Q.sum_pt_top40)
    if Q.n_dr_0p1_0p2 < 10.0 and Q.psi_0p2 > 0.9313699:
        z += 0.4611577 * (10.0 - Q.n_dr_0p1_0p2) * (Q.psi_0p2 - 0.9313699)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.sum_pt_top40 > 1095.686:
        z += 0.0001407898 * (8.0 - Q.n_dr_0p2_0p4) * (Q.sum_pt_top40 - 1095.686)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.3761489
    if Q.mass_top15 < 57.87349:
        z += -0.00784168 * Q.mass_top15 + 0.2210095
    if 57.87349 <= Q.mass_top15 < 69.02716:
        z += 0.00311733 * Q.mass_top15 - 0.4132267
    if 69.02716 <= Q.mass_top15 < 75.26407:
        z += 0.008888924 * Q.mass_top15 - 0.8116234
    if 75.26407 <= Q.mass_top15 < 91.19:
        z += 0.008954381 * Q.mass_top15 - 0.81655
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.03459559 * Q.n_dr_0p2_0p4 + 0.7723721
    if 15.0 <= Q.n_dr_0p2_0p4 < 26.0:
        z += -0.02303984 * Q.n_dr_0p2_0p4 + 0.5990359
    if Q.planar_flow < 0.3036026:
        z += -0.4476231 * Q.planar_flow + 0.1358995
    if Q.mass_top40 < 62.55:
        z += 0.01808395 * Q.mass_top40 - 1.995496
    if 62.55 <= Q.mass_top40 < 67.72643:
        z += 0.01551553 * Q.mass_top40 - 1.834842
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.02118866 * Q.mass_top40 - 2.219062
    if 83.32554 <= Q.mass_top40 < 163.2541:
        z += 0.005673888 * Q.mass_top40 - 0.9262857
    if Q.e3 >= 0.0001841806:
        z += 969.1545 * Q.e3 - 0.1784995
    if Q.mass_top30 < 60.43821:
        z += 0.00179683 * Q.mass_top30 - 0.3210469
    if 60.43821 <= Q.mass_top30 < 91.69753:
        z += 0.008930059 * Q.mass_top30 - 0.7521664
    if 91.69753 <= Q.mass_top30 < 120.6:
        z += -0.002307691 * Q.mass_top30 + 0.2783075
    if Q.mass < 74.25181:
        z += 0.03668702 * Q.mass - 1.987701
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.1005708 * Q.mass - 6.731188
    if 78.26182 <= Q.mass < 80.78464:
        z += 0.01850026 * Q.mass - 0.3081975
    if 80.78464 <= Q.mass < 86.4:
        z += 0.007490869 * Q.mass + 0.5811926
    if 86.4 <= Q.mass < 92.85979:
        z += -0.02871761 * Q.mass + 3.709605
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.05297029 * Q.mass + 5.961704
    if 101.0497 <= Q.mass < 120.6:
        z += -0.02078145 * Q.mass + 2.709031
    if 120.6 <= Q.mass < 143.7876:
        z += -0.00874554 * Q.mass + 1.2575
    if Q.psi_0p3 >= 0.9973959:
        z += 126.6518 * Q.psi_0p3 - 126.322
    if Q.sj3_pair_mass_min >= 32.51366:
        z += 0.01131476 * Q.sj3_pair_mass_min - 0.3678842
    if Q.n_particles >= 22.0:
        z += -0.008874469 * Q.n_particles + 0.1952383
    if Q.girth2_top15 < 0.004855289:
        z += 142.4273 * Q.girth2_top15 - 0.6915254
    if 0.00727763 <= Q.girth2_top15 < 0.01563836:
        z += -62.95652 * Q.girth2_top15 + 0.4581743
    if Q.girth2_top15 >= 0.01563836:
        z += 1.75787 * Q.girth2_top15 - 0.5538524
    if 0.02793599 <= Q.e2 < 0.04755309:
        z += 11.45354 * Q.e2 - 0.3199661
    if 0.04755309 <= Q.e2 < 0.06524004:
        z += 1.03146 * Q.e2 + 0.1756361
    if Q.e2 >= 0.06524004:
        z += -34.86418 * Q.e2 + 2.517469
    if 0.2232169 <= Q.sj2_dr < 0.2412757:
        z += 2.487364 * Q.sj2_dr - 0.5552216
    if Q.sj2_dr >= 0.2412757:
        z += -1.486749 * Q.sj2_dr + 0.4036351
    if 0.009606007 <= Q.e2_sq < 0.01396296:
        z += 137.8136 * Q.e2_sq - 1.323838
    if Q.e2_sq >= 0.01396296:
        z += 166.5582 * Q.e2_sq - 1.725198
    if Q.log_sum_pt >= 6.97212:
        z += -2.569322 * Q.log_sum_pt + 17.91362
    if Q.sj2_mass1 < 21.52288:
        z += 0.003705439 * Q.sj2_mass1 - 0.0797517
    if Q.pair_mass_0_12 >= 15.47191:
        z += -0.01310179 * Q.pair_mass_0_12 + 0.2027097
    if Q.sd_rg >= 0.2042612:
        z += -1.393466 * Q.sd_rg + 0.2846311
    if Q.girth < 0.03577037:
        z += -0.8689499 * Q.girth - 0.06474509
    if 0.03577037 <= Q.girth < 0.05660088:
        z += 11.05905 * Q.girth - 0.4914139
    if 0.05660088 <= Q.girth < 0.07031778:
        z += -9.808172 * Q.girth + 0.6896889
    if Q.girth >= 0.1207452:
        z += -14.21773 * Q.girth + 1.716722
    if 0.008241985 <= Q.lam1 < 0.01174405:
        z += -99.2739 * Q.lam1 + 0.818214
    if Q.lam1 >= 0.01174405:
        z += -71.88251 * Q.lam1 + 0.4965282
    if Q.psi_0p1 < 0.9909875:
        z += -0.5253145 * Q.psi_0p1 + 0.5205801
    if Q.lam2 >= 0.001776308:
        z += -14.22221 * Q.lam2 + 0.02526303
    if Q.C2 >= 0.06655881:
        z += -1.310901 * Q.C2 + 0.08725201
    if Q.n_dr_0p1_0p2 < 26.0:
        z += -0.009751208 * Q.n_dr_0p1_0p2 + 0.2535314
    if Q.psi_0p2 >= 0.9734513:
        z += -8.230492 * Q.psi_0p2 + 8.011982
    if Q.sum_pt >= 995.6769:
        z += -4.72902e-05 * Q.sum_pt + 0.04708576
    if Q.tau2 >= 0.05936026:
        z += 1.202387 * Q.tau2 - 0.07137403
    if Q.n_pt_above_10 >= 11.0:
        z += 0.01208707 * Q.n_pt_above_10 - 0.1329578
    if Q.n_pt_above_1 < 58.0:
        z += -0.009314043 * Q.n_pt_above_1 + 0.5402145
    if Q.D3 < 0.1900123:
        z += -1.04507 * Q.D3 + 0.1985762
    if Q.M3 < 0.03187688:
        z += 1.515609 * Q.M3 - 0.04831289
    if 0.007164202 <= Q.girth2_top5 < 0.0168592:
        z += -12.35711 * Q.girth2_top5 + 0.08852883
    if Q.girth2_top5 >= 0.0168592:
        z += 2.579439 * Q.girth2_top5 - 0.1632894
    if Q.mass_over_sum_pt < 0.06030419:
        z += 13.33861 * Q.mass_over_sum_pt - 0.8043741
    if Q.zdr_12 < 0.002227078:
        z += -27.27118 * Q.zdr_12 + 0.06073507
    if Q.planar_flow < 0.3036026 and Q.n_dr_0p1_0p2 > 13.0:
        z += -0.04913148 * (0.3036026 - Q.planar_flow) * (Q.n_dr_0p1_0p2 - 13.0)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.z_top50_slots < 1.0:
        z += -1.182226 * (15.0 - Q.n_dr_0p2_0p4) * (1.0 - Q.z_top50_slots)
    if Q.psi_0p3 > 0.9973959 and Q.n_dr_0_0p05 < 13.0:
        z += -12.65834 * (Q.psi_0p3 - 0.9973959) * (13.0 - Q.n_dr_0_0p05)
    if Q.n_particles > 22.0 and Q.pt_7 < 33.21875:
        z += -0.000425617 * (Q.n_particles - 22.0) * (33.21875 - Q.pt_7)
    if Q.psi_0p3 > 0.9973959 and Q.zdr_0 < 0.005911134:
        z += -4295.539 * (Q.psi_0p3 - 0.9973959) * (0.005911134 - Q.zdr_0)
    if Q.mass_top40 < 83.32554 and Q.zdr_0 > 0.001901263:
        z += -0.7195038 * (83.32554 - Q.mass_top40) * (Q.zdr_0 - 0.001901263)
    if Q.mass < 120.6 and Q.psi_0p3 < 0.9943058:
        z += -0.004888623 * (120.6 - Q.mass) * (0.9943058 - Q.psi_0p3)
    if Q.n_particles > 22.0 and Q.psi_0p3 > 0.9638082:
        z += 0.02760225 * (Q.n_particles - 22.0) * (Q.psi_0p3 - 0.9638082)
    if Q.n_particles > 22.0 and Q.n_dr_0_0p05 > 10.0:
        z += 0.000580848 * (Q.n_particles - 22.0) * (Q.n_dr_0_0p05 - 10.0)
    if Q.n_particles > 22.0 and Q.soft1_pt < 2.275391:
        z += 0.0063125 * (Q.n_particles - 22.0) * (2.275391 - Q.soft1_pt)
    if Q.mass < 101.0497 and Q.zdr_0 > 0.0008718296:
        z += -0.04756386 * (101.0497 - Q.mass) * (Q.zdr_0 - 0.0008718296)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.max_dr > 0.1939977:
        z += -0.04133928 * (15.0 - Q.n_dr_0p2_0p4) * (Q.max_dr - 0.1939977)
    if Q.n_particles > 22.0 and Q.sj3_mass1 > 32.50209:
        z += 0.0001504132 * (Q.n_particles - 22.0) * (Q.sj3_mass1 - 32.50209)
    if Q.girth2_top15 > 0.00727763 and Q.pt_11 < 33.78125:
        z += -0.6662374 * (Q.girth2_top15 - 0.00727763) * (33.78125 - Q.pt_11)
    if Q.n_dr_0p2_0p4 < 26.0 and Q.n_dr_0p1_0p2 > 11.0:
        z += -0.000392505 * (26.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 11.0)
    if Q.planar_flow < 0.3036026 and Q.z_dr_0p4_up > 0.0:
        z += -490.0858 * (0.3036026 - Q.planar_flow) * (Q.z_dr_0p4_up - 0.0)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.sj3_dr13 > 0.1348442:
        z += 0.1336613 * (15.0 - Q.n_dr_0p2_0p4) * (Q.sj3_dr13 - 0.1348442)
    if Q.mass_top40 < 83.32554 and Q.D2_b2 < 0.4953696:
        z += -0.01656949 * (83.32554 - Q.mass_top40) * (0.4953696 - Q.D2_b2)
    if Q.psi_0p1 < 0.9909875 and Q.pt_1 > 99.0:
        z += -0.004981588 * (0.9909875 - Q.psi_0p1) * (Q.pt_1 - 99.0)
    if Q.sj2_dr > 0.2232169 and Q.D2_b2 < 20.40995:
        z += -0.1081173 * (Q.sj2_dr - 0.2232169) * (20.40995 - Q.D2_b2)
    if Q.mass < 80.78464 and Q.D2_b2 < 3.852812:
        z += -0.02011203 * (80.78464 - Q.mass) * (3.852812 - Q.D2_b2)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.D2_b2 < 7.36624:
        z += 0.002695024 * (15.0 - Q.n_dr_0p2_0p4) * (7.36624 - Q.D2_b2)
    if Q.mass_top40 < 62.55 and Q.D2_b2 < 3.014827:
        z += 0.002479777 * (62.55 - Q.mass_top40) * (3.014827 - Q.D2_b2)
    if Q.mass < 101.0497 and Q.D2_b2 < 3.852812:
        z += 0.008031941 * (101.0497 - Q.mass) * (3.852812 - Q.D2_b2)
    if Q.mass_top40 < 83.32554 and Q.D2_b2 < 3.014827:
        z += -0.01478413 * (83.32554 - Q.mass_top40) * (3.014827 - Q.D2_b2)
    if Q.n_particles > 22.0 and Q.D2_b2 < 3.852812:
        z += -0.001229763 * (Q.n_particles - 22.0) * (3.852812 - Q.D2_b2)
    if Q.mass < 80.78464 and Q.D2_b2 < 0.4953696:
        z += -0.0649005 * (80.78464 - Q.mass) * (0.4953696 - Q.D2_b2)
    if Q.mass < 143.7876 and Q.C2_b2 > 0.002289486:
        z += -0.03143841 * (143.7876 - Q.mass) * (Q.C2_b2 - 0.002289486)
    if Q.girth > 0.1207452 and Q.abseta_3 > 0.005970001:
        z += 3.370142 * (Q.girth - 0.1207452) * (Q.abseta_3 - 0.005970001)
    if Q.sj2_mass1 < 21.52288 and Q.pair_mass_0_13 > 8.669189:
        z += -0.003339224 * (21.52288 - Q.sj2_mass1) * (Q.pair_mass_0_13 - 8.669189)
    if Q.girth2_top15 < 0.004855289 and Q.D3 < 0.1645959:
        z += -227.6621 * (0.004855289 - Q.girth2_top15) * (0.1645959 - Q.D3)
    if Q.mass < 92.85979 and Q.girth2_top2 < 0.005180665:
        z += 2.144223 * (92.85979 - Q.mass) * (0.005180665 - Q.girth2_top2)
    if Q.sj2_dr > 0.2232169 and Q.tau32 < 0.6616141:
        z += 6.266863 * (Q.sj2_dr - 0.2232169) * (0.6616141 - Q.tau32)
    if Q.mass_top15 < 91.19 and Q.mass_top2 > 3.19816:
        z += 0.0002800801 * (91.19 - Q.mass_top15) * (Q.mass_top2 - 3.19816)
    if Q.mass < 80.78464 and Q.mean_eta > 0.00249209:
        z += -14.40559 * (80.78464 - Q.mass) * (Q.mean_eta - 0.00249209)
    if Q.log_sum_pt > 6.97212 and Q.abseta_11 > 0.02558899:
        z += -16.73883 * (Q.log_sum_pt - 6.97212) * (Q.abseta_11 - 0.02558899)
    if Q.M3 < 0.03187688 and Q.soft6_dr0 > 0.06490217:
        z += -6.949686 * (0.03187688 - Q.M3) * (Q.soft6_dr0 - 0.06490217)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.7410341
    z += -0.02173357 * Q.n_particles + 1.390948
    if 0.07696632 <= Q.mass_over_sum_pt < 0.09046749:
        z += -21.28049 * Q.mass_over_sum_pt + 1.637881
    if 0.09046749 <= Q.mass_over_sum_pt < 0.1708801:
        z += -28.23097 * Q.mass_over_sum_pt + 2.266674
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 166.2568 * Q.mass_over_sum_pt - 30.96741
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -14.38693 * Q.log_sum_pt + 99.41557
    if 6.920349 <= Q.log_sum_pt < 6.935549:
        z += -23.61382 * Q.log_sum_pt + 163.2689
    if 6.935549 <= Q.log_sum_pt < 6.949481:
        z += -23.33154 * Q.log_sum_pt + 161.3111
    if 6.949481 <= Q.log_sum_pt < 6.98945:
        z += -20.99713 * Q.log_sum_pt + 145.0882
    if Q.log_sum_pt >= 6.98945:
        z += -14.14112 * Q.log_sum_pt + 97.16842
    if 907.9372 <= Q.sum_pt < 986.0565:
        z += 0.005777532 * Q.sum_pt - 5.245636
    if Q.sum_pt >= 986.0565:
        z += -0.004759427 * Q.sum_pt + 5.144401
    if 0.9638082 <= Q.psi_0p3 < 0.9973959:
        z += 6.07998 * Q.psi_0p3 - 5.859935
    if Q.psi_0p3 >= 0.9973959:
        z += 42.1407 * Q.psi_0p3 - 41.82675
    if Q.girth2_top50 >= 0.01951641:
        z += -20.73949 * Q.girth2_top50 + 0.4047605
    if Q.girth2 >= 0.004756928:
        z += 66.07145 * Q.girth2 - 0.3142971
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += 0.01132678 * Q.sum_pt_top50 - 10.58195
    if Q.sum_pt_top50 >= 959.0957:
        z += 0.01857896 * Q.sum_pt_top50 - 17.53748
    if Q.n_pt_above_1 >= 28.0:
        z += 0.005180761 * Q.n_pt_above_1 - 0.1450613
    if Q.sum_pt_top20 >= 1129.275:
        z += 0.003742717 * Q.sum_pt_top20 - 4.226556
    if Q.sum_pt_top3 < 787.6281:
        z += 0.0005145113 * Q.sum_pt_top3 - 0.4052436
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.02137783 * Q.n_dr_0p2_0p4 + 0.2351561
    if Q.n_dr_0p2_0p4 >= 21.0:
        z += -0.0149199 * Q.n_dr_0p2_0p4 + 0.3133178
    if 994.2695 <= Q.sum_pt_top40 < 1024.942:
        z += -0.003892643 * Q.sum_pt_top40 + 3.870337
    if 1024.942 <= Q.sum_pt_top40 < 1069.671:
        z += -0.007002099 * Q.sum_pt_top40 + 7.057349
    if Q.sum_pt_top40 >= 1069.671:
        z += -0.008860938 * Q.sum_pt_top40 + 9.045696
    if Q.sum_pt_top30 >= 933.1875:
        z += 0.004263229 * Q.sum_pt_top30 - 3.978392
    if Q.max_dr < 0.2404747:
        z += -1.090243 * Q.max_dr + 0.3747556
    if 0.2404747 <= Q.max_dr < 0.3437357:
        z += -2.177202 * Q.max_dr + 0.6361416
    if 0.3437357 <= Q.max_dr < 0.4357228:
        z += -1.086959 * Q.max_dr + 0.261386
    if Q.max_dr >= 0.4357228:
        z += -0.4271382 * Q.max_dr - 0.02611273
    if Q.z_top30_slots >= 0.9048492:
        z += -4.062948 * Q.z_top30_slots + 3.676355
    if Q.mass_top40 < 120.6:
        z += 0.00501743 * Q.mass_top40 - 0.752687
    if 120.6 <= Q.mass_top40 < 150.0144:
        z += -0.005871643 * Q.mass_top40 + 0.5605352
    if Q.mass_top40 >= 150.0144:
        z += -0.01088907 * Q.mass_top40 + 1.313222
    if Q.mass < 64.48544:
        z += -0.004427427 * Q.mass + 0.2855046
    if 74.25181 <= Q.mass < 91.19:
        z += 0.006285866 * Q.mass - 0.4667369
    if 91.19 <= Q.mass < 172.4888:
        z += -0.01063649 * Q.mass + 1.076413
    if 172.4888 <= Q.mass < 172.8:
        z += 0.3981626 * Q.mass - 69.43686
    if Q.mass >= 172.8:
        z += 0.02255 * Q.mass - 4.531003
    if Q.girth2_top30 < 0.006363916:
        z += 19.72642 * Q.girth2_top30 - 0.3565903
    if 0.006363916 <= Q.girth2_top30 < 0.007463985:
        z += 120.5525 * Q.girth2_top30 - 0.9982386
    if 0.007463985 <= Q.girth2_top30 < 0.01807679:
        z += 4.410896 * Q.girth2_top30 - 0.1313599
    if Q.girth2_top30 >= 0.01807679:
        z += -15.31552 * Q.girth2_top30 + 0.2252304
    if Q.z_dr_0_0p05 >= 0.9084912:
        z += 2.763067 * Q.z_dr_0_0p05 - 2.510222
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -484.201 * Q.mass_over_sum_pt_sq + 14.13868
    if Q.sj2_mass1 < 23.31647:
        z += -0.01082159 * Q.sj2_mass1 + 0.2523212
    if Q.tau21 < 0.5494307:
        z += 0.2035616 * Q.tau21 - 0.111843
    if 69.65633 <= Q.sd_mass < 86.4:
        z += 0.01046429 * Q.sd_mass - 0.7289039
    if Q.sd_mass >= 86.4:
        z += -0.002504862 * Q.sd_mass + 0.3916307
    if Q.mass_top50 >= 157.5448:
        z += -0.03763666 * Q.mass_top50 + 5.929461
    if Q.girth2_top15 < 0.0004816552:
        z += 672.7534 * Q.girth2_top15 - 0.2460053
    if 0.0004816552 <= Q.girth2_top15 < 0.005788041:
        z += -14.7049 * Q.girth2_top15 + 0.08511255
    if Q.sum_pt_top10 >= 943.6922:
        z += -0.0009650136 * Q.sum_pt_top10 + 0.9106758
    if Q.z_11 < 0.01375115:
        z += -60.60225 * Q.z_11 + 0.8333504
    if Q.n_pt_above_10 >= 31.0:
        z += -0.02694089 * Q.n_pt_above_10 + 0.8351676
    if Q.e2 < 0.01541561:
        z += 18.17664 * Q.e2 + 0.07522852
    if 0.01541561 <= Q.e2 < 0.03875945:
        z += -15.22596 * Q.e2 + 0.5901499
    if Q.M3 < 0.03787151:
        z += -10.28758 * Q.M3 + 0.3896063
    if Q.n_dr_0p1_0p2 < 8.0:
        z += -0.02959289 * Q.n_dr_0p1_0p2 + 0.1231376
    if 8.0 <= Q.n_dr_0p1_0p2 < 21.0:
        z += 0.008738883 * Q.n_dr_0p1_0p2 - 0.1835165
    if Q.mass_top10 >= 71.781:
        z += 0.002916837 * Q.mass_top10 - 0.2093735
    if Q.z_dr_0p2_0p4 < 0.1937447:
        z += 0.2852487 * Q.z_dr_0p2_0p4 - 0.05526542
    if Q.pt_11 < 14.14062:
        z += 0.0468238 * Q.pt_11 - 0.6621177
    if Q.ptdr0_10 < 1.20817:
        z += 0.0314556 * Q.ptdr0_10 - 0.03800372
    if Q.D2 < 1.976207:
        z += -0.1169847 * Q.D2 + 0.231186
    if Q.z_top20_slots >= 0.9102775:
        z += -2.312783 * Q.z_top20_slots + 2.105274
    if Q.pt_14 < 19.34375:
        z += -0.007478672 * Q.pt_14 + 0.1446656
    if Q.z_5 < 0.02946315:
        z += -15.54742 * Q.z_5 + 0.458076
    if Q.psi_0p1 < 0.6635952:
        z += 0.5126144 * Q.psi_0p1 - 0.3401684
    if Q.z_top50_slots >= 0.9906378:
        z += 0.3948263 * Q.z_top50_slots - 0.3911299
    z += 0.2954213 * Q.soft4_dr0
    if Q.girth2_top20 < 0.01083435:
        z += 12.932 * Q.girth2_top20 - 0.1401098
    if Q.tau2 < 0.04828819:
        z += 2.509214 * Q.tau2 - 0.1211654
    if Q.ptdr0_8 < 1.692473:
        z += 0.03002169 * Q.ptdr0_8 - 0.05081091
    if Q.sj2_dr >= 0.1411617:
        z += 1.146235 * Q.sj2_dr - 0.1618045
    if Q.n_particles < 64.0 and Q.z_8 < 0.03260972:
        z += 0.1733425 * (64.0 - Q.n_particles) * (0.03260972 - Q.z_8)
    if Q.n_particles < 64.0 and Q.D2 < 2.178951:
        z += -0.01071974 * (64.0 - Q.n_particles) * (2.178951 - Q.D2)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 27162.22 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    if Q.log_sum_pt > 6.910131 and Q.dr_12 < 0.1269782:
        z += -13.96657 * (Q.log_sum_pt - 6.910131) * (0.1269782 - Q.dr_12)
    if Q.psi_0p3 > 0.9973959 and Q.D2 < 3.345339:
        z += 48.16638 * (Q.psi_0p3 - 0.9973959) * (3.345339 - Q.D2)
    if Q.log_sum_pt > 6.910131 and Q.dr_11 < 0.2535773:
        z += -6.771729 * (Q.log_sum_pt - 6.910131) * (0.2535773 - Q.dr_11)
    if Q.log_sum_pt > 6.910131 and Q.absphi_13 < 0.1958008:
        z += -5.416876 * (Q.log_sum_pt - 6.910131) * (0.1958008 - Q.absphi_13)
    if Q.sum_pt_top20 > 1129.275 and Q.pair_mass_0_10 > 1.569656:
        z += 6.056359e-05 * (Q.sum_pt_top20 - 1129.275) * (Q.pair_mass_0_10 - 1.569656)
    if Q.log_sum_pt > 6.910131 and Q.dr0_10 > 0.06373449:
        z += 6.886713 * (Q.log_sum_pt - 6.910131) * (Q.dr0_10 - 0.06373449)
    if Q.z_top30_slots > 0.9048492 and Q.sj3_mass1 > 10.87918:
        z += -0.2428356 * (Q.z_top30_slots - 0.9048492) * (Q.sj3_mass1 - 10.87918)
    if Q.M3 < 0.03787151 and Q.soft2_dr > 0.1573586:
        z += -0.7996017 * (0.03787151 - Q.M3) * (Q.soft2_dr - 0.1573586)
    if Q.z_dr_0p2_0p4 < 0.1937447 and Q.absphi_9 > 0.03289795:
        z += -2.246463 * (0.1937447 - Q.z_dr_0p2_0p4) * (Q.absphi_9 - 0.03289795)
    if Q.mass_top10 > 71.781 and Q.tau43 > 0.7207148:
        z += -0.010387 * (Q.mass_top10 - 71.781) * (Q.tau43 - 0.7207148)
    if Q.n_pt_above_1 > 28.0 and Q.tau43 < 0.942303:
        z += 0.03199094 * (Q.n_pt_above_1 - 28.0) * (0.942303 - Q.tau43)
    if Q.pt_11 < 14.14062 and Q.n_real_top15 < 15.0:
        z += -0.02695549 * (14.14062 - Q.pt_11) * (15.0 - Q.n_real_top15)
    if Q.tau21 < 0.5494307 and Q.soft1_dr > 0.2690904:
        z += 0.324242 * (0.5494307 - Q.tau21) * (Q.soft1_dr - 0.2690904)
    if Q.M3 < 0.03787151 and Q.soft3_dr > 0.1968156:
        z += -7.745073 * (0.03787151 - Q.M3) * (Q.soft3_dr - 0.1968156)
    if Q.log_sum_pt > 6.98945 and Q.soft4_dr > 0.02734277:
        z += 3.088198 * (Q.log_sum_pt - 6.98945) * (Q.soft4_dr - 0.02734277)
    if Q.sum_pt > 907.9372 and Q.soft4_dr > 0.06076665:
        z += -0.00290944 * (Q.sum_pt - 907.9372) * (Q.soft4_dr - 0.06076665)
    if Q.n_dr_0p2_0p4 < 11.0 and Q.orientation_deg > -0.8247956:
        z += 7.030235e-05 * (11.0 - Q.n_dr_0p2_0p4) * (Q.orientation_deg - -0.8247956)
    if Q.mass_top10 > 71.781 and Q.pt_3 < 114.75:
        z += -4.081045e-05 * (Q.mass_top10 - 71.781) * (114.75 - Q.pt_3)
    if Q.sum_pt_top40 > 1069.671 and Q.dr1_12 < 0.07895359:
        z += 0.01282669 * (Q.sum_pt_top40 - 1069.671) * (0.07895359 - Q.dr1_12)
    if Q.z_top50_slots > 0.9906378 and Q.dr1_13 < 0.2696957:
        z += -2.108396 * (Q.z_top50_slots - 0.9906378) * (0.2696957 - Q.dr1_13)
    if Q.sum_pt_top40 > 994.2695 and Q.dr1_12 < 0.3241858:
        z += -0.00324017 * (Q.sum_pt_top40 - 994.2695) * (0.3241858 - Q.dr1_12)
    if Q.mass_over_sum_pt > 0.09046749 and Q.soft7_dr0 < 0.008031476:
        z += 15673.54 * (Q.mass_over_sum_pt - 0.09046749) * (0.008031476 - Q.soft7_dr0)
    if Q.psi_0p3 > 0.9638082 and Q.ptdr0_8 > 4.453578:
        z += -0.5646214 * (Q.psi_0p3 - 0.9638082) * (Q.ptdr0_8 - 4.453578)
    return max(0.0, z)


def neuron_6(Q):
    z = -0.1735495
    if Q.mass_top50 < 71.79516:
        z += -0.04300678 * Q.mass_top50 + 4.242121
    if 71.79516 <= Q.mass_top50 < 168.9698:
        z += -0.01077842 * Q.mass_top50 + 1.928281
    if 168.9698 <= Q.mass_top50 < 172.8:
        z += -0.02794986 * Q.mass_top50 + 4.829737
    if Q.mass < 86.4:
        z += 4.48823e-05 * Q.mass - 1.568756
    if 86.4 <= Q.mass < 92.85979:
        z += 0.02811119 * Q.mass - 3.993684
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.09773934 * Q.mass - 10.45934
    if 101.0497 <= Q.mass < 120.6:
        z += 0.02981072 * Q.mass - 3.595173
    if Q.e3 < 0.0003372339:
        z += -99.55822 * Q.e3 + 0.03357441
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += 0.03248577 * Q.n_dr_0p2_0p4 - 0.4872866
    if Q.girth2_top15 < 0.002197765:
        z += -96.45686 * Q.girth2_top15 + 0.2119895
    if 29.00832 <= Q.sj3_pair_mass_min < 76.60223:
        z += -0.008823439 * Q.sj3_pair_mass_min + 0.2559532
    if Q.sj3_pair_mass_min >= 76.60223:
        z += -0.02826823 * Q.sj3_pair_mass_min + 1.745468
    if Q.e2 < 0.02793599:
        z += -13.66091 * Q.e2 + 0.8912383
    if 0.02793599 <= Q.e2 < 0.04755309:
        z += -32.40358 * Q.e2 + 1.414833
    if 0.04755309 <= Q.e2 < 0.05557149:
        z += -9.020915 * Q.e2 + 0.3029156
    if 0.05557149 <= Q.e2 < 0.06524004:
        z += -18.92714 * Q.e2 + 0.853419
    if Q.e2 >= 0.06524004:
        z += -5.266225 * Q.e2 - 0.03781932
    if Q.sj3_mass1 >= 21.11128:
        z += -0.004105914 * Q.sj3_mass1 + 0.08668108
    if Q.log_sum_pt < 6.811175:
        z += 5.378897 * Q.log_sum_pt - 36.82266
    if 6.811175 <= Q.log_sum_pt < 7.017258:
        z += 0.9027999 * Q.log_sum_pt - 6.33518
    if Q.tau1 < 0.05444509:
        z += -6.155875 * Q.tau1 + 0.3884867
    if 0.05444509 <= Q.tau1 < 0.06310829:
        z += 3.415825 * Q.tau1 - 0.1326454
    if Q.tau1 >= 0.06310829:
        z += 9.5717 * Q.tau1 - 0.5211321
    if Q.M2 < 0.1011005:
        z += 2.74215 * Q.M2 - 0.2772327
    if Q.sj3_dr_min >= 0.1204829:
        z += -1.615879 * Q.sj3_dr_min + 0.1946858
    if Q.z_top40_slots < 0.9300465:
        z += -10.84704 * Q.z_top40_slots + 10.08825
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -1.142559 * Q.z_dr_0p1_0p2 + 0.1374998
    if Q.n_dr_0p1_0p2 >= 33.0:
        z += 0.02296406 * Q.n_dr_0p1_0p2 - 0.7578139
    if Q.mass_over_sum_pt < 0.02682209:
        z += 5.938013 * Q.mass_over_sum_pt - 0.4946332
    if 0.02682209 <= Q.mass_over_sum_pt < 0.08329945:
        z += 22.50334 * Q.mass_over_sum_pt - 0.9389498
    if 0.08329945 <= Q.mass_over_sum_pt < 0.09795415:
        z += 16.56533 * Q.mass_over_sum_pt - 0.4443166
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -21.46087 * Q.mass_over_sum_pt + 3.280507
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -36.84425 * Q.mass_over_sum_pt + 5.099221
    if Q.girth2_top30 < 0.00375223:
        z += -113.3925 * Q.girth2_top30 - 0.1206565
    if 0.00375223 <= Q.girth2_top30 < 0.008376291:
        z += 38.12554 * Q.girth2_top30 - 0.6891872
    if 0.008376291 <= Q.girth2_top30 < 0.01807679:
        z += 83.89356 * Q.girth2_top30 - 1.072554
    if 0.01807679 <= Q.girth2_top30 < 0.02809026:
        z += 45.76802 * Q.girth2_top30 - 0.3833663
    if Q.girth2_top30 >= 0.02809026:
        z += -22.10518 * Q.girth2_top30 + 1.523209
    if Q.girth2_top50 < 0.00242543:
        z += 264.8536 * Q.girth2_top50 - 1.05551
    if 0.00242543 <= Q.girth2_top50 < 0.003418057:
        z += 253.0959 * Q.girth2_top50 - 1.026992
    if 0.003418057 <= Q.girth2_top50 < 0.004573744:
        z += 140.0865 * Q.girth2_top50 - 0.6407198
    if Q.pt_14 >= 10.78125:
        z += -0.002732445 * Q.pt_14 + 0.02945917
    if Q.girth2_top20 >= 0.008031209:
        z += 50.62113 * Q.girth2_top20 - 0.4065489
    if Q.lam1 < 0.003811746:
        z += -327.388 * Q.lam1 + 1.43721
    if 0.003811746 <= Q.lam1 < 0.007259287:
        z += -130.9638 * Q.lam1 + 0.6884907
    if 0.007259287 <= Q.lam1 < 0.02048524:
        z += 19.82568 * Q.lam1 - 0.4061338
    if Q.lam2 < 0.003687605:
        z += -124.6745 * Q.lam2 + 0.4597505
    if Q.mass_top15 < 30.35994:
        z += 0.0005651405 * Q.mass_top15 + 0.3427522
    if 30.35994 <= Q.mass_top15 < 91.19:
        z += -0.005916645 * Q.mass_top15 + 0.5395388
    if Q.LHA >= 0.3719813:
        z += 3.095393 * Q.LHA - 1.151428
    if Q.girth2_top3 < 0.001592178:
        z += 42.92328 * Q.girth2_top3 - 0.06834149
    if Q.pt_6 < 19.46875:
        z += -0.0104675 * Q.pt_6 + 0.2037892
    if Q.n_dr_0_0p05 >= 9.0:
        z += -0.003410607 * Q.n_dr_0_0p05 + 0.03069546
    if Q.C2 < 0.04704395:
        z += -1.952417 * Q.C2 + 0.09184941
    if Q.girth2_top40 >= 0.008031986:
        z += 65.25285 * Q.girth2_top40 - 0.5241099
    if Q.sum_pt_top20 < 1129.275:
        z += 0.00126762 * Q.sum_pt_top20 - 1.431491
    if Q.z_dr_0p2_0p4 < 0.03700182:
        z += -3.74509 * Q.z_dr_0p2_0p4 + 0.1385751
    if Q.sj3_mass2 >= 11.91094:
        z += 0.009300471 * Q.sj3_mass2 - 0.1107773
    if Q.sum_pt_top10 >= 975.0641:
        z += -0.0003510777 * Q.sum_pt_top10 + 0.3423232
    if Q.D2 < 1.409617:
        z += 0.4368472 * Q.D2 - 0.6157873
    if Q.girth >= 0.02085222:
        z += -8.761348 * Q.girth + 0.1826935
    if Q.n_dr_0p05_0p1 >= 21.0:
        z += -0.002422031 * Q.n_dr_0p05_0p1 + 0.05086266
    if Q.sum_pt < 1260.541:
        z += -0.0006987411 * Q.sum_pt + 0.8807917
    if Q.sj2_dr >= 0.2595052:
        z += 1.136943 * Q.sj2_dr - 0.2950426
    if Q.eta_5 < -0.1135254:
        z += 1.509364 * Q.eta_5 + 0.1713511
    if Q.mass < 120.6 and Q.sum_pt < 1007.788:
        z += 8.291361e-05 * (120.6 - Q.mass) * (1007.788 - Q.sum_pt)
    if Q.sj3_pair_mass_min > 29.00832 and Q.psi_0p3 > 0.9896594:
        z += 0.6365916 * (Q.sj3_pair_mass_min - 29.00832) * (Q.psi_0p3 - 0.9896594)
    if Q.sj3_dr_min > 0.1204829 and Q.pt_11 < 33.78125:
        z += 0.06711535 * (Q.sj3_dr_min - 0.1204829) * (33.78125 - Q.pt_11)
    if Q.M2 < 0.1011005 and Q.eta_0 < -0.07794189:
        z += 9.940192 * (0.1011005 - Q.M2) * (-0.07794189 - Q.eta_0)
    if Q.mass_over_sum_pt > 0.1182259 and Q.C2_b2 > 0.02704832:
        z += 258.7458 * (Q.mass_over_sum_pt - 0.1182259) * (Q.C2_b2 - 0.02704832)
    if Q.e2 > 0.05557149 and Q.zdr_0 > 0.001369707:
        z += 407.0715 * (Q.e2 - 0.05557149) * (Q.zdr_0 - 0.001369707)
    if Q.sj3_dr_min > 0.1204829 and Q.sj3_mass2 < 7.863702:
        z += 0.2757131 * (Q.sj3_dr_min - 0.1204829) * (7.863702 - Q.sj3_mass2)
    if Q.e2 > 0.05557149 and Q.z_2 > 0.120204:
        z += 6257.546 * (Q.e2 - 0.05557149) * (Q.z_2 - 0.120204)
    if Q.n_dr_0p2_0p4 > 15.0 and Q.max_pair_mass > 33.3761:
        z += -0.001416258 * (Q.n_dr_0p2_0p4 - 15.0) * (Q.max_pair_mass - 33.3761)
    if Q.e2 > 0.05557149 and Q.D2_b2 < 1.67722:
        z += 36.50794 * (Q.e2 - 0.05557149) * (1.67722 - Q.D2_b2)
    if Q.girth2_top30 > 0.008376291 and Q.D2_b2 > 1.67722:
        z += -26.49956 * (Q.girth2_top30 - 0.008376291) * (Q.D2_b2 - 1.67722)
    if Q.mass_over_sum_pt > 0.1182259 and Q.D2_b2 < 7.36624:
        z += -3.398595 * (Q.mass_over_sum_pt - 0.1182259) * (7.36624 - Q.D2_b2)
    if Q.girth2_top30 > 0.008376291 and Q.z_dr_0p05_0p1 > 0.6469679:
        z += 1379.972 * (Q.girth2_top30 - 0.008376291) * (Q.z_dr_0p05_0p1 - 0.6469679)
    if Q.girth2_top30 > 0.008376291 and Q.n_real_top50 < 43.0:
        z += 12.71589 * (Q.girth2_top30 - 0.008376291) * (43.0 - Q.n_real_top50)
    if Q.e3 < 0.0003372339 and Q.dr_max_012 > 0.004415714:
        z += -2521.432 * (0.0003372339 - Q.e3) * (Q.dr_max_012 - 0.004415714)
    if Q.girth2_top30 > 0.008376291 and Q.orientation_deg > 26.6454:
        z += -0.1341551 * (Q.girth2_top30 - 0.008376291) * (Q.orientation_deg - 26.6454)
    if Q.n_dr_0p1_0p2 > 33.0 and Q.absphi_2 < 0.12854:
        z += -0.06101805 * (Q.n_dr_0p1_0p2 - 33.0) * (0.12854 - Q.absphi_2)
    if Q.girth2_top20 > 0.008031209 and Q.C2_b2 > 0.0008187529:
        z += -991.942 * (Q.girth2_top20 - 0.008031209) * (Q.C2_b2 - 0.0008187529)
    if Q.mass_top15 < 91.19 and Q.C2_b2 < 0.04008677:
        z += -0.05551919 * (91.19 - Q.mass_top15) * (0.04008677 - Q.C2_b2)
    if Q.n_dr_0p1_0p2 > 33.0 and Q.ptdr0_10 > 6.130737:
        z += -0.005795097 * (Q.n_dr_0p1_0p2 - 33.0) * (Q.ptdr0_10 - 6.130737)
    if Q.LHA > 0.3719813 and Q.z_dr_0p05_0p1 > 0.6469679:
        z += -808.4524 * (Q.LHA - 0.3719813) * (Q.z_dr_0p05_0p1 - 0.6469679)
    if Q.mass_over_sum_pt > 0.02682209 and Q.z_dr_0p05_0p1 < 0.7108211:
        z += -3.100476 * (Q.mass_over_sum_pt - 0.02682209) * (0.7108211 - Q.z_dr_0p05_0p1)
    if Q.n_dr_0p1_0p2 > 33.0 and Q.soft9_absphi < 0.03967285:
        z += -1.178889 * (Q.n_dr_0p1_0p2 - 33.0) * (0.03967285 - Q.soft9_absphi)
    if Q.e2 > 0.02793599 and Q.zdr_1 < 0.01718455:
        z += -377.882 * (Q.e2 - 0.02793599) * (0.01718455 - Q.zdr_1)
    if Q.mass_over_sum_pt > 0.02682209 and Q.eta_12 > 0.09442749:
        z += 19.58045 * (Q.mass_over_sum_pt - 0.02682209) * (Q.eta_12 - 0.09442749)
    if Q.mass < 120.6 and Q.mean_eta < -0.003213499:
        z += 15.85185 * (120.6 - Q.mass) * (-0.003213499 - Q.mean_eta)
    if Q.mass_over_sum_pt > 0.09795415 and Q.soft8_z < 0.001160626:
        z += -132.8364 * (Q.mass_over_sum_pt - 0.09795415) * (0.001160626 - Q.soft8_z)
    if Q.sj3_pair_mass_min > 29.00832 and Q.sj3_z1 > 0.7389287:
        z += -0.1275674 * (Q.sj3_pair_mass_min - 29.00832) * (Q.sj3_z1 - 0.7389287)
    if Q.e2 > 0.04755309 and Q.sj3_z1 > 0.7707617:
        z += -4793.212 * (Q.e2 - 0.04755309) * (Q.sj3_z1 - 0.7707617)
    if Q.e2 > 0.04755309 and Q.sj3_mass3 < 0.4118081:
        z += 885.1992 * (Q.e2 - 0.04755309) * (0.4118081 - Q.sj3_mass3)
    if Q.e2 > 0.02793599 and Q.max_dr < 0.4357228:
        z += 20.51389 * (Q.e2 - 0.02793599) * (0.4357228 - Q.max_dr)
    if Q.D2 < 1.409617 and Q.soft9_dr > 0.2400617:
        z += -8.440443 * (1.409617 - Q.D2) * (Q.soft9_dr - 0.2400617)
    if Q.mass_over_sum_pt > 0.1182259 and Q.soft3_dr0 > 0.1966723:
        z += -0.7902316 * (Q.mass_over_sum_pt - 0.1182259) * (Q.soft3_dr0 - 0.1966723)
    if Q.pt_14 > 10.78125 and Q.sj3_mass3 > 0.4118081:
        z += 0.0003528646 * (Q.pt_14 - 10.78125) * (Q.sj3_mass3 - 0.4118081)
    if Q.C2 < 0.04704395 and Q.dr1_8 > 0.2050905:
        z += 177.6618 * (0.04704395 - Q.C2) * (Q.dr1_8 - 0.2050905)
    if Q.lam1 < 0.007259287 and Q.n_dr_0p4_up > 0.0:
        z += 52.85601 * (0.007259287 - Q.lam1) * (Q.n_dr_0p4_up - 0.0)
    if Q.log_sum_pt < 7.017258 and Q.soft10_dr0 > 0.1732831:
        z += -3.719966 * (7.017258 - Q.log_sum_pt) * (Q.soft10_dr0 - 0.1732831)
    if Q.n_dr_0p2_0p4 > 15.0 and Q.soft5_dr0 > 0.3255096:
        z += 0.4326284 * (Q.n_dr_0p2_0p4 - 15.0) * (Q.soft5_dr0 - 0.3255096)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.1034787
    if Q.tau21_b2 < 0.2352054:
        z += -5.939089 * Q.tau21_b2 + 1.396906
    if Q.girth2 < 0.006403325:
        z += 55.91987 * Q.girth2 + 0.1681955
    if 0.006403325 <= Q.girth2 < 0.007877041:
        z += -15.14963 * Q.girth2 + 0.6232766
    if 0.007877041 <= Q.girth2 < 0.008190222:
        z += 128.7062 * Q.girth2 - 0.5098815
    if 0.008190222 <= Q.girth2 < 0.009614971:
        z += -381.9974 * Q.girth2 + 3.672894
    if Q.mass_over_sum_pt < 0.1182259:
        z += -12.22016 * Q.mass_over_sum_pt + 1.44474
    if Q.mass < 78.26182:
        z += 0.03538798 * Q.mass - 3.028935
    if 78.26182 <= Q.mass < 82.85409:
        z += 0.09192169 * Q.mass - 7.453366
    if 82.85409 <= Q.mass < 91.19:
        z += 0.006266266 * Q.mass - 0.3564643
    if 91.19 <= Q.mass < 92.85979:
        z += 0.02720497 * Q.mass - 2.265864
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.03305316 * Q.mass - 2.808926
    if 101.0497 <= Q.mass < 120.6:
        z += -0.02716511 * Q.mass + 3.276112
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.1148456 * Q.n_dr_0p2_0p4 + 0.7654561
    if 6.0 <= Q.n_dr_0p2_0p4 < 9.0:
        z += -0.0254609 * Q.n_dr_0p2_0p4 + 0.2291481
    if 0.9973959 <= Q.psi_0p3 < 0.9980008:
        z += 18.78559 * Q.psi_0p3 - 18.73667
    if 0.9980008 <= Q.psi_0p3 < 0.9995915:
        z += -138.1523 * Q.psi_0p3 + 137.8875
    if Q.psi_0p3 >= 0.9995915:
        z += -349.8408 * Q.psi_0p3 + 349.4895
    if Q.tau21 < 0.1578973:
        z += 1.324415 * Q.tau21 - 0.2971006
    if 0.1578973 <= Q.tau21 < 0.3861957:
        z += 0.3853689 * Q.tau21 - 0.1488278
    if Q.z_dr_0_0p05 < 0.09573228:
        z += -0.7378607 * Q.z_dr_0_0p05 + 0.07063708
    if Q.z_dr_0p05_0p1 >= 0.6469679:
        z += 0.564455 * Q.z_dr_0p05_0p1 - 0.3651843
    if Q.lam2 < 0.000404306:
        z += -406.5361 * Q.lam2 + 0.164365
    if Q.e2_sq < 0.00363788:
        z += 6.355462 * Q.e2_sq - 0.02312041
    if Q.tau1 < 0.07708632:
        z += -0.7069485 * Q.tau1 - 0.4451069
    if 0.07708632 <= Q.tau1 < 0.08786745:
        z += 7.707185 * Q.tau1 - 1.093721
    if 0.08786745 <= Q.tau1 < 0.09591084:
        z += 4.398701 * Q.tau1 - 0.8030134
    if 0.09591084 <= Q.tau1 < 0.1072713:
        z += 33.54899 * Q.tau1 - 3.598842
    if Q.n_dr_0_0p05 < 18.0:
        z += 0.004784291 * Q.n_dr_0_0p05 - 0.08611724
    if Q.lam1 < 0.006189818:
        z += -29.57871 * Q.lam1 + 0.02026599
    if 0.006189818 <= Q.lam1 < 0.008241985:
        z += 189.4529 * Q.lam1 - 1.3355
    if 0.008241985 <= Q.lam1 < 0.01174405:
        z += -64.52424 * Q.lam1 + 0.7577759
    if Q.sd_mass < 45.595:
        z += -0.001195407 * Q.sd_mass - 0.3118425
    if 45.595 <= Q.sd_mass < 69.65633:
        z += -0.008070539 * Q.sd_mass + 0.001629112
    if 69.65633 <= Q.sd_mass < 86.4:
        z += 0.01672245 * Q.sd_mass - 1.725359
    if 86.4 <= Q.sd_mass < 98.05743:
        z += -0.006875132 * Q.sd_mass + 0.3134717
    if Q.sd_mass >= 98.05743:
        z += 0.001652886 * Q.sd_mass - 0.5227639
    if 73.35236 <= Q.mass_top20 < 85.79457:
        z += 0.0147547 * Q.mass_top20 - 1.082292
    if Q.mass_top20 >= 85.79457:
        z += -0.0008456139 * Q.mass_top20 + 0.25613
    if Q.psi_0p2 >= 0.9313699:
        z += -2.040519 * Q.psi_0p2 + 1.900477
    if Q.max_dr < 0.1939977:
        z += 1.171149 * Q.max_dr + 0.03314211
    if 0.1939977 <= Q.max_dr < 0.2982:
        z += -2.49843 * Q.max_dr + 0.7450319
    if Q.sd_rg < 0.1596365:
        z += -0.7450538 * Q.sd_rg + 0.236812
    if 0.1596365 <= Q.sd_rg < 0.1881908:
        z += 5.371555 * Q.sd_rg - 0.739622
    if 0.1881908 <= Q.sd_rg < 0.2639816:
        z += -3.579 * Q.sd_rg + 0.94479
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1937688:
        z += -2.520665 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1937688)
    if Q.tau21_b2 < 0.2352054 and Q.mass_over_sum_pt_sq < 0.007873266:
        z += -791.251 * (0.2352054 - Q.tau21_b2) * (0.007873266 - Q.mass_over_sum_pt_sq)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.sj2_dr < 0.1825048:
        z += 0.1265691 * (6.0 - Q.n_dr_0p2_0p4) * (0.1825048 - Q.sj2_dr)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9638082:
        z += 9.600173 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.e3 < 7.876005e-05:
        z += -1271.837 * (6.0 - Q.n_dr_0p2_0p4) * (7.876005e-05 - Q.e3)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9777125:
        z += 7.617703 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 82.85409 and Q.psi_0p3 < 0.9985421:
        z += -0.001775139 * (82.85409 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9777125:
        z += -16.50796 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9777125:
        z += 8.59865 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.tau21_b2 < 0.2352054 and Q.n_real_top50 > 34.0:
        z += 0.04734691 * (0.2352054 - Q.tau21_b2) * (Q.n_real_top50 - 34.0)
    if Q.psi_0p3 > 0.9980008 and Q.n_dr_0p1_0p2 > 14.0:
        z += 5.959404 * (Q.psi_0p3 - 0.9980008) * (Q.n_dr_0p1_0p2 - 14.0)
    if Q.mass < 91.19 and Q.n_dr_0_0p05 < 6.0:
        z += -0.009378875 * (91.19 - Q.mass) * (6.0 - Q.n_dr_0_0p05)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9980008:
        z += -123.9432 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9980008:
        z += 57.72703 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9980008:
        z += 61.31384 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 1260.541:
        z += -0.007794799 * (0.2352054 - Q.tau21_b2) * (1260.541 - Q.sum_pt)
    if Q.tau21_b2 < 0.2352054 and Q.z_2 < 0.1063277:
        z += -14.51801 * (0.2352054 - Q.tau21_b2) * (0.1063277 - Q.z_2)
    if Q.mass < 92.85979 and Q.psi_0p3 > 0.9638082:
        z += -8.411594 * (92.85979 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.tau21_b2 < 0.2352054 and Q.orientation_deg > -9.840088:
        z += -0.0151543 * (0.2352054 - Q.tau21_b2) * (Q.orientation_deg - -9.840088)
    if Q.girth2 < 0.006403325 and Q.psi_0p3 > 0.9980008:
        z += 41362.34 * (0.006403325 - Q.girth2) * (Q.psi_0p3 - 0.9980008)
    if Q.z_dr_0_0p05 < 0.09573228 and Q.sj3_mass3 < 5.541989:
        z += 0.0925727 * (0.09573228 - Q.z_dr_0_0p05) * (5.541989 - Q.sj3_mass3)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.n_dr_0p1_0p2 > 15.0:
        z += 0.0008612423 * (6.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 15.0)
    if Q.tau21_b2 < 0.2352054 and Q.psi_0p3 > 0.9299135:
        z += -26.24106 * (0.2352054 - Q.tau21_b2) * (Q.psi_0p3 - 0.9299135)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.max_dr > 0.3618493:
        z += -0.2943555 * (6.0 - Q.n_dr_0p2_0p4) * (Q.max_dr - 0.3618493)
    if Q.mass < 92.85979 and Q.z_dr_0_0p05 < 0.4947602:
        z += -0.08666059 * (92.85979 - Q.mass) * (0.4947602 - Q.z_dr_0_0p05)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.eta_1 > 0.0001021922:
        z += -0.08030391 * (6.0 - Q.n_dr_0p2_0p4) * (Q.eta_1 - 0.0001021922)
    if Q.lam1 < 0.008241985 and Q.zdr_0 > 0.01564747:
        z += -22235.91 * (0.008241985 - Q.lam1) * (Q.zdr_0 - 0.01564747)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.ptdr0_2 > 11.87288:
        z += 0.003568255 * (6.0 - Q.n_dr_0p2_0p4) * (Q.ptdr0_2 - 11.87288)
    if Q.psi_0p3 > 0.9973959 and Q.z_top50_slots > 0.978741:
        z += 3241.02 * (Q.psi_0p3 - 0.9973959) * (Q.z_top50_slots - 0.978741)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.soft6_z < 0.004060732:
        z += 5.221534 * (6.0 - Q.n_dr_0p2_0p4) * (0.004060732 - Q.soft6_z)
    if Q.tau21_b2 < 0.2352054 and Q.zdr_3 > 0.00396157:
        z += -8.870001 * (0.2352054 - Q.tau21_b2) * (Q.zdr_3 - 0.00396157)
    if Q.tau21 < 0.3861957 and Q.pt1_dr01 < 28.39396:
        z += -0.01193265 * (0.3861957 - Q.tau21) * (28.39396 - Q.pt1_dr01)
    if Q.e2_sq < 0.00363788 and Q.ptdr0_4 > 12.44629:
        z += -2817.52 * (0.00363788 - Q.e2_sq) * (Q.ptdr0_4 - 12.44629)
    if Q.tau1 < 0.07708632 and Q.ptdr0_8 > 4.453578:
        z += -0.1917032 * (0.07708632 - Q.tau1) * (Q.ptdr0_8 - 4.453578)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.ptdr0_4 > 10.58726:
        z += 0.005678004 * (6.0 - Q.n_dr_0p2_0p4) * (Q.ptdr0_4 - 10.58726)
    if Q.psi_0p2 > 0.9313699 and Q.n_pt_above_50 < 6.0:
        z += 0.7447602 * (Q.psi_0p2 - 0.9313699) * (6.0 - Q.n_pt_above_50)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.phi_6 < -0.07593384:
        z += 0.4588341 * (6.0 - Q.n_dr_0p2_0p4) * (-0.07593384 - Q.phi_6)
    if Q.mass_top20 > 73.35236 and Q.pt_9 < 10.10938:
        z += -0.002700834 * (Q.mass_top20 - 73.35236) * (10.10938 - Q.pt_9)
    if Q.tau21 < 0.3861957 and Q.z_9 < 0.009867229:
        z += 451.0593 * (0.3861957 - Q.tau21) * (0.009867229 - Q.z_9)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.eta_12 < -0.03771973:
        z += 0.3893769 * (6.0 - Q.n_dr_0p2_0p4) * (-0.03771973 - Q.eta_12)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.049742
    if 0.07696632 <= Q.mass_over_sum_pt < 0.1182259:
        z += -13.42286 * Q.mass_over_sum_pt + 1.033108
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -31.62661 * Q.mass_over_sum_pt + 3.185263
    if Q.sum_pt_top40 < 1001.523:
        z += 0.001762732 * Q.sum_pt_top40 - 1.976375
    if 1001.523 <= Q.sum_pt_top40 < 1024.942:
        z += 0.009007867 * Q.sum_pt_top40 - 9.232545
    if Q.girth2_top40 < 0.005196966:
        z += -123.3395 * Q.girth2_top40 + 1.090387
    if 0.005196966 <= Q.girth2_top40 < 0.008840538:
        z += -52.19116 * Q.girth2_top40 + 0.720632
    if Q.girth2_top40 >= 0.008840538:
        z += 71.14832 * Q.girth2_top40 - 0.3697554
    if Q.n_dr_0p2_0p4 < 3.0:
        z += -0.004669413 * Q.n_dr_0p2_0p4 - 0.1735769
    if 3.0 <= Q.n_dr_0p2_0p4 < 9.0:
        z += 0.0223899 * Q.n_dr_0p2_0p4 - 0.2547548
    if 9.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += 0.06299291 * Q.n_dr_0p2_0p4 - 0.6201819
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += 0.02705932 * Q.n_dr_0p2_0p4 - 0.08117795
    if 64.48544 <= Q.mass < 74.25181:
        z += 0.02151377 * Q.mass - 1.387325
    if 74.25181 <= Q.mass < 87.36377:
        z += 0.05306403 * Q.mass - 3.729989
    if 87.36377 <= Q.mass < 101.0497:
        z += 0.1089262 * Q.mass - 8.61032
    if 101.0497 <= Q.mass < 125.1:
        z += 0.0645282 * Q.mass - 4.123913
    if Q.mass >= 125.1:
        z += 0.0144645 * Q.mass + 2.139056
    if 38.0 <= Q.n_particles < 51.0:
        z += 0.006152237 * Q.n_particles - 0.233785
    if Q.n_particles >= 51.0:
        z += 0.0100386 * Q.n_particles - 0.4319895
    if Q.sum_pt < 1017.435:
        z += -0.03192783 * Q.sum_pt + 32.68848
    if 1017.435 <= Q.sum_pt < 1028.184:
        z += -0.01897806 * Q.sum_pt + 19.51294
    if 0.006043209 <= Q.girth2_top20 < 0.008031209:
        z += -125.2469 * Q.girth2_top20 + 0.756893
    if Q.girth2_top20 >= 0.008031209:
        z += 71.12718 * Q.girth2_top20 - 0.8202281
    if Q.log_sum_pt < 6.930088:
        z += 9.82997 * Q.log_sum_pt - 68.12256
    if Q.log_sum_pt >= 7.139296:
        z += -0.5809211 * Q.log_sum_pt + 4.147368
    if Q.girth2_top30 < 0.007463985:
        z += -163.7515 * Q.girth2_top30 + 1.222239
    if Q.width < 0.009614971:
        z += 566.7755 * Q.width - 5.44953
    if Q.girth2 < 0.007877041:
        z += -275.3794 * Q.girth2 + 2.169175
    if Q.sj2_dr >= 0.2232169:
        z += 2.276485 * Q.sj2_dr - 0.5081499
    if Q.z_dr_0p1_0p2 >= 0.3340477:
        z += 1.21098 * Q.z_dr_0p1_0p2 - 0.4045251
    if Q.n_for_90pct < 7.0:
        z += 0.02629003 * Q.n_for_90pct - 1.025311
    if 7.0 <= Q.n_for_90pct < 39.0:
        z += -0.01337229 * Q.n_for_90pct - 0.7476748
    if Q.n_for_90pct >= 39.0:
        z += -0.03966232 * Q.n_for_90pct + 0.2776362
    if Q.psi_0p3 >= 0.9943058:
        z += -47.87234 * Q.psi_0p3 + 47.59974
    if 0.04755309 <= Q.e2 < 0.06524004:
        z += 33.46127 * Q.e2 - 1.591186
    if Q.e2 >= 0.06524004:
        z += 35.27241 * Q.e2 - 1.709345
    if Q.lam1 < 0.007671243:
        z += -61.59925 * Q.lam1 + 0.3379024
    if 0.007671243 <= Q.lam1 < 0.01174405:
        z += 33.05839 * Q.lam1 - 0.3882394
    if Q.tau2 < 0.0795038:
        z += -1.925593 * Q.tau2 + 0.153092
    if Q.C2 >= 0.06655881:
        z += 6.452837 * Q.C2 - 0.4294931
    if 5.13841e-05 <= Q.e3 < 0.0003372339:
        z += -1962.338 * Q.e3 + 0.100833
    if Q.e3 >= 0.0003372339:
        z += -392.5111 * Q.e3 - 0.428566
    if 82.04491 <= Q.mass_top50 < 117.0487:
        z += -0.03652709 * Q.mass_top50 + 2.996862
    if Q.mass_top50 >= 117.0487:
        z += 0.001199424 * Q.mass_top50 - 1.418979
    if Q.mass_top20 >= 119.2969:
        z += -0.01485904 * Q.mass_top20 + 1.772637
    if Q.n_dr_0p1_0p2 >= 19.0:
        z += 0.01150168 * Q.n_dr_0p1_0p2 - 0.218532
    if Q.z_top15_slots < 0.8316924:
        z += 0.05914902 * Q.z_top15_slots - 0.0491938
    if Q.eta_0 >= -0.01382675:
        z += 0.9290609 * Q.eta_0 + 0.01284589
    if Q.psi_0p1 < 0.3628388:
        z += 1.016856 * Q.psi_0p1 - 0.368955
    if Q.z_top50_slots >= 0.9704436:
        z += -16.51646 * Q.z_top50_slots + 16.0283
    if Q.soft5_pt < 1.219727:
        z += -0.2458086 * Q.soft5_pt + 0.2998193
    if Q.girth2_top15 < 0.02146578:
        z += 26.08413 * Q.girth2_top15 - 0.5599162
    if Q.sum_pt_top30 >= 978.0762:
        z += 0.002426375 * Q.sum_pt_top30 - 2.37318
    if Q.sum_pt_top50 >= 889.8503:
        z += -0.002231416 * Q.sum_pt_top50 + 1.985626
    if Q.mass_over_sum_pt > 0.07696632 and Q.sum_pt < 1115.723:
        z += 0.2829597 * (Q.mass_over_sum_pt - 0.07696632) * (1115.723 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.z_dr_0p1_0p2 > 0.3340477:
        z += 0.144072 * (15.0 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p1_0p2 - 0.3340477)
    if Q.girth2_top40 > 0.005196966 and Q.log_sum_pt < 7.017258:
        z += -681.1549 * (Q.girth2_top40 - 0.005196966) * (7.017258 - Q.log_sum_pt)
    if Q.n_particles > 51.0 and Q.soft1_pt < 1.091797:
        z += -0.009053229 * (Q.n_particles - 51.0) * (1.091797 - Q.soft1_pt)
    if Q.sum_pt < 1017.435 and Q.n_for_90pct < 13.0:
        z += 0.0003497703 * (1017.435 - Q.sum_pt) * (13.0 - Q.n_for_90pct)
    if Q.mass > 64.48544 and Q.zdr_0 > 0.0008718296:
        z += -0.02036446 * (Q.mass - 64.48544) * (Q.zdr_0 - 0.0008718296)
    if Q.girth2_top30 < 0.007463985 and Q.sj2_dr > 0.1512157:
        z += -707.0408 * (0.007463985 - Q.girth2_top30) * (Q.sj2_dr - 0.1512157)
    if Q.tau2 < 0.0795038 and Q.sj3_mass1 > 13.38202:
        z += 0.4575495 * (0.0795038 - Q.tau2) * (Q.sj3_mass1 - 13.38202)
    if Q.sum_pt < 1017.435 and Q.dr_max_012 > 0.1828389:
        z += -0.1911767 * (1017.435 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    if Q.sum_pt < 1028.184 and Q.dr_max_012 > 0.1828389:
        z += 0.1675988 * (1028.184 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    if Q.girth2_top20 > 0.006043209 and Q.sj3_z2 < 0.08173536:
        z += 15746.75 * (Q.girth2_top20 - 0.006043209) * (0.08173536 - Q.sj3_z2)
    if Q.log_sum_pt < 6.930088 and Q.dr01 < 0.05595395:
        z += 32.77618 * (6.930088 - Q.log_sum_pt) * (0.05595395 - Q.dr01)
    if Q.mass_top20 > 119.2969 and Q.absphi_1 > 0.07141113:
        z += -0.1072599 * (Q.mass_top20 - 119.2969) * (Q.absphi_1 - 0.07141113)
    if Q.sj2_dr > 0.2232169 and Q.mean_phi > 2.298159e-05:
        z += -263.4621 * (Q.sj2_dr - 0.2232169) * (Q.mean_phi - 2.298159e-05)
    if Q.sum_pt_top50 > 889.8503 and Q.eta_9 < -0.1315918:
        z += 0.01964821 * (Q.sum_pt_top50 - 889.8503) * (-0.1315918 - Q.eta_9)
    if Q.sum_pt < 1017.435 and Q.dr1_9 > 0.1651809:
        z += -0.01190571 * (1017.435 - Q.sum_pt) * (Q.dr1_9 - 0.1651809)
    if Q.mass > 74.25181 and Q.sj3_mass2 < 5.253859:
        z += -0.003089795 * (Q.mass - 74.25181) * (5.253859 - Q.sj3_mass2)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.9071266
    if Q.mass_top40 < 80.89043:
        z += -0.005792819 * Q.mass_top40 - 1.245158
    if 80.89043 <= Q.mass_top40 < 125.1:
        z += 0.01866163 * Q.mass_top40 - 3.223289
    if 125.1 <= Q.mass_top40 < 163.2541:
        z += 0.02329287 * Q.mass_top40 - 3.802657
    if Q.sum_pt_top40 < 956.2133:
        z += -0.008304287 * Q.sum_pt_top40 + 7.940669
    if Q.girth2_top40 < 0.00217213:
        z += -12.99818 * Q.girth2_top40 + 0.6975761
    if 0.00217213 <= Q.girth2_top40 < 0.006259772:
        z += -163.7478 * Q.girth2_top40 + 1.025024
    if Q.z_dr_0p1_0p2 >= 0.6882177:
        z += -2.281777 * Q.z_dr_0p1_0p2 + 1.57036
    if Q.mass_over_sum_pt >= 0.1708801:
        z += -4.131106 * Q.mass_over_sum_pt + 0.7059239
    if Q.mass < 64.48544:
        z += -0.0595363 * Q.mass + 6.44656
    if 64.48544 <= Q.mass < 82.85409:
        z += -0.09178703 * Q.mass + 8.526262
    if 82.85409 <= Q.mass < 92.85979:
        z += -0.03520067 * Q.mass + 3.837851
    if 92.85979 <= Q.mass < 143.7876:
        z += 0.03365689 * Q.mass - 2.556248
    if 143.7876 <= Q.mass < 160.8:
        z += -0.06033931 * Q.mass + 10.95924
    if 160.8 <= Q.mass < 162.8363:
        z += -0.2143603 * Q.mass + 35.72582
    if 162.8363 <= Q.mass < 172.8:
        z += -0.08231597 * Q.mass + 14.2242
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -1.376599 * Q.z_dr_0_0p05 + 1.209901
    if 0.09749958 <= Q.girth < 0.1402186:
        z += 19.33574 * Q.girth - 1.885227
    if Q.girth >= 0.1402186:
        z += -38.59404 * Q.girth + 6.237606
    if Q.z_top5 >= 0.7963975:
        z += 6.10819 * Q.z_top5 - 4.864547
    if Q.psi_0p3 >= 0.9943058:
        z += 42.35382 * Q.psi_0p3 - 42.11264
    if Q.tau1 < 0.05444509:
        z += 15.00651 * Q.tau1 - 0.8170309
    if Q.lam1 < 0.004673423:
        z += -167.9505 * Q.lam1 + 0.7849038
    if Q.lam1 >= 0.01174405:
        z += 25.7333 * Q.lam1 - 0.3022132
    if Q.girth2_top30 >= 0.0008564881:
        z += -26.76762 * Q.girth2_top30 + 0.02292615
    if Q.D2 >= 2.178951:
        z += 0.03788818 * Q.D2 - 0.08255648
    if Q.mass_top50 < 92.16545:
        z += -0.007802017 * Q.mass_top50 + 3.07905
    if 92.16545 <= Q.mass_top50 < 138.8977:
        z += -0.05049991 * Q.mass_top50 + 7.01432
    if Q.mass_top30 < 73.33139:
        z += -0.01431958 * Q.mass_top30 + 1.050075
    if Q.n_for_90pct >= 35.0:
        z += 0.03226546 * Q.n_for_90pct - 1.129291
    if Q.LHA >= 0.404204:
        z += 19.13058 * Q.LHA - 7.732657
    if Q.sj3_dr13 >= 0.1654269:
        z += 0.2530802 * Q.sj3_dr13 - 0.04186628
    if Q.z_top40_slots >= 0.9574183:
        z += -2.312153 * Q.z_top40_slots + 2.213698
    if Q.e3 >= 0.0003372339:
        z += 233.1185 * Q.e3 - 0.07861547
    if Q.sum_pt < 986.0565:
        z += -0.01742178 * Q.sum_pt + 17.17886
    if Q.log_sum_pt < 6.903423:
        z += 11.72852 * Q.log_sum_pt - 80.96696
    if Q.sum_pt_top40 < 956.2133 and Q.psi_0p1 > 0.9184255:
        z += -0.0142383 * (956.2133 - Q.sum_pt_top40) * (Q.psi_0p1 - 0.9184255)
    if Q.sum_pt_top40 < 956.2133 and Q.soft4_pt > 1.789258:
        z += -0.004618275 * (956.2133 - Q.sum_pt_top40) * (Q.soft4_pt - 1.789258)
    if Q.mass_top40 < 80.89043 and Q.sum_pt < 1034.834:
        z += -0.0001755509 * (80.89043 - Q.mass_top40) * (1034.834 - Q.sum_pt)
    if Q.mass_top40 < 80.89043 and Q.sum_pt_top40 < 906.6023:
        z += 0.000249994 * (80.89043 - Q.mass_top40) * (906.6023 - Q.sum_pt_top40)
    if Q.sum_pt_top40 < 956.2133 and Q.soft3_pt > 2.873047:
        z += 0.0008639836 * (956.2133 - Q.sum_pt_top40) * (Q.soft3_pt - 2.873047)
    if Q.mass < 92.85979 and Q.z_top50_slots > 0.9995789:
        z += -4.750073 * (92.85979 - Q.mass) * (Q.z_top50_slots - 0.9995789)
    if Q.mass_top40 < 125.1 and Q.n_dr_0p2_0p4 > 9.0:
        z += 0.001456834 * (125.1 - Q.mass_top40) * (Q.n_dr_0p2_0p4 - 9.0)
    if Q.sum_pt_top40 < 956.2133 and Q.lam2 < 0.001776308:
        z += -7.488486 * (956.2133 - Q.sum_pt_top40) * (0.001776308 - Q.lam2)
    if Q.z_dr_0p1_0p2 > 0.6882177 and Q.pt_2 < 154.25:
        z += 0.001206381 * (Q.z_dr_0p1_0p2 - 0.6882177) * (154.25 - Q.pt_2)
    if Q.sum_pt < 986.0565 and Q.soft5_z > 0.002036949:
        z += 4.196485 * (986.0565 - Q.sum_pt) * (Q.soft5_z - 0.002036949)
    return max(0.0, z)


def neuron_10(Q):
    z = 0.8892238
    if Q.girth < 0.1207452:
        z += 42.4678 * Q.girth - 5.127782
    if Q.mass < 64.48544:
        z += -0.03484099 * Q.mass + 2.710391
    if 64.48544 <= Q.mass < 78.26182:
        z += -0.07385937 * Q.mass + 5.226509
    if 78.26182 <= Q.mass < 86.4:
        z += -0.01070433 * Q.mass + 0.2838806
    if 86.4 <= Q.mass < 89.74183:
        z += 0.00355259 * Q.mass - 0.947917
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.05673859 * Q.mass - 5.720926
    if 101.0497 <= Q.mass < 143.7876:
        z += 0.02413666 * Q.mass - 2.426511
    if 143.7876 <= Q.mass < 162.8363:
        z += -0.05768833 * Q.mass + 9.33891
    if Q.mass >= 162.8363:
        z += -0.1659913 * Q.mass + 26.97457
    if Q.girth2_top30 < 0.005402331:
        z += 73.96577 * Q.girth2_top30 - 0.3995876
    if Q.sum_pt_top20 < 750.7313:
        z += 0.003233701 * Q.sum_pt_top20 - 2.42764
    if 136.785 <= Q.mass_top50 < 160.8:
        z += 0.059457 * Q.mass_top50 - 8.132826
    if 160.8 <= Q.mass_top50 < 172.8:
        z += 0.0639685 * Q.mass_top50 - 8.858275
    if Q.mass_top50 >= 172.8:
        z += 0.1281425 * Q.mass_top50 - 19.94754
    if Q.sj2_mass1 < 65.20727:
        z += 0.01293027 * Q.sj2_mass1 - 0.8431478
    if Q.D2 < 2.975532:
        z += -0.1659528 * Q.D2 + 0.4937979
    if 0.9980008 <= Q.psi_0p3 < 0.9985421:
        z += -228.7787 * Q.psi_0p3 + 228.3214
    if Q.psi_0p3 >= 0.9985421:
        z += -167.2131 * Q.psi_0p3 + 166.8455
    if Q.z_top2_slots < 0.5760704:
        z += -0.5245637 * Q.z_top2_slots + 0.3021856
    if Q.mass_top5 < 22.18342:
        z += 0.006488358 * Q.mass_top5 - 0.3854589
    if 22.18342 <= Q.mass_top5 < 59.40777:
        z += 0.0179956 * Q.mass_top5 - 0.6407289
    if Q.mass_top5 >= 59.40777:
        z += 0.01150724 * Q.mass_top5 - 0.25527
    if Q.sum_pt_top50 < 959.0957:
        z += 0.002518489 * Q.sum_pt_top50 - 2.415472
    if Q.girth2 < 0.002575211:
        z += 151.4747 * Q.girth2 - 0.3900794
    if Q.dr_0 < 0.06413297:
        z += -3.625765 * Q.dr_0 + 0.2325311
    if Q.z_dr_0_0p05 >= 0.7674734:
        z += -2.631435 * Q.z_dr_0_0p05 + 2.019557
    if Q.girth2_top10 < 0.01976735:
        z += 18.21479 * Q.girth2_top10 - 0.3600582
    if Q.e2 < 0.01256572:
        z += 21.72543 * Q.e2 - 1.477347
    if 0.01256572 <= Q.e2 < 0.01541561:
        z += 65.619 * Q.e2 - 2.028902
    if 0.01541561 <= Q.e2 < 0.03263075:
        z += 84.44777 * Q.e2 - 2.319159
    if 0.03263075 <= Q.e2 < 0.04358622:
        z += 89.48051 * Q.e2 - 2.483381
    if Q.e2 >= 0.04358622:
        z += 48.92631 * Q.e2 - 0.7157764
    if Q.sum_pt_top10 >= 943.6922:
        z += -0.005140377 * Q.sum_pt_top10 + 4.850934
    if Q.n_particles >= 41.0:
        z += 0.003980909 * Q.n_particles - 0.1632173
    if Q.mass_over_sum_pt >= 0.1606361:
        z += -82.2776 * Q.mass_over_sum_pt + 13.21675
    if Q.e3 < 0.0001086251:
        z += -3558.346 * Q.e3 + 0.3865257
    if Q.zdr_1 < 0.008824206:
        z += -31.29164 * Q.zdr_1 + 0.2761239
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.0505887 * Q.n_dr_0p2_0p4 - 0.5564757
    if Q.sum_pt_top15 < 935.1043:
        z += -0.002718952 * Q.sum_pt_top15 + 2.542504
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -4.072439 * Q.z_dr_0p2_0p4 + 0.371511
    if Q.mass_top15 < 72.18744:
        z += 0.006463596 * Q.mass_top15 - 0.4665904
    if Q.girth2_top20 < 0.005312783:
        z += 45.42336 * Q.girth2_top20 + 0.5265331
    if 0.005312783 <= Q.girth2_top20 < 0.01655983:
        z += -68.2719 * Q.girth2_top20 + 1.130571
    if Q.sj3_dr_min >= 0.1204829:
        z += 2.521715 * Q.sj3_dr_min - 0.3038236
    if 78.53034 <= Q.mass_top30 < 89.17293:
        z += -0.02205435 * Q.mass_top30 + 1.731936
    if Q.mass_top30 >= 89.17293:
        z += -0.00478874 * Q.mass_top30 + 0.1923104
    if Q.z_2nd < 0.09021906:
        z += 4.722215 * Q.z_2nd - 0.4260338
    if Q.tau1 < 0.04466492:
        z += 5.283794 * Q.tau1 + 2.509642
    if 0.04466492 <= Q.tau1 < 0.1507173:
        z += -25.8895 * Q.tau1 + 3.901995
    if Q.M2 >= 0.06621477:
        z += -4.237118 * Q.M2 + 0.2805598
    if Q.pt_dispersion >= 0.27462:
        z += -1.935763 * Q.pt_dispersion + 0.5315994
    if Q.zdr_2 < 0.007043525:
        z += -16.19986 * Q.zdr_2 + 0.1141041
    if Q.girth2_top3 < 0.006756161:
        z += 41.709 * Q.girth2_top3 - 0.2817927
    if Q.zdr_0 < 0.02078982:
        z += -13.13961 * Q.zdr_0 + 0.27317
    if Q.tau2 < 0.04828819:
        z += -8.469563 * Q.tau2 + 0.4089799
    if Q.sum_pt < 986.0565:
        z += 0.001163928 * Q.sum_pt - 1.147698
    if Q.girth2_top40 >= 0.02497133:
        z += 98.00572 * Q.girth2_top40 - 2.447333
    if Q.mass_top10 >= 71.781:
        z += -0.01031159 * Q.mass_top10 + 0.7401762
    if Q.max_dr < 0.2404747:
        z += -0.6723495 * Q.max_dr + 0.161683
    if Q.girth2_top30 < 0.005402331 and Q.sum_pt_top5 < 631.275:
        z += 0.1107559 * (0.005402331 - Q.girth2_top30) * (631.275 - Q.sum_pt_top5)
    if Q.mass_top50 > 160.8 and Q.pt_7 < 53.4375:
        z += -0.0007524513 * (Q.mass_top50 - 160.8) * (53.4375 - Q.pt_7)
    if Q.mass_top50 > 160.8 and Q.soft4_z > 0.001721109:
        z += -16.16622 * (Q.mass_top50 - 160.8) * (Q.soft4_z - 0.001721109)
    if Q.z_top2_slots < 0.5760704 and Q.soft9_z < 0.002970691:
        z += 544.3573 * (0.5760704 - Q.z_top2_slots) * (0.002970691 - Q.soft9_z)
    if Q.D2 < 2.975532 and Q.sj2_dr > 0.2070855:
        z += -1.520561 * (2.975532 - Q.D2) * (Q.sj2_dr - 0.2070855)
    if Q.sum_pt_top50 < 959.0957 and Q.absphi_4 > 0.004672432:
        z += -0.007755698 * (959.0957 - Q.sum_pt_top50) * (Q.absphi_4 - 0.004672432)
    if Q.e2 > 0.03263075 and Q.soft2_pt > 1.413232:
        z += -8.875598 * (Q.e2 - 0.03263075) * (Q.soft2_pt - 1.413232)
    if Q.girth2_top30 < 0.005402331 and Q.psi_0p3 > 0.9985421:
        z += 34495.94 * (0.005402331 - Q.girth2_top30) * (Q.psi_0p3 - 0.9985421)
    if Q.girth2_top10 < 0.01976735 and Q.psi_0p3 > 0.9985421:
        z += -15817.62 * (0.01976735 - Q.girth2_top10) * (Q.psi_0p3 - 0.9985421)
    if Q.mass < 101.0497 and Q.pt1_dr01 > 5.351077:
        z += 0.001315406 * (101.0497 - Q.mass) * (Q.pt1_dr01 - 5.351077)
    if Q.sum_pt_top10 > 943.6922 and Q.z_dr_0p05_0p1 < 0.6469679:
        z += 0.00130837 * (Q.sum_pt_top10 - 943.6922) * (0.6469679 - Q.z_dr_0p05_0p1)
    if Q.girth2_top10 < 0.01976735 and Q.max_dr < 0.4357228:
        z += -157.8624 * (0.01976735 - Q.girth2_top10) * (0.4357228 - Q.max_dr)
    if Q.mass_top5 < 59.40777 and Q.z_dr_0p05_0p1 < 0.8509215:
        z += 0.006309612 * (59.40777 - Q.mass_top5) * (0.8509215 - Q.z_dr_0p05_0p1)
    if Q.D2 < 2.975532 and Q.dr_12 > 0.1269782:
        z += -0.206155 * (2.975532 - Q.D2) * (Q.dr_12 - 0.1269782)
    if Q.mass_over_sum_pt > 0.1606361 and Q.z_3 < 0.09696199:
        z += 335.2513 * (Q.mass_over_sum_pt - 0.1606361) * (0.09696199 - Q.z_3)
    if Q.mass_top50 > 136.785 and Q.soft5_z > 0.001594761:
        z += -33.9423 * (Q.mass_top50 - 136.785) * (Q.soft5_z - 0.001594761)
    if Q.mass > 162.8363 and Q.soft5_z > 0.001434897:
        z += 48.49611 * (Q.mass - 162.8363) * (Q.soft5_z - 0.001434897)
    if Q.girth2_top10 < 0.01976735 and Q.tau32 < 0.8476928:
        z += -10.9641 * (0.01976735 - Q.girth2_top10) * (0.8476928 - Q.tau32)
    if Q.psi_0p3 > 0.9985421 and Q.soft5_z > 0.001434897:
        z += 35768.46 * (Q.psi_0p3 - 0.9985421) * (Q.soft5_z - 0.001434897)
    if Q.sum_pt_top50 < 959.0957 and Q.dr0_3 < 0.2748443:
        z += -0.005974923 * (959.0957 - Q.sum_pt_top50) * (0.2748443 - Q.dr0_3)
    if Q.e2 > 0.03263075 and Q.tau43 < 0.842732:
        z += 26.40649 * (Q.e2 - 0.03263075) * (0.842732 - Q.tau43)
    if Q.D2 < 2.975532 and Q.absphi_4 < 0.08728027:
        z += 1.627761 * (2.975532 - Q.D2) * (0.08728027 - Q.absphi_4)
    if Q.sum_pt_top10 > 943.6922 and Q.eta_0 < -0.07794189:
        z += -0.1977281 * (Q.sum_pt_top10 - 943.6922) * (-0.07794189 - Q.eta_0)
    if Q.mass > 78.26182 and Q.psi_0p3 > 0.9853273:
        z += -0.004613803 * (Q.mass - 78.26182) * (Q.psi_0p3 - 0.9853273)
    if Q.mass_top50 > 172.8 and Q.M2 > 0.04577853:
        z += -0.5478886 * (Q.mass_top50 - 172.8) * (Q.M2 - 0.04577853)
    if Q.sj2_mass1 < 65.20727 and Q.sj2_mass2 > 7.769597:
        z += 0.0001284105 * (65.20727 - Q.sj2_mass1) * (Q.sj2_mass2 - 7.769597)
    if Q.e2 < 0.01541561 and Q.pt2_over_pt0 < 0.7904923:
        z += 12.05206 * (0.01541561 - Q.e2) * (0.7904923 - Q.pt2_over_pt0)
    if Q.pt_dispersion > 0.27462 and Q.n_real_top30 < 30.0:
        z += 0.0174973 * (Q.pt_dispersion - 0.27462) * (30.0 - Q.n_real_top30)
    if Q.mass_top50 > 172.8 and Q.tau43 > 0.9187998:
        z += -0.3566129 * (Q.mass_top50 - 172.8) * (Q.tau43 - 0.9187998)
    if Q.sum_pt_top50 < 959.0957 and Q.dr_9 < 0.2384186:
        z += 0.01480575 * (959.0957 - Q.sum_pt_top50) * (0.2384186 - Q.dr_9)
    if Q.mass_top50 > 172.8 and Q.soft2_abseta > 0.3344727:
        z += -1.315108 * (Q.mass_top50 - 172.8) * (Q.soft2_abseta - 0.3344727)
    if Q.mass_top5 < 59.40777 and Q.zdr_5 > 0.005134657:
        z += 0.8828262 * (59.40777 - Q.mass_top5) * (Q.zdr_5 - 0.005134657)
    if Q.sum_pt_top50 < 959.0957 and Q.dr1_5 < 0.2277069:
        z += 0.001741221 * (959.0957 - Q.sum_pt_top50) * (0.2277069 - Q.dr1_5)
    if Q.D2 < 2.975532 and Q.soft2_dr > 0.2302212:
        z += 0.6938856 * (2.975532 - Q.D2) * (Q.soft2_dr - 0.2302212)
    if Q.mass > 78.26182 and Q.lam2 < 0.0006154841:
        z += -47.02182 * (Q.mass - 78.26182) * (0.0006154841 - Q.lam2)
    if Q.mass_top50 > 136.785 and Q.ptdr0_7 < 0.1938641:
        z += 0.1901134 * (Q.mass_top50 - 136.785) * (0.1938641 - Q.ptdr0_7)
    if Q.mass_over_sum_pt > 0.1606361 and Q.sj3_mass2 > 4.756459:
        z += 0.2094995 * (Q.mass_over_sum_pt - 0.1606361) * (Q.sj3_mass2 - 4.756459)
    if Q.sum_pt < 986.0565 and Q.dr_3 > 0.01436663:
        z += -0.01059444 * (986.0565 - Q.sum_pt) * (Q.dr_3 - 0.01436663)
    if Q.z_top2_slots < 0.5760704 and Q.absphi_13 > 0.1468506:
        z += -3.942055 * (0.5760704 - Q.z_top2_slots) * (Q.absphi_13 - 0.1468506)
    if Q.mass_top5 > 22.18342 and Q.lam2 > 0.0009731947:
        z += 1.600823 * (Q.mass_top5 - 22.18342) * (Q.lam2 - 0.0009731947)
    if Q.M2 > 0.06621477 and Q.sj3_dr13 > 0.1348442:
        z += -7.175539 * (Q.M2 - 0.06621477) * (Q.sj3_dr13 - 0.1348442)
    if Q.z_top2_slots < 0.5760704 and Q.dr0_12 > 0.2665639:
        z += -2.1314 * (0.5760704 - Q.z_top2_slots) * (Q.dr0_12 - 0.2665639)
    if Q.mass_top50 > 160.8 and Q.C3 < 0.01034918:
        z += -1.057616 * (Q.mass_top50 - 160.8) * (0.01034918 - Q.C3)
    if Q.z_top2_slots < 0.5760704 and Q.orientation_deg > -44.87344:
        z += 0.0006439452 * (0.5760704 - Q.z_top2_slots) * (Q.orientation_deg - -44.87344)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.07794032
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.02704739 * Q.n_dr_0p2_0p4 + 0.1909098
    if 6.0 <= Q.n_dr_0p2_0p4 < 10.0:
        z += -0.007156358 * Q.n_dr_0p2_0p4 + 0.07156358
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += 25.07338 * Q.mass_over_sum_pt_sq - 0.2405795
    if 0.9313699 <= Q.psi_0p2 < 0.9935324:
        z += 0.05190988 * Q.psi_0p2 - 0.04834729
    if Q.psi_0p2 >= 0.9935324:
        z += 22.11559 * Q.psi_0p2 - 21.96933
    if Q.mass < 64.48544:
        z += -0.02624938 * Q.mass + 3.138744
    if 64.48544 <= Q.mass < 80.4:
        z += -0.04469102 * Q.mass + 4.327962
    if 80.4 <= Q.mass < 92.85979:
        z += -0.06475027 * Q.mass + 5.940725
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.008787807 * Q.mass - 0.8880054
    if Q.girth2_top10 < 0.00130722:
        z += -197.8336 * Q.girth2_top10 + 0.1985881
    if 0.00130722 <= Q.girth2_top10 < 0.00406126:
        z += -31.01926 * Q.girth2_top10 - 0.01947501
    if 0.00406126 <= Q.girth2_top10 < 0.00625621:
        z += 66.26676 * Q.girth2_top10 - 0.4145788
    if Q.e2 < 0.02515919:
        z += -54.5512 * Q.e2 + 1.305771
    if 0.02515919 <= Q.e2 < 0.03029714:
        z += -9.606109 * Q.e2 + 0.1749886
    if 0.03029714 <= Q.e2 < 0.03875945:
        z += 13.71365 * Q.e2 - 0.5315336
    if Q.girth2_top30 < 0.005402331:
        z += 2.130249 * Q.girth2_top30 - 0.1983784
    if 0.005402331 <= Q.girth2_top30 < 0.006929741:
        z += 122.3444 * Q.girth2_top30 - 0.8478149
    if 2.178951 <= Q.D2 < 3.814159:
        z += 0.07105317 * Q.D2 - 0.1548214
    if 3.814159 <= Q.D2 < 5.378975:
        z += 0.001666501 * Q.D2 + 0.1098305
    if Q.D2 >= 5.378975:
        z += -0.02218959 * Q.D2 + 0.2381518
    if Q.girth2_top15 < 0.009962397:
        z += -11.76218 * Q.girth2_top15 + 0.1171795
    if Q.psi_0p1 >= 0.9770626:
        z += -9.277217 * Q.psi_0p1 + 9.064422
    if Q.mass_top30 < 60.43821:
        z += -0.000840391 * Q.mass_top30 - 0.233809
    if 60.43821 <= Q.mass_top30 < 86.4:
        z += 0.01096229 * Q.mass_top30 - 0.9471419
    if Q.mass_top40 < 67.72643:
        z += 0.008272748 * Q.mass_top40 - 0.8820122
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.02062481 * Q.mass_top40 - 1.718573
    if Q.girth < 0.07374472:
        z += 6.081771 * Q.girth - 0.2628034
    if 0.07374472 <= Q.girth < 0.076787:
        z += 10.62607 * Q.girth - 0.5979217
    if 0.076787 <= Q.girth < 0.08589404:
        z += -23.93999 * Q.girth + 2.056303
    if Q.e2_sq < 0.00818374:
        z += -189.2723 * Q.e2_sq + 1.548956
    if Q.mass_over_sum_pt < 0.06030419:
        z += 24.61439 * Q.mass_over_sum_pt - 1.799431
    if 0.06030419 <= Q.mass_over_sum_pt < 0.07999061:
        z += 16.00498 * Q.mass_over_sum_pt - 1.280248
    if Q.lam1 < 0.006716737:
        z += 84.14066 * Q.lam1 - 0.5651507
    if Q.z_top40_slots >= 0.996191:
        z += -16.56708 * Q.z_top40_slots + 16.50398
    if Q.e3 < 3.793233e-05:
        z += 9054.185 * Q.e3 - 0.3434464
    if Q.sj2_mass1 < 19.89956:
        z += 0.01995968 * Q.sj2_mass1 - 0.3971889
    if Q.mass_top50 < 80.35535:
        z += 0.03155278 * Q.mass_top50 - 2.134856
    if 80.35535 <= Q.mass_top50 < 97.93004:
        z += -0.02279291 * Q.mass_top50 + 2.23211
    if 0.9896594 <= Q.psi_0p3 < 0.9980008:
        z += -1.948203 * Q.psi_0p3 + 1.928058
    if Q.psi_0p3 >= 0.9980008:
        z += 102.1047 * Q.psi_0p3 - 101.9168
    if Q.C2_b2 >= 0.01330402:
        z += -4.335155 * Q.C2_b2 + 0.05767501
    if Q.soft10_pt < 1.916992:
        z += -0.01989239 * Q.soft10_pt + 0.03813355
    if Q.zdr_0 < 0.005911134:
        z += 7.964327 * Q.zdr_0 - 0.04707821
    if Q.z_top30_slots >= 0.9909379:
        z += -0.7061841 * Q.z_top30_slots + 0.6997847
    if Q.girth2_top2 < 0.001056655:
        z += -110.1838 * Q.girth2_top2 + 0.1164262
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.002469891 * Q.n_dr_0p1_0p2 + 0.06834457
    if 8.0 <= Q.n_dr_0p1_0p2 < 15.0:
        z += -0.01258624 * Q.n_dr_0p1_0p2 + 0.1887936
    z += -0.007544742 * Q.n_real_top50 + 0.3772371
    if Q.z_top5_slots >= 0.534626:
        z += -0.4939268 * Q.z_top5_slots + 0.2640661
    if Q.girth2_top20 < 0.008031209:
        z += 31.88619 * Q.girth2_top20 - 0.2560847
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_particles > 34.0:
        z += 0.0005919184 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_particles - 34.0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2 < 0.005532208:
        z += -36.55213 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.00484226 * (10.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.mass < 101.0497 and Q.planar_flow < 0.3563114:
        z += 0.08207064 * (101.0497 - Q.mass) * (0.3563114 - Q.planar_flow)
    if Q.mass_over_sum_pt_sq < 0.009595015 and Q.z_dr_0p1_0p2 > 0.1203437:
        z += 879.1092 * (0.009595015 - Q.mass_over_sum_pt_sq) * (Q.z_dr_0p1_0p2 - 0.1203437)
    if Q.mass < 101.0497 and Q.sum_pt_top10 < 891.875:
        z += -1.353049e-05 * (101.0497 - Q.mass) * (891.875 - Q.sum_pt_top10)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.z_6 > 0.02949408:
        z += 0.0329171 * (10.0 - Q.n_dr_0p2_0p4) * (Q.z_6 - 0.02949408)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.pair_mass_0_12 > 7.607175:
        z += 0.003351134 * (10.0 - Q.n_dr_0p2_0p4) * (Q.pair_mass_0_12 - 7.607175)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_real_top40 > 26.0:
        z += -0.001146106 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_real_top40 - 26.0)
    if Q.mass < 101.0497 and Q.pt1_dr01 > 12.6865:
        z += 0.000871257 * (101.0497 - Q.mass) * (Q.pt1_dr01 - 12.6865)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.pt_11 > 6.632422:
        z += 0.001262274 * (10.0 - Q.n_dr_0p2_0p4) * (Q.pt_11 - 6.632422)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.pair_mass_0_10 > 7.661644:
        z += 0.0006176524 * (10.0 - Q.n_dr_0p2_0p4) * (Q.pair_mass_0_10 - 7.661644)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.pair_mass_0_13 > 8.669189:
        z += 0.00103968 * (10.0 - Q.n_dr_0p2_0p4) * (Q.pair_mass_0_13 - 8.669189)
    if Q.mass < 101.0497 and Q.ptdr0_2 > 11.87288:
        z += 0.0004197073 * (101.0497 - Q.mass) * (Q.ptdr0_2 - 11.87288)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.dr_8 > 0.05067946:
        z += 0.01032137 * (10.0 - Q.n_dr_0p2_0p4) * (Q.dr_8 - 0.05067946)
    if Q.z_top40_slots > 0.996191 and Q.C2_b2 < 0.006580753:
        z += 7359.555 * (Q.z_top40_slots - 0.996191) * (0.006580753 - Q.C2_b2)
    if Q.sj2_mass1 < 19.89956 and Q.dr_9 < 0.1147522:
        z += 0.001105891 * (19.89956 - Q.sj2_mass1) * (0.1147522 - Q.dr_9)
    if Q.mass < 80.4 and Q.pt1_over_pt0 < 0.8128995:
        z += -0.006284916 * (80.4 - Q.mass) * (0.8128995 - Q.pt1_over_pt0)
    if Q.psi_0p2 > 0.9935324 and Q.dr_11 > 0.05788126:
        z += 106.5427 * (Q.psi_0p2 - 0.9935324) * (Q.dr_11 - 0.05788126)
    if Q.girth < 0.08589404 and Q.soft3_dr > 0.161871:
        z += -2.415489 * (0.08589404 - Q.girth) * (Q.soft3_dr - 0.161871)
    if Q.psi_0p2 > 0.9313699 and Q.soft5_dr0 < 0.3897349:
        z += -0.538797 * (Q.psi_0p2 - 0.9313699) * (0.3897349 - Q.soft5_dr0)
    if Q.psi_0p2 > 0.9313699 and Q.dr_13 < 0.08052407:
        z += 29.25406 * (Q.psi_0p2 - 0.9313699) * (0.08052407 - Q.dr_13)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.absphi_7 > 0.05749512:
        z += 0.1815032 * (10.0 - Q.n_dr_0p2_0p4) * (Q.absphi_7 - 0.05749512)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.phi_0 < 0.0402832:
        z += 0.06331148 * (10.0 - Q.n_dr_0p2_0p4) * (0.0402832 - Q.phi_0)
    if Q.sj2_mass1 < 19.89956 and Q.soft10_abseta < 0.002968979:
        z += 2.284435 * (19.89956 - Q.sj2_mass1) * (0.002968979 - Q.soft10_abseta)
    if Q.psi_0p3 > 0.9896594 and Q.max_dr > 0.2982:
        z += 121.424 * (Q.psi_0p3 - 0.9896594) * (Q.max_dr - 0.2982)
    if Q.girth < 0.076787 and Q.n_dr_0p4_up > 0.0:
        z += -7.638783 * (0.076787 - Q.girth) * (Q.n_dr_0p4_up - 0.0)
    if Q.girth2_top15 < 0.009962397 and Q.eta_0 < 0.02980347:
        z += 389.5099 * (0.009962397 - Q.girth2_top15) * (0.02980347 - Q.eta_0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.zdr_7 > 0.004742028:
        z += 17.87666 * (10.0 - Q.n_dr_0p2_0p4) * (Q.zdr_7 - 0.004742028)
    if Q.z_top40_slots > 0.996191 and Q.dr_10 > 0.06334002:
        z += 28.4665 * (Q.z_top40_slots - 0.996191) * (Q.dr_10 - 0.06334002)
    if Q.mass_top50 < 80.35535 and Q.zdr_4 < 0.003932029:
        z += 6.069353 * (80.35535 - Q.mass_top50) * (0.003932029 - Q.zdr_4)
    if Q.mass < 92.85979 and Q.zdr_4 < 0.004495205:
        z += -4.117025 * (92.85979 - Q.mass) * (0.004495205 - Q.zdr_4)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.zdr_3 > 0.00781909:
        z += 2.377874 * (10.0 - Q.n_dr_0p2_0p4) * (Q.zdr_3 - 0.00781909)
    if Q.mass < 80.4 and Q.z_9 > 0.01833434:
        z += -0.9024078 * (80.4 - Q.mass) * (Q.z_9 - 0.01833434)
    if Q.mass < 101.0497 and Q.z_9 > 0.01620892:
        z += 0.2944382 * (101.0497 - Q.mass) * (Q.z_9 - 0.01620892)
    if Q.psi_0p2 > 0.9313699 and Q.soft10_dr0 < 0.2943504:
        z += 0.3503159 * (Q.psi_0p2 - 0.9313699) * (0.2943504 - Q.soft10_dr0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.M2 > 0.06621477:
        z += 0.0560474 * (10.0 - Q.n_dr_0p2_0p4) * (Q.M2 - 0.06621477)
    if Q.psi_0p2 > 0.9313699 and Q.sj3_mass2 < 10.81508:
        z += 0.02292123 * (Q.psi_0p2 - 0.9313699) * (10.81508 - Q.sj3_mass2)
    if Q.e2 < 0.03029714 and Q.sj3_pairmin_over_m > 0.09540583:
        z += 12.34095 * (0.03029714 - Q.e2) * (Q.sj3_pairmin_over_m - 0.09540583)
    if Q.z_top40_slots > 0.996191 and Q.dr_12 < 0.1711407:
        z += 60.32952 * (Q.z_top40_slots - 0.996191) * (0.1711407 - Q.dr_12)
    if Q.e2 < 0.03875945 and Q.soft1_abseta > 0.008917999:
        z += -9.707878 * (0.03875945 - Q.e2) * (Q.soft1_abseta - 0.008917999)
    if Q.psi_0p2 > 0.9935324 and Q.abseta_14 < 0.1206146:
        z += 75.55278 * (Q.psi_0p2 - 0.9935324) * (0.1206146 - Q.abseta_14)
    if Q.e2_sq < 0.00818374 and Q.soft2_abseta > 0.1358643:
        z += -49.2827 * (0.00818374 - Q.e2_sq) * (Q.soft2_abseta - 0.1358643)
    if Q.psi_0p2 > 0.9935324 and Q.eta_1 < -0.01510849:
        z += -351.2418 * (Q.psi_0p2 - 0.9935324) * (-0.01510849 - Q.eta_1)
    if Q.mass < 64.48544 and Q.pt_5 > 35.5:
        z += -0.0002552942 * (64.48544 - Q.mass) * (Q.pt_5 - 35.5)
    if Q.z_top30_slots > 0.9909379 and Q.z_5 > 0.03252317:
        z += 230.8124 * (Q.z_top30_slots - 0.9909379) * (Q.z_5 - 0.03252317)
    if Q.soft10_pt < 1.916992 and Q.max_pair_mass > 18.09798:
        z += 0.008190404 * (1.916992 - Q.soft10_pt) * (Q.max_pair_mass - 18.09798)
    if Q.sj2_mass1 < 19.89956 and Q.phi_10 > -0.1351318:
        z += 0.000840585 * (19.89956 - Q.sj2_mass1) * (Q.phi_10 - -0.1351318)
    if Q.soft10_pt < 1.916992 and Q.sj3_mass1 > 5.112677:
        z += 0.002804116 * (1.916992 - Q.soft10_pt) * (Q.sj3_mass1 - 5.112677)
    if Q.psi_0p3 > 0.9896594 and Q.soft4_absphi < 0.04641724:
        z += 158.6473 * (Q.psi_0p3 - 0.9896594) * (0.04641724 - Q.soft4_absphi)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.253561
    if Q.mass < 53.87362:
        z += -0.0637332 * Q.mass + 5.584778
    if 53.87362 <= Q.mass < 62.55:
        z += -0.06374055 * Q.mass + 5.585174
    if 62.55 <= Q.mass < 74.25181:
        z += -0.09963986 * Q.mass + 7.830675
    if 74.25181 <= Q.mass < 82.85409:
        z += -0.0565673 * Q.mass + 4.63246
    if 82.85409 <= Q.mass < 86.4:
        z += 0.01533372 * Q.mass - 1.324833
    if Q.mass >= 160.8:
        z += 0.008855257 * Q.mass - 1.423925
    if Q.sd_mass >= 125.1:
        z += 0.03079812 * Q.sd_mass - 3.852845
    if Q.mass_top40 < 74.78616:
        z += 0.009629718 * Q.mass_top40 - 0.6333621
    if 74.78616 <= Q.mass_top40 < 89.6788:
        z += -0.005828895 * Q.mass_top40 + 0.5227283
    if Q.sj3_pair_mass_max >= 120.6:
        z += -0.004543188 * Q.sj3_pair_mass_max + 0.5479085
    if Q.mass_over_sum_pt < 0.05077291:
        z += 18.9992 * Q.mass_over_sum_pt - 1.338984
    if 0.05077291 <= Q.mass_over_sum_pt < 0.06895248:
        z += 20.59122 * Q.mass_over_sum_pt - 1.419815
    if Q.girth2_top10 < 0.0007431905:
        z += -207.4486 * Q.girth2_top10 + 0.1541739
    if Q.lam1 < 0.002752094:
        z += -115.589 * Q.lam1 + 0.3181117
    if Q.log_sum_pt < 6.856375:
        z += 2.180923 * Q.log_sum_pt - 14.95323
    if Q.girth2_top15 < 0.006142802:
        z += 2.0127 * Q.girth2_top15 - 0.01236362
    if Q.mass_top15 < 24.20196:
        z += -0.004910914 * Q.mass_top15 + 0.1188537
    if 0.9538343 <= Q.psi_0p1 < 0.9770626:
        z += 5.56911 * Q.psi_0p1 - 5.312008
    if Q.psi_0p1 >= 0.9770626:
        z += 22.21365 * Q.psi_0p1 - 21.57477
    if Q.z_dr_0_0p05 >= 0.9351131:
        z += 1.928636 * Q.z_dr_0_0p05 - 1.803493
    if Q.z_dr_0p1_0p2 >= 0.6882177:
        z += 1.051153 * Q.z_dr_0p1_0p2 - 0.7234224
    if Q.soft5_pt < 1.339844:
        z += 0.2059678 * Q.soft5_pt - 0.2759646
    if Q.psi_0p3 >= 0.9973959:
        z += -67.62041 * Q.psi_0p3 + 67.44432
    if Q.soft4_dr < 0.1464158:
        z += 1.008777 * Q.soft4_dr - 0.147701
    if Q.mass < 86.4 and Q.tau21 < 0.5100475:
        z += 0.05559102 * (86.4 - Q.mass) * (0.5100475 - Q.tau21)
    if Q.mass < 86.4 and Q.z_dr_0p1_0p2 < 0.06472584:
        z += -0.2147199 * (86.4 - Q.mass) * (0.06472584 - Q.z_dr_0p1_0p2)
    if Q.sd_mass > 125.1 and Q.sd_zg < 0.4199841:
        z += -0.07047059 * (Q.sd_mass - 125.1) * (0.4199841 - Q.sd_zg)
    if Q.mass_top40 < 89.6788 and Q.n_dr_0p05_0p1 > 1.0:
        z += 0.0005303568 * (89.6788 - Q.mass_top40) * (Q.n_dr_0p05_0p1 - 1.0)
    if Q.mass_top40 < 89.6788 and Q.pt_0 < 334.5:
        z += -1.127695e-05 * (89.6788 - Q.mass_top40) * (334.5 - Q.pt_0)
    if Q.mass < 86.4 and Q.psi_0p3 < 0.9985421:
        z += -2.068739 * (86.4 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.sj3_pair_mass_max > 120.6 and Q.sj3_pair_mass_min < 76.60223:
        z += 0.0003369997 * (Q.sj3_pair_mass_max - 120.6) * (76.60223 - Q.sj3_pair_mass_min)
    if Q.mass < 86.4 and Q.zdr_0 > 0.007860787:
        z += 1.588612 * (86.4 - Q.mass) * (Q.zdr_0 - 0.007860787)
    if Q.sj3_pair_mass_max > 120.6 and Q.z_dr_0p1_0p2 > 0.2864926:
        z += 0.02814306 * (Q.sj3_pair_mass_max - 120.6) * (Q.z_dr_0p1_0p2 - 0.2864926)
    if Q.sj3_pair_mass_max > 120.6 and Q.soft9_z < 0.00104408:
        z += 71.2541 * (Q.sj3_pair_mass_max - 120.6) * (0.00104408 - Q.soft9_z)
    if Q.girth2_top15 < 0.006142802 and Q.pt_5 < 48.28125:
        z += 2.092834 * (0.006142802 - Q.girth2_top15) * (48.28125 - Q.pt_5)
    if Q.mass < 86.4 and Q.lam2 < 0.003687605:
        z += 10.08036 * (86.4 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.girth2_top10 < 0.0007431905 and Q.sum_pt_top40 < 1069.671:
        z += 3.959504 * (0.0007431905 - Q.girth2_top10) * (1069.671 - Q.sum_pt_top40)
    if Q.girth2_top15 < 0.006142802 and Q.sum_pt_top40 > 1225.842:
        z += -0.398 * (0.006142802 - Q.girth2_top15) * (Q.sum_pt_top40 - 1225.842)
    if Q.girth2_top15 < 0.006142802 and Q.z_10 > 0.01143749:
        z += -1448.603 * (0.006142802 - Q.girth2_top15) * (Q.z_10 - 0.01143749)
    if Q.mass < 86.4 and Q.z_top50_slots < 0.985099:
        z += -2.050118 * (86.4 - Q.mass) * (0.985099 - Q.z_top50_slots)
    if Q.sj3_pair_mass_max > 120.6 and Q.sj3_mass1 < 32.50209:
        z += 0.0005949443 * (Q.sj3_pair_mass_max - 120.6) * (32.50209 - Q.sj3_mass1)
    if Q.z_dr_0p1_0p2 > 0.6882177 and Q.sj3_z3 < 0.2502892:
        z += -12.28785 * (Q.z_dr_0p1_0p2 - 0.6882177) * (0.2502892 - Q.sj3_z3)
    if Q.log_sum_pt < 6.856375 and Q.lam2 < 0.002396991:
        z += -2288.407 * (6.856375 - Q.log_sum_pt) * (0.002396991 - Q.lam2)
    if Q.mass < 74.25181 and Q.dr_max_012 > 0.1206357:
        z += 1.057944 * (74.25181 - Q.mass) * (Q.dr_max_012 - 0.1206357)
    if Q.z_dr_0p1_0p2 > 0.6882177 and Q.n_dr_0p4_up > 0.0:
        z += 3.868542 * (Q.z_dr_0p1_0p2 - 0.6882177) * (Q.n_dr_0p4_up - 0.0)
    if Q.sd_mass > 125.1 and Q.girth2_top3 > 0.01002369:
        z += 0.3268617 * (Q.sd_mass - 125.1) * (Q.girth2_top3 - 0.01002369)
    if Q.sd_mass > 125.1 and Q.lam2 > 0.0001679609:
        z += -2.001993 * (Q.sd_mass - 125.1) * (Q.lam2 - 0.0001679609)
    if Q.z_dr_0p1_0p2 > 0.6882177 and Q.soft4_absphi > 0.170166:
        z += -4.342629 * (Q.z_dr_0p1_0p2 - 0.6882177) * (Q.soft4_absphi - 0.170166)
    if Q.psi_0p3 > 0.9973959 and Q.sj3_z2 < 0.1873652:
        z += 228.5118 * (Q.psi_0p3 - 0.9973959) * (0.1873652 - Q.sj3_z2)
    if Q.girth2_top10 < 0.0007431905 and Q.n_real_top40 < 36.0:
        z += 1.694609 * (0.0007431905 - Q.girth2_top10) * (36.0 - Q.n_real_top40)
    if Q.sj3_pair_mass_max > 120.6 and Q.orientation_deg > 35.6254:
        z += -0.0001341096 * (Q.sj3_pair_mass_max - 120.6) * (Q.orientation_deg - 35.6254)
    if Q.sj3_pair_mass_max > 120.6 and Q.orientation_deg < -62.72059:
        z += -0.0002454134 * (Q.sj3_pair_mass_max - 120.6) * (-62.72059 - Q.orientation_deg)
    if Q.z_dr_0_0p05 > 0.9351131 and Q.n_real_top15 < 15.0:
        z += 5.685732 * (Q.z_dr_0_0p05 - 0.9351131) * (15.0 - Q.n_real_top15)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.181007
    if Q.sum_pt < 1007.788:
        z += 0.02107743 * Q.sum_pt - 22.4515
    if 1007.788 <= Q.sum_pt < 1034.834:
        z += 0.01735281 * Q.sum_pt - 18.69787
    if 1034.834 <= Q.sum_pt < 1085.125:
        z += 0.01127693 * Q.sum_pt - 12.41034
    if 1085.125 <= Q.sum_pt < 1115.723:
        z += 0.005669093 * Q.sum_pt - 6.325136
    if 74.25181 <= Q.mass < 101.0497:
        z += 0.02306807 * Q.mass - 1.712846
    if 101.0497 <= Q.mass < 136.785:
        z += -0.0005547125 * Q.mass + 0.6742295
    if 136.785 <= Q.mass < 143.7876:
        z += -0.04987009 * Q.mass + 7.419833
    if 143.7876 <= Q.mass < 160.8:
        z += -0.1342579 * Q.mass + 19.55375
    if 160.8 <= Q.mass < 162.8363:
        z += -0.3394772 * Q.mass + 52.55301
    if 162.8363 <= Q.mass < 172.8:
        z += -0.2572714 * Q.mass + 39.16693
    if Q.mass >= 172.8:
        z += -0.1236641 * Q.mass + 16.07959
    if Q.n_for_90pct < 16.0:
        z += 0.03965649 * Q.n_for_90pct - 0.6345038
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 260.126 * Q.mass_over_sum_pt - 44.45036
    if Q.log_sum_pt < 6.811175:
        z += -6.031199 * Q.log_sum_pt + 39.54185
    if 6.811175 <= Q.log_sum_pt < 6.910131:
        z += 15.53924 * Q.log_sum_pt - 107.3782
    if Q.log_sum_pt >= 7.139296:
        z += 0.554477 * Q.log_sum_pt - 3.958575
    if Q.sum_pt_top40 < 1007.44:
        z += -0.008751814 * Q.sum_pt_top40 + 9.072182
    if 1007.44 <= Q.sum_pt_top40 < 1053.047:
        z += -0.005596793 * Q.sum_pt_top40 + 5.893688
    if Q.mass_top10 < 23.38894:
        z += 0.001456542 * Q.mass_top10 - 0.1121371
    if 23.38894 <= Q.mass_top10 < 56.92192:
        z += -0.00249119 * Q.mass_top10 - 0.01980384
    if 56.92192 <= Q.mass_top10 < 76.9886:
        z += -0.001362235 * Q.mass_top10 - 0.08406615
    if Q.mass_top10 >= 76.9886:
        z += -0.002818776 * Q.mass_top10 + 0.02807095
    if Q.C2 >= 0.0977156:
        z += -3.087319 * Q.C2 + 0.3016792
    if 27.46086 <= Q.mass_top50 < 92.16545:
        z += 0.006109578 * Q.mass_top50 - 0.1677743
    if 92.16545 <= Q.mass_top50 < 136.785:
        z += 0.004456901 * Q.mass_top50 - 0.01545458
    if 136.785 <= Q.mass_top50 < 168.9698:
        z += 0.05030882 * Q.mass_top50 - 6.28731
    if Q.mass_top50 >= 168.9698:
        z += 0.03112209 * Q.mass_top50 - 3.045332
    if Q.n_dr_0p1_0p2 < 17.0:
        z += -0.005564835 * Q.n_dr_0p1_0p2 + 0.09460219
    if Q.girth2_top50 < 0.007820315:
        z += -70.58204 * Q.girth2_top50 + 1.299376
    if 0.007820315 <= Q.girth2_top50 < 0.01951641:
        z += -42.26102 * Q.girth2_top50 + 1.077896
    if 0.01951641 <= Q.girth2_top50 < 0.02550569:
        z += -86.11306 * Q.girth2_top50 + 1.933731
    if Q.girth2_top50 >= 0.02550569:
        z += -43.85205 * Q.girth2_top50 + 0.8558345
    if Q.pt_entropy < 2.752054:
        z += 0.02778482 * Q.pt_entropy - 0.07646532
    if Q.tau1 < 0.08286256:
        z += -9.229854 * Q.tau1 + 0.7648093
    if Q.tau1 >= 0.1751567:
        z += -2.880164 * Q.tau1 + 0.5044801
    if Q.mass_over_sum_pt_sq < 0.02580396:
        z += 65.6704 * Q.mass_over_sum_pt_sq - 1.694556
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -636.6726 * Q.mass_over_sum_pt_sq + 18.59085
    if 0.9638082 <= Q.psi_0p3 < 0.9973959:
        z += 5.428079 * Q.psi_0p3 - 5.231627
    if Q.psi_0p3 >= 0.9973959:
        z += 32.64651 * Q.psi_0p3 - 32.37917
    z += 0.01215932 * Q.n_particles
    if Q.mass_top40 < 160.8:
        z += -0.0008809055 * Q.mass_top40 + 0.1416496
    if Q.sum_pt_top50 < 997.0189:
        z += -0.002488828 * Q.sum_pt_top50 + 2.481409
    if Q.sum_pt_top30 < 966.0633:
        z += 0.004940428 * Q.sum_pt_top30 - 4.772766
    if Q.sum_pt_top30 >= 1111.245:
        z += 0.004494532 * Q.sum_pt_top30 - 4.994526
    if Q.mass_top5 >= 33.71058:
        z += 0.004712102 * Q.mass_top5 - 0.1588477
    if Q.pt_11 < 18.32812:
        z += 0.004294563 * Q.pt_11 - 0.07871129
    if Q.z_dr_0p1_0p2 < 0.08871546:
        z += -0.4111441 * Q.z_dr_0p1_0p2 + 0.03647484
    if Q.sum_pt_top20 >= 956.5062:
        z += -0.0005382932 * Q.sum_pt_top20 + 0.5148808
    if Q.pt_6 < 41.21875:
        z += -0.005483514 * Q.pt_6 + 0.2260236
    if 34.0 <= Q.n_pt_above_1 < 58.0:
        z += -5.787291e-05 * Q.n_pt_above_1 + 0.001967679
    if Q.n_pt_above_1 >= 58.0:
        z += 0.07817779 * Q.n_pt_above_1 - 4.535701
    if Q.psi_0p1 >= 0.9371031:
        z += 1.517344 * Q.psi_0p1 - 1.421907
    if Q.mass_top30 >= 138.3818:
        z += 0.008668928 * Q.mass_top30 - 1.199622
    if Q.mass_top20 >= 91.19:
        z += -0.00954149 * Q.mass_top20 + 0.8700885
    if Q.lam1 >= 0.01174405:
        z += -0.521109 * Q.lam1 + 0.00611993
    if Q.sum_pt < 1085.125 and Q.sum_pt_top40 < 858.8262:
        z += 1.070899e-05 * (1085.125 - Q.sum_pt) * (858.8262 - Q.sum_pt_top40)
    if Q.sum_pt < 1085.125 and Q.tau21_b2 < 0.8310045:
        z += -0.004219246 * (1085.125 - Q.sum_pt) * (0.8310045 - Q.tau21_b2)
    if Q.n_for_90pct < 16.0 and Q.D2 < 3.814159:
        z += 0.01860449 * (16.0 - Q.n_for_90pct) * (3.814159 - Q.D2)
    if Q.sum_pt_top40 < 1053.047 and Q.sj2_zsoft < 0.2179035:
        z += 0.01473913 * (1053.047 - Q.sum_pt_top40) * (0.2179035 - Q.sj2_zsoft)
    if Q.mass > 136.785 and Q.sum_pt < 1028.184:
        z += 0.0004824082 * (Q.mass - 136.785) * (1028.184 - Q.sum_pt)
    if Q.mass > 74.25181 and Q.sum_pt < 1028.184:
        z += -0.0001625796 * (Q.mass - 74.25181) * (1028.184 - Q.sum_pt)
    if Q.sum_pt < 1007.788 and Q.sum_pt_top3 > 512.4375:
        z += -2.435167e-05 * (1007.788 - Q.sum_pt) * (Q.sum_pt_top3 - 512.4375)
    if Q.log_sum_pt > 7.139296 and Q.C3 < 0.00317537:
        z += -82.84832 * (Q.log_sum_pt - 7.139296) * (0.00317537 - Q.C3)
    if Q.log_sum_pt > 7.139296 and Q.dr_12 < 0.05398073:
        z += 21.61304 * (Q.log_sum_pt - 7.139296) * (0.05398073 - Q.dr_12)
    if Q.log_sum_pt < 6.811175 and Q.dr_13 < 0.1312677:
        z += 49.55147 * (6.811175 - Q.log_sum_pt) * (0.1312677 - Q.dr_13)
    if Q.psi_0p3 > 0.9638082 and Q.max_dr < 0.4357228:
        z += 21.20343 * (Q.psi_0p3 - 0.9638082) * (0.4357228 - Q.max_dr)
    if Q.sum_pt < 1085.125 and Q.dr_13 < 0.1023813:
        z += -0.008521867 * (1085.125 - Q.sum_pt) * (0.1023813 - Q.dr_13)
    if Q.psi_0p3 > 0.9638082 and Q.sj3_dr13 > 0.2465017:
        z += -60.74078 * (Q.psi_0p3 - 0.9638082) * (Q.sj3_dr13 - 0.2465017)
    if Q.mass > 172.8 and Q.pt_6 < 56.53125:
        z += -0.001350305 * (Q.mass - 172.8) * (56.53125 - Q.pt_6)
    if Q.psi_0p3 > 0.9638082 and Q.C3 < 0.0223982:
        z += 341.0288 * (Q.psi_0p3 - 0.9638082) * (0.0223982 - Q.C3)
    if Q.psi_0p3 > 0.9638082 and Q.absphi_12 > 0.01163483:
        z += 37.07725 * (Q.psi_0p3 - 0.9638082) * (Q.absphi_12 - 0.01163483)
    if Q.mass > 172.8 and Q.z_6 < 0.04568661:
        z += 0.3836236 * (Q.mass - 172.8) * (0.04568661 - Q.z_6)
    if Q.log_sum_pt > 7.139296 and Q.sd_zg > 0.4462823:
        z += 402.8548 * (Q.log_sum_pt - 7.139296) * (Q.sd_zg - 0.4462823)
    if Q.log_sum_pt < 6.811175 and Q.zdr_11 < 0.005041702:
        z += 1030.691 * (6.811175 - Q.log_sum_pt) * (0.005041702 - Q.zdr_11)
    if Q.C2 > 0.0977156 and Q.orientation_deg < 53.9394:
        z += -0.02120619 * (Q.C2 - 0.0977156) * (53.9394 - Q.orientation_deg)
    if Q.pt_11 < 18.32812 and Q.dr0_14 < 0.2327893:
        z += 0.01761353 * (18.32812 - Q.pt_11) * (0.2327893 - Q.dr0_14)
    if Q.log_sum_pt > 7.139296 and Q.phi_11 > -0.1395264:
        z += 10.62891 * (Q.log_sum_pt - 7.139296) * (Q.phi_11 - -0.1395264)
    if Q.psi_0p3 > 0.9638082 and Q.abseta_0 > 0.02487183:
        z += 5.986402 * (Q.psi_0p3 - 0.9638082) * (Q.abseta_0 - 0.02487183)
    if Q.sum_pt_top30 > 1111.245 and Q.mratio_min_012 < 0.01190756:
        z += 1.385963 * (Q.sum_pt_top30 - 1111.245) * (0.01190756 - Q.mratio_min_012)
    if Q.mass > 162.8363 and Q.soft6_z > 0.0005748372:
        z += 22.79382 * (Q.mass - 162.8363) * (Q.soft6_z - 0.0005748372)
    if Q.mass_top50 > 136.785 and Q.soft6_z > 0.0008188601:
        z += -15.72531 * (Q.mass_top50 - 136.785) * (Q.soft6_z - 0.0008188601)
    if Q.mass > 172.8 and Q.soft5_z > 0.0004140594:
        z += 1.805939 * (Q.mass - 172.8) * (Q.soft5_z - 0.0004140594)
    if Q.mass_top50 > 92.16545 and Q.soft7_z < 0.001923089:
        z += -5.969453 * (Q.mass_top50 - 92.16545) * (0.001923089 - Q.soft7_z)
    if Q.mass > 172.8 and Q.soft6_z > 0.0004464147:
        z += -17.98766 * (Q.mass - 172.8) * (Q.soft6_z - 0.0004464147)
    if Q.sum_pt < 1085.125 and Q.dr_4 < 0.1289397:
        z += -0.003547428 * (1085.125 - Q.sum_pt) * (0.1289397 - Q.dr_4)
    if Q.sum_pt_top40 < 1053.047 and Q.dr0_3 < 0.1617791:
        z += -0.002609493 * (1053.047 - Q.sum_pt_top40) * (0.1617791 - Q.dr0_3)
    if Q.girth2_top50 < 0.02550569 and Q.psi_0p3 > 0.9638082:
        z += 875.2795 * (0.02550569 - Q.girth2_top50) * (Q.psi_0p3 - 0.9638082)
    if Q.log_sum_pt < 6.811175 and Q.D2 < 2.680253:
        z += 3.073041 * (6.811175 - Q.log_sum_pt) * (2.680253 - Q.D2)
    if Q.sum_pt < 1034.834 and Q.D2 < 2.975532:
        z += -0.003707533 * (1034.834 - Q.sum_pt) * (2.975532 - Q.D2)
    if Q.mass > 160.8 and Q.D2 > 0.8942376:
        z += 0.02979018 * (Q.mass - 160.8) * (Q.D2 - 0.8942376)
    if Q.mass > 136.785 and Q.D2 > 0.8942376:
        z += -0.01365857 * (Q.mass - 136.785) * (Q.D2 - 0.8942376)
    if Q.mass > 172.8 and Q.D2 > 0.8942376:
        z += -0.02752575 * (Q.mass - 172.8) * (Q.D2 - 0.8942376)
    if Q.n_for_90pct < 16.0 and Q.D2 < 9.676985:
        z += 0.0048242 * (16.0 - Q.n_for_90pct) * (9.676985 - Q.D2)
    if Q.mass_top30 > 138.3818 and Q.dr_8 < 0.1502914:
        z += -0.149151 * (Q.mass_top30 - 138.3818) * (0.1502914 - Q.dr_8)
    if Q.mass > 136.785 and Q.z_9 < 0.02291864:
        z += 0.7275472 * (Q.mass - 136.785) * (0.02291864 - Q.z_9)
    if Q.sum_pt < 1034.834 and Q.pt_9 < 16.64062:
        z += 0.0005189819 * (1034.834 - Q.sum_pt) * (16.64062 - Q.pt_9)
    if Q.log_sum_pt > 7.139296 and Q.dr_9 < 0.04047238:
        z += -44.12809 * (Q.log_sum_pt - 7.139296) * (0.04047238 - Q.dr_9)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.03572892
    if Q.tau21_b2 < 0.1722488:
        z += -1.881074 * Q.tau21_b2 + 0.4912315
    if 0.1722488 <= Q.tau21_b2 < 0.342495:
        z += -0.9822174 * Q.tau21_b2 + 0.3364045
    if Q.mass < 74.25181:
        z += 0.03589818 * Q.mass - 3.441975
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.05281005 * Q.mass - 4.697713
    if 78.26182 <= Q.mass < 80.4:
        z += 0.06516874 * Q.mass - 5.664926
    if 80.4 <= Q.mass < 82.85409:
        z += 0.1187883 * Q.mass - 9.975941
    if 82.85409 <= Q.mass < 89.74183:
        z += 0.1092431 * Q.mass - 9.185081
    if 89.74183 <= Q.mass < 91.19:
        z += 0.03452802 * Q.mass - 2.480011
    if 91.19 <= Q.mass < 125.1:
        z += -0.01971687 * Q.mass + 2.46658
    if Q.psi_0p3 >= 0.9924477:
        z += 49.09912 * Q.psi_0p3 - 48.72831
    if Q.N2 < 0.4226723:
        z += -0.4372829 * Q.N2 + 0.1848274
    if Q.n_dr_0p2_0p4 < 18.0:
        z += 0.002956579 * Q.n_dr_0p2_0p4 - 0.05321842
    if Q.pt_10 >= 28.15625:
        z += 0.002535384 * Q.pt_10 - 0.0713869
    if Q.e2 < 0.01256572:
        z += -24.30211 * Q.e2 + 0.9626457
    if 0.01256572 <= Q.e2 < 0.02210818:
        z += -16.71891 * Q.e2 + 0.8673573
    if 0.02210818 <= Q.e2 < 0.03029714:
        z += -34.98126 * Q.e2 + 1.271105
    if 0.03029714 <= Q.e2 < 0.04358622:
        z += -15.89821 * Q.e2 + 0.6929428
    if Q.mass_over_sum_pt < 0.07852883:
        z += -36.02236 * Q.mass_over_sum_pt + 5.507184
    if 0.07852883 <= Q.mass_over_sum_pt < 0.08873143:
        z += -26.03294 * Q.mass_over_sum_pt + 4.722727
    if 0.08873143 <= Q.mass_over_sum_pt < 0.09046749:
        z += -16.62898 * Q.mass_over_sum_pt + 3.8883
    if 0.09046749 <= Q.mass_over_sum_pt < 0.09795415:
        z += -95.85382 * Q.mass_over_sum_pt + 11.05557
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -54.12621 * Q.mass_over_sum_pt + 6.96818
    if 0.1182259 <= Q.mass_over_sum_pt < 0.140939:
        z += -25.05422 * Q.mass_over_sum_pt + 3.531118
    if Q.girth2_top20 < 0.006374178:
        z += -12.17075 * Q.girth2_top20 - 0.1591508
    if 0.006374178 <= Q.girth2_top20 < 0.01083435:
        z += 53.07625 * Q.girth2_top20 - 0.5750468
    if Q.sj3_mass1 < 32.50209:
        z += -0.001442211 * Q.sj3_mass1 + 0.04687488
    if Q.girth < 0.076787:
        z += 8.909588 * Q.girth - 0.6841405
    if Q.girth2_top5 < 0.007164202:
        z += -49.66665 * Q.girth2_top5 + 0.3558219
    if Q.sd_mass < 69.65633:
        z += -0.001534806 * Q.sd_mass + 0.106909
    if Q.sd_rg < 0.1778185:
        z += -1.888844 * Q.sd_rg + 0.1975905
    if 0.1778185 <= Q.sd_rg < 0.2042612:
        z += 2.956873 * Q.sd_rg - 0.6640676
    if 0.2042612 <= Q.sd_rg < 0.3017146:
        z += 0.616635 * Q.sd_rg - 0.1860478
    if Q.girth2_top30 < 0.002582316:
        z += 70.04347 * Q.girth2_top30 - 0.9070239
    if 0.002582316 <= Q.girth2_top30 < 0.01215787:
        z += 75.83366 * Q.girth2_top30 - 0.921976
    if Q.n_pt_above_1 >= 58.0:
        z += 0.01351596 * Q.n_pt_above_1 - 0.7839259
    if Q.tau1 < 0.06310829:
        z += -0.8139505 * Q.tau1 - 0.424402
    if 0.06310829 <= Q.tau1 < 0.1072713:
        z += 10.77303 * Q.tau1 - 1.155636
    if Q.mass_top30 < 82.66587:
        z += -0.004854362 * Q.mass_top30 + 0.40129
    if Q.sum_pt_top40 < 906.6023:
        z += 4.842808e-05 * Q.sum_pt_top40 + 0.19418
    if 906.6023 <= Q.sum_pt_top40 < 1024.942:
        z += -0.002011872 * Q.sum_pt_top40 + 2.062053
    if Q.girth2_top3 < 0.001592178:
        z += -4.563465 * Q.girth2_top3 + 0.007265846
    if Q.log_sum_pt < 6.941997:
        z += -2.507467 * Q.log_sum_pt + 17.34769
    if 6.941997 <= Q.log_sum_pt < 6.98945:
        z += 1.246053 * Q.log_sum_pt - 8.709225
    if Q.sum_pt_top50 < 976.277:
        z += 0.001292581 * Q.sum_pt_top50 - 1.261917
    if Q.lam2 < 0.001163277:
        z += -52.44729 * Q.lam2 + 0.06101074
    if Q.z_dr_0p2_0p4 < 0.1291856:
        z += 0.2498925 * Q.z_dr_0p2_0p4 - 0.03228252
    if Q.lam1 < 0.007259287:
        z += 9.880936 * Q.lam1 - 0.2744753
    if 0.007259287 <= Q.lam1 < 0.007671243:
        z += 23.21857 * Q.lam1 - 0.371297
    if 0.007671243 <= Q.lam1 < 0.008241985:
        z += 338.4745 * Q.lam1 - 2.789702
    if Q.e3 < 0.0001086251:
        z += 2777.515 * Q.e3 - 0.3017078
    if Q.psi_0p1 >= 0.9538343:
        z += -0.7589803 * Q.psi_0p1 + 0.7239414
    if Q.psi_0p2 >= 0.9804031:
        z += 0.06274103 * Q.psi_0p2 - 0.0615115
    if Q.tau2 < 0.0795038:
        z += 2.937171 * Q.tau2 - 0.2335163
    if Q.sj2_mass1 < 37.45803:
        z += -0.001049448 * Q.sj2_mass1 + 0.03931026
    if Q.z_top30_slots < 0.9734886:
        z += 0.9624627 * Q.z_top30_slots - 0.9369465
    if Q.mass_top50 < 71.79516:
        z += 0.01745443 * Q.mass_top50 - 1.443741
    if 71.79516 <= Q.mass_top50 < 77.37641:
        z += -0.006425388 * Q.mass_top50 + 0.270715
    if 77.37641 <= Q.mass_top50 < 85.8667:
        z += -0.01655932 * Q.mass_top50 + 1.054842
    if 85.8667 <= Q.mass_top50 < 97.93004:
        z += 0.01739022 * Q.mass_top50 - 1.860292
    if 97.93004 <= Q.mass_top50 < 117.0487:
        z += 0.008225852 * Q.mass_top50 - 0.9628256
    if Q.girth2_top40 < 0.01897915:
        z += 62.39593 * Q.girth2_top40 - 1.184222
    if Q.soft7_z < 0.001595561:
        z += 145.1856 * Q.soft7_z - 0.2316524
    if Q.soft7_pt < 1.650391:
        z += -0.272423 * Q.soft7_pt + 0.4496043
    if Q.soft9_pt < 3.080078:
        z += 0.005462564 * Q.soft9_pt - 0.01682512
    if Q.mass_top40 < 89.6788:
        z += -0.004627848 * Q.mass_top40 + 0.4150198
    if Q.tau21_b2 < 0.342495 and Q.girth2_top40 < 0.007709916:
        z += -872.2606 * (0.342495 - Q.tau21_b2) * (0.007709916 - Q.girth2_top40)
    if Q.tau21_b2 < 0.342495 and Q.lam1 < 0.02048524:
        z += 14.3733 * (0.342495 - Q.tau21_b2) * (0.02048524 - Q.lam1)
    if Q.tau21_b2 < 0.342495 and Q.girth2_top50 < 0.006118006:
        z += 187.5777 * (0.342495 - Q.tau21_b2) * (0.006118006 - Q.girth2_top50)
    if Q.N2 < 0.4226723 and Q.max_dr > 0.2404747:
        z += -3.706947 * (0.4226723 - Q.N2) * (Q.max_dr - 0.2404747)
    if Q.N2 < 0.4226723 and Q.dr_12 < 0.2053949:
        z += 0.8292578 * (0.4226723 - Q.N2) * (0.2053949 - Q.dr_12)
    if Q.mass < 125.1 and Q.mass_top2 > 28.78966:
        z += -0.0006915455 * (125.1 - Q.mass) * (Q.mass_top2 - 28.78966)
    if Q.psi_0p3 > 0.9924477 and Q.sd_rg > 0.2280025:
        z += 31.96377 * (Q.psi_0p3 - 0.9924477) * (Q.sd_rg - 0.2280025)
    if Q.psi_0p3 > 0.9924477 and Q.sd_rg < 0.1881908:
        z += -498.3257 * (Q.psi_0p3 - 0.9924477) * (0.1881908 - Q.sd_rg)
    if Q.tau21_b2 < 0.342495 and Q.n_real_top40 < 32.0:
        z += -0.167822 * (0.342495 - Q.tau21_b2) * (32.0 - Q.n_real_top40)
    if Q.tau21_b2 < 0.342495 and Q.n_for_50pct > 3.0:
        z += 0.01628357 * (0.342495 - Q.tau21_b2) * (Q.n_for_50pct - 3.0)
    if Q.mass_over_sum_pt < 0.09046749 and Q.psi_0p2 > 0.9087063:
        z += -420.0688 * (0.09046749 - Q.mass_over_sum_pt) * (Q.psi_0p2 - 0.9087063)
    if Q.girth2_top5 < 0.007164202 and Q.zdr_5 > 0.008949744:
        z += -22553.87 * (0.007164202 - Q.girth2_top5) * (Q.zdr_5 - 0.008949744)
    if Q.tau21_b2 < 0.342495 and Q.sj3_mass2 > 5.762132:
        z += -0.05848576 * (0.342495 - Q.tau21_b2) * (Q.sj3_mass2 - 5.762132)
    if Q.girth2_top5 < 0.007164202 and Q.pair_mass_0_4 > 8.587823:
        z += 1.150288 * (0.007164202 - Q.girth2_top5) * (Q.pair_mass_0_4 - 8.587823)
    if Q.psi_0p3 > 0.9924477 and Q.z_2nd < 0.1231747:
        z += -504.2266 * (Q.psi_0p3 - 0.9924477) * (0.1231747 - Q.z_2nd)
    if Q.tau21_b2 < 0.342495 and Q.n_pt_above_50 < 7.0:
        z += 0.04594657 * (0.342495 - Q.tau21_b2) * (7.0 - Q.n_pt_above_50)
    if Q.girth2_top5 < 0.007164202 and Q.n_pt_above_10 > 13.0:
        z += -1.604607 * (0.007164202 - Q.girth2_top5) * (Q.n_pt_above_10 - 13.0)
    if Q.girth < 0.076787 and Q.z_dr_0p2_0p4 < 0.05180474:
        z += 189.1105 * (0.076787 - Q.girth) * (0.05180474 - Q.z_dr_0p2_0p4)
    if Q.girth2_top30 < 0.01215787 and Q.pair_mass_0_2 > 22.41128:
        z += -5.57012 * (0.01215787 - Q.girth2_top30) * (Q.pair_mass_0_2 - 22.41128)
    if Q.mass < 91.19 and Q.n_dr_0p05_0p1 > 14.0:
        z += -0.001098768 * (91.19 - Q.mass) * (Q.n_dr_0p05_0p1 - 14.0)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.M3 < 0.03787151:
        z += 0.1662852 * (18.0 - Q.n_dr_0p2_0p4) * (0.03787151 - Q.M3)
    if Q.girth2_top5 < 0.007164202 and Q.sj2_mass2 > 14.45919:
        z += 2.09462 * (0.007164202 - Q.girth2_top5) * (Q.sj2_mass2 - 14.45919)
    if Q.tau21_b2 < 0.342495 and Q.sj3_dr13 < 0.3682013:
        z += 0.2389555 * (0.342495 - Q.tau21_b2) * (0.3682013 - Q.sj3_dr13)
    if Q.psi_0p2 > 0.9804031 and Q.pt2_over_pt0 > 0.1852611:
        z += 6.584083 * (Q.psi_0p2 - 0.9804031) * (Q.pt2_over_pt0 - 0.1852611)
    if Q.sj2_mass1 < 37.45803 and Q.phi_6 < -0.05432129:
        z += 0.04946242 * (37.45803 - Q.sj2_mass1) * (-0.05432129 - Q.phi_6)
    if Q.tau21_b2 < 0.342495 and Q.orientation_deg > -18.47737:
        z += -0.007441485 * (0.342495 - Q.tau21_b2) * (Q.orientation_deg - -18.47737)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.soft9_pt < 3.445312:
        z += 0.003478569 * (18.0 - Q.n_dr_0p2_0p4) * (3.445312 - Q.soft9_pt)
    if Q.z_dr_0p2_0p4 < 0.1291856 and Q.eta_11 > -0.04888916:
        z += -0.5088037 * (0.1291856 - Q.z_dr_0p2_0p4) * (Q.eta_11 - -0.04888916)
    if Q.N2 < 0.4226723 and Q.abseta_14 > 0.2005615:
        z += -4.003071 * (0.4226723 - Q.N2) * (Q.abseta_14 - 0.2005615)
    if Q.mass_top50 < 117.0487 and Q.zdr_6 > 0.006336433:
        z += -1.164914 * (117.0487 - Q.mass_top50) * (Q.zdr_6 - 0.006336433)
    if Q.psi_0p2 > 0.9804031 and Q.phi_2 < -0.0966217:
        z += 397.5065 * (Q.psi_0p2 - 0.9804031) * (-0.0966217 - Q.phi_2)
    if Q.tau21_b2 < 0.342495 and Q.absphi_5 < 0.06195068:
        z += 10.40767 * (0.342495 - Q.tau21_b2) * (0.06195068 - Q.absphi_5)
    if Q.tau2 < 0.0795038 and Q.absphi_1 < 0.1178619:
        z += 6.510404 * (0.0795038 - Q.tau2) * (0.1178619 - Q.absphi_1)
    if Q.N2 < 0.4226723 and Q.sj3_dr_min > 0.05940798:
        z += 4.064188 * (0.4226723 - Q.N2) * (Q.sj3_dr_min - 0.05940798)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.4122125
    if Q.z_dr_0_0p05 < 0.8459004:
        z += -0.2721642 * Q.z_dr_0_0p05 + 0.2302238
    if Q.girth2_top20 < 0.00287991:
        z += -5.652959 * Q.girth2_top20 + 0.01628002
    if Q.z_dr_0p1_0p2 < 0.06472584:
        z += -9.011945 * Q.z_dr_0p1_0p2 + 0.8727776
    if 0.06472584 <= Q.z_dr_0p1_0p2 < 0.1203437:
        z += -5.20466 * Q.z_dr_0p1_0p2 + 0.6263479
    if Q.girth2_top5 < 0.002270363:
        z += 43.84843 * Q.girth2_top5 + 0.3350694
    if 0.002270363 <= Q.girth2_top5 < 0.008329695:
        z += -71.72758 * Q.girth2_top5 + 0.5974689
    if Q.girth2_top2 < 0.0005124533:
        z += 51.56442 * Q.girth2_top2 - 0.2055788
    if 0.0005124533 <= Q.girth2_top2 < 0.001056655:
        z += 329.2059 * Q.girth2_top2 - 0.3478571
    if 0.8976117 <= Q.psi_0p1 < 0.9371031:
        z += -5.917023 * Q.psi_0p1 + 5.311189
    if Q.psi_0p1 >= 0.9371031:
        z += -11.74914 * Q.psi_0p1 + 10.77648
    if Q.sum_pt < 986.0565:
        z += 0.007143296 * Q.sum_pt - 7.113843
    if 986.0565 <= Q.sum_pt < 1002.379:
        z += 0.00429784 * Q.sum_pt - 4.308062
    if Q.log_sum_pt < 6.903423:
        z += -1.522252 * Q.log_sum_pt + 11.19402
    if 6.903423 <= Q.log_sum_pt < 7.017258:
        z += -6.019854 * Q.log_sum_pt + 42.24286
    if Q.psi_0p3 >= 0.9896594:
        z += -12.16826 * Q.psi_0p3 + 12.04243
    if 0.1666442 <= Q.sj2_dr < 0.2232169:
        z += -1.180361 * Q.sj2_dr + 0.1967003
    if 0.2232169 <= Q.sj2_dr < 0.2971569:
        z += 2.244199 * Q.sj2_dr - 0.5677193
    if Q.sj2_dr >= 0.2971569:
        z += -0.04730785 * Q.sj2_dr + 0.1132177
    if Q.girth2_top10 < 0.007678544:
        z += -63.02381 * Q.girth2_top10 + 0.37832
    if 0.007678544 <= Q.girth2_top10 < 0.008956554:
        z += 82.63715 * Q.girth2_top10 - 0.7401441
    if Q.lam1 < 0.004673423:
        z += -65.86295 * Q.lam1 - 0.172046
    if 0.004673423 <= Q.lam1 < 0.01174405:
        z += 67.86546 * Q.lam1 - 0.7970154
    if Q.lam1 >= 0.01649354:
        z += -25.90097 * Q.lam1 + 0.4271988
    if Q.n_dr_0p1_0p2 < 13.0:
        z += -0.02693897 * Q.n_dr_0p1_0p2 + 0.3502066
    if Q.D2 < 1.976207:
        z += 0.2676436 * Q.D2 - 0.5289193
    if Q.lam2 < 0.001776308:
        z += -91.18839 * Q.lam2 + 0.1619787
    if Q.tau1 < 0.0705748:
        z += 16.21921 * Q.tau1 - 1.144668
    if Q.sum_pt_top20 < 1017.778:
        z += 0.001064869 * Q.sum_pt_top20 - 1.0838
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 17.56262 * Q.mass_over_sum_pt - 3.001103
    if Q.z_dr_0p2_0p4 < 0.002288114:
        z += 111.2228 * Q.z_dr_0p2_0p4 - 0.2544906
    if 9.0 <= Q.n_dr_0p2_0p4 < 26.0:
        z += -0.01270826 * Q.n_dr_0p2_0p4 + 0.1143743
    if Q.n_dr_0p2_0p4 >= 26.0:
        z += -0.0456762 * Q.n_dr_0p2_0p4 + 0.9715407
    if Q.sd_mass < 45.595:
        z += 0.001877228 * Q.sd_mass + 0.2892906
    if 45.595 <= Q.sd_mass < 79.18312:
        z += -0.01116117 * Q.sd_mass + 0.8837766
    if Q.sj3_pair_mass_max < 35.30013:
        z += -0.01111791 * Q.sj3_pair_mass_max + 0.3924637
    if Q.mass_top50 < 52.15154:
        z += 0.001609735 * Q.mass_top50 - 0.4048874
    if 52.15154 <= Q.mass_top50 < 71.79516:
        z += 0.01633799 * Q.mass_top50 - 1.172989
    if Q.mass_top50 >= 97.93004:
        z += -0.01117207 * Q.mass_top50 + 1.094081
    if Q.sum_pt_top40 < 935.8189:
        z += 0.0007603425 * Q.sum_pt_top40 - 1.077725
    if 935.8189 <= Q.sum_pt_top40 < 1018.698:
        z += 0.004418281 * Q.sum_pt_top40 - 4.500893
    if Q.dr_7 >= 0.2229947:
        z += 1.39352 * Q.dr_7 - 0.3107477
    if Q.sum_pt_top30 < 933.1875:
        z += -0.002483564 * Q.sum_pt_top30 + 2.317631
    if Q.sd_rg < 0.1690338:
        z += -0.2849307 * Q.sd_rg - 0.3358356
    if 0.1690338 <= Q.sd_rg < 0.2042612:
        z += -0.8189635 * Q.sd_rg - 0.2455659
    if 0.2042612 <= Q.sd_rg < 0.3017146:
        z += 3.509294 * Q.sd_rg - 1.129661
    if Q.sd_rg >= 0.3017146:
        z += -0.5340328 * Q.sd_rg + 0.09026961
    if Q.e2 >= 0.03029714:
        z += -29.29556 * Q.e2 + 0.8875718
    if Q.mass_top30 < 80.24626:
        z += -0.003062097 * Q.mass_top30 + 0.2730561
    if 80.24626 <= Q.mass_top30 < 89.17293:
        z += 0.008603503 * Q.mass_top30 - 0.6630646
    if Q.mass_top30 >= 89.17293:
        z += 0.0116656 * Q.mass_top30 - 0.9361207
    if Q.N2 < 0.4226723:
        z += 1.005851 * Q.N2 - 0.4251453
    if Q.e3 >= 0.0001841806:
        z += 186.884 * Q.e3 - 0.03442042
    if Q.e2_sq < 0.006399858:
        z += 100.802 * Q.e2_sq - 0.6451184
    if Q.n_pt_above_5 < 42.0:
        z += -0.003953662 * Q.n_pt_above_5 + 0.1660538
    if Q.n_pt_above_10 < 22.0:
        z += 0.01179979 * Q.n_pt_above_10 - 0.2595954
    if Q.psi_0p2 >= 0.8706159:
        z += 0.772256 * Q.psi_0p2 - 0.6723384
    if Q.max_dr < 0.2982:
        z += 2.142217 * Q.max_dr - 0.6388092
    if Q.girth < 0.08589404:
        z += -15.64075 * Q.girth + 1.343447
    if Q.LHA < 0.2941033:
        z += 3.598741 * Q.LHA - 1.058401
    if Q.dr_3 < 0.05268713:
        z += 1.584874 * Q.dr_3 - 0.08350244
    if Q.dr_13 >= 0.2103034:
        z += 2.494017 * Q.dr_13 - 0.5245003
    if Q.sj2_zsoft < 0.2416266:
        z += -0.7190626 * Q.sj2_zsoft + 0.1737447
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.sd_mass < 76.29481:
        z += -0.00362893 * (0.1203437 - Q.z_dr_0p1_0p2) * (76.29481 - Q.sd_mass)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.planar_flow < 0.777038:
        z += 3.999968 * (0.1203437 - Q.z_dr_0p1_0p2) * (0.777038 - Q.planar_flow)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_particles < 58.0:
        z += -0.02651206 * (0.1203437 - Q.z_dr_0p1_0p2) * (58.0 - Q.n_particles)
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 949.9169:
        z += 0.01311996 * (0.8459004 - Q.z_dr_0_0p05) * (949.9169 - Q.sum_pt)
    if Q.girth2_top5 < 0.008329695 and Q.psi_0p2 < 0.948102:
        z += -183.3139 * (0.008329695 - Q.girth2_top5) * (0.948102 - Q.psi_0p2)
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 1260.541:
        z += -0.0006786493 * (0.8459004 - Q.z_dr_0_0p05) * (1260.541 - Q.sum_pt)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_dr_0p2_0p4 > 5.0:
        z += -0.1948522 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p2_0p4 - 5.0)
    if Q.girth2_top20 < 0.00287991 and Q.sum_pt > 1167.447:
        z += 0.01271286 * (0.00287991 - Q.girth2_top20) * (Q.sum_pt - 1167.447)
    if Q.sj2_dr > 0.2232169 and Q.C2_b2 < 0.04008677:
        z += 136.6107 * (Q.sj2_dr - 0.2232169) * (0.04008677 - Q.C2_b2)
    if Q.girth2_top5 < 0.008329695 and Q.z_6 < 0.04770182:
        z += 625.4411 * (0.008329695 - Q.girth2_top5) * (0.04770182 - Q.z_6)
    if Q.psi_0p3 > 0.9896594 and Q.sj2_mass2 < 14.45919:
        z += -2.246557 * (Q.psi_0p3 - 0.9896594) * (14.45919 - Q.sj2_mass2)
    if Q.girth2_top5 < 0.008329695 and Q.sj3_mass1 > 5.112677:
        z += -0.5579213 * (0.008329695 - Q.girth2_top5) * (Q.sj3_mass1 - 5.112677)
    if Q.n_dr_0p1_0p2 < 13.0 and Q.absphi_0 < 0.02970886:
        z += -0.7829884 * (13.0 - Q.n_dr_0p1_0p2) * (0.02970886 - Q.absphi_0)
    if Q.girth2_top5 < 0.008329695 and Q.n_real_top40 > 22.0:
        z += -0.9855042 * (0.008329695 - Q.girth2_top5) * (Q.n_real_top40 - 22.0)
    if Q.girth2_top20 < 0.00287991 and Q.max_dr < 0.4357228:
        z += 431.3841 * (0.00287991 - Q.girth2_top20) * (0.4357228 - Q.max_dr)
    if Q.girth2_top5 < 0.008329695 and Q.sj3_pairmin_over_m > 0.09540583:
        z += -51.52504 * (0.008329695 - Q.girth2_top5) * (Q.sj3_pairmin_over_m - 0.09540583)
    if Q.girth2_top5 < 0.008329695 and Q.dr0_8 > 0.2441824:
        z += 122.7474 * (0.008329695 - Q.girth2_top5) * (Q.dr0_8 - 0.2441824)
    if Q.n_dr_0p1_0p2 < 13.0 and Q.n_dr_0p05_0p1 > 8.0:
        z += -0.001390625 * (13.0 - Q.n_dr_0p1_0p2) * (Q.n_dr_0p05_0p1 - 8.0)
    if Q.lam2 < 0.001776308 and Q.dr_12 > 0.2053949:
        z += 479.3633 * (0.001776308 - Q.lam2) * (Q.dr_12 - 0.2053949)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_dr_0p4_up > 0.0:
        z += 0.3907259 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p4_up - 0.0)
    if Q.sd_mass < 79.18312 and Q.eta_2 > -0.03302612:
        z += -0.01037403 * (79.18312 - Q.sd_mass) * (Q.eta_2 - -0.03302612)
    if Q.sd_mass < 45.595 and Q.eta_1 < -0.009371567:
        z += 0.1993827 * (45.595 - Q.sd_mass) * (-0.009371567 - Q.eta_1)
    if Q.sj2_dr > 0.2232169 and Q.soft5_z < 0.0009102912:
        z += -997.4606 * (Q.sj2_dr - 0.2232169) * (0.0009102912 - Q.soft5_z)
    if Q.psi_0p3 > 0.9896594 and Q.pt_2 < 137.5:
        z += 0.1148191 * (Q.psi_0p3 - 0.9896594) * (137.5 - Q.pt_2)
    if Q.sd_mass < 79.18312 and Q.pt_10 > 8.304297:
        z += -4.427111e-05 * (79.18312 - Q.sd_mass) * (Q.pt_10 - 8.304297)
    if Q.sj2_dr > 0.2232169 and Q.absphi_9 < 0.02334595:
        z += -86.87531 * (Q.sj2_dr - 0.2232169) * (0.02334595 - Q.absphi_9)
    if Q.girth2_top5 < 0.008329695 and Q.dr1_11 > 0.2621232:
        z += 703.598 * (0.008329695 - Q.girth2_top5) * (Q.dr1_11 - 0.2621232)
    if Q.girth2_top2 < 0.001056655 and Q.dr1_11 > 0.1710892:
        z += -2169.737 * (0.001056655 - Q.girth2_top2) * (Q.dr1_11 - 0.1710892)
    if Q.D2 < 1.976207 and Q.dr_11 < 0.2535773:
        z += -0.2070311 * (1.976207 - Q.D2) * (0.2535773 - Q.dr_11)
    if Q.dr_7 > 0.2229947 and Q.zdr_10 < 0.002337294:
        z += 1383.481 * (Q.dr_7 - 0.2229947) * (0.002337294 - Q.zdr_10)
    if Q.lam2 < 0.001776308 and Q.dr1_3 > 0.1637906:
        z += -1118.039 * (0.001776308 - Q.lam2) * (Q.dr1_3 - 0.1637906)
    if Q.lam2 < 0.001776308 and Q.soft8_dr0 > 0.0373368:
        z += 241.8389 * (0.001776308 - Q.lam2) * (Q.soft8_dr0 - 0.0373368)
    if Q.lam2 < 0.001776308 and Q.ptdr0_10 > 3.719859:
        z += 65.00889 * (0.001776308 - Q.lam2) * (Q.ptdr0_10 - 3.719859)
    if Q.e2 > 0.03029714 and Q.sj3_z2 < 0.3166869:
        z += -65.06016 * (Q.e2 - 0.03029714) * (0.3166869 - Q.sj3_z2)
    if Q.log_sum_pt < 7.017258 and Q.n_dr_0p05_0p1 > 6.0:
        z += -0.01340625 * (7.017258 - Q.log_sum_pt) * (Q.n_dr_0p05_0p1 - 6.0)
    if Q.n_pt_above_10 < 22.0 and Q.soft1_dr < 0.3493565:
        z += 0.009448993 * (22.0 - Q.n_pt_above_10) * (0.3493565 - Q.soft1_dr)
    if Q.psi_0p3 > 0.9896594 and Q.soft3_z < 0.002741632:
        z += -2600.746 * (Q.psi_0p3 - 0.9896594) * (0.002741632 - Q.soft3_z)
    if Q.psi_0p3 > 0.9896594 and Q.ptdr0_8 > 3.519576:
        z += 3.025704 * (Q.psi_0p3 - 0.9896594) * (Q.ptdr0_8 - 3.519576)
    if Q.psi_0p2 > 0.8706159 and Q.zdr_13 < 0.002782871:
        z += -64.22034 * (Q.psi_0p2 - 0.8706159) * (0.002782871 - Q.zdr_13)
    if Q.log_sum_pt < 7.017258 and Q.sj3_mass2 < 13.23436:
        z += 0.01783774 * (7.017258 - Q.log_sum_pt) * (13.23436 - Q.sj3_mass2)
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    return [B[c] + sum(h[j] * W[j][c] for j in range(len(h))) for c in range(5)]


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
