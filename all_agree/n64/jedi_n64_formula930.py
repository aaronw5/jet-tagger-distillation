"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; all observables, tuned for agreement), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

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
    z = 1.225391
    if Q.mass < 53.87362:
        z += 0.02778109 * Q.mass - 2.960884
    if 53.87362 <= Q.mass < 74.25181:
        z += 0.03103724 * Q.mass - 3.136304
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.01191018 * Q.mass - 1.716085
    if 78.26182 <= Q.mass < 89.74183:
        z += -0.1181747 * Q.mass + 8.464592
    if 89.74183 <= Q.mass < 91.19:
        z += -0.02378673 * Q.mass - 0.005955855
    if 91.19 <= Q.mass < 92.85979:
        z += 0.08212846 * Q.mass - 9.664362
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.06760749 * Q.mass - 8.315947
    if Q.mass >= 101.0497:
        z += 0.03657024 * Q.mass - 5.179643
    if Q.girth2_top20 < 0.005312783:
        z += 30.54831 * Q.girth2_top20 - 0.2023651
    if 0.005312783 <= Q.girth2_top20 < 0.006374178:
        z += 102.7026 * Q.girth2_top20 - 0.5857054
    if 0.006374178 <= Q.girth2_top20 < 0.007538019:
        z += -59.23445 * Q.girth2_top20 + 0.4465104
    if Q.sum_pt < 972.0419:
        z += 0.005764766 * Q.sum_pt - 5.728331
    if 972.0419 <= Q.sum_pt < 1012.673:
        z += 0.00306999 * Q.sum_pt - 3.108895
    if Q.psi_0p3 >= 0.9956185:
        z += 55.72528 * Q.psi_0p3 - 55.48112
    if Q.log_sum_pt < 6.98945:
        z += -3.073754 * Q.log_sum_pt + 21.52554
    if 6.98945 <= Q.log_sum_pt < 7.017258:
        z += -1.499121 * Q.log_sum_pt + 10.51972
    if Q.mass_top30 < 80.4:
        z += -0.007299152 * Q.mass_top30 + 0.5868518
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.03027653 * Q.n_dr_0p2_0p4 + 0.3982372
    if 11.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += -0.01629883 * Q.n_dr_0p2_0p4 + 0.2444825
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += 3.70001 * Q.z_dr_0p2_0p4 - 0.2615946
    if 0.09122568 <= Q.z_dr_0p2_0p4 < 0.1291856:
        z += -2.000567 * Q.z_dr_0p2_0p4 + 0.2584445
    if Q.girth2_top30 < 0.006363916:
        z += 108.2197 * Q.girth2_top30 - 0.5468827
    if 0.006363916 <= Q.girth2_top30 < 0.007856958:
        z += -94.98602 * Q.girth2_top30 + 0.7463011
    if Q.lam1 < 0.005913555:
        z += 88.09339 * Q.lam1 - 0.5209451
    if Q.mass_over_sum_pt_sq < 0.004754444:
        z += -1050.829 * Q.mass_over_sum_pt_sq + 7.146595
    if 0.004754444 <= Q.mass_over_sum_pt_sq < 0.006938798:
        z += -826.554 * Q.mass_over_sum_pt_sq + 6.080291
    if 0.006938798 <= Q.mass_over_sum_pt_sq < 0.007873266:
        z += -369.194 * Q.mass_over_sum_pt_sq + 2.906762
    if Q.tau1 < 0.0705748:
        z += 23.54353 * Q.tau1 - 1.66158
    if Q.e2_sq < 0.00616708:
        z += 520.1777 * Q.e2_sq - 3.207977
    if Q.e2 < 0.01879315:
        z += -25.00436 * Q.e2 + 0.4699107
    if 71.79516 <= Q.mass_top50 < 82.04491:
        z += 0.02059603 * Q.mass_top50 - 1.478695
    if Q.mass_top50 >= 82.04491:
        z += -0.04150768 * Q.mass_top50 + 3.616598
    if Q.z_top50_slots < 0.9906378:
        z += 16.38618 * Q.z_top50_slots - 16.23277
    if Q.soft5_z < 0.002788182:
        z += 82.20515 * Q.soft5_z - 0.2292029
    if Q.psi_0p1 >= 0.9184255:
        z += -2.192949 * Q.psi_0p1 + 2.01406
    if Q.z_dr_0p05_0p1 >= 0.7108211:
        z += 1.139715 * Q.z_dr_0p05_0p1 - 0.8101332
    if Q.sum_pt_top40 < 858.8262:
        z += 0.004113939 * Q.sum_pt_top40 - 4.25576
    if 858.8262 <= Q.sum_pt_top40 < 1069.671:
        z += 0.00342717 * Q.sum_pt_top40 - 3.665945
    if Q.sum_pt_top20 < 846.1934:
        z += -0.0004440824 * Q.sum_pt_top20 + 0.3757796
    if Q.mass_top40 < 80.89043:
        z += 0.006229111 * Q.mass_top40 - 0.5038755
    if Q.e3 < 0.0005178279:
        z += -284.2399 * Q.e3 + 0.1471873
    if Q.C2 < 0.05602756:
        z += 2.090585 * Q.C2 - 0.1171304
    if Q.lam2 < 0.0006154841:
        z += -323.7275 * Q.lam2 + 0.1992491
    if Q.sum_pt_top50 < 1156.659:
        z += -0.002218857 * Q.sum_pt_top50 + 2.566462
    if Q.dr_2 < 0.03843804:
        z += -3.924509 * Q.dr_2 + 0.1508504
    if Q.sum_pt < 1012.673 and Q.M3 < 0.03457336:
        z += -0.03204689 * (1012.673 - Q.sum_pt) * (0.03457336 - Q.M3)
    if Q.lam1 < 0.005913555 and Q.z_7 < 0.03230249:
        z += 1063.084 * (0.005913555 - Q.lam1) * (0.03230249 - Q.z_7)
    if Q.log_sum_pt < 6.98945 and Q.sum_pt_top50 > 959.0957:
        z += 0.09714462 * (6.98945 - Q.log_sum_pt) * (Q.sum_pt_top50 - 959.0957)
    if Q.z_dr_0p2_0p4 < 0.09122568 and Q.pt_2 < 118.5:
        z += 0.02314869 * (0.09122568 - Q.z_dr_0p2_0p4) * (118.5 - Q.pt_2)
    if Q.soft5_z < 0.002788182 and Q.n_real_top30 < 30.0:
        z += 4.564719 * (0.002788182 - Q.soft5_z) * (30.0 - Q.n_real_top30)
    if Q.mass < 101.0497 and Q.eta_1 > 0.0001021922:
        z += -0.05395177 * (101.0497 - Q.mass) * (Q.eta_1 - 0.0001021922)
    if Q.psi_0p3 > 0.9956185 and Q.dr_max_012 > 0.06112084:
        z += -156.1708 * (Q.psi_0p3 - 0.9956185) * (Q.dr_max_012 - 0.06112084)
    if Q.girth2_top30 < 0.007856958 and Q.ptdr0_4 > 8.939062:
        z += -0.1761077 * (0.007856958 - Q.girth2_top30) * (Q.ptdr0_4 - 8.939062)
    if Q.log_sum_pt < 7.017258 and Q.soft4_dr < 0.199317:
        z += 2.446474 * (7.017258 - Q.log_sum_pt) * (0.199317 - Q.soft4_dr)
    if Q.mass_over_sum_pt_sq < 0.007873266 and Q.pt_10 < 34.0625:
        z += 1.200866 * (0.007873266 - Q.mass_over_sum_pt_sq) * (34.0625 - Q.pt_10)
    if Q.sum_pt < 1012.673 and Q.C3 < 0.0223982:
        z += -0.1043126 * (1012.673 - Q.sum_pt) * (0.0223982 - Q.C3)
    if Q.sum_pt < 972.0419 and Q.z_4 < 0.05554124:
        z += -0.06120415 * (972.0419 - Q.sum_pt) * (0.05554124 - Q.z_4)
    if Q.psi_0p3 > 0.9956185 and Q.pt_4 > 65.1875:
        z += -1.856289 * (Q.psi_0p3 - 0.9956185) * (Q.pt_4 - 65.1875)
    if Q.C2 < 0.05602756 and Q.phi_10 > 0.0647583:
        z += 197.3456 * (0.05602756 - Q.C2) * (Q.phi_10 - 0.0647583)
    if Q.mass < 101.0497 and Q.eta_0 < 0.02980347:
        z += 0.08899637 * (101.0497 - Q.mass) * (0.02980347 - Q.eta_0)
    if Q.psi_0p1 > 0.9184255 and Q.eta_4 < -0.02580643:
        z += 61.69694 * (Q.psi_0p1 - 0.9184255) * (-0.02580643 - Q.eta_4)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.6060924
    if Q.n_particles >= 38.0:
        z += 0.0930585 * Q.n_particles - 3.536223
    if Q.log_sum_pt < 6.893714:
        z += 9.251128 * Q.log_sum_pt - 66.04654
    if 6.893714 <= Q.log_sum_pt < 6.910131:
        z += 36.51656 * Q.log_sum_pt - 254.0066
    if 6.910131 <= Q.log_sum_pt < 6.959294:
        z += 55.68062 * Q.log_sum_pt - 386.4328
    if 6.959294 <= Q.log_sum_pt < 6.98945:
        z += 45.65388 * Q.log_sum_pt - 316.6538
    if 6.98945 <= Q.log_sum_pt < 7.139296:
        z += 29.26467 * Q.log_sum_pt - 202.1022
    if Q.log_sum_pt >= 7.139296:
        z += 20.01354 * Q.log_sum_pt - 136.0556
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.009263286 * Q.sum_pt_top50 + 8.884378
    if Q.psi_0p3 >= 0.9980008:
        z += -268.8486 * Q.psi_0p3 + 268.3111
    if Q.sum_pt_top2 < 689.25:
        z += -0.001944912 * Q.sum_pt_top2 + 1.340531
    if Q.z_top30_slots >= 0.9341838:
        z += 1.817088 * Q.z_top30_slots - 1.697494
    if Q.mass_top20 < 47.88842:
        z += 0.01553317 * Q.mass_top20 - 0.7438589
    if Q.sj3_mass1 < 32.50209:
        z += 0.02713516 * Q.sj3_mass1 - 0.8819494
    if Q.sum_pt_top40 < 1069.671:
        z += -0.008013516 * Q.sum_pt_top40 + 8.571827
    if Q.girth2_top15 < 0.003270031:
        z += -178.5764 * Q.girth2_top15 + 0.5839503
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.1306933 * Q.n_dr_0p2_0p4 - 0.9148528
    if Q.girth2_top3 < 0.0005522528:
        z += -1219.954 * Q.girth2_top3 + 0.673723
    if Q.M3 < 0.03187688:
        z += 9.749037 * Q.M3 - 0.3107689
    if Q.sj2_mass1 < 30.26161:
        z += 0.03375604 * Q.sj2_mass1 - 1.021512
    if Q.soft5_z < 0.001594761:
        z += -209.6611 * Q.soft5_z + 0.3343594
    if Q.pt_9 < 31.35938:
        z += 0.04308392 * Q.pt_9 - 1.351085
    if Q.lam1 < 0.004673423:
        z += -256.8841 * Q.lam1 + 1.200528
    if Q.mass < 120.6:
        z += 0.01879747 * Q.mass - 2.266975
    if Q.girth2_top30 < 0.02412652:
        z += -73.79081 * Q.girth2_top30 + 1.780316
    if Q.N2 >= 0.4678622:
        z += 2.094009 * Q.N2 - 0.9797079
    if Q.tau3 < 0.009068077:
        z += 74.1425 * Q.tau3 - 0.6723299
    if Q.D3 < 0.1416054:
        z += -3.162656 * Q.D3 + 0.4478492
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -5.153301 * Q.z_dr_0_0p05 + 4.529267
    if Q.mass_top30 < 42.41192:
        z += -0.03343432 * Q.mass_top30 + 1.418013
    if Q.n_dr_0_0p05 < 12.0:
        z += -0.01542708 * Q.n_dr_0_0p05 + 0.1851249
    if Q.mass_top10 >= 56.92192:
        z += 0.009281542 * Q.mass_top10 - 0.5283232
    z += -0.3776881 * Q.absphi_0
    if Q.sj3_dr_max >= 0.3276439:
        z += 1.865344 * Q.sj3_dr_max - 0.6111684
    if Q.soft1_pt < 1.521582:
        z += -0.01659648 * Q.soft1_pt + 0.02525291
    if Q.sum_pt < 1017.435:
        z += -0.01274548 * Q.sum_pt + 12.96769
    if Q.sum_pt_top30 >= 1191.938:
        z += 0.00744314 * Q.sum_pt_top30 - 8.87176
    if Q.z_top20_slots >= 0.8965411:
        z += 2.573305 * Q.z_top20_slots - 2.307074
    if Q.sum_pt_top5 >= 430.75:
        z += -0.0007100159 * Q.sum_pt_top5 + 0.3058394
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.06723274 * Q.n_dr_0p1_0p2 - 0.5378619
    if Q.pt_entropy >= 2.07371:
        z += 0.7372996 * Q.pt_entropy - 1.528945
    if Q.n_particles > 38.0 and Q.mass_top15 < 57.87349:
        z += 0.0007667565 * (Q.n_particles - 38.0) * (57.87349 - Q.mass_top15)
    if Q.z_top30_slots > 0.9341838 and Q.max_pair_mass > 13.04793:
        z += 0.2906851 * (Q.z_top30_slots - 0.9341838) * (Q.max_pair_mass - 13.04793)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.1119555:
        z += 0.379318 * (Q.n_particles - 38.0) * (0.1119555 - Q.dr_0)
    if Q.mass_top20 < 47.88842 and Q.n_real_top40 > 29.0:
        z += 0.004858119 * (47.88842 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 38.0 and Q.soft1_pt < 2.275391:
        z += -0.0275739 * (Q.n_particles - 38.0) * (2.275391 - Q.soft1_pt)
    if Q.z_top30_slots > 0.9341838 and Q.C2 < 0.07279889:
        z += 210.7623 * (Q.z_top30_slots - 0.9341838) * (0.07279889 - Q.C2)
    if Q.sj3_mass1 < 32.50209 and Q.sj3_mass2 < 18.68222:
        z += -0.002265619 * (32.50209 - Q.sj3_mass1) * (18.68222 - Q.sj3_mass2)
    if Q.girth2_top15 < 0.003270031 and Q.psi_0p3 > 0.9973959:
        z += 26563.93 * (0.003270031 - Q.girth2_top15) * (Q.psi_0p3 - 0.9973959)
    if Q.M3 < 0.03187688 and Q.M2 > 0.05568888:
        z += -176.1046 * (0.03187688 - Q.M3) * (Q.M2 - 0.05568888)
    if Q.n_particles > 38.0 and Q.dr_1 < 0.1611545:
        z += 0.2375987 * (Q.n_particles - 38.0) * (0.1611545 - Q.dr_1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 > 7.407874:
        z += 0.8021571 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_3 - 7.407874)
    if Q.M3 < 0.03187688 and Q.psi_0p3 < 1.0:
        z += 99.73737 * (0.03187688 - Q.M3) * (1.0 - Q.psi_0p3)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_4 > 6.984554:
        z += 0.1007016 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_4 - 6.984554)
    if Q.sum_pt_top2 < 689.25 and Q.tau43 < 0.9624339:
        z += -0.0003767863 * (689.25 - Q.sum_pt_top2) * (0.9624339 - Q.tau43)
    if Q.pt_9 < 31.35938 and Q.dr1_12 < 0.3241858:
        z += -0.007834151 * (31.35938 - Q.pt_9) * (0.3241858 - Q.dr1_12)
    if Q.log_sum_pt < 7.139296 and Q.pt1_dr01 < 5.351077:
        z += -0.217072 * (7.139296 - Q.log_sum_pt) * (5.351077 - Q.pt1_dr01)
    if Q.pt_9 < 31.35938 and Q.pair_mass_0_13 < 11.29505:
        z += 0.0005070287 * (31.35938 - Q.pt_9) * (11.29505 - Q.pair_mass_0_13)
    if Q.n_particles > 38.0 and Q.sj3_dr12 < 0.2313408:
        z += 0.1302419 * (Q.n_particles - 38.0) * (0.2313408 - Q.sj3_dr12)
    if Q.sum_pt_top5 > 430.75 and Q.eta_0 < 0.02980347:
        z += -0.008859698 * (Q.sum_pt_top5 - 430.75) * (0.02980347 - Q.eta_0)
    if Q.girth2_top30 < 0.02412652 and Q.zdr_14 > 0.0025721:
        z += 10656.79 * (0.02412652 - Q.girth2_top30) * (Q.zdr_14 - 0.0025721)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.soft10_abseta < 0.002968979:
        z += -11.3388 * (7.0 - Q.n_dr_0p2_0p4) * (0.002968979 - Q.soft10_abseta)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.4112215
    if 6.903423 <= Q.log_sum_pt < 6.959294:
        z += 0.4399849 * Q.log_sum_pt - 3.037402
    if 6.959294 <= Q.log_sum_pt < 7.062574:
        z += -7.652353 * Q.log_sum_pt + 53.27955
    if Q.log_sum_pt >= 7.062574:
        z += -9.255836 * Q.log_sum_pt + 64.60427
    if Q.sum_pt < 972.0419:
        z += 0.002344293 * Q.sum_pt - 3.053665
    if 972.0419 <= Q.sum_pt < 1017.435:
        z += 0.002686019 * Q.sum_pt - 3.385837
    if 1017.435 <= Q.sum_pt < 1115.723:
        z += 0.01197692 * Q.sum_pt - 12.83872
    if 1115.723 <= Q.sum_pt < 1260.541:
        z += 0.007700818 * Q.sum_pt - 8.067777
    if Q.sum_pt >= 1260.541:
        z += 0.005014799 * Q.sum_pt - 4.68194
    if Q.n_particles < 51.0:
        z += 0.00244229 * Q.n_particles - 0.1245568
    if Q.sum_pt_top20 >= 1129.275:
        z += 0.0007608865 * Q.sum_pt_top20 - 0.8592501
    if Q.sum_pt_top50 < 988.4554:
        z += -0.003486428 * Q.sum_pt_top50 + 3.532342
    if 988.4554 <= Q.sum_pt_top50 < 1048.098:
        z += -0.003774691 * Q.sum_pt_top50 + 3.817277
    if 1048.098 <= Q.sum_pt_top50 < 1078.994:
        z += 0.003653261 * Q.sum_pt_top50 - 3.967946
    if 1078.994 <= Q.sum_pt_top50 < 1156.659:
        z += -0.0002882626 * Q.sum_pt_top50 + 0.2849348
    if Q.sum_pt_top50 >= 1156.659:
        z += -4.928977e-05 * Q.sum_pt_top50 + 0.008524537
    if Q.z_top20_slots >= 0.7516206:
        z += 0.0118673 * Q.z_top20_slots - 0.008919706
    if Q.mass < 78.26182:
        z += 0.02702382 * Q.mass - 2.674805
    if 78.26182 <= Q.mass < 91.19:
        z += 0.03507536 * Q.mass - 3.304933
    if 91.19 <= Q.mass < 92.85979:
        z += 0.06372721 * Q.mass - 5.917695
    if Q.mass_top30 < 82.66587:
        z += -0.00654758 * Q.mass_top30 + 0.5412614
    if Q.mass_over_sum_pt < 0.09795415:
        z += -17.65884 * Q.mass_over_sum_pt + 1.729757
    if Q.sum_pt_top40 < 984.7009:
        z += 0.002202866 * Q.sum_pt_top40 - 2.188043
    if 984.7009 <= Q.sum_pt_top40 < 1018.698:
        z += 4.666811e-05 * Q.sum_pt_top40 - 0.06483298
    if 1018.698 <= Q.sum_pt_top40 < 1041.263:
        z += -0.004638414 * Q.sum_pt_top40 + 4.70785
    if 1041.263 <= Q.sum_pt_top40 < 1069.671:
        z += -0.002156198 * Q.sum_pt_top40 + 2.12321
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.001640622 * Q.sum_pt_top40 - 1.938139
    if Q.sum_pt_top30 < 966.0633:
        z += 0.001329435 * Q.sum_pt_top30 - 1.38929
    if 966.0633 <= Q.sum_pt_top30 < 996.8867:
        z += 0.003405567 * Q.sum_pt_top30 - 3.394964
    if Q.girth2_top30 < 0.004763596:
        z += 59.66061 * Q.girth2_top30 - 0.3710837
    if 0.004763596 <= Q.girth2_top30 < 0.006363916:
        z += 54.29203 * Q.girth2_top30 - 0.3455099
    if Q.psi_0p3 >= 0.9973959:
        z += -56.71088 * Q.psi_0p3 + 56.5632
    if Q.lam1 < 0.01174405:
        z += -14.88319 * Q.lam1 + 0.174789
    if Q.sj3_pair_mass_min >= 18.02031:
        z += 0.001611576 * Q.sj3_pair_mass_min - 0.0290411
    if Q.pt_6 < 56.53125:
        z += 0.0002661585 * Q.pt_6 - 0.01504627
    if Q.mass_top50 < 92.16545:
        z += -0.008175011 * Q.mass_top50 + 0.7534536
    if Q.log_sum_pt > 6.903423 and Q.mass_top40 > 77.93668:
        z += -0.02772078 * (Q.log_sum_pt - 6.903423) * (Q.mass_top40 - 77.93668)
    if Q.log_sum_pt > 6.903423 and Q.girth2_top15 < 0.02146578:
        z += 565.1508 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.girth2_top15)
    if Q.log_sum_pt > 6.903423 and Q.e3 < 1.433504e-05:
        z += -94660.29 * (Q.log_sum_pt - 6.903423) * (1.433504e-05 - Q.e3)
    if Q.log_sum_pt > 7.062574 and Q.girth2_top30 > 0.008376291:
        z += 63.37053 * (Q.log_sum_pt - 7.062574) * (Q.girth2_top30 - 0.008376291)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p3 > 0.9299135:
        z += -42.09064 * (Q.log_sum_pt - 6.903423) * (Q.psi_0p3 - 0.9299135)
    if Q.sum_pt > 1017.435 and Q.C2 > 0.02207346:
        z += 0.00736845 * (Q.sum_pt - 1017.435) * (Q.C2 - 0.02207346)
    if Q.log_sum_pt > 6.903423 and Q.soft5_z > 0.0005078411:
        z += 145.554 * (Q.log_sum_pt - 6.903423) * (Q.soft5_z - 0.0005078411)
    if Q.sum_pt_top50 > 988.4554 and Q.girth2_top15 < 0.02146578:
        z += -0.4036413 * (Q.sum_pt_top50 - 988.4554) * (0.02146578 - Q.girth2_top15)
    if Q.sum_pt < 972.0419 and Q.e4 < 5.8505e-08:
        z += 31174.0 * (972.0419 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.sum_pt_top20 > 1129.275 and Q.n_dr_0_0p05 < 1.0:
        z += 0.003784423 * (Q.sum_pt_top20 - 1129.275) * (1.0 - Q.n_dr_0_0p05)
    if Q.sum_pt > 1115.723 and Q.mean_phi > 0.0004404746:
        z += -0.3201101 * (Q.sum_pt - 1115.723) * (Q.mean_phi - 0.0004404746)
    if Q.sum_pt > 1115.723 and Q.eta_5 < -0.1135254:
        z += 0.01130507 * (Q.sum_pt - 1115.723) * (-0.1135254 - Q.eta_5)
    if Q.sum_pt < 1260.541 and Q.n_real_top30 < 30.0:
        z += 0.0001224388 * (1260.541 - Q.sum_pt) * (30.0 - Q.n_real_top30)
    if Q.log_sum_pt > 7.062574 and Q.z_10 < 0.01898316:
        z += 234.4611 * (Q.log_sum_pt - 7.062574) * (0.01898316 - Q.z_10)
    if Q.log_sum_pt > 6.903423 and Q.z_10 < 0.01585274:
        z += 201.0273 * (Q.log_sum_pt - 6.903423) * (0.01585274 - Q.z_10)
    if Q.z_top20_slots > 0.7516206 and Q.sj3_mass1 > 6.519856:
        z += -0.003909299 * (Q.z_top20_slots - 0.7516206) * (Q.sj3_mass1 - 6.519856)
    if Q.sum_pt_top50 < 1048.098 and Q.max_dr < 0.3742561:
        z += 0.002119148 * (1048.098 - Q.sum_pt_top50) * (0.3742561 - Q.max_dr)
    if Q.sum_pt < 1260.541 and Q.max_dr < 0.397021:
        z += -0.003635347 * (1260.541 - Q.sum_pt) * (0.397021 - Q.max_dr)
    if Q.sum_pt_top30 < 996.8867 and Q.soft7_dr < 0.03275811:
        z += 0.07170291 * (996.8867 - Q.sum_pt_top30) * (0.03275811 - Q.soft7_dr)
    if Q.sum_pt_top20 > 1129.275 and Q.eta_1 > 0.08734131:
        z += -0.4197282 * (Q.sum_pt_top20 - 1129.275) * (Q.eta_1 - 0.08734131)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.05795646
    if Q.lam2 < 0.0006154841:
        z += -240.7153 * Q.lam2 + 0.1481564
    if Q.n_dr_0p2_0p4 < 1.0:
        z += 0.2638494 * Q.n_dr_0p2_0p4 + 0.1245443
    if 1.0 <= Q.n_dr_0p2_0p4 < 5.0:
        z += -0.09438878 * Q.n_dr_0p2_0p4 + 0.4827826
    if 5.0 <= Q.n_dr_0p2_0p4 < 8.0:
        z += -0.003612887 * Q.n_dr_0p2_0p4 + 0.0289031
    if Q.n_particles < 46.0:
        z += -0.02585803 * Q.n_particles + 1.189469
    if Q.psi_0p2 >= 0.9985434:
        z += 238.3975 * Q.psi_0p2 - 238.0502
    if Q.tau21 < 0.3861957:
        z += 1.297527 * Q.tau21 - 0.5010994
    if Q.D2 < 2.410481:
        z += 0.01870767 * Q.D2 - 0.04509449
    if 0.9980008 <= Q.psi_0p3 < 0.9995915:
        z += 79.50938 * Q.psi_0p3 - 79.35042
    if Q.psi_0p3 >= 0.9995915:
        z += -198.3407 * Q.psi_0p3 + 198.3861
    if Q.mass_over_sum_pt < 0.08665515:
        z += -16.00962 * Q.mass_over_sum_pt + 1.387316
    if Q.z_top30_slots >= 0.9734886:
        z += -1.714092 * Q.z_top30_slots + 1.668649
    if Q.girth2_top40 < 0.006026828:
        z += -73.9397 * Q.girth2_top40 + 0.8128298
    if 0.006026828 <= Q.girth2_top40 < 0.008840538:
        z += -130.5067 * Q.girth2_top40 + 1.153749
    if Q.zdr_0 < 0.003235754:
        z += 145.937 * Q.zdr_0 - 0.4722163
    if Q.mass_top50 < 79.21004:
        z += 0.004523801 * Q.mass_top50 - 0.3583304
    if Q.mass < 53.87362:
        z += -0.002236349 * Q.mass + 0.5960685
    if 53.87362 <= Q.mass < 86.4:
        z += -0.01185102 * Q.mass + 1.114046
    if 86.4 <= Q.mass < 92.85979:
        z += -0.01395051 * Q.mass + 1.295441
    if Q.girth2 < 0.009614971:
        z += 173.4718 * Q.girth2 - 1.667927
    if Q.n_dr_0p1_0p2 < 10.0:
        z += -0.03985934 * Q.n_dr_0p1_0p2 + 0.3985934
    if Q.n_for_90pct < 11.0:
        z += -0.05711654 * Q.n_for_90pct + 0.628282
    if Q.sj2_mass1 < 27.56535:
        z += -0.01540934 * Q.sj2_mass1 + 0.4247639
    if Q.mass_top40 < 80.4:
        z += 0.01334458 * Q.mass_top40 - 1.072904
    if Q.lam1 < 0.001260456:
        z += 162.841 * Q.lam1 - 0.2052538
    if Q.tau4 < 0.01626937:
        z += -12.61361 * Q.tau4 + 0.2052155
    if Q.z_dr_0p1_0p2 < 0.04748396:
        z += -0.3954713 * Q.z_dr_0p1_0p2 + 0.01877854
    if Q.max_dr < 0.2738063:
        z += -1.339251 * Q.max_dr + 0.3666954
    if Q.n_dr_0p2_0p4 < 5.0 and Q.sum_pt_top30 < 988.4375:
        z += 0.00025688 * (5.0 - Q.n_dr_0p2_0p4) * (988.4375 - Q.sum_pt_top30)
    if Q.n_particles < 46.0 and Q.mass_top15 > 57.87349:
        z += -0.0007397829 * (46.0 - Q.n_particles) * (Q.mass_top15 - 57.87349)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_dr_0p1_0p2 > 9.0:
        z += -0.008774188 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 9.0)
    if Q.n_particles < 46.0 and Q.sum_pt_top30 > 800.732:
        z += 6.743304e-05 * (46.0 - Q.n_particles) * (Q.sum_pt_top30 - 800.732)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.z_10 < 0.03447112:
        z += -1.728256 * (5.0 - Q.n_dr_0p2_0p4) * (0.03447112 - Q.z_10)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.eccentricity > 0.7333655:
        z += 0.1391537 * (5.0 - Q.n_dr_0p2_0p4) * (Q.eccentricity - 0.7333655)
    if Q.psi_0p3 > 0.9980008 and Q.soft10_z < 0.004962409:
        z += 1341.759 * (Q.psi_0p3 - 0.9980008) * (0.004962409 - Q.soft10_z)
    if Q.D2 < 2.410481 and Q.psi_0p3 > 0.9985421:
        z += 11.21202 * (2.410481 - Q.D2) * (Q.psi_0p3 - 0.9985421)
    if Q.lam2 < 0.0006154841 and Q.orientation_deg < -9.840088:
        z += 5.130991 * (0.0006154841 - Q.lam2) * (-9.840088 - Q.orientation_deg)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.zdr_5 < 0.007099471:
        z += -1.57136 * (5.0 - Q.n_dr_0p2_0p4) * (0.007099471 - Q.zdr_5)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.psi_0p1 < 0.9538343:
        z += 0.0411759 * (5.0 - Q.n_dr_0p2_0p4) * (0.9538343 - Q.psi_0p1)
    if Q.tau21 < 0.3861957 and Q.lam1 > 0.007671243:
        z += -95.0449 * (0.3861957 - Q.tau21) * (Q.lam1 - 0.007671243)
    if Q.z_top30_slots > 0.9734886 and Q.eta_9 < 0.1269531:
        z += 14.42339 * (Q.z_top30_slots - 0.9734886) * (0.1269531 - Q.eta_9)
    if Q.D2 < 2.410481 and Q.max_dr > 0.4357228:
        z += 1.057885 * (2.410481 - Q.D2) * (Q.max_dr - 0.4357228)
    if Q.D2 < 2.410481 and Q.soft5_dr < 0.136982:
        z += 0.5070696 * (2.410481 - Q.D2) * (0.136982 - Q.soft5_dr)
    if Q.z_top30_slots > 0.9734886 and Q.n_dr_0_0p05 > 9.0:
        z += 0.7057497 * (Q.z_top30_slots - 0.9734886) * (Q.n_dr_0_0p05 - 9.0)
    if Q.psi_0p3 > 0.9980008 and Q.soft4_dr > 0.08815263:
        z += 44.46008 * (Q.psi_0p3 - 0.9980008) * (Q.soft4_dr - 0.08815263)
    if Q.lam2 < 0.0006154841 and Q.soft10_dr < 0.1309949:
        z += 2173.169 * (0.0006154841 - Q.lam2) * (0.1309949 - Q.soft10_dr)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.z_top50_slots > 0.978741:
        z += 2.6538 * (8.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.978741)
    if Q.lam2 < 0.0006154841 and Q.n_dr_0p4_up < 1.0:
        z += 131.6653 * (0.0006154841 - Q.lam2) * (1.0 - Q.n_dr_0p4_up)
    if Q.girth2 < 0.009614971 and Q.eta_0 < 0.02980347:
        z += 437.972 * (0.009614971 - Q.girth2) * (0.02980347 - Q.eta_0)
    if Q.mass_top50 < 79.21004 and Q.mass_top3 > 8.921413:
        z += -0.004041693 * (79.21004 - Q.mass_top50) * (Q.mass_top3 - 8.921413)
    if Q.girth2_top40 < 0.008840538 and Q.ptdr0_6 > 5.648151:
        z += 13.82791 * (0.008840538 - Q.girth2_top40) * (Q.ptdr0_6 - 5.648151)
    if Q.sj2_mass1 < 27.56535 and Q.phi_13 < 0.1455078:
        z += 0.03857909 * (27.56535 - Q.sj2_mass1) * (0.1455078 - Q.phi_13)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.eta_3 < 0.03329468:
        z += 0.1441905 * (5.0 - Q.n_dr_0p2_0p4) * (0.03329468 - Q.eta_3)
    if Q.psi_0p3 > 0.9980008 and Q.eta_1 > -0.04302979:
        z += -435.578 * (Q.psi_0p3 - 0.9980008) * (Q.eta_1 - -0.04302979)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.soft5_z < 0.002788182:
        z += 9.226031 * (8.0 - Q.n_dr_0p2_0p4) * (0.002788182 - Q.soft5_z)
    if Q.psi_0p2 > 0.9985434 and Q.absphi_4 < 0.05004883:
        z += 693.3171 * (Q.psi_0p2 - 0.9985434) * (0.05004883 - Q.absphi_4)
    if Q.psi_0p3 > 0.9980008 and Q.sd_zg < 0.254323:
        z += 85.80668 * (Q.psi_0p3 - 0.9980008) * (0.254323 - Q.sd_zg)
    if Q.lam2 < 0.0006154841 and Q.soft8_dr > 0.04226709:
        z += -38.55854 * (0.0006154841 - Q.lam2) * (Q.soft8_dr - 0.04226709)
    if Q.max_dr < 0.2738063 and Q.phi_14 > 0.102478:
        z += -37.98276 * (0.2738063 - Q.max_dr) * (Q.phi_14 - 0.102478)
    if Q.psi_0p2 > 0.9985434 and Q.abseta_4 < 0.08575439:
        z += 227.9449 * (Q.psi_0p2 - 0.9985434) * (0.08575439 - Q.abseta_4)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.061886
    if Q.mass_top15 < 57.87349:
        z += -0.009262464 * Q.mass_top15 + 0.2802831
    if 57.87349 <= Q.mass_top15 < 69.02716:
        z += 0.006142333 * Q.mass_top15 - 0.6112462
    if 69.02716 <= Q.mass_top15 < 75.26407:
        z += 0.008888943 * Q.mass_top15 - 0.8008369
    if 75.26407 <= Q.mass_top15 < 91.19:
        z += 0.008276998 * Q.mass_top15 - 0.7547795
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.0366273 * Q.n_dr_0p2_0p4 + 0.8127302
    if 15.0 <= Q.n_dr_0p2_0p4 < 26.0:
        z += -0.02393824 * Q.n_dr_0p2_0p4 + 0.6223943
    if Q.planar_flow < 0.3036026:
        z += -0.67772 * Q.planar_flow + 0.2057576
    if Q.mass_top40 < 62.55:
        z += 0.01952509 * Q.mass_top40 - 2.332967
    if 62.55 <= Q.mass_top40 < 67.72643:
        z += 0.01358129 * Q.mass_top40 - 1.961182
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.03154269 * Q.mass_top40 - 3.177644
    if 83.32554 <= Q.mass_top40 < 163.2541:
        z += 0.00687279 * Q.mass_top40 - 1.122011
    if Q.e3 >= 0.0001841806:
        z += 850.4339 * Q.e3 - 0.1566334
    if Q.mass_top30 < 60.43821:
        z += 0.001543609 * Q.mass_top30 - 0.3001095
    if 60.43821 <= Q.mass_top30 < 91.69753:
        z += 0.009571487 * Q.mass_top30 - 0.7853001
    if 91.69753 <= Q.mass_top30 < 120.6:
        z += -0.003196324 * Q.mass_top30 + 0.3854767
    if Q.mass < 74.25181:
        z += 0.03555721 * Q.mass - 1.590374
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.1079289 * Q.mass - 6.9641
    if 78.26182 <= Q.mass < 80.78464:
        z += 0.02863113 * Q.mass - 0.7581146
    if 80.78464 <= Q.mass < 86.4:
        z += -0.01563127 * Q.mass + 2.817607
    if 86.4 <= Q.mass < 92.85979:
        z += -0.04608586 * Q.mass + 5.448883
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.06438246 * Q.mass + 7.147902
    if 101.0497 <= Q.mass < 120.6:
        z += -0.0209547 * Q.mass + 2.759539
    if 120.6 <= Q.mass < 143.7876:
        z += -0.01002273 * Q.mass + 1.441144
    if Q.psi_0p3 >= 0.9973959:
        z += 116.9821 * Q.psi_0p3 - 116.6775
    if Q.sj3_pair_mass_min >= 32.51366:
        z += 0.01216642 * Q.sj3_pair_mass_min - 0.3955749
    if Q.n_particles >= 22.0:
        z += -0.01801859 * Q.n_particles + 0.396409
    if Q.girth2_top15 < 0.004855289:
        z += 150.741 * Q.girth2_top15 - 0.7318908
    if 0.00727763 <= Q.girth2_top15 < 0.01563836:
        z += -66.60628 * Q.girth2_top15 + 0.4847358
    if Q.girth2_top15 >= 0.01563836:
        z += -3.098755 * Q.girth2_top15 - 0.5084173
    if Q.e2 >= 0.06524004:
        z += -30.67366 * Q.e2 + 2.001151
    if Q.sj2_dr >= 0.2232169:
        z += -1.585409 * Q.sj2_dr + 0.35389
    if 0.009606007 <= Q.e2_sq < 0.01396296:
        z += 147.196 * Q.e2_sq - 1.413966
    if Q.e2_sq >= 0.01396296:
        z += 209.5155 * Q.e2_sq - 2.284131
    if Q.log_sum_pt >= 6.97212:
        z += -1.783909 * Q.log_sum_pt + 12.43763
    if Q.sj2_mass1 < 21.52288:
        z += 0.01016922 * Q.sj2_mass1 - 0.218871
    if Q.pair_mass_0_12 >= 15.47191:
        z += -0.01796894 * Q.pair_mass_0_12 + 0.2780138
    if Q.sd_rg >= 0.2042612:
        z += -1.248781 * Q.sd_rg + 0.2550775
    if Q.girth >= 0.1207452:
        z += -14.30023 * Q.girth + 1.726684
    if Q.lam1 >= 0.008241985:
        z += -117.4448 * Q.lam1 + 0.9679785
    if Q.psi_0p1 < 0.9909875:
        z += -0.4670547 * Q.psi_0p1 + 0.4628454
    if Q.lam2 >= 0.001776308:
        z += -55.40713 * Q.lam2 + 0.09842015
    if Q.planar_flow < 0.3036026 and Q.n_dr_0p1_0p2 > 13.0:
        z += -0.101619 * (0.3036026 - Q.planar_flow) * (Q.n_dr_0p1_0p2 - 13.0)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.z_top50_slots < 1.0:
        z += -1.100145 * (15.0 - Q.n_dr_0p2_0p4) * (1.0 - Q.z_top50_slots)
    if Q.psi_0p3 > 0.9973959 and Q.n_dr_0_0p05 < 13.0:
        z += -14.75717 * (Q.psi_0p3 - 0.9973959) * (13.0 - Q.n_dr_0_0p05)
    if Q.n_particles > 22.0 and Q.pt_7 < 33.21875:
        z += -0.0005471435 * (Q.n_particles - 22.0) * (33.21875 - Q.pt_7)
    if Q.psi_0p3 > 0.9973959 and Q.zdr_0 < 0.005911134:
        z += -13958.37 * (Q.psi_0p3 - 0.9973959) * (0.005911134 - Q.zdr_0)
    if Q.mass_top40 < 83.32554 and Q.zdr_0 > 0.001901263:
        z += -1.809721 * (83.32554 - Q.mass_top40) * (Q.zdr_0 - 0.001901263)
    if Q.mass < 120.6 and Q.psi_0p3 < 0.9943058:
        z += -0.03705757 * (120.6 - Q.mass) * (0.9943058 - Q.psi_0p3)
    if Q.n_particles > 22.0 and Q.psi_0p3 > 0.9638082:
        z += 0.051581 * (Q.n_particles - 22.0) * (Q.psi_0p3 - 0.9638082)
    if Q.n_particles > 22.0 and Q.n_dr_0_0p05 > 10.0:
        z += 0.0006359515 * (Q.n_particles - 22.0) * (Q.n_dr_0_0p05 - 10.0)
    if Q.n_particles > 22.0 and Q.soft1_pt < 2.275391:
        z += 0.006772245 * (Q.n_particles - 22.0) * (2.275391 - Q.soft1_pt)
    if Q.mass < 101.0497 and Q.zdr_0 > 0.0008718296:
        z += 0.01400004 * (101.0497 - Q.mass) * (Q.zdr_0 - 0.0008718296)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.max_dr > 0.1939977:
        z += -0.03436657 * (15.0 - Q.n_dr_0p2_0p4) * (Q.max_dr - 0.1939977)
    if Q.n_particles > 22.0 and Q.sj3_mass1 > 32.50209:
        z += 0.0001326194 * (Q.n_particles - 22.0) * (Q.sj3_mass1 - 32.50209)
    if Q.girth2_top15 > 0.00727763 and Q.pt_11 < 33.78125:
        z += -0.6788883 * (Q.girth2_top15 - 0.00727763) * (33.78125 - Q.pt_11)
    if Q.n_dr_0p2_0p4 < 26.0 and Q.n_dr_0p1_0p2 > 11.0:
        z += -0.0005133639 * (26.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 11.0)
    if Q.planar_flow < 0.3036026 and Q.z_dr_0p4_up > 0.0:
        z += -370.8067 * (0.3036026 - Q.planar_flow) * (Q.z_dr_0p4_up - 0.0)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.sj3_dr13 > 0.1348442:
        z += 0.1733166 * (15.0 - Q.n_dr_0p2_0p4) * (Q.sj3_dr13 - 0.1348442)
    if Q.mass_top40 < 83.32554 and Q.D2_b2 < 0.4953696:
        z += -0.1615284 * (83.32554 - Q.mass_top40) * (0.4953696 - Q.D2_b2)
    if Q.psi_0p1 < 0.9909875 and Q.pt_1 > 99.0:
        z += -0.005179089 * (0.9909875 - Q.psi_0p1) * (Q.pt_1 - 99.0)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.5048165
    z += -0.02018962 * Q.n_particles + 1.292136
    if 0.09046749 <= Q.mass_over_sum_pt < 0.1708801:
        z += -16.95851 * Q.mass_over_sum_pt + 1.534194
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 184.2841 * Q.mass_over_sum_pt - 32.85417
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -15.29677 * Q.log_sum_pt + 105.7027
    if 6.920349 <= Q.log_sum_pt < 6.935549:
        z += -26.91819 * Q.log_sum_pt + 186.127
    if 6.935549 <= Q.log_sum_pt < 6.98945:
        z += -21.24909 * Q.log_sum_pt + 146.8086
    if Q.log_sum_pt >= 6.98945:
        z += -14.78719 * Q.log_sum_pt + 101.6435
    if 907.9372 <= Q.sum_pt < 986.0565:
        z += 0.0035896 * Q.sum_pt - 3.259132
    if Q.sum_pt >= 986.0565:
        z += -0.00492387 * Q.sum_pt + 5.135631
    if Q.psi_0p3 >= 0.9973959:
        z += 54.97327 * Q.psi_0p3 - 54.83011
    if Q.girth2_top50 >= 0.01951641:
        z += -35.66297 * Q.girth2_top50 + 0.6960132
    if Q.girth2 >= 0.004756928:
        z += 11.25832 * Q.girth2 - 0.05355502
    if Q.sum_pt_top50 >= 934.2416:
        z += 0.01668984 * Q.sum_pt_top50 - 15.59234
    if Q.n_pt_above_1 >= 28.0:
        z += 0.007228493 * Q.n_pt_above_1 - 0.2023978
    if Q.sum_pt_top20 >= 1129.275:
        z += 0.00324418 * Q.sum_pt_top20 - 3.663571
    if Q.sum_pt_top3 < 787.6281:
        z += 0.0005322627 * Q.sum_pt_top3 - 0.4192251
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.02616902 * Q.n_dr_0p2_0p4 + 0.2878592
    if Q.sum_pt_top40 >= 1024.942:
        z += -0.007920411 * Q.sum_pt_top40 + 8.117965
    if Q.sum_pt_top30 >= 933.1875:
        z += 0.003711388 * Q.sum_pt_top30 - 3.46342
    if 0.2404747 <= Q.max_dr < 0.4357228:
        z += -1.023408 * Q.max_dr + 0.2461037
    if Q.max_dr >= 0.4357228:
        z += -0.6132773 * Q.max_dr + 0.06740047
    if Q.z_top30_slots >= 0.9048492:
        z += -3.779153 * Q.z_top30_slots + 3.419563
    if Q.mass_top40 < 120.6:
        z += 0.003657671 * Q.mass_top40 - 0.5487035
    if 120.6 <= Q.mass_top40 < 150.0144:
        z += -0.008779247 * Q.mass_top40 + 0.9511888
    if Q.mass_top40 >= 150.0144:
        z += -0.01243692 * Q.mass_top40 + 1.499892
    if 74.25181 <= Q.mass < 91.19:
        z += 0.009164209 * Q.mass - 0.680459
    if 91.19 <= Q.mass < 172.8:
        z += -0.007546583 * Q.mass + 0.8433981
    if Q.mass >= 172.8:
        z += 0.03185817 * Q.mass - 5.965744
    if Q.girth2_top30 >= 0.006363916:
        z += 14.1642 * Q.girth2_top30 - 0.09013976
    if Q.z_dr_0_0p05 >= 0.9084912:
        z += 2.267004 * Q.z_dr_0_0p05 - 2.059554
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -496.0437 * Q.mass_over_sum_pt_sq + 14.48448
    if Q.sj2_mass1 < 23.31647:
        z += -0.01372357 * Q.sj2_mass1 + 0.3199853
    if Q.tau21 < 0.5494307:
        z += 0.2198906 * Q.tau21 - 0.1208146
    if 69.65633 <= Q.sd_mass < 86.4:
        z += 0.01102354 * Q.sd_mass - 0.7678593
    if Q.sd_mass >= 86.4:
        z += -0.002864803 * Q.sd_mass + 0.4320936
    if Q.mass_top50 >= 157.5448:
        z += -0.02982169 * Q.mass_top50 + 4.698253
    if Q.girth2_top15 < 0.0004816552:
        z += 1127.044 * Q.girth2_top15 - 0.5428468
    if Q.sum_pt_top10 >= 943.6922:
        z += -0.001207093 * Q.sum_pt_top10 + 1.139124
    if Q.z_11 < 0.01375115:
        z += -55.03392 * Q.z_11 + 0.7567795
    if Q.n_pt_above_10 >= 31.0:
        z += -0.02477386 * Q.n_pt_above_10 + 0.7679897
    if Q.e2 < 0.03875945:
        z += -8.82996 * Q.e2 + 0.3422444
    if Q.M3 < 0.03787151:
        z += -11.42991 * Q.M3 + 0.432868
    if Q.n_dr_0p1_0p2 < 8.0:
        z += -0.0307324 * Q.n_dr_0p1_0p2 + 0.13324
    if 8.0 <= Q.n_dr_0p1_0p2 < 21.0:
        z += 0.008663013 * Q.n_dr_0p1_0p2 - 0.1819233
    if Q.mass_top10 >= 71.781:
        z += 0.003113923 * Q.mass_top10 - 0.2235205
    if Q.z_dr_0p2_0p4 < 0.1937447:
        z += 0.7364742 * Q.z_dr_0p2_0p4 - 0.142688
    if Q.pt_11 < 14.14062:
        z += 0.03396418 * Q.pt_11 - 0.4802748
    if Q.ptdr0_10 < 1.20817:
        z += 0.03058959 * Q.ptdr0_10 - 0.03695744
    if Q.D2 < 1.976207:
        z += -0.09749929 * Q.D2 + 0.1926788
    if Q.n_particles < 64.0 and Q.z_8 < 0.03260972:
        z += 0.2380499 * (64.0 - Q.n_particles) * (0.03260972 - Q.z_8)
    if Q.n_particles < 64.0 and Q.D2 < 2.178951:
        z += -0.01122519 * (64.0 - Q.n_particles) * (2.178951 - Q.D2)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 29443.11 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    if Q.log_sum_pt > 6.910131 and Q.dr_12 < 0.1269782:
        z += -17.2717 * (Q.log_sum_pt - 6.910131) * (0.1269782 - Q.dr_12)
    if Q.psi_0p3 > 0.9973959 and Q.D2 < 3.345339:
        z += 50.78727 * (Q.psi_0p3 - 0.9973959) * (3.345339 - Q.D2)
    if Q.log_sum_pt > 6.910131 and Q.dr_11 < 0.2535773:
        z += -8.093022 * (Q.log_sum_pt - 6.910131) * (0.2535773 - Q.dr_11)
    if Q.log_sum_pt > 6.910131 and Q.absphi_13 < 0.1958008:
        z += -8.173745 * (Q.log_sum_pt - 6.910131) * (0.1958008 - Q.absphi_13)
    if Q.sum_pt_top20 > 1129.275 and Q.pair_mass_0_10 > 1.569656:
        z += 4.695479e-05 * (Q.sum_pt_top20 - 1129.275) * (Q.pair_mass_0_10 - 1.569656)
    if Q.log_sum_pt > 6.910131 and Q.dr0_10 > 0.06373449:
        z += 7.461602 * (Q.log_sum_pt - 6.910131) * (Q.dr0_10 - 0.06373449)
    if Q.z_top30_slots > 0.9048492 and Q.sj3_mass1 > 10.87918:
        z += -0.2454379 * (Q.z_top30_slots - 0.9048492) * (Q.sj3_mass1 - 10.87918)
    if Q.M3 < 0.03787151 and Q.soft2_dr > 0.1573586:
        z += -5.739373 * (0.03787151 - Q.M3) * (Q.soft2_dr - 0.1573586)
    if Q.z_dr_0p2_0p4 < 0.1937447 and Q.absphi_9 > 0.03289795:
        z += 2.028028 * (0.1937447 - Q.z_dr_0p2_0p4) * (Q.absphi_9 - 0.03289795)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.210779
    if Q.mass_top50 < 71.79516:
        z += -0.033619 * Q.mass_top50 + 3.107624
    if 71.79516 <= Q.mass_top50 < 172.8:
        z += -0.006870389 * Q.mass_top50 + 1.187203
    if Q.mass < 86.4:
        z += 0.001177352 * Q.mass - 1.755059
    if 86.4 <= Q.mass < 92.85979:
        z += 0.03039071 * Q.mass - 4.279093
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.1043727 * Q.mass - 11.14904
    if 101.0497 <= Q.mass < 120.6:
        z += 0.03080339 * Q.mass - 3.714888
    if Q.e3 < 0.0003372339:
        z += -672.5353 * Q.e3 + 0.2268017
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += 0.03579256 * Q.n_dr_0p2_0p4 - 0.5368884
    if Q.girth2_top15 < 0.002197765:
        z += -41.38541 * Q.girth2_top15 + 0.09095539
    if 29.00832 <= Q.sj3_pair_mass_min < 76.60223:
        z += -0.009572719 * Q.sj3_pair_mass_min + 0.2776885
    if Q.sj3_pair_mass_min >= 76.60223:
        z += -0.04684667 * Q.sj3_pair_mass_min + 3.132956
    if 0.02793599 <= Q.e2 < 0.05557149:
        z += -17.035 * Q.e2 + 0.4758897
    if Q.e2 >= 0.05557149:
        z += -5.253058 * Q.e2 - 0.1788504
    if Q.sj3_mass1 >= 21.11128:
        z += -0.002178217 * Q.sj3_mass1 + 0.04598494
    if Q.log_sum_pt < 6.811175:
        z += 5.44779 * Q.log_sum_pt - 37.28246
    if 6.811175 <= Q.log_sum_pt < 7.017258:
        z += 0.857002 * Q.log_sum_pt - 6.013804
    if Q.tau1 < 0.06310829:
        z += -15.78867 * Q.tau1 + 0.9963956
    if Q.M2 < 0.1011005:
        z += 3.150634 * Q.M2 - 0.3185306
    if Q.sj3_dr_min >= 0.1204829:
        z += -1.945534 * Q.sj3_dr_min + 0.2344036
    if Q.z_top40_slots < 0.9300465:
        z += -12.74652 * Q.z_top40_slots + 11.85485
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -1.110206 * Q.z_dr_0p1_0p2 + 0.1336063
    if Q.n_dr_0p1_0p2 >= 33.0:
        z += 0.01761612 * Q.n_dr_0p1_0p2 - 0.5813318
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -15.28746 * Q.mass_over_sum_pt + 1.49747
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -19.91376 * Q.mass_over_sum_pt + 2.044419
    if Q.girth2_top30 < 0.00375223:
        z += -143.7729 * Q.girth2_top30 + 0.1000914
    if 0.00375223 <= Q.girth2_top30 < 0.008376291:
        z += 30.67303 * Q.girth2_top30 - 0.5544698
    if 0.008376291 <= Q.girth2_top30 < 0.01807679:
        z += 124.5791 * Q.girth2_top30 - 1.341054
    if 0.01807679 <= Q.girth2_top30 < 0.02809026:
        z += 93.90607 * Q.girth2_top30 - 0.7865846
    if Q.girth2_top30 >= 0.02809026:
        z += 24.01544 * Q.girth2_top30 + 1.176661
    if Q.girth2_top50 < 0.00242543:
        z += 369.0817 * Q.girth2_top50 - 1.374667
    if 0.00242543 <= Q.girth2_top50 < 0.004573744:
        z += 223.1914 * Q.girth2_top50 - 1.02082
    if Q.pt_14 >= 10.78125:
        z += -0.01007238 * Q.pt_14 + 0.1085929
    if Q.girth2_top20 >= 0.008031209:
        z += 50.59853 * Q.girth2_top20 - 0.4063674
    if Q.lam1 < 0.003811746:
        z += -258.278 * Q.lam1 + 0.9844902
    if Q.lam2 < 0.003687605:
        z += -76.55588 * Q.lam2 + 0.2823078
    if Q.mass_top15 < 91.19:
        z += -0.006852128 * Q.mass_top15 + 0.6248455
    if Q.LHA >= 0.3719813:
        z += 2.664742 * Q.LHA - 0.9912339
    if Q.girth2_top3 < 0.001592178:
        z += 47.99443 * Q.girth2_top3 - 0.07641565
    if Q.pt_6 < 19.46875:
        z += -0.01209557 * Q.pt_6 + 0.2354856
    if Q.n_dr_0_0p05 >= 9.0:
        z += -0.005378098 * Q.n_dr_0_0p05 + 0.04840288
    if Q.C2 < 0.04704395:
        z += 0.09418507 * Q.C2 - 0.004430838
    if Q.mass < 120.6 and Q.sum_pt < 1007.788:
        z += 8.256766e-05 * (120.6 - Q.mass) * (1007.788 - Q.sum_pt)
    if Q.sj3_pair_mass_min > 29.00832 and Q.psi_0p3 > 0.9896594:
        z += 0.9969018 * (Q.sj3_pair_mass_min - 29.00832) * (Q.psi_0p3 - 0.9896594)
    if Q.sj3_dr_min > 0.1204829 and Q.pt_11 < 33.78125:
        z += 0.07205912 * (Q.sj3_dr_min - 0.1204829) * (33.78125 - Q.pt_11)
    if Q.M2 < 0.1011005 and Q.eta_0 < -0.07794189:
        z += -9.043169 * (0.1011005 - Q.M2) * (-0.07794189 - Q.eta_0)
    if Q.mass_over_sum_pt > 0.1182259 and Q.C2_b2 > 0.02704832:
        z += 267.3215 * (Q.mass_over_sum_pt - 0.1182259) * (Q.C2_b2 - 0.02704832)
    if Q.e2 > 0.05557149 and Q.zdr_0 > 0.001369707:
        z += 355.9366 * (Q.e2 - 0.05557149) * (Q.zdr_0 - 0.001369707)
    if Q.sj3_dr_min > 0.1204829 and Q.sj3_mass2 < 7.863702:
        z += 0.2589862 * (Q.sj3_dr_min - 0.1204829) * (7.863702 - Q.sj3_mass2)
    if Q.e2 > 0.05557149 and Q.z_2 > 0.120204:
        z += 6575.832 * (Q.e2 - 0.05557149) * (Q.z_2 - 0.120204)
    if Q.n_dr_0p2_0p4 > 15.0 and Q.max_pair_mass > 33.3761:
        z += -0.001664943 * (Q.n_dr_0p2_0p4 - 15.0) * (Q.max_pair_mass - 33.3761)
    if Q.e2 > 0.05557149 and Q.D2_b2 < 1.67722:
        z += 37.87323 * (Q.e2 - 0.05557149) * (1.67722 - Q.D2_b2)
    if Q.girth2_top30 > 0.008376291 and Q.D2_b2 > 1.67722:
        z += -32.19874 * (Q.girth2_top30 - 0.008376291) * (Q.D2_b2 - 1.67722)
    if Q.mass_over_sum_pt > 0.1182259 and Q.D2_b2 < 7.36624:
        z += -4.755548 * (Q.mass_over_sum_pt - 0.1182259) * (7.36624 - Q.D2_b2)
    if Q.girth2_top30 > 0.008376291 and Q.z_dr_0p05_0p1 > 0.6469679:
        z += 1494.346 * (Q.girth2_top30 - 0.008376291) * (Q.z_dr_0p05_0p1 - 0.6469679)
    if Q.girth2_top30 > 0.008376291 and Q.n_real_top50 < 43.0:
        z += 16.11843 * (Q.girth2_top30 - 0.008376291) * (43.0 - Q.n_real_top50)
    if Q.e3 < 0.0003372339 and Q.dr_max_012 > 0.004415714:
        z += -3898.166 * (0.0003372339 - Q.e3) * (Q.dr_max_012 - 0.004415714)
    if Q.girth2_top30 > 0.008376291 and Q.orientation_deg > 26.6454:
        z += -0.1159933 * (Q.girth2_top30 - 0.008376291) * (Q.orientation_deg - 26.6454)
    if Q.n_dr_0p1_0p2 > 33.0 and Q.absphi_2 < 0.12854:
        z += -0.06403355 * (Q.n_dr_0p1_0p2 - 33.0) * (0.12854 - Q.absphi_2)
    if Q.girth2_top20 > 0.008031209 and Q.C2_b2 > 0.0008187529:
        z += -1113.6 * (Q.girth2_top20 - 0.008031209) * (Q.C2_b2 - 0.0008187529)
    if Q.mass_top15 < 91.19 and Q.C2_b2 < 0.04008677:
        z += -0.04857299 * (91.19 - Q.mass_top15) * (0.04008677 - Q.C2_b2)
    if Q.n_dr_0p1_0p2 > 33.0 and Q.ptdr0_10 > 6.130737:
        z += -0.004478677 * (Q.n_dr_0p1_0p2 - 33.0) * (Q.ptdr0_10 - 6.130737)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.1381217
    if Q.tau21_b2 < 0.2352054:
        z += -6.719413 * Q.tau21_b2 + 1.580443
    if Q.girth2 < 0.006403325:
        z += 122.9101 * Q.girth2 - 0.4499333
    if 0.006403325 <= Q.girth2 < 0.007877041:
        z += 52.29768 * Q.girth2 + 0.002221081
    if 0.007877041 <= Q.girth2 < 0.008190222:
        z += 214.3613 * Q.girth2 - 1.274361
    if 0.008190222 <= Q.girth2 < 0.009614971:
        z += -337.8179 * Q.girth2 + 3.24811
    if Q.mass_over_sum_pt < 0.1182259:
        z += -19.51665 * Q.mass_over_sum_pt + 2.307374
    if Q.mass < 78.26182:
        z += 0.03556828 * Q.mass - 2.991789
    if 78.26182 <= Q.mass < 82.85409:
        z += 0.09202975 * Q.mass - 7.410566
    if 82.85409 <= Q.mass < 91.19:
        z += 0.00130704 * Q.mass + 0.106181
    if 91.19 <= Q.mass < 92.85979:
        z += 0.02855279 * Q.mass - 2.378359
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.03661424 * Q.mass - 3.126943
    if 101.0497 <= Q.mass < 120.6:
        z += -0.02930467 * Q.mass + 3.534143
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.142237 * Q.n_dr_0p2_0p4 + 0.8534221
    if 0.9973959 <= Q.psi_0p3 < 0.9980008:
        z += 89.02727 * Q.psi_0p3 - 88.79543
    if Q.psi_0p3 >= 0.9980008:
        z += -119.4627 * Q.psi_0p3 + 119.2778
    if Q.tau21 < 0.3861957:
        z += 0.5887777 * Q.tau21 - 0.2273834
    if Q.z_dr_0_0p05 < 0.09573228:
        z += -0.3528375 * Q.z_dr_0_0p05 + 0.03377794
    if Q.z_dr_0p05_0p1 >= 0.6469679:
        z += 0.4411612 * Q.z_dr_0p05_0p1 - 0.2854171
    if Q.lam2 < 0.000404306:
        z += -441.7226 * Q.lam2 + 0.1785911
    if Q.e2_sq < 0.00363788:
        z += 10.8805 * Q.e2_sq - 0.03958196
    if Q.tau1 < 0.08786745:
        z += -2.57452 * Q.tau1 - 0.176225
    if 0.08786745 <= Q.tau1 < 0.09591084:
        z += 4.741884 * Q.tau1 - 0.8190987
    if 0.09591084 <= Q.tau1 < 0.1072713:
        z += 32.06755 * Q.tau1 - 3.439926
    if Q.n_dr_0_0p05 < 18.0:
        z += 0.001446961 * Q.n_dr_0_0p05 - 0.0260453
    if Q.lam1 < 0.006189818:
        z += -9.0186 * Q.lam1 - 0.37078
    if 0.006189818 <= Q.lam1 < 0.008241985:
        z += 207.8795 * Q.lam1 - 1.71334
    if Q.sd_mass < 45.595:
        z += -0.001583461 * Q.sd_mass - 0.206556
    if 45.595 <= Q.sd_mass < 69.65633:
        z += -0.008870753 * Q.sd_mass + 0.1257081
    if 69.65633 <= Q.sd_mass < 86.4:
        z += 0.01163651 * Q.sd_mass - 1.302753
    if 86.4 <= Q.sd_mass < 98.05743:
        z += -0.007287292 * Q.sd_mass + 0.3322641
    if Q.sd_mass >= 98.05743:
        z += 0.001724117 * Q.sd_mass - 0.5513716
    if 73.35236 <= Q.mass_top20 < 85.79457:
        z += 0.01843085 * Q.mass_top20 - 1.351946
    if Q.mass_top20 >= 85.79457:
        z += -0.0001851842 * Q.mass_top20 + 0.2452081
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1937688:
        z += -4.283913 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1937688)
    if Q.tau21_b2 < 0.2352054 and Q.mass_over_sum_pt_sq < 0.007873266:
        z += -793.8333 * (0.2352054 - Q.tau21_b2) * (0.007873266 - Q.mass_over_sum_pt_sq)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.sj2_dr < 0.1825048:
        z += 0.1370725 * (6.0 - Q.n_dr_0p2_0p4) * (0.1825048 - Q.sj2_dr)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9638082:
        z += 9.942455 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.e3 < 7.876005e-05:
        z += -1229.16 * (6.0 - Q.n_dr_0p2_0p4) * (7.876005e-05 - Q.e3)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9777125:
        z += 7.855925 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 82.85409 and Q.psi_0p3 < 0.9985421:
        z += 0.1468521 * (82.85409 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9777125:
        z += -17.03833 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9777125:
        z += 8.816855 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.tau21_b2 < 0.2352054 and Q.n_real_top50 > 34.0:
        z += 0.06371539 * (0.2352054 - Q.tau21_b2) * (Q.n_real_top50 - 34.0)
    if Q.psi_0p3 > 0.9980008 and Q.n_dr_0p1_0p2 > 14.0:
        z += 5.645804 * (Q.psi_0p3 - 0.9980008) * (Q.n_dr_0p1_0p2 - 14.0)
    if Q.mass < 91.19 and Q.n_dr_0_0p05 < 6.0:
        z += -0.012175 * (91.19 - Q.mass) * (6.0 - Q.n_dr_0_0p05)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9980008:
        z += -131.3616 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9980008:
        z += 60.92196 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9980008:
        z += 66.18608 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 1260.541:
        z += -0.009252801 * (0.2352054 - Q.tau21_b2) * (1260.541 - Q.sum_pt)
    if Q.tau21_b2 < 0.2352054 and Q.z_2 < 0.1063277:
        z += -15.90882 * (0.2352054 - Q.tau21_b2) * (0.1063277 - Q.z_2)
    if Q.mass < 92.85979 and Q.psi_0p3 > 0.9638082:
        z += -8.739812 * (92.85979 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.tau21_b2 < 0.2352054 and Q.orientation_deg > -9.840088:
        z += -0.01858098 * (0.2352054 - Q.tau21_b2) * (Q.orientation_deg - -9.840088)
    if Q.girth2 < 0.006403325 and Q.psi_0p3 > 0.9980008:
        z += 36554.9 * (0.006403325 - Q.girth2) * (Q.psi_0p3 - 0.9980008)
    if Q.z_dr_0_0p05 < 0.09573228 and Q.sj3_mass3 < 5.541989:
        z += 0.03495341 * (0.09573228 - Q.z_dr_0_0p05) * (5.541989 - Q.sj3_mass3)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.n_dr_0p1_0p2 > 15.0:
        z += 0.0007954502 * (6.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 15.0)
    if Q.tau21_b2 < 0.2352054 and Q.psi_0p3 > 0.9299135:
        z += -26.27606 * (0.2352054 - Q.tau21_b2) * (Q.psi_0p3 - 0.9299135)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.max_dr > 0.3618493:
        z += -0.5259468 * (6.0 - Q.n_dr_0p2_0p4) * (Q.max_dr - 0.3618493)
    if Q.mass < 92.85979 and Q.z_dr_0_0p05 < 0.4947602:
        z += -0.09501294 * (92.85979 - Q.mass) * (0.4947602 - Q.z_dr_0_0p05)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.eta_1 > 0.0001021922:
        z += -0.04815987 * (6.0 - Q.n_dr_0p2_0p4) * (Q.eta_1 - 0.0001021922)
    if Q.lam1 < 0.008241985 and Q.zdr_0 > 0.01564747:
        z += -17971.74 * (0.008241985 - Q.lam1) * (Q.zdr_0 - 0.01564747)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.ptdr0_2 > 11.87288:
        z += 0.0046146 * (6.0 - Q.n_dr_0p2_0p4) * (Q.ptdr0_2 - 11.87288)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.220754
    if 0.07696632 <= Q.mass_over_sum_pt < 0.1182259:
        z += -12.76862 * Q.mass_over_sum_pt + 0.9827539
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -29.46826 * Q.mass_over_sum_pt + 2.957085
    if Q.sum_pt_top40 < 1001.523:
        z += -0.00209914 * Q.sum_pt_top40 + 1.932828
    if 1001.523 <= Q.sum_pt_top40 < 1024.942:
        z += 0.007238048 * Q.sum_pt_top40 - 7.418582
    if Q.girth2_top40 < 0.005196966:
        z += -129.1372 * Q.girth2_top40 + 1.141642
    if 0.005196966 <= Q.girth2_top40 < 0.008840538:
        z += -63.06718 * Q.girth2_top40 + 0.7982787
    if Q.girth2_top40 >= 0.008840538:
        z += 66.06999 * Q.girth2_top40 - 0.3433635
    if Q.n_dr_0p2_0p4 < 3.0:
        z += -0.008268535 * Q.n_dr_0p2_0p4 - 0.1409743
    if 3.0 <= Q.n_dr_0p2_0p4 < 9.0:
        z += 0.01910963 * Q.n_dr_0p2_0p4 - 0.2231088
    if 9.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += 0.06327669 * Q.n_dr_0p2_0p4 - 0.6206123
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += 0.02737817 * Q.n_dr_0p2_0p4 - 0.0821345
    if 64.48544 <= Q.mass < 74.25181:
        z += 0.02364753 * Q.mass - 1.524922
    if 74.25181 <= Q.mass < 87.36377:
        z += 0.05295813 * Q.mass - 3.701287
    if 87.36377 <= Q.mass < 101.0497:
        z += 0.1065109 * Q.mass - 8.37986
    if 101.0497 <= Q.mass < 125.1:
        z += 0.06174363 * Q.mass - 3.856138
    if Q.mass >= 125.1:
        z += 0.009156464 * Q.mass + 2.722516
    if Q.n_particles >= 51.0:
        z += 0.01485614 * Q.n_particles - 0.7576633
    if Q.sum_pt < 1017.435:
        z += -0.03325817 * Q.sum_pt + 34.02565
    if 1017.435 <= Q.sum_pt < 1028.184:
        z += -0.01745545 * Q.sum_pt + 17.94741
    if 0.006043209 <= Q.girth2_top20 < 0.008031209:
        z += -128.1119 * Q.girth2_top20 + 0.7742069
    if Q.girth2_top20 >= 0.008031209:
        z += 70.87874 * Q.girth2_top20 - 0.8239285
    if Q.log_sum_pt < 6.930088:
        z += 13.15983 * Q.log_sum_pt - 91.19878
    if Q.log_sum_pt >= 7.139296:
        z += -0.8814397 * Q.log_sum_pt + 6.292859
    if Q.girth2_top30 < 0.007463985:
        z += -168.8955 * Q.girth2_top30 + 1.260634
    if Q.width < 0.009614971:
        z += 579.2718 * Q.width - 5.569682
    if Q.girth2 < 0.007877041:
        z += -297.4901 * Q.girth2 + 2.343342
    if Q.sj2_dr >= 0.2232169:
        z += 1.791604 * Q.sj2_dr - 0.3999163
    if Q.z_dr_0p1_0p2 >= 0.3340477:
        z += 1.110941 * Q.z_dr_0p1_0p2 - 0.3711073
    if Q.n_for_90pct < 7.0:
        z += 0.04005971 * Q.n_for_90pct - 1.562329
    if 7.0 <= Q.n_for_90pct < 39.0:
        z += -0.01631948 * Q.n_for_90pct - 1.167674
    if Q.n_for_90pct >= 39.0:
        z += -0.05637919 * Q.n_for_90pct + 0.3946543
    if Q.psi_0p3 >= 0.9943058:
        z += -52.08794 * Q.psi_0p3 + 51.79134
    if 0.04755309 <= Q.e2 < 0.06524004:
        z += 33.82515 * Q.e2 - 1.60849
    if Q.e2 >= 0.06524004:
        z += 37.29959 * Q.e2 - 1.835163
    if Q.lam1 < 0.007671243:
        z += -66.58161 * Q.lam1 + 0.3731573
    if 0.007671243 <= Q.lam1 < 0.01174405:
        z += 33.78661 * Q.lam1 - 0.3967917
    if Q.tau2 < 0.0795038:
        z += -1.666088 * Q.tau2 + 0.1324603
    if Q.C2 >= 0.06655881:
        z += 6.2935 * Q.C2 - 0.4188879
    if 5.13841e-05 <= Q.e3 < 0.0003372339:
        z += -1937.936 * Q.e3 + 0.09957909
    if Q.e3 >= 0.0003372339:
        z += -167.5221 * Q.e3 - 0.4974644
    if 82.04491 <= Q.mass_top50 < 117.0487:
        z += -0.03376884 * Q.mass_top50 + 2.770562
    if Q.mass_top50 >= 117.0487:
        z += 0.004866768 * Q.mass_top50 - 1.751688
    if Q.mass_top20 >= 119.2969:
        z += -0.01711505 * Q.mass_top20 + 2.041771
    if Q.n_dr_0p1_0p2 >= 19.0:
        z += 0.01371921 * Q.n_dr_0p1_0p2 - 0.2606649
    if Q.z_top15_slots < 0.8316924:
        z += 0.4491305 * Q.z_top15_slots - 0.3735384
    if Q.eta_0 >= -0.01382675:
        z += 1.13591 * Q.eta_0 + 0.01570595
    if Q.psi_0p1 < 0.3628388:
        z += 1.138378 * Q.psi_0p1 - 0.4130477
    if Q.z_top50_slots >= 0.9704436:
        z += -13.23679 * Q.z_top50_slots + 12.84556
    if Q.soft5_pt < 1.219727:
        z += -0.2772111 * Q.soft5_pt + 0.3381218
    if Q.girth2_top15 < 0.02146578:
        z += 27.33764 * Q.girth2_top15 - 0.5868237
    if Q.mass_over_sum_pt > 0.07696632 and Q.sum_pt < 1115.723:
        z += 0.2820247 * (Q.mass_over_sum_pt - 0.07696632) * (1115.723 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.z_dr_0p1_0p2 > 0.3340477:
        z += 0.1098798 * (15.0 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p1_0p2 - 0.3340477)
    if Q.girth2_top40 > 0.005196966 and Q.log_sum_pt < 7.017258:
        z += -687.1996 * (Q.girth2_top40 - 0.005196966) * (7.017258 - Q.log_sum_pt)
    if Q.n_particles > 51.0 and Q.soft1_pt < 1.091797:
        z += -0.01028661 * (Q.n_particles - 51.0) * (1.091797 - Q.soft1_pt)
    if Q.sum_pt < 1017.435 and Q.n_for_90pct < 13.0:
        z += 0.0002299996 * (1017.435 - Q.sum_pt) * (13.0 - Q.n_for_90pct)
    if Q.mass > 64.48544 and Q.zdr_0 > 0.0008718296:
        z += -0.02336669 * (Q.mass - 64.48544) * (Q.zdr_0 - 0.0008718296)
    if Q.girth2_top30 < 0.007463985 and Q.sj2_dr > 0.1512157:
        z += -662.4682 * (0.007463985 - Q.girth2_top30) * (Q.sj2_dr - 0.1512157)
    if Q.tau2 < 0.0795038 and Q.sj3_mass1 > 13.38202:
        z += 0.4817931 * (0.0795038 - Q.tau2) * (Q.sj3_mass1 - 13.38202)
    if Q.sum_pt < 1017.435 and Q.dr_max_012 > 0.1828389:
        z += -0.1927291 * (1017.435 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    if Q.sum_pt < 1028.184 and Q.dr_max_012 > 0.1828389:
        z += 0.1780424 * (1028.184 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    if Q.girth2_top20 > 0.006043209 and Q.sj3_z2 < 0.08173536:
        z += 13812.67 * (Q.girth2_top20 - 0.006043209) * (0.08173536 - Q.sj3_z2)
    if Q.log_sum_pt < 6.930088 and Q.dr01 < 0.05595395:
        z += 30.42732 * (6.930088 - Q.log_sum_pt) * (0.05595395 - Q.dr01)
    if Q.mass_top20 > 119.2969 and Q.absphi_1 > 0.07141113:
        z += -0.1204549 * (Q.mass_top20 - 119.2969) * (Q.absphi_1 - 0.07141113)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.9315234
    if Q.mass_top40 < 80.89043:
        z += -0.005373389 * Q.mass_top40 - 1.30686
    if 80.89043 <= Q.mass_top40 < 125.1:
        z += 0.01907854 * Q.mass_top40 - 3.284787
    if 125.1 <= Q.mass_top40 < 163.2541:
        z += 0.02353772 * Q.mass_top40 - 3.842631
    if Q.sum_pt_top40 < 956.2133:
        z += -0.008836194 * Q.sum_pt_top40 + 8.449286
    if Q.girth2_top40 < 0.00217213:
        z += -15.55061 * Q.girth2_top40 + 0.703475
    if 0.00217213 <= Q.girth2_top40 < 0.006259772:
        z += -163.8345 * Q.girth2_top40 + 1.025567
    if Q.z_dr_0p1_0p2 >= 0.6882177:
        z += -2.190543 * Q.z_dr_0p1_0p2 + 1.50757
    if Q.mass_over_sum_pt >= 0.1708801:
        z += -4.615656 * Q.mass_over_sum_pt + 0.7887238
    if Q.mass < 64.48544:
        z += -0.05849842 * Q.mass + 6.284754
    if 64.48544 <= Q.mass < 82.85409:
        z += -0.09068636 * Q.mass + 8.360407
    if 82.85409 <= Q.mass < 92.85979:
        z += -0.03414898 * Q.mass + 3.676054
    if 92.85979 <= Q.mass < 143.7876:
        z += 0.03460234 * Q.mass - 2.708179
    if 143.7876 <= Q.mass < 160.8:
        z += -0.05960625 * Q.mass + 10.83785
    if 160.8 <= Q.mass < 162.8363:
        z += -0.2138668 * Q.mass + 35.64294
    if 162.8363 <= Q.mass < 172.8:
        z += -0.08206394 * Q.mass + 14.18065
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -1.403086 * Q.z_dr_0_0p05 + 1.233181
    if 0.09749958 <= Q.girth < 0.1402186:
        z += 19.59899 * Q.girth - 1.910893
    if Q.girth >= 0.1402186:
        z += -37.84861 * Q.girth + 6.144328
    if Q.z_top5 >= 0.7963975:
        z += 6.177763 * Q.z_top5 - 4.919955
    if Q.psi_0p3 >= 0.9943058:
        z += 39.14283 * Q.psi_0p3 - 38.91995
    if Q.tau1 < 0.05444509:
        z += 14.98062 * Q.tau1 - 0.8156212
    if Q.lam1 < 0.004673423:
        z += -168.0424 * Q.lam1 + 0.7853331
    if Q.lam1 >= 0.01174405:
        z += 27.27021 * Q.lam1 - 0.3202627
    if Q.girth2_top30 >= 0.0008564881:
        z += -27.99563 * Q.girth2_top30 + 0.02397792
    if Q.D2 >= 2.178951:
        z += 0.03829594 * Q.D2 - 0.08344495
    if Q.mass_top50 < 92.16545:
        z += -0.007520817 * Q.mass_top50 + 3.043718
    if 92.16545 <= Q.mass_top50 < 138.8977:
        z += -0.05029846 * Q.mass_top50 + 6.986339
    if Q.mass_top30 < 73.33139:
        z += -0.01433596 * Q.mass_top30 + 1.051276
    if Q.n_for_90pct >= 35.0:
        z += 0.04761786 * Q.n_for_90pct - 1.666625
    if Q.LHA >= 0.404204:
        z += 19.7355 * Q.LHA - 7.977167
    if Q.sj3_dr13 >= 0.1654269:
        z += -0.01925327 * Q.sj3_dr13 + 0.003185009
    if Q.z_top40_slots >= 0.9574183:
        z += -3.370595 * Q.z_top40_slots + 3.227069
    if Q.e3 >= 0.0003372339:
        z += 66.58482 * Q.e3 - 0.02245466
    if Q.sum_pt < 986.0565:
        z += -0.01736079 * Q.sum_pt + 17.11872
    if Q.log_sum_pt < 6.903423:
        z += 11.64594 * Q.log_sum_pt - 80.39685
    if Q.sum_pt_top40 < 956.2133 and Q.psi_0p1 > 0.9184255:
        z += -0.004043487 * (956.2133 - Q.sum_pt_top40) * (Q.psi_0p1 - 0.9184255)
    if Q.sum_pt_top40 < 956.2133 and Q.soft4_pt > 1.789258:
        z += -0.004186431 * (956.2133 - Q.sum_pt_top40) * (Q.soft4_pt - 1.789258)
    if Q.mass_top40 < 80.89043 and Q.sum_pt < 1034.834:
        z += -0.0001764549 * (80.89043 - Q.mass_top40) * (1034.834 - Q.sum_pt)
    if Q.mass_top40 < 80.89043 and Q.sum_pt_top40 < 906.6023:
        z += 0.0002251443 * (80.89043 - Q.mass_top40) * (906.6023 - Q.sum_pt_top40)
    if Q.sum_pt_top40 < 956.2133 and Q.soft3_pt > 2.873047:
        z += 0.0005326021 * (956.2133 - Q.sum_pt_top40) * (Q.soft3_pt - 2.873047)
    if Q.mass < 92.85979 and Q.z_top50_slots > 0.9995789:
        z += -5.051931 * (92.85979 - Q.mass) * (Q.z_top50_slots - 0.9995789)
    if Q.mass_top40 < 125.1 and Q.n_dr_0p2_0p4 > 9.0:
        z += 0.001392812 * (125.1 - Q.mass_top40) * (Q.n_dr_0p2_0p4 - 9.0)
    if Q.sum_pt_top40 < 956.2133 and Q.lam2 < 0.001776308:
        z += -7.278216 * (956.2133 - Q.sum_pt_top40) * (0.001776308 - Q.lam2)
    if Q.z_dr_0p1_0p2 > 0.6882177 and Q.pt_2 < 154.25:
        z += 0.002444186 * (Q.z_dr_0p1_0p2 - 0.6882177) * (154.25 - Q.pt_2)
    if Q.sum_pt < 986.0565 and Q.soft5_z > 0.002036949:
        z += 4.453363 * (986.0565 - Q.sum_pt) * (Q.soft5_z - 0.002036949)
    return max(0.0, z)


def neuron_10(Q):
    z = 0.1383221
    if Q.girth < 0.1207452:
        z += 7.997161 * Q.girth - 0.9656186
    if Q.mass < 64.48544:
        z += -0.03263009 * Q.mass + 2.579021
    if 64.48544 <= Q.mass < 78.26182:
        z += -0.06528367 * Q.mass + 4.684702
    if 78.26182 <= Q.mass < 86.4:
        z += -0.006913065 * Q.mass + 0.1165122
    if 86.4 <= Q.mass < 89.74183:
        z += -0.01998878 * Q.mass + 1.246254
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.06046801 * Q.mass - 5.974086
    if 101.0497 <= Q.mass < 143.7876:
        z += 0.02571703 * Q.mass - 2.462509
    if 143.7876 <= Q.mass < 162.8363:
        z += -0.06461151 * Q.mass + 10.52562
    if Q.mass >= 162.8363:
        z += -0.1797817 * Q.mass + 29.27951
    if Q.girth2_top30 < 0.005402331:
        z += 119.9015 * Q.girth2_top30 - 0.6477478
    if Q.sum_pt_top20 < 750.7313:
        z += 0.003017007 * Q.sum_pt_top20 - 2.264961
    if 136.785 <= Q.mass_top50 < 160.8:
        z += 0.0691044 * Q.mass_top50 - 9.452445
    if 160.8 <= Q.mass_top50 < 172.8:
        z += 0.08023484 * Q.mass_top50 - 11.24222
    if Q.mass_top50 >= 172.8:
        z += 0.1343988 * Q.mass_top50 - 20.60176
    if Q.sj2_mass1 < 65.20727:
        z += 0.007063121 * Q.sj2_mass1 - 0.4605668
    if Q.D2 < 2.975532:
        z += -0.2790474 * Q.D2 + 0.8303143
    if Q.psi_0p3 >= 0.9985421:
        z += -107.6985 * Q.psi_0p3 + 107.5415
    if Q.z_top2_slots < 0.5760704:
        z += -1.526 * Q.z_top2_slots + 0.8790834
    if Q.mass_top5 < 22.18342:
        z += 0.007311 * Q.mass_top5 - 0.4343302
    if 22.18342 <= Q.mass_top5 < 59.40777:
        z += 0.01587187 * Q.mass_top5 - 0.6242397
    if Q.mass_top5 >= 59.40777:
        z += 0.008560874 * Q.mass_top5 - 0.1899094
    if Q.sum_pt_top50 < 959.0957:
        z += 0.006028456 * Q.sum_pt_top50 - 5.781867
    if Q.girth2 < 0.002575211:
        z += 263.0078 * Q.girth2 - 0.6773006
    if Q.dr_0 < 0.06413297:
        z += -3.949747 * Q.dr_0 + 0.253309
    if Q.z_dr_0_0p05 >= 0.7674734:
        z += -3.274045 * Q.z_dr_0_0p05 + 2.512743
    if Q.girth2_top10 < 0.01976735:
        z += 13.82966 * Q.girth2_top10 - 0.2733758
    if Q.e2 < 0.01256572:
        z += 32.29345 * Q.e2 - 1.407549
    if 0.01256572 <= Q.e2 < 0.03263075:
        z += 84.17791 * Q.e2 - 2.059515
    if 0.03263075 <= Q.e2 < 0.04358622:
        z += 81.08521 * Q.e2 - 1.958598
    if Q.e2 >= 0.04358622:
        z += 48.79176 * Q.e2 - 0.5510488
    if Q.sum_pt_top10 >= 943.6922:
        z += -0.005850376 * Q.sum_pt_top10 + 5.520954
    if Q.n_particles >= 41.0:
        z += 0.007650948 * Q.n_particles - 0.3136889
    if Q.mass_over_sum_pt >= 0.1606361:
        z += -51.39452 * Q.mass_over_sum_pt + 8.255815
    if Q.e3 < 0.0001086251:
        z += -3619.285 * Q.e3 + 0.3931451
    if Q.zdr_1 < 0.008824206:
        z += -13.58177 * Q.zdr_1 + 0.1198483
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.05113773 * Q.n_dr_0p2_0p4 - 0.562515
    if Q.sum_pt_top15 < 935.1043:
        z += -0.002943736 * Q.sum_pt_top15 + 2.7527
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -3.984759 * Q.z_dr_0p2_0p4 + 0.3635124
    if Q.mass_top15 < 72.18744:
        z += 0.01042329 * Q.mass_top15 - 0.7524307
    if Q.girth2_top20 < 0.01655983:
        z += -64.09991 * Q.girth2_top20 + 1.061484
    if Q.sj3_dr_min >= 0.1204829:
        z += 3.092952 * Q.sj3_dr_min - 0.3726478
    if Q.mass_top30 >= 78.53034:
        z += -0.008960475 * Q.mass_top30 + 0.7036692
    if Q.z_2nd < 0.09021906:
        z += 6.89411 * Q.z_2nd - 0.6219801
    if Q.girth2_top30 < 0.005402331 and Q.sum_pt_top5 < 631.275:
        z += 0.02214151 * (0.005402331 - Q.girth2_top30) * (631.275 - Q.sum_pt_top5)
    if Q.mass_top50 > 160.8 and Q.pt_7 < 53.4375:
        z += -0.0007008582 * (Q.mass_top50 - 160.8) * (53.4375 - Q.pt_7)
    if Q.mass_top50 > 160.8 and Q.soft4_z > 0.001721109:
        z += -17.43731 * (Q.mass_top50 - 160.8) * (Q.soft4_z - 0.001721109)
    if Q.z_top2_slots < 0.5760704 and Q.soft9_z < 0.002970691:
        z += 595.8749 * (0.5760704 - Q.z_top2_slots) * (0.002970691 - Q.soft9_z)
    if Q.D2 < 2.975532 and Q.sj2_dr > 0.2070855:
        z += -2.095055 * (2.975532 - Q.D2) * (Q.sj2_dr - 0.2070855)
    if Q.sum_pt_top50 < 959.0957 and Q.absphi_4 > 0.004672432:
        z += -0.006989059 * (959.0957 - Q.sum_pt_top50) * (Q.absphi_4 - 0.004672432)
    if Q.e2 > 0.03263075 and Q.soft2_pt > 1.413232:
        z += -12.07836 * (Q.e2 - 0.03263075) * (Q.soft2_pt - 1.413232)
    if Q.girth2_top30 < 0.005402331 and Q.psi_0p3 > 0.9985421:
        z += 48469.92 * (0.005402331 - Q.girth2_top30) * (Q.psi_0p3 - 0.9985421)
    if Q.girth2_top10 < 0.01976735 and Q.psi_0p3 > 0.9985421:
        z += -26195.15 * (0.01976735 - Q.girth2_top10) * (Q.psi_0p3 - 0.9985421)
    if Q.mass < 101.0497 and Q.pt1_dr01 > 5.351077:
        z += 0.001549625 * (101.0497 - Q.mass) * (Q.pt1_dr01 - 5.351077)
    if Q.sum_pt_top10 > 943.6922 and Q.z_dr_0p05_0p1 < 0.6469679:
        z += 0.001925218 * (Q.sum_pt_top10 - 943.6922) * (0.6469679 - Q.z_dr_0p05_0p1)
    if Q.girth2_top10 < 0.01976735 and Q.max_dr < 0.4357228:
        z += -144.9776 * (0.01976735 - Q.girth2_top10) * (0.4357228 - Q.max_dr)
    if Q.mass_top5 < 59.40777 and Q.z_dr_0p05_0p1 < 0.8509215:
        z += 0.006113249 * (59.40777 - Q.mass_top5) * (0.8509215 - Q.z_dr_0p05_0p1)
    if Q.D2 < 2.975532 and Q.dr_12 > 0.1269782:
        z += -0.1196652 * (2.975532 - Q.D2) * (Q.dr_12 - 0.1269782)
    if Q.mass_over_sum_pt > 0.1606361 and Q.z_3 < 0.09696199:
        z += 524.7985 * (Q.mass_over_sum_pt - 0.1606361) * (0.09696199 - Q.z_3)
    if Q.mass_top50 > 136.785 and Q.soft5_z > 0.001594761:
        z += -31.98989 * (Q.mass_top50 - 136.785) * (Q.soft5_z - 0.001594761)
    if Q.mass > 162.8363 and Q.soft5_z > 0.001434897:
        z += 44.63295 * (Q.mass - 162.8363) * (Q.soft5_z - 0.001434897)
    if Q.girth2_top10 < 0.01976735 and Q.tau32 < 0.8476928:
        z += -2.912864 * (0.01976735 - Q.girth2_top10) * (0.8476928 - Q.tau32)
    if Q.psi_0p3 > 0.9985421 and Q.soft5_z > 0.001434897:
        z += 15844.32 * (Q.psi_0p3 - 0.9985421) * (Q.soft5_z - 0.001434897)
    if Q.sum_pt_top50 < 959.0957 and Q.dr0_3 < 0.2748443:
        z += 0.001557019 * (959.0957 - Q.sum_pt_top50) * (0.2748443 - Q.dr0_3)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.07296445
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.03829611 * Q.n_dr_0p2_0p4 + 0.3829611
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += 11.81608 * Q.mass_over_sum_pt_sq - 0.1133754
    if 0.9313699 <= Q.psi_0p2 < 0.9935324:
        z += 0.04383267 * Q.psi_0p2 - 0.04082443
    if Q.psi_0p2 >= 0.9935324:
        z += 23.8842 * Q.psi_0p2 - 23.72701
    if Q.mass < 80.4:
        z += -0.01755385 * Q.mass + 2.435976
    if 80.4 <= Q.mass < 92.85979:
        z += -0.07899917 * Q.mass + 7.37618
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.00492479 * Q.mass + 0.4976486
    if Q.girth2_top10 < 0.00130722:
        z += -195.7859 * Q.girth2_top10 + 0.184245
    if 0.00130722 <= Q.girth2_top10 < 0.00406126:
        z += -32.5128 * Q.girth2_top10 - 0.02918892
    if 0.00406126 <= Q.girth2_top10 < 0.00625621:
        z += 73.4558 * Q.girth2_top10 - 0.459555
    if Q.e2 < 0.02515919:
        z += -51.32771 * Q.e2 + 1.147573
    if 0.02515919 <= Q.e2 < 0.03029714:
        z += -3.323488 * Q.e2 - 0.06017441
    if 0.03029714 <= Q.e2 < 0.03875945:
        z += 19.00979 * Q.e2 - 0.736809
    if Q.girth2_top30 < 0.005402331:
        z += 78.54224 * Q.girth2_top30 - 0.6879496
    if 0.005402331 <= Q.girth2_top30 < 0.006929741:
        z += 172.6048 * Q.girth2_top30 - 1.196107
    if 2.178951 <= Q.D2 < 5.378975:
        z += 0.05016959 * Q.D2 - 0.1093171
    if Q.D2 >= 5.378975:
        z += -0.02386743 * Q.D2 + 0.2889262
    if Q.girth2_top15 < 0.009962397:
        z += -6.817195 * Q.girth2_top15 + 0.06791561
    if Q.psi_0p1 >= 0.9770626:
        z += -16.48672 * Q.psi_0p1 + 16.10856
    if Q.mass_top30 < 60.43821:
        z += -0.006691034 * Q.mass_top30 + 0.4043941
    if Q.mass_top40 < 67.72643:
        z += 0.005091503 * Q.mass_top40 - 0.6854185
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.02183389 * Q.mass_top40 - 1.819321
    if Q.girth < 0.07374472:
        z += 7.919642 * Q.girth - 0.3885003
    if 0.07374472 <= Q.girth < 0.076787:
        z += 10.64752 * Q.girth - 0.5896669
    if 0.076787 <= Q.girth < 0.08589404:
        z += -25.02725 * Q.girth + 2.149692
    if Q.e2_sq < 0.00818374:
        z += -153.8366 * Q.e2_sq + 1.258959
    if Q.mass_over_sum_pt < 0.06030419:
        z += 21.25331 * Q.mass_over_sum_pt - 1.281664
    if Q.lam1 < 0.006716737:
        z += 97.15854 * Q.lam1 - 0.6525883
    if Q.z_top40_slots >= 0.996191:
        z += 18.09914 * Q.z_top40_slots - 18.0302
    if Q.e3 < 3.793233e-05:
        z += 10493.41 * Q.e3 - 0.3980394
    if Q.sj2_mass1 < 19.89956:
        z += 0.01950409 * Q.sj2_mass1 - 0.3881228
    if Q.mass_top50 < 80.35535:
        z += 0.0178226 * Q.mass_top50 - 1.432141
    if 0.9896594 <= Q.psi_0p3 < 0.9980008:
        z += 4.783407 * Q.psi_0p3 - 4.733943
    if Q.psi_0p3 >= 0.9980008:
        z += 100.7148 * Q.psi_0p3 - 100.4735
    if Q.C2_b2 >= 0.01330402:
        z += -4.779765 * Q.C2_b2 + 0.06359011
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_particles > 34.0:
        z += 0.0002796254 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_particles - 34.0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2 < 0.005532208:
        z += -39.65958 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.005436176 * (10.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.mass < 101.0497 and Q.planar_flow < 0.3563114:
        z += 0.09420407 * (101.0497 - Q.mass) * (0.3563114 - Q.planar_flow)
    if Q.mass_over_sum_pt_sq < 0.009595015 and Q.z_dr_0p1_0p2 > 0.1203437:
        z += 928.95 * (0.009595015 - Q.mass_over_sum_pt_sq) * (Q.z_dr_0p1_0p2 - 0.1203437)
    if Q.mass < 101.0497 and Q.sum_pt_top10 < 891.875:
        z += -9.332142e-06 * (101.0497 - Q.mass) * (891.875 - Q.sum_pt_top10)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.z_6 > 0.02949408:
        z += -0.01567129 * (10.0 - Q.n_dr_0p2_0p4) * (Q.z_6 - 0.02949408)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.pair_mass_0_12 > 7.607175:
        z += 0.002396064 * (10.0 - Q.n_dr_0p2_0p4) * (Q.pair_mass_0_12 - 7.607175)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_real_top40 > 26.0:
        z += -0.002516262 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_real_top40 - 26.0)
    if Q.mass < 101.0497 and Q.pt1_dr01 > 12.6865:
        z += 0.0009922065 * (101.0497 - Q.mass) * (Q.pt1_dr01 - 12.6865)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.pt_11 > 6.632422:
        z += 0.001482323 * (10.0 - Q.n_dr_0p2_0p4) * (Q.pt_11 - 6.632422)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.pair_mass_0_10 > 7.661644:
        z += -9.056742e-06 * (10.0 - Q.n_dr_0p2_0p4) * (Q.pair_mass_0_10 - 7.661644)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.pair_mass_0_13 > 8.669189:
        z += -0.0008564008 * (10.0 - Q.n_dr_0p2_0p4) * (Q.pair_mass_0_13 - 8.669189)
    if Q.mass < 101.0497 and Q.ptdr0_2 > 11.87288:
        z += 0.0003162032 * (101.0497 - Q.mass) * (Q.ptdr0_2 - 11.87288)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.dr_8 > 0.05067946:
        z += -0.02001731 * (10.0 - Q.n_dr_0p2_0p4) * (Q.dr_8 - 0.05067946)
    if Q.z_top40_slots > 0.996191 and Q.C2_b2 < 0.006580753:
        z += 6681.725 * (Q.z_top40_slots - 0.996191) * (0.006580753 - Q.C2_b2)
    if Q.sj2_mass1 < 19.89956 and Q.dr_9 < 0.1147522:
        z += 0.005359197 * (19.89956 - Q.sj2_mass1) * (0.1147522 - Q.dr_9)
    if Q.mass < 80.4 and Q.pt1_over_pt0 < 0.8128995:
        z += -0.003922661 * (80.4 - Q.mass) * (0.8128995 - Q.pt1_over_pt0)
    if Q.psi_0p2 > 0.9935324 and Q.dr_11 > 0.05788126:
        z += -145.8338 * (Q.psi_0p2 - 0.9935324) * (Q.dr_11 - 0.05788126)
    if Q.girth < 0.08589404 and Q.soft3_dr > 0.161871:
        z += -1.281056 * (0.08589404 - Q.girth) * (Q.soft3_dr - 0.161871)
    if Q.psi_0p2 > 0.9313699 and Q.soft5_dr0 < 0.3897349:
        z += -1.97594 * (Q.psi_0p2 - 0.9313699) * (0.3897349 - Q.soft5_dr0)
    if Q.psi_0p2 > 0.9313699 and Q.dr_13 < 0.08052407:
        z += 21.811 * (Q.psi_0p2 - 0.9313699) * (0.08052407 - Q.dr_13)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.absphi_7 > 0.05749512:
        z += 0.2945826 * (10.0 - Q.n_dr_0p2_0p4) * (Q.absphi_7 - 0.05749512)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.phi_0 < 0.0402832:
        z += 0.05826379 * (10.0 - Q.n_dr_0p2_0p4) * (0.0402832 - Q.phi_0)
    if Q.sj2_mass1 < 19.89956 and Q.soft10_abseta < 0.002968979:
        z += 1.08863 * (19.89956 - Q.sj2_mass1) * (0.002968979 - Q.soft10_abseta)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.2560775
    if Q.mass < 53.87362:
        z += -0.06287881 * Q.mass + 5.521636
    if 53.87362 <= Q.mass < 62.55:
        z += -0.06303564 * Q.mass + 5.530085
    if 62.55 <= Q.mass < 74.25181:
        z += -0.09907573 * Q.mass + 7.784393
    if 74.25181 <= Q.mass < 82.85409:
        z += -0.0561484 * Q.mass + 4.596961
    if 82.85409 <= Q.mass < 86.4:
        z += 0.01555703 * Q.mass - 1.344127
    if Q.mass >= 160.8:
        z += 0.01020373 * Q.mass - 1.64076
    if Q.sd_mass >= 125.1:
        z += 0.03299513 * Q.sd_mass - 4.127691
    if Q.mass_top40 < 74.78616:
        z += 0.00995324 * Q.mass_top40 - 0.6607649
    if 74.78616 <= Q.mass_top40 < 89.6788:
        z += -0.005613497 * Q.mass_top40 + 0.5034116
    if Q.sj3_pair_mass_max >= 120.6:
        z += -0.003140201 * Q.sj3_pair_mass_max + 0.3787083
    if Q.mass_over_sum_pt < 0.05077291:
        z += 19.19063 * Q.mass_over_sum_pt - 1.35014
    if 0.05077291 <= Q.mass_over_sum_pt < 0.06895248:
        z += 20.67022 * Q.mass_over_sum_pt - 1.425263
    if Q.girth2_top10 < 0.0007431905:
        z += -187.9143 * Q.girth2_top10 + 0.1396561
    if Q.lam1 < 0.002752094:
        z += -114.1806 * Q.lam1 + 0.3142357
    if Q.log_sum_pt < 6.856375:
        z += 2.520634 * Q.log_sum_pt - 17.28241
    if Q.girth2_top15 < 0.006142802:
        z += 6.543069 * Q.girth2_top15 - 0.04019278
    if Q.mass_top15 < 24.20196:
        z += -0.003102002 * Q.mass_top15 + 0.07507451
    if 0.9538343 <= Q.psi_0p1 < 0.9770626:
        z += 5.486583 * Q.psi_0p1 - 5.233291
    if Q.psi_0p1 >= 0.9770626:
        z += 21.96933 * Q.psi_0p1 - 21.33797
    if Q.z_dr_0_0p05 >= 0.9351131:
        z += 1.714455 * Q.z_dr_0_0p05 - 1.603209
    if Q.z_dr_0p1_0p2 >= 0.6882177:
        z += 1.550445 * Q.z_dr_0p1_0p2 - 1.067044
    if Q.soft5_pt < 1.339844:
        z += 0.2306321 * Q.soft5_pt - 0.309011
    if Q.psi_0p3 >= 0.9973959:
        z += -62.21038 * Q.psi_0p3 + 62.04837
    if Q.soft4_dr < 0.1464158:
        z += 1.086857 * Q.soft4_dr - 0.1591331
    if Q.mass < 86.4 and Q.tau21 < 0.5100475:
        z += 0.07386799 * (86.4 - Q.mass) * (0.5100475 - Q.tau21)
    if Q.mass < 86.4 and Q.z_dr_0p1_0p2 < 0.06472584:
        z += -0.2184681 * (86.4 - Q.mass) * (0.06472584 - Q.z_dr_0p1_0p2)
    if Q.sd_mass > 125.1 and Q.sd_zg < 0.4199841:
        z += -0.05579668 * (Q.sd_mass - 125.1) * (0.4199841 - Q.sd_zg)
    if Q.mass_top40 < 89.6788 and Q.n_dr_0p05_0p1 > 1.0:
        z += 0.0005826664 * (89.6788 - Q.mass_top40) * (Q.n_dr_0p05_0p1 - 1.0)
    if Q.mass_top40 < 89.6788 and Q.pt_0 < 334.5:
        z += -1.237738e-05 * (89.6788 - Q.mass_top40) * (334.5 - Q.pt_0)
    if Q.mass < 86.4 and Q.psi_0p3 < 0.9985421:
        z += -2.139421 * (86.4 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.sj3_pair_mass_max > 120.6 and Q.sj3_pair_mass_min < 76.60223:
        z += 0.0004023165 * (Q.sj3_pair_mass_max - 120.6) * (76.60223 - Q.sj3_pair_mass_min)
    if Q.mass < 86.4 and Q.zdr_0 > 0.007860787:
        z += 0.4412542 * (86.4 - Q.mass) * (Q.zdr_0 - 0.007860787)
    if Q.sj3_pair_mass_max > 120.6 and Q.z_dr_0p1_0p2 > 0.2864926:
        z += 0.03405564 * (Q.sj3_pair_mass_max - 120.6) * (Q.z_dr_0p1_0p2 - 0.2864926)
    if Q.sj3_pair_mass_max > 120.6 and Q.soft9_z < 0.00104408:
        z += 76.78864 * (Q.sj3_pair_mass_max - 120.6) * (0.00104408 - Q.soft9_z)
    if Q.girth2_top15 < 0.006142802 and Q.pt_5 < 48.28125:
        z += 1.886717 * (0.006142802 - Q.girth2_top15) * (48.28125 - Q.pt_5)
    if Q.mass < 86.4 and Q.lam2 < 0.003687605:
        z += 10.03014 * (86.4 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.girth2_top10 < 0.0007431905 and Q.sum_pt_top40 < 1069.671:
        z += 3.820784 * (0.0007431905 - Q.girth2_top10) * (1069.671 - Q.sum_pt_top40)
    if Q.girth2_top15 < 0.006142802 and Q.sum_pt_top40 > 1225.842:
        z += -0.2809406 * (0.006142802 - Q.girth2_top15) * (Q.sum_pt_top40 - 1225.842)
    if Q.girth2_top15 < 0.006142802 and Q.z_10 > 0.01143749:
        z += -1869.933 * (0.006142802 - Q.girth2_top15) * (Q.z_10 - 0.01143749)
    if Q.mass < 86.4 and Q.z_top50_slots < 0.985099:
        z += -1.983697 * (86.4 - Q.mass) * (0.985099 - Q.z_top50_slots)
    if Q.sj3_pair_mass_max > 120.6 and Q.sj3_mass1 < 32.50209:
        z += 0.0007513441 * (Q.sj3_pair_mass_max - 120.6) * (32.50209 - Q.sj3_mass1)
    if Q.z_dr_0p1_0p2 > 0.6882177 and Q.sj3_z3 < 0.2502892:
        z += -10.21707 * (Q.z_dr_0p1_0p2 - 0.6882177) * (0.2502892 - Q.sj3_z3)
    if Q.log_sum_pt < 6.856375 and Q.lam2 < 0.002396991:
        z += -2331.401 * (6.856375 - Q.log_sum_pt) * (0.002396991 - Q.lam2)
    if Q.mass < 74.25181 and Q.dr_max_012 > 0.1206357:
        z += 1.149657 * (74.25181 - Q.mass) * (Q.dr_max_012 - 0.1206357)
    if Q.z_dr_0p1_0p2 > 0.6882177 and Q.n_dr_0p4_up > 0.0:
        z += 3.553477 * (Q.z_dr_0p1_0p2 - 0.6882177) * (Q.n_dr_0p4_up - 0.0)
    if Q.sd_mass > 125.1 and Q.girth2_top3 > 0.01002369:
        z += 0.3741115 * (Q.sd_mass - 125.1) * (Q.girth2_top3 - 0.01002369)
    if Q.sd_mass > 125.1 and Q.lam2 > 0.0001679609:
        z += -2.225376 * (Q.sd_mass - 125.1) * (Q.lam2 - 0.0001679609)
    if Q.z_dr_0p1_0p2 > 0.6882177 and Q.soft4_absphi > 0.170166:
        z += -5.40767 * (Q.z_dr_0p1_0p2 - 0.6882177) * (Q.soft4_absphi - 0.170166)
    if Q.psi_0p3 > 0.9973959 and Q.sj3_z2 < 0.1873652:
        z += 183.2366 * (Q.psi_0p3 - 0.9973959) * (0.1873652 - Q.sj3_z2)
    if Q.girth2_top10 < 0.0007431905 and Q.n_real_top40 < 36.0:
        z += 1.905949 * (0.0007431905 - Q.girth2_top10) * (36.0 - Q.n_real_top40)
    if Q.sj3_pair_mass_max > 120.6 and Q.orientation_deg > 35.6254:
        z += -0.0001165377 * (Q.sj3_pair_mass_max - 120.6) * (Q.orientation_deg - 35.6254)
    if Q.sj3_pair_mass_max > 120.6 and Q.orientation_deg < -62.72059:
        z += -0.0004447508 * (Q.sj3_pair_mass_max - 120.6) * (-62.72059 - Q.orientation_deg)
    if Q.z_dr_0_0p05 > 0.9351131 and Q.n_real_top15 < 15.0:
        z += 6.95191 * (Q.z_dr_0_0p05 - 0.9351131) * (15.0 - Q.n_real_top15)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.428855
    if Q.sum_pt < 1007.788:
        z += 0.02183255 * Q.sum_pt - 22.83428
    if 1007.788 <= Q.sum_pt < 1085.125:
        z += 0.01075407 * Q.sum_pt - 11.66951
    if 74.25181 <= Q.mass < 136.785:
        z += 0.02935312 * Q.mass - 2.179522
    if 136.785 <= Q.mass < 143.7876:
        z += -0.05468041 * Q.mass + 9.315004
    if 143.7876 <= Q.mass < 160.8:
        z += -0.1610739 * Q.mass + 24.61307
    if 160.8 <= Q.mass < 162.8363:
        z += -0.3144988 * Q.mass + 49.2838
    if 162.8363 <= Q.mass < 172.8:
        z += -0.1972914 * Q.mass + 30.19816
    if Q.mass >= 172.8:
        z += -0.06354366 * Q.mass + 7.08656
    if Q.n_for_90pct < 16.0:
        z += 0.03237514 * Q.n_for_90pct - 0.5180022
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 229.3571 * Q.mass_over_sum_pt - 39.19256
    if Q.log_sum_pt < 6.811175:
        z += -10.6724 * Q.log_sum_pt + 71.34518
    if 6.811175 <= Q.log_sum_pt < 6.910131:
        z += 13.60634 * Q.log_sum_pt - 94.02161
    if Q.log_sum_pt >= 7.139296:
        z += 0.4847476 * Q.log_sum_pt - 3.460756
    if Q.sum_pt_top40 < 1007.44:
        z += -0.006989861 * Q.sum_pt_top40 + 7.217079
    if 1007.44 <= Q.sum_pt_top40 < 1053.047:
        z += -0.003841775 * Q.sum_pt_top40 + 4.045571
    if Q.mass_top10 < 56.92192:
        z += -0.0006944056 * Q.mass_top10 + 0.05346132
    if 56.92192 <= Q.mass_top10 < 76.9886:
        z += -0.003578427 * Q.mass_top10 + 0.2176254
    if Q.mass_top10 >= 76.9886:
        z += -0.002884022 * Q.mass_top10 + 0.1641641
    if Q.C2 >= 0.0977156:
        z += -2.839429 * Q.C2 + 0.2774565
    if 92.16545 <= Q.mass_top50 < 136.785:
        z += -0.02816828 * Q.mass_top50 + 2.596143
    if 136.785 <= Q.mass_top50 < 168.9698:
        z += 0.05244564 * Q.mass_top50 - 8.430632
    if Q.mass_top50 >= 168.9698:
        z += 0.003022686 * Q.mass_top50 - 0.07964809
    if Q.n_dr_0p1_0p2 < 17.0:
        z += -0.002600321 * Q.n_dr_0p1_0p2 + 0.04420545
    if Q.girth2_top50 < 0.007820315:
        z += -121.47 * Q.girth2_top50 + 1.731437
    if 0.007820315 <= Q.girth2_top50 < 0.02550569:
        z += -44.18922 * Q.girth2_top50 + 1.127077
    if Q.pt_entropy < 2.752054:
        z += -0.02273919 * Q.pt_entropy + 0.06257946
    if Q.tau1 < 0.08286256:
        z += -7.228122 * Q.tau1 + 0.5989407
    if Q.tau1 >= 0.1751567:
        z += -11.90456 * Q.tau1 + 2.085164
    if Q.mass_over_sum_pt_sq < 0.02580396:
        z += 49.51387 * Q.mass_over_sum_pt_sq - 1.277654
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -539.0728 * Q.mass_over_sum_pt_sq + 15.74093
    if Q.psi_0p3 >= 0.9638082:
        z += 9.767582 * Q.psi_0p3 - 9.414075
    z += 0.008967828 * Q.n_particles
    if Q.mass_top40 < 160.8:
        z += 0.00361606 * Q.mass_top40 - 0.5814625
    if Q.sum_pt_top50 < 997.0189:
        z += -0.004889853 * Q.sum_pt_top50 + 4.875276
    if Q.sum_pt_top30 < 966.0633:
        z += 0.00353835 * Q.sum_pt_top30 - 3.41827
    if Q.sum_pt_top30 >= 1111.245:
        z += 0.004705779 * Q.sum_pt_top30 - 5.229274
    if Q.mass_top5 >= 33.71058:
        z += 0.004023161 * Q.mass_top5 - 0.1356231
    if Q.pt_11 < 18.32812:
        z += -0.008209981 * Q.pt_11 + 0.1504736
    if Q.z_dr_0p1_0p2 < 0.08871546:
        z += -1.124946 * Q.z_dr_0p1_0p2 + 0.09980007
    if Q.sum_pt_top20 >= 956.5062:
        z += -0.0001520382 * Q.sum_pt_top20 + 0.1454255
    if Q.pt_6 < 41.21875:
        z += -0.009440022 * Q.pt_6 + 0.3891059
    if Q.sum_pt < 1085.125 and Q.sum_pt_top40 < 858.8262:
        z += 6.882021e-06 * (1085.125 - Q.sum_pt) * (858.8262 - Q.sum_pt_top40)
    if Q.sum_pt < 1085.125 and Q.tau21_b2 < 0.8310045:
        z += -0.005324858 * (1085.125 - Q.sum_pt) * (0.8310045 - Q.tau21_b2)
    if Q.n_for_90pct < 16.0 and Q.D2 < 3.814159:
        z += 0.02949713 * (16.0 - Q.n_for_90pct) * (3.814159 - Q.D2)
    if Q.sum_pt_top40 < 1053.047 and Q.sj2_zsoft < 0.2179035:
        z += 0.02315834 * (1053.047 - Q.sum_pt_top40) * (0.2179035 - Q.sj2_zsoft)
    if Q.mass > 136.785 and Q.sum_pt < 1028.184:
        z += 0.0004468836 * (Q.mass - 136.785) * (1028.184 - Q.sum_pt)
    if Q.mass > 74.25181 and Q.sum_pt < 1028.184:
        z += -0.0001658437 * (Q.mass - 74.25181) * (1028.184 - Q.sum_pt)
    if Q.sum_pt < 1007.788 and Q.sum_pt_top3 > 512.4375:
        z += 1.016203e-05 * (1007.788 - Q.sum_pt) * (Q.sum_pt_top3 - 512.4375)
    if Q.log_sum_pt > 7.139296 and Q.C3 < 0.00317537:
        z += -466.2538 * (Q.log_sum_pt - 7.139296) * (0.00317537 - Q.C3)
    if Q.log_sum_pt > 7.139296 and Q.dr_12 < 0.05398073:
        z += 37.99263 * (Q.log_sum_pt - 7.139296) * (0.05398073 - Q.dr_12)
    if Q.log_sum_pt < 6.811175 and Q.dr_13 < 0.1312677:
        z += 28.27872 * (6.811175 - Q.log_sum_pt) * (0.1312677 - Q.dr_13)
    if Q.psi_0p3 > 0.9638082 and Q.max_dr < 0.4357228:
        z += 20.00549 * (Q.psi_0p3 - 0.9638082) * (0.4357228 - Q.max_dr)
    if Q.sum_pt < 1085.125 and Q.dr_13 < 0.1023813:
        z += -0.007144035 * (1085.125 - Q.sum_pt) * (0.1023813 - Q.dr_13)
    if Q.psi_0p3 > 0.9638082 and Q.sj3_dr13 > 0.2465017:
        z += -72.20446 * (Q.psi_0p3 - 0.9638082) * (Q.sj3_dr13 - 0.2465017)
    if Q.mass > 172.8 and Q.pt_6 < 56.53125:
        z += -0.001609088 * (Q.mass - 172.8) * (56.53125 - Q.pt_6)
    if Q.psi_0p3 > 0.9638082 and Q.C3 < 0.0223982:
        z += 328.2464 * (Q.psi_0p3 - 0.9638082) * (0.0223982 - Q.C3)
    if Q.psi_0p3 > 0.9638082 and Q.absphi_12 > 0.01163483:
        z += 38.80463 * (Q.psi_0p3 - 0.9638082) * (Q.absphi_12 - 0.01163483)
    if Q.mass > 172.8 and Q.z_6 < 0.04568661:
        z += 0.7627918 * (Q.mass - 172.8) * (0.04568661 - Q.z_6)
    if Q.log_sum_pt > 7.139296 and Q.sd_zg > 0.4462823:
        z += 386.9222 * (Q.log_sum_pt - 7.139296) * (Q.sd_zg - 0.4462823)
    if Q.log_sum_pt < 6.811175 and Q.zdr_11 < 0.005041702:
        z += 574.7722 * (6.811175 - Q.log_sum_pt) * (0.005041702 - Q.zdr_11)
    if Q.C2 > 0.0977156 and Q.orientation_deg < 53.9394:
        z += -0.01241506 * (Q.C2 - 0.0977156) * (53.9394 - Q.orientation_deg)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.1230559
    if Q.tau21_b2 < 0.342495:
        z += -0.9760696 * Q.tau21_b2 + 0.3342989
    if Q.mass < 80.4:
        z += 0.03002538 * Q.mass - 2.872029
    if 80.4 <= Q.mass < 82.85409:
        z += 0.09156607 * Q.mass - 7.8199
    if 82.85409 <= Q.mass < 91.19:
        z += 0.08816495 * Q.mass - 7.538103
    if 91.19 <= Q.mass < 125.1:
        z += -0.01479382 * Q.mass + 1.850707
    if Q.psi_0p3 >= 0.9924477:
        z += 51.97144 * Q.psi_0p3 - 51.57893
    if Q.N2 < 0.4226723:
        z += -0.1057666 * Q.N2 + 0.04470463
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.005050383 * Q.n_dr_0p2_0p4 + 0.09090689
    if Q.pt_10 >= 28.15625:
        z += 0.0003685527 * Q.pt_10 - 0.01037706
    if Q.e2 < 0.02210818:
        z += -1.771788 * Q.e2 + 0.2158612
    if 0.02210818 <= Q.e2 < 0.03029714:
        z += -21.57663 * Q.e2 + 0.6537103
    if Q.mass_over_sum_pt < 0.07852883:
        z += -26.131 * Q.mass_over_sum_pt + 3.81265
    if 0.07852883 <= Q.mass_over_sum_pt < 0.08873143:
        z += -19.72547 * Q.mass_over_sum_pt + 3.309632
    if 0.08873143 <= Q.mass_over_sum_pt < 0.09046749:
        z += 17.32647 * Q.mass_over_sum_pt + 0.02196025
    if 0.09046749 <= Q.mass_over_sum_pt < 0.09795415:
        z += -74.59677 * Q.mass_over_sum_pt + 8.338025
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -41.93739 * Q.mass_over_sum_pt + 5.138904
    if 0.1182259 <= Q.mass_over_sum_pt < 0.140939:
        z += -7.960886 * Q.mass_over_sum_pt + 1.121999
    if Q.girth2_top20 < 0.006374178:
        z += -21.60123 * Q.girth2_top20 - 0.1234591
    if 0.006374178 <= Q.girth2_top20 < 0.01083435:
        z += 58.55133 * Q.girth2_top20 - 0.6343658
    if Q.sj3_mass1 < 32.50209:
        z += -0.002619785 * Q.sj3_mass1 + 0.08514848
    if Q.girth < 0.076787:
        z += 10.18829 * Q.girth - 0.7823281
    if Q.girth2_top5 < 0.007164202:
        z += -51.76278 * Q.girth2_top5 + 0.370839
    if Q.sd_mass < 69.65633:
        z += -0.002727838 * Q.sd_mass + 0.1900112
    if Q.sd_rg < 0.1778185:
        z += -1.560392 * Q.sd_rg + 0.1218517
    if 0.1778185 <= Q.sd_rg < 0.3017146:
        z += 1.256011 * Q.sd_rg - 0.3789568
    if Q.girth2_top30 < 0.01215787:
        z += 126.3735 * Q.girth2_top30 - 1.536433
    if Q.n_pt_above_1 >= 58.0:
        z += 8.891738e-05 * Q.n_pt_above_1 - 0.005157208
    if Q.tau1 < 0.06310829:
        z += -4.467049 * Q.tau1 - 0.05348473
    if 0.06310829 <= Q.tau1 < 0.1072713:
        z += 7.594429 * Q.tau1 - 0.814664
    if Q.mass_top30 < 82.66587:
        z += -0.008185985 * Q.mass_top30 + 0.6767016
    if Q.sum_pt_top40 < 906.6023:
        z += 0.0007768222 * Q.sum_pt_top40 - 0.5010188
    if 906.6023 <= Q.sum_pt_top40 < 1024.942:
        z += -0.001717508 * Q.sum_pt_top40 + 1.760347
    if Q.girth2_top3 < 0.001592178:
        z += -24.1811 * Q.girth2_top3 + 0.0385006
    if Q.log_sum_pt < 6.941997:
        z += -3.171865 * Q.log_sum_pt + 21.97703
    if 6.941997 <= Q.log_sum_pt < 6.98945:
        z += 0.8860317 * Q.log_sum_pt - 6.192875
    if Q.sum_pt_top50 < 976.277:
        z += 0.001170699 * Q.sum_pt_top50 - 1.142927
    if Q.lam2 < 0.001163277:
        z += -40.36939 * Q.lam2 + 0.0469608
    if Q.z_dr_0p2_0p4 < 0.1291856:
        z += -0.04418729 * Q.z_dr_0p2_0p4 + 0.005708361
    if Q.tau21_b2 < 0.342495 and Q.girth2_top40 < 0.007709916:
        z += -979.1493 * (0.342495 - Q.tau21_b2) * (0.007709916 - Q.girth2_top40)
    if Q.tau21_b2 < 0.342495 and Q.lam1 < 0.02048524:
        z += 6.19083 * (0.342495 - Q.tau21_b2) * (0.02048524 - Q.lam1)
    if Q.tau21_b2 < 0.342495 and Q.girth2_top50 < 0.006118006:
        z += 390.4126 * (0.342495 - Q.tau21_b2) * (0.006118006 - Q.girth2_top50)
    if Q.N2 < 0.4226723 and Q.max_dr > 0.2404747:
        z += -3.593737 * (0.4226723 - Q.N2) * (Q.max_dr - 0.2404747)
    if Q.N2 < 0.4226723 and Q.dr_12 < 0.2053949:
        z += 0.9870913 * (0.4226723 - Q.N2) * (0.2053949 - Q.dr_12)
    if Q.mass < 125.1 and Q.mass_top2 > 28.78966:
        z += -0.0006806714 * (125.1 - Q.mass) * (Q.mass_top2 - 28.78966)
    if Q.psi_0p3 > 0.9924477 and Q.sd_rg > 0.2280025:
        z += 47.335 * (Q.psi_0p3 - 0.9924477) * (Q.sd_rg - 0.2280025)
    if Q.psi_0p3 > 0.9924477 and Q.sd_rg < 0.1881908:
        z += -568.7717 * (Q.psi_0p3 - 0.9924477) * (0.1881908 - Q.sd_rg)
    if Q.tau21_b2 < 0.342495 and Q.n_real_top40 < 32.0:
        z += -0.183996 * (0.342495 - Q.tau21_b2) * (32.0 - Q.n_real_top40)
    if Q.tau21_b2 < 0.342495 and Q.n_for_50pct > 3.0:
        z += 0.03482064 * (0.342495 - Q.tau21_b2) * (Q.n_for_50pct - 3.0)
    if Q.mass_over_sum_pt < 0.09046749 and Q.psi_0p2 > 0.9087063:
        z += -391.335 * (0.09046749 - Q.mass_over_sum_pt) * (Q.psi_0p2 - 0.9087063)
    if Q.girth2_top5 < 0.007164202 and Q.zdr_5 > 0.008949744:
        z += -28252.96 * (0.007164202 - Q.girth2_top5) * (Q.zdr_5 - 0.008949744)
    if Q.tau21_b2 < 0.342495 and Q.sj3_mass2 > 5.762132:
        z += -0.04441259 * (0.342495 - Q.tau21_b2) * (Q.sj3_mass2 - 5.762132)
    if Q.girth2_top5 < 0.007164202 and Q.pair_mass_0_4 > 8.587823:
        z += 1.508144 * (0.007164202 - Q.girth2_top5) * (Q.pair_mass_0_4 - 8.587823)
    if Q.psi_0p3 > 0.9924477 and Q.z_2nd < 0.1231747:
        z += -537.88 * (Q.psi_0p3 - 0.9924477) * (0.1231747 - Q.z_2nd)
    if Q.tau21_b2 < 0.342495 and Q.n_pt_above_50 < 7.0:
        z += 0.08754131 * (0.342495 - Q.tau21_b2) * (7.0 - Q.n_pt_above_50)
    if Q.girth2_top5 < 0.007164202 and Q.n_pt_above_10 > 13.0:
        z += -1.963011 * (0.007164202 - Q.girth2_top5) * (Q.n_pt_above_10 - 13.0)
    if Q.girth < 0.076787 and Q.z_dr_0p2_0p4 < 0.05180474:
        z += 163.1277 * (0.076787 - Q.girth) * (0.05180474 - Q.z_dr_0p2_0p4)
    if Q.girth2_top30 < 0.01215787 and Q.pair_mass_0_2 > 22.41128:
        z += -5.890948 * (0.01215787 - Q.girth2_top30) * (Q.pair_mass_0_2 - 22.41128)
    if Q.mass < 91.19 and Q.n_dr_0p05_0p1 > 14.0:
        z += -0.001035644 * (91.19 - Q.mass) * (Q.n_dr_0p05_0p1 - 14.0)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.M3 < 0.03787151:
        z += 0.1639908 * (18.0 - Q.n_dr_0p2_0p4) * (0.03787151 - Q.M3)
    if Q.girth2_top5 < 0.007164202 and Q.sj2_mass2 > 14.45919:
        z += 1.866955 * (0.007164202 - Q.girth2_top5) * (Q.sj2_mass2 - 14.45919)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.1535886
    if Q.z_dr_0_0p05 < 0.8459004:
        z += -0.3475081 * Q.z_dr_0_0p05 + 0.2939572
    if Q.girth2_top20 < 0.00287991:
        z += -29.31889 * Q.girth2_top20 + 0.08443576
    if Q.z_dr_0p1_0p2 < 0.06472584:
        z += -9.269065 * Q.z_dr_0p1_0p2 + 0.884534
    if 0.06472584 <= Q.z_dr_0p1_0p2 < 0.1203437:
        z += -5.116812 * Q.z_dr_0p1_0p2 + 0.615776
    if Q.girth2_top5 < 0.002270363:
        z += 41.36952 * Q.girth2_top5 + 0.4322475
    if 0.002270363 <= Q.girth2_top5 < 0.008329695:
        z += -86.83652 * Q.girth2_top5 + 0.7233217
    if Q.girth2_top2 < 0.0005124533:
        z += -40.31256 * Q.girth2_top2 - 0.1593924
    if 0.0005124533 <= Q.girth2_top2 < 0.001056655:
        z += 330.8529 * Q.girth2_top2 - 0.3495974
    if 0.8976117 <= Q.psi_0p1 < 0.9371031:
        z += -5.923004 * Q.psi_0p1 + 5.316558
    if Q.psi_0p1 >= 0.9371031:
        z += -12.93388 * Q.psi_0p1 + 11.88647
    if Q.sum_pt < 986.0565:
        z += 0.01049902 * Q.sum_pt - 10.44461
    if 986.0565 <= Q.sum_pt < 1002.379:
        z += 0.005635399 * Q.sum_pt - 5.648803
    if Q.log_sum_pt < 6.903423:
        z += -0.5111456 * Q.log_sum_pt + 4.217186
    if 6.903423 <= Q.log_sum_pt < 7.017258:
        z += -6.048519 * Q.log_sum_pt + 42.44402
    if Q.psi_0p3 >= 0.9896594:
        z += -21.32325 * Q.psi_0p3 + 21.10276
    if 0.2232169 <= Q.sj2_dr < 0.2971569:
        z += 3.320331 * Q.sj2_dr - 0.741154
    if Q.sj2_dr >= 0.2971569:
        z += 1.184897 * Q.sj2_dr - 0.1065949
    if Q.girth2_top10 < 0.007678544:
        z += -62.14941 * Q.girth2_top10 + 0.477217
    if Q.lam1 < 0.004673423:
        z += -72.68507 * Q.lam1 + 0.3396881
    if Q.lam1 >= 0.01649354:
        z += -47.61117 * Q.lam1 + 0.785277
    if Q.n_dr_0p1_0p2 < 13.0:
        z += -0.02350984 * Q.n_dr_0p1_0p2 + 0.3056279
    if Q.D2 < 1.976207:
        z += 0.3566026 * Q.D2 - 0.7047206
    if Q.lam2 < 0.001776308:
        z += -122.5399 * Q.lam2 + 0.2176687
    if Q.tau1 < 0.0705748:
        z += 18.4698 * Q.tau1 - 1.303502
    if Q.sum_pt_top20 < 1017.778:
        z += 0.0006049339 * Q.sum_pt_top20 - 0.6156885
    if Q.mass_over_sum_pt >= 0.1708801:
        z += -0.3805955 * Q.mass_over_sum_pt + 0.0650362
    if Q.z_dr_0p2_0p4 < 0.002288114:
        z += 135.5464 * Q.z_dr_0p2_0p4 - 0.3101457
    if Q.n_dr_0p2_0p4 >= 9.0:
        z += -0.01675646 * Q.n_dr_0p2_0p4 + 0.1508082
    if Q.sd_mass < 45.595:
        z += -3.533717e-05 * Q.sd_mass + 0.3553142
    if 45.595 <= Q.sd_mass < 79.18312:
        z += -0.0105306 * Q.sd_mass + 0.8338457
    if Q.sj3_pair_mass_max < 35.30013:
        z += -0.02673659 * Q.sj3_pair_mass_max + 0.9438051
    if Q.mass_top50 < 71.79516:
        z += 0.009962874 * Q.mass_top50 - 0.7152862
    if Q.sum_pt_top40 < 935.8189:
        z += 0.000394007 * Q.sum_pt_top40 - 0.7147919
    if 935.8189 <= Q.sum_pt_top40 < 1018.698:
        z += 0.004175647 * Q.sum_pt_top40 - 4.253722
    if Q.dr_7 >= 0.2229947:
        z += 3.173988 * Q.dr_7 - 0.7077826
    if Q.sum_pt_top30 < 933.1875:
        z += -0.002853096 * Q.sum_pt_top30 + 2.662473
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.sd_mass < 76.29481:
        z += -0.007209626 * (0.1203437 - Q.z_dr_0p1_0p2) * (76.29481 - Q.sd_mass)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.planar_flow < 0.777038:
        z += 3.57041 * (0.1203437 - Q.z_dr_0p1_0p2) * (0.777038 - Q.planar_flow)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_particles < 58.0:
        z += -0.01627764 * (0.1203437 - Q.z_dr_0p1_0p2) * (58.0 - Q.n_particles)
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 949.9169:
        z += 0.006898817 * (0.8459004 - Q.z_dr_0_0p05) * (949.9169 - Q.sum_pt)
    if Q.girth2_top5 < 0.008329695 and Q.psi_0p2 < 0.948102:
        z += -315.4413 * (0.008329695 - Q.girth2_top5) * (0.948102 - Q.psi_0p2)
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 1260.541:
        z += -0.001825193 * (0.8459004 - Q.z_dr_0_0p05) * (1260.541 - Q.sum_pt)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_dr_0p2_0p4 > 5.0:
        z += -0.1632325 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p2_0p4 - 5.0)
    if Q.girth2_top20 < 0.00287991 and Q.sum_pt > 1167.447:
        z += -0.1792287 * (0.00287991 - Q.girth2_top20) * (Q.sum_pt - 1167.447)
    if Q.sj2_dr > 0.2232169 and Q.C2_b2 < 0.04008677:
        z += 154.7121 * (Q.sj2_dr - 0.2232169) * (0.04008677 - Q.C2_b2)
    if Q.girth2_top5 < 0.008329695 and Q.z_6 < 0.04770182:
        z += 926.6213 * (0.008329695 - Q.girth2_top5) * (0.04770182 - Q.z_6)
    if Q.psi_0p3 > 0.9896594 and Q.sj2_mass2 < 14.45919:
        z += -2.178301 * (Q.psi_0p3 - 0.9896594) * (14.45919 - Q.sj2_mass2)
    if Q.girth2_top5 < 0.008329695 and Q.sj3_mass1 > 5.112677:
        z += -0.427472 * (0.008329695 - Q.girth2_top5) * (Q.sj3_mass1 - 5.112677)
    if Q.n_dr_0p1_0p2 < 13.0 and Q.absphi_0 < 0.02970886:
        z += -0.8834177 * (13.0 - Q.n_dr_0p1_0p2) * (0.02970886 - Q.absphi_0)
    if Q.girth2_top5 < 0.008329695 and Q.n_real_top40 > 22.0:
        z += -0.9624585 * (0.008329695 - Q.girth2_top5) * (Q.n_real_top40 - 22.0)
    if Q.girth2_top20 < 0.00287991 and Q.max_dr < 0.4357228:
        z += 352.9802 * (0.00287991 - Q.girth2_top20) * (0.4357228 - Q.max_dr)
    if Q.girth2_top5 < 0.008329695 and Q.sj3_pairmin_over_m > 0.09540583:
        z += -67.6136 * (0.008329695 - Q.girth2_top5) * (Q.sj3_pairmin_over_m - 0.09540583)
    if Q.girth2_top5 < 0.008329695 and Q.dr0_8 > 0.2441824:
        z += 213.2233 * (0.008329695 - Q.girth2_top5) * (Q.dr0_8 - 0.2441824)
    if Q.n_dr_0p1_0p2 < 13.0 and Q.n_dr_0p05_0p1 > 8.0:
        z += -0.002358434 * (13.0 - Q.n_dr_0p1_0p2) * (Q.n_dr_0p05_0p1 - 8.0)
    if Q.lam2 < 0.001776308 and Q.dr_12 > 0.2053949:
        z += 627.7655 * (0.001776308 - Q.lam2) * (Q.dr_12 - 0.2053949)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_dr_0p4_up > 0.0:
        z += 0.07217871 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p4_up - 0.0)
    if Q.sd_mass < 79.18312 and Q.eta_2 > -0.03302612:
        z += -0.006448408 * (79.18312 - Q.sd_mass) * (Q.eta_2 - -0.03302612)
    if Q.sd_mass < 45.595 and Q.eta_1 < -0.009371567:
        z += 0.2341414 * (45.595 - Q.sd_mass) * (-0.009371567 - Q.eta_1)
    if Q.sj2_dr > 0.2232169 and Q.soft5_z < 0.0009102912:
        z += -964.4382 * (Q.sj2_dr - 0.2232169) * (0.0009102912 - Q.soft5_z)
    if Q.psi_0p3 > 0.9896594 and Q.pt_2 < 137.5:
        z += 0.1735669 * (Q.psi_0p3 - 0.9896594) * (137.5 - Q.pt_2)
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
