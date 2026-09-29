"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; all observables, tuned for agreement), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  6:  11.8%   (on for 68% of jets)
  neuron 13:  10.9%   (on for 86% of jets)
  neuron  7:  10.8%   (on for 67% of jets)
  neuron  5:   9.1%   (on for 66% of jets)
  neuron 10:   8.7%   (on for 87% of jets)
  neuron  3:   8.1%   (on for 31% of jets)
  neuron 11:   7.8%   (on for 76% of jets)
  neuron  9:   6.9%   (on for 59% of jets)
  neuron  2:   5.3%   (on for 59% of jets)
  neuron  0:   5.1%   (on for 78% of jets)
  neuron 14:   4.3%   (on for 37% of jets)
  neuron  1:   4.0%   (on for 73% of jets)
  neuron  4:   3.9%   (on for 88% of jets)
  neuron 15:   1.9%   (on for 42% of jets)
  neuron  8:   1.1%   (on for 33% of jets)
  neuron 12:   0.3%   (on for 8% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.4% (the network: 65.8%); same class as the network for 87.6% of jets.

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
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmax_over_m     largest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_2          mass of particles 0 and 2 [GeV]
  Q.pair_mass_0_4          mass of particles 0 and 4 [GeV]
  Q.pair_mass_0_5          mass of particles 0 and 5 [GeV]
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.pair_mass_0_7          mass of particles 0 and 7 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.mratio_max_012         largest pair mass / mass of the 3 hardest (dimensionless)
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.min_pair_mass          smallest pair mass among particles 0, 1, 2 [GeV]
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.m01                    mass of particles 0 and 1 [GeV]
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.ptdr0_7                pT7 · ΔR(0, 7) [GeV]
  Q.z_1                    pT of particle 1 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
  Q.pt1_over_pt0           pT1 / pT0
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.z_3rd                  3rd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.sd_nremoved            number of branches removed by soft drop
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_1               |Δη| of particle 1
  Q.abseta_2               |Δη| of particle 2
  Q.abseta_3               |Δη| of particle 3
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_5               |Δη| of particle 5
  Q.abseta_6               |Δη| of particle 6
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_2               |Δφ| of particle 2
  Q.absphi_3               |Δφ| of particle 3
  Q.absphi_5               |Δφ| of particle 5
  Q.absphi_6               |Δφ| of particle 6
  Q.absphi_7               |Δφ| of particle 7
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_3                  ΔR between particle 3 and the hardest particle
  Q.dr0_4                  ΔR between particle 4 and the hardest particle
  Q.dr0_5                  ΔR between particle 5 and the hardest particle
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_3                  ΔR between particle 3 and the 2nd-hardest particle
  Q.dr1_4                  ΔR between particle 4 and the 2nd-hardest particle
  Q.dr1_5                  ΔR between particle 5 and the 2nd-hardest particle
  Q.dr1_6                  ΔR between particle 6 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr02                   ΔR between particles 0 and 2
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_2                  Δη of particle 2
  Q.eta_3                  Δη of particle 3
  Q.eta_4                  Δη of particle 4
  Q.eta_5                  Δη of particle 5
  Q.eta_6                  Δη of particle 6
  Q.eta_7                  Δη of particle 7
  Q.phi_0                  Δφ of particle 0
  Q.phi_1                  Δφ of particle 1
  Q.phi_2                  Δφ of particle 2
  Q.phi_3                  Δφ of particle 3
  Q.phi_4                  Δφ of particle 4
  Q.phi_5                  Δφ of particle 5
  Q.phi_6                  Δφ of particle 6
  Q.phi_7                  Δφ of particle 7
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
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
  Q.centroid_offset        distance of the pT centroid from the jet axis
  Q.pt_dispersion          √(Σ pTᵢ²) / Σ pTᵢ
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
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_pairmax_over_m=max(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_2=pair_mass(0, 2),
        pair_mass_0_4=pair_mass(0, 4),
        pair_mass_0_5=pair_mass(0, 5),
        pair_mass_0_6=pair_mass(0, 6),
        pair_mass_0_7=pair_mass(0, 7),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        mratio_max_012=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        min_pair_mass=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        m01=pair_mass(0, 1),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        ptdr0_7=pt[7] * math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        z_1=z[1],
        z_3=z[3],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        z_3rd=zs[2],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        sd_nremoved=softdrop("removed"),
        abseta_0=abs(eta[0]),
        abseta_1=abs(eta[1]),
        abseta_2=abs(eta[2]),
        abseta_3=abs(eta[3]),
        abseta_4=abs(eta[4]),
        abseta_5=abs(eta[5]),
        abseta_6=abs(eta[6]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_2=abs(phi[2]),
        absphi_3=abs(phi[3]),
        absphi_5=abs(phi[5]),
        absphi_6=abs(phi[6]),
        absphi_7=abs(phi[7]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        dr0_3=math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        dr0_4=math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        dr0_5=math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_3=math.sqrt(dist2(1, 3)) if pt[3] > 0 else 0.0,
        dr1_4=math.sqrt(dist2(1, 4)) if pt[4] > 0 else 0.0,
        dr1_5=math.sqrt(dist2(1, 5)) if pt[5] > 0 else 0.0,
        dr1_6=math.sqrt(dist2(1, 6)) if pt[6] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr02=math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        eta_2=eta[2],
        eta_3=eta[3],
        eta_4=eta[4],
        eta_5=eta[5],
        eta_6=eta[6],
        eta_7=eta[7],
        phi_0=phi[0],
        phi_1=phi[1],
        phi_2=phi[2],
        phi_3=phi[3],
        phi_4=phi[4],
        phi_5=phi[5],
        phi_6=phi[6],
        phi_7=phi[7],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
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
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 24.36;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.35681 * (-0.07525098
        + 0.09438327 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +9.4%  girth2 < 0.01324
        + 0.08728629 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +8.7%  width < 0.008678
        + 0.06152747 * max(0.0, Q.sj3_dr_max - 0.1070199) / 0.08518534   # +6.2%  sj3_dr_max > 0.107
        + 0.05326974 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +5.3%  sj3_dr_max < 0.3012
        - 0.05095879 * max(0.0, 0.004372139 - Q.width) / 0.00156571   # -5.1%  width < 0.004372
        + 0.0438787 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # +4.4%  mass_over_sum_pt < 0.1309
        - 0.04100743 * max(0.0, 0.08475161 - Q.mass_over_sum_pt) / 0.03304118   # -4.1%  mass_over_sum_pt < 0.08475
        - 0.03871046 * max(0.0, 0.07608178 - Q.girth) / 0.02796784   # -3.9%  girth < 0.07608
        - 0.03639069 * max(0.0, 56.92035 - Q.mass) / 21.78475   # -3.6%  mass < 56.92
        - 0.03357122 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -3.4%  lam1 < 0.005433
        - 0.03053575 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -3.1%  girth < 0.08724
        - 0.02445232 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # -2.4%  sj3_dr_max > 0.179
        + 0.0201544 * max(0.0, 0.003904593 - Q.mass_over_sum_pt_sq) / 0.001485439   # +2.0%  mass_over_sum_pt_sq < 0.003905
        + 0.01943219 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +1.9%  log_sum_pt > 6.378
        + 0.01889607 * max(0.0, 0.03556091 - Q.e2) / 0.01371972   # +1.9%  e2 < 0.03556
        - 0.0188429 * max(0.0, 0.007929074 - Q.girth2_top3) / 0.004560262   # -1.9%  girth2_top3 < 0.007929
        - 0.01877722 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.9%  sj3_dr_max > 0.2337
        - 0.01610338 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -1.6%  log_sum_pt > 6.67
        + 0.01387686 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +1.4%  sum_pt_top5 > 687.4
        + 0.0126798 * max(0.0, 0.006756161 - Q.girth2_top3) / 0.003650633   # +1.3%  girth2_top3 < 0.006756
        + 0.01252392 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +1.3%  planar_flow < 0.1484
        + 0.01182662 * max(0.0, 0.0245477 - Q.e2) / 0.007455821   # +1.2%  e2 < 0.02455
        + 0.01150335 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 0.07232166 - Q.dr_4) / 0.001949106   # +1.2%  log_sum_pt > 6.67 and dr_4 < 0.07232
        + 0.01136093 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.001732318   # +1.1%  planar_flow < 0.1484 and centroid_offset < 0.0499
        + 0.01135777 * max(0.0, Q.n_dr_0_0p05 - 4.0) / 1.61275   # +1.1%  n_dr_0_0p05 > 4
        - 0.009635959 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, Q.z_7 - 0.01685855) / 0.0006290825   # -1.0%  log_sum_pt > 6.67 and z_7 > 0.01686
        + 0.009526529 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 1.002471 - Q.D2) / 0.001008123   # +1.0%  girth2 < 0.01324 and D2 < 1.002
        - 0.00927577 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # -0.9%  girth2_top5 < 0.00833
        - 0.008916941 * max(0.0, 0.0002758826 - Q.lam1) / 3.586469e-05   # -0.9%  lam1 < 0.0002759
        + 0.008296524 * max(0.0, 0.004372139 - Q.width) * max(0.0, 0.03532852 - Q.C3) / 3.6978e-05   # +0.8%  width < 0.004372 and C3 < 0.03533
        + 0.007189165 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.phi_0 - -0.008995056) / 0.0001281354   # +0.7%  girth2 < 0.01324 and phi_0 > -0.008995
        - 0.007006595 * max(0.0, 0.001563465 - Q.C2_b2) / 0.0008800039   # -0.7%  C2_b2 < 0.001563
        + 0.006968201 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 28.78966 - Q.m01) / 1.085363   # +0.7%  log_sum_pt > 6.67 and m01 < 28.79
        - 0.006730255 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.centroid_offset - 0.01837778) / 2.056482e-05   # -0.7%  girth2 < 0.01324 and centroid_offset > 0.01838
        - 0.006236173 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 0.06473447 - Q.z_7) / 3.579716e-05   # -0.6%  centroid_offset > 0.02077 and z_7 < 0.06473
        + 0.00595371 * max(0.0, Q.centroid_offset - 0.006789738) * max(0.0, 223.375 - Q.pt_1) / 1.309723   # +0.6%  centroid_offset > 0.00679 and pt_1 < 223.4
        + 0.005940795 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.08525808 - Q.dr_2) / 1.41829e-06   # +0.6%  lam2 < 7.301e-05 and dr_2 < 0.08526
        + 0.005591447 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05557716 - Q.z_7) / 0.001734349   # +0.6%  sj3_dr_max < 0.3012 and z_7 < 0.05558
        + 0.005560877 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2) / 2.36013e-06   # +0.6%  lam2 < 7.301e-05 and D2_b2 < 0.267
        + 0.005401437 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.4160096 - Q.pt_dispersion) / 0.0006933006   # +0.5%  planar_flow < 0.1484 and pt_dispersion < 0.416
        + 0.004926705 * max(0.0, 21.78408 - Q.mass) / 4.431023   # +0.5%  mass < 21.78
        - 0.004596138 * max(0.0, 7.300726e-05 - Q.lam2) / 2.767163e-05   # -0.5%  lam2 < 7.301e-05
        - 0.004586383 * max(0.0, Q.sum_pt - 901.5938) / 12.40362   # -0.5%  sum_pt > 901.6
        - 0.004460054 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -0.4%  centroid_offset > 0.00679
        + 0.004343338 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.02320757 - Q.z_7) / 2.048782e-05   # +0.4%  planar_flow < 0.1484 and z_7 < 0.02321
        - 0.004196803 * max(0.0, 0.01882765 - Q.girth2) / 0.01314296   # -0.4%  girth2 < 0.01883
        - 0.004043695 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, Q.dr0_6 - 0.1755206) / 0.0008435917   # -0.4%  planar_flow < 0.1484 and dr0_6 > 0.1755
        - 0.003843923 * max(0.0, 1.0 - Q.n_dr_0_0p05) / 0.2859429   # -0.4%  n_dr_0_0p05 < 1
        + 0.003754491 * max(0.0, 29.6447 - Q.mass) / 7.34723   # +0.4%  mass < 29.64
        - 0.003671843 * max(0.0, 0.004372139 - Q.width) * max(0.0, Q.eccentricity - 0.9598562) / 6.567391e-06   # -0.4%  width < 0.004372 and eccentricity > 0.9599
        + 0.003381168 * max(0.0, 0.08723651 - Q.girth) * max(0.0, Q.dr1_3 - 0.1637906) / 6.974897e-05   # +0.3%  girth < 0.08724 and dr1_3 > 0.1638
        + 0.003285849 * max(0.0, 0.01587232 - Q.zdr_1) / 0.007691685   # +0.3%  zdr_1 < 0.01587
        - 0.00315131 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # -0.3%  lam1 < 0.006507 and D2 < 0.8757
        + 0.002690284 * max(0.0, 0.01882765 - Q.girth2) * max(0.0, -0.0966217 - Q.phi_2) / 7.731109e-06   # +0.3%  girth2 < 0.01883 and phi_2 < -0.09662
        - 0.002544767 * max(0.0, 64.61873 - Q.mass) / 27.57279   # -0.3%  mass < 64.62
        - 0.002337248 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 0.05056028 - Q.dr_2) / 1.335632e-05   # -0.2%  centroid_offset > 0.02077 and dr_2 < 0.05056
        - 0.002084062 * max(0.0, 6.080494 - Q.log_sum_pt) / 0.006097657   # -0.2%  log_sum_pt < 6.08
        - 0.001935532 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.eccentricity - 0.9458207) / 0.1898472   # -0.2%  sum_pt > 901.6 and eccentricity > 0.9458
        + 0.00193543 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 2.0 - Q.n_dr_0p05_0p1) / 0.1809599   # +0.2%  sj3_dr_max < 0.3012 and n_dr_0p05_0p1 < 2
        - 0.001890256 * max(0.0, 0.01882765 - Q.girth2) * max(0.0, Q.mass_top2 - 28.78966) / 0.006400313   # -0.2%  girth2 < 0.01883 and mass_top2 > 28.79
        - 0.001852497 * max(0.0, 0.007929074 - Q.girth2_top3) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0005550791   # -0.2%  girth2_top3 < 0.007929 and D2_b2 < 0.5327
        - 0.001689024 * max(0.0, Q.centroid_offset - 0.02076709) / 0.004751537   # -0.2%  centroid_offset > 0.02077
        - 0.001629813 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 358.375 - Q.sum_pt_top2) / 2.426548   # -0.2%  planar_flow < 0.1484 and sum_pt_top2 < 358.4
        - 0.001468059 * max(0.0, 64.61873 - Q.mass) * max(0.0, 0.1830092 - Q.D2_b2) / 0.3382769   # -0.1%  mass < 64.62 and D2_b2 < 0.183
        + 0.00136175 * max(0.0, 0.04889979 - Q.sj3_dr_max) / 0.006377687   # +0.1%  sj3_dr_max < 0.0489
        - 0.001315396 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, 0.03363037 - Q.absphi_3) / 0.2707601   # -0.1%  sum_pt > 901.6 and absphi_3 < 0.03363
        + 0.001296651 * max(0.0, 0.004372139 - Q.width) * max(0.0, Q.pair_mass_0_6 - 20.59519) / 8.507382e-05   # +0.1%  width < 0.004372 and pair_mass_0_6 > 20.6
        - 0.00127853 * max(0.0, 0.004372139 - Q.width) * max(0.0, 0.7459513 - Q.D2) / 7.193909e-06   # -0.1%  width < 0.004372 and D2 < 0.746
        + 0.001220986 * max(0.0, 0.01882765 - Q.girth2) * max(0.0, 46.125 - Q.pt_6) / 0.1159788   # +0.1%  girth2 < 0.01883 and pt_6 < 46.12
        + 0.00120249 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, Q.phi_4 - 0.1096802) / 7.123371e-05   # +0.1%  log_sum_pt > 6.67 and phi_4 > 0.1097
        - 0.001171114 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05744392 - Q.D2_b2) / 0.0005632394   # -0.1%  sj3_dr_max < 0.3012 and D2_b2 < 0.05744
        - 0.001148938 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.02270492 - Q.dr_2) / 2.517754e-05   # -0.1%  planar_flow < 0.1484 and dr_2 < 0.0227
        + 0.001127104 * max(0.0, Q.centroid_offset - 0.006789738) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.02705531   # +0.1%  centroid_offset > 0.00679 and n_pt_above_50 < 7
        + 0.001090698 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.dr0_6 - 0.2347949) / 0.03872508   # +0.1%  sum_pt > 901.6 and dr0_6 > 0.2348
        - 0.001022167 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, 0.119512 - Q.dr12) / 1.24022   # -0.1%  sum_pt > 901.6 and dr12 < 0.1195
        + 0.0009446433 * max(0.0, Q.n_dr_0_0p05 - 4.0) * max(0.0, Q.dr_4 - 0.1999753) / 0.0009517533   # +0.1%  n_dr_0_0p05 > 4 and dr_4 > 0.2
        - 0.0008371405 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.1%  centroid_offset > 0.0499
        - 0.0008005415 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -0.1%  lam1 < 0.006507
        + 0.0007437078 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, -0.07373047 - Q.eta_5) / 0.1578737   # +0.1%  sum_pt_top5 > 687.4 and eta_5 < -0.07373
        - 0.0006951227 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.sj3_dr12 - 0.2182873) / 7.0923e-06   # -0.1%  girth2 < 0.01324 and sj3_dr12 > 0.2183
        - 0.0006787981 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.pt_7 - 33.21875) / 68.87156   # -0.1%  sum_pt > 901.6 and pt_7 > 33.22
        - 0.0006626424 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.1830092 - Q.D2_b2) / 0.003463294   # -0.1%  mass < 29.64 and D2_b2 < 0.183
        - 0.000584769 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.eta_2 - 0.02253723) / 1.528066e-05   # -0.1%  girth2_top5 < 0.00833 and eta_2 > 0.02254
        - 0.0005710132 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.01452637 - Q.phi_0) / 0.1160543   # -0.1%  mass < 29.64 and phi_0 < 0.01453
        + 0.0005581135 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, Q.eta_6 - 0.05477905) / 4.236986e-06   # +0.1%  lam1 < 0.005433 and eta_6 > 0.05478
        + 0.0004682086 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, Q.dr_4 - 0.03006824) / 0.002784857   # +0.0%  planar_flow < 0.1484 and dr_4 > 0.03007
        - 0.0004465276 * max(0.0, 56.92035 - Q.mass) * max(0.0, Q.dr_max_012 - 0.06112084) / 0.1887976   # -0.0%  mass < 56.92 and dr_max_012 > 0.06112
        - 0.000439235 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, Q.phi_3 - 0.04800415) / 1.473914e-06   # -0.0%  lam1 < 0.005433 and phi_3 > 0.048
        - 0.0004370482 * max(0.0, Q.n_dr_0_0p05 - 4.0) * max(0.0, Q.pair_mass_0_5 - 16.26599) / 0.6268949   # -0.0%  n_dr_0_0p05 > 4 and pair_mass_0_5 > 16.27
        - 0.0003983068 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.06336451 - Q.dr_4) / 1.344914   # -0.0%  sum_pt_top5 > 687.4 and dr_4 < 0.06336
        + 0.0003790013 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, Q.phi_1 - 0.01467133) / 0.0008710753   # +0.0%  sj3_dr_max < 0.3012 and phi_1 > 0.01467
        + 0.0003789863 * max(0.0, 10.97038 - Q.sd_mass) / 2.685704   # +0.0%  sd_mass < 10.97
        - 0.000261273 * max(0.0, 0.001563465 - Q.C2_b2) * max(0.0, Q.zdr_6 - 0.01392641) / 4.211449e-08   # -0.0%  C2_b2 < 0.001563 and zdr_6 > 0.01393
        - 0.0002226924 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, Q.phi_2 - 0.09649963) / 4.090789e-05   # -0.0%  log_sum_pt > 6.67 and phi_2 > 0.0965
        + 0.00019708 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, -0.003096655 - Q.mean_phi) / 2.294381e-05   # +0.0%  girth2 < 0.01324 and mean_phi < -0.003097
        + 0.0001313418 * max(0.0, 56.92035 - Q.mass) * max(0.0, 0.05744392 - Q.D2_b2) / 0.01156768   # +0.0%  mass < 56.92 and D2_b2 < 0.05744
        + 4.682559e-05 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.mean_eta - 6.288824e-05) / 0.000715688   # +0.0%  log_sum_pt > 6.378 and mean_eta > 6.289e-05
        - 2.955815e-05 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, -0.09552002 - Q.eta_2) / 0.009222008   # -0.0%  sum_pt > 901.6 and eta_2 < -0.09552
        + 1.500557e-05 * max(0.0, 0.008678045 - Q.width) * max(0.0, Q.eta_7 - 0.1219513) / 5.067324e-06   # +0.0%  width < 0.008678 and eta_7 > 0.122
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 21.42;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.41529 * (0.002064366
        - 0.08704382 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -8.7%  girth2 < 0.008678
        - 0.06386258 * max(0.0, 0.00609665 - Q.width) / 0.002544039   # -6.4%  width < 0.006097
        - 0.06262819 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -6.3%  sum_pt_top5 > 531.2
        + 0.05343572 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) / 0.002578466   # +5.3%  mass_over_sum_pt_sq < 0.005833
        + 0.04979723 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # +5.0%  girth < 0.0717
        + 0.04580967 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +4.6%  lam1 < 0.005954
        + 0.04400249 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # +4.4%  e2_sq < 0.008169
        + 0.04327819 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +4.3%  log_sum_pt > 6.503
        + 0.03477485 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # +3.5%  z_7 > 0.02321
        + 0.03458655 * max(0.0, 49.6681 - Q.mass) / 17.06893   # +3.5%  mass < 49.67
        + 0.03409162 * max(0.0, Q.sum_pt_top5 - 531.1875) * max(0.0, 0.01392641 - Q.zdr_6) / 1.245695   # +3.4%  sum_pt_top5 > 531.2 and zdr_6 < 0.01393
        - 0.0340051 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, 0.008654951 - Q.zdr_6) / 0.000799435   # -3.4%  log_sum_pt > 6.503 and zdr_6 < 0.008655
        - 0.033771 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # -3.4%  girth < 0.1019
        - 0.02892448 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -2.9%  z_7 < 0.06165
        + 0.0275948 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +2.8%  e3 < 0.0005117
        - 0.02512951 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1) / 0.1196218   # -2.5%  log_sum_pt > 6.378 and z_dr_0p05_0p1 < 0.7509
        - 0.02071856 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 1.627072e-05 - Q.e3) / 1.912338e-06   # -2.1%  log_sum_pt > 6.378 and e3 < 1.627e-05
        + 0.02049301 * max(0.0, Q.log_sum_pt - 6.605974) / 0.07073019   # +2.0%  log_sum_pt > 6.606
        + 0.02039492 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +2.0%  log_sum_pt > 6.378
        - 0.01904078 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -1.9%  centroid_offset < 0.01838
        + 0.01321712 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.centroid_offset - 0.009480685) / 0.0009010889   # +1.3%  log_sum_pt > 6.378 and centroid_offset > 0.009481
        + 0.01245659 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +1.2%  pt_7 > 34.53
        + 0.01167589 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +1.2%  lam1 < 0.008376
        + 0.01044375 * max(0.0, 1.627072e-05 - Q.e3) / 6.577181e-06   # +1.0%  e3 < 1.627e-05
        - 0.009465215 * max(0.0, Q.z_7 - 0.04624032) / 0.01204993   # -0.9%  z_7 > 0.04624
        + 0.008444594 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 1.432482 - Q.D2) / 0.01970563   # +0.8%  log_sum_pt > 6.606 and D2 < 1.432
        + 0.007397456 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +0.7%  sj3_dr_max > 0.169
        + 0.007035761 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 0.09925997 - Q.z_4) / 0.00251516   # +0.7%  log_sum_pt > 6.606 and z_4 < 0.09926
        - 0.007032965 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.04875823) / 0.002321449   # -0.7%  z_7 < 0.06165 and z_dr_0p05_0p1 > 0.04876
        - 0.006673121 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # -0.7%  tau1 < 0.07284
        - 0.006548523 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 52.90625 - Q.pt_6) / 13.55761   # -0.7%  pt_7 > 34.53 and pt_6 < 52.91
        - 0.006543943 * max(0.0, 0.0008722282 - Q.lam1) / 0.0001882079   # -0.7%  lam1 < 0.0008722
        - 0.006383749 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 1.679198 - Q.D2) / 0.005760154   # -0.6%  z_7 < 0.06165 and D2 < 1.679
        - 0.006126601 * max(0.0, 1.0 - Q.n_dr_0_0p05) / 0.2859429   # -0.6%  n_dr_0_0p05 < 1
        + 0.005999303 * max(0.0, Q.sum_pt_top5 - 367.5938) / 230.8403   # +0.6%  sum_pt_top5 > 367.6
        + 0.005954228 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.sj2_dr - 0.1294903) / 0.1877842   # +0.6%  pt_7 > 34.53 and sj2_dr > 0.1295
        + 0.005747029 * max(0.0, 0.008678045 - Q.girth2) * max(0.0, Q.centroid_offset - 0.02076709) / 7.418362e-06   # +0.6%  girth2 < 0.008678 and centroid_offset > 0.02077
        - 0.005698024 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # -0.6%  girth2 < 0.00502
        + 0.005375149 * max(0.0, Q.z_7 - 0.04624032) * max(0.0, 0.002412891 - Q.girth2_top2) / 7.577894e-06   # +0.5%  z_7 > 0.04624 and girth2_top2 < 0.002413
        + 0.004103536 * max(0.0, 0.006726135 - Q.tau21_b2) / 0.0004063558   # +0.4%  tau21_b2 < 0.006726
        - 0.003896421 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.5700307   # -0.4%  sj3_dr_max > 0.169 and sj3_pair_mass_min > 5.744
        + 0.003636137 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, Q.zdr_4 - 0.002832174) / 8.802138e-05   # +0.4%  log_sum_pt > 6.606 and zdr_4 > 0.002832
        + 0.003249055 * max(0.0, 1.0 - Q.n_dr_0_0p05) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.1952756   # +0.3%  n_dr_0_0p05 < 1 and n_pt_above_50 > 5
        + 0.002969185 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # +0.3%  zdr_0 < 0.02118
        - 0.002955191 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1484197 - Q.planar_flow) / 0.0001369594   # -0.3%  lam1 < 0.008376 and planar_flow < 0.1484
        + 0.002865872 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.1340569 - Q.sj3_pairmin_over_m) / 0.00601396   # +0.3%  log_sum_pt > 6.378 and sj3_pairmin_over_m < 0.1341
        - 0.002761424 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.z_dr_0p05_0p1 - 0.2919447) / 0.7430303   # -0.3%  pt_7 > 34.53 and z_dr_0p05_0p1 > 0.2919
        - 0.002625136 * max(0.0, Q.sj3_dr_min - 0.03628191) / 0.02376249   # -0.3%  sj3_dr_min > 0.03628
        - 0.002604894 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236) / 1.769648e-05   # -0.3%  mass_over_sum_pt_sq < 0.005833 and centroid_offset > 0.008092
        - 0.002544262 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, 0.03885671 - Q.D2_b2) / 4.310709e-07   # -0.3%  mass_over_sum_pt_sq < 0.005833 and D2_b2 < 0.03886
        - 0.002502397 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 77.625 - Q.pt_3) / 25.26458   # -0.3%  pt_7 > 34.53 and pt_3 < 77.62
        + 0.002285341 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.01403324 - Q.girth2_top2) / 0.0001680113   # +0.2%  z_7 < 0.06165 and girth2_top2 < 0.01403
        - 0.002220595 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 2.955458e-05 - Q.e3) / 7.137805e-05   # -0.2%  pt_7 > 34.53 and e3 < 2.955e-05
        - 0.00215236 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.dr1_7 - 0.1343804) / 0.0003460886   # -0.2%  z_7 < 0.06165 and dr1_7 > 0.1344
        - 0.002124404 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.pair_mass_0_4 - 13.64049) / 0.0375734   # -0.2%  z_7 < 0.06165 and pair_mass_0_4 > 13.64
        - 0.002046841 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.00323438   # -0.2%  lam1 < 0.008376 and n_pt_above_50 > 5
        - 0.001867971 * max(0.0, 0.0005116989 - Q.e3) * max(0.0, -0.04013062 - Q.phi_0) / 1.595938e-06   # -0.2%  e3 < 0.0005117 and phi_0 < -0.04013
        + 0.001599373 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.008921136 - Q.mean_phi2) / 0.0001024498   # +0.2%  z_7 < 0.06165 and mean_phi2 < 0.008921
        + 0.00157366 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # +0.2%  pt_7 > 34.53 and tau2 < 0.01713
        + 0.001490665 * max(0.0, 49.6681 - Q.mass) * max(0.0, Q.mass_top2 - 0.6957161) / 24.04133   # +0.1%  mass < 49.67 and mass_top2 > 0.6957
        + 0.001442516 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # +0.1%  lam1 < 0.008376 and centroid_offset > 0.02077
        + 0.001314831 * max(0.0, Q.sum_pt_top5 - 531.1875) * max(0.0, -0.04013062 - Q.phi_0) / 0.1583461   # +0.1%  sum_pt_top5 > 531.2 and phi_0 < -0.04013
        + 0.001062044 * max(0.0, Q.sum_pt_top5 - 531.1875) * max(0.0, Q.ptdr0_6 - 5.648151) / 42.02402   # +0.1%  sum_pt_top5 > 531.2 and ptdr0_6 > 5.648
        - 0.001056466 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.centroid_offset - 0.02076709) / 2.871254e-05   # -0.1%  z_7 < 0.06165 and centroid_offset > 0.02077
        + 0.0009470148 * max(0.0, 25.57812 - Q.pt_7) * max(0.0, Q.zdr_6 - 0.00177424) / 0.001000592   # +0.1%  pt_7 < 25.58 and zdr_6 > 0.001774
        - 0.0009279094 * max(0.0, 1.0 - Q.n_dr_0_0p05) * max(0.0, Q.zdr_7 - 0.0007928864) / 0.001861106   # -0.1%  n_dr_0_0p05 < 1 and zdr_7 > 0.0007929
        - 0.0008347336 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.ptdr0_6 - 4.218041) / 0.000461403   # -0.1%  lam1 < 0.005954 and ptdr0_6 > 4.218
        + 0.0008228442 * max(0.0, 25.57812 - Q.pt_7) / 1.327389   # +0.1%  pt_7 < 25.58
        - 0.0008153785 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 3.916009 - Q.D3) / 0.03613284   # -0.1%  z_7 < 0.06165 and D3 < 3.916
        + 0.000799961 * max(0.0, Q.sj3_dr_min - 0.03628191) * max(0.0, 0.09635947 - Q.sj3_z3) / 0.0003928608   # +0.1%  sj3_dr_min > 0.03628 and sj3_z3 < 0.09636
        - 0.0007625232 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.1%  pt_7 > 53.44
        - 0.0006001773 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, Q.zdr_6 - 0.0002697433) / 0.001181033   # -0.1%  pt_7 > 53.44 and zdr_6 > 0.0002697
        - 0.0005580345 * max(0.0, 49.6681 - Q.mass) * max(0.0, Q.mean_phi - 0.002834884) / 0.05152091   # -0.1%  mass < 49.67 and mean_phi > 0.002835
        + 0.0005433486 * max(0.0, Q.pt_6 - 62.25) / 0.3835962   # +0.1%  pt_6 > 62.25
        - 0.0004293508 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, 1.138966 - Q.sj2_mass2) / 0.02325489   # -0.0%  sj3_dr_max > 0.169 and sj2_mass2 < 1.139
        + 0.0003192942 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.06297984 - Q.tau2) / 0.0007776805   # +0.0%  z_7 < 0.06165 and tau2 < 0.06298
        - 0.0002712851 * max(0.0, 0.07283629 - Q.tau1) * max(0.0, Q.zdr_6 - 0.005547132) / 5.423046e-06   # -0.0%  tau1 < 0.07284 and zdr_6 > 0.005547
        - 0.0002502111 * max(0.0, 0.1019409 - Q.girth) * max(0.0, Q.pair_mass_0_6 - 4.037975) / 0.1185817   # -0.0%  girth < 0.1019 and pair_mass_0_6 > 4.038
        + 0.0002381745 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 3.175597   # +0.0%  pt_7 > 34.53 and n_dr_0p1_0p2 > 1
        - 0.0001406194 * max(0.0, 0.0717028 - Q.girth) * max(0.0, Q.zdr_6 - 0.01392641) / 2.850351e-07   # -0.0%  girth < 0.0717 and zdr_6 > 0.01393
        + 0.0001070323 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.1057739 - Q.abseta_0) / 0.001206885   # +0.0%  z_7 < 0.06165 and abseta_0 < 0.1058
        - 3.542988e-05 * max(0.0, 0.006726135 - Q.tau21_b2) * max(0.0, Q.mean_eta - 0.009367547) / 1.05134e-06   # -0.0%  tau21_b2 < 0.006726 and mean_eta > 0.009368
        - 2.356416e-05 * max(0.0, Q.z_7 - 0.08670959) / 0.000332184   # -0.0%  z_7 > 0.08671
        + 1.527092e-05 * max(0.0, 0.006726135 - Q.tau21_b2) * max(0.0, Q.sj3_pair_mass_min - 6.811308) / 2.612788e-07   # +0.0%  tau21_b2 < 0.006726 and sj3_pair_mass_min > 6.811
        + 3.127097e-06 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, Q.dr01 - 0.1894571) / 0.0009794156   # +0.0%  pt_7 > 53.44 and dr01 > 0.1895
        + 2.038507e-06 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, Q.phi_0 - 0.07977295) / 1.342031e-05   # +0.0%  log_sum_pt > 6.606 and phi_0 > 0.07977
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 20.43;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.42559 * (0.09556741
        - 0.1115056 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -11.2%  pt_7 < 53.44
        - 0.06891266 * max(0.0, Q.LHA - 0.1329373) / 0.1175814   # -6.9%  LHA > 0.1329
        + 0.05917513 * max(0.0, 813.4156 - Q.sum_pt) / 129.0758   # +5.9%  sum_pt < 813.4
        - 0.03980142 * max(0.0, Q.sum_pt - 527.1781) * max(0.0, 0.0001947983 - Q.lam2) / 0.02863163   # -4.0%  sum_pt > 527.2 and lam2 < 0.0001948
        + 0.03899387 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # +3.9%  log_sum_pt < 6.606
        + 0.0325689 * max(0.0, 0.1079857 - Q.mass_over_sum_pt) / 0.05207765   # +3.3%  mass_over_sum_pt < 0.108
        + 0.03251175 * max(0.0, 62.55 - Q.sj3_pair_mass_max) / 30.55764   # +3.3%  sj3_pair_mass_max < 62.55
        + 0.0319811 * max(0.0, 0.2507612 - Q.max_dr) / 0.1316542   # +3.2%  max_dr < 0.2508
        - 0.02954915 * max(0.0, 0.4926918 - Q.planar_flow) / 0.2728654   # -3.0%  planar_flow < 0.4927
        + 0.02891503 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +2.9%  lam1 < 0.005954
        - 0.02684292 * max(0.0, 62.55 - Q.sj3_pair_mass_max) * max(0.0, 0.06810151 - Q.z_7) / 0.6094489   # -2.7%  sj3_pair_mass_max < 62.55 and z_7 < 0.0681
        - 0.02584544 * max(0.0, 36.22941 - Q.mass) / 10.11393   # -2.6%  mass < 36.23
        - 0.02256521 * max(0.0, Q.z_7 - 0.03629544) / 0.01879829   # -2.3%  z_7 > 0.0363
        + 0.02138222 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # +2.1%  zdr_0 < 0.02118
        + 0.02052464 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # +2.1%  girth2 < 0.003563
        - 0.01986394 * max(0.0, 53.4375 - Q.pt_7) * max(0.0, 1.129616 - Q.D2_b2) / 9.433694   # -2.0%  pt_7 < 53.44 and D2_b2 < 1.13
        - 0.01770354 * max(0.0, Q.sum_pt - 527.1781) * max(0.0, 0.01922607 - Q.absphi_2) / 1.427803   # -1.8%  sum_pt > 527.2 and absphi_2 < 0.01923
        + 0.0171614 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.pt_6 - 19.46875) / 0.05232739   # +1.7%  lam1 < 0.005954 and pt_6 > 19.47
        + 0.01578156 * max(0.0, 36.22941 - Q.mass) * max(0.0, 73.75 - Q.pt_5) / 259.9887   # +1.6%  mass < 36.23 and pt_5 < 73.75
        + 0.01538061 * max(0.0, Q.N2 - 0.1150852) / 0.1151384   # +1.5%  N2 > 0.1151
        - 0.01488466 * max(0.0, Q.sum_pt - 527.1781) * max(0.0, 0.05648804 - Q.absphi_3) / 6.676969   # -1.5%  sum_pt > 527.2 and absphi_3 < 0.05649
        + 0.01480903 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # +1.5%  log_sum_pt < 6.464
        + 0.01459399 * max(0.0, Q.sum_pt - 868.5094) / 18.08463   # +1.5%  sum_pt > 868.5
        + 0.01355706 * max(0.0, Q.sum_pt - 527.1781) / 199.4719   # +1.4%  sum_pt > 527.2
        + 0.01248828 * max(0.0, 0.06729223 - Q.C2) / 0.04162086   # +1.2%  C2 < 0.06729
        - 0.01215324 * max(0.0, Q.z_7 - 0.04939969) * max(0.0, 0.9206502 - Q.D2_b2) / 0.004461719   # -1.2%  z_7 > 0.0494 and D2_b2 < 0.9207
        + 0.01139888 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +1.1%  lam2 < 0.0001948
        + 0.01018746 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # +1.0%  log_sum_pt > 6.843
        + 0.01002995 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +1.0%  pt_7 > 34.53
        - 0.009198754 * max(0.0, 0.03243272 - Q.z_7) / 0.002061024   # -0.9%  z_7 < 0.03243
        - 0.009125809 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, Q.D3 - 0.2213841) / 0.02598782   # -0.9%  z_7 < 0.07149 and D3 > 0.2214
        + 0.008891902 * max(0.0, 6.46415 - Q.log_sum_pt) * max(0.0, 1.129616 - Q.D2_b2) / 0.03658616   # +0.9%  log_sum_pt < 6.464 and D2_b2 < 1.13
        + 0.008603548 * max(0.0, 3.55957 - Q.m012) / 0.8681902   # +0.9%  m012 < 3.56
        - 0.008426538 * max(0.0, 0.004430536 - Q.centroid_offset) / 0.0004182362   # -0.8%  centroid_offset < 0.004431
        - 0.00834638 * max(0.0, 62.55 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.01096064) / 0.2139385   # -0.8%  sj3_pair_mass_max < 62.55 and centroid_offset > 0.01096
        - 0.007931575 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # -0.8%  girth < 0.007674
        - 0.007550004 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.05718994 - Q.abseta_7) / 0.0005837029   # -0.8%  z_7 < 0.07149 and abseta_7 < 0.05719
        - 0.007473251 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 0.0174408 - Q.abseta_0) / 0.216283   # -0.7%  sum_pt_top5 > 752.1 and abseta_0 < 0.01744
        + 0.007464112 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.0001947983 - Q.lam2) / 1.313144e-06   # +0.7%  log_sum_pt > 6.843 and lam2 < 0.0001948
        - 0.007377561 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.max_dr - 0.08050702) / 4.493327e-05   # -0.7%  lam1 < 0.005954 and max_dr > 0.08051
        + 0.006964266 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.4474937 - Q.z_dr_0p05_0p1) / 0.00315435   # +0.7%  log_sum_pt > 6.843 and z_dr_0p05_0p1 < 0.4475
        + 0.006745053 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.0174408 - Q.abseta_0) / 8.283244e-05   # +0.7%  log_sum_pt > 6.843 and abseta_0 < 0.01744
        + 0.006501155 * max(0.0, 0.01573181 - Q.absphi_2) / 0.003774997   # +0.7%  absphi_2 < 0.01573
        + 0.005883555 * max(0.0, Q.m012 - 40.2) / 1.307209   # +0.6%  m012 > 40.2
        - 0.005822194 * max(0.0, 0.004430536 - Q.centroid_offset) * max(0.0, 0.02490234 - Q.abseta_7) / 4.69344e-06   # -0.6%  centroid_offset < 0.004431 and abseta_7 < 0.0249
        + 0.005237139 * max(0.0, Q.pt_7 - 43.5) * max(0.0, 0.004032342 - Q.C2_b2) / 0.004969117   # +0.5%  pt_7 > 43.5 and C2_b2 < 0.004032
        + 0.004940476 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 0.03623337 - Q.max_dr) / 9.868306e-05   # +0.5%  log_sum_pt < 6.606 and max_dr < 0.03623
        + 0.004878598 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # +0.5%  lam1 < 0.001504
        - 0.004741991 * max(0.0, 0.03243272 - Q.z_7) * max(0.0, 0.9206502 - Q.D2_b2) / 0.0004360983   # -0.5%  z_7 < 0.03243 and D2_b2 < 0.9207
        - 0.004684667 * max(0.0, 53.4375 - Q.pt_7) * max(0.0, Q.abseta_0 - 0.02940369) / 0.2565312   # -0.5%  pt_7 < 53.44 and abseta_0 > 0.0294
        + 0.00465241 * max(0.0, Q.sum_pt_top5 - 839.9547) * max(0.0, 0.04881348 - Q.dr_7) / 0.1547132   # +0.5%  sum_pt_top5 > 840 and dr_7 < 0.04881
        - 0.004635174 * max(0.0, 0.2507612 - Q.max_dr) * max(0.0, Q.phi_0 - 0.002076054) / 0.001439957   # -0.5%  max_dr < 0.2508 and phi_0 > 0.002076
        + 0.00422693 * max(0.0, Q.pt_7 - 43.5) / 1.4368   # +0.4%  pt_7 > 43.5
        + 0.004216557 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 1.129616 - Q.D2_b2) / 6.295733   # +0.4%  sum_pt_top5 > 752.1 and D2_b2 < 1.13
        - 0.00395785 * max(0.0, 0.007673833 - Q.girth) * max(0.0, 3.885568 - Q.D2) / 0.0003532425   # -0.4%  girth < 0.007674 and D2 < 3.886
        - 0.00360541 * max(0.0, Q.sum_pt_top5 - 752.1) / 21.27415   # -0.4%  sum_pt_top5 > 752.1
        + 0.003527798 * max(0.0, Q.z_7 - 0.04939969) / 0.01023002   # +0.4%  z_7 > 0.0494
        + 0.003375898 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 0.01573181 - Q.absphi_2) / 0.0002701397   # +0.3%  log_sum_pt < 6.606 and absphi_2 < 0.01573
        - 0.003226719 * max(0.0, 0.007673833 - Q.girth) * max(0.0, 75.625 - Q.pt_4) / 0.004511241   # -0.3%  girth < 0.007674 and pt_4 < 75.62
        - 0.003036451 * max(0.0, 0.03243272 - Q.z_7) * max(0.0, Q.mass_top3 - 16.89912) / 0.01064848   # -0.3%  z_7 < 0.03243 and mass_top3 > 16.9
        - 0.002775806 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.D3 - 0.4387703) / 4.820328   # -0.3%  pt_7 > 34.53 and D3 > 0.4388
        + 0.002147001 * max(0.0, Q.z_7 - 0.04939969) * max(0.0, 0.0623951 - Q.sj3_dr_min) / 0.0002978985   # +0.2%  z_7 > 0.0494 and sj3_dr_min < 0.0624
        - 0.002138083 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 1.345805 - Q.D2_b2) / 0.002918988   # -0.2%  log_sum_pt > 6.843 and D2_b2 < 1.346
        - 0.00204886 * max(0.0, 3.849518e-05 - Q.girth2_top2) / 2.100982e-06   # -0.2%  girth2_top2 < 3.85e-05
        + 0.002042606 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.pt_6 - 41.21875) / 0.06090836   # +0.2%  log_sum_pt > 6.843 and pt_6 > 41.22
        - 0.002002737 * max(0.0, 0.03243272 - Q.z_7) * max(0.0, 8.0 - Q.n_pt_above_5) / 0.0001713068   # -0.2%  z_7 < 0.03243 and n_pt_above_5 < 8
        + 0.001851748 * max(0.0, 0.009668065 - Q.e2) / 0.001482576   # +0.2%  e2 < 0.009668
        + 0.00153132 * max(0.0, Q.z_7 - 0.03629544) * max(0.0, 0.004032342 - Q.C2_b2) / 4.986777e-05   # +0.2%  z_7 > 0.0363 and C2_b2 < 0.004032
        + 0.00129313 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, Q.eta_0 - 0.009292603) / 0.06320835   # +0.1%  sum_pt_top5 > 752.1 and eta_0 > 0.009293
        - 0.001227144 * max(0.0, Q.z_7 - 0.03629544) * max(0.0, Q.dr02 - 0.175862) / 0.0002861587   # -0.1%  z_7 > 0.0363 and dr02 > 0.1759
        + 0.0008616722 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 0.0531335 - Q.z_5) / 2.719561e-05   # +0.1%  log_sum_pt < 6.606 and z_5 < 0.05313
        - 0.0008134434 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.1%  sum_pt_top5 > 840
        + 0.0006416632 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, -0.007301331 - Q.phi_5) / 8.171123e-05   # +0.1%  log_sum_pt > 6.843 and phi_5 < -0.007301
        - 0.0005924801 * max(0.0, Q.sum_pt_top5 - 839.9547) * max(0.0, 0.001184464 - Q.absphi_2) / 0.0007589667   # -0.1%  sum_pt_top5 > 840 and absphi_2 < 0.001184
        - 0.0004604309 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # -0.0%  z_7 < 0.07149
        + 0.0003754693 * max(0.0, 0.03243272 - Q.z_7) * max(0.0, -0.00469376 - Q.mean_phi) / 1.933422e-06   # +0.0%  z_7 < 0.03243 and mean_phi < -0.004694
        + 0.0003104985 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.dr02 - 0.15647) / 2.387299e-05   # +0.0%  log_sum_pt > 6.843 and dr02 > 0.1565
        - 0.0002554899 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.eta_4 - 0.1080994) / 1.042813e-05   # -0.0%  log_sum_pt > 6.843 and eta_4 > 0.1081
        - 0.0001419772 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, -0.01284493 - Q.mean_eta) / 7.701745e-06   # -0.0%  zdr_0 < 0.02118 and mean_eta < -0.01284
        - 0.0001087747 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, -0.0216713 - Q.phi_0) / 1.182282e-05   # -0.0%  log_sum_pt > 6.843 and phi_0 < -0.02167
        - 5.572935e-05 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.eta_0 - 0.02130127) / 1.178041e-05   # -0.0%  log_sum_pt > 6.843 and eta_0 > 0.0213
        - 5.567785e-06 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.dr02 - 0.2623104) / 3.697963e-06   # -0.0%  log_sum_pt > 6.843 and dr02 > 0.2623
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 23.85;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.84971 * (-0.006764634
        - 0.1388775 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # -13.9%  girth2 > 0.008678
        - 0.1266503 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -12.7%  mass_over_sum_pt_sq < 0.01166
        + 0.06736295 * max(0.0, Q.width - 0.007520088) / 0.002470136   # +6.7%  width > 0.00752
        + 0.05222279 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +5.2%  lam1 > 0.008376
        + 0.04456676 * max(0.0, Q.sj2_dr - 0.1872617) / 0.02732165   # +4.5%  sj2_dr > 0.1873
        + 0.03521655 * max(0.0, Q.width - 0.005590289) / 0.003084256   # +3.5%  width > 0.00559
        - 0.03050171 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # -3.1%  girth2_top5 < 0.007164
        - 0.03038792 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # -3.0%  girth > 0.1019
        + 0.02971346 * max(0.0, Q.girth - 0.04081947) / 0.0270611   # +3.0%  girth > 0.04082
        + 0.02646718 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # +2.6%  girth > 0.08724
        - 0.02441535 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.9641201 - Q.z_dr_0p05_0p1) / 0.01959435   # -2.4%  sj2_dr > 0.1873 and z_dr_0p05_0p1 < 0.9641
        + 0.02378901 * max(0.0, 0.00817466 - Q.mass_over_sum_pt_sq) / 0.004299658   # +2.4%  mass_over_sum_pt_sq < 0.008175
        - 0.02340284 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # -2.3%  tau1 > 0.05357
        + 0.02329359 * max(0.0, Q.max_dr - 0.1027585) / 0.04505724   # +2.3%  max_dr > 0.1028
        + 0.02042501 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # +2.0%  sj3_dr_max > 0.1986
        + 0.02027381 * max(0.0, Q.mass_over_sum_pt - 0.0681391) / 0.01539157   # +2.0%  mass_over_sum_pt > 0.06814
        - 0.01541578 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.02673292   # -1.5%  max_dr > 0.1028 and z_dr_0p05_0p1 < 0.846
        - 0.01485501 * max(0.0, 0.9008535 - Q.z_dr_0_0p05) / 0.3811445   # -1.5%  z_dr_0_0p05 < 0.9009
        - 0.01392863 * max(0.0, Q.width - 0.01323868) / 0.001442684   # -1.4%  width > 0.01324
        + 0.01237425 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 0.07861328 - Q.abseta_0) / 0.0002969212   # +1.2%  centroid_offset > 0.01096 and abseta_0 < 0.07861
        + 0.01079377 * max(0.0, 0.004372139 - Q.girth2) / 0.00156571   # +1.1%  girth2 < 0.004372
        + 0.01066275 * max(0.0, 1.0 - Q.n_dr_0_0p05) / 0.2859429   # +1.1%  n_dr_0_0p05 < 1
        + 0.009392805 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +0.9%  tau1 > 0.1136
        + 0.008810003 * max(0.0, Q.girth - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494) / 0.008781747   # +0.9%  girth > 0.04082 and log_sum_pt > 6.08
        + 0.008094243 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50) / 0.02994202   # +0.8%  centroid_offset > 0.01096 and n_pt_above_50 < 8
        + 0.008032488 * max(0.0, -0.004664942 - Q.mean_eta) / 0.003754527   # +0.8%  mean_eta < -0.004665
        + 0.007244179 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 0.1626038 - Q.abseta_7) / 0.0008475577   # +0.7%  centroid_offset > 0.01096 and abseta_7 < 0.1626
        - 0.00706163 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr) / 0.0001196243   # -0.7%  mass_over_sum_pt > 0.06814 and sj2_dr < 0.218
        + 0.007029347 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +0.7%  girth2_top5 < 0.00833
        + 0.007027689 * max(0.0, Q.sj2_dr - 0.2687922) * max(0.0, 0.9641201 - Q.z_dr_0p05_0p1) / 0.005973659   # +0.7%  sj2_dr > 0.2688 and z_dr_0p05_0p1 < 0.9641
        + 0.006793507 * max(0.0, Q.LHA - 0.3467135) / 0.008898884   # +0.7%  LHA > 0.3467
        + 0.006397094 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.02100911   # +0.6%  sj2_dr > 0.1873 and n_dr_0p2_0p4 < 2
        + 0.006307591 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, Q.eta_2 - -0.04544525) / 0.002690131   # +0.6%  max_dr > 0.1028 and eta_2 > -0.04545
        - 0.006153554 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -0.6%  lam1 > 0.012
        + 0.005973215 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, Q.phi_7 - -0.05731659) / 0.0006418095   # +0.6%  centroid_offset > 0.01096 and phi_7 > -0.05732
        - 0.005791682 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # -0.6%  tau1 > 0.1028
        - 0.005705183 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, 50.3522 - Q.mass_top3) / 0.5259724   # -0.6%  tau1 > 0.05357 and mass_top3 < 50.35
        + 0.00511624 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, Q.pair_mass_0_5 - 16.26599) / 0.01853759   # +0.5%  centroid_offset > 0.01096 and pair_mass_0_5 > 16.27
        + 0.005062045 * max(0.0, Q.sd_mass - 62.55) / 3.348271   # +0.5%  sd_mass > 62.55
        - 0.004951089 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.05268713 - Q.dr_3) / 0.000155932   # -0.5%  sj2_dr > 0.1873 and dr_3 < 0.05269
        + 0.004474896 * max(0.0, Q.mean_eta - 0.01772426) / 0.001375178   # +0.4%  mean_eta > 0.01772
        - 0.004380095 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -0.4%  e2 > 0.06344
        + 0.003663921 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.4%  lam2 > 0.001131
        - 0.00357817 * max(0.0, Q.sj2_dr - 0.1492731) / 0.04449993   # -0.4%  sj2_dr > 0.1493
        + 0.003452494 * max(0.0, Q.sd_mass - 62.55) * max(0.0, 0.9206502 - Q.D2_b2) / 1.823781   # +0.3%  sd_mass > 62.55 and D2_b2 < 0.9207
        - 0.003373829 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113) / 0.39507   # -0.3%  sj2_dr > 0.1873 and sj2_mass1 > 2.25
        - 0.003362889 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # -0.3%  lam1 > 0.01643
        + 0.003278114 * max(0.0, Q.centroid_offset - 0.01096064) / 0.008813008   # +0.3%  centroid_offset > 0.01096
        + 0.002855383 * max(0.0, Q.girth - 0.08723651) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 5.945324e-05   # +0.3%  girth > 0.08724 and z_dr_0p05_0p1 > 0.6748
        - 0.002345729 * max(0.0, Q.e2 - 0.06344108) * max(0.0, 0.1332952 - Q.dr0_3) / 6.054313e-05   # -0.2%  e2 > 0.06344 and dr0_3 < 0.1333
        + 0.002211698 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.06081235 - Q.z_6) / 0.0004655277   # +0.2%  max_dr > 0.1028 and z_6 < 0.06081
        + 0.001959611 * max(0.0, Q.mass - 64.61873) * max(0.0, Q.eta_7 - -0.0803833) / 0.3229902   # +0.2%  mass > 64.62 and eta_7 > -0.08038
        - 0.001916571 * max(0.0, Q.sd_mass - 62.55) * max(0.0, 1.474928 - Q.pair_mass_0_4) / 0.6028639   # -0.2%  sd_mass > 62.55 and pair_mass_0_4 < 1.475
        - 0.001896175 * max(0.0, Q.sj3_dr_max - 0.1986272) * max(0.0, 0.03359985 - Q.absphi_6) / 0.0001684548   # -0.2%  sj3_dr_max > 0.1986 and absphi_6 < 0.0336
        - 0.001871843 * max(0.0, Q.width - 0.007520088) * max(0.0, Q.sj3_pairmin_over_m - 0.07708997) / 0.0004577253   # -0.2%  width > 0.00752 and sj3_pairmin_over_m > 0.07709
        + 0.001871539 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, Q.abseta_2 - 0.03250122) / 0.0002973544   # +0.2%  centroid_offset > 0.01096 and abseta_2 > 0.0325
        - 0.001793109 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.1950293   # -0.2%  mass_over_sum_pt > 0.06814 and sj3_pair_mass_min > 5.744
        + 0.00177067 * max(0.0, Q.sj3_dr_max - 0.1986272) * max(0.0, 0.5144873 - Q.sj3_mass2) / 0.004336836   # +0.2%  sj3_dr_max > 0.1986 and sj3_mass2 < 0.5145
        + 0.001710674 * max(0.0, Q.sj3_dr_max - 0.3012016) / 0.01214652   # +0.2%  sj3_dr_max > 0.3012
        - 0.001694617 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # -0.2%  sj3_pair_mass_min > 11.05
        - 0.001595962 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, Q.eta_2 - 0.09655762) / 0.0002867998   # -0.2%  max_dr > 0.1028 and eta_2 > 0.09656
        + 0.001553268 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.pt_6 - 31.90625) / 0.1358885   # +0.2%  mass_over_sum_pt > 0.06814 and pt_6 > 31.91
        - 0.001512634 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # -0.2%  sj2_dr > 0.2688
        - 0.001457595 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, Q.abseta_0 - 0.1057739) / 0.0002451453   # -0.1%  max_dr > 0.1028 and abseta_0 > 0.1058
        - 0.001439033 * max(0.0, 0.03320012 - Q.planar_flow) * max(0.0, Q.ptdr0_3 - 12.34207) / 0.006377968   # -0.1%  planar_flow < 0.0332 and ptdr0_3 > 12.34
        + 0.001420646 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, Q.pt_5 - 59.125) / 0.0351992   # +0.1%  tau1 > 0.05357 and pt_5 > 59.12
        - 0.00138679 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, -0.0892334 - Q.eta_1) / 0.0002499278   # -0.1%  max_dr > 0.1028 and eta_1 < -0.08923
        - 0.001257841 * max(0.0, Q.mean_eta2 - 0.01423545) / 0.0002893422   # -0.1%  mean_eta2 > 0.01424
        - 0.001242758 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, Q.pair_mass_0_5 - 16.26599) / 0.00413009   # -0.1%  girth2 > 0.01883 and pair_mass_0_5 > 16.27
        - 0.001202291 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 27.57812 - Q.pt_6) / 0.028355   # -0.1%  sj2_dr > 0.1873 and pt_6 < 27.58
        + 0.001180015 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, Q.z_6 - 0.05441452) / 6.527736e-06   # +0.1%  lam2 > 0.001131 and z_6 > 0.05441
        + 0.001157651 * max(0.0, Q.sd_mass - 86.4) / 0.7553564   # +0.1%  sd_mass > 86.4
        + 0.00107243 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +0.1%  e2 > 0.04447
        + 0.001024193 * max(0.0, -0.004664942 - Q.mean_eta) * max(0.0, Q.phi_5 - 0.07385864) / 2.965175e-05   # +0.1%  mean_eta < -0.004665 and phi_5 > 0.07386
        - 0.0009759527 * max(0.0, 0.03320012 - Q.planar_flow) / 0.004307781   # -0.1%  planar_flow < 0.0332
        + 0.0009631587 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, Q.phi_5 - 0.07385864) / 0.0006819949   # +0.1%  max_dr > 0.1028 and phi_5 > 0.07386
        - 0.0009419535 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.z_top5 - 0.7773372) / 2.699283e-05   # -0.1%  e2 > 0.06344 and z_top5 > 0.7773
        - 0.0008643033 * max(0.0, Q.girth2 - 0.01882765) / 0.0007605369   # -0.1%  girth2 > 0.01883
        + 0.0008426444 * max(0.0, Q.mass - 64.61873) / 3.274818   # +0.1%  mass > 64.62
        + 0.0007375332 * max(0.0, Q.mean_eta - 0.01772426) * max(0.0, 0.02612796 - Q.mean_phi) / 3.988786e-05   # +0.1%  mean_eta > 0.01772 and mean_phi < 0.02613
        + 0.0007373751 * max(0.0, 1.0 - Q.n_dr_0_0p05) * max(0.0, 4.0 - Q.n_dr_0p05_0p1) / 0.3078555   # +0.1%  n_dr_0_0p05 < 1 and n_dr_0p05_0p1 < 4
        + 0.00071882 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6) / 0.005634141   # +0.1%  lam2 > 0.001131 and pt_6 < 56.53
        - 0.0006907581 * max(0.0, Q.sj2_mass1 - 40.2) / 0.3062354   # -0.1%  sj2_mass1 > 40.2
        - 0.0005031477 * max(0.0, Q.mass - 64.61873) * max(0.0, Q.phi_6 - 0.1192017) / 0.03430397   # -0.1%  mass > 64.62 and phi_6 > 0.1192
        + 0.0004585763 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sj2_mass1 - 16.86126) / 0.02534836   # +0.0%  e2 > 0.06344 and sj2_mass1 > 16.86
        - 0.0004330345 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, Q.mratio_min_012 - 0.3373366) / 0.0003208192   # -0.0%  tau1 > 0.05357 and mratio_min_012 > 0.3373
        + 0.0004026467 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, Q.phi_4 - 0.1096802) / 0.0003590336   # +0.0%  max_dr > 0.1028 and phi_4 > 0.1097
        - 0.0003770225 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 2.662284e-07   # -0.0%  girth2 > 0.01883 and z_dr_0p05_0p1 > 0.7509
        + 0.0003581747 * max(0.0, -0.004664942 - Q.mean_eta) * max(0.0, Q.z_6 - 0.06727211) / 3.056899e-05   # +0.0%  mean_eta < -0.004665 and z_6 > 0.06727
        + 0.0003164761 * max(0.0, Q.lam1 - 0.01643375) * max(0.0, 6.0 - Q.n_for_90pct) / 5.140257e-06   # +0.0%  lam1 > 0.01643 and n_for_90pct < 6
        + 0.0003070251 * max(0.0, Q.girth - 0.07608178) * max(0.0, Q.eta_0 - 0.07952881) / 0.0001300296   # +0.0%  girth > 0.07608 and eta_0 > 0.07953
        - 0.0002710799 * max(0.0, Q.lam1 - 0.008375572) * max(0.0, 24.42188 - Q.pt_6) / 0.0001598819   # -0.0%  lam1 > 0.008376 and pt_6 < 24.42
        + 0.000215363 * max(0.0, Q.e2 - 0.06344108) * max(0.0, 0.01257746 - Q.zdr_3) / 1.576515e-06   # +0.0%  e2 > 0.06344 and zdr_3 < 0.01258
        - 0.0001418153 * max(0.0, Q.sd_mass - 62.55) * max(0.0, 0.01849752 - Q.zdr_1) / 0.007868008   # -0.0%  sd_mass > 62.55 and zdr_1 < 0.0185
        - 0.0001095876 * max(0.0, Q.sd_mass - 38.43971) / 11.50403   # -0.0%  sd_mass > 38.44
        - 7.813594e-05 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, Q.absphi_2 - 0.12854) / 4.568815e-05   # -0.0%  centroid_offset > 0.01096 and absphi_2 > 0.1285
        - 4.978023e-05 * max(0.0, Q.girth - 0.07608178) / 0.01059033   # -0.0%  girth > 0.07608
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 76.84;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 76.84239 * (-0.0113988
        + 0.2859222 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # +28.6%  e2_sq < 0.01166
        - 0.2513648 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -25.1%  mass_over_sum_pt_sq < 0.01166
        - 0.03760133 * max(0.0, 0.233678 - Q.sj3_dr_max) / 0.09057682   # -3.8%  sj3_dr_max < 0.2337
        - 0.02939926 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -2.9%  girth2 < 0.008678
        - 0.02910142 * max(0.0, 0.05028464 - Q.e2) / 0.02502152   # -2.9%  e2 < 0.05028
        + 0.02904942 * max(0.0, 0.06729223 - Q.C2) / 0.04162086   # +2.9%  C2 < 0.06729
        + 0.02835433 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # +2.8%  sj3_dr_max < 0.3456
        + 0.02749332 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +2.7%  N2 < 0.2233
        + 0.01912857 * max(0.0, Q.width - 0.003562611) / 0.004066872   # +1.9%  width > 0.003563
        - 0.01856011 * max(0.0, 0.04990367 - Q.centroid_offset) / 0.03352573   # -1.9%  centroid_offset < 0.0499
        - 0.01648665 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266) / 0.01381411   # -1.6%  N2 < 0.2233 and eccentricity > 0.7117
        + 0.01633347 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +1.6%  tau1 < 0.07284
        + 0.01437568 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +1.4%  sj3_dr_max < 0.2134
        + 0.01281956 * max(0.0, 0.007639643 - Q.girth2_top2) / 0.004543329   # +1.3%  girth2_top2 < 0.00764
        - 0.01179729 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -1.2%  mass_over_sum_pt > 0.09041
        + 0.01027488 * max(0.0, 0.001653836 - Q.width) / 0.0004289067   # +1.0%  width < 0.001654
        + 0.009656583 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +1.0%  lam2 < 0.003408
        + 0.009597383 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +1.0%  e3 < 0.0001869
        - 0.009419231 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -0.9%  lam2 < 0.001131
        - 0.009035069 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -0.9%  lam2 < 0.0005373
        - 0.008796051 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # -0.9%  max_dr < 0.1773
        - 0.008568544 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -0.9%  lam1 < 0.001504
        - 0.007782336 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.1410336 - Q.dr01) / 4.252003e-05   # -0.8%  lam2 < 0.0005373 and dr01 < 0.141
        - 0.007695113 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.1647949 - Q.absphi_7) / 4.92402e-05   # -0.8%  lam2 < 0.0005373 and absphi_7 < 0.1648
        + 0.00730476 * max(0.0, Q.mass_top5 - 14.54404) / 16.7137   # +0.7%  mass_top5 > 14.54
        + 0.00662433 * max(0.0, Q.sd_mass - 44.82259) / 8.759065   # +0.7%  sd_mass > 44.82
        + 0.004974435 * max(0.0, 0.002635418 - Q.girth2) / 0.0007941346   # +0.5%  girth2 < 0.002635
        - 0.004601212 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -0.5%  mass_over_sum_pt < 0.07269
        + 0.004286327 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +0.4%  max_dr < 0.1118
        + 0.003862065 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # +0.4%  girth2_top5 < 0.007164
        + 0.003490877 * max(0.0, 0.3467135 - Q.LHA) / 0.1128474   # +0.3%  LHA < 0.3467
        + 0.003387341 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +0.3%  e2 > 0.04447
        + 0.003372762 * max(0.0, 739.5 - Q.sum_pt) * max(0.0, Q.n_for_50pct - 1.0) / 170.3115   # +0.3%  sum_pt < 739.5 and n_for_50pct > 1
        + 0.003258614 * max(0.0, Q.max_dr - 0.1598486) / 0.0215652   # +0.3%  max_dr > 0.1598
        - 0.003238066 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.centroid_offset - 0.001308549) / 0.1760217   # -0.3%  sd_mass > 44.82 and centroid_offset > 0.001309
        + 0.003002063 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 27.42324 - Q.sj3_pair_mass_min) / 138.7479   # +0.3%  sd_mass > 44.82 and sj3_pair_mass_min < 27.42
        - 0.002862125 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7) / 0.8992052   # -0.3%  N2 < 0.2233 and pt_7 < 53.44
        - 0.002375709 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.2%  mass > 76.66
        - 0.001870061 * max(0.0, Q.LHA - 0.2809341) / 0.02587078   # -0.2%  LHA > 0.2809
        + 0.001656825 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, Q.dr1_5 - 0.1518778) / 0.000997393   # +0.2%  sj3_dr_max < 0.3456 and dr1_5 > 0.1519
        - 0.001653151 * max(0.0, Q.girth2_top5 - 0.001501708) / 0.004796547   # -0.2%  girth2_top5 > 0.001502
        + 0.001604825 * max(0.0, 739.5 - Q.sum_pt) * max(0.0, 0.05749512 - Q.absphi_7) / 1.67493   # +0.2%  sum_pt < 739.5 and absphi_7 < 0.0575
        - 0.001587123 * max(0.0, Q.mass - 45.7571) / 9.387753   # -0.2%  mass > 45.76
        + 0.00153196 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, Q.abseta_1 - 0.01179237) / 4.320063e-05   # +0.2%  girth2_top2 < 0.00764 and abseta_1 > 0.01179
        + 0.001437561 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, Q.pair_mass_0_6 - 15.55293) / 0.07779524   # +0.1%  sj3_dr_max < 0.3456 and pair_mass_0_6 > 15.55
        - 0.001383726 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.pt_2 - 69.875) / 1.400496   # -0.1%  N2 < 0.2233 and pt_2 > 69.88
        + 0.001379245 * max(0.0, 37.76455 - Q.mass_top5) / 16.71307   # +0.1%  mass_top5 < 37.76
        - 0.001361118 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1218872 - Q.abseta_7) / 0.003384012   # -0.1%  N2 < 0.2233 and abseta_7 < 0.1219
        - 0.001297051 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.07796252 - Q.dr1_6) / 1.697653e-05   # -0.1%  lam2 < 0.0005373 and dr1_6 < 0.07796
        + 0.001242076 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, -0.009352575 - Q.mean_phi) / 2.725001e-07   # +0.1%  e3 < 0.0001869 and mean_phi < -0.009353
        + 0.001152745 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 169.875 - Q.pt_1) / 2.311819   # +0.1%  N2 < 0.2233 and pt_1 < 169.9
        + 0.001144843 * max(0.0, Q.mass - 76.6557) * max(0.0, 0.107953 - Q.M3) / 0.09817562   # +0.1%  mass > 76.66 and M3 < 0.108
        - 0.001001716 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.e2_sq - 0.01165737) / 7.11886e-05   # -0.1%  N2 < 0.2233 and e2_sq > 0.01166
        - 0.0008606355 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.55 - Q.mass) / 0.3959152   # -0.1%  N2 < 0.2233 and mass < 62.55
        - 0.0008580555 * max(0.0, 1.340118e-05 - Q.e3) / 5.081834e-06   # -0.1%  e3 < 1.34e-05
        + 0.0008405277 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, -0.01284493 - Q.mean_eta) / 2.067889e-07   # +0.1%  e3 < 0.0001869 and mean_eta < -0.01284
        - 0.0007990931 * max(0.0, 0.03874536 - Q.M2) / 0.006229062   # -0.1%  M2 < 0.03875
        + 0.0007394866 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, Q.dr_5 - 0.1351015) / 0.0003073221   # +0.1%  sj3_dr_max < 0.3456 and dr_5 > 0.1351
        - 0.0006584305 * max(0.0, Q.sd_mass - 74.57663) / 1.60503   # -0.1%  sd_mass > 74.58
        + 0.0006450305 * max(0.0, 0.07283629 - Q.tau1) * max(0.0, Q.phi_1 - 0.01467133) / 5.855453e-05   # +0.1%  tau1 < 0.07284 and phi_1 > 0.01467
        + 0.0006238033 * max(0.0, 0.233678 - Q.sj3_dr_max) * max(0.0, Q.dr0_3 - 0.1815989) / 2.350264e-05   # +0.1%  sj3_dr_max < 0.2337 and dr0_3 > 0.1816
        - 0.000610443 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, -0.01275329 - Q.mean_phi) / 4.253124e-07   # -0.1%  lam2 < 0.0005373 and mean_phi < -0.01275
        + 0.0005280178 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, Q.mean_eta - 0.01271871) / 4.463111e-07   # +0.1%  lam2 < 0.0005373 and mean_eta > 0.01272
        + 0.0004346573 * max(0.0, 0.177305 - Q.max_dr) * max(0.0, Q.ptdr0_4 - 8.939062) / 0.01768503   # +0.0%  max_dr < 0.1773 and ptdr0_4 > 8.939
        - 0.0004148578 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -0.0%  sum_pt < 739.5
        - 0.0003648242 * max(0.0, Q.width - 0.003562611) * max(0.0, 0.1057566 - Q.sj3_z3) / 5.535701e-05   # -0.0%  width > 0.003563 and sj3_z3 < 0.1058
        - 0.0003640829 * max(0.0, 0.2233283 - Q.N2) * max(0.0, -0.04302979 - Q.eta_1) / 0.0004807701   # -0.0%  N2 < 0.2233 and eta_1 < -0.04303
        - 0.0003582408 * max(0.0, Q.max_dr - 0.1598486) * max(0.0, 0.0006435798 - Q.C2_b2) / 1.059262e-06   # -0.0%  max_dr > 0.1598 and C2_b2 < 0.0006436
        + 0.0003539071 * max(0.0, 0.008678045 - Q.girth2) * max(0.0, Q.pair_mass_0_6 - 24.79428) / 0.0003191005   # +0.0%  girth2 < 0.008678 and pair_mass_0_6 > 24.79
        + 0.0003378748 * max(0.0, 0.2233283 - Q.N2) * max(0.0, -0.009352575 - Q.mean_phi) / 0.0001205535   # +0.0%  N2 < 0.2233 and mean_phi < -0.009353
        - 0.0003302605 * max(0.0, 0.01165737 - Q.e2_sq) * max(0.0, Q.ptdr0_7 - 2.691363) / 0.005064196   # -0.0%  e2_sq < 0.01166 and ptdr0_7 > 2.691
        + 0.0003238561 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, -0.1218262 - Q.eta_7) / 9.876428e-05   # +0.0%  sj3_dr_max < 0.3456 and eta_7 < -0.1218
        - 0.0003154071 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, Q.phi_0 - 0.004838562) / 4.167552e-06   # -0.0%  lam2 < 0.0005373 and phi_0 > 0.004839
        - 0.0002715975 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.275762 - Q.sd_zg) / 0.3498594   # -0.0%  sd_mass > 44.82 and sd_zg < 0.2758
        + 0.0002659149 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, Q.absphi_0 - 0.1056549) / 4.162515e-05   # +0.0%  sj3_dr_max < 0.3456 and absphi_0 > 0.1057
        + 0.0002383478 * max(0.0, 0.233678 - Q.sj3_dr_max) * max(0.0, Q.dr02 - 0.2009694) / 3.403825e-06   # +0.0%  sj3_dr_max < 0.2337 and dr02 > 0.201
        - 0.0002120092 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -0.0%  girth > 0.08724
        + 0.0002060883 * max(0.0, 0.233678 - Q.sj3_dr_max) * max(0.0, Q.dr_4 - 0.1289397) / 2.599095e-05   # +0.0%  sj3_dr_max < 0.2337 and dr_4 > 0.1289
        - 0.0002058578 * max(0.0, Q.max_dr - 0.1598486) * max(0.0, 0.1236856 - Q.D2_b2) / 0.0003376539   # -0.0%  max_dr > 0.1598 and D2_b2 < 0.1237
        - 0.0001970885 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # -0.0%  mean_eta > 0.02644
        + 0.0001967734 * max(0.0, 0.233678 - Q.sj3_dr_max) * max(0.0, Q.ptdr0_5 - 7.737156) / 0.01098259   # +0.0%  sj3_dr_max < 0.2337 and ptdr0_5 > 7.737
        - 0.0001646593 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, Q.sj3_pairmax_over_m - 0.7715709) / 4.146965e-05   # -0.0%  mean_eta > 0.02644 and sj3_pairmax_over_m > 0.7716
        + 0.0001556933 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, -0.1135254 - Q.eta_5) / 9.71026e-05   # +0.0%  sj3_dr_max < 0.3456 and eta_5 < -0.1135
        - 0.000106566 * max(0.0, Q.sj3_pair_mass_max - 80.4) / 0.3254107   # -0.0%  sj3_pair_mass_max > 80.4
        + 0.0001035678 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 24.42188 - Q.pt_6) / 0.01371831   # +0.0%  N2 < 0.2233 and pt_6 < 24.42
        + 7.201976e-05 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, Q.dr1_5 - 0.2928518) / 8.694595e-06   # +0.0%  sj3_dr_max < 0.3456 and dr1_5 > 0.2929
        + 5.356518e-05 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.004918231 - Q.zdr_0) / 6.083288e-07   # +0.0%  N2 < 0.2233 and zdr_0 < 0.004918
        + 5.080609e-05 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.0006435798) / 6.028305e-06   # +0.0%  girth2_top2 < 0.00764 and C2_b2 > 0.0006436
        + 4.156385e-05 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, Q.orientation_deg - 45.28772) / 0.001508409   # +0.0%  mean_eta > 0.02644 and orientation_deg > 45.29
        - 4.041275e-05 * max(0.0, Q.mass - 76.6557) * max(0.0, Q.sj2_zsoft - 0.4448372) / 0.006058409   # -0.0%  mass > 76.66 and sj2_zsoft > 0.4448
        + 2.642398e-05 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, Q.pair_mass_0_6 - 24.79428) / 0.006550441   # +0.0%  sj3_dr_max < 0.3456 and pair_mass_0_6 > 24.79
        - 2.385842e-05 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.005440034 - Q.zdr_5) / 0.005279271   # -0.0%  sd_mass > 44.82 and zdr_5 < 0.00544
        + 2.07459e-05 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.phi_6 - 0.1192017) / 0.0001340425   # +0.0%  N2 < 0.2233 and phi_6 > 0.1192
        + 1.152997e-05 * max(0.0, 37.76455 - Q.mass_top5) * max(0.0, Q.abseta_3 - 0.1364807) / 0.00111574   # +0.0%  mass_top5 < 37.76 and abseta_3 > 0.1365
        - 1.001818e-05 * max(0.0, Q.mass_top5 - 14.54404) * max(0.0, Q.pt_6 - 42.78125) / 57.98201   # -0.0%  mass_top5 > 14.54 and pt_6 > 42.78
        - 6.130266e-06 * max(0.0, 0.213399 - Q.sj3_dr_max) * max(0.0, Q.mean_phi2 - 0.008921136) / 8.517119e-08   # -0.0%  sj3_dr_max < 0.2134 and mean_phi2 > 0.008921
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 12.08;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.08221 * (-0.07781071
        + 0.06838652 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset) / 7.0272e-06   # +6.8%  girth2 < 0.001654 and centroid_offset < 0.02355
        - 0.0614061 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -6.1%  zdr_0 < 0.02118
        + 0.04569347 * max(0.0, 0.003562611 - Q.width) / 0.001184249   # +4.6%  width < 0.003563
        + 0.04551372 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +4.6%  z_7 < 0.07149
        + 0.03971058 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, 0.01437952 - Q.centroid_offset) / 1.318015e-05   # +4.0%  e2_sq < 0.005285 and centroid_offset < 0.01438
        + 0.03959771 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +4.0%  LHA < 0.2161
        + 0.03852337 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 46.125 - Q.pt_6) / 2161.354   # +3.9%  sum_pt_top5 > 430.8 and pt_6 < 46.12
        + 0.03483815 * max(0.0, 0.001653836 - Q.girth2) / 0.0004289067   # +3.5%  girth2 < 0.001654
        + 0.03292847 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 28.39396 - Q.pt1_dr01) / 4201.911   # +3.3%  sum_pt_top5 > 430.8 and pt1_dr01 < 28.39
        + 0.03283367 * max(0.0, Q.sum_pt_top5 - 430.75) / 176.7518   # +3.3%  sum_pt_top5 > 430.8
        - 0.03135252 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -3.1%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.0308007 * max(0.0, 0.04939969 - Q.z_7) / 0.007457505   # +3.1%  z_7 < 0.0494
        - 0.02797228 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -2.8%  log_sum_pt > 6.701
        + 0.02632706 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 40.2 - Q.mass_top3) / 0.60086   # +2.6%  z_7 < 0.07149 and mass_top3 < 40.2
        + 0.0242582 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5) / 0.3059459   # +2.4%  z_7 < 0.0494 and mass_top5 < 62.55
        + 0.02306892 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.002127561 - Q.mean_phi2) / 2.516806e-05   # +2.3%  z_7 < 0.07149 and mean_phi2 < 0.002128
        + 0.02300926 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +2.3%  z_7 < 0.02807
        - 0.01899293 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # -1.9%  girth2_top3 < 0.002152
        + 0.01890564 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03532852 - Q.C3) / 0.0003034905   # +1.9%  z_7 < 0.07149 and C3 < 0.03533
        + 0.01858276 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0004149607   # +1.9%  z_7 < 0.07149 and centroid_offset < 0.03117
        + 0.01839774 * max(0.0, 0.006789738 - Q.centroid_offset) / 0.001012351   # +1.8%  centroid_offset < 0.00679
        + 0.01781071 * max(0.0, 0.005284669 - Q.e2_sq) / 0.00223735   # +1.8%  e2_sq < 0.005285
        - 0.01764782 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 501.625 - Q.sum_pt_top2) / 0.2299658   # -1.8%  z_7 < 0.0494 and sum_pt_top2 < 501.6
        - 0.01677821 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.001503553 - Q.lam1) / 4.027805e-05   # -1.7%  LHA < 0.2161 and lam1 < 0.001504
        + 0.01655737 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, 0.3220738 - Q.planar_flow) / 0.0001906084   # +1.7%  e2_sq < 0.005285 and planar_flow < 0.3221
        + 0.01620943 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +1.6%  sum_pt_top5 > 687.4
        + 0.01522325 * max(0.0, 0.02886576 - Q.z_6) / 0.0008381782   # +1.5%  z_6 < 0.02887
        + 0.01510696 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01341502 - Q.dr_2) / 0.0001347035   # +1.5%  log_sum_pt > 6.701 and dr_2 < 0.01342
        + 0.01216545 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # +1.2%  pt_5 < 24.58
        - 0.01180638 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # -1.2%  log_sum_pt > 6.843
        - 0.01126756 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.sj3_dr13 - 0.04995258) / 0.000417922   # -1.1%  zdr_0 < 0.02118 and sj3_dr13 > 0.04995
        + 0.01075391 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # +1.1%  girth < 0.007674
        + 0.0098808 * max(0.0, 0.08051087 - Q.z_6) / 0.02253808   # +1.0%  z_6 < 0.08051
        + 0.00927978 * max(0.0, 0.006789738 - Q.centroid_offset) * max(0.0, 56.4375 - Q.pt_5) / 0.01266727   # +0.9%  centroid_offset < 0.00679 and pt_5 < 56.44
        + 0.007910428 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 9.030369e-05 - Q.mean_eta2) / 9.99897e-07   # +0.8%  log_sum_pt > 6.701 and mean_eta2 < 9.03e-05
        + 0.0078961 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.07235619 - Q.z_3) / 0.0001569963   # +0.8%  LHA < 0.2161 and z_3 < 0.07236
        - 0.007698662 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.mean_phi - 0.002834884) / 4.214377e-05   # -0.8%  log_sum_pt > 6.701 and mean_phi > 0.002835
        - 0.007055358 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.02270492 - Q.dr_2) / 4.344748e-05   # -0.7%  log_sum_pt > 6.896 and dr_2 < 0.0227
        + 0.00704774 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 0.009661512 - Q.dr_2) / 1.13159e-05   # +0.7%  z_7 < 0.0494 and dr_2 < 0.009662
        - 0.007004377 * max(0.0, Q.pair_mass_0_4 - 23.26771) / 0.6641733   # -0.7%  pair_mass_0_4 > 23.27
        - 0.006708112 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.7%  log_sum_pt > 6.896
        - 0.006600355 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0007322447   # -0.7%  e2_sq < 0.005285 and n_pt_above_50 > 6
        + 0.006021906 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.centroid_offset - 0.009480685) / 0.7981241   # +0.6%  sum_pt_top5 > 430.8 and centroid_offset > 0.009481
        - 0.00595002 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, -0.003031633 - Q.mean_eta) / 4.360831e-05   # -0.6%  log_sum_pt > 6.701 and mean_eta < -0.003032
        + 0.005930049 * max(0.0, 45.05938 - Q.pt_3) / 0.7565874   # +0.6%  pt_3 < 45.06
        + 0.004139365 * max(0.0, 0.003562611 - Q.width) * max(0.0, Q.sj3_dr13 - 0.181053) / 2.382944e-06   # +0.4%  width < 0.003563 and sj3_dr13 > 0.1811
        - 0.00390786 * max(0.0, 24.57812 - Q.pt_5) * max(0.0, Q.mean_eta2 - 1.797789e-05) / 0.0002724447   # -0.4%  pt_5 < 24.58 and mean_eta2 > 1.798e-05
        - 0.003761731 * max(0.0, 37.15625 - Q.pt_7) / 5.803748   # -0.4%  pt_7 < 37.16
        - 0.003652063 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.phi_0 - -0.0297699) / 0.0002791968   # -0.4%  zdr_0 < 0.02118 and phi_0 > -0.02977
        - 0.003395101 * max(0.0, 0.02886576 - Q.z_6) * max(0.0, Q.mean_phi2 - 0.001125075) / 4.002571e-07   # -0.3%  z_6 < 0.02887 and mean_phi2 > 0.001125
        + 0.003109358 * max(0.0, Q.pair_mass_0_4 - 23.26771) * max(0.0, Q.eccentricity - 0.9704496) / 0.009376301   # +0.3%  pair_mass_0_4 > 23.27 and eccentricity > 0.9704
        - 0.003061464 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, Q.mean_phi - -0.02594505) / 0.0001045731   # -0.3%  log_sum_pt > 6.896 and mean_phi > -0.02595
        + 0.002999813 * max(0.0, 37.15625 - Q.pt_7) * max(0.0, Q.pt_5 - 24.57812) / 73.42646   # +0.3%  pt_7 < 37.16 and pt_5 > 24.58
        + 0.002588097 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.sj2_dr - 0.1682655) / 4.310225   # +0.3%  sum_pt_top5 > 430.8 and sj2_dr > 0.1683
        - 0.002174241 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, Q.C2_b2 - 0.002354783) / 3.68136e-05   # -0.2%  z_7 < 0.07149 and C2_b2 > 0.002355
        + 0.001914211 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.sj3_dr_max - 0.169029) / 0.0008305926   # +0.2%  log_sum_pt > 6.701 and sj3_dr_max > 0.169
        + 0.001603279 * max(0.0, 24.57812 - Q.pt_5) * max(0.0, 8.0 - Q.n_pt_above_5) / 0.08462184   # +0.2%  pt_5 < 24.58 and n_pt_above_5 < 8
        + 0.001324649 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.05744392 - Q.D2_b2) / 1.220721e-05   # +0.1%  log_sum_pt > 6.896 and D2_b2 < 0.05744
        + 0.001165487 * max(0.0, 0.002074109 - Q.e2_sq) / 0.000670556   # +0.1%  e2_sq < 0.002074
        + 0.001164172 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 0.001534584   # +0.1%  LHA < 0.2161 and n_dr_0p2_0p4 > 0
        + 0.001058499 * max(0.0, Q.pair_mass_0_4 - 23.26771) * max(0.0, 0.01423545 - Q.mean_eta2) / 0.004884555   # +0.1%  pair_mass_0_4 > 23.27 and mean_eta2 < 0.01424
        + 0.0007131547 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, Q.dr0_5 - 0.1122797) / 8.005166e-06   # +0.1%  e2_sq < 0.005285 and dr0_5 > 0.1123
        - 0.0004696695 * max(0.0, Q.pair_mass_0_4 - 23.26771) * max(0.0, Q.phi_4 - 0.1096802) / 0.01602175   # -0.0%  pair_mass_0_4 > 23.27 and phi_4 > 0.1097
        - 0.000387262 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, Q.mean_eta - 0.01271871) / 5.064669e-06   # -0.0%  z_7 < 0.0494 and mean_eta > 0.01272
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 24.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.43927 * (0.009587558
        - 0.1316081 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # -13.2%  width < 0.01324
        - 0.1042686 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -10.4%  girth2 < 0.008678
        + 0.09086441 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +9.1%  lam1 < 0.00733
        + 0.05891023 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +5.9%  lam1 < 0.012
        + 0.04232306 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +4.2%  lam2 < 0.003408
        + 0.03685915 * max(0.0, 0.1136369 - Q.tau1) * max(0.0, 0.01426135 - Q.mean_phi2) / 0.0007578925   # +3.7%  tau1 < 0.1136 and mean_phi2 < 0.01426
        + 0.03416269 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.003289169   # +3.4%  lam1 < 0.00733 and n_dr_0p2_0p4 < 1
        + 0.03127997 * max(0.0, 0.1789613 - Q.sj3_dr_max) / 0.05394779   # +3.1%  sj3_dr_max < 0.179
        - 0.02680014 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 1.341509   # -2.7%  mass < 60.63 and z_dr_0p2_0p4 < 0.05644
        + 0.02453058 * max(0.0, 60.63098 - Q.mass) / 24.47895   # +2.5%  mass < 60.63
        + 0.02402551 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # +2.4%  sj2_dr < 0.1873
        + 0.02241401 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt) / 1.016213   # +2.2%  pt_6 < 41.22 and log_sum_pt < 6.767
        + 0.0217116 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +2.2%  lam1 < 0.005954
        - 0.02165487 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -2.2%  sj3_dr_max < 0.3012
        + 0.02099023 * max(0.0, 0.0030133 - Q.e2_sq) / 0.001065944   # +2.1%  e2_sq < 0.003013
        + 0.0203473 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +2.0%  tau1 < 0.1136
        + 0.01473938 * max(0.0, 6.638339 - Q.log_sum_pt) * max(0.0, 0.0753896 - Q.z_7) / 0.001441873   # +1.5%  log_sum_pt < 6.638 and z_7 < 0.07539
        + 0.01297328 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +1.3%  pt_6 < 41.22
        - 0.01289975 * max(0.0, Q.sj3_dr_min - 0.02400746) / 0.02802924   # -1.3%  sj3_dr_min > 0.02401
        - 0.01282056 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -1.3%  lam2 < 0.001131
        - 0.01198265 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.psi_0p1 - 0.4008925) / 0.003485257   # -1.2%  centroid_offset > 0.008092 and psi_0p1 > 0.4009
        - 0.0102258 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # -1.0%  max_dr < 0.1452
        - 0.01020139 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 0.2534037 - Q.planar_flow) / 0.0006566196   # -1.0%  lam1 < 0.012 and planar_flow < 0.2534
        - 0.009700516 * max(0.0, Q.tau2 - 0.008780509) / 0.00811667   # -1.0%  tau2 > 0.008781
        - 0.009561613 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757) / 0.07098489   # -1.0%  pt_6 < 41.22 and z_7 > 0.02321
        + 0.009184213 * max(0.0, 0.02415398 - Q.C2_b2) / 0.02132459   # +0.9%  C2_b2 < 0.02415
        - 0.008299103 * max(0.0, Q.sj3_pair_mass_min - 4.501727) / 3.480268   # -0.8%  sj3_pair_mass_min > 4.502
        - 0.007518432 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -0.8%  pt_6 < 39.75
        - 0.007359328 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # -0.7%  centroid_offset > 0.008092
        - 0.007022644 * max(0.0, 21.78408 - Q.mass) / 4.431023   # -0.7%  mass < 21.78
        + 0.006558182 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +0.7%  sj3_dr_max > 0.1879
        - 0.006390951 * max(0.0, 0.1452311 - Q.max_dr) * max(0.0, Q.mean_phi - 0.004406178) / 9.704264e-05   # -0.6%  max_dr < 0.1452 and mean_phi > 0.004406
        - 0.0061854 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.00433128 - Q.mean_phi2) / 7.021694e-06   # -0.6%  centroid_offset > 0.01838 and mean_phi2 < 0.004331
        + 0.006050617 * max(0.0, 0.1789613 - Q.sj3_dr_max) * max(0.0, 0.1230713 - Q.tau21_b2) / 0.0006424247   # +0.6%  sj3_dr_max < 0.179 and tau21_b2 < 0.1231
        + 0.005721432 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 43.23531   # +0.6%  sum_pt < 615.9 and n_dr_0p2_0p4 < 2
        + 0.005346991 * max(0.0, 29.90625 - Q.pt_6) / 1.385454   # +0.5%  pt_6 < 29.91
        - 0.005099882 * max(0.0, Q.sj3_dr_max - 0.1879486) * max(0.0, 0.03250122 - Q.abseta_2) / 0.0002335307   # -0.5%  sj3_dr_max > 0.1879 and abseta_2 < 0.0325
        - 0.004665796 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.0002302115 - Q.mean_phi2) / 1.138469e-07   # -0.5%  centroid_offset > 0.008092 and mean_phi2 < 0.0002302
        - 0.00462302 * max(0.0, 45.595 - Q.mass) * max(0.0, Q.eta_3 - -0.00592804) / 0.1560163   # -0.5%  mass < 45.59 and eta_3 > -0.005928
        + 0.004516651 * max(0.0, 615.875 - Q.sum_pt) / 30.07766   # +0.5%  sum_pt < 615.9
        + 0.004376814 * max(0.0, 6.638339 - Q.log_sum_pt) / 0.1524936   # +0.4%  log_sum_pt < 6.638
        - 0.004246997 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.1818564 - Q.z_3rd) / 0.0002723086   # -0.4%  centroid_offset > 0.01838 and z_3rd < 0.1819
        - 0.0040287 * max(0.0, 45.595 - Q.mass) / 14.73532   # -0.4%  mass < 45.59
        + 0.00402192 * max(0.0, 6.267538 - Q.log_sum_pt) / 0.02292097   # +0.4%  log_sum_pt < 6.268
        + 0.003765952 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +0.4%  e3 < 0.0005117
        + 0.003677525 * max(0.0, Q.eccentricity - 0.927072) / 0.03232992   # +0.4%  eccentricity > 0.9271
        - 0.003623831 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.4%  sum_pt_top5 > 840
        + 0.003610304 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 3.0374e-08 - Q.e4) / 5.487247e-07   # +0.4%  sum_pt < 615.9 and e4 < 3.037e-08
        + 0.003520155 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, Q.sj3_dr12 - 0.1692253) / 0.0003039084   # +0.4%  centroid_offset > 0.01838 and sj3_dr12 > 0.1692
        + 0.003502588 * max(0.0, 0.1872617 - Q.sj2_dr) * max(0.0, Q.eccentricity - 0.927072) / 0.0009514438   # +0.4%  sj2_dr < 0.1873 and eccentricity > 0.9271
        + 0.003365414 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.M3 - 0.0782171) / 0.01402652   # +0.3%  pt_6 < 41.22 and M3 > 0.07822
        + 0.002874631 * max(0.0, 19.46875 - Q.pt_6) / 0.2515892   # +0.3%  pt_6 < 19.47
        - 0.002674859 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2) / 2.179362e-05   # -0.3%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        - 0.002559888 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.3%  z_6 < 0.0216
        + 0.002223119 * max(0.0, 0.01323868 - Q.width) * max(0.0, -0.02665591 - Q.mean_eta) / 1.690289e-06   # +0.2%  width < 0.01324 and mean_eta < -0.02666
        + 0.002148294 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.002127561 - Q.mean_phi2) / 0.007417827   # +0.2%  sum_pt > 988.4 and mean_phi2 < 0.002128
        + 0.002087143 * max(0.0, 6.267538 - Q.log_sum_pt) * max(0.0, 6.284176 - Q.sj2_mass2) / 0.102395   # +0.2%  log_sum_pt < 6.268 and sj2_mass2 < 6.284
        - 0.001922562 * max(0.0, 60.63098 - Q.mass) * max(0.0, Q.mean_phi2 - 0.0003573041) / 0.01556073   # -0.2%  mass < 60.63 and mean_phi2 > 0.0003573
        - 0.001894822 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308) / 0.05438016   # -0.2%  centroid_offset > 0.008092 and sj3_pair_mass_min > 6.811
        - 0.001820157 * max(0.0, Q.sj3_dr23 - 0.2207152) / 0.01962555   # -0.2%  sj3_dr23 > 0.2207
        + 0.001614135 * max(0.0, Q.sj3_dr13 - 0.181053) * max(0.0, Q.ptdr0_7 - 10.31097) / 0.01556911   # +0.2%  sj3_dr13 > 0.1811 and ptdr0_7 > 10.31
        + 0.001484513 * max(0.0, 0.01323868 - Q.width) * max(0.0, Q.mean_phi - 0.02612796) / 1.673749e-06   # +0.1%  width < 0.01324 and mean_phi > 0.02613
        + 0.001456873 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, Q.mean_phi - 0.01251955) / 0.1451915   # +0.1%  sum_pt < 615.9 and mean_phi > 0.01252
        - 0.001456708 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, Q.pt_4 - 65.1875) / 0.007630915   # -0.1%  centroid_offset > 0.01838 and pt_4 > 65.19
        - 0.001384617 * max(0.0, 60.63098 - Q.mass) * max(0.0, -0.008995056 - Q.phi_0) / 0.1146652   # -0.1%  mass < 60.63 and phi_0 < -0.008995
        + 0.001374973 * max(0.0, 0.01323868 - Q.width) * max(0.0, -0.02594505 - Q.mean_phi) / 1.701156e-06   # +0.1%  width < 0.01324 and mean_phi < -0.02595
        - 0.001184856 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, Q.orientation_deg - -36.09848) / 0.2437506   # -0.1%  centroid_offset > 0.01838 and orientation_deg > -36.1
        + 0.001119748 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # +0.1%  mean_eta > 0.02644
        + 0.001043067 * max(0.0, 0.04176067 - Q.sj2_zsoft) / 0.0006762529   # +0.1%  sj2_zsoft < 0.04176
        + 0.001036629 * max(0.0, Q.sj2_mass1 - 40.2) / 0.3062354   # +0.1%  sj2_mass1 > 40.2
        + 0.0009691043 * max(0.0, 29.90625 - Q.pt_6) * max(0.0, 8.0 - Q.n_pt_above_10) / 0.5895401   # +0.1%  pt_6 < 29.91 and n_pt_above_10 < 8
        + 0.0008941451 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.mean_eta - 0.02644207) / 1.54475e-06   # +0.1%  lam1 < 0.012 and mean_eta > 0.02644
        + 0.000879562 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 2.69383   # +0.1%  sj3_pair_mass_min > 4.502 and n_dr_0p2_0p4 > 1
        + 0.0008748112 * max(0.0, 19.46875 - Q.pt_6) * max(0.0, 1.813997 - Q.min_pair_mass) / 0.2569012   # +0.1%  pt_6 < 19.47 and min_pair_mass < 1.814
        - 0.0008256548 * max(0.0, Q.sj3_dr13 - 0.181053) * max(0.0, 0.02658081 - Q.abseta_5) / 8.728483e-05   # -0.1%  sj3_dr13 > 0.1811 and abseta_5 < 0.02658
        - 0.0007661852 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.n_pt_above_50 - 6.0) / 2.33176   # -0.1%  sum_pt > 988.4 and n_pt_above_50 > 6
        - 0.0007361119 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.02702951 - Q.C2) / 2.39871e-05   # -0.1%  centroid_offset > 0.01838 and C2 < 0.02703
        - 0.0006539021 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.002776626 - Q.mean_phi2) / 0.01000982   # -0.1%  sum_pt > 988.4 and mean_phi2 < 0.002777
        - 0.0006511734 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.eta_4 - -0.1065704) / 0.6037598   # -0.1%  pt_6 < 41.22 and eta_4 > -0.1066
        + 0.0006248915 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 24.57812 - Q.pt_5) / 1.825775   # +0.1%  sum_pt < 615.9 and pt_5 < 24.58
        + 0.0005981715 * max(0.0, 0.1872617 - Q.sj2_dr) * max(0.0, Q.dr1_6 - 0.07796252) / 0.0003473731   # +0.1%  sj2_dr < 0.1873 and dr1_6 > 0.07796
        + 0.0005829049 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.02980347 - Q.eta_0) / 0.0002676158   # +0.1%  centroid_offset > 0.01838 and eta_0 < 0.0298
        - 0.0005248466 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002658741   # -0.1%  lam2 < 0.003408 and n_dr_0_0p05 < 3
        + 0.000491946 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.0%  sum_pt > 988.4
        - 0.0004167052 * max(0.0, Q.sj2_mass1 - 40.2) * max(0.0, 0.4490772 - Q.tau21_b2) / 0.01473   # -0.0%  sj2_mass1 > 40.2 and tau21_b2 < 0.4491
        - 0.00041246 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # -0.0%  sj3_dr_min > 0.1278
        - 0.0003434243 * max(0.0, Q.centroid_offset - 0.01837778) / 0.005523694   # -0.0%  centroid_offset > 0.01838
        + 0.0002588388 * max(0.0, 0.04176067 - Q.sj2_zsoft) * max(0.0, -0.0216713 - Q.phi_0) / 2.739884e-07   # +0.0%  sj2_zsoft < 0.04176 and phi_0 < -0.02167
        + 0.0002579425 * max(0.0, Q.sj3_dr13 - 0.181053) / 0.02525513   # +0.0%  sj3_dr13 > 0.1811
        + 0.0001878532 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, Q.M3 - 0.06688759) / 9.396674e-06   # +0.0%  mean_eta > 0.02644 and M3 > 0.06689
        + 0.0001527827 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 0.05101919 - Q.z_3) / 0.0001257894   # +0.0%  sum_pt < 615.9 and z_3 < 0.05102
        + 0.0001209144 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, 0.003206041 - Q.tau21_b2) / 4.673745e-08   # +0.0%  mean_eta > 0.02644 and tau21_b2 < 0.003206
        - 0.000112769 * max(0.0, 29.90625 - Q.pt_6) * max(0.0, Q.z_dr_0p2_0p4 - 0.1009734) / 0.007157979   # -0.0%  pt_6 < 29.91 and z_dr_0p2_0p4 > 0.101
        + 8.820878e-05 * max(0.0, 0.01323868 - Q.width) * max(0.0, -0.0803833 - Q.eta_7) / 2.326074e-05   # +0.0%  width < 0.01324 and eta_7 < -0.08038
        + 5.162053e-05 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, 0.03747769 - Q.z_4) / 1.203658e-08   # +0.0%  mean_eta > 0.02644 and z_4 < 0.03748
        + 1.351754e-05 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.mean_phi2 - 0.008921136) / 0.000300354   # +0.0%  sum_pt > 988.4 and mean_phi2 > 0.008921
        - 7.672015e-06 * max(0.0, Q.eccentricity - 0.927072) * max(0.0, 8.0 - Q.n_pt_above_5) / 4.363806e-05   # -0.0%  eccentricity > 0.9271 and n_pt_above_5 < 8
        - 2.065323e-06 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 0.1343804 - Q.dr1_7) / 0.3834372   # -0.0%  pt_6 < 41.22 and dr1_7 < 0.1344
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 45.03;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 45.02604 * (0.2206373
        - 0.06112856 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -6.1%  girth < 0.08724
        + 0.05798639 * max(0.0, Q.width - 0.008678045) / 0.002214537   # +5.8%  width > 0.008678
        - 0.05767805 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -5.8%  mass_over_sum_pt > 0.09041
        - 0.05626612 * max(0.0, Q.mass_over_sum_pt - 0.01109984) / 0.05101851   # -5.6%  mass_over_sum_pt > 0.0111
        - 0.04968623 * max(0.0, 0.006679471 - Q.width) / 0.00293624   # -5.0%  width < 0.006679
        - 0.0474007 * max(0.0, Q.width - 0.005590289) / 0.003084256   # -4.7%  width > 0.00559
        - 0.04423215 * max(0.0, Q.girth2 - 0.004372139) / 0.003638805   # -4.4%  girth2 > 0.004372
        - 0.038519 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -3.9%  girth2 > 0.00752
        + 0.03612204 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +3.6%  sj2_dr < 0.1592
        + 0.03519421 * max(0.0, Q.sj3_dr_max - 0.1426152) / 0.06279029   # +3.5%  sj3_dr_max > 0.1426
        - 0.03458092 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -3.5%  sj2_dr < 0.1873
        + 0.03285153 * max(0.0, Q.mass_over_sum_pt - 0.07269073) / 0.01344138   # +3.3%  mass_over_sum_pt > 0.07269
        + 0.02545839 * max(0.0, Q.girth2 - 0.01323868) / 0.001442684   # +2.5%  girth2 > 0.01324
        + 0.02098903 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +2.1%  mass_over_sum_pt > 0.07992
        + 0.02026271 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # +2.0%  max_dr < 0.1773
        + 0.01949771 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +1.9%  LHA < 0.3033
        + 0.01829158 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +1.8%  mass_over_sum_pt > 0.08475
        - 0.01827803 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.004032342 - Q.C2_b2) / 2.736263e-05   # -1.8%  centroid_offset < 0.02077 and C2_b2 < 0.004032
        + 0.01802857 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.8%  tau1 < 0.05357
        + 0.01799188 * max(0.0, Q.girth2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +1.8%  girth2 > 0.004372 and planar_flow < 0.195
        - 0.01795483 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.width - 0.007520088) / 0.0001455029   # -1.8%  planar_flow < 0.195 and width > 0.00752
        - 0.01744412 * max(0.0, Q.girth2 - 0.001653836) / 0.005220304   # -1.7%  girth2 > 0.001654
        - 0.01516894 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # -1.5%  girth2 < 0.003563
        + 0.01403839 * max(0.0, Q.z_7 - 0.03243272) / 0.02180051   # +1.4%  z_7 > 0.03243
        + 0.01274632 * max(0.0, Q.mass - 36.22941) / 14.20528   # +1.3%  mass > 36.23
        + 0.01193685 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # +1.2%  e2 > 0.03556
        - 0.01114804 * max(0.0, 0.0009641429 - Q.girth2) / 0.0002052288   # -1.1%  girth2 < 0.0009641
        - 0.01079151 * max(0.0, Q.sj3_dr_max - 0.04889979) / 0.1256943   # -1.1%  sj3_dr_max > 0.0489
        + 0.01060897 * max(0.0, 0.001101266 - Q.e2_sq) / 0.000308004   # +1.1%  e2_sq < 0.001101
        - 0.009853404 * max(0.0, Q.sj3_dr_max - 0.213399) / 0.03061279   # -1.0%  sj3_dr_max > 0.2134
        - 0.009102034 * max(0.0, 0.04081947 - Q.girth) / 0.009176315   # -0.9%  girth < 0.04082
        - 0.008725997 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # -0.9%  e2 > 0.05028
        + 0.008222718 * max(0.0, 2.357246 - Q.D2) / 1.097329   # +0.8%  D2 < 2.357
        - 0.006857113 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # -0.7%  max_dr < 0.1118
        - 0.006644135 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -0.7%  centroid_offset < 0.02077
        - 0.006416786 * max(0.0, Q.centroid_offset - 0.03117077) / 0.002505019   # -0.6%  centroid_offset > 0.03117
        + 0.00626808 * max(0.0, 0.08723651 - Q.girth) * max(0.0, 0.004406178 - Q.mean_phi) / 0.0002325096   # +0.6%  girth < 0.08724 and mean_phi < 0.004406
        + 0.005989165 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) / 0.09532154   # +0.6%  z_dr_0p1_0p2 < 0.1586
        - 0.005457793 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -0.5%  sj3_dr_max > 0.2623
        + 0.005153801 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sd_mass - 38.43971) / 1.25962   # +0.5%  planar_flow < 0.195 and sd_mass > 38.44
        + 0.005044864 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.02656143 - Q.tau21_b2) / 4.805781e-05   # +0.5%  centroid_offset < 0.02077 and tau21_b2 < 0.02656
        - 0.004886642 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -0.5%  lam1 < 0.008376
        + 0.004863657 * max(0.0, 0.0003061234 - Q.lam2) * max(0.0, 0.04019753 - Q.tau21_b2) / 2.611351e-06   # +0.5%  lam2 < 0.0003061 and tau21_b2 < 0.0402
        - 0.004375915 * max(0.0, 0.001056655 - Q.girth2_top2) / 0.0003002514   # -0.4%  girth2_top2 < 0.001057
        - 0.004364956 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.4%  mass > 76.66
        - 0.004177643 * max(0.0, 0.04019753 - Q.tau21_b2) / 0.01109641   # -0.4%  tau21_b2 < 0.0402
        - 0.004029628 * max(0.0, Q.sd_rg - 0.2037854) / 0.01566501   # -0.4%  sd_rg > 0.2038
        + 0.003543003 * max(0.0, Q.mass_top5 - 53.60766) / 1.978814   # +0.4%  mass_top5 > 53.61
        + 0.003414119 * max(0.0, 0.08723651 - Q.girth) * max(0.0, 0.004032342 - Q.C2_b2) / 0.0001217265   # +0.3%  girth < 0.08724 and C2_b2 < 0.004032
        + 0.003057536 * max(0.0, 0.1950135 - Q.planar_flow) / 0.07573236   # +0.3%  planar_flow < 0.195
        - 0.003030411 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) * max(0.0, 2.080881 - Q.N3) / 0.06089036   # -0.3%  z_dr_0p1_0p2 < 0.1586 and N3 < 2.081
        + 0.002936009 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # +0.3%  sj2_dr < 0.1295
        - 0.002928425 * max(0.0, 8.10414e-05 - Q.girth2_top2) / 7.505045e-06   # -0.3%  girth2_top2 < 8.104e-05
        - 0.002906801 * max(0.0, Q.mass_over_sum_pt - 0.08475161) * max(0.0, 36.8125 - Q.pt_6) / 0.02275392   # -0.3%  mass_over_sum_pt > 0.08475 and pt_6 < 36.81
        + 0.002871805 * max(0.0, 0.001056655 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.006789738) / 7.886295e-07   # +0.3%  girth2_top2 < 0.001057 and centroid_offset > 0.00679
        - 0.002709766 * max(0.0, Q.girth2 - 0.01323868) * max(0.0, 38.25 - Q.pt_6) / 0.004519922   # -0.3%  girth2 > 0.01324 and pt_6 < 38.25
        - 0.002635207 * max(0.0, 29.04219 - Q.pt_7) / 2.179984   # -0.3%  pt_7 < 29.04
        - 0.002590665 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # -0.3%  lam2 < 0.0003061
        + 0.002153806 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, Q.sum_pt_top3 - 331.25) / 1.710819   # +0.2%  centroid_offset < 0.02077 and sum_pt_top3 > 331.2
        + 0.002101214 * max(0.0, 0.003562611 - Q.girth2) * max(0.0, -0.006779839 - Q.mean_eta) / 1.412644e-06   # +0.2%  girth2 < 0.003563 and mean_eta < -0.00678
        - 0.00197729 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -0.2%  e3 < 8.148e-05
        - 0.001755634 * max(0.0, Q.centroid_offset - 0.03117077) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.00188116   # -0.2%  centroid_offset > 0.03117 and n_pt_above_50 > 4
        - 0.001309132 * max(0.0, 0.05744392 - Q.D2_b2) / 0.005938983   # -0.1%  D2_b2 < 0.05744
        - 0.001196036 * max(0.0, Q.mass - 76.6557) * max(0.0, Q.zdr_6 - 0.006366792) / 0.006641425   # -0.1%  mass > 76.66 and zdr_6 > 0.006367
        + 0.00118513 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 0.01961688 - Q.dr0_5) / 0.0003875566   # +0.1%  planar_flow < 0.195 and dr0_5 < 0.01962
        + 0.001174057 * max(0.0, Q.girth2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9458207) / 2.125219e-05   # +0.1%  girth2 > 0.01324 and eccentricity > 0.9458
        + 0.001172592 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, Q.zdr_5 - 0.007104199) / 4.507664e-06   # +0.1%  centroid_offset < 0.02077 and zdr_5 > 0.007104
        - 0.001140105 * max(0.0, 29.04219 - Q.pt_7) * max(0.0, 0.2838437 - Q.tau21) / 0.1125981   # -0.1%  pt_7 < 29.04 and tau21 < 0.2838
        - 0.001131871 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6) / 0.1690249   # -0.1%  planar_flow < 0.195 and pt_6 < 35.28
        - 0.0009319228 * max(0.0, 0.001056655 - Q.girth2_top2) * max(0.0, 0.02656143 - Q.tau21_b2) / 2.239525e-07   # -0.1%  girth2_top2 < 0.001057 and tau21_b2 < 0.02656
        - 0.0009286647 * max(0.0, Q.mass_over_sum_pt - 0.01109984) * max(0.0, 35.28125 - Q.pt_6) / 0.1065511   # -0.1%  mass_over_sum_pt > 0.0111 and pt_6 < 35.28
        + 0.0007317593 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.pt_4 - 71.6875) / 0.2128009   # +0.1%  planar_flow < 0.195 and pt_4 > 71.69
        - 0.0007115928 * max(0.0, Q.mass_over_sum_pt - 0.1079857) / 0.005364315   # -0.1%  mass_over_sum_pt > 0.108
        - 0.0007071992 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # -0.1%  sd_rg > 0.2788
        - 0.0006609572 * max(0.0, 0.009213932 - Q.tau21_b2) / 0.0008512768   # -0.1%  tau21_b2 < 0.009214
        - 0.0005692937 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.1%  centroid_offset > 0.03776
        + 0.0004706417 * max(0.0, Q.mass_over_sum_pt - 0.09041383) * max(0.0, 36.8125 - Q.pt_6) / 0.01975833   # +0.0%  mass_over_sum_pt > 0.09041 and pt_6 < 36.81
        - 0.0003841177 * max(0.0, 0.177305 - Q.max_dr) * max(0.0, Q.dr0_4 - 0.1878822) / 3.314452e-05   # -0.0%  max_dr < 0.1773 and dr0_4 > 0.1879
        - 0.00032956 * max(0.0, Q.centroid_offset - 0.03776099) * max(0.0, Q.pt_4 - 81.375) / 0.0001936307   # -0.0%  centroid_offset > 0.03776 and pt_4 > 81.38
        - 0.0003122166 * max(0.0, 0.001101266 - Q.e2_sq) * max(0.0, -0.04275513 - Q.phi_1) / 2.373915e-08   # -0.0%  e2_sq < 0.001101 and phi_1 < -0.04276
        - 0.0002788281 * max(0.0, 0.001101266 - Q.e2_sq) * max(0.0, Q.phi_0 - 0.0402832) / 2.634889e-08   # -0.0%  e2_sq < 0.001101 and phi_0 > 0.04028
        - 0.0002321277 * max(0.0, 0.001101266 - Q.e2_sq) * max(0.0, -0.04544525 - Q.eta_2) / 2.425087e-08   # -0.0%  e2_sq < 0.001101 and eta_2 < -0.04545
        + 0.000211832 * max(0.0, Q.width - 0.008678045) * max(0.0, 38.25 - Q.pt_6) / 0.006941575   # +0.0%  width > 0.008678 and pt_6 < 38.25
        - 0.0001804807 * max(0.0, 0.009213932 - Q.tau21_b2) * max(0.0, Q.sj2_mass1 - 3.458544) / 0.0001032175   # -0.0%  tau21_b2 < 0.009214 and sj2_mass1 > 3.459
        + 0.0001665824 * max(0.0, Q.sj3_dr_max - 0.1426152) * max(0.0, 19.46875 - Q.pt_6) / 0.01292439   # +0.0%  sj3_dr_max > 0.1426 and pt_6 < 19.47
        - 0.0001504275 * max(0.0, 0.001101266 - Q.e2_sq) * max(0.0, Q.eta_2 - 0.04476929) / 2.389146e-08   # -0.0%  e2_sq < 0.001101 and eta_2 > 0.04477
        + 0.0001340733 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) * max(0.0, -0.01275329 - Q.mean_phi) / 0.0001186461   # +0.0%  z_dr_0p1_0p2 < 0.1586 and mean_phi < -0.01275
        + 0.0001088265 * max(0.0, 0.001101266 - Q.e2_sq) * max(0.0, -0.06384277 - Q.eta_2) / 6.602357e-09   # +0.0%  e2_sq < 0.001101 and eta_2 < -0.06384
        - 7.939644e-05 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, Q.n_dr_0p05_0p1 - 5.0) / 0.0005328768   # -0.0%  tau1 < 0.05357 and n_dr_0p05_0p1 > 5
        - 3.489096e-05 * max(0.0, 0.05744392 - Q.D2_b2) * max(0.0, 0.00177424 - Q.zdr_6) / 1.04577e-07   # -0.0%  D2_b2 < 0.05744 and zdr_6 < 0.001774
        + 2.029597e-05 * max(0.0, 0.1591713 - Q.sj2_dr) * max(0.0, Q.sj2_mass1 - 40.2) / 3.774486e-05   # +0.0%  sj2_dr < 0.1592 and sj2_mass1 > 40.2
        - 1.639895e-05 * max(0.0, 0.04019753 - Q.tau21_b2) * max(0.0, 0.01452637 - Q.phi_0) / 0.0003289225   # -0.0%  tau21_b2 < 0.0402 and phi_0 < 0.01453
        + 1.112734e-05 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj2_mass1 - 31.78116) / 1.424888e-07   # +0.0%  e3 < 8.148e-05 and sj2_mass1 > 31.78
        + 9.644077e-08 * max(0.0, 0.1294903 - Q.sj2_dr) * max(0.0, Q.sj2_mass1 - 31.78116) / 5.268517e-05   # +0.0%  sj2_dr < 0.1295 and sj2_mass1 > 31.78
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 16.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.23305 * (-0.02068117
        - 0.0900064 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.003562611 - Q.girth2) / 5.329563e-05   # -9.0%  girth < 0.06109 and girth2 < 0.003563
        + 0.0846332 * max(0.0, 0.003562611 - Q.width) / 0.001184249   # +8.5%  width < 0.003563
        + 0.06988263 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.girth2) / 0.0003703114   # +7.0%  sj3_dr_max < 0.1986 and girth2 < 0.006679
        - 0.06782393 * max(0.0, 0.06108601 - Q.girth) / 0.01868685   # -6.8%  girth < 0.06109
        + 0.05534114 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0001901112   # +5.5%  girth2 < 0.00502 and z_dr_0p2_0p4 < 0.101
        + 0.05141283 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.width) / 4.808955e-05   # +5.1%  tau1 < 0.05357 and width < 0.003563
        + 0.04899473 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +4.9%  girth2 < 0.006679 and centroid_offset < 0.02355
        + 0.02923316 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +2.9%  girth2 < 0.00502
        - 0.02885278 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.1231549   # -2.9%  mass < 29.64 and centroid_offset < 0.02686
        - 0.02794136 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.003562611 - Q.width) / 0.0001781324   # -2.8%  sj3_dr_max < 0.1986 and width < 0.003563
        - 0.0269054 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -2.7%  sj3_dr_max < 0.1426
        - 0.0252613 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.0001947983 - Q.lam2) / 0.0007178522   # -2.5%  mass < 21.78 and lam2 < 0.0001948
        + 0.02460338 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.0001272072   # +2.5%  mass_over_sum_pt < 0.03319 and centroid_offset < 0.02686
        + 0.02431003 * max(0.0, 0.002635418 - Q.width) / 0.0007941347   # +2.4%  width < 0.002635
        + 0.02000367 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.0001947983 - Q.lam2) / 9.822054e-06   # +2.0%  sj3_dr_max < 0.1986 and lam2 < 0.0001948
        + 0.01959049 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2) / 2.869243e-06   # +2.0%  girth < 0.06109 and lam2 < 0.0001948
        - 0.0195376 * max(0.0, 0.01289969 - Q.e2) * max(0.0, Q.psi_0p2 - 0.79448) / 0.0005184882   # -2.0%  e2 < 0.0129 and psi_0p2 > 0.7945
        + 0.01685848 * max(0.0, 49.6681 - Q.mass) * max(0.0, 0.0001330621 - Q.lam2) / 0.001509268   # +1.7%  mass < 49.67 and lam2 < 0.0001331
        - 0.0159734 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -1.6%  girth2 < 0.00502 and centroid_offset > 0.00679
        - 0.0155907 * max(0.0, 0.006679471 - Q.girth2) / 0.00293624   # -1.6%  girth2 < 0.006679
        - 0.01481195 * max(0.0, 0.003562611 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738) / 5.559665e-06   # -1.5%  width < 0.003563 and centroid_offset > 0.00679
        - 0.01372218 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # -1.4%  LHA < 0.1967
        - 0.01310882 * max(0.0, 21.78408 - Q.mass) / 4.431023   # -1.3%  mass < 21.78
        - 0.01160599 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.008678045 - Q.width) / 0.0002365852   # -1.2%  log_sum_pt > 6.701 and width < 0.008678
        - 0.01155272 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.005691733 - Q.girth2_top5) / 0.0003116052   # -1.2%  sj3_dr_max < 0.1986 and girth2_top5 < 0.005692
        + 0.01093223 * max(0.0, 0.0001330621 - Q.lam2) * max(0.0, 4.721224 - Q.D2_b2) / 0.0002340602   # +1.1%  lam2 < 0.0001331 and D2_b2 < 4.721
        - 0.0102569 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) / 0.00722828   # -1.0%  mass_over_sum_pt < 0.03319
        + 0.009414271 * max(0.0, 0.01289969 - Q.e2) / 0.002527207   # +0.9%  e2 < 0.0129
        - 0.008245852 * max(0.0, 0.0001330621 - Q.lam2) / 6.711676e-05   # -0.8%  lam2 < 0.0001331
        - 0.008125137 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # -0.8%  pt_7 > 34.53
        + 0.006710602 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.4007947 - Q.planar_flow) / 0.0003918103   # +0.7%  girth2 < 0.006679 and planar_flow < 0.4008
        - 0.006050942 * max(0.0, Q.z_dr_0_0p05 - 0.8477313) / 0.0544886   # -0.6%  z_dr_0_0p05 > 0.8477
        - 0.005985784 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 6.502799 - Q.log_sum_pt) / 8.085113e-05   # -0.6%  girth2 < 0.00502 and log_sum_pt < 6.503
        - 0.005788015 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # -0.6%  tau1 < 0.05357
        - 0.005132617 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 4.721224 - Q.D2_b2) / 0.008595177   # -0.5%  girth2 < 0.006679 and D2_b2 < 4.721
        + 0.004961122 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 48.71875 - Q.pt_7) / 0.776931   # +0.5%  log_sum_pt > 6.701 and pt_7 < 48.72
        - 0.004778418 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # -0.5%  sj3_dr_max < 0.1986
        - 0.004480963 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.003562611 - Q.width) / 0.02261167   # -0.4%  mass < 29.64 and width < 0.003563
        - 0.004422171 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.02355416 - Q.centroid_offset) / 0.0005980206   # -0.4%  log_sum_pt > 6.701 and centroid_offset < 0.02355
        + 0.004114958 * max(0.0, 49.6681 - Q.mass) / 17.06893   # +0.4%  mass < 49.67
        - 0.004092428 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.n_pt_above_50 - 3.0) / 0.006708018   # -0.4%  girth2 < 0.006679 and n_pt_above_50 > 3
        + 0.003751242 * max(0.0, 0.06108601 - Q.girth) * max(0.0, Q.width - 0.0003193707) / 1.048473e-05   # +0.4%  girth < 0.06109 and width > 0.0003194
        - 0.00351739 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 43.5 - Q.pt_7) / 0.02255472   # -0.4%  girth2 < 0.00502 and pt_7 < 43.5
        + 0.003375251 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.phi_0 - -0.04013062) / 0.0001189753   # +0.3%  girth2 < 0.006679 and phi_0 > -0.04013
        + 0.003014534 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0005611231 - Q.girth2) / 8.626944e-06   # +0.3%  log_sum_pt > 6.701 and girth2 < 0.0005611
        - 0.002945066 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 4.721224 - Q.D2_b2) / 0.04403777   # -0.3%  tau1 < 0.05357 and D2_b2 < 4.721
        - 0.00250844 * max(0.0, 0.007078158 - Q.e2) / 0.0007714962   # -0.3%  e2 < 0.007078
        - 0.002375396 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 32.99439 - Q.pair_mass_0_2) / 1030.436   # -0.2%  sum_pt_top5 > 687.4 and pair_mass_0_2 < 32.99
        - 0.002325939 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.zdr_0 - 0.007798268) / 0.0001443534   # -0.2%  log_sum_pt > 6.701 and zdr_0 > 0.007798
        - 0.002321219 * max(0.0, 0.0001330621 - Q.lam2) * max(0.0, 0.1340569 - Q.sj3_pairmin_over_m) / 2.166157e-06   # -0.2%  lam2 < 0.0001331 and sj3_pairmin_over_m < 0.1341
        + 0.002320402 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.mean_phi - -0.0008556753) / 5.44837e-05   # +0.2%  LHA < 0.1967 and mean_phi > -0.0008557
        - 0.002274973 * max(0.0, 0.008355823 - Q.dr_0) / 0.0005818924   # -0.2%  dr_0 < 0.008356
        - 0.002225394 * max(0.0, 0.003343241 - Q.centroid_offset) / 0.0002272052   # -0.2%  centroid_offset < 0.003343
        + 0.002153575 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, Q.phi_7 - -0.08051147) / 0.005314153   # +0.2%  sj3_dr_max < 0.1986 and phi_7 > -0.08051
        - 0.002057165 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.001011075   # -0.2%  girth < 0.06109 and z_dr_0p2_0p4 < 0.05644
        - 0.002056577 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, Q.abseta_4 - 0.02116394) / 0.0003559616   # -0.2%  sj3_dr_max < 0.1986 and abseta_4 > 0.02116
        + 0.002008552 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 3.10694e-07 - Q.e3) / 3.648373e-06   # +0.2%  sum_pt_top5 > 687.4 and e3 < 3.107e-07
        + 0.001727383 * max(0.0, 29.6447 - Q.mass) * max(0.0, 1.762929e-06 - Q.e3) / 9.656157e-06   # +0.2%  mass < 29.64 and e3 < 1.763e-06
        - 0.00164926 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 0.001219216   # -0.2%  tau1 < 0.05357 and n_dr_0p2_0p4 > 0
        - 0.001615756 * max(0.0, 0.001653836 - Q.girth2) / 0.0004289067   # -0.2%  girth2 < 0.001654
        + 0.001602616 * max(0.0, 0.03448406 - Q.z_6) / 0.001525592   # +0.2%  z_6 < 0.03448
        - 0.001415715 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.n_dr_0p05_0p1 - 0.0) / 0.002340464   # -0.1%  girth2 < 0.006679 and n_dr_0p05_0p1 > 0
        - 0.001404773 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.01627885) / 0.0001130126   # -0.1%  sj3_dr_max < 0.1426 and centroid_offset > 0.01628
        - 0.001368783 * max(0.0, 0.007078158 - Q.e2) * max(0.0, Q.girth2_top5 - 5.672047e-05) / 1.487483e-07   # -0.1%  e2 < 0.007078 and girth2_top5 > 5.672e-05
        - 0.001271781 * max(0.0, 0.003239423 - Q.dr_0) / 5.971839e-05   # -0.1%  dr_0 < 0.003239
        + 0.00107132 * max(0.0, Q.D2 - 3.885568) / 0.120662   # +0.1%  D2 > 3.886
        + 0.0009862486 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +0.1%  sj3_dr_max < 0.107
        - 0.000942268 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 2.955458e-05 - Q.e3) / 8.40609e-07   # -0.1%  log_sum_pt > 6.701 and e3 < 2.955e-05
        - 0.0008930398 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.2097233   # -0.1%  mass < 29.64 and D2_b2 < 0.7166
        - 0.000821202 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 0.01544159   # -0.1%  log_sum_pt > 6.701 and n_dr_0p05_0p1 > 2
        - 0.0007495587 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 142.5 - Q.pt_1) / 122.3165   # -0.1%  pt_7 > 34.53 and pt_1 < 142.5
        + 0.0007482439 * max(0.0, 0.06108601 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738) / 8.239263e-05   # +0.1%  girth < 0.06109 and centroid_offset > 0.00679
        - 0.0007449788 * max(0.0, 21.78408 - Q.mass) * max(0.0, 48.71875 - Q.pt_7) / 69.12746   # -0.1%  mass < 21.78 and pt_7 < 48.72
        - 0.0007355576 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.1%  sum_pt > 988.4
        + 0.0007178223 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.mean_eta - 0.003218391) / 1.870377e-05   # +0.1%  LHA < 0.1967 and mean_eta > 0.003218
        + 0.0006733795 * max(0.0, 0.06108601 - Q.girth) * max(0.0, Q.girth2 - 0.005019719) / 2.328219e-07   # +0.1%  girth < 0.06109 and girth2 > 0.00502
        - 0.0006650495 * max(0.0, 6.633799 - Q.mass) / 0.2714171   # -0.1%  mass < 6.634
        + 0.000549491 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # +0.1%  log_sum_pt > 6.701
        + 0.0005444225 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, -0.01586914 - Q.eta_2) / 0.0001976417   # +0.1%  sj3_dr_max < 0.1986 and eta_2 < -0.01587
        - 0.0005238065 * max(0.0, 29.6447 - Q.mass) / 7.34723   # -0.1%  mass < 29.64
        - 0.0005064575 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.0005394348 - Q.mean_phi2) / 0.01120348   # -0.1%  sum_pt_top5 > 687.4 and mean_phi2 < 0.0005394
        + 0.0004983624 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.m012 - 8.921413) / 0.004746988   # +0.0%  girth2 < 0.006679 and m012 > 8.921
        - 0.0004613093 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.n_dr_0p1_0p2 - 2.0) / 8.060906e-05   # -0.0%  girth2 < 0.006679 and n_dr_0p1_0p2 > 2
        - 0.0003753318 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.07002129   # -0.0%  mass < 21.78 and D2_b2 < 0.7166
        - 0.0003407719 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.centroid_offset - 0.01837778) / 3.05705e-06   # -0.0%  LHA < 0.1967 and centroid_offset > 0.01838
        + 0.0002675177 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.0%  sum_pt_top5 > 687.4
        + 0.0002406345 * max(0.0, Q.z_dr_0_0p05 - 0.8477313) * max(0.0, 96.6875 - Q.pt_2) / 0.5412326   # +0.0%  z_dr_0_0p05 > 0.8477 and pt_2 < 96.69
        - 0.0002144072 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.008506537   # -0.0%  LHA < 0.1967 and n_pt_above_50 > 6
        - 0.0002083606 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 8.0 - Q.n_pt_above_10) / 7.731645   # -0.0%  sum_pt_top5 > 687.4 and n_pt_above_10 < 8
        + 0.0001862888 * max(0.0, 0.003343241 - Q.centroid_offset) * max(0.0, 0.5327104 - Q.D2_b2) / 2.280793e-05   # +0.0%  centroid_offset < 0.003343 and D2_b2 < 0.5327
        - 0.0001746806 * max(0.0, 0.06108601 - Q.girth) * max(0.0, Q.ptdr0_2 - 21.99783) / 2.032717e-05   # -0.0%  girth < 0.06109 and ptdr0_2 > 22
        - 0.0001593031 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.pt_7 - 15.55391) / 0.4153091   # -0.0%  LHA < 0.1967 and pt_7 > 15.55
        + 0.0001446721 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.05062541 - Q.dr12) / 0.1693771   # +0.0%  sum_pt > 988.4 and dr12 < 0.05063
        - 0.0001258729 * max(0.0, Q.width - 0.02530566) / 0.0002851417   # -0.0%  width > 0.02531
        - 0.0001251141 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.eta_0 - 0.02130127) / 4.418729e-06   # -0.0%  girth2 < 0.006679 and eta_0 > 0.0213
        - 0.0001171654 * max(0.0, 0.07658656 - Q.LHA) / 0.0004806283   # -0.0%  LHA < 0.07659
        - 6.613651e-05 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, Q.m01 - 22.84498) / 0.01782551   # -0.0%  sj3_dr_max < 0.1986 and m01 > 22.84
        - 5.982045e-05 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.m012 - 28.3451) / 0.00013178   # -0.0%  girth2 < 0.00502 and m012 > 28.35
        + 1.416688e-05 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, -0.02665591 - Q.mean_eta) / 1.176967e-08   # +0.0%  LHA < 0.1967 and mean_eta < -0.02666
        - 9.389215e-06 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, 1.0 - Q.psi_0p3) / 5.234195e-06   # -0.0%  LHA < 0.1967 and psi_0p3 < 1
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 31.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.44061 * (-0.09078649
        + 0.09678301 * max(0.0, 0.00609665 - Q.girth2) / 0.002544039   # +9.7%  girth2 < 0.006097
        + 0.06847434 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.2973626   # +6.8%  mass < 53.33 and centroid_offset < 0.02686
        + 0.06704363 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # +6.7%  girth2 < 0.003563
        + 0.04770041 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +4.8%  sj3_dr_max < 0.2134
        - 0.04732609 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -4.7%  mass_over_sum_pt < 0.07269
        - 0.04364307 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -4.4%  sj3_dr_max < 0.1426
        + 0.04326838 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +4.3%  mass_over_sum_pt_sq < 0.007183
        - 0.03735748 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -3.7%  lam1 < 0.005954
        + 0.03660474 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +3.7%  centroid_offset < 0.01838
        - 0.03487587 * max(0.0, Q.lam1 - 0.003377388) / 0.003672352   # -3.5%  lam1 > 0.003377
        + 0.03465012 * max(0.0, Q.lam1 - 0.005433361) / 0.00267714   # +3.5%  lam1 > 0.005433
        + 0.03015357 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +3.0%  girth2 < 0.00502
        - 0.0296524 * max(0.0, 53.33237 - Q.mass) / 19.36004   # -3.0%  mass < 53.33
        + 0.01816214 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +1.8%  sum_pt < 988.4
        - 0.01758639 * max(0.0, 0.05464922 - Q.girth) / 0.01532978   # -1.8%  girth < 0.05465
        + 0.01543615 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # +1.5%  lam2 < 0.0003061
        + 0.01480603 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # +1.5%  tau1 < 0.0437
        - 0.01439317 * max(0.0, 29.6447 - Q.mass) / 7.34723   # -1.4%  mass < 29.64
        + 0.0128887 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd) / 0.0002178654   # +1.3%  centroid_offset < 0.01838 and z_2nd < 0.2056
        - 0.01285235 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -1.3%  e2 > 0.03556
        + 0.01275958 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.07073975 - Q.abseta_1) / 0.0003304963   # +1.3%  centroid_offset < 0.01838 and abseta_1 < 0.07074
        + 0.01274732 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +1.3%  max_dr < 0.1118
        + 0.01256136 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.0002201438   # +1.3%  girth2 < 0.006097 and planar_flow < 0.3221
        - 0.01082069 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266) / 0.0005312449   # -1.1%  lam1 > 0.005433 and eccentricity > 0.7117
        - 0.01073928 * max(0.0, 0.0030133 - Q.e2_sq) / 0.001065944   # -1.1%  e2_sq < 0.003013
        - 0.009750449 * max(0.0, Q.sd_mass - 45.595) / 8.450255   # -1.0%  sd_mass > 45.59
        - 0.009552613 * max(0.0, 2.371297e-05 - Q.e3) / 1.105328e-05   # -1.0%  e3 < 2.371e-05
        - 0.009539635 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.002834884 - Q.mean_phi) / 1.344572e-05   # -1.0%  girth2 < 0.006097 and mean_phi < 0.002835
        + 0.009239563 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt) / 5.175436   # +0.9%  mass < 53.33 and log_sum_pt < 6.843
        - 0.009159113 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # -0.9%  e3 > 8.148e-05
        - 0.008083384 * max(0.0, Q.girth - 0.0717028) / 0.01201981   # -0.8%  girth > 0.0717
        - 0.008058076 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255) / 0.000277001   # -0.8%  tau1 < 0.0437 and eccentricity > 0.9031
        + 0.007908915 * max(0.0, 0.002170915 - Q.zdr_2) / 0.0002841564   # +0.8%  zdr_2 < 0.002171
        + 0.00713951 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +0.7%  e3 > 0.0001869
        - 0.00685961 * max(0.0, 0.0284695 - Q.C3) / 0.008714909   # -0.7%  C3 < 0.02847
        + 0.006851891 * max(0.0, 0.0001721983 - Q.width) / 1.441896e-05   # +0.7%  width < 0.0001722
        + 0.005639351 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.mean_eta - 6.288824e-05) / 1.181456e-05   # +0.6%  centroid_offset < 0.01838 and mean_eta > 6.289e-05
        - 0.005603415 * Q.z_dr_0p2_0p4 / 0.02920532   # -0.6%  z_dr_0p2_0p4
        - 0.004804043 * max(0.0, 7.0 - Q.n_for_90pct) * max(0.0, Q.dr_2 - 0.00656258) / 0.02004733   # -0.5%  n_for_90pct < 7 and dr_2 > 0.006563
        + 0.004743389 * max(0.0, 0.06503035 - Q.z_5) / 0.007566857   # +0.5%  z_5 < 0.06503
        + 0.004393031 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +0.4%  e2 > 0.04447
        - 0.004378164 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1) / 0.1931178   # -0.4%  centroid_offset < 0.01838 and pt_1 < 159.2
        + 0.004245615 * max(0.0, 0.006292091 - Q.zdr_0) * max(0.0, Q.pt_2 - 88.4375) / 0.02706551   # +0.4%  zdr_0 < 0.006292 and pt_2 > 88.44
        + 0.004213617 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.008712838   # +0.4%  centroid_offset < 0.01838 and n_for_90pct > 5
        - 0.003958993 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.01108383   # -0.4%  n_dr_0p2_0p4 > 1 and sj3_dr_min < 0.209
        - 0.003901398 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.pair_mass_0_2 - 17.9037) / 0.01350362   # -0.4%  centroid_offset < 0.01838 and pair_mass_0_2 > 17.9
        + 0.003878403 * max(0.0, 0.006292091 - Q.zdr_0) / 0.001006321   # +0.4%  zdr_0 < 0.006292
        - 0.003509466 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) / 6.087676   # -0.4%  sj3_pair_mass_max < 24.23
        - 0.00350814 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.pt1_dr01 - 17.82896) / 0.009671885   # -0.4%  centroid_offset < 0.01838 and pt1_dr01 > 17.83
        - 0.003162495 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) * max(0.0, Q.absphi_2 - 0.007518768) / 0.03147941   # -0.3%  sj3_pair_mass_max < 24.23 and absphi_2 > 0.007519
        + 0.003124567 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +0.3%  sj3_dr_max < 0.107
        + 0.002948948 * max(0.0, 53.33237 - Q.mass) * max(0.0, Q.D2 - 2.843757) / 5.555775   # +0.3%  mass < 53.33 and D2 > 2.844
        + 0.00294846 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.6875) / 0.3914557   # +0.3%  sj3_dr_max < 0.1426 and pt_6 > 33.69
        + 0.00275286 * max(0.0, 0.02227783 - Q.absphi_1) / 0.006894148   # +0.3%  absphi_1 < 0.02228
        + 0.002637783 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.tau4 - 0.001224244) / 1.099241e-05   # +0.3%  centroid_offset < 0.01838 and tau4 > 0.001224
        - 0.002537351 * max(0.0, Q.girth2_top5 - 0.01148293) / 0.00154713   # -0.3%  girth2_top5 > 0.01148
        + 0.002528646 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top2 - 1.933087) / 0.0177962   # +0.3%  mass_over_sum_pt < 0.07269 and mass_top2 > 1.933
        + 0.002526877 * max(0.0, 0.0782171 - Q.M3) * max(0.0, Q.pt_2 - 80.875) / 0.152146   # +0.3%  M3 < 0.07822 and pt_2 > 80.88
        - 0.002510835 * max(0.0, Q.sd_mass - 45.595) * max(0.0, Q.n_dr_0p05_0p1 - 7.0) / 0.4890223   # -0.3%  sd_mass > 45.59 and n_dr_0p05_0p1 > 7
        - 0.00239804 * max(0.0, 0.05464922 - Q.girth) * max(0.0, Q.z_6 - 0.03448406) / 0.0003071415   # -0.2%  girth < 0.05465 and z_6 > 0.03448
        + 0.002392478 * max(0.0, 31.125 - Q.pt_4) / 0.3140792   # +0.2%  pt_4 < 31.12
        - 0.00218042 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, 0.02656143 - Q.tau21_b2) / 8.029519e-06   # -0.2%  tau1 < 0.0437 and tau21_b2 < 0.02656
        + 0.002173059 * max(0.0, 0.002316125 - Q.centroid_offset) / 9.872932e-05   # +0.2%  centroid_offset < 0.002316
        + 0.00215895 * max(0.0, Q.N2 - 0.2233283) / 0.04872624   # +0.2%  N2 > 0.2233
        + 0.001905075 * max(0.0, 0.05464922 - Q.girth) * max(0.0, Q.z_5 - 0.03672711) / 0.0004195788   # +0.2%  girth < 0.05465 and z_5 > 0.03673
        + 0.001852477 * max(0.0, 988.4078 - Q.sum_pt) * max(0.0, 0.01778111 - Q.dr_2) / 0.3401129   # +0.2%  sum_pt < 988.4 and dr_2 < 0.01778
        - 0.001695911 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 0.1551147 - Q.dr_4) / 7.495507e-06   # -0.2%  lam2 > 0.001131 and dr_4 < 0.1551
        - 0.001628924 * max(0.0, Q.mass - 45.595) / 9.461082   # -0.2%  mass > 45.59
        - 0.001501452 * max(0.0, Q.e2 - 0.04447357) * max(0.0, 0.2723288 - Q.sj3_mass3) / 0.0006651999   # -0.2%  e2 > 0.04447 and sj3_mass3 < 0.2723
        + 0.001445067 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.1%  lam2 > 0.001131
        - 0.001405876 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2) / 0.000627491   # -0.1%  log_sum_pt > 6.896 and D2_b2 < 0.7166
        + 0.001390041 * max(0.0, 2.371297e-05 - Q.e3) * max(0.0, Q.eccentricity - 0.8319502) / 8.425435e-07   # +0.1%  e3 < 2.371e-05 and eccentricity > 0.832
        - 0.001378391 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, -0.01275329 - Q.mean_phi) / 1.158446e-05   # -0.1%  tau1 < 0.0437 and mean_phi < -0.01275
        + 0.001366028 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.001255404 - Q.zdr_5) / 2.213187e-06   # +0.1%  log_sum_pt > 6.896 and zdr_5 < 0.001255
        - 0.001304052 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, Q.mean_phi - 0.02612796) / 3.162575e-07   # -0.1%  girth2 < 0.006097 and mean_phi > 0.02613
        - 0.001274819 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) * max(0.0, -0.004917145 - Q.phi_0) / 9.23841e-05   # -0.1%  mass_over_sum_pt < 0.07269 and phi_0 < -0.004917
        - 0.001073886 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # -0.1%  n_dr_0p2_0p4 > 1
        - 0.001005227 * max(0.0, Q.D2 - 2.055451) * max(0.0, 1.0 - Q.psi_0p3) / 0.0009851735   # -0.1%  D2 > 2.055 and psi_0p3 < 1
        - 0.0009401219 * max(0.0, Q.e3 - 0.0001869378) * max(0.0, Q.pt_2 - 56.5) / 0.0005941291   # -0.1%  e3 > 0.0001869 and pt_2 > 56.5
        + 0.0009022704 * max(0.0, Q.sum_pt_top5 - 839.9547) * max(0.0, 1.797343 - Q.pt1_dr01) / 5.637585   # +0.1%  sum_pt_top5 > 840 and pt1_dr01 < 1.797
        + 0.0008817719 * max(0.0, 31.125 - Q.pt_4) * max(0.0, Q.M3 - 0.05210278) / 0.0007324993   # +0.1%  pt_4 < 31.12 and M3 > 0.0521
        + 0.0007209576 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.n_dr_0p05_0p1 - 5.0) / 0.001271313   # +0.1%  sj3_dr_max < 0.1426 and n_dr_0p05_0p1 > 5
        + 0.0007131115 * max(0.0, 7.0 - Q.n_for_90pct) / 0.5603294   # +0.1%  n_for_90pct < 7
        - 0.0007073218 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.1%  log_sum_pt > 6.896
        + 0.0006255118 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, Q.abseta_1 - 0.03625488) / 8.135786e-06   # +0.1%  sj3_dr_max < 0.107 and abseta_1 > 0.03625
        + 0.0005971292 * max(0.0, Q.girth2 - 0.01882765) / 0.0007605369   # +0.1%  girth2 > 0.01883
        - 0.000506889 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.mean_phi - 0.02612796) / 2.994783e-06   # -0.1%  tau1 < 0.0437 and mean_phi > 0.02613
        - 0.0004974862 * max(0.0, 0.05464922 - Q.girth) * max(0.0, 0.02656143 - Q.tau21_b2) / 1.23798e-05   # -0.0%  girth < 0.05465 and tau21_b2 < 0.02656
        - 0.000400855 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.zdr_2 - 0.00759156) / 4.975701e-07   # -0.0%  tau1 < 0.0437 and zdr_2 > 0.007592
        - 0.0002689766 * max(0.0, Q.sum_pt_top5 - 839.9547) * max(0.0, 0.004811143 - Q.tau21_b2) / 0.0005843272   # -0.0%  sum_pt_top5 > 840 and tau21_b2 < 0.004811
        - 0.0002328448 * max(0.0, 31.125 - Q.pt_4) * max(0.0, 8.0 - Q.n_pt_above_1) / 0.003929533   # -0.0%  pt_4 < 31.12 and n_pt_above_1 < 8
        + 0.0001689703 * max(0.0, Q.e2 - 0.03556091) * max(0.0, Q.mean_phi - 0.009050008) / 3.742843e-05   # +0.0%  e2 > 0.03556 and mean_phi > 0.00905
        - 0.0001364853 * max(0.0, 31.125 - Q.pt_4) * max(0.0, Q.dr_min_012 - 0.01488897) / 0.0006591722   # -0.0%  pt_4 < 31.12 and dr_min_012 > 0.01489
        + 8.251655e-05 * max(0.0, Q.D2 - 2.055451) / 0.3061148   # +0.0%  D2 > 2.055
        - 5.267239e-05 * max(0.0, Q.e3 - 0.0001869378) * max(0.0, 33.6875 - Q.pt_6) / 4.695931e-05   # -0.0%  e3 > 0.0001869 and pt_6 < 33.69
        + 5.244931e-05 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # +0.0%  sum_pt_top5 > 840
        + 4.190136e-05 * max(0.0, Q.e2 - 0.03556091) * max(0.0, Q.mean_eta - 0.00189657) / 5.895833e-05   # +0.0%  e2 > 0.03556 and mean_eta > 0.001897
        - 2.914952e-05 * max(0.0, 0.0782171 - Q.M3) / 0.01197321   # -0.0%  M3 < 0.07822
        - 2.554824e-05 * max(0.0, 559.6875 - Q.sum_pt) / 16.18216   # -0.0%  sum_pt < 559.7
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 12.53;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.5267 * (0.1934633
        + 0.0975425 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # +9.8%  girth2 > 0.00752
        - 0.07358329 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -7.4%  lam1 < 0.005954
        + 0.07105596 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +7.1%  mass < 76.66
        - 0.06056444 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -6.1%  lam1 < 0.004184
        + 0.04799253 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1) / 0.002082998   # +4.8%  zdr_0 < 0.02118 and z_dr_0p05_0p1 < 0.2919
        + 0.04221333 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # +4.2%  girth2_top3 < 0.002152
        - 0.03992833 * max(0.0, 0.02355416 - Q.centroid_offset) / 0.0103615   # -4.0%  centroid_offset < 0.02355
        - 0.03958436 * max(0.0, 0.002635418 - Q.girth2) / 0.0007941346   # -4.0%  girth2 < 0.002635
        + 0.03781243 * max(0.0, Q.lam2 - 0.0003061234) / 0.0004216599   # +3.8%  lam2 > 0.0003061
        - 0.03658183 * max(0.0, 0.06473447 - Q.z_7) / 0.01631636   # -3.7%  z_7 < 0.06473
        - 0.03578956 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # -3.6%  lam1 > 0.00733
        + 0.02829921 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # +2.8%  tau1 > 0.05357
        + 0.02367547 * max(0.0, 45.75 - Q.pt_7) / 12.14632   # +2.4%  pt_7 < 45.75
        + 0.02141485 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # +2.1%  zdr_0 < 0.02118
        - 0.02122924 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -2.1%  LHA > 0.3033
        + 0.02051608 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.002471 - Q.D2) / 1.812028   # +2.1%  pt_7 < 45.75 and D2 < 1.002
        - 0.01838729 * max(0.0, 0.00325401 - Q.zdr_7) / 0.001049641   # -1.8%  zdr_7 < 0.003254
        + 0.01661907 * max(0.0, Q.mass_top5 - 22.18342) / 12.67741   # +1.7%  mass_top5 > 22.18
        + 0.01459797 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, 1.002471 - Q.D2) / 0.009558568   # +1.5%  tau1 > 0.05357 and D2 < 1.002
        + 0.01399826 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.01258764) / 3.400017e-05   # +1.4%  zdr_0 < 0.02118 and centroid_offset > 0.01259
        + 0.01370912 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +1.4%  e3 < 8.148e-05
        - 0.01269024 * max(0.0, Q.mass_top5 - 45.32077) / 3.61051   # -1.3%  mass_top5 > 45.32
        + 0.0114535 * max(0.0, Q.sj3_pair_mass_min - 15.95929) / 1.348548   # +1.1%  sj3_pair_mass_min > 15.96
        + 0.01105075 * Q.e2 / 0.02863215   # +1.1%  e2
        + 0.01034128 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +1.0%  sj3_dr_min > 0.1278
        + 0.009648102 * Q.mean_phi / 0.01077348   # +1.0%  mean_phi
        - 0.009053509 * max(0.0, 0.004918231 - Q.zdr_0) / 0.0006323791   # -0.9%  zdr_0 < 0.004918
        - 0.008303578 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.planar_flow - 0.04505724) / 0.0002791469   # -0.8%  lam2 > 0.0003061 and planar_flow > 0.04506
        - 0.008140123 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # -0.8%  sj3_dr_max > 0.1986
        + 0.007984214 * max(0.0, 45.75 - Q.pt_7) * max(0.0, Q.abseta_3 - 0.0013237) / 0.460532   # +0.8%  pt_7 < 45.75 and abseta_3 > 0.001324
        - 0.007807979 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.2919447) / 6.832392e-05   # -0.8%  lam1 < 0.005954 and z_dr_0p05_0p1 > 0.2919
        + 0.007718512 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # +0.8%  sj3_pair_mass_min > 11.05
        + 0.007495513 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0008259449   # +0.7%  lam1 < 0.005954 and n_pt_above_50 > 6
        + 0.007286493 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.sj3_pairmax_over_m - 0.6745796) / 3.382029e-05   # +0.7%  lam2 > 0.0003061 and sj3_pairmax_over_m > 0.6746
        - 0.007017747 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2) / 0.0002929492   # -0.7%  lam1 > 0.00733 and D2_b2 < 0.3809
        - 0.006864038 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -0.7%  e3 > 3.892e-05
        + 0.006280657 * max(0.0, Q.C2_b2 - 0.009032972) / 0.001703043   # +0.6%  C2_b2 > 0.009033
        - 0.006030201 * max(0.0, Q.girth2 - 0.02530566) / 0.0002851418   # -0.6%  girth2 > 0.02531
        + 0.004854573 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.dr_7 - 0.1078355) / 8.813877e-05   # +0.5%  zdr_0 < 0.02118 and dr_7 > 0.1078
        - 0.004472497 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -0.4%  lam2 > 0.003408
        - 0.004350997 * max(0.0, Q.n_pt_above_50 - 6.0) / 0.2884353   # -0.4%  n_pt_above_50 > 6
        + 0.004019035 * max(0.0, Q.LHA - 0.3033137) * max(0.0, Q.sj3_mass1 - 5.294926) / 0.009410502   # +0.4%  LHA > 0.3033 and sj3_mass1 > 5.295
        - 0.003921949 * max(0.0, 6.080494 - Q.log_sum_pt) / 0.006097657   # -0.4%  log_sum_pt < 6.08
        + 0.00379146 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, Q.eta_7 - -0.0803833) / 0.002483399   # +0.4%  tau1 > 0.05357 and eta_7 > -0.08038
        + 0.003691999 * max(0.0, Q.sj3_pair_mass_min - 11.051) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 3.623232   # +0.4%  sj3_pair_mass_min > 11.05 and n_dr_0p2_0p4 > 0
        + 0.003093397 * max(0.0, Q.e3 - 3.892127e-05) * max(0.0, Q.zdr_7 - 0.00279494) / 4.771428e-07   # +0.3%  e3 > 3.892e-05 and zdr_7 > 0.002795
        + 0.003071861 * max(0.0, 0.02563286 - Q.M2) / 0.001872375   # +0.3%  M2 < 0.02563
        + 0.002757036 * max(0.0, Q.phi_6 - 0.04013062) / 0.01096667   # +0.3%  phi_6 > 0.04013
        + 0.002691822 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.3%  sum_pt > 988.4
        - 0.00258689 * max(0.0, Q.sj3_pair_mass_min - 11.051) * max(0.0, Q.sj3_pairmin_over_m - 0.28737) / 0.2416857   # -0.3%  sj3_pair_mass_min > 11.05 and sj3_pairmin_over_m > 0.2874
        - 0.002374581 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, -0.004664942 - Q.mean_eta) / 3.825289e-05   # -0.2%  z_7 < 0.06473 and mean_eta < -0.004665
        - 0.002290516 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.3658817 - Q.z_dr_0_0p05) / 0.002463327   # -0.2%  sj3_dr_min > 0.1278 and z_dr_0_0p05 < 0.3659
        - 0.002270822 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # -0.2%  sj2_dr > 0.3004
        + 0.00214137 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # +0.2%  mean_eta > 0.02644
        + 0.002081127 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj3_dr13 - 0.181053) / 5.645056e-07   # +0.2%  e3 < 8.148e-05 and sj3_dr13 > 0.1811
        - 0.002021805 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, 31.90625 - Q.pt_6) / 0.0003664138   # -0.2%  lam2 > 0.0003061 and pt_6 < 31.91
        + 0.001984322 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 6.572938 - Q.log_sum_pt) / 1.257183   # +0.2%  pt_7 < 45.75 and log_sum_pt < 6.573
        + 0.001980042 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.phi_5 - 2.199411e-05) / 0.0001175681   # +0.2%  zdr_0 < 0.02118 and phi_5 > 2.199e-05
        - 0.001933414 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -0.2%  lam1 < 0.001504
        + 0.001560548 * max(0.0, 76.6557 - Q.mass) * max(0.0, Q.D3 - 1.173493) / 7.575827   # +0.2%  mass < 76.66 and D3 > 1.173
        - 0.001496466 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.2723288 - Q.sj3_mass3) / 0.0003425911   # -0.1%  lam1 > 0.00733 and sj3_mass3 < 0.2723
        + 0.001292242 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.mratio_max_012 - 0.8183982) / 2.871114e-06   # +0.1%  lam2 > 0.0003061 and mratio_max_012 > 0.8184
        - 0.001243245 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, Q.D2_b2 - 1.129616) / 0.0002590844   # -0.1%  lam1 > 0.00733 and D2_b2 > 1.13
        - 0.001173872 * max(0.0, 6.080494 - Q.log_sum_pt) * max(0.0, Q.mean_eta2 - 9.030369e-05) / 5.001864e-05   # -0.1%  log_sum_pt < 6.08 and mean_eta2 > 9.03e-05
        + 0.001142744 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.02695109 - Q.dr_min_012) / 0.0001231931   # +0.1%  sj3_dr_min > 0.1278 and dr_min_012 < 0.02695
        - 0.001083781 * max(0.0, Q.mass_top5 - 45.32077) * max(0.0, Q.eta_3 - -0.00592804) / 0.1673223   # -0.1%  mass_top5 > 45.32 and eta_3 > -0.005928
        - 0.001069232 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.dr1_6 - 0.07796252) / 0.07926774   # -0.1%  sum_pt > 988.4 and dr1_6 > 0.07796
        + 0.000829294 * max(0.0, Q.mass_top5 - 45.32077) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 3.955556   # +0.1%  mass_top5 > 45.32 and n_dr_0p2_0p4 < 2
        - 0.0007496969 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.03573274 - Q.sj2_mass2) / 7.668628e-05   # -0.1%  sj3_dr_min > 0.1278 and sj2_mass2 < 0.03573
        + 0.0007050611 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 1.0 - Q.psi_0p3) / 0.0002192504   # +0.1%  sj3_dr_min > 0.1278 and psi_0p3 < 1
        + 0.000648288 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1797097) / 5.905455e-07   # +0.1%  e3 < 8.148e-05 and sj3_dr23 > 0.1797
        - 0.0006105953 * max(0.0, 6.080494 - Q.log_sum_pt) * max(0.0, Q.absphi_7 - 0.03546143) / 0.0002718019   # -0.1%  log_sum_pt < 6.08 and absphi_7 > 0.03546
        + 0.0006057745 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.03607145 - Q.dr_7) / 6.704906e-07   # +0.1%  lam1 > 0.00733 and dr_7 < 0.03607
        - 0.0005821037 * max(0.0, Q.e3 - 3.892127e-05) * max(0.0, Q.abseta_7 - 0.05718994) / 3.23493e-06   # -0.1%  e3 > 3.892e-05 and abseta_7 > 0.05719
        + 0.0004023338 * max(0.0, 488.9312 - Q.sum_pt) / 6.000379   # +0.0%  sum_pt < 488.9
        + 0.0003966496 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 1.911889 - Q.D2_b2) / 0.005346342   # +0.0%  sj3_dr_min > 0.1278 and D2_b2 < 1.912
        + 0.0003813676 * max(0.0, Q.LHA - 0.3033137) * max(0.0, Q.sj3_pairmax_over_m - 0.901705) / 9.935534e-05   # +0.0%  LHA > 0.3033 and sj3_pairmax_over_m > 0.9017
        - 0.0003688961 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.5973755 - Q.sj3_mass2) / 0.001008755   # -0.0%  sj3_dr_min > 0.1278 and sj3_mass2 < 0.5974
        + 0.0003357799 * max(0.0, 76.6557 - Q.mass) * max(0.0, -0.0496521 - Q.phi_4) / 0.1061029   # +0.0%  mass < 76.66 and phi_4 < -0.04965
        - 0.0001819387 * max(0.0, 45.75 - Q.pt_7) * max(0.0, Q.tau4 - 0.007583927) / 0.001915061   # -0.0%  pt_7 < 45.75 and tau4 > 0.007584
        - 0.0001809655 * max(0.0, Q.girth2 - 0.02530566) * max(0.0, Q.tau43 - 0.7296607) / 1.131197e-06   # -0.0%  girth2 > 0.02531 and tau43 > 0.7297
        + 0.0001177413 * max(0.0, Q.girth2 - 0.007520088) * max(0.0, 0.003088708 - Q.dr1_4) / 4.730438e-08   # +0.0%  girth2 > 0.00752 and dr1_4 < 0.003089
        + 0.0001053363 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, Q.absphi_5 - 0.04455566) / 2.4347e-05   # +0.0%  mean_eta > 0.02644 and absphi_5 > 0.04456
        + 9.295701e-05 * max(0.0, Q.phi_6 - 0.04013062) * max(0.0, -0.1218262 - Q.eta_7) / 4.512671e-05   # +0.0%  phi_6 > 0.04013 and eta_7 < -0.1218
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 61.3;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 61.30412 * (-0.02170715
        - 0.2349114 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -23.5%  e2_sq < 0.008169
        + 0.2163745 * max(0.0, 0.00817466 - Q.mass_over_sum_pt_sq) / 0.004299658   # +21.6%  mass_over_sum_pt_sq < 0.008175
        + 0.09223747 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +9.2%  width < 0.008678
        + 0.05320777 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +5.3%  girth2 < 0.01324
        + 0.02982519 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +3.0%  centroid_offset < 0.03776
        - 0.02775842 * max(0.0, 0.169029 - Q.sj3_dr_max) / 0.04876366   # -2.8%  sj3_dr_max < 0.169
        - 0.02466554 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -2.5%  girth < 0.0717
        + 0.01940188 * max(0.0, 0.2623172 - Q.sj3_dr_max) / 0.1129006   # +1.9%  sj3_dr_max < 0.2623
        - 0.01867138 * max(0.0, 0.07637363 - Q.mass_over_sum_pt) / 0.02715027   # -1.9%  mass_over_sum_pt < 0.07637
        + 0.01864513 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +1.9%  z_7 > 0.01686
        + 0.01697835 * max(0.0, 0.006679471 - Q.width) / 0.00293624   # +1.7%  width < 0.006679
        - 0.01464011 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -1.5%  lam1 < 0.008376
        - 0.01328003 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # -1.3%  tau1 < 0.09539
        + 0.01296947 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +1.3%  sj3_dr_max < 0.1426
        + 0.01237918 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +1.2%  lam2 < 0.001131
        + 0.01236302 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +1.2%  planar_flow < 0.2534
        - 0.01202671 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -1.2%  mass < 69.61
        - 0.01182253 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 15.95929 - Q.sj3_pair_mass_min) / 0.05098851   # -1.2%  centroid_offset < 0.01438 and sj3_pair_mass_min < 15.96
        - 0.009856056 * max(0.0, 0.004372139 - Q.girth2) / 0.00156571   # -1.0%  girth2 < 0.004372
        + 0.007596264 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # +0.8%  max_dr < 0.1452
        - 0.006708539 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -0.7%  lam1 < 0.00733
        + 0.006068902 * max(0.0, 791.125 - Q.sum_pt_top5) / 212.2282   # +0.6%  sum_pt_top5 < 791.1
        - 0.005981182 * max(0.0, 0.02689598 - Q.girth) / 0.004330137   # -0.6%  girth < 0.0269
        + 0.005866808 * max(0.0, 0.09538712 - Q.tau1) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.03209574   # +0.6%  tau1 < 0.09539 and z_dr_0p05_0p1 < 0.846
        - 0.005505916 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt) / 3.513015   # -0.6%  centroid_offset < 0.03776 and sum_pt < 901.6
        - 0.005432073 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.03601074 - Q.abseta_4) / 8.179649e-05   # -0.5%  centroid_offset < 0.01438 and abseta_4 < 0.03601
        - 0.005039064 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.00279494 - Q.zdr_7) / 5.690817e-06   # -0.5%  centroid_offset < 0.01438 and zdr_7 < 0.002795
        + 0.004640976 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.pt_6 - 19.46875) / 0.1742442   # +0.5%  girth2 < 0.01324 and pt_6 > 19.47
        + 0.004585668 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0007565864   # +0.5%  centroid_offset < 0.01438 and D2_b2 < 0.5327
        + 0.004535554 * max(0.0, 15.45403 - Q.mass) * max(0.0, 0.003780225 - Q.zdr_7) / 0.006870481   # +0.5%  mass < 15.45 and zdr_7 < 0.00378
        - 0.004371933 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001130645 - Q.lam2) / 1.843299e-05   # -0.4%  sj2_dr > 0.1779 and lam2 < 0.001131
        - 0.004297704 * max(0.0, 0.1027585 - Q.max_dr) / 0.02411847   # -0.4%  max_dr < 0.1028
        - 0.00405237 * max(0.0, 0.02054282 - Q.girth) / 0.002592388   # -0.4%  girth < 0.02054
        - 0.003848994 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -0.4%  sj2_dr > 0.1779
        + 0.003420593 * max(0.0, 0.01655442 - Q.e2) / 0.003887347   # +0.3%  e2 < 0.01655
        - 0.003411013 * max(0.0, Q.n_dr_0p1_0p2 - 3.0) / 0.3318622   # -0.3%  n_dr_0p1_0p2 > 3
        - 0.003372844 * max(0.0, 0.0005049491 - Q.lam1) / 8.739619e-05   # -0.3%  lam1 < 0.0005049
        - 0.003245142 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 1.050302e-05 - Q.e3) / 1.771164e-07   # -0.3%  planar_flow < 0.2534 and e3 < 1.05e-05
        - 0.003079205 * max(0.0, 506.875 - Q.sum_pt_top5) / 34.75344   # -0.3%  sum_pt_top5 < 506.9
        - 0.002937798 * max(0.0, 0.02689598 - Q.girth) * max(0.0, Q.sj3_pair_mass_min - 1.590484) / 0.005084766   # -0.3%  girth < 0.0269 and sj3_pair_mass_min > 1.59
        + 0.002857225 * max(0.0, 1.050302e-05 - Q.e3) / 3.717236e-06   # +0.3%  e3 < 1.05e-05
        - 0.002661068 * max(0.0, 0.003676313 - Q.zdr_0) / 0.0003550098   # -0.3%  zdr_0 < 0.003676
        + 0.002344168 * max(0.0, Q.eccentricity - 0.9884745) / 0.001961982   # +0.2%  eccentricity > 0.9885
        + 0.002272838 * max(0.0, 15.45403 - Q.mass) / 2.385786   # +0.2%  mass < 15.45
        + 0.002241361 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 0.04977417 - Q.abseta_4) / 0.001815038   # +0.2%  planar_flow < 0.2534 and abseta_4 < 0.04977
        - 0.002014872 * max(0.0, 49.6681 - Q.mass) * max(0.0, 1.332146 - Q.D2) / 1.334408   # -0.2%  mass < 49.67 and D2 < 1.332
        + 0.002001833 * max(0.0, 506.875 - Q.sum_pt_top5) * max(0.0, Q.z_dr_0p1_0p2 - 0.04510668) / 8.800637   # +0.2%  sum_pt_top5 < 506.9 and z_dr_0p1_0p2 > 0.04511
        - 0.001861785 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.2%  planar_flow < 0.2534 and sum_pt < 840
        - 0.001856237 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 0.0345459 - Q.absphi_0) / 0.0003803849   # -0.2%  centroid_offset < 0.03776 and absphi_0 < 0.03455
        - 0.001831572 * max(0.0, 49.6681 - Q.mass) / 17.06893   # -0.2%  mass < 49.67
        - 0.001625389 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.0283055   # -0.2%  sj2_dr > 0.1779 and n_pt_above_50 > 4
        + 0.001616211 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.2%  sj2_dr > 0.2688
        - 0.001592032 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # -0.2%  centroid_offset > 0.0499 and D2 < 2.844
        + 0.001553013 * max(0.0, 0.01437952 - Q.centroid_offset) / 0.004306483   # +0.2%  centroid_offset < 0.01438
        - 0.00151564 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # -0.2%  C2 < 0.03579
        - 0.001493088 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 4.721224 - Q.D2_b2) / 16.05793   # -0.1%  pt_6 < 41.22 and D2_b2 < 4.721
        + 0.001294664 * max(0.0, 0.2623172 - Q.sj3_dr_max) * max(0.0, -0.0216713 - Q.phi_0) / 0.0003847124   # +0.1%  sj3_dr_max < 0.2623 and phi_0 < -0.02167
        + 0.001291942 * max(0.0, 0.004372139 - Q.girth2) * max(0.0, Q.sj3_dr13 - 0.1659434) / 5.846229e-06   # +0.1%  girth2 < 0.004372 and sj3_dr13 > 0.1659
        + 0.001284713 * max(0.0, Q.z_dr_0p05_0p1 - 0.8460335) / 0.01076887   # +0.1%  z_dr_0p05_0p1 > 0.846
        - 0.00126278 * max(0.0, Q.eccentricity - 0.9884745) * max(0.0, 2.771069 - Q.sj2_mass2) / 0.00304207   # -0.1%  eccentricity > 0.9885 and sj2_mass2 < 2.771
        - 0.0009460487 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 37.15625 - Q.pt_7) / 0.5537514   # -0.1%  planar_flow < 0.2534 and pt_7 < 37.16
        - 0.0008512318 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.tau3 - 0.001996306) / 6.194014e-06   # -0.1%  centroid_offset > 0.0499 and tau3 > 0.001996
        - 0.0008477569 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.1%  sum_pt > 988.4
        - 0.0007837074 * max(0.0, 15.45403 - Q.mass) * max(0.0, -0.004917145 - Q.phi_0) / 0.005624273   # -0.1%  mass < 15.45 and phi_0 < -0.004917
        - 0.000779096 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.5502779 - Q.tau32) / 0.0001096908   # -0.1%  centroid_offset > 0.0499 and tau32 < 0.5503
        - 0.0007047284 * max(0.0, Q.max_pair_mass - 33.3761) / 0.9279851   # -0.1%  max_pair_mass > 33.38
        + 0.0006941917 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 0.415246 - Q.D2) / 8.563062e-05   # +0.1%  girth2 < 0.01324 and D2 < 0.4152
        + 0.0006905412 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +0.1%  pt_6 < 41.22
        + 0.0006667715 * max(0.0, 0.008168571 - Q.e2_sq) * max(0.0, Q.mass_top2 - 6.779915) / 0.005654747   # +0.1%  e2_sq < 0.008169 and mass_top2 > 6.78
        - 0.000636563 * max(0.0, Q.max_pair_mass - 33.3761) * max(0.0, 4.966909 - Q.sj2_mass1) / 0.9445084   # -0.1%  max_pair_mass > 33.38 and sj2_mass1 < 4.967
        + 0.0006064077 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, Q.phi_1 - 0.002190304) / 4.583393e-05   # +0.1%  centroid_offset < 0.01438 and phi_1 > 0.00219
        + 0.0004875145 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.04881348 - Q.dr_7) / 8.444248e-05   # +0.0%  sj2_dr > 0.1779 and dr_7 < 0.04881
        - 0.0004871082 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 61.53125 - Q.pt_2) / 0.009295235   # -0.0%  girth2 < 0.01324 and pt_2 < 61.53
        + 0.0004377581 * max(0.0, 69.61135 - Q.mass) * max(0.0, 0.5502779 - Q.tau32) / 0.9545616   # +0.0%  mass < 69.61 and tau32 < 0.5503
        + 0.0004326243 * max(0.0, 69.61135 - Q.mass) * max(0.0, Q.pair_mass_0_2 - 11.71328) / 21.07106   # +0.0%  mass < 69.61 and pair_mass_0_2 > 11.71
        - 0.0004221131 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.zdr_6 - 0.007390416) / 3.11545e-06   # -0.0%  centroid_offset > 0.0499 and zdr_6 > 0.00739
        + 0.0003579518 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, Q.sj3_pair_mass_min - 1.282345) / 0.4025762   # +0.0%  planar_flow < 0.2534 and sj3_pair_mass_min > 1.282
        + 0.0003494784 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.5186963 - Q.tau32) / 0.0001128939   # +0.0%  centroid_offset < 0.01438 and tau32 < 0.5187
        + 0.0003362659 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # +0.0%  LHA < 0.3127
        + 0.0003011393 * max(0.0, 0.07368057 - Q.z_3rd) / 0.0007500377   # +0.0%  z_3rd < 0.07368
        + 0.0002562747 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, -0.001813533 - Q.mean_phi) / 4.860641e-05   # +0.0%  centroid_offset < 0.03776 and mean_phi < -0.001814
        + 0.0002412803 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, -0.005273438 - Q.eta_1) / 4.114128e-05   # +0.0%  centroid_offset < 0.01438 and eta_1 < -0.005273
        - 0.0001713885 * max(0.0, 15.45403 - Q.mass) * max(0.0, Q.abseta_5 - 0.06137085) / 0.0001841249   # -0.0%  mass < 15.45 and abseta_5 > 0.06137
        - 0.0001662872 * max(0.0, Q.eccentricity - 0.9884745) * max(0.0, 0.02130127 - Q.eta_0) / 6.650689e-05   # -0.0%  eccentricity > 0.9885 and eta_0 < 0.0213
        - 0.0001279048 * max(0.0, 49.6681 - Q.mass) * max(0.0, Q.dr1_7 - 0.1792439) / 0.02583476   # -0.0%  mass < 49.67 and dr1_7 > 0.1792
        + 9.105627e-05 * max(0.0, Q.pt_7 - 29.04219) / 7.785987   # +0.0%  pt_7 > 29.04
        - 9.045464e-05 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.0001991733   # -0.0%  centroid_offset < 0.03776 and dr_max_012 > 0.1828
        - 8.416795e-05 * max(0.0, 69.61135 - Q.mass) * max(0.0, 0.415246 - Q.D2) / 0.1191559   # -0.0%  mass < 69.61 and D2 < 0.4152
        + 7.040841e-05 * max(0.0, 24.42188 - Q.pt_6) * max(0.0, 0.110654 - Q.mratio_min_012) / 0.007626969   # +0.0%  pt_6 < 24.42 and mratio_min_012 < 0.1107
        + 6.810129e-05 * max(0.0, 24.42188 - Q.pt_6) / 0.6046572   # +0.0%  pt_6 < 24.42
        + 5.661514e-05 * max(0.0, Q.z_dr_0p05_0p1 - 0.8460335) * max(0.0, 0.3593346 - Q.pt1_over_pt0) / 6.5712e-05   # +0.0%  z_dr_0p05_0p1 > 0.846 and pt1_over_pt0 < 0.3593
        + 5.471788e-05 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.dr_min_012 - 0.01488897) / 0.001081167   # +0.0%  sum_pt > 988.4 and dr_min_012 > 0.01489
        - 4.796438e-05 * max(0.0, 0.169029 - Q.sj3_dr_max) * max(0.0, Q.pair_mass_0_4 - 4.619568) / 0.02101062   # -0.0%  sj3_dr_max < 0.169 and pair_mass_0_4 > 4.62
        - 4.211182e-05 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.phi_2 - 0.06414795) / 1.239903e-06   # -0.0%  sj3_dr_max < 0.1426 and phi_2 > 0.06415
        - 3.998861e-05 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.0%  centroid_offset > 0.0499
        - 3.944091e-05 * max(0.0, 15.45403 - Q.mass) * max(0.0, -0.05890198 - Q.phi_1) / 6.939758e-05   # -0.0%  mass < 15.45 and phi_1 < -0.0589
        + 2.822575e-05 * max(0.0, Q.n_dr_0p1_0p2 - 3.0) * max(0.0, 0.113617 - Q.z_1) / 0.0002649226   # +0.0%  n_dr_0p1_0p2 > 3 and z_1 < 0.1136
        + 1.029468e-05 * max(0.0, 0.2623172 - Q.sj3_dr_max) * max(0.0, Q.pair_mass_0_2 - 40.2) / 0.001291824   # +0.0%  sj3_dr_max < 0.2623 and pair_mass_0_2 > 40.2
        + 6.446827e-06 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, -0.07818909 - Q.phi_0) / 0.0007184678   # +0.0%  sum_pt > 988.4 and phi_0 < -0.07819
        + 2.794926e-06 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, Q.dr_min_012 - 0.02695109) / 6.832345e-07   # +0.0%  lam2 < 0.001131 and dr_min_012 > 0.02695
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.7347;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.7347079 * (-0.8247436
        + 0.315428 * max(0.0, Q.girth2 - 0.01882765) / 0.0007605369   # +31.5%  girth2 > 0.01883
        - 0.09559587 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -9.6%  e2 > 0.06344
        + 0.09319106 * max(0.0, Q.mass - 69.61135) * max(0.0, Q.centroid_offset - 0.009480685) / 0.04081507   # +9.3%  mass > 69.61 and centroid_offset > 0.009481
        + 0.06825085 * max(0.0, Q.mass - 91.19) / 0.5839191   # +6.8%  mass > 91.19
        - 0.06434963 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, 53.4375 - Q.pt_7) / 0.01439677   # -6.4%  girth2 > 0.01883 and pt_7 < 53.44
        + 0.05434148 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, Q.abseta_6 - 0.01612854) / 7.142361e-05   # +5.4%  girth2 > 0.01883 and abseta_6 > 0.01613
        - 0.05178605 * max(0.0, Q.zdr_0 - 0.03981924) / 0.0005695785   # -5.2%  zdr_0 > 0.03982
        - 0.05014027 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, Q.lam2 - 0.000537286) / 2.828867e-06   # -5.0%  girth2 > 0.01883 and lam2 > 0.0005373
        - 0.04700629 * max(0.0, Q.mass_over_sum_pt - 0.1309286) / 0.002541185   # -4.7%  mass_over_sum_pt > 0.1309
        + 0.0318867 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +3.2%  centroid_offset > 0.0499
        - 0.02995117 * max(0.0, Q.mass - 69.61135) / 2.406702   # -3.0%  mass > 69.61
        + 0.01897676 * max(0.0, Q.mass - 69.61135) * max(0.0, 6.423044 - Q.log_sum_pt) / 0.0982281   # +1.9%  mass > 69.61 and log_sum_pt < 6.423
        - 0.01491222 * max(0.0, Q.girth2_top2 - 0.01403324) * max(0.0, -0.006701703 - Q.mean_phi) / 9.987654e-06   # -1.5%  girth2_top2 > 0.01403 and mean_phi < -0.006702
        + 0.01368316 * max(0.0, Q.mean_phi - 0.02612796) / 0.0006999411   # +1.4%  mean_phi > 0.02613
        + 0.01325375 * max(0.0, Q.mass - 69.61135) * max(0.0, Q.dr_4 - 0.02477348) / 0.3143833   # +1.3%  mass > 69.61 and dr_4 > 0.02477
        - 0.01165099 * max(0.0, Q.girth2_top2 - 0.01403324) / 0.001129575   # -1.2%  girth2_top2 > 0.01403
        - 0.01110403 * max(0.0, Q.mass - 91.19) * max(0.0, Q.centroid_offset - 0.02076709) / 0.004710116   # -1.1%  mass > 91.19 and centroid_offset > 0.02077
        - 0.004264444 * max(0.0, Q.mass - 69.61135) * max(0.0, Q.abseta_5 - 0.1522217) / 0.02987685   # -0.4%  mass > 69.61 and abseta_5 > 0.1522
        - 0.003572786 * max(0.0, Q.mass - 91.19) * max(0.0, Q.eta_5 - 0.007119751) / 0.02885229   # -0.4%  mass > 91.19 and eta_5 > 0.00712
        + 0.002140461 * max(0.0, Q.mass - 91.19) * max(0.0, 50.3522 - Q.mass_top3) / 4.712306   # +0.2%  mass > 91.19 and mass_top3 < 50.35
        - 0.001780071 * max(0.0, Q.mass - 91.19) * max(0.0, Q.n_dr_0p2_0p4 - 2.0) / 0.4387749   # -0.2%  mass > 91.19 and n_dr_0p2_0p4 > 2
        + 0.001686093 * max(0.0, Q.mass - 91.19) * max(0.0, 0.117984 - Q.abseta_1) / 0.02338086   # +0.2%  mass > 91.19 and abseta_1 < 0.118
        + 0.00104785 * max(0.0, Q.mass - 91.19) * max(0.0, 0.002207727 - Q.tau4) / 3.151142e-05   # +0.1%  mass > 91.19 and tau4 < 0.002208
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 20.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.00981 * (0.007334643
        + 0.2752275 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # +27.5%  girth < 0.1484
        - 0.08236414 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -8.2%  e2 < 0.08001
        + 0.07349402 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +7.3%  width < 0.01324
        + 0.06772953 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +6.8%  lam1 < 0.01643
        - 0.06370747 * max(0.0, 0.007520088 - Q.width) / 0.003544991   # -6.4%  width < 0.00752
        - 0.06104366 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -6.1%  girth < 0.1484 and log_sum_pt < 6.804
        + 0.05007419 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +5.0%  lam1 < 0.006507
        - 0.04781862 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -4.8%  girth < 0.1484 and pt_7 < 38.53
        - 0.03276588 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 8.324263e-07   # -3.3%  e3 < 5.335e-05 and centroid_offset < 0.03776
        + 0.02572767 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7) / 781.5808   # +2.6%  sum_pt_top5 > 658.1 and pt_7 < 40.04
        + 0.02320988 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +2.3%  log_sum_pt > 6.503
        + 0.01767068 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # +1.8%  e3 < 5.335e-05
        - 0.01595465 * max(0.0, Q.mass - 49.6681) / 7.721594   # -1.6%  mass > 49.67
        - 0.01518491 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7) / 0.06765006   # -1.5%  pt_6 < 31.91 and z_7 < 0.05861
        - 0.01319583 * max(0.0, 0.0005124533 - Q.girth2_top2) / 0.0001104529   # -1.3%  girth2_top2 < 0.0005125
        + 0.01218086 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.tau2 - 0.008780509) / 0.000269945   # +1.2%  girth < 0.1484 and tau2 > 0.008781
        - 0.01094118 * max(0.0, 31.90625 - Q.pt_6) / 1.826854   # -1.1%  pt_6 < 31.91
        + 0.01001167 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +1.0%  sum_pt_top5 > 791.1
        + 0.009407804 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.07474969 - Q.M3) / 0.0006773733   # +0.9%  girth < 0.1484 and M3 < 0.07475
        + 0.008292622 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.z_7 - 0.06164517) / 0.0003465887   # +0.8%  girth < 0.1484 and z_7 > 0.06165
        + 0.007770909 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55592   # +0.8%  sum_pt_top5 > 658.1
        - 0.00718121 * max(0.0, Q.sj3_dr23 - 0.1974628) / 0.02478939   # -0.7%  sj3_dr23 > 0.1975
        - 0.006959492 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # -0.7%  girth < 0.007674
        + 0.006779393 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.D3 - 0.2213841) / 3.794981e-05   # +0.7%  e3 < 5.335e-05 and D3 > 0.2214
        + 0.006276973 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 31.90625 - Q.pt_6) / 134.9952   # +0.6%  sum_pt_top5 > 791.1 and pt_6 < 31.91
        + 0.005972684 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +0.6%  lam2 < 0.0001948
        + 0.005545851 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, 0.04681396 - Q.absphi_6) / 0.003005137   # +0.6%  log_sum_pt > 6.503 and absphi_6 < 0.04681
        - 0.0048255 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.000537286 - Q.lam2) / 4.021081e-05   # -0.5%  girth < 0.1484 and lam2 < 0.0005373
        - 0.003522732 * max(0.0, 0.1484084 - Q.girth) * max(0.0, -9.311297e-05 - Q.mean_phi) / 0.0003735644   # -0.4%  girth < 0.1484 and mean_phi < -9.311e-05
        + 0.003480793 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.3%  sum_pt > 988.4
        - 0.003440838 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # -0.3%  z_7 < 0.02807
        - 0.003238501 * max(0.0, Q.pt_7 - 48.71875) / 0.6759439   # -0.3%  pt_7 > 48.72
        - 0.003172333 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # -0.3%  z_5 < 0.02818
        - 0.003142541 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.sj2_mass1 - 31.78116) / 0.01215768   # -0.3%  girth < 0.1484 and sj2_mass1 > 31.78
        + 0.002775316 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.pair_mass_0_7 - 10.2219) / 0.2907061   # +0.3%  log_sum_pt > 6.503 and pair_mass_0_7 > 10.22
        + 0.002586726 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # +0.3%  z_6 < 0.0216
        - 0.002026418 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6) / 104.494   # -0.2%  sum_pt > 988.4 and pt_6 < 62.25
        + 0.001795947 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.08151794 - Q.M3) / 0.08390625   # +0.2%  sum_pt > 988.4 and M3 < 0.08152
        - 0.001194129 * max(0.0, Q.mass - 49.6681) * max(0.0, 0.4684459 - Q.z_dr_0p1_0p2) / 1.50172   # -0.1%  mass > 49.67 and z_dr_0p1_0p2 < 0.4684
        + 0.001086438 * max(0.0, Q.pt_7 - 48.71875) * max(0.0, Q.pt_2 - 126.75) / 3.467244   # +0.1%  pt_7 > 48.72 and pt_2 > 126.8
        - 0.0007102973 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.zdr_7 - 0.0007928864) / 0.0001818422   # -0.1%  log_sum_pt > 6.503 and zdr_7 > 0.0007929
        + 0.0004492758 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 74.5625 - Q.pt_3) / 205.6534   # +0.0%  sum_pt_top5 > 791.1 and pt_3 < 74.56
        + 6.29706e-05 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.02320757) / 0.2899084   # +0.0%  sum_pt_top5 > 658.1 and z_7 > 0.02321
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 33.91;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.90618 * (0.02673747
        - 0.07964069 * max(0.0, Q.width - 0.006679471) / 0.002702003   # -8.0%  width > 0.006679
        - 0.05624237 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -5.6%  girth2 > 0.00752
        + 0.05478888 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # +5.5%  girth2 > 0.008678
        + 0.05276721 * max(0.0, Q.e2_sq - 0.002074109) / 0.004484017   # +5.3%  e2_sq > 0.002074
        - 0.0524944 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -5.2%  mass_over_sum_pt > 0.09041
        + 0.05059105 * max(0.0, Q.width - 0.003562611) / 0.004066872   # +5.1%  width > 0.003563
        + 0.04057456 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # +4.1%  sj2_dr > 0.1592
        + 0.04004319 * max(0.0, Q.girth - 0.02689598) / 0.03613842   # +4.0%  girth > 0.0269
        - 0.03774008 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -3.8%  girth > 0.08724
        + 0.03283618 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +3.3%  mass_over_sum_pt > 0.07992
        - 0.03259027 * max(0.0, Q.e2_sq - 0.0030133) / 0.003940214   # -3.3%  e2_sq > 0.003013
        - 0.03231294 * max(0.0, Q.girth2 - 0.0009641429) / 0.00568632   # -3.2%  girth2 > 0.0009641
        - 0.02325345 * max(0.0, Q.e2 - 0.01655442) / 0.01596508   # -2.3%  e2 > 0.01655
        + 0.01926047 * max(0.0, Q.width - 0.005590289) / 0.003084256   # +1.9%  width > 0.00559
        + 0.01906573 * max(0.0, Q.width - 0.01323868) / 0.001442684   # +1.9%  width > 0.01324
        + 0.01902356 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +1.9%  e2 > 0.04111
        - 0.01868901 * max(0.0, 0.324646 - Q.sd_rg) / 0.2082097   # -1.9%  sd_rg < 0.3246
        + 0.01814957 * max(0.0, 49.91626 - Q.sd_mass) / 22.60916   # +1.8%  sd_mass < 49.92
        - 0.01630942 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -1.6%  planar_flow < 0.1115 and lam1 < 0.00733
        + 0.01591181 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +1.6%  LHA > 0.3033
        - 0.01572894 * max(0.0, 74.57663 - Q.sd_mass) / 42.04056   # -1.6%  sd_mass < 74.58
        + 0.01460248 * max(0.0, Q.girth - 0.04081947) / 0.0270611   # +1.5%  girth > 0.04082
        - 0.0140642 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -1.4%  e3 < 8.148e-05
        + 0.01353177 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # +1.4%  planar_flow < 0.1115 and lam1 < 0.01643
        - 0.01345977 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # -1.3%  sj2_dr > 0.1295
        + 0.01276767 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +1.3%  mass_over_sum_pt > 0.08475
        + 0.01149882 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0006014625   # +1.1%  sj2_dr > 0.1592 and D2_b2 < 0.08499
        + 0.01075119 * max(0.0, 5.0 - Q.n_dr_0_0p05) / 1.94322   # +1.1%  n_dr_0_0p05 < 5
        - 0.009477106 * max(0.0, 715.4688 - Q.sum_pt) / 69.91196   # -0.9%  sum_pt < 715.5
        + 0.00850659 * max(0.0, 0.1115136 - Q.planar_flow) / 0.03343187   # +0.9%  planar_flow < 0.1115
        - 0.008261575 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.8%  psi_0p1 > 0.9761
        - 0.007784288 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # -0.8%  e2 > 0.05028
        - 0.007680651 * max(0.0, Q.sj2_dr - 0.1294903) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0009393137   # -0.8%  sj2_dr > 0.1295 and D2_b2 < 0.08499
        - 0.00766167 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.8%  planar_flow < 0.1115 and centroid_offset < 0.01838
        - 0.007416072 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # -0.7%  e3 < 5.335e-05
        - 0.006638463 * max(0.0, Q.e2_sq - 0.002074109) * max(0.0, 0.1872617 - Q.sj2_dr) / 2.014151e-05   # -0.7%  e2_sq > 0.002074 and sj2_dr < 0.1873
        - 0.006069812 * max(0.0, Q.sj2_dr - 0.2001708) / 0.02317349   # -0.6%  sj2_dr > 0.2002
        - 0.005987974 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0003170016   # -0.6%  sj2_dr > 0.2002 and D2_b2 < 0.08499
        - 0.005891525 * max(0.0, Q.sd_rg - 0.2330919) / 0.01070351   # -0.6%  sd_rg > 0.2331
        + 0.005889217 * max(0.0, Q.girth - 0.08065885) / 0.009330634   # +0.6%  girth > 0.08066
        + 0.005822548 * max(0.0, Q.z_dr_0_0p05 - 0.1515405) / 0.4480651   # +0.6%  z_dr_0_0p05 > 0.1515
        - 0.005753307 * max(0.0, Q.z_dr_0_0p05 - 0.6080732) / 0.1760045   # -0.6%  z_dr_0_0p05 > 0.6081
        - 0.005343746 * max(0.0, Q.psi_0p1 - 0.8155839) / 0.1042073   # -0.5%  psi_0p1 > 0.8156
        - 0.005279115 * max(0.0, 0.004032342 - Q.C2_b2) / 0.002883542   # -0.5%  C2_b2 < 0.004032
        + 0.005223973 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067297   # +0.5%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        - 0.004856916 * max(0.0, Q.e2_sq - 0.01165737) / 0.001433269   # -0.5%  e2_sq > 0.01166
        - 0.004439732 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 0.02415398 - Q.C2_b2) / 0.0005249308   # -0.4%  z_dr_0p05_0p1 > 0.7509 and C2_b2 < 0.02415
        - 0.004230453 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -0.4%  sj2_dr > 0.1779
        + 0.004186126 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.width) / 3.351526e-05   # +0.4%  planar_flow < 0.1115 and width < 0.006097
        + 0.00377064 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, 150.625 - Q.pt_1) / 0.1837865   # +0.4%  centroid_offset > 0.02686 and pt_1 < 150.6
        - 0.00366539 * max(0.0, Q.sj2_dr - 0.06154135) / 0.1002688   # -0.4%  sj2_dr > 0.06154
        - 0.003390669 * max(0.0, 0.163898 - Q.z_dr_0p05_0p1) / 0.08682106   # -0.3%  z_dr_0p05_0p1 < 0.1639
        + 0.002913413 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1629004) / 4.043869e-07   # +0.3%  e3 < 5.335e-05 and sj3_dr23 > 0.1629
        - 0.002839419 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.3%  centroid_offset > 0.0499
        + 0.002341273 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # +0.2%  e2 > 0.03556
        - 0.002005762 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.03293672 - Q.dr_2) / 6.256053e-05   # -0.2%  sj2_dr > 0.1592 and dr_2 < 0.03294
        + 0.001754909 * max(0.0, Q.psi_0p1 - 0.8155839) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0006308268   # +0.2%  psi_0p1 > 0.8156 and D2_b2 < 0.08499
        - 0.001725487 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # -0.2%  girth > 0.1019
        - 0.001639975 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 0.1830092 - Q.D2_b2) / 0.008544403   # -0.2%  z_dr_0p05_0p1 < 0.5883 and D2_b2 < 0.183
        + 0.001493295 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.05056028 - Q.dr_2) / 0.0001260892   # +0.1%  sj2_dr > 0.2002 and dr_2 < 0.05056
        - 0.001466708 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 658.125 - Q.sum_pt_top5) / 2.950014   # -0.1%  planar_flow < 0.1115 and sum_pt_top5 < 658.1
        + 0.001406564 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 715.4688 - Q.sum_pt) / 22.02426   # +0.1%  z_dr_0p05_0p1 < 0.5883 and sum_pt < 715.5
        - 0.001299385 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 0.002231562   # -0.1%  planar_flow < 0.1115 and z_dr_0p05_0p1 > 0.6748
        + 0.001246867 * max(0.0, 0.07870506 - Q.z_dr_0p1_0p2) / 0.04317253   # +0.1%  z_dr_0p1_0p2 < 0.07871
        - 0.001208344 * max(0.0, Q.mass - 69.61135) / 2.406702   # -0.1%  mass > 69.61
        - 0.001101187 * max(0.0, Q.sd_rg - 0.2787955) * max(0.0, 0.2669656 - Q.D2_b2) / 0.0004100902   # -0.1%  sd_rg > 0.2788 and D2_b2 < 0.267
        + 0.001081008 * max(0.0, 0.08366273 - Q.planar_flow) / 0.02146305   # +0.1%  planar_flow < 0.08366
        + 0.001036928 * max(0.0, Q.psi_0p1 - 0.7143804) / 0.1805043   # +0.1%  psi_0p1 > 0.7144
        - 0.001008518 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.1%  centroid_offset > 0.03776
        - 0.0009274301 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.0008333816 - Q.C2_b2) / 1.341084e-07   # -0.1%  centroid_offset > 0.0499 and C2_b2 < 0.0008334
        + 0.0008370102 * max(0.0, Q.centroid_offset - 0.02685622) / 0.003254849   # +0.1%  centroid_offset > 0.02686
        + 0.00083613 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # +0.1%  z_dr_0p05_0p1 > 0.7509
        + 0.000813685 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 0.3201347 - Q.D3) / 0.004766833   # +0.1%  z_dr_0p05_0p1 < 0.5883 and D3 < 0.3201
        - 0.0008117289 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.3724174   # -0.1%  z_dr_0p05_0p1 < 0.5883
        - 0.0007995191 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # -0.1%  sd_rg > 0.2788
        + 0.0006847476 * max(0.0, 74.57663 - Q.sd_mass) * max(0.0, Q.sj2_mass2 - 3.697444) / 7.806142   # +0.1%  sd_mass < 74.58 and sj2_mass2 > 3.697
        + 0.0006693841 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # +0.1%  lam2 < 0.0005373
        - 0.0006648954 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, Q.mean_eta - -0.006779839) / 0.000320267   # -0.1%  planar_flow < 0.1115 and mean_eta > -0.00678
        - 0.000619114 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, Q.zdr_5 - 0.0155217) / 1.782544e-06   # -0.1%  z_dr_0p05_0p1 > 0.7509 and zdr_5 > 0.01552
        - 0.0005414782 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.001964442 - Q.zdr_7) / 5.711896e-06   # -0.1%  planar_flow < 0.1115 and zdr_7 < 0.001964
        - 0.0003906038 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.02358801 - Q.dr_3) / 2.035984e-05   # -0.0%  sj2_dr > 0.1592 and dr_3 < 0.02359
        - 0.0002842045 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, -0.05200653 - Q.phi_5) / 1.213647e-07   # -0.0%  e3 < 5.335e-05 and phi_5 < -0.05201
        - 0.0002290915 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, -0.08051147 - Q.phi_7) / 0.0002204999   # -0.0%  planar_flow < 0.1115 and phi_7 < -0.08051
        - 0.0001893818 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 2.468971 - Q.D3) / 0.0137974   # -0.0%  z_dr_0p05_0p1 > 0.7509 and D3 < 2.469
        - 0.0001725171 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.1058993 - Q.z_5) / 0.0007616364   # -0.0%  sj2_dr > 0.2002 and z_5 < 0.1059
        - 0.0001702964 * max(0.0, Q.mass_top5 - 49.18618) / 2.747087   # -0.0%  mass_top5 > 49.19
        - 0.000147608 * max(0.0, Q.sj3_pairmax_over_m - 0.9367476) / 0.002001707   # -0.0%  sj3_pairmax_over_m > 0.9367
        - 0.0001447658 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, 0.05268713 - Q.dr_3) / 7.198945e-06   # -0.0%  centroid_offset > 0.02686 and dr_3 < 0.05269
        - 0.0001059296 * max(0.0, Q.mass - 69.61135) * max(0.0, 0.003768113 - Q.dr0_7) / 6.828687e-05   # -0.0%  mass > 69.61 and dr0_7 < 0.003768
        + 0.0001057161 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.05268713 - Q.dr_3) / 0.0001291671   # +0.0%  sj2_dr > 0.2002 and dr_3 < 0.05269
        - 0.0001051481 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, Q.sj3_mass3 - 0.9601884) / 0.001720232   # -0.0%  planar_flow < 0.1115 and sj3_mass3 > 0.9602
        + 9.337948e-05 * max(0.0, Q.psi_0p1 - 0.8155839) * max(0.0, Q.phi_4 - 0.1096802) / 6.769968e-05   # +0.0%  psi_0p1 > 0.8156 and phi_4 > 0.1097
        - 4.396411e-05 * max(0.0, Q.mass_top5 - 49.18618) * max(0.0, Q.sd_nremoved - 1.0) / 0.1407813   # -0.0%  mass_top5 > 49.19 and sd_nremoved > 1
        + 3.639043e-05 * max(0.0, 49.91626 - Q.sd_mass) * max(0.0, Q.ptdr0_4 - 12.44629) / 0.8990836   # +0.0%  sd_mass < 49.92 and ptdr0_4 > 12.45
        + 2.571364e-05 * max(0.0, Q.sj3_pairmax_over_m - 0.9367476) * max(0.0, 0.01000023 - Q.dr0_4) / 2.886002e-06   # +0.0%  sj3_pairmax_over_m > 0.9367 and dr0_4 < 0.01
        - 1.954911e-05 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, Q.dr_4 - 0.05563519) / 0.001203856   # -0.0%  planar_flow < 0.1115 and dr_4 > 0.05564
        + 1.780745e-05 * max(0.0, 0.2568137 - Q.D2) / 0.0023137   # +0.0%  D2 < 0.2568
        + 1.658214e-05 * max(0.0, Q.psi_0p1 - 0.9761279) * max(0.0, Q.dr02 - 0.175862) / 3.966889e-07   # +0.0%  psi_0p1 > 0.9761 and dr02 > 0.1759
        - 9.914622e-06 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.n_dr_0p2_0p4 - 2.0) / 1.484336e-07   # -0.0%  e3 < 5.335e-05 and n_dr_0p2_0p4 > 2
        + 2.17809e-08 * max(0.0, Q.girth - 0.02689598) * max(0.0, Q.D2_b2 - 4.721224) / 0.0001187831   # +0.0%  girth > 0.0269 and D2_b2 > 4.721
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 30.46;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 30.45656 * (-0.006254851
        + 0.1404729 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +14.0%  width < 0.01324
        + 0.1132763 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +11.3%  mass_over_sum_pt_sq < 0.007183
        - 0.1026371 * max(0.0, 0.007520088 - Q.girth2) / 0.00354499   # -10.3%  girth2 < 0.00752
        - 0.09858512 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -9.9%  e2_sq < 0.008169
        + 0.08545578 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +8.5%  sj2_dr < 0.1592
        + 0.06515241 * max(0.0, 0.01882765 - Q.width) / 0.01314296   # +6.5%  width < 0.01883
        - 0.05968138 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # -6.0%  mass_over_sum_pt < 0.1309
        - 0.05861264 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # -5.9%  e2_sq < 0.01166
        + 0.0462506 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +4.6%  e2 < 0.04111
        - 0.04542811 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # -4.5%  girth < 0.1019
        - 0.04194028 * max(0.0, 0.2001708 - Q.sj2_dr) / 0.07331403   # -4.2%  sj2_dr < 0.2002
        - 0.02803671 * max(0.0, 0.1492731 - Q.sj2_dr) / 0.04374278   # -2.8%  sj2_dr < 0.1493
        - 0.02039247 * max(0.0, 0.00609665 - Q.width) / 0.002544039   # -2.0%  width < 0.006097
        + 0.01866683 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +1.9%  lam1 < 0.00484
        - 0.01372434 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # -1.4%  girth2_top3 < 0.002152
        - 0.01097224 * max(0.0, 0.007520088 - Q.girth2) * max(0.0, 0.7459513 - Q.D2) / 9.127003e-05   # -1.1%  girth2 < 0.00752 and D2 < 0.746
        + 0.01030085 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +1.0%  lam2 < 0.001131
        - 0.008562235 * max(0.0, 0.0005124533 - Q.girth2_top2) / 0.0001104529   # -0.9%  girth2_top2 < 0.0005125
        + 0.007491972 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2) / 0.003993721   # +0.7%  mass_over_sum_pt < 0.1309 and D2 < 0.746
        + 0.006620196 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +0.7%  N2 < 0.2233 and sum_pt_top5 > 430.8
        - 0.003685946 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.0114178   # -0.4%  N2 < 0.2233 and z_dr_0p05_0p1 < 0.5883
        - 0.003069032 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.121681 - Q.max_dr) / 0.0004905272   # -0.3%  N2 < 0.2233 and max_dr < 0.1217
        + 0.002678251 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.1094252   # +0.3%  N2 < 0.2233 and n_dr_0p1_0p2 < 4
        - 0.002670381 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # -0.3%  lam1 < 0.002464
        + 0.001780293 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 716.8828 - Q.sum_pt_top5) / 7.315623   # +0.2%  N2 < 0.2233 and sum_pt_top5 < 716.9
        + 0.001129461 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +0.1%  N2 < 0.2233
        + 0.00107199 * max(0.0, 0.00609665 - Q.width) * max(0.0, 0.7459513 - Q.D2) / 3.18925e-05   # +0.1%  width < 0.006097 and D2 < 0.746
        - 0.000492225 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sj3_pairmax_over_m - 0.9367476) / 0.0001955197   # -0.0%  N2 < 0.2233 and sj3_pairmax_over_m > 0.9367
        + 0.0004158613 * max(0.0, 0.03043859 - Q.sj3_pairmin_over_m) / 0.0003673495   # +0.0%  sj3_pairmin_over_m < 0.03044
        - 0.0003946712 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 1.54073 - Q.pt_entropy) / 0.000718138   # -0.0%  N2 < 0.2233 and pt_entropy < 1.541
        + 0.0002349983 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.4684459) / 0.0001955483   # +0.0%  mass_over_sum_pt < 0.1309 and z_dr_0p1_0p2 > 0.4684
        + 0.000116478 * max(0.0, Q.sj3_pair_mass_max - 80.4) / 0.3254107   # +0.0%  sj3_pair_mass_max > 80.4
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [2.3597970588235295, 1.728651050420168, 2.948559138655462, 1.7275409663865546, 2.840041176470588, 4.367259663865546, 3.060834033613445, 3.758525, 0.5167050420168067, 3.3715773109243696, 4.182419117647059, 4.993022899159664, 0.12036176470588235, 4.90431974789916, 0.9121279411764706, 0.5110901260504201]
T = [4.132812658383666, 2.2966775505514705, 6.994886085215335, 5.18303376772584, 5.3095133469012605]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +31%, n5 -20%, n1 +16%, n9 +14%, n0 -9%, n6 +8% ...
            + 0.306561 * h[2] / H_AVG[2]
            - 0.1981365 * h[5] / H_AVG[5]
            + 0.1633886 * h[1] / H_AVG[1]
            + 0.1402168 * h[9] / H_AVG[9]
            - 0.08921728 * h[0] / H_AVG[0]
            + 0.08100506 * h[6] / H_AVG[6]
            - 0.02147479 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +37%, n10 -23%, n6 +17%, n4 -12%, n5 +9%, n8 +1% ...
            + 0.3727404 * h[9] / H_AVG[9]
            - 0.2276342 * h[10] / H_AVG[10]
            + 0.1665903 * h[6] / H_AVG[6]
            - 0.11593 * h[4] / H_AVG[4]
            + 0.08913541 * h[5] / H_AVG[5]
            + 0.01406121 * h[8] / H_AVG[8]
            + 0.01390841 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +27%, n6 -14%, n3 -12%, n7 +12%, n0 +12%, n14 -10% ...
            + 0.2676789 * h[11] / H_AVG[11]
            - 0.1367443 * h[6] / H_AVG[6]
            - 0.123486 * h[3] / H_AVG[3]
            + 0.1175398 * h[7] / H_AVG[7]
            + 0.1159676 * h[0] / H_AVG[0]
            - 0.09779944 * h[14] / H_AVG[14]
            - 0.05023305 * h[15] / H_AVG[15]
            + 0.04929816 * h[13] / H_AVG[13]
            - 0.01846724 * h[8] / H_AVG[8]
            - 0.01506269 * h[9] / H_AVG[9]
            - 0.007722834 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +34%, n6 -22%, n3 -19%, n14 +7%, n13 +5%, n4 +4% ...
            + 0.3399184 * h[7] / H_AVG[7]
            - 0.2214558 * h[6] / H_AVG[6]
            - 0.1874851 * h[3] / H_AVG[3]
            + 0.06599378 * h[14] / H_AVG[14]
            + 0.05174672 * h[13] / H_AVG[13]
            + 0.04280856 * h[4] / H_AVG[4]
            + 0.04169014 * h[1] / H_AVG[1]
            - 0.02032821 * h[9] / H_AVG[9]
            - 0.01540755 * h[15] / H_AVG[15]
            + 0.01316573 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -38%, n10 +30%, n5 -21%, n4 +7%, n3 +2%, n8 +2% ...
            - 0.3752472 * h[13] / H_AVG[13]
            + 0.2953957 * h[10] / H_AVG[10]
            - 0.2056337 * h[5] / H_AVG[5]
            + 0.06686209 * h[4] / H_AVG[4]
            + 0.02033544 * h[3] / H_AVG[3]
            + 0.01824691 * h[8] / H_AVG[8]
            - 0.01133454 * h[12] / H_AVG[12]
            + 0.006944484 * h[0] / H_AVG[0]
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
