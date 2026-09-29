"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  12.8%   (on for 49% of jets)
  neuron  1:  11.9%   (on for 93% of jets)
  neuron  5:  11.3%   (on for 78% of jets)
  neuron  4:  10.1%   (on for 67% of jets)
  neuron  0:   7.7%   (on for 58% of jets)
  neuron 13:   7.4%   (on for 87% of jets)
  neuron  9:   6.8%   (on for 64% of jets)
  neuron 10:   6.8%   (on for 78% of jets)
  neuron 12:   4.9%   (on for 55% of jets)
  neuron  7:   4.5%   (on for 31% of jets)
  neuron  6:   4.0%   (on for 60% of jets)
  neuron 11:   3.4%   (on for 69% of jets)
  neuron  3:   3.3%   (on for 51% of jets)
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
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_pt               pT [GeV] of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_pt               pT [GeV] of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_z                pT share of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_12                 pT share × ΔR of particle 12 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_13              |Δφ| of particle 13
  Q.absphi_2               |Δφ| of particle 2
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft10_dr0             ΔR between the hardest and the 10. softest real particle (0 if among the 15 hardest)
  Q.soft3_dr0              ΔR between the hardest and the 3. softest real particle (0 if among the 15 hardest)
  Q.soft4_dr0              ΔR between the hardest and the 4. softest real particle (0 if among the 15 hardest)
  Q.soft8_dr0              ΔR between the hardest and the 8. softest real particle (0 if among the 15 hardest)
  Q.dr0_8                  ΔR between particle 8 and the hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_12                  ΔR of particle 12 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_9                   ΔR of particle 9 from the jet axis
  Q.soft10_dr              ΔR from the jet axis of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_dr               ΔR from the jet axis of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_dr               ΔR from the jet axis of the 4. softest real particle (0 if it is among the 15 hardest)
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
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.z_dr_0p4_up            pT share of the particles with 0.4 ≤ ΔR < 10
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
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
  Q.tau43                  N-subjettiness τ4/τ3
  Q.centroid_offset        distance of the pT centroid from the jet axis
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
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_7=pt[7],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft2_pt=softp(2, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        soft7_pt=softp(7, 'pt'),
        z_11=z[11],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        z_9=z[9],
        soft1_z=softp(1, 'z'),
        soft3_z=softp(3, 'z'),
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft7_z=softp(7, 'z'),
        soft8_z=softp(8, 'z'),
        z_top10_slots=sum(pt[:10]) / tot,
        z_top2_slots=sum(pt[:2]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_12=z[12] * dr[12],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_5=z[5] * dr[5],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        absphi_0=abs(phi[0]),
        absphi_13=abs(phi[13]),
        absphi_2=abs(phi[2]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft10_dr0=softp(10, 'dr0'),
        soft3_dr0=softp(3, 'dr0'),
        soft4_dr0=softp(4, 'dr0'),
        soft8_dr0=softp(8, 'dr0'),
        dr0_8=math.sqrt(dist2(0, 8)) if pt[8] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_12=dr[12] if pt[12] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_9=dr[9] if pt[9] > 0 else 0.0,
        soft10_dr=softp(10, 'dr'),
        soft3_dr=softp(3, 'dr'),
        soft4_dr=softp(4, 'dr'),
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
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        z_dr_0p4_up=sum(z[i] for i in real if 0.4 <= dr[i] < 10),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
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
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 21.26;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.25787 * (0.05871066
        - 0.2015016 * max(0.0, Q.mass - 92.85979) / 14.48273   # -20.2%  mass > 92.86
        - 0.1279401 * max(0.0, Q.mass - 78.26182) / 21.33658   # -12.8%  mass > 78.26
        + 0.1247826 * max(0.0, Q.mass - 91.03469) / 15.0694   # +12.5%  mass > 91.03
        - 0.05670624 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -5.7%  mass_top50 > 82.04
        - 0.04372538 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -4.4%  mass < 101
        + 0.04219751 * max(0.0, Q.mass_top50 - 92.16545) / 13.44564   # +4.2%  mass_top50 > 92.17
        + 0.03977121 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 0.002241068   # +4.0%  mass_over_sum_pt_sq < 0.007873
        + 0.02919727 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +2.9%  log_sum_pt < 7.017
        + 0.02778804 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq) / 0.001689525   # +2.8%  mass_over_sum_pt_sq < 0.006939
        - 0.02709632 * max(0.0, Q.mass - 74.25181) / 24.07648   # -2.7%  mass > 74.25
        + 0.02120836 * max(0.0, Q.mass_top50 - 71.79516) / 24.22501   # +2.1%  mass_top50 > 71.8
        - 0.02048151 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -2.0%  e2_sq < 0.006167
        - 0.02002444 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -2.0%  mass_top40 < 83.33
        + 0.0183845 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +1.8%  n_dr_0p2_0p4 < 15
        - 0.01635999 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -1.6%  girth2_top30 < 0.006364
        - 0.01481439 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # -1.5%  log_sum_pt < 6.989
        + 0.01456448 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +1.5%  mass_top40 < 80.89
        - 0.01450332 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # -1.5%  z_dr_0p2_0p4 < 0.09123
        - 0.0120501 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -1.2%  tau1 < 0.07057
        + 0.01121557 * max(0.0, 76.41544 - Q.mass_top30) / 12.67596   # +1.1%  mass_top30 < 76.42
        + 0.010642 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # +1.1%  e3 < 0.0005178
        - 0.008113884 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # -0.8%  girth2_top20 < 0.006374
        + 0.007538708 * max(0.0, 6.97212 - Q.log_sum_pt) / 0.05249646   # +0.8%  log_sum_pt < 6.972
        - 0.007421529 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -0.7%  sum_pt < 1013
        - 0.007376789 * max(0.0, Q.mass - 87.36377) / 16.57741   # -0.7%  mass > 87.36
        - 0.006878899 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # -0.7%  sum_pt_top40 < 1070
        + 0.006181277 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +0.6%  sum_pt_top50 < 1157
        + 0.005704642 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +0.6%  psi_0p3 > 0.9956
        + 0.00540714 * max(0.0, 0.007856958 - Q.girth2_top30) / 0.002601721   # +0.5%  girth2_top30 < 0.007857
        + 0.004718632 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957) / 1.852228   # +0.5%  log_sum_pt < 6.989 and sum_pt_top50 > 959.1
        + 0.004494969 * max(0.0, Q.sj2_dr - 0.1411617) / 0.06721719   # +0.4%  sj2_dr > 0.1412
        - 0.004098067 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # -0.4%  sj2_dr > 0.2232
        - 0.004089028 * max(0.0, 80.24626 - Q.mass_top30) / 14.57104   # -0.4%  mass_top30 < 80.25
        - 0.003704189 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -0.4%  lam1 < 0.005914
        - 0.003121378 * max(0.0, 0.9906378 - Q.z_top50_slots) / 0.004783315   # -0.3%  z_top50_slots < 0.9906
        - 0.003061341 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -0.3%  n_dr_0p2_0p4 < 11
        - 0.002919245 * max(0.0, 0.05602756 - Q.C2) / 0.009882647   # -0.3%  C2 < 0.05603
        + 0.002907454 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +0.3%  e2 < 0.01879
        - 0.002602515 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -0.3%  sum_pt < 972
        + 0.001965093 * max(0.0, 0.004754444 - Q.mass_over_sum_pt_sq) / 0.0007997283   # +0.2%  mass_over_sum_pt_sq < 0.004754
        + 0.001961191 * max(0.0, 950.9324 - Q.sum_pt_top30) / 25.09781   # +0.2%  sum_pt_top30 < 950.9
        + 0.001957124 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +0.2%  lam2 < 0.0006155
        + 0.001726124 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) * max(0.0, 118.5 - Q.pt_2) / 1.543055   # +0.2%  z_dr_0p2_0p4 < 0.09123 and pt_2 < 118.5
        + 0.001548955 * max(0.0, 53.87362 - Q.mass) / 3.56547   # +0.2%  mass < 53.87
        + 0.001418744 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) * max(0.0, Q.z_dr_0p05_0p1 - 0.01490648) / 0.0002465915   # +0.1%  mass_over_sum_pt_sq < 0.007873 and z_dr_0p05_0p1 > 0.01491
        - 0.001035837 * max(0.0, 0.00616708 - Q.e2_sq) * max(0.0, Q.z_dr_0p05_0p1 - 0.02069735) / 6.532427e-05   # -0.1%  e2_sq < 0.006167 and z_dr_0p05_0p1 > 0.0207
        - 0.0008782859 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # -0.1%  e2 > 0.05557
        + 0.0006600464 * max(0.0, Q.mass - 78.26182) * max(0.0, 0.06909411 - Q.sj2_zsoft) / 0.005641476   # +0.1%  mass > 78.26 and sj2_zsoft < 0.06909
        + 0.0006440741 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, Q.dr_9 - 0.04680031) / 1.099591   # +0.1%  sum_pt < 1013 and dr_9 > 0.0468
        - 0.0005771785 * max(0.0, Q.mass_top50 - 71.79516) * max(0.0, 0.06909411 - Q.sj2_zsoft) / 0.007761882   # -0.1%  mass_top50 > 71.8 and sj2_zsoft < 0.06909
        + 0.0003327008 * max(0.0, 858.8262 - Q.sum_pt_top40) / 3.625959   # +0.0%  sum_pt_top40 < 858.8
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 20.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.2325 * (0.1233632
        + 0.0838188 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # +8.4%  log_sum_pt > 6.894
        - 0.07960023 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -8.0%  log_sum_pt < 7.139
        + 0.0618613 * max(0.0, 1.521582 - Q.soft1_pt) / 0.9527349   # +6.2%  soft1_pt < 1.522
        + 0.04474383 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +4.5%  log_sum_pt > 6.91
        - 0.04412678 * max(0.0, 2.275391 - Q.soft1_pt) / 1.652278   # -4.4%  soft1_pt < 2.275
        - 0.04378419 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -4.4%  mass < 101
        - 0.04264425 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -4.3%  sum_pt_top50 > 959.1
        + 0.03979381 * max(0.0, 0.1751567 - Q.tau1) / 0.09101058   # +4.0%  tau1 < 0.1752
        - 0.03636884 * max(0.0, Q.z_top50_slots - 0.9586536) / 0.03427112   # -3.6%  z_top50_slots > 0.9587
        + 0.02997624 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.2734   # +3.0%  sum_pt_top2 < 689.2
        + 0.02627233 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # +2.6%  sum_pt_top40 < 1070
        + 0.02619699 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +2.6%  mass < 89.74
        - 0.02580537 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -2.6%  log_sum_pt > 6.959
        - 0.02490169 * max(0.0, 121.3913 - Q.mass) / 39.46288   # -2.5%  mass < 121.4
        + 0.01846172 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +1.8%  n_particles > 38
        - 0.01590954 * max(0.0, 47.88842 - Q.mass_top20) / 6.040308   # -1.6%  mass_top20 < 47.89
        - 0.01578349 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt) / 15.05788   # -1.6%  n_particles > 38 and soft1_pt < 2.275
        - 0.01449323 * max(0.0, 1.091797 - Q.soft1_pt) / 0.5750351   # -1.4%  soft1_pt < 1.092
        - 0.01432998 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # -1.4%  sj3_mass1 < 32.5
        + 0.01323822 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +1.3%  girth2_top5 < 0.00833
        - 0.0127805 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2) / 190.5117   # -1.3%  sj3_mass1 < 32.5 and sj3_mass2 < 18.68
        + 0.01132628 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +1.1%  sum_pt < 1017
        + 0.01049522 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4) / 0.08975043   # +1.0%  z_dr_0p2_0p4 < 0.1292
        + 0.00964983 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2) / 0.0008500795   # +1.0%  z_top30_slots > 0.9342 and C2 < 0.0728
        - 0.008991876 * max(0.0, 68.43422 - Q.mass_top5) / 42.12422   # -0.9%  mass_top5 < 68.43
        - 0.008653264 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.932092   # -0.9%  n_dr_0p2_0p4 < 13
        + 0.008616649 * max(0.0, 80.24626 - Q.mass_top30) / 14.57104   # +0.9%  mass_top30 < 80.25
        + 0.008591261 * max(0.0, Q.z_top30_slots - 0.9564984) / 0.01947087   # +0.9%  z_top30_slots > 0.9565
        - 0.008425432 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # -0.8%  log_sum_pt > 7.017
        + 0.008100882 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 39.23115   # +0.8%  mass_top20 < 47.89 and n_real_top40 > 29
        - 0.007992623 * max(0.0, 30.26161 - Q.sj2_mass1) / 7.997784   # -0.8%  sj2_mass1 < 30.26
        + 0.007568922 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.01395978 - Q.zdr_2) / 0.09391572   # +0.8%  n_particles > 38 and zdr_2 < 0.01396
        + 0.007526809 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # +0.8%  tau2 < 0.0795
        + 0.007486408 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1) / 1.005598   # +0.7%  n_particles > 38 and dr_1 < 0.1612
        - 0.00721737 * max(0.0, 33.78125 - Q.pt_11) / 12.89848   # -0.7%  pt_11 < 33.78
        - 0.007169047 * max(0.0, 50.3522 - Q.m012) / 35.90052   # -0.7%  m012 < 50.35
        + 0.00675896 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.7%  lam1 < 0.004673
        - 0.006715249 * max(0.0, Q.z_top30_slots - 0.9341838) / 0.03406818   # -0.7%  z_top30_slots > 0.9342
        + 0.006505994 * max(0.0, 0.03209934 - Q.tau3) / 0.01307604   # +0.7%  tau3 < 0.0321
        - 0.006168809 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -0.6%  n_dr_0p2_0p4 < 7
        + 0.005829831 * max(0.0, 0.00100099 - Q.girth2_top5) / 0.0002508867   # +0.6%  girth2_top5 < 0.001001
        - 0.005586047 * max(0.0, Q.z_dr_0_0p05 - 0.8459004) / 0.02484119   # -0.6%  z_dr_0_0p05 > 0.8459
        - 0.005532823 * max(0.0, 31.35938 - Q.pt_9) / 6.538304   # -0.6%  pt_9 < 31.36
        - 0.00526466 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.5%  psi_0p3 > 0.998
        + 0.005248147 * max(0.0, Q.mass_top10 - 31.33272) / 22.02168   # +0.5%  mass_top10 > 31.33
        - 0.005010474 * max(0.0, Q.zdr_0 - 0.001901263) / 0.007974889   # -0.5%  zdr_0 > 0.001901
        - 0.004927388 * max(0.0, Q.sum_pt_top3 - 353.0625) / 132.9972   # -0.5%  sum_pt_top3 > 353.1
        + 0.004642857 * max(0.0, 0.001823079 - Q.girth2_top20) / 0.000269929   # +0.5%  girth2_top20 < 0.001823
        - 0.004636118 * max(0.0, Q.eccentricity - 0.6414784) / 0.1723183   # -0.5%  eccentricity > 0.6415
        - 0.004245501 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, 7.880676 - Q.sj3_mass3) / 0.1008495   # -0.4%  z_top30_slots > 0.9565 and sj3_mass3 < 7.881
        - 0.004029348 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # -0.4%  M3 < 0.03188
        - 0.003849694 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -0.4%  log_sum_pt > 6.989
        + 0.003492782 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 0.3343369 - Q.soft4_dr0) / 2.709945   # +0.3%  sj3_mass1 < 32.5 and soft4_dr0 < 0.3343
        + 0.003449351 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0) / 0.570275   # +0.3%  n_particles > 38 and dr_0 < 0.112
        + 0.003268137 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793) / 0.1941151   # +0.3%  z_top30_slots > 0.9342 and max_pair_mass > 13.05
        + 0.003194825 * max(0.0, 10.0 - Q.n_dr_0_0p05) / 2.509415   # +0.3%  n_dr_0_0p05 < 10
        + 0.003049938 * max(0.0, 0.001594761 - Q.soft5_z) / 0.0005119901   # +0.3%  soft5_z < 0.001595
        + 0.003006938 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.0174229 - Q.mean_phi2) / 0.1185516   # +0.3%  n_particles > 38 and mean_phi2 < 0.01742
        - 0.002975062 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 0.2792343 - Q.soft3_dr) / 33.80538   # -0.3%  sum_pt_top2 < 689.2 and soft3_dr < 0.2792
        - 0.002947166 * max(0.0, 1.521582 - Q.soft1_pt) * max(0.0, 0.2664362 - Q.soft4_dr) / 0.09507844   # -0.3%  soft1_pt < 1.522 and soft4_dr < 0.2664
        + 0.002649491 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 0.3425105 - Q.soft3_dr0) / 46.70681   # +0.3%  sum_pt_top2 < 689.2 and soft3_dr0 < 0.3425
        - 0.002613374 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.3%  n_dr_0p1_0p2 < 8
        + 0.002375257 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 15.47191 - Q.pair_mass_0_12) / 61.85839   # +0.2%  pt_9 < 31.36 and pair_mass_0_12 < 15.47
        + 0.002370429 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.2898296 - Q.sj3_dr12) / 1.258789   # +0.2%  n_particles > 38 and sj3_dr12 < 0.2898
        + 0.002344568 * max(0.0, 0.08822608 - Q.D3) / 0.02694065   # +0.2%  D3 < 0.08823
        + 0.002264147 * max(0.0, 0.0007894752 - Q.girth2_top15) / 9.062109e-05   # +0.2%  girth2_top15 < 0.0007895
        + 0.002187671 * max(0.0, Q.psi_0p1 - 0.8747961) / 0.03440015   # +0.2%  psi_0p1 > 0.8748
        + 0.002124213 * max(0.0, 0.3100459 - Q.tau21) / 0.03839468   # +0.2%  tau21 < 0.31
        + 0.001983335 * max(0.0, 48.60659 - Q.mass_top40) / 2.938264   # +0.2%  mass_top40 < 48.61
        - 0.001957848 * max(0.0, 0.03107249 - Q.z_7) / 0.003035085   # -0.2%  z_7 < 0.03107
        - 0.001943961 * max(0.0, 0.03187688 - Q.M3) * max(0.0, Q.M2 - 0.05568888) / 0.0001450223   # -0.2%  M3 < 0.03188 and M2 > 0.05569
        - 0.001860435 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 0.9624339 - Q.tau43) / 39.08134   # -0.2%  sum_pt_top2 < 689.2 and tau43 < 0.9624
        + 0.001856235 * max(0.0, 0.003270031 - Q.girth2_top15) / 0.0007969965   # +0.2%  girth2_top15 < 0.00327
        + 0.001788568 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1546554 - Q.dr_9) / 0.7931187   # +0.2%  n_particles > 38 and dr_9 < 0.1547
        - 0.001688347 * max(0.0, 14.54404 - Q.mass_top5) / 4.336972   # -0.2%  mass_top5 < 14.54
        + 0.001687772 * max(0.0, Q.sj2_zsoft - 0.2912328) / 0.04122672   # +0.2%  sj2_zsoft > 0.2912
        - 0.001670403 * max(0.0, 0.003468228 - Q.C2_b2) / 0.0005968599   # -0.2%  C2_b2 < 0.003468
        + 0.001637278 * max(0.0, Q.sum_pt_top30 - 1191.938) / 6.938323   # +0.2%  sum_pt_top30 > 1192
        + 0.001390625 * max(0.0, 0.003270031 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 6.08612e-07   # +0.1%  girth2_top15 < 0.00327 and psi_0p3 > 0.9974
        + 0.001383334 * max(0.0, Q.n_particles - 38.0) * max(0.0, 57.87349 - Q.mass_top15) / 116.4836   # +0.1%  n_particles > 38 and mass_top15 < 57.87
        + 0.001372539 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, 30.0 - Q.n_real_top30) / 3.822398   # +0.1%  n_dr_0p2_0p4 < 7 and n_real_top30 < 30
        + 0.00121286 * max(0.0, Q.sj3_dr_max - 0.3483732) / 0.01384956   # +0.1%  sj3_dr_max > 0.3484
        + 0.001065099 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874) / 0.05244257   # +0.1%  z_top30_slots > 0.9342 and ptdr0_3 > 7.408
        + 0.0008590972 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, Q.ptdr0_4 - 4.470953) / 0.03202477   # +0.1%  z_top30_slots > 0.9565 and ptdr0_4 > 4.471
        - 0.0006360764 * max(0.0, Q.M3 - 0.04813493) / 0.001348782   # -0.1%  M3 > 0.04813
        + 0.0006023559 * max(0.0, Q.sd_rg - 0.3017146) / 0.004328764   # +0.1%  sd_rg > 0.3017
        - 0.0005541296 * max(0.0, Q.girth2_top5 - 0.0168592) / 0.0008909511   # -0.1%  girth2_top5 > 0.01686
        - 0.0004411918 * max(0.0, Q.N2 - 0.4678622) / 0.001556028   # -0.0%  N2 > 0.4679
        + 0.0004400675 * max(0.0, 0.0101072 - Q.girth) / 0.0001148761   # +0.0%  girth < 0.01011
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 6.749;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.748639 * (0.008334194
        - 0.1592741 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -15.9%  mass < 92.86
        + 0.08140263 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15) / 0.0009227558   # +8.1%  log_sum_pt > 6.903 and girth2_top15 < 0.02147
        - 0.0791359 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # -7.9%  sum_pt < 1261
        + 0.07412614 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # +7.4%  sum_pt > 1017
        + 0.07337698 * max(0.0, 143.7876 - Q.mass) / 57.99337   # +7.3%  mass < 143.8
        - 0.0709232 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15) / 1.045047   # -7.1%  sum_pt_top50 > 988.5 and girth2_top15 < 0.02147
        + 0.0556894 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +5.6%  mass < 121.4
        + 0.04850789 * max(0.0, 91.03469 - Q.mass) / 16.36287   # +4.9%  mass < 91.03
        - 0.03686876 * max(0.0, 132.4278 - Q.mass_top40) / 51.50818   # -3.7%  mass_top40 < 132.4
        - 0.0324827 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -3.2%  sum_pt > 1053
        + 0.029546 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +3.0%  sum_pt_top40 < 1053
        + 0.02918118 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # +2.9%  girth2_top50 < 0.008125
        - 0.02131124 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003463934   # -2.1%  log_sum_pt > 6.903 and psi_0p3 > 0.9299
        + 0.0205363 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +2.1%  mass_over_sum_pt < 0.09795
        - 0.01951839 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # -2.0%  lam2 < 0.001776
        + 0.0190903 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # +1.9%  mass_top40 < 83.33
        - 0.01646208 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -1.6%  lam1 < 0.006717
        + 0.0162548 * max(0.0, Q.z_top20_slots - 0.7516206) / 0.1429779   # +1.6%  z_top20_slots > 0.7516
        + 0.01588378 * max(0.0, Q.sum_pt_top50 - 988.4554) / 62.81258   # +1.6%  sum_pt_top50 > 988.5
        - 0.01564249 * max(0.0, 51.0 - Q.n_particles) / 9.158477   # -1.6%  n_particles < 51
        - 0.01529968 * max(0.0, 1003.329 - Q.sum_pt_top15) / 147.1939   # -1.5%  sum_pt_top15 < 1003
        + 0.01234799 * max(0.0, 787.6281 - Q.sum_pt_top3) / 328.0371   # +1.2%  sum_pt_top3 < 787.6
        + 0.011982 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +1.2%  mass < 78.26
        - 0.01197109 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # -1.2%  sum_pt > 1116
        + 0.0086479 * max(0.0, Q.sum_pt_top50 - 1156.659) / 13.71464   # +0.9%  sum_pt_top50 > 1157
        - 0.008421032 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.8%  log_sum_pt > 7.063
        - 0.006462045 * max(0.0, 996.8867 - Q.sum_pt_top30) / 42.94346   # -0.6%  sum_pt_top30 < 996.9
        - 0.003917966 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.mass_top40 - 77.93668) / 1.109603   # -0.4%  log_sum_pt > 6.903 and mass_top40 > 77.94
        - 0.003440332 * max(0.0, 0.006088416 - Q.girth2_top30) / 0.001533892   # -0.3%  girth2_top30 < 0.006088
        + 0.002295763 * max(0.0, Q.sum_pt_top20 - 1129.275) / 6.499168   # +0.2%  sum_pt_top20 > 1129
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 3.289;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.288879 * (-0.06646313
        - 0.1381353 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # -13.8%  girth2 < 0.009615
        + 0.09702696 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +9.7%  mass < 92.86
        - 0.07497761 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # -7.5%  mass_top40 < 80.89
        + 0.07212935 * max(0.0, 0.08665515 - Q.mass_over_sum_pt) / 0.01542259   # +7.2%  mass_over_sum_pt < 0.08666
        + 0.05898905 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +5.9%  girth2_top40 < 0.008841
        - 0.05829159 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # -5.8%  girth2_top40 < 0.00626
        + 0.0495051 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741) / 0.0491674   # +5.0%  n_dr_0p2_0p4 < 8 and z_top50_slots > 0.9787
        + 0.04697628 * max(0.0, 0.006026828 - Q.girth2_top40) / 0.001351778   # +4.7%  girth2_top40 < 0.006027
        + 0.04419403 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +4.4%  n_particles < 46
        + 0.03343506 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # +3.3%  girth2_top50 < 0.008125
        - 0.02927612 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -2.9%  tau21 < 0.3862
        + 0.02811371 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.psi_0p3 - 0.9924477) / 0.007646232   # +2.8%  n_dr_0p2_0p4 < 5 and psi_0p3 > 0.9924
        - 0.02635248 * max(0.0, 79.21004 - Q.mass_top50) / 10.65754   # -2.6%  mass_top50 < 79.21
        + 0.023933 * max(0.0, 1.36316 - Q.D2_b2) / 0.3773161   # +2.4%  D2_b2 < 1.363
        + 0.02347243 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +2.3%  lam2 < 0.0006155
        - 0.01806113 * max(0.0, Q.z_top30_slots - 0.9734886) / 0.01006171   # -1.8%  z_top30_slots > 0.9735
        - 0.01669317 * max(0.0, 2.410481 - Q.D2) / 0.587301   # -1.7%  D2 < 2.41
        - 0.01614623 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -1.6%  mass < 53.87
        + 0.01584797 * max(0.0, 0.009614971 - Q.girth2) * max(0.0, 26.76006 - Q.sj3_mass1) / 0.04798748   # +1.6%  girth2 < 0.009615 and sj3_mass1 < 26.76
        + 0.01453578 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732) / 1477.074   # +1.5%  n_particles < 46 and sum_pt_top30 > 800.7
        + 0.01400655 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1) / 0.1325294   # +1.4%  n_dr_0p2_0p4 < 5 and psi_0p1 < 0.9538
        - 0.01268161 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, 18.0 - Q.n_dr_0_0p05) / 19.28245   # -1.3%  n_dr_0p2_0p4 < 8 and n_dr_0_0p05 < 18
        + 0.01246213 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.9087063) / 0.1465289   # +1.2%  n_dr_0p1_0p2 < 10 and psi_0p2 > 0.9087
        - 0.01180958 * max(0.0, 0.019523 - Q.z_dr_0p2_0p4) / 0.007664401   # -1.2%  z_dr_0p2_0p4 < 0.01952
        - 0.01114039 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243) / 0.0001457263   # -1.1%  tau21 < 0.3862 and lam1 > 0.007671
        + 0.01076776 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421) / 0.000406438   # +1.1%  D2 < 2.41 and psi_0p3 > 0.9985
        - 0.009846089 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5) / 0.004484684   # -1.0%  n_dr_0p2_0p4 < 5 and zdr_5 < 0.007099
        + 0.009533717 * max(0.0, 0.001868041 - Q.lam1) / 0.000193543   # +1.0%  lam1 < 0.001868
        - 0.00833589 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0) / 3.186294   # -0.8%  n_dr_0p2_0p4 < 5 and n_dr_0p1_0p2 > 9
        + 0.007685071 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.eccentricity - 0.7333655) / 0.1710102   # +0.8%  n_dr_0p2_0p4 < 5 and eccentricity > 0.7334
        + 0.005638809 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p05_0p1 - 0.5106729) / 0.121515   # +0.6%  n_dr_0p2_0p4 < 5 and z_dr_0p05_0p1 > 0.5107
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 10.97;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.96924 * (0.01444682
        + 0.0617723 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.07479306   # +6.2%  mass < 101 and lam2 < 0.003688
        - 0.05923752 * max(0.0, 87.36377 - Q.mass) / 14.19996   # -5.9%  mass < 87.36
        + 0.05868124 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # +5.9%  mass_top40 < 83.33
        - 0.05610548 * max(0.0, Q.n_particles - 22.0) / 23.9717   # -5.6%  n_particles > 22
        - 0.05225508 * max(0.0, 79.65241 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.03386155   # -5.2%  mass < 79.65 and lam2 < 0.003688
        + 0.04973132 * max(0.0, 6.916121 - Q.D2) / 4.129327   # +5.0%  D2 < 6.916
        + 0.04610134 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +4.6%  mass < 101
        - 0.04550628 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # -4.6%  girth2_top10 < 0.007679
        + 0.04061386 * max(0.0, 0.008956554 - Q.girth2_top10) / 0.004473389   # +4.1%  girth2_top10 < 0.008957
        - 0.03706498 * max(0.0, 94.64253 - Q.mass_top40) / 21.02021   # -3.7%  mass_top40 < 94.64
        - 0.02796069 * max(0.0, 79.65241 - Q.mass) / 10.37103   # -2.8%  mass < 79.65
        + 0.02659896 * max(0.0, 0.008184368 - Q.mass_over_sum_pt_sq) / 0.002451608   # +2.7%  mass_over_sum_pt_sq < 0.008184
        - 0.02647553 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 6.916121 - Q.D2) / 32.02927   # -2.6%  mass_top40 < 83.33 and D2 < 6.916
        + 0.02503515 * max(0.0, Q.sj2_dr - 0.1512157) / 0.05988199   # +2.5%  sj2_dr > 0.1512
        + 0.02318928 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +2.3%  n_dr_0p2_0p4 < 15
        - 0.02218882 * max(0.0, 0.004855289 - Q.girth2_top15) / 0.001418414   # -2.2%  girth2_top15 < 0.004855
        - 0.02116824 * max(0.0, Q.sj2_dr - 0.1825048) / 0.04110551   # -2.1%  sj2_dr > 0.1825
        - 0.01855307 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -1.9%  mass < 92.86
        + 0.01460764 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +1.5%  mass < 121.4
        + 0.01428202 * max(0.0, Q.e2_sq - 0.01396296) / 0.002220692   # +1.4%  e2_sq > 0.01396
        - 0.01372869 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -1.4%  tau1 < 0.07057
        - 0.01266415 * max(0.0, 25.0 - Q.n_dr_0_0p05) / 13.07341   # -1.3%  n_dr_0_0p05 < 25
        + 0.01265152 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt) / 36.47388   # +1.3%  n_particles > 22 and soft1_pt < 2.275
        - 0.01252012 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # -1.3%  z_dr_0p2_0p4 < 0.06849
        + 0.01168585 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +1.2%  n_dr_0p2_0p4 < 26
        + 0.01131445 * max(0.0, 48.0 - Q.n_pt_above_1) / 9.959713   # +1.1%  n_pt_above_1 < 48
        - 0.01061301 * max(0.0, 1042.609 - Q.sum_pt) / 35.65948   # -1.1%  sum_pt < 1043
        - 0.01021333 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # -1.0%  lam2 < 0.001776
        - 0.00911368 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # -0.9%  mass_top40 < 67.73
        + 0.008573757 * max(0.0, Q.mass - 143.7876) / 3.946979   # +0.9%  mass > 143.8
        - 0.008217068 * max(0.0, 61.65972 - Q.mass_top50) / 5.455879   # -0.8%  mass_top50 < 61.66
        + 0.007686458 * max(0.0, 5.378975 - Q.D2) / 2.781156   # +0.8%  D2 < 5.379
        + 0.007657644 * Q.n_dr_0p05_0p1 / 11.55961   # +0.8%  n_dr_0p05_0p1
        - 0.007247025 * max(0.0, 0.005809485 - Q.girth2_top30) / 0.001402136   # -0.7%  girth2_top30 < 0.005809
        + 0.006778321 * max(0.0, Q.n_dr_0_0p05 - 5.0) / 8.271047   # +0.7%  n_dr_0_0p05 > 5
        + 0.006671128 * max(0.0, 69.02716 - Q.mass_top15) / 18.23029   # +0.7%  mass_top15 < 69.03
        + 0.006500285 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.7%  psi_0p3 > 0.9974
        + 0.005773972 * max(0.0, 57.87349 - Q.mass_top15) / 12.46085   # +0.6%  mass_top15 < 57.87
        + 0.005755549 * max(0.0, Q.n_dr_0_0p05 - 5.0) * max(0.0, 0.001163277 - Q.lam2) / 0.004222239   # +0.6%  n_dr_0_0p05 > 5 and lam2 < 0.001163
        - 0.005328421 * max(0.0, 121.737 - Q.mass_top30) / 46.42587   # -0.5%  mass_top30 < 121.7
        - 0.005253022 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # -0.5%  LHA > 0.372
        - 0.005123237 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # -0.5%  lam1 < 0.007671
        + 0.004898058 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, 1053.047 - Q.sum_pt_top40) / 2417.732   # +0.5%  mass_top30 < 121.7 and sum_pt_top40 < 1053
        - 0.004827491 * max(0.0, Q.mass_top50 - 138.8977) / 3.930268   # -0.5%  mass_top50 > 138.9
        + 0.004754519 * max(0.0, Q.e2 - 0.03680582) / 0.004603798   # +0.5%  e2 > 0.03681
        + 0.004685672 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.C2_b2 - 0.001126916) / 0.3455335   # +0.5%  n_particles > 22 and C2_b2 > 0.001127
        + 0.004475036 * max(0.0, 44.19225 - Q.mass_top10) / 11.22847   # +0.4%  mass_top10 < 44.19
        + 0.004457663 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +0.4%  e2 < 0.02516
        - 0.004437136 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.max_dr - 0.1939977) / 1.031549   # -0.4%  n_dr_0p2_0p4 < 15 and max_dr > 0.194
        - 0.004225439 * max(0.0, 0.02647293 - Q.z_dr_0p2_0p4) / 0.01168948   # -0.4%  z_dr_0p2_0p4 < 0.02647
        - 0.003965519 * max(0.0, 0.9978182 - Q.z_top50_slots) / 0.006875583   # -0.4%  z_top50_slots < 0.9978
        - 0.003678225 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # -0.4%  mass_top30 < 60.44
        - 0.003628391 * max(0.0, 0.4281458 - Q.tau21) / 0.08904986   # -0.4%  tau21 < 0.4281
        - 0.003606386 * max(0.0, Q.psi_0p1 - 0.8747961) / 0.03440015   # -0.4%  psi_0p1 > 0.8748
        - 0.003257628 * max(0.0, 74.25181 - Q.mass) / 8.587064   # -0.3%  mass < 74.25
        + 0.002935515 * max(0.0, Q.lam2 - 0.0004662705) / 0.001028518   # +0.3%  lam2 > 0.0004663
        - 0.002926092 * max(0.0, Q.mass_top20 - 103.6749) / 3.795511   # -0.3%  mass_top20 > 103.7
        - 0.00287799 * max(0.0, Q.girth2_top15 - 0.007887677) / 0.002755219   # -0.3%  girth2_top15 > 0.007888
        + 0.002825453 * max(0.0, 0.004855289 - Q.girth2_top15) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 0.007590625   # +0.3%  girth2_top15 < 0.004855 and n_dr_0p05_0p1 > 2
        + 0.002742594 * max(0.0, 0.7703036 - Q.psi_0p1) / 0.09235895   # +0.3%  psi_0p1 < 0.7703
        - 0.002531122 * max(0.0, Q.mass_top10 - 76.9886) / 2.774301   # -0.3%  mass_top10 > 76.99
        + 0.002450008 * max(0.0, Q.sj3_pair_mass_min - 32.51366) / 5.88267   # +0.2%  sj3_pair_mass_min > 32.51
        - 0.001794743 * max(0.0, 0.03577037 - Q.girth) / 0.004182456   # -0.2%  girth < 0.03577
        - 0.001654283 * max(0.0, Q.e3 - 6.567534e-05) / 6.326067e-05   # -0.2%  e3 > 6.568e-05
        - 0.001453775 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.zdr_0 - 0.006864207) / 0.008666302   # -0.1%  mass < 87.36 and zdr_0 > 0.006864
        + 0.001444351 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # +0.1%  sum_pt < 1013
        - 0.001267433 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.z_top50_slots) / 0.03204382   # -0.1%  n_dr_0p2_0p4 < 15 and z_top50_slots < 1
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 14.99;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.99403 * (0.06130822
        + 0.08484242 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +8.5%  sum_pt_top50 > 934.2
        + 0.0749912 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +7.5%  sum_pt > 907.9
        - 0.07495295 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # -7.5%  sum_pt > 986.1
        - 0.05241118 * max(0.0, 150.0144 - Q.mass_top40) / 66.86198   # -5.2%  mass_top40 < 150
        - 0.04585075 * max(0.0, Q.mass_over_sum_pt - 0.08873143) / 0.01477171   # -4.6%  mass_over_sum_pt > 0.08873
        - 0.04219371 * max(0.0, Q.mass - 91.03469) / 15.0694   # -4.2%  mass > 91.03
        - 0.0421918 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -4.2%  log_sum_pt > 6.91
        - 0.04004424 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -4.0%  log_sum_pt > 6.92
        + 0.03756123 * max(0.0, Q.mass_over_sum_pt - 0.09046749) / 0.01421039   # +3.8%  mass_over_sum_pt > 0.09047
        + 0.03700828 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # +3.7%  sum_pt_top50 > 959.1
        - 0.03325503 * max(0.0, Q.mass - 172.4888) / 0.7446233   # -3.3%  mass > 172.5
        + 0.02882246 * max(0.0, Q.mass - 87.36377) / 16.57741   # +2.9%  mass > 87.36
        + 0.02515044 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +2.5%  sum_pt > 907.9 and e4 < 5.851e-08
        + 0.0200438 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +2.0%  n_particles < 64
        - 0.01872877 * max(0.0, Q.girth2 - 0.004756928) / 0.005348122   # -1.9%  girth2 > 0.004757
        - 0.01839849 * max(0.0, Q.sum_pt_top40 - 994.2695) * max(0.0, 5.8505e-08 - Q.e4) / 2.919126e-06   # -1.8%  sum_pt_top40 > 994.3 and e4 < 5.851e-08
        + 0.0183957 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # +1.8%  e2 < 0.04083
        - 0.01787347 * max(0.0, 787.6281 - Q.sum_pt_top3) / 328.0371   # -1.8%  sum_pt_top3 < 787.6
        - 0.01633934 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -1.6%  mass_over_sum_pt_sq > 0.0292
        + 0.01536626 * max(0.0, Q.sum_pt_top40 - 994.2695) / 51.72055   # +1.5%  sum_pt_top40 > 994.3
        + 0.01527623 * max(0.0, Q.n_pt_above_1 - 28.0) / 14.80415   # +1.5%  n_pt_above_1 > 28
        - 0.01519564 * max(0.0, Q.n_real_top50 - 22.0) / 19.58685   # -1.5%  n_real_top50 > 22
        - 0.01341961 * max(0.0, Q.z_top30_slots - 0.9048492) / 0.05621641   # -1.3%  z_top30_slots > 0.9048
        + 0.0121124 * max(0.0, Q.girth2_top30 - 0.006363916) / 0.003802703   # +1.2%  girth2_top30 > 0.006364
        + 0.01173285 * max(0.0, Q.mass - 101.0497) / 12.31084   # +1.2%  mass > 101
        - 0.01065664 * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 9.82015   # -1.1%  n_dr_0p1_0p2 < 21
        + 0.01064414 * max(0.0, Q.sd_mass - 69.65633) / 11.73272   # +1.1%  sd_mass > 69.66
        - 0.01049517 * max(0.0, 0.007856958 - Q.girth2_top30) / 0.002601721   # -1.0%  girth2_top30 < 0.007857
        + 0.009739417 * max(0.0, Q.log_sum_pt - 6.935549) / 0.0371538   # +1.0%  log_sum_pt > 6.936
        - 0.008788465 * max(0.0, Q.mass_top40 - 111.2487) / 7.565029   # -0.9%  mass_top40 > 111.2
        - 0.008319469 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2) / 10.81389   # -0.8%  n_particles < 64 and D2 < 2.179
        + 0.007168282 * max(0.0, Q.sum_pt_top10 - 822.975) / 44.35907   # +0.7%  sum_pt_top10 > 823
        + 0.007037365 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # +0.7%  log_sum_pt > 6.989
        + 0.006870645 * max(0.0, Q.psi_0p3 - 0.9638082) / 0.02792824   # +0.7%  psi_0p3 > 0.9638
        - 0.006721856 * max(0.0, 0.5100475 - Q.tau21) / 0.1343519   # -0.7%  tau21 < 0.51
        - 0.006496247 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # -0.6%  tau1 < 0.1507
        + 0.006152056 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +0.6%  n_dr_0p2_0p4 < 11
        + 0.006105478 * max(0.0, 64.48544 - Q.mass) / 5.935866   # +0.6%  mass < 64.49
        - 0.005976612 * max(0.0, Q.sum_pt_top15 - 935.1043) / 29.65775   # -0.6%  sum_pt_top15 > 935.1
        - 0.005950055 * max(0.0, Q.sd_mass - 83.30647) / 6.989667   # -0.6%  sd_mass > 83.31
        + 0.005392432 * max(0.0, Q.sum_pt_top30 - 933.1875) / 79.88556   # +0.5%  sum_pt_top30 > 933.2
        + 0.005008446 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2) / 0.001543279   # +0.5%  psi_0p3 > 0.9974 and D2 < 3.345
        - 0.004872263 * max(0.0, 0.03700182 - Q.z_dr_0p2_0p4) / 0.0183096   # -0.5%  z_dr_0p2_0p4 < 0.037
        + 0.00445931 * max(0.0, Q.soft3_pt - 0.3395996) / 0.709118   # +0.4%  soft3_pt > 0.3396
        - 0.004401174 * max(0.0, Q.girth2_top50 - 0.01951641) / 0.001208668   # -0.4%  girth2_top50 > 0.01952
        + 0.004192985 * max(0.0, 1.976207 - Q.D2) / 0.367701   # +0.4%  D2 < 1.976
        + 0.00379456 * max(0.0, 0.01375115 - Q.z_11) / 0.0009450582   # +0.4%  z_11 < 0.01375
        + 0.003694493 * max(0.0, 0.03787151 - Q.M3) / 0.01370579   # +0.4%  M3 < 0.03787
        - 0.003414934 * max(0.0, Q.mass - 172.4888) * max(0.0, 0.161871 - Q.soft3_dr) / 0.01314483   # -0.3%  mass > 172.5 and soft3_dr < 0.1619
        - 0.003325119 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -0.3%  mass_top50 > 157.5
        - 0.002864274 * max(0.0, Q.z_top10_slots - 0.7887522) / 0.03614094   # -0.3%  z_top10_slots > 0.7888
        + 0.002836042 * max(0.0, 23.31647 - Q.sj2_mass1) / 3.96589   # +0.3%  sj2_mass1 < 23.32
        - 0.002798071 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -0.3%  sum_pt_top40 > 1025
        - 0.002649147 * max(0.0, Q.soft3_z - 0.0008504382) / 0.0003473759   # -0.3%  soft3_z > 0.0008504
        - 0.00251284 * max(0.0, 14.14062 - Q.pt_11) / 0.9740783   # -0.3%  pt_11 < 14.14
        + 0.002381867 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # +0.2%  n_dr_0p1_0p2 < 8
        - 0.002041849 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.1958008 - Q.absphi_13) / 0.007588689   # -0.2%  log_sum_pt > 6.91 and absphi_13 < 0.1958
        + 0.001918957 * max(0.0, Q.M2 - 0.06621477) / 0.0101271   # +0.2%  M2 > 0.06621
        - 0.001647962 * max(0.0, 0.0007522186 - Q.girth2_top20) / 5.065357e-05   # -0.2%  girth2_top20 < 0.0007522
        - 0.001610228 * max(0.0, 0.707925 - Q.psi_0p1) / 0.07386333   # -0.2%  psi_0p1 < 0.7079
        + 0.001504343 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # +0.2%  mass_top10 > 71.78
        + 0.001189236 * max(0.0, Q.z_dr_0_0p05 - 0.9084912) / 0.009227347   # +0.1%  z_dr_0_0p05 > 0.9085
        - 0.001117379 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.1%  sum_pt_top10 > 943.7
        - 0.001091971 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.1269782 - Q.dr_12) / 0.003311185   # -0.1%  log_sum_pt > 6.91 and dr_12 < 0.127
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 16.85;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.84523 * (-0.02734342
        + 0.1949522 * max(0.0, 0.02580859 - Q.e2_sq) / 0.01698103   # +19.5%  e2_sq < 0.02581
        - 0.1262618 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -12.6%  mass < 101
        - 0.0829925 * max(0.0, 0.02550569 - Q.girth2_top50) / 0.01683664   # -8.3%  girth2_top50 < 0.02551
        - 0.06801489 * max(0.0, 121.3913 - Q.mass) / 39.46288   # -6.8%  mass < 121.4
        - 0.06622069 * max(0.0, 0.02412652 - Q.girth2_top30) / 0.01613825   # -6.6%  girth2_top30 < 0.02413
        + 0.06438031 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +6.4%  mass < 92.86
        - 0.04613172 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # -4.6%  mass_over_sum_pt < 0.09795
        + 0.04519977 * max(0.0, 172.4888 - Q.mass) / 83.49222   # +4.5%  mass < 172.5
        + 0.03162329 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +3.2%  mass < 89.74
        + 0.02834378 * max(0.0, 65.20727 - Q.sj2_mass1) / 36.50254   # +2.8%  sj2_mass1 < 65.21
        + 0.02050765 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # +2.1%  girth2_top30 < 0.008376
        + 0.02038612 * max(0.0, Q.tau1 - 0.05444509) / 0.03990678   # +2.0%  tau1 > 0.05445
        + 0.0200431 * max(0.0, 0.003687605 - Q.lam2) / 0.002601874   # +2.0%  lam2 < 0.003688
        - 0.01807319 * max(0.0, Q.e2 - 0.02793599) / 0.008664879   # -1.8%  e2 > 0.02794
        + 0.01747315 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +1.7%  lam1 < 0.007259
        + 0.01703511 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +1.7%  mass < 82.85
        + 0.01272417 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # +1.3%  mass_top50 < 71.8
        - 0.01168965 * max(0.0, 51.2028 - Q.sj2_mass1) / 24.19897   # -1.2%  sj2_mass1 < 51.2
        - 0.01115197 * max(0.0, 0.7058597 - Q.tau21_b2) / 0.3912084   # -1.1%  tau21_b2 < 0.7059
        + 0.007355723 * max(0.0, Q.e2 - 0.02793599) * max(0.0, Q.psi_0p3 - 0.9896594) / 4.018702e-05   # +0.7%  e2 > 0.02794 and psi_0p3 > 0.9897
        + 0.006931971 * max(0.0, Q.mass_top40 - 94.64253) / 11.19703   # +0.7%  mass_top40 > 94.64
        + 0.006539454 * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) / 0.159722   # +0.7%  z_dr_0p1_0p2 < 0.2865
        + 0.006216695 * max(0.0, 65.20727 - Q.sj2_mass1) * max(0.0, Q.eccentricity - 0.3346888) / 16.90037   # +0.6%  sj2_mass1 < 65.21 and eccentricity > 0.3347
        - 0.005519297 * max(0.0, 89.74183 - Q.mass) * max(0.0, 1129.275 - Q.sum_pt_top20) / 2511.831   # -0.6%  mass < 89.74 and sum_pt_top20 < 1129
        - 0.005214885 * Q.sj2_mass2 / 11.61619   # -0.5%  sj2_mass2
        + 0.004484069 * max(0.0, 121.3913 - Q.mass) * max(0.0, 1003.544 - Q.sum_pt_top50) / 688.1559   # +0.4%  mass < 121.4 and sum_pt_top50 < 1004
        - 0.00401968 * max(0.0, 0.001425993 - Q.girth2_top2) / 0.0004557017   # -0.4%  girth2_top2 < 0.001426
        - 0.003931485 * max(0.0, Q.mass - 143.7876) / 3.946979   # -0.4%  mass > 143.8
        - 0.003725132 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # -0.4%  psi_0p3 > 0.9924
        - 0.003489407 * max(0.0, 1.409617 - Q.D2) / 0.1519846   # -0.3%  D2 < 1.41
        + 0.003434884 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.3%  e2 > 0.04755
        + 0.003199404 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +0.3%  tau1 < 0.06311
        - 0.002784699 * max(0.0, 0.2126484 - Q.z_dr_0p05_0p1) * max(0.0, Q.sj3_dr13 - 0.06988208) / 0.01261869   # -0.3%  z_dr_0p05_0p1 < 0.2126 and sj3_dr13 > 0.06988
        - 0.002672164 * max(0.0, 2.178951 - Q.D2) / 0.4648731   # -0.3%  D2 < 2.179
        - 0.002255267 * max(0.0, Q.sj3_pair_mass_min - 32.51366) / 5.88267   # -0.2%  sj3_pair_mass_min > 32.51
        + 0.002141756 * max(0.0, 0.1722488 - Q.tau21_b2) / 0.02681502   # +0.2%  tau21_b2 < 0.1722
        + 0.002032526 * max(0.0, 0.06310829 - Q.tau1) * max(0.0, Q.lam2 - 0.0002104765) / 2.6798e-06   # +0.2%  tau1 < 0.06311 and lam2 > 0.0002105
        + 0.001951041 * max(0.0, 0.002197765 - Q.girth2_top15) / 0.0004509993   # +0.2%  girth2_top15 < 0.002198
        + 0.001739292 * max(0.0, 0.00287991 - Q.girth2_top20) / 0.000559956   # +0.2%  girth2_top20 < 0.00288
        - 0.00166796 * max(0.0, 926.0902 - Q.sum_pt_top20) / 50.299   # -0.2%  sum_pt_top20 < 926.1
        + 0.001633581 * max(0.0, Q.n_dr_0p1_0p2 - 17.0) / 2.16319   # +0.2%  n_dr_0p1_0p2 > 17
        + 0.001588018 * max(0.0, Q.soft10_dr0 - 0.1909669) / 0.03473503   # +0.2%  soft10_dr0 > 0.191
        - 0.001490759 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.05799644 - Q.z_6) / 0.3718674   # -0.1%  mass < 92.86 and z_6 < 0.058
        + 0.00125715 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.zdr_0 - 0.0008718296) / 4.275428e-05   # +0.1%  e2 > 0.04755 and zdr_0 > 0.0008718
        - 0.001198148 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.1%  log_sum_pt < 6.811
        - 0.001189801 * max(0.0, Q.soft10_dr - 0.2075213) / 0.02066177   # -0.1%  soft10_dr > 0.2075
        + 0.001025373 * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) * max(0.0, Q.sj3_dr13 - 0.26623) / 0.002141374   # +0.1%  z_dr_0p1_0p2 < 0.2865 and sj3_dr13 > 0.2662
        + 0.000988336 * max(0.0, 0.8635933 - Q.z_top30_slots) / 0.003797907   # +0.1%  z_top30_slots < 0.8636
        - 0.0009476996 * max(0.0, Q.sj3_mass1 - 19.30204) / 2.042969   # -0.1%  sj3_mass1 > 19.3
        + 0.0008002499 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.zdr_1 - 0.003059578) / 2.442918e-05   # +0.1%  e2 > 0.04755 and zdr_1 > 0.00306
        - 0.0006671351 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.psi_0p3 - 0.9896594) / 5.449059e-06   # -0.1%  e2 > 0.04755 and psi_0p3 > 0.9897
        + 0.0006173272 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.zdr_2 - 0.002740381) / 1.874611e-05   # +0.1%  e2 > 0.04755 and zdr_2 > 0.00274
        + 0.0005890044 * max(0.0, Q.mass_top10 - 99.06678) / 0.7458098   # +0.1%  mass_top10 > 99.07
        - 0.0005309166 * max(0.0, Q.e2 - 0.04755309) * max(0.0, 0.8680812 - Q.eccentricity) / 0.0002874144   # -0.1%  e2 > 0.04755 and eccentricity < 0.8681
        - 0.0005025742 * max(0.0, Q.lam1 - 0.02457025) / 0.0002026642   # -0.1%  lam1 > 0.02457
        - 0.0004620636 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, Q.max_pair_mass - 25.32749) / 6.450547   # -0.0%  n_dr_0p2_0p4 > 15 and max_pair_mass > 25.33
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 42.33;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 42.32575 * (0.1929444
        - 0.1965107 * Q.log_sum_pt / 6.944607   # -19.7%  log_sum_pt
        - 0.09350514 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -9.4%  mass < 92.86
        + 0.07078096 * max(0.0, 91.03469 - Q.mass) / 16.36287   # +7.1%  mass < 91.03
        - 0.05428587 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5305504   # -5.4%  mass < 91.03 and psi_0p3 > 0.9638
        + 0.04140923 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.7612129   # +4.1%  mass < 101 and psi_0p3 > 0.9638
        + 0.03883373 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # +3.9%  LHA < 0.3098
        + 0.03803743 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # +3.8%  mass_over_sum_pt_sq < 0.009595
        - 0.02910592 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.01682193   # -2.9%  mass < 91.03 and psi_0p3 > 0.9974
        + 0.02839737 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.02554674   # +2.8%  mass < 101 and psi_0p3 > 0.9974
        + 0.02828792 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +2.8%  mass < 121.4
        - 0.02771144 * max(0.0, 0.008190222 - Q.width) / 0.002454223   # -2.8%  width < 0.00819
        - 0.02579767 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -2.6%  tau1 < 0.1073
        - 0.02480823 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # -2.5%  girth2_top30 < 0.008376
        - 0.02397001 * max(0.0, 73.35236 - Q.mass_top20) / 16.12893   # -2.4%  mass_top20 < 73.35
        + 0.02189858 * max(0.0, 70.42121 - Q.mass_top20) / 14.59536   # +2.2%  mass_top20 < 70.42
        - 0.02094202 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # -2.1%  LHA < 0.3332
        + 0.02090775 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +2.1%  mass_over_sum_pt < 0.1182
        - 0.02012081 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -2.0%  mass < 101
        + 0.01683195 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # +1.7%  tau1 < 0.08787
        + 0.01613492 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # +1.6%  girth2_top30 < 0.01216
        - 0.01460643 * max(0.0, 0.2845608 - Q.LHA) / 0.05172777   # -1.5%  LHA < 0.2846
        + 0.01353617 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +1.4%  tau1 < 0.07709
        - 0.01297856 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # -1.3%  LHA < 0.2601
        - 0.01252607 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -1.3%  lam1 < 0.008242
        - 0.01223517 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -1.2%  mass < 82.85
        + 0.009823467 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +1.0%  tau21_b2 < 0.2352
        + 0.007858742 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +0.8%  mass < 78.26
        - 0.007718268 * max(0.0, 0.003418057 - Q.girth2_top50) / 0.0004594629   # -0.8%  girth2_top50 < 0.003418
        - 0.006810413 * max(0.0, 1245.697 - Q.sum_pt_top50) / 217.4702   # -0.7%  sum_pt_top50 < 1246
        - 0.006375959 * max(0.0, 0.05936026 - Q.tau2) / 0.0302199   # -0.6%  tau2 < 0.05936
        + 0.005855012 * max(0.0, 0.006929741 - Q.girth2_top30) / 0.002004687   # +0.6%  girth2_top30 < 0.00693
        + 0.005169793 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9299135) / 1.084471   # +0.5%  mass < 91.03 and psi_0p3 > 0.9299
        - 0.004227213 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003362293   # -0.4%  tau21_b2 < 0.2352 and psi_0p3 > 0.9299
        + 0.0042152 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p3 - 0.9973959) / 4.396153e-05   # +0.4%  mass_over_sum_pt < 0.1182 and psi_0p3 > 0.9974
        + 0.003695352 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # +0.4%  lam1 < 0.00619
        + 0.003070671 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.3%  n_dr_0p2_0p4 < 6
        - 0.002901774 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.3%  psi_0p3 > 0.998
        - 0.002537175 * max(0.0, Q.z_top20_slots - 0.9102775) / 0.02689878   # -0.3%  z_top20_slots > 0.9103
        - 0.002467928 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.sd_mass - 55.96662) / 29.57136   # -0.2%  mass < 91.03 and sd_mass > 55.97
        - 0.002290631 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3) / 6.520272e-05   # -0.2%  n_dr_0p2_0p4 < 6 and e3 < 7.876e-05
        + 0.002200177 * max(0.0, 0.007877041 - Q.girth2) * max(0.0, 0.1134943 - Q.M2) / 8.436081e-05   # +0.2%  girth2 < 0.007877 and M2 < 0.1135
        + 0.001766926 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.01175018   # +0.2%  mass < 82.85 and psi_0p3 > 0.9974
        - 0.001579923 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.001696484   # -0.2%  tau21_b2 < 0.2352 and sj2_dr > 0.1938
        + 0.001525267 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 2.275391 - Q.soft1_pt) / 2.585031   # +0.2%  n_dr_0p2_0p4 < 6 and soft1_pt < 2.275
        - 0.001511787 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9995915 - Q.psi_0p3) / 0.0378912   # -0.2%  mass < 82.85 and psi_0p3 < 0.9996
        - 0.001404879 * max(0.0, 18.0 - Q.n_dr_0_0p05) / 7.289224   # -0.1%  n_dr_0_0p05 < 18
        + 0.001343169 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.sd_mass - 76.29481) / 33.79048   # +0.1%  mass < 121.4 and sd_mass > 76.29
        - 0.001209167 * max(0.0, 73.35236 - Q.mass_top20) * max(0.0, 0.002181998 - Q.soft1_z) / 0.02705102   # -0.1%  mass_top20 < 73.35 and soft1_z < 0.002182
        - 0.001081798 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.1825048 - Q.sj2_dr) / 0.06647299   # -0.1%  n_dr_0p2_0p4 < 6 and sj2_dr < 0.1825
        + 0.0009920558 * max(0.0, 90.4505 - Q.mass_top20) / 27.98123   # +0.1%  mass_top20 < 90.45
        + 0.0008605898 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.9973959 - Q.psi_0p3) / 0.03659629   # +0.1%  mass < 91.03 and psi_0p3 < 0.9974
        + 0.0007478559 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 11.65938 - Q.sj3_mass3) / 0.4349059   # +0.1%  tau21_b2 < 0.2352 and sj3_mass3 < 11.66
        - 0.0007456306 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.max_dr - 0.1939977) / 6.595287e-05   # -0.1%  psi_0p3 > 0.998 and max_dr > 0.194
        + 0.0007096212 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.3811408 - Q.soft8_dr0) / 0.4241418   # +0.1%  n_dr_0p2_0p4 < 6 and soft8_dr0 < 0.3811
        - 0.0006866599 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -0.1%  tau21 < 0.3862
        - 0.0006213651 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.n_real_top50 - 34.0) / 0.287919   # -0.1%  tau21_b2 < 0.2352 and n_real_top50 > 34
        - 0.0005571222 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.orientation_deg - -9.840088) / 0.01845442   # -0.1%  psi_0p3 > 0.998 and orientation_deg > -9.84
        + 0.0003602337 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.pt_4 - 60.03125) / 11.52427   # +0.0%  n_dr_0p2_0p4 < 6 and pt_4 > 60.03
        - 0.0003509566 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.3825328 - Q.pt2_over_pt0) / 0.002626916   # -0.0%  tau21_b2 < 0.2352 and pt2_over_pt0 < 0.3825
        - 0.0003402926 * max(0.0, 91.03469 - Q.mass) * max(0.0, 6.0 - Q.n_dr_0_0p05) / 1.610656   # -0.0%  mass < 91.03 and n_dr_0_0p05 < 6
        - 0.0002269437 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.n_dr_0p1_0p2 - 15.0) / 0.00106686   # -0.0%  psi_0p3 > 0.998 and n_dr_0p1_0p2 > 15
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 18.37;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.37162 * (-0.03718491
        + 0.09717509 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266) / 0.00762696   # +9.7%  mass_over_sum_pt_sq > 0.001785
        - 0.06956 * max(0.0, Q.e2_sq - 0.009606007) / 0.003182984   # -7.0%  e2_sq > 0.009606
        - 0.06461098 * max(0.0, Q.girth2_top40 - 0.006026828) / 0.00421794   # -6.5%  girth2_top40 > 0.006027
        + 0.05899807 * max(0.0, Q.mass - 80.78464) / 19.8061   # +5.9%  mass > 80.78
        + 0.05716933 * max(0.0, Q.e2_sq - 0.007872294) / 0.00365949   # +5.7%  e2_sq > 0.007872
        - 0.05715345 * max(0.0, Q.mass - 101.0497) / 12.31084   # -5.7%  mass > 101
        + 0.04932182 * max(0.0, Q.mass - 64.48544) / 31.19164   # +4.9%  mass > 64.49
        - 0.03949477 * max(0.0, Q.mass_top50 - 80.35535) / 18.59691   # -3.9%  mass_top50 > 80.36
        - 0.0330681 * max(0.0, 0.01563836 - Q.girth2_top15) / 0.009593305   # -3.3%  girth2_top15 < 0.01564
        - 0.0317424 * max(0.0, 0.0289594 - Q.girth2_top50) / 0.02002571   # -3.2%  girth2_top50 < 0.02896
        + 0.02802282 * max(0.0, Q.girth2_top40 - 0.005196966) / 0.004725809   # +2.8%  girth2_top40 > 0.005197
        - 0.02698281 * max(0.0, Q.tau1 - 0.06310829) / 0.03403397   # -2.7%  tau1 > 0.06311
        + 0.02678355 * max(0.0, Q.mass_top50 - 97.93004) / 11.91764   # +2.7%  mass_top50 > 97.93
        + 0.02610604 * max(0.0, 0.007887677 - Q.girth2_top15) / 0.00326329   # +2.6%  girth2_top15 < 0.007888
        + 0.02568849 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +2.6%  e3 < 0.0003372
        - 0.02459943 * max(0.0, Q.mass_top50 - 43.66601) / 46.00141   # -2.5%  mass_top50 > 43.67
        + 0.02357701 * max(0.0, Q.girth2_top40 - 0.008031986) / 0.003375597   # +2.4%  girth2_top40 > 0.008032
        - 0.02013121 * max(0.0, 0.02737453 - Q.girth2_top20) / 0.01974268   # -2.0%  girth2_top20 < 0.02737
        + 0.01913065 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # +1.9%  mass_over_sum_pt > 0.07697
        - 0.01622724 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -1.6%  n_dr_0p2_0p4 < 15
        + 0.01468669 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # +1.5%  log_sum_pt > 6.959
        + 0.01436828 * max(0.0, 0.007856958 - Q.girth2_top30) / 0.002601721   # +1.4%  girth2_top30 < 0.007857
        - 0.01306311 * max(0.0, Q.mass - 121.3913) / 7.81281   # -1.3%  mass > 121.4
        - 0.01272244 * max(0.0, Q.sum_pt_top50 - 1038.855) / 34.95069   # -1.3%  sum_pt_top50 > 1039
        + 0.01165358 * max(0.0, 6.935549 - Q.log_sum_pt) / 0.02809622   # +1.2%  log_sum_pt < 6.936
        + 0.01096907 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4) / 0.08975043   # +1.1%  z_dr_0p2_0p4 < 0.1292
        + 0.01045725 * max(0.0, 0.006630167 - Q.girth2_top40) / 0.001656929   # +1.0%  girth2_top40 < 0.00663
        - 0.008939241 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -0.9%  mass_over_sum_pt > 0.1182
        + 0.008341568 * max(0.0, Q.girth2_top15 - 0.001319197) / 0.006270373   # +0.8%  girth2_top15 > 0.001319
        + 0.007348539 * max(0.0, 1032.405 - Q.sum_pt_top40) / 43.16371   # +0.7%  sum_pt_top40 < 1032
        - 0.006609451 * max(0.0, 80.24626 - Q.mass_top30) / 14.57104   # -0.7%  mass_top30 < 80.25
        + 0.005962074 * max(0.0, Q.mass_top40 - 111.2487) / 7.565029   # +0.6%  mass_top40 > 111.2
        + 0.005875674 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +0.6%  LHA > 0.3332
        - 0.005152564 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.5%  log_sum_pt < 6.811
        + 0.005062161 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) / 3.639534   # +0.5%  n_dr_0p2_0p4 > 8
        + 0.004782965 * max(0.0, 1.409617 - Q.D2) / 0.1519846   # +0.5%  D2 < 1.41
        - 0.004737098 * max(0.0, 1032.405 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477) / 0.1123331   # -0.5%  sum_pt_top40 < 1032 and psi_0p3 > 0.9924
        - 0.004446838 * max(0.0, 0.007887677 - Q.girth2_top15) * max(0.0, 0.6133424 - Q.tau21_b2) / 0.0006475533   # -0.4%  girth2_top15 < 0.007888 and tau21_b2 < 0.6133
        + 0.003622536 * max(0.0, 907.9372 - Q.sum_pt) / 3.961002   # +0.4%  sum_pt < 907.9
        + 0.003462289 * max(0.0, 1003.544 - Q.sum_pt_top50) / 20.02299   # +0.3%  sum_pt_top50 < 1004
        - 0.003448797 * max(0.0, Q.mass - 143.7876) / 3.946979   # -0.3%  mass > 143.8
        + 0.003245579 * max(0.0, Q.n_dr_0p1_0p2 - 19.0) / 1.732133   # +0.3%  n_dr_0p1_0p2 > 19
        - 0.003192158 * max(0.0, 0.2555281 - Q.N2) / 0.02566857   # -0.3%  N2 < 0.2555
        - 0.003113667 * max(0.0, Q.sum_pt_top40 - 858.8262) / 166.7848   # -0.3%  sum_pt_top40 > 858.8
        + 0.00304335 * max(0.0, Q.mass - 89.74183) / 15.55572   # +0.3%  mass > 89.74
        + 0.002910269 * max(0.0, Q.n_pt_above_1 - 54.0) / 1.837408   # +0.3%  n_pt_above_1 > 54
        - 0.002716309 * max(0.0, 0.005383629 - Q.girth2_top15) / 0.001666876   # -0.3%  girth2_top15 < 0.005384
        + 0.002567782 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.3%  e2 > 0.04755
        + 0.00234118 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435) / 1.470253   # +0.2%  log_sum_pt > 6.959 and sj3_pair_mass_max > 28.35
        + 0.002259816 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.2%  sj2_dr > 0.2232
        + 0.002194472 * max(0.0, 1001.523 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477) / 0.05538252   # +0.2%  sum_pt_top40 < 1002 and psi_0p3 > 0.9924
        + 0.001899746 * max(0.0, Q.lam1 - 0.008241985) / 0.002595658   # +0.2%  lam1 > 0.008242
        + 0.001727519 * max(0.0, 818.9605 - Q.sum_pt_top15) / 38.21852   # +0.2%  sum_pt_top15 < 819
        + 0.001438038 * max(0.0, Q.C2 - 0.07996447) / 0.01049526   # +0.1%  C2 > 0.07996
        - 0.001333383 * max(0.0, Q.mass_top20 - 90.4505) / 6.038147   # -0.1%  mass_top20 > 90.45
        + 0.001329979 * max(0.0, 1001.523 - Q.sum_pt_top40) / 26.72886   # +0.1%  sum_pt_top40 < 1002
        - 0.001263399 * max(0.0, 1003.544 - Q.sum_pt_top50) * max(0.0, 21.0 - Q.n_dr_0p05_0p1) / 202.9375   # -0.1%  sum_pt_top50 < 1004 and n_dr_0p05_0p1 < 21
        - 0.0007588989 * max(0.0, Q.n_for_90pct - 39.0) / 0.1609176   # -0.1%  n_for_90pct > 39
        - 0.0006428529 * max(0.0, Q.n_pt_above_1 - 54.0) * max(0.0, 15.0 - Q.n_dr_0p05_0p1) / 5.973534   # -0.1%  n_pt_above_1 > 54 and n_dr_0p05_0p1 < 15
        - 0.0005492646 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.zdr_0 - 0.008905716) / 5.779062e-05   # -0.1%  log_sum_pt > 6.959 and zdr_0 > 0.008906
        - 0.0002168855 * max(0.0, 1032.405 - Q.sum_pt_top40) * max(0.0, 50.3522 - Q.mass_top3) / 1525.305   # -0.0%  sum_pt_top40 < 1032 and mass_top3 < 50.35
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 12.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.40618 * (0.02214713
        + 0.1979047 * max(0.0, 132.4278 - Q.mass_top40) / 51.50818   # +19.8%  mass_top40 < 132.4
        - 0.1904919 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -19.0%  mass_top40 < 163.3
        - 0.07431402 * max(0.0, 121.737 - Q.mass_top30) / 46.42587   # -7.4%  mass_top30 < 121.7
        + 0.05094776 * max(0.0, 87.36377 - Q.mass) / 14.19996   # +5.1%  mass < 87.36
        - 0.03122722 * max(0.0, Q.mass - 143.7876) / 3.946979   # -3.1%  mass > 143.8
        + 0.03048571 * max(0.0, 79.65241 - Q.mass) / 10.37103   # +3.0%  mass < 79.65
        + 0.02442177 * max(0.0, 152.6883 - Q.mass_top30) / 74.18247   # +2.4%  mass_top30 < 152.7
        + 0.02310807 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # +2.3%  girth2_top30 < 0.01216
        - 0.02305941 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -2.3%  mass < 64.49
        - 0.02185981 * max(0.0, 33.0 - Q.n_dr_0p1_0p2) / 20.72755   # -2.2%  n_dr_0p1_0p2 < 33
        + 0.02148915 * max(0.0, 0.2091025 - Q.LHA) / 0.02095782   # +2.1%  LHA < 0.2091
        - 0.02128528 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -2.1%  tau1 < 0.05445
        + 0.02010456 * max(0.0, Q.girth - 0.08589404) / 0.01080971   # +2.0%  girth > 0.08589
        + 0.01856711 * max(0.0, Q.mass - 101.0497) / 12.31084   # +1.9%  mass > 101
        - 0.01801337 * max(0.0, 89.6788 - Q.mass_top40) / 17.47899   # -1.8%  mass_top40 < 89.68
        - 0.01753121 * max(0.0, Q.girth - 0.08589404) * max(0.0, 0.002036949 - Q.soft5_z) / 7.737606e-06   # -1.8%  girth > 0.08589 and soft5_z < 0.002037
        - 0.01532847 * max(0.0, 0.05048381 - Q.girth) / 0.008533025   # -1.5%  girth < 0.05048
        + 0.01474751 * max(0.0, Q.girth - 0.09749958) * max(0.0, 0.002036949 - Q.soft5_z) / 5.873801e-06   # +1.5%  girth > 0.0975 and soft5_z < 0.002037
        - 0.01351055 * max(0.0, Q.z_top40_slots - 0.9574183) / 0.02831517   # -1.4%  z_top40_slots > 0.9574
        + 0.01308453 * max(0.0, 956.2133 - Q.sum_pt_top40) / 14.00451   # +1.3%  sum_pt_top40 < 956.2
        + 0.01201812 * max(0.0, 73.33139 - Q.mass_top30) / 11.37701   # +1.2%  mass_top30 < 73.33
        + 0.009772904 * max(0.0, 0.2187642 - Q.z_dr_0p1_0p2) / 0.1086627   # +1.0%  z_dr_0p1_0p2 < 0.2188
        + 0.009615132 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +1.0%  lam1 < 0.004673
        - 0.009141694 * max(0.0, Q.mass - 162.8363) / 1.509832   # -0.9%  mass > 162.8
        + 0.008844573 * max(0.0, 0.003636595 - Q.mass_over_sum_pt_sq) / 0.0004958684   # +0.9%  mass_over_sum_pt_sq < 0.003637
        + 0.008656095 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # +0.9%  girth2_top40 < 0.00626
        + 0.007952989 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # +0.8%  psi_0p3 > 0.9943
        - 0.007894897 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt) / 447.38   # -0.8%  mass_top40 < 80.89 and sum_pt < 1035
        + 0.007078894 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +0.7%  lam1 > 0.01174
        + 0.006275011 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # +0.6%  girth2_top20 < 0.01083
        + 0.006071703 * max(0.0, 0.00634935 - Q.girth2_top50) / 0.001425305   # +0.6%  girth2_top50 < 0.006349
        - 0.006057901 * max(0.0, 0.02783745 - Q.girth) / 0.002411075   # -0.6%  girth < 0.02784
        - 0.005058401 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft5_z - 0.0005078411) / 0.0203614   # -0.5%  sum_pt_top40 < 956.2 and soft5_z > 0.0005078
        + 0.004954758 * max(0.0, 0.004169954 - Q.girth2_top15) / 0.001130716   # +0.5%  girth2_top15 < 0.00417
        - 0.004559502 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258) / 5.834847   # -0.5%  sum_pt_top40 < 956.2 and soft4_pt > 1.789
        - 0.003854368 * max(0.0, 0.00217213 - Q.girth2_top40) / 0.0002075259   # -0.4%  girth2_top40 < 0.002172
        - 0.00346957 * max(0.0, Q.mass - 121.3913) / 7.81281   # -0.3%  mass > 121.4
        + 0.003318803 * max(0.0, Q.mass - 172.4888) / 0.7446233   # +0.3%  mass > 172.5
        + 0.002839532 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft4_pt - 1.375) / 4.279282   # +0.3%  sum_pt_top50 < 959.1 and soft4_pt > 1.375
        + 0.002409293 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.894531) / 2.293712   # +0.2%  sum_pt_top40 < 956.2 and soft5_pt > 2.895
        + 0.00225034 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, Q.n_dr_0p2_0p4 - 10.0) / 39.79287   # +0.2%  mass_top30 < 121.7 and n_dr_0p2_0p4 > 10
        + 0.002198684 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 934.2416 - Q.sum_pt_top50) / 121.4929   # +0.2%  mass_top40 < 80.89 and sum_pt_top50 < 934.2
        - 0.002184645 * max(0.0, Q.width - 0.0258669) / 0.0004653552   # -0.2%  width > 0.02587
        - 0.001992383 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # -0.2%  mass_over_sum_pt > 0.1709
        - 0.001928733 * max(0.0, Q.width - 0.0258669) * max(0.0, Q.soft5_pt - 1.022461) / 0.0004441883   # -0.2%  width > 0.02587 and soft5_pt > 1.022
        + 0.001752127 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.n_particles - 38.0) / 73.85286   # +0.2%  mass < 87.36 and n_particles > 38
        + 0.001724094 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft5_z - 0.0009919684) / 0.008783167   # +0.2%  sum_pt_top50 < 959.1 and soft5_z > 0.000992
        + 0.001701774 * max(0.0, 26.0 - Q.n_pt_above_1) / 0.8081244   # +0.2%  n_pt_above_1 < 26
        + 0.001578946 * max(0.0, Q.z_top5 - 0.7963975) * max(0.0, 7.740999 - Q.ptdr0_2) / 0.04194485   # +0.2%  z_top5 > 0.7964 and ptdr0_2 < 7.741
        + 0.001418679 * max(0.0, Q.lam1 - 0.01174405) * max(0.0, 2.894531 - Q.soft5_pt) / 0.002564166   # +0.1%  lam1 > 0.01174 and soft5_pt < 2.895
        + 0.001407614 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # +0.1%  e3 > 0.0003372
        - 0.001381444 * max(0.0, Q.lam1 - 0.01174405) * max(0.0, Q.sj3_pairmin_over_m - 0.1952481) / 0.0003017671   # -0.1%  lam1 > 0.01174 and sj3_pairmin_over_m > 0.1952
        - 0.001340888 * max(0.0, Q.sum_pt_top15 - 985.0781) / 16.07962   # -0.1%  sum_pt_top15 > 985.1
        - 0.00105477 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft5_pt - 2.894531) / 1.196279   # -0.1%  sum_pt_top50 < 959.1 and soft5_pt > 2.895
        + 0.001021026 * max(0.0, Q.mass_over_sum_pt - 0.1708801) * max(0.0, Q.soft4_pt - 1.789258) / 0.0003712571   # +0.1%  mass_over_sum_pt > 0.1709 and soft4_pt > 1.789
        - 0.001020134 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, 20.0 - Q.n_pt_above_10) / 29.14649   # -0.1%  sum_pt_top40 < 956.2 and n_pt_above_10 < 20
        - 0.0008517465 * max(0.0, Q.girth - 0.08589404) * max(0.0, Q.z_dr_0p05_0p1 - 0.5106729) / 0.0002060031   # -0.1%  girth > 0.08589 and z_dr_0p05_0p1 > 0.5107
        + 0.0007287759 * max(0.0, Q.mass - 143.7876) * max(0.0, Q.z_dr_0p05_0p1 - 0.4026646) / 0.1395143   # +0.1%  mass > 143.8 and z_dr_0p05_0p1 > 0.4027
        - 0.000571841 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.psi_0p1 - 0.9184255) / 0.111142   # -0.1%  sum_pt_top40 < 956.2 and psi_0p1 > 0.9184
        - 0.0005651131 * max(0.0, Q.lam1 - 0.01174405) * max(0.0, Q.soft5_z - 0.002346473) / 4.122063e-07   # -0.1%  lam1 > 0.01174 and soft5_z > 0.002346
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 24.56;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.55805 * (-0.08785734
        + 0.1099578 * Q.sum_pt / 1043.796   # +11.0%  sum_pt
        - 0.1011524 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -10.1%  girth < 0.1207
        - 0.07535859 * max(0.0, Q.mass - 64.48544) / 31.19164   # -7.5%  mass > 64.49
        + 0.07160614 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # +7.2%  tau1 < 0.1507
        + 0.05384166 * max(0.0, Q.mass - 80.78464) / 19.8061   # +5.4%  mass > 80.78
        + 0.04159383 * max(0.0, Q.mass - 89.74183) / 15.55572   # +4.2%  mass > 89.74
        + 0.03935976 * max(0.0, 132.4278 - Q.mass_top40) / 51.50818   # +3.9%  mass_top40 < 132.4
        - 0.03815366 * max(0.0, 80.3008 - Q.sj2_mass1) / 50.41572   # -3.8%  sj2_mass1 < 80.3
        + 0.0325245 * max(0.0, 0.01807679 - Q.girth2_top30) / 0.01083719   # +3.3%  girth2_top30 < 0.01808
        - 0.02570427 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -2.6%  mass_top40 < 83.33
        + 0.02374519 * max(0.0, Q.pt_entropy - 2.07371) / 0.8049786   # +2.4%  pt_entropy > 2.074
        - 0.02006542 * max(0.0, 0.06524004 - Q.e2) / 0.03476195   # -2.0%  e2 < 0.06524
        - 0.01754429 * max(0.0, Q.mass - 143.7876) / 3.946979   # -1.8%  mass > 143.8
        - 0.01751146 * max(0.0, Q.mass - 101.0497) / 12.31084   # -1.8%  mass > 101
        + 0.01734324 * max(0.0, 3.814159 - Q.D2) / 1.519965   # +1.7%  D2 < 3.814
        + 0.01724326 * max(0.0, Q.mass_top40 - 67.72643) / 24.79585   # +1.7%  mass_top40 > 67.73
        - 0.01498206 * max(0.0, Q.log_sum_pt - 6.903423) / 0.05662275   # -1.5%  log_sum_pt > 6.903
        + 0.01479618 * max(0.0, 0.009962397 - Q.girth2_top15) / 0.004894699   # +1.5%  girth2_top15 < 0.009962
        + 0.01471717 * max(0.0, 87.27603 - Q.mass_top40) / 15.9837   # +1.5%  mass_top40 < 87.28
        - 0.01387493 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -1.4%  tau1 < 0.05445
        + 0.01344354 * max(0.0, 111.2487 - Q.mass_top40) / 33.99436   # +1.3%  mass_top40 < 111.2
        + 0.01313946 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +1.3%  mass < 74.25
        + 0.01212066 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.2%  z_dr_0p2_0p4 < 0.09123
        + 0.01160887 * max(0.0, 839.9547 - Q.sum_pt_top5) / 255.1112   # +1.2%  sum_pt_top5 < 840
        - 0.01111127 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # -1.1%  e2 < 0.04083
        - 0.01068845 * max(0.0, 71.781 - Q.mass_top10) / 28.28521   # -1.1%  mass_top10 < 71.78
        - 0.01024967 * max(0.0, 75.26407 - Q.mass_top15) / 22.29039   # -1.0%  mass_top15 < 75.26
        - 0.009437413 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -0.9%  n_dr_0p2_0p4 < 11
        - 0.008653452 * max(0.0, Q.mass - 82.85409) / 18.72167   # -0.9%  mass > 82.85
        + 0.008651776 * max(0.0, 0.6133424 - Q.tau21_b2) / 0.3099701   # +0.9%  tau21_b2 < 0.6133
        + 0.008189628 * max(0.0, 0.2091025 - Q.LHA) / 0.02095782   # +0.8%  LHA < 0.2091
        + 0.007705744 * max(0.0, Q.mass_top50 - 138.8977) / 3.930268   # +0.8%  mass_top50 > 138.9
        - 0.007541211 * max(0.0, Q.lam1 - 0.001260456) / 0.006719699   # -0.8%  lam1 > 0.00126
        + 0.007025612 * max(0.0, 0.07852883 - Q.mass_over_sum_pt) / 0.01107516   # +0.7%  mass_over_sum_pt < 0.07853
        - 0.006897675 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # -0.7%  girth2_top30 < 0.005402
        - 0.006783884 * max(0.0, 3.814159 - Q.D2) * max(0.0, Q.n_real_top40 - 32.0) / 8.464645   # -0.7%  D2 < 3.814 and n_real_top40 > 32
        - 0.00642495 * max(0.0, 0.03577037 - Q.girth) / 0.004182456   # -0.6%  girth < 0.03577
        + 0.006270438 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +0.6%  dr_0 < 0.06413
        + 0.006258531 * max(0.0, Q.mass_top5 - 14.54404) / 16.7137   # +0.6%  mass_top5 > 14.54
        - 0.006132838 * max(0.0, Q.mass - 162.8363) / 1.509832   # -0.6%  mass > 162.8
        + 0.005477436 * max(0.0, Q.mass - 64.48544) * max(0.0, 2.537109 - Q.soft2_pt) / 47.07428   # +0.5%  mass > 64.49 and soft2_pt < 2.537
        + 0.00504982 * max(0.0, Q.n_real_top40 - 36.0) / 2.636901   # +0.5%  n_real_top40 > 36
        - 0.005049076 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # -0.5%  mass_over_sum_pt > 0.1709
        + 0.004430949 * max(0.0, 0.06940472 - Q.dr_1) / 0.0280884   # +0.4%  dr_1 < 0.0694
        + 0.00419353 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # +0.4%  mass_over_sum_pt_sq > 0.0292
        - 0.003616565 * max(0.0, 0.01807679 - Q.girth2_top30) * max(0.0, 0.0002836757 - Q.z_dr_0p4_up) / 2.758759e-06   # -0.4%  girth2_top30 < 0.01808 and z_dr_0p4_up < 0.0002837
        - 0.003521904 * max(0.0, Q.M2 - 0.04577853) / 0.02232023   # -0.4%  M2 > 0.04578
        - 0.002578854 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # -0.3%  psi_0p3 > 0.9956
        - 0.002078105 * max(0.0, 0.005312783 - Q.girth2_top20) / 0.001437214   # -0.2%  girth2_top20 < 0.005313
        - 0.002044168 * max(0.0, Q.psi_0p3 - 0.9985421) * max(0.0, Q.n_real_top30 - 22.0) / 0.002934868   # -0.2%  psi_0p3 > 0.9985 and n_real_top30 > 22
        + 0.002001395 * max(0.0, 0.01541561 - Q.e2) / 0.001436351   # +0.2%  e2 < 0.01542
        + 0.001955118 * max(0.0, 3.814159 - Q.D2) * max(0.0, Q.max_dr - 0.2404747) / 0.1627144   # +0.2%  D2 < 3.814 and max_dr > 0.2405
        + 0.001870239 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, 464.75 - Q.sum_pt_top2) / 5683.5   # +0.2%  sj2_mass1 < 80.3 and sum_pt_top2 < 464.8
        - 0.001665856 * max(0.0, 3.814159 - Q.D2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.04343614   # -0.2%  D2 < 3.814 and sj2_dr > 0.1938
        - 0.00152866 * max(0.0, 11.91979 - Q.sj2_mass2) / 3.453476   # -0.2%  sj2_mass2 < 11.92
        + 0.0014173 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.91979) / 145.9595   # +0.1%  sj2_mass1 < 80.3 and sj2_mass2 > 11.92
        + 0.001303798 * max(0.0, Q.mass - 82.85409) * max(0.0, Q.psi_0p3 - 0.9853273) / 0.08864606   # +0.1%  mass > 82.85 and psi_0p3 > 0.9853
        + 0.001275241 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.0830523 - Q.dr_4) / 0.002292504   # +0.1%  log_sum_pt > 6.903 and dr_4 < 0.08305
        - 0.001226324 * max(0.0, 0.1207452 - Q.girth) * max(0.0, 0.8125223 - Q.tau32) / 0.003712688   # -0.1%  girth < 0.1207 and tau32 < 0.8125
        - 0.001033015 * max(0.0, Q.z_dr_0_0p05 - 0.7674734) * max(0.0, 4.450169 - Q.D2) / 0.03687443   # -0.1%  z_dr_0_0p05 > 0.7675 and D2 < 4.45
        + 0.001028023 * max(0.0, 42.41192 - Q.mass_top30) / 2.481216   # +0.1%  mass_top30 < 42.41
        + 0.0009973312 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # +0.1%  mass_top50 > 157.5
        + 0.0008938658 * max(0.0, 918.0938 - Q.sum_pt_top15) * max(0.0, 0.07464736 - Q.dr_2) / 1.629019   # +0.1%  sum_pt_top15 < 918.1 and dr_2 < 0.07465
        + 0.0008808303 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.002245509 - Q.centroid_offset) / 5.948473e-05   # +0.1%  log_sum_pt > 6.903 and centroid_offset < 0.002246
        + 0.0008096002 * max(0.0, 111.2487 - Q.mass_top40) * max(0.0, Q.pt1_dr01 - 12.6865) / 28.78446   # +0.1%  mass_top40 < 111.2 and pt1_dr01 > 12.69
        + 0.0007827875 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.03259277 - Q.absphi_2) / 0.0008069178   # +0.1%  log_sum_pt > 6.903 and absphi_2 < 0.03259
        + 0.0006250281 * max(0.0, Q.mass - 162.8363) * max(0.0, 41.4375 - Q.pt_9) / 16.12776   # +0.1%  mass > 162.8 and pt_9 < 41.44
        - 0.0005007674 * max(0.0, 750.7313 - Q.sum_pt_top20) / 7.067235   # -0.1%  sum_pt_top20 < 750.7
        + 0.0003599228 * max(0.0, Q.mass_top40 - 163.2541) * max(0.0, Q.soft4_z - 0.001186042) / 0.0003797437   # +0.0%  mass_top40 > 163.3 and soft4_z > 0.001186
        - 0.0002061124 * max(0.0, Q.mass - 82.85409) * max(0.0, Q.n_dr_0p4_up - 0.0) / 4.047143   # -0.0%  mass > 82.85 and n_dr_0p4_up > 0
        + 0.0001874862 * max(0.0, Q.mass - 172.4888) * max(0.0, Q.M2 - 0.04260132) / 0.01594348   # +0.0%  mass > 172.5 and M2 > 0.0426
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 14.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.41306 * (-0.01633935
        - 0.1123886 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # -11.2%  LHA < 0.3098
        + 0.09979765 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +10.0%  mass < 92.86
        + 0.06183878 * max(0.0, 0.3203321 - Q.LHA) / 0.07499257   # +6.2%  LHA < 0.3203
        + 0.04573817 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # +4.6%  tau1 < 0.1073
        - 0.04501217 * max(0.0, 85.8667 - Q.mass_top50) / 13.95835   # -4.5%  mass_top50 < 85.87
        + 0.04286366 * max(0.0, 97.93004 - Q.mass_top50) / 22.03533   # +4.3%  mass_top50 < 97.93
        + 0.04128608 * max(0.0, 0.2941033 - Q.LHA) / 0.05718399   # +4.1%  LHA < 0.2941
        - 0.0367309 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # -3.7%  girth < 0.07374
        + 0.03076855 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +3.1%  e2_sq < 0.008184
        - 0.02860954 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # -2.9%  girth2_top10 < 0.007679
        - 0.02850401 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # -2.9%  tau1 < 0.08787
        - 0.02584173 * max(0.0, 79.65241 - Q.mass) / 10.37103   # -2.6%  mass < 79.65
        - 0.02453802 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -2.5%  mass < 101
        + 0.02194401 * max(0.0, 60.5496 - Q.mass_top10) / 20.08884   # +2.2%  mass_top10 < 60.55
        - 0.02171646 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # -2.2%  e2 < 0.04083
        + 0.0194556 * max(0.0, 0.02793599 - Q.e2) / 0.005715277   # +1.9%  e2 < 0.02794
        - 0.01895123 * max(0.0, 0.007639643 - Q.girth2_top2) / 0.004543329   # -1.9%  girth2_top2 < 0.00764
        - 0.0180951 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -1.8%  lam1 < 0.006717
        + 0.01779537 * max(0.0, 0.2845608 - Q.LHA) / 0.05172777   # +1.8%  LHA < 0.2846
        - 0.01730196 * max(0.0, 0.006929741 - Q.girth2_top30) / 0.002004687   # -1.7%  girth2_top30 < 0.00693
        - 0.01663692 * max(0.0, 38.52415 - Q.mass_top10) / 8.8473   # -1.7%  mass_top10 < 38.52
        + 0.01512015 * max(0.0, 0.004752876 - Q.girth2_top10) / 0.001623418   # +1.5%  girth2_top10 < 0.004753
        + 0.01490999 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 43.23581   # +1.5%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21
        + 0.01399705 * max(0.0, 0.004007842 - Q.girth2_top2) / 0.001907211   # +1.4%  girth2_top2 < 0.004008
        - 0.01188575 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2) / 0.005195774   # -1.2%  n_dr_0p2_0p4 < 10 and girth2 < 0.005532
        + 0.009840933 * max(0.0, 0.003213724 - Q.girth2_top10) / 0.0009423838   # +1.0%  girth2_top10 < 0.003214
        - 0.009505397 * max(0.0, Q.psi_0p3 - 0.9853273) / 0.009408723   # -1.0%  psi_0p3 > 0.9853
        + 0.008900718 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.mass_top10 - 38.52415) / 111.7968   # +0.9%  mass < 101 and mass_top10 > 38.52
        + 0.008898763 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # +0.9%  mass_over_sum_pt_sq < 0.009595
        + 0.008177351 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.8%  n_dr_0p2_0p4 < 6
        - 0.007812472 * max(0.0, 86.25221 - Q.mass_top30) / 18.20727   # -0.8%  mass_top30 < 86.25
        + 0.007742163 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +0.8%  mass_top30 < 60.44
        + 0.007156791 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +0.7%  e2 < 0.02516
        + 0.006072903 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +0.6%  n_dr_0p2_0p4 < 10
        - 0.005744891 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -0.6%  mass < 64.49
        + 0.005568156 * max(0.0, 15.0 - Q.n_dr_0p1_0p2) / 5.139044   # +0.6%  n_dr_0p1_0p2 < 15
        + 0.005563504 * max(0.0, 0.00130722 - Q.girth2_top10) / 0.0002794102   # +0.6%  girth2_top10 < 0.001307
        + 0.00518337 * max(0.0, Q.z_top40_slots - 0.9886962) / 0.005919309   # +0.5%  z_top40_slots > 0.9887
        - 0.005079129 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 51.2028 - Q.sj2_mass1) / 122.8376   # -0.5%  n_dr_0p2_0p4 < 10 and sj2_mass1 < 51.2
        - 0.004761051 * max(0.0, 0.1548383 - Q.z_dr_0p1_0p2) / 0.06710802   # -0.5%  z_dr_0p1_0p2 < 0.1548
        - 0.004370662 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_real_top30 - 22.0) / 9.916795   # -0.4%  n_dr_0p2_0p4 < 6 and n_real_top30 > 22
        + 0.00422558 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_pt - 907.9372) / 548.1551   # +0.4%  n_dr_0p2_0p4 < 10 and sum_pt > 907.9
        + 0.004147635 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.z_11 - 0.01184033) / 0.1688776   # +0.4%  mass < 101 and z_11 > 0.01184
        - 0.004138475 * max(0.0, 0.09573228 - Q.z_dr_0_0p05) / 0.02128126   # -0.4%  z_dr_0_0p05 < 0.09573
        + 0.003921298 * max(0.0, 0.05048381 - Q.girth) / 0.008533025   # +0.4%  girth < 0.05048
        - 0.0038936 * max(0.0, Q.z_top10_slots - 0.7271951) / 0.0653343   # -0.4%  z_top10_slots > 0.7272
        + 0.003493831 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.3%  psi_0p3 > 0.9974
        + 0.003452677 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 0.05112769 - Q.C2_b2) / 0.0003814199   # +0.3%  z_top30_slots > 0.9735 and C2_b2 < 0.05113
        - 0.002952806 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0) / 30.16986   # -0.3%  n_dr_0p2_0p4 < 10 and n_particles > 34
        - 0.002674953 * max(0.0, 0.005911134 - Q.zdr_0) / 0.001247991   # -0.3%  zdr_0 < 0.005911
        - 0.002662388 * max(0.0, 79.65241 - Q.mass) * max(0.0, Q.z_11 - 0.01375115) / 0.05008222   # -0.3%  mass < 79.65 and z_11 > 0.01375
        - 0.002635834 * max(0.0, 4.204324e-05 - Q.e3) / 1.197776e-05   # -0.3%  e3 < 4.204e-05
        + 0.002563915 * max(0.0, Q.psi_0p2 - 0.9935324) / 0.001465572   # +0.3%  psi_0p2 > 0.9935
        + 0.00240093 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_7 - 0.01968091) / 0.05641123   # +0.2%  n_dr_0p2_0p4 < 10 and z_7 > 0.01968
        - 0.002320633 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.2%  n_dr_0p1_0p2 < 8
        + 0.002205612 * max(0.0, 0.09573228 - Q.z_dr_0_0p05) * max(0.0, Q.max_dr - 0.1939977) / 0.003427714   # +0.2%  z_dr_0_0p05 < 0.09573 and max_dr > 0.194
        + 0.001829626 * max(0.0, Q.z_top30_slots - 0.9948118) / 0.001278706   # +0.2%  z_top30_slots > 0.9948
        - 0.001627484 * max(0.0, 0.07374472 - Q.girth) * max(0.0, Q.pt_7 - 20.125) / 0.2625894   # -0.2%  girth < 0.07374 and pt_7 > 20.12
        + 0.0014871 * max(0.0, 0.144845 - Q.tau21_b2) * max(0.0, Q.z_5 - 0.03252317) / 0.0003129002   # +0.1%  tau21_b2 < 0.1448 and z_5 > 0.03252
        + 0.001268928 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) * max(0.0, Q.z_dr_0p1_0p2 - 0.1548383) / 4.64505e-05   # +0.1%  mass_over_sum_pt_sq < 0.009595 and z_dr_0p1_0p2 > 0.1548
        - 0.001148724 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # -0.1%  sum_pt_top5 > 791.1
        - 0.001129963 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # -0.1%  sum_pt < 986.1
        - 0.001085972 * max(0.0, 79.65241 - Q.mass) * max(0.0, 3.814159 - Q.D2) / 4.810976   # -0.1%  mass < 79.65 and D2 < 3.814
        + 0.0006262995 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 1017.435 - Q.sum_pt) / 16.4952   # +0.1%  n_dr_0p2_0p4 < 6 and sum_pt < 1017
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 2.52;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.520374 * (-0.05392087
        + 0.2287238 * max(0.0, 80.78464 - Q.mass) / 10.84952   # +22.9%  mass < 80.78
        + 0.1499884 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +15.0%  mass < 82.85
        + 0.1099605 * max(0.0, 87.36377 - Q.mass) / 14.19996   # +11.0%  mass < 87.36
        - 0.08332149 * max(0.0, 77.93668 - Q.mass_top40) / 11.12028   # -8.3%  mass_top40 < 77.94
        + 0.07609112 * max(0.0, 94.64253 - Q.mass_top40) / 21.02021   # +7.6%  mass_top40 < 94.64
        - 0.06150496 * max(0.0, 0.06895248 - Q.mass_over_sum_pt) / 0.007729512   # -6.2%  mass_over_sum_pt < 0.06895
        - 0.05790809 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -5.8%  mass < 53.87
        - 0.04408159 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.6147674   # -4.4%  mass < 87.36 and z_dr_0p1_0p2 < 0.06473
        - 0.02967052 * max(0.0, 46.36558 - Q.mass_top15) / 7.938634   # -3.0%  mass_top15 < 46.37
        - 0.02671286 * max(0.0, 94.64253 - Q.mass_top40) * max(0.0, 0.5760704 - Q.z_top2_slots) / 3.935323   # -2.7%  mass_top40 < 94.64 and z_top2_slots < 0.5761
        - 0.02433964 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3) / 0.0414109   # -2.4%  mass < 87.36 and psi_0p3 < 0.999
        - 0.02091616 * max(0.0, 0.00363788 - Q.e2_sq) / 0.0004963098   # -2.1%  e2_sq < 0.003638
        + 0.01414877 * max(0.0, 24.20196 - Q.mass_top15) / 1.791326   # +1.4%  mass_top15 < 24.2
        + 0.01051849 * max(0.0, 0.0007431905 - Q.girth2_top10) / 0.0001243965   # +1.1%  girth2_top10 < 0.0007432
        - 0.01020559 * max(0.0, Q.z_dr_0_0p05 - 0.9084912) / 0.009227347   # -1.0%  z_dr_0_0p05 > 0.9085
        + 0.00988278 * max(0.0, 52.15154 - Q.mass_top50) / 3.332261   # +1.0%  mass_top50 < 52.15
        - 0.008316271 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.lam2 - 0.0003497174) / 0.002699158   # -0.8%  mass < 87.36 and lam2 > 0.0003497
        + 0.006136459 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.3036026 - Q.planar_flow) / 0.0008856905   # +0.6%  z_dr_0p1_0p2 > 0.4479 and planar_flow < 0.3036
        + 0.005585299 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.4026646) / 0.2505715   # +0.6%  mass < 87.36 and z_dr_0p05_0p1 > 0.4027
        - 0.005201835 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.001776308 - Q.lam2) / 5.387726e-06   # -0.5%  z_dr_0p1_0p2 > 0.4479 and lam2 < 0.001776
        + 0.004905107 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) / 0.02385119   # +0.5%  z_dr_0p1_0p2 > 0.4479
        - 0.003794572 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 0.05112769 - Q.C2_b2) / 0.2840034   # -0.4%  sum_pt < 972 and C2_b2 < 0.05113
        + 0.003715708 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 435.25 - Q.pt_0) / 490.7786   # +0.4%  sj3_pair_mass_max > 122.7 and pt_0 < 435.2
        + 0.002208418 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1) / 9.827377   # +0.2%  sd_mass > 133.3 and sj3_mass1 < 32.5
        + 0.002161625 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 76.60223 - Q.sj3_pair_mass_min) / 12.46005   # +0.2%  sd_mass > 133.3 and sj3_pair_mass_min < 76.6
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 7.493;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.49337 * (0.2306788
        - 0.116474 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -11.6%  sum_pt < 1085
        + 0.09353635 * max(0.0, Q.mass - 64.48544) / 31.19164   # +9.4%  mass > 64.49
        - 0.09233887 * max(0.0, Q.mass - 143.7876) / 3.946979   # -9.2%  mass > 143.8
        - 0.05553719 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -5.6%  sum_pt < 1013
        - 0.05321778 * max(0.0, Q.mass_top50 - 97.93004) / 11.91764   # -5.3%  mass_top50 > 97.93
        - 0.04403242 * max(0.0, Q.mass_over_sum_pt - 0.06030419) / 0.03186977   # -4.4%  mass_over_sum_pt > 0.0603
        + 0.04300194 * max(0.0, Q.mass_top50 - 138.8977) / 3.930268   # +4.3%  mass_top50 > 138.9
        + 0.03823721 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +3.8%  sum_pt_top40 < 1053
        + 0.03730921 * max(0.0, 0.02497133 - Q.girth2_top40) / 0.01655385   # +3.7%  girth2_top40 < 0.02497
        + 0.03322436 * max(0.0, 1008.935 - Q.sum_pt_top50) / 22.0436   # +3.3%  sum_pt_top50 < 1009
        + 0.03053264 * max(0.0, Q.mass - 78.26182) / 21.33658   # +3.1%  mass > 78.26
        - 0.02649629 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -2.6%  mass_top50 > 157.5
        + 0.02564887 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.5260785 - Q.D3) / 23.2296   # +2.6%  sum_pt_top40 < 1053 and D3 < 0.5261
        + 0.02109814 * max(0.0, Q.mass_over_sum_pt - 0.07435617) / 0.02188714   # +2.1%  mass_over_sum_pt > 0.07436
        + 0.02038492 * max(0.0, Q.mass - 172.4888) * max(0.0, 41.4375 - Q.pt_9) / 7.126564   # +2.0%  mass > 172.5 and pt_9 < 41.44
        + 0.01863055 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # +1.9%  tau1 < 0.08787
        - 0.01862027 * max(0.0, 3.654944 - Q.pt_entropy) / 0.8125705   # -1.9%  pt_entropy < 3.655
        + 0.01624977 * max(0.0, Q.mass_top40 - 150.0144) / 1.666872   # +1.6%  mass_top40 > 150
        - 0.01588667 * max(0.0, 6.879399 - Q.log_sum_pt) / 0.01083309   # -1.6%  log_sum_pt < 6.879
        - 0.01523418 * max(0.0, 19.0 - Q.n_dr_0p1_0p2) / 8.166005   # -1.5%  n_dr_0p1_0p2 < 19
        + 0.01210232 * max(0.0, Q.mass_top50 - 97.93004) * max(0.0, 2.873047 - Q.soft3_pt) / 19.23325   # +1.2%  mass_top50 > 97.93 and soft3_pt < 2.873
        - 0.01104711 * max(0.0, 950.9324 - Q.sum_pt_top30) / 25.09781   # -1.1%  sum_pt_top30 < 950.9
        - 0.01097919 * max(0.0, Q.sum_pt - 1260.541) / 7.655483   # -1.1%  sum_pt > 1261
        + 0.01091694 * max(0.0, Q.mass_top5 - 37.76455) / 5.869288   # +1.1%  mass_top5 > 37.76
        + 0.01021217 * max(0.0, Q.psi_0p3 - 0.9777125) / 0.01571542   # +1.0%  psi_0p3 > 0.9777
        - 0.009995698 * max(0.0, Q.mass - 172.4888) / 0.7446233   # -1.0%  mass > 172.5
        + 0.009031976 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # +0.9%  log_sum_pt > 7.139
        - 0.008805145 * max(0.0, Q.sum_pt_top15 - 951.1375) / 24.48977   # -0.9%  sum_pt_top15 > 951.1
        + 0.007895971 * max(0.0, Q.psi_0p3 - 0.9777125) * max(0.0, Q.max_dr - 0.2738063) / 0.001214328   # +0.8%  psi_0p3 > 0.9777 and max_dr > 0.2738
        + 0.007793194 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.1269782 - Q.dr_12) / 3.348364   # +0.8%  sum_pt < 1085 and dr_12 < 0.127
        + 0.007589246 * max(0.0, Q.mass_top10 - 60.5496) / 6.712581   # +0.8%  mass_top10 > 60.55
        - 0.007232053 * max(0.0, Q.mass - 172.4888) * max(0.0, 0.0301005 - Q.z_9) / 0.002811277   # -0.7%  mass > 172.5 and z_9 < 0.0301
        - 0.007035778 * max(0.0, 889.8383 - Q.sum_pt_top20) / 35.81237   # -0.7%  sum_pt_top20 < 889.8
        - 0.00574413 * max(0.0, 14.0 - Q.n_for_90pct) / 1.135089   # -0.6%  n_for_90pct < 14
        - 0.00570485 * max(0.0, Q.mass_top5 - 37.76455) * max(0.0, 27.0 - Q.n_dr_0p05_0p1) / 94.87774   # -0.6%  mass_top5 > 37.76 and n_dr_0p05_0p1 < 27
        + 0.005584485 * max(0.0, Q.sum_pt_top20 - 1064.139) / 11.11947   # +0.6%  sum_pt_top20 > 1064
        - 0.005244908 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # -0.5%  mass_top10 > 71.78
        + 0.005152623 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # +0.5%  sum_pt_top50 < 959.1
        - 0.004371323 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) / 0.03630126   # -0.4%  z_dr_0_0p05 > 0.8103
        - 0.004099831 * max(0.0, Q.max_dr - 0.3315857) / 0.04315475   # -0.4%  max_dr > 0.3316
        + 0.003837037 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.1023813 - Q.dr_13) / 2.159889   # +0.4%  sum_pt < 1085 and dr_13 < 0.1024
        + 0.003492276 * max(0.0, 14.0 - Q.n_for_90pct) * max(0.0, 3.345339 - Q.D2) / 0.7823248   # +0.3%  n_for_90pct < 14 and D2 < 3.345
        - 0.003261221 * max(0.0, Q.mass - 64.48544) * max(0.0, 0.002227078 - Q.zdr_12) / 0.01550191   # -0.3%  mass > 64.49 and zdr_12 < 0.002227
        + 0.002490593 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.01833434 - Q.z_9) / 0.01801529   # +0.2%  sum_pt < 1013 and z_9 < 0.01833
        - 0.002241955 * max(0.0, 20.54688 - Q.pt_9) / 1.525182   # -0.2%  pt_9 < 20.55
        + 0.002070199 * max(0.0, Q.lam1 - 0.01649354) / 0.001003295   # +0.2%  lam1 > 0.01649
        - 0.001831211 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.2%  log_sum_pt < 6.811
        + 0.001771256 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 800.732 - Q.sum_pt_top30) / 1062.214   # +0.2%  sum_pt < 1085 and sum_pt_top30 < 800.7
        - 0.001511689 * max(0.0, Q.D2 - 5.378975) / 0.5670303   # -0.2%  D2 > 5.379
        - 0.001284779 * max(0.0, Q.mass - 172.4888) * max(0.0, 0.002357824 - Q.soft6_z) / 0.0004574479   # -0.1%  mass > 172.5 and soft6_z < 0.002358
        - 0.001004257 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.00317537 - Q.C3) / 7.466262e-06   # -0.1%  log_sum_pt > 7.139 and C3 < 0.003175
        - 0.0009931811 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, -0.0892334 - Q.eta_1) / 1.48343e-05   # -0.1%  log_sum_pt < 6.811 and eta_1 < -0.08923
        - 0.0006397736 * max(0.0, Q.mass_top50 - 97.93004) * max(0.0, 0.0006388081 - Q.soft7_z) / 6.542946e-05   # -0.1%  mass_top50 > 97.93 and soft7_z < 0.0006388
        + 0.0005048294 * max(0.0, Q.mass_top50 - 97.93004) * max(0.0, 0.6713867 - Q.soft7_pt) / 0.06788123   # +0.1%  mass_top50 > 97.93 and soft7_pt < 0.6714
        - 0.0004783437 * max(0.0, 889.8383 - Q.sum_pt_top20) * max(0.0, 0.01833434 - Q.z_9) / 0.004165657   # -0.0%  sum_pt_top20 < 889.8 and z_9 < 0.01833
        - 0.0003567733 * max(0.0, Q.mass_top50 - 157.5448) * max(0.0, Q.sd_zg - 0.4462823) / 0.004617902   # -0.0%  mass_top50 > 157.5 and sd_zg > 0.4463
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 22.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.59081 * (-0.01079074
        + 0.1148084 * max(0.0, 0.01986381 - Q.mass_over_sum_pt_sq) / 0.0117775   # +11.5%  mass_over_sum_pt_sq < 0.01986
        + 0.0948092 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +9.5%  mass_over_sum_pt < 0.1182
        - 0.08716715 * max(0.0, 91.03469 - Q.mass) / 16.36287   # -8.7%  mass < 91.03
        - 0.08607261 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) / 0.0178873   # -8.6%  mass_over_sum_pt < 0.09047
        - 0.06819766 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -6.8%  mass_top50 > 157.5
        - 0.05865754 * max(0.0, 0.01292642 - Q.girth2_top40) / 0.006292908   # -5.9%  girth2_top40 < 0.01293
        - 0.04065718 * max(0.0, 0.01897915 - Q.girth2_top40) / 0.01130444   # -4.1%  girth2_top40 < 0.01898
        - 0.03923121 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -3.9%  sum_pt_top50 > 959.1
        + 0.03688651 * max(0.0, 0.08873143 - Q.mass_over_sum_pt) / 0.01671255   # +3.7%  mass_over_sum_pt < 0.08873
        + 0.03204149 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +3.2%  mass_over_sum_pt < 0.09795
        + 0.03035824 * max(0.0, 79.65241 - Q.mass) / 10.37103   # +3.0%  mass < 79.65
        - 0.02848759 * max(0.0, 0.1008497 - Q.tau1) / 0.02962944   # -2.8%  tau1 < 0.1008
        - 0.0275511 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -2.8%  girth2_top20 < 0.01083
        - 0.02742664 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -2.7%  mass < 82.85
        + 0.0196191 * max(0.0, Q.sum_pt_top40 - 906.6023) / 122.3672   # +2.0%  sum_pt_top40 > 906.6
        + 0.01812651 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # +1.8%  mass_top40 < 83.33
        + 0.01610863 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # +1.6%  girth2_top10 < 0.007679
        - 0.01264112 * max(0.0, 0.008956554 - Q.girth2_top10) / 0.004473389   # -1.3%  girth2_top10 < 0.008957
        + 0.01180794 * max(0.0, Q.log_sum_pt - 6.935549) / 0.0371538   # +1.2%  log_sum_pt > 6.936
        - 0.01064803 * max(0.0, 0.05936026 - Q.tau2) / 0.0302199   # -1.1%  tau2 < 0.05936
        + 0.0104214 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +1.0%  e2 < 0.0303
        + 0.01025648 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +1.0%  mass_top40 < 67.73
        + 0.00984227 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +1.0%  mass < 121.4
        + 0.009009009 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +0.9%  tau21_b2 < 0.3425
        - 0.008418436 * max(0.0, Q.n_pt_above_10 - 11.0) / 8.944079   # -0.8%  n_pt_above_10 > 11
        - 0.007109169 * max(0.0, Q.e2 - 0.02210818) / 0.01219843   # -0.7%  e2 > 0.02211
        - 0.006742423 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.02048524 - Q.lam1) / 0.001304679   # -0.7%  tau21_b2 < 0.3425 and lam1 < 0.02049
        + 0.006574006 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # +0.7%  girth2_top20 < 0.006374
        - 0.006052611 * max(0.0, 0.4226723 - Q.N2) / 0.1194948   # -0.6%  N2 < 0.4227
        - 0.005370681 * max(0.0, 7.876005e-05 - Q.e3) / 3.594075e-05   # -0.5%  e3 < 7.876e-05
        + 0.005098885 * max(0.0, 0.03680582 - Q.e2) / 0.01052402   # +0.5%  e2 < 0.03681
        + 0.004796483 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 1.059188 - Q.N3) / 0.002548597   # +0.5%  psi_0p3 > 0.9924 and N3 < 1.059
        + 0.004371329 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # +0.4%  sj3_mass1 < 32.5
        + 0.004328822 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.002405315 - Q.soft8_z) / 0.0166304   # +0.4%  mass < 91.03 and soft8_z < 0.002405
        + 0.004215886 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.01691783 - Q.C2_b2) / 0.001347445   # +0.4%  tau21_b2 < 0.3425 and C2_b2 < 0.01692
        - 0.003986258 * max(0.0, 19.0 - Q.n_dr_0p1_0p2) / 8.166005   # -0.4%  n_dr_0p1_0p2 < 19
        + 0.003367631 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +0.3%  psi_0p3 > 0.9924
        - 0.003131601 * max(0.0, 0.001595561 - Q.soft7_z) / 0.0003533115   # -0.3%  soft7_z < 0.001596
        + 0.002919962 * max(0.0, 0.001411894 - Q.lam2) / 0.0006681999   # +0.3%  lam2 < 0.001412
        - 0.002876488 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.007709916 - Q.girth2_top40) / 0.0001062984   # -0.3%  tau21_b2 < 0.3425 and girth2_top40 < 0.00771
        + 0.00261099 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 4.250195 - Q.soft6_pt) / 0.009807099   # +0.3%  psi_0p3 > 0.9924 and soft6_pt < 4.25
        + 0.002133956 * max(0.0, Q.e2 - 0.04358622) / 0.002786698   # +0.2%  e2 > 0.04359
        + 0.00209945 * max(0.0, 0.03948167 - Q.dr_3) / 0.0095916   # +0.2%  dr_3 < 0.03948
        + 0.001991866 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.34119 - Q.sj2_zsoft) / 0.009756021   # +0.2%  tau21_b2 < 0.3425 and sj2_zsoft < 0.3412
        + 0.001983564 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.05799644 - Q.z_6) / 0.001993324   # +0.2%  tau21_b2 < 0.3425 and z_6 < 0.058
        - 0.001699087 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.zdr_3 - 0.0003281655) / 0.08597332   # -0.2%  mass < 121.4 and zdr_3 > 0.0003282
        - 0.001695026 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.0458688 - Q.dr_3) / 4.832428e-05   # -0.2%  psi_0p3 > 0.9924 and dr_3 < 0.04587
        - 0.001483299 * max(0.0, 0.01083435 - Q.girth2_top20) * max(0.0, Q.mass_top3 - 16.89912) / 0.01539405   # -0.1%  girth2_top20 < 0.01083 and mass_top3 > 16.9
        + 0.00127364 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, Q.pt_3 - 60.59375) / 0.06871383   # +0.1%  psi_0p3 > 0.9924 and pt_3 > 60.59
        + 0.001184827 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.006118006 - Q.girth2_top50) / 2.65509e-05   # +0.1%  tau21_b2 < 0.3425 and girth2_top50 < 0.006118
        - 0.0006521604 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.1433879 - Q.sj2_zsoft) / 0.001093769   # -0.1%  tau21_b2 < 0.3425 and sj2_zsoft < 0.1434
        - 0.0006503677 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.n_for_50pct - 5.0) / 4.765654   # -0.1%  mass < 82.85 and n_for_50pct > 5
        - 0.0003480638 * max(0.0, 0.9586536 - Q.z_top50_slots) / 0.0006232493   # -0.0%  z_top50_slots < 0.9587
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 10.76;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.75779 * (-0.00806021
        + 0.08557592 * max(0.0, 0.140939 - Q.mass_over_sum_pt) / 0.05796036   # +8.6%  mass_over_sum_pt < 0.1409
        + 0.08420737 * max(0.0, 0.08068193 - Q.girth) / 0.02368455   # +8.4%  girth < 0.08068
        + 0.06620114 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +6.6%  log_sum_pt < 7.017
        - 0.05315021 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # -5.3%  e3 < 0.0005178
        - 0.05302634 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # -5.3%  LHA < 0.3024
        + 0.03951594 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +4.0%  girth2_top5 < 0.00833
        + 0.02966313 * max(0.0, 79.18312 - Q.sd_mass) / 30.46368   # +3.0%  sd_mass < 79.18
        + 0.02901763 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +2.9%  z_dr_0p1_0p2 < 0.1203
        + 0.02614784 * max(0.0, Q.sj3_dr_max - 0.2442262) / 0.04751597   # +2.6%  sj3_dr_max > 0.2442
        + 0.02411538 * max(0.0, Q.lam1 - 0.006189818) / 0.003311577   # +2.4%  lam1 > 0.00619
        - 0.02333222 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -2.3%  mass < 92.86
        + 0.02252245 * max(0.0, 80.24626 - Q.mass_top30) / 14.57104   # +2.3%  mass_top30 < 80.25
        - 0.02240065 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -2.2%  girth2_top30 < 0.01216
        - 0.02195889 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.2%  tau1 < 0.07057
        - 0.02170337 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # -2.2%  e2 > 0.0303
        - 0.02149435 * max(0.0, 0.07031778 - Q.girth) / 0.01719521   # -2.1%  girth < 0.07032
        - 0.02148049 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # -2.1%  lam1 < 0.01174
        - 0.01967156 * max(0.0, 0.08329945 - Q.mass_over_sum_pt) / 0.0135093   # -2.0%  mass_over_sum_pt < 0.0833
        + 0.01880554 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # +1.9%  girth2_top10 < 0.007679
        + 0.01849269 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +1.8%  lam2 < 0.001776
        - 0.01749822 * max(0.0, 1017.778 - Q.sum_pt_top20) / 107.0228   # -1.7%  sum_pt_top20 < 1018
        - 0.01728478 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -1.7%  psi_0p3 > 0.9897
        - 0.0158584 * max(0.0, Q.sj3_dr_max - 0.2841485) / 0.0316155   # -1.6%  sj3_dr_max > 0.2841
        - 0.01539486 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -1.5%  psi_0p1 > 0.8976
        - 0.01236594 * max(0.0, Q.sj3_dr_max - 0.1717401) / 0.08948791   # -1.2%  sj3_dr_max > 0.1717
        + 0.01165718 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) / 3.82104   # +1.2%  n_dr_0p1_0p2 < 13
        - 0.0113512 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -1.1%  sum_pt < 1002
        - 0.01073589 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # -1.1%  mass_top50 < 71.8
        - 0.01072105 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.n_real_top40 - 22.0) / 0.06106188   # -1.1%  girth2_top5 < 0.00833 and n_real_top40 > 22
        + 0.01059555 * max(0.0, 0.2091025 - Q.LHA) / 0.02095782   # +1.1%  LHA < 0.2091
        + 0.01028019 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +1.0%  lam1 < 0.004673
        - 0.009979972 * max(0.0, Q.sj2_dr - 0.1666442) / 0.04983259   # -1.0%  sj2_dr > 0.1666
        - 0.009702292 * max(0.0, 0.002270363 - Q.girth2_top5) / 0.0007634099   # -1.0%  girth2_top5 < 0.00227
        - 0.009677712 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 76.29481 - Q.sd_mass) / 2.433638   # -1.0%  z_dr_0p1_0p2 < 0.1203 and sd_mass < 76.29
        - 0.009023507 * max(0.0, 40.97891 - Q.sd_mass) / 11.60739   # -0.9%  sd_mass < 40.98
        - 0.008863431 * max(0.0, Q.lam1 - 0.01649354) / 0.001003295   # -0.9%  lam1 > 0.01649
        - 0.008074852 * max(0.0, 1018.832 - Q.sum_pt_top30) / 55.61816   # -0.8%  sum_pt_top30 < 1019
        + 0.007207519 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.7%  sj2_dr > 0.2232
        - 0.00668228 * max(0.0, 68.28697 - Q.mass_top30) / 9.510711   # -0.7%  mass_top30 < 68.29
        - 0.006225035 * max(0.0, 1.976207 - Q.D2) / 0.367701   # -0.6%  D2 < 1.976
        + 0.005815989 * max(0.0, 52.15154 - Q.mass_top50) / 3.332261   # +0.6%  mass_top50 < 52.15
        + 0.005727214 * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.01931881   # +0.6%  z_dr_0p1_0p2 < 0.06473
        - 0.004952588 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677) / 0.04548726   # -0.5%  girth2_top5 < 0.00833 and sj3_mass1 > 5.113
        - 0.004701579 * max(0.0, 0.001056655 - Q.girth2_top2) / 0.0003002514   # -0.5%  girth2_top2 < 0.001057
        - 0.004620383 * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 3.233647   # -0.5%  n_dr_0p2_0p4 > 9
        - 0.004572937 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -0.5%  log_sum_pt < 6.903
        - 0.004164785 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.02970886 - Q.absphi_0) / 0.06277088   # -0.4%  n_dr_0p1_0p2 < 13 and absphi_0 < 0.02971
        - 0.004013002 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # -0.4%  sum_pt_top40 < 1007
        - 0.003980448 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583) / 0.0009330603   # -0.4%  girth2_top5 < 0.00833 and sj3_pairmin_over_m > 0.09541
        + 0.003823307 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2) / 0.0004272674   # +0.4%  sj2_dr > 0.2232 and C2_b2 < 0.04009
        - 0.003799141 * max(0.0, Q.psi_0p1 - 0.9371031) / 0.01088804   # -0.4%  psi_0p1 > 0.9371
        - 0.003792996 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 14.45919 - Q.sj2_mass2) / 0.03398137   # -0.4%  psi_0p3 > 0.9897 and sj2_mass2 < 14.46
        + 0.003452954 * max(0.0, 0.2416266 - Q.sj2_zsoft) / 0.06095505   # +0.3%  sj2_zsoft < 0.2416
        + 0.003184391 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, 0.04770182 - Q.z_6) / 5.265762e-05   # +0.3%  girth2_top5 < 0.00833 and z_6 < 0.0477
        + 0.003077488 * max(0.0, 869.693 - Q.sum_pt_top20) / 29.2727   # +0.3%  sum_pt_top20 < 869.7
        - 0.002868776 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0) / 0.2039042   # -0.3%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p2_0p4 > 5
        + 0.002562106 * max(0.0, 0.00287991 - Q.girth2_top20) * max(0.0, 0.4357228 - Q.max_dr) / 4.319324e-05   # +0.3%  girth2_top20 < 0.00288 and max_dr < 0.4357
        + 0.002421627 * max(0.0, 0.0005124533 - Q.girth2_top2) / 0.0001104529   # +0.2%  girth2_top2 < 0.0005125
        - 0.002002119 * max(0.0, 0.2982 - Q.max_dr) / 0.01350087   # -0.2%  max_dr < 0.2982
        - 0.0015387 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, 0.948102 - Q.psi_0p2) / 3.786811e-05   # -0.2%  girth2_top5 < 0.00833 and psi_0p2 < 0.9481
        + 0.001010607 * max(0.0, 906.6023 - Q.sum_pt_top40) / 6.984569   # +0.1%  sum_pt_top40 < 906.6
        + 0.0009503613 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt) / 2.722316   # +0.1%  z_dr_0_0p05 < 0.8459 and sum_pt < 949.9
        + 0.0008643316 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.dr0_8 - 0.2441824) / 1.193182e-05   # +0.1%  girth2_top5 < 0.00833 and dr0_8 > 0.2442
        - 0.0004391965 * max(0.0, 0.002270363 - Q.girth2_top5) * max(0.0, Q.dr0_8 - 0.2441824) / 1.876332e-06   # -0.0%  girth2_top5 < 0.00227 and dr0_8 > 0.2442
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6463987920168067, 2.7584543592436974, 0.2113329831932773, 0.38055535714285715, 0.7673824579831933, 1.0328398109243697, 0.533767962184874, 0.4667276260504202, 1.1597211134453782, 0.9875698529411765, 1.2501849789915966, 0.7606677521008404, 0.5496060924369748, 1.5261159663865547, 0.26557883403361343, 0.4306641806722689]
T = [3.7808788865546212, 2.4479892758665964, 4.188886825433298, 4.28591722032563, 4.066018324087447]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +46%, n4 -18%, n9 +13%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.4559876 * h[1] / H_AVG[1]
            - 0.1775935 * h[4] / H_AVG[4]
            + 0.1346818 * h[9] / H_AVG[9]
            - 0.08536704 * h[5] / H_AVG[5]
            - 0.07548946 * h[3] / H_AVG[3]
            + 0.03406984 * h[12] / H_AVG[12]
            + 0.02205869 * h[6] / H_AVG[6]
            + 0.009585413 * h[8] / H_AVG[8]
            - 0.005166561 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +23%, n1 -21%, n11 -8%, n12 +8%, n6 +5% ...
            - 0.3330668 * h[4] / H_AVG[4]
            + 0.2269242 * h[9] / H_AVG[9]
            - 0.2112796 * h[1] / H_AVG[1]
            - 0.07768291 * h[11] / H_AVG[11]
            + 0.07717644 * h[12] / H_AVG[12]
            + 0.047697 * h[6] / H_AVG[6]
            + 0.01079115 * h[2] / H_AVG[2]
            - 0.007979667 * h[10] / H_AVG[10]
            + 0.007402256 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +14%, n0 +12%, n11 +11%, n14 -9%, n7 -7% ...
            - 0.2422496 * h[8] / H_AVG[8]
            + 0.1425463 * h[5] / H_AVG[5]
            + 0.1157346 * h[0] / H_AVG[0]
            + 0.1078202 * h[11] / H_AVG[11]
            - 0.08717612 * h[14] / H_AVG[14]
            - 0.06963778 * h[7] / H_AVG[7]
            + 0.06297323 * h[4] / H_AVG[4]
            - 0.05330234 * h[12] / H_AVG[12]
            - 0.05157239 * h[9] / H_AVG[9]
            + 0.03974635 * h[3] / H_AVG[3]
            - 0.01927709 * h[15] / H_AVG[15]
            + 0.007964048 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -25%, n0 -21%, n5 +17%, n6 -12%, n7 +10%, n15 +6% ...
            - 0.253677 * h[8] / H_AVG[8]
            - 0.2073765 * h[0] / H_AVG[0]
            + 0.1656769 * h[5] / H_AVG[5]
            - 0.1206481 * h[6] / H_AVG[6]
            + 0.09868877 * h[7] / H_AVG[7]
            + 0.056522 * h[15] / H_AVG[15]
            + 0.04007355 * h[12] / H_AVG[12]
            + 0.03884652 * h[3] / H_AVG[3]
            - 0.01849076 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +30%, n5 -12%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3401467 * h[13] / H_AVG[13]
            + 0.3026673 * h[10] / H_AVG[10]
            - 0.1190707 * h[5] / H_AVG[5]
            + 0.06016418 * h[8] / H_AVG[8]
            - 0.05068897 * h[12] / H_AVG[12]
            - 0.03971922 * h[15] / H_AVG[15]
            + 0.03538701 * h[4] / H_AVG[4]
            - 0.03228395 * h[7] / H_AVG[7]
            + 0.01987198 * h[0] / H_AVG[0]
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
