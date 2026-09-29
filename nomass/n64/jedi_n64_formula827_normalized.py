"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  1:  12.0%   (on for 93% of jets)
  neuron  5:  11.3%   (on for 80% of jets)
  neuron  4:  10.1%   (on for 68% of jets)
  neuron  0:   7.8%   (on for 61% of jets)
  neuron 13:   7.4%   (on for 88% of jets)
  neuron  9:   6.8%   (on for 66% of jets)
  neuron 10:   6.8%   (on for 81% of jets)
  neuron 12:   4.9%   (on for 58% of jets)
  neuron  7:   4.4%   (on for 27% of jets)
  neuron  6:   4.0%   (on for 66% of jets)
  neuron 11:   3.4%   (on for 70% of jets)
  neuron  3:   3.3%   (on for 51% of jets)
  neuron 15:   2.6%   (on for 67% of jets)
  neuron 14:   1.9%   (on for 34% of jets)
  neuron  2:   0.6%   (on for 41% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.5% (the network: 81.1%); same class as the network for 92.3% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top20           number of real particles among the 20 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_10                  pT of particle 10 [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_14                  pT of particle 14 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_8                pT8 · ΔR(0, 8) [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_pt              pT [GeV] of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_pt               pT [GeV] of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_pt               pT [GeV] of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_6                    pT of particle 6 / total pT
  Q.z_8                    pT of particle 8 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_z               pT share of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_z                pT share of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.soft10_abseta          |Δη| of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_2               |Δφ| of particle 2
  Q.absphi_3               |Δφ| of particle 3
  Q.absphi_5               |Δφ| of particle 5
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft2_dr0              ΔR between the hardest and the 2. softest real particle (0 if among the 15 hardest)
  Q.soft3_dr0              ΔR between the hardest and the 3. softest real particle (0 if among the 15 hardest)
  Q.soft4_dr0              ΔR between the hardest and the 4. softest real particle (0 if among the 15 hardest)
  Q.soft5_dr0              ΔR between the hardest and the 5. softest real particle (0 if among the 15 hardest)
  Q.soft8_dr0              ΔR between the hardest and the 8. softest real particle (0 if among the 15 hardest)
  Q.dr0_11                 ΔR between particle 11 and the hardest particle
  Q.dr0_12                 ΔR between particle 12 and the hardest particle
  Q.dr0_3                  ΔR between particle 3 and the hardest particle
  Q.dr0_4                  ΔR between particle 4 and the hardest particle
  Q.dr0_8                  ΔR between particle 8 and the hardest particle
  Q.dr1_12                 ΔR between particle 12 and the 2nd-hardest particle
  Q.dr1_9                  ΔR between particle 9 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_10                  ΔR of particle 10 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.soft2_dr               ΔR from the jet axis of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_dr               ΔR from the jet axis of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_dr               ΔR from the jet axis of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_dr               ΔR from the jet axis of the 5. softest real particle (0 if it is among the 15 hardest)
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
  Q.mean_eta2              pT-weighted mean Δη²
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
        z_top5=sum(zs[:5]),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        n_particles=len(real),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        n_real_top20=sum(1 for x in pt[:20] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_10=pt[10],
        pt_11=pt[11],
        pt_14=pt[14],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        ptdr0_8=pt[8] * math.sqrt(dist2(0, 8)) if pt[8] > 0 else 0.0,
        soft1_pt=softp(1, 'pt'),
        soft10_pt=softp(10, 'pt'),
        soft2_pt=softp(2, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        soft7_pt=softp(7, 'pt'),
        soft9_pt=softp(9, 'pt'),
        z_6=z[6],
        z_8=z[8],
        z_9=z[9],
        soft1_z=softp(1, 'z'),
        soft10_z=softp(10, 'z'),
        soft3_z=softp(3, 'z'),
        soft4_z=softp(4, 'z'),
        soft7_z=softp(7, 'z'),
        soft8_z=softp(8, 'z'),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        zdr_6=z[6] * dr[6],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        soft10_abseta=softp(10, 'abseta'),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_2=abs(phi[2]),
        absphi_3=abs(phi[3]),
        absphi_5=abs(phi[5]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft2_dr0=softp(2, 'dr0'),
        soft3_dr0=softp(3, 'dr0'),
        soft4_dr0=softp(4, 'dr0'),
        soft5_dr0=softp(5, 'dr0'),
        soft8_dr0=softp(8, 'dr0'),
        dr0_11=math.sqrt(dist2(0, 11)) if pt[11] > 0 else 0.0,
        dr0_12=math.sqrt(dist2(0, 12)) if pt[12] > 0 else 0.0,
        dr0_3=math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        dr0_4=math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        dr0_8=math.sqrt(dist2(0, 8)) if pt[8] > 0 else 0.0,
        dr1_12=math.sqrt(dist2(1, 12)) if pt[12] > 0 else 0.0,
        dr1_9=math.sqrt(dist2(1, 9)) if pt[9] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_10=dr[10] if pt[10] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        soft2_dr=softp(2, 'dr'),
        soft3_dr=softp(3, 'dr'),
        soft4_dr=softp(4, 'dr'),
        soft5_dr=softp(5, 'dr'),
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
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
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
    # scale S = 13.52;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.52172 * (-0.1065985
        + 0.1197472 * max(0.0, 0.007872294 - Q.e2_sq) / 0.002240658   # +12.0%  e2_sq < 0.007872
        + 0.08044512 * max(0.0, Q.sum_pt_top40 - 858.8262) / 166.7848   # +8.0%  sum_pt_top40 > 858.8
        - 0.06662962 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -6.7%  e2_sq < 0.006167
        + 0.06190631 * max(0.0, 0.006936725 - Q.e2_sq) / 0.001688614   # +6.2%  e2_sq < 0.006937
        - 0.05809841 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -5.8%  e2_sq < 0.009606
        + 0.05802439 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # +5.8%  girth2_top50 < 0.008125
        - 0.05057987 * max(0.0, Q.sum_pt - 1002.379) / 57.37467   # -5.1%  sum_pt > 1002
        + 0.04894856 * max(0.0, 1191.938 - Q.sum_pt_top30) / 206.0686   # +4.9%  sum_pt_top30 < 1192
        + 0.04334299 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +4.3%  girth2_top40 < 0.008841
        - 0.03601105 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -3.6%  log_sum_pt > 6.91
        - 0.03338341 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -3.3%  girth2_top30 < 0.006364
        - 0.02858391 * max(0.0, 0.008190222 - Q.girth2) * max(0.0, 0.006142802 - Q.girth2_top15) / 1.08234e-05   # -2.9%  girth2 < 0.00819 and girth2_top15 < 0.006143
        - 0.02617247 * max(0.0, 0.005852839 - Q.girth2_top50) * max(0.0, 1260.541 - Q.sum_pt) / 0.2467156   # -2.6%  girth2_top50 < 0.005853 and sum_pt < 1261
        + 0.02540361 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +2.5%  n_dr_0p2_0p4 < 18
        + 0.02453537 * max(0.0, 0.008190222 - Q.girth2) / 0.002454223   # +2.5%  girth2 < 0.00819
        + 0.02347861 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # +2.3%  girth2_top30 < 0.008376
        - 0.02302701 * max(0.0, 1073.473 - Q.sum_pt_top30) * max(0.0, 43.66338 - Q.D2_b2) / 3778.139   # -2.3%  sum_pt_top30 < 1073 and D2_b2 < 43.66
        - 0.02141999 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.1%  tau1 < 0.07057
        - 0.02087167 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -2.1%  psi_0p2 > 0.9087
        + 0.01434722 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.0001086251 - Q.e3) / 3.665746e-06   # +1.4%  log_sum_pt > 6.91 and e3 < 0.0001086
        - 0.01286807 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # -1.3%  girth2_top40 < 0.00626
        + 0.01067019 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # +1.1%  psi_0p3 > 0.9943
        + 0.008351103 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +0.8%  e2 < 0.01879
        + 0.008101971 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # +0.8%  sum_pt > 1116
        - 0.007754153 * max(0.0, Q.z_top20_slots - 0.8474481) / 0.06506311   # -0.8%  z_top20_slots > 0.8474
        - 0.007478872 * max(0.0, 0.005852839 - Q.girth2_top50) / 0.001204465   # -0.7%  girth2_top50 < 0.005853
        - 0.007306128 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -0.7%  lam1 < 0.005914
        + 0.007164405 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.01204531 - Q.mean_eta2) / 0.0004526298   # +0.7%  log_sum_pt > 6.91 and mean_eta2 < 0.01205
        - 0.006661342 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -0.7%  sum_pt_top40 > 1025
        + 0.006326785 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.932092   # +0.6%  n_dr_0p2_0p4 < 13
        - 0.00570696 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.log_sum_pt - 6.915514) / 0.4823065   # -0.6%  n_dr_0p2_0p4 < 18 and log_sum_pt > 6.916
        - 0.005608854 * max(0.0, 0.05602756 - Q.C2) / 0.009882647   # -0.6%  C2 < 0.05603
        + 0.005192246 * max(0.0, Q.sum_pt - 1167.447) / 14.21193   # +0.5%  sum_pt > 1167
        - 0.004541388 * max(0.0, Q.sum_pt_top20 - 1064.139) / 11.11947   # -0.5%  sum_pt_top20 > 1064
        + 0.004061648 * max(0.0, Q.sum_pt_top20 - 1035.416) / 14.61631   # +0.4%  sum_pt_top20 > 1035
        - 0.004033656 * max(0.0, 0.003113918 - Q.girth2_top40) / 0.0004182435   # -0.4%  girth2_top40 < 0.003114
        + 0.003504088 * max(0.0, 0.2708235 - Q.tau21_b2) / 0.06922804   # +0.4%  tau21_b2 < 0.2708
        - 0.003211115 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # -0.3%  sj2_dr > 0.2232
        + 0.003199636 * max(0.0, Q.sum_pt_top30 - 950.9324) / 66.97298   # +0.3%  sum_pt_top30 > 950.9
        - 0.003001779 * max(0.0, 1073.473 - Q.sum_pt_top30) / 97.85231   # -0.3%  sum_pt_top30 < 1073
        + 0.002384526 * max(0.0, Q.sum_pt_top10 - 845.5367) / 35.97281   # +0.2%  sum_pt_top10 > 845.5
        - 0.002077267 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # -0.2%  tau1 < 0.06311
        + 0.002052834 * max(0.0, 0.002270363 - Q.girth2_top5) * max(0.0, 0.0216878 - Q.C2_b2) / 7.233634e-06   # +0.2%  girth2_top5 < 0.00227 and C2_b2 < 0.02169
        + 0.001386224 * max(0.0, 1073.473 - Q.sum_pt_top30) * max(0.0, 0.0009793444 - Q.mean_eta2) / 0.006393901   # +0.1%  sum_pt_top30 < 1073 and mean_eta2 < 0.0009793
        - 0.001327585 * max(0.0, 0.001535432 - Q.girth2_top40) / 9.698891e-05   # -0.1%  girth2_top40 < 0.001535
        - 0.0008344386 * max(0.0, 613.7 - Q.sum_pt_top15) / 2.976772   # -0.1%  sum_pt_top15 < 613.7
        - 0.0002360149 * max(0.0, 613.7 - Q.sum_pt_top15) * max(0.0, 0.009867229 - Q.z_9) / 1.550576e-05   # -0.0%  sum_pt_top15 < 613.7 and z_9 < 0.009867
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 22.77;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.76931 * (0.0431523
        + 0.07996002 * max(0.0, Q.sum_pt - 972.0419) / 81.34075   # +8.0%  sum_pt > 972
        + 0.07962097 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +8.0%  log_sum_pt > 6.91
        - 0.07852328 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -7.9%  log_sum_pt < 7.139
        + 0.06418782 * max(0.0, 0.1953848 - Q.tau1) / 0.1097382   # +6.4%  tau1 < 0.1954
        + 0.05049876 * max(0.0, 1225.842 - Q.sum_pt_top40) / 211.117   # +5.0%  sum_pt_top40 < 1226
        - 0.03974124 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # -4.0%  sum_pt_top50 > 934.2
        + 0.03879199 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # +3.9%  sum_pt_top50 < 1079
        - 0.03508851 * max(0.0, 2.275391 - Q.soft1_pt) / 1.652278   # -3.5%  soft1_pt < 2.275
        - 0.03411515 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -3.4%  e2_sq < 0.009606
        + 0.03247195 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +3.2%  n_particles > 38
        - 0.03156702 * max(0.0, 0.01397874 - Q.girth2) / 0.006902497   # -3.2%  girth2 < 0.01398
        - 0.03101089 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -3.1%  sum_pt_top50 > 959.1
        - 0.02956095 * max(0.0, Q.tau1 - 0.02607472) / 0.06189568   # -3.0%  tau1 > 0.02607
        + 0.02683821 * max(0.0, 0.001452174 - Q.soft1_z) / 0.0009045584   # +2.7%  soft1_z < 0.001452
        - 0.02356608 * max(0.0, 1167.447 - Q.sum_pt) / 137.8628   # -2.4%  sum_pt < 1167
        - 0.02152317 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -2.2%  log_sum_pt > 6.989
        + 0.0199778 * max(0.0, 0.007463985 - Q.girth2_top30) / 0.00233708   # +2.0%  girth2_top30 < 0.007464
        + 0.01971592 * max(0.0, 605.875 - Q.sum_pt_top2) / 245.9717   # +2.0%  sum_pt_top2 < 605.9
        + 0.01629135 * max(0.0, Q.e2 - 0.01036127) / 0.02094596   # +1.6%  e2 > 0.01036
        - 0.01612776 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -1.6%  sum_pt > 1053
        - 0.01400753 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.002181998 - Q.soft1_z) / 0.01445558   # -1.4%  n_particles > 38 and soft1_z < 0.002182
        + 0.01110285 * max(0.0, Q.z_top30_slots - 0.9564984) / 0.01947087   # +1.1%  z_top30_slots > 0.9565
        - 0.009490727 * max(0.0, 0.3845054 - Q.soft2_dr) / 0.1840621   # -0.9%  soft2_dr < 0.3845
        + 0.007833667 * max(0.0, 0.4086381 - Q.soft2_dr0) / 0.1958521   # +0.8%  soft2_dr0 < 0.4086
        - 0.007209116 * max(0.0, 0.7058597 - Q.tau21_b2) / 0.3912084   # -0.7%  tau21_b2 < 0.7059
        + 0.006971314 * max(0.0, 1038.262 - Q.sum_pt_top30) / 69.34981   # +0.7%  sum_pt_top30 < 1038
        + 0.00691869 * max(0.0, 0.5100475 - Q.tau21) / 0.1343519   # +0.7%  tau21 < 0.51
        + 0.006637882 * max(0.0, 0.005240342 - Q.lam1) / 0.001165583   # +0.7%  lam1 < 0.00524
        - 0.006590715 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -0.7%  n_dr_0p2_0p4 < 7
        - 0.006241787 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # -0.6%  lam2 < 0.001776
        - 0.006229858 * max(0.0, 0.03457336 - Q.M3) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.0006031662   # -0.6%  M3 < 0.03457 and psi_0p3 > 0.9299
        - 0.005933162 * max(0.0, Q.z_top5 - 0.4670285) / 0.1261617   # -0.6%  z_top5 > 0.467
        - 0.005672457 * max(0.0, 29.04688 - Q.pt_11) / 8.607263   # -0.6%  pt_11 < 29.05
        - 0.005584612 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # -0.6%  psi_0p3 > 0.9966
        - 0.0054707 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) * max(0.0, 10.0 - Q.n_dr_0p05_0p1) / 0.2065903   # -0.5%  z_dr_0_0p05 > 0.8103 and n_dr_0p05_0p1 < 10
        - 0.005303886 * max(0.0, 0.02017767 - Q.tau4) / 0.005429427   # -0.5%  tau4 < 0.02018
        + 0.005078542 * max(0.0, 2.117188 - Q.soft5_pt) / 0.8827838   # +0.5%  soft5_pt < 2.117
        - 0.004984048 * max(0.0, 605.875 - Q.sum_pt_top2) * max(0.0, 0.3797 - Q.soft3_dr) / 46.98971   # -0.5%  sum_pt_top2 < 605.9 and soft3_dr < 0.3797
        + 0.00491446 * max(0.0, Q.tau1 - 0.1219132) / 0.01033631   # +0.5%  tau1 > 0.1219
        - 0.004879315 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, 10.17512 - Q.ptdr0_3) / 0.1427049   # -0.5%  z_top30_slots > 0.9565 and ptdr0_3 < 10.18
        + 0.004705748 * max(0.0, Q.n_particles - 38.0) * max(0.0, Q.tau32 - 0.3293142) / 3.693221   # +0.5%  n_particles > 38 and tau32 > 0.3293
        + 0.004371303 * max(0.0, 0.05602756 - Q.C2) / 0.009882647   # +0.4%  C2 < 0.05603
        + 0.004171477 * max(0.0, 0.0006570502 - Q.girth2_top5) / 0.0001395474   # +0.4%  girth2_top5 < 0.0006571
        + 0.004122129 * max(0.0, Q.n_dr_0_0p05 - 15.0) / 2.713461   # +0.4%  n_dr_0_0p05 > 15
        - 0.004090456 * max(0.0, Q.z_top5 - 0.6551948) * max(0.0, 28.39396 - Q.pt1_dr01) / 0.8332473   # -0.4%  z_top5 > 0.6552 and pt1_dr01 < 28.39
        + 0.00391634 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # +0.4%  LHA < 0.2454
        + 0.003825329 * max(0.0, 2.178951 - Q.D2) / 0.4648731   # +0.4%  D2 < 2.179
        + 0.003495373 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.009970338 - Q.zdr_0) / 0.03317992   # +0.3%  n_particles > 38 and zdr_0 < 0.00997
        + 0.003487245 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1178619 - Q.absphi_1) / 0.8192292   # +0.3%  n_particles > 38 and absphi_1 < 0.1179
        - 0.003168193 * max(0.0, 0.02210818 - Q.e2) / 0.00342102   # -0.3%  e2 < 0.02211
        - 0.003019289 * max(0.0, 3.852812 - Q.D2_b2) / 1.876829   # -0.3%  D2_b2 < 3.853
        - 0.003010195 * max(0.0, 605.875 - Q.sum_pt_top2) * max(0.0, Q.soft3_dr0 - 0.02918107) / 44.00985   # -0.3%  sum_pt_top2 < 605.9 and soft3_dr0 > 0.02918
        - 0.002887565 * max(0.0, 0.8495689 - Q.D2_b2) / 0.1691494   # -0.3%  D2_b2 < 0.8496
        - 0.002867147 * max(0.0, 7.0 - Q.n_dr_0p1_0p2) / 0.9740454   # -0.3%  n_dr_0p1_0p2 < 7
        - 0.002817407 * max(0.0, 0.0007143144 - Q.lam2) / 0.0002017367   # -0.3%  lam2 < 0.0007143
        + 0.002360737 * max(0.0, 0.002197765 - Q.girth2_top15) / 0.0004509993   # +0.2%  girth2_top15 < 0.002198
        + 0.002240985 * max(0.0, Q.psi_0p3 - 0.9638082) / 0.02792824   # +0.2%  psi_0p3 > 0.9638
        + 0.002104327 * max(0.0, 0.001153063 - Q.girth2_top20) / 0.0001200862   # +0.2%  girth2_top20 < 0.001153
        - 0.002054686 * max(0.0, 0.001784491 - Q.girth2) / 0.0001206749   # -0.2%  girth2 < 0.001784
        + 0.001990667 * max(0.0, 0.2140276 - Q.tau21) / 0.01160104   # +0.2%  tau21 < 0.214
        + 0.001974107 * max(0.0, 0.004763596 - Q.girth2_top30) / 0.0009958273   # +0.2%  girth2_top30 < 0.004764
        - 0.00190159 * max(0.0, 0.03044378 - Q.z_8) / 0.004402513   # -0.2%  z_8 < 0.03044
        + 0.001789296 * max(0.0, 2.5 - Q.soft9_pt) * max(0.0, 0.1156062 - Q.dr_13) / 0.03287379   # +0.2%  soft9_pt < 2.5 and dr_13 < 0.1156
        + 0.00173734 * max(0.0, 0.002197765 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9966167) / 5.451807e-07   # +0.2%  girth2_top15 < 0.002198 and psi_0p3 > 0.9966
        + 0.001697517 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, 0.1847499 - Q.mratio_min_012) / 0.001104779   # +0.2%  z_top30_slots > 0.9565 and mratio_min_012 < 0.1847
        + 0.001695787 * max(0.0, Q.z_top5 - 0.6551948) / 0.03331427   # +0.2%  z_top5 > 0.6552
        - 0.001638575 * max(0.0, Q.zdr_0 - 0.01240028) / 0.002210186   # -0.2%  zdr_0 > 0.0124
        + 0.001580993 * max(0.0, 1.409617 - Q.D2) / 0.1519846   # +0.2%  D2 < 1.41
        + 0.0015706 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.pt1_dr01 - 5.351077) / 0.1255531   # +0.2%  z_top30_slots > 0.9342 and pt1_dr01 > 5.351
        - 0.001512479 * max(0.0, 3.852812 - Q.D2_b2) * max(0.0, 0.3176645 - Q.soft4_dr) / 0.2843672   # -0.2%  D2_b2 < 3.853 and soft4_dr < 0.3177
        + 0.001478555 * max(0.0, 0.03962283 - Q.D3) / 0.008603272   # +0.1%  D3 < 0.03962
        + 0.001456775 * max(0.0, 29.04688 - Q.pt_11) * max(0.0, 0.1745719 - Q.dr1_12) / 0.7333922   # +0.1%  pt_11 < 29.05 and dr1_12 < 0.1746
        - 0.001316443 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.9624339 - Q.tau43) / 1.418537   # -0.1%  n_particles > 38 and tau43 < 0.9624
        - 0.001169355 * max(0.0, 0.2773918 - Q.D2_b2) / 0.01604103   # -0.1%  D2_b2 < 0.2774
        + 0.001106125 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_4 - 2.381691) / 0.0768104   # +0.1%  z_top30_slots > 0.9342 and ptdr0_4 > 2.382
        + 0.001100939 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_2 - 7.740999) / 0.07370506   # +0.1%  z_top30_slots > 0.9342 and ptdr0_2 > 7.741
        + 0.0008596521 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, Q.ptdr0_5 - 0.4637912) / 0.04806377   # +0.1%  z_top30_slots > 0.9565 and ptdr0_5 > 0.4638
        + 0.000825431 * max(0.0, Q.n_dr_0p05_0p1 - 17.0) / 1.571504   # +0.1%  n_dr_0p05_0p1 > 17
        + 0.000537148 * max(0.0, 0.03044378 - Q.z_8) * max(0.0, 0.002968979 - Q.soft10_abseta) / 3.114943e-06   # +0.1%  z_8 < 0.03044 and soft10_abseta < 0.002969
        - 0.000504222 * max(0.0, Q.n_dr_0_0p05 - 30.0) / 0.212479   # -0.1%  n_dr_0_0p05 > 30
        - 0.0004789528 * max(0.0, Q.n_dr_0_0p05 - 15.0) * max(0.0, 0.002292633 - Q.eta_1) / 0.0227646   # -0.0%  n_dr_0_0p05 > 15 and eta_1 < 0.002293
        - 0.0004200564 * max(0.0, Q.N2 - 0.4678622) / 0.001556028   # -0.0%  N2 > 0.4679
        - 0.0003921825 * max(0.0, Q.n_pt_above_5 - 46.0) / 0.2726588   # -0.0%  n_pt_above_5 > 46
        + 0.0003151265 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) * max(0.0, 20.0 - Q.n_real_top20) / 0.01239824   # +0.0%  z_dr_0_0p05 > 0.8103 and n_real_top20 < 20
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 2.852;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.851744 * (-0.08047665
        - 0.1457709 * max(0.0, 0.006941794 - Q.girth2) / 0.001690424   # -14.6%  girth2 < 0.006942
        + 0.1277997 * max(0.0, Q.sum_pt - 1007.788) / 53.71957   # +12.8%  sum_pt > 1008
        + 0.1192034 * max(0.0, Q.log_sum_pt - 6.903423) / 0.05662275   # +11.9%  log_sum_pt > 6.903
        - 0.1124323 * max(0.0, Q.sum_pt_top50 - 1003.544) / 52.17558   # -11.2%  sum_pt_top50 > 1004
        + 0.08377406 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +8.4%  lam1 < 0.01174
        + 0.08220399 * max(0.0, 0.00634935 - Q.girth2_top50) / 0.001425305   # +8.2%  girth2_top50 < 0.006349
        - 0.04673589 * max(0.0, Q.sum_pt - 1085.125) / 25.7173   # -4.7%  sum_pt > 1085
        + 0.0408093 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +4.1%  sum_pt_top40 > 1070
        - 0.03482272 * max(0.0, Q.sum_pt_top30 - 1011.524) / 32.30985   # -3.5%  sum_pt_top30 > 1012
        + 0.03313505 * max(0.0, Q.log_sum_pt - 6.930088) / 0.0397436   # +3.3%  log_sum_pt > 6.93
        + 0.02860641 * max(0.0, Q.log_sum_pt - 6.930088) * max(0.0, Q.z_top15_slots - 0.6655806) / 0.006105824   # +2.9%  log_sum_pt > 6.93 and z_top15_slots > 0.6656
        - 0.02208888 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -2.2%  psi_0p3 > 0.9943
        - 0.01863246 * max(0.0, Q.log_sum_pt - 6.930088) * max(0.0, Q.girth2_top30 - 0.00375223) / 0.0001399256   # -1.9%  log_sum_pt > 6.93 and girth2_top30 > 0.003752
        + 0.01715921 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15) / 0.0009227558   # +1.7%  log_sum_pt > 6.903 and girth2_top15 < 0.02147
        - 0.01377825 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -1.4%  log_sum_pt > 7.063
        - 0.01158864 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.8747961 - Q.psi_0p1) / 0.005466701   # -1.2%  log_sum_pt > 6.903 and psi_0p1 < 0.8748
        - 0.01056741 * max(0.0, Q.sum_pt_top20 - 993.5664) * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 1.779338   # -1.1%  sum_pt_top20 > 993.6 and z_dr_0p1_0p2 < 0.1203
        - 0.010431 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # -1.0%  log_sum_pt > 7.017
        - 0.009141121 * max(0.0, Q.sum_pt_top15 - 1003.329) / 12.91783   # -0.9%  sum_pt_top15 > 1003
        - 0.00792866 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.00781909 - Q.zdr_3) / 0.0002719899   # -0.8%  log_sum_pt > 6.903 and zdr_3 < 0.007819
        + 0.007519752 * max(0.0, Q.sum_pt_top30 - 1111.245) * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.9443349   # +0.8%  sum_pt_top30 > 1111 and z_dr_0p1_0p2 < 0.1203
        + 0.004655625 * max(0.0, Q.sum_pt_top20 - 1064.139) / 11.11947   # +0.5%  sum_pt_top20 > 1064
        + 0.004165079 * max(0.0, Q.sum_pt_top50 - 1003.544) * max(0.0, Q.z_dr_0p1_0p2 - 0.003353111) / 6.148304   # +0.4%  sum_pt_top50 > 1004 and z_dr_0p1_0p2 > 0.003353
        + 0.004145079 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 1.0 - Q.psi_0p3) / 0.0005826929   # +0.4%  log_sum_pt > 6.903 and psi_0p3 < 1
        + 0.002905091 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.girth2_top30 - 0.01215787) / 1.24419e-05   # +0.3%  log_sum_pt > 7.063 and girth2_top30 > 0.01216
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 2.474;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.474157 * (-0.04790822
        + 0.1122486 * max(0.0, 0.007511864 - Q.e2_sq) / 0.002016671   # +11.2%  e2_sq < 0.007512
        + 0.1069434 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +10.7%  girth2_top40 < 0.008841
        + 0.1057013 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +10.6%  n_particles < 46
        - 0.06736623 * max(0.0, 0.3719813 - Q.LHA) / 0.1170757   # -6.7%  LHA < 0.372
        + 0.06105242 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +6.1%  n_dr_0p2_0p4 < 5
        - 0.05558656 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # -5.6%  girth2 < 0.009615
        + 0.04764527 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.985099) / 0.03403522   # +4.8%  n_dr_0p2_0p4 < 8 and z_top50_slots > 0.9851
        - 0.04762558 * max(0.0, 0.347196 - Q.tau21) / 0.05239131   # -4.8%  tau21 < 0.3472
        - 0.04595439 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # -4.6%  girth2_top40 < 0.00626
        - 0.03568258 * max(0.0, 0.02647293 - Q.z_dr_0p2_0p4) / 0.01168948   # -3.6%  z_dr_0p2_0p4 < 0.02647
        + 0.03337937 * max(0.0, 0.2018786 - Q.tau21_b2) / 0.03799024   # +3.3%  tau21_b2 < 0.2019
        + 0.03210768 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +3.2%  lam2 < 0.0006155
        + 0.03184998 * max(0.0, 15.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.9313699) / 0.2389701   # +3.2%  n_dr_0p1_0p2 < 15 and psi_0p2 > 0.9314
        - 0.02974658 * max(0.0, 25.0 - Q.n_dr_0_0p05) / 13.07341   # -3.0%  n_dr_0_0p05 < 25
        - 0.02403573 * max(0.0, Q.psi_0p1 - 0.8509811) / 0.04568859   # -2.4%  psi_0p1 > 0.851
        - 0.02327556 * max(0.0, 40.0 - Q.n_pt_above_1) / 5.337714   # -2.3%  n_pt_above_1 < 40
        - 0.01843771 * max(0.0, 0.3719813 - Q.LHA) * max(0.0, Q.eta_1 - -0.04302979) / 0.005200433   # -1.8%  LHA < 0.372 and eta_1 > -0.04303
        - 0.01839878 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.e2 - 0.01036127) / 0.09655312   # -1.8%  n_particles < 46 and e2 > 0.01036
        - 0.01834592 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -1.8%  girth2_top30 < 0.006364
        - 0.01601161 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03029714 - Q.e2) / 0.006086253   # -1.6%  n_dr_0p2_0p4 < 5 and e2 < 0.0303
        + 0.01534764 * max(0.0, 1.788105 - Q.D2) / 0.2865857   # +1.5%  D2 < 1.788
        + 0.01176958 * max(0.0, Q.z_dr_0_0p05 - 0.7674734) / 0.05221088   # +1.2%  z_dr_0_0p05 > 0.7675
        + 0.01038959 * max(0.0, Q.psi_0p2 - 0.9976427) / 0.0003015943   # +1.0%  psi_0p2 > 0.9976
        + 0.009398475 * max(0.0, 0.2738063 - Q.max_dr) * max(0.0, Q.z_top50_slots - 0.9586536) / 0.0003528057   # +0.9%  max_dr < 0.2738 and z_top50_slots > 0.9587
        - 0.009195813 * max(0.0, 0.007511864 - Q.e2_sq) * max(0.0, 1024.942 - Q.sum_pt_top40) / 0.04926444   # -0.9%  e2_sq < 0.007512 and sum_pt_top40 < 1025
        - 0.007166955 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0) / 3.186294   # -0.7%  n_dr_0p2_0p4 < 5 and n_dr_0p1_0p2 > 9
        - 0.005336665 * max(0.0, 1.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.n_dr_0p4_up) / 0.05515294   # -0.5%  n_dr_0p2_0p4 < 1 and n_dr_0p4_up < 1
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 12.51;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.51213 * (-0.008769588
        + 0.07307403 * max(0.0, Q.girth2_top50 - 0.004573744) / 0.00533432   # +7.3%  girth2_top50 > 0.004574
        + 0.07271793 * max(0.0, Q.girth2_top15 - 0.002197765) / 0.005632841   # +7.3%  girth2_top15 > 0.002198
        + 0.06590333 * max(0.0, 0.009614971 - Q.width) / 0.003502545   # +6.6%  width < 0.009615
        - 0.05678795 * max(0.0, Q.girth2_top50 - 0.00242543) / 0.006946092   # -5.7%  girth2_top50 > 0.002425
        + 0.04461477 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +4.5%  n_dr_0p2_0p4 < 26
        + 0.04166726 * max(0.0, 0.01563836 - Q.girth2_top15) / 0.009593305   # +4.2%  girth2_top15 < 0.01564
        - 0.03847693 * max(0.0, 0.00592208 - Q.e2_sq) / 0.001192334   # -3.8%  e2_sq < 0.005922
        + 0.03794366 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # +3.8%  girth2_top20 < 0.01083
        - 0.03573685 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -3.6%  e2 < 0.04359
        - 0.03337558 * max(0.0, 1061.184 - Q.sum_pt_top50) / 53.8984   # -3.3%  sum_pt_top50 < 1061
        - 0.03250509 * max(0.0, Q.girth2_top15 - 0.004169954) / 0.004340368   # -3.3%  girth2_top15 > 0.00417
        + 0.03046321 * max(0.0, Q.girth - 0.076787) / 0.01351025   # +3.0%  girth > 0.07679
        - 0.02793474 * max(0.0, 0.008031986 - Q.girth2_top40) / 0.002514593   # -2.8%  girth2_top40 < 0.008032
        + 0.02775886 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +2.8%  sum_pt_top40 < 1053
        - 0.02523537 * max(0.0, Q.LHA - 0.2454112) / 0.05012361   # -2.5%  LHA > 0.2454
        - 0.02377754 * max(0.0, 0.007538019 - Q.girth2_top20) / 0.002740841   # -2.4%  girth2_top20 < 0.007538
        - 0.02230204 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -2.2%  lam1 < 0.006717
        - 0.02209658 * max(0.0, Q.n_particles - 29.0) / 17.61555   # -2.2%  n_particles > 29
        - 0.02175 * max(0.0, Q.girth2_top15 - 0.007887677) / 0.002755219   # -2.2%  girth2_top15 > 0.007888
        - 0.01880186 * max(0.0, Q.psi_0p1 - 0.3628388) / 0.4305572   # -1.9%  psi_0p1 > 0.3628
        - 0.01527914 * max(0.0, Q.LHA - 0.2845608) / 0.02922319   # -1.5%  LHA > 0.2846
        + 0.01451808 * max(0.0, 0.01397874 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.0001260069   # +1.5%  girth2 < 0.01398 and psi_0p3 > 0.9777
        + 0.01334917 * max(0.0, 0.01397874 - Q.girth2) / 0.006902497   # +1.3%  girth2 < 0.01398
        + 0.01321713 * max(0.0, Q.LHA - 0.1870291) / 0.08997036   # +1.3%  LHA > 0.187
        - 0.01270345 * max(0.0, Q.soft10_z - 0.0007540821) / 0.001847761   # -1.3%  soft10_z > 0.0007541
        + 0.01243759 * max(0.0, Q.sj2_dr - 0.1512157) / 0.05988199   # +1.2%  sj2_dr > 0.1512
        + 0.01142569 * max(0.0, Q.n_particles - 29.0) * max(0.0, 2.275391 - Q.soft1_pt) / 26.03979   # +1.1%  n_particles > 29 and soft1_pt < 2.275
        - 0.01119442 * max(0.0, Q.sj2_dr - 0.1825048) / 0.04110551   # -1.1%  sj2_dr > 0.1825
        + 0.01092499 * max(0.0, Q.LHA - 0.1870291) * max(0.0, Q.sj3_dr_max - 0.07466995) / 0.01953828   # +1.1%  LHA > 0.187 and sj3_dr_max > 0.07467
        - 0.01082331 * max(0.0, 0.1011219 - Q.tau2) / 0.06797109   # -1.1%  tau2 < 0.1011
        + 0.01052457 * max(0.0, 0.03263075 - Q.e2) / 0.008034085   # +1.1%  e2 < 0.03263
        + 0.009906297 * max(0.0, Q.soft10_pt - 1.603516) / 1.270676   # +1.0%  soft10_pt > 1.604
        - 0.008897125 * max(0.0, Q.n_pt_above_1 - 23.0) / 19.0629   # -0.9%  n_pt_above_1 > 23
        + 0.007331936 * max(0.0, 26.0 - Q.n_dr_0p1_0p2) / 14.20854   # +0.7%  n_dr_0p1_0p2 < 26
        - 0.006910101 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # -0.7%  z_dr_0p2_0p4 < 0.06849
        + 0.006185764 * max(0.0, Q.n_dr_0_0p05 - 10.0) / 5.012778   # +0.6%  n_dr_0_0p05 > 10
        + 0.005631287 * max(0.0, 0.005712208 - Q.girth2_top40) / 0.001219686   # +0.6%  girth2_top40 < 0.005712
        - 0.005610561 * max(0.0, Q.n_particles - 29.0) * max(0.0, Q.eccentricity - 0.3346888) / 7.265231   # -0.6%  n_particles > 29 and eccentricity > 0.3347
        - 0.004961716 * max(0.0, 9.0 - Q.n_for_50pct) / 4.104077   # -0.5%  n_for_50pct < 9
        - 0.00488564 * Q.zdr_0 / 0.009758003   # -0.5%  zdr_0
        + 0.004645778 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.sum_pt_top40 - 858.8262) / 0.172906   # +0.5%  psi_0p3 > 0.9974 and sum_pt_top40 > 858.8
        + 0.003740711 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.4%  psi_0p3 > 0.9974
        - 0.003546826 * max(0.0, Q.n_particles - 29.0) * max(0.0, 1.789258 - Q.soft4_pt) / 13.12058   # -0.4%  n_particles > 29 and soft4_pt < 1.789
        + 0.003305942 * max(0.0, 0.01397874 - Q.girth2) * max(0.0, Q.eccentricity - 0.8903081) / 0.0001316174   # +0.3%  girth2 < 0.01398 and eccentricity > 0.8903
        - 0.003296016 * max(0.0, 0.985099 - Q.z_top50_slots) / 0.003556075   # -0.3%  z_top50_slots < 0.9851
        - 0.003179109 * max(0.0, Q.LHA - 0.3719813) * max(0.0, Q.sj3_dr_max - 0.2268922) / 0.00109963   # -0.3%  LHA > 0.372 and sj3_dr_max > 0.2269
        - 0.003024298 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # -0.3%  C2 > 0.06656
        - 0.002715503 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # -0.3%  LHA > 0.3332
        - 0.002543373 * max(0.0, 0.009614971 - Q.width) * max(0.0, Q.soft10_z - 0.0007540821) / 5.992084e-06   # -0.3%  width < 0.009615 and soft10_z > 0.0007541
        - 0.00235538 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 13.0) / 43.30584   # -0.2%  n_dr_0p2_0p4 < 26 and n_dr_0p1_0p2 > 13
        - 0.002086251 * max(0.0, 0.001163277 - Q.lam2) * max(0.0, Q.z_dr_0p05_0p1 - 0.1514163) / 8.963301e-05   # -0.2%  lam2 < 0.001163 and z_dr_0p05_0p1 > 0.1514
        - 0.00205648 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.max_dr - 0.1939977) / 0.0001018378   # -0.2%  psi_0p3 > 0.9974 and max_dr > 0.194
        + 0.002046753 * max(0.0, Q.z_dr_0p05_0p1 - 0.4026646) / 0.08060812   # +0.2%  z_dr_0p05_0p1 > 0.4027
        + 0.001929466 * max(0.0, Q.sj2_dr - 0.2595052) * max(0.0, Q.soft4_pt - 0.6098633) / 0.009871605   # +0.2%  sj2_dr > 0.2595 and soft4_pt > 0.6099
        - 0.001813983 * max(0.0, 0.03263075 - Q.e2) * max(0.0, 1013.916 - Q.sum_pt_top50) / 0.1628293   # -0.2%  e2 < 0.03263 and sum_pt_top50 < 1014
        - 0.001425884 * max(0.0, Q.lam1 - 0.02048524) / 0.0005048405   # -0.1%  lam1 > 0.02049
        - 0.001388889 * max(0.0, Q.sj2_dr - 0.2595052) * max(0.0, 0.0329485 - Q.C2_b2) / 0.0001569167   # -0.1%  sj2_dr > 0.2595 and C2_b2 < 0.03295
        - 0.0012794 * max(0.0, Q.sj2_dr - 0.2595052) * max(0.0, Q.soft4_z - 0.0006393354) / 9.274593e-06   # -0.1%  sj2_dr > 0.2595 and soft4_z > 0.0006393
        + 0.001255575 * max(0.0, Q.e2_sq - 0.02580859) / 0.0004635665   # +0.1%  e2_sq > 0.02581
        + 0.001242082 * max(0.0, Q.n_particles - 29.0) * max(0.0, Q.e3 - 0.0001841806) / 0.001190643   # +0.1%  n_particles > 29 and e3 > 0.0001842
        - 0.0008345868 * max(0.0, Q.D2 - 5.378975) / 0.5670303   # -0.1%  D2 > 5.379
        - 0.0006461469 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # -0.1%  e2 > 0.06524
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 15.25;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.24582 * (0.05740136
        - 0.1007946 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -10.1%  log_sum_pt > 6.91
        + 0.09759489 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +9.8%  sum_pt_top50 > 934.2
        - 0.09324141 * max(0.0, Q.e2_sq - 0.007872294) / 0.00365949   # -9.3%  e2_sq > 0.007872
        + 0.0491436 * max(0.0, Q.e2_sq - 0.009606007) / 0.003182984   # +4.9%  e2_sq > 0.009606
        - 0.04173975 * max(0.0, Q.girth2 - 0.02928196) / 0.0002029538   # -4.2%  girth2 > 0.02928
        + 0.03826344 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +3.8%  sum_pt > 907.9 and e4 < 5.851e-08
        - 0.03811014 * max(0.0, Q.girth2 - 0.02928196) * max(0.0, 0.002741632 - Q.soft3_z) / 2.118876e-07   # -3.8%  girth2 > 0.02928 and soft3_z < 0.002742
        + 0.03115956 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +3.1%  n_particles < 64
        + 0.03020532 * max(0.0, Q.e2_sq - 0.007872294) * max(0.0, 0.003627839 - Q.soft7_z) / 6.713918e-06   # +3.0%  e2_sq > 0.007872 and soft7_z < 0.003628
        - 0.0291757 * max(0.0, Q.e2_sq - 0.009606007) * max(0.0, 0.003627839 - Q.soft7_z) / 5.80783e-06   # -2.9%  e2_sq > 0.009606 and soft7_z < 0.003628
        - 0.02857047 * max(0.0, 902.4062 - Q.sum_pt_top5) / 313.0794   # -2.9%  sum_pt_top5 < 902.4
        + 0.02794325 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +2.8%  sum_pt > 907.9
        - 0.02787123 * max(0.0, 0.01563836 - Q.girth2_top15) / 0.009593305   # -2.8%  girth2_top15 < 0.01564
        - 0.0239153 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -2.4%  girth2_top30 < 0.01216
        + 0.02322865 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # +2.3%  log_sum_pt > 6.959
        + 0.02121576 * max(0.0, Q.sum_pt_top30 - 911.9328) / 96.43087   # +2.1%  sum_pt_top30 > 911.9
        - 0.02021985 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -2.0%  tau1 < 0.1073
        - 0.0187739 * max(0.0, Q.sum_pt_top50 - 1048.098) / 31.96699   # -1.9%  sum_pt_top50 > 1048
        - 0.0185766 * max(0.0, 0.2708738 - Q.z_dr_0p2_0p4) / 0.2177567   # -1.9%  z_dr_0p2_0p4 < 0.2709
        - 0.01787463 * max(0.0, Q.z_top30_slots - 0.9203881) / 0.04411614   # -1.8%  z_top30_slots > 0.9204
        + 0.01776883 * max(0.0, Q.n_pt_above_1 - 26.0) / 16.44623   # +1.8%  n_pt_above_1 > 26
        + 0.01549391 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # +1.5%  e2 < 0.03876
        + 0.01453119 * max(0.0, Q.width - 0.006941794) / 0.004053246   # +1.5%  width > 0.006942
        - 0.01436646 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -1.4%  log_sum_pt > 6.92
        + 0.01218669 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # +1.2%  girth < 0.07374
        + 0.01005981 * max(0.0, 2.680253 - Q.D2) / 0.7435983   # +1.0%  D2 < 2.68
        + 0.01004486 * max(0.0, Q.n_for_90pct - 11.0) / 10.33888   # +1.0%  n_for_90pct > 11
        - 0.009734071 * max(0.0, Q.sum_pt_top50 - 1024.689) * max(0.0, 5.8505e-08 - Q.e4) / 2.280575e-06   # -1.0%  sum_pt_top50 > 1025 and e4 < 5.851e-08
        - 0.008626356 * max(0.0, Q.sum_pt_top15 - 951.1375) / 24.48977   # -0.9%  sum_pt_top15 > 951.1
        - 0.008408914 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # -0.8%  e3 < 0.0005178
        - 0.007495012 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.410481 - Q.D2) / 13.17726   # -0.7%  n_particles < 64 and D2 < 2.41
        - 0.006738158 * max(0.0, 0.3572263 - Q.N2) / 0.07385171   # -0.7%  N2 < 0.3572
        + 0.006531055 * max(0.0, Q.log_sum_pt - 6.935549) / 0.0371538   # +0.7%  log_sum_pt > 6.936
        + 0.005278348 * max(0.0, 0.005801034 - Q.girth2_top10) / 0.002192212   # +0.5%  girth2_top10 < 0.005801
        + 0.005056997 * max(0.0, Q.sum_pt_top10 - 822.975) / 44.35907   # +0.5%  sum_pt_top10 > 823
        - 0.004935075 * max(0.0, Q.max_dr - 0.1939977) / 0.1607927   # -0.5%  max_dr > 0.194
        + 0.004607241 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # +0.5%  psi_0p3 > 0.998
        + 0.004365189 * max(0.0, Q.sum_pt_top20 - 1017.778) / 17.66072   # +0.4%  sum_pt_top20 > 1018
        + 0.00407557 * max(0.0, 579.875 - Q.sum_pt_top5) / 65.87113   # +0.4%  sum_pt_top5 < 579.9
        - 0.003929392 * max(0.0, 0.008031209 - Q.girth2_top20) / 0.00309849   # -0.4%  girth2_top20 < 0.008031
        + 0.003788685 * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 4.239657   # +0.4%  n_dr_0p1_0p2 > 11
        - 0.00377665 * max(0.0, Q.girth - 0.1564779) / 0.0006617163   # -0.4%  girth > 0.1565
        + 0.003584234 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # +0.4%  n_dr_0p2_0p4 < 8
        - 0.003482851 * max(0.0, Q.pt_entropy - 2.903111) / 0.1962448   # -0.3%  pt_entropy > 2.903
        + 0.00337917 * max(0.0, Q.tau2 - 0.02164722) / 0.0156989   # +0.3%  tau2 > 0.02165
        + 0.003370503 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +0.3%  tau4 < 0.01627
        + 0.003277876 * max(0.0, 0.005383629 - Q.girth2_top15) / 0.001666876   # +0.3%  girth2_top15 < 0.005384
        + 0.002690061 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 12.6865 - Q.pt1_dr01) / 168.7239   # +0.3%  n_particles < 64 and pt1_dr01 < 12.69
        + 0.00229522 * max(0.0, 0.2708738 - Q.z_dr_0p2_0p4) * max(0.0, Q.zdr_0 - 0.005911134) / 0.0009498814   # +0.2%  z_dr_0p2_0p4 < 0.2709 and zdr_0 > 0.005911
        - 0.002261503 * max(0.0, 0.01256572 - Q.e2) / 0.0008030352   # -0.2%  e2 < 0.01257
        + 0.002203684 * max(0.0, Q.sd_rg - 0.1377546) / 0.046752   # +0.2%  sd_rg > 0.1378
        + 0.002183383 * max(0.0, Q.psi_0p1 - 0.9670742) / 0.003344718   # +0.2%  psi_0p1 > 0.9671
        + 0.00178097 * max(0.0, Q.z_dr_0_0p05 - 0.9084912) / 0.009227347   # +0.2%  z_dr_0_0p05 > 0.9085
        - 0.001675352 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.09468596 - Q.dr_11) / 0.00209608   # -0.2%  log_sum_pt > 6.91 and dr_11 < 0.09469
        - 0.001638066 * max(0.0, Q.z_dr_0p1_0p2 - 0.1872805) / 0.07218708   # -0.2%  z_dr_0p1_0p2 > 0.1873
        + 0.001349074 * max(0.0, Q.sum_pt_top50 - 1024.689) / 40.56456   # +0.1%  sum_pt_top50 > 1025
        - 0.001142437 * max(0.0, Q.psi_0p1 - 0.9670742) * max(0.0, Q.psi_0p3 - 0.9980008) / 2.935834e-06   # -0.1%  psi_0p1 > 0.9671 and psi_0p3 > 0.998
        - 0.001121013 * max(0.0, Q.sum_pt_top15 - 967.7705) / 19.94058   # -0.1%  sum_pt_top15 > 967.8
        + 0.001094216 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # +0.1%  log_sum_pt > 7.017
        - 0.0008776912 * max(0.0, Q.sum_pt_top30 - 1052.08) / 20.84425   # -0.1%  sum_pt_top30 > 1052
        - 0.0007215936 * max(0.0, Q.n_dr_0p2_0p4 - 21.0) / 0.6469597   # -0.1%  n_dr_0p2_0p4 > 21
        - 0.0004545612 * max(0.0, Q.n_pt_above_1 - 58.0) * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0002368421   # -0.0%  n_pt_above_1 > 58 and psi_0p3 > 0.9985
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 15.79;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.78906 * (0.08139773
        + 0.1729959 * max(0.0, 0.02580859 - Q.e2_sq) / 0.01698103   # +17.3%  e2_sq < 0.02581
        - 0.1485441 * max(0.0, 0.009614971 - Q.width) / 0.003502545   # -14.9%  width < 0.009615
        - 0.1041812 * max(0.0, 0.02497133 - Q.girth2_top40) / 0.01655385   # -10.4%  girth2_top40 < 0.02497
        - 0.06368467 * max(0.0, 0.01397874 - Q.girth2) / 0.006902497   # -6.4%  girth2 < 0.01398
        - 0.05132123 * max(0.0, Q.e2 - 0.01541561) / 0.01690634   # -5.1%  e2 > 0.01542
        + 0.04044727 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +4.0%  girth2_top40 < 0.008841
        + 0.03737961 * max(0.0, Q.sum_pt_top20 - 695.3094) / 236.2671   # +3.7%  sum_pt_top20 > 695.3
        + 0.03261126 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +3.3%  lam1 < 0.007671
        - 0.02374364 * max(0.0, Q.log_sum_pt - 6.811175) / 0.1383141   # -2.4%  log_sum_pt > 6.811
        + 0.02352143 * max(0.0, 0.008190222 - Q.width) / 0.002454223   # +2.4%  width < 0.00819
        + 0.02244679 * max(0.0, Q.LHA - 0.2091025) / 0.07391154   # +2.2%  LHA > 0.2091
        + 0.02187519 * max(0.0, 0.007511864 - Q.e2_sq) / 0.002016671   # +2.2%  e2_sq < 0.007512
        - 0.01861715 * max(0.0, 0.01397874 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.9853273) / 7.658135e-05   # -1.9%  girth2 < 0.01398 and psi_0p3 > 0.9853
        + 0.01837215 * max(0.0, 0.003811746 - Q.lam1) / 0.0006796118   # +1.8%  lam1 < 0.003812
        - 0.01706209 * max(0.0, Q.lam1 - 0.006189818) / 0.003311577   # -1.7%  lam1 > 0.00619
        - 0.01639256 * max(0.0, 0.02265114 - Q.girth2_top20) / 0.01537233   # -1.6%  girth2_top20 < 0.02265
        - 0.01488299 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -1.5%  girth2_top20 < 0.01083
        + 0.01292035 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +1.3%  lam2 < 0.001776
        + 0.01207838 * max(0.0, 0.007887677 - Q.girth2_top15) / 0.00326329   # +1.2%  girth2_top15 < 0.007888
        - 0.01112341 * max(0.0, 0.02580859 - Q.e2_sq) * max(0.0, 6.98945 - Q.log_sum_pt) / 0.001023242   # -1.1%  e2_sq < 0.02581 and log_sum_pt < 6.989
        + 0.0108114 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +1.1%  z_dr_0p1_0p2 < 0.1203
        - 0.01025457 * max(0.0, Q.lam2 - 0.0009731947) / 0.0007968622   # -1.0%  lam2 > 0.0009732
        + 0.00984662 * max(0.0, Q.girth2_top15 - 0.001319197) / 0.006270373   # +1.0%  girth2_top15 > 0.001319
        + 0.00981226 * max(0.0, Q.psi_0p3 - 0.9853273) / 0.009408723   # +1.0%  psi_0p3 > 0.9853
        + 0.008022711 * max(0.0, Q.girth2_top30 - 0.008376291) / 0.003092781   # +0.8%  girth2_top30 > 0.008376
        + 0.007874993 * max(0.0, 0.007511864 - Q.e2_sq) * max(0.0, 1007.788 - Q.sum_pt) / 0.02913759   # +0.8%  e2_sq < 0.007512 and sum_pt < 1008
        - 0.007278986 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 0.7108211 - Q.z_dr_0p05_0p1) / 0.02609827   # -0.7%  z_dr_0p1_0p2 < 0.1203 and z_dr_0p05_0p1 < 0.7108
        - 0.006545248 * max(0.0, Q.z_top20_slots - 0.7111557) / 0.1805694   # -0.7%  z_top20_slots > 0.7112
        - 0.005510364 * max(0.0, 7.876005e-05 - Q.e3) / 3.594075e-05   # -0.6%  e3 < 7.876e-05
        + 0.005456746 * max(0.0, Q.tau1 - 0.05444509) / 0.03990678   # +0.5%  tau1 > 0.05445
        - 0.005152755 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, 0.3384815 - Q.N2) / 1.669221e-05   # -0.5%  e3 < 0.0003372 and N2 < 0.3385
        - 0.004676936 * max(0.0, Q.psi_0p1 - 0.8509811) / 0.04568859   # -0.5%  psi_0p1 > 0.851
        - 0.004583597 * max(0.0, 0.002412891 - Q.girth2_top2) / 0.0009487944   # -0.5%  girth2_top2 < 0.002413
        + 0.004085267 * max(0.0, 0.00287991 - Q.girth2_top20) / 0.000559956   # +0.4%  girth2_top20 < 0.00288
        - 0.003759441 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, 0.001592178 - Q.girth2_top3) / 7.904451e-05   # -0.4%  log_sum_pt > 6.811 and girth2_top3 < 0.001592
        + 0.003617171 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # +0.4%  LHA < 0.2601
        - 0.003531587 * max(0.0, 0.003811746 - Q.lam1) * max(0.0, 1008.935 - Q.sum_pt_top50) / 0.0115984   # -0.4%  lam1 < 0.003812 and sum_pt_top50 < 1009
        + 0.00262409 * max(0.0, Q.sum_pt_top20 - 695.3094) * max(0.0, 1.36316 - Q.D2_b2) / 96.35963   # +0.3%  sum_pt_top20 > 695.3 and D2_b2 < 1.363
        + 0.002567753 * max(0.0, 0.2018786 - Q.tau21_b2) / 0.03799024   # +0.3%  tau21_b2 < 0.2019
        + 0.002197083 * max(0.0, 0.002412891 - Q.girth2_top2) * max(0.0, Q.M2 - 0.03940774) / 3.297792e-05   # +0.2%  girth2_top2 < 0.002413 and M2 > 0.03941
        + 0.001998912 * max(0.0, Q.e2 - 0.04755309) * max(0.0, 0.3289237 - Q.z_dr_0_0p05) / 0.0005973978   # +0.2%  e2 > 0.04755 and z_dr_0_0p05 < 0.3289
        + 0.001685512 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.2%  e2 > 0.04755
        + 0.001617829 * max(0.0, 0.001784301 - Q.e2_sq) / 0.0001206653   # +0.2%  e2_sq < 0.001784
        + 0.001318108 * max(0.0, Q.psi_0p3 - 0.9853273) * max(0.0, Q.n_dr_0p1_0p2 - 15.0) / 0.01929542   # +0.1%  psi_0p3 > 0.9853 and n_dr_0p1_0p2 > 15
        - 0.001254605 * max(0.0, Q.LHA - 0.2091025) * max(0.0, 0.001387595 - Q.soft8_z) / 1.01053e-05   # -0.1%  LHA > 0.2091 and soft8_z < 0.001388
        - 0.001103922 * max(0.0, Q.sum_pt_top40 - 1225.842) / 7.259753   # -0.1%  sum_pt_top40 > 1226
        - 0.001088254 * max(0.0, 19.46875 - Q.pt_6) / 0.2515892   # -0.1%  pt_6 < 19.47
        + 0.001022245 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # +0.1%  log_sum_pt > 7.139
        + 0.001003218 * max(0.0, Q.e2 - 0.04755309) * max(0.0, 26.14062 - Q.pt_14) / 0.01767661   # +0.1%  e2 > 0.04755 and pt_14 < 26.14
        + 0.000931323 * max(0.0, 0.01906139 - Q.z_6) / 0.0002479519   # +0.1%  z_6 < 0.01906
        - 0.0008833313 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, Q.dr_max_012 - 0.1686705) / 0.001990704   # -0.1%  log_sum_pt > 6.811 and dr_max_012 > 0.1687
        + 0.0008068364 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.zdr_0 - 0.0008718296) / 4.275428e-05   # +0.1%  e2 > 0.04755 and zdr_0 > 0.0008718
        + 0.0007723783 * max(0.0, Q.lam2 - 0.003687605) / 0.0003262682   # +0.1%  lam2 > 0.003688
        + 0.0007704178 * max(0.0, Q.z_top20_slots - 0.7111557) * max(0.0, Q.girth2_top3 - 0.01002369) / 0.0001325706   # +0.1%  z_top20_slots > 0.7112 and girth2_top3 > 0.01002
        - 0.0005778258 * max(0.0, Q.lam1 - 0.02457025) / 0.0002026642   # -0.1%  lam1 > 0.02457
        + 0.0004437031 * max(0.0, 0.01397874 - Q.girth2) * max(0.0, Q.soft1_pt - 1.521582) / 0.0002697325   # +0.0%  girth2 < 0.01398 and soft1_pt > 1.522
        + 0.0003105618 * max(0.0, Q.e2 - 0.06524004) * max(0.0, Q.psi_0p3 - 0.9896594) / 1.181132e-06   # +0.0%  e2 > 0.06524 and psi_0p3 > 0.9897
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 41.8;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 41.80107 * (-0.02703409
        + 0.1386485 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # +13.9%  e2_sq < 0.009606
        - 0.07970153 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # -8.0%  girth2_top50 < 0.008125
        - 0.07542521 * max(0.0, 0.009606007 - Q.e2_sq) * max(0.0, 1167.447 - Q.sum_pt) / 0.4400479   # -7.5%  e2_sq < 0.009606 and sum_pt < 1167
        + 0.05377256 * max(0.0, 0.01397874 - Q.width) / 0.006902497   # +5.4%  width < 0.01398
        - 0.0434154 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # -4.3%  girth2_top30 < 0.008376
        - 0.03906316 * max(0.0, 0.08589404 - Q.girth) / 0.02745918   # -3.9%  girth < 0.08589
        + 0.03805271 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +3.8%  lam1 < 0.007671
        - 0.03759162 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -3.8%  lam1 < 0.008242
        - 0.03336234 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # -3.3%  girth2 < 0.007877
        - 0.03211023 * max(0.0, 0.006936725 - Q.e2_sq) / 0.001688614   # -3.2%  e2_sq < 0.006937
        - 0.03041078 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # -3.0%  LHA < 0.2454
        + 0.02789595 * max(0.0, 0.05660088 - Q.girth) / 0.01080382   # +2.8%  girth < 0.0566
        + 0.02744594 * max(0.0, 0.007820315 - Q.girth2_top50) / 0.002264874   # +2.7%  girth2_top50 < 0.00782
        + 0.02467154 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # +2.5%  girth2_top30 < 0.01216
        + 0.02445248 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # +2.4%  LHA < 0.3024
        - 0.02261437 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -2.3%  max_dr > 0.2405
        - 0.01969876 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -2.0%  e2 < 0.04359
        + 0.01966936 * max(0.0, 0.005913555 - Q.lam1) * max(0.0, 1139.734 - Q.sum_pt_top40) / 0.1589332   # +2.0%  lam1 < 0.005914 and sum_pt_top40 < 1140
        - 0.01806838 * max(0.0, 0.00375223 - Q.girth2_top30) / 0.0006714094   # -1.8%  girth2_top30 < 0.003752
        + 0.01710493 * max(0.0, 0.08589404 - Q.girth) * max(0.0, 1156.659 - Q.sum_pt_top50) / 3.29885   # +1.7%  girth < 0.08589 and sum_pt_top50 < 1157
        + 0.01594176 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +1.6%  tau1 < 0.07709
        + 0.01445601 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # +1.4%  girth2_top30 < 0.006364
        - 0.01434741 * max(0.0, 0.005913555 - Q.lam1) * max(0.0, 1167.447 - Q.sum_pt) / 0.1821423   # -1.4%  lam1 < 0.005914 and sum_pt < 1167
        + 0.01274692 * max(0.0, Q.max_dr - 0.2982) / 0.06837446   # +1.3%  max_dr > 0.2982
        - 0.01086106 * max(0.0, 0.007511864 - Q.e2_sq) / 0.002016671   # -1.1%  e2_sq < 0.007512
        + 0.01058339 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +1.1%  e2 < 0.03481
        - 0.0104525 * max(0.0, 0.04828819 - Q.tau2) / 0.02103241   # -1.0%  tau2 < 0.04829
        - 0.01043596 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # -1.0%  LHA < 0.2601
        + 0.009441294 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +0.9%  n_dr_0p2_0p4 < 15
        - 0.009251129 * max(0.0, 0.005809485 - Q.girth2_top30) / 0.001402136   # -0.9%  girth2_top30 < 0.005809
        - 0.00686903 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -0.7%  lam1 < 0.005914
        + 0.006089819 * max(0.0, 0.002396991 - Q.lam2) / 0.001466253   # +0.6%  lam2 < 0.002397
        + 0.005906756 * max(0.0, 0.003811746 - Q.lam1) / 0.0006796118   # +0.6%  lam1 < 0.003812
        + 0.004405257 * max(0.0, 0.002396991 - Q.lam2) * max(0.0, 7.139296 - Q.log_sum_pt) / 0.0002814467   # +0.4%  lam2 < 0.002397 and log_sum_pt < 7.139
        + 0.004276979 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.01396296 - Q.e2_sq) / 0.000313768   # +0.4%  tau21_b2 < 0.2352 and e2_sq < 0.01396
        + 0.004224681 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # +0.4%  psi_0p3 > 0.9966
        - 0.003941315 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -0.4%  psi_0p2 > 0.9087
        - 0.003694958 * max(0.0, 36.0 - Q.n_pt_above_5) / 9.448955   # -0.4%  n_pt_above_5 < 36
        - 0.002861014 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007872294 - Q.e2_sq) / 5.103177e-05   # -0.3%  tau21_b2 < 0.2352 and e2_sq < 0.007872
        + 0.002684185 * max(0.0, Q.psi_0p3 - 0.9966167) * max(0.0, 2.275391 - Q.soft1_pt) / 0.002319433   # +0.3%  psi_0p3 > 0.9966 and soft1_pt < 2.275
        - 0.002626671 * max(0.0, Q.tau21 - 0.4281458) / 0.1150264   # -0.3%  tau21 > 0.4281
        + 0.002531807 * max(0.0, 0.0168592 - Q.girth2_top5) / 0.01188986   # +0.3%  girth2_top5 < 0.01686
        + 0.002526487 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 1225.842 - Q.sum_pt_top40) / 1472.634   # +0.3%  n_dr_0p2_0p4 < 15 and sum_pt_top40 < 1226
        + 0.002477822 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +0.2%  tau21_b2 < 0.2352
        - 0.002472268 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.001696484   # -0.2%  tau21_b2 < 0.2352 and sj2_dr > 0.1938
        + 0.002148934 * max(0.0, 0.007877041 - Q.girth2) * max(0.0, 0.004600222 - Q.zdr_6) / 7.782441e-06   # +0.2%  girth2 < 0.007877 and zdr_6 < 0.0046
        - 0.002100296 * max(0.0, 0.002396991 - Q.lam2) * max(0.0, Q.zdr_0 - 0.002524869) / 9.492656e-06   # -0.2%  lam2 < 0.002397 and zdr_0 > 0.002525
        + 0.00197901 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # +0.2%  psi_0p3 > 0.998
        - 0.001452081 * max(0.0, Q.psi_0p3 - 0.9995915) / 8.716828e-05   # -0.1%  psi_0p3 > 0.9996
        - 0.001294978 * max(0.0, Q.sum_pt_top15 - 1082.548) * max(0.0, Q.mean_phi - 2.298159e-05) / 0.0002044402   # -0.1%  sum_pt_top15 > 1083 and mean_phi > 2.298e-05
        + 0.00119265 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sum_pt_top30 - 978.0762) / 2.738836   # +0.1%  tau21_b2 < 0.2352 and sum_pt_top30 > 978.1
        + 0.0009644129 * max(0.0, Q.psi_0p3 - 0.9966167) * max(0.0, 0.3403782 - Q.soft8_dr0) / 0.0002962251   # +0.1%  psi_0p3 > 0.9966 and soft8_dr0 < 0.3404
        + 0.0009452753 * max(0.0, 0.05048381 - Q.girth) / 0.008533025   # +0.1%  girth < 0.05048
        + 0.0009060719 * max(0.0, 0.1219401 - Q.tau21_b2) / 0.01156068   # +0.1%  tau21_b2 < 0.1219
        + 0.0008224065 * max(0.0, Q.sum_pt_top15 - 1082.548) * max(0.0, Q.mean_phi - -3.355129e-05) / 0.0003894346   # +0.1%  sum_pt_top15 > 1083 and mean_phi > -3.355e-05
        - 0.0007958747 * max(0.0, 0.00130722 - Q.girth2_top10) / 0.0002794102   # -0.1%  girth2_top10 < 0.001307
        - 0.0007720352 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.n_pt_above_10 - 20.0) / 0.001532328   # -0.1%  psi_0p3 > 0.998 and n_pt_above_10 > 20
        - 0.0006837243 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, 0.0361727 - Q.dr_0) / 4.440919e-06   # -0.1%  psi_0p3 > 0.998 and dr_0 < 0.03617
        - 0.0006682924 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, 0.004298863 - Q.C2_b2) / 1.015813e-06   # -0.1%  psi_0p3 > 0.998 and C2_b2 < 0.004299
        - 0.000616588 * max(0.0, 0.007877041 - Q.girth2) * max(0.0, 0.3289237 - Q.z_dr_0_0p05) / 4.14127e-05   # -0.1%  girth2 < 0.007877 and z_dr_0_0p05 < 0.3289
        - 0.0005583187 * max(0.0, 0.0009793444 - Q.mean_eta2) / 0.000113516   # -0.1%  mean_eta2 < 0.0009793
        - 0.0004390299 * max(0.0, 0.008241985 - Q.lam1) * max(0.0, Q.soft1_z - 0.0007833979) / 2.117702e-07   # -0.0%  lam1 < 0.008242 and soft1_z > 0.0007834
        - 0.0004302384 * max(0.0, 0.007820315 - Q.girth2_top50) * max(0.0, Q.soft5_pt - 0.5297852) / 0.001781666   # -0.0%  girth2_top50 < 0.00782 and soft5_pt > 0.5298
        - 0.0003858675 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.orientation_deg - 17.2994) / 0.772537   # -0.0%  tau21_b2 < 0.2352 and orientation_deg > 17.3
        + 0.0003437935 * max(0.0, 0.009606007 - Q.e2_sq) * max(0.0, Q.girth2_top10 - 0.00625621) / 2.278575e-07   # +0.0%  e2_sq < 0.009606 and girth2_top10 > 0.006256
        - 0.0003180154 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 4.476506   # -0.0%  n_dr_0p2_0p4 < 15 and n_dr_0p1_0p2 > 21
        - 0.0002845125 * max(0.0, Q.sum_pt_top15 - 1082.548) * max(0.0, Q.soft4_dr0 - 0.02345786) / 0.9995171   # -0.0%  sum_pt_top15 > 1083 and soft4_dr0 > 0.02346
        + 0.0002523955 * max(0.0, Q.sum_pt_top15 - 1082.548) * max(0.0, 0.003468228 - Q.C2_b2) / 0.004508564   # +0.0%  sum_pt_top15 > 1083 and C2_b2 < 0.003468
        - 0.0002049489 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.log_sum_pt - 7.139296) / 0.0001642741   # -0.0%  tau21_b2 < 0.2352 and log_sum_pt > 7.139
        - 0.0001579871 * max(0.0, Q.sum_pt_top15 - 1082.548) * max(0.0, 0.002333533 - Q.zdr_6) / 0.008196118   # -0.0%  sum_pt_top15 > 1083 and zdr_6 < 0.002334
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 16.11;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.11391 * (0.0691021
        + 0.1688927 * max(0.0, 0.0258669 - Q.girth2) / 0.01702764   # +16.9%  girth2 < 0.02587
        - 0.1584647 * max(0.0, 0.0258669 - Q.girth2) * max(0.0, 7.139296 - Q.log_sum_pt) / 0.003279142   # -15.8%  girth2 < 0.02587 and log_sum_pt < 7.139
        - 0.1339268 * max(0.0, 0.01397874 - Q.girth2) / 0.006902497   # -13.4%  girth2 < 0.01398
        - 0.1107896 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -11.1%  e2_sq < 0.009606
        + 0.07135383 * max(0.0, 1245.697 - Q.sum_pt_top50) / 217.4702   # +7.1%  sum_pt_top50 < 1246
        + 0.06060127 * max(0.0, 0.1219132 - Q.tau1) / 0.04578087   # +6.1%  tau1 < 0.1219
        + 0.039057 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, 1260.541 - Q.sum_pt) / 0.4807996   # +3.9%  girth2_top30 < 0.007464 and sum_pt < 1261
        - 0.03829628 * max(0.0, 0.0258669 - Q.girth2) * max(0.0, Q.z_top50_slots - 0.9704436) / 0.000437459   # -3.8%  girth2 < 0.02587 and z_top50_slots > 0.9704
        + 0.02604084 * max(0.0, 0.008190222 - Q.width) / 0.002454223   # +2.6%  width < 0.00819
        - 0.02536464 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # -2.5%  n_dr_0p2_0p4 < 18
        + 0.02174757 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +2.2%  sum_pt_top50 < 1157
        + 0.02033973 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +2.0%  girth2_top40 < 0.008841
        + 0.01657531 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # +1.7%  sum_pt < 1085
        + 0.01556966 * max(0.0, 1028.184 - Q.sum_pt) / 26.95739   # +1.6%  sum_pt < 1028
        - 0.01452172 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, 1095.686 - Q.sum_pt_top40) / 0.172292   # -1.5%  girth2_top30 < 0.007464 and sum_pt_top40 < 1096
        + 0.01230106 * max(0.0, 0.01351431 - Q.girth2_top50) / 0.006620975   # +1.2%  girth2_top50 < 0.01351
        + 0.006793581 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sj2_zsoft - 0.1181474) / 1.531301   # +0.7%  n_dr_0p2_0p4 < 18 and sj2_zsoft > 0.1181
        - 0.006512737 * max(0.0, 0.02210818 - Q.e2) / 0.00342102   # -0.7%  e2 < 0.02211
        + 0.006325255 * max(0.0, Q.girth2_top15 - 0.007887677) / 0.002755219   # +0.6%  girth2_top15 > 0.007888
        + 0.005053321 * max(0.0, 0.01397874 - Q.girth2) * max(0.0, 986.0565 - Q.sum_pt) / 0.05983431   # +0.5%  girth2 < 0.01398 and sum_pt < 986.1
        - 0.004466691 * max(0.0, 0.1219132 - Q.tau1) * max(0.0, 0.5364935 - Q.planar_flow) / 0.004630037   # -0.4%  tau1 < 0.1219 and planar_flow < 0.5365
        + 0.004420908 * max(0.0, 1.788105 - Q.D2) / 0.2865857   # +0.4%  D2 < 1.788
        - 0.004117714 * max(0.0, Q.girth2_top15 - 0.005788041) / 0.003469237   # -0.4%  girth2_top15 > 0.005788
        + 0.003418981 * max(0.0, 1008.935 - Q.sum_pt_top50) / 22.0436   # +0.3%  sum_pt_top50 < 1009
        - 0.003360022 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -0.3%  psi_0p3 > 0.9943
        + 0.003248813 * max(0.0, 0.002574843 - Q.e2_sq) / 0.0002582574   # +0.3%  e2_sq < 0.002575
        + 0.002970942 * max(0.0, 0.003638856 - Q.girth2) / 0.0004964601   # +0.3%  girth2 < 0.003639
        - 0.002884424 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # -0.3%  n_dr_0p2_0p4 < 6
        - 0.002384787 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.2%  log_sum_pt < 6.811
        + 0.002241399 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.2%  sj2_dr > 0.2232
        + 0.001878858 * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 1.386277   # +0.2%  n_dr_0p1_0p2 > 21
        - 0.001857281 * max(0.0, 1008.935 - Q.sum_pt_top50) * max(0.0, 21.0 - Q.n_dr_0p05_0p1) / 223.6243   # -0.2%  sum_pt_top50 < 1009 and n_dr_0p05_0p1 < 21
        + 0.001460672 * max(0.0, Q.sum_pt_top20 - 926.0902) / 52.62483   # +0.1%  sum_pt_top20 > 926.1
        + 0.0007672582 * max(0.0, 0.0258669 - Q.girth2) * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.0001095108   # +0.1%  girth2 < 0.02587 and z_dr_0p1_0p2 > 0.334
        + 0.0007346387 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.01778043 - Q.zdr_0) / 4.999377e-05   # +0.1%  log_sum_pt < 6.811 and zdr_0 < 0.01778
        - 0.0006698942 * max(0.0, Q.sum_pt_top20 - 1017.778) / 17.66072   # -0.1%  sum_pt_top20 > 1018
        - 0.0005891659 * max(0.0, 1245.697 - Q.sum_pt_top50) * max(0.0, Q.n_for_90pct - 39.0) / 46.16464   # -0.1%  sum_pt_top50 < 1246 and n_for_90pct > 39
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 6.376;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.376221 * (-0.1458013
        + 0.1570786 * max(0.0, Q.lam1 - 0.004673423) / 0.004174908   # +15.7%  lam1 > 0.004673
        + 0.1426661 * max(0.0, 0.006941794 - Q.width) / 0.001690424   # +14.3%  width < 0.006942
        - 0.0913629 * max(0.0, Q.lam1 - 0.006716737) / 0.003087986   # -9.1%  lam1 > 0.006717
        - 0.07276796 * max(0.0, 0.02675364 - Q.girth2_top15) / 0.01959938   # -7.3%  girth2_top15 < 0.02675
        + 0.04905945 * max(0.0, 1191.938 - Q.sum_pt_top30) / 206.0686   # +4.9%  sum_pt_top30 < 1192
        + 0.0473308 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # +4.7%  girth2_top20 < 0.01083
        - 0.03799822 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -3.8%  log_sum_pt < 6.903
        + 0.03526536 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +3.5%  lam1 < 0.007259
        + 0.03161988 * max(0.0, 972.4111 - Q.sum_pt_top40) / 17.57725   # +3.2%  sum_pt_top40 < 972.4
        - 0.02844321 * max(0.0, Q.girth - 0.1207452) / 0.004285542   # -2.8%  girth > 0.1207
        + 0.02316313 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +2.3%  sum_pt < 986.1
        + 0.0201469 * max(0.0, Q.girth - 0.1207452) * max(0.0, 2.894531 - Q.soft5_pt) / 0.005763801   # +2.0%  girth > 0.1207 and soft5_pt < 2.895
        + 0.02000196 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +2.0%  LHA > 0.3332
        + 0.01741439 * max(0.0, 0.005712208 - Q.girth2_top40) / 0.001219686   # +1.7%  girth2_top40 < 0.005712
        + 0.01727695 * max(0.0, Q.LHA - 0.3332345) * max(0.0, 1042.609 - Q.sum_pt) / 0.8026572   # +1.7%  LHA > 0.3332 and sum_pt < 1043
        - 0.01680163 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # -1.7%  tau1 < 0.06311
        - 0.01644128 * max(0.0, 0.007259287 - Q.lam1) * max(0.0, Q.sum_pt - 1085.125) / 0.08770494   # -1.6%  lam1 < 0.007259 and sum_pt > 1085
        + 0.01428353 * max(0.0, 0.004169954 - Q.girth2_top15) / 0.001130716   # +1.4%  girth2_top15 < 0.00417
        - 0.01407793 * max(0.0, Q.e2_sq - 0.01989454) / 0.001200608   # -1.4%  e2_sq > 0.01989
        - 0.01320298 * max(0.0, Q.girth2 - 0.0258669) / 0.0004653552   # -1.3%  girth2 > 0.02587
        - 0.01196338 * max(0.0, Q.LHA - 0.3332345) * max(0.0, 2.117188 - Q.soft5_pt) / 0.01067255   # -1.2%  LHA > 0.3332 and soft5_pt < 2.117
        + 0.01152587 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +1.2%  psi_0p3 > 0.9956
        - 0.01050904 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -1.1%  z_dr_0_0p05 > 0.8789
        + 0.01035735 * max(0.0, 0.007259287 - Q.lam1) * max(0.0, Q.n_particles - 36.0) / 0.01882539   # +1.0%  lam1 < 0.007259 and n_particles > 36
        - 0.01018538 * max(0.0, 0.005712208 - Q.girth2_top40) * max(0.0, 1007.788 - Q.sum_pt) / 0.0192656   # -1.0%  girth2_top40 < 0.005712 and sum_pt < 1008
        - 0.007879389 * max(0.0, Q.eccentricity - 0.7333655) / 0.1074574   # -0.8%  eccentricity > 0.7334
        + 0.006943898 * max(0.0, 6.903423 - Q.log_sum_pt) * max(0.0, 0.4479367 - Q.z_dr_0p1_0p2) / 0.004178024   # +0.7%  log_sum_pt < 6.903 and z_dr_0p1_0p2 < 0.4479
        + 0.006787498 * max(0.0, 0.005712208 - Q.girth2_top40) * max(0.0, Q.log_sum_pt - 7.017258) / 3.653263e-05   # +0.7%  girth2_top40 < 0.005712 and log_sum_pt > 7.017
        + 0.006748311 * max(0.0, 2.485823 - Q.pt_entropy) / 0.09968085   # +0.7%  pt_entropy < 2.486
        - 0.006640109 * max(0.0, 972.4111 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.106445) / 11.84488   # -0.7%  sum_pt_top40 < 972.4 and soft4_pt > 1.106
        - 0.005622968 * max(0.0, 0.001319197 - Q.girth2_top15) / 0.0002099631   # -0.6%  girth2_top15 < 0.001319
        - 0.005166626 * max(0.0, 972.4111 - Q.sum_pt_top40) * max(0.0, 20.0 - Q.n_dr_0_0p05) / 189.2239   # -0.5%  sum_pt_top40 < 972.4 and n_dr_0_0p05 < 20
        - 0.003594802 * max(0.0, 972.4111 - Q.sum_pt_top40) * max(0.0, Q.sum_pt - 907.9372) / 488.8461   # -0.4%  sum_pt_top40 < 972.4 and sum_pt > 907.9
        + 0.00353666 * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 1.386277   # +0.4%  n_dr_0p1_0p2 > 21
        - 0.003326105 * max(0.0, Q.girth2_top15 - 0.02146578) / 0.0006182335   # -0.3%  girth2_top15 > 0.02147
        + 0.003312078 * max(0.0, Q.z_dr_0_0p05 - 0.878906) * max(0.0, Q.sum_pt_top40 - 1013.042) / 0.9125551   # +0.3%  z_dr_0_0p05 > 0.8789 and sum_pt_top40 > 1013
        + 0.003028052 * max(0.0, 0.004169954 - Q.girth2_top15) * max(0.0, Q.n_dr_0p2_0p4 - 8.0) / 0.001650576   # +0.3%  girth2_top15 < 0.00417 and n_dr_0p2_0p4 > 8
        + 0.00245803 * max(0.0, 986.0565 - Q.sum_pt) * max(0.0, Q.soft4_pt - 1.226562) / 3.636941   # +0.2%  sum_pt < 986.1 and soft4_pt > 1.227
        + 0.002402854 * max(0.0, 26.0 - Q.n_pt_above_1) / 0.8081244   # +0.2%  n_pt_above_1 < 26
        + 0.002271914 * max(0.0, 0.007259287 - Q.lam1) * max(0.0, 886.3438 - Q.sum_pt_top30) / 0.01167511   # +0.2%  lam1 < 0.007259 and sum_pt_top30 < 886.3
        + 0.002140817 * max(0.0, 6.903423 - Q.log_sum_pt) * max(0.0, 0.0830523 - Q.dr_4) / 0.0004939562   # +0.2%  log_sum_pt < 6.903 and dr_4 < 0.08305
        + 0.001762941 * max(0.0, Q.e3 - 0.0005178279) / 8.375623e-06   # +0.2%  e3 > 0.0005178
        - 0.001702035 * max(0.0, Q.e2_sq - 0.0292152) * max(0.0, 3.779492 - Q.soft7_pt) / 0.0003078674   # -0.2%  e2_sq > 0.02922 and soft7_pt < 3.779
        - 0.001695433 * max(0.0, 972.4111 - Q.sum_pt_top40) * max(0.0, 0.0369869 - Q.tau3) / 0.1284345   # -0.2%  sum_pt_top40 < 972.4 and tau3 < 0.03699
        + 0.001449629 * max(0.0, Q.girth - 0.1207452) * max(0.0, 0.4606099 - Q.sj3_pairmin_over_m) / 0.0004267265   # +0.1%  girth > 0.1207 and sj3_pairmin_over_m < 0.4606
        - 0.000378689 * max(0.0, 6.903423 - Q.log_sum_pt) * max(0.0, Q.soft4_pt - 1.789258) / 0.002713428   # -0.0%  log_sum_pt < 6.903 and soft4_pt > 1.789
        - 0.0002069182 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, 907.9372 - Q.sum_pt) / 0.03221192   # -0.0%  z_dr_0p1_0p2 > 0.6882 and sum_pt < 907.9
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 15.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.23612 * (0.1835324
        - 0.1900658 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -19.0%  girth < 0.1207
        + 0.1077231 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # +10.8%  tau1 < 0.1507
        - 0.07305238 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -7.3%  e2_sq < 0.009606
        + 0.06659184 * max(0.0, Q.e2 - 0.006720044) / 0.02422374   # +6.7%  e2 > 0.00672
        + 0.05962688 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +6.0%  e2_sq < 0.008184
        - 0.04754299 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # -4.8%  e3 < 0.0005178
        - 0.04269088 * max(0.0, Q.girth2_top20 - 0.005718442) / 0.003749064   # -4.3%  girth2_top20 > 0.005718
        - 0.03148253 * max(0.0, 3.539367 - Q.pt_entropy) / 0.7052473   # -3.1%  pt_entropy < 3.539
        + 0.028093 * max(0.0, Q.girth2_top20 - 0.008031209) / 0.002906852   # +2.8%  girth2_top20 > 0.008031
        + 0.02753758 * max(0.0, 0.00592208 - Q.e2_sq) / 0.001192334   # +2.8%  e2_sq < 0.005922
        - 0.02609733 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -2.6%  n_dr_0p2_0p4 < 15
        + 0.0228279 * max(0.0, 0.3719813 - Q.LHA) / 0.1170757   # +2.3%  LHA < 0.372
        - 0.01941889 * max(0.0, 0.03577037 - Q.girth) / 0.004182456   # -1.9%  girth < 0.03577
        - 0.0191778 * max(0.0, 0.01976735 - Q.girth2_top10) / 0.01371247   # -1.9%  girth2_top10 < 0.01977
        - 0.01645002 * max(0.0, Q.girth2 - 0.01397874) / 0.002228371   # -1.6%  girth2 > 0.01398
        + 0.01634656 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.6%  z_dr_0p2_0p4 < 0.09123
        + 0.01463975 * max(0.0, 0.08222447 - Q.M2) / 0.02103373   # +1.5%  M2 < 0.08222
        - 0.01425739 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # -1.4%  girth2_top10 < 0.007679
        + 0.01238797 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +1.2%  dr_0 < 0.06413
        + 0.01138679 * max(0.0, 2.410481 - Q.D2) / 0.587301   # +1.1%  D2 < 2.41
        + 0.010096 * max(0.0, 656.3844 - Q.sum_pt_top3) / 209.0067   # +1.0%  sum_pt_top3 < 656.4
        - 0.009892018 * max(0.0, Q.e2_sq - 0.02580859) / 0.0004635665   # -1.0%  e2_sq > 0.02581
        + 0.009015424 * max(0.0, 0.006403325 - Q.width) / 0.001406912   # +0.9%  width < 0.006403
        + 0.008502313 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # +0.9%  LHA < 0.187
        - 0.008126004 * max(0.0, Q.e2 - 0.006720044) * max(0.0, Q.log_sum_pt - 6.893714) / 0.001240306   # -0.8%  e2 > 0.00672 and log_sum_pt > 6.894
        - 0.007929318 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # -0.8%  lam1 < 0.00619
        - 0.006490905 * max(0.0, Q.z_dr_0_0p05 - 0.7128619) / 0.07526795   # -0.6%  z_dr_0_0p05 > 0.7129
        - 0.006123588 * max(0.0, 0.0259543 - Q.tau4) / 0.009964748   # -0.6%  tau4 < 0.02595
        + 0.006115051 * max(0.0, Q.girth2 - 0.01397874) * max(0.0, 114.75 - Q.pt_3) / 0.1232701   # +0.6%  girth2 > 0.01398 and pt_3 < 114.8
        + 0.005117104 * max(0.0, 0.06108421 - Q.dr_1) / 0.02249055   # +0.5%  dr_1 < 0.06108
        + 0.004828219 * max(0.0, Q.girth2_top40 - 0.02497133) / 0.0004755075   # +0.5%  girth2_top40 > 0.02497
        - 0.004804259 * max(0.0, 0.3062621 - Q.tau21_b2) / 0.08794306   # -0.5%  tau21_b2 < 0.3063
        + 0.004753491 * max(0.0, 0.006212866 - Q.zdr_2) / 0.00254432   # +0.5%  zdr_2 < 0.006213
        - 0.004751224 * max(0.0, 949.9169 - Q.sum_pt) / 6.913717   # -0.5%  sum_pt < 949.9
        + 0.004738322 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 40.0 - Q.n_real_top40) / 41.36308   # +0.5%  n_dr_0p2_0p4 < 15 and n_real_top40 < 40
        + 0.004183826 * max(0.0, Q.n_dr_0_0p05 - 6.0) / 7.551284   # +0.4%  n_dr_0_0p05 > 6
        + 0.003892361 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.max_dr - 0.2404747) / 0.7337551   # +0.4%  n_dr_0p2_0p4 < 15 and max_dr > 0.2405
        + 0.003676691 * max(0.0, Q.e2 - 0.006720044) * max(0.0, Q.sj3_pairmin_over_m - 0.1765064) / 0.002982204   # +0.4%  e2 > 0.00672 and sj3_pairmin_over_m > 0.1765
        - 0.003583305 * max(0.0, 0.1690338 - Q.sd_rg) / 0.06029557   # -0.4%  sd_rg < 0.169
        - 0.003335553 * max(0.0, 0.1722488 - Q.tau21_b2) / 0.02681502   # -0.3%  tau21_b2 < 0.1722
        + 0.003026196 * max(0.0, 656.3844 - Q.sum_pt_top3) * max(0.0, 0.9768165 - Q.eccentricity) / 43.8908   # +0.3%  sum_pt_top3 < 656.4 and eccentricity < 0.9768
        - 0.002977867 * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0004356739   # -0.3%  psi_0p3 > 0.9985
        - 0.002731596 * max(0.0, Q.soft2_pt - 0.291748) / 0.5590972   # -0.3%  soft2_pt > 0.2917
        + 0.002532883 * max(0.0, Q.dr_max_012 - 0.06112084) / 0.05574719   # +0.3%  dr_max_012 > 0.06112
        - 0.00244528 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, Q.pt_6 - 33.6875) / 0.01241674   # -0.2%  girth2_top30 < 0.005402 and pt_6 > 33.69
        + 0.002355882 * max(0.0, 949.9169 - Q.sum_pt) * max(0.0, 0.004495205 - Q.zdr_4) / 0.01304907   # +0.2%  sum_pt < 949.9 and zdr_4 < 0.004495
        - 0.002274424 * max(0.0, Q.psi_0p3 - 0.9985421) * max(0.0, Q.n_real_top40 - 26.0) / 0.003911714   # -0.2%  psi_0p3 > 0.9985 and n_real_top40 > 26
        - 0.00220533 * max(0.0, 0.1207452 - Q.girth) * max(0.0, 0.8476928 - Q.tau32) / 0.004807579   # -0.2%  girth < 0.1207 and tau32 < 0.8477
        + 0.001685185 * max(0.0, 0.1207452 - Q.girth) * max(0.0, Q.pt1_dr01 - 12.6865) / 0.05101569   # +0.2%  girth < 0.1207 and pt1_dr01 > 12.69
        - 0.001561318 * max(0.0, 0.002247756 - Q.girth2_top10) / 0.0005836006   # -0.2%  girth2_top10 < 0.002248
        - 0.001305263 * max(0.0, 907.9372 - Q.sum_pt) * max(0.0, 0.003932029 - Q.zdr_4) / 0.006421434   # -0.1%  sum_pt < 907.9 and zdr_4 < 0.003932
        + 0.00116438 * max(0.0, 0.1207452 - Q.girth) * max(0.0, Q.ptdr0_2 - 7.740999) / 0.05541042   # +0.1%  girth < 0.1207 and ptdr0_2 > 7.741
        + 0.0009493874 * max(0.0, 907.9372 - Q.sum_pt) / 3.961002   # +0.1%  sum_pt < 907.9
        + 0.0009342449 * max(0.0, 0.1207452 - Q.girth) * max(0.0, Q.ptdr0_3 - 7.407874) / 0.03961536   # +0.1%  girth < 0.1207 and ptdr0_3 > 7.408
        - 0.0008370734 * max(0.0, Q.soft2_pt - 0.291748) * max(0.0, Q.dr1_9 - 0.1110681) / 0.02615207   # -0.1%  soft2_pt > 0.2917 and dr1_9 > 0.1111
        - 0.0007256678 * max(0.0, 949.9169 - Q.sum_pt) * max(0.0, Q.eccentricity - 0.6414784) / 0.7814919   # -0.1%  sum_pt < 949.9 and eccentricity > 0.6415
        - 0.0006445397 * max(0.0, Q.girth2 - 0.02928196) / 0.0002029538   # -0.1%  girth2 > 0.02928
        + 0.0006339935 * max(0.0, 907.9372 - Q.sum_pt) * max(0.0, 0.2992503 - Q.z_dr_0p05_0p1) / 0.6100122   # +0.1%  sum_pt < 907.9 and z_dr_0p05_0p1 < 0.2993
        + 0.0004918256 * max(0.0, 949.9169 - Q.sum_pt) * max(0.0, 0.05648804 - Q.absphi_3) / 0.1921085   # +0.0%  sum_pt < 949.9 and absphi_3 < 0.05649
        + 0.0003962081 * max(0.0, Q.psi_0p3 - 0.9985421) * max(0.0, Q.n_dr_0p1_0p2 - 19.0) / 0.0004021163   # +0.0%  psi_0p3 > 0.9985 and n_dr_0p1_0p2 > 19
        - 0.0003401257 * max(0.0, 907.9372 - Q.sum_pt) * max(0.0, Q.dr0_11 - 0.07072039) / 0.2449422   # -0.0%  sum_pt < 907.9 and dr0_11 > 0.07072
        + 0.0002673481 * max(0.0, 949.9169 - Q.sum_pt) * max(0.0, Q.dr0_11 - 0.2612223) / 0.07695826   # +0.0%  sum_pt < 949.9 and dr0_11 > 0.2612
        + 0.000161676 * max(0.0, 949.9169 - Q.sum_pt) * max(0.0, Q.dr0_4 - 0.2821771) / 0.038088   # +0.0%  sum_pt < 949.9 and dr0_4 > 0.2822
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 13.11;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.11143 * (0.02384321
        + 0.1260293 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +12.6%  e2_sq < 0.008184
        + 0.06945679 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +6.9%  LHA < 0.3332
        + 0.06495321 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # +6.5%  tau1 < 0.1073
        - 0.05418756 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -5.4%  girth2_top30 < 0.006364
        - 0.05127679 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -5.1%  e2_sq < 0.009606
        + 0.04708774 * max(0.0, 0.009606007 - Q.e2_sq) * max(0.0, 1115.723 - Q.sum_pt) / 0.2875932   # +4.7%  e2_sq < 0.009606 and sum_pt < 1116
        - 0.04610608 * max(0.0, 0.00818374 - Q.e2_sq) * max(0.0, 1115.723 - Q.sum_pt) / 0.1997244   # -4.6%  e2_sq < 0.008184 and sum_pt < 1116
        - 0.04340194 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # -4.3%  LHA < 0.3098
        - 0.04096158 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # -4.1%  e2 < 0.03876
        + 0.03653688 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # +3.7%  girth2_top30 < 0.01216
        - 0.03354674 * max(0.0, 0.1008497 - Q.tau1) / 0.02962944   # -3.4%  tau1 < 0.1008
        - 0.03055086 * max(0.0, Q.n_real_top50 - 22.0) / 19.58685   # -3.1%  n_real_top50 > 22
        + 0.03042687 * max(0.0, 0.005809485 - Q.girth2_top30) / 0.001402136   # +3.0%  girth2_top30 < 0.005809
        - 0.0274279 * max(0.0, 0.00634935 - Q.girth2_top50) / 0.001425305   # -2.7%  girth2_top50 < 0.006349
        - 0.02656481 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # -2.7%  girth2_top5 < 0.007164
        + 0.02355028 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 43.23581   # +2.4%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21
        - 0.0219408 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # -2.2%  girth < 0.07374
        + 0.02171307 * max(0.0, 0.007511864 - Q.e2_sq) / 0.002016671   # +2.2%  e2_sq < 0.007512
        - 0.01773841 * max(0.0, 4.450169 - Q.D2) / 2.013555   # -1.8%  D2 < 4.45
        + 0.01609698 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +1.6%  e2 < 0.02516
        + 0.0139806 * max(0.0, 0.00327978 - Q.girth2_top5) / 0.001239029   # +1.4%  girth2_top5 < 0.00328
        - 0.01393052 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # -1.4%  n_dr_0p2_0p4 < 10
        - 0.01347644 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2) / 0.005195774   # -1.3%  n_dr_0p2_0p4 < 10 and girth2 < 0.005532
        - 0.01304455 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -1.3%  lam1 < 0.005914
        - 0.01245478 * max(0.0, 0.09749958 - Q.girth) / 0.03657082   # -1.2%  girth < 0.0975
        - 0.01175917 * max(0.0, 0.004754578 - Q.e2_sq) / 0.0007999118   # -1.2%  e2_sq < 0.004755
        - 0.009070543 * max(0.0, Q.sum_pt_top15 - 840.7969) / 74.67343   # -0.9%  sum_pt_top15 > 840.8
        + 0.007222619 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 1156.659 - Q.sum_pt_top50) / 469.6702   # +0.7%  n_dr_0p2_0p4 < 10 and sum_pt_top50 < 1157
        - 0.006986971 * max(0.0, Q.z_top30_slots - 0.9734886) / 0.01006171   # -0.7%  z_top30_slots > 0.9735
        + 0.006918634 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # +0.7%  psi_0p2 > 0.9314
        + 0.006520093 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.7%  n_dr_0p2_0p4 < 6
        + 0.006018135 * max(0.0, 0.02210818 - Q.e2) / 0.00342102   # +0.6%  e2 < 0.02211
        - 0.005401207 * max(0.0, 0.005911134 - Q.zdr_0) / 0.001247991   # -0.5%  zdr_0 < 0.005911
        + 0.00534332 * max(0.0, 0.00130722 - Q.girth2_top10) / 0.0002794102   # +0.5%  girth2_top10 < 0.001307
        + 0.004922818 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 36.0 - Q.n_pt_above_5) / 42.8312   # +0.5%  n_dr_0p2_0p4 < 10 and n_pt_above_5 < 36
        + 0.003832612 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +0.4%  tau21_b2 < 0.2352
        - 0.003764019 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 62.25 - Q.pt_6) / 80.51398   # -0.4%  n_dr_0p2_0p4 < 10 and pt_6 < 62.25
        + 0.003663492 * max(0.0, Q.psi_0p2 - 0.9935324) / 0.001465572   # +0.4%  psi_0p2 > 0.9935
        + 0.003037876 * max(0.0, 0.02645404 - Q.dr_0) / 0.005170181   # +0.3%  dr_0 < 0.02645
        - 0.002976895 * max(0.0, Q.C2_b2 - 0.002289486) / 0.01267025   # -0.3%  C2_b2 > 0.002289
        + 0.002938141 * max(0.0, 4.450169 - Q.D2) * max(0.0, Q.pt_3 - 39.3125) / 66.38598   # +0.3%  D2 < 4.45 and pt_3 > 39.31
        + 0.002816618 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 0.2462041 - Q.dr_10) / 0.001724259   # +0.3%  z_top30_slots > 0.9735 and dr_10 < 0.2462
        + 0.002695392 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 1.08774 - Q.D2_b2) / 0.003455048   # +0.3%  z_top30_slots > 0.9735 and D2_b2 < 1.088
        + 0.002454047 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 0.2665639 - Q.dr0_12) / 0.001739509   # +0.2%  z_top30_slots > 0.9735 and dr0_12 < 0.2666
        - 0.00232905 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.2%  n_dr_0p1_0p2 < 8
        + 0.002282859 * max(0.0, 0.009606007 - Q.e2_sq) * max(0.0, Q.z_dr_0p1_0p2 - 0.1548383) / 4.668117e-05   # +0.2%  e2_sq < 0.009606 and z_dr_0p1_0p2 > 0.1548
        + 0.0006040629 * max(0.0, Q.sum_pt_top50 - 1245.697) / 7.470315   # +0.1%  sum_pt_top50 > 1246
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 6.826;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.826464 * (-0.03487816
        - 0.2982355 * max(0.0, 0.007511864 - Q.e2_sq) * max(0.0, Q.log_sum_pt - 6.811175) / 0.0003183593   # -29.8%  e2_sq < 0.007512 and log_sum_pt > 6.811
        + 0.2394183 * max(0.0, 0.006403325 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.811175) / 0.0002246937   # +23.9%  girth2 < 0.006403 and log_sum_pt > 6.811
        + 0.2131467 * max(0.0, 0.007511864 - Q.e2_sq) / 0.002016671   # +21.3%  e2_sq < 0.007512
        - 0.06237943 * max(0.0, 0.006403325 - Q.girth2) / 0.001406912   # -6.2%  girth2 < 0.006403
        + 0.03251158 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +3.3%  psi_0p3 > 0.9897
        - 0.02615485 * max(0.0, 0.005809485 - Q.girth2_top30) / 0.001402136   # -2.6%  girth2_top30 < 0.005809
        - 0.01891814 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -1.9%  psi_0p3 > 0.9943
        - 0.01515574 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -1.5%  z_dr_0_0p05 > 0.8789
        + 0.01186207 * max(0.0, 0.006403325 - Q.girth2) * max(0.0, 9.0 - Q.n_for_50pct) / 0.007394057   # +1.2%  girth2 < 0.006403 and n_for_50pct < 9
        + 0.0112041 * max(0.0, Q.psi_0p3 - 0.9943058) * max(0.0, 0.00751625 - Q.girth2) / 6.462915e-06   # +1.1%  psi_0p3 > 0.9943 and girth2 < 0.007516
        - 0.01045808 * max(0.0, 0.002752094 - Q.lam1) / 0.0003912817   # -1.0%  lam1 < 0.002752
        + 0.009916937 * max(0.0, 0.005809485 - Q.girth2_top30) * max(0.0, 0.07906904 - Q.z_dr_0p05_0p1) / 5.728554e-05   # +1.0%  girth2_top30 < 0.005809 and z_dr_0p05_0p1 < 0.07907
        + 0.007904244 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +0.8%  n_dr_0p2_0p4 < 10
        - 0.007631095 * max(0.0, Q.psi_0p1 - 0.9371031) / 0.01088804   # -0.8%  psi_0p1 > 0.9371
        + 0.003850263 * max(0.0, 0.007511864 - Q.e2_sq) * max(0.0, 32.09375 - Q.pt_10) / 0.02371056   # +0.4%  e2_sq < 0.007512 and pt_10 < 32.09
        - 0.003145448 * max(0.0, 6.879399 - Q.log_sum_pt) / 0.01083309   # -0.3%  log_sum_pt < 6.879
        + 0.003091883 * max(0.0, 0.007511864 - Q.e2_sq) * max(0.0, Q.zdr_0 - 0.005911134) / 2.222328e-06   # +0.3%  e2_sq < 0.007512 and zdr_0 > 0.005911
        + 0.002827193 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +0.3%  LHA > 0.372
        + 0.002420203 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.001864148 - Q.girth2_top2) / 3.894739e-06   # +0.2%  psi_0p3 > 0.9897 and girth2_top2 < 0.001864
        + 0.002343554 * max(0.0, Q.LHA - 0.3719813) * max(0.0, 0.2352054 - Q.tau21_b2) / 0.0002543919   # +0.2%  LHA > 0.372 and tau21_b2 < 0.2352
        - 0.002340487 * max(0.0, 0.007511864 - Q.e2_sq) * max(0.0, 34.0 - Q.n_real_top40) / 0.007490368   # -0.2%  e2_sq < 0.007512 and n_real_top40 < 34
        + 0.002193978 * max(0.0, 0.0004816552 - Q.girth2_top15) / 3.66329e-05   # +0.2%  girth2_top15 < 0.0004817
        + 0.002116113 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # +0.2%  log_sum_pt < 6.811
        - 0.002091217 * max(0.0, 0.08133662 - Q.psi_0p1) / 0.002484547   # -0.2%  psi_0p1 < 0.08134
        + 0.001775189 * max(0.0, 0.08133662 - Q.psi_0p1) * max(0.0, Q.sum_pt_top50 - 889.8503) / 0.2667141   # +0.2%  psi_0p1 < 0.08134 and sum_pt_top50 > 889.9
        - 0.001740302 * max(0.0, Q.sum_pt - 1085.125) * max(0.0, 0.019523 - Q.z_dr_0p2_0p4) / 0.1925692   # -0.2%  sum_pt > 1085 and z_dr_0p2_0p4 < 0.01952
        + 0.001403991 * max(0.0, Q.psi_0p1 - 0.9371031) * max(0.0, Q.dr_max_012 - 0.06112084) / 0.0001515273   # +0.1%  psi_0p1 > 0.9371 and dr_max_012 > 0.06112
        + 0.00134981 * max(0.0, Q.sum_pt - 1085.125) * max(0.0, 0.001868041 - Q.lam1) / 0.006895658   # +0.1%  sum_pt > 1085 and lam1 < 0.001868
        + 0.0009024702 * max(0.0, 0.08133662 - Q.psi_0p1) * max(0.0, 0.2708738 - Q.z_dr_0p2_0p4) / 0.0002508144   # +0.1%  psi_0p1 < 0.08134 and z_dr_0p2_0p4 < 0.2709
        - 0.0007105322 * max(0.0, 0.08133662 - Q.psi_0p1) * max(0.0, 0.02809026 - Q.girth2_top30) / 5.106683e-06   # -0.1%  psi_0p1 < 0.08134 and girth2_top30 < 0.02809
        - 0.0004568255 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, Q.dr0_3 - 0.1617791) / 9.525836e-05   # -0.0%  log_sum_pt < 6.811 and dr0_3 > 0.1618
        + 0.0003437853 * max(0.0, 0.004754578 - Q.e2_sq) / 0.0007999118   # +0.0%  e2_sq < 0.004755
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 17.46;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.45856 * (0.06251273
        + 0.2269434 * max(0.0, Q.log_sum_pt - 6.811175) / 0.1383141   # +22.7%  log_sum_pt > 6.811
        - 0.09162742 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -9.2%  sum_pt < 1085
        - 0.07607017 * max(0.0, Q.sum_pt_top50 - 889.8503) / 149.655   # -7.6%  sum_pt_top50 > 889.9
        - 0.06456672 * max(0.0, Q.e2_sq - 0.0292152) / 0.0002016952   # -6.5%  e2_sq > 0.02922
        - 0.06316161 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # -6.3%  log_sum_pt > 6.894
        + 0.04898658 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # +4.9%  sum_pt_top50 < 1079
        + 0.03918149 * max(0.0, Q.girth2_top40 - 0.01292642) * max(0.0, 1245.697 - Q.sum_pt_top50) / 0.6025464   # +3.9%  girth2_top40 > 0.01293 and sum_pt_top50 < 1246
        - 0.03403526 * max(0.0, Q.girth2_top40 - 0.02862386) * max(0.0, Q.z_6 - 0.01906139) / 3.587165e-06   # -3.4%  girth2_top40 > 0.02862 and z_6 > 0.01906
        + 0.03276619 * max(0.0, Q.girth2_top50 - 0.005852839) / 0.004485718   # +3.3%  girth2_top50 > 0.005853
        - 0.03178299 * max(0.0, Q.girth2_top40 - 0.008840538) / 0.003161074   # -3.2%  girth2_top40 > 0.008841
        - 0.03169203 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # -3.2%  sum_pt > 1017
        + 0.02937389 * max(0.0, Q.sum_pt_top50 - 1013.916) / 45.94021   # +2.9%  sum_pt_top50 > 1014
        - 0.02510268 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -2.5%  girth < 0.1207
        + 0.02386802 * max(0.0, Q.girth2_top40 - 0.02862386) / 0.0001970429   # +2.4%  girth2_top40 > 0.02862
        - 0.02205343 * max(0.0, Q.girth2_top50 - 0.01351431) / 0.002240755   # -2.2%  girth2_top50 > 0.01351
        - 0.02166913 * max(0.0, Q.sum_pt_top40 - 972.4111) / 67.15108   # -2.2%  sum_pt_top40 > 972.4
        + 0.02089572 * max(0.0, Q.sum_pt_top50 - 976.277) / 72.29277   # +2.1%  sum_pt_top50 > 976.3
        - 0.01607738 * max(0.0, Q.width - 0.01991191) / 0.001207633   # -1.6%  width > 0.01991
        + 0.0141959 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, Q.tau21 - 0.1295048) / 22.44785   # +1.4%  sum_pt < 1085 and tau21 > 0.1295
        - 0.01083435 * max(0.0, Q.girth2_top40 - 0.01292642) / 0.002259475   # -1.1%  girth2_top40 > 0.01293
        - 0.009942746 * max(0.0, Q.girth2_top40 - 0.02862386) * max(0.0, 62.25 - Q.pt_6) / 0.00503292   # -1.0%  girth2_top40 > 0.02862 and pt_6 < 62.25
        + 0.007199533 * max(0.0, Q.psi_0p3 - 0.9777125) / 0.01571542   # +0.7%  psi_0p3 > 0.9777
        + 0.006973334 * max(0.0, Q.e2 - 0.03680582) / 0.004603798   # +0.7%  e2 > 0.03681
        + 0.006012631 * max(0.0, Q.girth2_top40 - 0.02862386) * max(0.0, Q.pt_6 - 19.46875) / 0.003416206   # +0.6%  girth2_top40 > 0.02862 and pt_6 > 19.47
        - 0.004544033 * max(0.0, Q.girth2_top20 - 0.01083435) / 0.002303468   # -0.5%  girth2_top20 > 0.01083
        - 0.004184804 * max(0.0, 25.0 - Q.n_pt_above_5) / 2.617361   # -0.4%  n_pt_above_5 < 25
        + 0.004108468 * max(0.0, 1078.994 - Q.sum_pt_top50) * max(0.0, Q.tau32 - 0.577419) / 11.40784   # +0.4%  sum_pt_top50 < 1079 and tau32 > 0.5774
        - 0.003363393 * max(0.0, 54.0 - Q.n_particles) / 11.024   # -0.3%  n_particles < 54
        + 0.00274262 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +0.3%  e2 < 0.03481
        - 0.002618498 * max(0.0, 795.6047 - Q.sum_pt_top15) / 30.63834   # -0.3%  sum_pt_top15 < 795.6
        - 0.002553454 * max(0.0, Q.girth2_top50 - 0.005852839) * max(0.0, Q.soft7_z - 0.00146164) / 2.905625e-06   # -0.3%  girth2_top50 > 0.005853 and soft7_z > 0.001462
        + 0.002498356 * max(0.0, 25.0 - Q.n_pt_above_5) * max(0.0, 5.378975 - Q.D2) / 4.808335   # +0.2%  n_pt_above_5 < 25 and D2 < 5.379
        - 0.002490167 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, Q.C2 - 0.06655881) / 0.001737527   # -0.2%  log_sum_pt > 6.811 and C2 > 0.06656
        - 0.002445029 * max(0.0, Q.psi_0p3 - 0.9777125) * max(0.0, Q.sj2_zsoft - 0.1181474) / 0.002366695   # -0.2%  psi_0p3 > 0.9777 and sj2_zsoft > 0.1181
        + 0.002376902 * max(0.0, Q.sum_pt_top20 - 1064.139) / 11.11947   # +0.2%  sum_pt_top20 > 1064
        - 0.001953328 * max(0.0, Q.tau2 - 0.05936026) / 0.004947723   # -0.2%  tau2 > 0.05936
        - 0.001717076 * max(0.0, Q.sum_pt_top10 - 867.9266) / 28.75627   # -0.2%  sum_pt_top10 > 867.9
        - 0.001656962 * max(0.0, Q.sum_pt_top10 - 867.9266) * max(0.0, 0.06112084 - Q.dr_max_012) / 1.182144   # -0.2%  sum_pt_top10 > 867.9 and dr_max_012 < 0.06112
        - 0.00161574 * max(0.0, 0.1207452 - Q.girth) * max(0.0, Q.z_dr_0p05_0p1 - 0.1514163) / 0.006181466   # -0.2%  girth < 0.1207 and z_dr_0p05_0p1 > 0.1514
        - 0.001314191 * max(0.0, Q.sum_pt - 1167.447) / 14.21193   # -0.1%  sum_pt > 1167
        + 0.001199199 * max(0.0, Q.sum_pt_top40 - 972.4111) * max(0.0, 2.275391 - Q.soft1_pt) / 111.3032   # +0.1%  sum_pt_top40 > 972.4 and soft1_pt < 2.275
        + 0.0005131125 * max(0.0, Q.e2 - 0.03680582) * max(0.0, 0.2416266 - Q.sj2_zsoft) / 5.019245e-05   # +0.1%  e2 > 0.03681 and sj2_zsoft < 0.2416
        - 0.0005028519 * max(0.0, Q.log_sum_pt - 6.893714) * max(0.0, 0.577419 - Q.tau32) / 0.001578279   # -0.1%  log_sum_pt > 6.894 and tau32 < 0.5774
        - 0.0004743898 * max(0.0, Q.sum_pt - 1017.435) * max(0.0, Q.absphi_2 - 0.03259277) / 0.596772   # -0.0%  sum_pt > 1017 and absphi_2 > 0.03259
        - 0.000114792 * max(0.0, Q.girth2_top50 - 0.005852839) * max(0.0, Q.pt_4 - 90.625) / 0.0005349998   # -0.0%  girth2_top50 > 0.005853 and pt_4 > 90.62
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 20.69;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.68939 * (-0.0007205716
        + 0.1021048 * max(0.0, 0.01989454 - Q.e2_sq) / 0.01180402   # +10.2%  e2_sq < 0.01989
        + 0.07551564 * max(0.0, 0.01396296 - Q.e2_sq) / 0.00689253   # +7.6%  e2_sq < 0.01396
        - 0.06316311 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # -6.3%  girth2_top50 < 0.008125
        - 0.05477047 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -5.5%  girth2_top30 < 0.01216
        - 0.05062237 * max(0.0, 0.008190222 - Q.girth2) / 0.002454223   # -5.1%  girth2 < 0.00819
        + 0.04759949 * max(0.0, 0.007820315 - Q.girth2_top50) / 0.002264874   # +4.8%  girth2_top50 < 0.00782
        - 0.04615226 * max(0.0, 0.001784491 - Q.girth2) / 0.0001206749   # -4.6%  girth2 < 0.001784
        + 0.04167568 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # +4.2%  e2_sq < 0.009606
        - 0.04079365 * max(0.0, 0.01292642 - Q.girth2_top40) / 0.006292908   # -4.1%  girth2_top40 < 0.01293
        - 0.03420413 * max(0.0, 0.01951641 - Q.girth2_top50) / 0.01159099   # -3.4%  girth2_top50 < 0.01952
        - 0.02857552 * max(0.0, 0.006936725 - Q.e2_sq) / 0.001688614   # -2.9%  e2_sq < 0.006937
        + 0.02823307 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +2.8%  tau21_b2 < 0.3425
        - 0.02504034 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -2.5%  girth2_top20 < 0.01083
        - 0.02405302 * max(0.0, 0.1008497 - Q.tau1) / 0.02962944   # -2.4%  tau1 < 0.1008
        + 0.0239066 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +2.4%  psi_0p3 > 0.9956
        + 0.02386869 * max(0.0, 0.01215787 - Q.girth2_top30) * max(0.0, Q.sum_pt - 907.9372) / 0.9148907   # +2.4%  girth2_top30 < 0.01216 and sum_pt > 907.9
        - 0.02151462 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, 1225.842 - Q.sum_pt_top40) / 0.3755165   # -2.2%  psi_0p3 > 0.9956 and sum_pt_top40 < 1226
        + 0.02067141 * max(0.0, 0.00634935 - Q.girth2_top50) / 0.001425305   # +2.1%  girth2_top50 < 0.006349
        - 0.01991006 * max(0.0, Q.e2_sq - 0.00363788) / 0.006149555   # -2.0%  e2_sq > 0.003638
        - 0.01879776 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.02048524 - Q.lam1) / 0.001304679   # -1.9%  tau21_b2 < 0.3425 and lam1 < 0.02049
        - 0.01869923 * max(0.0, 0.00751625 - Q.width) / 0.002018105   # -1.9%  width < 0.007516
        + 0.01412719 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +1.4%  e2 < 0.03481
        - 0.01317812 * max(0.0, 0.008190222 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.9956185) / 5.477284e-06   # -1.3%  girth2 < 0.00819 and psi_0p3 > 0.9956
        + 0.01292465 * max(0.0, 0.006088416 - Q.girth2_top30) / 0.001533892   # +1.3%  girth2_top30 < 0.006088
        - 0.01264734 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # -1.3%  girth2_top30 < 0.008376
        - 0.01227054 * max(0.0, 0.1632346 - Q.LHA) / 0.009624216   # -1.2%  LHA < 0.1632
        + 0.007900745 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +0.8%  psi_0p3 > 0.9897
        - 0.007805856 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # -0.8%  n_dr_0p1_0p2 < 17
        + 0.007240792 * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) / 0.159722   # +0.7%  z_dr_0p1_0p2 < 0.2865
        + 0.007191257 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # +0.7%  girth2_top20 < 0.006374
        + 0.006944656 * max(0.0, 0.03409838 - Q.tau4) / 0.01737852   # +0.7%  tau4 < 0.0341
        + 0.006235785 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # +0.6%  tau1 < 0.07057
        - 0.006105068 * max(0.0, 0.3572263 - Q.N2) / 0.07385171   # -0.6%  N2 < 0.3572
        - 0.005545882 * max(0.0, 0.5936969 - Q.tau21) / 0.1891413   # -0.6%  tau21 < 0.5937
        + 0.005340637 * max(0.0, 0.00588499 - Q.girth2_top3) / 0.003020157   # +0.5%  girth2_top3 < 0.005885
        - 0.005026552 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.5%  log_sum_pt > 7.063
        - 0.004880186 * max(0.0, 0.008824206 - Q.zdr_1) / 0.00383206   # -0.5%  zdr_1 < 0.008824
        + 0.004819984 * max(0.0, 0.00625621 - Q.girth2_top10) / 0.002474789   # +0.5%  girth2_top10 < 0.006256
        - 0.004621835 * max(0.0, 0.2091025 - Q.LHA) / 0.02095782   # -0.5%  LHA < 0.2091
        - 0.00414168 * max(0.0, 0.03480688 - Q.e2) * max(0.0, Q.sum_pt_top30 - 886.3438) / 1.374127   # -0.4%  e2 < 0.03481 and sum_pt_top30 > 886.3
        + 0.004034089 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # +0.4%  tau1 < 0.05445
        - 0.003655211 * max(0.0, 0.001595561 - Q.soft7_z) / 0.0003533115   # -0.4%  soft7_z < 0.001596
        + 0.003346334 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, 1024.942 - Q.sum_pt_top40) / 0.04251006   # +0.3%  psi_0p3 > 0.9956 and sum_pt_top40 < 1025
        - 0.003279847 * max(0.0, 0.6289751 - Q.z_top5) / 0.101999   # -0.3%  z_top5 < 0.629
        - 0.003174238 * max(0.0, 28.0 - Q.n_pt_above_5) / 4.053793   # -0.3%  n_pt_above_5 < 28
        + 0.003171922 * max(0.0, Q.sj2_dr - 0.1666442) / 0.04983259   # +0.3%  sj2_dr > 0.1666
        + 0.002669547 * max(0.0, 0.008190222 - Q.girth2) * max(0.0, 0.002171436 - Q.soft8_z) / 1.962773e-06   # +0.3%  girth2 < 0.00819 and soft8_z < 0.002171
        + 0.002546072 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, 4.250195 - Q.soft6_pt) / 0.004557084   # +0.3%  psi_0p3 > 0.9956 and soft6_pt < 4.25
        - 0.002503895 * max(0.0, Q.sj2_dr - 0.2412757) / 0.01833576   # -0.3%  sj2_dr > 0.2413
        - 0.002380384 * max(0.0, 0.001155057 - Q.girth2_top3) / 0.000329166   # -0.2%  girth2_top3 < 0.001155
        - 0.002110839 * max(0.0, Q.LHA - 0.3098384) / 0.0195892   # -0.2%  LHA > 0.3098
        + 0.001619707 * max(0.0, Q.sum_pt_top50 - 1245.697) / 7.470315   # +0.2%  sum_pt_top50 > 1246
        + 0.001604661 * max(0.0, 0.1722488 - Q.tau21_b2) / 0.02681502   # +0.2%  tau21_b2 < 0.1722
        + 0.001295415 * max(0.0, 0.003626563 - Q.zdr_1) / 0.0007706791   # +0.1%  zdr_1 < 0.003627
        - 0.0009585945 * max(0.0, Q.sum_pt - 1260.541) / 7.655483   # -0.1%  sum_pt > 1261
        + 0.0009525672 * max(0.0, 0.01396296 - Q.e2_sq) * max(0.0, 0.9906378 - Q.z_top50_slots) / 1.282821e-05   # +0.1%  e2_sq < 0.01396 and z_top50_slots < 0.9906
        - 0.0005244546 * max(0.0, Q.n_pt_above_10 - 28.0) / 0.3744807   # -0.1%  n_pt_above_10 > 28
        + 0.0005190434 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, 0.05230713 - Q.absphi_5) / 0.0003293827   # +0.1%  log_sum_pt > 7.063 and absphi_5 < 0.05231
        - 0.0004496651 * max(0.0, Q.z_dr_0p05_0p1 - 0.7108211) * max(0.0, 0.7393686 - Q.pt2_over_pt0) / 0.003187587   # -0.0%  z_dr_0p05_0p1 > 0.7108 and pt2_over_pt0 < 0.7394
        - 0.0003828935 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, Q.sj3_dr13 - 0.228329) / 9.135563e-06   # -0.0%  psi_0p3 > 0.9956 and sj3_dr13 > 0.2283
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 9.633;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.632848 * (-0.06063044
        - 0.09131907 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -9.1%  girth2_top30 < 0.01216
        + 0.07067534 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # +7.1%  girth < 0.1207
        - 0.05660504 * max(0.0, 0.2941033 - Q.LHA) / 0.05718399   # -5.7%  LHA < 0.2941
        + 0.05563552 * max(0.0, 0.08068193 - Q.girth) / 0.02368455   # +5.6%  girth < 0.08068
        + 0.04620562 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # +4.6%  e2 < 0.04083
        - 0.03921416 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -3.9%  tau1 < 0.07057
        + 0.03865318 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # +3.9%  girth2_top30 < 0.008376
        - 0.0326095 * max(0.0, 0.00751625 - Q.girth2) / 0.002018105   # -3.3%  girth2 < 0.007516
        + 0.0310339 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +3.1%  girth2_top5 < 0.00833
        + 0.0273463 * max(0.0, 0.002396991 - Q.lam2) / 0.001466253   # +2.7%  lam2 < 0.002397
        + 0.02470146 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +2.5%  sj2_dr > 0.2232
        + 0.02419777 * max(0.0, 6.97212 - Q.log_sum_pt) / 0.05249646   # +2.4%  log_sum_pt < 6.972
        + 0.02359314 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +2.4%  z_dr_0p1_0p2 < 0.1203
        + 0.0218854 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # +2.2%  girth2_top10 < 0.007679
        + 0.02185119 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # +2.2%  lam1 < 0.00619
        + 0.02043656 * max(0.0, Q.sum_pt_top50 - 1107.226) / 19.65456   # +2.0%  sum_pt_top50 > 1107
        - 0.0178415 * max(0.0, Q.sj2_dr - 0.1745007) / 0.04529905   # -1.8%  sj2_dr > 0.1745
        + 0.01775423 * max(0.0, Q.sum_pt_top40 - 858.8262) / 166.7848   # +1.8%  sum_pt_top40 > 858.8
        - 0.0160363 * max(0.0, Q.sum_pt_top50 - 1024.689) / 40.56456   # -1.6%  sum_pt_top50 > 1025
        - 0.01601798 * max(0.0, 0.1778185 - Q.sd_rg) * max(0.0, Q.soft5_dr - 0.04139378) / 0.009084795   # -1.6%  sd_rg < 0.1778 and soft5_dr > 0.04139
        + 0.01566254 * max(0.0, 0.1778185 - Q.sd_rg) * max(0.0, Q.soft5_dr0 - 0.03721986) / 0.009661509   # +1.6%  sd_rg < 0.1778 and soft5_dr0 > 0.03722
        - 0.01516773 * max(0.0, 0.002270363 - Q.girth2_top5) / 0.0007634099   # -1.5%  girth2_top5 < 0.00227
        - 0.01512566 * max(0.0, 4.646151e-05 - Q.e3) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 0.0002000803   # -1.5%  e3 < 4.646e-05 and n_dr_0p1_0p2 < 21
        - 0.01492953 * max(0.0, Q.sum_pt_top40 - 1095.686) / 18.59291   # -1.5%  sum_pt_top40 > 1096
        - 0.01490774 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -1.5%  psi_0p1 > 0.8976
        + 0.01392814 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 184.0097   # +1.4%  n_dr_0p2_0p4 < 26 and n_dr_0p1_0p2 < 21
        + 0.01387534 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +1.4%  tau1 < 0.06311
        - 0.01368875 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.0007392963   # -1.4%  z_dr_0p1_0p2 < 0.1203 and psi_0p3 > 0.9777
        + 0.01275602 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 0.002396991 - Q.lam2) / 8.322782e-05   # +1.3%  z_dr_0p1_0p2 < 0.1203 and lam2 < 0.002397
        - 0.01267466 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -1.3%  sum_pt < 1002
        - 0.01230261 * max(0.0, 0.9860575 - Q.z_top30_slots) / 0.03906901   # -1.2%  z_top30_slots < 0.9861
        - 0.01076964 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.0002825377   # -1.1%  girth2_top5 < 0.00833 and psi_0p3 > 0.9299
        + 0.01023311 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) / 0.3666529   # +1.0%  z_dr_0_0p05 < 0.8459
        + 0.009800634 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, 46.0 - Q.n_real_top50) / 0.03635692   # +1.0%  girth2_top5 < 0.00833 and n_real_top50 < 46
        - 0.009769857 * max(0.0, 1.976207 - Q.D2) / 0.367701   # -1.0%  D2 < 1.976
        - 0.009200753 * max(0.0, 0.02647293 - Q.z_dr_0p2_0p4) / 0.01168948   # -0.9%  z_dr_0p2_0p4 < 0.02647
        - 0.008313585 * max(0.0, 0.006805717 - Q.girth2_top50) / 0.001665679   # -0.8%  girth2_top50 < 0.006806
        - 0.008280372 * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0004356739   # -0.8%  psi_0p3 > 0.9985
        + 0.008081374 * max(0.0, Q.sum_pt_top20 - 869.693) / 87.9958   # +0.8%  sum_pt_top20 > 869.7
        + 0.006244074 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # +0.6%  girth2_top30 < 0.005402
        - 0.00614309 * max(0.0, 0.002247756 - Q.girth2_top10) / 0.0005836006   # -0.6%  girth2_top10 < 0.002248
        - 0.005627228 * max(0.0, Q.sj2_dr - 0.2780918) / 0.009227968   # -0.6%  sj2_dr > 0.2781
        - 0.00562687 * max(0.0, Q.sum_pt_top30 - 988.4375) / 43.28413   # -0.6%  sum_pt_top30 > 988.4
        + 0.0054283 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 0.05112769 - Q.C2_b2) / 0.001667822   # +0.5%  z_dr_0p1_0p2 < 0.1203 and C2_b2 < 0.05113
        + 0.004689037 * max(0.0, 4.646151e-05 - Q.e3) / 1.427904e-05   # +0.5%  e3 < 4.646e-05
        - 0.004066004 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 995.6769 - Q.sum_pt) / 0.6029747   # -0.4%  z_dr_0p1_0p2 < 0.1203 and sum_pt < 995.7
        - 0.004027085 * max(0.0, 0.02970886 - Q.absphi_0) / 0.01119853   # -0.4%  absphi_0 < 0.02971
        + 0.003418911 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) / 2.171439   # +0.3%  n_dr_0p1_0p2 < 10
        + 0.003259175 * max(0.0, 0.000341301 - Q.girth2_top3) / 6.040252e-05   # +0.3%  girth2_top3 < 0.0003413
        + 0.003203787 * max(0.0, Q.sum_pt_top30 - 988.4375) * max(0.0, 6.70332 - Q.soft10_pt) / 177.4515   # +0.3%  sum_pt_top30 > 988.4 and soft10_pt < 6.703
        - 0.00308147 * max(0.0, Q.sum_pt_top50 - 1003.544) / 52.17558   # -0.3%  sum_pt_top50 > 1004
        + 0.002977423 * max(0.0, 0.1778185 - Q.sd_rg) * max(0.0, 0.6025827 - Q.planar_flow) / 0.006056217   # +0.3%  sd_rg < 0.1778 and planar_flow < 0.6026
        + 0.002969945 * max(0.0, Q.psi_0p3 - 0.9985421) * max(0.0, 0.1613937 - Q.dr_10) / 3.823369e-05   # +0.3%  psi_0p3 > 0.9985 and dr_10 < 0.1614
        + 0.002690878 * max(0.0, Q.sum_pt_top50 - 1156.659) / 13.71464   # +0.3%  sum_pt_top50 > 1157
        + 0.002510354 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 986.0565 - Q.sum_pt) / 0.5137322   # +0.3%  z_dr_0p1_0p2 < 0.1203 and sum_pt < 986.1
        - 0.002169886 * max(0.0, 0.001155057 - Q.girth2_top3) * max(0.0, Q.eccentricity - 0.773315) / 9.102993e-06   # -0.2%  girth2_top3 < 0.001155 and eccentricity > 0.7733
        - 0.002069301 * max(0.0, Q.sum_pt_top50 - 1024.689) * max(0.0, 9.176976 - Q.ptdr0_8) / 282.9311   # -0.2%  sum_pt_top50 > 1025 and ptdr0_8 < 9.177
        - 0.001967938 * max(0.0, 0.1778185 - Q.sd_rg) / 0.06576479   # -0.2%  sd_rg < 0.1778
        - 0.001788334 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 10.0) / 0.006625563   # -0.2%  girth2_top5 < 0.00833 and n_dr_0p2_0p4 > 10
        - 0.001141566 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 0.8706159 - Q.psi_0p2) / 0.0003752255   # -0.1%  z_dr_0p1_0p2 < 0.1203 and psi_0p2 < 0.8706
        + 0.0009975734 * max(0.0, Q.sum_pt_top50 - 1156.659) * max(0.0, 0.1381171 - Q.dr0_8) / 1.27388   # +0.1%  sum_pt_top50 > 1157 and dr0_8 < 0.1381
        - 0.0008205579 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # -0.1%  log_sum_pt > 7.139
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6451536764705882, 2.7578877100840336, 0.21139579831932773, 0.380475, 0.7659806722689075, 1.0322903361344538, 0.5347237394957983, 0.4540234243697479, 1.158417331932773, 0.9860628151260504, 1.251057037815126, 0.7601461134453782, 0.547509243697479, 1.525241281512605, 0.2633726890756303, 0.43218960084033614]
T = [3.777919897255777, 2.4449049041491593, 4.173858439469538, 4.272209036896008, 4.061345674402573]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +46%, n4 -18%, n9 +13%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.456251 * h[1] / H_AVG[1]
            - 0.177408 * h[4] / H_AVG[4]
            + 0.1345816 * h[9] / H_AVG[9]
            - 0.08538845 * h[5] / H_AVG[5]
            - 0.07553264 * h[3] / H_AVG[3]
            + 0.03396644 * h[12] / H_AVG[12]
            + 0.0221155 * h[6] / H_AVG[6]
            + 0.009582136 * h[8] / H_AVG[8]
            - 0.005174214 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +23%, n1 -21%, n11 -8%, n12 +8%, n6 +5% ...
            - 0.3328778 * h[4] / H_AVG[4]
            + 0.2268638 * h[9] / H_AVG[9]
            - 0.2115027 * h[1] / H_AVG[1]
            - 0.07772757 * h[11] / H_AVG[11]
            + 0.07697899 * h[12] / H_AVG[12]
            + 0.04784269 * h[6] / H_AVG[6]
            + 0.01080798 * h[2] / H_AVG[2]
            - 0.007995307 * h[10] / H_AVG[10]
            + 0.007403262 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +14%, n0 +12%, n11 +11%, n14 -9%, n7 -7% ...
            - 0.2428485 * h[8] / H_AVG[8]
            + 0.1429835 * h[5] / H_AVG[5]
            + 0.1159276 * h[0] / H_AVG[0]
            + 0.1081342 * h[11] / H_AVG[11]
            - 0.08676323 * h[14] / H_AVG[14]
            - 0.06798617 * h[7] / H_AVG[7]
            + 0.06308452 * h[4] / H_AVG[4]
            - 0.05329017 * h[12] / H_AVG[12]
            - 0.0516791 * h[9] / H_AVG[9]
            + 0.03988104 * h[3] / H_AVG[3]
            - 0.01941502 * h[15] / H_AVG[15]
            + 0.008007036 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -25%, n0 -21%, n5 +17%, n6 -12%, n7 +10%, n15 +6% ...
            - 0.2542048 * h[8] / H_AVG[8]
            - 0.2076411 * h[0] / H_AVG[0]
            + 0.1661201 * h[5] / H_AVG[5]
            - 0.1212519 * h[6] / H_AVG[6]
            + 0.09631053 * h[7] / H_AVG[7]
            + 0.0569042 * h[15] / H_AVG[15]
            + 0.04004875 * h[12] / H_AVG[12]
            + 0.03896294 * h[3] / H_AVG[3]
            - 0.01855561 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +30%, n5 -12%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3403428 * h[13] / H_AVG[13]
            + 0.3032269 * h[10] / H_AVG[10]
            - 0.1191443 * h[5] / H_AVG[5]
            + 0.06016569 * h[8] / H_AVG[8]
            - 0.05055368 * h[12] / H_AVG[12]
            - 0.03990576 * h[15] / H_AVG[15]
            + 0.035363 * h[4] / H_AVG[4]
            - 0.03144132 * h[7] / H_AVG[7]
            + 0.01985652 * h[0] / H_AVG[0]
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
