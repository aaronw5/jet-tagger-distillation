"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.6%   (on for 42% of jets)
  neuron  5:  11.4%   (on for 79% of jets)
  neuron  1:  11.0%   (on for 93% of jets)
  neuron  4:  10.8%   (on for 68% of jets)
  neuron  0:   7.7%   (on for 59% of jets)
  neuron 13:   6.7%   (on for 86% of jets)
  neuron  9:   6.7%   (on for 60% of jets)
  neuron 10:   6.5%   (on for 82% of jets)
  neuron 12:   5.4%   (on for 39% of jets)
  neuron  7:   5.1%   (on for 54% of jets)
  neuron  6:   4.2%   (on for 51% of jets)
  neuron  3:   4.0%   (on for 70% of jets)
  neuron 11:   2.4%   (on for 57% of jets)
  neuron 14:   2.4%   (on for 35% of jets)
  neuron 15:   1.6%   (on for 48% of jets)
  neuron  2:   0.6%   (on for 74% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.0% (the network: 81.1%); same class as the network for 91.1% of jets.

Quantities:
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
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top20           number of real particles among the 20 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_10                  pT of particle 10 [GeV]
  Q.ptdr0_10               pT10 · ΔR(0, 10) [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_12                  pT of particle 12 [GeV]
  Q.pt_13                  pT of particle 13 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.pt_8                   pT of particle 8 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_pt               pT [GeV] of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_0                    pT of particle 0 / total pT
  Q.z_10                   pT of particle 10 / total pT
  Q.z_14                   pT of particle 14 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_8                    pT of particle 8 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_z               pT share of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_z                pT share of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_14                 pT share × ΔR of particle 14 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.z_1st                  largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_13              |Δφ| of particle 13
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft5_dr0              ΔR between the hardest and the 5. softest real particle (0 if among the 15 hardest)
  Q.dr0_11                 ΔR between particle 11 and the hardest particle
  Q.dr0_5                  ΔR between particle 5 and the hardest particle
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_11                 ΔR between particle 11 and the 2nd-hardest particle
  Q.dr1_12                 ΔR between particle 12 and the 2nd-hardest particle
  Q.dr1_13                 ΔR between particle 13 and the 2nd-hardest particle
  Q.dr1_9                  ΔR between particle 9 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_10                  ΔR of particle 10 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.soft10_dr              ΔR from the jet axis of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_dr               ΔR from the jet axis of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_dr               ΔR from the jet axis of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_9                  Δη of particle 9
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
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
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top20=sum(1 for x in pt[:20] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_0=pt[0],
        pt_1=pt[1],
        pt_10=pt[10],
        ptdr0_10=pt[10] * math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        pt_11=pt[11],
        pt_12=pt[12],
        pt_13=pt[13],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        pt_5=pt[5],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        pt_8=pt[8],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft2_pt=softp(2, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        soft9_pt=softp(9, 'pt'),
        z_0=z[0],
        z_10=z[10],
        z_14=z[14],
        z_3=z[3],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        z_8=z[8],
        soft1_z=softp(1, 'z'),
        soft10_z=softp(10, 'z'),
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft7_z=softp(7, 'z'),
        soft9_z=softp(9, 'z'),
        sj3_z3=subjets(3)["z"][2],
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top3_slots=sum(pt[:3]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_14=z[14] * dr[14],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        zdr_5=z[5] * dr[5],
        z_1st=zs[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_zg=softdrop("zg"),
        absphi_1=abs(phi[1]),
        absphi_13=abs(phi[13]),
        sj2_dr=subjets(2)["dr"][0],
        soft5_dr0=softp(5, 'dr0'),
        dr0_11=math.sqrt(dist2(0, 11)) if pt[11] > 0 else 0.0,
        dr0_5=math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_11=math.sqrt(dist2(1, 11)) if pt[11] > 0 else 0.0,
        dr1_12=math.sqrt(dist2(1, 12)) if pt[12] > 0 else 0.0,
        dr1_13=math.sqrt(dist2(1, 13)) if pt[13] > 0 else 0.0,
        dr1_9=math.sqrt(dist2(1, 9)) if pt[9] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_10=dr[10] if pt[10] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        soft10_dr=softp(10, 'dr'),
        soft6_dr=softp(6, 'dr'),
        soft7_dr=softp(7, 'dr'),
        eta_0=eta[0],
        eta_1=eta[1],
        eta_9=eta[9],
        phi_0=phi[0],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
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
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    # scale S = 29.88;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.8776 * (-0.02389944
        + 0.1531401 * max(0.0, 0.08068193 - Q.girth) / 0.02368455   # +15.3%  girth < 0.08068
        - 0.1506365 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # -15.1%  LHA < 0.3098
        + 0.07750006 * max(0.0, 0.007887677 - Q.girth2_top15) / 0.00326329   # +7.8%  girth2_top15 < 0.007888
        - 0.07591275 * max(0.0, 0.09749958 - Q.girth) / 0.03657082   # -7.6%  girth < 0.0975
        + 0.06721357 * max(0.0, 0.3203321 - Q.LHA) / 0.07499257   # +6.7%  LHA < 0.3203
        - 0.06361532 * max(0.0, Q.sj2_dr - 0.1512157) / 0.05988199   # -6.4%  sj2_dr > 0.1512
        - 0.06130947 * max(0.0, 0.009962397 - Q.girth2_top15) / 0.004894699   # -6.1%  girth2_top15 < 0.009962
        + 0.05325076 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +5.3%  LHA < 0.3332
        + 0.03716617 * max(0.0, Q.sj2_dr - 0.1411617) / 0.06721719   # +3.7%  sj2_dr > 0.1412
        + 0.02989012 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_pt_top50 - 889.8503) / 1563.946   # +3.0%  n_dr_0p2_0p4 < 18 and sum_pt_top50 > 889.9
        - 0.02210054 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # -2.2%  dr_0 < 0.06413
        + 0.01718562 * max(0.0, Q.sj2_dr - 0.1745007) / 0.04529905   # +1.7%  sj2_dr > 0.1745
        - 0.01629464 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.log_sum_pt - 6.920349) / 0.4496771   # -1.6%  n_dr_0p2_0p4 < 18 and log_sum_pt > 6.92
        - 0.01598613 * max(0.0, 0.04466492 - Q.tau1) / 0.005217805   # -1.6%  tau1 < 0.04466
        - 0.01576089 * max(0.0, 0.005383629 - Q.girth2_top15) / 0.001666876   # -1.6%  girth2_top15 < 0.005384
        + 0.01423227 * max(0.0, 0.05775119 - Q.dr_0) / 0.02089184   # +1.4%  dr_0 < 0.05775
        + 0.01098026 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +1.1%  psi_0p3 > 0.9897
        + 0.01026708 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +1.0%  n_dr_0p2_0p4 < 18
        - 0.009576847 * max(0.0, Q.sum_pt_top40 - 1001.523) / 47.19068   # -1.0%  sum_pt_top40 > 1002
        - 0.009497566 * max(0.0, 2.883342e-05 - Q.e3) / 6.471346e-06   # -0.9%  e3 < 2.883e-05
        - 0.008899144 * max(0.0, 0.05048381 - Q.girth) / 0.008533025   # -0.9%  girth < 0.05048
        - 0.007796656 * max(0.0, 0.1623049 - Q.sj3_dr_max) / 0.01012502   # -0.8%  sj3_dr_max < 0.1623
        + 0.007610032 * max(0.0, 0.2125209 - Q.sj3_dr_max) / 0.0267777   # +0.8%  sj3_dr_max < 0.2125
        + 0.006298489 * max(0.0, 5.13841e-05 - Q.e3) / 1.709557e-05   # +0.6%  e3 < 5.138e-05
        + 0.005580397 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, 26.0 - Q.n_dr_0p1_0p2) / 0.02963093   # +0.6%  psi_0p3 > 0.9956 and n_dr_0p1_0p2 < 26
        + 0.004955153 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # +0.5%  LHA < 0.187
        + 0.004936922 * max(0.0, 0.0008296372 - Q.lam2) / 0.0002681038   # +0.5%  lam2 < 0.0008296
        - 0.00448547 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0223982 - Q.C3) / 0.1512275   # -0.4%  n_dr_0p2_0p4 < 18 and C3 < 0.0224
        - 0.004326783 * max(0.0, Q.sj2_dr - 0.06289464) / 0.1327343   # -0.4%  sj2_dr > 0.06289
        - 0.004126418 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.girth2_top10 - 0.004752876) / 0.01391948   # -0.4%  n_dr_0p2_0p4 < 18 and girth2_top10 > 0.004753
        - 0.003327749 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # -0.3%  e2 < 0.0303
        - 0.003208063 * max(0.0, 0.1825048 - Q.sj2_dr) / 0.03087146   # -0.3%  sj2_dr < 0.1825
        - 0.003034512 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.tau21_b2 - 0.2018786) / 1.609082   # -0.3%  n_dr_0p2_0p4 < 18 and tau21_b2 > 0.2019
        + 0.002916673 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +0.3%  e2 < 0.01879
        + 0.002895346 * max(0.0, Q.psi_0p1 - 0.8747961) / 0.03440015   # +0.3%  psi_0p1 > 0.8748
        - 0.002215476 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # -0.2%  psi_0p3 > 0.9956
        + 0.002214678 * max(0.0, 0.1210264 - Q.sj3_dr_max) / 0.004637661   # +0.2%  sj3_dr_max < 0.121
        - 0.002154488 * max(0.0, 0.08589404 - Q.girth) / 0.02745918   # -0.2%  girth < 0.08589
        + 0.00163695 * max(0.0, Q.sum_pt_top40 - 1001.523) * max(0.0, 0.1512157 - Q.sj2_dr) / 1.305424   # +0.2%  sum_pt_top40 > 1002 and sj2_dr < 0.1512
        + 0.00115788 * max(0.0, Q.sum_pt_top2 - 405.0) / 51.45192   # +0.1%  sum_pt_top2 > 405
        + 0.001135572 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # +0.1%  log_sum_pt > 7.017
        + 0.001062665 * max(0.0, Q.sum_pt_top40 - 1001.523) * max(0.0, 0.02652372 - Q.dr_5) / 0.2748573   # +0.1%  sum_pt_top40 > 1002 and dr_5 < 0.02652
        + 0.0009040339 * max(0.0, 0.08589404 - Q.girth) * max(0.0, 0.4947602 - Q.z_dr_0_0p05) / 0.0006313374   # +0.1%  girth < 0.08589 and z_dr_0_0p05 < 0.4948
        - 0.0005687038 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, 0.02652372 - Q.dr_5) / 6.727825e-06   # -0.1%  psi_0p3 > 0.9956 and dr_5 < 0.02652
        + 0.0005461929 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # +0.1%  log_sum_pt > 7.063
        - 0.0004889572 * max(0.0, Q.sum_pt_top30 - 1052.08) / 20.84425   # -0.0%  sum_pt_top30 > 1052
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 16.88;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.88459 * (0.08192482
        - 0.1391608 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -13.9%  log_sum_pt < 7.139
        + 0.1236139 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +12.4%  log_sum_pt > 6.91
        + 0.1088898 * max(0.0, Q.sum_pt - 972.0419) / 81.34075   # +10.9%  sum_pt > 972
        - 0.06719881 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -6.7%  sum_pt_top50 > 959.1
        + 0.06700251 * max(0.0, 1225.842 - Q.sum_pt_top40) / 211.117   # +6.7%  sum_pt_top40 < 1226
        + 0.06108868 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # +6.1%  sum_pt_top50 < 1079
        - 0.04088478 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -4.1%  log_sum_pt > 6.989
        - 0.04045405 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # -4.0%  sum_pt_top50 > 934.2
        + 0.03036999 * max(0.0, 605.875 - Q.sum_pt_top2) / 245.9717   # +3.0%  sum_pt_top2 < 605.9
        + 0.02817848 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +2.8%  tau1 < 0.07709
        + 0.02680826 * max(0.0, Q.n_particles - 38.0) * max(0.0, Q.tau32 - 0.3293142) / 3.693221   # +2.7%  n_particles > 38 and tau32 > 0.3293
        - 0.02252136 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.002181998 - Q.soft1_z) / 0.01445558   # -2.3%  n_particles > 38 and soft1_z < 0.002182
        + 0.02119093 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +2.1%  n_particles > 38
        - 0.01906637 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # -1.9%  lam2 < 0.001776
        - 0.01624157 * max(0.0, 29.04688 - Q.pt_11) / 8.607263   # -1.6%  pt_11 < 29.05
        - 0.01370844 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -1.4%  n_dr_0p2_0p4 < 7
        - 0.01202652 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.9624339 - Q.tau43) / 1.418537   # -1.2%  n_particles > 38 and tau43 < 0.9624
        - 0.01168332 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) * max(0.0, 10.0 - Q.n_dr_0p05_0p1) / 0.2065903   # -1.2%  z_dr_0_0p05 > 0.8103 and n_dr_0p05_0p1 < 10
        - 0.01038787 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # -1.0%  psi_0p3 > 0.9966
        - 0.009885378 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -1.0%  sum_pt > 1053
        + 0.00898947 * max(0.0, Q.z_top30_slots - 0.9341838) / 0.03406818   # +0.9%  z_top30_slots > 0.9342
        + 0.007852788 * max(0.0, Q.n_dr_0_0p05 - 15.0) / 2.713461   # +0.8%  n_dr_0_0p05 > 15
        + 0.007736111 * max(0.0, 0.2140276 - Q.tau21) / 0.01160104   # +0.8%  tau21 < 0.214
        + 0.007190014 * max(0.0, 0.05602756 - Q.C2) / 0.009882647   # +0.7%  C2 < 0.05603
        - 0.006903531 * max(0.0, 3.852812 - Q.D2_b2) / 1.876829   # -0.7%  D2_b2 < 3.853
        - 0.006819382 * max(0.0, 7.0 - Q.n_dr_0p1_0p2) / 0.9740454   # -0.7%  n_dr_0p1_0p2 < 7
        - 0.006047914 * max(0.0, 0.03457336 - Q.M3) / 0.01113671   # -0.6%  M3 < 0.03457
        - 0.00600032 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 17.94219 - Q.ptdr0_3) / 0.4842322   # -0.6%  z_top30_slots > 0.9342 and ptdr0_3 < 17.94
        + 0.005987628 * max(0.0, 0.0007894752 - Q.girth2_top15) / 9.062109e-05   # +0.6%  girth2_top15 < 0.0007895
        - 0.005965678 * max(0.0, Q.z_top5 - 0.6551948) * max(0.0, 28.39396 - Q.pt1_dr01) / 0.8332473   # -0.6%  z_top5 > 0.6552 and pt1_dr01 < 28.39
        + 0.005905397 * max(0.0, 2.5 - Q.soft9_pt) / 0.7352391   # +0.6%  soft9_pt < 2.5
        - 0.005186881 * max(0.0, 0.03457336 - Q.M3) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.0006031662   # -0.5%  M3 < 0.03457 and psi_0p3 > 0.9299
        + 0.005086681 * max(0.0, 2.117188 - Q.soft5_pt) / 0.8827838   # +0.5%  soft5_pt < 2.117
        + 0.005055268 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1178619 - Q.absphi_1) / 0.8192292   # +0.5%  n_particles > 38 and absphi_1 < 0.1179
        + 0.004942146 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) / 0.03630126   # +0.5%  z_dr_0_0p05 > 0.8103
        - 0.004601797 * max(0.0, 2.117188 - Q.soft5_pt) * max(0.0, 0.4357228 - Q.max_dr) / 0.06450036   # -0.5%  soft5_pt < 2.117 and max_dr < 0.4357
        + 0.004334923 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.009970338 - Q.zdr_0) / 0.03317992   # +0.4%  n_particles > 38 and zdr_0 < 0.00997
        - 0.003820334 * max(0.0, 0.002197765 - Q.girth2_top15) / 0.0004509993   # -0.4%  girth2_top15 < 0.002198
        + 0.002921421 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.pt1_dr01 - 5.351077) / 0.1255531   # +0.3%  z_top30_slots > 0.9342 and pt1_dr01 > 5.351
        + 0.002904779 * max(0.0, 29.04688 - Q.pt_11) * max(0.0, 0.1745719 - Q.dr1_12) / 0.7333922   # +0.3%  pt_11 < 29.05 and dr1_12 < 0.1746
        - 0.002657782 * max(0.0, 36.65625 - Q.pt_8) / 8.068046   # -0.3%  pt_8 < 36.66
        - 0.002402409 * max(0.0, 0.2773918 - Q.D2_b2) / 0.01604103   # -0.2%  D2_b2 < 0.2774
        - 0.001905338 * max(0.0, Q.tau1 - 0.1219132) / 0.01033631   # -0.2%  tau1 > 0.1219
        - 0.001845212 * max(0.0, 0.0007143144 - Q.lam2) / 0.0002017367   # -0.2%  lam2 < 0.0007143
        + 0.001802675 * max(0.0, 0.002197765 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9966167) / 5.451807e-07   # +0.2%  girth2_top15 < 0.002198 and psi_0p3 > 0.9966
        + 0.001459714 * max(0.0, Q.z_top5 - 0.6551948) / 0.03331427   # +0.1%  z_top5 > 0.6552
        + 0.001317429 * max(0.0, Q.sj3_dr23 - 0.2799759) / 0.0237761   # +0.1%  sj3_dr23 > 0.28
        - 0.001191783 * max(0.0, Q.n_dr_0_0p05 - 30.0) / 0.212479   # -0.1%  n_dr_0_0p05 > 30
        + 0.0008034886 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) * max(0.0, 20.0 - Q.n_real_top20) / 0.01239824   # +0.1%  z_dr_0_0p05 > 0.8103 and n_real_top20 < 20
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 2.81;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.809793 * (-0.01227608
        - 0.1616592 * max(0.0, Q.sum_pt_top50 - 997.0189) / 56.58644   # -16.2%  sum_pt_top50 > 997
        - 0.0772908 * max(0.0, 0.0009731947 - Q.lam2) / 0.0003580574   # -7.7%  lam2 < 0.0009732
        + 0.06919668 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15) / 0.0009227558   # +6.9%  log_sum_pt > 6.903 and girth2_top15 < 0.02147
        + 0.06725862 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.8065577 - Q.tau21) / 0.01865613   # +6.7%  log_sum_pt > 6.903 and tau21 < 0.8066
        + 0.06574928 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +6.6%  lam2 < 0.001776
        + 0.06200401 * max(0.0, Q.log_sum_pt - 6.930088) / 0.0397436   # +6.2%  log_sum_pt > 6.93
        + 0.04736013 * max(0.0, Q.sum_pt_top50 - 997.0189) * max(0.0, 0.006794973 - Q.zdr_4) / 0.2427126   # +4.7%  sum_pt_top50 > 997 and zdr_4 < 0.006795
        - 0.04580628 * max(0.0, Q.sum_pt - 1085.125) / 25.7173   # -4.6%  sum_pt > 1085
        - 0.04178666 * max(0.0, 0.03577037 - Q.girth) / 0.004182456   # -4.2%  girth < 0.03577
        + 0.03775719 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.1206357 - Q.dr_max_012) / 0.00414901   # +3.8%  log_sum_pt > 6.903 and dr_max_012 < 0.1206
        + 0.03401824 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # +3.4%  LHA < 0.187
        + 0.03102706 * Q.M3 / 0.02697224   # +3.1%  M3
        + 0.0299969 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +3.0%  sum_pt_top40 > 1070
        - 0.02743595 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.8747961 - Q.psi_0p1) / 0.005466701   # -2.7%  log_sum_pt > 6.903 and psi_0p1 < 0.8748
        + 0.02383529 * max(0.0, 0.04568661 - Q.z_6) / 0.008954882   # +2.4%  z_6 < 0.04569
        + 0.02211562 * max(0.0, Q.sum_pt_top50 - 997.0189) * max(0.0, 0.2162512 - Q.dr_6) / 9.011274   # +2.2%  sum_pt_top50 > 997 and dr_6 < 0.2163
        - 0.0216486 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.zdr_3 - 0.00453462) / 5.033547e-05   # -2.2%  log_sum_pt > 6.903 and zdr_3 > 0.004535
        - 0.01824386 * max(0.0, Q.sum_pt_top50 - 1078.994) / 24.4796   # -1.8%  sum_pt_top50 > 1079
        - 0.01788303 * max(0.0, Q.sum_pt_top30 - 996.8867) / 38.86433   # -1.8%  sum_pt_top30 > 996.9
        - 0.01289785 * max(0.0, Q.sum_pt_top50 - 1038.855) / 34.95069   # -1.3%  sum_pt_top50 > 1039
        + 0.01269159 * max(0.0, Q.sum_pt_top20 - 1064.139) / 11.11947   # +1.3%  sum_pt_top20 > 1064
        - 0.0118537 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.dr_max_012 - 0.1206357) / 0.06050554   # -1.2%  sum_pt_top20 > 1129 and dr_max_012 > 0.1206
        + 0.007929427 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # +0.8%  log_sum_pt > 7.063
        - 0.006075916 * max(0.0, Q.sum_pt_top15 - 1082.548) / 5.972549   # -0.6%  sum_pt_top15 > 1083
        - 0.005578373 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 10.49219 - Q.pt_12) / 0.02319147   # -0.6%  log_sum_pt > 6.903 and pt_12 < 10.49
        + 0.005331495 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, 0.001658225 - Q.zdr_14) / 9.234506e-06   # +0.5%  log_sum_pt > 7.063 and zdr_14 < 0.001658
        - 0.004936578 * max(0.0, Q.sum_pt_top20 - 1129.275) / 6.499168   # -0.5%  sum_pt_top20 > 1129
        + 0.004084241 * max(0.0, Q.sum_pt_top15 - 985.0781) / 16.07962   # +0.4%  sum_pt_top15 > 985.1
        - 0.003616612 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.08734131) / 0.003497383   # -0.4%  sum_pt_top20 > 1129 and eta_1 > 0.08734
        - 0.003604016 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 1.0 - Q.psi_0p3) / 0.0005826929   # -0.4%  log_sum_pt > 6.903 and psi_0p3 < 1
        + 0.003278426 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.z_7 - 0.02696541) / 0.04610808   # +0.3%  sum_pt_top20 > 1129 and z_7 > 0.02697
        - 0.003225171 * max(0.0, Q.sum_pt_top15 - 985.0781) * max(0.0, 0.05840156 - Q.M3) / 0.3504214   # -0.3%  sum_pt_top15 > 985.1 and M3 < 0.0584
        + 0.003137814 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.girth2_top15 - 0.009962397) / 1.329441e-05   # +0.3%  log_sum_pt > 7.063 and girth2_top15 > 0.009962
        - 0.002860304 * max(0.0, Q.N2 - 0.4454953) / 0.003082384   # -0.3%  N2 > 0.4455
        - 0.002804073 * max(0.0, Q.sum_pt_top15 - 985.0781) * max(0.0, Q.pt_7 - 41.65625) / 104.7463   # -0.3%  sum_pt_top15 > 985.1 and pt_7 > 41.66
        + 0.002431917 * max(0.0, Q.sum_pt - 1085.125) * max(0.0, 0.01389212 - Q.z_10) / 0.01206953   # +0.2%  sum_pt > 1085 and z_10 < 0.01389
        + 0.001589078 * max(0.0, Q.psi_0p1 - 0.8976117) * max(0.0, 0.985099 - Q.z_top50_slots) / 1.493571e-05   # +0.2%  psi_0p1 > 0.8976 and z_top50_slots < 0.9851
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 2.838;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.838415 * (0.004495613
        + 0.175572 * max(0.0, 0.0174227 - Q.tau4) / 0.003628011   # +17.6%  tau4 < 0.01742
        - 0.1442741 * max(0.0, 0.347196 - Q.tau21) / 0.05239131   # -14.4%  tau21 < 0.3472
        + 0.07026349 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +7.0%  n_dr_0p2_0p4 < 5
        - 0.06276514 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # -6.3%  n_particles < 46
        + 0.04882793 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # +4.9%  n_dr_0p1_0p2 < 17
        - 0.04645279 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.e2 - 0.01036127) / 0.09655312   # -4.6%  n_particles < 46 and e2 > 0.01036
        + 0.0375536 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 858.8262) / 1128.322   # +3.8%  n_particles < 46 and sum_pt_top40 > 858.8
        + 0.03469641 * max(0.0, 0.347196 - Q.tau21) * max(0.0, 1085.125 - Q.sum_pt) / 3.187528   # +3.5%  tau21 < 0.3472 and sum_pt < 1085
        + 0.03431357 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.eccentricity - 0.8680812) / 3.480194e-05   # +3.4%  psi_0p3 > 0.998 and eccentricity > 0.8681
        + 0.02494132 * max(0.0, 0.0006154841 - Q.lam2) * max(0.0, 1.0 - Q.n_dr_0p4_up) / 0.0001383002   # +2.5%  lam2 < 0.0006155 and n_dr_0p4_up < 1
        + 0.02214357 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +2.2%  lam2 < 0.0006155
        - 0.02176255 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0) / 3.186294   # -2.2%  n_dr_0p2_0p4 < 5 and n_dr_0p1_0p2 > 9
        - 0.02138538 * max(0.0, 0.08871546 - Q.z_dr_0p1_0p2) / 0.03070029   # -2.1%  z_dr_0p1_0p2 < 0.08872
        - 0.02112021 * max(0.0, 0.003235754 - Q.zdr_0) / 0.0003842694   # -2.1%  zdr_0 < 0.003236
        - 0.01832844 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # -1.8%  n_dr_0p2_0p4 < 8
        + 0.01775212 * max(0.0, 46.0 - Q.n_particles) * max(0.0, 0.02980347 - Q.eta_0) / 0.211231   # +1.8%  n_particles < 46 and eta_0 < 0.0298
        - 0.01661193 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, 0.005967398 - Q.zdr_5) / 2.029154e-06   # -1.7%  psi_0p3 > 0.998 and zdr_5 < 0.005967
        + 0.01622243 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.004201797 - Q.soft10_z) / 0.0015266   # +1.6%  n_dr_0p2_0p4 < 5 and soft10_z < 0.004202
        - 0.01610561 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) / 2.171439   # -1.6%  n_dr_0p1_0p2 < 10
        + 0.01597636 * max(0.0, Q.psi_0p2 - 0.9985434) / 0.00014398   # +1.6%  psi_0p2 > 0.9985
        - 0.01509643 * max(0.0, 1.788105 - Q.D2) * max(0.0, 0.003582374 - Q.soft5_z) / 0.0005782761   # -1.5%  D2 < 1.788 and soft5_z < 0.003582
        + 0.01411744 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sd_rg - 0.1596365) / 0.01134857   # +1.4%  n_dr_0p2_0p4 < 5 and sd_rg > 0.1596
        - 0.01405349 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03029714 - Q.e2) / 0.006086253   # -1.4%  n_dr_0p2_0p4 < 5 and e2 < 0.0303
        - 0.01275307 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, 0.003052491 - Q.soft7_z) / 7.113004e-07   # -1.3%  psi_0p3 > 0.998 and soft7_z < 0.003052
        + 0.01234112 * max(0.0, 0.0174227 - Q.tau4) * max(0.0, Q.pt_11 - 17.09375) / 0.01303829   # +1.2%  tau4 < 0.01742 and pt_11 > 17.09
        + 0.01161972 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.pt1_dr01 - 12.6865) / 6.748406   # +1.2%  n_dr_0p2_0p4 < 8 and pt1_dr01 > 12.69
        + 0.0100951 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.planar_flow - 0.08857921) / 1.003549   # +1.0%  n_dr_0p1_0p2 < 10 and planar_flow > 0.08858
        - 0.008646463 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 988.4375 - Q.sum_pt_top30) / 17.46027   # -0.9%  n_dr_0p2_0p4 < 5 and sum_pt_top30 < 988.4
        + 0.007826432 * max(0.0, 0.347196 - Q.tau21) * max(0.0, 0.8976117 - Q.psi_0p1) / 0.008852048   # +0.8%  tau21 < 0.3472 and psi_0p1 < 0.8976
        + 0.007694914 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.sj2_dr - 0.1512157) / 1.762725e-05   # +0.8%  psi_0p3 > 0.998 and sj2_dr > 0.1512
        + 0.007661809 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.01452637 - Q.phi_0) / 0.02777525   # +0.8%  n_dr_0p2_0p4 < 5 and phi_0 < 0.01453
        - 0.004199916 * max(0.0, Q.sum_pt_top10 - 916.1578) / 16.81479   # -0.4%  sum_pt_top10 > 916.2
        - 0.002808162 * max(0.0, 0.0006154841 - Q.lam2) * max(0.0, -0.04559937 - Q.eta_9) / 1.248239e-06   # -0.3%  lam2 < 0.0006155 and eta_9 < -0.0456
        - 0.002111011 * max(0.0, 0.347196 - Q.tau21) * max(0.0, 0.03275811 - Q.soft7_dr) / 0.0001041529   # -0.2%  tau21 < 0.3472 and soft7_dr < 0.03276
        - 0.001905975 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sd_rg - 0.2042612) / 0.0007836064   # -0.2%  n_dr_0p2_0p4 < 5 and sd_rg > 0.2043
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 8.95;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.95019 * (0.1994048
        + 0.1614399 * max(0.0, 0.009962397 - Q.girth2_top15) / 0.004894699   # +16.1%  girth2_top15 < 0.009962
        - 0.07182243 * max(0.0, 0.007887677 - Q.girth2_top15) / 0.00326329   # -7.2%  girth2_top15 < 0.007888
        + 0.06980017 * max(0.0, Q.sj2_dr - 0.1219342) / 0.08223311   # +7.0%  sj2_dr > 0.1219
        - 0.06035159 * max(0.0, 1041.263 - Q.sum_pt_top40) / 49.16087   # -6.0%  sum_pt_top40 < 1041
        - 0.0590642 * max(0.0, Q.n_particles - 26.0) / 20.24921   # -5.9%  n_particles > 26
        - 0.05747132 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -5.7%  e2 < 0.04359
        - 0.04569291 * max(0.0, 0.005788041 - Q.girth2_top15) / 0.001877672   # -4.6%  girth2_top15 < 0.005788
        + 0.03877039 * max(0.0, 1011.524 - Q.sum_pt_top30) / 51.02618   # +3.9%  sum_pt_top30 < 1012
        - 0.0382284 * max(0.0, 0.05660088 - Q.girth) / 0.01080382   # -3.8%  girth < 0.0566
        - 0.03471773 * max(0.0, Q.sj2_dr - 0.1825048) / 0.04110551   # -3.5%  sj2_dr > 0.1825
        - 0.03432346 * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.02836816   # -3.4%  z_dr_0p2_0p4 < 0.0518
        + 0.03097104 * max(0.0, 0.09749958 - Q.girth) / 0.03657082   # +3.1%  girth < 0.0975
        + 0.02942217 * max(0.0, 0.09749958 - Q.girth) * max(0.0, Q.psi_0p3 - 0.9973959) / 3.542375e-05   # +2.9%  girth < 0.0975 and psi_0p3 > 0.9974
        - 0.02637754 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # -2.6%  e3 < 0.0005178
        - 0.02068306 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # -2.1%  girth < 0.07374
        + 0.02010657 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +2.0%  n_dr_0p2_0p4 < 18
        - 0.01897324 * max(0.0, 0.006876086 - Q.girth2_top10) / 0.002892387   # -1.9%  girth2_top10 < 0.006876
        + 0.0184649 * max(0.0, Q.n_particles - 26.0) * max(0.0, Q.tau21 - 0.1295048) / 7.108916   # +1.8%  n_particles > 26 and tau21 > 0.1295
        + 0.0171448 * max(0.0, 0.002197765 - Q.girth2_top15) / 0.0004509993   # +1.7%  girth2_top15 < 0.002198
        - 0.01700255 * max(0.0, 5.142486 - Q.D2_b2) / 2.813803   # -1.7%  D2_b2 < 5.142
        - 0.01620369 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -1.6%  psi_0p1 > 0.8976
        - 0.01267033 * max(0.0, 0.05660088 - Q.girth) * max(0.0, Q.psi_0p3 - 0.9973959) / 8.756067e-06   # -1.3%  girth < 0.0566 and psi_0p3 > 0.9974
        - 0.01193109 * max(0.0, Q.n_particles - 26.0) * max(0.0, Q.soft1_pt - 0.4909668) / 9.262955   # -1.2%  n_particles > 26 and soft1_pt > 0.491
        - 0.01188504 * max(0.0, 0.004169954 - Q.girth2_top15) / 0.001130716   # -1.2%  girth2_top15 < 0.00417
        + 0.009265086 * max(0.0, 0.04748396 - Q.z_dr_0p1_0p2) / 0.01200196   # +0.9%  z_dr_0p1_0p2 < 0.04748
        + 0.007929164 * max(0.0, 0.3563114 - Q.planar_flow) / 0.0664923   # +0.8%  planar_flow < 0.3563
        - 0.006395635 * max(0.0, 3.355186 - Q.pt_entropy) * max(0.0, Q.zdr_0 - 0.0008718296) / 0.004720646   # -0.6%  pt_entropy < 3.355 and zdr_0 > 0.0008718
        + 0.005827366 * max(0.0, 0.1416054 - Q.D3) / 0.05511785   # +0.6%  D3 < 0.1416
        - 0.005159698 * max(0.0, Q.sj2_dr - 0.2412757) / 0.01833576   # -0.5%  sj2_dr > 0.2413
        - 0.004745812 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.5%  psi_0p3 > 0.998
        + 0.004454756 * max(0.0, Q.n_particles - 26.0) * max(0.0, Q.sj3_dr_max - 0.3715619) / 0.2880801   # +0.4%  n_particles > 26 and sj3_dr_max > 0.3716
        + 0.004379407 * max(0.0, 3.355186 - Q.pt_entropy) / 0.5477678   # +0.4%  pt_entropy < 3.355
        - 0.003926726 * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 4.239657   # -0.4%  n_dr_0p1_0p2 > 11
        + 0.003467363 * max(0.0, 0.007887677 - Q.girth2_top15) * max(0.0, Q.sum_pt_top40 - 1095.686) / 0.09612434   # +0.3%  girth2_top15 < 0.007888 and sum_pt_top40 > 1096
        + 0.003363541 * max(0.0, Q.n_particles - 26.0) * max(0.0, 0.2982 - Q.max_dr) / 0.1671773   # +0.3%  n_particles > 26 and max_dr < 0.2982
        - 0.003312997 * max(0.0, 0.3563114 - Q.planar_flow) * max(0.0, Q.soft5_dr0 - 0.054583) / 0.007650575   # -0.3%  planar_flow < 0.3563 and soft5_dr0 > 0.05458
        + 0.00283672 * max(0.0, Q.n_dr_0p1_0p2 - 11.0) * max(0.0, Q.e3 - 0.0001841806) / 0.0005133991   # +0.3%  n_dr_0p1_0p2 > 11 and e3 > 0.0001842
        + 0.002503163 * max(0.0, 0.03263075 - Q.e2) / 0.008034085   # +0.3%  e2 < 0.03263
        - 0.002389477 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.eccentricity - 0.773315) / 7.903039e-05   # -0.2%  psi_0p3 > 0.998 and eccentricity > 0.7733
        - 0.002263096 * max(0.0, 0.09749958 - Q.girth) * max(0.0, Q.e3 - 4.646151e-05) / 1.294051e-07   # -0.2%  girth < 0.0975 and e3 > 4.646e-05
        + 0.001612574 * max(0.0, Q.soft1_pt - 2.275391) / 0.04593653   # +0.2%  soft1_pt > 2.275
        - 0.001500701 * max(0.0, 0.004169954 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 8.516876e-07   # -0.2%  girth2_top15 < 0.00417 and psi_0p3 > 0.9974
        + 0.001148223 * max(0.0, Q.n_dr_0p2_0p4 - 26.0) / 0.2669462   # +0.1%  n_dr_0p2_0p4 > 26
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.38;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.38153 * (-0.2068411
        + 0.2338529 * max(0.0, 0.1564779 - Q.girth) / 0.08789501   # +23.4%  girth < 0.1565
        + 0.06672834 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +6.7%  n_particles < 64
        - 0.06169833 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -6.2%  tau1 < 0.1073
        + 0.04865584 * max(0.0, Q.LHA - 0.1632346) / 0.1084458   # +4.9%  LHA > 0.1632
        - 0.04104146 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -4.1%  sum_pt < 1002
        + 0.040016 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # +4.0%  log_sum_pt < 7.139
        - 0.03812508 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # -3.8%  sum_pt_top50 < 1079
        + 0.03561543 * max(0.0, Q.z_dr_0_0p05 - 0.7128619) / 0.07526795   # +3.6%  z_dr_0_0p05 > 0.7129
        - 0.03257987 * max(0.0, Q.z_top20_slots - 0.8281581) / 0.07910948   # -3.3%  z_top20_slots > 0.8282
        + 0.02971485 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # +3.0%  sum_pt < 1085
        + 0.0261124 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # +2.6%  e3 < 0.0005178
        + 0.02538907 * max(0.0, 911.9328 - Q.sum_pt_top30) / 15.55609   # +2.5%  sum_pt_top30 < 911.9
        - 0.02466255 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 7.684973e-07   # -2.5%  sum_pt < 1002 and e4 < 5.851e-08
        - 0.02348064 * max(0.0, Q.z_top30_slots - 0.9460751) / 0.02601016   # -2.3%  z_top30_slots > 0.9461
        + 0.02179392 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 3.38926e-06   # +2.2%  sum_pt < 1085 and e4 < 5.851e-08
        + 0.02166592 * max(0.0, Q.LHA - 0.302389) / 0.02201444   # +2.2%  LHA > 0.3024
        + 0.0194752 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +1.9%  n_dr_0p2_0p4 < 11
        - 0.0144028 * max(0.0, 0.985099 - Q.z_top50_slots) / 0.003556075   # -1.4%  z_top50_slots < 0.9851
        - 0.01359625 * max(0.0, 6.567534e-05 - Q.e3) / 2.643172e-05   # -1.4%  e3 < 6.568e-05
        - 0.01357463 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 0.07996447 - Q.C2) / 0.5514341   # -1.4%  n_particles < 64 and C2 < 0.07996
        - 0.01285947 * max(0.0, 1013.042 - Q.sum_pt_top40) / 31.91616   # -1.3%  sum_pt_top40 < 1013
        - 0.01225163 * max(0.0, 0.05954375 - Q.z_dr_0p05_0p1) / 0.01428263   # -1.2%  z_dr_0p05_0p1 < 0.05954
        + 0.01206821 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +1.2%  tau4 < 0.01627
        - 0.01107689 * Q.z_top3_slots / 0.4446698   # -1.1%  z_top3_slots
        + 0.01103556 * max(0.0, 934.2416 - Q.sum_pt_top50) / 6.932864   # +1.1%  sum_pt_top50 < 934.2
        + 0.01100124 * max(0.0, 0.0005178279 - Q.e3) * max(0.0, Q.max_dr - 0.1939977) / 6.684346e-05   # +1.1%  e3 < 0.0005178 and max_dr > 0.194
        - 0.009657549 * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.02836816   # -1.0%  z_dr_0p2_0p4 < 0.0518
        - 0.00893815 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.ptdr0_10 - 3.719859) / 10.46372   # -0.9%  sum_pt < 1002 and ptdr0_10 > 3.72
        - 0.008689942 * max(0.0, 907.9372 - Q.sum_pt) / 3.961002   # -0.9%  sum_pt < 907.9
        - 0.008255337 * max(0.0, 8.0 - Q.n_dr_0_0p05) / 1.708339   # -0.8%  n_dr_0_0p05 < 8
        + 0.007476231 * max(0.0, 1.409617 - Q.D2) / 0.1519846   # +0.7%  D2 < 1.41
        - 0.006778382 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.absphi_1 - 0.02227783) / 0.4183469   # -0.7%  sum_pt < 1002 and absphi_1 > 0.02228
        + 0.005926361 * max(0.0, 0.1564779 - Q.girth) * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 0.1706859   # +0.6%  girth < 0.1565 and n_dr_0p1_0p2 > 11
        + 0.005752048 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.6%  psi_0p3 > 0.9974
        - 0.00543659 * max(0.0, 1.681792 - Q.ptdr0_10) / 0.6929801   # -0.5%  ptdr0_10 < 1.682
        - 0.004516935 * max(0.0, 0.0007894752 - Q.girth2_top15) / 9.062109e-05   # -0.5%  girth2_top15 < 0.0007895
        + 0.003889848 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # +0.4%  z_dr_0_0p05 > 0.8789
        - 0.003141872 * max(0.0, Q.C2 - 0.1235569) / 0.002419231   # -0.3%  C2 > 0.1236
        + 0.003099059 * max(0.0, 0.02476077 - Q.z_8) / 0.002032134   # +0.3%  z_8 < 0.02476
        + 0.003064932 * max(0.0, Q.z_dr_0_0p05 - 0.9351131) / 0.004598373   # +0.3%  z_dr_0_0p05 > 0.9351
        + 0.002959933 * max(0.0, Q.girth2_top10 - 0.005337976) / 0.003351803   # +0.3%  girth2_top10 > 0.005338
        - 0.002228667 * max(0.0, Q.sum_pt_top15 - 967.7705) * max(0.0, 0.01500702 - Q.absphi_1) / 0.1325052   # -0.2%  sum_pt_top15 > 967.8 and absphi_1 < 0.01501
        - 0.002201318 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, 0.199671 - Q.dr_11) / 1.707103   # -0.2%  sum_pt < 1002 and dr_11 < 0.1997
        - 0.002009556 * max(0.0, 0.1564779 - Q.girth) * max(0.0, 0.9638082 - Q.psi_0p3) / 0.0001348634   # -0.2%  girth < 0.1565 and psi_0p3 < 0.9638
        - 0.001353751 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.dr1_13 - 0.1778153) / 0.5300108   # -0.1%  sum_pt < 1002 and dr1_13 > 0.1778
        - 0.00118013 * max(0.0, 7.139296 - Q.log_sum_pt) * max(0.0, Q.soft2_pt - 1.413232) / 0.02596349   # -0.1%  log_sum_pt < 7.139 and soft2_pt > 1.413
        + 0.0005551578 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, 0.03626613 - Q.soft6_dr) / 0.02518702   # +0.1%  sum_pt < 1002 and soft6_dr < 0.03627
        + 0.0004137809 * max(0.0, 934.2416 - Q.sum_pt_top50) * max(0.0, 0.03626613 - Q.soft6_dr) / 0.009542907   # +0.0%  sum_pt_top50 < 934.2 and soft6_dr < 0.03627
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 13.38;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.38226 * (0.01474897
        - 0.2726101 * max(0.0, 0.09749958 - Q.girth) / 0.03657082   # -27.3%  girth < 0.0975
        + 0.08315974 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +8.3%  LHA < 0.3332
        + 0.06949995 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +6.9%  e3 < 0.0003372
        + 0.06897283 * max(0.0, 0.006427167 - Q.lam2) / 0.005146703   # +6.9%  lam2 < 0.006427
        - 0.05709254 * max(0.0, Q.e3 - 0.0001086251) / 5.31284e-05   # -5.7%  e3 > 0.0001086
        + 0.05232252 * max(0.0, Q.e3 - 5.13841e-05) / 6.821576e-05   # +5.2%  e3 > 5.138e-05
        + 0.04999321 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # +5.0%  tau1 < 0.05445
        + 0.04651737 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # +4.7%  tau2 < 0.0795
        - 0.03831713 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, 7.017258 - Q.log_sum_pt) / 2.140029e-05   # -3.8%  e3 < 0.0003372 and log_sum_pt < 7.017
        - 0.03780026 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -3.8%  psi_0p2 > 0.9087
        + 0.02243568 * max(0.0, 0.3676068 - Q.sj2_zsoft) / 0.1397909   # +2.2%  sj2_zsoft < 0.3676
        + 0.02003496 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # +2.0%  e2 < 0.04359
        - 0.01809994 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -1.8%  psi_0p3 > 0.9943
        + 0.01371376 * max(0.0, Q.psi_0p2 - 0.9087063) * max(0.0, 0.1512157 - Q.sj2_dr) / 0.001524835   # +1.4%  psi_0p2 > 0.9087 and sj2_dr < 0.1512
        + 0.01227002 * max(0.0, Q.z_dr_0_0p05 - 0.3289237) / 0.280304   # +1.2%  z_dr_0_0p05 > 0.3289
        - 0.01202115 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # -1.2%  z_dr_0p2_0p4 < 0.06849
        + 0.0113793 * max(0.0, 0.003270031 - Q.girth2_top15) / 0.0007969965   # +1.1%  girth2_top15 < 0.00327
        - 0.01059425 * max(0.0, 0.09749958 - Q.girth) * max(0.0, Q.eccentricity - 0.6414784) / 0.004627153   # -1.1%  girth < 0.0975 and eccentricity > 0.6415
        - 0.01046132 * max(0.0, Q.z_top20_slots - 0.7818983) / 0.116358   # -1.0%  z_top20_slots > 0.7819
        + 0.009287759 * max(0.0, Q.n_dr_0p1_0p2 - 15.0) * max(0.0, Q.psi_0p3 - 0.9853273) / 0.01929542   # +0.9%  n_dr_0p1_0p2 > 15 and psi_0p3 > 0.9853
        + 0.008566158 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +0.9%  sum_pt > 907.9
        + 0.006902746 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # +0.7%  e2 > 0.05557
        - 0.00624342 * max(0.0, 0.08589404 - Q.girth) * max(0.0, 0.1134943 - Q.M2) / 0.001132661   # -0.6%  girth < 0.08589 and M2 < 0.1135
        + 0.005745055 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +0.6%  e2 < 0.02516
        - 0.00554627 * max(0.0, 0.003270031 - Q.girth2_top15) * max(0.0, Q.sum_pt_top40 - 858.8262) / 0.1598282   # -0.6%  girth2_top15 < 0.00327 and sum_pt_top40 > 858.8
        - 0.005332169 * max(0.0, 0.001319197 - Q.girth2_top15) / 0.0002099631   # -0.5%  girth2_top15 < 0.001319
        - 0.004621999 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.03720487   # -0.5%  z_dr_0p1_0p2 > 0.334
        + 0.004535229 * max(0.0, 0.08589404 - Q.girth) * max(0.0, 1002.379 - Q.sum_pt) / 0.3684202   # +0.5%  girth < 0.08589 and sum_pt < 1002
        - 0.004330539 * max(0.0, Q.n_dr_0p1_0p2 - 15.0) * max(0.0, 0.006427167 - Q.lam2) / 0.008539361   # -0.4%  n_dr_0p1_0p2 > 15 and lam2 < 0.006427
        - 0.004105711 * max(0.0, Q.C2 - 0.1088881) / 0.004226185   # -0.4%  C2 > 0.1089
        + 0.003661253 * max(0.0, 0.04575675 - Q.sj2_zsoft) / 0.00226832   # +0.4%  sj2_zsoft < 0.04576
        + 0.003376056 * max(0.0, Q.sum_pt_top50 - 1078.994) / 24.4796   # +0.3%  sum_pt_top50 > 1079
        - 0.002976809 * max(0.0, 0.3676068 - Q.sj2_zsoft) * max(0.0, 0.008824206 - Q.zdr_1) / 0.0007676544   # -0.3%  sj2_zsoft < 0.3676 and zdr_1 < 0.008824
        - 0.00240106 * max(0.0, 0.003270031 - Q.girth2_top15) * max(0.0, Q.z_top15_slots - 0.7733683) / 9.822416e-05   # -0.2%  girth2_top15 < 0.00327 and z_top15_slots > 0.7734
        - 0.002208114 * max(0.0, Q.n_dr_0p1_0p2 - 26.0) / 0.7746672   # -0.2%  n_dr_0p1_0p2 > 26
        + 0.001817634 * max(0.0, 0.09749958 - Q.girth) * max(0.0, 0.9638082 - Q.psi_0p3) / 1.616441e-05   # +0.2%  girth < 0.0975 and psi_0p3 < 0.9638
        + 0.001625338 * max(0.0, 0.05444509 - Q.tau1) * max(0.0, 972.0419 - Q.sum_pt) / 0.0682246   # +0.2%  tau1 < 0.05445 and sum_pt < 972
        + 0.001357732 * max(0.0, Q.tau21_b2 - 0.7058597) / 0.01262433   # +0.1%  tau21_b2 > 0.7059
        + 0.001305393 * max(0.0, Q.psi_0p3 - 0.9943058) * max(0.0, Q.soft1_z - 0.0004000768) / 6.136976e-07   # +0.1%  psi_0p3 > 0.9943 and soft1_z > 0.0004001
        + 0.001283286 * max(0.0, Q.n_dr_0p1_0p2 - 15.0) / 2.705171   # +0.1%  n_dr_0p1_0p2 > 15
        - 0.001187781 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) / 1.539217   # -0.1%  n_dr_0p2_0p4 > 15
        - 0.001056651 * max(0.0, 0.04358622 - Q.e2) * max(0.0, 0.9638082 - Q.psi_0p3) / 1.224228e-05   # -0.1%  e2 < 0.04359 and psi_0p3 < 0.9638
        + 0.0009348649 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, 0.07952881 - Q.eta_0) / 0.1317641   # +0.1%  n_dr_0p2_0p4 > 15 and eta_0 < 0.07953
        + 0.0008436103 * max(0.0, 0.2126484 - Q.z_dr_0p05_0p1) * max(0.0, Q.zdr_0 - 0.01113024) / 0.0001762331   # +0.1%  z_dr_0p05_0p1 < 0.2126 and zdr_0 > 0.01113
        + 0.0006960611 * max(0.0, Q.e3 - 5.13841e-05) * max(0.0, 0.06796146 - Q.sj3_dr_min) / 4.178763e-08   # +0.1%  e3 > 5.138e-05 and sj3_dr_min < 0.06796
        + 0.0006383965 * max(0.0, Q.psi_0p2 - 0.9087063) * max(0.0, 0.9704436 - Q.z_top50_slots) / 3.390458e-05   # +0.1%  psi_0p2 > 0.9087 and z_top50_slots < 0.9704
        + 0.0001168655 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, 0.06988208 - Q.sj3_dr13) / 0.0005224292   # +0.0%  n_dr_0p2_0p4 > 15 and sj3_dr13 < 0.06988
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 29.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.06728 * (0.001245172
        - 0.1201011 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -12.0%  tau1 < 0.1073
        - 0.09194356 * max(0.0, 0.08068193 - Q.girth) / 0.02368455   # -9.2%  girth < 0.08068
        + 0.08197579 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # +8.2%  LHA < 0.3024
        + 0.07311132 * max(0.0, 0.1072713 - Q.tau1) * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) / 0.008148309   # +7.3%  tau1 < 0.1073 and z_dr_0p1_0p2 < 0.2865
        + 0.0726311 * max(0.0, 0.09749958 - Q.girth) / 0.03657082   # +7.3%  girth < 0.0975
        - 0.0606046 * max(0.0, 0.3332345 - Q.LHA) * max(0.0, 0.250441 - Q.z_dr_0p1_0p2) / 0.01705837   # -6.1%  LHA < 0.3332 and z_dr_0p1_0p2 < 0.2504
        + 0.04513394 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +4.5%  psi_0p3 > 0.9974
        + 0.04226216 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.009962397 - Q.girth2_top15) / 0.0001707165   # +4.2%  tau21_b2 < 0.2352 and girth2_top15 < 0.009962
        + 0.0384654 * max(0.0, 0.1219132 - Q.tau1) / 0.04578087   # +3.8%  tau1 < 0.1219
        - 0.03511278 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3) / 0.0005360215   # -3.5%  n_dr_0p2_0p4 < 21 and e3 < 7.876e-05
        - 0.03214414 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # -3.2%  LHA < 0.3332
        - 0.02968477 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007887677 - Q.girth2_top15) / 8.087416e-05   # -3.0%  tau21_b2 < 0.2352 and girth2_top15 < 0.007888
        + 0.02946852 * max(0.0, 0.05557149 - Q.e2) / 0.02580562   # +2.9%  e2 < 0.05557
        - 0.02189342 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.006802603 - Q.mean_eta2) / 3.803697e-06   # -2.2%  psi_0p3 > 0.9974 and mean_eta2 < 0.006803
        + 0.02176373 * max(0.0, 0.03680582 - Q.e2) / 0.01052402   # +2.2%  e2 < 0.03681
        - 0.02104361 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.006808102 - Q.mean_phi2) / 3.800578e-06   # -2.1%  psi_0p3 > 0.9974 and mean_phi2 < 0.006808
        + 0.02043448 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063   # +2.0%  n_dr_0p2_0p4 < 21
        - 0.01778584 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1245.697 - Q.sum_pt_top50) / 11.02846   # -1.8%  tau21_b2 < 0.2352 and sum_pt_top50 < 1246
        - 0.01627588 * max(0.0, 0.04755309 - Q.e2) / 0.01877457   # -1.6%  e2 < 0.04755
        + 0.01454955 * max(0.0, 6.567534e-05 - Q.e3) / 2.643172e-05   # +1.5%  e3 < 6.568e-05
        - 0.01448061 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 7.062574 - Q.log_sum_pt) / 0.0001122434   # -1.4%  psi_0p3 > 0.9974 and log_sum_pt < 7.063
        - 0.01376229 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -1.4%  n_dr_0p2_0p4 < 15
        + 0.01237744 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.003687605 - Q.lam2) / 2.993211e-06   # +1.2%  psi_0p3 > 0.9974 and lam2 < 0.003688
        + 0.01197845 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.1134943 - Q.M2) / 0.3618294   # +1.2%  n_dr_0p2_0p4 < 15 and M2 < 0.1135
        - 0.007970224 * max(0.0, 3.376709e-05 - Q.e3) / 8.295151e-06   # -0.8%  e3 < 3.377e-05
        - 0.006746059 * max(0.0, 0.347196 - Q.tau21) / 0.05239131   # -0.7%  tau21 < 0.3472
        - 0.006330327 * max(0.0, Q.psi_0p1 - 0.8509811) / 0.04568859   # -0.6%  psi_0p1 > 0.851
        - 0.005590543 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.girth2_top10 - 0.007678544) / 7.055851e-07   # -0.6%  psi_0p3 > 0.9974 and girth2_top10 > 0.007679
        + 0.005129054 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +0.5%  tau21_b2 < 0.2352
        + 0.005087638 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.06975954 - Q.M2) / 0.001355005   # +0.5%  tau21_b2 < 0.2352 and M2 < 0.06976
        - 0.00472306 * max(0.0, 5.727594e-05 - Q.e3) / 2.076531e-05   # -0.5%  e3 < 5.728e-05
        + 0.004228285 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.001696484   # +0.4%  tau21_b2 < 0.2352 and sj2_dr > 0.1938
        + 0.003659497 * max(0.0, 0.005180665 - Q.girth2_top2) / 0.002696378   # +0.4%  girth2_top2 < 0.005181
        + 0.002720679 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.4357228 - Q.max_dr) / 0.8636325   # +0.3%  n_dr_0p2_0p4 < 15 and max_dr < 0.4357
        + 0.002091716 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.girth2_top10 - 0.01414829) / 3.726039e-07   # +0.2%  psi_0p3 > 0.9974 and girth2_top10 > 0.01415
        + 0.001861534 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 3.376709e-05 - Q.e3) / 0.0001233727   # +0.2%  n_dr_0p2_0p4 < 21 and e3 < 3.377e-05
        + 0.001355663 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.005788041 - Q.girth2_top15) / 1.968469e-05   # +0.1%  tau21_b2 < 0.2352 and girth2_top15 < 0.005788
        - 0.0009538206 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 36.0 - Q.n_real_top40) / 0.1793513   # -0.1%  tau21_b2 < 0.2352 and n_real_top40 < 36
        + 0.0008576602 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.2618467 - Q.D3) / 0.003307359   # +0.1%  tau21_b2 < 0.2352 and D3 < 0.2618
        + 0.0006916446 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.001127591 - Q.zdr_14) / 3.211654e-07   # +0.1%  psi_0p3 > 0.9974 and zdr_14 < 0.001128
        + 0.0004317852 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 10.21558   # +0.0%  n_dr_0p2_0p4 < 21 and n_dr_0p1_0p2 > 21
        + 0.0003055098 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.985099 - Q.z_top50_slots) / 2.023206e-06   # +0.0%  psi_0p3 > 0.9974 and z_top50_slots < 0.9851
        - 0.0002808081 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) / 0.293679   # -0.0%  n_dr_0p1_0p2 > 33
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 24.93;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.92965 * (0.1057395
        - 0.1638063 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -16.4%  log_sum_pt < 7.139
        + 0.1106567 * max(0.0, Q.girth - 0.04362872) / 0.03194126   # +11.1%  girth > 0.04363
        + 0.09811513 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # +9.8%  sum_pt < 1261
        + 0.0762168 * max(0.0, Q.girth - 0.03577037) / 0.03765667   # +7.6%  girth > 0.03577
        - 0.067512 * max(0.0, Q.LHA - 0.2091025) / 0.07391154   # -6.8%  LHA > 0.2091
        - 0.04650597 * max(0.0, 0.009962397 - Q.girth2_top15) / 0.004894699   # -4.7%  girth2_top15 < 0.009962
        - 0.04475741 * max(0.0, Q.girth - 0.09749958) / 0.008315822   # -4.5%  girth > 0.0975
        - 0.03616023 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.139296 - Q.log_sum_pt) / 1.930358   # -3.6%  n_dr_0p2_0p4 < 18 and log_sum_pt < 7.139
        - 0.02935022 * max(0.0, Q.girth - 0.07374472) / 0.01465579   # -2.9%  girth > 0.07374
        - 0.02680657 * max(0.0, Q.psi_0p3 - 0.9777125) / 0.01571542   # -2.7%  psi_0p3 > 0.9777
        - 0.02187299 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 3.793233e-05 - Q.e3) / 0.0001192169   # -2.2%  n_dr_0p2_0p4 < 18 and e3 < 3.793e-05
        - 0.02139756 * max(0.0, Q.girth - 0.07031778) / 0.01612201   # -2.1%  girth > 0.07032
        - 0.02050466 * max(0.0, Q.e2 - 0.02210818) / 0.01219843   # -2.1%  e2 > 0.02211
        + 0.01545596 * max(0.0, Q.girth - 0.07031778) * max(0.0, 1191.938 - Q.sum_pt_top30) / 4.563404   # +1.5%  girth > 0.07032 and sum_pt_top30 < 1192
        + 0.01454596 * max(0.0, 1042.609 - Q.sum_pt) / 35.65948   # +1.5%  sum_pt < 1043
        + 0.01437775 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +1.4%  LHA > 0.3332
        + 0.0124705 * max(0.0, Q.z_top50_slots - 0.9586536) / 0.03427112   # +1.2%  z_top50_slots > 0.9587
        + 0.01178143 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # +1.2%  girth2_top5 < 0.007164
        - 0.01172626 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # -1.2%  n_dr_0p2_0p4 < 18
        - 0.01118142 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # -1.1%  n_dr_0p2_0p4 < 8
        + 0.01042358 * max(0.0, Q.girth - 0.07031778) * max(0.0, 0.0795038 - Q.tau2) / 0.0003075621   # +1.0%  girth > 0.07032 and tau2 < 0.0795
        + 0.01034628 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # +1.0%  sum_pt_top40 < 1007
        + 0.01030361 * max(0.0, Q.girth - 0.07031778) * max(0.0, Q.e2 - 0.04358622) / 0.000222409   # +1.0%  girth > 0.07032 and e2 > 0.04359
        + 0.009437714 * max(0.0, Q.girth - 0.07031778) * max(0.0, 1115.723 - Q.sum_pt) / 1.878598   # +0.9%  girth > 0.07032 and sum_pt < 1116
        - 0.009204093 * max(0.0, 0.02675364 - Q.girth2_top15) / 0.01959938   # -0.9%  girth2_top15 < 0.02675
        + 0.008109642 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +0.8%  LHA > 0.372
        + 0.00798119 * max(0.0, 0.08082334 - Q.dr_0) / 0.0377444   # +0.8%  dr_0 < 0.08082
        - 0.007326615 * max(0.0, Q.LHA - 0.3098384) / 0.0195892   # -0.7%  LHA > 0.3098
        + 0.005827987 * max(0.0, 3.793233e-05 - Q.e3) / 1.004564e-05   # +0.6%  e3 < 3.793e-05
        + 0.005284051 * max(0.0, 0.02675364 - Q.girth2_top15) * max(0.0, Q.tau4 - 0.01517184) / 5.673266e-05   # +0.5%  girth2_top15 < 0.02675 and tau4 > 0.01517
        - 0.00526292 * max(0.0, Q.n_dr_0p2_0p4 - 3.0) / 6.482203   # -0.5%  n_dr_0p2_0p4 > 3
        - 0.005030421 * max(0.0, Q.girth - 0.07031778) * max(0.0, 976.277 - Q.sum_pt_top50) / 0.4391148   # -0.5%  girth > 0.07032 and sum_pt_top50 < 976.3
        + 0.005018531 * max(0.0, 1042.609 - Q.sum_pt) * max(0.0, Q.max_dr - 0.1939977) / 6.226413   # +0.5%  sum_pt < 1043 and max_dr > 0.194
        - 0.004583266 * max(0.0, 2.178951 - Q.D2) / 0.4648731   # -0.5%  D2 < 2.179
        + 0.004256758 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # +0.4%  sum_pt < 1002
        + 0.004055584 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.8252004 - Q.psi_0p1) / 0.5392896   # +0.4%  n_dr_0p2_0p4 < 18 and psi_0p1 < 0.8252
        - 0.00398772 * max(0.0, 0.00727763 - Q.girth2_top15) / 0.00282147   # -0.4%  girth2_top15 < 0.007278
        + 0.003808115 * max(0.0, Q.n_particles - 43.0) / 7.774361   # +0.4%  n_particles > 43
        + 0.003676881 * max(0.0, Q.n_pt_above_1 - 54.0) / 1.837408   # +0.4%  n_pt_above_1 > 54
        - 0.003267279 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 0.7225388 - Q.planar_flow) / 60.52173   # -0.3%  sum_pt < 1261 and planar_flow < 0.7225
        - 0.003071068 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, Q.n_dr_0p05_0p1 - 7.0) / 1269.618   # -0.3%  sum_pt < 1261 and n_dr_0p05_0p1 > 7
        - 0.002854724 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.z_0 - 0.08872428) / 1.838848   # -0.3%  sum_pt < 1002 and z_0 > 0.08872
        + 0.002485183 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.2%  sj2_dr > 0.2232
        - 0.002129725 * max(0.0, Q.girth - 0.07031778) * max(0.0, Q.sum_pt_top20 - 869.693) / 0.4413306   # -0.2%  girth > 0.07032 and sum_pt_top20 > 869.7
        + 0.001676561 * max(0.0, 0.007164202 - Q.girth2_top5) * max(0.0, Q.sj3_dr23 - 0.2563461) / 8.744166e-05   # +0.2%  girth2_top5 < 0.007164 and sj3_dr23 > 0.2563
        - 0.001604991 * max(0.0, 1007.44 - Q.sum_pt_top40) * max(0.0, Q.sum_pt - 949.9169) / 555.6226   # -0.2%  sum_pt_top40 < 1007 and sum_pt > 949.9
        + 0.00109806 * max(0.0, Q.e2 - 0.04358622) / 0.002786698   # +0.1%  e2 > 0.04359
        - 0.00108434 * max(0.0, Q.LHA - 0.3203321) * max(0.0, 0.02037449 - Q.z_14) / 4.375313e-05   # -0.1%  LHA > 0.3203 and z_14 < 0.02037
        + 0.0008486647 * max(0.0, Q.psi_0p3 - 0.9777125) * max(0.0, Q.n_dr_0p1_0p2 - 17.0) / 0.02691594   # +0.1%  psi_0p3 > 0.9777 and n_dr_0p1_0p2 > 17
        - 0.0007525899 * max(0.0, Q.n_dr_0_0p05 - 30.0) / 0.212479   # -0.1%  n_dr_0_0p05 > 30
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 8.631;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.630916 * (-0.05371262
        - 0.09010739 * max(0.0, Q.sum_pt_top20 - 750.7313) / 184.752   # -9.0%  sum_pt_top20 > 750.7
        + 0.08399914 * max(0.0, 0.006142802 - Q.girth2_top15) * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.000105955   # +8.4%  girth2_top15 < 0.006143 and z_dr_0p2_0p4 < 0.06849
        + 0.07519628 * max(0.0, 0.09591084 - Q.tau1) / 0.02629125   # +7.5%  tau1 < 0.09591
        + 0.06887297 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # +6.9%  tau1 < 0.1073
        + 0.06085181 * max(0.0, 0.05660088 - Q.girth) / 0.01080382   # +6.1%  girth < 0.0566
        - 0.05795605 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # -5.8%  LHA < 0.2454
        - 0.05666395 * max(0.0, 0.006142802 - Q.girth2_top15) / 0.002080651   # -5.7%  girth2_top15 < 0.006143
        + 0.04615118 * max(0.0, Q.e2 - 0.02515919) / 0.01027642   # +4.6%  e2 > 0.02516
        + 0.03612745 * max(0.0, 2.178951 - Q.D2) / 0.4648731   # +3.6%  D2 < 2.179
        - 0.0323519 * max(0.0, Q.girth - 0.1207452) / 0.004285542   # -3.2%  girth > 0.1207
        + 0.02473398 * max(0.0, 988.4554 - Q.sum_pt_top50) / 15.57124   # +2.5%  sum_pt_top50 < 988.5
        - 0.0245606 * max(0.0, Q.girth - 0.1402186) / 0.001871547   # -2.5%  girth > 0.1402
        + 0.02320946 * max(0.0, 0.006142802 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9777125) / 3.516962e-05   # +2.3%  girth2_top15 < 0.006143 and psi_0p3 > 0.9777
        - 0.02303627 * max(0.0, 0.002412891 - Q.girth2_top2) / 0.0009487944   # -2.3%  girth2_top2 < 0.002413
        + 0.02292984 * max(0.0, Q.girth - 0.09749958) / 0.008315822   # +2.3%  girth > 0.0975
        - 0.0227603 * max(0.0, Q.z_top40_slots - 0.9674996) / 0.02046045   # -2.3%  z_top40_slots > 0.9675
        + 0.02153063 * max(0.0, 0.05660088 - Q.girth) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.0003340726   # +2.2%  girth < 0.0566 and psi_0p3 > 0.9638
        + 0.01986903 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) / 0.03630126   # +2.0%  z_dr_0_0p05 > 0.8103
        - 0.01964229 * max(0.0, 3.376709e-05 - Q.e3) / 8.295151e-06   # -2.0%  e3 < 3.377e-05
        + 0.01435403 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.09061548 - Q.dr_13) / 0.0001299587   # +1.4%  log_sum_pt < 6.811 and dr_13 < 0.09062
        + 0.01432158 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +1.4%  LHA > 0.372
        + 0.01415589 * max(0.0, 0.004855289 - Q.girth2_top15) / 0.001418414   # +1.4%  girth2_top15 < 0.004855
        - 0.01298563 * max(0.0, Q.girth2_top15 - 0.02146578) / 0.0006182335   # -1.3%  girth2_top15 > 0.02147
        - 0.01250417 * max(0.0, 0.0007894752 - Q.girth2_top15) / 9.062109e-05   # -1.3%  girth2_top15 < 0.0007895
        + 0.01211789 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +1.2%  LHA > 0.4042
        + 0.01057058 * max(0.0, 0.006142802 - Q.girth2_top15) * max(0.0, Q.n_dr_0p2_0p4 - 7.0) / 0.004066923   # +1.1%  girth2_top15 < 0.006143 and n_dr_0p2_0p4 > 7
        - 0.01036144 * max(0.0, Q.z_top40_slots - 0.9674996) * max(0.0, 0.01158441 - Q.C3) / 0.0001133203   # -1.0%  z_top40_slots > 0.9675 and C3 < 0.01158
        - 0.01000034 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -1.0%  log_sum_pt < 6.811
        - 0.009800212 * max(0.0, Q.sj3_dr13 - 0.1025461) / 0.0963422   # -1.0%  sj3_dr13 > 0.1025
        + 0.008972621 * max(0.0, 889.8503 - Q.sum_pt_top50) / 3.808547   # +0.9%  sum_pt_top50 < 889.9
        - 0.007756441 * max(0.0, 0.2595052 - Q.sj2_dr) / 0.08013882   # -0.8%  sj2_dr < 0.2595
        - 0.007150981 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, 0.1312677 - Q.dr_13) / 0.7213696   # -0.7%  sum_pt_top50 < 988.5 and dr_13 < 0.1313
        + 0.006373589 * max(0.0, Q.psi_0p1 - 0.8509811) / 0.04568859   # +0.6%  psi_0p1 > 0.851
        - 0.005960302 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # -0.6%  psi_0p3 > 0.9966
        - 0.003869663 * max(0.0, 0.05660088 - Q.girth) * max(0.0, 1007.788 - Q.sum_pt) / 0.1758936   # -0.4%  girth < 0.0566 and sum_pt < 1008
        - 0.00364072 * max(0.0, 0.1825048 - Q.sj2_dr) / 0.03087146   # -0.4%  sj2_dr < 0.1825
        + 0.003259577 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, Q.soft5_z - 0.0004140594) / 2.555122e-05   # +0.3%  z_dr_0p1_0p2 > 0.4479 and soft5_z > 0.0004141
        + 0.002906005 * max(0.0, Q.LHA - 0.3719813) * max(0.0, 0.006427167 - Q.lam2) / 1.406065e-05   # +0.3%  LHA > 0.372 and lam2 < 0.006427
        + 0.002695934 * max(0.0, Q.z_top5_slots - 0.7963976) / 0.006402521   # +0.3%  z_top5_slots > 0.7964
        + 0.002474808 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, 0.2640475 - Q.z_1st) / 1.531764   # +0.2%  sum_pt_top50 < 988.5 and z_1st < 0.264
        - 0.002142673 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, Q.soft5_pt - 1.652344) / 5.88648   # -0.2%  sum_pt_top50 < 988.5 and soft5_pt > 1.652
        - 0.002078932 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, Q.e3 - 0.0001086251) / 0.001799898   # -0.2%  sum_pt_top50 < 988.5 and e3 > 0.0001086
        + 0.002022089 * max(0.0, Q.girth - 0.1564779) / 0.0006617163   # +0.2%  girth > 0.1565
        + 0.001884302 * max(0.0, 0.2595052 - Q.sj2_dr) * max(0.0, Q.pt_0 - 213.5) / 5.810924   # +0.2%  sj2_dr < 0.2595 and pt_0 > 213.5
        - 0.001461803 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, Q.sum_pt_top20 - 789.7344) / 304.8764   # -0.1%  sum_pt_top50 < 988.5 and sum_pt_top20 > 789.7
        + 0.001370499 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) / 0.02385119   # +0.1%  z_dr_0p1_0p2 > 0.4479
        - 0.00130246 * max(0.0, Q.tau21_b2 - 0.6133424) / 0.02390338   # -0.1%  tau21_b2 > 0.6133
        + 0.0009543193 * max(0.0, Q.girth - 0.1402186) * max(0.0, Q.soft6_pt - 4.250195) / 9.168767e-05   # +0.1%  girth > 0.1402 and soft6_pt > 4.25
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 8.818;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.81797 * (0.6783347
        - 0.2365311 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -23.7%  girth < 0.1207
        - 0.1466284 * max(0.0, Q.LHA - 0.1870291) / 0.08997036   # -14.7%  LHA > 0.187
        - 0.06200461 * max(0.0, Q.sum_pt_top5 - 402.625) / 200.0399   # -6.2%  sum_pt_top5 > 402.6
        - 0.05601462 * max(0.0, 0.02441963 - Q.girth2_top5) / 0.01889954   # -5.6%  girth2_top5 < 0.02442
        - 0.03943719 * max(0.0, Q.girth - 0.09749958) / 0.008315822   # -3.9%  girth > 0.0975
        + 0.03777638 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +3.8%  LHA > 0.3332
        + 0.03615477 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # +3.6%  e2 > 0.0303
        + 0.03216622 * max(0.0, 0.1207452 - Q.girth) * max(0.0, Q.psi_0p2 - 0.948102) / 0.001922283   # +3.2%  girth < 0.1207 and psi_0p2 > 0.9481
        - 0.02941341 * max(0.0, 0.01414829 - Q.girth2_top10) / 0.00877908   # -2.9%  girth2_top10 < 0.01415
        + 0.02283458 * max(0.0, Q.psi_0p1 - 0.9184255) / 0.01693566   # +2.3%  psi_0p1 > 0.9184
        - 0.02190888 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # -2.2%  z_dr_0p1_0p2 < 0.1203
        + 0.02177391 * max(0.0, 2.410481 - Q.D2) / 0.587301   # +2.2%  D2 < 2.41
        + 0.02161629 * max(0.0, 0.05602756 - Q.C2) / 0.009882647   # +2.2%  C2 < 0.05603
        - 0.01758327 * max(0.0, Q.psi_0p3 - 0.9989733) / 0.0002740091   # -1.8%  psi_0p3 > 0.999
        - 0.01723112 * max(0.0, Q.sj2_dr - 0.09395198) / 0.1054596   # -1.7%  sj2_dr > 0.09395
        - 0.01717377 * max(0.0, 0.2018786 - Q.tau21_b2) / 0.03799024   # -1.7%  tau21_b2 < 0.2019
        - 0.01471592 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -1.5%  n_dr_0p2_0p4 < 11
        + 0.0145886 * max(0.0, Q.LHA - 0.1870291) * max(0.0, Q.sj3_pairmin_over_m - 0.1389615) / 0.01420104   # +1.5%  LHA > 0.187 and sj3_pairmin_over_m > 0.139
        - 0.01141987 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, 0.8644992 - Q.tau32) / 2.698478e-05   # -1.1%  e3 < 0.0003372 and tau32 < 0.8645
        - 0.01104328 * max(0.0, Q.sj2_dr - 0.09395198) * max(0.0, 0.6025827 - Q.planar_flow) / 0.02318142   # -1.1%  sj2_dr > 0.09395 and planar_flow < 0.6026
        + 0.01072998 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, 114.75 - Q.pt_3) / 0.04784342   # +1.1%  tau1 > 0.1954 and pt_3 < 114.8
        - 0.01040074 * max(0.0, Q.z_dr_0_0p05 - 0.7128619) / 0.07526795   # -1.0%  z_dr_0_0p05 > 0.7129
        - 0.009737479 * max(0.0, Q.sj2_dr - 0.09395198) * max(0.0, Q.M2 - 0.04260132) / 0.001979695   # -1.0%  sj2_dr > 0.09395 and M2 > 0.0426
        - 0.008281607 * max(0.0, 0.002672224 - Q.soft7_z) / 0.001098778   # -0.8%  soft7_z < 0.002672
        + 0.007862841 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +0.8%  sum_pt < 986.1
        - 0.007658424 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_real_top40 - 22.0) / 55.27538   # -0.8%  n_dr_0p2_0p4 < 11 and n_real_top40 > 22
        - 0.007617751 * max(0.0, 6.856375 - Q.log_sum_pt) / 0.008053459   # -0.8%  log_sum_pt < 6.856
        - 0.006812359 * max(0.0, Q.tau1 - 0.1953848) / 0.000822026   # -0.7%  tau1 > 0.1954
        - 0.006557738 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, 0.1082864 - Q.z_3) / 4.224064e-05   # -0.7%  tau1 > 0.1954 and z_3 < 0.1083
        - 0.005923536 * max(0.0, Q.LHA - 0.1870291) * max(0.0, 90.625 - Q.pt_4) / 3.201494   # -0.6%  LHA > 0.187 and pt_4 < 90.62
        - 0.004806032 * max(0.0, Q.tau1 - 0.1751567) / 0.002322569   # -0.5%  tau1 > 0.1752
        + 0.004506159 * max(0.0, 0.43805 - Q.tau32) / 0.01189481   # +0.5%  tau32 < 0.4381
        + 0.00429161 * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 1.386277   # +0.4%  n_dr_0p1_0p2 > 21
        + 0.003985009 * max(0.0, Q.psi_0p3 - 0.9989733) * max(0.0, 0.5912328 - Q.z_dr_0p05_0p1) / 7.440498e-05   # +0.4%  psi_0p3 > 0.999 and z_dr_0p05_0p1 < 0.5912
        - 0.003880706 * max(0.0, Q.psi_0p1 - 0.9848104) / 0.0006331202   # -0.4%  psi_0p1 > 0.9848
        - 0.003752884 * max(0.0, Q.sd_rg - 0.3017146) / 0.004328764   # -0.4%  sd_rg > 0.3017
        - 0.003640647 * max(0.0, Q.e2 - 0.03029714) * max(0.0, Q.log_sum_pt - 6.910131) / 0.0002282547   # -0.4%  e2 > 0.0303 and log_sum_pt > 6.91
        + 0.003239738 * max(0.0, 0.1207452 - Q.girth) * max(0.0, 997.0189 - Q.sum_pt_top50) / 0.7202273   # +0.3%  girth < 0.1207 and sum_pt_top50 < 997
        - 0.00308321 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.pt_7 - 34.53125) / 0.2313795   # -0.3%  z_dr_0p1_0p2 < 0.1203 and pt_7 > 34.53
        - 0.003079316 * max(0.0, Q.LHA - 0.1870291) * max(0.0, 0.9973959 - Q.psi_0p3) / 0.001748639   # -0.3%  LHA > 0.187 and psi_0p3 < 0.9974
        + 0.002422226 * max(0.0, Q.ptdr0_2 - 7.740999) / 2.384946   # +0.2%  ptdr0_2 > 7.741
        + 0.00239847 * max(0.0, 0.1207452 - Q.girth) * max(0.0, Q.pt1_dr01 - 5.351077) / 0.09847904   # +0.2%  girth < 0.1207 and pt1_dr01 > 5.351
        + 0.00221744 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +0.2%  sum_pt_top5 > 791.1
        + 0.001424839 * max(0.0, 886.3438 - Q.sum_pt_top30) / 11.17185   # +0.1%  sum_pt_top30 < 886.3
        - 0.001374453 * max(0.0, Q.n_pt_above_5 - 42.0) / 0.549916   # -0.1%  n_pt_above_5 > 42
        - 0.001128251 * max(0.0, 889.8503 - Q.sum_pt_top50) * max(0.0, Q.planar_flow - 0.4118472) / 1.038799   # -0.1%  sum_pt_top50 < 889.9 and planar_flow > 0.4118
        - 0.0004666747 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, 0.1651809 - Q.dr1_9) / 3.911359e-05   # -0.0%  tau1 > 0.1954 and dr1_9 < 0.1652
        + 0.0003686497 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, Q.sum_pt - 1167.447) / 0.008922112   # +0.0%  tau1 > 0.1954 and sum_pt > 1167
        + 0.0003350945 * max(0.0, Q.e2 - 0.03029714) * max(0.0, Q.z_dr_0p4_up - 0.0) / 3.251128e-06   # +0.0%  e2 > 0.0303 and z_dr_0p4_up > 0
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 15.02;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.0236 * (-0.01747209
        - 0.1426614 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # -14.3%  LHA < 0.3098
        + 0.1416962 * max(0.0, 0.08589404 - Q.girth) / 0.02745918   # +14.2%  girth < 0.08589
        + 0.1051071 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +10.5%  LHA < 0.3332
        - 0.104169 * max(0.0, 0.09749958 - Q.girth) / 0.03657082   # -10.4%  girth < 0.0975
        - 0.06510389 * max(0.0, 0.006142802 - Q.girth2_top15) / 0.002080651   # -6.5%  girth2_top15 < 0.006143
        - 0.04714527 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # -4.7%  e2 < 0.04083
        + 0.03753254 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +3.8%  e2 < 0.02516
        + 0.0329972 * max(0.0, 0.007887677 - Q.girth2_top15) / 0.00326329   # +3.3%  girth2_top15 < 0.007888
        + 0.03091747 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # +3.1%  z_dr_0p2_0p4 < 0.06849
        + 0.02409885 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # +2.4%  e2 < 0.04359
        + 0.01903235 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) / 0.03630126   # +1.9%  z_dr_0_0p05 > 0.8103
        - 0.01843606 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # -1.8%  log_sum_pt > 6.894
        + 0.0176853 * max(0.0, 19.0 - Q.n_dr_0p1_0p2) / 8.166005   # +1.8%  n_dr_0p1_0p2 < 19
        + 0.0169644 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 33.0 - Q.n_dr_0p1_0p2) / 87.77824   # +1.7%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 33
        + 0.01562597 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +1.6%  n_dr_0p2_0p4 < 15
        - 0.01428186 * max(0.0, Q.psi_0p1 - 0.6635952) / 0.1705715   # -1.4%  psi_0p1 > 0.6636
        + 0.01380301 * max(0.0, 1.409617 - Q.D2) / 0.1519846   # +1.4%  D2 < 1.41
        - 0.01351289 * max(0.0, 0.1623049 - Q.sj3_dr_max) / 0.01012502   # -1.4%  sj3_dr_max < 0.1623
        + 0.01342122 * max(0.0, 0.1894436 - Q.sj3_dr_max) / 0.01747403   # +1.3%  sj3_dr_max < 0.1894
        - 0.01241057 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -1.2%  psi_0p3 > 0.9897
        - 0.01170248 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005337976 - Q.girth2_top10) / 0.007083386   # -1.2%  n_dr_0p2_0p4 < 10 and girth2_top10 < 0.005338
        + 0.01069961 * max(0.0, 0.00130722 - Q.girth2_top10) / 0.0002794102   # +1.1%  girth2_top10 < 0.001307
        + 0.01032878 * max(0.0, 0.09338587 - Q.dr_0) / 0.04813256   # +1.0%  dr_0 < 0.09339
        - 0.009293812 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # -0.9%  sum_pt > 986.1
        + 0.008482848 * max(0.0, Q.log_sum_pt - 6.893714) * max(0.0, 0.3213081 - Q.sd_zg) / 0.005513471   # +0.8%  log_sum_pt > 6.894 and sd_zg < 0.3213
        - 0.007672156 * max(0.0, 3.376709e-05 - Q.e3) / 8.295151e-06   # -0.8%  e3 < 3.377e-05
        - 0.007490059 * max(0.0, 0.1548383 - Q.z_dr_0p1_0p2) / 0.06710802   # -0.7%  z_dr_0p1_0p2 < 0.1548
        + 0.006916722 * max(0.0, Q.psi_0p2 - 0.995185) / 0.0009240812   # +0.7%  psi_0p2 > 0.9952
        + 0.004929221 * max(0.0, 19.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.6025827 - Q.planar_flow) / 1.538522   # +0.5%  n_dr_0p1_0p2 < 19 and planar_flow < 0.6026
        - 0.004547254 * max(0.0, 1.409617 - Q.D2) * max(0.0, Q.n_real_top50 - 22.0) / 2.18239   # -0.5%  D2 < 1.41 and n_real_top50 > 22
        - 0.003857606 * max(0.0, 0.1732514 - Q.N2) / 0.004810221   # -0.4%  N2 < 0.1733
        + 0.003729816 * max(0.0, Q.log_sum_pt - 6.893714) * max(0.0, Q.z_top50_slots - 0.9704436) / 0.001405037   # +0.4%  log_sum_pt > 6.894 and z_top50_slots > 0.9704
        + 0.003508315 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +0.4%  n_dr_0p2_0p4 < 10
        - 0.003268156 * max(0.0, 1.409617 - Q.D2) * max(0.0, 0.04963857 - Q.z_7) / 0.002083888   # -0.3%  D2 < 1.41 and z_7 < 0.04964
        - 0.002695204 * max(0.0, 9.0 - Q.n_dr_0p1_0p2) / 1.720126   # -0.3%  n_dr_0p1_0p2 < 9
        + 0.002356468 * max(0.0, Q.z_dr_0p05_0p1 - 0.8509215) / 0.003556426   # +0.2%  z_dr_0p05_0p1 > 0.8509
        - 0.002297286 * max(0.0, 0.09749958 - Q.girth) * max(0.0, Q.tau4 - 0.008810529) / 0.0001863032   # -0.2%  girth < 0.0975 and tau4 > 0.008811
        - 0.002237623 * max(0.0, 1.409617 - Q.D2) * max(0.0, Q.z_5 - 0.03693777) / 0.002235851   # -0.2%  D2 < 1.41 and z_5 > 0.03694
        - 0.001914106 * max(0.0, 1.409617 - Q.D2) * max(0.0, 0.2191044 - Q.dr0_11) / 0.01751416   # -0.2%  D2 < 1.41 and dr0_11 < 0.2191
        - 0.00134468 * max(0.0, 1.409617 - Q.D2) * max(0.0, 0.09223087 - Q.dr_10) / 0.002765493   # -0.1%  D2 < 1.41 and dr_10 < 0.09223
        - 0.001178585 * max(0.0, 0.09749958 - Q.girth) * max(0.0, Q.pt1_dr01 - 12.6865) / 0.02199468   # -0.1%  girth < 0.0975 and pt1_dr01 > 12.69
        - 0.0008932259 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.dr0_7 - 0.1600212) / 0.02736434   # -0.1%  n_dr_0p2_0p4 < 10 and dr0_7 > 0.16
        - 0.0007490782 * max(0.0, Q.psi_0p2 - 0.995185) * max(0.0, 0.03638736 - Q.dr_10) / 3.315226e-06   # -0.1%  psi_0p2 > 0.9952 and dr_10 < 0.03639
        - 0.0006867538 * max(0.0, 0.08589404 - Q.girth) * max(0.0, Q.n_dr_0p4_up - 0.0) / 0.003406155   # -0.1%  girth < 0.08589 and n_dr_0p4_up > 0
        - 0.000617556 * max(0.0, 1.409617 - Q.D2) * max(0.0, 0.07954628 - Q.soft10_dr) / 0.002533336   # -0.1%  D2 < 1.41 and soft10_dr < 0.07955
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 9.286;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.285997 * (-0.02224629
        + 0.1053286 * max(0.0, 0.06171014 - Q.girth) / 0.01295056   # +10.5%  girth < 0.06171
        - 0.08686237 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, 0.01204531 - Q.mean_eta2) / 0.001166867   # -8.7%  log_sum_pt > 6.811 and mean_eta2 < 0.01205
        - 0.07428588 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -7.4%  psi_0p3 > 0.9897
        + 0.05043105 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.006615185 - Q.girth2_top15) / 1.491736e-05   # +5.0%  psi_0p3 > 0.9897 and girth2_top15 < 0.006615
        - 0.04961386 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # -5.0%  LHA < 0.2454
        - 0.04878633 * max(0.0, 0.2628766 - Q.sj3_dr_max) * max(0.0, Q.girth2_top10 - 0.004752876) / 3.407988e-05   # -4.9%  sj3_dr_max < 0.2629 and girth2_top10 > 0.004753
        + 0.04685433 * max(0.0, 0.05048381 - Q.girth) / 0.008533025   # +4.7%  girth < 0.05048
        - 0.043387 * max(0.0, 0.2284021 - Q.LHA) / 0.02716657   # -4.3%  LHA < 0.2284
        + 0.03699737 * max(0.0, Q.psi_0p2 - 0.9734513) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 0.1343004   # +3.7%  psi_0p2 > 0.9735 and n_dr_0p1_0p2 < 21
        - 0.03588156 * max(0.0, Q.log_sum_pt - 6.811175) / 0.1383141   # -3.6%  log_sum_pt > 6.811
        + 0.03408383 * max(0.0, 0.005850286 - Q.mean_eta2) / 0.00275351   # +3.4%  mean_eta2 < 0.00585
        + 0.03236547 * max(0.0, 0.05048381 - Q.girth) * max(0.0, Q.log_sum_pt - 6.811175) / 0.001311076   # +3.2%  girth < 0.05048 and log_sum_pt > 6.811
        - 0.02937438 * max(0.0, 0.01414829 - Q.girth2_top10) / 0.00877908   # -2.9%  girth2_top10 < 0.01415
        + 0.02684471 * max(0.0, 4.450169 - Q.D2) / 2.013555   # +2.7%  D2 < 4.45
        - 0.02542268 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # -2.5%  e2 > 0.0303
        + 0.0253219 * max(0.0, 0.2628766 - Q.sj3_dr_max) / 0.05345657   # +2.5%  sj3_dr_max < 0.2629
        + 0.02353464 * max(0.0, 0.2628766 - Q.sj3_dr_max) * max(0.0, Q.eccentricity - 0.5245966) / 0.01227931   # +2.4%  sj3_dr_max < 0.2629 and eccentricity > 0.5246
        + 0.01958854 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +2.0%  n_dr_0p2_0p4 < 10
        + 0.01885334 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 7.139296 - Q.log_sum_pt) / 0.00115378   # +1.9%  psi_0p3 > 0.9897 and log_sum_pt < 7.139
        - 0.01841049 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # -1.8%  LHA < 0.187
        + 0.01802421 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +1.8%  LHA > 0.372
        - 0.01709053 * max(0.0, 0.1506299 - Q.sj3_dr_max) / 0.008148644   # -1.7%  sj3_dr_max < 0.1506
        + 0.01390432 * max(0.0, 0.1840219 - Q.sj3_dr12) / 0.05790731   # +1.4%  sj3_dr12 < 0.184
        - 0.01339188 * max(0.0, Q.eccentricity - 0.8903081) / 0.02227339   # -1.3%  eccentricity > 0.8903
        - 0.0121398 * max(0.0, 0.1048824 - Q.sj3_dr12) / 0.02123626   # -1.2%  sj3_dr12 < 0.1049
        - 0.0115456 * max(0.0, 846.1934 - Q.sum_pt_top20) / 22.83716   # -1.2%  sum_pt_top20 < 846.2
        - 0.01121537 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # -1.1%  LHA > 0.4042
        + 0.009755963 * max(0.0, 0.1999777 - Q.sj3_dr_max) / 0.02143   # +1.0%  sj3_dr_max < 0.2
        + 0.007688994 * max(0.0, Q.LHA - 0.3719813) * max(0.0, Q.log_sum_pt - 6.856375) / 0.0004589561   # +0.8%  LHA > 0.372 and log_sum_pt > 6.856
        + 0.006505024 * max(0.0, 1167.447 - Q.sum_pt) / 137.8628   # +0.7%  sum_pt < 1167
        - 0.006123494 * max(0.0, Q.psi_0p2 - 0.9734513) / 0.0116938   # -0.6%  psi_0p2 > 0.9735
        + 0.005548863 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, Q.z_1st - 0.1586697) / 0.0005591086   # +0.6%  psi_0p3 > 0.9897 and z_1st > 0.1587
        - 0.005205823 * max(0.0, Q.psi_0p2 - 0.9734513) * max(0.0, 40.0 - Q.n_real_top40) / 0.06806351   # -0.5%  psi_0p2 > 0.9735 and n_real_top40 < 40
        + 0.00412153 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, Q.sd_zg - 0.1961626) / 0.0005987903   # +0.4%  psi_0p3 > 0.9897 and sd_zg > 0.1962
        + 0.003432442 * max(0.0, Q.LHA - 0.404204) * max(0.0, 0.0329485 - Q.C2_b2) / 2.756829e-05   # +0.3%  LHA > 0.4042 and C2_b2 < 0.03295
        - 0.003421817 * max(0.0, 0.2628766 - Q.sj3_dr_max) * max(0.0, Q.pt_10 - 11.74219) / 0.6747505   # -0.3%  sj3_dr_max < 0.2629 and pt_10 > 11.74
        + 0.003224456 * max(0.0, Q.LHA - 0.3719813) * max(0.0, 0.003687605 - Q.lam2) / 4.619984e-06   # +0.3%  LHA > 0.372 and lam2 < 0.003688
        - 0.002405639 * max(0.0, 0.1210264 - Q.sj3_dr_max) / 0.004637661   # -0.2%  sj3_dr_max < 0.121
        + 0.002262911 * max(0.0, Q.LHA - 0.3719813) * max(0.0, Q.pt_5 - 48.28125) / 0.01704158   # +0.2%  LHA > 0.372 and pt_5 > 48.28
        - 0.002220527 * max(0.0, Q.n_dr_0_0p05 - 20.0) * max(0.0, 3.259863 - Q.soft4_pt) / 2.781449   # -0.2%  n_dr_0_0p05 > 20 and soft4_pt < 3.26
        + 0.001923027 * max(0.0, 0.06171014 - Q.girth) * max(0.0, Q.max_dr - 0.316623) / 0.0007980345   # +0.2%  girth < 0.06171 and max_dr > 0.3166
        + 0.001856587 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # +0.2%  e2 > 0.06524
        + 0.001653482 * max(0.0, Q.n_dr_0_0p05 - 20.0) / 1.33142   # +0.2%  n_dr_0_0p05 > 20
        - 0.001652686 * max(0.0, Q.LHA - 0.3719813) * max(0.0, Q.z_5 - 0.04812833) / 1.433613e-05   # -0.2%  LHA > 0.372 and z_5 > 0.04813
        - 0.0007310764 * max(0.0, Q.girth2_top3 - 0.02353672) / 0.0004246139   # -0.1%  girth2_top3 > 0.02354
        - 0.000290422 * max(0.0, Q.ptdr0_6 - 11.6057) * max(0.0, 0.0003690273 - Q.soft4_z) / 2.476812e-07   # -0.0%  ptdr0_6 > 11.61 and soft4_z < 0.000369
        + 0.0002629104 * max(0.0, Q.LHA - 0.404204) * max(0.0, Q.z_dr_0p4_up - 0.0) / 3.831631e-06   # +0.0%  LHA > 0.4042 and z_dr_0p4_up > 0
        - 0.0001723285 * max(0.0, Q.pt_13 - 28.20312) * max(0.0, Q.eta_9 - 0.1269531) / 0.0002273219   # -0.0%  pt_13 > 28.2 and eta_9 > 0.127
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 34.31;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.30775 * (0.06244938
        - 0.4081718 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # -40.8%  log_sum_pt < 6.989
        + 0.3728289 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # +37.3%  sum_pt < 1085
        + 0.04260803 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # +4.3%  e3 < 0.0005178
        - 0.01781306 * max(0.0, Q.girth - 0.08589404) / 0.01080971   # -1.8%  girth > 0.08589
        - 0.01645879 * max(0.0, 0.0001841806 - Q.e3) / 0.0001220279   # -1.6%  e3 < 0.0001842
        - 0.01610569 * max(0.0, Q.girth - 0.09749958) / 0.008315822   # -1.6%  girth > 0.0975
        - 0.01588987 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -1.6%  psi_0p2 > 0.9087
        + 0.01065569 * max(0.0, Q.tau4 - 0.01517184) / 0.004291999   # +1.1%  tau4 > 0.01517
        + 0.0101741 * max(0.0, Q.LHA - 0.3203321) / 0.01671666   # +1.0%  LHA > 0.3203
        - 0.009318369 * max(0.0, 26.0 - Q.n_dr_0p1_0p2) / 14.20854   # -0.9%  n_dr_0p1_0p2 < 26
        + 0.007097742 * max(0.0, 41.4375 - Q.pt_9) / 15.12338   # +0.7%  pt_9 < 41.44
        + 0.007047716 * max(0.0, Q.e2 - 0.04082832) / 0.003398869   # +0.7%  e2 > 0.04083
        + 0.00615986 * max(0.0, 1048.098 - Q.sum_pt_top50) / 44.36839   # +0.6%  sum_pt_top50 < 1048
        - 0.005314705 * max(0.0, Q.girth2_top15 - 0.009962397) / 0.002311908   # -0.5%  girth2_top15 > 0.009962
        + 0.005313053 * max(0.0, Q.girth - 0.09749958) * max(0.0, 1167.447 - Q.sum_pt) / 1.406455   # +0.5%  girth > 0.0975 and sum_pt < 1167
        + 0.005062434 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 6.811175 - Q.log_sum_pt) / 1.755002   # +0.5%  sum_pt < 1085 and log_sum_pt < 6.811
        - 0.004879041 * max(0.0, Q.sum_pt - 1167.447) / 14.21193   # -0.5%  sum_pt > 1167
        - 0.004861693 * max(0.0, Q.LHA - 0.4331369) / 0.001006142   # -0.5%  LHA > 0.4331
        - 0.004435286 * max(0.0, 0.0001841806 - Q.e3) * max(0.0, 41.4375 - Q.pt_9) / 0.00190888   # -0.4%  e3 < 0.0001842 and pt_9 < 41.44
        + 0.004209614 * max(0.0, 0.1872805 - Q.z_dr_0p1_0p2) / 0.08737974   # +0.4%  z_dr_0p1_0p2 < 0.1873
        - 0.004000338 * max(0.0, 23.0 - Q.n_pt_above_5) / 1.859217   # -0.4%  n_pt_above_5 < 23
        + 0.003841244 * max(0.0, Q.sum_pt_top30 - 1111.245) / 12.62573   # +0.4%  sum_pt_top30 > 1111
        + 0.002637478 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # +0.3%  sum_pt_top50 < 959.1
        + 0.002610958 * max(0.0, Q.girth2_top15 - 0.009962397) * max(0.0, 1225.842 - Q.sum_pt_top40) / 0.6413342   # +0.3%  girth2_top15 > 0.009962 and sum_pt_top40 < 1226
        - 0.001561228 * max(0.0, 708.7945 - Q.sum_pt_top15) / 11.69336   # -0.2%  sum_pt_top15 < 708.8
        + 0.001223734 * max(0.0, 1048.098 - Q.sum_pt_top50) * max(0.0, 0.0003125151 - Q.girth2_top2) / 0.002122269   # +0.1%  sum_pt_top50 < 1048 and girth2_top2 < 0.0003125
        + 0.001080267 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.3437357 - Q.max_dr) / 1.395101   # +0.1%  sum_pt < 1085 and max_dr < 0.3437
        - 0.001056859 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # -0.1%  C2 > 0.06656
        - 0.001013346 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # -0.1%  log_sum_pt > 7.139
        + 0.001003303 * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 1.386277   # +0.1%  n_dr_0p1_0p2 > 21
        + 0.0009005334 * max(0.0, Q.girth - 0.1564779) / 0.0006617163   # +0.1%  girth > 0.1565
        + 0.0008738121 * max(0.0, 23.0 - Q.n_pt_above_5) * max(0.0, 3.345339 - Q.D2) / 1.366679   # +0.1%  n_pt_above_5 < 23 and D2 < 3.345
        + 0.0006433007 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.psi_0p3 - 0.9777125) / 8.852804e-05   # +0.1%  log_sum_pt > 7.139 and psi_0p3 > 0.9777
        - 0.0006382772 * max(0.0, 26.0 - Q.n_dr_0p1_0p2) * max(0.0, 1027.303 - Q.sum_pt_top30) / 631.9123   # -0.1%  n_dr_0p1_0p2 < 26 and sum_pt_top30 < 1027
        + 0.0006369927 * max(0.0, Q.tau4 - 0.01517184) * max(0.0, 0.02529394 - Q.dr0_5) / 2.364705e-05   # +0.1%  tau4 > 0.01517 and dr0_5 < 0.02529
        - 0.0004924689 * max(0.0, 0.0005178279 - Q.e3) * max(0.0, 0.9966167 - Q.psi_0p3) / 2.589793e-06   # -0.0%  e3 < 0.0005178 and psi_0p3 < 0.9966
        - 0.0003774986 * max(0.0, Q.girth - 0.1207452) / 0.004285542   # -0.0%  girth > 0.1207
        + 0.0002023604 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, -0.004917145 - Q.phi_0) / 4.999945e-05   # +0.0%  log_sum_pt > 7.139 and phi_0 < -0.004917
        - 0.0002017154 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, -0.04013062 - Q.phi_0) / 1.626186e-05   # -0.0%  log_sum_pt > 7.139 and phi_0 < -0.04013
        + 0.0001450976 * max(0.0, Q.girth - 0.1402186) * max(0.0, Q.soft6_pt - 4.250195) / 9.168767e-05   # +0.0%  girth > 0.1402 and soft6_pt > 4.25
        - 0.0001233652 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.absphi_13 - 0.01228943) / 0.0001757278   # -0.0%  log_sum_pt > 7.139 and absphi_13 > 0.01229
        + 0.0001046886 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.01134873 - Q.dr_13) / 2.590289e-06   # +0.0%  log_sum_pt > 7.139 and dr_13 < 0.01135
        - 0.0001039025 * max(0.0, Q.girth2_top15 - 0.005383629) / 0.003662853   # -0.0%  girth2_top15 > 0.005384
        + 8.144425e-05 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, -0.07818909 - Q.phi_0) / 6.160563e-06   # +0.0%  log_sum_pt > 7.139 and phi_0 < -0.07819
        - 4.040029e-05 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, -0.0297699 - Q.phi_0) / 2.18725e-05   # -0.0%  log_sum_pt > 7.139 and phi_0 < -0.02977
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 18.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.22941 * (0.1270524
        - 0.1397898 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # -14.0%  LHA < 0.3332
        - 0.09652514 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # -9.7%  tau1 < 0.1507
        + 0.07619707 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # +7.6%  LHA < 0.3024
        - 0.06196501 * max(0.0, Q.girth - 0.076787) / 0.01351025   # -6.2%  girth > 0.07679
        + 0.05096173 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # +5.1%  sd_rg < 0.3017
        - 0.04787416 * max(0.0, 7.876005e-05 - Q.e3) / 3.594075e-05   # -4.8%  e3 < 7.876e-05
        + 0.03879139 * max(0.0, Q.sd_rg - 0.1596365) / 0.03553981   # +3.9%  sd_rg > 0.1596
        - 0.03860247 * max(0.0, Q.girth - 0.08068193) / 0.0122472   # -3.9%  girth > 0.08068
        + 0.03735931 * max(0.0, Q.girth - 0.09749958) / 0.008315822   # +3.7%  girth > 0.0975
        - 0.0335209 * max(0.0, 0.1751567 - Q.tau1) / 0.09101058   # -3.4%  tau1 < 0.1752
        + 0.03272667 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +3.3%  tau21_b2 < 0.3425
        - 0.03012142 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 1156.659 - Q.sum_pt_top50) / 14.04661   # -3.0%  tau21_b2 < 0.3425 and sum_pt_top50 < 1157
        - 0.02822665 * max(0.0, Q.girth - 0.08589404) / 0.01080971   # -2.8%  girth > 0.08589
        - 0.02508029 * max(0.0, Q.sd_rg - 0.2042612) / 0.02046744   # -2.5%  sd_rg > 0.2043
        + 0.0244659 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.01976735 - Q.girth2_top10) / 0.001359663   # +2.4%  tau21_b2 < 0.3425 and girth2_top10 < 0.01977
        - 0.02280457 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # -2.3%  n_dr_0p1_0p2 < 17
        - 0.02075768 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.00727763 - Q.girth2_top15) / 0.0001401748   # -2.1%  tau21_b2 < 0.3425 and girth2_top15 < 0.007278
        + 0.01969763 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +2.0%  z_dr_0p1_0p2 < 0.1203
        + 0.01828808 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +1.8%  e2 < 0.03481
        - 0.01513447 * max(0.0, 3.376709e-05 - Q.e3) / 8.295151e-06   # -1.5%  e3 < 3.377e-05
        + 0.01485081 * max(0.0, Q.sd_rg - 0.15159) / 0.03939137   # +1.5%  sd_rg > 0.1516
        + 0.0121704 * max(0.0, 0.2941033 - Q.LHA) / 0.05718399   # +1.2%  LHA < 0.2941
        + 0.01049007 * max(0.0, Q.girth - 0.076787) * max(0.0, 0.4021783 - Q.max_dr) / 0.0004881945   # +1.0%  girth > 0.07679 and max_dr < 0.4022
        + 0.009950498 * max(0.0, Q.girth - 0.1207452) / 0.004285542   # +1.0%  girth > 0.1207
        + 0.008874129 * max(0.0, Q.z_dr_0_0p05 - 0.6283153) / 0.1153015   # +0.9%  z_dr_0_0p05 > 0.6283
        + 0.008167102 * max(0.0, 0.03787151 - Q.M3) / 0.01370579   # +0.8%  M3 < 0.03787
        - 0.007357183 * max(0.0, 0.001838217 - Q.soft9_z) / 0.0003773283   # -0.7%  soft9_z < 0.001838
        + 0.007119777 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +0.7%  psi_0p3 > 0.9924
        + 0.007099872 * max(0.0, 0.05347848 - Q.dr_6) / 0.01424699   # +0.7%  dr_6 < 0.05348
        + 0.005984934 * max(0.0, 0.03480688 - Q.e2) * max(0.0, 0.001838217 - Q.soft9_z) / 5.241978e-06   # +0.6%  e2 < 0.03481 and soft9_z < 0.001838
        + 0.005847302 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.006142802 - Q.girth2_top15) / 7.19088e-05   # +0.6%  tau21_b2 < 0.3425 and girth2_top15 < 0.006143
        - 0.005727715 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.05347848 - Q.dr_6) / 5.520862e-05   # -0.6%  psi_0p3 > 0.9924 and dr_6 < 0.05348
        + 0.004828487 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) * max(0.0, 17.82896 - Q.pt1_dr01) / 93.02539   # +0.5%  n_dr_0p1_0p2 < 17 and pt1_dr01 < 17.83
        - 0.004658023 * max(0.0, Q.girth - 0.09749958) * max(0.0, 0.4021783 - Q.max_dr) / 0.0002725179   # -0.5%  girth > 0.0975 and max_dr < 0.4022
        + 0.00455869 * max(0.0, Q.girth2_top10 - 0.0001295334) / 0.006638188   # +0.5%  girth2_top10 > 0.0001295
        + 0.004466584 * max(0.0, Q.n_real_top50 - 43.0) * max(0.0, Q.tau32 - 0.577419) / 0.480675   # +0.4%  n_real_top50 > 43 and tau32 > 0.5774
        - 0.003744308 * max(0.0, Q.psi_0p3 - 0.9638082) * max(0.0, 182.125 - Q.pt_1) / 1.522171   # -0.4%  psi_0p3 > 0.9638 and pt_1 < 182.1
        - 0.003602572 * max(0.0, Q.girth - 0.1207452) * max(0.0, 0.4021783 - Q.max_dr) / 0.0001405604   # -0.4%  girth > 0.1207 and max_dr < 0.4022
        + 0.002642629 * max(0.0, 0.5100475 - Q.tau21) / 0.1343519   # +0.3%  tau21 < 0.51
        + 0.002427045 * max(0.0, 0.1219401 - Q.tau21_b2) / 0.01156068   # +0.2%  tau21_b2 < 0.1219
        + 0.00201592 * max(0.0, Q.sd_rg - 0.15159) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.0411792   # +0.2%  sd_rg > 0.1516 and n_pt_above_50 > 4
        - 0.001570141 * max(0.0, 0.003398536 - Q.z_dr_0p2_0p4) / 0.0005454047   # -0.2%  z_dr_0p2_0p4 < 0.003399
        - 0.001181967 * max(0.0, Q.z_dr_0p05_0p1 - 0.7108211) / 0.01402576   # -0.1%  z_dr_0p05_0p1 > 0.7108
        + 0.0009511705 * max(0.0, 0.003398536 - Q.z_dr_0p2_0p4) * max(0.0, 0.04002298 - Q.dr_6) / 2.69213e-06   # +0.1%  z_dr_0p2_0p4 < 0.003399 and dr_6 < 0.04002
        + 0.0008223703 * max(0.0, 0.003398536 - Q.z_dr_0p2_0p4) * max(0.0, 0.002970691 - Q.soft9_z) / 3.781212e-07   # +0.1%  z_dr_0p2_0p4 < 0.003399 and soft9_z < 0.002971
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 6.679;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.679213 * (-0.04100568
        - 0.08740358 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # -8.7%  tau1 < 0.07709
        - 0.05896415 * max(0.0, Q.log_sum_pt - 6.915514) / 0.04811413   # -5.9%  log_sum_pt > 6.916
        + 0.05557638 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +5.6%  girth2_top5 < 0.00833
        + 0.0553481 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +5.5%  sum_pt > 907.9
        - 0.05067579 * max(0.0, 0.02647293 - Q.z_dr_0p2_0p4) / 0.01168948   # -5.1%  z_dr_0p2_0p4 < 0.02647
        + 0.04731423 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +4.7%  lam2 < 0.001776
        + 0.04610006 * max(0.0, Q.sum_pt_top30 - 886.3438) / 117.6357   # +4.6%  sum_pt_top30 > 886.3
        + 0.04541099 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +4.5%  psi_0p3 > 0.9897
        - 0.04323892 * max(0.0, Q.sj2_dr - 0.1512157) / 0.05988199   # -4.3%  sj2_dr > 0.1512
        + 0.04134309 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +4.1%  z_dr_0p1_0p2 < 0.1203
        + 0.04129471 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +4.1%  sj2_dr > 0.2232
        + 0.03840186 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # +3.8%  log_sum_pt > 7.017
        + 0.03496631 * max(0.0, 0.1881908 - Q.sd_rg) / 0.07278694   # +3.5%  sd_rg < 0.1882
        - 0.03374971 * max(0.0, Q.sum_pt_top40 - 1001.523) / 47.19068   # -3.4%  sum_pt_top40 > 1002
        + 0.02726164 * max(0.0, Q.psi_0p1 - 0.9184255) / 0.01693566   # +2.7%  psi_0p1 > 0.9184
        + 0.02683219 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # +2.7%  girth2_top10 < 0.007679
        + 0.02297355 * max(0.0, 0.08068193 - Q.girth) / 0.02368455   # +2.3%  girth < 0.08068
        - 0.02230551 * max(0.0, Q.sum_pt_top30 - 1038.262) / 23.89568   # -2.2%  sum_pt_top30 > 1038
        - 0.02156553 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) / 3.639534   # -2.2%  n_dr_0p2_0p4 > 8
        - 0.02092634 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) / 0.3666529   # -2.1%  z_dr_0_0p05 < 0.8459
        - 0.02037646 * max(0.0, Q.psi_0p1 - 0.707925) / 0.1360654   # -2.0%  psi_0p1 > 0.7079
        - 0.0179831 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, 689.25 - Q.sum_pt_top2) / 1.278535   # -1.8%  girth2_top5 < 0.00833 and sum_pt_top2 < 689.2
        - 0.01737661 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 1.480811 - Q.soft5_pt) / 0.02368753   # -1.7%  z_dr_0p1_0p2 < 0.1203 and soft5_pt < 1.481
        - 0.01443833 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.4299592 - Q.tau21_b2) / 0.001167326   # -1.4%  psi_0p3 > 0.9897 and tau21_b2 < 0.43
        - 0.01198728 * max(0.0, 0.04748396 - Q.z_dr_0p1_0p2) / 0.01200196   # -1.2%  z_dr_0p1_0p2 < 0.04748
        + 0.00830979 * max(0.0, 0.0004140594 - Q.soft5_z) / 1.144764e-05   # +0.8%  soft5_z < 0.0004141
        - 0.007945787 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 0.4823776 - Q.tau21_b2) / 0.006691474   # -0.8%  z_dr_0p1_0p2 < 0.1203 and tau21_b2 < 0.4824
        - 0.007037791 * max(0.0, 0.003213724 - Q.girth2_top10) / 0.0009423838   # -0.7%  girth2_top10 < 0.003214
        - 0.006710149 * max(0.0, 0.5297852 - Q.soft5_pt) / 0.01859865   # -0.7%  soft5_pt < 0.5298
        + 0.006642516 * max(0.0, 0.002787636 - Q.mean_phi2) / 0.0008254854   # +0.7%  mean_phi2 < 0.002788
        - 0.006203369 * max(0.0, 2.410481 - Q.D2) / 0.587301   # -0.6%  D2 < 2.41
        - 0.006015804 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, Q.sj3_dr23 - 0.2563461) / 0.002209452   # -0.6%  sj2_dr > 0.2232 and sj3_dr23 > 0.2563
        + 0.006005423 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 0.05112769 - Q.C2_b2) / 0.001667822   # +0.6%  z_dr_0p1_0p2 < 0.1203 and C2_b2 < 0.05113
        + 0.00540749 * max(0.0, Q.psi_0p1 - 0.707925) * max(0.0, Q.sj3_dr13 - 0.228329) / 0.002890489   # +0.5%  psi_0p1 > 0.7079 and sj3_dr13 > 0.2283
        - 0.005072697 * max(0.0, Q.z_dr_0p05_0p1 - 0.8509215) / 0.003556426   # -0.5%  z_dr_0p05_0p1 > 0.8509
        - 0.004228512 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.03006824 - Q.dr_4) / 0.000106252   # -0.4%  sj2_dr > 0.2232 and dr_4 < 0.03007
        - 0.003759571 * max(0.0, Q.psi_0p2 - 0.9976427) / 0.0003015943   # -0.4%  psi_0p2 > 0.9976
        - 0.003609876 * max(0.0, Q.sj3_dr13 - 0.3682013) / 0.001622675   # -0.4%  sj3_dr13 > 0.3682
        - 0.003460662 * max(0.0, Q.sj2_dr - 0.2780918) / 0.009227968   # -0.3%  sj2_dr > 0.2781
        + 0.003211305 * max(0.0, Q.z_dr_0p05_0p1 - 0.7108211) / 0.01402576   # +0.3%  z_dr_0p05_0p1 > 0.7108
        + 0.003069238 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, 40.0 - Q.n_real_top40) / 0.02186912   # +0.3%  girth2_top5 < 0.00833 and n_real_top40 < 40
        + 0.002526309 * max(0.0, Q.psi_0p1 - 0.707925) * max(0.0, Q.dr1_11 - 0.2621232) / 0.0003748677   # +0.3%  psi_0p1 > 0.7079 and dr1_11 > 0.2621
        - 0.002510962 * max(0.0, 0.008293585 - Q.sj3_z3) / 0.000142753   # -0.3%  sj3_z3 < 0.008294
        - 0.002293442 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p4_up - 0.0) / 0.006297153   # -0.2%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p4_up > 0
        - 0.001274929 * max(0.0, 0.08068193 - Q.girth) * max(0.0, Q.dr1_11 - 0.1911348) / 0.0001794349   # -0.1%  girth < 0.08068 and dr1_11 > 0.1911
        - 0.0008899503 * max(0.0, Q.psi_0p1 - 0.707925) * max(0.0, Q.dr1_11 - 0.2195171) / 0.000707107   # -0.1%  psi_0p1 > 0.7079 and dr1_11 > 0.2195
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.710368224789916, 2.811699527310924, 0.2703170168067227, 0.508775262605042, 0.909759768907563, 1.1580067226890756, 0.6294261554621848, 0.5860892857142858, 1.368996113445378, 1.0739155462184875, 1.3264737394957984, 0.6051550420168067, 0.677815756302521, 1.5375052521008403, 0.3674170693277311, 0.2975621848739496]
T = [4.171266439075629, 2.695772152376575, 4.7716683035714285, 4.900374002100841, 4.320680315290177]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +42%, n4 -19%, n9 +13%, n3 -9%, n5 -9%, n12 +4% ...
            + 0.4212898 * h[1] / H_AVG[1]
            - 0.1908389 * h[4] / H_AVG[4]
            + 0.1327505 * h[9] / H_AVG[9]
            - 0.09147856 * h[3] / H_AVG[3]
            - 0.08675473 * h[5] / H_AVG[5]
            + 0.03808509 * h[12] / H_AVG[12]
            + 0.02357745 * h[6] / H_AVG[6]
            + 0.01025615 * h[8] / H_AVG[8]
            - 0.004968791 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -36%, n9 +22%, n1 -20%, n12 +9%, n11 -6%, n6 +5% ...
            - 0.3585688 * h[4] / H_AVG[4]
            + 0.2240833 * h[9] / H_AVG[9]
            - 0.1955631 * h[1] / H_AVG[1]
            + 0.08643133 * h[12] / H_AVG[12]
            - 0.05612075 * h[11] / H_AVG[11]
            + 0.05107515 * h[6] / H_AVG[6]
            + 0.0125343 * h[2] / H_AVG[2]
            + 0.007934856 * h[8] / H_AVG[8]
            - 0.007688392 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +14%, n0 +11%, n14 -11%, n7 -8%, n11 +8% ...
            - 0.2510383 * h[8] / H_AVG[8]
            + 0.1403016 * h[5] / H_AVG[5]
            + 0.1116541 * h[0] / H_AVG[0]
            - 0.1058746 * h[14] / H_AVG[14]
            - 0.07676682 * h[7] / H_AVG[7]
            + 0.07530088 * h[11] / H_AVG[11]
            + 0.06553891 * h[4] / H_AVG[4]
            - 0.05770784 * h[12] / H_AVG[12]
            - 0.04923205 * h[9] / H_AVG[9]
            + 0.04664808 * h[3] / H_AVG[3]
            - 0.01169254 * h[15] / H_AVG[15]
            + 0.008244315 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -20%, n5 +16%, n6 -12%, n7 +11%, n3 +5% ...
            - 0.2619053 * h[8] / H_AVG[8]
            - 0.1993228 * h[0] / H_AVG[0]
            + 0.162463 * h[5] / H_AVG[5]
            - 0.1244306 * h[6] / H_AVG[6]
            + 0.1083883 * h[7] / H_AVG[7]
            + 0.0454229 * h[3] / H_AVG[3]
            + 0.04322475 * h[12] / H_AVG[12]
            + 0.03415632 * h[15] / H_AVG[15]
            - 0.02068595 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -32%, n10 +30%, n5 -13%, n8 +7%, n12 -6%, n4 +4% ...
            - 0.3224872 * h[13] / H_AVG[13]
            + 0.3022088 * h[10] / H_AVG[10]
            - 0.125632 * h[5] / H_AVG[5]
            + 0.06683499 * h[8] / H_AVG[8]
            - 0.05882891 * h[12] / H_AVG[12]
            + 0.03947988 * h[4] / H_AVG[4]
            - 0.03815085 * h[7] / H_AVG[7]
            - 0.02582598 * h[15] / H_AVG[15]
            + 0.0205514 * h[0] / H_AVG[0]
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
