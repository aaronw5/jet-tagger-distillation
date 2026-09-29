"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  12.9%   (on for 53% of jets)
  neuron  1:  12.0%   (on for 93% of jets)
  neuron  5:  11.3%   (on for 80% of jets)
  neuron  4:  10.1%   (on for 68% of jets)
  neuron  0:   7.7%   (on for 62% of jets)
  neuron 13:   7.4%   (on for 88% of jets)
  neuron  9:   6.8%   (on for 64% of jets)
  neuron 10:   6.8%   (on for 80% of jets)
  neuron 12:   4.9%   (on for 56% of jets)
  neuron  7:   4.4%   (on for 36% of jets)
  neuron  6:   4.0%   (on for 68% of jets)
  neuron 11:   3.4%   (on for 70% of jets)
  neuron  3:   3.3%   (on for 52% of jets)
  neuron 15:   2.6%   (on for 66% of jets)
  neuron 14:   1.9%   (on for 37% of jets)
  neuron  2:   0.6%   (on for 40% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 79.2% (the network: 81.1%); same class as the network for 90.4% of jets.

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
  Q.ptdr0_10               pT10 · ΔR(0, 10) [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.ptdr0_12               pT12 · ΔR(0, 12) [GeV]
  Q.pt_13                  pT of particle 13 [GeV]
  Q.ptdr0_14               pT14 · ΔR(0, 14) [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.pt_8                   pT of particle 8 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_pt              pT [GeV] of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_0                    pT of particle 0 / total pT
  Q.z_1                    pT of particle 1 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_8                    pT of particle 8 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.z_1st                  largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_12              |Δη| of particle 12
  Q.abseta_9               |Δη| of particle 9
  Q.soft8_abseta           |Δη| of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.absphi_1               |Δφ| of particle 1
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft2_dr0              ΔR between the hardest and the 2. softest real particle (0 if among the 15 hardest)
  Q.soft4_dr0              ΔR between the hardest and the 4. softest real particle (0 if among the 15 hardest)
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_11                 ΔR between particle 11 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_12                  ΔR of particle 12 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.soft2_dr               ΔR from the jet axis of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_dr               ΔR from the jet axis of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_dr               ΔR from the jet axis of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
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
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
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
        ptdr0_10=pt[10] * math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        pt_11=pt[11],
        ptdr0_12=pt[12] * math.sqrt(dist2(0, 12)) if pt[12] > 0 else 0.0,
        pt_13=pt[13],
        ptdr0_14=pt[14] * math.sqrt(dist2(0, 14)) if pt[14] > 0 else 0.0,
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        pt_8=pt[8],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft10_pt=softp(10, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        soft9_pt=softp(9, 'pt'),
        z_0=z[0],
        z_1=z[1],
        z_3=z[3],
        z_7=z[7],
        z_8=z[8],
        soft1_z=softp(1, 'z'),
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
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
        z_1st=zs[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_zg=softdrop("zg"),
        abseta_12=abs(eta[12]),
        abseta_9=abs(eta[9]),
        soft8_abseta=softp(8, 'abseta'),
        absphi_1=abs(phi[1]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft2_dr0=softp(2, 'dr0'),
        soft4_dr0=softp(4, 'dr0'),
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_11=math.sqrt(dist2(1, 11)) if pt[11] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_12=dr[12] if pt[12] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        soft2_dr=softp(2, 'dr'),
        soft4_dr=softp(4, 'dr'),
        soft6_dr=softp(6, 'dr'),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        phi_0=phi[0],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
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
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
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
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 31.17;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.16507 * (-0.01866051
        - 0.1537787 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # -15.4%  LHA < 0.3098
        + 0.1476227 * max(0.0, 0.08068193 - Q.sum_z_dr) / 0.02368455   # +14.8%  sum_z_dr < 0.08068
        + 0.08695265 * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) / 0.00326329   # +8.7%  sum_z_dr2_top15 < 0.007888
        - 0.06111375 * max(0.0, Q.sj2_dr - 0.1512157) / 0.05988199   # -6.1%  sj2_dr > 0.1512
        - 0.05903196 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # -5.9%  sum_z_dr < 0.0975
        + 0.05235215 * max(0.0, 0.3203321 - Q.LHA) / 0.07499257   # +5.2%  LHA < 0.3203
        + 0.04468861 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_pt_top50 - 889.8503) / 1563.946   # +4.5%  n_dr_0p2_0p4 < 18 and sum_pt_top50 > 889.9
        + 0.04244447 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +4.2%  LHA < 0.3332
        + 0.03974702 * max(0.0, Q.sj2_dr - 0.1411617) / 0.06721719   # +4.0%  sj2_dr > 0.1412
        - 0.03745427 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # -3.7%  sum_z_dr2_top15 < 0.009962
        - 0.03382377 * max(0.0, 0.005383629 - Q.sum_z_dr2_top15) / 0.001666876   # -3.4%  sum_z_dr2_top15 < 0.005384
        - 0.02614945 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # -2.6%  dr_0 < 0.06413
        + 0.02285137 * max(0.0, Q.sj2_dr - 0.1745007) / 0.04529905   # +2.3%  sj2_dr > 0.1745
        + 0.02207751 * max(0.0, 0.05775119 - Q.dr_0) / 0.02089184   # +2.2%  dr_0 < 0.05775
        - 0.02095913 * max(0.0, Q.sj2_dr - 0.06289464) / 0.1327343   # -2.1%  sj2_dr > 0.06289
        - 0.01937901 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.log_sum_pt - 6.920349) / 0.4496771   # -1.9%  n_dr_0p2_0p4 < 18 and log_sum_pt > 6.92
        - 0.01547741 * max(0.0, 0.04466492 - Q.tau1) / 0.005217805   # -1.5%  tau1 < 0.04466
        + 0.0130471 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # +1.3%  LHA < 0.187
        - 0.01167179 * max(0.0, 0.05048381 - Q.sum_z_dr) / 0.008533025   # -1.2%  sum_z_dr < 0.05048
        - 0.009466332 * max(0.0, 2.883342e-05 - Q.e3) / 6.471346e-06   # -0.9%  e3 < 2.883e-05
        - 0.007818911 * max(0.0, 0.1825048 - Q.sj2_dr) / 0.03087146   # -0.8%  sj2_dr < 0.1825
        - 0.006848499 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # -0.7%  log_sum_pt > 6.894
        + 0.006737677 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +0.7%  e2 < 0.01879
        + 0.006134785 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +0.6%  psi_0p3 > 0.9897
        + 0.005862064 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # +0.6%  log_sum_pt > 7.017
        - 0.004603427 * max(0.0, 0.1623049 - Q.sj3_dr_max) / 0.01012502   # -0.5%  sj3_dr_max < 0.1623
        - 0.003876614 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0223982 - Q.C3) / 0.1512275   # -0.4%  n_dr_0p2_0p4 < 18 and C3 < 0.0224
        + 0.00381835 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, 26.0 - Q.n_dr_0p1_0p2) / 0.02963093   # +0.4%  psi_0p3 > 0.9956 and n_dr_0p1_0p2 < 26
        - 0.003770129 * max(0.0, Q.sum_pt_top40 - 1001.523) / 47.19068   # -0.4%  sum_pt_top40 > 1002
        + 0.00358181 * max(0.0, 0.2125209 - Q.sj3_dr_max) / 0.0267777   # +0.4%  sj3_dr_max < 0.2125
        - 0.003394874 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # -0.3%  n_dr_0p2_0p4 < 18
        + 0.002480537 * max(0.0, 0.0008296372 - Q.lam2) / 0.0002681038   # +0.2%  lam2 < 0.0008296
        - 0.002234047 * max(0.0, Q.sum_pt_top30 - 1052.08) / 20.84425   # -0.2%  sum_pt_top30 > 1052
        - 0.001946503 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.tau21_b2 - 0.2018786) / 1.609082   # -0.2%  n_dr_0p2_0p4 < 18 and tau21_b2 > 0.2019
        + 0.001783655 * max(0.0, 5.13841e-05 - Q.e3) / 1.709557e-05   # +0.2%  e3 < 5.138e-05
        + 0.00177889 * max(0.0, 0.1210264 - Q.sj3_dr_max) / 0.004637661   # +0.2%  sj3_dr_max < 0.121
        - 0.001727792 * max(0.0, Q.psi_0p1 - 0.8747961) / 0.03440015   # -0.2%  psi_0p1 > 0.8748
        + 0.001620295 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9278036 - Q.z_top15_slots) / 0.8344987   # +0.2%  n_dr_0p2_0p4 < 18 and z_top15_slots < 0.9278
        + 0.001435575 * max(0.0, 0.08589404 - Q.sum_z_dr) / 0.02745918   # +0.1%  sum_z_dr < 0.08589
        + 0.001265604 * max(0.0, 0.08589404 - Q.sum_z_dr) * max(0.0, 0.4947602 - Q.z_dr_0_0p05) / 0.0006313374   # +0.1%  sum_z_dr < 0.08589 and z_dr_0_0p05 < 0.4948
        - 0.001050652 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # -0.1%  psi_0p3 > 0.9956
        + 0.001003528 * max(0.0, Q.sum_pt_top2 - 405.0) / 51.45192   # +0.1%  sum_pt_top2 > 405
        + 0.0009362841 * max(0.0, Q.tau21_b2 - 0.7058597) / 0.01262433   # +0.1%  tau21_b2 > 0.7059
        + 0.0009355838 * max(0.0, Q.sum_pt_top40 - 1001.523) * max(0.0, 0.1512157 - Q.sj2_dr) / 1.305424   # +0.1%  sum_pt_top40 > 1002 and sj2_dr < 0.1512
        + 0.0006525139 * max(0.0, Q.sum_pt_top40 - 1001.523) * max(0.0, 0.02652372 - Q.dr_5) / 0.2748573   # +0.1%  sum_pt_top40 > 1002 and dr_5 < 0.02652
        - 0.0006521086 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.1%  log_sum_pt > 7.063
        - 0.0006025964 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, 0.02652372 - Q.dr_5) / 6.727825e-06   # -0.1%  psi_0p3 > 0.9956 and dr_5 < 0.02652
        + 0.0004973623 * max(0.0, 0.0005078411 - Q.soft5_z) / 1.827872e-05   # +0.0%  soft5_z < 0.0005078
        - 0.0003486207 * max(0.0, 0.5297852 - Q.soft5_pt) / 0.01859865   # -0.0%  soft5_pt < 0.5298
        + 0.0003015191 * max(0.0, Q.z_dr_0p05_0p1 - 0.8509215) / 0.003556426   # +0.0%  z_dr_0p05_0p1 > 0.8509
        - 0.0002099631 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, 0.439209 - Q.soft5_pt) / 3.433702e-05   # -0.0%  psi_0p3 > 0.9956 and soft5_pt < 0.4392
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 17.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.00894 * (0.02779898
        + 0.1086826 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +10.9%  log_sum_pt > 6.91
        + 0.1076749 * max(0.0, Q.sum_pt - 972.0419) / 81.34075   # +10.8%  sum_pt > 972
        - 0.09296409 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -9.3%  log_sum_pt < 7.139
        + 0.0585026 * max(0.0, 1225.842 - Q.sum_pt_top40) / 211.117   # +5.9%  sum_pt_top40 < 1226
        - 0.05243376 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # -5.2%  sum_pt_top50 > 934.2
        - 0.04848904 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -4.8%  sum_pt_top50 > 959.1
        + 0.04153019 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # +4.2%  sum_pt_top50 < 1079
        + 0.03786176 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +3.8%  n_particles > 38
        + 0.02657439 * max(0.0, 605.875 - Q.sum_pt_top2) / 245.9717   # +2.7%  sum_pt_top2 < 605.9
        - 0.02359065 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -2.4%  log_sum_pt > 6.989
        - 0.0215806 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -2.2%  sum_pt > 1053
        + 0.01931192 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +1.9%  tau1 < 0.07709
        - 0.01901182 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.002181998 - Q.soft1_z) / 0.01445558   # -1.9%  n_particles > 38 and soft1_z < 0.002182
        - 0.01832949 * max(0.0, Q.tau1 - 0.1219132) / 0.01033631   # -1.8%  tau1 > 0.1219
        - 0.01819017 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # -1.8%  lam2 < 0.001776
        - 0.01474905 * max(0.0, 12.6865 - Q.pt1_dr01) * max(0.0, 0.3845054 - Q.soft2_dr) / 1.613797   # -1.5%  pt1_dr01 < 12.69 and soft2_dr < 0.3845
        + 0.01336154 * max(0.0, 12.6865 - Q.pt1_dr01) * max(0.0, 0.4086381 - Q.soft2_dr0) / 1.741015   # +1.3%  pt1_dr01 < 12.69 and soft2_dr0 < 0.4086
        + 0.01167633 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +1.2%  LHA > 0.3332
        - 0.0111078 * max(0.0, 0.02178815 - Q.tau4) / 0.006603045   # -1.1%  tau4 < 0.02179
        - 0.01058555 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.932092   # -1.1%  n_dr_0p2_0p4 < 13
        + 0.01045231 * max(0.0, 2.117188 - Q.soft5_pt) / 0.8827838   # +1.0%  soft5_pt < 2.117
        + 0.009797775 * max(0.0, 1.521582 - Q.soft1_pt) / 0.9527349   # +1.0%  soft1_pt < 1.522
        - 0.00964399 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 17.94219 - Q.ptdr0_3) / 0.4842322   # -1.0%  z_top30_slots > 0.9342 and ptdr0_3 < 17.94
        - 0.007995276 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) * max(0.0, 10.0 - Q.n_dr_0p05_0p1) / 0.2065903   # -0.8%  z_dr_0_0p05 > 0.8103 and n_dr_0p05_0p1 < 10
        - 0.007838063 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # -0.8%  psi_0p3 > 0.9966
        - 0.007702881 * max(0.0, 0.03457336 - Q.M3) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.0006031662   # -0.8%  M3 < 0.03457 and psi_0p3 > 0.9299
        + 0.007554234 * max(0.0, 0.3572263 - Q.N2) / 0.07385171   # +0.8%  N2 < 0.3572
        + 0.007423383 * max(0.0, Q.z_top30_slots - 0.9564984) / 0.01947087   # +0.7%  z_top30_slots > 0.9565
        + 0.007394929 * max(0.0, 1.0 - Q.z_top50_slots) / 0.007698523   # +0.7%  z_top50_slots < 1
        - 0.006931144 * max(0.0, 3.852812 - Q.D2_b2) / 1.876829   # -0.7%  D2_b2 < 3.853
        - 0.006742082 * max(0.0, 0.0005358203 - Q.lam2) / 0.0001122488   # -0.7%  lam2 < 0.0005358
        + 0.00673829 * max(0.0, Q.z_top30_slots - 0.9341838) / 0.03406818   # +0.7%  z_top30_slots > 0.9342
        + 0.006563111 * max(0.0, 0.006756161 - Q.sum_z_dr2_top3) / 0.003650633   # +0.7%  sum_z_dr2_top3 < 0.006756
        - 0.00654549 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -0.7%  n_dr_0p2_0p4 < 7
        + 0.006275562 * max(0.0, 0.05602756 - Q.C2) / 0.009882647   # +0.6%  C2 < 0.05603
        + 0.005849867 * max(0.0, 0.0006570502 - Q.sum_z_dr2_top5) / 0.0001395474   # +0.6%  sum_z_dr2_top5 < 0.0006571
        + 0.005514039 * max(0.0, 0.002197765 - Q.sum_z_dr2_top15) / 0.0004509993   # +0.6%  sum_z_dr2_top15 < 0.002198
        - 0.005359634 * max(0.0, 29.04688 - Q.pt_11) / 8.607263   # -0.5%  pt_11 < 29.05
        + 0.005064409 * max(0.0, Q.n_dr_0_0p05 - 15.0) * max(0.0, 0.05520935 - Q.eta_0) / 0.1503213   # +0.5%  n_dr_0_0p05 > 15 and eta_0 < 0.05521
        + 0.004820627 * max(0.0, 0.2134181 - Q.sj3_dr_min) / 0.1208971   # +0.5%  sj3_dr_min < 0.2134
        + 0.004665687 * max(0.0, Q.n_particles - 38.0) * max(0.0, Q.tau32 - 0.3293142) / 3.693221   # +0.5%  n_particles > 38 and tau32 > 0.3293
        - 0.004461327 * max(0.0, Q.z_top5 - 0.6551948) * max(0.0, 28.39396 - Q.pt1_dr01) / 0.8332473   # -0.4%  z_top5 > 0.6552 and pt1_dr01 < 28.39
        + 0.004226756 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1178619 - Q.absphi_1) / 0.8192292   # +0.4%  n_particles > 38 and absphi_1 < 0.1179
        + 0.00416709 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.009970338 - Q.zdr_0) / 0.03317992   # +0.4%  n_particles > 38 and zdr_0 < 0.00997
        - 0.00402958 * max(0.0, 0.8495689 - Q.D2_b2) / 0.1691494   # -0.4%  D2_b2 < 0.8496
        - 0.003980761 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 11.6057 - Q.ptdr0_6) / 0.3086732   # -0.4%  z_top30_slots > 0.9342 and ptdr0_6 < 11.61
        - 0.003968866 * max(0.0, 605.875 - Q.sum_pt_top2) * max(0.0, 0.2664362 - Q.soft4_dr) / 24.98241   # -0.4%  sum_pt_top2 < 605.9 and soft4_dr < 0.2664
        + 0.003830108 * max(0.0, 605.875 - Q.sum_pt_top2) * max(0.0, 0.3636138 - Q.soft4_dr0) / 41.76948   # +0.4%  sum_pt_top2 < 605.9 and soft4_dr0 < 0.3636
        - 0.003784028 * max(0.0, 7.0 - Q.n_dr_0p1_0p2) / 0.9740454   # -0.4%  n_dr_0p1_0p2 < 7
        + 0.003736628 * max(0.0, 0.3572263 - Q.N2) * max(0.0, 0.7904923 - Q.pt2_over_pt0) / 0.02349909   # +0.4%  N2 < 0.3572 and pt2_over_pt0 < 0.7905
        - 0.003641739 * max(0.0, 36.65625 - Q.pt_8) / 8.068046   # -0.4%  pt_8 < 36.66
        + 0.003414661 * max(0.0, Q.eccentricity - 0.9252623) / 0.009894939   # +0.3%  eccentricity > 0.9253
        - 0.003369147 * max(0.0, 2.117188 - Q.soft5_pt) * max(0.0, 0.4357228 - Q.max_dr) / 0.06450036   # -0.3%  soft5_pt < 2.117 and max_dr < 0.4357
        - 0.003299214 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.3%  log_sum_pt > 7.063
        + 0.003147439 * max(0.0, 0.2140276 - Q.tau21) / 0.01160104   # +0.3%  tau21 < 0.214
        + 0.003081528 * max(0.0, 2.5 - Q.soft9_pt) * max(0.0, 0.1224381 - Q.dr_11) / 0.03966978   # +0.3%  soft9_pt < 2.5 and dr_11 < 0.1224
        - 0.002987903 * max(0.0, 17.53125 - Q.pt_13) / 3.094926   # -0.3%  pt_13 < 17.53
        - 0.002929269 * max(0.0, Q.zdr_0 - 0.01113024) / 0.002624099   # -0.3%  zdr_0 > 0.01113
        + 0.002815035 * max(0.0, 1.601009 - Q.D2) / 0.21515   # +0.3%  D2 < 1.601
        + 0.002781262 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.pt1_dr01 - 5.351077) / 0.1255531   # +0.3%  z_top30_slots > 0.9342 and pt1_dr01 > 5.351
        + 0.002562424 * max(0.0, 0.002197765 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.9966167) / 5.451807e-07   # +0.3%  sum_z_dr2_top15 < 0.002198 and psi_0p3 > 0.9966
        + 0.002371829 * max(0.0, Q.sum_pt_top30 - 1038.262) / 23.89568   # +0.2%  sum_pt_top30 > 1038
        + 0.002181645 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_2 - 7.740999) / 0.07370506   # +0.2%  z_top30_slots > 0.9342 and ptdr0_2 > 7.741
        + 0.002151481 * max(0.0, 0.006756161 - Q.sum_z_dr2_top3) * max(0.0, Q.eta_0 - -0.004464722) / 3.725323e-05   # +0.2%  sum_z_dr2_top3 < 0.006756 and eta_0 > -0.004465
        - 0.001890492 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) / 0.03630126   # -0.2%  z_dr_0_0p05 > 0.8103
        + 0.001878373 * max(0.0, Q.n_dr_0_0p05 - 15.0) / 2.713461   # +0.2%  n_dr_0_0p05 > 15
        + 0.001877023 * max(0.0, 0.0007894752 - Q.sum_z_dr2_top15) / 9.062109e-05   # +0.2%  sum_z_dr2_top15 < 0.0007895
        + 0.001814394 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.dr12 - 0.003946546) / 0.001521699   # +0.2%  z_top30_slots > 0.9342 and dr12 > 0.003947
        + 0.001775434 * max(0.0, Q.sj3_dr23 - 0.2799759) / 0.0237761   # +0.2%  sj3_dr23 > 0.28
        + 0.001575487 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_4 - 2.381691) / 0.0768104   # +0.2%  z_top30_slots > 0.9342 and ptdr0_4 > 2.382
        - 0.001506672 * max(0.0, 0.2773918 - Q.D2_b2) / 0.01604103   # -0.2%  D2_b2 < 0.2774
        - 0.001278854 * max(0.0, Q.n_dr_0_0p05 - 15.0) * max(0.0, 0.0002882481 - Q.eta_0) / 0.01885152   # -0.1%  n_dr_0_0p05 > 15 and eta_0 < 0.0002882
        + 0.001153947 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # +0.1%  e2 > 0.05557
        - 0.0008503297 * max(0.0, Q.n_pt_above_5 - 46.0) / 0.2726588   # -0.1%  n_pt_above_5 > 46
        - 0.0007874968 * max(0.0, Q.n_dr_0_0p05 - 30.0) / 0.212479   # -0.1%  n_dr_0_0p05 > 30
        + 0.0007073088 * max(0.0, Q.sum_z_dr - 0.1564779) / 0.0006617163   # +0.1%  sum_z_dr > 0.1565
        - 0.0006541766 * max(0.0, 0.2140276 - Q.tau21) * max(0.0, 0.2834473 - Q.soft8_abseta) / 0.002471228   # -0.1%  tau21 < 0.214 and soft8_abseta < 0.2834
        + 0.0006190241 * max(0.0, Q.z_top5 - 0.6551948) * max(0.0, Q.ptdr0_5 - 0.5368514) / 0.0440253   # +0.1%  z_top5 > 0.6552 and ptdr0_5 > 0.5369
        + 0.000606392 * max(0.0, Q.z_top15_slots - 0.9885666) / 0.0003255751   # +0.1%  z_top15_slots > 0.9886
        - 0.0005291662 * max(0.0, Q.N2 - 0.4678622) / 0.001556028   # -0.1%  N2 > 0.4679
        + 0.0004946282 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) * max(0.0, 20.0 - Q.n_real_top20) / 0.01239824   # +0.0%  z_dr_0_0p05 > 0.8103 and n_real_top20 < 20
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 5.641;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.640671 * (0.01857028
        - 0.1035337 * max(0.0, Q.sd_rg - 0.1690338) / 0.03150941   # -10.4%  sd_rg > 0.169
        - 0.07457392 * max(0.0, Q.z_top50_slots - 0.9586536) / 0.03427112   # -7.5%  z_top50_slots > 0.9587
        + 0.07224751 * max(0.0, Q.log_sum_pt - 6.903423) / 0.05662275   # +7.2%  log_sum_pt > 6.903
        + 0.06670126 * max(0.0, Q.sd_rg - 0.1596365) / 0.03553981   # +6.7%  sd_rg > 0.1596
        - 0.05104804 * max(0.0, Q.sum_pt_top50 - 997.0189) / 56.58644   # -5.1%  sum_pt_top50 > 997
        + 0.04985284 * max(0.0, Q.sd_rg - 0.1690338) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1) / 0.01876614   # +5.0%  sd_rg > 0.169 and z_dr_0p05_0p1 < 0.8509
        + 0.04928421 * max(0.0, Q.log_sum_pt - 6.930088) / 0.0397436   # +4.9%  log_sum_pt > 6.93
        - 0.0444064 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -4.4%  sum_pt > 1053
        + 0.03586509 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.sum_z_dr2_top15) / 0.0009227558   # +3.6%  log_sum_pt > 6.903 and sum_z_dr2_top15 < 0.02147
        + 0.03461089 * max(0.0, Q.z_top15_slots - 0.6225177) / 0.2124803   # +3.5%  z_top15_slots > 0.6225
        + 0.03229552 * max(0.0, Q.sum_pt_top50 - 1048.098) / 31.96699   # +3.2%  sum_pt_top50 > 1048
        - 0.03078676 * max(0.0, 0.006615185 - Q.sum_z_dr2_top15) / 0.002375805   # -3.1%  sum_z_dr2_top15 < 0.006615
        + 0.03065371 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # +3.1%  sum_z_dr2_top15 < 0.009962
        - 0.0303437 * max(0.0, Q.sd_rg - 0.1596365) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1) / 0.0209016   # -3.0%  sd_rg > 0.1596 and z_dr_0p05_0p1 < 0.8509
        + 0.02503404 * max(0.0, Q.z_top40_slots - 0.9300465) / 0.0517789   # +2.5%  z_top40_slots > 0.93
        + 0.02195099 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +2.2%  sum_pt_top40 > 1070
        - 0.02014808 * max(0.0, 0.03577037 - Q.sum_z_dr) / 0.004182456   # -2.0%  sum_z_dr < 0.03577
        - 0.01973053 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # -2.0%  sd_rg < 0.3017
        - 0.01715019 * max(0.0, Q.z_top20_slots - 0.8281581) * max(0.0, Q.z_top50_slots - 0.9995789) / 3.023732e-05   # -1.7%  z_top20_slots > 0.8282 and z_top50_slots > 0.9996
        + 0.01526163 * max(0.0, Q.sd_rg - 0.2639816) / 0.00903147   # +1.5%  sd_rg > 0.264
        - 0.01399191 * max(0.0, Q.log_sum_pt - 6.930088) * max(0.0, Q.C2 - 0.06655881) / 0.0004521485   # -1.4%  log_sum_pt > 6.93 and C2 > 0.06656
        - 0.01387537 * max(0.0, Q.sum_pt_top50 - 1078.994) / 24.4796   # -1.4%  sum_pt_top50 > 1079
        - 0.01292148 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # -1.3%  psi_0p3 > 0.9966
        + 0.01290645 * max(0.0, Q.sum_pt_top50 - 997.0189) * max(0.0, Q.C2 - 0.06655881) / 0.5584339   # +1.3%  sum_pt_top50 > 997 and C2 > 0.06656
        - 0.01169908 * max(0.0, Q.sum_pt_top30 - 996.8867) / 38.86433   # -1.2%  sum_pt_top30 > 996.9
        + 0.01071423 * max(0.0, Q.sum_pt_top30 - 852.7492) / 147.1655   # +1.1%  sum_pt_top30 > 852.7
        + 0.009343235 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # +0.9%  LHA < 0.187
        + 0.009217199 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.2002199 - Q.dr_max_012) / 0.007813213   # +0.9%  log_sum_pt > 6.903 and dr_max_012 < 0.2002
        - 0.008503875 * max(0.0, Q.sd_rg - 0.2639816) * max(0.0, 0.5912328 - Q.z_dr_0p05_0p1) / 0.003149033   # -0.9%  sd_rg > 0.264 and z_dr_0p05_0p1 < 0.5912
        + 0.008466438 * max(0.0, 0.02793599 - Q.e2) / 0.005715277   # +0.8%  e2 < 0.02794
        - 0.008390134 * max(0.0, Q.sum_pt - 1085.125) / 25.7173   # -0.8%  sum_pt > 1085
        - 0.008173112 * max(0.0, Q.sum_pt_top40 - 1018.698) / 38.17804   # -0.8%  sum_pt_top40 > 1019
        - 0.007640333 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.1206357 - Q.dr_max_012) / 0.00414901   # -0.8%  log_sum_pt > 6.903 and dr_max_012 < 0.1206
        - 0.00754921 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # -0.8%  log_sum_pt > 7.017
        - 0.005960668 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.6%  log_sum_pt > 7.063
        - 0.005670373 * max(0.0, Q.sum_pt_top15 - 985.0781) / 16.07962   # -0.6%  sum_pt_top15 > 985.1
        + 0.005353527 * max(0.0, Q.sum_pt_top20 - 1064.139) / 11.11947   # +0.5%  sum_pt_top20 > 1064
        - 0.005349038 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.8747961 - Q.psi_0p1) / 0.005466701   # -0.5%  log_sum_pt > 6.903 and psi_0p1 < 0.8748
        + 0.002947606 * max(0.0, Q.log_sum_pt - 6.930088) * max(0.0, Q.C2 - 0.1088881) / 9.583602e-05   # +0.3%  log_sum_pt > 6.93 and C2 > 0.1089
        - 0.002301978 * max(0.0, Q.sum_pt_top50 - 997.0189) * max(0.0, Q.C2 - 0.1088881) / 0.1050644   # -0.2%  sum_pt_top50 > 997 and C2 > 0.1089
        + 0.001404755 * max(0.0, Q.sum_pt_top15 - 1082.548) / 5.972549   # +0.1%  sum_pt_top15 > 1083
        + 0.000751134 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.sum_z_dr2_top15 - 0.009962397) / 1.329441e-05   # +0.1%  log_sum_pt > 7.063 and sum_z_dr2_top15 > 0.009962
        - 0.000732167 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.z_7 - 0.02696541) / 0.04610808   # -0.1%  sum_pt_top20 > 1129 and z_7 > 0.02697
        + 0.0006576819 * max(0.0, Q.sum_pt_top15 - 985.0781) * max(0.0, Q.pt_7 - 41.65625) / 104.7463   # +0.1%  sum_pt_top15 > 985.1 and pt_7 > 41.66
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 1.973;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.972777 * (-0.2259376
        + 0.1107889 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +11.1%  n_particles < 46
        + 0.09790433 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # +9.8%  n_dr_0p1_0p2 < 17
        + 0.08764538 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +8.8%  n_dr_0p2_0p4 < 5
        - 0.07839931 * max(0.0, 0.2989787 - Q.N2) / 0.04313186   # -7.8%  N2 < 0.299
        + 0.06869439 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # +6.9%  n_dr_0p2_0p4 < 8
        + 0.04847906 * max(0.0, 5.727594e-05 - Q.e3) / 2.076531e-05   # +4.8%  e3 < 5.728e-05
        + 0.04636622 * max(0.0, 1.788105 - Q.D2) / 0.2865857   # +4.6%  D2 < 1.788
        - 0.04531835 * max(0.0, 43.0 - Q.n_pt_above_1) / 6.916252   # -4.5%  n_pt_above_1 < 43
        + 0.03592832 * max(0.0, 0.0174227 - Q.tau4) / 0.003628011   # +3.6%  tau4 < 0.01742
        + 0.03474678 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +3.5%  lam2 < 0.0006155
        + 0.03423874 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.psi_0p3 - 0.9966167) / 0.006523952   # +3.4%  n_dr_0p2_0p4 < 8 and psi_0p3 > 0.9966
        + 0.02724597 * max(0.0, 0.2738063 - Q.max_dr) * max(0.0, Q.z_top50_slots - 0.9586536) / 0.0003528057   # +2.7%  max_dr < 0.2738 and z_top50_slots > 0.9587
        - 0.02562705 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03029714 - Q.e2) / 0.006086253   # -2.6%  n_dr_0p2_0p4 < 5 and e2 < 0.0303
        - 0.02511143 * max(0.0, 0.08871546 - Q.z_dr_0p1_0p2) / 0.03070029   # -2.5%  z_dr_0p1_0p2 < 0.08872
        - 0.02442442 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.e2 - 0.01036127) / 0.09655312   # -2.4%  n_particles < 46 and e2 > 0.01036
        + 0.0225562 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 858.8262) / 1128.322   # +2.3%  n_particles < 46 and sum_pt_top40 > 858.8
        - 0.02233288 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 30.51645   # -2.2%  n_dr_0p2_0p4 < 8 and n_dr_0p05_0p1 > 2
        - 0.02162852 * max(0.0, 0.347196 - Q.tau21) / 0.05239131   # -2.2%  tau21 < 0.3472
        - 0.01938624 * max(0.0, 0.2738063 - Q.max_dr) / 0.009268927   # -1.9%  max_dr < 0.2738
        - 0.0193039 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.1221251   # -1.9%  n_particles < 46 and psi_0p3 > 0.9777
        - 0.01587812 * max(0.0, 0.02647293 - Q.z_dr_0p2_0p4) / 0.01168948   # -1.6%  z_dr_0p2_0p4 < 0.02647
        + 0.0153177 * max(0.0, 1.67722 - Q.D2_b2) / 0.5276868   # +1.5%  D2_b2 < 1.677
        + 0.01521318 * max(0.0, 0.347196 - Q.tau21) * max(0.0, 1085.125 - Q.sum_pt) / 3.187528   # +1.5%  tau21 < 0.3472 and sum_pt < 1085
        + 0.01354336 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.eccentricity - 0.8680812) / 3.480194e-05   # +1.4%  psi_0p3 > 0.998 and eccentricity > 0.8681
        - 0.01213211 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0) / 3.186294   # -1.2%  n_dr_0p2_0p4 < 5 and n_dr_0p1_0p2 > 9
        + 0.01211383 * max(0.0, 1.67722 - Q.D2_b2) * max(0.0, Q.psi_0p3 - 0.9989733) / 0.0002383665   # +1.2%  D2_b2 < 1.677 and psi_0p3 > 0.999
        - 0.01037342 * max(0.0, 1.788105 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9995915) / 4.53478e-05   # -1.0%  D2 < 1.788 and psi_0p3 > 0.9996
        - 0.005912684 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) * max(0.0, 1013.042 - Q.sum_pt_top40) / 41.6591   # -0.6%  n_dr_0p1_0p2 < 10 and sum_pt_top40 < 1013
        - 0.002983669 * max(0.0, 1.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.psi_0p3 - 0.9993087) / 3.893503e-05   # -0.3%  n_dr_0p2_0p4 < 1 and psi_0p3 > 0.9993
        - 0.0004055829 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.006755326   # -0.0%  n_dr_0p1_0p2 < 17 and psi_0p3 > 0.9974
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 9.435;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.434601 * (0.1778351
        + 0.103123 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # +10.3%  sum_z_dr2_top15 < 0.009962
        - 0.07120673 * max(0.0, 0.05660088 - Q.sum_z_dr) / 0.01080382   # -7.1%  sum_z_dr < 0.0566
        - 0.06861348 * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) / 0.00326329   # -6.9%  sum_z_dr2_top15 < 0.007888
        + 0.05535299 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # +5.5%  sum_z_dr > 0.0975
        + 0.05042452 * max(0.0, Q.sj2_dr - 0.1411617) / 0.06721719   # +5.0%  sj2_dr > 0.1412
        - 0.04876502 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -4.9%  e2 < 0.04359
        - 0.04820212 * max(0.0, Q.n_particles - 26.0) / 20.24921   # -4.8%  n_particles > 26
        + 0.03981329 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # +4.0%  LHA < 0.2454
        - 0.03800819 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # -3.8%  LHA > 0.3332
        - 0.03545065 * max(0.0, Q.sj2_dr - 0.1825048) / 0.04110551   # -3.5%  sj2_dr > 0.1825
        + 0.0348521 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +3.5%  n_dr_0p2_0p4 < 18
        - 0.03403892 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # -3.4%  sum_z_dr < 0.0975
        - 0.03267429 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # -3.3%  z_dr_0p2_0p4 < 0.06849
        + 0.03044637 * max(0.0, Q.psi_0p2 - 0.8706159) / 0.08984765   # +3.0%  psi_0p2 > 0.8706
        - 0.02746763 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # -2.7%  e3 < 0.0005178
        + 0.01992575 * max(0.0, 0.09749958 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.9973959) / 3.542375e-05   # +2.0%  sum_z_dr < 0.0975 and psi_0p3 > 0.9974
        + 0.01950651 * max(0.0, 0.03263075 - Q.e2) / 0.008034085   # +2.0%  e2 < 0.03263
        + 0.01903162 * max(0.0, Q.soft9_pt - 1.458984) / 1.15228   # +1.9%  soft9_pt > 1.459
        - 0.0181712 * max(0.0, 0.005788041 - Q.sum_z_dr2_top15) / 0.001877672   # -1.8%  sum_z_dr2_top15 < 0.005788
        - 0.0153993 * max(0.0, Q.soft9_z - 0.001545795) / 0.001021437   # -1.5%  soft9_z > 0.001546
        + 0.01388713 * max(0.0, 0.07374472 - Q.sum_z_dr) / 0.01915593   # +1.4%  sum_z_dr < 0.07374
        + 0.0136056 * max(0.0, 1011.524 - Q.sum_pt_top30) / 51.02618   # +1.4%  sum_pt_top30 < 1012
        - 0.01336852 * max(0.0, Q.sum_z_dr - 0.01525414) / 0.05449303   # -1.3%  sum_z_dr > 0.01525
        - 0.01115749 * max(0.0, 1038.855 - Q.sum_pt_top50) / 38.10896   # -1.1%  sum_pt_top50 < 1039
        - 0.009841626 * max(0.0, 1041.263 - Q.sum_pt_top40) / 49.16087   # -1.0%  sum_pt_top40 < 1041
        - 0.009139185 * max(0.0, 0.006876086 - Q.sum_z_dr2_top10) / 0.002892387   # -0.9%  sum_z_dr2_top10 < 0.006876
        - 0.008746367 * max(0.0, 0.05660088 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.9973959) / 8.756067e-06   # -0.9%  sum_z_dr < 0.0566 and psi_0p3 > 0.9974
        - 0.008395791 * max(0.0, Q.sum_z_dr2_top15 - 0.004855289) / 0.003942732   # -0.8%  sum_z_dr2_top15 > 0.004855
        - 0.007810623 * max(0.0, Q.n_particles - 26.0) * max(0.0, Q.soft1_pt - 0.4909668) / 9.262955   # -0.8%  n_particles > 26 and soft1_pt > 0.491
        + 0.006826062 * max(0.0, 0.001163277 - Q.lam2) / 0.0004869716   # +0.7%  lam2 < 0.001163
        - 0.006361693 * max(0.0, 0.05660088 - Q.sum_z_dr) * max(0.0, 1007.44 - Q.sum_pt_top40) / 0.1929141   # -0.6%  sum_z_dr < 0.0566 and sum_pt_top40 < 1007
        - 0.00627174 * max(0.0, Q.C2 - 0.06116874) / 0.01742347   # -0.6%  C2 > 0.06117
        - 0.005958888 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -0.6%  psi_0p1 > 0.8976
        + 0.005835869 * max(0.0, Q.n_dr_0_0p05 - 14.0) / 3.092148   # +0.6%  n_dr_0_0p05 > 14
        + 0.005712477 * max(0.0, 0.04748396 - Q.z_dr_0p1_0p2) * max(0.0, 0.01003298 - Q.zdr_1) / 8.822948e-05   # +0.6%  z_dr_0p1_0p2 < 0.04748 and zdr_1 < 0.01003
        - 0.005499425 * max(0.0, Q.soft1_pt - 1.521582) / 0.1002024   # -0.5%  soft1_pt > 1.522
        - 0.00529332 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # -0.5%  e2 < 0.01879
        + 0.004823691 * max(0.0, 0.002197765 - Q.sum_z_dr2_top15) / 0.0004509993   # +0.5%  sum_z_dr2_top15 < 0.002198
        + 0.004741405 * max(0.0, Q.soft1_pt - 1.091797) / 0.1522878   # +0.5%  soft1_pt > 1.092
        - 0.00444751 * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 4.239657   # -0.4%  n_dr_0p1_0p2 > 11
        - 0.004311451 * max(0.0, Q.n_pt_above_1 - 32.0) / 11.7996   # -0.4%  n_pt_above_1 > 32
        - 0.004108655 * max(0.0, 3.355186 - Q.pt_entropy) * max(0.0, Q.zdr_0 - 0.0008718296) / 0.004720646   # -0.4%  pt_entropy < 3.355 and zdr_0 > 0.0008718
        + 0.004042616 * max(0.0, Q.n_particles - 26.0) * max(0.0, Q.tau21 - 0.1295048) / 7.108916   # +0.4%  n_particles > 26 and tau21 > 0.1295
        - 0.003293512 * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) * max(0.0, Q.soft9_z - 0.0007402181) / 5.306836e-05   # -0.3%  z_dr_0p2_0p4 < 0.0518 and soft9_z > 0.0007402
        - 0.003015024 * max(0.0, Q.sum_z_dr2_top15 - 0.004855289) * max(0.0, 0.4090302 - Q.sj3_pairmin_over_m) / 0.0003619533   # -0.3%  sum_z_dr2_top15 > 0.004855 and sj3_pairmin_over_m < 0.409
        + 0.002880638 * max(0.0, Q.C2 - 0.0977156) / 0.006150774   # +0.3%  C2 > 0.09772
        + 0.002709555 * max(0.0, Q.soft1_pt - 2.275391) / 0.04593653   # +0.3%  soft1_pt > 2.275
        - 0.002587344 * max(0.0, 0.004169954 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 8.516876e-07   # -0.3%  sum_z_dr2_top15 < 0.00417 and psi_0p3 > 0.9974
        - 0.00192757 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # -0.2%  e2 > 0.05557
        + 0.001260864 * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) * max(0.0, Q.sum_pt_top40 - 1095.686) / 0.09612434   # +0.1%  sum_z_dr2_top15 < 0.007888 and sum_pt_top40 > 1096
        + 0.001191242 * max(0.0, Q.n_particles - 26.0) * max(0.0, Q.lam2 - 0.003687605) / 0.01132634   # +0.1%  n_particles > 26 and lam2 > 0.003688
        - 0.0004734541 * max(0.0, Q.soft1_pt - 2.275391) * max(0.0, 0.4021783 - Q.max_dr) / 0.003815609   # -0.0%  soft1_pt > 2.275 and max_dr < 0.4022
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 13.8;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.79811 * (-0.3334822
        + 0.3098423 * max(0.0, 0.1564779 - Q.sum_z_dr) / 0.08789501   # +31.0%  sum_z_dr < 0.1565
        + 0.07604402 * max(0.0, Q.sum_z_dr - 0.02783745) / 0.04381821   # +7.6%  sum_z_dr > 0.02784
        + 0.05415425 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # +5.4%  log_sum_pt < 7.139
        - 0.04754614 * max(0.0, 0.08068193 - Q.sum_z_dr) / 0.02368455   # -4.8%  sum_z_dr < 0.08068
        + 0.04733163 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +4.7%  n_particles < 64
        - 0.04505765 * max(0.0, Q.LHA - 0.1632346) / 0.1084458   # -4.5%  LHA > 0.1632
        - 0.03929676 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -3.9%  tau1 < 0.1073
        + 0.032443 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # +3.2%  e3 < 0.0005178
        - 0.02302413 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -2.3%  sum_pt < 1002
        + 0.02128572 * max(0.0, Q.sum_z_dr2_top10 - 0.005337976) / 0.003351803   # +2.1%  sum_z_dr2_top10 > 0.005338
        + 0.02092553 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # +2.1%  sum_pt < 1085
        + 0.0184971 * max(0.0, Q.LHA - 0.302389) / 0.02201444   # +1.8%  LHA > 0.3024
        - 0.01695788 * max(0.0, 6.893714 - Q.log_sum_pt) / 0.0132811   # -1.7%  log_sum_pt < 6.894
        - 0.01677985 * max(0.0, Q.sum_z_dr2_top10 - 0.00130722) / 0.0057341   # -1.7%  sum_z_dr2_top10 > 0.001307
        + 0.01671476 * Q.z_top3_slots / 0.4446698   # +1.7%  z_top3_slots
        - 0.01439688 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 7.684973e-07   # -1.4%  sum_pt < 1002 and e4 < 5.851e-08
        + 0.01266643 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # +1.3%  log_sum_pt < 6.903
        - 0.01213095 * max(0.0, 58.0 - Q.n_pt_above_1) / 17.32351   # -1.2%  n_pt_above_1 < 58
        + 0.01093213 * max(0.0, Q.n_dr_0p1_0p2 - 8.0) / 5.88681   # +1.1%  n_dr_0p1_0p2 > 8
        - 0.01080872 * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.02836816   # -1.1%  z_dr_0p2_0p4 < 0.0518
        + 0.009269487 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 3.38926e-06   # +0.9%  sum_pt < 1085 and e4 < 5.851e-08
        - 0.009126533 * max(0.0, Q.sum_z_dr2_top10 - 0.008956554) / 0.002278745   # -0.9%  sum_z_dr2_top10 > 0.008957
        - 0.00857519 * max(0.0, Q.z_top30_slots - 0.9460751) / 0.02601016   # -0.9%  z_top30_slots > 0.9461
        + 0.008083479 * max(0.0, Q.z_dr_0_0p05 - 0.181086) / 0.3722859   # +0.8%  z_dr_0_0p05 > 0.1811
        + 0.006844778 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +0.7%  n_dr_0p2_0p4 < 11
        - 0.006720793 * max(0.0, Q.z_top40_slots - 0.9457345) / 0.03799209   # -0.7%  z_top40_slots > 0.9457
        - 0.006204538 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # -0.6%  sum_pt_top50 < 1079
        - 0.006155214 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 0.07996447 - Q.C2) / 0.5514341   # -0.6%  n_particles < 64 and C2 < 0.07996
        + 0.006154336 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # +0.6%  sum_pt_top50 < 959.1
        - 0.005408593 * max(0.0, 0.985099 - Q.z_top50_slots) / 0.003556075   # -0.5%  z_top50_slots < 0.9851
        + 0.005254324 * max(0.0, Q.n_dr_0p05_0p1 - 6.0) / 6.42277   # +0.5%  n_dr_0p05_0p1 > 6
        + 0.005081866 * max(0.0, 934.2416 - Q.sum_pt_top50) / 6.932864   # +0.5%  sum_pt_top50 < 934.2
        - 0.004699005 * max(0.0, Q.LHA - 0.4331369) / 0.001006142   # -0.5%  LHA > 0.4331
        - 0.004549647 * max(0.0, Q.sum_pt_top15 - 967.7705) / 19.94058   # -0.5%  sum_pt_top15 > 967.8
        - 0.004437593 * max(0.0, 0.0005178279 - Q.e3) * max(0.0, Q.max_dr - 0.1939977) / 6.684346e-05   # -0.4%  e3 < 0.0005178 and max_dr > 0.194
        - 0.004412462 * max(0.0, 934.2416 - Q.sum_pt_top50) * max(0.0, Q.e4 - 3.53e-10) / 1.423498e-07   # -0.4%  sum_pt_top50 < 934.2 and e4 > 3.53e-10
        + 0.004400041 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.4%  psi_0p3 > 0.9974
        + 0.003805291 * max(0.0, Q.z_dr_0_0p05 - 0.7128619) / 0.07526795   # +0.4%  z_dr_0_0p05 > 0.7129
        + 0.003767269 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # +0.4%  z_dr_0_0p05 > 0.8789
        + 0.003721719 * max(0.0, Q.M2 - 0.05233747) / 0.01772949   # +0.4%  M2 > 0.05234
        - 0.003396188 * max(0.0, 8.0 - Q.n_dr_0_0p05) / 1.708339   # -0.3%  n_dr_0_0p05 < 8
        - 0.003294831 * max(0.0, Q.z_top20_slots - 0.9358352) / 0.01535337   # -0.3%  z_top20_slots > 0.9358
        + 0.003000002 * max(0.0, Q.sum_pt_top10 - 822.975) / 44.35907   # +0.3%  sum_pt_top10 > 823
        + 0.00284236 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +0.3%  tau4 < 0.01627
        - 0.002710951 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.absphi_1 - 0.02227783) / 0.4183469   # -0.3%  sum_pt < 1002 and absphi_1 > 0.02228
        - 0.002594912 * max(0.0, 0.0007894752 - Q.sum_z_dr2_top15) / 9.062109e-05   # -0.3%  sum_z_dr2_top15 < 0.0007895
        + 0.002510538 * max(0.0, Q.zdr_0 - 0.005911134) / 0.00509486   # +0.3%  zdr_0 > 0.005911
        + 0.002277908 * max(0.0, Q.sum_pt_top5 - 430.75) / 176.7518   # +0.2%  sum_pt_top5 > 430.8
        + 0.002266808 * max(0.0, 907.9372 - Q.sum_pt) / 3.961002   # +0.2%  sum_pt < 907.9
        - 0.002229076 * max(0.0, 0.05954375 - Q.z_dr_0p05_0p1) / 0.01428263   # -0.2%  z_dr_0p05_0p1 < 0.05954
        + 0.001412328 * max(0.0, 911.9328 - Q.sum_pt_top30) / 15.55609   # +0.1%  sum_pt_top30 < 911.9
        - 0.001301666 * max(0.0, 907.9372 - Q.sum_pt) * max(0.0, 0.2535773 - Q.dr_11) / 0.6391318   # -0.1%  sum_pt < 907.9 and dr_11 < 0.2536
        + 0.001270779 * max(0.0, 0.02476077 - Q.z_8) / 0.002032134   # +0.1%  z_8 < 0.02476
        + 0.001171211 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, 0.199671 - Q.dr_11) / 1.707103   # +0.1%  sum_pt < 1002 and dr_11 < 0.1997
        - 0.001169489 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.ptdr0_12 - 2.024972) / 14.44148   # -0.1%  sum_pt < 1002 and ptdr0_12 > 2.025
        - 0.001023358 * max(0.0, 0.1564779 - Q.sum_z_dr) * max(0.0, 0.9638082 - Q.psi_0p3) / 0.0001348634   # -0.1%  sum_z_dr < 0.1565 and psi_0p3 < 0.9638
        + 0.0006850076 * max(0.0, Q.sum_pt_top15 - 1003.329) / 12.91783   # +0.1%  sum_pt_top15 > 1003
        + 0.0004253612 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, 0.03626613 - Q.soft6_dr) / 0.02518702   # +0.0%  sum_pt < 1002 and soft6_dr < 0.03627
        - 0.0003870461 * max(0.0, 708.7945 - Q.sum_pt_top15) / 11.69336   # -0.0%  sum_pt_top15 < 708.8
        - 0.0002966202 * max(0.0, 934.2416 - Q.sum_pt_top50) * max(0.0, 0.03626613 - Q.soft6_dr) / 0.009542907   # -0.0%  sum_pt_top50 < 934.2 and soft6_dr < 0.03627
        + 0.0002255614 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.sj3_dr23 - 0.1834565) / 1.474397   # +0.0%  sum_pt < 1002 and sj3_dr23 > 0.1835
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 14.91;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.90796 * (0.01532604
        - 0.2562699 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # -25.6%  sum_z_dr < 0.0975
        + 0.1632578 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +16.3%  LHA < 0.3332
        + 0.0441921 * max(0.0, 0.006427167 - Q.lam2) / 0.005146703   # +4.4%  lam2 < 0.006427
        + 0.03764529 * max(0.0, 0.05048381 - Q.sum_z_dr) / 0.008533025   # +3.8%  sum_z_dr < 0.05048
        + 0.02752005 * max(0.0, 0.04755309 - Q.e2) / 0.01877457   # +2.8%  e2 < 0.04755
        - 0.02737212 * max(0.0, 0.2284021 - Q.LHA) / 0.02716657   # -2.7%  LHA < 0.2284
        - 0.02617372 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # -2.6%  sum_z_dr2_top15 < 0.009962
        + 0.02571253 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # +2.6%  e2 < 0.04359
        + 0.0231234 * max(0.0, Q.z_dr_0_0p05 - 0.3289237) / 0.280304   # +2.3%  z_dr_0_0p05 > 0.3289
        - 0.022411 * max(0.0, 0.7108211 - Q.z_dr_0p05_0p1) / 0.4588085   # -2.2%  z_dr_0p05_0p1 < 0.7108
        + 0.02148943 * max(0.0, Q.e3 - 5.13841e-05) / 6.821576e-05   # +2.1%  e3 > 5.138e-05
        - 0.02126409 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -2.1%  psi_0p2 > 0.9087
        + 0.02070231 * max(0.0, 0.00727763 - Q.sum_z_dr2_top15) / 0.00282147   # +2.1%  sum_z_dr2_top15 < 0.007278
        + 0.01965954 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # +2.0%  tau2 < 0.0795
        - 0.01920405 * max(0.0, Q.e3 - 0.0001086251) / 5.31284e-05   # -1.9%  e3 > 0.0001086
        + 0.01892126 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +1.9%  e3 < 0.0003372
        - 0.01830487 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # -1.8%  LHA < 0.3024
        - 0.01668872 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, 7.017258 - Q.log_sum_pt) / 2.140029e-05   # -1.7%  e3 < 0.0003372 and log_sum_pt < 7.017
        - 0.01639774 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # -1.6%  sum_pt > 907.9
        + 0.0124776 * max(0.0, 0.08589404 - Q.sum_z_dr) / 0.02745918   # +1.2%  sum_z_dr < 0.08589
        + 0.01083335 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # +1.1%  tau1 < 0.05445
        + 0.009497926 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) / 0.003638607   # +0.9%  sum_z_dr2_top5 < 0.007164
        + 0.009231521 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # +0.9%  z_dr_0p2_0p4 < 0.06849
        - 0.009167295 * max(0.0, 0.2091025 - Q.LHA) / 0.02095782   # -0.9%  LHA < 0.2091
        - 0.009030285 * max(0.0, 0.2126484 - Q.z_dr_0p05_0p1) / 0.09386891   # -0.9%  z_dr_0p05_0p1 < 0.2126
        + 0.008003451 * max(0.0, 0.003270031 - Q.sum_z_dr2_top15) / 0.0007969965   # +0.8%  sum_z_dr2_top15 < 0.00327
        + 0.00741288 * max(0.0, Q.z_top20_slots - 0.7818983) / 0.116358   # +0.7%  z_top20_slots > 0.7819
        - 0.00673356 * max(0.0, 1003.329 - Q.sum_pt_top15) / 147.1939   # -0.7%  sum_pt_top15 < 1003
        + 0.006179475 * max(0.0, 0.3676068 - Q.sj2_zsoft) / 0.1397909   # +0.6%  sj2_zsoft < 0.3676
        - 0.005228016 * max(0.0, 0.002270363 - Q.sum_z_dr2_top5) / 0.0007634099   # -0.5%  sum_z_dr2_top5 < 0.00227
        + 0.005199144 * max(0.0, 0.08589404 - Q.sum_z_dr) * max(0.0, 1002.379 - Q.sum_pt) / 0.3684202   # +0.5%  sum_z_dr < 0.08589 and sum_pt < 1002
        + 0.005195873 * max(0.0, Q.M2 - 0.03112708) / 0.03490112   # +0.5%  M2 > 0.03113
        - 0.004636927 * max(0.0, 0.003270031 - Q.sum_z_dr2_top15) * max(0.0, Q.sum_pt_top40 - 858.8262) / 0.1598282   # -0.5%  sum_z_dr2_top15 < 0.00327 and sum_pt_top40 > 858.8
        - 0.004587539 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # -0.5%  e2 < 0.02516
        + 0.004260225 * max(0.0, 0.003270031 - Q.sum_z_dr2_top15) * max(0.0, Q.z_top15_slots - 0.7733683) / 9.822416e-05   # +0.4%  sum_z_dr2_top15 < 0.00327 and z_top15_slots > 0.7734
        - 0.00407086 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) / 1.539217   # -0.4%  n_dr_0p2_0p4 > 15
        - 0.003921021 * max(0.0, 0.3676068 - Q.sj2_zsoft) * max(0.0, 0.008824206 - Q.zdr_1) / 0.0007676544   # -0.4%  sj2_zsoft < 0.3676 and zdr_1 < 0.008824
        + 0.003557856 * max(0.0, Q.sum_pt_top50 - 1078.994) / 24.4796   # +0.4%  sum_pt_top50 > 1079
        - 0.0034441 * max(0.0, 0.04649465 - Q.dr_0) / 0.01415326   # -0.3%  dr_0 < 0.04649
        + 0.003150979 * max(0.0, Q.psi_0p2 - 0.9087063) * max(0.0, 0.1512157 - Q.sj2_dr) / 0.001524835   # +0.3%  psi_0p2 > 0.9087 and sj2_dr < 0.1512
        - 0.003112571 * max(0.0, 0.09749958 - Q.sum_z_dr) * max(0.0, Q.eccentricity - 0.6414784) / 0.004627153   # -0.3%  sum_z_dr < 0.0975 and eccentricity > 0.6415
        + 0.002918549 * max(0.0, Q.n_dr_0p1_0p2 - 15.0) * max(0.0, Q.psi_0p3 - 0.9853273) / 0.01929542   # +0.3%  n_dr_0p1_0p2 > 15 and psi_0p3 > 0.9853
        + 0.002861517 * max(0.0, 0.09749958 - Q.sum_z_dr) * max(0.0, 0.9638082 - Q.psi_0p3) / 1.616441e-05   # +0.3%  sum_z_dr < 0.0975 and psi_0p3 < 0.9638
        - 0.002824695 * max(0.0, 0.3332345 - Q.LHA) * max(0.0, 0.9638082 - Q.psi_0p3) / 5.130459e-05   # -0.3%  LHA < 0.3332 and psi_0p3 < 0.9638
        - 0.00270136 * max(0.0, 0.08589404 - Q.sum_z_dr) * max(0.0, 0.1134943 - Q.M2) / 0.001132661   # -0.3%  sum_z_dr < 0.08589 and M2 < 0.1135
        - 0.002609158 * max(0.0, 0.001319197 - Q.sum_z_dr2_top15) / 0.0002099631   # -0.3%  sum_z_dr2_top15 < 0.001319
        + 0.002270915 * max(0.0, 0.0004005745 - Q.sum_z_dr2_top5) / 6.9335e-05   # +0.2%  sum_z_dr2_top5 < 0.0004006
        - 0.00221492 * max(0.0, Q.n_dr_0p1_0p2 - 15.0) / 2.705171   # -0.2%  n_dr_0p1_0p2 > 15
        + 0.002163287 * max(0.0, 0.1825048 - Q.sj2_dr) / 0.03087146   # +0.2%  sj2_dr < 0.1825
        + 0.001858111 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.03720487   # +0.2%  z_dr_0p1_0p2 > 0.334
        + 0.00139768 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, 0.07952881 - Q.eta_0) / 0.1317641   # +0.1%  n_dr_0p2_0p4 > 15 and eta_0 < 0.07953
        + 0.001372676 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # +0.1%  e2 > 0.05557
        + 0.001082486 * max(0.0, 0.05444509 - Q.tau1) * max(0.0, 951.1375 - Q.sum_pt_top15) / 0.2413807   # +0.1%  tau1 < 0.05445 and sum_pt_top15 < 951.1
        + 0.001034435 * max(0.0, Q.n_dr_0p1_0p2 - 26.0) / 0.7746672   # +0.1%  n_dr_0p1_0p2 > 26
        - 0.001018682 * max(0.0, 0.05444509 - Q.tau1) * max(0.0, 972.0419 - Q.sum_pt) / 0.0682246   # -0.1%  tau1 < 0.05445 and sum_pt < 972
        + 0.0009722743 * max(0.0, 0.04575675 - Q.sj2_zsoft) / 0.00226832   # +0.1%  sj2_zsoft < 0.04576
        - 0.0009310423 * max(0.0, Q.tau21_b2 - 0.7058597) / 0.01262433   # -0.1%  tau21_b2 > 0.7059
        + 0.0008481839 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) * max(0.0, 1.0 - Q.n_dr_0p4_up) / 0.03141518   # +0.1%  z_dr_0p1_0p2 > 0.334 and n_dr_0p4_up < 1
        - 0.0008400392 * max(0.0, Q.sj3_dr23 - 0.3357015) / 0.01270175   # -0.1%  sj3_dr23 > 0.3357
        + 0.0007677703 * max(0.0, 0.2126484 - Q.z_dr_0p05_0p1) * max(0.0, Q.zdr_0 - 0.01113024) / 0.0001762331   # +0.1%  z_dr_0p05_0p1 < 0.2126 and zdr_0 > 0.01113
        + 0.0005809089 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.psi_0p3 - 0.9896594) / 3.001287e-06   # +0.1%  e2 > 0.05557 and psi_0p3 > 0.9897
        - 0.0005150805 * max(0.0, Q.n_dr_0p1_0p2 - 26.0) * max(0.0, Q.sum_pt - 986.0565) / 45.26133   # -0.1%  n_dr_0p1_0p2 > 26 and sum_pt > 986.1
        - 0.0004994262 * max(0.0, 0.04358622 - Q.e2) * max(0.0, 0.9638082 - Q.psi_0p3) / 1.224228e-05   # -0.0%  e2 < 0.04359 and psi_0p3 < 0.9638
        + 0.0004743978 * max(0.0, Q.psi_0p2 - 0.9087063) * max(0.0, 0.9704436 - Q.z_top50_slots) / 3.390458e-05   # +0.0%  psi_0p2 > 0.9087 and z_top50_slots < 0.9704
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 37.3;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 37.30495 * (-0.03076682
        + 0.08485572 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # +8.5%  sum_z_dr < 0.0975
        - 0.08129225 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -8.1%  tau1 < 0.1073
        - 0.06503031 * max(0.0, 0.08068193 - Q.sum_z_dr) / 0.02368455   # -6.5%  sum_z_dr < 0.08068
        - 0.05869246 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # -5.9%  LHA < 0.3332
        + 0.05730333 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # +5.7%  LHA < 0.3024
        + 0.0557748 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +5.6%  psi_0p3 > 0.9974
        + 0.04836707 * max(0.0, 0.1219132 - Q.tau1) / 0.04578087   # +4.8%  tau1 < 0.1219
        + 0.04720757 * max(0.0, 0.1072713 - Q.tau1) * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) / 0.008148309   # +4.7%  tau1 < 0.1073 and z_dr_0p1_0p2 < 0.2865
        + 0.03717806 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063   # +3.7%  n_dr_0p2_0p4 < 21
        + 0.03292933 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.0001707165   # +3.3%  tau21_b2 < 0.2352 and sum_z_dr2_top15 < 0.009962
        - 0.03062858 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.006808102 - Q.mean_phi2) / 3.800578e-06   # -3.1%  psi_0p3 > 0.9974 and mean_phi2 < 0.006808
        - 0.02679268 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3) / 0.0005360215   # -2.7%  n_dr_0p2_0p4 < 21 and e3 < 7.876e-05
        - 0.02512422 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.006802603 - Q.mean_eta2) / 3.803697e-06   # -2.5%  psi_0p3 > 0.9974 and mean_eta2 < 0.006803
        - 0.02370094 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) / 0.293679   # -2.4%  n_dr_0p1_0p2 > 33
        - 0.02305862 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) / 8.087416e-05   # -2.3%  tau21_b2 < 0.2352 and sum_z_dr2_top15 < 0.007888
        + 0.02227123 * max(0.0, 6.567534e-05 - Q.e3) / 2.643172e-05   # +2.2%  e3 < 6.568e-05
        - 0.02158724 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # -2.2%  tau21_b2 < 0.2352
        - 0.02034775 * max(0.0, 0.3332345 - Q.LHA) * max(0.0, 0.250441 - Q.z_dr_0p1_0p2) / 0.01705837   # -2.0%  LHA < 0.3332 and z_dr_0p1_0p2 < 0.2504
        - 0.01844416 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.sum_z_dr2_top10 - 0.01414829) / 3.726039e-07   # -1.8%  psi_0p3 > 0.9974 and sum_z_dr2_top10 > 0.01415
        - 0.01609724 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -1.6%  n_dr_0p2_0p4 < 15
        + 0.01565739 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1512157) / 0.00318489   # +1.6%  tau21_b2 < 0.2352 and sj2_dr > 0.1512
        + 0.01512488 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.z_top50_slots - 0.9704436) / 0.001432928   # +1.5%  tau21_b2 < 0.2352 and z_top50_slots > 0.9704
        + 0.01403537 * max(0.0, 0.03680582 - Q.e2) / 0.01052402   # +1.4%  e2 < 0.03681
        - 0.01390337 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 7.062574 - Q.log_sum_pt) / 0.0001122434   # -1.4%  psi_0p3 > 0.9974 and log_sum_pt < 7.063
        - 0.01266625 * max(0.0, 0.04755309 - Q.e2) / 0.01877457   # -1.3%  e2 < 0.04755
        + 0.011414 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 3.376709e-05 - Q.e3) / 0.0001233727   # +1.1%  n_dr_0p2_0p4 < 21 and e3 < 3.377e-05
        - 0.009932134 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1245.697 - Q.sum_pt_top50) / 11.02846   # -1.0%  tau21_b2 < 0.2352 and sum_pt_top50 < 1246
        - 0.009331034 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.001696484   # -0.9%  tau21_b2 < 0.2352 and sj2_dr > 0.1938
        - 0.009136205 * max(0.0, 5.727594e-05 - Q.e3) / 2.076531e-05   # -0.9%  e3 < 5.728e-05
        + 0.009120227 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.003687605 - Q.lam2) / 2.993211e-06   # +0.9%  psi_0p3 > 0.9974 and lam2 < 0.003688
        + 0.007756134 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.psi_0p3 - 0.9956185) / 0.02161643   # +0.8%  n_dr_0p2_0p4 < 15 and psi_0p3 > 0.9956
        - 0.006813719 * max(0.0, 3.376709e-05 - Q.e3) / 8.295151e-06   # -0.7%  e3 < 3.377e-05
        - 0.005652515 * max(0.0, 0.03197846 - Q.tau2) / 0.009015905   # -0.6%  tau2 < 0.03198
        - 0.005421652 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.541744 - Q.tau21_b2) / 0.0003029439   # -0.5%  psi_0p3 > 0.9974 and tau21_b2 < 0.5417
        + 0.005170169 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.1134943 - Q.M2) / 0.3618294   # +0.5%  n_dr_0p2_0p4 < 15 and M2 < 0.1135
        + 0.004634145 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.005203178 - Q.mean_phi2) / 2.479991e-06   # +0.5%  psi_0p3 > 0.9974 and mean_phi2 < 0.005203
        - 0.004505195 * max(0.0, 0.03263075 - Q.e2) / 0.008034085   # -0.5%  e2 < 0.03263
        + 0.004132238 * max(0.0, 0.05557149 - Q.e2) / 0.02580562   # +0.4%  e2 < 0.05557
        - 0.00409145 * max(0.0, Q.psi_0p1 - 0.8509811) / 0.04568859   # -0.4%  psi_0p1 > 0.851
        + 0.003785006 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.4823776 - Q.tau21_b2) / 2.786788   # +0.4%  n_dr_0p2_0p4 < 21 and tau21_b2 < 0.4824
        - 0.00331673 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.985099 - Q.z_top50_slots) / 2.023206e-06   # -0.3%  psi_0p3 > 0.9974 and z_top50_slots < 0.9851
        - 0.002595432 * max(0.0, 0.04362872 - Q.sum_z_dr) / 0.006325401   # -0.3%  sum_z_dr < 0.04363
        + 0.00240081 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.06975954 - Q.M2) / 0.001355005   # +0.2%  tau21_b2 < 0.2352 and M2 < 0.06976
        - 0.001741242 * max(0.0, 0.04755309 - Q.e2) * max(0.0, 0.2352054 - Q.tau21_b2) / 0.0005663274   # -0.2%  e2 < 0.04755 and tau21_b2 < 0.2352
        - 0.001673995 * max(0.0, 0.347196 - Q.tau21) / 0.05239131   # -0.2%  tau21 < 0.3472
        - 0.001593133 * max(0.0, 0.1219132 - Q.tau1) * max(0.0, Q.z_dr_0p05_0p1 - 0.4026646) / 0.001603182   # -0.2%  tau1 < 0.1219 and z_dr_0p05_0p1 > 0.4027
        + 0.001555681 * max(0.0, 0.347196 - Q.tau21) * max(0.0, 0.2070855 - Q.sj2_dr) / 0.001346201   # +0.2%  tau21 < 0.3472 and sj2_dr < 0.2071
        + 0.001517716 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.005788041 - Q.sum_z_dr2_top15) / 1.968469e-05   # +0.2%  tau21_b2 < 0.2352 and sum_z_dr2_top15 < 0.005788
        - 0.001496242 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.sum_z_dr2_top10 - 0.007678544) / 7.055851e-07   # -0.1%  psi_0p3 > 0.9974 and sum_z_dr2_top10 > 0.007679
        - 0.001390822 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) * max(0.0, Q.pt_2 - 92.5) / 0.4280383   # -0.1%  n_dr_0p1_0p2 > 33 and pt_2 > 92.5
        + 0.001326542 * max(0.0, 0.005180665 - Q.sum_z_dr2_top2) / 0.002696378   # +0.1%  sum_z_dr2_top2 < 0.005181
        + 0.001220998 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.4357228 - Q.max_dr) / 0.8636325   # +0.1%  n_dr_0p2_0p4 < 15 and max_dr < 0.4357
        - 0.001204484 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0004404746 - Q.mean_phi) / 0.003432598   # -0.1%  n_dr_0p2_0p4 < 15 and mean_phi < 0.0004405
        - 0.001077754 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 10.21558   # -0.1%  n_dr_0p2_0p4 < 21 and n_dr_0p1_0p2 > 21
        - 0.0008003903 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sj3_dr23 - 0.2351209) / 0.2078418   # -0.1%  n_dr_0p2_0p4 < 21 and sj3_dr23 > 0.2351
        + 0.0006912009 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.2618467 - Q.D3) / 0.003307359   # +0.1%  tau21_b2 < 0.2352 and D3 < 0.2618
        + 0.0006690883 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.ptdr0_2 - 11.87288) / 9.289026   # +0.1%  n_dr_0p2_0p4 < 15 and ptdr0_2 > 11.87
        - 0.0005921175 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.ptdr0_2 - 0.7280557) / 0.3442961   # -0.1%  tau21_b2 < 0.2352 and ptdr0_2 > 0.7281
        + 0.0005840199 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 44.7593 - Q.orientation_deg) / 2.633912   # +0.1%  tau21_b2 < 0.2352 and orientation_deg < 44.76
        - 0.0005737714 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 972.0419 - Q.sum_pt) / 0.2148301   # -0.1%  tau21_b2 < 0.2352 and sum_pt < 972
        + 0.000548276 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.eta_1 - 0.002292633) / 0.1076072   # +0.1%  n_dr_0p2_0p4 < 15 and eta_1 > 0.002293
        - 0.0005211145 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 36.0 - Q.n_real_top40) / 0.1793513   # -0.1%  tau21_b2 < 0.2352 and n_real_top40 < 36
        - 0.0004237895 * max(0.0, 0.1219132 - Q.tau1) * max(0.0, Q.zdr_0 - 0.0138766) / 1.915744e-05   # -0.0%  tau1 < 0.1219 and zdr_0 > 0.01388
        + 0.0004222828 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, -0.009371567 - Q.eta_1) / 0.08700179   # +0.0%  n_dr_0p2_0p4 < 15 and eta_1 < -0.009372
        + 0.0003868502 * max(0.0, Q.sj2_zsoft - 0.4184936) * max(0.0, Q.eccentricity - 0.8680812) / 0.0002528926   # +0.0%  sj2_zsoft > 0.4185 and eccentricity > 0.8681
        - 0.0002473835 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.ptdr0_14 - 4.336333) / 0.007065186   # -0.0%  tau21_b2 < 0.2352 and ptdr0_14 > 4.336
        - 0.0002105295 * max(0.0, 0.08068193 - Q.sum_z_dr) * max(0.0, Q.sum_z_dr2_top3 - 0.003952582) / 2.162737e-06   # -0.0%  sum_z_dr < 0.08068 and sum_z_dr2_top3 > 0.003953
        - 0.0001290382 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_z_dr2_top5 - 0.008329695) / 0.007978129   # -0.0%  n_dr_0p2_0p4 < 21 and sum_z_dr2_top5 > 0.00833
        - 0.0001219286 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.ptdr0_10 - 7.702385) / 0.3817228   # -0.0%  n_dr_0p2_0p4 < 21 and ptdr0_10 > 7.702
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 22.49;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.4908 * (0.08929069
        - 0.1658023 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -16.6%  log_sum_pt < 7.139
        + 0.1529182 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # +15.3%  sum_pt < 1261
        + 0.09868318 * max(0.0, Q.sum_z_dr - 0.04362872) / 0.03194126   # +9.9%  sum_z_dr > 0.04363
        - 0.08884977 * max(0.0, Q.LHA - 0.2091025) / 0.07391154   # -8.9%  LHA > 0.2091
        + 0.06528515 * max(0.0, Q.sum_z_dr - 0.03577037) / 0.03765667   # +6.5%  sum_z_dr > 0.03577
        - 0.04172075 * max(0.0, 0.02675364 - Q.sum_z_dr2_top15) / 0.01959938   # -4.2%  sum_z_dr2_top15 < 0.02675
        - 0.03628726 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # -3.6%  sum_z_dr > 0.0975
        - 0.02696326 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # -2.7%  n_dr_0p2_0p4 < 18
        - 0.02693548 * max(0.0, Q.sum_z_dr - 0.07374472) / 0.01465579   # -2.7%  sum_z_dr > 0.07374
        - 0.02221732 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # -2.2%  sum_z_dr2_top15 < 0.009962
        - 0.02137219 * max(0.0, Q.sum_z_dr - 0.07031778) / 0.01612201   # -2.1%  sum_z_dr > 0.07032
        - 0.02014266 * max(0.0, Q.z_top50_slots - 0.9586536) / 0.03427112   # -2.0%  z_top50_slots > 0.9587
        + 0.01826064 * max(0.0, 1042.609 - Q.sum_pt) / 35.65948   # +1.8%  sum_pt < 1043
        + 0.01792214 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +1.8%  LHA > 0.3332
        + 0.01529724 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, 1191.938 - Q.sum_pt_top30) / 4.563404   # +1.5%  sum_z_dr > 0.07032 and sum_pt_top30 < 1192
        + 0.01474787 * max(0.0, 0.00727763 - Q.sum_z_dr2_top15) / 0.00282147   # +1.5%  sum_z_dr2_top15 < 0.007278
        - 0.01424493 * max(0.0, Q.e2 - 0.02210818) / 0.01219843   # -1.4%  e2 > 0.02211
        + 0.01356608 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, 1115.723 - Q.sum_pt) / 1.878598   # +1.4%  sum_z_dr > 0.07032 and sum_pt < 1116
        - 0.0135294 * max(0.0, 1191.938 - Q.sum_pt_top30) / 206.0686   # -1.4%  sum_pt_top30 < 1192
        + 0.009283552 * max(0.0, Q.psi_0p1 - 0.6635952) / 0.1705715   # +0.9%  psi_0p1 > 0.6636
        + 0.00888406 * max(0.0, Q.LHA - 0.3203321) / 0.01671666   # +0.9%  LHA > 0.3203
        + 0.007126689 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # +0.7%  sum_pt < 1002
        + 0.00654118 * max(0.0, Q.psi_0p2 - 0.7286738) / 0.218083   # +0.7%  psi_0p2 > 0.7287
        - 0.005805378 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.139296 - Q.log_sum_pt) / 1.930358   # -0.6%  n_dr_0p2_0p4 < 18 and log_sum_pt < 7.139
        - 0.005634911 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 4.450169 - Q.D2) / 457.6573   # -0.6%  sum_pt < 1261 and D2 < 4.45
        + 0.005577053 * max(0.0, Q.LHA - 0.3098384) / 0.0195892   # +0.6%  LHA > 0.3098
        + 0.005460524 * max(0.0, Q.n_particles - 43.0) / 7.774361   # +0.5%  n_particles > 43
        - 0.005056862 * max(0.0, 0.0001086251 - Q.e3) / 5.924919e-05   # -0.5%  e3 < 0.0001086
        + 0.004952278 * max(0.0, Q.e2 - 0.04358622) / 0.002786698   # +0.5%  e2 > 0.04359
        + 0.004696366 * max(0.0, 0.08082334 - Q.dr_0) / 0.0377444   # +0.5%  dr_0 < 0.08082
        - 0.004511816 * max(0.0, 6.959294 - Q.log_sum_pt) / 0.0431467   # -0.5%  log_sum_pt < 6.959
        - 0.004476609 * max(0.0, Q.psi_0p3 - 0.9777125) / 0.01571542   # -0.4%  psi_0p3 > 0.9777
        - 0.004438891 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, 976.277 - Q.sum_pt_top50) / 0.4391148   # -0.4%  sum_z_dr > 0.07032 and sum_pt_top50 < 976.3
        + 0.004035029 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 3.793233e-05 - Q.e3) / 0.0001192169   # +0.4%  n_dr_0p2_0p4 < 18 and e3 < 3.793e-05
        - 0.003925337 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -0.4%  sum_z_dr > 0.1207
        + 0.003419759 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.8252004 - Q.psi_0p1) / 0.5392896   # +0.3%  n_dr_0p2_0p4 < 18 and psi_0p1 < 0.8252
        - 0.003330231 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -0.3%  psi_0p3 > 0.9943
        + 0.003129505 * max(0.0, 2.178951 - Q.D2) / 0.4648731   # +0.3%  D2 < 2.179
        + 0.002650008 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.3%  sj2_dr > 0.2232
        + 0.002435935 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # +0.2%  sum_pt_top40 < 1007
        + 0.002266892 * max(0.0, Q.n_dr_0p2_0p4 - 3.0) / 6.482203   # +0.2%  n_dr_0p2_0p4 > 3
        + 0.002180112 * max(0.0, Q.n_pt_above_1 - 54.0) / 1.837408   # +0.2%  n_pt_above_1 > 54
        - 0.002046387 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, Q.n_dr_0p05_0p1 - 7.0) / 1269.618   # -0.2%  sum_pt < 1261 and n_dr_0p05_0p1 > 7
        + 0.001992225 * max(0.0, 0.02675364 - Q.sum_z_dr2_top15) * max(0.0, Q.tau4 - 0.01517184) / 5.673266e-05   # +0.2%  sum_z_dr2_top15 < 0.02675 and tau4 > 0.01517
        - 0.001696024 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) * max(0.0, 0.3315857 - Q.max_dr) / 9.802766e-05   # -0.2%  sum_z_dr2_top15 < 0.009962 and max_dr < 0.3316
        - 0.00151109 * max(0.0, 1007.44 - Q.sum_pt_top40) * max(0.0, Q.sum_pt - 949.9169) / 555.6226   # -0.2%  sum_pt_top40 < 1007 and sum_pt > 949.9
        - 0.001464224 * max(0.0, 1027.303 - Q.sum_pt_top30) / 61.36161   # -0.1%  sum_pt_top30 < 1027
        + 0.001451463 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, Q.e2 - 0.04358622) / 0.000222409   # +0.1%  sum_z_dr > 0.07032 and e2 > 0.04359
        - 0.001421837 * max(0.0, 3.793233e-05 - Q.e3) / 1.004564e-05   # -0.1%  e3 < 3.793e-05
        + 0.001383084 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, 0.0795038 - Q.tau2) / 0.0003075621   # +0.1%  sum_z_dr > 0.07032 and tau2 < 0.0795
        + 0.001378335 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +0.1%  sum_pt_top50 < 1157
        - 0.0005569809 * max(0.0, 0.04575675 - Q.sj2_zsoft) / 0.00226832   # -0.1%  sj2_zsoft < 0.04576
        - 0.0005335528 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.z_0 - 0.08872428) / 1.838848   # -0.1%  sum_pt < 1002 and z_0 > 0.08872
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 6.43;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.430027 * (-0.09589385
        + 0.1293415 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # +12.9%  tau1 < 0.1073
        + 0.08083637 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.000105955   # +8.1%  sum_z_dr2_top15 < 0.006143 and z_dr_0p2_0p4 < 0.06849
        - 0.07791865 * max(0.0, Q.sum_pt_top20 - 750.7313) / 184.752   # -7.8%  sum_pt_top20 > 750.7
        - 0.06625461 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) / 0.002080651   # -6.6%  sum_z_dr2_top15 < 0.006143
        + 0.06066887 * max(0.0, 0.05660088 - Q.sum_z_dr) / 0.01080382   # +6.1%  sum_z_dr < 0.0566
        - 0.05876205 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # -5.9%  LHA < 0.2454
        - 0.05000436 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -5.0%  sum_z_dr > 0.1207
        + 0.0411674 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # +4.1%  sum_z_dr > 0.0975
        + 0.03372059 * max(0.0, 0.09591084 - Q.tau1) / 0.02629125   # +3.4%  tau1 < 0.09591
        - 0.03274982 * max(0.0, Q.sum_z_dr - 0.1402186) / 0.001871547   # -3.3%  sum_z_dr > 0.1402
        + 0.03199243 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.9777125) / 3.516962e-05   # +3.2%  sum_z_dr2_top15 < 0.006143 and psi_0p3 > 0.9777
        + 0.02897692 * max(0.0, 988.4554 - Q.sum_pt_top50) / 15.57124   # +2.9%  sum_pt_top50 < 988.5
        + 0.02765456 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +2.8%  LHA > 0.4042
        + 0.02655529 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +2.7%  LHA > 0.372
        + 0.0234199 * max(0.0, Q.psi_0p1 - 0.8509811) / 0.04568859   # +2.3%  psi_0p1 > 0.851
        + 0.02038521 * max(0.0, Q.e2 - 0.02515919) / 0.01027642   # +2.0%  e2 > 0.02516
        + 0.02013018 * max(0.0, 2.178951 - Q.D2) / 0.4648731   # +2.0%  D2 < 2.179
        + 0.01709543 * max(0.0, 0.004855289 - Q.sum_z_dr2_top15) / 0.001418414   # +1.7%  sum_z_dr2_top15 < 0.004855
        + 0.0152745 * max(0.0, 0.05660088 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.0003340726   # +1.5%  sum_z_dr < 0.0566 and psi_0p3 > 0.9638
        - 0.01347791 * max(0.0, 0.002412891 - Q.sum_z_dr2_top2) / 0.0009487944   # -1.3%  sum_z_dr2_top2 < 0.002413
        - 0.01319173 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -1.3%  log_sum_pt < 6.811
        - 0.01277264 * max(0.0, Q.z_top40_slots - 0.9674996) / 0.02046045   # -1.3%  z_top40_slots > 0.9675
        + 0.01275427 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, Q.soft5_z - 0.0004140594) / 2.555122e-05   # +1.3%  z_dr_0p1_0p2 > 0.4479 and soft5_z > 0.0004141
        - 0.01231058 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) / 0.03630126   # -1.2%  z_dr_0_0p05 > 0.8103
        + 0.01181189 * max(0.0, 889.8503 - Q.sum_pt_top50) / 3.808547   # +1.2%  sum_pt_top50 < 889.9
        - 0.01108233 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, Q.soft5_pt - 0.5297852) / 0.02338419   # -1.1%  z_dr_0p1_0p2 > 0.4479 and soft5_pt > 0.5298
        + 0.009205567 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # +0.9%  psi_0p3 > 0.9966
        + 0.008143283 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) * max(0.0, Q.n_dr_0p2_0p4 - 7.0) / 0.004066923   # +0.8%  sum_z_dr2_top15 < 0.006143 and n_dr_0p2_0p4 > 7
        - 0.007893833 * max(0.0, 0.05660088 - Q.sum_z_dr) * max(0.0, 1007.788 - Q.sum_pt) / 0.1758936   # -0.8%  sum_z_dr < 0.0566 and sum_pt < 1008
        - 0.007428635 * max(0.0, Q.z_top40_slots - 0.9674996) * max(0.0, 0.01158441 - Q.C3) / 0.0001133203   # -0.7%  z_top40_slots > 0.9675 and C3 < 0.01158
        - 0.006620525 * max(0.0, 0.0007894752 - Q.sum_z_dr2_top15) / 9.062109e-05   # -0.7%  sum_z_dr2_top15 < 0.0007895
        - 0.003342921 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, Q.sum_pt_top20 - 789.7344) / 304.8764   # -0.3%  sum_pt_top50 < 988.5 and sum_pt_top20 > 789.7
        - 0.003135242 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, 0.1312677 - Q.dr_13) / 0.7213696   # -0.3%  sum_pt_top50 < 988.5 and dr_13 < 0.1313
        + 0.002880481 * max(0.0, Q.z_top5_slots - 0.7963976) / 0.006402521   # +0.3%  z_top5_slots > 0.7964
        - 0.002615356 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) / 0.02385119   # -0.3%  z_dr_0p1_0p2 > 0.4479
        + 0.002607 * max(0.0, 0.2595052 - Q.sj2_dr) * max(0.0, Q.pt_0 - 213.5) / 5.810924   # +0.3%  sj2_dr < 0.2595 and pt_0 > 213.5
        + 0.002493032 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, Q.e3 - 0.0001086251) / 0.001799898   # +0.2%  sum_pt_top50 < 988.5 and e3 > 0.0001086
        + 0.002058895 * max(0.0, Q.LHA - 0.3719813) * max(0.0, 0.006427167 - Q.lam2) / 1.406065e-05   # +0.2%  LHA > 0.372 and lam2 < 0.006427
        - 0.00201141 * max(0.0, Q.LHA - 0.3719813) * max(0.0, Q.soft4_z - 0.002015695) / 2.138221e-06   # -0.2%  LHA > 0.372 and soft4_z > 0.002016
        - 0.001664365 * max(0.0, Q.sum_z_dr2_top15 - 0.02146578) / 0.0006182335   # -0.2%  sum_z_dr2_top15 > 0.02147
        + 0.001628528 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.09061548 - Q.dr_13) / 0.0001299587   # +0.2%  log_sum_pt < 6.811 and dr_13 < 0.09062
        - 0.00124854 * max(0.0, Q.sum_z_dr - 0.1564779) / 0.0006617163   # -0.1%  sum_z_dr > 0.1565
        + 0.001246968 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, 0.001416411 - Q.C3) / 0.00386608   # +0.1%  sum_pt_top50 < 988.5 and C3 < 0.001416
        - 0.001097891 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, Q.soft10_pt - 2.498047) / 6.651723   # -0.1%  sum_pt_top50 < 988.5 and soft10_pt > 2.498
        - 0.001059205 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, Q.soft5_pt - 1.652344) / 5.88648   # -0.1%  sum_pt_top50 < 988.5 and soft5_pt > 1.652
        - 0.0007794576 * max(0.0, Q.sum_z_dr - 0.1564779) * max(0.0, Q.soft6_pt - 1.931641) / 0.0004276585   # -0.1%  sum_z_dr > 0.1565 and soft6_pt > 1.932
        + 0.0005288929 * max(0.0, Q.sum_z_dr - 0.1402186) * max(0.0, Q.soft6_pt - 4.250195) / 9.168767e-05   # +0.1%  sum_z_dr > 0.1402 and soft6_pt > 4.25
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 14.05;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.0506 * (0.2680513
        - 0.1892329 * max(0.0, 0.1207452 - Q.sum_z_dr) / 0.05578614   # -18.9%  sum_z_dr < 0.1207
        + 0.1369443 * max(0.0, Q.e2 - 0.008484542) / 0.02258988   # +13.7%  e2 > 0.008485
        - 0.05684159 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # -5.7%  sum_z_dr > 0.0975
        - 0.04096232 * max(0.0, Q.tau1 - 0.04466492) / 0.04702157   # -4.1%  tau1 > 0.04466
        - 0.03837018 * max(0.0, Q.LHA - 0.1870291) / 0.08997036   # -3.8%  LHA > 0.187
        + 0.03233846 * max(0.0, 0.1207452 - Q.sum_z_dr) * max(0.0, Q.psi_0p2 - 0.948102) / 0.001922283   # +3.2%  sum_z_dr < 0.1207 and psi_0p2 > 0.9481
        - 0.02802347 * max(0.0, 0.02441963 - Q.sum_z_dr2_top5) / 0.01889954   # -2.8%  sum_z_dr2_top5 < 0.02442
        + 0.0259746 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +2.6%  LHA > 0.3332
        + 0.02456581 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +2.5%  tau1 < 0.07709
        + 0.02341395 * max(0.0, Q.pt_entropy - 2.07371) / 0.8049786   # +2.3%  pt_entropy > 2.074
        - 0.02293183 * max(0.0, Q.sum_pt_top5 - 402.625) / 200.0399   # -2.3%  sum_pt_top5 > 402.6
        + 0.02268581 * max(0.0, Q.sum_z_dr - 0.09749958) * max(0.0, 1156.659 - Q.sum_pt_top50) / 1.495133   # +2.3%  sum_z_dr > 0.0975 and sum_pt_top50 < 1157
        - 0.01960232 * max(0.0, 0.01414829 - Q.sum_z_dr2_top10) / 0.00877908   # -2.0%  sum_z_dr2_top10 < 0.01415
        + 0.01925579 * max(0.0, 0.0001086251 - Q.e3) / 5.924919e-05   # +1.9%  e3 < 0.0001086
        - 0.01652118 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -1.7%  psi_0p2 > 0.9087
        - 0.01608737 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # -1.6%  e2 > 0.0303
        - 0.01573522 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # -1.6%  sum_pt_top40 < 1025
        - 0.01472624 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # -1.5%  e3 < 0.0003372
        - 0.01436325 * max(0.0, Q.sum_z_dr - 0.1402186) / 0.001871547   # -1.4%  sum_z_dr > 0.1402
        - 0.01434192 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # -1.4%  n_dr_0p2_0p4 < 18
        - 0.01402916 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_real_top40 - 22.0) / 55.27538   # -1.4%  n_dr_0p2_0p4 < 11 and n_real_top40 > 22
        + 0.0136397 * max(0.0, 0.004007842 - Q.sum_z_dr2_top2) / 0.001907211   # +1.4%  sum_z_dr2_top2 < 0.004008
        + 0.01291294 * max(0.0, Q.sj2_dr - 0.09395198) / 0.1054596   # +1.3%  sj2_dr > 0.09395
        - 0.01103144 * max(0.0, Q.z_dr_0_0p05 - 0.7128619) / 0.07526795   # -1.1%  z_dr_0_0p05 > 0.7129
        - 0.009965033 * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0004356739   # -1.0%  psi_0p3 > 0.9985
        + 0.009603537 * max(0.0, 0.1207452 - Q.sum_z_dr) * max(0.0, 997.0189 - Q.sum_pt_top50) / 0.7202273   # +1.0%  sum_z_dr < 0.1207 and sum_pt_top50 < 997
        + 0.00955702 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +1.0%  LHA > 0.4042
        + 0.009379624 * max(0.0, 5.13841e-05 - Q.e3) / 1.709557e-05   # +0.9%  e3 < 5.138e-05
        + 0.008288645 * max(0.0, 2.410481 - Q.D2) / 0.587301   # +0.8%  D2 < 2.41
        - 0.008011617 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # -0.8%  sum_pt < 986.1
        - 0.00787475 * max(0.0, 0.2018786 - Q.tau21_b2) / 0.03799024   # -0.8%  tau21_b2 < 0.2019
        + 0.007200841 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, 114.75 - Q.pt_3) / 0.04784342   # +0.7%  tau1 > 0.1954 and pt_3 < 114.8
        + 0.007041456 * max(0.0, Q.psi_0p3 - 0.9989733) / 0.0002740091   # +0.7%  psi_0p3 > 0.999
        - 0.006956304 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # -0.7%  z_dr_0p1_0p2 < 0.1203
        - 0.005799436 * max(0.0, 2.026142 - Q.D2_b2) / 0.7121724   # -0.6%  D2_b2 < 2.026
        + 0.005638919 * max(0.0, Q.psi_0p1 - 0.9184255) / 0.01693566   # +0.6%  psi_0p1 > 0.9184
        - 0.005412897 * max(0.0, Q.sj2_dr - 0.09395198) * max(0.0, Q.M2 - 0.04260132) / 0.001979695   # -0.5%  sj2_dr > 0.09395 and M2 > 0.0426
        - 0.004971655 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, 0.1082864 - Q.z_3) / 4.224064e-05   # -0.5%  tau1 > 0.1954 and z_3 < 0.1083
        + 0.004777539 * max(0.0, 889.8383 - Q.sum_pt_top20) / 35.81237   # +0.5%  sum_pt_top20 < 889.8
        + 0.004759155 * max(0.0, Q.LHA - 0.1870291) * max(0.0, Q.sj3_pairmin_over_m - 0.1389615) / 0.01420104   # +0.5%  LHA > 0.187 and sj3_pairmin_over_m > 0.139
        + 0.00443538 * max(0.0, 0.002672224 - Q.soft7_z) / 0.001098778   # +0.4%  soft7_z < 0.002672
        - 0.004434436 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, Q.sum_pt - 1167.447) / 0.008922112   # -0.4%  tau1 > 0.1954 and sum_pt > 1167
        + 0.004180158 * max(0.0, 0.05602756 - Q.C2) / 0.009882647   # +0.4%  C2 < 0.05603
        - 0.00402709 * max(0.0, Q.tau1 - 0.1953848) / 0.000822026   # -0.4%  tau1 > 0.1954
        - 0.003857706 * max(0.0, 0.02944575 - Q.M3) / 0.00752156   # -0.4%  M3 < 0.02945
        + 0.003327861 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.max_dr - 0.2404747) / 0.4088487   # +0.3%  n_dr_0p2_0p4 < 11 and max_dr > 0.2405
        + 0.003219509 * max(0.0, 1024.942 - Q.sum_pt_top40) * max(0.0, Q.n_dr_0_0p05 - 1.0) / 377.2098   # +0.3%  sum_pt_top40 < 1025 and n_dr_0_0p05 > 1
        - 0.003111779 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, 0.8644992 - Q.tau32) / 2.698478e-05   # -0.3%  e3 < 0.0003372 and tau32 < 0.8645
        + 0.003009992 * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 1.386277   # +0.3%  n_dr_0p1_0p2 > 21
        - 0.002800334 * max(0.0, Q.psi_0p3 - 0.9989733) * max(0.0, Q.eccentricity - 0.5245966) / 9.193571e-05   # -0.3%  psi_0p3 > 0.999 and eccentricity > 0.5246
        - 0.002642698 * max(0.0, Q.e2 - 0.03029714) * max(0.0, Q.soft1_pt - 1.091797) / 0.001350423   # -0.3%  e2 > 0.0303 and soft1_pt > 1.092
        + 0.002604855 * max(0.0, Q.tau1 - 0.1751567) / 0.002322569   # +0.3%  tau1 > 0.1752
        + 0.002469355 * max(0.0, 0.1207452 - Q.sum_z_dr) * max(0.0, Q.pt1_dr01 - 5.351077) / 0.09847904   # +0.2%  sum_z_dr < 0.1207 and pt1_dr01 > 5.351
        - 0.002167575 * max(0.0, Q.sj2_dr - 0.09395198) * max(0.0, 0.6025827 - Q.planar_flow) / 0.02318142   # -0.2%  sj2_dr > 0.09395 and planar_flow < 0.6026
        - 0.001996369 * max(0.0, Q.LHA - 0.1870291) * max(0.0, 0.9973959 - Q.psi_0p3) / 0.001748639   # -0.2%  LHA > 0.187 and psi_0p3 < 0.9974
        - 0.001908971 * max(0.0, Q.e2 - 0.03029714) * max(0.0, Q.log_sum_pt - 6.910131) / 0.0002282547   # -0.2%  e2 > 0.0303 and log_sum_pt > 6.91
        - 0.001794322 * max(0.0, 6.856375 - Q.log_sum_pt) / 0.008053459   # -0.2%  log_sum_pt < 6.856
        - 0.001524878 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.pt_7 - 34.53125) / 0.2313795   # -0.2%  z_dr_0p1_0p2 < 0.1203 and pt_7 > 34.53
        + 0.001194861 * max(0.0, Q.ptdr0_2 - 7.740999) / 2.384946   # +0.1%  ptdr0_2 > 7.741
        + 0.001007247 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0_0p05 - 20.0) / 5.811896   # +0.1%  n_dr_0p2_0p4 < 11 and n_dr_0_0p05 > 20
        - 0.0009378365 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # -0.1%  sum_pt_top5 > 791.1
        - 0.0009377703 * max(0.0, Q.sd_rg - 0.3017146) / 0.004328764   # -0.1%  sd_rg > 0.3017
        - 0.0008195489 * max(0.0, 889.8503 - Q.sum_pt_top50) * max(0.0, 0.2126484 - Q.z_dr_0p05_0p1) / 0.340937   # -0.1%  sum_pt_top50 < 889.9 and z_dr_0p05_0p1 < 0.2126
        + 0.0007878445 * max(0.0, 889.8503 - Q.sum_pt_top50) / 3.808547   # +0.1%  sum_pt_top50 < 889.9
        - 0.0004915776 * max(0.0, 986.0565 - Q.sum_pt) * max(0.0, 0.02358801 - Q.dr_3) / 0.04247891   # -0.0%  sum_pt < 986.1 and dr_3 < 0.02359
        - 0.0003271421 * max(0.0, Q.n_pt_above_5 - 42.0) / 0.549916   # -0.0%  n_pt_above_5 > 42
        - 0.0002127263 * max(0.0, Q.e2 - 0.03029714) * max(0.0, Q.z_dr_0p4_up - 0.0) / 3.251128e-06   # -0.0%  e2 > 0.0303 and z_dr_0p4_up > 0
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 16.38;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.38217 * (-0.03844281
        - 0.1291282 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # -12.9%  LHA < 0.3098
        + 0.1259308 * max(0.0, 0.08589404 - Q.sum_z_dr) / 0.02745918   # +12.6%  sum_z_dr < 0.08589
        + 0.09149357 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +9.1%  LHA < 0.3332
        - 0.06548096 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) / 0.002080651   # -6.5%  sum_z_dr2_top15 < 0.006143
        + 0.05475687 * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) / 0.00326329   # +5.5%  sum_z_dr2_top15 < 0.007888
        - 0.04906626 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # -4.9%  sum_z_dr < 0.0975
        + 0.03952594 * max(0.0, 0.006615185 - Q.sum_z_dr2_top15) / 0.002375805   # +4.0%  sum_z_dr2_top15 < 0.006615
        - 0.03937995 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # -3.9%  e2 < 0.04083
        + 0.03018773 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +3.0%  z_dr_0p2_0p4 < 0.09123
        - 0.02510442 * max(0.0, 0.09338587 - Q.dr_0) / 0.04813256   # -2.5%  dr_0 < 0.09339
        + 0.02071261 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +2.1%  n_dr_0p2_0p4 < 15
        + 0.01856009 * max(0.0, 0.05775119 - Q.dr_0) / 0.02089184   # +1.9%  dr_0 < 0.05775
        - 0.01807796 * max(0.0, Q.psi_0p2 - 0.8706159) / 0.08984765   # -1.8%  psi_0p2 > 0.8706
        + 0.01662789 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # +1.7%  e2 < 0.04359
        - 0.01643735 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # -1.6%  log_sum_pt > 6.894
        - 0.01572737 * max(0.0, Q.psi_0p1 - 0.6635952) / 0.1705715   # -1.6%  psi_0p1 > 0.6636
        - 0.01451994 * max(0.0, 0.007678544 - Q.sum_z_dr2_top10) / 0.00347358   # -1.5%  sum_z_dr2_top10 < 0.007679
        + 0.0130055 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 33.0 - Q.n_dr_0p1_0p2) / 87.77824   # +1.3%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 33
        - 0.01251389 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005337976 - Q.sum_z_dr2_top10) / 0.007083386   # -1.3%  n_dr_0p2_0p4 < 10 and sum_z_dr2_top10 < 0.005338
        - 0.01210028 * max(0.0, 0.03577037 - Q.sum_z_dr) / 0.004182456   # -1.2%  sum_z_dr < 0.03577
        + 0.01151575 * max(0.0, 19.0 - Q.n_dr_0p1_0p2) / 8.166005   # +1.2%  n_dr_0p1_0p2 < 19
        - 0.01089796 * max(0.0, 0.1623049 - Q.sj3_dr_max) / 0.01012502   # -1.1%  sj3_dr_max < 0.1623
        + 0.01087968 * max(0.0, 0.004752876 - Q.sum_z_dr2_top10) / 0.001623418   # +1.1%  sum_z_dr2_top10 < 0.004753
        + 0.01028396 * max(0.0, 1.409617 - Q.D2) / 0.1519846   # +1.0%  D2 < 1.41
        - 0.009873434 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # -1.0%  z_dr_0p2_0p4 < 0.06849
        + 0.009289599 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +0.9%  e2 < 0.02516
        + 0.00912174 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # +0.9%  LHA < 0.187
        + 0.008674586 * max(0.0, 0.076787 - Q.sum_z_dr) / 0.02105267   # +0.9%  sum_z_dr < 0.07679
        + 0.008088435 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # +0.8%  sum_pt > 986.1
        - 0.007554864 * max(0.0, 3.376709e-05 - Q.e3) / 8.295151e-06   # -0.8%  e3 < 3.377e-05
        + 0.007287465 * max(0.0, 0.01564747 - Q.zdr_0) / 0.007296635   # +0.7%  zdr_0 < 0.01565
        - 0.006770759 * max(0.0, 0.1548383 - Q.z_dr_0p1_0p2) / 0.06710802   # -0.7%  z_dr_0p1_0p2 < 0.1548
        + 0.00633878 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +0.6%  e2 < 0.01879
        + 0.005835016 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +0.6%  n_dr_0p2_0p4 < 10
        - 0.005593198 * max(0.0, 1.409617 - Q.D2) * max(0.0, Q.n_real_top50 - 22.0) / 2.18239   # -0.6%  D2 < 1.41 and n_real_top50 > 22
        + 0.005513663 * max(0.0, 19.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.6025827 - Q.planar_flow) / 1.538522   # +0.6%  n_dr_0p1_0p2 < 19 and planar_flow < 0.6026
        + 0.005247581 * max(0.0, 0.1506299 - Q.sj3_dr_max) / 0.008148644   # +0.5%  sj3_dr_max < 0.1506
        - 0.005151909 * max(0.0, 0.09573228 - Q.z_dr_0_0p05) / 0.02128126   # -0.5%  z_dr_0_0p05 < 0.09573
        - 0.005000071 * max(0.0, 0.005911134 - Q.zdr_0) / 0.001247991   # -0.5%  zdr_0 < 0.005911
        + 0.004878314 * max(0.0, 1.788105 - Q.D2) / 0.2865857   # +0.5%  D2 < 1.788
        + 0.004606205 * max(0.0, 0.1894436 - Q.sj3_dr_max) / 0.01747403   # +0.5%  sj3_dr_max < 0.1894
        - 0.003404827 * max(0.0, Q.n_dr_0p1_0p2 - 14.0) / 3.026787   # -0.3%  n_dr_0p1_0p2 > 14
        + 0.003311434 * max(0.0, 0.00130722 - Q.sum_z_dr2_top10) / 0.0002794102   # +0.3%  sum_z_dr2_top10 < 0.001307
        - 0.002839299 * max(0.0, 9.0 - Q.n_dr_0p1_0p2) / 1.720126   # -0.3%  n_dr_0p1_0p2 < 9
        + 0.002575592 * max(0.0, Q.psi_0p2 - 0.995185) / 0.0009240812   # +0.3%  psi_0p2 > 0.9952
        + 0.002328436 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # +0.2%  psi_0p3 > 0.998
        + 0.002306994 * max(0.0, Q.log_sum_pt - 6.893714) * max(0.0, Q.M2 - 0.04894324) / 0.00143094   # +0.2%  log_sum_pt > 6.894 and M2 > 0.04894
        - 0.002261717 * max(0.0, 0.076787 - Q.sum_z_dr) * max(0.0, 3.814159 - Q.D2) / 0.01309078   # -0.2%  sum_z_dr < 0.07679 and D2 < 3.814
        - 0.002102592 * max(0.0, 0.09749958 - Q.sum_z_dr) * max(0.0, Q.tau4 - 0.008810529) / 0.0001863032   # -0.2%  sum_z_dr < 0.0975 and tau4 > 0.008811
        + 0.002018284 * max(0.0, Q.z_dr_0p05_0p1 - 0.4026646) / 0.08060812   # +0.2%  z_dr_0p05_0p1 > 0.4027
        - 0.001747257 * max(0.0, Q.log_sum_pt - 6.893714) * max(0.0, Q.dr1_7 - 0.08891009) / 0.002073639   # -0.2%  log_sum_pt > 6.894 and dr1_7 > 0.08891
        + 0.001722153 * max(0.0, 0.09573228 - Q.z_dr_0_0p05) * max(0.0, Q.max_dr - 0.1939977) / 0.003427714   # +0.2%  z_dr_0_0p05 < 0.09573 and max_dr > 0.194
        - 0.001632683 * max(0.0, 1.409617 - Q.D2) * max(0.0, 0.04963857 - Q.z_7) / 0.002083888   # -0.2%  D2 < 1.41 and z_7 < 0.04964
        - 0.001480822 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0) / 30.16986   # -0.1%  n_dr_0p2_0p4 < 10 and n_particles > 34
        - 0.001166842 * max(0.0, 0.1732514 - Q.N2) / 0.004810221   # -0.1%  N2 < 0.1733
        + 0.0008963532 * max(0.0, 0.1210264 - Q.sj3_dr_max) / 0.004637661   # +0.1%  sj3_dr_max < 0.121
        + 0.0007557078 * max(0.0, Q.sum_pt - 986.0565) * max(0.0, Q.dr1_7 - 0.1792439) / 0.7892819   # +0.1%  sum_pt > 986.1 and dr1_7 > 0.1792
        + 0.0007084719 * max(0.0, Q.z_dr_0p05_0p1 - 0.8509215) / 0.003556426   # +0.1%  z_dr_0p05_0p1 > 0.8509
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 4.27;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.27041 * (-0.1234401
        + 0.1117903 * max(0.0, 0.06171014 - Q.sum_z_dr) / 0.01295056   # +11.2%  sum_z_dr < 0.06171
        - 0.07503208 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, 0.01204531 - Q.mean_eta2) / 0.001166867   # -7.5%  log_sum_pt > 6.811 and mean_eta2 < 0.01205
        + 0.04909794 * max(0.0, 4.646151e-05 - Q.e3) / 1.427904e-05   # +4.9%  e3 < 4.646e-05
        - 0.04533553 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # -4.5%  LHA < 0.2454
        + 0.03916502 * max(0.0, 0.05048381 - Q.sum_z_dr) / 0.008533025   # +3.9%  sum_z_dr < 0.05048
        - 0.03676644 * max(0.0, 0.1048824 - Q.sj3_dr12) / 0.02123626   # -3.7%  sj3_dr12 < 0.1049
        + 0.03488532 * max(0.0, 0.05048381 - Q.sum_z_dr) * max(0.0, Q.log_sum_pt - 6.811175) / 0.001311076   # +3.5%  sum_z_dr < 0.05048 and log_sum_pt > 6.811
        - 0.03438897 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # -3.4%  e2 < 0.02516
        + 0.03301505 * max(0.0, 0.005850286 - Q.mean_eta2) / 0.00275351   # +3.3%  mean_eta2 < 0.00585
        + 0.03089326 * max(0.0, 0.005788041 - Q.sum_z_dr2_top15) / 0.001877672   # +3.1%  sum_z_dr2_top15 < 0.005788
        + 0.03078122 * max(0.0, Q.psi_0p2 - 0.9734513) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 0.1343004   # +3.1%  psi_0p2 > 0.9735 and n_dr_0p1_0p2 < 21
        + 0.02942943 * max(0.0, 1167.447 - Q.sum_pt) / 137.8628   # +2.9%  sum_pt < 1167
        - 0.0290363 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # -2.9%  sum_pt > 907.9
        + 0.02775234 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +2.8%  psi_0p3 > 0.9897
        + 0.0265175 * max(0.0, 0.1840219 - Q.sj3_dr12) / 0.05790731   # +2.7%  sj3_dr12 < 0.184
        - 0.02499597 * max(0.0, 0.2284021 - Q.LHA) / 0.02716657   # -2.5%  LHA < 0.2284
        + 0.02211723 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +2.2%  n_dr_0p2_0p4 < 10
        + 0.02140183 * max(0.0, 0.001411894 - Q.lam2) / 0.0006681999   # +2.1%  lam2 < 0.001412
        + 0.02085724 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.006615185 - Q.sum_z_dr2_top15) / 1.491736e-05   # +2.1%  psi_0p3 > 0.9897 and sum_z_dr2_top15 < 0.006615
        - 0.01920336 * max(0.0, 0.1506299 - Q.sj3_dr_max) / 0.008148644   # -1.9%  sj3_dr_max < 0.1506
        + 0.01778599 * max(0.0, 0.2628766 - Q.sj3_dr_max) / 0.05345657   # +1.8%  sj3_dr_max < 0.2629
        - 0.01768336 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, 0.1903878 - Q.sj3_z3) / 0.01422725   # -1.8%  log_sum_pt > 6.811 and sj3_z3 < 0.1904
        - 0.01395477 * max(0.0, 0.04362872 - Q.sum_z_dr) / 0.006325401   # -1.4%  sum_z_dr < 0.04363
        + 0.01307899 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +1.3%  LHA > 0.372
        + 0.0129894 * max(0.0, 0.1690079 - Q.sj3_z3) / 0.08254186   # +1.3%  sj3_z3 < 0.169
        + 0.01211175 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # +1.2%  LHA < 0.187
        + 0.01199572 * max(0.0, 0.06069672 - Q.sj3_dr12) / 0.008100502   # +1.2%  sj3_dr12 < 0.0607
        - 0.01183288 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # -1.2%  LHA > 0.4042
        - 0.01129482 * max(0.0, 0.002197765 - Q.sum_z_dr2_top15) / 0.0004509993   # -1.1%  sum_z_dr2_top15 < 0.002198
        - 0.01081927 * max(0.0, 1167.447 - Q.sum_pt) * max(0.0, 0.05910514 - Q.M2) / 0.9368297   # -1.1%  sum_pt < 1167 and M2 < 0.05911
        + 0.01075127 * max(0.0, 0.1999777 - Q.sj3_dr_max) / 0.02143   # +1.1%  sj3_dr_max < 0.2
        - 0.009545505 * max(0.0, 4.646151e-05 - Q.e3) * max(0.0, 0.2140047 - Q.z_1) / 1.137126e-06   # -1.0%  e3 < 4.646e-05 and z_1 < 0.214
        - 0.009474128 * max(0.0, 0.01414829 - Q.sum_z_dr2_top10) / 0.00877908   # -0.9%  sum_z_dr2_top10 < 0.01415
        - 0.0082799 * max(0.0, Q.psi_0p2 - 0.9734513) * max(0.0, 40.0 - Q.n_real_top40) / 0.06806351   # -0.8%  psi_0p2 > 0.9735 and n_real_top40 < 40
        + 0.00803443 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, Q.z_1st - 0.1586697) / 0.0005591086   # +0.8%  psi_0p3 > 0.9897 and z_1st > 0.1587
        + 0.007933159 * max(0.0, Q.pt_dispersion - 0.3227599) / 0.0409829   # +0.8%  pt_dispersion > 0.3228
        + 0.007412091 * max(0.0, Q.LHA - 0.3719813) * max(0.0, Q.log_sum_pt - 6.856375) / 0.0004589561   # +0.7%  LHA > 0.372 and log_sum_pt > 6.856
        - 0.007240728 * max(0.0, 0.2628766 - Q.sj3_dr_max) * max(0.0, Q.sum_z_dr2_top10 - 0.004752876) / 3.407988e-05   # -0.7%  sj3_dr_max < 0.2629 and sum_z_dr2_top10 > 0.004753
        - 0.006734131 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -0.7%  z_dr_0_0p05 > 0.8789
        + 0.006536039 * max(0.0, 0.1210264 - Q.sj3_dr_max) / 0.004637661   # +0.7%  sj3_dr_max < 0.121
        - 0.006466312 * max(0.0, 0.02620804 - Q.z_dr_0p1_0p2) / 0.004479696   # -0.6%  z_dr_0p1_0p2 < 0.02621
        - 0.005720376 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # -0.6%  e2 > 0.0303
        - 0.005640124 * max(0.0, Q.n_dr_0_0p05 - 20.0) / 1.33142   # -0.6%  n_dr_0_0p05 > 20
        + 0.004957516 * max(0.0, 0.004169954 - Q.sum_z_dr2_top15) / 0.001130716   # +0.5%  sum_z_dr2_top15 < 0.00417
        + 0.004040531 * max(0.0, Q.LHA - 0.404204) * max(0.0, 0.0329485 - Q.C2_b2) / 2.756829e-05   # +0.4%  LHA > 0.4042 and C2_b2 < 0.03295
        + 0.003971776 * max(0.0, Q.n_dr_0_0p05 - 20.0) * max(0.0, 3.259863 - Q.soft4_pt) / 2.781449   # +0.4%  n_dr_0_0p05 > 20 and soft4_pt < 3.26
        + 0.003071516 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # +0.3%  e2 > 0.06524
        - 0.002988868 * max(0.0, Q.sum_pt_top10 - 916.1578) / 16.81479   # -0.3%  sum_pt_top10 > 916.2
        - 0.002828764 * max(0.0, Q.sum_z_dr2_top3 - 0.02353672) / 0.0004246139   # -0.3%  sum_z_dr2_top3 > 0.02354
        + 0.002363586 * max(0.0, Q.LHA - 0.3719813) * max(0.0, 0.003687605 - Q.lam2) / 4.619984e-06   # +0.2%  LHA > 0.372 and lam2 < 0.003688
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 33.3;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.3009 * (0.06127277
        - 0.3633532 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # -36.3%  log_sum_pt < 6.989
        + 0.3490579 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # +34.9%  sum_pt < 1085
        + 0.03689911 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # +3.7%  e3 < 0.0005178
        + 0.02293558 * max(0.0, 1048.098 - Q.sum_pt_top50) / 44.36839   # +2.3%  sum_pt_top50 < 1048
        - 0.02243221 * max(0.0, 6.959294 - Q.log_sum_pt) / 0.0431467   # -2.2%  log_sum_pt < 6.959
        - 0.01719622 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # -1.7%  sum_z_dr > 0.0975
        + 0.01639091 * max(0.0, Q.LHA - 0.3203321) / 0.01671666   # +1.6%  LHA > 0.3203
        - 0.01507232 * max(0.0, Q.sum_z_dr - 0.08589404) / 0.01080971   # -1.5%  sum_z_dr > 0.08589
        + 0.0148083 * max(0.0, Q.sum_z_dr2_top15 - 0.005383629) / 0.003662853   # +1.5%  sum_z_dr2_top15 > 0.005384
        - 0.01343939 * max(0.0, 1115.723 - Q.sum_pt) / 92.38491   # -1.3%  sum_pt < 1116
        - 0.01319199 * max(0.0, Q.sum_z_dr2_top15 - 0.00727763) / 0.002923446   # -1.3%  sum_z_dr2_top15 > 0.007278
        + 0.009162786 * max(0.0, Q.sum_z_dr - 0.09749958) * max(0.0, 1167.447 - Q.sum_pt) / 1.406455   # +0.9%  sum_z_dr > 0.0975 and sum_pt < 1167
        - 0.007619122 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -0.8%  psi_0p2 > 0.9087
        + 0.007156804 * max(0.0, Q.sum_z_dr2_top15 - 0.009962397) * max(0.0, 1225.842 - Q.sum_pt_top40) / 0.6413342   # +0.7%  sum_z_dr2_top15 > 0.009962 and sum_pt_top40 < 1226
        - 0.006869913 * max(0.0, Q.sum_pt - 1167.447) / 14.21193   # -0.7%  sum_pt > 1167
        - 0.006627256 * max(0.0, 1115.723 - Q.sum_pt) * max(0.0, 2.0948e-08 - Q.e4) / 1.57227e-06   # -0.7%  sum_pt < 1116 and e4 < 2.095e-08
        + 0.005428215 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # +0.5%  log_sum_pt > 7.063
        - 0.005047634 * max(0.0, 0.0001841806 - Q.e3) / 0.0001220279   # -0.5%  e3 < 0.0001842
        + 0.004698863 * max(0.0, Q.e2 - 0.04082832) / 0.003398869   # +0.5%  e2 > 0.04083
        - 0.004692108 * max(0.0, Q.sum_z_dr2_top15 - 0.009962397) / 0.002311908   # -0.5%  sum_z_dr2_top15 > 0.009962
        - 0.004581756 * max(0.0, 26.0 - Q.n_dr_0p1_0p2) / 14.20854   # -0.5%  n_dr_0p1_0p2 < 26
        - 0.003779209 * max(0.0, 0.0001841806 - Q.e3) * max(0.0, 41.4375 - Q.pt_9) / 0.00190888   # -0.4%  e3 < 0.0001842 and pt_9 < 41.44
        - 0.003542717 * max(0.0, Q.sum_pt_top20 - 956.5062) / 37.46564   # -0.4%  sum_pt_top20 > 956.5
        - 0.003354491 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -0.3%  sum_z_dr > 0.1207
        + 0.003127537 * max(0.0, 6.959294 - Q.log_sum_pt) * max(0.0, 0.01311308 - Q.C3) / 0.0002829716   # +0.3%  log_sum_pt < 6.959 and C3 < 0.01311
        + 0.002432138 * max(0.0, 41.4375 - Q.pt_9) / 15.12338   # +0.2%  pt_9 < 41.44
        + 0.002280372 * max(0.0, Q.sum_z_dr2_top5 - 0.00422683) / 0.003365384   # +0.2%  sum_z_dr2_top5 > 0.004227
        - 0.00227151 * max(0.0, Q.sum_z_dr2_top15 - 0.01563836) / 0.001334556   # -0.2%  sum_z_dr2_top15 > 0.01564
        - 0.002158957 * max(0.0, 0.1872805 - Q.z_dr_0p1_0p2) / 0.08737974   # -0.2%  z_dr_0p1_0p2 < 0.1873
        - 0.002087417 * max(0.0, 0.9431554 - Q.z_top15_slots) / 0.1161773   # -0.2%  z_top15_slots < 0.9432
        + 0.002052365 * max(0.0, Q.sum_pt_top30 - 1111.245) / 12.62573   # +0.2%  sum_pt_top30 > 1111
        - 0.001823504 * max(0.0, 23.0 - Q.n_pt_above_5) / 1.859217   # -0.2%  n_pt_above_5 < 23
        - 0.001812341 * max(0.0, Q.sum_z_dr - 0.1402186) * max(0.0, Q.soft6_pt - 2.162109) / 0.0007576602   # -0.2%  sum_z_dr > 0.1402 and soft6_pt > 2.162
        + 0.001761535 * max(0.0, Q.sum_z_dr - 0.1402186) * max(0.0, Q.soft6_z - 0.002707742) / 4.721914e-07   # +0.2%  sum_z_dr > 0.1402 and soft6_z > 0.002708
        - 0.001666353 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.6025827 - Q.planar_flow) / 11.76045   # -0.2%  sum_pt < 1085 and planar_flow < 0.6026
        + 0.00159813 * max(0.0, 26.0 - Q.n_dr_0p1_0p2) * max(0.0, 1027.303 - Q.sum_pt_top30) / 631.9123   # +0.2%  n_dr_0p1_0p2 < 26 and sum_pt_top30 < 1027
        - 0.001500206 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # -0.2%  C2 > 0.06656
        + 0.001345992 * max(0.0, 1048.098 - Q.sum_pt_top50) * max(0.0, 0.0003125151 - Q.sum_z_dr2_top2) / 0.002122269   # +0.1%  sum_pt_top50 < 1048 and sum_z_dr2_top2 < 0.0003125
        + 0.001250953 * max(0.0, Q.tau4 - 0.01517184) / 0.004291999   # +0.1%  tau4 > 0.01517
        - 0.00114641 * max(0.0, Q.sum_z_dr - 0.1402186) / 0.001871547   # -0.1%  sum_z_dr > 0.1402
        - 0.001070162 * max(0.0, Q.sum_z_dr2_top15 - 0.005383629) * max(0.0, Q.soft6_pt - 1.740234) / 0.001771781   # -0.1%  sum_z_dr2_top15 > 0.005384 and soft6_pt > 1.74
        - 0.001043236 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, Q.sd_zg - 0.1634067) / 7.780975   # -0.1%  sum_pt < 1085 and sd_zg > 0.1634
        + 0.001020616 * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 1.386277   # +0.1%  n_dr_0p1_0p2 > 21
        + 0.0009928317 * max(0.0, 23.0 - Q.n_pt_above_5) * max(0.0, 3.345339 - Q.D2) / 1.366679   # +0.1%  n_pt_above_5 < 23 and D2 < 3.345
        - 0.0009419308 * max(0.0, Q.sum_z_dr - 0.1564779) / 0.0006617163   # -0.1%  sum_z_dr > 0.1565
        - 0.0008133205 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # -0.1%  log_sum_pt > 7.139
        + 0.0007472416 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.psi_0p3 - 0.9777125) / 8.852804e-05   # +0.1%  log_sum_pt > 7.139 and psi_0p3 > 0.9777
        - 0.0007394699 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, 0.0001021922 - Q.eta_1) / 0.0001366493   # -0.1%  log_sum_pt > 7.063 and eta_1 < 0.0001022
        + 0.0007378873 * max(0.0, Q.sum_pt - 1167.447) * max(0.0, 0.002292633 - Q.eta_1) / 0.1946292   # +0.1%  sum_pt > 1167 and eta_1 < 0.002293
        - 0.0007090853 * max(0.0, 13.46875 - Q.pt_13) / 1.49457   # -0.1%  pt_13 < 13.47
        - 0.0006040156 * max(0.0, 0.0005178279 - Q.e3) * max(0.0, 0.9966167 - Q.psi_0p3) / 2.589793e-06   # -0.1%  e3 < 0.0005178 and psi_0p3 < 0.9966
        - 0.0005808114 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 6.811175 - Q.log_sum_pt) / 1.755002   # -0.1%  sum_pt < 1085 and log_sum_pt < 6.811
        - 0.0005342058 * max(0.0, Q.sum_z_dr2_top15 - 0.02675364) / 0.0002253448   # -0.1%  sum_z_dr2_top15 > 0.02675
        - 0.0004469077 * max(0.0, Q.sum_z_dr - 0.1402186) * max(0.0, Q.soft6_pt - 3.328125) / 0.0002562638   # -0.0%  sum_z_dr > 0.1402 and soft6_pt > 3.328
        - 0.0003323221 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.dr0_7 - 0.1350754) / 0.4140323   # -0.0%  sum_pt_top50 < 959.1 and dr0_7 > 0.1351
        + 0.0003029255 * max(0.0, Q.sum_pt - 1167.447) * max(0.0, 0.06868286 - Q.abseta_12) / 0.5479112   # +0.0%  sum_pt > 1167 and abseta_12 < 0.06868
        - 0.0002721019 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, -0.0297699 - Q.phi_0) / 2.18725e-05   # -0.0%  log_sum_pt > 7.139 and phi_0 < -0.02977
        + 0.0001501869 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, -0.04013062 - Q.phi_0) / 1.626186e-05   # +0.0%  log_sum_pt > 7.139 and phi_0 < -0.04013
        - 0.000142774 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.04736241 - Q.dr_12) / 7.835712e-05   # -0.0%  log_sum_pt > 7.139 and dr_12 < 0.04736
        + 0.0001308503 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, -0.004917145 - Q.phi_0) / 4.999945e-05   # +0.0%  log_sum_pt > 7.139 and phi_0 < -0.004917
        - 3.343911e-05 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.abseta_9 - 0.1726074) / 1.740504e-05   # -0.0%  log_sum_pt > 7.063 and abseta_9 > 0.1726
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 20.49;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.48526 * (0.1094313
        - 0.1644181 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # -16.4%  LHA < 0.3332
        + 0.07484161 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # +7.5%  LHA < 0.3024
        - 0.06560641 * max(0.0, 0.07031778 - Q.sum_z_dr) / 0.01719521   # -6.6%  sum_z_dr < 0.07032
        + 0.06269475 * max(0.0, 0.2941033 - Q.LHA) / 0.05718399   # +6.3%  LHA < 0.2941
        + 0.04227746 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # +4.2%  sum_z_dr > 0.0975
        - 0.03935046 * max(0.0, 0.1751567 - Q.tau1) / 0.09101058   # -3.9%  tau1 < 0.1752
        - 0.03832064 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # -3.8%  tau1 < 0.1507
        + 0.03763048 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063   # +3.8%  n_dr_0p2_0p4 < 21
        - 0.03035351 * max(0.0, Q.sum_z_dr - 0.08068193) / 0.0122472   # -3.0%  sum_z_dr > 0.08068
        - 0.02718609 * max(0.0, Q.sum_z_dr - 0.076787) / 0.01351025   # -2.7%  sum_z_dr > 0.07679
        - 0.02513027 * max(0.0, Q.sum_z_dr - 0.08589404) / 0.01080971   # -2.5%  sum_z_dr > 0.08589
        + 0.02460012 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # +2.5%  sd_rg < 0.3017
        - 0.0239437 * max(0.0, Q.sj3_dr_max - 0.1999777) / 0.07046487   # -2.4%  sj3_dr_max > 0.2
        + 0.02284488 * max(0.0, Q.sj3_dr_max - 0.1717401) / 0.08948791   # +2.3%  sj3_dr_max > 0.1717
        + 0.01837413 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # +1.8%  LHA < 0.2601
        + 0.0174198 * max(0.0, Q.psi_0p3 - 0.9638082) / 0.02792824   # +1.7%  psi_0p3 > 0.9638
        - 0.01722165 * max(0.0, 0.3715619 - Q.sj3_dr_max) / 0.1322803   # -1.7%  sj3_dr_max < 0.3716
        + 0.01694737 * max(0.0, Q.sd_rg - 0.1596365) / 0.03553981   # +1.7%  sd_rg > 0.1596
        + 0.0166381 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +1.7%  e2 < 0.03481
        - 0.01555467 * max(0.0, 7.876005e-05 - Q.e3) / 3.594075e-05   # -1.6%  e3 < 7.876e-05
        - 0.01313526 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -1.3%  n_dr_0p2_0p4 < 11
        - 0.01309951 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # -1.3%  n_dr_0p1_0p2 < 17
        + 0.01269826 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +1.3%  tau21_b2 < 0.3425
        - 0.01219289 * max(0.0, Q.sd_rg - 0.2042612) / 0.02046744   # -1.2%  sd_rg > 0.2043
        - 0.01131088 * max(0.0, 0.07996447 - Q.C2) / 0.02445622   # -1.1%  C2 < 0.07996
        - 0.01087737 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.00727763 - Q.sum_z_dr2_top15) / 0.0001401748   # -1.1%  tau21_b2 < 0.3425 and sum_z_dr2_top15 < 0.007278
        - 0.0107769 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -1.1%  sum_z_dr > 0.1207
        + 0.008434313 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +0.8%  psi_0p3 > 0.9924
        + 0.008406061 * max(0.0, Q.z_dr_0_0p05 - 0.6283153) / 0.1153015   # +0.8%  z_dr_0_0p05 > 0.6283
        + 0.008386127 * max(0.0, 3.376709e-05 - Q.e3) / 8.295151e-06   # +0.8%  e3 < 3.377e-05
        - 0.008170602 * max(0.0, Q.LHA - 0.09165437) / 0.1710459   # -0.8%  LHA > 0.09165
        - 0.008150345 * max(0.0, Q.sum_z_dr2_top10 - 0.0001295334) / 0.006638188   # -0.8%  sum_z_dr2_top10 > 0.0001295
        + 0.00769559 * max(0.0, Q.sd_rg - 0.15159) / 0.03939137   # +0.8%  sd_rg > 0.1516
        - 0.006532513 * max(0.0, Q.sj3_dr_max - 0.2628766) / 0.03959245   # -0.7%  sj3_dr_max > 0.2629
        + 0.006423087 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +0.6%  lam2 < 0.001776
        + 0.006342005 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +0.6%  z_dr_0p1_0p2 < 0.1203
        - 0.005832816 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 1156.659 - Q.sum_pt_top50) / 14.04661   # -0.6%  tau21_b2 < 0.3425 and sum_pt_top50 < 1157
        + 0.005740096 * max(0.0, 0.05347848 - Q.dr_6) / 0.01424699   # +0.6%  dr_6 < 0.05348
        - 0.004388521 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.05347848 - Q.dr_6) / 5.520862e-05   # -0.4%  psi_0p3 > 0.9924 and dr_6 < 0.05348
        + 0.004133199 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) / 7.19088e-05   # +0.4%  tau21_b2 < 0.3425 and sum_z_dr2_top15 < 0.006143
        - 0.003960226 * max(0.0, 0.009970338 - Q.zdr_0) / 0.003273027   # -0.4%  zdr_0 < 0.00997
        + 0.003778528 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) * max(0.0, 17.82896 - Q.pt1_dr01) / 93.02539   # +0.4%  n_dr_0p1_0p2 < 17 and pt1_dr01 < 17.83
        + 0.003764072 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.01976735 - Q.sum_z_dr2_top10) / 0.001359663   # +0.4%  tau21_b2 < 0.3425 and sum_z_dr2_top10 < 0.01977
        + 0.003159921 * max(0.0, 0.03787151 - Q.M3) / 0.01370579   # +0.3%  M3 < 0.03787
        - 0.003010942 * max(0.0, Q.sum_z_dr - 0.09749958) * max(0.0, 0.4021783 - Q.max_dr) / 0.0002725179   # -0.3%  sum_z_dr > 0.0975 and max_dr < 0.4022
        + 0.002984291 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +0.3%  dr_0 < 0.06413
        + 0.002871133 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 2.275391 - Q.soft1_pt) / 0.1867267   # +0.3%  tau21_b2 < 0.3425 and soft1_pt < 2.275
        - 0.002661022 * max(0.0, Q.n_real_top50 - 43.0) / 3.389518   # -0.3%  n_real_top50 > 43
        + 0.002286479 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) / 3.82104   # +0.2%  n_dr_0p1_0p2 < 13
        + 0.002050312 * max(0.0, Q.sum_z_dr - 0.076787) * max(0.0, 0.4021783 - Q.max_dr) / 0.0004881945   # +0.2%  sum_z_dr > 0.07679 and max_dr < 0.4022
        + 0.002045223 * max(0.0, Q.n_real_top50 - 43.0) * max(0.0, 0.4357228 - Q.max_dr) / 0.2546702   # +0.2%  n_real_top50 > 43 and max_dr < 0.4357
        + 0.001959255 * max(0.0, Q.sum_z_dr - 0.1207452) * max(0.0, 0.4021783 - Q.max_dr) / 0.0001405604   # +0.2%  sum_z_dr > 0.1207 and max_dr < 0.4022
        + 0.001831571 * max(0.0, 0.07996447 - Q.C2) * max(0.0, Q.pt_3 - 55.15625) / 0.5532265   # +0.2%  C2 < 0.07996 and pt_3 > 55.16
        - 0.001706989 * max(0.0, Q.sd_rg - 0.15159) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.0411792   # -0.2%  sd_rg > 0.1516 and n_pt_above_50 > 4
        + 0.001540663 * max(0.0, 0.1219401 - Q.tau21_b2) / 0.01156068   # +0.2%  tau21_b2 < 0.1219
        - 0.001299774 * max(0.0, 0.003398536 - Q.z_dr_0p2_0p4) / 0.0005454047   # -0.1%  z_dr_0p2_0p4 < 0.003399
        - 0.001097554 * max(0.0, Q.sum_z_dr2_top10 - 0.0001295334) * max(0.0, 36.0 - Q.n_real_top40) / 0.006448485   # -0.1%  sum_z_dr2_top10 > 0.0001295 and n_real_top40 < 36
        - 0.001084045 * max(0.0, 0.001838217 - Q.soft9_z) / 0.0003773283   # -0.1%  soft9_z < 0.001838
        - 0.0009954101 * max(0.0, Q.z_dr_0p05_0p1 - 0.7108211) / 0.01402576   # -0.1%  z_dr_0p05_0p1 > 0.7108
        - 0.0008735617 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, Q.n_pt_above_10 - 23.0) / 0.09465507   # -0.1%  tau21_b2 < 0.3425 and n_pt_above_10 > 23
        - 0.0007871053 * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 1.386277   # -0.1%  n_dr_0p1_0p2 > 21
        + 0.0001713864 * max(0.0, Q.sum_z_dr - 0.08589404) * max(0.0, 36.0 - Q.n_real_top40) / 0.0005904753   # +0.0%  sum_z_dr > 0.08589 and n_real_top40 < 36
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 8.16;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.159965 * (-0.01374295
        + 0.08072565 * max(0.0, 0.1881908 - Q.sd_rg) / 0.07278694   # +8.1%  sd_rg < 0.1882
        + 0.06322356 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +6.3%  sum_pt > 907.9
        - 0.05876416 * max(0.0, Q.log_sum_pt - 6.915514) / 0.04811413   # -5.9%  log_sum_pt > 6.916
        + 0.05274234 * max(0.0, Q.sum_pt_top30 - 886.3438) / 117.6357   # +5.3%  sum_pt_top30 > 886.3
        - 0.05152887 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # -5.2%  tau1 < 0.07709
        - 0.05068847 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # -5.1%  sd_rg < 0.3017
        + 0.04312599 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # +4.3%  e2 < 0.04359
        + 0.04268876 * max(0.0, 0.08068193 - Q.sum_z_dr) / 0.02368455   # +4.3%  sum_z_dr < 0.08068
        - 0.04164722 * max(0.0, 0.1690338 - Q.sd_rg) / 0.06029557   # -4.2%  sd_rg < 0.169
        + 0.02935191 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +2.9%  sj2_dr > 0.2232
        + 0.02889656 * max(0.0, 0.007678544 - Q.sum_z_dr2_top10) / 0.00347358   # +2.9%  sum_z_dr2_top10 < 0.007679
        + 0.02540001 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +2.5%  lam2 < 0.001776
        - 0.02528464 * max(0.0, Q.LHA - 0.302389) / 0.02201444   # -2.5%  LHA > 0.3024
        - 0.02022071 * max(0.0, Q.sum_pt_top40 - 1001.523) / 47.19068   # -2.0%  sum_pt_top40 > 1002
        - 0.01950572 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # -2.0%  sum_z_dr2_top15 < 0.009962
        - 0.01900874 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.4299592 - Q.tau21_b2) / 0.001167326   # -1.9%  psi_0p3 > 0.9897 and tau21_b2 < 0.43
        - 0.01823444 * max(0.0, 2.410481 - Q.D2) / 0.587301   # -1.8%  D2 < 2.41
        - 0.01814362 * max(0.0, Q.psi_0p1 - 0.9184255) / 0.01693566   # -1.8%  psi_0p1 > 0.9184
        - 0.01664123 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -1.7%  psi_0p3 > 0.9897
        + 0.01646631 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # +1.6%  sum_z_dr2_top5 < 0.00833
        + 0.01597433 * max(0.0, Q.psi_0p1 - 0.707925) * max(0.0, 0.1828389 - Q.dr_max_012) / 0.01897164   # +1.6%  psi_0p1 > 0.7079 and dr_max_012 < 0.1828
        + 0.01589899 * max(0.0, Q.psi_0p1 - 0.707925) / 0.1360654   # +1.6%  psi_0p1 > 0.7079
        + 0.01580726 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +1.6%  tau21_b2 < 0.3425
        - 0.01461048 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # -1.5%  sum_z_dr2_top3 < 0.002152
        - 0.0144496 * max(0.0, 0.02793599 - Q.e2) / 0.005715277   # -1.4%  e2 < 0.02794
        + 0.01414232 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # +1.4%  log_sum_pt > 7.017
        + 0.01354473 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) / 0.3666529   # +1.4%  z_dr_0_0p05 < 0.8459
        - 0.01326924 * max(0.0, Q.sj2_dr - 0.1512157) / 0.05988199   # -1.3%  sj2_dr > 0.1512
        - 0.01293929 * max(0.0, 0.02647293 - Q.z_dr_0p2_0p4) / 0.01168948   # -1.3%  z_dr_0p2_0p4 < 0.02647
        + 0.01264196 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +1.3%  z_dr_0p1_0p2 < 0.1203
        + 0.01216081 * max(0.0, 0.04748396 - Q.z_dr_0p1_0p2) / 0.01200196   # +1.2%  z_dr_0p1_0p2 < 0.04748
        - 0.0107192 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) / 3.639534   # -1.1%  n_dr_0p2_0p4 > 8
        - 0.009704648 * max(0.0, 0.003213724 - Q.sum_z_dr2_top10) / 0.0009423838   # -1.0%  sum_z_dr2_top10 < 0.003214
        + 0.009190663 * max(0.0, 0.004169954 - Q.sum_z_dr2_top15) / 0.001130716   # +0.9%  sum_z_dr2_top15 < 0.00417
        - 0.008434353 * max(0.0, 0.04748396 - Q.z_dr_0p1_0p2) * max(0.0, 0.03105164 - Q.absphi_1) / 0.0002432398   # -0.8%  z_dr_0p1_0p2 < 0.04748 and absphi_1 < 0.03105
        - 0.008362522 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, 689.25 - Q.sum_pt_top2) / 1.278535   # -0.8%  sum_z_dr2_top5 < 0.00833 and sum_pt_top2 < 689.2
        - 0.007220842 * max(0.0, Q.sum_pt_top30 - 1038.262) / 23.89568   # -0.7%  sum_pt_top30 > 1038
        - 0.007010973 * max(0.0, Q.tau1 - 0.1953848) / 0.000822026   # -0.7%  tau1 > 0.1954
        + 0.006673032 * max(0.0, Q.psi_0p1 - 0.9184255) * max(0.0, 0.03105164 - Q.absphi_1) / 0.0003613644   # +0.7%  psi_0p1 > 0.9184 and absphi_1 < 0.03105
        - 0.006646322 * max(0.0, Q.n_dr_0p05_0p1 - 6.0) / 6.42277   # -0.7%  n_dr_0p05_0p1 > 6
        - 0.005932859 * max(0.0, 0.001155057 - Q.sum_z_dr2_top3) / 0.000329166   # -0.6%  sum_z_dr2_top3 < 0.001155
        + 0.004782621 * max(0.0, 2.410481 - Q.D2) * max(0.0, 0.199671 - Q.dr_11) / 0.06314657   # +0.5%  D2 < 2.41 and dr_11 < 0.1997
        - 0.004531337 * max(0.0, 0.1881908 - Q.sd_rg) * max(0.0, Q.eta_1 - -0.04302979) / 0.003190231   # -0.5%  sd_rg < 0.1882 and eta_1 > -0.04303
        - 0.004360144 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 1.480811 - Q.soft5_pt) / 0.02368753   # -0.4%  z_dr_0p1_0p2 < 0.1203 and soft5_pt < 1.481
        - 0.004256917 * max(0.0, Q.sj2_dr - 0.2780918) / 0.009227968   # -0.4%  sj2_dr > 0.2781
        - 0.004154894 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, Q.sj3_pairmin_over_m - 0.1581667) / 0.00375618   # -0.4%  sj2_dr > 0.2232 and sj3_pairmin_over_m > 0.1582
        + 0.003452079 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, 40.0 - Q.n_real_top40) / 0.02186912   # +0.3%  sum_z_dr2_top5 < 0.00833 and n_real_top40 < 40
        + 0.002258298 * max(0.0, 0.0004140594 - Q.soft5_z) / 1.144764e-05   # +0.2%  soft5_z < 0.0004141
        + 0.002057552 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 0.4823776 - Q.tau21_b2) / 0.006691474   # +0.2%  z_dr_0p1_0p2 < 0.1203 and tau21_b2 < 0.4824
        - 0.001971001 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, Q.sj3_dr23 - 0.2563461) / 0.002209452   # -0.2%  sj2_dr > 0.2232 and sj3_dr23 > 0.2563
        + 0.001874754 * max(0.0, Q.log_sum_pt - 7.017258) * max(0.0, 6.70332 - Q.soft10_pt) / 0.06588738   # +0.2%  log_sum_pt > 7.017 and soft10_pt < 6.703
        - 0.001821916 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.03006824 - Q.dr_4) / 0.000106252   # -0.2%  sj2_dr > 0.2232 and dr_4 < 0.03007
        - 0.001812781 * max(0.0, 0.5297852 - Q.soft5_pt) / 0.01859865   # -0.2%  soft5_pt < 0.5298
        + 0.001473196 * max(0.0, Q.log_sum_pt - 7.017258) * max(0.0, 2.275391 - Q.soft1_pt) / 0.02241794   # +0.1%  log_sum_pt > 7.017 and soft1_pt < 2.275
        + 0.001422555 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, Q.sj3_pairmin_over_m - 0.2777088) / 0.001760992   # +0.1%  sj2_dr > 0.2232 and sj3_pairmin_over_m > 0.2777
        + 0.00117374 * max(0.0, Q.psi_0p1 - 0.707925) * max(0.0, Q.dr1_11 - 0.2195171) / 0.000707107   # +0.1%  psi_0p1 > 0.7079 and dr1_11 > 0.2195
        - 0.0009728855 * max(0.0, 0.08068193 - Q.sum_z_dr) * max(0.0, Q.dr1_11 - 0.1911348) / 0.0001794349   # -0.1%  sum_z_dr < 0.08068 and dr1_11 > 0.1911
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6391830882352941, 2.756777731092437, 0.2114468487394958, 0.37984306722689076, 0.7653981092436974, 1.032876575630252, 0.5340361344537815, 0.45530273109243696, 1.1611585084033613, 0.9815495798319328, 1.2506364495798319, 0.7601011554621848, 0.545872268907563, 1.526819275210084, 0.2556149159663866, 0.43368193277310924]
T = [3.773686516872374, 2.440857392331932, 4.1603317013524155, 4.267536180409664, 4.062664741169906]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +46%, n4 -18%, n9 +13%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.456579 * h[1] / H_AVG[1]
            - 0.1774719 * h[4] / H_AVG[4]
            + 0.1341159 * h[9] / H_AVG[9]
            - 0.08553279 * h[5] / H_AVG[5]
            - 0.07549178 * h[3] / H_AVG[3]
            + 0.03390287 * h[12] / H_AVG[12]
            + 0.02211184 * h[6] / H_AVG[6]
            + 0.009615585 * h[8] / H_AVG[8]
            - 0.005178277 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +23%, n1 -21%, n11 -8%, n12 +8%, n6 +5% ...
            - 0.3331762 * h[4] / H_AVG[4]
            + 0.2261999 * h[9] / H_AVG[9]
            - 0.2117681 * h[1] / H_AVG[1]
            - 0.07785186 * h[11] / H_AVG[11]
            + 0.0768761 * h[12] / H_AVG[12]
            + 0.0478604 * h[6] / H_AVG[6]
            + 0.01082851 * h[2] / H_AVG[2]
            - 0.008005873 * h[10] / H_AVG[10]
            + 0.007433086 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +14%, n0 +12%, n11 +11%, n14 -8%, n7 -7% ...
            - 0.2442146 * h[8] / H_AVG[8]
            + 0.1435298 * h[5] / H_AVG[5]
            + 0.1152281 * h[0] / H_AVG[0]
            + 0.1084793 * h[11] / H_AVG[11]
            - 0.08448137 * h[14] / H_AVG[14]
            - 0.0683994 * h[7] / H_AVG[7]
            + 0.0632415 * h[4] / H_AVG[4]
            - 0.05330359 * h[12] / H_AVG[12]
            - 0.05160982 * h[9] / H_AVG[9]
            + 0.03994425 * h[3] / H_AVG[3]
            - 0.0195454 * h[15] / H_AVG[15]
            + 0.00802274 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -21%, n5 +17%, n6 -12%, n7 +10%, n15 +6% ...
            - 0.2550854 * h[8] / H_AVG[8]
            - 0.2059448 * h[0] / H_AVG[0]
            + 0.1663964 * h[5] / H_AVG[5]
            - 0.1212286 * h[6] / H_AVG[6]
            + 0.09668766 * h[7] / H_AVG[7]
            + 0.05716321 * h[15] / H_AVG[15]
            + 0.03997273 * h[12] / H_AVG[12]
            + 0.03894082 * h[3] / H_AVG[3]
            - 0.01858041 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +30%, n5 -12%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3405843 * h[13] / H_AVG[13]
            + 0.3030265 * h[10] / H_AVG[10]
            - 0.1191732 * h[5] / H_AVG[5]
            + 0.06028848 * h[8] / H_AVG[8]
            - 0.05038617 * h[12] / H_AVG[12]
            - 0.04003056 * h[15] / H_AVG[15]
            + 0.03532463 * h[4] / H_AVG[4]
            - 0.03151968 * h[7] / H_AVG[7]
            + 0.01966637 * h[0] / H_AVG[0]
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
