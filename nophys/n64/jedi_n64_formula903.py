"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_12                 pT share × ΔR of particle 12 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
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
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
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
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.z_dr_0p4_up            pT share of the particles with 0.4 ≤ ΔR < 10
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_phi2              pT-weighted mean Δφ²
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
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
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
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        z_dr_0p4_up=sum(z[i] for i in real if 0.4 <= dr[i] < 10),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
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
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = 1.248064
    if Q.mass < 53.87362:
        z += 0.03011862 * Q.mass - 3.479154
    if 53.87362 <= Q.mass < 74.25181:
        z += 0.03935372 * Q.mass - 3.976682
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.01542954 * Q.mass - 2.200268
    if 78.26182 <= Q.mass < 87.36377:
        z += -0.1120387 * Q.mass + 7.775627
    if 87.36377 <= Q.mass < 91.03469:
        z += -0.1214982 * Q.mass + 8.602049
    if 91.03469 <= Q.mass < 92.85979:
        z += 0.05452819 * Q.mass - 7.422463
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.2412375 * Q.mass + 20.04228
    if Q.mass >= 101.0497:
        z += -0.2805912 * Q.mass + 24.01896
    if Q.sum_pt < 972.0419:
        z += 0.01384313 * Q.sum_pt - 13.78409
    if 972.0419 <= Q.sum_pt < 1012.673:
        z += 0.008072315 * Q.sum_pt - 8.174614
    if Q.psi_0p3 >= 0.9956185:
        z += 62.58381 * Q.psi_0p3 - 62.3096
    if Q.log_sum_pt < 6.97212:
        z += -5.253175 * Q.log_sum_pt + 36.8578
    if 6.97212 <= Q.log_sum_pt < 6.98945:
        z += -2.200457 * Q.log_sum_pt + 15.57388
    if 6.98945 <= Q.log_sum_pt < 7.017258:
        z += -6.973002 * Q.log_sum_pt + 48.93135
    if Q.mass_top30 < 76.41544:
        z += -0.01284323 * Q.mass_top30 + 0.958568
    if 76.41544 <= Q.mass_top30 < 80.24626:
        z += 0.005965534 * Q.mass_top30 - 0.4787118
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.03772601 * Q.n_dr_0p2_0p4 + 0.6238977
    if 11.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += -0.0522279 * Q.n_dr_0p2_0p4 + 0.7834185
    if Q.sum_z_dr2_top30 < 0.006363916:
        z += 163.0007 * Q.sum_z_dr2_top30 - 0.9713597
    if 0.006363916 <= Q.sum_z_dr2_top30 < 0.007856958:
        z += -44.18011 * Q.sum_z_dr2_top30 + 0.3471213
    if Q.lam1 < 0.005913555:
        z += 53.78912 * Q.lam1 - 0.3180849
    if Q.mass_over_sum_pt_sq < 0.004754444:
        z += -779.1221 * Q.mass_over_sum_pt_sq + 5.644603
    if 0.004754444 <= Q.mass_over_sum_pt_sq < 0.006938798:
        z += -726.8872 * Q.mass_over_sum_pt_sq + 5.396255
    if 0.006938798 <= Q.mass_over_sum_pt_sq < 0.007873266:
        z += -377.2538 * Q.mass_over_sum_pt_sq + 2.970219
    if Q.tau1 < 0.0705748:
        z += 19.05416 * Q.tau1 - 1.344744
    if Q.sum_zz_dr2 < 0.00616708:
        z += 336.0671 * Q.sum_zz_dr2 - 2.072553
    if Q.e2 < 0.01879315:
        z += -26.24128 * Q.e2 + 0.4931562
    if Q.e2 >= 0.05557149:
        z += -16.67409 * Q.e2 + 0.9266039
    if 71.79516 <= Q.mass_top50 < 82.04491:
        z += 0.01861072 * Q.mass_top50 - 1.336159
    if 82.04491 <= Q.mass_top50 < 92.16545:
        z += -0.04945338 * Q.mass_top50 + 4.248153
    if Q.mass_top50 >= 92.16545:
        z += 0.01726189 * Q.mass_top50 - 1.90069
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += 5.293653 * Q.z_dr_0p2_0p4 - 0.4829171
    if Q.z_top50_slots < 0.9906378:
        z += 13.87194 * Q.z_top50_slots - 13.74207
    if Q.sum_pt_top40 < 858.8262:
        z += 0.0001179475 * Q.sum_pt_top40 - 0.5374227
    if 858.8262 <= Q.sum_pt_top40 < 1069.671:
        z += 0.002068469 * Q.sum_pt_top40 - 2.212582
    if Q.mass_top40 < 80.89043:
        z += 0.006133797 * Q.mass_top40 - 0.5717364
    if 80.89043 <= Q.mass_top40 < 83.32554:
        z += 0.03103396 * Q.mass_top40 - 2.585921
    if Q.sum_z_dr2_top20 < 0.006374178:
        z += 86.77341 * Q.sum_z_dr2_top20 - 0.5531091
    if Q.e3 < 0.0005178279:
        z += -533.9315 * Q.e3 + 0.2764846
    if Q.C2 < 0.05602756:
        z += 6.279384 * Q.C2 - 0.3518186
    if Q.lam2 < 0.0006154841:
        z += -277.4536 * Q.lam2 + 0.1707683
    if Q.sum_pt_top50 < 1156.659:
        z += -0.0009756712 * Q.sum_pt_top50 + 1.128519
    if 0.1411617 <= Q.sj2_dr < 0.2232169:
        z += 1.421563 * Q.sj2_dr - 0.2006703
    if Q.sj2_dr >= 0.2232169:
        z += -2.185369 * Q.sj2_dr + 0.6044578
    if Q.sum_pt_top30 < 950.9324:
        z += -0.001661131 * Q.sum_pt_top30 + 1.579623
    if Q.log_sum_pt < 6.98945 and Q.sum_pt_top50 > 959.0957:
        z += 0.05415536 * (6.98945 - Q.log_sum_pt) * (Q.sum_pt_top50 - 959.0957)
    if Q.z_dr_0p2_0p4 < 0.09122568 and Q.pt_2 < 118.5:
        z += 0.02377992 * (0.09122568 - Q.z_dr_0p2_0p4) * (118.5 - Q.pt_2)
    if Q.sum_pt < 1012.673 and Q.dr_9 > 0.04680031:
        z += 0.01245158 * (1012.673 - Q.sum_pt) * (Q.dr_9 - 0.04680031)
    if Q.sum_zz_dr2 < 0.00616708 and Q.z_dr_0p05_0p1 > 0.02069735:
        z += -337.0829 * (0.00616708 - Q.sum_zz_dr2) * (Q.z_dr_0p05_0p1 - 0.02069735)
    if Q.mass_over_sum_pt_sq < 0.007873266 and Q.z_dr_0p05_0p1 > 0.01490648:
        z += 122.3054 * (0.007873266 - Q.mass_over_sum_pt_sq) * (Q.z_dr_0p05_0p1 - 0.01490648)
    if Q.mass > 78.26182 and Q.sj2_zsoft < 0.06909411:
        z += 2.487148 * (Q.mass - 78.26182) * (0.06909411 - Q.sj2_zsoft)
    if Q.mass_top50 > 71.79516 and Q.sj2_zsoft < 0.06909411:
        z += -1.580749 * (Q.mass_top50 - 71.79516) * (0.06909411 - Q.sj2_zsoft)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.495947
    if Q.n_particles >= 38.0:
        z += 0.03458814 * Q.n_particles - 1.314349
    if Q.log_sum_pt < 6.893714:
        z += 8.046852 * Q.log_sum_pt - 57.44886
    if 6.893714 <= Q.log_sum_pt < 6.910131:
        z += 34.47278 * Q.log_sum_pt - 239.6217
    if 6.910131 <= Q.log_sum_pt < 6.959294:
        z += 51.96581 * Q.log_sum_pt - 360.5008
    if 6.959294 <= Q.log_sum_pt < 6.98945:
        z += 33.62054 * Q.log_sum_pt - 232.8307
    if 6.98945 <= Q.log_sum_pt < 7.017258:
        z += 29.93658 * Q.log_sum_pt - 207.0818
    if 7.017258 <= Q.log_sum_pt < 7.139296:
        z += 19.51674 * Q.log_sum_pt - 133.9631
    if Q.log_sum_pt >= 7.139296:
        z += 11.46989 * Q.log_sum_pt - 76.51421
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.009970071 * Q.sum_pt_top50 + 9.562252
    if Q.psi_0p3 >= 0.9980008:
        z += -159.8194 * Q.psi_0p3 + 159.4998
    if Q.sum_pt_top2 < 689.25:
        z += -0.001876104 * Q.sum_pt_top2 + 1.293105
    if 0.9341838 <= Q.z_top30_slots < 0.9564984:
        z += -3.98807 * Q.z_top30_slots + 3.725591
    if Q.z_top30_slots >= 0.9564984:
        z += 4.93925 * Q.z_top30_slots - 4.813377
    if Q.mass_top20 < 47.88842:
        z += 0.05329029 * Q.mass_top20 - 2.551988
    if Q.sj3_mass1 < 32.50209:
        z += 0.01654806 * Q.sj3_mass1 - 0.5378465
    if Q.sum_pt_top40 < 1069.671:
        z += -0.00751897 * Q.sum_pt_top40 + 8.042826
    if Q.sum_z_dr2_top15 < 0.0007894752:
        z += -552.6265 * Q.sum_z_dr2_top15 + 0.5531744
    if 0.0007894752 <= Q.sum_z_dr2_top15 < 0.003270031:
        z += -47.12225 * Q.sum_z_dr2_top15 + 0.1540912
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.09079704 * Q.n_dr_0p2_0p4 - 0.8126606
    if 7.0 <= Q.n_dr_0p2_0p4 < 13.0:
        z += 0.02951356 * Q.n_dr_0p2_0p4 - 0.3836763
    if Q.M3 < 0.03187688:
        z += 8.889705 * Q.M3 - 0.2833761
    if Q.M3 >= 0.04813493:
        z += -9.541511 * Q.M3 + 0.45928
    if Q.sj2_mass1 < 30.26161:
        z += 0.02021945 * Q.sj2_mass1 - 0.6118729
    if Q.soft5_z < 0.001594761:
        z += -120.5255 * Q.soft5_z + 0.1922095
    if Q.pt_9 < 31.35938:
        z += 0.01712108 * Q.pt_9 - 0.5369065
    if Q.lam1 < 0.004673423:
        z += -143.0874 * Q.lam1 + 0.6687079
    if Q.mass < 89.74183:
        z += 0.0162012 * Q.mass - 2.282106
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.0502729 * Q.mass - 5.339764
    if 101.0497 <= Q.mass < 121.3913:
        z += 0.01276703 * Q.mass - 1.549806
    if Q.N2 >= 0.4678622:
        z += -5.736668 * Q.N2 + 2.68397
    if Q.D3 < 0.08822608:
        z += -1.760777 * Q.D3 + 0.1553464
    if Q.z_dr_0_0p05 >= 0.8459004:
        z += -4.549689 * Q.z_dr_0_0p05 + 3.848584
    if Q.soft1_pt < 1.091797:
        z += -0.2634174 * Q.soft1_pt + 0.212662
    if 1.091797 <= Q.soft1_pt < 1.521582:
        z += -0.7733589 * Q.soft1_pt + 0.7694146
    if 1.521582 <= Q.soft1_pt < 2.275391:
        z += 0.5403421 * Q.soft1_pt - 1.229489
    if Q.sj2_zsoft >= 0.2912328:
        z += 0.8282939 * Q.sj2_zsoft - 0.2412263
    if Q.sum_pt < 1017.435:
        z += -0.01062563 * Q.sum_pt + 10.81089
    if Q.sum_pt_top30 >= 1191.938:
        z += 0.004774386 * Q.sum_pt_top30 - 5.690771
    if Q.mass_top30 < 80.24626:
        z += -0.01196458 * Q.mass_top30 + 0.9601125
    if Q.n_dr_0_0p05 < 10.0:
        z += -0.02575871 * Q.n_dr_0_0p05 + 0.2575871
    if Q.tau1 < 0.1751567:
        z += -8.846537 * Q.tau1 + 1.54953
    if Q.mass_top10 >= 31.33272:
        z += 0.004821755 * Q.mass_top10 - 0.1510787
    if Q.zdr_0 >= 0.001901263:
        z += -12.7117 * Q.zdr_0 + 0.02416829
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.04003619 * Q.n_dr_0p1_0p2 - 0.3202895
    if Q.sum_pt_top3 >= 353.0625:
        z += -0.0007495903 * Q.sum_pt_top3 + 0.2646522
    if Q.sum_z_dr2_top5 < 0.00100099:
        z += -529.083 * Q.sum_z_dr2_top5 + 0.9615773
    if 0.00100099 <= Q.sum_z_dr2_top5 < 0.008329695:
        z += -58.94226 * Q.sum_z_dr2_top5 + 0.490971
    if Q.sum_z_dr2_top5 >= 0.0168592:
        z += -12.58366 * Q.sum_z_dr2_top5 + 0.2121505
    if Q.mass_top5 < 14.54404:
        z += 0.01219519 * Q.mass_top5 - 0.4101109
    if 14.54404 <= Q.mass_top5 < 68.43422:
        z += 0.004318849 * Q.mass_top5 - 0.2955571
    if Q.tau2 < 0.0795038:
        z += -3.176481 * Q.tau2 + 0.2525423
    if Q.sj3_dr_max >= 0.3483732:
        z += 1.771839 * Q.sj3_dr_max - 0.6172613
    if Q.eccentricity >= 0.6414784:
        z += -0.544343 * Q.eccentricity + 0.3491842
    if Q.m012 < 50.3522:
        z += 0.004040269 * Q.m012 - 0.2034364
    if Q.sum_z_dr2_top20 < 0.001823079:
        z += -348.0049 * Q.sum_z_dr2_top20 + 0.6344405
    if Q.sum_z_dr < 0.0101072:
        z += -77.50672 * Q.sum_z_dr + 0.7833759
    if Q.psi_0p1 >= 0.8747961:
        z += 1.286682 * Q.psi_0p1 - 1.125584
    if Q.z_dr_0p2_0p4 < 0.1291856:
        z += -2.365945 * Q.z_dr_0p2_0p4 + 0.305646
    if Q.z_top50_slots >= 0.9586536:
        z += -21.47092 * Q.z_top50_slots + 20.58318
    if Q.C2_b2 < 0.003468228:
        z += 56.62374 * Q.C2_b2 - 0.1963841
    if Q.tau21 < 0.3100459:
        z += -1.119377 * Q.tau21 + 0.3470584
    if Q.mass_top40 < 48.60659:
        z += -0.01365699 * Q.mass_top40 + 0.6638196
    if Q.z_7 < 0.03107249:
        z += 13.05142 * Q.z_7 - 0.4055401
    if Q.pt_11 < 33.78125:
        z += 0.01132114 * Q.pt_11 - 0.3824422
    if Q.tau3 < 0.03209934:
        z += -10.0667 * Q.tau3 + 0.3231345
    if Q.sd_rg >= 0.3017146:
        z += 2.815392 * Q.sd_rg - 0.8494448
    if Q.n_particles > 38.0 and Q.mass_top15 < 57.87349:
        z += 0.0002402769 * (Q.n_particles - 38.0) * (57.87349 - Q.mass_top15)
    if Q.z_top30_slots > 0.9341838 and Q.max_pair_mass > 13.04793:
        z += 0.3406361 * (Q.z_top30_slots - 0.9341838) * (Q.max_pair_mass - 13.04793)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.1119555:
        z += 0.1223778 * (Q.n_particles - 38.0) * (0.1119555 - Q.dr_0)
    if Q.mass_top20 < 47.88842 and Q.n_real_top40 > 29.0:
        z += 0.004177831 * (47.88842 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 38.0 and Q.soft1_pt < 2.275391:
        z += -0.02120746 * (Q.n_particles - 38.0) * (2.275391 - Q.soft1_pt)
    if Q.z_top30_slots > 0.9341838 and Q.C2 < 0.07279889:
        z += 229.6729 * (Q.z_top30_slots - 0.9341838) * (0.07279889 - Q.C2)
    if Q.sj3_mass1 < 32.50209 and Q.sj3_mass2 < 18.68222:
        z += -0.0013573 * (32.50209 - Q.sj3_mass1) * (18.68222 - Q.sj3_mass2)
    if Q.sum_z_dr2_top15 < 0.003270031 and Q.psi_0p3 > 0.9973959:
        z += 46229.49 * (0.003270031 - Q.sum_z_dr2_top15) * (Q.psi_0p3 - 0.9973959)
    if Q.M3 < 0.03187688 and Q.M2 > 0.05568888:
        z += -271.2079 * (0.03187688 - Q.M3) * (Q.M2 - 0.05568888)
    if Q.n_particles > 38.0 and Q.dr_1 < 0.1611545:
        z += 0.1506256 * (Q.n_particles - 38.0) * (0.1611545 - Q.dr_1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 > 7.407874:
        z += 0.4109186 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_3 - 7.407874)
    if Q.n_particles > 38.0 and Q.mean_phi2 < 0.0174229:
        z += 0.5131763 * (Q.n_particles - 38.0) * (0.0174229 - Q.mean_phi2)
    if Q.pt_9 < 31.35938 and Q.pair_mass_0_12 < 15.47191:
        z += 0.0007768938 * (31.35938 - Q.pt_9) * (15.47191 - Q.pair_mass_0_12)
    if Q.n_particles > 38.0 and Q.zdr_2 < 0.01395978:
        z += 1.630592 * (Q.n_particles - 38.0) * (0.01395978 - Q.zdr_2)
    if Q.sum_pt_top2 < 689.25 and Q.tau43 < 0.9624339:
        z += -0.0009631517 * (689.25 - Q.sum_pt_top2) * (0.9624339 - Q.tau43)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.n_real_top30 < 30.0:
        z += 0.007265044 * (7.0 - Q.n_dr_0p2_0p4) * (30.0 - Q.n_real_top30)
    if Q.n_particles > 38.0 and Q.sj3_dr12 < 0.2898296:
        z += 0.03809988 * (Q.n_particles - 38.0) * (0.2898296 - Q.sj3_dr12)
    if Q.z_top30_slots > 0.9564984 and Q.ptdr0_4 > 4.470953:
        z += 0.5427575 * (Q.z_top30_slots - 0.9564984) * (Q.ptdr0_4 - 4.470953)
    if Q.z_top30_slots > 0.9564984 and Q.sj3_mass3 < 7.880676:
        z += -0.8517359 * (Q.z_top30_slots - 0.9564984) * (7.880676 - Q.sj3_mass3)
    if Q.sum_pt_top2 < 689.25 and Q.soft3_dr < 0.2792343:
        z += -0.001780573 * (689.25 - Q.sum_pt_top2) * (0.2792343 - Q.soft3_dr)
    if Q.soft1_pt < 1.521582 and Q.soft4_dr < 0.2664362:
        z += -0.627151 * (1.521582 - Q.soft1_pt) * (0.2664362 - Q.soft4_dr)
    if Q.sj3_mass1 < 32.50209 and Q.soft4_dr0 < 0.3343369:
        z += 0.02607718 * (32.50209 - Q.sj3_mass1) * (0.3343369 - Q.soft4_dr0)
    if Q.n_particles > 38.0 and Q.dr_9 < 0.1546554:
        z += 0.04562647 * (Q.n_particles - 38.0) * (0.1546554 - Q.dr_9)
    if Q.sum_pt_top2 < 689.25 and Q.soft3_dr0 < 0.3425105:
        z += 0.001147709 * (689.25 - Q.sum_pt_top2) * (0.3425105 - Q.soft3_dr0)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.05624447
    if Q.log_sum_pt >= 7.062574:
        z += -5.223949 * Q.log_sum_pt + 36.89453
    if Q.sum_pt < 1017.435:
        z += 0.002379939 * Q.sum_pt - 3.000011
    if 1017.435 <= Q.sum_pt < 1052.889:
        z += 0.01281753 * Q.sum_pt - 13.61958
    if 1052.889 <= Q.sum_pt < 1115.723:
        z += 0.006279889 * Q.sum_pt - 6.736164
    if 1115.723 <= Q.sum_pt < 1260.541:
        z += 0.002330893 * Q.sum_pt - 2.33018
    if Q.sum_pt >= 1260.541:
        z += -4.904624e-05 * Q.sum_pt + 0.6698311
    if Q.n_particles < 51.0:
        z += 0.01152654 * Q.n_particles - 0.5878533
    if Q.sum_pt_top20 >= 1129.275:
        z += 0.002383886 * Q.sum_pt_top20 - 2.692063
    if 988.4554 <= Q.sum_pt_top50 < 1156.659:
        z += 0.001706567 * Q.sum_pt_top50 - 1.686865
    if Q.sum_pt_top50 >= 1156.659:
        z += 0.005961988 * Q.sum_pt_top50 - 6.608939
    if Q.z_top20_slots >= 0.7516206:
        z += 0.7672358 * Q.z_top20_slots - 0.5766702
    if Q.mass < 78.26182:
        z += 0.0147962 * Q.mass - 0.8236537
    if 78.26182 <= Q.mass < 91.03469:
        z += 0.02299958 * Q.mass - 1.465665
    if 91.03469 <= Q.mass < 92.85979:
        z += 0.043006 * Q.mass - 3.286943
    if 92.85979 <= Q.mass < 121.3913:
        z += -0.01806239 * Q.mass + 2.383855
    if 121.3913 <= Q.mass < 143.7876:
        z += -0.008538816 * Q.mass + 1.227776
    if Q.mass_over_sum_pt < 0.09795415:
        z += -5.92463 * Q.mass_over_sum_pt + 0.580342
    if Q.sum_pt_top30 < 996.8867:
        z += 0.001015522 * Q.sum_pt_top30 - 1.01236
    if Q.sum_z_dr2_top30 < 0.006088416:
        z += 15.13637 * Q.sum_z_dr2_top30 - 0.09215652
    if Q.sum_pt_top3 < 787.6281:
        z += -0.0002540325 * Q.sum_pt_top3 + 0.2000831
    if Q.sum_pt_top15 < 1003.329:
        z += 0.000701469 * Q.sum_pt_top15 - 0.7038043
    if Q.sum_z_dr2_top50 < 0.008124776:
        z += -79.71175 * Q.sum_z_dr2_top50 + 0.6476401
    if Q.mass_top40 < 83.32554:
        z += -0.004562029 * Q.mass_top40 + 0.1429413
    if 83.32554 <= Q.mass_top40 < 132.4278:
        z += 0.004830571 * Q.mass_top40 - 0.6397022
    if Q.lam2 < 0.001776308:
        z += 138.3762 * Q.lam2 - 0.2457988
    if Q.lam1 < 0.006716737:
        z += 58.10173 * Q.lam1 - 0.390254
    if Q.sum_pt_top40 < 1053.047:
        z += -0.003453337 * Q.sum_pt_top40 + 3.636528
    if Q.log_sum_pt > 6.903423 and Q.mass_top40 > 77.93668:
        z += -0.0238292 * (Q.log_sum_pt - 6.903423) * (Q.mass_top40 - 77.93668)
    if Q.log_sum_pt > 6.903423 and Q.sum_z_dr2_top15 < 0.02146578:
        z += 595.3438 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.sum_z_dr2_top15)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p3 > 0.9299135:
        z += -41.5198 * (Q.log_sum_pt - 6.903423) * (Q.psi_0p3 - 0.9299135)
    if Q.sum_pt_top50 > 988.4554 and Q.sum_z_dr2_top15 < 0.02146578:
        z += -0.4580032 * (Q.sum_pt_top50 - 988.4554) * (0.02146578 - Q.sum_z_dr2_top15)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.2185892
    if Q.lam2 < 0.0006154841:
        z += -514.8233 * Q.lam2 + 0.3168655
    if Q.n_particles < 46.0:
        z += -0.02273683 * Q.n_particles + 1.045894
    if Q.tau21 < 0.3861957:
        z += 1.394783 * Q.tau21 - 0.5386592
    if Q.D2 < 2.410481:
        z += 0.0934816 * Q.D2 - 0.2253356
    if Q.mass_over_sum_pt < 0.08665515:
        z += -15.38164 * Q.mass_over_sum_pt + 1.332898
    if Q.z_top30_slots >= 0.9734886:
        z += -5.903654 * Q.z_top30_slots + 5.74714
    if Q.sum_z_dr2_top40 < 0.006026828:
        z += -45.55582 * Q.sum_z_dr2_top40 + 0.4196097
    if 0.006026828 <= Q.sum_z_dr2_top40 < 0.006259772:
        z += 68.7376 * Q.sum_z_dr2_top40 - 0.2692171
    if 0.006259772 <= Q.sum_z_dr2_top40 < 0.008840538:
        z += -62.40961 * Q.sum_z_dr2_top40 + 0.5517346
    if Q.mass_top50 < 79.21004:
        z += 0.008132284 * Q.mass_top50 - 0.6441585
    if Q.sum_z_dr2 < 0.009614971:
        z += 129.7087 * Q.sum_z_dr2 - 1.247145
    if Q.mass < 53.87362:
        z += -0.003236213 * Q.mass + 0.8811621
    if 53.87362 <= Q.mass < 92.85979:
        z += -0.0181299 * Q.mass + 1.683539
    if Q.sum_z_dr2_top50 < 0.008124776:
        z += -44.50956 * Q.sum_z_dr2_top50 + 0.3616302
    if Q.lam1 < 0.001868041:
        z += -162.0066 * Q.lam1 + 0.3026349
    if Q.mass_top40 < 80.89043:
        z += 0.01983201 * Q.mass_top40 - 1.60422
    if Q.z_dr_0p2_0p4 < 0.019523:
        z += 5.067622 * Q.z_dr_0p2_0p4 - 0.09893519
    if Q.D2_b2 < 1.36316:
        z += -0.2086122 * Q.D2_b2 + 0.2843719
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_dr_0p1_0p2 > 9.0:
        z += -0.00860427 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 9.0)
    if Q.n_particles < 46.0 and Q.sum_pt_top30 > 800.732:
        z += 3.236561e-05 * (46.0 - Q.n_particles) * (Q.sum_pt_top30 - 800.732)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.eccentricity > 0.7333655:
        z += 0.1477998 * (5.0 - Q.n_dr_0p2_0p4) * (Q.eccentricity - 0.7333655)
    if Q.D2 < 2.410481 and Q.psi_0p3 > 0.9985421:
        z += 87.13229 * (2.410481 - Q.D2) * (Q.psi_0p3 - 0.9985421)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.zdr_5 < 0.007099471:
        z += -7.220709 * (5.0 - Q.n_dr_0p2_0p4) * (0.007099471 - Q.zdr_5)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.psi_0p1 < 0.9538343:
        z += 0.3475895 * (5.0 - Q.n_dr_0p2_0p4) * (0.9538343 - Q.psi_0p1)
    if Q.tau21 < 0.3861957 and Q.lam1 > 0.007671243:
        z += -251.4259 * (0.3861957 - Q.tau21) * (Q.lam1 - 0.007671243)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.z_top50_slots > 0.978741:
        z += 3.311469 * (8.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.978741)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.n_dr_0_0p05 < 18.0:
        z += -0.002163019 * (8.0 - Q.n_dr_0p2_0p4) * (18.0 - Q.n_dr_0_0p05)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.z_dr_0p05_0p1 > 0.5106729:
        z += 0.1526179 * (5.0 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p05_0p1 - 0.5106729)
    if Q.n_dr_0p1_0p2 < 10.0 and Q.psi_0p2 > 0.9087063:
        z += 0.2797157 * (10.0 - Q.n_dr_0p1_0p2) * (Q.psi_0p2 - 0.9087063)
    if Q.sum_z_dr2 < 0.009614971 and Q.sj3_mass1 < 26.76006:
        z += 1.086159 * (0.009614971 - Q.sum_z_dr2) * (26.76006 - Q.sj3_mass1)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.psi_0p3 > 0.9924477:
        z += 12.09257 * (5.0 - Q.n_dr_0p2_0p4) * (Q.psi_0p3 - 0.9924477)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.1584707
    if Q.mass_top15 < 57.87349:
        z += -0.009096852 * Q.mass_top15 + 0.5712379
    if 57.87349 <= Q.mass_top15 < 69.02716:
        z += -0.004014045 * Q.mass_top15 + 0.2770782
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.04144143 * Q.n_dr_0p2_0p4 + 0.7035496
    if 15.0 <= Q.n_dr_0p2_0p4 < 26.0:
        z += -0.007448015 * Q.n_dr_0p2_0p4 + 0.1936484
    if Q.mass_top40 < 67.72643:
        z += -0.01460782 * Q.mass_top40 + 1.200758
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += -0.027586 * Q.mass_top40 + 2.079724
    if 83.32554 <= Q.mass_top40 < 94.64253:
        z += 0.01934209 * Q.mass_top40 - 1.830584
    if Q.mass_top30 < 60.43821:
        z += 0.007029447 * Q.mass_top30 - 0.5020205
    if 60.43821 <= Q.mass_top30 < 121.737:
        z += 0.001258969 * Q.mass_top30 - 0.1532632
    if Q.mass < 74.25181:
        z += 0.06558657 * Q.mass - 5.079637
    if 74.25181 <= Q.mass < 79.65241:
        z += 0.06142522 * Q.mass - 4.77065
    if 79.65241 <= Q.mass < 87.36377:
        z += 0.03185174 * Q.mass - 2.415051
    if 87.36377 <= Q.mass < 92.85979:
        z += -0.01390829 * Q.mass + 1.582719
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.02547068 * Q.mass + 2.6564
    if 101.0497 <= Q.mass < 121.3913:
        z += -0.004060391 * Q.mass + 0.4928961
    if Q.mass >= 143.7876:
        z += 0.02382775 * Q.mass - 3.426135
    if Q.psi_0p3 >= 0.9973959:
        z += 74.69768 * Q.psi_0p3 - 74.50316
    if Q.sj3_pair_mass_min >= 32.51366:
        z += 0.004568458 * Q.sj3_pair_mass_min - 0.1485373
    if Q.n_particles >= 22.0:
        z += -0.02567338 * Q.n_particles + 0.5648144
    if Q.sum_z_dr2_top15 < 0.004855289:
        z += 171.5962 * Q.sum_z_dr2_top15 - 0.8331491
    if Q.sum_z_dr2_top15 >= 0.007887677:
        z += -11.45802 * Q.sum_z_dr2_top15 + 0.09037719
    if Q.n_dr_0_0p05 < 5.0:
        z += 0.01062585 * Q.n_dr_0_0p05 - 0.2656462
    if 5.0 <= Q.n_dr_0_0p05 < 25.0:
        z += 0.01961541 * Q.n_dr_0_0p05 - 0.310594
    if Q.n_dr_0_0p05 >= 25.0:
        z += 0.008989557 * Q.n_dr_0_0p05 - 0.04494778
    if Q.sum_zz_dr2 >= 0.01396296:
        z += 70.54691 * Q.sum_zz_dr2 - 0.9850439
    if Q.sum_pt < 1012.673:
        z += 0.002454025 * Q.sum_pt - 2.582856
    if 1012.673 <= Q.sum_pt < 1042.609:
        z += 0.003264677 * Q.sum_pt - 3.403781
    if Q.sum_z_dr2_top10 < 0.007678544:
        z += 44.11493 * Q.sum_z_dr2_top10 - 0.2114618
    if 0.007678544 <= Q.sum_z_dr2_top10 < 0.008956554:
        z += -99.58966 * Q.sum_z_dr2_top10 + 0.8919801
    if Q.z_dr_0p2_0p4 < 0.02647293:
        z += 7.353047 * Q.z_dr_0p2_0p4 - 0.3370137
    if 0.02647293 <= Q.z_dr_0p2_0p4 < 0.0684915:
        z += 3.387955 * Q.z_dr_0p2_0p4 - 0.2320461
    if Q.n_pt_above_1 < 48.0:
        z += -0.0124613 * Q.n_pt_above_1 + 0.5981424
    if Q.lam2 < 0.0004662705:
        z += 117.6915 * Q.lam2 - 0.2090564
    if 0.0004662705 <= Q.lam2 < 0.001776308:
        z += 148.999 * Q.lam2 - 0.2236541
    if Q.lam2 >= 0.001776308:
        z += 31.30754 * Q.lam2 - 0.01459778
    if Q.D2 < 5.378975:
        z += -0.1624238 * Q.D2 + 1.076742
    if 5.378975 <= Q.D2 < 6.916121:
        z += -0.1321075 * Q.D2 + 0.9136712
    if Q.e2 < 0.02515919:
        z += -10.74659 * Q.e2 + 0.2703756
    if Q.e2 >= 0.03680582:
        z += 11.32836 * Q.e2 - 0.4169495
    if Q.tau21 < 0.4281458:
        z += 0.4469485 * Q.tau21 - 0.1913591
    if Q.psi_0p1 < 0.7703036:
        z += -0.3257311 * Q.psi_0p1 + 0.2509118
    if Q.psi_0p1 >= 0.8747961:
        z += -1.149975 * Q.psi_0p1 + 1.005994
    z += 0.007266554 * Q.n_dr_0p05_0p1
    if Q.mass_top10 < 44.19225:
        z += -0.004371721 * Q.mass_top10 + 0.1931962
    if Q.mass_top10 >= 76.9886:
        z += -0.01000774 * Q.mass_top10 + 0.7704819
    if Q.mass_top50 < 61.65972:
        z += 0.01652071 * Q.mass_top50 - 1.018663
    if Q.mass_top50 >= 138.8977:
        z += -0.01347336 * Q.mass_top50 + 1.871418
    if Q.mass_top20 >= 103.6749:
        z += -0.008456572 * Q.mass_top20 + 0.8767343
    if Q.z_top50_slots < 0.9978182:
        z += 6.326553 * Q.z_top50_slots - 6.31275
    if 0.1512157 <= Q.sj2_dr < 0.1825048:
        z += 4.585963 * Q.sj2_dr - 0.6934696
    if Q.sj2_dr >= 0.1825048:
        z += -1.062903 * Q.sj2_dr + 0.3374758
    if Q.LHA >= 0.3719813:
        z += -8.058196 * Q.LHA + 2.997498
    if Q.mass_over_sum_pt_sq < 0.008184368:
        z += -119.0118 * Q.mass_over_sum_pt_sq + 0.9740366
    if Q.sum_z_dr2_top30 < 0.005809485:
        z += 56.69521 * Q.sum_z_dr2_top30 - 0.3293699
    if Q.tau1 < 0.0705748:
        z += 11.20173 * Q.tau1 - 0.7905596
    if Q.e3 >= 6.567534e-05:
        z += -286.8484 * Q.e3 + 0.01883887
    if Q.lam1 < 0.007671243:
        z += 22.23417 * Q.lam1 - 0.1705637
    if Q.sum_z_dr < 0.03577037:
        z += 4.707036 * Q.sum_z_dr - 0.1683724
    if Q.n_dr_0p2_0p4 < 15.0 and Q.z_top50_slots < 1.0:
        z += -0.4338677 * (15.0 - Q.n_dr_0p2_0p4) * (1.0 - Q.z_top50_slots)
    if Q.mass_top40 < 83.32554 and Q.D2 < 6.916121:
        z += -0.009067223 * (83.32554 - Q.mass_top40) * (6.916121 - Q.D2)
    if Q.n_particles > 22.0 and Q.soft1_pt < 2.275391:
        z += 0.003804847 * (Q.n_particles - 22.0) * (2.275391 - Q.soft1_pt)
    if Q.mass < 87.36377 and Q.zdr_0 > 0.006864207:
        z += -1.840093 * (87.36377 - Q.mass) * (Q.zdr_0 - 0.006864207)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.max_dr > 0.1939977:
        z += -0.04718342 * (15.0 - Q.n_dr_0p2_0p4) * (Q.max_dr - 0.1939977)
    if Q.mass < 79.65241 and Q.lam2 < 0.003687605:
        z += -16.92771 * (79.65241 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.n_dr_0_0p05 > 5.0 and Q.lam2 < 0.001163277:
        z += 14.95273 * (Q.n_dr_0_0p05 - 5.0) * (0.001163277 - Q.lam2)
    if Q.mass_top30 < 121.737 and Q.sum_pt_top40 < 1053.047:
        z += 2.222248e-05 * (121.737 - Q.mass_top30) * (1053.047 - Q.sum_pt_top40)
    if Q.mass < 101.0497 and Q.lam2 < 0.003687605:
        z += 9.059602 * (101.0497 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.sum_z_dr2_top15 < 0.004855289 and Q.n_dr_0p05_0p1 > 2.0:
        z += 4.083073 * (0.004855289 - Q.sum_z_dr2_top15) * (Q.n_dr_0p05_0p1 - 2.0)
    if Q.n_particles > 22.0 and Q.C2_b2 > 0.001126916:
        z += 0.1487505 * (Q.n_particles - 22.0) * (Q.C2_b2 - 0.001126916)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.9192573
    z += -0.01652687 * Q.n_particles + 1.05772
    if 0.08873143 <= Q.mass_over_sum_pt < 0.09046749:
        z += -46.54081 * Q.mass_over_sum_pt + 4.129633
    if Q.mass_over_sum_pt >= 0.09046749:
        z += -6.908257 * Q.mass_over_sum_pt + 0.5441747
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -12.22443 * Q.log_sum_pt + 84.47245
    if 6.920349 <= Q.log_sum_pt < 6.935549:
        z += -25.53987 * Q.log_sum_pt + 176.6199
    if 6.935549 <= Q.log_sum_pt < 6.98945:
        z += -21.60937 * Q.log_sum_pt + 149.3597
    if Q.log_sum_pt >= 6.98945:
        z += -16.6186 * Q.log_sum_pt + 114.477
    if 907.9372 <= Q.sum_pt < 986.0565:
        z += 0.008041933 * Q.sum_pt - 7.30157
    if Q.sum_pt >= 986.0565:
        z += -0.008076653 * Q.sum_pt + 8.592267
    if Q.sum_z_dr2_top50 >= 0.01951641:
        z += -54.59838 * Q.sum_z_dr2_top50 + 1.065564
    if Q.sum_z_dr2 >= 0.004756928:
        z += -52.50812 * Q.sum_z_dr2 + 0.2497774
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += 0.0117368 * Q.sum_pt_top50 - 10.96501
    if Q.sum_pt_top50 >= 959.0957:
        z += 0.01814898 * Q.sum_pt_top50 - 17.1149
    if Q.n_pt_above_1 >= 28.0:
        z += 0.01547216 * Q.n_pt_above_1 - 0.4332206
    if Q.sum_pt_top3 < 787.6281:
        z += 0.0008169666 * Q.sum_pt_top3 - 0.6434659
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.02055568 * Q.n_dr_0p2_0p4 + 0.2261125
    if 994.2695 <= Q.sum_pt_top40 < 1024.942:
        z += 0.004454751 * Q.sum_pt_top40 - 4.429223
    if Q.sum_pt_top40 >= 1024.942:
        z += 0.003273927 * Q.sum_pt_top40 - 3.218946
    if Q.sum_pt_top30 >= 933.1875:
        z += 0.001012127 * Q.sum_pt_top30 - 0.944504
    if Q.z_top30_slots >= 0.9048492:
        z += -3.579277 * Q.z_top30_slots + 3.238706
    if Q.mass_top40 < 111.2487:
        z += 0.01175339 * Q.mass_top40 - 1.763178
    if 111.2487 <= Q.mass_top40 < 150.0144:
        z += -0.005665516 * Q.mass_top40 + 0.1746519
    if Q.mass_top40 >= 150.0144:
        z += -0.01741891 * Q.mass_top40 + 1.93783
    if Q.mass < 64.48544:
        z += -0.01542247 * Q.mass + 0.994525
    if 87.36377 <= Q.mass < 91.03469:
        z += 0.02606951 * Q.mass - 2.277531
    if 91.03469 <= Q.mass < 101.0497:
        z += -0.01591318 * Q.mass + 1.54435
    if 101.0497 <= Q.mass < 172.4888:
        z += -0.001623115 * Q.mass + 0.1003435
    if Q.mass >= 172.4888:
        z += -0.6712597 * Q.mass + 115.6052
    if Q.sum_z_dr2_top30 < 0.006363916:
        z += 60.48492 * Q.sum_z_dr2_top30 - 0.4752275
    if 0.006363916 <= Q.sum_z_dr2_top30 < 0.007856958:
        z += 108.244 * Q.sum_z_dr2_top30 - 0.7791624
    if Q.sum_z_dr2_top30 >= 0.007856958:
        z += 47.75911 * Q.sum_z_dr2_top30 - 0.3039349
    if Q.z_dr_0_0p05 >= 0.9084912:
        z += 1.932456 * Q.z_dr_0_0p05 - 1.75562
    if Q.mass_top50 >= 157.5448:
        z += -0.03201871 * Q.mass_top50 + 5.044382
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -1207.887 * Q.mass_over_sum_pt_sq + 35.27032
    if Q.sj2_mass1 < 23.31647:
        z += -0.01072236 * Q.sj2_mass1 + 0.2500077
    if 69.65633 <= Q.sd_mass < 83.30647:
        z += 0.01360287 * Q.sd_mass - 0.947526
    if Q.sd_mass >= 83.30647:
        z += 0.0008389838 * Q.sd_mass + 0.1157884
    if Q.tau21 < 0.5100475:
        z += 0.7501771 * Q.tau21 - 0.382626
    if 822.975 <= Q.sum_pt_top10 < 943.6922:
        z += 0.002422987 * Q.sum_pt_top10 - 1.994058
    if Q.sum_pt_top10 >= 943.6922:
        z += 0.001033439 * Q.sum_pt_top10 - 0.6827519
    if Q.z_11 < 0.01375115:
        z += -60.20345 * Q.z_11 + 0.8278663
    if Q.pt_11 < 14.14062:
        z += 0.03868026 * Q.pt_11 - 0.546963
    if Q.e2 < 0.04082832:
        z += -20.67412 * Q.e2 + 0.8440896
    if Q.tau1 < 0.1507173:
        z += 1.399562 * Q.tau1 - 0.2109381
    if Q.z_dr_0p2_0p4 < 0.03700182:
        z += 3.989977 * Q.z_dr_0p2_0p4 - 0.1476364
    if Q.n_dr_0p1_0p2 < 8.0:
        z += -0.01077068 * Q.n_dr_0p1_0p2 - 0.1253606
    if 8.0 <= Q.n_dr_0p1_0p2 < 21.0:
        z += 0.01627124 * Q.n_dr_0p1_0p2 - 0.341696
    if Q.mass_top10 >= 71.781:
        z += 0.006133478 * Q.mass_top10 - 0.4402672
    if Q.M3 < 0.03787151:
        z += -4.041749 * Q.M3 + 0.1530671
    if Q.psi_0p1 < 0.707925:
        z += 0.3268714 * Q.psi_0p1 - 0.2314004
    if Q.D2 < 1.976207:
        z += -0.1709806 * Q.D2 + 0.3378931
    if Q.M2 >= 0.06621477:
        z += 2.841178 * Q.M2 - 0.188128
    if Q.sum_z_dr2_top20 < 0.0007522186:
        z += 487.8156 * Q.sum_z_dr2_top20 - 0.366944
    if Q.psi_0p3 >= 0.9638082:
        z += 3.688692 * Q.psi_0p3 - 3.555191
    if Q.z_top10_slots >= 0.7887522:
        z += -1.188321 * Q.z_top10_slots + 0.9372904
    if Q.sum_pt_top15 >= 935.1043:
        z += -0.003021589 * Q.sum_pt_top15 + 2.8255
    if Q.soft3_z >= 0.0008504382:
        z += -114.347 * Q.soft3_z + 0.09724505
    if Q.soft3_pt >= 0.3395996:
        z += 0.09429043 * Q.soft3_pt - 0.03202099
    if Q.n_real_top50 >= 22.0:
        z += -0.01163249 * Q.n_real_top50 + 0.2559148
    if Q.n_particles < 64.0 and Q.D2 < 2.178951:
        z += -0.01153539 * (64.0 - Q.n_particles) * (2.178951 - Q.D2)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 49666.75 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    if Q.log_sum_pt > 6.910131 and Q.dr_12 < 0.1269782:
        z += -4.944772 * (Q.log_sum_pt - 6.910131) * (0.1269782 - Q.dr_12)
    if Q.psi_0p3 > 0.9973959 and Q.D2 < 3.345339:
        z += 48.66055 * (Q.psi_0p3 - 0.9973959) * (3.345339 - Q.D2)
    if Q.log_sum_pt > 6.910131 and Q.absphi_13 < 0.1958008:
        z += -4.034366 * (Q.log_sum_pt - 6.910131) * (0.1958008 - Q.absphi_13)
    if Q.mass > 172.4888 and Q.soft3_dr < 0.161871:
        z += -3.895343 * (Q.mass - 172.4888) * (0.161871 - Q.soft3_dr)
    if Q.sum_pt_top40 > 994.2695 and Q.e4 < 5.8505e-08:
        z += -94503.5 * (Q.sum_pt_top40 - 994.2695) * (5.8505e-08 - Q.e4)
    return max(0.0, z)


def neuron_6(Q):
    z = -0.4606062
    if Q.mass_top50 < 71.79516:
        z += -0.02611435 * Q.mass_top50 + 1.874884
    if Q.mass < 82.85409:
        z += -0.01014283 * Q.mass - 0.2472066
    if 82.85409 <= Q.mass < 89.74183:
        z += 0.01410487 * Q.mass - 2.256228
    if 89.74183 <= Q.mass < 92.85979:
        z += 0.04834828 * Q.mass - 5.329294
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.1099631 * Q.mass - 11.05083
    if 101.0497 <= Q.mass < 121.3913:
        z += 0.0199136 * Q.mass - 1.951358
    if 121.3913 <= Q.mass < 143.7876:
        z += -0.009119419 * Q.mass + 1.572998
    if 143.7876 <= Q.mass < 172.4888:
        z += -0.02589853 * Q.mass + 3.985625
    if Q.mass >= 172.4888:
        z += -0.01677911 * Q.mass + 2.412628
    if Q.sum_z_dr2_top15 < 0.002197765:
        z += -72.87313 * Q.sum_z_dr2_top15 + 0.160158
    if Q.sj3_pair_mass_min >= 32.51366:
        z += -0.006458036 * Q.sj3_pair_mass_min + 0.2099744
    if 0.02793599 <= Q.e2 < 0.04755309:
        z += -35.13575 * Q.e2 + 0.9815522
    if Q.e2 >= 0.04755309:
        z += -7.675255 * Q.e2 - 0.3242793
    if Q.log_sum_pt < 6.811175:
        z += 4.133828 * Q.log_sum_pt - 28.15622
    if Q.tau1 < 0.05444509:
        z += -5.049356 * Q.tau1 + 0.3186562
    if 0.05444509 <= Q.tau1 < 0.06310829:
        z += 3.555919 * Q.tau1 - 0.1498588
    if Q.tau1 >= 0.06310829:
        z += 8.605275 * Q.tau1 - 0.468515
    if Q.z_top30_slots < 0.8635933:
        z += -4.383665 * Q.z_top30_slots + 3.785703
    if Q.sj3_mass1 >= 19.30204:
        z += -0.007814223 * Q.sj3_mass1 + 0.1508305
    if Q.lam2 < 0.003687605:
        z += -129.7645 * Q.lam2 + 0.47852
    if Q.z_dr_0p1_0p2 < 0.2864926:
        z += -0.6896895 * Q.z_dr_0p1_0p2 + 0.1975909
    if Q.sum_z_dr2_top2 < 0.001425993:
        z += 148.5894 * Q.sum_z_dr2_top2 - 0.2118875
    if Q.D2 < 1.409617:
        z += 0.4835779 * Q.D2 - 0.7561536
    if 1.409617 <= Q.D2 < 2.178951:
        z += 0.09682904 * Q.D2 - 0.2109857
    if Q.lam1 < 0.007259287:
        z += -130.8308 * Q.lam1 + 0.9497382
    if Q.lam1 >= 0.02457025:
        z += -41.77342 * Q.lam1 + 1.026383
    if Q.mass_over_sum_pt < 0.09795415:
        z += 33.21998 * Q.mass_over_sum_pt - 3.254035
    if Q.sum_zz_dr2 < 0.02580859:
        z += -193.3932 * Q.sum_zz_dr2 + 4.991206
    if Q.sum_z_dr2_top50 < 0.02550569:
        z += 83.03487 * Q.sum_z_dr2_top50 - 2.117862
    if Q.sum_z_dr2_top30 < 0.008376291:
        z += -46.7613 * Q.sum_z_dr2_top30 - 0.6969959
    if 0.008376291 <= Q.sum_z_dr2_top30 < 0.02412652:
        z += 69.12167 * Q.sum_z_dr2_top30 - 1.667665
    if Q.tau21_b2 < 0.1722488:
        z += -0.865256 * Q.tau21_b2 - 0.1071996
    if 0.1722488 <= Q.tau21_b2 < 0.7058597:
        z += 0.480198 * Q.tau21_b2 - 0.3389524
    if Q.sum_z_dr2_top20 < 0.00287991:
        z += -52.32335 * Q.sum_z_dr2_top20 + 0.1506866
    if Q.sj2_mass1 < 51.2028:
        z += -0.004942793 * Q.sj2_mass1 + 0.436265
    if 51.2028 <= Q.sj2_mass1 < 65.20727:
        z += -0.01308012 * Q.sj2_mass1 + 0.8529186
    if Q.soft10_dr >= 0.2075213:
        z += -0.9700265 * Q.soft10_dr + 0.2013012
    if Q.n_dr_0p1_0p2 >= 17.0:
        z += 0.01272105 * Q.n_dr_0p1_0p2 - 0.2162579
    if Q.psi_0p3 >= 0.9924477:
        z += -15.63568 * Q.psi_0p3 + 15.5176
    if Q.sum_pt_top20 < 926.0902:
        z += 0.000558603 * Q.sum_pt_top20 - 0.5173168
    if Q.mass_top10 >= 99.06678:
        z += 0.01330355 * Q.mass_top10 - 1.31794
    if Q.mass_top40 >= 94.64253:
        z += 0.01042872 * Q.mass_top40 - 0.9870005
    z += -0.007562369 * Q.sj2_mass2
    if Q.soft10_dr0 >= 0.1909669:
        z += 0.770131 * Q.soft10_dr0 - 0.1470695
    if Q.mass < 121.3913 and Q.sum_pt_top50 < 1003.544:
        z += 0.0001097646 * (121.3913 - Q.mass) * (1003.544 - Q.sum_pt_top50)
    if Q.e2 > 0.04755309 and Q.psi_0p3 > 0.9896594:
        z += -2062.383 * (Q.e2 - 0.04755309) * (Q.psi_0p3 - 0.9896594)
    if Q.tau1 < 0.06310829 and Q.lam2 > 0.0002104765:
        z += 12776.46 * (0.06310829 - Q.tau1) * (Q.lam2 - 0.0002104765)
    if Q.mass < 89.74183 and Q.sum_pt_top20 < 1129.275:
        z += -3.701437e-05 * (89.74183 - Q.mass) * (1129.275 - Q.sum_pt_top20)
    if Q.e2 > 0.04755309 and Q.zdr_0 > 0.0008718296:
        z += 495.3185 * (Q.e2 - 0.04755309) * (Q.zdr_0 - 0.0008718296)
    if Q.n_dr_0p2_0p4 > 15.0 and Q.max_pair_mass > 25.32749:
        z += -0.001206652 * (Q.n_dr_0p2_0p4 - 15.0) * (Q.max_pair_mass - 25.32749)
    if Q.e2 > 0.04755309 and Q.zdr_2 > 0.002740381:
        z += 554.7294 * (Q.e2 - 0.04755309) * (Q.zdr_2 - 0.002740381)
    if Q.e2 > 0.04755309 and Q.zdr_1 > 0.003059578:
        z += 551.8152 * (Q.e2 - 0.04755309) * (Q.zdr_1 - 0.003059578)
    if Q.z_dr_0p1_0p2 < 0.2864926 and Q.sj3_dr13 > 0.26623:
        z += 8.066151 * (0.2864926 - Q.z_dr_0p1_0p2) * (Q.sj3_dr13 - 0.26623)
    if Q.e2 > 0.02793599 and Q.psi_0p3 > 0.9896594:
        z += 3083.305 * (Q.e2 - 0.02793599) * (Q.psi_0p3 - 0.9896594)
    if Q.z_dr_0p05_0p1 < 0.2126484 and Q.sj3_dr13 > 0.06988208:
        z += -3.717415 * (0.2126484 - Q.z_dr_0p05_0p1) * (Q.sj3_dr13 - 0.06988208)
    if Q.mass < 92.85979 and Q.z_6 < 0.05799644:
        z += -0.06752994 * (92.85979 - Q.mass) * (0.05799644 - Q.z_6)
    if Q.e2 > 0.04755309 and Q.eccentricity < 0.8680812:
        z += -31.11679 * (Q.e2 - 0.04755309) * (0.8680812 - Q.eccentricity)
    if Q.sj2_mass1 < 65.20727 and Q.eccentricity > 0.3346888:
        z += 0.006196411 * (65.20727 - Q.sj2_mass1) * (Q.eccentricity - 0.3346888)
    return max(0.0, z)


def neuron_7(Q):
    z = 8.166514
    if Q.tau21_b2 < 0.2352054:
        z += -7.961034 * Q.tau21_b2 + 1.872479
    if Q.mass_over_sum_pt < 0.1182259:
        z += -22.59148 * Q.mass_over_sum_pt + 2.670899
    if Q.mass < 78.26182:
        z += 0.05749267 * Q.mass - 5.157332
    if 78.26182 <= Q.mass < 82.85409:
        z += 0.09123735 * Q.mass - 7.798252
    if 82.85409 <= Q.mass < 91.03469:
        z += 0.04747876 * Q.mass - 4.172674
    if 91.03469 <= Q.mass < 92.85979:
        z += 0.2305675 * Q.mass - 20.8401
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.005716318 * Q.mass + 0.03953313
    if 101.0497 <= Q.mass < 121.3913:
        z += -0.0303401 * Q.mass + 3.683023
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.08429382 * Q.n_dr_0p2_0p4 + 0.5057629
    if Q.psi_0p3 >= 0.9980008:
        z += -184.2798 * Q.psi_0p3 + 183.9114
    z += -1.197686 * Q.log_sum_pt
    if Q.tau1 < 0.07708632:
        z += -36.7088 * Q.tau1 + 2.225641
    if 0.07708632 <= Q.tau1 < 0.08786745:
        z += -1.292429 * Q.tau1 - 0.504477
    if 0.08786745 <= Q.tau1 < 0.1072713:
        z += 31.85145 * Q.tau1 - 3.416745
    if Q.LHA < 0.2601462:
        z += 11.82622 * Q.LHA - 2.917986
    if 0.2601462 <= Q.LHA < 0.2845608:
        z += -2.019906 * Q.LHA + 0.6840313
    if 0.2845608 <= Q.LHA < 0.3098384:
        z += -13.97148 * Q.LHA + 4.08498
    if 0.3098384 <= Q.LHA < 0.3332345:
        z += 10.42564 * Q.LHA - 3.474183
    if Q.tau21 < 0.3861957:
        z += 0.4210091 * Q.tau21 - 0.1625919
    if Q.mass_top20 < 70.42121:
        z += -0.002102921 * Q.mass_top20 - 0.006229993
    if 70.42121 <= Q.mass_top20 < 73.35236:
        z += 0.06140177 * Q.mass_top20 - 4.478307
    if 73.35236 <= Q.mass_top20 < 90.4505:
        z += -0.001500631 * Q.mass_top20 + 0.1357329
    if Q.z_top20_slots >= 0.9102775:
        z += -3.992295 * Q.z_top20_slots + 3.634096
    if Q.sum_z_dr2_top50 < 0.003418057:
        z += 711.0073 * Q.sum_z_dr2_top50 - 2.430263
    if Q.lam1 < 0.006189818:
        z += 82.80202 * Q.lam1 - 0.8819687
    if 0.006189818 <= Q.lam1 < 0.008241985:
        z += 180.024 * Q.lam1 - 1.483755
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += -461.3988 * Q.mass_over_sum_pt_sq + 4.427128
    if Q.lam1_plus_lam2 < 0.008190222:
        z += 477.9139 * Q.lam1_plus_lam2 - 3.914221
    if Q.tau2 < 0.05936026:
        z += 8.930116 * Q.tau2 - 0.530094
    if Q.sum_pt_top50 < 1245.697:
        z += 0.001325496 * Q.sum_pt_top50 - 1.651166
    if Q.n_dr_0_0p05 < 18.0:
        z += 0.0081576 * Q.n_dr_0_0p05 - 0.1468368
    if Q.sum_z_dr2_top30 < 0.006929741:
        z += 113.5525 * Q.sum_z_dr2_top30 - 0.6948653
    if 0.006929741 <= Q.sum_z_dr2_top30 < 0.008376291:
        z += 237.1717 * Q.sum_z_dr2_top30 - 1.551514
    if 0.008376291 <= Q.sum_z_dr2_top30 < 0.01215787:
        z += -115.059 * Q.sum_z_dr2_top30 + 1.398873
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1937688:
        z += -39.41764 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1937688)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.sj2_dr < 0.1825048:
        z += -0.6888198 * (6.0 - Q.n_dr_0p2_0p4) * (0.1825048 - Q.sj2_dr)
    if Q.mass < 91.03469 and Q.psi_0p3 > 0.9299135:
        z += 0.2017715 * (91.03469 - Q.mass) * (Q.psi_0p3 - 0.9299135)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.e3 < 7.876005e-05:
        z += -1486.942 * (6.0 - Q.n_dr_0p2_0p4) * (7.876005e-05 - Q.e3)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9638082:
        z += 2.302479 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.mass < 82.85409 and Q.psi_0p3 < 0.9995915:
        z += -1.688717 * (82.85409 - Q.mass) * (0.9995915 - Q.psi_0p3)
    if Q.mass < 91.03469 and Q.psi_0p3 > 0.9638082:
        z += -4.330767 * (91.03469 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.tau21_b2 < 0.2352054 and Q.n_real_top50 > 34.0:
        z += -0.09134425 * (0.2352054 - Q.tau21_b2) * (Q.n_real_top50 - 34.0)
    if Q.psi_0p3 > 0.9980008 and Q.n_dr_0p1_0p2 > 15.0:
        z += -9.003579 * (Q.psi_0p3 - 0.9980008) * (Q.n_dr_0p1_0p2 - 15.0)
    if Q.mass < 91.03469 and Q.n_dr_0_0p05 < 6.0:
        z += -0.008942405 * (91.03469 - Q.mass) * (6.0 - Q.n_dr_0_0p05)
    if Q.mass < 91.03469 and Q.psi_0p3 < 0.9973959:
        z += 0.9953224 * (91.03469 - Q.mass) * (0.9973959 - Q.psi_0p3)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9973959:
        z += 6.364707 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9973959)
    if Q.mass_over_sum_pt < 0.1182259 and Q.psi_0p3 > 0.9973959:
        z += 4058.355 * (0.1182259 - Q.mass_over_sum_pt) * (Q.psi_0p3 - 0.9973959)
    if Q.mass < 91.03469 and Q.psi_0p3 > 0.9973959:
        z += -73.23356 * (91.03469 - Q.mass) * (Q.psi_0p3 - 0.9973959)
    if Q.tau21_b2 < 0.2352054 and Q.pt2_over_pt0 < 0.3825328:
        z += -5.65473 * (0.2352054 - Q.tau21_b2) * (0.3825328 - Q.pt2_over_pt0)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9973959:
        z += 47.04866 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9973959)
    if Q.psi_0p3 > 0.9980008 and Q.orientation_deg > -9.840088:
        z += -1.277776 * (Q.psi_0p3 - 0.9980008) * (Q.orientation_deg - -9.840088)
    if Q.psi_0p3 > 0.9980008 and Q.max_dr > 0.1939977:
        z += -478.5141 * (Q.psi_0p3 - 0.9980008) * (Q.max_dr - 0.1939977)
    if Q.mass < 121.3913 and Q.sd_mass > 76.29481:
        z += 0.001682445 * (121.3913 - Q.mass) * (Q.sd_mass - 76.29481)
    if Q.mass < 91.03469 and Q.sd_mass > 55.96662:
        z += -0.003532367 * (91.03469 - Q.mass) * (Q.sd_mass - 55.96662)
    if Q.tau21_b2 < 0.2352054 and Q.psi_0p3 > 0.9299135:
        z += -53.21367 * (0.2352054 - Q.tau21_b2) * (Q.psi_0p3 - 0.9299135)
    if Q.tau21_b2 < 0.2352054 and Q.sj3_mass3 < 11.65938:
        z += 0.07278255 * (0.2352054 - Q.tau21_b2) * (11.65938 - Q.sj3_mass3)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.soft1_pt < 2.275391:
        z += 0.0249738 * (6.0 - Q.n_dr_0p2_0p4) * (2.275391 - Q.soft1_pt)
    if Q.sum_z_dr2 < 0.007877041 and Q.M2 < 0.1134943:
        z += 1103.879 * (0.007877041 - Q.sum_z_dr2) * (0.1134943 - Q.M2)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.pt_4 > 60.03125:
        z += 0.001323049 * (6.0 - Q.n_dr_0p2_0p4) * (Q.pt_4 - 60.03125)
    if Q.mass_top20 < 73.35236 and Q.soft1_z < 0.002181998:
        z += -1.891939 * (73.35236 - Q.mass_top20) * (0.002181998 - Q.soft1_z)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.soft8_dr0 < 0.3811408:
        z += 0.07081416 * (6.0 - Q.n_dr_0p2_0p4) * (0.3811408 - Q.soft8_dr0)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.6831469
    if 0.07696632 <= Q.mass_over_sum_pt < 0.1182259:
        z += 17.36376 * Q.mass_over_sum_pt - 1.336425
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -3.865704 * Q.mass_over_sum_pt + 1.173449
    if Q.sum_pt_top40 < 858.8262:
        z += -0.004041871 * Q.sum_pt_top40 + 4.144618
    if 858.8262 <= Q.sum_pt_top40 < 1001.523:
        z += -0.004384846 * Q.sum_pt_top40 + 4.439174
    if 1001.523 <= Q.sum_pt_top40 < 1032.405:
        z += -0.003470708 * Q.sum_pt_top40 + 3.523644
    if Q.sum_pt_top40 >= 1032.405:
        z += -0.0003429756 * Q.sum_pt_top40 + 0.2945564
    if Q.sum_z_dr2_top40 < 0.005196966:
        z += -115.9474 * Q.sum_z_dr2_top40 + 0.768751
    if 0.005196966 <= Q.sum_z_dr2_top40 < 0.006026828:
        z += -7.008553 * Q.sum_z_dr2_top40 + 0.2025993
    if 0.006026828 <= Q.sum_z_dr2_top40 < 0.006630167:
        z += -288.4275 * Q.sum_z_dr2_top40 + 1.898663
    if 0.006630167 <= Q.sum_z_dr2_top40 < 0.008031986:
        z += -172.4801 * Q.sum_z_dr2_top40 + 1.129912
    if Q.sum_z_dr2_top40 >= 0.008031986:
        z += -44.1627 * Q.sum_z_dr2_top40 + 0.09926872
    if Q.n_dr_0p2_0p4 < 8.0:
        z += 0.03984031 * Q.n_dr_0p2_0p4 - 0.5976047
    if 8.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += 0.06539305 * Q.n_dr_0p2_0p4 - 0.8020266
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += 0.02555274 * Q.n_dr_0p2_0p4 - 0.2044219
    if 64.48544 <= Q.mass < 80.78464:
        z += 0.02905014 * Q.mass - 1.873311
    if 80.78464 <= Q.mass < 89.74183:
        z += 0.08377521 * Q.mass - 6.294256
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.08736947 * Q.mass - 6.616812
    if 101.0497 <= Q.mass < 121.3913:
        z += 0.002078692 * Q.mass + 2.001796
    if 121.3913 <= Q.mass < 143.7876:
        z += -0.02863886 * Q.mass + 5.73064
    if Q.mass >= 143.7876:
        z += -0.04469164 * Q.mass + 8.03883
    if Q.log_sum_pt < 6.811175:
        z += 11.76805 * Q.log_sum_pt - 79.2065
    if 6.811175 <= Q.log_sum_pt < 6.935549:
        z += -7.62007 * Q.log_sum_pt + 52.84937
    if Q.log_sum_pt >= 6.959294:
        z += 9.480602 * Q.log_sum_pt - 65.97829
    if Q.mass_top20 >= 90.4505:
        z += -0.00405694 * Q.mass_top20 + 0.3669522
    if Q.n_for_90pct >= 39.0:
        z += -0.08664183 * Q.n_for_90pct + 3.379031
    if Q.sum_z_dr2_top15 < 0.001319197:
        z += -53.70638 * Q.sum_z_dr2_top15 + 0.007758428
    if 0.001319197 <= Q.sum_z_dr2_top15 < 0.005383629:
        z += -29.26635 * Q.sum_z_dr2_top15 - 0.02448279
    if 0.005383629 <= Q.sum_z_dr2_top15 < 0.007887677:
        z += -59.20439 * Q.sum_z_dr2_top15 + 0.1366925
    if 0.007887677 <= Q.sum_z_dr2_top15 < 0.01563836:
        z += 87.76694 * Q.sum_z_dr2_top15 - 1.02257
    if Q.sum_z_dr2_top15 >= 0.01563836:
        z += 24.44003 * Q.sum_z_dr2_top15 - 0.03224122
    if 43.66601 <= Q.mass_top50 < 80.35535:
        z += -0.009824293 * Q.mass_top50 + 0.4289877
    if 80.35535 <= Q.mass_top50 < 97.93004:
        z += -0.0488406 * Q.mass_top50 + 3.564157
    if Q.mass_top50 >= 97.93004:
        z += -0.007552484 * Q.mass_top50 - 0.4791902
    if 0.007872294 <= Q.sum_zz_dr2 < 0.009606007:
        z += 287.0053 * Q.sum_zz_dr2 - 2.25939
    if Q.sum_zz_dr2 >= 0.009606007:
        z += -114.4826 * Q.sum_zz_dr2 + 1.597306
    if Q.sj2_dr >= 0.2232169:
        z += 1.718935 * Q.sj2_dr - 0.3836953
    if Q.e2 >= 0.04755309:
        z += 22.3885 * Q.e2 - 1.064642
    if Q.sum_pt_top50 < 1003.544:
        z += -0.003176741 * Q.sum_pt_top50 + 3.188
    if Q.sum_pt_top50 >= 1038.855:
        z += -0.006687472 * Q.sum_pt_top50 + 6.947314
    if Q.n_dr_0p1_0p2 >= 19.0:
        z += 0.03442376 * Q.n_dr_0p1_0p2 - 0.6540515
    if Q.D2 < 1.409617:
        z += -0.5781561 * Q.D2 + 0.8149788
    if Q.tau1 >= 0.06310829:
        z += -14.56538 * Q.tau1 + 0.9191964
    if Q.n_pt_above_1 >= 54.0:
        z += 0.02909878 * Q.n_pt_above_1 - 1.571334
    if Q.sum_pt < 907.9372:
        z += -0.01680177 * Q.sum_pt + 15.25495
    if Q.z_dr_0p2_0p4 < 0.1291856:
        z += -2.245332 * Q.z_dr_0p2_0p4 + 0.2900645
    if Q.mass_top40 >= 111.2487:
        z += 0.01447885 * Q.mass_top40 - 1.610753
    if Q.LHA >= 0.3332345:
        z += 7.798651 * Q.LHA - 2.59878
    if Q.N2 < 0.2555281:
        z += 2.284704 * Q.N2 - 0.5838063
    if Q.e3 < 0.0003372339:
        z += -1840.476 * Q.e3 + 0.6206708
    if Q.sum_z_dr2_top20 < 0.02737453:
        z += 18.73317 * Q.sum_z_dr2_top20 - 0.5128116
    if Q.C2 >= 0.07996447:
        z += 2.51724 * Q.C2 - 0.2012897
    if Q.sum_z_dr2_top30 < 0.007856958:
        z += -101.4592 * Q.sum_z_dr2_top30 + 0.7971608
    if Q.mass_over_sum_pt_sq >= 0.001785266:
        z += 234.0727 * Q.mass_over_sum_pt_sq - 0.4178822
    if Q.lam1 >= 0.008241985:
        z += 13.44607 * Q.lam1 - 0.1108223
    if Q.mass_top30 < 80.24626:
        z += 0.008333398 * Q.mass_top30 - 0.668724
    if Q.sum_z_dr2_top50 < 0.0289594:
        z += 29.12053 * Q.sum_z_dr2_top50 - 0.8433128
    if Q.sum_pt_top15 < 818.9605:
        z += -0.0008304173 * Q.sum_pt_top15 + 0.680079
    if Q.sum_pt_top40 < 1001.523 and Q.psi_0p3 > 0.9924477:
        z += 0.7279555 * (1001.523 - Q.sum_pt_top40) * (Q.psi_0p3 - 0.9924477)
    if Q.log_sum_pt > 6.959294 and Q.sj3_pair_mass_max > 28.35435:
        z += 0.02925433 * (Q.log_sum_pt - 6.959294) * (Q.sj3_pair_mass_max - 28.35435)
    if Q.sum_z_dr2_top15 < 0.007887677 and Q.tau21_b2 < 0.6133424:
        z += -126.1604 * (0.007887677 - Q.sum_z_dr2_top15) * (0.6133424 - Q.tau21_b2)
    if Q.sum_pt_top40 < 1032.405 and Q.psi_0p3 > 0.9924477:
        z += -0.7747326 * (1032.405 - Q.sum_pt_top40) * (Q.psi_0p3 - 0.9924477)
    if Q.sum_pt_top50 < 1003.544 and Q.n_dr_0p05_0p1 < 21.0:
        z += -0.0001143736 * (1003.544 - Q.sum_pt_top50) * (21.0 - Q.n_dr_0p05_0p1)
    if Q.log_sum_pt > 6.959294 and Q.zdr_0 > 0.008905716:
        z += -174.611 * (Q.log_sum_pt - 6.959294) * (Q.zdr_0 - 0.008905716)
    if Q.n_pt_above_1 > 54.0 and Q.n_dr_0p05_0p1 < 15.0:
        z += -0.001977095 * (Q.n_pt_above_1 - 54.0) * (15.0 - Q.n_dr_0p05_0p1)
    if Q.sum_pt_top40 < 1032.405 and Q.mass_top3 < 50.3522:
        z += -2.612289e-06 * (1032.405 - Q.sum_pt_top40) * (50.3522 - Q.mass_top3)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.2747612
    if Q.sum_pt_top40 < 956.2133:
        z += -0.0115912 * Q.sum_pt_top40 + 11.08366
    if Q.sum_z_dr2_top40 < 0.00217213:
        z += 156.9569 * Q.sum_z_dr2_top40 - 0.04064256
    if 0.00217213 <= Q.sum_z_dr2_top40 < 0.006259772:
        z += -73.46243 * Q.sum_z_dr2_top40 + 0.459858
    if Q.mass_over_sum_pt >= 0.1708801:
        z += -44.64774 * Q.mass_over_sum_pt + 7.629412
    if Q.mass_top30 < 73.33139:
        z += 0.002669066 * Q.mass_top30 - 0.8328803
    if 73.33139 <= Q.mass_top30 < 121.737:
        z += 0.01577434 * Q.mass_top30 - 1.793908
    if 121.737 <= Q.mass_top30 < 152.6883:
        z += -0.004084264 * Q.mass_top30 + 0.6236192
    if Q.mass < 64.48544:
        z += -0.03278489 * Q.mass + 3.685615
    if 64.48544 <= Q.mass < 79.65241:
        z += -0.08097991 * Q.mass + 6.793492
    if 79.65241 <= Q.mass < 87.36377:
        z += -0.04451188 * Q.mass + 3.888726
    if 101.0497 <= Q.mass < 121.3913:
        z += 0.01871089 * Q.mass - 1.89073
    if 121.3913 <= Q.mass < 143.7876:
        z += 0.01320147 * Q.mass - 1.221934
    if 143.7876 <= Q.mass < 162.8363:
        z += -0.08495222 * Q.mass + 12.89135
    if 162.8363 <= Q.mass < 172.4888:
        z += -0.1600688 * Q.mass + 25.12307
    if Q.mass >= 172.4888:
        z += -0.1047742 * Q.mass + 15.58536
    if Q.sum_z_dr < 0.02783745:
        z += 53.45698 * Q.sum_z_dr - 1.992805
    if 0.02783745 <= Q.sum_z_dr < 0.05048381:
        z += 22.28608 * Q.sum_z_dr - 1.125086
    if Q.sum_z_dr >= 0.08589404:
        z += 23.07376 * Q.sum_z_dr - 1.981899
    if Q.psi_0p3 >= 0.9943058:
        z += 35.77334 * Q.psi_0p3 - 35.56964
    if Q.z_top40_slots >= 0.9574183:
        z += -5.919594 * Q.z_top40_slots + 5.667528
    if Q.lam1 < 0.004673423:
        z += -124.8145 * Q.lam1 + 0.5833112
    if Q.lam1 >= 0.01174405:
        z += 48.05518 * Q.lam1 - 0.5643625
    if Q.sum_z_dr2_top20 < 0.01083435:
        z += -14.69332 * Q.sum_z_dr2_top20 + 0.1591927
    if Q.mass_top40 < 89.6788:
        z += -0.004994931 * Q.mass_top40 + 0.2867411
    if 89.6788 <= Q.mass_top40 < 132.4278:
        z += -0.0177804 * Q.mass_top40 + 1.433327
    if 132.4278 <= Q.mass_top40 < 163.2541:
        z += 0.02988662 * Q.mass_top40 - 4.879114
    if Q.tau1 < 0.05444509:
        z += 33.49772 * Q.tau1 - 1.823786
    if Q.n_pt_above_1 < 26.0:
        z += -0.02612533 * Q.n_pt_above_1 + 0.6792586
    if Q.e3 >= 0.0003372339:
        z += 805.0203 * Q.e3 - 0.2714802
    if Q.LHA < 0.2091025:
        z += -12.7207 * Q.LHA + 2.659931
    if Q.z_dr_0p1_0p2 < 0.2187642:
        z += -1.115787 * Q.z_dr_0p1_0p2 + 0.2440943
    if Q.sum_pt_top15 >= 985.0781:
        z += -0.001034558 * Q.sum_pt_top15 + 1.01912
    if Q.mass_over_sum_pt_sq < 0.003636595:
        z += -221.2832 * Q.mass_over_sum_pt_sq + 0.8047174
    if Q.lam1_plus_lam2 >= 0.0258669:
        z += -58.24174 * Q.lam1_plus_lam2 + 1.506533
    if Q.n_dr_0p1_0p2 < 33.0:
        z += 0.01308387 * Q.n_dr_0p1_0p2 - 0.4317678
    if Q.sum_z_dr2_top30 < 0.01215787:
        z += -48.30042 * Q.sum_z_dr2_top30 + 0.5872304
    if Q.sum_z_dr2_top15 < 0.004169954:
        z += -54.36344 * Q.sum_z_dr2_top15 + 0.226693
    if Q.sum_z_dr2_top50 < 0.00634935:
        z += -52.84948 * Q.sum_z_dr2_top50 + 0.3355599
    if Q.sum_pt_top40 < 956.2133 and Q.psi_0p1 > 0.9184255:
        z += -0.06383151 * (956.2133 - Q.sum_pt_top40) * (Q.psi_0p1 - 0.9184255)
    if Q.sum_pt_top40 < 956.2133 and Q.soft4_pt > 1.789258:
        z += -0.009694512 * (956.2133 - Q.sum_pt_top40) * (Q.soft4_pt - 1.789258)
    if Q.mass_top40 < 80.89043 and Q.sum_pt < 1034.834:
        z += -0.0002189314 * (80.89043 - Q.mass_top40) * (1034.834 - Q.sum_pt)
    if Q.mass < 87.36377 and Q.n_particles > 38.0:
        z += 0.0002943311 * (87.36377 - Q.mass) * (Q.n_particles - 38.0)
    if Q.mass_top40 < 80.89043 and Q.sum_pt_top50 < 934.2416:
        z += 0.0002245173 * (80.89043 - Q.mass_top40) * (934.2416 - Q.sum_pt_top50)
    if Q.mass_top30 < 121.737 and Q.n_dr_0p2_0p4 > 10.0:
        z += 0.000701586 * (121.737 - Q.mass_top30) * (Q.n_dr_0p2_0p4 - 10.0)
    if Q.sum_pt_top40 < 956.2133 and Q.n_pt_above_10 < 20.0:
        z += -0.0004342193 * (956.2133 - Q.sum_pt_top40) * (20.0 - Q.n_pt_above_10)
    if Q.sum_pt_top40 < 956.2133 and Q.soft5_pt > 2.894531:
        z += 0.01303133 * (956.2133 - Q.sum_pt_top40) * (Q.soft5_pt - 2.894531)
    if Q.sum_pt_top50 < 959.0957 and Q.soft4_pt > 1.375:
        z += 0.008232161 * (959.0957 - Q.sum_pt_top50) * (Q.soft4_pt - 1.375)
    if Q.sum_pt_top50 < 959.0957 and Q.soft5_pt > 2.894531:
        z += -0.01093864 * (959.0957 - Q.sum_pt_top50) * (Q.soft5_pt - 2.894531)
    if Q.sum_pt_top40 < 956.2133 and Q.soft5_z > 0.0005078411:
        z += -3.082078 * (956.2133 - Q.sum_pt_top40) * (Q.soft5_z - 0.0005078411)
    if Q.mass > 143.7876 and Q.z_dr_0p05_0p1 > 0.4026646:
        z += 0.06480569 * (Q.mass - 143.7876) * (Q.z_dr_0p05_0p1 - 0.4026646)
    if Q.sum_z_dr > 0.08589404 and Q.z_dr_0p05_0p1 > 0.5106729:
        z += -51.29494 * (Q.sum_z_dr - 0.08589404) * (Q.z_dr_0p05_0p1 - 0.5106729)
    if Q.lam1 > 0.01174405 and Q.soft5_pt < 2.894531:
        z += 6.86398 * (Q.lam1 - 0.01174405) * (2.894531 - Q.soft5_pt)
    if Q.mass_over_sum_pt > 0.1708801 and Q.soft4_pt > 1.789258:
        z += 34.1193 * (Q.mass_over_sum_pt - 0.1708801) * (Q.soft4_pt - 1.789258)
    if Q.lam1 > 0.01174405 and Q.soft5_z > 0.002346473:
        z += -17008.22 * (Q.lam1 - 0.01174405) * (Q.soft5_z - 0.002346473)
    if Q.lam1 > 0.01174405 and Q.sj3_pairmin_over_m > 0.1952481:
        z += -56.79358 * (Q.lam1 - 0.01174405) * (Q.sj3_pairmin_over_m - 0.1952481)
    if Q.lam1_plus_lam2 > 0.0258669 and Q.soft5_pt > 1.022461:
        z += -53.86951 * (Q.lam1_plus_lam2 - 0.0258669) * (Q.soft5_pt - 1.022461)
    if Q.sum_pt_top50 < 959.0957 and Q.soft5_z > 0.0009919684:
        z += 2.435274 * (959.0957 - Q.sum_pt_top50) * (Q.soft5_z - 0.0009919684)
    if Q.sum_z_dr > 0.08589404 and Q.soft5_z < 0.002036949:
        z += -28108.86 * (Q.sum_z_dr - 0.08589404) * (0.002036949 - Q.soft5_z)
    if Q.sum_z_dr > 0.09749958 and Q.soft5_z < 0.002036949:
        z += 31148.54 * (Q.sum_z_dr - 0.09749958) * (0.002036949 - Q.soft5_z)
    if Q.z_top5 > 0.7963975 and Q.ptdr0_2 < 7.740999:
        z += 0.4670104 * (Q.z_top5 - 0.7963975) * (7.740999 - Q.ptdr0_2)
    return max(0.0, z)


def neuron_10(Q):
    z = -2.157605
    if Q.sum_z_dr < 0.03577037:
        z += 82.25433 * Q.sum_z_dr - 6.726116
    if 0.03577037 <= Q.sum_z_dr < 0.1207452:
        z += 44.52906 * Q.sum_z_dr - 5.37667
    if Q.mass < 64.48544:
        z += -0.03757741 * Q.mass + 2.79019
    if 64.48544 <= Q.mass < 74.25181:
        z += -0.09690931 * Q.mass + 6.616235
    if 74.25181 <= Q.mass < 80.78464:
        z += -0.05933191 * Q.mass + 3.826044
    if 80.78464 <= Q.mass < 82.85409:
        z += 0.00742764 * Q.mass - 1.567102
    if 82.85409 <= Q.mass < 89.74183:
        z += -0.003923478 * Q.mass - 0.6266154
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.06174134 * Q.mass - 6.519497
    if 101.0497 <= Q.mass < 143.7876:
        z += 0.02680894 * Q.mass - 2.989587
    if 143.7876 <= Q.mass < 162.8363:
        z += -0.08235141 * Q.mass + 12.70632
    if Q.mass >= 162.8363:
        z += -0.1821046 * Q.mass + 28.94976
    if Q.sum_z_dr2_top30 < 0.005402331:
        z += 63.87016 * Q.sum_z_dr2_top30 + 0.589103
    if 0.005402331 <= Q.sum_z_dr2_top30 < 0.01807679:
        z += -73.70341 * Q.sum_z_dr2_top30 + 1.332321
    if Q.sum_pt_top20 < 750.7313:
        z += 0.001740125 * Q.sum_pt_top20 - 1.306366
    if Q.sj2_mass1 < 80.3008:
        z += 0.01858506 * Q.sj2_mass1 - 1.492395
    if Q.mass_top40 < 67.72643:
        z += -0.005068846 * Q.mass_top40 + 1.704328
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.01200905 * Q.mass_top40 + 0.5477037
    if 83.32554 <= Q.mass_top40 < 87.27603:
        z += -0.03401195 * Q.mass_top40 + 4.382428
    if 87.27603 <= Q.mass_top40 < 111.2487:
        z += -0.01139986 * Q.mass_top40 + 2.408934
    if 111.2487 <= Q.mass_top40 < 132.4278:
        z += -0.001688039 * Q.mass_top40 + 1.328507
    if Q.mass_top40 >= 132.4278:
        z += 0.01707789 * Q.mass_top40 - 1.156625
    if Q.D2 < 3.814159:
        z += -0.2802144 * Q.D2 + 1.068783
    if Q.mass_top5 >= 14.54404:
        z += 0.009195887 * Q.mass_top5 - 0.1337453
    if 138.8977 <= Q.mass_top50 < 157.5448:
        z += 0.04814889 * Q.mass_top50 - 6.687768
    if Q.mass_top50 >= 157.5448:
        z += 0.06387826 * Q.mass_top50 - 9.16585
    if Q.log_sum_pt >= 6.903423:
        z += -6.497919 * Q.log_sum_pt + 44.85788
    z += 0.002587048 * Q.sum_pt
    if Q.mass_top15 < 75.26407:
        z += 0.01129239 * Q.mass_top15 - 0.8499114
    if Q.mass_over_sum_pt < 0.07852883:
        z += -15.57858 * Q.mass_over_sum_pt + 1.223368
    if Q.mass_over_sum_pt >= 0.1708801:
        z += -223.9724 * Q.mass_over_sum_pt + 38.27242
    if Q.sum_pt_top5 < 839.9547:
        z += -0.001117517 * Q.sum_pt_top5 + 0.9386637
    if Q.e2 < 0.01541561:
        z += 0.4092255 * Q.e2 - 1.232351
    if 0.01541561 <= Q.e2 < 0.04082832:
        z += 34.62814 * Q.e2 - 1.759857
    if 0.04082832 <= Q.e2 < 0.06524004:
        z += 14.17548 * Q.e2 - 0.9248091
    if Q.dr_0 < 0.06413297:
        z += -6.113236 * Q.dr_0 + 0.39206
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.05164639 * Q.n_dr_0p2_0p4 - 0.5681103
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -5.11079 * Q.z_dr_0p2_0p4 + 0.4662353
    if Q.mass_top10 < 71.781:
        z += 0.009280029 * Q.mass_top10 - 0.6661297
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += 507.7465 * Q.mass_over_sum_pt_sq - 14.82621
    if Q.dr_1 < 0.06940472:
        z += -3.874035 * Q.dr_1 + 0.2688763
    if Q.tau1 < 0.05444509:
        z += 17.95668 * Q.tau1 + 1.454864
    if 0.05444509 <= Q.tau1 < 0.1507173:
        z += -25.26708 * Q.tau1 + 3.808186
    if Q.tau21_b2 < 0.6133424:
        z += -0.6854554 * Q.tau21_b2 + 0.4204189
    if Q.pt_entropy >= 2.07371:
        z += 0.7244113 * Q.pt_entropy - 1.502219
    if Q.M2 >= 0.04577853:
        z += -3.875009 * Q.M2 + 0.1773922
    if Q.mass_top30 < 42.41192:
        z += -0.01017495 * Q.mass_top30 + 0.431539
    if Q.sj2_mass2 < 11.91979:
        z += 0.01087047 * Q.sj2_mass2 - 0.1295737
    if Q.lam1 >= 0.001260456:
        z += -27.56038 * Q.lam1 + 0.03473863
    if Q.sum_z_dr2_top15 < 0.009962397:
        z += -74.23649 * Q.sum_z_dr2_top15 + 0.7395734
    if Q.n_real_top40 >= 36.0:
        z += 0.0470301 * Q.n_real_top40 - 1.693084
    if Q.LHA < 0.2091025:
        z += -9.596479 * Q.LHA + 2.006648
    if Q.sum_z_dr2_top20 < 0.005312783:
        z += 35.50912 * Q.sum_z_dr2_top20 - 0.1886522
    if Q.psi_0p3 >= 0.9956185:
        z += -32.68394 * Q.psi_0p3 + 32.54073
    if Q.mass > 162.8363 and Q.pt_9 < 41.4375:
        z += 0.0009517419 * (Q.mass - 162.8363) * (41.4375 - Q.pt_9)
    if Q.sj2_mass1 < 80.3008 and Q.sum_pt_top2 < 464.75:
        z += 8.081188e-06 * (80.3008 - Q.sj2_mass1) * (464.75 - Q.sum_pt_top2)
    if Q.D2 < 3.814159 and Q.sj2_dr > 0.1937688:
        z += -0.9418463 * (3.814159 - Q.D2) * (Q.sj2_dr - 0.1937688)
    if Q.mass_top40 > 163.2541 and Q.soft4_z > 0.001186042:
        z += 23.27623 * (Q.mass_top40 - 163.2541) * (Q.soft4_z - 0.001186042)
    if Q.mass > 64.48544 and Q.soft2_pt < 2.537109:
        z += 0.002857508 * (Q.mass - 64.48544) * (2.537109 - Q.soft2_pt)
    if Q.mass > 82.85409 and Q.psi_0p3 > 0.9853273:
        z += 0.3611975 * (Q.mass - 82.85409) * (Q.psi_0p3 - 0.9853273)
    if Q.log_sum_pt > 6.903423 and Q.dr_4 < 0.0830523:
        z += 13.6608 * (Q.log_sum_pt - 6.903423) * (0.0830523 - Q.dr_4)
    if Q.mass_top40 < 111.2487 and Q.pt1_dr01 > 12.6865:
        z += 0.0006907269 * (111.2487 - Q.mass_top40) * (Q.pt1_dr01 - 12.6865)
    if Q.D2 < 3.814159 and Q.max_dr > 0.2404747:
        z += 0.2950807 * (3.814159 - Q.D2) * (Q.max_dr - 0.2404747)
    if Q.psi_0p3 > 0.9985421 and Q.n_real_top30 > 22.0:
        z += -17.10495 * (Q.psi_0p3 - 0.9985421) * (Q.n_real_top30 - 22.0)
    if Q.sum_z_dr < 0.1207452 and Q.tau32 < 0.8125223:
        z += -8.111674 * (0.1207452 - Q.sum_z_dr) * (0.8125223 - Q.tau32)
    if Q.mass > 172.4888 and Q.M2 > 0.04260132:
        z += 0.2887887 * (Q.mass - 172.4888) * (Q.M2 - 0.04260132)
    if Q.z_dr_0_0p05 > 0.7674734 and Q.D2 < 4.450169:
        z += -0.687979 * (Q.z_dr_0_0p05 - 0.7674734) * (4.450169 - Q.D2)
    if Q.log_sum_pt > 6.903423 and Q.absphi_2 < 0.03259277:
        z += 23.82366 * (Q.log_sum_pt - 6.903423) * (0.03259277 - Q.absphi_2)
    if Q.sj2_mass1 < 80.3008 and Q.sj2_mass2 > 11.91979:
        z += 0.0002384642 * (80.3008 - Q.sj2_mass1) * (Q.sj2_mass2 - 11.91979)
    if Q.sum_pt_top15 < 918.0938 and Q.dr_2 < 0.07464736:
        z += 0.01347535 * (918.0938 - Q.sum_pt_top15) * (0.07464736 - Q.dr_2)
    if Q.sum_z_dr2_top30 < 0.01807679 and Q.z_dr_0p4_up < 0.0002836757:
        z += -32194.11 * (0.01807679 - Q.sum_z_dr2_top30) * (0.0002836757 - Q.z_dr_0p4_up)
    if Q.mass > 82.85409 and Q.n_dr_0p4_up > 0.0:
        z += -0.001250689 * (Q.mass - 82.85409) * (Q.n_dr_0p4_up - 0.0)
    if Q.D2 < 3.814159 and Q.n_real_top40 > 32.0:
        z += -0.01968174 * (3.814159 - Q.D2) * (Q.n_real_top40 - 32.0)
    if Q.log_sum_pt > 6.903423 and Q.centroid_offset < 0.002245509:
        z += 363.6475 * (Q.log_sum_pt - 6.903423) * (0.002245509 - Q.centroid_offset)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.2355001
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.09939298 * Q.n_dr_0p2_0p4 + 0.6881656
    if 6.0 <= Q.n_dr_0p2_0p4 < 10.0:
        z += -0.02295193 * Q.n_dr_0p2_0p4 + 0.2295193
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += -36.75754 * Q.mass_over_sum_pt_sq + 0.3526891
    if Q.psi_0p2 >= 0.9935324:
        z += 25.21463 * Q.psi_0p2 - 25.05155
    if Q.mass < 64.48544:
        z += -0.01688426 * Q.mass + 2.31536
    if 64.48544 <= Q.mass < 79.65241:
        z += -0.0308336 * Q.mass + 3.21489
    if 79.65241 <= Q.mass < 92.85979:
        z += -0.06674695 * Q.mass + 6.075474
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.01497367 * Q.mass - 1.513085
    if Q.sum_z_dr2_top10 < 0.00130722:
        z += -453.0262 * Q.sum_z_dr2_top10 + 0.5853535
    if 0.00130722 <= Q.sum_z_dr2_top10 < 0.003213724:
        z += -166.0392 * Q.sum_z_dr2_top10 + 0.2101982
    if 0.003213724 <= Q.sum_z_dr2_top10 < 0.004752876:
        z += -15.5294 * Q.sum_z_dr2_top10 - 0.2734987
    if 0.004752876 <= Q.sum_z_dr2_top10 < 0.007678544:
        z += 118.7107 * Q.sum_z_dr2_top10 - 0.9115251
    if Q.sum_z_dr < 0.05048381:
        z += 21.01316 * Q.sum_z_dr - 1.703677
    if 0.05048381 <= Q.sum_z_dr < 0.07374472:
        z += 27.63659 * Q.sum_z_dr - 2.038053
    if Q.lam1 < 0.006716737:
        z += 136.3971 * Q.lam1 - 0.9161437
    if Q.mass_top10 < 38.52415:
        z += 0.01135898 * Q.mass_top10 - 0.09082472
    if 38.52415 <= Q.mass_top10 < 60.5496:
        z += -0.01574408 * Q.mass_top10 + 0.9532978
    if Q.e2 < 0.02515919:
        z += -48.27407 * Q.e2 + 0.9831721
    if 0.02515919 <= Q.e2 < 0.02793599:
        z += -25.60355 * Q.e2 + 0.4128001
    if 0.02793599 <= Q.e2 < 0.04082832:
        z += 23.46052 * Q.e2 - 0.9578535
    if Q.sum_z_dr2_top30 < 0.006929741:
        z += 124.3956 * Q.sum_z_dr2_top30 - 0.8620291
    if Q.sum_pt_top5 >= 791.125:
        z += -0.001137713 * Q.sum_pt_top5 + 0.9000733
    if Q.sum_zz_dr2 < 0.00818374:
        z += -180.9041 * Q.sum_zz_dr2 + 1.480472
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.009709314 * Q.n_dr_0p1_0p2 + 0.03164139
    if 8.0 <= Q.n_dr_0p1_0p2 < 15.0:
        z += -0.01561656 * Q.n_dr_0p1_0p2 + 0.2342483
    if Q.zdr_0 < 0.005911134:
        z += 30.89306 * Q.zdr_0 - 0.182613
    if Q.LHA < 0.2845608:
        z += -3.205597 * Q.LHA + 0.8288597
    if 0.2845608 <= Q.LHA < 0.2941033:
        z += 1.752779 * Q.LHA - 0.5820999
    if 0.2941033 <= Q.LHA < 0.3098384:
        z += 12.15882 * Q.LHA - 3.64255
    if 0.3098384 <= Q.LHA < 0.3203321:
        z += -11.88499 * Q.LHA + 3.807145
    if Q.z_dr_0p1_0p2 < 0.1548383:
        z += 1.02255 * Q.z_dr_0p1_0p2 - 0.1583299
    if Q.mass_top30 < 60.43821:
        z += -0.00977492 * Q.mass_top30 + 0.4311337
    if 60.43821 <= Q.mass_top30 < 86.25221:
        z += 0.006184434 * Q.mass_top30 - 0.5334211
    if Q.mass_top50 < 85.8667:
        z += 0.01844184 * Q.mass_top50 - 1.245324
    if 85.8667 <= Q.mass_top50 < 97.93004:
        z += -0.02803664 * Q.mass_top50 + 2.745629
    if Q.z_top30_slots >= 0.9948118:
        z += 20.6228 * Q.z_top30_slots - 20.5158
    if Q.z_top10_slots >= 0.7271951:
        z += -0.8589468 * Q.z_top10_slots + 0.6246219
    if Q.z_dr_0_0p05 < 0.09573228:
        z += 2.802845 * Q.z_dr_0_0p05 - 0.2683228
    if Q.tau1 < 0.08786745:
        z += -0.117094 * Q.tau1 + 0.3834239
    if 0.08786745 <= Q.tau1 < 0.1072713:
        z += -19.22999 * Q.tau1 + 2.062826
    if Q.z_top40_slots >= 0.9886962:
        z += 12.6211 * Q.z_top40_slots - 12.47844
    if 0.9853273 <= Q.psi_0p3 < 0.9973959:
        z += -14.56115 * Q.psi_0p3 + 14.3475
    if Q.psi_0p3 >= 0.9973959:
        z += 38.19294 * Q.psi_0p3 - 38.26922
    if Q.sum_z_dr2_top2 < 0.004007842:
        z += -45.65761 * Q.sum_z_dr2_top2 - 0.03535573
    if 0.004007842 <= Q.sum_z_dr2_top2 < 0.007639643:
        z += 60.12008 * Q.sum_z_dr2_top2 - 0.4592959
    if Q.sum_pt < 986.0565:
        z += 0.001358954 * Q.sum_pt - 1.340006
    if Q.e3 < 4.204324e-05:
        z += 3171.749 * Q.e3 - 0.1333506
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_particles > 34.0:
        z += -0.001410645 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_particles - 34.0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.sum_z_dr2 < 0.005532208:
        z += -32.97103 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.sum_z_dr2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.004970383 * (10.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.mass < 101.0497 and Q.mass_top10 > 38.52415:
        z += 0.001147498 * (101.0497 - Q.mass) * (Q.mass_top10 - 38.52415)
    if Q.mass_over_sum_pt_sq < 0.009595015 and Q.z_dr_0p1_0p2 > 0.1548383:
        z += 393.7338 * (0.009595015 - Q.mass_over_sum_pt_sq) * (Q.z_dr_0p1_0p2 - 0.1548383)
    if Q.mass < 79.65241 and Q.D2 < 3.814159:
        z += -0.003253431 * (79.65241 - Q.mass) * (3.814159 - Q.D2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.z_7 > 0.01968091:
        z += 0.6134372 * (10.0 - Q.n_dr_0p2_0p4) * (Q.z_7 - 0.01968091)
    if Q.z_top30_slots > 0.9734886 and Q.C2_b2 < 0.05112769:
        z += 130.4695 * (Q.z_top30_slots - 0.9734886) * (0.05112769 - Q.C2_b2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.sj2_mass1 < 51.2028:
        z += -0.000595956 * (10.0 - Q.n_dr_0p2_0p4) * (51.2028 - Q.sj2_mass1)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.n_real_top30 > 22.0:
        z += -0.006352315 * (6.0 - Q.n_dr_0p2_0p4) * (Q.n_real_top30 - 22.0)
    if Q.mass < 79.65241 and Q.z_11 > 0.01375115:
        z += -0.7662032 * (79.65241 - Q.mass) * (Q.z_11 - 0.01375115)
    if Q.mass < 101.0497 and Q.z_11 > 0.01184033:
        z += 0.3539848 * (101.0497 - Q.mass) * (Q.z_11 - 0.01184033)
    if Q.z_dr_0_0p05 < 0.09573228 and Q.max_dr > 0.1939977:
        z += 9.274291 * (0.09573228 - Q.z_dr_0_0p05) * (Q.max_dr - 0.1939977)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.sum_pt < 1017.435:
        z += 0.0005472436 * (6.0 - Q.n_dr_0p2_0p4) * (1017.435 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.sum_pt > 907.9372:
        z += 0.0001111064 * (10.0 - Q.n_dr_0p2_0p4) * (Q.sum_pt - 907.9372)
    if Q.sum_z_dr < 0.07374472 and Q.pt_7 > 20.125:
        z += -0.08932965 * (0.07374472 - Q.sum_z_dr) * (Q.pt_7 - 20.125)
    if Q.tau21_b2 < 0.144845 and Q.z_5 > 0.03252317:
        z += 68.5 * (0.144845 - Q.tau21_b2) * (Q.z_5 - 0.03252317)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.1359007
    if Q.mass < 53.87362:
        z += -0.06365862 * Q.mass + 6.438732
    if 53.87362 <= Q.mass < 80.78464:
        z += -0.1045929 * Q.mass + 8.644011
    if 80.78464 <= Q.mass < 82.85409:
        z += -0.05145974 * Q.mass + 4.351666
    if 82.85409 <= Q.mass < 87.36377:
        z += -0.01951706 * Q.mass + 1.705084
    if Q.mass_top40 < 77.93668:
        z += 0.009761031 * Q.mass_top40 - 0.6083263
    if 77.93668 <= Q.mass_top40 < 94.64253:
        z += -0.009123507 * Q.mass_top40 + 0.8634718
    if Q.mass_top50 < 52.15154:
        z += -0.007474894 * Q.mass_top50 + 0.3898272
    if Q.z_dr_0p1_0p2 >= 0.4479367:
        z += 0.5183264 * Q.z_dr_0p1_0p2 - 0.2321774
    if Q.sum_zz_dr2 < 0.00363788:
        z += 106.217 * Q.sum_zz_dr2 - 0.3864049
    if Q.mass_top15 < 24.20196:
        z += -0.01048729 * Q.mass_top15 + 0.04503486
    if 24.20196 <= Q.mass_top15 < 46.36558:
        z += 0.009419858 * Q.mass_top15 - 0.4367571
    if Q.mass_over_sum_pt < 0.06895248:
        z += 20.05502 * Q.mass_over_sum_pt - 1.382843
    if Q.z_dr_0_0p05 >= 0.9084912:
        z += -2.787574 * Q.z_dr_0_0p05 + 2.532486
    if Q.sum_z_dr2_top10 < 0.0007431905:
        z += -213.1132 * Q.sum_z_dr2_top10 + 0.1583837
    if Q.mass < 87.36377 and Q.z_dr_0p1_0p2 < 0.06472584:
        z += -0.1807221 * (87.36377 - Q.mass) * (0.06472584 - Q.z_dr_0p1_0p2)
    if Q.mass < 87.36377 and Q.psi_0p3 < 0.9989733:
        z += -1.481373 * (87.36377 - Q.mass) * (0.9989733 - Q.psi_0p3)
    if Q.sd_mass > 133.2575 and Q.sj3_pair_mass_min < 76.60223:
        z += 0.0004372458 * (Q.sd_mass - 133.2575) * (76.60223 - Q.sj3_pair_mass_min)
    if Q.mass < 87.36377 and Q.lam2 > 0.0003497174:
        z += -7.765427 * (87.36377 - Q.mass) * (Q.lam2 - 0.0003497174)
    if Q.mass < 87.36377 and Q.z_dr_0p05_0p1 > 0.4026646:
        z += 0.05617974 * (87.36377 - Q.mass) * (Q.z_dr_0p05_0p1 - 0.4026646)
    if Q.sj3_pair_mass_max > 122.7494 and Q.pt_0 < 435.25:
        z += 1.908187e-05 * (Q.sj3_pair_mass_max - 122.7494) * (435.25 - Q.pt_0)
    if Q.mass_top40 < 94.64253 and Q.z_top2_slots < 0.5760704:
        z += -0.01710823 * (94.64253 - Q.mass_top40) * (0.5760704 - Q.z_top2_slots)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.planar_flow < 0.3036026:
        z += 17.46227 * (Q.z_dr_0p1_0p2 - 0.4479367) * (0.3036026 - Q.planar_flow)
    if Q.sd_mass > 133.2575 and Q.sj3_mass1 < 32.50209:
        z += 0.0005663808 * (Q.sd_mass - 133.2575) * (32.50209 - Q.sj3_mass1)
    if Q.sum_pt < 972.0419 and Q.C2_b2 < 0.05112769:
        z += -0.03367474 * (972.0419 - Q.sum_pt) * (0.05112769 - Q.C2_b2)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.lam2 < 0.001776308:
        z += -2433.414 * (Q.z_dr_0p1_0p2 - 0.4479367) * (0.001776308 - Q.lam2)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.728562
    if Q.sum_pt < 1012.673:
        z += 0.03431105 * Q.sum_pt - 35.68902
    if 1012.673 <= Q.sum_pt < 1085.125:
        z += 0.0130176 * Q.sum_pt - 14.12573
    if Q.sum_pt >= 1260.541:
        z += -0.0107467 * Q.sum_pt + 13.54665
    if 64.48544 <= Q.mass < 78.26182:
        z += 0.02247084 * Q.mass - 1.449042
    if 78.26182 <= Q.mass < 143.7876:
        z += 0.03319385 * Q.mass - 2.288245
    if 143.7876 <= Q.mass < 172.4888:
        z += -0.1421122 * Q.mass + 22.9186
    if Q.mass >= 172.4888:
        z += -0.242702 * Q.mass + 40.2692
    if Q.n_for_90pct < 14.0:
        z += 0.03792028 * Q.n_for_90pct - 0.5308839
    if Q.log_sum_pt < 6.811175:
        z += 13.79947 * Q.log_sum_pt - 94.74035
    if 6.811175 <= Q.log_sum_pt < 6.879399:
        z += 10.98899 * Q.log_sum_pt - 75.59767
    if Q.log_sum_pt >= 7.139296:
        z += 12.41264 * Q.log_sum_pt - 88.61751
    if Q.sum_pt_top40 < 1053.047:
        z += -0.004962351 * Q.sum_pt_top40 + 5.225591
    if 60.5496 <= Q.mass_top10 < 71.781:
        z += 0.008472007 * Q.mass_top10 - 0.5129766
    if Q.mass_top10 >= 71.781:
        z += -0.002215009 * Q.mass_top10 + 0.2541481
    if Q.sum_z_dr2_top40 < 0.02497133:
        z += -16.88863 * Q.sum_z_dr2_top40 + 0.4217315
    if 97.93004 <= Q.mass_top50 < 138.8977:
        z += -0.03346135 * Q.mass_top50 + 3.276872
    if 138.8977 <= Q.mass_top50 < 157.5448:
        z += 0.04852528 * Q.mass_top50 - 8.110881
    if Q.mass_top50 >= 157.5448:
        z += -0.07898363 * Q.mass_top50 + 11.97749
    if Q.n_dr_0p1_0p2 < 19.0:
        z += 0.01397934 * Q.n_dr_0p1_0p2 - 0.2656075
    if Q.sum_pt_top20 < 889.8383:
        z += 0.001472164 * Q.sum_pt_top20 - 1.309988
    if Q.sum_pt_top20 >= 1064.139:
        z += 0.003763365 * Q.sum_pt_top20 - 4.004743
    if 0.06030419 <= Q.mass_over_sum_pt < 0.07435617:
        z += -10.35311 * Q.mass_over_sum_pt + 0.6243359
    if Q.mass_over_sum_pt >= 0.07435617:
        z += -3.129865 * Q.mass_over_sum_pt + 0.0872431
    if Q.pt_entropy < 3.654944:
        z += 0.1717126 * Q.pt_entropy - 0.6275998
    if Q.sum_pt_top50 < 959.0957:
        z += -0.01517925 * Q.sum_pt_top50 + 15.12125
    if 959.0957 <= Q.sum_pt_top50 < 1008.935:
        z += -0.01129409 * Q.sum_pt_top50 + 11.39501
    if Q.psi_0p3 >= 0.9777125:
        z += 4.869331 * Q.psi_0p3 - 4.760806
    if Q.mass_top5 >= 37.76455:
        z += 0.01393776 * Q.mass_top5 - 0.5263531
    if Q.D2 >= 5.378975:
        z += -0.01997714 * Q.D2 + 0.1074565
    if Q.sum_pt_top15 >= 951.1375:
        z += -0.002694194 * Q.sum_pt_top15 + 2.562549
    if Q.pt_9 < 20.54688:
        z += 0.01101494 * Q.pt_9 - 0.2263227
    if Q.mass_top40 >= 150.0144:
        z += 0.07305034 * Q.mass_top40 - 10.95861
    if Q.tau1 < 0.08786745:
        z += -6.49482 * Q.tau1 + 0.5706833
    if Q.z_dr_0_0p05 >= 0.8103116:
        z += -0.9023362 * Q.z_dr_0_0p05 + 0.7311735
    if Q.max_dr >= 0.3315857:
        z += -0.7118928 * Q.max_dr + 0.2360535
    if Q.lam1 >= 0.01649354:
        z += 15.46181 * Q.lam1 - 0.2550201
    if Q.sum_pt_top30 < 950.9324:
        z += 0.0032983 * Q.sum_pt_top30 - 3.13646
    if Q.sum_pt < 1085.125 and Q.sum_pt_top30 < 800.732:
        z += 1.24953e-05 * (1085.125 - Q.sum_pt) * (800.732 - Q.sum_pt_top30)
    if Q.n_for_90pct < 14.0 and Q.D2 < 3.345339:
        z += 0.0334502 * (14.0 - Q.n_for_90pct) * (3.345339 - Q.D2)
    if Q.log_sum_pt > 7.139296 and Q.C3 < 0.00317537:
        z += -1007.903 * (Q.log_sum_pt - 7.139296) * (0.00317537 - Q.C3)
    if Q.sum_pt < 1012.673 and Q.z_9 < 0.01833434:
        z += 1.03595 * (1012.673 - Q.sum_pt) * (0.01833434 - Q.z_9)
    if Q.mass > 172.4888 and Q.pt_9 < 41.4375:
        z += 0.02143414 * (Q.mass - 172.4888) * (41.4375 - Q.pt_9)
    if Q.sum_pt_top40 < 1053.047 and Q.D3 < 0.5260785:
        z += 0.008273774 * (1053.047 - Q.sum_pt_top40) * (0.5260785 - Q.D3)
    if Q.sum_pt_top20 < 889.8383 and Q.z_9 < 0.01833434:
        z += -0.8604661 * (889.8383 - Q.sum_pt_top20) * (0.01833434 - Q.z_9)
    if Q.mass > 172.4888 and Q.z_9 < 0.0301005:
        z += -19.27681 * (Q.mass - 172.4888) * (0.0301005 - Q.z_9)
    if Q.sum_pt < 1085.125 and Q.dr_12 < 0.1269782:
        z += 0.01744054 * (1085.125 - Q.sum_pt) * (0.1269782 - Q.dr_12)
    if Q.mass_top50 > 157.5448 and Q.sd_zg > 0.4462823:
        z += -0.5789284 * (Q.mass_top50 - 157.5448) * (Q.sd_zg - 0.4462823)
    if Q.mass_top50 > 97.93004 and Q.soft3_pt < 2.873047:
        z += 0.004715123 * (Q.mass_top50 - 97.93004) * (2.873047 - Q.soft3_pt)
    if Q.mass_top5 > 37.76455 and Q.n_dr_0p05_0p1 < 27.0:
        z += -0.0004505646 * (Q.mass_top5 - 37.76455) * (27.0 - Q.n_dr_0p05_0p1)
    if Q.mass_top50 > 97.93004 and Q.soft7_z < 0.0006388081:
        z += -73.27068 * (Q.mass_top50 - 97.93004) * (0.0006388081 - Q.soft7_z)
    if Q.psi_0p3 > 0.9777125 and Q.max_dr > 0.2738063:
        z += 48.72442 * (Q.psi_0p3 - 0.9777125) * (Q.max_dr - 0.2738063)
    if Q.log_sum_pt < 6.811175 and Q.eta_1 < -0.0892334:
        z += -501.6935 * (6.811175 - Q.log_sum_pt) * (-0.0892334 - Q.eta_1)
    if Q.sum_pt < 1085.125 and Q.dr_13 < 0.1023813:
        z += 0.01331195 * (1085.125 - Q.sum_pt) * (0.1023813 - Q.dr_13)
    if Q.mass > 172.4888 and Q.soft6_z < 0.002357824:
        z += -21.04573 * (Q.mass - 172.4888) * (0.002357824 - Q.soft6_z)
    if Q.mass_top50 > 97.93004 and Q.soft7_pt < 0.6713867:
        z += 0.05572783 * (Q.mass_top50 - 97.93004) * (0.6713867 - Q.soft7_pt)
    if Q.mass > 64.48544 and Q.zdr_12 < 0.002227078:
        z += -1.576421 * (Q.mass - 64.48544) * (0.002227078 - Q.zdr_12)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.2437716
    if Q.tau21_b2 < 0.342495:
        z += -1.868561 * Q.tau21_b2 + 0.6399727
    if Q.mass < 79.65241:
        z += 0.1009362 * Q.mass - 9.342055
    if 79.65241 <= Q.mass < 82.85409:
        z += 0.1670643 * Q.mass - 14.60932
    if 82.85409 <= Q.mass < 91.03469:
        z += 0.1147099 * Q.mass - 10.27155
    if 91.03469 <= Q.mass < 121.3913:
        z += -0.005634278 * Q.mass + 0.6839523
    if Q.psi_0p3 >= 0.9924477:
        z += 18.95633 * Q.psi_0p3 - 18.81317
    if Q.N2 < 0.4226723:
        z += 1.144262 * Q.N2 - 0.4836478
    if Q.n_dr_0p1_0p2 < 19.0:
        z += 0.01102776 * Q.n_dr_0p1_0p2 - 0.2095275
    if Q.sj3_mass1 < 32.50209:
        z += -0.005636338 * Q.sj3_mass1 + 0.1831928
    if Q.e2 < 0.02210818:
        z += -45.45241 * Q.e2 + 1.448317
    if 0.02210818 <= Q.e2 < 0.03029714:
        z += -58.61819 * Q.e2 + 1.739389
    if 0.03029714 <= Q.e2 < 0.03680582:
        z += -24.11102 * Q.e2 + 0.6939199
    if 0.03680582 <= Q.e2 < 0.04358622:
        z += -13.16578 * Q.e2 + 0.2910714
    if Q.e2 >= 0.04358622:
        z += 4.133476 * Q.e2 - 0.4629377
    if Q.mass_over_sum_pt < 0.08873143:
        z += -26.77655 * Q.mass_over_sum_pt + 4.085293
    if 0.08873143 <= Q.mass_over_sum_pt < 0.09046749:
        z += 23.08392 * Q.mass_over_sum_pt - 0.3388976
    if 0.09046749 <= Q.mass_over_sum_pt < 0.09795415:
        z += -85.62165 * Q.mass_over_sum_pt + 9.495423
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -54.67831 * Q.mass_over_sum_pt + 6.464395
    if Q.sum_z_dr2_top20 < 0.006374178:
        z += 42.75944 * Q.sum_z_dr2_top20 - 0.7965067
    if 0.006374178 <= Q.sum_z_dr2_top20 < 0.01083435:
        z += 117.4731 * Q.sum_z_dr2_top20 - 1.272745
    if Q.mass_over_sum_pt_sq < 0.01986381:
        z += -220.2179 * Q.mass_over_sum_pt_sq + 4.374366
    if Q.soft7_z < 0.001595561:
        z += 200.2352 * Q.soft7_z - 0.3194874
    if Q.n_pt_above_10 >= 11.0:
        z += -0.02126314 * Q.n_pt_above_10 + 0.2338946
    if Q.mass_top40 < 67.72643:
        z += -0.05993372 * Q.mass_top40 + 4.524793
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += -0.02985402 * Q.mass_top40 + 2.487602
    if Q.sum_z_dr2_top40 < 0.01292642:
        z += 291.8231 * Q.sum_z_dr2_top40 - 4.264009
    if 0.01292642 <= Q.sum_z_dr2_top40 < 0.01897915:
        z += 81.24934 * Q.sum_z_dr2_top40 - 1.542043
    if Q.tau2 < 0.05936026:
        z += 7.959907 * Q.tau2 - 0.4725022
    if Q.e3 < 7.876005e-05:
        z += 3375.778 * Q.e3 - 0.2658765
    if Q.tau1 < 0.1008497:
        z += 21.72021 * Q.tau1 - 2.190477
    if Q.dr_3 < 0.03948167:
        z += -4.94477 * Q.dr_3 + 0.1952278
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.01024122 * Q.sum_pt_top50 + 9.822307
    if Q.log_sum_pt >= 6.935549:
        z += 7.179636 * Q.log_sum_pt - 49.79472
    if Q.mass_top50 >= 157.5448:
        z += -0.9894171 * Q.mass_top50 + 155.8775
    if Q.sum_pt_top40 >= 906.6023:
        z += 0.003621978 * Q.sum_pt_top40 - 3.283693
    if Q.sum_z_dr2_top10 < 0.007678544:
        z += -40.926 * Q.sum_z_dr2_top10 + 0.2326662
    if 0.007678544 <= Q.sum_z_dr2_top10 < 0.008956554:
        z += 63.83822 * Q.sum_z_dr2_top10 - 0.5717705
    if Q.lam2 < 0.001411894:
        z += -98.71939 * Q.lam2 + 0.1393813
    if Q.z_top50_slots < 0.9586536:
        z += 12.61621 * Q.z_top50_slots - 12.09457
    if Q.tau21_b2 < 0.342495 and Q.sum_z_dr2_top40 < 0.007709916:
        z += -611.3184 * (0.342495 - Q.tau21_b2) * (0.007709916 - Q.sum_z_dr2_top40)
    if Q.tau21_b2 < 0.342495 and Q.lam1 < 0.02048524:
        z += -116.7466 * (0.342495 - Q.tau21_b2) * (0.02048524 - Q.lam1)
    if Q.tau21_b2 < 0.342495 and Q.sum_z_dr2_top50 < 0.006118006:
        z += 1008.109 * (0.342495 - Q.tau21_b2) * (0.006118006 - Q.sum_z_dr2_top50)
    if Q.mass < 82.85409 and Q.n_for_50pct > 5.0:
        z += -0.003082962 * (82.85409 - Q.mass) * (Q.n_for_50pct - 5.0)
    if Q.sum_z_dr2_top20 < 0.01083435 and Q.mass_top3 > 16.89912:
        z += -2.176745 * (0.01083435 - Q.sum_z_dr2_top20) * (Q.mass_top3 - 16.89912)
    if Q.tau21_b2 < 0.342495 and Q.C2_b2 < 0.01691783:
        z += 70.68212 * (0.342495 - Q.tau21_b2) * (0.01691783 - Q.C2_b2)
    if Q.psi_0p3 > 0.9924477 and Q.dr_3 < 0.0458688:
        z += -792.3967 * (Q.psi_0p3 - 0.9924477) * (0.0458688 - Q.dr_3)
    if Q.mass < 121.3913 and Q.zdr_3 > 0.0003281655:
        z += -0.4464612 * (121.3913 - Q.mass) * (Q.zdr_3 - 0.0003281655)
    if Q.mass < 91.03469 and Q.soft8_z < 0.002405315:
        z += 5.880292 * (91.03469 - Q.mass) * (0.002405315 - Q.soft8_z)
    if Q.psi_0p3 > 0.9924477 and Q.pt_3 > 60.59375:
        z += 0.41873 * (Q.psi_0p3 - 0.9924477) * (Q.pt_3 - 60.59375)
    if Q.psi_0p3 > 0.9924477 and Q.soft6_pt < 4.250195:
        z += 6.014457 * (Q.psi_0p3 - 0.9924477) * (4.250195 - Q.soft6_pt)
    if Q.tau21_b2 < 0.342495 and Q.sj2_zsoft < 0.1433879:
        z += -13.46979 * (0.342495 - Q.tau21_b2) * (0.1433879 - Q.sj2_zsoft)
    if Q.tau21_b2 < 0.342495 and Q.z_6 < 0.05799644:
        z += 22.48019 * (0.342495 - Q.tau21_b2) * (0.05799644 - Q.z_6)
    if Q.psi_0p3 > 0.9924477 and Q.N3 < 1.059188:
        z += 42.51611 * (Q.psi_0p3 - 0.9924477) * (1.059188 - Q.N3)
    if Q.tau21_b2 < 0.342495 and Q.sj2_zsoft < 0.34119:
        z += 4.612316 * (0.342495 - Q.tau21_b2) * (0.34119 - Q.sj2_zsoft)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.08671004
    if Q.z_dr_0p1_0p2 < 0.06472584:
        z += -9.790044 * Q.z_dr_0p1_0p2 + 1.000792
    if 0.06472584 <= Q.z_dr_0p1_0p2 < 0.1203437:
        z += -6.600811 * Q.z_dr_0p1_0p2 + 0.7943659
    if Q.sum_z_dr2_top5 < 0.002270363:
        z += 43.17254 * Q.sum_z_dr2_top5 + 0.4688322
    if 0.002270363 <= Q.sum_z_dr2_top5 < 0.008329695:
        z += -93.54983 * Q.sum_z_dr2_top5 + 0.7792416
    if Q.sum_z_dr2_top2 < 0.0005124533:
        z += -67.4052 * Q.sum_z_dr2_top2 - 0.05713106
    if 0.0005124533 <= Q.sum_z_dr2_top2 < 0.001056655:
        z += 168.4542 * Q.sum_z_dr2_top2 - 0.177998
    if 0.8976117 <= Q.psi_0p1 < 0.9371031:
        z += -6.699335 * Q.psi_0p1 + 6.013401
    if Q.psi_0p1 >= 0.9371031:
        z += -10.45303 * Q.psi_0p1 + 9.530999
    if Q.log_sum_pt < 6.903423:
        z += -4.814595 * Q.log_sum_pt + 34.14798
    if 6.903423 <= Q.log_sum_pt < 7.017258:
        z += -8.001035 * Q.log_sum_pt + 56.14532
    if Q.psi_0p3 >= 0.9896594:
        z += -30.78449 * Q.psi_0p3 + 30.46616
    if Q.sum_pt < 1002.379:
        z += 0.007652513 * Q.sum_pt - 7.670715
    if 0.1666442 <= Q.sj2_dr < 0.2232169:
        z += -2.154462 * Q.sj2_dr + 0.3590286
    if Q.sj2_dr >= 0.2232169:
        z += 1.055854 * Q.sj2_dr - 0.3575683
    if Q.sum_z_dr2_top10 < 0.007678544:
        z += -58.24136 * Q.sum_z_dr2_top10 + 0.4472088
    if Q.lam1 < 0.004673423:
        z += -75.0257 * Q.lam1 + 0.06291575
    if 0.004673423 <= Q.lam1 < 0.006189818:
        z += 40.69102 * Q.lam1 - 0.4778775
    if 0.006189818 <= Q.lam1 < 0.01174405:
        z += 119.0308 * Q.lam1 - 0.9627864
    if 0.01174405 <= Q.lam1 < 0.01649354:
        z += 78.33978 * Q.lam1 - 0.484909
    if Q.lam1 >= 0.01649354:
        z += -16.69795 * Q.lam1 + 1.0826
    if Q.n_dr_0p1_0p2 < 13.0:
        z += -0.03281971 * Q.n_dr_0p1_0p2 + 0.4266563
    if Q.D2 < 1.976207:
        z += 0.1821252 * Q.D2 - 0.3599171
    if Q.lam2 < 0.001776308:
        z += -208.9894 * Q.lam2 + 0.3712297
    if Q.tau1 < 0.0705748:
        z += 17.57166 * Q.tau1 - 1.240116
    if Q.sum_pt_top20 < 869.693:
        z += 0.0006279137 * Q.sum_pt_top20 - 0.8065589
    if 869.693 <= Q.sum_pt_top20 < 1017.778:
        z += 0.001758898 * Q.sum_pt_top20 - 1.790168
    if Q.n_dr_0p2_0p4 >= 9.0:
        z += -0.01537122 * Q.n_dr_0p2_0p4 + 0.138341
    if Q.sd_mass < 40.97891:
        z += -0.002112051 * Q.sd_mass + 0.4867419
    if 40.97891 <= Q.sd_mass < 79.18312:
        z += -0.01047509 * Q.sd_mass + 0.82945
    if Q.mass_top50 < 52.15154:
        z += -0.004704912 * Q.mass_top50 - 0.03104252
    if 52.15154 <= Q.mass_top50 < 71.79516:
        z += 0.01407128 * Q.mass_top50 - 1.01025
    if Q.sum_pt_top40 < 906.6023:
        z += -8.058338e-05 * Q.sum_pt_top40 - 0.07577705
    if 906.6023 <= Q.sum_pt_top40 < 1007.44:
        z += 0.001475977 * Q.sum_pt_top40 - 1.486958
    if Q.sum_pt_top30 < 1018.832:
        z += 0.001561856 * Q.sum_pt_top30 - 1.591268
    if Q.sum_z_dr < 0.07031778:
        z += -24.80049 * Q.sum_z_dr + 2.140323
    if 0.07031778 <= Q.sum_z_dr < 0.08068193:
        z += -38.24794 * Q.sum_z_dr + 3.085917
    if Q.sj2_zsoft < 0.2416266:
        z += -0.6094023 * Q.sj2_zsoft + 0.1472478
    if Q.mass_top30 < 68.28697:
        z += -0.009069821 * Q.mass_top30 + 0.8182134
    if 68.28697 <= Q.mass_top30 < 80.24626:
        z += -0.01662831 * Q.mass_top30 + 1.334359
    if Q.e2 >= 0.03029714:
        z += -31.50445 * Q.e2 + 0.9544947
    if Q.e3 < 0.0005178279:
        z += 1349.492 * Q.e3 - 0.6988046
    if Q.mass_over_sum_pt < 0.08329945:
        z += -0.2184486 * Q.mass_over_sum_pt + 0.9337091
    if 0.08329945 <= Q.mass_over_sum_pt < 0.140939:
        z += -15.8834 * Q.mass_over_sum_pt + 2.238591
    if Q.LHA < 0.2091025:
        z += 3.710737 * Q.LHA - 1.62945
    if 0.2091025 <= Q.LHA < 0.302389:
        z += 9.149503 * Q.LHA - 2.766709
    if 0.1717401 <= Q.sj3_dr_max < 0.2442262:
        z += -1.486571 * Q.sj3_dr_max + 0.2553039
    if 0.2442262 <= Q.sj3_dr_max < 0.2841485:
        z += 4.433396 * Q.sj3_dr_max - 1.190507
    if Q.sj3_dr_max >= 0.2841485:
        z += -0.9627329 * Q.sj3_dr_max + 0.3427943
    if Q.sum_z_dr2_top30 < 0.01215787:
        z += 40.60063 * Q.sum_z_dr2_top30 - 0.4936173
    if Q.mass < 92.85979:
        z += 0.01426048 * Q.mass - 1.324226
    if Q.max_dr < 0.2982:
        z += 1.595332 * Q.max_dr - 0.475728
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.sd_mass < 76.29481:
        z += -0.0427799 * (0.1203437 - Q.z_dr_0p1_0p2) * (76.29481 - Q.sd_mass)
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 949.9169:
        z += 0.003755547 * (0.8459004 - Q.z_dr_0_0p05) * (949.9169 - Q.sum_pt)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.psi_0p2 < 0.948102:
        z += -437.1228 * (0.008329695 - Q.sum_z_dr2_top5) * (0.948102 - Q.psi_0p2)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_dr_0p2_0p4 > 5.0:
        z += -0.1513539 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p2_0p4 - 5.0)
    if Q.sj2_dr > 0.2232169 and Q.C2_b2 < 0.04008677:
        z += 96.26366 * (Q.sj2_dr - 0.2232169) * (0.04008677 - Q.C2_b2)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.z_6 < 0.04770182:
        z += 650.5613 * (0.008329695 - Q.sum_z_dr2_top5) * (0.04770182 - Q.z_6)
    if Q.psi_0p3 > 0.9896594 and Q.sj2_mass2 < 14.45919:
        z += -1.200783 * (Q.psi_0p3 - 0.9896594) * (14.45919 - Q.sj2_mass2)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.sj3_mass1 > 5.112677:
        z += -1.171293 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.sj3_mass1 - 5.112677)
    if Q.n_dr_0p1_0p2 < 13.0 and Q.absphi_0 < 0.02970886:
        z += -0.7137685 * (13.0 - Q.n_dr_0p1_0p2) * (0.02970886 - Q.absphi_0)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.n_real_top40 > 22.0:
        z += -1.888819 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.n_real_top40 - 22.0)
    if Q.sum_z_dr2_top20 < 0.00287991 and Q.max_dr < 0.4357228:
        z += 638.123 * (0.00287991 - Q.sum_z_dr2_top20) * (0.4357228 - Q.max_dr)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.sj3_pairmin_over_m > 0.09540583:
        z += -45.89288 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.sj3_pairmin_over_m - 0.09540583)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.dr0_8 > 0.2441824:
        z += 779.2861 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.dr0_8 - 0.2441824)
    if Q.sum_z_dr2_top5 < 0.002270363 and Q.dr0_8 > 0.2441824:
        z += -2518.095 * (0.002270363 - Q.sum_z_dr2_top5) * (Q.dr0_8 - 0.2441824)
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
