"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; all observables, tuned for agreement), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
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
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
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
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
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
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -1.832874
    if Q.planar_flow < 0.1484197:
        z += -5.969105 * Q.planar_flow + 0.8859328
    if Q.lam1_plus_lam2 < 0.004372139:
        z += 314.694 * Q.lam1_plus_lam2 + 0.6825144
    if 0.004372139 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -478.0412 * Q.lam1_plus_lam2 + 4.148463
    if Q.sum_z_dr2 < 0.01323868:
        z += -271.3434 * Q.sum_z_dr2 + 3.548758
    if 0.01323868 <= Q.sum_z_dr2 < 0.01882765:
        z += 7.777606 * Q.sum_z_dr2 - 0.1464341
    if Q.mass < 21.78408:
        z += 0.00340716 * Q.mass - 1.502272
    if 21.78408 <= Q.mass < 29.6447:
        z += 0.03048867 * Q.mass - 2.092218
    if 29.6447 <= Q.mass < 56.92035:
        z += 0.04293519 * Q.mass - 2.461191
    if 56.92035 <= Q.mass < 64.61873:
        z += 0.002247956 * Q.mass - 0.1452601
    if Q.sum_z_dr2_top3 < 0.006756161:
        z += 16.04288 * Q.sum_z_dr2_top3 - 0.2264324
    if 0.006756161 <= Q.sum_z_dr2_top3 < 0.007929074:
        z += 100.6418 * Q.sum_z_dr2_top3 - 0.7979962
    if Q.lam1 < 0.0002758826:
        z += 6435.183 * Q.lam1 - 3.739424
    if 0.0002758826 <= Q.lam1 < 0.005433361:
        z += 379.4157 * Q.lam1 - 2.068743
    if 0.005433361 <= Q.lam1 < 0.006506576:
        z += 6.746752 * Q.lam1 - 0.04389825
    if Q.sum_pt >= 901.5938:
        z += -0.009006218 * Q.sum_pt + 8.11995
    if Q.C2_b2 < 0.001563465:
        z += 193.929 * Q.C2_b2 - 0.3032014
    if Q.sj3_dr_max < 0.04889979:
        z += -14.14064 * Q.sj3_dr_max + 2.947058
    if 0.04889979 <= Q.sj3_dr_max < 0.1070199:
        z += -8.940023 * Q.sj3_dr_max + 2.692749
    if 0.1070199 <= Q.sj3_dr_max < 0.1789613:
        z += 8.652359 * Q.sj3_dr_max + 0.8100135
    if 0.1789613 <= Q.sj3_dr_max < 0.233678:
        z += -5.13332 * Q.sj3_dr_max + 3.277117
    if 0.233678 <= Q.sj3_dr_max < 0.3012016:
        z += -23.34351 * Q.sj3_dr_max + 7.532437
    if Q.sj3_dr_max >= 0.3012016:
        z += -14.40348 * Q.sj3_dr_max + 4.839688
    if 0.006789738 <= Q.centroid_offset < 0.02076709:
        z += -9.523377 * Q.centroid_offset + 0.06466124
    if 0.02076709 <= Q.centroid_offset < 0.04990367:
        z += -18.18147 * Q.centroid_offset + 0.2444646
    if Q.centroid_offset >= 0.04990367:
        z += -43.46677 * Q.centroid_offset + 1.506294
    if Q.sum_z_dr < 0.07608178:
        z += 54.15323 * Q.sum_z_dr - 4.348086
    if 0.07608178 <= Q.sum_z_dr < 0.08723651:
        z += 20.44082 * Q.sum_z_dr - 1.783186
    if Q.e2 < 0.0245477:
        z += -72.18186 * Q.e2 + 2.141353
    if 0.0245477 <= Q.e2 < 0.03556091:
        z += -33.54645 * Q.e2 + 1.192942
    if Q.sum_z_dr2_top5 < 0.008329695:
        z += 49.7185 * Q.sum_z_dr2_top5 - 0.41414
    if Q.log_sum_pt < 6.080494:
        z += 8.32469 * Q.log_sum_pt - 50.61823
    if 6.377723 <= Q.log_sum_pt < 6.670067:
        z += 2.258485 * Q.log_sum_pt - 14.40399
    if Q.log_sum_pt >= 6.670067:
        z += -6.401287 * Q.log_sum_pt + 43.35727
    if Q.mass_over_sum_pt_sq < 0.003904593:
        z += -330.4725 * Q.mass_over_sum_pt_sq + 1.290361
    if Q.n_dr_0_0p05 < 1.0:
        z += 0.327428 * Q.n_dr_0_0p05 - 0.327428
    if Q.n_dr_0_0p05 >= 4.0:
        z += 0.1715326 * Q.n_dr_0_0p05 - 0.6861303
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.009099075 * Q.sum_pt_top5 - 6.255045
    if Q.lam2 < 7.300726e-05:
        z += 4045.56 * Q.lam2 - 0.2953552
    if Q.zdr_1 < 0.01587232:
        z += -10.40511 * Q.zdr_1 + 0.1651532
    if Q.mass_over_sum_pt < 0.08475161:
        z += 15.42618 * Q.mass_over_sum_pt - 0.6238311
    if 0.08475161 <= Q.mass_over_sum_pt < 0.1309286:
        z += -14.80308 * Q.mass_over_sum_pt + 1.938147
    if Q.sd_mass < 10.97038:
        z += -0.003437049 * Q.sd_mass + 0.03770572
    if Q.planar_flow < 0.1484197 and Q.centroid_offset < 0.04990367:
        z += 159.7375 * (0.1484197 - Q.planar_flow) * (0.04990367 - Q.centroid_offset)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.D2 < 0.7459513:
        z += -4328.79 * (0.004372139 - Q.lam1_plus_lam2) * (0.7459513 - Q.D2)
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += -964.0042 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.D2 < 1.002471:
        z += 230.1663 * (0.01323868 - Q.sum_z_dr2) * (1.002471 - Q.D2)
    if Q.planar_flow < 0.1484197 and Q.sum_pt_top2 < 358.375:
        z += -0.01635947 * (0.1484197 - Q.planar_flow) * (358.375 - Q.sum_pt_top2)
    if Q.sum_pt > 901.5938 and Q.dr12 < 0.119512:
        z += -0.02007443 * (Q.sum_pt - 901.5938) * (0.119512 - Q.dr12)
    if Q.sum_z_dr2 < 0.01323868 and Q.mean_phi < -0.003096655:
        z += 209.2172 * (0.01323868 - Q.sum_z_dr2) * (-0.003096655 - Q.mean_phi)
    if Q.sum_z_dr2 < 0.01882765 and Q.mass_top2 > 28.78966:
        z += -7.193494 * (0.01882765 - Q.sum_z_dr2) * (Q.mass_top2 - 28.78966)
    if Q.sum_z_dr2 < 0.01323868 and Q.centroid_offset > 0.01837778:
        z += -7971.26 * (0.01323868 - Q.sum_z_dr2) * (Q.centroid_offset - 0.01837778)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.eccentricity > 0.9598562:
        z += -13617.95 * (0.004372139 - Q.lam1_plus_lam2) * (Q.eccentricity - 0.9598562)
    if Q.mass < 29.6447 and Q.phi_0 < 0.01452637:
        z += -0.119841 * (29.6447 - Q.mass) * (0.01452637 - Q.phi_0)
    if Q.sum_pt > 901.5938 and Q.absphi_3 < 0.03363037:
        z += -0.1183293 * (Q.sum_pt - 901.5938) * (0.03363037 - Q.absphi_3)
    if Q.sum_z_dr2 < 0.01323868 and Q.phi_0 > -0.008995056:
        z += 1366.563 * (0.01323868 - Q.sum_z_dr2) * (Q.phi_0 - -0.008995056)
    if Q.lam1_plus_lam2 < 0.008678045 and Q.eta_7 > 0.1219513:
        z += 72.12639 * (0.008678045 - Q.lam1_plus_lam2) * (Q.eta_7 - 0.1219513)
    if Q.sum_pt > 901.5938 and Q.eccentricity > 0.9458207:
        z += -0.2483227 * (Q.sum_pt - 901.5938) * (Q.eccentricity - 0.9458207)
    if Q.planar_flow < 0.1484197 and Q.z_7 < 0.02320757:
        z += 5163.549 * (0.1484197 - Q.planar_flow) * (0.02320757 - Q.z_7)
    if Q.lam1 < 0.005433361 and Q.eta_6 > 0.05477905:
        z += 3208.381 * (0.005433361 - Q.lam1) * (Q.eta_6 - 0.05477905)
    if Q.sum_pt > 901.5938 and Q.dr0_6 > 0.2347949:
        z += 0.6860136 * (Q.sum_pt - 901.5938) * (Q.dr0_6 - 0.2347949)
    if Q.log_sum_pt > 6.670067 and Q.dr_4 < 0.07232166:
        z += 143.7504 * (Q.log_sum_pt - 6.670067) * (0.07232166 - Q.dr_4)
    if Q.log_sum_pt > 6.670067 and Q.phi_2 > 0.09649963:
        z += -132.5924 * (Q.log_sum_pt - 6.670067) * (Q.phi_2 - 0.09649963)
    if Q.sum_z_dr2 < 0.01882765 and Q.pt_6 < 46.125:
        z += 0.2564204 * (0.01882765 - Q.sum_z_dr2) * (46.125 - Q.pt_6)
    if Q.sum_z_dr2 < 0.01323868 and Q.sj3_dr12 > 0.2182873:
        z += -2387.233 * (0.01323868 - Q.sum_z_dr2) * (Q.sj3_dr12 - 0.2182873)
    if Q.mass < 64.61873 and Q.D2_b2 < 0.1830092:
        z += -0.105704 * (64.61873 - Q.mass) * (0.1830092 - Q.D2_b2)
    if Q.lam2 < 7.300726e-05 and Q.D2_b2 < 0.2669656:
        z += 57388.89 * (7.300726e-05 - Q.lam2) * (0.2669656 - Q.D2_b2)
    if Q.sj3_dr_max < 0.3012016 and Q.D2_b2 < 0.05744392:
        z += -50.64384 * (0.3012016 - Q.sj3_dr_max) * (0.05744392 - Q.D2_b2)
    if Q.sj3_dr_max < 0.3012016 and Q.phi_1 > 0.01467133:
        z += 10.59755 * (0.3012016 - Q.sj3_dr_max) * (Q.phi_1 - 0.01467133)
    if Q.n_dr_0_0p05 > 4.0 and Q.pair_mass_0_5 > 16.26599:
        z += -0.01698068 * (Q.n_dr_0_0p05 - 4.0) * (Q.pair_mass_0_5 - 16.26599)
    if Q.lam1 < 0.005433361 and Q.phi_3 > 0.04800415:
        z += -7258.474 * (0.005433361 - Q.lam1) * (Q.phi_3 - 0.04800415)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.pair_mass_0_6 > 20.59519:
        z += 371.2338 * (0.004372139 - Q.lam1_plus_lam2) * (Q.pair_mass_0_6 - 20.59519)
    if Q.planar_flow < 0.1484197 and Q.dr0_6 > 0.1755206:
        z += -116.7526 * (0.1484197 - Q.planar_flow) * (Q.dr0_6 - 0.1755206)
    if Q.mass < 56.92035 and Q.dr_max_012 > 0.06112084:
        z += -0.05760659 * (56.92035 - Q.mass) * (Q.dr_max_012 - 0.06112084)
    if Q.log_sum_pt > 6.377723 and Q.mean_eta > 6.288824e-05:
        z += 1.593602 * (Q.log_sum_pt - 6.377723) * (Q.mean_eta - 6.288824e-05)
    if Q.sum_pt_top5 > 687.4375 and Q.dr_4 < 0.06336451:
        z += -0.007213458 * (Q.sum_pt_top5 - 687.4375) * (0.06336451 - Q.dr_4)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.eta_2 > 0.02253723:
        z += -932.1002 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.eta_2 - 0.02253723)
    if Q.lam2 < 7.300726e-05 and Q.dr_2 < 0.08525808:
        z += 102023.4 * (7.300726e-05 - Q.lam2) * (0.08525808 - Q.dr_2)
    if Q.sum_pt > 901.5938 and Q.pt_7 > 33.21875:
        z += -0.0002400607 * (Q.sum_pt - 901.5938) * (Q.pt_7 - 33.21875)
    if Q.n_dr_0_0p05 > 4.0 and Q.dr_4 > 0.1999753:
        z += 24.17485 * (Q.n_dr_0_0p05 - 4.0) * (Q.dr_4 - 0.1999753)
    if Q.planar_flow < 0.1484197 and Q.dr_4 > 0.03006824:
        z += 4.095028 * (0.1484197 - Q.planar_flow) * (Q.dr_4 - 0.03006824)
    if Q.mass < 56.92035 and Q.D2_b2 < 0.05744392:
        z += 0.2765523 * (56.92035 - Q.mass) * (0.05744392 - Q.D2_b2)
    if Q.mass < 29.6447 and Q.D2_b2 < 0.1830092:
        z += -4.660262 * (29.6447 - Q.mass) * (0.1830092 - Q.D2_b2)
    if Q.log_sum_pt > 6.670067 and Q.z_7 > 0.01685855:
        z += -373.0849 * (Q.log_sum_pt - 6.670067) * (Q.z_7 - 0.01685855)
    if Q.sj3_dr_max < 0.3012016 and Q.z_7 < 0.05557716:
        z += 78.52505 * (0.3012016 - Q.sj3_dr_max) * (0.05557716 - Q.z_7)
    if Q.C2_b2 < 0.001563465 and Q.zdr_6 > 0.01392641:
        z += -151106.6 * (0.001563465 - Q.C2_b2) * (Q.zdr_6 - 0.01392641)
    if Q.sum_pt_top5 > 687.4375 and Q.eta_5 < -0.07373047:
        z += 0.1147395 * (Q.sum_pt_top5 - 687.4375) * (-0.07373047 - Q.eta_5)
    if Q.sj3_dr_max < 0.3012016 and Q.n_dr_0p05_0p1 < 2.0:
        z += 0.2605046 * (0.3012016 - Q.sj3_dr_max) * (2.0 - Q.n_dr_0p05_0p1)
    if Q.log_sum_pt > 6.670067 and Q.m01 < 28.78966:
        z += 0.1563745 * (Q.log_sum_pt - 6.670067) * (28.78966 - Q.m01)
    if Q.planar_flow < 0.1484197 and Q.dr_2 < 0.02270492:
        z += -1111.485 * (0.1484197 - Q.planar_flow) * (0.02270492 - Q.dr_2)
    if Q.sum_z_dr2_top3 < 0.007929074 and Q.D2_b2 < 0.5327104:
        z += -81.28738 * (0.007929074 - Q.sum_z_dr2_top3) * (0.5327104 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.01882765 and Q.phi_2 < -0.0966217:
        z += 8475.724 * (0.01882765 - Q.sum_z_dr2) * (-0.0966217 - Q.phi_2)
    if Q.centroid_offset > 0.02076709 and Q.z_7 < 0.06473447:
        z += -4243.165 * (Q.centroid_offset - 0.02076709) * (0.06473447 - Q.z_7)
    if Q.planar_flow < 0.1484197 and Q.pt_dispersion < 0.4160096:
        z += 189.7615 * (0.1484197 - Q.planar_flow) * (0.4160096 - Q.pt_dispersion)
    if Q.centroid_offset > 0.006789738 and Q.n_pt_above_50 < 7.0:
        z += 1.014687 * (Q.centroid_offset - 0.006789738) * (7.0 - Q.n_pt_above_50)
    if Q.sum_pt > 901.5938 and Q.eta_2 < -0.09552002:
        z += -0.07806785 * (Q.sum_pt - 901.5938) * (-0.09552002 - Q.eta_2)
    if Q.centroid_offset > 0.02076709 and Q.dr_2 < 0.05056028:
        z += -4262.245 * (Q.centroid_offset - 0.02076709) * (0.05056028 - Q.dr_2)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.C3 < 0.03532852:
        z += 5464.787 * (0.004372139 - Q.lam1_plus_lam2) * (0.03532852 - Q.C3)
    if Q.centroid_offset > 0.006789738 and Q.pt_1 < 223.375:
        z += 0.1107207 * (Q.centroid_offset - 0.006789738) * (223.375 - Q.pt_1)
    if Q.log_sum_pt > 6.670067 and Q.phi_4 > 0.1096802:
        z += 411.1653 * (Q.log_sum_pt - 6.670067) * (Q.phi_4 - 0.1096802)
    if Q.sum_z_dr < 0.08723651 and Q.dr1_3 > 0.1637906:
        z += 1180.727 * (0.08723651 - Q.sum_z_dr) * (Q.dr1_3 - 0.1637906)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.04420901
    if Q.lam1 < 0.0008722282:
        z += 296.8752 * Q.lam1 + 2.157183
    if 0.0008722282 <= Q.lam1 < 0.00595415:
        z += -447.7295 * Q.lam1 + 2.806648
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += -58.14746 * Q.lam1 + 0.4870183
    if Q.pt_7 < 25.57812:
        z += -0.01327527 * Q.pt_7 + 0.3395566
    if 34.53125 <= Q.pt_7 < 53.4375:
        z += 0.05962934 * Q.pt_7 - 2.059076
    if Q.pt_7 >= 53.4375:
        z += 0.01038938 * Q.pt_7 + 0.5721847
    if 6.377723 <= Q.log_sum_pt < 6.502799:
        z += 2.084112 * Q.log_sum_pt - 13.29189
    if 6.502799 <= Q.log_sum_pt < 6.605974:
        z += 9.512514 * Q.log_sum_pt - 61.5973
    if Q.log_sum_pt >= 6.605974:
        z += 15.71728 * Q.log_sum_pt - 102.5858
    if Q.mass_over_sum_pt_sq < 0.005832932:
        z += -443.8072 * Q.mass_over_sum_pt_sq + 2.588697
    if Q.tau21_b2 < 0.006726135:
        z += -216.2598 * Q.tau21_b2 + 1.454593
    if Q.z_7 < 0.02320757:
        z += 43.56186 * Q.z_7 - 2.685379
    if 0.02320757 <= Q.z_7 < 0.04624032:
        z += 68.65281 * Q.z_7 - 3.267678
    if 0.04624032 <= Q.z_7 < 0.06164517:
        z += 51.83111 * Q.z_7 - 2.489837
    if 0.06164517 <= Q.z_7 < 0.08670959:
        z += 8.269243 * Q.z_7 + 0.1955412
    if Q.z_7 >= 0.08670959:
        z += 6.750105 * Q.z_7 + 0.3272651
    if Q.sum_z_dr2 < 0.005019719:
        z += 483.2499 * Q.sum_z_dr2 - 3.959136
    if 0.005019719 <= Q.sum_z_dr2 < 0.008678045:
        z += 419.1418 * Q.sum_z_dr2 - 3.637331
    if Q.mass < 49.6681:
        z += -0.04339353 * Q.mass + 2.155274
    if Q.sj3_dr_min >= 0.03628191:
        z += -2.365833 * Q.sj3_dr_min + 0.08583692
    if Q.sj3_dr_max >= 0.169029:
        z += 3.303761 * Q.sj3_dr_max - 0.5584314
    if Q.sum_zz_dr2 < 0.008168571:
        z += -219.4137 * Q.sum_zz_dr2 + 1.792296
    if Q.e3 < 1.627072e-05:
        z += -35315.06 * Q.e3 + 1.223722
    if 1.627072e-05 <= Q.e3 < 0.0005116989:
        z += -1310.222 * Q.e3 + 0.6704389
    if Q.n_dr_0_0p05 < 1.0:
        z += 0.4588433 * Q.n_dr_0_0p05 - 0.4588433
    if 367.5938 <= Q.sum_pt_top5 < 531.1875:
        z += 0.0005565617 * Q.sum_pt_top5 - 0.2045886
    if Q.sum_pt_top5 >= 531.1875:
        z += -0.01207908 * Q.sum_pt_top5 + 6.507309
    if Q.lam1_plus_lam2 < 0.00609665:
        z += 537.5845 * Q.lam1_plus_lam2 - 3.277465
    if Q.sum_z_dr < 0.0717028:
        z += -27.75584 * Q.sum_z_dr + 1.540537
    if 0.0717028 <= Q.sum_z_dr < 0.1019409:
        z += 14.86977 * Q.sum_z_dr - 1.515838
    if Q.centroid_offset < 0.01837778:
        z += 60.70504 * Q.centroid_offset - 1.115624
    if Q.tau1 < 0.07283629:
        z += 5.199175 * Q.tau1 - 0.3786886
    if Q.pt_6 >= 62.25:
        z += 0.0303339 * Q.pt_6 - 1.888285
    if Q.zdr_0 < 0.0211821:
        z += -7.034801 * Q.zdr_0 + 0.1490119
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += 4155.669 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.pt_7 > 34.53125 and Q.e3 < 2.955458e-05:
        z += -666.2371 * (Q.pt_7 - 34.53125) * (2.955458e-05 - Q.e3)
    if Q.log_sum_pt > 6.377723 and Q.e3 < 1.627072e-05:
        z += -232016.6 * (Q.log_sum_pt - 6.377723) * (1.627072e-05 - Q.e3)
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += 0.8828005 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.tau2 < 0.06297984:
        z += 8.792529 * (0.06164517 - Q.z_7) * (0.06297984 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.D2 < 1.679198:
        z += -23.73372 * (0.06164517 - Q.z_7) * (1.679198 - Q.D2)
    if Q.pt_7 > 34.53125 and Q.sj2_dr > 0.1294903:
        z += 0.6790324 * (Q.pt_7 - 34.53125) * (Q.sj2_dr - 0.1294903)
    if Q.log_sum_pt > 6.377723 and Q.z_dr_0p05_0p1 < 0.7509095:
        z += -4.49881 * (Q.log_sum_pt - 6.377723) * (0.7509095 - Q.z_dr_0p05_0p1)
    if Q.tau21_b2 < 0.006726135 and Q.sj3_pair_mass_min > 6.811308:
        z += 1251.656 * (0.006726135 - Q.tau21_b2) * (Q.sj3_pair_mass_min - 6.811308)
    if Q.log_sum_pt > 6.377723 and Q.sj3_pairmin_over_m < 0.1340569:
        z += 10.20517 * (Q.log_sum_pt - 6.377723) * (0.1340569 - Q.sj3_pairmin_over_m)
    if Q.z_7 < 0.06164517 and Q.z_dr_0p05_0p1 > 0.04875823:
        z += -64.87888 * (0.06164517 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.04875823)
    if Q.pt_7 > 34.53125 and Q.n_dr_0p1_0p2 > 1.0:
        z += 0.001606179 * (Q.pt_7 - 34.53125) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.pt_7 > 34.53125 and Q.pt_6 < 52.90625:
        z += -0.0103439 * (Q.pt_7 - 34.53125) * (52.90625 - Q.pt_6)
    if Q.log_sum_pt > 6.377723 and Q.centroid_offset > 0.009480685:
        z += 314.1182 * (Q.log_sum_pt - 6.377723) * (Q.centroid_offset - 0.009480685)
    if Q.log_sum_pt > 6.605974 and Q.D2 < 1.432482:
        z += 9.17725 * (Q.log_sum_pt - 6.605974) * (1.432482 - Q.D2)
    if Q.lam1 < 0.008375572 and Q.planar_flow < 0.1484197:
        z += -462.0807 * (0.008375572 - Q.lam1) * (0.1484197 - Q.planar_flow)
    if Q.pt_7 > 34.53125 and Q.z_dr_0p05_0p1 > 0.2919447:
        z += -0.07958854 * (Q.pt_7 - 34.53125) * (Q.z_dr_0p05_0p1 - 0.2919447)
    if Q.z_7 < 0.06164517 and Q.D3 < 3.916009:
        z += -0.4832604 * (0.06164517 - Q.z_7) * (3.916009 - Q.D3)
    if Q.mass < 49.6681 and Q.mean_phi > 0.002834884:
        z += -0.2319539 * (49.6681 - Q.mass) * (Q.mean_phi - 0.002834884)
    if Q.z_7 < 0.06164517 and Q.centroid_offset > 0.02076709:
        z += -787.9669 * (0.06164517 - Q.z_7) * (Q.centroid_offset - 0.02076709)
    if Q.sj3_dr_max > 0.169029 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.1463833 * (Q.sj3_dr_max - 0.169029) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.log_sum_pt > 6.605974 and Q.phi_0 > 0.07977295:
        z += 3.252923 * (Q.log_sum_pt - 6.605974) * (Q.phi_0 - 0.07977295)
    if Q.mass < 49.6681 and Q.mass_top2 > 0.6957161:
        z += 0.001327839 * (49.6681 - Q.mass) * (Q.mass_top2 - 0.6957161)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.D2_b2 < 0.03885671:
        z += -126397.1 * (0.005832932 - Q.mass_over_sum_pt_sq) * (0.03885671 - Q.D2_b2)
    if Q.log_sum_pt > 6.605974 and Q.zdr_4 > 0.002832174:
        z += 884.6595 * (Q.log_sum_pt - 6.605974) * (Q.zdr_4 - 0.002832174)
    if Q.z_7 < 0.06164517 and Q.pair_mass_0_4 > 13.64049:
        z += -1.210823 * (0.06164517 - Q.z_7) * (Q.pair_mass_0_4 - 13.64049)
    if Q.pt_7 > 53.4375 and Q.dr01 > 0.1894571:
        z += 0.06837517 * (Q.pt_7 - 53.4375) * (Q.dr01 - 0.1894571)
    if Q.n_dr_0_0p05 < 1.0 and Q.n_pt_above_50 > 5.0:
        z += 0.3563142 * (1.0 - Q.n_dr_0_0p05) * (Q.n_pt_above_50 - 5.0)
    if Q.lam1 < 0.008375572 and Q.n_pt_above_50 > 5.0:
        z += -13.55243 * (0.008375572 - Q.lam1) * (Q.n_pt_above_50 - 5.0)
    if Q.log_sum_pt > 6.605974 and Q.z_4 < 0.09925997:
        z += 59.90588 * (Q.log_sum_pt - 6.605974) * (0.09925997 - Q.z_4)
    if Q.sum_z_dr2 < 0.008678045 and Q.centroid_offset > 0.02076709:
        z += 16590.5 * (0.008678045 - Q.sum_z_dr2) * (Q.centroid_offset - 0.02076709)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.centroid_offset > 0.00809236:
        z += -3152.297 * (0.005832932 - Q.mass_over_sum_pt_sq) * (Q.centroid_offset - 0.00809236)
    if Q.n_dr_0_0p05 < 1.0 and Q.zdr_7 > 0.0007928864:
        z += -10.67723 * (1.0 - Q.n_dr_0_0p05) * (Q.zdr_7 - 0.0007928864)
    if Q.z_7 < 0.06164517 and Q.dr1_7 > 0.1343804:
        z += -133.1839 * (0.06164517 - Q.z_7) * (Q.dr1_7 - 0.1343804)
    if Q.z_7 < 0.06164517 and Q.mean_phi2 < 0.008921136:
        z += 334.32 * (0.06164517 - Q.z_7) * (0.008921136 - Q.mean_phi2)
    if Q.z_7 > 0.04624032 and Q.sum_z_dr2_top2 < 0.002412891:
        z += 15190.29 * (Q.z_7 - 0.04624032) * (0.002412891 - Q.sum_z_dr2_top2)
    if Q.z_7 < 0.06164517 and Q.sum_z_dr2_top2 < 0.01403324:
        z += 291.2974 * (0.06164517 - Q.z_7) * (0.01403324 - Q.sum_z_dr2_top2)
    if Q.sj3_dr_max > 0.169029 and Q.sj2_mass2 < 1.138966:
        z += -0.3953868 * (Q.sj3_dr_max - 0.169029) * (1.138966 - Q.sj2_mass2)
    if Q.tau21_b2 < 0.006726135 and Q.mean_eta > 0.009367547:
        z += -721.6899 * (0.006726135 - Q.tau21_b2) * (Q.mean_eta - 0.009367547)
    if Q.sum_pt_top5 > 531.1875 and Q.phi_0 < -0.04013062:
        z += 0.1778224 * (Q.sum_pt_top5 - 531.1875) * (-0.04013062 - Q.phi_0)
    if Q.e3 < 0.0005116989 and Q.phi_0 < -0.04013062:
        z += -25065.59 * (0.0005116989 - Q.e3) * (-0.04013062 - Q.phi_0)
    if Q.sj3_dr_min > 0.03628191 and Q.sj3_z3 < 0.09635947:
        z += 43.6068 * (Q.sj3_dr_min - 0.03628191) * (0.09635947 - Q.sj3_z3)
    if Q.log_sum_pt > 6.502799 and Q.zdr_6 < 0.008654951:
        z += -910.9299 * (Q.log_sum_pt - 6.502799) * (0.008654951 - Q.zdr_6)
    if Q.sum_z_dr < 0.1019409 and Q.pair_mass_0_6 > 4.037975:
        z += -0.04518695 * (0.1019409 - Q.sum_z_dr) * (Q.pair_mass_0_6 - 4.037975)
    if Q.sum_pt_top5 > 531.1875 and Q.zdr_6 < 0.01392641:
        z += 0.5860842 * (Q.sum_pt_top5 - 531.1875) * (0.01392641 - Q.zdr_6)
    if Q.lam1 < 0.00595415 and Q.ptdr0_6 > 4.218041:
        z += -38.74285 * (0.00595415 - Q.lam1) * (Q.ptdr0_6 - 4.218041)
    if Q.tau1 < 0.07283629 and Q.zdr_6 > 0.005547132:
        z += -1071.289 * (0.07283629 - Q.tau1) * (Q.zdr_6 - 0.005547132)
    if Q.pt_7 > 34.53125 and Q.pt_3 < 77.625:
        z += -0.002121134 * (Q.pt_7 - 34.53125) * (77.625 - Q.pt_3)
    if Q.pt_7 > 53.4375 and Q.zdr_6 > 0.0002697433:
        z += -10.88283 * (Q.pt_7 - 53.4375) * (Q.zdr_6 - 0.0002697433)
    if Q.z_7 < 0.06164517 and Q.abseta_0 < 0.1057739:
        z += 1.89921 * (0.06164517 - Q.z_7) * (0.1057739 - Q.abseta_0)
    if Q.sum_pt_top5 > 531.1875 and Q.ptdr0_6 > 5.648151:
        z += 0.000541214 * (Q.sum_pt_top5 - 531.1875) * (Q.ptdr0_6 - 5.648151)
    if Q.sum_z_dr < 0.0717028 and Q.zdr_6 > 0.01392641:
        z += -10565.04 * (0.0717028 - Q.sum_z_dr) * (Q.zdr_6 - 0.01392641)
    if Q.pt_7 < 25.57812 and Q.zdr_6 > 0.00177424:
        z += 20.2686 * (25.57812 - Q.pt_7) * (Q.zdr_6 - 0.00177424)
    return max(0.0, z)


def neuron_2(Q):
    z = 1.952021
    if Q.sj3_pair_mass_max < 62.55:
        z += -0.02173177 * Q.sj3_pair_mass_max + 1.359322
    if Q.log_sum_pt < 6.46415:
        z += -10.26898 * Q.log_sum_pt + 67.22465
    if 6.46415 <= Q.log_sum_pt < 6.605974:
        z += -5.953928 * Q.log_sum_pt + 39.3315
    if Q.log_sum_pt >= 6.842717:
        z += 26.4204 * Q.log_sum_pt - 180.7873
    if Q.lam1 < 0.001503553:
        z += -488.3438 * Q.lam1 + 1.778091
    if 0.001503553 <= Q.lam1 < 0.00595415:
        z += -234.5395 * Q.lam1 + 1.396483
    if Q.z_7 < 0.03243272:
        z += 91.60277 * Q.z_7 - 2.988087
    if 0.03243272 <= Q.z_7 < 0.03629544:
        z += 0.4393659 * Q.z_7 - 0.03140967
    if 0.03629544 <= Q.z_7 < 0.04939969:
        z += -24.07924 * Q.z_7 + 0.8585038
    if 0.04939969 <= Q.z_7 < 0.07148865:
        z += -17.03552 * Q.z_7 + 0.5105464
    if Q.z_7 >= 0.07148865:
        z += -17.47489 * Q.z_7 + 0.5419561
    if Q.sum_z_dr < 0.007673833:
        z += 699.7631 * Q.sum_z_dr - 5.369865
    if Q.pt_7 < 34.53125:
        z += 0.1191138 * Q.pt_7 - 6.365142
    if 34.53125 <= Q.pt_7 < 43.5:
        z += 0.1649079 * Q.pt_7 - 7.946471
    if 43.5 <= Q.pt_7 < 53.4375:
        z += 0.2249981 * Q.pt_7 - 10.56039
    if Q.pt_7 >= 53.4375:
        z += 0.1058843 * Q.pt_7 - 4.19525
    if Q.LHA >= 0.1329373:
        z += -11.97113 * Q.LHA + 1.591409
    if 752.1 <= Q.sum_pt_top5 < 839.9547:
        z += -0.003461601 * Q.sum_pt_top5 + 2.60347
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.005392263 * Q.sum_pt_top5 + 4.225138
    if Q.planar_flow < 0.4926918:
        z += 2.211929 * Q.planar_flow - 1.089799
    if Q.mass < 36.22941:
        z += 0.05219616 * Q.mass - 1.891036
    if Q.N2 >= 0.1150852:
        z += 2.728526 * Q.N2 - 0.314013
    if Q.max_dr < 0.2507612:
        z += -4.96173 * Q.max_dr + 1.244209
    if Q.sum_z_dr2 < 0.003562611:
        z += -354.0029 * Q.sum_z_dr2 + 1.261175
    if Q.m012 < 3.55957:
        z += -0.2024125 * Q.m012 + 0.7205013
    if Q.m012 >= 40.2:
        z += 0.09193258 * Q.m012 - 3.69569
    if Q.sum_pt < 527.1781:
        z += -0.009364161 * Q.sum_pt + 7.616955
    if 527.1781 <= Q.sum_pt < 813.4156:
        z += -0.00797594 * Q.sum_pt + 6.885115
    if 813.4156 <= Q.sum_pt < 868.5094:
        z += 0.00138822 * Q.sum_pt - 0.7318395
    if Q.sum_pt >= 868.5094:
        z += 0.01787133 * Q.sum_pt - 15.04757
    if Q.centroid_offset < 0.004430536:
        z += 411.5306 * Q.centroid_offset - 1.823301
    if Q.zdr_0 < 0.0211821:
        z += -48.31899 * Q.zdr_0 + 1.023498
    if Q.sum_z_dr2_top2 < 3.849518e-05:
        z += 19918.86 * Q.sum_z_dr2_top2 - 0.7667801
    if Q.mass_over_sum_pt < 0.1079857:
        z += -12.77398 * Q.mass_over_sum_pt + 1.379407
    if Q.e2 < 0.009668065:
        z += -25.51171 * Q.e2 + 0.2466489
    if Q.lam2 < 0.0001947983:
        z += -2077.42 * Q.lam2 + 0.4046779
    if Q.C2 < 0.06729223:
        z += -6.128669 * Q.C2 + 0.4124118
    if Q.absphi_2 < 0.01573181:
        z += -35.17616 * Q.absphi_2 + 0.5533848
    if Q.sj3_pair_mass_max < 62.55 and Q.z_7 < 0.06810151:
        z += -0.8996362 * (62.55 - Q.sj3_pair_mass_max) * (0.06810151 - Q.z_7)
    if Q.sj3_pair_mass_max < 62.55 and Q.centroid_offset > 0.01096064:
        z += -0.7968631 * (62.55 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.01096064)
    if Q.pt_7 > 34.53125 and Q.D3 > 0.4387703:
        z += -0.01176216 * (Q.pt_7 - 34.53125) * (Q.D3 - 0.4387703)
    if Q.lam1 < 0.00595415 and Q.max_dr > 0.08050702:
        z += -3353.662 * (0.00595415 - Q.lam1) * (Q.max_dr - 0.08050702)
    if Q.z_7 > 0.04939969 and Q.sj3_dr_min < 0.0623951:
        z += 147.2104 * (Q.z_7 - 0.04939969) * (0.0623951 - Q.sj3_dr_min)
    if Q.log_sum_pt > 6.842717 and Q.pt_6 > 41.21875:
        z += 0.684987 * (Q.log_sum_pt - 6.842717) * (Q.pt_6 - 41.21875)
    if Q.z_7 > 0.04939969 and Q.D2_b2 < 0.9206502:
        z += -55.63708 * (Q.z_7 - 0.04939969) * (0.9206502 - Q.D2_b2)
    if Q.z_7 < 0.03243272 and Q.mass_top3 > 16.89912:
        z += -5.824428 * (0.03243272 - Q.z_7) * (Q.mass_top3 - 16.89912)
    if Q.sum_z_dr < 0.007673833 and Q.D2 < 3.885568:
        z += -228.8553 * (0.007673833 - Q.sum_z_dr) * (3.885568 - Q.D2)
    if Q.lam1 < 0.00595415 and Q.pt_6 > 19.46875:
        z += 6.698817 * (0.00595415 - Q.lam1) * (Q.pt_6 - 19.46875)
    if Q.sum_z_dr < 0.007673833 and Q.pt_4 < 75.625:
        z += -14.60965 * (0.007673833 - Q.sum_z_dr) * (75.625 - Q.pt_4)
    if Q.log_sum_pt < 6.605974 and Q.z_5 < 0.0531335:
        z += 647.1693 * (6.605974 - Q.log_sum_pt) * (0.0531335 - Q.z_5)
    if Q.log_sum_pt < 6.605974 and Q.max_dr < 0.03623337:
        z += 1022.588 * (6.605974 - Q.log_sum_pt) * (0.03623337 - Q.max_dr)
    if Q.mass < 36.22941 and Q.pt_5 < 73.75:
        z += 0.001239853 * (36.22941 - Q.mass) * (73.75 - Q.pt_5)
    if Q.log_sum_pt < 6.46415 and Q.D2_b2 < 1.129616:
        z += 4.964235 * (6.46415 - Q.log_sum_pt) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt > 6.842717 and Q.phi_0 < -0.0216713:
        z += -187.9237 * (Q.log_sum_pt - 6.842717) * (-0.0216713 - Q.phi_0)
    if Q.z_7 < 0.03243272 and Q.D2_b2 < 0.9206502:
        z += -222.1012 * (0.03243272 - Q.z_7) * (0.9206502 - Q.D2_b2)
    if Q.max_dr < 0.2507612 and Q.phi_0 > 0.002076054:
        z += -65.7493 * (0.2507612 - Q.max_dr) * (Q.phi_0 - 0.002076054)
    if Q.pt_7 < 53.4375 and Q.D2_b2 < 1.129616:
        z += -0.04300888 * (53.4375 - Q.pt_7) * (1.129616 - Q.D2_b2)
    if Q.sum_pt > 527.1781 and Q.absphi_3 < 0.05648804:
        z += -0.04553383 * (Q.sum_pt - 527.1781) * (0.05648804 - Q.absphi_3)
    if Q.log_sum_pt > 6.842717 and Q.eta_0 > 0.02130127:
        z += -96.62688 * (Q.log_sum_pt - 6.842717) * (Q.eta_0 - 0.02130127)
    if Q.sum_pt_top5 > 752.1 and Q.eta_0 > 0.009292603:
        z += 0.417871 * (Q.sum_pt_top5 - 752.1) * (Q.eta_0 - 0.009292603)
    if Q.log_sum_pt > 6.842717 and Q.z_dr_0p05_0p1 < 0.4474937:
        z += 45.09621 * (Q.log_sum_pt - 6.842717) * (0.4474937 - Q.z_dr_0p05_0p1)
    if Q.sum_pt_top5 > 752.1 and Q.D2_b2 < 1.129616:
        z += 0.01368 * (Q.sum_pt_top5 - 752.1) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt > 6.842717 and Q.D2_b2 < 1.345805:
        z += -14.96121 * (Q.log_sum_pt - 6.842717) * (1.345805 - Q.D2_b2)
    if Q.z_7 < 0.03243272 and Q.n_pt_above_5 < 8.0:
        z += -238.7942 * (0.03243272 - Q.z_7) * (8.0 - Q.n_pt_above_5)
    if Q.zdr_0 < 0.0211821 and Q.mean_eta < -0.01284493:
        z += -376.5339 * (0.0211821 - Q.zdr_0) * (-0.01284493 - Q.mean_eta)
    if Q.sum_pt_top5 > 752.1 and Q.abseta_0 < 0.0174408:
        z += -0.7057678 * (Q.sum_pt_top5 - 752.1) * (0.0174408 - Q.abseta_0)
    if Q.log_sum_pt > 6.842717 and Q.abseta_0 < 0.0174408:
        z += 1663.257 * (Q.log_sum_pt - 6.842717) * (0.0174408 - Q.abseta_0)
    if Q.pt_7 < 53.4375 and Q.abseta_0 > 0.02940369:
        z += -0.3730037 * (53.4375 - Q.pt_7) * (Q.abseta_0 - 0.02940369)
    if Q.sum_pt > 527.1781 and Q.lam2 < 0.0001947983:
        z += -28.39403 * (Q.sum_pt - 527.1781) * (0.0001947983 - Q.lam2)
    if Q.sum_pt_top5 > 839.9547 and Q.dr_7 < 0.04881348:
        z += 0.6142219 * (Q.sum_pt_top5 - 839.9547) * (0.04881348 - Q.dr_7)
    if Q.centroid_offset < 0.004430536 and Q.abseta_7 < 0.02490234:
        z += -25337.86 * (0.004430536 - Q.centroid_offset) * (0.02490234 - Q.abseta_7)
    if Q.z_7 < 0.07148865 and Q.abseta_7 < 0.05718994:
        z += -264.1982 * (0.07148865 - Q.z_7) * (0.05718994 - Q.abseta_7)
    if Q.log_sum_pt > 6.842717 and Q.phi_5 < -0.007301331:
        z += 160.3984 * (Q.log_sum_pt - 6.842717) * (-0.007301331 - Q.phi_5)
    if Q.log_sum_pt > 6.842717 and Q.eta_4 > 0.1080994:
        z += -500.4283 * (Q.log_sum_pt - 6.842717) * (Q.eta_4 - 0.1080994)
    if Q.log_sum_pt > 6.842717 and Q.dr02 > 0.2623104:
        z += -30.75349 * (Q.log_sum_pt - 6.842717) * (Q.dr02 - 0.2623104)
    if Q.log_sum_pt > 6.842717 and Q.dr02 > 0.15647:
        z += 265.6606 * (Q.log_sum_pt - 6.842717) * (Q.dr02 - 0.15647)
    if Q.log_sum_pt > 6.842717 and Q.lam2 < 0.0001947983:
        z += 116102.2 * (Q.log_sum_pt - 6.842717) * (0.0001947983 - Q.lam2)
    if Q.z_7 < 0.07148865 and Q.D3 > 0.2213841:
        z += -7.172592 * (0.07148865 - Q.z_7) * (Q.D3 - 0.2213841)
    if Q.z_7 < 0.03243272 and Q.mean_phi < -0.00469376:
        z += 3966.635 * (0.03243272 - Q.z_7) * (-0.00469376 - Q.mean_phi)
    if Q.z_7 > 0.03629544 and Q.C2_b2 < 0.004032342:
        z += 627.2209 * (Q.z_7 - 0.03629544) * (0.004032342 - Q.C2_b2)
    if Q.pt_7 > 43.5 and Q.C2_b2 < 0.004032342:
        z += 21.5273 * (Q.pt_7 - 43.5) * (0.004032342 - Q.C2_b2)
    if Q.z_7 > 0.03629544 and Q.dr02 > 0.175862:
        z += -87.59175 * (Q.z_7 - 0.03629544) * (Q.dr02 - 0.175862)
    if Q.sum_pt_top5 > 839.9547 and Q.absphi_2 < 0.001184464:
        z += -15.94504 * (Q.sum_pt_top5 - 839.9547) * (0.001184464 - Q.absphi_2)
    if Q.sum_pt > 527.1781 and Q.absphi_2 < 0.01922607:
        z += -0.2532598 * (Q.sum_pt - 527.1781) * (0.01922607 - Q.absphi_2)
    if Q.log_sum_pt < 6.605974 and Q.absphi_2 < 0.01573181:
        z += 255.2557 * (6.605974 - Q.log_sum_pt) * (0.01573181 - Q.absphi_2)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.1613346
    if Q.mass_over_sum_pt >= 0.0681391:
        z += 31.41488 * Q.mass_over_sum_pt - 2.140582
    if Q.centroid_offset >= 0.01096064:
        z += 8.871214 * Q.centroid_offset - 0.09723417
    if 0.05356915 <= Q.tau1 < 0.1027642:
        z += -20.85496 * Q.tau1 + 1.117182
    if 0.1027642 <= Q.tau1 < 0.1136369:
        z += -36.33749 * Q.tau1 + 2.708233
    if Q.tau1 >= 0.1136369:
        z += -4.540733 * Q.tau1 - 0.9050517
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += 676.5582 * Q.lam1 - 5.666562
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 557.136 * Q.lam1 - 4.23305
    if Q.lam1 >= 0.01643375:
        z += 439.8287 * Q.lam1 - 2.305252
    if 0.04081947 <= Q.sum_z_dr < 0.07608178:
        z += 26.18731 * Q.sum_z_dr - 1.068952
    if 0.07608178 <= Q.sum_z_dr < 0.08723651:
        z += 26.0752 * Q.sum_z_dr - 1.060423
    if 0.08723651 <= Q.sum_z_dr < 0.1019409:
        z += 106.4521 * Q.sum_z_dr - 8.072221
    if Q.sum_z_dr >= 0.1019409:
        z += -27.75956 * Q.sum_z_dr + 5.609438
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.007520088:
        z += 272.32 * Q.lam1_plus_lam2 - 1.522347
    if 0.007520088 <= Q.lam1_plus_lam2 < 0.01323868:
        z += 922.7242 * Q.lam1_plus_lam2 - 6.413445
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += 692.4631 * Q.lam1_plus_lam2 - 3.365092
    if 0.04447357 <= Q.e2 < 0.06344108:
        z += 5.877206 * Q.e2 - 0.2613803
    if Q.e2 >= 0.06344108:
        z += -53.41464 * Q.e2 + 3.500158
    if Q.sum_z_dr2 < 0.004372139:
        z += -164.4163 * Q.sum_z_dr2 + 0.7188512
    if 0.008678045 <= Q.sum_z_dr2 < 0.01882765:
        z += -1495.658 * Q.sum_z_dr2 + 12.97938
    if Q.sum_z_dr2 >= 0.01882765:
        z += -1522.761 * Q.sum_z_dr2 + 13.48968
    if 0.1492731 <= Q.sj2_dr < 0.1872617:
        z += -1.917718 * Q.sj2_dr + 0.2862637
    if 0.1872617 <= Q.sj2_dr < 0.2687922:
        z += 36.98566 * Q.sj2_dr - 6.998849
    if Q.sj2_dr >= 0.2687922:
        z += 32.72312 * Q.sj2_dr - 5.853113
    if Q.mean_eta < -0.004664942:
        z += -51.02441 * Q.mean_eta - 0.2380259
    if Q.mean_eta >= 0.01772426:
        z += 77.60816 * Q.mean_eta - 1.375547
    if Q.sum_z_dr2_top5 < 0.007164202:
        z += 163.0342 * Q.sum_z_dr2_top5 - 1.125011
    if 0.007164202 <= Q.sum_z_dr2_top5 < 0.008329695:
        z += -36.89316 * Q.sum_z_dr2_top5 + 0.3073088
    if Q.max_dr >= 0.1027585:
        z += 12.32977 * Q.max_dr - 1.266989
    if Q.lam2 >= 0.001130645:
        z += 280.118 * Q.lam2 - 0.316714
    if Q.mass >= 64.61873:
        z += 0.006136776 * Q.mass - 0.3965507
    if 38.43971 <= Q.sd_mass < 62.55:
        z += -0.0002271927 * Q.sd_mass + 0.008733223
    if 62.55 <= Q.sd_mass < 86.4:
        z += 0.03582973 * Q.sd_mass - 2.246627
    if Q.sd_mass >= 86.4:
        z += 0.07238153 * Q.sd_mass - 5.404703
    if Q.sj2_mass1 >= 40.2:
        z += -0.05379647 * Q.sj2_mass1 + 2.162618
    if Q.sj3_pair_mass_min >= 11.051:
        z += -0.02079949 * Q.sj3_pair_mass_min + 0.2298552
    if 0.1986272 <= Q.sj3_dr_max < 0.3012016:
        z += 13.77049 * Q.sj3_dr_max - 2.735193
    if Q.sj3_dr_max >= 0.3012016:
        z += 17.1294 * Q.sj3_dr_max - 3.746903
    if Q.n_dr_0_0p05 < 1.0:
        z += -0.8893508 * Q.n_dr_0_0p05 + 0.8893508
    if Q.z_dr_0_0p05 < 0.9008535:
        z += 0.9295364 * Q.z_dr_0_0p05 - 0.8373761
    if Q.mean_eta2 >= 0.01423545:
        z += -103.6805 * Q.mean_eta2 + 1.475938
    if Q.mass_over_sum_pt_sq < 0.00817466:
        z += 287.214 * Q.mass_over_sum_pt_sq - 3.809202
    if 0.00817466 <= Q.mass_over_sum_pt_sq < 0.0116609:
        z += 419.1689 * Q.mass_over_sum_pt_sq - 4.887888
    if Q.LHA >= 0.3467135:
        z += 18.20714 * Q.LHA - 6.312661
    if Q.planar_flow < 0.03320012:
        z += 5.403291 * Q.planar_flow - 0.1793899
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.2192754 * (Q.mass_over_sum_pt - 0.0681391) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.mass_over_sum_pt > 0.0681391 and Q.pt_6 > 31.90625:
        z += 0.2726133 * (Q.mass_over_sum_pt - 0.0681391) * (Q.pt_6 - 31.90625)
    if Q.sum_z_dr > 0.04081947 and Q.log_sum_pt > 6.080494:
        z += 23.92645 * (Q.sum_z_dr - 0.04081947) * (Q.log_sum_pt - 6.080494)
    if Q.e2 > 0.06344108 and Q.sj2_mass1 > 16.86126:
        z += 0.4314643 * (Q.e2 - 0.06344108) * (Q.sj2_mass1 - 16.86126)
    if Q.sj2_dr > 0.1872617 and Q.sj2_mass1 > 2.250113:
        z += -0.2036724 * (Q.sj2_dr - 0.1872617) * (Q.sj2_mass1 - 2.250113)
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj2_dr < 0.2179769:
        z += -1407.89 * (Q.mass_over_sum_pt - 0.0681391) * (0.2179769 - Q.sj2_dr)
    if Q.lam1_plus_lam2 > 0.007520088 and Q.sj3_pairmin_over_m > 0.07708997:
        z += -97.53214 * (Q.lam1_plus_lam2 - 0.007520088) * (Q.sj3_pairmin_over_m - 0.07708997)
    if Q.centroid_offset > 0.01096064 and Q.n_pt_above_50 < 8.0:
        z += 6.447306 * (Q.centroid_offset - 0.01096064) * (8.0 - Q.n_pt_above_50)
    if Q.centroid_offset > 0.01096064 and Q.phi_7 > -0.05731659:
        z += 221.9654 * (Q.centroid_offset - 0.01096064) * (Q.phi_7 - -0.05731659)
    if Q.sj2_dr > 0.1872617 and Q.pt_6 < 27.57812:
        z += -1.011261 * (Q.sj2_dr - 0.1872617) * (27.57812 - Q.pt_6)
    if Q.centroid_offset > 0.01096064 and Q.abseta_0 < 0.07861328:
        z += 993.9412 * (Q.centroid_offset - 0.01096064) * (0.07861328 - Q.abseta_0)
    if Q.mean_eta < -0.004664942 and Q.phi_5 > 0.07385864:
        z += 823.7863 * (-0.004664942 - Q.mean_eta) * (Q.phi_5 - 0.07385864)
    if Q.centroid_offset > 0.01096064 and Q.pair_mass_0_5 > 16.26599:
        z += 6.582346 * (Q.centroid_offset - 0.01096064) * (Q.pair_mass_0_5 - 16.26599)
    if Q.sj2_dr > 0.1872617 and Q.n_dr_0p2_0p4 < 2.0:
        z += 7.262035 * (Q.sj2_dr - 0.1872617) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.tau1 > 0.05356915 and Q.mass_top3 < 50.3522:
        z += -0.2586961 * (Q.tau1 - 0.05356915) * (50.3522 - Q.mass_top3)
    if Q.lam2 > 0.001130645 and Q.pt_6 < 56.53125:
        z += 3.042815 * (Q.lam2 - 0.001130645) * (56.53125 - Q.pt_6)
    if Q.sd_mass > 62.55 and Q.D2_b2 < 0.9206502:
        z += 0.04514852 * (Q.sd_mass - 62.55) * (0.9206502 - Q.D2_b2)
    if Q.e2 > 0.06344108 and Q.z_top5 > 0.7773372:
        z += -832.2701 * (Q.e2 - 0.06344108) * (Q.z_top5 - 0.7773372)
    if Q.mean_eta < -0.004664942 and Q.z_6 > 0.06727211:
        z += 279.4455 * (-0.004664942 - Q.mean_eta) * (Q.z_6 - 0.06727211)
    if Q.sj2_dr > 0.1872617 and Q.dr_3 < 0.05268713:
        z += -757.2666 * (Q.sj2_dr - 0.1872617) * (0.05268713 - Q.dr_3)
    if Q.mass > 64.61873 and Q.phi_6 > 0.1192017:
        z += -0.3498117 * (Q.mass - 64.61873) * (Q.phi_6 - 0.1192017)
    if Q.tau1 > 0.05356915 and Q.pt_5 > 59.125:
        z += 0.9625787 * (Q.tau1 - 0.05356915) * (Q.pt_5 - 59.125)
    if Q.e2 > 0.06344108 and Q.dr0_3 < 0.1332952:
        z += -924.0516 * (Q.e2 - 0.06344108) * (0.1332952 - Q.dr0_3)
    if Q.max_dr > 0.1027585 and Q.phi_4 > 0.1096802:
        z += 26.74683 * (Q.max_dr - 0.1027585) * (Q.phi_4 - 0.1096802)
    if Q.mass > 64.61873 and Q.eta_7 > -0.0803833:
        z += 0.1446984 * (Q.mass - 64.61873) * (Q.eta_7 - -0.0803833)
    if Q.mean_eta > 0.01772426 and Q.mean_phi < 0.02612796:
        z += 440.9852 * (Q.mean_eta - 0.01772426) * (0.02612796 - Q.mean_phi)
    if Q.sum_z_dr > 0.07608178 and Q.eta_0 > 0.07952881:
        z += 56.31382 * (Q.sum_z_dr - 0.07608178) * (Q.eta_0 - 0.07952881)
    if Q.lam2 > 0.001130645 and Q.z_6 > 0.05441452:
        z += 4311.299 * (Q.lam2 - 0.001130645) * (Q.z_6 - 0.05441452)
    if Q.lam1 > 0.008375572 and Q.pt_6 < 24.42188:
        z += -40.4372 * (Q.lam1 - 0.008375572) * (24.42188 - Q.pt_6)
    if Q.lam1 > 0.01643375 and Q.n_for_90pct < 6.0:
        z += 1468.383 * (Q.lam1 - 0.01643375) * (6.0 - Q.n_for_90pct)
    if Q.max_dr > 0.1027585 and Q.z_6 < 0.06081235:
        z += 113.3088 * (Q.max_dr - 0.1027585) * (0.06081235 - Q.z_6)
    if Q.max_dr > 0.1027585 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += -13.75316 * (Q.max_dr - 0.1027585) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.e2 > 0.06344108 and Q.zdr_3 < 0.01257746:
        z += 3258.038 * (Q.e2 - 0.06344108) * (0.01257746 - Q.zdr_3)
    if Q.tau1 > 0.05356915 and Q.mratio_min_012 > 0.3373366:
        z += -32.1918 * (Q.tau1 - 0.05356915) * (Q.mratio_min_012 - 0.3373366)
    if Q.max_dr > 0.1027585 and Q.eta_1 < -0.0892334:
        z += -132.3364 * (Q.max_dr - 0.1027585) * (-0.0892334 - Q.eta_1)
    if Q.sd_mass > 62.55 and Q.zdr_1 < 0.01849752:
        z += -0.4298744 * (Q.sd_mass - 62.55) * (0.01849752 - Q.zdr_1)
    if Q.sd_mass > 62.55 and Q.pair_mass_0_4 < 1.474928:
        z += -0.07582087 * (Q.sd_mass - 62.55) * (1.474928 - Q.pair_mass_0_4)
    if Q.sum_z_dr2 > 0.01882765 and Q.pair_mass_0_5 > 16.26599:
        z += -7.176459 * (Q.sum_z_dr2 - 0.01882765) * (Q.pair_mass_0_5 - 16.26599)
    if Q.max_dr > 0.1027585 and Q.abseta_0 > 0.1057739:
        z += -141.8066 * (Q.max_dr - 0.1027585) * (Q.abseta_0 - 0.1057739)
    if Q.sj3_dr_max > 0.1986272 and Q.sj3_mass2 < 0.5144873:
        z += 9.737506 * (Q.sj3_dr_max - 0.1986272) * (0.5144873 - Q.sj3_mass2)
    if Q.max_dr > 0.1027585 and Q.eta_2 > 0.09655762:
        z += -132.7171 * (Q.max_dr - 0.1027585) * (Q.eta_2 - 0.09655762)
    if Q.sj2_dr > 0.2687922 and Q.z_dr_0p05_0p1 < 0.9641201:
        z += 28.05791 * (Q.sj2_dr - 0.2687922) * (0.9641201 - Q.z_dr_0p05_0p1)
    if Q.sj2_dr > 0.1872617 and Q.z_dr_0p05_0p1 < 0.9641201:
        z += -29.71771 * (Q.sj2_dr - 0.1872617) * (0.9641201 - Q.z_dr_0p05_0p1)
    if Q.n_dr_0_0p05 < 1.0 and Q.n_dr_0p05_0p1 < 4.0:
        z += 0.05712481 * (1.0 - Q.n_dr_0_0p05) * (4.0 - Q.n_dr_0p05_0p1)
    if Q.sum_z_dr2 > 0.01882765 and Q.z_dr_0p05_0p1 > 0.7509095:
        z += -33775.05 * (Q.sum_z_dr2 - 0.01882765) * (Q.z_dr_0p05_0p1 - 0.7509095)
    if Q.sum_z_dr > 0.08723651 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += 1145.439 * (Q.sum_z_dr - 0.08723651) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.centroid_offset > 0.01096064 and Q.abseta_2 > 0.03250122:
        z += 150.1093 * (Q.centroid_offset - 0.01096064) * (Q.abseta_2 - 0.03250122)
    if Q.max_dr > 0.1027585 and Q.eta_2 > -0.04544525:
        z += 55.92079 * (Q.max_dr - 0.1027585) * (Q.eta_2 - -0.04544525)
    if Q.max_dr > 0.1027585 and Q.phi_5 > 0.07385864:
        z += 33.68216 * (Q.max_dr - 0.1027585) * (Q.phi_5 - 0.07385864)
    if Q.centroid_offset > 0.01096064 and Q.abseta_7 < 0.1626038:
        z += 203.8464 * (Q.centroid_offset - 0.01096064) * (0.1626038 - Q.abseta_7)
    if Q.centroid_offset > 0.01096064 and Q.absphi_2 > 0.12854:
        z += -40.78782 * (Q.centroid_offset - 0.01096064) * (Q.absphi_2 - 0.12854)
    if Q.sj3_dr_max > 0.1986272 and Q.absphi_6 < 0.03359985:
        z += -268.4591 * (Q.sj3_dr_max - 0.1986272) * (0.03359985 - Q.absphi_6)
    if Q.planar_flow < 0.03320012 and Q.ptdr0_3 > 12.34207:
        z += -5.381105 * (0.03320012 - Q.planar_flow) * (Q.ptdr0_3 - 12.34207)
    return max(0.0, z)


def neuron_4(Q):
    z = -0.8759114
    if Q.N2 < 0.2233283:
        z += -41.27917 * Q.N2 + 9.218805
    if Q.lam2 < 0.000537286:
        z += 2323.494 * Q.lam2 - 1.016797
    if 0.000537286 <= Q.lam2 < 0.001130645:
        z += 547.7714 * Q.lam2 - 0.06272663
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -244.3682 * Q.lam2 + 0.8329017
    if Q.mass_over_sum_pt < 0.07269073:
        z += 14.22249 * Q.mass_over_sum_pt - 1.033843
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -109.2399 * Q.mass_over_sum_pt + 9.876795
    if Q.lam1_plus_lam2 < 0.001653836:
        z += -1840.834 * Q.lam1_plus_lam2 + 3.044438
    if Q.lam1_plus_lam2 >= 0.003562611:
        z += 361.429 * Q.lam1_plus_lam2 - 1.287631
    if Q.e2 < 0.04447357:
        z += 89.37199 * Q.e2 - 4.494038
    if 0.04447357 <= Q.e2 < 0.05028464:
        z += 149.1826 * Q.e2 - 7.154032
    if Q.e2 >= 0.05028464:
        z += 59.81065 * Q.e2 - 2.659993
    if Q.sum_pt < 739.5:
        z += 0.0003858944 * Q.sum_pt - 0.2853689
    if Q.max_dr < 0.1117619:
        z += -2.001991 * Q.max_dr - 0.4053226
    if 0.1117619 <= Q.max_dr < 0.1598486:
        z += 9.597786 * Q.max_dr - 1.701735
    if 0.1598486 <= Q.max_dr < 0.177305:
        z += 21.20907 * Q.max_dr - 3.557784
    if Q.max_dr >= 0.177305:
        z += 11.61129 * Q.max_dr - 1.856049
    if Q.C2 < 0.06729223:
        z += -53.6324 * Q.C2 + 3.609044
    if Q.sum_z_dr >= 0.08723651:
        z += -2.074416 * Q.sum_z_dr + 0.1809648
    if 44.82259 <= Q.sd_mass < 74.57663:
        z += 0.05811458 * Q.sd_mass - 2.604846
    if Q.sd_mass >= 74.57663:
        z += 0.02659157 * Q.sd_mass - 0.2539665
    if Q.e3 < 1.340118e-05:
        z += 8042.438 * Q.e3 + 0.7481419
    if 1.340118e-05 <= Q.e3 < 0.0001869378:
        z += -4932.215 * Q.e3 + 0.9220176
    if Q.sj3_dr_max < 0.213399:
        z += 5.489557 * Q.sj3_dr_max - 0.2530985
    if 0.213399 <= Q.sj3_dr_max < 0.233678:
        z += 20.06382 * Q.sj3_dr_max - 3.363231
    if 0.233678 <= Q.sj3_dr_max < 0.3456459:
        z += -11.83591 * Q.sj3_dr_max + 4.091033
    if Q.sum_z_dr2 < 0.002635418:
        z += 26.62943 * Q.sum_z_dr2 - 3.139639
    if 0.002635418 <= Q.sum_z_dr2 < 0.008678045:
        z += 507.9678 * Q.sum_z_dr2 - 4.408167
    if Q.sum_z_dr2_top5 < 0.001501708:
        z += -81.56152 * Q.sum_z_dr2_top5 + 0.5843232
    if 0.001501708 <= Q.sum_z_dr2_top5 < 0.007164202:
        z += -108.0456 * Q.sum_z_dr2_top5 + 0.6240945
    if Q.sum_z_dr2_top5 >= 0.007164202:
        z += -26.48407 * Q.sum_z_dr2_top5 + 0.03977135
    if Q.centroid_offset < 0.04990367:
        z += 42.54055 * Q.centroid_offset - 2.12293
    if Q.sum_z_dr2_top2 < 0.007639643:
        z += -216.8202 * Q.sum_z_dr2_top2 + 1.656429
    if Q.mass_top5 < 14.54404:
        z += -0.006341416 * Q.mass_top5 + 0.2394807
    if 14.54404 <= Q.mass_top5 < 37.76455:
        z += 0.02724272 * Q.mass_top5 - 0.2489683
    if Q.mass_top5 >= 37.76455:
        z += 0.03358414 * Q.mass_top5 - 0.488449
    if 45.7571 <= Q.mass < 76.6557:
        z += -0.01299122 * Q.mass + 0.5944404
    if Q.mass >= 76.6557:
        z += -0.1310699 * Q.mass + 9.645847
    if Q.sum_zz_dr2 < 0.01165737:
        z += -3050.219 * Q.sum_zz_dr2 + 35.55754
    if Q.tau1 < 0.07283629:
        z += -45.66261 * Q.tau1 + 3.325895
    if Q.mass_over_sum_pt_sq < 0.0116609:
        z += 2680.432 * Q.mass_over_sum_pt_sq - 31.25627
    if Q.LHA < 0.2809341:
        z += -2.377081 * Q.LHA + 0.824166
    if 0.2809341 <= Q.LHA < 0.3467135:
        z += -7.931609 * Q.LHA + 2.384622
    if Q.LHA >= 0.3467135:
        z += -5.554529 * Q.LHA + 1.560457
    if Q.lam1 < 0.001503553:
        z += 1677.016 * Q.lam1 - 2.521484
    if Q.mean_eta >= 0.02644207:
        z += -20.47267 * Q.mean_eta + 0.5413398
    if Q.M2 < 0.03874536:
        z += 9.8577 * Q.M2 - 0.3819401
    if Q.sj3_pair_mass_max >= 80.4:
        z += -0.02516446 * Q.sj3_pair_mass_max + 2.023223
    if Q.N2 < 0.2233283 and Q.mass < 62.55:
        z += -0.167039 * (0.2233283 - Q.N2) * (62.55 - Q.mass)
    if Q.N2 < 0.2233283 and Q.sum_zz_dr2 > 0.01165737:
        z += -1081.273 * (0.2233283 - Q.N2) * (Q.sum_zz_dr2 - 0.01165737)
    if Q.N2 < 0.2233283 and Q.pt_7 < 53.4375:
        z += -0.2445855 * (0.2233283 - Q.N2) * (53.4375 - Q.pt_7)
    if Q.N2 < 0.2233283 and Q.mean_phi < -0.009352575:
        z += 215.3658 * (0.2233283 - Q.N2) * (-0.009352575 - Q.mean_phi)
    if Q.N2 < 0.2233283 and Q.eccentricity > 0.7117266:
        z += -91.70866 * (0.2233283 - Q.N2) * (Q.eccentricity - 0.7117266)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1218872:
        z += -30.90755 * (0.2233283 - Q.N2) * (0.1218872 - Q.abseta_7)
    if Q.e3 < 0.0001869378 and Q.mean_eta < -0.01284493:
        z += 312338.6 * (0.0001869378 - Q.e3) * (-0.01284493 - Q.mean_eta)
    if Q.N2 < 0.2233283 and Q.zdr_0 < 0.004918231:
        z += 6766.204 * (0.2233283 - Q.N2) * (0.004918231 - Q.zdr_0)
    if Q.sd_mass > 44.82259 and Q.centroid_offset > 0.001308549:
        z += -1.41358 * (Q.sd_mass - 44.82259) * (Q.centroid_offset - 0.001308549)
    if Q.sd_mass > 44.82259 and Q.sd_zg < 0.275762:
        z += -0.05965311 * (Q.sd_mass - 44.82259) * (0.275762 - Q.sd_zg)
    if Q.lam1_plus_lam2 > 0.003562611 and Q.sj3_z3 < 0.1057566:
        z += -506.4212 * (Q.lam1_plus_lam2 - 0.003562611) * (0.1057566 - Q.sj3_z3)
    if Q.N2 < 0.2233283 and Q.pt_2 > 69.875:
        z += -0.07592227 * (0.2233283 - Q.N2) * (Q.pt_2 - 69.875)
    if Q.sj3_dr_max < 0.213399 and Q.mean_phi2 > 0.008921136:
        z += -5530.794 * (0.213399 - Q.sj3_dr_max) * (Q.mean_phi2 - 0.008921136)
    if Q.lam2 < 0.000537286 and Q.dr01 < 0.1410336:
        z += -14064.27 * (0.000537286 - Q.lam2) * (0.1410336 - Q.dr01)
    if Q.sj3_dr_max < 0.3456459 and Q.pair_mass_0_6 > 15.55293:
        z += 1.419954 * (0.3456459 - Q.sj3_dr_max) * (Q.pair_mass_0_6 - 15.55293)
    if Q.N2 < 0.2233283 and Q.phi_6 > 0.1192017:
        z += 11.89297 * (0.2233283 - Q.N2) * (Q.phi_6 - 0.1192017)
    if Q.max_dr > 0.1598486 and Q.C2_b2 < 0.0006435798:
        z += -25987.98 * (Q.max_dr - 0.1598486) * (0.0006435798 - Q.C2_b2)
    if Q.sum_z_dr2_top2 < 0.007639643 and Q.C2_b2 > 0.0006435798:
        z += 647.6217 * (0.007639643 - Q.sum_z_dr2_top2) * (Q.C2_b2 - 0.0006435798)
    if Q.sj3_dr_max < 0.3456459 and Q.eta_5 < -0.1135254:
        z += 123.2083 * (0.3456459 - Q.sj3_dr_max) * (-0.1135254 - Q.eta_5)
    if Q.max_dr < 0.177305 and Q.ptdr0_4 > 8.939062:
        z += 1.888609 * (0.177305 - Q.max_dr) * (Q.ptdr0_4 - 8.939062)
    if Q.lam2 < 0.000537286 and Q.phi_0 > 0.004838562:
        z += -5815.557 * (0.000537286 - Q.lam2) * (Q.phi_0 - 0.004838562)
    if Q.lam2 < 0.000537286 and Q.mean_phi < -0.01275329:
        z += -110290.5 * (0.000537286 - Q.lam2) * (-0.01275329 - Q.mean_phi)
    if Q.e3 < 0.0001869378 and Q.mean_phi < -0.009352575:
        z += 350253.4 * (0.0001869378 - Q.e3) * (-0.009352575 - Q.mean_phi)
    if Q.sj3_dr_max < 0.3456459 and Q.absphi_0 > 0.1056549:
        z += 490.894 * (0.3456459 - Q.sj3_dr_max) * (Q.absphi_0 - 0.1056549)
    if Q.sum_zz_dr2 < 0.01165737 and Q.ptdr0_7 > 2.691363:
        z += -5.011261 * (0.01165737 - Q.sum_zz_dr2) * (Q.ptdr0_7 - 2.691363)
    if Q.sj3_dr_max < 0.3456459 and Q.dr1_5 > 0.1518778:
        z += 127.6472 * (0.3456459 - Q.sj3_dr_max) * (Q.dr1_5 - 0.1518778)
    if Q.N2 < 0.2233283 and Q.pt_1 < 169.875:
        z += 0.03831602 * (0.2233283 - Q.N2) * (169.875 - Q.pt_1)
    if Q.sj3_dr_max < 0.233678 and Q.dr0_3 > 0.1815989:
        z += 2039.538 * (0.233678 - Q.sj3_dr_max) * (Q.dr0_3 - 0.1815989)
    if Q.N2 < 0.2233283 and Q.pt_6 < 24.42188:
        z += 0.5801295 * (0.2233283 - Q.N2) * (24.42188 - Q.pt_6)
    if Q.mass > 76.6557 and Q.sj2_zsoft > 0.4448372:
        z += -0.5125788 * (Q.mass - 76.6557) * (Q.sj2_zsoft - 0.4448372)
    if Q.sd_mass > 44.82259 and Q.zdr_5 < 0.005440034:
        z += -0.347271 * (Q.sd_mass - 44.82259) * (0.005440034 - Q.zdr_5)
    if Q.lam2 < 0.000537286 and Q.dr1_6 < 0.07796252:
        z += -5870.959 * (0.000537286 - Q.lam2) * (0.07796252 - Q.dr1_6)
    if Q.mass_top5 > 14.54404 and Q.pt_6 > 42.78125:
        z += -1.32769e-05 * (Q.mass_top5 - 14.54404) * (Q.pt_6 - 42.78125)
    if Q.mass > 76.6557 and Q.M3 < 0.107953:
        z += 0.8960725 * (Q.mass - 76.6557) * (0.107953 - Q.M3)
    if Q.sd_mass > 44.82259 and Q.sj3_pair_mass_min < 27.42324:
        z += 0.001662624 * (Q.sd_mass - 44.82259) * (27.42324 - Q.sj3_pair_mass_min)
    if Q.sj3_dr_max < 0.3456459 and Q.eta_7 < -0.1218262:
        z += 251.9725 * (0.3456459 - Q.sj3_dr_max) * (-0.1218262 - Q.eta_7)
    if Q.lam2 < 0.000537286 and Q.mean_eta > 0.01271871:
        z += 90910.01 * (0.000537286 - Q.lam2) * (Q.mean_eta - 0.01271871)
    if Q.mean_eta > 0.02644207 and Q.orientation_deg > 45.28772:
        z += 2.117374 * (Q.mean_eta - 0.02644207) * (Q.orientation_deg - 45.28772)
    if Q.mean_eta > 0.02644207 and Q.sj3_pairmax_over_m > 0.7715709:
        z += -305.1103 * (Q.mean_eta - 0.02644207) * (Q.sj3_pairmax_over_m - 0.7715709)
    if Q.tau1 < 0.07283629 and Q.phi_1 > 0.01467133:
        z += 846.4875 * (0.07283629 - Q.tau1) * (Q.phi_1 - 0.01467133)
    if Q.sj3_dr_max < 0.233678 and Q.dr_4 > 0.1289397:
        z += 609.3014 * (0.233678 - Q.sj3_dr_max) * (Q.dr_4 - 0.1289397)
    if Q.sj3_dr_max < 0.233678 and Q.ptdr0_5 > 7.737156:
        z += 1.376774 * (0.233678 - Q.sj3_dr_max) * (Q.ptdr0_5 - 7.737156)
    if Q.lam2 < 0.000537286 and Q.absphi_7 < 0.1647949:
        z += -12008.7 * (0.000537286 - Q.lam2) * (0.1647949 - Q.absphi_7)
    if Q.sj3_dr_max < 0.3456459 and Q.pair_mass_0_6 > 24.79428:
        z += 0.3099763 * (0.3456459 - Q.sj3_dr_max) * (Q.pair_mass_0_6 - 24.79428)
    if Q.mass_top5 < 37.76455 and Q.abseta_3 > 0.1364807:
        z += 0.7940834 * (37.76455 - Q.mass_top5) * (Q.abseta_3 - 0.1364807)
    if Q.sum_pt < 739.5 and Q.absphi_7 < 0.05749512:
        z += 0.0736261 * (739.5 - Q.sum_pt) * (0.05749512 - Q.absphi_7)
    if Q.sj3_dr_max < 0.233678 and Q.dr02 > 0.2009694:
        z += 5380.775 * (0.233678 - Q.sj3_dr_max) * (Q.dr02 - 0.2009694)
    if Q.sum_z_dr2_top2 < 0.007639643 and Q.abseta_1 > 0.01179237:
        z += 2724.948 * (0.007639643 - Q.sum_z_dr2_top2) * (Q.abseta_1 - 0.01179237)
    if Q.sum_pt < 739.5 and Q.n_for_50pct > 1.0:
        z += 0.001521748 * (739.5 - Q.sum_pt) * (Q.n_for_50pct - 1.0)
    if Q.sum_z_dr2 < 0.008678045 and Q.pair_mass_0_6 > 24.79428:
        z += 85.22415 * (0.008678045 - Q.sum_z_dr2) * (Q.pair_mass_0_6 - 24.79428)
    if Q.sj3_dr_max < 0.3456459 and Q.dr_5 > 0.1351015:
        z += 184.9002 * (0.3456459 - Q.sj3_dr_max) * (Q.dr_5 - 0.1351015)
    if Q.max_dr > 0.1598486 and Q.D2_b2 < 0.1236856:
        z += -46.84857 * (Q.max_dr - 0.1598486) * (0.1236856 - Q.D2_b2)
    if Q.sj3_dr_max < 0.3456459 and Q.dr1_5 > 0.2928518:
        z += 636.5071 * (0.3456459 - Q.sj3_dr_max) * (Q.dr1_5 - 0.2928518)
    if Q.N2 < 0.2233283 and Q.eta_1 < -0.04302979:
        z += -58.19206 * (0.2233283 - Q.N2) * (-0.04302979 - Q.eta_1)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.9401255
    if Q.LHA < 0.2160559:
        z += -14.8448 * Q.LHA + 3.207307
    if Q.z_7 < 0.02807091:
        z += -287.838 * Q.z_7 + 10.25965
    if 0.02807091 <= Q.z_7 < 0.04939969:
        z += -75.59219 * Q.z_7 + 4.301712
    if 0.04939969 <= Q.z_7 < 0.07148865:
        z += -25.69071 * Q.z_7 + 1.836594
    if 6.701242 <= Q.log_sum_pt < 6.842717:
        z += -9.590785 * Q.log_sum_pt + 64.27017
    if 6.842717 <= Q.log_sum_pt < 6.896095:
        z += -27.70261 * Q.log_sum_pt + 188.2043
    if Q.log_sum_pt >= 6.896095:
        z += -47.78957 * Q.log_sum_pt + 326.7258
    if Q.sum_zz_dr2 < 0.002074109:
        z += -117.182 * Q.sum_zz_dr2 + 0.5518462
    if 0.002074109 <= Q.sum_zz_dr2 < 0.005284669:
        z += -96.18198 * Q.sum_zz_dr2 + 0.5082899
    if Q.z_6 < 0.02886576:
        z += -224.7378 * Q.z_6 + 6.760785
    if 0.02886576 <= Q.z_6 < 0.08051087:
        z += -5.296899 * Q.z_6 + 0.4264579
    if Q.sum_z_dr2 < 0.001653836:
        z += -981.3835 * Q.sum_z_dr2 + 1.623048
    if Q.sum_z_dr < 0.007673833:
        z += -561.2161 * Q.sum_z_dr + 4.306678
    if Q.pt_7 < 37.15625:
        z += 0.007831152 * Q.pt_7 - 0.2909762
    if Q.zdr_0 < 0.0211821:
        z += 82.0821 * Q.zdr_0 - 1.738671
    if Q.pair_mass_0_4 >= 23.26771:
        z += -0.1274191 * Q.pair_mass_0_4 + 2.964751
    if 430.75 <= Q.sum_pt_top5 < 687.4375:
        z += 0.002244408 * Q.sum_pt_top5 - 0.966779
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.007516707 * Q.sum_pt_top5 - 4.591155
    if Q.pt_5 < 24.57812:
        z += -0.5175981 * Q.pt_5 + 12.72159
    if Q.lam1_plus_lam2 < 0.003562611:
        z += -466.184 * Q.lam1_plus_lam2 + 1.660832
    if Q.centroid_offset < 0.006789738:
        z += -219.5734 * Q.centroid_offset + 1.490846
    if Q.pt_3 < 45.05938:
        z += -0.09469905 * Q.pt_3 + 4.26708
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 294.5935 * Q.sum_z_dr2_top3 - 0.6338379
    if Q.z_7 < 0.04939969 and Q.mass_top5 < 62.55:
        z += 0.957989 * (0.04939969 - Q.z_7) * (62.55 - Q.mass_top5)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -92.99337 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.sum_zz_dr2 < 0.005284669 and Q.centroid_offset < 0.01437952:
        z += 36402.6 * (0.005284669 - Q.sum_zz_dr2) * (0.01437952 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.centroid_offset < 0.03117077:
        z += 541.0654 * (0.07148865 - Q.z_7) * (0.03117077 - Q.centroid_offset)
    if Q.z_7 < 0.04939969 and Q.sum_pt_top2 < 501.625:
        z += -0.927202 * (0.04939969 - Q.z_7) * (501.625 - Q.sum_pt_top2)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.05744392:
        z += 1311.085 * (Q.log_sum_pt - 6.896095) * (0.05744392 - Q.D2_b2)
    if Q.z_7 < 0.07148865 and Q.mass_top3 < 40.2:
        z += 0.5293897 * (0.07148865 - Q.z_7) * (40.2 - Q.mass_top3)
    if Q.z_7 < 0.07148865 and Q.C3 < 0.03532852:
        z += 752.6493 * (0.07148865 - Q.z_7) * (0.03532852 - Q.C3)
    if Q.sum_z_dr2 < 0.001653836 and Q.centroid_offset < 0.02355416:
        z += 117580.3 * (0.001653836 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.sum_zz_dr2 < 0.005284669 and Q.n_pt_above_50 > 6.0:
        z += -108.9074 * (0.005284669 - Q.sum_zz_dr2) * (Q.n_pt_above_50 - 6.0)
    if Q.LHA < 0.2160559 and Q.lam1 < 0.001503553:
        z += -5032.963 * (0.2160559 - Q.LHA) * (0.001503553 - Q.lam1)
    if Q.z_7 < 0.07148865 and Q.C2_b2 > 0.002354783:
        z += -713.5853 * (0.07148865 - Q.z_7) * (Q.C2_b2 - 0.002354783)
    if Q.log_sum_pt > 6.701242 and Q.sj3_dr_max > 0.169029:
        z += 27.84506 * (Q.log_sum_pt - 6.701242) * (Q.sj3_dr_max - 0.169029)
    if Q.log_sum_pt > 6.701242 and Q.dr_2 < 0.01341502:
        z += 1355.016 * (Q.log_sum_pt - 6.701242) * (0.01341502 - Q.dr_2)
    if Q.log_sum_pt > 6.896095 and Q.dr_2 < 0.02270492:
        z += -1962.009 * (Q.log_sum_pt - 6.896095) * (0.02270492 - Q.dr_2)
    if Q.LHA < 0.2160559 and Q.n_dr_0p2_0p4 > 0.0:
        z += 9.165855 * (0.2160559 - Q.LHA) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.sum_pt_top5 > 430.75 and Q.centroid_offset > 0.009480685:
        z += 0.09116119 * (Q.sum_pt_top5 - 430.75) * (Q.centroid_offset - 0.009480685)
    if Q.z_7 < 0.04939969 and Q.dr_2 < 0.009661512:
        z += 7525.01 * (0.04939969 - Q.z_7) * (0.009661512 - Q.dr_2)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi > 0.002834884:
        z += -2207.132 * (Q.log_sum_pt - 6.701242) * (Q.mean_phi - 0.002834884)
    if Q.sum_pt_top5 > 430.75 and Q.pt1_dr01 < 28.39396:
        z += 9.468281e-05 * (Q.sum_pt_top5 - 430.75) * (28.39396 - Q.pt1_dr01)
    if Q.sum_pt_top5 > 430.75 and Q.sj2_dr > 0.1682655:
        z += 0.007254827 * (Q.sum_pt_top5 - 430.75) * (Q.sj2_dr - 0.1682655)
    if Q.z_6 < 0.02886576 and Q.mean_phi2 > 0.001125075:
        z += -102485.0 * (0.02886576 - Q.z_6) * (Q.mean_phi2 - 0.001125075)
    if Q.pt_7 < 37.15625 and Q.pt_5 > 24.57812:
        z += 0.0004936147 * (37.15625 - Q.pt_7) * (Q.pt_5 - 24.57812)
    if Q.sum_pt_top5 > 430.75 and Q.pt_6 < 46.125:
        z += 0.00021535 * (Q.sum_pt_top5 - 430.75) * (46.125 - Q.pt_6)
    if Q.LHA < 0.2160559 and Q.z_3 < 0.07235619:
        z += 607.6726 * (0.2160559 - Q.LHA) * (0.07235619 - Q.z_3)
    if Q.pair_mass_0_4 > 23.26771 and Q.eccentricity > 0.9704496:
        z += 4.00669 * (Q.pair_mass_0_4 - 23.26771) * (Q.eccentricity - 0.9704496)
    if Q.zdr_0 < 0.0211821 and Q.phi_0 > -0.0297699:
        z += -158.0426 * (0.0211821 - Q.zdr_0) * (Q.phi_0 - -0.0297699)
    if Q.z_7 < 0.07148865 and Q.mean_phi2 < 0.002127561:
        z += 11074.5 * (0.07148865 - Q.z_7) * (0.002127561 - Q.mean_phi2)
    if Q.pt_5 < 24.57812 and Q.mean_eta2 > 1.797789e-05:
        z += -173.3034 * (24.57812 - Q.pt_5) * (Q.mean_eta2 - 1.797789e-05)
    if Q.pt_5 < 24.57812 and Q.n_pt_above_5 < 8.0:
        z += 0.2289144 * (24.57812 - Q.pt_5) * (8.0 - Q.n_pt_above_5)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta2 < 9.030369e-05:
        z += 95585.31 * (Q.log_sum_pt - 6.701242) * (9.030369e-05 - Q.mean_eta2)
    if Q.pair_mass_0_4 > 23.26771 and Q.mean_eta2 < 0.01423545:
        z += 2.618255 * (Q.pair_mass_0_4 - 23.26771) * (0.01423545 - Q.mean_eta2)
    if Q.lam1_plus_lam2 < 0.003562611 and Q.sj3_dr13 > 0.181053:
        z += 20987.77 * (0.003562611 - Q.lam1_plus_lam2) * (Q.sj3_dr13 - 0.181053)
    if Q.sum_zz_dr2 < 0.005284669 and Q.planar_flow < 0.3220738:
        z += 1049.532 * (0.005284669 - Q.sum_zz_dr2) * (0.3220738 - Q.planar_flow)
    if Q.zdr_0 < 0.0211821 and Q.sj3_dr13 > 0.04995258:
        z += -325.7476 * (0.0211821 - Q.zdr_0) * (Q.sj3_dr13 - 0.04995258)
    if Q.centroid_offset < 0.006789738 and Q.pt_5 < 56.4375:
        z += 8.851177 * (0.006789738 - Q.centroid_offset) * (56.4375 - Q.pt_5)
    if Q.sum_zz_dr2 < 0.005284669 and Q.dr0_5 > 0.1122797:
        z += 1076.366 * (0.005284669 - Q.sum_zz_dr2) * (Q.dr0_5 - 0.1122797)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta < -0.003031633:
        z += -1648.525 * (Q.log_sum_pt - 6.701242) * (-0.003031633 - Q.mean_eta)
    if Q.z_7 < 0.04939969 and Q.mean_eta > 0.01271871:
        z += -923.8474 * (0.04939969 - Q.z_7) * (Q.mean_eta - 0.01271871)
    if Q.log_sum_pt > 6.896095 and Q.mean_phi > -0.02594505:
        z += -353.7167 * (Q.log_sum_pt - 6.896095) * (Q.mean_phi - -0.02594505)
    if Q.pair_mass_0_4 > 23.26771 and Q.phi_4 > 0.1096802:
        z += -0.354184 * (Q.pair_mass_0_4 - 23.26771) * (Q.phi_4 - 0.1096802)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.234313
    if 0.00809236 <= Q.centroid_offset < 0.01837778:
        z += -17.08179 * Q.centroid_offset + 0.138232
    if Q.centroid_offset >= 0.01837778:
        z += -18.60125 * Q.centroid_offset + 0.1661563
    if Q.lam1_plus_lam2 < 0.01323868:
        z += 390.5241 * Q.lam1_plus_lam2 - 5.170022
    if Q.mass < 21.78408:
        z += 0.02092432 * Q.mass + 0.3364755
    if 21.78408 <= Q.mass < 45.595:
        z += -0.01780902 * Q.mass + 1.180246
    if 45.595 <= Q.mass < 60.63098:
        z += -0.02449082 * Q.mass + 1.484902
    if Q.pt_6 < 19.46875:
        z += -0.3923971 * Q.pt_6 + 9.09089
    if 19.46875 <= Q.pt_6 < 29.90625:
        z += -0.1131566 * Q.pt_6 + 3.654427
    if 29.90625 <= Q.pt_6 < 39.75:
        z += -0.01883616 * Q.pt_6 + 0.8336572
    if 39.75 <= Q.pt_6 < 41.21875:
        z += -0.05781781 * Q.pt_6 + 2.383178
    if Q.lam2 < 0.001130645:
        z += 2.27829 * Q.lam2 + 0.773297
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -340.6321 * Q.lam2 + 1.161007
    if Q.tau1 < 0.1136369:
        z += -8.677744 * Q.tau1 + 0.9861119
    if 0.02400746 <= Q.sj3_dr_min < 0.1278212:
        z += -11.24756 * Q.sj3_dr_min + 0.2700254
    if Q.sj3_dr_min >= 0.1278212:
        z += -12.47538 * Q.sj3_dr_min + 0.4269664
    if Q.lam1 < 0.00595415:
        z += -1044.366 * Q.lam1 + 8.285052
    if 0.00595415 <= Q.lam1 < 0.00733008:
        z += -833.65 * Q.lam1 + 7.030417
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += -196.7833 * Q.lam1 + 2.362133
    if Q.sj3_pair_mass_min >= 4.501727:
        z += -0.05827829 * Q.sj3_pair_mass_min + 0.2623529
    if Q.sum_pt < 615.875:
        z += -0.003669956 * Q.sum_pt + 2.260234
    if Q.sum_pt >= 988.4078:
        z += 0.002730917 * Q.sum_pt - 2.69926
    if Q.sum_zz_dr2 < 0.0030133:
        z += -481.2506 * Q.sum_zz_dr2 + 1.450153
    if Q.sj3_dr13 >= 0.181053:
        z += 0.2496098 * Q.sj3_dr13 - 0.0451926
    if Q.eccentricity >= 0.927072:
        z += 2.779965 * Q.eccentricity - 2.577228
    if Q.sj3_dr_max < 0.1789613:
        z += -10.52382 * Q.sj3_dr_max + 1.437602
    if 0.1789613 <= Q.sj3_dr_max < 0.1879486:
        z += 3.646545 * Q.sj3_dr_max - 1.098345
    if 0.1879486 <= Q.sj3_dr_max < 0.3012016:
        z += 7.717474 * Q.sj3_dr_max - 1.86347
    if Q.sj3_dr_max >= 0.3012016:
        z += 4.07093 * Q.sj3_dr_max - 0.7651253
    if Q.max_dr < 0.1452311:
        z += 5.226623 * Q.max_dr - 0.7590683
    if Q.mean_eta >= 0.02644207:
        z += 36.99311 * Q.mean_eta - 0.9781744
    if Q.e3 < 0.0005116989:
        z += -204.0593 * Q.e3 + 0.1044169
    if Q.sum_z_dr2 < 0.008678045:
        z += 572.9819 * Q.sum_z_dr2 - 4.972363
    if Q.C2_b2 < 0.02415398:
        z += -10.52567 * Q.C2_b2 + 0.2542368
    if Q.sj2_dr < 0.1872617:
        z += -9.095858 * Q.sj2_dr + 1.703306
    if Q.sj2_zsoft < 0.04176067:
        z += -37.69565 * Q.sj2_zsoft + 1.574196
    if Q.sj3_dr23 >= 0.2207152:
        z += -2.266603 * Q.sj3_dr23 + 0.5002737
    if Q.sj2_mass1 >= 40.2:
        z += 0.08272871 * Q.sj2_mass1 - 3.325694
    if Q.log_sum_pt < 6.267538:
        z += -4.989782 * Q.log_sum_pt + 31.53375
    if 6.267538 <= Q.log_sum_pt < 6.638339:
        z += -0.7014471 * Q.log_sum_pt + 4.656443
    if Q.z_6 < 0.02160287:
        z += 201.4363 * Q.z_6 - 4.351604
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.01029107 * Q.sum_pt_top5 + 8.644032
    if Q.tau2 >= 0.008780509:
        z += -29.20823 * Q.tau2 + 0.2564631
    if Q.centroid_offset > 0.00809236 and Q.sj3_pair_mass_min > 6.811308:
        z += -0.8515618 * (Q.centroid_offset - 0.00809236) * (Q.sj3_pair_mass_min - 6.811308)
    if Q.pt_6 < 41.21875 and Q.log_sum_pt < 6.766778:
        z += 0.5390424 * (41.21875 - Q.pt_6) * (6.766778 - Q.log_sum_pt)
    if Q.centroid_offset > 0.00809236 and Q.psi_0p1 > 0.4008925:
        z += -84.02459 * (Q.centroid_offset - 0.00809236) * (Q.psi_0p1 - 0.4008925)
    if Q.centroid_offset > 0.01837778 and Q.C2 < 0.02702951:
        z += -749.9882 * (Q.centroid_offset - 0.01837778) * (0.02702951 - Q.C2)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.008921136:
        z += -2999.576 * (Q.centroid_offset - 0.01837778) * (0.008921136 - Q.mean_phi2)
    if Q.lam1 < 0.01200373 and Q.planar_flow < 0.2534037:
        z += -379.6941 * (0.01200373 - Q.lam1) * (0.2534037 - Q.planar_flow)
    if Q.pt_6 < 41.21875 and Q.z_7 > 0.02320757:
        z += -3.291952 * (41.21875 - Q.pt_6) * (Q.z_7 - 0.02320757)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.mean_eta < -0.02665591:
        z += 32143.27 * (0.01323868 - Q.lam1_plus_lam2) * (-0.02665591 - Q.mean_eta)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.mean_phi < -0.02594505:
        z += 19753.24 * (0.01323868 - Q.lam1_plus_lam2) * (-0.02594505 - Q.mean_phi)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.mean_phi > 0.02612796:
        z += 21676.14 * (0.01323868 - Q.lam1_plus_lam2) * (Q.mean_phi - 0.02612796)
    if Q.sum_pt < 615.875 and Q.mean_phi > 0.01251955:
        z += 0.2452274 * (615.875 - Q.sum_pt) * (Q.mean_phi - 0.01251955)
    if Q.pt_6 < 29.90625 and Q.n_pt_above_10 < 8.0:
        z += 0.04017404 * (29.90625 - Q.pt_6) * (8.0 - Q.n_pt_above_10)
    if Q.mass < 60.63098 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -0.4882383 * (60.63098 - Q.mass) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sum_pt < 615.875 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.003234107 * (615.875 - Q.sum_pt) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.sj3_dr_max < 0.1789613 and Q.tau21_b2 < 0.1230713:
        z += 230.179 * (0.1789613 - Q.sj3_dr_max) * (0.1230713 - Q.tau21_b2)
    if Q.lam2 < 0.003408389 and Q.n_dr_0_0p05 < 3.0:
        z += -4.824414 * (0.003408389 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.max_dr < 0.1452311 and Q.mean_phi > 0.004406178:
        z += -1609.501 * (0.1452311 - Q.max_dr) * (Q.mean_phi - 0.004406178)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.eta_7 < -0.0803833:
        z += 92.67797 * (0.01323868 - Q.lam1_plus_lam2) * (-0.0803833 - Q.eta_7)
    if Q.pt_6 < 41.21875 and Q.dr1_7 < 0.1343804:
        z += -0.0001316383 * (41.21875 - Q.pt_6) * (0.1343804 - Q.dr1_7)
    if Q.lam1 < 0.01200373 and Q.mean_eta > 0.02644207:
        z += 14146.15 * (0.01200373 - Q.lam1) * (Q.mean_eta - 0.02644207)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.00433128:
        z += -21528.52 * (Q.centroid_offset - 0.01837778) * (0.00433128 - Q.mean_phi2)
    if Q.sum_pt < 615.875 and Q.z_3 < 0.05101919:
        z += 29.68373 * (615.875 - Q.sum_pt) * (0.05101919 - Q.z_3)
    if Q.pt_6 < 41.21875 and Q.M3 > 0.0782171:
        z += 5.863767 * (41.21875 - Q.pt_6) * (Q.M3 - 0.0782171)
    if Q.lam1 < 0.00733008 and Q.n_dr_0p2_0p4 < 1.0:
        z += 253.8365 * (0.00733008 - Q.lam1) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.sj3_dr13 > 0.181053 and Q.abseta_5 < 0.02658081:
        z += -231.1788 * (Q.sj3_dr13 - 0.181053) * (0.02658081 - Q.abseta_5)
    if Q.mass < 45.595 and Q.eta_3 > -0.00592804:
        z += -0.7241761 * (45.595 - Q.mass) * (Q.eta_3 - -0.00592804)
    if Q.centroid_offset > 0.01837778 and Q.eta_0 < 0.02980347:
        z += 53.23219 * (Q.centroid_offset - 0.01837778) * (0.02980347 - Q.eta_0)
    if Q.mean_eta > 0.02644207 and Q.M3 > 0.06688759:
        z += 488.5767 * (Q.mean_eta - 0.02644207) * (Q.M3 - 0.06688759)
    if Q.sj3_pair_mass_min > 4.501727 and Q.n_dr_0p2_0p4 > 1.0:
        z += 0.007979662 * (Q.sj3_pair_mass_min - 4.501727) * (Q.n_dr_0p2_0p4 - 1.0)
    if Q.sj3_dr_max > 0.1879486 and Q.abseta_2 < 0.03250122:
        z += -533.709 * (Q.sj3_dr_max - 0.1879486) * (0.03250122 - Q.abseta_2)
    if Q.sj3_dr13 > 0.181053 and Q.ptdr0_7 > 10.31097:
        z += 2.533754 * (Q.sj3_dr13 - 0.181053) * (Q.ptdr0_7 - 10.31097)
    if Q.mean_eta > 0.02644207 and Q.z_4 < 0.03747769:
        z += 104811.1 * (Q.mean_eta - 0.02644207) * (0.03747769 - Q.z_4)
    if Q.centroid_offset > 0.01837778 and Q.z_3rd < 0.1818564:
        z += -381.1614 * (Q.centroid_offset - 0.01837778) * (0.1818564 - Q.z_3rd)
    if Q.sj2_dr < 0.1872617 and Q.eccentricity > 0.927072:
        z += 89.96927 * (0.1872617 - Q.sj2_dr) * (Q.eccentricity - 0.927072)
    if Q.pt_6 < 41.21875 and Q.eta_4 > -0.1065704:
        z += -0.02635851 * (41.21875 - Q.pt_6) * (Q.eta_4 - -0.1065704)
    if Q.sj2_mass1 > 40.2 and Q.tau21_b2 < 0.4490772:
        z += -0.6913762 * (Q.sj2_mass1 - 40.2) * (0.4490772 - Q.tau21_b2)
    if Q.eccentricity > 0.927072 and Q.n_pt_above_5 < 8.0:
        z += -4.296673 * (Q.eccentricity - 0.927072) * (8.0 - Q.n_pt_above_5)
    if Q.sum_pt < 615.875 and Q.pt_5 < 24.57812:
        z += 0.008364609 * (615.875 - Q.sum_pt) * (24.57812 - Q.pt_5)
    if Q.centroid_offset > 0.01837778 and Q.pt_4 > 65.1875:
        z += -4.665349 * (Q.centroid_offset - 0.01837778) * (Q.pt_4 - 65.1875)
    if Q.centroid_offset > 0.00809236 and Q.mean_phi2 < 0.0002302115:
        z += -1001597.0 * (Q.centroid_offset - 0.00809236) * (0.0002302115 - Q.mean_phi2)
    if Q.sum_pt > 988.4078 and Q.mean_phi2 < 0.002776626:
        z += -1.596521 * (Q.sum_pt - 988.4078) * (0.002776626 - Q.mean_phi2)
    if Q.sum_pt > 988.4078 and Q.mean_phi2 > 0.008921136:
        z += 1.099898 * (Q.sum_pt - 988.4078) * (Q.mean_phi2 - 0.008921136)
    if Q.sum_pt > 988.4078 and Q.mean_phi2 < 0.002127561:
        z += 7.077916 * (Q.sum_pt - 988.4078) * (0.002127561 - Q.mean_phi2)
    if Q.tau1 < 0.1136369 and Q.mean_phi2 < 0.01426135:
        z += 1188.573 * (0.1136369 - Q.tau1) * (0.01426135 - Q.mean_phi2)
    if Q.sj2_dr < 0.1872617 and Q.dr1_6 > 0.07796252:
        z += 42.08408 * (0.1872617 - Q.sj2_dr) * (Q.dr1_6 - 0.07796252)
    if Q.log_sum_pt < 6.638339 and Q.z_7 < 0.0753896:
        z += 249.8278 * (6.638339 - Q.log_sum_pt) * (0.0753896 - Q.z_7)
    if Q.mass < 60.63098 and Q.mean_phi2 > 0.0003573041:
        z += -3.019526 * (60.63098 - Q.mass) * (Q.mean_phi2 - 0.0003573041)
    if Q.centroid_offset > 0.01837778 and Q.orientation_deg > -36.09848:
        z += -0.1187978 * (Q.centroid_offset - 0.01837778) * (Q.orientation_deg - -36.09848)
    if Q.sum_pt > 988.4078 and Q.n_pt_above_50 > 6.0:
        z += -0.008030418 * (Q.sum_pt - 988.4078) * (Q.n_pt_above_50 - 6.0)
    if Q.sum_pt < 615.875 and Q.e4 < 3.0374e-08:
        z += 160796.8 * (615.875 - Q.sum_pt) * (3.0374e-08 - Q.e4)
    if Q.mean_eta > 0.02644207 and Q.tau21_b2 < 0.003206041:
        z += 63226.83 * (Q.mean_eta - 0.02644207) * (0.003206041 - Q.tau21_b2)
    if Q.log_sum_pt < 6.267538 and Q.sj2_mass2 < 6.284176:
        z += 0.498152 * (6.267538 - Q.log_sum_pt) * (6.284176 - Q.sj2_mass2)
    if Q.sj2_zsoft < 0.04176067 and Q.phi_0 < -0.0216713:
        z += 23087.95 * (0.04176067 - Q.sj2_zsoft) * (-0.0216713 - Q.phi_0)
    if Q.centroid_offset > 0.01837778 and Q.sj3_dr12 > 0.1692253:
        z += 283.0788 * (Q.centroid_offset - 0.01837778) * (Q.sj3_dr12 - 0.1692253)
    if Q.pt_6 < 29.90625 and Q.z_dr_0p2_0p4 > 0.1009734:
        z += -0.3850238 * (29.90625 - Q.pt_6) * (Q.z_dr_0p2_0p4 - 0.1009734)
    if Q.pt_6 < 19.46875 and Q.min_pair_mass < 1.813997:
        z += 0.08322169 * (19.46875 - Q.pt_6) * (1.813997 - Q.min_pair_mass)
    if Q.mass < 60.63098 and Q.phi_0 < -0.008995056:
        z += -0.2951116 * (60.63098 - Q.mass) * (-0.008995056 - Q.phi_0)
    return max(0.0, z)


def neuron_7(Q):
    z = 9.934425
    if Q.planar_flow < 0.1950135:
        z += -1.817832 * Q.planar_flow + 0.3545019
    if Q.sum_z_dr2_top2 < 8.10414e-05:
        z += 18225.11 * Q.sum_z_dr2_top2 - 2.117203
    if 8.10414e-05 <= Q.sum_z_dr2_top2 < 0.001056655:
        z += 656.2172 * Q.sum_z_dr2_top2 - 0.6933953
    if Q.sum_z_dr2 < 0.0009641429:
        z += 3022.551 * Q.sum_z_dr2 - 4.412796
    if 0.0009641429 <= Q.sum_z_dr2 < 0.001653836:
        z += 576.7341 * Q.sum_z_dr2 - 2.054679
    if 0.001653836 <= Q.sum_z_dr2 < 0.003562611:
        z += 426.2756 * Q.sum_z_dr2 - 1.805846
    if 0.003562611 <= Q.sum_z_dr2 < 0.004372139:
        z += -150.4586 * Q.sum_z_dr2 + 0.2488338
    if 0.004372139 <= Q.sum_z_dr2 < 0.007520088:
        z += -697.7806 * Q.sum_z_dr2 + 2.641802
    if 0.007520088 <= Q.sum_z_dr2 < 0.01323868:
        z += -1399.911 * Q.sum_z_dr2 + 7.921886
    if Q.sum_z_dr2 >= 0.01323868:
        z += -605.3567 * Q.sum_z_dr2 - 2.596963
    if 0.01109984 <= Q.mass_over_sum_pt < 0.07269073:
        z += -49.65728 * Q.mass_over_sum_pt + 0.551188
    if 0.07269073 <= Q.mass_over_sum_pt < 0.07992374:
        z += 60.38898 * Q.mass_over_sum_pt - 7.448156
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 147.1902 * Q.mass_over_sum_pt - 14.38564
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 233.3235 * Q.mass_over_sum_pt - 21.68557
    if 0.09041383 <= Q.mass_over_sum_pt < 0.1079857:
        z += -79.62463 * Q.mass_over_sum_pt + 6.60927
    if Q.mass_over_sum_pt >= 0.1079857:
        z += -85.59748 * Q.mass_over_sum_pt + 7.254251
    if Q.tau1 < 0.05356915:
        z += -47.87725 * Q.tau1 + 2.564744
    if Q.sum_z_dr < 0.04081947:
        z += 120.3061 * Q.sum_z_dr - 8.422024
    if 0.04081947 <= Q.sum_z_dr < 0.08723651:
        z += 75.6445 * Q.sum_z_dr - 6.598962
    if 36.22941 <= Q.mass < 76.6557:
        z += 0.04040162 * Q.mass - 1.463727
    if Q.mass >= 76.6557:
        z += -0.08672047 * Q.mass + 8.280906
    if Q.z_dr_0p1_0p2 < 0.1585582:
        z += -2.829039 * Q.z_dr_0p1_0p2 + 0.4485674
    if Q.centroid_offset < 0.02076709:
        z += 35.89497 * Q.centroid_offset - 0.7454341
    if 0.03117077 <= Q.centroid_offset < 0.03776099:
        z += -115.3374 * Q.centroid_offset + 3.595157
    if Q.centroid_offset >= 0.03776099:
        z += -130.5106 * Q.centroid_offset + 4.168109
    if Q.lam1_plus_lam2 < 0.005590289:
        z += 761.918 * Q.lam1_plus_lam2 - 5.08921
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.006679471:
        z += 69.93085 * Q.lam1_plus_lam2 - 1.220802
    if 0.006679471 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -691.9872 * Q.lam1_plus_lam2 + 3.868408
    if Q.lam1_plus_lam2 >= 0.008678045:
        z += 486.9941 * Q.lam1_plus_lam2 - 6.362845
    if Q.z_7 >= 0.03243272:
        z += 28.99441 * Q.z_7 - 0.9403677
    if 0.03556091 <= Q.e2 < 0.05028464:
        z += 79.14471 * Q.e2 - 2.814458
    if Q.e2 >= 0.05028464:
        z += -37.47555 * Q.e2 + 3.04975
    if Q.sj2_dr < 0.1294903:
        z += -13.21354 * Q.sj2_dr + 1.314922
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -9.48237 * Q.sj2_dr + 0.8317713
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 24.12032 * Q.sj2_dr - 4.516813
    if Q.sum_zz_dr2 < 0.001101266:
        z += -1550.888 * Q.sum_zz_dr2 + 1.707941
    if Q.LHA < 0.3033137:
        z += -11.18466 * Q.LHA + 3.392461
    if Q.pt_7 < 29.04219:
        z += 0.05442836 * Q.pt_7 - 1.580719
    if 0.2037854 <= Q.sd_rg < 0.2787955:
        z += -11.58239 * Q.sd_rg + 2.360321
    if Q.sd_rg >= 0.2787955:
        z += -17.79934 * Q.sd_rg + 4.09358
    if Q.lam1 < 0.008375572:
        z += 51.16714 * Q.lam1 - 0.4285541
    if Q.e3 < 8.147744e-05:
        z += 1580.14 * Q.e3 - 0.1287457
    if Q.tau21_b2 < 0.009213932:
        z += 51.91126 * Q.tau21_b2 - 1.003531
    if 0.009213932 <= Q.tau21_b2 < 0.04019753:
        z += 16.95167 * Q.tau21_b2 - 0.6814153
    if Q.mass_top5 >= 53.60766:
        z += 0.08061767 * Q.mass_top5 - 4.321725
    if Q.D2_b2 < 0.05744392:
        z += 9.925106 * Q.D2_b2 - 0.570137
    if 0.04889979 <= Q.sj3_dr_max < 0.1426152:
        z += -3.865721 * Q.sj3_dr_max + 0.1890329
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += 21.37155 * Q.sj3_dr_max - 3.410185
    if 0.213399 <= Q.sj3_dr_max < 0.2623172:
        z += 6.878925 * Q.sj3_dr_max - 0.317473
    if Q.sj3_dr_max >= 0.2623172:
        z += -6.192647 * Q.sj3_dr_max + 3.111425
    if Q.lam2 < 0.0003061234:
        z += 586.4347 * Q.lam2 - 0.1795214
    if Q.D2 < 2.357246:
        z += -0.3373979 * Q.D2 + 0.7953301
    if Q.max_dr < 0.1117619:
        z += -2.081704 * Q.max_dr + 1.081778
    if 0.1117619 <= Q.max_dr < 0.177305:
        z += -12.95518 * Q.max_dr + 2.297019
    if Q.planar_flow < 0.1950135 and Q.lam1_plus_lam2 > 0.007520088:
        z += -5556.14 * (0.1950135 - Q.planar_flow) * (Q.lam1_plus_lam2 - 0.007520088)
    if Q.sum_z_dr2_top2 < 0.001056655 and Q.centroid_offset > 0.006789738:
        z += 163962.9 * (0.001056655 - Q.sum_z_dr2_top2) * (Q.centroid_offset - 0.006789738)
    if Q.planar_flow < 0.1950135 and Q.pt_6 < 35.28125:
        z += -0.3015159 * (0.1950135 - Q.planar_flow) * (35.28125 - Q.pt_6)
    if Q.planar_flow < 0.1950135 and Q.sd_mass > 38.43971:
        z += 0.1842263 * (0.1950135 - Q.planar_flow) * (Q.sd_mass - 38.43971)
    if Q.z_dr_0p1_0p2 < 0.1585582 and Q.N3 < 2.080881:
        z += -2.24087 * (0.1585582 - Q.z_dr_0p1_0p2) * (2.080881 - Q.N3)
    if Q.centroid_offset > 0.03117077 and Q.n_pt_above_50 > 4.0:
        z += -42.02154 * (Q.centroid_offset - 0.03117077) * (Q.n_pt_above_50 - 4.0)
    if Q.tau1 < 0.05356915 and Q.n_dr_0p05_0p1 > 5.0:
        z += -6.708694 * (0.05356915 - Q.tau1) * (Q.n_dr_0p05_0p1 - 5.0)
    if Q.centroid_offset < 0.02076709 and Q.sum_pt_top3 > 331.25:
        z += 0.05668475 * (0.02076709 - Q.centroid_offset) * (Q.sum_pt_top3 - 331.25)
    if Q.sum_z_dr < 0.08723651 and Q.mean_phi < 0.004406178:
        z += 1213.829 * (0.08723651 - Q.sum_z_dr) * (0.004406178 - Q.mean_phi)
    if Q.sum_zz_dr2 < 0.001101266 and Q.phi_1 < -0.04275513:
        z += -592181.3 * (0.001101266 - Q.sum_zz_dr2) * (-0.04275513 - Q.phi_1)
    if Q.sum_z_dr2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 3070.846 * (Q.sum_z_dr2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    if Q.sum_z_dr2 > 0.01323868 and Q.eccentricity > 0.9458207:
        z += 2487.421 * (Q.sum_z_dr2 - 0.01323868) * (Q.eccentricity - 0.9458207)
    if Q.e3 < 8.147744e-05 and Q.sj2_mass1 > 31.78116:
        z += 3516.206 * (8.147744e-05 - Q.e3) * (Q.sj2_mass1 - 31.78116)
    if Q.sum_zz_dr2 < 0.001101266 and Q.eta_2 < -0.06384277:
        z += 742163.1 * (0.001101266 - Q.sum_zz_dr2) * (-0.06384277 - Q.eta_2)
    if Q.centroid_offset > 0.03776099 and Q.pt_4 > 81.375:
        z += -76.63445 * (Q.centroid_offset - 0.03776099) * (Q.pt_4 - 81.375)
    if Q.sum_zz_dr2 < 0.001101266 and Q.phi_0 > 0.0402832:
        z += -476472.7 * (0.001101266 - Q.sum_zz_dr2) * (Q.phi_0 - 0.0402832)
    if Q.planar_flow < 0.1950135 and Q.pt_4 > 71.6875:
        z += 0.1548312 * (0.1950135 - Q.planar_flow) * (Q.pt_4 - 71.6875)
    if Q.sum_z_dr2 < 0.003562611 and Q.mean_eta < -0.006779839:
        z += 66973.25 * (0.003562611 - Q.sum_z_dr2) * (-0.006779839 - Q.mean_eta)
    if Q.centroid_offset < 0.02076709 and Q.C2_b2 < 0.004032342:
        z += -30077.05 * (0.02076709 - Q.centroid_offset) * (0.004032342 - Q.C2_b2)
    if Q.sum_z_dr < 0.08723651 and Q.C2_b2 < 0.004032342:
        z += 1262.866 * (0.08723651 - Q.sum_z_dr) * (0.004032342 - Q.C2_b2)
    if Q.centroid_offset < 0.02076709 and Q.tau21_b2 < 0.02656143:
        z += 4726.604 * (0.02076709 - Q.centroid_offset) * (0.02656143 - Q.tau21_b2)
    if Q.sum_z_dr2_top2 < 0.001056655 and Q.tau21_b2 < 0.02656143:
        z += -187364.7 * (0.001056655 - Q.sum_z_dr2_top2) * (0.02656143 - Q.tau21_b2)
    if Q.centroid_offset < 0.02076709 and Q.zdr_5 > 0.007104199:
        z += 11712.76 * (0.02076709 - Q.centroid_offset) * (Q.zdr_5 - 0.007104199)
    if Q.pt_7 < 29.04219 and Q.tau21 < 0.2838437:
        z += -0.4559086 * (29.04219 - Q.pt_7) * (0.2838437 - Q.tau21)
    if Q.planar_flow < 0.1950135 and Q.dr0_5 < 0.01961688:
        z += 137.6876 * (0.1950135 - Q.planar_flow) * (0.01961688 - Q.dr0_5)
    if Q.tau21_b2 < 0.009213932 and Q.sj2_mass1 > 3.458544:
        z += -78.7302 * (0.009213932 - Q.tau21_b2) * (Q.sj2_mass1 - 3.458544)
    if Q.lam2 < 0.0003061234 and Q.tau21_b2 < 0.04019753:
        z += 83861.27 * (0.0003061234 - Q.lam2) * (0.04019753 - Q.tau21_b2)
    if Q.mass_over_sum_pt > 0.09041383 and Q.pt_6 < 36.8125:
        z += 1.072516 * (Q.mass_over_sum_pt - 0.09041383) * (36.8125 - Q.pt_6)
    if Q.sj3_dr_max > 0.1426152 and Q.pt_6 < 19.46875:
        z += 0.5803406 * (Q.sj3_dr_max - 0.1426152) * (19.46875 - Q.pt_6)
    if Q.sum_z_dr2 > 0.01323868 and Q.pt_6 < 38.25:
        z += -26.99383 * (Q.sum_z_dr2 - 0.01323868) * (38.25 - Q.pt_6)
    if Q.mass_over_sum_pt > 0.01109984 and Q.pt_6 < 35.28125:
        z += -0.3924323 * (Q.mass_over_sum_pt - 0.01109984) * (35.28125 - Q.pt_6)
    if Q.sj2_dr < 0.1294903 and Q.sj2_mass1 > 31.78116:
        z += 0.08242065 * (0.1294903 - Q.sj2_dr) * (Q.sj2_mass1 - 31.78116)
    if Q.sj2_dr < 0.1591713 and Q.sj2_mass1 > 40.2:
        z += 24.21117 * (0.1591713 - Q.sj2_dr) * (Q.sj2_mass1 - 40.2)
    if Q.sum_zz_dr2 < 0.001101266 and Q.eta_2 > 0.04476929:
        z += -283497.0 * (0.001101266 - Q.sum_zz_dr2) * (Q.eta_2 - 0.04476929)
    if Q.tau21_b2 < 0.04019753 and Q.phi_0 < 0.01452637:
        z += -2.244845 * (0.04019753 - Q.tau21_b2) * (0.01452637 - Q.phi_0)
    if Q.lam1_plus_lam2 > 0.008678045 and Q.pt_6 < 38.25:
        z += 1.374034 * (Q.lam1_plus_lam2 - 0.008678045) * (38.25 - Q.pt_6)
    if Q.sum_zz_dr2 < 0.001101266 and Q.eta_2 < -0.04544525:
        z += -430986.3 * (0.001101266 - Q.sum_zz_dr2) * (-0.04544525 - Q.eta_2)
    if Q.D2_b2 < 0.05744392 and Q.zdr_6 < 0.00177424:
        z += -15022.44 * (0.05744392 - Q.D2_b2) * (0.00177424 - Q.zdr_6)
    if Q.z_dr_0p1_0p2 < 0.1585582 and Q.mean_phi < -0.01275329:
        z += 50.88065 * (0.1585582 - Q.z_dr_0p1_0p2) * (-0.01275329 - Q.mean_phi)
    if Q.max_dr < 0.177305 and Q.dr0_4 > 0.1878822:
        z += -521.8148 * (0.177305 - Q.max_dr) * (Q.dr0_4 - 0.1878822)
    if Q.mass > 76.6557 and Q.zdr_6 > 0.006366792:
        z += -8.108617 * (Q.mass - 76.6557) * (Q.zdr_6 - 0.006366792)
    if Q.mass_over_sum_pt > 0.08475161 and Q.pt_6 < 36.8125:
        z += -5.75205 * (Q.mass_over_sum_pt - 0.08475161) * (36.8125 - Q.pt_6)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.3357185
    if Q.sum_z_dr2 < 0.001653836:
        z += -101.9645 * Q.sum_z_dr2 + 0.5746052
    if 0.001653836 <= Q.sum_z_dr2 < 0.005019719:
        z += -163.1169 * Q.sum_z_dr2 + 0.6757412
    if 0.005019719 <= Q.sum_z_dr2 < 0.006679471:
        z += 86.19344 * Q.sum_z_dr2 - 0.5757266
    if Q.tau1 < 0.05356915:
        z += 5.541585 * Q.tau1 - 0.296858
    if Q.LHA < 0.07658656:
        z += 12.844 * Q.LHA - 2.051452
    if 0.07658656 <= Q.LHA < 0.1967397:
        z += 8.88678 * Q.LHA - 1.748383
    if Q.log_sum_pt >= 6.701242:
        z += 0.2531282 * Q.log_sum_pt - 1.696274
    if Q.mass < 6.633799:
        z += 0.08504363 * Q.mass - 1.14996
    if 6.633799 <= Q.mass < 21.78408:
        z += 0.04526802 * Q.mass - 0.8860966
    if 21.78408 <= Q.mass < 29.6447:
        z += -0.002756141 * Q.mass + 0.1600655
    if 29.6447 <= Q.mass < 49.6681:
        z += -0.003913445 * Q.mass + 0.1943734
    if Q.sum_z_dr < 0.06108601:
        z += 58.91788 * Q.sum_z_dr - 3.599058
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.0001169064 * Q.sum_pt_top5 - 0.08036586
    if Q.centroid_offset < 0.003343241:
        z += 158.9969 * Q.centroid_offset - 0.5315652
    if Q.lam2 < 0.0001330621:
        z += 1994.365 * Q.lam2 - 0.2653744
    if Q.dr_0 < 0.003239423:
        z += 409.169 * Q.dr_0 - 1.650184
    if 0.003239423 <= Q.dr_0 < 0.008355823:
        z += 63.46493 * Q.dr_0 - 0.5303017
    if Q.z_dr_0_0p05 >= 0.8477313:
        z += -1.802675 * Q.z_dr_0_0p05 + 1.528184
    if Q.lam1_plus_lam2 < 0.002635418:
        z += -1657.032 * Q.lam1_plus_lam2 + 5.442614
    if 0.002635418 <= Q.lam1_plus_lam2 < 0.003562611:
        z += -1160.106 * Q.lam1_plus_lam2 + 4.133007
    if Q.lam1_plus_lam2 >= 0.02530566:
        z += -7.165914 * Q.lam1_plus_lam2 + 0.1813382
    if Q.z_6 < 0.03448406:
        z += -17.05263 * Q.z_6 + 0.5880437
    if Q.mass_over_sum_pt < 0.03319429:
        z += 23.03464 * Q.mass_over_sum_pt - 0.7646186
    if Q.sj3_dr_max < 0.1070199:
        z += 12.25594 * Q.sj3_dr_max - 1.837683
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 12.92333 * Q.sj3_dr_max - 1.909107
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += 1.179104 * Q.sj3_dr_max - 0.2342022
    if Q.e2 < 0.007078158:
        z += -7.690769 * Q.e2 + 0.4064696
    if 0.007078158 <= Q.e2 < 0.01289969:
        z += -60.47086 * Q.e2 + 0.7800554
    if Q.D2 >= 3.885568:
        z += 0.1441281 * Q.D2 - 0.5600197
    if Q.pt_7 >= 34.53125:
        z += -0.02948273 * Q.pt_7 + 1.018075
    if Q.sum_pt >= 988.4078:
        z += -0.002712187 * Q.sum_pt + 2.680747
    if Q.sum_z_dr2 < 0.006679471 and Q.D2_b2 < 4.721224:
        z += -9.69358 * (0.006679471 - Q.sum_z_dr2) * (4.721224 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 20482.03 * (0.006679471 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.sum_z_dr2 < 0.005019719 and Q.log_sum_pt < 6.502799:
        z += -1201.808 * (0.005019719 - Q.sum_z_dr2) * (6.502799 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.006679471 and Q.planar_flow < 0.4007947:
        z += 278.0262 * (0.006679471 - Q.sum_z_dr2) * (0.4007947 - Q.planar_flow)
    if Q.sum_z_dr2 < 0.005019719 and Q.pt_7 < 43.5:
        z += -2.531531 * (0.005019719 - Q.sum_z_dr2) * (43.5 - Q.pt_7)
    if Q.sum_z_dr2 < 0.006679471 and Q.n_dr_0p05_0p1 > 0.0:
        z += -9.819159 * (0.006679471 - Q.sum_z_dr2) * (Q.n_dr_0p05_0p1 - 0.0)
    if Q.LHA < 0.1967397 and Q.mean_phi > -0.0008556753:
        z += 691.3481 * (0.1967397 - Q.LHA) * (Q.mean_phi - -0.0008556753)
    if Q.sum_z_dr2 < 0.005019719 and Q.m012 > 28.3451:
        z += -7.368864 * (0.005019719 - Q.sum_z_dr2) * (Q.m012 - 28.3451)
    if Q.sum_z_dr2 < 0.006679471 and Q.m012 > 8.921413:
        z += 1.704227 * (0.006679471 - Q.sum_z_dr2) * (Q.m012 - 8.921413)
    if Q.sum_z_dr2 < 0.006679471 and Q.n_dr_0p1_0p2 > 2.0:
        z += -92.89847 * (0.006679471 - Q.sum_z_dr2) * (Q.n_dr_0p1_0p2 - 2.0)
    if Q.sum_z_dr2 < 0.006679471 and Q.phi_0 > -0.04013062:
        z += 460.5211 * (0.006679471 - Q.sum_z_dr2) * (Q.phi_0 - -0.04013062)
    if Q.sum_z_dr2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 4725.421 * (0.005019719 - Q.sum_z_dr2) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.sum_z_dr < 0.06108601 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -33.02829 * (0.06108601 - Q.sum_z_dr) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt > 6.701242 and Q.zdr_0 > 0.007798268:
        z += -261.5601 * (Q.log_sum_pt - 6.701242) * (Q.zdr_0 - 0.007798268)
    if Q.sum_pt_top5 > 687.4375 and Q.e3 < 3.10694e-07:
        z += 8936.843 * (Q.sum_pt_top5 - 687.4375) * (3.10694e-07 - Q.e3)
    if Q.mass < 29.6447 and Q.D2_b2 < 0.716559:
        z += -0.06912328 * (29.6447 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.sum_z_dr < 0.06108601 and Q.lam2 < 0.0001947983:
        z += 110835.3 * (0.06108601 - Q.sum_z_dr) * (0.0001947983 - Q.lam2)
    if Q.lam2 < 0.0001330621 and Q.D2_b2 < 4.721224:
        z += 758.1955 * (0.0001330621 - Q.lam2) * (4.721224 - Q.D2_b2)
    if Q.tau1 < 0.05356915 and Q.D2_b2 < 4.721224:
        z += -1.0856 * (0.05356915 - Q.tau1) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.701242 and Q.e3 < 2.955458e-05:
        z += -18196.2 * (Q.log_sum_pt - 6.701242) * (2.955458e-05 - Q.e3)
    if Q.mass < 21.78408 and Q.lam2 < 0.0001947983:
        z += -571.243 * (21.78408 - Q.mass) * (0.0001947983 - Q.lam2)
    if Q.mass < 29.6447 and Q.e3 < 1.762929e-06:
        z += 2903.918 * (29.6447 - Q.mass) * (1.762929e-06 - Q.e3)
    if Q.LHA < 0.1967397 and Q.mean_eta > 0.003218391:
        z += 623.0001 * (0.1967397 - Q.LHA) * (Q.mean_eta - 0.003218391)
    if Q.mass < 49.6681 and Q.lam2 < 0.0001330621:
        z += 181.3227 * (49.6681 - Q.mass) * (0.0001330621 - Q.lam2)
    if Q.centroid_offset < 0.003343241 and Q.D2_b2 < 0.5327104:
        z += 132.587 * (0.003343241 - Q.centroid_offset) * (0.5327104 - Q.D2_b2)
    if Q.LHA < 0.1967397 and Q.psi_0p3 < 1.0:
        z += -29.11921 * (0.1967397 - Q.LHA) * (1.0 - Q.psi_0p3)
    if Q.LHA < 0.1967397 and Q.n_pt_above_50 > 6.0:
        z += -0.409154 * (0.1967397 - Q.LHA) * (Q.n_pt_above_50 - 6.0)
    if Q.mass < 21.78408 and Q.pt_7 < 48.71875:
        z += -0.0001749418 * (21.78408 - Q.mass) * (48.71875 - Q.pt_7)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 < 48.71875:
        z += 0.1036567 * (Q.log_sum_pt - 6.701242) * (48.71875 - Q.pt_7)
    if Q.mass < 21.78408 and Q.D2_b2 < 0.716559:
        z += -0.08701324 * (21.78408 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.sum_pt_top5 > 687.4375 and Q.n_pt_above_10 < 8.0:
        z += -0.0004374655 * (Q.sum_pt_top5 - 687.4375) * (8.0 - Q.n_pt_above_10)
    if Q.sum_z_dr2 < 0.006679471 and Q.n_pt_above_50 > 3.0:
        z += -9.903461 * (0.006679471 - Q.sum_z_dr2) * (Q.n_pt_above_50 - 3.0)
    if Q.sum_z_dr2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -24362.04 * (0.005019719 - Q.sum_z_dr2) * (Q.centroid_offset - 0.006789738)
    if Q.LHA < 0.1967397 and Q.pt_7 > 15.55391:
        z += -0.00622663 * (0.1967397 - Q.LHA) * (Q.pt_7 - 15.55391)
    if Q.sum_z_dr < 0.06108601 and Q.lam1_plus_lam2 > 0.0003193707:
        z += 5807.888 * (0.06108601 - Q.sum_z_dr) * (Q.lam1_plus_lam2 - 0.0003193707)
    if Q.log_sum_pt > 6.701242 and Q.lam1_plus_lam2 < 0.008678045:
        z += -796.3335 * (Q.log_sum_pt - 6.701242) * (0.008678045 - Q.lam1_plus_lam2)
    if Q.mass < 29.6447 and Q.lam1_plus_lam2 < 0.003562611:
        z += -3.21691 * (29.6447 - Q.mass) * (0.003562611 - Q.lam1_plus_lam2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2 < 0.006679471:
        z += 3063.39 * (0.1986272 - Q.sj3_dr_max) * (0.006679471 - Q.sum_z_dr2)
    if Q.tau1 < 0.05356915 and Q.n_dr_0p2_0p4 > 0.0:
        z += -21.95881 * (0.05356915 - Q.tau1) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.tau1 < 0.05356915 and Q.lam1_plus_lam2 < 0.003562611:
        z += 17354.86 * (0.05356915 - Q.tau1) * (0.003562611 - Q.lam1_plus_lam2)
    if Q.log_sum_pt > 6.701242 and Q.centroid_offset < 0.02355416:
        z += -120.0382 * (Q.log_sum_pt - 6.701242) * (0.02355416 - Q.centroid_offset)
    if Q.sum_z_dr < 0.06108601 and Q.sum_z_dr2 > 0.005019719:
        z += 46950.07 * (0.06108601 - Q.sum_z_dr) * (Q.sum_z_dr2 - 0.005019719)
    if Q.z_dr_0_0p05 > 0.8477313 and Q.pt_2 < 96.6875:
        z += 0.007217289 * (Q.z_dr_0_0p05 - 0.8477313) * (96.6875 - Q.pt_2)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2 < 0.0005611231:
        z += 5672.354 * (Q.log_sum_pt - 6.701242) * (0.0005611231 - Q.sum_z_dr2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2_top5 < 0.005691733:
        z += -601.838 * (0.1986272 - Q.sj3_dr_max) * (0.005691733 - Q.sum_z_dr2_top5)
    if Q.sj3_dr_max < 0.1986272 and Q.lam2 < 0.0001947983:
        z += 33060.36 * (0.1986272 - Q.sj3_dr_max) * (0.0001947983 - Q.lam2)
    if Q.pt_7 > 34.53125 and Q.pt_1 < 142.5:
        z += -9.947659e-05 * (Q.pt_7 - 34.53125) * (142.5 - Q.pt_1)
    if Q.sj3_dr_max < 0.1986272 and Q.m01 > 22.84498:
        z += -0.06022815 * (0.1986272 - Q.sj3_dr_max) * (Q.m01 - 22.84498)
    if Q.LHA < 0.1967397 and Q.centroid_offset > 0.01837778:
        z += -1809.512 * (0.1967397 - Q.LHA) * (Q.centroid_offset - 0.01837778)
    if Q.log_sum_pt > 6.701242 and Q.n_dr_0p05_0p1 > 2.0:
        z += -0.8632926 * (Q.log_sum_pt - 6.701242) * (Q.n_dr_0p05_0p1 - 2.0)
    if Q.sum_z_dr < 0.06108601 and Q.centroid_offset > 0.006789738:
        z += 147.4195 * (0.06108601 - Q.sum_z_dr) * (Q.centroid_offset - 0.006789738)
    if Q.lam1_plus_lam2 < 0.003562611 and Q.centroid_offset > 0.006789738:
        z += -43247.78 * (0.003562611 - Q.lam1_plus_lam2) * (Q.centroid_offset - 0.006789738)
    if Q.mass < 29.6447 and Q.centroid_offset < 0.02685622:
        z += -3.803087 * (29.6447 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass_over_sum_pt < 0.03319429 and Q.centroid_offset < 0.02685622:
        z += 3139.665 * (0.03319429 - Q.mass_over_sum_pt) * (0.02685622 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.1986272 and Q.lam1_plus_lam2 < 0.003562611:
        z += -2546.273 * (0.1986272 - Q.sj3_dr_max) * (0.003562611 - Q.lam1_plus_lam2)
    if Q.sum_z_dr < 0.06108601 and Q.ptdr0_2 > 21.99783:
        z += -139.498 * (0.06108601 - Q.sum_z_dr) * (Q.ptdr0_2 - 21.99783)
    if Q.sum_pt_top5 > 687.4375 and Q.mean_phi2 < 0.0005394348:
        z += -0.7338214 * (Q.sum_pt_top5 - 687.4375) * (0.0005394348 - Q.mean_phi2)
    if Q.e2 < 0.007078158 and Q.sum_z_dr2_top5 > 5.672047e-05:
        z += -149376.7 * (0.007078158 - Q.e2) * (Q.sum_z_dr2_top5 - 5.672047e-05)
    if Q.LHA < 0.1967397 and Q.mean_eta < -0.02665591:
        z += 19539.35 * (0.1967397 - Q.LHA) * (-0.02665591 - Q.mean_eta)
    if Q.sum_z_dr < 0.06108601 and Q.sum_z_dr2 < 0.003562611:
        z += -27414.6 * (0.06108601 - Q.sum_z_dr) * (0.003562611 - Q.sum_z_dr2)
    if Q.sj3_dr_max < 0.1426152 and Q.centroid_offset > 0.01627885:
        z += -201.7805 * (0.1426152 - Q.sj3_dr_max) * (Q.centroid_offset - 0.01627885)
    if Q.sj3_dr_max < 0.1986272 and Q.phi_7 > -0.08051147:
        z += 6.578489 * (0.1986272 - Q.sj3_dr_max) * (Q.phi_7 - -0.08051147)
    if Q.sj3_dr_max < 0.1986272 and Q.eta_2 < -0.01586914:
        z += 44.71545 * (0.1986272 - Q.sj3_dr_max) * (-0.01586914 - Q.eta_2)
    if Q.e2 < 0.01289969 and Q.psi_0p2 > 0.79448:
        z += -611.6916 * (0.01289969 - Q.e2) * (Q.psi_0p2 - 0.79448)
    if Q.sum_z_dr2 < 0.006679471 and Q.eta_0 > 0.02130127:
        z += -459.6308 * (0.006679471 - Q.sum_z_dr2) * (Q.eta_0 - 0.02130127)
    if Q.sj3_dr_max < 0.1986272 and Q.abseta_4 > 0.02116394:
        z += -93.78685 * (0.1986272 - Q.sj3_dr_max) * (Q.abseta_4 - 0.02116394)
    if Q.sum_pt_top5 > 687.4375 and Q.pair_mass_0_2 < 32.99439:
        z += -3.742098e-05 * (Q.sum_pt_top5 - 687.4375) * (32.99439 - Q.pair_mass_0_2)
    if Q.sum_pt > 988.4078 and Q.dr12 < 0.05062541:
        z += 0.01386533 * (Q.sum_pt - 988.4078) * (0.05062541 - Q.dr12)
    if Q.lam2 < 0.0001330621 and Q.sj3_pairmin_over_m < 0.1340569:
        z += -17395.08 * (0.0001330621 - Q.lam2) * (0.1340569 - Q.sj3_pairmin_over_m)
    return max(0.0, z)


def neuron_9(Q):
    z = -2.854382
    if Q.sum_z_dr < 0.05464922:
        z += 36.06879 * Q.sum_z_dr - 1.971131
    if Q.sum_z_dr >= 0.0717028:
        z += -21.14396 * Q.sum_z_dr + 1.516081
    if Q.tau1 < 0.04369778:
        z += -37.81849 * Q.tau1 + 1.652584
    if Q.mass < 29.6447:
        z += 0.1097473 * Q.mass - 4.394113
    if 29.6447 <= Q.mass < 45.595:
        z += 0.04815536 * Q.mass - 2.56824
    if 45.595 <= Q.mass < 53.33237:
        z += 0.0427422 * Q.mass - 2.321426
    if Q.mass >= 53.33237:
        z += -0.00541316 * Q.mass + 0.246813
    if Q.e3 < 2.371297e-05:
        z += 27172.01 * Q.e3 - 0.6443292
    if 8.147744e-05 <= Q.e3 < 0.0001869378:
        z += -5761.865 * Q.e3 + 0.469462
    if Q.e3 >= 0.0001869378:
        z += 192.3667 * Q.e3 - 0.6436092
    if Q.sum_z_dr2 < 0.003562611:
        z += -3474.11 * Q.sum_z_dr2 + 16.13361
    if 0.003562611 <= Q.sum_z_dr2 < 0.005019719:
        z += -1694.171 * Q.sum_z_dr2 + 9.792375
    if 0.005019719 <= Q.sum_z_dr2 < 0.00609665:
        z += -1196.097 * Q.sum_z_dr2 + 7.292182
    if Q.sum_z_dr2 >= 0.01882765:
        z += 24.68533 * Q.sum_z_dr2 - 0.4647668
    if Q.sj3_dr_max < 0.1070199:
        z += 13.01527 * Q.sj3_dr_max - 0.6013744
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 17.11043 * Q.sj3_dr_max - 1.039638
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -19.78656 * Q.sj3_dr_max + 4.222432
    if Q.lam2 < 0.0003061234:
        z += -2439.914 * Q.lam2 + 0.7469146
    if Q.lam2 >= 0.001130645:
        z += 145.6433 * Q.lam2 - 0.1646709
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += -0.2208423 * Q.n_dr_0p2_0p4 + 0.2208423
    if Q.lam1 < 0.003377388:
        z += 466.4297 * Q.lam1 - 2.777192
    if 0.003377388 <= Q.lam1 < 0.005433361:
        z += 167.8421 * Q.lam1 - 1.768746
    if 0.005433361 <= Q.lam1 < 0.00595415:
        z += 574.7767 * Q.lam1 - 3.979768
    if Q.lam1 >= 0.00595415:
        z += 108.347 * Q.lam1 - 1.202576
    if Q.centroid_offset < 0.002316125:
        z += -863.3504 * Q.centroid_offset + 4.751538
    if 0.002316125 <= Q.centroid_offset < 0.01837778:
        z += -171.3342 * Q.centroid_offset + 3.148742
    if 0.03556091 <= Q.e2 < 0.04447357:
        z += -59.50344 * Q.e2 + 2.115997
    if Q.e2 >= 0.04447357:
        z += -27.76585 * Q.e2 + 0.7045127
    if Q.sj3_pair_mass_max < 24.23013:
        z += 0.0181251 * Q.sj3_pair_mass_max - 0.4391735
    if Q.lam1_plus_lam2 < 0.0001721983:
        z += -14940.57 * Q.lam1_plus_lam2 + 2.572742
    if Q.C3 < 0.0284695:
        z += 24.74728 * Q.C3 - 0.7045426
    if Q.zdr_0 < 0.006292091:
        z += -121.1734 * Q.zdr_0 + 0.7624338
    if Q.mass_over_sum_pt < 0.07269073:
        z += 59.85414 * Q.mass_over_sum_pt - 4.350841
    if Q.sum_pt < 559.6875:
        z += -0.002013575 * Q.sum_pt + 2.011514
    if 559.6875 <= Q.sum_pt < 988.4078:
        z += -0.002063213 * Q.sum_pt + 2.039296
    z += -6.032282 * Q.z_dr_0p2_0p4
    if Q.log_sum_pt >= 6.896095:
        z += -5.511571 * Q.log_sum_pt + 38.00832
    if Q.sum_pt_top5 >= 839.9547:
        z += 0.0001916174 * Q.sum_pt_top5 - 0.1609499
    if Q.pt_4 < 31.125:
        z += -0.2394968 * Q.pt_4 + 7.454337
    if Q.n_for_90pct < 7.0:
        z += -0.04001335 * Q.n_for_90pct + 0.2800935
    if Q.max_dr < 0.1117619:
        z += -14.11475 * Q.max_dr + 1.577491
    if Q.M3 < 0.0782171:
        z += 0.07654407 * Q.M3 - 0.005987055
    if Q.absphi_1 < 0.02227783:
        z += -12.55436 * Q.absphi_1 + 0.2796839
    if Q.sum_z_dr2_top5 >= 0.01148293:
        z += -51.56377 * Q.sum_z_dr2_top5 + 0.5921029
    if Q.N2 >= 0.2233283:
        z += 1.393062 * Q.N2 - 0.3111102
    if Q.z_5 < 0.06503035:
        z += -19.70898 * Q.z_5 + 1.281682
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -385.5037 * Q.mass_over_sum_pt_sq + 2.769009
    if Q.zdr_2 < 0.002170915:
        z += -875.0853 * Q.zdr_2 + 1.899736
    if Q.D2 >= 2.055451:
        z += 0.008475154 * Q.D2 - 0.01742026
    if Q.sd_mass >= 45.595:
        z += -0.0362782 * Q.sd_mass + 1.654105
    if Q.sum_zz_dr2 < 0.0030133:
        z += 316.7609 * Q.sum_zz_dr2 - 0.9544958
    if Q.mass < 53.33237 and Q.centroid_offset < 0.02685622:
        z += 7.239899 * (53.33237 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass < 53.33237 and Q.log_sum_pt < 6.842717:
        z += 0.05613005 * (53.33237 - Q.mass) * (6.842717 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.00609665 and Q.planar_flow < 0.3220738:
        z += 1793.995 * (0.00609665 - Q.sum_z_dr2) * (0.3220738 - Q.planar_flow)
    if Q.sum_z_dr2 < 0.00609665 and Q.mean_phi < 0.002834884:
        z += -22306.87 * (0.00609665 - Q.sum_z_dr2) * (0.002834884 - Q.mean_phi)
    if Q.sum_z_dr < 0.05464922 and Q.z_5 > 0.03672711:
        z += 142.7544 * (0.05464922 - Q.sum_z_dr) * (Q.z_5 - 0.03672711)
    if Q.mass < 53.33237 and Q.D2 > 2.843757:
        z += 0.01668835 * (53.33237 - Q.mass) * (Q.D2 - 2.843757)
    if Q.lam1 > 0.005433361 and Q.eccentricity > 0.7117266:
        z += -640.3996 * (Q.lam1 - 0.005433361) * (Q.eccentricity - 0.7117266)
    if Q.sj3_pair_mass_max < 24.23013 and Q.absphi_2 > 0.007518768:
        z += -3.158597 * (24.23013 - Q.sj3_pair_mass_max) * (Q.absphi_2 - 0.007518768)
    if Q.tau1 < 0.04369778 and Q.eccentricity > 0.9031255:
        z += -914.6205 * (0.04369778 - Q.tau1) * (Q.eccentricity - 0.9031255)
    if Q.e3 < 2.371297e-05 and Q.eccentricity > 0.8319502:
        z += 51871.2 * (2.371297e-05 - Q.e3) * (Q.eccentricity - 0.8319502)
    if Q.e2 > 0.03556091 and Q.mean_eta > 0.00189657:
        z += 22.34466 * (Q.e2 - 0.03556091) * (Q.mean_eta - 0.00189657)
    if Q.sj3_dr_max < 0.1070199 and Q.abseta_1 > 0.03625488:
        z += 2417.28 * (0.1070199 - Q.sj3_dr_max) * (Q.abseta_1 - 0.03625488)
    if Q.centroid_offset < 0.01837778 and Q.abseta_1 < 0.07073975:
        z += 1213.838 * (0.01837778 - Q.centroid_offset) * (0.07073975 - Q.abseta_1)
    if Q.tau1 < 0.04369778 and Q.mean_phi < -0.01275329:
        z += -3740.997 * (0.04369778 - Q.tau1) * (-0.01275329 - Q.mean_phi)
    if Q.tau1 < 0.04369778 and Q.mean_phi > 0.02612796:
        z += -5321.553 * (0.04369778 - Q.tau1) * (Q.mean_phi - 0.02612796)
    if Q.centroid_offset < 0.01837778 and Q.tau4 > 0.001224244:
        z += 7544.616 * (0.01837778 - Q.centroid_offset) * (Q.tau4 - 0.001224244)
    if Q.sum_z_dr2 < 0.00609665 and Q.mean_phi > 0.02612796:
        z += -129641.8 * (0.00609665 - Q.sum_z_dr2) * (Q.mean_phi - 0.02612796)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.716559:
        z += -70.4418 * (Q.log_sum_pt - 6.896095) * (0.716559 - Q.D2_b2)
    if Q.centroid_offset < 0.01837778 and Q.n_for_90pct > 5.0:
        z += 15.205 * (0.01837778 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.sj3_dr_max < 0.1426152 and Q.pt_6 > 33.6875:
        z += 0.2368119 * (0.1426152 - Q.sj3_dr_max) * (Q.pt_6 - 33.6875)
    if Q.pt_4 < 31.125 and Q.dr_min_012 > 0.01488897:
        z += -6.509953 * (31.125 - Q.pt_4) * (Q.dr_min_012 - 0.01488897)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 < 159.25:
        z += -0.7127884 * (0.01837778 - Q.centroid_offset) * (159.25 - Q.pt_1)
    if Q.centroid_offset < 0.01837778 and Q.z_2nd < 0.2055511:
        z += 1859.994 * (0.01837778 - Q.centroid_offset) * (0.2055511 - Q.z_2nd)
    if Q.pt_4 < 31.125 and Q.n_pt_above_1 < 8.0:
        z += -1.863016 * (31.125 - Q.pt_4) * (8.0 - Q.n_pt_above_1)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.sj3_dr_min < 0.2089872:
        z += -11.23016 * (Q.n_dr_0p2_0p4 - 1.0) * (0.2089872 - Q.sj3_dr_min)
    if Q.sum_z_dr < 0.05464922 and Q.tau21_b2 < 0.02656143:
        z += -1263.451 * (0.05464922 - Q.sum_z_dr) * (0.02656143 - Q.tau21_b2)
    if Q.pt_4 < 31.125 and Q.M3 > 0.05210278:
        z += 37.84774 * (31.125 - Q.pt_4) * (Q.M3 - 0.05210278)
    if Q.tau1 < 0.04369778 and Q.tau21_b2 < 0.02656143:
        z += -8537.714 * (0.04369778 - Q.tau1) * (0.02656143 - Q.tau21_b2)
    if Q.sum_pt_top5 > 839.9547 and Q.tau21_b2 < 0.004811143:
        z += -14.47269 * (Q.sum_pt_top5 - 839.9547) * (0.004811143 - Q.tau21_b2)
    if Q.mass_over_sum_pt < 0.07269073 and Q.mass_top2 > 1.933087:
        z += 4.467369 * (0.07269073 - Q.mass_over_sum_pt) * (Q.mass_top2 - 1.933087)
    if Q.zdr_0 < 0.006292091 and Q.pt_2 > 88.4375:
        z += 4.931911 * (0.006292091 - Q.zdr_0) * (Q.pt_2 - 88.4375)
    if Q.e2 > 0.03556091 and Q.mean_phi > 0.009050008:
        z += 141.9383 * (Q.e2 - 0.03556091) * (Q.mean_phi - 0.009050008)
    if Q.mass_over_sum_pt < 0.07269073 and Q.phi_0 < -0.004917145:
        z += -433.8527 * (0.07269073 - Q.mass_over_sum_pt) * (-0.004917145 - Q.phi_0)
    if Q.centroid_offset < 0.01837778 and Q.mean_eta > 6.288824e-05:
        z += 15007.29 * (0.01837778 - Q.centroid_offset) * (Q.mean_eta - 6.288824e-05)
    if Q.sum_pt_top5 > 839.9547 and Q.pt1_dr01 < 1.797343:
        z += 0.005031929 * (Q.sum_pt_top5 - 839.9547) * (1.797343 - Q.pt1_dr01)
    if Q.centroid_offset < 0.01837778 and Q.pt1_dr01 > 17.82896:
        z += -11.40399 * (0.01837778 - Q.centroid_offset) * (Q.pt1_dr01 - 17.82896)
    if Q.e3 > 0.0001869378 and Q.pt_6 < 33.6875:
        z += -35.26568 * (Q.e3 - 0.0001869378) * (33.6875 - Q.pt_6)
    if Q.sum_z_dr < 0.05464922 and Q.z_6 > 0.03448406:
        z += -245.4758 * (0.05464922 - Q.sum_z_dr) * (Q.z_6 - 0.03448406)
    if Q.centroid_offset < 0.01837778 and Q.pair_mass_0_2 > 17.9037:
        z += -9.083664 * (0.01837778 - Q.centroid_offset) * (Q.pair_mass_0_2 - 17.9037)
    if Q.sum_pt < 988.4078 and Q.dr_2 < 0.01778111:
        z += 0.1712461 * (988.4078 - Q.sum_pt) * (0.01778111 - Q.dr_2)
    if Q.n_for_90pct < 7.0 and Q.dr_2 > 0.00656258:
        z += -7.534269 * (7.0 - Q.n_for_90pct) * (Q.dr_2 - 0.00656258)
    if Q.e2 > 0.04447357 and Q.sj3_mass3 < 0.2723288:
        z += -70.96596 * (Q.e2 - 0.04447357) * (0.2723288 - Q.sj3_mass3)
    if Q.lam2 > 0.001130645 and Q.dr_4 < 0.1551147:
        z += -7113.659 * (Q.lam2 - 0.001130645) * (0.1551147 - Q.dr_4)
    if Q.e3 > 0.0001869378 and Q.pt_2 > 56.5:
        z += -49.75013 * (Q.e3 - 0.0001869378) * (Q.pt_2 - 56.5)
    if Q.M3 < 0.0782171 and Q.pt_2 > 80.875:
        z += 0.522173 * (0.0782171 - Q.M3) * (Q.pt_2 - 80.875)
    if Q.tau1 < 0.04369778 and Q.zdr_2 > 0.00759156:
        z += -25329.35 * (0.04369778 - Q.tau1) * (Q.zdr_2 - 0.00759156)
    if Q.D2 > 2.055451 and Q.psi_0p3 < 1.0:
        z += -32.08059 * (Q.D2 - 2.055451) * (1.0 - Q.psi_0p3)
    if Q.log_sum_pt > 6.896095 and Q.zdr_5 < 0.001255404:
        z += 19405.84 * (Q.log_sum_pt - 6.896095) * (0.001255404 - Q.zdr_5)
    if Q.sj3_dr_max < 0.1426152 and Q.n_dr_0p05_0p1 > 5.0:
        z += 17.82986 * (0.1426152 - Q.sj3_dr_max) * (Q.n_dr_0p05_0p1 - 5.0)
    if Q.sd_mass > 45.595 and Q.n_dr_0p05_0p1 > 7.0:
        z += -0.1614286 * (Q.sd_mass - 45.595) * (Q.n_dr_0p05_0p1 - 7.0)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.423457
    z += 4.834756 * Q.e2
    if 11.051 <= Q.sj3_pair_mass_min < 15.95929:
        z += 0.04975862 * Q.sj3_pair_mass_min - 0.5498826
    if Q.sj3_pair_mass_min >= 15.95929:
        z += 0.1561506 * Q.sj3_pair_mass_min - 2.247822
    if Q.lam1 < 0.001503553:
        z += 929.1414 * Q.lam1 - 4.370039
    if 0.001503553 <= Q.lam1 < 0.004183811:
        z += 867.4548 * Q.lam1 - 4.277289
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += 366.0444 * Q.lam1 - 2.179483
    if Q.lam1 >= 0.00733008:
        z += -216.2549 * Q.lam1 + 1.585166
    if 0.0003061234 <= Q.lam2 < 0.003408389:
        z += 1123.335 * Q.lam2 - 0.343879
    if Q.lam2 >= 0.003408389:
        z += 766.5515 * Q.lam2 + 0.8721771
    if Q.pt_7 < 45.75:
        z += -0.02441691 * Q.pt_7 + 1.117074
    if Q.sj3_dr_min >= 0.1278212:
        z += 15.77883 * Q.sj3_dr_min - 2.016869
    if Q.e3 < 3.892127e-05:
        z += -3047.949 * Q.e3 + 0.2483391
    if 3.892127e-05 <= Q.e3 < 8.147744e-05:
        z += -4530.347 * Q.e3 + 0.3060359
    if Q.e3 >= 8.147744e-05:
        z += -1482.398 * Q.e3 + 0.05769683
    if Q.log_sum_pt < 6.080494:
        z += 8.057045 * Q.log_sum_pt - 48.99082
    if 22.18342 <= Q.mass_top5 < 45.32077:
        z += 0.01642151 * Q.mass_top5 - 0.3642852
    if Q.mass_top5 >= 45.32077:
        z += -0.02760742 * Q.mass_top5 + 1.63114
    if Q.zdr_0 < 0.004918231:
        z += 149.6611 * Q.zdr_0 - 0.25338
    if 0.004918231 <= Q.zdr_0 < 0.0211821:
        z += -29.67853 * Q.zdr_0 + 0.6286537
    if Q.M2 < 0.02563286:
        z += -20.5516 * Q.M2 + 0.5267963
    z += 11.21819 * Q.mean_phi
    if Q.sum_pt < 488.9312:
        z += -0.000839933 * Q.sum_pt + 0.4106695
    if Q.sum_pt >= 988.4078:
        z += 0.007659245 * Q.sum_pt - 7.570457
    if Q.z_7 < 0.06473447:
        z += 28.08529 * Q.z_7 - 1.818087
    if Q.mean_eta >= 0.02644207:
        z += 36.2611 * Q.mean_eta - 0.9588186
    if Q.LHA >= 0.3033137:
        z += -14.82088 * Q.LHA + 4.495376
    if Q.tau1 >= 0.05356915:
        z += 13.24551 * Q.tau1 - 0.7095508
    if Q.sum_z_dr2 < 0.002635418:
        z += 624.405 * Q.sum_z_dr2 - 1.645568
    if 0.007520088 <= Q.sum_z_dr2 < 0.02530566:
        z += 494.6634 * Q.sum_z_dr2 - 3.719913
    if Q.sum_z_dr2 >= 0.02530566:
        z += 229.7477 * Q.sum_z_dr2 + 2.983956
    if Q.mass < 76.6557:
        z += -0.0234972 * Q.mass + 1.801194
    if Q.phi_6 >= 0.04013062:
        z += 3.14923 * Q.phi_6 - 0.1263805
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += -678.8458 * Q.sum_z_dr2_top3 + 1.460583
    if Q.sj2_dr >= 0.3003793:
        z += -6.216743 * Q.sj2_dr + 1.867381
    if Q.sj3_dr_max >= 0.1986272:
        z += -2.882515 * Q.sj3_dr_max + 0.572546
    if Q.C2_b2 >= 0.009032972:
        z += 46.19727 * Q.C2_b2 - 0.4172987
    if Q.n_pt_above_50 >= 6.0:
        z += -0.1889632 * Q.n_pt_above_50 + 1.133779
    if Q.zdr_7 < 0.00325401:
        z += 219.439 * Q.zdr_7 - 0.7140567
    if Q.centroid_offset < 0.02355416:
        z += 48.27203 * Q.centroid_offset - 1.137007
    if Q.lam2 > 0.0003061234 and Q.sj3_pairmax_over_m > 0.6745796:
        z += 2698.845 * (Q.lam2 - 0.0003061234) * (Q.sj3_pairmax_over_m - 0.6745796)
    if Q.e3 < 8.147744e-05 and Q.sj3_dr13 > 0.181053:
        z += 46181.4 * (8.147744e-05 - Q.e3) * (Q.sj3_dr13 - 0.181053)
    if Q.sj3_pair_mass_min > 11.051 and Q.n_dr_0p2_0p4 > 0.0:
        z += 0.01276446 * (Q.sj3_pair_mass_min - 11.051) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.pt_7 < 45.75 and Q.D2 < 1.002471:
        z += 0.1418293 * (45.75 - Q.pt_7) * (1.002471 - Q.D2)
    if Q.zdr_0 < 0.0211821 and Q.dr_7 > 0.1078355:
        z += 689.9552 * (0.0211821 - Q.zdr_0) * (Q.dr_7 - 0.1078355)
    if Q.zdr_0 < 0.0211821 and Q.centroid_offset > 0.01258764:
        z += 5157.39 * (0.0211821 - Q.zdr_0) * (Q.centroid_offset - 0.01258764)
    if Q.pt_7 < 45.75 and Q.log_sum_pt < 6.572938:
        z += 0.01977199 * (45.75 - Q.pt_7) * (6.572938 - Q.log_sum_pt)
    if Q.lam2 > 0.0003061234 and Q.planar_flow > 0.04505724:
        z += -372.6227 * (Q.lam2 - 0.0003061234) * (Q.planar_flow - 0.04505724)
    if Q.sj3_dr_min > 0.1278212 and Q.z_dr_0_0p05 < 0.3658817:
        z += -11.64791 * (Q.sj3_dr_min - 0.1278212) * (0.3658817 - Q.z_dr_0_0p05)
    if Q.sj3_dr_min > 0.1278212 and Q.dr_min_012 < 0.02695109:
        z += 116.1982 * (Q.sj3_dr_min - 0.1278212) * (0.02695109 - Q.dr_min_012)
    if Q.mass_top5 > 45.32077 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.002626261 * (Q.mass_top5 - 45.32077) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.zdr_0 < 0.0211821 and Q.z_dr_0p05_0p1 < 0.2919447:
        z += 288.6168 * (0.0211821 - Q.zdr_0) * (0.2919447 - Q.z_dr_0p05_0p1)
    if Q.log_sum_pt < 6.080494 and Q.mean_eta2 > 9.030369e-05:
        z += -293.9854 * (6.080494 - Q.log_sum_pt) * (Q.mean_eta2 - 9.030369e-05)
    if Q.lam1 > 0.00733008 and Q.sj3_mass3 < 0.2723288:
        z += -54.71766 * (Q.lam1 - 0.00733008) * (0.2723288 - Q.sj3_mass3)
    if Q.e3 < 8.147744e-05 and Q.sj3_dr23 > 0.1797097:
        z += 13751.54 * (8.147744e-05 - Q.e3) * (Q.sj3_dr23 - 0.1797097)
    if Q.lam1 > 0.00733008 and Q.D2_b2 < 0.380911:
        z += -300.0836 * (Q.lam1 - 0.00733008) * (0.380911 - Q.D2_b2)
    if Q.sj3_dr_min > 0.1278212 and Q.D2_b2 < 1.911889:
        z += 0.9293667 * (Q.sj3_dr_min - 0.1278212) * (1.911889 - Q.D2_b2)
    if Q.mean_eta > 0.02644207 and Q.absphi_5 > 0.04455566:
        z += 54.1963 * (Q.mean_eta - 0.02644207) * (Q.absphi_5 - 0.04455566)
    if Q.z_7 < 0.06473447 and Q.mean_eta < -0.004664942:
        z += -777.6058 * (0.06473447 - Q.z_7) * (-0.004664942 - Q.mean_eta)
    if Q.phi_6 > 0.04013062 and Q.eta_7 < -0.1218262:
        z += 25.8039 * (Q.phi_6 - 0.04013062) * (-0.1218262 - Q.eta_7)
    if Q.sj3_dr_min > 0.1278212 and Q.sj2_mass2 < 0.03573274:
        z += -122.463 * (Q.sj3_dr_min - 0.1278212) * (0.03573274 - Q.sj2_mass2)
    if Q.LHA > 0.3033137 and Q.sj3_mass1 > 5.294926:
        z += 5.349902 * (Q.LHA - 0.3033137) * (Q.sj3_mass1 - 5.294926)
    if Q.LHA > 0.3033137 and Q.sj3_pairmax_over_m > 0.901705:
        z += 48.08276 * (Q.LHA - 0.3033137) * (Q.sj3_pairmax_over_m - 0.901705)
    if Q.lam1 > 0.00733008 and Q.D2_b2 > 1.129616:
        z += -60.11076 * (Q.lam1 - 0.00733008) * (Q.D2_b2 - 1.129616)
    if Q.sum_z_dr2 > 0.007520088 and Q.dr1_4 < 0.003088708:
        z += 31179.15 * (Q.sum_z_dr2 - 0.007520088) * (0.003088708 - Q.dr1_4)
    if Q.pt_7 < 45.75 and Q.tau4 > 0.007583927:
        z += -1.190089 * (45.75 - Q.pt_7) * (Q.tau4 - 0.007583927)
    if Q.log_sum_pt < 6.080494 and Q.absphi_7 > 0.03546143:
        z += -28.14088 * (6.080494 - Q.log_sum_pt) * (Q.absphi_7 - 0.03546143)
    if Q.tau1 > 0.05356915 and Q.eta_7 > -0.0803833:
        z += 19.1248 * (Q.tau1 - 0.05356915) * (Q.eta_7 - -0.0803833)
    if Q.pt_7 < 45.75 and Q.abseta_3 > 0.0013237:
        z += 0.2171747 * (45.75 - Q.pt_7) * (Q.abseta_3 - 0.0013237)
    if Q.sj3_dr_min > 0.1278212 and Q.psi_0p3 < 1.0:
        z += 40.28313 * (Q.sj3_dr_min - 0.1278212) * (1.0 - Q.psi_0p3)
    if Q.lam1 > 0.00733008 and Q.dr_7 < 0.03607145:
        z += 11317.62 * (Q.lam1 - 0.00733008) * (0.03607145 - Q.dr_7)
    if Q.mass < 76.6557 and Q.phi_4 < -0.0496521:
        z += 0.03964281 * (76.6557 - Q.mass) * (-0.0496521 - Q.phi_4)
    if Q.sj3_pair_mass_min > 11.051 and Q.sj3_pairmin_over_m > 0.28737:
        z += -0.1340799 * (Q.sj3_pair_mass_min - 11.051) * (Q.sj3_pairmin_over_m - 0.28737)
    if Q.sj3_dr_min > 0.1278212 and Q.sj3_mass2 < 0.5973755:
        z += -4.580946 * (Q.sj3_dr_min - 0.1278212) * (0.5973755 - Q.sj3_mass2)
    if Q.e3 > 3.892127e-05 and Q.zdr_7 > 0.00279494:
        z += 81212.72 * (Q.e3 - 3.892127e-05) * (Q.zdr_7 - 0.00279494)
    if Q.mass < 76.6557 and Q.D3 > 1.173493:
        z += 0.002580382 * (76.6557 - Q.mass) * (Q.D3 - 1.173493)
    if Q.lam1 < 0.00595415 and Q.z_dr_0p05_0p1 > 0.2919447:
        z += -1431.537 * (0.00595415 - Q.lam1) * (Q.z_dr_0p05_0p1 - 0.2919447)
    if Q.zdr_0 < 0.0211821 and Q.phi_5 > 2.199411e-05:
        z += 210.9704 * (0.0211821 - Q.zdr_0) * (Q.phi_5 - 2.199411e-05)
    if Q.lam2 > 0.0003061234 and Q.pt_6 < 31.90625:
        z += -69.12009 * (Q.lam2 - 0.0003061234) * (31.90625 - Q.pt_6)
    if Q.lam1 < 0.00595415 and Q.n_pt_above_50 > 6.0:
        z += 113.6808 * (0.00595415 - Q.lam1) * (Q.n_pt_above_50 - 6.0)
    if Q.mass_top5 > 45.32077 and Q.eta_3 > -0.00592804:
        z += -0.08113807 * (Q.mass_top5 - 45.32077) * (Q.eta_3 - -0.00592804)
    if Q.tau1 > 0.05356915 and Q.D2 < 1.002471:
        z += 19.13095 * (Q.tau1 - 0.05356915) * (1.002471 - Q.D2)
    if Q.sum_z_dr2 > 0.02530566 and Q.tau43 > 0.7296607:
        z += -2003.984 * (Q.sum_z_dr2 - 0.02530566) * (Q.tau43 - 0.7296607)
    if Q.lam2 > 0.0003061234 and Q.mratio_max_012 > 0.8183982:
        z += 5638.066 * (Q.lam2 - 0.0003061234) * (Q.mratio_max_012 - 0.8183982)
    if Q.e3 > 3.892127e-05 and Q.abseta_7 > 0.05718994:
        z += -2254.095 * (Q.e3 - 3.892127e-05) * (Q.abseta_7 - 0.05718994)
    if Q.sum_pt > 988.4078 and Q.dr1_6 > 0.07796252:
        z += -0.168971 * (Q.sum_pt - 988.4078) * (Q.dr1_6 - 0.07796252)
    return max(0.0, z)


def neuron_11(Q):
    z = -1.330738
    if Q.planar_flow < 0.2534037:
        z += -6.92055 * Q.planar_flow + 1.753693
    if 0.1778793 <= Q.sj2_dr < 0.2687922:
        z += -7.655884 * Q.sj2_dr + 1.361823
    if Q.sj2_dr >= 0.2687922:
        z += 4.05093 * Q.sj2_dr - 1.784876
    if Q.mass < 15.45403:
        z += -0.02856342 * Q.mass - 1.043359
    if 15.45403 <= Q.mass < 49.6681:
        z += 0.02983844 * Q.mass - 1.945904
    if 49.6681 <= Q.mass < 69.61135:
        z += 0.02326024 * Q.mass - 1.619177
    if Q.sum_z_dr2 < 0.004372139:
        z += -10.13663 * Q.sum_z_dr2 + 3.555844
    if 0.004372139 <= Q.sum_z_dr2 < 0.01323868:
        z += -396.0425 * Q.sum_z_dr2 + 5.243078
    if Q.tau1 < 0.09538712:
        z += 19.11884 * Q.tau1 - 1.823691
    if Q.LHA < 0.3127275:
        z += -0.2416791 * Q.LHA + 0.0755797
    if Q.centroid_offset < 0.01437952:
        z += -104.2241 * Q.centroid_offset + 3.418696
    if 0.01437952 <= Q.centroid_offset < 0.03776099:
        z += -82.11646 * Q.centroid_offset + 3.100799
    if Q.centroid_offset >= 0.04990367:
        z += -3.040012 * Q.centroid_offset + 0.1517078
    if Q.lam1 < 0.0005049491:
        z += 2692.543 * Q.lam1 - 3.807302
    if 0.0005049491 <= Q.lam1 < 0.00733008:
        z += 326.6598 * Q.lam1 - 2.612651
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 208.7136 * Q.lam1 - 1.748096
    if Q.sj3_dr_max < 0.1426152:
        z += 2.982489 * Q.sj3_dr_max - 0.08604407
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 24.36194 * Q.sj3_dr_max - 3.135078
    if 0.169029 <= Q.sj3_dr_max < 0.2623172:
        z += -10.53506 * Q.sj3_dr_max + 2.763528
    if Q.lam1_plus_lam2 < 0.006679471:
        z += -1625.922 * Q.lam1_plus_lam2 + 13.40137
    if 0.006679471 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -1271.44 * Q.lam1_plus_lam2 + 11.03362
    if Q.sum_z_dr < 0.02054282:
        z += 240.9478 * Q.sum_z_dr - 8.579815
    if 0.02054282 <= Q.sum_z_dr < 0.02689598:
        z += 145.1185 * Q.sum_z_dr - 6.61121
    if 0.02689598 <= Q.sum_z_dr < 0.0717028:
        z += 60.4396 * Q.sum_z_dr - 4.333689
    if Q.sum_zz_dr2 < 0.008168571:
        z += 3353.175 * Q.sum_zz_dr2 - 27.39064
    if Q.z_7 >= 0.01685855:
        z += 32.13374 * Q.z_7 - 0.5417281
    if Q.z_dr_0p05_0p1 >= 0.8460335:
        z += 7.31351 * Q.z_dr_0p05_0p1 - 6.187475
    if Q.pt_6 < 24.42188:
        z += -0.0146243 * Q.pt_6 + 0.4868203
    if 24.42188 <= Q.pt_6 < 41.21875:
        z += -0.007719741 * Q.pt_6 + 0.3181981
    if Q.max_pair_mass >= 33.3761:
        z += -0.04655544 * Q.max_pair_mass + 1.553839
    if Q.pt_7 >= 29.04219:
        z += 0.0007169449 * Q.pt_7 - 0.02082165
    if Q.eccentricity >= 0.9884745:
        z += 73.24588 * Q.eccentricity - 72.40169
    if Q.C2 < 0.03578649:
        z += 6.174473 * Q.C2 - 0.2209627
    if Q.max_dr < 0.1027585:
        z += 1.184625 * Q.max_dr + 0.2919205
    if 0.1027585 <= Q.max_dr < 0.1452311:
        z += -9.739242 * Q.max_dr + 1.414441
    if Q.e3 < 1.050302e-05:
        z += -47120.94 * Q.e3 + 0.4949122
    if Q.sum_pt_top5 < 506.875:
        z += 0.003678573 * Q.sum_pt_top5 - 1.36627
    if 506.875 <= Q.sum_pt_top5 < 791.125:
        z += -0.00175306 * Q.sum_pt_top5 + 1.386889
    if Q.sum_pt >= 988.4078:
        z += -0.01180494 * Q.sum_pt + 11.6681
    if Q.mass_over_sum_pt < 0.07637363:
        z += 42.15914 * Q.mass_over_sum_pt - 3.219847
    if Q.e2 < 0.01655442:
        z += -53.94333 * Q.e2 + 0.8930005
    if Q.zdr_0 < 0.003676313:
        z += 459.5209 * Q.zdr_0 - 1.689343
    if Q.z_3rd < 0.07368057:
        z += -24.61354 * Q.z_3rd + 1.813539
    if Q.lam2 < 0.001130645:
        z += -830.5521 * Q.lam2 + 0.9390594
    if Q.mass_over_sum_pt_sq < 0.00817466:
        z += -3085.046 * Q.mass_over_sum_pt_sq + 25.21921
    if Q.n_dr_0p1_0p2 >= 3.0:
        z += -0.6301085 * Q.n_dr_0p1_0p2 + 1.890325
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.007822361 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.pt_7 < 37.15625:
        z += -0.1047342 * (0.2534037 - Q.planar_flow) * (37.15625 - Q.pt_7)
    if Q.tau1 < 0.09538712 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += 11.20583 * (0.09538712 - Q.tau1) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.planar_flow < 0.2534037 and Q.e3 < 1.050302e-05:
        z += -1123219.0 * (0.2534037 - Q.planar_flow) * (1.050302e-05 - Q.e3)
    if Q.sj2_dr > 0.1778793 and Q.lam2 < 0.001130645:
        z += -14540.1 * (Q.sj2_dr - 0.1778793) * (0.001130645 - Q.lam2)
    if Q.mass < 49.6681 and Q.D2 < 1.332146:
        z += -0.09256538 * (49.6681 - Q.mass) * (1.332146 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.pt_6 > 19.46875:
        z += 1.632828 * (0.01323868 - Q.sum_z_dr2) * (Q.pt_6 - 19.46875)
    if Q.centroid_offset < 0.03776099 and Q.sum_pt < 901.5938:
        z += -0.09608138 * (0.03776099 - Q.centroid_offset) * (901.5938 - Q.sum_pt)
    if Q.centroid_offset < 0.03776099 and Q.mean_phi < -0.001813533:
        z += 323.2227 * (0.03776099 - Q.centroid_offset) * (-0.001813533 - Q.mean_phi)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += -71.62124 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.D2 < 0.415246:
        z += 496.9812 * (0.01323868 - Q.sum_z_dr2) * (0.415246 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.pt_2 < 61.53125:
        z += -3.212586 * (0.01323868 - Q.sum_z_dr2) * (61.53125 - Q.pt_2)
    if Q.centroid_offset < 0.03776099 and Q.absphi_0 < 0.0345459:
        z += -299.1574 * (0.03776099 - Q.centroid_offset) * (0.0345459 - Q.absphi_0)
    if Q.mass < 49.6681 and Q.dr1_7 > 0.1792439:
        z += -0.3035093 * (49.6681 - Q.mass) * (Q.dr1_7 - 0.1792439)
    if Q.mass < 15.45403 and Q.phi_1 < -0.05890198:
        z += -34.84113 * (15.45403 - Q.mass) * (-0.05890198 - Q.phi_1)
    if Q.mass < 69.61135 and Q.D2 < 0.415246:
        z += -0.0433033 * (69.61135 - Q.mass) * (0.415246 - Q.D2)
    if Q.sj2_dr > 0.1778793 and Q.n_pt_above_50 > 4.0:
        z += -3.520272 * (Q.sj2_dr - 0.1778793) * (Q.n_pt_above_50 - 4.0)
    if Q.centroid_offset < 0.01437952 and Q.phi_1 > 0.002190304:
        z += 811.0866 * (0.01437952 - Q.centroid_offset) * (Q.phi_1 - 0.002190304)
    if Q.centroid_offset < 0.03776099 and Q.dr_max_012 > 0.1828389:
        z += -27.84129 * (0.03776099 - Q.centroid_offset) * (Q.dr_max_012 - 0.1828389)
    if Q.centroid_offset < 0.01437952 and Q.eta_1 < -0.005273438:
        z += 359.5289 * (0.01437952 - Q.centroid_offset) * (-0.005273438 - Q.eta_1)
    if Q.centroid_offset < 0.01437952 and Q.abseta_4 < 0.03601074:
        z += -4071.183 * (0.01437952 - Q.centroid_offset) * (0.03601074 - Q.abseta_4)
    if Q.planar_flow < 0.2534037 and Q.abseta_4 < 0.04977417:
        z += 75.70347 * (0.2534037 - Q.planar_flow) * (0.04977417 - Q.abseta_4)
    if Q.centroid_offset < 0.01437952 and Q.D2_b2 < 0.5327104:
        z += 371.564 * (0.01437952 - Q.centroid_offset) * (0.5327104 - Q.D2_b2)
    if Q.pt_6 < 41.21875 and Q.D2_b2 < 4.721224:
        z += -0.005700139 * (41.21875 - Q.pt_6) * (4.721224 - Q.D2_b2)
    if Q.centroid_offset > 0.04990367 and Q.tau32 < 0.5502779:
        z += -435.4219 * (Q.centroid_offset - 0.04990367) * (0.5502779 - Q.tau32)
    if Q.centroid_offset < 0.01437952 and Q.tau32 < 0.5186963:
        z += 189.7752 * (0.01437952 - Q.centroid_offset) * (0.5186963 - Q.tau32)
    if Q.sum_zz_dr2 < 0.008168571 and Q.mass_top2 > 6.779915:
        z += 7.228589 * (0.008168571 - Q.sum_zz_dr2) * (Q.mass_top2 - 6.779915)
    if Q.eccentricity > 0.9884745 and Q.sj2_mass2 < 2.771069:
        z += -25.44768 * (Q.eccentricity - 0.9884745) * (2.771069 - Q.sj2_mass2)
    if Q.eccentricity > 0.9884745 and Q.eta_0 < 0.02130127:
        z += -153.2787 * (Q.eccentricity - 0.9884745) * (0.02130127 - Q.eta_0)
    if Q.max_pair_mass > 33.3761 and Q.sj2_mass1 < 4.966909:
        z += -0.04131666 * (Q.max_pair_mass - 33.3761) * (4.966909 - Q.sj2_mass1)
    if Q.sum_pt > 988.4078 and Q.phi_0 < -0.07818909:
        z += 0.5500832 * (Q.sum_pt - 988.4078) * (-0.07818909 - Q.phi_0)
    if Q.mass < 15.45403 and Q.phi_0 < -0.004917145:
        z += -8.542347 * (15.45403 - Q.mass) * (-0.004917145 - Q.phi_0)
    if Q.centroid_offset < 0.01437952 and Q.zdr_7 < 0.00279494:
        z += -54283.14 * (0.01437952 - Q.centroid_offset) * (0.00279494 - Q.zdr_7)
    if Q.mass < 15.45403 and Q.zdr_7 < 0.003780225:
        z += 40.46997 * (15.45403 - Q.mass) * (0.003780225 - Q.zdr_7)
    if Q.sj2_dr > 0.1778793 and Q.dr_7 < 0.04881348:
        z += 353.929 * (Q.sj2_dr - 0.1778793) * (0.04881348 - Q.dr_7)
    if Q.sj3_dr_max < 0.2623172 and Q.phi_0 < -0.0216713:
        z += 206.3054 * (0.2623172 - Q.sj3_dr_max) * (-0.0216713 - Q.phi_0)
    if Q.centroid_offset < 0.01437952 and Q.sj3_pair_mass_min < 15.95929:
        z += -14.21438 * (0.01437952 - Q.centroid_offset) * (15.95929 - Q.sj3_pair_mass_min)
    if Q.sum_z_dr < 0.02689598 and Q.sj3_pair_mass_min > 1.590484:
        z += -35.41936 * (0.02689598 - Q.sum_z_dr) * (Q.sj3_pair_mass_min - 1.590484)
    if Q.mass < 69.61135 and Q.tau32 < 0.5502779:
        z += 0.02811383 * (69.61135 - Q.mass) * (0.5502779 - Q.tau32)
    if Q.planar_flow < 0.2534037 and Q.sj3_pair_mass_min > 1.282345:
        z += 0.05450873 * (0.2534037 - Q.planar_flow) * (Q.sj3_pair_mass_min - 1.282345)
    if Q.pt_6 < 24.42188 and Q.mratio_min_012 < 0.110654:
        z += 0.5659294 * (24.42188 - Q.pt_6) * (0.110654 - Q.mratio_min_012)
    if Q.sum_pt > 988.4078 and Q.dr_min_012 > 0.01488897:
        z += 3.102604 * (Q.sum_pt - 988.4078) * (Q.dr_min_012 - 0.01488897)
    if Q.sj3_dr_max < 0.169029 and Q.pair_mass_0_4 > 4.619568:
        z += -0.1399489 * (0.169029 - Q.sj3_dr_max) * (Q.pair_mass_0_4 - 4.619568)
    if Q.mass < 15.45403 and Q.abseta_5 > 0.06137085:
        z += -57.06353 * (15.45403 - Q.mass) * (Q.abseta_5 - 0.06137085)
    if Q.sj3_dr_max < 0.1426152 and Q.phi_2 > 0.06414795:
        z += -2082.121 * (0.1426152 - Q.sj3_dr_max) * (Q.phi_2 - 0.06414795)
    if Q.sj3_dr_max < 0.2623172 and Q.pair_mass_0_2 > 40.2:
        z += 0.488539 * (0.2623172 - Q.sj3_dr_max) * (Q.pair_mass_0_2 - 40.2)
    if Q.mass < 69.61135 and Q.pair_mass_0_2 > 11.71328:
        z += 0.001258676 * (69.61135 - Q.mass) * (Q.pair_mass_0_2 - 11.71328)
    if Q.centroid_offset > 0.04990367 and Q.zdr_6 > 0.007390416:
        z += -8306.111 * (Q.centroid_offset - 0.04990367) * (Q.zdr_6 - 0.007390416)
    if Q.lam2 < 0.001130645 and Q.dr_min_012 > 0.02695109:
        z += 250.7784 * (0.001130645 - Q.lam2) * (Q.dr_min_012 - 0.02695109)
    if Q.z_dr_0p05_0p1 > 0.8460335 and Q.pt1_over_pt0 < 0.3593346:
        z += 52.81746 * (Q.z_dr_0p05_0p1 - 0.8460335) * (0.3593346 - Q.pt1_over_pt0)
    if Q.sum_pt_top5 < 506.875 and Q.z_dr_0p1_0p2 > 0.04510668:
        z += 0.01394451 * (506.875 - Q.sum_pt_top5) * (Q.z_dr_0p1_0p2 - 0.04510668)
    if Q.n_dr_0p1_0p2 > 3.0 and Q.z_1 < 0.113617:
        z += 6.53155 * (Q.n_dr_0p1_0p2 - 3.0) * (0.113617 - Q.z_1)
    if Q.centroid_offset > 0.04990367 and Q.tau3 > 0.001996306:
        z += -8424.91 * (Q.centroid_offset - 0.04990367) * (Q.tau3 - 0.001996306)
    if Q.sum_z_dr2 < 0.004372139 and Q.sj3_dr13 > 0.1659434:
        z += 13547.42 * (0.004372139 - Q.sum_z_dr2) * (Q.sj3_dr13 - 0.1659434)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.6059456
    if Q.sum_z_dr2 >= 0.01882765:
        z += 304.7156 * Q.sum_z_dr2 - 5.73708
    if 69.61135 <= Q.mass < 91.19:
        z += -0.009143366 * Q.mass + 0.6364821
    if Q.mass >= 91.19:
        z += 0.0767323 * Q.mass - 7.19452
    if Q.mass_over_sum_pt >= 0.1309286:
        z += -13.59047 * Q.mass_over_sum_pt + 1.779381
    if Q.zdr_0 >= 0.03981924:
        z += -66.7996 * Q.zdr_0 + 2.65991
    if Q.sum_z_dr2_top2 >= 0.01403324:
        z += -7.578138 * Q.sum_z_dr2_top2 + 0.1063458
    if Q.mean_phi >= 0.02612796:
        z += 14.36282 * Q.mean_phi - 0.3752712
    if Q.e2 >= 0.06344108:
        z += -39.86412 * Q.e2 + 2.529022
    if Q.centroid_offset >= 0.04990367:
        z += 29.05184 * Q.centroid_offset - 1.449793
    if Q.sum_z_dr2 > 0.01882765 and Q.lam2 > 0.000537286:
        z += -13022.33 * (Q.sum_z_dr2 - 0.01882765) * (Q.lam2 - 0.000537286)
    if Q.mass > 91.19 and Q.mass_top3 < 50.3522:
        z += 0.0003337248 * (Q.mass - 91.19) * (50.3522 - Q.mass_top3)
    if Q.mass > 91.19 and Q.centroid_offset > 0.02076709:
        z += -1.732064 * (Q.mass - 91.19) * (Q.centroid_offset - 0.02076709)
    if Q.mass > 91.19 and Q.tau4 < 0.002207727:
        z += 24.43127 * (Q.mass - 91.19) * (0.002207727 - Q.tau4)
    if Q.mass > 91.19 and Q.n_dr_0p2_0p4 > 2.0:
        z += -0.002980644 * (Q.mass - 91.19) * (Q.n_dr_0p2_0p4 - 2.0)
    if Q.sum_z_dr2 > 0.01882765 and Q.pt_7 < 53.4375:
        z += -3.283944 * (Q.sum_z_dr2 - 0.01882765) * (53.4375 - Q.pt_7)
    if Q.mass > 69.61135 and Q.centroid_offset > 0.009480685:
        z += 1.677523 * (Q.mass - 69.61135) * (Q.centroid_offset - 0.009480685)
    if Q.mass > 69.61135 and Q.abseta_5 > 0.1522217:
        z += -0.1048678 * (Q.mass - 69.61135) * (Q.abseta_5 - 0.1522217)
    if Q.mass > 69.61135 and Q.dr_4 > 0.02477348:
        z += 0.03097376 * (Q.mass - 69.61135) * (Q.dr_4 - 0.02477348)
    if Q.mass > 91.19 and Q.eta_5 > 0.007119751:
        z += -0.09097905 * (Q.mass - 91.19) * (Q.eta_5 - 0.007119751)
    if Q.sum_z_dr2 > 0.01882765 and Q.abseta_6 > 0.01612854:
        z += 558.9904 * (Q.sum_z_dr2 - 0.01882765) * (Q.abseta_6 - 0.01612854)
    if Q.mass > 69.61135 and Q.log_sum_pt < 6.423044:
        z += 0.1419387 * (Q.mass - 69.61135) * (6.423044 - Q.log_sum_pt)
    if Q.mass > 91.19 and Q.abseta_1 < 0.117984:
        z += 0.05298289 * (Q.mass - 91.19) * (0.117984 - Q.abseta_1)
    if Q.sum_z_dr2_top2 > 0.01403324 and Q.mean_phi < -0.006701703:
        z += -1096.967 * (Q.sum_z_dr2_top2 - 0.01403324) * (-0.006701703 - Q.mean_phi)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.1467648
    if Q.sum_z_dr < 0.007673833:
        z += 540.716 * Q.sum_z_dr + 4.405394
    if 0.007673833 <= Q.sum_z_dr < 0.1484084:
        z += -60.78647 * Q.sum_z_dr + 9.021224
    if Q.lam1 < 0.006506576:
        z += -467.6878 * Q.lam1 + 4.244165
    if 0.006506576 <= Q.lam1 < 0.01643375:
        z += -120.993 * Q.lam1 + 1.988369
    if 658.125 <= Q.sum_pt_top5 < 791.125:
        z += 0.003339949 * Q.sum_pt_top5 - 2.198104
    if Q.sum_pt_top5 >= 791.125:
        z += 0.01710603 * Q.sum_pt_top5 - 13.08879
    if Q.e3 < 5.334511e-05:
        z += -10705.3 * Q.e3 + 0.5710754
    if Q.pt_6 < 31.90625:
        z += 0.1198405 * Q.pt_6 - 3.823659
    if Q.sum_pt >= 988.4078:
        z += 0.01582064 * Q.sum_pt - 15.63724
    if Q.e2 < 0.08000524:
        z += 31.73418 * Q.e2 - 2.538901
    if Q.mass >= 49.6681:
        z += -0.04134501 * Q.mass + 2.053528
    if Q.z_5 < 0.02818362:
        z += 166.8593 * Q.z_5 - 4.7027
    if Q.z_7 < 0.02807091:
        z += 52.56506 * Q.z_7 - 1.475549
    if Q.sj3_dr23 >= 0.1974628:
        z += -5.796618 * Q.sj3_dr23 + 1.144616
    if Q.log_sum_pt >= 6.502799:
        z += 3.722357 * Q.log_sum_pt - 24.20574
    if Q.pt_7 >= 48.71875:
        z += -0.09586857 * Q.pt_7 + 4.670597
    if Q.sum_z_dr2_top2 < 0.0005124533:
        z += 2390.576 * Q.sum_z_dr2_top2 - 1.225059
    if Q.lam1_plus_lam2 < 0.007520088:
        z += 181.0437 * Q.lam1_plus_lam2 - 0.3403829
    if 0.007520088 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -178.555 * Q.lam1_plus_lam2 + 2.363831
    if Q.lam2 < 0.0001947983:
        z += -1066.35 * Q.lam2 + 0.2077232
    if Q.z_6 < 0.02160287:
        z += -166.6564 * Q.z_6 + 3.600257
    if Q.sum_z_dr < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -61.12756 * (0.1484084 - Q.sum_z_dr) * (6.804164 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.1484084 and Q.pt_7 < 38.53125:
        z += -1.426377 * (0.1484084 - Q.sum_z_dr) * (38.53125 - Q.pt_7)
    if Q.sum_pt_top5 > 658.125 and Q.z_7 > 0.02320757:
        z += 0.004346302 * (Q.sum_pt_top5 - 658.125) * (Q.z_7 - 0.02320757)
    if Q.e3 < 5.334511e-05 and Q.centroid_offset < 0.03776099:
        z += -787623.9 * (5.334511e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 40.04062:
        z += 0.0006586725 * (Q.sum_pt_top5 - 658.125) * (40.04062 - Q.pt_7)
    if Q.pt_6 < 31.90625 and Q.z_7 < 0.0586137:
        z += -4.491453 * (31.90625 - Q.pt_6) * (0.0586137 - Q.z_7)
    if Q.sum_z_dr < 0.1484084 and Q.z_7 > 0.06164517:
        z += 478.7627 * (0.1484084 - Q.sum_z_dr) * (Q.z_7 - 0.06164517)
    if Q.sum_z_dr < 0.1484084 and Q.lam2 < 0.000537286:
        z += -2401.278 * (0.1484084 - Q.sum_z_dr) * (0.000537286 - Q.lam2)
    if Q.sum_z_dr < 0.1484084 and Q.sj2_mass1 > 31.78116:
        z += -5.172176 * (0.1484084 - Q.sum_z_dr) * (Q.sj2_mass1 - 31.78116)
    if Q.sum_pt > 988.4078 and Q.M3 < 0.08151794:
        z += 0.428294 * (Q.sum_pt - 988.4078) * (0.08151794 - Q.M3)
    if Q.sum_z_dr < 0.1484084 and Q.M3 < 0.07474969:
        z += 277.9093 * (0.1484084 - Q.sum_z_dr) * (0.07474969 - Q.M3)
    if Q.sum_z_dr < 0.1484084 and Q.mean_phi < -9.311297e-05:
        z += -188.6935 * (0.1484084 - Q.sum_z_dr) * (-9.311297e-05 - Q.mean_phi)
    if Q.pt_7 > 48.71875 and Q.pt_2 > 126.75:
        z += 0.006269943 * (Q.pt_7 - 48.71875) * (Q.pt_2 - 126.75)
    if Q.e3 < 5.334511e-05 and Q.D3 > 0.2213841:
        z += 3574.572 * (5.334511e-05 - Q.e3) * (Q.D3 - 0.2213841)
    if Q.sum_z_dr < 0.1484084 and Q.tau2 > 0.008780509:
        z += 902.9124 * (0.1484084 - Q.sum_z_dr) * (Q.tau2 - 0.008780509)
    if Q.sum_pt_top5 > 791.125 and Q.pt_3 < 74.5625:
        z += 4.371394e-05 * (Q.sum_pt_top5 - 791.125) * (74.5625 - Q.pt_3)
    if Q.mass > 49.6681 and Q.z_dr_0p1_0p2 < 0.4684459:
        z += -0.01591128 * (Q.mass - 49.6681) * (0.4684459 - Q.z_dr_0p1_0p2)
    if Q.log_sum_pt > 6.502799 and Q.absphi_6 < 0.04681396:
        z += 36.92723 * (Q.log_sum_pt - 6.502799) * (0.04681396 - Q.absphi_6)
    if Q.log_sum_pt > 6.502799 and Q.zdr_7 > 0.0007928864:
        z += -78.1607 * (Q.log_sum_pt - 6.502799) * (Q.zdr_7 - 0.0007928864)
    if Q.log_sum_pt > 6.502799 and Q.pair_mass_0_7 > 10.2219:
        z += 0.1910298 * (Q.log_sum_pt - 6.502799) * (Q.pair_mass_0_7 - 10.2219)
    if Q.sum_pt_top5 > 791.125 and Q.pt_6 < 31.90625:
        z += 0.0009304112 * (Q.sum_pt_top5 - 791.125) * (31.90625 - Q.pt_6)
    if Q.sum_pt > 988.4078 and Q.pt_6 < 62.25:
        z += -0.0003880438 * (Q.sum_pt - 988.4078) * (62.25 - Q.pt_6)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.9065652
    if Q.planar_flow < 0.08366273:
        z += -10.33499 * Q.planar_flow + 1.104931
    if 0.08366273 <= Q.planar_flow < 0.1115136:
        z += -8.627275 * Q.planar_flow + 0.9620582
    if Q.z_dr_0p05_0p1 < 0.163898:
        z += 1.398059 * Q.z_dr_0p05_0p1 - 0.2605005
    if 0.163898 <= Q.z_dr_0p05_0p1 < 0.5882598:
        z += 0.07390262 * Q.z_dr_0p05_0p1 - 0.04347394
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += 1.242701 * Q.z_dr_0p05_0p1 - 0.933156
    if 0.7143804 <= Q.psi_0p1 < 0.8155839:
        z += 0.194778 * Q.psi_0p1 - 0.1391456
    if 0.8155839 <= Q.psi_0p1 < 0.9761279:
        z += -1.543929 * Q.psi_0p1 + 1.278916
    if Q.psi_0p1 >= 0.9761279:
        z += -28.208 * Q.psi_0p1 + 27.30646
    if 0.0009641429 <= Q.sum_z_dr2 < 0.007520088:
        z += -192.6744 * Q.sum_z_dr2 + 0.1857656
    if 0.007520088 <= Q.sum_z_dr2 < 0.008678045:
        z += -964.6819 * Q.sum_z_dr2 + 5.99133
    if Q.sum_z_dr2 >= 0.008678045:
        z += -125.8241 * Q.sum_z_dr2 - 1.288315
    if 0.002074109 <= Q.sum_zz_dr2 < 0.0030133:
        z += 399.0025 * Q.sum_zz_dr2 - 0.8275748
    if 0.0030133 <= Q.sum_zz_dr2 < 0.01165737:
        z += 118.558 * Q.sum_zz_dr2 + 0.01748865
    if Q.sum_zz_dr2 >= 0.01165737:
        z += 3.660248 * Q.sum_zz_dr2 + 1.356895
    if 0.003562611 <= Q.lam1_plus_lam2 < 0.005590289:
        z += 421.7858 * Q.lam1_plus_lam2 - 1.502659
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.006679471:
        z += 633.5221 * Q.lam1_plus_lam2 - 2.686326
    if 0.006679471 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -365.852 * Q.lam1_plus_lam2 + 3.988965
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += 82.23389 * Q.lam1_plus_lam2 - 1.943098
    if 0.01655442 <= Q.e2 < 0.03556091:
        z += -49.385 * Q.e2 + 0.81754
    if 0.03556091 <= Q.e2 < 0.04110972:
        z += -37.6954 * Q.e2 + 0.4018473
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += 88.60578 * Q.e2 - 4.790359
    if Q.e2 >= 0.05028464:
        z += 10.26409 * Q.e2 - 0.8509747
    if 0.02685622 <= Q.centroid_offset < 0.03776099:
        z += 8.719241 * Q.centroid_offset - 0.2341659
    if 0.03776099 <= Q.centroid_offset < 0.04990367:
        z += -11.522 * Q.centroid_offset + 0.5301633
    if Q.centroid_offset >= 0.04990367:
        z += -130.9092 * Q.centroid_offset + 6.488021
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 102.259 * Q.mass_over_sum_pt - 8.172918
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 147.5327 * Q.mass_over_sum_pt - 12.00994
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -66.94883 * Q.mass_over_sum_pt + 7.382155
    if Q.n_dr_0_0p05 < 5.0:
        z += -0.1875915 * Q.n_dr_0_0p05 + 0.9379575
    if Q.e3 < 5.334511e-05:
        z += 16076.59 * Q.e3 - 1.095709
    if 5.334511e-05 <= Q.e3 < 8.147744e-05:
        z += 8463.603 * Q.e3 - 0.6895927
    if Q.sd_mass < 49.91626:
        z += -0.01453271 * Q.sd_mass + 0.4125879
    if 49.91626 <= Q.sd_mass < 74.57663:
        z += 0.01268557 * Q.sd_mass - 0.9460467
    if 0.02689598 <= Q.sum_z_dr < 0.04081947:
        z += 37.56975 * Q.sum_z_dr - 1.010475
    if 0.04081947 <= Q.sum_z_dr < 0.08065885:
        z += 55.86591 * Q.sum_z_dr - 1.757315
    if 0.08065885 <= Q.sum_z_dr < 0.08723651:
        z += 77.26647 * Q.sum_z_dr - 3.48346
    if 0.08723651 <= Q.sum_z_dr < 0.1019409:
        z += -85.67136 * Q.sum_z_dr + 10.73067
    if Q.sum_z_dr >= 0.1019409:
        z += -96.50556 * Q.sum_z_dr + 11.83512
    if Q.mass >= 69.61135:
        z += -0.01702342 * Q.mass + 1.185024
    if Q.mass_top5 >= 49.18618:
        z += -0.002101899 * Q.mass_top5 + 0.1033844
    if 0.06154135 <= Q.sj2_dr < 0.1294903:
        z += -1.239462 * Q.sj2_dr + 0.07627815
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -9.393233 * Q.sj2_dr + 1.132112
    if 0.1591713 <= Q.sj2_dr < 0.1778793:
        z += 25.64752 * Q.sj2_dr - 4.44537
    if 0.1778793 <= Q.sj2_dr < 0.2001708:
        z += 20.99354 * Q.sj2_dr - 3.617524
    if Q.sj2_dr >= 0.2001708:
        z += 12.11253 * Q.sj2_dr - 1.839804
    if Q.lam2 < 0.000537286:
        z += -58.04928 * Q.lam2 + 0.03118906
    if Q.C2_b2 < 0.004032342:
        z += 62.07456 * Q.C2_b2 - 0.2503059
    if Q.LHA >= 0.3033137:
        z += 30.06776 * Q.LHA - 9.119966
    if Q.sum_pt < 715.4688:
        z += 0.004596244 * Q.sum_pt - 3.288469
    if Q.sd_rg < 0.2330919:
        z += 3.043436 * Q.sd_rg - 0.9880393
    if 0.2330919 <= Q.sd_rg < 0.2787955:
        z += -15.61951 * Q.sd_rg + 3.362143
    if 0.2787955 <= Q.sd_rg < 0.324646:
        z += -20.91224 * Q.sd_rg + 4.837732
    if Q.sd_rg >= 0.324646:
        z += -23.95568 * Q.sd_rg + 5.825771
    if Q.D2 < 0.2568137:
        z += -0.2609598 * Q.D2 + 0.06701807
    if Q.sj3_pairmax_over_m >= 0.9367476:
        z += -2.500278 * Q.sj3_pairmax_over_m + 2.34213
    if 0.1515405 <= Q.z_dr_0_0p05 < 0.6080732:
        z += 0.4406064 * Q.z_dr_0_0p05 - 0.0667697
    if Q.z_dr_0_0p05 >= 0.6080732:
        z += -0.6677328 * Q.z_dr_0_0p05 + 0.6071817
    if Q.z_dr_0p1_0p2 < 0.07870506:
        z += -0.9792447 * Q.z_dr_0p1_0p2 + 0.07707152
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -9352.688 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += 1433.24 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1_plus_lam2 < 0.00609665:
        z += 4234.952 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.lam1_plus_lam2)
    if Q.planar_flow < 0.1115136 and Q.sum_pt_top5 < 658.125:
        z += -0.0168577 * (0.1115136 - Q.planar_flow) * (658.125 - Q.sum_pt_top5)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -1177.3 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.planar_flow < 0.1115136 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -19.74274 * (0.1115136 - Q.planar_flow) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.sum_zz_dr2 > 0.002074109 and Q.sj2_dr < 0.1872617:
        z += -11175.18 * (Q.sum_zz_dr2 - 0.002074109) * (0.1872617 - Q.sj2_dr)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.D3 < 0.3201347:
        z += 5.787689 * (0.5882598 - Q.z_dr_0p05_0p1) * (0.3201347 - Q.D3)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 1.0:
        z += 8.56795 * (Q.z_dr_0p05_0p1 - 0.7509095) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.sd_mass < 74.57663 and Q.sj2_mass2 > 3.697444:
        z += 0.002974218 * (74.57663 - Q.sd_mass) * (Q.sj2_mass2 - 3.697444)
    if Q.e3 < 5.334511e-05 and Q.n_dr_0p2_0p4 > 2.0:
        z += -2264.763 * (5.334511e-05 - Q.e3) * (Q.n_dr_0p2_0p4 - 2.0)
    if Q.planar_flow < 0.1115136 and Q.phi_7 < -0.08051147:
        z += -35.22731 * (0.1115136 - Q.planar_flow) * (-0.08051147 - Q.phi_7)
    if Q.planar_flow < 0.1115136 and Q.zdr_7 < 0.001964442:
        z += -3214.249 * (0.1115136 - Q.planar_flow) * (0.001964442 - Q.zdr_7)
    if Q.e3 < 5.334511e-05 and Q.sj3_dr23 > 0.1629004:
        z += 244277.7 * (5.334511e-05 - Q.e3) * (Q.sj3_dr23 - 0.1629004)
    if Q.sj2_dr > 0.2001708 and Q.z_5 < 0.1058993:
        z += -7.680037 * (Q.sj2_dr - 0.2001708) * (0.1058993 - Q.z_5)
    if Q.e3 < 5.334511e-05 and Q.phi_5 < -0.05200653:
        z += -79399.42 * (5.334511e-05 - Q.e3) * (-0.05200653 - Q.phi_5)
    if Q.centroid_offset > 0.02685622 and Q.dr_3 < 0.05268713:
        z += -681.83 * (Q.centroid_offset - 0.02685622) * (0.05268713 - Q.dr_3)
    if Q.psi_0p1 > 0.8155839 and Q.phi_4 > 0.1096802:
        z += 46.76744 * (Q.psi_0p1 - 0.8155839) * (Q.phi_4 - 0.1096802)
    if Q.planar_flow < 0.1115136 and Q.dr_4 > 0.05563519:
        z += -0.5505939 * (0.1115136 - Q.planar_flow) * (Q.dr_4 - 0.05563519)
    if Q.sj2_dr > 0.1591713 and Q.D2_b2 < 0.08499387:
        z += 648.2217 * (Q.sj2_dr - 0.1591713) * (0.08499387 - Q.D2_b2)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.D2_b2 < 0.1830092:
        z += -6.507802 * (0.5882598 - Q.z_dr_0p05_0p1) * (0.1830092 - Q.D2_b2)
    if Q.sj2_dr > 0.2001708 and Q.D2_b2 < 0.08499387:
        z += -640.4677 * (Q.sj2_dr - 0.2001708) * (0.08499387 - Q.D2_b2)
    if Q.sj2_dr > 0.1294903 and Q.D2_b2 < 0.08499387:
        z += -277.2466 * (Q.sj2_dr - 0.1294903) * (0.08499387 - Q.D2_b2)
    if Q.psi_0p1 > 0.8155839 and Q.D2_b2 < 0.08499387:
        z += 94.32426 * (Q.psi_0p1 - 0.8155839) * (0.08499387 - Q.D2_b2)
    if Q.sj2_dr > 0.1591713 and Q.dr_3 < 0.02358801:
        z += -650.4902 * (Q.sj2_dr - 0.1591713) * (0.02358801 - Q.dr_3)
    if Q.sj2_dr > 0.2001708 and Q.dr_3 < 0.05268713:
        z += 27.75034 * (Q.sj2_dr - 0.2001708) * (0.05268713 - Q.dr_3)
    if Q.planar_flow < 0.1115136 and Q.mean_eta > -0.006779839:
        z += -70.39146 * (0.1115136 - Q.planar_flow) * (Q.mean_eta - -0.006779839)
    if Q.psi_0p1 > 0.9761279 and Q.dr02 > 0.175862:
        z += 1417.325 * (Q.psi_0p1 - 0.9761279) * (Q.dr02 - 0.175862)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.sum_pt < 715.4688:
        z += 0.002165394 * (0.5882598 - Q.z_dr_0p05_0p1) * (715.4688 - Q.sum_pt)
    if Q.centroid_offset > 0.04990367 and Q.C2_b2 < 0.0008333816:
        z += -234479.0 * (Q.centroid_offset - 0.04990367) * (0.0008333816 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.C2_b2 < 0.02415398:
        z += -286.7698 * (Q.z_dr_0p05_0p1 - 0.7509095) * (0.02415398 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.zdr_5 > 0.0155217:
        z += -11776.31 * (Q.z_dr_0p05_0p1 - 0.7509095) * (Q.zdr_5 - 0.0155217)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.D3 < 2.468971:
        z += -0.4653931 * (Q.z_dr_0p05_0p1 - 0.7509095) * (2.468971 - Q.D3)
    if Q.sd_rg > 0.2787955 and Q.D2_b2 < 0.2669656:
        z += -91.04595 * (Q.sd_rg - 0.2787955) * (0.2669656 - Q.D2_b2)
    if Q.sum_z_dr > 0.02689598 and Q.D2_b2 > 4.721224:
        z += 0.006217274 * (Q.sum_z_dr - 0.02689598) * (Q.D2_b2 - 4.721224)
    if Q.sj2_dr > 0.1591713 and Q.dr_2 < 0.03293672:
        z += -1087.071 * (Q.sj2_dr - 0.1591713) * (0.03293672 - Q.dr_2)
    if Q.sj2_dr > 0.2001708 and Q.dr_2 < 0.05056028:
        z += 401.5562 * (Q.sj2_dr - 0.2001708) * (0.05056028 - Q.dr_2)
    if Q.sd_mass < 49.91626 and Q.ptdr0_4 > 12.44629:
        z += 0.001372353 * (49.91626 - Q.sd_mass) * (Q.ptdr0_4 - 12.44629)
    if Q.mass_top5 > 49.18618 and Q.sd_nremoved > 1.0:
        z += -0.01058844 * (Q.mass_top5 - 49.18618) * (Q.sd_nremoved - 1.0)
    if Q.sj3_pairmax_over_m > 0.9367476 and Q.dr0_4 < 0.01000023:
        z += 302.0966 * (Q.sj3_pairmax_over_m - 0.9367476) * (0.01000023 - Q.dr0_4)
    if Q.centroid_offset > 0.02685622 and Q.pt_1 < 150.625:
        z += 0.6956331 * (Q.centroid_offset - 0.02685622) * (150.625 - Q.pt_1)
    if Q.planar_flow < 0.1115136 and Q.sj3_mass3 > 0.9601884:
        z += -2.072492 * (0.1115136 - Q.planar_flow) * (Q.sj3_mass3 - 0.9601884)
    if Q.mass > 69.61135 and Q.dr0_7 < 0.003768113:
        z += -52.59677 * (Q.mass - 69.61135) * (0.003768113 - Q.dr0_7)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.1905012
    if Q.N2 < 0.2233283:
        z += -0.6721324 * Q.N2 + 0.1501062
    if Q.sum_z_dr2 < 0.007520088:
        z += 881.7999 * Q.sum_z_dr2 - 6.631213
    if Q.mass_over_sum_pt < 0.1309286:
        z += 25.17663 * Q.mass_over_sum_pt - 3.296342
    if Q.lam1_plus_lam2 < 0.00609665:
        z += -426.3044 * Q.lam1_plus_lam2 + 8.231132
    if 0.00609665 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -670.4376 * Q.lam1_plus_lam2 + 9.719527
    if 0.01323868 <= Q.lam1_plus_lam2 < 0.01882765:
        z += -150.9796 * Q.lam1_plus_lam2 + 2.842591
    if Q.e2 < 0.04110972:
        z += -80.10643 * Q.e2 + 3.293153
    if Q.sum_zz_dr2 < 0.008168571:
        z += 946.9547 * Q.sum_zz_dr2 - 8.599898
    if 0.008168571 <= Q.sum_zz_dr2 < 0.01165737:
        z += 247.8302 * Q.sum_zz_dr2 - 2.88905
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -977.6573 * Q.mass_over_sum_pt_sq + 7.022352
    if Q.lam1 < 0.002464291:
        z += -197.9748 * Q.lam1 + 1.215943
    if 0.002464291 <= Q.lam1 < 0.00483998:
        z += -306.4691 * Q.lam1 + 1.483305
    if Q.lam2 < 0.001130645:
        z += -343.3518 * Q.lam2 + 0.3882089
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 536.6078 * Q.sum_z_dr2_top3 - 1.154548
    if Q.sum_z_dr < 0.1019409:
        z += 28.44734 * Q.sum_z_dr - 2.899948
    if Q.sj2_dr < 0.1492731:
        z += -16.82845 * Q.sj2_dr + 2.157491
    if 0.1492731 <= Q.sj2_dr < 0.1591713:
        z += -36.34942 * Q.sj2_dr + 5.071447
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 17.42308 * Q.sj2_dr - 3.487592
    if Q.sum_z_dr2_top2 < 0.0005124533:
        z += 2360.972 * Q.sum_z_dr2_top2 - 1.209888
    if Q.sj3_pair_mass_max >= 80.4:
        z += 0.01090167 * Q.sj3_pair_mass_max - 0.8764942
    if Q.sj3_pairmin_over_m < 0.03043859:
        z += -34.47862 * Q.sj3_pairmin_over_m + 1.049481
    if Q.N2 < 0.2233283 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -9.832127 * (0.2233283 - Q.N2) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.N2 < 0.2233283 and Q.max_dr < 0.121681:
        z += -190.5544 * (0.2233283 - Q.N2) * (0.121681 - Q.max_dr)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 < 716.8828:
        z += 0.007411754 * (0.2233283 - Q.N2) * (716.8828 - Q.sum_pt_top5)
    if Q.sum_z_dr2 < 0.007520088 and Q.D2 < 0.7459513:
        z += -3661.406 * (0.007520088 - Q.sum_z_dr2) * (0.7459513 - Q.D2)
    if Q.lam1_plus_lam2 < 0.00609665 and Q.D2 < 0.7459513:
        z += 1023.724 * (0.00609665 - Q.lam1_plus_lam2) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.D2 < 0.7459513:
        z += 57.13461 * (0.1309286 - Q.mass_over_sum_pt) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.z_dr_0p1_0p2 > 0.4684459:
        z += 36.60087 * (0.1309286 - Q.mass_over_sum_pt) * (Q.z_dr_0p1_0p2 - 0.4684459)
    if Q.N2 < 0.2233283 and Q.n_dr_0p1_0p2 < 4.0:
        z += 0.7454437 * (0.2233283 - Q.N2) * (4.0 - Q.n_dr_0p1_0p2)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.02293568 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
    if Q.N2 < 0.2233283 and Q.pt_entropy < 1.54073:
        z += -16.73818 * (0.2233283 - Q.N2) * (1.54073 - Q.pt_entropy)
    if Q.N2 < 0.2233283 and Q.sj3_pairmax_over_m > 0.9367476:
        z += -76.67504 * (0.2233283 - Q.N2) * (Q.sj3_pairmax_over_m - 0.9367476)
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
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:8] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:8] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:8] + [0.0] * 0
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
