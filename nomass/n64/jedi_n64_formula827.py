"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; no mass observables), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
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
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta2              pT-weighted mean Δη²
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
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
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
    )


def neuron_0(Q):
    z = -1.441395
    if Q.n_dr_0p2_0p4 < 13.0:
        z += -0.04889586 * Q.n_dr_0p2_0p4 + 0.8080185
    if 13.0 <= Q.n_dr_0p2_0p4 < 18.0:
        z += -0.03447447 * Q.n_dr_0p2_0p4 + 0.6205405
    if Q.sum_z_dr2_top40 < 0.001535432:
        z += 245.9898 * Q.sum_z_dr2_top40 + 0.231362
    if 0.001535432 <= Q.sum_z_dr2_top40 < 0.003113918:
        z += 60.90453 * Q.sum_z_dr2_top40 + 0.5155478
    if 0.003113918 <= Q.sum_z_dr2_top40 < 0.006259772:
        z += -69.50262 * Q.sum_z_dr2_top40 + 0.921625
    if 0.006259772 <= Q.sum_z_dr2_top40 < 0.008840538:
        z += -188.531 * Q.sum_z_dr2_top40 + 1.666716
    if Q.sum_z_dr2 < 0.008190222:
        z += -135.1793 * Q.sum_z_dr2 + 1.107149
    if Q.sum_z_dr2_top50 < 0.005852839:
        z += -233.6143 * Q.sum_z_dr2_top50 + 2.088816
    if 0.005852839 <= Q.sum_z_dr2_top50 < 0.008124776:
        z += -317.5746 * Q.sum_z_dr2_top50 + 2.580222
    if Q.sum_zz_dr2 < 0.00616708:
        z += -298.3539 * Q.sum_zz_dr2 + 2.681402
    if 0.00616708 <= Q.sum_zz_dr2 < 0.006936725:
        z += -993.7678 * Q.sum_zz_dr2 + 6.970074
    if 0.006936725 <= Q.sum_zz_dr2 < 0.007872294:
        z += -498.0478 * Q.sum_zz_dr2 + 3.531401
    if 0.007872294 <= Q.sum_zz_dr2 < 0.009606007:
        z += 224.5913 * Q.sum_zz_dr2 - 2.157426
    if 1002.379 <= Q.sum_pt < 1115.723:
        z += -0.01192036 * Q.sum_pt + 11.94871
    if 1115.723 <= Q.sum_pt < 1167.447:
        z += -0.00656536 * Q.sum_pt + 5.974018
    if Q.sum_pt >= 1167.447:
        z += -0.001625279 * Q.sum_pt + 0.2067365
    if 858.8262 <= Q.sum_pt_top40 < 1024.942:
        z += 0.006521916 * Q.sum_pt_top40 - 5.601192
    if Q.sum_pt_top40 >= 1024.942:
        z += 0.003986779 * Q.sum_pt_top40 - 3.002822
    if Q.psi_0p3 >= 0.9943058:
        z += 52.31122 * Q.psi_0p3 - 52.01335
    if Q.lam1 < 0.005913555:
        z += 67.48396 * Q.lam1 - 0.39907
    if Q.tau1 < 0.06310829:
        z += 24.17577 * Q.tau1 - 1.686551
    if 0.06310829 <= Q.tau1 < 0.0705748:
        z += 21.5442 * Q.tau1 - 1.520478
    if Q.psi_0p2 >= 0.9087063:
        z += -4.846468 * Q.psi_0p2 + 4.404016
    if Q.e2 < 0.01879315:
        z += -47.9433 * Q.e2 + 0.9010055
    if Q.C2 < 0.05602756:
        z += 7.674192 * Q.C2 - 0.4299662
    if Q.log_sum_pt >= 6.910131:
        z += -9.409142 * Q.log_sum_pt + 65.0184
    if Q.sum_pt_top30 < 950.9324:
        z += -0.002797084 * Q.sum_pt_top30 + 3.383089
    if 950.9324 <= Q.sum_pt_top30 < 1073.473:
        z += -0.002151083 * Q.sum_pt_top30 + 2.768786
    if 1073.473 <= Q.sum_pt_top30 < 1191.938:
        z += -0.002565884 * Q.sum_pt_top30 + 3.214064
    if Q.sum_pt_top30 >= 1191.938:
        z += 0.0006460004 * Q.sum_pt_top30 - 0.6143027
    if Q.sum_z_dr2_top30 < 0.006363916:
        z += 162.4159 * Q.sum_z_dr2_top30 - 0.8192922
    if 0.006363916 <= Q.sum_z_dr2_top30 < 0.008376291:
        z += -106.4954 * Q.sum_z_dr2_top30 + 0.8920367
    if Q.tau21_b2 < 0.2708235:
        z += -0.6844233 * Q.tau21_b2 + 0.185358
    if 1035.416 <= Q.sum_pt_top20 < 1064.139:
        z += 0.003757477 * Q.sum_pt_top20 - 3.890553
    if Q.sum_pt_top20 >= 1064.139:
        z += -0.001765031 * Q.sum_pt_top20 + 1.986164
    if Q.sum_pt_top15 < 613.7:
        z += 0.003790361 * Q.sum_pt_top15 - 2.326144
    if Q.sj2_dr >= 0.2232169:
        z += -1.797739 * Q.sj2_dr + 0.4012858
    if Q.sum_pt_top10 >= 845.5367:
        z += 0.0008963128 * Q.sum_pt_top10 - 0.7578654
    if Q.z_top20_slots >= 0.8474481:
        z += -1.611504 * Q.z_top20_slots + 1.365666
    if Q.sum_z_dr2 < 0.008190222 and Q.sum_z_dr2_top15 < 0.006142802:
        z += -35710.01 * (0.008190222 - Q.sum_z_dr2) * (0.006142802 - Q.sum_z_dr2_top15)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.log_sum_pt > 6.915514:
        z += -0.1599976 * (18.0 - Q.n_dr_0p2_0p4) * (Q.log_sum_pt - 6.915514)
    if Q.sum_z_dr2_top50 < 0.005852839 and Q.sum_pt < 1260.541:
        z += -1.434432 * (0.005852839 - Q.sum_z_dr2_top50) * (1260.541 - Q.sum_pt)
    if Q.log_sum_pt > 6.910131 and Q.e3 < 0.0001086251:
        z += 52922.13 * (Q.log_sum_pt - 6.910131) * (0.0001086251 - Q.e3)
    if Q.sum_pt_top30 < 1073.473 and Q.mean_eta2 < 0.0009793444:
        z += 2.931563 * (1073.473 - Q.sum_pt_top30) * (0.0009793444 - Q.mean_eta2)
    if Q.log_sum_pt > 6.910131 and Q.mean_eta2 < 0.01204531:
        z += 214.0271 * (Q.log_sum_pt - 6.910131) * (0.01204531 - Q.mean_eta2)
    if Q.sum_pt_top15 < 613.7 and Q.z_9 < 0.009867229:
        z += -205.8156 * (613.7 - Q.sum_pt_top15) * (0.009867229 - Q.z_9)
    if Q.sum_pt_top30 < 1073.473 and Q.D2_b2 < 43.66338:
        z += -8.241217e-05 * (1073.473 - Q.sum_pt_top30) * (43.66338 - Q.D2_b2)
    if Q.sum_z_dr2_top5 < 0.002270363 and Q.C2_b2 < 0.0216878:
        z += 3837.33 * (0.002270363 - Q.sum_z_dr2_top5) * (0.0216878 - Q.C2_b2)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.9825482
    if Q.n_particles >= 38.0:
        z += 0.0684642 * Q.n_particles - 2.60164
    if Q.log_sum_pt < 6.910131:
        z += 8.93327 * Q.log_sum_pt - 63.77726
    if 6.910131 <= Q.log_sum_pt < 6.98945:
        z += 43.96485 * Q.log_sum_pt - 305.85
    if 6.98945 <= Q.log_sum_pt < 7.139296:
        z += 20.78584 * Q.log_sum_pt - 143.8415
    if Q.log_sum_pt >= 7.139296:
        z += 11.85257 * Q.log_sum_pt - 80.06428
    if Q.sum_pt_top50 < 934.2416:
        z += -0.01303197 * Q.sum_pt_top50 + 14.06142
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += -0.02138049 * Q.sum_pt_top50 + 21.86095
    if 959.0957 <= Q.sum_pt_top50 < 1078.994:
        z += -0.02953978 * Q.sum_pt_top50 + 29.68649
    if Q.sum_pt_top50 >= 1078.994:
        z += -0.01650781 * Q.sum_pt_top50 + 15.62508
    if Q.sum_z_dr2_top15 < 0.002197765:
        z += -119.185 * Q.sum_z_dr2_top15 + 0.2619406
    if 0.4670285 <= Q.z_top5 < 0.6551948:
        z += -1.070801 * Q.z_top5 + 0.5000944
    if Q.z_top5 >= 0.6551948:
        z += 0.08821893 * Q.z_top5 - 0.2592892
    if 0.9638082 <= Q.psi_0p3 < 0.9966167:
        z += 1.827029 * Q.psi_0p3 - 1.760905
    if Q.psi_0p3 >= 0.9966167:
        z += -91.42628 * Q.psi_0p3 + 91.1769
    if Q.tau21 < 0.2140276:
        z += -5.079621 * Q.tau21 + 1.434276
    if 0.2140276 <= Q.tau21 < 0.5100475:
        z += -1.172546 * Q.tau21 + 0.5980543
    if Q.soft5_pt < 2.117188:
        z += -0.1309889 * Q.soft5_pt + 0.2773281
    if Q.lam2 < 0.0007143144:
        z += 467.2908 * Q.lam2 - 0.4923484
    if 0.0007143144 <= Q.lam2 < 0.001776308:
        z += 149.3001 * Q.lam2 - 0.265203
    if Q.C2 < 0.05602756:
        z += -10.07135 * Q.C2 + 0.564273
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.07368429 * Q.n_dr_0p2_0p4 - 0.51579
    if Q.pt_11 < 29.04688:
        z += 0.01500569 * Q.pt_11 - 0.4358685
    if Q.D2_b2 < 0.2773918:
        z += 2.085159 * Q.D2_b2 - 0.931775
    if 0.2773918 <= Q.D2_b2 < 0.8495689:
        z += 0.4253265 * Q.D2_b2 - 0.4713512
    if 0.8495689 <= Q.D2_b2 < 3.852812:
        z += 0.03662941 * Q.D2_b2 - 0.1411262
    if Q.sum_pt < 972.0419:
        z += 0.003892156 * Q.sum_pt - 4.543885
    if 972.0419 <= Q.sum_pt < 1052.889:
        z += 0.02627497 * Q.sum_pt - 26.30092
    if 1052.889 <= Q.sum_pt < 1167.447:
        z += 0.01532339 * Q.sum_pt - 14.77011
    if Q.sum_pt >= 1167.447:
        z += 0.01143123 * Q.sum_pt - 10.22623
    if Q.sum_z_dr2_top5 < 0.0006570502:
        z += -680.6409 * Q.sum_z_dr2_top5 + 0.4472152
    if Q.sum_pt_top2 < 605.875:
        z += -0.001825079 * Q.sum_pt_top2 + 1.10577
    if 15.0 <= Q.n_dr_0_0p05 < 30.0:
        z += 0.03458979 * Q.n_dr_0_0p05 - 0.5188468
    if Q.n_dr_0_0p05 >= 30.0:
        z += -0.01944279 * Q.n_dr_0_0p05 + 1.102131
    if Q.N2 >= 0.4678622:
        z += -6.146674 * Q.N2 + 2.875796
    if Q.n_dr_0p1_0p2 < 7.0:
        z += 0.0670225 * Q.n_dr_0p1_0p2 - 0.4691575
    if Q.tau1 < 0.02607472:
        z += -13.31818 * Q.tau1 + 2.60217
    if 0.02607472 <= Q.tau1 < 0.1219132:
        z += -24.19264 * Q.tau1 + 2.885718
    if 0.1219132 <= Q.tau1 < 0.1953848:
        z += -13.36684 * Q.tau1 + 1.565909
    if Q.tau1 >= 0.1953848:
        z += -0.04866219 * Q.tau1 - 1.03626
    if Q.sum_z_dr2_top30 < 0.004763596:
        z += -239.7738 * Q.sum_z_dr2_top30 + 1.667779
    if 0.004763596 <= Q.sum_z_dr2_top30 < 0.007463985:
        z += -194.6364 * Q.sum_z_dr2_top30 + 1.452763
    if Q.sum_zz_dr2 < 0.009606007:
        z += 222.0721 * Q.sum_zz_dr2 - 2.133226
    if Q.sum_pt_top40 < 1225.842:
        z += -0.005446374 * Q.sum_pt_top40 + 6.676395
    if Q.e2 < 0.01036127:
        z += 21.08657 * Q.e2 - 0.4661857
    if 0.01036127 <= Q.e2 < 0.02210818:
        z += 38.79608 * Q.e2 - 0.6496788
    if Q.e2 >= 0.02210818:
        z += 17.70951 * Q.e2 - 0.1834931
    if Q.D3 < 0.03962283:
        z += -3.913126 * Q.D3 + 0.1550491
    if Q.sum_z_dr2_top20 < 0.001153063:
        z += -398.9975 * Q.sum_z_dr2_top20 + 0.4600692
    if Q.sum_z_dr2 < 0.001784491:
        z += 491.8148 * Q.sum_z_dr2 - 2.147431
    if 0.001784491 <= Q.sum_z_dr2 < 0.01397874:
        z += 104.1303 * Q.sum_z_dr2 - 1.455611
    if Q.z_top30_slots >= 0.9564984:
        z += 12.98372 * Q.z_top30_slots - 12.4189
    if Q.LHA < 0.2454112:
        z += -2.663562 * Q.LHA + 0.6536681
    if Q.zdr_0 >= 0.01240028:
        z += -16.88058 * Q.zdr_0 + 0.2093239
    if Q.lam1 < 0.005240342:
        z += -129.6691 * Q.lam1 + 0.6795102
    if Q.soft2_dr < 0.3845054:
        z += 1.174046 * Q.soft2_dr - 0.4514269
    if Q.soft2_dr0 < 0.4086381:
        z += -0.910724 * Q.soft2_dr0 + 0.3721566
    if Q.tau4 < 0.02017767:
        z += 22.24283 * Q.tau4 - 0.4488086
    if Q.z_8 < 0.03044378:
        z += 9.834811 * Q.z_8 - 0.2994088
    if Q.n_dr_0p05_0p1 >= 17.0:
        z += 0.01195956 * Q.n_dr_0p05_0p1 - 0.2033125
    if Q.soft1_z < 0.001452174:
        z += -675.5645 * Q.soft1_z + 0.9810375
    if Q.D2 < 1.409617:
        z += -0.424217 * Q.D2 + 0.7421283
    if 1.409617 <= Q.D2 < 2.178951:
        z += -0.1873632 * Q.D2 + 0.4082552
    if Q.n_pt_above_5 >= 46.0:
        z += -0.03275055 * Q.n_pt_above_5 + 1.506525
    if Q.soft1_pt < 2.275391:
        z += 0.4835393 * Q.soft1_pt - 1.100241
    if Q.tau21_b2 < 0.7058597:
        z += 0.4195887 * Q.tau21_b2 - 0.2961707
    if Q.sum_pt_top30 < 1038.262:
        z += -0.00228886 * Q.sum_pt_top30 + 2.376436
    if Q.n_particles > 38.0 and Q.zdr_0 < 0.009970338:
        z += 2.398657 * (Q.n_particles - 38.0) * (0.009970338 - Q.zdr_0)
    if Q.n_particles > 38.0 and Q.soft1_z < 0.002181998:
        z += -22.06359 * (Q.n_particles - 38.0) * (0.002181998 - Q.soft1_z)
    if Q.z_top5 > 0.6551948 and Q.pt1_dr01 < 28.39396:
        z += -0.1117758 * (Q.z_top5 - 0.6551948) * (28.39396 - Q.pt1_dr01)
    if Q.sum_z_dr2_top15 < 0.002197765 and Q.psi_0p3 > 0.9966167:
        z += 72559.48 * (0.002197765 - Q.sum_z_dr2_top15) * (Q.psi_0p3 - 0.9966167)
    if Q.n_particles > 38.0 and Q.absphi_1 < 0.1178619:
        z += 0.09692302 * (Q.n_particles - 38.0) * (0.1178619 - Q.absphi_1)
    if Q.z_top30_slots > 0.9341838 and Q.pt1_dr01 > 5.351077:
        z += 0.2848315 * (Q.z_top30_slots - 0.9341838) * (Q.pt1_dr01 - 5.351077)
    if Q.n_particles > 38.0 and Q.tau32 > 0.3293142:
        z += 0.02901171 * (Q.n_particles - 38.0) * (Q.tau32 - 0.3293142)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_2 > 7.740999:
        z += 0.3401073 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_2 - 7.740999)
    if Q.M3 < 0.03457336 and Q.psi_0p3 > 0.9299135:
        z += -235.1749 * (0.03457336 - Q.M3) * (Q.psi_0p3 - 0.9299135)
    if Q.n_particles > 38.0 and Q.tau43 < 0.9624339:
        z += -0.02113058 * (Q.n_particles - 38.0) * (0.9624339 - Q.tau43)
    if Q.z_dr_0_0p05 > 0.8103116 and Q.n_dr_0p05_0p1 < 10.0:
        z += -0.6029522 * (Q.z_dr_0_0p05 - 0.8103116) * (10.0 - Q.n_dr_0p05_0p1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_4 > 2.381691:
        z += 0.3278946 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_4 - 2.381691)
    if Q.pt_11 < 29.04688 and Q.dr1_12 < 0.1745719:
        z += 0.04522786 * (29.04688 - Q.pt_11) * (0.1745719 - Q.dr1_12)
    if Q.z_dr_0_0p05 > 0.8103116 and Q.n_real_top20 < 20.0:
        z += 0.5787283 * (Q.z_dr_0_0p05 - 0.8103116) * (20.0 - Q.n_real_top20)
    if Q.z_top30_slots > 0.9564984 and Q.ptdr0_3 < 10.17512:
        z += -0.7785204 * (Q.z_top30_slots - 0.9564984) * (10.17512 - Q.ptdr0_3)
    if Q.sum_pt_top2 < 605.875 and Q.soft3_dr < 0.3797:
        z += -0.002415068 * (605.875 - Q.sum_pt_top2) * (0.3797 - Q.soft3_dr)
    if Q.n_dr_0_0p05 > 15.0 and Q.eta_1 < 0.002292633:
        z += -0.4790518 * (Q.n_dr_0_0p05 - 15.0) * (0.002292633 - Q.eta_1)
    if Q.D2_b2 < 3.852812 and Q.soft4_dr < 0.3176645:
        z += -0.1211043 * (3.852812 - Q.D2_b2) * (0.3176645 - Q.soft4_dr)
    if Q.z_top30_slots > 0.9564984 and Q.mratio_min_012 < 0.1847499:
        z += 34.98553 * (Q.z_top30_slots - 0.9564984) * (0.1847499 - Q.mratio_min_012)
    if Q.sum_pt_top2 < 605.875 and Q.soft3_dr0 > 0.02918107:
        z += -0.00155738 * (605.875 - Q.sum_pt_top2) * (Q.soft3_dr0 - 0.02918107)
    if Q.z_top30_slots > 0.9564984 and Q.ptdr0_5 > 0.4637912:
        z += 0.4072441 * (Q.z_top30_slots - 0.9564984) * (Q.ptdr0_5 - 0.4637912)
    if Q.soft9_pt < 2.5 and Q.dr_13 < 0.1156062:
        z += 1.239317 * (2.5 - Q.soft9_pt) * (0.1156062 - Q.dr_13)
    if Q.z_8 < 0.03044378 and Q.soft10_abseta < 0.002968979:
        z += 3926.393 * (0.03044378 - Q.z_8) * (0.002968979 - Q.soft10_abseta)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.2294988
    if 6.903423 <= Q.log_sum_pt < 6.930088:
        z += 6.003552 * Q.log_sum_pt - 41.44506
    if 6.930088 <= Q.log_sum_pt < 7.017258:
        z += 8.38111 * Q.log_sum_pt - 57.92174
    if 7.017258 <= Q.log_sum_pt < 7.062574:
        z += 6.562849 * Q.log_sum_pt - 45.16254
    if Q.log_sum_pt >= 7.062574:
        z += 2.951062 * Q.log_sum_pt - 19.65402
    if 1007.788 <= Q.sum_pt < 1085.125:
        z += 0.006784347 * Q.sum_pt - 6.837187
    if Q.sum_pt >= 1085.125:
        z += 0.00160189 * Q.sum_pt - 1.213573
    if Q.sum_z_dr2 < 0.006941794:
        z += 245.9154 * Q.sum_z_dr2 - 1.707094
    if Q.sum_z_dr2_top50 < 0.00634935:
        z += -164.4734 * Q.sum_z_dr2_top50 + 1.044299
    if Q.sum_pt_top50 >= 1003.544:
        z += -0.006145179 * Q.sum_pt_top50 + 6.166959
    if Q.psi_0p3 >= 0.9943058:
        z += -22.83891 * Q.psi_0p3 + 22.70886
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.00505793 * Q.sum_pt_top40 - 5.410322
    if Q.sum_pt_top30 >= 1011.524:
        z += -0.003073537 * Q.sum_pt_top30 + 3.108956
    if Q.sum_pt_top15 >= 1003.329:
        z += -0.002017997 * Q.sum_pt_top15 + 2.024716
    if Q.sum_pt_top20 >= 1064.139:
        z += 0.001194001 * Q.sum_pt_top20 - 1.270583
    if Q.lam1 < 0.01174405:
        z += -42.06797 * Q.lam1 + 0.4940483
    if Q.log_sum_pt > 6.903423 and Q.sum_z_dr2_top15 < 0.02146578:
        z += 53.02993 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.sum_z_dr2_top15)
    if Q.log_sum_pt > 7.062574 and Q.sum_z_dr2_top30 > 0.01215787:
        z += 665.8612 * (Q.log_sum_pt - 7.062574) * (Q.sum_z_dr2_top30 - 0.01215787)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p1 < 0.8747961:
        z += -6.045296 * (Q.log_sum_pt - 6.903423) * (0.8747961 - Q.psi_0p1)
    if Q.log_sum_pt > 6.930088 and Q.sum_z_dr2_top30 > 0.00375223:
        z += -379.7378 * (Q.log_sum_pt - 6.930088) * (Q.sum_z_dr2_top30 - 0.00375223)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p3 < 1.0:
        z += 20.28634 * (Q.log_sum_pt - 6.903423) * (1.0 - Q.psi_0p3)
    if Q.log_sum_pt > 6.903423 and Q.zdr_3 < 0.00781909:
        z += -83.12996 * (Q.log_sum_pt - 6.903423) * (0.00781909 - Q.zdr_3)
    if Q.log_sum_pt > 6.930088 and Q.z_top15_slots > 0.6655806:
        z += 13.36071 * (Q.log_sum_pt - 6.930088) * (Q.z_top15_slots - 0.6655806)
    if Q.sum_pt_top20 > 993.5664 and Q.z_dr_0p1_0p2 < 0.1203437:
        z += -0.01693638 * (Q.sum_pt_top20 - 993.5664) * (0.1203437 - Q.z_dr_0p1_0p2)
    if Q.sum_pt_top30 > 1111.245 and Q.z_dr_0p1_0p2 < 0.1203437:
        z += 0.02270848 * (Q.sum_pt_top30 - 1111.245) * (0.1203437 - Q.z_dr_0p1_0p2)
    if Q.sum_pt_top50 > 1003.544 and Q.z_dr_0p1_0p2 > 0.003353111:
        z += 0.001931873 * (Q.sum_pt_top50 - 1003.544) * (Q.z_dr_0p1_0p2 - 0.003353111)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.1185324
    if Q.lam2 < 0.0006154841:
        z += -529.7711 * Q.lam2 + 0.3260656
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.1368123 * Q.n_dr_0p2_0p4 + 0.6840617
    if Q.n_particles < 46.0:
        z += -0.04090966 * Q.n_particles + 1.881844
    if Q.tau21 < 0.347196:
        z += 2.249097 * Q.tau21 - 0.7808774
    if Q.D2 < 1.788105:
        z += -0.1324995 * Q.D2 + 0.236923
    if Q.sum_zz_dr2 < 0.007511864:
        z += -137.7125 * Q.sum_zz_dr2 + 1.034477
    if Q.LHA < 0.3719813:
        z += 1.423648 * Q.LHA - 0.5295703
    if Q.sum_z_dr2 < 0.009614971:
        z += 39.2657 * Q.sum_z_dr2 - 0.3775386
    if Q.n_pt_above_1 < 40.0:
        z += 0.01078877 * Q.n_pt_above_1 - 0.4315509
    if Q.z_dr_0p2_0p4 < 0.02647293:
        z += 7.552457 * Q.z_dr_0p2_0p4 - 0.1999356
    if Q.psi_0p2 >= 0.9976427:
        z += 85.23192 * Q.psi_0p2 - 85.03101
    if Q.sum_z_dr2_top40 < 0.006259772:
        z += -7.337975 * Q.sum_z_dr2_top40 + 0.2655997
    if 0.006259772 <= Q.sum_z_dr2_top40 < 0.008840538:
        z += -85.11645 * Q.sum_z_dr2_top40 + 0.7524752
    if Q.n_dr_0_0p05 < 25.0:
        z += 0.005629572 * Q.n_dr_0_0p05 - 0.1407393
    if Q.tau21_b2 < 0.2018786:
        z += -2.173868 * Q.tau21_b2 + 0.4388575
    if Q.psi_0p1 >= 0.8509811:
        z += -1.301598 * Q.psi_0p1 + 1.107635
    if Q.z_dr_0_0p05 >= 0.7674734:
        z += 0.557734 * Q.z_dr_0_0p05 - 0.428046
    if Q.sum_z_dr2_top30 < 0.006363916:
        z += 27.04041 * Q.sum_z_dr2_top30 - 0.1720829
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_dr_0p1_0p2 > 9.0:
        z += -0.005565139 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 9.0)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.e2 < 0.03029714:
        z += -6.50897 * (5.0 - Q.n_dr_0p2_0p4) * (0.03029714 - Q.e2)
    if Q.n_particles < 46.0 and Q.e2 > 0.01036127:
        z += -0.4714656 * (46.0 - Q.n_particles) * (Q.e2 - 0.01036127)
    if Q.LHA < 0.3719813 and Q.eta_1 > -0.04302979:
        z += -8.771918 * (0.3719813 - Q.LHA) * (Q.eta_1 - -0.04302979)
    if Q.sum_zz_dr2 < 0.007511864 and Q.sum_pt_top40 < 1024.942:
        z += -0.4618317 * (0.007511864 - Q.sum_zz_dr2) * (1024.942 - Q.sum_pt_top40)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.z_top50_slots > 0.985099:
        z += 3.463526 * (8.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.985099)
    if Q.max_dr < 0.2738063 and Q.z_top50_slots > 0.9586536:
        z += 65.90965 * (0.2738063 - Q.max_dr) * (Q.z_top50_slots - 0.9586536)
    if Q.n_dr_0p2_0p4 < 1.0 and Q.n_dr_0p4_up < 1.0:
        z += -0.2394024 * (1.0 - Q.n_dr_0p2_0p4) * (1.0 - Q.n_dr_0p4_up)
    if Q.n_dr_0p1_0p2 < 15.0 and Q.psi_0p2 > 0.9313699:
        z += 0.329756 * (15.0 - Q.n_dr_0p1_0p2) * (Q.psi_0p2 - 0.9313699)
    return max(0.0, z)


def neuron_4(Q):
    z = -0.1097262
    if Q.e2 < 0.03263075:
        z += 12.48086 * Q.e2 - 0.723562
    if 0.03263075 <= Q.e2 < 0.04358622:
        z += 28.87163 * Q.e2 - 1.258405
    if Q.e2 >= 0.06524004:
        z += -19.83924 * Q.e2 + 1.294313
    if Q.sum_zz_dr2 < 0.00592208:
        z += 403.7699 * Q.sum_zz_dr2 - 2.391157
    if Q.sum_zz_dr2 >= 0.02580859:
        z += 33.88924 * Q.sum_zz_dr2 - 0.8746335
    if Q.psi_0p3 >= 0.9973959:
        z += 49.03244 * Q.psi_0p3 - 48.90476
    if Q.n_particles >= 29.0:
        z += -0.01569496 * Q.n_particles + 0.4551537
    if Q.n_dr_0_0p05 >= 10.0:
        z += 0.01543996 * Q.n_dr_0_0p05 - 0.1543996
    if Q.sum_pt_top50 < 1061.184:
        z += 0.007747904 * Q.sum_pt_top50 - 8.221951
    if Q.sum_z_dr2 < 0.01397874:
        z += -24.198 * Q.sum_z_dr2 + 0.3382576
    if Q.sum_z_dr2_top20 < 0.007538019:
        z += 18.93995 * Q.sum_z_dr2_top20 + 0.1526024
    if 0.007538019 <= Q.sum_z_dr2_top20 < 0.01083435:
        z += -89.60621 * Q.sum_z_dr2_top20 + 0.9708253
    if Q.tau2 < 0.1011219:
        z += 1.992356 * Q.tau2 - 0.2014709
    if Q.n_for_50pct < 9.0:
        z += 0.01512682 * Q.n_for_50pct - 0.1361414
    if 0.00242543 <= Q.sum_z_dr2_top50 < 0.004573744:
        z += -102.2932 * Q.sum_z_dr2_top50 + 0.2481051
    if Q.sum_z_dr2_top50 >= 0.004573744:
        z += 69.10854 * Q.sum_z_dr2_top50 - 0.5358426
    if Q.sum_z_dr2_top15 < 0.002197765:
        z += -54.3448 * Q.sum_z_dr2_top15 + 0.8498633
    if 0.002197765 <= Q.sum_z_dr2_top15 < 0.004169954:
        z += 107.1823 * Q.sum_z_dr2_top15 + 0.4948648
    if 0.004169954 <= Q.sum_z_dr2_top15 < 0.007887677:
        z += 13.4787 * Q.sum_z_dr2_top15 + 0.8856044
    if 0.007887677 <= Q.sum_z_dr2_top15 < 0.01563836:
        z += -85.29343 * Q.sum_z_dr2_top15 + 1.664687
    if Q.sum_z_dr2_top15 >= 0.01563836:
        z += -30.94862 * Q.sum_z_dr2_top15 + 0.8148237
    if Q.n_dr_0p2_0p4 < 26.0:
        z += -0.03243497 * Q.n_dr_0p2_0p4 + 0.8433093
    z += -6.264578 * Q.zdr_0
    if Q.lam1 < 0.006716737:
        z += 145.9365 * Q.lam1 - 0.9802173
    if Q.lam1 >= 0.02048524:
        z += -35.33957 * Q.lam1 + 0.7239394
    if Q.lam1_plus_lam2 < 0.009614971:
        z += -235.4263 * Q.lam1_plus_lam2 + 2.263617
    if 0.1870291 <= Q.LHA < 0.2454112:
        z += 1.838099 * Q.LHA - 0.343778
    if 0.2454112 <= Q.LHA < 0.2845608:
        z += -4.461294 * Q.LHA + 1.202164
    if 0.2845608 <= Q.LHA < 0.3332345:
        z += -11.00317 * Q.LHA + 3.063727
    if Q.LHA >= 0.3332345:
        z += -13.45786 * Q.LHA + 3.881713
    if Q.sum_z_dr >= 0.076787:
        z += 28.21262 * Q.sum_z_dr - 2.166363
    if Q.sum_z_dr2_top40 < 0.005712208:
        z += 81.22944 * Q.sum_z_dr2_top40 - 0.7864438
    if 0.005712208 <= Q.sum_z_dr2_top40 < 0.008031986:
        z += 138.9979 * Q.sum_z_dr2_top40 - 1.116429
    if Q.n_dr_0p1_0p2 < 26.0:
        z += -0.00645655 * Q.n_dr_0p1_0p2 + 0.1678703
    if Q.z_top50_slots < 0.985099:
        z += 11.59711 * Q.z_top50_slots - 11.4243
    if Q.sum_pt_top40 < 1053.047:
        z += -0.006015297 * Q.sum_pt_top40 + 6.334393
    if Q.soft10_pt >= 1.603516:
        z += 0.09754565 * Q.soft10_pt - 0.156416
    if Q.soft10_z >= 0.0007540821:
        z += -86.02155 * Q.soft10_z + 0.06486731
    if Q.C2 >= 0.06655881:
        z += -2.499365 * Q.C2 + 0.1663548
    if Q.D2 >= 5.378975:
        z += -0.01841605 * Q.D2 + 0.09905948
    if Q.psi_0p1 >= 0.3628388:
        z += -0.546388 * Q.psi_0p1 + 0.1982508
    if Q.z_dr_0p05_0p1 >= 0.4026646:
        z += 0.3177005 * Q.z_dr_0p05_0p1 - 0.1279268
    if 0.1512157 <= Q.sj2_dr < 0.1825048:
        z += 2.598792 * Q.sj2_dr - 0.3929781
    if Q.sj2_dr >= 0.1825048:
        z += -0.8086855 * Q.sj2_dr + 0.228903
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += 2.132889 * Q.z_dr_0p2_0p4 - 0.1460848
    if Q.n_pt_above_1 >= 23.0:
        z += -0.005839719 * Q.n_pt_above_1 + 0.1343135
    if Q.n_particles > 29.0 and Q.e3 > 0.0001841806:
        z += 13.05269 * (Q.n_particles - 29.0) * (Q.e3 - 0.0001841806)
    if Q.sum_z_dr2 < 0.01397874 and Q.eccentricity > 0.8903081:
        z += 314.2774 * (0.01397874 - Q.sum_z_dr2) * (Q.eccentricity - 0.8903081)
    if Q.sum_z_dr2 < 0.01397874 and Q.psi_0p3 > 0.9777125:
        z += 1441.605 * (0.01397874 - Q.sum_z_dr2) * (Q.psi_0p3 - 0.9777125)
    if Q.n_particles > 29.0 and Q.soft1_pt < 2.275391:
        z += 0.005490049 * (Q.n_particles - 29.0) * (2.275391 - Q.soft1_pt)
    if Q.e2 < 0.03263075 and Q.sum_pt_top50 < 1013.916:
        z += -0.1393901 * (0.03263075 - Q.e2) * (1013.916 - Q.sum_pt_top50)
    if Q.psi_0p3 > 0.9973959 and Q.sum_pt_top40 > 858.8262:
        z += 0.336186 * (Q.psi_0p3 - 0.9973959) * (Q.sum_pt_top40 - 858.8262)
    if Q.n_dr_0p2_0p4 < 26.0 and Q.n_dr_0p1_0p2 > 13.0:
        z += -0.0006805276 * (26.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 13.0)
    if Q.lam2 < 0.001163277 and Q.z_dr_0p05_0p1 > 0.1514163:
        z += -291.2258 * (0.001163277 - Q.lam2) * (Q.z_dr_0p05_0p1 - 0.1514163)
    if Q.n_particles > 29.0 and Q.eccentricity > 0.3346888:
        z += -0.00966247 * (Q.n_particles - 29.0) * (Q.eccentricity - 0.3346888)
    if Q.psi_0p3 > 0.9973959 and Q.max_dr > 0.1939977:
        z += -252.666 * (Q.psi_0p3 - 0.9973959) * (Q.max_dr - 0.1939977)
    if Q.lam1_plus_lam2 < 0.009614971 and Q.soft10_z > 0.0007540821:
        z += -5310.844 * (0.009614971 - Q.lam1_plus_lam2) * (Q.soft10_z - 0.0007540821)
    if Q.sj2_dr > 0.2595052 and Q.C2_b2 < 0.0329485:
        z += -110.7464 * (Q.sj2_dr - 0.2595052) * (0.0329485 - Q.C2_b2)
    if Q.sj2_dr > 0.2595052 and Q.soft4_z > 0.0006393354:
        z += -1726.008 * (Q.sj2_dr - 0.2595052) * (Q.soft4_z - 0.0006393354)
    if Q.sj2_dr > 0.2595052 and Q.soft4_pt > 0.6098633:
        z += 2.445573 * (Q.sj2_dr - 0.2595052) * (Q.soft4_pt - 0.6098633)
    if Q.n_particles > 29.0 and Q.soft4_pt < 1.789258:
        z += -0.003382348 * (Q.n_particles - 29.0) * (1.789258 - Q.soft4_pt)
    if Q.LHA > 0.3719813 and Q.sj3_dr_max > 0.2268922:
        z += -36.17347 * (Q.LHA - 0.3719813) * (Q.sj3_dr_max - 0.2268922)
    if Q.LHA > 0.1870291 and Q.sj3_dr_max > 0.07466995:
        z += 6.99626 * (Q.LHA - 0.1870291) * (Q.sj3_dr_max - 0.07466995)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.875131
    z += -0.02612368 * Q.n_particles + 1.671915
    if 0.007872294 <= Q.sum_zz_dr2 < 0.009606007:
        z += -388.4536 * Q.sum_zz_dr2 + 3.058021
    if Q.sum_zz_dr2 >= 0.009606007:
        z += -153.0661 * Q.sum_zz_dr2 + 0.7968868
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -29.69413 * Q.log_sum_pt + 205.1903
    if 6.920349 <= Q.log_sum_pt < 6.935549:
        z += -34.55146 * Q.log_sum_pt + 238.8047
    if 6.935549 <= Q.log_sum_pt < 6.959294:
        z += -31.87148 * Q.log_sum_pt + 220.2176
    if 6.959294 <= Q.log_sum_pt < 7.017258:
        z += -19.42807 * Q.log_sum_pt + 133.6203
    if Q.log_sum_pt >= 7.017258:
        z += -18.40837 * Q.log_sum_pt + 126.4648
    if Q.sum_pt >= 907.9372:
        z += 0.003046909 * Q.sum_pt - 2.766402
    if Q.sum_z_dr2 >= 0.02928196:
        z += -3135.477 * Q.sum_z_dr2 + 91.81289
    if Q.psi_0p3 >= 0.9980008:
        z += 105.3905 * Q.psi_0p3 - 105.1798
    if Q.lam1_plus_lam2 >= 0.006941794:
        z += 54.65741 * Q.lam1_plus_lam2 - 0.3794205
    if 934.2416 <= Q.sum_pt_top50 < 1024.689:
        z += 0.01372766 * Q.sum_pt_top50 - 12.82495
    if 1024.689 <= Q.sum_pt_top50 < 1048.098:
        z += 0.01423469 * Q.sum_pt_top50 - 13.3445
    if Q.sum_pt_top50 >= 1048.098:
        z += 0.005280969 * Q.sum_pt_top50 - 3.96012
    if Q.n_pt_above_1 >= 26.0:
        z += 0.01647189 * Q.n_pt_above_1 - 0.4282692
    if Q.sum_pt_top20 >= 1017.778:
        z += 0.003768301 * Q.sum_pt_top20 - 3.835295
    if Q.n_for_90pct >= 11.0:
        z += 0.01481226 * Q.n_for_90pct - 0.1629348
    if Q.sum_pt_top5 < 579.875:
        z += 0.00044799 * Q.sum_pt_top5 - 0.7085087
    if 579.875 <= Q.sum_pt_top5 < 902.4062:
        z += 0.001391278 * Q.sum_pt_top5 - 1.255498
    if Q.sum_z_dr2_top15 < 0.005383629:
        z += 14.3128 * Q.sum_z_dr2_top15 - 0.5312713
    if 0.005383629 <= Q.sum_z_dr2_top15 < 0.01563836:
        z += 44.29338 * Q.sum_z_dr2_top15 - 0.6926756
    if Q.z_dr_0_0p05 >= 0.9084912:
        z += 2.942596 * Q.z_dr_0_0p05 - 2.673323
    if Q.max_dr >= 0.1939977:
        z += -0.4679273 * Q.max_dr + 0.0907768
    if 911.9328 <= Q.sum_pt_top30 < 1052.08:
        z += 0.003354234 * Q.sum_pt_top30 - 3.058836
    if Q.sum_pt_top30 >= 1052.08:
        z += 0.002712276 * Q.sum_pt_top30 - 2.383445
    if Q.z_top30_slots >= 0.9203881:
        z += -6.177184 * Q.z_top30_slots + 5.685407
    if Q.n_dr_0p2_0p4 < 8.0:
        z += -0.0211538 * Q.n_dr_0p2_0p4 + 0.1692304
    if Q.n_dr_0p2_0p4 >= 21.0:
        z += -0.0170046 * Q.n_dr_0p2_0p4 + 0.3570966
    if Q.sum_z_dr2_top10 < 0.005801034:
        z += -36.70848 * Q.sum_z_dr2_top10 + 0.2129472
    if Q.sum_z_dr2_top30 < 0.01215787:
        z += 61.42935 * Q.sum_z_dr2_top30 - 0.7468502
    if Q.psi_0p1 >= 0.9670742:
        z += 9.95225 * Q.psi_0p1 - 9.624564
    if Q.tau4 < 0.01626937:
        z += -17.3425 * Q.tau4 + 0.2821514
    if Q.pt_entropy >= 2.903111:
        z += -0.2705749 * Q.pt_entropy + 0.785509
    if 951.1375 <= Q.sum_pt_top15 < 967.7705:
        z += -0.005370237 * Q.sum_pt_top15 + 5.107834
    if Q.sum_pt_top15 >= 967.7705:
        z += -0.006227322 * Q.sum_pt_top15 + 5.937295
    if Q.sum_pt_top10 >= 822.975:
        z += 0.001738046 * Q.sum_pt_top10 - 1.430368
    if Q.sum_z_dr2_top20 < 0.008031209:
        z += 19.33419 * Q.sum_z_dr2_top20 - 0.155277
    if Q.tau2 >= 0.02164722:
        z += 3.281646 * Q.tau2 - 0.07103852
    if Q.sd_rg >= 0.1377546:
        z += 0.7186211 * Q.sd_rg - 0.09899337
    if Q.z_dr_0p2_0p4 < 0.2708738:
        z += 1.300606 * Q.z_dr_0p2_0p4 - 0.3522999
    if Q.e2 < 0.01256572:
        z += 22.98641 * Q.e2 + 0.2336925
    if 0.01256572 <= Q.e2 < 0.03875945:
        z += -19.9488 * Q.e2 + 0.7732044
    if Q.tau1 < 0.1072713:
        z += 8.992345 * Q.tau1 - 0.9646201
    if Q.n_dr_0p1_0p2 >= 11.0:
        z += 0.01362413 * Q.n_dr_0p1_0p2 - 0.1498654
    if Q.sum_z_dr < 0.07374472:
        z += -9.699143 * Q.sum_z_dr + 0.7152606
    if Q.sum_z_dr >= 0.1564779:
        z += -87.01334 * Q.sum_z_dr + 13.61566
    if Q.e3 < 0.0005178279:
        z += 302.5751 * Q.e3 - 0.1566818
    if Q.z_dr_0p1_0p2 >= 0.1872805:
        z += -0.3459576 * Q.z_dr_0p1_0p2 + 0.06479111
    if Q.D2 < 2.680253:
        z += -0.2062539 * Q.D2 + 0.5528126
    if Q.N2 < 0.3572263:
        z += 1.391014 * Q.N2 - 0.4969069
    if Q.n_particles < 64.0 and Q.D2 < 2.410481:
        z += -0.008671577 * (64.0 - Q.n_particles) * (2.410481 - Q.D2)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 76831.03 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    if Q.psi_0p1 > 0.9670742 and Q.psi_0p3 > 0.9980008:
        z += -5932.691 * (Q.psi_0p1 - 0.9670742) * (Q.psi_0p3 - 0.9980008)
    if Q.n_pt_above_1 > 58.0 and Q.psi_0p3 > 0.9985421:
        z += -29.26068 * (Q.n_pt_above_1 - 58.0) * (Q.psi_0p3 - 0.9985421)
    if Q.z_dr_0p2_0p4 < 0.2708738 and Q.zdr_0 > 0.005911134:
        z += 36.83883 * (0.2708738 - Q.z_dr_0p2_0p4) * (Q.zdr_0 - 0.005911134)
    if Q.n_particles < 64.0 and Q.pt1_dr01 < 12.6865:
        z += 0.0002430728 * (64.0 - Q.n_particles) * (12.6865 - Q.pt1_dr01)
    if Q.log_sum_pt > 6.910131 and Q.dr_11 < 0.09468596:
        z += -12.18566 * (Q.log_sum_pt - 6.910131) * (0.09468596 - Q.dr_11)
    if Q.sum_zz_dr2 > 0.009606007 and Q.soft7_z < 0.003627839:
        z += -76587.58 * (Q.sum_zz_dr2 - 0.009606007) * (0.003627839 - Q.soft7_z)
    if Q.sum_z_dr2 > 0.02928196 and Q.soft3_z < 0.002741632:
        z += -2742116.0 * (Q.sum_z_dr2 - 0.02928196) * (0.002741632 - Q.soft3_z)
    if Q.sum_zz_dr2 > 0.007872294 and Q.soft7_z < 0.003627839:
        z += 68589.6 * (Q.sum_zz_dr2 - 0.007872294) * (0.003627839 - Q.soft7_z)
    if Q.sum_pt_top50 > 1024.689 and Q.e4 < 5.8505e-08:
        z += -65073.02 * (Q.sum_pt_top50 - 1024.689) * (5.8505e-08 - Q.e4)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.285193
    if Q.lam1 < 0.003811746:
        z += -630.5456 * Q.lam1 + 3.189718
    if 0.003811746 <= Q.lam1 < 0.006189818:
        z += -203.7153 * Q.lam1 + 1.562749
    if 0.006189818 <= Q.lam1 < 0.007671243:
        z += -285.0645 * Q.lam1 + 2.066286
    if 0.007671243 <= Q.lam1 < 0.02457025:
        z += -81.34925 * Q.lam1 + 0.5035371
    if Q.lam1 >= 0.02457025:
        z += -126.3662 * Q.lam1 + 1.609615
    if Q.sum_z_dr2 < 0.01397874:
        z += 145.6749 * Q.sum_z_dr2 - 2.036353
    if Q.psi_0p3 >= 0.9853273:
        z += 16.46624 * Q.psi_0p3 - 16.22464
    if Q.sum_zz_dr2 < 0.001784301:
        z += -543.8124 * Q.sum_zz_dr2 + 5.815636
    if 0.001784301 <= Q.sum_zz_dr2 < 0.007511864:
        z += -332.1194 * Q.sum_zz_dr2 + 5.437912
    if 0.007511864 <= Q.sum_zz_dr2 < 0.02580859:
        z += -160.8526 * Q.sum_zz_dr2 + 4.151379
    if 0.01541561 <= Q.e2 < 0.04755309:
        z += -47.9296 * Q.e2 + 0.7388639
    if Q.e2 >= 0.04755309:
        z += -35.29948 * Q.e2 + 0.1382629
    if Q.lam2 < 0.0009731947:
        z += -214.3047 * Q.lam2 + 0.3806713
    if 0.0009731947 <= Q.lam2 < 0.001776308:
        z += -417.489 * Q.lam2 + 0.5784092
    if 0.001776308 <= Q.lam2 < 0.003687605:
        z += -203.1843 * Q.lam2 + 0.1977379
    if Q.lam2 >= 0.003687605:
        z += -165.8067 * Q.lam2 + 0.05990405
    if Q.lam1_plus_lam2 < 0.008190222:
        z += 518.2961 * Q.lam1_plus_lam2 - 5.199
    if 0.008190222 <= Q.lam1_plus_lam2 < 0.009614971:
        z += 669.6194 * Q.lam1_plus_lam2 - 6.438371
    if Q.sum_z_dr2_top15 < 0.001319197:
        z += -58.43985 * Q.sum_z_dr2_top15 + 0.4609547
    if 0.001319197 <= Q.sum_z_dr2_top15 < 0.007887677:
        z += -33.64565 * Q.sum_z_dr2_top15 + 0.4282463
    if Q.sum_z_dr2_top15 >= 0.007887677:
        z += 24.7942 * Q.sum_z_dr2_top15 - 0.03270843
    if Q.sum_z_dr2_top2 < 0.002412891:
        z += 76.27647 * Q.sum_z_dr2_top2 - 0.1840468
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -3.609529 * Q.z_dr_0p1_0p2 + 0.434384
    if Q.psi_0p1 >= 0.8509811:
        z += -1.616255 * Q.psi_0p1 + 1.375403
    if Q.sum_z_dr2_top20 < 0.00287991:
        z += -54.00309 * Q.sum_z_dr2_top20 - 0.5301591
    if 0.00287991 <= Q.sum_z_dr2_top20 < 0.01083435:
        z += 61.18903 * Q.sum_z_dr2_top20 - 0.861902
    if 0.01083435 <= Q.sum_z_dr2_top20 < 0.02265114:
        z += 16.83694 * Q.sum_z_dr2_top20 - 0.3813759
    if Q.LHA < 0.2091025:
        z += -1.439536 * Q.LHA + 0.3744897
    if 0.2091025 <= Q.LHA < 0.2601462:
        z += 3.355569 * Q.LHA - 0.6281788
    if Q.LHA >= 0.2601462:
        z += 4.795105 * Q.LHA - 1.002668
    if Q.z_top20_slots >= 0.7111557:
        z += -0.572319 * Q.z_top20_slots + 0.4070079
    if Q.sum_z_dr2_top40 < 0.008840538:
        z += -106.0684 * Q.sum_z_dr2_top40 - 0.6651825
    if 0.008840538 <= Q.sum_z_dr2_top40 < 0.02497133:
        z += 99.368 * Q.sum_z_dr2_top40 - 2.481351
    if Q.sum_z_dr2_top30 >= 0.008376291:
        z += 40.957 * Q.sum_z_dr2_top30 - 0.3430677
    if Q.sum_pt_top40 >= 1225.842:
        z += -0.002400893 * Q.sum_pt_top40 + 2.943116
    if 6.811175 <= Q.log_sum_pt < 7.139296:
        z += -2.710423 * Q.log_sum_pt + 18.46117
    if Q.log_sum_pt >= 7.139296:
        z += 0.2497373 * Q.log_sum_pt - 2.672295
    if Q.sum_pt_top20 >= 695.3094:
        z += 0.002497973 * Q.sum_pt_top20 - 1.736864
    if Q.e3 < 7.876005e-05:
        z += 2420.746 * Q.e3 - 0.1906581
    if Q.pt_6 < 19.46875:
        z += 0.06829587 * Q.pt_6 - 1.329635
    if Q.tau21_b2 < 0.2018786:
        z += -1.067179 * Q.tau21_b2 + 0.2154407
    if Q.tau1 >= 0.05444509:
        z += 2.158953 * Q.tau1 - 0.1175444
    if Q.z_6 < 0.01906139:
        z += -59.30468 * Q.z_6 + 1.130429
    if Q.sum_z_dr2 < 0.01397874 and Q.psi_0p3 > 0.9853273:
        z += -3838.367 * (0.01397874 - Q.sum_z_dr2) * (Q.psi_0p3 - 0.9853273)
    if Q.lam1 < 0.003811746 and Q.sum_pt_top50 < 1008.935:
        z += -4.807598 * (0.003811746 - Q.lam1) * (1008.935 - Q.sum_pt_top50)
    if Q.e2 > 0.04755309 and Q.zdr_0 > 0.0008718296:
        z += 297.9628 * (Q.e2 - 0.04755309) * (Q.zdr_0 - 0.0008718296)
    if Q.sum_z_dr2 < 0.01397874 and Q.soft1_pt > 1.521582:
        z += 25.9726 * (0.01397874 - Q.sum_z_dr2) * (Q.soft1_pt - 1.521582)
    if Q.e2 > 0.04755309 and Q.pt_14 < 26.14062:
        z += 0.896092 * (Q.e2 - 0.04755309) * (26.14062 - Q.pt_14)
    if Q.e3 < 0.0003372339 and Q.N2 < 0.3384815:
        z += -4873.96 * (0.0003372339 - Q.e3) * (0.3384815 - Q.N2)
    if Q.e2 > 0.04755309 and Q.z_dr_0_0p05 < 0.3289237:
        z += 52.83067 * (Q.e2 - 0.04755309) * (0.3289237 - Q.z_dr_0_0p05)
    if Q.psi_0p3 > 0.9853273 and Q.n_dr_0p1_0p2 > 15.0:
        z += 1.078582 * (Q.psi_0p3 - 0.9853273) * (Q.n_dr_0p1_0p2 - 15.0)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.z_dr_0p05_0p1 < 0.7108211:
        z += -4.403676 * (0.1203437 - Q.z_dr_0p1_0p2) * (0.7108211 - Q.z_dr_0p05_0p1)
    if Q.sum_z_dr2_top2 < 0.002412891 and Q.M2 > 0.03940774:
        z += 1051.912 * (0.002412891 - Q.sum_z_dr2_top2) * (Q.M2 - 0.03940774)
    if Q.sum_zz_dr2 < 0.007511864 and Q.sum_pt < 1007.788:
        z += 4.267296 * (0.007511864 - Q.sum_zz_dr2) * (1007.788 - Q.sum_pt)
    if Q.sum_zz_dr2 < 0.02580859 and Q.log_sum_pt < 6.98945:
        z += -171.6389 * (0.02580859 - Q.sum_zz_dr2) * (6.98945 - Q.log_sum_pt)
    if Q.LHA > 0.2091025 and Q.soft8_z < 0.001387595:
        z += -1960.262 * (Q.LHA - 0.2091025) * (0.001387595 - Q.soft8_z)
    if Q.sum_pt_top20 > 695.3094 and Q.D2_b2 < 1.36316:
        z += 0.0004299716 * (Q.sum_pt_top20 - 695.3094) * (1.36316 - Q.D2_b2)
    if Q.log_sum_pt > 6.811175 and Q.dr_max_012 > 0.1686705:
        z += -7.00605 * (Q.log_sum_pt - 6.811175) * (Q.dr_max_012 - 0.1686705)
    if Q.log_sum_pt > 6.811175 and Q.sum_z_dr2_top3 < 0.001592178:
        z += -750.9444 * (Q.log_sum_pt - 6.811175) * (0.001592178 - Q.sum_z_dr2_top3)
    if Q.z_top20_slots > 0.7111557 and Q.sum_z_dr2_top3 > 0.01002369:
        z += 91.75616 * (Q.z_top20_slots - 0.7111557) * (Q.sum_z_dr2_top3 - 0.01002369)
    if Q.e2 > 0.06524004 and Q.psi_0p3 > 0.9896594:
        z += 4151.507 * (Q.e2 - 0.06524004) * (Q.psi_0p3 - 0.9896594)
    return max(0.0, z)


def neuron_7(Q):
    z = -1.130054
    if Q.tau21_b2 < 0.1219401:
        z += -5.259331 * Q.tau21_b2 + 0.8659466
    if 0.1219401 <= Q.tau21_b2 < 0.2352054:
        z += -1.983159 * Q.tau21_b2 + 0.4664498
    if Q.sum_z_dr2 < 0.007877041:
        z += 621.9433 * Q.sum_z_dr2 - 4.899073
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.05274117 * Q.n_dr_0p2_0p4 + 0.7911175
    if Q.lam1 < 0.003811746:
        z += -262.9239 * Q.lam1 + 0.6549871
    if 0.003811746 <= Q.lam1 < 0.005913555:
        z += 100.3846 * Q.lam1 - 0.7298525
    if 0.005913555 <= Q.lam1 < 0.007671243:
        z += -95.75458 * Q.lam1 + 0.4300271
    if 0.007671243 <= Q.lam1 < 0.008241985:
        z += 533.5673 * Q.lam1 - 4.397654
    if Q.lam2 < 0.002396991:
        z += -173.6132 * Q.lam2 + 0.4161494
    if 0.9966167 <= Q.psi_0p3 < 0.9980008:
        z += 129.5098 * Q.psi_0p3 - 129.0716
    if 0.9980008 <= Q.psi_0p3 < 0.9995915:
        z += 253.6307 * Q.psi_0p3 - 252.9444
    if Q.psi_0p3 >= 0.9995915:
        z += -442.7069 * Q.psi_0p3 + 443.1088
    if Q.tau1 < 0.07708632:
        z += -41.19338 * Q.tau1 + 3.175446
    if Q.sum_zz_dr2 < 0.006936725:
        z += -636.9089 * Q.sum_zz_dr2 + 8.711351
    if 0.006936725 <= Q.sum_zz_dr2 < 0.007511864:
        z += -1431.787 * Q.sum_zz_dr2 + 14.2252
    if 0.007511864 <= Q.sum_zz_dr2 < 0.009606007:
        z += -1656.912 * Q.sum_zz_dr2 + 15.91631
    if Q.sum_z_dr < 0.05048381:
        z += -53.09715 * Q.sum_z_dr + 1.235077
    if 0.05048381 <= Q.sum_z_dr < 0.05660088:
        z += -48.46649 * Q.sum_z_dr + 1.001304
    if 0.05660088 <= Q.sum_z_dr < 0.08589404:
        z += 59.4658 * Q.sum_z_dr - 5.107758
    if Q.lam1_plus_lam2 < 0.01397874:
        z += -325.6432 * Q.lam1_plus_lam2 + 4.552082
    if Q.sum_z_dr2_top50 < 0.007820315:
        z += 841.9709 * Q.sum_z_dr2_top50 - 6.99505
    if 0.007820315 <= Q.sum_z_dr2_top50 < 0.008124776:
        z += 1348.52 * Q.sum_z_dr2_top50 - 10.95642
    if Q.tau2 < 0.04828819:
        z += 20.77393 * Q.tau2 - 1.003135
    if Q.LHA < 0.2454112:
        z += 32.57185 * Q.LHA - 7.221405
    if 0.2454112 <= Q.LHA < 0.2601462:
        z += -5.398766 * Q.LHA + 2.097011
    if 0.2601462 <= Q.LHA < 0.302389:
        z += -16.39431 * Q.LHA + 4.95746
    if Q.e2 < 0.03480688:
        z += 5.486576 * Q.e2 - 0.6577503
    if 0.03480688 <= Q.e2 < 0.04358622:
        z += 53.16796 * Q.e2 - 2.31739
    if 0.2404747 <= Q.max_dr < 0.2982:
        z += -8.029031 * Q.max_dr + 1.930779
    if Q.max_dr >= 0.2982:
        z += -0.2361374 * Q.max_dr - 0.3930622
    if Q.mean_eta2 < 0.0009793444:
        z += 205.595 * Q.mean_eta2 - 0.2013483
    if Q.tau21 >= 0.4281458:
        z += -0.9545437 * Q.tau21 + 0.4086839
    if Q.sum_z_dr2_top30 < 0.00375223:
        z += 1475.753 * Q.sum_z_dr2_top30 - 6.519101
    if 0.00375223 <= Q.sum_z_dr2_top30 < 0.005809485:
        z += 350.8388 * Q.sum_z_dr2_top30 - 2.298166
    if 0.005809485 <= Q.sum_z_dr2_top30 < 0.006363916:
        z += 75.04016 * Q.sum_z_dr2_top30 - 0.6959177
    if 0.006363916 <= Q.sum_z_dr2_top30 < 0.008376291:
        z += 435.0235 * Q.sum_z_dr2_top30 - 2.986822
    if 0.008376291 <= Q.sum_z_dr2_top30 < 0.01215787:
        z += -173.7532 * Q.sum_z_dr2_top30 + 2.112469
    if Q.sum_z_dr2_top5 < 0.0168592:
        z += -8.901055 * Q.sum_z_dr2_top5 + 0.1500646
    if Q.n_pt_above_5 < 36.0:
        z += 0.01634607 * Q.n_pt_above_5 - 0.5884584
    if Q.sum_z_dr2_top10 < 0.00130722:
        z += 119.0666 * Q.sum_z_dr2_top10 - 0.1556462
    if Q.psi_0p2 >= 0.9087063:
        z += -2.829209 * Q.psi_0p2 + 2.57092
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1937688:
        z += -60.91626 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1937688)
    if Q.tau21_b2 < 0.2352054 and Q.sum_zz_dr2 < 0.007872294:
        z += -2343.51 * (0.2352054 - Q.tau21_b2) * (0.007872294 - Q.sum_zz_dr2)
    if Q.tau21_b2 < 0.2352054 and Q.sum_zz_dr2 < 0.01396296:
        z += 569.7915 * (0.2352054 - Q.tau21_b2) * (0.01396296 - Q.sum_zz_dr2)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt_top30 > 978.0762:
        z += 0.01820264 * (0.2352054 - Q.tau21_b2) * (Q.sum_pt_top30 - 978.0762)
    if Q.tau21_b2 < 0.2352054 and Q.log_sum_pt > 7.139296:
        z += -52.15116 * (0.2352054 - Q.tau21_b2) * (Q.log_sum_pt - 7.139296)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.sum_pt_top40 < 1225.842:
        z += 7.171494e-05 * (15.0 - Q.n_dr_0p2_0p4) * (1225.842 - Q.sum_pt_top40)
    if Q.sum_z_dr2 < 0.007877041 and Q.z_dr_0_0p05 < 0.3289237:
        z += -622.3704 * (0.007877041 - Q.sum_z_dr2) * (0.3289237 - Q.z_dr_0_0p05)
    if Q.sum_zz_dr2 < 0.009606007 and Q.sum_z_dr2_top10 > 0.00625621:
        z += 63069.84 * (0.009606007 - Q.sum_zz_dr2) * (Q.sum_z_dr2_top10 - 0.00625621)
    if Q.lam1 < 0.005913555 and Q.sum_pt_top40 < 1139.734:
        z += 5.173244 * (0.005913555 - Q.lam1) * (1139.734 - Q.sum_pt_top40)
    if Q.psi_0p3 > 0.9980008 and Q.dr_0 < 0.0361727:
        z += -6435.697 * (Q.psi_0p3 - 0.9980008) * (0.0361727 - Q.dr_0)
    if Q.sum_zz_dr2 < 0.009606007 and Q.sum_pt < 1167.447:
        z += -7.164799 * (0.009606007 - Q.sum_zz_dr2) * (1167.447 - Q.sum_pt)
    if Q.lam1 < 0.005913555 and Q.sum_pt < 1167.447:
        z += -3.292685 * (0.005913555 - Q.lam1) * (1167.447 - Q.sum_pt)
    if Q.lam2 < 0.002396991 and Q.log_sum_pt < 7.139296:
        z += 654.2783 * (0.002396991 - Q.lam2) * (7.139296 - Q.log_sum_pt)
    if Q.psi_0p3 > 0.9980008 and Q.n_pt_above_10 > 20.0:
        z += -21.0607 * (Q.psi_0p3 - 0.9980008) * (Q.n_pt_above_10 - 20.0)
    if Q.lam2 < 0.002396991 and Q.zdr_0 > 0.002524869:
        z += -9248.689 * (0.002396991 - Q.lam2) * (Q.zdr_0 - 0.002524869)
    if Q.tau21_b2 < 0.2352054 and Q.orientation_deg > 17.2994:
        z += -0.02087884 * (0.2352054 - Q.tau21_b2) * (Q.orientation_deg - 17.2994)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.n_dr_0p1_0p2 > 21.0:
        z += -0.00296959 * (15.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 21.0)
    if Q.sum_z_dr2_top50 < 0.007820315 and Q.soft5_pt > 0.5297852:
        z += -10.09416 * (0.007820315 - Q.sum_z_dr2_top50) * (Q.soft5_pt - 0.5297852)
    if Q.sum_z_dr < 0.08589404 and Q.sum_pt_top50 < 1156.659:
        z += 0.2167435 * (0.08589404 - Q.sum_z_dr) * (1156.659 - Q.sum_pt_top50)
    if Q.sum_pt_top15 > 1082.548 and Q.mean_phi > 2.298159e-05:
        z += -264.7791 * (Q.sum_pt_top15 - 1082.548) * (Q.mean_phi - 2.298159e-05)
    if Q.psi_0p3 > 0.9966167 and Q.soft1_pt < 2.275391:
        z += 48.37468 * (Q.psi_0p3 - 0.9966167) * (2.275391 - Q.soft1_pt)
    if Q.sum_pt_top15 > 1082.548 and Q.soft4_dr0 > 0.02345786:
        z += -0.01189867 * (Q.sum_pt_top15 - 1082.548) * (Q.soft4_dr0 - 0.02345786)
    if Q.psi_0p3 > 0.9980008 and Q.C2_b2 < 0.004298863:
        z += -27500.47 * (Q.psi_0p3 - 0.9980008) * (0.004298863 - Q.C2_b2)
    if Q.sum_pt_top15 > 1082.548 and Q.C2_b2 < 0.003468228:
        z += 2.340081 * (Q.sum_pt_top15 - 1082.548) * (0.003468228 - Q.C2_b2)
    if Q.sum_pt_top15 > 1082.548 and Q.mean_phi > -3.355129e-05:
        z += 88.27534 * (Q.sum_pt_top15 - 1082.548) * (Q.mean_phi - -3.355129e-05)
    if Q.psi_0p3 > 0.9966167 and Q.soft8_dr0 < 0.3403782:
        z += 136.0907 * (Q.psi_0p3 - 0.9966167) * (0.3403782 - Q.soft8_dr0)
    if Q.sum_pt_top15 > 1082.548 and Q.zdr_6 < 0.002333533:
        z += -0.8057508 * (Q.sum_pt_top15 - 1082.548) * (0.002333533 - Q.zdr_6)
    if Q.sum_z_dr2 < 0.007877041 and Q.zdr_6 < 0.004600222:
        z += 11542.36 * (0.007877041 - Q.sum_z_dr2) * (0.004600222 - Q.zdr_6)
    if Q.lam1 < 0.008241985 and Q.soft1_z > 0.0007833979:
        z += -86659.62 * (0.008241985 - Q.lam1) * (Q.soft1_z - 0.0007833979)
    return max(0.0, z)


def neuron_8(Q):
    z = 1.113505
    if Q.sum_z_dr2 < 0.003638856:
        z += 56.39335 * Q.sum_z_dr2 + 0.1146999
    if 0.003638856 <= Q.sum_z_dr2 < 0.01397874:
        z += 152.823 * Q.sum_z_dr2 - 0.2361937
    if 0.01397874 <= Q.sum_z_dr2 < 0.0258669:
        z += -159.8296 * Q.sum_z_dr2 + 4.134297
    if Q.n_dr_0p2_0p4 < 6.0:
        z += 0.07116557 * Q.n_dr_0p2_0p4 - 0.9192383
    if 6.0 <= Q.n_dr_0p2_0p4 < 18.0:
        z += 0.04102041 * Q.n_dr_0p2_0p4 - 0.7383673
    if Q.sum_zz_dr2 < 0.002574843:
        z += 307.6745 * Q.sum_zz_dr2 - 4.380803
    if 0.002574843 <= Q.sum_zz_dr2 < 0.009606007:
        z += 510.3834 * Q.sum_zz_dr2 - 4.902747
    if Q.sum_pt < 1028.184:
        z += -0.01329055 * Q.sum_pt + 13.89196
    if 1028.184 <= Q.sum_pt < 1085.125:
        z += -0.003983706 * Q.sum_pt + 4.322819
    if Q.sum_z_dr2_top50 < 0.01351431:
        z += -29.93791 * Q.sum_z_dr2_top50 + 0.4045902
    if Q.sj2_dr >= 0.2232169:
        z += 1.495406 * Q.sj2_dr - 0.3337998
    if Q.sum_pt_top50 < 1008.935:
        z += -0.01038845 * Q.sum_pt_top50 + 12.11744
    if 1008.935 <= Q.sum_pt_top50 < 1156.659:
        z += -0.007889169 * Q.sum_pt_top50 + 9.595832
    if 1156.659 <= Q.sum_pt_top50 < 1245.697:
        z += -0.005287111 * Q.sum_pt_top50 + 6.586136
    if Q.lam1_plus_lam2 < 0.008190222:
        z += -170.9786 * Q.lam1_plus_lam2 + 1.400353
    if Q.log_sum_pt < 6.811175:
        z += 7.870735 * Q.log_sum_pt - 53.60896
    if Q.n_dr_0p1_0p2 >= 21.0:
        z += 0.02183961 * Q.n_dr_0p1_0p2 - 0.4586317
    if Q.psi_0p3 >= 0.9943058:
        z += -19.63062 * Q.psi_0p3 + 19.51884
    if Q.tau1 < 0.1219132:
        z += -21.33038 * Q.tau1 + 2.600456
    if Q.D2 < 1.788105:
        z += -0.2485752 * Q.D2 + 0.4444786
    if Q.e2 < 0.02210818:
        z += 30.67671 * Q.e2 - 0.6782063
    if 0.005788041 <= Q.sum_z_dr2_top15 < 0.007887677:
        z += -19.12595 * Q.sum_z_dr2_top15 + 0.1107018
    if Q.sum_z_dr2_top15 >= 0.007887677:
        z += 17.86732 * Q.sum_z_dr2_top15 - 0.1810892
    if Q.sum_z_dr2_top40 < 0.008840538:
        z += -105.4334 * Q.sum_z_dr2_top40 + 0.9320879
    if 926.0902 <= Q.sum_pt_top20 < 1017.778:
        z += 0.000447263 * Q.sum_pt_top20 - 0.4142059
    if Q.sum_pt_top20 >= 1017.778:
        z += -0.0001639587 * Q.sum_pt_top20 + 0.2078822
    if Q.sum_z_dr2 < 0.0258669 and Q.log_sum_pt < 7.139296:
        z += -778.7054 * (0.0258669 - Q.sum_z_dr2) * (7.139296 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.0258669 and Q.z_dr_0p1_0p2 > 0.3340477:
        z += 112.8978 * (0.0258669 - Q.sum_z_dr2) * (Q.z_dr_0p1_0p2 - 0.3340477)
    if Q.sum_z_dr2_top30 < 0.007463985 and Q.sum_pt < 1260.541:
        z += 1.308988 * (0.007463985 - Q.sum_z_dr2_top30) * (1260.541 - Q.sum_pt)
    if Q.sum_z_dr2_top30 < 0.007463985 and Q.sum_pt_top40 < 1095.686:
        z += -1.358168 * (0.007463985 - Q.sum_z_dr2_top30) * (1095.686 - Q.sum_pt_top40)
    if Q.sum_z_dr2 < 0.01397874 and Q.sum_pt < 986.0565:
        z += 1.360904 * (0.01397874 - Q.sum_z_dr2) * (986.0565 - Q.sum_pt)
    if Q.sum_z_dr2 < 0.0258669 and Q.z_top50_slots > 0.9704436:
        z += -1410.653 * (0.0258669 - Q.sum_z_dr2) * (Q.z_top50_slots - 0.9704436)
    if Q.sum_pt_top50 < 1245.697 and Q.n_for_90pct > 39.0:
        z += -0.0002056502 * (1245.697 - Q.sum_pt_top50) * (Q.n_for_90pct - 39.0)
    if Q.tau1 < 0.1219132 and Q.planar_flow < 0.5364935:
        z += -15.54542 * (0.1219132 - Q.tau1) * (0.5364935 - Q.planar_flow)
    if Q.sum_pt_top50 < 1008.935 and Q.n_dr_0p05_0p1 < 21.0:
        z += -0.0001338319 * (1008.935 - Q.sum_pt_top50) * (21.0 - Q.n_dr_0p05_0p1)
    if Q.log_sum_pt < 6.811175 and Q.zdr_0 < 0.01778043:
        z += 236.7875 * (6.811175 - Q.log_sum_pt) * (0.01778043 - Q.zdr_0)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.sj2_zsoft > 0.1181474:
        z += 0.07148895 * (18.0 - Q.n_dr_0p2_0p4) * (Q.sj2_zsoft - 0.1181474)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.9296614
    if Q.sum_z_dr2_top40 < 0.005712208:
        z += -91.03817 * Q.sum_z_dr2_top40 + 0.5200289
    if Q.sum_pt_top40 < 972.4111:
        z += -0.01147025 * Q.sum_pt_top40 + 11.1538
    if Q.sum_z_dr2 >= 0.0258669:
        z += -180.905 * Q.sum_z_dr2 + 4.679452
    if Q.LHA >= 0.3332345:
        z += 9.214046 * Q.LHA - 3.070438
    if Q.lam1 < 0.004673423:
        z += -99.94787 * Q.lam1 + 0.7255503
    if 0.004673423 <= Q.lam1 < 0.006716737:
        z += 139.9539 * Q.lam1 - 0.3956124
    if 0.006716737 <= Q.lam1 < 0.007259287:
        z += -48.69656 * Q.lam1 + 0.8715033
    if Q.lam1 >= 0.007259287:
        z += 51.25131 * Q.lam1 + 0.145953
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -4.231569 * Q.z_dr_0_0p05 + 3.719151
    if Q.sum_pt_top30 < 1191.938:
        z += -0.001518008 * Q.sum_pt_top30 + 1.809372
    if Q.lam1_plus_lam2 < 0.006941794:
        z += -538.1319 * Q.lam1_plus_lam2 + 3.735601
    if Q.sum_z_dr2_top15 < 0.001319197:
        z += 113.8872 * Q.sum_z_dr2_top15 - 0.5227425
    if 0.001319197 <= Q.sum_z_dr2_top15 < 0.004169954:
        z += -56.87281 * Q.sum_z_dr2_top15 - 0.2974764
    if 0.004169954 <= Q.sum_z_dr2_top15 < 0.02146578:
        z += 23.67344 * Q.sum_z_dr2_top15 - 0.6333506
    if 0.02146578 <= Q.sum_z_dr2_top15 < 0.02675364:
        z += -10.63072 * Q.sum_z_dr2_top15 + 0.103015
    if Q.sum_z_dr2_top15 >= 0.02675364:
        z += -34.30416 * Q.sum_z_dr2_top15 + 0.7363656
    if Q.n_pt_above_1 < 26.0:
        z += -0.01895888 * Q.n_pt_above_1 + 0.4929308
    if Q.log_sum_pt < 6.903423:
        z += 15.69329 * Q.log_sum_pt - 108.3374
    if Q.sum_pt < 986.0565:
        z += -0.01232382 * Q.sum_pt + 12.15198
    if Q.sum_z_dr >= 0.1207452:
        z += -42.31908 * Q.sum_z_dr + 5.109825
    if Q.psi_0p3 >= 0.9956185:
        z += 37.92721 * Q.psi_0p3 - 37.76103
    if Q.n_dr_0p1_0p2 >= 21.0:
        z += 0.01626696 * Q.n_dr_0p1_0p2 - 0.3416063
    if Q.e3 >= 0.0005178279:
        z += 1342.098 * Q.e3 - 0.6949755
    if Q.sum_z_dr2_top20 < 0.01083435:
        z += -56.96062 * Q.sum_z_dr2_top20 + 0.6171315
    if Q.sum_zz_dr2 >= 0.01989454:
        z += -74.76545 * Q.sum_zz_dr2 + 1.487424
    if Q.eccentricity >= 0.7333655:
        z += -0.4675409 * Q.eccentricity + 0.3428784
    if Q.pt_entropy < 2.485823:
        z += -0.4316649 * Q.pt_entropy + 1.073042
    if Q.tau1 < 0.06310829:
        z += 10.03702 * Q.tau1 - 0.6334189
    if Q.sum_z_dr2_top40 < 0.005712208 and Q.sum_pt < 1007.788:
        z += -3.370994 * (0.005712208 - Q.sum_z_dr2_top40) * (1007.788 - Q.sum_pt)
    if Q.sum_z_dr2_top40 < 0.005712208 and Q.log_sum_pt > 7.017258:
        z += 1184.656 * (0.005712208 - Q.sum_z_dr2_top40) * (Q.log_sum_pt - 7.017258)
    if Q.sum_pt_top40 < 972.4111 and Q.soft4_pt > 1.106445:
        z += -0.003574437 * (972.4111 - Q.sum_pt_top40) * (Q.soft4_pt - 1.106445)
    if Q.LHA > 0.3332345 and Q.sum_pt < 1042.609:
        z += 0.1372462 * (Q.LHA - 0.3332345) * (1042.609 - Q.sum_pt)
    if Q.lam1 < 0.007259287 and Q.sum_pt_top30 < 886.3438:
        z += 1.240779 * (0.007259287 - Q.lam1) * (886.3438 - Q.sum_pt_top30)
    if Q.lam1 < 0.007259287 and Q.sum_pt > 1085.125:
        z += -1.195295 * (0.007259287 - Q.lam1) * (Q.sum_pt - 1085.125)
    if Q.z_dr_0p1_0p2 > 0.6882177 and Q.sum_pt < 907.9372:
        z += -0.04095863 * (Q.z_dr_0p1_0p2 - 0.6882177) * (907.9372 - Q.sum_pt)
    if Q.z_dr_0_0p05 > 0.878906 and Q.sum_pt_top40 > 1013.042:
        z += 0.02314221 * (Q.z_dr_0_0p05 - 0.878906) * (Q.sum_pt_top40 - 1013.042)
    if Q.sum_pt_top40 < 972.4111 and Q.sum_pt > 907.9372:
        z += -4.688848e-05 * (972.4111 - Q.sum_pt_top40) * (Q.sum_pt - 907.9372)
    if Q.sum_pt_top40 < 972.4111 and Q.n_dr_0_0p05 < 20.0:
        z += -0.0001740983 * (972.4111 - Q.sum_pt_top40) * (20.0 - Q.n_dr_0_0p05)
    if Q.lam1 < 0.007259287 and Q.n_particles > 36.0:
        z += 3.508068 * (0.007259287 - Q.lam1) * (Q.n_particles - 36.0)
    if Q.sum_pt_top40 < 972.4111 and Q.tau3 < 0.0369869:
        z += -0.08417094 * (972.4111 - Q.sum_pt_top40) * (0.0369869 - Q.tau3)
    if Q.sum_z_dr2_top15 < 0.004169954 and Q.n_dr_0p2_0p4 > 8.0:
        z += 11.69745 * (0.004169954 - Q.sum_z_dr2_top15) * (Q.n_dr_0p2_0p4 - 8.0)
    if Q.sum_pt < 986.0565 and Q.soft4_pt > 1.226562:
        z += 0.004309376 * (986.0565 - Q.sum_pt) * (Q.soft4_pt - 1.226562)
    if Q.log_sum_pt < 6.903423 and Q.z_dr_0p1_0p2 < 0.4479367:
        z += 10.59731 * (6.903423 - Q.log_sum_pt) * (0.4479367 - Q.z_dr_0p1_0p2)
    if Q.log_sum_pt < 6.903423 and Q.soft4_pt > 1.789258:
        z += -0.8898726 * (6.903423 - Q.log_sum_pt) * (Q.soft4_pt - 1.789258)
    if Q.sum_z_dr > 0.1207452 and Q.sj3_pairmin_over_m < 0.4606099:
        z += 21.66061 * (Q.sum_z_dr - 0.1207452) * (0.4606099 - Q.sj3_pairmin_over_m)
    if Q.log_sum_pt < 6.903423 and Q.dr_4 < 0.0830523:
        z += 27.63469 * (6.903423 - Q.log_sum_pt) * (0.0830523 - Q.dr_4)
    if Q.sum_z_dr > 0.1207452 and Q.soft5_pt < 2.894531:
        z += 22.28757 * (Q.sum_z_dr - 0.1207452) * (2.894531 - Q.soft5_pt)
    if Q.sum_zz_dr2 > 0.0292152 and Q.soft7_pt < 3.779492:
        z += -35.25073 * (Q.sum_zz_dr2 - 0.0292152) * (3.779492 - Q.soft7_pt)
    if Q.LHA > 0.3332345 and Q.soft5_pt < 2.117188:
        z += -7.147416 * (Q.LHA - 0.3332345) * (2.117188 - Q.soft5_pt)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.796321
    if Q.sum_z_dr < 0.03577037:
        z += 122.6505 * Q.sum_z_dr - 8.798307
    if 0.03577037 <= Q.sum_z_dr < 0.1207452:
        z += 51.91012 * Q.sum_z_dr - 6.267896
    if 0.01397874 <= Q.sum_z_dr2 < 0.02928196:
        z += -112.4743 * Q.sum_z_dr2 + 1.57225
    if Q.sum_z_dr2 >= 0.02928196:
        z += -160.8611 * Q.sum_z_dr2 + 2.98911
    if Q.soft2_pt >= 0.291748:
        z += -0.0744395 * Q.soft2_pt + 0.02171758
    if Q.e2 >= 0.006720044:
        z += 41.88459 * Q.e2 - 0.2814663
    if Q.sum_zz_dr2 < 0.00592208:
        z += -404.279 * Q.sum_zz_dr2 + 2.060096
    if 0.00592208 <= Q.sum_zz_dr2 < 0.00818374:
        z += -52.3927 * Q.sum_zz_dr2 - 0.02380293
    if 0.00818374 <= Q.sum_zz_dr2 < 0.009606007:
        z += 318.204 * Q.sum_zz_dr2 - 3.05667
    if Q.sum_zz_dr2 >= 0.02580859:
        z += -325.1227 * Q.sum_zz_dr2 + 8.390957
    if Q.psi_0p3 >= 0.9985421:
        z += -104.1401 * Q.psi_0p3 + 103.9883
    if Q.sum_pt < 907.9372:
        z += 0.006818673 * Q.sum_pt - 6.630476
    if 907.9372 <= Q.sum_pt < 949.9169:
        z += 0.01047052 * Q.sum_pt - 9.946125
    if Q.z_dr_0_0p05 >= 0.7128619:
        z += -1.313922 * Q.z_dr_0_0p05 + 0.9366449
    if Q.M2 < 0.08222447:
        z += -10.60454 * Q.M2 + 0.8719524
    if Q.dr_max_012 >= 0.06112084:
        z += 0.6922557 * Q.dr_max_012 - 0.04231125
    if Q.lam1_plus_lam2 < 0.006403325:
        z += -97.63231 * Q.lam1_plus_lam2 + 0.6251714
    if Q.n_dr_0_0p05 >= 6.0:
        z += 0.008441648 * Q.n_dr_0_0p05 - 0.05064989
    if Q.sum_pt_top3 < 656.3844:
        z += -0.000735976 * Q.sum_pt_top3 + 0.4830831
    if Q.tau21_b2 < 0.1722488:
        z += 2.727576 * Q.tau21_b2 - 0.581366
    if 0.1722488 <= Q.tau21_b2 < 0.3062621:
        z += 0.832337 * Q.tau21_b2 - 0.2549133
    if Q.dr_0 < 0.06413297:
        z += -7.492969 * Q.dr_0 + 0.4805464
    if Q.D2 < 2.410481:
        z += -0.2954029 * Q.D2 + 0.7120632
    if Q.e3 < 0.0005178279:
        z += 1709.634 * Q.e3 - 0.8852962
    if Q.tau4 < 0.0259543:
        z += 9.362979 * Q.tau4 - 0.2430095
    if Q.pt_entropy < 3.539367:
        z += 0.6801467 * Q.pt_entropy - 2.407289
    if Q.sum_z_dr2_top10 < 0.002247756:
        z += 124.6072 * Q.sum_z_dr2_top10 - 0.9930322
    if 0.002247756 <= Q.sum_z_dr2_top10 < 0.007678544:
        z += 83.84573 * Q.sum_z_dr2_top10 - 0.9014103
    if 0.007678544 <= Q.sum_z_dr2_top10 < 0.01976735:
        z += 21.30873 * Q.sum_z_dr2_top10 - 0.4212172
    if Q.n_dr_0p2_0p4 < 15.0:
        z += 0.05313752 * Q.n_dr_0p2_0p4 - 0.7970628
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -4.276308 * Q.z_dr_0p2_0p4 + 0.3901091
    if Q.LHA < 0.1870291:
        z += -11.63975 * Q.LHA + 2.726429
    if 0.1870291 <= Q.LHA < 0.3719813:
        z += -2.9708 * Q.LHA + 1.105082
    if Q.sum_z_dr2_top40 >= 0.02497133:
        z += 154.7049 * Q.sum_z_dr2_top40 - 3.863187
    if 0.005718442 <= Q.sum_z_dr2_top20 < 0.008031209:
        z += -173.4949 * Q.sum_z_dr2_top20 + 0.9921204
    if Q.sum_z_dr2_top20 >= 0.008031209:
        z += -26.24686 * Q.sum_z_dr2_top20 - 0.1904592
    if Q.tau1 < 0.1507173:
        z += -23.58273 * Q.tau1 + 3.554325
    if Q.sd_rg < 0.1690338:
        z += 0.9054673 * Q.sd_rg - 0.1530546
    if Q.dr_1 < 0.06108421:
        z += -3.466558 * Q.dr_1 + 0.211752
    if Q.zdr_2 < 0.006212866:
        z += -28.46527 * Q.zdr_2 + 0.1768509
    if Q.lam1 < 0.006189818:
        z += 75.09553 * Q.lam1 - 0.4648277
    if Q.e2 > 0.006720044 and Q.log_sum_pt > 6.893714:
        z += -99.82114 * (Q.e2 - 0.006720044) * (Q.log_sum_pt - 6.893714)
    if Q.e2 > 0.006720044 and Q.sj3_pairmin_over_m > 0.1765064:
        z += 18.78426 * (Q.e2 - 0.006720044) * (Q.sj3_pairmin_over_m - 0.1765064)
    if Q.sum_pt < 907.9372 and Q.z_dr_0p05_0p1 < 0.2992503:
        z += 0.0158351 * (907.9372 - Q.sum_pt) * (0.2992503 - Q.z_dr_0p05_0p1)
    if Q.psi_0p3 > 0.9985421 and Q.n_real_top40 > 26.0:
        z += -8.858877 * (Q.psi_0p3 - 0.9985421) * (Q.n_real_top40 - 26.0)
    if Q.sum_z_dr < 0.1207452 and Q.pt1_dr01 > 12.6865:
        z += 0.5032899 * (0.1207452 - Q.sum_z_dr) * (Q.pt1_dr01 - 12.6865)
    if Q.psi_0p3 > 0.9985421 and Q.n_dr_0p1_0p2 > 19.0:
        z += 15.01226 * (Q.psi_0p3 - 0.9985421) * (Q.n_dr_0p1_0p2 - 19.0)
    if Q.sum_z_dr < 0.1207452 and Q.tau32 < 0.8476928:
        z += -6.989105 * (0.1207452 - Q.sum_z_dr) * (0.8476928 - Q.tau32)
    if Q.sum_z_dr < 0.1207452 and Q.ptdr0_3 > 7.407874:
        z += 0.3593118 * (0.1207452 - Q.sum_z_dr) * (Q.ptdr0_3 - 7.407874)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.n_real_top40 < 40.0:
        z += 0.001745364 * (15.0 - Q.n_dr_0p2_0p4) * (40.0 - Q.n_real_top40)
    if Q.sum_z_dr2 > 0.01397874 and Q.pt_3 < 114.75:
        z += 0.7558168 * (Q.sum_z_dr2 - 0.01397874) * (114.75 - Q.pt_3)
    if Q.sum_pt < 949.9169 and Q.zdr_4 < 0.004495205:
        z += 2.750731 * (949.9169 - Q.sum_pt) * (0.004495205 - Q.zdr_4)
    if Q.soft2_pt > 0.291748 and Q.dr1_9 > 0.1110681:
        z += -0.4876764 * (Q.soft2_pt - 0.291748) * (Q.dr1_9 - 0.1110681)
    if Q.sum_pt < 907.9372 and Q.zdr_4 < 0.003932029:
        z += -3.096993 * (907.9372 - Q.sum_pt) * (0.003932029 - Q.zdr_4)
    if Q.sum_pt < 949.9169 and Q.eccentricity > 0.6414784:
        z += -0.01414776 * (949.9169 - Q.sum_pt) * (Q.eccentricity - 0.6414784)
    if Q.sum_pt < 949.9169 and Q.absphi_3 < 0.05648804:
        z += 0.03900667 * (949.9169 - Q.sum_pt) * (0.05648804 - Q.absphi_3)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.max_dr > 0.2404747:
        z += 0.08082325 * (15.0 - Q.n_dr_0p2_0p4) * (Q.max_dr - 0.2404747)
    if Q.sum_pt < 949.9169 and Q.dr0_11 > 0.2612223:
        z += 0.05292931 * (949.9169 - Q.sum_pt) * (Q.dr0_11 - 0.2612223)
    if Q.sum_z_dr2_top30 < 0.005402331 and Q.pt_6 > 33.6875:
        z += -3.000512 * (0.005402331 - Q.sum_z_dr2_top30) * (Q.pt_6 - 33.6875)
    if Q.sum_z_dr < 0.1207452 and Q.ptdr0_2 > 7.740999:
        z += 0.3201677 * (0.1207452 - Q.sum_z_dr) * (Q.ptdr0_2 - 7.740999)
    if Q.sum_pt < 949.9169 and Q.dr0_4 > 0.2821771:
        z += 0.06467429 * (949.9169 - Q.sum_pt) * (Q.dr0_4 - 0.2821771)
    if Q.sum_pt_top3 < 656.3844 and Q.eccentricity < 0.9768165:
        z += 0.001050505 * (656.3844 - Q.sum_pt_top3) * (0.9768165 - Q.eccentricity)
    if Q.sum_pt < 907.9372 and Q.dr0_11 > 0.07072039:
        z += -0.02115681 * (907.9372 - Q.sum_pt) * (Q.dr0_11 - 0.07072039)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.3126185
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.00755056 * Q.n_dr_0p2_0p4 - 0.1462739
    if 6.0 <= Q.n_dr_0p2_0p4 < 10.0:
        z += 0.04789433 * Q.n_dr_0p2_0p4 - 0.4789433
    if Q.sum_zz_dr2 < 0.004754578:
        z += -430.2883 * Q.sum_zz_dr2 + 3.814107
    if 0.004754578 <= Q.sum_zz_dr2 < 0.007511864:
        z += -623.0339 * Q.sum_zz_dr2 + 4.730531
    if 0.007511864 <= Q.sum_zz_dr2 < 0.00818374:
        z += -481.8659 * Q.sum_zz_dr2 + 3.670097
    if 0.00818374 <= Q.sum_zz_dr2 < 0.009606007:
        z += 192.2063 * Q.sum_zz_dr2 - 1.846335
    if 0.9313699 <= Q.psi_0p2 < 0.9935324:
        z += 2.235246 * Q.psi_0p2 - 2.081841
    if Q.psi_0p2 >= 0.9935324:
        z += 35.0099 * Q.psi_0p2 - 34.64452
    if Q.lam1 < 0.005913555:
        z += 116.8316 * Q.lam1 - 0.6908902
    if Q.sum_z_dr2_top50 < 0.00634935:
        z += 252.3102 * Q.sum_z_dr2_top50 - 1.602006
    if Q.e2 < 0.02210818:
        z += -24.09491 * Q.e2 - 0.08101187
    if 0.02210818 <= Q.e2 < 0.02515919:
        z += -1.029758 * Q.e2 - 0.5909404
    if 0.02515919 <= Q.e2 < 0.03875945:
        z += 45.35564 * Q.e2 - 1.75796
    if Q.z_top30_slots >= 0.9734886:
        z += -9.104729 * Q.z_top30_slots + 8.86335
    if Q.D2 < 4.450169:
        z += 0.1155051 * Q.D2 - 0.5140171
    if Q.C2_b2 >= 0.002289486:
        z += -3.080549 * Q.C2_b2 + 0.007052875
    if Q.sum_z_dr2_top30 < 0.005809485:
        z += 58.01575 * Q.sum_z_dr2_top30 - 0.05932187
    if 0.005809485 <= Q.sum_z_dr2_top30 < 0.006363916:
        z += 342.5386 * Q.sum_z_dr2_top30 - 1.712253
    if 0.006363916 <= Q.sum_z_dr2_top30 < 0.01215787:
        z += -80.7106 * Q.sum_z_dr2_top30 + 0.9812692
    if Q.zdr_0 < 0.005911134:
        z += 56.74522 * Q.zdr_0 - 0.3354286
    if Q.sum_z_dr2_top5 < 0.00327978:
        z += -52.2189 * Q.sum_z_dr2_top5 - 0.2005665
    if 0.00327978 <= Q.sum_z_dr2_top5 < 0.007164202:
        z += 95.72415 * Q.sum_z_dr2_top5 - 0.6857872
    if Q.n_real_top50 >= 22.0:
        z += -0.02045073 * Q.n_real_top50 + 0.449916
    if Q.sum_z_dr2_top10 < 0.00130722:
        z += -250.7372 * Q.sum_z_dr2_top10 + 0.3277688
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.02312227 * Q.n_dr_0p1_0p2 - 0.1849781
    if Q.sum_pt_top15 >= 840.7969:
        z += -0.001592638 * Q.sum_pt_top15 + 1.339085
    if Q.sum_pt_top50 >= 1245.697:
        z += 0.001060213 * Q.sum_pt_top50 - 1.320704
    if Q.dr_0 < 0.02645404:
        z += -7.703964 * Q.dr_0 + 0.203801
    if Q.sum_z_dr < 0.07374472:
        z += 19.48285 * Q.sum_z_dr - 1.54283
    if 0.07374472 <= Q.sum_z_dr < 0.09749958:
        z += 4.465307 * Q.sum_z_dr - 0.4353656
    if Q.LHA < 0.3098384:
        z += -2.264711 * Q.LHA + 0.9522989
    if 0.3098384 <= Q.LHA < 0.3332345:
        z += -10.71135 * Q.LHA + 3.569391
    if Q.tau21_b2 < 0.2352054:
        z += -0.9621546 * Q.tau21_b2 + 0.226304
    if Q.tau1 < 0.1008497:
        z += -9.997581 * Q.tau1 + 1.16778
    if 0.1008497 <= Q.tau1 < 0.1072713:
        z += -24.84247 * Q.tau1 + 2.664883
    if Q.n_dr_0p2_0p4 < 10.0 and Q.sum_z_dr2 < 0.005532208:
        z += -34.00751 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.sum_z_dr2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.007141714 * (10.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.sum_zz_dr2 < 0.009606007 and Q.z_dr_0p1_0p2 > 0.1548383:
        z += 641.1909 * (0.009606007 - Q.sum_zz_dr2) * (Q.z_dr_0p1_0p2 - 0.1548383)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.sum_pt_top50 < 1156.659:
        z += 0.0002016284 * (10.0 - Q.n_dr_0p2_0p4) * (1156.659 - Q.sum_pt_top50)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_pt_above_5 < 36.0:
        z += 0.001506966 * (10.0 - Q.n_dr_0p2_0p4) * (36.0 - Q.n_pt_above_5)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.pt_6 < 62.25:
        z += -0.0006129576 * (10.0 - Q.n_dr_0p2_0p4) * (62.25 - Q.pt_6)
    if Q.sum_zz_dr2 < 0.009606007 and Q.sum_pt < 1115.723:
        z += 2.146739 * (0.009606007 - Q.sum_zz_dr2) * (1115.723 - Q.sum_pt)
    if Q.z_top30_slots > 0.9734886 and Q.D2_b2 < 1.08774:
        z += 10.22864 * (Q.z_top30_slots - 0.9734886) * (1.08774 - Q.D2_b2)
    if Q.sum_zz_dr2 < 0.00818374 and Q.sum_pt < 1115.723:
        z += -3.026753 * (0.00818374 - Q.sum_zz_dr2) * (1115.723 - Q.sum_pt)
    if Q.D2 < 4.450169 and Q.pt_3 > 39.3125:
        z += 0.0005802914 * (4.450169 - Q.D2) * (Q.pt_3 - 39.3125)
    if Q.z_top30_slots > 0.9734886 and Q.dr_10 < 0.2462041:
        z += 21.41782 * (Q.z_top30_slots - 0.9734886) * (0.2462041 - Q.dr_10)
    if Q.z_top30_slots > 0.9734886 and Q.dr0_12 < 0.2665639:
        z += 18.49721 * (Q.z_top30_slots - 0.9734886) * (0.2665639 - Q.dr0_12)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.2380945
    if Q.sum_z_dr2 < 0.006403325:
        z += 302.6706 * Q.sum_z_dr2 - 1.938098
    if Q.sum_z_dr2_top30 < 0.005809485:
        z += 127.338 * Q.sum_z_dr2_top30 - 0.7397682
    if Q.psi_0p1 < 0.08133662:
        z += 5.745765 * Q.psi_0p1 - 0.4673411
    if Q.psi_0p1 >= 0.9371031:
        z += -4.784461 * Q.psi_0p1 + 4.483533
    if Q.sum_zz_dr2 < 0.004754578:
        z += -724.4388 * Q.sum_zz_dr2 + 5.433797
    if 0.004754578 <= Q.sum_zz_dr2 < 0.007511864:
        z += -721.5049 * Q.sum_zz_dr2 + 5.419847
    if 0.9896594 <= Q.psi_0p3 < 0.9943058:
        z += 36.74337 * Q.psi_0p3 - 36.36342
    if Q.psi_0p3 >= 0.9943058:
        z += -10.08028 * Q.psi_0p3 + 10.1936
    if Q.LHA >= 0.3719813:
        z += 2.699002 * Q.LHA - 1.003978
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -6.53353 * Q.z_dr_0_0p05 + 5.742359
    if Q.lam1 < 0.002752094:
        z += 182.456 * Q.lam1 - 0.5021359
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.01414891 * Q.n_dr_0p2_0p4 + 0.1414891
    if Q.log_sum_pt < 6.811175:
        z += -0.9765877 * Q.log_sum_pt + 6.516483
    if 6.811175 <= Q.log_sum_pt < 6.879399:
        z += 1.982103 * Q.log_sum_pt - 13.63567
    if Q.sum_z_dr2_top15 < 0.0004816552:
        z += -408.8431 * Q.sum_z_dr2_top15 + 0.1969214
    if Q.sum_z_dr2 < 0.006403325 and Q.log_sum_pt > 6.811175:
        z += 7273.816 * (0.006403325 - Q.sum_z_dr2) * (Q.log_sum_pt - 6.811175)
    if Q.sum_zz_dr2 < 0.007511864 and Q.log_sum_pt > 6.811175:
        z += -6394.956 * (0.007511864 - Q.sum_zz_dr2) * (Q.log_sum_pt - 6.811175)
    if Q.psi_0p1 < 0.08133662 and Q.sum_pt_top50 > 889.8503:
        z += 0.04543541 * (0.08133662 - Q.psi_0p1) * (Q.sum_pt_top50 - 889.8503)
    if Q.sum_zz_dr2 < 0.007511864 and Q.zdr_0 > 0.005911134:
        z += 9497.529 * (0.007511864 - Q.sum_zz_dr2) * (Q.zdr_0 - 0.005911134)
    if Q.psi_0p1 < 0.08133662 and Q.sum_z_dr2_top30 < 0.02809026:
        z += -949.8187 * (0.08133662 - Q.psi_0p1) * (0.02809026 - Q.sum_z_dr2_top30)
    if Q.sum_pt > 1085.125 and Q.lam1 < 0.001868041:
        z += 1.336265 * (Q.sum_pt - 1085.125) * (0.001868041 - Q.lam1)
    if Q.psi_0p3 > 0.9943058 and Q.sum_z_dr2 < 0.00751625:
        z += 11834.35 * (Q.psi_0p3 - 0.9943058) * (0.00751625 - Q.sum_z_dr2)
    if Q.sum_zz_dr2 < 0.007511864 and Q.n_real_top40 < 34.0:
        z += -2.133039 * (0.007511864 - Q.sum_zz_dr2) * (34.0 - Q.n_real_top40)
    if Q.LHA > 0.3719813 and Q.tau21_b2 < 0.2352054:
        z += 62.88795 * (Q.LHA - 0.3719813) * (0.2352054 - Q.tau21_b2)
    if Q.sum_z_dr2 < 0.006403325 and Q.n_for_50pct < 9.0:
        z += 10.9515 * (0.006403325 - Q.sum_z_dr2) * (9.0 - Q.n_for_50pct)
    if Q.sum_z_dr2_top30 < 0.005809485 and Q.z_dr_0p05_0p1 < 0.07906904:
        z += 1181.757 * (0.005809485 - Q.sum_z_dr2_top30) * (0.07906904 - Q.z_dr_0p05_0p1)
    if Q.sum_pt > 1085.125 and Q.z_dr_0p2_0p4 < 0.019523:
        z += -0.06169269 * (Q.sum_pt - 1085.125) * (0.019523 - Q.z_dr_0p2_0p4)
    if Q.psi_0p1 < 0.08133662 and Q.z_dr_0p2_0p4 < 0.2708738:
        z += 24.56271 * (0.08133662 - Q.psi_0p1) * (0.2708738 - Q.z_dr_0p2_0p4)
    if Q.sum_zz_dr2 < 0.007511864 and Q.pt_10 < 32.09375:
        z += 1.108523 * (0.007511864 - Q.sum_zz_dr2) * (32.09375 - Q.pt_10)
    if Q.log_sum_pt < 6.811175 and Q.dr0_3 > 0.1617791:
        z += -32.73732 * (6.811175 - Q.log_sum_pt) * (Q.dr0_3 - 0.1617791)
    if Q.psi_0p1 > 0.9371031 and Q.dr_max_012 > 0.06112084:
        z += 63.25126 * (Q.psi_0p1 - 0.9371031) * (Q.dr_max_012 - 0.06112084)
    if Q.psi_0p3 > 0.9896594 and Q.sum_z_dr2_top2 < 0.001864148:
        z += 4241.987 * (Q.psi_0p3 - 0.9896594) * (0.001864148 - Q.sum_z_dr2_top2)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.091382
    if Q.sum_pt < 1017.435:
        z += 0.02385935 * Q.sum_pt - 25.89038
    if 1017.435 <= Q.sum_pt < 1085.125:
        z += 0.01231496 * Q.sum_pt - 14.14471
    if 1085.125 <= Q.sum_pt < 1167.447:
        z += -0.0115444 * Q.sum_pt + 11.74567
    if Q.sum_pt >= 1167.447:
        z += -0.01315881 * Q.sum_pt + 13.63041
    if 0.008840538 <= Q.sum_z_dr2_top40 < 0.01292642:
        z += -175.537 * Q.sum_z_dr2_top40 + 1.551841
    if 0.01292642 <= Q.sum_z_dr2_top40 < 0.02862386:
        z += -259.2521 * Q.sum_z_dr2_top40 + 2.633978
    if Q.sum_z_dr2_top40 >= 0.02862386:
        z += 1855.522 * Q.sum_z_dr2_top40 - 57.89903
    if Q.n_pt_above_5 < 25.0:
        z += 0.02791386 * Q.n_pt_above_5 - 0.6978465
    if Q.sum_z_dr < 0.1207452:
        z += 7.856013 * Q.sum_z_dr - 0.9485757
    if Q.sum_pt_top40 >= 972.4111:
        z += -0.005633742 * Q.sum_pt_top40 + 5.478313
    if Q.e2 < 0.03480688:
        z += -5.160726 * Q.e2 + 0.1796287
    if Q.e2 >= 0.03680582:
        z += 26.44434 * Q.e2 - 0.9733054
    if Q.sum_pt_top50 < 889.8503:
        z += -0.01261838 * Q.sum_pt_top50 + 13.61515
    if 889.8503 <= Q.sum_pt_top50 < 976.277:
        z += -0.02149263 * Q.sum_pt_top50 + 21.51191
    if 976.277 <= Q.sum_pt_top50 < 1013.916:
        z += -0.01644635 * Q.sum_pt_top50 + 16.58535
    if 1013.916 <= Q.sum_pt_top50 < 1078.994:
        z += -0.005283454 * Q.sum_pt_top50 + 5.26711
    if Q.sum_pt_top50 >= 1078.994:
        z += 0.007334922 * Q.sum_pt_top50 - 8.348045
    if Q.tau2 >= 0.05936026:
        z += -6.892525 * Q.tau2 + 0.4091421
    if 6.811175 <= Q.log_sum_pt < 6.893714:
        z += 28.64572 * Q.log_sum_pt - 195.111
    if Q.log_sum_pt >= 6.893714:
        z += 11.46264 * Q.log_sum_pt - 76.65579
    if Q.lam1_plus_lam2 >= 0.01991191:
        z += -232.4282 * Q.lam1_plus_lam2 + 4.628089
    if Q.n_particles < 54.0:
        z += 0.005326562 * Q.n_particles - 0.2876344
    if Q.sum_zz_dr2 >= 0.0292152:
        z += -5588.839 * Q.sum_zz_dr2 + 163.279
    if 0.005852839 <= Q.sum_z_dr2_top50 < 0.01351431:
        z += 127.5271 * Q.sum_z_dr2_top50 - 0.7463955
    if Q.sum_z_dr2_top50 >= 0.01351431:
        z += -44.29942 * Q.sum_z_dr2_top50 + 1.575721
    if Q.sum_pt_top20 >= 1064.139:
        z += 0.003731949 * Q.sum_pt_top20 - 3.971313
    if Q.sum_pt_top10 >= 867.9266:
        z += -0.001042474 * Q.sum_pt_top10 + 0.9047911
    if Q.psi_0p3 >= 0.9777125:
        z += 7.998101 * Q.psi_0p3 - 7.819843
    if Q.sum_z_dr2_top20 >= 0.01083435:
        z += -34.44036 * Q.sum_z_dr2_top20 + 0.3731391
    if Q.sum_pt_top15 < 795.6047:
        z += 0.001492092 * Q.sum_pt_top15 - 1.187115
    if Q.sum_z_dr2_top40 > 0.01292642 and Q.sum_pt_top50 < 1245.697:
        z += 1.135269 * (Q.sum_z_dr2_top40 - 0.01292642) * (1245.697 - Q.sum_pt_top50)
    if Q.sum_pt_top40 > 972.4111 and Q.soft1_pt < 2.275391:
        z += 0.0001881015 * (Q.sum_pt_top40 - 972.4111) * (2.275391 - Q.soft1_pt)
    if Q.n_pt_above_5 < 25.0 and Q.D2 < 5.378975:
        z += 0.009071271 * (25.0 - Q.n_pt_above_5) * (5.378975 - Q.D2)
    if Q.sum_pt < 1085.125 and Q.tau21 > 0.1295048:
        z += 0.0110407 * (1085.125 - Q.sum_pt) * (Q.tau21 - 0.1295048)
    if Q.sum_z_dr < 0.1207452 and Q.z_dr_0p05_0p1 > 0.1514163:
        z += -4.5634 * (0.1207452 - Q.sum_z_dr) * (Q.z_dr_0p05_0p1 - 0.1514163)
    if Q.sum_z_dr2_top40 > 0.02862386 and Q.pt_6 > 19.46875:
        z += 30.72762 * (Q.sum_z_dr2_top40 - 0.02862386) * (Q.pt_6 - 19.46875)
    if Q.sum_z_dr2_top40 > 0.02862386 and Q.z_6 > 0.01906139:
        z += -165648.0 * (Q.sum_z_dr2_top40 - 0.02862386) * (Q.z_6 - 0.01906139)
    if Q.sum_z_dr2_top40 > 0.02862386 and Q.pt_6 < 62.25:
        z += -34.49012 * (Q.sum_z_dr2_top40 - 0.02862386) * (62.25 - Q.pt_6)
    if Q.log_sum_pt > 6.893714 and Q.tau32 < 0.577419:
        z += -5.562431 * (Q.log_sum_pt - 6.893714) * (0.577419 - Q.tau32)
    if Q.e2 > 0.03680582 and Q.sj2_zsoft < 0.2416266:
        z += 178.4771 * (Q.e2 - 0.03680582) * (0.2416266 - Q.sj2_zsoft)
    if Q.log_sum_pt > 6.811175 and Q.C2 > 0.06655881:
        z += -25.02103 * (Q.log_sum_pt - 6.811175) * (Q.C2 - 0.06655881)
    if Q.sum_z_dr2_top50 > 0.005852839 and Q.pt_4 > 90.625:
        z += -3.745988 * (Q.sum_z_dr2_top50 - 0.005852839) * (Q.pt_4 - 90.625)
    if Q.sum_pt_top50 < 1078.994 and Q.tau32 > 0.577419:
        z += 0.006287604 * (1078.994 - Q.sum_pt_top50) * (Q.tau32 - 0.577419)
    if Q.sum_z_dr2_top50 > 0.005852839 and Q.soft7_z > 0.00146164:
        z += -15342.53 * (Q.sum_z_dr2_top50 - 0.005852839) * (Q.soft7_z - 0.00146164)
    if Q.sum_pt > 1017.435 and Q.absphi_2 > 0.03259277:
        z += -0.01387827 * (Q.sum_pt - 1017.435) * (Q.absphi_2 - 0.03259277)
    if Q.sum_pt_top10 > 867.9266 and Q.dr_max_012 < 0.06112084:
        z += -0.02447094 * (Q.sum_pt_top10 - 867.9266) * (0.06112084 - Q.dr_max_012)
    if Q.psi_0p3 > 0.9777125 and Q.sj2_zsoft > 0.1181474:
        z += -18.03642 * (Q.psi_0p3 - 0.9777125) * (Q.sj2_zsoft - 0.1181474)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.01490819
    if Q.tau21_b2 < 0.1722488:
        z += -6.601048 * Q.tau21_b2 + 2.050045
    if 0.1722488 <= Q.tau21_b2 < 0.342495:
        z += -5.362957 * Q.tau21_b2 + 1.836786
    if Q.n_dr_0p1_0p2 < 17.0:
        z += 0.02448035 * Q.n_dr_0p1_0p2 - 0.4161659
    if 0.9896594 <= Q.psi_0p3 < 0.9956185:
        z += 27.06206 * Q.psi_0p3 - 26.78222
    if Q.psi_0p3 >= 0.9956185:
        z += 282.32 * Q.psi_0p3 - 280.9218
    if Q.LHA < 0.1632346:
        z += 30.94089 * Q.LHA - 5.259903
    if 0.1632346 <= Q.LHA < 0.2091025:
        z += 4.562638 * Q.LHA - 0.9540591
    if Q.LHA >= 0.3098384:
        z += -2.22939 * Q.LHA + 0.6907507
    if Q.lam1_plus_lam2 < 0.00751625:
        z += 191.7025 * Q.lam1_plus_lam2 - 1.440884
    if Q.sum_zz_dr2 < 0.00363788:
        z += -302.0295 * Q.sum_zz_dr2 + 6.664742
    if 0.00363788 <= Q.sum_zz_dr2 < 0.006936725:
        z += -369.0144 * Q.sum_zz_dr2 + 6.908425
    if 0.006936725 <= Q.sum_zz_dr2 < 0.009606007:
        z += -719.13 * Q.sum_zz_dr2 + 9.337081
    if 0.009606007 <= Q.sum_zz_dr2 < 0.01396296:
        z += -472.6242 * Q.sum_zz_dr2 + 6.969144
    if 0.01396296 <= Q.sum_zz_dr2 < 0.01989454:
        z += -245.948 * Q.sum_zz_dr2 + 3.804073
    if Q.sum_zz_dr2 >= 0.01989454:
        z += -66.98484 * Q.sum_zz_dr2 + 0.2436828
    if Q.tau1 < 0.05444509:
        z += -3.388521 * Q.tau1 - 0.4401114
    if 0.05444509 <= Q.tau1 < 0.0705748:
        z += 7.198921 * Q.tau1 - 1.016546
    if 0.0705748 <= Q.tau1 < 0.1008497:
        z += 16.79554 * Q.tau1 - 1.693825
    if Q.sum_z_dr2 < 0.001784491:
        z += 8339.433 * Q.sum_z_dr2 - 17.6153
    if 0.001784491 <= Q.sum_z_dr2 < 0.008190222:
        z += 426.7525 * Q.sum_z_dr2 - 3.495197
    if Q.sum_z_dr2_top30 < 0.006088416:
        z += 104.3622 * Q.sum_z_dr2_top30 - 1.994979
    if 0.006088416 <= Q.sum_z_dr2_top30 < 0.008376291:
        z += 278.692 * Q.sum_z_dr2_top30 - 3.056371
    if 0.008376291 <= Q.sum_z_dr2_top30 < 0.01215787:
        z += 190.9164 * Q.sum_z_dr2_top30 - 2.321137
    if Q.N2 < 0.3572263:
        z += 1.710321 * Q.N2 - 0.6109717
    if Q.soft7_z < 0.001595561:
        z += 214.0436 * Q.soft7_z - 0.3415197
    if Q.sum_z_dr2_top3 < 0.001155057:
        z += 113.0309 * Q.sum_z_dr2_top3 + 0.04249074
    if 0.001155057 <= Q.sum_z_dr2_top3 < 0.00588499:
        z += -36.58568 * Q.sum_z_dr2_top3 + 0.2153064
    if Q.n_pt_above_10 >= 28.0:
        z += -0.02897518 * Q.n_pt_above_10 + 0.8113051
    if Q.e2 < 0.03480688:
        z += -31.50213 * Q.e2 + 1.096491
    if Q.z_top5 < 0.6289751:
        z += 0.6652809 * Q.z_top5 - 0.4184451
    if Q.tau21 < 0.5936969:
        z += 0.6066413 * Q.tau21 - 0.360161
    if Q.sum_z_dr2_top50 < 0.00634935:
        z += -144.8749 * Q.sum_z_dr2_top50 - 0.1835368
    if 0.00634935 <= Q.sum_z_dr2_top50 < 0.007820315:
        z += 155.1863 * Q.sum_z_dr2_top50 - 2.088731
    if 0.007820315 <= Q.sum_z_dr2_top50 < 0.008124776:
        z += 590.0027 * Q.sum_z_dr2_top50 - 5.489131
    if 0.008124776 <= Q.sum_z_dr2_top50 < 0.01951641:
        z += 61.05285 * Q.sum_z_dr2_top50 - 1.191532
    if Q.sum_z_dr2_top10 < 0.00625621:
        z += -40.29536 * Q.sum_z_dr2_top10 + 0.2520963
    if Q.sum_z_dr2_top20 < 0.006374178:
        z += 22.93146 * Q.sum_z_dr2_top20 - 0.5822906
    if 0.006374178 <= Q.sum_z_dr2_top20 < 0.01083435:
        z += 97.78123 * Q.sum_z_dr2_top20 - 1.059396
    if Q.log_sum_pt >= 7.062574:
        z += -9.559502 * Q.log_sum_pt + 67.51469
    if Q.sum_pt >= 1260.541:
        z += -0.002590657 * Q.sum_pt + 3.26563
    if Q.sum_pt_top50 >= 1245.697:
        z += 0.004485855 * Q.sum_pt_top50 - 5.588014
    if Q.z_dr_0p1_0p2 < 0.2864926:
        z += -0.9379266 * Q.z_dr_0p1_0p2 + 0.268709
    if 0.1666442 <= Q.sj2_dr < 0.2412757:
        z += 1.316912 * Q.sj2_dr - 0.2194557
    if Q.sj2_dr >= 0.2412757:
        z += -1.50839 * Q.sj2_dr + 0.4622209
    if Q.sum_z_dr2_top40 < 0.01292642:
        z += 134.1185 * Q.sum_z_dr2_top40 - 1.733673
    if Q.zdr_1 < 0.003626563:
        z += -8.428026 * Q.zdr_1 - 0.106384
    if 0.003626563 <= Q.zdr_1 < 0.008824206:
        z += 26.34825 * Q.zdr_1 - 0.2325023
    if Q.n_pt_above_5 < 28.0:
        z += 0.01620039 * Q.n_pt_above_5 - 0.453611
    if Q.tau4 < 0.03409838:
        z += -8.267717 * Q.tau4 + 0.2819158
    if Q.tau21_b2 < 0.342495 and Q.lam1 < 0.02048524:
        z += -298.0919 * (0.342495 - Q.tau21_b2) * (0.02048524 - Q.lam1)
    if Q.sum_zz_dr2 < 0.01396296 and Q.z_top50_slots < 0.9906378:
        z += 1536.304 * (0.01396296 - Q.sum_zz_dr2) * (0.9906378 - Q.z_top50_slots)
    if Q.psi_0p3 > 0.9956185 and Q.sum_pt_top40 < 1225.842:
        z += -1.185366 * (Q.psi_0p3 - 0.9956185) * (1225.842 - Q.sum_pt_top40)
    if Q.sum_z_dr2 < 0.008190222 and Q.soft8_z < 0.002171436:
        z += 28139.42 * (0.008190222 - Q.sum_z_dr2) * (0.002171436 - Q.soft8_z)
    if Q.psi_0p3 > 0.9956185 and Q.sj3_dr13 > 0.228329:
        z += -867.1423 * (Q.psi_0p3 - 0.9956185) * (Q.sj3_dr13 - 0.228329)
    if Q.psi_0p3 > 0.9956185 and Q.soft6_pt < 4.250195:
        z += 11.55929 * (Q.psi_0p3 - 0.9956185) * (4.250195 - Q.soft6_pt)
    if Q.psi_0p3 > 0.9956185 and Q.sum_pt_top40 < 1024.942:
        z += 1.628641 * (Q.psi_0p3 - 0.9956185) * (1024.942 - Q.sum_pt_top40)
    if Q.sum_z_dr2_top30 < 0.01215787 and Q.sum_pt > 907.9372:
        z += 0.5397678 * (0.01215787 - Q.sum_z_dr2_top30) * (Q.sum_pt - 907.9372)
    if Q.e2 < 0.03480688 and Q.sum_pt_top30 > 886.3438:
        z += -0.06235874 * (0.03480688 - Q.e2) * (Q.sum_pt_top30 - 886.3438)
    if Q.log_sum_pt > 7.062574 and Q.absphi_5 < 0.05230713:
        z += 32.60247 * (Q.log_sum_pt - 7.062574) * (0.05230713 - Q.absphi_5)
    if Q.sum_z_dr2 < 0.008190222 and Q.psi_0p3 > 0.9956185:
        z += -49777.8 * (0.008190222 - Q.sum_z_dr2) * (Q.psi_0p3 - 0.9956185)
    if Q.z_dr_0p05_0p1 > 0.7108211 and Q.pt2_over_pt0 < 0.7393686:
        z += -2.918602 * (Q.z_dr_0p05_0p1 - 0.7108211) * (0.7393686 - Q.pt2_over_pt0)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.5840438
    if Q.z_dr_0_0p05 < 0.8459004:
        z += -0.2688482 * Q.z_dr_0_0p05 + 0.2274188
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -4.805656 * Q.z_dr_0p1_0p2 + 0.5783303
    if Q.sd_rg < 0.1778185:
        z += 0.2882523 * Q.sd_rg - 0.05125658
    if Q.psi_0p1 >= 0.8976117:
        z += -5.80897 * Q.psi_0p1 + 5.2142
    if Q.sum_z_dr2_top5 < 0.002270363:
        z += 125.6025 * Q.sum_z_dr2_top5 + 0.1134608
    if 0.002270363 <= Q.sum_z_dr2_top5 < 0.008329695:
        z += -65.7868 * Q.sum_z_dr2_top5 + 0.547984
    if Q.sum_z_dr2_top10 < 0.002247756:
        z += 40.7051 * Q.sum_z_dr2_top10 + 0.2381106
    if 0.002247756 <= Q.sum_z_dr2_top10 < 0.007678544:
        z += -60.69206 * Q.sum_z_dr2_top10 + 0.4660267
    if Q.e3 < 4.646151e-05:
        z += -3163.293 * Q.e3 + 0.1469714
    if Q.n_dr_0p1_0p2 < 10.0:
        z += -0.01516683 * Q.n_dr_0p1_0p2 + 0.1516683
    if Q.log_sum_pt < 6.97212:
        z += -4.440175 * Q.log_sum_pt + 30.95743
    if Q.log_sum_pt >= 7.139296:
        z += -1.449667 * Q.log_sum_pt + 10.3496
    if Q.sum_pt_top30 >= 988.4375:
        z += -0.001252255 * Q.sum_pt_top30 + 1.237776
    if 1003.544 <= Q.sum_pt_top50 < 1024.689:
        z += -0.0005689124 * Q.sum_pt_top50 + 0.5709288
    if 1024.689 <= Q.sum_pt_top50 < 1107.226:
        z += -0.004377045 * Q.sum_pt_top50 + 4.473081
    if 1107.226 <= Q.sum_pt_top50 < 1156.659:
        z += 0.00563907 * Q.sum_pt_top50 - 6.617021
    if Q.sum_pt_top50 >= 1156.659:
        z += 0.007529081 * Q.sum_pt_top50 - 8.803121
    if Q.tau1 < 0.06310829:
        z += 15.57567 * Q.tau1 - 1.192749
    if 0.06310829 <= Q.tau1 < 0.0705748:
        z += 28.0981 * Q.tau1 - 1.983018
    if Q.sum_z_dr < 0.08068193:
        z += -34.8316 * Q.sum_z_dr + 3.299206
    if 0.08068193 <= Q.sum_z_dr < 0.1207452:
        z += -12.20384 * Q.sum_z_dr + 1.473554
    if Q.sum_pt_top20 >= 869.693:
        z += 0.0008846633 * Q.sum_pt_top20 - 0.7693854
    if Q.D2 < 1.976207:
        z += 0.2559458 * Q.D2 - 0.505802
    if Q.sum_z_dr2_top30 < 0.005402331:
        z += -25.545 * Q.sum_z_dr2_top30 - 0.491757
    if 0.005402331 <= Q.sum_z_dr2_top30 < 0.008376291:
        z += 23.30463 * Q.sum_z_dr2_top30 - 0.7556589
    if 0.008376291 <= Q.sum_z_dr2_top30 < 0.01215787:
        z += 148.2058 * Q.sum_z_dr2_top30 - 1.801868
    if Q.absphi_0 < 0.02970886:
        z += 3.464053 * Q.absphi_0 - 0.1029131
    if Q.psi_0p3 >= 0.9985421:
        z += -183.0809 * Q.psi_0p3 + 182.8139
    if Q.lam2 < 0.002396991:
        z += -179.6571 * Q.lam2 + 0.4306365
    if Q.z_dr_0p2_0p4 < 0.02647293:
        z += 7.581984 * Q.z_dr_0p2_0p4 - 0.2007173
    if 0.1745007 <= Q.sj2_dr < 0.2232169:
        z += -3.793998 * Q.sj2_dr + 0.6620553
    if 0.2232169 <= Q.sj2_dr < 0.2780918:
        z += 6.057819 * Q.sj2_dr - 1.537037
    if Q.sj2_dr >= 0.2780918:
        z += 0.1836951 * Q.sj2_dr + 0.09650875
    if Q.e2 < 0.04082832:
        z += -33.36121 * Q.e2 + 1.362082
    if Q.z_top30_slots < 0.9860575:
        z += 3.033329 * Q.z_top30_slots - 2.991037
    if Q.lam1 < 0.006189818:
        z += -130.8379 * Q.lam1 + 0.8098628
    if Q.LHA < 0.2941033:
        z += 9.535321 * Q.LHA - 2.804369
    if Q.sum_z_dr2_top50 < 0.006805717:
        z += 48.07859 * Q.sum_z_dr2_top50 - 0.3272093
    if 858.8262 <= Q.sum_pt_top40 < 1095.686:
        z += 0.001025416 * Q.sum_pt_top40 - 0.8806545
    if Q.sum_pt_top40 >= 1095.686:
        z += -0.006709461 * Q.sum_pt_top40 + 7.594346
    if Q.sum_z_dr2 < 0.00751625:
        z += 155.6522 * Q.sum_z_dr2 - 1.169921
    if Q.sum_z_dr2_top3 < 0.000341301:
        z += -519.7653 * Q.sum_z_dr2_top3 + 0.1773964
    if Q.sum_pt < 1002.379:
        z += 0.007651213 * Q.sum_pt - 7.669411
    if Q.sd_rg < 0.1778185 and Q.planar_flow < 0.6025827:
        z += 4.735805 * (0.1778185 - Q.sd_rg) * (0.6025827 - Q.planar_flow)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.sum_pt < 986.0565:
        z += 0.04707095 * (0.1203437 - Q.z_dr_0p1_0p2) * (986.0565 - Q.sum_pt)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.psi_0p2 < 0.8706159:
        z += -29.30645 * (0.1203437 - Q.z_dr_0p1_0p2) * (0.8706159 - Q.psi_0p2)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.n_real_top50 < 46.0:
        z += 2.5967 * (0.008329695 - Q.sum_z_dr2_top5) * (46.0 - Q.n_real_top50)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.psi_0p3 > 0.9777125:
        z += -178.361 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.psi_0p3 - 0.9777125)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.n_dr_0p2_0p4 > 10.0:
        z += -2.600043 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.n_dr_0p2_0p4 - 10.0)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.C2_b2 < 0.05112769:
        z += 31.35225 * (0.1203437 - Q.z_dr_0p1_0p2) * (0.05112769 - Q.C2_b2)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.sum_pt < 995.6769:
        z += -0.06495662 * (0.1203437 - Q.z_dr_0p1_0p2) * (995.6769 - Q.sum_pt)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.psi_0p3 > 0.9299135:
        z += -367.1804 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.psi_0p3 - 0.9299135)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.lam2 < 0.002396991:
        z += 1476.391 * (0.1203437 - Q.z_dr_0p1_0p2) * (0.002396991 - Q.lam2)
    if Q.sd_rg < 0.1778185 and Q.soft5_dr > 0.04139378:
        z += -16.98429 * (0.1778185 - Q.sd_rg) * (Q.soft5_dr - 0.04139378)
    if Q.sd_rg < 0.1778185 and Q.soft5_dr0 > 0.03721986:
        z += 15.61607 * (0.1778185 - Q.sd_rg) * (Q.soft5_dr0 - 0.03721986)
    if Q.psi_0p3 > 0.9985421 and Q.dr_10 < 0.1613937:
        z += 748.2675 * (Q.psi_0p3 - 0.9985421) * (0.1613937 - Q.dr_10)
    if Q.sum_z_dr2_top3 < 0.001155057 and Q.eccentricity > 0.773315:
        z += -2296.187 * (0.001155057 - Q.sum_z_dr2_top3) * (Q.eccentricity - 0.773315)
    if Q.e3 < 4.646151e-05 and Q.n_dr_0p1_0p2 < 21.0:
        z += -728.2236 * (4.646151e-05 - Q.e3) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 26.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.0007291337 * (26.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.sum_pt_top50 > 1156.659 and Q.dr0_8 < 0.1381171:
        z += 0.007543466 * (Q.sum_pt_top50 - 1156.659) * (0.1381171 - Q.dr0_8)
    if Q.sum_pt_top30 > 988.4375 and Q.soft10_pt < 6.70332:
        z += 0.0001739157 * (Q.sum_pt_top30 - 988.4375) * (6.70332 - Q.soft10_pt)
    if Q.sum_pt_top50 > 1024.689 and Q.ptdr0_8 < 9.176976:
        z += -7.045272e-05 * (Q.sum_pt_top50 - 1024.689) * (9.176976 - Q.ptdr0_8)
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
