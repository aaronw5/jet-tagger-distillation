"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; all observables), as if-statements.

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
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_13         mass of particles 0 and 13 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
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
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_10                  pT of particle 10 [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_pt              pT [GeV] of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_pt               pT [GeV] of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.z_2                    pT of particle 2 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_4                    pT of particle 4 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
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
  Q.zdr_11                 pT share × ΔR of particle 11 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_13              |Δφ| of particle 13
  Q.absphi_4               |Δφ| of particle 4
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft10_dr0             ΔR between the hardest and the 10. softest real particle (0 if among the 15 hardest)
  Q.soft3_dr0              ΔR between the hardest and the 3. softest real particle (0 if among the 15 hardest)
  Q.dr1_12                 ΔR between particle 12 and the 2nd-hardest particle
  Q.dr1_5                  ΔR between particle 5 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_9                   ΔR of particle 9 from the jet axis
  Q.soft3_dr               ΔR from the jet axis of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.eta_0                  Δη of particle 0
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
        pair_mass_0_13=pair_mass(0, 13),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
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
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_0=pt[0],
        pt_10=pt[10],
        pt_11=pt[11],
        pt_2=pt[2],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_6=pt[6],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft10_pt=softp(10, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft7_pt=softp(7, 'pt'),
        soft9_pt=softp(9, 'pt'),
        z_11=z[11],
        z_2=z[2],
        z_3=z[3],
        z_4=z[4],
        z_6=z[6],
        z_7=z[7],
        z_9=z[9],
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft7_z=softp(7, 'z'),
        soft8_z=softp(8, 'z'),
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
        zdr_11=z[11] * dr[11],
        zdr_2=z[2] * dr[2],
        zdr_4=z[4] * dr[4],
        zdr_5=z[5] * dr[5],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        absphi_0=abs(phi[0]),
        absphi_13=abs(phi[13]),
        absphi_4=abs(phi[4]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft10_dr0=softp(10, 'dr0'),
        soft3_dr0=softp(3, 'dr0'),
        dr1_12=math.sqrt(dist2(1, 12)) if pt[12] > 0 else 0.0,
        dr1_5=math.sqrt(dist2(1, 5)) if pt[5] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_9=dr[9] if pt[9] > 0 else 0.0,
        soft3_dr=softp(3, 'dr'),
        eta_0=eta[0],
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
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = 1.196448
    if Q.mass < 53.87362:
        z += 0.03636014 * Q.mass - 3.902042
    if 53.87362 <= Q.mass < 74.25181:
        z += 0.04119014 * Q.mass - 4.162252
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.01743595 * Q.mass - 2.39846
    if 78.26182 <= Q.mass < 89.74183:
        z += -0.0949753 * Q.mass + 6.399049
    if 89.74183 <= Q.mass < 91.19:
        z += -0.05627124 * Q.mass + 2.925675
    if 91.19 <= Q.mass < 92.85979:
        z += 0.003758887 * Q.mass - 2.548472
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.04472201 * Q.mass - 6.352299
    if Q.mass >= 101.0497:
        z += 0.003531868 * Q.mass - 2.190047
    if Q.girth2_top20 < 0.005312783:
        z += 72.04876 * Q.girth2_top20 - 0.460692
    if 0.005312783 <= Q.girth2_top20 < 0.006374178:
        z += 108.0394 * Q.girth2_top20 - 0.6519027
    if 0.006374178 <= Q.girth2_top20 < 0.007538019:
        z += -31.58501 * Q.girth2_top20 + 0.2380884
    if Q.sum_pt < 972.0419:
        z += 0.01394428 * Q.sum_pt - 13.92256
    if 972.0419 <= Q.sum_pt < 1012.673:
        z += 0.00906047 * Q.sum_pt - 9.175293
    if Q.psi_0p3 >= 0.9956185:
        z += 64.65437 * Q.psi_0p3 - 64.37108
    if Q.mass_top30 < 80.4:
        z += -0.01379006 * Q.mass_top30 + 1.108721
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.04049617 * Q.n_dr_0p2_0p4 + 0.6743537
    if 11.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += -0.05722395 * Q.n_dr_0p2_0p4 + 0.8583593
    if Q.girth2_top30 < 0.006363916:
        z += 197.5911 * Q.girth2_top30 - 1.231289
    if 0.006363916 <= Q.girth2_top30 < 0.007856958:
        z += -17.52419 * Q.girth2_top30 + 0.1376869
    if Q.lam1 < 0.005913555:
        z += 64.80396 * Q.lam1 - 0.3832218
    if Q.mass_over_sum_pt_sq < 0.004754444:
        z += -882.635 * Q.mass_over_sum_pt_sq + 6.268914
    if 0.004754444 <= Q.mass_over_sum_pt_sq < 0.006938798:
        z += -770.7587 * Q.mass_over_sum_pt_sq + 5.737005
    if 0.006938798 <= Q.mass_over_sum_pt_sq < 0.007873266:
        z += -416.1364 * Q.mass_over_sum_pt_sq + 3.276352
    if Q.tau1 < 0.0705748:
        z += 17.96582 * Q.tau1 - 1.267935
    if Q.e2_sq < 0.00616708:
        z += 454.1041 * Q.e2_sq - 2.800496
    if Q.e2 < 0.01879315:
        z += -34.57227 * Q.e2 + 0.6497218
    if Q.e2 >= 0.05557149:
        z += -22.03074 * Q.e2 + 1.224281
    if 71.79516 <= Q.mass_top50 < 82.04491:
        z += 0.01463104 * Q.mass_top50 - 1.050438
    if Q.mass_top50 >= 82.04491:
        z += -0.05733503 * Q.mass_top50 + 4.854012
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += 6.011036 * Q.z_dr_0p2_0p4 - 0.5483609
    if Q.z_top50_slots < 0.9906378:
        z += 14.63265 * Q.z_top50_slots - 14.49565
    if Q.log_sum_pt < 7.017258:
        z += -5.912028 * Q.log_sum_pt + 41.48622
    if Q.sum_pt_top40 < 858.8262:
        z += 0.000319886 * Q.sum_pt_top40 - 0.8198325
    if 858.8262 <= Q.sum_pt_top40 < 1069.671:
        z += 0.00258534 * Q.sum_pt_top40 - 2.765463
    if Q.sum_pt_top20 < 846.1934:
        z += -0.0009665317 * Q.sum_pt_top20 + 0.8178727
    if Q.mass_top40 < 80.89043:
        z += 0.005089942 * Q.mass_top40 - 0.4117276
    if Q.e3 < 0.0005178279:
        z += -550.7749 * Q.e3 + 0.2852066
    if Q.C2 < 0.05602756:
        z += 4.184849 * Q.C2 - 0.2344669
    if Q.lam2 < 0.0006154841:
        z += -270.6656 * Q.lam2 + 0.1665904
    if Q.sum_pt_top50 < 1156.659:
        z += -0.001254235 * Q.sum_pt_top50 + 1.450723
    if Q.sj2_dr >= 0.2232169:
        z += -2.140493 * Q.sj2_dr + 0.4777943
    if Q.log_sum_pt < 6.98945 and Q.sum_pt_top50 > 959.0957:
        z += 0.03669033 * (6.98945 - Q.log_sum_pt) * (Q.sum_pt_top50 - 959.0957)
    if Q.z_dr_0p2_0p4 < 0.09122568 and Q.pt_2 < 118.5:
        z += 0.02193628 * (0.09122568 - Q.z_dr_0p2_0p4) * (118.5 - Q.pt_2)
    if Q.mass_over_sum_pt_sq < 0.007873266 and Q.pt_10 < 34.0625:
        z += 1.333149 * (0.007873266 - Q.mass_over_sum_pt_sq) * (34.0625 - Q.pt_10)
    if Q.sum_pt < 972.0419 and Q.z_4 < 0.05554124:
        z += 0.1011357 * (972.0419 - Q.sum_pt) * (0.05554124 - Q.z_4)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.346088
    if Q.n_particles >= 38.0:
        z += 0.03397825 * Q.n_particles - 1.291173
    if Q.log_sum_pt < 6.893714:
        z += 6.239219 * Q.log_sum_pt - 44.54363
    if 6.893714 <= Q.log_sum_pt < 6.910131:
        z += 31.5486 * Q.log_sum_pt - 219.0192
    if 6.910131 <= Q.log_sum_pt < 6.959294:
        z += 51.46406 * Q.log_sum_pt - 356.6377
    if 6.959294 <= Q.log_sum_pt < 6.98945:
        z += 35.50874 * Q.log_sum_pt - 245.6
    if 6.98945 <= Q.log_sum_pt < 7.062574:
        z += 23.68655 * Q.log_sum_pt - 162.9693
    if 7.062574 <= Q.log_sum_pt < 7.139296:
        z += 18.14891 * Q.log_sum_pt - 123.8594
    if Q.log_sum_pt >= 7.139296:
        z += 11.90969 * Q.log_sum_pt - 79.31572
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.01004309 * Q.sum_pt_top50 + 9.632282
    if Q.psi_0p3 >= 0.9980008:
        z += -130.4187 * Q.psi_0p3 + 130.158
    if Q.sum_pt_top2 < 689.25:
        z += -0.001243227 * Q.sum_pt_top2 + 0.8568941
    if Q.z_top30_slots >= 0.9341838:
        z += -2.982263 * Q.z_top30_slots + 2.785982
    if Q.mass_top20 < 47.88842:
        z += 0.04015335 * Q.mass_top20 - 1.92288
    if Q.sj3_mass1 < 32.50209:
        z += 0.02075537 * Q.sj3_mass1 - 0.6745929
    if Q.sum_pt_top40 < 1069.671:
        z += -0.007508149 * Q.sum_pt_top40 + 8.031251
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.08225519 * Q.n_dr_0p2_0p4 - 0.7489202
    if 7.0 <= Q.n_dr_0p2_0p4 < 13.0:
        z += 0.02885564 * Q.n_dr_0p2_0p4 - 0.3751233
    if Q.M3 < 0.03187688:
        z += 11.74263 * Q.M3 - 0.3743183
    if Q.sj2_mass1 < 30.26161:
        z += 0.02000129 * Q.sj2_mass1 - 0.6052712
    if Q.soft5_z < 0.001594761:
        z += -171.3551 * Q.soft5_z + 0.2732704
    if Q.pt_9 < 31.35938:
        z += 0.0260225 * Q.pt_9 - 0.8160493
    if Q.lam1 < 0.004673423:
        z += -189.7385 * Q.lam1 + 0.8867282
    if Q.mass < 120.6:
        z += 0.02065314 * Q.mass - 2.490768
    if Q.girth2_top30 < 0.02412652:
        z += -20.2124 * Q.girth2_top30 + 0.487655
    if Q.N2 >= 0.4678622:
        z += -5.393069 * Q.N2 + 2.523213
    if Q.tau3 < 0.009068077:
        z += -96.07765 * Q.tau3 + 1.530043
    if 0.009068077 <= Q.tau3 < 0.04516808:
        z += -18.24942 * Q.tau3 + 0.824291
    if Q.D3 < 0.1416054:
        z += -1.05143 * Q.D3 + 0.1488882
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -4.592127 * Q.z_dr_0_0p05 + 4.036048
    if Q.mass_top30 < 42.41192:
        z += -0.01029689 * Q.mass_top30 + 0.436711
    if Q.n_dr_0_0p05 < 12.0:
        z += -0.02922194 * Q.n_dr_0_0p05 + 0.3506633
    if Q.mass_top10 >= 56.92192:
        z += 0.004060274 * Q.mass_top10 - 0.2311186
    z += -1.197004 * Q.absphi_0
    if Q.sj3_dr_max >= 0.3276439:
        z += 1.650764 * Q.sj3_dr_max - 0.5408629
    if Q.soft1_pt < 1.521582:
        z += -0.2105484 * Q.soft1_pt - 0.06812377
    if 1.521582 <= Q.soft1_pt < 2.275391:
        z += 0.5153701 * Q.soft1_pt - 1.172668
    if Q.sum_pt < 1017.435:
        z += -0.008645657 * Q.sum_pt + 8.796392
    if Q.sum_pt_top30 >= 1191.938:
        z += 0.004775573 * Q.sum_pt_top30 - 5.692186
    if Q.z_top20_slots >= 0.8965411:
        z += 6.050461 * Q.z_top20_slots - 5.424487
    if Q.sum_pt_top5 >= 430.75:
        z += -0.000489376 * Q.sum_pt_top5 + 0.2107987
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.04105492 * Q.n_dr_0p1_0p2 - 0.3284394
    if Q.pt_entropy >= 2.07371:
        z += 0.9903908 * Q.pt_entropy - 2.053783
    if Q.girth2_top15 < 0.0007894752:
        z += -428.7 * Q.girth2_top15 + 0.3384481
    if Q.girth2_top20 < 0.001823079:
        z += -298.5443 * Q.girth2_top20 + 0.8426104
    if 0.001823079 <= Q.girth2_top20 < 0.007538019:
        z += -52.20365 * Q.girth2_top20 + 0.3935121
    if Q.psi_0p1 < 0.3628388:
        z += -0.5503041 * Q.psi_0p1 + 0.1996717
    if Q.psi_0p1 >= 0.8747961:
        z += 1.756543 * Q.psi_0p1 - 1.536617
    if Q.mass_top50 < 80.35535:
        z += -0.01418723 * Q.mass_top50 + 1.14002
    if Q.LHA < 0.2601462:
        z += -3.097315 * Q.LHA + 0.8057547
    if Q.sj3_mass3 < 6.160019:
        z += 0.04737605 * Q.sj3_mass3 - 0.2918374
    if Q.sj3_z3 < 0.08974974:
        z += -1.408604 * Q.sj3_z3 + 0.1264218
    if Q.e2 < 0.02515919:
        z += 22.85673 * Q.e2 - 0.575057
    if Q.e2 >= 0.06524004:
        z += 15.45552 * Q.e2 - 1.008318
    if Q.girth2_top5 < 0.0006570502:
        z += -448.7396 * Q.girth2_top5 + 0.2948444
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += 243.5154 * Q.mass_over_sum_pt_sq - 2.336534
    if Q.girth2_top40 < 0.008031986:
        z += -185.4781 * Q.girth2_top40 + 1.489758
    if Q.z_top50_slots >= 0.9586536:
        z += -27.48304 * Q.z_top50_slots + 26.34671
    if Q.n_for_90pct >= 11.0:
        z += -0.02334232 * Q.n_for_90pct + 0.2567656
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -3.50827 * Q.z_dr_0p2_0p4 + 0.3200444
    if Q.tau1 < 0.1953848:
        z += -11.24708 * Q.tau1 + 2.197508
    if Q.C2_b2 < 0.003468228:
        z += 55.26474 * Q.C2_b2 - 0.1916707
    if Q.D2 < 1.788105:
        z += -0.1763954 * Q.D2 + 0.3154136
    if Q.z_7 < 0.02274107:
        z += 19.21888 * Q.z_7 - 0.4370579
    if Q.n_particles > 38.0 and Q.mass_top15 < 57.87349:
        z += 0.0002578524 * (Q.n_particles - 38.0) * (57.87349 - Q.mass_top15)
    if Q.z_top30_slots > 0.9341838 and Q.max_pair_mass > 13.04793:
        z += 0.3045565 * (Q.z_top30_slots - 0.9341838) * (Q.max_pair_mass - 13.04793)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.1119555:
        z += 0.199281 * (Q.n_particles - 38.0) * (0.1119555 - Q.dr_0)
    if Q.mass_top20 < 47.88842 and Q.n_real_top40 > 29.0:
        z += 0.002987846 * (47.88842 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 38.0 and Q.soft1_pt < 2.275391:
        z += -0.01601772 * (Q.n_particles - 38.0) * (2.275391 - Q.soft1_pt)
    if Q.z_top30_slots > 0.9341838 and Q.C2 < 0.07279889:
        z += 200.3801 * (Q.z_top30_slots - 0.9341838) * (0.07279889 - Q.C2)
    if Q.sj3_mass1 < 32.50209 and Q.sj3_mass2 < 18.68222:
        z += -0.001345347 * (32.50209 - Q.sj3_mass1) * (18.68222 - Q.sj3_mass2)
    if Q.girth2_top15 < 0.003270031 and Q.psi_0p3 > 0.9973959:
        z += 41229.88 * (0.003270031 - Q.girth2_top15) * (Q.psi_0p3 - 0.9973959)
    if Q.M3 < 0.03187688 and Q.M2 > 0.05568888:
        z += -278.0018 * (0.03187688 - Q.M3) * (Q.M2 - 0.05568888)
    if Q.n_particles > 38.0 and Q.dr_1 < 0.1611545:
        z += 0.1474947 * (Q.n_particles - 38.0) * (0.1611545 - Q.dr_1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 > 7.407874:
        z += 0.3706634 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_3 - 7.407874)
    if Q.M3 < 0.03187688 and Q.psi_0p3 < 1.0:
        z += 96.97748 * (0.03187688 - Q.M3) * (1.0 - Q.psi_0p3)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_4 > 6.984554:
        z += 0.3644672 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_4 - 6.984554)
    if Q.pt_9 < 31.35938 and Q.dr1_12 < 0.3241858:
        z += 0.0361837 * (31.35938 - Q.pt_9) * (0.3241858 - Q.dr1_12)
    if Q.log_sum_pt < 7.139296 and Q.pt1_dr01 < 5.351077:
        z += -0.09430569 * (7.139296 - Q.log_sum_pt) * (5.351077 - Q.pt1_dr01)
    if Q.pt_9 < 31.35938 and Q.pair_mass_0_13 < 11.29505:
        z += 0.001293831 * (31.35938 - Q.pt_9) * (11.29505 - Q.pair_mass_0_13)
    if Q.n_particles > 38.0 and Q.sj3_dr12 < 0.2313408:
        z += 0.04121637 * (Q.n_particles - 38.0) * (0.2313408 - Q.sj3_dr12)
    if Q.sum_pt_top5 > 430.75 and Q.eta_0 < 0.02980347:
        z += -0.005703947 * (Q.sum_pt_top5 - 430.75) * (0.02980347 - Q.eta_0)
    if Q.z_top20_slots > 0.8965411 and Q.dr_2 < 0.03843804:
        z += -395.4911 * (Q.z_top20_slots - 0.8965411) * (0.03843804 - Q.dr_2)
    if Q.mass < 120.6 and Q.dr_2 < 0.03843804:
        z += 0.2122934 * (120.6 - Q.mass) * (0.03843804 - Q.dr_2)
    if Q.girth2_top20 < 0.007538019 and Q.eta_1 > -0.04302979:
        z += 469.3326 * (0.007538019 - Q.girth2_top20) * (Q.eta_1 - -0.04302979)
    if Q.sum_pt_top2 < 689.25 and Q.sj3_mass2 > 2.12465:
        z += 2.299574e-05 * (689.25 - Q.sum_pt_top2) * (Q.sj3_mass2 - 2.12465)
    if Q.sj2_mass1 < 30.26161 and Q.soft10_dr0 < 0.2366434:
        z += 0.03138356 * (30.26161 - Q.sj2_mass1) * (0.2366434 - Q.soft10_dr0)
    if Q.n_particles > 38.0 and Q.dr_9 < 0.1546554:
        z += 0.04020954 * (Q.n_particles - 38.0) * (0.1546554 - Q.dr_9)
    if Q.pt_entropy > 2.07371 and Q.soft3_dr < 0.3797:
        z += -0.6223645 * (Q.pt_entropy - 2.07371) * (0.3797 - Q.soft3_dr)
    if Q.sum_pt_top2 < 689.25 and Q.soft3_dr0 > 0.02918107:
        z += -0.001281918 * (689.25 - Q.sum_pt_top2) * (Q.soft3_dr0 - 0.02918107)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.408462
    if 6.903423 <= Q.log_sum_pt < 6.959294:
        z += 2.337713 * Q.log_sum_pt - 16.13822
    if Q.log_sum_pt >= 6.959294:
        z += -9.216833 * Q.log_sum_pt + 64.27326
    if Q.sum_pt < 972.0419:
        z += 0.07712352 * Q.sum_pt - 75.7888
    if 972.0419 <= Q.sum_pt < 1017.435:
        z += 0.0028475 * Q.sum_pt - 3.589391
    if 1017.435 <= Q.sum_pt < 1115.723:
        z += 0.01210135 * Q.sum_pt - 13.00458
    if 1115.723 <= Q.sum_pt < 1260.541:
        z += 0.007119322 * Q.sum_pt - 7.446017
    if Q.sum_pt >= 1260.541:
        z += 0.004271822 * Q.sum_pt - 3.856626
    if Q.n_particles < 51.0:
        z += 0.01164534 * Q.n_particles - 0.5939122
    if Q.sum_pt_top20 >= 1129.275:
        z += 0.001969127 * Q.sum_pt_top20 - 2.223686
    if Q.z_top20_slots >= 0.7516206:
        z += 0.8401728 * Q.z_top20_slots - 0.6314912
    if Q.mass < 78.26182:
        z += 0.03774766 * Q.mass - 3.649439
    if 78.26182 <= Q.mass < 91.19:
        z += 0.045336 * Q.mass - 4.243316
    if 91.19 <= Q.mass < 92.85979:
        z += 0.06535333 * Q.mass - 6.068696
    if Q.mass_top30 < 82.66587:
        z += -0.007260543 * Q.mass_top30 + 0.6001991
    if Q.mass_over_sum_pt < 0.09795415:
        z += -19.30346 * Q.mass_over_sum_pt + 1.890853
    if Q.sum_pt_top40 < 984.7009:
        z += -0.001378928 * Q.sum_pt_top40 + 1.435827
    if 984.7009 <= Q.sum_pt_top40 < 1041.263:
        z += -0.00221665 * Q.sum_pt_top40 + 2.260732
    if 1041.263 <= Q.sum_pt_top40 < 1069.671:
        z += -0.0008377215 * Q.sum_pt_top40 + 0.8249051
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.003762222 * Q.sum_pt_top40 - 4.095522
    if Q.sum_pt_top30 < 996.8867:
        z += 0.002081112 * Q.sum_pt_top30 - 2.074633
    if Q.girth2_top30 < 0.006363916:
        z += 76.12233 * Q.girth2_top30 - 0.4844361
    if Q.lam1 < 0.01174405:
        z += -33.99655 * Q.lam1 + 0.3992572
    if Q.sum_pt_top50 < 1048.098:
        z += -0.003662276 * Q.sum_pt_top50 + 3.751506
    if 1048.098 <= Q.sum_pt_top50 < 1078.994:
        z += 0.00281327 * Q.sum_pt_top50 - 3.035502
    if Q.mass_top50 < 92.16545:
        z += -0.01150131 * Q.mass_top50 + 1.060023
    if Q.log_sum_pt > 6.903423 and Q.mass_top40 > 77.93668:
        z += -0.02466664 * (Q.log_sum_pt - 6.903423) * (Q.mass_top40 - 77.93668)
    if Q.log_sum_pt > 6.903423 and Q.girth2_top15 < 0.02146578:
        z += 642.2581 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.girth2_top15)
    if Q.log_sum_pt > 7.062574 and Q.girth2_top30 > 0.008376291:
        z += 151.1944 * (Q.log_sum_pt - 7.062574) * (Q.girth2_top30 - 0.008376291)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p3 > 0.9299135:
        z += -38.83402 * (Q.log_sum_pt - 6.903423) * (Q.psi_0p3 - 0.9299135)
    if Q.sum_pt_top50 > 988.4554 and Q.girth2_top15 < 0.02146578:
        z += -0.4731257 * (Q.sum_pt_top50 - 988.4554) * (0.02146578 - Q.girth2_top15)
    if Q.sum_pt < 972.0419 and Q.e4 < 5.8505e-08:
        z += -1431230.0 * (972.0419 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.sum_pt < 1260.541 and Q.max_dr < 0.397021:
        z += -0.003757013 * (1260.541 - Q.sum_pt) * (0.397021 - Q.max_dr)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.169784
    if Q.lam2 < 0.0006154841:
        z += -569.372 * Q.lam2 + 0.3504394
    if Q.n_particles < 46.0:
        z += -0.01853046 * Q.n_particles + 0.8524012
    if Q.psi_0p2 >= 0.9985434:
        z += 105.1068 * Q.psi_0p2 - 104.9537
    if Q.tau21 < 0.3861957:
        z += 1.621362 * Q.tau21 - 0.6261632
    if Q.D2 < 2.410481:
        z += 0.281138 * Q.D2 - 0.6776778
    if Q.mass_over_sum_pt < 0.08665515:
        z += -12.35791 * Q.mass_over_sum_pt + 1.070876
    if Q.z_top30_slots >= 0.9734886:
        z += -5.62557 * Q.z_top30_slots + 5.476428
    if Q.girth2_top40 < 0.006026828:
        z += -153.8061 * Q.girth2_top40 + 1.134912
    if 0.006026828 <= Q.girth2_top40 < 0.008840538:
        z += -73.90554 * Q.girth2_top40 + 0.6533648
    if Q.mass_top50 < 79.21004:
        z += 0.005581185 * Q.mass_top50 - 0.4420859
    if Q.mass < 40.2:
        z += 0.003627073 * Q.mass + 0.6432414
    if 40.2 <= Q.mass < 53.87362:
        z += -0.009488967 * Q.mass + 1.170506
    if 53.87362 <= Q.mass < 86.4:
        z += -0.01743363 * Q.mass + 1.598514
    if 86.4 <= Q.mass < 92.85979:
        z += -0.01428039 * Q.mass + 1.326074
    if Q.girth2 < 0.009614971:
        z += 108.1296 * Q.girth2 - 1.039663
    if Q.sj2_mass1 < 27.56535:
        z += -0.008534345 * Q.sj2_mass1 + 0.2352522
    if Q.mass_top40 < 80.4:
        z += 0.02286829 * Q.mass_top40 - 1.83861
    if Q.lam1 < 0.001260456:
        z += -337.8557 * Q.lam1 + 0.4258521
    if Q.max_dr < 0.2738063:
        z += -2.189698 * Q.max_dr + 0.5995532
    if Q.girth2_top30 < 0.006363916:
        z += 112.6652 * Q.girth2_top30 - 0.716992
    if Q.z_dr_0p2_0p4 < 0.019523:
        z += 7.537192 * Q.z_dr_0p2_0p4 - 0.1471486
    if Q.girth2_top50 < 0.008124776:
        z += -60.20436 * Q.girth2_top50 + 0.489147
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_dr_0p1_0p2 > 9.0:
        z += -0.008844582 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 9.0)
    if Q.n_particles < 46.0 and Q.sum_pt_top30 > 800.732:
        z += 2.898151e-05 * (46.0 - Q.n_particles) * (Q.sum_pt_top30 - 800.732)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.eccentricity > 0.7333655:
        z += 0.2107914 * (5.0 - Q.n_dr_0p2_0p4) * (Q.eccentricity - 0.7333655)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.zdr_5 < 0.007099471:
        z += -8.065869 * (5.0 - Q.n_dr_0p2_0p4) * (0.007099471 - Q.zdr_5)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.psi_0p1 < 0.9538343:
        z += 0.2323833 * (5.0 - Q.n_dr_0p2_0p4) * (0.9538343 - Q.psi_0p1)
    if Q.tau21 < 0.3861957 and Q.lam1 > 0.007671243:
        z += -239.5858 * (0.3861957 - Q.tau21) * (Q.lam1 - 0.007671243)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.z_top50_slots > 0.978741:
        z += 3.234462 * (8.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.978741)
    if Q.n_dr_0p2_0p4 < 1.0 and Q.n_dr_0p4_up < 1.0:
        z += -0.2361418 * (1.0 - Q.n_dr_0p2_0p4) * (1.0 - Q.n_dr_0p4_up)
    if Q.D2 < 2.410481 and Q.D2_b2 < 5.142486:
        z += 0.06846599 * (2.410481 - Q.D2) * (5.142486 - Q.D2_b2)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.psi_0p3 > 0.9924477:
        z += 11.0426 * (5.0 - Q.n_dr_0p2_0p4) * (Q.psi_0p3 - 0.9924477)
    if Q.n_dr_0p1_0p2 < 10.0 and Q.psi_0p2 > 0.9313699:
        z += 0.4033343 * (10.0 - Q.n_dr_0p1_0p2) * (Q.psi_0p2 - 0.9313699)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.546955
    if Q.mass_top15 < 57.87349:
        z += -0.00533202 * Q.mass_top15 - 0.1240738
    if 57.87349 <= Q.mass_top15 < 69.02716:
        z += 0.008809855 * Q.mass_top15 - 0.9425135
    if 69.02716 <= Q.mass_top15 < 75.26407:
        z += 0.01663981 * Q.mass_top15 - 1.482993
    if 75.26407 <= Q.mass_top15 < 91.19:
        z += 0.01448036 * Q.mass_top15 - 1.320464
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.06644712 * Q.n_dr_0p2_0p4 + 1.240066
    if 15.0 <= Q.n_dr_0p2_0p4 < 26.0:
        z += -0.02212352 * Q.n_dr_0p2_0p4 + 0.5752116
    if Q.e3 >= 0.0001841806:
        z += 350.6957 * Q.e3 - 0.06459136
    if Q.mass < 74.25181:
        z += 0.08576495 * Q.mass - 5.880368
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.09861071 * Q.mass - 6.834188
    if 78.26182 <= Q.mass < 80.78464:
        z += 0.04485657 * Q.mass - 2.627291
    if 80.78464 <= Q.mass < 86.4:
        z += 0.01678186 * Q.mass - 0.3592867
    if 86.4 <= Q.mass < 92.85979:
        z += -0.02636326 * Q.mass + 3.368452
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.0447258 * Q.mass + 5.073594
    if 101.0497 <= Q.mass < 120.6:
        z += -0.01915591 * Q.mass + 2.489764
    if 120.6 <= Q.mass < 143.7876:
        z += -0.00774383 * Q.mass + 1.113467
    if Q.psi_0p3 >= 0.9973959:
        z += 119.78 * Q.psi_0p3 - 119.4681
    if Q.sj3_pair_mass_min >= 32.51366:
        z += 0.006400413 * Q.sj3_pair_mass_min - 0.2081009
    if Q.n_particles >= 22.0:
        z += -0.02015739 * Q.n_particles + 0.4434625
    if Q.girth2_top15 < 0.004855289:
        z += 105.1382 * Q.girth2_top15 - 0.5104764
    if 0.00727763 <= Q.girth2_top15 < 0.01563836:
        z += -90.73711 * Q.girth2_top15 + 0.6603511
    if Q.girth2_top15 >= 0.01563836:
        z += -2.084572 * Q.girth2_top15 - 0.7260288
    if 0.2232169 <= Q.sj2_dr < 0.2412757:
        z += 4.787043 * Q.sj2_dr - 1.068549
    if Q.sj2_dr >= 0.2412757:
        z += 0.04233742 * Q.sj2_dr + 0.07623316
    if 0.009606007 <= Q.e2_sq < 0.01396296:
        z += 165.4922 * Q.e2_sq - 1.589719
    if Q.e2_sq >= 0.01396296:
        z += 173.187 * Q.e2_sq - 1.697161
    if Q.mass_top40 < 62.55:
        z += 0.02438131 * Q.mass_top40 - 2.263286
    if 62.55 <= Q.mass_top40 < 67.72643:
        z += -0.001754821 * Q.mass_top40 - 0.6284713
    if 67.72643 <= Q.mass_top40 < 163.2541:
        z += 0.007823062 * Q.mass_top40 - 1.277147
    if Q.mass_top30 < 91.69753:
        z += 0.0116998 * Q.mass_top30 - 1.072842
    if Q.log_sum_pt >= 6.97212:
        z += -2.314414 * Q.log_sum_pt + 16.13637
    if Q.sd_rg >= 0.2042612:
        z += -1.044736 * Q.sd_rg + 0.213399
    if Q.girth < 0.05660088:
        z += 7.955188 * Q.girth - 0.2629024
    if 0.05660088 <= Q.girth < 0.07031778:
        z += -13.65966 * Q.girth + 0.9605167
    if Q.girth >= 0.1207452:
        z += -19.83808 * Q.girth + 2.395352
    if 0.008241985 <= Q.lam1 < 0.01174405:
        z += -98.60207 * Q.lam1 + 0.8126768
    if Q.lam1 >= 0.01174405:
        z += -65.44479 * Q.lam1 + 0.4232761
    if Q.psi_0p1 < 0.9909875:
        z += -0.427838 * Q.psi_0p1 + 0.4239821
    if Q.lam2 >= 0.001776308:
        z += -35.84906 * Q.lam2 + 0.06367898
    if Q.C2 >= 0.06655881:
        z += -2.503966 * Q.C2 + 0.166661
    if Q.n_dr_0p1_0p2 < 26.0:
        z += -0.006371046 * Q.n_dr_0p1_0p2 + 0.1656472
    if Q.psi_0p2 >= 0.9734513:
        z += -6.407831 * Q.psi_0p2 + 6.237711
    if Q.sum_pt >= 995.6769:
        z += 0.00098615 * Q.sum_pt - 0.9818868
    if Q.n_pt_above_10 >= 11.0:
        z += 0.01069734 * Q.n_pt_above_10 - 0.1176707
    if Q.D3 < 0.1900123:
        z += -0.4462115 * Q.D3 + 0.08478568
    if 0.007164202 <= Q.girth2_top5 < 0.0168592:
        z += -11.84656 * Q.girth2_top5 + 0.08487114
    if Q.girth2_top5 >= 0.0168592:
        z += -4.393662 * Q.girth2_top5 - 0.0407787
    if 0.02793599 <= Q.e2 < 0.04755309:
        z += 10.86487 * Q.e2 - 0.3035209
    if Q.e2 >= 0.04755309:
        z += -10.75447 * Q.e2 + 0.7245453
    if Q.mass_over_sum_pt < 0.06030419:
        z += -19.03022 * Q.mass_over_sum_pt + 1.147602
    if Q.n_dr_0p2_0p4 < 15.0 and Q.z_top50_slots < 1.0:
        z += -0.6009331 * (15.0 - Q.n_dr_0p2_0p4) * (1.0 - Q.z_top50_slots)
    if Q.psi_0p3 > 0.9973959 and Q.n_dr_0_0p05 < 13.0:
        z += -9.043998 * (Q.psi_0p3 - 0.9973959) * (13.0 - Q.n_dr_0_0p05)
    if Q.mass_top40 < 83.32554 and Q.zdr_0 > 0.001901263:
        z += 1.238813 * (83.32554 - Q.mass_top40) * (Q.zdr_0 - 0.001901263)
    if Q.mass < 120.6 and Q.psi_0p3 < 0.9943058:
        z += -0.1923813 * (120.6 - Q.mass) * (0.9943058 - Q.psi_0p3)
    if Q.n_particles > 22.0 and Q.n_dr_0_0p05 > 10.0:
        z += 0.0003209306 * (Q.n_particles - 22.0) * (Q.n_dr_0_0p05 - 10.0)
    if Q.n_particles > 22.0 and Q.soft1_pt < 2.275391:
        z += 0.004206846 * (Q.n_particles - 22.0) * (2.275391 - Q.soft1_pt)
    if Q.mass < 101.0497 and Q.zdr_0 > 0.0008718296:
        z += -0.6808734 * (101.0497 - Q.mass) * (Q.zdr_0 - 0.0008718296)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.max_dr > 0.1939977:
        z += -0.05166703 * (15.0 - Q.n_dr_0p2_0p4) * (Q.max_dr - 0.1939977)
    if Q.girth2_top15 > 0.00727763 and Q.pt_11 < 33.78125:
        z += -0.9094142 * (Q.girth2_top15 - 0.00727763) * (33.78125 - Q.pt_11)
    if Q.n_dr_0p2_0p4 < 26.0 and Q.n_dr_0p1_0p2 > 11.0:
        z += -0.0005806284 * (26.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 11.0)
    if Q.sj2_dr > 0.2232169 and Q.D2_b2 < 20.40995:
        z += -0.1067553 * (Q.sj2_dr - 0.2232169) * (20.40995 - Q.D2_b2)
    if Q.mass < 80.78464 and Q.D2_b2 < 3.852812:
        z += -0.005814323 * (80.78464 - Q.mass) * (3.852812 - Q.D2_b2)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.D2_b2 < 7.36624:
        z += -0.001837908 * (15.0 - Q.n_dr_0p2_0p4) * (7.36624 - Q.D2_b2)
    if Q.mass < 101.0497 and Q.D2_b2 < 3.852812:
        z += 0.005714849 * (101.0497 - Q.mass) * (3.852812 - Q.D2_b2)
    if Q.mass_top40 < 83.32554 and Q.D2_b2 < 3.014827:
        z += -0.008874662 * (83.32554 - Q.mass_top40) * (3.014827 - Q.D2_b2)
    if Q.mass < 92.85979 and Q.girth2_top2 < 0.005180665:
        z += 2.551271 * (92.85979 - Q.mass) * (0.005180665 - Q.girth2_top2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.5829186
    z += -0.02643767 * Q.n_particles + 1.692011
    if 907.9372 <= Q.sum_pt < 986.0565:
        z += 0.00969428 * Q.sum_pt - 8.801797
    if Q.sum_pt >= 986.0565:
        z += -0.005377054 * Q.sum_pt + 6.059391
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -14.91556 * Q.log_sum_pt + 103.0685
    if 6.920349 <= Q.log_sum_pt < 6.949481:
        z += -25.326 * Q.log_sum_pt + 175.1123
    if 6.949481 <= Q.log_sum_pt < 6.98945:
        z += -21.74155 * Q.log_sum_pt + 150.2023
    if Q.log_sum_pt >= 6.98945:
        z += -16.7991 * Q.log_sum_pt + 115.6573
    if 0.07696632 <= Q.mass_over_sum_pt < 0.1708801:
        z += -19.80674 * Q.mass_over_sum_pt + 1.524452
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 197.7102 * Q.mass_over_sum_pt - 35.64486
    if Q.girth2_top50 >= 0.01951641:
        z += -58.74206 * Q.girth2_top50 + 1.146434
    if Q.girth2 >= 0.004756928:
        z += 3.320572 * Q.girth2 - 0.01579572
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += 0.01229251 * Q.sum_pt_top50 - 11.48417
    if Q.sum_pt_top50 >= 959.0957:
        z += 0.01787547 * Q.sum_pt_top50 - 16.83877
    if Q.n_pt_above_1 >= 28.0:
        z += 0.01098792 * Q.n_pt_above_1 - 0.3076617
    if Q.sum_pt_top3 < 787.6281:
        z += 0.000867815 * Q.sum_pt_top3 - 0.6835155
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.0198407 * Q.n_dr_0p2_0p4 + 0.2182477
    if Q.n_dr_0p2_0p4 >= 21.0:
        z += -0.02239018 * Q.n_dr_0p2_0p4 + 0.4701938
    if 994.2695 <= Q.sum_pt_top40 < 1024.942:
        z += -0.0008428401 * Q.sum_pt_top40 + 0.8380102
    if 1024.942 <= Q.sum_pt_top40 < 1069.671:
        z += -0.002203052 * Q.sum_pt_top40 + 2.232149
    if Q.sum_pt_top40 >= 1069.671:
        z += -0.003425177 * Q.sum_pt_top40 + 3.539421
    if Q.sum_pt_top30 >= 933.1875:
        z += 0.001670331 * Q.sum_pt_top30 - 1.558732
    if Q.max_dr >= 0.2404747:
        z += -0.4755229 * Q.max_dr + 0.1143512
    if Q.z_top30_slots >= 0.9048492:
        z += -3.910743 * Q.z_top30_slots + 3.538632
    if Q.mass_top40 < 120.6:
        z += 0.009237668 * Q.mass_top40 - 1.385784
    if 120.6 <= Q.mass_top40 < 150.0144:
        z += -0.003175631 * Q.mass_top40 + 0.1112602
    if Q.mass_top40 >= 150.0144:
        z += -0.0124133 * Q.mass_top40 + 1.497044
    if Q.mass < 64.48544:
        z += -0.007030173 * Q.mass + 0.4533438
    if 74.25181 <= Q.mass < 91.19:
        z += 0.01741617 * Q.mass - 1.293182
    if 91.19 <= Q.mass < 172.4888:
        z += -0.0003251396 * Q.mass + 0.324648
    if 172.4888 <= Q.mass < 172.8:
        z += 0.1066931 * Q.mass - 18.1348
    if Q.mass >= 172.8:
        z += -0.05319352 * Q.mass + 9.493606
    if Q.girth2_top30 < 0.006363916:
        z += 13.87325 * Q.girth2_top30 - 0.2507838
    if 0.006363916 <= Q.girth2_top30 < 0.007463985:
        z += 141.1555 * Q.girth2_top30 - 1.060797
    if 0.007463985 <= Q.girth2_top30 < 0.01807679:
        z += 27.20363 * Q.girth2_top30 - 0.2102623
    if Q.girth2_top30 >= 0.01807679:
        z += 13.33038 * Q.girth2_top30 + 0.04052148
    if Q.z_dr_0_0p05 >= 0.9084912:
        z += 3.309452 * Q.z_dr_0_0p05 - 3.006608
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -408.5088 * Q.mass_over_sum_pt_sq + 11.92846
    if Q.sj2_mass1 < 23.31647:
        z += -0.009429108 * Q.sj2_mass1 + 0.2198536
    if Q.tau21 < 0.5494307:
        z += 0.7741124 * Q.tau21 - 0.4253211
    if 69.65633 <= Q.sd_mass < 86.4:
        z += 0.01204094 * Q.sd_mass - 0.8387278
    if Q.sd_mass >= 86.4:
        z += -0.0006298935 * Q.sd_mass + 0.2560324
    if Q.mass_top50 >= 157.5448:
        z += -0.02750142 * Q.mass_top50 + 4.332706
    if Q.sum_pt_top10 >= 943.6922:
        z += -0.001696047 * Q.sum_pt_top10 + 1.600546
    if Q.z_11 < 0.01375115:
        z += -80.03162 * Q.z_11 + 1.100527
    if Q.e2 < 0.01541561:
        z += 17.45737 * Q.e2 + 0.2205416
    if 0.01541561 <= Q.e2 < 0.03875945:
        z += -20.97589 * Q.e2 + 0.8130137
    if Q.M3 < 0.03787151:
        z += -3.822879 * Q.M3 + 0.1447782
    if Q.n_dr_0p1_0p2 < 8.0:
        z += -0.01177531 * Q.n_dr_0p1_0p2 - 0.0835528
    if 8.0 <= Q.n_dr_0p1_0p2 < 21.0:
        z += 0.01367349 * Q.n_dr_0p1_0p2 - 0.2871432
    if Q.mass_top10 >= 71.781:
        z += 0.004950045 * Q.mass_top10 - 0.3553192
    if Q.z_dr_0p2_0p4 < 0.1937447:
        z += 0.6687849 * Q.z_dr_0p2_0p4 - 0.1295735
    if Q.pt_11 < 14.14062:
        z += 0.05246283 * Q.pt_11 - 0.7418572
    if Q.D2 < 1.976207:
        z += -0.1911588 * Q.D2 + 0.3777695
    if Q.z_top20_slots >= 0.9102775:
        z += -1.302906 * Q.z_top20_slots + 1.186006
    if Q.psi_0p1 < 0.6635952:
        z += 0.2896958 * Q.psi_0p1 - 0.1922408
    if Q.z_top50_slots >= 0.9906378:
        z += -21.33309 * Q.z_top50_slots + 21.13337
    if Q.psi_0p3 >= 0.9638082:
        z += 4.334316 * Q.psi_0p3 - 4.177449
    if Q.n_particles < 64.0 and Q.D2 < 2.178951:
        z += -0.01262696 * (64.0 - Q.n_particles) * (2.178951 - Q.D2)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 14913.5 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    if Q.psi_0p3 > 0.9973959 and Q.D2 < 3.345339:
        z += 42.76685 * (Q.psi_0p3 - 0.9973959) * (3.345339 - Q.D2)
    if Q.log_sum_pt > 6.910131 and Q.dr_11 < 0.2535773:
        z += -2.603775 * (Q.log_sum_pt - 6.910131) * (0.2535773 - Q.dr_11)
    if Q.log_sum_pt > 6.910131 and Q.absphi_13 < 0.1958008:
        z += -4.13683 * (Q.log_sum_pt - 6.910131) * (0.1958008 - Q.absphi_13)
    if Q.sum_pt_top40 > 994.2695 and Q.dr1_12 < 0.3241858:
        z += -0.001940352 * (Q.sum_pt_top40 - 994.2695) * (0.3241858 - Q.dr1_12)
    return max(0.0, z)


def neuron_6(Q):
    z = -0.07771583
    if Q.mass_top50 < 71.79516:
        z += -0.03844262 * Q.mass_top50 + 4.15205
    if 71.79516 <= Q.mass_top50 < 168.9698:
        z += -0.01314102 * Q.mass_top50 + 2.335518
    if 168.9698 <= Q.mass_top50 < 172.8:
        z += -0.03004591 * Q.mass_top50 + 5.191933
    if Q.mass < 86.4:
        z += 0.006437268 * Q.mass - 2.324617
    if 86.4 <= Q.mass < 92.85979:
        z += 0.0401907 * Q.mass - 5.240914
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.1043975 * Q.mass - 11.20314
    if 101.0497 <= Q.mass < 120.6:
        z += 0.03344232 * Q.mass - 4.033144
    if Q.girth2_top15 < 0.002197765:
        z += -74.9618 * Q.girth2_top15 + 0.1647484
    if Q.sj3_pair_mass_min >= 29.00832:
        z += -0.01212167 * Q.sj3_pair_mass_min + 0.3516294
    if Q.sj3_mass1 >= 21.11128:
        z += -0.01262385 * Q.sj3_mass1 + 0.2665056
    if Q.log_sum_pt < 6.811175:
        z += 5.224311 * Q.log_sum_pt - 35.80941
    if 6.811175 <= Q.log_sum_pt < 7.017258:
        z += 1.095253 * Q.log_sum_pt - 7.685672
    if Q.tau1 < 0.05444509:
        z += -8.319003 * Q.tau1 + 0.524998
    if 0.05444509 <= Q.tau1 < 0.06310829:
        z += 0.8182917 * Q.tau1 + 0.02751716
    if Q.tau1 >= 0.06310829:
        z += 9.137295 * Q.tau1 - 0.4974809
    if Q.M2 < 0.1011005:
        z += 3.323159 * Q.M2 - 0.3359729
    if Q.sj3_dr_min >= 0.1204829:
        z += -0.8786048 * Q.sj3_dr_min + 0.1058569
    if Q.z_top40_slots < 0.9300465:
        z += -4.951291 * Q.z_top40_slots + 4.604931
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -1.319646 * Q.z_dr_0p1_0p2 + 0.1588111
    if Q.mass_over_sum_pt < 0.02682209:
        z += 6.891417 * Q.mass_over_sum_pt - 0.5740512
    if 0.02682209 <= Q.mass_over_sum_pt < 0.08329945:
        z += 28.65728 * Q.mass_over_sum_pt - 1.157857
    if 0.08329945 <= Q.mass_over_sum_pt < 0.09795415:
        z += 21.76586 * Q.mass_over_sum_pt - 0.5838058
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -30.86495 * Q.mass_over_sum_pt + 4.5716
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -55.35698 * Q.mass_over_sum_pt + 7.467193
    if Q.girth2_top30 < 0.00375223:
        z += -69.95614 * Q.girth2_top30 - 0.9165757
    if 0.00375223 <= Q.girth2_top30 < 0.008376291:
        z += 82.31091 * Q.girth2_top30 - 1.487917
    if 0.008376291 <= Q.girth2_top30 < 0.01807679:
        z += 134.3831 * Q.girth2_top30 - 1.924089
    if 0.01807679 <= Q.girth2_top30 < 0.02809026:
        z += 52.07219 * Q.girth2_top30 - 0.4361718
    if Q.girth2_top30 >= 0.02809026:
        z += -31.76278 * Q.girth2_top30 + 1.918774
    if Q.girth2_top20 >= 0.008031209:
        z += 71.68741 * Q.girth2_top20 - 0.5757366
    if Q.lam1 < 0.003811746:
        z += -326.28 * Q.lam1 + 1.347474
    if 0.003811746 <= Q.lam1 < 0.007259287:
        z += -138.541 * Q.lam1 + 0.6318609
    if 0.007259287 <= Q.lam1 < 0.02048524:
        z += 28.26626 * Q.lam1 - 0.579041
    if Q.girth2_top50 < 0.003418057:
        z += 389.4682 * Q.girth2_top50 - 1.563631
    if 0.003418057 <= Q.girth2_top50 < 0.004573744:
        z += 201.098 * Q.girth2_top50 - 0.9197709
    if Q.lam2 < 0.003687605:
        z += -149.8813 * Q.lam2 + 0.5527032
    if Q.mass_top15 < 30.35994:
        z += 0.001568723 * Q.mass_top15 + 0.3283808
    if 30.35994 <= Q.mass_top15 < 91.19:
        z += -0.006181273 * Q.mass_top15 + 0.5636703
    if Q.LHA >= 0.3719813:
        z += 12.3655 * Q.LHA - 4.599736
    if Q.girth2_top3 < 0.001592178:
        z += 160.6715 * Q.girth2_top3 - 0.2558176
    if Q.e2 < 0.02793599:
        z += -14.95572 * Q.e2 + 0.9757117
    if 0.02793599 <= Q.e2 < 0.04755309:
        z += -36.2932 * Q.e2 + 1.571795
    if 0.04755309 <= Q.e2 < 0.06524004:
        z += -7.753627 * Q.e2 + 0.2146506
    if Q.e2 >= 0.06524004:
        z += 7.202093 * Q.e2 - 0.7610611
    if Q.C2 < 0.04704395:
        z += -5.950109 * Q.C2 + 0.2799166
    if Q.girth2_top40 >= 0.008031986:
        z += 68.31851 * Q.girth2_top40 - 0.5487333
    if Q.sum_pt_top20 < 1129.275:
        z += 0.001500451 * Q.sum_pt_top20 - 1.694421
    if Q.z_dr_0p2_0p4 < 0.03700182:
        z += -4.207995 * Q.z_dr_0p2_0p4 + 0.1557035
    if Q.D2 < 1.409617:
        z += 0.2265395 * Q.D2 - 0.319334
    if Q.girth >= 0.02085222:
        z += -8.168769 * Q.girth + 0.1703369
    if Q.sum_pt < 1260.541:
        z += -0.0008019512 * Q.sum_pt + 1.010892
    if Q.mass < 120.6 and Q.sum_pt < 1007.788:
        z += 0.0001045414 * (120.6 - Q.mass) * (1007.788 - Q.sum_pt)
    if Q.sj3_pair_mass_min > 29.00832 and Q.psi_0p3 > 0.9896594:
        z += 0.705649 * (Q.sj3_pair_mass_min - 29.00832) * (Q.psi_0p3 - 0.9896594)
    if Q.mass_over_sum_pt > 0.1182259 and Q.C2_b2 > 0.02704832:
        z += 380.5478 * (Q.mass_over_sum_pt - 0.1182259) * (Q.C2_b2 - 0.02704832)
    if Q.e2 > 0.05557149 and Q.zdr_0 > 0.001369707:
        z += 544.0334 * (Q.e2 - 0.05557149) * (Q.zdr_0 - 0.001369707)
    if Q.e2 > 0.05557149 and Q.D2_b2 < 1.67722:
        z += 15.62194 * (Q.e2 - 0.05557149) * (1.67722 - Q.D2_b2)
    if Q.girth2_top30 > 0.008376291 and Q.D2_b2 > 1.67722:
        z += -13.0615 * (Q.girth2_top30 - 0.008376291) * (Q.D2_b2 - 1.67722)
    if Q.mass_over_sum_pt > 0.1182259 and Q.D2_b2 < 7.36624:
        z += -2.520029 * (Q.mass_over_sum_pt - 0.1182259) * (7.36624 - Q.D2_b2)
    if Q.e3 < 0.0003372339 and Q.dr_max_012 > 0.004415714:
        z += -2779.814 * (0.0003372339 - Q.e3) * (Q.dr_max_012 - 0.004415714)
    if Q.girth2_top30 > 0.008376291 and Q.orientation_deg > 26.6454:
        z += -0.2530078 * (Q.girth2_top30 - 0.008376291) * (Q.orientation_deg - 26.6454)
    if Q.girth2_top20 > 0.008031209 and Q.C2_b2 > 0.0008187529:
        z += -693.6863 * (Q.girth2_top20 - 0.008031209) * (Q.C2_b2 - 0.0008187529)
    if Q.mass_top15 < 91.19 and Q.C2_b2 < 0.04008677:
        z += -0.06655912 * (91.19 - Q.mass_top15) * (0.04008677 - Q.C2_b2)
    if Q.mass_over_sum_pt > 0.02682209 and Q.z_dr_0p05_0p1 < 0.7108211:
        z += -5.070729 * (Q.mass_over_sum_pt - 0.02682209) * (0.7108211 - Q.z_dr_0p05_0p1)
    if Q.e2 > 0.02793599 and Q.zdr_1 < 0.01718455:
        z += -482.2806 * (Q.e2 - 0.02793599) * (0.01718455 - Q.zdr_1)
    if Q.mass_over_sum_pt > 0.09795415 and Q.soft8_z < 0.001160626:
        z += -10929.89 * (Q.mass_over_sum_pt - 0.09795415) * (0.001160626 - Q.soft8_z)
    if Q.e2 > 0.02793599 and Q.max_dr < 0.4357228:
        z += 65.77059 * (Q.e2 - 0.02793599) * (0.4357228 - Q.max_dr)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.8300914
    if Q.tau21_b2 < 0.2352054:
        z += -9.92863 * Q.tau21_b2 + 2.335268
    if Q.girth2 < 0.006403325:
        z += 118.4303 * Q.girth2 - 0.3234047
    if 0.006403325 <= Q.girth2 < 0.007877041:
        z += 32.92186 * Q.girth2 + 0.2241334
    if 0.007877041 <= Q.girth2 < 0.008190222:
        z += 163.0224 * Q.girth2 - 0.8006742
    if 0.008190222 <= Q.girth2 < 0.009614971:
        z += -375.1647 * Q.girth2 + 3.607198
    if Q.mass_over_sum_pt < 0.1182259:
        z += -30.15753 * Q.mass_over_sum_pt + 3.565402
    if Q.mass < 78.26182:
        z += 0.2366436 * Q.mass - 18.28468
    if 78.26182 <= Q.mass < 82.85409:
        z += 0.1782434 * Q.mass - 13.71418
    if 82.85409 <= Q.mass < 91.19:
        z += 0.02424574 * Q.mass - 0.9548375
    if 91.19 <= Q.mass < 92.85979:
        z += -0.01544763 * Q.mass + 2.664801
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.02444142 * Q.mass - 1.039289
    if 101.0497 <= Q.mass < 120.6:
        z += -0.07317077 * Q.mass + 8.824395
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.1385494 * Q.n_dr_0p2_0p4 + 1.049887
    if 6.0 <= Q.n_dr_0p2_0p4 < 9.0:
        z += -0.07286367 * Q.n_dr_0p2_0p4 + 0.655773
    if 0.9973959 <= Q.psi_0p3 < 0.9980008:
        z += 145.8992 * Q.psi_0p3 - 145.5193
    if 0.9980008 <= Q.psi_0p3 < 0.9995915:
        z += -383.2341 * Q.psi_0p3 + 382.5563
    if Q.psi_0p3 >= 0.9995915:
        z += -764.8767 * Q.psi_0p3 + 764.0429
    if Q.tau21 < 0.3861957:
        z += 1.255969 * Q.tau21 - 0.48505
    if Q.z_dr_0_0p05 < 0.09573228:
        z += -2.183471 * Q.z_dr_0_0p05 + 0.2090286
    if Q.lam2 < 0.000404306:
        z += -641.8649 * Q.lam2 + 0.2595098
    if Q.e2_sq < 0.00363788:
        z += 707.3696 * Q.e2_sq - 2.573326
    if Q.tau1 < 0.07708632:
        z += 7.524137 * Q.tau1 - 1.557568
    if 0.07708632 <= Q.tau1 < 0.08786745:
        z += 21.76863 * Q.tau1 - 2.655624
    if 0.08786745 <= Q.tau1 < 0.09591084:
        z += 26.82485 * Q.tau1 - 3.099901
    if 0.09591084 <= Q.tau1 < 0.1072713:
        z += 46.39861 * Q.tau1 - 4.977237
    if Q.n_dr_0_0p05 < 18.0:
        z += 0.01847601 * Q.n_dr_0_0p05 - 0.3325682
    if Q.lam1 < 0.006189818:
        z += 16.62009 * Q.lam1 - 0.3385776
    if 0.006189818 <= Q.lam1 < 0.008241985:
        z += 181.6996 * Q.lam1 - 1.36039
    if 0.008241985 <= Q.lam1 < 0.01174405:
        z += -39.16995 * Q.lam1 + 0.4600139
    if Q.sd_mass < 45.595:
        z += -0.004559731 * Q.sd_mass - 0.1935932
    if 45.595 <= Q.sd_mass < 69.65633:
        z += -0.005849331 * Q.sd_mass - 0.1347939
    if 69.65633 <= Q.sd_mass < 86.4:
        z += 0.02924177 * Q.sd_mass - 2.579111
    if 86.4 <= Q.sd_mass < 98.05743:
        z += -0.001289599 * Q.sd_mass + 0.05879929
    if Q.sd_mass >= 98.05743:
        z += -0.1480089 * Q.sd_mass + 14.44572
    if 73.35236 <= Q.mass_top20 < 85.79457:
        z += 0.006378733 * Q.mass_top20 - 0.4678952
    if Q.mass_top20 >= 85.79457:
        z += 0.01953278 * Q.mass_top20 - 1.596441
    if Q.psi_0p2 >= 0.9313699:
        z += -4.482201 * Q.psi_0p2 + 4.174587
    if Q.max_dr < 0.1939977:
        z += 1.596298 * Q.max_dr + 0.03046809
    if 0.1939977 <= Q.max_dr < 0.2982:
        z += -3.264283 * Q.max_dr + 0.9734093
    if Q.sd_rg < 0.1596365:
        z += -0.5295718 * Q.sd_rg - 0.05239971
    if 0.1596365 <= Q.sd_rg < 0.1881908:
        z += 7.001627 * Q.sd_rg - 1.254654
    if 0.1881908 <= Q.sd_rg < 0.2639816:
        z += -0.8310764 * Q.sd_rg + 0.2193889
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1937688:
        z += -26.84219 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1937688)
    if Q.tau21_b2 < 0.2352054 and Q.mass_over_sum_pt_sq < 0.007873266:
        z += -1005.32 * (0.2352054 - Q.tau21_b2) * (0.007873266 - Q.mass_over_sum_pt_sq)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9638082:
        z += 9.433125 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.e3 < 7.876005e-05:
        z += -1381.177 * (6.0 - Q.n_dr_0p2_0p4) * (7.876005e-05 - Q.e3)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9777125:
        z += 5.495725 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 82.85409 and Q.psi_0p3 < 0.9985421:
        z += 2.807716 * (82.85409 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9777125:
        z += -16.48547 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9777125:
        z += 5.212725 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9980008:
        z += -66.73368 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9980008:
        z += 56.93549 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 1260.541:
        z += -0.01520078 * (0.2352054 - Q.tau21_b2) * (1260.541 - Q.sum_pt)
    if Q.tau21_b2 < 0.2352054 and Q.z_2 < 0.1063277:
        z += -21.3131 * (0.2352054 - Q.tau21_b2) * (0.1063277 - Q.z_2)
    if Q.mass < 92.85979 and Q.psi_0p3 > 0.9638082:
        z += -6.701954 * (92.85979 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.tau21_b2 < 0.2352054 and Q.orientation_deg > -9.840088:
        z += -0.01603749 * (0.2352054 - Q.tau21_b2) * (Q.orientation_deg - -9.840088)
    if Q.girth2 < 0.006403325 and Q.psi_0p3 > 0.9980008:
        z += 83619.85 * (0.006403325 - Q.girth2) * (Q.psi_0p3 - 0.9980008)
    if Q.tau21_b2 < 0.2352054 and Q.psi_0p3 > 0.9299135:
        z += -37.40229 * (0.2352054 - Q.tau21_b2) * (Q.psi_0p3 - 0.9299135)
    if Q.mass < 92.85979 and Q.z_dr_0_0p05 < 0.4947602:
        z += -0.1339489 * (92.85979 - Q.mass) * (0.4947602 - Q.z_dr_0_0p05)
    if Q.lam1 < 0.008241985 and Q.zdr_0 > 0.01564747:
        z += -23033.84 * (0.008241985 - Q.lam1) * (Q.zdr_0 - 0.01564747)
    if Q.psi_0p3 > 0.9973959 and Q.z_top50_slots > 0.978741:
        z += 8751.724 * (Q.psi_0p3 - 0.9973959) * (Q.z_top50_slots - 0.978741)
    return max(0.0, z)


def neuron_8(Q):
    z = 1.877666
    if 0.07696632 <= Q.mass_over_sum_pt < 0.1182259:
        z += -18.66767 * Q.mass_over_sum_pt + 1.436782
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -35.91783 * Q.mass_over_sum_pt + 3.476199
    if Q.sum_pt_top40 < 1001.523:
        z += 0.003408914 * Q.sum_pt_top40 - 3.52217
    if 1001.523 <= Q.sum_pt_top40 < 1024.942:
        z += 0.004614286 * Q.sum_pt_top40 - 4.729377
    if Q.girth2_top40 < 0.005196966:
        z += -44.0942 * Q.girth2_top40 + 0.3898165
    if 0.005196966 <= Q.girth2_top40 < 0.008840538:
        z += 23.59124 * Q.girth2_top40 + 0.03805757
    if Q.girth2_top40 >= 0.008840538:
        z += 67.68544 * Q.girth2_top40 - 0.3517589
    if Q.n_dr_0p2_0p4 < 3.0:
        z += 0.02453491 * Q.n_dr_0p2_0p4 - 0.3680236
    if 3.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += 0.04301 * Q.n_dr_0p2_0p4 - 0.4234489
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += 0.01847509 * Q.n_dr_0p2_0p4 - 0.05542527
    if 64.48544 <= Q.mass < 74.25181:
        z += 0.02432797 * Q.mass - 1.5688
    if 74.25181 <= Q.mass < 87.36377:
        z += 0.05654689 * Q.mass - 3.961113
    if 87.36377 <= Q.mass < 101.0497:
        z += 0.09675331 * Q.mass - 7.473698
    if 101.0497 <= Q.mass < 125.1:
        z += 0.04582163 * Q.mass - 2.327066
    if Q.mass >= 125.1:
        z += -0.006657265 * Q.mass + 4.238044
    if Q.sum_pt < 1017.435:
        z += -0.03131322 * Q.sum_pt + 32.0762
    if 1017.435 <= Q.sum_pt < 1028.184:
        z += -0.02019245 * Q.sum_pt + 20.76155
    if 0.006043209 <= Q.girth2_top20 < 0.008031209:
        z += -106.5096 * Q.girth2_top20 + 0.6436597
    if Q.girth2_top20 >= 0.008031209:
        z += 45.19176 * Q.girth2_top20 - 0.5746856
    if Q.log_sum_pt < 6.930088:
        z += 14.8715 * Q.log_sum_pt - 103.0608
    if Q.log_sum_pt >= 7.139296:
        z += -2.20725 * Q.log_sum_pt + 15.75821
    if Q.girth2_top30 < 0.007463985:
        z += -193.6351 * Q.girth2_top30 + 1.445289
    if Q.width < 0.009614971:
        z += 403.6885 * Q.width - 3.881453
    if Q.girth2 < 0.007877041:
        z += -160.4424 * Q.girth2 + 1.263812
    if Q.sj2_dr >= 0.2232169:
        z += 2.124167 * Q.sj2_dr - 0.4741499
    if Q.z_dr_0p1_0p2 >= 0.3340477:
        z += 0.8984174 * Q.z_dr_0p1_0p2 - 0.3001143
    if Q.n_for_90pct < 7.0:
        z += 0.02829529 * Q.n_for_90pct - 1.103516
    if 7.0 <= Q.n_for_90pct < 39.0:
        z += -0.01326513 * Q.n_for_90pct - 0.8125935
    if Q.n_for_90pct >= 39.0:
        z += -0.04156043 * Q.n_for_90pct + 0.290923
    if Q.psi_0p3 >= 0.9943058:
        z += -28.57408 * Q.psi_0p3 + 28.41137
    if 0.04755309 <= Q.e2 < 0.06524004:
        z += 34.82302 * Q.e2 - 1.655942
    if Q.e2 >= 0.06524004:
        z += 58.67783 * Q.e2 - 3.212231
    if Q.lam1 < 0.007671243:
        z += -113.9218 * Q.lam1 + 0.7785141
    if 0.007671243 <= Q.lam1 < 0.01174405:
        z += 23.42544 * Q.lam1 - 0.2751096
    if Q.tau2 < 0.0795038:
        z += -3.131002 * Q.tau2 + 0.2489266
    if Q.C2 >= 0.06655881:
        z += 5.801353 * Q.C2 - 0.3861312
    if 5.13841e-05 <= Q.e3 < 0.0003372339:
        z += -2429.404 * Q.e3 + 0.1248328
    if Q.e3 >= 0.0003372339:
        z += -1251.085 * Q.e3 - 0.2725364
    if 82.04491 <= Q.mass_top50 < 117.0487:
        z += -0.03043775 * Q.mass_top50 + 2.497263
    if Q.mass_top50 >= 117.0487:
        z += 0.003994247 * Q.mass_top50 - 1.532959
    if Q.mass_top20 >= 119.2969:
        z += -0.01256272 * Q.mass_top20 + 1.498693
    if Q.n_dr_0p1_0p2 >= 19.0:
        z += 0.01732325 * Q.n_dr_0p1_0p2 - 0.3291418
    if Q.z_top15_slots < 0.8316924:
        z += -1.152501 * Q.z_top15_slots + 0.9585265
    if Q.psi_0p1 < 0.3628388:
        z += 0.9662785 * Q.psi_0p1 - 0.3506033
    if Q.z_top50_slots >= 0.9704436:
        z += -25.7789 * Q.z_top50_slots + 25.01697
    if Q.girth2_top15 < 0.02146578:
        z += 29.79818 * Q.girth2_top15 - 0.6396412
    if Q.sum_pt_top30 >= 978.0762:
        z += 0.001826433 * Q.sum_pt_top30 - 1.78639
    if Q.sum_pt_top50 >= 889.8503:
        z += -0.0009281069 * Q.sum_pt_top50 + 0.8258762
    if Q.n_particles >= 38.0:
        z += 0.01010072 * Q.n_particles - 0.3838274
    if Q.mass_over_sum_pt > 0.07696632 and Q.sum_pt < 1115.723:
        z += 0.2468195 * (Q.mass_over_sum_pt - 0.07696632) * (1115.723 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.z_dr_0p1_0p2 > 0.3340477:
        z += 0.07153378 * (15.0 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p1_0p2 - 0.3340477)
    if Q.girth2_top40 > 0.005196966 and Q.log_sum_pt < 7.017258:
        z += -667.5116 * (Q.girth2_top40 - 0.005196966) * (7.017258 - Q.log_sum_pt)
    if Q.sum_pt < 1017.435 and Q.n_for_90pct < 13.0:
        z += -0.0004561279 * (1017.435 - Q.sum_pt) * (13.0 - Q.n_for_90pct)
    if Q.mass > 64.48544 and Q.zdr_0 > 0.0008718296:
        z += -0.06492222 * (Q.mass - 64.48544) * (Q.zdr_0 - 0.0008718296)
    if Q.girth2_top30 < 0.007463985 and Q.sj2_dr > 0.1512157:
        z += -280.378 * (0.007463985 - Q.girth2_top30) * (Q.sj2_dr - 0.1512157)
    if Q.sum_pt < 1017.435 and Q.dr_max_012 > 0.1828389:
        z += -0.1769754 * (1017.435 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    if Q.sum_pt < 1028.184 and Q.dr_max_012 > 0.1828389:
        z += 0.1595312 * (1028.184 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.797959
    if Q.mass_top40 < 80.89043:
        z += 0.002803249 * Q.mass_top40 - 1.421166
    if 80.89043 <= Q.mass_top40 < 163.2541:
        z += 0.01450165 * Q.mass_top40 - 2.367455
    if Q.sum_pt_top40 < 956.2133:
        z += -0.008543123 * Q.sum_pt_top40 + 8.169048
    if Q.girth2_top40 < 0.00217213:
        z += 103.5067 * Q.girth2_top40 + 0.2459248
    if 0.00217213 <= Q.girth2_top40 < 0.006259772:
        z += -115.1654 * Q.girth2_top40 + 0.720909
    if Q.mass_over_sum_pt >= 0.1708801:
        z += -54.41776 * Q.mass_over_sum_pt + 9.298913
    if Q.mass < 64.48544:
        z += -0.04620976 * Q.mass + 5.576532
    if 64.48544 <= Q.mass < 82.85409:
        z += -0.08526501 * Q.mass + 8.095027
    if 82.85409 <= Q.mass < 92.85979:
        z += -0.02739616 * Q.mass + 3.300356
    if 92.85979 <= Q.mass < 143.7876:
        z += 0.03643212 * Q.mass - 2.626724
    if 143.7876 <= Q.mass < 160.8:
        z += -0.07141353 * Q.mass + 12.88014
    if 160.8 <= Q.mass < 162.8363:
        z += -0.2267769 * Q.mass + 37.86257
    if 162.8363 <= Q.mass < 172.8:
        z += -0.09384623 * Q.mass + 16.21663
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -3.033251 * Q.z_dr_0_0p05 + 2.665942
    if 0.09749958 <= Q.girth < 0.1402186:
        z += 29.56463 * Q.girth - 2.882539
    if Q.girth >= 0.1402186:
        z += -39.83962 * Q.girth + 6.849227
    if Q.z_top5 >= 0.7963975:
        z += 3.77469 * Q.z_top5 - 3.006154
    if Q.psi_0p3 >= 0.9943058:
        z += 37.95817 * Q.psi_0p3 - 37.74203
    if Q.tau1 < 0.05444509:
        z += 17.94511 * Q.tau1 - 0.9770234
    if Q.lam1 < 0.004673423:
        z += -132.7175 * Q.lam1 + 0.6202451
    if Q.lam1 >= 0.01174405:
        z += 62.41311 * Q.lam1 - 0.7329827
    if Q.girth2_top30 >= 0.0008564881:
        z += -27.96024 * Q.girth2_top30 + 0.02394761
    if Q.D2 >= 2.178951:
        z += 0.01753274 * Q.D2 - 0.03820297
    if Q.mass_top50 < 92.16545:
        z += -0.002087284 * Q.mass_top50 + 2.252722
    if 92.16545 <= Q.mass_top50 < 138.8977:
        z += -0.04408834 * Q.mass_top50 + 6.123768
    if Q.mass_top30 < 73.33139:
        z += -0.009531924 * Q.mass_top30 + 0.6989892
    if Q.n_for_90pct >= 35.0:
        z += -0.02950861 * Q.n_for_90pct + 1.032801
    if Q.LHA >= 0.404204:
        z += 35.46211 * Q.LHA - 14.33393
    if Q.z_top40_slots >= 0.9574183:
        z += -7.610425 * Q.z_top40_slots + 7.286361
    if Q.e3 >= 0.0003372339:
        z += 1044.822 * Q.e3 - 0.3523495
    if Q.sum_pt < 986.0565:
        z += -0.01913989 * Q.sum_pt + 18.87301
    if Q.log_sum_pt < 6.903423:
        z += 15.4205 * Q.log_sum_pt - 106.4542
    if Q.sum_pt_top40 < 956.2133 and Q.psi_0p1 > 0.9184255:
        z += -0.05123008 * (956.2133 - Q.sum_pt_top40) * (Q.psi_0p1 - 0.9184255)
    if Q.sum_pt_top40 < 956.2133 and Q.soft4_pt > 1.789258:
        z += -0.004951749 * (956.2133 - Q.sum_pt_top40) * (Q.soft4_pt - 1.789258)
    if Q.mass_top40 < 80.89043 and Q.sum_pt < 1034.834:
        z += -0.0001960662 * (80.89043 - Q.mass_top40) * (1034.834 - Q.sum_pt)
    if Q.mass_top40 < 80.89043 and Q.sum_pt_top40 < 906.6023:
        z += 0.0002209588 * (80.89043 - Q.mass_top40) * (906.6023 - Q.sum_pt_top40)
    if Q.sum_pt_top40 < 956.2133 and Q.soft3_pt > 2.873047:
        z += 0.00391624 * (956.2133 - Q.sum_pt_top40) * (Q.soft3_pt - 2.873047)
    if Q.mass < 92.85979 and Q.z_top50_slots > 0.9995789:
        z += -9.015381 * (92.85979 - Q.mass) * (Q.z_top50_slots - 0.9995789)
    if Q.mass_top40 < 125.1 and Q.n_dr_0p2_0p4 > 9.0:
        z += 0.0007234638 * (125.1 - Q.mass_top40) * (Q.n_dr_0p2_0p4 - 9.0)
    if Q.sum_pt_top40 < 956.2133 and Q.lam2 < 0.001776308:
        z += -2.297141 * (956.2133 - Q.sum_pt_top40) * (0.001776308 - Q.lam2)
    if Q.sum_pt < 986.0565 and Q.soft5_z > 0.002036949:
        z += 1.432015 * (986.0565 - Q.sum_pt) * (Q.soft5_z - 0.002036949)
    return max(0.0, z)


def neuron_10(Q):
    z = 0.9249052
    if Q.girth < 0.1207452:
        z += 42.36887 * Q.girth - 5.115837
    if Q.mass < 64.48544:
        z += -0.04265328 * Q.mass + 3.30926
    if 64.48544 <= Q.mass < 78.26182:
        z += -0.08701837 * Q.mass + 6.170162
    if 78.26182 <= Q.mass < 86.4:
        z += -0.02161426 * Q.mass + 1.051518
    if 86.4 <= Q.mass < 89.74183:
        z += 0.01052694 * Q.mass - 1.725482
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.05739537 * Q.mass - 5.931541
    if 101.0497 <= Q.mass < 143.7876:
        z += 0.02103902 * Q.mass - 2.257742
    if 143.7876 <= Q.mass < 162.8363:
        z += -0.0766274 * Q.mass + 11.78548
    if Q.mass >= 162.8363:
        z += -0.1829964 * Q.mass + 29.10623
    if Q.girth2_top30 < 0.005402331:
        z += 91.31437 * Q.girth2_top30 - 0.4933104
    if 136.785 <= Q.mass_top50 < 160.8:
        z += 0.06671787 * Q.mass_top50 - 9.126004
    if 160.8 <= Q.mass_top50 < 172.8:
        z += 0.1009939 * Q.mass_top50 - 14.63759
    if Q.mass_top50 >= 172.8:
        z += 0.08241337 * Q.mass_top50 - 11.42687
    if Q.sj2_mass1 < 65.20727:
        z += 0.01306472 * Q.sj2_mass1 - 0.8519144
    if Q.D2 < 2.975532:
        z += -0.09710442 * Q.D2 + 0.2889373
    if 0.9980008 <= Q.psi_0p3 < 0.9985421:
        z += -226.9137 * Q.psi_0p3 + 226.4601
    if Q.psi_0p3 >= 0.9985421:
        z += 4.040771 * Q.psi_0p3 - 4.15771
    if Q.z_top2_slots < 0.5760704:
        z += -0.479939 * Q.z_top2_slots + 0.2764787
    if Q.mass_top5 < 22.18342:
        z += 0.003372508 * Q.mass_top5 - 0.2003532
    if 22.18342 <= Q.mass_top5 < 59.40777:
        z += 0.01618548 * Q.mass_top5 - 0.4845886
    if Q.mass_top5 >= 59.40777:
        z += 0.01281297 * Q.mass_top5 - 0.2842355
    if Q.sum_pt_top50 < 959.0957:
        z += 0.005313099 * Q.sum_pt_top50 - 5.09577
    if Q.girth2 < 0.002575211:
        z += 77.46606 * Q.girth2 - 0.1994914
    if Q.dr_0 < 0.06413297:
        z += -4.299663 * Q.dr_0 + 0.2757502
    if Q.z_dr_0_0p05 >= 0.7674734:
        z += -2.861373 * Q.z_dr_0_0p05 + 2.196028
    if Q.girth2_top10 < 0.01976735:
        z += 21.22635 * Q.girth2_top10 - 0.4195889
    if Q.e2 < 0.01256572:
        z += -5.199337 * Q.e2 - 0.9252058
    if 0.01256572 <= Q.e2 < 0.01541561:
        z += 35.9356 * Q.e2 - 1.442096
    if 0.01541561 <= Q.e2 < 0.03263075:
        z += 76.82309 * Q.e2 - 2.072402
    if 0.03263075 <= Q.e2 < 0.04358622:
        z += 70.38432 * Q.e2 - 1.8623
    if Q.e2 >= 0.04358622:
        z += 34.69618 * Q.e2 - 0.3067886
    if Q.sum_pt_top10 >= 943.6922:
        z += -0.003230028 * Q.sum_pt_top10 + 3.048152
    if Q.n_particles >= 41.0:
        z += 0.01056824 * Q.n_particles - 0.433298
    if Q.mass_over_sum_pt >= 0.1606361:
        z += -65.72315 * Q.mass_over_sum_pt + 10.55751
    if Q.e3 < 0.0001086251:
        z += -4393.906 * Q.e3 + 0.4772884
    if Q.zdr_1 < 0.008824206:
        z += -33.34055 * Q.zdr_1 + 0.2942039
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.04690091 * Q.n_dr_0p2_0p4 - 0.51591
    if Q.sum_pt_top15 < 935.1043:
        z += -0.002048098 * Q.sum_pt_top15 + 1.915185
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -4.057953 * Q.z_dr_0p2_0p4 + 0.3701896
    if Q.mass_top15 < 72.18744:
        z += 0.008687707 * Q.mass_top15 - 0.6271433
    if Q.girth2_top20 < 0.005312783:
        z += 9.927757 * Q.girth2_top20 + 0.9368922
    if 0.005312783 <= Q.girth2_top20 < 0.01655983:
        z += -87.99074 * Q.girth2_top20 + 1.457112
    if Q.sj3_dr_min >= 0.1204829:
        z += 1.293141 * Q.sj3_dr_min - 0.1558014
    if 78.53034 <= Q.mass_top30 < 89.17293:
        z += -0.02239699 * Q.mass_top30 + 1.758843
    if Q.mass_top30 >= 89.17293:
        z += -0.007580986 * Q.mass_top30 + 0.4376567
    if Q.tau1 < 0.04466492:
        z += 9.624397 * Q.tau1 + 2.375635
    if 0.04466492 <= Q.tau1 < 0.1507173:
        z += -26.45399 * Q.tau1 + 3.987073
    if Q.pt_dispersion >= 0.27462:
        z += -2.024482 * Q.pt_dispersion + 0.5559634
    if Q.zdr_2 < 0.007043525:
        z += -27.00499 * Q.zdr_2 + 0.1902103
    if Q.girth2_top3 < 0.006756161:
        z += 19.12875 * Q.girth2_top3 - 0.1292369
    if Q.zdr_0 < 0.02078982:
        z += -8.550984 * Q.zdr_0 + 0.1777734
    if Q.tau2 < 0.04828819:
        z += -12.50191 * Q.tau2 + 0.6036948
    if Q.sum_pt < 986.0565:
        z += 0.002137751 * Q.sum_pt - 2.107943
    if Q.girth2_top40 >= 0.02497133:
        z += 76.30509 * Q.girth2_top40 - 1.90544
    if Q.mass_top10 >= 71.781:
        z += -0.007817912 * Q.mass_top10 + 0.5611775
    if Q.max_dr < 0.2404747:
        z += -2.497747 * Q.max_dr + 0.6006451
    if Q.girth2_top30 < 0.005402331 and Q.sum_pt_top5 < 631.275:
        z += 0.2615858 * (0.005402331 - Q.girth2_top30) * (631.275 - Q.sum_pt_top5)
    if Q.mass_top50 > 160.8 and Q.soft4_z > 0.001721109:
        z += -27.10064 * (Q.mass_top50 - 160.8) * (Q.soft4_z - 0.001721109)
    if Q.D2 < 2.975532 and Q.sj2_dr > 0.2070855:
        z += -1.120068 * (2.975532 - Q.D2) * (Q.sj2_dr - 0.2070855)
    if Q.girth2_top30 < 0.005402331 and Q.psi_0p3 > 0.9985421:
        z += 78836.46 * (0.005402331 - Q.girth2_top30) * (Q.psi_0p3 - 0.9985421)
    if Q.girth2_top10 < 0.01976735 and Q.psi_0p3 > 0.9985421:
        z += -8195.183 * (0.01976735 - Q.girth2_top10) * (Q.psi_0p3 - 0.9985421)
    if Q.mass < 101.0497 and Q.pt1_dr01 > 5.351077:
        z += 0.000477571 * (101.0497 - Q.mass) * (Q.pt1_dr01 - 5.351077)
    if Q.sum_pt_top10 > 943.6922 and Q.z_dr_0p05_0p1 < 0.6469679:
        z += 0.0020184 * (Q.sum_pt_top10 - 943.6922) * (0.6469679 - Q.z_dr_0p05_0p1)
    if Q.girth2_top10 < 0.01976735 and Q.max_dr < 0.4357228:
        z += -143.4779 * (0.01976735 - Q.girth2_top10) * (0.4357228 - Q.max_dr)
    if Q.mass_top5 < 59.40777 and Q.z_dr_0p05_0p1 < 0.8509215:
        z += 0.007560101 * (59.40777 - Q.mass_top5) * (0.8509215 - Q.z_dr_0p05_0p1)
    if Q.mass_over_sum_pt > 0.1606361 and Q.z_3 < 0.09696199:
        z += 317.1513 * (Q.mass_over_sum_pt - 0.1606361) * (0.09696199 - Q.z_3)
    if Q.mass_top50 > 136.785 and Q.soft5_z > 0.001594761:
        z += -34.238 * (Q.mass_top50 - 136.785) * (Q.soft5_z - 0.001594761)
    if Q.mass > 162.8363 and Q.soft5_z > 0.001434897:
        z += 48.23783 * (Q.mass - 162.8363) * (Q.soft5_z - 0.001434897)
    if Q.D2 < 2.975532 and Q.absphi_4 < 0.08728027:
        z += 0.855471 * (2.975532 - Q.D2) * (0.08728027 - Q.absphi_4)
    if Q.sum_pt_top10 > 943.6922 and Q.eta_0 < -0.07794189:
        z += -0.2547449 * (Q.sum_pt_top10 - 943.6922) * (-0.07794189 - Q.eta_0)
    if Q.mass > 78.26182 and Q.psi_0p3 > 0.9853273:
        z += 0.2192135 * (Q.mass - 78.26182) * (Q.psi_0p3 - 0.9853273)
    if Q.sj2_mass1 < 65.20727 and Q.sj2_mass2 > 7.769597:
        z += 0.0002017543 * (65.20727 - Q.sj2_mass1) * (Q.sj2_mass2 - 7.769597)
    if Q.pt_dispersion > 0.27462 and Q.n_real_top30 < 30.0:
        z += 0.09506699 * (Q.pt_dispersion - 0.27462) * (30.0 - Q.n_real_top30)
    if Q.sum_pt_top50 < 959.0957 and Q.dr_9 < 0.2384186:
        z += 0.009775961 * (959.0957 - Q.sum_pt_top50) * (0.2384186 - Q.dr_9)
    if Q.sum_pt_top50 < 959.0957 and Q.dr1_5 < 0.2277069:
        z += 0.00493378 * (959.0957 - Q.sum_pt_top50) * (0.2277069 - Q.dr1_5)
    if Q.sum_pt < 986.0565 and Q.dr_3 > 0.01436663:
        z += -0.02011732 * (986.0565 - Q.sum_pt) * (Q.dr_3 - 0.01436663)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.438903
    if 0.9313699 <= Q.psi_0p2 < 0.9935324:
        z += 3.883739 * Q.psi_0p2 - 3.617198
    if Q.psi_0p2 >= 0.9935324:
        z += 31.72353 * Q.psi_0p2 - 31.27693
    if Q.girth2_top10 < 0.00130722:
        z += -198.1393 * Q.girth2_top10 + 0.02146142
    if 0.00130722 <= Q.girth2_top10 < 0.00406126:
        z += -22.84155 * Q.girth2_top10 - 0.2076914
    if 0.00406126 <= Q.girth2_top10 < 0.00625621:
        z += 136.8855 * Q.girth2_top10 - 0.8563844
    if Q.mass < 64.48544:
        z += -0.03513559 * Q.mass + 3.966701
    if 64.48544 <= Q.mass < 80.4:
        z += -0.0492944 * Q.mass + 4.879738
    if 80.4 <= Q.mass < 92.85979:
        z += -0.07355406 * Q.mass + 6.830215
    if Q.e2 < 0.02515919:
        z += -54.69605 * Q.e2 + 1.165882
    if 0.02515919 <= Q.e2 < 0.03029714:
        z += -3.394745 * Q.e2 - 0.1248177
    if 0.03029714 <= Q.e2 < 0.03875945:
        z += 26.90388 * Q.e2 - 1.04278
    if Q.girth2_top30 < 0.005402331:
        z += -14.72102 * Q.girth2_top30 - 0.1160606
    if 0.005402331 <= Q.girth2_top30 < 0.006929741:
        z += 128.0523 * Q.girth2_top30 - 0.8873695
    if 2.178951 <= Q.D2 < 3.814159:
        z += 0.09761345 * Q.D2 - 0.2126949
    if 3.814159 <= Q.D2 < 5.378975:
        z += 0.003787801 * Q.D2 + 0.1451711
    if Q.D2 >= 5.378975:
        z += -0.01742071 * Q.D2 + 0.2592511
    if Q.mass_top30 < 60.43821:
        z += -0.00431648 * Q.mass_top30 - 0.1050032
    if 60.43821 <= Q.mass_top30 < 86.4:
        z += 0.01409315 * Q.mass_top30 - 1.217648
    if Q.mass_top40 < 67.72643:
        z += 0.004301504 * Q.mass_top40 - 0.5393099
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.01589735 * Q.mass_top40 - 1.324655
    if Q.girth < 0.07374472:
        z += 0.2172065 * Q.girth + 0.2397118
    if 0.07374472 <= Q.girth < 0.076787:
        z += 3.967327 * Q.girth - 0.03683977
    if 0.076787 <= Q.girth < 0.08589404:
        z += -29.40575 * Q.girth + 2.525778
    if Q.e2_sq < 0.00818374:
        z += -150.9723 * Q.e2_sq + 1.235518
    if Q.mass_over_sum_pt < 0.06030419:
        z += 25.34056 * Q.mass_over_sum_pt - 1.823094
    if 0.06030419 <= Q.mass_over_sum_pt < 0.07999061:
        z += 14.9825 * Q.mass_over_sum_pt - 1.198459
    if Q.lam1 < 0.006716737:
        z += 108.6604 * Q.lam1 - 0.7298436
    if Q.e3 < 3.793233e-05:
        z += 6198.555 * Q.e3 - 0.2351257
    if Q.sj2_mass1 < 19.89956:
        z += 0.005466916 * Q.sj2_mass1 - 0.1087892
    if Q.mass_top50 < 80.35535:
        z += 0.02877086 * Q.mass_top50 - 1.822126
    if 80.35535 <= Q.mass_top50 < 97.93004:
        z += -0.02786773 * Q.mass_top50 + 2.729088
    if Q.psi_0p3 >= 0.9980008:
        z += 50.03943 * Q.psi_0p3 - 49.93939
    if Q.zdr_0 < 0.005911134:
        z += 28.17769 * Q.zdr_0 - 0.1665621
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.04473868 * Q.n_dr_0p2_0p4 + 0.2684321
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.02192589 * Q.n_dr_0p1_0p2 - 0.0823461
    if 8.0 <= Q.n_dr_0p1_0p2 < 15.0:
        z += -0.01329444 * Q.n_dr_0p1_0p2 + 0.1994165
    z += -0.01313714 * Q.n_real_top50 + 0.656857
    if Q.z_top5_slots >= 0.534626:
        z += -0.812729 * Q.z_top5_slots + 0.434506
    if Q.girth2_top20 < 0.008031209:
        z += 27.45788 * Q.girth2_top20 - 0.22052
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2 < 0.005532208:
        z += -29.61731 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.006619297 * (10.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.mass_over_sum_pt_sq < 0.009595015 and Q.z_dr_0p1_0p2 > 0.1203437:
        z += 598.9899 * (0.009595015 - Q.mass_over_sum_pt_sq) * (Q.z_dr_0p1_0p2 - 0.1203437)
    if Q.mass < 101.0497 and Q.sum_pt_top10 < 891.875:
        z += -2.174857e-05 * (101.0497 - Q.mass) * (891.875 - Q.sum_pt_top10)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_real_top40 > 26.0:
        z += -0.002020238 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_real_top40 - 26.0)
    if Q.mass < 101.0497 and Q.pt1_dr01 > 12.6865:
        z += 0.0008211709 * (101.0497 - Q.mass) * (Q.pt1_dr01 - 12.6865)
    if Q.z_top40_slots > 0.996191 and Q.C2_b2 < 0.006580753:
        z += 6740.534 * (Q.z_top40_slots - 0.996191) * (0.006580753 - Q.C2_b2)
    if Q.mass_top50 < 80.35535 and Q.zdr_4 < 0.003932029:
        z += 5.63947 * (80.35535 - Q.mass_top50) * (0.003932029 - Q.zdr_4)
    if Q.mass < 92.85979 and Q.zdr_4 < 0.004495205:
        z += -3.270321 * (92.85979 - Q.mass) * (0.004495205 - Q.zdr_4)
    if Q.mass < 80.4 and Q.z_9 > 0.01833434:
        z += -0.6795102 * (80.4 - Q.mass) * (Q.z_9 - 0.01833434)
    if Q.mass < 101.0497 and Q.z_9 > 0.01620892:
        z += 0.3457607 * (101.0497 - Q.mass) * (Q.z_9 - 0.01620892)
    if Q.soft10_pt < 1.916992 and Q.sj3_mass1 > 5.112677:
        z += -0.006136917 * (1.916992 - Q.soft10_pt) * (Q.sj3_mass1 - 5.112677)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.009581723
    if Q.mass < 53.87362:
        z += -0.04878608 * Q.mass + 4.896578
    if 53.87362 <= Q.mass < 62.55:
        z += -0.05519557 * Q.mass + 5.241881
    if 62.55 <= Q.mass < 74.25181:
        z += -0.1026729 * Q.mass + 8.211588
    if 74.25181 <= Q.mass < 82.85409:
        z += -0.07137657 * Q.mass + 5.887778
    if 82.85409 <= Q.mass < 86.4:
        z += 0.007349894 * Q.mass - 0.6350308
    if Q.sd_mass >= 125.1:
        z += 0.02074482 * Q.sd_mass - 2.595176
    if Q.sj3_pair_mass_max >= 120.6:
        z += -0.01171918 * Q.sj3_pair_mass_max + 1.413333
    if Q.girth2_top10 < 0.0007431905:
        z += -251.4245 * Q.girth2_top10 + 0.1868563
    if Q.mass_top40 < 74.78616:
        z += 0.0188277 * Q.mass_top40 - 1.408051
    if Q.log_sum_pt < 6.856375:
        z += 1.324313 * Q.log_sum_pt - 9.079988
    if Q.girth2_top15 < 0.006142802:
        z += 74.59572 * Q.girth2_top15 - 0.4582267
    if Q.mass_over_sum_pt < 0.06895248:
        z += 24.55829 * Q.mass_over_sum_pt - 1.693355
    if Q.mass < 86.4 and Q.z_dr_0p1_0p2 < 0.06472584:
        z += -0.1950479 * (86.4 - Q.mass) * (0.06472584 - Q.z_dr_0p1_0p2)
    if Q.sd_mass > 125.1 and Q.sd_zg < 0.4199841:
        z += -0.06415118 * (Q.sd_mass - 125.1) * (0.4199841 - Q.sd_zg)
    if Q.mass_top40 < 89.6788 and Q.n_dr_0p05_0p1 > 1.0:
        z += 0.000293637 * (89.6788 - Q.mass_top40) * (Q.n_dr_0p05_0p1 - 1.0)
    if Q.mass_top40 < 89.6788 and Q.pt_0 < 334.5:
        z += -3.027758e-05 * (89.6788 - Q.mass_top40) * (334.5 - Q.pt_0)
    if Q.mass < 86.4 and Q.psi_0p3 < 0.9985421:
        z += -1.51924 * (86.4 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.sj3_pair_mass_max > 120.6 and Q.sj3_pair_mass_min < 76.60223:
        z += 0.0003133436 * (Q.sj3_pair_mass_max - 120.6) * (76.60223 - Q.sj3_pair_mass_min)
    if Q.sj3_pair_mass_max > 120.6 and Q.z_dr_0p1_0p2 > 0.2864926:
        z += 0.02058296 * (Q.sj3_pair_mass_max - 120.6) * (Q.z_dr_0p1_0p2 - 0.2864926)
    if Q.mass < 86.4 and Q.lam2 < 0.003687605:
        z += 8.554694 * (86.4 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.girth2_top10 < 0.0007431905 and Q.sum_pt_top40 < 1069.671:
        z += 2.653534 * (0.0007431905 - Q.girth2_top10) * (1069.671 - Q.sum_pt_top40)
    if Q.girth2_top10 < 0.0007431905 and Q.n_real_top40 < 36.0:
        z += -21.08249 * (0.0007431905 - Q.girth2_top10) * (36.0 - Q.n_real_top40)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.177254
    if Q.sum_pt < 1085.125:
        z += 0.007404295 * Q.sum_pt - 8.152789
    if 1085.125 <= Q.sum_pt < 1115.723:
        z += 0.003863144 * Q.sum_pt - 4.310197
    if 74.25181 <= Q.mass < 101.0497:
        z += 0.0249059 * Q.mass - 1.849308
    if 101.0497 <= Q.mass < 136.785:
        z += -0.006033707 * Q.mass + 1.27713
    if 136.785 <= Q.mass < 143.7876:
        z += -0.03424952 * Q.mass + 5.13663
    if 143.7876 <= Q.mass < 160.8:
        z += -0.1184806 * Q.mass + 17.24802
    if 160.8 <= Q.mass < 162.8363:
        z += -0.3561032 * Q.mass + 55.45773
    if 162.8363 <= Q.mass < 172.8:
        z += -0.2474285 * Q.mass + 37.76154
    if Q.mass >= 172.8:
        z += -0.1725323 * Q.mass + 24.81948
    if Q.n_for_90pct < 16.0:
        z += 0.03123426 * Q.n_for_90pct - 0.4997482
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 365.624 * Q.mass_over_sum_pt - 62.47787
    if Q.log_sum_pt < 6.811175:
        z += 38.88156 * Q.log_sum_pt - 266.7536
    if 6.811175 <= Q.log_sum_pt < 6.910131:
        z += 19.44768 * Q.log_sum_pt - 134.386
    if Q.log_sum_pt >= 7.139296:
        z += -3.300328 * Q.log_sum_pt + 23.56202
    if Q.sum_pt_top40 < 1007.44:
        z += -0.009062559 * Q.sum_pt_top40 + 9.341241
    if 1007.44 <= Q.sum_pt_top40 < 1053.047:
        z += -0.004632057 * Q.sum_pt_top40 + 4.877775
    if 27.46086 <= Q.mass_top50 < 136.785:
        z += 0.007261759 * Q.mass_top50 - 0.1994141
    if 136.785 <= Q.mass_top50 < 168.9698:
        z += 0.06038224 * Q.mass_top50 - 7.465499
    if Q.mass_top50 >= 168.9698:
        z += 0.07818224 * Q.mass_top50 - 10.47316
    if Q.girth2_top50 < 0.007820315:
        z += -82.14511 * Q.girth2_top50 + 1.005672
    if 0.007820315 <= Q.girth2_top50 < 0.02550569:
        z += -20.54077 * Q.girth2_top50 + 0.5239066
    if Q.pt_entropy < 2.752054:
        z += 0.2496401 * Q.pt_entropy - 0.687023
    if Q.tau1 < 0.08286256:
        z += -9.518492 * Q.tau1 + 0.7887266
    if Q.tau1 >= 0.1751567:
        z += -8.496208 * Q.tau1 + 1.488168
    if Q.mass_over_sum_pt_sq < 0.02580396:
        z += 96.55163 * Q.mass_over_sum_pt_sq - 2.491414
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -868.8082 * Q.mass_over_sum_pt_sq + 25.36921
    z += 0.01325481 * Q.n_particles
    if Q.sum_pt_top50 < 997.0189:
        z += -0.007344356 * Q.sum_pt_top50 + 7.322462
    if Q.sum_pt_top30 < 966.0633:
        z += 0.004356963 * Q.sum_pt_top30 - 4.209102
    if Q.sum_pt_top30 >= 1111.245:
        z += 0.002184687 * Q.sum_pt_top30 - 2.427723
    if Q.mass_top5 >= 33.71058:
        z += 0.006292685 * Q.mass_top5 - 0.21213
    if Q.pt_11 < 18.32812:
        z += 0.01633071 * Q.pt_11 - 0.2993113
    if Q.z_dr_0p1_0p2 < 0.08871546:
        z += 2.100781 * Q.z_dr_0p1_0p2 - 0.1863717
    if Q.mass_top10 < 76.9886:
        z += 0.001243895 * Q.mass_top10 - 0.0957657
    if Q.sum_pt_top20 >= 956.5062:
        z += -0.001109768 * Q.sum_pt_top20 + 1.0615
    if 34.0 <= Q.n_pt_above_1 < 58.0:
        z += -0.008118726 * Q.n_pt_above_1 + 0.2760367
    if Q.n_pt_above_1 >= 58.0:
        z += 0.01719646 * Q.n_pt_above_1 - 1.192244
    if Q.psi_0p1 >= 0.9371031:
        z += 2.375097 * Q.psi_0p1 - 2.225711
    if Q.mass_top30 >= 138.3818:
        z += 0.01553386 * Q.mass_top30 - 2.149604
    if Q.mass_top20 >= 91.19:
        z += -0.007889173 * Q.mass_top20 + 0.7194137
    if Q.lam1 >= 0.01174405:
        z += 11.44084 * Q.lam1 - 0.1343618
    if Q.sum_pt < 1085.125 and Q.sum_pt_top40 < 858.8262:
        z += 3.549665e-05 * (1085.125 - Q.sum_pt) * (858.8262 - Q.sum_pt_top40)
    if Q.sum_pt < 1085.125 and Q.tau21_b2 < 0.8310045:
        z += -0.002316056 * (1085.125 - Q.sum_pt) * (0.8310045 - Q.tau21_b2)
    if Q.n_for_90pct < 16.0 and Q.D2 < 3.814159:
        z += 0.01466995 * (16.0 - Q.n_for_90pct) * (3.814159 - Q.D2)
    if Q.mass > 136.785 and Q.sum_pt < 1028.184:
        z += 0.0004310917 * (Q.mass - 136.785) * (1028.184 - Q.sum_pt)
    if Q.mass > 74.25181 and Q.sum_pt < 1028.184:
        z += -0.0002583906 * (Q.mass - 74.25181) * (1028.184 - Q.sum_pt)
    if Q.sum_pt < 1007.788 and Q.sum_pt_top3 > 512.4375:
        z += 2.274156e-05 * (1007.788 - Q.sum_pt) * (Q.sum_pt_top3 - 512.4375)
    if Q.log_sum_pt > 7.139296 and Q.C3 < 0.00317537:
        z += -1105.551 * (Q.log_sum_pt - 7.139296) * (0.00317537 - Q.C3)
    if Q.log_sum_pt < 6.811175 and Q.dr_13 < 0.1312677:
        z += 46.72717 * (6.811175 - Q.log_sum_pt) * (0.1312677 - Q.dr_13)
    if Q.mass > 172.8 and Q.pt_6 < 56.53125:
        z += 0.001094266 * (Q.mass - 172.8) * (56.53125 - Q.pt_6)
    if Q.log_sum_pt < 6.811175 and Q.zdr_11 < 0.005041702:
        z += -705.1431 * (6.811175 - Q.log_sum_pt) * (0.005041702 - Q.zdr_11)
    if Q.mass > 162.8363 and Q.soft6_z > 0.0005748372:
        z += 32.27709 * (Q.mass - 162.8363) * (Q.soft6_z - 0.0005748372)
    if Q.mass_top50 > 136.785 and Q.soft6_z > 0.0008188601:
        z += -19.40429 * (Q.mass_top50 - 136.785) * (Q.soft6_z - 0.0008188601)
    if Q.mass > 172.8 and Q.soft5_z > 0.0004140594:
        z += -9.879546 * (Q.mass - 172.8) * (Q.soft5_z - 0.0004140594)
    if Q.mass_top50 > 92.16545 and Q.soft7_z < 0.001923089:
        z += -3.720455 * (Q.mass_top50 - 92.16545) * (0.001923089 - Q.soft7_z)
    if Q.mass > 172.8 and Q.soft6_z > 0.0004464147:
        z += 7.458095 * (Q.mass - 172.8) * (Q.soft6_z - 0.0004464147)
    if Q.sum_pt < 1085.125 and Q.dr_4 < 0.1289397:
        z += 0.008449466 * (1085.125 - Q.sum_pt) * (0.1289397 - Q.dr_4)
    if Q.girth2_top50 < 0.02550569 and Q.psi_0p3 > 0.9638082:
        z += 658.705 * (0.02550569 - Q.girth2_top50) * (Q.psi_0p3 - 0.9638082)
    if Q.sum_pt < 1034.834 and Q.D2 < 2.975532:
        z += -0.002142016 * (1034.834 - Q.sum_pt) * (2.975532 - Q.D2)
    if Q.mass > 160.8 and Q.D2 > 0.8942376:
        z += 0.0345473 * (Q.mass - 160.8) * (Q.D2 - 0.8942376)
    if Q.mass > 136.785 and Q.D2 > 0.8942376:
        z += -0.01596312 * (Q.mass - 136.785) * (Q.D2 - 0.8942376)
    if Q.mass > 172.8 and Q.D2 > 0.8942376:
        z += -0.02088977 * (Q.mass - 172.8) * (Q.D2 - 0.8942376)
    if Q.n_for_90pct < 16.0 and Q.D2 < 9.676985:
        z += 0.003570209 * (16.0 - Q.n_for_90pct) * (9.676985 - Q.D2)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.5213972
    if Q.tau21_b2 < 0.342495:
        z += -4.590833 * Q.tau21_b2 + 1.572337
    if Q.mass < 74.25181:
        z += 0.1446619 * Q.mass - 11.53237
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.1403843 * Q.mass - 11.21476
    if 78.26182 <= Q.mass < 80.4:
        z += 0.0983989 * Q.mass - 7.928902
    if 80.4 <= Q.mass < 82.85409:
        z += 0.1624589 * Q.mass - 13.07933
    if 82.85409 <= Q.mass < 89.74183:
        z += 0.0790119 * Q.mass - 6.165401
    if 89.74183 <= Q.mass < 91.19:
        z += 0.03557922 * Q.mass - 2.267672
    if 91.19 <= Q.mass < 125.1:
        z += -0.02880556 * Q.mass + 3.603575
    if Q.psi_0p3 >= 0.9924477:
        z += 86.91881 * Q.psi_0p3 - 86.26237
    if Q.N2 < 0.4226723:
        z += 0.7072569 * Q.N2 - 0.2989379
    if Q.n_dr_0p2_0p4 < 18.0:
        z += 0.01639013 * Q.n_dr_0p2_0p4 - 0.2950223
    if Q.e2 < 0.03029714:
        z += -43.51496 * Q.e2 + 1.639046
    if 0.03029714 <= Q.e2 < 0.04358622:
        z += -24.1301 * Q.e2 + 1.05174
    if Q.mass_over_sum_pt < 0.07852883:
        z += -72.56539 * Q.mass_over_sum_pt + 9.800384
    if 0.07852883 <= Q.mass_over_sum_pt < 0.08873143:
        z += -66.25715 * Q.mass_over_sum_pt + 9.305006
    if 0.08873143 <= Q.mass_over_sum_pt < 0.09046749:
        z += -53.06797 * Q.mass_over_sum_pt + 8.134711
    if 0.09046749 <= Q.mass_over_sum_pt < 0.09795415:
        z += -116.2991 * Q.mass_over_sum_pt + 13.85508
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -76.28764 * Q.mass_over_sum_pt + 9.935784
    if 0.1182259 <= Q.mass_over_sum_pt < 0.140939:
        z += -40.35587 * Q.mass_over_sum_pt + 5.687717
    if Q.girth2_top20 < 0.006374178:
        z += -9.333858 * Q.girth2_top20 - 0.1483845
    if 0.006374178 <= Q.girth2_top20 < 0.01083435:
        z += 46.60808 * Q.girth2_top20 - 0.5049684
    if Q.girth < 0.076787:
        z += 15.01528 * Q.girth - 1.152978
    if Q.girth2_top5 < 0.007164202:
        z += -68.82537 * Q.girth2_top5 + 0.4930789
    if Q.sd_mass < 69.65633:
        z += -0.008122631 * Q.sd_mass + 0.5657926
    if Q.sd_rg < 0.1778185:
        z += 0.4673007 * Q.sd_rg - 0.6261638
    if 0.1778185 <= Q.sd_rg < 0.3017146:
        z += 4.383263 * Q.sd_rg - 1.322494
    if Q.girth2_top30 < 0.002582316:
        z += 346.6105 * Q.girth2_top30 - 2.471413
    if 0.002582316 <= Q.girth2_top30 < 0.01215787:
        z += 164.6228 * Q.girth2_top30 - 2.001463
    if Q.n_pt_above_1 >= 58.0:
        z += 0.03889683 * Q.n_pt_above_1 - 2.256016
    if Q.tau1 < 0.06310829:
        z += 22.35548 * Q.tau1 - 2.157015
    if 0.06310829 <= Q.tau1 < 0.1072713:
        z += 16.89647 * Q.tau1 - 1.812506
    if Q.mass_top30 < 82.66587:
        z += -0.01650774 * Q.mass_top30 + 1.364627
    if Q.log_sum_pt < 6.941997:
        z += -3.965124 * Q.log_sum_pt + 27.38427
    if 6.941997 <= Q.log_sum_pt < 6.98945:
        z += 2.984253 * Q.log_sum_pt - 20.85829
    if Q.sum_pt_top50 < 976.277:
        z += 0.00891909 * Q.sum_pt_top50 - 8.707502
    if Q.sum_pt_top40 < 1024.942:
        z += -0.003228556 * Q.sum_pt_top40 + 3.309084
    if Q.lam1 < 0.007259287:
        z += 18.1225 * Q.lam1 - 0.312692
    if 0.007259287 <= Q.lam1 < 0.007671243:
        z += -10.81027 * Q.lam1 - 0.1026607
    if 0.007671243 <= Q.lam1 < 0.008241985:
        z += 325.171 * Q.lam1 - 2.680055
    if Q.e3 < 0.0001086251:
        z += 2333.429 * Q.e3 - 0.2534689
    if Q.tau2 < 0.0795038:
        z += 6.162935 * Q.tau2 - 0.4899767
    if Q.sj2_mass1 < 37.45803:
        z += -0.006204966 * Q.sj2_mass1 + 0.2324258
    if Q.z_top30_slots < 0.9734886:
        z += 3.121898 * Q.z_top30_slots - 3.039132
    if Q.mass_top50 < 71.79516:
        z += -0.009785042 * Q.mass_top50 - 0.04642499
    if 71.79516 <= Q.mass_top50 < 77.37641:
        z += -0.03054675 * Q.mass_top50 + 1.444165
    if 77.37641 <= Q.mass_top50 < 85.8667:
        z += 0.01486301 * Q.mass_top50 - 2.069479
    if 85.8667 <= Q.mass_top50 < 97.93004:
        z += 0.0462736 * Q.mass_top50 - 4.766603
    if 97.93004 <= Q.mass_top50 < 117.0487:
        z += 0.01229304 * Q.mass_top50 - 1.438885
    if Q.girth2_top40 < 0.01897915:
        z += 46.40902 * Q.girth2_top40 - 0.8808037
    if Q.soft7_z < 0.001595561:
        z += 337.4528 * Q.soft7_z - 0.5384265
    if Q.soft7_pt < 1.650391:
        z += -0.250595 * Q.soft7_pt + 0.4135796
    if Q.soft9_pt < 3.080078:
        z += 0.1347698 * Q.soft9_pt - 0.4151014
    if Q.mass_top40 < 89.6788:
        z += -0.01437121 * Q.mass_top40 + 1.288793
    if Q.tau21_b2 < 0.342495 and Q.girth2_top40 < 0.007709916:
        z += -402.7838 * (0.342495 - Q.tau21_b2) * (0.007709916 - Q.girth2_top40)
    if Q.tau21_b2 < 0.342495 and Q.lam1 < 0.02048524:
        z += -213.2067 * (0.342495 - Q.tau21_b2) * (0.02048524 - Q.lam1)
    if Q.tau21_b2 < 0.342495 and Q.girth2_top50 < 0.006118006:
        z += 770.8011 * (0.342495 - Q.tau21_b2) * (0.006118006 - Q.girth2_top50)
    if Q.N2 < 0.4226723 and Q.max_dr > 0.2404747:
        z += -2.461158 * (0.4226723 - Q.N2) * (Q.max_dr - 0.2404747)
    if Q.psi_0p3 > 0.9924477 and Q.sd_rg > 0.2280025:
        z += -981.0286 * (Q.psi_0p3 - 0.9924477) * (Q.sd_rg - 0.2280025)
    if Q.psi_0p3 > 0.9924477 and Q.sd_rg < 0.1881908:
        z += -322.0522 * (Q.psi_0p3 - 0.9924477) * (0.1881908 - Q.sd_rg)
    if Q.tau21_b2 < 0.342495 and Q.n_real_top40 < 32.0:
        z += -0.1121232 * (0.342495 - Q.tau21_b2) * (32.0 - Q.n_real_top40)
    if Q.tau21_b2 < 0.342495 and Q.n_for_50pct > 3.0:
        z += -0.2032216 * (0.342495 - Q.tau21_b2) * (Q.n_for_50pct - 3.0)
    if Q.mass_over_sum_pt < 0.09046749 and Q.psi_0p2 > 0.9087063:
        z += -436.8595 * (0.09046749 - Q.mass_over_sum_pt) * (Q.psi_0p2 - 0.9087063)
    if Q.girth2_top5 < 0.007164202 and Q.n_pt_above_10 > 13.0:
        z += -3.310304 * (0.007164202 - Q.girth2_top5) * (Q.n_pt_above_10 - 13.0)
    if Q.girth < 0.076787 and Q.z_dr_0p2_0p4 < 0.05180474:
        z += 224.5042 * (0.076787 - Q.girth) * (0.05180474 - Q.z_dr_0p2_0p4)
    if Q.mass < 91.19 and Q.n_dr_0p05_0p1 > 14.0:
        z += -0.001111273 * (91.19 - Q.mass) * (Q.n_dr_0p05_0p1 - 14.0)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.M3 < 0.03787151:
        z += 0.3200042 * (18.0 - Q.n_dr_0p2_0p4) * (0.03787151 - Q.M3)
    if Q.psi_0p2 > 0.9804031 and Q.pt2_over_pt0 > 0.1852611:
        z += 17.31748 * (Q.psi_0p2 - 0.9804031) * (Q.pt2_over_pt0 - 0.1852611)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.soft9_pt < 3.445312:
        z += 0.01044272 * (18.0 - Q.n_dr_0p2_0p4) * (3.445312 - Q.soft9_pt)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.388526
    if Q.z_dr_0_0p05 < 0.8459004:
        z += -0.1580061 * Q.z_dr_0_0p05 + 0.1336574
    if Q.z_dr_0p1_0p2 < 0.06472584:
        z += -5.199818 * Q.z_dr_0p1_0p2 + 0.5323185
    if 0.06472584 <= Q.z_dr_0p1_0p2 < 0.1203437:
        z += -3.51966 * Q.z_dr_0p1_0p2 + 0.4235688
    if Q.girth2_top5 < 0.002270363:
        z += 58.42512 * Q.girth2_top5 + 0.4720279
    if 0.002270363 <= Q.girth2_top5 < 0.008329695:
        z += -99.7922 * Q.girth2_top5 + 0.8312386
    if Q.girth2_top2 < 0.0005124533:
        z += -286.4177 * Q.girth2_top2 - 0.02073672
    if 0.0005124533 <= Q.girth2_top2 < 0.001056655:
        z += 307.813 * Q.girth2_top2 - 0.3252523
    if 0.8976117 <= Q.psi_0p1 < 0.9371031:
        z += -4.133491 * Q.psi_0p1 + 3.71027
    if Q.psi_0p1 >= 0.9371031:
        z += -8.313652 * Q.psi_0p1 + 7.627512
    if Q.log_sum_pt < 6.903423:
        z += -4.957483 * Q.log_sum_pt + 35.06594
    if 6.903423 <= Q.log_sum_pt < 7.017258:
        z += -7.399645 * Q.log_sum_pt + 51.92522
    if Q.psi_0p3 >= 0.9896594:
        z += -32.59095 * Q.psi_0p3 + 32.25394
    if Q.sum_pt < 1002.379:
        z += 0.007981303 * Q.sum_pt - 8.000287
    if 0.1666442 <= Q.sj2_dr < 0.2232169:
        z += -2.816818 * Q.sj2_dr + 0.4694063
    if 0.2232169 <= Q.sj2_dr < 0.2971569:
        z += 2.332266 * Q.sj2_dr - 0.6799563
    if Q.sj2_dr >= 0.2971569:
        z += -0.4121208 * Q.sj2_dr + 0.1355573
    if Q.girth2_top10 < 0.007678544:
        z += -51.71608 * Q.girth2_top10 + 0.3433898
    if 0.007678544 <= Q.girth2_top10 < 0.008956554:
        z += 42.02965 * Q.girth2_top10 - 0.3764408
    if Q.lam1 < 0.004673423:
        z += -100.0009 * Q.lam1 + 0.08520482
    if 0.004673423 <= Q.lam1 < 0.01174405:
        z += 54.04638 * Q.lam1 - 0.6347234
    if Q.lam1 >= 0.01649354:
        z += -28.64703 * Q.lam1 + 0.4724911
    if Q.n_dr_0p1_0p2 < 13.0:
        z += -0.03430781 * Q.n_dr_0p1_0p2 + 0.4460015
    if Q.D2 < 1.976207:
        z += 0.234097 * Q.D2 - 0.4626243
    if Q.lam2 < 0.001776308:
        z += -223.0822 * Q.lam2 + 0.3962628
    if Q.tau1 < 0.0705748:
        z += 16.05369 * Q.tau1 - 1.132986
    if Q.sum_pt_top20 < 1017.778:
        z += 0.001697572 * Q.sum_pt_top20 - 1.727752
    if Q.n_dr_0p2_0p4 >= 9.0:
        z += -0.02114753 * Q.n_dr_0p2_0p4 + 0.1903278
    if Q.sd_mass < 45.595:
        z += 9.711273e-05 * Q.sd_mass + 0.4178717
    if 45.595 <= Q.sd_mass < 79.18312:
        z += -0.01257289 * Q.sd_mass + 0.9955603
    if Q.mass_top50 < 52.15154:
        z += -0.001168553 * Q.mass_top50 - 0.3384634
    if 52.15154 <= Q.mass_top50 < 71.79516:
        z += 0.02033257 * Q.mass_top50 - 1.45978
    if Q.mass_top50 >= 97.93004:
        z += -0.01339276 * Q.mass_top50 + 1.311553
    if Q.sum_pt_top40 < 1018.698:
        z += 0.002922178 * Q.sum_pt_top40 - 2.976816
    if Q.sum_pt_top30 < 933.1875:
        z += -0.001310941 * Q.sum_pt_top30 + 1.223353
    if Q.sd_rg < 0.2042612:
        z += -0.4073768 * Q.sd_rg - 0.4216555
    if 0.2042612 <= Q.sd_rg < 0.3017146:
        z += 5.180598 * Q.sd_rg - 1.563062
    if Q.e2 >= 0.03029714:
        z += -43.53739 * Q.e2 + 1.319059
    if Q.mass_top30 < 80.24626:
        z += -0.005544153 * Q.mass_top30 + 0.4943884
    if 80.24626 <= Q.mass_top30 < 89.17293:
        z += 0.01501098 * Q.mass_top30 - 1.155084
    if Q.mass_top30 >= 89.17293:
        z += 0.02055513 * Q.mass_top30 - 1.649472
    if Q.N2 < 0.4226723:
        z += 1.074128 * Q.N2 - 0.4540043
    if Q.e3 >= 0.0001841806:
        z += 751.3179 * Q.e3 - 0.1383782
    if Q.e2_sq < 0.006399858:
        z += 97.93663 * Q.e2_sq - 0.6267805
    if Q.n_pt_above_5 < 42.0:
        z += -0.008788703 * Q.n_pt_above_5 + 0.3691255
    if Q.max_dr < 0.2982:
        z += 1.579188 * Q.max_dr - 0.4709138
    if Q.girth < 0.08589404:
        z += -17.23809 * Q.girth + 1.480649
    if Q.LHA < 0.2941033:
        z += 6.034204 * Q.LHA - 1.774679
    if Q.sj2_zsoft < 0.2416266:
        z += -0.6136191 * Q.sj2_zsoft + 0.1482667
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.planar_flow < 0.777038:
        z += 2.216261 * (0.1203437 - Q.z_dr_0p1_0p2) * (0.777038 - Q.planar_flow)
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 949.9169:
        z += 0.005470984 * (0.8459004 - Q.z_dr_0_0p05) * (949.9169 - Q.sum_pt)
    if Q.girth2_top5 < 0.008329695 and Q.psi_0p2 < 0.948102:
        z += -327.9656 * (0.008329695 - Q.girth2_top5) * (0.948102 - Q.psi_0p2)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_dr_0p2_0p4 > 5.0:
        z += -0.1214662 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p2_0p4 - 5.0)
    if Q.girth2_top20 < 0.00287991 and Q.sum_pt > 1167.447:
        z += 0.4539286 * (0.00287991 - Q.girth2_top20) * (Q.sum_pt - 1167.447)
    if Q.sj2_dr > 0.2232169 and Q.C2_b2 < 0.04008677:
        z += 87.07018 * (Q.sj2_dr - 0.2232169) * (0.04008677 - Q.C2_b2)
    if Q.girth2_top5 < 0.008329695 and Q.z_6 < 0.04770182:
        z += 600.6721 * (0.008329695 - Q.girth2_top5) * (0.04770182 - Q.z_6)
    if Q.psi_0p3 > 0.9896594 and Q.sj2_mass2 < 14.45919:
        z += -1.06532 * (Q.psi_0p3 - 0.9896594) * (14.45919 - Q.sj2_mass2)
    if Q.girth2_top5 < 0.008329695 and Q.sj3_mass1 > 5.112677:
        z += -1.08473 * (0.008329695 - Q.girth2_top5) * (Q.sj3_mass1 - 5.112677)
    if Q.n_dr_0p1_0p2 < 13.0 and Q.absphi_0 < 0.02970886:
        z += -0.762694 * (13.0 - Q.n_dr_0p1_0p2) * (0.02970886 - Q.absphi_0)
    if Q.girth2_top5 < 0.008329695 and Q.n_real_top40 > 22.0:
        z += -1.708396 * (0.008329695 - Q.girth2_top5) * (Q.n_real_top40 - 22.0)
    if Q.girth2_top20 < 0.00287991 and Q.max_dr < 0.4357228:
        z += 549.3087 * (0.00287991 - Q.girth2_top20) * (0.4357228 - Q.max_dr)
    if Q.girth2_top5 < 0.008329695 and Q.sj3_pairmin_over_m > 0.09540583:
        z += -47.1901 * (0.008329695 - Q.girth2_top5) * (Q.sj3_pairmin_over_m - 0.09540583)
    if Q.D2 < 1.976207 and Q.dr_11 < 0.2535773:
        z += 0.7831748 * (1.976207 - Q.D2) * (0.2535773 - Q.dr_11)
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
