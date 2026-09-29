"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

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
    z = -0.7140578
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.03078671 * Q.n_dr_0p2_0p4 + 0.5541607
    if Q.girth2_top15 < 0.005383629:
        z += -52.82416 * Q.girth2_top15 + 0.3476233
    if 0.005383629 <= Q.girth2_top15 < 0.007887677:
        z += -335.3272 * Q.girth2_top15 + 1.868515
    if 0.007887677 <= Q.girth2_top15 < 0.009962397:
        z += 374.2375 * Q.girth2_top15 - 3.728303
    if Q.tau1 < 0.04466492:
        z += 91.53793 * Q.tau1 - 4.088534
    if 0.9896594 <= Q.psi_0p3 < 0.9956185:
        z += 54.31296 * Q.psi_0p3 - 53.75133
    if Q.psi_0p3 >= 0.9956185:
        z += 20.15228 * Q.psi_0p3 - 19.74033
    if 7.017258 <= Q.log_sum_pt < 7.062574:
        z += 2.073863 * Q.log_sum_pt - 14.55283
    if Q.log_sum_pt >= 7.062574:
        z += 3.573925 * Q.log_sum_pt - 25.14713
    if Q.sum_pt_top40 >= 1001.523:
        z += -0.006063341 * Q.sum_pt_top40 + 6.072576
    if Q.girth < 0.05048381:
        z += -97.66033 * Q.girth + 7.765149
    if 0.05048381 <= Q.girth < 0.08068193:
        z += -128.8199 * Q.girth + 9.3382
    if 0.08068193 <= Q.girth < 0.08589404:
        z += 64.36337 * Q.girth - 6.248195
    if 0.08589404 <= Q.girth < 0.09749958:
        z += 62.01913 * Q.girth - 6.046839
    if Q.sj3_dr_max < 0.1210264:
        z += 0.2481041 * Q.sj3_dr_max - 0.2028366
    if 0.1210264 <= Q.sj3_dr_max < 0.1623049:
        z += 14.51591 * Q.sj3_dr_max - 1.929618
    if 0.1623049 <= Q.sj3_dr_max < 0.2125209:
        z += -8.491 * Q.sj3_dr_max + 1.804515
    if Q.sum_pt_top2 >= 405.0:
        z += 0.0006723691 * Q.sum_pt_top2 - 0.2723095
    if Q.sj2_dr < 0.06289464:
        z += 3.104784 * Q.sj2_dr - 0.5666381
    if 0.06289464 <= Q.sj2_dr < 0.1411617:
        z += 2.130855 * Q.sj2_dr - 0.5053832
    if 0.1411617 <= Q.sj2_dr < 0.1512157:
        z += 18.65097 * Q.sj2_dr - 2.837391
    if 0.1512157 <= Q.sj2_dr < 0.1745007:
        z += -13.08934 * Q.sj2_dr + 1.962242
    if 0.1745007 <= Q.sj2_dr < 0.1825048:
        z += -1.754333 * Q.sj2_dr - 0.01572511
    if Q.sj2_dr >= 0.1825048:
        z += -4.859118 * Q.sj2_dr + 0.550913
    if Q.dr_0 < 0.05775119:
        z += 5.859997 * Q.dr_0 - 0.5057117
    if 0.05775119 <= Q.dr_0 < 0.06413297:
        z += 26.21368 * Q.dr_0 - 1.681161
    if Q.LHA < 0.1870291:
        z += 11.40463 * Q.LHA - 4.031481
    if 0.1870291 <= Q.LHA < 0.3098384:
        z += 21.31199 * Q.LHA - 5.884446
    if 0.3098384 <= Q.LHA < 0.3203321:
        z += -45.49171 * Q.LHA + 14.8139
    if 0.3203321 <= Q.LHA < 0.3332345:
        z += -18.71332 * Q.LHA + 6.235926
    if Q.e3 < 2.883342e-05:
        z += 32841.62 * Q.e3 - 0.6987041
    if 2.883342e-05 <= Q.e3 < 5.13841e-05:
        z += -11007.75 * Q.e3 + 0.5656233
    if Q.e2 < 0.01879315:
        z += -22.42567 * Q.e2 + 0.2538016
    if 0.01879315 <= Q.e2 < 0.03029714:
        z += 14.57297 * Q.e2 - 0.4415192
    if Q.psi_0p1 >= 0.8747961:
        z += 2.514698 * Q.psi_0p1 - 2.199848
    if Q.lam2 < 0.0008296372:
        z += -550.1727 * Q.lam2 + 0.4564437
    if Q.sum_pt_top30 >= 1052.08:
        z += -0.0007008584 * Q.sum_pt_top30 + 0.7373591
    if Q.n_dr_0p2_0p4 < 18.0 and Q.girth2_top10 > 0.004752876:
        z += -8.85719 * (18.0 - Q.n_dr_0p2_0p4) * (Q.girth2_top10 - 0.004752876)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.tau21_b2 > 0.2018786:
        z += -0.05634512 * (18.0 - Q.n_dr_0p2_0p4) * (Q.tau21_b2 - 0.2018786)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.log_sum_pt > 6.920349:
        z += -1.082654 * (18.0 - Q.n_dr_0p2_0p4) * (Q.log_sum_pt - 6.920349)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.sum_pt_top50 > 889.8503:
        z += 0.0005710203 * (18.0 - Q.n_dr_0p2_0p4) * (Q.sum_pt_top50 - 889.8503)
    if Q.psi_0p3 > 0.9956185 and Q.n_dr_0p1_0p2 < 26.0:
        z += 5.626852 * (Q.psi_0p3 - 0.9956185) * (26.0 - Q.n_dr_0p1_0p2)
    if Q.girth < 0.08589404 and Q.z_dr_0_0p05 < 0.4947602:
        z += 42.78277 * (0.08589404 - Q.girth) * (0.4947602 - Q.z_dr_0_0p05)
    if Q.sum_pt_top40 > 1001.523 and Q.sj2_dr < 0.1512157:
        z += 0.03746532 * (Q.sum_pt_top40 - 1001.523) * (0.1512157 - Q.sj2_dr)
    if Q.sum_pt_top40 > 1001.523 and Q.dr_5 < 0.02652372:
        z += 0.115514 * (Q.sum_pt_top40 - 1001.523) * (0.02652372 - Q.dr_5)
    if Q.psi_0p3 > 0.9956185 and Q.dr_5 < 0.02652372:
        z += -2525.557 * (Q.psi_0p3 - 0.9956185) * (0.02652372 - Q.dr_5)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.C3 < 0.0223982:
        z += -0.8861818 * (18.0 - Q.n_dr_0p2_0p4) * (0.0223982 - Q.C3)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.383267
    if Q.n_particles >= 38.0:
        z += 0.03313186 * Q.n_particles - 1.259011
    if Q.log_sum_pt < 6.910131:
        z += 11.74004 * Q.log_sum_pt - 83.8156
    if 6.910131 <= Q.log_sum_pt < 6.98945:
        z += 52.07114 * Q.log_sum_pt - 362.5088
    if 6.98945 <= Q.log_sum_pt < 7.139296:
        z += 19.42056 * Q.log_sum_pt - 134.2992
    if Q.log_sum_pt >= 7.139296:
        z += 7.680519 * Q.log_sum_pt - 50.48357
    if Q.sum_pt_top50 < 934.2416:
        z += -0.01521841 * Q.sum_pt_top50 + 16.42057
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += -0.0215203 * Q.sum_pt_top50 + 22.30806
    if 959.0957 <= Q.sum_pt_top50 < 1078.994:
        z += -0.03463144 * Q.sum_pt_top50 + 34.8829
    if Q.sum_pt_top50 >= 1078.994:
        z += -0.01941303 * Q.sum_pt_top50 + 18.46232
    if Q.girth2_top15 < 0.0007894752:
        z += -972.5931 * Q.girth2_top15 + 0.5664157
    if 0.0007894752 <= Q.girth2_top15 < 0.002197765:
        z += 143.0263 * Q.girth2_top15 - 0.3143382
    if Q.z_top5 >= 0.6551948:
        z += 0.7398232 * Q.z_top5 - 0.4847284
    if Q.psi_0p3 >= 0.9966167:
        z += -128.6288 * Q.psi_0p3 + 128.1936
    if Q.tau21 < 0.2140276:
        z += -11.25943 * Q.tau21 + 2.409828
    if Q.soft5_pt < 2.117188:
        z += -0.09729057 * Q.soft5_pt + 0.2059824
    if Q.lam2 < 0.0007143144:
        z += 492.6264 * Q.lam2 - 0.7110451
    if 0.0007143144 <= Q.lam2 < 0.001776308:
        z += 338.1892 * Q.lam2 - 0.6007284
    if Q.C2 < 0.05602756:
        z += -12.2842 * Q.C2 + 0.688254
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.1136504 * Q.n_dr_0p2_0p4 - 0.7955531
    if Q.M3 < 0.03457336:
        z += 9.169365 * Q.M3 - 0.3170157
    if Q.pt_11 < 29.04688:
        z += 0.03186058 * Q.pt_11 - 0.9254501
    if Q.D2_b2 < 0.2773918:
        z += 2.590853 * Q.D2_b2 - 0.9407385
    if 0.2773918 <= Q.D2_b2 < 3.852812:
        z += 0.06210653 * Q.D2_b2 - 0.2392848
    if Q.z_dr_0_0p05 >= 0.8103116:
        z += 2.298711 * Q.z_dr_0_0p05 - 1.862672
    if 972.0419 <= Q.sum_pt < 1052.889:
        z += 0.02260317 * Q.sum_pt - 21.97123
    if Q.sum_pt >= 1052.889:
        z += 0.01762538 * Q.sum_pt - 16.73017
    if Q.sum_pt_top2 < 605.875:
        z += -0.002084731 * Q.sum_pt_top2 + 1.263086
    if Q.z_top30_slots >= 0.9341838:
        z += 4.455287 * Q.z_top30_slots - 4.162057
    if 15.0 <= Q.n_dr_0_0p05 < 30.0:
        z += 0.04886422 * Q.n_dr_0_0p05 - 0.7329632
    if Q.n_dr_0_0p05 >= 30.0:
        z += -0.04584055 * Q.n_dr_0_0p05 + 2.10818
    if Q.n_dr_0p1_0p2 < 7.0:
        z += 0.1182106 * Q.n_dr_0p1_0p2 - 0.8274741
    if Q.soft9_pt < 2.5:
        z += -0.135616 * Q.soft9_pt + 0.3390401
    if Q.sj3_dr23 >= 0.2799759:
        z += 0.9355721 * Q.sj3_dr23 - 0.2619376
    if Q.tau1 < 0.07708632:
        z += -29.41114 * Q.tau1 + 2.267196
    if Q.tau1 >= 0.1219132:
        z += -3.112412 * Q.tau1 + 0.3794442
    if Q.sum_pt_top40 < 1225.842:
        z += -0.005358689 * Q.sum_pt_top40 + 6.568906
    if Q.pt_8 < 36.65625:
        z += 0.005562135 * Q.pt_8 - 0.203887
    if Q.n_particles > 38.0 and Q.zdr_0 < 0.009970338:
        z += 2.205955 * (Q.n_particles - 38.0) * (0.009970338 - Q.zdr_0)
    if Q.n_particles > 38.0 and Q.soft1_z < 0.002181998:
        z += -26.30569 * (Q.n_particles - 38.0) * (0.002181998 - Q.soft1_z)
    if Q.z_top5 > 0.6551948 and Q.pt1_dr01 < 28.39396:
        z += -0.1208861 * (Q.z_top5 - 0.6551948) * (28.39396 - Q.pt1_dr01)
    if Q.girth2_top15 < 0.002197765 and Q.psi_0p3 > 0.9966167:
        z += 55830.0 * (0.002197765 - Q.girth2_top15) * (Q.psi_0p3 - 0.9966167)
    if Q.n_particles > 38.0 and Q.absphi_1 < 0.1178619:
        z += 0.1041908 * (Q.n_particles - 38.0) * (0.1178619 - Q.absphi_1)
    if Q.z_top30_slots > 0.9341838 and Q.pt1_dr01 > 5.351077:
        z += 0.3928775 * (Q.z_top30_slots - 0.9341838) * (Q.pt1_dr01 - 5.351077)
    if Q.n_particles > 38.0 and Q.tau32 > 0.3293142:
        z += 0.1225615 * (Q.n_particles - 38.0) * (Q.tau32 - 0.3293142)
    if Q.M3 < 0.03457336 and Q.psi_0p3 > 0.9299135:
        z += -145.1977 * (0.03457336 - Q.M3) * (Q.psi_0p3 - 0.9299135)
    if Q.n_particles > 38.0 and Q.tau43 < 0.9624339:
        z += -0.1431495 * (Q.n_particles - 38.0) * (0.9624339 - Q.tau43)
    if Q.z_dr_0_0p05 > 0.8103116 and Q.n_dr_0p05_0p1 < 10.0:
        z += -0.9548762 * (Q.z_dr_0_0p05 - 0.8103116) * (10.0 - Q.n_dr_0p05_0p1)
    if Q.z_dr_0_0p05 > 0.8103116 and Q.n_real_top20 < 20.0:
        z += 1.094234 * (Q.z_dr_0_0p05 - 0.8103116) * (20.0 - Q.n_real_top20)
    if Q.pt_11 < 29.04688 and Q.dr1_12 < 0.1745719:
        z += 0.06687555 * (29.04688 - Q.pt_11) * (0.1745719 - Q.dr1_12)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 < 17.94219:
        z += -0.2092239 * (Q.z_top30_slots - 0.9341838) * (17.94219 - Q.ptdr0_3)
    if Q.soft5_pt < 2.117188 and Q.max_dr < 0.4357228:
        z += -1.204636 * (2.117188 - Q.soft5_pt) * (0.4357228 - Q.max_dr)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.03449324
    if 6.930088 <= Q.log_sum_pt < 7.062574:
        z += 4.38356 * Q.log_sum_pt - 30.37846
    if Q.log_sum_pt >= 7.062574:
        z += 6.431577 * Q.log_sum_pt - 44.84273
    if Q.lam2 < 0.0009731947:
        z += 412.4525 * Q.lam2 - 0.2455334
    if 0.0009731947 <= Q.lam2 < 0.001776308:
        z += -194.0737 * Q.lam2 + 0.3447347
    if Q.girth < 0.03577037:
        z += 28.07248 * Q.girth - 1.004163
    if 1064.139 <= Q.sum_pt_top20 < 1129.275:
        z += 0.003207055 * Q.sum_pt_top20 - 3.412752
    if Q.sum_pt_top20 >= 1129.275:
        z += 0.001072818 * Q.sum_pt_top20 - 1.002612
    if 997.0189 <= Q.sum_pt_top50 < 1038.855:
        z += -0.008027171 * Q.sum_pt_top50 + 8.003241
    if 1038.855 <= Q.sum_pt_top50 < 1078.994:
        z += -0.009064069 * Q.sum_pt_top50 + 9.080427
    if Q.sum_pt_top50 >= 1078.994:
        z += -0.01115812 * Q.sum_pt_top50 + 11.33989
    if Q.sum_pt >= 1085.125:
        z += -0.005004654 * Q.sum_pt + 5.430674
    if 985.0781 <= Q.sum_pt_top15 < 1082.548:
        z += 0.0007136905 * Q.sum_pt_top15 - 0.7030409
    if Q.sum_pt_top15 >= 1082.548:
        z += -0.002144732 * Q.sum_pt_top15 + 2.391338
    if Q.LHA < 0.1870291:
        z += -6.396485 * Q.LHA + 1.196329
    z += 3.232198 * Q.M3
    if Q.N2 >= 0.4454953:
        z += -2.607353 * Q.N2 + 1.161563
    if Q.z_6 < 0.04568661:
        z += -7.478852 * Q.z_6 + 0.3416834
    if Q.sum_pt_top30 >= 996.8867:
        z += -0.001292898 * Q.sum_pt_top30 + 1.288873
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.003663142 * Q.sum_pt_top40 - 3.918358
    if Q.log_sum_pt > 6.903423 and Q.girth2_top15 < 0.02146578:
        z += 210.704 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.girth2_top15)
    if Q.log_sum_pt > 6.903423 and Q.tau21 < 0.8065577:
        z += 10.1298 * (Q.log_sum_pt - 6.903423) * (0.8065577 - Q.tau21)
    if Q.log_sum_pt > 7.062574 and Q.girth2_top15 > 0.009962397:
        z += 663.1816 * (Q.log_sum_pt - 7.062574) * (Q.girth2_top15 - 0.009962397)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p1 < 0.8747961:
        z += -14.10162 * (Q.log_sum_pt - 6.903423) * (0.8747961 - Q.psi_0p1)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p3 < 1.0:
        z += -17.37886 * (Q.log_sum_pt - 6.903423) * (1.0 - Q.psi_0p3)
    if Q.sum_pt_top15 > 985.0781 and Q.M3 < 0.05840156:
        z += -0.02586048 * (Q.sum_pt_top15 - 985.0781) * (0.05840156 - Q.M3)
    if Q.sum_pt_top50 > 997.0189 and Q.zdr_4 < 0.006794973:
        z += 0.5482706 * (Q.sum_pt_top50 - 997.0189) * (0.006794973 - Q.zdr_4)
    if Q.sum_pt_top20 > 1129.275 and Q.z_7 > 0.02696541:
        z += 0.1997849 * (Q.sum_pt_top20 - 1129.275) * (Q.z_7 - 0.02696541)
    if Q.sum_pt_top20 > 1129.275 and Q.dr_max_012 > 0.1206357:
        z += -0.5504696 * (Q.sum_pt_top20 - 1129.275) * (Q.dr_max_012 - 0.1206357)
    if Q.log_sum_pt > 6.903423 and Q.dr_max_012 < 0.1206357:
        z += 25.56994 * (Q.log_sum_pt - 6.903423) * (0.1206357 - Q.dr_max_012)
    if Q.log_sum_pt > 6.903423 and Q.zdr_3 > 0.00453462:
        z += -1208.454 * (Q.log_sum_pt - 6.903423) * (Q.zdr_3 - 0.00453462)
    if Q.sum_pt_top50 > 997.0189 and Q.dr_6 < 0.2162512:
        z += 0.006895841 * (Q.sum_pt_top50 - 997.0189) * (0.2162512 - Q.dr_6)
    if Q.sum_pt_top15 > 985.0781 and Q.pt_7 > 41.65625:
        z += -7.521854e-05 * (Q.sum_pt_top15 - 985.0781) * (Q.pt_7 - 41.65625)
    if Q.log_sum_pt > 7.062574 and Q.zdr_14 < 0.001658225:
        z += 1622.22 * (Q.log_sum_pt - 7.062574) * (0.001658225 - Q.zdr_14)
    if Q.psi_0p1 > 0.8976117 and Q.z_top50_slots < 0.985099:
        z += 298.9467 * (Q.psi_0p1 - 0.8976117) * (0.985099 - Q.z_top50_slots)
    if Q.log_sum_pt > 6.903423 and Q.pt_12 < 10.49219:
        z += -0.6758553 * (Q.log_sum_pt - 6.903423) * (10.49219 - Q.pt_12)
    if Q.sum_pt > 1085.125 and Q.z_10 < 0.01389212:
        z += 0.5661517 * (Q.sum_pt - 1085.125) * (0.01389212 - Q.z_10)
    if Q.sum_pt_top20 > 1129.275 and Q.eta_1 > 0.08734131:
        z += -2.905582 * (Q.sum_pt_top20 - 1129.275) * (Q.eta_1 - 0.08734131)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.01276041
    if Q.lam2 < 0.0006154841:
        z += -419.156 * Q.lam2 + 0.2579838
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.1604953 * Q.n_dr_0p2_0p4 + 0.7420589
    if 5.0 <= Q.n_dr_0p2_0p4 < 8.0:
        z += 0.02013921 * Q.n_dr_0p2_0p4 - 0.1611137
    if Q.n_particles < 46.0:
        z += 0.02786845 * Q.n_particles - 1.281949
    if Q.tau21 < 0.347196:
        z += 7.816369 * Q.tau21 - 2.713812
    if Q.psi_0p2 >= 0.9985434:
        z += 314.9572 * Q.psi_0p2 - 314.4984
    if Q.n_dr_0p1_0p2 < 10.0:
        z += 4.415587e-05 * Q.n_dr_0p1_0p2 + 0.1466175
    if 10.0 <= Q.n_dr_0p1_0p2 < 17.0:
        z += -0.02100843 * Q.n_dr_0p1_0p2 + 0.3571433
    if Q.sum_pt_top10 >= 916.1578:
        z += -0.0007089652 * Q.sum_pt_top10 + 0.649524
    if Q.tau4 < 0.0174227:
        z += -137.3607 * Q.tau4 + 2.393193
    if Q.zdr_0 < 0.003235754:
        z += 156.0049 * Q.zdr_0 - 0.5047933
    if Q.z_dr_0p1_0p2 < 0.08871546:
        z += 1.977199 * Q.z_dr_0p1_0p2 - 0.1754081
    if Q.n_dr_0p2_0p4 < 5.0 and Q.sum_pt_top30 < 988.4375:
        z += -0.001405605 * (5.0 - Q.n_dr_0p2_0p4) * (988.4375 - Q.sum_pt_top30)
    if Q.n_particles < 46.0 and Q.sum_pt_top40 > 858.8262:
        z += 9.447006e-05 * (46.0 - Q.n_particles) * (Q.sum_pt_top40 - 858.8262)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_dr_0p1_0p2 > 9.0:
        z += -0.01938652 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 9.0)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.soft10_z < 0.004201797:
        z += 30.16244 * (5.0 - Q.n_dr_0p2_0p4) * (0.004201797 - Q.soft10_z)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.e2 < 0.03029714:
        z += -6.554055 * (5.0 - Q.n_dr_0p2_0p4) * (0.03029714 - Q.e2)
    if Q.n_particles < 46.0 and Q.e2 > 0.01036127:
        z += -1.365593 * (46.0 - Q.n_particles) * (Q.e2 - 0.01036127)
    if Q.D2 < 1.788105 and Q.soft5_z < 0.003582374:
        z += -74.09944 * (1.788105 - Q.D2) * (0.003582374 - Q.soft5_z)
    if Q.lam2 < 0.0006154841 and Q.eta_9 < -0.04559937:
        z += -6385.58 * (0.0006154841 - Q.lam2) * (-0.04559937 - Q.eta_9)
    if Q.tau21 < 0.347196 and Q.psi_0p1 < 0.8976117:
        z += 2.50955 * (0.347196 - Q.tau21) * (0.8976117 - Q.psi_0p1)
    if Q.n_particles < 46.0 and Q.eta_0 < 0.02980347:
        z += 0.238544 * (46.0 - Q.n_particles) * (0.02980347 - Q.eta_0)
    if Q.psi_0p3 > 0.9980008 and Q.zdr_5 < 0.005967398:
        z += -23237.06 * (Q.psi_0p3 - 0.9980008) * (0.005967398 - Q.zdr_5)
    if Q.tau21 < 0.347196 and Q.sum_pt < 1085.125:
        z += 0.0308963 * (0.347196 - Q.tau21) * (1085.125 - Q.sum_pt)
    if Q.psi_0p3 > 0.9980008 and Q.eccentricity > 0.8680812:
        z += 2798.584 * (Q.psi_0p3 - 0.9980008) * (Q.eccentricity - 0.8680812)
    if Q.psi_0p3 > 0.9980008 and Q.sj2_dr > 0.1512157:
        z += 1239.068 * (Q.psi_0p3 - 0.9980008) * (Q.sj2_dr - 0.1512157)
    if Q.tau4 < 0.0174227 and Q.pt_11 > 17.09375:
        z += 2.68664 * (0.0174227 - Q.tau4) * (Q.pt_11 - 17.09375)
    if Q.n_dr_0p1_0p2 < 10.0 and Q.planar_flow > 0.08857921:
        z += 0.02855276 * (10.0 - Q.n_dr_0p1_0p2) * (Q.planar_flow - 0.08857921)
    if Q.psi_0p3 > 0.9980008 and Q.soft7_z < 0.003052491:
        z += -50890.6 * (Q.psi_0p3 - 0.9980008) * (0.003052491 - Q.soft7_z)
    if Q.tau21 < 0.347196 and Q.soft7_dr < 0.03275811:
        z += -57.53009 * (0.347196 - Q.tau21) * (0.03275811 - Q.soft7_dr)
    if Q.lam2 < 0.0006154841 and Q.n_dr_0p4_up < 1.0:
        z += 511.8852 * (0.0006154841 - Q.lam2) * (1.0 - Q.n_dr_0p4_up)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.phi_0 < 0.01452637:
        z += 0.7829773 * (5.0 - Q.n_dr_0p2_0p4) * (0.01452637 - Q.phi_0)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.pt1_dr01 > 12.6865:
        z += 0.004887315 * (8.0 - Q.n_dr_0p2_0p4) * (Q.pt1_dr01 - 12.6865)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.sd_rg > 0.1596365:
        z += 3.530943 * (5.0 - Q.n_dr_0p2_0p4) * (Q.sd_rg - 0.1596365)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.sd_rg > 0.2042612:
        z += -6.903909 * (5.0 - Q.n_dr_0p2_0p4) * (Q.sd_rg - 0.2042612)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.784711
    if Q.e2 < 0.03263075:
        z += 30.42434 * Q.e2 - 1.356632
    if 0.03263075 <= Q.e2 < 0.04358622:
        z += 33.21293 * Q.e2 - 1.447626
    if Q.psi_0p3 >= 0.9980008:
        z += -63.73122 * Q.psi_0p3 + 63.60381
    if Q.n_dr_0p1_0p2 >= 11.0:
        z += -0.008289572 * Q.n_dr_0p1_0p2 + 0.09118529
    if Q.sum_pt_top40 < 1041.263:
        z += 0.01098756 * Q.sum_pt_top40 - 11.44094
    if Q.psi_0p1 >= 0.8976117:
        z += -5.866498 * Q.psi_0p1 + 5.265838
    if Q.n_particles >= 26.0:
        z += -0.02610649 * Q.n_particles + 0.6787686
    if Q.girth2_top15 < 0.002197765:
        z += -126.5788 * Q.girth2_top15 + 0.4819729
    if 0.002197765 <= Q.girth2_top15 < 0.004169954:
        z += 213.664 * Q.girth2_top15 - 0.2658007
    if 0.004169954 <= Q.girth2_top15 < 0.005788041:
        z += 119.5878 * Q.girth2_top15 + 0.1264924
    if 0.005788041 <= Q.girth2_top15 < 0.007887677:
        z += -98.21391 * Q.girth2_top15 + 1.387138
    if 0.007887677 <= Q.girth2_top15 < 0.009962397:
        z += -295.2005 * Q.girth2_top15 + 2.940905
    if Q.pt_entropy < 3.355186:
        z += -0.07155683 * Q.pt_entropy + 0.2400865
    if Q.girth < 0.05660088:
        z += 33.75348 * Q.girth - 1.766149
    if 0.05660088 <= Q.girth < 0.07374472:
        z += 2.083986 * Q.girth + 0.02637233
    if 0.07374472 <= Q.girth < 0.09749958:
        z += -7.579723 * Q.girth + 0.7390198
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.01806096 * Q.n_dr_0p2_0p4 + 0.3250972
    if Q.n_dr_0p2_0p4 >= 26.0:
        z += 0.03849768 * Q.n_dr_0p2_0p4 - 1.00094
    if Q.D3 < 0.1416054:
        z += -0.9462638 * Q.D3 + 0.1339961
    if Q.girth2_top10 < 0.006876086:
        z += 58.71069 * Q.girth2_top10 - 0.4036998
    if Q.e3 < 0.0005178279:
        z += 557.1972 * Q.e3 - 0.2885322
    if Q.sum_pt_top30 < 1011.524:
        z += -0.006800476 * Q.sum_pt_top30 + 6.878844
    if Q.soft1_pt >= 2.275391:
        z += 0.314191 * Q.soft1_pt - 0.7149072
    if Q.D2_b2 < 5.142486:
        z += 0.05408198 * Q.D2_b2 - 0.2781158
    if 0.1219342 <= Q.sj2_dr < 0.1825048:
        z += 7.596998 * Q.sj2_dr - 0.926334
    if 0.1825048 <= Q.sj2_dr < 0.2412757:
        z += 0.03766441 * Q.sj2_dr + 0.4532809
    if Q.sj2_dr >= 0.2412757:
        z += -2.480927 * Q.sj2_dr + 1.060956
    if Q.planar_flow < 0.3563114:
        z += -1.067304 * Q.planar_flow + 0.3802927
    if Q.z_dr_0p1_0p2 < 0.04748396:
        z += -6.909226 * Q.z_dr_0p1_0p2 + 0.3280774
    if Q.z_dr_0p2_0p4 < 0.05180474:
        z += 10.8291 * Q.z_dr_0p2_0p4 - 0.5609985
    if Q.n_dr_0p1_0p2 > 11.0 and Q.e3 > 0.0001841806:
        z += 49.45311 * (Q.n_dr_0p1_0p2 - 11.0) * (Q.e3 - 0.0001841806)
    if Q.psi_0p3 > 0.9980008 and Q.eccentricity > 0.773315:
        z += -270.6082 * (Q.psi_0p3 - 0.9980008) * (Q.eccentricity - 0.773315)
    if Q.n_particles > 26.0 and Q.tau21 > 0.1295048:
        z += 0.02324747 * (Q.n_particles - 26.0) * (Q.tau21 - 0.1295048)
    if Q.n_particles > 26.0 and Q.soft1_pt > 0.4909668:
        z += -0.01152824 * (Q.n_particles - 26.0) * (Q.soft1_pt - 0.4909668)
    if Q.pt_entropy < 3.355186 and Q.zdr_0 > 0.0008718296:
        z += -12.12592 * (3.355186 - Q.pt_entropy) * (Q.zdr_0 - 0.0008718296)
    if Q.n_particles > 26.0 and Q.sj3_dr_max > 0.3715619:
        z += 0.1384022 * (Q.n_particles - 26.0) * (Q.sj3_dr_max - 0.3715619)
    if Q.girth < 0.09749958 and Q.e3 > 4.646151e-05:
        z += -156525.0 * (0.09749958 - Q.girth) * (Q.e3 - 4.646151e-05)
    if Q.girth2_top15 < 0.007887677 and Q.sum_pt_top40 > 1095.686:
        z += 0.3228481 * (0.007887677 - Q.girth2_top15) * (Q.sum_pt_top40 - 1095.686)
    if Q.n_particles > 26.0 and Q.max_dr < 0.2982:
        z += 0.1800743 * (Q.n_particles - 26.0) * (0.2982 - Q.max_dr)
    if Q.girth < 0.09749958 and Q.psi_0p3 > 0.9973959:
        z += 7433.827 * (0.09749958 - Q.girth) * (Q.psi_0p3 - 0.9973959)
    if Q.girth < 0.05660088 and Q.psi_0p3 > 0.9973959:
        z += -12951.23 * (0.05660088 - Q.girth) * (Q.psi_0p3 - 0.9973959)
    if Q.girth2_top15 < 0.004169954 and Q.psi_0p3 > 0.9973959:
        z += -15770.52 * (0.004169954 - Q.girth2_top15) * (Q.psi_0p3 - 0.9973959)
    if Q.planar_flow < 0.3563114 and Q.soft5_dr0 > 0.054583:
        z += -3.87578 * (0.3563114 - Q.planar_flow) * (Q.soft5_dr0 - 0.054583)
    return max(0.0, z)


def neuron_5(Q):
    z = -2.147327
    z += -0.03809464 * Q.n_particles + 2.438057
    if Q.sum_pt < 907.9372:
        z += 0.04487545 * Q.sum_pt - 42.45049
    if 907.9372 <= Q.sum_pt < 1002.379:
        z += 0.02209967 * Q.sum_pt - 21.77151
    if 1002.379 <= Q.sum_pt < 1085.125:
        z += -0.004601079 * Q.sum_pt + 4.992745
    if Q.log_sum_pt < 7.139296:
        z += -2.075665 * Q.log_sum_pt + 14.81879
    if Q.e3 < 6.567534e-05:
        z += 4700.363 * Q.e3 - 0.01940651
    if 6.567534e-05 <= Q.e3 < 0.0005178279:
        z += -639.8094 * Q.e3 + 0.3313112
    if Q.sum_pt_top50 < 934.2416:
        z += -0.01068537 * Q.sum_pt_top50 + 9.13741
    if 934.2416 <= Q.sum_pt_top50 < 1078.994:
        z += 0.005839695 * Q.sum_pt_top50 - 6.300996
    if Q.z_top30_slots >= 0.9460751:
        z += -9.371917 * Q.z_top30_slots + 8.866537
    z += -0.2586077 * Q.z_top3_slots
    if Q.psi_0p3 >= 0.9973959:
        z += 62.55789 * Q.psi_0p3 - 62.39498
    if Q.girth < 0.1564779:
        z += -27.62104 * Q.girth + 4.322082
    if 0.1632346 <= Q.LHA < 0.302389:
        z += 4.657831 * Q.LHA - 0.7603193
    if Q.LHA >= 0.302389:
        z += 14.87501 * Q.LHA - 3.849882
    if Q.z_top20_slots >= 0.8281581:
        z += -4.275455 * Q.z_top20_slots + 3.540753
    if 0.7128619 <= Q.z_dr_0_0p05 < 0.878906:
        z += 4.912353 * Q.z_dr_0_0p05 - 3.501829
    if 0.878906 <= Q.z_dr_0_0p05 < 0.9351131:
        z += 7.462523 * Q.z_dr_0_0p05 - 5.743189
    if Q.z_dr_0_0p05 >= 0.9351131:
        z += 14.38208 * Q.z_dr_0_0p05 - 12.21375
    if Q.tau1 < 0.1072713:
        z += 18.6844 * Q.tau1 - 2.004299
    if Q.girth2_top15 < 0.0007894752:
        z += 517.4592 * Q.girth2_top15 - 0.4085212
    if Q.C2 >= 0.1235569:
        z += -13.48257 * Q.C2 + 1.665865
    if Q.tau4 < 0.01626937:
        z += -42.28345 * Q.tau4 + 0.6879249
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.04505433 * Q.n_dr_0p2_0p4 + 0.4955976
    if Q.D2 < 1.409617:
        z += -0.5106751 * Q.D2 + 0.7198565
    if Q.z_dr_0p05_0p1 < 0.05954375:
        z += 8.905272 * Q.z_dr_0p05_0p1 - 0.5302533
    if Q.z_top50_slots < 0.985099:
        z += 42.04724 * Q.z_top50_slots - 41.42069
    if Q.z_dr_0p2_0p4 < 0.05180474:
        z += 3.53425 * Q.z_dr_0p2_0p4 - 0.1830909
    if Q.sum_pt_top30 < 911.9328:
        z += -0.01694368 * Q.sum_pt_top30 + 15.45149
    if Q.z_8 < 0.02476077:
        z += -15.83212 * Q.z_8 + 0.3920154
    if Q.n_dr_0_0p05 < 8.0:
        z += 0.05016746 * Q.n_dr_0_0p05 - 0.4013397
    if Q.sum_pt_top40 < 1013.042:
        z += 0.004182865 * Q.sum_pt_top40 - 4.23742
    if Q.girth2_top10 >= 0.005337976:
        z += 9.167794 * Q.girth2_top10 - 0.04893747
    if Q.ptdr0_10 < 1.681792:
        z += 0.08144554 * Q.ptdr0_10 - 0.1369745
    if Q.log_sum_pt < 7.139296 and Q.soft2_pt > 1.413232:
        z += -0.4718765 * (7.139296 - Q.log_sum_pt) * (Q.soft2_pt - 1.413232)
    if Q.n_particles < 64.0 and Q.C2 < 0.07996447:
        z += -0.2555617 * (64.0 - Q.n_particles) * (0.07996447 - Q.C2)
    if Q.girth < 0.1564779 and Q.n_dr_0p1_0p2 > 11.0:
        z += 0.3604557 * (0.1564779 - Q.girth) * (Q.n_dr_0p1_0p2 - 11.0)
    if Q.e3 < 0.0005178279 and Q.max_dr > 0.1939977:
        z += 1708.615 * (0.0005178279 - Q.e3) * (Q.max_dr - 0.1939977)
    if Q.sum_pt < 1002.379 and Q.dr_11 < 0.199671:
        z += -0.01338704 * (1002.379 - Q.sum_pt) * (0.199671 - Q.dr_11)
    if Q.sum_pt < 1002.379 and Q.dr1_13 > 0.1778153:
        z += -0.02651645 * (1002.379 - Q.sum_pt) * (Q.dr1_13 - 0.1778153)
    if Q.girth < 0.1564779 and Q.psi_0p3 < 0.9638082:
        z += -154.6919 * (0.1564779 - Q.girth) * (0.9638082 - Q.psi_0p3)
    if Q.sum_pt < 1002.379 and Q.ptdr0_10 > 3.719859:
        z += -0.008867947 * (1002.379 - Q.sum_pt) * (Q.ptdr0_10 - 3.719859)
    if Q.sum_pt_top50 < 934.2416 and Q.soft6_dr < 0.03626613:
        z += 0.4501438 * (934.2416 - Q.sum_pt_top50) * (0.03626613 - Q.soft6_dr)
    if Q.sum_pt < 1002.379 and Q.soft6_dr < 0.03626613:
        z += 0.2288237 * (1002.379 - Q.sum_pt) * (0.03626613 - Q.soft6_dr)
    if Q.sum_pt_top15 > 967.7705 and Q.absphi_1 < 0.01500702:
        z += -0.1746119 * (Q.sum_pt_top15 - 967.7705) * (0.01500702 - Q.absphi_1)
    if Q.sum_pt < 1002.379 and Q.absphi_1 > 0.02227783:
        z += -0.1682097 * (1002.379 - Q.sum_pt) * (Q.absphi_1 - 0.02227783)
    if Q.sum_pt < 1002.379 and Q.e4 < 5.8505e-08:
        z += -333163.3 * (1002.379 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.sum_pt < 1085.125 and Q.e4 < 5.8505e-08:
        z += 66756.24 * (1085.125 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.1973746
    if Q.tau1 < 0.05444509:
        z += -84.86691 * Q.tau1 + 4.620587
    if 15.0 <= Q.n_dr_0p1_0p2 < 26.0:
        z += 0.006348311 * Q.n_dr_0p1_0p2 - 0.09522466
    if Q.n_dr_0p1_0p2 >= 26.0:
        z += -0.03179654 * Q.n_dr_0p1_0p2 + 0.8965414
    if Q.psi_0p2 >= 0.9087063:
        z += -8.686818 * Q.psi_0p2 + 7.893766
    if Q.C2 >= 0.1088881:
        z += -13.00078 * Q.C2 + 1.41563
    if Q.girth2_top15 < 0.001319197:
        z += 148.7842 * Q.girth2_top15 + 0.1764671
    if 0.001319197 <= Q.girth2_top15 < 0.003270031:
        z += -191.0684 * Q.girth2_top15 + 0.6247995
    if Q.girth < 0.09749958:
        z += 99.75546 * Q.girth - 9.726116
    if Q.e3 < 5.13841e-05:
        z += -3627.088 * Q.e3 + 1.223177
    if 5.13841e-05 <= Q.e3 < 0.0001086251:
        z += 6637.309 * Q.e3 + 0.6957503
    if 0.0001086251 <= Q.e3 < 0.0003372339:
        z += -7743.46 * Q.e3 + 2.257862
    if Q.e3 >= 0.0003372339:
        z += -4116.372 * Q.e3 + 1.034685
    if Q.sj2_zsoft < 0.04575675:
        z += -23.74784 * Q.sj2_zsoft + 1.777887
    if 0.04575675 <= Q.sj2_zsoft < 0.3676068:
        z += -2.14778 * Q.sj2_zsoft + 0.7895385
    if Q.e2 < 0.02515919:
        z += -34.20882 * Q.e2 + 1.179671
    if 0.02515919 <= Q.e2 < 0.04358622:
        z += -17.31178 * Q.e2 + 0.7545551
    if Q.e2 >= 0.05557149:
        z += 82.49693 * Q.e2 - 4.584477
    if Q.z_dr_0_0p05 >= 0.3289237:
        z += 0.585795 * Q.z_dr_0_0p05 - 0.1926818
    if Q.tau21_b2 >= 0.7058597:
        z += 1.439247 * Q.tau21_b2 - 1.015907
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += -0.01032681 * Q.n_dr_0p2_0p4 + 0.1549022
    if Q.tau2 < 0.0795038:
        z += -12.98466 * Q.tau2 + 1.03233
    if Q.psi_0p3 >= 0.9943058:
        z += -87.82087 * Q.psi_0p3 + 87.3208
    if Q.lam2 < 0.006427167:
        z += -179.3405 * Q.lam2 + 1.152652
    if Q.LHA < 0.3332345:
        z += -13.08947 * Q.LHA + 4.361864
    if Q.z_top20_slots >= 0.7818983:
        z += -1.203149 * Q.z_top20_slots + 0.9407404
    if Q.sum_pt_top50 >= 1078.994:
        z += 0.001845588 * Q.sum_pt_top50 - 1.991379
    if Q.sum_pt >= 907.9372:
        z += 0.0008198745 * Q.sum_pt - 0.7443945
    if Q.z_dr_0p1_0p2 >= 0.3340477:
        z += -1.662492 * Q.z_dr_0p1_0p2 + 0.5553517
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += 3.968516 * Q.z_dr_0p2_0p4 - 0.2718096
    if Q.n_dr_0p1_0p2 > 15.0 and Q.lam2 < 0.006427167:
        z += -6.786503 * (Q.n_dr_0p1_0p2 - 15.0) * (0.006427167 - Q.lam2)
    if Q.psi_0p2 > 0.9087063 and Q.sj2_dr < 0.1512157:
        z += 120.3547 * (Q.psi_0p2 - 0.9087063) * (0.1512157 - Q.sj2_dr)
    if Q.psi_0p2 > 0.9087063 and Q.z_top50_slots < 0.9704436:
        z += 251.9774 * (Q.psi_0p2 - 0.9087063) * (0.9704436 - Q.z_top50_slots)
    if Q.z_dr_0p05_0p1 < 0.2126484 and Q.zdr_0 > 0.01113024:
        z += 64.05955 * (0.2126484 - Q.z_dr_0p05_0p1) * (Q.zdr_0 - 0.01113024)
    if Q.girth2_top15 < 0.003270031 and Q.z_top15_slots > 0.7733683:
        z += -327.1253 * (0.003270031 - Q.girth2_top15) * (Q.z_top15_slots - 0.7733683)
    if Q.girth < 0.08589404 and Q.sum_pt < 1002.379:
        z += 0.1647348 * (0.08589404 - Q.girth) * (1002.379 - Q.sum_pt)
    if Q.e3 < 0.0003372339 and Q.log_sum_pt < 7.017258:
        z += -23960.88 * (0.0003372339 - Q.e3) * (7.017258 - Q.log_sum_pt)
    if Q.tau1 < 0.05444509 and Q.sum_pt < 972.0419:
        z += 0.3188101 * (0.05444509 - Q.tau1) * (972.0419 - Q.sum_pt)
    if Q.e3 > 5.13841e-05 and Q.sj3_dr_min < 0.06796146:
        z += 222909.8 * (Q.e3 - 5.13841e-05) * (0.06796146 - Q.sj3_dr_min)
    if Q.n_dr_0p1_0p2 > 15.0 and Q.psi_0p3 > 0.9853273:
        z += 6.441488 * (Q.n_dr_0p1_0p2 - 15.0) * (Q.psi_0p3 - 0.9853273)
    if Q.girth < 0.08589404 and Q.M2 < 0.1134943:
        z += -73.76533 * (0.08589404 - Q.girth) * (0.1134943 - Q.M2)
    if Q.sj2_zsoft < 0.3676068 and Q.zdr_1 < 0.008824206:
        z += -51.8937 * (0.3676068 - Q.sj2_zsoft) * (0.008824206 - Q.zdr_1)
    if Q.girth2_top15 < 0.003270031 and Q.sum_pt_top40 > 858.8262:
        z += -0.464384 * (0.003270031 - Q.girth2_top15) * (Q.sum_pt_top40 - 858.8262)
    if Q.girth < 0.09749958 and Q.eccentricity > 0.6414784:
        z += -30.6398 * (0.09749958 - Q.girth) * (Q.eccentricity - 0.6414784)
    if Q.psi_0p3 > 0.9943058 and Q.soft1_z > 0.0004000768:
        z += 28465.33 * (Q.psi_0p3 - 0.9943058) * (Q.soft1_z - 0.0004000768)
    if Q.n_dr_0p2_0p4 > 15.0 and Q.eta_0 < 0.07952881:
        z += 0.09494701 * (Q.n_dr_0p2_0p4 - 15.0) * (0.07952881 - Q.eta_0)
    if Q.girth < 0.09749958 and Q.psi_0p3 < 0.9638082:
        z += 1504.791 * (0.09749958 - Q.girth) * (0.9638082 - Q.psi_0p3)
    if Q.e2 < 0.04358622 and Q.psi_0p3 < 0.9638082:
        z += -1155.044 * (0.04358622 - Q.e2) * (0.9638082 - Q.psi_0p3)
    if Q.n_dr_0p2_0p4 > 15.0 and Q.sj3_dr13 < 0.06988208:
        z += 2.993562 * (Q.n_dr_0p2_0p4 - 15.0) * (0.06988208 - Q.sj3_dr13)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.03619377
    if Q.tau21_b2 < 0.2352054:
        z += -2.854577 * Q.tau21_b2 + 0.6714121
    if Q.e3 < 3.376709e-05:
        z += 18539.71 * Q.e3 - 0.2709151
    if 3.376709e-05 <= Q.e3 < 5.727594e-05:
        z += -9388.979 * Q.e3 + 0.6721557
    if 5.727594e-05 <= Q.e3 < 6.567534e-05:
        z += -16000.32 * Q.e3 + 1.050826
    if Q.n_dr_0p2_0p4 < 15.0:
        z += 0.006283671 * Q.n_dr_0p2_0p4 + 0.1888006
    if 15.0 <= Q.n_dr_0p2_0p4 < 21.0:
        z += -0.04717594 * Q.n_dr_0p2_0p4 + 0.9906948
    if Q.psi_0p3 >= 0.9973959:
        z += 1374.377 * Q.psi_0p3 - 1370.798
    if Q.psi_0p1 >= 0.8509811:
        z += -4.027382 * Q.psi_0p1 + 3.427226
    if Q.tau1 < 0.1072713:
        z += 77.41214 * Q.tau1 - 7.946503
    if 0.1072713 <= Q.tau1 < 0.1219132:
        z += -24.42253 * Q.tau1 + 2.97743
    if Q.LHA < 0.302389:
        z += -27.22874 * Q.LHA + 7.894689
    if 0.302389 <= Q.LHA < 0.3332345:
        z += 10.98969 * Q.LHA - 3.662146
    if Q.e2 < 0.03680582:
        z += -68.10569 * Q.e2 + 2.858759
    if 0.03680582 <= Q.e2 < 0.04755309:
        z += -7.9944 * Q.e2 + 0.6463144
    if 0.04755309 <= Q.e2 < 0.05557149:
        z += -33.19315 * Q.e2 + 1.844593
    if Q.girth < 0.08068193:
        z += 55.11058 * Q.girth - 3.475566
    if 0.08068193 <= Q.girth < 0.09749958:
        z += -57.72877 * Q.girth + 5.628531
    if Q.tau21 < 0.347196:
        z += 3.742788 * Q.tau21 - 1.299481
    if Q.girth2_top2 < 0.005180665:
        z += -39.44983 * Q.girth2_top2 + 0.2043763
    if Q.n_dr_0p1_0p2 >= 33.0:
        z += -0.02779337 * Q.n_dr_0p1_0p2 + 0.9171812
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1937688:
        z += 72.44674 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1937688)
    if Q.tau21_b2 < 0.2352054 and Q.girth2_top15 < 0.007887677:
        z += -10669.11 * (0.2352054 - Q.tau21_b2) * (0.007887677 - Q.girth2_top15)
    if Q.tau21_b2 < 0.2352054 and Q.girth2_top15 < 0.009962397:
        z += 7195.823 * (0.2352054 - Q.tau21_b2) * (0.009962397 - Q.girth2_top15)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt_top50 < 1245.697:
        z += -0.04687746 * (0.2352054 - Q.tau21_b2) * (1245.697 - Q.sum_pt_top50)
    if Q.tau21_b2 < 0.2352054 and Q.girth2_top15 < 0.005788041:
        z += 2001.832 * (0.2352054 - Q.tau21_b2) * (0.005788041 - Q.girth2_top15)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.e3 < 7.876005e-05:
        z += -1904.09 * (21.0 - Q.n_dr_0p2_0p4) * (7.876005e-05 - Q.e3)
    if Q.tau21_b2 < 0.2352054 and Q.D3 < 0.2618467:
        z += 7.53769 * (0.2352054 - Q.tau21_b2) * (0.2618467 - Q.D3)
    if Q.psi_0p3 > 0.9973959 and Q.mean_eta2 < 0.006802603:
        z += -167306.2 * (Q.psi_0p3 - 0.9973959) * (0.006802603 - Q.mean_eta2)
    if Q.psi_0p3 > 0.9973959 and Q.z_top50_slots < 0.985099:
        z += 4389.241 * (Q.psi_0p3 - 0.9973959) * (0.985099 - Q.z_top50_slots)
    if Q.psi_0p3 > 0.9973959 and Q.log_sum_pt < 7.062574:
        z += -3749.992 * (Q.psi_0p3 - 0.9973959) * (7.062574 - Q.log_sum_pt)
    if Q.psi_0p3 > 0.9973959 and Q.mean_phi2 < 0.006808102:
        z += -160944.1 * (Q.psi_0p3 - 0.9973959) * (0.006808102 - Q.mean_phi2)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.e3 < 3.376709e-05:
        z += 438.5875 * (21.0 - Q.n_dr_0p2_0p4) * (3.376709e-05 - Q.e3)
    if Q.psi_0p3 > 0.9973959 and Q.girth2_top10 > 0.007678544:
        z += -230308.0 * (Q.psi_0p3 - 0.9973959) * (Q.girth2_top10 - 0.007678544)
    if Q.psi_0p3 > 0.9973959 and Q.girth2_top10 > 0.01414829:
        z += 163177.3 * (Q.psi_0p3 - 0.9973959) * (Q.girth2_top10 - 0.01414829)
    if Q.tau1 < 0.1072713 and Q.z_dr_0p1_0p2 < 0.2864926:
        z += 260.8084 * (0.1072713 - Q.tau1) * (0.2864926 - Q.z_dr_0p1_0p2)
    if Q.LHA < 0.3332345 and Q.z_dr_0p1_0p2 < 0.250441:
        z += -103.2696 * (0.3332345 - Q.LHA) * (0.250441 - Q.z_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.n_dr_0p1_0p2 > 21.0:
        z += 0.001228597 * (21.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 21.0)
    if Q.tau21_b2 < 0.2352054 and Q.M2 < 0.06975954:
        z += 109.139 * (0.2352054 - Q.tau21_b2) * (0.06975954 - Q.M2)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.M2 < 0.1134943:
        z += 0.9622794 * (15.0 - Q.n_dr_0p2_0p4) * (0.1134943 - Q.M2)
    if Q.tau21_b2 < 0.2352054 and Q.n_real_top40 < 36.0:
        z += -0.1545847 * (0.2352054 - Q.tau21_b2) * (36.0 - Q.n_real_top40)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.max_dr < 0.4357228:
        z += 0.09156989 * (15.0 - Q.n_dr_0p2_0p4) * (0.4357228 - Q.max_dr)
    if Q.psi_0p3 > 0.9973959 and Q.lam2 < 0.003687605:
        z += 120198.2 * (Q.psi_0p3 - 0.9973959) * (0.003687605 - Q.lam2)
    if Q.psi_0p3 > 0.9973959 and Q.zdr_14 < 0.001127591:
        z += 62597.74 * (Q.psi_0p3 - 0.9973959) * (0.001127591 - Q.zdr_14)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.63605
    if 0.03577037 <= Q.girth < 0.04362872:
        z += 50.45742 * Q.girth - 1.804881
    if 0.04362872 <= Q.girth < 0.07031778:
        z += 136.8233 * Q.girth - 5.572912
    if 0.07031778 <= Q.girth < 0.07374472:
        z += 103.736 * Q.girth - 3.246287
    if 0.07374472 <= Q.girth < 0.09749958:
        z += 53.81096 * Q.girth + 0.4354201
    if Q.girth >= 0.09749958:
        z += -80.36541 * Q.girth + 13.51756
    if Q.sum_pt_top40 < 1007.44:
        z += -0.008818357 * Q.sum_pt_top40 + 8.883966
    if Q.n_dr_0p2_0p4 < 3.0:
        z += 0.1372473 * Q.n_dr_0p2_0p4 - 1.391369
    if 3.0 <= Q.n_dr_0p2_0p4 < 8.0:
        z += 0.1170068 * Q.n_dr_0p2_0p4 - 1.330647
    if 8.0 <= Q.n_dr_0p2_0p4 < 18.0:
        z += 0.009098601 * Q.n_dr_0p2_0p4 - 0.4673817
    if Q.n_dr_0p2_0p4 >= 18.0:
        z += -0.02024046 * Q.n_dr_0p2_0p4 + 0.06072138
    if Q.n_pt_above_1 >= 54.0:
        z += 0.04988731 * Q.n_pt_above_1 - 2.693915
    if Q.sum_pt < 1002.379:
        z += -0.02771937 * Q.sum_pt + 31.00839
    if 1002.379 <= Q.sum_pt < 1042.609:
        z += -0.02106917 * Q.sum_pt + 24.34238
    if 1042.609 <= Q.sum_pt < 1260.541:
        z += -0.01090005 * Q.sum_pt + 13.73995
    if 0.2091025 <= Q.LHA < 0.3098384:
        z += -22.77115 * Q.LHA + 4.761504
    if 0.3098384 <= Q.LHA < 0.3332345:
        z += -32.09516 * Q.LHA + 7.650442
    if 0.3332345 <= Q.LHA < 0.3719813:
        z += -6.199833 * Q.LHA - 0.9787766
    if Q.LHA >= 0.3719813:
        z += 22.07304 * Q.LHA - 11.49576
    if Q.psi_0p3 >= 0.9777125:
        z += -42.52375 * Q.psi_0p3 + 41.576
    if Q.z_top50_slots >= 0.9586536:
        z += 9.07135 * Q.z_top50_slots - 8.696283
    if Q.e3 < 3.793233e-05:
        z += -14462.96 * Q.e3 + 0.5486137
    if Q.girth2_top15 < 0.00727763:
        z += 283.8055 * Q.girth2_top15 - 2.929366
    if 0.00727763 <= Q.girth2_top15 < 0.009962397:
        z += 248.5712 * Q.girth2_top15 - 2.672944
    if 0.009962397 <= Q.girth2_top15 < 0.02675364:
        z += 11.70725 * Q.girth2_top15 - 0.3132116
    if Q.log_sum_pt < 7.139296:
        z += 20.4037 * Q.log_sum_pt - 145.668
    if Q.n_particles >= 43.0:
        z += 0.01221129 * Q.n_particles - 0.5250855
    if Q.girth2_top5 < 0.007164202:
        z += -80.71962 * Q.girth2_top5 + 0.5782917
    if Q.n_dr_0_0p05 >= 30.0:
        z += -0.08829957 * Q.n_dr_0_0p05 + 2.648987
    if 0.02210818 <= Q.e2 < 0.04358622:
        z += -41.90488 * Q.e2 + 0.9264408
    if Q.e2 >= 0.04358622:
        z += -32.0817 * Q.e2 + 0.4982852
    if Q.D2 < 2.178951:
        z += 0.2457859 * Q.D2 - 0.5355553
    if Q.sj2_dr >= 0.2232169:
        z += 2.565155 * Q.sj2_dr - 0.5725859
    if Q.dr_0 < 0.08082334:
        z += -5.271465 * Q.dr_0 + 0.4260574
    if Q.n_dr_0p2_0p4 < 18.0 and Q.log_sum_pt < 7.139296:
        z += -0.4669921 * (18.0 - Q.n_dr_0p2_0p4) * (7.139296 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.psi_0p1 < 0.8252004:
        z += 0.1874768 * (18.0 - Q.n_dr_0p2_0p4) * (0.8252004 - Q.psi_0p1)
    if Q.girth > 0.07031778 and Q.sum_pt_top20 > 869.693:
        z += -0.1203028 * (Q.girth - 0.07031778) * (Q.sum_pt_top20 - 869.693)
    if Q.girth > 0.07031778 and Q.e2 > 0.04358622:
        z += 1154.923 * (Q.girth - 0.07031778) * (Q.e2 - 0.04358622)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.e3 < 3.793233e-05:
        z += -4573.898 * (18.0 - Q.n_dr_0p2_0p4) * (3.793233e-05 - Q.e3)
    if Q.girth > 0.07031778 and Q.tau2 < 0.0795038:
        z += 844.8902 * (Q.girth - 0.07031778) * (0.0795038 - Q.tau2)
    if Q.psi_0p3 > 0.9777125 and Q.n_dr_0p1_0p2 > 17.0:
        z += 0.7860366 * (Q.psi_0p3 - 0.9777125) * (Q.n_dr_0p1_0p2 - 17.0)
    if Q.girth > 0.07031778 and Q.sum_pt_top30 < 1191.938:
        z += 0.08443517 * (Q.girth - 0.07031778) * (1191.938 - Q.sum_pt_top30)
    if Q.sum_pt_top40 < 1007.44 and Q.sum_pt > 949.9169:
        z += -7.201269e-05 * (1007.44 - Q.sum_pt_top40) * (Q.sum_pt - 949.9169)
    if Q.girth > 0.07031778 and Q.sum_pt_top50 < 976.277:
        z += -0.2855896 * (Q.girth - 0.07031778) * (976.277 - Q.sum_pt_top50)
    if Q.girth > 0.07031778 and Q.sum_pt < 1115.723:
        z += 0.1252418 * (Q.girth - 0.07031778) * (1115.723 - Q.sum_pt)
    if Q.sum_pt < 1042.609 and Q.max_dr > 0.1939977:
        z += 0.02009347 * (1042.609 - Q.sum_pt) * (Q.max_dr - 0.1939977)
    if Q.sum_pt < 1002.379 and Q.z_0 > 0.08872428:
        z += -0.0387021 * (1002.379 - Q.sum_pt) * (Q.z_0 - 0.08872428)
    if Q.girth2_top15 < 0.02675364 and Q.tau4 > 0.01517184:
        z += 2321.935 * (0.02675364 - Q.girth2_top15) * (Q.tau4 - 0.01517184)
    if Q.girth2_top5 < 0.007164202 and Q.sj3_dr23 > 0.2563461:
        z += 477.9883 * (0.007164202 - Q.girth2_top5) * (Q.sj3_dr23 - 0.2563461)
    if Q.LHA > 0.3203321 and Q.z_14 < 0.02037449:
        z += -617.8351 * (Q.LHA - 0.3203321) * (0.02037449 - Q.z_14)
    if Q.sum_pt < 1260.541 and Q.n_dr_0p05_0p1 > 7.0:
        z += -6.030214e-05 * (1260.541 - Q.sum_pt) * (Q.n_dr_0p05_0p1 - 7.0)
    if Q.sum_pt < 1260.541 and Q.planar_flow < 0.7225388:
        z += -0.001345833 * (1260.541 - Q.sum_pt) * (0.7225388 - Q.planar_flow)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.4635891
    if Q.e3 < 3.376709e-05:
        z += 20437.36 * Q.e3 - 0.6901101
    if Q.sum_pt_top50 < 889.8503:
        z += -0.03404342 * Q.sum_pt_top50 + 31.64539
    if 889.8503 <= Q.sum_pt_top50 < 988.4554:
        z += -0.0137097 * Q.sum_pt_top50 + 13.55142
    if Q.girth < 0.05660088:
        z += -48.61309 * Q.girth + 2.751544
    if 0.09749958 <= Q.girth < 0.1207452:
        z += 23.79867 * Q.girth - 2.32036
    if 0.1207452 <= Q.girth < 0.1402186:
        z += -41.35681 * Q.girth + 5.54685
    if 0.1402186 <= Q.girth < 0.1564779:
        z += -154.6216 * Q.girth + 21.42868
    if Q.girth >= 0.1564779:
        z += -128.2471 * Q.girth + 17.30165
    if Q.sj2_dr < 0.1825048:
        z += 1.853223 * Q.sj2_dr - 0.4025456
    if 0.1825048 <= Q.sj2_dr < 0.2595052:
        z += 0.8353653 * Q.sj2_dr - 0.2167816
    if Q.z_dr_0_0p05 >= 0.8103116:
        z += 4.724021 * Q.z_dr_0_0p05 - 3.827929
    if Q.z_dr_0p1_0p2 >= 0.4479367:
        z += 0.495936 * Q.z_dr_0p1_0p2 - 0.2221479
    if Q.tau21_b2 >= 0.6133424:
        z += -0.4702859 * Q.tau21_b2 + 0.2884463
    if Q.z_top5_slots >= 0.7963976:
        z += 3.634252 * Q.z_top5_slots - 2.89431
    if Q.LHA < 0.2454112:
        z += 14.9413 * Q.LHA - 3.666762
    if 0.3719813 <= Q.LHA < 0.404204:
        z += 17.28621 * Q.LHA - 6.430145
    if Q.LHA >= 0.404204:
        z += 50.40779 * Q.LHA - 19.81802
    if Q.sum_pt_top20 >= 750.7313:
        z += -0.004209476 * Q.sum_pt_top20 + 3.160185
    if Q.girth2_top15 < 0.0007894752:
        z += 1339.835 * Q.girth2_top15 - 1.96586
    if 0.0007894752 <= Q.girth2_top15 < 0.004855289:
        z += 148.9151 * Q.girth2_top15 - 1.025659
    if 0.004855289 <= Q.girth2_top15 < 0.006142802:
        z += 235.0523 * Q.girth2_top15 - 1.44388
    if Q.girth2_top15 >= 0.02146578:
        z += -181.2872 * Q.girth2_top15 + 3.891472
    if Q.D2 < 2.178951:
        z += -0.6707485 * Q.D2 + 1.461528
    if Q.psi_0p1 >= 0.8509811:
        z += 1.204019 * Q.psi_0p1 - 1.024597
    if Q.z_top40_slots >= 0.9674996:
        z += -9.601073 * Q.z_top40_slots + 9.289035
    if Q.tau1 < 0.09591084:
        z += -42.02554 * Q.tau1 + 4.227694
    if 0.09591084 <= Q.tau1 < 0.1072713:
        z += -17.34003 * Q.tau1 + 1.860087
    if Q.e2 >= 0.02515919:
        z += 38.76126 * Q.e2 - 0.975202
    if Q.psi_0p3 >= 0.9966167:
        z += -37.72649 * Q.psi_0p3 + 37.59886
    if Q.log_sum_pt < 6.811175:
        z += 17.67814 * Q.log_sum_pt - 120.4089
    if Q.girth2_top2 < 0.002412891:
        z += 209.5545 * Q.girth2_top2 - 0.5056321
    if Q.sj3_dr13 >= 0.1025461:
        z += -0.8779621 * Q.sj3_dr13 + 0.0900316
    if Q.girth < 0.05660088 and Q.sum_pt < 1007.788:
        z += -0.1898803 * (0.05660088 - Q.girth) * (1007.788 - Q.sum_pt)
    if Q.sum_pt_top50 < 988.4554 and Q.sum_pt_top20 > 789.7344:
        z += -4.1383e-05 * (988.4554 - Q.sum_pt_top50) * (Q.sum_pt_top20 - 789.7344)
    if Q.girth < 0.05660088 and Q.psi_0p3 > 0.9638082:
        z += 556.2535 * (0.05660088 - Q.girth) * (Q.psi_0p3 - 0.9638082)
    if Q.girth2_top15 < 0.006142802 and Q.psi_0p3 > 0.9777125:
        z += 5695.794 * (0.006142802 - Q.girth2_top15) * (Q.psi_0p3 - 0.9777125)
    if Q.girth2_top15 < 0.006142802 and Q.z_dr_0p2_0p4 < 0.0684915:
        z += 6842.431 * (0.006142802 - Q.girth2_top15) * (0.0684915 - Q.z_dr_0p2_0p4)
    if Q.girth2_top15 < 0.006142802 and Q.n_dr_0p2_0p4 > 7.0:
        z += 22.43311 * (0.006142802 - Q.girth2_top15) * (Q.n_dr_0p2_0p4 - 7.0)
    if Q.sum_pt_top50 < 988.4554 and Q.e3 > 0.0001086251:
        z += -9.968943 * (988.4554 - Q.sum_pt_top50) * (Q.e3 - 0.0001086251)
    if Q.LHA > 0.3719813 and Q.lam2 < 0.006427167:
        z += 1783.807 * (Q.LHA - 0.3719813) * (0.006427167 - Q.lam2)
    if Q.sum_pt_top50 < 988.4554 and Q.z_1st < 0.2640475:
        z += 0.01394461 * (988.4554 - Q.sum_pt_top50) * (0.2640475 - Q.z_1st)
    if Q.sj2_dr < 0.2595052 and Q.pt_0 > 213.5:
        z += 0.002798738 * (0.2595052 - Q.sj2_dr) * (Q.pt_0 - 213.5)
    if Q.z_top40_slots > 0.9674996 and Q.C3 < 0.01158441:
        z += -789.1676 * (Q.z_top40_slots - 0.9674996) * (0.01158441 - Q.C3)
    if Q.sum_pt_top50 < 988.4554 and Q.soft5_pt > 1.652344:
        z += -0.003141645 * (988.4554 - Q.sum_pt_top50) * (Q.soft5_pt - 1.652344)
    if Q.girth > 0.1402186 and Q.soft6_pt > 4.250195:
        z += 89.83377 * (Q.girth - 0.1402186) * (Q.soft6_pt - 4.250195)
    if Q.log_sum_pt < 6.811175 and Q.dr_13 < 0.09061548:
        z += 953.291 * (6.811175 - Q.log_sum_pt) * (0.09061548 - Q.dr_13)
    if Q.sum_pt_top50 < 988.4554 and Q.dr_13 < 0.1312677:
        z += -0.0855588 * (988.4554 - Q.sum_pt_top50) * (0.1312677 - Q.dr_13)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.soft5_z > 0.0004140594:
        z += 1101.049 * (Q.z_dr_0p1_0p2 - 0.4479367) * (Q.soft5_z - 0.0004140594)
    return max(0.0, z)


def neuron_10(Q):
    z = 5.981535
    if Q.girth < 0.09749958:
        z += 37.38785 * Q.girth - 4.514403
    if 0.09749958 <= Q.girth < 0.1207452:
        z += -4.43074 * Q.girth - 0.4371074
    if Q.girth >= 0.1207452:
        z += -41.81859 * Q.girth + 4.077295
    if 0.1751567 <= Q.tau1 < 0.1953848:
        z += -18.2468 * Q.tau1 + 3.196049
    if Q.tau1 >= 0.1953848:
        z += -91.32378 * Q.tau1 + 17.47418
    if Q.e2 >= 0.03029714:
        z += 43.01856 * Q.e2 - 1.30334
    if 402.625 <= Q.sum_pt_top5 < 791.125:
        z += -0.002733228 * Q.sum_pt_top5 + 1.100466
    if Q.sum_pt_top5 >= 791.125:
        z += -0.001389593 * Q.sum_pt_top5 + 0.03748286
    if Q.soft7_z < 0.002672224:
        z += 66.46197 * Q.soft7_z - 0.1776013
    if Q.sj2_dr >= 0.09395198:
        z += -1.440774 * Q.sj2_dr + 0.1353635
    if Q.psi_0p3 >= 0.9989733:
        z += -565.8525 * Q.psi_0p3 + 565.2716
    if Q.sd_rg >= 0.3017146:
        z += -7.644865 * Q.sd_rg + 2.306567
    if 0.1870291 <= Q.LHA < 0.3332345:
        z += -14.37101 * Q.LHA + 2.687797
    if Q.LHA >= 0.3332345:
        z += 9.694958 * Q.LHA - 5.331814
    if Q.n_dr_0p1_0p2 >= 21.0:
        z += 0.0272985 * Q.n_dr_0p1_0p2 - 0.5732685
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.02891672 * Q.n_dr_0p2_0p4 - 0.318084
    if Q.girth2_top10 < 0.01414829:
        z += 29.5437 * Q.girth2_top10 - 0.4179928
    if Q.z_dr_0_0p05 >= 0.7128619:
        z += -1.218492 * Q.z_dr_0_0p05 + 0.8686164
    if Q.C2 < 0.05602756:
        z += -19.28752 * Q.C2 + 1.080633
    if Q.ptdr0_2 >= 7.740999:
        z += 0.008955806 * Q.ptdr0_2 - 0.06932688
    if Q.tau32 < 0.43805:
        z += -3.340547 * Q.tau32 + 1.463327
    if Q.tau21_b2 < 0.2018786:
        z += 3.986228 * Q.tau21_b2 - 0.8047341
    if Q.log_sum_pt < 6.856375:
        z += 8.3409 * Q.log_sum_pt - 57.18834
    if Q.D2 < 2.410481:
        z += -0.3269221 * Q.D2 + 0.7880397
    if Q.sum_pt < 986.0565:
        z += -0.005785391 * Q.sum_pt + 5.704723
    if 0.9184255 <= Q.psi_0p1 < 0.9848104:
        z += 11.88939 * Q.psi_0p1 - 10.91952
    if Q.psi_0p1 >= 0.9848104:
        z += -42.16029 * Q.psi_0p1 + 42.30917
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += 4.085085 * Q.z_dr_0p1_0p2 - 0.4916142
    if Q.n_pt_above_5 >= 42.0:
        z += -0.02203952 * Q.n_pt_above_5 + 0.92566
    if Q.sum_pt_top30 < 886.3438:
        z += -0.001124629 * Q.sum_pt_top30 + 0.996808
    if Q.girth2_top5 < 0.02441963:
        z += 26.13477 * Q.girth2_top5 - 0.6382015
    if Q.girth < 0.1207452 and Q.psi_0p2 > 0.948102:
        z += 147.5541 * (0.1207452 - Q.girth) * (Q.psi_0p2 - 0.948102)
    if Q.e2 > 0.03029714 and Q.log_sum_pt > 6.910131:
        z += -140.646 * (Q.e2 - 0.03029714) * (Q.log_sum_pt - 6.910131)
    if Q.girth < 0.1207452 and Q.pt1_dr01 > 5.351077:
        z += 0.2147629 * (0.1207452 - Q.girth) * (Q.pt1_dr01 - 5.351077)
    if Q.n_dr_0p2_0p4 < 11.0 and Q.n_real_top40 > 22.0:
        z += -0.001221733 * (11.0 - Q.n_dr_0p2_0p4) * (Q.n_real_top40 - 22.0)
    if Q.LHA > 0.1870291 and Q.psi_0p3 < 0.9973959:
        z += -15.52825 * (Q.LHA - 0.1870291) * (0.9973959 - Q.psi_0p3)
    if Q.sum_pt_top50 < 889.8503 and Q.planar_flow > 0.4118472:
        z += -0.00957729 * (889.8503 - Q.sum_pt_top50) * (Q.planar_flow - 0.4118472)
    if Q.sj2_dr > 0.09395198 and Q.planar_flow < 0.6025827:
        z += -4.200746 * (Q.sj2_dr - 0.09395198) * (0.6025827 - Q.planar_flow)
    if Q.tau1 > 0.1953848 and Q.sum_pt > 1167.447:
        z += 0.3643467 * (Q.tau1 - 0.1953848) * (Q.sum_pt - 1167.447)
    if Q.sj2_dr > 0.09395198 and Q.M2 > 0.04260132:
        z += -43.37274 * (Q.sj2_dr - 0.09395198) * (Q.M2 - 0.04260132)
    if Q.tau1 > 0.1953848 and Q.pt_3 < 114.75:
        z += 1.977632 * (Q.tau1 - 0.1953848) * (114.75 - Q.pt_3)
    if Q.e3 < 0.0003372339 and Q.tau32 < 0.8644992:
        z += -3731.737 * (0.0003372339 - Q.e3) * (0.8644992 - Q.tau32)
    if Q.LHA > 0.1870291 and Q.pt_4 < 90.625:
        z += -0.01631537 * (Q.LHA - 0.1870291) * (90.625 - Q.pt_4)
    if Q.girth < 0.1207452 and Q.sum_pt_top50 < 997.0189:
        z += 0.03966513 * (0.1207452 - Q.girth) * (997.0189 - Q.sum_pt_top50)
    if Q.tau1 > 0.1953848 and Q.z_3 < 0.1082864:
        z += -1368.965 * (Q.tau1 - 0.1953848) * (0.1082864 - Q.z_3)
    if Q.e2 > 0.03029714 and Q.z_dr_0p4_up > 0.0:
        z += 908.8702 * (Q.e2 - 0.03029714) * (Q.z_dr_0p4_up - 0.0)
    if Q.psi_0p3 > 0.9989733 and Q.z_dr_0p05_0p1 < 0.5912328:
        z += 472.276 * (Q.psi_0p3 - 0.9989733) * (0.5912328 - Q.z_dr_0p05_0p1)
    if Q.LHA > 0.1870291 and Q.sj3_pairmin_over_m > 0.1389615:
        z += 9.058622 * (Q.LHA - 0.1870291) * (Q.sj3_pairmin_over_m - 0.1389615)
    if Q.tau1 > 0.1953848 and Q.dr1_9 < 0.1651809:
        z += -105.2096 * (Q.tau1 - 0.1953848) * (0.1651809 - Q.dr1_9)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.pt_7 > 34.53125:
        z += -0.1175024 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.pt_7 - 34.53125)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.2624938
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.04519369 * Q.n_dr_0p2_0p4 + 0.6088004
    if 10.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += -0.03137269 * Q.n_dr_0p2_0p4 + 0.4705904
    if Q.girth < 0.08589404:
        z += -34.73205 * Q.girth + 2.486634
    if 0.08589404 <= Q.girth < 0.09749958:
        z += 42.79351 * Q.girth - 4.172349
    if Q.D2 < 1.409617:
        z += -1.364421 * Q.D2 + 1.923311
    if Q.girth2_top10 < 0.00130722:
        z += -575.3074 * Q.girth2_top10 + 0.7520535
    if Q.N2 < 0.1732514:
        z += 12.04833 * Q.N2 - 2.08739
    if Q.log_sum_pt >= 6.893714:
        z += -4.316001 * Q.log_sum_pt + 29.75328
    if Q.psi_0p2 >= 0.995185:
        z += 112.4513 * Q.psi_0p2 - 111.9098
    if Q.z_dr_0p1_0p2 < 0.1548383:
        z += 1.676814 * Q.z_dr_0p1_0p2 - 0.2596351
    if Q.z_dr_0p05_0p1 >= 0.8509215:
        z += 9.954556 * Q.z_dr_0p05_0p1 - 8.470546
    if Q.n_dr_0p1_0p2 < 9.0:
        z += -0.008997012 * Q.n_dr_0p1_0p2 + 0.4063427
    if 9.0 <= Q.n_dr_0p1_0p2 < 19.0:
        z += -0.03253696 * Q.n_dr_0p1_0p2 + 0.6182023
    if Q.girth2_top15 < 0.006142802:
        z += 318.1778 * Q.girth2_top15 - 1.689434
    if 0.006142802 <= Q.girth2_top15 < 0.007887677:
        z += -151.9132 * Q.girth2_top15 + 1.198243
    if Q.LHA < 0.3098384:
        z += 13.2399 * Q.LHA - 3.66769
    if 0.3098384 <= Q.LHA < 0.3332345:
        z += -18.57315 * Q.LHA + 6.189217
    if Q.e2 < 0.02515919:
        z += -94.21617 * Q.e2 + 1.969318
    if 0.02515919 <= Q.e2 < 0.04082832:
        z += 29.71172 * Q.e2 - 1.148607
    if 0.04082832 <= Q.e2 < 0.04358622:
        z += -23.37729 * Q.e2 + 1.018928
    if Q.e3 < 3.376709e-05:
        z += 13895.28 * Q.e3 - 0.4692032
    if Q.sum_pt >= 986.0565:
        z += -0.002002571 * Q.sum_pt + 1.974648
    if Q.psi_0p1 >= 0.6635952:
        z += -1.257918 * Q.psi_0p1 + 0.8347483
    if Q.dr_0 < 0.09338587:
        z += -3.22392 * Q.dr_0 + 0.3010685
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += -11.45858 * Q.z_dr_0p2_0p4 + 0.7848151
    if Q.sj3_dr_max < 0.1623049:
        z += 8.511439 * Q.sj3_dr_max - 1.06829
    if 0.1623049 <= Q.sj3_dr_max < 0.1894436:
        z += -11.53913 * Q.sj3_dr_max + 2.186015
    if Q.psi_0p3 >= 0.9896594:
        z += -30.86817 * Q.psi_0p3 + 30.54898
    if Q.z_dr_0_0p05 >= 0.8103116:
        z += 7.876709 * Q.z_dr_0_0p05 - 6.382588
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2_top10 < 0.005337976:
        z += -24.82054 * (10.0 - Q.n_dr_0p2_0p4) * (0.005337976 - Q.girth2_top10)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 33.0:
        z += 0.002903527 * (10.0 - Q.n_dr_0p2_0p4) * (33.0 - Q.n_dr_0p1_0p2)
    if Q.D2 < 1.409617 and Q.z_5 > 0.03693777:
        z += -15.03551 * (1.409617 - Q.D2) * (Q.z_5 - 0.03693777)
    if Q.D2 < 1.409617 and Q.n_real_top50 > 22.0:
        z += -0.03130336 * (1.409617 - Q.D2) * (Q.n_real_top50 - 22.0)
    if Q.girth < 0.09749958 and Q.tau4 > 0.008810529:
        z += -185.2546 * (0.09749958 - Q.girth) * (Q.tau4 - 0.008810529)
    if Q.girth < 0.09749958 and Q.pt1_dr01 > 12.6865:
        z += -0.8050404 * (0.09749958 - Q.girth) * (Q.pt1_dr01 - 12.6865)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.dr0_7 > 0.1600212:
        z += -0.4904 * (10.0 - Q.n_dr_0p2_0p4) * (Q.dr0_7 - 0.1600212)
    if Q.D2 < 1.409617 and Q.z_7 < 0.04963857:
        z += -23.56148 * (1.409617 - Q.D2) * (0.04963857 - Q.z_7)
    if Q.girth < 0.08589404 and Q.n_dr_0p4_up > 0.0:
        z += -3.02908 * (0.08589404 - Q.girth) * (Q.n_dr_0p4_up - 0.0)
    if Q.n_dr_0p1_0p2 < 19.0 and Q.planar_flow < 0.6025827:
        z += 0.04813365 * (19.0 - Q.n_dr_0p1_0p2) * (0.6025827 - Q.planar_flow)
    if Q.log_sum_pt > 6.893714 and Q.z_top50_slots > 0.9704436:
        z += 39.88171 * (Q.log_sum_pt - 6.893714) * (Q.z_top50_slots - 0.9704436)
    if Q.log_sum_pt > 6.893714 and Q.sd_zg < 0.3213081:
        z += 23.11483 * (Q.log_sum_pt - 6.893714) * (0.3213081 - Q.sd_zg)
    if Q.D2 < 1.409617 and Q.soft10_dr < 0.07954628:
        z += -3.662332 * (1.409617 - Q.D2) * (0.07954628 - Q.soft10_dr)
    if Q.D2 < 1.409617 and Q.dr0_11 < 0.2191044:
        z += -1.641916 * (1.409617 - Q.D2) * (0.2191044 - Q.dr0_11)
    if Q.psi_0p2 > 0.995185 and Q.dr_10 < 0.03638736:
        z += -3394.597 * (Q.psi_0p2 - 0.995185) * (0.03638736 - Q.dr_10)
    if Q.D2 < 1.409617 and Q.dr_10 < 0.09223087:
        z += -7.305004 * (1.409617 - Q.D2) * (0.09223087 - Q.dr_10)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.206579
    if Q.girth < 0.05048381:
        z += -126.513 * Q.girth + 7.23472
    if 0.05048381 <= Q.girth < 0.06171014:
        z += -75.52419 * Q.girth + 4.660608
    if Q.sj3_dr_max < 0.1210264:
        z += 15.66665 * Q.sj3_dr_max - 1.514917
    if 0.1210264 <= Q.sj3_dr_max < 0.1506299:
        z += 10.84983 * Q.sj3_dr_max - 0.9319548
    if 0.1506299 <= Q.sj3_dr_max < 0.1999777:
        z += -8.626123 * Q.sj3_dr_max + 2.001705
    if 0.1999777 <= Q.sj3_dr_max < 0.2628766:
        z += -4.398693 * Q.sj3_dr_max + 1.156314
    if Q.psi_0p3 >= 0.9896594:
        z += -114.2036 * Q.psi_0p3 + 113.0227
    if Q.D2 < 4.450169:
        z += -0.1238009 * Q.D2 + 0.5509349
    if Q.LHA < 0.1870291:
        z += 40.03247 * Q.LHA - 8.904239
    if 0.1870291 <= Q.LHA < 0.2284021:
        z += 28.59186 * Q.LHA - 6.764512
    if 0.2284021 <= Q.LHA < 0.2454112:
        z += 13.76145 * Q.LHA - 3.377214
    if 0.3719813 <= Q.LHA < 0.404204:
        z += 23.40651 * Q.LHA - 8.706785
    if Q.LHA >= 0.404204:
        z += -9.574905 * Q.LHA + 4.624437
    if Q.psi_0p2 >= 0.9734513:
        z += -4.862638 * Q.psi_0p2 + 4.733542
    if Q.log_sum_pt >= 6.811175:
        z += -2.408982 * Q.log_sum_pt + 16.408
    if Q.n_dr_0_0p05 >= 20.0:
        z += 0.01153222 * Q.n_dr_0_0p05 - 0.2306444
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.0476977 * Q.n_dr_0p2_0p4 + 0.476977
    if Q.sj3_dr12 < 0.1048824:
        z += 3.078689 * Q.sj3_dr12 - 0.1464438
    if 0.1048824 <= Q.sj3_dr12 < 0.1840219:
        z += -2.229692 * Q.sj3_dr12 + 0.410312
    if Q.mean_eta2 < 0.005850286:
        z += -114.945 * Q.mean_eta2 + 0.6724613
    if Q.sum_pt < 1167.447:
        z += -0.0004381577 * Q.sum_pt + 0.5115257
    if Q.eccentricity >= 0.8903081:
        z += -5.583206 * Q.eccentricity + 4.970773
    if Q.girth2_top3 >= 0.02353672:
        z += -15.98811 * Q.girth2_top3 + 0.3763076
    if 0.03029714 <= Q.e2 < 0.06524004:
        z += -31.85455 * Q.e2 + 0.965102
    if Q.e2 >= 0.06524004:
        z += 10.45188 * Q.e2 - 1.794971
    if Q.girth2_top10 < 0.01414829:
        z += 31.0705 * Q.girth2_top10 - 0.4395943
    if Q.sum_pt_top20 < 846.1934:
        z += 0.004694649 * Q.sum_pt_top20 - 3.97258
    if Q.girth < 0.05048381 and Q.log_sum_pt > 6.811175:
        z += 229.2358 * (0.05048381 - Q.girth) * (Q.log_sum_pt - 6.811175)
    if Q.sj3_dr_max < 0.2628766 and Q.girth2_top10 > 0.004752876:
        z += -13293.17 * (0.2628766 - Q.sj3_dr_max) * (Q.girth2_top10 - 0.004752876)
    if Q.psi_0p3 > 0.9896594 and Q.girth2_top15 < 0.006615185:
        z += 31393.12 * (Q.psi_0p3 - 0.9896594) * (0.006615185 - Q.girth2_top15)
    if Q.psi_0p3 > 0.9896594 and Q.log_sum_pt < 7.139296:
        z += 151.7378 * (Q.psi_0p3 - 0.9896594) * (7.139296 - Q.log_sum_pt)
    if Q.LHA > 0.3719813 and Q.pt_5 > 48.28125:
        z += 1.233065 * (Q.LHA - 0.3719813) * (Q.pt_5 - 48.28125)
    if Q.LHA > 0.3719813 and Q.z_5 > 0.04812833:
        z += -1070.5 * (Q.LHA - 0.3719813) * (Q.z_5 - 0.04812833)
    if Q.psi_0p3 > 0.9896594 and Q.z_1st > 0.1586697:
        z += 92.1587 * (Q.psi_0p3 - 0.9896594) * (Q.z_1st - 0.1586697)
    if Q.LHA > 0.3719813 and Q.lam2 < 0.003687605:
        z += 6481.038 * (Q.LHA - 0.3719813) * (0.003687605 - Q.lam2)
    if Q.LHA > 0.3719813 and Q.log_sum_pt > 6.856375:
        z += 155.5704 * (Q.LHA - 0.3719813) * (Q.log_sum_pt - 6.856375)
    if Q.psi_0p2 > 0.9734513 and Q.n_dr_0p1_0p2 < 21.0:
        z += 2.558126 * (Q.psi_0p2 - 0.9734513) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.psi_0p2 > 0.9734513 and Q.n_real_top40 < 40.0:
        z += -0.7102374 * (Q.psi_0p2 - 0.9734513) * (40.0 - Q.n_real_top40)
    if Q.psi_0p3 > 0.9896594 and Q.sd_zg > 0.1961626:
        z += 63.91639 * (Q.psi_0p3 - 0.9896594) * (Q.sd_zg - 0.1961626)
    if Q.sj3_dr_max < 0.2628766 and Q.pt_10 > 11.74219:
        z += -0.04709145 * (0.2628766 - Q.sj3_dr_max) * (Q.pt_10 - 11.74219)
    if Q.log_sum_pt > 6.811175 and Q.mean_eta2 < 0.01204531:
        z += -691.2561 * (Q.log_sum_pt - 6.811175) * (0.01204531 - Q.mean_eta2)
    if Q.sj3_dr_max < 0.2628766 and Q.eccentricity > 0.5245966:
        z += 17.79763 * (0.2628766 - Q.sj3_dr_max) * (Q.eccentricity - 0.5245966)
    if Q.LHA > 0.404204 and Q.C2_b2 < 0.0329485:
        z += 1156.171 * (Q.LHA - 0.404204) * (0.0329485 - Q.C2_b2)
    if Q.girth < 0.06171014 and Q.max_dr > 0.316623:
        z += 22.3765 * (0.06171014 - Q.girth) * (Q.max_dr - 0.316623)
    if Q.n_dr_0_0p05 > 20.0 and Q.soft4_pt < 3.259863:
        z += -0.007413333 * (Q.n_dr_0_0p05 - 20.0) * (3.259863 - Q.soft4_pt)
    if Q.LHA > 0.404204 and Q.z_dr_0p4_up > 0.0:
        z += 637.1661 * (Q.LHA - 0.404204) * (Q.z_dr_0p4_up - 0.0)
    if Q.pt_13 > 28.20312 and Q.eta_9 > 0.1269531:
        z += -7.039539 * (Q.pt_13 - 28.20312) * (Q.eta_9 - 0.1269531)
    if Q.ptdr0_6 > 11.6057 and Q.soft4_z < 0.0003690273:
        z += -10888.43 * (Q.ptdr0_6 - 11.6057) * (0.0003690273 - Q.soft4_z)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.142498
    if Q.sum_pt < 1085.125:
        z += -0.1907772 * Q.sum_pt + 207.0171
    if Q.sum_pt >= 1167.447:
        z += -0.01177806 * Q.sum_pt + 13.75026
    if 0.005383629 <= Q.girth2_top15 < 0.009962397:
        z += -0.9731922 * Q.girth2_top15 + 0.005239306
    if Q.girth2_top15 >= 0.009962397:
        z += -79.8412 * Q.girth2_top15 + 0.7909537
    if Q.n_pt_above_5 < 23.0:
        z += 0.07381742 * Q.n_pt_above_5 - 1.697801
    if Q.n_dr_0p1_0p2 < 21.0:
        z += 0.02250001 * Q.n_dr_0p1_0p2 - 0.5850002
    if 21.0 <= Q.n_dr_0p1_0p2 < 26.0:
        z += 0.04732985 * Q.n_dr_0p1_0p2 - 1.106427
    if Q.n_dr_0p1_0p2 >= 26.0:
        z += 0.02482985 * Q.n_dr_0p1_0p2 - 0.5214268
    if Q.log_sum_pt < 6.98945:
        z += 212.2177 * Q.log_sum_pt - 1483.285
    if Q.log_sum_pt >= 7.139296:
        z += -6.376088 * Q.log_sum_pt + 45.52078
    if 0.08589404 <= Q.girth < 0.09749958:
        z += -56.53488 * Q.girth + 4.85601
    if 0.09749958 <= Q.girth < 0.1207452:
        z += -122.9805 * Q.girth + 11.33443
    if 0.1207452 <= Q.girth < 0.1564779:
        z += -126.0026 * Q.girth + 11.69933
    if Q.girth >= 0.1564779:
        z += -79.31297 * Q.girth + 4.39344
    if 0.3203321 <= Q.LHA < 0.4331369:
        z += 20.8804 * Q.LHA - 6.688662
    if Q.LHA >= 0.4331369:
        z += -144.8951 * Q.LHA + 65.11484
    if Q.e2 >= 0.04082832:
        z += 71.13873 * Q.e2 - 2.904475
    if Q.sum_pt_top50 < 959.0957:
        z += -0.0138682 * Q.sum_pt_top50 + 13.72485
    if 959.0957 <= Q.sum_pt_top50 < 1048.098:
        z += -0.004763096 * Q.sum_pt_top50 + 4.992192
    if Q.e3 < 0.0001841806:
        z += 1177.278 * Q.e3 + 0.9342695
    if 0.0001841806 <= Q.e3 < 0.0005178279:
        z += -3450.055 * Q.e3 + 1.786535
    if Q.psi_0p2 >= 0.9087063:
        z += -9.361577 * Q.psi_0p2 + 8.506924
    if Q.sum_pt_top15 < 708.7945:
        z += 0.004580567 * Q.sum_pt_top15 - 3.246681
    if Q.sum_pt_top30 >= 1111.245:
        z += 0.01043777 * Q.sum_pt_top30 - 11.59892
    if Q.C2 >= 0.06655881:
        z += -2.394876 * Q.C2 + 0.1594001
    if Q.tau4 >= 0.01517184:
        z += 85.17542 * Q.tau4 - 1.292268
    if Q.pt_9 < 41.4375:
        z += -0.01610139 * Q.pt_9 + 0.6672015
    if Q.z_dr_0p1_0p2 < 0.1872805:
        z += -1.652813 * Q.z_dr_0p1_0p2 + 0.3095396
    if Q.girth2_top15 > 0.009962397 and Q.sum_pt_top40 < 1225.842:
        z += 0.1396715 * (Q.girth2_top15 - 0.009962397) * (1225.842 - Q.sum_pt_top40)
    if Q.n_pt_above_5 < 23.0 and Q.D2 < 3.345339:
        z += 0.02193531 * (23.0 - Q.n_pt_above_5) * (3.345339 - Q.D2)
    if Q.sum_pt < 1085.125 and Q.log_sum_pt < 6.811175:
        z += 0.09896322 * (1085.125 - Q.sum_pt) * (6.811175 - Q.log_sum_pt)
    if Q.n_dr_0p1_0p2 < 26.0 and Q.sum_pt_top30 < 1027.303:
        z += -3.465331e-05 * (26.0 - Q.n_dr_0p1_0p2) * (1027.303 - Q.sum_pt_top30)
    if Q.girth > 0.09749958 and Q.sum_pt < 1167.447:
        z += 0.1296017 * (Q.girth - 0.09749958) * (1167.447 - Q.sum_pt)
    if Q.log_sum_pt > 7.139296 and Q.psi_0p3 > 0.9777125:
        z += 249.3018 * (Q.log_sum_pt - 7.139296) * (Q.psi_0p3 - 0.9777125)
    if Q.e3 < 0.0005178279 and Q.psi_0p3 < 0.9966167:
        z += -6523.88 * (0.0005178279 - Q.e3) * (0.9966167 - Q.psi_0p3)
    if Q.sum_pt < 1085.125 and Q.max_dr < 0.3437357:
        z += 0.02656548 * (1085.125 - Q.sum_pt) * (0.3437357 - Q.max_dr)
    if Q.sum_pt_top50 < 1048.098 and Q.girth2_top2 < 0.0003125151:
        z += 19.78239 * (1048.098 - Q.sum_pt_top50) * (0.0003125151 - Q.girth2_top2)
    if Q.girth > 0.1402186 and Q.soft6_pt > 4.250195:
        z += 54.29271 * (Q.girth - 0.1402186) * (Q.soft6_pt - 4.250195)
    if Q.e3 < 0.0001841806 and Q.pt_9 < 41.4375:
        z += -79.7141 * (0.0001841806 - Q.e3) * (41.4375 - Q.pt_9)
    if Q.log_sum_pt > 7.139296 and Q.dr_13 < 0.01134873:
        z += 1386.575 * (Q.log_sum_pt - 7.139296) * (0.01134873 - Q.dr_13)
    if Q.log_sum_pt > 7.139296 and Q.absphi_13 > 0.01228943:
        z += -24.08487 * (Q.log_sum_pt - 7.139296) * (Q.absphi_13 - 0.01228943)
    if Q.log_sum_pt > 7.139296 and Q.phi_0 < -0.07818909:
        z += 453.5573 * (Q.log_sum_pt - 7.139296) * (-0.07818909 - Q.phi_0)
    if Q.log_sum_pt > 7.139296 and Q.phi_0 < -0.0297699:
        z += -63.3692 * (Q.log_sum_pt - 7.139296) * (-0.0297699 - Q.phi_0)
    if Q.log_sum_pt > 7.139296 and Q.phi_0 < -0.04013062:
        z += -425.5601 * (Q.log_sum_pt - 7.139296) * (-0.04013062 - Q.phi_0)
    if Q.log_sum_pt > 7.139296 and Q.phi_0 < -0.004917145:
        z += 138.8521 * (Q.log_sum_pt - 7.139296) * (-0.004917145 - Q.phi_0)
    if Q.tau4 > 0.01517184 and Q.dr0_5 < 0.02529394:
        z += 924.1656 * (Q.tau4 - 0.01517184) * (0.02529394 - Q.dr0_5)
    return max(0.0, z)


def neuron_14(Q):
    z = 2.31609
    if Q.tau21_b2 < 0.1219401:
        z += -9.304456 * Q.tau21_b2 + 2.342649
    if 0.1219401 <= Q.tau21_b2 < 0.342495:
        z += -5.477381 * Q.tau21_b2 + 1.875975
    if Q.n_dr_0p1_0p2 < 17.0:
        z += 0.06301498 * Q.n_dr_0p1_0p2 - 1.071255
    if 0.076787 <= Q.girth < 0.08068193:
        z += -83.6095 * Q.girth + 6.420123
    if 0.08068193 <= Q.girth < 0.08589404:
        z += -141.0675 * Q.girth + 11.05595
    if 0.08589404 <= Q.girth < 0.09749958:
        z += -188.6687 * Q.girth + 15.14461
    if 0.09749958 <= Q.girth < 0.1207452:
        z += -106.772 * Q.girth + 7.159715
    if Q.girth >= 0.1207452:
        z += -64.44562 * Q.girth + 2.049003
    if Q.girth2_top10 >= 0.0001295334:
        z += 12.51881 * Q.girth2_top10 - 0.001621604
    if Q.e3 < 3.376709e-05:
        z += 57541.6 * Q.e3 - 3.035537
    if 3.376709e-05 <= Q.e3 < 7.876005e-05:
        z += 24282.12 * Q.e3 - 1.912461
    if Q.psi_0p3 >= 0.9924477:
        z += 32.33979 * Q.psi_0p3 - 32.09555
    if Q.e2 < 0.03480688:
        z += -35.93167 * Q.e2 + 1.250669
    if Q.tau1 < 0.1507173:
        z += 31.99697 * Q.tau1 - 4.986588
    if 0.1507173 <= Q.tau1 < 0.1751567:
        z += 6.714233 * Q.tau1 - 1.176043
    if Q.z_dr_0_0p05 >= 0.6283153:
        z += 1.403018 * Q.z_dr_0_0p05 - 0.8815379
    if Q.z_dr_0p2_0p4 < 0.003398536:
        z += 52.47982 * Q.z_dr_0p2_0p4 - 0.1783546
    if Q.soft9_z < 0.001838217:
        z += 355.4388 * Q.soft9_z - 0.6533736
    if Q.dr_6 < 0.05347848:
        z += -9.084477 * Q.dr_6 + 0.4858241
    if Q.z_dr_0p05_0p1 >= 0.7108211:
        z += -1.536214 * Q.z_dr_0p05_0p1 + 1.091973
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -7.592746 * Q.z_dr_0p1_0p2 + 0.9137389
    if Q.LHA < 0.2941033:
        z += 3.814167 * Q.LHA - 2.110036
    if 0.2941033 <= Q.LHA < 0.302389:
        z += 7.693911 * Q.LHA - 3.251081
    if 0.302389 <= Q.LHA < 0.3332345:
        z += 29.97281 * Q.LHA - 9.987975
    if Q.sd_rg < 0.15159:
        z += -5.603296 * Q.sd_rg + 1.690596
    if 0.15159 <= Q.sd_rg < 0.1596365:
        z += 1.269312 * Q.sd_rg + 0.6487773
    if 0.1596365 <= Q.sd_rg < 0.2042612:
        z += 21.16655 * Q.sd_rg - 2.527548
    if 0.2042612 <= Q.sd_rg < 0.3017146:
        z += -1.171312 * Q.sd_rg + 2.035211
    if Q.sd_rg >= 0.3017146:
        z += 4.431984 * Q.sd_rg + 0.3446145
    if Q.M3 < 0.03787151:
        z += -10.86267 * Q.M3 + 0.4113857
    if Q.tau21 < 0.5100475:
        z += -0.3585626 * Q.tau21 + 0.182884
    if Q.tau21_b2 < 0.342495 and Q.girth2_top15 < 0.00727763:
        z += -2699.488 * (0.342495 - Q.tau21_b2) * (0.00727763 - Q.girth2_top15)
    if Q.tau21_b2 < 0.342495 and Q.girth2_top10 < 0.01976735:
        z += 328.0217 * (0.342495 - Q.tau21_b2) * (0.01976735 - Q.girth2_top10)
    if Q.tau21_b2 < 0.342495 and Q.girth2_top15 < 0.006142802:
        z += 1482.334 * (0.342495 - Q.tau21_b2) * (0.006142802 - Q.girth2_top15)
    if Q.tau21_b2 < 0.342495 and Q.sum_pt_top50 < 1156.659:
        z += -0.03909098 * (0.342495 - Q.tau21_b2) * (1156.659 - Q.sum_pt_top50)
    if Q.psi_0p3 > 0.9924477 and Q.dr_6 < 0.05347848:
        z += -1891.242 * (Q.psi_0p3 - 0.9924477) * (0.05347848 - Q.dr_6)
    if Q.n_dr_0p1_0p2 < 17.0 and Q.pt1_dr01 < 17.82896:
        z += 0.0009461984 * (17.0 - Q.n_dr_0p1_0p2) * (17.82896 - Q.pt1_dr01)
    if Q.girth > 0.09749958 and Q.max_dr < 0.4021783:
        z += -311.5869 * (Q.girth - 0.09749958) * (0.4021783 - Q.max_dr)
    if Q.girth > 0.076787 and Q.max_dr < 0.4021783:
        z += 391.704 * (Q.girth - 0.076787) * (0.4021783 - Q.max_dr)
    if Q.e2 < 0.03480688 and Q.soft9_z < 0.001838217:
        z += 20813.1 * (0.03480688 - Q.e2) * (0.001838217 - Q.soft9_z)
    if Q.girth > 0.1207452 and Q.max_dr < 0.4021783:
        z += -467.2208 * (Q.girth - 0.1207452) * (0.4021783 - Q.max_dr)
    if Q.n_real_top50 > 43.0 and Q.tau32 > 0.577419:
        z += 0.1693935 * (Q.n_real_top50 - 43.0) * (Q.tau32 - 0.577419)
    if Q.sd_rg > 0.15159 and Q.n_pt_above_50 > 4.0:
        z += 0.8924172 * (Q.sd_rg - 0.15159) * (Q.n_pt_above_50 - 4.0)
    if Q.z_dr_0p2_0p4 < 0.003398536 and Q.soft9_z < 0.002970691:
        z += 39646.88 * (0.003398536 - Q.z_dr_0p2_0p4) * (0.002970691 - Q.soft9_z)
    if Q.psi_0p3 > 0.9638082 and Q.pt_1 < 182.125:
        z += -0.04484156 * (Q.psi_0p3 - 0.9638082) * (182.125 - Q.pt_1)
    if Q.z_dr_0p2_0p4 < 0.003398536 and Q.dr_6 < 0.04002298:
        z += 6440.727 * (0.003398536 - Q.z_dr_0p2_0p4) * (0.04002298 - Q.dr_6)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.2738857
    if Q.z_dr_0_0p05 < 0.8459004:
        z += 0.3812093 * Q.z_dr_0_0p05 - 0.3224651
    if Q.z_dr_0p1_0p2 < 0.04748396:
        z += 0.8320155 * Q.z_dr_0p1_0p2 + 0.3859225
    if 0.04748396 <= Q.z_dr_0p1_0p2 < 0.1203437:
        z += -5.839028 * Q.z_dr_0p1_0p2 + 0.7026901
    if 0.7108211 <= Q.z_dr_0p05_0p1 < 0.8509215:
        z += 1.529257 * Q.z_dr_0p05_0p1 - 1.087028
    if Q.z_dr_0p05_0p1 >= 0.8509215:
        z += -7.997617 * Q.z_dr_0p05_0p1 + 7.019594
    if Q.girth2_top5 < 0.008329695:
        z += -81.68893 * Q.girth2_top5 + 0.6804439
    if 6.915514 <= Q.log_sum_pt < 7.017258:
        z += -8.185416 * Q.log_sum_pt + 56.60636
    if Q.log_sum_pt >= 7.017258:
        z += 7.49282 * Q.log_sum_pt - 53.41186
    if Q.sum_pt >= 907.9372:
        z += 0.00264399 * Q.sum_pt - 2.400576
    if Q.psi_0p3 >= 0.9896594:
        z += 50.21476 * Q.psi_0p3 - 49.6955
    if Q.n_dr_0p2_0p4 >= 8.0:
        z += -0.03957671 * Q.n_dr_0p2_0p4 + 0.3166136
    if 0.707925 <= Q.psi_0p1 < 0.9184255:
        z += -1.000245 * Q.psi_0p1 + 0.7080985
    if Q.psi_0p1 >= 0.9184255:
        z += 9.75141 * Q.psi_0p1 - 9.166496
    if Q.girth2_top10 < 0.003213724:
        z += -1.71373 * Q.girth2_top10 + 0.235868
    if 0.003213724 <= Q.girth2_top10 < 0.007678544:
        z += -51.59459 * Q.girth2_top10 + 0.3961713
    if 886.3438 <= Q.sum_pt_top30 < 1038.262:
        z += 0.002617506 * Q.sum_pt_top30 - 2.32001
    if Q.sum_pt_top30 >= 1038.262:
        z += -0.003617231 * Q.sum_pt_top30 + 4.153279
    if 0.1512157 <= Q.sj2_dr < 0.2232169:
        z += -4.822852 * Q.sj2_dr + 0.7292908
    if 0.2232169 <= Q.sj2_dr < 0.2780918:
        z += 6.596957 * Q.sj2_dr - 1.819803
    if Q.sj2_dr >= 0.2780918:
        z += 4.092126 * Q.sj2_dr - 1.123231
    if Q.psi_0p2 >= 0.9976427:
        z += -83.26077 * Q.psi_0p2 + 83.06451
    if Q.lam2 < 0.001776308:
        z += -331.9849 * Q.lam2 + 0.5897075
    if Q.D2 < 2.410481:
        z += 0.07054921 * Q.D2 - 0.1700575
    if Q.tau1 < 0.07708632:
        z += 36.08761 * Q.tau1 - 2.781861
    if Q.sum_pt_top40 >= 1001.523:
        z += -0.004776823 * Q.sum_pt_top40 + 4.784099
    if Q.mean_phi2 < 0.002787636:
        z += -53.74629 * Q.mean_phi2 + 0.1498251
    if Q.sd_rg < 0.1881908:
        z += -3.208645 * Q.sd_rg + 0.6038374
    if Q.soft5_z < 0.0004140594:
        z += -4848.41 * Q.soft5_z + 2.00753
    if Q.girth < 0.08068193:
        z += -6.478705 * Q.girth + 0.5227144
    if Q.sj3_dr13 >= 0.3682013:
        z += -14.85888 * Q.sj3_dr13 + 5.471061
    if Q.sj3_z3 < 0.008293585:
        z += 117.4844 * Q.sj3_z3 - 0.9743666
    if Q.soft5_pt < 0.5297852:
        z += 2.409772 * Q.soft5_pt - 1.276662
    if Q.z_dr_0p2_0p4 < 0.02647293:
        z += 28.95547 * Q.z_dr_0p2_0p4 - 0.7665361
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.tau21_b2 < 0.4823776:
        z += -7.931228 * (0.1203437 - Q.z_dr_0p1_0p2) * (0.4823776 - Q.tau21_b2)
    if Q.girth2_top5 < 0.008329695 and Q.n_real_top40 < 40.0:
        z += 0.9373991 * (0.008329695 - Q.girth2_top5) * (40.0 - Q.n_real_top40)
    if Q.girth2_top5 < 0.008329695 and Q.sum_pt_top2 < 689.25:
        z += -0.09394576 * (0.008329695 - Q.girth2_top5) * (689.25 - Q.sum_pt_top2)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.C2_b2 < 0.05112769:
        z += 24.05022 * (0.1203437 - Q.z_dr_0p1_0p2) * (0.05112769 - Q.C2_b2)
    if Q.sj2_dr > 0.2232169 and Q.sj3_dr23 > 0.2563461:
        z += -18.18589 * (Q.sj2_dr - 0.2232169) * (Q.sj3_dr23 - 0.2563461)
    if Q.psi_0p3 > 0.9896594 and Q.tau21_b2 < 0.4299592:
        z += -82.61332 * (Q.psi_0p3 - 0.9896594) * (0.4299592 - Q.tau21_b2)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.soft5_pt < 1.480811:
        z += -4.899713 * (0.1203437 - Q.z_dr_0p1_0p2) * (1.480811 - Q.soft5_pt)
    if Q.psi_0p1 > 0.707925 and Q.sj3_dr13 > 0.228329:
        z += 12.49539 * (Q.psi_0p1 - 0.707925) * (Q.sj3_dr13 - 0.228329)
    if Q.sj2_dr > 0.2232169 and Q.dr_4 < 0.03006824:
        z += -265.8126 * (Q.sj2_dr - 0.2232169) * (0.03006824 - Q.dr_4)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_dr_0p4_up > 0.0:
        z += -2.43259 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p4_up - 0.0)
    if Q.psi_0p1 > 0.707925 and Q.dr1_11 > 0.2621232:
        z += 45.01256 * (Q.psi_0p1 - 0.707925) * (Q.dr1_11 - 0.2621232)
    if Q.girth < 0.08068193 and Q.dr1_11 > 0.1911348:
        z += -47.45744 * (0.08068193 - Q.girth) * (Q.dr1_11 - 0.1911348)
    if Q.psi_0p1 > 0.707925 and Q.dr1_11 > 0.2195171:
        z += -8.406321 * (Q.psi_0p1 - 0.707925) * (Q.dr1_11 - 0.2195171)
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
