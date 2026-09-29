"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
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
  Q.girth                  pT-weighted mean ΔR
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
        girth=sum(z[i] * dr[i] for i in P),
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
    z = -0.581556
    if Q.n_dr_0p2_0p4 < 18.0:
        z += 0.01061848 * Q.n_dr_0p2_0p4 - 0.1911326
    if Q.girth2_top15 < 0.005383629:
        z += 40.45284 * Q.girth2_top15 + 0.7696929
    if 0.005383629 <= Q.girth2_top15 < 0.007887677:
        z += -591.9397 * Q.girth2_top15 + 4.17426
    if 0.007887677 <= Q.girth2_top15 < 0.009962397:
        z += 238.4753 * Q.girth2_top15 - 2.375786
    if Q.tau1 < 0.04466492:
        z += 92.44395 * Q.tau1 - 4.129002
    if 0.9896594 <= Q.psi_0p3 < 0.9956185:
        z += 31.65283 * Q.psi_0p3 - 31.32552
    if Q.psi_0p3 >= 0.9956185:
        z += 14.75461 * Q.psi_0p3 - 14.50134
    if 6.893714 <= Q.log_sum_pt < 7.017258:
        z += -3.325851 * Q.log_sum_pt + 22.92746
    if 7.017258 <= Q.log_sum_pt < 7.062574:
        z += 7.841194 * Q.log_sum_pt - 55.43457
    if Q.log_sum_pt >= 7.062574:
        z += 5.973071 * Q.log_sum_pt - 42.24081
    if Q.sum_pt_top40 >= 1001.523:
        z += -0.002489821 * Q.sum_pt_top40 + 2.493613
    if Q.girth < 0.05048381:
        z += -102.9423 * Q.girth + 8.755356
    if 0.05048381 <= Q.girth < 0.08068193:
        z += -145.5711 * Q.girth + 10.90742
    if 0.08068193 <= Q.girth < 0.08589404:
        z += 48.67677 * Q.girth - 4.764874
    if 0.08589404 <= Q.girth < 0.09749958:
        z += 50.30609 * Q.girth - 4.904823
    if Q.sj3_dr_max < 0.1210264:
        z += -1.953336 * Q.sj3_dr_max + 0.03292204
    if 0.1210264 <= Q.sj3_dr_max < 0.1623049:
        z += 10.0008 * Q.sj3_dr_max - 1.413844
    if 0.1623049 <= Q.sj3_dr_max < 0.2125209:
        z += -4.168668 * Q.sj3_dr_max + 0.8859291
    if Q.sum_pt_top2 >= 405.0:
        z += 0.0006078492 * Q.sum_pt_top2 - 0.2461789
    if Q.sj2_dr < 0.06289464:
        z += 7.893274 * Q.sj2_dr - 1.440561
    if 0.06289464 <= Q.sj2_dr < 0.1411617:
        z += 2.972221 * Q.sj2_dr - 1.131053
    if 0.1411617 <= Q.sj2_dr < 0.1512157:
        z += 21.40082 * Q.sj2_dr - 3.732465
    if 0.1512157 <= Q.sj2_dr < 0.1745007:
        z += -10.40531 * Q.sj2_dr + 1.07712
    if 0.1745007 <= Q.sj2_dr < 0.1825048:
        z += 5.316092 * Q.sj2_dr - 1.666276
    if Q.sj2_dr >= 0.1825048:
        z += -2.577182 * Q.sj2_dr - 0.2257149
    if Q.dr_0 < 0.05775119:
        z += -0.5811157 * Q.dr_0 - 0.1729074
    if 0.05775119 <= Q.dr_0 < 0.06413297:
        z += 32.35266 * Q.dr_0 - 2.074872
    if Q.LHA < 0.1870291:
        z += 6.610593 * Q.LHA - 4.797608
    if 0.1870291 <= Q.LHA < 0.3098384:
        z += 33.82114 * Q.LHA - 9.886773
    if 0.3098384 <= Q.LHA < 0.3203321:
        z += -37.3148 * Q.LHA + 12.15387
    if 0.3203321 <= Q.LHA < 0.3332345:
        z += -15.55853 * Q.LHA + 5.184641
    if Q.z_dr_0p05_0p1 >= 0.8509215:
        z += 2.642221 * Q.z_dr_0p05_0p1 - 2.248323
    if Q.e3 < 2.883342e-05:
        z += 42336.9 * Q.e3 - 1.147392
    if 2.883342e-05 <= Q.e3 < 5.13841e-05:
        z += -3251.586 * Q.e3 + 0.1670799
    if Q.e2 < 0.01879315:
        z += -89.15188 * Q.e2 + 1.675444
    if Q.psi_0p1 >= 0.8747961:
        z += -1.565306 * Q.psi_0p1 + 1.369323
    if Q.lam2 < 0.0008296372:
        z += -288.3439 * Q.lam2 + 0.2392209
    if Q.sum_pt_top30 >= 1052.08:
        z += -0.003340212 * Q.sum_pt_top30 + 3.51417
    if Q.tau21_b2 >= 0.7058597:
        z += 2.311359 * Q.tau21_b2 - 1.631495
    if Q.soft5_z < 0.0005078411:
        z += -847.9985 * Q.soft5_z + 0.4306485
    if Q.soft5_pt < 0.5297852:
        z += 0.5841706 * Q.soft5_pt - 0.3094849
    if Q.n_dr_0p2_0p4 < 18.0 and Q.tau21_b2 > 0.2018786:
        z += -0.03770031 * (18.0 - Q.n_dr_0p2_0p4) * (Q.tau21_b2 - 0.2018786)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.log_sum_pt > 6.920349:
        z += -1.343071 * (18.0 - Q.n_dr_0p2_0p4) * (Q.log_sum_pt - 6.920349)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.sum_pt_top50 > 889.8503:
        z += 0.0008905188 * (18.0 - Q.n_dr_0p2_0p4) * (Q.sum_pt_top50 - 889.8503)
    if Q.psi_0p3 > 0.9956185 and Q.n_dr_0p1_0p2 < 26.0:
        z += 4.016044 * (Q.psi_0p3 - 0.9956185) * (26.0 - Q.n_dr_0p1_0p2)
    if Q.girth < 0.08589404 and Q.z_dr_0_0p05 < 0.4947602:
        z += 62.47472 * (0.08589404 - Q.girth) * (0.4947602 - Q.z_dr_0_0p05)
    if Q.sum_pt_top40 > 1001.523 and Q.sj2_dr < 0.1512157:
        z += 0.02233568 * (Q.sum_pt_top40 - 1001.523) * (0.1512157 - Q.sj2_dr)
    if Q.sum_pt_top40 > 1001.523 and Q.dr_5 < 0.02652372:
        z += 0.07398618 * (Q.sum_pt_top40 - 1001.523) * (0.02652372 - Q.dr_5)
    if Q.psi_0p3 > 0.9956185 and Q.dr_5 < 0.02652372:
        z += -2791.386 * (Q.psi_0p3 - 0.9956185) * (0.02652372 - Q.dr_5)
    if Q.psi_0p3 > 0.9956185 and Q.soft5_pt < 0.439209:
        z += -190.5673 * (Q.psi_0p3 - 0.9956185) * (0.439209 - Q.soft5_pt)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.z_top15_slots < 0.9278036:
        z += 0.06051129 * (18.0 - Q.n_dr_0p2_0p4) * (0.9278036 - Q.z_top15_slots)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.C3 < 0.0223982:
        z += -0.7988953 * (18.0 - Q.n_dr_0p2_0p4) * (0.0223982 - Q.C3)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.4728312
    if Q.n_particles >= 38.0:
        z += 0.05963255 * Q.n_particles - 2.266037
    if Q.log_sum_pt < 6.910131:
        z += 7.9005 * Q.log_sum_pt - 56.40401
    if 6.910131 <= Q.log_sum_pt < 6.98945:
        z += 43.62118 * Q.log_sum_pt - 303.2386
    if 6.98945 <= Q.log_sum_pt < 7.062574:
        z += 24.64294 * Q.log_sum_pt - 170.5911
    if 7.062574 <= Q.log_sum_pt < 7.139296:
        z += 19.48466 * Q.log_sum_pt - 134.1604
    if Q.log_sum_pt >= 7.139296:
        z += 11.58416 * Q.log_sum_pt - 77.75638
    if Q.sum_pt_top50 < 934.2416:
        z += -0.01042219 * Q.sum_pt_top50 + 11.24548
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += -0.01865043 * Q.sum_pt_top50 + 18.93264
    if 959.0957 <= Q.sum_pt_top50 < 1078.994:
        z += -0.02818078 * Q.sum_pt_top50 + 28.07316
    if Q.sum_pt_top50 >= 1078.994:
        z += -0.01775859 * Q.sum_pt_top50 + 16.82768
    if Q.girth2_top15 < 0.0007894752:
        z += -560.2599 * Q.girth2_top15 + 0.7351734
    if 0.0007894752 <= Q.girth2_top15 < 0.002197765:
        z += -207.9559 * Q.girth2_top15 + 0.4570381
    if Q.psi_0p3 >= 0.9966167:
        z += -97.77039 * Q.psi_0p3 + 97.4396
    if Q.tau21 < 0.2140276:
        z += -4.614638 * Q.tau21 + 0.9876598
    if Q.soft5_pt < 2.117188:
        z += -0.2013888 * Q.soft5_pt + 0.4263779
    if Q.C2 < 0.05602756:
        z += -10.80082 * Q.C2 + 0.6051434
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.08501699 * Q.n_dr_0p2_0p4 - 0.777229
    if 7.0 <= Q.n_dr_0p2_0p4 < 13.0:
        z += 0.03035168 * Q.n_dr_0p2_0p4 - 0.3945718
    if Q.pt_11 < 29.04688:
        z += 0.01059125 * Q.pt_11 - 0.3076428
    if Q.D2_b2 < 0.2773918:
        z += 2.065596 * Q.D2_b2 - 1.029411
    if 0.2773918 <= Q.D2_b2 < 0.8495689:
        z += 0.4680115 * Q.D2_b2 - 0.5862543
    if 0.8495689 <= Q.D2_b2 < 3.852812:
        z += 0.06281416 * Q.D2_b2 - 0.2420112
    if Q.z_dr_0_0p05 >= 0.8103116:
        z += -0.8857892 * Q.z_dr_0_0p05 + 0.7177653
    if 972.0419 <= Q.sum_pt < 1052.889:
        z += 0.02251561 * Q.sum_pt - 21.88612
    if Q.sum_pt >= 1052.889:
        z += 0.01156865 * Q.sum_pt - 10.36018
    if Q.girth2_top5 < 0.0006570502:
        z += -713.0196 * Q.girth2_top5 + 0.4684897
    if Q.sum_pt_top2 < 605.875:
        z += -0.001837618 * Q.sum_pt_top2 + 1.113367
    if 0.9341838 <= Q.z_top30_slots < 0.9564984:
        z += 3.364171 * Q.z_top30_slots - 3.142754
    if Q.z_top30_slots >= 0.9564984:
        z += 9.848928 * Q.z_top30_slots - 9.345414
    if 15.0 <= Q.n_dr_0_0p05 < 30.0:
        z += 0.01177431 * Q.n_dr_0_0p05 - 0.1766147
    if Q.n_dr_0_0p05 >= 30.0:
        z += -0.0512648 * Q.n_dr_0_0p05 + 1.714559
    if Q.N2 < 0.3572263:
        z += -1.739831 * Q.N2 + 0.6215136
    if Q.N2 >= 0.4678622:
        z += -5.784317 * Q.N2 + 2.706263
    if Q.n_dr_0p1_0p2 < 7.0:
        z += 0.06607731 * Q.n_dr_0p1_0p2 - 0.4625412
    if Q.sj3_dr23 >= 0.2799759:
        z += 1.27011 * Q.sj3_dr23 - 0.3556001
    if Q.tau1 < 0.07708632:
        z += -20.30515 * Q.tau1 + 1.56525
    if Q.tau1 >= 0.1219132:
        z += -30.16213 * Q.tau1 + 3.677164
    if Q.lam2 < 0.0005358203:
        z += 1346.644 * Q.lam2 - 1.124747
    if 0.0005358203 <= Q.lam2 < 0.001776308:
        z += 325.0238 * Q.lam2 - 0.5773425
    if Q.sum_pt_top40 < 1225.842:
        z += -0.004713345 * Q.sum_pt_top40 + 5.777818
    if Q.pt_8 < 36.65625:
        z += 0.007677462 * Q.pt_8 - 0.281427
    if Q.eccentricity >= 0.9252623:
        z += 5.869643 * Q.eccentricity - 5.430959
    if Q.n_pt_above_5 >= 46.0:
        z += -0.05304507 * Q.n_pt_above_5 + 2.440073
    if Q.girth >= 0.1564779:
        z += 18.18086 * Q.girth - 2.844903
    if Q.zdr_0 >= 0.01113024:
        z += -18.987 * Q.zdr_0 + 0.2113298
    if Q.pt_13 < 17.53125:
        z += 0.01642077 * Q.pt_13 - 0.2878766
    if Q.tau4 < 0.02178815:
        z += 28.61284 * Q.tau4 - 0.6234207
    if Q.sj3_dr_min < 0.2134181:
        z += -0.6782114 * Q.sj3_dr_min + 0.1447426
    if Q.e2 >= 0.05557149:
        z += 17.52869 * Q.e2 - 0.9740955
    if Q.LHA >= 0.3332345:
        z += 14.34822 * Q.LHA - 4.781323
    if Q.z_top50_slots < 1.0:
        z += -16.33819 * Q.z_top50_slots + 16.33819
    if Q.girth2_top3 < 0.006756161:
        z += -30.57869 * Q.girth2_top3 + 0.2065945
    if Q.D2 < 1.601009:
        z += -0.222546 * Q.D2 + 0.3562982
    if Q.z_top15_slots >= 0.9885666:
        z += 31.67958 * Q.z_top15_slots - 31.31738
    if Q.sum_pt_top30 >= 1038.262:
        z += 0.001688267 * Q.sum_pt_top30 - 1.752863
    if Q.soft1_pt < 1.521582:
        z += -0.1749173 * Q.soft1_pt + 0.2661509
    if Q.n_particles > 38.0 and Q.zdr_0 < 0.009970338:
        z += 2.136165 * (Q.n_particles - 38.0) * (0.009970338 - Q.zdr_0)
    if Q.n_particles > 38.0 and Q.soft1_z < 0.002181998:
        z += -22.36997 * (Q.n_particles - 38.0) * (0.002181998 - Q.soft1_z)
    if Q.z_top5 > 0.6551948 and Q.pt1_dr01 < 28.39396:
        z += -0.09106833 * (Q.z_top5 - 0.6551948) * (28.39396 - Q.pt1_dr01)
    if Q.girth2_top15 < 0.002197765 and Q.psi_0p3 > 0.9966167:
        z += 79944.34 * (0.002197765 - Q.girth2_top15) * (Q.psi_0p3 - 0.9966167)
    if Q.n_particles > 38.0 and Q.absphi_1 < 0.1178619:
        z += 0.08775645 * (Q.n_particles - 38.0) * (0.1178619 - Q.absphi_1)
    if Q.z_top30_slots > 0.9341838 and Q.pt1_dr01 > 5.351077:
        z += 0.3767833 * (Q.z_top30_slots - 0.9341838) * (Q.pt1_dr01 - 5.351077)
    if Q.n_particles > 38.0 and Q.tau32 > 0.3293142:
        z += 0.02148759 * (Q.n_particles - 38.0) * (Q.tau32 - 0.3293142)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_2 > 7.740999:
        z += 0.5034589 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_2 - 7.740999)
    if Q.M3 < 0.03457336 and Q.psi_0p3 > 0.9299135:
        z += -217.2168 * (0.03457336 - Q.M3) * (Q.psi_0p3 - 0.9299135)
    if Q.z_dr_0_0p05 > 0.8103116 and Q.n_dr_0p05_0p1 < 10.0:
        z += -0.6582651 * (Q.z_dr_0_0p05 - 0.8103116) * (10.0 - Q.n_dr_0p05_0p1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_4 > 2.381691:
        z += 0.3488767 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_4 - 2.381691)
    if Q.z_dr_0_0p05 > 0.8103116 and Q.n_real_top20 < 20.0:
        z += 0.6785722 * (Q.z_dr_0_0p05 - 0.8103116) * (20.0 - Q.n_real_top20)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 < 17.94219:
        z += -0.3387508 * (Q.z_top30_slots - 0.9341838) * (17.94219 - Q.ptdr0_3)
    if Q.soft5_pt < 2.117188 and Q.max_dr < 0.4357228:
        z += -0.8884543 * (2.117188 - Q.soft5_pt) * (0.4357228 - Q.max_dr)
    if Q.n_dr_0_0p05 > 15.0 and Q.eta_0 < 0.0002882481:
        z += -1.153856 * (Q.n_dr_0_0p05 - 15.0) * (0.0002882481 - Q.eta_0)
    if Q.sum_pt_top2 < 605.875 and Q.soft4_dr < 0.2664362:
        z += -0.00270215 * (605.875 - Q.sum_pt_top2) * (0.2664362 - Q.soft4_dr)
    if Q.z_top30_slots > 0.9341838 and Q.dr12 > 0.003946546:
        z += 20.28057 * (Q.z_top30_slots - 0.9341838) * (Q.dr12 - 0.003946546)
    if Q.sum_pt_top2 < 605.875 and Q.soft4_dr0 < 0.3636138:
        z += 0.001559657 * (605.875 - Q.sum_pt_top2) * (0.3636138 - Q.soft4_dr0)
    if Q.z_top5 > 0.6551948 and Q.ptdr0_5 > 0.5368514:
        z += 0.2391566 * (Q.z_top5 - 0.6551948) * (Q.ptdr0_5 - 0.5368514)
    if Q.tau21 < 0.2140276 and Q.soft8_abseta < 0.2834473:
        z += -4.50256 * (0.2140276 - Q.tau21) * (0.2834473 - Q.soft8_abseta)
    if Q.N2 < 0.3572263 and Q.pt2_over_pt0 < 0.7904923:
        z += 2.704619 * (0.3572263 - Q.N2) * (0.7904923 - Q.pt2_over_pt0)
    if Q.girth2_top3 < 0.006756161 and Q.eta_0 > -0.004464722:
        z += 982.3152 * (0.006756161 - Q.girth2_top3) * (Q.eta_0 - -0.004464722)
    if Q.soft9_pt < 2.5 and Q.dr_11 < 0.1224381:
        z += 1.321246 * (2.5 - Q.soft9_pt) * (0.1224381 - Q.dr_11)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_6 < 11.6057:
        z += -0.2193534 * (Q.z_top30_slots - 0.9341838) * (11.6057 - Q.ptdr0_6)
    if Q.pt1_dr01 < 12.6865 and Q.soft2_dr < 0.3845054:
        z += -0.1554506 * (12.6865 - Q.pt1_dr01) * (0.3845054 - Q.soft2_dr)
    if Q.pt1_dr01 < 12.6865 and Q.soft2_dr0 < 0.4086381:
        z += 0.1305363 * (12.6865 - Q.pt1_dr01) * (0.4086381 - Q.soft2_dr0)
    if Q.n_dr_0_0p05 > 15.0 and Q.eta_0 < 0.05520935:
        z += 0.5730405 * (Q.n_dr_0_0p05 - 15.0) * (0.05520935 - Q.eta_0)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.1047488
    if 6.903423 <= Q.log_sum_pt < 6.930088:
        z += 7.197186 * Q.log_sum_pt - 49.68521
    if 6.930088 <= Q.log_sum_pt < 7.017258:
        z += 14.19192 * Q.log_sum_pt - 98.15935
    if 7.017258 <= Q.log_sum_pt < 7.062574:
        z += 11.58905 * Q.log_sum_pt - 79.89436
    if Q.log_sum_pt >= 7.062574:
        z += 8.498453 * Q.log_sum_pt - 58.06676
    if Q.girth < 0.03577037:
        z += 27.17271 * Q.girth - 0.9719779
    if Q.sd_rg < 0.1596365:
        z += 0.6712685 * Q.sd_rg - 0.2025315
    if 0.1596365 <= Q.sd_rg < 0.1690338:
        z += 11.2577 * Q.sd_rg - 1.892513
    if 0.1690338 <= Q.sd_rg < 0.2639816:
        z += -7.27643 * Q.sd_rg + 1.240383
    if 0.2639816 <= Q.sd_rg < 0.3017146:
        z += 2.25533 * Q.sd_rg - 1.275827
    if Q.sd_rg >= 0.3017146:
        z += 1.584062 * Q.sd_rg - 1.073295
    if 997.0189 <= Q.sum_pt_top50 < 1048.098:
        z += -0.005088591 * Q.sum_pt_top50 + 5.073421
    if 1048.098 <= Q.sum_pt_top50 < 1078.994:
        z += 0.0006100517 * Q.sum_pt_top50 - 0.8993158
    if Q.sum_pt_top50 >= 1078.994:
        z += -0.002587157 * Q.sum_pt_top50 + 2.550453
    if 1052.889 <= Q.sum_pt < 1085.125:
        z += -0.007470148 * Q.sum_pt + 7.86524
    if Q.sum_pt >= 1085.125:
        z += -0.009310387 * Q.sum_pt + 9.862129
    if 985.0781 <= Q.sum_pt_top15 < 1082.548:
        z += -0.001989146 * Q.sum_pt_top15 + 1.959464
    if Q.sum_pt_top15 >= 1082.548:
        z += -0.0006624486 * Q.sum_pt_top15 + 0.5232511
    if Q.LHA < 0.1870291:
        z += -3.526819 * Q.LHA + 0.6596178
    if Q.sum_pt_top20 >= 1064.139:
        z += 0.002715731 * Q.sum_pt_top20 - 2.889915
    if Q.psi_0p3 >= 0.9966167:
        z += -53.45204 * Q.psi_0p3 + 53.2712
    if 852.7492 <= Q.sum_pt_top30 < 996.8867:
        z += 0.0004106631 * Q.sum_pt_top30 - 0.3501926
    if Q.sum_pt_top30 >= 996.8867:
        z += -0.001287312 * Q.sum_pt_top30 + 1.342496
    if 1018.698 <= Q.sum_pt_top40 < 1069.671:
        z += -0.001207549 * Q.sum_pt_top40 + 1.230127
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.004173761 * Q.sum_pt_top40 - 4.526105
    if Q.z_top50_slots >= 0.9586536:
        z += -12.27409 * Q.z_top50_slots + 11.76661
    if Q.z_top40_slots >= 0.9300465:
        z += 2.727149 * Q.z_top40_slots - 2.536375
    if Q.z_top15_slots >= 0.6225177:
        z += 0.9188083 * Q.z_top15_slots - 0.5719744
    if Q.girth2_top15 < 0.006615185:
        z += 37.76891 * Q.girth2_top15 - 0.1316065
    if 0.006615185 <= Q.girth2_top15 < 0.009962397:
        z += -35.32546 * Q.girth2_top15 + 0.3519263
    if Q.e2 < 0.02793599:
        z += -8.355919 * Q.e2 + 0.2334309
    if Q.log_sum_pt > 6.903423 and Q.girth2_top15 < 0.02146578:
        z += 219.2381 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.girth2_top15)
    if Q.log_sum_pt > 7.062574 and Q.girth2_top15 > 0.009962397:
        z += 318.6978 * (Q.log_sum_pt - 7.062574) * (Q.girth2_top15 - 0.009962397)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p1 < 0.8747961:
        z += -5.519263 * (Q.log_sum_pt - 6.903423) * (0.8747961 - Q.psi_0p1)
    if Q.sum_pt_top20 > 1129.275 and Q.z_7 > 0.02696541:
        z += -0.08957027 * (Q.sum_pt_top20 - 1129.275) * (Q.z_7 - 0.02696541)
    if Q.log_sum_pt > 6.903423 and Q.dr_max_012 < 0.1206357:
        z += -10.3872 * (Q.log_sum_pt - 6.903423) * (0.1206357 - Q.dr_max_012)
    if Q.sum_pt_top15 > 985.0781 and Q.pt_7 > 41.65625:
        z += 3.541668e-05 * (Q.sum_pt_top15 - 985.0781) * (Q.pt_7 - 41.65625)
    if Q.z_top20_slots > 0.8281581 and Q.z_top50_slots > 0.9995789:
        z += -3199.311 * (Q.z_top20_slots - 0.8281581) * (Q.z_top50_slots - 0.9995789)
    if Q.log_sum_pt > 6.930088 and Q.C2 > 0.06655881:
        z += -174.5528 * (Q.log_sum_pt - 6.930088) * (Q.C2 - 0.06655881)
    if Q.sd_rg > 0.1596365 and Q.z_dr_0p05_0p1 < 0.8509215:
        z += -8.188789 * (Q.sd_rg - 0.1596365) * (0.8509215 - Q.z_dr_0p05_0p1)
    if Q.sum_pt_top50 > 997.0189 and Q.C2 > 0.06655881:
        z += 0.1303664 * (Q.sum_pt_top50 - 997.0189) * (Q.C2 - 0.06655881)
    if Q.log_sum_pt > 6.930088 and Q.C2 > 0.1088881:
        z += 173.4888 * (Q.log_sum_pt - 6.930088) * (Q.C2 - 0.1088881)
    if Q.sum_pt_top50 > 997.0189 and Q.C2 > 0.1088881:
        z += -0.123588 * (Q.sum_pt_top50 - 997.0189) * (Q.C2 - 0.1088881)
    if Q.log_sum_pt > 6.903423 and Q.dr_max_012 < 0.2002199:
        z += 6.654264 * (Q.log_sum_pt - 6.903423) * (0.2002199 - Q.dr_max_012)
    if Q.sd_rg > 0.2639816 and Q.z_dr_0p05_0p1 < 0.5912328:
        z += -15.23247 * (Q.sd_rg - 0.2639816) * (0.5912328 - Q.z_dr_0p05_0p1)
    if Q.sd_rg > 0.1690338 and Q.z_dr_0p05_0p1 < 0.8509215:
        z += 14.98462 * (Q.sd_rg - 0.1690338) * (0.8509215 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.4457246
    if Q.lam2 < 0.0006154841:
        z += -457.1353 * Q.lam2 + 0.2813595
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.2090653 * Q.n_dr_0p2_0p4 + 1.202711
    if 5.0 <= Q.n_dr_0p2_0p4 < 8.0:
        z += -0.05246147 * Q.n_dr_0p2_0p4 + 0.4196917
    if Q.n_particles < 46.0:
        z += -0.03418949 * Q.n_particles + 1.572716
    if Q.tau21 < 0.347196:
        z += 0.8144147 * Q.tau21 - 0.2827615
    if Q.D2 < 1.788105:
        z += -0.3191723 * Q.D2 + 0.5707137
    if Q.tau4 < 0.0174227:
        z += -19.53649 * Q.tau4 + 0.3403782
    if Q.n_pt_above_1 < 43.0:
        z += 0.01292651 * Q.n_pt_above_1 - 0.55584
    if Q.e3 < 5.727594e-05:
        z += -4605.682 * Q.e3 + 0.2637947
    if Q.z_dr_0p1_0p2 < 0.08871546:
        z += 1.613641 * Q.z_dr_0p1_0p2 - 0.1431549
    if Q.n_dr_0p1_0p2 < 17.0:
        z += -0.02927719 * Q.n_dr_0p1_0p2 + 0.4977122
    if Q.N2 < 0.2989787:
        z += 3.58585 * Q.N2 - 1.072093
    if Q.D2_b2 < 1.67722:
        z += -0.0572658 * Q.D2_b2 + 0.09604737
    if Q.max_dr < 0.2738063:
        z += 4.126123 * Q.max_dr - 1.129758
    if Q.z_dr_0p2_0p4 < 0.02647293:
        z += 2.679674 * Q.z_dr_0p2_0p4 - 0.07093881
    if Q.n_particles < 46.0 and Q.sum_pt_top40 > 858.8262:
        z += 3.943762e-05 * (46.0 - Q.n_particles) * (Q.sum_pt_top40 - 858.8262)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_dr_0p1_0p2 > 9.0:
        z += -0.007511532 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 9.0)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.e2 < 0.03029714:
        z += -8.306664 * (5.0 - Q.n_dr_0p2_0p4) * (0.03029714 - Q.e2)
    if Q.n_particles < 46.0 and Q.e2 > 0.01036127:
        z += -0.4990408 * (46.0 - Q.n_particles) * (Q.e2 - 0.01036127)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.n_dr_0p05_0p1 > 2.0:
        z += -0.00144374 * (8.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p05_0p1 - 2.0)
    if Q.n_dr_0p1_0p2 < 10.0 and Q.sum_pt_top40 < 1013.042:
        z += -0.0002799967 * (10.0 - Q.n_dr_0p1_0p2) * (1013.042 - Q.sum_pt_top40)
    if Q.tau21 < 0.347196 and Q.sum_pt < 1085.125:
        z += 0.009415518 * (0.347196 - Q.tau21) * (1085.125 - Q.sum_pt)
    if Q.psi_0p3 > 0.9980008 and Q.eccentricity > 0.8680812:
        z += 767.7167 * (Q.psi_0p3 - 0.9980008) * (Q.eccentricity - 0.8680812)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.psi_0p3 > 0.9966167:
        z += 10.35345 * (8.0 - Q.n_dr_0p2_0p4) * (Q.psi_0p3 - 0.9966167)
    if Q.n_dr_0p2_0p4 < 1.0 and Q.psi_0p3 > 0.9993087:
        z += -151.1779 * (1.0 - Q.n_dr_0p2_0p4) * (Q.psi_0p3 - 0.9993087)
    if Q.n_dr_0p1_0p2 < 17.0 and Q.psi_0p3 > 0.9973959:
        z += -0.1184435 * (17.0 - Q.n_dr_0p1_0p2) * (Q.psi_0p3 - 0.9973959)
    if Q.n_particles < 46.0 and Q.psi_0p3 > 0.9777125:
        z += -0.3118304 * (46.0 - Q.n_particles) * (Q.psi_0p3 - 0.9777125)
    if Q.max_dr < 0.2738063 and Q.z_top50_slots > 0.9586536:
        z += 152.3508 * (0.2738063 - Q.max_dr) * (Q.z_top50_slots - 0.9586536)
    if Q.D2 < 1.788105 and Q.psi_0p3 > 0.9995915:
        z += -451.2774 * (1.788105 - Q.D2) * (Q.psi_0p3 - 0.9995915)
    if Q.D2_b2 < 1.67722 and Q.psi_0p3 > 0.9989733:
        z += 100.2569 * (1.67722 - Q.D2_b2) * (Q.psi_0p3 - 0.9989733)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.677804
    if Q.e2 < 0.01879315:
        z += 28.00319 * Q.e2 - 0.9458134
    if 0.01879315 <= Q.e2 < 0.03263075:
        z += 6.799862 * Q.e2 - 0.5473362
    if 0.03263075 <= Q.e2 < 0.04358622:
        z += 29.70678 * Q.e2 - 1.294806
    if Q.e2 >= 0.05557149:
        z += -16.24127 * Q.e2 + 0.9025517
    if Q.lam2 < 0.001163277:
        z += -132.2483 * Q.lam2 + 0.1538415
    if Q.n_dr_0p1_0p2 >= 11.0:
        z += -0.00989714 * Q.n_dr_0p1_0p2 + 0.1088685
    if Q.sum_pt_top40 < 1041.263:
        z += 0.001888734 * Q.sum_pt_top40 - 1.966669
    if Q.psi_0p1 >= 0.8976117:
        z += -2.274163 * Q.psi_0p1 + 2.041315
    if Q.n_particles >= 26.0:
        z += -0.02245854 * Q.n_particles + 0.5839221
    if Q.n_dr_0_0p05 >= 14.0:
        z += 0.0178061 * Q.n_dr_0_0p05 - 0.2492854
    if Q.girth2_top15 < 0.002197765:
        z += -10.00533 * Q.girth2_top15 + 0.1088575
    if 0.002197765 <= Q.girth2_top15 < 0.004855289:
        z += 90.90303 * Q.girth2_top15 - 0.1129153
    if 0.004855289 <= Q.girth2_top15 < 0.005788041:
        z += 70.81266 * Q.girth2_top15 - 0.01537078
    if 0.005788041 <= Q.girth2_top15 < 0.007887677:
        z += -20.49085 * Q.girth2_top15 + 0.5130977
    if 0.007887677 <= Q.girth2_top15 < 0.009962397:
        z += -218.8615 * Q.girth2_top15 + 2.077781
    if Q.girth2_top15 >= 0.009962397:
        z += -20.09037 * Q.girth2_top15 + 0.09754453
    if Q.girth < 0.01525414:
        z += 64.12418 * Q.girth - 3.871376
    if 0.01525414 <= Q.girth < 0.05660088:
        z += 61.80963 * Q.girth - 3.836069
    if 0.05660088 <= Q.girth < 0.07374472:
        z += -0.3727598 * Q.girth - 0.3164914
    if 0.07374472 <= Q.girth < 0.09749958:
        z += 6.46687 * Q.girth - 0.820878
    if Q.girth >= 0.09749958:
        z += 60.48542 * Q.girth - 6.087664
    if Q.girth2_top10 < 0.006876086:
        z += 29.81086 * Q.girth2_top10 - 0.2049821
    if Q.e3 < 0.0005178279:
        z += 611.6276 * Q.e3 - 0.3167178
    if Q.sum_pt_top30 < 1011.524:
        z += -0.002515638 * Q.sum_pt_top30 + 2.544628
    if 1.091797 <= Q.soft1_pt < 1.521582:
        z += 0.2937417 * Q.soft1_pt - 0.3207063
    if 1.521582 <= Q.soft1_pt < 2.275391:
        z += -0.2240591 * Q.soft1_pt + 0.4671702
    if Q.soft1_pt >= 2.275391:
        z += 0.3324384 * Q.soft1_pt - 0.7990792
    if 0.1411617 <= Q.sj2_dr < 0.1825048:
        z += 7.077582 * Q.sj2_dr - 0.9990837
    if Q.sj2_dr >= 0.1825048:
        z += -1.059108 * Q.sj2_dr + 0.4859015
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.03300069 * Q.n_dr_0p2_0p4 + 0.5940123
    if 0.06116874 <= Q.C2 < 0.0977156:
        z += -3.396072 * Q.C2 + 0.2077335
    if Q.C2 >= 0.0977156:
        z += 1.022505 * Q.C2 - 0.2240305
    if Q.LHA < 0.2454112:
        z += -11.21978 * Q.LHA + 2.75346
    if Q.LHA >= 0.3332345:
        z += -25.90688 * Q.LHA + 8.633068
    if Q.psi_0p2 >= 0.8706159:
        z += 3.197072 * Q.psi_0p2 - 2.783421
    if Q.soft9_pt >= 1.458984:
        z += 0.1558265 * Q.soft9_pt - 0.2273484
    if Q.soft9_z >= 0.001545795:
        z += -142.2371 * Q.soft9_z + 0.2198694
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += 7.604703 * Q.z_dr_0p2_0p4 - 0.5208575
    if Q.sum_pt_top50 < 1038.855:
        z += 0.002762249 * Q.sum_pt_top50 - 2.869576
    if Q.n_pt_above_1 >= 32.0:
        z += -0.003447305 * Q.n_pt_above_1 + 0.1103138
    if Q.n_particles > 26.0 and Q.tau21 > 0.1295048:
        z += 0.005365159 * (Q.n_particles - 26.0) * (Q.tau21 - 0.1295048)
    if Q.n_particles > 26.0 and Q.soft1_pt > 0.4909668:
        z += -0.007955357 * (Q.n_particles - 26.0) * (Q.soft1_pt - 0.4909668)
    if Q.n_particles > 26.0 and Q.lam2 > 0.003687605:
        z += 0.9922793 * (Q.n_particles - 26.0) * (Q.lam2 - 0.003687605)
    if Q.pt_entropy < 3.355186 and Q.zdr_0 > 0.0008718296:
        z += -8.211488 * (3.355186 - Q.pt_entropy) * (Q.zdr_0 - 0.0008718296)
    if Q.girth < 0.05660088 and Q.sum_pt_top40 < 1007.44:
        z += -0.3111232 * (0.05660088 - Q.girth) * (1007.44 - Q.sum_pt_top40)
    if Q.girth2_top15 < 0.007887677 and Q.sum_pt_top40 > 1095.686:
        z += 0.1237537 * (0.007887677 - Q.girth2_top15) * (Q.sum_pt_top40 - 1095.686)
    if Q.soft1_pt > 2.275391 and Q.max_dr < 0.4021783:
        z += -1.170678 * (Q.soft1_pt - 2.275391) * (0.4021783 - Q.max_dr)
    if Q.girth < 0.09749958 and Q.psi_0p3 > 0.9973959:
        z += 5306.933 * (0.09749958 - Q.girth) * (Q.psi_0p3 - 0.9973959)
    if Q.girth < 0.05660088 and Q.psi_0p3 > 0.9973959:
        z += -9424.149 * (0.05660088 - Q.girth) * (Q.psi_0p3 - 0.9973959)
    if Q.girth2_top15 < 0.004169954 and Q.psi_0p3 > 0.9973959:
        z += -28661.4 * (0.004169954 - Q.girth2_top15) * (Q.psi_0p3 - 0.9973959)
    if Q.z_dr_0p2_0p4 < 0.05180474 and Q.soft9_z > 0.0007402181:
        z += -585.5272 * (0.05180474 - Q.z_dr_0p2_0p4) * (Q.soft9_z - 0.0007402181)
    if Q.girth2_top15 > 0.004855289 and Q.sj3_pairmin_over_m < 0.4090302:
        z += -78.589 * (Q.girth2_top15 - 0.004855289) * (0.4090302 - Q.sj3_pairmin_over_m)
    if Q.z_dr_0p1_0p2 < 0.04748396 and Q.zdr_1 < 0.01003298:
        z += 610.8495 * (0.04748396 - Q.z_dr_0p1_0p2) * (0.01003298 - Q.zdr_1)
    return max(0.0, z)


def neuron_5(Q):
    z = -4.601425
    z += -0.03591396 * Q.n_particles + 2.298494
    if Q.sum_pt < 907.9372:
        z += 0.007705797 * Q.sum_pt - 8.113528
    if 907.9372 <= Q.sum_pt < 1002.379:
        z += 0.0156022 * Q.sum_pt - 15.28297
    if 1002.379 <= Q.sum_pt < 1085.125:
        z += -0.004306464 * Q.sum_pt + 4.673051
    if Q.log_sum_pt < 6.893714:
        z += 2.564151 * Q.log_sum_pt - 16.64973
    if 6.893714 <= Q.log_sum_pt < 6.903423:
        z += -15.05388 * Q.log_sum_pt + 104.8039
    if 6.903423 <= Q.log_sum_pt < 7.139296:
        z += -3.733485 * Q.log_sum_pt + 26.65446
    if Q.e3 < 0.0005178279:
        z += -1056.533 * Q.e3 + 0.5471022
    if Q.sum_pt_top50 < 934.2416:
        z += -0.01739589 * Q.sum_pt_top50 + 16.2815
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += -0.007281725 * Q.sum_pt_top50 + 6.832424
    if 959.0957 <= Q.sum_pt_top50 < 1078.994:
        z += 0.001263127 * Q.sum_pt_top50 - 1.362907
    if Q.z_top30_slots >= 0.9460751:
        z += -4.549048 * Q.z_top30_slots + 4.303741
    z += 0.5186594 * Q.z_top3_slots
    if Q.psi_0p3 >= 0.9973959:
        z += 63.60256 * Q.psi_0p3 - 63.43693
    if Q.sum_pt_top15 < 708.7945:
        z += 0.0004567127 * Q.sum_pt_top15 - 0.3237155
    if 967.7705 <= Q.sum_pt_top15 < 1003.329:
        z += -0.003148181 * Q.sum_pt_top15 + 3.046716
    if Q.sum_pt_top15 >= 1003.329:
        z += -0.002416493 * Q.sum_pt_top15 + 2.312593
    if Q.girth < 0.02783745:
        z += -20.94093 * Q.girth + 5.376292
    if 0.02783745 <= Q.girth < 0.08068193:
        z += 3.004919 * Q.girth + 4.7097
    if 0.08068193 <= Q.girth < 0.1564779:
        z += -24.69445 * Q.girth + 6.944539
    if Q.girth >= 0.1564779:
        z += 23.94585 * Q.girth - 0.6665912
    if 0.1632346 <= Q.LHA < 0.302389:
        z += -5.732914 * Q.LHA + 0.9358102
    if 0.302389 <= Q.LHA < 0.4331369:
        z += 5.860618 * Q.LHA - 2.569947
    if Q.LHA >= 0.4331369:
        z += -58.58099 * Q.LHA + 25.34209
    if Q.sum_pt_top5 >= 430.75:
        z += 0.0001778247 * Q.sum_pt_top5 - 0.07659798
    if 0.181086 <= Q.z_dr_0_0p05 < 0.7128619:
        z += 0.2995998 * Q.z_dr_0_0p05 - 0.05425332
    if 0.7128619 <= Q.z_dr_0_0p05 < 0.878906:
        z += 0.9971853 * Q.z_dr_0_0p05 - 0.5515355
    if Q.z_dr_0_0p05 >= 0.878906:
        z += 4.27981 * Q.z_dr_0_0p05 - 3.436654
    if Q.tau1 < 0.1072713:
        z += 15.81687 * Q.tau1 - 1.696696
    if Q.girth2_top15 < 0.0007894752:
        z += 395.1054 * Q.girth2_top15 - 0.311926
    if Q.n_pt_above_1 < 58.0:
        z += 0.009662256 * Q.n_pt_above_1 - 0.5604108
    if Q.tau4 < 0.01626937:
        z += -13.23624 * Q.tau4 + 0.2153453
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.02104613 * Q.n_dr_0p2_0p4 + 0.2315074
    if Q.z_dr_0p05_0p1 < 0.05954375:
        z += 2.153458 * Q.z_dr_0p05_0p1 - 0.128225
    if Q.z_top50_slots < 0.985099:
        z += 20.98617 * Q.z_top50_slots - 20.67345
    if Q.z_dr_0p2_0p4 < 0.05180474:
        z += 5.257302 * Q.z_dr_0p2_0p4 - 0.2723532
    if Q.sum_pt_top30 < 911.9328:
        z += -0.001252722 * Q.sum_pt_top30 + 1.142399
    if Q.z_8 < 0.02476077:
        z += -8.628543 * Q.z_8 + 0.2136494
    if Q.n_dr_0_0p05 < 8.0:
        z += 0.02743072 * Q.n_dr_0_0p05 - 0.2194458
    if 0.00130722 <= Q.girth2_top10 < 0.005337976:
        z += -40.37779 * Q.girth2_top10 + 0.05278266
    if 0.005337976 <= Q.girth2_top10 < 0.008956554:
        z += 47.24752 * Q.girth2_top10 - 0.4149591
    if Q.girth2_top10 >= 0.008956554:
        z += -8.014889 * Q.girth2_top10 + 0.08000155
    if Q.M2 >= 0.05233747:
        z += 2.896457 * Q.M2 - 0.1515932
    if Q.z_top20_slots >= 0.9358352:
        z += -2.961073 * Q.z_top20_slots + 2.771076
    if Q.sum_pt_top10 >= 822.975:
        z += 0.0009331659 * Q.sum_pt_top10 - 0.7679722
    if Q.n_dr_0p1_0p2 >= 8.0:
        z += 0.02562385 * Q.n_dr_0p1_0p2 - 0.2049908
    if Q.n_dr_0p05_0p1 >= 6.0:
        z += 0.01128793 * Q.n_dr_0p05_0p1 - 0.06772757
    if Q.z_top40_slots >= 0.9457345:
        z += -2.440883 * Q.z_top40_slots + 2.308427
    if Q.zdr_0 >= 0.005911134:
        z += 6.799144 * Q.zdr_0 - 0.04019065
    if Q.n_particles < 64.0 and Q.C2 < 0.07996447:
        z += -0.1540172 * (64.0 - Q.n_particles) * (0.07996447 - Q.C2)
    if Q.e3 < 0.0005178279 and Q.max_dr > 0.1939977:
        z += -916.027 * (0.0005178279 - Q.e3) * (Q.max_dr - 0.1939977)
    if Q.sum_pt < 1002.379 and Q.dr_11 < 0.199671:
        z += 0.009466621 * (1002.379 - Q.sum_pt) * (0.199671 - Q.dr_11)
    if Q.girth < 0.1564779 and Q.psi_0p3 < 0.9638082:
        z += -104.7017 * (0.1564779 - Q.girth) * (0.9638082 - Q.psi_0p3)
    if Q.sum_pt_top50 < 934.2416 and Q.soft6_dr < 0.03626613:
        z += -0.4288839 * (934.2416 - Q.sum_pt_top50) * (0.03626613 - Q.soft6_dr)
    if Q.sum_pt < 1002.379 and Q.soft6_dr < 0.03626613:
        z += 0.233024 * (1002.379 - Q.sum_pt) * (0.03626613 - Q.soft6_dr)
    if Q.sum_pt < 1002.379 and Q.sj3_dr23 > 0.1834565:
        z += 0.002110912 * (1002.379 - Q.sum_pt) * (Q.sj3_dr23 - 0.1834565)
    if Q.sum_pt < 1002.379 and Q.absphi_1 > 0.02227783:
        z += -0.08941386 * (1002.379 - Q.sum_pt) * (Q.absphi_1 - 0.02227783)
    if Q.sum_pt < 1002.379 and Q.ptdr0_12 > 2.024972:
        z += -0.001117388 * (1002.379 - Q.sum_pt) * (Q.ptdr0_12 - 2.024972)
    if Q.sum_pt < 1002.379 and Q.e4 < 5.8505e-08:
        z += -258491.2 * (1002.379 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.sum_pt < 1085.125 and Q.e4 < 5.8505e-08:
        z += 37737.27 * (1085.125 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.sum_pt < 907.9372 and Q.dr_11 < 0.2535773:
        z += -0.02810147 * (907.9372 - Q.sum_pt) * (0.2535773 - Q.dr_11)
    if Q.sum_pt_top50 < 934.2416 and Q.e4 > 3.53e-10:
        z += -427704.5 * (934.2416 - Q.sum_pt_top50) * (Q.e4 - 3.53e-10)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.2284799
    if Q.tau1 < 0.05444509:
        z += -20.48702 * Q.tau1 + 1.115418
    if 15.0 <= Q.n_dr_0p1_0p2 < 26.0:
        z += -0.01220623 * Q.n_dr_0p1_0p2 + 0.1830934
    if Q.n_dr_0p1_0p2 >= 26.0:
        z += 0.007700787 * Q.n_dr_0p1_0p2 - 0.334489
    if Q.psi_0p2 >= 0.9087063:
        z += -5.44379 * Q.psi_0p2 + 4.946807
    if Q.z_dr_0p05_0p1 < 0.2126484:
        z += 2.162356 * Q.z_dr_0p05_0p1 - 0.8225885
    if 0.2126484 <= Q.z_dr_0p05_0p1 < 0.7108211:
        z += 0.7281953 * Q.z_dr_0p05_0p1 - 0.5176166
    if Q.girth2_top15 < 0.001319197:
        z += 5.883743 * Q.girth2_top15 + 0.2470379
    if 0.001319197 <= Q.girth2_top15 < 0.003270031:
        z += -179.3737 * Q.girth2_top15 + 0.4914289
    if 0.003270031 <= Q.girth2_top15 < 0.00727763:
        z += -29.66772 * Q.girth2_top15 + 0.001885837
    if 0.00727763 <= Q.girth2_top15 < 0.009962397:
        z += 79.71823 * Q.girth2_top15 - 0.7941847
    if Q.girth < 0.05048381:
        z += 31.92351 * Q.girth - 6.283361
    if 0.05048381 <= Q.girth < 0.08589404:
        z += 97.6932 * Q.girth - 9.603665
    if 0.08589404 <= Q.girth < 0.09749958:
        z += 104.4675 * Q.girth - 10.18553
    if Q.e3 < 5.13841e-05:
        z += -1100.049 * Q.e3 + 0.370974
    if 5.13841e-05 <= Q.e3 < 0.0001086251:
        z += 3596.278 * Q.e3 + 0.1296574
    if 0.0001086251 <= Q.e3 < 0.0003372339:
        z += -1792.424 * Q.e3 + 0.7150056
    if Q.e3 >= 0.0003372339:
        z += -692.3745 * Q.e3 + 0.3440316
    if Q.sj2_zsoft < 0.04575675:
        z += -7.049035 * Q.sj2_zsoft + 0.5346428
    if 0.04575675 <= Q.sj2_zsoft < 0.3676068:
        z += -0.6590081 * Q.sj2_zsoft + 0.2422559
    if Q.e2 < 0.02515919:
        z += -31.57206 * Q.e2 + 1.739767
    if 0.02515919 <= Q.e2 < 0.04358622:
        z += -46.60296 * Q.e2 + 2.117932
    if 0.04358622 <= Q.e2 < 0.04755309:
        z += -21.85231 * Q.e2 + 1.039145
    if Q.e2 >= 0.05557149:
        z += 18.27563 * Q.e2 - 1.015604
    if Q.z_dr_0_0p05 >= 0.3289237:
        z += 1.229817 * Q.z_dr_0_0p05 - 0.404516
    if Q.tau21_b2 >= 0.7058597:
        z += -1.09946 * Q.tau21_b2 + 0.7760642
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += -0.03942797 * Q.n_dr_0p2_0p4 + 0.5914196
    if Q.girth2_top5 < 0.0004005745:
        z += -425.0985 * Q.girth2_top5 + 0.2425941
    if 0.0004005745 <= Q.girth2_top5 < 0.002270363:
        z += 63.17877 * Q.girth2_top5 + 0.04700271
    if 0.002270363 <= Q.girth2_top5 < 0.007164202:
        z += -38.91453 * Q.girth2_top5 + 0.2787916
    if Q.tau2 < 0.0795038:
        z += -6.113321 * Q.tau2 + 0.4860323
    if Q.lam2 < 0.006427167:
        z += -128.007 * Q.lam2 + 0.8227222
    if Q.sj3_dr23 >= 0.3357015:
        z += -0.9859479 * Q.sj3_dr23 + 0.3309842
    if Q.LHA < 0.2091025:
        z += -2.708065 * Q.LHA + 3.421554
    if 0.2091025 <= Q.LHA < 0.2284021:
        z += -9.229051 * Q.LHA + 4.785109
    if 0.2284021 <= Q.LHA < 0.302389:
        z += -24.24981 * Q.LHA + 8.215881
    if 0.302389 <= Q.LHA < 0.3332345:
        z += -28.62672 * Q.LHA + 9.539411
    if Q.z_top20_slots >= 0.7818983:
        z += 0.9497489 * Q.z_top20_slots - 0.7426071
    if Q.dr_0 < 0.04649465:
        z += 3.62775 * Q.dr_0 - 0.168671
    if Q.sum_pt_top50 >= 1078.994:
        z += 0.002166717 * Q.sum_pt_top50 - 2.337875
    if Q.sum_pt >= 907.9372:
        z += -0.001748372 * Q.sum_pt + 1.587412
    if Q.z_dr_0p1_0p2 >= 0.3340477:
        z += 0.7445435 * Q.z_dr_0p1_0p2 - 0.2487131
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += -3.395032 * Q.z_dr_0p2_0p4 + 0.2325309
    if Q.M2 >= 0.03112708:
        z += 2.219409 * Q.M2 - 0.06908372
    if Q.sj2_dr < 0.1825048:
        z += -1.044661 * Q.sj2_dr + 0.1906556
    if Q.sum_pt_top15 < 1003.329:
        z += 0.000681982 * Q.sum_pt_top15 - 0.6842524
    if Q.tau1 < 0.05444509 and Q.sum_pt_top15 < 951.1375:
        z += 0.06685562 * (0.05444509 - Q.tau1) * (951.1375 - Q.sum_pt_top15)
    if Q.psi_0p2 > 0.9087063 and Q.sj2_dr < 0.1512157:
        z += 30.80639 * (Q.psi_0p2 - 0.9087063) * (0.1512157 - Q.sj2_dr)
    if Q.psi_0p2 > 0.9087063 and Q.z_top50_slots < 0.9704436:
        z += 208.5943 * (Q.psi_0p2 - 0.9087063) * (0.9704436 - Q.z_top50_slots)
    if Q.z_dr_0p05_0p1 < 0.2126484 and Q.zdr_0 > 0.01113024:
        z += 64.94743 * (0.2126484 - Q.z_dr_0p05_0p1) * (Q.zdr_0 - 0.01113024)
    if Q.girth2_top15 < 0.003270031 and Q.z_top15_slots > 0.7733683:
        z += 646.595 * (0.003270031 - Q.girth2_top15) * (Q.z_top15_slots - 0.7733683)
    if Q.girth < 0.08589404 and Q.sum_pt < 1002.379:
        z += 0.210381 * (0.08589404 - Q.girth) * (1002.379 - Q.sum_pt)
    if Q.e3 < 0.0003372339 and Q.log_sum_pt < 7.017258:
        z += -11625.76 * (0.0003372339 - Q.e3) * (7.017258 - Q.log_sum_pt)
    if Q.tau1 < 0.05444509 and Q.sum_pt < 972.0419:
        z += -0.2225952 * (0.05444509 - Q.tau1) * (972.0419 - Q.sum_pt)
    if Q.n_dr_0p1_0p2 > 26.0 and Q.sum_pt > 986.0565:
        z += -0.0001696547 * (Q.n_dr_0p1_0p2 - 26.0) * (Q.sum_pt - 986.0565)
    if Q.n_dr_0p1_0p2 > 15.0 and Q.psi_0p3 > 0.9853273:
        z += 2.254919 * (Q.n_dr_0p1_0p2 - 15.0) * (Q.psi_0p3 - 0.9853273)
    if Q.girth < 0.08589404 and Q.M2 < 0.1134943:
        z += -35.555 * (0.08589404 - Q.girth) * (0.1134943 - Q.M2)
    if Q.sj2_zsoft < 0.3676068 and Q.zdr_1 < 0.008824206:
        z += -76.14678 * (0.3676068 - Q.sj2_zsoft) * (0.008824206 - Q.zdr_1)
    if Q.girth2_top15 < 0.003270031 and Q.sum_pt_top40 > 858.8262:
        z += -0.4325089 * (0.003270031 - Q.girth2_top15) * (Q.sum_pt_top40 - 858.8262)
    if Q.e2 > 0.05557149 and Q.psi_0p3 > 0.9896594:
        z += 2885.484 * (Q.e2 - 0.05557149) * (Q.psi_0p3 - 0.9896594)
    if Q.girth < 0.09749958 and Q.eccentricity > 0.6414784:
        z += -10.02821 * (0.09749958 - Q.girth) * (Q.eccentricity - 0.6414784)
    if Q.n_dr_0p2_0p4 > 15.0 and Q.eta_0 < 0.07952881:
        z += 0.1581353 * (Q.n_dr_0p2_0p4 - 15.0) * (0.07952881 - Q.eta_0)
    if Q.girth < 0.09749958 and Q.psi_0p3 < 0.9638082:
        z += 2639.094 * (0.09749958 - Q.girth) * (0.9638082 - Q.psi_0p3)
    if Q.e2 < 0.04358622 and Q.psi_0p3 < 0.9638082:
        z += -608.173 * (0.04358622 - Q.e2) * (0.9638082 - Q.psi_0p3)
    if Q.LHA < 0.3332345 and Q.psi_0p3 < 0.9638082:
        z += -820.7927 * (0.3332345 - Q.LHA) * (0.9638082 - Q.psi_0p3)
    if Q.z_dr_0p1_0p2 > 0.3340477 and Q.n_dr_0p4_up < 1.0:
        z += 0.4025026 * (Q.z_dr_0p1_0p2 - 0.3340477) * (1.0 - Q.n_dr_0p4_up)
    return max(0.0, z)


def neuron_7(Q):
    z = -1.147755
    if Q.tau21_b2 < 0.2352054:
        z += 15.41926 * Q.tau21_b2 - 3.626695
    if Q.e3 < 3.376709e-05:
        z += 15622.92 * Q.e3 + 0.08957398
    if 3.376709e-05 <= Q.e3 < 5.727594e-05:
        z += -15019.73 * Q.e3 + 1.124287
    if 5.727594e-05 <= Q.e3 < 6.567534e-05:
        z += -31432.96 * Q.e3 + 2.06437
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.02990473 * Q.n_dr_0p2_0p4 + 1.109503
    if 15.0 <= Q.n_dr_0p2_0p4 < 21.0:
        z += -0.1101554 * Q.n_dr_0p2_0p4 + 2.313263
    if Q.psi_0p3 >= 0.9973959:
        z += 2179.729 * Q.psi_0p3 - 2174.053
    if Q.psi_0p1 >= 0.8509811:
        z += -3.340688 * Q.psi_0p1 + 2.842862
    if Q.e2 < 0.03263075:
        z += -9.638495 * Q.e2 + 0.2837056
    if 0.03263075 <= Q.e2 < 0.03680582:
        z += -30.55763 * Q.e2 + 0.9663126
    if 0.03680582 <= Q.e2 < 0.04755309:
        z += 19.19415 * Q.e2 - 0.8648422
    if 0.04755309 <= Q.e2 < 0.05557149:
        z += -5.973619 * Q.e2 + 0.3319629
    if Q.tau1 < 0.1072713:
        z += 49.05027 * Q.tau1 - 4.684609
    if 0.1072713 <= Q.tau1 < 0.1219132:
        z += -39.41233 * Q.tau1 + 4.804886
    if Q.LHA < 0.302389:
        z += -8.533947 * Q.LHA + 1.786206
    if 0.302389 <= Q.LHA < 0.3332345:
        z += 25.75303 * Q.LHA - 8.581797
    if Q.girth < 0.04362872:
        z += 31.17545 * Q.girth - 0.4924037
    if 0.04362872 <= Q.girth < 0.08068193:
        z += 15.86852 * Q.girth + 0.175418
    if 0.08068193 <= Q.girth < 0.09749958:
        z += -86.55911 * Q.girth + 8.439477
    if Q.tau21 < 0.347196:
        z += 1.191959 * Q.tau21 - 0.4138435
    if Q.girth2_top2 < 0.005180665:
        z += -18.35298 * Q.girth2_top2 + 0.09508065
    if Q.n_dr_0p1_0p2 >= 33.0:
        z += -3.010643 * Q.n_dr_0p1_0p2 + 99.35121
    if Q.tau2 < 0.03197846:
        z += 23.38831 * Q.tau2 - 0.7479222
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1937688:
        z += -205.1853 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1937688)
    if Q.tau21_b2 < 0.2352054 and Q.girth2_top15 < 0.007887677:
        z += -10636.29 * (0.2352054 - Q.tau21_b2) * (0.007887677 - Q.girth2_top15)
    if Q.tau21_b2 < 0.2352054 and Q.girth2_top15 < 0.009962397:
        z += 7195.711 * (0.2352054 - Q.tau21_b2) * (0.009962397 - Q.girth2_top15)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt_top50 < 1245.697:
        z += -0.03359652 * (0.2352054 - Q.tau21_b2) * (1245.697 - Q.sum_pt_top50)
    if Q.tau21_b2 < 0.2352054 and Q.girth2_top15 < 0.005788041:
        z += 2876.263 * (0.2352054 - Q.tau21_b2) * (0.005788041 - Q.girth2_top15)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.e3 < 7.876005e-05:
        z += -1864.663 * (21.0 - Q.n_dr_0p2_0p4) * (7.876005e-05 - Q.e3)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.girth2_top5 > 0.008329695:
        z += -0.6033698 * (21.0 - Q.n_dr_0p2_0p4) * (Q.girth2_top5 - 0.008329695)
    if Q.tau21_b2 < 0.2352054 and Q.D3 < 0.2618467:
        z += 7.796315 * (0.2352054 - Q.tau21_b2) * (0.2618467 - Q.D3)
    if Q.psi_0p3 > 0.9973959 and Q.mean_eta2 < 0.006802603:
        z += -246407.0 * (Q.psi_0p3 - 0.9973959) * (0.006802603 - Q.mean_eta2)
    if Q.psi_0p3 > 0.9973959 and Q.z_top50_slots < 0.985099:
        z += -61155.63 * (Q.psi_0p3 - 0.9973959) * (0.985099 - Q.z_top50_slots)
    if Q.psi_0p3 > 0.9973959 and Q.log_sum_pt < 7.062574:
        z += -4620.889 * (Q.psi_0p3 - 0.9973959) * (7.062574 - Q.log_sum_pt)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 972.0419:
        z += -0.09963459 * (0.2352054 - Q.tau21_b2) * (972.0419 - Q.sum_pt)
    if Q.psi_0p3 > 0.9973959 and Q.mean_phi2 < 0.006808102:
        z += -300637.8 * (Q.psi_0p3 - 0.9973959) * (0.006808102 - Q.mean_phi2)
    if Q.tau1 < 0.1219132 and Q.z_dr_0p05_0p1 > 0.4026646:
        z += -37.07112 * (0.1219132 - Q.tau1) * (Q.z_dr_0p05_0p1 - 0.4026646)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.e3 < 3.376709e-05:
        z += 3451.321 * (21.0 - Q.n_dr_0p2_0p4) * (3.376709e-05 - Q.e3)
    if Q.psi_0p3 > 0.9973959 and Q.girth2_top10 > 0.007678544:
        z += -79107.74 * (Q.psi_0p3 - 0.9973959) * (Q.girth2_top10 - 0.007678544)
    if Q.psi_0p3 > 0.9973959 and Q.girth2_top10 > 0.01414829:
        z += -1846621.0 * (Q.psi_0p3 - 0.9973959) * (Q.girth2_top10 - 0.01414829)
    if Q.tau1 < 0.1219132 and Q.zdr_0 > 0.0138766:
        z += -825.2379 * (0.1219132 - Q.tau1) * (Q.zdr_0 - 0.0138766)
    if Q.tau1 < 0.1072713 and Q.z_dr_0p1_0p2 < 0.2864926:
        z += 216.1278 * (0.1072713 - Q.tau1) * (0.2864926 - Q.z_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.ptdr0_10 > 7.702385:
        z += -0.01191582 * (21.0 - Q.n_dr_0p2_0p4) * (Q.ptdr0_10 - 7.702385)
    if Q.LHA < 0.3332345 and Q.z_dr_0p1_0p2 < 0.250441:
        z += -44.49848 * (0.3332345 - Q.LHA) * (0.250441 - Q.z_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.n_dr_0p1_0p2 > 21.0:
        z += -0.003935711 * (21.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 21.0)
    if Q.tau21_b2 < 0.2352054 and Q.M2 < 0.06975954:
        z += 66.09725 * (0.2352054 - Q.tau21_b2) * (0.06975954 - Q.M2)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.M2 < 0.1134943:
        z += 0.5330493 * (15.0 - Q.n_dr_0p2_0p4) * (0.1134943 - Q.M2)
    if Q.tau21_b2 < 0.2352054 and Q.n_real_top40 < 36.0:
        z += -0.1083914 * (0.2352054 - Q.tau21_b2) * (36.0 - Q.n_real_top40)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.max_dr < 0.4357228:
        z += 0.0527415 * (15.0 - Q.n_dr_0p2_0p4) * (0.4357228 - Q.max_dr)
    if Q.psi_0p3 > 0.9973959 and Q.lam2 < 0.003687605:
        z += 113667.1 * (Q.psi_0p3 - 0.9973959) * (0.003687605 - Q.lam2)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sj3_dr23 > 0.2351209:
        z += -0.1436598 * (21.0 - Q.n_dr_0p2_0p4) * (Q.sj3_dr23 - 0.2351209)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.eta_1 > 0.002292633:
        z += 0.1900747 * (15.0 - Q.n_dr_0p2_0p4) * (Q.eta_1 - 0.002292633)
    if Q.tau21_b2 < 0.2352054 and Q.z_top50_slots > 0.9704436:
        z += 393.7624 * (0.2352054 - Q.tau21_b2) * (Q.z_top50_slots - 0.9704436)
    if Q.tau21_b2 < 0.2352054 and Q.orientation_deg < 44.7593:
        z += 0.008271663 * (0.2352054 - Q.tau21_b2) * (44.7593 - Q.orientation_deg)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.mean_phi < 0.0004404746:
        z += -13.09015 * (15.0 - Q.n_dr_0p2_0p4) * (0.0004404746 - Q.mean_phi)
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1512157:
        z += 183.3967 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1512157)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.psi_0p3 > 0.9956185:
        z += 13.38529 * (15.0 - Q.n_dr_0p2_0p4) * (Q.psi_0p3 - 0.9956185)
    if Q.tau21 < 0.347196 and Q.sj2_dr < 0.2070855:
        z += 43.1099 * (0.347196 - Q.tau21) * (0.2070855 - Q.sj2_dr)
    if Q.psi_0p3 > 0.9973959 and Q.tau21_b2 < 0.541744:
        z += -667.63 * (Q.psi_0p3 - 0.9973959) * (0.541744 - Q.tau21_b2)
    if Q.sj2_zsoft > 0.4184936 and Q.eccentricity > 0.8680812:
        z += 57.06543 * (Q.sj2_zsoft - 0.4184936) * (Q.eccentricity - 0.8680812)
    if Q.e2 < 0.04755309 and Q.tau21_b2 < 0.2352054:
        z += -114.6986 * (0.04755309 - Q.e2) * (0.2352054 - Q.tau21_b2)
    if Q.tau21_b2 < 0.2352054 and Q.ptdr0_14 > 4.336333:
        z += -1.306212 * (0.2352054 - Q.tau21_b2) * (Q.ptdr0_14 - 4.336333)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.tau21_b2 < 0.4823776:
        z += 0.05066746 * (21.0 - Q.n_dr_0p2_0p4) * (0.4823776 - Q.tau21_b2)
    if Q.n_dr_0p1_0p2 > 33.0 and Q.pt_2 > 92.5:
        z += -0.1212147 * (Q.n_dr_0p1_0p2 - 33.0) * (Q.pt_2 - 92.5)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.eta_1 < -0.009371567:
        z += 0.181068 * (15.0 - Q.n_dr_0p2_0p4) * (-0.009371567 - Q.eta_1)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.ptdr0_2 > 11.87288:
        z += 0.002687074 * (15.0 - Q.n_dr_0p2_0p4) * (Q.ptdr0_2 - 11.87288)
    if Q.tau21_b2 < 0.2352054 and Q.ptdr0_2 > 0.7280557:
        z += -0.06415673 * (0.2352054 - Q.tau21_b2) * (Q.ptdr0_2 - 0.7280557)
    if Q.girth < 0.08068193 and Q.girth2_top3 > 0.003952582:
        z += -3631.415 * (0.08068193 - Q.girth) * (Q.girth2_top3 - 0.003952582)
    if Q.psi_0p3 > 0.9973959 and Q.mean_phi2 < 0.005203178:
        z += 69708.55 * (Q.psi_0p3 - 0.9973959) * (0.005203178 - Q.mean_phi2)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.008219
    if 0.03577037 <= Q.girth < 0.04362872:
        z += 38.99218 * Q.girth - 1.394765
    if 0.04362872 <= Q.girth < 0.07031778:
        z += 108.478 * Q.girth - 4.426341
    if 0.07031778 <= Q.girth < 0.07374472:
        z += 78.66297 * Q.girth - 2.329816
    if 0.07374472 <= Q.girth < 0.09749958:
        z += 37.32774 * Q.girth + 0.7184387
    if 0.09749958 <= Q.girth < 0.1207452:
        z += -60.81404 * Q.girth + 10.28722
    if Q.girth >= 0.1207452:
        z += -81.41446 * Q.girth + 12.77462
    if Q.sum_pt_top40 < 1007.44:
        z += -0.001873086 * Q.sum_pt_top40 + 1.887022
    if Q.n_dr_0p2_0p4 < 3.0:
        z += 0.06086224 * Q.n_dr_0p2_0p4 - 1.09552
    if 3.0 <= Q.n_dr_0p2_0p4 < 18.0:
        z += 0.0687275 * Q.n_dr_0p2_0p4 - 1.119116
    if Q.n_dr_0p2_0p4 >= 18.0:
        z += 0.007865261 * Q.n_dr_0p2_0p4 - 0.02359578
    if Q.n_pt_above_1 >= 54.0:
        z += 0.02668567 * Q.n_pt_above_1 - 1.441026
    if Q.sum_pt < 1002.379:
        z += -0.03688816 * Q.sum_pt + 41.39595
    if 1002.379 <= Q.sum_pt < 1042.609:
        z += -0.02684358 * Q.sum_pt + 31.32747
    if 1042.609 <= Q.sum_pt < 1260.541:
        z += -0.01532641 * Q.sum_pt + 19.31956
    if 0.2091025 <= Q.LHA < 0.3098384:
        z += -27.03641 * Q.LHA + 5.653381
    if 0.3098384 <= Q.LHA < 0.3203321:
        z += -20.63327 * Q.LHA + 3.669443
    if 0.3203321 <= Q.LHA < 0.3332345:
        z += -8.680541 * Q.LHA - 0.1594009
    if Q.LHA >= 0.3332345:
        z += 20.44067 * Q.LHA - 9.863593
    if Q.log_sum_pt < 6.959294:
        z += 20.98376 * Q.log_sum_pt - 149.386
    if 6.959294 <= Q.log_sum_pt < 7.139296:
        z += 18.63192 * Q.log_sum_pt - 133.0188
    if 0.9777125 <= Q.psi_0p3 < 0.9943058:
        z += -6.406609 * Q.psi_0p3 + 6.263822
    if Q.psi_0p3 >= 0.9943058:
        z += -33.56289 * Q.psi_0p3 + 33.26547
    if Q.z_top50_slots >= 0.9586536:
        z += -13.21884 * Q.z_top50_slots + 12.67229
    if Q.psi_0p1 >= 0.6635952:
        z += 1.224088 * Q.psi_0p1 - 0.8122989
    if Q.e3 < 3.793233e-05:
        z += 5102.865 * Q.e3 - 0.3292632
    if 3.793233e-05 <= Q.e3 < 0.0001086251:
        z += 1919.569 * Q.e3 - 0.2085133
    if Q.girth2_top15 < 0.00727763:
        z += 32.40292 * Q.girth2_top15 - 1.442323
    if 0.00727763 <= Q.girth2_top15 < 0.009962397:
        z += 149.9627 * Q.girth2_top15 - 2.29788
    if 0.009962397 <= Q.girth2_top15 < 0.02675364:
        z += 47.87566 * Q.girth2_top15 - 1.280848
    if Q.sum_pt_top30 < 1027.303:
        z += 0.00201331 * Q.sum_pt_top30 - 2.311385
    if 1027.303 <= Q.sum_pt_top30 < 1191.938:
        z += 0.00147663 * Q.sum_pt_top30 - 1.760051
    if Q.n_particles >= 43.0:
        z += 0.015797 * Q.n_particles - 0.6792709
    if 0.02210818 <= Q.e2 < 0.04358622:
        z += -26.26403 * Q.e2 + 0.5806499
    if Q.e2 >= 0.04358622:
        z += 13.70468 * Q.e2 - 1.161435
    if Q.D2 < 2.178951:
        z += -0.1514071 * Q.D2 + 0.3299085
    if Q.sj2_dr >= 0.2232169:
        z += 2.467693 * Q.sj2_dr - 0.5508308
    if Q.dr_0 < 0.08082334:
        z += -2.798429 * Q.dr_0 + 0.2261784
    if Q.sj2_zsoft < 0.04575675:
        z += 5.522567 * Q.sj2_zsoft - 0.2526948
    if Q.psi_0p2 >= 0.7286738:
        z += 0.6745889 * Q.psi_0p2 - 0.4915552
    if Q.sum_pt_top50 < 1156.659:
        z += -0.0002301787 * Q.sum_pt_top50 + 0.2662383
    if Q.n_dr_0p2_0p4 < 18.0 and Q.log_sum_pt < 7.139296:
        z += -0.06763906 * (18.0 - Q.n_dr_0p2_0p4) * (7.139296 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.psi_0p1 < 0.8252004:
        z += 0.1426193 * (18.0 - Q.n_dr_0p2_0p4) * (0.8252004 - Q.psi_0p1)
    if Q.girth > 0.07031778 and Q.e2 > 0.04358622:
        z += 146.7772 * (Q.girth - 0.07031778) * (Q.e2 - 0.04358622)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.e3 < 3.793233e-05:
        z += 761.2261 * (18.0 - Q.n_dr_0p2_0p4) * (3.793233e-05 - Q.e3)
    if Q.girth > 0.07031778 and Q.tau2 < 0.0795038:
        z += 101.1395 * (Q.girth - 0.07031778) * (0.0795038 - Q.tau2)
    if Q.girth > 0.07031778 and Q.sum_pt_top30 < 1191.938:
        z += 0.07539268 * (Q.girth - 0.07031778) * (1191.938 - Q.sum_pt_top30)
    if Q.sum_pt_top40 < 1007.44 and Q.sum_pt > 949.9169:
        z += -6.116676e-05 * (1007.44 - Q.sum_pt_top40) * (Q.sum_pt - 949.9169)
    if Q.girth > 0.07031778 and Q.sum_pt_top50 < 976.277:
        z += -0.2273533 * (Q.girth - 0.07031778) * (976.277 - Q.sum_pt_top50)
    if Q.girth > 0.07031778 and Q.sum_pt < 1115.723:
        z += 0.1624148 * (Q.girth - 0.07031778) * (1115.723 - Q.sum_pt)
    if Q.sum_pt < 1260.541 and Q.D2 < 4.450169:
        z += -0.0002769183 * (1260.541 - Q.sum_pt) * (4.450169 - Q.D2)
    if Q.sum_pt < 1002.379 and Q.z_0 > 0.08872428:
        z += -0.006525842 * (1002.379 - Q.sum_pt) * (Q.z_0 - 0.08872428)
    if Q.girth2_top15 < 0.02675364 and Q.tau4 > 0.01517184:
        z += 789.7875 * (0.02675364 - Q.girth2_top15) * (Q.tau4 - 0.01517184)
    if Q.sum_pt < 1260.541 and Q.n_dr_0p05_0p1 > 7.0:
        z += -3.625098e-05 * (1260.541 - Q.sum_pt) * (Q.n_dr_0p05_0p1 - 7.0)
    if Q.girth2_top15 < 0.009962397 and Q.max_dr < 0.3315857:
        z += -389.1243 * (0.009962397 - Q.girth2_top15) * (0.3315857 - Q.max_dr)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.6166
    if Q.sum_pt_top50 < 889.8503:
        z += -0.031908 * Q.sum_pt_top50 + 29.57323
    if 889.8503 <= Q.sum_pt_top50 < 988.4554:
        z += -0.0119658 * Q.sum_pt_top50 + 11.82766
    if Q.girth < 0.05660088:
        z += -36.10784 * Q.girth + 2.043735
    if 0.09749958 <= Q.girth < 0.1207452:
        z += 31.83179 * Q.girth - 3.103586
    if 0.1207452 <= Q.girth < 0.1402186:
        z += -43.19475 * Q.girth + 5.955506
    if 0.1402186 <= Q.girth < 0.1564779:
        z += -155.7125 * Q.girth + 21.73258
    if Q.girth >= 0.1564779:
        z += -167.8448 * Q.girth + 23.63102
    if Q.z_dr_0_0p05 >= 0.8103116:
        z += -2.180568 * Q.z_dr_0_0p05 + 1.766939
    if Q.z_dr_0p1_0p2 >= 0.4479367:
        z += -0.7050719 * Q.z_dr_0p1_0p2 + 0.3158276
    if Q.z_top5_slots >= 0.7963976:
        z += 2.892856 * Q.z_top5_slots - 2.303863
    if Q.LHA < 0.2454112:
        z += 11.28606 * Q.LHA - 2.769726
    if 0.3719813 <= Q.LHA < 0.404204:
        z += 23.87898 * Q.LHA - 8.882533
    if Q.LHA >= 0.404204:
        z += 80.19172 * Q.LHA - 31.64437
    if Q.sum_pt_top20 >= 750.7313:
        z += -0.002711846 * Q.sum_pt_top20 + 2.035867
    if Q.girth2_top15 < 0.0007894752:
        z += 597.0147 * Q.girth2_top15 - 1.252345
    if 0.0007894752 <= Q.girth2_top15 < 0.004855289:
        z += 127.2549 * Q.girth2_top15 - 0.881481
    if 0.004855289 <= Q.girth2_top15 < 0.006142802:
        z += 204.7527 * Q.girth2_top15 - 1.257756
    if Q.girth2_top15 >= 0.02146578:
        z += -17.31047 * Q.girth2_top15 + 0.3715827
    if Q.D2 < 2.178951:
        z += -0.2784364 * Q.D2 + 0.6066992
    if Q.psi_0p1 >= 0.8509811:
        z += 3.296022 * Q.psi_0p1 - 2.804853
    if Q.z_top40_slots >= 0.9674996:
        z += -4.014008 * Q.z_top40_slots + 3.883551
    if Q.tau1 < 0.09591084:
        z += -32.50724 * Q.tau1 + 3.393402
    if 0.09591084 <= Q.tau1 < 0.1072713:
        z += -24.26022 * Q.tau1 + 2.602424
    if Q.e2 >= 0.02515919:
        z += 12.75517 * Q.e2 - 0.3209098
    if Q.psi_0p3 >= 0.9966167:
        z += 43.40948 * Q.psi_0p3 - 43.26262
    if Q.log_sum_pt < 6.811175:
        z += 17.37318 * Q.log_sum_pt - 118.3318
    if Q.girth2_top2 < 0.002412891:
        z += 91.34048 * Q.girth2_top2 - 0.2203946
    if Q.girth < 0.05660088 and Q.sum_pt < 1007.788:
        z += -0.2885696 * (0.05660088 - Q.girth) * (1007.788 - Q.sum_pt)
    if Q.sum_pt_top50 < 988.4554 and Q.sum_pt_top20 > 789.7344:
        z += -7.05042e-05 * (988.4554 - Q.sum_pt_top50) * (Q.sum_pt_top20 - 789.7344)
    if Q.sum_pt_top50 < 988.4554 and Q.soft10_pt > 2.498047:
        z += -0.0010613 * (988.4554 - Q.sum_pt_top50) * (Q.soft10_pt - 2.498047)
    if Q.girth < 0.05660088 and Q.psi_0p3 > 0.9638082:
        z += 293.9943 * (0.05660088 - Q.girth) * (Q.psi_0p3 - 0.9638082)
    if Q.girth2_top15 < 0.006142802 and Q.psi_0p3 > 0.9777125:
        z += 5849.144 * (0.006142802 - Q.girth2_top15) * (Q.psi_0p3 - 0.9777125)
    if Q.girth2_top15 < 0.006142802 and Q.z_dr_0p2_0p4 < 0.0684915:
        z += 4905.669 * (0.006142802 - Q.girth2_top15) * (0.0684915 - Q.z_dr_0p2_0p4)
    if Q.girth2_top15 < 0.006142802 and Q.n_dr_0p2_0p4 > 7.0:
        z += 12.87497 * (0.006142802 - Q.girth2_top15) * (Q.n_dr_0p2_0p4 - 7.0)
    if Q.sum_pt_top50 < 988.4554 and Q.e3 > 0.0001086251:
        z += 8.906204 * (988.4554 - Q.sum_pt_top50) * (Q.e3 - 0.0001086251)
    if Q.LHA > 0.3719813 and Q.lam2 < 0.006427167:
        z += 941.546 * (Q.LHA - 0.3719813) * (0.006427167 - Q.lam2)
    if Q.sj2_dr < 0.2595052 and Q.pt_0 > 213.5:
        z += 0.002884752 * (0.2595052 - Q.sj2_dr) * (Q.pt_0 - 213.5)
    if Q.sum_pt_top50 < 988.4554 and Q.C3 < 0.001416411:
        z += 2.073945 * (988.4554 - Q.sum_pt_top50) * (0.001416411 - Q.C3)
    if Q.z_top40_slots > 0.9674996 and Q.C3 < 0.01158441:
        z += -421.516 * (Q.z_top40_slots - 0.9674996) * (0.01158441 - Q.C3)
    if Q.sum_pt_top50 < 988.4554 and Q.soft5_pt > 1.652344:
        z += -0.00115701 * (988.4554 - Q.sum_pt_top50) * (Q.soft5_pt - 1.652344)
    if Q.girth > 0.1402186 and Q.soft6_pt > 4.250195:
        z += 37.09109 * (Q.girth - 0.1402186) * (Q.soft6_pt - 4.250195)
    if Q.log_sum_pt < 6.811175 and Q.dr_13 < 0.09061548:
        z += 80.57544 * (6.811175 - Q.log_sum_pt) * (0.09061548 - Q.dr_13)
    if Q.sum_pt_top50 < 988.4554 and Q.dr_13 < 0.1312677:
        z += -0.02794641 * (988.4554 - Q.sum_pt_top50) * (0.1312677 - Q.dr_13)
    if Q.LHA > 0.3719813 and Q.soft4_z > 0.002015695:
        z += -6048.682 * (Q.LHA - 0.3719813) * (Q.soft4_z - 0.002015695)
    if Q.girth > 0.1564779 and Q.soft6_pt > 1.931641:
        z += -11.71947 * (Q.girth - 0.1564779) * (Q.soft6_pt - 1.931641)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.soft5_z > 0.0004140594:
        z += 3209.644 * (Q.z_dr_0p1_0p2 - 0.4479367) * (Q.soft5_z - 0.0004140594)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.soft5_pt > 0.5297852:
        z += -3.047344 * (Q.z_dr_0p1_0p2 - 0.4479367) * (Q.soft5_pt - 0.5297852)
    return max(0.0, z)


def neuron_10(Q):
    z = 3.766281
    if Q.girth < 0.09749958:
        z += 47.66122 * Q.girth - 5.754863
    if 0.09749958 <= Q.girth < 0.1207452:
        z += -48.37959 * Q.girth + 3.609077
    if 0.1207452 <= Q.girth < 0.1402186:
        z += -96.04082 * Q.girth + 9.36394
    if Q.girth >= 0.1402186:
        z += -203.8726 * Q.girth + 24.48396
    if Q.tau1 < 0.04466492:
        z += -21.33681 * Q.tau1 + 1.644777
    if 0.04466492 <= Q.tau1 < 0.07708632:
        z += -33.57684 * Q.tau1 + 2.191476
    if 0.07708632 <= Q.tau1 < 0.1751567:
        z += -12.24002 * Q.tau1 + 0.5466996
    if 0.1751567 <= Q.tau1 < 0.1953848:
        z += 3.51829 * Q.tau1 - 2.213474
    if Q.tau1 >= 0.1953848:
        z += -65.31532 * Q.tau1 + 11.23557
    if 0.008484542 <= Q.e2 < 0.03029714:
        z += 85.17748 * Q.e2 - 0.722692
    if Q.e2 >= 0.03029714:
        z += 54.67736 * Q.e2 + 0.2013748
    if 402.625 <= Q.sum_pt_top5 < 791.125:
        z += -0.001610708 * Q.sum_pt_top5 + 0.6485114
    if Q.sum_pt_top5 >= 791.125:
        z += -0.002516196 * Q.sum_pt_top5 + 1.364866
    if Q.soft7_z < 0.002672224:
        z += -56.71732 * Q.soft7_z + 0.1515614
    if Q.sj2_dr >= 0.09395198:
        z += 1.720417 * Q.sj2_dr - 0.1616366
    if 0.9985421 <= Q.psi_0p3 < 0.9989733:
        z += -321.3749 * Q.psi_0p3 + 320.9064
    if Q.psi_0p3 >= 0.9989733:
        z += 39.6958 * Q.psi_0p3 - 39.79362
    if Q.sd_rg >= 0.3017146:
        z += -3.043879 * Q.sd_rg + 0.9183826
    if Q.sum_pt_top50 < 889.8503:
        z += -0.002906538 * Q.sum_pt_top50 + 2.586384
    if 0.1870291 <= Q.LHA < 0.3332345:
        z += -5.99224 * Q.LHA + 1.120723
    if 0.3332345 <= Q.LHA < 0.404204:
        z += 20.3746 * Q.LHA - 7.665619
    if Q.LHA >= 0.404204:
        z += 62.89962 * Q.LHA - 24.8544
    if Q.n_dr_0p1_0p2 >= 21.0:
        z += 0.03050774 * Q.n_dr_0p1_0p2 - 0.6406625
    if Q.girth2_top10 < 0.01414829:
        z += 31.3728 * Q.girth2_top10 - 0.4438714
    if Q.z_dr_0_0p05 >= 0.7128619:
        z += -2.059288 * Q.z_dr_0_0p05 + 1.467988
    if Q.C2 < 0.05602756:
        z += -5.943116 * Q.C2 + 0.3329783
    if Q.e3 < 5.13841e-05:
        z += -11468.45 * Q.e3 + 0.6200232
    if 5.13841e-05 <= Q.e3 < 0.0001086251:
        z += -3759.478 * Q.e3 + 0.2239044
    if 0.0001086251 <= Q.e3 < 0.0003372339:
        z += 806.9205 * Q.e3 - 0.2721209
    if Q.ptdr0_2 >= 7.740999:
        z += 0.007039365 * Q.ptdr0_2 - 0.05449172
    if Q.tau21_b2 < 0.2018786:
        z += 2.912457 * Q.tau21_b2 - 0.5879628
    if Q.log_sum_pt < 6.856375:
        z += 3.130494 * Q.log_sum_pt - 21.46384
    if Q.D2 < 2.410481:
        z += -0.1982977 * Q.D2 + 0.4779928
    if Q.sum_pt < 986.0565:
        z += 0.009392899 * Q.sum_pt - 9.261929
    if Q.psi_0p1 >= 0.9184255:
        z += 4.678307 * Q.psi_0p1 - 4.296676
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += 2.066739 * Q.z_dr_0p1_0p2 - 0.248719
    if Q.n_pt_above_5 >= 42.0:
        z += -0.008358628 * Q.n_pt_above_5 + 0.3510624
    if Q.girth2_top5 < 0.02441963:
        z += 20.83365 * Q.girth2_top5 - 0.5087501
    if Q.sum_pt_top40 < 1024.942:
        z += 0.005744496 * Q.sum_pt_top40 - 5.887777
    if Q.n_dr_0p2_0p4 < 18.0:
        z += 0.02022426 * Q.n_dr_0p2_0p4 - 0.3640366
    if Q.sum_pt_top20 < 889.8383:
        z += -0.001874416 * Q.sum_pt_top20 + 1.667927
    if Q.girth2_top2 < 0.004007842:
        z += -100.4849 * Q.girth2_top2 + 0.4027275
    if Q.psi_0p2 >= 0.9087063:
        z += -3.986323 * Q.psi_0p2 + 3.622396
    if Q.pt_entropy >= 2.07371:
        z += 0.4086817 * Q.pt_entropy - 0.8474871
    if Q.D2_b2 < 2.026142:
        z += 0.1144183 * Q.D2_b2 - 0.2318277
    if Q.M3 < 0.02944575:
        z += 7.20636 * Q.M3 - 0.2121967
    if Q.girth < 0.1207452 and Q.psi_0p2 > 0.948102:
        z += 236.3725 * (0.1207452 - Q.girth) * (Q.psi_0p2 - 0.948102)
    if Q.e2 > 0.03029714 and Q.log_sum_pt > 6.910131:
        z += -117.5099 * (Q.e2 - 0.03029714) * (Q.log_sum_pt - 6.910131)
    if Q.e2 > 0.03029714 and Q.soft1_pt > 1.091797:
        z += -27.4962 * (Q.e2 - 0.03029714) * (Q.soft1_pt - 1.091797)
    if Q.girth < 0.1207452 and Q.pt1_dr01 > 5.351077:
        z += 0.3523178 * (0.1207452 - Q.girth) * (Q.pt1_dr01 - 5.351077)
    if Q.n_dr_0p2_0p4 < 11.0 and Q.n_real_top40 > 22.0:
        z += -0.00356611 * (11.0 - Q.n_dr_0p2_0p4) * (Q.n_real_top40 - 22.0)
    if Q.LHA > 0.1870291 and Q.psi_0p3 < 0.9973959:
        z += -16.04115 * (Q.LHA - 0.1870291) * (0.9973959 - Q.psi_0p3)
    if Q.sj2_dr > 0.09395198 and Q.planar_flow < 0.6025827:
        z += -1.313799 * (Q.sj2_dr - 0.09395198) * (0.6025827 - Q.planar_flow)
    if Q.tau1 > 0.1953848 and Q.sum_pt > 1167.447:
        z += -6.983377 * (Q.tau1 - 0.1953848) * (Q.sum_pt - 1167.447)
    if Q.n_dr_0p2_0p4 < 11.0 and Q.max_dr > 0.2404747:
        z += 0.1143661 * (11.0 - Q.n_dr_0p2_0p4) * (Q.max_dr - 0.2404747)
    if Q.sj2_dr > 0.09395198 and Q.M2 > 0.04260132:
        z += -38.41725 * (Q.sj2_dr - 0.09395198) * (Q.M2 - 0.04260132)
    if Q.tau1 > 0.1953848 and Q.pt_3 < 114.75:
        z += 2.114734 * (Q.tau1 - 0.1953848) * (114.75 - Q.pt_3)
    if Q.e3 < 0.0003372339 and Q.tau32 < 0.8644992:
        z += -1620.26 * (0.0003372339 - Q.e3) * (0.8644992 - Q.tau32)
    if Q.sum_pt_top50 < 889.8503 and Q.z_dr_0p05_0p1 < 0.2126484:
        z += -0.03377501 * (889.8503 - Q.sum_pt_top50) * (0.2126484 - Q.z_dr_0p05_0p1)
    if Q.girth < 0.1207452 and Q.sum_pt_top50 < 997.0189:
        z += 0.1873512 * (0.1207452 - Q.girth) * (997.0189 - Q.sum_pt_top50)
    if Q.tau1 > 0.1953848 and Q.z_3 < 0.1082864:
        z += -1653.733 * (Q.tau1 - 0.1953848) * (0.1082864 - Q.z_3)
    if Q.e2 > 0.03029714 and Q.z_dr_0p4_up > 0.0:
        z += -919.3525 * (Q.e2 - 0.03029714) * (Q.z_dr_0p4_up - 0.0)
    if Q.LHA > 0.1870291 and Q.sj3_pairmin_over_m > 0.1389615:
        z += 4.708738 * (Q.LHA - 0.1870291) * (Q.sj3_pairmin_over_m - 0.1389615)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.pt_7 > 34.53125:
        z += -0.09259875 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.pt_7 - 34.53125)
    if Q.girth > 0.09749958 and Q.sum_pt_top50 < 1156.659:
        z += 0.2131912 * (Q.girth - 0.09749958) * (1156.659 - Q.sum_pt_top50)
    if Q.sum_pt_top40 < 1024.942 and Q.n_dr_0_0p05 > 1.0:
        z += 0.0001199227 * (1024.942 - Q.sum_pt_top40) * (Q.n_dr_0_0p05 - 1.0)
    if Q.n_dr_0p2_0p4 < 11.0 and Q.n_dr_0_0p05 > 20.0:
        z += 0.002435079 * (11.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0_0p05 - 20.0)
    if Q.sum_pt < 986.0565 and Q.dr_3 < 0.02358801:
        z += -0.1625974 * (986.0565 - Q.sum_pt) * (0.02358801 - Q.dr_3)
    if Q.psi_0p3 > 0.9989733 and Q.eccentricity > 0.5245966:
        z += -427.977 * (Q.psi_0p3 - 0.9989733) * (Q.eccentricity - 0.5245966)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.6297765
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.07041149 * Q.n_dr_0p2_0p4 + 0.9308438
    if 10.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += -0.04534578 * Q.n_dr_0p2_0p4 + 0.6801866
    if Q.girth < 0.03577037:
        z += -12.50565 * Q.girth + 3.13323
    if 0.03577037 <= Q.girth < 0.076787:
        z += -59.90097 * Q.girth + 4.828578
    if 0.076787 <= Q.girth < 0.08589404:
        z += -53.15083 * Q.girth + 4.310255
    if 0.08589404 <= Q.girth < 0.09749958:
        z += 21.97959 * Q.girth - 2.143
    if Q.D2 < 1.409617:
        z += -1.387351 * Q.D2 + 2.06118
    if 1.409617 <= Q.D2 < 1.788105:
        z += -0.2788602 * Q.D2 + 0.4986313
    if Q.girth2_top10 < 0.00130722:
        z += -235.4628 * Q.girth2_top10 + 0.2497921
    if 0.00130722 <= Q.girth2_top10 < 0.004752876:
        z += -41.30936 * Q.girth2_top10 - 0.004009188
    if 0.004752876 <= Q.girth2_top10 < 0.007678544:
        z += 68.47923 * Q.girth2_top10 - 0.5258207
    if Q.N2 < 0.1732514:
        z += 3.973914 * Q.N2 - 0.6884861
    if Q.log_sum_pt >= 6.893714:
        z += -4.196066 * Q.log_sum_pt + 28.92648
    if 0.8706159 <= Q.psi_0p2 < 0.995185:
        z += -3.296203 * Q.psi_0p2 + 2.869727
    if Q.psi_0p2 >= 0.995185:
        z += 42.36403 * Q.psi_0p2 - 42.57065
    if Q.z_dr_0p1_0p2 < 0.1548383:
        z += 1.652853 * Q.z_dr_0p1_0p2 - 0.255925
    if 0.4026646 <= Q.z_dr_0p05_0p1 < 0.8509215:
        z += 0.4101802 * Q.z_dr_0p05_0p1 - 0.1651651
    if Q.z_dr_0p05_0p1 >= 0.8509215:
        z += 3.673654 * Q.z_dr_0p05_0p1 - 2.942126
    if Q.n_dr_0p1_0p2 < 9.0:
        z += 0.003938738 * Q.n_dr_0p1_0p2 + 0.1955737
    if 9.0 <= Q.n_dr_0p1_0p2 < 14.0:
        z += -0.02310223 * Q.n_dr_0p1_0p2 + 0.4389424
    if 14.0 <= Q.n_dr_0p1_0p2 < 19.0:
        z += -0.0415305 * Q.n_dr_0p1_0p2 + 0.6969381
    if Q.n_dr_0p1_0p2 >= 19.0:
        z += -0.01842827 * Q.n_dr_0p1_0p2 + 0.2579958
    if Q.girth2_top15 < 0.006142802:
        z += -31.86542 * Q.girth2_top15 + 0.8041336
    if 0.006142802 <= Q.girth2_top15 < 0.006615185:
        z += -547.4349 * Q.girth2_top15 + 3.971175
    if 0.006615185 <= Q.girth2_top15 < 0.007887677:
        z += -274.887 * Q.girth2_top15 + 2.16822
    if Q.LHA < 0.1870291:
        z += 3.769444 * Q.LHA - 1.983559
    if 0.1870291 <= Q.LHA < 0.3098384:
        z += 13.76954 * Q.LHA - 3.853868
    if 0.3098384 <= Q.LHA < 0.3332345:
        z += -17.62955 * Q.LHA + 5.874777
    if Q.e2 < 0.01879315:
        z += -46.76967 * Q.e2 + 0.4624452
    if 0.01879315 <= Q.e2 < 0.02515919:
        z += -2.680769 * Q.e2 - 0.3661239
    if 0.02515919 <= Q.e2 < 0.04082832:
        z += 30.76608 * Q.e2 - 1.20762
    if 0.04082832 <= Q.e2 < 0.04358622:
        z += -17.58863 * Q.e2 + 0.766622
    if Q.e3 < 3.376709e-05:
        z += 14920.17 * Q.e3 - 0.5038106
    if Q.sum_pt >= 986.0565:
        z += 0.001900446 * Q.sum_pt - 1.873947
    if Q.psi_0p1 >= 0.6635952:
        z += -1.510501 * Q.psi_0p1 + 1.002361
    if Q.dr_0 < 0.05775119:
        z += -6.009326 * Q.dr_0 + 0.04256817
    if 0.05775119 <= Q.dr_0 < 0.09338587:
        z += 8.544418 * Q.dr_0 - 0.797928
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += -4.50104 * Q.z_dr_0p2_0p4 + 0.5013238
    if 0.0684915 <= Q.z_dr_0p2_0p4 < 0.09122568:
        z += -8.491216 * Q.z_dr_0p2_0p4 + 0.774617
    if Q.sj3_dr_max < 0.1210264:
        z += -0.4017222 * Q.sj3_dr_max - 0.07147176
    if 0.1210264 <= Q.sj3_dr_max < 0.1506299:
        z += 2.764574 * Q.sj3_dr_max - 0.4546773
    if 0.1506299 <= Q.sj3_dr_max < 0.1623049:
        z += 13.3144 * Q.sj3_dr_max - 2.043796
    if 0.1623049 <= Q.sj3_dr_max < 0.1894436:
        z += -4.318388 * Q.sj3_dr_max + 0.8180911
    if Q.psi_0p3 >= 0.9980008:
        z += 57.23282 * Q.psi_0p3 - 57.1184
    if Q.z_dr_0_0p05 < 0.09573228:
        z += 3.965903 * Q.z_dr_0_0p05 - 0.3796649
    if Q.zdr_0 < 0.005911134:
        z += 49.2735 * Q.zdr_0 - 0.1319604
    if 0.005911134 <= Q.zdr_0 < 0.01564747:
        z += -16.36158 * Q.zdr_0 + 0.2560174
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_particles > 34.0:
        z += -0.000804083 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_particles - 34.0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2_top10 < 0.005337976:
        z += -28.94161 * (10.0 - Q.n_dr_0p2_0p4) * (0.005337976 - Q.girth2_top10)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 33.0:
        z += 0.002427232 * (10.0 - Q.n_dr_0p2_0p4) * (33.0 - Q.n_dr_0p1_0p2)
    if Q.D2 < 1.409617 and Q.n_real_top50 > 22.0:
        z += -0.04198548 * (1.409617 - Q.D2) * (Q.n_real_top50 - 22.0)
    if Q.girth < 0.09749958 and Q.tau4 > 0.008810529:
        z += -184.8869 * (0.09749958 - Q.girth) * (Q.tau4 - 0.008810529)
    if Q.D2 < 1.409617 and Q.z_7 < 0.04963857:
        z += -12.83509 * (1.409617 - Q.D2) * (0.04963857 - Q.z_7)
    if Q.n_dr_0p1_0p2 < 19.0 and Q.planar_flow < 0.6025827:
        z += 0.05870943 * (19.0 - Q.n_dr_0p1_0p2) * (0.6025827 - Q.planar_flow)
    if Q.log_sum_pt > 6.893714 and Q.M2 > 0.04894324:
        z += 26.4117 * (Q.log_sum_pt - 6.893714) * (Q.M2 - 0.04894324)
    if Q.girth < 0.076787 and Q.D2 < 3.814159:
        z += -2.830376 * (0.076787 - Q.girth) * (3.814159 - Q.D2)
    if Q.z_dr_0_0p05 < 0.09573228 and Q.max_dr > 0.1939977:
        z += 8.230732 * (0.09573228 - Q.z_dr_0_0p05) * (Q.max_dr - 0.1939977)
    if Q.sum_pt > 986.0565 and Q.dr1_7 > 0.1792439:
        z += 0.01568531 * (Q.sum_pt - 986.0565) * (Q.dr1_7 - 0.1792439)
    if Q.log_sum_pt > 6.893714 and Q.dr1_7 > 0.08891009:
        z += -13.80369 * (Q.log_sum_pt - 6.893714) * (Q.dr1_7 - 0.08891009)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.5271399
    if Q.girth < 0.04362872:
        z += -47.04176 * Q.girth + 2.853261
    if 0.04362872 <= Q.girth < 0.05048381:
        z += -56.46292 * Q.girth + 3.264294
    if 0.05048381 <= Q.girth < 0.06171014:
        z += -36.86251 * Q.girth + 2.274791
    if Q.sj3_dr_max < 0.1210264:
        z += 0.4820539 * Q.sj3_dr_max + 0.01443099
    if 0.1210264 <= Q.sj3_dr_max < 0.1506299:
        z += 6.500511 * Q.sj3_dr_max - 0.7139615
    if 0.1506299 <= Q.sj3_dr_max < 0.1999777:
        z += -3.563277 * Q.sj3_dr_max + 0.8019455
    if 0.1999777 <= Q.sj3_dr_max < 0.2628766:
        z += -1.420845 * Q.sj3_dr_max + 0.3735069
    if Q.psi_0p3 >= 0.9896594:
        z += 19.62069 * Q.psi_0p3 - 19.4178
    if Q.LHA < 0.1870291:
        z += 6.250802 * Q.LHA - 1.669259
    if 0.1870291 <= Q.LHA < 0.2284021:
        z += 9.712042 * Q.LHA - 2.316612
    if 0.2284021 <= Q.LHA < 0.2454112:
        z += 5.782836 * Q.LHA - 1.419173
    if 0.3719813 <= Q.LHA < 0.404204:
        z += 7.810806 * Q.LHA - 2.905474
    if Q.LHA >= 0.404204:
        z += -8.191666 * Q.LHA + 3.56279
    if Q.n_dr_0_0p05 >= 20.0:
        z += -0.01809019 * Q.n_dr_0_0p05 + 0.3618038
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -1.816043 * Q.z_dr_0_0p05 + 1.596131
    if Q.z_dr_0p1_0p2 < 0.02620804:
        z += 6.164215 * Q.z_dr_0p1_0p2 - 0.161552
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.02476664 * Q.n_dr_0p2_0p4 + 0.2476664
    if Q.sj3_dr12 < 0.06069672:
        z += -0.8860542 * Q.sj3_dr12 - 0.03173279
    if 0.06069672 <= Q.sj3_dr12 < 0.1048824:
        z += 5.437833 * Q.sj3_dr12 - 0.415572
    if 0.1048824 <= Q.sj3_dr12 < 0.1840219:
        z += -1.955549 * Q.sj3_dr12 + 0.3598638
    if Q.mean_eta2 < 0.005850286:
        z += -51.20294 * Q.mean_eta2 + 0.2995518
    if Q.sum_pt < 907.9372:
        z += -0.0009116001 * Q.sum_pt + 1.064245
    if 907.9372 <= Q.sum_pt < 1167.447:
        z += -0.001798435 * Q.sum_pt + 1.869435
    if Q.sum_pt >= 1167.447:
        z += -0.0008868346 * Q.sum_pt + 0.8051901
    if Q.girth2_top3 >= 0.02353672:
        z += -28.44933 * Q.girth2_top3 + 0.6696039
    if Q.e2 < 0.02515919:
        z += 32.2757 * Q.e2 - 0.8120306
    if 0.03029714 <= Q.e2 < 0.06524004:
        z += -3.296217 * Q.e2 + 0.09986598
    if Q.e2 >= 0.06524004:
        z += 28.89111 * Q.e2 - 2.000036
    if Q.girth2_top10 < 0.01414829:
        z += 4.608503 * Q.girth2_top10 - 0.06520242
    if Q.e3 < 4.646151e-05:
        z += -14683.65 * Q.e3 + 0.6822245
    if Q.lam2 < 0.001411894:
        z += -136.7773 * Q.lam2 + 0.1931151
    if Q.girth2_top15 < 0.002197765:
        z += 17.96397 * Q.girth2_top15 + 0.2497012
    if 0.002197765 <= Q.girth2_top15 < 0.004169954:
        z += -88.98412 * Q.girth2_top15 + 0.4847479
    if 0.004169954 <= Q.girth2_top15 < 0.005788041:
        z += -70.26091 * Q.girth2_top15 + 0.406673
    if Q.sj3_z3 < 0.1690079:
        z += -0.6720235 * Q.sj3_z3 + 0.1135773
    if Q.sum_pt_top10 >= 916.1578:
        z += -0.0007590752 * Q.sum_pt_top10 + 0.6954326
    if Q.pt_dispersion >= 0.3227599:
        z += 0.8266336 * Q.pt_dispersion - 0.2668042
    if Q.girth < 0.05048381 and Q.log_sum_pt > 6.811175:
        z += 113.6277 * (0.05048381 - Q.girth) * (Q.log_sum_pt - 6.811175)
    if Q.sj3_dr_max < 0.2628766 and Q.girth2_top10 > 0.004752876:
        z += -907.3058 * (0.2628766 - Q.sj3_dr_max) * (Q.girth2_top10 - 0.004752876)
    if Q.psi_0p3 > 0.9896594 and Q.girth2_top15 < 0.006615185:
        z += 5970.825 * (Q.psi_0p3 - 0.9896594) * (0.006615185 - Q.girth2_top15)
    if Q.psi_0p3 > 0.9896594 and Q.z_1st > 0.1586697:
        z += 61.3661 * (Q.psi_0p3 - 0.9896594) * (Q.z_1st - 0.1586697)
    if Q.LHA > 0.3719813 and Q.lam2 < 0.003687605:
        z += 2184.744 * (Q.LHA - 0.3719813) * (0.003687605 - Q.lam2)
    if Q.LHA > 0.3719813 and Q.log_sum_pt > 6.856375:
        z += 68.96666 * (Q.LHA - 0.3719813) * (Q.log_sum_pt - 6.856375)
    if Q.psi_0p2 > 0.9734513 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.9787642 * (Q.psi_0p2 - 0.9734513) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.psi_0p2 > 0.9734513 and Q.n_real_top40 < 40.0:
        z += -0.5194938 * (Q.psi_0p2 - 0.9734513) * (40.0 - Q.n_real_top40)
    if Q.log_sum_pt > 6.811175 and Q.sj3_z3 < 0.1903878:
        z += -5.307787 * (Q.log_sum_pt - 6.811175) * (0.1903878 - Q.sj3_z3)
    if Q.log_sum_pt > 6.811175 and Q.mean_eta2 < 0.01204531:
        z += -274.5967 * (Q.log_sum_pt - 6.811175) * (0.01204531 - Q.mean_eta2)
    if Q.LHA > 0.404204 and Q.C2_b2 < 0.0329485:
        z += 625.8904 * (Q.LHA - 0.404204) * (0.0329485 - Q.C2_b2)
    if Q.n_dr_0_0p05 > 20.0 and Q.soft4_pt < 3.259863:
        z += 0.006097942 * (Q.n_dr_0_0p05 - 20.0) * (3.259863 - Q.soft4_pt)
    if Q.e3 < 4.646151e-05 and Q.z_1 < 0.2140047:
        z += -35847.57 * (4.646151e-05 - Q.e3) * (0.2140047 - Q.z_1)
    if Q.sum_pt < 1167.447 and Q.M2 < 0.05910514:
        z += -0.04931819 * (1167.447 - Q.sum_pt) * (0.05910514 - Q.M2)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.040438
    if Q.sum_pt < 1085.125:
        z += -0.1685273 * Q.sum_pt + 182.725
    if 1085.125 <= Q.sum_pt < 1115.723:
        z += 0.004844338 * Q.sum_pt - 5.404938
    if Q.sum_pt >= 1167.447:
        z += -0.01609734 * Q.sum_pt + 18.79279
    if 0.005383629 <= Q.girth2_top15 < 0.00727763:
        z += 134.6299 * Q.girth2_top15 - 0.7247976
    if 0.00727763 <= Q.girth2_top15 < 0.009962397:
        z += -15.63972 * Q.girth2_top15 + 0.3688093
    if 0.009962397 <= Q.girth2_top15 < 0.01563836:
        z += -83.22521 * Q.girth2_top15 + 1.042123
    if 0.01563836 <= Q.girth2_top15 < 0.02675364:
        z += -139.9057 * Q.girth2_top15 + 1.928513
    if Q.girth2_top15 >= 0.02675364:
        z += -218.8493 * Q.girth2_top15 + 4.040541
    if Q.n_pt_above_5 < 23.0:
        z += 0.03266124 * Q.n_pt_above_5 - 0.7512086
    if Q.n_dr_0p1_0p2 < 21.0:
        z += 0.01073837 * Q.n_dr_0p1_0p2 - 0.2791977
    if 21.0 <= Q.n_dr_0p1_0p2 < 26.0:
        z += 0.03525542 * Q.n_dr_0p1_0p2 - 0.7940557
    if Q.n_dr_0p1_0p2 >= 26.0:
        z += 0.02451705 * Q.n_dr_0p1_0p2 - 0.514858
    if Q.log_sum_pt < 6.959294:
        z += 200.6845 * Q.log_sum_pt - 1402.153
    if 6.959294 <= Q.log_sum_pt < 6.98945:
        z += 183.3712 * Q.log_sum_pt - 1281.664
    if 7.062574 <= Q.log_sum_pt < 7.139296:
        z += 16.61615 * Q.log_sum_pt - 117.3528
    if Q.log_sum_pt >= 7.139296:
        z += 11.64883 * Q.log_sum_pt - 81.88964
    if 0.08589404 <= Q.girth < 0.09749958:
        z += -46.43249 * Q.girth + 3.988274
    if 0.09749958 <= Q.girth < 0.1207452:
        z += -115.2951 * Q.girth + 10.70235
    if 0.1207452 <= Q.girth < 0.1402186:
        z += -141.3613 * Q.girth + 13.84971
    if 0.1402186 <= Q.girth < 0.1564779:
        z += -161.7596 * Q.girth + 16.70994
    if Q.girth >= 0.1564779:
        z += -209.1623 * Q.girth + 24.12741
    if Q.LHA >= 0.3203321:
        z += 32.65199 * Q.LHA - 10.45948
    if Q.e2 >= 0.04082832:
        z += 46.03777 * Q.e2 - 1.879645
    if Q.sum_pt_top50 < 1048.098:
        z += -0.0172144 * Q.sum_pt_top50 + 18.04238
    if Q.e3 < 0.0001841806:
        z += -1522.631 * Q.e3 + 1.248052
    if 0.0001841806 <= Q.e3 < 0.0005178279:
        z += -2900.108 * Q.e3 + 1.501757
    if Q.psi_0p2 >= 0.9087063:
        z += -4.357098 * Q.psi_0p2 + 3.959322
    if Q.sum_pt_top30 >= 1111.245:
        z += 0.005413201 * Q.sum_pt_top30 - 6.015393
    if Q.sum_pt_top20 >= 956.5062:
        z += -0.003148903 * Q.sum_pt_top20 + 3.011945
    if Q.C2 >= 0.06655881:
        z += -3.299747 * Q.C2 + 0.2196272
    if Q.tau4 >= 0.01517184:
        z += 9.705932 * Q.tau4 - 0.1472568
    if Q.girth2_top5 >= 0.00422683:
        z += 22.56456 * Q.girth2_top5 - 0.09537657
    if Q.pt_9 < 41.4375:
        z += -0.00535544 * Q.pt_9 + 0.221916
    if Q.z_dr_0p1_0p2 < 0.1872805:
        z += 0.8227904 * Q.z_dr_0p1_0p2 - 0.1540926
    if Q.pt_13 < 13.46875:
        z += 0.01579931 * Q.pt_13 - 0.212797
    if Q.z_top15_slots < 0.9431554:
        z += 0.5983342 * Q.z_top15_slots - 0.5643221
    if Q.girth2_top15 > 0.009962397 and Q.sum_pt_top40 < 1225.842:
        z += 0.3716128 * (Q.girth2_top15 - 0.009962397) * (1225.842 - Q.sum_pt_top40)
    if Q.n_pt_above_5 < 23.0 and Q.D2 < 3.345339:
        z += 0.02419163 * (23.0 - Q.n_pt_above_5) * (3.345339 - Q.D2)
    if Q.sum_pt < 1085.125 and Q.log_sum_pt < 6.811175:
        z += -0.0110208 * (1085.125 - Q.sum_pt) * (6.811175 - Q.log_sum_pt)
    if Q.n_dr_0p1_0p2 < 26.0 and Q.sum_pt_top30 < 1027.303:
        z += 8.421921e-05 * (26.0 - Q.n_dr_0p1_0p2) * (1027.303 - Q.sum_pt_top30)
    if Q.girth > 0.09749958 and Q.sum_pt < 1167.447:
        z += 0.216949 * (Q.girth - 0.09749958) * (1167.447 - Q.sum_pt)
    if Q.log_sum_pt > 7.139296 and Q.psi_0p3 > 0.9777125:
        z += 281.084 * (Q.log_sum_pt - 7.139296) * (Q.psi_0p3 - 0.9777125)
    if Q.e3 < 0.0005178279 and Q.psi_0p3 < 0.9966167:
        z += -7766.747 * (0.0005178279 - Q.e3) * (0.9966167 - Q.psi_0p3)
    if Q.sum_pt_top50 < 1048.098 and Q.girth2_top2 < 0.0003125151:
        z += 21.1202 * (1048.098 - Q.sum_pt_top50) * (0.0003125151 - Q.girth2_top2)
    if Q.sum_pt < 1085.125 and Q.planar_flow < 0.6025827:
        z += -0.004718444 * (1085.125 - Q.sum_pt) * (0.6025827 - Q.planar_flow)
    if Q.e3 < 0.0001841806 and Q.pt_9 < 41.4375:
        z += -65.92927 * (0.0001841806 - Q.e3) * (41.4375 - Q.pt_9)
    if Q.sum_pt < 1085.125 and Q.sd_zg > 0.1634067:
        z += -0.004464824 * (1085.125 - Q.sum_pt) * (Q.sd_zg - 0.1634067)
    if Q.log_sum_pt > 7.139296 and Q.phi_0 < -0.0297699:
        z += -414.2753 * (Q.log_sum_pt - 7.139296) * (-0.0297699 - Q.phi_0)
    if Q.log_sum_pt > 7.139296 and Q.phi_0 < -0.04013062:
        z += 307.5515 * (Q.log_sum_pt - 7.139296) * (-0.04013062 - Q.phi_0)
    if Q.log_sum_pt > 7.139296 and Q.phi_0 < -0.004917145:
        z += 87.1496 * (Q.log_sum_pt - 7.139296) * (-0.004917145 - Q.phi_0)
    if Q.log_sum_pt > 7.139296 and Q.dr_12 < 0.04736241:
        z += -60.67735 * (Q.log_sum_pt - 7.139296) * (0.04736241 - Q.dr_12)
    if Q.sum_pt > 1167.447 and Q.abseta_12 < 0.06868286:
        z += 0.01841118 * (Q.sum_pt - 1167.447) * (0.06868286 - Q.abseta_12)
    if Q.sum_pt > 1167.447 and Q.eta_1 < 0.002292633:
        z += 0.1262519 * (Q.sum_pt - 1167.447) * (0.002292633 - Q.eta_1)
    if Q.sum_pt_top50 < 959.0957 and Q.dr0_7 > 0.1350754:
        z += -0.02672889 * (959.0957 - Q.sum_pt_top50) * (Q.dr0_7 - 0.1350754)
    if Q.log_sum_pt > 7.062574 and Q.abseta_9 > 0.1726074:
        z += -63.97874 * (Q.log_sum_pt - 7.062574) * (Q.abseta_9 - 0.1726074)
    if Q.log_sum_pt > 7.062574 and Q.eta_1 < 0.0001021922:
        z += -180.2059 * (Q.log_sum_pt - 7.062574) * (0.0001021922 - Q.eta_1)
    if Q.girth > 0.1402186 and Q.soft6_pt > 3.328125:
        z += -58.07465 * (Q.girth - 0.1402186) * (Q.soft6_pt - 3.328125)
    if Q.girth > 0.1402186 and Q.soft6_z > 0.002707742:
        z += 124230.8 * (Q.girth - 0.1402186) * (Q.soft6_z - 0.002707742)
    if Q.girth > 0.1402186 and Q.soft6_pt > 2.162109:
        z += -79.65651 * (Q.girth - 0.1402186) * (Q.soft6_pt - 2.162109)
    if Q.girth2_top15 > 0.005383629 and Q.soft6_pt > 1.740234:
        z += -20.11386 * (Q.girth2_top15 - 0.005383629) * (Q.soft6_pt - 1.740234)
    if Q.sum_pt < 1115.723 and Q.e4 < 2.0948e-08:
        z += -140366.2 * (1115.723 - Q.sum_pt) * (2.0948e-08 - Q.e4)
    if Q.log_sum_pt < 6.959294 and Q.C3 < 0.01311308:
        z += 368.0574 * (6.959294 - Q.log_sum_pt) * (0.01311308 - Q.C3)
    return max(0.0, z)


def neuron_14(Q):
    z = 2.241729
    if Q.tau21_b2 < 0.1219401:
        z += -5.118292 * Q.tau21_b2 + 1.15087
    if 0.1219401 <= Q.tau21_b2 < 0.342495:
        z += -2.388273 * Q.tau21_b2 + 0.8179716
    if Q.n_dr_0p1_0p2 < 13.0:
        z += 0.0284185 * Q.n_dr_0p1_0p2 - 0.5321473
    if 13.0 <= Q.n_dr_0p1_0p2 < 17.0:
        z += 0.04067671 * Q.n_dr_0p1_0p2 - 0.6915041
    if Q.n_dr_0p1_0p2 >= 21.0:
        z += -0.01163119 * Q.n_dr_0p1_0p2 + 0.244255
    if Q.girth < 0.07031778:
        z += 78.15922 * Q.girth - 5.495983
    if 0.076787 <= Q.girth < 0.08068193:
        z += -41.2216 * Q.girth + 3.165283
    if 0.08068193 <= Q.girth < 0.08589404:
        z += -91.99234 * Q.girth + 7.261564
    if 0.08589404 <= Q.girth < 0.09749958:
        z += -139.6162 * Q.girth + 11.35217
    if 0.09749958 <= Q.girth < 0.1207452:
        z += -35.46958 * Q.girth + 1.197919
    if Q.girth >= 0.1207452:
        z += -86.9841 * Q.girth + 7.418049
    if Q.girth2_top10 >= 0.0001295334:
        z += -25.15173 * Q.girth2_top10 + 0.003257989
    if Q.e3 < 3.376709e-05:
        z += -11844.19 * Q.e3 + 0.001047793
    if 3.376709e-05 <= Q.e3 < 7.876005e-05:
        z += 8865.742 * Q.e3 - 0.6982663
    if 0.9638082 <= Q.psi_0p3 < 0.9924477:
        z += 12.77736 * Q.psi_0p3 - 12.31492
    if Q.psi_0p3 >= 0.9924477:
        z += 55.82896 * Q.psi_0p3 - 55.04138
    if Q.e2 < 0.03480688:
        z += -36.73516 * Q.e2 + 1.278636
    if Q.tau1 < 0.1507173:
        z += 20.13664 * Q.tau1 - 3.251406
    if 0.1507173 <= Q.tau1 < 0.1751567:
        z += 8.85726 * Q.tau1 - 1.551408
    if Q.z_dr_0_0p05 >= 0.6283153:
        z += 1.493479 * Q.z_dr_0_0p05 - 0.9383755
    if Q.n_real_top50 >= 43.0:
        z += -0.01608244 * Q.n_real_top50 + 0.6915451
    if Q.z_dr_0p2_0p4 < 0.003398536:
        z += 48.81915 * Q.z_dr_0p2_0p4 - 0.1659136
    if Q.soft9_z < 0.001838217:
        z += 58.8531 * Q.soft9_z - 0.1081848
    if Q.dr_6 < 0.05347848:
        z += -8.253487 * Q.dr_6 + 0.4413839
    if Q.z_dr_0p05_0p1 >= 0.7108211:
        z += -1.453842 * Q.z_dr_0p05_0p1 + 1.033421
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -2.747138 * Q.z_dr_0p1_0p2 + 0.3306006
    if Q.LHA < 0.09165437:
        z += -16.92127 * Q.LHA + 3.307966
    if 0.09165437 <= Q.LHA < 0.2601462:
        z += -17.89982 * Q.LHA + 3.397654
    if 0.2601462 <= Q.LHA < 0.2941033:
        z += -8.412462 * Q.LHA + 0.9295533
    if 0.2941033 <= Q.LHA < 0.302389:
        z += 14.04694 * Q.LHA - 5.675831
    if 0.302389 <= Q.LHA < 0.3332345:
        z += 38.63744 * Q.LHA - 13.11173
    if Q.LHA >= 0.3332345:
        z += -0.9785498 * Q.LHA + 0.08968836
    if Q.sd_rg < 0.15159:
        z += -3.039523 * Q.sd_rg + 0.9170684
    if 0.15159 <= Q.sd_rg < 0.1596365:
        z += 0.9625249 * Q.sd_rg + 0.3103979
    if 0.1596365 <= Q.sd_rg < 0.2042612:
        z += 10.73104 * Q.sd_rg - 1.249014
    if 0.2042612 <= Q.sd_rg < 0.3017146:
        z += -1.47246 * Q.sd_rg + 1.243688
    if Q.sd_rg >= 0.3017146:
        z += 1.567063 * Q.sd_rg + 0.3266196
    if Q.M3 < 0.03787151:
        z += -4.722955 * Q.M3 + 0.1788654
    if Q.C2 < 0.07996447:
        z += 9.474333 * Q.C2 - 0.75761
    if Q.lam2 < 0.001776308:
        z += -138.2249 * Q.lam2 + 0.2455301
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.001264095 * Q.n_dr_0p2_0p4 + 0.6261619
    if 11.0 <= Q.n_dr_0p2_0p4 < 21.0:
        z += -0.06122569 * Q.n_dr_0p2_0p4 + 1.285739
    if Q.zdr_0 < 0.009970338:
        z += 24.78631 * Q.zdr_0 - 0.2471279
    if Q.dr_0 < 0.06413297:
        z += -2.426957 * Q.dr_0 + 0.155648
    if Q.sj3_dr_max < 0.1717401:
        z += 2.666989 * Q.sj3_dr_max - 0.9909516
    if 0.1717401 <= Q.sj3_dr_max < 0.1999777:
        z += 7.896558 * Q.sj3_dr_max - 1.889078
    if 0.1999777 <= Q.sj3_dr_max < 0.2628766:
        z += 0.9357448 * Q.sj3_dr_max - 0.4970708
    if 0.2628766 <= Q.sj3_dr_max < 0.3715619:
        z += -2.444198 * Q.sj3_dr_max + 0.3914371
    if Q.sj3_dr_max >= 0.3715619:
        z += -5.111187 * Q.sj3_dr_max + 1.382389
    if Q.tau21_b2 < 0.342495 and Q.girth2_top15 < 0.00727763:
        z += -1589.627 * (0.342495 - Q.tau21_b2) * (0.00727763 - Q.girth2_top15)
    if Q.tau21_b2 < 0.342495 and Q.girth2_top10 < 0.01976735:
        z += 56.71111 * (0.342495 - Q.tau21_b2) * (0.01976735 - Q.girth2_top10)
    if Q.tau21_b2 < 0.342495 and Q.girth2_top15 < 0.006142802:
        z += 1177.459 * (0.342495 - Q.tau21_b2) * (0.006142802 - Q.girth2_top15)
    if Q.tau21_b2 < 0.342495 and Q.sum_pt_top50 < 1156.659:
        z += -0.008506448 * (0.342495 - Q.tau21_b2) * (1156.659 - Q.sum_pt_top50)
    if Q.psi_0p3 > 0.9924477 and Q.dr_6 < 0.05347848:
        z += -1628.369 * (Q.psi_0p3 - 0.9924477) * (0.05347848 - Q.dr_6)
    if Q.n_dr_0p1_0p2 < 17.0 and Q.pt1_dr01 < 17.82896:
        z += 0.0008320753 * (17.0 - Q.n_dr_0p1_0p2) * (17.82896 - Q.pt1_dr01)
    if Q.girth > 0.09749958 and Q.max_dr < 0.4021783:
        z += -226.3335 * (Q.girth - 0.09749958) * (0.4021783 - Q.max_dr)
    if Q.girth > 0.076787 and Q.max_dr < 0.4021783:
        z += 86.03368 * (Q.girth - 0.076787) * (0.4021783 - Q.max_dr)
    if Q.tau21_b2 < 0.342495 and Q.soft1_pt < 2.275391:
        z += 0.3149838 * (0.342495 - Q.tau21_b2) * (2.275391 - Q.soft1_pt)
    if Q.girth > 0.1207452 and Q.max_dr < 0.4021783:
        z += 285.5416 * (Q.girth - 0.1207452) * (0.4021783 - Q.max_dr)
    if Q.sd_rg > 0.15159 and Q.n_pt_above_50 > 4.0:
        z += -0.8491694 * (Q.sd_rg - 0.15159) * (Q.n_pt_above_50 - 4.0)
    if Q.n_real_top50 > 43.0 and Q.max_dr < 0.4357228:
        z += 0.1645145 * (Q.n_real_top50 - 43.0) * (0.4357228 - Q.max_dr)
    if Q.tau21_b2 < 0.342495 and Q.n_pt_above_10 > 23.0:
        z += -0.1890563 * (0.342495 - Q.tau21_b2) * (Q.n_pt_above_10 - 23.0)
    if Q.C2 < 0.07996447 and Q.pt_3 > 55.15625:
        z += 0.06782069 * (0.07996447 - Q.C2) * (Q.pt_3 - 55.15625)
    if Q.girth > 0.08589404 and Q.n_real_top40 < 36.0:
        z += 5.94588 * (Q.girth - 0.08589404) * (36.0 - Q.n_real_top40)
    if Q.girth2_top10 > 0.0001295334 and Q.n_real_top40 < 36.0:
        z += -3.48666 * (Q.girth2_top10 - 0.0001295334) * (36.0 - Q.n_real_top40)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.112142
    if Q.z_dr_0_0p05 < 0.8459004:
        z += -0.3014418 * Q.z_dr_0_0p05 + 0.2549897
    if Q.z_dr_0p1_0p2 < 0.04748396:
        z += -10.44926 * Q.z_dr_0p1_0p2 + 0.6551009
    if 0.04748396 <= Q.z_dr_0p1_0p2 < 0.1203437:
        z += -2.181298 * Q.z_dr_0p1_0p2 + 0.2625054
    if Q.girth2_top5 < 0.008329695:
        z += -29.56869 * Q.girth2_top5 + 0.2462982
    if 6.915514 <= Q.log_sum_pt < 7.017258:
        z += -9.966168 * Q.log_sum_pt + 68.92117
    if Q.log_sum_pt >= 7.017258:
        z += -2.912281 * Q.log_sum_pt + 19.42223
    if Q.sum_pt >= 907.9372:
        z += 0.003689767 * Q.sum_pt - 3.350077
    if Q.psi_0p3 >= 0.9896594:
        z += -22.48116 * Q.psi_0p3 + 22.24869
    if Q.n_dr_0p2_0p4 >= 8.0:
        z += -0.02403283 * Q.n_dr_0p2_0p4 + 0.1922626
    if Q.girth2_top3 < 0.001155057:
        z += 300.1259 * Q.girth2_top3 - 0.49918
    if 0.001155057 <= Q.girth2_top3 < 0.002151568:
        z += 153.0515 * Q.girth2_top3 - 0.3293006
    if 0.707925 <= Q.psi_0p1 < 0.9184255:
        z += 0.9534768 * Q.psi_0p1 - 0.67499
    if Q.psi_0p1 >= 0.9184255:
        z += -7.788511 * Q.psi_0p1 + 7.353875
    if Q.girth2_top10 < 0.003213724:
        z += 16.14876 * Q.girth2_top10 + 0.2511849
    if 0.003213724 <= Q.girth2_top10 < 0.007678544:
        z += -67.88239 * Q.girth2_top10 + 0.5212379
    if 886.3438 <= Q.sum_pt_top30 < 1038.262:
        z += 0.003658546 * Q.sum_pt_top30 - 3.24273
    if Q.sum_pt_top30 >= 1038.262:
        z += 0.001192753 * Q.sum_pt_top30 - 0.6825906
    if 0.1512157 <= Q.sj2_dr < 0.2232169:
        z += -1.808165 * Q.sj2_dr + 0.273423
    if 0.2232169 <= Q.sj2_dr < 0.2780918:
        z += 8.108454 * Q.sj2_dr - 1.940134
    if Q.sj2_dr >= 0.2780918:
        z += 4.344213 * Q.sj2_dr - 0.8933297
    if Q.lam2 < 0.001776308:
        z += -217.7326 * Q.lam2 + 0.3867602
    if Q.D2 < 2.410481:
        z += 0.2533494 * Q.D2 - 0.610694
    if Q.tau1 < 0.07708632:
        z += 25.99217 * Q.tau1 - 2.003641
    if Q.tau1 >= 0.1953848:
        z += -69.59547 * Q.tau1 + 13.5979
    if Q.sum_pt_top40 >= 1001.523:
        z += -0.003496458 * Q.sum_pt_top40 + 3.501784
    if Q.sd_rg < 0.1690338:
        z += -0.918988 * Q.sd_rg - 0.002293414
    if 0.1690338 <= Q.sd_rg < 0.1881908:
        z += -6.55522 * Q.sd_rg + 0.9504205
    if 0.1881908 <= Q.sd_rg < 0.3017146:
        z += 2.494734 * Q.sd_rg - 0.7526976
    if Q.soft5_z < 0.0004140594:
        z += -1609.732 * Q.soft5_z + 0.6665246
    if Q.girth < 0.08068193:
        z += -14.70743 * Q.girth + 1.186624
    if Q.soft5_pt < 0.5297852:
        z += 0.7953386 * Q.soft5_pt - 0.4213586
    if Q.z_dr_0p2_0p4 < 0.02647293:
        z += 9.032409 * Q.z_dr_0p2_0p4 - 0.2391143
    if Q.tau21_b2 < 0.342495:
        z += -1.18425 * Q.tau21_b2 + 0.4055997
    if Q.girth2_top15 < 0.004169954:
        z += -33.80762 * Q.girth2_top15 - 0.04738267
    if 0.004169954 <= Q.girth2_top15 < 0.009962397:
        z += 32.51804 * Q.girth2_top15 - 0.3239576
    if Q.e2 < 0.02793599:
        z += -2.091877 * Q.e2 + 0.4140467
    if 0.02793599 <= Q.e2 < 0.04358622:
        z += -22.72223 * Q.e2 + 0.9903763
    if Q.n_dr_0p05_0p1 >= 6.0:
        z += -0.008443981 * Q.n_dr_0p05_0p1 + 0.05066389
    if Q.LHA >= 0.302389:
        z += -9.372112 * Q.LHA + 2.834024
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.tau21_b2 < 0.4823776:
        z += 2.509096 * (0.1203437 - Q.z_dr_0p1_0p2) * (0.4823776 - Q.tau21_b2)
    if Q.girth2_top5 < 0.008329695 and Q.n_real_top40 < 40.0:
        z += 1.288064 * (0.008329695 - Q.girth2_top5) * (40.0 - Q.n_real_top40)
    if Q.girth2_top5 < 0.008329695 and Q.sum_pt_top2 < 689.25:
        z += -0.05337192 * (0.008329695 - Q.girth2_top5) * (689.25 - Q.sum_pt_top2)
    if Q.sj2_dr > 0.2232169 and Q.sj3_dr23 > 0.2563461:
        z += -7.279315 * (Q.sj2_dr - 0.2232169) * (Q.sj3_dr23 - 0.2563461)
    if Q.psi_0p3 > 0.9896594 and Q.tau21_b2 < 0.4299592:
        z += -132.8769 * (Q.psi_0p3 - 0.9896594) * (0.4299592 - Q.tau21_b2)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.soft5_pt < 1.480811:
        z += -1.501998 * (0.1203437 - Q.z_dr_0p1_0p2) * (1.480811 - Q.soft5_pt)
    if Q.sj2_dr > 0.2232169 and Q.sj3_pairmin_over_m > 0.1581667:
        z += -9.026135 * (Q.sj2_dr - 0.2232169) * (Q.sj3_pairmin_over_m - 0.1581667)
    if Q.sd_rg < 0.1881908 and Q.eta_1 > -0.04302979:
        z += -11.59024 * (0.1881908 - Q.sd_rg) * (Q.eta_1 - -0.04302979)
    if Q.sj2_dr > 0.2232169 and Q.dr_4 < 0.03006824:
        z += -139.9199 * (Q.sj2_dr - 0.2232169) * (0.03006824 - Q.dr_4)
    if Q.psi_0p1 > 0.707925 and Q.dr_max_012 < 0.1828389:
        z += 6.870783 * (Q.psi_0p1 - 0.707925) * (0.1828389 - Q.dr_max_012)
    if Q.girth < 0.08068193 and Q.dr1_11 > 0.1911348:
        z += -44.24285 * (0.08068193 - Q.girth) * (Q.dr1_11 - 0.1911348)
    if Q.psi_0p1 > 0.707925 and Q.dr1_11 > 0.2195171:
        z += 13.54487 * (Q.psi_0p1 - 0.707925) * (Q.dr1_11 - 0.2195171)
    if Q.D2 < 2.410481 and Q.dr_11 < 0.199671:
        z += 0.6180229 * (2.410481 - Q.D2) * (0.199671 - Q.dr_11)
    if Q.z_dr_0p1_0p2 < 0.04748396 and Q.absphi_1 < 0.03105164:
        z += -282.9472 * (0.04748396 - Q.z_dr_0p1_0p2) * (0.03105164 - Q.absphi_1)
    if Q.psi_0p1 > 0.9184255 and Q.absphi_1 < 0.03105164:
        z += 150.6836 * (Q.psi_0p1 - 0.9184255) * (0.03105164 - Q.absphi_1)
    if Q.sj2_dr > 0.2232169 and Q.sj3_pairmin_over_m > 0.2777088:
        z += 6.591737 * (Q.sj2_dr - 0.2232169) * (Q.sj3_pairmin_over_m - 0.2777088)
    if Q.log_sum_pt > 7.017258 and Q.soft1_pt < 2.275391:
        z += 0.5362324 * (Q.log_sum_pt - 7.017258) * (2.275391 - Q.soft1_pt)
    if Q.log_sum_pt > 7.017258 and Q.soft10_pt < 6.70332:
        z += 0.232183 * (Q.log_sum_pt - 7.017258) * (6.70332 - Q.soft10_pt)
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
