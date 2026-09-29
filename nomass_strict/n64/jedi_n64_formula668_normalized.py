"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.4%   (on for 41% of jets)
  neuron  1:  12.3%   (on for 96% of jets)
  neuron  5:  11.3%   (on for 76% of jets)
  neuron  4:   9.9%   (on for 67% of jets)
  neuron  0:   7.7%   (on for 61% of jets)
  neuron 13:   7.3%   (on for 88% of jets)
  neuron  9:   6.6%   (on for 60% of jets)
  neuron 10:   6.2%   (on for 80% of jets)
  neuron 12:   5.6%   (on for 35% of jets)
  neuron  7:   4.2%   (on for 45% of jets)
  neuron  6:   3.8%   (on for 43% of jets)
  neuron  3:   3.7%   (on for 53% of jets)
  neuron 11:   3.4%   (on for 70% of jets)
  neuron 15:   2.2%   (on for 58% of jets)
  neuron 14:   1.9%   (on for 29% of jets)
  neuron  2:   0.6%   (on for 37% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.7% (the network: 81.1%); same class as the network for 92.8% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
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
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.ptdr0_10               pT10 · ΔR(0, 10) [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.ptdr0_12               pT12 · ΔR(0, 12) [GeV]
  Q.ptdr0_14               pT14 · ΔR(0, 14) [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_8                   pT of particle 8 [GeV]
  Q.ptdr0_8                pT8 · ΔR(0, 8) [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_0                    pT of particle 0 / total pT
  Q.z_14                   pT of particle 14 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_z                pT share of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_12              |Δη| of particle 12
  Q.abseta_9               |Δη| of particle 9
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_13              |Δφ| of particle 13
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft2_dr0              ΔR between the hardest and the 2. softest real particle (0 if among the 15 hardest)
  Q.dr0_12                 ΔR between particle 12 and the hardest particle
  Q.dr1_11                 ΔR between particle 11 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.soft2_dr               ΔR from the jet axis of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_dr               ΔR from the jet axis of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.phi_0                  Δφ of particle 0
  Q.phi_10                 Δφ of particle 10
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
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.mean_eta               pT-weighted mean Δη
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
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        ptdr0_10=pt[10] * math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        pt_11=pt[11],
        ptdr0_12=pt[12] * math.sqrt(dist2(0, 12)) if pt[12] > 0 else 0.0,
        ptdr0_14=pt[14] * math.sqrt(dist2(0, 14)) if pt[14] > 0 else 0.0,
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_8=pt[8],
        ptdr0_8=pt[8] * math.sqrt(dist2(0, 8)) if pt[8] > 0 else 0.0,
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        soft9_pt=softp(9, 'pt'),
        z_0=z[0],
        z_14=z[14],
        z_3=z[3],
        z_7=z[7],
        soft1_z=softp(1, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft9_z=softp(9, 'z'),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top3_slots=sum(pt[:3]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        sd_zg=softdrop("zg"),
        abseta_12=abs(eta[12]),
        abseta_9=abs(eta[9]),
        absphi_1=abs(phi[1]),
        absphi_13=abs(phi[13]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft2_dr0=softp(2, 'dr0'),
        dr0_12=math.sqrt(dist2(0, 12)) if pt[12] > 0 else 0.0,
        dr1_11=math.sqrt(dist2(1, 11)) if pt[11] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        soft2_dr=softp(2, 'dr'),
        soft6_dr=softp(6, 'dr'),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        phi_0=phi[0],
        phi_10=phi[10],
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
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
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
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 28.02;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.02365 * (-0.01571672
        + 0.1686681 * max(0.0, 0.08068193 - Q.sum_z_dr) / 0.02368455   # +16.9%  sum_z_dr < 0.08068
        - 0.1682003 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # -16.8%  LHA < 0.3098
        + 0.07229841 * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) / 0.00326329   # +7.2%  sum_z_dr2_top15 < 0.007888
        - 0.06675144 * max(0.0, Q.sj2_dr - 0.1512157) / 0.05988199   # -6.7%  sj2_dr > 0.1512
        + 0.05528988 * max(0.0, 0.3203321 - Q.LHA) / 0.07499257   # +5.5%  LHA < 0.3203
        - 0.05438107 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # -5.4%  sum_z_dr < 0.0975
        - 0.05145904 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # -5.1%  sum_z_dr2_top15 < 0.009962
        + 0.0496461 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +5.0%  LHA < 0.3332
        + 0.03427426 * max(0.0, Q.sj2_dr - 0.1411617) / 0.06721719   # +3.4%  sj2_dr > 0.1412
        + 0.03023958 * max(0.0, Q.sj2_dr - 0.1745007) / 0.04529905   # +3.0%  sj2_dr > 0.1745
        + 0.02952995 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_pt_top50 - 889.8503) / 1563.946   # +3.0%  n_dr_0p2_0p4 < 18 and sum_pt_top50 > 889.9
        - 0.02553237 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # -2.6%  dr_0 < 0.06413
        - 0.02175456 * max(0.0, Q.sj2_dr - 0.06289464) / 0.1327343   # -2.2%  sj2_dr > 0.06289
        + 0.01806147 * max(0.0, 0.05775119 - Q.dr_0) / 0.02089184   # +1.8%  dr_0 < 0.05775
        - 0.01792366 * max(0.0, 0.04466492 - Q.tau1) / 0.005217805   # -1.8%  tau1 < 0.04466
        - 0.01755463 * max(0.0, 0.005383629 - Q.sum_z_dr2_top15) / 0.001666876   # -1.8%  sum_z_dr2_top15 < 0.005384
        - 0.0170522 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # -1.7%  log_sum_pt > 6.894
        - 0.01449221 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.log_sum_pt - 6.920349) / 0.4496771   # -1.4%  n_dr_0p2_0p4 < 18 and log_sum_pt > 6.92
        + 0.01177385 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +1.2%  n_dr_0p2_0p4 < 18
        + 0.01108434 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # +1.1%  LHA < 0.187
        - 0.01081917 * max(0.0, 0.05048381 - Q.sum_z_dr) / 0.008533025   # -1.1%  sum_z_dr < 0.05048
        + 0.009984443 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +1.0%  psi_0p3 > 0.9897
        - 0.007190845 * max(0.0, 2.883342e-05 - Q.e3) / 6.471346e-06   # -0.7%  e3 < 2.883e-05
        - 0.006930333 * max(0.0, 0.1825048 - Q.sj2_dr) / 0.03087146   # -0.7%  sj2_dr < 0.1825
        + 0.004512673 * max(0.0, 0.4947602 - Q.z_dr_0_0p05) / 0.1759565   # +0.5%  z_dr_0_0p05 < 0.4948
        + 0.003666863 * max(0.0, 5.13841e-05 - Q.e3) / 1.709557e-05   # +0.4%  e3 < 5.138e-05
        - 0.003561304 * max(0.0, 0.1623049 - Q.sj3_dr_max) / 0.01012502   # -0.4%  sj3_dr_max < 0.1623
        + 0.003465939 * max(0.0, 0.2125209 - Q.sj3_dr_max) / 0.0267777   # +0.3%  sj3_dr_max < 0.2125
        - 0.002328299 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.tau21_b2 - 0.2018786) / 1.609082   # -0.2%  n_dr_0p2_0p4 < 18 and tau21_b2 > 0.2019
        + 0.002314265 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, 26.0 - Q.n_dr_0p1_0p2) / 0.02963093   # +0.2%  psi_0p3 > 0.9956 and n_dr_0p1_0p2 < 26
        + 0.002073225 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # +0.2%  log_sum_pt > 7.017
        + 0.001992349 * max(0.0, Q.sum_pt_top40 - 1001.523) * max(0.0, 0.1512157 - Q.sj2_dr) / 1.305424   # +0.2%  sum_pt_top40 > 1002 and sj2_dr < 0.1512
        - 0.001861684 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_z_dr2_top10 - 0.004752876) / 0.01391948   # -0.2%  n_dr_0p2_0p4 < 18 and sum_z_dr2_top10 > 0.004753
        + 0.001733545 * max(0.0, Q.sum_pt_top2 - 405.0) / 51.45192   # +0.2%  sum_pt_top2 > 405
        + 0.0008179951 * max(0.0, 0.1210264 - Q.sj3_dr_max) / 0.004637661   # +0.1%  sj3_dr_max < 0.121
        - 0.0007795837 * max(0.0, Q.sum_pt_top30 - 1052.08) / 20.84425   # -0.1%  sum_pt_top30 > 1052
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 16.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.07122 * (0.03093632
        + 0.1279272 * max(0.0, Q.sum_pt - 972.0419) / 81.34075   # +12.8%  sum_pt > 972
        + 0.10378 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +10.4%  log_sum_pt > 6.91
        - 0.08456355 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -8.5%  log_sum_pt < 7.139
        - 0.07368699 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # -7.4%  sum_pt_top50 > 934.2
        + 0.04848209 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +4.8%  n_particles > 38
        + 0.04600689 * max(0.0, 1225.842 - Q.sum_pt_top40) / 211.117   # +4.6%  sum_pt_top40 < 1226
        + 0.04179287 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # +4.2%  sum_pt_top50 < 1079
        - 0.03938721 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -3.9%  sum_pt_top50 > 959.1
        - 0.03165109 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -3.2%  sum_pt > 1053
        + 0.02844783 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +2.8%  tau1 < 0.07709
        + 0.02760245 * max(0.0, 605.875 - Q.sum_pt_top2) / 245.9717   # +2.8%  sum_pt_top2 < 605.9
        - 0.02235423 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.002181998 - Q.soft1_z) / 0.01445558   # -2.2%  n_particles > 38 and soft1_z < 0.002182
        + 0.02144374 * max(0.0, Q.z_top30_slots - 0.9341838) / 0.03406818   # +2.1%  z_top30_slots > 0.9342
        - 0.01940651 * max(0.0, 0.02178815 - Q.tau4) / 0.006603045   # -1.9%  tau4 < 0.02179
        - 0.01919521 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -1.9%  log_sum_pt > 6.989
        + 0.0189947 * max(0.0, 0.006756161 - Q.sum_z_dr2_top3) / 0.003650633   # +1.9%  sum_z_dr2_top3 < 0.006756
        - 0.01650607 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 17.94219 - Q.ptdr0_3) / 0.4842322   # -1.7%  z_top30_slots > 0.9342 and ptdr0_3 < 17.94
        - 0.01558645 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # -1.6%  lam2 < 0.001776
        - 0.0146935 * max(0.0, 12.6865 - Q.pt1_dr01) * max(0.0, 0.3845054 - Q.soft2_dr) / 1.613797   # -1.5%  pt1_dr01 < 12.69 and soft2_dr < 0.3845
        + 0.0143932 * max(0.0, 1.521582 - Q.soft1_pt) / 0.9527349   # +1.4%  soft1_pt < 1.522
        - 0.01281947 * max(0.0, Q.tau1 - 0.1219132) / 0.01033631   # -1.3%  tau1 > 0.1219
        + 0.01231367 * max(0.0, 12.6865 - Q.pt1_dr01) * max(0.0, 0.4086381 - Q.soft2_dr0) / 1.741015   # +1.2%  pt1_dr01 < 12.69 and soft2_dr0 < 0.4086
        - 0.01168402 * max(0.0, 0.03457336 - Q.M3) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.0006031662   # -1.2%  M3 < 0.03457 and psi_0p3 > 0.9299
        - 0.01072117 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # -1.1%  psi_0p3 > 0.9966
        + 0.01038056 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +1.0%  LHA > 0.3332
        - 0.009098937 * max(0.0, 29.04688 - Q.pt_11) / 8.607263   # -0.9%  pt_11 < 29.05
        - 0.008623656 * max(0.0, Q.z_top5 - 0.6551948) * max(0.0, 28.39396 - Q.pt1_dr01) / 0.8332473   # -0.9%  z_top5 > 0.6552 and pt1_dr01 < 28.39
        - 0.007105911 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) * max(0.0, 10.0 - Q.n_dr_0p05_0p1) / 0.2065903   # -0.7%  z_dr_0_0p05 > 0.8103 and n_dr_0p05_0p1 < 10
        + 0.005418473 * max(0.0, 0.05602756 - Q.C2) / 0.009882647   # +0.5%  C2 < 0.05603
        + 0.005399889 * max(0.0, 0.002197765 - Q.sum_z_dr2_top15) / 0.0004509993   # +0.5%  sum_z_dr2_top15 < 0.002198
        + 0.00534137 * max(0.0, Q.n_dr_0_0p05 - 15.0) / 2.713461   # +0.5%  n_dr_0_0p05 > 15
        + 0.005168098 * max(0.0, 0.0006570502 - Q.sum_z_dr2_top5) / 0.0001395474   # +0.5%  sum_z_dr2_top5 < 0.0006571
        - 0.00516252 * max(0.0, Q.zdr_0 - 0.01113024) / 0.002624099   # -0.5%  zdr_0 > 0.01113
        + 0.005061236 * max(0.0, 0.3572263 - Q.N2) / 0.07385171   # +0.5%  N2 < 0.3572
        + 0.005040554 * max(0.0, 0.3572263 - Q.N2) * max(0.0, 0.7904923 - Q.pt2_over_pt0) / 0.02349909   # +0.5%  N2 < 0.3572 and pt2_over_pt0 < 0.7905
        - 0.005014289 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 11.6057 - Q.ptdr0_6) / 0.3086732   # -0.5%  z_top30_slots > 0.9342 and ptdr0_6 < 11.61
        - 0.004922067 * max(0.0, 0.8495689 - Q.D2_b2) / 0.1691494   # -0.5%  D2_b2 < 0.8496
        + 0.004726902 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1178619 - Q.absphi_1) / 0.8192292   # +0.5%  n_particles > 38 and absphi_1 < 0.1179
        + 0.004592542 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.009970338 - Q.zdr_0) / 0.03317992   # +0.5%  n_particles > 38 and zdr_0 < 0.00997
        + 0.004576055 * max(0.0, Q.n_particles - 38.0) * max(0.0, Q.tau32 - 0.3293142) / 3.693221   # +0.5%  n_particles > 38 and tau32 > 0.3293
        - 0.004439 * max(0.0, 7.0 - Q.n_dr_0p1_0p2) / 0.9740454   # -0.4%  n_dr_0p1_0p2 < 7
        - 0.004028765 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -0.4%  n_dr_0p2_0p4 < 7
        + 0.003696976 * max(0.0, Q.sum_pt_top30 - 1038.262) / 23.89568   # +0.4%  sum_pt_top30 > 1038
        + 0.003235799 * max(0.0, 1.0 - Q.z_top50_slots) / 0.007698523   # +0.3%  z_top50_slots < 1
        + 0.003144117 * max(0.0, 2.5 - Q.soft9_pt) * max(0.0, 0.1224381 - Q.dr_11) / 0.03966978   # +0.3%  soft9_pt < 2.5 and dr_11 < 0.1224
        - 0.003107924 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.3%  log_sum_pt > 7.063
        + 0.00293678 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.pt1_dr01 - 5.351077) / 0.1255531   # +0.3%  z_top30_slots > 0.9342 and pt1_dr01 > 5.351
        + 0.002749917 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.dr12 - 0.003946546) / 0.001521699   # +0.3%  z_top30_slots > 0.9342 and dr12 > 0.003947
        + 0.002747426 * max(0.0, 0.2140276 - Q.tau21) / 0.01160104   # +0.3%  tau21 < 0.214
        + 0.002139026 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_4 - 2.381691) / 0.0768104   # +0.2%  z_top30_slots > 0.9342 and ptdr0_4 > 2.382
        + 0.001770331 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_2 - 7.740999) / 0.07370506   # +0.2%  z_top30_slots > 0.9342 and ptdr0_2 > 7.741
        + 0.001738567 * max(0.0, 0.002197765 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.9966167) / 5.451807e-07   # +0.2%  sum_z_dr2_top15 < 0.002198 and psi_0p3 > 0.9966
        + 0.00158722 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # +0.2%  e2 > 0.05557
        + 0.001285724 * max(0.0, Q.z_top5 - 0.6551948) * max(0.0, Q.ptdr0_5 - 0.5368514) / 0.0440253   # +0.1%  z_top5 > 0.6552 and ptdr0_5 > 0.5369
        + 0.001233217 * max(0.0, Q.z_top15_slots - 0.9885666) / 0.0003255751   # +0.1%  z_top15_slots > 0.9886
        - 0.001086099 * max(0.0, Q.n_dr_0_0p05 - 30.0) / 0.212479   # -0.1%  n_dr_0_0p05 > 30
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 8.347;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.346784 * (-0.07662843
        + 0.1119735 * max(0.0, Q.sd_rg - 0.1596365) / 0.03553981   # +11.2%  sd_rg > 0.1596
        - 0.067543 * max(0.0, Q.z_top50_slots - 0.9586536) / 0.03427112   # -6.8%  z_top50_slots > 0.9587
        - 0.06614266 * max(0.0, Q.sd_rg - 0.2639816) / 0.00903147   # -6.6%  sd_rg > 0.264
        + 0.05156666 * max(0.0, Q.log_sum_pt - 6.930088) / 0.0397436   # +5.2%  log_sum_pt > 6.93
        + 0.04534308 * max(0.0, Q.sd_rg - 0.2639816) * max(0.0, 0.5912328 - Q.z_dr_0p05_0p1) / 0.003149033   # +4.5%  sd_rg > 0.264 and z_dr_0p05_0p1 < 0.5912
        - 0.04234346 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -4.2%  sum_pt > 1053
        + 0.04190263 * max(0.0, 0.01414829 - Q.sum_z_dr2_top10) / 0.00877908   # +4.2%  sum_z_dr2_top10 < 0.01415
        + 0.04082726 * max(0.0, Q.z_top15_slots - 0.6225177) / 0.2124803   # +4.1%  z_top15_slots > 0.6225
        + 0.03788783 * max(0.0, Q.log_sum_pt - 6.903423) / 0.05662275   # +3.8%  log_sum_pt > 6.903
        - 0.03700696 * max(0.0, Q.sum_pt_top50 - 997.0189) / 56.58644   # -3.7%  sum_pt_top50 > 997
        - 0.03614639 * max(0.0, Q.sd_rg - 0.1690338) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1) / 0.01876614   # -3.6%  sd_rg > 0.169 and z_dr_0p05_0p1 < 0.8509
        + 0.03166027 * max(0.0, Q.z_top40_slots - 0.9300465) / 0.0517789   # +3.2%  z_top40_slots > 0.93
        - 0.03062129 * max(0.0, Q.sd_rg - 0.1690338) / 0.03150941   # -3.1%  sd_rg > 0.169
        - 0.03036637 * max(0.0, Q.psi_0p2 - 0.9734513) / 0.0116938   # -3.0%  psi_0p2 > 0.9735
        - 0.02895006 * max(0.0, 0.03577037 - Q.sum_z_dr) / 0.004182456   # -2.9%  sum_z_dr < 0.03577
        - 0.02694556 * max(0.0, Q.sd_rg - 0.1596365) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1) / 0.0209016   # -2.7%  sd_rg > 0.1596 and z_dr_0p05_0p1 < 0.8509
        + 0.02540158 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.sum_z_dr2_top15) / 0.0009227558   # +2.5%  log_sum_pt > 6.903 and sum_z_dr2_top15 < 0.02147
        + 0.02387313 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.2002199 - Q.dr_max_012) / 0.007813213   # +2.4%  log_sum_pt > 6.903 and dr_max_012 < 0.2002
        + 0.02251931 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +2.3%  sum_pt_top40 > 1070
        - 0.02076704 * max(0.0, Q.sum_pt_top40 - 1018.698) / 38.17804   # -2.1%  sum_pt_top40 > 1019
        + 0.01829306 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # +1.8%  LHA < 0.187
        + 0.01586909 * max(0.0, 0.006615185 - Q.sum_z_dr2_top15) / 0.002375805   # +1.6%  sum_z_dr2_top15 < 0.006615
        - 0.01475357 * max(0.0, Q.sum_pt_top50 - 997.0189) * max(0.0, Q.C2 - 0.06655881) / 0.5584339   # -1.5%  sum_pt_top50 > 997 and C2 > 0.06656
        - 0.01323284 * max(0.0, Q.sum_pt_top30 - 996.8867) / 38.86433   # -1.3%  sum_pt_top30 > 996.9
        - 0.01140023 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.sum_z_dr2_top15 - 0.009962397) / 1.329441e-05   # -1.1%  log_sum_pt > 7.063 and sum_z_dr2_top15 > 0.009962
        + 0.01096423 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # +1.1%  psi_0p3 > 0.9966
        - 0.01069116 * max(0.0, Q.z_top20_slots - 0.8281581) * max(0.0, Q.z_top50_slots - 0.9995789) / 3.023732e-05   # -1.1%  z_top20_slots > 0.8282 and z_top50_slots > 0.9996
        - 0.01023645 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.1206357 - Q.dr_max_012) / 0.00414901   # -1.0%  log_sum_pt > 6.903 and dr_max_012 < 0.1206
        + 0.01003676 * max(0.0, Q.log_sum_pt - 6.930088) * max(0.0, Q.C2 - 0.06655881) / 0.0004521485   # +1.0%  log_sum_pt > 6.93 and C2 > 0.06656
        + 0.007889864 * max(0.0, Q.sum_pt_top50 - 997.0189) * max(0.0, Q.C2 - 0.1088881) / 0.1050644   # +0.8%  sum_pt_top50 > 997 and C2 > 0.1089
        + 0.007521089 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.8065577 - Q.tau21) / 0.01865613   # +0.8%  log_sum_pt > 6.903 and tau21 < 0.8066
        - 0.007386976 * max(0.0, Q.log_sum_pt - 6.930088) * max(0.0, Q.C2 - 0.1088881) / 9.583602e-05   # -0.7%  log_sum_pt > 6.93 and C2 > 0.1089
        - 0.007099243 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.7%  log_sum_pt > 7.063
        - 0.007007818 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.8747961 - Q.psi_0p1) / 0.005466701   # -0.7%  log_sum_pt > 6.903 and psi_0p1 < 0.8748
        + 0.006328398 * max(0.0, Q.sum_pt_top50 - 997.0189) * max(0.0, 0.006794973 - Q.zdr_4) / 0.2427126   # +0.6%  sum_pt_top50 > 997 and zdr_4 < 0.006795
        + 0.004479076 * max(0.0, Q.sum_pt_top20 - 1064.139) / 11.11947   # +0.4%  sum_pt_top20 > 1064
        - 0.003819863 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.dr_max_012 - 0.1206357) / 0.06050554   # -0.4%  sum_pt_top20 > 1129 and dr_max_012 > 0.1206
        + 0.003079984 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.z_dr_0p1_0p2 - 0.003353111) / 0.007404432   # +0.3%  log_sum_pt > 6.903 and z_dr_0p1_0p2 > 0.003353
        - 0.002560232 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.zdr_3 - 0.00453462) / 5.033547e-05   # -0.3%  log_sum_pt > 6.903 and zdr_3 > 0.004535
        + 0.002464092 * max(0.0, Q.sum_pt_top20 - 1129.275) / 6.499168   # +0.2%  sum_pt_top20 > 1129
        - 0.002183399 * max(0.0, Q.sum_pt_top15 - 1082.548) / 5.972549   # -0.2%  sum_pt_top15 > 1083
        - 0.001893802 * max(0.0, Q.sum_pt - 1052.889) * max(0.0, 0.5106729 - Q.z_dr_0p05_0p1) / 11.33525   # -0.2%  sum_pt > 1053 and z_dr_0p05_0p1 < 0.5107
        - 0.0006921545 * max(0.0, Q.sum_pt_top15 - 1082.548) * max(0.0, Q.C2 - 0.1423914) / 0.0003134975   # -0.1%  sum_pt_top15 > 1083 and C2 > 0.1424
        - 0.0003285627 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.08734131) / 0.003497383   # -0.0%  sum_pt_top20 > 1129 and eta_1 > 0.08734
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 1.691;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.690811 * (-0.2418342
        + 0.1996352 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +20.0%  n_particles < 46
        + 0.1405459 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # +14.1%  n_dr_0p1_0p2 < 17
        + 0.1345409 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +13.5%  n_dr_0p2_0p4 < 5
        + 0.1036434 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # +10.4%  n_dr_0p2_0p4 < 8
        + 0.08363298 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +8.4%  lam2 < 0.0006155
        - 0.08182704 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.e2 - 0.01036127) / 0.09655312   # -8.2%  n_particles < 46 and e2 > 0.01036
        - 0.06286858 * max(0.0, 0.347196 - Q.tau21) / 0.05239131   # -6.3%  tau21 < 0.3472
        - 0.05974358 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.1221251   # -6.0%  n_particles < 46 and psi_0p3 > 0.9777
        - 0.04829633 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 30.51645   # -4.8%  n_dr_0p2_0p4 < 8 and n_dr_0p05_0p1 > 2
        + 0.0375323 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.eccentricity - 0.8680812) / 3.480194e-05   # +3.8%  psi_0p3 > 0.998 and eccentricity > 0.8681
        - 0.02795319 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03029714 - Q.e2) / 0.006086253   # -2.8%  n_dr_0p2_0p4 < 5 and e2 < 0.0303
        - 0.01167734 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) * max(0.0, 1013.042 - Q.sum_pt_top40) / 41.6591   # -1.2%  n_dr_0p1_0p2 < 10 and sum_pt_top40 < 1013
        + 0.008103244 * max(0.0, Q.psi_0p2 - 0.9985434) / 0.00014398   # +0.8%  psi_0p2 > 0.9985
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 8.771;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.77112 * (0.2117175
        + 0.1089349 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # +10.9%  sum_z_dr2_top15 < 0.009962
        - 0.06402627 * max(0.0, Q.sj2_dr - 0.1825048) / 0.04110551   # -6.4%  sj2_dr > 0.1825
        - 0.06328275 * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) / 0.00326329   # -6.3%  sum_z_dr2_top15 < 0.007888
        + 0.05920533 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # +5.9%  sum_z_dr > 0.0975
        - 0.05758525 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # -5.8%  LHA > 0.3332
        + 0.05442113 * max(0.0, Q.sj2_dr - 0.1219342) / 0.08223311   # +5.4%  sj2_dr > 0.1219
        - 0.04564472 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -4.6%  e2 < 0.04359
        + 0.04191463 * max(0.0, Q.sj2_dr - 0.1411617) / 0.06721719   # +4.2%  sj2_dr > 0.1412
        - 0.04136419 * max(0.0, 1038.855 - Q.sum_pt_top50) / 38.10896   # -4.1%  sum_pt_top50 < 1039
        - 0.03745898 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # -3.7%  sum_z_dr < 0.0975
        - 0.03397404 * max(0.0, 0.07374472 - Q.sum_z_dr) / 0.01915593   # -3.4%  sum_z_dr < 0.07374
        + 0.03239539 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # +3.2%  LHA < 0.2454
        - 0.02946025 * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.02836816   # -2.9%  z_dr_0p2_0p4 < 0.0518
        - 0.02823587 * max(0.0, 0.005788041 - Q.sum_z_dr2_top15) / 0.001877672   # -2.8%  sum_z_dr2_top15 < 0.005788
        + 0.02637135 * max(0.0, 0.09749958 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.9973959) / 3.542375e-05   # +2.6%  sum_z_dr < 0.0975 and psi_0p3 > 0.9974
        - 0.02519472 * max(0.0, 0.006876086 - Q.sum_z_dr2_top10) / 0.002892387   # -2.5%  sum_z_dr2_top10 < 0.006876
        + 0.02423382 * max(0.0, 0.03263075 - Q.e2) / 0.008034085   # +2.4%  e2 < 0.03263
        - 0.02368543 * max(0.0, Q.n_particles - 26.0) / 20.24921   # -2.4%  n_particles > 26
        - 0.02120532 * max(0.0, 3.355186 - Q.pt_entropy) / 0.5477678   # -2.1%  pt_entropy < 3.355
        - 0.01731145 * max(0.0, 0.05660088 - Q.sum_z_dr) / 0.01080382   # -1.7%  sum_z_dr < 0.0566
        - 0.01641396 * max(0.0, 0.05660088 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.9973959) / 8.756067e-06   # -1.6%  sum_z_dr < 0.0566 and psi_0p3 > 0.9974
        + 0.01625268 * max(0.0, Q.psi_0p2 - 0.8706159) / 0.08984765   # +1.6%  psi_0p2 > 0.8706
        - 0.01481491 * max(0.0, Q.n_pt_above_1 - 32.0) / 11.7996   # -1.5%  n_pt_above_1 > 32
        - 0.014616 * max(0.0, Q.soft1_pt - 1.521582) / 0.1002024   # -1.5%  soft1_pt > 1.522
        + 0.01207653 * max(0.0, 0.004169954 - Q.sum_z_dr2_top15) / 0.001130716   # +1.2%  sum_z_dr2_top15 < 0.00417
        - 0.01028096 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -1.0%  psi_0p1 > 0.8976
        - 0.009269173 * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) * max(0.0, Q.soft9_z - 0.0007402181) / 5.306836e-05   # -0.9%  z_dr_0p2_0p4 < 0.0518 and soft9_z > 0.0007402
        + 0.009223299 * max(0.0, 1011.524 - Q.sum_pt_top30) / 51.02618   # +0.9%  sum_pt_top30 < 1012
        + 0.008911663 * max(0.0, Q.soft1_pt - 1.091797) / 0.1522878   # +0.9%  soft1_pt > 1.092
        - 0.008299386 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # -0.8%  e2 < 0.01879
        + 0.007510092 * max(0.0, Q.soft9_pt - 1.458984) / 1.15228   # +0.8%  soft9_pt > 1.459
        - 0.006599553 * max(0.0, Q.n_particles - 26.0) * max(0.0, Q.soft1_pt - 0.4909668) / 9.262955   # -0.7%  n_particles > 26 and soft1_pt > 0.491
        + 0.006057749 * max(0.0, Q.soft1_pt - 2.275391) / 0.04593653   # +0.6%  soft1_pt > 2.275
        + 0.006037877 * max(0.0, Q.n_dr_0_0p05 - 14.0) / 3.092148   # +0.6%  n_dr_0_0p05 > 14
        - 0.00494551 * max(0.0, Q.sum_z_dr2_top15 - 0.004855289) * max(0.0, 0.4090302 - Q.sj3_pairmin_over_m) / 0.0003619533   # -0.5%  sum_z_dr2_top15 > 0.004855 and sj3_pairmin_over_m < 0.409
        - 0.004928882 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # -0.5%  e2 > 0.05557
        - 0.002934116 * max(0.0, 0.004169954 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 8.516876e-07   # -0.3%  sum_z_dr2_top15 < 0.00417 and psi_0p3 > 0.9974
        + 0.002549282 * max(0.0, Q.n_particles - 26.0) * max(0.0, Q.lam2 - 0.003687605) / 0.01132634   # +0.3%  n_particles > 26 and lam2 > 0.003688
        + 0.002372555 * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) * max(0.0, Q.sum_pt_top40 - 1095.686) / 0.09612434   # +0.2%  sum_z_dr2_top15 < 0.007888 and sum_pt_top40 > 1096
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 13.79;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.79152 * (-0.3240145
        + 0.273413 * max(0.0, 0.1564779 - Q.sum_z_dr) / 0.08789501   # +27.3%  sum_z_dr < 0.1565
        + 0.1439997 * max(0.0, Q.sum_z_dr - 0.02783745) / 0.04381821   # +14.4%  sum_z_dr > 0.02784
        + 0.07024203 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +7.0%  n_particles < 64
        - 0.05429899 * max(0.0, Q.LHA - 0.1632346) / 0.1084458   # -5.4%  LHA > 0.1632
        - 0.02910409 * max(0.0, Q.sum_z_dr2_top10 - 0.008956554) / 0.002278745   # -2.9%  sum_z_dr2_top10 > 0.008957
        + 0.02665432 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # +2.7%  sum_pt < 1085
        - 0.02603304 * max(0.0, 0.08068193 - Q.sum_z_dr) / 0.02368455   # -2.6%  sum_z_dr < 0.08068
        + 0.02568178 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +2.6%  n_dr_0p2_0p4 < 11
        + 0.02477354 * Q.z_top3_slots / 0.4446698   # +2.5%  z_top3_slots
        - 0.02325965 * max(0.0, 58.0 - Q.n_pt_above_1) / 17.32351   # -2.3%  n_pt_above_1 < 58
        - 0.02154819 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 7.684973e-07   # -2.2%  sum_pt < 1002 and e4 < 5.851e-08
        - 0.01971855 * max(0.0, 907.9372 - Q.sum_pt) / 3.961002   # -2.0%  sum_pt < 907.9
        - 0.0185681 * max(0.0, 6.893714 - Q.log_sum_pt) / 0.0132811   # -1.9%  log_sum_pt < 6.894
        + 0.01833226 * max(0.0, 1013.042 - Q.sum_pt_top40) / 31.91616   # +1.8%  sum_pt_top40 < 1013
        - 0.01746129 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -1.7%  sum_pt < 1002
        - 0.0166386 * max(0.0, Q.z_top30_slots - 0.9460751) / 0.02601016   # -1.7%  z_top30_slots > 0.9461
        - 0.01532808 * max(0.0, Q.n_dr_0p1_0p2 - 8.0) / 5.88681   # -1.5%  n_dr_0p1_0p2 > 8
        - 0.01489827 * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.02836816   # -1.5%  z_dr_0p2_0p4 < 0.0518
        - 0.01262577 * max(0.0, Q.LHA - 0.4331369) / 0.001006142   # -1.3%  LHA > 0.4331
        - 0.009897379 * max(0.0, 0.985099 - Q.z_top50_slots) / 0.003556075   # -1.0%  z_top50_slots < 0.9851
        + 0.009196411 * max(0.0, Q.sum_z_dr2_top10 - 0.005337976) / 0.003351803   # +0.9%  sum_z_dr2_top10 > 0.005338
        + 0.009150381 * max(0.0, 0.1564779 - Q.sum_z_dr) * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 0.1706859   # +0.9%  sum_z_dr < 0.1565 and n_dr_0p1_0p2 > 11
        + 0.009013303 * max(0.0, Q.LHA - 0.302389) / 0.02201444   # +0.9%  LHA > 0.3024
        + 0.008937832 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 3.38926e-06   # +0.9%  sum_pt < 1085 and e4 < 5.851e-08
        - 0.008590968 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 0.07996447 - Q.C2) / 0.5514341   # -0.9%  n_particles < 64 and C2 < 0.07996
        + 0.007883779 * max(0.0, 934.2416 - Q.sum_pt_top50) / 6.932864   # +0.8%  sum_pt_top50 < 934.2
        + 0.007563967 * max(0.0, Q.M2 - 0.05233747) / 0.01772949   # +0.8%  M2 > 0.05234
        + 0.006864759 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +0.7%  tau4 < 0.01627
        + 0.005539116 * max(0.0, Q.sum_pt_top15 - 1003.329) / 12.91783   # +0.6%  sum_pt_top15 > 1003
        - 0.005491838 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -0.5%  tau1 < 0.1073
        + 0.005481068 * max(0.0, Q.z_dr_0_0p05 - 0.7128619) / 0.07526795   # +0.5%  z_dr_0_0p05 > 0.7129
        + 0.005222978 * max(0.0, 907.9372 - Q.sum_pt) * max(0.0, 0.1569963 - Q.dr0_12) / 0.3024992   # +0.5%  sum_pt < 907.9 and dr0_12 < 0.157
        - 0.005181435 * max(0.0, 0.0007894752 - Q.sum_z_dr2_top15) / 9.062109e-05   # -0.5%  sum_z_dr2_top15 < 0.0007895
        - 0.005082178 * max(0.0, 0.0004005745 - Q.sum_z_dr2_top5) / 6.9335e-05   # -0.5%  sum_z_dr2_top5 < 0.0004006
        - 0.00507231 * max(0.0, Q.sum_pt_top15 - 967.7705) / 19.94058   # -0.5%  sum_pt_top15 > 967.8
        + 0.005068831 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # +0.5%  sum_pt_top50 < 959.1
        - 0.004807285 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.absphi_1 - 0.02227783) / 0.4183469   # -0.5%  sum_pt < 1002 and absphi_1 > 0.02228
        - 0.004536612 * max(0.0, Q.z_top20_slots - 0.9358352) / 0.01535337   # -0.5%  z_top20_slots > 0.9358
        + 0.003841209 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, 0.199671 - Q.dr_11) / 1.707103   # +0.4%  sum_pt < 1002 and dr_11 < 0.1997
        - 0.003305776 * max(0.0, Q.C2 - 0.1235569) / 0.002419231   # -0.3%  C2 > 0.1236
        - 0.002861088 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.3%  sum_pt_top10 > 943.7
        - 0.002475022 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.ptdr0_10 - 3.719859) / 10.46372   # -0.2%  sum_pt < 1002 and ptdr0_10 > 3.72
        + 0.002337924 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # +0.2%  log_sum_pt < 6.903
        - 0.002019025 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.ptdr0_12 - 2.024972) / 14.44148   # -0.2%  sum_pt < 1002 and ptdr0_12 > 2.025
        - 0.001419281 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, -0.003213499 - Q.mean_eta) / 7.829287e-08   # -0.1%  psi_0p3 > 0.9974 and mean_eta < -0.003213
        + 0.000475822 * max(0.0, 934.2416 - Q.sum_pt_top50) * max(0.0, 0.03626613 - Q.soft6_dr) / 0.009542907   # +0.0%  sum_pt_top50 < 934.2 and soft6_dr < 0.03627
        + 0.0001032421 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.sj3_dr23 - 0.1834565) / 1.474397   # +0.0%  sum_pt < 1002 and sj3_dr23 > 0.1835
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 18.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.23691 * (0.02299938
        - 0.2438983 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # -24.4%  sum_z_dr < 0.0975
        + 0.1243715 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +12.4%  LHA < 0.3332
        + 0.0552295 * max(0.0, 0.006427167 - Q.lam2) / 0.005146703   # +5.5%  lam2 < 0.006427
        + 0.04518425 * max(0.0, 0.05048381 - Q.sum_z_dr) / 0.008533025   # +4.5%  sum_z_dr < 0.05048
        + 0.04266215 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +4.3%  e3 < 0.0003372
        - 0.03780242 * max(0.0, Q.e3 - 0.0001086251) / 5.31284e-05   # -3.8%  e3 > 0.0001086
        - 0.03206094 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, 7.017258 - Q.log_sum_pt) / 2.140029e-05   # -3.2%  e3 < 0.0003372 and log_sum_pt < 7.017
        - 0.03176726 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # -3.2%  LHA < 0.3024
        + 0.03006319 * max(0.0, 0.003270031 - Q.sum_z_dr2_top15) / 0.0007969965   # +3.0%  sum_z_dr2_top15 < 0.00327
        + 0.02815571 * max(0.0, Q.e3 - 5.13841e-05) / 6.821576e-05   # +2.8%  e3 > 5.138e-05
        - 0.02737879 * max(0.0, 0.00727763 - Q.sum_z_dr2_top15) / 0.00282147   # -2.7%  sum_z_dr2_top15 < 0.007278
        + 0.02622235 * max(0.0, 0.04755309 - Q.e2) / 0.01877457   # +2.6%  e2 < 0.04755
        + 0.01970833 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # +2.0%  tau1 < 0.05445
        + 0.01697546 * max(0.0, Q.z_dr_0_0p05 - 0.3289237) / 0.280304   # +1.7%  z_dr_0_0p05 > 0.3289
        + 0.01689886 * max(0.0, Q.psi_0p2 - 0.9087063) * max(0.0, 0.1512157 - Q.sj2_dr) / 0.001524835   # +1.7%  psi_0p2 > 0.9087 and sj2_dr < 0.1512
        - 0.01688577 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # -1.7%  z_dr_0p2_0p4 < 0.06849
        - 0.01580106 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -1.6%  psi_0p2 > 0.9087
        - 0.01346589 * max(0.0, 0.1825048 - Q.sj2_dr) / 0.03087146   # -1.3%  sj2_dr < 0.1825
        - 0.01264058 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # -1.3%  sum_z_dr2_top15 < 0.009962
        - 0.01242409 * max(0.0, 0.2284021 - Q.LHA) / 0.02716657   # -1.2%  LHA < 0.2284
        + 0.01193393 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) / 0.003638607   # +1.2%  sum_z_dr2_top5 < 0.007164
        + 0.01129061 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # +1.1%  e2 < 0.04359
        + 0.01055183 * max(0.0, 0.3676068 - Q.sj2_zsoft) / 0.1397909   # +1.1%  sj2_zsoft < 0.3676
        + 0.01004589 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +1.0%  sum_pt > 907.9
        - 0.008700401 * max(0.0, 0.001319197 - Q.sum_z_dr2_top15) / 0.0002099631   # -0.9%  sum_z_dr2_top15 < 0.001319
        - 0.008487727 * max(0.0, 0.7108211 - Q.z_dr_0p05_0p1) / 0.4588085   # -0.8%  z_dr_0p05_0p1 < 0.7108
        - 0.008292797 * max(0.0, 0.003270031 - Q.sum_z_dr2_top15) * max(0.0, Q.sum_pt_top40 - 858.8262) / 0.1598282   # -0.8%  sum_z_dr2_top15 < 0.00327 and sum_pt_top40 > 858.8
        - 0.007826209 * max(0.0, 0.3332345 - Q.LHA) * max(0.0, 0.9638082 - Q.psi_0p3) / 5.130459e-05   # -0.8%  LHA < 0.3332 and psi_0p3 < 0.9638
        + 0.007705249 * max(0.0, Q.n_dr_0p1_0p2 - 15.0) * max(0.0, Q.psi_0p3 - 0.9853273) / 0.01929542   # +0.8%  n_dr_0p1_0p2 > 15 and psi_0p3 > 0.9853
        + 0.007381405 * max(0.0, 0.09749958 - Q.sum_z_dr) * max(0.0, 0.9638082 - Q.psi_0p3) / 1.616441e-05   # +0.7%  sum_z_dr < 0.0975 and psi_0p3 < 0.9638
        - 0.007223409 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) / 1.539217   # -0.7%  n_dr_0p2_0p4 > 15
        + 0.006437644 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # +0.6%  e2 > 0.05557
        - 0.005169609 * max(0.0, 0.3676068 - Q.sj2_zsoft) * max(0.0, 0.008824206 - Q.zdr_1) / 0.0007676544   # -0.5%  sj2_zsoft < 0.3676 and zdr_1 < 0.008824
        - 0.005111665 * max(0.0, Q.z_dr_0p2_0p4 - 0.2708738) / 0.004515795   # -0.5%  z_dr_0p2_0p4 > 0.2709
        - 0.005010969 * max(0.0, 0.08589404 - Q.sum_z_dr) * max(0.0, 0.1134943 - Q.M2) / 0.001132661   # -0.5%  sum_z_dr < 0.08589 and M2 < 0.1135
        - 0.004345122 * max(0.0, Q.n_dr_0p1_0p2 - 15.0) * max(0.0, 0.006427167 - Q.lam2) / 0.008539361   # -0.4%  n_dr_0p1_0p2 > 15 and lam2 < 0.006427
        + 0.004149956 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # +0.4%  C2 > 0.06656
        - 0.003827239 * max(0.0, Q.C2 - 0.1088881) / 0.004226185   # -0.4%  C2 > 0.1089
        + 0.003688108 * max(0.0, Q.sum_pt_top50 - 1078.994) / 24.4796   # +0.4%  sum_pt_top50 > 1079
        - 0.003370041 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.03720487   # -0.3%  z_dr_0p1_0p2 > 0.334
        + 0.002910116 * max(0.0, 0.05444509 - Q.tau1) * max(0.0, 972.0419 - Q.sum_pt) / 0.0682246   # +0.3%  tau1 < 0.05445 and sum_pt < 972
        + 0.001663142 * max(0.0, 0.04575675 - Q.sj2_zsoft) / 0.00226832   # +0.2%  sj2_zsoft < 0.04576
        + 0.00148721 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, 0.07952881 - Q.eta_0) / 0.1317641   # +0.1%  n_dr_0p2_0p4 > 15 and eta_0 < 0.07953
        - 0.001244819 * max(0.0, Q.e3 - 5.13841e-05) * max(0.0, Q.orientation_deg - 26.6454) / 0.0007609216   # -0.1%  e3 > 5.138e-05 and orientation_deg > 26.65
        - 0.001023739 * max(0.0, 0.04358622 - Q.e2) * max(0.0, 0.9638082 - Q.psi_0p3) / 1.224228e-05   # -0.1%  e2 < 0.04359 and psi_0p3 < 0.9638
        - 0.0009353965 * max(0.0, Q.n_dr_0p1_0p2 - 26.0) * max(0.0, Q.ptdr0_10 - 3.023504) / 2.185115   # -0.1%  n_dr_0p1_0p2 > 26 and ptdr0_10 > 3.024
        - 0.0005893989 * max(0.0, Q.n_dr_0p1_0p2 - 26.0) * max(0.0, Q.sum_pt - 986.0565) / 45.26133   # -0.1%  n_dr_0p1_0p2 > 26 and sum_pt > 986.1
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 26.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.18159 * (-0.01186578
        - 0.09419917 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -9.4%  tau1 < 0.1073
        + 0.08393688 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +8.4%  psi_0p3 > 0.9974
        + 0.07648537 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # +7.6%  sum_z_dr < 0.0975
        + 0.07341263 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # +7.3%  LHA < 0.3024
        - 0.06900554 * max(0.0, 0.08068193 - Q.sum_z_dr) / 0.02368455   # -6.9%  sum_z_dr < 0.08068
        + 0.04747442 * max(0.0, 0.1072713 - Q.tau1) * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) / 0.008148309   # +4.7%  tau1 < 0.1073 and z_dr_0p1_0p2 < 0.2865
        + 0.04298374 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063   # +4.3%  n_dr_0p2_0p4 < 21
        + 0.04030453 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.0001707165   # +4.0%  tau21_b2 < 0.2352 and sum_z_dr2_top15 < 0.009962
        - 0.03990645 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # -4.0%  LHA < 0.3332
        - 0.03865678 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.006802603 - Q.mean_eta2) / 3.803697e-06   # -3.9%  psi_0p3 > 0.9974 and mean_eta2 < 0.006803
        - 0.03623678 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.006808102 - Q.mean_phi2) / 3.800578e-06   # -3.6%  psi_0p3 > 0.9974 and mean_phi2 < 0.006808
        - 0.03593387 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -3.6%  n_dr_0p2_0p4 < 15
        - 0.03434964 * max(0.0, 0.3332345 - Q.LHA) * max(0.0, 0.250441 - Q.z_dr_0p1_0p2) / 0.01705837   # -3.4%  LHA < 0.3332 and z_dr_0p1_0p2 < 0.2504
        - 0.03412727 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1245.697 - Q.sum_pt_top50) / 11.02846   # -3.4%  tau21_b2 < 0.2352 and sum_pt_top50 < 1246
        + 0.02616765 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1512157) / 0.00318489   # +2.6%  tau21_b2 < 0.2352 and sj2_dr > 0.1512
        - 0.02431413 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3) / 0.0005360215   # -2.4%  n_dr_0p2_0p4 < 21 and e3 < 7.876e-05
        + 0.02177915 * max(0.0, 0.1219132 - Q.tau1) / 0.04578087   # +2.2%  tau1 < 0.1219
        - 0.02092706 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 7.062574 - Q.log_sum_pt) / 0.0001122434   # -2.1%  psi_0p3 > 0.9974 and log_sum_pt < 7.063
        - 0.01906836 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) / 8.087416e-05   # -1.9%  tau21_b2 < 0.2352 and sum_z_dr2_top15 < 0.007888
        + 0.01439291 * max(0.0, 0.05557149 - Q.e2) / 0.02580562   # +1.4%  e2 < 0.05557
        + 0.01324645 * max(0.0, 0.03680582 - Q.e2) / 0.01052402   # +1.3%  e2 < 0.03681
        + 0.01312156 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.1134943 - Q.M2) / 0.3618294   # +1.3%  n_dr_0p2_0p4 < 15 and M2 < 0.1135
        - 0.01165682 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # -1.2%  tau21_b2 < 0.2352
        + 0.01147618 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1115.723 - Q.sum_pt) / 4.512378   # +1.1%  tau21_b2 < 0.2352 and sum_pt < 1116
        + 0.01096031 * max(0.0, 6.567534e-05 - Q.e3) / 2.643172e-05   # +1.1%  e3 < 6.568e-05
        - 0.009402721 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.001696484   # -0.9%  tau21_b2 < 0.2352 and sj2_dr > 0.1938
        + 0.008237659 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.psi_0p3 - 0.9956185) / 0.02161643   # +0.8%  n_dr_0p2_0p4 < 15 and psi_0p3 > 0.9956
        - 0.008166852 * max(0.0, 0.04755309 - Q.e2) * max(0.0, 0.2352054 - Q.tau21_b2) / 0.0005663274   # -0.8%  e2 < 0.04755 and tau21_b2 < 0.2352
        - 0.007760325 * max(0.0, Q.psi_0p1 - 0.8509811) / 0.04568859   # -0.8%  psi_0p1 > 0.851
        - 0.005573217 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.sum_z_dr2_top10 - 0.007678544) / 7.055851e-07   # -0.6%  psi_0p3 > 0.9974 and sum_z_dr2_top10 > 0.007679
        - 0.005042342 * max(0.0, 0.04362872 - Q.sum_z_dr) / 0.006325401   # -0.5%  sum_z_dr < 0.04363
        - 0.004985389 * max(0.0, 3.376709e-05 - Q.e3) / 8.295151e-06   # -0.5%  e3 < 3.377e-05
        - 0.003570602 * max(0.0, 0.03263075 - Q.e2) / 0.008034085   # -0.4%  e2 < 0.03263
        + 0.003501632 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.003687605 - Q.lam2) / 2.993211e-06   # +0.4%  psi_0p3 > 0.9974 and lam2 < 0.003688
        + 0.003218669 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.4823776 - Q.tau21_b2) / 2.786788   # +0.3%  n_dr_0p2_0p4 < 21 and tau21_b2 < 0.4824
        + 0.003029967 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 0.5616627   # +0.3%  tau21_b2 < 0.2352 and n_dr_0p05_0p1 > 2
        + 0.002655305 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.4357228 - Q.max_dr) / 0.8636325   # +0.3%  n_dr_0p2_0p4 < 15 and max_dr < 0.4357
        - 0.000731672 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 972.0419 - Q.sum_pt) / 0.2148301   # -0.1%  tau21_b2 < 0.2352 and sum_pt < 972
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 28.52;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.51806 * (0.07672601
        - 0.1444887 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -14.4%  log_sum_pt < 7.139
        + 0.1157943 * max(0.0, Q.sum_z_dr - 0.04362872) / 0.03194126   # +11.6%  sum_z_dr > 0.04363
        + 0.1051409 * max(0.0, Q.sum_z_dr - 0.03577037) / 0.03765667   # +10.5%  sum_z_dr > 0.03577
        - 0.0956638 * max(0.0, Q.LHA - 0.2091025) / 0.07391154   # -9.6%  LHA > 0.2091
        + 0.08702678 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # +8.7%  sum_pt < 1261
        - 0.0504484 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # -5.0%  sum_z_dr > 0.0975
        - 0.03026858 * max(0.0, Q.sum_z_dr - 0.07374472) / 0.01465579   # -3.0%  sum_z_dr > 0.07374
        + 0.02944002 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +2.9%  sum_pt_top50 < 1157
        - 0.02597548 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.139296 - Q.log_sum_pt) / 1.930358   # -2.6%  n_dr_0p2_0p4 < 18 and log_sum_pt < 7.139
        - 0.02574342 * max(0.0, Q.z_top50_slots - 0.9586536) / 0.03427112   # -2.6%  z_top50_slots > 0.9587
        + 0.02516746 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +2.5%  LHA > 0.3332
        - 0.02255028 * max(0.0, Q.e2 - 0.02210818) / 0.01219843   # -2.3%  e2 > 0.02211
        + 0.019051 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, 1191.938 - Q.sum_pt_top30) / 4.563404   # +1.9%  sum_z_dr > 0.07032 and sum_pt_top30 < 1192
        - 0.01764311 * max(0.0, 1191.938 - Q.sum_pt_top30) / 206.0686   # -1.8%  sum_pt_top30 < 1192
        - 0.01666399 * max(0.0, 0.00727763 - Q.sum_z_dr2_top15) / 0.00282147   # -1.7%  sum_z_dr2_top15 < 0.007278
        - 0.01470139 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 3.793233e-05 - Q.e3) / 0.0001192169   # -1.5%  n_dr_0p2_0p4 < 18 and e3 < 3.793e-05
        - 0.01442248 * max(0.0, Q.psi_0p3 - 0.9777125) / 0.01571542   # -1.4%  psi_0p3 > 0.9777
        - 0.01321372 * max(0.0, Q.sum_z_dr - 0.07031778) / 0.01612201   # -1.3%  sum_z_dr > 0.07032
        + 0.0126699 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, 1115.723 - Q.sum_pt) / 1.878598   # +1.3%  sum_z_dr > 0.07032 and sum_pt < 1116
        - 0.01152937 * max(0.0, 1027.303 - Q.sum_pt_top30) / 61.36161   # -1.2%  sum_pt_top30 < 1027
        - 0.01125797 * max(0.0, Q.LHA - 0.3203321) / 0.01671666   # -1.1%  LHA > 0.3203
        - 0.01075669 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # -1.1%  n_dr_0p2_0p4 < 18
        + 0.00872778 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # +0.9%  sum_pt < 1002
        + 0.007533405 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, Q.e2 - 0.04358622) / 0.000222409   # +0.8%  sum_z_dr > 0.07032 and e2 > 0.04359
        - 0.006914384 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 4.450169 - Q.D2) / 457.6573   # -0.7%  sum_pt < 1261 and D2 < 4.45
        + 0.006254016 * max(0.0, 1042.609 - Q.sum_pt) / 35.65948   # +0.6%  sum_pt < 1043
        + 0.005833721 * max(0.0, 0.08082334 - Q.dr_0) / 0.0377444   # +0.6%  dr_0 < 0.08082
        + 0.00562988 * max(0.0, 3.793233e-05 - Q.e3) / 1.004564e-05   # +0.6%  e3 < 3.793e-05
        + 0.005369027 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.5%  sj2_dr > 0.2232
        + 0.004978325 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # +0.5%  sum_pt_top40 < 1007
        - 0.004930245 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 0.7225388 - Q.planar_flow) / 60.52173   # -0.5%  sum_pt < 1261 and planar_flow < 0.7225
        + 0.004689133 * max(0.0, 1042.609 - Q.sum_pt) * max(0.0, Q.max_dr - 0.1939977) / 6.226413   # +0.5%  sum_pt < 1043 and max_dr > 0.194
        - 0.004589597 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, 976.277 - Q.sum_pt_top50) / 0.4391148   # -0.5%  sum_z_dr > 0.07032 and sum_pt_top50 < 976.3
        + 0.003725672 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, 0.0795038 - Q.tau2) / 0.0003075621   # +0.4%  sum_z_dr > 0.07032 and tau2 < 0.0795
        - 0.003701499 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # -0.4%  n_dr_0p2_0p4 < 8
        + 0.00302015 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +0.3%  LHA > 0.372
        + 0.002970171 * max(0.0, 0.02675364 - Q.sum_z_dr2_top15) * max(0.0, Q.tau4 - 0.01517184) / 5.673266e-05   # +0.3%  sum_z_dr2_top15 < 0.02675 and tau4 > 0.01517
        + 0.002802453 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.8252004 - Q.psi_0p1) / 0.5392896   # +0.3%  n_dr_0p2_0p4 < 18 and psi_0p1 < 0.8252
        - 0.002379429 * max(0.0, 2.178951 - Q.D2) / 0.4648731   # -0.2%  D2 < 2.179
        + 0.00231557 * max(0.0, Q.e2 - 0.04358622) / 0.002786698   # +0.2%  e2 > 0.04359
        - 0.002279856 * max(0.0, 0.04575675 - Q.sj2_zsoft) / 0.00226832   # -0.2%  sj2_zsoft < 0.04576
        - 0.002268459 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) * max(0.0, 0.3315857 - Q.max_dr) / 9.802766e-05   # -0.2%  sum_z_dr2_top15 < 0.009962 and max_dr < 0.3316
        + 0.001977526 * max(0.0, Q.psi_0p3 - 0.9777125) * max(0.0, Q.n_dr_0p1_0p2 - 17.0) / 0.02691594   # +0.2%  psi_0p3 > 0.9777 and n_dr_0p1_0p2 > 17
        - 0.001874974 * max(0.0, 966.0633 - Q.sum_pt_top30) / 29.99923   # -0.2%  sum_pt_top30 < 966.1
        - 0.001309223 * max(0.0, Q.LHA - 0.3098384) / 0.0195892   # -0.1%  LHA > 0.3098
        + 0.00128844 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) * max(0.0, Q.sj3_dr23 - 0.2563461) / 8.744166e-05   # +0.1%  sum_z_dr2_top5 < 0.007164 and sj3_dr23 > 0.2563
        - 0.001197745 * max(0.0, Q.LHA - 0.3203321) * max(0.0, 0.02037449 - Q.z_14) / 4.375313e-05   # -0.1%  LHA > 0.3203 and z_14 < 0.02037
        + 0.0009878162 * max(0.0, 0.08082334 - Q.dr_0) * max(0.0, Q.n_dr_0p4_up - 0.0) / 0.004652169   # +0.1%  dr_0 < 0.08082 and n_dr_0p4_up > 0
        - 0.0008336769 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.z_0 - 0.08872428) / 1.838848   # -0.1%  sum_pt < 1002 and z_0 > 0.08872
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 7.613;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.613059 * (-0.1610857
        + 0.1069707 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # +10.7%  tau1 < 0.1073
        + 0.09606464 * max(0.0, 0.09591084 - Q.tau1) / 0.02629125   # +9.6%  tau1 < 0.09591
        + 0.07639112 * max(0.0, 0.2595052 - Q.sj2_dr) / 0.08013882   # +7.6%  sj2_dr < 0.2595
        + 0.06787975 * max(0.0, Q.e2 - 0.02515919) / 0.01027642   # +6.8%  e2 > 0.02516
        - 0.06526734 * max(0.0, Q.sum_pt_top20 - 750.7313) / 184.752   # -6.5%  sum_pt_top20 > 750.7
        - 0.06327498 * max(0.0, Q.z_top40_slots - 0.9674996) / 0.02046045   # -6.3%  z_top40_slots > 0.9675
        + 0.06031206 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.000105955   # +6.0%  sum_z_dr2_top15 < 0.006143 and z_dr_0p2_0p4 < 0.06849
        - 0.0512258 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) / 0.002080651   # -5.1%  sum_z_dr2_top15 < 0.006143
        - 0.04808999 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -4.8%  sum_z_dr > 0.1207
        + 0.04178417 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # +4.2%  sum_z_dr > 0.0975
        - 0.04087694 * max(0.0, 0.1825048 - Q.sj2_dr) / 0.03087146   # -4.1%  sj2_dr < 0.1825
        + 0.03003267 * max(0.0, 2.178951 - Q.D2) / 0.4648731   # +3.0%  D2 < 2.179
        - 0.02772516 * max(0.0, Q.sum_z_dr - 0.1402186) / 0.001871547   # -2.8%  sum_z_dr > 0.1402
        - 0.02047133 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # -2.0%  LHA < 0.2454
        + 0.02036874 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +2.0%  LHA > 0.4042
        + 0.01866485 * max(0.0, 0.05660088 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.0003340726   # +1.9%  sum_z_dr < 0.0566 and psi_0p3 > 0.9638
        - 0.01585523 * max(0.0, 3.376709e-05 - Q.e3) / 8.295151e-06   # -1.6%  e3 < 3.377e-05
        + 0.01502889 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.9777125) / 3.516962e-05   # +1.5%  sum_z_dr2_top15 < 0.006143 and psi_0p3 > 0.9777
        + 0.01486072 * max(0.0, Q.psi_0p1 - 0.8509811) / 0.04568859   # +1.5%  psi_0p1 > 0.851
        + 0.01182698 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +1.2%  LHA > 0.372
        + 0.01160834 * max(0.0, 889.8503 - Q.sum_pt_top50) / 3.808547   # +1.2%  sum_pt_top50 < 889.9
        + 0.01140733 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) * max(0.0, Q.n_dr_0p2_0p4 - 7.0) / 0.004066923   # +1.1%  sum_z_dr2_top15 < 0.006143 and n_dr_0p2_0p4 > 7
        + 0.0110014 * max(0.0, 0.05660088 - Q.sum_z_dr) * max(0.0, 1007.788 - Q.sum_pt) / 0.1758936   # +1.1%  sum_z_dr < 0.0566 and sum_pt < 1008
        - 0.01088951 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, Q.sum_pt_top20 - 789.7344) / 304.8764   # -1.1%  sum_pt_top50 < 988.5 and sum_pt_top20 > 789.7
        + 0.009022448 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) / 0.03630126   # +0.9%  z_dr_0_0p05 > 0.8103
        - 0.008395984 * max(0.0, Q.sum_z_dr2_top15 - 0.02146578) / 0.0006182335   # -0.8%  sum_z_dr2_top15 > 0.02147
        + 0.008342384 * max(0.0, Q.LHA - 0.3719813) * max(0.0, 0.006427167 - Q.lam2) / 1.406065e-05   # +0.8%  LHA > 0.372 and lam2 < 0.006427
        + 0.007857706 * max(0.0, 0.05660088 - Q.sum_z_dr) / 0.01080382   # +0.8%  sum_z_dr < 0.0566
        - 0.006040191 * max(0.0, 0.0007894752 - Q.sum_z_dr2_top15) / 9.062109e-05   # -0.6%  sum_z_dr2_top15 < 0.0007895
        + 0.00512065 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.09061548 - Q.dr_13) / 0.0001299587   # +0.5%  log_sum_pt < 6.811 and dr_13 < 0.09062
        + 0.004969866 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, 0.001416411 - Q.C3) / 0.00386608   # +0.5%  sum_pt_top50 < 988.5 and C3 < 0.001416
        + 0.004832893 * max(0.0, Q.sum_z_dr - 0.1564779) / 0.0006617163   # +0.5%  sum_z_dr > 0.1565
        - 0.003876869 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.4%  log_sum_pt < 6.811
        + 0.003662388 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, Q.soft5_z - 0.0004140594) / 2.555122e-05   # +0.4%  z_dr_0p1_0p2 > 0.4479 and soft5_z > 0.0004141
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 14.36;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.35618 * (0.2461614
        + 0.1885963 * max(0.0, Q.e2 - 0.008484542) / 0.02258988   # +18.9%  e2 > 0.008485
        - 0.1446188 * max(0.0, 0.1207452 - Q.sum_z_dr) / 0.05578614   # -14.5%  sum_z_dr < 0.1207
        - 0.08270345 * max(0.0, Q.LHA - 0.1870291) / 0.08997036   # -8.3%  LHA > 0.187
        - 0.07042114 * max(0.0, Q.tau1 - 0.04466492) / 0.04702157   # -7.0%  tau1 > 0.04466
        - 0.05617943 * max(0.0, Q.sum_pt_top5 - 402.625) / 200.0399   # -5.6%  sum_pt_top5 > 402.6
        + 0.04292846 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +4.3%  LHA > 0.3332
        - 0.04153825 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # -4.2%  sum_z_dr > 0.0975
        - 0.03025995 * max(0.0, 0.02441963 - Q.sum_z_dr2_top5) / 0.01889954   # -3.0%  sum_z_dr2_top5 < 0.02442
        + 0.02609746 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +2.6%  n_dr_0p2_0p4 < 18
        - 0.0240465 * max(0.0, Q.pt_entropy - 2.07371) / 0.8049786   # -2.4%  pt_entropy > 2.074
        + 0.02126632 * max(0.0, 0.004007842 - Q.sum_z_dr2_top2) / 0.001907211   # +2.1%  sum_z_dr2_top2 < 0.004008
        - 0.01958635 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # -2.0%  e2 > 0.0303
        + 0.01917028 * max(0.0, 2.410481 - Q.D2) / 0.587301   # +1.9%  D2 < 2.41
        + 0.01677867 * max(0.0, 5.13841e-05 - Q.e3) / 1.709557e-05   # +1.7%  e3 < 5.138e-05
        + 0.01583819 * max(0.0, Q.sum_z_dr - 0.09749958) * max(0.0, 1156.659 - Q.sum_pt_top50) / 1.495133   # +1.6%  sum_z_dr > 0.0975 and sum_pt_top50 < 1157
        + 0.01569035 * max(0.0, 0.1207452 - Q.sum_z_dr) * max(0.0, Q.psi_0p2 - 0.948102) / 0.001922283   # +1.6%  sum_z_dr < 0.1207 and psi_0p2 > 0.9481
        - 0.014795 * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0004356739   # -1.5%  psi_0p3 > 0.9985
        + 0.01442466 * max(0.0, 0.0001086251 - Q.e3) / 5.924919e-05   # +1.4%  e3 < 0.0001086
        + 0.01367203 * max(0.0, Q.psi_0p3 - 0.9989733) / 0.0002740091   # +1.4%  psi_0p3 > 0.999
        - 0.01200554 * max(0.0, 2.026142 - Q.D2_b2) / 0.7121724   # -1.2%  D2_b2 < 2.026
        - 0.01068269 * max(0.0, 0.01414829 - Q.sum_z_dr2_top10) / 0.00877908   # -1.1%  sum_z_dr2_top10 < 0.01415
        - 0.01055171 * max(0.0, 0.2018786 - Q.tau21_b2) / 0.03799024   # -1.1%  tau21_b2 < 0.2019
        - 0.01026647 * max(0.0, Q.sum_z_dr - 0.1402186) / 0.001871547   # -1.0%  sum_z_dr > 0.1402
        - 0.01025576 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # -1.0%  z_dr_0p1_0p2 < 0.1203
        + 0.009564808 * max(0.0, Q.psi_0p1 - 0.9184255) / 0.01693566   # +1.0%  psi_0p1 > 0.9184
        - 0.009031416 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_real_top40 - 22.0) / 55.27538   # -0.9%  n_dr_0p2_0p4 < 11 and n_real_top40 > 22
        + 0.007097651 * max(0.0, 889.8383 - Q.sum_pt_top20) / 35.81237   # +0.7%  sum_pt_top20 < 889.8
        - 0.007031096 * max(0.0, Q.sj2_dr - 0.09395198) * max(0.0, Q.M2 - 0.04260132) / 0.001979695   # -0.7%  sj2_dr > 0.09395 and M2 > 0.0426
        - 0.006992216 * max(0.0, Q.psi_0p3 - 0.9989733) * max(0.0, Q.eccentricity - 0.5245966) / 9.193571e-05   # -0.7%  psi_0p3 > 0.999 and eccentricity > 0.5246
        - 0.006480409 * max(0.0, 0.02944575 - Q.M3) / 0.00752156   # -0.6%  M3 < 0.02945
        - 0.00500019 * max(0.0, Q.z_dr_0_0p05 - 0.7128619) / 0.07526795   # -0.5%  z_dr_0_0p05 > 0.7129
        - 0.004120887 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # -0.4%  sum_pt_top5 > 791.1
        + 0.003287113 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, 0.1082864 - Q.z_3) / 4.224064e-05   # +0.3%  tau1 > 0.1954 and z_3 < 0.1083
        + 0.003236147 * max(0.0, 1024.942 - Q.sum_pt_top40) * max(0.0, Q.n_dr_0_0p05 - 1.0) / 377.2098   # +0.3%  sum_pt_top40 < 1025 and n_dr_0_0p05 > 1
        - 0.003175267 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, Q.sum_pt - 1167.447) / 0.008922112   # -0.3%  tau1 > 0.1954 and sum_pt > 1167
        + 0.003164935 * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 1.386277   # +0.3%  n_dr_0p1_0p2 > 21
        + 0.0031535 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +0.3%  LHA > 0.4042
        - 0.003114379 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, 114.75 - Q.pt_3) / 0.04784342   # -0.3%  tau1 > 0.1954 and pt_3 < 114.8
        - 0.002515848 * max(0.0, 6.856375 - Q.log_sum_pt) / 0.008053459   # -0.3%  log_sum_pt < 6.856
        + 0.002515054 * max(0.0, 0.1207452 - Q.sum_z_dr) * max(0.0, Q.pt1_dr01 - 5.351077) / 0.09847904   # +0.3%  sum_z_dr < 0.1207 and pt1_dr01 > 5.351
        + 0.002140459 * max(0.0, 0.1207452 - Q.sum_z_dr) * max(0.0, 997.0189 - Q.sum_pt_top50) / 0.7202273   # +0.2%  sum_z_dr < 0.1207 and sum_pt_top50 < 997
        + 0.001899594 * max(0.0, Q.tau1 - 0.1751567) / 0.002322569   # +0.2%  tau1 > 0.1752
        - 0.001706556 * max(0.0, Q.tau1 - 0.1953848) / 0.000822026   # -0.2%  tau1 > 0.1954
        - 0.001341055 * max(0.0, Q.e2 - 0.03029714) * max(0.0, Q.soft1_pt - 1.091797) / 0.001350423   # -0.1%  e2 > 0.0303 and soft1_pt > 1.092
        - 0.001057738 * max(0.0, 889.8503 - Q.sum_pt_top50) / 3.808547   # -0.1%  sum_pt_top50 < 889.9
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 15.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.67494 * (-0.02979237
        - 0.1574493 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # -15.7%  LHA < 0.3098
        + 0.1165741 * max(0.0, 0.08589404 - Q.sum_z_dr) / 0.02745918   # +11.7%  sum_z_dr < 0.08589
        + 0.08832188 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +8.8%  LHA < 0.3332
        - 0.07704066 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) / 0.002080651   # -7.7%  sum_z_dr2_top15 < 0.006143
        + 0.07500626 * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) / 0.00326329   # +7.5%  sum_z_dr2_top15 < 0.007888
        - 0.04739741 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # -4.7%  e2 < 0.04083
        - 0.0412927 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # -4.1%  sum_z_dr < 0.0975
        + 0.03305835 * max(0.0, 0.004752876 - Q.sum_z_dr2_top10) / 0.001623418   # +3.3%  sum_z_dr2_top10 < 0.004753
        + 0.03247282 * max(0.0, Q.psi_0p2 - 0.8706159) / 0.08984765   # +3.2%  psi_0p2 > 0.8706
        - 0.03208936 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # -3.2%  z_dr_0p2_0p4 < 0.06849
        + 0.02692517 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +2.7%  z_dr_0p2_0p4 < 0.09123
        - 0.02650297 * max(0.0, 0.007678544 - Q.sum_z_dr2_top10) / 0.00347358   # -2.7%  sum_z_dr2_top10 < 0.007679
        + 0.02265004 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +2.3%  e2 < 0.02516
        + 0.01931493 * max(0.0, 0.05775119 - Q.dr_0) / 0.02089184   # +1.9%  dr_0 < 0.05775
        + 0.01812267 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # +1.8%  e2 < 0.04359
        + 0.01580206 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 33.0 - Q.n_dr_0p1_0p2) / 87.77824   # +1.6%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 33
        + 0.01537456 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +1.5%  psi_0p3 > 0.9897
        + 0.01278346 * max(0.0, 1.409617 - Q.D2) / 0.1519846   # +1.3%  D2 < 1.41
        - 0.01150236 * max(0.0, 3.376709e-05 - Q.e3) / 8.295151e-06   # -1.2%  e3 < 3.377e-05
        - 0.01097828 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -1.1%  n_dr_0p2_0p4 < 15
        + 0.01087287 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +1.1%  n_dr_0p2_0p4 < 10
        + 0.01027007 * max(0.0, 0.01564747 - Q.zdr_0) / 0.007296635   # +1.0%  zdr_0 < 0.01565
        + 0.01024674 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +1.0%  e2 < 0.01879
        - 0.008450977 * max(0.0, 0.2891675 - Q.z_0) / 0.09032963   # -0.8%  z_0 < 0.2892
        - 0.008173335 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005337976 - Q.sum_z_dr2_top10) / 0.007083386   # -0.8%  n_dr_0p2_0p4 < 10 and sum_z_dr2_top10 < 0.005338
        + 0.00813625 * max(0.0, 19.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.6025827 - Q.planar_flow) / 1.538522   # +0.8%  n_dr_0p1_0p2 < 19 and planar_flow < 0.6026
        - 0.007840861 * max(0.0, 0.1623049 - Q.sj3_dr_max) / 0.01012502   # -0.8%  sj3_dr_max < 0.1623
        - 0.007501898 * max(0.0, 0.09749958 - Q.sum_z_dr) * max(0.0, Q.tau4 - 0.008810529) / 0.0001863032   # -0.8%  sum_z_dr < 0.0975 and tau4 > 0.008811
        - 0.006869962 * max(0.0, 1.409617 - Q.D2) * max(0.0, Q.n_real_top50 - 22.0) / 2.18239   # -0.7%  D2 < 1.41 and n_real_top50 > 22
        - 0.006396948 * max(0.0, 0.09338587 - Q.dr_0) * max(0.0, Q.eta_0 - -0.02893372) / 0.001461651   # -0.6%  dr_0 < 0.09339 and eta_0 > -0.02893
        - 0.0055939 * max(0.0, Q.z_dr_0p05_0p1 - 0.4026646) * max(0.0, 0.02704832 - Q.C2_b2) / 0.001759463   # -0.6%  z_dr_0p05_0p1 > 0.4027 and C2_b2 < 0.02705
        + 0.005402922 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) / 0.03630126   # +0.5%  z_dr_0_0p05 > 0.8103
        + 0.005020803 * max(0.0, Q.psi_0p2 - 0.995185) / 0.0009240812   # +0.5%  psi_0p2 > 0.9952
        - 0.005017703 * max(0.0, Q.n_dr_0p1_0p2 - 14.0) / 3.026787   # -0.5%  n_dr_0p1_0p2 > 14
        + 0.004738447 * max(0.0, Q.z_dr_0p05_0p1 - 0.4026646) / 0.08060812   # +0.5%  z_dr_0p05_0p1 > 0.4027
        - 0.003849948 * max(0.0, 1.409617 - Q.D2) * max(0.0, 0.04963857 - Q.z_7) / 0.002083888   # -0.4%  D2 < 1.41 and z_7 < 0.04964
        + 0.003099923 * max(0.0, 0.1506299 - Q.sj3_dr_max) / 0.008148644   # +0.3%  sj3_dr_max < 0.1506
        - 0.00185709 * max(0.0, 0.01505687 - Q.tau2) / 0.0009360369   # -0.2%  tau2 < 0.01506
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 11.96;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.95762 * (-0.05736803
        - 0.1303011 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # -13.0%  sum_pt > 907.9
        + 0.117358 * max(0.0, 0.06171014 - Q.sum_z_dr) / 0.01295056   # +11.7%  sum_z_dr < 0.06171
        - 0.06761919 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # -6.8%  e2 < 0.02516
        + 0.05926109 * max(0.0, 0.005788041 - Q.sum_z_dr2_top15) / 0.001877672   # +5.9%  sum_z_dr2_top15 < 0.005788
        - 0.05511925 * max(0.0, Q.log_sum_pt - 6.811175) / 0.1383141   # -5.5%  log_sum_pt > 6.811
        + 0.05337237 * max(0.0, 4.646151e-05 - Q.e3) / 1.427904e-05   # +5.3%  e3 < 4.646e-05
        + 0.0508605 * max(0.0, 0.05048381 - Q.sum_z_dr) * max(0.0, Q.log_sum_pt - 6.811175) / 0.001311076   # +5.1%  sum_z_dr < 0.05048 and log_sum_pt > 6.811
        + 0.04289865 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +4.3%  n_dr_0p2_0p4 < 10
        + 0.03238111 * max(0.0, Q.psi_0p2 - 0.9734513) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 0.1343004   # +3.2%  psi_0p2 > 0.9735 and n_dr_0p1_0p2 < 21
        - 0.03231187 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # -3.2%  LHA < 0.2454
        + 0.03204187 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.006615185 - Q.sum_z_dr2_top15) / 1.491736e-05   # +3.2%  psi_0p3 > 0.9897 and sum_z_dr2_top15 < 0.006615
        - 0.03012811 * max(0.0, 0.2284021 - Q.LHA) / 0.02716657   # -3.0%  LHA < 0.2284
        - 0.02934864 * max(0.0, 0.002197765 - Q.sum_z_dr2_top15) / 0.0004509993   # -2.9%  sum_z_dr2_top15 < 0.002198
        - 0.02327248 * max(0.0, 0.1048824 - Q.sj3_dr12) / 0.02123626   # -2.3%  sj3_dr12 < 0.1049
        - 0.02234139 * max(0.0, 0.2628766 - Q.sj3_dr_max) * max(0.0, Q.sum_z_dr2_top10 - 0.004752876) / 3.407988e-05   # -2.2%  sj3_dr_max < 0.2629 and sum_z_dr2_top10 > 0.004753
        - 0.02134315 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # -2.1%  LHA < 0.187
        + 0.02033038 * max(0.0, 0.1840219 - Q.sj3_dr12) / 0.05790731   # +2.0%  sj3_dr12 < 0.184
        + 0.01981975 * max(0.0, 0.1999777 - Q.sj3_dr_max) / 0.02143   # +2.0%  sj3_dr_max < 0.2
        - 0.01948403 * max(0.0, 0.1506299 - Q.sj3_dr_max) / 0.008148644   # -1.9%  sj3_dr_max < 0.1506
        - 0.01741114 * max(0.0, 0.01414829 - Q.sum_z_dr2_top10) / 0.00877908   # -1.7%  sum_z_dr2_top10 < 0.01415
        + 0.01401828 * max(0.0, 0.005850286 - Q.mean_eta2) / 0.00275351   # +1.4%  mean_eta2 < 0.00585
        - 0.01383911 * max(0.0, 0.05910514 - Q.M2) / 0.00697321   # -1.4%  M2 < 0.05911
        + 0.008689801 * max(0.0, 1167.447 - Q.sum_pt) * max(0.0, 0.05910514 - Q.M2) / 0.9368297   # +0.9%  sum_pt < 1167 and M2 < 0.05911
        + 0.008613414 * max(0.0, 0.2628766 - Q.sj3_dr_max) * max(0.0, Q.eccentricity - 0.5245966) / 0.01227931   # +0.9%  sj3_dr_max < 0.2629 and eccentricity > 0.5246
        + 0.007952693 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +0.8%  LHA > 0.372
        - 0.007564116 * max(0.0, Q.eccentricity - 0.8903081) / 0.02227339   # -0.8%  eccentricity > 0.8903
        + 0.007432887 * max(0.0, Q.pt_dispersion - 0.3227599) / 0.0409829   # +0.7%  pt_dispersion > 0.3228
        - 0.007358276 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.008152108 - Q.C3) / 1.7068e-05   # -0.7%  psi_0p3 > 0.9897 and C3 < 0.008152
        + 0.007018535 * max(0.0, Q.LHA - 0.3719813) * max(0.0, Q.log_sum_pt - 6.856375) / 0.0004589561   # +0.7%  LHA > 0.372 and log_sum_pt > 6.856
        - 0.006131889 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # -0.6%  LHA > 0.4042
        + 0.006071683 * max(0.0, 0.06069672 - Q.sj3_dr12) / 0.008100502   # +0.6%  sj3_dr12 < 0.0607
        + 0.005700277 * max(0.0, 0.2628766 - Q.sj3_dr_max) / 0.05345657   # +0.6%  sj3_dr_max < 0.2629
        - 0.00422269 * max(0.0, 0.1999777 - Q.sj3_dr_max) * max(0.0, 2.084411 - Q.ptdr0_14) / 0.02836232   # -0.4%  sj3_dr_max < 0.2 and ptdr0_14 < 2.084
        + 0.00404375 * max(0.0, 0.1210264 - Q.sj3_dr_max) / 0.004637661   # +0.4%  sj3_dr_max < 0.121
        + 0.003749407 * max(0.0, 0.05048381 - Q.sum_z_dr) / 0.008533025   # +0.4%  sum_z_dr < 0.05048
        - 0.003367744 * max(0.0, 0.1999777 - Q.sj3_dr_max) * max(0.0, 3.519576 - Q.ptdr0_8) / 0.04934164   # -0.3%  sj3_dr_max < 0.2 and ptdr0_8 < 3.52
        + 0.002508385 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # +0.3%  e2 > 0.06524
        - 0.002051054 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, Q.ptdr0_12 - 2.533834) / 0.09052767   # -0.2%  log_sum_pt > 6.811 and ptdr0_12 > 2.534
        - 0.001402672 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, 0.01204531 - Q.mean_eta2) / 0.001166867   # -0.1%  log_sum_pt > 6.811 and mean_eta2 < 0.01205
        + 0.001259334 * max(0.0, Q.LHA - 0.404204) * max(0.0, 0.0329485 - Q.C2_b2) / 2.756829e-05   # +0.1%  LHA > 0.4042 and C2_b2 < 0.03295
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 33.94;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.93558 * (0.05514705
        - 0.3610562 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # -36.1%  log_sum_pt < 6.989
        + 0.3327522 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # +33.3%  sum_pt < 1085
        + 0.04249315 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # +4.2%  e3 < 0.0005178
        - 0.03049718 * max(0.0, 6.959294 - Q.log_sum_pt) / 0.0431467   # -3.0%  log_sum_pt < 6.959
        + 0.02569254 * max(0.0, 1048.098 - Q.sum_pt_top50) / 44.36839   # +2.6%  sum_pt_top50 < 1048
        - 0.02273664 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # -2.3%  sum_z_dr > 0.0975
        + 0.0211219 * max(0.0, 1115.723 - Q.sum_pt) / 92.38491   # +2.1%  sum_pt < 1116
        - 0.01787951 * max(0.0, Q.sum_z_dr - 0.08589404) / 0.01080971   # -1.8%  sum_z_dr > 0.08589
        - 0.01651603 * max(0.0, 0.0001841806 - Q.e3) / 0.0001220279   # -1.7%  e3 < 0.0001842
        + 0.01612581 * max(0.0, Q.LHA - 0.3203321) / 0.01671666   # +1.6%  LHA > 0.3203
        + 0.01266543 * max(0.0, Q.sum_z_dr - 0.09749958) * max(0.0, 1167.447 - Q.sum_pt) / 1.406455   # +1.3%  sum_z_dr > 0.0975 and sum_pt < 1167
        - 0.01012403 * max(0.0, Q.sum_z_dr2_top15 - 0.00727763) / 0.002923446   # -1.0%  sum_z_dr2_top15 > 0.007278
        - 0.008256827 * max(0.0, Q.sum_pt - 1167.447) / 14.21193   # -0.8%  sum_pt > 1167
        - 0.007255043 * max(0.0, 1115.723 - Q.sum_pt) * max(0.0, 2.0948e-08 - Q.e4) / 1.57227e-06   # -0.7%  sum_pt < 1116 and e4 < 2.095e-08
        - 0.006396232 * max(0.0, 26.0 - Q.n_dr_0p1_0p2) / 14.20854   # -0.6%  n_dr_0p1_0p2 < 26
        + 0.005743413 * max(0.0, Q.sum_z_dr2_top15 - 0.005383629) / 0.003662853   # +0.6%  sum_z_dr2_top15 > 0.005384
        - 0.005195697 * max(0.0, 0.0001841806 - Q.e3) * max(0.0, 41.4375 - Q.pt_9) / 0.00190888   # -0.5%  e3 < 0.0001842 and pt_9 < 41.44
        + 0.005100305 * max(0.0, Q.sum_z_dr2_top15 - 0.009962397) * max(0.0, 1225.842 - Q.sum_pt_top40) / 0.6413342   # +0.5%  sum_z_dr2_top15 > 0.009962 and sum_pt_top40 < 1226
        + 0.005049729 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 6.811175 - Q.log_sum_pt) / 1.755002   # +0.5%  sum_pt < 1085 and log_sum_pt < 6.811
        + 0.004793042 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # +0.5%  log_sum_pt > 7.063
        + 0.004699486 * max(0.0, Q.e2 - 0.04082832) / 0.003398869   # +0.5%  e2 > 0.04083
        - 0.003974346 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -0.4%  psi_0p2 > 0.9087
        + 0.003653192 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # +0.4%  sum_pt_top50 < 959.1
        + 0.003247083 * max(0.0, Q.sum_z_dr2_top5 - 0.00422683) / 0.003365384   # +0.3%  sum_z_dr2_top5 > 0.004227
        - 0.002903007 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -0.3%  sum_z_dr > 0.1207
        - 0.002724805 * max(0.0, Q.sum_z_dr - 0.1564779) / 0.0006617163   # -0.3%  sum_z_dr > 0.1565
        + 0.001986777 * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 1.386277   # +0.2%  n_dr_0p1_0p2 > 21
        - 0.001887348 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, Q.sd_zg - 0.1634067) / 7.780975   # -0.2%  sum_pt < 1085 and sd_zg > 0.1634
        - 0.001571455 * max(0.0, Q.sum_z_dr2_top15 - 0.005383629) * max(0.0, Q.soft6_pt - 1.740234) / 0.001771781   # -0.2%  sum_z_dr2_top15 > 0.005384 and soft6_pt > 1.74
        - 0.00154893 * max(0.0, 23.0 - Q.n_pt_above_5) / 1.859217   # -0.2%  n_pt_above_5 < 23
        - 0.001343001 * max(0.0, Q.sum_z_dr2_top15 - 0.01563836) / 0.001334556   # -0.1%  sum_z_dr2_top15 > 0.01564
        - 0.001259992 * max(0.0, 0.0005178279 - Q.e3) * max(0.0, 0.9966167 - Q.psi_0p3) / 2.589793e-06   # -0.1%  e3 < 0.0005178 and psi_0p3 < 0.9966
        - 0.001171413 * max(0.0, 708.7945 - Q.sum_pt_top15) / 11.69336   # -0.1%  sum_pt_top15 < 708.8
        + 0.001152839 * max(0.0, 6.959294 - Q.log_sum_pt) * max(0.0, 0.01311308 - Q.C3) / 0.0002829716   # +0.1%  log_sum_pt < 6.959 and C3 < 0.01311
        + 0.001103609 * max(0.0, Q.sum_pt_top30 - 1111.245) / 12.62573   # +0.1%  sum_pt_top30 > 1111
        - 0.001095469 * max(0.0, Q.sum_z_dr - 0.1402186) * max(0.0, Q.soft6_pt - 3.328125) / 0.0002562638   # -0.1%  sum_z_dr > 0.1402 and soft6_pt > 3.328
        + 0.001091784 * max(0.0, 1048.098 - Q.sum_pt_top50) * max(0.0, 0.0003125151 - Q.sum_z_dr2_top2) / 0.002122269   # +0.1%  sum_pt_top50 < 1048 and sum_z_dr2_top2 < 0.0003125
        - 0.00101366 * max(0.0, Q.sum_z_dr - 0.1402186) / 0.001871547   # -0.1%  sum_z_dr > 0.1402
        + 0.000864276 * max(0.0, Q.sum_z_dr - 0.1402186) * max(0.0, Q.soft6_z - 0.002707742) / 4.721914e-07   # +0.1%  sum_z_dr > 0.1402 and soft6_z > 0.002708
        - 0.0008480408 * max(0.0, Q.sum_z_dr - 0.1402186) * max(0.0, Q.soft6_pt - 4.250195) / 9.168767e-05   # -0.1%  sum_z_dr > 0.1402 and soft6_pt > 4.25
        + 0.0006747857 * max(0.0, Q.sum_pt - 1167.447) * max(0.0, 0.06868286 - Q.abseta_12) / 0.5479112   # +0.1%  sum_pt > 1167 and abseta_12 < 0.06868
        - 0.0006063541 * max(0.0, Q.sum_z_dr - 0.1402186) * max(0.0, Q.soft6_pt - 2.162109) / 0.0007576602   # -0.1%  sum_z_dr > 0.1402 and soft6_pt > 2.162
        + 0.0004546967 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.psi_0p3 - 0.9777125) / 8.852804e-05   # +0.0%  log_sum_pt > 7.139 and psi_0p3 > 0.9777
        + 0.0003888891 * max(0.0, Q.sum_z_dr - 0.1564779) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.000215543   # +0.0%  sum_z_dr > 0.1565 and n_pt_above_50 > 5
        - 0.0002742077 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, 0.0001021922 - Q.eta_1) / 0.0001366493   # -0.0%  log_sum_pt > 7.063 and eta_1 < 0.0001022
        - 0.0002737328 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.absphi_13 - 0.01228943) / 0.0001757278   # -0.0%  log_sum_pt > 7.139 and absphi_13 > 0.01229
        + 0.0002246622 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, -0.004917145 - Q.phi_0) / 4.999945e-05   # +0.0%  log_sum_pt > 7.139 and phi_0 < -0.004917
        + 0.0001709945 * max(0.0, Q.sum_pt - 1167.447) * max(0.0, 0.002292633 - Q.eta_1) / 0.1946292   # +0.0%  sum_pt > 1167 and eta_1 < 0.002293
        - 0.0001665821 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, -0.0297699 - Q.phi_0) / 2.18725e-05   # -0.0%  log_sum_pt > 7.139 and phi_0 < -0.02977
        - 9.390927e-05 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.abseta_9 - 0.1726074) / 8.25197e-06   # -0.0%  log_sum_pt > 7.139 and abseta_9 > 0.1726
        - 7.976958e-05 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.phi_10 - 0.1347656) / 1.899899e-05   # -0.0%  log_sum_pt > 7.063 and phi_10 > 0.1348
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 34.8;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.80028 * (0.0706294
        - 0.1130627 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -11.3%  sum_z_dr > 0.1207
        - 0.08271139 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # -8.3%  LHA < 0.3332
        - 0.08008463 * max(0.0, 0.07031778 - Q.sum_z_dr) / 0.01719521   # -8.0%  sum_z_dr < 0.07032
        - 0.06132899 * max(0.0, Q.sum_z_dr - 0.1207452) * max(0.0, 0.4021783 - Q.max_dr) / 0.0001405604   # -6.1%  sum_z_dr > 0.1207 and max_dr < 0.4022
        - 0.05344644 * max(0.0, Q.sum_z_dr - 0.076787) / 0.01351025   # -5.3%  sum_z_dr > 0.07679
        - 0.05038749 * max(0.0, Q.sum_z_dr - 0.08589404) / 0.01080971   # -5.0%  sum_z_dr > 0.08589
        - 0.04408672 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # -4.4%  tau1 < 0.1507
        + 0.04071584 * max(0.0, Q.sum_z_dr - 0.08068193) / 0.0122472   # +4.1%  sum_z_dr > 0.08068
        + 0.04060419 * max(0.0, 0.2941033 - Q.LHA) / 0.05718399   # +4.1%  LHA < 0.2941
        - 0.03860395 * max(0.0, 0.1751567 - Q.tau1) / 0.09101058   # -3.9%  tau1 < 0.1752
        + 0.03712738 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # +3.7%  LHA < 0.3024
        + 0.02996826 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # +3.0%  LHA < 0.2601
        + 0.02848113 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # +2.8%  sd_rg < 0.3017
        + 0.02339817 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063   # +2.3%  n_dr_0p2_0p4 < 21
        + 0.01959223 * max(0.0, Q.sd_rg - 0.1596365) / 0.03553981   # +2.0%  sd_rg > 0.1596
        + 0.01662401 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +1.7%  tau21_b2 < 0.3425
        + 0.01639111 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +1.6%  e2 < 0.03481
        - 0.01516637 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 1156.659 - Q.sum_pt_top50) / 14.04661   # -1.5%  tau21_b2 < 0.3425 and sum_pt_top50 < 1157
        - 0.01417473 * max(0.0, Q.sd_rg - 0.2042612) / 0.02046744   # -1.4%  sd_rg > 0.2043
        - 0.01370571 * max(0.0, 0.07996447 - Q.C2) / 0.02445622   # -1.4%  C2 < 0.07996
        - 0.01323653 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # -1.3%  n_dr_0p1_0p2 < 17
        + 0.0128533 * max(0.0, Q.z_dr_0_0p05 - 0.6283153) / 0.1153015   # +1.3%  z_dr_0_0p05 > 0.6283
        - 0.01252896 * max(0.0, Q.sj3_dr_max - 0.2628766) / 0.03959245   # -1.3%  sj3_dr_max > 0.2629
        - 0.01251965 * max(0.0, 0.3715619 - Q.sj3_dr_max) / 0.1322803   # -1.3%  sj3_dr_max < 0.3716
        + 0.01206416 * max(0.0, Q.sj3_dr_max - 0.1717401) / 0.08948791   # +1.2%  sj3_dr_max > 0.1717
        - 0.01135376 * max(0.0, 7.876005e-05 - Q.e3) / 3.594075e-05   # -1.1%  e3 < 7.876e-05
        - 0.009909758 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -1.0%  n_dr_0p2_0p4 < 11
        + 0.00803019 * max(0.0, Q.sum_z_dr2_top10 - 0.0001295334) / 0.006638188   # +0.8%  sum_z_dr2_top10 > 0.0001295
        + 0.007063873 * max(0.0, Q.sd_rg - 0.15159) / 0.03939137   # +0.7%  sd_rg > 0.1516
        + 0.006190817 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +0.6%  lam2 < 0.001776
        + 0.006064642 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) / 3.82104   # +0.6%  n_dr_0p1_0p2 < 13
        + 0.0057722 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +0.6%  e2 < 0.01879
        + 0.005460104 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +0.5%  z_dr_0p1_0p2 < 0.1203
        + 0.005430699 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.01976735 - Q.sum_z_dr2_top10) / 0.001359663   # +0.5%  tau21_b2 < 0.3425 and sum_z_dr2_top10 < 0.01977
        + 0.005400454 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +0.5%  psi_0p3 > 0.9924
        - 0.004217219 * max(0.0, Q.psi_0p3 - 0.9638082) * max(0.0, 182.125 - Q.pt_1) / 1.522171   # -0.4%  psi_0p3 > 0.9638 and pt_1 < 182.1
        + 0.004015033 * max(0.0, 0.05347848 - Q.dr_6) / 0.01424699   # +0.4%  dr_6 < 0.05348
        + 0.00397918 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, Q.pt_8 - 16.73281) / 0.05815116   # +0.4%  psi_0p3 > 0.9924 and pt_8 > 16.73
        - 0.003786774 * max(0.0, Q.sj3_dr_max - 0.1999777) / 0.07046487   # -0.4%  sj3_dr_max > 0.2
        - 0.003784381 * max(0.0, 3.376709e-05 - Q.e3) / 8.295151e-06   # -0.4%  e3 < 3.377e-05
        + 0.003322202 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +0.3%  dr_0 < 0.06413
        - 0.003249516 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.05347848 - Q.dr_6) / 5.520862e-05   # -0.3%  psi_0p3 > 0.9924 and dr_6 < 0.05348
        + 0.00316428 * max(0.0, 0.03787151 - Q.M3) / 0.01370579   # +0.3%  M3 < 0.03787
        - 0.003157825 * max(0.0, 0.009970338 - Q.zdr_0) / 0.003273027   # -0.3%  zdr_0 < 0.00997
        + 0.002962234 * max(0.0, 0.5100475 - Q.tau21) / 0.1343519   # +0.3%  tau21 < 0.51
        + 0.002775959 * max(0.0, 0.07996447 - Q.C2) * max(0.0, Q.pt_3 - 55.15625) / 0.5532265   # +0.3%  C2 < 0.07996 and pt_3 > 55.16
        + 0.002359019 * max(0.0, Q.sum_z_dr - 0.076787) * max(0.0, 0.4021783 - Q.max_dr) / 0.0004881945   # +0.2%  sum_z_dr > 0.07679 and max_dr < 0.4022
        - 0.002343022 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.00727763 - Q.sum_z_dr2_top15) / 0.0001401748   # -0.2%  tau21_b2 < 0.3425 and sum_z_dr2_top15 < 0.007278
        - 0.001151318 * max(0.0, Q.sum_z_dr2_top10 - 0.0001295334) * max(0.0, 36.0 - Q.n_real_top40) / 0.006448485   # -0.1%  sum_z_dr2_top10 > 0.0001295 and n_real_top40 < 36
        - 0.0008776406 * max(0.0, Q.sd_rg - 0.15159) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.0411792   # -0.1%  sd_rg > 0.1516 and n_pt_above_50 > 4
        - 0.0007910449 * max(0.0, Q.z_dr_0p05_0p1 - 0.7108211) / 0.01402576   # -0.1%  z_dr_0p05_0p1 > 0.7108
        + 0.0005228776 * max(0.0, Q.sum_z_dr - 0.08589404) * max(0.0, 36.0 - Q.n_real_top40) / 0.0005904753   # +0.1%  sum_z_dr > 0.08589 and n_real_top40 < 36
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 10.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.00525 * (-0.0172191
        - 0.08554245 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # -8.6%  sd_rg < 0.3017
        - 0.07259879 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # -7.3%  sum_z_dr2_top15 < 0.009962
        + 0.06488913 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # +6.5%  e2 < 0.04359
        + 0.06280386 * max(0.0, 0.1881908 - Q.sd_rg) / 0.07278694   # +6.3%  sd_rg < 0.1882
        + 0.05628564 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # +5.6%  sum_z_dr2_top5 < 0.00833
        + 0.04480137 * max(0.0, 0.08068193 - Q.sum_z_dr) / 0.02368455   # +4.5%  sum_z_dr < 0.08068
        - 0.04446333 * max(0.0, Q.log_sum_pt - 6.915514) / 0.04811413   # -4.4%  log_sum_pt > 6.916
        + 0.04388316 * max(0.0, Q.sum_pt_top30 - 886.3438) / 117.6357   # +4.4%  sum_pt_top30 > 886.3
        + 0.04218997 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +4.2%  sj2_dr > 0.2232
        - 0.04171173 * max(0.0, Q.sj2_dr - 0.1512157) / 0.05988199   # -4.2%  sj2_dr > 0.1512
        - 0.0415381 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # -4.2%  tau1 < 0.07709
        + 0.03714834 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +3.7%  sum_pt > 907.9
        + 0.0337387 * max(0.0, 0.007678544 - Q.sum_z_dr2_top10) / 0.00347358   # +3.4%  sum_z_dr2_top10 < 0.007679
        - 0.03160888 * max(0.0, 0.02647293 - Q.z_dr_0p2_0p4) / 0.01168948   # -3.2%  z_dr_0p2_0p4 < 0.02647
        + 0.03009875 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) / 0.3666529   # +3.0%  z_dr_0_0p05 < 0.8459
        + 0.02179079 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +2.2%  psi_0p3 > 0.9897
        + 0.01877945 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +1.9%  z_dr_0p1_0p2 < 0.1203
        - 0.01865971 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) / 3.639534   # -1.9%  n_dr_0p2_0p4 > 8
        - 0.01853276 * max(0.0, Q.sum_pt_top40 - 1001.523) / 47.19068   # -1.9%  sum_pt_top40 > 1002
        - 0.01775311 * max(0.0, 0.001155057 - Q.sum_z_dr2_top3) / 0.000329166   # -1.8%  sum_z_dr2_top3 < 0.001155
        + 0.01706101 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +1.7%  lam2 < 0.001776
        - 0.01685213 * max(0.0, 2.883342e-05 - Q.e3) / 6.471346e-06   # -1.7%  e3 < 2.883e-05
        - 0.01593449 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, 689.25 - Q.sum_pt_top2) / 1.278535   # -1.6%  sum_z_dr2_top5 < 0.00833 and sum_pt_top2 < 689.2
        + 0.01525122 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # +1.5%  log_sum_pt > 7.017
        - 0.01281705 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.4299592 - Q.tau21_b2) / 0.001167326   # -1.3%  psi_0p3 > 0.9897 and tau21_b2 < 0.43
        - 0.0125846 * max(0.0, 0.02793599 - Q.e2) / 0.005715277   # -1.3%  e2 < 0.02794
        - 0.01142132 * max(0.0, Q.n_dr_0p05_0p1 - 6.0) / 6.42277   # -1.1%  n_dr_0p05_0p1 > 6
        + 0.01133641 * max(0.0, Q.psi_0p1 - 0.707925) * max(0.0, 0.1828389 - Q.dr_max_012) / 0.01897164   # +1.1%  psi_0p1 > 0.7079 and dr_max_012 < 0.1828
        - 0.01132613 * max(0.0, 0.1690338 - Q.sd_rg) / 0.06029557   # -1.1%  sd_rg < 0.169
        + 0.01109566 * max(0.0, 0.004169954 - Q.sum_z_dr2_top15) / 0.001130716   # +1.1%  sum_z_dr2_top15 < 0.00417
        - 0.008487028 * max(0.0, Q.LHA - 0.302389) / 0.02201444   # -0.8%  LHA > 0.3024
        - 0.006974811 * max(0.0, Q.sj2_dr - 0.2780918) / 0.009227968   # -0.7%  sj2_dr > 0.2781
        - 0.004368826 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 1.480811 - Q.soft5_pt) / 0.02368753   # -0.4%  z_dr_0p1_0p2 < 0.1203 and soft5_pt < 1.481
        - 0.004230593 * max(0.0, Q.tau1 - 0.1953848) / 0.000822026   # -0.4%  tau1 > 0.1954
        + 0.002430861 * max(0.0, 0.0004140594 - Q.soft5_z) / 1.144764e-05   # +0.2%  soft5_z < 0.0004141
        - 0.002300928 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.03006824 - Q.dr_4) / 0.000106252   # -0.2%  sj2_dr > 0.2232 and dr_4 < 0.03007
        - 0.002276189 * max(0.0, 0.08068193 - Q.sum_z_dr) * max(0.0, Q.dr1_11 - 0.1911348) / 0.0001794349   # -0.2%  sum_z_dr < 0.08068 and dr1_11 > 0.1911
        + 0.002223782 * max(0.0, Q.psi_0p1 - 0.707925) * max(0.0, Q.dr1_11 - 0.2195171) / 0.000707107   # +0.2%  psi_0p1 > 0.7079 and dr1_11 > 0.2195
        - 0.002208931 * max(0.0, 0.5297852 - Q.soft5_pt) / 0.01859865   # -0.2%  soft5_pt < 0.5298
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6641906512605043, 2.925503151260504, 0.21794180672268906, 0.4467440651260504, 0.7748081932773109, 1.0707384453781512, 0.5223641806722689, 0.4501723739495798, 1.2520199579831932, 0.9801596638655462, 1.179989600840336, 0.7778616596638656, 0.6515361344537816, 1.5572642857142858, 0.2684021008403361, 0.3751047268907563]
T = [3.9733414243040968, 2.521046357996324, 4.368814321822478, 4.428952708114495, 4.078731299238445]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +46%, n4 -17%, n9 +13%, n3 -8%, n5 -8%, n12 +4% ...
            + 0.4601768 * h[1] / H_AVG[1]
            - 0.1706265 * h[4] / H_AVG[4]
            + 0.1271964 * h[9] / H_AVG[9]
            - 0.08432652 * h[3] / H_AVG[3]
            - 0.08421269 * h[5] / H_AVG[5]
            + 0.03843208 * h[12] / H_AVG[12]
            + 0.02054175 * h[6] / H_AVG[6]
            + 0.009847033 * h[8] / H_AVG[8]
            - 0.00464026 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +22%, n1 -22%, n12 +9%, n11 -8%, n6 +5% ...
            - 0.3265445 * h[4] / H_AVG[4]
            + 0.2186948 * h[9] / H_AVG[9]
            - 0.217581 * h[1] / H_AVG[1]
            + 0.08883833 * h[12] / H_AVG[12]
            - 0.07713679 * h[11] / H_AVG[11]
            + 0.04532529 * h[6] / H_AVG[6]
            + 0.01080612 * h[2] / H_AVG[2]
            + 0.007759799 * h[8] / H_AVG[8]
            - 0.007313367 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +14%, n0 +11%, n11 +11%, n14 -8%, n7 -6% ...
            - 0.2507585 * h[8] / H_AVG[8]
            + 0.1416908 * h[5] / H_AVG[5]
            + 0.1140225 * h[0] / H_AVG[0]
            + 0.1057164 * h[11] / H_AVG[11]
            - 0.08447438 * h[14] / H_AVG[14]
            - 0.06440139 * h[7] / H_AVG[7]
            + 0.06096398 * h[4] / H_AVG[4]
            - 0.06058544 * h[12] / H_AVG[12]
            - 0.04907737 * h[9] / H_AVG[9]
            + 0.04473766 * h[3] / H_AVG[3]
            - 0.01609868 * h[15] / H_AVG[15]
            + 0.007472911 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -27%, n0 -21%, n5 +17%, n6 -11%, n7 +9%, n15 +5% ...
            - 0.2650217 * h[8] / H_AVG[8]
            - 0.2062027 * h[0] / H_AVG[0]
            + 0.1662092 * h[5] / H_AVG[5]
            - 0.1142573 * h[6] / H_AVG[6]
            + 0.09211404 * h[7] / H_AVG[7]
            + 0.04764025 * h[15] / H_AVG[15]
            + 0.04597137 * h[12] / H_AVG[12]
            + 0.04413019 * h[3] / H_AVG[3]
            - 0.01845316 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -35%, n10 +28%, n5 -12%, n8 +6%, n12 -6%, n4 +4% ...
            - 0.3460073 * h[13] / H_AVG[13]
            + 0.2847827 * h[10] / H_AVG[10]
            - 0.1230551 * h[5] / H_AVG[5]
            + 0.06475003 * h[8] / H_AVG[8]
            - 0.05990246 * h[12] / H_AVG[12]
            + 0.03561807 * h[4] / H_AVG[4]
            - 0.03448726 * h[15] / H_AVG[15]
            - 0.03104176 * h[7] / H_AVG[7]
            + 0.02035531 * h[0] / H_AVG[0]
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
