"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.
  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1
             avg_k = the term's average size on the entire training set (term / avg is 1 on average), so share_k is the
             fraction of the neuron's average input that comes from that if-statement (sign: pushes it up / down).
  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1
             h_j = neuron j (after max(0, .) and the network's rounding), avg_j = its average on the training jets.

How much each neuron matters (share of all class scores, averaged over the training jets):
  neuron 13:  14.1%   (on for 88% of jets)
  neuron  9:  12.0%   (on for 64% of jets)
  neuron  7:  10.3%   (on for 59% of jets)
  neuron  3:   8.7%   (on for 24% of jets)
  neuron 10:   8.1%   (on for 80% of jets)
  neuron  5:   7.3%   (on for 65% of jets)
  neuron  6:   7.3%   (on for 35% of jets)
  neuron  2:   7.3%   (on for 90% of jets)
  neuron 11:   6.4%   (on for 77% of jets)
  neuron  0:   3.8%   (on for 41% of jets)
  neuron 14:   3.7%   (on for 27% of jets)
  neuron  1:   3.4%   (on for 65% of jets)
  neuron  4:   3.2%   (on for 57% of jets)
  neuron 15:   2.8%   (on for 29% of jets)
  neuron  8:   1.1%   (on for 39% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.7% (the network: 65.8%); same class as the network for 90.5% of jets.

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
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.pair_mass_0_7          mass of particles 0 and 7 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.m01                    mass of particles 0 and 1 [GeV]
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.ptdr0_7                pT7 · ΔR(0, 7) [GeV]
  Q.z_2                    pT of particle 2 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.z_1st                  largest pT share
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_1               |Δη| of particle 1
  Q.abseta_2               |Δη| of particle 2
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_6               |Δη| of particle 6
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_5               |Δφ| of particle 5
  Q.absphi_6               |Δφ| of particle 6
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_3                  ΔR between particle 3 and the hardest particle
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr1_3                  ΔR between particle 3 and the 2nd-hardest particle
  Q.dr1_6                  ΔR between particle 6 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.phi_0                  Δφ of particle 0
  Q.phi_1                  Δφ of particle 1
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
  Q.tau4                   N-subjettiness τ4 (β=1)
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
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_6=pair_mass(0, 6),
        pair_mass_0_7=pair_mass(0, 7),
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        m01=pair_mass(0, 1),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        pt_0=pt[0],
        pt_1=pt[1],
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        pt_5=pt[5],
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        pt_7=pt[7],
        ptdr0_7=pt[7] * math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        z_2=z[2],
        z_3=z[3],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        z_1st=zs[0],
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_0=abs(eta[0]),
        abseta_1=abs(eta[1]),
        abseta_2=abs(eta[2]),
        abseta_4=abs(eta[4]),
        abseta_6=abs(eta[6]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_5=abs(phi[5]),
        absphi_6=abs(phi[6]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_3=math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr1_3=math.sqrt(dist2(1, 3)) if pt[3] > 0 else 0.0,
        dr1_6=math.sqrt(dist2(1, 6)) if pt[6] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        phi_0=phi[0],
        phi_1=phi[1],
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
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 28.55;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.54947 * (-0.06969721
        + 0.07941846 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +7.9%  sum_z_dr2 < 0.01324
        - 0.07345269 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) / 0.00156571   # -7.3%  lam1_plus_lam2 < 0.004372
        + 0.07316449 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +7.3%  lam1_plus_lam2 < 0.008678
        + 0.05279684 * max(0.0, Q.sj3_dr_max - 0.1070199) / 0.08518534   # +5.3%  sj3_dr_max > 0.107
        - 0.05164182 * max(0.0, 0.08475161 - Q.mass_over_sum_pt) / 0.03304118   # -5.2%  mass_over_sum_pt < 0.08475
        + 0.04501164 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +4.5%  sj3_dr_max < 0.3012
        + 0.03808275 * max(0.0, 0.03556091 - Q.e2) / 0.01371972   # +3.8%  e2 < 0.03556
        - 0.03460331 * max(0.0, 0.07608178 - Q.sum_z_dr) / 0.02796784   # -3.5%  sum_z_dr < 0.07608
        + 0.02751958 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # +2.8%  mass_over_sum_pt < 0.1309
        - 0.02718926 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # -2.7%  sj3_dr_max > 0.179
        + 0.02699011 * max(0.0, 0.006756161 - Q.sum_z_dr2_top3) / 0.003650633   # +2.7%  sum_z_dr2_top3 < 0.006756
        - 0.02649449 * max(0.0, 0.007929074 - Q.sum_z_dr2_top3) / 0.004560262   # -2.6%  sum_z_dr2_top3 < 0.007929
        - 0.02568075 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -2.6%  lam1 < 0.005433
        - 0.02514117 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -2.5%  centroid_offset > 0.00679
        + 0.02407945 * max(0.0, 0.003904593 - Q.mass_over_sum_pt_sq) / 0.001485439   # +2.4%  mass_over_sum_pt_sq < 0.003905
        + 0.02355444 * max(0.0, 0.0245477 - Q.e2) / 0.007455821   # +2.4%  e2 < 0.02455
        - 0.02270201 * max(0.0, 64.61873 - Q.mass) / 27.57279   # -2.3%  mass < 64.62
        - 0.02215593 * max(0.0, 0.001563465 - Q.C2_b2) / 0.0008800039   # -2.2%  C2_b2 < 0.001563
        - 0.02191194 * max(0.0, 29.6447 - Q.mass) / 7.34723   # -2.2%  mass < 29.64
        + 0.02133509 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +2.1%  planar_flow < 0.1484
        - 0.0205955 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -2.1%  centroid_offset > 0.0499
        - 0.0204109 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -2.0%  sum_z_dr < 0.08724
        + 0.01878535 * max(0.0, 0.0002758826 - Q.lam1) / 3.586469e-05   # +1.9%  lam1 < 0.0002759
        + 0.01489228 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) * max(0.0, 0.03532852 - Q.C3) / 3.6978e-05   # +1.5%  lam1_plus_lam2 < 0.004372 and C3 < 0.03533
        - 0.01488164 * max(0.0, 7.300726e-05 - Q.lam2) / 2.767163e-05   # -1.5%  lam2 < 7.301e-05
        - 0.01277583 * max(0.0, 56.92035 - Q.mass) / 21.78475   # -1.3%  mass < 56.92
        - 0.01200963 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.2%  sj3_dr_max > 0.2337
        - 0.01078784 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05557716 - Q.z_7) / 0.001734349   # -1.1%  sj3_dr_max < 0.3012 and z_7 < 0.05558
        - 0.01018602 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, Q.z_7 - 0.01685855) / 0.0006290825   # -1.0%  log_sum_pt > 6.67 and z_7 > 0.01686
        - 0.010171 * max(0.0, 0.01882765 - Q.sum_z_dr2) / 0.01314296   # -1.0%  sum_z_dr2 < 0.01883
        + 0.008834614 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 0.07232166 - Q.dr_4) / 0.001949106   # +0.9%  log_sum_pt > 6.67 and dr_4 < 0.07232
        + 0.008572867 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +0.9%  lam1 < 0.006507
        + 0.007842437 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +0.8%  log_sum_pt > 6.378
        + 0.007812143 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, 1.002471 - Q.D2) / 0.001008123   # +0.8%  sum_z_dr2 < 0.01324 and D2 < 1.002
        - 0.006900554 * max(0.0, Q.sum_pt - 901.5938) / 12.40362   # -0.7%  sum_pt > 901.6
        + 0.006587052 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2) / 2.36013e-06   # +0.7%  lam2 < 7.301e-05 and D2_b2 < 0.267
        + 0.005183822 * max(0.0, Q.centroid_offset - 0.006789738) * max(0.0, 223.375 - Q.pt_1) / 1.309723   # +0.5%  centroid_offset > 0.00679 and pt_1 < 223.4
        + 0.004880318 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.08525808 - Q.dr_2) / 1.41829e-06   # +0.5%  lam2 < 7.301e-05 and dr_2 < 0.08526
        + 0.004586139 * max(0.0, 21.78408 - Q.mass) / 4.431023   # +0.5%  mass < 21.78
        + 0.004558353 * max(0.0, Q.centroid_offset - 0.006789738) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.02705531   # +0.5%  centroid_offset > 0.00679 and n_pt_above_50 < 7
        - 0.004174532 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.06336451 - Q.dr_4) / 1.344914   # -0.4%  sum_pt_top5 > 687.4 and dr_4 < 0.06336
        + 0.004125399 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.4%  sum_pt_top5 > 687.4
        - 0.004077694 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # -0.4%  sum_z_dr2_top5 < 0.00833
        - 0.003577086 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 2.0 - Q.n_dr_0p05_0p1) / 0.1809599   # -0.4%  sj3_dr_max < 0.3012 and n_dr_0p05_0p1 < 2
        - 0.003122864 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # -0.3%  lam1 < 0.006507 and D2 < 0.8757
        - 0.002896639 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 358.375 - Q.sum_pt_top2) / 2.426548   # -0.3%  planar_flow < 0.1484 and sum_pt_top2 < 358.4
        + 0.002826668 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, Q.phi_0 - -0.008995056) / 0.0001281354   # +0.3%  sum_z_dr2 < 0.01324 and phi_0 > -0.008995
        - 0.002372091 * max(0.0, 0.04889979 - Q.sj3_dr_max) / 0.006377687   # -0.2%  sj3_dr_max < 0.0489
        - 0.001770121 * max(0.0, 0.01882765 - Q.sum_z_dr2) * max(0.0, 46.125 - Q.pt_6) / 0.1159788   # -0.2%  sum_z_dr2 < 0.01883 and pt_6 < 46.12
        - 0.001759018 * max(0.0, 0.007929074 - Q.sum_z_dr2_top3) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0005550791   # -0.2%  sum_z_dr2_top3 < 0.007929 and D2_b2 < 0.5327
        - 0.001714063 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 0.06473447 - Q.z_7) / 3.579716e-05   # -0.2%  centroid_offset > 0.02077 and z_7 < 0.06473
        - 0.00150094 * max(0.0, 64.61873 - Q.mass) * max(0.0, 0.1830092 - Q.D2_b2) / 0.3382769   # -0.2%  mass < 64.62 and D2_b2 < 0.183
        + 0.001442853 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.4160096 - Q.pt_dispersion) / 0.0006933006   # +0.1%  planar_flow < 0.1484 and pt_dispersion < 0.416
        - 0.001386794 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05744392 - Q.D2_b2) / 0.0005632394   # -0.1%  sj3_dr_max < 0.3012 and D2_b2 < 0.05744
        + 0.001236281 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 28.78966 - Q.m01) / 1.085363   # +0.1%  log_sum_pt > 6.67 and m01 < 28.79
        - 0.00123546 * max(0.0, 6.080494 - Q.log_sum_pt) / 0.006097657   # -0.1%  log_sum_pt < 6.08
        - 0.001131597 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.01837778) / 2.056482e-05   # -0.1%  sum_z_dr2 < 0.01324 and centroid_offset > 0.01838
        - 0.001037087 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.mean_eta - 6.288824e-05) / 0.000715688   # -0.1%  log_sum_pt > 6.378 and mean_eta > 6.289e-05
        - 0.0009347677 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -0.1%  log_sum_pt > 6.67
        + 0.0007099213 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, -0.003096655 - Q.mean_phi) / 2.294381e-05   # +0.1%  sum_z_dr2 < 0.01324 and mean_phi < -0.003097
        - 0.0006751872 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, Q.dr0_6 - 0.1755206) / 0.0008435917   # -0.1%  planar_flow < 0.1484 and dr0_6 > 0.1755
        - 0.000630571 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.02270492 - Q.dr_2) / 2.517754e-05   # -0.1%  planar_flow < 0.1484 and dr_2 < 0.0227
        - 0.0006046138 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 0.05056028 - Q.dr_2) / 1.335632e-05   # -0.1%  centroid_offset > 0.02077 and dr_2 < 0.05056
        - 0.0005801875 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) * max(0.0, 0.7459513 - Q.D2) / 7.193909e-06   # -0.1%  lam1_plus_lam2 < 0.004372 and D2 < 0.746
        - 0.0002970543 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.pt_7 - 33.21875) / 68.87156   # -0.0%  sum_pt > 901.6 and pt_7 > 33.22
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 19.31;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.30848 * (-0.003234525
        - 0.1081488 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -10.8%  sum_z_dr2 < 0.008678
        + 0.07996963 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) / 0.002578466   # +8.0%  mass_over_sum_pt_sq < 0.005833
        - 0.07903868 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -7.9%  sum_pt_top5 > 531.2
        - 0.07371151 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 0.002544039   # -7.4%  lam1_plus_lam2 < 0.006097
        - 0.0625322 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -6.3%  sum_z_dr < 0.1019
        + 0.06132686 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +6.1%  log_sum_pt > 6.503
        + 0.04445232 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # +4.4%  sum_zz_dr2 < 0.008169
        + 0.04396317 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # +4.4%  z_7 > 0.02321
        + 0.0408944 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +4.1%  log_sum_pt > 6.378
        + 0.03973142 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +4.0%  lam1 < 0.005954
        + 0.03235199 * max(0.0, Q.sum_pt_top5 - 531.1875) * max(0.0, 0.01392641 - Q.zdr_6) / 1.245695   # +3.2%  sum_pt_top5 > 531.2 and zdr_6 < 0.01393
        - 0.03004738 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -3.0%  z_7 < 0.06165
        + 0.02099357 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +2.1%  e3 < 0.0005117
        - 0.02058514 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, 0.008654951 - Q.zdr_6) / 0.000799435   # -2.1%  log_sum_pt > 6.503 and zdr_6 < 0.008655
        + 0.02055339 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # +2.1%  sum_z_dr < 0.0717
        + 0.01955145 * max(0.0, Q.sum_pt_top5 - 367.5938) / 230.8403   # +2.0%  sum_pt_top5 > 367.6
        - 0.01883288 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1) / 0.1196218   # -1.9%  log_sum_pt > 6.378 and z_dr_0p05_0p1 < 0.7509
        + 0.01554797 * max(0.0, Q.log_sum_pt - 6.605974) / 0.07073019   # +1.6%  log_sum_pt > 6.606
        - 0.01490542 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # -1.5%  tau1 < 0.07284
        + 0.01433149 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.06297984 - Q.tau2) / 0.0007776805   # +1.4%  z_7 < 0.06165 and tau2 < 0.06298
        + 0.01253311 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +1.3%  lam1 < 0.008376
        + 0.01233374 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.01403324 - Q.sum_z_dr2_top2) / 0.0001680113   # +1.2%  z_7 < 0.06165 and sum_z_dr2_top2 < 0.01403
        - 0.01222674 * max(0.0, Q.z_7 - 0.04624032) / 0.01204993   # -1.2%  z_7 > 0.04624
        + 0.01182104 * max(0.0, 49.6681 - Q.mass) / 17.06893   # +1.2%  mass < 49.67
        + 0.01107625 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.008921136 - Q.mean_phi2) / 0.0001024498   # +1.1%  z_7 < 0.06165 and mean_phi2 < 0.008921
        - 0.008647261 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # -0.9%  sum_z_dr2 < 0.00502
        + 0.00864253 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +0.9%  sj3_dr_max > 0.169
        + 0.008173883 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +0.8%  pt_7 > 34.53
        + 0.008026392 * max(0.0, 0.008678045 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.02076709) / 7.418362e-06   # +0.8%  sum_z_dr2 < 0.008678 and centroid_offset > 0.02077
        - 0.007600937 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -0.8%  zdr_0 < 0.02118
        - 0.007146661 * max(0.0, 0.0008722282 - Q.lam1) / 0.0001882079   # -0.7%  lam1 < 0.0008722
        - 0.005765262 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # -0.6%  lam1 < 0.008376 and centroid_offset > 0.02077
        - 0.004595106 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236) / 1.769648e-05   # -0.5%  mass_over_sum_pt_sq < 0.005833 and centroid_offset > 0.008092
        - 0.00439572 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.00323438   # -0.4%  lam1 < 0.008376 and n_pt_above_50 > 5
        - 0.004360357 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 1.679198 - Q.D2) / 0.005760154   # -0.4%  z_7 < 0.06165 and D2 < 1.679
        - 0.0043433 * max(0.0, 25.57812 - Q.pt_7) / 1.327389   # -0.4%  pt_7 < 25.58
        - 0.004250958 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.04875823) / 0.002321449   # -0.4%  z_7 < 0.06165 and z_dr_0p05_0p1 > 0.04876
        + 0.003872851 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.centroid_offset - 0.009480685) / 0.0009010889   # +0.4%  log_sum_pt > 6.378 and centroid_offset > 0.009481
        - 0.003576588 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1484197 - Q.planar_flow) / 0.0001369594   # -0.4%  lam1 < 0.008376 and planar_flow < 0.1484
        + 0.003267738 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.sj2_dr - 0.1294903) / 0.1877842   # +0.3%  pt_7 > 34.53 and sj2_dr > 0.1295
        - 0.002644161 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.5700307   # -0.3%  sj3_dr_max > 0.169 and sj3_pair_mass_min > 5.744
        + 0.002022426 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 1.432482 - Q.D2) / 0.01970563   # +0.2%  log_sum_pt > 6.606 and D2 < 1.432
        - 0.002010451 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 52.90625 - Q.pt_6) / 13.55761   # -0.2%  pt_7 > 34.53 and pt_6 < 52.91
        - 0.001846054 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.2%  pt_7 > 53.44
        - 0.001644077 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 77.625 - Q.pt_3) / 25.26458   # -0.2%  pt_7 > 34.53 and pt_3 < 77.62
        + 0.001132502 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 3.175597   # +0.1%  pt_7 > 34.53 and n_dr_0p1_0p2 > 1
        - 0.0005741934 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, 0.03885671 - Q.D2_b2) / 4.310709e-07   # -0.1%  mass_over_sum_pt_sq < 0.005833 and D2_b2 < 0.03886
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 16.19;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.1908 * (0.236915
        - 0.1099991 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # -11.0%  z_7 > 0.02321
        - 0.105453 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -10.5%  pt_7 < 53.44
        - 0.07278264 * max(0.0, Q.LHA - 0.111565) / 0.1352185   # -7.3%  LHA > 0.1116
        - 0.06512014 * max(0.0, 716.8828 - Q.sum_pt_top5) / 152.5573   # -6.5%  sum_pt_top5 < 716.9
        + 0.06107471 * max(0.0, Q.pt_7 - 20.125) / 15.07003   # +6.1%  pt_7 > 20.12
        + 0.05142255 * max(0.0, 788.4484 - Q.sum_pt) / 112.1576   # +5.1%  sum_pt < 788.4
        + 0.04142345 * max(0.0, 43.5 - Q.pt_7) / 10.28861   # +4.1%  pt_7 < 43.5
        + 0.03956184 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +4.0%  lam1 < 0.012
        + 0.03674315 * max(0.0, 63.68899 - Q.sj3_pair_mass_max) / 31.57762   # +3.7%  sj3_pair_mass_max < 63.69
        + 0.03200178 * max(0.0, 840.0195 - Q.sum_pt) / 148.4153   # +3.2%  sum_pt < 840
        + 0.03056496 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # +3.1%  log_sum_pt < 6.606
        + 0.02645142 * max(0.0, Q.pt_6 - 27.57812) / 13.69978   # +2.6%  pt_6 > 27.58
        - 0.0238279 * max(0.0, 0.0003193707 - Q.sum_z_dr2) / 4.035777e-05   # -2.4%  sum_z_dr2 < 0.0003194
        - 0.02252376 * max(0.0, Q.z_6 - 0.02886576) / 0.03195851   # -2.3%  z_6 > 0.02887
        + 0.02252109 * max(0.0, 0.0003193707 - Q.sum_z_dr2) * max(0.0, 36.76827 - Q.mass_top2) / 0.001427268   # +2.3%  sum_z_dr2 < 0.0003194 and mass_top2 < 36.77
        - 0.02247483 * max(0.0, 63.68899 - Q.sj3_pair_mass_max) * max(0.0, 0.06810151 - Q.z_7) / 0.6284113   # -2.2%  sj3_pair_mass_max < 63.69 and z_7 < 0.0681
        - 0.02215404 * max(0.0, 0.1484084 - Q.sum_z_dr) / 0.0905999   # -2.2%  sum_z_dr < 0.1484
        - 0.02099589 * max(0.0, Q.sum_pt_top5 - 605.1875) / 67.49708   # -2.1%  sum_pt_top5 > 605.2
        + 0.0176043 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +1.8%  z_7 < 0.07149
        - 0.01739984 * max(0.0, Q.mass - 36.22941) / 14.20528   # -1.7%  mass > 36.23
        + 0.01091501 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # +1.1%  zdr_0 < 0.02118
        + 0.01058256 * max(0.0, Q.z_7 - 0.04939969) / 0.01023002   # +1.1%  z_7 > 0.0494
        + 0.009921029 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +1.0%  sum_z_dr2 < 0.00502
        + 0.007942665 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # +0.8%  log_sum_pt < 6.464
        + 0.007663527 * max(0.0, 840.0195 - Q.sum_pt) * max(0.0, 0.02164863 - Q.dr_5) / 0.1765901   # +0.8%  sum_pt < 840 and dr_5 < 0.02165
        + 0.007642254 * max(0.0, Q.mass - 15.45403) / 27.25251   # +0.8%  mass > 15.45
        - 0.00736795 * max(0.0, 63.68899 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.01096064) / 0.2223898   # -0.7%  sj3_pair_mass_max < 63.69 and centroid_offset > 0.01096
        + 0.006398823 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # +0.6%  log_sum_pt > 6.843
        - 0.006344299 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # -0.6%  lam2 < 0.0001948
        - 0.00582979 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, 0.02164863 - Q.dr_5) / 0.1197721   # -0.6%  sum_pt < 788.4 and dr_5 < 0.02165
        - 0.005685483 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.6947818 - Q.planar_flow) / 0.009090069   # -0.6%  z_7 < 0.07149 and planar_flow < 0.6948
        - 0.005555248 * max(0.0, Q.pt_6 - 27.57812) * max(0.0, 0.7974684 - Q.planar_flow) / 7.391558   # -0.6%  pt_6 > 27.58 and planar_flow < 0.7975
        + 0.005279177 * max(0.0, 559.6875 - Q.sum_pt) * max(0.0, 4.721224 - Q.D2_b2) / 63.42935   # +0.5%  sum_pt < 559.7 and D2_b2 < 4.721
        - 0.005017092 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.max_dr - 0.0931108) / 3.745922e-05   # -0.5%  lam1 < 0.005954 and max_dr > 0.09311
        - 0.004578972 * max(0.0, 43.5 - Q.pt_7) * max(0.0, 3.032555 - Q.D2_b2) / 18.19911   # -0.5%  pt_7 < 43.5 and D2_b2 < 3.033
        - 0.004149659 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # -0.4%  pt_6 < 41.22
        - 0.004020176 * max(0.0, 0.03243272 - Q.z_7) / 0.002061024   # -0.4%  z_7 < 0.03243
        - 0.003758222 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.4%  sum_pt_top5 > 840
        - 0.003603774 * max(0.0, Q.z_7 - 0.04939969) * max(0.0, 0.0623951 - Q.sj3_dr_min) / 0.0002978985   # -0.4%  z_7 > 0.0494 and sj3_dr_min < 0.0624
        + 0.003565803 * max(0.0, Q.z_top5 - 0.8770155) / 0.00720125   # +0.4%  z_top5 > 0.877
        - 0.003419368 * max(0.0, 559.6875 - Q.sum_pt) / 16.18216   # -0.3%  sum_pt < 559.7
        - 0.002971926 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # -0.3%  sum_z_dr < 0.007674
        - 0.002517464 * max(0.0, 0.02164863 - Q.dr_5) / 0.002569084   # -0.3%  dr_5 < 0.02165
        + 0.002397377 * max(0.0, 15.95929 - Q.sj3_pair_mass_min) * max(0.0, Q.D2 - 0.6203774) / 10.04714   # +0.2%  sj3_pair_mass_min < 15.96 and D2 > 0.6204
        + 0.002179653 * max(0.0, Q.sum_pt_top5 - 839.9547) * max(0.0, 2.326647 - Q.D2_b2) / 5.890686   # +0.2%  sum_pt_top5 > 840 and D2_b2 < 2.327
        + 0.002066726 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 0.0003061234 - Q.lam2) / 1.525985e-05   # +0.2%  log_sum_pt < 6.606 and lam2 < 0.0003061
        - 0.001998199 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 3.032555 - Q.D2_b2) / 0.005136309   # -0.2%  log_sum_pt > 6.896 and D2_b2 < 3.033
        - 0.001994859 * max(0.0, 33.02656 - Q.pt_5) / 1.052976   # -0.2%  pt_5 < 33.03
        - 0.001892768 * max(0.0, 0.004430536 - Q.centroid_offset) / 0.0004182362   # -0.2%  centroid_offset < 0.004431
        + 0.001835919 * max(0.0, Q.mass_top5 - 49.18618) / 2.747087   # +0.2%  mass_top5 > 49.19
        + 0.001534732 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.2%  log_sum_pt > 6.896
        + 0.001489308 * max(0.0, Q.m012 - 42.18339) / 1.09247   # +0.1%  m012 > 42.18
        + 0.001165648 * max(0.0, 0.03672711 - Q.z_5) / 0.001007435   # +0.1%  z_5 < 0.03673
        + 0.001156928 * max(0.0, 0.007673833 - Q.sum_z_dr) * max(0.0, 71.6875 - Q.pt_4) / 0.003882715   # +0.1%  sum_z_dr < 0.007674 and pt_4 < 71.69
        - 0.0009620276 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.04416271 - Q.dr_5) / 9.036365e-05   # -0.1%  log_sum_pt > 6.896 and dr_5 < 0.04416
        + 0.0009360998 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.04416271 - Q.dr_5) / 0.0001724025   # +0.1%  log_sum_pt > 6.843 and dr_5 < 0.04416
        + 0.0007148187 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.002460108 - Q.zdr_5) / 5.982837e-06   # +0.1%  log_sum_pt > 6.896 and zdr_5 < 0.00246
        - 0.000540094 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.pt_6 - 41.21875) / 0.06090836   # -0.1%  log_sum_pt > 6.843 and pt_6 > 41.22
        + 0.0003041947 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, 0.03623337 - Q.max_dr) / 0.1060521   # +0.0%  sum_pt < 788.4 and max_dr < 0.03623
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 17.33;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.33226 * (-0.2487111
        - 0.1943663 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # -19.4%  sum_z_dr2 > 0.008678
        + 0.1108617 * max(0.0, Q.sum_z_dr - 0.04081947) / 0.0270611   # +11.1%  sum_z_dr > 0.04082
        - 0.08184555 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # -8.2%  tau1 > 0.05357
        + 0.06847416 * max(0.0, Q.lam1_plus_lam2 - 0.007520088) / 0.002470136   # +6.8%  lam1_plus_lam2 > 0.00752
        + 0.06129691 * max(0.0, Q.mass_over_sum_pt - 0.0681391) / 0.01539157   # +6.1%  mass_over_sum_pt > 0.06814
        + 0.05795703 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +5.8%  lam1 > 0.008376
        - 0.05217356 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # -5.2%  lam1_plus_lam2 > 0.01324
        + 0.0484001 * max(0.0, Q.sum_z_dr - 0.07608178) / 0.01059033   # +4.8%  sum_z_dr > 0.07608
        + 0.04358429 * max(0.0, Q.sj2_dr - 0.1872617) / 0.02732165   # +4.4%  sj2_dr > 0.1873
        + 0.03979935 * max(0.0, Q.sum_z_dr - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494) / 0.008781747   # +4.0%  sum_z_dr > 0.04082 and log_sum_pt > 6.08
        + 0.0345096 * max(0.0, Q.max_dr - 0.1027585) / 0.04505724   # +3.5%  max_dr > 0.1028
        - 0.01873665 * max(0.0, Q.mass - 64.61873) / 3.274818   # -1.9%  mass > 64.62
        + 0.01748268 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # +1.7%  tau1 > 0.1028
        - 0.0173791 * max(0.0, 0.004372139 - Q.sum_z_dr2) / 0.00156571   # -1.7%  sum_z_dr2 < 0.004372
        + 0.01656346 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # +1.7%  lam1 > 0.01643
        + 0.01249535 * max(0.0, Q.lam1_plus_lam2 - 0.007520088) * max(0.0, Q.sj3_pairmin_over_m - 0.07708997) / 0.0004577253   # +1.2%  lam1_plus_lam2 > 0.00752 and sj3_pairmin_over_m > 0.07709
        + 0.01170976 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # +1.2%  e2 > 0.06344
        - 0.01068263 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr) / 0.0001196243   # -1.1%  mass_over_sum_pt > 0.06814 and sj2_dr < 0.218
        - 0.009925351 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113) / 0.39507   # -1.0%  sj2_dr > 0.1873 and sj2_mass1 > 2.25
        + 0.009491925 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 0.07861328 - Q.abseta_0) / 0.0002969212   # +0.9%  centroid_offset > 0.01096 and abseta_0 < 0.07861
        + 0.00924964 * max(0.0, 0.05226226 - Q.z_dr_0_0p05) / 0.01510942   # +0.9%  z_dr_0_0p05 < 0.05226
        + 0.008306522 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6) / 0.005634141   # +0.8%  lam2 > 0.001131 and pt_6 < 56.53
        - 0.008236714 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # -0.8%  sum_z_dr > 0.08724
        - 0.006980255 * max(0.0, Q.sj2_dr - 0.1492731) / 0.04449993   # -0.7%  sj2_dr > 0.1493
        + 0.006730135 * max(0.0, Q.sd_mass - 62.73432) / 3.311364   # +0.7%  sd_mass > 62.73
        + 0.005934852 * max(0.0, -0.004664942 - Q.mean_eta) / 0.003754527   # +0.6%  mean_eta < -0.004665
        - 0.005366745 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.pt_6 - 31.90625) / 0.1358885   # -0.5%  mass_over_sum_pt > 0.06814 and pt_6 > 31.91
        + 0.004028012 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, Q.z_6 - 0.05096142) / 7.479719e-06   # +0.4%  lam2 > 0.001131 and z_6 > 0.05096
        - 0.003912034 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.01084112 - Q.zdr_1) / 0.0001011412   # -0.4%  max_dr > 0.1028 and zdr_1 < 0.01084
        + 0.00375645 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # +0.4%  lam1 > 0.012
        - 0.003731144 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.1950293   # -0.4%  mass_over_sum_pt > 0.06814 and sj3_pair_mass_min > 5.744
        - 0.003226456 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.05268713 - Q.dr_3) / 0.000155932   # -0.3%  sj2_dr > 0.1873 and dr_3 < 0.05269
        + 0.003188214 * max(0.0, Q.mean_eta - 0.01772426) / 0.001375178   # +0.3%  mean_eta > 0.01772
        + 0.001928564 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sj2_mass1 - 16.86126) / 0.02534836   # +0.2%  e2 > 0.06344 and sj2_mass1 > 16.86
        - 0.001792041 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # -0.2%  sum_z_dr2 > 0.01883
        - 0.001577788 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.05744392 - Q.D2_b2) / 0.0002412597   # -0.2%  max_dr > 0.1028 and D2_b2 < 0.05744
        - 0.0008884229 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.z_top5 - 0.7773372) / 2.699283e-05   # -0.1%  e2 > 0.06344 and z_top5 > 0.7773
        - 0.0008694115 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, -0.05963135 - Q.eta_1) / 0.0004225811   # -0.1%  max_dr > 0.1028 and eta_1 < -0.05963
        - 0.0008409209 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, Q.abseta_0 - 0.1057739) / 0.0002451453   # -0.1%  max_dr > 0.1028 and abseta_0 > 0.1058
        + 0.0008401515 * max(0.0, Q.sum_z_dr - 0.07608178) * max(0.0, Q.eta_0 - 0.07952881) / 0.0001300296   # +0.1%  sum_z_dr > 0.07608 and eta_0 > 0.07953
        + 0.000653995 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, Q.pt_5 - 59.125) / 0.0351992   # +0.1%  tau1 > 0.05357 and pt_5 > 59.12
        - 0.0002260443 * max(0.0, Q.sj2_dr - 0.2687922) * max(0.0, 0.02160287 - Q.z_6) / 2.621561e-06   # -0.0%  sj2_dr > 0.2688 and z_6 < 0.0216
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 99.21;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 99.21161 * (-0.03785883
        + 0.2875011 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # +28.8%  sum_zz_dr2 < 0.01166
        - 0.2318847 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -23.2%  mass_over_sum_pt_sq < 0.01166
        + 0.04566903 * max(0.0, 0.01716248 - Q.sum_zz_dr2) / 0.0120354   # +4.6%  sum_zz_dr2 < 0.01716
        - 0.0455957 * max(0.0, 0.01882765 - Q.sum_z_dr2) / 0.01314296   # -4.6%  sum_z_dr2 < 0.01883
        - 0.03219951 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -3.2%  sum_z_dr2 < 0.008678
        - 0.03054844 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # -3.1%  lam1_plus_lam2 < 0.01324
        + 0.02716046 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # +2.7%  C2_b2 < 0.009033
        + 0.02677903 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +2.7%  mass < 76.66
        + 0.02537252 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +2.5%  N2 < 0.2233
        + 0.01872206 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # +1.9%  sum_z_dr2 < 0.002635
        + 0.01734371 * max(0.0, 0.1245537 - Q.sum_z_dr) / 0.06848162   # +1.7%  sum_z_dr < 0.1246
        - 0.01438981 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -1.4%  mass < 69.61
        - 0.01310027 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -1.3%  lam2 < 0.0005373
        + 0.01230345 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +1.2%  sj3_dr_max > 0.169
        - 0.01189361 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -1.2%  lam2 < 0.001131
        - 0.01173551 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.2%  sj3_dr_max > 0.2337
        - 0.01044016 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266) / 0.01381411   # -1.0%  N2 < 0.2233 and eccentricity > 0.7117
        + 0.01028292 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # +1.0%  sum_z_dr2_top5 < 0.00833
        + 0.0101128 * max(0.0, Q.e2 - 0.009668065) / 0.02044666   # +1.0%  e2 > 0.009668
        - 0.009120867 * max(0.0, Q.sum_z_dr - 0.05464922) / 0.01938482   # -0.9%  sum_z_dr > 0.05465
        + 0.009021203 * max(0.0, Q.sum_z_dr2 - 0.003562611) / 0.004066872   # +0.9%  sum_z_dr2 > 0.003563
        - 0.007751749 * max(0.0, 0.05119235 - Q.C2) / 0.02746756   # -0.8%  C2 < 0.05119
        - 0.007559161 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # -0.8%  lam1 < 0.002464
        + 0.007021668 * max(0.0, 0.03981924 - Q.zdr_0) / 0.02538306   # +0.7%  zdr_0 < 0.03982
        - 0.006975968 * max(0.0, 49.91626 - Q.sd_mass) / 22.60916   # -0.7%  sd_mass < 49.92
        - 0.006777094 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -0.7%  sum_pt < 739.5
        - 0.004574815 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.002796532   # -0.5%  centroid_offset < 0.01838 and z_dr_0p05_0p1 < 0.5883
        - 0.004536061 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -0.5%  mass_over_sum_pt > 0.09041
        - 0.004133913 * max(0.0, 76.6557 - Q.mass) * max(0.0, 0.875672 - Q.D2) / 2.2476   # -0.4%  mass < 76.66 and D2 < 0.8757
        + 0.00378668 * max(0.0, Q.max_dr - 0.1117619) / 0.04033009   # +0.4%  max_dr > 0.1118
        + 0.003753533 * max(0.0, Q.tau1 - 0.07283629) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.001837197   # +0.4%  tau1 > 0.07284 and sj3_dr_min < 0.209
        - 0.003650415 * max(0.0, 1.002471 - Q.D2) * max(0.0, 53.4375 - Q.pt_7) / 3.084564   # -0.4%  D2 < 1.002 and pt_7 < 53.44
        + 0.003649048 * max(0.0, 0.875672 - Q.D2) / 0.1390856   # +0.4%  D2 < 0.8757
        - 0.003379617 * max(0.0, 1.340118e-05 - Q.e3) / 5.081834e-06   # -0.3%  e3 < 1.34e-05
        - 0.00314811 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.1591797 - Q.absphi_6) / 4.786038e-05   # -0.3%  lam2 < 0.0005373 and absphi_6 < 0.1592
        + 0.002540564 * max(0.0, 0.01165737 - Q.sum_zz_dr2) * max(0.0, 0.875672 - Q.D2) / 0.0005867961   # +0.3%  sum_zz_dr2 < 0.01166 and D2 < 0.8757
        + 0.002321066 * max(0.0, 1.002471 - Q.D2) / 0.1871081   # +0.2%  D2 < 1.002
        - 0.002077074 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 7.11886e-05   # -0.2%  N2 < 0.2233 and sum_zz_dr2 > 0.01166
        + 0.002063947 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +0.2%  tau1 > 0.1136
        - 0.002011563 * max(0.0, 37.76455 - Q.mass_top5) / 16.71307   # -0.2%  mass_top5 < 37.76
        - 0.001864279 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1626038 - Q.abseta_7) / 0.005313035   # -0.2%  N2 < 0.2233 and abseta_7 < 0.1626
        - 0.001547686 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 60.63098 - Q.mass) / 0.3406587   # -0.2%  N2 < 0.2233 and mass < 60.63
        - 0.001486284 * max(0.0, Q.sd_rg - 0.1667615) / 0.02547986   # -0.1%  sd_rg > 0.1668
        + 0.001398538 * max(0.0, Q.sum_z_dr - 0.1019409) / 0.005400003   # +0.1%  sum_z_dr > 0.1019
        - 0.00115141 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # -0.1%  e2 > 0.04447
        - 0.001041336 * max(0.0, Q.sd_rg - 0.1667615) * max(0.0, 0.3331409 - Q.z_1st) / 0.001855799   # -0.1%  sd_rg > 0.1668 and z_1st < 0.3331
        + 0.001035695 * max(0.0, Q.max_dr - 0.2215867) / 0.008157178   # +0.1%  max_dr > 0.2216
        + 0.0009635077 * max(0.0, Q.sd_rg - 0.2037854) * max(0.0, 176.25 - Q.pt_0) / 0.6276792   # +0.1%  sd_rg > 0.2038 and pt_0 < 176.2
        - 0.0009134179 * max(0.0, Q.sd_rg - 0.2037854) / 0.01566501   # -0.1%  sd_rg > 0.2038
        + 0.0008808161 * max(0.0, Q.sj3_dr_max - 0.3456459) / 0.006655619   # +0.1%  sj3_dr_max > 0.3456
        + 0.0006633278 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.abseta_0 - 0.02487183) / 0.001302228   # +0.1%  N2 < 0.2233 and abseta_0 > 0.02487
        + 0.000636988 * max(0.0, 739.5 - Q.sum_pt) * max(0.0, Q.n_dr_0p05_0p1 - 0.0) / 212.6759   # +0.1%  sum_pt < 739.5 and n_dr_0p05_0p1 > 0
        - 0.0006329498 * max(0.0, 1.002471 - Q.D2) * max(0.0, 0.1132812 - Q.absphi_5) / 0.01081292   # -0.1%  D2 < 1.002 and absphi_5 < 0.1133
        + 0.0004824596 * max(0.0, Q.sd_rg - 0.324646) / 0.001748965   # +0.0%  sd_rg > 0.3246
        + 0.0003994194 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, 130.375 - Q.pt_0) / 0.287519   # +0.0%  sj3_dr_max > 0.2337 and pt_0 < 130.4
        + 0.0003666447 * max(0.0, Q.mean_eta - 0.006823012) / 0.003151938   # +0.0%  mean_eta > 0.006823
        - 0.0002511441 * max(0.0, Q.sd_mass - 38.43971) * max(0.0, 153.375 - Q.pt_0) / 125.7046   # -0.0%  sd_mass > 38.44 and pt_0 < 153.4
        - 0.0002180488 * max(0.0, 0.002635418 - Q.sum_z_dr2) * max(0.0, Q.dr1_7 - 0.1343804) / 1.289265e-06   # -0.0%  sum_z_dr2 < 0.002635 and dr1_7 > 0.1344
        - 0.000191431 * max(0.0, 0.03981924 - Q.zdr_0) * max(0.0, Q.ptdr0_7 - 10.31097) / 0.001410312   # -0.0%  zdr_0 < 0.03982 and ptdr0_7 > 10.31
        - 0.0001687966 * max(0.0, 0.05119235 - Q.C2) * max(0.0, Q.mean_eta - 0.02644207) / 1.096062e-05   # -0.0%  C2 < 0.05119 and mean_eta > 0.02644
        + 0.0001671662 * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) * max(0.0, -0.01790907 - Q.mean_eta) / 2.392698e-06   # +0.0%  sum_z_dr2_top2 < 0.00764 and mean_eta < -0.01791
        - 0.0001671401 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # -0.0%  mean_eta > 0.02644
        - 0.0001427475 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, -0.01753483 - Q.mean_phi) / 2.64217e-07   # -0.0%  lam2 < 0.0005373 and mean_phi < -0.01753
        - 0.0001185605 * max(0.0, Q.sj3_dr_max - 0.3456459) * max(0.0, 118.1875 - Q.pt_0) / 0.05295229   # -0.0%  sj3_dr_max > 0.3456 and pt_0 < 118.2
        + 0.0001107193 * max(0.0, 0.2233283 - Q.N2) * max(0.0, -0.02594505 - Q.mean_phi) / 3.611955e-05   # +0.0%  N2 < 0.2233 and mean_phi < -0.02595
        + 0.0001106208 * max(0.0, 0.2233283 - Q.N2) * max(0.0, -0.009352575 - Q.mean_phi) / 0.0001205535   # +0.0%  N2 < 0.2233 and mean_phi < -0.009353
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 12.63;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.62726 * (0.05393676
        + 0.06744532 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +6.7%  tau1 < 0.09539
        + 0.06389725 * max(0.0, Q.log_sum_pt - 6.267538) / 0.2983143   # +6.4%  log_sum_pt > 6.268
        - 0.06024207 * max(0.0, 0.09041383 - Q.mass_over_sum_pt) / 0.03744004   # -6.0%  mass_over_sum_pt < 0.09041
        - 0.05589994 * max(0.0, Q.pt_7 - 23.21641) / 12.35308   # -5.6%  pt_7 > 23.22
        - 0.04743486 * max(0.0, 37.15625 - Q.pt_7) / 5.803748   # -4.7%  pt_7 < 37.16
        + 0.04581939 * max(0.0, 0.001653836 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 7.0272e-06   # +4.6%  sum_z_dr2 < 0.001654 and centroid_offset < 0.02355
        + 0.03703116 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # +3.7%  sum_z_dr2 < 0.002635
        - 0.03382013 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0004149607   # -3.4%  z_7 < 0.07149 and centroid_offset < 0.03117
        - 0.02783081 * max(0.0, 0.02383244 - Q.zdr_0) / 0.01109706   # -2.8%  zdr_0 < 0.02383
        + 0.02469814 * max(0.0, 0.1879486 - Q.sj3_dr_max) / 0.05910331   # +2.5%  sj3_dr_max < 0.1879
        - 0.02347828 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -2.3%  log_sum_pt > 6.701
        + 0.02101799 * max(0.0, 0.04939969 - Q.z_7) / 0.007457505   # +2.1%  z_7 < 0.0494
        - 0.02043883 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # -2.0%  sj3_dr_max < 0.107
        + 0.02041864 * max(0.0, Q.sum_pt_top5 - 716.8828) / 29.12383   # +2.0%  sum_pt_top5 > 716.9
        + 0.01976215 * max(0.0, Q.log_sum_pt - 6.267538) * max(0.0, 0.0113471 - Q.C3) / 0.0005276169   # +2.0%  log_sum_pt > 6.268 and C3 < 0.01135
        - 0.0188131 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.00424745 - Q.mean_eta2) / 0.0001193758   # -1.9%  log_sum_pt > 6.701 and mean_eta2 < 0.004247
        - 0.01791728 * max(0.0, Q.log_sum_pt - 6.766778) / 0.01900249   # -1.8%  log_sum_pt > 6.767
        - 0.01782719 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -1.8%  sj3_dr_max < 0.3012
        + 0.0164697 * max(0.0, Q.log_sum_pt - 6.267538) * max(0.0, 7.737156 - Q.ptdr0_5) / 1.615178   # +1.6%  log_sum_pt > 6.268 and ptdr0_5 < 7.737
        - 0.0161937 * max(0.0, 0.002915531 - Q.sum_z_dr2_top3) / 0.001178811   # -1.6%  sum_z_dr2_top3 < 0.002916
        - 0.01571746 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -1.6%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.01568808 * max(0.0, 0.04737278 - Q.z_6) / 0.004344387   # +1.6%  z_6 < 0.04737
        + 0.01518962 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 9.030369e-05 - Q.mean_eta2) / 9.99897e-07   # +1.5%  log_sum_pt > 6.701 and mean_eta2 < 9.03e-05
        + 0.01378154 * max(0.0, Q.log_sum_pt - 6.267538) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.1905581   # +1.4%  log_sum_pt > 6.268 and z_dr_0p05_0p1 < 0.846
        + 0.01358441 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 68.43422 - Q.mass_top5) / 0.3465446   # +1.4%  z_7 < 0.0494 and mass_top5 < 68.43
        + 0.01356259 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.06112084 - Q.dr_max_012) / 0.001464096   # +1.4%  log_sum_pt > 6.701 and dr_max_012 < 0.06112
        + 0.01251953 * max(0.0, 0.001653836 - Q.sum_z_dr2) / 0.0004289067   # +1.3%  sum_z_dr2 < 0.001654
        + 0.01185834 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +1.2%  z_7 < 0.02807
        + 0.01169022 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +1.2%  z_7 < 0.07149
        + 0.01071265 * max(0.0, 0.07870506 - Q.z_dr_0p1_0p2) / 0.04317253   # +1.1%  z_dr_0p1_0p2 < 0.07871
        - 0.01053144 * max(0.0, 0.009518026 - Q.C3) / 0.0007952626   # -1.1%  C3 < 0.009518
        - 0.01007584 * max(0.0, Q.sum_pt_top5 - 716.8828) * max(0.0, 9.030369e-05 - Q.mean_eta2) / 0.0008286902   # -1.0%  sum_pt_top5 > 716.9 and mean_eta2 < 9.03e-05
        + 0.00900948 * max(0.0, 0.0586137 - Q.z_7) / 0.01231218   # +0.9%  z_7 < 0.05861
        - 0.008927471 * max(0.0, 0.002915531 - Q.sum_z_dr2_top3) * max(0.0, 0.02186584 - Q.phi_1) / 2.667753e-05   # -0.9%  sum_z_dr2_top3 < 0.002916 and phi_1 < 0.02187
        - 0.008807497 * max(0.0, 0.02383244 - Q.zdr_0) * max(0.0, 0.02090454 - Q.abseta_0) / 0.0001122283   # -0.9%  zdr_0 < 0.02383 and abseta_0 < 0.0209
        + 0.008732291 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.001736329 - Q.zdr_3) / 2.305525e-05   # +0.9%  log_sum_pt > 6.701 and zdr_3 < 0.001736
        + 0.008694471 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0002302115 - Q.mean_phi2) / 3.738179e-06   # +0.9%  log_sum_pt > 6.701 and mean_phi2 < 0.0002302
        + 0.008282303 * max(0.0, 60.63098 - Q.mass) / 24.47895   # +0.8%  mass < 60.63
        - 0.00792563 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 4.721224 - Q.D2_b2) / 0.09403434   # -0.8%  log_sum_pt > 6.701 and D2_b2 < 4.721
        - 0.007636643 * max(0.0, Q.log_sum_pt - 6.267538) * max(0.0, 0.03601074 - Q.abseta_4) / 0.004915608   # -0.8%  log_sum_pt > 6.268 and abseta_4 < 0.03601
        + 0.007453318 * max(0.0, 0.09538712 - Q.tau1) * max(0.0, 0.4926918 - Q.planar_flow) / 0.008723222   # +0.7%  tau1 < 0.09539 and planar_flow < 0.4927
        + 0.007376373 * max(0.0, 0.005284669 - Q.sum_zz_dr2) * max(0.0, 0.01437952 - Q.centroid_offset) / 1.318015e-05   # +0.7%  sum_zz_dr2 < 0.005285 and centroid_offset < 0.01438
        - 0.007349742 * max(0.0, 0.002915531 - Q.sum_z_dr2_top3) * max(0.0, 0.006646257 - Q.tau3) / 3.390069e-06   # -0.7%  sum_z_dr2_top3 < 0.002916 and tau3 < 0.006646
        + 0.00709331 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 42.18339 - Q.mass_top3) / 0.6375506   # +0.7%  z_7 < 0.07149 and mass_top3 < 42.18
        + 0.006292824 * max(0.0, 60.63098 - Q.mass) * max(0.0, 3.885568 - Q.D2) / 50.41033   # +0.6%  mass < 60.63 and D2 < 3.886
        - 0.006094348 * max(0.0, 37.15625 - Q.pt_7) * max(0.0, 0.02079512 - Q.C3) / 0.02781713   # -0.6%  pt_7 < 37.16 and C3 < 0.0208
        - 0.005979057 * max(0.0, Q.pt_7 - 23.21641) * max(0.0, 0.07141113 - Q.absphi_1) / 0.5007982   # -0.6%  pt_7 > 23.22 and absphi_1 < 0.07141
        + 0.005957662 * max(0.0, 37.15625 - Q.pt_7) * max(0.0, 0.01257746 - Q.zdr_3) / 0.04815576   # +0.6%  pt_7 < 37.16 and zdr_3 < 0.01258
        + 0.005320019 * max(0.0, 0.005284669 - Q.sum_zz_dr2) / 0.00223735   # +0.5%  sum_zz_dr2 < 0.005285
        - 0.005111763 * max(0.0, 0.001653836 - Q.sum_z_dr2) * max(0.0, 0.01452637 - Q.absphi_0) / 3.722671e-06   # -0.5%  sum_z_dr2 < 0.001654 and absphi_0 < 0.01453
        - 0.004708502 * max(0.0, Q.max_pair_mass - 21.83614) * max(0.0, 0.2366102 - Q.dr_max_012) / 0.07844215   # -0.5%  max_pair_mass > 21.84 and dr_max_012 < 0.2366
        - 0.003958911 * max(0.0, 0.04737278 - Q.z_6) * max(0.0, 0.001736329 - Q.zdr_3) / 2.629876e-06   # -0.4%  z_6 < 0.04737 and zdr_3 < 0.001736
        - 0.003826117 * max(0.0, Q.max_pair_mass - 21.83614) / 2.887317   # -0.4%  max_pair_mass > 21.84
        - 0.003630249 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.002727284 - Q.zdr_3) / 6.209553e-06   # -0.4%  log_sum_pt > 6.896 and zdr_3 < 0.002727
        - 0.003285415 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.mean_phi - 0.0006973656) / 6.386425e-05   # -0.3%  log_sum_pt > 6.701 and mean_phi > 0.0006974
        + 0.003156989 * max(0.0, 37.15625 - Q.pt_7) * max(0.0, 0.1847499 - Q.mratio_min_012) / 0.2915428   # +0.3%  pt_7 < 37.16 and mratio_min_012 < 0.1847
        + 0.003062994 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01402779 - Q.abseta_4) / 0.0001821359   # +0.3%  log_sum_pt > 6.701 and abseta_4 < 0.01403
        - 0.002692453 * max(0.0, Q.log_sum_pt - 6.766778) * max(0.0, 0.05481757 - Q.C3) / 0.0006853593   # -0.3%  log_sum_pt > 6.767 and C3 < 0.05482
        + 0.002627906 * max(0.0, 0.02383244 - Q.zdr_0) * max(0.0, Q.z_5 - 0.02818362) / 0.0004584623   # +0.3%  zdr_0 < 0.02383 and z_5 > 0.02818
        + 0.002623358 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.3%  log_sum_pt > 6.896
        + 0.002258035 * max(0.0, 9.121392e-05 - Q.sum_z_dr2) / 4.152703e-06   # +0.2%  sum_z_dr2 < 9.121e-05
        + 0.00216061 * max(0.0, 0.09041383 - Q.mass_over_sum_pt) * max(0.0, Q.dr0_3 - 0.06671811) / 0.0002756001   # +0.2%  mass_over_sum_pt < 0.09041 and dr0_3 > 0.06672
        - 0.002070841 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.0002302115 - Q.mean_phi2) / 4.941235e-07   # -0.2%  log_sum_pt > 6.896 and mean_phi2 < 0.0002302
        + 0.001976641 * max(0.0, 0.02160287 - Q.z_6) * max(0.0, 0.1583252 - Q.abseta_6) / 3.551538e-05   # +0.2%  z_6 < 0.0216 and abseta_6 < 0.1583
        + 0.001705933 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, Q.mass_top2 - 22.84498) / 0.1158122   # +0.2%  sj3_dr_max < 0.3012 and mass_top2 > 22.84
        + 0.001697484 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.mean_phi2 - 0.0007823696) / 2.732408e-05   # +0.2%  log_sum_pt > 6.701 and mean_phi2 > 0.0007824
        + 0.001666091 * max(0.0, 0.02807091 - Q.z_7) * max(0.0, Q.pt_5 - 24.57812) / 0.009960011   # +0.2%  z_7 < 0.02807 and pt_5 > 24.58
        - 0.001488322 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, Q.sj3_pair_mass_max - 17.83163) / 0.05562116   # -0.1%  log_sum_pt > 6.896 and sj3_pair_mass_max > 17.83
        + 0.001459684 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # +0.1%  z_6 < 0.0216
        - 0.001357065 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, Q.C2_b2 - 0.001109927) / 4.204538e-05   # -0.1%  z_7 < 0.07149 and C2_b2 > 0.00111
        + 0.001310723 * max(0.0, 0.02818362 - Q.z_5) * max(0.0, 0.1626038 - Q.abseta_7) / 4.227915e-05   # +0.1%  z_5 < 0.02818 and abseta_7 < 0.1626
        - 0.00124361 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.03086731 - Q.dr_max_012) / 7.260647e-05   # -0.1%  log_sum_pt > 6.896 and dr_max_012 < 0.03087
        + 0.001213683 * max(0.0, 0.02886576 - Q.z_6) / 0.0008381782   # +0.1%  z_6 < 0.02887
        - 0.001054751 * max(0.0, 0.09041383 - Q.mass_over_sum_pt) * max(0.0, Q.ptdr0_3 - 12.34207) / 0.002779189   # -0.1%  mass_over_sum_pt < 0.09041 and ptdr0_3 > 12.34
        - 0.000284952 * max(0.0, 0.02886576 - Q.z_6) * max(0.0, 8.0 - Q.n_pt_above_5) / 0.0001362087   # -0.0%  z_6 < 0.02887 and n_pt_above_5 < 8
        + 6.683812e-05 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, Q.mean_phi - 0.01746433) / 3.778817e-07   # +0.0%  log_sum_pt > 6.896 and mean_phi > 0.01746
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 31.25;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.24787 * (0.01349341
        - 0.1504319 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -15.0%  sum_z_dr2 < 0.008678
        - 0.09893752 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # -9.9%  lam1_plus_lam2 < 0.01324
        + 0.08684411 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +8.7%  lam1 < 0.012
        + 0.05404321 * max(0.0, 0.1789613 - Q.sj3_dr_max) / 0.05394779   # +5.4%  sj3_dr_max < 0.179
        + 0.04527788 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +4.5%  tau1 < 0.1136
        - 0.04480509 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # -4.5%  max_dr < 0.1452
        + 0.04042114 * max(0.0, 0.09041383 - Q.mass_over_sum_pt) / 0.03744004   # +4.0%  mass_over_sum_pt < 0.09041
        - 0.03935598 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -3.9%  sj3_dr_max < 0.3012
        - 0.03747987 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 1.341509   # -3.7%  mass < 60.63 and z_dr_0p2_0p4 < 0.05644
        - 0.03444533 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -3.4%  lam2 < 0.001131
        + 0.0335461 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +3.4%  lam2 < 0.003408
        + 0.03286117 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 988.4078 - Q.sum_pt) / 1177.276   # +3.3%  pt_6 < 41.22 and sum_pt < 988.4
        + 0.03121884 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +3.1%  e3 < 0.0005117
        + 0.02363157 * max(0.0, 0.02415398 - Q.C2_b2) / 0.02132459   # +2.4%  C2_b2 < 0.02415
        + 0.0196499 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +2.0%  sj3_dr_max > 0.1879
        - 0.01826731 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -1.8%  pt_6 < 39.75
        + 0.01513854 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +1.5%  centroid_offset > 0.008092
        + 0.01449199 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +1.4%  lam1 < 0.00733
        - 0.01416966 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757) / 0.07098489   # -1.4%  pt_6 < 41.22 and z_7 > 0.02321
        - 0.01410528 * max(0.0, Q.sj3_dr_min - 0.02400746) / 0.02802924   # -1.4%  sj3_dr_min > 0.02401
        + 0.01392068 * max(0.0, 60.63098 - Q.mass) / 24.47895   # +1.4%  mass < 60.63
        - 0.01077947 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 0.2534037 - Q.planar_flow) / 0.0006566196   # -1.1%  lam1 < 0.012 and planar_flow < 0.2534
        - 0.01058016 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # -1.1%  pt_6 < 41.22
        + 0.009788143 * max(0.0, 45.7571 - Q.mass) / 14.82409   # +1.0%  mass < 45.76
        - 0.009549357 * max(0.0, Q.centroid_offset - 0.01837778) / 0.005523694   # -1.0%  centroid_offset > 0.01838
        + 0.009418243 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.003289169   # +0.9%  lam1 < 0.00733 and n_dr_0p2_0p4 < 1
        + 0.008787004 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # +0.9%  sj2_dr < 0.1873
        + 0.007104329 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002658741   # +0.7%  lam2 < 0.003408 and n_dr_0_0p05 < 3
        + 0.007066305 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # +0.7%  sum_zz_dr2 < 0.003013
        + 0.005971722 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2) / 2.179362e-05   # +0.6%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        - 0.004829293 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -0.5%  lam1 < 0.005954
        + 0.004427368 * max(0.0, 0.03932388 - Q.z_6) / 0.002360011   # +0.4%  z_6 < 0.03932
        + 0.004233838 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.psi_0p1 - 0.4008925) / 0.003485257   # +0.4%  centroid_offset > 0.008092 and psi_0p1 > 0.4009
        + 0.003812841 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.0753896 - Q.z_7) / 0.963884   # +0.4%  sum_pt < 763.8 and z_7 < 0.07539
        - 0.003735788 * max(0.0, 763.825 - Q.sum_pt) / 96.68366   # -0.4%  sum_pt < 763.8
        - 0.003155498 * max(0.0, Q.sj3_pair_mass_min - 4.501727) / 3.480268   # -0.3%  sj3_pair_mass_min > 4.502
        + 0.003075264 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +0.3%  sj3_dr_min > 0.1278
        + 0.003036571 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 43.23531   # +0.3%  sum_pt < 615.9 and n_dr_0p2_0p4 < 2
        - 0.002905774 * max(0.0, Q.eccentricity - 0.927072) / 0.03232992   # -0.3%  eccentricity > 0.9271
        + 0.001817025 * max(0.0, 488.9312 - Q.sum_pt) / 6.000379   # +0.2%  sum_pt < 488.9
        + 0.001765892 * max(0.0, 615.875 - Q.sum_pt) / 30.07766   # +0.2%  sum_pt < 615.9
        + 0.001707344 * max(0.0, 19.46875 - Q.pt_6) / 0.2515892   # +0.2%  pt_6 < 19.47
        + 0.001656391 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 0.1343804 - Q.dr1_7) / 0.3834372   # +0.2%  pt_6 < 41.22 and dr1_7 < 0.1344
        - 0.001586328 * max(0.0, 29.90625 - Q.pt_6) / 1.385454   # -0.2%  pt_6 < 29.91
        - 0.001489619 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # -0.1%  mean_eta > 0.02644
        + 0.00147747 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt) / 1.016213   # +0.1%  pt_6 < 41.22 and log_sum_pt < 6.767
        + 0.001363319 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308) / 0.05438016   # +0.1%  centroid_offset > 0.008092 and sj3_pair_mass_min > 6.811
        - 0.001303012 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 2.69383   # -0.1%  sj3_pair_mass_min > 4.502 and n_dr_0p2_0p4 > 1
        - 0.001293805 * max(0.0, Q.sj3_dr13 - 0.181053) / 0.02525513   # -0.1%  sj3_dr13 > 0.1811
        - 0.001225579 * max(0.0, Q.sj3_dr23 - 0.2207152) / 0.01962555   # -0.1%  sj3_dr23 > 0.2207
        + 0.001065213 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, Q.sum_pt_top5 - 658.125) / 0.02654014   # +0.1%  centroid_offset > 0.01838 and sum_pt_top5 > 658.1
        + 0.001011112 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.M3 - 0.0782171) / 0.01402652   # +0.1%  pt_6 < 41.22 and M3 > 0.07822
        - 0.0009959012 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.1818564 - Q.z_2) / 0.0002723086   # -0.1%  centroid_offset > 0.01838 and z_2 < 0.1819
        + 0.0007789079 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.02702951 - Q.C2) / 2.39871e-05   # +0.1%  centroid_offset > 0.01838 and C2 < 0.02703
        - 0.0007509384 * max(0.0, Q.sj3_dr_max - 0.1879486) * max(0.0, 0.03250122 - Q.abseta_2) / 0.0002335307   # -0.1%  sj3_dr_max > 0.1879 and abseta_2 < 0.0325
        - 0.0007018236 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.00433128 - Q.mean_phi2) / 7.021694e-06   # -0.1%  centroid_offset > 0.01838 and mean_phi2 < 0.004331
        - 0.0006790971 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.02980347 - Q.eta_0) / 0.0002676158   # -0.1%  centroid_offset > 0.01838 and eta_0 < 0.0298
        - 0.0005831784 * max(0.0, 0.1452311 - Q.max_dr) * max(0.0, Q.mean_phi - 0.004406178) / 9.704264e-05   # -0.1%  max_dr < 0.1452 and mean_phi > 0.004406
        - 0.0004236941 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, Q.M3 - 0.06688759) / 9.396674e-06   # -0.0%  mean_eta > 0.02644 and M3 > 0.06689
        + 0.0003625877 * max(0.0, 29.90625 - Q.pt_6) * max(0.0, 8.0 - Q.n_pt_above_10) / 0.5895401   # +0.0%  pt_6 < 29.91 and n_pt_above_10 < 8
        + 0.0003367755 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.mean_eta - 0.02644207) / 1.54475e-06   # +0.0%  lam1 < 0.012 and mean_eta > 0.02644
        - 0.0002228116 * max(0.0, Q.sj3_dr13 - 0.181053) * max(0.0, Q.ptdr0_7 - 10.31097) / 0.01556911   # -0.0%  sj3_dr13 > 0.1811 and ptdr0_7 > 10.31
        + 6.314668e-05 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, Q.mass_top5 - 68.43422) / 1.626202e-05   # +0.0%  lam1 < 0.00733 and mass_top5 > 68.43
        + 3.901379e-05 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 0.05101919 - Q.z_3) / 0.0001257894   # +0.0%  sum_pt < 615.9 and z_3 < 0.05102
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 40.39;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 40.39026 * (0.2410141
        - 0.06625511 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -6.6%  sum_z_dr < 0.08724
        - 0.06575514 * max(0.0, Q.mass_over_sum_pt - 0.01109984) / 0.05101851   # -6.6%  mass_over_sum_pt > 0.0111
        - 0.06240398 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) / 0.00293624   # -6.2%  lam1_plus_lam2 < 0.006679
        - 0.05700723 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -5.7%  mass_over_sum_pt > 0.09041
        - 0.05344099 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # -5.3%  lam1_plus_lam2 > 0.00559
        + 0.04577418 * max(0.0, Q.sum_z_dr2 - 0.01323868) / 0.001442684   # +4.6%  sum_z_dr2 > 0.01324
        - 0.03932246 * max(0.0, Q.sum_z_dr2 - 0.004372139) / 0.003638805   # -3.9%  sum_z_dr2 > 0.004372
        + 0.03550702 * max(0.0, Q.sj3_dr_max - 0.1426152) / 0.06279029   # +3.6%  sj3_dr_max > 0.1426
        + 0.03541133 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +3.5%  mass_over_sum_pt > 0.08475
        - 0.03336141 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -3.3%  sj2_dr < 0.1873
        + 0.03178904 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +3.2%  sj2_dr < 0.1592
        + 0.03119566 * max(0.0, Q.mass_over_sum_pt - 0.07269073) / 0.01344138   # +3.1%  mass_over_sum_pt > 0.07269
        - 0.03098139 * max(0.0, Q.mass_over_sum_pt - 0.1079857) / 0.005364315   # -3.1%  mass_over_sum_pt > 0.108
        - 0.02727496 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -2.7%  sum_z_dr2 > 0.00752
        - 0.02276682 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -2.3%  lam1 < 0.008376
        - 0.02009022 * max(0.0, Q.lam1_plus_lam2 - 0.008678045) * max(0.0, 38.25 - Q.pt_6) / 0.006941575   # -2.0%  lam1_plus_lam2 > 0.008678 and pt_6 < 38.25
        - 0.01960258 * max(0.0, Q.sj3_dr_max - 0.04889979) / 0.1256943   # -2.0%  sj3_dr_max > 0.0489
        - 0.01873838 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # -1.9%  e2 > 0.05028
        + 0.01738798 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +1.7%  mass_over_sum_pt > 0.07992
        + 0.01717455 * max(0.0, Q.lam1_plus_lam2 - 0.008678045) / 0.002214537   # +1.7%  lam1_plus_lam2 > 0.008678
        + 0.01643859 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.6%  tau1 < 0.05357
        + 0.01603991 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +1.6%  LHA < 0.3033
        + 0.01527502 * max(0.0, Q.mass - 36.22941) / 14.20528   # +1.5%  mass > 36.23
        + 0.01419847 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # +1.4%  max_dr < 0.1773
        - 0.01174205 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.004032342 - Q.C2_b2) / 2.736263e-05   # -1.2%  centroid_offset < 0.02077 and C2_b2 < 0.004032
        - 0.01044161 * max(0.0, Q.sum_z_dr2 - 0.001653836) / 0.005220304   # -1.0%  sum_z_dr2 > 0.001654
        - 0.01019771 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # -1.0%  max_dr < 0.1118
        - 0.01002536 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -1.0%  sj3_dr_max > 0.2623
        + 0.009872593 * max(0.0, 0.08723651 - Q.sum_z_dr) * max(0.0, 0.004032342 - Q.C2_b2) / 0.0001217265   # +1.0%  sum_z_dr < 0.08724 and C2_b2 < 0.004032
        + 0.009839649 * max(0.0, 0.0003061234 - Q.lam2) * max(0.0, 0.04019753 - Q.tau21_b2) / 2.611351e-06   # +1.0%  lam2 < 0.0003061 and tau21_b2 < 0.0402
        - 0.008658065 * max(0.0, 0.04081947 - Q.sum_z_dr) / 0.009176315   # -0.9%  sum_z_dr < 0.04082
        - 0.0084753 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # -0.8%  lam2 < 0.0003061
        - 0.008393604 * max(0.0, Q.sd_rg - 0.2037854) / 0.01566501   # -0.8%  sd_rg > 0.2038
        + 0.008356736 * max(0.0, Q.mass_over_sum_pt - 0.09041383) * max(0.0, 36.8125 - Q.pt_6) / 0.01975833   # +0.8%  mass_over_sum_pt > 0.09041 and pt_6 < 36.81
        + 0.007833639 * max(0.0, Q.sum_z_dr2 - 0.01323868) * max(0.0, 38.25 - Q.pt_6) / 0.004519922   # +0.8%  sum_z_dr2 > 0.01324 and pt_6 < 38.25
        - 0.007660035 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.8%  mass > 76.66
        - 0.006965254 * max(0.0, 0.04019753 - Q.tau21_b2) / 0.01109641   # -0.7%  tau21_b2 < 0.0402
        + 0.006595305 * max(0.0, Q.z_7 - 0.03243272) / 0.02180051   # +0.7%  z_7 > 0.03243
        - 0.005980131 * max(0.0, 0.0009641429 - Q.sum_z_dr2) / 0.0002052288   # -0.6%  sum_z_dr2 < 0.0009641
        + 0.005391437 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # +0.5%  sd_rg > 0.2788
        - 0.00435915 * max(0.0, 0.1950135 - Q.planar_flow) / 0.07573236   # -0.4%  planar_flow < 0.195
        + 0.004316496 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) / 0.0003002514   # +0.4%  sum_z_dr2_top2 < 0.001057
        + 0.004276423 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # +0.4%  sum_z_dr2 < 0.003563
        + 0.004119277 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sd_mass - 38.43971) / 1.25962   # +0.4%  planar_flow < 0.195 and sd_mass > 38.44
        - 0.004033462 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -0.4%  centroid_offset < 0.02077
        + 0.003814095 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.lam1_plus_lam2 - 0.007520088) / 0.0001455029   # +0.4%  planar_flow < 0.195 and lam1_plus_lam2 > 0.00752
        + 0.003685727 * max(0.0, Q.mass_top5 - 53.60766) / 1.978814   # +0.4%  mass_top5 > 53.61
        + 0.00363795 * max(0.0, Q.sum_z_dr2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +0.4%  sum_z_dr2 > 0.004372 and planar_flow < 0.195
        + 0.003409074 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) / 0.09532154   # +0.3%  z_dr_0p1_0p2 < 0.1586
        - 0.003238054 * max(0.0, 0.05744392 - Q.D2_b2) / 0.005938983   # -0.3%  D2_b2 < 0.05744
        + 0.003156695 * max(0.0, Q.sj3_dr_max - 0.213399) / 0.03061279   # +0.3%  sj3_dr_max > 0.2134
        + 0.002951704 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.02656143 - Q.tau21_b2) / 4.805781e-05   # +0.3%  centroid_offset < 0.02077 and tau21_b2 < 0.02656
        - 0.002739328 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.3%  centroid_offset > 0.03776
        + 0.002547195 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # +0.3%  sj2_dr < 0.1295
        + 0.002357107 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, Q.sum_pt_top3 - 331.25) / 1.710819   # +0.2%  centroid_offset < 0.02077 and sum_pt_top3 > 331.2
        + 0.002211236 * max(0.0, Q.sum_z_dr2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9458207) / 2.125219e-05   # +0.2%  sum_z_dr2 > 0.01324 and eccentricity > 0.9458
        - 0.002094593 * max(0.0, Q.mass - 69.61135) / 2.406702   # -0.2%  mass > 69.61
        - 0.002067603 * max(0.0, Q.mass_over_sum_pt - 0.01109984) * max(0.0, 35.28125 - Q.pt_6) / 0.1065511   # -0.2%  mass_over_sum_pt > 0.0111 and pt_6 < 35.28
        - 0.001819903 * max(0.0, 29.04219 - Q.pt_7) / 2.179984   # -0.2%  pt_7 < 29.04
        - 0.001764234 * max(0.0, Q.sum_z_dr2 - 0.01323868) * max(0.0, 0.06970457 - Q.dr_6) / 2.606496e-06   # -0.2%  sum_z_dr2 > 0.01324 and dr_6 < 0.0697
        + 0.001597514 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) * max(0.0, 0.006452173 - Q.mean_phi) / 2.379971e-05   # +0.2%  lam1_plus_lam2 < 0.006679 and mean_phi < 0.006452
        - 0.001533097 * max(0.0, Q.mass - 76.6557) * max(0.0, Q.zdr_6 - 0.006366792) / 0.006641425   # -0.2%  mass > 76.66 and zdr_6 > 0.006367
        + 0.00109071 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # +0.1%  e2 > 0.03556
        - 0.0008488967 * max(0.0, Q.centroid_offset - 0.03117077) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.00188116   # -0.1%  centroid_offset > 0.03117 and n_pt_above_50 > 4
        + 0.0007385851 * max(0.0, Q.sd_rg - 0.2037854) * max(0.0, 0.06970457 - Q.dr_6) / 4.971602e-05   # +0.1%  sd_rg > 0.2038 and dr_6 < 0.0697
        - 0.0005723198 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) * max(0.0, 0.02656143 - Q.tau21_b2) / 2.239525e-07   # -0.1%  sum_z_dr2_top2 < 0.001057 and tau21_b2 < 0.02656
        + 0.0005011016 * max(0.0, 0.009213932 - Q.tau21_b2) / 0.0008512768   # +0.1%  tau21_b2 < 0.009214
        - 0.0004599341 * max(0.0, 29.04219 - Q.pt_7) * max(0.0, 0.2838437 - Q.tau21) / 0.1125981   # -0.0%  pt_7 < 29.04 and tau21 < 0.2838
        - 0.0002987739 * max(0.0, Q.sj3_dr_max - 0.1426152) * max(0.0, 19.46875 - Q.pt_6) / 0.01292439   # -0.0%  sj3_dr_max > 0.1426 and pt_6 < 19.47
        - 0.0001388274 * max(0.0, Q.centroid_offset - 0.03776099) * max(0.0, Q.pt_4 - 81.375) / 0.0001936307   # -0.0%  centroid_offset > 0.03776 and pt_4 > 81.38
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 19.75;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.75408 * (-0.01814984
        - 0.0803283 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.003562611 - Q.sum_z_dr2) / 5.329563e-05   # -8.0%  sum_z_dr < 0.06109 and sum_z_dr2 < 0.003563
        + 0.07227843 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.001184249   # +7.2%  lam1_plus_lam2 < 0.003563
        - 0.06901774 * max(0.0, 0.06108601 - Q.sum_z_dr) / 0.01868685   # -6.9%  sum_z_dr < 0.06109
        + 0.06227112 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0001901112   # +6.2%  sum_z_dr2 < 0.00502 and z_dr_0p2_0p4 < 0.101
        + 0.0602838 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 4.808955e-05   # +6.0%  tau1 < 0.05357 and lam1_plus_lam2 < 0.003563
        + 0.05815754 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.0003703114   # +5.8%  sj3_dr_max < 0.1986 and sum_z_dr2 < 0.006679
        + 0.03948142 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +3.9%  sum_z_dr2 < 0.00502
        - 0.03921999 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # -3.9%  LHA < 0.1967
        - 0.03622338 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.1231549   # -3.6%  mass < 29.64 and centroid_offset < 0.02686
        + 0.03472821 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +3.5%  sum_z_dr2 < 0.006679 and centroid_offset < 0.02355
        + 0.03373416 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.0001272072   # +3.4%  mass_over_sum_pt < 0.03319 and centroid_offset < 0.02686
        - 0.03066693 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.0001781324   # -3.1%  sj3_dr_max < 0.1986 and lam1_plus_lam2 < 0.003563
        - 0.027836 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -2.8%  sj3_dr_max < 0.1426
        + 0.02331203 * max(0.0, 0.002635418 - Q.lam1_plus_lam2) / 0.0007941347   # +2.3%  lam1_plus_lam2 < 0.002635
        - 0.02129488 * max(0.0, 49.6681 - Q.mass) / 17.06893   # -2.1%  mass < 49.67
        - 0.01814784 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.0001947983 - Q.lam2) / 0.0007178522   # -1.8%  mass < 21.78 and lam2 < 0.0001948
        - 0.01794679 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.005691733 - Q.sum_z_dr2_top5) / 0.0003116052   # -1.8%  sj3_dr_max < 0.1986 and sum_z_dr2_top5 < 0.005692
        - 0.01699281 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # -1.7%  sj3_dr_max < 0.1986
        - 0.01481139 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.001011075   # -1.5%  sum_z_dr < 0.06109 and z_dr_0p2_0p4 < 0.05644
        + 0.01409011 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.0001947983 - Q.lam2) / 2.869243e-06   # +1.4%  sum_z_dr < 0.06109 and lam2 < 0.0001948
        - 0.01363282 * max(0.0, 0.01289969 - Q.e2) * max(0.0, Q.psi_0p2 - 0.79448) / 0.0005184882   # -1.4%  e2 < 0.0129 and psi_0p2 > 0.7945
        + 0.01314497 * max(0.0, 0.0001330621 - Q.lam2) * max(0.0, 4.721224 - Q.D2_b2) / 0.0002340602   # +1.3%  lam2 < 0.0001331 and D2_b2 < 4.721
        - 0.0128883 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.01627885) / 0.0001130126   # -1.3%  sj3_dr_max < 0.1426 and centroid_offset > 0.01628
        + 0.01217751 * max(0.0, 29.6447 - Q.mass) * max(0.0, 1.762929e-06 - Q.e3) / 9.656157e-06   # +1.2%  mass < 29.64 and e3 < 1.763e-06
        - 0.01207258 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # -1.2%  tau1 < 0.05357
        - 0.01199729 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -1.2%  sum_z_dr2 < 0.00502 and centroid_offset > 0.00679
        - 0.01186065 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 43.5 - Q.pt_7) / 0.02255472   # -1.2%  sum_z_dr2 < 0.00502 and pt_7 < 43.5
        - 0.01163791 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, Q.centroid_offset - 0.006789738) / 8.239263e-05   # -1.2%  sum_z_dr < 0.06109 and centroid_offset > 0.00679
        - 0.01013414 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 2.955458e-05 - Q.e3) / 8.40609e-07   # -1.0%  log_sum_pt > 6.701 and e3 < 2.955e-05
        + 0.01006546 * max(0.0, 0.01289969 - Q.e2) / 0.002527207   # +1.0%  e2 < 0.0129
        - 0.009492846 * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.00293624   # -0.9%  sum_z_dr2 < 0.006679
        - 0.007778204 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) / 0.00722828   # -0.8%  mass_over_sum_pt < 0.03319
        + 0.007363839 * max(0.0, 21.78408 - Q.mass) * max(0.0, 48.71875 - Q.pt_7) / 69.12746   # +0.7%  mass < 21.78 and pt_7 < 48.72
        + 0.007294274 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) * max(0.0, Q.centroid_offset - 0.006789738) / 5.559665e-06   # +0.7%  lam1_plus_lam2 < 0.003563 and centroid_offset > 0.00679
        + 0.007183911 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.pt_7 - 15.55391) / 0.4153091   # +0.7%  LHA < 0.1967 and pt_7 > 15.55
        - 0.007023598 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 4.721224 - Q.D2_b2) / 0.04403777   # -0.7%  tau1 < 0.05357 and D2_b2 < 4.721
        + 0.006914753 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, Q.phi_0 - -0.04013062) / 0.0001189753   # +0.7%  sum_z_dr2 < 0.006679 and phi_0 > -0.04013
        + 0.006522621 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 48.71875 - Q.pt_7) / 0.776931   # +0.7%  log_sum_pt > 6.701 and pt_7 < 48.72
        + 0.00570487 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.0001947983 - Q.lam2) / 9.822054e-06   # +0.6%  sj3_dr_max < 0.1986 and lam2 < 0.0001948
        + 0.005601788 * max(0.0, 49.6681 - Q.mass) * max(0.0, 0.0001330621 - Q.lam2) / 0.001509268   # +0.6%  mass < 49.67 and lam2 < 0.0001331
        - 0.005112251 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # -0.5%  pt_7 > 34.53
        + 0.0045037 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0005611231 - Q.sum_z_dr2) / 8.626944e-06   # +0.5%  log_sum_pt > 6.701 and sum_z_dr2 < 0.0005611
        - 0.003959978 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.02355416 - Q.centroid_offset) / 0.0005980206   # -0.4%  log_sum_pt > 6.701 and centroid_offset < 0.02355
        + 0.003274621 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, Q.lam1_plus_lam2 - 0.0003193707) / 1.048473e-05   # +0.3%  sum_z_dr < 0.06109 and lam1_plus_lam2 > 0.0003194
        + 0.002873795 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +0.3%  sj3_dr_max < 0.107
        - 0.002451129 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 8.0 - Q.n_pt_above_10) / 7.731645   # -0.2%  sum_pt_top5 > 687.4 and n_pt_above_10 < 8
        + 0.002254167 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 142.5 - Q.pt_1) / 122.3165   # +0.2%  pt_7 > 34.53 and pt_1 < 142.5
        - 0.001920298 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.2097233   # -0.2%  mass < 29.64 and D2_b2 < 0.7166
        + 0.001790289 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.centroid_offset - 0.01837778) / 3.05705e-06   # +0.2%  LHA < 0.1967 and centroid_offset > 0.01838
        + 0.001611861 * max(0.0, 0.008355823 - Q.dr_0) / 0.0005818924   # +0.2%  dr_0 < 0.008356
        - 0.00157752 * max(0.0, 0.007078158 - Q.e2) * max(0.0, Q.sum_z_dr2_top5 - 5.672047e-05) / 1.487483e-07   # -0.2%  e2 < 0.007078 and sum_z_dr2_top5 > 5.672e-05
        + 0.001524915 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.mean_phi - -0.0008556753) / 5.44837e-05   # +0.2%  LHA < 0.1967 and mean_phi > -0.0008557
        - 0.00104028 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.1%  sum_pt > 988.4
        - 0.0009732179 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.zdr_0 - 0.007798268) / 0.0001443534   # -0.1%  log_sum_pt > 6.701 and zdr_0 > 0.007798
        - 0.0008962299 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 0.01544159   # -0.1%  log_sum_pt > 6.701 and n_dr_0p05_0p1 > 2
        + 0.0007190949 * max(0.0, 0.07658656 - Q.LHA) / 0.0004806283   # +0.1%  LHA < 0.07659
        + 0.000680499 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.05062541 - Q.dr12) / 0.1693771   # +0.1%  sum_pt > 988.4 and dr12 < 0.05063
        + 0.0006676214 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.07002129   # +0.1%  mass < 21.78 and D2_b2 < 0.7166
        - 0.0005750694 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.0002365852   # -0.1%  log_sum_pt > 6.701 and lam1_plus_lam2 < 0.008678
        + 0.0002782615 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, 1.0 - Q.psi_0p3) / 5.234195e-06   # +0.0%  LHA < 0.1967 and psi_0p3 < 1
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 32.6;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 32.59986 * (-0.08880021
        + 0.09585175 * max(0.0, 0.00609665 - Q.sum_z_dr2) / 0.002544039   # +9.6%  sum_z_dr2 < 0.006097
        + 0.07107972 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # +7.1%  sum_z_dr2 < 0.003563
        - 0.05996089 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -6.0%  mass_over_sum_pt < 0.07269
        - 0.05002829 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -5.0%  sj3_dr_max < 0.1426
        + 0.04840902 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +4.8%  sj3_dr_max < 0.2134
        - 0.04512901 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -4.5%  lam1 < 0.005954
        + 0.04503095 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +4.5%  centroid_offset < 0.01838
        + 0.04471996 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +4.5%  sum_pt < 988.4
        - 0.04429407 * max(0.0, 53.33237 - Q.mass) / 19.36004   # -4.4%  mass < 53.33
        + 0.04353305 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.2973626   # +4.4%  mass < 53.33 and centroid_offset < 0.02686
        + 0.03549378 * max(0.0, Q.lam1 - 0.005433361) / 0.00267714   # +3.5%  lam1 > 0.005433
        - 0.0267743 * max(0.0, Q.lam1 - 0.003377388) / 0.003672352   # -2.7%  lam1 > 0.003377
        + 0.02476485 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +2.5%  max_dr < 0.1118
        + 0.02475674 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +2.5%  mass_over_sum_pt_sq < 0.007183
        + 0.02394981 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # +2.4%  tau1 < 0.0437
        + 0.02372288 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +2.4%  sum_z_dr2 < 0.00502
        + 0.02116606 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt) / 5.175436   # +2.1%  mass < 53.33 and log_sum_pt < 6.843
        - 0.02052388 * max(0.0, 29.6447 - Q.mass) / 7.34723   # -2.1%  mass < 29.64
        - 0.01890493 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # -1.9%  sum_zz_dr2 < 0.003013
        - 0.01855611 * max(0.0, 2.371297e-05 - Q.e3) / 1.105328e-05   # -1.9%  e3 < 2.371e-05
        + 0.0124283 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +1.2%  e2 > 0.04447
        + 0.01197237 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # +1.2%  lam2 < 0.0003061
        - 0.01081112 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1) / 0.1931178   # -1.1%  centroid_offset < 0.01838 and pt_1 < 159.2
        - 0.01034769 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266) / 0.0005312449   # -1.0%  lam1 > 0.005433 and eccentricity > 0.7117
        - 0.01003345 * max(0.0, Q.sum_z_dr - 0.0717028) / 0.01201981   # -1.0%  sum_z_dr > 0.0717
        - 0.009474734 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.008712838   # -0.9%  centroid_offset < 0.01838 and n_for_90pct > 5
        - 0.009455047 * max(0.0, Q.mass - 45.7571) / 9.387753   # -0.9%  mass > 45.76
        - 0.00938422 * max(0.0, 0.05464922 - Q.sum_z_dr) / 0.01532978   # -0.9%  sum_z_dr < 0.05465
        - 0.009004551 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -0.9%  e2 > 0.03556
        - 0.008090582 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # -0.8%  e3 > 8.148e-05
        - 0.007290097 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) / 6.087676   # -0.7%  sj3_pair_mass_max < 24.23
        - 0.006660687 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, Q.z_6 - 0.03448406) / 0.0003071415   # -0.7%  sum_z_dr < 0.05465 and z_6 > 0.03448
        + 0.006427982 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd) / 0.0002178654   # +0.6%  centroid_offset < 0.01838 and z_2nd < 0.2056
        - 0.005710334 * max(0.0, 0.0284695 - Q.C3) / 0.008714909   # -0.6%  C3 < 0.02847
        + 0.005244966 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +0.5%  e3 > 0.0001869
        - 0.005178309 * max(0.0, 0.006292091 - Q.zdr_0) / 0.001006321   # -0.5%  zdr_0 < 0.006292
        + 0.00490455 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.5%  lam2 > 0.001131
        + 0.004727857 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.0002201438   # +0.5%  sum_z_dr2 < 0.006097 and planar_flow < 0.3221
        - 0.004232927 * max(0.0, 0.02227783 - Q.absphi_1) / 0.006894148   # -0.4%  absphi_1 < 0.02228
        + 0.004010821 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.6875) / 0.3914557   # +0.4%  sj3_dr_max < 0.1426 and pt_6 > 33.69
        - 0.003744378 * max(0.0, Q.D2 - 2.055451) / 0.3061148   # -0.4%  D2 > 2.055
        - 0.003472245 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, 0.002834884 - Q.mean_phi) / 1.344572e-05   # -0.3%  sum_z_dr2 < 0.006097 and mean_phi < 0.002835
        + 0.003253787 * max(0.0, 0.0001721983 - Q.lam1_plus_lam2) / 1.441896e-05   # +0.3%  lam1_plus_lam2 < 0.0001722
        + 0.003143712 * max(0.0, 0.06503035 - Q.z_5) / 0.007566857   # +0.3%  z_5 < 0.06503
        + 0.003034216 * max(0.0, 0.0782171 - Q.M3) / 0.01197321   # +0.3%  M3 < 0.07822
        + 0.002850988 * max(0.0, Q.N2 - 0.2233283) / 0.04872624   # +0.3%  N2 > 0.2233
        + 0.002715315 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.3%  n_dr_0p2_0p4 > 1
        - 0.002634316 * max(0.0, 7.0 - Q.n_for_90pct) / 0.5603294   # -0.3%  n_for_90pct < 7
        + 0.002328666 * max(0.0, Q.sum_z_dr2_top5 - 0.01148293) / 0.00154713   # +0.2%  sum_z_dr2_top5 > 0.01148
        - 0.002304671 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.tau4 - 0.001224244) / 1.099241e-05   # -0.2%  centroid_offset < 0.01838 and tau4 > 0.001224
        - 0.001845214 * max(0.0, 988.4078 - Q.sum_pt) * max(0.0, 0.01778111 - Q.dr_2) / 0.3401129   # -0.2%  sum_pt < 988.4 and dr_2 < 0.01778
        + 0.001714163 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.mean_eta - 6.288824e-05) / 1.181456e-05   # +0.2%  centroid_offset < 0.01838 and mean_eta > 6.289e-05
        - 0.001692394 * Q.z_dr_0p2_0p4 / 0.02920532   # -0.2%  z_dr_0p2_0p4
        - 0.001671273 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255) / 0.000277001   # -0.2%  tau1 < 0.0437 and eccentricity > 0.9031
        + 0.001603449 * max(0.0, 53.33237 - Q.mass) * max(0.0, Q.D2 - 2.843757) / 5.555775   # +0.2%  mass < 53.33 and D2 > 2.844
        + 0.00159369 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, 0.02656143 - Q.tau21_b2) / 1.23798e-05   # +0.2%  sum_z_dr < 0.05465 and tau21_b2 < 0.02656
        - 0.001568121 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.2%  sum_pt_top5 > 840
        + 0.001500571 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) * max(0.0, -0.004917145 - Q.phi_0) / 9.23841e-05   # +0.2%  mass_over_sum_pt < 0.07269 and phi_0 < -0.004917
        + 0.001273861 * max(0.0, 0.002170915 - Q.zdr_2) / 0.0002841564   # +0.1%  zdr_2 < 0.002171
        - 0.001206073 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.n_dr_0p05_0p1 - 7.0) / 0.5176771   # -0.1%  sd_mass > 44.82 and n_dr_0p05_0p1 > 7
        + 0.001176883 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.1%  log_sum_pt > 6.896
        - 0.001169556 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, 0.02656143 - Q.tau21_b2) / 8.029519e-06   # -0.1%  tau1 < 0.0437 and tau21_b2 < 0.02656
        - 0.001007632 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top2 - 1.933087) / 0.0177962   # -0.1%  mass_over_sum_pt < 0.07269 and mass_top2 > 1.933
        - 0.001002708 * max(0.0, 0.002316125 - Q.centroid_offset) / 9.872932e-05   # -0.1%  centroid_offset < 0.002316
        - 0.0009609376 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.01108383   # -0.1%  n_dr_0p2_0p4 > 1 and sj3_dr_min < 0.209
        - 0.0009077071 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.pt1_dr01 - 17.82896) / 0.009671885   # -0.1%  centroid_offset < 0.01838 and pt1_dr01 > 17.83
        + 0.0008680254 * max(0.0, 0.006292091 - Q.zdr_0) * max(0.0, Q.pt_2 - 88.4375) / 0.02706551   # +0.1%  zdr_0 < 0.006292 and pt_2 > 88.44
        + 0.0008177211 * max(0.0, Q.e2 - 0.04447357) * max(0.0, 0.2723288 - Q.sj3_mass3) / 0.0006651999   # +0.1%  e2 > 0.04447 and sj3_mass3 < 0.2723
        - 0.0006836098 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2) / 0.000627491   # -0.1%  log_sum_pt > 6.896 and D2_b2 < 0.7166
        + 0.000669744 * max(0.0, Q.e3 - 0.0001869378) * max(0.0, Q.pt_2 - 56.5) / 0.0005941291   # +0.1%  e3 > 0.0001869 and pt_2 > 56.5
        + 0.0006213696 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, -0.01275329 - Q.mean_phi) / 1.158446e-05   # +0.1%  tau1 < 0.0437 and mean_phi < -0.01275
        + 0.0004979724 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.mean_phi - 0.02612796) / 2.994783e-06   # +0.0%  tau1 < 0.0437 and mean_phi > 0.02613
        - 0.0004791216 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, Q.mean_phi - 0.02612796) / 3.162575e-07   # -0.0%  sum_z_dr2 < 0.006097 and mean_phi > 0.02613
        + 0.000411914 * max(0.0, 31.125 - Q.pt_4) / 0.3140792   # +0.0%  pt_4 < 31.12
        - 0.0003656698 * max(0.0, 53.33237 - Q.mass) * max(0.0, Q.dr1_7 - 0.2025074) / 0.02402033   # -0.0%  mass < 53.33 and dr1_7 > 0.2025
        - 0.0003503702 * max(0.0, Q.sum_pt_top5 - 839.9547) * max(0.0, 1.797343 - Q.pt1_dr01) / 5.637585   # -0.0%  sum_pt_top5 > 840 and pt1_dr01 < 1.797
        - 0.0001973215 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.n_dr_0p05_0p1 - 5.0) / 0.0003364245   # -0.0%  tau1 < 0.0437 and n_dr_0p05_0p1 > 5
        - 0.0001838028 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, Q.abseta_1 - 0.03625488) / 8.135786e-06   # -0.0%  sj3_dr_max < 0.107 and abseta_1 > 0.03625
        - 0.0001828161 * max(0.0, Q.e2 - 0.03556091) * max(0.0, Q.mean_phi - 0.009050008) / 3.742843e-05   # -0.0%  e2 > 0.03556 and mean_phi > 0.00905
        + 0.0001712511 * max(0.0, 31.125 - Q.pt_4) * max(0.0, Q.M3 - 0.05210278) / 0.0007324993   # +0.0%  pt_4 < 31.12 and M3 > 0.0521
        - 8.211929e-05 * max(0.0, Q.sum_pt_top5 - 839.9547) * max(0.0, 0.004811143 - Q.tau21_b2) / 0.0005843272   # -0.0%  sum_pt_top5 > 840 and tau21_b2 < 0.004811
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 8.2;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.200368 * (0.2138968
        + 0.1258827 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # +12.6%  sum_z_dr2 > 0.00752
        + 0.1125084 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +11.3%  mass < 76.66
        - 0.08533752 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -8.5%  lam1 < 0.004184
        - 0.06626881 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # -6.6%  lam1 > 0.00733
        - 0.05169165 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -5.2%  LHA > 0.3033
        + 0.0493569 * max(0.0, Q.lam2 - 0.0003061234) / 0.0004216599   # +4.9%  lam2 > 0.0003061
        + 0.03694663 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1) / 0.002082998   # +3.7%  zdr_0 < 0.02118 and z_dr_0p05_0p1 < 0.2919
        + 0.03456423 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # +3.5%  tau1 > 0.05357
        - 0.03453801 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -3.5%  lam1 < 0.005954
        - 0.03042175 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # -3.0%  sum_z_dr2 < 0.002635
        - 0.02748552 * Q.e2 / 0.02863215   # -2.7%  e2
        - 0.02739576 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -2.7%  zdr_0 < 0.02118
        + 0.0261893 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # +2.6%  sum_z_dr2_top3 < 0.002152
        + 0.02377948 * max(0.0, Q.mass_top5 - 22.18342) / 12.67741   # +2.4%  mass_top5 > 22.18
        - 0.02364833 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -2.4%  lam1 < 0.001504
        - 0.02015227 * max(0.0, 45.75 - Q.pt_7) / 12.14632   # -2.0%  pt_7 < 45.75
        - 0.01825518 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.planar_flow - 0.04505724) / 0.0002791469   # -1.8%  lam2 > 0.0003061 and planar_flow > 0.04506
        + 0.01781786 * max(0.0, 0.02355416 - Q.centroid_offset) / 0.0103615   # +1.8%  centroid_offset < 0.02355
        + 0.0177156 * max(0.0, Q.sj3_pair_mass_min - 15.95929) / 1.348548   # +1.8%  sj3_pair_mass_min > 15.96
        + 0.01685513 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.002471 - Q.D2) / 1.812028   # +1.7%  pt_7 < 45.75 and D2 < 1.002
        + 0.01671033 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, 1.002471 - Q.D2) / 0.009558568   # +1.7%  tau1 > 0.05357 and D2 < 1.002
        + 0.01308875 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +1.3%  sj3_dr_min > 0.1278
        - 0.01194074 * max(0.0, 0.06473447 - Q.z_7) / 0.01631636   # -1.2%  z_7 < 0.06473
        - 0.01153902 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 6.572938 - Q.log_sum_pt) / 1.257183   # -1.2%  pt_7 < 45.75 and log_sum_pt < 6.573
        - 0.009381743 * max(0.0, 0.00325401 - Q.zdr_7) / 0.001049641   # -0.9%  zdr_7 < 0.003254
        - 0.009055079 * max(0.0, Q.mass_top5 - 45.32077) / 3.61051   # -0.9%  mass_top5 > 45.32
        - 0.008657246 * max(0.0, 0.004918231 - Q.zdr_0) / 0.0006323791   # -0.9%  zdr_0 < 0.004918
        - 0.007676666 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2) / 0.0002929492   # -0.8%  lam1 > 0.00733 and D2_b2 < 0.3809
        - 0.007615903 * max(0.0, 0.02563286 - Q.M2) / 0.001872375   # -0.8%  M2 < 0.02563
        - 0.007503826 * max(0.0, Q.sj3_pair_mass_min - 11.051) * max(0.0, Q.sj3_pairmin_over_m - 0.28737) / 0.2416857   # -0.8%  sj3_pair_mass_min > 11.05 and sj3_pairmin_over_m > 0.2874
        - 0.007167574 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.3658817 - Q.z_dr_0_0p05) / 0.002463327   # -0.7%  sj3_dr_min > 0.1278 and z_dr_0_0p05 < 0.3659
        + 0.006647394 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.01258764) / 3.400017e-05   # +0.7%  zdr_0 < 0.02118 and centroid_offset > 0.01259
        - 0.006356804 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1797097) / 5.905455e-07   # -0.6%  e3 < 8.148e-05 and sj3_dr23 > 0.1797
        + 0.005880115 * max(0.0, Q.C2_b2 - 0.009032972) / 0.001703043   # +0.6%  C2_b2 > 0.009033
        - 0.003518132 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.4%  sum_pt > 988.4
        + 0.00341137 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # +0.3%  sj3_pair_mass_min > 11.05
        - 0.003179772 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.2919447) / 6.832392e-05   # -0.3%  lam1 < 0.005954 and z_dr_0p05_0p1 > 0.2919
        - 0.003177144 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -0.3%  e3 > 3.892e-05
        - 0.002762314 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -0.3%  lam2 > 0.003408
        - 0.00254288 * max(0.0, Q.sum_z_dr2 - 0.02530566) / 0.0002851418   # -0.3%  sum_z_dr2 > 0.02531
        + 0.001692115 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.dr_7 - 0.1078355) / 8.813877e-05   # +0.2%  zdr_0 < 0.02118 and dr_7 > 0.1078
        - 0.001655227 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, Q.D2_b2 - 1.129616) / 0.0002590844   # -0.2%  lam1 > 0.00733 and D2_b2 > 1.13
        - 0.001096094 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, 31.90625 - Q.pt_6) / 0.0003664138   # -0.1%  lam2 > 0.0003061 and pt_6 < 31.91
        - 0.0009327862 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.5973755 - Q.sj3_mass2) / 0.001008755   # -0.1%  sj3_dr_min > 0.1278 and sj3_mass2 < 0.5974
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 31.7;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.69984 * (-0.06447368
        + 0.1790135 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +17.9%  lam1_plus_lam2 < 0.008678
        + 0.1027445 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +10.3%  sum_z_dr2 < 0.01324
        + 0.05095156 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +5.1%  centroid_offset < 0.03776
        - 0.04977787 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -5.0%  lam1 < 0.008376
        - 0.0476322 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -4.8%  sum_zz_dr2 < 0.008169
        + 0.04684498 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +4.7%  z_7 > 0.01686
        - 0.04644382 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -4.6%  mass < 69.61
        - 0.04477245 * max(0.0, 0.169029 - Q.sj3_dr_max) / 0.04876366   # -4.5%  sj3_dr_max < 0.169
        - 0.04356252 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # -4.4%  sum_z_dr < 0.0717
        - 0.0429179 * max(0.0, 0.07637363 - Q.mass_over_sum_pt) / 0.02715027   # -4.3%  mass_over_sum_pt < 0.07637
        + 0.03906827 * max(0.0, 0.2623172 - Q.sj3_dr_max) / 0.1129006   # +3.9%  sj3_dr_max < 0.2623
        + 0.03879295 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) / 0.00293624   # +3.9%  lam1_plus_lam2 < 0.006679
        - 0.02421916 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # -2.4%  tau1 < 0.09539
        - 0.02353951 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -2.4%  lam1 < 0.00733
        + 0.02312949 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +2.3%  planar_flow < 0.2534
        - 0.01899239 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt) / 3.513015   # -1.9%  centroid_offset < 0.03776 and sum_pt < 901.6
        - 0.01545558 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 15.95929 - Q.sj3_pair_mass_min) / 0.05098851   # -1.5%  centroid_offset < 0.01438 and sj3_pair_mass_min < 15.96
        - 0.01113587 * max(0.0, Q.pt_7 - 29.04219) / 7.785987   # -1.1%  pt_7 > 29.04
        + 0.01002339 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # +1.0%  max_dr < 0.1452
        + 0.009965055 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +1.0%  sj3_dr_max < 0.1426
        - 0.009800064 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.tau4 - 0.002207727) / 2.478129e-06   # -1.0%  centroid_offset > 0.0499 and tau4 > 0.002208
        + 0.00957278 * max(0.0, 15.45403 - Q.mass) / 2.385786   # +1.0%  mass < 15.45
        - 0.008570788 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # -0.9%  C2 < 0.03579
        + 0.007862551 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +0.8%  centroid_offset > 0.0499
        + 0.007720424 * max(0.0, 0.01655442 - Q.e2) / 0.003887347   # +0.8%  e2 < 0.01655
        + 0.007547792 * max(0.0, 1.050302e-05 - Q.e3) / 3.717236e-06   # +0.8%  e3 < 1.05e-05
        - 0.006770856 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.7%  planar_flow < 0.2534 and sum_pt < 840
        - 0.005250947 * max(0.0, 0.02689598 - Q.sum_z_dr) / 0.004330137   # -0.5%  sum_z_dr < 0.0269
        - 0.00466959 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # -0.5%  centroid_offset > 0.0499 and D2 < 2.844
        + 0.004392791 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # +0.4%  sj2_dr > 0.1779
        + 0.004137055 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0007565864   # +0.4%  centroid_offset < 0.01438 and D2_b2 < 0.5327
        + 0.004073502 * max(0.0, 0.09538712 - Q.tau1) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.03209574   # +0.4%  tau1 < 0.09539 and z_dr_0p05_0p1 < 0.846
        - 0.003969485 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 0.0345459 - Q.absphi_0) / 0.0003803849   # -0.4%  centroid_offset < 0.03776 and absphi_0 < 0.03455
        - 0.003909642 * max(0.0, 506.875 - Q.sum_pt_top5) / 34.75344   # -0.4%  sum_pt_top5 < 506.9
        + 0.003515189 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.4%  sj2_dr > 0.2688
        - 0.003490936 * max(0.0, Q.n_dr_0p1_0p2 - 3.0) / 0.3318622   # -0.3%  n_dr_0p1_0p2 > 3
        + 0.00326949 * max(0.0, Q.eccentricity - 0.9884745) / 0.001961982   # +0.3%  eccentricity > 0.9885
        - 0.002837657 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 4.721224 - Q.D2_b2) / 16.05793   # -0.3%  pt_6 < 41.22 and D2_b2 < 4.721
        + 0.002794743 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +0.3%  pt_6 < 41.22
        - 0.00270828 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 1.050302e-05 - Q.e3) / 1.771164e-07   # -0.3%  planar_flow < 0.2534 and e3 < 1.05e-05
        - 0.002595382 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, Q.sj3_pair_mass_min - 1.282345) / 0.4025762   # -0.3%  planar_flow < 0.2534 and sj3_pair_mass_min > 1.282
        - 0.002398137 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 37.15625 - Q.pt_7) / 0.5537514   # -0.2%  planar_flow < 0.2534 and pt_7 < 37.16
        - 0.002222685 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001130645 - Q.lam2) / 1.843299e-05   # -0.2%  sj2_dr > 0.1779 and lam2 < 0.001131
        - 0.001739393 * max(0.0, 0.0005049491 - Q.lam1) / 8.739619e-05   # -0.2%  lam1 < 0.0005049
        - 0.001676818 * max(0.0, Q.eccentricity - 0.9884745) * max(0.0, 2.771069 - Q.sj2_mass2) / 0.00304207   # -0.2%  eccentricity > 0.9885 and sj2_mass2 < 2.771
        - 0.001600147 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.03601074 - Q.abseta_4) / 8.179649e-05   # -0.2%  centroid_offset < 0.01438 and abseta_4 < 0.03601
        - 0.001499627 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.zdr_7 - 0.004322471) / 4.458454e-06   # -0.1%  centroid_offset > 0.0499 and zdr_7 > 0.004322
        - 0.001304582 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, -0.001813533 - Q.mean_phi) / 4.860641e-05   # -0.1%  centroid_offset < 0.03776 and mean_phi < -0.001814
        - 0.001228736 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.00279494 - Q.zdr_7) / 5.690817e-06   # -0.1%  centroid_offset < 0.01438 and zdr_7 < 0.002795
        + 0.001101416 * max(0.0, 0.01437952 - Q.centroid_offset) / 0.004306483   # +0.1%  centroid_offset < 0.01438
        - 0.001085381 * max(0.0, 24.42188 - Q.pt_6) / 0.6046572   # -0.1%  pt_6 < 24.42
        + 0.001064216 * max(0.0, 0.2623172 - Q.sj3_dr_max) * max(0.0, -0.0216713 - Q.phi_0) / 0.0003847124   # +0.1%  sj3_dr_max < 0.2623 and phi_0 < -0.02167
        - 0.0009197763 * max(0.0, 0.02689598 - Q.sum_z_dr) * max(0.0, Q.sj3_pair_mass_min - 1.590484) / 0.005084766   # -0.1%  sum_z_dr < 0.0269 and sj3_pair_mass_min > 1.59
        - 0.0008785177 * max(0.0, 0.008168571 - Q.sum_zz_dr2) * max(0.0, Q.mass_top2 - 6.779915) / 0.005654747   # -0.1%  sum_zz_dr2 < 0.008169 and mass_top2 > 6.78
        - 0.000870812 * max(0.0, 0.004372139 - Q.sum_z_dr2) / 0.00156571   # -0.1%  sum_z_dr2 < 0.004372
        - 0.0006515177 * max(0.0, Q.max_pair_mass - 33.3761) / 0.9279851   # -0.1%  max_pair_mass > 33.38
        + 0.0005239615 * max(0.0, 506.875 - Q.sum_pt_top5) * max(0.0, Q.n_dr_0p1_0p2 - 0.0) / 79.19632   # +0.1%  sum_pt_top5 < 506.9 and n_dr_0p1_0p2 > 0
        + 0.0004927365 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.04881348 - Q.dr_7) / 8.444248e-05   # +0.0%  sj2_dr > 0.1779 and dr_7 < 0.04881
        - 0.0002848752 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.0%  sum_pt > 988.4
        + 1.369147e-05 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, -0.07818909 - Q.phi_0) / 0.0007184678   # +0.0%  sum_pt > 988.4 and phi_0 < -0.07819
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.5661;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5661452 * (-0.769889
        + 0.4730696 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # +47.3%  sum_z_dr2 > 0.01883
        - 0.2379095 * max(0.0, Q.mass_over_sum_pt - 0.1309286) / 0.002541185   # -23.8%  mass_over_sum_pt > 0.1309
        + 0.08506414 * max(0.0, Q.mass - 88.15578) / 0.7232262   # +8.5%  mass > 88.16
        - 0.07231243 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, 53.4375 - Q.pt_7) / 0.01439677   # -7.2%  sum_z_dr2 > 0.01883 and pt_7 < 53.44
        - 0.05922705 * max(0.0, Q.sum_z_dr2_top2 - 0.01403324) / 0.001129575   # -5.9%  sum_z_dr2_top2 > 0.01403
        + 0.02669294 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, Q.lam2 - 0.000537286) / 2.828867e-06   # +2.7%  sum_z_dr2 > 0.01883 and lam2 > 0.0005373
        - 0.02505108 * max(0.0, Q.zdr_0 - 0.03981924) / 0.0005695785   # -2.5%  zdr_0 > 0.03982
        + 0.02067323 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, Q.abseta_6 - 0.02839661) / 6.31192e-05   # +2.1%  sum_z_dr2 > 0.01883 and abseta_6 > 0.0284
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 16.95;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.94552 * (0.02259309
        + 0.3305968 * max(0.0, 0.1484084 - Q.sum_z_dr) / 0.0905999   # +33.1%  sum_z_dr < 0.1484
        - 0.1401638 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -14.0%  e2 < 0.08001
        + 0.09770145 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +9.8%  lam1 < 0.01643
        - 0.07565348 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -7.6%  sum_z_dr < 0.1484 and log_sum_pt < 6.804
        - 0.04414711 * max(0.0, 0.007520088 - Q.lam1_plus_lam2) / 0.003544991   # -4.4%  lam1_plus_lam2 < 0.00752
        - 0.04319805 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 8.324263e-07   # -4.3%  e3 < 5.335e-05 and centroid_offset < 0.03776
        - 0.0378655 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -3.8%  sum_z_dr < 0.1484 and pt_7 < 38.53
        + 0.03088253 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +3.1%  lam1 < 0.006507
        + 0.02908383 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7) / 781.5808   # +2.9%  sum_pt_top5 > 658.1 and pt_7 < 40.04
        + 0.02670644 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +2.7%  lam1_plus_lam2 < 0.01324
        + 0.02051581 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # +2.1%  e3 < 5.335e-05
        - 0.0181049 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.000537286 - Q.lam2) / 4.021081e-05   # -1.8%  sum_z_dr < 0.1484 and lam2 < 0.0005373
        - 0.01294244 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7) / 0.06765006   # -1.3%  pt_6 < 31.91 and z_7 < 0.05861
        - 0.009849177 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # -1.0%  z_7 < 0.02807
        + 0.00935893 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 31.90625 - Q.pt_6) / 134.9952   # +0.9%  sum_pt_top5 > 791.1 and pt_6 < 31.91
        + 0.008363795 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.D3 - 0.2213841) / 3.794981e-05   # +0.8%  e3 < 5.335e-05 and D3 > 0.2214
        + 0.007540423 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +0.8%  log_sum_pt > 6.503
        + 0.007034336 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.z_7 - 0.06164517) / 0.0003465887   # +0.7%  sum_z_dr < 0.1484 and z_7 > 0.06165
        - 0.006274798 * max(0.0, Q.mass - 49.6681) / 7.721594   # -0.6%  mass > 49.67
        + 0.005968482 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +0.6%  sum_pt_top5 > 791.1
        + 0.004376042 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.tau2 - 0.008780509) / 0.000269945   # +0.4%  sum_z_dr < 0.1484 and tau2 > 0.008781
        + 0.004162232 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.07474969 - Q.M3) / 0.0006773733   # +0.4%  sum_z_dr < 0.1484 and M3 < 0.07475
        - 0.003887423 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # -0.4%  z_5 < 0.02818
        - 0.003813895 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6) / 104.494   # -0.4%  sum_pt > 988.4 and pt_6 < 62.25
        - 0.003658061 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.zdr_7 - 0.0007928864) / 0.0001818422   # -0.4%  log_sum_pt > 6.503 and zdr_7 > 0.0007929
        - 0.003598444 * max(0.0, Q.sj3_dr23 - 0.1974628) / 0.02478939   # -0.4%  sj3_dr23 > 0.1975
        - 0.002556804 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # -0.3%  sum_z_dr < 0.007674
        - 0.002013893 * max(0.0, 31.90625 - Q.pt_6) / 1.826854   # -0.2%  pt_6 < 31.91
        - 0.001902268 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.2%  z_6 < 0.0216
        - 0.001635087 * max(0.0, Q.mass - 49.6681) * max(0.0, 0.4684459 - Q.z_dr_0p1_0p2) / 1.50172   # -0.2%  mass > 49.67 and z_dr_0p1_0p2 < 0.4684
        + 0.001509061 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.pair_mass_0_7 - 10.2219) / 0.2907061   # +0.2%  log_sum_pt > 6.503 and pair_mass_0_7 > 10.22
        - 0.001453226 * max(0.0, Q.pt_7 - 48.71875) / 0.6759439   # -0.1%  pt_7 > 48.72
        - 0.001419564 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.02320757) / 0.2899084   # -0.1%  sum_pt_top5 > 658.1 and z_7 > 0.02321
        - 0.001075463 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.sj2_mass1 - 31.78116) / 0.01215768   # -0.1%  sum_z_dr < 0.1484 and sj2_mass1 > 31.78
        - 0.0009864584 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.1%  sum_pt > 988.4
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 40.94;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 40.93912 * (0.002923318
        - 0.08966516 * max(0.0, Q.lam1_plus_lam2 - 0.006679471) / 0.002702003   # -9.0%  lam1_plus_lam2 > 0.006679
        - 0.05588995 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -5.6%  sum_z_dr2 > 0.00752
        + 0.05187291 * max(0.0, Q.sum_z_dr - 0.02689598) / 0.03613842   # +5.2%  sum_z_dr > 0.0269
        - 0.04929158 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # -4.9%  sum_z_dr > 0.08724
        - 0.04417476 * max(0.0, Q.e2 - 0.01655442) / 0.01596508   # -4.4%  e2 > 0.01655
        - 0.04354708 * max(0.0, 0.324646 - Q.sd_rg) / 0.2082097   # -4.4%  sd_rg < 0.3246
        + 0.04260011 * max(0.0, Q.lam1_plus_lam2 - 0.003562611) / 0.004066872   # +4.3%  lam1_plus_lam2 > 0.003563
        + 0.04092854 * max(0.0, Q.sum_zz_dr2 - 0.002074109) / 0.004484017   # +4.1%  sum_zz_dr2 > 0.002074
        + 0.03886599 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +3.9%  mass_over_sum_pt > 0.08475
        + 0.03376588 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # +3.4%  sj2_dr > 0.1592
        + 0.0318933 * max(0.0, Q.sum_z_dr - 0.04081947) / 0.0270611   # +3.2%  sum_z_dr > 0.04082
        + 0.03100478 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +3.1%  mass_over_sum_pt > 0.07992
        + 0.02965206 * max(0.0, 49.91626 - Q.sd_mass) / 22.60916   # +3.0%  sd_mass < 49.92
        - 0.02894295 * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 0.001433269   # -2.9%  sum_zz_dr2 > 0.01166
        - 0.02848294 * max(0.0, Q.sum_z_dr2 - 0.0009641429) / 0.00568632   # -2.8%  sum_z_dr2 > 0.0009641
        - 0.02729715 * max(0.0, Q.sum_zz_dr2 - 0.0030133) / 0.003940214   # -2.7%  sum_zz_dr2 > 0.003013
        + 0.02679169 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +2.7%  e2 > 0.05028
        - 0.01814214 * max(0.0, Q.sum_z_dr - 0.1019409) / 0.005400003   # -1.8%  sum_z_dr > 0.1019
        - 0.01797999 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -1.8%  e3 < 8.148e-05
        - 0.0162217 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -1.6%  centroid_offset > 0.0499
        + 0.01504896 * max(0.0, Q.z_dr_0_0p05 - 0.1515405) / 0.4480651   # +1.5%  z_dr_0_0p05 > 0.1515
        - 0.01247128 * max(0.0, 74.57663 - Q.sd_mass) / 42.04056   # -1.2%  sd_mass < 74.58
        + 0.01168182 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +1.2%  LHA > 0.3033
        - 0.01153181 * max(0.0, Q.sj2_dr - 0.06154135) / 0.1002688   # -1.2%  sj2_dr > 0.06154
        - 0.01090466 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # -1.1%  sj2_dr > 0.1295
        - 0.01070748 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # -1.1%  sum_z_dr2 > 0.008678
        + 0.01043743 * max(0.0, Q.sum_z_dr - 0.08065885) / 0.009330634   # +1.0%  sum_z_dr > 0.08066
        - 0.01033414 * max(0.0, 0.004032342 - Q.C2_b2) / 0.002883542   # -1.0%  C2_b2 < 0.004032
        + 0.009876073 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # +1.0%  lam2 < 0.0005373
        + 0.009825359 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # +1.0%  lam1_plus_lam2 > 0.00559
        + 0.008791429 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # +0.9%  planar_flow < 0.1115 and lam1 < 0.01643
        + 0.008703667 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0006014625   # +0.9%  sj2_dr > 0.1592 and D2_b2 < 0.08499
        + 0.008594299 * max(0.0, 5.0 - Q.n_dr_0_0p05) / 1.94322   # +0.9%  n_dr_0_0p05 < 5
        - 0.008346812 * max(0.0, 715.4688 - Q.sum_pt) / 69.91196   # -0.8%  sum_pt < 715.5
        - 0.007915243 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # -0.8%  lam1_plus_lam2 > 0.01324
        + 0.006636108 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +0.7%  e2 > 0.04111
        - 0.00628334 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # -0.6%  e3 < 5.335e-05
        - 0.005541454 * max(0.0, Q.sj2_dr - 0.2001708) / 0.02317349   # -0.6%  sj2_dr > 0.2002
        - 0.005422408 * max(0.0, 0.163898 - Q.z_dr_0p05_0p1) / 0.08682106   # -0.5%  z_dr_0p05_0p1 < 0.1639
        - 0.005075382 * max(0.0, Q.sj2_dr - 0.1294903) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0009393137   # -0.5%  sj2_dr > 0.1295 and D2_b2 < 0.08499
        - 0.004796376 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -0.5%  sj2_dr > 0.1779
        - 0.004789861 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -0.5%  planar_flow < 0.1115 and lam1 < 0.00733
        - 0.004784144 * max(0.0, Q.sd_rg - 0.2330919) / 0.01070351   # -0.5%  sd_rg > 0.2331
        + 0.004652473 * max(0.0, 0.1115136 - Q.planar_flow) / 0.03343187   # +0.5%  planar_flow < 0.1115
        - 0.003782938 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0003170016   # -0.4%  sj2_dr > 0.2002 and D2_b2 < 0.08499
        - 0.003772752 * max(0.0, 0.08366273 - Q.planar_flow) / 0.02146305   # -0.4%  planar_flow < 0.08366
        - 0.003771373 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.4%  psi_0p1 > 0.9761
        - 0.003694597 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 0.02415398 - Q.C2_b2) / 0.0005249308   # -0.4%  z_dr_0p05_0p1 > 0.7509 and C2_b2 < 0.02415
        + 0.003472556 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067297   # +0.3%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        + 0.00340449 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # +0.3%  sd_rg > 0.2788
        + 0.003398618 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.0008333816 - Q.C2_b2) / 1.341084e-07   # +0.3%  centroid_offset > 0.0499 and C2_b2 < 0.0008334
        - 0.002979823 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.3%  centroid_offset > 0.03776
        - 0.002751351 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -0.3%  e2 > 0.03556
        - 0.002593685 * max(0.0, Q.psi_0p1 - 0.7143804) / 0.1805043   # -0.3%  psi_0p1 > 0.7144
        - 0.00159562 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.2%  planar_flow < 0.1115 and centroid_offset < 0.01838
        + 0.001591403 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1629004) / 4.043869e-07   # +0.2%  e3 < 5.335e-05 and sj3_dr23 > 0.1629
        + 0.001208621 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 715.4688 - Q.sum_pt) / 22.02426   # +0.1%  z_dr_0p05_0p1 < 0.5883 and sum_pt < 715.5
        - 0.001193864 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 658.125 - Q.sum_pt_top5) / 2.950014   # -0.1%  planar_flow < 0.1115 and sum_pt_top5 < 658.1
        - 0.0009975634 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 0.1830092 - Q.D2_b2) / 0.008544403   # -0.1%  z_dr_0p05_0p1 < 0.5883 and D2_b2 < 0.183
        - 0.00091282 * max(0.0, Q.mass - 69.61135) / 2.406702   # -0.1%  mass > 69.61
        - 0.0008783169 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, Q.mean_eta - -0.006779839) / 0.000320267   # -0.1%  planar_flow < 0.1115 and mean_eta > -0.00678
        + 0.0008272958 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 3.351526e-05   # +0.1%  planar_flow < 0.1115 and lam1_plus_lam2 < 0.006097
        - 0.0008252512 * max(0.0, Q.sum_zz_dr2 - 0.002074109) * max(0.0, 0.1872617 - Q.sj2_dr) / 2.014151e-05   # -0.1%  sum_zz_dr2 > 0.002074 and sj2_dr < 0.1873
        - 0.0007850698 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.1058993 - Q.z_5) / 0.0007616364   # -0.1%  sj2_dr > 0.2002 and z_5 < 0.1059
        - 0.0007692931 * max(0.0, Q.sd_rg - 0.2787955) * max(0.0, 0.2669656 - Q.D2_b2) / 0.0004100902   # -0.1%  sd_rg > 0.2788 and D2_b2 < 0.267
        - 0.000769114 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.03293672 - Q.dr_2) / 6.256053e-05   # -0.1%  sj2_dr > 0.1592 and dr_2 < 0.03294
        + 0.0007604874 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.05056028 - Q.dr_2) / 0.0001260892   # +0.1%  sj2_dr > 0.2002 and dr_2 < 0.05056
        - 0.0007352494 * max(0.0, Q.mass_top5 - 49.18618) / 2.747087   # -0.1%  mass_top5 > 49.19
        + 0.0005813496 * max(0.0, Q.psi_0p1 - 0.8155839) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0006308268   # +0.1%  psi_0p1 > 0.8156 and D2_b2 < 0.08499
        + 0.0005702357 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, 150.625 - Q.pt_1) / 0.1837865   # +0.1%  centroid_offset > 0.02686 and pt_1 < 150.6
        - 0.0004220516 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -0.0%  mass_over_sum_pt > 0.09041
        + 0.0004130268 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 0.3201347 - Q.D3) / 0.004766833   # +0.0%  z_dr_0p05_0p1 < 0.5883 and D3 < 0.3201
        - 0.0002157639 * max(0.0, Q.sum_z_dr - 0.02689598) * max(0.0, Q.D2_b2 - 4.721224) / 0.0001187831   # -0.0%  sum_z_dr > 0.0269 and D2_b2 > 4.721
        - 0.0001627386 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, Q.zdr_5 - 0.0155217) / 1.782544e-06   # -0.0%  z_dr_0p05_0p1 > 0.7509 and zdr_5 > 0.01552
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 43.93;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 43.93107 * (-0.02975453
        + 0.09461361 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +9.5%  lam1_plus_lam2 < 0.01324
        + 0.07416195 * max(0.0, 0.01882765 - Q.lam1_plus_lam2) * max(0.0, 0.003408389 - Q.lam2) / 4.313596e-05   # +7.4%  lam1_plus_lam2 < 0.01883 and lam2 < 0.003408
        - 0.06308692 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -6.3%  sum_zz_dr2 < 0.008169
        + 0.05920273 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +5.9%  e2 < 0.04111
        - 0.0521997 * max(0.0, 0.007520088 - Q.sum_z_dr2) / 0.00354499   # -5.2%  sum_z_dr2 < 0.00752
        - 0.05159113 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # -5.2%  sum_zz_dr2 < 0.01166
        + 0.05123112 * max(0.0, 0.06345984 - Q.tau1) / 0.02211428   # +5.1%  tau1 < 0.06346
        + 0.04953748 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +5.0%  sj2_dr < 0.1592
        - 0.04840533 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -4.8%  sum_z_dr < 0.1019
        - 0.04413195 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -4.4%  sum_z_dr < 0.08724
        - 0.0407164 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 0.002544039   # -4.1%  lam1_plus_lam2 < 0.006097
        + 0.03854364 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +3.9%  mass_over_sum_pt_sq < 0.007183
        - 0.03281835 * max(0.0, 0.2001708 - Q.sj2_dr) / 0.07331403   # -3.3%  sj2_dr < 0.2002
        - 0.02212974 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # -2.2%  lam1 < 0.00484
        - 0.02051678 * max(0.0, 0.1492731 - Q.sj2_dr) / 0.04374278   # -2.1%  sj2_dr < 0.1493
        - 0.01984104 * max(0.0, 0.03846696 - Q.e2) / 0.01567344   # -2.0%  e2 < 0.03847
        + 0.01731612 * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) / 0.004543329   # +1.7%  sum_z_dr2_top2 < 0.00764
        - 0.01671533 * max(0.0, 0.06344108 - Q.e2) / 0.03657078   # -1.7%  e2 < 0.06344
        + 0.01530055 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +1.5%  lam2 < 0.001131
        - 0.01406373 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.01216943 - Q.zdr_0) / 3.637364e-05   # -1.4%  centroid_offset < 0.01838 and zdr_0 < 0.01217
        + 0.01181553 * max(0.0, 0.006756161 - Q.sum_z_dr2_top3) / 0.003650633   # +1.2%  sum_z_dr2_top3 < 0.006756
        - 0.01171678 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2512159 - Q.dr01) / 0.001381924   # -1.2%  centroid_offset < 0.01838 and dr01 < 0.2512
        + 0.01169089 * max(0.0, Q.mass - 45.7571) / 9.387753   # +1.2%  mass > 45.76
        + 0.01135196 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # +1.1%  lam1 < 0.005433
        + 0.01061221 * max(0.0, 0.0005124533 - Q.sum_z_dr2_top2) / 0.0001104529   # +1.1%  sum_z_dr2_top2 < 0.0005125
        - 0.00959238 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # -1.0%  N2 < 0.2233
        - 0.009388933 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # -0.9%  lam2 < 0.0003061
        + 0.008564563 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +0.9%  N2 < 0.2233 and sum_pt_top5 > 430.8
        + 0.007985774 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.04262929   # +0.8%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 2
        - 0.007850403 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # -0.8%  z_dr_0p05_0p1 > 0.7509
        + 0.006372773 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 716.8828 - Q.sum_pt_top5) / 7.315623   # +0.6%  N2 < 0.2233 and sum_pt_top5 < 716.9
        + 0.005700432 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) * max(0.0, 0.01272888 - Q.tau21_b2) / 8.968194e-06   # +0.6%  lam1_plus_lam2 < 0.01324 and tau21_b2 < 0.01273
        - 0.005697777 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -0.6%  centroid_offset < 0.01838
        - 0.005548447 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.01272888 - Q.tau21_b2) / 7.280144e-05   # -0.6%  mass_over_sum_pt < 0.1309 and tau21_b2 < 0.01273
        + 0.004921873 * max(0.0, 0.01165737 - Q.sum_zz_dr2) * max(0.0, Q.lam2 - 6.531723e-06) / 7.375237e-07   # +0.5%  sum_zz_dr2 < 0.01166 and lam2 > 6.532e-06
        - 0.004775503 * max(0.0, 0.007520088 - Q.sum_z_dr2) * max(0.0, 0.7459513 - Q.D2) / 9.127003e-05   # -0.5%  sum_z_dr2 < 0.00752 and D2 < 0.746
        - 0.004473065 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, Q.mean_phi - -0.01753483) / 1.673325e-05   # -0.4%  lam2 < 0.001131 and mean_phi > -0.01753
        - 0.004385855 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, Q.n_for_50pct - 1.0) / 0.001375293   # -0.4%  lam2 < 0.001131 and n_for_50pct > 1
        - 0.003468995 * max(0.0, Q.sj3_pair_mass_max - 37.79615) / 7.959239   # -0.3%  sj3_pair_mass_max > 37.8
        + 0.003267318 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.1094252   # +0.3%  N2 < 0.2233 and n_dr_0p1_0p2 < 4
        + 0.003105349 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) * max(0.0, 0.7459513 - Q.D2) / 3.18925e-05   # +0.3%  lam1_plus_lam2 < 0.006097 and D2 < 0.746
        + 0.00229281 * max(0.0, 0.0262518 - Q.tau1) / 0.005366404   # +0.2%  tau1 < 0.02625
        - 0.002263829 * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 0.03639977   # -0.2%  z_dr_0p05_0p1 > 0.6748
        - 0.00178009 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.0114178   # -0.2%  N2 < 0.2233 and z_dr_0p05_0p1 < 0.5883
        - 0.00177837 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.2%  mass > 76.66
        + 0.001765446 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +0.2%  LHA < 0.3033
        + 0.001696498 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2) / 0.003993721   # +0.2%  mass_over_sum_pt < 0.1309 and D2 < 0.746
        + 0.001309909 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # +0.1%  sum_z_dr2_top3 < 0.002152
        - 0.001065839 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.121681 - Q.max_dr) / 0.0004905272   # -0.1%  N2 < 0.2233 and max_dr < 0.1217
        + 0.0009622038 * max(0.0, 0.01882765 - Q.lam1_plus_lam2) * max(0.0, Q.dr1_3 - 0.0717773) / 0.0002620031   # +0.1%  lam1_plus_lam2 < 0.01883 and dr1_3 > 0.07178
        + 0.0007844378 * max(0.0, 0.06344108 - Q.e2) * max(0.0, Q.dr12 - 0.119512) / 0.0002640953   # +0.1%  e2 < 0.06344 and dr12 > 0.1195
        - 0.0007808201 * max(0.0, 6.080494 - Q.log_sum_pt) / 0.006097657   # -0.1%  log_sum_pt < 6.08
        - 0.0006664152 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) * max(0.0, 0.08499387 - Q.D2_b2) / 3.927041e-06   # -0.1%  lam1_plus_lam2 < 0.006097 and D2_b2 < 0.08499
        - 0.0006402856 * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) * max(0.0, 0.238708 - Q.z_2nd) / 0.002174277   # -0.1%  z_dr_0p05_0p1 > 0.6748 and z_2nd < 0.2387
        + 0.0005040666 * max(0.0, 0.01882765 - Q.lam1_plus_lam2) * max(0.0, Q.ptdr0_2 - 14.78048) / 0.005175612   # +0.1%  lam1_plus_lam2 < 0.01883 and ptdr0_2 > 14.78
        - 0.0004815411 * max(0.0, Q.mass - 45.7571) * max(0.0, 23.21641 - Q.pt_7) / 5.139381   # -0.0%  mass > 45.76 and pt_7 < 23.22
        - 0.0004668814 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 6.0 - Q.n_for_90pct) / 0.005228592   # -0.0%  N2 < 0.2233 and n_for_90pct < 6
        - 0.0004211816 * max(0.0, 0.01882765 - Q.lam1_plus_lam2) * max(0.0, Q.lam2 - 0.001130645) / 2.453968e-07   # -0.0%  lam1_plus_lam2 < 0.01883 and lam2 > 0.001131
        - 0.0003744132 * max(0.0, 0.1019409 - Q.sum_z_dr) * max(0.0, Q.dr1_3 - 0.1637906) / 0.0001271335   # -0.0%  sum_z_dr < 0.1019 and dr1_3 > 0.1638
        - 0.0003673093 * max(0.0, Q.sj3_pair_mass_max - 72.58129) * max(0.0, 1.666763 - Q.N3) / 0.3484543   # -0.0%  sj3_pair_mass_max > 72.58 and N3 < 1.667
        + 0.0003477049 * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) * max(0.0, Q.dr1_6 - 0.1261715) / 0.0009098173   # +0.0%  z_dr_0p05_0p1 > 0.6748 and dr1_6 > 0.1262
        + 0.0002925162 * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) * max(0.0, Q.pt_3 - 68.75) / 0.383852   # +0.0%  z_dr_0p05_0p1 > 0.6748 and pt_3 > 68.75
        + 0.0002760626 * max(0.0, Q.sj3_pair_mass_max - 37.79615) * max(0.0, Q.n_pt_above_50 - 7.0) / 0.5841427   # +0.0%  sj3_pair_mass_max > 37.8 and n_pt_above_50 > 7
        - 0.0001587018 * max(0.0, 0.0262518 - Q.tau1) * max(0.0, Q.zdr_3 - 0.008191788) / 3.916286e-08   # -0.0%  tau1 < 0.02625 and zdr_3 > 0.008192
        - 0.0001122511 * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) * max(0.0, Q.pair_mass_0_6 - 24.79428) / 0.00796161   # -0.0%  z_dr_0p05_0p1 > 0.6748 and pair_mass_0_6 > 24.79
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9251262605042017, 0.7820002100840336, 2.098969432773109, 0.9591974789915967, 1.2118710084033613, 1.8212344537815126, 0.9866806722689075, 1.8603088235294118, 0.28549243697478993, 3.041075630252101, 2.0015236344537817, 2.122805987394958, 0.09282247899159664, 3.3088756302521007, 0.41072689075630253, 0.3862151260504202]
T = [2.361876239988183, 1.3866386423319328, 3.306011016281513, 2.492814279149159, 2.8759405888918064]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +38%, n9 +22%, n5 -14%, n1 +13%, n0 -6%, n6 +5% ...
            + 0.3818578 * h[2] / H_AVG[2]
            + 0.2213007 * h[9] / H_AVG[9]
            - 0.1445806 * h[5] / H_AVG[5]
            + 0.1293331 * h[1] / H_AVG[1]
            - 0.06120176 * h[0] / H_AVG[0]
            + 0.04569172 * h[6] / H_AVG[6]
            - 0.01603427 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -18%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5568488 * h[9] / H_AVG[9]
            - 0.1804295 * h[10] / H_AVG[10]
            + 0.08894537 * h[6] / H_AVG[6]
            - 0.08193404 * h[4] / H_AVG[4]
            + 0.06156641 * h[5] / H_AVG[5]
            + 0.01740788 * h[15] / H_AVG[15]
            + 0.01286801 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -15%, n7 +12%, n0 +10%, n6 -9%, n14 -9% ...
            + 0.2407894 * h[11] / H_AVG[11]
            - 0.1450687 * h[3] / H_AVG[3]
            + 0.1230917 * h[7] / H_AVG[7]
            + 0.0961921 * h[0] / H_AVG[0]
            - 0.09326578 * h[6] / H_AVG[6]
            - 0.0931773 * h[14] / H_AVG[14]
            - 0.08031519 * h[15] / H_AVG[15]
            + 0.07037342 * h[13] / H_AVG[13]
            - 0.0287457 * h[9] / H_AVG[9]
            - 0.02158889 * h[8] / H_AVG[8]
            - 0.007391841 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -22%, n6 -15%, n13 +7%, n14 +6%, n1 +4% ...
            + 0.3498134 * h[7] / H_AVG[7]
            - 0.2164415 * h[3] / H_AVG[3]
            - 0.1484287 * h[6] / H_AVG[6]
            + 0.0725903 * h[13] / H_AVG[13]
            + 0.06178663 * h[14] / H_AVG[14]
            + 0.03921272 * h[1] / H_AVG[1]
            - 0.03812302 * h[9] / H_AVG[9]
            + 0.03798013 * h[4] / H_AVG[4]
            - 0.02420803 * h[15] / H_AVG[15]
            + 0.01141553 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +26%, n5 -16%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4674056 * h[13] / H_AVG[13]
            + 0.2609829 * h[10] / H_AVG[10]
            - 0.1583164 * h[5] / H_AVG[5]
            + 0.05267281 * h[4] / H_AVG[4]
            + 0.0208453 * h[3] / H_AVG[3]
            + 0.01861298 * h[8] / H_AVG[8]
            - 0.01613776 * h[12] / H_AVG[12]
            + 0.005026216 * h[0] / H_AVG[0]
        ),
    ]]


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
