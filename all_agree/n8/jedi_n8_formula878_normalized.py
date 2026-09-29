"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; all observables, tuned for agreement), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  6:  11.8%   (on for 72% of jets)
  neuron 13:  11.5%   (on for 87% of jets)
  neuron  7:  10.5%   (on for 72% of jets)
  neuron  5:   9.2%   (on for 69% of jets)
  neuron  3:   8.8%   (on for 34% of jets)
  neuron 10:   8.6%   (on for 88% of jets)
  neuron 11:   7.3%   (on for 87% of jets)
  neuron  9:   6.7%   (on for 66% of jets)
  neuron  2:   5.4%   (on for 64% of jets)
  neuron  0:   5.3%   (on for 83% of jets)
  neuron  1:   4.1%   (on for 75% of jets)
  neuron 14:   4.0%   (on for 39% of jets)
  neuron  4:   3.7%   (on for 89% of jets)
  neuron 15:   1.7%   (on for 35% of jets)
  neuron  8:   1.1%   (on for 37% of jets)
  neuron 12:   0.3%   (on for 8% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.3% (the network: 65.8%); same class as the network for 87.1% of jets.

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
  Q.sj3_pairmax_over_m     largest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_4          mass of particles 0 and 4 [GeV]
  Q.pair_mass_0_5          mass of particles 0 and 5 [GeV]
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.pair_mass_0_7          mass of particles 0 and 7 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.ptdr0_7                pT7 · ΔR(0, 7) [GeV]
  Q.z_3                    pT of particle 3 / total pT
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_1               |Δη| of particle 1
  Q.abseta_2               |Δη| of particle 2
  Q.abseta_3               |Δη| of particle 3
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_5               |Δη| of particle 5
  Q.abseta_6               |Δη| of particle 6
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_2               |Δφ| of particle 2
  Q.absphi_3               |Δφ| of particle 3
  Q.absphi_5               |Δφ| of particle 5
  Q.absphi_6               |Δφ| of particle 6
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_3                  ΔR between particle 3 and the hardest particle
  Q.dr0_5                  ΔR between particle 5 and the hardest particle
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr1_4                  ΔR between particle 4 and the 2nd-hardest particle
  Q.dr1_5                  ΔR between particle 5 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_2                  Δη of particle 2
  Q.eta_3                  Δη of particle 3
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
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.centroid_offset        distance of the pT centroid from the jet axis
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
        sj3_pairmax_over_m=max(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_4=pair_mass(0, 4),
        pair_mass_0_5=pair_mass(0, 5),
        pair_mass_0_6=pair_mass(0, 6),
        pair_mass_0_7=pair_mass(0, 7),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass3=subjets(3)["mass"][2],
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_90pct=ncum(0.9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        pt_2=pt[2],
        pt_3=pt[3],
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        ptdr0_7=pt[7] * math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        z_3=z[3],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        zdr_5=z[5] * dr[5],
        zdr_7=z[7] * dr[7],
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        abseta_0=abs(eta[0]),
        abseta_1=abs(eta[1]),
        abseta_2=abs(eta[2]),
        abseta_3=abs(eta[3]),
        abseta_4=abs(eta[4]),
        abseta_5=abs(eta[5]),
        abseta_6=abs(eta[6]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_2=abs(phi[2]),
        absphi_3=abs(phi[3]),
        absphi_5=abs(phi[5]),
        absphi_6=abs(phi[6]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_3=math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        dr0_5=math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr1_4=math.sqrt(dist2(1, 4)) if pt[4] > 0 else 0.0,
        dr1_5=math.sqrt(dist2(1, 5)) if pt[5] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        eta_2=eta[2],
        eta_3=eta[3],
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
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 17.83;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.8326 * (-0.08364017
        + 0.156541 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +15.7%  girth2 < 0.01324
        + 0.1074815 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +10.7%  width < 0.008678
        - 0.07238149 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -7.2%  girth < 0.08724
        - 0.06272888 * max(0.0, 0.004372139 - Q.width) / 0.00156571   # -6.3%  width < 0.004372
        + 0.05970566 * max(0.0, Q.sj3_dr_max - 0.1070199) / 0.08518534   # +6.0%  sj3_dr_max > 0.107
        + 0.05391072 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +5.4%  sj3_dr_max < 0.3012
        - 0.05137868 * max(0.0, 56.92035 - Q.mass) / 21.78475   # -5.1%  mass < 56.92
        - 0.04967416 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -5.0%  lam1 < 0.005433
        + 0.04376574 * max(0.0, 0.01882765 - Q.girth2) / 0.01314296   # +4.4%  girth2 < 0.01883
        + 0.02765514 * max(0.0, 0.0245477 - Q.e2) / 0.007455821   # +2.8%  e2 < 0.02455
        - 0.02755441 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -2.8%  sj3_dr_max > 0.2337
        + 0.02656925 * max(0.0, 0.003904593 - Q.mass_over_sum_pt_sq) / 0.001485439   # +2.7%  mass_over_sum_pt_sq < 0.003905
        + 0.01794878 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.001732318   # +1.8%  planar_flow < 0.1484 and centroid_offset < 0.0499
        + 0.01530978 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +1.5%  sum_pt_top5 > 687.4
        - 0.01529157 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # -1.5%  girth2_top5 < 0.00833
        + 0.01426121 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 1.002471 - Q.D2) / 0.001008123   # +1.4%  girth2 < 0.01324 and D2 < 1.002
        - 0.01376224 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.centroid_offset - 0.01837778) / 2.056482e-05   # -1.4%  girth2 < 0.01324 and centroid_offset > 0.01838
        - 0.01346644 * max(0.0, 64.61873 - Q.mass) / 27.57279   # -1.3%  mass < 64.62
        + 0.01259586 * max(0.0, Q.n_dr_0_0p05 - 4.0) / 1.61275   # +1.3%  n_dr_0_0p05 > 4
        + 0.01121285 * max(0.0, 29.6447 - Q.mass) / 7.34723   # +1.1%  mass < 29.64
        - 0.0109509 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -1.1%  log_sum_pt > 6.67
        + 0.01062845 * max(0.0, 0.01587232 - Q.zdr_1) / 0.007691685   # +1.1%  zdr_1 < 0.01587
        - 0.0102984 * max(0.0, 0.007929074 - Q.girth2_top3) / 0.004560262   # -1.0%  girth2_top3 < 0.007929
        + 0.009759716 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +1.0%  planar_flow < 0.1484
        - 0.009593558 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -1.0%  lam1 < 0.006507
        - 0.008413919 * max(0.0, 0.0002758826 - Q.lam1) / 3.586469e-05   # -0.8%  lam1 < 0.0002759
        + 0.008182842 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 0.07232166 - Q.dr_4) / 0.001949106   # +0.8%  log_sum_pt > 6.67 and dr_4 < 0.07232
        - 0.007012884 * max(0.0, 0.001563465 - Q.C2_b2) / 0.0008800039   # -0.7%  C2_b2 < 0.001563
        - 0.006020542 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # -0.6%  lam1 < 0.006507 and D2 < 0.8757
        + 0.00596722 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.phi_0 - -0.008995056) / 0.0001281354   # +0.6%  girth2 < 0.01324 and phi_0 > -0.008995
        - 0.005639812 * max(0.0, Q.sum_pt - 901.5938) / 12.40362   # -0.6%  sum_pt > 901.6
        + 0.005580894 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.02320757 - Q.z_7) / 2.048782e-05   # +0.6%  planar_flow < 0.1484 and z_7 < 0.02321
        - 0.00542071 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, Q.dr0_6 - 0.1755206) / 0.0008435917   # -0.5%  planar_flow < 0.1484 and dr0_6 > 0.1755
        + 0.004812738 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2) / 2.36013e-06   # +0.5%  lam2 < 7.301e-05 and D2_b2 < 0.267
        - 0.004573276 * max(0.0, 0.004372139 - Q.width) * max(0.0, Q.eccentricity - 0.9598562) / 6.567391e-06   # -0.5%  width < 0.004372 and eccentricity > 0.9599
        + 0.004072634 * max(0.0, 7.300726e-05 - Q.lam2) / 2.767163e-05   # +0.4%  lam2 < 7.301e-05
        + 0.003427524 * max(0.0, 0.01882765 - Q.girth2) * max(0.0, 46.125 - Q.pt_6) / 0.1159788   # +0.3%  girth2 < 0.01883 and pt_6 < 46.12
        - 0.002793223 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.01452637 - Q.phi_0) / 0.1160543   # -0.3%  mass < 29.64 and phi_0 < 0.01453
        - 0.002394683 * max(0.0, 6.080494 - Q.log_sum_pt) / 0.006097657   # -0.2%  log_sum_pt < 6.08
        + 0.001920742 * max(0.0, 0.004372139 - Q.width) * max(0.0, Q.pair_mass_0_6 - 20.59519) / 8.507382e-05   # +0.2%  width < 0.004372 and pair_mass_0_6 > 20.6
        - 0.001821638 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.eccentricity - 0.9458207) / 0.1898472   # -0.2%  sum_pt > 901.6 and eccentricity > 0.9458
        - 0.001780483 * max(0.0, 0.01882765 - Q.girth2) * max(0.0, Q.mass_top2 - 28.78966) / 0.006400313   # -0.2%  girth2 < 0.01883 and mass_top2 > 28.79
        - 0.001508733 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, -0.003096655 - Q.mean_phi) / 2.294381e-05   # -0.2%  girth2 < 0.01324 and mean_phi < -0.003097
        - 0.001507952 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.sj3_dr12 - 0.2182873) / 7.0923e-06   # -0.2%  girth2 < 0.01324 and sj3_dr12 > 0.2183
        - 0.001422067 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, 0.119512 - Q.dr12) / 1.24022   # -0.1%  sum_pt > 901.6 and dr12 < 0.1195
        + 0.001393464 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.dr0_6 - 0.2347949) / 0.03872508   # +0.1%  sum_pt > 901.6 and dr0_6 > 0.2348
        + 0.001346101 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 358.375 - Q.sum_pt_top2) / 2.426548   # +0.1%  planar_flow < 0.1484 and sum_pt_top2 < 358.4
        - 0.001279165 * max(0.0, 0.004372139 - Q.width) * max(0.0, 0.7459513 - Q.D2) / 7.193909e-06   # -0.1%  width < 0.004372 and D2 < 0.746
        + 0.00111993 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, Q.phi_1 - 0.01467133) / 0.0008710753   # +0.1%  sj3_dr_max < 0.3012 and phi_1 > 0.01467
        + 0.001087621 * max(0.0, 56.92035 - Q.mass) * max(0.0, Q.dr_max_012 - 0.06112084) / 0.1887976   # +0.1%  mass < 56.92 and dr_max_012 > 0.06112
        - 0.001049249 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05744392 - Q.D2_b2) / 0.0005632394   # -0.1%  sj3_dr_max < 0.3012 and D2_b2 < 0.05744
        - 0.0009139321 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, 0.03363037 - Q.absphi_3) / 0.2707601   # -0.1%  sum_pt > 901.6 and absphi_3 < 0.03363
        + 0.0008529893 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, Q.eta_6 - 0.05477905) / 4.236986e-06   # +0.1%  lam1 < 0.005433 and eta_6 > 0.05478
        - 0.000492424 * max(0.0, Q.n_dr_0_0p05 - 4.0) * max(0.0, Q.pair_mass_0_5 - 16.26599) / 0.6268949   # -0.0%  n_dr_0_0p05 > 4 and pair_mass_0_5 > 16.27
        - 0.0004341452 * max(0.0, 64.61873 - Q.mass) * max(0.0, 0.1830092 - Q.D2_b2) / 0.3382769   # -0.0%  mass < 64.62 and D2_b2 < 0.183
        - 0.0004098854 * max(0.0, 21.78408 - Q.mass) / 4.431023   # -0.0%  mass < 21.78
        - 0.0003223436 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, Q.phi_3 - 0.04800415) / 1.473914e-06   # -0.0%  lam1 < 0.005433 and phi_3 > 0.048
        + 0.0003138698 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +0.0%  centroid_offset > 0.0499
        - 0.0002808733 * max(0.0, 0.008678045 - Q.width) * max(0.0, Q.eta_7 - 0.1219513) / 5.067324e-06   # -0.0%  width < 0.008678 and eta_7 > 0.122
        + 3.020117e-06 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, Q.phi_2 - 0.09649963) / 4.090789e-05   # +0.0%  log_sum_pt > 6.67 and phi_2 > 0.0965
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 15.96;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.96228 * (0.01952112
        - 0.1502365 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -15.0%  girth2 < 0.008678
        + 0.0770961 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # +7.7%  e2_sq < 0.008169
        - 0.07300735 * max(0.0, 0.00609665 - Q.width) / 0.002544039   # -7.3%  width < 0.006097
        + 0.0681794 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +6.8%  lam1 < 0.005954
        + 0.06134871 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) / 0.002578466   # +6.1%  mass_over_sum_pt_sq < 0.005833
        + 0.04523777 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # +4.5%  z_7 > 0.02321
        - 0.04481174 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -4.5%  sum_pt_top5 > 531.2
        + 0.04174069 * max(0.0, 49.6681 - Q.mass) / 17.06893   # +4.2%  mass < 49.67
        - 0.03868926 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1) / 0.1196218   # -3.9%  log_sum_pt > 6.378 and z_dr_0p05_0p1 < 0.7509
        - 0.03182676 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 1.627072e-05 - Q.e3) / 1.912338e-06   # -3.2%  log_sum_pt > 6.378 and e3 < 1.627e-05
        - 0.02989501 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -3.0%  z_7 < 0.06165
        + 0.0253544 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +2.5%  log_sum_pt > 6.378
        + 0.02393493 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # +2.4%  girth < 0.0717
        + 0.02292935 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +2.3%  log_sum_pt > 6.503
        + 0.01886204 * max(0.0, Q.log_sum_pt - 6.605974) / 0.07073019   # +1.9%  log_sum_pt > 6.606
        + 0.0169869 * max(0.0, 1.627072e-05 - Q.e3) / 6.577181e-06   # +1.7%  e3 < 1.627e-05
        + 0.01633652 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.centroid_offset - 0.009480685) / 0.0009010889   # +1.6%  log_sum_pt > 6.378 and centroid_offset > 0.009481
        + 0.01462787 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +1.5%  pt_7 > 34.53
        + 0.01407232 * max(0.0, Q.sum_pt_top5 - 367.5938) / 230.8403   # +1.4%  sum_pt_top5 > 367.6
        + 0.01322541 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +1.3%  lam1 < 0.008376
        + 0.0117375 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 1.432482 - Q.D2) / 0.01970563   # +1.2%  log_sum_pt > 6.606 and D2 < 1.432
        - 0.01090378 * max(0.0, 0.0008722282 - Q.lam1) / 0.0001882079   # -1.1%  lam1 < 0.0008722
        - 0.01071653 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 1.679198 - Q.D2) / 0.005760154   # -1.1%  z_7 < 0.06165 and D2 < 1.679
        - 0.00986712 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 52.90625 - Q.pt_6) / 13.55761   # -1.0%  pt_7 > 34.53 and pt_6 < 52.91
        - 0.009512414 * max(0.0, Q.z_7 - 0.04624032) / 0.01204993   # -1.0%  z_7 > 0.04624
        + 0.008404445 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +0.8%  sj3_dr_max > 0.169
        - 0.007913156 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.04875823) / 0.002321449   # -0.8%  z_7 < 0.06165 and z_dr_0p05_0p1 > 0.04876
        + 0.007601166 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.1340569 - Q.sj3_pairmin_over_m) / 0.00601396   # +0.8%  log_sum_pt > 6.378 and sj3_pairmin_over_m < 0.1341
        + 0.007595007 * max(0.0, 0.008678045 - Q.girth2) * max(0.0, Q.centroid_offset - 0.02076709) / 7.418362e-06   # +0.8%  girth2 < 0.008678 and centroid_offset > 0.02077
        + 0.007439621 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 0.09925997 - Q.z_4) / 0.00251516   # +0.7%  log_sum_pt > 6.606 and z_4 < 0.09926
        - 0.007116545 * max(0.0, 1.0 - Q.n_dr_0_0p05) / 0.2859429   # -0.7%  n_dr_0_0p05 < 1
        - 0.007007845 * max(0.0, Q.sj3_dr_min - 0.03628191) / 0.02376249   # -0.7%  sj3_dr_min > 0.03628
        + 0.0068956 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.sj2_dr - 0.1294903) / 0.1877842   # +0.7%  pt_7 > 34.53 and sj2_dr > 0.1295
        + 0.005741493 * max(0.0, 0.006726135 - Q.tau21_b2) / 0.0004063558   # +0.6%  tau21_b2 < 0.006726
        + 0.005404408 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, Q.zdr_4 - 0.002832174) / 8.802138e-05   # +0.5%  log_sum_pt > 6.606 and zdr_4 > 0.002832
        - 0.004777367 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.5700307   # -0.5%  sj3_dr_max > 0.169 and sj3_pair_mass_min > 5.744
        + 0.004621949 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.06297984 - Q.tau2) / 0.0007776805   # +0.5%  z_7 < 0.06165 and tau2 < 0.06298
        + 0.00460281 * max(0.0, 1.0 - Q.n_dr_0_0p05) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.1952756   # +0.5%  n_dr_0_0p05 < 1 and n_pt_above_50 > 5
        - 0.004122574 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.z_dr_0p05_0p1 - 0.2919447) / 0.7430303   # -0.4%  pt_7 > 34.53 and z_dr_0p05_0p1 > 0.2919
        + 0.003665951 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # +0.4%  pt_7 > 34.53 and tau2 < 0.01713
        - 0.003300827 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236) / 1.769648e-05   # -0.3%  mass_over_sum_pt_sq < 0.005833 and centroid_offset > 0.008092
        - 0.00293024 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.pair_mass_0_4 - 13.64049) / 0.0375734   # -0.3%  z_7 < 0.06165 and pair_mass_0_4 > 13.64
        - 0.002832516 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.008921136 - Q.mean_phi2) / 0.0001024498   # -0.3%  z_7 < 0.06165 and mean_phi2 < 0.008921
        + 0.002770094 * max(0.0, 49.6681 - Q.mass) * max(0.0, Q.mass_top2 - 0.6957161) / 24.04133   # +0.3%  mass < 49.67 and mass_top2 > 0.6957
        + 0.002516296 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # +0.3%  lam1 < 0.008376 and centroid_offset > 0.02077
        - 0.002439718 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, 0.03885671 - Q.D2_b2) / 4.310709e-07   # -0.2%  mass_over_sum_pt_sq < 0.005833 and D2_b2 < 0.03886
        - 0.002034035 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 3.916009 - Q.D3) / 0.03613284   # -0.2%  z_7 < 0.06165 and D3 < 3.916
        - 0.00194594 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.00323438   # -0.2%  lam1 < 0.008376 and n_pt_above_50 > 5
        - 0.001837153 * max(0.0, 1.0 - Q.n_dr_0_0p05) * max(0.0, Q.zdr_7 - 0.0007928864) / 0.001861106   # -0.2%  n_dr_0_0p05 < 1 and zdr_7 > 0.0007929
        - 0.0008989586 * max(0.0, Q.z_7 - 0.08670959) / 0.000332184   # -0.1%  z_7 > 0.08671
        - 0.0008495837 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.dr1_7 - 0.1343804) / 0.0003460886   # -0.1%  z_7 < 0.06165 and dr1_7 > 0.1344
        - 0.000413536 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.centroid_offset - 0.02076709) / 2.871254e-05   # -0.0%  z_7 < 0.06165 and centroid_offset > 0.02077
        + 0.0003573252 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1484197 - Q.planar_flow) / 0.0001369594   # +0.0%  lam1 < 0.008376 and planar_flow < 0.1484
        + 0.0003123656 * max(0.0, 49.6681 - Q.mass) * max(0.0, Q.mean_phi - 0.002834884) / 0.05152091   # +0.0%  mass < 49.67 and mean_phi > 0.002835
        - 0.0002276194 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.0%  pt_7 > 53.44
        - 0.0001602794 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 3.175597   # -0.0%  pt_7 > 34.53 and n_dr_0p1_0p2 > 1
        + 5.978073e-05 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, Q.phi_0 - 0.07977295) / 1.342031e-05   # +0.0%  log_sum_pt > 6.606 and phi_0 > 0.07977
        + 3.175732e-05 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, Q.dr01 - 0.1894571) / 0.0009794156   # +0.0%  pt_7 > 53.44 and dr01 > 0.1895
        + 1.994591e-05 * max(0.0, 0.006726135 - Q.tau21_b2) * max(0.0, Q.sj3_pair_mass_min - 6.811308) / 2.612788e-07   # +0.0%  tau21_b2 < 0.006726 and sj3_pair_mass_min > 6.811
        - 1.569615e-05 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 2.955458e-05 - Q.e3) / 7.137805e-05   # -0.0%  pt_7 > 34.53 and e3 < 2.955e-05
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 16.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.78184 * (0.1028705
        - 0.1266767 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -12.7%  pt_7 < 53.44
        + 0.08151696 * max(0.0, 813.4156 - Q.sum_pt) / 129.0758   # +8.2%  sum_pt < 813.4
        - 0.07866512 * max(0.0, Q.LHA - 0.1329373) / 0.1175814   # -7.9%  LHA > 0.1329
        + 0.04416249 * max(0.0, 62.55 - Q.sj3_pair_mass_max) / 30.55764   # +4.4%  sj3_pair_mass_max < 62.55
        - 0.03838736 * max(0.0, Q.z_7 - 0.03629544) / 0.01879829   # -3.8%  z_7 > 0.0363
        + 0.03802435 * max(0.0, 0.2507612 - Q.max_dr) / 0.1316542   # +3.8%  max_dr < 0.2508
        + 0.03704222 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # +3.7%  log_sum_pt < 6.606
        + 0.03508604 * max(0.0, 0.1079857 - Q.mass_over_sum_pt) / 0.05207765   # +3.5%  mass_over_sum_pt < 0.108
        - 0.03059836 * max(0.0, Q.sum_pt - 527.1781) * max(0.0, 0.0001947983 - Q.lam2) / 0.02863163   # -3.1%  sum_pt > 527.2 and lam2 < 0.0001948
        - 0.03043448 * max(0.0, 62.55 - Q.sj3_pair_mass_max) * max(0.0, 0.06810151 - Q.z_7) / 0.6094489   # -3.0%  sj3_pair_mass_max < 62.55 and z_7 < 0.0681
        + 0.02780177 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # +2.8%  girth2 < 0.003563
        - 0.02692492 * max(0.0, 36.22941 - Q.mass) / 10.11393   # -2.7%  mass < 36.23
        + 0.02635038 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +2.6%  lam1 < 0.005954
        + 0.02580566 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # +2.6%  zdr_0 < 0.02118
        - 0.02330271 * max(0.0, 0.4926918 - Q.planar_flow) / 0.2728654   # -2.3%  planar_flow < 0.4927
        + 0.02265243 * max(0.0, Q.sum_pt - 527.1781) / 199.4719   # +2.3%  sum_pt > 527.2
        + 0.02048446 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.pt_6 - 19.46875) / 0.05232739   # +2.0%  lam1 < 0.005954 and pt_6 > 19.47
        - 0.02005562 * max(0.0, 53.4375 - Q.pt_7) * max(0.0, 1.129616 - Q.D2_b2) / 9.433694   # -2.0%  pt_7 < 53.44 and D2_b2 < 1.13
        - 0.01877853 * max(0.0, Q.sum_pt - 527.1781) * max(0.0, 0.05648804 - Q.absphi_3) / 6.676969   # -1.9%  sum_pt > 527.2 and absphi_3 < 0.05649
        + 0.01781968 * max(0.0, 36.22941 - Q.mass) * max(0.0, 73.75 - Q.pt_5) / 259.9887   # +1.8%  mass < 36.23 and pt_5 < 73.75
        + 0.01773618 * max(0.0, Q.N2 - 0.1150852) / 0.1151384   # +1.8%  N2 > 0.1151
        + 0.01531929 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # +1.5%  log_sum_pt > 6.843
        + 0.01514902 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # +1.5%  log_sum_pt < 6.464
        - 0.01315207 * max(0.0, 0.004430536 - Q.centroid_offset) / 0.0004182362   # -1.3%  centroid_offset < 0.004431
        - 0.01284541 * max(0.0, Q.z_7 - 0.04939969) * max(0.0, 0.9206502 - Q.D2_b2) / 0.004461719   # -1.3%  z_7 > 0.0494 and D2_b2 < 0.9207
        + 0.01181844 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +1.2%  pt_7 > 34.53
        + 0.01113692 * max(0.0, 6.46415 - Q.log_sum_pt) * max(0.0, 1.129616 - Q.D2_b2) / 0.03658616   # +1.1%  log_sum_pt < 6.464 and D2_b2 < 1.13
        - 0.01102071 * max(0.0, 62.55 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.01096064) / 0.2139385   # -1.1%  sj3_pair_mass_max < 62.55 and centroid_offset > 0.01096
        - 0.01027382 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # -1.0%  girth < 0.007674
        + 0.009131291 * max(0.0, 3.55957 - Q.m012) / 0.8681902   # +0.9%  m012 < 3.56
        + 0.008341765 * max(0.0, Q.z_7 - 0.04939969) / 0.01023002   # +0.8%  z_7 > 0.0494
        - 0.008129964 * max(0.0, 0.03243272 - Q.z_7) / 0.002061024   # -0.8%  z_7 < 0.03243
        + 0.00796621 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 1.129616 - Q.D2_b2) / 6.295733   # +0.8%  sum_pt_top5 > 752.1 and D2_b2 < 1.13
        + 0.007183486 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.0174408 - Q.abseta_0) / 8.283244e-05   # +0.7%  log_sum_pt > 6.843 and abseta_0 < 0.01744
        + 0.007083458 * max(0.0, Q.pt_7 - 43.5) / 1.4368   # +0.7%  pt_7 > 43.5
        + 0.006673988 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.4474937 - Q.z_dr_0p05_0p1) / 0.00315435   # +0.7%  log_sum_pt > 6.843 and z_dr_0p05_0p1 < 0.4475
        + 0.006507186 * max(0.0, Q.m012 - 40.2) / 1.307209   # +0.7%  m012 > 40.2
        - 0.005665088 * max(0.0, 0.007673833 - Q.girth) * max(0.0, 3.885568 - Q.D2) / 0.0003532425   # -0.6%  girth < 0.007674 and D2 < 3.886
        - 0.005167414 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.max_dr - 0.08050702) / 4.493327e-05   # -0.5%  lam1 < 0.005954 and max_dr > 0.08051
        + 0.004559173 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 0.03623337 - Q.max_dr) / 9.868306e-05   # +0.5%  log_sum_pt < 6.606 and max_dr < 0.03623
        - 0.004144167 * max(0.0, 53.4375 - Q.pt_7) * max(0.0, Q.abseta_0 - 0.02940369) / 0.2565312   # -0.4%  pt_7 < 53.44 and abseta_0 > 0.0294
        - 0.003780262 * max(0.0, 0.2507612 - Q.max_dr) * max(0.0, Q.phi_0 - 0.002076054) / 0.001439957   # -0.4%  max_dr < 0.2508 and phi_0 > 0.002076
        - 0.003418817 * max(0.0, 3.849518e-05 - Q.girth2_top2) / 2.100982e-06   # -0.3%  girth2_top2 < 3.85e-05
        - 0.003334613 * max(0.0, 0.03243272 - Q.z_7) * max(0.0, Q.mass_top3 - 16.89912) / 0.01064848   # -0.3%  z_7 < 0.03243 and mass_top3 > 16.9
        - 0.003036232 * max(0.0, 0.007673833 - Q.girth) * max(0.0, 75.625 - Q.pt_4) / 0.004511241   # -0.3%  girth < 0.007674 and pt_4 < 75.62
        - 0.002788535 * max(0.0, 0.03243272 - Q.z_7) * max(0.0, 0.9206502 - Q.D2_b2) / 0.0004360983   # -0.3%  z_7 < 0.03243 and D2_b2 < 0.9207
        + 0.002034365 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # +0.2%  sum_pt_top5 > 840
        - 0.001928938 * max(0.0, 0.03243272 - Q.z_7) * max(0.0, 8.0 - Q.n_pt_above_5) / 0.0001713068   # -0.2%  z_7 < 0.03243 and n_pt_above_5 < 8
        + 0.001922486 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.pt_6 - 41.21875) / 0.06090836   # +0.2%  log_sum_pt > 6.843 and pt_6 > 41.22
        + 0.001867396 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +0.2%  z_7 < 0.07149
        + 0.001603295 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, Q.eta_0 - 0.009292603) / 0.06320835   # +0.2%  sum_pt_top5 > 752.1 and eta_0 > 0.009293
        + 0.00139836 * max(0.0, Q.z_7 - 0.04939969) * max(0.0, 0.0623951 - Q.sj3_dr_min) / 0.0002978985   # +0.1%  z_7 > 0.0494 and sj3_dr_min < 0.0624
        + 0.0009675575 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 0.0531335 - Q.z_5) / 2.719561e-05   # +0.1%  log_sum_pt < 6.606 and z_5 < 0.05313
        + 0.0006696797 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 1.345805 - Q.D2_b2) / 0.002918988   # +0.1%  log_sum_pt > 6.843 and D2_b2 < 1.346
        - 0.0004453247 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.D3 - 0.4387703) / 4.820328   # -0.0%  pt_7 > 34.53 and D3 > 0.4388
        - 0.0004238514 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, -0.0216713 - Q.phi_0) / 1.182282e-05   # -0.0%  log_sum_pt > 6.843 and phi_0 < -0.02167
        - 0.0003767068 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, -0.01284493 - Q.mean_eta) / 7.701745e-06   # -0.0%  zdr_0 < 0.02118 and mean_eta < -0.01284
        - 0.0002055789 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 0.0174408 - Q.abseta_0) / 0.216283   # -0.0%  sum_pt_top5 > 752.1 and abseta_0 < 0.01744
        - 0.000126001 * max(0.0, Q.sum_pt_top5 - 752.1) / 21.27415   # -0.0%  sum_pt_top5 > 752.1
        + 9.670961e-05 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.eta_0 - 0.02130127) / 1.178041e-05   # +0.0%  log_sum_pt > 6.843 and eta_0 > 0.0213
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 15.91;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.9139 * (-0.08506471
        - 0.185682 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # -18.6%  girth2 > 0.008678
        + 0.1293353 * max(0.0, Q.width - 0.007520088) / 0.002470136   # +12.9%  width > 0.00752
        + 0.09189877 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +9.2%  lam1 > 0.008376
        - 0.05770181 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # -5.8%  girth2_top5 < 0.007164
        + 0.05537108 * max(0.0, Q.mass_over_sum_pt - 0.0681391) / 0.01539157   # +5.5%  mass_over_sum_pt > 0.06814
        + 0.05343639 * max(0.0, Q.girth - 0.04081947) / 0.0270611   # +5.3%  girth > 0.04082
        + 0.0469133 * max(0.0, Q.sj2_dr - 0.1872617) / 0.02732165   # +4.7%  sj2_dr > 0.1873
        - 0.04509382 * max(0.0, Q.width - 0.01323868) / 0.001442684   # -4.5%  width > 0.01324
        - 0.03843078 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # -3.8%  tau1 > 0.05357
        + 0.03451843 * max(0.0, Q.max_dr - 0.1027585) / 0.04505724   # +3.5%  max_dr > 0.1028
        - 0.02207109 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.02673292   # -2.2%  max_dr > 0.1028 and z_dr_0p05_0p1 < 0.846
        + 0.01948394 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # +1.9%  girth > 0.08724
        - 0.01797798 * max(0.0, Q.sj2_dr - 0.1492731) / 0.04449993   # -1.8%  sj2_dr > 0.1493
        + 0.01660831 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 0.07861328 - Q.abseta_0) / 0.0002969212   # +1.7%  centroid_offset > 0.01096 and abseta_0 < 0.07861
        - 0.01635117 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -1.6%  lam1 > 0.012
        + 0.01448099 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # +1.4%  tau1 > 0.1028
        + 0.0122638 * max(0.0, Q.girth - 0.07608178) / 0.01059033   # +1.2%  girth > 0.07608
        + 0.0118263 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50) / 0.02994202   # +1.2%  centroid_offset > 0.01096 and n_pt_above_50 < 8
        - 0.01075354 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr) / 0.0001196243   # -1.1%  mass_over_sum_pt > 0.06814 and sj2_dr < 0.218
        + 0.008528588 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, Q.phi_7 - -0.05731659) / 0.0006418095   # +0.9%  centroid_offset > 0.01096 and phi_7 > -0.05732
        + 0.008354392 * max(0.0, -0.004664942 - Q.mean_eta) / 0.003754527   # +0.8%  mean_eta < -0.004665
        + 0.008110256 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.02100911   # +0.8%  sj2_dr > 0.1873 and n_dr_0p2_0p4 < 2
        - 0.0072312 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -0.7%  e2 > 0.06344
        - 0.006687294 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.1950293   # -0.7%  mass_over_sum_pt > 0.06814 and sj3_pair_mass_min > 5.744
        + 0.006670158 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, Q.pair_mass_0_5 - 16.26599) / 0.01853759   # +0.7%  centroid_offset > 0.01096 and pair_mass_0_5 > 16.27
        - 0.005957237 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, 50.3522 - Q.mass_top3) / 0.5259724   # -0.6%  tau1 > 0.05357 and mass_top3 < 50.35
        + 0.005635366 * max(0.0, Q.mean_eta - 0.01772426) / 0.001375178   # +0.6%  mean_eta > 0.01772
        + 0.005518324 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.6%  lam2 > 0.001131
        + 0.00518164 * max(0.0, Q.sd_mass - 62.55) / 3.348271   # +0.5%  sd_mass > 62.55
        + 0.004934783 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.5%  sj2_dr > 0.2688
        - 0.0044793 * max(0.0, 0.004372139 - Q.girth2) / 0.00156571   # -0.4%  girth2 < 0.004372
        + 0.004420015 * max(0.0, Q.girth - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494) / 0.008781747   # +0.4%  girth > 0.04082 and log_sum_pt > 6.08
        + 0.003907891 * max(0.0, Q.sd_mass - 62.55) * max(0.0, 0.9206502 - Q.D2_b2) / 1.823781   # +0.4%  sd_mass > 62.55 and D2_b2 < 0.9207
        + 0.003499813 * max(0.0, Q.mass - 64.61873) * max(0.0, Q.eta_7 - -0.0803833) / 0.3229902   # +0.3%  mass > 64.62 and eta_7 > -0.08038
        - 0.003360919 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.05268713 - Q.dr_3) / 0.000155932   # -0.3%  sj2_dr > 0.1873 and dr_3 < 0.05269
        + 0.002868785 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +0.3%  girth2_top5 < 0.00833
        + 0.00262767 * max(0.0, Q.girth2 - 0.01882765) / 0.0007605369   # +0.3%  girth2 > 0.01883
        - 0.002519554 * max(0.0, Q.e2 - 0.06344108) * max(0.0, 0.1332952 - Q.dr0_3) / 6.054313e-05   # -0.3%  e2 > 0.06344 and dr0_3 < 0.1333
        - 0.002347058 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # -0.2%  lam1 > 0.01643
        - 0.002149547 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113) / 0.39507   # -0.2%  sj2_dr > 0.1873 and sj2_mass1 > 2.25
        + 0.001937203 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, Q.pt_5 - 59.125) / 0.0351992   # +0.2%  tau1 > 0.05357 and pt_5 > 59.12
        + 0.001663741 * max(0.0, -0.004664942 - Q.mean_eta) * max(0.0, Q.phi_5 - 0.07385864) / 2.965175e-05   # +0.2%  mean_eta < -0.004665 and phi_5 > 0.07386
        - 0.001273884 * max(0.0, Q.mass - 64.61873) / 3.274818   # -0.1%  mass > 64.62
        + 0.001256795 * max(0.0, Q.mass - 64.61873) * max(0.0, Q.phi_6 - 0.1192017) / 0.03430397   # +0.1%  mass > 64.62 and phi_6 > 0.1192
        + 0.00124948 * max(0.0, Q.centroid_offset - 0.01096064) / 0.008813008   # +0.1%  centroid_offset > 0.01096
        - 0.0009943911 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.z_top5 - 0.7773372) / 2.699283e-05   # -0.1%  e2 > 0.06344 and z_top5 > 0.7773
        + 0.0009904729 * max(0.0, Q.mean_eta - 0.01772426) * max(0.0, 0.02612796 - Q.mean_phi) / 3.988786e-05   # +0.1%  mean_eta > 0.01772 and mean_phi < 0.02613
        - 0.0008912041 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 27.57812 - Q.pt_6) / 0.028355   # -0.1%  sj2_dr > 0.1873 and pt_6 < 27.58
        + 0.0007107402 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, Q.phi_4 - 0.1096802) / 0.0003590336   # +0.1%  max_dr > 0.1028 and phi_4 > 0.1097
        - 0.0006186368 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.pt_6 - 31.90625) / 0.1358885   # -0.1%  mass_over_sum_pt > 0.06814 and pt_6 > 31.91
        + 0.0006142412 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6) / 0.005634141   # +0.1%  lam2 > 0.001131 and pt_6 < 56.53
        - 0.0006042416 * max(0.0, Q.girth - 0.07608178) * max(0.0, Q.eta_0 - 0.07952881) / 0.0001300296   # -0.1%  girth > 0.07608 and eta_0 > 0.07953
        - 0.0005126906 * max(0.0, Q.lam1 - 0.008375572) * max(0.0, 24.42188 - Q.pt_6) / 0.0001598819   # -0.1%  lam1 > 0.008376 and pt_6 < 24.42
        + 0.0004356286 * max(0.0, Q.lam1 - 0.01643375) * max(0.0, 6.0 - Q.n_for_90pct) / 5.140257e-06   # +0.0%  lam1 > 0.01643 and n_for_90pct < 6
        - 0.0003553118 * max(0.0, Q.width - 0.007520088) * max(0.0, Q.sj3_pairmin_over_m - 0.07708997) / 0.0004577253   # -0.0%  width > 0.00752 and sj3_pairmin_over_m > 0.07709
        + 0.0002205066 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.06081235 - Q.z_6) / 0.0004655277   # +0.0%  max_dr > 0.1028 and z_6 < 0.06081
        + 0.0002203019 * max(0.0, -0.004664942 - Q.mean_eta) * max(0.0, Q.z_6 - 0.06727211) / 3.056899e-05   # +0.0%  mean_eta < -0.004665 and z_6 > 0.06727
        + 0.0001441748 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, Q.z_6 - 0.05441452) / 6.527736e-06   # +0.0%  lam2 > 0.001131 and z_6 > 0.05441
        + 8.214974e-05 * max(0.0, Q.e2 - 0.06344108) * max(0.0, 0.01257746 - Q.zdr_3) / 1.576515e-06   # +0.0%  e2 > 0.06344 and zdr_3 < 0.01258
        + 3.571476e-05 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sj2_mass1 - 16.86126) / 0.02534836   # +0.0%  e2 > 0.06344 and sj2_mass1 > 16.86
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 71.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 71.63846 * (-0.01298341
        + 0.3154286 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # +31.5%  e2_sq < 0.01166
        - 0.2760691 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -27.6%  mass_over_sum_pt_sq < 0.01166
        + 0.03837191 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # +3.8%  sj3_dr_max < 0.3456
        - 0.03717607 * max(0.0, 0.233678 - Q.sj3_dr_max) / 0.09057682   # -3.7%  sj3_dr_max < 0.2337
        + 0.03092031 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +3.1%  N2 < 0.2233
        - 0.03029976 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -3.0%  girth2 < 0.008678
        - 0.02930992 * max(0.0, 0.05028464 - Q.e2) / 0.02502152   # -2.9%  e2 < 0.05028
        + 0.02792742 * max(0.0, 0.06729223 - Q.C2) / 0.04162086   # +2.8%  C2 < 0.06729
        - 0.01930792 * max(0.0, 0.04990367 - Q.centroid_offset) / 0.03352573   # -1.9%  centroid_offset < 0.0499
        - 0.01876808 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266) / 0.01381411   # -1.9%  N2 < 0.2233 and eccentricity > 0.7117
        + 0.01818117 * max(0.0, Q.width - 0.003562611) / 0.004066872   # +1.8%  width > 0.003563
        + 0.01467632 * max(0.0, 0.007639643 - Q.girth2_top2) / 0.004543329   # +1.5%  girth2_top2 < 0.00764
        - 0.01394403 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # -1.4%  max_dr < 0.1773
        - 0.01297911 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -1.3%  mass_over_sum_pt > 0.09041
        + 0.01266811 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +1.3%  e3 < 0.0001869
        - 0.01214027 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -1.2%  lam2 < 0.0005373
        + 0.01156177 * max(0.0, Q.sd_mass - 44.82259) / 8.759065   # +1.2%  sd_mass > 44.82
        + 0.01010307 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +1.0%  tau1 < 0.07284
        + 0.008893437 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +0.9%  sj3_dr_max < 0.2134
        - 0.006882276 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.1410336 - Q.dr01) / 4.252003e-05   # -0.7%  lam2 < 0.0005373 and dr01 < 0.141
        + 0.005710518 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +0.6%  max_dr < 0.1118
        + 0.004135159 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # +0.4%  girth2_top5 < 0.007164
        + 0.004060513 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +0.4%  e2 > 0.04447
        + 0.004026392 * max(0.0, Q.max_dr - 0.1598486) / 0.0215652   # +0.4%  max_dr > 0.1598
        - 0.003236011 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.centroid_offset - 0.001308549) / 0.1760217   # -0.3%  sd_mass > 44.82 and centroid_offset > 0.001309
        + 0.002988644 * max(0.0, 0.3467135 - Q.LHA) / 0.1128474   # +0.3%  LHA < 0.3467
        - 0.002945047 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7) / 0.8992052   # -0.3%  N2 < 0.2233 and pt_7 < 53.44
        + 0.002205694 * max(0.0, 0.001653836 - Q.width) / 0.0004289067   # +0.2%  width < 0.001654
        + 0.002157991 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, Q.dr1_5 - 0.1518778) / 0.000997393   # +0.2%  sj3_dr_max < 0.3456 and dr1_5 > 0.1519
        - 0.0020892 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.55 - Q.mass) / 0.3959152   # -0.2%  N2 < 0.2233 and mass < 62.55
        - 0.001762982 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.pt_2 - 69.875) / 1.400496   # -0.2%  N2 < 0.2233 and pt_2 > 69.88
        + 0.001742534 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, Q.pair_mass_0_6 - 15.55293) / 0.07779524   # +0.2%  sj3_dr_max < 0.3456 and pair_mass_0_6 > 15.55
        - 0.001486126 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.1%  mass > 76.66
        + 0.001403584 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, -0.009352575 - Q.mean_phi) / 2.725001e-07   # +0.1%  e3 < 0.0001869 and mean_phi < -0.009353
        - 0.001385525 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -0.1%  girth > 0.08724
        - 0.001361991 * max(0.0, Q.sd_mass - 74.57663) / 1.60503   # -0.1%  sd_mass > 74.58
        + 0.00121664 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # +0.1%  sum_pt < 739.5
        + 0.001154847 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 169.875 - Q.pt_1) / 2.311819   # +0.1%  N2 < 0.2233 and pt_1 < 169.9
        - 0.00113143 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1218872 - Q.abseta_7) / 0.003384012   # -0.1%  N2 < 0.2233 and abseta_7 < 0.1219
        + 0.001076147 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, -0.01284493 - Q.mean_eta) / 2.067889e-07   # +0.1%  e3 < 0.0001869 and mean_eta < -0.01284
        - 0.0008046519 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.e2_sq - 0.01165737) / 7.11886e-05   # -0.1%  N2 < 0.2233 and e2_sq > 0.01166
        + 0.0007422191 * max(0.0, 0.177305 - Q.max_dr) * max(0.0, Q.ptdr0_4 - 8.939062) / 0.01768503   # +0.1%  max_dr < 0.1773 and ptdr0_4 > 8.939
        + 0.0007361999 * max(0.0, 0.233678 - Q.sj3_dr_max) * max(0.0, Q.dr0_3 - 0.1815989) / 2.350264e-05   # +0.1%  sj3_dr_max < 0.2337 and dr0_3 > 0.1816
        + 0.0005954826 * max(0.0, 0.2233283 - Q.N2) * max(0.0, -0.009352575 - Q.mean_phi) / 0.0001205535   # +0.1%  N2 < 0.2233 and mean_phi < -0.009353
        - 0.0005628368 * max(0.0, Q.max_dr - 0.1598486) * max(0.0, 0.0006435798 - Q.C2_b2) / 1.059262e-06   # -0.1%  max_dr > 0.1598 and C2_b2 < 0.0006436
        + 0.000511096 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.005440034 - Q.zdr_5) / 0.005279271   # +0.1%  sd_mass > 44.82 and zdr_5 < 0.00544
        - 0.0005035519 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.275762 - Q.sd_zg) / 0.3498594   # -0.1%  sd_mass > 44.82 and sd_zg < 0.2758
        - 0.0003978258 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, -0.01275329 - Q.mean_phi) / 4.253124e-07   # -0.0%  lam2 < 0.0005373 and mean_phi < -0.01275
        - 0.0003652263 * max(0.0, 37.76455 - Q.mass_top5) / 16.71307   # -0.0%  mass_top5 < 37.76
        + 0.0003604505 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, Q.absphi_0 - 0.1056549) / 4.162515e-05   # +0.0%  sj3_dr_max < 0.3456 and absphi_0 > 0.1057
        + 0.0003445301 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, -0.1135254 - Q.eta_5) / 9.71026e-05   # +0.0%  sj3_dr_max < 0.3456 and eta_5 < -0.1135
        + 0.0003212977 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, Q.phi_0 - 0.004838562) / 4.167552e-06   # +0.0%  lam2 < 0.0005373 and phi_0 > 0.004839
        + 0.0002897032 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.0006435798) / 6.028305e-06   # +0.0%  girth2_top2 < 0.00764 and C2_b2 > 0.0006436
        + 0.0001594444 * max(0.0, 0.01165737 - Q.e2_sq) * max(0.0, Q.ptdr0_7 - 2.691363) / 0.005064196   # +0.0%  e2_sq < 0.01166 and ptdr0_7 > 2.691
        - 0.00015294 * max(0.0, Q.width - 0.003562611) * max(0.0, 0.1057566 - Q.sj3_z3) / 5.535701e-05   # -0.0%  width > 0.003563 and sj3_z3 < 0.1058
        + 0.0001457402 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 24.42188 - Q.pt_6) / 0.01371831   # +0.0%  N2 < 0.2233 and pt_6 < 24.42
        - 7.415372e-05 * max(0.0, Q.mass - 76.6557) * max(0.0, Q.sj2_zsoft - 0.4448372) / 0.006058409   # -0.0%  mass > 76.66 and sj2_zsoft > 0.4448
        - 3.953913e-05 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.004918231 - Q.zdr_0) / 6.083288e-07   # -0.0%  N2 < 0.2233 and zdr_0 < 0.004918
        + 5.061464e-06 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.phi_6 - 0.1192017) / 0.0001340425   # +0.0%  N2 < 0.2233 and phi_6 > 0.1192
        + 2.388731e-06 * max(0.0, 0.213399 - Q.sj3_dr_max) * max(0.0, Q.mean_phi2 - 0.008921136) / 8.517119e-08   # +0.0%  sj3_dr_max < 0.2134 and mean_phi2 > 0.008921
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.99;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.99259 * (-0.0709682
        + 0.07302186 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset) / 7.0272e-06   # +7.3%  girth2 < 0.001654 and centroid_offset < 0.02355
        - 0.05891039 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -5.9%  zdr_0 < 0.02118
        + 0.04694429 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +4.7%  LHA < 0.2161
        + 0.04343573 * max(0.0, 0.003562611 - Q.width) / 0.001184249   # +4.3%  width < 0.003563
        + 0.04094384 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, 0.01437952 - Q.centroid_offset) / 1.318015e-05   # +4.1%  e2_sq < 0.005285 and centroid_offset < 0.01438
        + 0.04021033 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +4.0%  z_7 < 0.07149
        + 0.03596268 * max(0.0, 0.04939969 - Q.z_7) / 0.007457505   # +3.6%  z_7 < 0.0494
        + 0.0355651 * max(0.0, 0.001653836 - Q.girth2) / 0.0004289067   # +3.6%  girth2 < 0.001654
        + 0.03423136 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 46.125 - Q.pt_6) / 2161.354   # +3.4%  sum_pt_top5 > 430.8 and pt_6 < 46.12
        - 0.03330204 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -3.3%  log_sum_pt > 6.701
        + 0.03323567 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 28.39396 - Q.pt1_dr01) / 4201.911   # +3.3%  sum_pt_top5 > 430.8 and pt1_dr01 < 28.39
        + 0.03263805 * max(0.0, Q.sum_pt_top5 - 430.75) / 176.7518   # +3.3%  sum_pt_top5 > 430.8
        - 0.03059141 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -3.1%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.02853618 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 40.2 - Q.mass_top3) / 0.60086   # +2.9%  z_7 < 0.07149 and mass_top3 < 40.2
        + 0.02421624 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.002127561 - Q.mean_phi2) / 2.516806e-05   # +2.4%  z_7 < 0.07149 and mean_phi2 < 0.002128
        + 0.02388677 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5) / 0.3059459   # +2.4%  z_7 < 0.0494 and mass_top5 < 62.55
        + 0.02210755 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +2.2%  z_7 < 0.02807
        + 0.02209122 * max(0.0, 0.006789738 - Q.centroid_offset) / 0.001012351   # +2.2%  centroid_offset < 0.00679
        + 0.02107435 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03532852 - Q.C3) / 0.0003034905   # +2.1%  z_7 < 0.07149 and C3 < 0.03533
        - 0.02081928 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # -2.1%  girth2_top3 < 0.002152
        + 0.01972108 * max(0.0, 0.005284669 - Q.e2_sq) / 0.00223735   # +2.0%  e2_sq < 0.005285
        + 0.0189101 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0004149607   # +1.9%  z_7 < 0.07149 and centroid_offset < 0.03117
        + 0.01691159 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, 0.3220738 - Q.planar_flow) / 0.0001906084   # +1.7%  e2_sq < 0.005285 and planar_flow < 0.3221
        - 0.01538077 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.001503553 - Q.lam1) / 4.027805e-05   # -1.5%  LHA < 0.2161 and lam1 < 0.001504
        + 0.01434557 * max(0.0, 0.02886576 - Q.z_6) / 0.0008381782   # +1.4%  z_6 < 0.02887
        + 0.01427842 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01341502 - Q.dr_2) / 0.0001347035   # +1.4%  log_sum_pt > 6.701 and dr_2 < 0.01342
        - 0.01400639 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 501.625 - Q.sum_pt_top2) / 0.2299658   # -1.4%  z_7 < 0.0494 and sum_pt_top2 < 501.6
        + 0.01374823 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.centroid_offset - 0.009480685) / 0.7981241   # +1.4%  sum_pt_top5 > 430.8 and centroid_offset > 0.009481
        + 0.01334216 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # +1.3%  girth < 0.007674
        + 0.01314167 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +1.3%  sum_pt_top5 > 687.4
        + 0.01193549 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # +1.2%  pt_5 < 24.58
        + 0.009460772 * max(0.0, 0.006789738 - Q.centroid_offset) * max(0.0, 56.4375 - Q.pt_5) / 0.01266727   # +0.9%  centroid_offset < 0.00679 and pt_5 < 56.44
        - 0.009052566 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.sj3_dr13 - 0.04995258) / 0.000417922   # -0.9%  zdr_0 < 0.02118 and sj3_dr13 > 0.04995
        + 0.00804691 * max(0.0, 45.05938 - Q.pt_3) / 0.7565874   # +0.8%  pt_3 < 45.06
        - 0.007824029 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.8%  log_sum_pt > 6.896
        - 0.007518513 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.02270492 - Q.dr_2) / 4.344748e-05   # -0.8%  log_sum_pt > 6.896 and dr_2 < 0.0227
        - 0.00725056 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.mean_phi - 0.002834884) / 4.214377e-05   # -0.7%  log_sum_pt > 6.701 and mean_phi > 0.002835
        - 0.006999888 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0007322447   # -0.7%  e2_sq < 0.005285 and n_pt_above_50 > 6
        + 0.006973586 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.07235619 - Q.z_3) / 0.0001569963   # +0.7%  LHA < 0.2161 and z_3 < 0.07236
        + 0.006650032 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 9.030369e-05 - Q.mean_eta2) / 9.99897e-07   # +0.7%  log_sum_pt > 6.701 and mean_eta2 < 9.03e-05
        - 0.006147156 * max(0.0, Q.pair_mass_0_4 - 23.26771) / 0.6641733   # -0.6%  pair_mass_0_4 > 23.27
        - 0.006106481 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, -0.003031633 - Q.mean_eta) / 4.360831e-05   # -0.6%  log_sum_pt > 6.701 and mean_eta < -0.003032
        + 0.006021257 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 0.009661512 - Q.dr_2) / 1.13159e-05   # +0.6%  z_7 < 0.0494 and dr_2 < 0.009662
        + 0.005553227 * max(0.0, 37.15625 - Q.pt_7) * max(0.0, Q.pt_5 - 24.57812) / 73.42646   # +0.6%  pt_7 < 37.16 and pt_5 > 24.58
        - 0.005128324 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.phi_0 - -0.0297699) / 0.0002791968   # -0.5%  zdr_0 < 0.02118 and phi_0 > -0.02977
        - 0.004060605 * max(0.0, 0.002074109 - Q.e2_sq) / 0.000670556   # -0.4%  e2_sq < 0.002074
        - 0.003979321 * max(0.0, 24.57812 - Q.pt_5) * max(0.0, Q.mean_eta2 - 1.797789e-05) / 0.0002724447   # -0.4%  pt_5 < 24.58 and mean_eta2 > 1.798e-05
        - 0.003533203 * max(0.0, 37.15625 - Q.pt_7) / 5.803748   # -0.4%  pt_7 < 37.16
        + 0.003337944 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.sj2_dr - 0.1682655) / 4.310225   # +0.3%  sum_pt_top5 > 430.8 and sj2_dr > 0.1683
        - 0.003293754 * max(0.0, 0.02886576 - Q.z_6) * max(0.0, Q.mean_phi2 - 0.001125075) / 4.002571e-07   # -0.3%  z_6 < 0.02887 and mean_phi2 > 0.001125
        + 0.002931901 * max(0.0, Q.pair_mass_0_4 - 23.26771) * max(0.0, Q.eccentricity - 0.9704496) / 0.009376301   # +0.3%  pair_mass_0_4 > 23.27 and eccentricity > 0.9704
        + 0.002498594 * max(0.0, 0.003562611 - Q.width) * max(0.0, Q.sj3_dr13 - 0.181053) / 2.382944e-06   # +0.2%  width < 0.003563 and sj3_dr13 > 0.1811
        - 0.002456973 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, Q.C2_b2 - 0.002354783) / 3.68136e-05   # -0.2%  z_7 < 0.07149 and C2_b2 > 0.002355
        + 0.001941205 * max(0.0, 24.57812 - Q.pt_5) * max(0.0, 8.0 - Q.n_pt_above_5) / 0.08462184   # +0.2%  pt_5 < 24.58 and n_pt_above_5 < 8
        + 0.001905582 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, Q.dr0_5 - 0.1122797) / 8.005166e-06   # +0.2%  e2_sq < 0.005285 and dr0_5 > 0.1123
        + 0.001628695 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.05744392 - Q.D2_b2) / 1.220721e-05   # +0.2%  log_sum_pt > 6.896 and D2_b2 < 0.05744
        + 0.0009139554 * max(0.0, Q.pair_mass_0_4 - 23.26771) * max(0.0, 0.01423545 - Q.mean_eta2) / 0.004884555   # +0.1%  pair_mass_0_4 > 23.27 and mean_eta2 < 0.01424
        - 0.0006175014 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.sj3_dr_max - 0.169029) / 0.0008305926   # -0.1%  log_sum_pt > 6.701 and sj3_dr_max > 0.169
        - 0.0004940871 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, Q.mean_eta - 0.01271871) / 5.064669e-06   # -0.0%  z_7 < 0.0494 and mean_eta > 0.01272
        + 0.000227565 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 0.001534584   # +0.0%  LHA < 0.2161 and n_dr_0p2_0p4 > 0
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 23.74;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.73606 * (0.00390046
        - 0.1614141 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # -16.1%  width < 0.01324
        - 0.1088345 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -10.9%  girth2 < 0.008678
        + 0.09817671 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +9.8%  lam1 < 0.00733
        + 0.05805665 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +5.8%  lam1 < 0.012
        + 0.05411541 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +5.4%  lam2 < 0.003408
        + 0.04905914 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +4.9%  tau1 < 0.1136
        + 0.03898482 * max(0.0, 60.63098 - Q.mass) / 24.47895   # +3.9%  mass < 60.63
        + 0.03577174 * max(0.0, 0.1789613 - Q.sj3_dr_max) / 0.05394779   # +3.6%  sj3_dr_max < 0.179
        + 0.03471574 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.003289169   # +3.5%  lam1 < 0.00733 and n_dr_0p2_0p4 < 1
        - 0.03137897 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 1.341509   # -3.1%  mass < 60.63 and z_dr_0p2_0p4 < 0.05644
        - 0.02754904 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -2.8%  sj3_dr_max < 0.3012
        + 0.02658206 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt) / 1.016213   # +2.7%  pt_6 < 41.22 and log_sum_pt < 6.767
        + 0.02347453 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # +2.3%  sj2_dr < 0.1873
        + 0.02265687 * max(0.0, 0.0030133 - Q.e2_sq) / 0.001065944   # +2.3%  e2_sq < 0.003013
        + 0.0223666 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +2.2%  lam1 < 0.005954
        + 0.02145773 * max(0.0, 0.02415398 - Q.C2_b2) / 0.02132459   # +2.1%  C2_b2 < 0.02415
        + 0.01751931 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +1.8%  pt_6 < 41.22
        - 0.01448871 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # -1.4%  max_dr < 0.1452
        - 0.01341458 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -1.3%  pt_6 < 39.75
        - 0.01059226 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.psi_0p1 - 0.4008925) / 0.003485257   # -1.1%  centroid_offset > 0.008092 and psi_0p1 > 0.4009
        - 0.01038358 * max(0.0, Q.sj3_pair_mass_min - 4.501727) / 3.480268   # -1.0%  sj3_pair_mass_min > 4.502
        + 0.009130752 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +0.9%  e3 < 0.0005117
        - 0.008793029 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -0.9%  lam2 < 0.001131
        - 0.008721468 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757) / 0.07098489   # -0.9%  pt_6 < 41.22 and z_7 > 0.02321
        + 0.007360127 * max(0.0, 0.1789613 - Q.sj3_dr_max) * max(0.0, 0.1230713 - Q.tau21_b2) / 0.0006424247   # +0.7%  sj3_dr_max < 0.179 and tau21_b2 < 0.1231
        + 0.006693394 * max(0.0, 615.875 - Q.sum_pt) / 30.07766   # +0.7%  sum_pt < 615.9
        + 0.006237507 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 43.23531   # +0.6%  sum_pt < 615.9 and n_dr_0p2_0p4 < 2
        - 0.006147933 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # -0.6%  centroid_offset > 0.008092
        - 0.005934925 * max(0.0, 45.595 - Q.mass) * max(0.0, Q.eta_3 - -0.00592804) / 0.1560163   # -0.6%  mass < 45.59 and eta_3 > -0.005928
        - 0.005934052 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.00433128 - Q.mean_phi2) / 7.021694e-06   # -0.6%  centroid_offset > 0.01838 and mean_phi2 < 0.004331
        - 0.00548668 * max(0.0, 0.1452311 - Q.max_dr) * max(0.0, Q.mean_phi - 0.004406178) / 9.704264e-05   # -0.5%  max_dr < 0.1452 and mean_phi > 0.004406
        - 0.005309091 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 0.2534037 - Q.planar_flow) / 0.0006566196   # -0.5%  lam1 < 0.012 and planar_flow < 0.2534
        + 0.004479638 * max(0.0, 29.90625 - Q.pt_6) / 1.385454   # +0.4%  pt_6 < 29.91
        - 0.004314046 * max(0.0, Q.sj3_dr_max - 0.1879486) * max(0.0, 0.03250122 - Q.abseta_2) / 0.0002335307   # -0.4%  sj3_dr_max > 0.1879 and abseta_2 < 0.0325
        + 0.003687874 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.M3 - 0.0782171) / 0.01402652   # +0.4%  pt_6 < 41.22 and M3 > 0.07822
        - 0.003320538 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # -0.3%  sj3_dr_min > 0.1278
        - 0.003226311 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308) / 0.05438016   # -0.3%  centroid_offset > 0.008092 and sj3_pair_mass_min > 6.811
        - 0.002744024 * max(0.0, Q.centroid_offset - 0.01837778) / 0.005523694   # -0.3%  centroid_offset > 0.01838
        - 0.002740927 * max(0.0, 45.595 - Q.mass) / 14.73532   # -0.3%  mass < 45.59
        + 0.002243025 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +0.2%  sj3_dr_max > 0.1879
        + 0.002078919 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, Q.mean_phi - 0.01251955) / 0.1451915   # +0.2%  sum_pt < 615.9 and mean_phi > 0.01252
        + 0.002043706 * max(0.0, 0.01323868 - Q.width) * max(0.0, -0.02665591 - Q.mean_eta) / 1.690289e-06   # +0.2%  width < 0.01324 and mean_eta < -0.02666
        - 0.001659065 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2) / 2.179362e-05   # -0.2%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        - 0.001502452 * max(0.0, Q.sj3_dr13 - 0.181053) * max(0.0, 0.02658081 - Q.abseta_5) / 8.728483e-05   # -0.2%  sj3_dr13 > 0.1811 and abseta_5 < 0.02658
        + 0.001137615 * max(0.0, 29.90625 - Q.pt_6) * max(0.0, 8.0 - Q.n_pt_above_10) / 0.5895401   # +0.1%  pt_6 < 29.91 and n_pt_above_10 < 8
        + 0.001050867 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # +0.1%  mean_eta > 0.02644
        + 0.001032966 * max(0.0, Q.sj3_dr13 - 0.181053) / 0.02525513   # +0.1%  sj3_dr13 > 0.1811
        - 0.0008750913 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.02702951 - Q.C2) / 2.39871e-05   # -0.1%  centroid_offset > 0.01838 and C2 < 0.02703
        + 0.0008492458 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.mean_eta - 0.02644207) / 1.54475e-06   # +0.1%  lam1 < 0.012 and mean_eta > 0.02644
        + 0.0008095796 * max(0.0, 0.01323868 - Q.width) * max(0.0, -0.02594505 - Q.mean_phi) / 1.701156e-06   # +0.1%  width < 0.01324 and mean_phi < -0.02595
        + 0.0007370373 * max(0.0, 0.01323868 - Q.width) * max(0.0, Q.mean_phi - 0.02612796) / 1.673749e-06   # +0.1%  width < 0.01324 and mean_phi > 0.02613
        + 0.0005837538 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002658741   # +0.1%  lam2 < 0.003408 and n_dr_0_0p05 < 3
        + 0.0004928983 * max(0.0, 0.04176067 - Q.sj2_zsoft) / 0.0006762529   # +0.0%  sj2_zsoft < 0.04176
        + 0.0004090085 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, Q.M3 - 0.06688759) / 9.396674e-06   # +0.0%  mean_eta > 0.02644 and M3 > 0.06689
        + 0.0003275167 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 2.69383   # +0.0%  sj3_pair_mass_min > 4.502 and n_dr_0p2_0p4 > 1
        + 0.0003106085 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.02980347 - Q.eta_0) / 0.0002676158   # +0.0%  centroid_offset > 0.01838 and eta_0 < 0.0298
        + 0.0002238425 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 0.05101919 - Q.z_3) / 0.0001257894   # +0.0%  sum_pt < 615.9 and z_3 < 0.05102
        - 0.0002065221 * max(0.0, Q.eccentricity - 0.927072) / 0.03232992   # -0.0%  eccentricity > 0.9271
        + 0.0001240251 * max(0.0, 0.01323868 - Q.width) * max(0.0, -0.0803833 - Q.eta_7) / 2.326074e-05   # +0.0%  width < 0.01324 and eta_7 < -0.08038
        + 4.683662e-05 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 0.1343804 - Q.dr1_7) / 0.3834372   # +0.0%  pt_6 < 41.22 and dr1_7 < 0.1344
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 42.76;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 42.75647 * (0.1990124
        - 0.07992047 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -8.0%  girth < 0.08724
        - 0.07797618 * max(0.0, Q.width - 0.005590289) / 0.003084256   # -7.8%  width > 0.00559
        + 0.07320819 * max(0.0, Q.width - 0.008678045) / 0.002214537   # +7.3%  width > 0.008678
        - 0.06907504 * max(0.0, Q.mass_over_sum_pt - 0.01109984) / 0.05101851   # -6.9%  mass_over_sum_pt > 0.0111
        - 0.06820385 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -6.8%  girth2 > 0.00752
        - 0.06057652 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -6.1%  mass_over_sum_pt > 0.09041
        - 0.05205211 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -5.2%  sj2_dr < 0.1873
        + 0.04952477 * max(0.0, Q.mass_over_sum_pt - 0.07269073) / 0.01344138   # +5.0%  mass_over_sum_pt > 0.07269
        + 0.04762294 * max(0.0, Q.girth2 - 0.001653836) / 0.005220304   # +4.8%  girth2 > 0.001654
        + 0.0468568 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +4.7%  sj2_dr < 0.1592
        + 0.03237654 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +3.2%  mass_over_sum_pt > 0.08475
        - 0.03231549 * max(0.0, Q.girth2 - 0.004372139) / 0.003638805   # -3.2%  girth2 > 0.004372
        + 0.02733294 * max(0.0, Q.girth2 - 0.01323868) / 0.001442684   # +2.7%  girth2 > 0.01324
        + 0.02519685 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +2.5%  LHA < 0.3033
        + 0.02002841 * max(0.0, Q.girth2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +2.0%  girth2 > 0.004372 and planar_flow < 0.195
        - 0.01921222 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.004032342 - Q.C2_b2) / 2.736263e-05   # -1.9%  centroid_offset < 0.02077 and C2_b2 < 0.004032
        - 0.01801623 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.width - 0.007520088) / 0.0001455029   # -1.8%  planar_flow < 0.195 and width > 0.00752
        - 0.01768257 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # -1.8%  girth2 < 0.003563
        + 0.01503743 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # +1.5%  e2 > 0.03556
        + 0.01440793 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.4%  tau1 < 0.05357
        + 0.01344638 * max(0.0, Q.z_7 - 0.03243272) / 0.02180051   # +1.3%  z_7 > 0.03243
        + 0.01259742 * max(0.0, Q.mass - 36.22941) / 14.20528   # +1.3%  mass > 36.23
        - 0.01161903 * max(0.0, 0.04081947 - Q.girth) / 0.009176315   # -1.2%  girth < 0.04082
        - 0.01020525 * max(0.0, 0.0009641429 - Q.girth2) / 0.0002052288   # -1.0%  girth2 < 0.0009641
        + 0.00980252 * max(0.0, 0.001101266 - Q.e2_sq) / 0.000308004   # +1.0%  e2_sq < 0.001101
        + 0.008373703 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # +0.8%  sj2_dr < 0.1295
        - 0.007784041 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # -0.8%  e2 > 0.05028
        + 0.006230223 * max(0.0, 0.08723651 - Q.girth) * max(0.0, 0.004406178 - Q.mean_phi) / 0.0002325096   # +0.6%  girth < 0.08724 and mean_phi < 0.004406
        - 0.006067325 * max(0.0, Q.centroid_offset - 0.03117077) / 0.002505019   # -0.6%  centroid_offset > 0.03117
        + 0.00582171 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sd_mass - 38.43971) / 1.25962   # +0.6%  planar_flow < 0.195 and sd_mass > 38.44
        - 0.005644456 * max(0.0, 0.001056655 - Q.girth2_top2) / 0.0003002514   # -0.6%  girth2_top2 < 0.001057
        + 0.004631957 * max(0.0, 0.08723651 - Q.girth) * max(0.0, 0.004032342 - Q.C2_b2) / 0.0001217265   # +0.5%  girth < 0.08724 and C2_b2 < 0.004032
        - 0.004586273 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -0.5%  e3 < 8.148e-05
        + 0.004518099 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.02656143 - Q.tau21_b2) / 4.805781e-05   # +0.5%  centroid_offset < 0.02077 and tau21_b2 < 0.02656
        - 0.004047655 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -0.4%  lam1 < 0.008376
        - 0.004033008 * max(0.0, 8.10414e-05 - Q.girth2_top2) / 7.505045e-06   # -0.4%  girth2_top2 < 8.104e-05
        + 0.003717342 * max(0.0, 0.1950135 - Q.planar_flow) / 0.07573236   # +0.4%  planar_flow < 0.195
        - 0.003635529 * max(0.0, 29.04219 - Q.pt_7) / 2.179984   # -0.4%  pt_7 < 29.04
        + 0.003479291 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) / 0.09532154   # +0.3%  z_dr_0p1_0p2 < 0.1586
        - 0.003247113 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.3%  mass > 76.66
        - 0.00304618 * max(0.0, Q.sd_rg - 0.2037854) / 0.01566501   # -0.3%  sd_rg > 0.2038
        + 0.002310113 * max(0.0, 0.003562611 - Q.girth2) * max(0.0, -0.006779839 - Q.mean_eta) / 1.412644e-06   # +0.2%  girth2 < 0.003563 and mean_eta < -0.00678
        + 0.002038995 * max(0.0, 0.001056655 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.006789738) / 7.886295e-07   # +0.2%  girth2_top2 < 0.001057 and centroid_offset > 0.00679
        - 0.001838788 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6) / 0.1690249   # -0.2%  planar_flow < 0.195 and pt_6 < 35.28
        - 0.001670737 * max(0.0, Q.centroid_offset - 0.03117077) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.00188116   # -0.2%  centroid_offset > 0.03117 and n_pt_above_50 > 4
        + 0.001568834 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, Q.sum_pt_top3 - 331.25) / 1.710819   # +0.2%  centroid_offset < 0.02077 and sum_pt_top3 > 331.2
        - 0.001485369 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -0.1%  centroid_offset < 0.02077
        + 0.001323314 * max(0.0, Q.girth2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9458207) / 2.125219e-05   # +0.1%  girth2 > 0.01324 and eccentricity > 0.9458
        - 0.001208541 * max(0.0, Q.mass_over_sum_pt - 0.1079857) / 0.005364315   # -0.1%  mass_over_sum_pt > 0.108
        + 0.001061316 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.pt_4 - 71.6875) / 0.2128009   # +0.1%  planar_flow < 0.195 and pt_4 > 71.69
        + 0.000508524 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) * max(0.0, 2.080881 - Q.N3) / 0.06089036   # +0.1%  z_dr_0p1_0p2 < 0.1586 and N3 < 2.081
        - 0.0004135833 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.0%  centroid_offset > 0.03776
        - 0.0003061827 * max(0.0, 0.001101266 - Q.e2_sq) * max(0.0, -0.04275513 - Q.phi_1) / 2.373915e-08   # -0.0%  e2_sq < 0.001101 and phi_1 < -0.04276
        - 0.0002941193 * max(0.0, 0.001101266 - Q.e2_sq) * max(0.0, Q.phi_0 - 0.0402832) / 2.634889e-08   # -0.0%  e2_sq < 0.001101 and phi_0 > 0.04028
        - 0.0002541995 * max(0.0, Q.centroid_offset - 0.03776099) * max(0.0, Q.pt_4 - 81.375) / 0.0001936307   # -0.0%  centroid_offset > 0.03776 and pt_4 > 81.38
        - 0.0002150002 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, Q.n_dr_0p05_0p1 - 5.0) / 0.0005328768   # -0.0%  tau1 < 0.05357 and n_dr_0p05_0p1 > 5
        + 0.0001510897 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # +0.0%  sd_rg > 0.2788
        + 0.0001368693 * max(0.0, 0.009213932 - Q.tau21_b2) / 0.0008512768   # +0.0%  tau21_b2 < 0.009214
        - 3.977005e-05 * max(0.0, 0.001101266 - Q.e2_sq) * max(0.0, -0.06384277 - Q.eta_2) / 6.602357e-09   # -0.0%  e2_sq < 0.001101 and eta_2 < -0.06384
        + 1.667483e-05 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj2_mass1 - 31.78116) / 1.424888e-07   # +0.0%  e3 < 8.148e-05 and sj2_mass1 > 31.78
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 9.532;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.532263 * (-0.03540479
        - 0.1150769 * max(0.0, 0.06108601 - Q.girth) / 0.01868685   # -11.5%  girth < 0.06109
        + 0.09822981 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0001901112   # +9.8%  girth2 < 0.00502 and z_dr_0p2_0p4 < 0.101
        + 0.07415623 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +7.4%  girth2 < 0.006679 and centroid_offset < 0.02355
        + 0.05748458 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +5.7%  girth2 < 0.00502
        - 0.05497127 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.003562611 - Q.width) / 0.02261167   # -5.5%  mass < 29.64 and width < 0.003563
        + 0.04722038 * max(0.0, 0.003562611 - Q.width) / 0.001184249   # +4.7%  width < 0.003563
        + 0.04012835 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2) / 2.869243e-06   # +4.0%  girth < 0.06109 and lam2 < 0.0001948
        + 0.04006065 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.girth2) / 0.0003703114   # +4.0%  sj3_dr_max < 0.1986 and girth2 < 0.006679
        + 0.03615346 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.width) / 4.808955e-05   # +3.6%  tau1 < 0.05357 and width < 0.003563
        + 0.03589653 * max(0.0, 49.6681 - Q.mass) * max(0.0, 0.0001330621 - Q.lam2) / 0.001509268   # +3.6%  mass < 49.67 and lam2 < 0.0001331
        - 0.03576924 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.0001947983 - Q.lam2) / 0.0007178522   # -3.6%  mass < 21.78 and lam2 < 0.0001948
        - 0.03358818 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # -3.4%  LHA < 0.1967
        - 0.03153809 * max(0.0, 0.0001330621 - Q.lam2) / 6.711676e-05   # -3.2%  lam2 < 0.0001331
        - 0.02838023 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -2.8%  sj3_dr_max < 0.1426
        - 0.02813392 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -2.8%  girth2 < 0.00502 and centroid_offset > 0.00679
        + 0.02707727 * max(0.0, 0.0001330621 - Q.lam2) * max(0.0, 4.721224 - Q.D2_b2) / 0.0002340602   # +2.7%  lam2 < 0.0001331 and D2_b2 < 4.721
        - 0.01743955 * max(0.0, 0.006679471 - Q.girth2) / 0.00293624   # -1.7%  girth2 < 0.006679
        - 0.0150575 * max(0.0, Q.z_dr_0_0p05 - 0.8477313) / 0.0544886   # -1.5%  z_dr_0_0p05 > 0.8477
        - 0.014565 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.n_pt_above_50 - 3.0) / 0.006708018   # -1.5%  girth2 < 0.006679 and n_pt_above_50 > 3
        - 0.01236786 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.008678045 - Q.width) / 0.0002365852   # -1.2%  log_sum_pt > 6.701 and width < 0.008678
        - 0.01194711 * max(0.0, 21.78408 - Q.mass) / 4.431023   # -1.2%  mass < 21.78
        + 0.01131235 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) / 0.00722828   # +1.1%  mass_over_sum_pt < 0.03319
        - 0.01107548 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -1.1%  log_sum_pt > 6.701
        - 0.01090971 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 4.721224 - Q.D2_b2) / 0.04403777   # -1.1%  tau1 < 0.05357 and D2_b2 < 4.721
        + 0.01064242 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +1.1%  sj3_dr_max < 0.1986
        - 0.009501304 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 4.721224 - Q.D2_b2) / 0.008595177   # -1.0%  girth2 < 0.006679 and D2_b2 < 4.721
        + 0.007754102 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 48.71875 - Q.pt_7) / 0.776931   # +0.8%  log_sum_pt > 6.701 and pt_7 < 48.72
        + 0.007184951 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.pt_7 - 15.55391) / 0.4153091   # +0.7%  LHA < 0.1967 and pt_7 > 15.55
        + 0.007097179 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.phi_0 - -0.04013062) / 0.0001189753   # +0.7%  girth2 < 0.006679 and phi_0 > -0.04013
        + 0.006820391 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.4007947 - Q.planar_flow) / 0.0003918103   # +0.7%  girth2 < 0.006679 and planar_flow < 0.4008
        - 0.005887129 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 43.5 - Q.pt_7) / 0.02255472   # -0.6%  girth2 < 0.00502 and pt_7 < 43.5
        + 0.005818942 * max(0.0, 0.06108601 - Q.girth) * max(0.0, Q.width - 0.0003193707) / 1.048473e-05   # +0.6%  girth < 0.06109 and width > 0.0003194
        - 0.005533511 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 6.502799 - Q.log_sum_pt) / 8.085113e-05   # -0.6%  girth2 < 0.00502 and log_sum_pt < 6.503
        - 0.005261822 * max(0.0, 0.001653836 - Q.girth2) / 0.0004289067   # -0.5%  girth2 < 0.001654
        - 0.004865764 * max(0.0, 29.6447 - Q.mass) / 7.34723   # -0.5%  mass < 29.64
        + 0.004305211 * max(0.0, 0.003343241 - Q.centroid_offset) / 0.0002272052   # +0.4%  centroid_offset < 0.003343
        - 0.003797906 * max(0.0, 49.6681 - Q.mass) / 17.06893   # -0.4%  mass < 49.67
        - 0.003686296 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.zdr_0 - 0.007798268) / 0.0001443534   # -0.4%  log_sum_pt > 6.701 and zdr_0 > 0.007798
        - 0.003006816 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 2.955458e-05 - Q.e3) / 8.40609e-07   # -0.3%  log_sum_pt > 6.701 and e3 < 2.955e-05
        + 0.002188816 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 3.10694e-07 - Q.e3) / 3.648373e-06   # +0.2%  sum_pt_top5 > 687.4 and e3 < 3.107e-07
        + 0.002049859 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.2%  sum_pt_top5 > 687.4
        - 0.001891513 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.n_dr_0p05_0p1 - 0.0) / 0.002340464   # -0.2%  girth2 < 0.006679 and n_dr_0p05_0p1 > 0
        + 0.001832025 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.mean_phi - -0.0008556753) / 5.44837e-05   # +0.2%  LHA < 0.1967 and mean_phi > -0.0008557
        - 0.001494236 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 0.001219216   # -0.1%  tau1 < 0.05357 and n_dr_0p2_0p4 > 0
        + 0.001435501 * max(0.0, 0.03448406 - Q.z_6) / 0.001525592   # +0.1%  z_6 < 0.03448
        + 0.001073583 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.m012 - 8.921413) / 0.004746988   # +0.1%  girth2 < 0.006679 and m012 > 8.921
        - 0.001016856 * max(0.0, 29.6447 - Q.mass) * max(0.0, 1.762929e-06 - Q.e3) / 9.656157e-06   # -0.1%  mass < 29.64 and e3 < 1.763e-06
        - 0.0009451515 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.mean_eta - 0.003218391) / 1.870377e-05   # -0.1%  LHA < 0.1967 and mean_eta > 0.003218
        - 0.0009156376 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.008506537   # -0.1%  LHA < 0.1967 and n_pt_above_50 > 6
        - 0.000852485 * max(0.0, 0.003239423 - Q.dr_0) / 5.971839e-05   # -0.1%  dr_0 < 0.003239
        - 0.0008038392 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.2097233   # -0.1%  mass < 29.64 and D2_b2 < 0.7166
        - 0.0007701376 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # -0.1%  tau1 < 0.05357
        - 0.000701878 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.n_dr_0p1_0p2 - 2.0) / 8.060906e-05   # -0.1%  girth2 < 0.006679 and n_dr_0p1_0p2 > 2
        - 0.0005377023 * max(0.0, 0.003343241 - Q.centroid_offset) * max(0.0, 0.5327104 - Q.D2_b2) / 2.280793e-05   # -0.1%  centroid_offset < 0.003343 and D2_b2 < 0.5327
        - 0.0004595007 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, 1.0 - Q.psi_0p3) / 5.234195e-06   # -0.0%  LHA < 0.1967 and psi_0p3 < 1
        + 0.0004467301 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.001011075   # +0.0%  girth < 0.06109 and z_dr_0p2_0p4 < 0.05644
        - 0.0002624527 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 8.0 - Q.n_pt_above_10) / 7.731645   # -0.0%  sum_pt_top5 > 687.4 and n_pt_above_10 < 8
        + 0.0002610136 * max(0.0, 21.78408 - Q.mass) * max(0.0, 48.71875 - Q.pt_7) / 69.12746   # +0.0%  mass < 21.78 and pt_7 < 48.72
        - 0.000189735 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.m012 - 28.3451) / 0.00013178   # -0.0%  girth2 < 0.00502 and m012 > 28.35
        + 0.0001679725 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.07002129   # +0.0%  mass < 21.78 and D2_b2 < 0.7166
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 23.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.6369 * (-0.05450366
        + 0.1605797 * max(0.0, 0.00609665 - Q.girth2) / 0.002544039   # +16.1%  girth2 < 0.006097
        + 0.09421088 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.2973626   # +9.4%  mass < 53.33 and centroid_offset < 0.02686
        + 0.06722908 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # +6.7%  girth2 < 0.003563
        + 0.06319983 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +6.3%  sj3_dr_max < 0.2134
        - 0.05707006 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -5.7%  sj3_dr_max < 0.1426
        - 0.05145391 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -5.1%  lam1 < 0.005954
        + 0.04863932 * max(0.0, Q.lam1 - 0.005433361) / 0.00267714   # +4.9%  lam1 > 0.005433
        - 0.04644983 * max(0.0, Q.lam1 - 0.003377388) / 0.003672352   # -4.6%  lam1 > 0.003377
        + 0.0438522 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +4.4%  centroid_offset < 0.01838
        - 0.03962911 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -4.0%  mass_over_sum_pt < 0.07269
        - 0.02256037 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -2.3%  e2 > 0.03556
        - 0.02042963 * max(0.0, 0.05464922 - Q.girth) / 0.01532978   # -2.0%  girth < 0.05465
        - 0.019288 * max(0.0, 53.33237 - Q.mass) / 19.36004   # -1.9%  mass < 53.33
        + 0.01657332 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.07073975 - Q.abseta_1) / 0.0003304963   # +1.7%  centroid_offset < 0.01838 and abseta_1 < 0.07074
        - 0.01651899 * max(0.0, 2.371297e-05 - Q.e3) / 1.105328e-05   # -1.7%  e3 < 2.371e-05
        + 0.01571005 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd) / 0.0002178654   # +1.6%  centroid_offset < 0.01838 and z_2nd < 0.2056
        + 0.01529754 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt) / 5.175436   # +1.5%  mass < 53.33 and log_sum_pt < 6.843
        - 0.0130921 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266) / 0.0005312449   # -1.3%  lam1 > 0.005433 and eccentricity > 0.7117
        + 0.01275494 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +1.3%  max_dr < 0.1118
        - 0.01258067 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255) / 0.000277001   # -1.3%  tau1 < 0.0437 and eccentricity > 0.9031
        - 0.01256572 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) / 6.087676   # -1.3%  sj3_pair_mass_max < 24.23
        - 0.01255137 * max(0.0, Q.girth - 0.0717028) / 0.01201981   # -1.3%  girth > 0.0717
        + 0.01232015 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.0002201438   # +1.2%  girth2 < 0.006097 and planar_flow < 0.3221
        + 0.01009631 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +1.0%  e2 > 0.04447
        + 0.009649884 * max(0.0, 0.0001721983 - Q.width) / 1.441896e-05   # +1.0%  width < 0.0001722
        - 0.009458689 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # -0.9%  e3 > 8.148e-05
        - 0.008454317 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.002834884 - Q.mean_phi) / 1.344572e-05   # -0.8%  girth2 < 0.006097 and mean_phi < 0.002835
        - 0.007731417 * max(0.0, 0.0284695 - Q.C3) / 0.008714909   # -0.8%  C3 < 0.02847
        - 0.007275489 * Q.z_dr_0p2_0p4 / 0.02920532   # -0.7%  z_dr_0p2_0p4
        + 0.00680069 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # +0.7%  tau1 < 0.0437
        + 0.006795902 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +0.7%  sum_pt < 988.4
        + 0.006381365 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +0.6%  e3 > 0.0001869
        - 0.004831654 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1) / 0.1931178   # -0.5%  centroid_offset < 0.01838 and pt_1 < 159.2
        + 0.004182686 * max(0.0, 0.006292091 - Q.zdr_0) / 0.001006321   # +0.4%  zdr_0 < 0.006292
        - 0.004145302 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.01108383   # -0.4%  n_dr_0p2_0p4 > 1 and sj3_dr_min < 0.209
        + 0.004106359 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.008712838   # +0.4%  centroid_offset < 0.01838 and n_for_90pct > 5
        + 0.003922642 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.6875) / 0.3914557   # +0.4%  sj3_dr_max < 0.1426 and pt_6 > 33.69
        + 0.003819991 * max(0.0, 0.002316125 - Q.centroid_offset) / 9.872932e-05   # +0.4%  centroid_offset < 0.002316
        + 0.002852196 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +0.3%  sj3_dr_max < 0.107
        + 0.002673367 * max(0.0, 31.125 - Q.pt_4) / 0.3140792   # +0.3%  pt_4 < 31.12
        - 0.00253891 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) * max(0.0, Q.absphi_2 - 0.007518768) / 0.03147941   # -0.3%  sj3_pair_mass_max < 24.23 and absphi_2 > 0.007519
        + 0.00253227 * max(0.0, 559.6875 - Q.sum_pt) / 16.18216   # +0.3%  sum_pt < 559.7
        + 0.002355528 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.2%  lam2 > 0.001131
        + 0.001966054 * max(0.0, 7.0 - Q.n_for_90pct) / 0.5603294   # +0.2%  n_for_90pct < 7
        - 0.001923771 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2) / 0.000627491   # -0.2%  log_sum_pt > 6.896 and D2_b2 < 0.7166
        + 0.001769868 * max(0.0, 53.33237 - Q.mass) * max(0.0, Q.D2 - 2.843757) / 5.555775   # +0.2%  mass < 53.33 and D2 > 2.844
        + 0.001616137 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.2%  log_sum_pt > 6.896
        - 0.001452058 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, Q.mean_phi - 0.02612796) / 3.162575e-07   # -0.1%  girth2 < 0.006097 and mean_phi > 0.02613
        - 0.001193824 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, -0.01275329 - Q.mean_phi) / 1.158446e-05   # -0.1%  tau1 < 0.0437 and mean_phi < -0.01275
        + 0.001023208 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.tau4 - 0.001224244) / 1.099241e-05   # +0.1%  centroid_offset < 0.01838 and tau4 > 0.001224
        - 0.0009010277 * max(0.0, 2.371297e-05 - Q.e3) * max(0.0, Q.eccentricity - 0.8319502) / 8.425435e-07   # -0.1%  e3 < 2.371e-05 and eccentricity > 0.832
        - 0.0006682112 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.1%  sum_pt_top5 > 840
        + 0.0006321324 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.1%  n_dr_0p2_0p4 > 1
        + 0.0003596806 * max(0.0, 0.05464922 - Q.girth) * max(0.0, Q.z_5 - 0.03672711) / 0.0004195788   # +0.0%  girth < 0.05465 and z_5 > 0.03673
        - 0.0003301377 * max(0.0, 0.05464922 - Q.girth) * max(0.0, 0.02656143 - Q.tau21_b2) / 1.23798e-05   # -0.0%  girth < 0.05465 and tau21_b2 < 0.02656
        - 0.0002805118 * max(0.0, 31.125 - Q.pt_4) * max(0.0, Q.dr_min_012 - 0.01488897) / 0.0006591722   # -0.0%  pt_4 < 31.12 and dr_min_012 > 0.01489
        - 0.0002355426 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.mean_phi - 0.02612796) / 2.994783e-06   # -0.0%  tau1 < 0.0437 and mean_phi > 0.02613
        + 0.0002198946 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, Q.abseta_1 - 0.03625488) / 8.135786e-06   # +0.0%  sj3_dr_max < 0.107 and abseta_1 > 0.03625
        - 0.0002093663 * max(0.0, 31.125 - Q.pt_4) * max(0.0, 8.0 - Q.n_pt_above_1) / 0.003929533   # -0.0%  pt_4 < 31.12 and n_pt_above_1 < 8
        - 5.686698e-05 * max(0.0, Q.e2 - 0.03556091) * max(0.0, Q.mean_eta - 0.00189657) / 5.895833e-05   # -0.0%  e2 > 0.03556 and mean_eta > 0.001897
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 11.03;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.0256 * (0.197289
        + 0.1221994 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # +12.2%  girth2 > 0.00752
        - 0.0944079 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -9.4%  lam1 < 0.005954
        - 0.06346352 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -6.3%  lam1 < 0.004184
        + 0.05767465 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +5.8%  mass < 76.66
        - 0.05485466 * max(0.0, 0.06473447 - Q.z_7) / 0.01631636   # -5.5%  z_7 < 0.06473
        - 0.05374905 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # -5.4%  lam1 > 0.00733
        + 0.04798208 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # +4.8%  tau1 > 0.05357
        + 0.04779513 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1) / 0.002082998   # +4.8%  zdr_0 < 0.02118 and z_dr_0p05_0p1 < 0.2919
        + 0.03917425 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # +3.9%  girth2_top3 < 0.002152
        + 0.03560061 * max(0.0, Q.lam2 - 0.0003061234) / 0.0004216599   # +3.6%  lam2 > 0.0003061
        - 0.03466605 * max(0.0, 0.002635418 - Q.girth2) / 0.0007941346   # -3.5%  girth2 < 0.002635
        + 0.03175441 * max(0.0, 45.75 - Q.pt_7) / 12.14632   # +3.2%  pt_7 < 45.75
        + 0.02606462 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.002471 - Q.D2) / 1.812028   # +2.6%  pt_7 < 45.75 and D2 < 1.002
        + 0.02595625 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +2.6%  e3 < 8.148e-05
        + 0.02292311 * Q.e2 / 0.02863215   # +2.3%  e2
        - 0.02275267 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -2.3%  LHA > 0.3033
        + 0.01624355 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.01258764) / 3.400017e-05   # +1.6%  zdr_0 < 0.02118 and centroid_offset > 0.01259
        + 0.01583099 * Q.mean_phi / 0.01077348   # +1.6%  mean_phi
        + 0.01557247 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # +1.6%  sj3_pair_mass_min > 11.05
        - 0.01473069 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -1.5%  lam1 < 0.001504
        + 0.01036054 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +1.0%  sj3_dr_min > 0.1278
        + 0.01026104 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # +1.0%  zdr_0 < 0.02118
        - 0.009540846 * max(0.0, Q.mass_top5 - 45.32077) / 3.61051   # -1.0%  mass_top5 > 45.32
        - 0.008936114 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.planar_flow - 0.04505724) / 0.0002791469   # -0.9%  lam2 > 0.0003061 and planar_flow > 0.04506
        + 0.008563953 * max(0.0, Q.C2_b2 - 0.009032972) / 0.001703043   # +0.9%  C2_b2 > 0.009033
        + 0.008500686 * max(0.0, 45.75 - Q.pt_7) * max(0.0, Q.abseta_3 - 0.0013237) / 0.460532   # +0.9%  pt_7 < 45.75 and abseta_3 > 0.001324
        + 0.008182407 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.sj3_pairmax_over_m - 0.6745796) / 3.382029e-05   # +0.8%  lam2 > 0.0003061 and sj3_pairmax_over_m > 0.6746
        - 0.007892012 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # -0.8%  sj3_dr_max > 0.1986
        - 0.007740113 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -0.8%  e3 > 3.892e-05
        - 0.006708608 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2) / 0.0002929492   # -0.7%  lam1 > 0.00733 and D2_b2 < 0.3809
        + 0.006213986 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.dr_7 - 0.1078355) / 8.813877e-05   # +0.6%  zdr_0 < 0.02118 and dr_7 > 0.1078
        - 0.005934638 * max(0.0, Q.girth2 - 0.02530566) / 0.0002851418   # -0.6%  girth2 > 0.02531
        - 0.005175832 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -0.5%  lam2 > 0.003408
        + 0.004668054 * max(0.0, Q.sj3_pair_mass_min - 11.051) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 3.623232   # +0.5%  sj3_pair_mass_min > 11.05 and n_dr_0p2_0p4 > 0
        + 0.004487239 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, Q.eta_7 - -0.0803833) / 0.002483399   # +0.4%  tau1 > 0.05357 and eta_7 > -0.08038
        + 0.004141753 * max(0.0, 0.02563286 - Q.M2) / 0.001872375   # +0.4%  M2 < 0.02563
        + 0.004117554 * max(0.0, Q.phi_6 - 0.04013062) / 0.01096667   # +0.4%  phi_6 > 0.04013
        + 0.003733882 * max(0.0, Q.LHA - 0.3033137) * max(0.0, Q.sj3_mass1 - 5.294926) / 0.009410502   # +0.4%  LHA > 0.3033 and sj3_mass1 > 5.295
        + 0.003436837 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.3%  sum_pt > 988.4
        + 0.003429516 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj3_dr13 - 0.181053) / 5.645056e-07   # +0.3%  e3 < 8.148e-05 and sj3_dr13 > 0.1811
        - 0.003159028 * max(0.0, 6.080494 - Q.log_sum_pt) / 0.006097657   # -0.3%  log_sum_pt < 6.08
        + 0.00284619 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # +0.3%  mean_eta > 0.02644
        - 0.002807274 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.3658817 - Q.z_dr_0_0p05) / 0.002463327   # -0.3%  sj3_dr_min > 0.1278 and z_dr_0_0p05 < 0.3659
        + 0.002279123 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.02695109 - Q.dr_min_012) / 0.0001231931   # +0.2%  sj3_dr_min > 0.1278 and dr_min_012 < 0.02695
        - 0.002128803 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # -0.2%  sj2_dr > 0.3004
        - 0.001904579 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 6.572938 - Q.log_sum_pt) / 1.257183   # -0.2%  pt_7 < 45.75 and log_sum_pt < 6.573
        - 0.001782229 * max(0.0, 6.080494 - Q.log_sum_pt) * max(0.0, Q.mean_eta2 - 9.030369e-05) / 5.001864e-05   # -0.2%  log_sum_pt < 6.08 and mean_eta2 > 9.03e-05
        - 0.001580772 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, -0.004664942 - Q.mean_eta) / 3.825289e-05   # -0.2%  z_7 < 0.06473 and mean_eta < -0.004665
        - 0.001365384 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.2723288 - Q.sj3_mass3) / 0.0003425911   # -0.1%  lam1 > 0.00733 and sj3_mass3 < 0.2723
        + 0.001362442 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1797097) / 5.905455e-07   # +0.1%  e3 < 8.148e-05 and sj3_dr23 > 0.1797
        - 0.0009564941 * max(0.0, 6.080494 - Q.log_sum_pt) * max(0.0, Q.absphi_7 - 0.03546143) / 0.0002718019   # -0.1%  log_sum_pt < 6.08 and absphi_7 > 0.03546
        - 0.0006552168 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, Q.D2_b2 - 1.129616) / 0.0002590844   # -0.1%  lam1 > 0.00733 and D2_b2 > 1.13
        + 0.000472923 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 1.911889 - Q.D2_b2) / 0.005346342   # +0.0%  sj3_dr_min > 0.1278 and D2_b2 < 1.912
        + 0.0002861589 * max(0.0, Q.phi_6 - 0.04013062) * max(0.0, -0.1218262 - Q.eta_7) / 4.512671e-05   # +0.0%  phi_6 > 0.04013 and eta_7 < -0.1218
        - 0.0002602062 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.03573274 - Q.sj2_mass2) / 7.668628e-05   # -0.0%  sj3_dr_min > 0.1278 and sj2_mass2 < 0.03573
        + 0.0002544699 * max(0.0, Q.LHA - 0.3033137) * max(0.0, Q.sj3_pairmax_over_m - 0.901705) / 9.935534e-05   # +0.0%  LHA > 0.3033 and sj3_pairmax_over_m > 0.9017
        - 0.0002113441 * max(0.0, Q.mass_top5 - 45.32077) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 3.955556   # -0.0%  mass_top5 > 45.32 and n_dr_0p2_0p4 < 2
        + 0.0001771659 * max(0.0, Q.girth2 - 0.007520088) * max(0.0, 0.003088708 - Q.dr1_4) / 4.730438e-08   # +0.0%  girth2 > 0.00752 and dr1_4 < 0.003089
        - 4.862116e-05 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, Q.absphi_5 - 0.04455566) / 2.4347e-05   # -0.0%  mean_eta > 0.02644 and absphi_5 > 0.04456
        + 3.987224e-05 * max(0.0, 45.75 - Q.pt_7) * max(0.0, Q.tau4 - 0.007583927) / 0.001915061   # +0.0%  pt_7 < 45.75 and tau4 > 0.007584
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 29.72;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.72399 * (-0.02824941
        + 0.2058826 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +20.6%  width < 0.008678
        + 0.1247006 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +12.5%  girth2 < 0.01324
        - 0.06462083 * max(0.0, 0.169029 - Q.sj3_dr_max) / 0.04876366   # -6.5%  sj3_dr_max < 0.169
        - 0.057607 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -5.8%  lam1 < 0.008376
        + 0.05692605 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +5.7%  centroid_offset < 0.03776
        - 0.05003468 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -5.0%  girth < 0.0717
        - 0.04566714 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -4.6%  e2_sq < 0.008169
        + 0.04126267 * max(0.0, 0.2623172 - Q.sj3_dr_max) / 0.1129006   # +4.1%  sj3_dr_max < 0.2623
        - 0.03565391 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # -3.6%  tau1 < 0.09539
        + 0.03093631 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +3.1%  z_7 > 0.01686
        + 0.02645728 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +2.6%  sj3_dr_max < 0.1426
        + 0.02192822 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +2.2%  planar_flow < 0.2534
        + 0.01759241 * max(0.0, 0.09538712 - Q.tau1) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.03209574   # +1.8%  tau1 < 0.09539 and z_dr_0p05_0p1 < 0.846
        - 0.01750438 * max(0.0, 0.004372139 - Q.girth2) / 0.00156571   # -1.8%  girth2 < 0.004372
        - 0.01562982 * max(0.0, 0.01437952 - Q.centroid_offset) / 0.004306483   # -1.6%  centroid_offset < 0.01438
        - 0.0148726 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -1.5%  mass < 69.61
        + 0.01250549 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # +1.3%  max_dr < 0.1452
        - 0.01226984 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.03601074 - Q.abseta_4) / 8.179649e-05   # -1.2%  centroid_offset < 0.01438 and abseta_4 < 0.03601
        - 0.01116026 * max(0.0, 0.02054282 - Q.girth) / 0.002592388   # -1.1%  girth < 0.02054
        - 0.009996561 * max(0.0, 49.6681 - Q.mass) / 17.06893   # -1.0%  mass < 49.67
        - 0.008862164 * max(0.0, 0.0005049491 - Q.lam1) / 8.739619e-05   # -0.9%  lam1 < 0.0005049
        - 0.007905291 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -0.8%  lam1 < 0.00733
        + 0.007829373 * max(0.0, 15.45403 - Q.mass) / 2.385786   # +0.8%  mass < 15.45
        + 0.007694613 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.pt_6 - 19.46875) / 0.1742442   # +0.8%  girth2 < 0.01324 and pt_6 > 19.47
        - 0.007540861 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 0.0345459 - Q.absphi_0) / 0.0003803849   # -0.8%  centroid_offset < 0.03776 and absphi_0 < 0.03455
        + 0.0074376 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0007565864   # +0.7%  centroid_offset < 0.01438 and D2_b2 < 0.5327
        - 0.006762368 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -0.7%  sj2_dr > 0.1779
        - 0.006419136 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt) / 3.513015   # -0.6%  centroid_offset < 0.03776 and sum_pt < 901.6
        - 0.006379666 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001130645 - Q.lam2) / 1.843299e-05   # -0.6%  sj2_dr > 0.1779 and lam2 < 0.001131
        + 0.005343683 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.5%  sj2_dr > 0.2688
        - 0.004743698 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 1.050302e-05 - Q.e3) / 1.771164e-07   # -0.5%  planar_flow < 0.2534 and e3 < 1.05e-05
        + 0.004476044 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 0.04977417 - Q.abseta_4) / 0.001815038   # +0.4%  planar_flow < 0.2534 and abseta_4 < 0.04977
        - 0.003816794 * max(0.0, 49.6681 - Q.mass) * max(0.0, 1.332146 - Q.D2) / 1.334408   # -0.4%  mass < 49.67 and D2 < 1.332
        + 0.003616609 * max(0.0, Q.eccentricity - 0.9884745) / 0.001961982   # +0.4%  eccentricity > 0.9885
        - 0.00357814 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.0283055   # -0.4%  sj2_dr > 0.1779 and n_pt_above_50 > 4
        - 0.003300177 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # -0.3%  centroid_offset > 0.0499 and D2 < 2.844
        - 0.002987093 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # -0.3%  C2 < 0.03579
        - 0.00290263 * max(0.0, Q.eccentricity - 0.9884745) * max(0.0, 2.771069 - Q.sj2_mass2) / 0.00304207   # -0.3%  eccentricity > 0.9885 and sj2_mass2 < 2.771
        + 0.002730904 * max(0.0, Q.pt_7 - 29.04219) / 7.785987   # +0.3%  pt_7 > 29.04
        + 0.002532595 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, -0.001813533 - Q.mean_phi) / 4.860641e-05   # +0.3%  centroid_offset < 0.03776 and mean_phi < -0.001814
        - 0.002197423 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 4.721224 - Q.D2_b2) / 16.05793   # -0.2%  pt_6 < 41.22 and D2_b2 < 4.721
        - 0.002154408 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 37.15625 - Q.pt_7) / 0.5537514   # -0.2%  planar_flow < 0.2534 and pt_7 < 37.16
        + 0.002032295 * max(0.0, Q.z_dr_0p05_0p1 - 0.8460335) / 0.01076887   # +0.2%  z_dr_0p05_0p1 > 0.846
        + 0.001841217 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.5186963 - Q.tau32) / 0.0001128939   # +0.2%  centroid_offset < 0.01438 and tau32 < 0.5187
        - 0.001818118 * max(0.0, Q.max_pair_mass - 33.3761) / 0.9279851   # -0.2%  max_pair_mass > 33.38
        - 0.001672373 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.2%  planar_flow < 0.2534 and sum_pt < 840
        - 0.001581837 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.5502779 - Q.tau32) / 0.0001096908   # -0.2%  centroid_offset > 0.0499 and tau32 < 0.5503
        + 0.0009558899 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, Q.phi_1 - 0.002190304) / 4.583393e-05   # +0.1%  centroid_offset < 0.01438 and phi_1 > 0.00219
        + 0.0009522867 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +0.1%  pt_6 < 41.22
        - 0.0008041829 * max(0.0, Q.eccentricity - 0.9884745) * max(0.0, 0.02130127 - Q.eta_0) / 6.650689e-05   # -0.1%  eccentricity > 0.9885 and eta_0 < 0.0213
        + 0.0007437088 * max(0.0, 0.008168571 - Q.e2_sq) * max(0.0, Q.mass_top2 - 6.779915) / 0.005654747   # +0.1%  e2_sq < 0.008169 and mass_top2 > 6.78
        + 0.000738351 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, -0.005273438 - Q.eta_1) / 4.114128e-05   # +0.1%  centroid_offset < 0.01438 and eta_1 < -0.005273
        - 0.0006290537 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # -0.1%  LHA < 0.3127
        + 0.0004675531 * max(0.0, 49.6681 - Q.mass) * max(0.0, Q.dr1_7 - 0.1792439) / 0.02583476   # +0.0%  mass < 49.67 and dr1_7 > 0.1792
        + 0.0004082633 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 0.415246 - Q.D2) / 8.563062e-05   # +0.0%  girth2 < 0.01324 and D2 < 0.4152
        - 0.0003397682 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 61.53125 - Q.pt_2) / 0.009295235   # -0.0%  girth2 < 0.01324 and pt_2 < 61.53
        - 0.0003319348 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.0%  centroid_offset > 0.0499
        + 0.0001154744 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.0001991733   # +0.0%  centroid_offset < 0.03776 and dr_max_012 > 0.1828
        - 9.03635e-05 * max(0.0, 15.45403 - Q.mass) * max(0.0, -0.05890198 - Q.phi_1) / 6.939758e-05   # -0.0%  mass < 15.45 and phi_1 < -0.0589
        - 5.743179e-05 * max(0.0, 69.61135 - Q.mass) * max(0.0, 0.415246 - Q.D2) / 0.1191559   # -0.0%  mass < 69.61 and D2 < 0.4152
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.7144;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.7144429 * (-0.757889
        + 0.3158466 * max(0.0, Q.girth2 - 0.01882765) / 0.0007605369   # +31.6%  girth2 > 0.01883
        - 0.105591 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -10.6%  e2 > 0.06344
        + 0.09087164 * max(0.0, Q.mass - 69.61135) * max(0.0, Q.centroid_offset - 0.009480685) / 0.04081507   # +9.1%  mass > 69.61 and centroid_offset > 0.009481
        + 0.06930721 * max(0.0, Q.mass - 91.19) / 0.5839191   # +6.9%  mass > 91.19
        + 0.06182336 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, Q.abseta_6 - 0.01612854) / 7.142361e-05   # +6.2%  girth2 > 0.01883 and abseta_6 > 0.01613
        - 0.05966574 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, 53.4375 - Q.pt_7) / 0.01439677   # -6.0%  girth2 > 0.01883 and pt_7 < 53.44
        - 0.0573852 * max(0.0, Q.mass_over_sum_pt - 0.1309286) / 0.002541185   # -5.7%  mass_over_sum_pt > 0.1309
        - 0.05212526 * max(0.0, Q.zdr_0 - 0.03981924) / 0.0005695785   # -5.2%  zdr_0 > 0.03982
        - 0.03495066 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, Q.lam2 - 0.000537286) / 2.828867e-06   # -3.5%  girth2 > 0.01883 and lam2 > 0.0005373
        + 0.03036168 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +3.0%  centroid_offset > 0.0499
        + 0.02729564 * max(0.0, Q.mass - 69.61135) * max(0.0, 6.423044 - Q.log_sum_pt) / 0.0982281   # +2.7%  mass > 69.61 and log_sum_pt < 6.423
        - 0.02418447 * max(0.0, Q.mass - 69.61135) / 2.406702   # -2.4%  mass > 69.61
        - 0.01704665 * max(0.0, Q.girth2_top2 - 0.01403324) * max(0.0, -0.006701703 - Q.mean_phi) / 9.987654e-06   # -1.7%  girth2_top2 > 0.01403 and mean_phi < -0.006702
        - 0.01069279 * max(0.0, Q.mass - 91.19) * max(0.0, Q.centroid_offset - 0.02076709) / 0.004710116   # -1.1%  mass > 91.19 and centroid_offset > 0.02077
        - 0.01016779 * max(0.0, Q.girth2_top2 - 0.01403324) / 0.001129575   # -1.0%  girth2_top2 > 0.01403
        + 0.009908945 * max(0.0, Q.mean_phi - 0.02612796) / 0.0006999411   # +1.0%  mean_phi > 0.02613
        + 0.008555989 * max(0.0, Q.mass - 69.61135) * max(0.0, Q.dr_4 - 0.02477348) / 0.3143833   # +0.9%  mass > 69.61 and dr_4 > 0.02477
        - 0.00350666 * max(0.0, Q.mass - 69.61135) * max(0.0, Q.abseta_5 - 0.1522217) / 0.02987685   # -0.4%  mass > 69.61 and abseta_5 > 0.1522
        - 0.003178684 * max(0.0, Q.mass - 91.19) * max(0.0, Q.eta_5 - 0.007119751) / 0.02885229   # -0.3%  mass > 91.19 and eta_5 > 0.00712
        + 0.003066923 * max(0.0, Q.mass - 91.19) * max(0.0, 50.3522 - Q.mass_top3) / 4.712306   # +0.3%  mass > 91.19 and mass_top3 < 50.35
        + 0.002617998 * max(0.0, Q.mass - 91.19) * max(0.0, 0.117984 - Q.abseta_1) / 0.02338086   # +0.3%  mass > 91.19 and abseta_1 < 0.118
        - 0.001304305 * max(0.0, Q.mass - 91.19) * max(0.0, Q.n_dr_0p2_0p4 - 2.0) / 0.4387749   # -0.1%  mass > 91.19 and n_dr_0p2_0p4 > 2
        + 0.000544858 * max(0.0, Q.mass - 91.19) * max(0.0, 0.002207727 - Q.tau4) / 3.151142e-05   # +0.1%  mass > 91.19 and tau4 < 0.002208
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 19.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.77919 * (0.009115789
        + 0.2793173 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # +27.9%  girth < 0.1484
        - 0.08206522 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -8.2%  e2 < 0.08001
        + 0.07335448 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +7.3%  width < 0.01324
        + 0.0682309 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +6.8%  lam1 < 0.01643
        - 0.06496634 * max(0.0, 0.007520088 - Q.width) / 0.003544991   # -6.5%  width < 0.00752
        - 0.06147854 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -6.1%  girth < 0.1484 and log_sum_pt < 6.804
        + 0.05060379 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +5.1%  lam1 < 0.006507
        - 0.04637842 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -4.6%  girth < 0.1484 and pt_7 < 38.53
        - 0.03067857 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 8.324263e-07   # -3.1%  e3 < 5.335e-05 and centroid_offset < 0.03776
        + 0.026247 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7) / 781.5808   # +2.6%  sum_pt_top5 > 658.1 and pt_7 < 40.04
        + 0.02314268 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +2.3%  log_sum_pt > 6.503
        + 0.01905909 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # +1.9%  e3 < 5.335e-05
        - 0.01443168 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7) / 0.06765006   # -1.4%  pt_6 < 31.91 and z_7 < 0.05861
        - 0.0140624 * max(0.0, Q.mass - 49.6681) / 7.721594   # -1.4%  mass > 49.67
        - 0.01165358 * max(0.0, 0.0005124533 - Q.girth2_top2) / 0.0001104529   # -1.2%  girth2_top2 < 0.0005125
        + 0.01140149 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.tau2 - 0.008780509) / 0.000269945   # +1.1%  girth < 0.1484 and tau2 > 0.008781
        + 0.01015992 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.07474969 - Q.M3) / 0.0006773733   # +1.0%  girth < 0.1484 and M3 < 0.07475
        - 0.00967535 * max(0.0, 31.90625 - Q.pt_6) / 1.826854   # -1.0%  pt_6 < 31.91
        + 0.009148389 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +0.9%  sum_pt_top5 > 791.1
        + 0.007816163 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.z_7 - 0.06164517) / 0.0003465887   # +0.8%  girth < 0.1484 and z_7 > 0.06165
        + 0.007759632 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 31.90625 - Q.pt_6) / 134.9952   # +0.8%  sum_pt_top5 > 791.1 and pt_6 < 31.91
        - 0.007348467 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # -0.7%  girth < 0.007674
        + 0.007226962 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, 0.04681396 - Q.absphi_6) / 0.003005137   # +0.7%  log_sum_pt > 6.503 and absphi_6 < 0.04681
        - 0.007152806 * max(0.0, Q.sj3_dr23 - 0.1974628) / 0.02478939   # -0.7%  sj3_dr23 > 0.1975
        + 0.006934047 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55592   # +0.7%  sum_pt_top5 > 658.1
        + 0.006736928 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +0.7%  lam2 < 0.0001948
        - 0.005786225 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.000537286 - Q.lam2) / 4.021081e-05   # -0.6%  girth < 0.1484 and lam2 < 0.0005373
        + 0.00573478 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.D3 - 0.2213841) / 3.794981e-05   # +0.6%  e3 < 5.335e-05 and D3 > 0.2214
        - 0.004451997 * max(0.0, 0.1484084 - Q.girth) * max(0.0, -9.311297e-05 - Q.mean_phi) / 0.0003735644   # -0.4%  girth < 0.1484 and mean_phi < -9.311e-05
        - 0.004042327 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # -0.4%  z_7 < 0.02807
        + 0.003196356 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 74.5625 - Q.pt_3) / 205.6534   # +0.3%  sum_pt_top5 > 791.1 and pt_3 < 74.56
        - 0.002803779 * max(0.0, Q.pt_7 - 48.71875) / 0.6759439   # -0.3%  pt_7 > 48.72
        - 0.00260369 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.sj2_mass1 - 31.78116) / 0.01215768   # -0.3%  girth < 0.1484 and sj2_mass1 > 31.78
        + 0.002503598 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.pair_mass_0_7 - 10.2219) / 0.2907061   # +0.3%  log_sum_pt > 6.503 and pair_mass_0_7 > 10.22
        - 0.002353476 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # -0.2%  z_5 < 0.02818
        + 0.001940422 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.08151794 - Q.M3) / 0.08390625   # +0.2%  sum_pt > 988.4 and M3 < 0.08152
        + 0.001881608 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # +0.2%  z_6 < 0.0216
        - 0.001813838 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6) / 104.494   # -0.2%  sum_pt > 988.4 and pt_6 < 62.25
        - 0.001167049 * max(0.0, Q.mass - 49.6681) * max(0.0, 0.4684459 - Q.z_dr_0p1_0p2) / 1.50172   # -0.1%  mass > 49.67 and z_dr_0p1_0p2 < 0.4684
        + 0.001159966 * max(0.0, Q.pt_7 - 48.71875) * max(0.0, Q.pt_2 - 126.75) / 3.467244   # +0.1%  pt_7 > 48.72 and pt_2 > 126.8
        - 0.0007439812 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.02320757) / 0.2899084   # -0.1%  sum_pt_top5 > 658.1 and z_7 > 0.02321
        + 0.0005380348 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.1%  sum_pt > 988.4
        - 0.0002487444 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.zdr_7 - 0.0007928864) / 0.0001818422   # -0.0%  log_sum_pt > 6.503 and zdr_7 > 0.0007929
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 29.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.08798 * (0.001284995
        - 0.09447375 * max(0.0, Q.width - 0.006679471) / 0.002702003   # -9.4%  width > 0.006679
        - 0.06878625 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -6.9%  girth2 > 0.00752
        + 0.06835855 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # +6.8%  girth2 > 0.008678
        - 0.06514309 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -6.5%  mass_over_sum_pt > 0.09041
        + 0.05480491 * max(0.0, Q.e2_sq - 0.002074109) / 0.004484017   # +5.5%  e2_sq > 0.002074
        + 0.04950954 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # +5.0%  sj2_dr > 0.1592
        + 0.04673967 * max(0.0, Q.width - 0.003562611) / 0.004066872   # +4.7%  width > 0.003563
        - 0.04381292 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -4.4%  girth > 0.08724
        + 0.03706086 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +3.7%  mass_over_sum_pt > 0.07992
        + 0.03422186 * max(0.0, Q.girth - 0.02689598) / 0.03613842   # +3.4%  girth > 0.0269
        - 0.03085802 * max(0.0, Q.e2_sq - 0.0030133) / 0.003940214   # -3.1%  e2_sq > 0.003013
        - 0.02857922 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # -2.9%  sj2_dr > 0.1295
        - 0.02707822 * max(0.0, Q.e2 - 0.01655442) / 0.01596508   # -2.7%  e2 > 0.01655
        - 0.02364627 * max(0.0, 74.57663 - Q.sd_mass) / 42.04056   # -2.4%  sd_mass < 74.58
        + 0.02286096 * max(0.0, Q.width - 0.01323868) / 0.001442684   # +2.3%  width > 0.01324
        + 0.02143535 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +2.1%  e2 > 0.04111
        - 0.02118334 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -2.1%  planar_flow < 0.1115 and lam1 < 0.00733
        + 0.01929261 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +1.9%  mass_over_sum_pt > 0.08475
        + 0.01830184 * max(0.0, Q.width - 0.005590289) / 0.003084256   # +1.8%  width > 0.00559
        + 0.01694498 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # +1.7%  planar_flow < 0.1115 and lam1 < 0.01643
        + 0.01530977 * max(0.0, Q.girth - 0.04081947) / 0.0270611   # +1.5%  girth > 0.04082
        + 0.01485324 * max(0.0, 49.91626 - Q.sd_mass) / 22.60916   # +1.5%  sd_mass < 49.92
        + 0.01471988 * max(0.0, Q.girth - 0.08065885) / 0.009330634   # +1.5%  girth > 0.08066
        + 0.01447063 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +1.4%  LHA > 0.3033
        - 0.0116755 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # -1.2%  e3 < 5.335e-05
        - 0.0108792 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # -1.1%  e2 > 0.05028
        + 0.01018177 * max(0.0, 5.0 - Q.n_dr_0_0p05) / 1.94322   # +1.0%  n_dr_0_0p05 < 5
        - 0.01016253 * max(0.0, Q.sj2_dr - 0.2001708) / 0.02317349   # -1.0%  sj2_dr > 0.2002
        - 0.009471648 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.9%  psi_0p1 > 0.9761
        - 0.008817004 * max(0.0, 0.004032342 - Q.C2_b2) / 0.002883542   # -0.9%  C2_b2 < 0.004032
        + 0.0088164 * max(0.0, 0.1115136 - Q.planar_flow) / 0.03343187   # +0.9%  planar_flow < 0.1115
        - 0.008393354 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.8%  planar_flow < 0.1115 and centroid_offset < 0.01838
        - 0.008310702 * max(0.0, Q.e2_sq - 0.002074109) * max(0.0, 0.1872617 - Q.sj2_dr) / 2.014151e-05   # -0.8%  e2_sq > 0.002074 and sj2_dr < 0.1873
        + 0.006726086 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.width) / 3.351526e-05   # +0.7%  planar_flow < 0.1115 and width < 0.006097
        - 0.006720634 * max(0.0, Q.e2_sq - 0.01165737) / 0.001433269   # -0.7%  e2_sq > 0.01166
        + 0.005860439 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067297   # +0.6%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        - 0.005060665 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # -0.5%  z_dr_0p05_0p1 > 0.7509
        - 0.005020172 * max(0.0, Q.sj2_dr - 0.06154135) / 0.1002688   # -0.5%  sj2_dr > 0.06154
        + 0.004637632 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # +0.5%  e2 > 0.03556
        + 0.003579619 * max(0.0, Q.psi_0p1 - 0.7143804) / 0.1805043   # +0.4%  psi_0p1 > 0.7144
        - 0.003052481 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.3724174   # -0.3%  z_dr_0p05_0p1 < 0.5883
        - 0.003007047 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 658.125 - Q.sum_pt_top5) / 2.950014   # -0.3%  planar_flow < 0.1115 and sum_pt_top5 < 658.1
        - 0.002586938 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.3%  centroid_offset > 0.0499
        - 0.002477832 * max(0.0, Q.psi_0p1 - 0.8155839) / 0.1042073   # -0.2%  psi_0p1 > 0.8156
        + 0.002374282 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1629004) / 4.043869e-07   # +0.2%  e3 < 5.335e-05 and sj3_dr23 > 0.1629
        - 0.001988836 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # -0.2%  girth > 0.1019
        + 0.001092855 * max(0.0, Q.mass_top5 - 49.18618) / 2.747087   # +0.1%  mass_top5 > 49.19
        - 0.001001676 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.1058993 - Q.z_5) / 0.0007616364   # -0.1%  sj2_dr > 0.2002 and z_5 < 0.1059
        - 0.0009629396 * max(0.0, Q.mass - 69.61135) / 2.406702   # -0.1%  mass > 69.61
        + 0.0008710727 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # +0.1%  lam2 < 0.0005373
        + 0.0008478698 * max(0.0, 74.57663 - Q.sd_mass) * max(0.0, Q.sj2_mass2 - 3.697444) / 7.806142   # +0.1%  sd_mass < 74.58 and sj2_mass2 > 3.697
        + 0.0008093393 * max(0.0, Q.centroid_offset - 0.02685622) / 0.003254849   # +0.1%  centroid_offset > 0.02686
        - 0.0006872948 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.001964442 - Q.zdr_7) / 5.711896e-06   # -0.1%  planar_flow < 0.1115 and zdr_7 < 0.001964
        + 0.0005959558 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 0.3201347 - Q.D3) / 0.004766833   # +0.1%  z_dr_0p05_0p1 < 0.5883 and D3 < 0.3201
        - 0.0002395018 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, -0.05200653 - Q.phi_5) / 1.213647e-07   # -0.0%  e3 < 5.335e-05 and phi_5 < -0.05201
        - 0.0002246854 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, 0.05268713 - Q.dr_3) / 7.198945e-06   # -0.0%  centroid_offset > 0.02686 and dr_3 < 0.05269
        - 0.0001958637 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, -0.08051147 - Q.phi_7) / 0.0002204999   # -0.0%  planar_flow < 0.1115 and phi_7 < -0.08051
        - 0.0001386521 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 0.002231562   # -0.0%  planar_flow < 0.1115 and z_dr_0p05_0p1 > 0.6748
        + 7.687919e-05 * max(0.0, Q.psi_0p1 - 0.8155839) * max(0.0, Q.phi_4 - 0.1096802) / 6.769968e-05   # +0.0%  psi_0p1 > 0.8156 and phi_4 > 0.1097
        - 8.882713e-06 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.n_dr_0p2_0p4 - 2.0) / 1.484336e-07   # -0.0%  e3 < 5.335e-05 and n_dr_0p2_0p4 > 2
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 29.94;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.93648 * (-0.006223341
        + 0.1444431 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +14.4%  width < 0.01324
        + 0.113908 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +11.4%  mass_over_sum_pt_sq < 0.007183
        - 0.1024583 * max(0.0, 0.007520088 - Q.girth2) / 0.00354499   # -10.2%  girth2 < 0.00752
        - 0.101138 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -10.1%  e2_sq < 0.008169
        + 0.06754692 * max(0.0, 0.01882765 - Q.width) / 0.01314296   # +6.8%  width < 0.01883
        + 0.06560665 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +6.6%  sj2_dr < 0.1592
        - 0.06299997 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # -6.3%  mass_over_sum_pt < 0.1309
        - 0.05989866 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # -6.0%  e2_sq < 0.01166
        - 0.04985526 * max(0.0, 0.2001708 - Q.sj2_dr) / 0.07331403   # -5.0%  sj2_dr < 0.2002
        - 0.04392265 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # -4.4%  girth < 0.1019
        + 0.04374538 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +4.4%  e2 < 0.04111
        - 0.04017747 * max(0.0, 0.1492731 - Q.sj2_dr) / 0.04374278   # -4.0%  sj2_dr < 0.1493
        + 0.01992912 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +2.0%  lam1 < 0.00484
        - 0.01932944 * max(0.0, 0.00609665 - Q.width) / 0.002544039   # -1.9%  width < 0.006097
        - 0.01085908 * max(0.0, 0.007520088 - Q.girth2) * max(0.0, 0.7459513 - Q.D2) / 9.127003e-05   # -1.1%  girth2 < 0.00752 and D2 < 0.746
        + 0.009389836 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +0.9%  lam2 < 0.001131
        - 0.008378808 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # -0.8%  girth2_top3 < 0.002152
        + 0.007153514 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +0.7%  N2 < 0.2233 and sum_pt_top5 > 430.8
        + 0.007061016 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2) / 0.003993721   # +0.7%  mass_over_sum_pt < 0.1309 and D2 < 0.746
        - 0.006904316 * max(0.0, 0.0005124533 - Q.girth2_top2) / 0.0001104529   # -0.7%  girth2_top2 < 0.0005125
        - 0.00417434 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.0114178   # -0.4%  N2 < 0.2233 and z_dr_0p05_0p1 < 0.5883
        + 0.002383117 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.1094252   # +0.2%  N2 < 0.2233 and n_dr_0p1_0p2 < 4
        + 0.001600729 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +0.2%  lam1 < 0.002464
        + 0.001571556 * max(0.0, 0.00609665 - Q.width) * max(0.0, 0.7459513 - Q.D2) / 3.18925e-05   # +0.2%  width < 0.006097 and D2 < 0.746
        - 0.001281452 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.121681 - Q.max_dr) / 0.0004905272   # -0.1%  N2 < 0.2233 and max_dr < 0.1217
        + 0.001167429 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 716.8828 - Q.sum_pt_top5) / 7.315623   # +0.1%  N2 < 0.2233 and sum_pt_top5 < 716.9
        + 0.0008685073 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +0.1%  N2 < 0.2233
        - 0.000612487 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sj3_pairmax_over_m - 0.9367476) / 0.0001955197   # -0.1%  N2 < 0.2233 and sj3_pairmax_over_m > 0.9367
        - 0.0005264215 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 1.54073 - Q.pt_entropy) / 0.000718138   # -0.1%  N2 < 0.2233 and pt_entropy < 1.541
        + 0.0004502421 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.4684459) / 0.0001955483   # +0.0%  mass_over_sum_pt < 0.1309 and z_dr_0p1_0p2 > 0.4684
        + 0.0004328738 * max(0.0, 0.03043859 - Q.sj3_pairmin_over_m) / 0.0003673495   # +0.0%  sj3_pairmin_over_m < 0.03044
        + 0.0002252444 * max(0.0, Q.sj3_pair_mass_max - 80.4) / 0.3254107   # +0.0%  sj3_pair_mass_max > 80.4
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [2.4142550420168067, 1.7863600840336133, 2.950650105042017, 1.8549308823529411, 2.6522693277310925, 4.352203151260504, 3.0219069327731094, 3.6214155462184876, 0.5377109243697479, 3.2190886554621847, 4.066521428571429, 4.625032457983194, 0.1289659663865546, 5.10598781512605, 0.8420441176470588, 0.4404323529411765]
T = [4.125605165933561, 2.2171940011160713, 6.81247645417542, 5.137073312762605, 5.337797236081933]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +31%, n5 -20%, n1 +17%, n9 +13%, n0 -9%, n6 +8% ...
            + 0.3073143 * h[2] / H_AVG[2]
            - 0.1977984 * h[5] / H_AVG[5]
            + 0.1691381 * h[1] / H_AVG[1]
            + 0.134109 * h[9] / H_AVG[9]
            - 0.09143564 * h[0] / H_AVG[0]
            + 0.08011457 * h[6] / H_AVG[6]
            - 0.02009 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +37%, n10 -23%, n6 +17%, n4 -11%, n5 +9%, n8 +2% ...
            + 0.3686401 * h[9] / H_AVG[9]
            - 0.2292606 * h[10] / H_AVG[10]
            + 0.1703678 * h[6] / H_AVG[6]
            - 0.1121464 * h[4] / H_AVG[4]
            + 0.09201248 * h[5] / H_AVG[5]
            + 0.01515742 * h[8] / H_AVG[8]
            + 0.01241525 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +25%, n6 -14%, n3 -14%, n0 +12%, n7 +12%, n14 -9% ...
            + 0.2545898 * h[11] / H_AVG[11]
            - 0.1386201 * h[6] / H_AVG[6]
            - 0.1361422 * h[3] / H_AVG[3]
            + 0.1218206 * h[0] / H_AVG[0]
            + 0.1162844 * h[7] / H_AVG[7]
            - 0.09270243 * h[14] / H_AVG[14]
            + 0.0526996 * h[13] / H_AVG[13]
            - 0.04444746 * h[15] / H_AVG[15]
            - 0.01973258 * h[8] / H_AVG[8]
            - 0.01476651 * h[9] / H_AVG[9]
            - 0.008194341 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +33%, n6 -22%, n3 -20%, n14 +6%, n13 +5%, n1 +4% ...
            + 0.3304486 * h[7] / H_AVG[7]
            - 0.2205955 * h[6] / H_AVG[6]
            - 0.2031115 * h[3] / H_AVG[3]
            + 0.06146818 * h[14] / H_AVG[14]
            + 0.05435657 * h[13] / H_AVG[13]
            + 0.04346736 * h[1] / H_AVG[1]
            + 0.04033591 * h[4] / H_AVG[4]
            - 0.01958246 * h[9] / H_AVG[9]
            - 0.01339626 * h[15] / H_AVG[15]
            + 0.01323773 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -39%, n10 +29%, n5 -20%, n4 +6%, n3 +2%, n8 +2% ...
            - 0.3886074 * h[13] / H_AVG[13]
            + 0.2856882 * h[10] / H_AVG[10]
            - 0.2038389 * h[5] / H_AVG[5]
            + 0.06211058 * h[4] / H_AVG[4]
            + 0.02171929 * h[3] / H_AVG[3]
            + 0.01888809 * h[8] / H_AVG[8]
            - 0.01208045 * h[12] / H_AVG[12]
            + 0.007067098 * h[0] / H_AVG[0]
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
