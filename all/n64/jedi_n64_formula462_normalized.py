"""JEDI-linear jet tagger, 64 particles, 3 features: one term per observable per neuron (from the 690), re-tuned on the network's predictions (all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  1:  13.0%   (on for 98% of jets)
  neuron  8:  12.8%   (on for 39% of jets)
  neuron  5:  12.3%   (on for 78% of jets)
  neuron  4:   9.1%   (on for 68% of jets)
  neuron  9:   7.7%   (on for 78% of jets)
  neuron 13:   7.2%   (on for 77% of jets)
  neuron  0:   6.7%   (on for 55% of jets)
  neuron 10:   5.7%   (on for 78% of jets)
  neuron  6:   4.9%   (on for 60% of jets)
  neuron 12:   4.1%   (on for 41% of jets)
  neuron  3:   4.1%   (on for 65% of jets)
  neuron  7:   4.1%   (on for 43% of jets)
  neuron 11:   3.8%   (on for 73% of jets)
  neuron 15:   2.3%   (on for 50% of jets)
  neuron 14:   1.8%   (on for 31% of jets)
  neuron  2:   0.5%   (on for 36% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.0% (the network: 81.1%); same class as the network for 93.1% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_13         mass of particles 0 and 13 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
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
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_11                  pT of particle 11 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.sj3_z2                 pT share of subjet 2 of 3 (by pT)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its term in ΣzΔR)
  Q.zdr_1                  pT share × ΔR of particle 1 (its term in ΣzΔR)
  Q.zdr_11                 pT share × ΔR of particle 11 (its term in ΣzΔR)
  Q.zdr_4                  pT share × ΔR of particle 4 (its term in ΣzΔR)
  Q.zdr_5                  pT share × ΔR of particle 5 (its term in ΣzΔR)
  Q.z_2nd                  2nd-largest pT share
  Q.pt2_over_pt0           pT2 / pT0
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft10_dr0             ΔR between the hardest and the 10. softest real particle (0 if among the 15 hardest)
  Q.soft3_dr0              ΔR between the hardest and the 3. softest real particle (0 if among the 15 hardest)
  Q.dr1_12                 ΔR between particle 12 and the 2nd-hardest particle
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_9                   ΔR of particle 9 from the jet axis
  Q.soft3_dr               ΔR from the jet axis of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.phi_0                  Δφ of particle 0
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
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
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
  Q.tau43                  N-subjettiness τ4/τ3
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
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_13=pair_mass(0, 13),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
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
        n_for_90pct=ncum(0.9),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_11=pt[11],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_6=pt[6],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        z_11=z[11],
        z_6=z[6],
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft7_z=softp(7, 'z'),
        soft8_z=softp(8, 'z'),
        sj3_z2=subjets(3)["z"][1],
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_11=z[11] * dr[11],
        zdr_4=z[4] * dr[4],
        zdr_5=z[5] * dr[5],
        z_2nd=zs[1],
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft10_dr0=softp(10, 'dr0'),
        soft3_dr0=softp(3, 'dr0'),
        dr1_12=math.sqrt(dist2(1, 12)) if pt[12] > 0 else 0.0,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_9=dr[9] if pt[9] > 0 else 0.0,
        soft3_dr=softp(3, 'dr'),
        eta_0=eta[0],
        eta_1=eta[1],
        phi_0=phi[0],
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
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
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
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 10.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.43674 * (-0.05964834
        - 0.3628512 * max(0.0, Q.mass - 87.3546) / 16.58154   # -36.3%  mass > 87.35
        + 0.1928284 * max(0.0, 0.007514722 - Q.mass_over_sum_pt_sq) / 0.002018138   # +19.3%  mass_over_sum_pt_sq < 0.007515
        + 0.08840871 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +8.8%  sum_pt_top50 < 1157
        - 0.08161008 * max(0.0, 0.00616708 - Q.sum_zz_dr2) / 0.001295555   # -8.2%  sum_zz_dr2 < 0.006167
        + 0.03542992 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +3.5%  n_dr_0p2_0p4 < 15
        - 0.03356675 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # -3.4%  sum_pt_top40 < 1070
        - 0.03098452 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -3.1%  tau1 < 0.07057
        - 0.02862524 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # -2.9%  z_dr_0p2_0p4 < 0.09123
        + 0.02341477 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +2.3%  log_sum_pt < 7.017
        - 0.02336019 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -2.3%  lam1 < 0.005914
        - 0.02086718 * max(0.0, 0.00581533 - Q.sum_z_dr2_top30) / 0.001404756   # -2.1%  sum_z_dr2_top30 < 0.005815
        + 0.0167321 * max(0.0, 80.4 - Q.mass_top30) / 14.65577   # +1.7%  mass_top30 < 80.4
        - 0.01536177 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -1.5%  sum_pt < 1013
        - 0.0130874 * max(0.0, 0.9906378 - Q.z_top50_slots) / 0.004783315   # -1.3%  z_top50_slots < 0.9906
        - 0.01269474 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # -1.3%  mass_top40 < 80.89
        + 0.007589516 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +0.8%  psi_0p3 > 0.9956
        + 0.004890852 * max(0.0, 846.1934 - Q.sum_pt_top20) / 22.83716   # +0.5%  sum_pt_top20 < 846.2
        + 0.004274399 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957) / 1.852228   # +0.4%  log_sum_pt < 6.989 and sum_pt_top50 > 959.1
        - 0.002048883 * max(0.0, Q.mass_top50 - 80.38538) / 18.58031   # -0.2%  mass_top50 > 80.39
        - 0.001373346 * max(0.0, 0.008023552 - Q.sum_z_dr2_top20) / 0.003092749   # -0.1%  sum_z_dr2_top20 < 0.008024
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 15.54;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.53862 * (-0.08625485
        + 0.1378966 * max(0.0, Q.log_sum_pt - 6.811325) / 0.1381714   # +13.8%  log_sum_pt > 6.811
        + 0.09607626 * max(0.0, 0.1953848 - Q.tau1) / 0.1097382   # +9.6%  tau1 < 0.1954
        + 0.07235508 * max(0.0, Q.pt_entropy - 2.07371) / 0.8049786   # +7.2%  pt_entropy > 2.074
        + 0.05883062 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +5.9%  n_particles > 38
        - 0.05403381 * max(0.0, 120.6 - Q.mass) / 38.8279   # -5.4%  mass < 120.6
        - 0.04720197 * max(0.0, Q.z_top50_slots - 0.9586536) / 0.03427112   # -4.7%  z_top50_slots > 0.9587
        + 0.0373044 * max(0.0, 0.008031986 - Q.sum_z_dr2_top40) / 0.002514593   # +3.7%  sum_z_dr2_top40 < 0.008032
        - 0.0317703 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # -3.2%  mass_over_sum_pt_sq < 0.009595
        + 0.02532938 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +2.5%  sum_pt < 1017
        - 0.02341982 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # -2.3%  sum_pt_top40 < 1070
        - 0.02328189 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2) / 190.5117   # -2.3%  sj3_mass1 < 32.5 and sj3_mass2 < 18.68
        + 0.02231509 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # +2.2%  LHA < 0.2601
        - 0.02218232 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -2.2%  sum_pt_top50 > 959.1
        + 0.02030464 * max(0.0, Q.z_top20_slots - 0.8965411) / 0.0341307   # +2.0%  z_top20_slots > 0.8965
        + 0.01885256 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.2734   # +1.9%  sum_pt_top2 < 689.2
        - 0.01877952 * max(0.0, 47.88842 - Q.mass_top20) / 6.040308   # -1.9%  mass_top20 < 47.89
        - 0.01696082 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt) / 15.05788   # -1.7%  n_particles > 38 and soft1_pt < 2.275
        + 0.01543435 * max(0.0, 0.02412652 - Q.sum_z_dr2_top30) / 0.01613825   # +1.5%  sum_z_dr2_top30 < 0.02413
        + 0.01487145 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2) / 0.0008500795   # +1.5%  z_top30_slots > 0.9342 and C2 < 0.0728
        + 0.01314539 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.3%  z_dr_0p2_0p4 < 0.09123
        - 0.01301024 * max(0.0, 31.35938 - Q.pt_9) / 6.538304   # -1.3%  pt_9 < 31.36
        + 0.01298719 * max(0.0, 0.04516808 - Q.tau3) / 0.02499032   # +1.3%  tau3 < 0.04517
        - 0.01087589 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # -1.1%  e2 < 0.02516
        - 0.0104212 * max(0.0, Q.z_top20_slots - 0.8965411) * max(0.0, 0.03843804 - Q.dr_2) / 0.0005290849   # -1.0%  z_top20_slots > 0.8965 and dr_2 < 0.03844
        - 0.01033495 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # -1.0%  M3 < 0.03188
        - 0.0100228 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -1.0%  n_dr_0p2_0p4 < 7
        - 0.009930771 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -1.0%  psi_0p3 > 0.998
        - 0.00951642 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # -1.0%  sj3_mass1 < 32.5
        - 0.008974574 * max(0.0, 30.26161 - Q.sj2_mass1) / 7.997784   # -0.9%  sj2_mass1 < 30.26
        + 0.008552764 * max(0.0, 0.002830721 - Q.sum_z_dr2_top20) / 0.0005452981   # +0.9%  sum_z_dr2_top20 < 0.002831
        + 0.007854945 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 39.23115   # +0.8%  mass_top20 < 47.89 and n_real_top40 > 29
        + 0.007766234 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0) / 0.570275   # +0.8%  n_particles > 38 and dr_0 < 0.112
        - 0.007523787 * max(0.0, Q.pt_entropy - 2.07371) * max(0.0, 0.3797 - Q.soft3_dr) / 0.1539306   # -0.8%  pt_entropy > 2.074 and soft3_dr < 0.3797
        + 0.00719516 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1) / 1.005598   # +0.7%  n_particles > 38 and dr_1 < 0.1612
        + 0.007086209 * max(0.0, 12.0 - Q.n_dr_0_0p05) / 3.466761   # +0.7%  n_dr_0_0p05 < 12
        - 0.006355601 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, Q.soft3_dr0 - 0.02918107) / 57.76427   # -0.6%  sum_pt_top2 < 689.2 and soft3_dr0 > 0.02918
        - 0.006312464 * max(0.0, Q.z_top30_slots - 0.9341838) / 0.03406818   # -0.6%  z_top30_slots > 0.9342
        - 0.005857697 * max(0.0, Q.n_for_90pct - 11.0) / 10.33888   # -0.6%  n_for_90pct > 11
        + 0.005516401 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793) / 0.1941151   # +0.6%  z_top30_slots > 0.9342 and max_pair_mass > 13.05
        + 0.005419442 * max(0.0, 0.1416054 - Q.D3) / 0.05511785   # +0.5%  D3 < 0.1416
        - 0.005275944 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.5%  n_dr_0p1_0p2 < 8
        + 0.005262429 * max(0.0, 0.007538019 - Q.sum_z_dr2_top20) * max(0.0, Q.eta_1 - -0.04302979) / 0.000120198   # +0.5%  sum_z_dr2_top20 < 0.007538 and eta_1 > -0.04303
        + 0.00515371 * max(0.0, 120.6 - Q.mass) * max(0.0, 0.03843804 - Q.dr_2) / 0.6524502   # +0.5%  mass < 120.6 and dr_2 < 0.03844
        + 0.005142267 * max(0.0, 0.0007894752 - Q.sum_z_dr2_top15) / 9.062109e-05   # +0.5%  sum_z_dr2_top15 < 0.0007895
        - 0.004103388 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -0.4%  z_dr_0_0p05 > 0.8789
        + 0.004021167 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 0.3241858 - Q.dr1_12) / 1.371943   # +0.4%  pt_9 < 31.36 and dr1_12 < 0.3242
        + 0.003933308 * max(0.0, Q.mass_top10 - 56.92192) / 8.071231   # +0.4%  mass_top10 > 56.92
        + 0.00358049 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 11.29505 - Q.pair_mass_0_13) / 38.90723   # +0.4%  pt_9 < 31.36 and pair_mass_0_13 < 11.3
        + 0.003489674 * max(0.0, Q.psi_0p1 - 0.8747961) / 0.03440015   # +0.3%  psi_0p1 > 0.8748
        + 0.003129248 * max(0.0, 30.26161 - Q.sj2_mass1) * max(0.0, 0.2366434 - Q.soft10_dr0) / 1.029958   # +0.3%  sj2_mass1 < 30.26 and soft10_dr0 < 0.2366
        - 0.003092379 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 0.02980347 - Q.eta_0) / 5.83951   # -0.3%  sum_pt_top5 > 430.8 and eta_0 < 0.0298
        + 0.002711523 * max(0.0, 0.0006570502 - Q.sum_z_dr2_top5) / 0.0001395474   # +0.3%  sum_z_dr2_top5 < 0.0006571
        + 0.002450432 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874) / 0.05244257   # +0.2%  z_top30_slots > 0.9342 and ptdr0_3 > 7.408
        + 0.002410919 * max(0.0, Q.soft1_pt - 1.505859) / 0.1017706   # +0.2%  soft1_pt > 1.506
        + 0.002104757 * max(0.0, 0.003270031 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 6.08612e-07   # +0.2%  sum_z_dr2_top15 < 0.00327 and psi_0p3 > 0.9974
        - 0.001810114 * max(0.0, Q.sum_pt_top30 - 1191.938) / 6.938323   # -0.2%  sum_pt_top30 > 1192
        + 0.0004672872 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.0%  lam1 < 0.004673
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 12.45;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.44864 * (-0.6979967
        + 0.6898324 * Q.sum_pt / 1043.796   # +69.0%  sum_pt
        + 0.06823479 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.sum_z_dr2_top15) / 0.0009227558   # +6.8%  log_sum_pt > 6.903 and sum_z_dr2_top15 < 0.02147
        + 0.05636567 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # +5.6%  mass_top50 < 92.17
        - 0.05262402 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # -5.3%  mass_over_sum_pt < 0.09795
        - 0.03198514 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.sum_z_dr2_top15) / 1.045047   # -3.2%  sum_pt_top50 > 988.5 and sum_z_dr2_top15 < 0.02147
        - 0.02828147 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -2.8%  mass < 92.86
        - 0.02746767 * max(0.0, Q.log_sum_pt - 6.971093) / 0.02523845   # -2.7%  log_sum_pt > 6.971
        + 0.019161 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +1.9%  lam1 < 0.01174
        + 0.007549787 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 4.657698e-07   # +0.8%  sum_pt < 972 and e4 < 5.851e-08
        - 0.006408715 * max(0.0, 1011.14 - Q.sum_pt_top30) / 50.79521   # -0.6%  sum_pt_top30 < 1011
        - 0.006350336 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003463934   # -0.6%  log_sum_pt > 6.903 and psi_0p3 > 0.9299
        + 0.003802007 * max(0.0, 1018.525 - Q.sum_pt_top40) / 34.79588   # +0.4%  sum_pt_top40 < 1019
        - 0.001868963 * max(0.0, Q.sum_pt_top50 - 1060.151) / 28.66803   # -0.2%  sum_pt_top50 > 1060
        - 6.801552e-05 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.08734131) / 0.003497383   # -0.0%  sum_pt_top20 > 1129 and eta_1 > 0.08734
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 2.74;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.739731 * (-0.03507953
        + 0.1432332 * max(0.0, 0.008840538 - Q.sum_z_dr2_top40) / 0.003108622   # +14.3%  sum_z_dr2_top40 < 0.008841
        - 0.1125247 * max(0.0, 2.410481 - Q.D2) / 0.587301   # -11.3%  D2 < 2.41
        + 0.09154942 * max(0.0, 2.410481 - Q.D2) * max(0.0, 5.142486 - Q.D2_b2) / 2.680544   # +9.2%  D2 < 2.41 and D2_b2 < 5.142
        - 0.06809066 * max(0.0, 0.009614971 - Q.sum_z_dr2) / 0.003502545   # -6.8%  sum_z_dr2 < 0.009615
        + 0.0630265 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741) / 0.0491674   # +6.3%  n_dr_0p2_0p4 < 8 and z_top50_slots > 0.9787
        - 0.06099427 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -6.1%  tau21 < 0.3862
        - 0.05758207 * max(0.0, 79.21004 - Q.mass_top50) / 10.65754   # -5.8%  mass_top50 < 79.21
        - 0.05120998 * max(0.0, 80.4 - Q.mass_top40) / 12.19371   # -5.1%  mass_top40 < 80.4
        + 0.04805622 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +4.8%  n_particles < 46
        + 0.04769468 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +4.8%  lam2 < 0.0006155
        - 0.04518892 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243) / 0.0001457263   # -4.5%  tau21 < 0.3862 and lam1 > 0.007671
        + 0.03606722 * max(0.0, 27.56535 - Q.sj2_mass1) / 6.309897   # +3.6%  sj2_mass1 < 27.57
        + 0.03369223 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +3.4%  n_dr_0p2_0p4 < 5
        + 0.03081156 * max(0.0, 0.008124776 - Q.sum_z_dr2_top50) / 0.002470567   # +3.1%  sum_z_dr2_top50 < 0.008125
        + 0.02810285 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1) / 0.1325294   # +2.8%  n_dr_0p2_0p4 < 5 and psi_0p1 < 0.9538
        + 0.02387063 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.9313699) / 0.1032909   # +2.4%  n_dr_0p1_0p2 < 10 and psi_0p2 > 0.9314
        - 0.0211959 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5) / 0.004484684   # -2.1%  n_dr_0p2_0p4 < 5 and zdr_5 < 0.007099
        + 0.01655914 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +1.7%  tau4 < 0.01627
        - 0.009313497 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0) / 3.186294   # -0.9%  n_dr_0p2_0p4 < 5 and n_dr_0p1_0p2 > 9
        - 0.007466273 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349) / 64.17204   # -0.7%  n_particles < 46 and mass_top15 > 57.87
        + 0.002149984 * max(0.0, 0.08665515 - Q.mass_over_sum_pt) / 0.01542259   # +0.2%  mass_over_sum_pt < 0.08666
        + 0.001620147 * max(0.0, 53.52607 - Q.mass) / 3.497069   # +0.2%  mass < 53.53
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 5.589;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.588905 * (0.0693262
        - 0.2356294 * max(0.0, 74.25181 - Q.mass) / 8.587064   # -23.6%  mass < 74.25
        - 0.1101925 * max(0.0, Q.n_particles - 22.0) / 23.9717   # -11.0%  n_particles > 22
        + 0.08500738 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +8.5%  n_dr_0p2_0p4 < 26
        + 0.07280535 * max(0.0, Q.sum_zz_dr2 - 0.009606007) / 0.003182984   # +7.3%  sum_zz_dr2 > 0.009606
        + 0.0546399 * max(0.0, 57.37264 - Q.mass_top40) / 4.890706   # +5.5%  mass_top40 < 57.37
        + 0.05447309 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +5.4%  psi_0p3 > 0.9974
        - 0.05297075 * max(0.0, 0.03552114 - Q.sum_z_dr) / 0.004120775   # -5.3%  sum_z_dr < 0.03552
        + 0.05089001 * max(0.0, 0.02429622 - Q.sum_z_dr2_top5) / 0.01878229   # +5.1%  sum_z_dr2_top5 < 0.0243
        - 0.04968114 * max(0.0, 0.004175169 - Q.sum_z_dr2_top15) / 0.001132783   # -5.0%  sum_z_dr2_top15 < 0.004175
        - 0.0416667 * max(0.0, Q.lam1 - 0.007259145) / 0.002883149   # -4.2%  lam1 > 0.007259
        - 0.02734692 * max(0.0, 80.78464 - Q.mass) * max(0.0, 3.852812 - Q.D2_b2) / 3.983646   # -2.7%  mass < 80.78 and D2_b2 < 3.853
        + 0.02552295 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt) / 36.47388   # +2.6%  n_particles > 22 and soft1_pt < 2.275
        + 0.01778012 * max(0.0, Q.sum_pt - 995.6769) / 62.24161   # +1.8%  sum_pt > 995.7
        + 0.01591705 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.005180665 - Q.sum_z_dr2_top2) / 0.07860143   # +1.6%  mass < 92.86 and sum_z_dr2_top2 < 0.005181
        + 0.01560558 * max(0.0, 57.91414 - Q.mass_top15) / 12.47898   # +1.6%  mass_top15 < 57.91
        - 0.01525651 * max(0.0, 100.7509 - Q.mass_top30) / 29.10389   # -1.5%  mass_top30 < 100.8
        - 0.01279972 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 56.19962   # -1.3%  n_dr_0p2_0p4 < 26 and n_dr_0p1_0p2 > 11
        + 0.01143476 * max(0.0, 0.9909875 - Q.psi_0p1) / 0.2210377   # +1.1%  psi_0p1 < 0.991
        - 0.01098706 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 20.40995 - Q.D2_b2) / 0.3948355   # -1.1%  sj2_dr > 0.2232 and D2_b2 < 20.41
        + 0.01058945 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.n_dr_0_0p05 - 10.0) / 143.1065   # +1.1%  n_particles > 22 and n_dr_0_0p05 > 10
        - 0.008252644 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 13.0 - Q.n_dr_0_0p05) / 0.003836025   # -0.8%  psi_0p3 > 0.9974 and n_dr_0_0p05 < 13
        - 0.005761011 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # -0.6%  C2 > 0.06656
        - 0.004062682 * max(0.0, Q.n_particles - 22.0) * max(0.0, 3.852812 - Q.D2_b2) / 43.35921   # -0.4%  n_particles > 22 and D2_b2 < 3.853
        + 0.00389316 * max(0.0, Q.log_sum_pt - 6.97212) / 0.02498281   # +0.4%  log_sum_pt > 6.972
        + 0.002561386 * max(0.0, Q.sj2_dr - 0.1666353) / 0.04983795   # +0.3%  sj2_dr > 0.1666
        - 0.00238638 * max(0.0, Q.e2 - 0.06510219) / 0.0004144068   # -0.2%  e2 > 0.0651
        + 0.001725396 * max(0.0, Q.e3 - 0.0001841806) / 4.035159e-05   # +0.2%  e3 > 0.0001842
        - 0.0001610327 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 3.014827 - Q.D2_b2) / 4.336836   # -0.0%  mass_top40 < 83.33 and D2_b2 < 3.015
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.69;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.69037 * (0.1029901
        + 0.1959138 * max(0.0, Q.sum_pt_top50 - 935.0507) / 107.6589   # +19.6%  sum_pt_top50 > 935.1
        - 0.130703 * max(0.0, Q.log_sum_pt - 6.903354) / 0.05667477   # -13.1%  log_sum_pt > 6.903
        + 0.06356676 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +6.4%  n_particles < 64
        - 0.05166117 * max(0.0, Q.sum_pt - 1012.538) / 50.74831   # -5.2%  sum_pt > 1013
        - 0.04723195 * max(0.0, Q.mass_over_sum_pt - 0.07437106) / 0.02187741   # -4.7%  mass_over_sum_pt > 0.07437
        + 0.04625865 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +4.6%  sum_pt > 907.9 and e4 < 5.851e-08
        - 0.03230471 * max(0.0, 150.0144 - Q.mass_top40) / 66.86198   # -3.2%  mass_top40 < 150
        + 0.0294246 * max(0.0, Q.sum_z_dr2 - 0.004756928) / 0.005348122   # +2.9%  sum_z_dr2 > 0.004757
        - 0.02773356 * max(0.0, Q.sum_z_dr2_top50 - 0.01951641) / 0.001208668   # -2.8%  sum_z_dr2_top50 > 0.01952
        - 0.02608362 * max(0.0, 0.01083435 - Q.sum_z_dr2_top20) / 0.00529825   # -2.6%  sum_z_dr2_top20 < 0.01083
        - 0.02451813 * max(0.0, Q.z_top50_slots - 0.9906378) / 0.006446962   # -2.5%  z_top50_slots > 0.9906
        + 0.02246974 * max(0.0, Q.sum_pt_top30 - 933.1875) / 79.88556   # +2.2%  sum_pt_top30 > 933.2
        - 0.02218891 * max(0.0, Q.z_top30_slots - 0.9048492) / 0.05621641   # -2.2%  z_top30_slots > 0.9048
        + 0.02205483 * max(0.0, 0.4367319 - Q.max_dr) / 0.09131945   # +2.2%  max_dr < 0.4367
        - 0.02187797 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -2.2%  mass_top50 > 157.5
        - 0.02085946 * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 9.82015   # -2.1%  n_dr_0p1_0p2 < 21
        + 0.02019267 * max(0.0, Q.sum_z_dr2_top30 - 0.005417092) / 0.004308041   # +2.0%  sum_z_dr2_top30 > 0.005417
        - 0.01845206 * max(0.0, Q.sum_pt_top40 - 1007.262) / 43.89205   # -1.8%  sum_pt_top40 > 1007
        - 0.01752266 * max(0.0, Q.mass - 172.8) / 0.7291439   # -1.8%  mass > 172.8
        + 0.0156569 * max(0.0, 0.04334991 - Q.e2) / 0.0152982   # +1.6%  e2 < 0.04335
        - 0.01525157 * max(0.0, 787.6281 - Q.sum_pt_top3) / 328.0371   # -1.5%  sum_pt_top3 < 787.6
        + 0.01473609 * max(0.0, 1.976207 - Q.D2) / 0.367701   # +1.5%  D2 < 1.976
        - 0.01355832 * max(0.0, Q.z_top20_slots - 0.9102775) / 0.02689878   # -1.4%  z_top20_slots > 0.9103
        + 0.01346558 * max(0.0, 0.01375115 - Q.z_11) / 0.0009450582   # +1.3%  z_11 < 0.01375
        - 0.01297015 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2) / 10.81389   # -1.3%  n_particles < 64 and D2 < 2.179
        - 0.01268616 * max(0.0, 0.5494307 - Q.tau21) / 0.1591168   # -1.3%  tau21 < 0.5494
        - 0.01246392 * max(0.0, 14.14062 - Q.pt_11) / 0.9740783   # -1.2%  pt_11 < 14.14
        - 0.01015865 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # -1.0%  psi_0p3 > 0.9974
        - 0.009573093 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -1.0%  mass_over_sum_pt_sq > 0.0292
        + 0.005975205 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # +0.6%  mass_top10 > 71.78
        + 0.005038974 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +0.5%  n_dr_0p2_0p4 < 11
        + 0.004836083 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2) / 0.001543279   # +0.5%  psi_0p3 > 0.9974 and D2 < 3.345
        + 0.003659337 * max(0.0, Q.n_pt_above_1 - 28.0) * max(0.0, 0.942303 - Q.tau43) / 1.657469   # +0.4%  n_pt_above_1 > 28 and tau43 < 0.9423
        + 0.003628832 * max(0.0, Q.z_dr_0_0p05 - 0.9084912) / 0.009227347   # +0.4%  z_dr_0_0p05 > 0.9085
        - 0.002727233 * max(0.0, 0.005788041 - Q.sum_z_dr2_top15) / 0.001877672   # -0.3%  sum_z_dr2_top15 < 0.005788
        - 0.002595663 * max(0.0, Q.sd_mass - 115.8435) / 2.045575   # -0.3%  sd_mass > 115.8
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 8.219;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.219015 * (0.1489055
        - 0.2075157 * max(0.0, 0.02809026 - Q.sum_z_dr2_top30) / 0.01980159   # -20.8%  sum_z_dr2_top30 < 0.02809
        + 0.09997832 * max(0.0, 0.04755309 - Q.e2) / 0.01877457   # +10.0%  e2 < 0.04755
        - 0.09729363 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -9.7%  mass_over_sum_pt > 0.1182
        + 0.07348668 * max(0.0, Q.mass - 86.4) / 17.01755   # +7.3%  mass > 86.4
        + 0.06664379 * max(0.0, 0.005623108 - Q.lam1) / 0.001326678   # +6.7%  lam1 < 0.005623
        - 0.06456786 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2) / 0.04240098   # -6.5%  mass_over_sum_pt > 0.1182 and D2_b2 < 7.366
        + 0.05789622 * max(0.0, Q.sum_z_dr2_top20 - 0.008031209) / 0.002906852   # +5.8%  sum_z_dr2_top20 > 0.008031
        - 0.05043844 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # -5.0%  e3 < 0.0003372
        + 0.04872396 * max(0.0, Q.sum_z_dr2_top40 - 0.008031986) / 0.003375597   # +4.9%  sum_z_dr2_top40 > 0.008032
        - 0.04431857 * max(0.0, Q.sj3_pair_mass_min - 60.17253) / 1.603492   # -4.4%  sj3_pair_mass_min > 60.17
        - 0.03891329 * max(0.0, Q.sum_z_dr2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529) / 8.566262e-05   # -3.9%  sum_z_dr2_top20 > 0.008031 and C2_b2 > 0.0008188
        + 0.02785505 * max(0.0, 0.003687605 - Q.lam2) / 0.002601874   # +2.8%  lam2 < 0.003688
        + 0.02772534 * max(0.0, Q.tau1 - 0.0920727) / 0.01829616   # +2.8%  tau1 > 0.09207
        - 0.0176215 * max(0.0, 0.004566318 - Q.sum_z_dr2_top50) / 0.0007717659   # -1.8%  sum_z_dr2_top50 < 0.004566
        - 0.01522363 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # -1.5%  sum_pt < 1261
        + 0.009944279 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # +1.0%  log_sum_pt < 6.811
        - 0.009389552 * max(0.0, Q.mass_over_sum_pt - 0.02682209) * max(0.0, 0.7108211 - Q.z_dr_0p05_0p1) / 0.02513932   # -0.9%  mass_over_sum_pt > 0.02682 and z_dr_0p05_0p1 < 0.7108
        + 0.007335586 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832) / 8.191828e-05   # +0.7%  mass_over_sum_pt > 0.1182 and C2_b2 > 0.02705
        + 0.006599416 * max(0.0, Q.sum_z_dr - 0.02085222) / 0.04959222   # +0.7%  sum_z_dr > 0.02085
        - 0.006562159 * max(0.0, Q.sum_z_dr2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722) / 0.001787472   # -0.7%  sum_z_dr2_top30 > 0.008376 and D2_b2 > 1.677
        - 0.005773936 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt) / 683.4113   # -0.6%  mass < 120.6 and sum_pt < 1008
        + 0.00481949 * max(0.0, Q.e2 - 0.02793599) * max(0.0, 0.4357228 - Q.max_dr) / 0.0007248268   # +0.5%  e2 > 0.02794 and max_dr < 0.4357
        - 0.003084964 * max(0.0, Q.mass_over_sum_pt - 0.09795415) * max(0.0, 0.001160626 - Q.soft8_z) / 8.555108e-07   # -0.3%  mass_over_sum_pt > 0.09795 and soft8_z < 0.001161
        + 0.002820053 * max(0.0, 0.9300465 - Q.z_top40_slots) / 0.002595351   # +0.3%  z_top40_slots < 0.93
        - 0.002022911 * max(0.0, Q.sj3_mass1 - 21.11128) / 1.638992   # -0.2%  sj3_mass1 > 21.11
        + 0.001827775 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.soft3_dr0 - 0.1966723) / 0.0006797808   # +0.2%  mass_over_sum_pt > 0.1182 and soft3_dr0 > 0.1967
        - 0.001240682 * max(0.0, 115.5062 - Q.mass_top50) / 35.7055   # -0.1%  mass_top50 < 115.5
        - 0.0003772456 * max(0.0, Q.sj3_pair_mass_min - 29.00832) * max(0.0, Q.psi_0p3 - 0.9896594) / 0.01664863   # -0.0%  sj3_pair_mass_min > 29.01 and psi_0p3 > 0.9897
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 7.128;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.127876 * (0.07232475
        - 0.1674399 * max(0.0, 87.3546 - Q.mass) / 14.19492   # -16.7%  mass < 87.35
        - 0.1398754 * max(0.0, 0.1199231 - Q.tau1) / 0.04418675   # -14.0%  tau1 < 0.1199
        - 0.1330815 * max(0.0, Q.sd_mass - 97.07106) / 4.424274   # -13.3%  sd_mass > 97.07
        + 0.1300798 * max(0.0, 0.01372009 - Q.sum_z_dr2) / 0.006695312   # +13.0%  sum_z_dr2 < 0.01372
        + 0.09741018 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +9.7%  tau21_b2 < 0.2352
        - 0.06227308 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt) / 11.66143   # -6.2%  tau21_b2 < 0.2352 and sum_pt < 1261
        + 0.04909473 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +4.9%  mass_over_sum_pt < 0.1182
        + 0.03303666 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # +3.3%  n_dr_0p2_0p4 < 8
        - 0.02685234 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # -2.7%  lam1 < 0.01174
        + 0.02615106 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.z_top50_slots - 0.978741) / 1.779326e-05   # +2.6%  psi_0p3 > 0.9974 and z_top50_slots > 0.9787
        - 0.02515482 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 5.106465e-05   # -2.5%  tau21_b2 < 0.2352 and mass_over_sum_pt_sq < 0.007873
        - 0.01988047 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.4385899   # -2.0%  mass < 101 and psi_0p3 > 0.9777
        - 0.01925158 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # -1.9%  psi_0p2 > 0.9314
        - 0.01856901 * max(0.0, 0.00363788 - Q.sum_zz_dr2) / 0.0004963098   # -1.9%  sum_zz_dr2 < 0.003638
        - 0.01721296 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.4947602 - Q.z_dr_0_0p05) / 0.572942   # -1.7%  mass < 92.86 and z_dr_0_0p05 < 0.4948
        - 0.01711392 * max(0.0, Q.mass_top20 - 103.0534) / 3.888733   # -1.7%  mass_top20 > 103.1
        + 0.01123008 * max(0.0, 0.1776078 - Q.sd_rg) / 0.06562807   # +1.1%  sd_rg < 0.1776
        - 0.004486464 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.4%  psi_0p3 > 0.998
        - 0.001805945 * max(0.0, 0.008241985 - Q.lam1) * max(0.0, Q.zdr_0 - 0.01564747) / 7.458275e-07   # -0.2%  lam1 < 0.008242 and zdr_0 > 0.01565
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 10.61;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.60799 * (0.07585297
        - 0.1820149 * max(0.0, 0.009614971 - Q.lam1_plus_lam2) / 0.003502545   # -18.2%  lam1_plus_lam2 < 0.009615
        + 0.1417145 * max(0.0, Q.mass - 64.06581) / 31.50753   # +14.2%  mass > 64.07
        + 0.08612563 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt) / 2.336698   # +8.6%  mass_over_sum_pt > 0.07697 and sum_pt < 1116
        + 0.06855495 * max(0.0, 1022.377 - Q.sum_pt) / 23.90959   # +6.9%  sum_pt < 1022
        - 0.0654686 * max(0.0, Q.z_top50_slots - 0.9704436) / 0.02331682   # -6.5%  z_top50_slots > 0.9704
        - 0.0474867 * max(0.0, 6.930088 - Q.log_sum_pt) / 0.02522479   # -4.7%  log_sum_pt < 6.93
        - 0.04151716 * max(0.0, 1032.05 - Q.sum_pt_top40) / 42.933   # -4.2%  sum_pt_top40 < 1032
        - 0.04040952 * max(0.0, Q.mass_over_sum_pt - 0.08330792) / 0.01699661   # -4.0%  mass_over_sum_pt > 0.08331
        - 0.03974667 * max(0.0, Q.mass_top50 - 77.33624) / 20.44301   # -4.0%  mass_top50 > 77.34
        + 0.03715579 * max(0.0, 0.007463985 - Q.sum_z_dr2_top30) / 0.00233708   # +3.7%  sum_z_dr2_top30 < 0.007464
        - 0.02795038 * max(0.0, Q.sum_z_dr2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt) / 0.0005328922   # -2.8%  sum_z_dr2_top40 > 0.005197 and log_sum_pt < 7.017
        - 0.02411872 * max(0.0, Q.e3 - 3.392004e-05) / 7.694005e-05   # -2.4%  e3 > 3.392e-05
        - 0.02389408 * max(0.0, 0.02146578 - Q.sum_z_dr2_top15) / 0.01470441   # -2.4%  sum_z_dr2_top15 < 0.02147
        + 0.02282286 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +2.3%  e2 > 0.04755
        + 0.02174131 * max(0.0, Q.n_dr_0p2_0p4 - 4.0) / 5.786086   # +2.2%  n_dr_0p2_0p4 > 4
        + 0.01831953 * max(0.0, Q.sum_pt_top30 - 978.0762) / 49.23409   # +1.8%  sum_pt_top30 > 978.1
        + 0.01645068 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +1.6%  lam1 < 0.007671
        - 0.0158442 * max(0.0, 0.00572743 - Q.sum_z_dr2_top40) / 0.001225729   # -1.6%  sum_z_dr2_top40 < 0.005727
        + 0.01523891 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +1.5%  sj2_dr > 0.2232
        - 0.01110491 * max(0.0, 0.007877041 - Q.sum_z_dr2) / 0.002242297   # -1.1%  sum_z_dr2 < 0.007877
        - 0.0105618 * max(0.0, Q.mass_top20 - 119.2969) / 1.8402   # -1.1%  mass_top20 > 119.3
        + 0.007178734 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # +0.7%  C2 > 0.06656
        + 0.005817658 * max(0.0, Q.n_dr_0p1_0p2 - 19.0) / 1.732133   # +0.6%  n_dr_0p1_0p2 > 19
        + 0.005264301 * max(0.0, 0.0795038 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.38202) / 0.1361121   # +0.5%  tau2 < 0.0795 and sj3_mass1 > 13.38
        - 0.004836312 * max(0.0, 0.007463985 - Q.sum_z_dr2_top30) * max(0.0, Q.sj2_dr - 0.1512157) / 6.78729e-05   # -0.5%  sum_z_dr2_top30 < 0.007464 and sj2_dr > 0.1512
        + 0.004413819 * max(0.0, Q.n_for_90pct - 16.0) / 6.639766   # +0.4%  n_for_90pct > 16
        - 0.003433155 * max(0.0, Q.mass - 64.48544) * max(0.0, Q.zdr_0 - 0.0008718296) / 0.4297651   # -0.3%  mass > 64.49 and zdr_0 > 0.0008718
        - 0.003314401 * max(0.0, 0.3628388 - Q.psi_0p1) / 0.02326895   # -0.3%  psi_0p1 < 0.3628
        + 0.002557977 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.09686657   # +0.3%  n_dr_0p2_0p4 < 15 and z_dr_0p1_0p2 > 0.334
        - 0.001352775 * max(0.0, 0.8316924 - Q.z_top15_slots) / 0.04702674   # -0.1%  z_top15_slots < 0.8317
        + 0.00131816 * max(0.0, Q.sum_pt_top50 - 889.8503) / 149.655   # +0.1%  sum_pt_top50 > 889.9
        - 0.0008715463 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.03720487   # -0.1%  z_dr_0p1_0p2 > 0.334
        - 0.0007065069 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.5693287   # -0.1%  sum_pt < 1017 and dr_max_012 > 0.1828
        + 0.0006928289 * max(0.0, Q.sum_z_dr2_top20 - 0.01062209) / 0.0023458   # +0.1%  sum_z_dr2_top20 > 0.01062
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 3.699;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.698654 * (0.180598
        - 0.1752453 * max(0.0, Q.mass_top50 - 71.69608) / 24.29469   # -17.5%  mass_top50 > 71.7
        + 0.1391727 * max(0.0, Q.mass_top40 - 80.89043) / 16.36297   # +13.9%  mass_top40 > 80.89
        + 0.105827 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +10.6%  mass < 82.85
        + 0.1042979 * max(0.0, 0.006628677 - Q.sum_z_dr2_top40) / 0.001656111   # +10.4%  sum_z_dr2_top40 < 0.006629
        - 0.1036787 * max(0.0, Q.sum_z_dr2_top30 - 0.0008564881) / 0.007661497   # -10.4%  sum_z_dr2_top30 > 0.0008565
        + 0.0809209 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +8.1%  sum_pt < 986.1
        - 0.06810234 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -6.8%  log_sum_pt < 6.903
        + 0.03854133 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40) / 104.3982   # +3.9%  mass_top40 < 80.89 and sum_pt_top40 < 906.6
        + 0.02868277 * max(0.0, 956.2133 - Q.sum_pt_top40) / 14.00451   # +2.9%  sum_pt_top40 < 956.2
        - 0.0276419 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -2.8%  tau1 < 0.05445
        + 0.02675806 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt) / 447.38   # +2.7%  mass_top40 < 80.89 and sum_pt < 1035
        - 0.02391256 * max(0.0, 73.33139 - Q.mass_top30) / 11.37701   # -2.4%  mass_top30 < 73.33
        + 0.02389757 * max(0.0, 125.1 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 42.49581   # +2.4%  mass_top40 < 125.1 and n_dr_0p2_0p4 > 9
        - 0.0214089 * max(0.0, Q.z_top40_slots - 0.9574183) / 0.02831517   # -2.1%  z_top40_slots > 0.9574
        + 0.01214745 * max(0.0, Q.D2 - 2.178951) / 1.450771   # +1.2%  D2 > 2.179
        + 0.007776021 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258) / 5.834847   # +0.8%  sum_pt_top40 < 956.2 and soft4_pt > 1.789
        - 0.004406402 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047) / 1.704178   # -0.4%  sum_pt_top40 < 956.2 and soft3_pt > 2.873
        + 0.003076068 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +0.3%  LHA > 0.4042
        + 0.002557896 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +0.3%  lam1 > 0.01174
        + 0.001948153 * max(0.0, Q.sum_z_dr - 0.1564422) / 0.0006635094   # +0.2%  sum_z_dr > 0.1564
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 11.04;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.03527 * (0.5303657
        - 0.2352743 * Q.mass / 89.74122   # -23.5%  mass
        - 0.192859 * max(0.0, 0.1207452 - Q.sum_z_dr) / 0.05578614   # -19.3%  sum_z_dr < 0.1207
        - 0.07848732 * max(0.0, 65.20727 - Q.sj2_mass1) / 36.50254   # -7.8%  sj2_mass1 < 65.21
        - 0.07035286 * max(0.0, Q.tau1 - 0.05426326) / 0.04003458   # -7.0%  tau1 > 0.05426
        + 0.06359139 * max(0.0, Q.e2 - 0.01256762) / 0.01912138   # +6.4%  e2 > 0.01257
        + 0.0451916 * max(0.0, 935.1043 - Q.sum_pt_top15) / 95.709   # +4.5%  sum_pt_top15 < 935.1
        + 0.03777557 * max(0.0, 0.04828819 - Q.tau2) / 0.02103241   # +3.8%  tau2 < 0.04829
        + 0.03212569 * max(0.0, 2.975532 - Q.D2) / 0.9290697   # +3.2%  D2 < 2.976
        - 0.02338521 * max(0.0, Q.psi_0p3 - 0.995608) / 0.001943985   # -2.3%  psi_0p3 > 0.9956
        + 0.02248816 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +2.2%  dr_0 < 0.06413
        - 0.01805062 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -1.8%  n_dr_0p2_0p4 < 11
        - 0.01492336 * max(0.0, 72.18744 - Q.mass_top15) / 20.20734   # -1.5%  mass_top15 < 72.19
        - 0.0141913 * max(0.0, Q.pt_dispersion - 0.27462) / 0.06481184   # -1.4%  pt_dispersion > 0.2746
        + 0.01322433 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +1.3%  mass_top50 > 136.8
        - 0.01190247 * max(0.0, 0.01976735 - Q.sum_z_dr2_top10) * max(0.0, Q.psi_0p3 - 0.9985421) / 6.34706e-06   # -1.2%  sum_z_dr2_top10 < 0.01977 and psi_0p3 > 0.9985
        + 0.01105952 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.1%  z_dr_0p2_0p4 < 0.09123
        - 0.01065586 * max(0.0, 0.01976735 - Q.sum_z_dr2_top10) * max(0.0, 0.4357228 - Q.max_dr) / 0.001259784   # -1.1%  sum_z_dr2_top10 < 0.01977 and max_dr < 0.4357
        - 0.01032174 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -1.0%  sum_pt_top10 > 943.7
        - 0.009600545 * max(0.0, Q.mass_over_sum_pt - 0.1606361) / 0.001344467   # -1.0%  mass_over_sum_pt > 0.1606
        + 0.008879147 * max(0.0, 0.008824206 - Q.zdr_1) / 0.00383206   # +0.9%  zdr_1 < 0.008824
        + 0.008482624 * max(0.0, Q.mass_top5 - 22.18342) / 12.67741   # +0.8%  mass_top5 > 22.18
        - 0.008013571 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # -0.8%  sum_pt_top50 < 959.1
        - 0.007632543 * max(0.0, 2.975532 - Q.D2) * max(0.0, Q.sj2_dr - 0.2070855) / 0.0198522   # -0.8%  D2 < 2.976 and sj2_dr > 0.2071
        - 0.007304075 * max(0.0, Q.z_dr_0_0p05 - 0.7674734) / 0.05221088   # -0.7%  z_dr_0_0p05 > 0.7675
        - 0.006618058 * max(0.0, Q.mass_top30 - 60.13856) / 25.93177   # -0.7%  mass_top30 > 60.14
        + 0.006488239 * max(0.0, 0.005402331 - Q.sum_z_dr2_top30) * max(0.0, Q.psi_0p3 - 0.9985421) / 4.280363e-07   # +0.6%  sum_z_dr2_top30 < 0.005402 and psi_0p3 > 0.9985
        + 0.006438197 * max(0.0, Q.sum_z_dr2_top40 - 0.02497133) / 0.0004755075   # +0.6%  sum_z_dr2_top40 > 0.02497
        + 0.005212707 * max(0.0, Q.sum_z_dr2_top20 - 0.007536681) / 0.003042796   # +0.5%  sum_z_dr2_top20 > 0.007537
        + 0.005116332 * max(0.0, 65.20727 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 7.769597) / 168.6958   # +0.5%  sj2_mass1 < 65.21 and sj2_mass2 > 7.77
        + 0.004131905 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +0.4%  sum_pt < 986.1
        - 0.002718204 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897) / 0.001013644   # -0.3%  mass > 162.8 and soft5_z > 0.001435
        - 0.002635842 * max(0.0, 0.01541561 - Q.e2) * max(0.0, 0.7904923 - Q.pt2_over_pt0) / 0.000518365   # -0.3%  e2 < 0.01542 and pt2_over_pt0 < 0.7905
        - 0.0020019 * max(0.0, 0.005402331 - Q.sum_z_dr2_top30) / 0.001231293   # -0.2%  sum_z_dr2_top30 < 0.005402
        - 0.001300879 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109) / 0.0004735247   # -0.1%  mass_top50 > 160.8 and soft4_z > 0.001721
        + 0.0007309911 * max(0.0, 59.40777 - Q.mass_top5) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1) / 22.61156   # +0.1%  mass_top5 < 59.41 and z_dr_0p05_0p1 < 0.8509
        + 0.0006892643 * max(0.0, Q.mass_top50 - 136.785) * max(0.0, Q.soft5_z - 0.001594761) / 0.001639485   # +0.1%  mass_top50 > 136.8 and soft5_z > 0.001595
        - 0.0001446378 * max(0.0, Q.sum_pt_top10 - 943.6922) * max(0.0, -0.07794189 - Q.eta_0) / 0.002015064   # -0.0%  sum_pt_top10 > 943.7 and eta_0 < -0.07794
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 9.061;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.060988 * (-0.02455028
        + 0.1648753 * max(0.0, 100.0884 - Q.mass) / 22.89756   # +16.5%  mass < 100.1
        + 0.07934191 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # +7.9%  mass_over_sum_pt_sq < 0.009595
        + 0.07641058 * max(0.0, 0.00818374 - Q.sum_zz_dr2) / 0.002451404   # +7.6%  sum_zz_dr2 < 0.008184
        + 0.07440864 * max(0.0, 0.02804652 - Q.e2) / 0.005764608   # +7.4%  e2 < 0.02805
        - 0.06834274 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -6.8%  lam1 < 0.006717
        - 0.06342139 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -6.3%  mass_top40 < 83.33
        + 0.05321531 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # +5.3%  n_dr_0p2_0p4 < 8
        - 0.04716028 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # -4.7%  psi_0p2 > 0.9314
        - 0.04123582 * max(0.0, 71.69608 - Q.mass_top50) / 8.178407   # -4.1%  mass_top50 < 71.7
        - 0.04053769 * max(0.0, 0.07032099 - Q.sum_z_dr) / 0.01719697   # -4.1%  sum_z_dr < 0.07032
        - 0.03751026 * max(0.0, 0.006929741 - Q.sum_z_dr2_top30) / 0.002004687   # -3.8%  sum_z_dr2_top30 < 0.00693
        - 0.03608614 * max(0.0, 0.07699881 - Q.mass_over_sum_pt) / 0.01042971   # -3.6%  mass_over_sum_pt < 0.077
        + 0.03405344 * max(0.0, 60.13856 - Q.mass_top30) / 6.902954   # +3.4%  mass_top30 < 60.14
        + 0.03305872 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +3.3%  psi_0p3 > 0.9897
        - 0.02720495 * max(0.0, 0.008031209 - Q.sum_z_dr2_top20) / 0.00309849   # -2.7%  sum_z_dr2_top20 < 0.008031
        + 0.02433159 * max(0.0, 15.0 - Q.n_dr_0p1_0p2) / 5.139044   # +2.4%  n_dr_0p1_0p2 < 15
        - 0.01943664 * max(0.0, 101.0497 - Q.mass) * max(0.0, 891.875 - Q.sum_pt_top10) / 2090.674   # -1.9%  mass < 101 and sum_pt_top10 < 891.9
        - 0.01763057 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.sum_z_dr2) / 0.005195774   # -1.8%  n_dr_0p2_0p4 < 10 and sum_z_dr2 < 0.005532
        - 0.01561252 * max(0.0, Q.z_top5_slots - 0.534626) / 0.08390497   # -1.6%  z_top5_slots > 0.5346
        - 0.01175436 * max(0.0, 3.793233e-05 - Q.e3) / 1.004564e-05   # -1.2%  e3 < 3.793e-05
        - 0.01083518 * max(0.0, 5.435412 - Q.D2) / 2.829169   # -1.1%  D2 < 5.435
        - 0.009296123 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0) / 30.16986   # -0.9%  n_dr_0p2_0p4 < 10 and n_particles > 34
        + 0.005638386 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0402832 - Q.phi_0) / 0.166757   # +0.6%  n_dr_0p2_0p4 < 10 and phi_0 < 0.04028
        + 0.00448991 * max(0.0, 80.35535 - Q.mass_top50) * max(0.0, 0.003932029 - Q.zdr_4) / 0.03420305   # +0.4%  mass_top50 < 80.36 and zdr_4 < 0.003932
        + 0.003420266 * max(0.0, 0.001291487 - Q.sum_z_dr2_top10) / 0.0002747458   # +0.3%  sum_z_dr2_top10 < 0.001291
        - 0.000691234 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.004495205 - Q.zdr_4) / 0.05697234   # -0.1%  mass < 92.86 and zdr_4 < 0.004495
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 2.287;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.287115 * (-0.3007386
        + 0.3879364 * max(0.0, 86.4 - Q.mass) / 13.67632   # +38.8%  mass < 86.4
        + 0.2087981 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.04417088   # +20.9%  mass < 86.4 and lam2 < 0.003688
        - 0.1091363 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.5993597   # -10.9%  mass < 86.4 and z_dr_0p1_0p2 < 0.06473
        - 0.09413203 * max(0.0, 0.06895248 - Q.mass_over_sum_pt) / 0.007729512   # -9.4%  mass_over_sum_pt < 0.06895
        - 0.06045187 * max(0.0, 74.78616 - Q.mass_top40) / 9.958349   # -6.0%  mass_top40 < 74.79
        + 0.03476116 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, Q.n_dr_0p05_0p1 - 1.0) / 124.4066   # +3.5%  mass_top40 < 89.68 and n_dr_0p05_0p1 > 1
        - 0.0335044 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.03580539   # -3.4%  mass < 86.4 and psi_0p3 < 0.9985
        + 0.01785524 * max(0.0, Q.sd_mass - 125.1) / 1.231011   # +1.8%  sd_mass > 125.1
        + 0.01103029 * max(0.0, Q.psi_0p1 - 0.9538343) / 0.006310723   # +1.1%  psi_0p1 > 0.9538
        + 0.009457687 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min) / 54.09437   # +0.9%  sj3_pair_mass_max > 120.6 and sj3_pair_mass_min < 76.6
        - 0.008148965 * max(0.0, 6.856375 - Q.log_sum_pt) * max(0.0, 0.002396991 - Q.lam2) / 7.289393e-06   # -0.8%  log_sum_pt < 6.856 and lam2 < 0.002397
        - 0.006641168 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.985099 - Q.z_top50_slots) / 0.008644046   # -0.7%  mass < 86.4 and z_top50_slots < 0.9851
        + 0.006376135 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926) / 0.5295314   # +0.6%  sj3_pair_mass_max > 120.6 and z_dr_0p1_0p2 > 0.2865
        - 0.005844124 * max(0.0, Q.sj3_pair_mass_max - 120.6) / 2.130243   # -0.6%  sj3_pair_mass_max > 120.6
        + 0.003698884 * max(0.0, 6.856375 - Q.log_sum_pt) / 0.008053459   # +0.4%  log_sum_pt < 6.856
        - 0.002052043 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) / 0.002080651   # -0.2%  sum_z_dr2_top15 < 0.006143
        - 0.0001751343 * max(0.0, 0.0007431905 - Q.sum_z_dr2_top10) * max(0.0, 1069.671 - Q.sum_pt_top40) / 0.006568426   # -0.0%  sum_z_dr2_top10 < 0.0007432 and sum_pt_top40 < 1070
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 7.682;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.681987 * (0.1564385
        + 0.1060205 * max(0.0, Q.sum_pt_top20 - 956.5062) / 37.46564   # +10.6%  sum_pt_top20 > 956.5
        + 0.1030675 * max(0.0, 0.02550569 - Q.sum_z_dr2_top50) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.0005189214   # +10.3%  sum_z_dr2_top50 < 0.02551 and psi_0p3 > 0.9638
        - 0.08880641 * max(0.0, Q.mass - 160.8) / 1.724663   # -8.9%  mass > 160.8
        - 0.07145763 * max(0.0, 16.0 - Q.n_for_90pct) / 1.776607   # -7.1%  n_for_90pct < 16
        - 0.05696969 * max(0.0, 1028.007 - Q.sum_pt) / 26.86015   # -5.7%  sum_pt < 1028
        - 0.05386182 * max(0.0, Q.mass - 172.8) * max(0.0, Q.soft5_z - 0.0004140594) / 0.001251279   # -5.4%  mass > 172.8 and soft5_z > 0.0004141
        - 0.0513676 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.8310045 - Q.tau21_b2) / 33.42141   # -5.1%  sum_pt < 1085 and tau21_b2 < 0.831
        + 0.0476116 * max(0.0, 1024.662 - Q.sum_pt_top40) / 38.31903   # +4.8%  sum_pt_top40 < 1025
        + 0.0443346 * max(0.0, 0.01327867 - Q.sum_z_dr2_top50) / 0.006432255   # +4.4%  sum_z_dr2_top50 < 0.01328
        + 0.04202052 * Q.n_particles / 45.81523   # +4.2%  n_particles
        - 0.04154482 * max(0.0, 160.8 - Q.mass_top40) / 76.75884   # -4.2%  mass_top40 < 160.8
        - 0.03796678 * max(0.0, 6.910131 - Q.log_sum_pt) / 0.017275   # -3.8%  log_sum_pt < 6.91
        - 0.03425333 * max(0.0, Q.mass_top50 - 92.16545) * max(0.0, 0.001923089 - Q.soft7_z) / 0.006887627   # -3.4%  mass_top50 > 92.17 and soft7_z < 0.001923
        - 0.03189895 * max(0.0, Q.mass_top50 - 136.785) * max(0.0, Q.soft6_z - 0.0008188601) / 0.003727182   # -3.2%  mass_top50 > 136.8 and soft6_z > 0.0008189
        - 0.02394892 * max(0.0, Q.mass - 172.8) * max(0.0, 0.04568661 - Q.z_6) / 0.006788705   # -2.4%  mass > 172.8 and z_6 < 0.04569
        + 0.02246388 * max(0.0, Q.mass - 74.25181) * max(0.0, 1028.184 - Q.sum_pt) / 733.8204   # +2.2%  mass > 74.25 and sum_pt < 1028
        + 0.01901849 * max(0.0, 16.0 - Q.n_for_90pct) * max(0.0, 9.676985 - Q.D2) / 8.10578   # +1.9%  n_for_90pct < 16 and D2 < 9.677
        - 0.01842838 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -1.8%  mass_over_sum_pt_sq > 0.0292
        - 0.0168391 * max(0.0, Q.mass_top30 - 138.3818) / 1.758216   # -1.7%  mass_top30 > 138.4
        + 0.01670852 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +1.7%  mass_over_sum_pt > 0.1709
        - 0.01556843 * max(0.0, Q.mass_top10 - 23.38894) / 27.40926   # -1.6%  mass_top10 > 23.39
        - 0.01468628 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.00317537 - Q.C3) / 7.466262e-06   # -1.5%  log_sum_pt > 7.139 and C3 < 0.003175
        - 0.008284479 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft6_z - 0.0005748372) / 0.002073713   # -0.8%  mass > 162.8 and soft6_z > 0.0005748
        - 0.008069271 * max(0.0, 966.0633 - Q.sum_pt_top30) / 29.99923   # -0.8%  sum_pt_top30 < 966.1
        + 0.007659952 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.2179035 - Q.sj2_zsoft) / 2.389974   # +0.8%  sum_pt_top40 < 1053 and sj2_zsoft < 0.2179
        + 0.005172563 * max(0.0, 1034.834 - Q.sum_pt) * max(0.0, 16.64062 - Q.pt_9) / 29.01849   # +0.5%  sum_pt < 1035 and pt_9 < 16.64
        - 0.003040648 * max(0.0, Q.mass - 172.8) * max(0.0, 56.53125 - Q.pt_6) / 9.768822   # -0.3%  mass > 172.8 and pt_6 < 56.53
        + 0.002534825 * max(0.0, Q.mass - 172.8) * max(0.0, Q.D2 - 0.8942376) / 0.9099067   # +0.3%  mass > 172.8 and D2 > 0.8942
        + 0.001337618 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 2.680253 - Q.D2) / 0.001923611   # +0.1%  log_sum_pt < 6.811 and D2 < 2.68
        - 0.001214176 * max(0.0, Q.mass_top50 - 138.2966) / 4.020522   # -0.1%  mass_top50 > 138.3
        + 0.001179233 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.1312677 - Q.dr_13) / 0.0002562188   # +0.1%  log_sum_pt < 6.811 and dr_13 < 0.1313
        + 0.0009289807 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.005041702 - Q.zdr_11) / 1.572361e-05   # +0.1%  log_sum_pt < 6.811 and zdr_11 < 0.005042
        - 0.0008273884 * max(0.0, 997.0189 - Q.sum_pt_top50) / 17.90852   # -0.1%  sum_pt_top50 < 997
        - 0.0004696776 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.sd_zg - 0.4462823) / 1.289737e-05   # -0.0%  log_sum_pt > 7.139 and sd_zg > 0.4463
        + 0.0002968887 * max(0.0, Q.mass_top5 - 33.71058) / 7.394508   # +0.0%  mass_top5 > 33.71
        - 0.0001405385 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.04047238 - Q.dr_9) / 6.697833e-05   # -0.0%  log_sum_pt > 7.139 and dr_9 < 0.04047
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 13.08;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.07844 * (-0.08060429
        - 0.2953156 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -29.5%  mass < 82.85
        + 0.1973752 * max(0.0, 0.140939 - Q.mass_over_sum_pt) / 0.05796036   # +19.7%  mass_over_sum_pt < 0.1409
        + 0.05764692 * max(0.0, 0.03266801 - Q.e2) / 0.008054442   # +5.8%  e2 < 0.03267
        + 0.0508584 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +5.1%  psi_0p3 > 0.9924
        - 0.04586082 * max(0.0, 0.01083435 - Q.sum_z_dr2_top20) / 0.00529825   # -4.6%  sum_z_dr2_top20 < 0.01083
        - 0.04490577 * max(0.0, 0.076787 - Q.sum_z_dr) / 0.02105267   # -4.5%  sum_z_dr < 0.07679
        - 0.04192336 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p2 - 0.9087063) / 0.00142529   # -4.2%  mass_over_sum_pt < 0.09047 and psi_0p2 > 0.9087
        - 0.03541827 * max(0.0, 0.01897915 - Q.sum_z_dr2_top40) / 0.01130444   # -3.5%  sum_z_dr2_top40 < 0.01898
        + 0.02579613 * max(0.0, 61.39113 - Q.mass_top50) / 5.389654   # +2.6%  mass_top50 < 61.39
        - 0.02151507 * max(0.0, 0.002582316 - Q.sum_z_dr2_top30) / 0.0003518773   # -2.2%  sum_z_dr2_top30 < 0.002582
        - 0.02123315 * max(0.0, 0.076787 - Q.sum_z_dr) * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.0007421723   # -2.1%  sum_z_dr < 0.07679 and z_dr_0p2_0p4 < 0.0518
        - 0.02033534 * max(0.0, 82.66587 - Q.mass_top30) / 15.96638   # -2.0%  mass_top30 < 82.67
        + 0.0193386 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +1.9%  n_dr_0p2_0p4 < 18
        - 0.01238299 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1881908 - Q.sd_rg) / 0.0002726472   # -1.2%  psi_0p3 > 0.9924 and sd_rg < 0.1882
        + 0.01191114 * max(0.0, 0.1377378 - Q.sd_rg) / 0.04425146   # +1.2%  sd_rg < 0.1377
        - 0.01161313 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # -1.2%  tau2 < 0.0795
        + 0.01044736 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) / 0.003638607   # +1.0%  sum_z_dr2_top5 < 0.007164
        - 0.0102438 * max(0.0, Q.psi_0p2 - 0.9804031) / 0.007668897   # -1.0%  psi_0p2 > 0.9804
        - 0.00991488 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2404747) / 0.01288721   # -1.0%  N2 < 0.4227 and max_dr > 0.2405
        - 0.009402176 * max(0.0, 0.07729606 - Q.tau1) / 0.01627043   # -0.9%  tau1 < 0.0773
        - 0.006388967 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.n_dr_0p05_0p1 - 14.0) / 15.57557   # -0.6%  mass < 91.19 and n_dr_0p05_0p1 > 14
        + 0.006375677 * max(0.0, Q.psi_0p2 - 0.9804031) * max(0.0, Q.pt2_over_pt0 - 0.1852611) / 0.002296628   # +0.6%  psi_0p2 > 0.9804 and pt2_over_pt0 > 0.1853
        - 0.004898929 * max(0.0, 0.02037163 - Q.lam1) / 0.01299525   # -0.5%  lam1 < 0.02037
        - 0.004843384 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) * max(0.0, Q.n_pt_above_10 - 13.0) / 0.02038471   # -0.5%  sum_z_dr2_top5 < 0.007164 and n_pt_above_10 > 13
        + 0.00399804 * max(0.0, 976.277 - Q.sum_pt_top50) / 12.87298   # +0.4%  sum_pt_top50 < 976.3
        - 0.003943599 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -18.47737) / 3.570094   # -0.4%  tau21_b2 < 0.3425 and orientation_deg > -18.48
        - 0.003094745 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1231747 - Q.z_2nd) / 6.106735e-05   # -0.3%  psi_0p3 > 0.9924 and z_2nd < 0.1232
        - 0.002815342 * max(0.0, 0.001592178 - Q.sum_z_dr2_top3) / 0.000513961   # -0.3%  sum_z_dr2_top3 < 0.001592
        + 0.00235441 * max(0.0, 6.924973 - Q.log_sum_pt) / 0.02279676   # +0.2%  log_sum_pt < 6.925
        + 0.002233052 * max(0.0, 89.6788 - Q.mass_top40) / 17.47899   # +0.2%  mass_top40 < 89.68
        - 0.002011492 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 32.0 - Q.n_real_top40) / 0.1595466   # -0.2%  tau21_b2 < 0.3425 and n_real_top40 < 32
        + 0.001889195 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +0.2%  tau21_b2 < 0.3425
        - 0.001715042 * max(0.0, 859.7766 - Q.sum_pt_top40) / 3.672354   # -0.2%  sum_pt_top40 < 859.8
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 8.109;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.109194 * (-0.09751207
        + 0.1099887 * max(0.0, 7.138608 - Q.log_sum_pt) / 0.1994883   # +11.0%  log_sum_pt < 7.139
        + 0.09323205 * max(0.0, 0.08589404 - Q.sum_z_dr) / 0.02745918   # +9.3%  sum_z_dr < 0.08589
        - 0.08599189 * max(0.0, Q.psi_0p2 - 0.8706159) / 0.08984765   # -8.6%  psi_0p2 > 0.8706
        - 0.07281266 * max(0.0, 0.2941033 - Q.LHA) / 0.05718399   # -7.3%  LHA < 0.2941
        + 0.07040341 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +7.0%  n_dr_0p2_0p4 < 26
        - 0.06828373 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # -6.8%  e2 > 0.0303
        + 0.06350376 * max(0.0, 0.01115288 - Q.sum_z_dr2_top5) / 0.006888859   # +6.4%  sum_z_dr2_top5 < 0.01115
        - 0.039982 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -4.0%  tau1 < 0.07057
        - 0.03850091 * max(0.0, 994.137 - Q.sum_pt_top40) / 23.95866   # -3.9%  sum_pt_top40 < 994.1
        + 0.03590213 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +3.6%  z_dr_0p1_0p2 < 0.1203
        - 0.03318651 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -3.3%  psi_0p3 > 0.9897
        - 0.02780937 * max(0.0, 1017.778 - Q.sum_pt_top20) / 107.0228   # -2.8%  sum_pt_top20 < 1018
        - 0.02505477 * max(0.0, 0.4226723 - Q.N2) / 0.1194948   # -2.5%  N2 < 0.4227
        + 0.02434908 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +2.4%  lam2 < 0.001776
        + 0.02384063 * max(0.0, 0.006265841 - Q.sum_z_dr2_top10) / 0.002481013   # +2.4%  sum_z_dr2_top10 < 0.006266
        + 0.0197497 * max(0.0, 933.1875 - Q.sum_pt_top30) / 20.26547   # +2.0%  sum_pt_top30 < 933.2
        + 0.01880801 * max(0.0, 97.07106 - Q.sd_mass) / 44.66664   # +1.9%  sd_mass < 97.07
        - 0.01757011 * max(0.0, Q.e3 - 0.0001841806) / 4.035159e-05   # -1.8%  e3 > 0.0001842
        + 0.01494756 * max(0.0, Q.sd_rg - 0.1776078) / 0.0282679   # +1.5%  sd_rg > 0.1776
        + 0.01443194 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) / 3.82104   # +1.4%  n_dr_0p1_0p2 < 13
        - 0.01308761 * max(0.0, 972.3665 - Q.sum_pt) / 9.634955   # -1.3%  sum_pt < 972.4
        - 0.009086085 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, Q.sj3_mass1 - 5.112677) / 0.04548726   # -0.9%  sum_z_dr2_top5 < 0.00833 and sj3_mass1 > 5.113
        + 0.008988749 * max(0.0, Q.e2 - 0.03029714) * max(0.0, 0.3166869 - Q.sj3_z2) / 0.0002537782   # +0.9%  e2 > 0.0303 and sj3_z2 < 0.3167
        - 0.008792059 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # -0.9%  lam1 < 0.01174
        + 0.008743312 * max(0.0, 73.27438 - Q.mass_top30) / 11.35435   # +0.9%  mass_top30 < 73.27
        - 0.008033229 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583) / 0.0009330603   # -0.8%  sum_z_dr2_top5 < 0.00833 and sj3_pairmin_over_m > 0.09541
        - 0.006945981 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0) / 0.2039042   # -0.7%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p2_0p4 > 5
        - 0.006610126 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # -0.7%  mass_top50 < 71.8
        + 0.006504789 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt) / 2.722316   # +0.7%  z_dr_0_0p05 < 0.8459 and sum_pt < 949.9
        + 0.005462781 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2) / 0.0004272674   # +0.5%  sj2_dr > 0.2232 and C2_b2 < 0.04009
        - 0.004580753 * max(0.0, 0.2982 - Q.max_dr) / 0.01350087   # -0.5%  max_dr < 0.2982
        - 0.003966803 * max(0.0, 0.006399858 - Q.sum_zz_dr2) / 0.001405963   # -0.4%  sum_zz_dr2 < 0.0064
        + 0.003755142 * max(0.0, Q.sj2_dr - 0.258586) / 0.01360148   # +0.4%  sj2_dr > 0.2586
        + 0.003706301 * max(0.0, 1.976207 - Q.D2) / 0.367701   # +0.4%  D2 < 1.976
        + 0.003387369 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # +0.3%  psi_0p1 > 0.8976
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.5493018907563025, 2.982434086134454, 0.1911518907563025, 0.46559768907563026, 0.6882759453781513, 1.117165231092437, 0.6519573529411765, 0.4169338235294118, 1.1489648109243698, 1.1004603991596638, 1.0374346638655463, 0.8364740546218488, 0.45456029411764703, 1.4758620273109244, 0.2366529411764706, 0.37762878151260504]
T = [3.9925210231748953, 2.475554321494223, 4.1230570886948525, 4.2397728991596635, 3.751803693704044]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +47%, n4 -15%, n9 +14%, n3 -9%, n5 -9%, n12 +3% ...
            + 0.4668783 * h[1] / H_AVG[1]
            - 0.1508424 * h[4] / H_AVG[4]
            + 0.142122 * h[9] / H_AVG[9]
            - 0.0874631 * h[3] / H_AVG[3]
            - 0.08744203 * h[5] / H_AVG[5]
            + 0.02668429 * h[12] / H_AVG[12]
            + 0.02551479 * h[6] / H_AVG[6]
            + 0.008993102 * h[8] / H_AVG[8]
            - 0.00406007 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -30%, n9 +25%, n1 -23%, n11 -8%, n12 +6%, n6 +6% ...
            - 0.2954058 * h[4] / H_AVG[4]
            + 0.2500486 * h[9] / H_AVG[9]
            - 0.2258914 * h[1] / H_AVG[1]
            - 0.08447341 * h[11] / H_AVG[11]
            + 0.06311924 * h[12] / H_AVG[12]
            + 0.05760959 * h[6] / H_AVG[6]
            + 0.009651974 * h[2] / H_AVG[2]
            + 0.007251942 * h[8] / H_AVG[8]
            - 0.006547995 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +16%, n11 +12%, n0 +10%, n14 -8%, n7 -6% ...
            - 0.2438347 * h[8] / H_AVG[8]
            + 0.1566462 * h[5] / H_AVG[5]
            + 0.1204583 * h[11] / H_AVG[11]
            + 0.09992013 * h[0] / H_AVG[0]
            - 0.07892149 * h[14] / H_AVG[14]
            - 0.06320156 * h[7] / H_AVG[7]
            - 0.05838525 * h[9] / H_AVG[9]
            + 0.05738336 * h[4] / H_AVG[4]
            + 0.04940484 * h[3] / H_AVG[3]
            - 0.0447884 * h[12] / H_AVG[12]
            - 0.01717303 * h[15] / H_AVG[15]
            + 0.009882797 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -25%, n5 +18%, n0 -18%, n6 -15%, n7 +9%, n15 +5% ...
            - 0.2540595 * h[8] / H_AVG[8]
            + 0.1811538 * h[5] / H_AVG[5]
            - 0.178144 * h[0] / H_AVG[0]
            - 0.1489664 * h[6] / H_AVG[6]
            + 0.08911946 * h[7] / H_AVG[7]
            + 0.05010084 * h[15] / H_AVG[15]
            + 0.04804479 * h[3] / H_AVG[3]
            + 0.03350417 * h[12] / H_AVG[12]
            - 0.01690703 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -36%, n10 +27%, n5 -14%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3564952 * h[13] / H_AVG[13]
            + 0.2721957 * h[10] / H_AVG[10]
            - 0.1395785 * h[5] / H_AVG[5]
            + 0.0645982 * h[8] / H_AVG[8]
            - 0.04543418 * h[12] / H_AVG[12]
            - 0.03774472 * h[15] / H_AVG[15]
            + 0.03439725 * h[4] / H_AVG[4]
            - 0.031255 * h[7] / H_AVG[7]
            + 0.01830126 * h[0] / H_AVG[0]
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
