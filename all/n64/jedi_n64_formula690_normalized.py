"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.2%   (on for 52% of jets)
  neuron  1:  12.5%   (on for 94% of jets)
  neuron  5:  11.9%   (on for 80% of jets)
  neuron  4:   9.8%   (on for 65% of jets)
  neuron  0:   7.6%   (on for 62% of jets)
  neuron 13:   7.5%   (on for 87% of jets)
  neuron  9:   7.0%   (on for 58% of jets)
  neuron 10:   6.3%   (on for 76% of jets)
  neuron 12:   4.5%   (on for 44% of jets)
  neuron  6:   3.9%   (on for 50% of jets)
  neuron  7:   3.8%   (on for 46% of jets)
  neuron  3:   3.6%   (on for 63% of jets)
  neuron 11:   3.4%   (on for 74% of jets)
  neuron 15:   2.3%   (on for 51% of jets)
  neuron 14:   2.0%   (on for 35% of jets)
  neuron  2:   0.6%   (on for 43% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.4% (the network: 81.1%); same class as the network for 94.1% of jets.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_11                 pT share × ΔR of particle 11 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
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
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 12.22;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.21952 * (0.1022582
        - 0.2203067 * max(0.0, Q.mass - 78.26182) / 21.33658   # -22.0%  mass > 78.26
        - 0.1102735 * max(0.0, Q.mass - 92.85979) / 14.48273   # -11.0%  mass > 92.86
        - 0.06316905 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -6.3%  mass < 101
        - 0.06033994 * max(0.0, Q.mass - 74.25181) / 24.07648   # -6.0%  mass > 74.25
        + 0.05875674 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 0.002241068   # +5.9%  mass_over_sum_pt_sq < 0.007873
        + 0.05837291 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq) / 0.001689525   # +5.8%  mass_over_sum_pt_sq < 0.006939
        + 0.05618842 * max(0.0, Q.mass - 91.19) / 15.01545   # +5.6%  mass > 91.19
        - 0.04617034 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -4.6%  e2_sq < 0.006167
        + 0.03596417 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +3.6%  sum_pt_top50 < 1157
        + 0.03506986 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # +3.5%  mass_top50 > 82.04
        + 0.02831413 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +2.8%  n_dr_0p2_0p4 < 15
        - 0.02256854 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # -2.3%  z_dr_0p2_0p4 < 0.09123
        - 0.02137387 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.1%  tau1 < 0.07057
        - 0.01938223 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -1.9%  girth2_top30 < 0.006364
        - 0.01833949 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # -1.8%  sum_pt_top40 < 1070
        + 0.01656644 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +1.7%  log_sum_pt < 7.017
        - 0.01414638 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -1.4%  sum_pt < 1013
        + 0.0133542 * max(0.0, Q.mass_top50 - 71.79516) / 24.22501   # +1.3%  mass_top50 > 71.8
        - 0.0122792 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # -1.2%  girth2_top20 < 0.006374
        - 0.01123988 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -1.1%  lam1 < 0.005914
        + 0.01075757 * max(0.0, 80.4 - Q.mass_top30) / 14.65577   # +1.1%  mass_top30 < 80.4
        - 0.01007228 * max(0.0, 0.007538019 - Q.girth2_top20) / 0.002740841   # -1.0%  girth2_top20 < 0.007538
        - 0.009727444 * max(0.0, 0.9906378 - Q.z_top50_slots) / 0.004783315   # -1.0%  z_top50_slots < 0.9906
        + 0.009249794 * max(0.0, 0.007856958 - Q.girth2_top30) / 0.002601721   # +0.9%  girth2_top30 < 0.007857
        + 0.008715306 * max(0.0, 0.005312783 - Q.girth2_top20) / 0.001437214   # +0.9%  girth2_top20 < 0.005313
        + 0.007678974 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +0.8%  psi_0p3 > 0.9956
        + 0.007375752 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957) / 1.852228   # +0.7%  log_sum_pt < 6.989 and sum_pt_top50 > 959.1
        + 0.005679336 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +0.6%  mass_top40 < 80.89
        + 0.004969861 * max(0.0, 0.004754444 - Q.mass_over_sum_pt_sq) / 0.0007997283   # +0.5%  mass_over_sum_pt_sq < 0.004754
        + 0.003597689 * max(0.0, 846.1934 - Q.sum_pt_top20) / 22.83716   # +0.4%  sum_pt_top20 < 846.2
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 21.11;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.10704 * (0.06420026
        - 0.08174294 * max(0.0, 2.275391 - Q.soft1_pt) / 1.652278   # -8.2%  soft1_pt < 2.275
        + 0.07885441 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # +7.9%  log_sum_pt > 6.894
        + 0.06268054 * max(0.0, 0.1953848 - Q.tau1) / 0.1097382   # +6.3%  tau1 < 0.1954
        + 0.0567438 * max(0.0, 1.521582 - Q.soft1_pt) / 0.9527349   # +5.7%  soft1_pt < 1.522
        - 0.05102621 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -5.1%  log_sum_pt < 7.139
        + 0.04575946 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +4.6%  log_sum_pt > 6.91
        + 0.04169668 * max(0.0, Q.pt_entropy - 2.07371) / 0.8049786   # +4.2%  pt_entropy > 2.074
        - 0.03881331 * max(0.0, Q.z_top50_slots - 0.9586536) / 0.03427112   # -3.9%  z_top50_slots > 0.9587
        - 0.03745944 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -3.7%  sum_pt_top50 > 959.1
        - 0.03667211 * max(0.0, 120.6 - Q.mass) / 38.8279   # -3.7%  mass < 120.6
        + 0.03076043 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +3.1%  n_particles > 38
        - 0.02697468 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # -2.7%  mass_over_sum_pt_sq < 0.009595
        - 0.02564298 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -2.6%  log_sum_pt > 6.959
        + 0.02502068 * max(0.0, 0.008031986 - Q.girth2_top40) / 0.002514593   # +2.5%  girth2_top40 < 0.008032
        + 0.02468453 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # +2.5%  sum_pt_top40 < 1070
        - 0.02202235 * max(0.0, Q.n_for_90pct - 11.0) / 10.33888   # -2.2%  n_for_90pct > 11
        + 0.02189083 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.2734   # +2.2%  sum_pt_top2 < 689.2
        + 0.01568987 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # +1.6%  LHA < 0.2601
        - 0.01467641 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2) / 190.5117   # -1.5%  sj3_mass1 < 32.5 and sj3_mass2 < 18.68
        + 0.0139473 * max(0.0, 0.04516808 - Q.tau3) / 0.02499032   # +1.4%  tau3 < 0.04517
        - 0.01295183 * max(0.0, 47.88842 - Q.mass_top20) / 6.040308   # -1.3%  mass_top20 < 47.89
        - 0.01239601 * max(0.0, Q.z_top30_slots - 0.9341838) / 0.03406818   # -1.2%  z_top30_slots > 0.9342
        - 0.01158994 * max(0.0, 31.35938 - Q.pt_9) / 6.538304   # -1.2%  pt_9 < 31.36
        + 0.01130938 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.1%  z_dr_0p2_0p4 < 0.09123
        - 0.01030502 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt) / 15.05788   # -1.0%  n_particles > 38 and soft1_pt < 2.275
        + 0.008952729 * max(0.0, Q.z_top20_slots - 0.8965411) / 0.0341307   # +0.9%  z_top20_slots > 0.8965
        - 0.008468883 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # -0.8%  e2 < 0.02516
        + 0.008410455 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2) / 0.0008500795   # +0.8%  z_top30_slots > 0.9342 and C2 < 0.0728
        - 0.008337615 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -0.8%  log_sum_pt > 6.989
        - 0.007949598 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # -0.8%  sj3_mass1 < 32.5
        - 0.007892154 * max(0.0, Q.z_top20_slots - 0.8965411) * max(0.0, 0.03843804 - Q.dr_2) / 0.0005290849   # -0.8%  z_top20_slots > 0.8965 and dr_2 < 0.03844
        + 0.007694148 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +0.8%  sum_pt < 1017
        - 0.0072987 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # -0.7%  M3 < 0.03188
        - 0.007196733 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -0.7%  n_dr_0p2_0p4 < 7
        - 0.006837367 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.7%  psi_0p3 > 0.998
        + 0.006389075 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1) / 1.005598   # +0.6%  n_particles > 38 and dr_1 < 0.1612
        - 0.006218807 * max(0.0, 30.26161 - Q.sj2_mass1) / 7.997784   # -0.6%  sj2_mass1 < 30.26
        + 0.006165862 * max(0.0, 0.02412652 - Q.girth2_top30) / 0.01613825   # +0.6%  girth2_top30 < 0.02413
        + 0.006037844 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 39.23115   # +0.6%  mass_top20 < 47.89 and n_real_top40 > 29
        + 0.005950835 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0) / 0.570275   # +0.6%  n_particles > 38 and dr_0 < 0.112
        - 0.005492055 * max(0.0, Q.pt_entropy - 2.07371) * max(0.0, 0.3797 - Q.soft3_dr) / 0.1539306   # -0.5%  pt_entropy > 2.074 and soft3_dr < 0.3797
        + 0.005354685 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.5%  lam1 < 0.004673
        + 0.005338209 * max(0.0, 0.007538019 - Q.girth2_top20) / 0.002740841   # +0.5%  girth2_top20 < 0.007538
        - 0.004435894 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, Q.soft3_dr0 - 0.02918107) / 57.76427   # -0.4%  sum_pt_top2 < 689.2 and soft3_dr0 > 0.02918
        + 0.004276052 * max(0.0, 0.1416054 - Q.D3) / 0.05511785   # +0.4%  D3 < 0.1416
        + 0.004172621 * max(0.0, 120.6 - Q.mass) * max(0.0, 0.03843804 - Q.dr_2) / 0.6524502   # +0.4%  mass < 120.6 and dr_2 < 0.03844
        - 0.003632698 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.4%  log_sum_pt > 7.063
        + 0.003570824 * max(0.0, 0.001823079 - Q.girth2_top20) / 0.000269929   # +0.4%  girth2_top20 < 0.001823
        + 0.003519931 * max(0.0, 0.007538019 - Q.girth2_top20) * max(0.0, Q.eta_1 - -0.04302979) / 0.000120198   # +0.4%  girth2_top20 < 0.007538 and eta_1 > -0.04303
        + 0.003491939 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793) / 0.1941151   # +0.3%  z_top30_slots > 0.9342 and max_pair_mass > 13.05
        - 0.003393503 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.3%  n_dr_0p1_0p2 < 8
        + 0.003244586 * max(0.0, 12.0 - Q.n_dr_0_0p05) / 3.466761   # +0.3%  n_dr_0_0p05 < 12
        + 0.003193749 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 11.29505 - Q.pair_mass_0_13) / 38.90723   # +0.3%  pt_9 < 31.36 and pair_mass_0_13 < 11.3
        + 0.003153146 * max(0.0, Q.mass_top10 - 56.92192) / 8.071231   # +0.3%  mass_top10 > 56.92
        - 0.002972174 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -0.3%  z_dr_0_0p05 > 0.8789
        + 0.002928119 * max(0.0, Q.psi_0p1 - 0.8747961) / 0.03440015   # +0.3%  psi_0p1 > 0.8748
        + 0.002917805 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 0.3241858 - Q.dr1_12) / 1.371943   # +0.3%  pt_9 < 31.36 and dr1_12 < 0.3242
        - 0.002818036 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 0.02980347 - Q.eta_0) / 5.83951   # -0.3%  sum_pt_top5 > 430.8 and eta_0 < 0.0298
        + 0.002305716 * max(0.0, 0.0006570502 - Q.girth2_top5) / 0.0001395474   # +0.2%  girth2_top5 < 0.0006571
        + 0.002297311 * max(0.0, 30.26161 - Q.sj2_mass1) * max(0.0, 0.2366434 - Q.soft10_dr0) / 1.029958   # +0.2%  sj2_mass1 < 30.26 and soft10_dr0 < 0.2366
        + 0.002116638 * max(0.0, 0.0007894752 - Q.girth2_top15) / 9.062109e-05   # +0.2%  girth2_top15 < 0.0007895
        + 0.001570145 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874) / 0.05244257   # +0.2%  z_top30_slots > 0.9342 and ptdr0_3 > 7.408
        + 0.001550413 * max(0.0, Q.sum_pt_top30 - 1191.938) / 6.938323   # +0.2%  sum_pt_top30 > 1192
        + 0.001141802 * max(0.0, 0.003270031 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 6.08612e-07   # +0.1%  girth2_top15 < 0.00327 and psi_0p3 > 0.9974
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 6.426;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.425573 * (0.04066011
        - 0.1757267 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -17.6%  mass < 92.86
        + 0.09887205 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15) / 0.0009227558   # +9.9%  log_sum_pt > 6.903 and girth2_top15 < 0.02147
        + 0.07573705 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # +7.6%  sum_pt > 1017
        + 0.06679073 * max(0.0, 91.19 - Q.mass) / 16.46423   # +6.7%  mass < 91.19
        - 0.06154472 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # -6.2%  sum_pt < 1261
        - 0.05903038 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15) / 1.045047   # -5.9%  sum_pt_top50 > 988.5 and girth2_top15 < 0.02147
        - 0.05542771 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -5.5%  log_sum_pt > 6.959
        + 0.05408936 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # +5.4%  mass_top50 < 92.17
        + 0.05219244 * max(0.0, 1041.263 - Q.sum_pt_top40) / 49.16087   # +5.2%  sum_pt_top40 < 1041
        - 0.0477861 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # -4.8%  sum_pt_top50 < 1079
        - 0.0437653 * max(0.0, 996.8867 - Q.sum_pt_top30) / 42.94346   # -4.4%  sum_pt_top30 < 996.9
        + 0.03162281 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +3.2%  lam1 < 0.01174
        + 0.03029364 * max(0.0, 1048.098 - Q.sum_pt_top50) / 44.36839   # +3.0%  sum_pt_top50 < 1048
        + 0.02186808 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +2.2%  mass_over_sum_pt < 0.09795
        + 0.01908251 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +1.9%  sum_pt_top40 > 1070
        - 0.01863898 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -1.9%  sum_pt < 972
        + 0.01732174 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 4.657698e-07   # +1.7%  sum_pt < 972 and e4 < 5.851e-08
        - 0.01505309 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # -1.5%  sum_pt > 1116
        - 0.01417706 * max(0.0, Q.sum_pt_top40 - 984.7009) / 58.19535   # -1.4%  sum_pt_top40 > 984.7
        + 0.01183753 * max(0.0, 966.0633 - Q.sum_pt_top30) / 29.99923   # +1.2%  sum_pt_top30 < 966.1
        - 0.01140603 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003463934   # -1.1%  log_sum_pt > 6.903 and psi_0p3 > 0.9299
        - 0.01038669 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -1.0%  log_sum_pt > 7.063
        + 0.007000264 * max(0.0, Q.sum_pt_top50 - 1156.659) / 13.71464   # +0.7%  sum_pt_top50 > 1157
        - 0.0003490518 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.08734131) / 0.003497383   # -0.0%  sum_pt_top20 > 1129 and eta_1 > 0.08734
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 3.352;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.352488 * (-0.02924461
        - 0.1537693 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # -15.4%  girth2 < 0.009615
        + 0.1221387 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +12.2%  girth2_top40 < 0.008841
        - 0.08328543 * max(0.0, 2.410481 - Q.D2) / 0.587301   # -8.3%  D2 < 2.41
        + 0.07104008 * max(0.0, 2.410481 - Q.D2) * max(0.0, 5.142486 - Q.D2_b2) / 2.680544   # +7.1%  D2 < 2.41 and D2_b2 < 5.142
        + 0.04866335 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +4.9%  n_particles < 46
        - 0.04634584 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -4.6%  tau21 < 0.3862
        + 0.04513328 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # +4.5%  girth2_top50 < 0.008125
        + 0.04100914 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741) / 0.0491674   # +4.1%  n_dr_0p2_0p4 < 8 and z_top50_slots > 0.9787
        + 0.03771049 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +3.8%  lam2 < 0.0006155
        - 0.03439032 * max(0.0, 79.21004 - Q.mass_top50) / 10.65754   # -3.4%  mass_top50 < 79.21
        + 0.03407182 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +3.4%  n_dr_0p2_0p4 < 5
        + 0.03262064 * max(0.0, 0.08665515 - Q.mass_over_sum_pt) / 0.01542259   # +3.3%  mass_over_sum_pt < 0.08666
        + 0.03254798 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +3.3%  mass < 92.86
        + 0.02708958 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +2.7%  tau4 < 0.01627
        + 0.0257361 * max(0.0, 27.56535 - Q.sj2_mass1) / 6.309897   # +2.6%  sj2_mass1 < 27.57
        - 0.02205388 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -2.2%  mass < 53.87
        - 0.02191676 * max(0.0, 80.4 - Q.mass_top40) / 12.19371   # -2.2%  mass_top40 < 80.4
        - 0.02185535 * max(0.0, 86.4 - Q.mass) / 13.67632   # -2.2%  mass < 86.4
        + 0.02110153 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1) / 0.1325294   # +2.1%  n_dr_0p2_0p4 < 5 and psi_0p1 < 0.9538
        - 0.02037017 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5) / 0.004484684   # -2.0%  n_dr_0p2_0p4 < 5 and zdr_5 < 0.007099
        + 0.02019525 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.9313699) / 0.1032909   # +2.0%  n_dr_0p1_0p2 < 10 and psi_0p2 > 0.9314
        - 0.01424919 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349) / 64.17204   # -1.4%  n_particles < 46 and mass_top15 > 57.87
        - 0.01228967 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243) / 0.0001457263   # -1.2%  tau21 < 0.3862 and lam1 > 0.007671
        - 0.01041617 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0) / 3.186294   # -1.0%  n_dr_0p2_0p4 < 5 and n_dr_0p1_0p2 > 9
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 10.75;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.75443 * (0.05526394
        - 0.09771076 * max(0.0, 86.4 - Q.mass) / 13.67632   # -9.8%  mass < 86.4
        + 0.08274784 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +8.3%  mass < 101
        - 0.08256644 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -8.3%  mass_top40 < 163.3
        + 0.06299415 * max(0.0, 120.6 - Q.mass) / 38.8279   # +6.3%  mass < 120.6
        - 0.0601146 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -6.0%  mass < 92.86
        + 0.04767301 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +4.8%  n_dr_0p2_0p4 < 26
        - 0.04127145 * max(0.0, 91.69753 - Q.mass_top30) / 22.0127   # -4.1%  mass_top30 < 91.7
        + 0.03703208 * max(0.0, Q.e2_sq - 0.009606007) / 0.003182984   # +3.7%  e2_sq > 0.009606
        - 0.03661506 * max(0.0, Q.n_particles - 22.0) / 23.9717   # -3.7%  n_particles > 22
        - 0.0318477 * max(0.0, 78.26182 - Q.mass) / 9.857173   # -3.2%  mass < 78.26
        + 0.02476706 * max(0.0, 62.55 - Q.mass_top40) / 6.231505   # +2.5%  mass_top40 < 62.55
        - 0.02370329 * max(0.0, Q.lam1 - 0.008241985) / 0.002595658   # -2.4%  lam1 > 0.008242
        - 0.02312774 * max(0.0, 0.004855289 - Q.girth2_top15) / 0.001418414   # -2.3%  girth2_top15 < 0.004855
        - 0.02168873 * max(0.0, 74.25181 - Q.mass) / 8.587064   # -2.2%  mass < 74.25
        + 0.02156184 * max(0.0, 80.78464 - Q.mass) / 10.84952   # +2.2%  mass < 80.78
        + 0.01842196 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt) / 36.47388   # +1.8%  n_particles > 22 and soft1_pt < 2.275
        + 0.01819214 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +1.8%  psi_0p3 > 0.9974
        - 0.01709003 * max(0.0, Q.girth2_top15 - 0.00727763) / 0.002923446   # -1.7%  girth2_top15 > 0.007278
        + 0.01667276 * max(0.0, 143.7876 - Q.mass) / 57.99337   # +1.7%  mass < 143.8
        - 0.01661133 * max(0.0, 0.03577037 - Q.girth) / 0.004182456   # -1.7%  girth < 0.03577
        + 0.01548069 * max(0.0, 101.0497 - Q.mass) * max(0.0, 3.852812 - Q.D2_b2) / 25.01551   # +1.5%  mass < 101 and D2_b2 < 3.853
        + 0.01409806 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.005180665 - Q.girth2_top2) / 0.07860143   # +1.4%  mass < 92.86 and girth2_top2 < 0.005181
        + 0.0136592 * max(0.0, 57.87349 - Q.mass_top15) / 12.46085   # +1.4%  mass_top15 < 57.87
        + 0.01248644 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +1.2%  mass_top30 < 60.44
        + 0.01245196 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # +1.2%  mass_top40 < 83.33
        + 0.01195081 * max(0.0, 0.9909875 - Q.psi_0p1) / 0.2210377   # +1.2%  psi_0p1 < 0.991
        + 0.01029207 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +1.0%  sj2_dr > 0.2232
        + 0.009153132 * max(0.0, Q.e2 - 0.02793599) / 0.008664879   # +0.9%  e2 > 0.02794
        - 0.009004611 * max(0.0, Q.n_particles - 22.0) * max(0.0, 3.852812 - Q.D2_b2) / 43.35921   # -0.9%  n_particles > 22 and D2_b2 < 3.853
        - 0.008391605 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # -0.8%  C2 > 0.06656
        + 0.008347047 * max(0.0, Q.girth2_top15 - 0.01563836) / 0.001334556   # +0.8%  girth2_top15 > 0.01564
        - 0.007988491 * max(0.0, Q.girth - 0.1207452) / 0.004285542   # -0.8%  girth > 0.1207
        + 0.007947092 * max(0.0, Q.sum_pt - 995.6769) / 62.24161   # +0.8%  sum_pt > 995.7
        + 0.007900799 * max(0.0, Q.e2_sq - 0.01396296) / 0.002220692   # +0.8%  e2_sq > 0.01396
        - 0.00787155 * max(0.0, Q.sj2_dr - 0.2412757) / 0.01833576   # -0.8%  sj2_dr > 0.2413
        + 0.00703471 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +0.7%  lam1 > 0.01174
        + 0.006498341 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.n_dr_0_0p05 - 10.0) / 143.1065   # +0.6%  n_particles > 22 and n_dr_0_0p05 > 10
        - 0.005980945 * max(0.0, 80.78464 - Q.mass) * max(0.0, 3.852812 - Q.D2_b2) / 3.983646   # -0.6%  mass < 80.78 and D2_b2 < 3.853
        - 0.005730332 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 56.19962   # -0.6%  n_dr_0p2_0p4 < 26 and n_dr_0p1_0p2 > 11
        - 0.005106176 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 20.40995 - Q.D2_b2) / 0.3948355   # -0.5%  sj2_dr > 0.2232 and D2_b2 < 20.41
        - 0.005089847 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # -0.5%  e2 > 0.04755
        - 0.005006981 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 3.014827 - Q.D2_b2) / 4.336836   # -0.5%  mass_top40 < 83.33 and D2_b2 < 3.015
        - 0.004545871 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 13.0 - Q.n_dr_0_0p05) / 0.003836025   # -0.5%  psi_0p3 > 0.9974 and n_dr_0_0p05 < 13
        - 0.004027456 * max(0.0, Q.girth2_top5 - 0.007164202) / 0.002334697   # -0.4%  girth2_top5 > 0.007164
        + 0.003552901 * max(0.0, Q.e3 - 0.0001841806) / 4.035159e-05   # +0.4%  e3 > 0.0001842
        - 0.003276521 * max(0.0, Q.log_sum_pt - 6.97212) / 0.02498281   # -0.3%  log_sum_pt > 6.972
        + 0.001737803 * max(0.0, Q.girth2_top5 - 0.0168592) / 0.0008909511   # +0.2%  girth2_top5 > 0.01686
        + 0.001666208 * max(0.0, 91.19 - Q.mass_top15) / 34.88803   # +0.2%  mass_top15 < 91.19
        - 0.001035925 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # -0.1%  e2 > 0.06524
        - 0.0002764457 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # -0.0%  mass_top40 < 67.73
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 14.37;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.37343 * (0.06365938
        + 0.09315937 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +9.3%  sum_pt_top50 > 934.2
        - 0.06925017 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # -6.9%  sum_pt > 986.1
        + 0.06783787 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +6.8%  sum_pt > 907.9
        - 0.06094705 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -6.1%  log_sum_pt > 6.91
        + 0.04720615 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +4.7%  n_particles < 64
        + 0.04283682 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # +4.3%  sum_pt_top50 > 959.1
        - 0.03966817 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # -4.0%  mass_over_sum_pt > 0.07697
        + 0.03434353 * max(0.0, Q.mass - 74.25181) / 24.07648   # +3.4%  mass > 74.25
        - 0.02976764 * max(0.0, Q.mass - 172.8) / 0.7291439   # -3.0%  mass > 172.8
        - 0.02921881 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -2.9%  log_sum_pt > 6.92
        - 0.02858013 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -2.9%  mass_top50 > 157.5
        + 0.02502814 * max(0.0, Q.sd_mass - 69.65633) / 11.73272   # +2.5%  sd_mass > 69.66
        + 0.02442777 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # +2.4%  e2 < 0.03876
        - 0.02406824 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -2.4%  max_dr > 0.2405
        - 0.02037281 * max(0.0, Q.sd_mass - 86.4) / 6.275027   # -2.0%  sd_mass > 86.4
        - 0.01995793 * max(0.0, 150.0144 - Q.mass_top40) / 66.86198   # -2.0%  mass_top40 < 150
        - 0.01886296 * max(0.0, Q.z_top30_slots - 0.9048492) / 0.05621641   # -1.9%  z_top30_slots > 0.9048
        + 0.01830345 * max(0.0, Q.girth2 - 0.004756928) / 0.005348122   # +1.8%  girth2 > 0.004757
        - 0.0168972 * max(0.0, Q.z_top50_slots - 0.9906378) / 0.006446962   # -1.7%  z_top50_slots > 0.9906
        + 0.01611014 * max(0.0, Q.mass - 172.4888) / 0.7446233   # +1.6%  mass > 172.5
        + 0.01500375 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +1.5%  sum_pt > 907.9 and e4 < 5.851e-08
        - 0.01480551 * max(0.0, Q.mass - 91.19) / 15.01545   # -1.5%  mass > 91.19
        + 0.01374274 * max(0.0, Q.girth2_top30 - 0.006363916) / 0.003802703   # +1.4%  girth2_top30 > 0.006364
        - 0.01347052 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2) / 10.81389   # -1.3%  n_particles < 64 and D2 < 2.179
        - 0.01230114 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -1.2%  girth2_top20 < 0.01083
        + 0.01215447 * max(0.0, Q.sum_pt_top30 - 933.1875) / 79.88556   # +1.2%  sum_pt_top30 > 933.2
        - 0.01176028 * max(0.0, 787.6281 - Q.sum_pt_top3) / 328.0371   # -1.2%  sum_pt_top3 < 787.6
        - 0.01142655 * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 9.82015   # -1.1%  n_dr_0p1_0p2 < 21
        - 0.01131687 * max(0.0, Q.sum_pt_top40 - 994.2695) / 51.72055   # -1.1%  sum_pt_top40 > 994.3
        - 0.01108768 * max(0.0, 0.5494307 - Q.tau21) / 0.1591168   # -1.1%  tau21 < 0.5494
        - 0.01071101 * max(0.0, Q.mass_over_sum_pt - 0.09046749) / 0.01421039   # -1.1%  mass_over_sum_pt > 0.09047
        - 0.009543325 * max(0.0, Q.girth2_top50 - 0.01951641) / 0.001208668   # -1.0%  girth2_top50 > 0.01952
        - 0.009477714 * max(0.0, Q.z_top20_slots - 0.9102775) / 0.02689878   # -0.9%  z_top20_slots > 0.9103
        - 0.008613124 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # -0.9%  psi_0p3 > 0.9974
        + 0.00848075 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # +0.8%  log_sum_pt > 6.989
        + 0.008200858 * max(0.0, 1.976207 - Q.D2) / 0.367701   # +0.8%  D2 < 1.976
        - 0.008051558 * max(0.0, 0.005788041 - Q.girth2_top15) / 0.001877672   # -0.8%  girth2_top15 < 0.005788
        - 0.007941898 * max(0.0, Q.girth2_top30 - 0.007463985) / 0.00336109   # -0.8%  girth2_top30 > 0.007464
        + 0.007838031 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +0.8%  n_dr_0p2_0p4 < 11
        + 0.007184185 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +0.7%  mass_over_sum_pt > 0.1709
        - 0.007120077 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -0.7%  mass_over_sum_pt_sq > 0.0292
        + 0.006309952 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2) / 0.001543279   # +0.6%  psi_0p3 > 0.9974 and D2 < 3.345
        + 0.006174068 * max(0.0, 0.01375115 - Q.z_11) / 0.0009450582   # +0.6%  z_11 < 0.01375
        - 0.00578735 * max(0.0, 14.14062 - Q.pt_11) / 0.9740783   # -0.6%  pt_11 < 14.14
        - 0.005199295 * max(0.0, 0.01541561 - Q.e2) / 0.001436351   # -0.5%  e2 < 0.01542
        + 0.005026773 * max(0.0, Q.log_sum_pt - 6.935549) / 0.0371538   # +0.5%  log_sum_pt > 6.936
        + 0.004746802 * max(0.0, 64.48544 - Q.mass) / 5.935866   # +0.5%  mass < 64.49
        - 0.004536039 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -0.5%  sum_pt_top40 > 1025
        + 0.004353307 * max(0.0, Q.log_sum_pt - 6.949481) / 0.03162473   # +0.4%  log_sum_pt > 6.949
        + 0.003357072 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # +0.3%  mass_top10 > 71.78
        + 0.00316486 * max(0.0, Q.n_pt_above_1 - 28.0) * max(0.0, 0.942303 - Q.tau43) / 1.657469   # +0.3%  n_pt_above_1 > 28 and tau43 < 0.9423
        + 0.002876371 * max(0.0, Q.z_dr_0_0p05 - 0.9084912) / 0.009227347   # +0.3%  z_dr_0_0p05 > 0.9085
        + 0.001391729 * max(0.0, Q.max_dr - 0.4357228) / 0.007711928   # +0.1%  max_dr > 0.4357
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 16.82;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.81898 * (-0.007786393
        + 0.1407881 * max(0.0, 172.8 - Q.mass_top50) / 85.49388   # +14.1%  mass_top50 < 172.8
        - 0.09384769 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -9.4%  mass < 101
        - 0.08572014 * max(0.0, 168.9698 - Q.mass_top50) / 81.82258   # -8.6%  mass_top50 < 169
        - 0.07710869 * max(0.0, 120.6 - Q.mass) / 38.8279   # -7.7%  mass < 120.6
        + 0.0751699 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +7.5%  mass < 92.86
        + 0.06877524 * max(0.0, Q.mass_over_sum_pt - 0.02682209) / 0.06022589   # +6.9%  mass_over_sum_pt > 0.02682
        - 0.03517506 * max(0.0, 0.01807679 - Q.girth2_top30) / 0.01083719   # -3.5%  girth2_top30 < 0.01808
        + 0.03373761 * max(0.0, 86.4 - Q.mass) / 13.67632   # +3.4%  mass < 86.4
        - 0.03319954 * max(0.0, Q.mass_over_sum_pt - 0.09795415) / 0.01222897   # -3.3%  mass_over_sum_pt > 0.09795
        + 0.03212622 * max(0.0, 0.06524004 - Q.e2) / 0.03476195   # +3.2%  e2 < 0.06524
        - 0.03139731 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # -3.1%  e3 < 0.0003372
        - 0.02591722 * max(0.0, Q.e2 - 0.02793599) / 0.008664879   # -2.6%  e2 > 0.02794
        + 0.02551848 * max(0.0, 0.003687605 - Q.lam2) / 0.002601874   # +2.6%  lam2 < 0.003688
        + 0.02028541 * max(0.0, Q.girth2_top30 - 0.008376291) / 0.003092781   # +2.0%  girth2_top30 > 0.008376
        - 0.02009357 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # -2.0%  sum_pt < 1261
        + 0.01816638 * max(0.0, Q.girth2_top20 - 0.008031209) / 0.002906852   # +1.8%  girth2_top20 > 0.008031
        + 0.01722289 * max(0.0, Q.tau1 - 0.05444509) / 0.03990678   # +1.7%  tau1 > 0.05445
        + 0.01382226 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +1.4%  lam1 < 0.007259
        - 0.01369723 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -1.4%  mass_over_sum_pt > 0.1182
        - 0.01281978 * max(0.0, Q.girth2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529) / 8.566262e-05   # -1.3%  girth2_top20 > 0.008031 and C2_b2 > 0.0008188
        + 0.01243969 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # +1.2%  mass_top50 < 71.8
        - 0.01219523 * max(0.0, 0.004573744 - Q.girth2_top50) / 0.0007739713   # -1.2%  girth2_top50 < 0.004574
        + 0.009028787 * max(0.0, Q.girth2_top40 - 0.008031986) / 0.003375597   # +0.9%  girth2_top40 > 0.008032
        - 0.008956141 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2) / 0.04240098   # -0.9%  mass_over_sum_pt > 0.1182 and D2_b2 < 7.366
        - 0.008515714 * max(0.0, Q.mass_over_sum_pt - 0.02682209) * max(0.0, 0.7108211 - Q.z_dr_0p05_0p1) / 0.02513932   # -0.9%  mass_over_sum_pt > 0.02682 and z_dr_0p05_0p1 < 0.7108
        + 0.008338741 * max(0.0, 0.003811746 - Q.lam1) / 0.0006796118   # +0.8%  lam1 < 0.003812
        + 0.006793169 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +0.7%  tau1 < 0.06311
        - 0.005605015 * max(0.0, Q.sj3_pair_mass_min - 29.00832) / 6.839984   # -0.6%  sj3_pair_mass_min > 29.01
        - 0.00547784 * max(0.0, Q.girth - 0.02085222) / 0.04959222   # -0.5%  girth > 0.02085
        + 0.005376025 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.5%  e2 > 0.04755
        + 0.004164679 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt) / 683.4113   # +0.4%  mass < 120.6 and sum_pt < 1008
        + 0.004136681 * max(0.0, Q.e2 - 0.02793599) * max(0.0, 0.4357228 - Q.max_dr) / 0.0007248268   # +0.4%  e2 > 0.02794 and max_dr < 0.4357
        - 0.004000352 * max(0.0, 0.08329945 - Q.mass_over_sum_pt) / 0.0135093   # -0.4%  mass_over_sum_pt < 0.0833
        + 0.003977755 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # +0.4%  e2 > 0.05557
        - 0.003553388 * max(0.0, 0.003418057 - Q.girth2_top50) / 0.0004594629   # -0.4%  girth2_top50 < 0.003418
        - 0.003043642 * max(0.0, Q.girth2_top30 - 0.02809026) / 0.0001993235   # -0.3%  girth2_top30 > 0.02809
        + 0.003009462 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832) / 8.191828e-05   # +0.3%  mass_over_sum_pt > 0.1182 and C2_b2 > 0.02705
        + 0.002997993 * max(0.0, 0.00375223 - Q.girth2_top30) / 0.0006714094   # +0.3%  girth2_top30 < 0.003752
        - 0.002456717 * max(0.0, Q.sj3_pair_mass_min - 76.60223) / 0.3297656   # -0.2%  sj3_pair_mass_min > 76.6
        - 0.002043407 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722) / 0.001787472   # -0.2%  girth2_top30 > 0.008376 and D2_b2 > 1.677
        - 0.001879958 * max(0.0, Q.sj3_mass1 - 21.11128) / 1.638992   # -0.2%  sj3_mass1 > 21.11
        + 0.001733606 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.soft3_dr0 - 0.1966723) / 0.0006797808   # +0.2%  mass_over_sum_pt > 0.1182 and soft3_dr0 > 0.1967
        + 0.0016579 * max(0.0, 0.9300465 - Q.z_top40_slots) / 0.002595351   # +0.2%  z_top40_slots < 0.93
        + 0.001644325 * max(0.0, Q.sj3_pair_mass_min - 29.00832) * max(0.0, Q.psi_0p3 - 0.9896594) / 0.01664863   # +0.2%  sj3_pair_mass_min > 29.01 and psi_0p3 > 0.9897
        - 0.00147532 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.1%  log_sum_pt < 6.811
        - 0.0009097618 * max(0.0, Q.mass_over_sum_pt - 0.09795415) * max(0.0, 0.001160626 - Q.soft8_z) / 8.555108e-07   # -0.1%  mass_over_sum_pt > 0.09795 and soft8_z < 0.001161
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 39.77;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 39.76504 * (-0.003741351
        + 0.1319863 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5337743   # +13.2%  mass < 91.19 and psi_0p3 > 0.9638
        - 0.1259329 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.3075308   # -12.6%  mass < 91.19 and psi_0p3 > 0.9777
        - 0.1182437 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5700006   # -11.8%  mass < 92.86 and psi_0p3 > 0.9638
        + 0.08680964 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.4385899   # +8.7%  mass < 101 and psi_0p3 > 0.9777
        + 0.0438181 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.2231625   # +4.4%  mass < 82.85 and psi_0p3 > 0.9777
        - 0.0400021 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -4.0%  mass < 101
        + 0.03314371 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.01745475   # +3.3%  mass < 101 and psi_0p3 > 0.998
        - 0.03180191 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -3.2%  tau1 < 0.1073
        + 0.03111189 * max(0.0, 120.6 - Q.mass) / 38.8279   # +3.1%  mass < 120.6
        - 0.02994399 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.01143143   # -3.0%  mass < 91.19 and psi_0p3 > 0.998
        - 0.02900786 * max(0.0, 0.008190222 - Q.girth2) / 0.002454223   # -2.9%  girth2 < 0.00819
        + 0.02696427 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # +2.7%  girth2 < 0.009615
        - 0.02654118 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -2.7%  mass < 82.85
        + 0.02259346 * max(0.0, 69.65633 - Q.sd_mass) / 24.56034   # +2.3%  sd_mass < 69.66
        - 0.0216697 * max(0.0, 86.4 - Q.sd_mass) / 35.84633   # -2.2%  sd_mass < 86.4
        - 0.01665192 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -1.7%  lam1 < 0.008242
        - 0.01621865 * max(0.0, 0.1881908 - Q.sd_rg) / 0.07278694   # -1.6%  sd_rg < 0.1882
        + 0.01558426 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +1.6%  mass < 78.26
        + 0.01468119 * max(0.0, 0.09591084 - Q.tau1) / 0.02629125   # +1.5%  tau1 < 0.09591
        + 0.01297157 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # +1.3%  girth2 < 0.007877
        + 0.01116139 * max(0.0, 91.19 - Q.mass) / 16.46423   # +1.1%  mass < 91.19
        - 0.0100055 * max(0.0, Q.sd_mass - 98.05743) / 4.276978   # -1.0%  sd_mass > 98.06
        - 0.009458363 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.9%  psi_0p3 > 0.998
        + 0.009066096 * max(0.0, 0.2639816 - Q.sd_rg) / 0.1327654   # +0.9%  sd_rg < 0.264
        + 0.008057387 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # +0.8%  lam1 < 0.00619
        + 0.007040681 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +0.7%  mass_over_sum_pt < 0.1182
        + 0.006873707 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +0.7%  lam1 < 0.01174
        + 0.005932651 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +0.6%  tau21_b2 < 0.2352
        - 0.005543906 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # -0.6%  psi_0p2 > 0.9314
        + 0.005361736 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.007828415   # +0.5%  mass < 82.85 and psi_0p3 > 0.998
        + 0.005121504 * max(0.0, 0.1596365 - Q.sd_rg) / 0.05492863   # +0.5%  sd_rg < 0.1596
        + 0.004774186 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +0.5%  tau1 < 0.07709
        - 0.004327005 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt) / 11.66143   # -0.4%  tau21_b2 < 0.2352 and sum_pt < 1261
        - 0.004254308 * max(0.0, 0.00363788 - Q.e2_sq) / 0.0004963098   # -0.4%  e2_sq < 0.003638
        - 0.0041656 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # -0.4%  tau1 < 0.08787
        + 0.004037908 * max(0.0, 9.0 - Q.n_dr_0p2_0p4) / 3.177318   # +0.4%  n_dr_0p2_0p4 < 9
        + 0.003767268 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.z_top50_slots - 0.978741) / 1.779326e-05   # +0.4%  psi_0p3 > 0.9974 and z_top50_slots > 0.9787
        + 0.003680866 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +0.4%  mass < 92.86
        + 0.003094837 * max(0.0, Q.mass_top20 - 85.79457) / 7.06893   # +0.3%  mass_top20 > 85.79
        - 0.002890302 * max(0.0, Q.mass_top20 - 73.35236) / 11.28399   # -0.3%  mass_top20 > 73.35
        + 0.001575981 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.02947847   # +0.2%  mass < 82.85 and psi_0p3 < 0.9985
        - 0.001352736 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.4947602 - Q.z_dr_0_0p05) / 0.572942   # -0.1%  mass < 92.86 and z_dr_0_0p05 < 0.4948
        + 0.001176459 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.1%  n_dr_0p2_0p4 < 6
        - 0.001133592 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 5.106465e-05   # -0.1%  tau21_b2 < 0.2352 and mass_over_sum_pt_sq < 0.007873
        - 0.0004677528 * max(0.0, 0.008241985 - Q.lam1) * max(0.0, Q.zdr_0 - 0.01564747) / 7.458275e-07   # -0.0%  lam1 < 0.008242 and zdr_0 > 0.01565
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 15.25;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.2481 * (0.1213569
        - 0.08924777 * max(0.0, 0.009614971 - Q.width) / 0.003502545   # -8.9%  width < 0.009615
        + 0.05906325 * max(0.0, Q.mass - 74.25181) / 24.07648   # +5.9%  mass > 74.25
        + 0.05350909 * max(0.0, Q.mass - 87.36377) / 16.57741   # +5.4%  mass > 87.36
        - 0.04985975 * max(0.0, 39.0 - Q.n_for_90pct) / 18.29776   # -5.0%  n_for_90pct < 39
        - 0.04847355 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -4.8%  mass_top50 > 82.04
        + 0.04707082 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt) / 2.336698   # +4.7%  mass_over_sum_pt > 0.07697 and sum_pt < 1116
        - 0.04702506 * max(0.0, Q.mass - 101.0497) / 12.31084   # -4.7%  mass > 101
        - 0.04451733 * max(0.0, Q.n_for_90pct - 7.0) / 13.93571   # -4.5%  n_for_90pct > 7
        + 0.04200049 * max(0.0, Q.mass - 64.48544) / 31.19164   # +4.2%  mass > 64.49
        + 0.04129392 * max(0.0, Q.girth2_top20 - 0.008031209) / 0.002906852   # +4.1%  girth2_top20 > 0.008031
        + 0.03496515 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +3.5%  girth2_top40 < 0.008841
        - 0.031903 * max(0.0, Q.z_top50_slots - 0.9704436) / 0.02331682   # -3.2%  z_top50_slots > 0.9704
        - 0.0312693 * max(0.0, Q.girth2_top20 - 0.006043209) / 0.003593885   # -3.1%  girth2_top20 > 0.006043
        + 0.02941686 * max(0.0, 1028.184 - Q.sum_pt) / 26.95739   # +2.9%  sum_pt < 1028
        - 0.0273448 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # -2.7%  mass_over_sum_pt > 0.07697
        + 0.0268644 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +2.7%  sum_pt < 1017
        - 0.02499037 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt) / 0.0005328922   # -2.5%  girth2_top40 > 0.005197 and log_sum_pt < 7.017
        - 0.02430682 * max(0.0, 6.930088 - Q.log_sum_pt) / 0.02522479   # -2.4%  log_sum_pt < 6.93
        - 0.02285675 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # -2.3%  sum_pt_top40 < 1025
        + 0.02190723 * max(0.0, 0.007463985 - Q.girth2_top30) / 0.00233708   # +2.2%  girth2_top30 < 0.007464
        - 0.01920476 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -1.9%  n_dr_0p2_0p4 < 15
        + 0.01521022 * max(0.0, Q.girth2_top40 - 0.005196966) / 0.004725809   # +1.5%  girth2_top40 > 0.005197
        + 0.01401777 * max(0.0, Q.n_dr_0p2_0p4 - 3.0) / 6.482203   # +1.4%  n_dr_0p2_0p4 > 3
        - 0.01365582 * max(0.0, Q.mass - 125.1) / 7.098976   # -1.4%  mass > 125.1
        + 0.01314967 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +1.3%  lam1 < 0.007671
        - 0.01307733 * max(0.0, Q.e3 - 5.13841e-05) / 6.821576e-05   # -1.3%  e3 > 5.138e-05
        + 0.01053675 * max(0.0, Q.mass_top50 - 117.0487) / 7.705579   # +1.1%  mass_top50 > 117
        + 0.009579034 * max(0.0, 9.0 - Q.n_dr_0p2_0p4) / 3.177318   # +1.0%  n_dr_0p2_0p4 < 9
        - 0.00881227 * max(0.0, 0.02146578 - Q.girth2_top15) / 0.01470441   # -0.9%  girth2_top15 < 0.02147
        + 0.008711805 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.6773544   # +0.9%  sum_pt < 1028 and dr_max_012 > 0.1828
        - 0.008286333 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -0.8%  mass_over_sum_pt > 0.1182
        - 0.008135119 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.5693287   # -0.8%  sum_pt < 1017 and dr_max_012 > 0.1828
        + 0.007906192 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.8%  e2 > 0.04755
        + 0.007439038 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.7%  sj2_dr > 0.2232
        + 0.005059654 * max(0.0, Q.sum_pt_top30 - 978.0762) / 49.23409   # +0.5%  sum_pt_top30 > 978.1
        - 0.005016208 * max(0.0, Q.sum_pt_top50 - 889.8503) / 149.655   # -0.5%  sum_pt_top50 > 889.9
        + 0.004133172 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # +0.4%  C2 > 0.06656
        - 0.003934628 * max(0.0, Q.mass - 64.48544) * max(0.0, Q.zdr_0 - 0.0008718296) / 0.4297651   # -0.4%  mass > 64.49 and zdr_0 > 0.0008718
        + 0.003700802 * max(0.0, 0.8316924 - Q.z_top15_slots) / 0.04702674   # +0.4%  z_top15_slots < 0.8317
        + 0.003563161 * max(0.0, 0.0795038 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.38202) / 0.1361121   # +0.4%  tau2 < 0.0795 and sj3_mass1 > 13.38
        - 0.003069391 * max(0.0, 0.3628388 - Q.psi_0p1) / 0.02326895   # -0.3%  psi_0p1 < 0.3628
        + 0.002937423 * max(0.0, 1001.523 - Q.sum_pt_top40) / 26.72886   # +0.3%  sum_pt_top40 < 1002
        + 0.002604708 * max(0.0, Q.n_dr_0p1_0p2 - 19.0) / 1.732133   # +0.3%  n_dr_0p1_0p2 > 19
        - 0.002573873 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, Q.sj2_dr - 0.1512157) / 6.78729e-05   # -0.3%  girth2_top30 < 0.007464 and sj2_dr > 0.1512
        + 0.002504709 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.03720487   # +0.3%  z_dr_0p1_0p2 > 0.334
        + 0.002230159 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # +0.2%  e3 > 0.0003372
        - 0.00181804 * max(0.0, Q.mass_top20 - 119.2969) / 1.8402   # -0.2%  mass_top20 > 119.3
        + 0.0009207756 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.09686657   # +0.1%  n_dr_0p2_0p4 < 15 and z_dr_0p1_0p2 > 0.334
        - 0.0003254909 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # -0.0%  girth2 < 0.007877
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 43.83;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 43.82828 * (-0.02299392
        - 0.2580119 * max(0.0, 160.8 - Q.mass) / 72.78344   # -25.8%  mass < 160.8
        + 0.2216049 * max(0.0, 162.8363 - Q.mass) / 74.60496   # +22.2%  mass < 162.8
        + 0.1516394 * max(0.0, 172.8 - Q.mass) / 83.78792   # +15.2%  mass < 172.8
        - 0.1256991 * max(0.0, 143.7876 - Q.mass) / 57.99337   # -12.6%  mass < 143.8
        + 0.06582409 * max(0.0, 138.8977 - Q.mass_top50) / 55.01559   # +6.6%  mass_top50 < 138.9
        - 0.04755897 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -4.8%  mass_top40 < 163.3
        + 0.02621773 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +2.6%  mass < 92.86
        - 0.01744485 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # -1.7%  mass_top50 < 92.17
        + 0.01471079 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +1.5%  mass < 82.85
        - 0.008120774 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -0.8%  mass < 64.49
        + 0.006557576 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # +0.7%  girth2_top40 < 0.00626
        + 0.006406278 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +0.6%  mass_top40 < 80.89
        + 0.005913648 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +0.6%  sum_pt < 986.1
        - 0.005355362 * max(0.0, Q.z_top40_slots - 0.9574183) / 0.02831517   # -0.5%  z_top40_slots > 0.9574
        - 0.005051863 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -0.5%  log_sum_pt < 6.903
        + 0.004985233 * max(0.0, 125.1 - Q.mass_top40) / 45.33615   # +0.5%  mass_top40 < 125.1
        + 0.004666095 * max(0.0, Q.girth - 0.09749958) / 0.008315822   # +0.5%  girth > 0.0975
        - 0.004298253 * max(0.0, Q.girth2_top30 - 0.0008564881) / 0.007661497   # -0.4%  girth2_top30 > 0.0008565
        - 0.003185612 * max(0.0, Q.girth - 0.1402186) / 0.001871547   # -0.3%  girth > 0.1402
        + 0.002829038 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +0.3%  LHA > 0.4042
        + 0.002331924 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +0.2%  lam1 > 0.01174
        - 0.002064337 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt) / 447.38   # -0.2%  mass_top40 < 80.89 and sum_pt < 1035
        + 0.001833152 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.2%  lam1 < 0.004673
        + 0.001426032 * max(0.0, 73.33139 - Q.mass_top30) / 11.37701   # +0.1%  mass_top30 < 73.33
        - 0.001225487 * max(0.0, 0.00217213 - Q.girth2_top40) / 0.0002075259   # -0.1%  girth2_top40 < 0.002172
        + 0.00112889 * max(0.0, 956.2133 - Q.sum_pt_top40) / 14.00451   # +0.1%  sum_pt_top40 < 956.2
        - 0.001007441 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -0.1%  tau1 < 0.05445
        + 0.0009965752 * max(0.0, 125.1 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 42.49581   # +0.1%  mass_top40 < 125.1 and n_dr_0p2_0p4 > 9
        + 0.0008558196 * max(0.0, Q.D2 - 2.178951) / 1.450771   # +0.1%  D2 > 2.179
        + 0.0005236203 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40) / 104.3982   # +0.1%  mass_top40 < 80.89 and sum_pt_top40 < 906.6
        + 0.0003170177 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258) / 5.834847   # +0.0%  sum_pt_top40 < 956.2 and soft4_pt > 1.789
        - 0.0002082403 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047) / 1.704178   # -0.0%  sum_pt_top40 < 956.2 and soft3_pt > 2.873
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 17.14;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.13562 * (0.03785081
        - 0.1243784 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -12.4%  girth < 0.1207
        + 0.123425 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # +12.3%  tau1 < 0.1507
        - 0.07703577 * max(0.0, Q.mass - 64.48544) / 31.19164   # -7.7%  mass > 64.49
        + 0.07587429 * max(0.0, Q.mass - 78.26182) / 21.33658   # +7.6%  mass > 78.26
        + 0.0463047 * max(0.0, Q.e2 - 0.01256572) / 0.01912291   # +4.6%  e2 > 0.01257
        + 0.04500426 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +4.5%  mass < 89.74
        - 0.04392543 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -4.4%  mass < 101
        - 0.04299585 * max(0.0, 65.20727 - Q.sj2_mass1) / 36.50254   # -4.3%  sj2_mass1 < 65.21
        + 0.02731017 * max(0.0, 86.4 - Q.mass) / 13.67632   # +2.7%  mass < 86.4
        + 0.02469611 * max(0.0, 935.1043 - Q.sum_pt_top15) / 95.709   # +2.5%  sum_pt_top15 < 935.1
        - 0.02267793 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -2.3%  psi_0p3 > 0.998
        + 0.02169439 * max(0.0, 2.975532 - Q.D2) / 0.9290697   # +2.2%  D2 < 2.976
        + 0.02133352 * max(0.0, 0.01655983 - Q.girth2_top20) / 0.01003486   # +2.1%  girth2_top20 < 0.01656
        - 0.02038223 * max(0.0, 0.04466492 - Q.tau1) / 0.005217805   # -2.0%  tau1 < 0.04466
        + 0.019934 * max(0.0, 0.04828819 - Q.tau2) / 0.02103241   # +2.0%  tau2 < 0.04829
        - 0.01955165 * max(0.0, Q.mass - 143.7876) / 3.946979   # -2.0%  mass > 143.8
        - 0.01906661 * max(0.0, Q.mass_top30 - 78.53034) / 14.31024   # -1.9%  mass_top30 > 78.53
        + 0.01888068 * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0004356739   # +1.9%  psi_0p3 > 0.9985
        - 0.01455311 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -1.5%  e2 < 0.04359
        + 0.01392606 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.4%  z_dr_0p2_0p4 < 0.09123
        - 0.01228071 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -1.2%  n_dr_0p2_0p4 < 11
        + 0.01188624 * max(0.0, Q.mass_top30 - 89.17293) / 10.17054   # +1.2%  mass_top30 > 89.17
        + 0.01128685 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +1.1%  mass_top50 > 136.8
        + 0.01037047 * max(0.0, Q.mass_top5 - 22.18342) / 12.67741   # +1.0%  mass_top5 > 22.18
        + 0.01023338 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +1.0%  dr_0 < 0.06413
        - 0.01013616 * max(0.0, Q.pt_dispersion - 0.27462) / 0.06481184   # -1.0%  pt_dispersion > 0.2746
        - 0.009899867 * max(0.0, 72.18744 - Q.mass_top15) / 20.20734   # -1.0%  mass_top15 < 72.19
        - 0.00868254 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # -0.9%  girth2_top30 < 0.005402
        - 0.008233656 * max(0.0, 0.005312783 - Q.girth2_top20) / 0.001437214   # -0.8%  girth2_top20 < 0.005313
        - 0.007943451 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.9985421) / 6.34706e-06   # -0.8%  girth2_top10 < 0.01977 and psi_0p3 > 0.9985
        - 0.007767614 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, 0.4357228 - Q.max_dr) / 0.001259784   # -0.8%  girth2_top10 < 0.01977 and max_dr < 0.4357
        + 0.007667791 * max(0.0, 0.008824206 - Q.zdr_1) / 0.00383206   # +0.8%  zdr_1 < 0.008824
        - 0.006408697 * max(0.0, Q.z_dr_0_0p05 - 0.7674734) / 0.05221088   # -0.6%  z_dr_0_0p05 > 0.7675
        - 0.005722536 * max(0.0, Q.mass - 162.8363) / 1.509832   # -0.6%  mass > 162.8
        - 0.00568678 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # -0.6%  sum_pt < 986.1
        - 0.005650259 * max(0.0, Q.e2 - 0.03263075) / 0.006288927   # -0.6%  e2 > 0.03263
        - 0.005582941 * max(0.0, Q.mass_over_sum_pt - 0.1606361) / 0.001344467   # -0.6%  mass_over_sum_pt > 0.1606
        + 0.004159534 * max(0.0, 0.01541561 - Q.e2) / 0.001436351   # +0.4%  e2 < 0.01542
        - 0.003616034 * max(0.0, 2.975532 - Q.D2) * max(0.0, Q.sj2_dr - 0.2070855) / 0.0198522   # -0.4%  D2 < 2.976 and sj2_dr > 0.2071
        + 0.003325459 * max(0.0, Q.girth2_top40 - 0.02497133) / 0.0004755075   # +0.3%  girth2_top40 > 0.02497
        + 0.003138172 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, Q.psi_0p3 - 0.9985421) / 4.280363e-07   # +0.3%  girth2_top30 < 0.005402 and psi_0p3 > 0.9985
        + 0.002964192 * max(0.0, 65.20727 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 7.769597) / 168.6958   # +0.3%  sj2_mass1 < 65.21 and sj2_mass2 > 7.77
        - 0.002656547 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.3%  sum_pt_top10 > 943.7
        - 0.002617448 * max(0.0, 0.01541561 - Q.e2) * max(0.0, 0.7904923 - Q.pt2_over_pt0) / 0.000518365   # -0.3%  e2 < 0.01542 and pt2_over_pt0 < 0.7905
        + 0.002029314 * max(0.0, 59.40777 - Q.mass_top5) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1) / 22.61156   # +0.2%  mass_top5 < 59.41 and z_dr_0p05_0p1 < 0.8509
        - 0.001765757 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # -0.2%  sum_pt_top50 < 959.1
        + 0.001483527 * max(0.0, Q.mass_top50 - 160.8) / 1.246461   # +0.1%  mass_top50 > 160.8
        - 0.001429546 * max(0.0, Q.mass_top50 - 136.785) * max(0.0, Q.soft5_z - 0.001594761) / 0.001639485   # -0.1%  mass_top50 > 136.8 and soft5_z > 0.001595
        + 0.001056637 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897) / 0.001013644   # +0.1%  mass > 162.8 and soft5_z > 0.001435
        - 0.000749692 * max(0.0, Q.mass_top50 - 172.8) / 0.5062389   # -0.1%  mass_top50 > 172.8
        - 0.0005114084 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109) / 0.0004735247   # -0.1%  mass_top50 > 160.8 and soft4_z > 0.001721
        - 0.0001066122 * max(0.0, Q.sum_pt_top10 - 943.6922) * max(0.0, -0.07794189 - Q.eta_0) / 0.002015064   # -0.0%  sum_pt_top10 > 943.7 and eta_0 < -0.07794
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 10.53;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.52817 * (-0.03656695
        + 0.1258844 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +12.6%  mass < 92.86
        - 0.07794433 * max(0.0, 0.076787 - Q.girth) / 0.02105267   # -7.8%  girth < 0.07679
        + 0.06957858 * max(0.0, 97.93004 - Q.mass_top50) / 22.03533   # +7.0%  mass_top50 < 97.93
        - 0.06155811 * max(0.0, 80.35535 - Q.mass_top50) / 11.1399   # -6.2%  mass_top50 < 80.36
        + 0.04738283 * max(0.0, 0.08589404 - Q.girth) / 0.02745918   # +4.7%  girth < 0.08589
        - 0.04204443 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -4.2%  lam1 < 0.006717
        + 0.0405594 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +4.1%  e2_sq < 0.008184
        + 0.03507406 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +3.5%  e2 < 0.0303
        - 0.0314101 * max(0.0, 0.006929741 - Q.girth2_top30) / 0.002004687   # -3.1%  girth2_top30 < 0.00693
        - 0.03093885 * max(0.0, 0.00625621 - Q.girth2_top10) / 0.002474789   # -3.1%  girth2_top10 < 0.006256
        + 0.03035533 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +3.0%  psi_0p3 > 0.9897
        + 0.02994548 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # +3.0%  mass_over_sum_pt_sq < 0.009595
        + 0.02696662 * max(0.0, 0.00406126 - Q.girth2_top10) / 0.001298509   # +2.7%  girth2_top10 < 0.004061
        - 0.02451193 * max(0.0, 80.4 - Q.mass) / 10.6806   # -2.5%  mass < 80.4
        - 0.02437212 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -2.4%  mass_top40 < 83.33
        - 0.02377238 * max(0.0, 0.07999061 - Q.mass_over_sum_pt) / 0.01176803   # -2.4%  mass_over_sum_pt < 0.07999
        + 0.02350391 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +2.4%  mass_top30 < 60.44
        + 0.02262013 * max(0.0, 15.0 - Q.n_dr_0p1_0p2) / 5.139044   # +2.3%  n_dr_0p1_0p2 < 15
        - 0.02248288 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # -2.2%  psi_0p2 > 0.9314
        + 0.02244852 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +2.2%  n_dr_0p2_0p4 < 10
        + 0.01976978 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +2.0%  e2 < 0.02516
        - 0.01404158 * max(0.0, 0.008031209 - Q.girth2_top20) / 0.00309849   # -1.4%  girth2_top20 < 0.008031
        + 0.01369832 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +1.4%  n_dr_0p2_0p4 < 6
        - 0.01278956 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2) / 0.005195774   # -1.3%  n_dr_0p2_0p4 < 10 and girth2 < 0.005532
        - 0.01245747 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.004495205 - Q.zdr_4) / 0.05697234   # -1.2%  mass < 92.86 and zdr_4 < 0.004495
        + 0.01240318 * max(0.0, 80.35535 - Q.mass_top50) * max(0.0, 0.003932029 - Q.zdr_4) / 0.03420305   # +1.2%  mass_top50 < 80.36 and zdr_4 < 0.003932
        - 0.0115656 * max(0.0, 86.4 - Q.mass_top30) / 18.30352   # -1.2%  mass_top30 < 86.4
        + 0.01140852 * max(0.0, Q.D2 - 2.178951) / 1.450771   # +1.1%  D2 > 2.179
        - 0.01076209 * max(0.0, Q.z_top5_slots - 0.534626) / 0.08390497   # -1.1%  z_top5_slots > 0.5346
        - 0.00994083 * max(0.0, 3.793233e-05 - Q.e3) / 1.004564e-05   # -1.0%  e3 < 3.793e-05
        - 0.009857847 * max(0.0, 101.0497 - Q.mass) * max(0.0, 891.875 - Q.sum_pt_top10) / 2090.674   # -1.0%  mass < 101 and sum_pt_top10 < 891.9
        + 0.009595958 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # +1.0%  girth < 0.07374
        - 0.00894636 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0) / 30.16986   # -0.9%  n_dr_0p2_0p4 < 10 and n_particles > 34
        - 0.006653121 * max(0.0, Q.D2 - 3.814159) / 0.870654   # -0.7%  D2 > 3.814
        - 0.006514169 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # -0.7%  e2 < 0.03876
        + 0.005360129 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0402832 - Q.phi_0) / 0.166757   # +0.5%  n_dr_0p2_0p4 < 10 and phi_0 < 0.04028
        - 0.004184091 * max(0.0, 0.06030419 - Q.mass_over_sum_pt) / 0.005383371   # -0.4%  mass_over_sum_pt < 0.0603
        + 0.003749004 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # +0.4%  girth2_top30 < 0.005402
        - 0.002948026 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -0.3%  mass < 64.49
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 3.678;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.677508 * (-0.06293411
        + 0.2659071 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +26.6%  mass < 82.85
        + 0.1319198 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.04417088   # +13.2%  mass < 86.4 and lam2 < 0.003688
        + 0.1136932 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +11.4%  mass < 74.25
        - 0.09291432 * max(0.0, 62.55 - Q.mass) / 5.464058   # -9.3%  mass < 62.55
        - 0.06794137 * max(0.0, 0.006142802 - Q.girth2_top15) / 0.002080651   # -6.8%  girth2_top15 < 0.006143
        - 0.0525404 * max(0.0, 0.06895248 - Q.mass_over_sum_pt) / 0.007729512   # -5.3%  mass_over_sum_pt < 0.06895
        - 0.0449353 * max(0.0, 74.78616 - Q.mass_top40) / 9.958349   # -4.5%  mass_top40 < 74.79
        - 0.04298842 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.5993597   # -4.3%  mass < 86.4 and z_dr_0p1_0p2 < 0.06473
        - 0.03707696 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.03580539   # -3.7%  mass < 86.4 and psi_0p3 < 0.9985
        - 0.03210129 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -3.2%  mass < 53.87
        - 0.02901112 * max(0.0, 86.4 - Q.mass) / 13.67632   # -2.9%  mass < 86.4
        + 0.0172757 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, Q.n_dr_0p05_0p1 - 1.0) / 124.4066   # +1.7%  mass_top40 < 89.68 and n_dr_0p05_0p1 > 1
        + 0.01618988 * max(0.0, Q.psi_0p1 - 0.9538343) / 0.006310723   # +1.6%  psi_0p1 > 0.9538
        - 0.00963058 * max(0.0, 6.856375 - Q.log_sum_pt) * max(0.0, 0.002396991 - Q.lam2) / 7.289393e-06   # -1.0%  log_sum_pt < 6.856 and lam2 < 0.002397
        - 0.009218064 * max(0.0, Q.sj3_pair_mass_max - 120.6) / 2.130243   # -0.9%  sj3_pair_mass_max > 120.6
        - 0.00776822 * max(0.0, 6.856375 - Q.log_sum_pt) / 0.008053459   # -0.8%  log_sum_pt < 6.856
        + 0.007554423 * max(0.0, 0.0007431905 - Q.girth2_top10) * max(0.0, 1069.671 - Q.sum_pt_top40) / 0.006568426   # +0.8%  girth2_top10 < 0.0007432 and sum_pt_top40 < 1070
        + 0.007190341 * max(0.0, Q.sd_mass - 125.1) / 1.231011   # +0.7%  sd_mass > 125.1
        + 0.006551594 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min) / 54.09437   # +0.7%  sj3_pair_mass_max > 120.6 and sj3_pair_mass_min < 76.6
        + 0.004331683 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926) / 0.5295314   # +0.4%  sj3_pair_mass_max > 120.6 and z_dr_0p1_0p2 > 0.2865
        - 0.003260264 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.985099 - Q.z_top50_slots) / 0.008644046   # -0.3%  mass < 86.4 and z_top50_slots < 0.9851
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 8.528;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.528282 * (0.2377973
        - 0.1257614 * max(0.0, 0.02580396 - Q.mass_over_sum_pt_sq) / 0.01697662   # -12.6%  mass_over_sum_pt_sq < 0.0258
        + 0.07133143 * max(0.0, 0.02550569 - Q.girth2_top50) / 0.01683664   # +7.1%  girth2_top50 < 0.02551
        - 0.06615734 * max(0.0, 160.8 - Q.mass_top40) / 76.75884   # -6.6%  mass_top40 < 160.8
        + 0.05157682 * Q.n_particles / 45.81523   # +5.2%  n_particles
        + 0.05113308 * max(0.0, 0.02550569 - Q.girth2_top50) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.0005189214   # +5.1%  girth2_top50 < 0.02551 and psi_0p3 > 0.9638
        + 0.04217197 * max(0.0, Q.mass_top50 - 27.46086) / 60.61506   # +4.2%  mass_top50 > 27.46
        - 0.04088969 * max(0.0, 6.910131 - Q.log_sum_pt) / 0.017275   # -4.1%  log_sum_pt < 6.91
        - 0.04064525 * max(0.0, Q.mass - 143.7876) / 3.946979   # -4.1%  mass > 143.8
        + 0.03875432 * max(0.0, Q.mass - 74.25181) / 24.07648   # +3.9%  mass > 74.25
        - 0.03009498 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.8310045 - Q.tau21_b2) / 33.42141   # -3.0%  sum_pt < 1085 and tau21_b2 < 0.831
        + 0.02773959 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +2.8%  sum_pt_top40 < 1053
        - 0.0272922 * max(0.0, Q.mass - 101.0497) / 12.31084   # -2.7%  mass > 101
        + 0.02571571 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +2.6%  mass_top50 > 136.8
        - 0.02543305 * max(0.0, Q.mass - 172.8) * max(0.0, Q.soft5_z - 0.0004140594) / 0.001251279   # -2.5%  mass > 172.8 and soft5_z > 0.0004141
        - 0.02505633 * max(0.0, Q.mass_top10 - 23.38894) / 27.40926   # -2.5%  mass_top10 > 23.39
        + 0.02023273 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # +2.0%  sum_pt_top40 < 1007
        - 0.02017939 * max(0.0, 1034.834 - Q.sum_pt) / 30.78807   # -2.0%  sum_pt < 1035
        + 0.01879807 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +1.9%  mass_over_sum_pt > 0.1709
        + 0.01646649 * max(0.0, 0.007820315 - Q.girth2_top50) / 0.002264874   # +1.6%  girth2_top50 < 0.00782
        - 0.01621946 * max(0.0, Q.mass - 160.8) / 1.724663   # -1.6%  mass > 160.8
        - 0.01581317 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -1.6%  mass_over_sum_pt_sq > 0.0292
        - 0.01558176 * max(0.0, 16.0 - Q.n_for_90pct) / 1.776607   # -1.6%  n_for_90pct < 16
        - 0.01522934 * max(0.0, 966.0633 - Q.sum_pt_top30) / 29.99923   # -1.5%  sum_pt_top30 < 966.1
        - 0.01317341 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -1.3%  sum_pt < 1085
        + 0.0119807 * max(0.0, Q.mass_top5 - 33.71058) / 7.394508   # +1.2%  mass_top5 > 33.71
        + 0.01106467 * max(0.0, 997.0189 - Q.sum_pt_top50) / 17.90852   # +1.1%  sum_pt_top50 < 997
        - 0.01085308 * max(0.0, 1007.788 - Q.sum_pt) / 17.7122   # -1.1%  sum_pt < 1008
        - 0.01008318 * max(0.0, Q.mass - 136.785) * max(0.0, Q.D2 - 0.8942376) / 6.48176   # -1.0%  mass > 136.8 and D2 > 0.8942
        - 0.009897838 * max(0.0, Q.mass_top50 - 136.785) * max(0.0, Q.soft6_z - 0.0008188601) / 0.003727182   # -1.0%  mass_top50 > 136.8 and soft6_z > 0.0008189
        - 0.009470537 * max(0.0, Q.mass - 172.8) / 0.7291439   # -0.9%  mass > 172.8
        - 0.008505364 * max(0.0, Q.mass_top50 - 92.16545) / 13.44564   # -0.9%  mass_top50 > 92.17
        - 0.008357855 * max(0.0, Q.mass_top50 - 92.16545) * max(0.0, 0.001923089 - Q.soft7_z) / 0.006887627   # -0.8%  mass_top50 > 92.17 and soft7_z < 0.001923
        - 0.00830288 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft6_z - 0.0005748372) / 0.002073713   # -0.8%  mass > 162.8 and soft6_z > 0.0005748
        + 0.007244069 * max(0.0, 16.0 - Q.n_for_90pct) * max(0.0, 9.676985 - Q.D2) / 8.10578   # +0.7%  n_for_90pct < 16 and D2 < 9.677
        + 0.007061922 * max(0.0, Q.mass - 172.8) * max(0.0, Q.D2 - 0.8942376) / 0.9099067   # +0.7%  mass > 172.8 and D2 > 0.8942
        + 0.005628588 * max(0.0, Q.mass_top30 - 138.3818) / 1.758216   # +0.6%  mass_top30 > 138.4
        + 0.005340142 * max(0.0, Q.mass_top50 - 168.9698) / 0.6651775   # +0.5%  mass_top50 > 169
        + 0.00518467 * max(0.0, Q.mass - 172.8) * max(0.0, Q.soft6_z - 0.0004464147) / 0.001308464   # +0.5%  mass > 172.8 and soft6_z > 0.0004464
        + 0.005174782 * max(0.0, Q.sum_pt_top20 - 956.5062) / 37.46564   # +0.5%  sum_pt_top20 > 956.5
        + 0.004673918 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.2179035 - Q.sj2_zsoft) / 2.389974   # +0.5%  sum_pt_top40 < 1053 and sj2_zsoft < 0.2179
        - 0.004060783 * max(0.0, Q.mass - 136.785) / 5.043396   # -0.4%  mass > 136.8
        + 0.003123454 * max(0.0, Q.mass - 74.25181) * max(0.0, 1028.184 - Q.sum_pt) / 733.8204   # +0.3%  mass > 74.25 and sum_pt < 1028
        + 0.003099626 * max(0.0, 1034.834 - Q.sum_pt) * max(0.0, 16.64062 - Q.pt_9) / 29.01849   # +0.3%  sum_pt < 1035 and pt_9 < 16.64
        + 0.002968833 * max(0.0, Q.mass - 160.8) * max(0.0, Q.D2 - 0.8942376) / 2.121004   # +0.3%  mass > 160.8 and D2 > 0.8942
        + 0.002949328 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.1312677 - Q.dr_13) / 0.0002562188   # +0.3%  log_sum_pt < 6.811 and dr_13 < 0.1313
        - 0.00255124 * max(0.0, Q.mass - 172.8) * max(0.0, 0.04568661 - Q.z_6) / 0.006788705   # -0.3%  mass > 172.8 and z_6 < 0.04569
        + 0.002335719 * max(0.0, Q.mass - 172.8) * max(0.0, 56.53125 - Q.pt_6) / 9.768822   # +0.2%  mass > 172.8 and pt_6 < 56.53
        - 0.001996055 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.005041702 - Q.zdr_11) / 1.572361e-05   # -0.2%  log_sum_pt < 6.811 and zdr_11 < 0.005042
        + 0.001784221 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 2.680253 - Q.D2) / 0.001923611   # +0.2%  log_sum_pt < 6.811 and D2 < 2.68
        + 0.001702878 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.sd_zg - 0.4462823) / 1.289737e-05   # +0.2%  log_sum_pt > 7.139 and sd_zg > 0.4463
        - 0.001086531 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.00317537 - Q.C3) / 7.466262e-06   # -0.1%  log_sum_pt > 7.139 and C3 < 0.003175
        - 0.001055071 * max(0.0, Q.mass - 162.8363) / 1.509832   # -0.1%  mass > 162.8
        - 0.001015071 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.04047238 - Q.dr_9) / 6.697833e-05   # -0.1%  log_sum_pt > 7.139 and dr_9 < 0.04047
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 24.54;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.53805 * (-0.01063817
        - 0.07709221 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -7.7%  mass < 82.85
        + 0.0706638 * max(0.0, 0.140939 - Q.mass_over_sum_pt) / 0.05796036   # +7.1%  mass_over_sum_pt < 0.1409
        - 0.05408183 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) / 0.0178873   # -5.4%  mass_over_sum_pt < 0.09047
        + 0.05167004 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +5.2%  mass_over_sum_pt < 0.1182
        - 0.05096786 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -5.1%  lam1 < 0.008242
        - 0.04348495 * max(0.0, 0.002582316 - Q.girth2_top30) / 0.0003518773   # -4.3%  girth2_top30 < 0.002582
        - 0.04258384 * max(0.0, 91.19 - Q.mass) / 16.46423   # -4.3%  mass < 91.19
        + 0.0369835 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +3.7%  mass_over_sum_pt < 0.09795
        - 0.0364851 * max(0.0, 97.93004 - Q.mass_top50) / 22.03533   # -3.6%  mass_top50 < 97.93
        + 0.03155661 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +3.2%  lam1 < 0.007671
        - 0.0315479 * max(0.0, 89.74183 - Q.mass) / 15.55633   # -3.2%  mass < 89.74
        + 0.02973955 * max(0.0, 77.37641 - Q.mass_top50) / 9.980855   # +3.0%  mass_top50 < 77.38
        + 0.02718342 * max(0.0, 89.6788 - Q.mass_top40) / 17.47899   # +2.7%  mass_top40 < 89.68
        + 0.02476486 * max(0.0, 125.1 - Q.mass) / 42.45775   # +2.5%  mass < 125.1
        - 0.02410301 * max(0.0, 117.0487 - Q.mass_top50) / 36.94196   # -2.4%  mass_top50 < 117
        - 0.02338564 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -2.3%  girth2_top30 < 0.01216
        + 0.02277519 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +2.3%  psi_0p3 > 0.9924
        - 0.02048604 * max(0.0, 0.01897915 - Q.girth2_top40) / 0.01130444   # -2.0%  girth2_top40 < 0.01898
        + 0.01939569 * max(0.0, 0.08873143 - Q.mass_over_sum_pt) / 0.01671255   # +1.9%  mass_over_sum_pt < 0.08873
        - 0.01724618 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -1.7%  girth2_top20 < 0.01083
        + 0.01616214 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +1.6%  e2 < 0.0303
        - 0.01599174 * max(0.0, 80.4 - Q.mass) / 10.6806   # -1.6%  mass < 80.4
        + 0.01473508 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # +1.5%  girth2_top5 < 0.007164
        - 0.01471484 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # -1.5%  sd_rg < 0.3017
        + 0.01272097 * max(0.0, 6.941997 - Q.log_sum_pt) / 0.03180972   # +1.3%  log_sum_pt < 6.942
        + 0.01147561 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +1.1%  lam1 < 0.007259
        + 0.01111159 * max(0.0, 0.1778185 - Q.sd_rg) / 0.06576479   # +1.1%  sd_rg < 0.1778
        - 0.01091357 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p2 - 0.9087063) / 0.00142529   # -1.1%  mass_over_sum_pt < 0.09047 and psi_0p2 > 0.9087
        + 0.01049273 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # +1.0%  mass_top50 < 71.8
        - 0.01048284 * max(0.0, 0.076787 - Q.girth) / 0.02105267   # -1.0%  girth < 0.07679
        - 0.01047116 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # -1.0%  log_sum_pt < 6.989
        - 0.01026412 * max(0.0, 0.07852883 - Q.mass_over_sum_pt) / 0.01107516   # -1.0%  mass_over_sum_pt < 0.07853
        - 0.01005245 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # -1.0%  tau2 < 0.0795
        - 0.007416281 * max(0.0, 0.007164202 - Q.girth2_top5) * max(0.0, Q.n_pt_above_10 - 13.0) / 0.02038471   # -0.7%  girth2_top5 < 0.007164 and n_pt_above_10 > 13
        + 0.007046043 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +0.7%  n_dr_0p2_0p4 < 18
        + 0.006975666 * max(0.0, 85.8667 - Q.mass_top50) / 13.95835   # +0.7%  mass_top50 < 85.87
        + 0.006519247 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # +0.7%  e2 < 0.04359
        - 0.005828311 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1881908 - Q.sd_rg) / 0.0002726472   # -0.6%  psi_0p3 > 0.9924 and sd_rg < 0.1882
        + 0.005713576 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +0.6%  tau21_b2 < 0.3425
        - 0.005383797 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -0.5%  tau1 < 0.1073
        - 0.005158624 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2404747) / 0.01288721   # -0.5%  N2 < 0.4227 and max_dr > 0.2405
        + 0.004942027 * max(0.0, 82.66587 - Q.mass_top30) / 15.96638   # +0.5%  mass_top30 < 82.67
        - 0.004928269 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # -0.5%  tau1 < 0.06311
        + 0.00474731 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # +0.5%  sum_pt_top40 < 1025
        - 0.004566117 * max(0.0, 0.001592178 - Q.girth2_top3) / 0.000513961   # -0.5%  girth2_top3 < 0.001592
        - 0.004283961 * max(0.0, 0.076787 - Q.girth) * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.0007421723   # -0.4%  girth < 0.07679 and z_dr_0p2_0p4 < 0.0518
        - 0.003591506 * max(0.0, Q.psi_0p2 - 0.9804031) / 0.007668897   # -0.4%  psi_0p2 > 0.9804
        + 0.003523072 * max(0.0, 0.02210818 - Q.e2) / 0.00342102   # +0.4%  e2 < 0.02211
        + 0.00331101 * max(0.0, Q.psi_0p2 - 0.9804031) * max(0.0, Q.pt2_over_pt0 - 0.1852611) / 0.002296628   # +0.3%  psi_0p2 > 0.9804 and pt2_over_pt0 > 0.1853
        - 0.003298924 * max(0.0, 0.01256572 - Q.e2) / 0.0008030352   # -0.3%  e2 < 0.01257
        - 0.002653139 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1231747 - Q.z_2nd) / 6.106735e-05   # -0.3%  psi_0p3 > 0.9924 and z_2nd < 0.1232
        - 0.002408989 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -18.47737) / 3.570094   # -0.2%  tau21_b2 < 0.3425 and orientation_deg > -18.48
        + 0.002408149 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +0.2%  mass < 74.25
        - 0.001986539 * max(0.0, 906.6023 - Q.sum_pt_top40) / 6.984569   # -0.2%  sum_pt_top40 < 906.6
        - 0.001905962 * max(0.0, 976.277 - Q.sum_pt_top50) / 12.87298   # -0.2%  sum_pt_top50 < 976.3
        + 0.001602878 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +0.2%  mass < 78.26
        - 0.001541337 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.n_dr_0p05_0p1 - 14.0) / 15.57557   # -0.2%  mass < 91.19 and n_dr_0p05_0p1 > 14
        - 0.00143235 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 32.0 - Q.n_real_top40) / 0.1595466   # -0.1%  tau21_b2 < 0.3425 and n_real_top40 < 32
        - 0.001040847 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, Q.sd_rg - 0.2280025) / 1.846945e-05   # -0.1%  psi_0p3 > 0.9924 and sd_rg > 0.228
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 11.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.43914 * (0.04026641
        - 0.07270138 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # -7.3%  sd_rg < 0.3017
        + 0.07237065 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +7.2%  log_sum_pt < 7.017
        + 0.05608791 * max(0.0, 0.08589404 - Q.girth) / 0.02745918   # +5.6%  girth < 0.08589
        + 0.05215233 * max(0.0, 79.18312 - Q.sd_mass) / 30.46368   # +5.2%  sd_mass < 79.18
        + 0.05065934 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +5.1%  girth2_top5 < 0.00833
        - 0.04441972 * max(0.0, Q.psi_0p2 - 0.8706159) / 0.08984765   # -4.4%  psi_0p2 > 0.8706
        + 0.04020564 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # +4.0%  girth2_top10 < 0.007679
        - 0.03887759 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # -3.9%  e2 > 0.0303
        - 0.03791131 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # -3.8%  lam1 < 0.01174
        + 0.03737639 * max(0.0, 0.2042612 - Q.sd_rg) / 0.08448097   # +3.7%  sd_rg < 0.2043
        - 0.03136055 * max(0.0, 0.2941033 - Q.LHA) / 0.05718399   # -3.1%  LHA < 0.2941
        + 0.03135473 * max(0.0, 89.17293 - Q.mass_top30) / 20.1761   # +3.1%  mass_top30 < 89.17
        - 0.03077933 * max(0.0, 1018.698 - Q.sum_pt_top40) / 34.89083   # -3.1%  sum_pt_top40 < 1019
        - 0.03043627 * max(0.0, 0.008956554 - Q.girth2_top10) / 0.004473389   # -3.0%  girth2_top10 < 0.008957
        - 0.02643038 * max(0.0, 1017.778 - Q.sum_pt_top20) / 107.0228   # -2.6%  sum_pt_top20 < 1018
        - 0.02580085 * max(0.0, 45.595 - Q.sd_mass) / 13.46843   # -2.6%  sd_mass < 45.59
        - 0.0250105 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -2.5%  psi_0p3 > 0.9897
        + 0.02001141 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +2.0%  lam2 < 0.001776
        - 0.01816651 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -1.8%  tau1 < 0.07057
        + 0.01694928 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +1.7%  z_dr_0p1_0p2 < 0.1203
        - 0.01597688 * max(0.0, 0.002270363 - Q.girth2_top5) / 0.0007634099   # -1.6%  girth2_top5 < 0.00227
        - 0.0144232 * max(0.0, 0.006399858 - Q.e2_sq) / 0.001405963   # -1.4%  e2_sq < 0.0064
        - 0.01432893 * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 3.233647   # -1.4%  n_dr_0p2_0p4 > 9
        - 0.01293551 * max(0.0, 0.4226723 - Q.N2) / 0.1194948   # -1.3%  N2 < 0.4227
        - 0.01203374 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # -1.2%  mass_top50 < 71.8
        - 0.0117425 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -1.2%  log_sum_pt < 6.903
        + 0.01171081 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) / 3.82104   # +1.2%  n_dr_0p1_0p2 < 13
        - 0.01155589 * max(0.0, Q.sj2_dr - 0.1666442) / 0.04983259   # -1.2%  sj2_dr > 0.1666
        + 0.01142566 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +1.1%  sj2_dr > 0.2232
        + 0.01117115 * max(0.0, 933.1875 - Q.sum_pt_top30) / 20.26547   # +1.1%  sum_pt_top30 < 933.2
        + 0.01077267 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 1260.541 - Q.sum_pt) / 87.05552   # +1.1%  z_dr_0_0p05 < 0.8459 and sum_pt < 1261
        + 0.009878715 * max(0.0, Q.mass_top30 - 80.24626) / 13.49216   # +1.0%  mass_top30 > 80.25
        + 0.009205859 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.9%  lam1 < 0.004673
        - 0.008598569 * max(0.0, Q.e3 - 0.0001841806) / 4.035159e-05   # -0.9%  e3 > 0.0001842
        + 0.008130508 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +0.8%  sum_pt < 986.1
        - 0.007760319 * max(0.0, 935.8189 - Q.sum_pt_top40) / 10.51928   # -0.8%  sum_pt_top40 < 935.8
        - 0.007371776 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677) / 0.04548726   # -0.7%  girth2_top5 < 0.00833 and sj3_mass1 > 5.113
        - 0.007331598 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583) / 0.0009330603   # -0.7%  girth2_top5 < 0.00833 and sj3_pairmin_over_m > 0.09541
        + 0.006903342 * max(0.0, 1.976207 - Q.D2) / 0.367701   # +0.7%  D2 < 1.976
        - 0.005570032 * max(0.0, 0.2982 - Q.max_dr) / 0.01350087   # -0.6%  max_dr < 0.2982
        + 0.004888959 * max(0.0, Q.sd_rg - 0.1690338) / 0.03150941   # +0.5%  sd_rg > 0.169
        - 0.004555149 * max(0.0, Q.lam1 - 0.01649354) / 0.001003295   # -0.5%  lam1 > 0.01649
        + 0.004534204 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2) / 0.0004272674   # +0.5%  sj2_dr > 0.2232 and C2_b2 < 0.04009
        - 0.003546657 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0) / 0.2039042   # -0.4%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p2_0p4 > 5
        - 0.003435901 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -0.3%  sum_pt < 1002
        - 0.003296285 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -0.3%  psi_0p1 > 0.8976
        + 0.003239624 * max(0.0, Q.e2 - 0.03029714) * max(0.0, 0.3166869 - Q.sj3_z2) / 0.0002537782   # +0.3%  e2 > 0.0303 and sj3_z2 < 0.3167
        + 0.002442629 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt) / 2.722316   # +0.2%  z_dr_0_0p05 < 0.8459 and sum_pt < 949.9
        + 0.00217086 * max(0.0, Q.n_dr_0p2_0p4 - 26.0) / 0.2669462   # +0.2%  n_dr_0p2_0p4 > 26
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6222871848739496, 2.8446268907563024, 0.225868487394958, 0.41432752100840337, 0.7344672268907563, 1.0800987394957984, 0.5080813025210084, 0.38602394957983194, 1.1820932773109243, 1.0054289915966386, 1.1483496848739496, 0.7548853991596639, 0.49358781512605043, 1.5373880777310924, 0.2709059348739496, 0.37775294117647057]
T = [3.837207177324055, 2.4134744567358193, 4.144235724133404, 4.181164371060924, 3.9301292738970592]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +46%, n4 -17%, n9 +14%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.4633296 * h[1] / H_AVG[1]
            - 0.1674809 * h[4] / H_AVG[4]
            + 0.1351046 * h[9] / H_AVG[9]
            - 0.08796264 * h[5] / H_AVG[5]
            - 0.08098224 * h[3] / H_AVG[3]
            + 0.03014814 * h[12] / H_AVG[12]
            + 0.02068893 * h[6] / H_AVG[6]
            + 0.009626901 * h[8] / H_AVG[8]
            - 0.004676048 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -32%, n9 +23%, n1 -22%, n11 -8%, n12 +7%, n6 +5% ...
            - 0.3233394 * h[4] / H_AVG[4]
            + 0.2343318 * h[9] / H_AVG[9]
            - 0.2209957 * h[1] / H_AVG[1]
            - 0.07819488 * h[11] / H_AVG[11]
            + 0.07030147 * h[12] / H_AVG[12]
            + 0.04605095 * h[6] / H_AVG[6]
            + 0.01169831 * h[2] / H_AVG[2]
            + 0.007652953 * h[8] / H_AVG[8]
            - 0.007434495 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +15%, n0 +11%, n11 +11%, n14 -9%, n4 +6% ...
            - 0.2495832 * h[8] / H_AVG[8]
            + 0.1506748 * h[5] / H_AVG[5]
            + 0.112618 * h[0] / H_AVG[0]
            + 0.1081534 * h[11] / H_AVG[11]
            - 0.08988284 * h[14] / H_AVG[14]
            + 0.06092151 * h[4] / H_AVG[4]
            - 0.058217 * h[7] / H_AVG[7]
            - 0.05307072 * h[9] / H_AVG[9]
            - 0.04838529 * h[12] / H_AVG[12]
            + 0.04373986 * h[3] / H_AVG[3]
            - 0.01709089 * h[15] / H_AVG[15]
            + 0.00766247 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -27%, n0 -20%, n5 +18%, n6 -12%, n7 +8%, n15 +5% ...
            - 0.2650488 * h[8] / H_AVG[8]
            - 0.2046427 * h[0] / H_AVG[0]
            + 0.1775983 * h[5] / H_AVG[5]
            - 0.1177193 * h[6] / H_AVG[6]
            + 0.08366909 * h[7] / H_AVG[7]
            + 0.05081982 * h[15] / H_AVG[15]
            + 0.04335354 * h[3] / H_AVG[3]
            + 0.03689073 * h[12] / H_AVG[12]
            - 0.02025768 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -35%, n10 +29%, n5 -13%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3545069 * h[13] / H_AVG[13]
            + 0.2876258 * h[10] / H_AVG[10]
            - 0.1288243 * h[5] / H_AVG[5]
            + 0.06344519 * h[8] / H_AVG[8]
            - 0.04709652 * h[12] / H_AVG[12]
            - 0.03604394 * h[15] / H_AVG[15]
            + 0.03504022 * h[4] / H_AVG[4]
            - 0.02762485 * h[7] / H_AVG[7]
            + 0.0197922 * h[0] / H_AVG[0]
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
