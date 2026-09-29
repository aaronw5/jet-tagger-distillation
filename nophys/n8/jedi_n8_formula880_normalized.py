"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.2%   (on for 88% of jets)
  neuron  9:  11.9%   (on for 70% of jets)
  neuron  7:  10.2%   (on for 61% of jets)
  neuron  3:   8.7%   (on for 29% of jets)
  neuron 10:   8.2%   (on for 80% of jets)
  neuron  2:   7.3%   (on for 87% of jets)
  neuron  5:   7.1%   (on for 63% of jets)
  neuron  6:   7.0%   (on for 48% of jets)
  neuron 11:   6.5%   (on for 71% of jets)
  neuron 14:   4.5%   (on for 31% of jets)
  neuron  0:   4.3%   (on for 43% of jets)
  neuron  1:   3.2%   (on for 56% of jets)
  neuron  4:   3.0%   (on for 43% of jets)
  neuron 15:   2.6%   (on for 29% of jets)
  neuron  8:   1.0%   (on for 34% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.9% (the network: 65.8%); same class as the network for 88.6% of jets.

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
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_2          mass of particles 0 and 2 [GeV]
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
  Q.m01                    mass of particles 0 and 1 [GeV]
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
  Q.z_1st                  largest pT share
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_2               |Δη| of particle 2
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_6               |Δη| of particle 6
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_5               |Δφ| of particle 5
  Q.absphi_6               |Δφ| of particle 6
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.phi_0                  Δφ of particle 0
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
  Q.n_pt_above_10          number of particles with pT > 10 GeV
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
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
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
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_2=pair_mass(0, 2),
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
        m01=pair_mass(0, 1),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        pt_0=pt[0],
        pt_1=pt[1],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_4=z[4],
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
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_0=abs(eta[0]),
        abseta_2=abs(eta[2]),
        abseta_4=abs(eta[4]),
        abseta_6=abs(eta[6]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_5=abs(phi[5]),
        absphi_6=abs(phi[6]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        eta_0=eta[0],
        eta_1=eta[1],
        phi_0=phi[0],
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
        n_pt_above_10=sum(1 for x in pt if x > 10),
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
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 25.33;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.33066 * (-0.06896827
        + 0.1112772 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +11.1%  girth2 < 0.01324
        + 0.08822384 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +8.8%  width < 0.008678
        + 0.06065368 * max(0.0, Q.sj3_dr_max - 0.1070199) / 0.08518534   # +6.1%  sj3_dr_max > 0.107
        - 0.05412358 * max(0.0, 0.07608178 - Q.girth) / 0.02796784   # -5.4%  girth < 0.07608
        - 0.04685551 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -4.7%  girth < 0.08724
        - 0.04281304 * max(0.0, 0.004372139 - Q.width) / 0.00156571   # -4.3%  width < 0.004372
        - 0.04154728 * max(0.0, 64.61873 - Q.mass) / 27.57279   # -4.2%  mass < 64.62
        - 0.04088378 * max(0.0, 21.78408 - Q.mass) / 4.431023   # -4.1%  mass < 21.78
        - 0.03445796 * max(0.0, 0.007929074 - Q.girth2_top3) / 0.004560262   # -3.4%  girth2_top3 < 0.007929
        - 0.02850073 * max(0.0, 0.08475161 - Q.mass_over_sum_pt) / 0.03304118   # -2.9%  mass_over_sum_pt < 0.08475
        - 0.02446037 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # -2.4%  sj3_dr_max > 0.179
        - 0.02349946 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -2.3%  lam1 < 0.005433
        + 0.02255931 * max(0.0, 0.01882765 - Q.girth2) / 0.01314296   # +2.3%  girth2 < 0.01883
        + 0.02187133 * max(0.0, 0.006756161 - Q.girth2_top3) / 0.003650633   # +2.2%  girth2_top3 < 0.006756
        + 0.02175823 * max(0.0, 0.03556091 - Q.e2) / 0.01371972   # +2.2%  e2 < 0.03556
        + 0.01973084 * max(0.0, 0.003904593 - Q.mass_over_sum_pt_sq) / 0.001485439   # +2.0%  mass_over_sum_pt_sq < 0.003905
        - 0.01947386 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -1.9%  centroid_offset > 0.00679
        + 0.01909482 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # +1.9%  mass_over_sum_pt < 0.1309
        + 0.01891789 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +1.9%  planar_flow < 0.1484
        + 0.01805891 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +1.8%  sj3_dr_max < 0.3012
        - 0.01713806 * max(0.0, 0.001563465 - Q.C2_b2) / 0.0008800039   # -1.7%  C2_b2 < 0.001563
        + 0.01466262 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +1.5%  log_sum_pt > 6.378
        + 0.01445139 * max(0.0, Q.n_dr_0_0p05 - 4.0) / 1.61275   # +1.4%  n_dr_0_0p05 > 4
        - 0.01212561 * max(0.0, 56.92035 - Q.mass) / 21.78475   # -1.2%  mass < 56.92
        + 0.01143173 * max(0.0, 0.004372139 - Q.width) * max(0.0, 0.03532852 - Q.C3) / 3.6978e-05   # +1.1%  width < 0.004372 and C3 < 0.03533
        - 0.0113378 * max(0.0, 7.300726e-05 - Q.lam2) / 2.767163e-05   # -1.1%  lam2 < 7.301e-05
        - 0.01126281 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -1.1%  centroid_offset > 0.0499
        - 0.009944086 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 2.0 - Q.n_dr_0p05_0p1) / 0.1809599   # -1.0%  sj3_dr_max < 0.3012 and n_dr_0p05_0p1 < 2
        - 0.009510505 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # -1.0%  girth2_top5 < 0.00833
        - 0.008390582 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -0.8%  sj3_dr_max > 0.2337
        - 0.008200469 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05557716 - Q.z_7) / 0.001734349   # -0.8%  sj3_dr_max < 0.3012 and z_7 < 0.05558
        + 0.007885047 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.8%  sum_pt_top5 > 687.4
        - 0.007651034 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, Q.z_7 - 0.01685855) / 0.0006290825   # -0.8%  log_sum_pt > 6.67 and z_7 > 0.01686
        - 0.006984085 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -0.7%  log_sum_pt > 6.67
        + 0.006953554 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 0.07232166 - Q.dr_4) / 0.001949106   # +0.7%  log_sum_pt > 6.67 and dr_4 < 0.07232
        - 0.006928253 * max(0.0, 0.0002758826 - Q.lam1) / 3.586469e-05   # -0.7%  lam1 < 0.0002759
        - 0.006369039 * max(0.0, Q.sum_pt - 901.5938) / 12.40362   # -0.6%  sum_pt > 901.6
        + 0.005537294 * max(0.0, 0.0245477 - Q.e2) / 0.007455821   # +0.6%  e2 < 0.02455
        + 0.00517976 * max(0.0, Q.centroid_offset - 0.006789738) * max(0.0, 223.375 - Q.pt_1) / 1.309723   # +0.5%  centroid_offset > 0.00679 and pt_1 < 223.4
        + 0.005021055 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2) / 2.36013e-06   # +0.5%  lam2 < 7.301e-05 and D2_b2 < 0.267
        - 0.004859567 * max(0.0, 0.04889979 - Q.sj3_dr_max) / 0.006377687   # -0.5%  sj3_dr_max < 0.0489
        - 0.004645915 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.06336451 - Q.dr_4) / 1.344914   # -0.5%  sum_pt_top5 > 687.4 and dr_4 < 0.06336
        + 0.004381069 * max(0.0, 0.01587232 - Q.zdr_1) / 0.007691685   # +0.4%  zdr_1 < 0.01587
        + 0.004329891 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 28.78966 - Q.m01) / 1.085363   # +0.4%  log_sum_pt > 6.67 and m01 < 28.79
        - 0.004074839 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 358.375 - Q.sum_pt_top2) / 2.426548   # -0.4%  planar_flow < 0.1484 and sum_pt_top2 < 358.4
        + 0.00401217 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.08525808 - Q.dr_2) / 1.41829e-06   # +0.4%  lam2 < 7.301e-05 and dr_2 < 0.08526
        + 0.003810392 * max(0.0, Q.centroid_offset - 0.006789738) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.02705531   # +0.4%  centroid_offset > 0.00679 and n_pt_above_50 < 7
        + 0.003344492 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 1.002471 - Q.D2) / 0.001008123   # +0.3%  girth2 < 0.01324 and D2 < 1.002
        - 0.002809018 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # -0.3%  lam1 < 0.006507 and D2 < 0.8757
        - 0.002482497 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, -0.003096655 - Q.mean_phi) / 2.294381e-05   # -0.2%  girth2 < 0.01324 and mean_phi < -0.003097
        - 0.002203777 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 0.06473447 - Q.z_7) / 3.579716e-05   # -0.2%  centroid_offset > 0.02077 and z_7 < 0.06473
        + 0.002005976 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.4160096 - Q.pt_dispersion) / 0.0006933006   # +0.2%  planar_flow < 0.1484 and pt_dispersion < 0.416
        - 0.00180211 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.centroid_offset - 0.01837778) / 2.056482e-05   # -0.2%  girth2 < 0.01324 and centroid_offset > 0.01838
        - 0.001742739 * max(0.0, 0.007929074 - Q.girth2_top3) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0005550791   # -0.2%  girth2_top3 < 0.007929 and D2_b2 < 0.5327
        - 0.001686086 * max(0.0, Q.centroid_offset - 0.02076709) / 0.004751537   # -0.2%  centroid_offset > 0.02077
        - 0.001411494 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.mean_eta - 6.288824e-05) / 0.000715688   # -0.1%  log_sum_pt > 6.378 and mean_eta > 6.289e-05
        - 0.001245161 * max(0.0, 56.92035 - Q.mass) * max(0.0, Q.dr_max_012 - 0.06112084) / 0.1887976   # -0.1%  mass < 56.92 and dr_max_012 > 0.06112
        - 0.001140777 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05744392 - Q.D2_b2) / 0.0005632394   # -0.1%  sj3_dr_max < 0.3012 and D2_b2 < 0.05744
        + 0.0008490695 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.pt_7 - 33.21875) / 68.87156   # +0.1%  sum_pt > 901.6 and pt_7 > 33.22
        + 0.0008275753 * max(0.0, 0.01882765 - Q.girth2) * max(0.0, Q.mass_top2 - 28.78966) / 0.006400313   # +0.1%  girth2 < 0.01883 and mass_top2 > 28.79
        - 0.0006089941 * max(0.0, 6.080494 - Q.log_sum_pt) / 0.006097657   # -0.1%  log_sum_pt < 6.08
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 19.06;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.05766 * (0.004050052
        - 0.08854393 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -8.9%  girth2 < 0.008678
        - 0.08749032 * max(0.0, 0.00609665 - Q.width) / 0.002544039   # -8.7%  width < 0.006097
        - 0.08152053 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -8.2%  sum_pt_top5 > 531.2
        + 0.056997 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +5.7%  log_sum_pt > 6.503
        - 0.05135252 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # -5.1%  girth < 0.1019
        + 0.05081444 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +5.1%  log_sum_pt > 6.378
        + 0.04519568 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +4.5%  lam1 < 0.005954
        + 0.04446024 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # +4.4%  z_7 > 0.02321
        + 0.04115696 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # +4.1%  girth < 0.0717
        + 0.03131296 * max(0.0, 49.6681 - Q.mass) / 17.06893   # +3.1%  mass < 49.67
        + 0.02723186 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.06297984 - Q.tau2) / 0.0007776805   # +2.7%  z_7 < 0.06165 and tau2 < 0.06298
        - 0.02606461 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -2.6%  z_7 < 0.06165
        + 0.0255675 * max(0.0, Q.sum_pt_top5 - 367.5938) / 230.8403   # +2.6%  sum_pt_top5 > 367.6
        - 0.02483705 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, 0.008654951 - Q.zdr_6) / 0.000799435   # -2.5%  log_sum_pt > 6.503 and zdr_6 < 0.008655
        + 0.02085984 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # +2.1%  e2_sq < 0.008169
        - 0.02009135 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1) / 0.1196218   # -2.0%  log_sum_pt > 6.378 and z_dr_0p05_0p1 < 0.7509
        + 0.01940854 * max(0.0, Q.sum_pt_top5 - 531.1875) * max(0.0, 0.01392641 - Q.zdr_6) / 1.245695   # +1.9%  sum_pt_top5 > 531.2 and zdr_6 < 0.01393
        - 0.01824432 * max(0.0, Q.z_7 - 0.04624032) / 0.01204993   # -1.8%  z_7 > 0.04624
        + 0.01803537 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +1.8%  e3 < 0.0005117
        + 0.01765493 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) / 0.002578466   # +1.8%  mass_over_sum_pt_sq < 0.005833
        + 0.01585573 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +1.6%  pt_7 > 34.53
        + 0.01497146 * max(0.0, Q.log_sum_pt - 6.605974) / 0.07073019   # +1.5%  log_sum_pt > 6.606
        - 0.01217171 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 1.627072e-05 - Q.e3) / 1.912338e-06   # -1.2%  log_sum_pt > 6.378 and e3 < 1.627e-05
        + 0.01141705 * max(0.0, 1.627072e-05 - Q.e3) / 6.577181e-06   # +1.1%  e3 < 1.627e-05
        - 0.0106866 * max(0.0, 0.0008722282 - Q.lam1) / 0.0001882079   # -1.1%  lam1 < 0.0008722
        - 0.01053821 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # -1.1%  girth2 < 0.00502
        + 0.01042941 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +1.0%  tau1 < 0.07284
        - 0.008342411 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.00323438   # -0.8%  lam1 < 0.008376 and n_pt_above_50 > 5
        + 0.007681635 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +0.8%  lam1 < 0.008376
        + 0.007511149 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.01403324 - Q.girth2_top2) / 0.0001680113   # +0.8%  z_7 < 0.06165 and girth2_top2 < 0.01403
        + 0.007502131 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +0.8%  sj3_dr_max > 0.169
        - 0.007394045 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 1.679198 - Q.D2) / 0.005760154   # -0.7%  z_7 < 0.06165 and D2 < 1.679
        + 0.006958067 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.008921136 - Q.mean_phi2) / 0.0001024498   # +0.7%  z_7 < 0.06165 and mean_phi2 < 0.008921
        - 0.006493504 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -0.6%  centroid_offset < 0.01838
        + 0.005313179 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.1057739 - Q.abseta_0) / 0.001206885   # +0.5%  z_7 < 0.06165 and abseta_0 < 0.1058
        - 0.005258684 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1484197 - Q.planar_flow) / 0.0001369594   # -0.5%  lam1 < 0.008376 and planar_flow < 0.1484
        - 0.00502279 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 2.955458e-05 - Q.e3) / 7.137805e-05   # -0.5%  pt_7 > 34.53 and e3 < 2.955e-05
        - 0.004931359 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -0.5%  zdr_0 < 0.02118
        + 0.004454025 * max(0.0, 0.008678045 - Q.girth2) * max(0.0, Q.centroid_offset - 0.02076709) / 7.418362e-06   # +0.4%  girth2 < 0.008678 and centroid_offset > 0.02077
        - 0.004302538 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 52.90625 - Q.pt_6) / 13.55761   # -0.4%  pt_7 > 34.53 and pt_6 < 52.91
        + 0.003936604 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 1.432482 - Q.D2) / 0.01970563   # +0.4%  log_sum_pt > 6.606 and D2 < 1.432
        + 0.003884604 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # +0.4%  pt_7 > 34.53 and tau2 < 0.01713
        - 0.003544536 * max(0.0, 25.57812 - Q.pt_7) / 1.327389   # -0.4%  pt_7 < 25.58
        - 0.003406283 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.04875823) / 0.002321449   # -0.3%  z_7 < 0.06165 and z_dr_0p05_0p1 > 0.04876
        + 0.003072888 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.centroid_offset - 0.009480685) / 0.0009010889   # +0.3%  log_sum_pt > 6.378 and centroid_offset > 0.009481
        + 0.002889605 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 0.09925997 - Q.z_4) / 0.00251516   # +0.3%  log_sum_pt > 6.606 and z_4 < 0.09926
        - 0.002485888 * max(0.0, Q.sj3_dr_min - 0.03628191) / 0.02376249   # -0.2%  sj3_dr_min > 0.03628
        - 0.002485315 * max(0.0, 0.1019409 - Q.girth) * max(0.0, Q.pair_mass_0_6 - 4.037975) / 0.1185817   # -0.2%  girth < 0.1019 and pair_mass_0_6 > 4.038
        + 0.002081263 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.sj2_dr - 0.1294903) / 0.1877842   # +0.2%  pt_7 > 34.53 and sj2_dr > 0.1295
        - 0.001987824 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236) / 1.769648e-05   # -0.2%  mass_over_sum_pt_sq < 0.005833 and centroid_offset > 0.008092
        - 0.001668672 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.2%  pt_7 > 53.44
        - 0.001538719 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.5700307   # -0.2%  sj3_dr_max > 0.169 and sj3_pair_mass_min > 5.744
        - 0.001252534 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.centroid_offset - 0.02076709) / 2.871254e-05   # -0.1%  z_7 < 0.06165 and centroid_offset > 0.02077
        + 0.001078431 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 3.175597   # +0.1%  pt_7 > 34.53 and n_dr_0p1_0p2 > 1
        - 0.0006111768 * max(0.0, Q.pt_6 - 62.25) / 0.3835962   # -0.1%  pt_6 > 62.25
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 20.21;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.20827 * (0.1946283
        - 0.09282644 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -9.3%  pt_7 < 53.44
        - 0.09058476 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # -9.1%  z_7 > 0.02321
        - 0.06077774 * max(0.0, 716.8828 - Q.sum_pt_top5) / 152.5573   # -6.1%  sum_pt_top5 < 716.9
        - 0.05350754 * max(0.0, Q.LHA - 0.111565) / 0.1352185   # -5.4%  LHA > 0.1116
        + 0.04851134 * max(0.0, 788.4484 - Q.sum_pt) / 112.1576   # +4.9%  sum_pt < 788.4
        + 0.04429492 * max(0.0, Q.pt_7 - 20.125) / 15.07003   # +4.4%  pt_7 > 20.12
        - 0.04006801 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # -4.0%  girth < 0.1484
        + 0.03968832 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # +4.0%  log_sum_pt < 6.606
        + 0.03781595 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +3.8%  lam1 < 0.012
        + 0.03174103 * max(0.0, Q.mass - 15.45403) / 27.25251   # +3.2%  mass > 15.45
        - 0.03012702 * max(0.0, Q.z_6 - 0.02886576) / 0.03195851   # -3.0%  z_6 > 0.02887
        + 0.02919523 * max(0.0, 43.5 - Q.pt_7) / 10.28861   # +2.9%  pt_7 < 43.5
        + 0.02883326 * max(0.0, Q.pt_6 - 27.57812) / 13.69978   # +2.9%  pt_6 > 27.58
        + 0.02640923 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +2.6%  z_7 < 0.07149
        - 0.02626132 * max(0.0, Q.mass - 36.22941) / 14.20528   # -2.6%  mass > 36.23
        + 0.02567987 * max(0.0, 840.0195 - Q.sum_pt) / 148.4153   # +2.6%  sum_pt < 840
        - 0.02302038 * max(0.0, 0.0003193707 - Q.girth2) / 4.035777e-05   # -2.3%  girth2 < 0.0003194
        + 0.02153104 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +2.2%  girth2 < 0.00502
        + 0.02107572 * max(0.0, 0.0003193707 - Q.girth2) * max(0.0, 36.76827 - Q.mass_top2) / 0.001427268   # +2.1%  girth2 < 0.0003194 and mass_top2 < 36.77
        - 0.01808398 * max(0.0, 15.95929 - Q.sj3_pair_mass_min) / 10.45654   # -1.8%  sj3_pair_mass_min < 15.96
        - 0.01687199 * max(0.0, 63.68899 - Q.sj3_pair_mass_max) * max(0.0, 0.06810151 - Q.z_7) / 0.6284113   # -1.7%  sj3_pair_mass_max < 63.69 and z_7 < 0.0681
        + 0.01482499 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +1.5%  lam1 < 0.005954
        + 0.0146316 * max(0.0, 63.68899 - Q.sj3_pair_mass_max) / 31.57762   # +1.5%  sj3_pair_mass_max < 63.69
        - 0.01196537 * max(0.0, Q.sum_pt_top5 - 605.1875) / 67.49708   # -1.2%  sum_pt_top5 > 605.2
        + 0.01160038 * max(0.0, 840.0195 - Q.sum_pt) * max(0.0, 0.02164863 - Q.dr_5) / 0.1765901   # +1.2%  sum_pt < 840 and dr_5 < 0.02165
        - 0.01115082 * max(0.0, 43.5 - Q.pt_7) * max(0.0, 3.916009 - Q.D3) / 26.77823   # -1.1%  pt_7 < 43.5 and D3 < 3.916
        + 0.008903486 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # +0.9%  log_sum_pt > 6.843
        - 0.008402975 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, 0.02164863 - Q.dr_5) / 0.1197721   # -0.8%  sum_pt < 788.4 and dr_5 < 0.02165
        + 0.007045993 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # +0.7%  zdr_0 < 0.02118
        - 0.006237177 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # -0.6%  lam2 < 0.0001948
        - 0.00615357 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.6%  sum_pt_top5 > 840
        + 0.005449764 * max(0.0, 559.6875 - Q.sum_pt) * max(0.0, 4.721224 - Q.D2_b2) / 63.42935   # +0.5%  sum_pt < 559.7 and D2_b2 < 4.721
        - 0.005350176 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # -0.5%  pt_6 < 41.22
        + 0.004921197 * max(0.0, Q.z_7 - 0.04939969) / 0.01023002   # +0.5%  z_7 > 0.0494
        - 0.004677439 * max(0.0, 43.5 - Q.pt_7) * max(0.0, 3.032555 - Q.D2_b2) / 18.19911   # -0.5%  pt_7 < 43.5 and D2_b2 < 3.033
        - 0.003936261 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.6947818 - Q.planar_flow) / 0.009090069   # -0.4%  z_7 < 0.07149 and planar_flow < 0.6948
        + 0.003803166 * max(0.0, Q.z_top5 - 0.8770155) / 0.00720125   # +0.4%  z_top5 > 0.877
        - 0.00368923 * max(0.0, 0.03243272 - Q.z_7) / 0.002061024   # -0.4%  z_7 < 0.03243
        - 0.003564761 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.max_dr - 0.0931108) / 3.745922e-05   # -0.4%  lam1 < 0.005954 and max_dr > 0.09311
        - 0.003509375 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # -0.4%  girth < 0.007674
        - 0.003478224 * max(0.0, Q.pt_6 - 27.57812) * max(0.0, 0.7974684 - Q.planar_flow) / 7.391558   # -0.3%  pt_6 > 27.58 and planar_flow < 0.7975
        + 0.003368297 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.1242755 - Q.dr_7) / 0.001381084   # +0.3%  z_7 < 0.07149 and dr_7 < 0.1243
        - 0.003148392 * max(0.0, 559.6875 - Q.sum_pt) / 16.18216   # -0.3%  sum_pt < 559.7
        + 0.003104516 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # +0.3%  log_sum_pt < 6.464
        - 0.003046785 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.centroid_offset - 0.01627885) / 2.281746e-05   # -0.3%  lam1 < 0.012 and centroid_offset > 0.01628
        + 0.003031308 * max(0.0, 0.8233866 - Q.z_top5) / 0.02979882   # +0.3%  z_top5 < 0.8234
        + 0.00283185 * max(0.0, 15.95929 - Q.sj3_pair_mass_min) * max(0.0, Q.D2 - 0.6203774) / 10.04714   # +0.3%  sj3_pair_mass_min < 15.96 and D2 > 0.6204
        - 0.002701253 * max(0.0, 63.68899 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.01096064) / 0.2223898   # -0.3%  sj3_pair_mass_max < 63.69 and centroid_offset > 0.01096
        + 0.002550965 * max(0.0, Q.sum_pt_top5 - 839.9547) * max(0.0, 2.326647 - Q.D2_b2) / 5.890686   # +0.3%  sum_pt_top5 > 840 and D2_b2 < 2.327
        + 0.002494242 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 0.0003061234 - Q.lam2) / 1.525985e-05   # +0.2%  log_sum_pt < 6.606 and lam2 < 0.0003061
        + 0.002415246 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.1286621 - Q.abseta_2) / 0.4240622   # +0.2%  pt_7 > 34.53 and abseta_2 < 0.1287
        - 0.002303812 * max(0.0, 0.02164863 - Q.dr_5) / 0.002569084   # -0.2%  dr_5 < 0.02165
        - 0.002221221 * max(0.0, Q.z_7 - 0.04939969) * max(0.0, 0.0623951 - Q.sj3_dr_min) / 0.0002978985   # -0.2%  z_7 > 0.0494 and sj3_dr_min < 0.0624
        + 0.00219634 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, 0.03623337 - Q.max_dr) / 0.1060521   # +0.2%  sum_pt < 788.4 and max_dr < 0.03623
        - 0.002133116 * max(0.0, 33.02656 - Q.pt_5) / 1.052976   # -0.2%  pt_5 < 33.03
        - 0.002020671 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 3.032555 - Q.D2_b2) / 0.005136309   # -0.2%  log_sum_pt > 6.896 and D2_b2 < 3.033
        - 0.001559933 * max(0.0, Q.z_6 - 0.02886576) * max(0.0, 0.02164863 - Q.dr_5) / 6.468094e-05   # -0.2%  z_6 > 0.02887 and dr_5 < 0.02165
        - 0.001250025 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 0.03623337 - Q.max_dr) / 9.868306e-05   # -0.1%  log_sum_pt < 6.606 and max_dr < 0.03623
        - 0.001177565 * max(0.0, Q.pt_6 - 27.57812) * max(0.0, Q.centroid_offset - 0.02076709) / 0.053518   # -0.1%  pt_6 > 27.58 and centroid_offset > 0.02077
        - 0.001130275 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.04416271 - Q.dr_5) / 0.0001724025   # -0.1%  log_sum_pt > 6.843 and dr_5 < 0.04416
        + 0.001082581 * max(0.0, 0.007673833 - Q.girth) * max(0.0, 71.6875 - Q.pt_4) / 0.003882715   # +0.1%  girth < 0.007674 and pt_4 < 71.69
        + 0.001074654 * max(0.0, 0.03672711 - Q.z_5) / 0.001007435   # +0.1%  z_5 < 0.03673
        + 0.0009057972 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.1%  log_sum_pt > 6.896
        + 0.0008146088 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.002460108 - Q.zdr_5) / 5.982837e-06   # +0.1%  log_sum_pt > 6.896 and zdr_5 < 0.00246
        - 0.0006570101 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.pt_6 - 41.21875) / 0.06090836   # -0.1%  log_sum_pt > 6.843 and pt_6 > 41.22
        + 0.0005784796 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.04416271 - Q.dr_5) / 9.036365e-05   # +0.1%  log_sum_pt > 6.896 and dr_5 < 0.04416
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 14.61;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.60648 * (-0.1288602
        - 0.2046657 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # -20.5%  girth2 > 0.008678
        + 0.1364338 * max(0.0, Q.width - 0.007520088) / 0.002470136   # +13.6%  width > 0.00752
        + 0.1228298 * max(0.0, Q.girth - 0.04081947) / 0.0270611   # +12.3%  girth > 0.04082
        + 0.09822567 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +9.8%  lam1 > 0.008376
        - 0.04300396 * max(0.0, 0.004372139 - Q.girth2) / 0.00156571   # -4.3%  girth2 < 0.004372
        - 0.04167122 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # -4.2%  tau1 > 0.05357
        - 0.04031037 * max(0.0, Q.width - 0.01323868) / 0.001442684   # -4.0%  width > 0.01324
        + 0.03605586 * max(0.0, Q.sj2_dr - 0.1872617) / 0.02732165   # +3.6%  sj2_dr > 0.1873
        + 0.03337979 * max(0.0, Q.mass_over_sum_pt - 0.0681391) / 0.01539157   # +3.3%  mass_over_sum_pt > 0.06814
        - 0.02195828 * max(0.0, Q.sj2_dr - 0.1492731) / 0.04449993   # -2.2%  sj2_dr > 0.1493
        + 0.02023815 * max(0.0, Q.max_dr - 0.1027585) / 0.04505724   # +2.0%  max_dr > 0.1028
        + 0.01900391 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # +1.9%  tau1 > 0.1028
        - 0.01784729 * max(0.0, Q.centroid_offset - 0.01096064) / 0.008813008   # -1.8%  centroid_offset > 0.01096
        - 0.01676576 * max(0.0, Q.girth - 0.07608178) / 0.01059033   # -1.7%  girth > 0.07608
        + 0.01538741 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +1.5%  girth2_top5 < 0.00833
        - 0.01266888 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -1.3%  lam1 > 0.012
        - 0.01192737 * max(0.0, Q.mass - 64.61873) / 3.274818   # -1.2%  mass > 64.62
        - 0.01102196 * max(0.0, Q.girth - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494) / 0.008781747   # -1.1%  girth > 0.04082 and log_sum_pt > 6.08
        + 0.01093885 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # +1.1%  girth > 0.08724
        + 0.007769577 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.8%  lam2 > 0.001131
        - 0.007699678 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # -0.8%  sj2_dr > 0.2688
        + 0.007342528 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50) / 0.02994202   # +0.7%  centroid_offset > 0.01096 and n_pt_above_50 < 8
        + 0.007119461 * max(0.0, Q.girth2 - 0.01882765) / 0.0007605369   # +0.7%  girth2 > 0.01883
        + 0.006897627 * max(0.0, 0.05226226 - Q.z_dr_0_0p05) / 0.01510942   # +0.7%  z_dr_0_0p05 < 0.05226
        - 0.005316659 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -0.5%  e2 > 0.06344
        + 0.004258055 * max(0.0, Q.sd_mass - 62.73432) * max(0.0, 0.9206502 - Q.D2_b2) / 1.800433   # +0.4%  sd_mass > 62.73 and D2_b2 < 0.9207
        + 0.003914245 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 0.07861328 - Q.abseta_0) / 0.0002969212   # +0.4%  centroid_offset > 0.01096 and abseta_0 < 0.07861
        + 0.003758393 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6) / 0.005634141   # +0.4%  lam2 > 0.001131 and pt_6 < 56.53
        - 0.003730391 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.pt_6 - 31.90625) / 0.1358885   # -0.4%  mass_over_sum_pt > 0.06814 and pt_6 > 31.91
        + 0.003625813 * max(0.0, Q.sd_mass - 62.73432) / 3.311364   # +0.4%  sd_mass > 62.73
        - 0.003289515 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr) / 0.0001196243   # -0.3%  mass_over_sum_pt > 0.06814 and sj2_dr < 0.218
        + 0.003144329 * max(0.0, -0.004664942 - Q.mean_eta) / 0.003754527   # +0.3%  mean_eta < -0.004665
        - 0.003108621 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, 50.3522 - Q.mass_top3) / 0.5259724   # -0.3%  tau1 > 0.05357 and mass_top3 < 50.35
        + 0.002860042 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.05792236 - Q.eta_1) / 0.003045389   # +0.3%  max_dr > 0.1028 and eta_1 < 0.05792
        - 0.002539575 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113) / 0.39507   # -0.3%  sj2_dr > 0.1873 and sj2_mass1 > 2.25
        + 0.00191331 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # +0.2%  lam1 > 0.01643
        + 0.001723982 * max(0.0, Q.mean_eta - 0.01772426) / 0.001375178   # +0.2%  mean_eta > 0.01772
        + 0.001375253 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, Q.z_6 - 0.05096142) / 7.479719e-06   # +0.1%  lam2 > 0.001131 and z_6 > 0.05096
        + 0.001337435 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sj2_mass1 - 16.86126) / 0.02534836   # +0.1%  e2 > 0.06344 and sj2_mass1 > 16.86
        - 0.001094515 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.05268713 - Q.dr_3) / 0.000155932   # -0.1%  sj2_dr > 0.1873 and dr_3 < 0.05269
        - 0.001033887 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, -0.05963135 - Q.eta_1) / 0.0004225811   # -0.1%  max_dr > 0.1028 and eta_1 < -0.05963
        - 0.0008131439 * max(0.0, Q.mean_eta - 0.01772426) * max(0.0, 0.02612796 - Q.mean_phi) / 3.988786e-05   # -0.1%  mean_eta > 0.01772 and mean_phi < 0.02613
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 99.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 99.28606 * (-0.03204877
        + 0.28863 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # +28.9%  e2_sq < 0.01166
        - 0.2317997 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -23.2%  mass_over_sum_pt_sq < 0.01166
        - 0.05838221 * max(0.0, 0.01882765 - Q.girth2) / 0.01314296   # -5.8%  girth2 < 0.01883
        + 0.05407298 * max(0.0, 0.01716248 - Q.e2_sq) / 0.0120354   # +5.4%  e2_sq < 0.01716
        - 0.03386688 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # -3.4%  width < 0.01324
        - 0.02698418 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -2.7%  girth2 < 0.008678
        + 0.02622794 * max(0.0, 0.1245537 - Q.girth) / 0.06848162   # +2.6%  girth < 0.1246
        - 0.02325467 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -2.3%  mass < 69.61
        + 0.01879197 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # +1.9%  C2_b2 < 0.009033
        + 0.01820735 * max(0.0, Q.e2 - 0.009668065) / 0.02044666   # +1.8%  e2 > 0.009668
        + 0.01792921 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +1.8%  N2 < 0.2233
        - 0.01304505 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -1.3%  lam2 < 0.001131
        - 0.01132362 * max(0.0, 0.05119235 - Q.C2) / 0.02746756   # -1.1%  C2 < 0.05119
        + 0.01050541 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +1.1%  mass < 76.66
        + 0.008841355 * max(0.0, 0.03981924 - Q.zdr_0) / 0.02538306   # +0.9%  zdr_0 < 0.03982
        + 0.008062639 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +0.8%  girth2_top5 < 0.00833
        - 0.007973709 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -0.8%  sj3_dr_max > 0.2337
        - 0.007899742 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -0.8%  lam2 < 0.0005373
        - 0.007612978 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -0.8%  mass_over_sum_pt > 0.09041
        + 0.007466015 * max(0.0, Q.girth2 - 0.003562611) / 0.004066872   # +0.7%  girth2 > 0.003563
        + 0.007253267 * max(0.0, 1.002471 - Q.D2) / 0.1871081   # +0.7%  D2 < 1.002
        + 0.007085774 * max(0.0, Q.max_dr - 0.1117619) / 0.04033009   # +0.7%  max_dr > 0.1118
        - 0.007081166 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266) / 0.01381411   # -0.7%  N2 < 0.2233 and eccentricity > 0.7117
        + 0.006771298 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +0.7%  sj3_dr_max > 0.169
        + 0.005932971 * max(0.0, Q.sd_mass - 38.43971) / 11.50403   # +0.6%  sd_mass > 38.44
        + 0.005748481 * max(0.0, 0.002635418 - Q.girth2) / 0.0007941346   # +0.6%  girth2 < 0.002635
        - 0.005695653 * max(0.0, Q.tau1 - 0.07283629) / 0.01802787   # -0.6%  tau1 > 0.07284
        - 0.004464624 * max(0.0, 0.2838437 - Q.tau21) / 0.07817133   # -0.4%  tau21 < 0.2838
        + 0.00416119 * max(0.0, Q.tau1 - 0.07283629) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.001837197   # +0.4%  tau1 > 0.07284 and sj3_dr_min < 0.209
        - 0.003797375 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.002796532   # -0.4%  centroid_offset < 0.01838 and z_dr_0p05_0p1 < 0.5883
        + 0.003668426 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +0.4%  e2 > 0.04447
        - 0.003651966 * max(0.0, Q.sd_rg - 0.2037854) / 0.01566501   # -0.4%  sd_rg > 0.2038
        - 0.003602741 * max(0.0, Q.girth - 0.05464922) / 0.01938482   # -0.4%  girth > 0.05465
        - 0.003593934 * max(0.0, Q.C2 - 0.01867771) / 0.01385506   # -0.4%  C2 > 0.01868
        - 0.003238569 * max(0.0, 76.6557 - Q.mass) * max(0.0, 0.875672 - Q.D2) / 2.2476   # -0.3%  mass < 76.66 and D2 < 0.8757
        - 0.002598063 * max(0.0, 1.002471 - Q.D2) * max(0.0, 53.4375 - Q.pt_7) / 3.084564   # -0.3%  D2 < 1.002 and pt_7 < 53.44
        - 0.002590745 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # -0.3%  tau1 > 0.1136
        - 0.002444999 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -0.2%  sum_pt < 739.5
        + 0.002286987 * max(0.0, 0.007639643 - Q.girth2_top2) / 0.004543329   # +0.2%  girth2_top2 < 0.00764
        - 0.002224408 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.1591797 - Q.absphi_6) / 4.786038e-05   # -0.2%  lam2 < 0.0005373 and absphi_6 < 0.1592
        + 0.002057611 * max(0.0, 0.875672 - Q.D2) / 0.1390856   # +0.2%  D2 < 0.8757
        + 0.001926457 * max(0.0, Q.sd_rg - 0.2037854) * max(0.0, 176.25 - Q.pt_0) / 0.6276792   # +0.2%  sd_rg > 0.2038 and pt_0 < 176.2
        - 0.00186331 * max(0.0, 37.76455 - Q.mass_top5) / 16.71307   # -0.2%  mass_top5 < 37.76
        - 0.00168764 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1626038 - Q.abseta_7) / 0.005313035   # -0.2%  N2 < 0.2233 and abseta_7 < 0.1626
        - 0.001394891 * max(0.0, Q.sd_rg - 0.1667615) * max(0.0, 0.3331409 - Q.z_1st) / 0.001855799   # -0.1%  sd_rg > 0.1668 and z_1st < 0.3331
        - 0.001362497 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.e2_sq - 0.01165737) / 7.11886e-05   # -0.1%  N2 < 0.2233 and e2_sq > 0.01166
        - 0.001351648 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # -0.1%  lam1 < 0.002464
        - 0.001277801 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.25 - Q.pt_6) / 1.048693   # -0.1%  N2 < 0.2233 and pt_6 < 62.25
        + 0.001271543 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # +0.1%  girth > 0.1019
        + 0.0009871974 * max(0.0, Q.sd_rg - 0.1667615) / 0.02547986   # +0.1%  sd_rg > 0.1668
        - 0.0009339379 * max(0.0, Q.sd_mass - 38.43971) * max(0.0, 153.375 - Q.pt_0) / 125.7046   # -0.1%  sd_mass > 38.44 and pt_0 < 153.4
        - 0.000871637 * max(0.0, Q.sj3_dr_max - 0.3456459) / 0.006655619   # -0.1%  sj3_dr_max > 0.3456
        + 0.0008407408 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, Q.phi_0 - -0.0297699) / 1.3376e-05   # +0.1%  lam2 < 0.0005373 and phi_0 > -0.02977
        - 0.000806914 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1551147 - Q.dr_4) / 0.003500706   # -0.1%  N2 < 0.2233 and dr_4 < 0.1551
        - 0.0007955892 * max(0.0, 1.340118e-05 - Q.e3) / 5.081834e-06   # -0.1%  e3 < 1.34e-05
        + 0.0007673405 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.abseta_0 - 0.02487183) / 0.001302228   # +0.1%  N2 < 0.2233 and abseta_0 > 0.02487
        - 0.0006986455 * max(0.0, 0.03981924 - Q.zdr_0) * max(0.0, Q.sj3_dr23 - 0.1414609) / 0.000822063   # -0.1%  zdr_0 < 0.03982 and sj3_dr23 > 0.1415
        - 0.0006944815 * max(0.0, Q.e2 - 0.009668065) * max(0.0, Q.mean_phi - -0.009352575) / 0.0002724497   # -0.1%  e2 > 0.009668 and mean_phi > -0.009353
        + 0.0006930131 * max(0.0, 0.01165737 - Q.e2_sq) * max(0.0, 0.875672 - Q.D2) / 0.0005867961   # +0.1%  e2_sq < 0.01166 and D2 < 0.8757
        - 0.0006749471 * max(0.0, 1.002471 - Q.D2) * max(0.0, 0.1132812 - Q.absphi_5) / 0.01081292   # -0.1%  D2 < 1.002 and absphi_5 < 0.1133
        + 0.0006748887 * max(0.0, Q.sd_rg - 0.324646) / 0.001748965   # +0.1%  sd_rg > 0.3246
        - 0.0006689458 * max(0.0, Q.mass - 88.15578) / 0.7232262   # -0.1%  mass > 88.16
        + 0.0006012479 * max(0.0, Q.mean_eta - 0.006823012) / 0.003151938   # +0.1%  mean_eta > 0.006823
        + 0.0005860542 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 60.63098 - Q.mass) / 0.3406587   # +0.1%  N2 < 0.2233 and mass < 60.63
        - 0.0004706431 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, 0.02117036 - Q.dr01) / 0.0003970056   # -0.0%  sj3_dr_max > 0.169 and dr01 < 0.02117
        + 0.0004609153 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, 130.375 - Q.pt_0) / 0.287519   # +0.0%  sj3_dr_max > 0.2337 and pt_0 < 130.4
        - 0.000459668 * max(0.0, Q.max_dr - 0.2215867) / 0.008157178   # -0.0%  max_dr > 0.2216
        + 0.0004314375 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, Q.absphi_7 - 0.02111816) / 2.405358e-05   # +0.0%  lam2 < 0.001131 and absphi_7 > 0.02112
        - 0.0004216292 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # -0.0%  mean_eta > 0.02644
        - 0.0004215482 * max(0.0, 739.5 - Q.sum_pt) * max(0.0, Q.n_dr_0p05_0p1 - 0.0) / 212.6759   # -0.0%  sum_pt < 739.5 and n_dr_0p05_0p1 > 0
        - 0.0003603446 * max(0.0, Q.sj3_dr_max - 0.3456459) * max(0.0, 118.1875 - Q.pt_0) / 0.05295229   # -0.0%  sj3_dr_max > 0.3456 and pt_0 < 118.2
        + 0.0003595923 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, -0.01790907 - Q.mean_eta) / 2.392698e-06   # +0.0%  girth2_top2 < 0.00764 and mean_eta < -0.01791
        + 0.0003435252 * max(0.0, 0.2233283 - Q.N2) * max(0.0, -0.009352575 - Q.mean_phi) / 0.0001205535   # +0.0%  N2 < 0.2233 and mean_phi < -0.009353
        - 0.0003369866 * max(0.0, 0.004811143 - Q.tau21_b2) / 0.0001640305   # -0.0%  tau21_b2 < 0.004811
        - 0.0003305076 * max(0.0, 430.75 - Q.sum_pt_top5) / 14.05246   # -0.0%  sum_pt_top5 < 430.8
        - 0.0002202343 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 25.57812 - Q.pt_7) / 0.037942   # -0.0%  N2 < 0.2233 and pt_7 < 25.58
        - 0.0002112374 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, -0.01753483 - Q.mean_phi) / 2.64217e-07   # -0.0%  lam2 < 0.0005373 and mean_phi < -0.01753
        + 0.0002028837 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, -0.01753483 - Q.mean_phi) / 2.381606e-06   # +0.0%  girth2_top2 < 0.00764 and mean_phi < -0.01753
        + 0.0001965908 * max(0.0, 76.6557 - Q.mass) * max(0.0, Q.mean_eta2 - 0.006395693) / 0.005490776   # +0.0%  mass < 76.66 and mean_eta2 > 0.006396
        - 0.0001696282 * max(0.0, 0.05119235 - Q.C2) * max(0.0, Q.mean_eta - 0.02644207) / 1.096062e-05   # -0.0%  C2 < 0.05119 and mean_eta > 0.02644
        - 0.00014027 * max(0.0, 0.2838437 - Q.tau21) * max(0.0, Q.dr1_7 - 0.2411619) / 0.0005548094   # -0.0%  tau21 < 0.2838 and dr1_7 > 0.2412
        + 0.0001317 * max(0.0, 0.004811143 - Q.tau21_b2) * max(0.0, 43.5 - Q.pt_7) / 0.001191607   # +0.0%  tau21_b2 < 0.004811 and pt_7 < 43.5
        - 7.136303e-05 * max(0.0, 0.002635418 - Q.girth2) * max(0.0, Q.dr1_7 - 0.1343804) / 1.289265e-06   # -0.0%  girth2 < 0.002635 and dr1_7 > 0.1344
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 11.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.78131 * (0.06732041
        + 0.0925312 * max(0.0, Q.log_sum_pt - 6.267538) / 0.2983143   # +9.3%  log_sum_pt > 6.268
        + 0.06693951 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +6.7%  tau1 < 0.09539
        - 0.0554905 * max(0.0, Q.pt_7 - 23.21641) / 12.35308   # -5.5%  pt_7 > 23.22
        - 0.05502403 * max(0.0, 0.09041383 - Q.mass_over_sum_pt) / 0.03744004   # -5.5%  mass_over_sum_pt < 0.09041
        - 0.0490308 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -4.9%  sj3_dr_max < 0.3012
        + 0.04273733 * max(0.0, 0.1879486 - Q.sj3_dr_max) / 0.05910331   # +4.3%  sj3_dr_max < 0.1879
        + 0.03991997 * max(0.0, 0.002635418 - Q.girth2) / 0.0007941346   # +4.0%  girth2 < 0.002635
        - 0.03573325 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -3.6%  mass_over_sum_pt_sq < 0.01166
        + 0.03565288 * max(0.0, 0.005284669 - Q.e2_sq) / 0.00223735   # +3.6%  e2_sq < 0.005285
        + 0.03499135 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset) / 7.0272e-06   # +3.5%  girth2 < 0.001654 and centroid_offset < 0.02355
        + 0.03465637 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +3.5%  z_7 < 0.07149
        - 0.03350759 * max(0.0, 43.5 - Q.pt_7) / 10.28861   # -3.4%  pt_7 < 43.5
        - 0.02617282 * max(0.0, 37.15625 - Q.pt_7) / 5.803748   # -2.6%  pt_7 < 37.16
        - 0.02283673 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0004149607   # -2.3%  z_7 < 0.07149 and centroid_offset < 0.03117
        - 0.02254873 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # -2.3%  sj3_dr_max < 0.107
        - 0.02001477 * max(0.0, 0.002915531 - Q.girth2_top3) / 0.001178811   # -2.0%  girth2_top3 < 0.002916
        - 0.01933632 * max(0.0, Q.log_sum_pt - 6.267538) * max(0.0, 0.03981924 - Q.zdr_0) / 0.008301571   # -1.9%  log_sum_pt > 6.268 and zdr_0 < 0.03982
        + 0.01891867 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, 0.01437952 - Q.centroid_offset) / 1.318015e-05   # +1.9%  e2_sq < 0.005285 and centroid_offset < 0.01438
        + 0.01779032 * max(0.0, Q.sum_pt_top5 - 716.8828) / 29.12383   # +1.8%  sum_pt_top5 > 716.9
        - 0.01772061 * max(0.0, 0.02383244 - Q.zdr_0) / 0.01109706   # -1.8%  zdr_0 < 0.02383
        - 0.01646748 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -1.6%  log_sum_pt > 6.701
        + 0.0155291 * max(0.0, 60.63098 - Q.mass) * max(0.0, 3.885568 - Q.D2) / 50.41033   # +1.6%  mass < 60.63 and D2 < 3.886
        + 0.01518352 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +1.5%  z_7 < 0.02807
        + 0.01390552 * max(0.0, 0.0586137 - Q.z_7) / 0.01231218   # +1.4%  z_7 < 0.05861
        - 0.01297695 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.00424745 - Q.mean_eta2) / 0.0001193758   # -1.3%  log_sum_pt > 6.701 and mean_eta2 < 0.004247
        + 0.01281662 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +1.3%  LHA < 0.2161
        - 0.01224431 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -1.2%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.0120195 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 68.43422 - Q.mass_top5) / 0.3465446   # +1.2%  z_7 < 0.0494 and mass_top5 < 68.43
        + 0.01005527 * max(0.0, 0.04939969 - Q.z_7) / 0.007457505   # +1.0%  z_7 < 0.0494
        + 0.009695971 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.06112084 - Q.dr_max_012) / 0.001464096   # +1.0%  log_sum_pt > 6.701 and dr_max_012 < 0.06112
        - 0.009667047 * max(0.0, Q.pt_7 - 23.21641) * max(0.0, 0.07141113 - Q.absphi_1) / 0.5007982   # -1.0%  pt_7 > 23.22 and absphi_1 < 0.07141
        + 0.009659098 * max(0.0, 0.04737278 - Q.z_6) / 0.004344387   # +1.0%  z_6 < 0.04737
        + 0.007579228 * max(0.0, Q.log_sum_pt - 6.267538) * max(0.0, 0.0113471 - Q.C3) / 0.0005276169   # +0.8%  log_sum_pt > 6.268 and C3 < 0.01135
        + 0.007255176 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 9.030369e-05 - Q.mean_eta2) / 9.99897e-07   # +0.7%  log_sum_pt > 6.701 and mean_eta2 < 9.03e-05
        + 0.006844796 * max(0.0, 37.15625 - Q.pt_7) * max(0.0, 0.01257746 - Q.zdr_3) / 0.04815576   # +0.7%  pt_7 < 37.16 and zdr_3 < 0.01258
        - 0.006436008 * max(0.0, 0.02383244 - Q.zdr_0) * max(0.0, 0.02090454 - Q.abseta_0) / 0.0001122283   # -0.6%  zdr_0 < 0.02383 and abseta_0 < 0.0209
        - 0.006021062 * max(0.0, Q.log_sum_pt - 6.267538) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.1905581   # -0.6%  log_sum_pt > 6.268 and z_dr_0p05_0p1 < 0.846
        + 0.005748106 * max(0.0, 0.09538712 - Q.tau1) * max(0.0, 0.4926918 - Q.planar_flow) / 0.008723222   # +0.6%  tau1 < 0.09539 and planar_flow < 0.4927
        + 0.005155518 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.001736329 - Q.zdr_3) / 2.305525e-05   # +0.5%  log_sum_pt > 6.701 and zdr_3 < 0.001736
        - 0.005042853 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 4.721224 - Q.D2_b2) / 0.09403434   # -0.5%  log_sum_pt > 6.701 and D2_b2 < 4.721
        - 0.00498557 * max(0.0, 0.002915531 - Q.girth2_top3) * max(0.0, 0.006646257 - Q.tau3) / 3.390069e-06   # -0.5%  girth2_top3 < 0.002916 and tau3 < 0.006646
        - 0.0047553 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.01452637 - Q.absphi_0) / 3.722671e-06   # -0.5%  girth2 < 0.001654 and absphi_0 < 0.01453
        - 0.004620522 * max(0.0, Q.sum_pt_top5 - 716.8828) * max(0.0, 9.030369e-05 - Q.mean_eta2) / 0.0008286902   # -0.5%  sum_pt_top5 > 716.9 and mean_eta2 < 9.03e-05
        - 0.004593236 * max(0.0, Q.log_sum_pt - 6.766778) / 0.01900249   # -0.5%  log_sum_pt > 6.767
        + 0.004453138 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0002302115 - Q.mean_phi2) / 3.738179e-06   # +0.4%  log_sum_pt > 6.701 and mean_phi2 < 0.0002302
        + 0.003912569 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03532852 - Q.C3) / 0.0003034905   # +0.4%  z_7 < 0.07149 and C3 < 0.03533
        - 0.003903965 * max(0.0, 0.009518026 - Q.C3) / 0.0007952626   # -0.4%  C3 < 0.009518
        - 0.003442393 * max(0.0, Q.max_pair_mass - 21.83614) / 2.887317   # -0.3%  max_pair_mass > 21.84
        - 0.00327383 * max(0.0, Q.log_sum_pt - 6.766778) * max(0.0, 0.05481757 - Q.C3) / 0.0006853593   # -0.3%  log_sum_pt > 6.767 and C3 < 0.05482
        - 0.002996504 * max(0.0, Q.max_pair_mass - 21.83614) * max(0.0, 0.2366102 - Q.dr_max_012) / 0.07844215   # -0.3%  max_pair_mass > 21.84 and dr_max_012 < 0.2366
        + 0.002273745 * max(0.0, 0.02160287 - Q.z_6) * max(0.0, 0.1583252 - Q.abseta_6) / 3.551538e-05   # +0.2%  z_6 < 0.0216 and abseta_6 < 0.1583
        - 0.002121939 * max(0.0, 0.04737278 - Q.z_6) * max(0.0, 0.001736329 - Q.zdr_3) / 2.629876e-06   # -0.2%  z_6 < 0.04737 and zdr_3 < 0.001736
        - 0.001869857 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.002727284 - Q.zdr_3) / 6.209553e-06   # -0.2%  log_sum_pt > 6.896 and zdr_3 < 0.002727
        + 0.001641564 * max(0.0, 9.121392e-05 - Q.girth2) / 4.152703e-06   # +0.2%  girth2 < 9.121e-05
        - 0.001510443 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.03086731 - Q.dr_max_012) / 7.260647e-05   # -0.2%  log_sum_pt > 6.896 and dr_max_012 < 0.03087
        + 0.001487021 * max(0.0, 0.02818362 - Q.z_5) * max(0.0, 0.1626038 - Q.abseta_7) / 4.227915e-05   # +0.1%  z_5 < 0.02818 and abseta_7 < 0.1626
        + 0.001470811 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # +0.1%  pt_7 > 53.44
        - 0.001429214 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, Q.C2_b2 - 0.001109927) / 4.204538e-05   # -0.1%  z_7 < 0.07149 and C2_b2 > 0.00111
        - 0.001427826 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.mean_phi - 0.0006973656) / 6.386425e-05   # -0.1%  log_sum_pt > 6.701 and mean_phi > 0.0006974
        + 0.001018884 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.mean_phi2 - 0.0007823696) / 2.732408e-05   # +0.1%  log_sum_pt > 6.701 and mean_phi2 > 0.0007824
        - 0.0009547843 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.0002302115 - Q.mean_phi2) / 4.941235e-07   # -0.1%  log_sum_pt > 6.896 and mean_phi2 < 0.0002302
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 24.47;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.46547 * (0.03976136
        - 0.155956 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -15.6%  girth2 < 0.008678
        - 0.1386645 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # -13.9%  width < 0.01324
        + 0.0822412 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +8.2%  lam1 < 0.012
        + 0.06952481 * max(0.0, 0.09041383 - Q.mass_over_sum_pt) / 0.03744004   # +7.0%  mass_over_sum_pt < 0.09041
        + 0.06728689 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +6.7%  lam1 < 0.00733
        + 0.06040254 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +6.0%  lam2 < 0.003408
        + 0.04490745 * max(0.0, 0.1789613 - Q.sj3_dr_max) / 0.05394779   # +4.5%  sj3_dr_max < 0.179
        - 0.03328811 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # -3.3%  max_dr < 0.1452
        + 0.03137592 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +3.1%  tau1 < 0.1136
        - 0.02996184 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 1.341509   # -3.0%  mass < 60.63 and z_dr_0p2_0p4 < 0.05644
        + 0.02929616 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 988.4078 - Q.sum_pt) / 1177.276   # +2.9%  pt_6 < 41.22 and sum_pt < 988.4
        - 0.02925868 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -2.9%  sj3_dr_max < 0.3012
        - 0.02789605 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -2.8%  lam2 < 0.001131
        + 0.0250832 * max(0.0, 0.02415398 - Q.C2_b2) / 0.02132459   # +2.5%  C2_b2 < 0.02415
        - 0.01803893 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # -1.8%  pt_6 < 41.22
        + 0.01677562 * max(0.0, 60.63098 - Q.mass) / 24.47895   # +1.7%  mass < 60.63
        - 0.0154938 * max(0.0, 45.7571 - Q.mass) / 14.82409   # -1.5%  mass < 45.76
        - 0.01295066 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 0.2534037 - Q.planar_flow) / 0.0006566196   # -1.3%  lam1 < 0.012 and planar_flow < 0.2534
        - 0.01111939 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -1.1%  lam1 < 0.005954
        + 0.01051195 * max(0.0, 0.0030133 - Q.e2_sq) / 0.001065944   # +1.1%  e2_sq < 0.003013
        - 0.009695962 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -1.0%  pt_6 < 39.75
        + 0.009412814 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +0.9%  centroid_offset > 0.008092
        + 0.008570511 * max(0.0, Q.eccentricity - 0.927072) / 0.03232992   # +0.9%  eccentricity > 0.9271
        - 0.008169047 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757) / 0.07098489   # -0.8%  pt_6 < 41.22 and z_7 > 0.02321
        + 0.006258527 * max(0.0, 0.03932388 - Q.z_6) / 0.002360011   # +0.6%  z_6 < 0.03932
        - 0.005561118 * max(0.0, Q.sj3_dr_min - 0.02400746) / 0.02802924   # -0.6%  sj3_dr_min > 0.02401
        + 0.0041091 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +0.4%  sj3_dr_max > 0.1879
        + 0.004004188 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002658741   # +0.4%  lam2 < 0.003408 and n_dr_0_0p05 < 3
        + 0.003979107 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.0753896 - Q.z_7) / 0.963884   # +0.4%  sum_pt < 763.8 and z_7 < 0.07539
        - 0.003931792 * max(0.0, Q.sj3_pair_mass_min - 4.501727) / 3.480268   # -0.4%  sj3_pair_mass_min > 4.502
        - 0.003208692 * max(0.0, Q.centroid_offset - 0.01837778) / 0.005523694   # -0.3%  centroid_offset > 0.01838
        + 0.002930631 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 43.23531   # +0.3%  sum_pt < 615.9 and n_dr_0p2_0p4 < 2
        - 0.002842328 * max(0.0, 763.825 - Q.sum_pt) / 96.68366   # -0.3%  sum_pt < 763.8
        + 0.002751496 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2) / 2.179362e-05   # +0.3%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        + 0.002304085 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.psi_0p1 - 0.4008925) / 0.003485257   # +0.2%  centroid_offset > 0.008092 and psi_0p1 > 0.4009
        + 0.002130495 * max(0.0, 615.875 - Q.sum_pt) / 30.07766   # +0.2%  sum_pt < 615.9
        + 0.001698842 * max(0.0, 488.9312 - Q.sum_pt) / 6.000379   # +0.2%  sum_pt < 488.9
        + 0.00163338 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 0.1343804 - Q.dr1_7) / 0.3834372   # +0.2%  pt_6 < 41.22 and dr1_7 < 0.1344
        + 0.001502474 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # +0.2%  sj2_dr < 0.1873
        - 0.0009955516 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 2.69383   # -0.1%  sj3_pair_mass_min > 4.502 and n_dr_0p2_0p4 > 1
        + 0.0009463879 * max(0.0, 19.46875 - Q.pt_6) / 0.2515892   # +0.1%  pt_6 < 19.47
        - 0.0006400045 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # -0.1%  mean_eta > 0.02644
        + 0.0005821318 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.mean_eta - 0.02644207) / 1.54475e-06   # +0.1%  lam1 < 0.012 and mean_eta > 0.02644
        + 0.0005633495 * max(0.0, 0.01323868 - Q.width) * max(0.0, -0.02665591 - Q.mean_eta) / 1.690289e-06   # +0.1%  width < 0.01324 and mean_eta < -0.02666
        + 0.0004237204 * max(0.0, 0.01323868 - Q.width) * max(0.0, -0.02594505 - Q.mean_phi) / 1.701156e-06   # +0.0%  width < 0.01324 and mean_phi < -0.02595
        + 0.0003986104 * max(0.0, 0.01323868 - Q.width) * max(0.0, Q.mean_phi - 0.02612796) / 1.673749e-06   # +0.0%  width < 0.01324 and mean_phi > 0.02613
        + 0.0003705797 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, Q.sum_pt_top5 - 658.125) / 0.02654014   # +0.0%  centroid_offset > 0.01838 and sum_pt_top5 > 658.1
        + 0.000351359 * max(0.0, 29.90625 - Q.pt_6) * max(0.0, 8.0 - Q.n_pt_above_10) / 0.5895401   # +0.0%  pt_6 < 29.91 and n_pt_above_10 < 8
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 44.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 44.17968 * (0.2457684
        - 0.09265799 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -9.3%  girth2 > 0.00752
        - 0.0699754 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -7.0%  girth < 0.08724
        - 0.06672384 * max(0.0, Q.width - 0.005590289) / 0.003084256   # -6.7%  width > 0.00559
        - 0.06508108 * max(0.0, Q.mass_over_sum_pt - 0.01109984) / 0.05101851   # -6.5%  mass_over_sum_pt > 0.0111
        - 0.06061875 * max(0.0, 0.006679471 - Q.width) / 0.00293624   # -6.1%  width < 0.006679
        + 0.05542455 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +5.5%  mass_over_sum_pt > 0.08475
        - 0.04513703 * max(0.0, Q.girth2 - 0.004372139) / 0.003638805   # -4.5%  girth2 > 0.004372
        + 0.04462442 * max(0.0, Q.mass_over_sum_pt - 0.07269073) / 0.01344138   # +4.5%  mass_over_sum_pt > 0.07269
        + 0.03288531 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +3.3%  sj2_dr < 0.1592
        - 0.03206458 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -3.2%  sj2_dr < 0.1873
        + 0.03149728 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +3.1%  LHA < 0.3033
        + 0.02852018 * max(0.0, Q.sj3_dr_max - 0.1426152) / 0.06279029   # +2.9%  sj3_dr_max > 0.1426
        + 0.02817616 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +2.8%  mass_over_sum_pt > 0.07992
        - 0.02565429 * max(0.0, Q.girth2 - 0.001653836) / 0.005220304   # -2.6%  girth2 > 0.001654
        - 0.0251843 * max(0.0, Q.girth2 - 0.01323868) / 0.001442684   # -2.5%  girth2 > 0.01324
        - 0.02458165 * max(0.0, Q.sj3_dr_max - 0.04889979) / 0.1256943   # -2.5%  sj3_dr_max > 0.0489
        - 0.01669543 * max(0.0, Q.width - 0.008678045) / 0.002214537   # -1.7%  width > 0.008678
        + 0.01628886 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.6%  tau1 < 0.05357
        - 0.01496948 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -1.5%  lam1 < 0.008376
        + 0.01195325 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +1.2%  e2 > 0.05028
        - 0.01081312 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # -1.1%  girth2 < 0.003563
        + 0.0103209 * max(0.0, Q.mass - 36.22941) / 14.20528   # +1.0%  mass > 36.23
        - 0.01024286 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.004032342 - Q.C2_b2) / 2.736263e-05   # -1.0%  centroid_offset < 0.02077 and C2_b2 < 0.004032
        + 0.01017535 * max(0.0, 0.08723651 - Q.girth) * max(0.0, 0.004032342 - Q.C2_b2) / 0.0001217265   # +1.0%  girth < 0.08724 and C2_b2 < 0.004032
        + 0.01017106 * max(0.0, Q.girth2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +1.0%  girth2 > 0.004372 and planar_flow < 0.195
        - 0.00806916 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # -0.8%  max_dr < 0.1118
        - 0.007829881 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # -0.8%  lam2 < 0.0003061
        + 0.007656289 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # +0.8%  max_dr < 0.1773
        + 0.007608667 * max(0.0, Q.width - 0.008678045) * max(0.0, 38.25 - Q.pt_6) / 0.006941575   # +0.8%  width > 0.008678 and pt_6 < 38.25
        - 0.007133764 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -0.7%  sj3_dr_max > 0.2623
        - 0.006728512 * max(0.0, Q.sd_rg - 0.2037854) / 0.01566501   # -0.7%  sd_rg > 0.2038
        - 0.006162408 * max(0.0, 0.04019753 - Q.tau21_b2) / 0.01109641   # -0.6%  tau21_b2 < 0.0402
        + 0.006162221 * max(0.0, 0.0003061234 - Q.lam2) * max(0.0, 0.04019753 - Q.tau21_b2) / 2.611351e-06   # +0.6%  lam2 < 0.0003061 and tau21_b2 < 0.0402
        - 0.006007842 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -0.6%  centroid_offset < 0.02077
        - 0.005765321 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.width - 0.007520088) / 0.0001455029   # -0.6%  planar_flow < 0.195 and width > 0.00752
        - 0.005752091 * max(0.0, Q.mass - 76.6557) * max(0.0, Q.zdr_6 - 0.006366792) / 0.006641425   # -0.6%  mass > 76.66 and zdr_6 > 0.006367
        - 0.005558209 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.6%  mass > 76.66
        - 0.005534453 * max(0.0, Q.girth2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9458207) / 2.125219e-05   # -0.6%  girth2 > 0.01324 and eccentricity > 0.9458
        + 0.005119521 * max(0.0, Q.z_7 - 0.03243272) / 0.02180051   # +0.5%  z_7 > 0.03243
        + 0.004959499 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # +0.5%  e2 > 0.03556
        - 0.004892016 * max(0.0, 0.0009641429 - Q.girth2) / 0.0002052288   # -0.5%  girth2 < 0.0009641
        - 0.004502584 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -0.5%  mass_over_sum_pt > 0.09041
        - 0.004337127 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.4%  centroid_offset > 0.03776
        + 0.004317347 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # +0.4%  sd_rg > 0.2788
        + 0.00426999 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, Q.sum_pt_top3 - 331.25) / 1.710819   # +0.4%  centroid_offset < 0.02077 and sum_pt_top3 > 331.2
        + 0.004268773 * max(0.0, 2.357246 - Q.D2) / 1.097329   # +0.4%  D2 < 2.357
        - 0.004054289 * max(0.0, 0.04081947 - Q.girth) / 0.009176315   # -0.4%  girth < 0.04082
        + 0.003743373 * max(0.0, Q.mass_over_sum_pt - 0.1079857) / 0.005364315   # +0.4%  mass_over_sum_pt > 0.108
        + 0.003364771 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sd_mass - 38.43971) / 1.25962   # +0.3%  planar_flow < 0.195 and sd_mass > 38.44
        + 0.002978031 * max(0.0, 0.001056655 - Q.girth2_top2) / 0.0003002514   # +0.3%  girth2_top2 < 0.001057
        - 0.002362092 * max(0.0, Q.mass_over_sum_pt - 0.09041383) * max(0.0, 36.8125 - Q.pt_6) / 0.01975833   # -0.2%  mass_over_sum_pt > 0.09041 and pt_6 < 36.81
        - 0.00234471 * max(0.0, Q.sj3_dr_max - 0.213399) / 0.03061279   # -0.2%  sj3_dr_max > 0.2134
        + 0.001843138 * max(0.0, 0.006679471 - Q.width) * max(0.0, 0.006452173 - Q.mean_phi) / 2.379971e-05   # +0.2%  width < 0.006679 and mean_phi < 0.006452
        - 0.001764505 * max(0.0, 29.04219 - Q.pt_7) / 2.179984   # -0.2%  pt_7 < 29.04
        - 0.001678511 * max(0.0, Q.mass_over_sum_pt - 0.01109984) * max(0.0, 35.28125 - Q.pt_6) / 0.1065511   # -0.2%  mass_over_sum_pt > 0.0111 and pt_6 < 35.28
        + 0.001664114 * max(0.0, Q.girth2 - 0.01323868) * max(0.0, 0.06970457 - Q.dr_6) / 2.606496e-06   # +0.2%  girth2 > 0.01324 and dr_6 < 0.0697
        + 0.001611601 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.02656143 - Q.tau21_b2) / 4.805781e-05   # +0.2%  centroid_offset < 0.02077 and tau21_b2 < 0.02656
        + 0.001360482 * max(0.0, Q.girth2 - 0.01323868) * max(0.0, 38.25 - Q.pt_6) / 0.004519922   # +0.1%  girth2 > 0.01324 and pt_6 < 38.25
        - 0.001244874 * max(0.0, 8.10414e-05 - Q.girth2_top2) / 7.505045e-06   # -0.1%  girth2_top2 < 8.104e-05
        + 0.0012146 * max(0.0, Q.mass_top5 - 53.60766) / 1.978814   # +0.1%  mass_top5 > 53.61
        - 0.001101619 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -0.1%  sj2_dr < 0.1295
        + 0.001035392 * max(0.0, Q.mass - 69.61135) / 2.406702   # +0.1%  mass > 69.61
        - 0.0008611012 * max(0.0, 0.05744392 - Q.D2_b2) / 0.005938983   # -0.1%  D2_b2 < 0.05744
        - 0.0005759292 * max(0.0, 0.08723651 - Q.girth) * max(0.0, 0.004406178 - Q.mean_phi) / 0.0002325096   # -0.1%  girth < 0.08724 and mean_phi < 0.004406
        - 0.0005364692 * max(0.0, Q.centroid_offset - 0.03117077) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.00188116   # -0.1%  centroid_offset > 0.03117 and n_pt_above_50 > 4
        + 0.0004765728 * max(0.0, Q.sd_rg - 0.2037854) * max(0.0, 0.06970457 - Q.dr_6) / 4.971602e-05   # +0.0%  sd_rg > 0.2038 and dr_6 < 0.0697
        - 0.0004628155 * max(0.0, 29.04219 - Q.pt_7) * max(0.0, 0.2838437 - Q.tau21) / 0.1125981   # -0.0%  pt_7 < 29.04 and tau21 < 0.2838
        - 0.0002800124 * max(0.0, 0.001056655 - Q.girth2_top2) * max(0.0, 0.02656143 - Q.tau21_b2) / 2.239525e-07   # -0.0%  girth2_top2 < 0.001057 and tau21_b2 < 0.02656
        - 0.0002643659 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, Q.n_dr_0p05_0p1 - 5.0) / 0.0005328768   # -0.0%  tau1 < 0.05357 and n_dr_0p05_0p1 > 5
        - 0.0001038387 * max(0.0, Q.centroid_offset - 0.03776099) * max(0.0, Q.pt_4 - 81.375) / 0.0001936307   # -0.0%  centroid_offset > 0.03776 and pt_4 > 81.38
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 15.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.17717 * (-0.01610969
        - 0.1072949 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.003562611 - Q.girth2) / 5.329563e-05   # -10.7%  girth < 0.06109 and girth2 < 0.003563
        - 0.08711617 * max(0.0, 0.06108601 - Q.girth) / 0.01868685   # -8.7%  girth < 0.06109
        + 0.07583137 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.girth2) / 0.0003703114   # +7.6%  sj3_dr_max < 0.1986 and girth2 < 0.006679
        + 0.07321812 * max(0.0, 0.003562611 - Q.width) / 0.001184249   # +7.3%  width < 0.003563
        + 0.06733954 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.width) / 4.808955e-05   # +6.7%  tau1 < 0.05357 and width < 0.003563
        + 0.05610484 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0001901112   # +5.6%  girth2 < 0.00502 and z_dr_0p2_0p4 < 0.101
        + 0.0334693 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +3.3%  girth2 < 0.006679 and centroid_offset < 0.02355
        + 0.03321889 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.0001272072   # +3.3%  mass_over_sum_pt < 0.03319 and centroid_offset < 0.02686
        + 0.02936707 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +2.9%  girth2 < 0.00502
        - 0.02877986 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -2.9%  sj3_dr_max < 0.1426
        - 0.02783377 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.1231549   # -2.8%  mass < 29.64 and centroid_offset < 0.02686
        - 0.02736692 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.003562611 - Q.width) / 0.0001781324   # -2.7%  sj3_dr_max < 0.1986 and width < 0.003563
        - 0.02473953 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.0001947983 - Q.lam2) / 0.0007178522   # -2.5%  mass < 21.78 and lam2 < 0.0001948
        - 0.02454031 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.005691733 - Q.girth2_top5) / 0.0003116052   # -2.5%  sj3_dr_max < 0.1986 and girth2_top5 < 0.005692
        - 0.01973183 * max(0.0, 0.01289969 - Q.e2) * max(0.0, Q.psi_0p2 - 0.79448) / 0.0005184882   # -2.0%  e2 < 0.0129 and psi_0p2 > 0.7945
        + 0.01926611 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2) / 2.869243e-06   # +1.9%  girth < 0.06109 and lam2 < 0.0001948
        - 0.01889282 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -1.9%  log_sum_pt > 6.701
        - 0.01786134 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) / 0.00722828   # -1.8%  mass_over_sum_pt < 0.03319
        - 0.01773692 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -1.8%  girth2 < 0.00502 and centroid_offset > 0.00679
        - 0.01533011 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # -1.5%  LHA < 0.1967
        + 0.01521604 * max(0.0, 0.0001330621 - Q.lam2) * max(0.0, 4.721224 - Q.D2_b2) / 0.0002340602   # +1.5%  lam2 < 0.0001331 and D2_b2 < 4.721
        - 0.01425118 * max(0.0, 0.0001330621 - Q.lam2) / 6.711676e-05   # -1.4%  lam2 < 0.0001331
        + 0.01305597 * max(0.0, 0.01289969 - Q.e2) / 0.002527207   # +1.3%  e2 < 0.0129
        - 0.0120752 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 4.721224 - Q.D2_b2) / 0.008595177   # -1.2%  girth2 < 0.006679 and D2_b2 < 4.721
        - 0.01168655 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 43.5 - Q.pt_7) / 0.02255472   # -1.2%  girth2 < 0.00502 and pt_7 < 43.5
        - 0.01070778 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.001011075   # -1.1%  girth < 0.06109 and z_dr_0p2_0p4 < 0.05644
        + 0.01058086 * max(0.0, 29.6447 - Q.mass) * max(0.0, 1.762929e-06 - Q.e3) / 9.656157e-06   # +1.1%  mass < 29.64 and e3 < 1.763e-06
        + 0.01031278 * max(0.0, 0.002635418 - Q.width) / 0.0007941347   # +1.0%  width < 0.002635
        + 0.01014606 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.0001947983 - Q.lam2) / 9.822054e-06   # +1.0%  sj3_dr_max < 0.1986 and lam2 < 0.0001948
        - 0.008158203 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.01627885) / 0.0001130126   # -0.8%  sj3_dr_max < 0.1426 and centroid_offset > 0.01628
        + 0.007577697 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.8%  sum_pt_top5 > 687.4
        + 0.007497679 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.phi_0 - -0.04013062) / 0.0001189753   # +0.7%  girth2 < 0.006679 and phi_0 > -0.04013
        + 0.0067222 * max(0.0, 49.6681 - Q.mass) * max(0.0, 0.0001330621 - Q.lam2) / 0.001509268   # +0.7%  mass < 49.67 and lam2 < 0.0001331
        + 0.006705871 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +0.7%  sj3_dr_max < 0.107
        + 0.00656141 * max(0.0, 0.06108601 - Q.girth) * max(0.0, Q.width - 0.0003193707) / 1.048473e-05   # +0.7%  girth < 0.06109 and width > 0.0003194
        - 0.005829613 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # -0.6%  sj3_dr_max < 0.1986
        + 0.005027355 * max(0.0, 21.78408 - Q.mass) * max(0.0, 48.71875 - Q.pt_7) / 69.12746   # +0.5%  mass < 21.78 and pt_7 < 48.72
        - 0.004825402 * max(0.0, 21.78408 - Q.mass) / 4.431023   # -0.5%  mass < 21.78
        - 0.004257947 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.02355416 - Q.centroid_offset) / 0.0005980206   # -0.4%  log_sum_pt > 6.701 and centroid_offset < 0.02355
        + 0.003886786 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 2.955458e-05 - Q.e3) / 8.40609e-07   # +0.4%  log_sum_pt > 6.701 and e3 < 2.955e-05
        + 0.003844829 * max(0.0, 29.6447 - Q.mass) / 7.34723   # +0.4%  mass < 29.64
        + 0.003388136 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 48.71875 - Q.pt_7) / 0.776931   # +0.3%  log_sum_pt > 6.701 and pt_7 < 48.72
        + 0.002587401 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.pt_7 - 15.55391) / 0.4153091   # +0.3%  LHA < 0.1967 and pt_7 > 15.55
        - 0.002544657 * max(0.0, 0.03448406 - Q.z_6) / 0.001525592   # -0.3%  z_6 < 0.03448
        + 0.002010371 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0005611231 - Q.girth2) / 8.626944e-06   # +0.2%  log_sum_pt > 6.701 and girth2 < 0.0005611
        - 0.001972602 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.2097233   # -0.2%  mass < 29.64 and D2_b2 < 0.7166
        - 0.001518385 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.008678045 - Q.width) / 0.0002365852   # -0.2%  log_sum_pt > 6.701 and width < 0.008678
        + 0.0007917497 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.07002129   # +0.1%  mass < 21.78 and D2_b2 < 0.7166
        - 0.0006971052 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 8.0 - Q.n_pt_above_10) / 7.731645   # -0.1%  sum_pt_top5 > 687.4 and n_pt_above_10 < 8
        - 0.0005224512 * max(0.0, 0.003562611 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738) / 5.559665e-06   # -0.1%  width < 0.003563 and centroid_offset > 0.00679
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 31.1;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.09684 * (-0.09930283
        + 0.1102157 * max(0.0, 0.00609665 - Q.girth2) / 0.002544039   # +11.0%  girth2 < 0.006097
        - 0.0846062 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -8.5%  mass_over_sum_pt < 0.07269
        + 0.06904361 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # +6.9%  girth2 < 0.003563
        + 0.05179963 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.2973626   # +5.2%  mass < 53.33 and centroid_offset < 0.02686
        - 0.05118286 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -5.1%  sj3_dr_max < 0.1426
        + 0.04789956 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +4.8%  sj3_dr_max < 0.2134
        + 0.04705066 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +4.7%  mass_over_sum_pt_sq < 0.007183
        + 0.04330361 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +4.3%  sum_pt < 988.4
        - 0.04233844 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -4.2%  lam1 < 0.005954
        + 0.04150839 * max(0.0, Q.lam1 - 0.005433361) / 0.00267714   # +4.2%  lam1 > 0.005433
        + 0.03333164 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +3.3%  centroid_offset < 0.01838
        - 0.02765034 * max(0.0, 53.33237 - Q.mass) / 19.36004   # -2.8%  mass < 53.33
        + 0.0256581 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +2.6%  girth2 < 0.00502
        - 0.0247285 * max(0.0, 0.05464922 - Q.girth) / 0.01532978   # -2.5%  girth < 0.05465
        - 0.02453233 * max(0.0, 0.0030133 - Q.e2_sq) / 0.001065944   # -2.5%  e2_sq < 0.003013
        - 0.02065616 * max(0.0, Q.lam1 - 0.003377388) / 0.003672352   # -2.1%  lam1 > 0.003377
        + 0.0205571 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # +2.1%  tau1 < 0.0437
        - 0.0166068 * max(0.0, 2.371297e-05 - Q.e3) / 1.105328e-05   # -1.7%  e3 < 2.371e-05
        - 0.01522573 * max(0.0, 29.6447 - Q.mass) / 7.34723   # -1.5%  mass < 29.64
        + 0.01273402 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt) / 5.175436   # +1.3%  mass < 53.33 and log_sum_pt < 6.843
        + 0.01221181 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # +1.2%  lam2 < 0.0003061
        + 0.01153665 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +1.2%  max_dr < 0.1118
        - 0.01054607 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1) / 0.1931178   # -1.1%  centroid_offset < 0.01838 and pt_1 < 159.2
        - 0.008184691 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.008712838   # -0.8%  centroid_offset < 0.01838 and n_for_90pct > 5
        + 0.007972291 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +0.8%  sj3_dr_max < 0.107
        - 0.007909035 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266) / 0.0005312449   # -0.8%  lam1 > 0.005433 and eccentricity > 0.7117
        - 0.007824838 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # -0.8%  e3 > 8.148e-05
        - 0.00747027 * max(0.0, Q.girth - 0.0717028) / 0.01201981   # -0.7%  girth > 0.0717
        - 0.00743509 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -0.7%  e2 > 0.03556
        + 0.006899452 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +0.7%  e2 > 0.04447
        - 0.006612289 * max(0.0, 0.05464922 - Q.girth) * max(0.0, Q.z_6 - 0.03448406) / 0.0003071415   # -0.7%  girth < 0.05465 and z_6 > 0.03448
        + 0.006503873 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd) / 0.0002178654   # +0.7%  centroid_offset < 0.01838 and z_2nd < 0.2056
        + 0.006071618 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +0.6%  e3 > 0.0001869
        + 0.005199848 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.0002201438   # +0.5%  girth2 < 0.006097 and planar_flow < 0.3221
        - 0.004696797 * max(0.0, 7.0 - Q.n_for_90pct) / 0.5603294   # -0.5%  n_for_90pct < 7
        - 0.004251881 * max(0.0, 0.0284695 - Q.C3) / 0.008714909   # -0.4%  C3 < 0.02847
        + 0.004150613 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.6875) / 0.3914557   # +0.4%  sj3_dr_max < 0.1426 and pt_6 > 33.69
        - 0.00369656 * max(0.0, Q.sd_mass - 44.82259) / 8.759065   # -0.4%  sd_mass > 44.82
        - 0.003428753 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.002834884 - Q.mean_phi) / 1.344572e-05   # -0.3%  girth2 < 0.006097 and mean_phi < 0.002835
        + 0.003424933 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.3%  lam2 > 0.001131
        + 0.003193371 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.3%  n_dr_0p2_0p4 > 1
        + 0.003144447 * max(0.0, Q.N2 - 0.2233283) / 0.04872624   # +0.3%  N2 > 0.2233
        - 0.003131351 * Q.z_dr_0p2_0p4 / 0.02920532   # -0.3%  z_dr_0p2_0p4
        - 0.003125878 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255) / 0.000277001   # -0.3%  tau1 < 0.0437 and eccentricity > 0.9031
        + 0.003086749 * max(0.0, 0.0001721983 - Q.width) / 1.441896e-05   # +0.3%  width < 0.0001722
        - 0.003060955 * max(0.0, 0.02227783 - Q.absphi_1) / 0.006894148   # -0.3%  absphi_1 < 0.02228
        - 0.002721384 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) / 6.087676   # -0.3%  sj3_pair_mass_max < 24.23
        + 0.002577537 * max(0.0, Q.mass - 45.7571) / 9.387753   # +0.3%  mass > 45.76
        + 0.002508995 * max(0.0, 0.0782171 - Q.M3) / 0.01197321   # +0.3%  M3 < 0.07822
        + 0.002479689 * max(0.0, 2.371297e-05 - Q.e3) * max(0.0, Q.eccentricity - 0.8319502) / 8.425435e-07   # +0.2%  e3 < 2.371e-05 and eccentricity > 0.832
        + 0.002310531 * max(0.0, 0.06503035 - Q.z_5) / 0.007566857   # +0.2%  z_5 < 0.06503
        - 0.002006327 * max(0.0, 0.006292091 - Q.zdr_0) / 0.001006321   # -0.2%  zdr_0 < 0.006292
        - 0.001785986 * max(0.0, Q.D2 - 2.055451) / 0.3061148   # -0.2%  D2 > 2.055
        + 0.001738224 * max(0.0, 0.002170915 - Q.zdr_2) / 0.0002841564   # +0.2%  zdr_2 < 0.002171
        - 0.001707512 * max(0.0, 988.4078 - Q.sum_pt) * max(0.0, 0.01778111 - Q.dr_2) / 0.3401129   # -0.2%  sum_pt < 988.4 and dr_2 < 0.01778
        - 0.001499245 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.1%  sum_pt_top5 > 840
        - 0.00147021 * max(0.0, Q.girth2_top5 - 0.01148293) / 0.00154713   # -0.1%  girth2_top5 > 0.01148
        + 0.001251332 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.1%  log_sum_pt > 6.896
        + 0.001116856 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) * max(0.0, -0.004917145 - Q.phi_0) / 9.23841e-05   # +0.1%  mass_over_sum_pt < 0.07269 and phi_0 < -0.004917
        - 0.0009297417 * max(0.0, 0.002316125 - Q.centroid_offset) / 9.872932e-05   # -0.1%  centroid_offset < 0.002316
        + 0.0009058512 * max(0.0, 7.0 - Q.n_for_90pct) * max(0.0, Q.dr_2 - 0.00656258) / 0.02004733   # +0.1%  n_for_90pct < 7 and dr_2 > 0.006563
        + 0.0008852097 * max(0.0, 0.05464922 - Q.girth) * max(0.0, 0.02656143 - Q.tau21_b2) / 1.23798e-05   # +0.1%  girth < 0.05465 and tau21_b2 < 0.02656
        - 0.0008798903 * max(0.0, 559.6875 - Q.sum_pt) / 16.18216   # -0.1%  sum_pt < 559.7
        - 0.0008492284 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.01108383   # -0.1%  n_dr_0p2_0p4 > 1 and sj3_dr_min < 0.209
        - 0.0007279967 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.pair_mass_0_2 - 17.9037) / 0.01350362   # -0.1%  centroid_offset < 0.01838 and pair_mass_0_2 > 17.9
        - 0.0006744364 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, 0.02656143 - Q.tau21_b2) / 8.029519e-06   # -0.1%  tau1 < 0.0437 and tau21_b2 < 0.02656
        - 0.0006511778 * max(0.0, Q.e2 - 0.04447357) * max(0.0, 0.2723288 - Q.sj3_mass3) / 0.0006651999   # -0.1%  e2 > 0.04447 and sj3_mass3 < 0.2723
        - 0.0005582901 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top2 - 1.933087) / 0.0177962   # -0.1%  mass_over_sum_pt < 0.07269 and mass_top2 > 1.933
        - 0.0005240647 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.n_dr_0p05_0p1 - 7.0) / 0.5176771   # -0.1%  sd_mass > 44.82 and n_dr_0p05_0p1 > 7
        + 0.0004418951 * max(0.0, 53.33237 - Q.mass) * max(0.0, Q.D2 - 2.843757) / 5.555775   # +0.0%  mass < 53.33 and D2 > 2.844
        + 0.0004052177 * max(0.0, 31.125 - Q.pt_4) / 0.3140792   # +0.0%  pt_4 < 31.12
        - 0.000367826 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2) / 0.000627491   # -0.0%  log_sum_pt > 6.896 and D2_b2 < 0.7166
        - 0.0003133923 * max(0.0, 53.33237 - Q.mass) * max(0.0, Q.dr1_7 - 0.2025074) / 0.02402033   # -0.0%  mass < 53.33 and dr1_7 > 0.2025
        - 0.0002898156 * max(0.0, Q.e3 - 0.0001869378) * max(0.0, 33.6875 - Q.pt_6) / 4.695931e-05   # -0.0%  e3 > 0.0001869 and pt_6 < 33.69
        - 2.266668e-05 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.mean_phi - 0.02612796) / 2.994783e-06   # -0.0%  tau1 < 0.0437 and mean_phi > 0.02613
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 8.249;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.24868 * (0.2914739
        + 0.1504593 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # +15.0%  girth2 > 0.00752
        - 0.1119479 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -11.2%  lam1 < 0.005954
        + 0.0743224 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +7.4%  mass < 76.66
        - 0.06127387 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # -6.1%  lam1 > 0.00733
        + 0.0493421 * max(0.0, Q.lam2 - 0.0003061234) / 0.0004216599   # +4.9%  lam2 > 0.0003061
        - 0.04285999 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -4.3%  LHA > 0.3033
        - 0.04128371 * max(0.0, 0.002635418 - Q.girth2) / 0.0007941346   # -4.1%  girth2 < 0.002635
        - 0.0401987 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -4.0%  lam1 < 0.004184
        - 0.03322967 * max(0.0, 0.06473447 - Q.z_7) / 0.01631636   # -3.3%  z_7 < 0.06473
        + 0.03059664 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1) / 0.002082998   # +3.1%  zdr_0 < 0.02118 and z_dr_0p05_0p1 < 0.2919
        + 0.02580926 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # +2.6%  girth2_top3 < 0.002152
        + 0.0253991 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # +2.5%  tau1 > 0.05357
        - 0.02505182 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 6.572938 - Q.log_sum_pt) / 1.257183   # -2.5%  pt_7 < 45.75 and log_sum_pt < 6.573
        + 0.02461865 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +2.5%  e3 < 8.148e-05
        - 0.02354714 * Q.e2 / 0.02863215   # -2.4%  e2
        - 0.02333668 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -2.3%  lam1 < 0.001504
        - 0.02154157 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # -2.2%  sj3_dr_max > 0.1986
        - 0.01545788 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.planar_flow - 0.04505724) / 0.0002791469   # -1.5%  lam2 > 0.0003061 and planar_flow > 0.04506
        - 0.01293011 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -1.3%  e3 > 3.892e-05
        + 0.01245595 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +1.2%  sj3_dr_min > 0.1278
        + 0.01219436 * max(0.0, Q.sj3_pair_mass_min - 15.95929) / 1.348548   # +1.2%  sj3_pair_mass_min > 15.96
        + 0.01215762 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.002471 - Q.D2) / 1.812028   # +1.2%  pt_7 < 45.75 and D2 < 1.002
        - 0.01124797 * max(0.0, Q.mass_top5 - 45.32077) / 3.61051   # -1.1%  mass_top5 > 45.32
        - 0.01087558 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2) / 0.0002929492   # -1.1%  lam1 > 0.00733 and D2_b2 < 0.3809
        + 0.01035342 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, 1.002471 - Q.D2) / 0.009558568   # +1.0%  tau1 > 0.05357 and D2 < 1.002
        + 0.009303017 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # +0.9%  sj3_pair_mass_min > 11.05
        - 0.009279417 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -0.9%  lam2 > 0.003408
        + 0.008937087 * max(0.0, Q.mass_top5 - 22.18342) / 12.67741   # +0.9%  mass_top5 > 22.18
        + 0.007635215 * max(0.0, Q.C2_b2 - 0.009032972) / 0.001703043   # +0.8%  C2_b2 > 0.009033
        - 0.007362226 * max(0.0, Q.sj3_pair_mass_min - 11.051) * max(0.0, Q.sj3_pairmin_over_m - 0.28737) / 0.2416857   # -0.7%  sj3_pair_mass_min > 11.05 and sj3_pairmin_over_m > 0.2874
        - 0.006661403 * max(0.0, 0.02563286 - Q.M2) / 0.001872375   # -0.7%  M2 < 0.02563
        - 0.005865965 * max(0.0, 0.004918231 - Q.zdr_0) / 0.0006323791   # -0.6%  zdr_0 < 0.004918
        + 0.005784296 * Q.mean_phi / 0.01077348   # +0.6%  mean_phi
        - 0.00509269 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.3658817 - Q.z_dr_0_0p05) / 0.002463327   # -0.5%  sj3_dr_min > 0.1278 and z_dr_0_0p05 < 0.3659
        + 0.004882728 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0008259449   # +0.5%  lam1 < 0.005954 and n_pt_above_50 > 6
        - 0.004541402 * max(0.0, Q.n_pt_above_50 - 6.0) / 0.2884353   # -0.5%  n_pt_above_50 > 6
        - 0.004198934 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.2723288 - Q.sj3_mass3) / 0.0003425911   # -0.4%  lam1 > 0.00733 and sj3_mass3 < 0.2723
        - 0.003738683 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1797097) / 5.905455e-07   # -0.4%  e3 < 8.148e-05 and sj3_dr23 > 0.1797
        + 0.003543384 * max(0.0, Q.sj3_pair_mass_min - 11.051) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 3.623232   # +0.4%  sj3_pair_mass_min > 11.05 and n_dr_0p2_0p4 > 0
        + 0.003319342 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +0.3%  sj2_dr > 0.3004
        - 0.002662334 * max(0.0, Q.girth2 - 0.02530566) / 0.0002851418   # -0.3%  girth2 > 0.02531
        + 0.001812318 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.2%  sum_pt > 988.4
        - 0.001508261 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.5973755 - Q.sj3_mass2) / 0.001008755   # -0.2%  sj3_dr_min > 0.1278 and sj3_mass2 < 0.5974
        - 0.001379948 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, 31.90625 - Q.pt_6) / 0.0003664138   # -0.1%  lam2 > 0.0003061 and pt_6 < 31.91
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 33.79;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.78747 * (-0.06402332
        + 0.1686808 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +16.9%  width < 0.008678
        + 0.146043 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +14.6%  girth2 < 0.01324
        - 0.0645277 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -6.5%  e2_sq < 0.008169
        - 0.05674205 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -5.7%  lam1 < 0.008376
        - 0.0511077 * max(0.0, 0.169029 - Q.sj3_dr_max) / 0.04876366   # -5.1%  sj3_dr_max < 0.169
        + 0.04708766 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +4.7%  centroid_offset < 0.03776
        + 0.04415559 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +4.4%  z_7 > 0.01686
        - 0.04004729 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -4.0%  girth < 0.0717
        - 0.03876757 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # -3.9%  tau1 < 0.09539
        + 0.02975569 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +3.0%  sj3_dr_max < 0.1426
        - 0.02923384 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -2.9%  mass < 69.61
        + 0.02304501 * max(0.0, 0.006679471 - Q.width) / 0.00293624   # +2.3%  width < 0.006679
        - 0.0196666 * max(0.0, 0.07637363 - Q.mass_over_sum_pt) / 0.02715027   # -2.0%  mass_over_sum_pt < 0.07637
        - 0.01937187 * max(0.0, 0.004372139 - Q.girth2) / 0.00156571   # -1.9%  girth2 < 0.004372
        + 0.01910954 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +1.9%  planar_flow < 0.2534
        - 0.01827581 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -1.8%  lam1 < 0.00733
        - 0.01698861 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt) / 3.513015   # -1.7%  centroid_offset < 0.03776 and sum_pt < 901.6
        - 0.01342214 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 15.95929 - Q.sj3_pair_mass_min) / 0.05098851   # -1.3%  centroid_offset < 0.01438 and sj3_pair_mass_min < 15.96
        + 0.01211579 * max(0.0, 0.2623172 - Q.sj3_dr_max) / 0.1129006   # +1.2%  sj3_dr_max < 0.2623
        + 0.011156 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # +1.1%  max_dr < 0.1452
        - 0.01040915 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # -1.0%  C2 < 0.03579
        + 0.009765025 * max(0.0, 0.01655442 - Q.e2) / 0.003887347   # +1.0%  e2 < 0.01655
        + 0.009450278 * max(0.0, 0.09538712 - Q.tau1) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.03209574   # +0.9%  tau1 < 0.09539 and z_dr_0p05_0p1 < 0.846
        - 0.009298546 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -0.9%  sj2_dr > 0.1779
        - 0.009045665 * max(0.0, 0.02689598 - Q.girth) / 0.004330137   # -0.9%  girth < 0.0269
        - 0.008893996 * max(0.0, Q.pt_7 - 29.04219) / 7.785987   # -0.9%  pt_7 > 29.04
        + 0.007088927 * max(0.0, 1.050302e-05 - Q.e3) / 3.717236e-06   # +0.7%  e3 < 1.05e-05
        - 0.0051048 * max(0.0, 506.875 - Q.sum_pt_top5) / 34.75344   # -0.5%  sum_pt_top5 < 506.9
        + 0.004290622 * max(0.0, 49.6681 - Q.mass) / 17.06893   # +0.4%  mass < 49.67
        + 0.003751121 * max(0.0, Q.eccentricity - 0.9884745) / 0.001961982   # +0.4%  eccentricity > 0.9885
        + 0.003372652 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0007565864   # +0.3%  centroid_offset < 0.01438 and D2_b2 < 0.5327
        + 0.003249916 * max(0.0, 0.01437952 - Q.centroid_offset) / 0.004306483   # +0.3%  centroid_offset < 0.01438
        - 0.003195436 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.3%  centroid_offset > 0.0499
        - 0.00307436 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 0.0345459 - Q.absphi_0) / 0.0003803849   # -0.3%  centroid_offset < 0.03776 and absphi_0 < 0.03455
        - 0.002890248 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.pt_6 - 19.46875) / 0.1742442   # -0.3%  girth2 < 0.01324 and pt_6 > 19.47
        + 0.002873283 * max(0.0, 15.45403 - Q.mass) * max(0.0, 0.003780225 - Q.zdr_7) / 0.006870481   # +0.3%  mass < 15.45 and zdr_7 < 0.00378
        - 0.002808183 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 1.050302e-05 - Q.e3) / 1.771164e-07   # -0.3%  planar_flow < 0.2534 and e3 < 1.05e-05
        - 0.002797993 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 4.721224 - Q.D2_b2) / 16.05793   # -0.3%  pt_6 < 41.22 and D2_b2 < 4.721
        - 0.002756378 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.5502779 - Q.tau32) / 0.0001096908   # -0.3%  centroid_offset > 0.0499 and tau32 < 0.5503
        + 0.002417204 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # +0.2%  centroid_offset > 0.0499 and D2 < 2.844
        + 0.002321211 * max(0.0, 15.45403 - Q.mass) / 2.385786   # +0.2%  mass < 15.45
        - 0.002167687 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, Q.sj3_pair_mass_min - 1.282345) / 0.4025762   # -0.2%  planar_flow < 0.2534 and sj3_pair_mass_min > 1.282
        - 0.002110569 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.00279494 - Q.zdr_7) / 5.690817e-06   # -0.2%  centroid_offset < 0.01438 and zdr_7 < 0.002795
        - 0.001835372 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.2%  planar_flow < 0.2534 and sum_pt < 840
        - 0.001661905 * max(0.0, Q.eccentricity - 0.9884745) * max(0.0, 2.771069 - Q.sj2_mass2) / 0.00304207   # -0.2%  eccentricity > 0.9885 and sj2_mass2 < 2.771
        - 0.001474676 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.03601074 - Q.abseta_4) / 8.179649e-05   # -0.1%  centroid_offset < 0.01438 and abseta_4 < 0.03601
        + 0.001315905 * max(0.0, 0.003676313 - Q.zdr_0) / 0.0003550098   # +0.1%  zdr_0 < 0.003676
        + 0.001224155 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, -0.001813533 - Q.mean_phi) / 4.860641e-05   # +0.1%  centroid_offset < 0.03776 and mean_phi < -0.001814
        + 0.001222201 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.1%  sj2_dr > 0.2688
        + 0.001215602 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 0.415246 - Q.D2) / 8.563062e-05   # +0.1%  girth2 < 0.01324 and D2 < 0.4152
        - 0.00109439 * max(0.0, 0.02689598 - Q.girth) * max(0.0, Q.sj3_pair_mass_min - 1.590484) / 0.005084766   # -0.1%  girth < 0.0269 and sj3_pair_mass_min > 1.59
        - 0.001017969 * max(0.0, Q.n_dr_0p1_0p2 - 3.0) / 0.3318622   # -0.1%  n_dr_0p1_0p2 > 3
        - 0.0009853499 * max(0.0, 49.6681 - Q.mass) * max(0.0, 1.332146 - Q.D2) / 1.334408   # -0.1%  mass < 49.67 and D2 < 1.332
        - 0.0008686748 * max(0.0, 24.42188 - Q.pt_6) / 0.6046572   # -0.1%  pt_6 < 24.42
        - 0.0007470405 * max(0.0, Q.eccentricity - 0.9884745) * max(0.0, 0.02130127 - Q.eta_0) / 6.650689e-05   # -0.1%  eccentricity > 0.9885 and eta_0 < 0.0213
        - 0.0006882442 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.tau4 - 0.002207727) / 2.478129e-06   # -0.1%  centroid_offset > 0.0499 and tau4 > 0.002208
        - 0.000669758 * max(0.0, 0.008168571 - Q.e2_sq) * max(0.0, Q.mass_top2 - 6.779915) / 0.005654747   # -0.1%  e2_sq < 0.008169 and mass_top2 > 6.78
        + 0.0006551522 * max(0.0, Q.max_pair_mass - 33.3761) / 0.9279851   # +0.1%  max_pair_mass > 33.38
        - 0.000623105 * max(0.0, 69.61135 - Q.mass) * max(0.0, 0.415246 - Q.D2) / 0.1191559   # -0.1%  mass < 69.61 and D2 < 0.4152
        + 0.0002669492 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.0%  sum_pt > 988.4
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.7442;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.7442266 * (-0.563495
        + 0.4624171 * max(0.0, Q.girth2 - 0.01882765) / 0.0007605369   # +46.2%  girth2 > 0.01883
        - 0.2487907 * max(0.0, Q.mass_over_sum_pt - 0.1309286) / 0.002541185   # -24.9%  mass_over_sum_pt > 0.1309
        - 0.07145657 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, 53.4375 - Q.pt_7) / 0.01439677   # -7.1%  girth2 > 0.01883 and pt_7 < 53.44
        + 0.05630157 * max(0.0, Q.mass - 88.15578) / 0.7232262   # +5.6%  mass > 88.16
        - 0.05259032 * max(0.0, Q.girth2_top2 - 0.01403324) / 0.001129575   # -5.3%  girth2_top2 > 0.01403
        - 0.0416587 * max(0.0, 437.2453 - Q.sum_pt) / 2.379313   # -4.2%  sum_pt < 437.2
        + 0.02526003 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, Q.lam2 - 0.000537286) / 2.828867e-06   # +2.5%  girth2 > 0.01883 and lam2 > 0.0005373
        - 0.01898156 * max(0.0, Q.zdr_0 - 0.03981924) / 0.0005695785   # -1.9%  zdr_0 > 0.03982
        + 0.01576579 * max(0.0, Q.mean_phi - 0.02612796) / 0.0006999411   # +1.6%  mean_phi > 0.02613
        + 0.006777598 * max(0.0, Q.mass - 88.15578) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 1.003466   # +0.7%  mass > 88.16 and n_dr_0p2_0p4 > 1
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 20.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.24343 * (0.02326792
        + 0.3331686 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # +33.3%  girth < 0.1484
        - 0.1475284 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -14.8%  e2 < 0.08001
        - 0.08492794 * max(0.0, 0.007520088 - Q.width) / 0.003544991   # -8.5%  width < 0.00752
        + 0.07763848 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +7.8%  lam1 < 0.01643
        - 0.06350688 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -6.4%  girth < 0.1484 and log_sum_pt < 6.804
        + 0.05262153 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +5.3%  lam1 < 0.006507
        - 0.03759955 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 8.324263e-07   # -3.8%  e3 < 5.335e-05 and centroid_offset < 0.03776
        + 0.03465346 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # +3.5%  e3 < 5.335e-05
        - 0.03357358 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -3.4%  girth < 0.1484 and pt_7 < 38.53
        + 0.02319153 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7) / 781.5808   # +2.3%  sum_pt_top5 > 658.1 and pt_7 < 40.04
        + 0.01547268 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +1.5%  width < 0.01324
        - 0.01521938 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.000537286 - Q.lam2) / 4.021081e-05   # -1.5%  girth < 0.1484 and lam2 < 0.0005373
        + 0.01094576 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +1.1%  log_sum_pt > 6.503
        - 0.008901064 * max(0.0, Q.mass - 49.6681) / 7.721594   # -0.9%  mass > 49.67
        - 0.007194521 * max(0.0, 31.90625 - Q.pt_6) / 1.826854   # -0.7%  pt_6 < 31.91
        - 0.006872321 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # -0.7%  z_7 < 0.02807
        - 0.006539138 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # -0.7%  lam2 < 0.0001948
        + 0.005877665 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.z_7 - 0.06164517) / 0.0003465887   # +0.6%  girth < 0.1484 and z_7 > 0.06165
        - 0.005030375 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55592   # -0.5%  sum_pt_top5 > 658.1
        + 0.004113391 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 31.90625 - Q.pt_6) / 134.9952   # +0.4%  sum_pt_top5 > 791.1 and pt_6 < 31.91
        - 0.003025562 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.zdr_7 - 0.0007928864) / 0.0001818422   # -0.3%  log_sum_pt > 6.503 and zdr_7 > 0.0007929
        + 0.0029867 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.07474969 - Q.M3) / 0.0006773733   # +0.3%  girth < 0.1484 and M3 < 0.07475
        - 0.002838612 * max(0.0, Q.sj3_dr23 - 0.1974628) / 0.02478939   # -0.3%  sj3_dr23 > 0.1975
        + 0.002791899 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.tau2 - 0.008780509) / 0.000269945   # +0.3%  girth < 0.1484 and tau2 > 0.008781
        - 0.002439153 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7) / 0.06765006   # -0.2%  pt_6 < 31.91 and z_7 < 0.05861
        + 0.002398937 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +0.2%  sum_pt_top5 > 791.1
        - 0.002212425 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.2%  z_6 < 0.0216
        - 0.001751896 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # -0.2%  z_5 < 0.02818
        - 0.001511178 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6) / 104.494   # -0.2%  sum_pt > 988.4 and pt_6 < 62.25
        + 0.001229863 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.pair_mass_0_7 - 10.2219) / 0.2907061   # +0.1%  log_sum_pt > 6.503 and pair_mass_0_7 > 10.22
        - 0.001132571 * max(0.0, Q.pt_7 - 48.71875) / 0.6759439   # -0.1%  pt_7 > 48.72
        - 0.000683047 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.sj2_mass1 - 31.78116) / 0.01215768   # -0.1%  girth < 0.1484 and sj2_mass1 > 31.78
        - 0.0004219462 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.08151794 - Q.M3) / 0.08390625   # -0.0%  sum_pt > 988.4 and M3 < 0.08152
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 38.45;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 38.44545 * (-0.0169653
        - 0.09577321 * max(0.0, Q.width - 0.006679471) / 0.002702003   # -9.6%  width > 0.006679
        - 0.07847471 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -7.8%  girth2 > 0.00752
        + 0.06029489 * max(0.0, Q.girth - 0.02689598) / 0.03613842   # +6.0%  girth > 0.0269
        + 0.04939724 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # +4.9%  sj2_dr > 0.1592
        + 0.04591468 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +4.6%  mass_over_sum_pt > 0.08475
        + 0.04496668 * max(0.0, Q.width - 0.003562611) / 0.004066872   # +4.5%  width > 0.003563
        - 0.0448305 * max(0.0, 0.324646 - Q.sd_rg) / 0.2082097   # -4.5%  sd_rg < 0.3246
        + 0.04436255 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +4.4%  mass_over_sum_pt > 0.07992
        - 0.03826798 * max(0.0, Q.e2 - 0.01655442) / 0.01596508   # -3.8%  e2 > 0.01655
        + 0.03602574 * max(0.0, Q.e2_sq - 0.002074109) / 0.004484017   # +3.6%  e2_sq > 0.002074
        - 0.0332744 * max(0.0, Q.e2_sq - 0.0030133) / 0.003940214   # -3.3%  e2_sq > 0.003013
        + 0.03030412 * max(0.0, Q.girth - 0.04081947) / 0.0270611   # +3.0%  girth > 0.04082
        + 0.02832831 * max(0.0, 49.91626 - Q.sd_mass) / 22.60916   # +2.8%  sd_mass < 49.92
        - 0.02540712 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -2.5%  girth > 0.08724
        + 0.02454244 * max(0.0, Q.e2_sq - 0.01165737) / 0.001433269   # +2.5%  e2_sq > 0.01166
        - 0.02296541 * max(0.0, Q.sj2_dr - 0.2001708) / 0.02317349   # -2.3%  sj2_dr > 0.2002
        - 0.01591617 * max(0.0, Q.sj2_dr - 0.06154135) / 0.1002688   # -1.6%  sj2_dr > 0.06154
        - 0.0153695 * max(0.0, Q.girth2 - 0.0009641429) / 0.00568632   # -1.5%  girth2 > 0.0009641
        - 0.01397982 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -1.4%  sj2_dr > 0.1779
        - 0.01392522 * max(0.0, 0.004032342 - Q.C2_b2) / 0.002883542   # -1.4%  C2_b2 < 0.004032
        + 0.01333395 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # +1.3%  lam2 < 0.0005373
        - 0.01311954 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -1.3%  mass_over_sum_pt > 0.09041
        + 0.01239532 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +1.2%  e2 > 0.04111
        - 0.01185518 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -1.2%  e3 < 8.148e-05
        + 0.01047025 * max(0.0, 0.1115136 - Q.planar_flow) / 0.03343187   # +1.0%  planar_flow < 0.1115
        - 0.009388307 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.9%  psi_0p1 > 0.9761
        + 0.009245666 * max(0.0, 0.07870506 - Q.z_dr_0p1_0p2) / 0.04317253   # +0.9%  z_dr_0p1_0p2 < 0.07871
        - 0.009122667 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.9%  centroid_offset > 0.0499
        + 0.008738089 * max(0.0, Q.z_dr_0_0p05 - 0.1515405) / 0.4480651   # +0.9%  z_dr_0_0p05 > 0.1515
        - 0.008608466 * max(0.0, 74.57663 - Q.sd_mass) / 42.04056   # -0.9%  sd_mass < 74.58
        + 0.008281513 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +0.8%  LHA > 0.3033
        - 0.008173559 * max(0.0, 715.4688 - Q.sum_pt) / 69.91196   # -0.8%  sum_pt < 715.5
        - 0.00795479 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -0.8%  planar_flow < 0.1115 and lam1 < 0.00733
        + 0.007383885 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0006014625   # +0.7%  sj2_dr > 0.1592 and D2_b2 < 0.08499
        + 0.006100193 * max(0.0, 5.0 - Q.n_dr_0_0p05) / 1.94322   # +0.6%  n_dr_0_0p05 < 5
        - 0.005869905 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.6%  centroid_offset > 0.03776
        - 0.005775433 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # -0.6%  e3 < 5.335e-05
        - 0.00551572 * max(0.0, Q.sj2_dr - 0.1294903) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0009393137   # -0.6%  sj2_dr > 0.1295 and D2_b2 < 0.08499
        - 0.005221426 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 0.02415398 - Q.C2_b2) / 0.0005249308   # -0.5%  z_dr_0p05_0p1 > 0.7509 and C2_b2 < 0.02415
        - 0.004993602 * max(0.0, Q.centroid_offset - 0.02685622) / 0.003254849   # -0.5%  centroid_offset > 0.02686
        + 0.00474888 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # +0.5%  girth2 > 0.008678
        + 0.004431227 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067297   # +0.4%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        + 0.004253278 * max(0.0, Q.psi_0p1 - 0.7143804) / 0.1805043   # +0.4%  psi_0p1 > 0.7144
        - 0.0042229 * max(0.0, 0.163898 - Q.z_dr_0p05_0p1) / 0.08682106   # -0.4%  z_dr_0p05_0p1 < 0.1639
        - 0.004141166 * max(0.0, Q.sd_rg - 0.2330919) / 0.01070351   # -0.4%  sd_rg > 0.2331
        + 0.003726763 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.width) / 3.351526e-05   # +0.4%  planar_flow < 0.1115 and width < 0.006097
        + 0.003644228 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # +0.4%  sd_rg > 0.2788
        - 0.00295153 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # -0.3%  sj2_dr > 0.1295
        + 0.002900914 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, 150.625 - Q.pt_1) / 0.1837865   # +0.3%  centroid_offset > 0.02686 and pt_1 < 150.6
        + 0.002712574 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +0.3%  e2 > 0.05028
        - 0.002656933 * max(0.0, Q.mass - 69.61135) / 2.406702   # -0.3%  mass > 69.61
        - 0.002567954 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0003170016   # -0.3%  sj2_dr > 0.2002 and D2_b2 < 0.08499
        - 0.00249705 * max(0.0, Q.width - 0.005590289) / 0.003084256   # -0.2%  width > 0.00559
        - 0.002430133 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.2%  planar_flow < 0.1115 and centroid_offset < 0.01838
        + 0.002400295 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.0008333816 - Q.C2_b2) / 1.341084e-07   # +0.2%  centroid_offset > 0.0499 and C2_b2 < 0.0008334
        - 0.002292205 * max(0.0, Q.width - 0.01323868) / 0.001442684   # -0.2%  width > 0.01324
        + 0.001965503 * max(0.0, Q.mass_top5 - 49.18618) / 2.747087   # +0.2%  mass_top5 > 49.19
        + 0.001886383 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.05056028 - Q.dr_2) / 0.0001260892   # +0.2%  sj2_dr > 0.2002 and dr_2 < 0.05056
        + 0.001741842 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 715.4688 - Q.sum_pt) / 22.02426   # +0.2%  z_dr_0p05_0p1 < 0.5883 and sum_pt < 715.5
        + 0.001583151 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1629004) / 4.043869e-07   # +0.2%  e3 < 5.335e-05 and sj3_dr23 > 0.1629
        + 0.001407345 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.3724174   # +0.1%  z_dr_0p05_0p1 < 0.5883
        - 0.001177123 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 658.125 - Q.sum_pt_top5) / 2.950014   # -0.1%  planar_flow < 0.1115 and sum_pt_top5 < 658.1
        - 0.0009484984 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.03293672 - Q.dr_2) / 6.256053e-05   # -0.1%  sj2_dr > 0.1592 and dr_2 < 0.03294
        + 0.0007953501 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.05268713 - Q.dr_3) / 0.0001291671   # +0.1%  sj2_dr > 0.2002 and dr_3 < 0.05269
        - 0.0007131207 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 0.002231562   # -0.1%  planar_flow < 0.1115 and z_dr_0p05_0p1 > 0.6748
        + 0.0006234176 * max(0.0, Q.psi_0p1 - 0.8155839) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0006308268   # +0.1%  psi_0p1 > 0.8156 and D2_b2 < 0.08499
        - 0.0004462898 * max(0.0, Q.sd_rg - 0.2787955) * max(0.0, 0.2669656 - Q.D2_b2) / 0.0004100902   # -0.0%  sd_rg > 0.2788 and D2_b2 < 0.267
        - 0.0002658046 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.02358801 - Q.dr_3) / 2.035984e-05   # -0.0%  sj2_dr > 0.1592 and dr_3 < 0.02359
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 47.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 47.56797 * (-0.03410314
        + 0.09740928 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +9.7%  width < 0.01324
        + 0.08122336 * max(0.0, 0.01882765 - Q.width) * max(0.0, 0.003408389 - Q.lam2) / 4.313596e-05   # +8.1%  width < 0.01883 and lam2 < 0.003408
        - 0.07235054 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # -7.2%  e2_sq < 0.01166
        + 0.0649057 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +6.5%  e2 < 0.04111
        + 0.05443464 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +5.4%  sj2_dr < 0.1592
        - 0.05108405 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # -5.1%  girth < 0.1019
        - 0.05030239 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -5.0%  e2_sq < 0.008169
        + 0.0423708 * max(0.0, 0.01882765 - Q.width) / 0.01314296   # +4.2%  width < 0.01883
        - 0.04148433 * max(0.0, 0.06344108 - Q.e2) / 0.03657078   # -4.1%  e2 < 0.06344
        - 0.03936176 * max(0.0, 0.007520088 - Q.girth2) / 0.00354499   # -3.9%  girth2 < 0.00752
        - 0.03781454 * max(0.0, 0.00609665 - Q.width) / 0.002544039   # -3.8%  width < 0.006097
        - 0.03535023 * max(0.0, 0.2001708 - Q.sj2_dr) / 0.07331403   # -3.5%  sj2_dr < 0.2002
        - 0.03312393 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # -3.3%  mass_over_sum_pt < 0.1309
        - 0.03251256 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -3.3%  girth < 0.08724
        + 0.02824311 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +2.8%  mass_over_sum_pt_sq < 0.007183
        - 0.02492641 * max(0.0, 0.1492731 - Q.sj2_dr) / 0.04374278   # -2.5%  sj2_dr < 0.1493
        + 0.02415286 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # +2.4%  lam1 < 0.005433
        + 0.02252456 * max(0.0, 0.06345984 - Q.tau1) / 0.02211428   # +2.3%  tau1 < 0.06346
        + 0.01442988 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +1.4%  lam2 < 0.001131
        - 0.01394874 * max(0.0, 0.03846696 - Q.e2) / 0.01567344   # -1.4%  e2 < 0.03847
        + 0.01266076 * max(0.0, 0.007639643 - Q.girth2_top2) / 0.004543329   # +1.3%  girth2_top2 < 0.00764
        + 0.009927355 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +1.0%  lam1 < 0.008376
        - 0.009564737 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # -1.0%  N2 < 0.2233
        - 0.008849914 * max(0.0, 0.007520088 - Q.girth2) * max(0.0, 0.7459513 - Q.D2) / 9.127003e-05   # -0.9%  girth2 < 0.00752 and D2 < 0.746
        + 0.00715277 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +0.7%  N2 < 0.2233 and sum_pt_top5 > 430.8
        - 0.007132183 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # -0.7%  z_dr_0p05_0p1 > 0.7509
        + 0.006705745 * max(0.0, Q.mass - 45.7571) / 9.387753   # +0.7%  mass > 45.76
        - 0.00651644 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.01272888 - Q.tau21_b2) / 7.280144e-05   # -0.7%  mass_over_sum_pt < 0.1309 and tau21_b2 < 0.01273
        + 0.006482689 * max(0.0, 0.01323868 - Q.width) * max(0.0, 0.01272888 - Q.tau21_b2) / 8.968194e-06   # +0.6%  width < 0.01324 and tau21_b2 < 0.01273
        + 0.006465212 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +0.6%  lam1 < 0.002464
        + 0.005644427 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.04262929   # +0.6%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 2
        - 0.005436997 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2512159 - Q.dr01) / 0.001381924   # -0.5%  centroid_offset < 0.01838 and dr01 < 0.2512
        - 0.00516443 * max(0.0, 0.0005124533 - Q.girth2_top2) / 0.0001104529   # -0.5%  girth2_top2 < 0.0005125
        - 0.00466803 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # -0.5%  girth2_top3 < 0.002152
        + 0.004336236 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) * max(0.0, 0.7459513 - Q.D2) / 8.253816e-05   # +0.4%  mass_over_sum_pt_sq < 0.007183 and D2 < 0.746
        + 0.004110787 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 716.8828 - Q.sum_pt_top5) / 7.315623   # +0.4%  N2 < 0.2233 and sum_pt_top5 < 716.9
        + 0.003308949 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +0.3%  lam1 < 0.00484
        - 0.003210884 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, Q.n_for_50pct - 1.0) / 0.001375293   # -0.3%  lam2 < 0.001131 and n_for_50pct > 1
        - 0.00309562 * max(0.0, 0.004007842 - Q.girth2_top2) / 0.001907211   # -0.3%  girth2_top2 < 0.004008
        + 0.002314696 * max(0.0, 0.006756161 - Q.girth2_top3) / 0.003650633   # +0.2%  girth2_top3 < 0.006756
        + 0.002260397 * max(0.0, 0.00609665 - Q.width) * max(0.0, 0.7459513 - Q.D2) / 3.18925e-05   # +0.2%  width < 0.006097 and D2 < 0.746
        - 0.002112663 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.2%  mass > 76.66
        + 0.00205325 * max(0.0, 0.01165737 - Q.e2_sq) * max(0.0, Q.lam2 - 6.531723e-06) / 7.375237e-07   # +0.2%  e2_sq < 0.01166 and lam2 > 6.532e-06
        - 0.00188499 * max(0.0, 0.0262518 - Q.tau1) * max(0.0, Q.zdr_3 - 0.008191788) / 3.916286e-08   # -0.2%  tau1 < 0.02625 and zdr_3 > 0.008192
        + 0.001558737 * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 0.03639977   # +0.2%  z_dr_0p05_0p1 > 0.6748
        + 0.001476282 * max(0.0, 0.0003061234 - Q.lam2) * max(0.0, 0.08499387 - Q.D2_b2) / 2.928603e-06   # +0.1%  lam2 < 0.0003061 and D2_b2 < 0.08499
        + 0.001456917 * max(0.0, 0.008355823 - Q.dr_0) / 0.0005818924   # +0.1%  dr_0 < 0.008356
        - 0.001055626 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.0114178   # -0.1%  N2 < 0.2233 and z_dr_0p05_0p1 < 0.5883
        - 0.0005540566 * max(0.0, 0.00609665 - Q.width) * max(0.0, 0.08499387 - Q.D2_b2) / 3.927041e-06   # -0.1%  width < 0.006097 and D2_b2 < 0.08499
        - 0.0003210778 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 6.0 - Q.n_for_90pct) / 0.005228592   # -0.0%  N2 < 0.2233 and n_for_90pct < 6
        + 0.0002406222 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.4684459) / 0.0001955483   # +0.0%  mass_over_sum_pt < 0.1309 and z_dr_0p1_0p2 > 0.4684
        - 0.0002121751 * max(0.0, Q.mass - 45.7571) * max(0.0, 23.21641 - Q.pt_7) / 5.139381   # -0.0%  mass > 45.76 and pt_7 < 23.22
        - 9.744267e-05 * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) * max(0.0, Q.pair_mass_0_6 - 24.79428) / 0.00796161   # -0.0%  z_dr_0p05_0p1 > 0.6748 and pair_mass_0_6 > 24.79
        - 1.323592e-05 * max(0.0, Q.sj3_pair_mass_max - 72.58129) * max(0.0, 0.01039257 - Q.dr_3) / 1.53327e-06   # -0.0%  sj3_pair_mass_max > 72.58 and dr_3 < 0.01039
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.020131512605042, 0.7224735294117647, 2.0816745798319327, 0.9452512605042017, 1.125544537815126, 1.7507569327731092, 0.9339023109243697, 1.8262569327731093, 0.2416903361344538, 2.9836453781512606, 2.0131336134453783, 2.1291556722689076, 0.09457478991596639, 3.290012605042017, 0.49694327731092436, 0.34824936974789916]
T = [2.3144811236213236, 1.3504034631039916, 3.3327641018907563, 2.4575022485556723, 2.837497117909664]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +39%, n9 +22%, n5 -14%, n1 +12%, n0 -7%, n6 +4% ...
            + 0.3864666 * h[2] / H_AVG[2]
            + 0.2215676 * h[9] / H_AVG[9]
            - 0.1418318 * h[5] / H_AVG[5]
            + 0.121935 * h[1] / H_AVG[1]
            - 0.0688688 * h[0] / H_AVG[0]
            + 0.04413325 * h[6] / H_AVG[6]
            - 0.01519704 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -19%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5609925 * h[9] / H_AVG[9]
            - 0.1863456 * h[10] / H_AVG[10]
            + 0.0864466 * h[6] / H_AVG[6]
            - 0.07813946 * h[4] / H_AVG[4]
            + 0.06077201 * h[5] / H_AVG[5]
            + 0.01611784 * h[15] / H_AVG[15]
            + 0.01118602 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +12%, n14 -11%, n0 +11%, n6 -9% ...
            + 0.2395709 * h[11] / H_AVG[11]
            - 0.1418119 * h[3] / H_AVG[3]
            + 0.1198686 * h[7] / H_AVG[7]
            - 0.1118313 * h[14] / H_AVG[14]
            + 0.105219 * h[0] / H_AVG[0]
            - 0.0875683 * h[6] / H_AVG[6]
            - 0.0718387 * h[15] / H_AVG[15]
            + 0.06941056 * h[13] / H_AVG[13]
            - 0.02797645 * h[9] / H_AVG[9]
            - 0.01812987 * h[8] / H_AVG[8]
            - 0.006774346 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -22%, n6 -14%, n14 +8%, n13 +7%, n9 -4% ...
            + 0.3483447 * h[7] / H_AVG[7]
            - 0.2163594 * h[3] / H_AVG[3]
            - 0.1425079 * h[6] / H_AVG[6]
            + 0.07583054 * h[14] / H_AVG[14]
            + 0.07321359 * h[13] / H_AVG[13]
            - 0.03794052 * h[9] / H_AVG[9]
            + 0.03674837 * h[1] / H_AVG[1]
            + 0.03578152 * h[4] / H_AVG[4]
            - 0.02214198 * h[15] / H_AVG[15]
            + 0.01113146 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +27%, n5 -15%, n4 +5%, n3 +2%, n12 -2% ...
            - 0.4710375 * h[13] / H_AVG[13]
            + 0.2660532 * h[10] / H_AVG[10]
            - 0.1542519 * h[5] / H_AVG[5]
            + 0.04958351 * h[4] / H_AVG[4]
            + 0.02082053 * h[3] / H_AVG[3]
            - 0.01666518 * h[12] / H_AVG[12]
            + 0.01597074 * h[8] / H_AVG[8]
            + 0.00561747 * h[0] / H_AVG[0]
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
