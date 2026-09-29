"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  9:  11.8%   (on for 71% of jets)
  neuron  7:  10.3%   (on for 62% of jets)
  neuron  3:   8.6%   (on for 32% of jets)
  neuron 10:   8.2%   (on for 80% of jets)
  neuron  2:   7.3%   (on for 87% of jets)
  neuron  5:   7.1%   (on for 62% of jets)
  neuron  6:   7.0%   (on for 50% of jets)
  neuron 11:   6.5%   (on for 73% of jets)
  neuron 14:   4.5%   (on for 30% of jets)
  neuron  0:   4.3%   (on for 43% of jets)
  neuron  1:   3.2%   (on for 56% of jets)
  neuron  4:   3.0%   (on for 43% of jets)
  neuron 15:   2.5%   (on for 27% of jets)
  neuron  8:   1.0%   (on for 33% of jets)
  neuron 12:   0.4%   (on for 8% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.0% (the network: 65.8%); same class as the network for 88.6% of jets.

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
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.ptdr0_7                pT7 · ΔR(0, 7) [GeV]
  Q.z_2                    pT of particle 2 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_nremoved            number of branches removed by soft drop
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_2               |Δη| of particle 2
  Q.abseta_3               |Δη| of particle 3
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_4                  ΔR between particle 4 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.phi_0                  Δφ of particle 0
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
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        pt_1=pt[1],
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        pt_4=pt[4],
        pt_5=pt[5],
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        ptdr0_7=pt[7] * math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        z_2=z[2],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        z_top3_slots=sum(pt[:3]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_nremoved=softdrop("removed"),
        abseta_0=abs(eta[0]),
        abseta_2=abs(eta[2]),
        abseta_3=abs(eta[3]),
        abseta_4=abs(eta[4]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_4=math.sqrt(dist2(1, 4)) if pt[4] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        eta_0=eta[0],
        phi_0=phi[0],
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
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 20.2;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.20314 * (-0.09079081
        + 0.1654866 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +16.5%  e3 < 0.0005117
        + 0.1008388 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +10.1%  lam1 < 0.012
        - 0.06510098 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # -6.5%  sj3_dr_max < 0.3456
        - 0.06179663 * max(0.0, 0.08065885 - Q.girth) / 0.03128522   # -6.2%  girth < 0.08066
        - 0.05655317 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -5.7%  lam1 < 0.004184
        + 0.04990329 * max(0.0, Q.sj3_dr_max - 0.1426152) / 0.06279029   # +5.0%  sj3_dr_max > 0.1426
        - 0.04700043 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # -4.7%  sj3_dr_max > 0.169
        + 0.04513435 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +4.5%  lam1 < 0.01643
        - 0.03232023 * max(0.0, 763.825 - Q.sum_pt) / 96.68366   # -3.2%  sum_pt < 763.8
        + 0.02733904 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +2.7%  e3 < 8.148e-05
        + 0.0239126 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +2.4%  tau1 < 0.05357
        - 0.02364639 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -2.4%  centroid_offset > 0.00679
        - 0.02274699 * max(0.0, 0.001563465 - Q.C2_b2) / 0.0008800039   # -2.3%  C2_b2 < 0.001563
        - 0.02195141 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.002399898   # -2.2%  lam1 < 0.005954 and n_dr_0p2_0p4 < 1
        + 0.02190842 * max(0.0, Q.LHA - 0.1546891) / 0.1006902   # +2.2%  LHA > 0.1547
        - 0.0214539 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -2.1%  lam1 < 0.005954
        + 0.02115866 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +2.1%  planar_flow < 0.1484
        - 0.01999173 * max(0.0, 0.01713288 - Q.tau2) / 0.008065871   # -2.0%  tau2 < 0.01713
        - 0.01571307 * max(0.0, 0.007929074 - Q.girth2_top3) / 0.004560262   # -1.6%  girth2_top3 < 0.007929
        - 0.01386984 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # -1.4%  sj3_dr_max > 0.1986
        + 0.01232347 * max(0.0, 0.004183811 - Q.lam1) * max(0.0, 763.825 - Q.sum_pt) / 0.08579672   # +1.2%  lam1 < 0.004184 and sum_pt < 763.8
        + 0.01085529 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, Q.centroid_offset - 0.01096064) / 1.576925   # +1.1%  sum_pt < 763.8 and centroid_offset > 0.01096
        + 0.01018767 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +1.0%  sum_pt_top5 > 791.1
        - 0.01015091 * max(0.0, Q.log_sum_pt - 6.766778) / 0.01900249   # -1.0%  log_sum_pt > 6.767
        + 0.01005319 * max(0.0, 0.04505724 - Q.planar_flow) * max(0.0, 0.05932655 - Q.tau21_b2) / 0.0003618271   # +1.0%  planar_flow < 0.04506 and tau21_b2 < 0.05933
        - 0.009664228 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.0005116989 - Q.e3) / 0.03717154   # -1.0%  sum_pt < 763.8 and e3 < 0.0005117
        + 0.009378974 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, Q.z_7 - 0.01685855) / 5.053524   # +0.9%  sum_pt < 763.8 and z_7 > 0.01686
        - 0.009332415 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.centroid_offset - 0.01837778) / 3.28605e-05   # -0.9%  lam1 < 0.01643 and centroid_offset > 0.01838
        - 0.008116464 * max(0.0, 0.04505724 - Q.planar_flow) / 0.007613167   # -0.8%  planar_flow < 0.04506
        + 0.007803401 * max(0.0, 0.01713288 - Q.tau2) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1) / 0.001574598   # +0.8%  tau2 < 0.01713 and z_dr_0p05_0p1 < 0.2919
        + 0.00771505 * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.8069227   # +0.8%  n_dr_0p2_0p4 < 1
        + 0.005934678 * max(0.0, 0.02045966 - Q.e2) / 0.005531925   # +0.6%  e2 < 0.02046
        - 0.004746654 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # -0.5%  log_sum_pt > 6.843
        - 0.004600736 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -0.5%  e2 > 0.06344
        - 0.004520932 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -0.5%  LHA > 0.3033
        - 0.002461905 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.380911 - Q.D2_b2) / 10.72478   # -0.2%  sum_pt < 763.8 and D2_b2 < 0.3809
        - 0.002160743 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 0.05744392 - Q.D2_b2) / 3.145076e-07   # -0.2%  e3 < 8.148e-05 and D2_b2 < 0.05744
        - 0.002138215 * max(0.0, 0.007929074 - Q.girth2_top3) * max(0.0, 2.482534e-05 - Q.phi_0) / 3.764038e-05   # -0.2%  girth2_top3 < 0.007929 and phi_0 < 2.483e-05
        - 0.002050786 * max(0.0, 27.57812 - Q.pt_6) / 0.9875435   # -0.2%  pt_6 < 27.58
        - 0.001333982 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 0.7459513 - Q.D2) / 2.95026e-05   # -0.1%  lam1 < 0.005954 and D2 < 0.746
        - 0.001282244 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, Q.C3 - 0.04857476) / 0.0001233564   # -0.1%  planar_flow < 0.1484 and C3 > 0.04857
        + 0.001226363 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 0.04037476 - Q.eta_0) / 0.5936549   # +0.1%  sum_pt_top5 > 791.1 and eta_0 < 0.04037
        + 0.001207588 * max(0.0, Q.sj3_dr_max - 0.1426152) * max(0.0, Q.D2 - 0.2568137) / 0.08032253   # +0.1%  sj3_dr_max > 0.1426 and D2 > 0.2568
        + 0.0009075953 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.02807091 - Q.z_7) / 4.425646e-05   # +0.1%  planar_flow < 0.1484 and z_7 < 0.02807
        + 0.0007143141 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.03885671 - Q.D2_b2) / 0.1899852   # +0.1%  sum_pt < 763.8 and D2_b2 < 0.03886
        - 0.0005538479 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.02358801 - Q.dr_3) / 2.846286e-05   # -0.1%  planar_flow < 0.1484 and dr_3 < 0.02359
        + 0.0004701834 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.006597523   # +0.0%  log_sum_pt > 6.843 and n_pt_above_50 > 5
        - 0.0002816618 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.04019753 - Q.tau21_b2) / 5.609094e-05   # -0.0%  log_sum_pt > 6.843 and tau21_b2 < 0.0402
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 17.34;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.34215 * (0.1396586
        - 0.09196634 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.001130645 - Q.lam2) / 4.482383e-06   # -9.2%  lam1 < 0.008376 and lam2 < 0.001131
        - 0.07289544 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # -7.3%  girth < 0.1019
        + 0.06969343 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +7.0%  log_sum_pt > 6.378
        - 0.04946916 * max(0.0, 0.06473447 - Q.z_7) / 0.01631636   # -4.9%  z_7 < 0.06473
        - 0.0453887 * max(0.0, 0.08670959 - Q.z_7) / 0.03486957   # -4.5%  z_7 < 0.08671
        + 0.03893542 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +3.9%  lam2 < 0.001131
        - 0.03746523 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.006756161 - Q.girth2_top3) / 0.0004453888   # -3.7%  log_sum_pt > 6.573 and girth2_top3 < 0.006756
        + 0.0347057 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # +3.5%  tau1 < 0.1028
        - 0.03360078 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # -3.4%  sj3_dr_max < 0.3456
        + 0.03171523 * max(0.0, 0.06108601 - Q.girth) / 0.01868685   # +3.2%  girth < 0.06109
        + 0.03015094 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +3.0%  lam1 < 0.012
        + 0.02748464 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, 0.007929074 - Q.girth2_top3) / 9.283431e-05   # +2.7%  z_7 < 0.06473 and girth2_top3 < 0.007929
        - 0.02685661 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -2.7%  lam1 < 0.008376
        + 0.02607743 * max(0.0, 0.07608178 - Q.girth) / 0.02796784   # +2.6%  girth < 0.07608
        + 0.02499049 * max(0.0, 0.169029 - Q.sj3_dr_max) / 0.04876366   # +2.5%  sj3_dr_max < 0.169
        + 0.02277731 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, 0.0005116989 - Q.e3) / 7.903391e-06   # +2.3%  z_7 < 0.06473 and e3 < 0.0005117
        - 0.02216284 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.004007842 - Q.girth2_top2) / 0.0005390217   # -2.2%  log_sum_pt > 6.378 and girth2_top2 < 0.004008
        - 0.01764584 * max(0.0, 0.04737278 - Q.z_6) / 0.004344387   # -1.8%  z_6 < 0.04737
        + 0.01549665 * max(0.0, Q.log_sum_pt - 6.572938) / 0.08631512   # +1.5%  log_sum_pt > 6.573
        + 0.01484699 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.1206357 - Q.dr_max_012) / 0.007392713   # +1.5%  log_sum_pt > 6.573 and dr_max_012 < 0.1206
        - 0.01440244 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.1147987 - Q.dr_2) / 0.01592659   # -1.4%  log_sum_pt > 6.378 and dr_2 < 0.1148
        - 0.01428127 * max(0.0, Q.z_7 - 0.04939969) / 0.01023002   # -1.4%  z_7 > 0.0494
        + 0.01348151 * max(0.0, Q.sum_pt - 763.825) / 48.90223   # +1.3%  sum_pt > 763.8
        - 0.0130043 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -1.3%  lam1 < 0.005433
        - 0.01283235 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 8.147744e-05 - Q.e3) / 0.0002667627   # -1.3%  pt_7 > 34.53 and e3 < 8.148e-05
        + 0.01262329 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +1.3%  pt_7 > 34.53
        - 0.01228019 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -1.2%  centroid_offset < 0.02077
        - 0.01224854 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # -1.2%  max_dr < 0.1118
        - 0.01057091 * max(0.0, 0.1950135 - Q.planar_flow) / 0.07573236   # -1.1%  planar_flow < 0.195
        - 0.01024223 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.00233503   # -1.0%  lam1 < 0.012 and n_pt_above_50 > 6
        + 0.010029 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +1.0%  tau1 < 0.07284
        + 0.009179646 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +0.9%  sj2_dr < 0.1592
        + 0.008687939 * max(0.0, 0.04737278 - Q.z_6) * max(0.0, 0.004007842 - Q.girth2_top2) / 1.318326e-05   # +0.9%  z_6 < 0.04737 and girth2_top2 < 0.004008
        - 0.008631049 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 0.1550922 - Q.dr_max_012) / 0.0002929109   # -0.9%  lam1 < 0.005433 and dr_max_012 < 0.1551
        + 0.00788817 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.716559 - Q.D2_b2) / 0.05721253   # +0.8%  log_sum_pt > 6.378 and D2_b2 < 0.7166
        + 0.007569196 * max(0.0, Q.pt_7 - 41.65625) / 1.850204   # +0.8%  pt_7 > 41.66
        - 0.007567822 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, 0.716559 - Q.D2_b2) / 0.004098393   # -0.8%  z_7 < 0.06473 and D2_b2 < 0.7166
        + 0.006615423 * max(0.0, Q.n_pt_above_50 - 6.0) * max(0.0, 8.147744e-05 - Q.e3) / 1.821055e-05   # +0.7%  n_pt_above_50 > 6 and e3 < 8.148e-05
        - 0.006427832 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -0.6%  zdr_0 < 0.02118
        - 0.006368427 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.dr_0 - 0.005517012) / 0.0001856778   # -0.6%  lam1 < 0.012 and dr_0 > 0.005517
        - 0.005904739 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.0155217 - Q.zdr_5) / 0.002604826   # -0.6%  log_sum_pt > 6.378 and zdr_5 < 0.01552
        + 0.005644361 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # +0.6%  lam1 < 0.008376 and centroid_offset > 0.02077
        + 0.005198763 * max(0.0, 0.04737278 - Q.z_6) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0003931974   # +0.5%  z_6 < 0.04737 and z_dr_0p2_0p4 < 0.101
        - 0.00518312 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.01058212 - Q.zdr_6) / 0.001664913   # -0.5%  log_sum_pt > 6.378 and zdr_6 < 0.01058
        + 0.004855855 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # +0.5%  pt_7 > 34.53 and tau2 < 0.01713
        + 0.004758923 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # +0.5%  sum_pt_top5 > 840
        - 0.004266677 * max(0.0, Q.sj3_dr_min - 0.02913153) / 0.02609727   # -0.4%  sj3_dr_min > 0.02913
        + 0.003370276 * max(0.0, Q.n_pt_above_50 - 6.0) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.5428739   # +0.3%  n_pt_above_50 > 6 and n_dr_0p2_0p4 < 2
        + 0.003155031 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 1.340118e-05 - Q.e3) / 1.490664e-06   # +0.3%  log_sum_pt > 6.378 and e3 < 1.34e-05
        - 0.002970705 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.3%  sum_pt > 988.4
        - 0.002928735 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 2.955458e-05 - Q.e3) / 7.137805e-05   # -0.3%  pt_7 > 34.53 and e3 < 2.955e-05
        - 0.002795839 * max(0.0, Q.sum_pt_top5 - 839.9547) * max(0.0, 3.892127e-05 - Q.e3) / 0.0003012957   # -0.3%  sum_pt_top5 > 840 and e3 < 3.892e-05
        - 0.00264657 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.1212677 - Q.dr_3) / 0.3088448   # -0.3%  pt_7 > 34.53 and dr_3 < 0.1213
        + 0.002175796 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 5.0 - Q.n_dr_0_0p05) / 8.652865   # +0.2%  pt_7 > 34.53 and n_dr_0_0p05 < 5
        - 0.001923676 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.2%  pt_7 > 53.44
        + 0.001667108 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.0095303 - Q.girth2_top2) / 0.03873367   # +0.2%  sum_pt > 988.4 and girth2_top2 < 0.00953
        - 0.001297144 * max(0.0, Q.sum_pt_top5 - 839.9547) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 15.08352   # -0.1%  sum_pt_top5 > 840 and n_dr_0p2_0p4 < 2
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 21.32;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.3186 * (0.3253482
        - 0.1553123 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # -15.5%  z_7 > 0.02321
        - 0.1019329 * max(0.0, Q.log_sum_pt - 6.080494) / 0.4685351   # -10.2%  log_sum_pt > 6.08
        + 0.0976931 * max(0.0, Q.pt_7 - 20.125) / 15.07003   # +9.8%  pt_7 > 20.12
        - 0.08010906 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -8.0%  pt_7 < 53.44
        - 0.05591601 * max(0.0, Q.LHA - 0.111565) / 0.1352185   # -5.6%  LHA > 0.1116
        + 0.03831809 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.2507612 - Q.max_dr) / 0.00062784   # +3.8%  lam1 < 0.00733 and max_dr < 0.2508
        + 0.03809258 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +3.8%  lam1 < 0.012
        + 0.0379235 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # +3.8%  log_sum_pt < 6.606
        + 0.03261181 * max(0.0, Q.z_7 - 0.02807091) / 0.02541111   # +3.3%  z_7 > 0.02807
        - 0.03253519 * max(0.0, Q.log_sum_pt - 6.080494) * max(0.0, 8.836697e-05 - Q.mean_phi2) / 6.819601e-06   # -3.3%  log_sum_pt > 6.08 and mean_phi2 < 8.837e-05
        + 0.03106202 * max(0.0, 8.836697e-05 - Q.mean_phi2) / 1.02303e-05   # +3.1%  mean_phi2 < 8.837e-05
        - 0.02827148 * Q.pt_7 / 34.64819   # -2.8%  pt_7
        - 0.02728904 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -2.7%  e2 < 0.08001
        - 0.02007424 * max(0.0, 752.1 - Q.sum_pt_top5) / 179.9248   # -2.0%  sum_pt_top5 < 752.1
        + 0.0183321 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # +1.8%  girth < 0.1019
        + 0.01682548 * max(0.0, 0.0586137 - Q.z_7) / 0.01231218   # +1.7%  z_7 < 0.05861
        - 0.01440336 * max(0.0, Q.log_sum_pt - 6.080494) * max(0.0, 0.0001947983 - Q.lam2) / 6.2639e-05   # -1.4%  log_sum_pt > 6.08 and lam2 < 0.0001948
        - 0.01437707 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 62.25 - Q.pt_6) / 0.07984241   # -1.4%  lam1 < 0.00733 and pt_6 < 62.25
        - 0.01348104 * max(0.0, 0.0423228 - Q.C2) / 0.02012665   # -1.3%  C2 < 0.04232
        - 0.01287084 * max(0.0, Q.z_7 - 0.02320757) * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.002539696   # -1.3%  z_7 > 0.02321 and sj3_dr_min < 0.1278
        + 0.01110016 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # +1.1%  lam1 < 0.004184
        - 0.009994397 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.1056549 - Q.absphi_0) / 0.419244   # -1.0%  sum_pt > 988.4 and absphi_0 < 0.1057
        + 0.009590638 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.1056549 - Q.absphi_0) / 0.0003841548   # +1.0%  log_sum_pt > 6.896 and absphi_0 < 0.1057
        - 0.009294767 * max(0.0, 0.0586137 - Q.z_7) * max(0.0, 0.07897949 - Q.absphi_0) / 0.0007310394   # -0.9%  z_7 < 0.05861 and absphi_0 < 0.07898
        - 0.008785248 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, Q.D2_b2 - 0.01618699) / 0.1028513   # -0.9%  log_sum_pt < 6.606 and D2_b2 > 0.01619
        + 0.00822947 * max(0.0, 6.701242 - Q.log_sum_pt) * max(0.0, Q.D2_b2 - 0.380911) / 0.1008885   # +0.8%  log_sum_pt < 6.701 and D2_b2 > 0.3809
        + 0.007285463 * max(0.0, 6.701242 - Q.log_sum_pt) * max(0.0, 0.0003061234 - Q.lam2) / 2.493328e-05   # +0.7%  log_sum_pt < 6.701 and lam2 < 0.0003061
        + 0.007002462 * max(0.0, 641.7188 - Q.sum_pt) / 38.47522   # +0.7%  sum_pt < 641.7
        + 0.006405126 * max(0.0, 6.701242 - Q.log_sum_pt) / 0.1935491   # +0.6%  log_sum_pt < 6.701
        + 0.005695423 * max(0.0, Q.sum_pt - 868.5094) * max(0.0, 4.721224 - Q.D2_b2) / 44.90046   # +0.6%  sum_pt > 868.5 and D2_b2 < 4.721
        - 0.004892653 * max(0.0, 0.1294903 - Q.sj2_dr) * max(0.0, 4.721224 - Q.D2_b2) / 0.08996587   # -0.5%  sj2_dr < 0.1295 and D2_b2 < 4.721
        + 0.004033555 * max(0.0, Q.sum_pt - 937.0312) / 8.087955   # +0.4%  sum_pt > 937
        - 0.003568201 * max(0.0, 0.1019409 - Q.girth) * max(0.0, Q.centroid_offset - 0.01837778) / 0.0001009228   # -0.4%  girth < 0.1019 and centroid_offset > 0.01838
        - 0.003403755 * max(0.0, 752.1 - Q.sum_pt_top5) * max(0.0, Q.D2_b2 - 0.1830092) / 113.9873   # -0.3%  sum_pt_top5 < 752.1 and D2_b2 > 0.183
        + 0.003063888 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.3%  sum_pt > 988.4
        - 0.002404214 * max(0.0, 641.7188 - Q.sum_pt) * max(0.0, 0.0003061234 - Q.lam2) / 0.003656548   # -0.2%  sum_pt < 641.7 and lam2 < 0.0003061
        - 0.002316658 * max(0.0, 0.01517359 - Q.girth) / 0.001389803   # -0.2%  girth < 0.01517
        - 0.002288634 * max(0.0, 6.701242 - Q.log_sum_pt) * max(0.0, 8.836697e-05 - Q.mean_phi2) / 5.122736e-07   # -0.2%  log_sum_pt < 6.701 and mean_phi2 < 8.837e-05
        + 0.002164312 * max(0.0, 53.4375 - Q.pt_7) * max(0.0, 0.1236856 - Q.D2_b2) / 0.3762556   # +0.2%  pt_7 < 53.44 and D2_b2 < 0.1237
        - 0.002111585 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 4.721224 - Q.D2_b2) / 0.009941678   # -0.2%  log_sum_pt > 6.896 and D2_b2 < 4.721
        + 0.001985203 * max(0.0, 0.01517359 - Q.girth) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.002541507   # +0.2%  girth < 0.01517 and n_pt_above_50 < 7
        + 0.001845896 * max(0.0, Q.sum_pt - 937.0312) * max(0.0, 8.836697e-05 - Q.mean_phi2) / 0.0002643409   # +0.2%  sum_pt > 937 and mean_phi2 < 8.837e-05
        - 0.001832808 * max(0.0, 23.21641 - Q.pt_7) / 0.9212932   # -0.2%  pt_7 < 23.22
        + 0.001682003 * max(0.0, 6.701242 - Q.log_sum_pt) * max(0.0, 0.1056549 - Q.absphi_0) / 0.01213397   # +0.2%  log_sum_pt < 6.701 and absphi_0 < 0.1057
        - 0.00155634 * max(0.0, Q.pt_7 - 20.125) * max(0.0, Q.sd_rg - 0.1667615) / 0.3767135   # -0.2%  pt_7 > 20.12 and sd_rg > 0.1668
        - 0.001344544 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # -0.1%  girth < 0.007674
        + 0.001298271 * max(0.0, 0.0586137 - Q.z_7) * max(0.0, 0.006721497 - Q.absphi_0) / 1.973944e-05   # +0.1%  z_7 < 0.05861 and absphi_0 < 0.006721
        - 0.001243821 * max(0.0, Q.sum_pt - 868.5094) * max(0.0, 0.01452637 - Q.absphi_0) / 0.1405135   # -0.1%  sum_pt > 868.5 and absphi_0 < 0.01453
        - 0.001217505 * max(0.0, 0.005576073 - Q.centroid_offset) * max(0.0, 0.03607145 - Q.dr_7) / 9.086096e-06   # -0.1%  centroid_offset < 0.005576 and dr_7 < 0.03607
        + 0.001052603 * max(0.0, 324.5922 - Q.sum_pt_top5) / 1.968832   # +0.1%  sum_pt_top5 < 324.6
        - 0.0009342249 * max(0.0, Q.sum_pt - 868.5094) * max(0.0, 0.01558685 - Q.abseta_2) / 0.1422729   # -0.1%  sum_pt > 868.5 and abseta_2 < 0.01559
        + 0.0007861846 * max(0.0, 53.4375 - Q.pt_7) * max(0.0, Q.pt1_dr01 - 17.82896) / 27.27641   # +0.1%  pt_7 < 53.44 and pt1_dr01 > 17.83
        - 0.0006773791 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.004071886   # -0.1%  log_sum_pt > 6.896 and n_pt_above_50 > 5
        + 0.0005571025 * max(0.0, 367.5938 - Q.sum_pt_top5) / 4.984642   # +0.1%  sum_pt_top5 < 367.6
        - 0.0004728993 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.0%  log_sum_pt > 6.896
        - 0.0004514727 * max(0.0, Q.sum_pt - 937.0312) * max(0.0, Q.absphi_0 - 0.003417683) / 0.06584349   # -0.0%  sum_pt > 937 and absphi_0 > 0.003418
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 6.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.010251 * (0.07362292
        - 0.135204 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # -13.5%  lam2 < 0.003408
        + 0.1266367 * max(0.0, Q.lam1 - 0.00595415) / 0.002480365   # +12.7%  lam1 > 0.005954
        - 0.0835291 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # -8.4%  girth > 0.1019
        - 0.07911333 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -7.9%  lam1 > 0.012
        + 0.07670385 * max(0.0, Q.sj3_dr_max - 0.213399) / 0.03061279   # +7.7%  sj3_dr_max > 0.2134
        + 0.07621386 * max(0.0, Q.girth - 0.06663269) / 0.01393206   # +7.6%  girth > 0.06663
        - 0.04552094 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -4.6%  sj3_dr_max > 0.2623
        + 0.04486785 * max(0.0, Q.LHA - 0.3255822) / 0.01245304   # +4.5%  LHA > 0.3256
        + 0.03936319 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +3.9%  sj2_dr > 0.1683
        + 0.03377365 * max(0.0, Q.LHA - 0.3255822) * max(0.0, Q.eccentricity - 0.7117266) / 0.00207611   # +3.4%  LHA > 0.3256 and eccentricity > 0.7117
        - 0.03014114 * max(0.0, Q.LHA - 0.2931906) / 0.02125042   # -3.0%  LHA > 0.2932
        - 0.02940691 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.log_sum_pt - 6.080494) / 0.01252901   # -2.9%  sj2_dr > 0.1683 and log_sum_pt > 6.08
        + 0.02377805 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +2.4%  tau1 > 0.1136
        + 0.02288167 * max(0.0, Q.girth - 0.1245537) / 0.002632136   # +2.3%  girth > 0.1246
        + 0.0206128 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.eccentricity - 0.9458207) / 0.0008867862   # +2.1%  sj2_dr > 0.1683 and eccentricity > 0.9458
        - 0.0175919 * max(0.0, 1.138757 - Q.N3) * max(0.0, Q.log_sum_pt - 6.080494) / 0.01204456   # -1.8%  N3 < 1.139 and log_sum_pt > 6.08
        - 0.0171412 * max(0.0, 0.2654572 - Q.N2) / 0.07451069   # -1.7%  N2 < 0.2655
        - 0.01278758 * max(0.0, Q.girth - 0.06663269) * max(0.0, Q.n_pt_above_50 - 3.0) / 0.0246836   # -1.3%  girth > 0.06663 and n_pt_above_50 > 3
        + 0.01111965 * max(0.0, 1.138757 - Q.N3) * max(0.0, 0.08670959 - Q.z_7) / 0.0009624291   # +1.1%  N3 < 1.139 and z_7 < 0.08671
        + 0.01081148 * max(0.0, Q.sj3_dr_max - 0.213399) * max(0.0, Q.n_dr_0p05_0p1 - 4.0) / 0.007898646   # +1.1%  sj3_dr_max > 0.2134 and n_dr_0p05_0p1 > 4
        - 0.009022535 * max(0.0, Q.girth - 0.1245537) * max(0.0, Q.eccentricity - 0.7792127) / 0.0002736432   # -0.9%  girth > 0.1246 and eccentricity > 0.7792
        + 0.007812274 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.09482124 - Q.C2) / 0.001379963   # +0.8%  sj2_dr > 0.1683 and C2 < 0.09482
        - 0.007640005 * max(0.0, Q.sj3_dr_max - 0.2623172) * max(0.0, Q.n_dr_0p05_0p1 - 4.0) / 0.004621037   # -0.8%  sj3_dr_max > 0.2623 and n_dr_0p05_0p1 > 4
        + 0.007001671 * max(0.0, Q.sd_rg - 0.1876504) * max(0.0, 0.1515405 - Q.z_dr_0_0p05) / 0.002283966   # +0.7%  sd_rg > 0.1877 and z_dr_0_0p05 < 0.1515
        - 0.006974628 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.n_dr_0_0p05 - 2.0) / 0.04103234   # -0.7%  sj2_dr > 0.1683 and n_dr_0_0p05 > 2
        + 0.006198351 * max(0.0, -0.006779839 - Q.mean_eta) / 0.003173922   # +0.6%  mean_eta < -0.00678
        - 0.00513774 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # -0.5%  sd_rg > 0.2788
        + 0.00349778 * max(0.0, Q.sd_rg - 0.1876504) / 0.01921774   # +0.3%  sd_rg > 0.1877
        + 0.003282373 * max(0.0, Q.sd_rg - 0.1876504) * max(0.0, 0.000537286 - Q.lam2) / 3.14287e-06   # +0.3%  sd_rg > 0.1877 and lam2 < 0.0005373
        - 0.002548952 * max(0.0, -0.006779839 - Q.mean_eta) * max(0.0, -0.03967285 - Q.eta_0) / 9.048956e-05   # -0.3%  mean_eta < -0.00678 and eta_0 < -0.03967
        - 0.002145019 * max(0.0, Q.LHA - 0.3255822) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 0.0001028066   # -0.2%  LHA > 0.3256 and z_dr_0p05_0p1 > 0.6748
        + 0.00153988 * max(0.0, -0.006779839 - Q.mean_eta) * max(0.0, 6.267538 - Q.log_sum_pt) / 0.0001795705   # +0.2%  mean_eta < -0.00678 and log_sum_pt < 6.268
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 43.05;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 43.04721 * (0.04514568
        + 0.1144557 * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.166579   # +11.4%  sj3_dr_min < 0.209
        - 0.1026185 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -10.3%  lam1 < 0.008376
        - 0.08835462 * max(0.0, 0.2089872 - Q.sj3_dr_min) * max(0.0, 0.003408389 - Q.lam2) / 0.000543125   # -8.8%  sj3_dr_min < 0.209 and lam2 < 0.003408
        + 0.06603025 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +6.6%  lam2 < 0.003408
        + 0.05236047 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # +5.2%  LHA < 0.3127
        - 0.04924034 * max(0.0, Q.eccentricity - 0.7117266) / 0.1947312   # -4.9%  eccentricity > 0.7117
        - 0.04896807 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -4.9%  lam1 < 0.006507
        + 0.03530483 * max(0.0, 0.2507612 - Q.max_dr) / 0.1316542   # +3.5%  max_dr < 0.2508
        + 0.03484148 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +3.5%  e3 < 0.0001869
        - 0.03331941 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # -3.3%  sum_pt < 988.4
        + 0.03301656 * max(0.0, Q.log_sum_pt - 6.327379) / 0.2487223   # +3.3%  log_sum_pt > 6.327
        - 0.02808113 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -2.8%  lam2 < 0.0005373
        - 0.02600765 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -2.6%  sj3_dr_max < 0.1426
        + 0.02280714 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +2.3%  N2 < 0.2233
        + 0.01883089 * max(0.0, Q.max_dr - 0.121681) / 0.03562206   # +1.9%  max_dr > 0.1217
        + 0.01434101 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, Q.eccentricity - 0.7117266) / 0.0004354758   # +1.4%  lam1 < 0.006507 and eccentricity > 0.7117
        + 0.01331013 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +1.3%  sj3_dr_max > 0.169
        - 0.01322262 * max(0.0, Q.C2 - 0.01867771) / 0.01385506   # -1.3%  C2 > 0.01868
        - 0.01287985 * max(0.0, Q.sj2_dr - 0.2001708) / 0.02317349   # -1.3%  sj2_dr > 0.2002
        - 0.01173557 * max(0.0, 0.33555 - Q.tau21) / 0.105517   # -1.2%  tau21 < 0.3356
        + 0.01158868 * max(0.0, 0.01002369 - Q.girth2_top3) / 0.006299101   # +1.2%  girth2_top3 < 0.01002
        + 0.01071393 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +1.1%  lam1 < 0.002464
        + 0.01008257 * max(0.0, 0.08082334 - Q.dr_0) * max(0.0, Q.centroid_offset - 0.009480685) / 0.000183564   # +1.0%  dr_0 < 0.08082 and centroid_offset > 0.009481
        - 0.009209405 * max(0.0, 0.08082334 - Q.dr_0) / 0.0377444   # -0.9%  dr_0 < 0.08082
        + 0.008581754 * max(0.0, 0.07264452 - Q.sj3_dr_max) / 0.01287024   # +0.9%  sj3_dr_max < 0.07264
        + 0.00825815 * max(0.0, Q.max_dr - 0.121681) * max(0.0, 0.009032972 - Q.C2_b2) / 0.0001556695   # +0.8%  max_dr > 0.1217 and C2_b2 < 0.009033
        - 0.006750231 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, 7.0 - Q.n_dr_0p05_0p1) / 1.337925   # -0.7%  log_sum_pt > 6.327 and n_dr_0p05_0p1 < 7
        - 0.00643179 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, Q.log_sum_pt - 6.267538) / 0.004076983   # -0.6%  sj3_dr_max > 0.2337 and log_sum_pt > 6.268
        - 0.005937432 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7) / 0.8992052   # -0.6%  N2 < 0.2233 and pt_7 < 53.44
        - 0.005849284 * max(0.0, Q.sj3_dr_max - 0.3012016) / 0.01214652   # -0.6%  sj3_dr_max > 0.3012
        - 0.005679417 * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.8069227   # -0.6%  n_dr_0p2_0p4 < 1
        - 0.005235292 * max(0.0, 763.825 - Q.sum_pt) / 96.68366   # -0.5%  sum_pt < 763.8
        - 0.005232352 * max(0.0, 0.04435703 - Q.M2) / 0.008629948   # -0.5%  M2 < 0.04436
        - 0.004990058 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -0.5%  lam1 < 0.00733
        - 0.004154086 * max(0.0, Q.max_dr - 0.121681) * max(0.0, 0.07861328 - Q.abseta_0) / 0.001374529   # -0.4%  max_dr > 0.1217 and abseta_0 < 0.07861
        - 0.00412207 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, 0.009799324 - Q.zdr_2) / 0.001337409   # -0.4%  log_sum_pt > 6.327 and zdr_2 < 0.009799
        - 0.004032432 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, 0.006197061 - Q.zdr_5) / 0.0008789743   # -0.4%  log_sum_pt > 6.327 and zdr_5 < 0.006197
        - 0.003350121 * max(0.0, 0.01705377 - Q.zdr_0) / 0.006142531   # -0.3%  zdr_0 < 0.01705
        - 0.003339 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 840.0195 - Q.sum_pt) / 6.935225   # -0.3%  N2 < 0.2233 and sum_pt < 840
        - 0.003273469 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, 7.998907 - Q.ptdr0_6) / 1.442602   # -0.3%  log_sum_pt > 6.327 and ptdr0_6 < 7.999
        + 0.003193513 * max(0.0, Q.sj2_dr - 0.2179769) / 0.01834468   # +0.3%  sj2_dr > 0.218
        - 0.003114085 * max(0.0, 0.009612129 - Q.zdr_1) / 0.003260319   # -0.3%  zdr_1 < 0.009612
        + 0.003002291 * max(0.0, 1.002471 - Q.D2) / 0.1871081   # +0.3%  D2 < 1.002
        - 0.002968324 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, 0.08575439 - Q.abseta_4) / 0.014315   # -0.3%  log_sum_pt > 6.327 and abseta_4 < 0.08575
        - 0.00283694 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # -0.3%  tau1 < 0.0437
        + 0.002507931 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, Q.absphi_7 - 0.001937771) / 3.668163e-05   # +0.3%  lam2 < 0.001131 and absphi_7 > 0.001938
        - 0.002410403 * max(0.0, Q.sj2_dr - 0.2179769) * max(0.0, 0.009032972 - Q.C2_b2) / 7.384952e-05   # -0.2%  sj2_dr > 0.218 and C2_b2 < 0.009033
        - 0.002347964 * max(0.0, 25.57812 - Q.pt_7) / 1.327389   # -0.2%  pt_7 < 25.58
        - 0.002159378 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.2832687 - Q.sj2_zsoft) / 0.002686709   # -0.2%  N2 < 0.2233 and sj2_zsoft < 0.2833
        + 0.002084534 * max(0.0, Q.eccentricity - 0.7117266) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.05767584   # +0.2%  eccentricity > 0.7117 and n_pt_above_50 > 6
        - 0.002042159 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1626038 - Q.abseta_7) / 0.005313035   # -0.2%  N2 < 0.2233 and abseta_7 < 0.1626
        - 0.001884865 * max(0.0, Q.sj3_dr_max - 0.3456459) * max(0.0, 0.1167998 - Q.sj3_z3) / 0.0001997328   # -0.2%  sj3_dr_max > 0.3456 and sj3_z3 < 0.1168
        - 0.001883856 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, 8.336138 - Q.ptdr0_7) / 1.552691   # -0.2%  log_sum_pt > 6.327 and ptdr0_7 < 8.336
        - 0.001757264 * max(0.0, Q.sj2_dr - 0.2179769) * max(0.0, 0.5327104 - Q.D2_b2) / 0.003818074   # -0.2%  sj2_dr > 0.218 and D2_b2 < 0.5327
        - 0.001524388 * max(0.0, 1.002471 - Q.D2) * max(0.0, 0.0155217 - Q.zdr_5) / 0.001630632   # -0.2%  D2 < 1.002 and zdr_5 < 0.01552
        + 0.001505422 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.n_dr_0p1_0p2 - 2.0) / 0.0465202   # +0.2%  N2 < 0.2233 and n_dr_0p1_0p2 > 2
        + 0.0014688 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, 0.05966518 - Q.sj2_zsoft) / 6.432356e-05   # +0.1%  sj3_dr_max > 0.2337 and sj2_zsoft < 0.05967
        + 0.001451482 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.06108421 - Q.dr_1) / 1.247729   # +0.1%  sum_pt < 763.8 and dr_1 < 0.06108
        - 0.001330863 * max(0.0, 0.08082334 - Q.dr_0) * max(0.0, Q.centroid_offset - 0.02355416) / 4.26964e-05   # -0.1%  dr_0 < 0.08082 and centroid_offset > 0.02355
        + 0.001092287 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # +0.1%  lam1 > 0.012
        + 0.001038862 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, -0.01790907 - Q.mean_eta) / 1.318391e-07   # +0.1%  e3 < 0.0001869 and mean_eta < -0.01791
        + 0.0009162597 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, -0.0008556753 - Q.mean_phi) / 6.216568e-07   # +0.1%  e3 < 0.0001869 and mean_phi < -0.0008557
        + 0.0009026815 * max(0.0, Q.sj3_dr_max - 0.3456459) / 0.006655619   # +0.1%  sj3_dr_max > 0.3456
        - 0.0008492127 * max(0.0, Q.sj2_dr - 0.2179769) * max(0.0, 0.0006435798 - Q.C2_b2) / 1.186443e-06   # -0.1%  sj2_dr > 0.218 and C2_b2 < 0.0006436
        - 0.0008086635 * max(0.0, Q.max_dr - 0.121681) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.004992694   # -0.1%  max_dr > 0.1217 and n_pt_above_50 > 6
        + 0.0007965757 * max(0.0, Q.sj2_dr - 0.2179769) * max(0.0, Q.lam2 - 0.001130645) / 2.486958e-05   # +0.1%  sj2_dr > 0.218 and lam2 > 0.001131
        - 0.0007089838 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, -0.01452103 - Q.phi_0) / 0.001594834   # -0.1%  log_sum_pt > 6.327 and phi_0 < -0.01452
        + 0.0006732424 * max(0.0, 0.2089872 - Q.sj3_dr_min) * max(0.0, -0.02594505 - Q.mean_phi) / 8.519257e-05   # +0.1%  sj3_dr_min < 0.209 and mean_phi < -0.02595
        + 0.0006678061 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, Q.lam2 - 0.000537286) / 5.527649e-08   # +0.1%  lam1 < 0.00733 and lam2 > 0.0005373
        - 0.0005589138 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, Q.dr0_7 - 0.09118326) / 1.143817e-05   # -0.1%  lam2 < 0.0005373 and dr0_7 > 0.09118
        - 0.0005411928 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, -0.02594505 - Q.mean_phi) / 3.376228e-07   # -0.1%  lam2 < 0.001131 and mean_phi < -0.02595
        + 0.000454074 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.planar_flow - 0.08366273) / 0.001421863   # +0.0%  N2 < 0.2233 and planar_flow > 0.08366
        - 0.0003588809 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, Q.dr1_4 - 0.1883455) / 0.001466951   # -0.0%  log_sum_pt > 6.327 and dr1_4 > 0.1883
        - 0.0003501426 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.lam2 - 0.001130645) / 2.416387e-06   # -0.0%  N2 < 0.2233 and lam2 > 0.001131
        - 0.0003156832 * max(0.0, 0.1329373 - Q.LHA) / 0.007753684   # -0.0%  LHA < 0.1329
        - 0.0003085075 * max(0.0, 25.57812 - Q.pt_7) * max(0.0, 3.032555 - Q.D2_b2) / 1.428175   # -0.0%  pt_7 < 25.58 and D2_b2 < 3.033
        + 0.0002981771 * max(0.0, 0.2233283 - Q.N2) * max(0.0, -0.02594505 - Q.mean_phi) / 3.611955e-05   # +0.0%  N2 < 0.2233 and mean_phi < -0.02595
        - 0.0002198045 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, Q.ptdr0_5 - 13.11362) / 0.02640147   # -0.0%  log_sum_pt > 6.327 and ptdr0_5 > 13.11
        + 0.0001278101 * max(0.0, Q.lam1 - 0.01200373) * max(0.0, 0.02460965 - Q.C3) / 5.693219e-07   # +0.0%  lam1 > 0.012 and C3 < 0.02461
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 11.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.23991 * (0.09902405
        - 0.1082372 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -10.8%  sj3_dr_max < 0.3012
        - 0.08287484 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -8.3%  sum_pt < 739.5
        + 0.06071019 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +6.1%  sj3_dr_max < 0.1986
        + 0.05293302 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +5.3%  e3 < 0.0001869
        - 0.05277169 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # -5.3%  centroid_offset < 0.03776
        + 0.0517711 * max(0.0, 0.002464291 - Q.lam1) * max(0.0, 0.01837778 - Q.centroid_offset) / 8.036566e-06   # +5.2%  lam1 < 0.002464 and centroid_offset < 0.01838
        + 0.04603805 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +4.6%  LHA < 0.2161
        - 0.04377238 * max(0.0, 0.03556091 - Q.e2) / 0.01371972   # -4.4%  e2 < 0.03556
        - 0.04055788 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # -4.1%  dr_0 < 0.06413
        + 0.03599647 * max(0.0, 0.03556091 - Q.e2) * max(0.0, 0.0003061234 - Q.lam2) / 3.480488e-06   # +3.6%  e2 < 0.03556 and lam2 < 0.0003061
        + 0.03595921 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +3.6%  lam1 < 0.00484
        + 0.03220452 * max(0.0, 0.03556091 - Q.e2) * max(0.0, 0.0753896 - Q.z_7) / 0.0004290247   # +3.2%  e2 < 0.03556 and z_7 < 0.07539
        + 0.0246961 * max(0.0, 579.875 - Q.sum_pt_top5) / 65.87113   # +2.5%  sum_pt_top5 < 579.9
        + 0.01740186 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +1.7%  lam1 < 0.002464
        - 0.01719231 * max(0.0, 0.02054282 - Q.girth) / 0.002592388   # -1.7%  girth < 0.02054
        - 0.01645752 * max(0.0, 0.266913 - Q.LHA) / 0.05602629   # -1.6%  LHA < 0.2669
        - 0.01609475 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -1.6%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.01588127 * max(0.0, 739.5 - Q.sum_pt) * max(0.0, 0.008675069 - Q.mean_eta2) / 0.3937849   # +1.6%  sum_pt < 739.5 and mean_eta2 < 0.008675
        + 0.0158123 * max(0.0, 0.06081235 - Q.z_6) * max(0.0, 28.39396 - Q.pt1_dr01) / 0.2343338   # +1.6%  z_6 < 0.06081 and pt1_dr01 < 28.39
        + 0.01511153 * max(0.0, 0.06081235 - Q.z_6) * max(0.0, 0.0005116989 - Q.e3) / 4.735652e-06   # +1.5%  z_6 < 0.06081 and e3 < 0.0005117
        - 0.01483054 * max(0.0, 0.06081235 - Q.z_6) / 0.009654642   # -1.5%  z_6 < 0.06081
        + 0.01314188 * max(0.0, 0.02054282 - Q.girth) * max(0.0, Q.sum_pt_top5 - 716.8828) / 0.2273844   # +1.3%  girth < 0.02054 and sum_pt_top5 > 716.9
        - 0.01311185 * max(0.0, Q.pt_7 - 37.15625) / 3.295689   # -1.3%  pt_7 > 37.16
        + 0.01248648 * max(0.0, 0.03117077 - Q.centroid_offset) / 0.01649145   # +1.2%  centroid_offset < 0.03117
        + 0.01217476 * max(0.0, 0.03117077 - Q.centroid_offset) * max(0.0, 105.9562 - Q.pt_2) / 0.278628   # +1.2%  centroid_offset < 0.03117 and pt_2 < 106
        - 0.01189157 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # -1.2%  sj3_dr_max < 0.107
        - 0.01158227 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.09543973 - Q.z_6) / 0.001514041   # -1.2%  LHA < 0.2161 and z_6 < 0.09544
        + 0.01111892 * max(0.0, 0.04939969 - Q.z_7) / 0.007457505   # +1.1%  z_7 < 0.0494
        + 0.01098608 * max(0.0, 0.03629544 - Q.z_7) / 0.002921513   # +1.1%  z_7 < 0.0363
        + 0.0100165 * max(0.0, 0.0008722282 - Q.lam1) / 0.0001882079   # +1.0%  lam1 < 0.0008722
        - 0.008916648 * max(0.0, Q.log_sum_pt - 6.804164) * max(0.0, 5.334511e-05 - Q.e3) / 6.116454e-07   # -0.9%  log_sum_pt > 6.804 and e3 < 5.335e-05
        + 0.008507145 * max(0.0, Q.log_sum_pt - 6.804164) * max(0.0, 0.01251955 - Q.mean_phi) / 0.0001601797   # +0.9%  log_sum_pt > 6.804 and mean_phi < 0.01252
        + 0.007494954 * max(0.0, 0.02320757 - Q.z_7) / 0.0007159291   # +0.7%  z_7 < 0.02321
        - 0.006918668 * max(0.0, 0.03117077 - Q.centroid_offset) * max(0.0, 0.1364165 - Q.z_2) / 0.0002646871   # -0.7%  centroid_offset < 0.03117 and z_2 < 0.1364
        - 0.006867112 * max(0.0, 20.125 - Q.pt_7) / 0.5468342   # -0.7%  pt_7 < 20.12
        - 0.006426402 * max(0.0, 0.8233866 - Q.z_top5) / 0.02979882   # -0.6%  z_top5 < 0.8234
        + 0.005498478 * max(0.0, 7.12483e-05 - Q.lam1) / 3.380156e-06   # +0.5%  lam1 < 7.125e-05
        + 0.005461528 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +0.5%  z_7 < 0.02807
        - 0.00506713 * max(0.0, 0.03117077 - Q.centroid_offset) * max(0.0, Q.D3 - 0.5730377) / 0.01595865   # -0.5%  centroid_offset < 0.03117 and D3 > 0.573
        + 0.00456172 * max(0.0, 0.03556091 - Q.e2) * max(0.0, Q.zdr_0 - 0.004918231) / 3.97457e-05   # +0.5%  e2 < 0.03556 and zdr_0 > 0.004918
        - 0.00442786 * max(0.0, Q.log_sum_pt - 6.804164) / 0.01257682   # -0.4%  log_sum_pt > 6.804
        - 0.003393129 * max(0.0, 0.03556091 - Q.e2) * max(0.0, Q.log_sum_pt - 6.896095) / 0.0001063848   # -0.3%  e2 < 0.03556 and log_sum_pt > 6.896
        - 0.002554259 * max(0.0, 0.02054282 - Q.girth) * max(0.0, 0.09607851 - Q.M3) / 4.66701e-05   # -0.3%  girth < 0.02054 and M3 < 0.09608
        + 0.002501444 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # +0.3%  pt_5 < 24.58
        + 0.002315161 * max(0.0, 0.02886576 - Q.z_6) / 0.0008381782   # +0.2%  z_6 < 0.02887
        - 0.002202666 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, Q.ptdr0_2 - 11.87288) / 0.08620714   # -0.2%  sj3_dr_max < 0.3012 and ptdr0_2 > 11.87
        - 0.002170989 * max(0.0, 7.12483e-05 - Q.lam1) * max(0.0, 73.75 - Q.pt_5) / 8.367553e-05   # -0.2%  lam1 < 7.125e-05 and pt_5 < 73.75
        + 0.001940887 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # +0.2%  z_6 < 0.0216
        - 0.00187929 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.007798268 - Q.zdr_0) / 1.475768e-05   # -0.2%  log_sum_pt > 6.896 and zdr_0 < 0.007798
        - 0.001877941 * max(0.0, Q.z_top5_slots - 0.908903) / 0.002542271   # -0.2%  z_top5_slots > 0.9089
        - 0.001416731 * max(0.0, Q.log_sum_pt - 6.804164) * max(0.0, 0.00161889 - Q.mean_phi) / 3.623207e-05   # -0.1%  log_sum_pt > 6.804 and mean_phi < 0.001619
        + 0.001343244 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, Q.mean_phi - -0.01275329) / 5.207735e-05   # +0.1%  log_sum_pt > 6.896 and mean_phi > -0.01275
        + 0.001049174 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # +0.1%  pt_7 > 53.44
        - 0.0005373907 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # -0.1%  z_5 < 0.02818
        + 0.0004642423 * max(0.0, 0.06081235 - Q.z_6) * max(0.0, Q.ptdr0_6 - 6.814768) / 0.001372224   # +0.0%  z_6 < 0.06081 and ptdr0_6 > 6.815
        - 0.0003907325 * max(0.0, 0.02054282 - Q.girth) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 4.486726e-05   # -0.0%  girth < 0.02054 and n_dr_0p2_0p4 > 0
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 16.53;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.52611 * (0.0372652
        - 0.1155586 * max(0.0, Q.sj3_dr_max - 0.03464708) / 0.1367546   # -11.6%  sj3_dr_max > 0.03465
        - 0.08293638 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # -8.3%  max_dr < 0.1773
        + 0.08044657 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +8.0%  lam2 < 0.003408
        - 0.06286252 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -6.3%  girth > 0.08724
        + 0.05972359 * max(0.0, Q.LHA - 0.251526) / 0.03924325   # +6.0%  LHA > 0.2515
        - 0.04611024 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -4.6%  lam2 < 0.001131
        + 0.04246427 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +4.2%  sj3_dr_max > 0.179
        + 0.04012366 * max(0.0, Q.LHA - 0.3255822) / 0.01245304   # +4.0%  LHA > 0.3256
        + 0.03678854 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +3.7%  sj2_dr > 0.1683
        + 0.03385979 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # +3.4%  C2 < 0.03579
        - 0.0280973 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -2.8%  sj2_dr > 0.1592
        - 0.02687326 * max(0.0, Q.psi_0p2 - 0.8990266) / 0.08559522   # -2.7%  psi_0p2 > 0.899
        + 0.02315213 * max(0.0, 33.21875 - Q.pt_7) / 3.73296   # +2.3%  pt_7 < 33.22
        + 0.0225362 * max(0.0, 39.75 - Q.pt_6) * max(0.0, 711.875 - Q.sum_pt_top3) / 684.1746   # +2.3%  pt_6 < 39.75 and sum_pt_top3 < 711.9
        - 0.01988413 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # -2.0%  tau1 > 0.05357
        - 0.01962931 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -2.0%  pt_6 < 39.75
        + 0.01763746 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.001460719   # +1.8%  centroid_offset > 0.008092 and sj3_dr_min < 0.209
        - 0.01594711 * max(0.0, 0.03978449 - Q.z_7) / 0.003874427   # -1.6%  z_7 < 0.03978
        - 0.0111393 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 5.0 - Q.n_dr_0_0p05) / 0.01797539   # -1.1%  centroid_offset > 0.02077 and n_dr_0_0p05 < 5
        + 0.01097077 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.01426135 - Q.mean_phi2) / 9.496308e-05   # +1.1%  centroid_offset > 0.008092 and mean_phi2 < 0.01426
        + 0.01068469 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, 0.6144369 - Q.pt_dispersion) / 0.0053957   # +1.1%  tau1 > 0.05357 and pt_dispersion < 0.6144
        + 0.01033544 * max(0.0, 0.08082334 - Q.dr_0) / 0.0377444   # +1.0%  dr_0 < 0.08082
        + 0.009359513 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +0.9%  centroid_offset < 0.01838
        + 0.009281703 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, Q.sj2_zsoft - 0.05966518) / 0.005829511   # +0.9%  sj2_dr > 0.1592 and sj2_zsoft > 0.05967
        + 0.009044824 * max(0.0, 7.330536e-06 - Q.e3) / 2.382852e-06   # +0.9%  e3 < 7.331e-06
        + 0.009012292 * max(0.0, Q.sj3_dr_max - 0.1789613) * max(0.0, 1.59265 - Q.D2_b2) / 0.03741161   # +0.9%  sj3_dr_max > 0.179 and D2_b2 < 1.593
        + 0.008935577 * max(0.0, 6.423044 - Q.log_sum_pt) / 0.05674989   # +0.9%  log_sum_pt < 6.423
        + 0.008581216 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +0.9%  centroid_offset > 0.008092
        + 0.0085297 * max(0.0, Q.centroid_offset - 0.02685622) / 0.003254849   # +0.9%  centroid_offset > 0.02686
        - 0.00844387 * max(0.0, 39.75 - Q.pt_6) * max(0.0, 4.721224 - Q.D2_b2) / 13.45505   # -0.8%  pt_6 < 39.75 and D2_b2 < 4.721
        + 0.007613954 * max(0.0, 0.04355037 - Q.z_6) / 0.003302426   # +0.8%  z_6 < 0.04355
        - 0.007283508 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # -0.7%  girth > 0.1019
        - 0.005961188 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.6%  z_6 < 0.0216
        - 0.005648378 * max(0.0, 39.75 - Q.pt_6) * max(0.0, 0.7861046 - Q.z_top3_slots) / 0.3107099   # -0.6%  pt_6 < 39.75 and z_top3_slots < 0.7861
        + 0.00541475 * max(0.0, 19.46875 - Q.pt_6) / 0.2515892   # +0.5%  pt_6 < 19.47
        - 0.004957547 * max(0.0, Q.girth - 0.1019409) * max(0.0, Q.pt_7 - 15.55391) / 0.1042854   # -0.5%  girth > 0.1019 and pt_7 > 15.55
        - 0.004836925 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 0.1246692   # -0.5%  sj2_dr > 0.1592 and n_dr_0p05_0p1 < 5
        + 0.004822949 * max(0.0, 31.90625 - Q.pt_6) / 1.826854   # +0.5%  pt_6 < 31.91
        - 0.004511646 * max(0.0, 5.794419e-05 - Q.lam2) / 1.900658e-05   # -0.5%  lam2 < 5.794e-05
        - 0.004472947 * max(0.0, Q.sum_pt - 840.0195) * max(0.0, Q.D2_b2 - 0.716559) / 84.38587   # -0.4%  sum_pt > 840 and D2_b2 > 0.7166
        + 0.004358342 * max(0.0, Q.girth - 0.08723651) * max(0.0, 0.380911 - Q.D2_b2) / 0.001021187   # +0.4%  girth > 0.08724 and D2_b2 < 0.3809
        + 0.004122007 * max(0.0, Q.sum_pt - 840.0195) / 24.43934   # +0.4%  sum_pt > 840
        + 0.003924083 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 0.3661115 - Q.sj3_dr23) / 0.000832188   # +0.4%  centroid_offset > 0.02077 and sj3_dr23 < 0.3661
        - 0.003841025 * max(0.0, Q.sj3_dr_max - 0.03464708) * max(0.0, 0.01488897 - Q.dr_min_012) / 0.001063317   # -0.4%  sj3_dr_max > 0.03465 and dr_min_012 < 0.01489
        - 0.003032855 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.1115136 - Q.planar_flow) / 0.0002206557   # -0.3%  centroid_offset < 0.01838 and planar_flow < 0.1115
        + 0.002929197 * max(0.0, Q.girth - 0.1484084) / 0.0008957407   # +0.3%  girth > 0.1484
        + 0.002823815 * max(0.0, Q.sj3_dr_max - 0.1789613) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.03477763   # +0.3%  sj3_dr_max > 0.179 and n_dr_0p2_0p4 < 2
        + 0.002808912 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.09482124 - Q.C2) / 0.001613212   # +0.3%  sj2_dr > 0.1592 and C2 < 0.09482
        - 0.00273276 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.n_dr_0_0p05 - 6.0) / 0.002859347   # -0.3%  centroid_offset > 0.008092 and n_dr_0_0p05 > 6
        - 0.002684344 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, 0.380911 - Q.D2_b2) / 0.0003886793   # -0.3%  centroid_offset > 0.02686 and D2_b2 < 0.3809
        + 0.002318753 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 0.002094451   # +0.2%  lam2 < 0.003408 and n_dr_0p1_0p2 > 1
        - 0.002238103 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.lam2 - 0.000537286) / 1.101827e-05   # -0.2%  centroid_offset > 0.008092 and lam2 > 0.0005373
        - 0.002231781 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.D3 - 0.8272948) / 0.005997404   # -0.2%  centroid_offset > 0.008092 and D3 > 0.8273
        - 0.001897374 * max(0.0, 0.07708997 - Q.sj3_pairmin_over_m) / 0.00772004   # -0.2%  sj3_pairmin_over_m < 0.07709
        - 0.001822465 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.mean_phi - -0.01753483) / 0.00072428   # -0.2%  sj2_dr > 0.1683 and mean_phi > -0.01753
        + 0.001735668 * max(0.0, 24.42188 - Q.pt_6) / 0.6046572   # +0.2%  pt_6 < 24.42
        + 0.001702544 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.8319502 - Q.eccentricity) / 0.0004572695   # +0.2%  centroid_offset > 0.008092 and eccentricity < 0.832
        - 0.001610234 * max(0.0, Q.tau1 - 0.1748444) / 0.001246453   # -0.2%  tau1 > 0.1748
        + 0.001483395 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, Q.C3 - 0.04165477) / 2.138046e-06   # +0.1%  lam2 < 0.001131 and C3 > 0.04165
        + 0.001346578 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, Q.pt_2 - 69.875) / 0.03159951   # +0.1%  centroid_offset > 0.02686 and pt_2 > 69.88
        - 0.001322182 * max(0.0, Q.girth - 0.1019409) * max(0.0, 0.2669656 - Q.D2_b2) / 0.0004057011   # -0.1%  girth > 0.1019 and D2_b2 < 0.267
        - 0.0008738558 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, Q.mean_eta - 0.02644207) / 7.861676e-05   # -0.1%  sj2_dr > 0.1592 and mean_eta > 0.02644
        + 0.0008645752 * max(0.0, Q.z_dr_0p1_0p2 - 0.1585582) / 0.08365029   # +0.1%  z_dr_0p1_0p2 > 0.1586
        + 0.0008214004 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_dr_0p05_0p1 - 0.0) / 0.01198963   # +0.1%  centroid_offset < 0.01838 and n_dr_0p05_0p1 > 0
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 38.9;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 38.89643 * (0.05563288
        + 0.05395159 * max(0.0, Q.girth - 0.0479157) * max(0.0, 2.357246 - Q.D2) / 0.03555018   # +5.4%  girth > 0.04792 and D2 < 2.357
        - 0.04960958 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 2.843757 - Q.D2) / 0.004089948   # -5.0%  lam1 > 0.00733 and D2 < 2.844
        + 0.04653925 * max(0.0, 0.08865369 - Q.tau1) / 0.03771145   # +4.7%  tau1 < 0.08865
        - 0.04522611 * max(0.0, Q.girth - 0.0479157) / 0.02294968   # -4.5%  girth > 0.04792
        - 0.04455704 * max(0.0, 0.4235925 - Q.LHA) / 0.1821673   # -4.5%  LHA < 0.4236
        + 0.03992575 * max(0.0, 0.1594012 - Q.sj3_dr_max) / 0.04422261   # +4.0%  sj3_dr_max < 0.1594
        + 0.03654308 * max(0.0, 0.04990367 - Q.centroid_offset) / 0.03352573   # +3.7%  centroid_offset < 0.0499
        + 0.03397527 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +3.4%  lam2 < 0.003408
        + 0.03204792 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # +3.2%  tau1 > 0.1028
        + 0.03029301 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +3.0%  lam1 < 0.008376
        - 0.03000689 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # -3.0%  sj3_dr_max < 0.2134
        - 0.02965553 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -3.0%  e3 < 8.148e-05
        - 0.02703498 * max(0.0, 0.06108601 - Q.girth) / 0.01868685   # -2.7%  girth < 0.06109
        - 0.02648061 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -2.6%  girth > 0.08724
        - 0.02615342 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # -2.6%  lam1 > 0.00733
        + 0.02576138 * max(0.0, Q.lam1 - 0.002464291) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0003749324   # +2.6%  lam1 > 0.002464 and planar_flow < 0.195
        - 0.02508901 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # -2.5%  lam1 < 0.00484
        + 0.02191987 * max(0.0, Q.lam1 - 0.002464291) / 0.004201699   # +2.2%  lam1 > 0.002464
        + 0.0217459 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 90.625 - Q.pt_4) / 0.08206501   # +2.2%  lam1 > 0.00733 and pt_4 < 90.62
        - 0.02066035 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # -2.1%  lam1 > 0.01643
        - 0.02044566 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # -2.0%  lam1 < 0.003377
        - 0.01894718 * max(0.0, 0.02685622 - Q.centroid_offset) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.01150938   # -1.9%  centroid_offset < 0.02686 and n_dr_0p2_0p4 < 1
        + 0.01752947 * max(0.0, Q.lam1 - 0.01643375) * max(0.0, 3.885568 - Q.D2) / 0.002114372   # +1.8%  lam1 > 0.01643 and D2 < 3.886
        + 0.01595451 * max(0.0, Q.tau1 - 0.06345984) / 0.02203215   # +1.6%  tau1 > 0.06346
        - 0.01349233 * max(0.0, Q.sj3_dr_max - 0.07264452) / 0.1084421   # -1.3%  sj3_dr_max > 0.07264
        - 0.0133407 * max(0.0, Q.girth - 0.08065885) / 0.009330634   # -1.3%  girth > 0.08066
        + 0.01325483 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # +1.3%  lam1 > 0.012
        + 0.01289379 * max(0.0, Q.N2 - 0.1333619) / 0.1019708   # +1.3%  N2 > 0.1334
        - 0.0126981 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.lam1 - 0.008375572) / 0.0001274849   # -1.3%  planar_flow < 0.195 and lam1 > 0.008376
        + 0.01240135 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # +1.2%  girth > 0.1019
        - 0.0122979 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # -1.2%  lam2 < 0.0003061
        - 0.01139241 * max(0.0, Q.girth - 0.08065885) * max(0.0, 2.843757 - Q.D2) / 0.01853389   # -1.1%  girth > 0.08066 and D2 < 2.844
        - 0.01122972 * max(0.0, Q.LHA - 0.3467135) / 0.008898884   # -1.1%  LHA > 0.3467
        + 0.01006793 * max(0.0, 0.2346184 - Q.LHA) / 0.04003744   # +1.0%  LHA < 0.2346
        - 0.009833366 * max(0.0, Q.lam1 - 0.002464291) * max(0.0, Q.D2 - 0.2568137) / 0.002599874   # -1.0%  lam1 > 0.002464 and D2 > 0.2568
        - 0.009583965 * max(0.0, 0.0931108 - Q.max_dr) / 0.02003526   # -1.0%  max_dr < 0.09311
        + 0.008160212 * max(0.0, Q.psi_0p1 - 0.7688952) / 0.1381236   # +0.8%  psi_0p1 > 0.7689
        - 0.006761874 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # -0.7%  tau1 > 0.1136
        + 0.006714354 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +0.7%  LHA > 0.3033
        + 0.006538997 * max(0.0, 0.04990367 - Q.centroid_offset) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.06026392   # +0.7%  centroid_offset < 0.0499 and n_dr_0p2_0p4 < 2
        + 0.006234197 * max(0.0, 0.03360421 - Q.girth) / 0.006496997   # +0.6%  girth < 0.0336
        - 0.006069247 * max(0.0, Q.girth - 0.08065885) * max(0.0, 0.09384951 - Q.sj2_zsoft) / 5.444145e-06   # -0.6%  girth > 0.08066 and sj2_zsoft < 0.09385
        - 0.006009919 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -0.6%  sj3_dr_max < 0.1426
        + 0.005796096 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.46415) / 0.01159499   # +0.6%  planar_flow < 0.195 and log_sum_pt > 6.464
        - 0.005654743 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.001397205   # -0.6%  z_dr_0p1_0p2 < 0.1586 and centroid_offset < 0.02686
        - 0.005399381 * max(0.0, 1.432482 - Q.D2) / 0.4017356   # -0.5%  D2 < 1.432
        - 0.005360395 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) / 0.09532154   # -0.5%  z_dr_0p1_0p2 < 0.1586
        - 0.005001441 * max(0.0, Q.N2 - 0.1754643) / 0.07481845   # -0.5%  N2 > 0.1755
        - 0.004458932 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 43.5 - Q.pt_7) / 0.0006101229   # -0.4%  e3 < 8.148e-05 and pt_7 < 43.5
        + 0.004392656 * max(0.0, 0.02685622 - Q.centroid_offset) / 0.01292674   # +0.4%  centroid_offset < 0.02686
        - 0.003721486 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 0.2507612 - Q.max_dr) / 0.008139731   # -0.4%  planar_flow < 0.195 and max_dr < 0.2508
        - 0.003532729 * max(0.0, 4.192959e-06 - Q.e3) / 1.214837e-06   # -0.4%  e3 < 4.193e-06
        + 0.00331201 * max(0.0, 1.050302e-05 - Q.e3) / 3.717236e-06   # +0.3%  e3 < 1.05e-05
        - 0.003164664 * max(0.0, Q.lam1 - 0.002464291) * max(0.0, 0.1879486 - Q.sj3_dr_max) / 1.34726e-05   # -0.3%  lam1 > 0.002464 and sj3_dr_max < 0.1879
        + 0.003102761 * max(0.0, 0.08050702 - Q.max_dr) / 0.01536431   # +0.3%  max_dr < 0.08051
        - 0.0023163 * max(0.0, 0.0006570502 - Q.girth2_top5) / 0.0001395474   # -0.2%  girth2_top5 < 0.0006571
        - 0.002158903 * max(0.0, 0.04990367 - Q.centroid_offset) * max(0.0, Q.tau4 - 0.002042374) / 4.300788e-05   # -0.2%  centroid_offset < 0.0499 and tau4 > 0.002042
        - 0.001746042 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 56.53125 - Q.pt_6) / 1.191827   # -0.2%  planar_flow < 0.195 and pt_6 < 56.53
        - 0.00173118 * max(0.0, 0.02685622 - Q.centroid_offset) * max(0.0, 1.808379 - Q.N3) / 0.005372382   # -0.2%  centroid_offset < 0.02686 and N3 < 1.808
        - 0.001658574 * max(0.0, Q.girth - 0.08065885) * max(0.0, Q.D2_b2 - 1.911889) / 0.0002280963   # -0.2%  girth > 0.08066 and D2_b2 > 1.912
        + 0.001308252 * max(0.0, 0.06108601 - Q.girth) * max(0.0, -0.00469376 - Q.mean_phi) / 2.708081e-05   # +0.1%  girth < 0.06109 and mean_phi < -0.004694
        + 0.001267 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, 1.432482 - Q.D2) / 0.003360448   # +0.1%  sj3_dr_max < 0.1426 and D2 < 1.432
        - 0.001026326 * max(0.0, 33.02656 - Q.pt_5) / 1.052976   # -0.1%  pt_5 < 33.03
        + 0.001020889 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, -0.004664942 - Q.mean_eta) / 9.65369e-06   # +0.1%  lam1 < 0.008376 and mean_eta < -0.004665
        - 0.0009570785 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 29.04219 - Q.pt_7) / 0.1265631   # -0.1%  planar_flow < 0.195 and pt_7 < 29.04
        - 0.0006857303 * max(0.0, 0.02481095 - Q.sj3_dr_max) / 0.001457689   # -0.1%  sj3_dr_max < 0.02481
        + 0.0006273433 * max(0.0, 0.00483998 - Q.lam1) * max(0.0, Q.mean_eta - 0.01271871) / 1.337782e-06   # +0.1%  lam1 < 0.00484 and mean_eta > 0.01272
        + 0.0004730181 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, Q.mean_phi - 0.01251955) / 6.443268e-07   # +0.0%  lam1 < 0.003377 and mean_phi > 0.01252
        + 0.0003606823 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, -0.009391681 - Q.mean_eta) / 9.810597e-07   # +0.0%  lam1 < 0.003377 and mean_eta < -0.009392
        + 0.0003047664 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) * max(0.0, Q.pt1_dr01 - 17.82896) / 0.05125238   # +0.0%  z_dr_0p1_0p2 < 0.1586 and pt1_dr01 > 17.83
        - 0.0002772112 * max(0.0, Q.ptdr0_6 - 11.6057) / 0.1402639   # -0.0%  ptdr0_6 > 11.61
        - 0.0001548548 * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) * max(0.0, -0.02665591 - Q.mean_eta) / 3.402016e-05   # -0.0%  z_dr_0p05_0p1 > 0.6748 and mean_eta < -0.02666
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 16.54;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.54219 * (-0.05121658
        - 0.1312306 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # -13.1%  sj3_dr_max < 0.107
        + 0.1057173 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, 0.0153634 - Q.girth2_top3) / 0.0003596312   # +10.6%  sj3_dr_max < 0.107 and girth2_top3 < 0.01536
        - 0.07571566 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, 0.0153634 - Q.girth2_top3) / 0.0005547224   # -7.6%  sj3_dr_max < 0.1426 and girth2_top3 < 0.01536
        + 0.04923962 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +4.9%  sj3_dr_max < 0.1426
        + 0.03383835 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +3.4%  lam2 < 0.0001948
        - 0.0313335 * max(0.0, 0.1767241 - Q.LHA) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.00104409   # -3.1%  LHA < 0.1767 and z_dr_0p2_0p4 < 0.05644
        + 0.03130809 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +3.1%  sj3_dr_max < 0.1986
        + 0.03116394 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.0001209282   # +3.1%  lam1 < 0.005433 and z_dr_0p2_0p4 < 0.05644
        - 0.03071527 * max(0.0, 0.1594012 - Q.sj3_dr_max) * max(0.0, 0.001864148 - Q.girth2_top2) / 6.497216e-05   # -3.1%  sj3_dr_max < 0.1594 and girth2_top2 < 0.001864
        + 0.0289579 * max(0.0, 0.1767241 - Q.LHA) / 0.01861764   # +2.9%  LHA < 0.1767
        + 0.02878722 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, 0.001864148 - Q.girth2_top2) / 3.719196e-05   # +2.9%  sj3_dr_max < 0.107 and girth2_top2 < 0.001864
        - 0.0262819 * max(0.0, 0.03360421 - Q.girth) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0001637288   # -2.6%  girth < 0.0336 and centroid_offset < 0.03117
        - 0.0232353 * max(0.0, Q.max_dr - 0.290267) / 0.001726409   # -2.3%  max_dr > 0.2903
        + 0.02181298 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 0.03117077 - Q.centroid_offset) / 4.498827e-05   # +2.2%  lam1 < 0.005433 and centroid_offset < 0.03117
        + 0.02127161 * max(0.0, 0.1594012 - Q.sj3_dr_max) / 0.04422261   # +2.1%  sj3_dr_max < 0.1594
        - 0.02039997 * max(0.0, 0.0262518 - Q.tau1) / 0.005366404   # -2.0%  tau1 < 0.02625
        - 0.01932254 * max(0.0, 0.251526 - Q.LHA) / 0.04800428   # -1.9%  LHA < 0.2515
        + 0.01869099 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # +1.9%  lam1 < 0.003377
        + 0.01851383 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.02355416 - Q.centroid_offset) / 4.507403e-05   # +1.9%  lam1 < 0.00733 and centroid_offset < 0.02355
        + 0.01749124 * max(0.0, 0.03459477 - Q.tau1) / 0.008474553   # +1.7%  tau1 < 0.03459
        + 0.01663009 * max(0.0, 0.0262518 - Q.tau1) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0001164763   # +1.7%  tau1 < 0.02625 and centroid_offset < 0.03117
        - 0.01626423 * max(0.0, 0.03360421 - Q.girth) / 0.006496997   # -1.6%  girth < 0.0336
        - 0.01617045 * max(0.0, 0.01713288 - Q.tau2) / 0.008065871   # -1.6%  tau2 < 0.01713
        - 0.0121538 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -1.2%  centroid_offset > 0.00679
        + 0.01201123 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.001864148 - Q.girth2_top2) / 8.694567e-05   # +1.2%  sj3_dr_max < 0.1986 and girth2_top2 < 0.001864
        - 0.01125165 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -1.1%  log_sum_pt > 6.701
        - 0.0109392 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, 22.38578 - Q.pt1_dr01) / 0.780888   # -1.1%  sj3_dr_max < 0.1426 and pt1_dr01 < 22.39
        + 0.01070383 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.00327978 - Q.girth2_top5) / 7.935857e-05   # +1.1%  log_sum_pt > 6.701 and girth2_top5 < 0.00328
        - 0.01017187 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.00633598 - Q.girth2_top5) / 0.0001712067   # -1.0%  log_sum_pt > 6.701 and girth2_top5 < 0.006336
        - 0.009126966 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.001056655 - Q.girth2_top2) / 2.034239e-06   # -0.9%  lam1 < 0.00733 and girth2_top2 < 0.001057
        + 0.008996989 * max(0.0, 0.03459477 - Q.tau1) * max(0.0, 0.001056655 - Q.girth2_top2) / 6.77448e-06   # +0.9%  tau1 < 0.03459 and girth2_top2 < 0.001057
        + 0.008904932 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55592   # +0.9%  sum_pt_top5 > 658.1
        - 0.007859124 * max(0.0, 0.02054282 - Q.girth) / 0.002592388   # -0.8%  girth < 0.02054
        - 0.007628861 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 38.53125 - Q.pt_7) / 0.01790589   # -0.8%  lam1 < 0.005433 and pt_7 < 38.53
        + 0.00706279 * max(0.0, 0.03360421 - Q.girth) * max(0.0, Q.psi_0p1 - 0.9761279) / 0.0001475117   # +0.7%  girth < 0.0336 and psi_0p1 > 0.9761
        + 0.007043346 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 5.351077 - Q.pt1_dr01) / 0.008800757   # +0.7%  lam1 < 0.005433 and pt1_dr01 < 5.351
        + 0.006462589 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 0.2752975   # +0.6%  sj3_dr_max < 0.1986 and n_dr_0p05_0p1 < 5
        - 0.006458535 * max(0.0, 0.1594012 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.003343241) / 0.00037024   # -0.6%  sj3_dr_max < 0.1594 and centroid_offset > 0.003343
        - 0.0051471 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01002369 - Q.girth2_top3) / 0.0003076784   # -0.5%  log_sum_pt > 6.701 and girth2_top3 < 0.01002
        - 0.005123262 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, Q.centroid_offset - 0.006789738) / 5.377342e-06   # -0.5%  lam1 < 0.003377 and centroid_offset > 0.00679
        - 0.004820272 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.01627885) / 7.113e-05   # -0.5%  sj3_dr_max < 0.107 and centroid_offset > 0.01628
        + 0.004717175 * max(0.0, 0.04355037 - Q.z_6) / 0.003302426   # +0.5%  z_6 < 0.04355
        - 0.004147637 * max(0.0, 0.04355037 - Q.z_6) * max(0.0, 0.07141113 - Q.absphi_1) / 0.0001834987   # -0.4%  z_6 < 0.04355 and absphi_1 < 0.07141
        + 0.003217457 * max(0.0, 0.0002758826 - Q.lam1) / 3.586469e-05   # +0.3%  lam1 < 0.0002759
        - 0.00289 * max(0.0, 0.01357518 - Q.tau1) * max(0.0, 0.001056655 - Q.girth2_top2) / 1.347821e-06   # -0.3%  tau1 < 0.01358 and girth2_top2 < 0.001057
        + 0.002787059 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.5925897 - Q.planar_flow) / 0.0009146444   # +0.3%  lam1 < 0.00733 and planar_flow < 0.5926
        - 0.002722284 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -0.3%  lam1 < 0.00733
        - 0.002650066 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 6.502799 - Q.log_sum_pt) / 0.0001002502   # -0.3%  lam1 < 0.005433 and log_sum_pt < 6.503
        + 0.002571562 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.002915531 - Q.girth2_top3) / 7.495703e-05   # +0.3%  log_sum_pt > 6.701 and girth2_top3 < 0.002916
        + 0.002297934 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 33.21875 - Q.pt_7) / 0.3351491   # +0.2%  log_sum_pt > 6.701 and pt_7 < 33.22
        - 0.001953509 * max(0.0, Q.z_dr_0_0p05 - 0.9008535) / 0.03206403   # -0.2%  z_dr_0_0p05 > 0.9009
        - 0.001933476 * max(0.0, Q.z_top5_slots - 0.8919245) / 0.004621614   # -0.2%  z_top5_slots > 0.8919
        - 0.001418327 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, Q.centroid_offset - 0.01627885) / 1.955234e-06   # -0.1%  lam1 < 0.003377 and centroid_offset > 0.01628
        - 0.0007346439 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 8.0 - Q.n_pt_above_10) / 8.553193   # -0.1%  sum_pt_top5 > 658.1 and n_pt_above_10 < 8
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 20.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.17672 * (-0.1987429
        + 0.1207599 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 0.001130645 - Q.lam2) / 2.659012e-06   # +12.1%  lam1 < 0.005954 and lam2 < 0.001131
        + 0.08743899 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +8.7%  sum_pt < 988.4
        + 0.06646122 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +6.6%  sj3_dr_max < 0.2134
        + 0.06537389 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +6.5%  lam1 < 0.005954
        - 0.05531182 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -5.5%  sj3_dr_max < 0.1426
        - 0.05144865 * max(0.0, 0.05464922 - Q.girth) / 0.01532978   # -5.1%  girth < 0.05465
        + 0.04884779 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0002544934   # +4.9%  tau1 < 0.0437 and centroid_offset < 0.03117
        + 0.04595901 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # +4.6%  lam1 < 0.003377
        + 0.03934528 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, 0.00733008 - Q.lam1) / 4.507403e-05   # +3.9%  centroid_offset < 0.02355 and lam1 < 0.00733
        - 0.03619679 * max(0.0, 0.04081947 - Q.girth) / 0.009176315   # -3.6%  girth < 0.04082
        - 0.03173571 * max(0.0, 813.4156 - Q.sum_pt) / 129.0758   # -3.2%  sum_pt < 813.4
        + 0.02200584 * max(0.0, 0.5925897 - Q.planar_flow) * max(0.0, 4.721224 - Q.D2_b2) / 1.452457   # +2.2%  planar_flow < 0.5926 and D2_b2 < 4.721
        - 0.02187343 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 0.08543881 - Q.sj3_dr_min) / 0.0001709388   # -2.2%  lam1 < 0.005954 and sj3_dr_min < 0.08544
        + 0.02084734 * max(0.0, 0.01627885 - Q.centroid_offset) / 0.005402048   # +2.1%  centroid_offset < 0.01628
        + 0.02007919 * max(0.0, 0.03117077 - Q.centroid_offset) / 0.01649145   # +2.0%  centroid_offset < 0.03117
        + 0.01885704 * max(0.0, 0.01627885 - Q.centroid_offset) * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.0005664198   # +1.9%  centroid_offset < 0.01628 and sj3_dr_min < 0.1278
        - 0.01734107 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -1.7%  sj2_dr < 0.1295
        - 0.01576435 * max(0.0, 1.960701e-05 - Q.e3) / 8.488392e-06   # -1.6%  e3 < 1.961e-05
        - 0.01546148 * max(0.0, 0.02689598 - Q.girth) / 0.004330137   # -1.5%  girth < 0.0269
        + 0.01528336 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +1.5%  lam2 < 0.0001948
        - 0.01379338 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # -1.4%  C2_b2 < 0.009033
        - 0.01333253 * max(0.0, 988.4078 - Q.sum_pt) * max(0.0, 0.01627885 - Q.centroid_offset) / 1.063436   # -1.3%  sum_pt < 988.4 and centroid_offset < 0.01628
        + 0.01321082 * max(0.0, 813.4156 - Q.sum_pt) * max(0.0, 3.0374e-08 - Q.e4) / 2.845153e-06   # +1.3%  sum_pt < 813.4 and e4 < 3.037e-08
        + 0.01276491 * max(0.0, Q.lam2 - 0.000537286) / 0.0003825703   # +1.3%  lam2 > 0.0005373
        + 0.01167233 * max(0.0, 0.1778793 - Q.sj2_dr) / 0.05866968   # +1.2%  sj2_dr < 0.1779
        - 0.01041635 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # -1.0%  tau1 < 0.0437
        + 0.007750653 * max(0.0, 813.4156 - Q.sum_pt) * max(0.0, Q.psi_0p2 - 0.79448) / 21.32423   # +0.8%  sum_pt < 813.4 and psi_0p2 > 0.7945
        + 0.007523118 * max(0.0, 0.1027585 - Q.max_dr) / 0.02411847   # +0.8%  max_dr < 0.1028
        + 0.007284924 * max(0.0, 6.638339 - Q.log_sum_pt) / 0.1524936   # +0.7%  log_sum_pt < 6.638
        - 0.006165944 * max(0.0, 0.5925897 - Q.planar_flow) / 0.3505892   # -0.6%  planar_flow < 0.5926
        + 0.004858312 * max(0.0, 0.0005049491 - Q.lam1) / 8.739619e-05   # +0.5%  lam1 < 0.0005049
        + 0.004683701 * max(0.0, 0.01675541 - Q.dr_0) / 0.002278483   # +0.5%  dr_0 < 0.01676
        + 0.004667572 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.5%  n_dr_0p2_0p4 > 1
        - 0.004339234 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, -0.006779839 - Q.mean_eta) / 4.043269e-06   # -0.4%  lam1 < 0.005954 and mean_eta < -0.00678
        + 0.004322532 * max(0.0, 0.0001413912 - Q.lam1) / 1.223988e-05   # +0.4%  lam1 < 0.0001414
        - 0.004320228 * max(0.0, 0.01627885 - Q.centroid_offset) * max(0.0, 0.1830092 - Q.D2_b2) / 0.0002309693   # -0.4%  centroid_offset < 0.01628 and D2_b2 < 0.183
        - 0.004002162 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.mean_eta - 0.009367547) / 3.008598e-06   # -0.4%  lam1 < 0.005954 and mean_eta > 0.009368
        + 0.003884206 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +0.4%  e3 > 0.0001869
        - 0.003410614 * max(0.0, Q.lam2 - 0.000537286) * max(0.0, 0.9979797 - Q.eccentricity) / 0.000146617   # -0.3%  lam2 > 0.0005373 and eccentricity < 0.998
        + 0.003355819 * max(0.0, 988.4078 - Q.sum_pt) * max(0.0, 0.05753583 - Q.z_5) / 0.5190201   # +0.3%  sum_pt < 988.4 and z_5 < 0.05754
        + 0.003151309 * max(0.0, 813.4156 - Q.sum_pt) * max(0.0, 0.0095303 - Q.girth2_top2) / 0.5861327   # +0.3%  sum_pt < 813.4 and girth2_top2 < 0.00953
        + 0.003129148 * max(0.0, 0.007673833 - Q.girth) * max(0.0, 0.08074951 - Q.absphi_7) / 1.592008e-05   # +0.3%  girth < 0.007674 and absphi_7 < 0.08075
        - 0.00302571 * max(0.0, 0.05464922 - Q.girth) * max(0.0, 0.0006973656 - Q.mean_phi) / 4.723126e-05   # -0.3%  girth < 0.05465 and mean_phi < 0.0006974
        - 0.002989227 * max(0.0, 0.01627885 - Q.centroid_offset) * max(0.0, 1.808379 - Q.N3) / 0.00218053   # -0.3%  centroid_offset < 0.01628 and N3 < 1.808
        - 0.002586953 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.mean_phi - 0.01251955) / 2.019727e-06   # -0.3%  lam1 < 0.005954 and mean_phi > 0.01252
        - 0.002493758 * max(0.0, 0.007078158 - Q.e2) / 0.0007714962   # -0.2%  e2 < 0.007078
        - 0.002490057 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, -0.009352575 - Q.mean_phi) / 2.9369e-06   # -0.2%  lam1 < 0.005954 and mean_phi < -0.009353
        + 0.002170612 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # +0.2%  lam1 > 0.01643
        - 0.002127213 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # -0.2%  girth < 0.007674
        - 0.001948131 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 1.911889 - Q.D2_b2) / 0.1770159   # -0.2%  n_dr_0p2_0p4 > 1 and D2_b2 < 1.912
        - 0.001762807 * max(0.0, 0.004918231 - Q.zdr_0) / 0.0006323791   # -0.2%  zdr_0 < 0.004918
        - 0.001689282 * max(0.0, 0.002316125 - Q.centroid_offset) / 9.872932e-05   # -0.2%  centroid_offset < 0.002316
        - 0.00167352 * max(0.0, 0.004918231 - Q.zdr_0) * max(0.0, 182.125 - Q.pt_1) / 0.02327674   # -0.2%  zdr_0 < 0.004918 and pt_1 < 182.1
        - 0.001352765 * max(0.0, 813.4156 - Q.sum_pt) * max(0.0, 0.0003125151 - Q.girth2_top2) / 0.002308904   # -0.1%  sum_pt < 813.4 and girth2_top2 < 0.0003125
        - 0.001273867 * max(0.0, 4.192959e-06 - Q.e3) * max(0.0, Q.n_dr_0p05_0p1 - 3.0) / 6.634517e-08   # -0.1%  e3 < 4.193e-06 and n_dr_0p05_0p1 > 3
        + 0.001073352 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +0.1%  sj3_dr_min > 0.1278
        - 0.0009157005 * max(0.0, Q.e3 - 0.0005116989) / 1.444524e-05   # -0.1%  e3 > 0.0005117
        - 0.0008066618 * max(0.0, 0.05464922 - Q.girth) * max(0.0, Q.max_dr - 0.1326115) / 0.0001423127   # -0.1%  girth < 0.05465 and max_dr > 0.1326
        - 0.000674627 * max(0.0, Q.lam2 - 0.000537286) * max(0.0, Q.n_dr_0p05_0p1 - 1.0) / 0.0004357558   # -0.1%  lam2 > 0.0005373 and n_dr_0p05_0p1 > 1
        - 0.0005138157 * max(0.0, Q.e3 - 0.0001869378) * max(0.0, 31.85938 - Q.pt_7) / 5.959986e-05   # -0.1%  e3 > 0.0001869 and pt_7 < 31.86
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 15.51;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.50963 * (0.2425054
        + 0.1661424 * Q.lam1 / 0.00591636   # +16.6%  lam1
        - 0.138855 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # -13.9%  girth < 0.1484
        - 0.06328398 * Q.e2 / 0.02863215   # -6.3%  e2
        - 0.05831317 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -5.8%  sj2_dr > 0.1592
        - 0.0580593 * max(0.0, Q.LHA - 0.1967397) / 0.07109091   # -5.8%  LHA > 0.1967
        + 0.04500429 * max(0.0, Q.lam2 - 0.0001947983) / 0.0004461515   # +4.5%  lam2 > 0.0001948
        + 0.03555505 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # +3.6%  sj2_dr > 0.1295
        + 0.03344514 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +3.3%  sj2_dr > 0.1683
        + 0.03021051 * max(0.0, Q.tau1 - 0.04369778) / 0.031989   # +3.0%  tau1 > 0.0437
        - 0.02567002 * max(0.0, 0.06810151 - Q.z_7) / 0.01877013   # -2.6%  z_7 < 0.0681
        - 0.02373693 * max(0.0, Q.lam1 - 0.0008722282) / 0.00523234   # -2.4%  lam1 > 0.0008722
        - 0.02195809 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # -2.2%  lam1 < 0.003377
        + 0.01898657 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # +1.9%  C2_b2 < 0.009033
        + 0.01800913 * max(0.0, 0.06810151 - Q.z_7) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.0007057412   # +1.8%  z_7 < 0.0681 and centroid_offset < 0.0499
        - 0.01773098 * max(0.0, Q.lam1 - 0.00483998) / 0.002931468   # -1.8%  lam1 > 0.00484
        - 0.01500667 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, 0.2229947 - Q.dr_7) / 0.0002186438   # -1.5%  lam1 < 0.003377 and dr_7 < 0.223
        - 0.01424211 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.6947818 - Q.planar_flow) / 0.0166918   # -1.4%  sj2_dr > 0.1683 and planar_flow < 0.6948
        - 0.0133613 * max(0.0, Q.lam1 - 0.00483998) * max(0.0, 0.5042684 - Q.sj3_pairmin_over_m) / 0.0008624329   # -1.3%  lam1 > 0.00484 and sj3_pairmin_over_m < 0.5043
        + 0.01145483 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +1.1%  n_dr_0p2_0p4 > 1
        + 0.01136354 * max(0.0, 0.2669656 - Q.D2_b2) / 0.07454565   # +1.1%  D2_b2 < 0.267
        + 0.01119871 * max(0.0, 0.002915531 - Q.girth2_top3) / 0.001178811   # +1.1%  girth2_top3 < 0.002916
        + 0.01112621 * max(0.0, Q.pt_7 - 20.125) / 15.07003   # +1.1%  pt_7 > 20.12
        - 0.01062625 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.planar_flow - 0.06126205) / 0.0002819968   # -1.1%  lam2 > 0.0001948 and planar_flow > 0.06126
        - 0.009875537 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -1.0%  log_sum_pt > 6.67
        + 0.009769077 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.158545   # +1.0%  n_dr_0p05_0p1 < 5
        - 0.008956797 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 4.721224 - Q.D2_b2) / 0.5862342   # -0.9%  n_dr_0p2_0p4 > 1 and D2_b2 < 4.721
        - 0.008676526 * max(0.0, Q.pt_7 - 20.125) * max(0.0, 0.2669656 - Q.D2_b2) / 1.268076   # -0.9%  pt_7 > 20.12 and D2_b2 < 0.267
        - 0.008400384 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -0.8%  lam2 > 0.003408
        - 0.008277052 * max(0.0, 48.03125 - Q.pt_6) / 9.943776   # -0.8%  pt_6 < 48.03
        - 0.008157296 * max(0.0, Q.LHA - 0.3127275) * max(0.0, 0.6947818 - Q.planar_flow) / 0.005887138   # -0.8%  LHA > 0.3127 and planar_flow < 0.6948
        - 0.008084154 * max(0.0, Q.LHA - 0.3127275) / 0.01533444   # -0.8%  LHA > 0.3127
        + 0.006866952 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # +0.7%  e3 > 8.148e-05
        - 0.006797515 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, 6.538559 - Q.log_sum_pt) / 0.0001231869   # -0.7%  lam2 > 0.0001948 and log_sum_pt < 6.539
        - 0.005766296 * max(0.0, 0.02563286 - Q.M2) * max(0.0, 0.1818564 - Q.z_2) / 0.0001032987   # -0.6%  M2 < 0.02563 and z_2 < 0.1819
        - 0.00554336 * max(0.0, 48.03125 - Q.pt_6) * max(0.0, 6.46415 - Q.log_sum_pt) / 0.7570358   # -0.6%  pt_6 < 48.03 and log_sum_pt < 6.464
        - 0.005231751 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.380911 - Q.D2_b2) / 0.005863346   # -0.5%  sj2_dr > 0.1592 and D2_b2 < 0.3809
        + 0.005019557 * max(0.0, 1.002471 - Q.D2) / 0.1871081   # +0.5%  D2 < 1.002
        - 0.004474248 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.eccentricity - 0.4829602) / 8.513077e-05   # -0.4%  lam2 > 0.0001948 and eccentricity > 0.483
        + 0.004181202 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 2.0 - Q.n_dr_0p1_0p2) / 0.06887712   # +0.4%  log_sum_pt > 6.67 and n_dr_0p1_0p2 < 2
        + 0.0035078 * max(0.0, 0.02563286 - Q.M2) * max(0.0, 118.5 - Q.pt_2) / 0.04960343   # +0.4%  M2 < 0.02563 and pt_2 < 118.5
        + 0.003133541 * Q.mean_phi / 0.01077348   # +0.3%  mean_phi
        - 0.003015833 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.02644207 - Q.mean_eta) / 0.000997125   # -0.3%  sj2_dr > 0.1683 and mean_eta < 0.02644
        + 0.002922466 * max(0.0, Q.lam2 - 0.003408389) * max(0.0, 111.75 - Q.pt_2) / 0.006488771   # +0.3%  lam2 > 0.003408 and pt_2 < 111.8
        + 0.002607479 * max(0.0, 48.03125 - Q.pt_6) * max(0.0, 0.01705377 - Q.zdr_0) / 0.0620657   # +0.3%  pt_6 < 48.03 and zdr_0 < 0.01705
        + 0.00220261 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +0.2%  sj2_dr > 0.3004
        + 0.002049264 * max(0.0, Q.pt_7 - 20.125) * max(0.0, 0.07530049 - Q.dr_5) / 0.4143559   # +0.2%  pt_7 > 20.12 and dr_5 < 0.0753
        - 0.002022772 * max(0.0, Q.e3 - 0.0005116989) / 1.444524e-05   # -0.2%  e3 > 0.0005117
        + 0.001927854 * max(0.0, Q.centroid_offset - 0.02355416) / 0.003991673   # +0.2%  centroid_offset > 0.02355
        - 0.001767465 * max(0.0, Q.LHA - 0.1967397) * max(0.0, Q.sj3_dr23 - 0.05931259) / 0.009608229   # -0.2%  LHA > 0.1967 and sj3_dr23 > 0.05931
        - 0.001597993 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, 0.01265416 - Q.zdr_7) / 1.587918e-06   # -0.2%  lam2 > 0.0001948 and zdr_7 < 0.01265
        + 0.001545825 * max(0.0, 48.03125 - Q.pt_6) * max(0.0, 0.1236856 - Q.D2_b2) / 0.1641634   # +0.2%  pt_6 < 48.03 and D2_b2 < 0.1237
        + 0.001345531 * max(0.0, 0.04581318 - Q.M3) / 0.002043736   # +0.1%  M3 < 0.04581
        + 0.001333473 * max(0.0, Q.lam2 - 0.003408389) * max(0.0, Q.pt2_over_pt0 - 0.1852611) / 6.009766e-05   # +0.1%  lam2 > 0.003408 and pt2_over_pt0 > 0.1853
        + 0.0006611304 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.04702759 - Q.abseta_3) / 0.1514747   # +0.1%  sum_pt > 988.4 and abseta_3 < 0.04703
        + 0.0006496758 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.1%  sum_pt > 988.4
        - 0.0002913625 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 45.05938 - Q.pt_3) / 0.009462343   # -0.0%  sj3_dr_min > 0.1278 and pt_3 < 45.06
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 19.83;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.82884 * (-0.02305259
        + 0.1994037 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +19.9%  lam1 < 0.012
        + 0.07921551 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 0.001130645 - Q.lam2) / 1.136697e-05   # +7.9%  lam1 < 0.01643 and lam2 < 0.001131
        - 0.06927993 * max(0.0, 0.04447357 - Q.e2) / 0.02019334   # -6.9%  e2 < 0.04447
        - 0.06343067 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -6.3%  girth < 0.0717
        + 0.05934516 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +5.9%  planar_flow < 0.2534
        - 0.04309769 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # -4.3%  sj3_dr_max > 0.179
        - 0.04214795 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, 0.02685622 - Q.centroid_offset) / 2.007754e-05   # -4.2%  lam1 < 0.003377 and centroid_offset < 0.02686
        - 0.03716117 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -3.7%  sj2_dr > 0.1592
        + 0.03543419 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +3.5%  tau1 < 0.07284
        + 0.03450542 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # +3.5%  sj2_dr > 0.1295
        + 0.03368351 * max(0.0, Q.sj3_dr_max - 0.1426152) / 0.06279029   # +3.4%  sj3_dr_max > 0.1426
        + 0.0288673 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +2.9%  centroid_offset < 0.01838
        - 0.02645342 * max(0.0, 0.06108601 - Q.girth) / 0.01868685   # -2.6%  girth < 0.06109
        - 0.01728225 * max(0.0, 45.75 - Q.pt_7) / 12.14632   # -1.7%  pt_7 < 45.75
        - 0.01503999 * max(0.0, 7.330536e-06 - Q.e3) / 2.382852e-06   # -1.5%  e3 < 7.331e-06
        + 0.014784 * max(0.0, Q.sum_pt_top5 - 506.875) / 121.3278   # +1.5%  sum_pt_top5 > 506.9
        - 0.01448372 * max(0.0, Q.sum_pt_top5 - 506.875) * max(0.0, 2.371297e-05 - Q.e3) / 0.001928824   # -1.4%  sum_pt_top5 > 506.9 and e3 < 2.371e-05
        + 0.01197673 * max(0.0, 0.05028464 - Q.e2) / 0.02502152   # +1.2%  e2 < 0.05028
        - 0.01139558 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 3.916009 - Q.D3) / 0.2321659   # -1.1%  planar_flow < 0.2534 and D3 < 3.916
        - 0.009346846 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -0.9%  lam2 < 0.0005373
        - 0.009062929 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 937.0312 - Q.sum_pt) / 1.062777   # -0.9%  centroid_offset < 0.01838 and sum_pt < 937
        + 0.008157124 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.050302e-05 - Q.e3) / 5.099048e-05   # +0.8%  pt_7 < 45.75 and e3 < 1.05e-05
        - 0.008009972 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 0.716559 - Q.D2_b2) / 0.0002593059   # -0.8%  lam1 < 0.005954 and D2_b2 < 0.7166
        + 0.007828888 * max(0.0, 0.01133547 - Q.tau2) / 0.003675653   # +0.8%  tau2 < 0.01134
        - 0.00753085 * max(0.0, 0.002127561 - Q.mean_phi2) / 0.0009769232   # -0.8%  mean_phi2 < 0.002128
        + 0.007501992 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # +0.8%  lam1 < 0.003377
        + 0.007417771 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.7%  sj2_dr > 0.2688
        - 0.006812974 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.centroid_offset - 0.01837778) / 3.28605e-05   # -0.7%  lam1 < 0.01643 and centroid_offset > 0.01838
        - 0.006790948 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.centroid_offset - 0.01627885) / 2.281746e-05   # -0.7%  lam1 < 0.012 and centroid_offset > 0.01628
        - 0.00678402 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.003652679   # -0.7%  planar_flow < 0.2534 and centroid_offset < 0.0499
        - 0.006612887 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 9.296551   # -0.7%  pt_7 < 45.75 and n_dr_0p2_0p4 < 1
        + 0.00660945 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.716559 - Q.D2_b2) / 0.0007061152   # +0.7%  lam1 < 0.008376 and D2_b2 < 0.7166
        - 0.006122724 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 0.380911 - Q.D2_b2) / 0.001095249   # -0.6%  lam1 < 0.01643 and D2_b2 < 0.3809
        - 0.005869199 * max(0.0, 0.1357675 - Q.tau21) / 0.01683052   # -0.6%  tau21 < 0.1358
        - 0.005707263 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.6%  planar_flow < 0.2534 and sum_pt < 840
        + 0.0056172 * max(0.0, 0.002915531 - Q.girth2_top3) / 0.001178811   # +0.6%  girth2_top3 < 0.002916
        - 0.005125596 * max(0.0, Q.centroid_offset - 0.02355416) / 0.003991673   # -0.5%  centroid_offset > 0.02355
        - 0.004687859 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, 0.2534037 - Q.planar_flow) / 4.073218e-05   # -0.5%  lam1 < 0.003377 and planar_flow < 0.2534
        + 0.004349088 * max(0.0, Q.sum_pt_top5 - 506.875) * max(0.0, 0.380911 - Q.D2_b2) / 13.07354   # +0.4%  sum_pt_top5 > 506.9 and D2_b2 < 0.3809
        - 0.003271011 * max(0.0, 33.6875 - Q.pt_6) * max(0.0, 4.721224 - Q.D2_b2) / 5.679374   # -0.3%  pt_6 < 33.69 and D2_b2 < 4.721
        + 0.003197305 * max(0.0, Q.sj2_dr - 0.2414351) * max(0.0, 739.5 - Q.sum_pt) / 2.049171   # +0.3%  sj2_dr > 0.2414 and sum_pt < 739.5
        - 0.003046581 * max(0.0, Q.sj2_dr - 0.2414351) / 0.01315894   # -0.3%  sj2_dr > 0.2414
        - 0.002807686 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 6.423044 - Q.log_sum_pt) / 0.0004075136   # -0.3%  lam1 < 0.01643 and log_sum_pt < 6.423
        + 0.002530769 * max(0.0, 45.75 - Q.pt_7) * max(0.0, Q.centroid_offset - 0.01096064) / 0.1020192   # +0.3%  pt_7 < 45.75 and centroid_offset > 0.01096
        - 0.002064949 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, Q.eccentricity - 0.927072) / 0.001136932   # -0.2%  sj2_dr > 0.1779 and eccentricity > 0.9271
        - 0.002024284 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.345805 - Q.D2_b2) / 7.233479   # -0.2%  pt_7 < 45.75 and D2_b2 < 1.346
        + 0.001741304 * max(0.0, 7.330536e-06 - Q.e3) * max(0.0, 0.716559 - Q.D2_b2) / 1.072852e-07   # +0.2%  e3 < 7.331e-06 and D2_b2 < 0.7166
        + 0.001328071 * max(0.0, 0.06108601 - Q.girth) * max(0.0, Q.dr1_7 - 0.2025074) / 5.186613e-05   # +0.1%  girth < 0.06109 and dr1_7 > 0.2025
        + 0.001251276 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +0.1%  lam1 < 0.01643
        - 0.001092274 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.dr1_7 - 0.2411619) / 1.071397e-05   # -0.1%  lam1 < 0.012 and dr1_7 > 0.2412
        + 0.0009740777 * max(0.0, Q.n_dr_0p05_0p1 - 6.0) / 0.1402639   # +0.1%  n_dr_0p05_0p1 > 6
        - 0.0009218544 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, Q.C3 - 0.04473419) / 6.06784e-07   # -0.1%  lam2 < 0.0005373 and C3 > 0.04473
        - 0.0005386099 * max(0.0, Q.sum_pt_top5 - 506.875) * max(0.0, Q.mean_phi - 0.01746433) / 0.03493588   # -0.1%  sum_pt_top5 > 506.9 and mean_phi > 0.01746
        - 0.0002765108 * max(0.0, 0.004918231 - Q.zdr_0) * max(0.0, Q.dr0_7 - 0.09118326) / 2.337194e-06   # -0.0%  zdr_0 < 0.004918 and dr0_7 > 0.09118
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 1.435;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.435044 * (-0.2751372
        + 0.245975 * max(0.0, Q.girth - 0.1245537) / 0.002632136   # +24.6%  girth > 0.1246
        - 0.163142 * max(0.0, Q.sd_rg - 0.324646) * max(0.0, 6.896095 - Q.log_sum_pt) / 0.001091861   # -16.3%  sd_rg > 0.3246 and log_sum_pt < 6.896
        - 0.1596275 * max(0.0, Q.LHA - 0.3855647) / 0.004130434   # -16.0%  LHA > 0.3856
        + 0.1018691 * max(0.0, Q.sd_rg - 0.324646) * max(0.0, 6.701242 - Q.log_sum_pt) / 0.0007568975   # +10.2%  sd_rg > 0.3246 and log_sum_pt < 6.701
        + 0.07287097 * max(0.0, Q.sd_rg - 0.324646) / 0.001748965   # +7.3%  sd_rg > 0.3246
        - 0.04333791 * max(0.0, Q.girth - 0.1245537) * max(0.0, 62.25 - Q.pt_6) / 0.06207836   # -4.3%  girth > 0.1246 and pt_6 < 62.25
        - 0.04037748 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -4.0%  e2 > 0.06344
        + 0.03274403 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # +3.3%  centroid_offset > 0.03776
        + 0.03204829 * max(0.0, Q.girth - 0.1484084) / 0.0008957407   # +3.2%  girth > 0.1484
        + 0.02904806 * max(0.0, Q.n_dr_0p2_0p4 - 2.0) / 0.05970756   # +2.9%  n_dr_0p2_0p4 > 2
        + 0.01887724 * max(0.0, Q.girth - 0.1245537) * max(0.0, Q.C2_b2 - 0.009032972) / 2.778481e-05   # +1.9%  girth > 0.1246 and C2_b2 > 0.009033
        - 0.0125278 * max(0.0, Q.zdr_0 - 0.03981924) / 0.0005695785   # -1.3%  zdr_0 > 0.03982
        + 0.01146711 * max(0.0, Q.girth - 0.1245537) * max(0.0, Q.log_sum_pt - 6.502799) / 3.140724e-05   # +1.1%  girth > 0.1246 and log_sum_pt > 6.503
        - 0.009619384 * max(0.0, Q.centroid_offset - 0.03776099) * max(0.0, 615.875 - Q.sum_pt) / 0.1794415   # -1.0%  centroid_offset > 0.03776 and sum_pt < 615.9
        - 0.008549027 * max(0.0, Q.n_dr_0p2_0p4 - 2.0) * max(0.0, 0.003408389 - Q.lam2) / 9.478981e-05   # -0.9%  n_dr_0p2_0p4 > 2 and lam2 < 0.003408
        - 0.008334785 * max(0.0, Q.n_dr_0p2_0p4 - 2.0) * max(0.0, 667.0063 - Q.sum_pt) / 6.802776   # -0.8%  n_dr_0p2_0p4 > 2 and sum_pt < 667
        - 0.006414884 * max(0.0, Q.girth - 0.1484084) * max(0.0, 588.5859 - Q.sum_pt) / 0.08773343   # -0.6%  girth > 0.1484 and sum_pt < 588.6
        - 0.003169413 * max(0.0, Q.girth - 0.1484084) * max(0.0, Q.log_sum_pt - 6.502799) / 8.799927e-06   # -0.3%  girth > 0.1484 and log_sum_pt > 6.503
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 21.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.64151 * (0.01379162
        + 0.3439342 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # +34.4%  girth < 0.1484
        + 0.1019732 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +10.2%  lam1 < 0.01643
        - 0.0756958 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -7.6%  girth < 0.1484 and log_sum_pt < 6.804
        - 0.06916229 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -6.9%  tau1 < 0.1136
        - 0.04497575 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -4.5%  e2 < 0.08001
        - 0.04016832 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 1.399607e-06   # -4.0%  e3 < 8.148e-05 and centroid_offset < 0.03776
        - 0.02526058 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # -2.5%  C2_b2 < 0.009033
        - 0.02523641 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.z_dr_0_0p05 - 0.3658817) / 0.004662966   # -2.5%  lam1 < 0.01643 and z_dr_0_0p05 > 0.3659
        - 0.02484756 * max(0.0, Q.pt_7 - 31.85938) / 5.945918   # -2.5%  pt_7 > 31.86
        - 0.02079341 * max(0.0, 48.71875 - Q.pt_7) / 14.7465   # -2.1%  pt_7 < 48.72
        + 0.01983973 * max(0.0, Q.z_7 - 0.04624032) / 0.01204993   # +2.0%  z_7 > 0.04624
        - 0.01964236 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.0003061234 - Q.lam2) / 2.082734e-05   # -2.0%  girth < 0.1484 and lam2 < 0.0003061
        + 0.01960705 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +2.0%  tau1 < 0.09539
        - 0.01778372 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -1.8%  girth < 0.1484 and pt_7 < 38.53
        + 0.01506249 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 40.04062 - Q.pt_7) / 652.1541   # +1.5%  sum_pt_top5 > 687.4 and pt_7 < 40.04
        - 0.01481157 * max(0.0, 50.25 - Q.pt_6) / 11.66482   # -1.5%  pt_6 < 50.25
        - 0.01239992 * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.09151624   # -1.2%  sj3_dr_min < 0.1278
        + 0.01233156 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +1.2%  e3 < 8.148e-05
        + 0.01187359 * max(0.0, Q.z_dr_0_0p05 - 0.6080732) / 0.1760045   # +1.2%  z_dr_0_0p05 > 0.6081
        - 0.007299873 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -0.7%  log_sum_pt > 6.67
        - 0.006965573 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 25.57812 - Q.pt_7) / 0.01870322   # -0.7%  lam1 < 0.01643 and pt_7 < 25.58
        - 0.006243476 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -0.6%  sj3_dr_max > 0.2337
        - 0.005768967 * max(0.0, Q.z_7 - 0.0586137) * max(0.0, 31.125 - Q.pt_4) / 0.0001667469   # -0.6%  z_7 > 0.05861 and pt_4 < 31.12
        + 0.00448781 * max(0.0, Q.pt_7 - 31.85938) * max(0.0, 0.01228369 - Q.zdr_1) / 0.03012054   # +0.4%  pt_7 > 31.86 and zdr_1 < 0.01228
        + 0.00422048 * max(0.0, 50.25 - Q.pt_6) * max(0.0, Q.zdr_0 - 0.001113887) / 0.1706944   # +0.4%  pt_6 < 50.25 and zdr_0 > 0.001114
        - 0.003981114 * max(0.0, Q.z_7 - 0.04624032) * max(0.0, 0.01084112 - Q.zdr_1) / 3.654417e-05   # -0.4%  z_7 > 0.04624 and zdr_1 < 0.01084
        + 0.003809725 * max(0.0, Q.z_7 - 0.0586137) / 0.00587069   # +0.4%  z_7 > 0.05861
        + 0.003780162 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +0.4%  sum_pt_top5 > 791.1
        + 0.003766452 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.4%  sum_pt_top5 > 687.4
        - 0.003306638 * max(0.0, 27.57812 - Q.pt_6) / 0.9875435   # -0.3%  pt_6 < 27.58
        - 0.003051968 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # -0.3%  pt_5 < 24.58
        + 0.002864426 * max(0.0, Q.pt_7 - 31.85938) * max(0.0, 0.1598486 - Q.max_dr) / 0.4085666   # +0.3%  pt_7 > 31.86 and max_dr < 0.1598
        + 0.00283628 * max(0.0, Q.pt_7 - 31.85938) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1) / 2.82555   # +0.3%  pt_7 > 31.86 and z_dr_0p05_0p1 < 0.7509
        + 0.002667362 * max(0.0, 0.08000524 - Q.e2) * max(0.0, 60.03125 - Q.pt_4) / 0.4322666   # +0.3%  e2 < 0.08001 and pt_4 < 60.03
        - 0.002554436 * max(0.0, Q.z_top5 - 0.8770155) / 0.00720125   # -0.3%  z_top5 > 0.877
        + 0.002494663 * max(0.0, 0.0294431 - Q.M2) / 0.002929467   # +0.2%  M2 < 0.02944
        + 0.002194535 * max(0.0, 6.46415 - Q.log_sum_pt) * max(0.0, 0.01084112 - Q.zdr_1) / 0.0001493841   # +0.2%  log_sum_pt < 6.464 and zdr_1 < 0.01084
        - 0.001808571 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.02788929 - Q.zdr_1) / 0.1073153   # -0.2%  sum_pt > 988.4 and zdr_1 < 0.02789
        + 0.001669927 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # +0.2%  z_5 < 0.02818
        - 0.001526298 * max(0.0, 61.53125 - Q.pt_2) / 1.459365   # -0.2%  pt_2 < 61.53
        + 0.001456967 * max(0.0, 0.1136369 - Q.tau1) * max(0.0, 69.875 - Q.pt_2) / 0.1415216   # +0.1%  tau1 < 0.1136 and pt_2 < 69.88
        - 0.00140339 * max(0.0, 6.46415 - Q.log_sum_pt) * max(0.0, 29.875 - Q.pt_5) / 0.02086227   # -0.1%  log_sum_pt < 6.464 and pt_5 < 29.88
        + 0.001102699 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.08525808 - Q.dr_2) / 2.250996   # +0.1%  sum_pt_top5 > 687.4 and dr_2 < 0.08526
        - 0.0009910312 * max(0.0, Q.z_7 - 0.0586137) * max(0.0, Q.lam2 - 4.64158e-05) / 5.481848e-06   # -0.1%  z_7 > 0.05861 and lam2 > 4.642e-05
        - 0.0008026163 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 0.0005193389   # -0.1%  sj3_dr_max > 0.2337 and z_dr_0p05_0p1 > 0.6748
        + 0.0005690781 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # +0.1%  sum_pt_top5 > 840
        - 0.0004077149 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.07092092 - Q.M3) / 0.05338926   # -0.0%  sum_pt > 988.4 and M3 < 0.07092
        + 0.0003053824 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.z_6 - 0.03932388) / 0.02599753   # +0.0%  sum_pt > 988.4 and z_6 > 0.03932
        - 0.000262871 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, Q.zdr_1 - 0.007459436) / 0.01546861   # -0.0%  sum_pt_top5 > 791.1 and zdr_1 > 0.007459
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 40;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 40.00023 * (-0.1141981
        + 0.07809697 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +7.8%  lam1 < 0.012
        - 0.0717448 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # -7.2%  lam1 < 0.002464
        + 0.06859711 * max(0.0, 0.1667615 - Q.sd_rg) / 0.07405603   # +6.9%  sd_rg < 0.1668
        + 0.05838978 * max(0.0, Q.girth - 0.02689598) / 0.03613842   # +5.8%  girth > 0.0269
        + 0.0540469 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +5.4%  lam1 < 0.01643
        - 0.04900291 * max(0.0, 0.1416226 - Q.tau1) / 0.08190733   # -4.9%  tau1 < 0.1416
        + 0.0480931 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +4.8%  e2 < 0.04111
        - 0.03634141 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -3.6%  lam1 < 0.006507
        - 0.03529256 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -3.5%  lam1 < 0.00733
        + 0.03526146 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +3.5%  tau1 < 0.09539
        - 0.03201845 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -3.2%  centroid_offset > 0.0499
        - 0.03097192 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # -3.1%  lam2 < 0.003408
        - 0.02991984 * max(0.0, 0.1773029 - Q.sd_rg) / 0.08115541   # -3.0%  sd_rg < 0.1773
        + 0.02985173 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.sd_rg - 0.1572969) / 6.238901e-05   # +3.0%  lam2 < 0.003408 and sd_rg > 0.1573
        - 0.02515978 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -2.5%  e3 < 8.148e-05
        - 0.02461966 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 1.0 - Q.psi_0p3) / 0.0007331701   # -2.5%  z_dr_0p05_0p1 > 0.1639 and psi_0p3 < 1
        - 0.02275836 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -2.3%  girth > 0.08724
        + 0.02114224 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.LHA - 0.1767241) / 0.0002170323   # +2.1%  lam2 < 0.003408 and LHA > 0.1767
        - 0.01624698 * max(0.0, 0.2330919 - Q.sd_rg) / 0.1256101   # -1.6%  sd_rg < 0.2331
        + 0.0160012 * max(0.0, 0.1115136 - Q.planar_flow) / 0.03343187   # +1.6%  planar_flow < 0.1115
        + 0.01595308 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.1644126   # +1.6%  z_dr_0p05_0p1 > 0.1639 and n_dr_0p2_0p4 < 1
        - 0.01522344 * max(0.0, Q.girth - 0.08065885) / 0.009330634   # -1.5%  girth > 0.08066
        - 0.0145141 * max(0.0, Q.sd_rg - 0.1876504) / 0.01921774   # -1.5%  sd_rg > 0.1877
        - 0.01359 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -1.4%  tau1 < 0.1136
        - 0.01325071 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.158545   # -1.3%  n_dr_0p05_0p1 < 5
        - 0.01312283 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -1.3%  planar_flow < 0.1115 and lam1 < 0.00733
        + 0.0130597 * max(0.0, 0.03556091 - Q.e2) / 0.01371972   # +1.3%  e2 < 0.03556
        + 0.0116331 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +1.2%  LHA > 0.3033
        - 0.008979749 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 0.20552 - Q.z_dr_0p2_0p4) / 0.03609844   # -0.9%  z_dr_0p05_0p1 > 0.1639 and z_dr_0p2_0p4 < 0.2055
        - 0.008736784 * max(0.0, Q.sd_rg - 0.1572969) / 0.02905066   # -0.9%  sd_rg > 0.1573
        - 0.008716208 * max(0.0, 0.06545715 - Q.sd_rg) / 0.02200467   # -0.9%  sd_rg < 0.06546
        - 0.007011298 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 1.0 - Q.sd_nremoved) / 0.02680832   # -0.7%  planar_flow < 0.1115 and sd_nremoved < 1
        + 0.006442879 * max(0.0, 6.0 - Q.n_dr_0p05_0p1) / 4.008323   # +0.6%  n_dr_0p05_0p1 < 6
        - 0.006423939 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) / 0.1950558   # -0.6%  z_dr_0p05_0p1 > 0.1639
        - 0.005255507 * max(0.0, 0.1116471 - Q.sd_rg) / 0.04385005   # -0.5%  sd_rg < 0.1116
        - 0.004930971 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # -0.5%  e3 < 5.335e-05
        - 0.004737511 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.sd_rg - 0.2037854) / 2.928331e-05   # -0.5%  lam2 < 0.003408 and sd_rg > 0.2038
        + 0.004366364 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.3724174   # +0.4%  z_dr_0p05_0p1 < 0.5883
        + 0.00345826 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00595415 - Q.lam1) / 3.161575e-05   # +0.3%  planar_flow < 0.1115 and lam1 < 0.005954
        - 0.003434306 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.centroid_offset - 0.02685622) / 6.831551e-06   # -0.3%  lam2 < 0.003408 and centroid_offset > 0.02686
        + 0.003121711 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 0.04510668 - Q.z_dr_0p1_0p2) / 0.01057962   # +0.3%  z_dr_0p05_0p1 < 0.5883 and z_dr_0p1_0p2 < 0.04511
        + 0.002985163 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # +0.3%  planar_flow < 0.1115 and lam1 < 0.01643
        - 0.002980297 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.3%  psi_0p1 > 0.9761
        - 0.002946408 * max(0.0, Q.sd_rg - 0.1449048) / 0.03437807   # -0.3%  sd_rg > 0.1449
        - 0.002863222 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # -0.3%  girth2_top3 < 0.002152
        - 0.002562321 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02685622) / 1.516228e-05   # -0.3%  lam1 < 0.01643 and centroid_offset > 0.02686
        - 0.002364317 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 6.670067 - Q.log_sum_pt) / 0.03946963   # -0.2%  z_dr_0p05_0p1 > 0.1639 and log_sum_pt < 6.67
        - 0.002246463 * max(0.0, 0.01289969 - Q.e2) / 0.002527207   # -0.2%  e2 < 0.0129
        - 0.002060014 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 763.825 - Q.sum_pt) / 2.369641   # -0.2%  planar_flow < 0.1115 and sum_pt < 763.8
        + 0.001554321 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.z_dr_0p2_0p4 - 0.0) / 5.139675e-05   # +0.2%  lam2 < 0.003408 and z_dr_0p2_0p4 > 0
        + 0.001231722 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 0.415246 - Q.D2) / 0.0001335382   # +0.1%  lam1 < 0.01643 and D2 < 0.4152
        - 0.001130219 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.415246 - Q.D2) / 1.113844e-05   # -0.1%  lam1 < 0.00733 and D2 < 0.4152
        + 0.001061922 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.2832687 - Q.sj2_zsoft) / 0.002541531   # +0.1%  planar_flow < 0.1115 and sj2_zsoft < 0.2833
        + 0.00102154 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.1974628) / 1.23895e-05   # +0.1%  lam1 < 0.006507 and sj3_dr23 > 0.1975
        + 0.0009671977 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, Q.centroid_offset - 0.03117077) / 1.256127e-06   # +0.1%  lam1 < 0.006507 and centroid_offset > 0.03117
        - 0.0009161986 * max(0.0, 0.1136369 - Q.tau1) * max(0.0, 0.415246 - Q.D2) / 0.0002136968   # -0.1%  tau1 < 0.1136 and D2 < 0.4152
        + 0.0007664177 * max(0.0, 0.04110972 - Q.e2) * max(0.0, 0.415246 - Q.D2) / 2.89594e-05   # +0.1%  e2 < 0.04111 and D2 < 0.4152
        - 0.0005908606 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.0002706389   # -0.1%  lam2 < 0.003408 and n_dr_0p2_0p4 > 1
        - 0.0002317835 * max(0.0, 0.06545715 - Q.sd_rg) * max(0.0, 1.0 - Q.psi_0p3) / 3.393963e-05   # -0.0%  sd_rg < 0.06546 and psi_0p3 < 1
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 26.69;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.68595 * (-0.1058336
        + 0.1221861 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +12.2%  lam1 < 0.01643
        - 0.1014368 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -10.1%  girth < 0.0717
        - 0.07381704 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -7.4%  lam1 < 0.00733
        + 0.06184254 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +6.2%  lam2 < 0.003408
        - 0.05673759 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -5.7%  girth < 0.08724
        + 0.04553762 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +4.6%  LHA < 0.3033
        + 0.04378592 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +4.4%  e2 < 0.04111
        - 0.03317683 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # -3.3%  lam1 < 0.012
        + 0.02989294 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +3.0%  lam1 < 0.002464
        - 0.026952 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -2.7%  lam1 < 0.008376
        - 0.02642702 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # -2.6%  LHA < 0.3127
        + 0.02313106 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # +2.3%  lam1 < 0.005433
        + 0.02231651 * max(0.0, Q.psi_0p1 - 0.7143804) / 0.1805043   # +2.2%  psi_0p1 > 0.7144
        + 0.02173278 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # +2.2%  girth2_top3 < 0.002152
        + 0.01896884 * max(0.0, 0.032347 - Q.e2) / 0.01171807   # +1.9%  e2 < 0.03235
        + 0.01695621 * max(0.0, 0.007639643 - Q.girth2_top2) / 0.004543329   # +1.7%  girth2_top2 < 0.00764
        - 0.01594327 * max(0.0, Q.tau1 - 0.1416226) / 0.003662393   # -1.6%  tau1 > 0.1416
        + 0.01586932 * max(0.0, Q.tau1 - 0.08865369) / 0.01243546   # +1.6%  tau1 > 0.08865
        + 0.01572961 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # +1.6%  girth < 0.1019
        - 0.01459171 * max(0.0, 0.03360421 - Q.girth) / 0.006496997   # -1.5%  girth < 0.0336
        + 0.0110353 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.1%  tau1 < 0.05357
        - 0.01078382 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1872617 - Q.sj2_dr) / 0.0007896743   # -1.1%  N2 < 0.2233 and sj2_dr < 0.1873
        + 0.01023274 * max(0.0, Q.tau1 - 0.1027642) * max(0.0, 0.5327104 - Q.D2_b2) / 0.001862765   # +1.0%  tau1 > 0.1028 and D2_b2 < 0.5327
        - 0.009721667 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.lam1 - 0.008375572) / 0.0001169809   # -1.0%  N2 < 0.2233 and lam1 > 0.008376
        + 0.009641616 * max(0.0, Q.eccentricity - 0.9458207) * max(0.0, Q.sum_pt_top5 - 367.5938) / 5.10997   # +1.0%  eccentricity > 0.9458 and sum_pt_top5 > 367.6
        + 0.008733912 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.2931906) / 0.001536651   # +0.9%  N2 < 0.2233 and LHA > 0.2932
        - 0.008437671 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 0.0002261574   # -0.8%  lam1 < 0.008376 and D2 < 0.8757
        - 0.007463406 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 0.01849752 - Q.zdr_1) / 3.216563e-05   # -0.7%  lam2 < 0.003408 and zdr_1 < 0.0185
        - 0.007103793 * max(0.0, 0.009233892 - Q.zdr_0) / 0.002026168   # -0.7%  zdr_0 < 0.009234
        - 0.006646988 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3255822) / 0.0007869707   # -0.7%  N2 < 0.2233 and LHA > 0.3256
        - 0.006513215 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -0.7%  lam1 < 0.006507
        - 0.006471289 * max(0.0, Q.tau1 - 0.08865369) * max(0.0, 0.5327104 - Q.D2_b2) / 0.002910551   # -0.6%  tau1 > 0.08865 and D2_b2 < 0.5327
        - 0.006042436 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.4235925) / 7.321856e-05   # -0.6%  N2 < 0.2233 and LHA > 0.4236
        - 0.00599184 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -0.6%  zdr_0 < 0.02118
        + 0.00598912 * max(0.0, 0.1515405 - Q.z_dr_0_0p05) / 0.0478325   # +0.6%  z_dr_0_0p05 < 0.1515
        - 0.005936128 * max(0.0, 7.330536e-06 - Q.e3) / 2.382852e-06   # -0.6%  e3 < 7.331e-06
        + 0.005848071 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +0.6%  tau1 > 0.1136
        + 0.005781087 * max(0.0, Q.tau1 - 0.1416226) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0006707461   # +0.6%  tau1 > 0.1416 and D2_b2 < 0.5327
        + 0.005612746 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1591713 - Q.sj2_dr) / 0.0002571871   # +0.6%  N2 < 0.2233 and sj2_dr < 0.1592
        - 0.00556832 * max(0.0, Q.tau1 - 0.09538712) / 0.01057268   # -0.6%  tau1 > 0.09539
        - 0.005273522 * max(0.0, 0.002151568 - Q.girth2_top3) * max(0.0, 0.1950135 - Q.planar_flow) / 2.579988e-05   # -0.5%  girth2_top3 < 0.002152 and planar_flow < 0.195
        - 0.005107828 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # -0.5%  tau1 > 0.1028
        - 0.004989176 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # -0.5%  N2 < 0.2233
        + 0.004530395 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 402.625) / 10.05301   # +0.5%  N2 < 0.2233 and sum_pt_top5 > 402.6
        + 0.004150435 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 0.001116597   # +0.4%  lam1 < 0.01643 and D2 < 0.8757
        - 0.003958446 * max(0.0, 0.1329373 - Q.LHA) / 0.007753684   # -0.4%  LHA < 0.1329
        - 0.003312548 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.centroid_offset - 0.03776099) / 5.89929e-06   # -0.3%  lam1 < 0.01643 and centroid_offset > 0.03776
        + 0.003194481 * max(0.0, Q.sj3_dr13 - 0.181053) / 0.02525513   # +0.3%  sj3_dr13 > 0.1811
        + 0.003010404 * max(0.0, Q.eccentricity - 0.9458207) / 0.02147681   # +0.3%  eccentricity > 0.9458
        - 0.002824867 * max(0.0, Q.sj3_dr13 - 0.249546) / 0.01061611   # -0.3%  sj3_dr13 > 0.2495
        + 0.002750559 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.1974628) / 2.285306e-05   # +0.3%  lam1 < 0.008376 and sj3_dr23 > 0.1975
        - 0.002736366 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # -0.3%  z_dr_0p05_0p1 > 0.7509
        + 0.002673453 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # +0.3%  lam1 < 0.006507 and D2 < 0.8757
        - 0.002155247 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # -0.2%  tau1 < 0.0437
        - 0.002116252 * max(0.0, Q.eccentricity - 0.9458207) * max(0.0, Q.mean_phi - -0.01753483) / 0.0004008253   # -0.2%  eccentricity > 0.9458 and mean_phi > -0.01753
        - 0.001677149 * max(0.0, Q.sj3_dr13 - 0.2871004) / 0.006026879   # -0.2%  sj3_dr13 > 0.2871
        - 0.001651044 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3855647) / 0.0002330351   # -0.2%  N2 < 0.2233 and LHA > 0.3856
        + 0.001195212 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, Q.centroid_offset - 0.03776099) / 7.468637e-07   # +0.1%  lam1 < 0.00733 and centroid_offset > 0.03776
        - 0.001026414 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, Q.sj3_dr23 - 0.1797097) / 0.0003211663   # -0.1%  z_dr_0p05_0p1 > 0.7509 and sj3_dr23 > 0.1797
        + 0.0009294598 * max(0.0, 0.009233892 - Q.zdr_0) * max(0.0, 0.716559 - Q.D2_b2) / 9.532153e-05   # +0.1%  zdr_0 < 0.009234 and D2_b2 < 0.7166
        + 0.0008917655 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, 0.2213841 - Q.D3) / 1.704383e-05   # +0.1%  girth2_top2 < 0.00764 and D3 < 0.2214
        - 0.0008135355 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.C2_b2 - 0.009032972) / 3.381092e-06   # -0.1%  lam1 < 0.01643 and C2_b2 > 0.009033
        - 0.0008133929 * max(0.0, 0.2213841 - Q.D3) / 0.002871295   # -0.1%  D3 < 0.2214
        - 0.000592292 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.1974628) / 9.808733e-06   # -0.1%  lam1 < 0.005954 and sj3_dr23 > 0.1975
        - 0.0005725305 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.centroid_offset - 0.04990367) / 2.174774e-06   # -0.1%  lam1 < 0.01643 and centroid_offset > 0.0499
        - 0.0004663182 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.02320757 - Q.z_7) / 1.527561e-05   # -0.0%  N2 < 0.2233 and z_7 < 0.02321
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0206300420168066, 0.7228649159663866, 2.0818220588235294, 0.9449350840336135, 1.130813025210084, 1.7531737394957982, 0.9338039915966386, 1.8341323529411764, 0.2410966386554622, 2.981253361344538, 2.016842962184874, 2.1330581932773107, 0.09510042016806723, 3.290354831932773, 0.49433718487394956, 0.3430640756302521]
T = [2.314971184020483, 1.3504935152967439, 3.3302265116202725, 2.45963389738708, 2.840429441307773]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +39%, n9 +22%, n5 -14%, n1 +12%, n0 -7%, n6 +4% ...
            + 0.3864121 * h[2] / H_AVG[2]
            + 0.2213431 * h[9] / H_AVG[9]
            - 0.1419975 * h[5] / H_AVG[5]
            + 0.1219752 * h[1] / H_AVG[1]
            - 0.06888787 * h[0] / H_AVG[0]
            + 0.04411926 * h[6] / H_AVG[6]
            - 0.01526494 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -19%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5605054 * h[9] / H_AVG[9]
            - 0.1866765 * h[10] / H_AVG[10]
            + 0.08643174 * h[6] / H_AVG[6]
            - 0.07849999 * h[4] / H_AVG[4]
            + 0.06085184 * h[5] / H_AVG[5]
            + 0.01587679 * h[15] / H_AVG[15]
            + 0.0111578 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +12%, n14 -11%, n0 +11%, n6 -9% ...
            + 0.2401929 * h[11] / H_AVG[11]
            - 0.1418725 * h[3] / H_AVG[3]
            + 0.1204772 * h[7] / H_AVG[7]
            - 0.1113296 * h[14] / H_AVG[14]
            + 0.1053507 * h[0] / H_AVG[0]
            - 0.0876258 * h[6] / H_AVG[6]
            - 0.07082298 * h[15] / H_AVG[15]
            + 0.06947067 * h[13] / H_AVG[13]
            - 0.02797532 * h[9] / H_AVG[9]
            - 0.01809912 * h[8] / H_AVG[8]
            - 0.006783181 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -22%, n6 -14%, n14 +8%, n13 +7%, n9 -4% ...
            + 0.3495437 * h[7] / H_AVG[7]
            - 0.2160996 * h[3] / H_AVG[3]
            - 0.1423694 * h[6] / H_AVG[6]
            + 0.07536749 * h[14] / H_AVG[14]
            + 0.07315775 * h[13] / H_AVG[13]
            - 0.03787725 * h[9] / H_AVG[9]
            + 0.03673641 * h[1] / H_AVG[1]
            + 0.03591785 * h[4] / H_AVG[4]
            - 0.02179339 * h[15] / H_AVG[15]
            + 0.01113716 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +27%, n5 -15%, n4 +5%, n3 +2%, n12 -2% ...
            - 0.4706002 * h[13] / H_AVG[13]
            + 0.2662682 * h[10] / H_AVG[10]
            - 0.1543053 * h[5] / H_AVG[5]
            + 0.04976418 * h[4] / H_AVG[4]
            + 0.02079208 * h[3] / H_AVG[3]
            - 0.0167405 * h[12] / H_AVG[12]
            + 0.01591507 * h[8] / H_AVG[8]
            + 0.005614413 * h[0] / H_AVG[0]
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
