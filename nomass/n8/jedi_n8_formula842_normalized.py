"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  9:  11.8%   (on for 70% of jets)
  neuron  7:  10.2%   (on for 61% of jets)
  neuron  3:   8.7%   (on for 30% of jets)
  neuron 10:   8.2%   (on for 80% of jets)
  neuron  2:   7.3%   (on for 88% of jets)
  neuron  5:   7.1%   (on for 62% of jets)
  neuron  6:   7.0%   (on for 51% of jets)
  neuron 11:   6.5%   (on for 71% of jets)
  neuron 14:   4.5%   (on for 31% of jets)
  neuron  0:   4.3%   (on for 44% of jets)
  neuron  1:   3.2%   (on for 57% of jets)
  neuron  4:   3.0%   (on for 42% of jets)
  neuron 15:   2.6%   (on for 30% of jets)
  neuron  8:   1.0%   (on for 35% of jets)
  neuron 12:   0.4%   (on for 8% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.9% (the network: 65.8%); same class as the network for 88.6% of jets.

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
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z1                 pT share of subjet 1 of 3 (by pT)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.pt1_over_pt0           pT1 / pT0
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.z_3rd                  3rd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_6               |Δη| of particle 6
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_2               |Δφ| of particle 2
  Q.absphi_5               |Δφ| of particle 5
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_5                  ΔR between particle 5 and the hardest particle
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_5                  Δη of particle 5
  Q.phi_0                  Δφ of particle 0
  Q.phi_1                  Δφ of particle 1
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
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
        max_dr=max(dr[i] for i in real),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        n_for_90pct=ncum(0.9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z1=subjets(3)["z"][0],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        z_3rd=zs[2],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        abseta_0=abs(eta[0]),
        abseta_4=abs(eta[4]),
        abseta_6=abs(eta[6]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_2=abs(phi[2]),
        absphi_5=abs(phi[5]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_5=math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_5=eta[5],
        phi_0=phi[0],
        phi_1=phi[1],
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 35.68;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.6799 * (0.202647
        - 0.2140116 * Q.log_sum_pt / 6.542932   # -21.4%  log_sum_pt
        + 0.1030077 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +10.3%  lam1_plus_lam2 < 0.008678
        - 0.07234743 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -7.2%  lam1 < 0.008376
        + 0.06824808 * max(0.0, 0.01882765 - Q.sum_z_dr2) / 0.01314296   # +6.8%  sum_z_dr2 < 0.01883
        + 0.05435015 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +5.4%  sum_z_dr2 < 0.01324
        + 0.0371592 * max(0.0, 0.1116471 - Q.sd_rg) / 0.04385005   # +3.7%  sd_rg < 0.1116
        - 0.03712234 * max(0.0, 0.005590289 - Q.sum_z_dr2) / 0.002229311   # -3.7%  sum_z_dr2 < 0.00559
        - 0.03448602 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -3.4%  sum_z_dr < 0.08724
        - 0.03113782 * max(0.0, 0.07608178 - Q.sum_z_dr) / 0.02796784   # -3.1%  sum_z_dr < 0.07608
        - 0.03095055 * max(0.0, 0.1773029 - Q.sd_rg) / 0.08115541   # -3.1%  sd_rg < 0.1773
        - 0.02410017 * max(0.0, 0.00718279 - Q.sum_zz_dr2) / 0.003528787   # -2.4%  sum_zz_dr2 < 0.007183
        - 0.0199171 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.001184249   # -2.0%  lam1_plus_lam2 < 0.003563
        - 0.0161366 * max(0.0, 0.007929074 - Q.sum_z_dr2_top3) / 0.004560262   # -1.6%  sum_z_dr2_top3 < 0.007929
        + 0.01578571 * max(0.0, 631.275 - Q.sum_pt_top5) / 94.34106   # +1.6%  sum_pt_top5 < 631.3
        - 0.01573043 * max(0.0, 788.4484 - Q.sum_pt) / 112.1576   # -1.6%  sum_pt < 788.4
        + 0.01436268 * max(0.0, 0.06345984 - Q.tau1) / 0.02211428   # +1.4%  tau1 < 0.06346
        + 0.0134158 * max(0.0, 0.032347 - Q.e2) / 0.01171807   # +1.3%  e2 < 0.03235
        + 0.01294199 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +1.3%  e3 < 8.148e-05
        - 0.0126752 * max(0.0, 739.5 - Q.sum_pt) * max(0.0, 1.345805 - Q.D2_b2) / 59.18221   # -1.3%  sum_pt < 739.5 and D2_b2 < 1.346
        + 0.01265107 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # +1.3%  sum_zz_dr2 < 0.01166
        - 0.01227897 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -1.2%  sum_pt < 739.5
        - 0.01194774 * max(0.0, 0.01882765 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.009480685) / 8.306781e-05   # -1.2%  sum_z_dr2 < 0.01883 and centroid_offset > 0.009481
        - 0.0118899 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -1.2%  centroid_offset > 0.0499
        - 0.01181457 * max(0.0, 0.06545715 - Q.sd_rg) / 0.02200467   # -1.2%  sd_rg < 0.06546
        + 0.008822166 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +0.9%  planar_flow < 0.1484
        + 0.008806049 * max(0.0, Q.n_dr_0_0p05 - 5.0) / 1.093146   # +0.9%  n_dr_0_0p05 > 5
        + 0.008482153 * max(0.0, 631.275 - Q.sum_pt_top5) * max(0.0, 1.345805 - Q.D2_b2) / 69.0494   # +0.8%  sum_pt_top5 < 631.3 and D2_b2 < 1.346
        - 0.006428278 * max(0.0, 0.01357153 - Q.tau2) / 0.005299336   # -0.6%  tau2 < 0.01357
        + 0.006346825 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +0.6%  lam1 < 0.005954
        - 0.005774184 * max(0.0, 0.01517359 - Q.sum_z_dr) / 0.001389803   # -0.6%  sum_z_dr < 0.01517
        + 0.005766285 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.001732318   # +0.6%  planar_flow < 0.1484 and centroid_offset < 0.0499
        - 0.005558818 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 0.2669656 - Q.D2_b2) / 3.951293e-06   # -0.6%  e3 < 8.148e-05 and D2_b2 < 0.267
        + 0.00555659 * max(0.0, 0.01882765 - Q.sum_z_dr2) * max(0.0, 0.09029177 - Q.M3) / 0.0001995224   # +0.6%  sum_z_dr2 < 0.01883 and M3 < 0.09029
        + 0.005431237 * max(0.0, 0.004007842 - Q.sum_z_dr2_top2) / 0.001907211   # +0.5%  sum_z_dr2_top2 < 0.004008
        - 0.004595773 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -0.5%  centroid_offset < 0.01838
        - 0.003778654 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.0402832 - Q.phi_0) / 0.0001765572   # -0.4%  lam1 < 0.008376 and phi_0 < 0.04028
        + 0.00349189 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 1.129616 - Q.D2_b2) / 0.04715373   # +0.3%  planar_flow < 0.1484 and D2_b2 < 1.13
        - 0.00346869 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) * max(0.0, 0.0425415 - Q.phi_1) / 5.045407e-05   # -0.3%  lam1_plus_lam2 < 0.003563 and phi_1 < 0.04254
        - 0.003456679 * max(0.0, Q.pt_6 - 35.28125) / 7.833466   # -0.3%  pt_6 > 35.28
        - 0.003450061 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, Q.eccentricity - 0.9598562) / 1.260811   # -0.3%  sum_pt < 788.4 and eccentricity > 0.9599
        - 0.002632397 * max(0.0, 0.380911 - Q.D2_b2) * max(0.0, Q.pt_5 - 24.57812) / 3.196263   # -0.3%  D2_b2 < 0.3809 and pt_5 > 24.58
        + 0.002616781 * max(0.0, Q.eccentricity - 0.9884745) * max(0.0, 0.001109927 - Q.C2_b2) / 1.494669e-06   # +0.3%  eccentricity > 0.9885 and C2_b2 < 0.00111
        - 0.002551177 * max(0.0, 0.005590289 - Q.sum_z_dr2) * max(0.0, 0.09310137 - Q.M3) / 3.230359e-05   # -0.3%  sum_z_dr2 < 0.00559 and M3 < 0.0931
        + 0.002070191 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, Q.M2 - 0.02563286) / 4.402579   # +0.2%  sum_pt < 788.4 and M2 > 0.02563
        - 0.002042308 * max(0.0, Q.eccentricity - 0.9884745) / 0.001961982   # -0.2%  eccentricity > 0.9885
        - 0.0019605 * max(0.0, 739.5 - Q.sum_pt) * max(0.0, Q.dr0_6 - 0.1979161) / 1.730861   # -0.2%  sum_pt < 739.5 and dr0_6 > 0.1979
        - 0.001628112 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.08753082 - Q.M3) / 0.0006875892   # -0.2%  planar_flow < 0.1484 and M3 < 0.08753
        + 0.00160027 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +0.2%  z_7 < 0.02807
        + 0.001255662 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, 0.380911 - Q.D2_b2) / 12.71118   # +0.1%  sum_pt < 788.4 and D2_b2 < 0.3809
        - 0.0007933624 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) * max(0.0, Q.eccentricity - 0.9598562) / 6.567391e-06   # -0.1%  lam1_plus_lam2 < 0.004372 and eccentricity > 0.9599
        + 0.000787909 * max(0.0, 739.5 - Q.sum_pt) * max(0.0, Q.mean_eta - 0.009367547) / 0.4149784   # +0.1%  sum_pt < 739.5 and mean_eta > 0.009368
        - 0.0007805495 * max(0.0, 0.006390125 - Q.sum_zz_dr2) * max(0.0, Q.ptdr0_2 - 3.351396) / 0.001639884   # -0.1%  sum_zz_dr2 < 0.00639 and ptdr0_2 > 3.351
        - 0.0003697104 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.dr0_7 - 0.2405707) / 1.062493e-07   # -0.0%  e3 < 8.148e-05 and dr0_7 > 0.2406
        + 0.0003589038 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) * max(0.0, Q.dr0_7 - 0.1786203) / 1.674303e-06   # +0.0%  lam1_plus_lam2 < 0.003563 and dr0_7 > 0.1786
        + 0.0003027599 * max(0.0, 0.005590289 - Q.sum_z_dr2) * max(0.0, Q.dr0_5 - 0.171998) / 2.694851e-06   # +0.0%  sum_z_dr2 < 0.00559 and dr0_5 > 0.172
        + 0.0002874854 * max(0.0, 0.06344108 - Q.e2) / 0.03657078   # +0.0%  e2 < 0.06344
        - 0.0002787672 * max(0.0, 0.06345984 - Q.tau1) * max(0.0, Q.dr_5 - 0.1631114) / 3.584581e-05   # -0.0%  tau1 < 0.06346 and dr_5 > 0.1631
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 20.06;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.05687 * (0.08818298
        - 0.08364308 * max(0.0, 0.00609665 - Q.sum_z_dr2) / 0.002544039   # -8.4%  sum_z_dr2 < 0.006097
        - 0.06673543 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -6.7%  sum_pt_top5 > 531.2
        + 0.0641767 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1442881 - Q.dr_0) / 0.0005125595   # +6.4%  lam1 < 0.008376 and dr_0 < 0.1443
        - 0.06163835 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # -6.2%  lam1_plus_lam2 < 0.008678
        - 0.05447301 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -5.4%  sum_z_dr < 0.1019
        + 0.052494 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +5.2%  log_sum_pt > 6.378
        + 0.04172501 * max(0.0, Q.sum_pt - 667.0063) / 97.02637   # +4.2%  sum_pt > 667
        - 0.04040173 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -4.0%  z_7 < 0.06165
        + 0.03585227 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +3.6%  lam1 < 0.005954
        - 0.0333249 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # -3.3%  sj3_dr_max < 0.3456
        + 0.032124 * max(0.0, 0.1594012 - Q.sj3_dr_max) / 0.04422261   # +3.2%  sj3_dr_max < 0.1594
        + 0.03164252 * max(0.0, 0.07608178 - Q.sum_z_dr) / 0.02796784   # +3.2%  sum_z_dr < 0.07608
        - 0.02911356 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.08082334 - Q.dr_0) / 0.01037756   # -2.9%  log_sum_pt > 6.378 and dr_0 < 0.08082
        - 0.0269989 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -2.7%  lam1 < 0.008376
        + 0.02575991 * max(0.0, 0.005834489 - Q.sum_zz_dr2) / 0.002579453   # +2.6%  sum_zz_dr2 < 0.005834
        + 0.02481624 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # +2.5%  tau1 < 0.1028
        + 0.02462475 * max(0.0, Q.log_sum_pt - 6.572938) / 0.08631512   # +2.5%  log_sum_pt > 6.573
        + 0.02439956 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.0001869378 - Q.e3) / 2.398316e-06   # +2.4%  z_7 < 0.06165 and e3 < 0.0001869
        + 0.02094614 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.01403324 - Q.sum_z_dr2_top2) / 0.0001680113   # +2.1%  z_7 < 0.06165 and sum_z_dr2_top2 < 0.01403
        - 0.01920406 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.pt_5 - 24.57812) / 0.1029929   # -1.9%  lam1 < 0.008376 and pt_5 > 24.58
        + 0.01412639 * max(0.0, 0.01403324 - Q.sum_z_dr2_top2) / 0.01007219   # +1.4%  sum_z_dr2_top2 < 0.01403
        - 0.01381511 * max(0.0, 0.1326115 - Q.max_dr) / 0.03990327   # -1.4%  max_dr < 0.1326
        - 0.0132164 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # -1.3%  sum_pt < 988.4
        + 0.01294745 * max(0.0, Q.pt_7 - 38.53125) / 2.781159   # +1.3%  pt_7 > 38.53
        + 0.01250225 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.03981924 - Q.zdr_0) / 0.0003895962   # +1.3%  z_7 < 0.06165 and zdr_0 < 0.03982
        - 0.01203158 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.007929074 - Q.sum_z_dr2_top3) / 0.0005408911   # -1.2%  log_sum_pt > 6.573 and sum_z_dr2_top3 < 0.007929
        + 0.01121765 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 0.2691019 - Q.tau21_b2) / 0.0004590397   # +1.1%  lam2 < 0.003408 and tau21_b2 < 0.2691
        - 0.01080236 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -1.1%  zdr_0 < 0.02118
        - 0.01070826 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -1.1%  centroid_offset < 0.02077
        - 0.01026917 * max(0.0, 0.05240025 - Q.z_7) / 0.008882498   # -1.0%  z_7 < 0.0524
        - 0.008927884 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.1410336 - Q.dr01) / 0.001621893   # -0.9%  z_7 < 0.06165 and dr01 < 0.141
        - 0.006333851 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 2.955458e-05 - Q.e3) / 7.137805e-05   # -0.6%  pt_7 > 34.53 and e3 < 2.955e-05
        + 0.004966683 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # +0.5%  lam1 < 0.008376 and centroid_offset > 0.02077
        + 0.004921232 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +0.5%  lam1 < 0.002464
        + 0.004893476 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7681386 - Q.z_dr_0_0p05) / 0.0392858   # +0.5%  log_sum_pt > 6.378 and z_dr_0_0p05 < 0.7681
        - 0.004858825 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) * max(0.0, 0.1484197 - Q.planar_flow) / 0.0001479746   # -0.5%  lam1_plus_lam2 < 0.008678 and planar_flow < 0.1484
        + 0.004840264 * max(0.0, Q.log_sum_pt - 6.638339) * max(0.0, 0.1410336 - Q.dr01) / 0.006766763   # +0.5%  log_sum_pt > 6.638 and dr01 < 0.141
        - 0.003915781 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.008654951 - Q.zdr_6) / 0.0005705688   # -0.4%  log_sum_pt > 6.573 and zdr_6 < 0.008655
        + 0.003766117 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # +0.4%  pt_7 > 34.53 and tau2 < 0.01713
        + 0.003629167 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +0.4%  tau1 < 0.07284
        + 0.003582631 * max(0.0, Q.z_6 - 0.06727211) / 0.005920115   # +0.4%  z_6 > 0.06727
        - 0.003511006 * max(0.0, 0.1027642 - Q.tau1) * max(0.0, Q.mean_phi - -0.01753483) / 0.0008830797   # -0.4%  tau1 < 0.1028 and mean_phi > -0.01753
        - 0.003119592 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # -0.3%  e3 > 0.0001869
        + 0.003028068 * max(0.0, Q.log_sum_pt - 6.638339) / 0.05708669   # +0.3%  log_sum_pt > 6.638
        - 0.002816158 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.06994629 - Q.abseta_4) / 0.004093636   # -0.3%  log_sum_pt > 6.573 and abseta_4 < 0.06995
        - 0.002721469 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.04875823) / 0.002321449   # -0.3%  z_7 < 0.06165 and z_dr_0p05_0p1 > 0.04876
        - 0.002604913 * max(0.0, Q.pt_7 - 38.53125) * max(0.0, 0.007782684 - Q.zdr_7) / 0.01134668   # -0.3%  pt_7 > 38.53 and zdr_7 < 0.007783
        + 0.001899295 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.zdr_1 - 0.005533857) / 0.02081763   # +0.2%  pt_7 > 34.53 and zdr_1 > 0.005534
        - 0.001871 * max(0.0, 0.05240025 - Q.z_7) * max(0.0, 0.1236856 - Q.D2_b2) / 0.000152235   # -0.2%  z_7 < 0.0524 and D2_b2 < 0.1237
        - 0.001773909 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.001387159   # -0.2%  lam1 < 0.008376 and n_pt_above_50 > 6
        - 0.001638828 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 77.625 - Q.pt_3) / 25.26458   # -0.2%  pt_7 > 34.53 and pt_3 < 77.62
        + 0.001500857 * max(0.0, Q.log_sum_pt - 6.638339) * max(0.0, 0.01817981 - Q.tau21_b2) / 0.0001689858   # +0.2%  log_sum_pt > 6.638 and tau21_b2 < 0.01818
        + 0.00113031 * max(0.0, Q.e3 - 0.0005116989) / 1.444524e-05   # +0.1%  e3 > 0.0005117
        - 0.0009369755 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.1%  pt_7 > 53.44
        - 0.0006788841 * max(0.0, Q.pt_6 - 62.25) / 0.3835962   # -0.1%  pt_6 > 62.25
        - 0.0004020796 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, Q.zdr_1 - 0.001590562) / 0.001619476   # -0.0%  pt_7 > 53.44 and zdr_1 > 0.001591
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 24.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.00908 * (0.03379677
        + 0.147517 * Q.pt_7 / 34.64819   # +14.8%  pt_7
        - 0.1359288 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # -13.6%  z_7 > 0.02321
        - 0.07588801 * max(0.0, Q.LHA - 0.111565) / 0.1352185   # -7.6%  LHA > 0.1116
        - 0.07207937 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -7.2%  pt_7 < 53.44
        + 0.066549 * max(0.0, 840.0195 - Q.sum_pt) / 148.4153   # +6.7%  sum_pt < 840
        + 0.0439664 * max(0.0, Q.LHA - 0.09323897) / 0.1512502   # +4.4%  LHA > 0.09324
        + 0.04243725 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +4.2%  lam1_plus_lam2 < 0.01324
        - 0.03885803 * max(0.0, 0.0423228 - Q.C2) * max(0.0, 0.02415398 - Q.C2_b2) / 0.0004792616   # -3.9%  C2 < 0.04232 and C2_b2 < 0.02415
        - 0.03632146 * max(0.0, 0.00718279 - Q.sum_zz_dr2) / 0.003528787   # -3.6%  sum_zz_dr2 < 0.007183
        + 0.0338635 * max(0.0, 0.0423228 - Q.C2) / 0.02012665   # +3.4%  C2 < 0.04232
        + 0.0269841 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.2507612 - Q.max_dr) / 0.00062784   # +2.7%  lam1 < 0.00733 and max_dr < 0.2508
        - 0.02536677 * max(0.0, 6.701242 - Q.log_sum_pt) * max(0.0, 1.129616 - Q.D2_b2) / 0.1081022   # -2.5%  log_sum_pt < 6.701 and D2_b2 < 1.13
        + 0.02388174 * max(0.0, 0.007520088 - Q.lam1_plus_lam2) / 0.003544991   # +2.4%  lam1_plus_lam2 < 0.00752
        + 0.02167519 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 1.129616 - Q.D2_b2) / 0.07303201   # +2.2%  log_sum_pt < 6.606 and D2_b2 < 1.13
        + 0.02063371 * max(0.0, 0.005590289 - Q.sum_z_dr2) / 0.002229311   # +2.1%  sum_z_dr2 < 0.00559
        - 0.02054729 * max(0.0, 716.8828 - Q.sum_pt_top5) / 152.5573   # -2.1%  sum_pt_top5 < 716.9
        + 0.01608127 * max(0.0, 6.701242 - Q.log_sum_pt) / 0.1935491   # +1.6%  log_sum_pt < 6.701
        + 0.01536445 * max(0.0, 763.825 - Q.sum_pt) / 96.68366   # +1.5%  sum_pt < 763.8
        - 0.01144792 * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.09151624   # -1.1%  sj3_dr_min < 0.1278
        - 0.01121929 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # -1.1%  sum_zz_dr2 < 0.003013
        + 0.009351836 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # +0.9%  lam1 < 0.003377
        + 0.007695471 * max(0.0, 0.001653836 - Q.lam1_plus_lam2) / 0.0004289067   # +0.8%  lam1_plus_lam2 < 0.001654
        + 0.007428321 * max(0.0, Q.z_7 - 0.02807091) / 0.02541111   # +0.7%  z_7 > 0.02807
        - 0.006782627 * max(0.0, 0.04447357 - Q.e2) / 0.02019334   # -0.7%  e2 < 0.04447
        - 0.006361009 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 62.25 - Q.pt_6) / 0.07984241   # -0.6%  lam1 < 0.00733 and pt_6 < 62.25
        + 0.005967962 * max(0.0, 0.02383244 - Q.zdr_0) / 0.01109706   # +0.6%  zdr_0 < 0.02383
        - 0.00586976 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -0.6%  sj2_dr < 0.1295
        + 0.005534361 * max(0.0, 615.875 - Q.sum_pt) / 30.07766   # +0.6%  sum_pt < 615.9
        + 0.005254941 * max(0.0, Q.sum_pt - 937.0312) / 8.087955   # +0.5%  sum_pt > 937
        - 0.004952538 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.5%  sum_pt > 988.4
        + 0.00459092 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.5%  log_sum_pt > 6.896
        + 0.004539869 * max(0.0, Q.z_7 - 0.05557716) / 0.00714993   # +0.5%  z_7 > 0.05558
        + 0.004508272 * max(0.0, Q.z_7 - 0.02807091) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.0007742112   # +0.5%  z_7 > 0.02807 and centroid_offset < 0.0499
        - 0.004045788 * max(0.0, 0.001101266 - Q.sum_zz_dr2) / 0.000308004   # -0.4%  sum_zz_dr2 < 0.001101
        - 0.003548242 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # -0.4%  log_sum_pt < 6.606
        - 0.003114431 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 0.01425934 - Q.abseta_0) / 0.1176352   # -0.3%  sum_pt_top5 > 791.1 and abseta_0 < 0.01426
        + 0.003039744 * max(0.0, 53.4375 - Q.pt_7) * max(0.0, 0.08499387 - Q.D2_b2) / 0.1962679   # +0.3%  pt_7 < 53.44 and D2_b2 < 0.08499
        - 0.002580125 * max(0.0, 0.08499387 - Q.D2_b2) / 0.01226862   # -0.3%  D2_b2 < 0.08499
        + 0.002390954 * max(0.0, 20.125 - Q.pt_7) / 0.5468342   # +0.2%  pt_7 < 20.12
        - 0.002325918 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # -0.2%  sum_z_dr < 0.007674
        - 0.002024026 * max(0.0, 0.06081235 - Q.z_6) * max(0.0, 0.07720947 - Q.absphi_2) / 0.0005288824   # -0.2%  z_6 < 0.06081 and absphi_2 < 0.07721
        - 0.001855171 * max(0.0, 0.005576073 - Q.centroid_offset) * max(0.0, 0.01685631 - Q.C3) / 3.692635e-06   # -0.2%  centroid_offset < 0.005576 and C3 < 0.01686
        - 0.001792655 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # -0.2%  sum_pt_top5 > 791.1
        + 0.001655744 * max(0.0, 367.5938 - Q.sum_pt_top5) / 4.984642   # +0.2%  sum_pt_top5 < 367.6
        + 0.001276011 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 1.129616 - Q.D2_b2) / 3.947956   # +0.1%  sum_pt_top5 > 791.1 and D2_b2 < 1.13
        + 0.001181995 * max(0.0, 0.007673833 - Q.sum_z_dr) * max(0.0, 75.625 - Q.pt_4) / 0.004511241   # +0.1%  sum_z_dr < 0.007674 and pt_4 < 75.62
        + 0.001175977 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.01425934 - Q.abseta_0) / 0.03515712   # +0.1%  sum_pt > 988.4 and abseta_0 < 0.01426
        - 0.0007998763 * max(0.0, Q.sum_pt - 937.0312) * max(0.0, Q.n_pt_above_50 - 5.0) / 6.962984   # -0.1%  sum_pt > 937 and n_pt_above_50 > 5
        - 0.0006389164 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 1.59265 - Q.D2_b2) / 0.00192904   # -0.1%  log_sum_pt > 6.896 and D2_b2 < 1.593
        - 0.0005744869 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.9419776 - Q.ptdr0_2) / 0.0009724569   # -0.1%  log_sum_pt > 6.896 and ptdr0_2 < 0.942
        + 0.0005314627 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.9419776 - Q.ptdr0_2) / 1.052477   # +0.1%  sum_pt > 988.4 and ptdr0_2 < 0.942
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 13.03;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.03437 * (-0.06184045
        - 0.1549428 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # -15.5%  sum_z_dr2 > 0.008678
        + 0.1437397 * max(0.0, Q.sum_zz_dr2 - 0.00718279) / 0.002233568   # +14.4%  sum_zz_dr2 > 0.007183
        + 0.1234254 * max(0.0, Q.lam1_plus_lam2 - 0.00609665) / 0.002892623   # +12.3%  lam1_plus_lam2 > 0.006097
        - 0.09949681 * max(0.0, Q.sum_zz_dr2 - 0.006390125) / 0.002450953   # -9.9%  sum_zz_dr2 > 0.00639
        + 0.0643473 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +6.4%  lam1 > 0.008376
        + 0.04296632 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # +4.3%  sj2_dr > 0.1779
        - 0.03618592 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -3.6%  sj2_dr > 0.1592
        - 0.02888864 * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 0.001433269   # -2.9%  sum_zz_dr2 > 0.01166
        + 0.02797752 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, Q.eccentricity - 0.9458207) / 0.000763705   # +2.8%  sj2_dr > 0.1779 and eccentricity > 0.9458
        - 0.02668746 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # -2.7%  lam1_plus_lam2 > 0.01324
        - 0.02532968 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.eccentricity - 0.9458207) / 0.0008867862   # -2.5%  sj2_dr > 0.1683 and eccentricity > 0.9458
        + 0.02290218 * max(0.0, Q.lam1 - 0.002464291) / 0.004201699   # +2.3%  lam1 > 0.002464
        + 0.02071046 * max(0.0, Q.sum_z_dr - 0.06663269) / 0.01393206   # +2.1%  sum_z_dr > 0.06663
        - 0.01803997 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 7.0 - Q.n_dr_0p05_0p1) / 0.1780556   # -1.8%  sj2_dr > 0.1683 and n_dr_0p05_0p1 < 7
        + 0.01541694 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # +1.5%  sum_z_dr > 0.08724
        - 0.01359018 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # -1.4%  sj2_dr > 0.2688
        + 0.01342108 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +1.3%  sj2_dr > 0.1683
        - 0.01206413 * max(0.0, Q.tau1 - 0.0811449) / 0.01489378   # -1.2%  tau1 > 0.08114
        + 0.01152292 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +1.2%  tau1 > 0.1136
        + 0.01133163 * max(0.0, Q.max_dr - 0.1117619) / 0.04033009   # +1.1%  max_dr > 0.1118
        - 0.01125137 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -1.1%  lam1 > 0.012
        + 0.01013964 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 988.4078 - Q.sum_pt) / 12.22843   # +1.0%  sj2_dr > 0.1683 and sum_pt < 988.4
        + 0.01005329 * max(0.0, Q.lam2 - 0.000537286) / 0.0003825703   # +1.0%  lam2 > 0.0005373
        + 0.007492537 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.06297984 - Q.tau2) / 0.001263954   # +0.7%  sj2_dr > 0.1683 and tau2 < 0.06298
        - 0.007375268 * max(0.0, Q.lam1_plus_lam2 - 0.00609665) * max(0.0, Q.log_sum_pt - 6.192222) / 0.0004573947   # -0.7%  lam1_plus_lam2 > 0.006097 and log_sum_pt > 6.192
        - 0.007183695 * max(0.0, Q.lam1_plus_lam2 - 0.00609665) * max(0.0, Q.n_pt_above_50 - 3.0) / 0.004753839   # -0.7%  lam1_plus_lam2 > 0.006097 and n_pt_above_50 > 3
        + 0.004068967 * max(0.0, Q.lam1_plus_lam2 - 0.01882765) / 0.0007605369   # +0.4%  lam1_plus_lam2 > 0.01883
        + 0.003826033 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 0.07848552   # +0.4%  sj2_dr > 0.1592 and n_dr_0p1_0p2 > 1
        + 0.00376348 * max(0.0, Q.sum_z_dr - 0.08723651) * max(0.0, 0.0003061234 - Q.lam2) / 3.733963e-07   # +0.4%  sum_z_dr > 0.08724 and lam2 < 0.0003061
        + 0.003500061 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +0.4%  sj2_dr > 0.3004
        + 0.002971191 * max(0.0, -0.006779839 - Q.mean_eta) / 0.003173922   # +0.3%  mean_eta < -0.00678
        - 0.002793613 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001864148 - Q.sum_z_dr2_top2) / 7.226227e-06   # -0.3%  sj2_dr > 0.1779 and sum_z_dr2_top2 < 0.001864
        + 0.002254904 * max(0.0, Q.lam1_plus_lam2 - 0.01882765) * max(0.0, Q.n_pt_above_50 - 3.0) / 0.001116483   # +0.2%  lam1_plus_lam2 > 0.01883 and n_pt_above_50 > 3
        - 0.002152103 * max(0.0, Q.lam1_plus_lam2 - 0.00609665) * max(0.0, Q.pt_6 - 31.90625) / 0.02357333   # -0.2%  lam1_plus_lam2 > 0.006097 and pt_6 > 31.91
        + 0.001900308 * max(0.0, Q.sum_zz_dr2 - 0.01165737) * max(0.0, 0.07148865 - Q.z_7) / 1.35438e-05   # +0.2%  sum_zz_dr2 > 0.01166 and z_7 < 0.07149
        - 0.00179347 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9884745) / 1.823692e-06   # -0.2%  lam1_plus_lam2 > 0.01324 and eccentricity > 0.9885
        + 0.001286356 * max(0.0, Q.sj2_dr - 0.2687922) * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) / 1.542054e-05   # +0.1%  sj2_dr > 0.2688 and sum_z_dr2_top2 < 0.00764
        - 0.001269729 * max(0.0, -0.006779839 - Q.mean_eta) * max(0.0, -0.03967285 - Q.eta_0) / 9.048956e-05   # -0.1%  mean_eta < -0.00678 and eta_0 < -0.03967
        - 0.001187404 * max(0.0, Q.sj2_dr - 0.3003793) * max(0.0, Q.z_dr_0p05_0p1 - 0.08878489) / 0.001005291   # -0.1%  sj2_dr > 0.3004 and z_dr_0p05_0p1 > 0.08878
        + 0.000749533 * max(0.0, -0.006779839 - Q.mean_eta) * max(0.0, 6.267538 - Q.log_sum_pt) / 0.0001795705   # +0.1%  mean_eta < -0.00678 and log_sum_pt < 6.268
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 53.69;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 53.69425 * (-0.0250233
        + 0.1067371 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # +10.7%  sum_zz_dr2 < 0.01166
        - 0.09865792 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # -9.9%  sum_z_dr2 < 0.01324
        - 0.0768369 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -7.7%  sum_z_dr2 < 0.008678
        + 0.0529148 * max(0.0, 0.02415398 - Q.C2_b2) / 0.02132459   # +5.3%  C2_b2 < 0.02415
        + 0.05255425 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +5.3%  e3 < 0.0001869
        - 0.0492985 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -4.9%  lam2 < 0.001131
        + 0.03711109 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +3.7%  N2 < 0.2233
        - 0.0364397 * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.00293624   # -3.6%  sum_z_dr2 < 0.006679
        - 0.03287812 * max(0.0, 0.04447357 - Q.e2) / 0.02019334   # -3.3%  e2 < 0.04447
        - 0.02976449 * max(0.0, Q.sum_zz_dr2 - 0.0030133) / 0.003940214   # -3.0%  sum_zz_dr2 > 0.003013
        + 0.02961419 * max(0.0, Q.e2 - 0.02045966) / 0.01370442   # +3.0%  e2 > 0.02046
        - 0.02903026 * max(0.0, 0.01165737 - Q.sum_zz_dr2) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.0003843383   # -2.9%  sum_zz_dr2 < 0.01166 and z_dr_0p2_0p4 < 0.05644
        + 0.02532236 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +2.5%  lam1 < 0.012
        + 0.02457195 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # +2.5%  C2_b2 < 0.009033
        - 0.01998185 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 988.4078 - Q.sum_pt) / 13.77568   # -2.0%  N2 < 0.2233 and sum_pt < 988.4
        - 0.01912309 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 6.733425 - Q.log_sum_pt) / 0.02442595   # -1.9%  sj3_dr_max < 0.3012 and log_sum_pt < 6.733
        + 0.01844461 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, Q.psi_0p2 - 0.9435576) / 0.0004391185   # +1.8%  sum_z_dr2 < 0.01324 and psi_0p2 > 0.9436
        + 0.01570446 * max(0.0, Q.max_dr - 0.0931108) / 0.05062176   # +1.6%  max_dr > 0.09311
        + 0.01433352 * max(0.0, 0.006390125 - Q.sum_zz_dr2) / 0.002953508   # +1.4%  sum_zz_dr2 < 0.00639
        + 0.01281399 * max(0.0, 0.07608178 - Q.sum_z_dr) / 0.02796784   # +1.3%  sum_z_dr < 0.07608
        + 0.01239496 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) / 0.003638607   # +1.2%  sum_z_dr2_top5 < 0.007164
        + 0.01217829 * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.166579   # +1.2%  sj3_dr_min < 0.209
        - 0.01161678 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, 813.4156 - Q.sum_pt) / 0.01550241   # -1.2%  e3 < 0.0001869 and sum_pt < 813.4
        + 0.01101801 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 868.5094 - Q.sum_pt) / 8.122677   # +1.1%  N2 < 0.2233 and sum_pt < 868.5
        + 0.0106438 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # +1.1%  sum_z_dr2 < 0.002635
        - 0.01006456 * max(0.0, 0.33555 - Q.tau21) / 0.105517   # -1.0%  tau21 < 0.3356
        + 0.009455106 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 6.701242 - Q.log_sum_pt) / 0.0003664792   # +0.9%  sum_z_dr2 < 0.006679 and log_sum_pt < 6.701
        + 0.009252134 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +0.9%  sj3_dr_max > 0.179
        - 0.008997105 * max(0.0, Q.sj2_dr - 0.2179769) / 0.01834468   # -0.9%  sj2_dr > 0.218
        - 0.008339599 * max(0.0, Q.sj3_dr_max - 0.213399) / 0.03061279   # -0.8%  sj3_dr_max > 0.2134
        + 0.008289808 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +0.8%  lam1 < 0.002464
        - 0.007765728 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.4007947 - Q.planar_flow) / 0.01755893   # -0.8%  N2 < 0.2233 and planar_flow < 0.4008
        + 0.006744596 * max(0.0, Q.sj2_dr - 0.2414351) * max(0.0, 154.25 - Q.pt_2) / 0.9967085   # +0.7%  sj2_dr > 0.2414 and pt_2 < 154.2
        - 0.006444456 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.20552 - Q.z_dr_0p2_0p4) / 0.02967662   # -0.6%  sj3_dr_max < 0.3012 and z_dr_0p2_0p4 < 0.2055
        + 0.006395105 * max(0.0, 0.006299534 - Q.sum_z_dr2_top2) / 0.003504831   # +0.6%  sum_z_dr2_top2 < 0.0063
        - 0.006146919 * max(0.0, 0.01357153 - Q.tau2) / 0.005299336   # -0.6%  tau2 < 0.01357
        - 0.005585214 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7) / 0.8992052   # -0.6%  N2 < 0.2233 and pt_7 < 53.44
        + 0.005459365 * max(0.0, 0.02415398 - Q.C2_b2) * max(0.0, 0.02612796 - Q.mean_phi) / 0.0005686318   # +0.5%  C2_b2 < 0.02415 and mean_phi < 0.02613
        - 0.004923448 * max(0.0, 1.340118e-05 - Q.e3) / 5.081834e-06   # -0.5%  e3 < 1.34e-05
        + 0.004709997 * max(0.0, 0.006299534 - Q.sum_z_dr2_top2) * max(0.0, Q.centroid_offset - 0.01096064) / 1.639932e-05   # +0.5%  sum_z_dr2_top2 < 0.0063 and centroid_offset > 0.01096
        + 0.004127167 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +0.4%  sj3_dr_max < 0.3012
        + 0.004037891 * max(0.0, 0.00483998 - Q.lam1) * max(0.0, Q.pt_entropy - 1.797618) / 0.0001621538   # +0.4%  lam1 < 0.00484 and pt_entropy > 1.798
        - 0.003978257 * max(0.0, Q.sj2_dr - 0.2179769) * max(0.0, 0.1818564 - Q.z_3rd) / 0.0009573735   # -0.4%  sj2_dr > 0.218 and z_3rd < 0.1819
        - 0.003660852 * max(0.0, 6.638339 - Q.log_sum_pt) / 0.1524936   # -0.4%  log_sum_pt < 6.638
        - 0.003649269 * max(0.0, Q.C2 - 0.06729223) / 0.002854902   # -0.4%  C2 > 0.06729
        - 0.003486301 * max(0.0, Q.sj2_dr - 0.2414351) * max(0.0, 1.129616 - Q.D2_b2) / 0.007762894   # -0.3%  sj2_dr > 0.2414 and D2_b2 < 1.13
        - 0.003458327 * max(0.0, Q.C2 - 0.02384388) / 0.01155414   # -0.3%  C2 > 0.02384
        - 0.003234601 * max(0.0, Q.sj3_dr_max - 0.3456459) / 0.006655619   # -0.3%  sj3_dr_max > 0.3456
        - 0.00276 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1626038 - Q.abseta_7) / 0.005313035   # -0.3%  N2 < 0.2233 and abseta_7 < 0.1626
        + 0.002735546 * max(0.0, Q.sj2_dr - 0.2414351) / 0.01315894   # +0.3%  sj2_dr > 0.2414
        - 0.00218116 * max(0.0, 0.006299534 - Q.sum_z_dr2_top2) * max(0.0, Q.z_4 - 0.06095291) / 7.857809e-05   # -0.2%  sum_z_dr2_top2 < 0.0063 and z_4 > 0.06095
        + 0.002078669 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, Q.phi_1 - -0.0001367927) / 1.344929e-05   # +0.2%  lam2 < 0.001131 and phi_1 > -0.0001368
        + 0.001765691 * max(0.0, 0.7459513 - Q.D2) * max(0.0, Q.pt_6 - 19.46875) / 2.3396   # +0.2%  D2 < 0.746 and pt_6 > 19.47
        + 0.001735743 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, Q.phi_0 - -0.0297699) / 3.166937e-05   # +0.2%  lam2 < 0.001131 and phi_0 > -0.02977
        - 0.001480549 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, Q.phi_1 - 0.009028244) / 1.507926e-06   # -0.1%  e3 < 0.0001869 and phi_1 > 0.009028
        + 0.001375328 * max(0.0, Q.max_dr - 0.0931108) * max(0.0, 0.04019165 - Q.absphi_0) / 0.0006099482   # +0.1%  max_dr > 0.09311 and absphi_0 < 0.04019
        + 0.001204167 * max(0.0, Q.sj3_dr_max - 0.3456459) * max(0.0, 1.129616 - Q.D2_b2) / 0.001618684   # +0.1%  sj3_dr_max > 0.3456 and D2_b2 < 1.13
        + 0.0009898116 * max(0.0, 0.2089872 - Q.sj3_dr_min) * max(0.0, -0.01790907 - Q.mean_eta) / 0.0001714584   # +0.1%  sj3_dr_min < 0.209 and mean_eta < -0.01791
        - 0.0005413349 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.centroid_offset - 0.02355416) / 0.0001983437   # -0.1%  N2 < 0.2233 and centroid_offset > 0.02355
        + 0.000515572 * max(0.0, 0.009032972 - Q.C2_b2) * max(0.0, Q.ptdr0_6 - 5.648151) / 0.00495401   # +0.1%  C2_b2 < 0.009033 and ptdr0_6 > 5.648
        + 0.0004039088 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, Q.dr_4 - 0.1100973) / 0.0002928023   # +0.0%  sj3_dr_max < 0.3012 and dr_4 > 0.1101
        + 0.0003832811 * max(0.0, 0.2233283 - Q.N2) * max(0.0, -0.02594505 - Q.mean_phi) / 3.611955e-05   # +0.0%  N2 < 0.2233 and mean_phi < -0.02595
        + 0.0003381055 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, Q.dr_7 - 0.1242755) / 5.192592e-05   # +0.0%  sum_z_dr2 < 0.01324 and dr_7 > 0.1243
        - 0.0003328795 * max(0.0, Q.sj2_dr - 0.2414351) * max(0.0, Q.z_dr_0p05_0p1 - 0.5882598) / 0.0005826806   # -0.0%  sj2_dr > 0.2414 and z_dr_0p05_0p1 > 0.5883
        + 0.0002717586 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, Q.mean_phi - 0.02612796) / 6.213564e-08   # +0.0%  e3 < 0.0001869 and mean_phi > 0.02613
        - 0.0002643505 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 24.42188 - Q.pt_6) / 0.01371831   # -0.0%  N2 < 0.2233 and pt_6 < 24.42
        - 0.0002454433 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.008074527   # -0.0%  N2 < 0.2233 and n_dr_0p2_0p4 > 1
        - 0.0002058628 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, Q.ptdr0_6 - 11.6057) / 6.612383e-06   # -0.0%  e3 < 0.0001869 and ptdr0_6 > 11.61
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 11.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.40521 * (0.06288181
        + 0.1034015 * max(0.0, 0.03117077 - Q.centroid_offset) / 0.01649145   # +10.3%  centroid_offset < 0.03117
        - 0.0998486 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # -10.0%  centroid_offset < 0.03776
        + 0.08248645 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +8.2%  LHA < 0.2161
        - 0.06949926 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -6.9%  sj3_dr_max < 0.3012
        + 0.04549006 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +4.5%  e3 < 0.0005117
        + 0.0450042 * max(0.0, 0.002635418 - Q.sum_z_dr2) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.001577364   # +4.5%  sum_z_dr2 < 0.002635 and n_dr_0p2_0p4 < 2
        + 0.04341104 * max(0.0, 0.002635418 - Q.sum_z_dr2) * max(0.0, 0.01627885 - Q.centroid_offset) / 7.136253e-06   # +4.3%  sum_z_dr2 < 0.002635 and centroid_offset < 0.01628
        - 0.04258631 * max(0.0, 715.4688 - Q.sum_pt) / 69.91196   # -4.3%  sum_pt < 715.5
        - 0.04066018 * max(0.0, 0.04649465 - Q.dr_0) / 0.01415326   # -4.1%  dr_0 < 0.04649
        - 0.03456049 * max(0.0, 0.03117077 - Q.centroid_offset) * max(0.0, 0.09549375 - Q.z_5) / 0.0004977925   # -3.5%  centroid_offset < 0.03117 and z_5 < 0.09549
        - 0.02997046 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -3.0%  LHA < 0.2161 and log_sum_pt < 6.804
        - 0.02952668 * max(0.0, 788.4484 - Q.sum_pt) / 112.1576   # -3.0%  sum_pt < 788.4
        - 0.02813019 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.09543973 - Q.z_6) / 0.001514041   # -2.8%  LHA < 0.2161 and z_6 < 0.09544
        + 0.02688309 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # +2.7%  max_dr < 0.1773
        - 0.0232309 * max(0.0, 0.03117077 - Q.centroid_offset) * max(0.0, Q.pt_5 - 29.875) / 0.3233752   # -2.3%  centroid_offset < 0.03117 and pt_5 > 29.88
        + 0.02107398 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +2.1%  sum_z_dr2 < 0.00502
        + 0.01812814 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, Q.n_for_90pct - 5.0) / 215.169   # +1.8%  sum_pt < 788.4 and n_for_90pct > 5
        + 0.0167596 * max(0.0, 0.06081235 - Q.z_6) / 0.009654642   # +1.7%  z_6 < 0.06081
        + 0.01405397 * max(0.0, 0.03556091 - Q.e2) * max(0.0, 0.0753896 - Q.z_7) / 0.0004290247   # +1.4%  e2 < 0.03556 and z_7 < 0.07539
        + 0.01402419 * max(0.0, 0.0005116989 - Q.e3) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.0009530518   # +1.4%  e3 < 0.0005117 and n_dr_0p1_0p2 < 3
        + 0.0127694 * max(0.0, 0.04939969 - Q.z_7) / 0.007457505   # +1.3%  z_7 < 0.0494
        + 0.01104058 * max(0.0, 0.03243272 - Q.z_7) / 0.002061024   # +1.1%  z_7 < 0.03243
        + 0.01100214 * max(0.0, 715.4688 - Q.sum_pt) * max(0.0, 0.0361727 - Q.dr_0) / 0.2640582   # +1.1%  sum_pt < 715.5 and dr_0 < 0.03617
        + 0.01086577 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 0.04678345 - Q.absphi_0) / 0.000232957   # +1.1%  z_7 < 0.0494 and absphi_0 < 0.04678
        + 0.01028007 * max(0.0, 0.0009641429 - Q.sum_z_dr2) / 0.0002052288   # +1.0%  sum_z_dr2 < 0.0009641
        - 0.01015573 * max(0.0, 0.002635418 - Q.sum_z_dr2) * max(0.0, Q.pt_6 - 27.57812) / 0.01123489   # -1.0%  sum_z_dr2 < 0.002635 and pt_6 > 27.58
        - 0.009366177 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # -0.9%  pt_7 > 34.53
        - 0.007902618 * max(0.0, 0.1767241 - Q.LHA) / 0.01861764   # -0.8%  LHA < 0.1767
        - 0.00720627 * max(0.0, 0.03459477 - Q.tau1) / 0.008474553   # -0.7%  tau1 < 0.03459
        - 0.007108794 * max(0.0, 0.1767241 - Q.LHA) * max(0.0, 0.006292091 - Q.zdr_0) / 7.498285e-05   # -0.7%  LHA < 0.1767 and zdr_0 < 0.006292
        - 0.006691703 * max(0.0, Q.pt_7 - 43.5) / 1.4368   # -0.7%  pt_7 > 43.5
        + 0.006426983 * max(0.0, 0.02320757 - Q.z_7) / 0.0007159291   # +0.6%  z_7 < 0.02321
        + 0.005884531 * max(0.0, 0.04649465 - Q.dr_0) * max(0.0, 2.080881 - Q.N3) / 0.008324961   # +0.6%  dr_0 < 0.04649 and N3 < 2.081
        - 0.005882069 * max(0.0, 20.125 - Q.pt_7) / 0.5468342   # -0.6%  pt_7 < 20.12
        + 0.005754889 * max(0.0, 0.03556091 - Q.e2) * max(0.0, Q.zdr_0 - 0.004918231) / 3.97457e-05   # +0.6%  e2 < 0.03556 and zdr_0 > 0.004918
        - 0.004311737 * max(0.0, 430.75 - Q.sum_pt_top5) / 14.05246   # -0.4%  sum_pt_top5 < 430.8
        + 0.004189056 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # +0.4%  sum_z_dr < 0.007674
        + 0.00406809 * max(0.0, 0.03556091 - Q.e2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.001328529   # +0.4%  e2 < 0.03556 and planar_flow < 0.3221
        + 0.003891048 * max(0.0, 0.02886576 - Q.z_6) / 0.0008381782   # +0.4%  z_6 < 0.02887
        - 0.003259496 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, Q.pt1_dr01 - 12.6865) / 0.1599605   # -0.3%  sj3_dr_max < 0.3012 and pt1_dr01 > 12.69
        + 0.002701639 * max(0.0, 0.0005116989 - Q.e3) * max(0.0, Q.absphi_0 - 0.04678345) / 2.519179e-06   # +0.3%  e3 < 0.0005117 and absphi_0 > 0.04678
        - 0.002416564 * max(0.0, 715.4688 - Q.sum_pt) * max(0.0, 0.007798268 - Q.zdr_0) / 0.05114364   # -0.2%  sum_pt < 715.5 and zdr_0 < 0.007798
        - 0.002411042 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, Q.ptdr0_2 - 11.87288) / 0.08620714   # -0.2%  sj3_dr_max < 0.3012 and ptdr0_2 > 11.87
        + 0.002395463 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +0.2%  z_7 < 0.02807
        - 0.00206314 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.2%  log_sum_pt > 6.896
        + 0.002049038 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # +0.2%  pt_7 > 53.44
        + 0.001902462 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # +0.2%  z_6 < 0.0216
        + 0.001763541 * max(0.0, 31.125 - Q.pt_4) / 0.3140792   # +0.2%  pt_4 < 31.12
        + 0.001742572 * max(0.0, 24.57812 - Q.pt_5) * max(0.0, 0.1583252 - Q.abseta_6) / 0.03135727   # +0.2%  pt_5 < 24.58 and abseta_6 < 0.1583
        - 0.001427246 * max(0.0, 0.03556091 - Q.e2) * max(0.0, Q.log_sum_pt - 6.896095) / 0.0001063848   # -0.1%  e2 < 0.03556 and log_sum_pt > 6.896
        - 0.001372174 * max(0.0, 0.03747769 - Q.z_4) / 0.000472225   # -0.1%  z_4 < 0.03748
        - 0.0009686726 * max(0.0, 0.02886576 - Q.z_6) * max(0.0, Q.zdr_0 - 0.004918231) / 3.900474e-06   # -0.1%  z_6 < 0.02887 and zdr_0 > 0.004918
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 23.83;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.83052 * (0.07977522
        - 0.2190474 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -21.9%  sum_z_dr2 < 0.008678
        + 0.08139354 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # +8.1%  sum_zz_dr2 < 0.008169
        + 0.0745478 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # +7.5%  tau1 < 0.1028
        + 0.0699453 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +7.0%  lam1 < 0.008376
        - 0.06423637 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # -6.4%  lam1_plus_lam2 < 0.01324
        - 0.04513886 * max(0.0, Q.sj3_dr_max - 0.07264452) / 0.1084421   # -4.5%  sj3_dr_max > 0.07264
        + 0.04385588 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +4.4%  lam2 < 0.003408
        + 0.03576586 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +3.6%  sj3_dr_max > 0.179
        - 0.03549207 * max(0.0, 0.1598486 - Q.max_dr) / 0.05771655   # -3.5%  max_dr < 0.1598
        - 0.03479379 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -3.5%  tau1 < 0.1136
        - 0.02496993 * max(0.0, Q.centroid_offset - 0.01627885) * max(0.0, 988.4078 - Q.sum_pt) / 2.602791   # -2.5%  centroid_offset > 0.01628 and sum_pt < 988.4
        - 0.02079048 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -2.1%  lam2 < 0.001131
        + 0.01983087 * max(0.0, Q.centroid_offset - 0.01627885) / 0.006307541   # +2.0%  centroid_offset > 0.01628
        - 0.01882113 * max(0.0, Q.sj3_dr_max - 0.07264452) * max(0.0, Q.eccentricity - 0.9031255) / 0.006127495   # -1.9%  sj3_dr_max > 0.07264 and eccentricity > 0.9031
        + 0.01703312 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +1.7%  lam1 < 0.012
        + 0.0140163 * max(0.0, 0.03853752 - Q.C3) / 0.01496309   # +1.4%  C3 < 0.03854
        + 0.01250772 * max(0.0, 0.02689598 - Q.sum_z_dr) / 0.004330137   # +1.3%  sum_z_dr < 0.0269
        + 0.01193248 * max(0.0, Q.sj3_dr_max - 0.1789613) * max(0.0, Q.eccentricity - 0.9031255) / 0.002046829   # +1.2%  sj3_dr_max > 0.179 and eccentricity > 0.9031
        + 0.01111614 * max(0.0, 0.0095303 - Q.sum_z_dr2_top2) / 0.006109573   # +1.1%  sum_z_dr2_top2 < 0.00953
        + 0.01097193 * max(0.0, 6.638339 - Q.log_sum_pt) / 0.1524936   # +1.1%  log_sum_pt < 6.638
        - 0.01014013 * max(0.0, Q.sj3_dr_min - 0.02022982) / 0.02964063   # -1.0%  sj3_dr_min > 0.02023
        + 0.008820839 * max(0.0, 29.90625 - Q.pt_6) / 1.385454   # +0.9%  pt_6 < 29.91
        - 0.00873461 * max(0.0, 1.651088 - Q.D3) / 0.5901079   # -0.9%  D3 < 1.651
        - 0.008685424 * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 0.001433269   # -0.9%  sum_zz_dr2 > 0.01166
        + 0.008666074 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 715.4688 - Q.sum_pt) / 1.381606   # +0.9%  centroid_offset > 0.008092 and sum_pt < 715.5
        + 0.007266641 * max(0.0, 6.327379 - Q.log_sum_pt) / 0.0331691   # +0.7%  log_sum_pt < 6.327
        - 0.007082107 * max(0.0, 0.03448406 - Q.z_6) / 0.001525592   # -0.7%  z_6 < 0.03448
        + 0.006865369 * max(0.0, 1.627072e-05 - Q.e3) / 6.577181e-06   # +0.7%  e3 < 1.627e-05
        + 0.00621626 * max(0.0, 1.332146 - Q.D2) / 0.3438169   # +0.6%  D2 < 1.332
        - 0.005996607 * max(0.0, 0.1055393 - Q.dr_1) / 0.05636111   # -0.6%  dr_1 < 0.1055
        + 0.005935795 * max(0.0, Q.LHA - 0.266913) * max(0.0, Q.eccentricity - 0.9031255) / 0.001613144   # +0.6%  LHA > 0.2669 and eccentricity > 0.9031
        + 0.005533204 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.01426135 - Q.mean_phi2) / 9.496308e-05   # +0.6%  centroid_offset > 0.008092 and mean_phi2 < 0.01426
        - 0.005090195 * max(0.0, Q.sj3_dr_max - 0.1789613) * max(0.0, 6.0 - Q.n_dr_0p05_0p1) / 0.1786734   # -0.5%  sj3_dr_max > 0.179 and n_dr_0p05_0p1 < 6
        - 0.004195071 * max(0.0, 6.327379 - Q.log_sum_pt) * max(0.0, Q.pt_7 - 25.57812) / 0.2533514   # -0.4%  log_sum_pt < 6.327 and pt_7 > 25.58
        - 0.004194754 * max(0.0, Q.eccentricity - 0.9031255) / 0.04739094   # -0.4%  eccentricity > 0.9031
        + 0.00393764 * max(0.0, Q.LHA - 0.266913) / 0.03187832   # +0.4%  LHA > 0.2669
        - 0.00380354 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.4%  z_6 < 0.0216
        - 0.003410795 * max(0.0, Q.sj3_dr_max - 0.3012016) / 0.01214652   # -0.3%  sj3_dr_max > 0.3012
        + 0.002913506 * max(0.0, 6.638339 - Q.log_sum_pt) * max(0.0, 0.0586137 - Q.z_7) / 0.0003525255   # +0.3%  log_sum_pt < 6.638 and z_7 < 0.05861
        + 0.002878011 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 19.46875 - Q.pt_6) / 0.002700092   # +0.3%  lam1 < 0.012 and pt_6 < 19.47
        + 0.002674208 * max(0.0, Q.sum_zz_dr2 - 0.01716248) / 0.0007604864   # +0.3%  sum_zz_dr2 > 0.01716
        + 0.002478715 * max(0.0, Q.sj3_dr_min - 0.02022982) * max(0.0, 0.4246762 - Q.sj3_pairmin_over_m) / 0.003883862   # +0.2%  sj3_dr_min > 0.02023 and sj3_pairmin_over_m < 0.4247
        + 0.002288942 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.1682655 - Q.sj2_dr) / 0.0003126334   # +0.2%  centroid_offset > 0.008092 and sj2_dr < 0.1683
        - 0.001654544 * max(0.0, Q.eccentricity - 0.9031255) * max(0.0, 0.9827275 - Q.D3) / 0.007764278   # -0.2%  eccentricity > 0.9031 and D3 < 0.9827
        + 0.001284793 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 0.09397911 - Q.dr_7) / 7.480263e-05   # +0.1%  centroid_offset > 0.02077 and dr_7 < 0.09398
        + 0.0009426102 * max(0.0, 6.638339 - Q.log_sum_pt) * max(0.0, Q.n_dr_0p1_0p2 - 3.0) / 0.1046422   # +0.1%  log_sum_pt < 6.638 and n_dr_0p1_0p2 > 3
        + 0.0007979321 * max(0.0, 0.1598486 - Q.max_dr) * max(0.0, Q.ptdr0_4 - 8.939062) / 0.01180828   # +0.1%  max_dr < 0.1598 and ptdr0_4 > 8.939
        + 0.0004110865 * max(0.0, 0.0095303 - Q.sum_z_dr2_top2) * max(0.0, -0.02665591 - Q.mean_eta) / 1.319786e-06   # +0.0%  sum_z_dr2_top2 < 0.00953 and mean_eta < -0.02666
        + 0.0003400348 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.dr01 - 0.1410336) / 0.01228651   # +0.0%  sum_pt > 988.4 and dr01 > 0.141
        + 0.0002844518 * max(0.0, 0.02160287 - Q.z_6) * max(0.0, 6.842717 - Q.log_sum_pt) / 4.457061e-06   # +0.0%  z_6 < 0.0216 and log_sum_pt < 6.843
        - 0.000269183 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.dr01 - 0.1662967) / 0.008508326   # -0.0%  sum_pt > 988.4 and dr01 > 0.1663
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 50.56;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 50.56188 * (0.201066
        - 0.1069942 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -10.7%  sum_zz_dr2 < 0.008169
        - 0.09152988 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -9.2%  sum_z_dr2 > 0.00752
        + 0.0712002 * max(0.0, 0.00718279 - Q.sum_zz_dr2) / 0.003528787   # +7.1%  sum_zz_dr2 < 0.007183
        - 0.05994281 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -6.0%  tau1 < 0.1136
        - 0.05865184 * max(0.0, 0.005590289 - Q.lam1_plus_lam2) / 0.002229311   # -5.9%  lam1_plus_lam2 < 0.00559
        + 0.05330495 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +5.3%  tau1 < 0.09539
        + 0.04668456 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # +4.7%  sum_z_dr2 > 0.008678
        + 0.04404816 * max(0.0, 0.005284669 - Q.sum_zz_dr2) / 0.00223735   # +4.4%  sum_zz_dr2 < 0.005285
        - 0.04276318 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # -4.3%  sum_zz_dr2 < 0.01166
        + 0.03827483 * max(0.0, Q.sum_z_dr2 - 0.01323868) / 0.001442684   # +3.8%  sum_z_dr2 > 0.01324
        - 0.03824621 * max(0.0, Q.sum_z_dr2 - 0.004372139) / 0.003638805   # -3.8%  sum_z_dr2 > 0.004372
        - 0.03401952 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -3.4%  sum_z_dr < 0.08724
        - 0.03303497 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # -3.3%  sj3_dr_max < 0.2134
        - 0.03058098 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -3.1%  sj2_dr < 0.1873
        + 0.02927373 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +2.9%  sj2_dr < 0.1592
        + 0.02150688 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # +2.2%  sj3_dr_max < 0.3456
        - 0.01791871 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # -1.8%  e3 < 0.0001869
        + 0.01476777 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +1.5%  sj3_dr_max < 0.1986
        - 0.01441215 * max(0.0, Q.sum_z_dr2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9458207) / 2.125219e-05   # -1.4%  sum_z_dr2 > 0.01324 and eccentricity > 0.9458
        + 0.01323949 * max(0.0, 0.2931906 - Q.LHA) / 0.07167606   # +1.3%  LHA < 0.2932
        + 0.01203681 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +1.2%  sj3_dr_max < 0.1426
        + 0.01201906 * max(0.0, 0.1979689 - Q.max_dr) / 0.08649773   # +1.2%  max_dr < 0.198
        + 0.01125999 * max(0.0, Q.sum_z_dr2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +1.1%  sum_z_dr2 > 0.004372 and planar_flow < 0.195
        + 0.01018727 * max(0.0, 0.0479157 - Q.sum_z_dr) / 0.01216112   # +1.0%  sum_z_dr < 0.04792
        + 0.00971329 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +1.0%  lam2 < 0.001131
        - 0.008259483 * max(0.0, Q.centroid_offset - 0.009480685) / 0.009662931   # -0.8%  centroid_offset > 0.009481
        - 0.008103702 * max(0.0, Q.sum_z_dr2 - 0.008678045) * max(0.0, Q.eccentricity - 0.9458207) / 3.56764e-05   # -0.8%  sum_z_dr2 > 0.008678 and eccentricity > 0.9458
        - 0.007231202 * max(0.0, 0.121681 - Q.max_dr) / 0.03360572   # -0.7%  max_dr < 0.1217
        - 0.007165936 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # -0.7%  LHA < 0.2161
        - 0.006762085 * max(0.0, 0.1416226 - Q.tau1) / 0.08190733   # -0.7%  tau1 < 0.1416
        - 0.006367791 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 4.721224 - Q.D2_b2) / 0.02812613   # -0.6%  centroid_offset < 0.02077 and D2_b2 < 4.721
        - 0.006025103 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -0.6%  lam2 < 0.0005373
        + 0.005868053 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.423044) / 0.01391616   # +0.6%  planar_flow < 0.195 and log_sum_pt > 6.423
        - 0.005047758 * max(0.0, 0.01165737 - Q.sum_zz_dr2) * max(0.0, 48.71875 - Q.pt_7) / 0.1103404   # -0.5%  sum_zz_dr2 < 0.01166 and pt_7 < 48.72
        - 0.003547631 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -0.4%  sj2_dr < 0.1295
        + 0.002858841 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) / 0.0003002514   # +0.3%  sum_z_dr2_top2 < 0.001057
        - 0.002677117 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.lam1_plus_lam2 - 0.007520088) / 0.0001455029   # -0.3%  planar_flow < 0.195 and lam1_plus_lam2 > 0.00752
        + 0.001927801 * max(0.0, 0.005590289 - Q.lam1_plus_lam2) * max(0.0, -0.00469376 - Q.mean_phi) / 4.24156e-06   # +0.2%  lam1_plus_lam2 < 0.00559 and mean_phi < -0.004694
        - 0.001842643 * max(0.0, 0.0005611231 - Q.sum_z_dr2) / 9.465572e-05   # -0.2%  sum_z_dr2 < 0.0005611
        - 0.001786264 * max(0.0, 0.213399 - Q.sj3_dr_max) * max(0.0, Q.eccentricity - 0.9458207) / 0.0008147252   # -0.2%  sj3_dr_max < 0.2134 and eccentricity > 0.9458
        + 0.001683296 * max(0.0, 0.005590289 - Q.lam1_plus_lam2) * max(0.0, -0.006779839 - Q.mean_eta) / 3.381615e-06   # +0.2%  lam1_plus_lam2 < 0.00559 and mean_eta < -0.00678
        + 0.001504809 * max(0.0, 0.001101266 - Q.sum_zz_dr2) / 0.000308004   # +0.2%  sum_zz_dr2 < 0.001101
        + 0.001306338 * max(0.0, 0.005590289 - Q.lam1_plus_lam2) * max(0.0, Q.mean_eta - 0.009367547) / 2.50486e-06   # +0.1%  lam1_plus_lam2 < 0.00559 and mean_eta > 0.009368
        + 0.001227068 * max(0.0, 0.005590289 - Q.lam1_plus_lam2) * max(0.0, Q.mean_phi - 0.009050008) / 2.537131e-06   # +0.1%  lam1_plus_lam2 < 0.00559 and mean_phi > 0.00905
        - 0.0009114331 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 33.6875 - Q.pt_6) / 0.1349842   # -0.1%  planar_flow < 0.195 and pt_6 < 33.69
        - 0.0006679487 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 0.05240025 - Q.z_7) / 0.000631283   # -0.1%  planar_flow < 0.195 and z_7 < 0.0524
        + 0.000551961 * max(0.0, 0.00718279 - Q.sum_zz_dr2) * max(0.0, Q.sj3_dr13 - 0.1982159) / 1.577335e-05   # +0.1%  sum_zz_dr2 < 0.007183 and sj3_dr13 > 0.1982
        - 0.0003136606 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 6.679047e-05   # -0.0%  tau1 < 0.05357 and z_dr_0p05_0p1 > 0.6748
        - 0.0002454319 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) * max(0.0, 0.380911 - Q.D2_b2) / 3.440047e-06   # -0.0%  sum_z_dr2_top2 < 0.001057 and D2_b2 < 0.3809
        - 0.0002130221 * max(0.0, 0.01165737 - Q.sum_zz_dr2) * max(0.0, Q.sj3_dr23 - 0.2989421) / 7.117187e-06   # -0.0%  sum_zz_dr2 < 0.01166 and sj3_dr23 > 0.2989
        - 0.0001504608 * max(0.0, Q.sum_z_dr2 - 0.008678045) * max(0.0, Q.sj3_z1 - 0.8740683) / 2.104095e-07   # -0.0%  sum_z_dr2 > 0.008678 and sj3_z1 > 0.8741
        - 0.0001417275 * max(0.0, Q.sum_z_dr2 - 0.004372139) * max(0.0, Q.sj3_z1 - 0.8740683) / 7.896779e-07   # -0.0%  sum_z_dr2 > 0.004372 and sj3_z1 > 0.8741
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 11.92;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.92031 * (-0.01385661
        + 0.08539262 * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.00293624   # +8.5%  sum_z_dr2 < 0.006679
        - 0.07168654 * max(0.0, 0.05464922 - Q.sum_z_dr) / 0.01532978   # -7.2%  sum_z_dr < 0.05465
        - 0.07069849 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, 0.005590289 - Q.lam1_plus_lam2) / 7.638845e-05   # -7.1%  sum_z_dr < 0.05465 and lam1_plus_lam2 < 0.00559
        - 0.05852356 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, 0.002635418 - Q.lam1_plus_lam2) / 3.204761e-05   # -5.9%  sum_z_dr < 0.05465 and lam1_plus_lam2 < 0.002635
        - 0.05352866 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -5.4%  sj3_dr_max < 0.1426
        - 0.051545 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, 0.20552 - Q.z_dr_0p2_0p4) / 0.005135396   # -5.2%  LHA < 0.1967 and z_dr_0p2_0p4 < 0.2055
        + 0.04955393 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.002635418 - Q.sum_z_dr2) / 3.338443e-05   # +5.0%  tau1 < 0.05357 and sum_z_dr2 < 0.002635
        + 0.04331469 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +4.3%  sum_z_dr2 < 0.006679 and centroid_offset < 0.02355
        + 0.04151708 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # +4.2%  LHA < 0.1967
        + 0.03721272 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +3.7%  sum_z_dr2 < 0.00502
        + 0.03533958 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.0003061234 - Q.lam2) / 1.676724e-05   # +3.5%  sj3_dr_max < 0.1986 and lam2 < 0.0003061
        + 0.03482473 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, 0.0001947983 - Q.lam2) / 2.397672e-06   # +3.5%  sum_z_dr < 0.05465 and lam2 < 0.0001948
        + 0.030152 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +3.0%  sj3_dr_max < 0.107
        + 0.02933416 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.0001053955   # +2.9%  sum_z_dr2 < 0.00502 and z_dr_0p2_0p4 < 0.05644
        + 0.02783305 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +2.8%  sum_pt_top5 > 687.4
        - 0.02491991 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, 0.0003061234 - Q.lam2) / 6.518054e-06   # -2.5%  sj3_dr_max < 0.107 and lam2 < 0.0003061
        - 0.02394018 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -2.4%  log_sum_pt > 6.701
        - 0.02316324 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -2.3%  sum_z_dr2 < 0.00502 and centroid_offset > 0.00679
        + 0.02257228 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +2.3%  tau1 < 0.05357
        - 0.01803268 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 43.5 - Q.pt_7) / 0.02255472   # -1.8%  sum_z_dr2 < 0.00502 and pt_7 < 43.5
        + 0.01733958 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.002635418 - Q.lam1_plus_lam2) / 5.489096e-05   # +1.7%  log_sum_pt > 6.701 and lam1_plus_lam2 < 0.002635
        - 0.01706866 * max(0.0, 0.01713288 - Q.tau2) / 0.008065871   # -1.7%  tau2 < 0.01713
        - 0.0146376 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, Q.lam1_plus_lam2 - 0.001653836) / 1.467156e-06   # -1.5%  sum_z_dr2 < 0.006679 and lam1_plus_lam2 > 0.001654
        - 0.01389127 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, 0.0001947983 - Q.lam2) / 4.231311e-06   # -1.4%  LHA < 0.1967 and lam2 < 0.0001948
        - 0.01270223 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.01002369 - Q.sum_z_dr2_top3) / 0.3268181   # -1.3%  sum_pt_top5 > 687.4 and sum_z_dr2_top3 < 0.01002
        + 0.01134874 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # +1.1%  lam1 < 0.001504
        + 0.009650955 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +1.0%  sj3_dr_max < 0.1986
        + 0.009649317 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 43.5 - Q.pt_7) / 0.2124887   # +1.0%  tau1 < 0.05357 and pt_7 < 43.5
        - 0.007314108 * max(0.0, Q.z_dr_0_0p05 - 0.9008535) * max(0.0, 0.0001947983 - Q.lam2) / 5.136077e-06   # -0.7%  z_dr_0_0p05 > 0.9009 and lam2 < 0.0001948
        + 0.006809797 * max(0.0, 0.0003193707 - Q.sum_z_dr2) / 4.035777e-05   # +0.7%  sum_z_dr2 < 0.0003194
        - 0.006744481 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.006679471 - Q.lam1_plus_lam2) / 0.0001692711   # -0.7%  log_sum_pt > 6.701 and lam1_plus_lam2 < 0.006679
        + 0.006529395 * max(0.0, Q.n_dr_0_0p05 - 5.0) / 1.093146   # +0.7%  n_dr_0_0p05 > 5
        - 0.005597527 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.0009641429 - Q.sum_z_dr2) / 0.01750133   # -0.6%  sum_pt_top5 > 687.4 and sum_z_dr2 < 0.0009641
        - 0.004217043 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.006445344 - Q.mean_phi2) / 0.0001920358   # -0.4%  log_sum_pt > 6.701 and mean_phi2 < 0.006445
        + 0.004047397 * max(0.0, 0.1531958 - Q.N2) / 0.02098023   # +0.4%  N2 < 0.1532
        - 0.003802748 * max(0.0, 24.42188 - Q.pt_6) / 0.6046572   # -0.4%  pt_6 < 24.42
        + 0.003757657 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, Q.sum_z_dr2_top3 - 0.001155057) / 3.499346e-05   # +0.4%  sj3_dr_max < 0.1986 and sum_z_dr2_top3 > 0.001155
        - 0.003573077 * max(0.0, 0.01357518 - Q.tau1) / 0.001531397   # -0.4%  tau1 < 0.01358
        - 0.003162416 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 6.502799 - Q.log_sum_pt) / 8.085113e-05   # -0.3%  sum_z_dr2 < 0.00502 and log_sum_pt < 6.503
        - 0.003087211 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, Q.lam1_plus_lam2 - 0.002635418) / 2.147931e-05   # -0.3%  sj3_dr_max < 0.1986 and lam1_plus_lam2 > 0.002635
        - 0.00198371 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 3.526798e-07   # -0.2%  sj3_dr_max < 0.1426 and lam1_plus_lam2 > 0.00559
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 26.95;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.95418 * (-0.1009056
        + 0.1213747 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) / 0.00293624   # +12.1%  lam1_plus_lam2 < 0.006679
        + 0.102389 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +10.2%  sum_z_dr2 < 0.00502
        + 0.0699199 * max(0.0, 6.896095 - Q.log_sum_pt) / 0.3571984   # +7.0%  log_sum_pt < 6.896
        + 0.06705916 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # +6.7%  sum_z_dr2 < 0.003563
        - 0.06411032 * max(0.0, 0.004641801 - Q.sum_zz_dr2) / 0.001869767   # -6.4%  sum_zz_dr2 < 0.004642
        - 0.04192161 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -4.2%  sj3_dr_max < 0.1426
        - 0.03653261 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # -3.7%  sum_zz_dr2 < 0.003013
        - 0.0336572 * max(0.0, 0.06663269 - Q.sum_z_dr) / 0.02186049   # -3.4%  sum_z_dr < 0.06663
        + 0.03324504 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +3.3%  sj3_dr_max < 0.1986
        - 0.03221029 * max(0.0, 0.02378091 - Q.sum_zz_dr2) / 0.01817124   # -3.2%  sum_zz_dr2 < 0.02378
        + 0.03114584 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.06729223 - Q.C2) / 0.0003300761   # +3.1%  centroid_offset < 0.01838 and C2 < 0.06729
        - 0.02876514 * max(0.0, 0.05464922 - Q.sum_z_dr) / 0.01532978   # -2.9%  sum_z_dr < 0.05465
        + 0.02572431 * max(0.0, 0.02355416 - Q.centroid_offset) / 0.0103615   # +2.6%  centroid_offset < 0.02355
        - 0.0247265 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -2.5%  sj2_dr < 0.1295
        - 0.01902361 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -1.9%  centroid_offset < 0.01838
        - 0.01384816 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -1.4%  e3 > 3.892e-05
        + 0.01319952 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +1.3%  lam2 < 0.0001948
        + 0.01174904 * max(0.0, 0.1778793 - Q.sj2_dr) / 0.05866968   # +1.2%  sj2_dr < 0.1779
        + 0.01164635 * max(0.0, 0.01705377 - Q.zdr_0) / 0.006142531   # +1.2%  zdr_0 < 0.01705
        + 0.01121688 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +1.1%  lam2 > 0.001131
        - 0.01048823 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # -1.0%  dr_0 < 0.06413
        + 0.01021884 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +1.0%  e3 > 0.0001869
        + 0.00966623 * max(0.0, 7.300726e-05 - Q.lam2) / 2.767163e-05   # +1.0%  lam2 < 7.301e-05
        - 0.009434005 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # -0.9%  lam1 < 0.00484
        + 0.009354989 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.pt_1 - 93.25) / 0.3775907   # +0.9%  centroid_offset < 0.01838 and pt_1 > 93.25
        + 0.009338289 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +0.9%  max_dr < 0.1118
        - 0.008737185 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.pt1_over_pt0 - 0.2131291) / 0.002760106   # -0.9%  centroid_offset < 0.01838 and pt1_over_pt0 > 0.2131
        + 0.008581972 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.01364517 - Q.tau3) / 6.201347e-05   # +0.9%  centroid_offset < 0.01838 and tau3 < 0.01365
        + 0.008511682 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, Q.N3 - 0.7686247) / 0.007184418   # +0.9%  centroid_offset < 0.02355 and N3 > 0.7686
        + 0.007793389 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.00739495 - Q.tau3) / 2.189213e-05   # +0.8%  centroid_offset < 0.01838 and tau3 < 0.007395
        - 0.007620657 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.04435703 - Q.M2) / 5.61399e-05   # -0.8%  centroid_offset < 0.01838 and M2 < 0.04436
        - 0.006756541 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, 0.380911 - Q.D2_b2) / 0.001264948   # -0.7%  centroid_offset < 0.02355 and D2_b2 < 0.3809
        + 0.005981332 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 0.0234802   # +0.6%  centroid_offset < 0.01838 and n_dr_0p05_0p1 < 5
        - 0.005818415 * max(0.0, 0.001101266 - Q.sum_zz_dr2) / 0.000308004   # -0.6%  sum_zz_dr2 < 0.001101
        - 0.005113646 * max(0.0, 0.00483998 - Q.lam1) * max(0.0, 0.1792439 - Q.dr1_7) / 0.0002681756   # -0.5%  lam1 < 0.00484 and dr1_7 < 0.1792
        + 0.005064039 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, Q.tau21_b2 - 0.004811143) / 0.002193894   # +0.5%  centroid_offset < 0.02355 and tau21_b2 > 0.004811
        - 0.00494069 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, Q.planar_flow - 0.1115136) / 0.07272605   # -0.5%  log_sum_pt < 6.896 and planar_flow > 0.1115
        - 0.004831655 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, Q.planar_flow - 0.2534037) / 0.0001586905   # -0.5%  lam2 > 0.001131 and planar_flow > 0.2534
        + 0.004465759 * max(0.0, 0.001101266 - Q.sum_zz_dr2) * max(0.0, Q.pt_7 - 15.55391) / 0.005706103   # +0.4%  sum_zz_dr2 < 0.001101 and pt_7 > 15.55
        - 0.00440793 * max(0.0, 0.06663269 - Q.sum_z_dr) * max(0.0, 0.00161889 - Q.mean_phi) / 8.763302e-05   # -0.4%  sum_z_dr < 0.06663 and mean_phi < 0.001619
        + 0.004329068 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, 0.0825754 - Q.dr_7) / 0.0003946939   # +0.4%  centroid_offset < 0.02355 and dr_7 < 0.08258
        - 0.004193411 * max(0.0, 0.04505724 - Q.planar_flow) / 0.007613167   # -0.4%  planar_flow < 0.04506
        + 0.003923282 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.4%  n_dr_0p2_0p4 > 1
        - 0.003253232 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, 0.9206502 - Q.D2_b2) / 0.1512968   # -0.3%  log_sum_pt < 6.896 and D2_b2 < 0.9207
        - 0.002944769 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, Q.z_dr_0p2_0p4 - 0.0) / 0.0166461   # -0.3%  log_sum_pt < 6.896 and z_dr_0p2_0p4 > 0
        - 0.002855482 * max(0.0, 0.001101266 - Q.sum_zz_dr2) * max(0.0, Q.eccentricity - 0.7117266) / 3.682686e-05   # -0.3%  sum_zz_dr2 < 0.001101 and eccentricity > 0.7117
        + 0.00284166 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +0.3%  LHA < 0.2161
        - 0.002807824 * max(0.0, 0.004918231 - Q.zdr_0) / 0.0006323791   # -0.3%  zdr_0 < 0.004918
        + 0.00253304 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, 0.05753583 - Q.z_5) / 0.0005902159   # +0.3%  log_sum_pt < 6.896 and z_5 < 0.05754
        - 0.002006101 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) * max(0.0, 0.8272948 - Q.D3) / 0.0002524894   # -0.2%  lam1_plus_lam2 < 0.006679 and D3 < 0.8273
        + 0.001856374 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) * max(0.0, Q.tau3 - 0.003447406) / 3.346084e-06   # +0.2%  lam1_plus_lam2 < 0.006679 and tau3 > 0.003447
        - 0.001825804 * max(0.0, 309.625 - Q.sum_pt_top3) / 11.24887   # -0.2%  sum_pt_top3 < 309.6
        - 0.00164281 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) * max(0.0, -0.006779839 - Q.mean_eta) / 4.770336e-06   # -0.2%  lam1_plus_lam2 < 0.006679 and mean_eta < -0.00678
        - 0.001613475 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) * max(0.0, Q.mean_eta - 0.009367547) / 3.542905e-06   # -0.2%  lam1_plus_lam2 < 0.006679 and mean_eta > 0.009368
        + 0.001612422 * max(0.0, 0.4309923 - Q.tau32) / 0.03475647   # +0.2%  tau32 < 0.431
        - 0.001564535 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, Q.pt_7 - 27.375) / 0.0002613758   # -0.2%  lam2 < 7.301e-05 and pt_7 > 27.38
        - 0.001498218 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.003218391 - Q.mean_eta) / 0.0001345871   # -0.1%  LHA < 0.2161 and mean_eta < 0.003218
        - 0.001464093 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.07735553 - Q.mratio_min_012) / 0.0001043432   # -0.1%  centroid_offset < 0.01838 and mratio_min_012 < 0.07736
        - 0.001419586 * max(0.0, 0.06663269 - Q.sum_z_dr) * max(0.0, Q.mean_phi - 0.004406178) / 3.557874e-05   # -0.1%  sum_z_dr < 0.06663 and mean_phi > 0.004406
        - 0.00126899 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, 0.5186963 - Q.tau32) / 0.0003401702   # -0.1%  centroid_offset < 0.02355 and tau32 < 0.5187
        - 0.001148222 * max(0.0, Q.pt_1 - 159.25) * max(0.0, 0.1132812 - Q.absphi_5) / 0.8646748   # -0.1%  pt_1 > 159.2 and absphi_5 < 0.1133
        - 0.001129636 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, 0.01778111 - Q.dr_2) / 0.0004075145   # -0.1%  log_sum_pt < 6.896 and dr_2 < 0.01778
        + 0.00107953 * max(0.0, 0.004918231 - Q.zdr_0) * max(0.0, Q.pt_3 - 71.5625) / 0.01103846   # +0.1%  zdr_0 < 0.004918 and pt_3 > 71.56
        + 0.001015412 * max(0.0, 0.1294903 - Q.sj2_dr) * max(0.0, -0.006779839 - Q.mean_eta) / 5.903935e-05   # +0.1%  sj2_dr < 0.1295 and mean_eta < -0.00678
        - 0.001001145 * max(0.0, Q.pt_1 - 159.25) * max(0.0, 1.129616 - Q.D2_b2) / 4.081098   # -0.1%  pt_1 > 159.2 and D2_b2 < 1.13
        + 0.0009659787 * max(0.0, 0.003562611 - Q.sum_z_dr2) * max(0.0, 0.8272948 - Q.D3) / 7.149804e-05   # +0.1%  sum_z_dr2 < 0.003563 and D3 < 0.8273
        + 0.0009631734 * max(0.0, 0.1294903 - Q.sj2_dr) * max(0.0, Q.mean_eta - 0.01271871) / 3.354705e-05   # +0.1%  sj2_dr < 0.1295 and mean_eta > 0.01272
        + 0.0008161283 * max(0.0, Q.lam1_plus_lam2 - 0.02530566) / 0.0002851417   # +0.1%  lam1_plus_lam2 > 0.02531
        + 0.000797262 * max(0.0, 0.06663269 - Q.sum_z_dr) * max(0.0, 0.2669656 - Q.D2_b2) / 0.0002788187   # +0.1%  sum_z_dr < 0.06663 and D2_b2 < 0.267
        - 0.0007955955 * max(0.0, 0.003562611 - Q.sum_z_dr2) * max(0.0, Q.mean_eta - 0.01271871) / 6.620797e-07   # -0.1%  sum_z_dr2 < 0.003563 and mean_eta > 0.01272
        - 0.0006179114 * max(0.0, 0.003562611 - Q.sum_z_dr2) * max(0.0, -0.01284493 - Q.mean_eta) / 6.468207e-07   # -0.1%  sum_z_dr2 < 0.003563 and mean_eta < -0.01284
        - 0.0006045711 * max(0.0, 0.002316125 - Q.centroid_offset) * max(0.0, 0.1167998 - Q.sj3_z3) / 3.71719e-06   # -0.1%  centroid_offset < 0.002316 and sj3_z3 < 0.1168
        - 0.0005428588 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, Q.tau32 - 0.2691692) / 0.02298418   # -0.1%  n_dr_0p2_0p4 > 1 and tau32 > 0.2692
        - 0.0004777713 * max(0.0, Q.e3 - 0.0001869378) * max(0.0, 33.21875 - Q.pt_7) / 8.091322e-05   # -0.0%  e3 > 0.0001869 and pt_7 < 33.22
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 14.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.64364 * (0.1760849
        + 0.1989288 * Q.lam1 / 0.00591636   # +19.9%  lam1
        - 0.1126899 * max(0.0, Q.sum_z_dr2 - 0.002635418) / 0.004603951   # -11.3%  sum_z_dr2 > 0.002635
        - 0.06875466 * max(0.0, 0.1245537 - Q.sum_z_dr) / 0.06848162   # -6.9%  sum_z_dr < 0.1246
        - 0.05284057 * max(0.0, Q.lam1 - 0.00483998) / 0.002931468   # -5.3%  lam1 > 0.00484
        - 0.0489593 * max(0.0, Q.LHA - 0.1967397) / 0.07109091   # -4.9%  LHA > 0.1967
        + 0.04819984 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # +4.8%  sum_z_dr2 > 0.00752
        + 0.04666315 * max(0.0, Q.lam2 - 0.0001947983) / 0.0004461515   # +4.7%  lam2 > 0.0001948
        - 0.04532034 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -4.5%  lam1 < 0.006507
        + 0.03186064 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # +3.2%  sj2_dr > 0.1295
        + 0.02890893 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +2.9%  sj3_dr_max < 0.3012
        + 0.02686256 * max(0.0, Q.tau1 - 0.04369778) / 0.031989   # +2.7%  tau1 > 0.0437
        - 0.0202137 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # -2.0%  lam1 < 0.003377
        - 0.01762871 * Q.e2 / 0.02863215   # -1.8%  e2
        - 0.0168851 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.6947818 - Q.planar_flow) / 0.0166918   # -1.7%  sj2_dr > 0.1683 and planar_flow < 0.6948
        + 0.01564627 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # +1.6%  e3 > 8.148e-05
        - 0.0139155 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -1.4%  log_sum_pt > 6.67
        - 0.01327194 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.planar_flow - 0.06126205) / 0.0002819968   # -1.3%  lam2 > 0.0001948 and planar_flow > 0.06126
        + 0.01307296 * max(0.0, 0.002915531 - Q.sum_z_dr2_top3) / 0.001178811   # +1.3%  sum_z_dr2_top3 < 0.002916
        - 0.01297864 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # -1.3%  sj2_dr > 0.1683
        - 0.01123136 * max(0.0, 48.03125 - Q.pt_6) / 9.943776   # -1.1%  pt_6 < 48.03
        + 0.01092814 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.158545   # +1.1%  n_dr_0p05_0p1 < 5
        + 0.009992675 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +1.0%  n_dr_0p2_0p4 > 1
        - 0.009671686 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.1647949 - Q.absphi_7) / 0.0004021457   # -1.0%  lam1 < 0.006507 and absphi_7 < 0.1648
        - 0.009578497 * max(0.0, Q.LHA - 0.3127275) * max(0.0, 0.6947818 - Q.planar_flow) / 0.005887138   # -1.0%  LHA > 0.3127 and planar_flow < 0.6948
        - 0.009475218 * max(0.0, 0.06810151 - Q.z_7) / 0.01877013   # -0.9%  z_7 < 0.0681
        + 0.009347747 * max(0.0, 0.001501708 - Q.sum_z_dr2_top5) / 0.0004379639   # +0.9%  sum_z_dr2_top5 < 0.001502
        - 0.00790863 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 4.721224 - Q.D2_b2) / 0.5862342   # -0.8%  n_dr_0p2_0p4 > 1 and D2_b2 < 4.721
        - 0.00748369 * max(0.0, Q.LHA - 0.3127275) / 0.01533444   # -0.7%  LHA > 0.3127
        + 0.007469816 * max(0.0, 1.002471 - Q.D2) / 0.1871081   # +0.7%  D2 < 1.002
        - 0.006068701 * max(0.0, 48.03125 - Q.pt_6) * max(0.0, 6.46415 - Q.log_sum_pt) / 0.7570358   # -0.6%  pt_6 < 48.03 and log_sum_pt < 6.464
        - 0.005920906 * max(0.0, 0.0009641429 - Q.lam1_plus_lam2) / 0.0002052288   # -0.6%  lam1_plus_lam2 < 0.0009641
        + 0.00523459 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 0.04000854 - Q.abseta_0) / 0.001250144   # +0.5%  log_sum_pt > 6.67 and abseta_0 < 0.04001
        - 0.005106044 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.eccentricity - 0.4829602) / 8.513077e-05   # -0.5%  lam2 > 0.0001948 and eccentricity > 0.483
        - 0.004383902 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, 6.538559 - Q.log_sum_pt) / 0.0001231869   # -0.4%  lam2 > 0.0001948 and log_sum_pt < 6.539
        - 0.004359589 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -0.4%  lam2 > 0.003408
        + 0.003991358 * max(0.0, Q.centroid_offset - 0.02355416) / 0.003991673   # +0.4%  centroid_offset > 0.02355
        + 0.003913042 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.109269   # +0.4%  log_sum_pt > 6.67 and n_dr_0p1_0p2 < 3
        - 0.003906021 * max(0.0, 0.02563286 - Q.M2) / 0.001872375   # -0.4%  M2 < 0.02563
        - 0.003367305 * max(0.0, Q.n_pt_above_50 - 6.0) / 0.2884353   # -0.3%  n_pt_above_50 > 6
        + 0.003345316 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0009425079   # +0.3%  lam1 < 0.006507 and n_pt_above_50 > 6
        + 0.003138949 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, Q.pt_3 - 60.59375) / 0.05411994   # +0.3%  lam1 < 0.006507 and pt_3 > 60.59
        + 0.003074802 * max(0.0, 48.03125 - Q.pt_6) * max(0.0, 0.1236856 - Q.D2_b2) / 0.1641634   # +0.3%  pt_6 < 48.03 and D2_b2 < 0.1237
        - 0.003058759 * max(0.0, Q.sum_z_dr2_top5 - 0.01148293) / 0.00154713   # -0.3%  sum_z_dr2_top5 > 0.01148
        - 0.003005148 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.02644207 - Q.mean_eta) / 0.000997125   # -0.3%  sj2_dr > 0.1683 and mean_eta < 0.02644
        + 0.002805307 * max(0.0, 0.04581318 - Q.M3) / 0.002043736   # +0.3%  M3 < 0.04581
        - 0.002351773 * max(0.0, Q.e3 - 0.0005116989) / 1.444524e-05   # -0.2%  e3 > 0.0005117
        + 0.001677232 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +0.2%  sj2_dr > 0.3004
        - 0.001532911 * max(0.0, Q.e3 - 8.147744e-05) * max(0.0, 33.21875 - Q.pt_7) / 0.0001092603   # -0.2%  e3 > 8.148e-05 and pt_7 < 33.22
        - 0.001373932 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, 0.01265416 - Q.zdr_7) / 1.587918e-06   # -0.1%  lam2 > 0.0001948 and zdr_7 < 0.01265
        + 0.001135166 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.1%  sum_pt > 988.4
        + 0.0005602833 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) * max(0.0, Q.sj3_dr23 - 0.3661115) / 0.01048748   # +0.1%  n_dr_0p05_0p1 < 5 and sj3_dr23 > 0.3661
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 43.3;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 43.29823 * (-0.04853188
        + 0.1315306 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +13.2%  lam1_plus_lam2 < 0.008678
        + 0.1220632 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +12.2%  lam1_plus_lam2 < 0.01324
        + 0.06737676 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +6.7%  LHA < 0.3033
        - 0.06482458 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # -6.5%  LHA < 0.3127
        - 0.06285214 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # -6.3%  sum_z_dr < 0.0717
        - 0.04926313 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -4.9%  sum_zz_dr2 < 0.008169
        - 0.04706815 * max(0.0, 0.006390125 - Q.sum_zz_dr2) / 0.002953508   # -4.7%  sum_zz_dr2 < 0.00639
        - 0.04496183 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -4.5%  lam1 < 0.008376
        + 0.03491767 * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.00293624   # +3.5%  sum_z_dr2 < 0.006679
        + 0.03312579 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +3.3%  centroid_offset < 0.03776
        - 0.01937432 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -1.9%  sj2_dr > 0.1779
        + 0.01888145 * max(0.0, 0.03459477 - Q.tau1) * max(0.0, 0.002915531 - Q.sum_z_dr2_top3) / 2.198359e-05   # +1.9%  tau1 < 0.03459 and sum_z_dr2_top3 < 0.002916
        - 0.01682062 * max(0.0, 0.005019719 - Q.lam1_plus_lam2) / 0.001903424   # -1.7%  lam1_plus_lam2 < 0.00502
        - 0.01677496 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # -1.7%  sum_z_dr2 < 0.003563
        - 0.01495334 * max(0.0, 0.04447357 - Q.e2) / 0.02019334   # -1.5%  e2 < 0.04447
        - 0.01444508 * max(0.0, 0.03459477 - Q.tau1) / 0.008474553   # -1.4%  tau1 < 0.03459
        + 0.01374275 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +1.4%  z_7 > 0.01686
        + 0.01251332 * max(0.0, Q.sj2_dr - 0.09419022) / 0.07816795   # +1.3%  sj2_dr > 0.09419
        + 0.01220293 * max(0.0, 0.2809341 - Q.LHA) / 0.06403989   # +1.2%  LHA < 0.2809
        - 0.01195898 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 118.5 - Q.pt_2) / 0.5820432   # -1.2%  centroid_offset < 0.03776 and pt_2 < 118.5
        + 0.01066029 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # +1.1%  sum_zz_dr2 < 0.003013
        - 0.0101065 * max(0.0, 0.03360421 - Q.sum_z_dr) / 0.006496997   # -1.0%  sum_z_dr < 0.0336
        + 0.01000832 * max(0.0, Q.log_sum_pt - 6.572938) / 0.08631512   # +1.0%  log_sum_pt > 6.573
        + 0.0096573 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +1.0%  planar_flow < 0.2534
        + 0.009362559 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # +0.9%  lam2 < 0.0005373
        + 0.008856768 * max(0.0, 0.006390125 - Q.sum_zz_dr2) * max(0.0, 118.5 - Q.pt_2) / 0.07535924   # +0.9%  sum_zz_dr2 < 0.00639 and pt_2 < 118.5
        + 0.00839617 * max(0.0, 0.02045966 - Q.e2) / 0.005531925   # +0.8%  e2 < 0.02046
        - 0.008345219 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 53.4375 - Q.pt_7) / 0.4342437   # -0.8%  centroid_offset < 0.03776 and pt_7 < 53.44
        - 0.008211878 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # -0.8%  tau1 < 0.09539
        + 0.007674957 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 0.1561228 - Q.z_3rd) / 0.0006510707   # +0.8%  centroid_offset < 0.03776 and z_3rd < 0.1561
        - 0.006179407 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.01364517 - Q.tau3) / 4.041223e-05   # -0.6%  centroid_offset < 0.01438 and tau3 < 0.01365
        - 0.006022338 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 3.916009 - Q.D3) / 0.2165298   # -0.6%  log_sum_pt > 6.573 and D3 < 3.916
        - 0.005907515 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) * max(0.0, 28.39396 - Q.pt1_dr01) / 0.2052503   # -0.6%  lam1_plus_lam2 < 0.01324 and pt1_dr01 < 28.39
        + 0.005905311 * max(0.0, Q.sj2_dr - 0.1872617) / 0.02732165   # +0.6%  sj2_dr > 0.1873
        - 0.005721219 * max(0.0, 0.006390125 - Q.sum_zz_dr2) * max(0.0, 0.1471389 - Q.z_3rd) / 6.771565e-05   # -0.6%  sum_zz_dr2 < 0.00639 and z_3rd < 0.1471
        + 0.004932878 * max(0.0, 0.09538712 - Q.tau1) * max(0.0, 62.25 - Q.pt_6) / 0.9985459   # +0.5%  tau1 < 0.09539 and pt_6 < 62.25
        - 0.004784517 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.08543881 - Q.sj3_dr_min) / 0.0002784225   # -0.5%  centroid_offset < 0.01438 and sj3_dr_min < 0.08544
        + 0.004621456 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.0586137 - Q.z_7) / 0.002292896   # +0.5%  log_sum_pt > 6.573 and z_7 < 0.05861
        - 0.004073535 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # -0.4%  centroid_offset > 0.0499 and D2 < 2.844
        - 0.003914272 * max(0.0, Q.pt_7 - 29.04219) / 7.785987   # -0.4%  pt_7 > 29.04
        - 0.003746816 * max(0.0, 62.25 - Q.pt_6) / 22.34324   # -0.4%  pt_6 < 62.25
        - 0.003581681 * max(0.0, 0.01437952 - Q.centroid_offset) / 0.004306483   # -0.4%  centroid_offset < 0.01438
        - 0.003558688 * max(0.0, 0.001570267 - Q.mean_phi2) / 0.0006563898   # -0.4%  mean_phi2 < 0.00157
        + 0.003424424 * max(0.0, Q.z_dr_0_0p05 - 0.6080732) / 0.1760045   # +0.3%  z_dr_0_0p05 > 0.6081
        - 0.003278454 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) * max(0.0, 6.502799 - Q.log_sum_pt) / 0.0004341009   # -0.3%  lam1_plus_lam2 < 0.01324 and log_sum_pt < 6.503
        - 0.00247755 * max(0.0, Q.log_sum_pt - 6.733425) / 0.02644661   # -0.2%  log_sum_pt > 6.733
        + 0.00244482 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.05932655 - Q.tau21_b2) / 7.260318e-05   # +0.2%  centroid_offset < 0.01438 and tau21_b2 < 0.05933
        - 0.002264783 * max(0.0, 0.02689598 - Q.sum_z_dr) / 0.004330137   # -0.2%  sum_z_dr < 0.0269
        - 0.002205562 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.2%  planar_flow < 0.2534 and sum_pt < 840
        + 0.002139015 * max(0.0, 0.001570267 - Q.mean_phi2) * max(0.0, Q.M2 - 0.01411669) / 4.456624e-05   # +0.2%  mean_phi2 < 0.00157 and M2 > 0.01412
        + 0.00194565 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.09607851 - Q.M3) / 9.192261e-05   # +0.2%  centroid_offset < 0.01438 and M3 < 0.09608
        + 0.001714899 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.2%  sj2_dr > 0.2688
        - 0.001626495 * max(0.0, 29.90625 - Q.pt_6) / 1.385454   # -0.2%  pt_6 < 29.91
        - 0.001599355 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001130645 - Q.lam2) / 1.843299e-05   # -0.2%  sj2_dr > 0.1779 and lam2 < 0.001131
        + 0.001394858 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.2691019 - Q.tau21_b2) / 0.000521804   # +0.1%  centroid_offset < 0.01438 and tau21_b2 < 0.2691
        - 0.001340284 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.0005116989 - Q.e3) / 2.655339e-07   # -0.1%  centroid_offset > 0.0499 and e3 < 0.0005117
        - 0.001331777 * max(0.0, 0.04447357 - Q.e2) * max(0.0, 1.002471 - Q.D2) / 0.001107726   # -0.1%  e2 < 0.04447 and D2 < 1.002
        - 0.001280711 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -0.1%  pt_6 < 39.75
        + 0.001278123 * max(0.0, Q.eccentricity - 0.9884745) / 0.001961982   # +0.1%  eccentricity > 0.9885
        - 0.001238223 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 37.15625 - Q.pt_7) / 0.5537514   # -0.1%  planar_flow < 0.2534 and pt_7 < 37.16
        - 0.001226572 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 1.050302e-05 - Q.e3) / 1.771164e-07   # -0.1%  planar_flow < 0.2534 and e3 < 1.05e-05
        + 0.0007476052 * max(0.0, 0.005019719 - Q.lam1_plus_lam2) * max(0.0, Q.sj3_dr23 - 0.1414609) / 1.353256e-05   # +0.1%  lam1_plus_lam2 < 0.00502 and sj3_dr23 > 0.1415
        - 0.0007399437 * max(0.0, 0.008168571 - Q.sum_zz_dr2) * max(0.0, Q.dr01 - 0.05595395) / 2.976641e-05   # -0.1%  sum_zz_dr2 < 0.008169 and dr01 > 0.05595
        - 0.0002217446 * max(0.0, 0.0030133 - Q.sum_zz_dr2) * max(0.0, Q.sum_z_dr2_top3 - 0.003952582) / 2.119623e-08   # -0.0%  sum_zz_dr2 < 0.003013 and sum_z_dr2_top3 > 0.003953
        + 0.0001978766 * max(0.0, Q.log_sum_pt - 6.733425) * max(0.0, Q.n_pt_above_50 - 7.0) / 0.002727108   # +0.0%  log_sum_pt > 6.733 and n_pt_above_50 > 7
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.8558;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.855799 * (-0.3747154
        + 0.2806161 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # +28.1%  sum_z_dr2 > 0.01883
        + 0.1154889 * max(0.0, Q.sum_z_dr2 - 0.02530566) / 0.0002851418   # +11.5%  sum_z_dr2 > 0.02531
        - 0.108081 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -10.8%  e2 > 0.06344
        - 0.09183308 * max(0.0, Q.ptdr0_4 - 15.26525) * max(0.0, 988.4078 - Q.sum_pt) / 57.72599   # -9.2%  ptdr0_4 > 15.27 and sum_pt < 988.4
        - 0.06732254 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, 53.4375 - Q.pt_7) / 0.01439677   # -6.7%  sum_z_dr2 > 0.01883 and pt_7 < 53.44
        + 0.06665884 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) * max(0.0, Q.log_sum_pt - 6.327379) / 8.505384e-05   # +6.7%  lam1_plus_lam2 > 0.01324 and log_sum_pt > 6.327
        - 0.06464755 * max(0.0, Q.sum_z_dr2 - 0.02530566) * max(0.0, 868.5094 - Q.sum_pt) / 0.1051521   # -6.5%  sum_z_dr2 > 0.02531 and sum_pt < 868.5
        - 0.05748363 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, 0.7974684 - Q.planar_flow) / 0.000314313   # -5.7%  sum_z_dr2 > 0.01883 and planar_flow < 0.7975
        + 0.04795755 * max(0.0, Q.ptdr0_4 - 15.26525) / 0.1831584   # +4.8%  ptdr0_4 > 15.27
        + 0.04423012 * max(0.0, Q.ptdr0_4 - 15.26525) * max(0.0, 763.825 - Q.sum_pt) / 21.50512   # +4.4%  ptdr0_4 > 15.27 and sum_pt < 763.8
        - 0.0194512 * max(0.0, Q.zdr_0 - 0.03981924) / 0.0005695785   # -1.9%  zdr_0 > 0.03982
        + 0.01547036 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +1.5%  centroid_offset > 0.0499
        - 0.01299567 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sum_pt - 559.6875) / 0.05572984   # -1.3%  e2 > 0.06344 and sum_pt > 559.7
        - 0.007763435 * max(0.0, Q.sum_z_dr2_top2 - 0.02260535) * max(0.0, Q.sum_pt - 527.1781) / 0.01862775   # -0.8%  sum_z_dr2_top2 > 0.02261 and sum_pt > 527.2
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 26.71;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.70927 * (0.03081201
        + 0.2382534 * max(0.0, 0.1484084 - Q.sum_z_dr) / 0.0905999   # +23.8%  sum_z_dr < 0.1484
        + 0.05809018 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +5.8%  lam1_plus_lam2 < 0.01324
        - 0.05392758 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # -5.4%  e3 < 0.0001869
        - 0.05088391 * max(0.0, Q.sum_pt_top5 - 482.2812) / 137.9506   # -5.1%  sum_pt_top5 > 482.3
        + 0.04526011 * max(0.0, Q.sum_pt_top5 - 482.2812) * max(0.0, 3.0374e-08 - Q.e4) / 4.059025e-06   # +4.5%  sum_pt_top5 > 482.3 and e4 < 3.037e-08
        - 0.04363305 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -4.4%  sum_z_dr < 0.1484 and log_sum_pt < 6.804
        - 0.0435955 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -4.4%  e2 < 0.08001
        + 0.04332181 * max(0.0, 0.02530566 - Q.sum_z_dr2) / 0.01914557   # +4.3%  sum_z_dr2 < 0.02531
        - 0.03996989 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # -4.0%  lam1_plus_lam2 < 0.008678
        + 0.03568003 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +3.6%  lam1 < 0.006507
        + 0.03093203 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +3.1%  lam1 < 0.01643
        - 0.02819022 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 1.399607e-06   # -2.8%  e3 < 8.148e-05 and centroid_offset < 0.03776
        - 0.02739933 * max(0.0, 0.007520088 - Q.sum_z_dr2) / 0.00354499   # -2.7%  sum_z_dr2 < 0.00752
        + 0.02722262 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +2.7%  e3 < 8.148e-05
        - 0.02201584 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -2.2%  tau1 < 0.1136
        + 0.02083884 * max(0.0, Q.z_7 - 0.04624032) / 0.01204993   # +2.1%  z_7 > 0.04624
        - 0.02007326 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -2.0%  sum_z_dr < 0.1484 and pt_7 < 38.53
        - 0.01751827 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # -1.8%  C2_b2 < 0.009033
        - 0.01751213 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.z_dr_0_0p05 - 0.3658817) / 0.004662966   # -1.8%  lam1 < 0.01643 and z_dr_0_0p05 > 0.3659
        - 0.01552802 * max(0.0, Q.pt_7 - 31.85938) / 5.945918   # -1.6%  pt_7 > 31.86
        - 0.01510431 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.0003061234 - Q.lam2) / 2.082734e-05   # -1.5%  sum_z_dr < 0.1484 and lam2 < 0.0003061
        - 0.01241908 * max(0.0, 50.25 - Q.pt_6) / 11.66482   # -1.2%  pt_6 < 50.25
        + 0.01055793 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 40.04062 - Q.pt_7) / 652.1541   # +1.1%  sum_pt_top5 > 687.4 and pt_7 < 40.04
        - 0.008044617 * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.09151624   # -0.8%  sj3_dr_min < 0.1278
        - 0.008017737 * max(0.0, 788.4484 - Q.sum_pt) / 112.1576   # -0.8%  sum_pt < 788.4
        + 0.006399618 * max(0.0, Q.z_dr_0_0p05 - 0.6080732) / 0.1760045   # +0.6%  z_dr_0_0p05 > 0.6081
        + 0.006138976 * max(0.0, Q.sum_pt_top5 - 482.2812) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 93.54358   # +0.6%  sum_pt_top5 > 482.3 and z_dr_0p05_0p1 < 0.846
        - 0.005022697 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 25.57812 - Q.pt_7) / 0.01870322   # -0.5%  lam1 < 0.01643 and pt_7 < 25.58
        - 0.004661605 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # -0.5%  log_sum_pt < 6.464
        + 0.004136825 * max(0.0, Q.pt_7 - 31.85938) * max(0.0, 0.01391455 - Q.zdr_1) / 0.03740834   # +0.4%  pt_7 > 31.86 and zdr_1 < 0.01391
        - 0.004068589 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, 0.01391455 - Q.zdr_1) / 0.4751719   # -0.4%  sum_pt < 788.4 and zdr_1 < 0.01391
        - 0.003665913 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, 2.468971 - Q.D3) / 151.1611   # -0.4%  sum_pt < 788.4 and D3 < 2.469
        - 0.002763697 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -0.3%  sj3_dr_max > 0.2337
        - 0.002737636 * max(0.0, 27.57812 - Q.pt_6) / 0.9875435   # -0.3%  pt_6 < 27.58
        + 0.002649644 * max(0.0, 6.46415 - Q.log_sum_pt) * max(0.0, 0.06809619 - Q.C3) / 0.002112398   # +0.3%  log_sum_pt < 6.464 and C3 < 0.0681
        - 0.002624826 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # -0.3%  pt_5 < 24.58
        + 0.002536701 * max(0.0, 0.08000524 - Q.e2) * max(0.0, 60.03125 - Q.pt_4) / 0.4322666   # +0.3%  e2 < 0.08001 and pt_4 < 60.03
        + 0.002444496 * max(0.0, 50.25 - Q.pt_6) * max(0.0, Q.zdr_0 - 0.001113887) / 0.1706944   # +0.2%  pt_6 < 50.25 and zdr_0 > 0.001114
        + 0.002134773 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # +0.2%  sum_pt_top5 > 840
        - 0.001823845 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.2%  sum_pt > 988.4
        + 0.001710915 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.2%  sum_pt_top5 > 687.4
        + 0.001482716 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # +0.1%  z_5 < 0.02818
        + 0.001402712 * max(0.0, 6.46415 - Q.log_sum_pt) * max(0.0, 0.009612129 - Q.zdr_1) / 0.0001139799   # +0.1%  log_sum_pt < 6.464 and zdr_1 < 0.009612
        + 0.001135413 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.08525808 - Q.dr_2) / 2.250996   # +0.1%  sum_pt_top5 > 687.4 and dr_2 < 0.08526
        - 0.001132909 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, 0.4309923 - Q.tau32) / 0.003864959   # -0.1%  sj3_dr_max > 0.2337 and tau32 < 0.431
        - 0.0009491479 * max(0.0, Q.z_top5 - 0.8919245) / 0.004621611   # -0.1%  z_top5 > 0.8919
        - 0.0009123164 * max(0.0, 24.57812 - Q.pt_5) * max(0.0, 169.875 - Q.pt_1) / 8.157701   # -0.1%  pt_5 < 24.58 and pt_1 < 169.9
        + 0.0009101217 * max(0.0, 24.57812 - Q.pt_5) * max(0.0, 0.1809419 - Q.z_2nd) / 0.00809881   # +0.1%  pt_5 < 24.58 and z_2nd < 0.1809
        - 0.000794657 * max(0.0, 6.46415 - Q.log_sum_pt) * max(0.0, 31.125 - Q.pt_4) / 0.01308589   # -0.1%  log_sum_pt < 6.464 and pt_4 < 31.12
        - 0.0006007679 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 0.0005193389   # -0.1%  sj3_dr_max > 0.2337 and z_dr_0p05_0p1 > 0.6748
        - 0.0005847275 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, Q.pt_7 - 37.15625) / 0.0438374   # -0.1%  sj3_dr_max > 0.2337 and pt_7 > 37.16
        - 0.0003247791 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.07092092 - Q.M3) / 0.05338926   # -0.0%  sum_pt > 988.4 and M3 < 0.07092
        + 0.0002592472 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.z_6 - 0.03932388) / 0.02599753   # +0.0%  sum_pt > 988.4 and z_6 > 0.03932
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 35.34;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.3432 * (-0.004359117
        - 0.09137599 * max(0.0, Q.sum_z_dr2 - 0.006679471) / 0.002702003   # -9.1%  sum_z_dr2 > 0.006679
        - 0.08691383 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -8.7%  sum_z_dr2 > 0.00752
        + 0.06535012 * max(0.0, Q.sum_zz_dr2 - 0.006390125) / 0.002450953   # +6.5%  sum_zz_dr2 > 0.00639
        + 0.0524515 * max(0.0, Q.sum_z_dr - 0.04081947) / 0.0270611   # +5.2%  sum_z_dr > 0.04082
        - 0.04651639 * max(0.0, 0.233678 - Q.sj3_dr_max) / 0.09057682   # -4.7%  sj3_dr_max < 0.2337
        + 0.04008642 * max(0.0, Q.sum_zz_dr2 - 0.00718279) / 0.002233568   # +4.0%  sum_zz_dr2 > 0.007183
        + 0.0350338 * max(0.0, Q.sum_z_dr - 0.02689598) / 0.03613842   # +3.5%  sum_z_dr > 0.0269
        + 0.03353682 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # +3.4%  sum_z_dr2 > 0.008678
        - 0.0327963 * max(0.0, Q.sum_zz_dr2 - 0.01716248) / 0.0007604864   # -3.3%  sum_zz_dr2 > 0.01716
        - 0.03232859 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # -3.2%  sum_z_dr > 0.08724
        - 0.03095535 * max(0.0, Q.sum_zz_dr2 - 0.008168571) / 0.002013747   # -3.1%  sum_zz_dr2 > 0.008169
        + 0.02863058 * max(0.0, 0.1115136 - Q.planar_flow) / 0.03343187   # +2.9%  planar_flow < 0.1115
        + 0.02766119 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +2.8%  sj2_dr < 0.1592
        - 0.02317017 * max(0.0, Q.tau1 - 0.03459477) / 0.03725748   # -2.3%  tau1 > 0.03459
        - 0.02287172 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -2.3%  sj2_dr < 0.1295
        - 0.02235907 * max(0.0, 0.005005443 - Q.sum_z_dr2_top5) / 0.002177006   # -2.2%  sum_z_dr2_top5 < 0.005005
        + 0.02210716 * max(0.0, 0.00422683 - Q.sum_z_dr2_top5) / 0.001731922   # +2.2%  sum_z_dr2_top5 < 0.004227
        - 0.01764662 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.158545   # -1.8%  n_dr_0p05_0p1 < 5
        - 0.01764285 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # -1.8%  planar_flow < 0.1115 and lam1 < 0.01643
        - 0.01635768 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # -1.6%  sj3_dr_max < 0.2134
        - 0.01594747 * max(0.0, Q.sum_z_dr2 - 0.008678045) * max(0.0, Q.log_sum_pt - 6.267538) / 0.0002118474   # -1.6%  sum_z_dr2 > 0.008678 and log_sum_pt > 6.268
        - 0.0152254 * max(0.0, Q.sum_zz_dr2 - 0.002074109) / 0.004484017   # -1.5%  sum_zz_dr2 > 0.002074
        - 0.01347416 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -1.3%  centroid_offset > 0.03776
        + 0.01263435 * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 0.001433269   # +1.3%  sum_zz_dr2 > 0.01166
        + 0.01245315 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) / 0.09532154   # +1.2%  z_dr_0p1_0p2 < 0.1586
        + 0.01235556 * max(0.0, Q.tau1 - 0.09538712) / 0.01057268   # +1.2%  tau1 > 0.09539
        - 0.01188052 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # -1.2%  lam1_plus_lam2 > 0.01324
        + 0.01039978 * max(0.0, Q.sum_zz_dr2 - 0.002074109) * max(0.0, Q.log_sum_pt - 6.267538) / 0.0007382066   # +1.0%  sum_zz_dr2 > 0.002074 and log_sum_pt > 6.268
        - 0.009936607 * max(0.0, Q.psi_0p1 - 0.8155839) / 0.1042073   # -1.0%  psi_0p1 > 0.8156
        + 0.008535416 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +0.9%  LHA > 0.3033
        + 0.008499645 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.3724174   # +0.8%  z_dr_0p05_0p1 < 0.5883
        + 0.008366386 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067297   # +0.8%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        - 0.008092608 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # -0.8%  C2 < 0.03579
        - 0.008089876 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # -0.8%  z_dr_0p05_0p1 > 0.7509
        + 0.007537295 * max(0.0, Q.n_dr_0_0p05 - 5.0) / 1.093146   # +0.8%  n_dr_0_0p05 > 5
        + 0.007313285 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +0.7%  e2 > 0.04111
        + 0.006779829 * max(0.0, Q.sum_zz_dr2 - 0.008168571) * max(0.0, Q.sum_pt - 488.9312) / 0.1696086   # +0.7%  sum_zz_dr2 > 0.008169 and sum_pt > 488.9
        + 0.006706304 * max(0.0, 0.1594012 - Q.sj3_dr_max) / 0.04422261   # +0.7%  sj3_dr_max < 0.1594
        + 0.006389413 * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 0.8908874   # +0.6%  n_dr_0p1_0p2 > 1
        + 0.006233991 * max(0.0, Q.psi_0p1 - 0.6332416) * max(0.0, Q.eccentricity - 0.9704496) / 0.001856694   # +0.6%  psi_0p1 > 0.6332 and eccentricity > 0.9704
        + 0.005777458 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) * max(0.0, Q.sum_pt - 488.9312) / 0.1069057   # +0.6%  lam1_plus_lam2 > 0.01324 and sum_pt > 488.9
        - 0.004860172 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.5%  centroid_offset > 0.0499
        - 0.004766665 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -0.5%  planar_flow < 0.1115 and lam1 < 0.00733
        - 0.004574811 * max(0.0, Q.sum_zz_dr2 - 0.002074109) * max(0.0, 0.1872617 - Q.sj2_dr) / 2.014151e-05   # -0.5%  sum_zz_dr2 > 0.002074 and sj2_dr < 0.1873
        + 0.004163702 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # +0.4%  e2 > 0.03556
        - 0.003610822 * max(0.0, Q.eccentricity - 0.9781608) / 0.005665615   # -0.4%  eccentricity > 0.9782
        - 0.003585634 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.4%  psi_0p1 > 0.9761
        - 0.003128081 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.3%  planar_flow < 0.1115 and centroid_offset < 0.01838
        - 0.002997679 * max(0.0, Q.centroid_offset - 0.02355416) / 0.003991673   # -0.3%  centroid_offset > 0.02355
        + 0.002865356 * max(0.0, 1.960701e-05 - Q.e3) / 8.488392e-06   # +0.3%  e3 < 1.961e-05
        - 0.002539214 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 0.002231562   # -0.3%  planar_flow < 0.1115 and z_dr_0p05_0p1 > 0.6748
        - 0.002294534 * max(0.0, Q.n_dr_0p1_0p2 - 2.0) / 0.5480118   # -0.2%  n_dr_0p1_0p2 > 2
        - 0.001934429 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.eta_5 - 0.1122437) / 6.758882e-06   # -0.2%  centroid_offset > 0.0499 and eta_5 > 0.1122
        + 0.001798347 * max(0.0, Q.psi_0p1 - 0.8155839) * max(0.0, Q.centroid_offset - 0.03776099) / 4.931851e-05   # +0.2%  psi_0p1 > 0.8156 and centroid_offset > 0.03776
        + 0.001466869 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 3.351526e-05   # +0.1%  planar_flow < 0.1115 and lam1_plus_lam2 < 0.006097
        - 0.001461476 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 658.125 - Q.sum_pt_top5) / 2.950014   # -0.1%  planar_flow < 0.1115 and sum_pt_top5 < 658.1
        + 0.001455119 * max(0.0, Q.eccentricity - 0.9781608) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0001884489   # +0.1%  eccentricity > 0.9782 and D2_b2 < 0.08499
        + 0.001006793 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +0.1%  e2 > 0.05028
        - 0.0009072265 * max(0.0, Q.LHA - 0.2160559) * max(0.0, Q.sum_pt - 813.4156) / 0.5098489   # -0.1%  LHA > 0.2161 and sum_pt > 813.4
        - 0.0006171028 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, Q.dr0_5 - 0.1925897) / 0.0001018035   # -0.1%  z_dr_0p05_0p1 > 0.7509 and dr0_5 > 0.1926
        + 0.0005759922 * max(0.0, Q.centroid_offset - 0.02355416) * max(0.0, Q.eccentricity - 0.9841966) / 9.844275e-06   # +0.1%  centroid_offset > 0.02355 and eccentricity > 0.9842
        + 0.0003441666 * max(0.0, Q.e2 - 0.03556091) * max(0.0, Q.sum_pt_top5 - 658.125) / 0.0353694   # +0.0%  e2 > 0.03556 and sum_pt_top5 > 658.1
        - 0.0003319235 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, Q.dr0_5 - 0.290089) / 2.769865e-05   # -0.0%  z_dr_0p05_0p1 > 0.7509 and dr0_5 > 0.2901
        + 0.0002932597 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, Q.dr0_5 - 0.171998) / 0.0001408058   # +0.0%  z_dr_0p05_0p1 > 0.7509 and dr0_5 > 0.172
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 39.48;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 39.48497 * (-0.02952543
        + 0.1289051 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +12.9%  sum_z_dr2 < 0.01324
        - 0.119604 * max(0.0, 0.007520088 - Q.sum_z_dr2) / 0.00354499   # -12.0%  sum_z_dr2 < 0.00752
        - 0.09549272 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # -9.5%  sum_zz_dr2 < 0.01166
        - 0.0626111 * max(0.0, 0.06344108 - Q.e2) / 0.03657078   # -6.3%  e2 < 0.06344
        - 0.06056296 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -6.1%  tau1 < 0.1136
        + 0.0573704 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +5.7%  lam1 < 0.00733
        + 0.05287954 * max(0.0, 0.006390125 - Q.sum_zz_dr2) / 0.002953508   # +5.3%  sum_zz_dr2 < 0.00639
        + 0.04754774 * max(0.0, 0.05028464 - Q.e2) / 0.02502152   # +4.8%  e2 < 0.05028
        - 0.03845119 * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.00293624   # -3.8%  sum_z_dr2 < 0.006679
        + 0.03795488 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +3.8%  e2 < 0.04111
        + 0.0314877 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # +3.1%  tau1 < 0.1028
        - 0.02635514 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -2.6%  lam1 < 0.008376
        + 0.02440857 * max(0.0, 0.01716248 - Q.sum_zz_dr2) / 0.0120354   # +2.4%  sum_zz_dr2 < 0.01716
        + 0.02219621 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +2.2%  lam2 < 0.003408
        + 0.01827185 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +1.8%  sj3_dr_max > 0.179
        + 0.01327305 * max(0.0, Q.psi_0p1 - 0.6332416) / 0.2477542   # +1.3%  psi_0p1 > 0.6332
        + 0.0100744 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +1.0%  lam2 < 0.001131
        + 0.009731557 * max(0.0, Q.sj3_dr_max - 0.1879486) * max(0.0, 169.875 - Q.pt_1) / 2.480285   # +1.0%  sj3_dr_max > 0.1879 and pt_1 < 169.9
        - 0.009062639 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # -0.9%  sum_z_dr2_top3 < 0.002152
        + 0.008807327 * max(0.0, 0.002635418 - Q.lam1_plus_lam2) / 0.0007941347   # +0.9%  lam1_plus_lam2 < 0.002635
        - 0.007962021 * max(0.0, Q.sj3_dr_max - 0.1426152) * max(0.0, 169.875 - Q.pt_1) / 3.727834   # -0.8%  sj3_dr_max > 0.1426 and pt_1 < 169.9
        + 0.007897249 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.2931906) / 0.001536651   # +0.8%  N2 < 0.2233 and LHA > 0.2932
        - 0.007728602 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.0001197728   # -0.8%  N2 < 0.2233 and sum_z_dr2 > 0.008678
        + 0.007293299 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +0.7%  lam1 < 0.005954
        - 0.006859631 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1872617 - Q.sj2_dr) / 0.0007896743   # -0.7%  N2 < 0.2233 and sj2_dr < 0.1873
        + 0.006123165 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sum_pt_top5 - 402.625) / 15.59919   # +0.6%  planar_flow < 0.195 and sum_pt_top5 > 402.6
        - 0.005963146 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # -0.6%  z_dr_0p05_0p1 > 0.7509
        - 0.005899367 * max(0.0, Q.psi_0p1 - 0.4008925) / 0.4528526   # -0.6%  psi_0p1 > 0.4009
        - 0.005602531 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -0.6%  sj3_dr_max > 0.2337
        + 0.004712123 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +0.5%  N2 < 0.2233
        + 0.004640857 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.001172117   # +0.5%  z_dr_0p05_0p1 > 0.7509 and z_dr_0p2_0p4 < 0.05644
        + 0.004470018 * max(0.0, 0.1515405 - Q.z_dr_0_0p05) / 0.0478325   # +0.4%  z_dr_0_0p05 < 0.1515
        - 0.004437332 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 8.147744e-05 - Q.e3) / 2.656078e-06   # -0.4%  N2 < 0.2233 and e3 < 8.148e-05
        - 0.00414677 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -0.4%  sj3_dr_max > 0.2623
        + 0.004024934 * max(0.0, 0.08082334 - Q.dr_0) / 0.0377444   # +0.4%  dr_0 < 0.08082
        - 0.003848449 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.4%  psi_0p1 > 0.9761
        - 0.003653744 * max(0.0, Q.max_dr - 0.121681) / 0.03562206   # -0.4%  max_dr > 0.1217
        - 0.00358495 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # -0.4%  sj3_dr_max > 0.1879
        - 0.003459471 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -0.3%  zdr_0 < 0.02118
        - 0.00303213 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3255822) / 0.0007869707   # -0.3%  N2 < 0.2233 and LHA > 0.3256
        + 0.002965305 * max(0.0, 1.0 - Q.n_dr_0p1_0p2) / 0.5188185   # +0.3%  n_dr_0p1_0p2 < 1
        + 0.002796837 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1591713 - Q.sj2_dr) / 0.0002571871   # +0.3%  N2 < 0.2233 and sj2_dr < 0.1592
        - 0.002700858 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, 182.125 - Q.pt_1) / 1.935314   # -0.3%  sj3_dr_max > 0.2337 and pt_1 < 182.1
        + 0.002019428 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +0.2%  N2 < 0.2233 and sum_pt_top5 > 430.8
        - 0.001412902 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) * max(0.0, Q.eccentricity - 0.9598562) / 4.406774e-06   # -0.1%  sum_z_dr2_top3 < 0.002152 and eccentricity > 0.9599
        - 0.001410461 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.mean_phi - -0.01753483) / 0.001413063   # -0.1%  planar_flow < 0.195 and mean_phi > -0.01753
        - 0.001396529 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 2.0 - Q.n_dr_0p05_0p1) / 0.05544434   # -0.1%  planar_flow < 0.195 and n_dr_0p05_0p1 < 2
        - 0.001146093 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, Q.pt_entropy - 1.797618) / 0.0002596149   # -0.1%  sum_z_dr2 < 0.006679 and pt_entropy > 1.798
        - 0.0009375292 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.04990367) / 8.981184e-07   # -0.1%  sum_z_dr2 < 0.01324 and centroid_offset > 0.0499
        + 0.0008849496 * max(0.0, Q.max_dr - 0.2215867) / 0.008157178   # +0.1%  max_dr > 0.2216
        - 0.0007996866 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, Q.dr12 - 0.1587481) / 1.436546e-06   # -0.1%  sum_z_dr2 < 0.006679 and dr12 > 0.1587
        - 0.0006300611 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 25.57812 - Q.pt_7) / 0.037942   # -0.1%  N2 < 0.2233 and pt_7 < 25.58
        + 0.0005115756 * max(0.0, 0.006390125 - Q.sum_zz_dr2) * max(0.0, Q.dr12 - 0.1587481) / 1.354517e-06   # +0.1%  sum_zz_dr2 < 0.00639 and dr12 > 0.1587
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0188747899159665, 0.7218254201680673, 2.081607142857143, 0.9463997899159664, 1.12399243697479, 1.7518224789915966, 0.9370163865546218, 1.827526680672269, 0.24232394957983194, 2.9819718487394957, 2.0170699579831934, 2.131138025210084, 0.0961218487394958, 3.290703361344538, 0.49688560924369746, 0.3502144957983193]
T = [2.3142068671218485, 1.3509267052914917, 3.336342850577731, 2.459996550026261, 2.8402707195378154]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +39%, n9 +22%, n5 -14%, n1 +12%, n0 -7%, n6 +4% ...
            + 0.3864998 * h[2] / H_AVG[2]
            + 0.2214696 * h[9] / H_AVG[9]
            - 0.1419349 * h[5] / H_AVG[5]
            + 0.12184 * h[1] / H_AVG[1]
            - 0.06879212 * h[0] / H_AVG[0]
            + 0.04428566 * h[6] / H_AVG[6]
            - 0.01517788 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -19%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5604607 * h[9] / H_AVG[9]
            - 0.1866376 * h[10] / H_AVG[10]
            + 0.08670126 * h[6] / H_AVG[6]
            - 0.07800149 * h[4] / H_AVG[4]
            + 0.06078544 * h[5] / H_AVG[5]
            + 0.01620251 * h[15] / H_AVG[15]
            + 0.01121101 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +12%, n14 -11%, n0 +10%, n6 -9% ...
            + 0.2395368 * h[11] / H_AVG[11]
            - 0.1418319 * h[3] / H_AVG[3]
            + 0.1198233 * h[7] / H_AVG[7]
            - 0.1116984 * h[14] / H_AVG[14]
            + 0.1049767 * h[0] / H_AVG[0]
            - 0.08776605 * h[6] / H_AVG[6]
            - 0.07216658 * h[15] / H_AVG[15]
            + 0.06935066 * h[13] / H_AVG[13]
            - 0.02793077 * h[9] / H_AVG[9]
            - 0.0181579 * h[8] / H_AVG[8]
            - 0.006761009 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -22%, n6 -14%, n14 +8%, n13 +7%, n9 -4% ...
            + 0.3482335 * h[7] / H_AVG[7]
            - 0.2164027 * h[3] / H_AVG[3]
            - 0.1428381 * h[6] / H_AVG[6]
            + 0.07574486 * h[14] / H_AVG[14]
            + 0.07315471 * h[13] / H_AVG[13]
            - 0.03788079 * h[9] / H_AVG[9]
            + 0.03667817 * h[1] / H_AVG[1]
            + 0.03569595 * h[4] / H_AVG[4]
            - 0.02224435 * h[15] / H_AVG[15]
            + 0.01112694 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +27%, n5 -15%, n4 +5%, n3 +2%, n12 -2% ...
            - 0.4706763 * h[13] / H_AVG[13]
            + 0.2663131 * h[10] / H_AVG[10]
            - 0.154195 * h[5] / H_AVG[5]
            + 0.04946678 * h[4] / H_AVG[4]
            + 0.02082548 * h[3] / H_AVG[3]
            - 0.01692125 * h[12] / H_AVG[12]
            + 0.01599698 * h[8] / H_AVG[8]
            + 0.005605071 * h[0] / H_AVG[0]
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
