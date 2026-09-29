"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.3%   (on for 86% of jets)
  neuron  9:  11.8%   (on for 66% of jets)
  neuron  7:  10.5%   (on for 59% of jets)
  neuron  6:   8.4%   (on for 36% of jets)
  neuron  3:   8.3%   (on for 23% of jets)
  neuron 10:   7.9%   (on for 76% of jets)
  neuron  2:   7.3%   (on for 90% of jets)
  neuron  5:   6.9%   (on for 51% of jets)
  neuron 11:   6.3%   (on for 74% of jets)
  neuron 14:   3.8%   (on for 26% of jets)
  neuron  0:   3.8%   (on for 41% of jets)
  neuron  1:   3.5%   (on for 65% of jets)
  neuron  4:   3.2%   (on for 52% of jets)
  neuron 15:   2.7%   (on for 27% of jets)
  neuron  8:   1.0%   (on for 29% of jets)
  neuron 12:   0.4%   (on for 8% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.7% (the network: 65.8%); same class as the network for 90.2% of jets.

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
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z1                 pT share of subjet 1 of 3 (by pT)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.pt1_over_pt0           pT1 / pT0
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.z_3rd                  3rd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_6               |Δη| of particle 6
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_5                  ΔR between particle 5 and the hardest particle
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_5                  Δη of particle 5
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
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
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
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
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        n_for_90pct=ncum(0.9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z1=subjets(3)["z"][0],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        zdr_3=z[3] * dr[3],
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        z_3rd=zs[2],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        abseta_0=abs(eta[0]),
        abseta_6=abs(eta[6]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_5=math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_5=eta[5],
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
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
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
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
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 35.73;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.72522 * (0.1911598
        - 0.2197361 * Q.log_sum_pt / 6.542932   # -22.0%  log_sum_pt
        + 0.1131628 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +11.3%  width < 0.008678
        + 0.05610524 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +5.6%  girth2 < 0.01324
        - 0.05509446 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -5.5%  lam1 < 0.008376
        + 0.04264325 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # +4.3%  e2_sq < 0.01166
        - 0.03891333 * max(0.0, 0.005590289 - Q.girth2) / 0.002229311   # -3.9%  girth2 < 0.00559
        - 0.03820859 * max(0.0, 0.07608178 - Q.girth) / 0.02796784   # -3.8%  girth < 0.07608
        - 0.03549865 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -3.5%  girth < 0.08724
        + 0.03480514 * max(0.0, 0.01882765 - Q.girth2) / 0.01314296   # +3.5%  girth2 < 0.01883
        + 0.03433885 * max(0.0, 0.032347 - Q.e2) / 0.01171807   # +3.4%  e2 < 0.03235
        - 0.0340615 * max(0.0, 0.00718279 - Q.e2_sq) / 0.003528787   # -3.4%  e2_sq < 0.007183
        + 0.02352126 * max(0.0, 0.1116471 - Q.sd_rg) / 0.04385005   # +2.4%  sd_rg < 0.1116
        - 0.02194649 * max(0.0, 0.01357153 - Q.tau2) / 0.005299336   # -2.2%  tau2 < 0.01357
        + 0.0212346 * max(0.0, 0.06344108 - Q.e2) / 0.03657078   # +2.1%  e2 < 0.06344
        - 0.0211738 * max(0.0, 0.003562611 - Q.width) / 0.001184249   # -2.1%  width < 0.003563
        - 0.02038695 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # -2.0%  tau1 < 0.1028
        - 0.01995232 * max(0.0, 0.1773029 - Q.sd_rg) / 0.08115541   # -2.0%  sd_rg < 0.1773
        - 0.01899858 * max(0.0, 788.4484 - Q.sum_pt) / 112.1576   # -1.9%  sum_pt < 788.4
        - 0.01607177 * max(0.0, 0.004372139 - Q.width) / 0.00156571   # -1.6%  width < 0.004372
        + 0.01518783 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +1.5%  planar_flow < 0.1484
        + 0.01107946 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +1.1%  lam1 < 0.005954
        + 0.01107597 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 1.129616 - Q.D2_b2) / 0.04715373   # +1.1%  planar_flow < 0.1484 and D2_b2 < 1.13
        + 0.0107961 * max(0.0, Q.n_dr_0_0p05 - 5.0) / 1.093146   # +1.1%  n_dr_0_0p05 > 5
        - 0.01007622 * max(0.0, 0.01882765 - Q.girth2) * max(0.0, Q.centroid_offset - 0.009480685) / 8.306781e-05   # -1.0%  girth2 < 0.01883 and centroid_offset > 0.009481
        + 0.009337566 * max(0.0, 631.275 - Q.sum_pt_top5) / 94.34106   # +0.9%  sum_pt_top5 < 631.3
        - 0.008666888 * max(0.0, 0.007929074 - Q.girth2_top3) / 0.004560262   # -0.9%  girth2_top3 < 0.007929
        - 0.007971172 * max(0.0, 0.01517359 - Q.girth) / 0.001389803   # -0.8%  girth < 0.01517
        - 0.00714254 * max(0.0, 739.5 - Q.sum_pt) * max(0.0, 1.345805 - Q.D2_b2) / 59.18221   # -0.7%  sum_pt < 739.5 and D2_b2 < 1.346
        - 0.006132868 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.6%  centroid_offset > 0.0499
        - 0.006093342 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.001732318   # -0.6%  planar_flow < 0.1484 and centroid_offset < 0.0499
        - 0.005265225 * max(0.0, 0.06545715 - Q.sd_rg) / 0.02200467   # -0.5%  sd_rg < 0.06546
        + 0.003898401 * max(0.0, 0.01882765 - Q.girth2) * max(0.0, 0.09029177 - Q.M3) / 0.0001995224   # +0.4%  girth2 < 0.01883 and M3 < 0.09029
        + 0.003799207 * max(0.0, 0.007929074 - Q.girth2_top3) * max(0.0, Q.N3 - 0.7686247) / 0.00341809   # +0.4%  girth2_top3 < 0.007929 and N3 > 0.7686
        - 0.003656822 * max(0.0, 0.005590289 - Q.girth2) * max(0.0, 0.09310137 - Q.M3) / 3.230359e-05   # -0.4%  girth2 < 0.00559 and M3 < 0.0931
        - 0.002516055 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 0.2669656 - Q.D2_b2) / 3.951293e-06   # -0.3%  e3 < 8.148e-05 and D2_b2 < 0.267
        - 0.002454634 * max(0.0, 0.380911 - Q.D2_b2) * max(0.0, Q.pt_5 - 24.57812) / 3.196263   # -0.2%  D2_b2 < 0.3809 and pt_5 > 24.58
        - 0.002343167 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, Q.eccentricity - 0.9598562) / 1.260811   # -0.2%  sum_pt < 788.4 and eccentricity > 0.9599
        - 0.001565111 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.dr0_7 - 0.2405707) / 1.062493e-07   # -0.2%  e3 < 8.148e-05 and dr0_7 > 0.2406
        - 0.001316382 * max(0.0, 0.06345984 - Q.tau1) * max(0.0, Q.dr0_6 - 0.1755206) / 7.050329e-05   # -0.1%  tau1 < 0.06346 and dr0_6 > 0.1755
        + 0.0009903924 * max(0.0, 0.003562611 - Q.width) * max(0.0, Q.dr0_7 - 0.1786203) / 1.674303e-06   # +0.1%  width < 0.003563 and dr0_7 > 0.1786
        - 0.0009580703 * max(0.0, 0.06345984 - Q.tau1) * max(0.0, Q.dr_5 - 0.1631114) / 3.584581e-05   # -0.1%  tau1 < 0.06346 and dr_5 > 0.1631
        + 0.0009579834 * max(0.0, 0.004372139 - Q.width) * max(0.0, Q.dr0_6 - 0.1755206) / 1.823868e-06   # +0.1%  width < 0.004372 and dr0_6 > 0.1755
        + 0.0008649104 * max(0.0, 0.005590289 - Q.girth2) * max(0.0, Q.dr0_5 - 0.171998) / 2.694851e-06   # +0.1%  girth2 < 0.00559 and dr0_5 > 0.172
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 19.37;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.36895 * (0.1097621
        + 0.07289363 * max(0.0, 0.005834489 - Q.e2_sq) / 0.002579453   # +7.3%  e2_sq < 0.005834
        - 0.07201105 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # -7.2%  girth < 0.1019
        - 0.06906777 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # -6.9%  width < 0.008678
        - 0.06790674 * max(0.0, 0.00609665 - Q.girth2) / 0.002544039   # -6.8%  girth2 < 0.006097
        + 0.06635485 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1442881 - Q.dr_0) / 0.0005125595   # +6.6%  lam1 < 0.008376 and dr_0 < 0.1443
        + 0.0546296 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +5.5%  log_sum_pt > 6.378
        - 0.05155204 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -5.2%  z_7 < 0.06165
        + 0.04067229 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +4.1%  lam1 < 0.005954
        - 0.04049274 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # -4.0%  sj3_dr_max < 0.3456
        + 0.03940734 * max(0.0, Q.sum_pt - 667.0063) / 97.02637   # +3.9%  sum_pt > 667
        - 0.03924259 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.08082334 - Q.dr_0) / 0.01037756   # -3.9%  log_sum_pt > 6.378 and dr_0 < 0.08082
        - 0.03363074 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -3.4%  sum_pt_top5 > 531.2
        + 0.02935361 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.0001869378 - Q.e3) / 2.398316e-06   # +2.9%  z_7 < 0.06165 and e3 < 0.0001869
        + 0.02920078 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # +2.9%  tau1 < 0.1028
        - 0.0273123 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # -2.7%  sum_pt < 988.4
        + 0.0248031 * max(0.0, Q.log_sum_pt - 6.572938) / 0.08631512   # +2.5%  log_sum_pt > 6.573
        - 0.02478231 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # -2.5%  tau1 < 0.07284
        + 0.01986704 * max(0.0, 0.1594012 - Q.sj3_dr_max) / 0.04422261   # +2.0%  sj3_dr_max < 0.1594
        - 0.01893744 * max(0.0, 0.05240025 - Q.z_7) / 0.008882498   # -1.9%  z_7 < 0.0524
        - 0.01880393 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.007929074 - Q.girth2_top3) / 0.0005408911   # -1.9%  log_sum_pt > 6.573 and girth2_top3 < 0.007929
        - 0.01857216 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -1.9%  zdr_0 < 0.02118
        + 0.01660236 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +1.7%  lam1 < 0.002464
        + 0.01621418 * max(0.0, Q.pt_7 - 38.53125) / 2.781159   # +1.6%  pt_7 > 38.53
        + 0.01553812 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.03981924 - Q.zdr_0) / 0.0003895962   # +1.6%  z_7 < 0.06165 and zdr_0 < 0.03982
        - 0.01534808 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -1.5%  lam1 < 0.008376
        - 0.01500388 * max(0.0, 0.1326115 - Q.max_dr) / 0.03990327   # -1.5%  max_dr < 0.1326
        + 0.01061981 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 0.2691019 - Q.tau21_b2) / 0.0004590397   # +1.1%  lam2 < 0.003408 and tau21_b2 < 0.2691
        + 0.009850487 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.01403324 - Q.girth2_top2) / 0.0001680113   # +1.0%  z_7 < 0.06165 and girth2_top2 < 0.01403
        + 0.008173945 * max(0.0, Q.log_sum_pt - 6.638339) * max(0.0, 0.1410336 - Q.dr01) / 0.006766763   # +0.8%  log_sum_pt > 6.638 and dr01 < 0.141
        - 0.008033182 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 2.955458e-05 - Q.e3) / 7.137805e-05   # -0.8%  pt_7 > 34.53 and e3 < 2.955e-05
        - 0.007071697 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.pt_5 - 24.57812) / 0.1029929   # -0.7%  lam1 < 0.008376 and pt_5 > 24.58
        - 0.006004382 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.1410336 - Q.dr01) / 0.001621893   # -0.6%  z_7 < 0.06165 and dr01 < 0.141
        + 0.003816516 * max(0.0, Q.z_6 - 0.06727211) / 0.005920115   # +0.4%  z_6 > 0.06727
        + 0.003202616 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # +0.3%  lam1 < 0.008376 and centroid_offset > 0.02077
        - 0.002962524 * max(0.0, 0.05240025 - Q.z_7) * max(0.0, 0.1236856 - Q.D2_b2) / 0.000152235   # -0.3%  z_7 < 0.0524 and D2_b2 < 0.1237
        - 0.002064176 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.2%  pt_7 > 53.44
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 21.1;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.10299 * (0.03172094
        + 0.1527759 * Q.pt_7 / 34.64819   # +15.3%  pt_7
        - 0.1372227 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # -13.7%  z_7 > 0.02321
        - 0.1027621 * max(0.0, Q.LHA - 0.111565) / 0.1352185   # -10.3%  LHA > 0.1116
        + 0.06051776 * max(0.0, 840.0195 - Q.sum_pt) / 148.4153   # +6.1%  sum_pt < 840
        + 0.04619044 * max(0.0, Q.LHA - 0.09323897) / 0.1512502   # +4.6%  LHA > 0.09324
        - 0.04485117 * max(0.0, 0.0423228 - Q.C2) * max(0.0, 0.02415398 - Q.C2_b2) / 0.0004792616   # -4.5%  C2 < 0.04232 and C2_b2 < 0.02415
        + 0.04447373 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +4.4%  width < 0.01324
        - 0.04146019 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -4.1%  pt_7 < 53.44
        - 0.03924475 * max(0.0, 716.8828 - Q.sum_pt_top5) / 152.5573   # -3.9%  sum_pt_top5 < 716.9
        + 0.03693192 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.2507612 - Q.max_dr) / 0.00062784   # +3.7%  lam1 < 0.00733 and max_dr < 0.2508
        + 0.03620508 * max(0.0, Q.z_7 - 0.02807091) / 0.02541111   # +3.6%  z_7 > 0.02807
        + 0.03476821 * max(0.0, 0.0423228 - Q.C2) / 0.02012665   # +3.5%  C2 < 0.04232
        - 0.02841306 * max(0.0, 0.00718279 - Q.e2_sq) / 0.003528787   # -2.8%  e2_sq < 0.007183
        + 0.02543969 * max(0.0, 763.825 - Q.sum_pt) / 96.68366   # +2.5%  sum_pt < 763.8
        - 0.01779908 * max(0.0, 6.701242 - Q.log_sum_pt) * max(0.0, 1.129616 - Q.D2_b2) / 0.1081022   # -1.8%  log_sum_pt < 6.701 and D2_b2 < 1.13
        + 0.01755332 * max(0.0, 0.007520088 - Q.width) / 0.003544991   # +1.8%  width < 0.00752
        + 0.01581423 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 1.129616 - Q.D2_b2) / 0.07303201   # +1.6%  log_sum_pt < 6.606 and D2_b2 < 1.13
        + 0.01442665 * max(0.0, 6.701242 - Q.log_sum_pt) / 0.1935491   # +1.4%  log_sum_pt < 6.701
        - 0.01366379 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 62.25 - Q.pt_6) / 0.07984241   # -1.4%  lam1 < 0.00733 and pt_6 < 62.25
        + 0.0124408 * max(0.0, 0.02383244 - Q.zdr_0) / 0.01109706   # +1.2%  zdr_0 < 0.02383
        + 0.01060216 * max(0.0, 615.875 - Q.sum_pt) / 30.07766   # +1.1%  sum_pt < 615.9
        - 0.008778541 * max(0.0, 0.04447357 - Q.e2) / 0.02019334   # -0.9%  e2 < 0.04447
        - 0.008286314 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -0.8%  lam1 < 0.00733
        + 0.00690256 * max(0.0, 0.001653836 - Q.width) / 0.0004289067   # +0.7%  width < 0.001654
        + 0.006750987 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # +0.7%  lam1 < 0.003377
        - 0.006697768 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.7%  sum_pt > 988.4
        - 0.006630774 * max(0.0, 0.0030133 - Q.e2_sq) / 0.001065944   # -0.7%  e2_sq < 0.003013
        + 0.00559785 * max(0.0, Q.sum_pt - 937.0312) / 8.087955   # +0.6%  sum_pt > 937
        - 0.004566657 * max(0.0, 0.001101266 - Q.e2_sq) / 0.000308004   # -0.5%  e2_sq < 0.001101
        + 0.003943071 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.4%  log_sum_pt > 6.896
        - 0.003043226 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 0.01425934 - Q.abseta_0) / 0.1176352   # -0.3%  sum_pt_top5 > 791.1 and abseta_0 < 0.01426
        - 0.002987515 * max(0.0, 0.005576073 - Q.centroid_offset) * max(0.0, 0.01685631 - Q.C3) / 3.692635e-06   # -0.3%  centroid_offset < 0.005576 and C3 < 0.01686
        + 0.001129975 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.01425934 - Q.abseta_0) / 0.03515712   # +0.1%  sum_pt > 988.4 and abseta_0 < 0.01426
        - 0.001128023 * max(0.0, Q.sum_pt - 937.0312) * max(0.0, Q.n_pt_above_50 - 5.0) / 6.962984   # -0.1%  sum_pt > 937 and n_pt_above_50 > 5
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 16.34;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.34096 * (-0.1725871
        - 0.1473754 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # -14.7%  girth2 > 0.008678
        + 0.1315923 * max(0.0, Q.girth - 0.06663269) / 0.01393206   # +13.2%  girth > 0.06663
        + 0.09981793 * max(0.0, Q.e2_sq - 0.00718279) / 0.002233568   # +10.0%  e2_sq > 0.007183
        + 0.08387622 * max(0.0, Q.width - 0.00609665) / 0.002892623   # +8.4%  width > 0.006097
        - 0.06283519 * max(0.0, Q.e2_sq - 0.006390125) / 0.002450953   # -6.3%  e2_sq > 0.00639
        + 0.05781684 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # +5.8%  sj2_dr > 0.1779
        - 0.0546549 * max(0.0, Q.tau1 - 0.0811449) / 0.01489378   # -5.5%  tau1 > 0.08114
        - 0.04736241 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 7.0 - Q.n_dr_0p05_0p1) / 0.1780556   # -4.7%  sj2_dr > 0.1683 and n_dr_0p05_0p1 < 7
        + 0.04002253 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +4.0%  sj2_dr > 0.1683
        + 0.03651802 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, Q.eccentricity - 0.9458207) / 0.000763705   # +3.7%  sj2_dr > 0.1779 and eccentricity > 0.9458
        - 0.02857792 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.eccentricity - 0.9458207) / 0.0008867862   # -2.9%  sj2_dr > 0.1683 and eccentricity > 0.9458
        + 0.02757829 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +2.8%  lam1 > 0.008376
        - 0.02260917 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 988.4078 - Q.sum_pt) / 12.22843   # -2.3%  sj2_dr > 0.1683 and sum_pt < 988.4
        - 0.02076759 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # -2.1%  sj2_dr > 0.2688
        + 0.01826058 * max(0.0, Q.max_dr - 0.1117619) / 0.04033009   # +1.8%  max_dr > 0.1118
        + 0.01799602 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.06297984 - Q.tau2) / 0.001263954   # +1.8%  sj2_dr > 0.1683 and tau2 < 0.06298
        - 0.01734249 * max(0.0, Q.width - 0.01323868) / 0.001442684   # -1.7%  width > 0.01324
        - 0.01329816 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001864148 - Q.girth2_top2) / 7.226227e-06   # -1.3%  sj2_dr > 0.1779 and girth2_top2 < 0.001864
        + 0.01177784 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +1.2%  tau1 > 0.1136
        + 0.01156215 * max(0.0, Q.width - 0.01882765) / 0.0007605369   # +1.2%  width > 0.01883
        - 0.01057976 * max(0.0, Q.width - 0.00609665) * max(0.0, Q.log_sum_pt - 6.192222) / 0.0004573947   # -1.1%  width > 0.006097 and log_sum_pt > 6.192
        + 0.009020474 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 0.07848552   # +0.9%  sj2_dr > 0.1592 and n_dr_0p1_0p2 > 1
        - 0.008049324 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -0.8%  girth > 0.08724
        + 0.004646379 * max(0.0, Q.lam2 - 0.000537286) / 0.0003825703   # +0.5%  lam2 > 0.0005373
        - 0.00419409 * max(0.0, Q.width - 0.00609665) * max(0.0, Q.pt_6 - 31.90625) / 0.02357333   # -0.4%  width > 0.006097 and pt_6 > 31.91
        + 0.003815253 * max(0.0, Q.sj2_dr - 0.2687922) * max(0.0, 0.007639643 - Q.girth2_top2) / 1.542054e-05   # +0.4%  sj2_dr > 0.2688 and girth2_top2 < 0.00764
        + 0.003560724 * max(0.0, Q.e2_sq - 0.01165737) / 0.001433269   # +0.4%  e2_sq > 0.01166
        - 0.003398152 * max(0.0, Q.sj2_dr - 0.3003793) * max(0.0, Q.z_dr_0p05_0p1 - 0.08878489) / 0.001005291   # -0.3%  sj2_dr > 0.3004 and z_dr_0p05_0p1 > 0.08878
        - 0.001093875 * max(0.0, Q.width - 0.01323868) * max(0.0, Q.eccentricity - 0.9884745) / 1.823692e-06   # -0.1%  width > 0.01324 and eccentricity > 0.9885
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 51.15;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 51.1521 * (-0.01619874
        + 0.1260789 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # +12.6%  e2_sq < 0.01166
        - 0.1063965 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # -10.6%  girth2 < 0.01324
        - 0.0824729 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -8.2%  girth2 < 0.008678
        + 0.05221245 * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.166579   # +5.2%  sj3_dr_min < 0.209
        + 0.04930358 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.psi_0p2 - 0.9435576) / 0.0004391185   # +4.9%  girth2 < 0.01324 and psi_0p2 > 0.9436
        + 0.04691228 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +4.7%  e3 < 0.0001869
        - 0.03966981 * max(0.0, 0.01165737 - Q.e2_sq) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.0003843383   # -4.0%  e2_sq < 0.01166 and z_dr_0p2_0p4 < 0.05644
        - 0.03839735 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -3.8%  lam2 < 0.001131
        - 0.03133442 * max(0.0, 0.04447357 - Q.e2) / 0.02019334   # -3.1%  e2 < 0.04447
        - 0.03131171 * max(0.0, 0.006679471 - Q.girth2) / 0.00293624   # -3.1%  girth2 < 0.006679
        + 0.02683256 * max(0.0, 0.002635418 - Q.girth2) / 0.0007941346   # +2.7%  girth2 < 0.002635
        + 0.02556745 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +2.6%  N2 < 0.2233
        - 0.02534914 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 6.733425 - Q.log_sum_pt) / 0.02442595   # -2.5%  sj3_dr_max < 0.3012 and log_sum_pt < 6.733
        - 0.0250681 * max(0.0, Q.e2_sq - 0.0030133) / 0.003940214   # -2.5%  e2_sq > 0.003013
        + 0.02252782 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +2.3%  lam1 < 0.012
        + 0.02007133 * max(0.0, 0.07608178 - Q.girth) / 0.02796784   # +2.0%  girth < 0.07608
        - 0.01981115 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 988.4078 - Q.sum_pt) / 13.77568   # -2.0%  N2 < 0.2233 and sum_pt < 988.4
        + 0.01762067 * max(0.0, 0.02415398 - Q.C2_b2) / 0.02132459   # +1.8%  C2_b2 < 0.02415
        + 0.01609156 * max(0.0, Q.e2 - 0.02045966) / 0.01370442   # +1.6%  e2 > 0.02046
        + 0.01504022 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +1.5%  sj3_dr_max > 0.179
        + 0.01474194 * max(0.0, 0.006390125 - Q.e2_sq) / 0.002953508   # +1.5%  e2_sq < 0.00639
        + 0.01289496 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 868.5094 - Q.sum_pt) / 8.122677   # +1.3%  N2 < 0.2233 and sum_pt < 868.5
        - 0.01259605 * max(0.0, 0.04447357 - Q.e2) * max(0.0, Q.psi_0p2 - 0.9435576) / 0.001063378   # -1.3%  e2 < 0.04447 and psi_0p2 > 0.9436
        + 0.01241819 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # +1.2%  C2_b2 < 0.009033
        + 0.0112157 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # +1.1%  girth2_top5 < 0.007164
        - 0.01111132 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.4007947 - Q.planar_flow) / 0.01755893   # -1.1%  N2 < 0.2233 and planar_flow < 0.4008
        - 0.01082081 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, 0.2002199 - Q.dr_max_012) / 0.0001240988   # -1.1%  lam2 < 0.001131 and dr_max_012 < 0.2002
        + 0.01043186 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +1.0%  sj3_dr_max < 0.3012
        - 0.009905921 * max(0.0, Q.sj2_dr - 0.2179769) / 0.01834468   # -1.0%  sj2_dr > 0.218
        - 0.008142267 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # -0.8%  lam1 < 0.002464
        + 0.007433445 * max(0.0, Q.max_dr - 0.0931108) / 0.05062176   # +0.7%  max_dr > 0.09311
        - 0.007413784 * max(0.0, 0.01357153 - Q.tau2) / 0.005299336   # -0.7%  tau2 < 0.01357
        + 0.006231833 * max(0.0, 0.006299534 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.01096064) / 1.639932e-05   # +0.6%  girth2_top2 < 0.0063 and centroid_offset > 0.01096
        - 0.005697166 * max(0.0, 6.638339 - Q.log_sum_pt) / 0.1524936   # -0.6%  log_sum_pt < 6.638
        - 0.00562869 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.20552 - Q.z_dr_0p2_0p4) / 0.02967662   # -0.6%  sj3_dr_max < 0.3012 and z_dr_0p2_0p4 < 0.2055
        + 0.005025427 * max(0.0, Q.sj2_dr - 0.2414351) * max(0.0, 154.25 - Q.pt_2) / 0.9967085   # +0.5%  sj2_dr > 0.2414 and pt_2 < 154.2
        - 0.00466947 * max(0.0, Q.sj3_dr_max - 0.213399) / 0.03061279   # -0.5%  sj3_dr_max > 0.2134
        - 0.004525565 * max(0.0, Q.C2 - 0.06729223) / 0.002854902   # -0.5%  C2 > 0.06729
        + 0.004100782 * max(0.0, 0.7459513 - Q.D2) * max(0.0, Q.pt_6 - 19.46875) / 2.3396   # +0.4%  D2 < 0.746 and pt_6 > 19.47
        - 0.00370286 * max(0.0, Q.sj2_dr - 0.2414351) * max(0.0, 1.129616 - Q.D2_b2) / 0.007762894   # -0.4%  sj2_dr > 0.2414 and D2_b2 < 1.13
        - 0.003262166 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1626038 - Q.abseta_7) / 0.005313035   # -0.3%  N2 < 0.2233 and abseta_7 < 0.1626
        - 0.002705557 * max(0.0, Q.sj2_dr - 0.2179769) * max(0.0, 0.1818564 - Q.z_3rd) / 0.0009573735   # -0.3%  sj2_dr > 0.218 and z_3rd < 0.1819
        + 0.001953726 * max(0.0, Q.sj2_dr - 0.2414351) * max(0.0, 0.1989187 - Q.sj3_z3) / 0.001224215   # +0.2%  sj2_dr > 0.2414 and sj3_z3 < 0.1989
        + 0.001551541 * max(0.0, Q.max_dr - 0.0931108) * max(0.0, 0.04019165 - Q.absphi_0) / 0.0006099482   # +0.2%  max_dr > 0.09311 and absphi_0 < 0.04019
        + 0.001269545 * max(0.0, Q.ptdr0_3 - 10.17512) / 1.13516   # +0.1%  ptdr0_3 > 10.18
        - 0.0006933118 * max(0.0, Q.ptdr0_3 - 17.94219) / 0.2195106   # -0.1%  ptdr0_3 > 17.94
        - 0.0006842201 * max(0.0, Q.sj3_dr_max - 0.3456459) / 0.006655619   # -0.1%  sj3_dr_max > 0.3456
        + 0.0006145055 * max(0.0, Q.sj3_dr_max - 0.3456459) * max(0.0, 1.129616 - Q.D2_b2) / 0.001618684   # +0.1%  sj3_dr_max > 0.3456 and D2_b2 < 1.13
        - 0.0004875311 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 24.42188 - Q.pt_6) / 0.01371831   # -0.0%  N2 < 0.2233 and pt_6 < 24.42
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 13.76;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.75544 * (0.008676879
        + 0.0923532 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +9.2%  LHA < 0.2161
        - 0.09121053 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -9.1%  sj3_dr_max < 0.3012
        + 0.070966 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +7.1%  e3 < 0.0005117
        - 0.05870971 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -5.9%  LHA < 0.2161 and log_sum_pt < 6.804
        - 0.05654098 * max(0.0, 715.4688 - Q.sum_pt) / 69.91196   # -5.7%  sum_pt < 715.5
        + 0.049021 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # +4.9%  max_dr < 0.1773
        + 0.04067061 * max(0.0, 0.002635418 - Q.girth2) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.001577364   # +4.1%  girth2 < 0.002635 and n_dr_0p2_0p4 < 2
        + 0.03981071 * max(0.0, 0.03117077 - Q.centroid_offset) / 0.01649145   # +4.0%  centroid_offset < 0.03117
        - 0.03811836 * max(0.0, 430.75 - Q.sum_pt_top5) / 14.05246   # -3.8%  sum_pt_top5 < 430.8
        + 0.03473619 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, Q.n_for_90pct - 5.0) / 215.169   # +3.5%  sum_pt < 788.4 and n_for_90pct > 5
        + 0.03000433 * max(0.0, 0.002635418 - Q.girth2) * max(0.0, 0.01627885 - Q.centroid_offset) / 7.136253e-06   # +3.0%  girth2 < 0.002635 and centroid_offset < 0.01628
        - 0.02978241 * max(0.0, 0.03117077 - Q.centroid_offset) * max(0.0, 0.09549375 - Q.z_5) / 0.0004977925   # -3.0%  centroid_offset < 0.03117 and z_5 < 0.09549
        + 0.02960661 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +3.0%  girth2 < 0.00502
        - 0.02619223 * max(0.0, 788.4484 - Q.sum_pt) / 112.1576   # -2.6%  sum_pt < 788.4
        - 0.02581834 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.09543973 - Q.z_6) / 0.001514041   # -2.6%  LHA < 0.2161 and z_6 < 0.09544
        - 0.02434649 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # -2.4%  centroid_offset < 0.03776
        + 0.02397429 * max(0.0, 0.0009641429 - Q.girth2) / 0.0002052288   # +2.4%  girth2 < 0.0009641
        - 0.02317754 * max(0.0, 0.04649465 - Q.dr_0) / 0.01415326   # -2.3%  dr_0 < 0.04649
        + 0.02106884 * max(0.0, 0.06081235 - Q.z_6) / 0.009654642   # +2.1%  z_6 < 0.06081
        - 0.02027218 * max(0.0, 0.03117077 - Q.centroid_offset) * max(0.0, Q.pt_5 - 29.875) / 0.3233752   # -2.0%  centroid_offset < 0.03117 and pt_5 > 29.88
        - 0.01781199 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # -1.8%  pt_7 > 34.53
        - 0.01322933 * max(0.0, Q.pt_7 - 43.5) / 1.4368   # -1.3%  pt_7 > 43.5
        + 0.0114153 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 0.04678345 - Q.absphi_0) / 0.000232957   # +1.1%  z_7 < 0.0494 and absphi_0 < 0.04678
        + 0.01136967 * max(0.0, 715.4688 - Q.sum_pt) * max(0.0, 0.0361727 - Q.dr_0) / 0.2640582   # +1.1%  sum_pt < 715.5 and dr_0 < 0.03617
        + 0.01102649 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +1.1%  z_7 < 0.02807
        - 0.01079254 * max(0.0, 0.1767241 - Q.LHA) * max(0.0, 0.006292091 - Q.zdr_0) / 7.498285e-05   # -1.1%  LHA < 0.1767 and zdr_0 < 0.006292
        - 0.009219975 * max(0.0, 0.03459477 - Q.tau1) / 0.008474553   # -0.9%  tau1 < 0.03459
        - 0.00903 * max(0.0, 715.4688 - Q.sum_pt) * max(0.0, 1.50326e-07 - Q.e4) / 8.236047e-06   # -0.9%  sum_pt < 715.5 and e4 < 1.503e-07
        + 0.008092919 * max(0.0, 0.03556091 - Q.e2) * max(0.0, Q.zdr_0 - 0.004918231) / 3.97457e-05   # +0.8%  e2 < 0.03556 and zdr_0 > 0.004918
        - 0.007586262 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, Q.pt1_dr01 - 12.6865) / 0.1599605   # -0.8%  sj3_dr_max < 0.3012 and pt1_dr01 > 12.69
        - 0.007000718 * max(0.0, 0.1767241 - Q.LHA) / 0.01861764   # -0.7%  LHA < 0.1767
        - 0.006988773 * max(0.0, 20.125 - Q.pt_7) / 0.5468342   # -0.7%  pt_7 < 20.12
        - 0.006467726 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.6%  log_sum_pt > 6.896
        + 0.006162713 * max(0.0, 0.03556091 - Q.e2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.001328529   # +0.6%  e2 < 0.03556 and planar_flow < 0.3221
        + 0.005680782 * max(0.0, 0.03243272 - Q.z_7) / 0.002061024   # +0.6%  z_7 < 0.03243
        + 0.005103333 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # +0.5%  girth < 0.007674
        + 0.004282584 * max(0.0, Q.pt_7 - 43.5) * max(0.0, 0.06970457 - Q.dr_6) / 0.04319889   # +0.4%  pt_7 > 43.5 and dr_6 < 0.0697
        + 0.004096337 * max(0.0, 715.4688 - Q.sum_pt) * max(0.0, 0.007798268 - Q.zdr_0) / 0.05114364   # +0.4%  sum_pt < 715.5 and zdr_0 < 0.007798
        - 0.003481164 * max(0.0, 0.03747769 - Q.z_4) / 0.000472225   # -0.3%  z_4 < 0.03748
        + 0.003292225 * max(0.0, 715.4688 - Q.sum_pt) * max(0.0, 0.003299436 - Q.zdr_3) / 0.01707731   # +0.3%  sum_pt < 715.5 and zdr_3 < 0.003299
        + 0.003211748 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # +0.3%  z_6 < 0.0216
        - 0.003180904 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, Q.ptdr0_2 - 11.87288) / 0.08620714   # -0.3%  sj3_dr_max < 0.3012 and ptdr0_2 > 11.87
        + 0.0030335 * max(0.0, 31.125 - Q.pt_4) / 0.3140792   # +0.3%  pt_4 < 31.12
        + 0.002062462 * max(0.0, 24.57812 - Q.pt_5) * max(0.0, 0.1583252 - Q.abseta_6) / 0.03135727   # +0.2%  pt_5 < 24.58 and abseta_6 < 0.1583
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 33.38;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.37615 * (0.0489515
        - 0.1912726 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -19.1%  girth2 < 0.008678
        - 0.06017323 * max(0.0, Q.centroid_offset - 0.01627885) * max(0.0, 988.4078 - Q.sum_pt) / 2.602791   # -6.0%  centroid_offset > 0.01628 and sum_pt < 988.4
        + 0.05950438 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # +6.0%  tau1 < 0.1028
        + 0.04912794 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +4.9%  lam1 < 0.012
        + 0.04861889 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # +4.9%  e2_sq < 0.008169
        - 0.04312661 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -4.3%  lam1 < 0.00733
        - 0.03950093 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # -4.0%  width < 0.01324
        + 0.03396983 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +3.4%  sj3_dr_max > 0.179
        - 0.03328381 * max(0.0, Q.sj3_dr_max - 0.07264452) / 0.1084421   # -3.3%  sj3_dr_max > 0.07264
        - 0.0329451 * max(0.0, 0.1598486 - Q.max_dr) / 0.05771655   # -3.3%  max_dr < 0.1598
        - 0.03253392 * max(0.0, Q.sj3_dr_max - 0.07264452) * max(0.0, Q.eccentricity - 0.9031255) / 0.006127495   # -3.3%  sj3_dr_max > 0.07264 and eccentricity > 0.9031
        + 0.02678766 * max(0.0, 0.02689598 - Q.girth) / 0.004330137   # +2.7%  girth < 0.0269
        + 0.02542349 * max(0.0, 0.02188405 - Q.lam1) / 0.01625498   # +2.5%  lam1 < 0.02188
        + 0.02438599 * max(0.0, Q.centroid_offset - 0.01627885) / 0.006307541   # +2.4%  centroid_offset > 0.01628
        + 0.02139727 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +2.1%  centroid_offset > 0.008092
        + 0.02036929 * max(0.0, 0.0095303 - Q.girth2_top2) / 0.006109573   # +2.0%  girth2_top2 < 0.00953
        + 0.01871176 * max(0.0, 0.03853752 - Q.C3) / 0.01496309   # +1.9%  C3 < 0.03854
        - 0.01816731 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -1.8%  lam2 < 0.001131
        + 0.01784791 * max(0.0, Q.sj3_dr_max - 0.1789613) * max(0.0, Q.eccentricity - 0.9031255) / 0.002046829   # +1.8%  sj3_dr_max > 0.179 and eccentricity > 0.9031
        + 0.01747291 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 715.4688 - Q.sum_pt) / 1.381606   # +1.7%  centroid_offset > 0.008092 and sum_pt < 715.5
        - 0.01391302 * max(0.0, 1.651088 - Q.D3) / 0.5901079   # -1.4%  D3 < 1.651
        + 0.011918 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +1.2%  lam1 < 0.008376
        + 0.01184717 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.01426135 - Q.mean_phi2) / 9.496308e-05   # +1.2%  centroid_offset > 0.008092 and mean_phi2 < 0.01426
        + 0.01145834 * max(0.0, Q.LHA - 0.266913) * max(0.0, Q.eccentricity - 0.9031255) / 0.001613144   # +1.1%  LHA > 0.2669 and eccentricity > 0.9031
        - 0.0111078 * max(0.0, Q.sj3_dr_min - 0.02022982) / 0.02964063   # -1.1%  sj3_dr_min > 0.02023
        - 0.009699194 * max(0.0, Q.eccentricity - 0.9031255) / 0.04739094   # -1.0%  eccentricity > 0.9031
        + 0.009566262 * max(0.0, 6.327379 - Q.log_sum_pt) / 0.0331691   # +1.0%  log_sum_pt < 6.327
        - 0.009373402 * max(0.0, 0.01323868 - Q.width) * max(0.0, 0.3220738 - Q.planar_flow) / 0.001091057   # -0.9%  width < 0.01324 and planar_flow < 0.3221
        + 0.009217211 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +0.9%  tau1 < 0.1136
        - 0.009155419 * max(0.0, Q.e2_sq - 0.01165737) / 0.001433269   # -0.9%  e2_sq > 0.01166
        + 0.008566166 * max(0.0, 3.0 - Q.n_dr_0_0p05) / 1.029338   # +0.9%  n_dr_0_0p05 < 3
        + 0.008287148 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 6.0 - Q.n_dr_0p05_0p1) / 0.03827938   # +0.8%  centroid_offset > 0.008092 and n_dr_0p05_0p1 < 6
        + 0.007026498 * max(0.0, 29.90625 - Q.pt_6) / 1.385454   # +0.7%  pt_6 < 29.91
        + 0.006675105 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.1682655 - Q.sj2_dr) / 0.0003126334   # +0.7%  centroid_offset > 0.008092 and sj2_dr < 0.1683
        + 0.005730778 * max(0.0, 6.638339 - Q.log_sum_pt) / 0.1524936   # +0.6%  log_sum_pt < 6.638
        + 0.004751198 * max(0.0, 1.332146 - Q.D2) / 0.3438169   # +0.5%  D2 < 1.332
        + 0.004374703 * max(0.0, 6.638339 - Q.log_sum_pt) * max(0.0, 0.0586137 - Q.z_7) / 0.0003525255   # +0.4%  log_sum_pt < 6.638 and z_7 < 0.05861
        - 0.004275668 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.4%  z_6 < 0.0216
        - 0.004157348 * max(0.0, 0.03448406 - Q.z_6) / 0.001525592   # -0.4%  z_6 < 0.03448
        - 0.004094027 * max(0.0, Q.sj3_dr_max - 0.1789613) * max(0.0, 6.0 - Q.n_dr_0p05_0p1) / 0.1786734   # -0.4%  sj3_dr_max > 0.179 and n_dr_0p05_0p1 < 6
        - 0.004044081 * max(0.0, 6.327379 - Q.log_sum_pt) * max(0.0, Q.pt_7 - 25.57812) / 0.2533514   # -0.4%  log_sum_pt < 6.327 and pt_7 > 25.58
        - 0.002903725 * max(0.0, Q.sj3_dr_max - 0.07264452) * max(0.0, 0.09310137 - Q.M3) / 0.003474724   # -0.3%  sj3_dr_max > 0.07264 and M3 < 0.0931
        + 0.002817017 * max(0.0, Q.e2_sq - 0.01716248) / 0.0007604864   # +0.3%  e2_sq > 0.01716
        + 0.002301844 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 19.46875 - Q.pt_6) / 0.002700092   # +0.2%  lam1 < 0.012 and pt_6 < 19.47
        + 0.002269323 * max(0.0, Q.centroid_offset - 0.02076709) / 0.004751537   # +0.2%  centroid_offset > 0.02077
        - 0.001669556 * max(0.0, Q.tau2 - 0.01713288) / 0.005826319   # -0.2%  tau2 > 0.01713
        + 0.00161653 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.2%  sum_pt > 988.4
        + 0.0007398495 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.dr01 - 0.1410336) / 0.01228651   # +0.1%  sum_pt > 988.4 and dr01 > 0.141
        - 0.0006461958 * max(0.0, Q.eccentricity - 0.9031255) * max(0.0, Q.mean_eta - 0.01772426) / 5.963698e-05   # -0.1%  eccentricity > 0.9031 and mean_eta > 0.01772
        + 0.0006196776 * max(0.0, 0.02160287 - Q.z_6) * max(0.0, 6.842717 - Q.log_sum_pt) / 4.457061e-06   # +0.1%  z_6 < 0.0216 and log_sum_pt < 6.843
        - 0.000556955 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.dr01 - 0.1662967) / 0.008508326   # -0.1%  sum_pt > 988.4 and dr01 > 0.1663
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 40.62;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 40.61841 * (0.2063297
        - 0.1624302 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -16.2%  e2_sq < 0.008169
        - 0.08571486 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -8.6%  girth2 > 0.00752
        + 0.05934405 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +5.9%  tau1 < 0.09539
        + 0.05450939 * max(0.0, 0.00718279 - Q.e2_sq) / 0.003528787   # +5.5%  e2_sq < 0.007183
        - 0.05090655 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -5.1%  girth < 0.08724
        - 0.04965496 * max(0.0, 0.005590289 - Q.width) / 0.002229311   # -5.0%  width < 0.00559
        - 0.04624459 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -4.6%  tau1 < 0.1136
        - 0.044369 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # -4.4%  e2_sq < 0.01166
        - 0.04396251 * max(0.0, Q.girth2 - 0.004372139) / 0.003638805   # -4.4%  girth2 > 0.004372
        + 0.03831994 * max(0.0, 0.1416226 - Q.tau1) / 0.08190733   # +3.8%  tau1 < 0.1416
        + 0.03215988 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +3.2%  sj2_dr < 0.1592
        - 0.03101445 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -3.1%  sj2_dr < 0.1873
        + 0.02773228 * max(0.0, 0.005284669 - Q.e2_sq) / 0.00223735   # +2.8%  e2_sq < 0.005285
        + 0.02677526 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # +2.7%  girth2 > 0.008678
        - 0.02453408 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # -2.5%  LHA < 0.2161
        + 0.01992759 * max(0.0, 0.1979689 - Q.max_dr) / 0.08649773   # +2.0%  max_dr < 0.198
        + 0.01754919 * max(0.0, 0.2931906 - Q.LHA) / 0.07167606   # +1.8%  LHA < 0.2932
        + 0.01369815 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +1.4%  sj3_dr_max < 0.1986
        - 0.01302758 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 4.721224 - Q.D2_b2) / 0.02812613   # -1.3%  centroid_offset < 0.02077 and D2_b2 < 4.721
        + 0.01277852 * max(0.0, Q.girth2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9458207) / 2.125219e-05   # +1.3%  girth2 > 0.01324 and eccentricity > 0.9458
        - 0.01274863 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -1.3%  lam2 < 0.0005373
        + 0.01186927 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.423044) / 0.01391616   # +1.2%  planar_flow < 0.195 and log_sum_pt > 6.423
        - 0.01079357 * max(0.0, Q.girth2 - 0.008678045) * max(0.0, Q.eccentricity - 0.9458207) / 3.56764e-05   # -1.1%  girth2 > 0.008678 and eccentricity > 0.9458
        + 0.01019239 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # +1.0%  centroid_offset < 0.02077
        + 0.009183186 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +0.9%  lam2 < 0.001131
        + 0.009118443 * max(0.0, Q.girth2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +0.9%  girth2 > 0.004372 and planar_flow < 0.195
        + 0.008902601 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.width - 0.007520088) / 0.0001455029   # +0.9%  planar_flow < 0.195 and width > 0.00752
        - 0.008200766 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # -0.8%  sj3_dr_max < 0.2134
        + 0.007147092 * max(0.0, 0.0479157 - Q.girth) / 0.01216112   # +0.7%  girth < 0.04792
        + 0.006228766 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +0.6%  sj3_dr_max < 0.1426
        + 0.006003288 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +0.6%  tau1 < 0.05357
        + 0.005528872 * max(0.0, 0.0005611231 - Q.girth2) / 9.465572e-05   # +0.6%  girth2 < 0.0005611
        - 0.004942004 * max(0.0, 0.121681 - Q.max_dr) / 0.03360572   # -0.5%  max_dr < 0.1217
        + 0.004709539 * max(0.0, 0.001056655 - Q.girth2_top2) / 0.0003002514   # +0.5%  girth2_top2 < 0.001057
        + 0.004145637 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # +0.4%  lam1 < 0.004184
        - 0.004127143 * max(0.0, 0.01165737 - Q.e2_sq) * max(0.0, 48.71875 - Q.pt_7) / 0.1103404   # -0.4%  e2_sq < 0.01166 and pt_7 < 48.72
        - 0.003328163 * max(0.0, 0.213399 - Q.sj3_dr_max) * max(0.0, Q.eccentricity - 0.9458207) / 0.0008147252   # -0.3%  sj3_dr_max < 0.2134 and eccentricity > 0.9458
        - 0.003313709 * max(0.0, 0.0009641429 - Q.girth2) / 0.0002052288   # -0.3%  girth2 < 0.0009641
        + 0.003132863 * max(0.0, 0.001101266 - Q.e2_sq) / 0.000308004   # +0.3%  e2_sq < 0.001101
        - 0.002412184 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 0.05240025 - Q.z_7) / 0.000631283   # -0.2%  planar_flow < 0.195 and z_7 < 0.0524
        + 0.001867459 * max(0.0, 0.005590289 - Q.width) * max(0.0, -0.00469376 - Q.mean_phi) / 4.24156e-06   # +0.2%  width < 0.00559 and mean_phi < -0.004694
        + 0.001631918 * max(0.0, 0.005590289 - Q.width) * max(0.0, -0.006779839 - Q.mean_eta) / 3.381615e-06   # +0.2%  width < 0.00559 and mean_eta < -0.00678
        + 0.001296048 * max(0.0, 0.005590289 - Q.width) * max(0.0, Q.mean_phi - 0.009050008) / 2.537131e-06   # +0.1%  width < 0.00559 and mean_phi > 0.00905
        + 0.001205837 * max(0.0, 0.005590289 - Q.width) * max(0.0, Q.mean_eta - 0.009367547) / 2.50486e-06   # +0.1%  width < 0.00559 and mean_eta > 0.009368
        - 0.001137455 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 33.6875 - Q.pt_6) / 0.1349842   # -0.1%  planar_flow < 0.195 and pt_6 < 33.69
        + 0.00107866 * max(0.0, 0.00718279 - Q.e2_sq) * max(0.0, Q.sj3_dr13 - 0.1982159) / 1.577335e-05   # +0.1%  e2_sq < 0.007183 and sj3_dr13 > 0.1982
        - 0.0007096011 * max(0.0, 0.1979689 - Q.max_dr) * max(0.0, Q.n_dr_0p1_0p2 - 4.0) / 0.004130109   # -0.1%  max_dr < 0.198 and n_dr_0p1_0p2 > 4
        - 0.0002090363 * max(0.0, Q.girth2 - 0.004372139) * max(0.0, Q.sj3_z1 - 0.8740683) / 7.896779e-07   # -0.0%  girth2 > 0.004372 and sj3_z1 > 0.8741
        + 0.0001828379 * max(0.0, Q.girth2 - 0.008678045) * max(0.0, Q.sj3_z1 - 0.8740683) / 2.104095e-07   # +0.0%  girth2 > 0.008678 and sj3_z1 > 0.8741
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 14.69;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.69319 * (-0.0372254
        + 0.06886455 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.002635418 - Q.girth2) / 3.338443e-05   # +6.9%  tau1 < 0.05357 and girth2 < 0.002635
        - 0.06534121 * max(0.0, 0.05464922 - Q.girth) * max(0.0, 0.005590289 - Q.width) / 7.638845e-05   # -6.5%  girth < 0.05465 and width < 0.00559
        + 0.06257579 * max(0.0, 0.006679471 - Q.girth2) / 0.00293624   # +6.3%  girth2 < 0.006679
        - 0.05579543 * max(0.0, 0.05464922 - Q.girth) / 0.01532978   # -5.6%  girth < 0.05465
        - 0.05542353 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, 0.20552 - Q.z_dr_0p2_0p4) / 0.005135396   # -5.5%  LHA < 0.1967 and z_dr_0p2_0p4 < 0.2055
        + 0.04964476 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.0003061234 - Q.lam2) / 1.676724e-05   # +5.0%  sj3_dr_max < 0.1986 and lam2 < 0.0003061
        - 0.04939379 * max(0.0, 0.05464922 - Q.girth) * max(0.0, 0.002635418 - Q.width) / 3.204761e-05   # -4.9%  girth < 0.05465 and width < 0.002635
        - 0.0443609 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -4.4%  sj3_dr_max < 0.1426
        + 0.0413153 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +4.1%  girth2 < 0.00502
        + 0.03574325 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +3.6%  girth2 < 0.006679 and centroid_offset < 0.02355
        + 0.03440197 * max(0.0, 0.05464922 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2) / 2.397672e-06   # +3.4%  girth < 0.05465 and lam2 < 0.0001948
        - 0.03369033 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, 0.0003061234 - Q.lam2) / 6.518054e-06   # -3.4%  sj3_dr_max < 0.107 and lam2 < 0.0003061
        + 0.03314173 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.0001053955   # +3.3%  girth2 < 0.00502 and z_dr_0p2_0p4 < 0.05644
        - 0.02878585 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -2.9%  girth2 < 0.00502 and centroid_offset > 0.00679
        + 0.02715895 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +2.7%  sj3_dr_max < 0.107
        + 0.02667219 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # +2.7%  LHA < 0.1967
        - 0.0234542 * max(0.0, 0.01713288 - Q.tau2) / 0.008065871   # -2.3%  tau2 < 0.01713
        - 0.02035548 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, 0.0001947983 - Q.lam2) / 4.231311e-06   # -2.0%  LHA < 0.1967 and lam2 < 0.0001948
        + 0.01816364 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.8%  tau1 < 0.05357
        - 0.0168791 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 43.5 - Q.pt_7) / 0.02255472   # -1.7%  girth2 < 0.00502 and pt_7 < 43.5
        + 0.01514143 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +1.5%  sum_pt_top5 > 687.4
        - 0.01435441 * max(0.0, Q.z_dr_0_0p05 - 0.9008535) * max(0.0, 0.0001947983 - Q.lam2) / 5.136077e-06   # -1.4%  z_dr_0_0p05 > 0.9009 and lam2 < 0.0001948
        - 0.01380156 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.0003061234 - Q.lam2) / 4.443043e-06   # -1.4%  tau1 < 0.05357 and lam2 < 0.0003061
        + 0.01353879 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.002151568 - Q.girth2_top3) / 5.290325e-05   # +1.4%  log_sum_pt > 6.701 and girth2_top3 < 0.002152
        + 0.01299191 * max(0.0, Q.n_dr_0_0p05 - 5.0) / 1.093146   # +1.3%  n_dr_0_0p05 > 5
        + 0.01227296 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 43.5 - Q.pt_7) / 0.2124887   # +1.2%  tau1 < 0.05357 and pt_7 < 43.5
        - 0.0120212 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.006679471 - Q.width) / 0.0001692711   # -1.2%  log_sum_pt > 6.701 and width < 0.006679
        + 0.01056996 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.4007947 - Q.planar_flow) / 0.0003918103   # +1.1%  girth2 < 0.006679 and planar_flow < 0.4008
        + 0.01032548 * max(0.0, 0.0003193707 - Q.girth2) / 4.035777e-05   # +1.0%  girth2 < 0.0003194
        + 0.009366049 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.002635418 - Q.width) / 5.489096e-05   # +0.9%  log_sum_pt > 6.701 and width < 0.002635
        + 0.009232296 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # +0.9%  lam1 < 0.001504
        - 0.007995731 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.006445344 - Q.mean_phi2) / 0.0001920358   # -0.8%  log_sum_pt > 6.701 and mean_phi2 < 0.006445
        - 0.007977862 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -0.8%  log_sum_pt > 6.701
        - 0.007745135 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.width - 0.001653836) / 1.467156e-06   # -0.8%  girth2 < 0.006679 and width > 0.001654
        + 0.007670556 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.psi_0p2 - 0.9435576) / 0.001795607   # +0.8%  log_sum_pt > 6.701 and psi_0p2 > 0.9436
        - 0.007630658 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.01002369 - Q.girth2_top3) / 0.3268181   # -0.8%  sum_pt_top5 > 687.4 and girth2_top3 < 0.01002
        + 0.006663982 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.pt_7 - 15.55391) / 0.4153091   # +0.7%  LHA < 0.1967 and pt_7 > 15.55
        - 0.006452374 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.pt_7 - 15.55391) / 0.4785904   # -0.6%  log_sum_pt > 6.701 and pt_7 > 15.55
        - 0.005896474 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.0009641429 - Q.girth2) / 0.01750133   # -0.6%  sum_pt_top5 > 687.4 and girth2 < 0.0009641
        - 0.005469463 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 6.502799 - Q.log_sum_pt) / 8.085113e-05   # -0.5%  girth2 < 0.00502 and log_sum_pt < 6.503
        - 0.004814652 * max(0.0, 0.01357518 - Q.tau1) / 0.001531397   # -0.5%  tau1 < 0.01358
        - 0.003612443 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.centroid_offset - 0.01627885) / 5.281971e-06   # -0.4%  LHA < 0.1967 and centroid_offset > 0.01628
        + 0.003292676 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, Q.girth2_top3 - 0.001155057) / 3.499346e-05   # +0.3%  sj3_dr_max < 0.1986 and girth2_top3 > 0.001155
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 27.95;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.95398 * (-0.1025176
        + 0.1120318 * max(0.0, 0.006679471 - Q.width) / 0.00293624   # +11.2%  width < 0.006679
        + 0.1028812 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +10.3%  girth2 < 0.00502
        + 0.0829383 * max(0.0, 6.896095 - Q.log_sum_pt) / 0.3571984   # +8.3%  log_sum_pt < 6.896
        + 0.06049983 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # +6.0%  girth2 < 0.003563
        - 0.05702325 * max(0.0, 0.02378091 - Q.e2_sq) / 0.01817124   # -5.7%  e2_sq < 0.02378
        + 0.04578184 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +4.6%  sj3_dr_max < 0.1986
        - 0.04355684 * max(0.0, 0.004641801 - Q.e2_sq) / 0.001869767   # -4.4%  e2_sq < 0.004642
        - 0.03931162 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -3.9%  sj3_dr_max < 0.1426
        - 0.03469886 * max(0.0, 0.0030133 - Q.e2_sq) / 0.001065944   # -3.5%  e2_sq < 0.003013
        - 0.03204599 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -3.2%  sj2_dr < 0.1295
        + 0.0311154 * max(0.0, 0.02355416 - Q.centroid_offset) / 0.0103615   # +3.1%  centroid_offset < 0.02355
        - 0.02537489 * max(0.0, 0.05464922 - Q.girth) / 0.01532978   # -2.5%  girth < 0.05465
        + 0.02226516 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.01364517 - Q.tau3) / 6.201347e-05   # +2.2%  centroid_offset < 0.01838 and tau3 < 0.01365
        - 0.02052139 * max(0.0, 0.06663269 - Q.girth) / 0.02186049   # -2.1%  girth < 0.06663
        + 0.0176417 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +1.8%  lam2 < 0.0001948
        - 0.0174809 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -1.7%  e3 > 3.892e-05
        + 0.01706627 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +1.7%  max_dr < 0.1118
        + 0.01634647 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +1.6%  e3 > 0.0001869
        - 0.01592158 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # -1.6%  dr_0 < 0.06413
        - 0.01384455 * max(0.0, 0.00483998 - Q.lam1) * max(0.0, 0.1792439 - Q.dr1_7) / 0.0002681756   # -1.4%  lam1 < 0.00484 and dr1_7 < 0.1792
        + 0.0122262 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 0.0234802   # +1.2%  centroid_offset < 0.01838 and n_dr_0p05_0p1 < 5
        - 0.01108234 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, 0.9206502 - Q.D2_b2) / 0.1512968   # -1.1%  log_sum_pt < 6.896 and D2_b2 < 0.9207
        + 0.01041892 * max(0.0, 0.1778793 - Q.sj2_dr) / 0.05866968   # +1.0%  sj2_dr < 0.1779
        + 0.0099836 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +1.0%  lam2 > 0.001131
        - 0.009963926 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # -1.0%  lam1 < 0.00484
        + 0.009820055 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # +1.0%  tau1 < 0.0437
        - 0.009525565 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # -1.0%  LHA < 0.2161
        + 0.0091166 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, 0.0825754 - Q.dr_7) / 0.0003946939   # +0.9%  centroid_offset < 0.02355 and dr_7 < 0.08258
        - 0.00824606 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.pt1_over_pt0 - 0.2131291) / 0.002760106   # -0.8%  centroid_offset < 0.01838 and pt1_over_pt0 > 0.2131
        - 0.007833097 * max(0.0, 0.001101266 - Q.e2_sq) / 0.000308004   # -0.8%  e2_sq < 0.001101
        + 0.007500478 * max(0.0, 7.300726e-05 - Q.lam2) / 2.767163e-05   # +0.8%  lam2 < 7.301e-05
        + 0.007403253 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.pt_1 - 93.25) / 0.3775907   # +0.7%  centroid_offset < 0.01838 and pt_1 > 93.25
        - 0.00685525 * max(0.0, 0.006679471 - Q.width) * max(0.0, 0.8272948 - Q.D3) / 0.0002524894   # -0.7%  width < 0.006679 and D3 < 0.8273
        - 0.006235413 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, Q.planar_flow - 0.1115136) / 0.07272605   # -0.6%  log_sum_pt < 6.896 and planar_flow > 0.1115
        + 0.005395961 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, Q.tau21_b2 - 0.004811143) / 0.002193894   # +0.5%  centroid_offset < 0.02355 and tau21_b2 > 0.004811
        + 0.005179457 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, Q.N3 - 0.7686247) / 0.007184418   # +0.5%  centroid_offset < 0.02355 and N3 > 0.7686
        - 0.005045455 * max(0.0, 0.06663269 - Q.girth) * max(0.0, 0.00161889 - Q.mean_phi) / 8.763302e-05   # -0.5%  girth < 0.06663 and mean_phi < 0.001619
        - 0.004984543 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.04435703 - Q.M2) / 5.61399e-05   # -0.5%  centroid_offset < 0.01838 and M2 < 0.04436
        - 0.004281894 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, 0.380911 - Q.D2_b2) / 0.001264948   # -0.4%  centroid_offset < 0.02355 and D2_b2 < 0.3809
        - 0.004251965 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, Q.planar_flow - 0.2534037) / 0.0001586905   # -0.4%  lam2 > 0.001131 and planar_flow > 0.2534
        - 0.003903586 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -0.4%  sj3_dr_max > 0.2623
        + 0.003765504 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, 0.05753583 - Q.z_5) / 0.0005902159   # +0.4%  log_sum_pt < 6.896 and z_5 < 0.05754
        + 0.003492089 * max(0.0, 0.003562611 - Q.girth2) * max(0.0, 0.8272948 - Q.D3) / 7.149804e-05   # +0.3%  girth2 < 0.003563 and D3 < 0.8273
        - 0.003106954 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.07735553 - Q.mratio_min_012) / 0.0001043432   # -0.3%  centroid_offset < 0.01838 and mratio_min_012 < 0.07736
        - 0.002971304 * max(0.0, 0.006679471 - Q.width) * max(0.0, -0.006779839 - Q.mean_eta) / 4.770336e-06   # -0.3%  width < 0.006679 and mean_eta < -0.00678
        - 0.002925315 * max(0.0, 0.06663269 - Q.girth) * max(0.0, Q.mean_phi - 0.004406178) / 3.557874e-05   # -0.3%  girth < 0.06663 and mean_phi > 0.004406
        - 0.002862878 * max(0.0, 0.001101266 - Q.e2_sq) * max(0.0, Q.eccentricity - 0.7117266) / 3.682686e-05   # -0.3%  e2_sq < 0.001101 and eccentricity > 0.7117
        - 0.002668416 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.003218391 - Q.mean_eta) / 0.0001345871   # -0.3%  LHA < 0.2161 and mean_eta < 0.003218
        - 0.002232026 * max(0.0, 0.006679471 - Q.width) * max(0.0, Q.mean_eta - 0.009367547) / 3.542905e-06   # -0.2%  width < 0.006679 and mean_eta > 0.009368
        - 0.002128374 * max(0.0, 0.003562611 - Q.girth2) * max(0.0, Q.mean_eta - 0.01271871) / 6.620797e-07   # -0.2%  girth2 < 0.003563 and mean_eta > 0.01272
        + 0.001932494 * max(0.0, 0.1294903 - Q.sj2_dr) * max(0.0, -0.006779839 - Q.mean_eta) / 5.903935e-05   # +0.2%  sj2_dr < 0.1295 and mean_eta < -0.00678
        + 0.001159193 * max(0.0, 0.1294903 - Q.sj2_dr) * max(0.0, Q.mean_eta - 0.01271871) / 3.354705e-05   # +0.1%  sj2_dr < 0.1295 and mean_eta > 0.01272
        - 0.001136667 * max(0.0, Q.ptdr0_2 - 21.99783) / 0.2665586   # -0.1%  ptdr0_2 > 22
        + 0.001112628 * max(0.0, 0.06663269 - Q.girth) * max(0.0, Q.mean_eta - 0.01271871) / 1.223782e-05   # +0.1%  girth < 0.06663 and mean_eta > 0.01272
        - 0.0009047848 * max(0.0, 0.003562611 - Q.girth2) * max(0.0, -0.01284493 - Q.mean_eta) / 6.468207e-07   # -0.1%  girth2 < 0.003563 and mean_eta < -0.01284
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 14.55;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.55059 * (0.1883063
        + 0.2144879 * Q.lam1 / 0.00591636   # +21.4%  lam1
        - 0.09788269 * max(0.0, 0.1245537 - Q.girth) / 0.06848162   # -9.8%  girth < 0.1246
        - 0.09736298 * max(0.0, Q.girth2 - 0.002635418) / 0.004603951   # -9.7%  girth2 > 0.002635
        - 0.08595961 * max(0.0, Q.LHA - 0.1967397) / 0.07109091   # -8.6%  LHA > 0.1967
        - 0.05093455 * max(0.0, Q.lam1 - 0.00483998) / 0.002931468   # -5.1%  lam1 > 0.00484
        - 0.042379 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -4.2%  lam1 < 0.006507
        + 0.03990765 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # +4.0%  girth2 > 0.00752
        + 0.0360536 * max(0.0, Q.tau1 - 0.04369778) / 0.031989   # +3.6%  tau1 > 0.0437
        + 0.03590234 * max(0.0, Q.lam2 - 0.0001947983) / 0.0004461515   # +3.6%  lam2 > 0.0001948
        + 0.03123411 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +3.1%  sj3_dr_max < 0.3012
        + 0.03121346 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # +3.1%  sj2_dr > 0.1295
        - 0.02464419 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.6947818 - Q.planar_flow) / 0.0166918   # -2.5%  sj2_dr > 0.1683 and planar_flow < 0.6948
        + 0.0226465 * max(0.0, 0.002915531 - Q.girth2_top3) / 0.001178811   # +2.3%  girth2_top3 < 0.002916
        - 0.02080489 * max(0.0, 0.06810151 - Q.z_7) / 0.01877013   # -2.1%  z_7 < 0.0681
        - 0.01814047 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # -1.8%  lam1 < 0.003377
        - 0.01690488 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.1647949 - Q.absphi_7) / 0.0004021457   # -1.7%  lam1 < 0.006507 and absphi_7 < 0.1648
        + 0.01652817 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.158545   # +1.7%  n_dr_0p05_0p1 < 5
        - 0.01439673 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # -1.4%  sj2_dr > 0.1683
        - 0.0124406 * max(0.0, Q.LHA - 0.3127275) / 0.01533444   # -1.2%  LHA > 0.3127
        + 0.009811882 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # +1.0%  e3 > 8.148e-05
        - 0.009137462 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -0.9%  log_sum_pt > 6.67
        + 0.008878716 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.9%  n_dr_0p2_0p4 > 1
        - 0.008625553 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.planar_flow - 0.06126205) / 0.0002819968   # -0.9%  lam2 > 0.0001948 and planar_flow > 0.06126
        - 0.007835398 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 4.721224 - Q.D2_b2) / 0.5862342   # -0.8%  n_dr_0p2_0p4 > 1 and D2_b2 < 4.721
        - 0.007461067 * max(0.0, Q.LHA - 0.3127275) * max(0.0, 0.6947818 - Q.planar_flow) / 0.005887138   # -0.7%  LHA > 0.3127 and planar_flow < 0.6948
        + 0.006246124 * max(0.0, 0.001501708 - Q.girth2_top5) / 0.0004379639   # +0.6%  girth2_top5 < 0.001502
        - 0.00480778 * max(0.0, 48.03125 - Q.pt_6) * max(0.0, 6.46415 - Q.log_sum_pt) / 0.7570358   # -0.5%  pt_6 < 48.03 and log_sum_pt < 6.464
        - 0.004791744 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.eccentricity - 0.4829602) / 8.513077e-05   # -0.5%  lam2 > 0.0001948 and eccentricity > 0.483
        - 0.004297666 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, 6.538559 - Q.log_sum_pt) / 0.0001231869   # -0.4%  lam2 > 0.0001948 and log_sum_pt < 6.539
        + 0.004169439 * max(0.0, Q.centroid_offset - 0.02355416) / 0.003991673   # +0.4%  centroid_offset > 0.02355
        + 0.00387115 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +0.4%  sj2_dr > 0.3004
        + 0.003421677 * max(0.0, 48.03125 - Q.pt_6) * max(0.0, 0.1236856 - Q.D2_b2) / 0.1641634   # +0.3%  pt_6 < 48.03 and D2_b2 < 0.1237
        + 0.002496268 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.2%  sum_pt > 988.4
        - 0.002346584 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -0.2%  lam2 > 0.003408
        - 0.001977155 * max(0.0, Q.girth2 - 0.02530566) / 0.0002851418   # -0.2%  girth2 > 0.02531
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 39.52;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 39.5212 * (-0.05228874
        + 0.1489296 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +14.9%  width < 0.008678
        + 0.0951116 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +9.5%  width < 0.01324
        - 0.08182399 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # -8.2%  LHA < 0.3127
        + 0.06965198 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +7.0%  LHA < 0.3033
        - 0.06517451 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -6.5%  girth < 0.0717
        - 0.05706727 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -5.7%  lam1 < 0.008376
        - 0.05074598 * max(0.0, 0.006390125 - Q.e2_sq) / 0.002953508   # -5.1%  e2_sq < 0.00639
        + 0.04158737 * max(0.0, 0.006679471 - Q.girth2) / 0.00293624   # +4.2%  girth2 < 0.006679
        - 0.03873309 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -3.9%  e2_sq < 0.008169
        + 0.03566145 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +3.6%  centroid_offset < 0.03776
        - 0.02343846 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -2.3%  sj2_dr > 0.1779
        + 0.02157164 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +2.2%  z_7 > 0.01686
        - 0.01526061 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -1.5%  lam1 < 0.00733
        + 0.01425993 * max(0.0, Q.sj2_dr - 0.09419022) / 0.07816795   # +1.4%  sj2_dr > 0.09419
        - 0.01315927 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 118.5 - Q.pt_2) / 0.5820432   # -1.3%  centroid_offset < 0.03776 and pt_2 < 118.5
        + 0.01308941 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +1.3%  planar_flow < 0.2534
        - 0.01271798 * max(0.0, 0.03459477 - Q.tau1) / 0.008474553   # -1.3%  tau1 < 0.03459
        + 0.01154732 * max(0.0, 0.2809341 - Q.LHA) / 0.06403989   # +1.2%  LHA < 0.2809
        - 0.01137151 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 53.4375 - Q.pt_7) / 0.4342437   # -1.1%  centroid_offset < 0.03776 and pt_7 < 53.44
        + 0.01094659 * max(0.0, 0.02045966 - Q.e2) / 0.005531925   # +1.1%  e2 < 0.02046
        + 0.01068167 * max(0.0, 0.03459477 - Q.tau1) * max(0.0, 0.002915531 - Q.girth2_top3) / 2.198359e-05   # +1.1%  tau1 < 0.03459 and girth2_top3 < 0.002916
        - 0.01034784 * max(0.0, 0.005019719 - Q.width) / 0.001903424   # -1.0%  width < 0.00502
        + 0.0103059 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # +1.0%  lam2 < 0.0005373
        + 0.01005788 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 0.1561228 - Q.z_3rd) / 0.0006510707   # +1.0%  centroid_offset < 0.03776 and z_3rd < 0.1561
        - 0.009690061 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.08543881 - Q.sj3_dr_min) / 0.0002784225   # -1.0%  centroid_offset < 0.01438 and sj3_dr_min < 0.08544
        - 0.009407068 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.M3 - 0.03843235) / 2.574583e-05   # -0.9%  centroid_offset > 0.0499 and M3 > 0.03843
        + 0.008729172 * max(0.0, 0.006390125 - Q.e2_sq) * max(0.0, 118.5 - Q.pt_2) / 0.07535924   # +0.9%  e2_sq < 0.00639 and pt_2 < 118.5
        + 0.008230217 * max(0.0, Q.log_sum_pt - 6.572938) / 0.08631512   # +0.8%  log_sum_pt > 6.573
        - 0.008113645 * max(0.0, 0.03360421 - Q.girth) / 0.006496997   # -0.8%  girth < 0.0336
        + 0.006784306 * max(0.0, Q.sj2_dr - 0.1872617) / 0.02732165   # +0.7%  sj2_dr > 0.1873
        + 0.006258038 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.2691019 - Q.tau21_b2) / 0.000521804   # +0.6%  centroid_offset < 0.01438 and tau21_b2 < 0.2691
        - 0.005980174 * max(0.0, 0.006390125 - Q.e2_sq) * max(0.0, 0.1471389 - Q.z_3rd) / 6.771565e-05   # -0.6%  e2_sq < 0.00639 and z_3rd < 0.1471
        - 0.005707994 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.6%  planar_flow < 0.2534 and sum_pt < 840
        - 0.005103758 * max(0.0, 0.001570267 - Q.mean_phi2) / 0.0006563898   # -0.5%  mean_phi2 < 0.00157
        + 0.004783198 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.0586137 - Q.z_7) / 0.002292896   # +0.5%  log_sum_pt > 6.573 and z_7 < 0.05861
        - 0.004496632 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # -0.4%  girth2 < 0.003563
        - 0.00445489 * max(0.0, Q.pt_7 - 29.04219) / 7.785987   # -0.4%  pt_7 > 29.04
        - 0.004294325 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 1.332146 - Q.D2) / 0.001440798   # -0.4%  centroid_offset < 0.01438 and D2 < 1.332
        + 0.003647183 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.05932655 - Q.tau21_b2) / 7.260318e-05   # +0.4%  centroid_offset < 0.01438 and tau21_b2 < 0.05933
        + 0.003476749 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.3%  sj2_dr > 0.2688
        - 0.003324479 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # -0.3%  centroid_offset > 0.0499 and D2 < 2.844
        - 0.003305952 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.01364517 - Q.tau3) / 4.041223e-05   # -0.3%  centroid_offset < 0.01438 and tau3 < 0.01365
        - 0.00316648 * max(0.0, 0.01323868 - Q.width) * max(0.0, 6.502799 - Q.log_sum_pt) / 0.0004341009   # -0.3%  width < 0.01324 and log_sum_pt < 6.503
        + 0.003151356 * max(0.0, 0.001570267 - Q.mean_phi2) * max(0.0, Q.M2 - 0.01411669) / 4.456624e-05   # +0.3%  mean_phi2 < 0.00157 and M2 > 0.01412
        - 0.002999311 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 37.15625 - Q.pt_7) / 0.5537514   # -0.3%  planar_flow < 0.2534 and pt_7 < 37.16
        + 0.002664815 * max(0.0, 0.09538712 - Q.tau1) * max(0.0, 62.25 - Q.pt_6) / 0.9985459   # +0.3%  tau1 < 0.09539 and pt_6 < 62.25
        - 0.002394708 * max(0.0, Q.log_sum_pt - 6.733425) / 0.02644661   # -0.2%  log_sum_pt > 6.733
        - 0.00203764 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 3.916009 - Q.D3) / 0.2165298   # -0.2%  log_sum_pt > 6.573 and D3 < 3.916
        - 0.001570205 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001130645 - Q.lam2) / 1.843299e-05   # -0.2%  sj2_dr > 0.1779 and lam2 < 0.001131
        - 0.001067854 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 1.050302e-05 - Q.e3) / 1.771164e-07   # -0.1%  planar_flow < 0.2534 and e3 < 1.05e-05
        + 0.0009540439 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.0001991733   # +0.1%  centroid_offset < 0.03776 and dr_max_012 > 0.1828
        + 0.0007305621 * max(0.0, 0.005019719 - Q.width) * max(0.0, Q.sj3_dr23 - 0.1414609) / 1.353256e-05   # +0.1%  width < 0.00502 and sj3_dr23 > 0.1415
        - 0.0002323485 * max(0.0, 0.0030133 - Q.e2_sq) * max(0.0, Q.girth2_top3 - 0.003952582) / 2.119623e-08   # -0.0%  e2_sq < 0.003013 and girth2_top3 > 0.003953
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.7875;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.7874883 * (-0.6582681
        + 0.2795935 * max(0.0, Q.girth2 - 0.01882765) / 0.0007605369   # +28.0%  girth2 > 0.01883
        - 0.1869772 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -18.7%  e2 > 0.06344
        + 0.1674 * max(0.0, Q.width - 0.01323868) / 0.001442684   # +16.7%  width > 0.01324
        - 0.06330303 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, 53.4375 - Q.pt_7) / 0.01439677   # -6.3%  girth2 > 0.01883 and pt_7 < 53.44
        + 0.05701013 * max(0.0, Q.girth2 - 0.02530566) / 0.0002851418   # +5.7%  girth2 > 0.02531
        - 0.05556975 * max(0.0, Q.ptdr0_4 - 15.26525) * max(0.0, 988.4078 - Q.sum_pt) / 57.72599   # -5.6%  ptdr0_4 > 15.27 and sum_pt < 988.4
        - 0.05419004 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, 96.6875 - Q.pt_2) / 0.02137101   # -5.4%  girth2 > 0.01883 and pt_2 < 96.69
        - 0.04961677 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, 0.7974684 - Q.planar_flow) / 0.000314313   # -5.0%  girth2 > 0.01883 and planar_flow < 0.7975
        + 0.03264073 * max(0.0, Q.ptdr0_4 - 15.26525) / 0.1831584   # +3.3%  ptdr0_4 > 15.27
        + 0.02941949 * max(0.0, Q.width - 0.01323868) * max(0.0, Q.log_sum_pt - 6.327379) / 8.505384e-05   # +2.9%  width > 0.01324 and log_sum_pt > 6.327
        + 0.02427932 * max(0.0, Q.ptdr0_4 - 15.26525) * max(0.0, 763.825 - Q.sum_pt) / 21.50512   # +2.4%  ptdr0_4 > 15.27 and sum_pt < 763.8
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 24.42;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.41948 * (0.05525479
        + 0.2644981 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # +26.4%  girth < 0.1484
        + 0.07371654 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +7.4%  width < 0.01324
        - 0.06816339 * max(0.0, Q.sum_pt_top5 - 482.2812) / 137.9506   # -6.8%  sum_pt_top5 > 482.3
        + 0.06730683 * max(0.0, Q.sum_pt_top5 - 482.2812) * max(0.0, 3.0374e-08 - Q.e4) / 4.059025e-06   # +6.7%  sum_pt_top5 > 482.3 and e4 < 3.037e-08
        - 0.06038478 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # -6.0%  C2_b2 < 0.009033
        - 0.0537264 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -5.4%  tau1 < 0.1136
        - 0.04412374 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -4.4%  girth < 0.1484 and log_sum_pt < 6.804
        - 0.04144614 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # -4.1%  e3 < 0.0001869
        - 0.03930831 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -3.9%  e2 < 0.08001
        - 0.03700006 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 1.399607e-06   # -3.7%  e3 < 8.148e-05 and centroid_offset < 0.03776
        + 0.03357286 * max(0.0, 0.02530566 - Q.girth2) / 0.01914557   # +3.4%  girth2 < 0.02531
        + 0.02869963 * max(0.0, Q.z_7 - 0.04624032) / 0.01204993   # +2.9%  z_7 > 0.04624
        + 0.01903937 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 40.04062 - Q.pt_7) / 652.1541   # +1.9%  sum_pt_top5 > 687.4 and pt_7 < 40.04
        - 0.01881463 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -1.9%  girth < 0.1484 and pt_7 < 38.53
        + 0.01757811 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +1.8%  e3 < 8.148e-05
        - 0.01714842 * max(0.0, Q.pt_7 - 31.85938) / 5.945918   # -1.7%  pt_7 > 31.86
        + 0.01459569 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +1.5%  lam1 < 0.01643
        - 0.01108311 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.0003061234 - Q.lam2) / 2.082734e-05   # -1.1%  girth < 0.1484 and lam2 < 0.0003061
        - 0.01078079 * max(0.0, 50.25 - Q.pt_6) / 11.66482   # -1.1%  pt_6 < 50.25
        - 0.008809995 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # -0.9%  width < 0.008678
        - 0.008436179 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 25.57812 - Q.pt_7) / 0.01870322   # -0.8%  lam1 < 0.01643 and pt_7 < 25.58
        - 0.007982091 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -0.8%  sj3_dr_max > 0.2337
        - 0.007677045 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # -0.8%  log_sum_pt < 6.464
        - 0.006206029 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.z_dr_0_0p05 - 0.3658817) / 0.004662966   # -0.6%  lam1 < 0.01643 and z_dr_0_0p05 > 0.3659
        + 0.006076663 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # +0.6%  LHA < 0.3127
        - 0.005652384 * max(0.0, 788.4484 - Q.sum_pt) / 112.1576   # -0.6%  sum_pt < 788.4
        + 0.005391848 * max(0.0, 0.08000524 - Q.e2) * max(0.0, 60.03125 - Q.pt_4) / 0.4322666   # +0.5%  e2 < 0.08001 and pt_4 < 60.03
        - 0.004194284 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, 2.468971 - Q.D3) / 151.1611   # -0.4%  sum_pt < 788.4 and D3 < 2.469
        + 0.003968577 * max(0.0, 0.0294431 - Q.M2) / 0.002929467   # +0.4%  M2 < 0.02944
        - 0.003623834 * max(0.0, 27.57812 - Q.pt_6) / 0.9875435   # -0.4%  pt_6 < 27.58
        - 0.003301607 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, 0.4309923 - Q.tau32) / 0.003864959   # -0.3%  sj3_dr_max > 0.2337 and tau32 < 0.431
        - 0.00207518 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # -0.2%  pt_5 < 24.58
        - 0.001746061 * max(0.0, Q.z_top5 - 0.8919245) / 0.004621611   # -0.2%  z_top5 > 0.8919
        - 0.001414 * max(0.0, Q.z_7 - 0.0586137) * max(0.0, Q.lam2 - 4.64158e-05) / 5.481848e-06   # -0.1%  z_7 > 0.05861 and lam2 > 4.642e-05
        - 0.001093538 * max(0.0, 24.57812 - Q.pt_5) * max(0.0, 169.875 - Q.pt_1) / 8.157701   # -0.1%  pt_5 < 24.58 and pt_1 < 169.9
        + 0.0009264061 * max(0.0, 24.57812 - Q.pt_5) * max(0.0, 0.1809419 - Q.z_2nd) / 0.00809881   # +0.1%  pt_5 < 24.58 and z_2nd < 0.1809
        + 0.0004374149 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.z_6 - 0.03932388) / 0.02599753   # +0.0%  sum_pt > 988.4 and z_6 > 0.03932
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 36.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 36.57133 * (-0.02216123
        - 0.08473153 * max(0.0, Q.e2_sq - 0.01165737) / 0.001433269   # -8.5%  e2_sq > 0.01166
        - 0.0718236 * max(0.0, Q.girth2 - 0.006679471) / 0.002702003   # -7.2%  girth2 > 0.006679
        + 0.06892902 * max(0.0, Q.girth - 0.04081947) / 0.0270611   # +6.9%  girth > 0.04082
        - 0.06125841 * max(0.0, 0.233678 - Q.sj3_dr_max) / 0.09057682   # -6.1%  sj3_dr_max < 0.2337
        - 0.05543249 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # -5.5%  girth > 0.1019
        - 0.05380274 * max(0.0, Q.girth2 - 0.008678045) * max(0.0, Q.log_sum_pt - 6.267538) / 0.0002118474   # -5.4%  girth2 > 0.008678 and log_sum_pt > 6.268
        - 0.05354263 * max(0.0, Q.tau1 - 0.03459477) / 0.03725748   # -5.4%  tau1 > 0.03459
        - 0.04637178 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -4.6%  girth2 > 0.00752
        - 0.04421632 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -4.4%  girth > 0.08724
        + 0.03917523 * max(0.0, Q.e2_sq - 0.00718279) / 0.002233568   # +3.9%  e2_sq > 0.007183
        + 0.03903329 * max(0.0, Q.girth - 0.02689598) / 0.03613842   # +3.9%  girth > 0.0269
        - 0.03682434 * max(0.0, Q.width - 0.01323868) / 0.001442684   # -3.7%  width > 0.01324
        + 0.02831373 * max(0.0, Q.e2_sq - 0.008168571) / 0.002013747   # +2.8%  e2_sq > 0.008169
        + 0.02530103 * max(0.0, Q.width - 0.005590289) / 0.003084256   # +2.5%  width > 0.00559
        - 0.02273659 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.158545   # -2.3%  n_dr_0p05_0p1 < 5
        + 0.0223691 * max(0.0, Q.e2_sq - 0.006390125) / 0.002450953   # +2.2%  e2_sq > 0.00639
        + 0.01665197 * max(0.0, Q.e2_sq - 0.008168571) * max(0.0, Q.sum_pt - 488.9312) / 0.1696086   # +1.7%  e2_sq > 0.008169 and sum_pt > 488.9
        + 0.01663469 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +1.7%  sj2_dr < 0.1592
        - 0.01560812 * max(0.0, Q.width - 0.01323868) * max(0.0, Q.sum_pt - 488.9312) / 0.1069057   # -1.6%  width > 0.01324 and sum_pt > 488.9
        - 0.01349279 * max(0.0, 0.005005443 - Q.girth2_top5) / 0.002177006   # -1.3%  girth2_top5 < 0.005005
        + 0.01211387 * max(0.0, Q.LHA - 0.2160559) / 0.05893777   # +1.2%  LHA > 0.2161
        + 0.01139383 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.3724174   # +1.1%  z_dr_0p05_0p1 < 0.5883
        + 0.01119275 * max(0.0, 0.00422683 - Q.girth2_top5) / 0.001731922   # +1.1%  girth2_top5 < 0.004227
        - 0.01117892 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -1.1%  centroid_offset > 0.03776
        + 0.01083553 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +1.1%  LHA > 0.3033
        + 0.01072269 * max(0.0, Q.n_dr_0_0p05 - 5.0) / 1.093146   # +1.1%  n_dr_0_0p05 > 5
        - 0.009181489 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.9%  centroid_offset > 0.0499
        - 0.00885011 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, Q.dr0_5 - 0.290089) / 2.769865e-05   # -0.9%  z_dr_0p05_0p1 > 0.7509 and dr0_5 > 0.2901
        + 0.007646937 * max(0.0, Q.psi_0p1 - 0.6332416) * max(0.0, Q.eccentricity - 0.9704496) / 0.001856694   # +0.8%  psi_0p1 > 0.6332 and eccentricity > 0.9704
        + 0.007532349 * max(0.0, Q.e2_sq - 0.002074109) * max(0.0, Q.log_sum_pt - 6.267538) / 0.0007382066   # +0.8%  e2_sq > 0.002074 and log_sum_pt > 6.268
        + 0.007061199 * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 0.8908874   # +0.7%  n_dr_0p1_0p2 > 1
        - 0.006853573 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.eta_5 - 0.1122437) / 6.758882e-06   # -0.7%  centroid_offset > 0.0499 and eta_5 > 0.1122
        - 0.006414502 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # -0.6%  C2 < 0.03579
        - 0.006366715 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # -0.6%  sj3_dr_max < 0.2134
        + 0.006022195 * max(0.0, 1.960701e-05 - Q.e3) / 8.488392e-06   # +0.6%  e3 < 1.961e-05
        + 0.005993897 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) / 0.09532154   # +0.6%  z_dr_0p1_0p2 < 0.1586
        + 0.005835438 * max(0.0, Q.tau1 - 0.09538712) / 0.01057268   # +0.6%  tau1 > 0.09539
        - 0.005284433 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.5%  psi_0p1 > 0.9761
        - 0.004710672 * max(0.0, Q.centroid_offset - 0.02355416) / 0.003991673   # -0.5%  centroid_offset > 0.02355
        + 0.004707323 * max(0.0, Q.e2_sq - 0.002074109) / 0.004484017   # +0.5%  e2_sq > 0.002074
        - 0.004583705 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -0.5%  planar_flow < 0.1115 and lam1 < 0.00733
        - 0.004368925 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -0.4%  e2 > 0.03556
        - 0.004291729 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -0.4%  sj2_dr < 0.1295
        - 0.003081124 * max(0.0, Q.e2_sq - 0.002074109) * max(0.0, 0.1872617 - Q.sj2_dr) / 2.014151e-05   # -0.3%  e2_sq > 0.002074 and sj2_dr < 0.1873
        - 0.002429783 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.2%  planar_flow < 0.1115 and centroid_offset < 0.01838
        - 0.001993954 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 0.002231562   # -0.2%  planar_flow < 0.1115 and z_dr_0p05_0p1 > 0.6748
        + 0.001806087 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +0.2%  e2 > 0.05028
        + 0.00129687 * max(0.0, Q.psi_0p1 - 0.8155839) * max(0.0, Q.centroid_offset - 0.03776099) / 4.931851e-05   # +0.1%  psi_0p1 > 0.8156 and centroid_offset > 0.03776
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 38.19;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 38.19448 * (-0.03937534
        - 0.1176883 * max(0.0, 0.007520088 - Q.girth2) / 0.00354499   # -11.8%  girth2 < 0.00752
        + 0.1076237 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +10.8%  girth2 < 0.01324
        - 0.06404689 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # -6.4%  e2_sq < 0.01166
        + 0.05522075 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # +5.5%  tau1 < 0.1028
        - 0.0493627 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -4.9%  tau1 < 0.1136
        + 0.04424401 * max(0.0, 0.006390125 - Q.e2_sq) / 0.002953508   # +4.4%  e2_sq < 0.00639
        + 0.04378468 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +4.4%  lam1 < 0.00733
        + 0.03657454 * max(0.0, 0.0811449 - Q.tau1) / 0.03266098   # +3.7%  tau1 < 0.08114
        + 0.03440742 * max(0.0, 0.01716248 - Q.e2_sq) / 0.0120354   # +3.4%  e2_sq < 0.01716
        - 0.03416602 * max(0.0, 0.006679471 - Q.girth2) / 0.00293624   # -3.4%  girth2 < 0.006679
        - 0.0295034 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -3.0%  lam1 < 0.008376
        - 0.02777613 * max(0.0, 0.06344108 - Q.e2) / 0.03657078   # -2.8%  e2 < 0.06344
        + 0.0264279 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +2.6%  e2 < 0.04111
        + 0.02460947 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +2.5%  N2 < 0.2233
        - 0.02159357 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.girth2 - 0.008678045) / 0.0001197728   # -2.2%  N2 < 0.2233 and girth2 > 0.008678
        + 0.02148399 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +2.1%  sj3_dr_max > 0.179
        - 0.02003197 * max(0.0, 0.002635418 - Q.width) / 0.0007941347   # -2.0%  width < 0.002635
        - 0.01794879 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -1.8%  lam1 < 0.005433
        + 0.01661582 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sum_pt_top5 - 402.625) / 15.59919   # +1.7%  planar_flow < 0.195 and sum_pt_top5 > 402.6
        - 0.01468181 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -1.5%  sj3_dr_max > 0.2623
        - 0.01371917 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 8.147744e-05 - Q.e3) / 2.656078e-06   # -1.4%  N2 < 0.2233 and e3 < 8.148e-05
        - 0.01343838 * max(0.0, Q.max_dr - 0.121681) / 0.03562206   # -1.3%  max_dr > 0.1217
        - 0.0126366 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -1.3%  lam2 < 0.001131
        + 0.01090922 * max(0.0, 0.05028464 - Q.e2) / 0.02502152   # +1.1%  e2 < 0.05028
        - 0.01080338 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # -1.1%  girth2_top3 < 0.002152
        - 0.0102811 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.4235925) / 7.321856e-05   # -1.0%  N2 < 0.2233 and LHA > 0.4236
        - 0.01022449 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # -1.0%  z_dr_0p05_0p1 > 0.7509
        + 0.009863679 * max(0.0, 0.005834489 - Q.e2_sq) / 0.002579453   # +1.0%  e2_sq < 0.005834
        + 0.007855538 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # +0.8%  sj3_dr_max > 0.2337
        - 0.007756089 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 0.07148865 - Q.z_7) / 0.001619322   # -0.8%  planar_flow < 0.195 and z_7 < 0.07149
        + 0.007715964 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.2931906) / 0.001536651   # +0.8%  N2 < 0.2233 and LHA > 0.2932
        + 0.006704209 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3255822) / 0.0007869707   # +0.7%  N2 < 0.2233 and LHA > 0.3256
        + 0.006588497 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +0.7%  sj3_dr_max > 0.1879
        + 0.006458157 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.001172117   # +0.6%  z_dr_0p05_0p1 > 0.7509 and z_dr_0p2_0p4 < 0.05644
        - 0.006291438 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1872617 - Q.sj2_dr) / 0.0007896743   # -0.6%  N2 < 0.2233 and sj2_dr < 0.1873
        + 0.006147559 * max(0.0, 0.08082334 - Q.dr_0) / 0.0377444   # +0.6%  dr_0 < 0.08082
        + 0.005662574 * max(0.0, Q.sj3_dr_max - 0.1879486) * max(0.0, 169.875 - Q.pt_1) / 2.480285   # +0.6%  sj3_dr_max > 0.1879 and pt_1 < 169.9
        - 0.005496044 * max(0.0, Q.sj3_dr_max - 0.1426152) * max(0.0, 169.875 - Q.pt_1) / 3.727834   # -0.5%  sj3_dr_max > 0.1426 and pt_1 < 169.9
        - 0.004576087 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.pt_entropy - 1.797618) / 0.0002596149   # -0.5%  girth2 < 0.006679 and pt_entropy > 1.798
        + 0.004498582 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +0.4%  N2 < 0.2233 and sum_pt_top5 > 430.8
        - 0.004270241 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.0114178   # -0.4%  N2 < 0.2233 and z_dr_0p05_0p1 < 0.5883
        - 0.00409145 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.mean_phi - -0.01753483) / 0.001413063   # -0.4%  planar_flow < 0.195 and mean_phi > -0.01753
        + 0.003244663 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 4.721224 - Q.D2_b2) / 0.1022787   # +0.3%  z_dr_0p05_0p1 > 0.7509 and D2_b2 < 4.721
        - 0.00288296 * max(0.0, 0.002151568 - Q.girth2_top3) * max(0.0, Q.eccentricity - 0.9598562) / 4.406774e-06   # -0.3%  girth2_top3 < 0.002152 and eccentricity > 0.9599
        + 0.002707763 * max(0.0, 0.1515405 - Q.z_dr_0_0p05) / 0.0478325   # +0.3%  z_dr_0_0p05 < 0.1515
        - 0.002084725 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3855647) / 0.0002330351   # -0.2%  N2 < 0.2233 and LHA > 0.3856
        - 0.001874232 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, 182.125 - Q.pt_1) / 1.935314   # -0.2%  sj3_dr_max > 0.2337 and pt_1 < 182.1
        - 0.001125296 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.dr12 - 0.1587481) / 1.436546e-06   # -0.1%  girth2 < 0.006679 and dr12 > 0.1587
        - 0.0009699922 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 25.57812 - Q.pt_7) / 0.037942   # -0.1%  N2 < 0.2233 and pt_7 < 25.58
        + 0.0008849646 * max(0.0, 0.006390125 - Q.e2_sq) * max(0.0, Q.dr12 - 0.1587481) / 1.354517e-06   # +0.1%  e2_sq < 0.00639 and dr12 > 0.1587
        - 0.0004450604 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.centroid_offset - 0.04990367) / 8.981184e-07   # -0.0%  girth2 < 0.01324 and centroid_offset > 0.0499
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9090254201680672, 0.7858619747899159, 2.0980659663865544, 0.9102918067226891, 1.2175873949579832, 1.7099191176470587, 1.126332142857143, 1.8875762605042017, 0.25706827731092435, 2.9838747899159666, 1.9502444327731092, 2.0767448529411765, 0.08944096638655462, 3.3167813025210084, 0.42295115546218487, 0.362040756302521]
T = [2.3452307740283613, 1.3751921218487395, 3.2926877281381297, 2.5290974313944328, 2.8224798155199577]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +38%, n9 +22%, n5 -14%, n1 +13%, n0 -6%, n6 +5% ...
            + 0.3844026 * h[2] / H_AVG[2]
            + 0.2186793 * h[9] / H_AVG[9]
            - 0.1367072 * h[5] / H_AVG[5]
            + 0.1308943 * h[1] / H_AVG[1]
            - 0.06056343 * h[0] / H_AVG[0]
            + 0.05252898 * h[6] / H_AVG[6]
            - 0.01622425 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +55%, n10 -18%, n6 +10%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5509226 * h[9] / H_AVG[9]
            - 0.1772702 * h[10] / H_AVG[10]
            + 0.1023795 * h[6] / H_AVG[6]
            - 0.08300572 * h[4] / H_AVG[4]
            + 0.05828455 * h[5] / H_AVG[5]
            + 0.0164541 * h[15] / H_AVG[15]
            + 0.01168329 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +13%, n6 -11%, n14 -10%, n0 +9% ...
            + 0.2365178 * h[11] / H_AVG[11]
            - 0.1382293 * h[3] / H_AVG[3]
            + 0.1254013 * h[7] / H_AVG[7]
            - 0.1068971 * h[6] / H_AVG[6]
            - 0.09633873 * h[14] / H_AVG[14]
            + 0.09490043 * h[0] / H_AVG[0]
            - 0.07559266 * h[15] / H_AVG[15]
            + 0.070827 * h[13] / H_AVG[13]
            - 0.02831914 * h[9] / H_AVG[9]
            - 0.01951812 * h[8] / H_AVG[8]
            - 0.007458401 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -20%, n6 -17%, n13 +7%, n14 +6%, n1 +4% ...
            + 0.3498487 * h[7] / H_AVG[7]
            - 0.2024592 * h[3] / H_AVG[3]
            - 0.167006 * h[6] / H_AVG[6]
            + 0.07171985 * h[13] / H_AVG[13]
            + 0.06271276 * h[14] / H_AVG[14]
            + 0.03884103 * h[1] / H_AVG[1]
            + 0.03761184 * h[4] / H_AVG[4]
            - 0.03686931 * h[9] / H_AVG[9]
            - 0.02236722 * h[15] / H_AVG[15]
            + 0.01056404 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -48%, n10 +26%, n5 -15%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4773966 * h[13] / H_AVG[13]
            + 0.2591132 * h[10] / H_AVG[10]
            - 0.1514554 * h[5] / H_AVG[5]
            + 0.05392365 * h[4] / H_AVG[4]
            + 0.02015718 * h[3] / H_AVG[3]
            + 0.01707729 * h[8] / H_AVG[8]
            - 0.01584439 * h[12] / H_AVG[12]
            + 0.005032285 * h[0] / H_AVG[0]
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
