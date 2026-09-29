"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network (step 4; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.2%   (on for 86% of jets)
  neuron  9:  12.0%   (on for 69% of jets)
  neuron  7:  10.4%   (on for 61% of jets)
  neuron  3:   8.4%   (on for 25% of jets)
  neuron 10:   7.9%   (on for 75% of jets)
  neuron  6:   7.6%   (on for 33% of jets)
  neuron  2:   7.2%   (on for 89% of jets)
  neuron  5:   7.2%   (on for 73% of jets)
  neuron 11:   6.3%   (on for 73% of jets)
  neuron 14:   3.9%   (on for 29% of jets)
  neuron  0:   3.8%   (on for 42% of jets)
  neuron  1:   3.6%   (on for 60% of jets)
  neuron  4:   3.3%   (on for 45% of jets)
  neuron 15:   2.6%   (on for 25% of jets)
  neuron  8:   1.1%   (on for 35% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 89.8% of jets.

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
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_nremoved            number of branches removed by soft drop
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
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
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        pt_2=pt[2],
        pt_3=pt[3],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_nremoved=softdrop("removed"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        eta_0=eta[0],
        sum_pt_top3=sum(pt[:3]),
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
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 14.72;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.72001 * (0.1107336
        + 0.1127768 * max(0.0, 0.012 - Q.lam1) / 0.00731311   # +11.3%  lam1 < 0.012
        - 0.1022975 * max(0.0, 0.08 - Q.girth) / 0.03079388   # -10.2%  girth < 0.08
        - 0.08198173 * max(0.0, 0.0043 - Q.lam1) / 0.001571318   # -8.2%  lam1 < 0.0043
        - 0.07502459 * max(0.0, 760.0 - Q.sum_pt) / 94.38999   # -7.5%  sum_pt < 760
        - 0.06906003 * max(0.0, 0.018 - Q.tau2) / 0.008763488   # -6.9%  tau2 < 0.018
        - 0.06863321 * max(0.0, Q.sj3_dr_max - 0.19) / 0.03856038   # -6.9%  sj3_dr_max > 0.19
        - 0.06452314 * max(0.0, Q.centroid_offset - 0.0064) / 0.01168243   # -6.5%  centroid_offset > 0.0064
        + 0.05947759 * max(0.0, 0.052 - Q.tau1) / 0.0161832   # +5.9%  tau1 < 0.052
        + 0.05728692 * max(0.0, Q.sj3_dr_max - 0.14) / 0.06437131   # +5.7%  sj3_dr_max > 0.14
        + 0.05052148 * max(0.0, 8e-05 - Q.e3) / 5.508718e-05   # +5.1%  e3 < 8e-05
        - 0.04823708 * max(0.0, 0.0059 - Q.lam1) * max(0.0, 0.98 - Q.n_dr_0p2_0p4) / 0.002320426   # -4.8%  lam1 < 0.0059 and n_dr_0p2_0p4 < 0.98
        + 0.03402722 * max(0.0, 0.99 - Q.n_dr_0p2_0p4) / 0.7988535   # +3.4%  n_dr_0p2_0p4 < 0.99
        - 0.03157459 * max(0.0, Q.LHA - 0.3) / 0.01897055   # -3.2%  LHA > 0.3
        - 0.02411768 * max(0.0, Q.log_sum_pt - 6.8) / 0.01319749   # -2.4%  log_sum_pt > 6.8
        + 0.02170432 * max(0.0, 770.0 - Q.sum_pt) * max(0.0, Q.centroid_offset - 0.011) / 1.621766   # +2.2%  sum_pt < 770 and centroid_offset > 0.011
        + 0.01882768 * max(0.0, 0.15 - Q.planar_flow) / 0.05189956   # +1.9%  planar_flow < 0.15
        + 0.01649997 * max(0.0, Q.sum_pt_top5 - 790.0) / 14.71999   # +1.6%  sum_pt_top5 > 790
        + 0.01613045 * max(0.0, 720.0 - Q.sum_pt) * max(0.0, Q.z_7 - 0.017) / 3.886094   # +1.6%  sum_pt < 720 and z_7 > 0.017
        + 0.01396196 * max(0.0, 0.0044 - Q.lam1) * max(0.0, 770.0 - Q.sum_pt) / 0.09740293   # +1.4%  lam1 < 0.0044 and sum_pt < 770
        + 0.01156585 * max(0.0, Q.sj3_dr_max - 0.12) * max(0.0, Q.D2 - -0.076) / 0.119894   # +1.2%  sj3_dr_max > 0.12 and D2 > -0.076
        + 0.006651678 * max(0.0, 0.043 - Q.planar_flow) * max(0.0, 0.057 - Q.tau21_b2) / 0.0003189342   # +0.7%  planar_flow < 0.043 and tau21_b2 < 0.057
        + 0.004093737 * max(0.0, Q.log_sum_pt - 6.8) * max(0.0, 0.046 - Q.z_dr_0p1_0p2) / 0.0004939333   # +0.4%  log_sum_pt > 6.8 and z_dr_0p1_0p2 < 0.046
        - 0.003603886 * max(0.0, 27.0 - Q.pt_6) / 0.9052773   # -0.4%  pt_6 < 27
        - 0.002697545 * max(0.0, 0.0042 - Q.lam1) * max(0.0, 0.75 - Q.D2) / 6.834405e-06   # -0.3%  lam1 < 0.0042 and D2 < 0.75
        + 0.002444449 * max(0.0, 0.14 - Q.planar_flow) * max(0.0, Q.C3 - 0.05) / 0.0001022225   # +0.2%  planar_flow < 0.14 and C3 > 0.05
        + 0.002278897 * max(0.0, Q.sj3_dr_max - 0.15) * max(0.0, 0.067 - Q.dr_7) / 0.0003829383   # +0.2%  sj3_dr_max > 0.15 and dr_7 < 0.067
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 13.19;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.19341 * (0.09929196
        - 0.1609266 * max(0.0, 0.11 - Q.girth) / 0.05558039   # -16.1%  girth < 0.11
        - 0.1159548 * max(0.0, 0.064 - Q.z_7) / 0.01580413   # -11.6%  z_7 < 0.064
        - 0.1119913 * max(0.0, 0.0084 - Q.lam1) * max(0.0, 0.0011 - Q.lam2) / 4.371442e-06   # -11.2%  lam1 < 0.0084 and lam2 < 0.0011
        + 0.10596 * max(0.0, Q.log_sum_pt - 6.4) / 0.1930903   # +10.6%  log_sum_pt > 6.4
        + 0.09018688 * max(0.0, 0.1 - Q.tau1) / 0.0461191   # +9.0%  tau1 < 0.1
        + 0.085616 * max(0.0, 0.064 - Q.z_7) * max(0.0, 0.0005 - Q.e3) / 7.480579e-06   # +8.6%  z_7 < 0.064 and e3 < 0.0005
        + 0.05224179 * max(0.0, 0.0013 - Q.lam2) / 0.001066947   # +5.2%  lam2 < 0.0013
        - 0.04708425 * max(0.0, 0.34 - Q.sj3_dr_max) / 0.1790208   # -4.7%  sj3_dr_max < 0.34
        + 0.03817749 * max(0.0, Q.log_sum_pt - 6.6) / 0.0734244   # +3.8%  log_sum_pt > 6.6
        - 0.03676984 * max(0.0, 0.048 - Q.z_6) / 0.00453383   # -3.7%  z_6 < 0.048
        - 0.03626852 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0066 - Q.girth2_top3) / 0.0003738325   # -3.6%  log_sum_pt > 6.6 and girth2_top3 < 0.0066
        + 0.03068337 * max(0.0, 0.0025 - Q.lam1) / 0.0007638084   # +3.1%  lam1 < 0.0025
        + 0.02065281 * max(0.0, 0.046 - Q.z_6) * max(0.0, 0.1 - Q.z_dr_0p2_0p4) / 0.0003538715   # +2.1%  z_6 < 0.046 and z_dr_0p2_0p4 < 0.1
        + 0.01581262 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.12 - Q.dr_max_012) / 0.006321891   # +1.6%  log_sum_pt > 6.6 and dr_max_012 < 0.12
        - 0.01139904 * max(0.0, 0.058 - Q.z_7) * max(0.0, 0.76 - Q.D2_b2) / 0.003081807   # -1.1%  z_7 < 0.058 and D2_b2 < 0.76
        + 0.01125817 * max(0.0, Q.pt_7 - 42.0) / 1.766156   # +1.1%  pt_7 > 42
        + 0.007670916 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 5.9 - Q.n_dr_0_0p05) / 11.42275   # +0.8%  pt_7 > 34 and n_dr_0_0p05 < 5.9
        + 0.006319234 * max(0.0, 0.0087 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021) / 7.7918e-06   # +0.6%  lam1 < 0.0087 and centroid_offset > 0.021
        - 0.004890246 * max(0.0, 0.0093 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.1) / 0.001510985   # -0.5%  lam1 < 0.0093 and n_pt_above_50 > 6.1
        + 0.00347403 * max(0.0, 0.15 - Q.sj2_dr) * max(0.0, 0.73 - Q.D2_b2) / 0.00224678   # +0.3%  sj2_dr < 0.15 and D2_b2 < 0.73
        - 0.00338551 * max(0.0, Q.sum_pt - 990.0) * max(0.0, 0.0086 - Q.girth2_top2) / 0.03409651   # -0.3%  sum_pt > 990 and girth2_top2 < 0.0086
        - 0.003276567 * max(0.0, Q.pt_7 - 54.0) / 0.3044303   # -0.3%  pt_7 > 54
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 13.39;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.39094 * (0.5720287
        - 0.239275 * max(0.0, Q.log_sum_pt - 6.1) / 0.4500163   # -23.9%  log_sum_pt > 6.1
        - 0.1254285 * max(0.0, Q.LHA - 0.11) / 0.1365532   # -12.5%  LHA > 0.11
        - 0.08238958 * max(0.0, 42.0 - Q.pt_7) / 9.117964   # -8.2%  pt_7 < 42
        - 0.07809102 * max(0.0, 54.0 - Q.pt_7) / 19.65624   # -7.8%  pt_7 < 54
        + 0.06852509 * max(0.0, 0.053 - Q.z_7) / 0.009185338   # +6.9%  z_7 < 0.053
        + 0.06116587 * max(0.0, 0.012 - Q.lam1) / 0.00731311   # +6.1%  lam1 < 0.012
        - 0.05971898 * max(0.0, Q.z_7 - 0.052) / 0.008855959   # -6.0%  z_7 > 0.052
        + 0.05376001 * max(0.0, 0.0072 - Q.lam1) * max(0.0, 0.25 - Q.max_dr) / 0.0006100822   # +5.4%  lam1 < 0.0072 and max_dr < 0.25
        - 0.03536449 * max(0.0, Q.log_sum_pt - 6.1) * max(0.0, 8.7e-05 - Q.mean_phi2) / 6.451821e-06   # -3.5%  log_sum_pt > 6.1 and mean_phi2 < 8.7e-05
        - 0.03464029 * max(0.0, 0.045 - Q.e2) / 0.02061626   # -3.5%  e2 < 0.045
        + 0.03280077 * max(0.0, 8.7e-05 - Q.mean_phi2) / 9.959933e-06   # +3.3%  mean_phi2 < 8.7e-05
        + 0.03048768 * max(0.0, 650.0 - Q.sum_pt) / 41.44757   # +3.0%  sum_pt < 650
        - 0.02015124 * max(0.0, 0.0072 - Q.lam1) * max(0.0, 62.0 - Q.pt_6) / 0.07687863   # -2.0%  lam1 < 0.0072 and pt_6 < 62
        - 0.01761385 * max(0.0, 0.04 - Q.C2) / 0.01828419   # -1.8%  C2 < 0.04
        + 0.01464 * max(0.0, Q.pt_7 - 42.0) / 1.766156   # +1.5%  pt_7 > 42
        + 0.0144484 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.11 - Q.absphi_0) / 0.0003831239   # +1.4%  log_sum_pt > 6.9 and absphi_0 < 0.11
        - 0.008682533 * max(0.0, Q.sum_pt - 990.0) * max(0.0, 0.1 - Q.absphi_0) / 0.3875575   # -0.9%  sum_pt > 990 and absphi_0 < 0.1
        + 0.008135317 * max(0.0, Q.sum_pt - 860.0) * max(0.0, 4.8 - Q.D2_b2) / 50.90632   # +0.8%  sum_pt > 860 and D2_b2 < 4.8
        + 0.00349133 * max(0.0, 320.0 - Q.sum_pt_top5) / 1.757601   # +0.3%  sum_pt_top5 < 320
        - 0.003192606 * max(0.0, 0.0055 - Q.centroid_offset) * max(0.0, 0.037 - Q.dr_7) / 9.23369e-06   # -0.3%  centroid_offset < 0.0055 and dr_7 < 0.037
        - 0.002682258 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 4.6 - Q.D2_b2) / 0.009162742   # -0.3%  log_sum_pt > 6.9 and D2_b2 < 4.6
        - 0.002529785 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 8.7e-05 - Q.mean_phi2) / 4.909593e-07   # -0.3%  log_sum_pt < 6.7 and mean_phi2 < 8.7e-05
        + 0.002011304 * max(0.0, Q.sum_pt - 940.0) * max(0.0, 8.3e-05 - Q.mean_phi2) / 0.0002321831   # +0.2%  sum_pt > 940 and mean_phi2 < 8.3e-05
        - 0.0007741872 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.n_pt_above_50 - 5.1) / 0.003742633   # -0.1%  log_sum_pt > 6.9 and n_pt_above_50 > 5.1
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 8.89;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.89018 * (0.1092216
        - 0.4055435 * max(0.0, 0.0036 - Q.lam2) / 0.003219067   # -40.6%  lam2 < 0.0036
        + 0.09293933 * max(0.0, Q.LHA - 0.29) / 0.02239153   # +9.3%  LHA > 0.29
        + 0.08641543 * max(0.0, Q.sj2_dr - 0.17) / 0.03414439   # +8.6%  sj2_dr > 0.17
        - 0.06122271 * max(0.0, Q.LHA - 0.33) / 0.01160514   # -6.1%  LHA > 0.33
        + 0.04720966 * max(0.0, Q.sj2_dr - 0.16) * max(0.0, 0.1 - Q.C2) / 0.001756077   # +4.7%  sj2_dr > 0.16 and C2 < 0.1
        - 0.0470204 * max(0.0, Q.lam1 - 0.012) / 0.00122947   # -4.7%  lam1 > 0.012
        + 0.0455357 * max(0.0, Q.sj3_dr_max - 0.2) / 0.03489832   # +4.6%  sj3_dr_max > 0.2
        + 0.04367401 * max(0.0, Q.girth - 0.065) * max(0.0, Q.eccentricity - 0.68) / 0.003131208   # +4.4%  girth > 0.065 and eccentricity > 0.68
        - 0.03593861 * max(0.0, Q.sj2_dr - 0.17) * max(0.0, Q.log_sum_pt - 6.0) / 0.01472354   # -3.6%  sj2_dr > 0.17 and log_sum_pt > 6
        + 0.03517586 * max(0.0, Q.sj2_dr - 0.17) * max(0.0, Q.eccentricity - 0.95) / 0.0007740587   # +3.5%  sj2_dr > 0.17 and eccentricity > 0.95
        - 0.02147281 * max(0.0, Q.sj2_dr - 0.17) * max(0.0, Q.n_dr_0_0p05 - 2.1) / 0.03887926   # -2.1%  sj2_dr > 0.17 and n_dr_0_0p05 > 2.1
        + 0.01760573 * max(0.0, Q.girth - 0.13) / 0.002132399   # +1.8%  girth > 0.13
        + 0.01329871 * max(0.0, -0.007 - Q.mean_eta) / 0.003119471   # +1.3%  mean_eta < -0.007
        - 0.01098965 * max(0.0, Q.sj3_dr_max - 0.27) * max(0.0, Q.n_dr_0p05_0p1 - 3.9) / 0.004630329   # -1.1%  sj3_dr_max > 0.27 and n_dr_0p05_0p1 > 3.9
        + 0.00979004 * max(0.0, Q.sj3_dr_max - 0.21) * max(0.0, Q.n_dr_0p05_0p1 - 3.9) / 0.008945038   # +1.0%  sj3_dr_max > 0.21 and n_dr_0p05_0p1 > 3.9
        - 0.00960621 * max(0.0, Q.sd_rg - 0.29) / 0.004086169   # -1.0%  sd_rg > 0.29
        + 0.006042623 * max(0.0, Q.sd_rg - 0.17) * max(0.0, 0.00057 - Q.lam2) / 4.928441e-06   # +0.6%  sd_rg > 0.17 and lam2 < 0.00057
        - 0.005562934 * max(0.0, Q.max_dr - 0.25) / 0.004665612   # -0.6%  max_dr > 0.25
        - 0.004956048 * max(0.0, -0.011 - Q.mean_eta) * max(0.0, -0.026 - Q.eta_0) / 9.856859e-05   # -0.5%  mean_eta < -0.011 and eta_0 < -0.026
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 22.19;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.18821 * (0.1735156
        + 0.1544267 * max(0.0, 0.00019 - Q.e3) / 0.0001522867   # +15.4%  e3 < 0.00019
        - 0.1541222 * max(0.0, 0.0067 - Q.lam1) / 0.00302628   # -15.4%  lam1 < 0.0067
        - 0.1335547 * max(0.0, 0.0084 - Q.lam1) / 0.004319738   # -13.4%  lam1 < 0.0084
        - 0.09815488 * max(0.0, Q.eccentricity - 0.71) / 0.1962055   # -9.8%  eccentricity > 0.71
        - 0.08963351 * max(0.0, 0.00064 - Q.lam2) / 0.0004792306   # -9.0%  lam2 < 0.00064
        + 0.07732117 * max(0.0, 0.0068 - Q.lam1) * max(0.0, Q.eccentricity - 0.73) / 0.0004310599   # +7.7%  lam1 < 0.0068 and eccentricity > 0.73
        + 0.04618708 * max(0.0, 0.22 - Q.N2) / 0.04950767   # +4.6%  N2 < 0.22
        + 0.04605185 * max(0.0, Q.log_sum_pt - 6.3) / 0.2710366   # +4.6%  log_sum_pt > 6.3
        - 0.04318082 * max(0.0, 0.14 - Q.sj3_dr_max) / 0.03615491   # -4.3%  sj3_dr_max < 0.14
        + 0.02577413 * max(0.0, Q.max_dr - 0.12) * max(0.0, 0.0089 - Q.C2_b2) / 0.0001571104   # +2.6%  max_dr > 0.12 and C2_b2 < 0.0089
        - 0.02343304 * max(0.0, 0.043 - Q.M2) / 0.008023723   # -2.3%  M2 < 0.043
        - 0.01971179 * max(0.0, 750.0 - Q.sum_pt) / 88.53628   # -2.0%  sum_pt < 750
        - 0.01511706 * max(0.0, Q.sj3_dr_max - 0.23) * max(0.0, Q.log_sum_pt - 6.3) / 0.003735193   # -1.5%  sj3_dr_max > 0.23 and log_sum_pt > 6.3
        + 0.01357063 * max(0.0, 0.082 - Q.dr_0) * max(0.0, Q.centroid_offset - 0.0083) / 0.0002120478   # +1.4%  dr_0 < 0.082 and centroid_offset > 0.0083
        - 0.01181372 * max(0.0, 800.0 - Q.sum_pt) * max(0.0, 0.1 - Q.M3) / 3.244124   # -1.2%  sum_pt < 800 and M3 < 0.1
        - 0.01002244 * max(0.0, 0.21 - Q.N2) * max(0.0, 0.29 - Q.sj2_zsoft) / 0.002398921   # -1.0%  N2 < 0.21 and sj2_zsoft < 0.29
        - 0.009725064 * max(0.0, Q.sj2_dr - 0.23) * max(0.0, 0.65 - Q.D2_b2) / 0.004222735   # -1.0%  sj2_dr > 0.23 and D2_b2 < 0.65
        - 0.009095722 * max(0.0, Q.sj2_dr - 0.19) * max(0.0, Q.lam2 - 0.0018) / 2.734658e-05   # -0.9%  sj2_dr > 0.19 and lam2 > 0.0018
        - 0.00803688 * max(0.0, 27.0 - Q.pt_7) / 1.636   # -0.8%  pt_7 < 27
        + 0.006210377 * max(0.0, 0.24 - Q.N2) * max(0.0, Q.n_dr_0p1_0p2 - 1.5) / 0.07030466   # +0.6%  N2 < 0.24 and n_dr_0p1_0p2 > 1.5
        + 0.001950846 * max(0.0, Q.sj3_dr_max - 0.2) * max(0.0, 0.06 - Q.sj2_zsoft) / 9.492496e-05   # +0.2%  sj3_dr_max > 0.2 and sj2_zsoft < 0.06
        + 0.001935022 * max(0.0, 0.0073 - Q.lam1) * max(0.0, Q.lam2 - 0.00059) / 4.935019e-08   # +0.2%  lam1 < 0.0073 and lam2 > 0.00059
        + 0.0009703326 * max(0.0, 0.19 - Q.sj3_dr_min) * max(0.0, -0.023 - Q.mean_phi) / 9.360844e-05   # +0.1%  sj3_dr_min < 0.19 and mean_phi < -0.023
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.04;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.0385 * (0.1544055
        - 0.1246791 * max(0.0, 740.0 - Q.sum_pt) / 82.88687   # -12.5%  sum_pt < 740
        + 0.1136281 * max(0.0, 580.0 - Q.sum_pt_top5) / 65.93389   # +11.4%  sum_pt_top5 < 580
        + 0.0904874 * max(0.0, 0.21 - Q.LHA) / 0.0298802   # +9.0%  LHA < 0.21
        - 0.0892178 * max(0.0, 0.063 - Q.dr_0) / 0.02440363   # -8.9%  dr_0 < 0.063
        + 0.07071681 * max(0.0, 0.038 - Q.e2) * max(0.0, 0.00028 - Q.lam2) / 3.479858e-06   # +7.1%  e2 < 0.038 and lam2 < 0.00028
        - 0.06248178 * max(0.0, 0.038 - Q.centroid_offset) / 0.02248113   # -6.2%  centroid_offset < 0.038
        - 0.05449462 * max(0.0, 0.21 - Q.LHA) * max(0.0, 0.096 - Q.z_6) / 0.001428314   # -5.4%  LHA < 0.21 and z_6 < 0.096
        + 0.05121548 * max(0.0, 0.0022 - Q.lam1) / 0.0006467003   # +5.1%  lam1 < 0.0022
        + 0.0489296 * max(0.0, 0.0023 - Q.lam1) * max(0.0, 0.021 - Q.centroid_offset) / 9.012477e-06   # +4.9%  lam1 < 0.0023 and centroid_offset < 0.021
        - 0.04665338 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt) / 0.004257547   # -4.7%  LHA < 0.22 and log_sum_pt < 6.8
        + 0.04531073 * max(0.0, 0.038 - Q.e2) * max(0.0, 0.075 - Q.z_7) / 0.0004703743   # +4.5%  e2 < 0.038 and z_7 < 0.075
        - 0.03788139 * max(0.0, 0.82 - Q.z_top5) / 0.02796121   # -3.8%  z_top5 < 0.82
        - 0.0371268 * max(0.0, 0.025 - Q.tau1) / 0.004936391   # -3.7%  tau1 < 0.025
        + 0.03063215 * max(0.0, 0.061 - Q.z_6) * max(0.0, 29.0 - Q.pt1_dr01) / 0.2421267   # +3.1%  z_6 < 0.061 and pt1_dr01 < 29
        - 0.02472898 * max(0.0, Q.log_sum_pt - 6.8) * max(0.0, 6e-05 - Q.e3) / 7.279822e-07   # -2.5%  log_sum_pt > 6.8 and e3 < 6e-05
        + 0.01918071 * max(0.0, 0.022 - Q.girth) * max(0.0, Q.sum_pt_top5 - 720.0) / 0.2481259   # +1.9%  girth < 0.022 and sum_pt_top5 > 720
        - 0.01660617 * max(0.0, Q.pt_7 - 39.0) / 2.621086   # -1.7%  pt_7 > 39
        + 0.01614436 * max(0.0, 0.033 - Q.z_7) / 0.002175372   # +1.6%  z_7 < 0.033
        + 0.009635963 * max(0.0, 7.2e-05 - Q.lam1) / 3.454666e-06   # +1.0%  lam1 < 7.2e-05
        - 0.006031935 * max(0.0, 8.5e-05 - Q.lam1) * max(0.0, 73.0 - Q.pt_5) / 0.0001173481   # -0.6%  lam1 < 8.5e-05 and pt_5 < 73
        + 0.004216702 * max(0.0, 0.022 - Q.z_6) / 0.0003306982   # +0.4%  z_6 < 0.022
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 20.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.00993 * (0.1089459
        - 0.1186923 * max(0.0, Q.sj3_dr_max - 0.032) / 0.1388903   # -11.9%  sj3_dr_max > 0.032
        - 0.112258 * max(0.0, 0.18 - Q.max_dr) / 0.07246046   # -11.2%  max_dr < 0.18
        - 0.1039385 * max(0.0, Q.girth - 0.088) / 0.007702969   # -10.4%  girth > 0.088
        - 0.07642351 * max(0.0, 0.001 - Q.lam2) / 0.0007964735   # -7.6%  lam2 < 0.001
        + 0.07448531 * max(0.0, 0.036 - Q.C2) / 0.01520863   # +7.4%  C2 < 0.036
        + 0.05916038 * max(0.0, Q.sj3_dr_max - 0.18) / 0.04273628   # +5.9%  sj3_dr_max > 0.18
        + 0.05401445 * max(0.0, Q.LHA - 0.26) / 0.03509173   # +5.4%  LHA > 0.26
        - 0.04586755 * max(0.0, 0.021 - Q.centroid_offset) / 0.008498207   # -4.6%  centroid_offset < 0.021
        + 0.03537934 * max(0.0, Q.sj2_dr - 0.16) * max(0.0, Q.sj2_zsoft - 0.054) / 0.005949059   # +3.5%  sj2_dr > 0.16 and sj2_zsoft > 0.054
        - 0.03251583 * max(0.0, Q.psi_0p2 - 0.9) / 0.08471869   # -3.3%  psi_0p2 > 0.9
        + 0.02738904 * max(0.0, Q.sj3_dr_max - 0.17) * max(0.0, 1.6 - Q.D2_b2) / 0.04248471   # +2.7%  sj3_dr_max > 0.17 and D2_b2 < 1.6
        + 0.02633059 * max(0.0, Q.LHA - 0.33) / 0.01160514   # +2.6%  LHA > 0.33
        + 0.02444402 * max(0.0, Q.centroid_offset - 0.0073) * max(0.0, 0.21 - Q.sj3_dr_min) / 0.001552772   # +2.4%  centroid_offset > 0.0073 and sj3_dr_min < 0.21
        + 0.0187909 * max(0.0, 0.02 - Q.centroid_offset) * max(0.0, Q.n_dr_0p05_0p1 - -0.52) / 0.01816447   # +1.9%  centroid_offset < 0.02 and n_dr_0p05_0p1 > -0.52
        + 0.01832941 * max(0.0, 39.0 - Q.pt_6) * max(0.0, 720.0 - Q.sum_pt_top3) / 630.1895   # +1.8%  pt_6 < 39 and sum_pt_top3 < 720
        - 0.0172119 * max(0.0, 39.0 - Q.pt_6) / 4.348597   # -1.7%  pt_6 < 39
        - 0.0165542 * max(0.0, 5.8e-05 - Q.lam2) / 1.903726e-05   # -1.7%  lam2 < 5.8e-05
        + 0.01608763 * max(0.0, 32.0 - Q.pt_6) / 1.850071   # +1.6%  pt_6 < 32
        + 0.01502634 * max(0.0, Q.centroid_offset - 0.01) * max(0.0, 0.015 - Q.mean_phi2) / 8.869501e-05   # +1.5%  centroid_offset > 0.01 and mean_phi2 < 0.015
        + 0.01190171 * max(0.0, Q.sum_pt - 850.0) / 22.05114   # +1.2%  sum_pt > 850
        - 0.01118337 * max(0.0, 39.0 - Q.pt_6) * max(0.0, 4.7 - Q.D2_b2) / 12.16188   # -1.1%  pt_6 < 39 and D2_b2 < 4.7
        + 0.01038171 * max(0.0, 0.0038 - Q.lam2) * max(0.0, Q.n_dr_0p1_0p2 - 0.55) / 0.003059461   # +1.0%  lam2 < 0.0038 and n_dr_0p1_0p2 > 0.55
        - 0.009899497 * max(0.0, 0.022 - Q.z_6) / 0.0003306982   # -1.0%  z_6 < 0.022
        + 0.009189886 * max(0.0, 20.0 - Q.pt_6) / 0.2786197   # +0.9%  pt_6 < 20
        - 0.009140735 * max(0.0, Q.sj3_dr_max - 0.043) * max(0.0, 0.015 - Q.dr_min_012) / 0.001021818   # -0.9%  sj3_dr_max > 0.043 and dr_min_012 < 0.015
        + 0.008366377 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 45.0 - Q.pt_7) / 0.8857705   # +0.8%  log_sum_pt < 6.5 and pt_7 < 45
        - 0.008334987 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, 0.13 - Q.planar_flow) / 0.0002910689   # -0.8%  centroid_offset < 0.019 and planar_flow < 0.13
        + 0.007034914 * max(0.0, Q.centroid_offset - 0.0096) * max(0.0, Q.N2 - 0.16) / 0.0006703244   # +0.7%  centroid_offset > 0.0096 and N2 > 0.16
        - 0.006173239 * max(0.0, Q.tau1 - 0.17) / 0.001500925   # -0.6%  tau1 > 0.17
        + 0.005405176 * max(0.0, Q.girth - 0.15) / 0.0008193726   # +0.5%  girth > 0.15
        - 0.003958578 * max(0.0, Q.sum_pt - 770.0) * max(0.0, Q.D2_b2 - 0.74) / 129.8539   # -0.4%  sum_pt > 770 and D2_b2 > 0.74
        - 0.002259402 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, Q.D3 - 0.72) / 0.004225278   # -0.2%  centroid_offset > 0.015 and D3 > 0.72
        + 0.00214794 * max(0.0, Q.centroid_offset - 0.027) * max(0.0, Q.pt_2 - 74.0) / 0.02528242   # +0.2%  centroid_offset > 0.027 and pt_2 > 74
        - 0.001723333 * max(0.0, Q.sj2_dr - 0.17) * max(0.0, Q.mean_eta - 0.027) / 6.980519e-05   # -0.2%  sj2_dr > 0.17 and mean_eta > 0.027
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 27.79;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.78685 * (0.1295577
        - 0.1127576 * max(0.0, Q.LHA - 0.35) / 0.008422526   # -11.3%  LHA > 0.35
        + 0.0814927 * max(0.0, Q.girth - 0.1) / 0.005689511   # +8.1%  girth > 0.1
        - 0.07076623 * max(0.0, 0.42 - Q.LHA) / 0.178761   # -7.1%  LHA < 0.42
        - 0.07051618 * max(0.0, Q.girth - 0.087) / 0.007900897   # -7.1%  girth > 0.087
        - 0.06166668 * max(0.0, 0.0047 - Q.lam1) / 0.001779359   # -6.2%  lam1 < 0.0047
        - 0.04560201 * max(0.0, Q.lam1 - 0.0073) / 0.002080684   # -4.6%  lam1 > 0.0073
        + 0.04383921 * max(0.0, Q.girth - 0.045) * max(0.0, 2.2 - Q.D2) / 0.03421779   # +4.4%  girth > 0.045 and D2 < 2.2
        + 0.04098986 * max(0.0, 0.16 - Q.sj3_dr_max) / 0.04449137   # +4.1%  sj3_dr_max < 0.16
        + 0.04000509 * max(0.0, 0.087 - Q.tau1) / 0.03656629   # +4.0%  tau1 < 0.087
        + 0.03622798 * max(0.0, Q.tau1 - 0.1) / 0.009496805   # +3.6%  tau1 > 0.1
        + 0.03406946 * max(0.0, Q.sj3_dr_max - 0.073) / 0.1081924   # +3.4%  sj3_dr_max > 0.073
        - 0.03137117 * max(0.0, Q.lam1 - 0.0024) * max(0.0, Q.D2 - 0.26) / 0.002617735   # -3.1%  lam1 > 0.0024 and D2 > 0.26
        + 0.03122909 * max(0.0, Q.lam1 - 0.0025) * max(0.0, 0.2 - Q.planar_flow) / 0.0003856703   # +3.1%  lam1 > 0.0025 and planar_flow < 0.2
        - 0.03116201 * max(0.0, Q.lam1 - 0.016) / 0.0007276421   # -3.1%  lam1 > 0.016
        + 0.03092828 * max(0.0, Q.lam1 - 0.012) / 0.00122947   # +3.1%  lam1 > 0.012
        - 0.02462071 * max(0.0, Q.girth - 0.081) * max(0.0, 2.8 - Q.D2) / 0.01795622   # -2.5%  girth > 0.081 and D2 < 2.8
        - 0.02353973 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, Q.lam1 - 0.0083) / 0.0001143523   # -2.4%  planar_flow < 0.18 and lam1 > 0.0083
        + 0.01879918 * max(0.0, Q.lam1 - 0.0077) * max(0.0, 93.0 - Q.pt_4) / 0.08344567   # +1.9%  lam1 > 0.0077 and pt_4 < 93
        - 0.01843464 * max(0.0, 0.0035 - Q.lam1) / 0.001188493   # -1.8%  lam1 < 0.0035
        - 0.0178602 * max(0.0, 0.034 - Q.girth) / 0.006634741   # -1.8%  girth < 0.034
        - 0.01713494 * max(0.0, Q.lam1 - 0.017) * max(0.0, 3.9 - Q.D2) / 0.001959366   # -1.7%  lam1 > 0.017 and D2 < 3.9
        + 0.01636949 * max(0.0, Q.N2 - 0.13) / 0.1043249   # +1.6%  N2 > 0.13
        - 0.01575057 * max(0.0, 1.5 - Q.D2) / 0.4434231   # -1.6%  D2 < 1.5
        + 0.01408699 * max(0.0, 1e-05 - Q.e3) / 3.494939e-06   # +1.4%  e3 < 1e-05
        + 0.01323326 * max(0.0, Q.tau1 - 0.062) / 0.02269819   # +1.3%  tau1 > 0.062
        + 0.01131609 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.5) / 0.01004596   # +1.1%  planar_flow < 0.2 and log_sum_pt > 6.5
        - 0.01094067 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, 0.24 - Q.max_dr) / 0.008261055   # -1.1%  planar_flow < 0.21 and max_dr < 0.24
        - 0.007404876 * max(0.0, 7.2e-05 - Q.e3) * max(0.0, 45.0 - Q.pt_7) / 0.0005845403   # -0.7%  e3 < 7.2e-05 and pt_7 < 45
        - 0.006859412 * max(0.0, 0.025 - Q.centroid_offset) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.01024739   # -0.7%  centroid_offset < 0.025 and n_dr_0p2_0p4 < 1
        - 0.006566596 * max(0.0, Q.lam1 - 0.0025) * max(0.0, 0.19 - Q.sj3_dr_max) / 1.45972e-05   # -0.7%  lam1 > 0.0025 and sj3_dr_max < 0.19
        - 0.005251892 * max(0.0, 0.24 - Q.planar_flow) * max(0.0, 56.0 - Q.pt_6) / 1.560786   # -0.5%  planar_flow < 0.24 and pt_6 < 56
        - 0.004497043 * max(0.0, 0.046 - Q.max_dr) / 0.004998346   # -0.4%  max_dr < 0.046
        - 0.001936373 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, 29.0 - Q.pt_7) / 0.1401191   # -0.2%  planar_flow < 0.21 and pt_7 < 29
        + 0.001677084 * max(0.0, 0.13 - Q.sj3_dr_max) * max(0.0, 1.4 - Q.D2) / 0.002377596   # +0.2%  sj3_dr_max < 0.13 and D2 < 1.4
        + 0.001096669 * max(0.0, Q.girth - 0.084) * max(0.0, 0.088 - Q.sj2_zsoft) / 3.252185e-06   # +0.1%  girth > 0.084 and sj2_zsoft < 0.088
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 10.35;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.34727 * (-0.103409
        + 0.1132218 * max(0.0, 0.0002 - Q.lam2) / 0.0001159936   # +11.3%  lam2 < 0.0002
        + 0.1006436 * max(0.0, 0.0055 - Q.lam1) * max(0.0, 0.058 - Q.z_dr_0p2_0p4) / 0.0001265354   # +10.1%  lam1 < 0.0055 and z_dr_0p2_0p4 < 0.058
        + 0.09408825 * max(0.0, 0.2 - Q.sj3_dr_max) / 0.06668192   # +9.4%  sj3_dr_max < 0.2
        - 0.09216185 * max(0.0, 0.11 - Q.sj3_dr_max) / 0.02502948   # -9.2%  sj3_dr_max < 0.11
        - 0.08461733 * max(0.0, 0.19 - Q.LHA) * max(0.0, 0.053 - Q.z_dr_0p2_0p4) / 0.001196118   # -8.5%  LHA < 0.19 and z_dr_0p2_0p4 < 0.053
        + 0.06953465 * max(0.0, 0.1 - Q.sj3_dr_max) * max(0.0, 0.0017 - Q.girth2_top2) / 3.023082e-05   # +7.0%  sj3_dr_max < 0.1 and girth2_top2 < 0.0017
        - 0.06730977 * max(0.0, 0.017 - Q.tau2) / 0.007959681   # -6.7%  tau2 < 0.017
        - 0.06562435 * max(0.0, Q.centroid_offset - 0.0074) / 0.01098758   # -6.6%  centroid_offset > 0.0074
        - 0.06521548 * max(0.0, 0.16 - Q.sj3_dr_max) * max(0.0, 0.0016 - Q.girth2_top2) / 5.441951e-05   # -6.5%  sj3_dr_max < 0.16 and girth2_top2 < 0.0016
        + 0.05992988 * max(0.0, 0.026 - Q.tau1) * max(0.0, 0.032 - Q.centroid_offset) / 0.0001187951   # +6.0%  tau1 < 0.026 and centroid_offset < 0.032
        + 0.03992563 * max(0.0, 0.0069 - Q.lam1) * max(0.0, 0.025 - Q.centroid_offset) / 4.529836e-05   # +4.0%  lam1 < 0.0069 and centroid_offset < 0.025
        - 0.0367695 * max(0.0, 0.027 - Q.tau1) / 0.005628162   # -3.7%  tau1 < 0.027
        - 0.02994094 * max(0.0, Q.log_sum_pt - 6.7) / 0.03560998   # -3.0%  log_sum_pt > 6.7
        - 0.02647025 * max(0.0, 0.11 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.017) / 7.02294e-05   # -2.6%  sj3_dr_max < 0.11 and centroid_offset > 0.017
        - 0.01862263 * max(0.0, 0.0044 - Q.lam1) * max(0.0, 39.0 - Q.pt_7) / 0.01396328   # -1.9%  lam1 < 0.0044 and pt_7 < 39
        + 0.01420976 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.0029 - Q.girth2_top5) / 6.968349e-05   # +1.4%  log_sum_pt > 6.7 and girth2_top5 < 0.0029
        + 0.01415991 * max(0.0, 0.044 - Q.z_6) / 0.0034153   # +1.4%  z_6 < 0.044
        - 0.007554453 * max(0.0, 0.0049 - Q.lam1) * max(0.0, 6.5 - Q.log_sum_pt) / 8.262995e-05   # -0.8%  lam1 < 0.0049 and log_sum_pt < 6.5
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 19.17;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.17484 * (-0.2758823
        + 0.2043374 * max(0.0, 0.006 - Q.lam1) * max(0.0, 0.0011 - Q.lam2) / 2.612092e-06   # +20.4%  lam1 < 0.006 and lam2 < 0.0011
        + 0.08330383 * max(0.0, 990.0 - Q.sum_pt) / 278.2819   # +8.3%  sum_pt < 990
        + 0.08098749 * max(0.0, 0.031 - Q.centroid_offset) / 0.01634655   # +8.1%  centroid_offset < 0.031
        + 0.07205076 * max(0.0, 0.044 - Q.tau1) * max(0.0, 0.032 - Q.centroid_offset) / 0.0002667108   # +7.2%  tau1 < 0.044 and centroid_offset < 0.032
        - 0.06239479 * max(0.0, 0.042 - Q.girth) / 0.00964847   # -6.2%  girth < 0.042
        + 0.06126874 * max(0.0, 0.21 - Q.sj3_dr_max) / 0.07342616   # +6.1%  sj3_dr_max < 0.21
        - 0.05941177 * max(0.0, 0.054 - Q.girth) / 0.01500937   # -5.9%  girth < 0.054
        + 0.05850359 * max(0.0, 0.017 - Q.centroid_offset) * max(0.0, 0.13 - Q.sj3_dr_min) / 0.0006232206   # +5.9%  centroid_offset < 0.017 and sj3_dr_min < 0.13
        - 0.05147521 * max(0.0, 0.14 - Q.sj3_dr_max) / 0.03615491   # -5.1%  sj3_dr_max < 0.14
        - 0.04534745 * max(0.0, 0.0059 - Q.lam1) * max(0.0, 0.087 - Q.sj3_dr_min) / 0.0001725258   # -4.5%  lam1 < 0.0059 and sj3_dr_min < 0.087
        + 0.04526236 * max(0.0, 0.0034 - Q.lam1) / 0.001143476   # +4.5%  lam1 < 0.0034
        + 0.03567201 * max(0.0, 0.025 - Q.centroid_offset) * max(0.0, 0.0069 - Q.lam1) / 4.529836e-05   # +3.6%  centroid_offset < 0.025 and lam1 < 0.0069
        + 0.03348075 * max(0.0, 820.0 - Q.sum_pt) * max(0.0, 2.6e-08 - Q.e4) / 2.498008e-06   # +3.3%  sum_pt < 820 and e4 < 2.6e-08
        - 0.02433664 * max(0.0, 0.027 - Q.girth) / 0.004361226   # -2.4%  girth < 0.027
        - 0.01855266 * max(0.0, 940.0 - Q.sum_pt) * max(0.0, 0.017 - Q.centroid_offset) / 0.9192358   # -1.9%  sum_pt < 940 and centroid_offset < 0.017
        + 0.008267468 * max(0.0, Q.lam2 - 0.0004) / 0.0004044067   # +0.8%  lam2 > 0.0004
        - 0.008176833 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, 0.17 - Q.D2_b2) / 0.0002784893   # -0.8%  centroid_offset < 0.019 and D2_b2 < 0.17
        + 0.007291059 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.057 - Q.z_5) / 0.5418795   # +0.7%  sum_pt < 1000 and z_5 < 0.057
        + 0.006389734 * max(0.0, Q.e3 - 0.00017) / 3.939619e-05   # +0.6%  e3 > 0.00017
        - 0.005730377 * max(0.0, 0.065 - Q.girth) * max(0.0, -9e-05 - Q.mean_phi) / 6.207857e-05   # -0.6%  girth < 0.065 and mean_phi < -9e-05
        + 0.005177641 * max(0.0, 0.00014 - Q.lam1) / 1.2034e-05   # +0.5%  lam1 < 0.00014
        - 0.005076057 * max(0.0, 0.0059 - Q.lam1) * max(0.0, -0.0066 - Q.mean_eta) / 4.055525e-06   # -0.5%  lam1 < 0.0059 and mean_eta < -0.0066
        + 0.004461211 * max(0.0, Q.lam1 - 0.014) / 0.000956857   # +0.4%  lam1 > 0.014
        - 0.004149738 * max(0.0, 0.0059 - Q.lam1) * max(0.0, Q.mean_eta - 0.0092) / 3.014037e-06   # -0.4%  lam1 < 0.0059 and mean_eta > 0.0092
        - 0.003243782 * max(0.0, 0.0024 - Q.centroid_offset) / 0.0001072397   # -0.3%  centroid_offset < 0.0024
        - 0.003094876 * max(0.0, 0.0061 - Q.lam1) * max(0.0, Q.mean_phi - 0.013) / 1.998107e-06   # -0.3%  lam1 < 0.0061 and mean_phi > 0.013
        - 0.001763932 * max(0.0, 4.6e-06 - Q.e3) * max(0.0, Q.n_dr_0p05_0p1 - 2.9) / 7.884178e-08   # -0.2%  e3 < 4.6e-06 and n_dr_0p05_0p1 > 2.9
        + 0.0007918393 * max(0.0, 6.9 - Q.log_sum_pt) * max(0.0, 39.0 - Q.pt_3) / 0.09202057   # +0.1%  log_sum_pt < 6.9 and pt_3 < 39
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 9.701;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.701192 * (0.8081481
        - 0.4201394 * max(0.0, 0.016 - Q.lam1) / 0.01081128   # -42.0%  lam1 < 0.016
        + 0.09008389 * max(0.0, Q.sj2_dr - 0.13) / 0.05566377   # +9.0%  sj2_dr > 0.13
        - 0.08887225 * max(0.0, Q.LHA - 0.2) / 0.06897334   # -8.9%  LHA > 0.2
        - 0.06198269 * max(0.0, 0.0036 - Q.lam1) * max(0.0, 0.27 - Q.dr_7) / 0.0002947579   # -6.2%  lam1 < 0.0036 and dr_7 < 0.27
        - 0.06166541 * max(0.0, Q.sj2_dr - 0.16) / 0.03884597   # -6.2%  sj2_dr > 0.16
        - 0.04374168 * max(0.0, 0.068 - Q.z_7) / 0.01869368   # -4.4%  z_7 < 0.068
        - 0.04219768 * max(0.0, Q.sj2_dr - 0.17) * max(0.0, 0.7 - Q.planar_flow) / 0.01644048   # -4.2%  sj2_dr > 0.17 and planar_flow < 0.7
        + 0.02655192 * max(0.0, Q.lam1 - 0.016) / 0.0007276421   # +2.7%  lam1 > 0.016
        + 0.02496372 * max(0.0, Q.lam2 - 0.00016) / 0.0004552215   # +2.5%  lam2 > 0.00016
        - 0.02215792 * max(0.0, Q.lam1 - 0.0058) * max(0.0, 0.51 - Q.sj3_pairmin_over_m) / 0.000738688   # -2.2%  lam1 > 0.0058 and sj3_pairmin_over_m < 0.51
        - 0.01914586 * max(0.0, Q.pt_7 - 20.0) * max(0.0, 0.28 - Q.D2_b2) / 1.365718   # -1.9%  pt_7 > 20 and D2_b2 < 0.28
        + 0.0172312 * max(0.0, 0.26 - Q.D2_b2) / 0.07174387   # +1.7%  D2_b2 < 0.26
        - 0.01717879 * max(0.0, Q.log_sum_pt - 6.7) / 0.03560998   # -1.7%  log_sum_pt > 6.7
        - 0.01509114 * max(0.0, Q.LHA - 0.32) * max(0.0, 0.82 - Q.planar_flow) / 0.006506757   # -1.5%  LHA > 0.32 and planar_flow < 0.82
        + 0.01083582 * max(0.0, Q.n_dr_0p2_0p4 - 0.99) / 0.1548165   # +1.1%  n_dr_0p2_0p4 > 0.99
        - 0.008489858 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 4.6 - Q.D2_b2) / 0.568012   # -0.8%  n_dr_0p2_0p4 > 1 and D2_b2 < 4.6
        + 0.008251174 * max(0.0, Q.centroid_offset - 0.022) / 0.004398144   # +0.8%  centroid_offset > 0.022
        - 0.007549516 * max(0.0, Q.lam2 - -2.1e-05) * max(0.0, 6.5 - Q.log_sum_pt) / 0.0001222693   # -0.8%  lam2 > -2.1e-05 and log_sum_pt < 6.5
        - 0.006232905 * max(0.0, 48.0 - Q.pt_6) * max(0.0, 6.5 - Q.log_sum_pt) / 0.8687731   # -0.6%  pt_6 < 48 and log_sum_pt < 6.5
        + 0.004489352 * max(0.0, Q.sj2_dr - 0.3) / 0.004613566   # +0.4%  sj2_dr > 0.3
        + 0.00314781 * max(0.0, Q.sum_pt - 990.0) / 4.325426   # +0.3%  sum_pt > 990
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 14.91;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.91162 * (-0.005706959
        + 0.2241267 * max(0.0, 0.012 - Q.lam1) / 0.00731311   # +22.4%  lam1 < 0.012
        - 0.104657 * max(0.0, 0.072 - Q.girth) / 0.02521171   # -10.5%  girth < 0.072
        + 0.0972563 * max(0.0, 0.016 - Q.lam1) * max(0.0, 0.0011 - Q.lam2) / 1.066359e-05   # +9.7%  lam1 < 0.016 and lam2 < 0.0011
        + 0.08525662 * max(0.0, 0.26 - Q.planar_flow) / 0.1135102   # +8.5%  planar_flow < 0.26
        - 0.06849674 * max(0.0, Q.sj3_dr_max - 0.18) / 0.04273628   # -6.8%  sj3_dr_max > 0.18
        + 0.06515477 * max(0.0, 0.072 - Q.tau1) / 0.02698786   # +6.5%  tau1 < 0.072
        - 0.04963666 * max(0.0, 0.06 - Q.girth) / 0.01809689   # -5.0%  girth < 0.06
        - 0.04540128 * max(0.0, 0.0034 - Q.lam1) * max(0.0, 0.027 - Q.centroid_offset) / 2.039176e-05   # -4.5%  lam1 < 0.0034 and centroid_offset < 0.027
        - 0.03398252 * max(0.0, 46.0 - Q.pt_7) / 12.35938   # -3.4%  pt_7 < 46
        + 0.03285965 * max(0.0, 0.018 - Q.centroid_offset) / 0.006472794   # +3.3%  centroid_offset < 0.018
        + 0.02489174 * max(0.0, Q.sj2_dr - 0.12) / 0.06175976   # +2.5%  sj2_dr > 0.12
        - 0.02354591 * max(0.0, 7.6e-06 - Q.e3) / 2.490125e-06   # -2.4%  e3 < 7.6e-06
        - 0.02246301 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.016) / 2.342376e-05   # -2.2%  lam1 < 0.012 and centroid_offset > 0.016
        - 0.01545484 * max(0.0, 0.26 - Q.planar_flow) * max(0.0, 840.0 - Q.sum_pt) / 15.16162   # -1.5%  planar_flow < 0.26 and sum_pt < 840
        - 0.01370946 * max(0.0, 0.26 - Q.planar_flow) * max(0.0, 4.0 - Q.D3) / 0.2499148   # -1.4%  planar_flow < 0.26 and D3 < 4
        + 0.01256754 * max(0.0, Q.sj3_dr_max - 0.26) / 0.01926026   # +1.3%  sj3_dr_max > 0.26
        - 0.01255954 * max(0.0, 0.0021 - Q.mean_phi2) / 0.0009604256   # -1.3%  mean_phi2 < 0.0021
        - 0.01135203 * max(0.0, Q.centroid_offset - 0.024) / 0.003882503   # -1.1%  centroid_offset > 0.024
        - 0.01102597 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, 940.0 - Q.sum_pt) / 1.034057   # -1.1%  centroid_offset < 0.018 and sum_pt < 940
        - 0.008573567 * max(0.0, 0.13 - Q.tau21) / 0.01511179   # -0.9%  tau21 < 0.13
        - 0.008254039 * max(0.0, 0.0034 - Q.lam1) * max(0.0, 0.26 - Q.planar_flow) / 4.349155e-05   # -0.8%  lam1 < 0.0034 and planar_flow < 0.26
        + 0.007057195 * max(0.0, Q.sj2_dr - 0.27) / 0.008286158   # +0.7%  sj2_dr > 0.27
        - 0.006742538 * max(0.0, 0.0056 - Q.lam1) * max(0.0, 0.77 - Q.D2_b2) / 0.0002458243   # -0.7%  lam1 < 0.0056 and D2_b2 < 0.77
        - 0.003728788 * max(0.0, 0.00055 - Q.lam2) * max(0.0, Q.C3 - 0.044) / 6.691007e-07   # -0.4%  lam2 < 0.00055 and C3 > 0.044
        + 0.003174845 * max(0.0, 7.6e-06 - Q.e3) * max(0.0, 0.78 - Q.D2_b2) / 1.400653e-07   # +0.3%  e3 < 7.6e-06 and D2_b2 < 0.78
        + 0.003040039 * max(0.0, 0.06 - Q.girth) * max(0.0, Q.dr1_7 - 0.2) / 5.122249e-05   # +0.3%  girth < 0.06 and dr1_7 > 0.2
        - 0.00300936 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.dr1_7 - 0.24) / 1.094498e-05   # -0.3%  lam1 < 0.012 and dr1_7 > 0.24
        - 0.001384394 * max(0.0, 0.011 - Q.tau2) * max(0.0, Q.pt1_dr01 - 22.0) / 0.002152613   # -0.1%  tau2 < 0.011 and pt1_dr01 > 22
        - 0.0006370139 * max(0.0, 0.0049 - Q.zdr_0) * max(0.0, Q.dr0_7 - 0.095) / 2.209049e-06   # -0.1%  zdr_0 < 0.0049 and dr0_7 > 0.095
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.4345;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.4345083 * (-1.889492
        + 0.3646357 * max(0.0, Q.girth - 0.13) / 0.002132399   # +36.5%  girth > 0.13
        - 0.1588558 * max(0.0, Q.girth - 0.12) * max(0.0, 61.0 - Q.pt_6) / 0.06909325   # -15.9%  girth > 0.12 and pt_6 < 61
        - 0.1044616 * max(0.0, Q.e2 - 0.067) / 0.001431843   # -10.4%  e2 > 0.067
        + 0.07798687 * max(0.0, Q.girth - 0.12) * max(0.0, Q.C2_b2 - 0.0089) / 3.258264e-05   # +7.8%  girth > 0.12 and C2_b2 > 0.0089
        + 0.07763533 * max(0.0, Q.centroid_offset - 0.039) / 0.001568986   # +7.8%  centroid_offset > 0.039
        + 0.07031821 * max(0.0, Q.n_dr_0p2_0p4 - 1.8) / 0.07834319   # +7.0%  n_dr_0p2_0p4 > 1.8
        + 0.06132227 * max(0.0, Q.sd_rg - 0.32) / 0.001988436   # +6.1%  sd_rg > 0.32
        - 0.0320896 * max(0.0, Q.girth - 0.15) * max(0.0, 570.0 - Q.sum_pt) / 0.06971598   # -3.2%  girth > 0.15 and sum_pt < 570
        + 0.0267776 * max(0.0, Q.girth - 0.15) / 0.0008193726   # +2.7%  girth > 0.15
        + 0.02591713 * max(0.0, Q.girth - 0.1) * max(0.0, Q.log_sum_pt - 6.5) / 8.729619e-05   # +2.6%  girth > 0.1 and log_sum_pt > 6.5
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 19.63;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.62714 * (-0.003663295
        + 0.3956412 * max(0.0, 0.15 - Q.girth) / 0.09211511   # +39.6%  girth < 0.15
        + 0.1127144 * max(0.0, 0.015 - Q.lam1) / 0.009920451   # +11.3%  lam1 < 0.015
        - 0.08635953 * max(0.0, 0.12 - Q.tau1) / 0.06277742   # -8.6%  tau1 < 0.12
        - 0.0733499 * max(0.0, 0.0087 - Q.C2_b2) / 0.006988585   # -7.3%  C2_b2 < 0.0087
        - 0.06870795 * max(0.0, 0.14 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt) / 0.01767418   # -6.9%  girth < 0.14 and log_sum_pt < 6.8
        - 0.04540502 * max(0.0, 8.5e-05 - Q.e3) * max(0.0, 0.036 - Q.centroid_offset) / 1.373144e-06   # -4.5%  e3 < 8.5e-05 and centroid_offset < 0.036
        + 0.04019929 * max(0.0, Q.z_7 - 0.045) / 0.01280839   # +4.0%  z_7 > 0.045
        - 0.02819459 * max(0.0, 0.14 - Q.girth) * max(0.0, 43.0 - Q.pt_7) / 0.8825823   # -2.8%  girth < 0.14 and pt_7 < 43
        - 0.02756517 * max(0.0, Q.pt_7 - 32.0) / 5.861596   # -2.8%  pt_7 > 32
        + 0.02452175 * max(0.0, Q.sum_pt_top5 - 690.0) * max(0.0, 40.0 - Q.pt_7) / 640.0155   # +2.5%  sum_pt_top5 > 690 and pt_7 < 40
        - 0.01576988 * max(0.0, 0.02 - Q.lam1) * max(0.0, 26.0 - Q.pt_7) / 0.02476142   # -1.6%  lam1 < 0.02 and pt_7 < 26
        - 0.01565473 * max(0.0, Q.sj3_dr_max - 0.23) / 0.02603877   # -1.6%  sj3_dr_max > 0.23
        - 0.01518823 * max(0.0, 50.0 - Q.pt_6) / 11.46544   # -1.5%  pt_6 < 50
        - 0.01247914 * max(0.0, Q.z_dr_0_0p05 - 0.61) / 0.1749499   # -1.2%  z_dr_0_0p05 > 0.61
        - 0.01113529 * max(0.0, 0.16 - Q.girth) * max(0.0, 0.00025 - Q.lam2) / 1.80623e-05   # -1.1%  girth < 0.16 and lam2 < 0.00025
        - 0.01110562 * max(0.0, 6.5 - Q.log_sum_pt) / 0.08351396   # -1.1%  log_sum_pt < 6.5
        + 0.007149501 * max(0.0, 0.087 - Q.e2) * max(0.0, 60.0 - Q.pt_4) / 0.487237   # +0.7%  e2 < 0.087 and pt_4 < 60
        - 0.005124185 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, Q.C2_b2 - -0.0047) / 0.0002073672   # -0.5%  log_sum_pt > 6.7 and C2_b2 > -0.0047
        - 0.003734681 * max(0.0, 26.0 - Q.pt_5) / 0.3610891   # -0.4%  pt_5 < 26
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 21.83;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.83006 * (-0.2166738
        + 0.1451933 * max(0.0, Q.girth - 0.026) / 0.03677005   # +14.5%  girth > 0.026
        - 0.1075945 * max(0.0, 0.0065 - Q.lam1) / 0.002885496   # -10.8%  lam1 < 0.0065
        + 0.07923795 * max(0.0, 0.095 - Q.tau1) / 0.04229264   # +7.9%  tau1 < 0.095
        - 0.0763667 * max(0.0, Q.girth - 0.087) / 0.007900897   # -7.6%  girth > 0.087
        + 0.06633037 * max(0.0, 0.012 - Q.lam1) / 0.00731311   # +6.6%  lam1 < 0.012
        + 0.05435951 * max(0.0, 0.041 - Q.e2) / 0.01750252   # +5.4%  e2 < 0.041
        + 0.05094837 * max(0.0, 0.17 - Q.sd_rg) / 0.07617847   # +5.1%  sd_rg < 0.17
        - 0.04846685 * max(0.0, Q.centroid_offset - 0.05) / 0.0008015409   # -4.8%  centroid_offset > 0.05
        + 0.04383782 * max(0.0, Q.sd_rg - 0.16) / 0.02798193   # +4.4%  sd_rg > 0.16
        - 0.04296478 * max(0.0, Q.sd_rg - 0.19) / 0.01864659   # -4.3%  sd_rg > 0.19
        - 0.0410157 * max(0.0, Q.z_dr_0p05_0p1 - 0.16) * max(0.0, 1.0 - Q.psi_0p3) / 0.0007399794   # -4.1%  z_dr_0p05_0p1 > 0.16 and psi_0p3 < 1
        + 0.0284007 * max(0.0, Q.girth - 0.068) / 0.01339069   # +2.8%  girth > 0.068
        - 0.02636587 * max(0.0, 8.4e-05 - Q.e3) / 5.849273e-05   # -2.6%  e3 < 8.4e-05
        - 0.02228193 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.158545   # -2.2%  n_dr_0p05_0p1 < 5
        - 0.02069751 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.0073 - Q.lam1) / 5.704895e-05   # -2.1%  planar_flow < 0.11 and lam1 < 0.0073
        + 0.01839558 * max(0.0, 0.0033 - Q.lam2) * max(0.0, Q.sd_rg - 0.16) / 5.720464e-05   # +1.8%  lam2 < 0.0033 and sd_rg > 0.16
        - 0.01598157 * max(0.0, Q.z_dr_0p05_0p1 - 0.16) * max(0.0, 0.21 - Q.z_dr_0p2_0p4) / 0.03723357   # -1.6%  z_dr_0p05_0p1 > 0.16 and z_dr_0p2_0p4 < 0.21
        + 0.01472317 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.016 - Q.lam1) / 0.0003003809   # +1.5%  planar_flow < 0.11 and lam1 < 0.016
        + 0.0145061 * max(0.0, Q.z_dr_0p05_0p1 - 0.17) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.1623943   # +1.5%  z_dr_0p05_0p1 > 0.17 and n_dr_0p2_0p4 < 1
        - 0.01416578 * max(0.0, 0.066 - Q.sd_rg) / 0.02224746   # -1.4%  sd_rg < 0.066
        + 0.01047596 * max(0.0, 0.59 - Q.z_dr_0p05_0p1) * max(0.0, 0.045 - Q.z_dr_0p1_0p2) / 0.01058754   # +1.0%  z_dr_0p05_0p1 < 0.59 and z_dr_0p1_0p2 < 0.045
        - 0.008608923 * max(0.0, 0.0022 - Q.girth2_top3) / 0.0008031336   # -0.9%  girth2_top3 < 0.0022
        - 0.008295991 * max(0.0, 0.016 - Q.lam1) * max(0.0, Q.centroid_offset - 0.027) / 1.414859e-05   # -0.8%  lam1 < 0.016 and centroid_offset > 0.027
        - 0.007417532 * max(0.0, 0.013 - Q.e2) / 0.002562107   # -0.7%  e2 < 0.013
        - 0.006066662 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 1.0 - Q.sd_nremoved) / 0.0262769   # -0.6%  planar_flow < 0.11 and sd_nremoved < 1
        - 0.00588773 * max(0.0, Q.z_dr_0p05_0p1 - 0.16) * max(0.0, 6.7 - Q.log_sum_pt) / 0.04462829   # -0.6%  z_dr_0p05_0p1 > 0.16 and log_sum_pt < 6.7
        + 0.005215071 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.006 - Q.lam1) / 3.153609e-05   # +0.5%  planar_flow < 0.11 and lam1 < 0.006
        + 0.004573787 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.29 - Q.sj2_zsoft) / 0.00261377   # +0.5%  planar_flow < 0.11 and sj2_zsoft < 0.29
        - 0.0036479 * max(0.0, 0.0034 - Q.lam2) * max(0.0, Q.z_dr_0p2_0p4 - 0.00043) / 5.104735e-05   # -0.4%  lam2 < 0.0034 and z_dr_0p2_0p4 > 0.00043
        + 0.003097182 * max(0.0, 0.0035 - Q.lam2) * max(0.0, Q.centroid_offset - 0.027) / 6.999136e-06   # +0.3%  lam2 < 0.0035 and centroid_offset > 0.027
        - 0.002037494 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.016 - Q.centroid_offset) / 0.000165348   # -0.2%  planar_flow < 0.11 and centroid_offset < 0.016
        + 0.001503516 * max(0.0, 0.0065 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.2) / 1.180642e-05   # +0.2%  lam1 < 0.0065 and sj3_dr23 > 0.2
        + 0.0007091255 * max(0.0, 0.016 - Q.lam1) * max(0.0, 0.42 - Q.D2) / 0.0001323098   # +0.1%  lam1 < 0.016 and D2 < 0.42
        + 0.0006290406 * max(0.0, 0.043 - Q.e2) * max(0.0, -0.027 - Q.mean_eta) / 5.996503e-06   # +0.1%  e2 < 0.043 and mean_eta < -0.027
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 25.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.63509 * (-0.02746235
        - 0.1386713 * max(0.0, 0.072 - Q.girth) / 0.02521171   # -13.9%  girth < 0.072
        - 0.1280182 * max(0.0, 0.086 - Q.girth) / 0.03540191   # -12.8%  girth < 0.086
        + 0.1014341 * max(0.0, 0.017 - Q.lam1) / 0.01171294   # +10.1%  lam1 < 0.017
        - 0.08560307 * max(0.0, 0.0074 - Q.lam1) / 0.003539424   # -8.6%  lam1 < 0.0074
        + 0.06670529 * max(0.0, 0.041 - Q.e2) / 0.01750252   # +6.7%  e2 < 0.041
        + 0.05112363 * max(0.0, Q.tau1 - 0.1) / 0.009496805   # +5.1%  tau1 > 0.1
        - 0.04959169 * max(0.0, Q.tau1 - 0.096) / 0.01042039   # -5.0%  tau1 > 0.096
        + 0.04308646 * max(0.0, 0.0054 - Q.lam1) / 0.002174262   # +4.3%  lam1 < 0.0054
        + 0.04112489 * max(0.0, 0.058 - Q.tau1) / 0.01920292   # +4.1%  tau1 < 0.058
        + 0.0402644 * max(0.0, 0.0021 - Q.girth2_top3) / 0.0007534173   # +4.0%  girth2_top3 < 0.0021
        + 0.03995073 * max(0.0, 0.0076 - Q.girth2_top2) / 0.004511633   # +4.0%  girth2_top2 < 0.0076
        + 0.02329678 * max(0.0, Q.eccentricity - 0.94) * max(0.0, Q.sum_pt_top5 - 410.0) / 4.895205   # +2.3%  eccentricity > 0.94 and sum_pt_top5 > 410
        - 0.01926403 * max(0.0, 0.23 - Q.N2) * max(0.0, Q.lam1 - 0.0087) / 0.0001201546   # -1.9%  N2 < 0.23 and lam1 > 0.0087
        - 0.01712344 * max(0.0, 0.033 - Q.girth) / 0.006288841   # -1.7%  girth < 0.033
        + 0.01662573 * max(0.0, 0.22 - Q.N2) * max(0.0, Q.LHA - 0.29) / 0.001590306   # +1.7%  N2 < 0.22 and LHA > 0.29
        - 0.01500455 * max(0.0, 0.0034 - Q.lam2) * max(0.0, 0.019 - Q.zdr_1) / 3.344722e-05   # -1.5%  lam2 < 0.0034 and zdr_1 < 0.019
        + 0.01316738 * max(0.0, Q.sj3_dr13 - 0.18) / 0.02557175   # +1.3%  sj3_dr13 > 0.18
        - 0.01124477 * max(0.0, 0.22 - Q.N2) * max(0.0, 0.19 - Q.sj2_dr) / 0.0008307224   # -1.1%  N2 < 0.22 and sj2_dr < 0.19
        - 0.00960674 * max(0.0, 0.0021 - Q.girth2_top3) * max(0.0, 0.21 - Q.planar_flow) / 2.801703e-05   # -1.0%  girth2_top3 < 0.0021 and planar_flow < 0.21
        - 0.009556328 * max(0.0, 0.0082 - Q.lam1) * max(0.0, 0.89 - Q.D2) / 0.0002187298   # -1.0%  lam1 < 0.0082 and D2 < 0.89
        + 0.009335862 * max(0.0, 0.0084 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.2) / 2.195649e-05   # +0.9%  lam1 < 0.0084 and sj3_dr23 > 0.2
        - 0.008682738 * max(0.0, Q.sj3_dr13 - 0.25) / 0.01054895   # -0.9%  sj3_dr13 > 0.25
        - 0.008232586 * max(0.0, Q.tau1 - 0.13) / 0.00490798   # -0.8%  tau1 > 0.13
        + 0.008160461 * max(0.0, Q.tau1 - 0.13) * max(0.0, 0.54 - Q.D2_b2) / 0.0009423162   # +0.8%  tau1 > 0.13 and D2_b2 < 0.54
        - 0.007548737 * max(0.0, Q.eccentricity - 0.95) * max(0.0, Q.mean_phi - -0.02) / 0.0004014784   # -0.8%  eccentricity > 0.95 and mean_phi > -0.02
        + 0.006803661 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.89 - Q.D2) / 8.345095e-05   # +0.7%  lam1 < 0.0065 and D2 < 0.89
        - 0.005676471 * max(0.0, 0.018 - Q.lam1) * max(0.0, Q.centroid_offset - 0.038) / 7.239645e-06   # -0.6%  lam1 < 0.018 and centroid_offset > 0.038
        - 0.005042938 * max(0.0, 0.19 - Q.N2) * max(0.0, Q.LHA - 0.38) / 0.0001890003   # -0.5%  N2 < 0.19 and LHA > 0.38
        - 0.005040244 * max(0.0, 0.006 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.2) / 9.570899e-06   # -0.5%  lam1 < 0.006 and sj3_dr23 > 0.2
        - 0.004345502 * max(0.0, 0.22 - Q.D3) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 0.0005086637   # -0.4%  D3 < 0.22 and n_dr_0p1_0p2 > 1
        - 0.002945354 * max(0.0, 0.22 - Q.N2) * max(0.0, Q.LHA - 0.44) / 3.665263e-05   # -0.3%  N2 < 0.22 and LHA > 0.44
        - 0.002302826 * max(0.0, 0.22 - Q.N2) * max(0.0, Q.z_dr_0p2_0p4 - 0.21) / 0.0002771509   # -0.2%  N2 < 0.22 and z_dr_0p2_0p4 > 0.21
        + 0.002009123 * max(0.0, 0.0078 - Q.lam1) * max(0.0, Q.centroid_offset - 0.038) / 8.555489e-07   # +0.2%  lam1 < 0.0078 and centroid_offset > 0.038
        - 0.001503207 * max(0.0, 0.017 - Q.lam1) * max(0.0, Q.C2_b2 - 0.007) / 4.52287e-06   # -0.2%  lam1 < 0.017 and C2_b2 > 0.007
        - 0.0009581439 * max(0.0, 0.24 - Q.N2) * max(0.0, 0.023 - Q.z_7) / 1.729726e-05   # -0.1%  N2 < 0.24 and z_7 < 0.023
        - 0.0009486812 * max(0.0, Q.z_dr_0p05_0p1 - 0.74) * max(0.0, Q.sj3_dr23 - 0.18) / 0.0003597564   # -0.1%  z_dr_0p05_0p1 > 0.74 and sj3_dr23 > 0.18
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9061798319327731, 0.8130035714285714, 2.0795697478991597, 0.9265800420168068, 1.230274369747899, 1.7817172268907564, 1.02766743697479, 1.8810529411764705, 0.28019138655462184, 3.0450264705882355, 1.9609004201680673, 2.091340336134454, 0.08794537815126051, 3.312491806722689, 0.431190231092437, 0.36186197478991594]
T = [2.3610183495273107, 1.385706786699055, 3.2873620158219534, 2.5084463350183825, 2.848829943539916]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +38%, n9 +22%, n5 -14%, n1 +13%, n0 -6%, n6 +5% ...
            + 0.378466 * h[2] / H_AVG[2]
            + 0.2216687 * h[9] / H_AVG[9]
            - 0.1414949 * h[5] / H_AVG[5]
            + 0.1345096 * h[1] / H_AVG[1]
            - 0.05997014 * h[0] / H_AVG[0]
            + 0.04760705 * h[6] / H_AVG[6]
            - 0.01628368 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -18%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5579472 * h[9] / H_AVG[9]
            - 0.1768863 * h[10] / H_AVG[10]
            + 0.09270246 * h[6] / H_AVG[6]
            - 0.08323422 * h[4] / H_AVG[4]
            + 0.06027104 * h[5] / H_AVG[5]
            + 0.01632118 * h[15] / H_AVG[15]
            + 0.01263757 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +13%, n14 -10%, n6 -10%, n0 +9% ...
            + 0.2385659 * h[11] / H_AVG[11]
            - 0.1409306 * h[3] / H_AVG[3]
            + 0.1251704 * h[7] / H_AVG[7]
            - 0.09837452 * h[14] / H_AVG[14]
            - 0.09769112 * h[6] / H_AVG[6]
            + 0.09475662 * h[0] / H_AVG[0]
            - 0.07567773 * h[15] / H_AVG[15]
            + 0.07084999 * h[13] / H_AVG[13]
            - 0.02894633 * h[9] / H_AVG[9]
            - 0.02130822 * h[8] / H_AVG[8]
            - 0.007728495 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -21%, n6 -15%, n13 +7%, n14 +6%, n1 +4% ...
            + 0.3515098 * h[7] / H_AVG[7]
            - 0.2077785 * h[3] / H_AVG[3]
            - 0.1536311 * h[6] / H_AVG[6]
            + 0.07221677 * h[13] / H_AVG[13]
            + 0.06446075 * h[14] / H_AVG[14]
            + 0.0405133 * h[1] / H_AVG[1]
            + 0.03831662 * h[4] / H_AVG[4]
            - 0.03793467 * h[9] / H_AVG[9]
            - 0.02254022 * h[15] / H_AVG[15]
            + 0.01109824 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +26%, n5 -16%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4723693 * h[13] / H_AVG[13]
            + 0.2581192 * h[10] / H_AVG[10]
            - 0.1563552 * h[5] / H_AVG[5]
            + 0.05398156 * h[4] / H_AVG[4]
            + 0.02032808 * h[3] / H_AVG[3]
            + 0.01844121 * h[8] / H_AVG[8]
            - 0.01543535 * h[12] / H_AVG[12]
            + 0.004970132 * h[0] / H_AVG[0]
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
