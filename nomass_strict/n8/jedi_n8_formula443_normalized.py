"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.8%   (on for 86% of jets)
  neuron  9:  12.5%   (on for 71% of jets)
  neuron  7:  10.0%   (on for 58% of jets)
  neuron  3:   8.5%   (on for 25% of jets)
  neuron 10:   8.3%   (on for 76% of jets)
  neuron  6:   7.3%   (on for 32% of jets)
  neuron  2:   7.3%   (on for 92% of jets)
  neuron  5:   6.8%   (on for 71% of jets)
  neuron 11:   6.2%   (on for 75% of jets)
  neuron 14:   3.9%   (on for 28% of jets)
  neuron  0:   3.9%   (on for 43% of jets)
  neuron  1:   3.3%   (on for 53% of jets)
  neuron  4:   3.3%   (on for 64% of jets)
  neuron 15:   2.6%   (on for 27% of jets)
  neuron  8:   1.0%   (on for 28% of jets)
  neuron 12:   0.3%   (on for 6% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.3% (the network: 65.8%); same class as the network for 89.5% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.eta_0                  Δη of particle 0
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
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        pt_2=pt[2],
        pt_5=pt[5],
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        z_top3_slots=sum(pt[:3]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        abseta_0=abs(eta[0]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        eta_0=eta[0],
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
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 19.61;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.61387 * (-0.09053428
        + 0.1240142 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +12.4%  lam1 < 0.012
        - 0.0923092 * max(0.0, 0.08065885 - Q.sum_z_dr) / 0.03128522   # -9.2%  sum_z_dr < 0.08066
        + 0.08596342 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +8.6%  e3 < 0.0005117
        + 0.07736462 * max(0.0, Q.sj3_dr_max - 0.1426152) / 0.06279029   # +7.7%  sj3_dr_max > 0.1426
        - 0.06526719 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -6.5%  lam1 < 0.004184
        + 0.05893049 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +5.9%  tau1 < 0.05357
        - 0.05317231 * max(0.0, 0.01713288 - Q.tau2) / 0.008065871   # -5.3%  tau2 < 0.01713
        + 0.05156596 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +5.2%  lam1 < 0.01643
        - 0.04681253 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # -4.7%  sj3_dr_max > 0.169
        - 0.03852495 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -3.9%  centroid_offset > 0.00679
        - 0.03404803 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # -3.4%  sj3_dr_max > 0.1986
        - 0.03183347 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -3.2%  lam1 < 0.005954
        + 0.03012249 * max(0.0, Q.LHA - 0.1546891) / 0.1006902   # +3.0%  LHA > 0.1547
        - 0.02730893 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -2.7%  LHA > 0.3033
        + 0.02621484 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +2.6%  e3 < 8.148e-05
        + 0.02119283 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, Q.centroid_offset - 0.01096064) / 1.576925   # +2.1%  sum_pt < 763.8 and centroid_offset > 0.01096
        - 0.02089656 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.0005116989 - Q.e3) / 0.03717154   # -2.1%  sum_pt < 763.8 and e3 < 0.0005117
        - 0.01895743 * max(0.0, 763.825 - Q.sum_pt) / 96.68366   # -1.9%  sum_pt < 763.8
        - 0.01531055 * max(0.0, 0.007929074 - Q.sum_z_dr2_top3) / 0.004560262   # -1.5%  sum_z_dr2_top3 < 0.007929
        + 0.01352475 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +1.4%  planar_flow < 0.1484
        + 0.01221396 * max(0.0, 0.004183811 - Q.lam1) * max(0.0, 763.825 - Q.sum_pt) / 0.08579672   # +1.2%  lam1 < 0.004184 and sum_pt < 763.8
        - 0.01173261 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # -1.2%  log_sum_pt > 6.843
        + 0.01042916 * max(0.0, 0.01713288 - Q.tau2) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1) / 0.001574598   # +1.0%  tau2 < 0.01713 and z_dr_0p05_0p1 < 0.2919
        + 0.009538592 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +1.0%  sum_pt_top5 > 791.1
        + 0.006781822 * max(0.0, 0.04505724 - Q.planar_flow) * max(0.0, 0.05932655 - Q.tau21_b2) / 0.0003618271   # +0.7%  planar_flow < 0.04506 and tau21_b2 < 0.05933
        + 0.005436991 * max(0.0, Q.sj3_dr_max - 0.1426152) * max(0.0, Q.D2 - 0.2568137) / 0.08032253   # +0.5%  sj3_dr_max > 0.1426 and D2 > 0.2568
        - 0.00537098 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.centroid_offset - 0.01837778) / 3.28605e-05   # -0.5%  lam1 < 0.01643 and centroid_offset > 0.01838
        + 0.002906224 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.03843804 - Q.dr_2) / 0.0001824973   # +0.3%  log_sum_pt > 6.843 and dr_2 < 0.03844
        - 0.002254875 * max(0.0, 0.004183811 - Q.lam1) * max(0.0, 0.7459513 - Q.D2) / 6.576264e-06   # -0.2%  lam1 < 0.004184 and D2 < 0.746
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 15.11;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.10552 * (0.08452675
        + 0.1241914 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +12.4%  log_sum_pt > 6.378
        - 0.09965494 * max(0.0, 0.06473447 - Q.z_7) / 0.01631636   # -10.0%  z_7 < 0.06473
        - 0.07174556 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -7.2%  lam1 < 0.008376
        + 0.06694215 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # +6.7%  lam1 < 0.005433
        + 0.05976875 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, 0.0005116989 - Q.e3) / 7.903391e-06   # +6.0%  z_7 < 0.06473 and e3 < 0.0005117
        - 0.05775595 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.001130645 - Q.lam2) / 4.482383e-06   # -5.8%  lam1 < 0.008376 and lam2 < 0.001131
        - 0.05554961 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.006756161 - Q.sum_z_dr2_top3) / 0.0004453888   # -5.6%  log_sum_pt > 6.573 and sum_z_dr2_top3 < 0.006756
        - 0.04755692 * max(0.0, 0.06108601 - Q.sum_z_dr) / 0.01868685   # -4.8%  sum_z_dr < 0.06109
        + 0.04146047 * max(0.0, Q.log_sum_pt - 6.572938) / 0.08631512   # +4.1%  log_sum_pt > 6.573
        - 0.04061249 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 0.1550922 - Q.dr_max_012) / 0.0002929109   # -4.1%  lam1 < 0.005433 and dr_max_012 < 0.1551
        - 0.03564612 * max(0.0, 0.04737278 - Q.z_6) / 0.004344387   # -3.6%  z_6 < 0.04737
        + 0.03548886 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +3.5%  lam1 < 0.002464
        + 0.02979597 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +3.0%  pt_7 > 34.53
        + 0.02532096 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.1206357 - Q.dr_max_012) / 0.007392713   # +2.5%  log_sum_pt > 6.573 and dr_max_012 < 0.1206
        + 0.02343327 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # +2.3%  tau1 < 0.1028
        - 0.02164691 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -2.2%  zdr_0 < 0.02118
        - 0.0200625 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.dr_0 - 0.005517012) / 0.0001856778   # -2.0%  lam1 < 0.012 and dr_0 > 0.005517
        - 0.01924259 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # -1.9%  max_dr < 0.1118
        - 0.01692559 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 8.147744e-05 - Q.e3) / 0.0002667627   # -1.7%  pt_7 > 34.53 and e3 < 8.148e-05
        + 0.01664645 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 1.340118e-05 - Q.e3) / 1.490664e-06   # +1.7%  log_sum_pt > 6.378 and e3 < 1.34e-05
        - 0.01464597 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -1.5%  centroid_offset < 0.02077
        + 0.01334888 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +1.3%  sj2_dr < 0.1592
        + 0.01322707 * max(0.0, 0.04737278 - Q.z_6) * max(0.0, 0.004007842 - Q.sum_z_dr2_top2) / 1.318326e-05   # +1.3%  z_6 < 0.04737 and sum_z_dr2_top2 < 0.004008
        - 0.01050319 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, 0.005180665 - Q.sum_z_dr2_top2) / 5.709888e-05   # -1.1%  z_7 < 0.06473 and sum_z_dr2_top2 < 0.005181
        - 0.009838855 * max(0.0, 0.06473447 - Q.z_7) * max(0.0, 0.716559 - Q.D2_b2) / 0.004098393   # -1.0%  z_7 < 0.06473 and D2_b2 < 0.7166
        + 0.00798302 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # +0.8%  sum_pt_top5 > 840
        - 0.006704343 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, Q.tau21_b2 - 0.04019753) / 0.0006219781   # -0.7%  lam1 < 0.005433 and tau21_b2 > 0.0402
        - 0.006554185 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.0095303 - Q.sum_z_dr2_top2) / 0.03873367   # -0.7%  sum_pt > 988.4 and sum_z_dr2_top2 < 0.00953
        + 0.005687937 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # +0.6%  lam1 < 0.008376 and centroid_offset > 0.02077
        - 0.00205907 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.2%  sum_pt > 988.4
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 16.89;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.8855 * (0.3369798
        - 0.1889558 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # -18.9%  z_7 > 0.02321
        - 0.1170032 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -11.7%  pt_7 < 53.44
        + 0.1086014 * max(0.0, Q.pt_7 - 20.125) / 15.07003   # +10.9%  pt_7 > 20.12
        - 0.105944 * Q.pt_7 / 34.64819   # -10.6%  pt_7
        - 0.07874181 * max(0.0, Q.LHA - 0.111565) / 0.1352185   # -7.9%  LHA > 0.1116
        + 0.06180181 * max(0.0, 6.701242 - Q.log_sum_pt) / 0.1935491   # +6.2%  log_sum_pt < 6.701
        + 0.04036293 * max(0.0, Q.z_7 - 0.02807091) / 0.02541111   # +4.0%  z_7 > 0.02807
        + 0.04009919 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # +4.0%  log_sum_pt < 6.606
        - 0.03857485 * max(0.0, Q.log_sum_pt - 6.080494) * max(0.0, 8.836697e-05 - Q.mean_phi2) / 6.819601e-06   # -3.9%  log_sum_pt > 6.08 and mean_phi2 < 8.837e-05
        + 0.03662564 * max(0.0, 8.836697e-05 - Q.mean_phi2) / 1.02303e-05   # +3.7%  mean_phi2 < 8.837e-05
        + 0.03622706 * max(0.0, 0.0586137 - Q.z_7) / 0.01231218   # +3.6%  z_7 < 0.05861
        - 0.03442691 * max(0.0, Q.log_sum_pt - 6.080494) / 0.4685351   # -3.4%  log_sum_pt > 6.08
        + 0.03294441 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # +3.3%  sum_z_dr < 0.1019
        + 0.02460858 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.2507612 - Q.max_dr) / 0.00062784   # +2.5%  lam1 < 0.00733 and max_dr < 0.2508
        - 0.01774644 * max(0.0, 0.0586137 - Q.z_7) * max(0.0, 0.1057739 - Q.abseta_0) / 0.001054575   # -1.8%  z_7 < 0.05861 and abseta_0 < 0.1058
        - 0.01081454 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 62.25 - Q.pt_6) / 0.07984241   # -1.1%  lam1 < 0.00733 and pt_6 < 62.25
        + 0.007209047 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.7%  log_sum_pt > 6.896
        - 0.007176429 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.7%  sum_pt > 988.4
        - 0.004794289 * max(0.0, 23.21641 - Q.pt_7) / 0.9212932   # -0.5%  pt_7 < 23.22
        - 0.003172583 * max(0.0, 6.701242 - Q.log_sum_pt) * max(0.0, 8.836697e-05 - Q.mean_phi2) / 5.122736e-07   # -0.3%  log_sum_pt < 6.701 and mean_phi2 < 8.837e-05
        + 0.002351735 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.1056549 - Q.absphi_0) / 0.0003841548   # +0.2%  log_sum_pt > 6.896 and absphi_0 < 0.1057
        + 0.00181742 * max(0.0, Q.sum_pt - 937.0312) * max(0.0, 8.836697e-05 - Q.mean_phi2) / 0.0002643409   # +0.2%  sum_pt > 937 and mean_phi2 < 8.837e-05
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 10.54;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.54283 * (0.06853217
        - 0.3131171 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # -31.3%  lam2 < 0.003408
        + 0.08179543 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +8.2%  sj2_dr > 0.1683
        - 0.07218717 * max(0.0, Q.sum_z_dr - 0.1019409) / 0.005400003   # -7.2%  sum_z_dr > 0.1019
        - 0.06133553 * max(0.0, Q.lam1 - 0.00595415) / 0.002480365   # -6.1%  lam1 > 0.005954
        + 0.05564004 * max(0.0, Q.sum_z_dr - 0.06663269) / 0.01393206   # +5.6%  sum_z_dr > 0.06663
        + 0.05457404 * max(0.0, Q.sum_z_dr - 0.06663269) * max(0.0, Q.eccentricity - 0.7117266) / 0.0025938   # +5.5%  sum_z_dr > 0.06663 and eccentricity > 0.7117
        + 0.0445881 * max(0.0, Q.sj3_dr_max - 0.213399) / 0.03061279   # +4.5%  sj3_dr_max > 0.2134
        + 0.03854174 * max(0.0, Q.sum_z_dr - 0.1245537) / 0.002632136   # +3.9%  sum_z_dr > 0.1246
        + 0.03806091 * max(0.0, Q.LHA - 0.2931906) / 0.02125042   # +3.8%  LHA > 0.2932
        + 0.03058671 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.eccentricity - 0.9458207) / 0.0008867862   # +3.1%  sj2_dr > 0.1683 and eccentricity > 0.9458
        + 0.02928262 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.09482124 - Q.C2) / 0.001379963   # +2.9%  sj2_dr > 0.1683 and C2 < 0.09482
        - 0.02557841 * max(0.0, Q.LHA - 0.3255822) / 0.01245304   # -2.6%  LHA > 0.3256
        - 0.02295055 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.log_sum_pt - 6.080494) / 0.01252901   # -2.3%  sj2_dr > 0.1683 and log_sum_pt > 6.08
        + 0.02241537 * max(0.0, Q.sd_rg - 0.1876504) / 0.01921774   # +2.2%  sd_rg > 0.1877
        - 0.02171955 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.n_dr_0_0p05 - 2.0) / 0.04103234   # -2.2%  sj2_dr > 0.1683 and n_dr_0_0p05 > 2
        + 0.0154288 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +1.5%  tau1 > 0.1136
        - 0.0147783 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # -1.5%  sd_rg > 0.2788
        - 0.01475413 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -1.5%  lam1 > 0.012
        + 0.009856203 * max(0.0, -0.006779839 - Q.mean_eta) / 0.003173922   # +1.0%  mean_eta < -0.00678
        - 0.009636399 * max(0.0, Q.sj3_dr_max - 0.2623172) * max(0.0, Q.n_dr_0p05_0p1 - 4.0) / 0.004621037   # -1.0%  sj3_dr_max > 0.2623 and n_dr_0p05_0p1 > 4
        + 0.006905104 * max(0.0, Q.sj3_dr_max - 0.213399) * max(0.0, Q.n_dr_0p05_0p1 - 4.0) / 0.007898646   # +0.7%  sj3_dr_max > 0.2134 and n_dr_0p05_0p1 > 4
        - 0.006881721 * max(0.0, Q.max_dr - 0.2507612) / 0.004590338   # -0.7%  max_dr > 0.2508
        - 0.005717577 * max(0.0, Q.sum_z_dr - 0.1245537) * max(0.0, Q.eccentricity - 0.7792127) / 0.0002736432   # -0.6%  sum_z_dr > 0.1246 and eccentricity > 0.7792
        - 0.003668525 * max(0.0, -0.006779839 - Q.mean_eta) * max(0.0, -0.03967285 - Q.eta_0) / 9.048956e-05   # -0.4%  mean_eta < -0.00678 and eta_0 < -0.03967
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 29.27;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.26587 * (0.05641773
        - 0.1852274 * max(0.0, 0.2089872 - Q.sj3_dr_min) * max(0.0, 0.003408389 - Q.lam2) / 0.000543125   # -18.5%  sj3_dr_min < 0.209 and lam2 < 0.003408
        + 0.1772868 * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.166579   # +17.7%  sj3_dr_min < 0.209
        + 0.1473126 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +14.7%  e3 < 0.0001869
        - 0.06667186 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -6.7%  lam1 < 0.008376
        + 0.06124034 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # +6.1%  LHA < 0.3127
        - 0.04563626 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -4.6%  lam1 < 0.006507
        - 0.04007607 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -4.0%  lam2 < 0.0005373
        + 0.03674823 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +3.7%  sj3_dr_max > 0.169
        - 0.03359444 * max(0.0, 763.825 - Q.sum_pt) / 96.68366   # -3.4%  sum_pt < 763.8
        - 0.01999686 * max(0.0, 0.01002369 - Q.sum_z_dr2_top3) / 0.006299101   # -2.0%  sum_z_dr2_top3 < 0.01002
        - 0.01870972 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.9%  sj3_dr_max > 0.2337
        + 0.01705551 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +1.7%  N2 < 0.2233
        - 0.01459309 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, 0.009799324 - Q.zdr_2) / 0.001337409   # -1.5%  log_sum_pt > 6.327 and zdr_2 < 0.009799
        + 0.01356041 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +1.4%  lam1 < 0.002464
        - 0.01338405 * max(0.0, Q.sj2_dr - 0.2179769) / 0.01834468   # -1.3%  sj2_dr > 0.218
        - 0.01227634 * max(0.0, Q.C2 - 0.01867771) / 0.01385506   # -1.2%  C2 > 0.01868
        + 0.01135167 * max(0.0, Q.max_dr - 0.121681) * max(0.0, 0.009032972 - Q.C2_b2) / 0.0001556695   # +1.1%  max_dr > 0.1217 and C2_b2 < 0.009033
        - 0.01097102 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, 7.0 - Q.n_dr_0p05_0p1) / 1.337925   # -1.1%  log_sum_pt > 6.327 and n_dr_0p05_0p1 < 7
        - 0.01003713 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1626038 - Q.abseta_7) / 0.005313035   # -1.0%  N2 < 0.2233 and abseta_7 < 0.1626
        - 0.009783201 * max(0.0, 25.57812 - Q.pt_7) / 1.327389   # -1.0%  pt_7 < 25.58
        - 0.009660196 * max(0.0, Q.log_sum_pt - 6.327379) * max(0.0, 7.998907 - Q.ptdr0_6) / 1.442602   # -1.0%  log_sum_pt > 6.327 and ptdr0_6 < 7.999
        - 0.009633785 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, Q.log_sum_pt - 6.267538) / 0.004076983   # -1.0%  sj3_dr_max > 0.2337 and log_sum_pt > 6.268
        + 0.009118247 * max(0.0, 0.08082334 - Q.dr_0) * max(0.0, Q.centroid_offset - 0.009480685) / 0.000183564   # +0.9%  dr_0 < 0.08082 and centroid_offset > 0.009481
        - 0.005961173 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -0.6%  lam1 < 0.00733
        - 0.005653831 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.2832687 - Q.sj2_zsoft) / 0.002686709   # -0.6%  N2 < 0.2233 and sj2_zsoft < 0.2833
        - 0.004929094 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.107953 - Q.M3) / 3.326739   # -0.5%  sum_pt < 763.8 and M3 < 0.108
        + 0.002214943 * max(0.0, Q.sj3_dr_max - 0.233678) * max(0.0, 0.05966518 - Q.sj2_zsoft) / 6.432356e-05   # +0.2%  sj3_dr_max > 0.2337 and sj2_zsoft < 0.05967
        - 0.001687887 * max(0.0, Q.sj2_dr - 0.2179769) * max(0.0, 0.0006435798 - Q.C2_b2) / 1.186443e-06   # -0.2%  sj2_dr > 0.218 and C2_b2 < 0.0006436
        + 0.001623666 * max(0.0, 25.57812 - Q.pt_7) * max(0.0, Q.eccentricity - 0.9458207) / 0.01996572   # +0.2%  pt_7 < 25.58 and eccentricity > 0.9458
        + 0.001542706 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.planar_flow - 0.08366273) / 0.001421863   # +0.2%  N2 < 0.2233 and planar_flow > 0.08366
        - 0.001247415 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.lam2 - 0.001130645) / 2.416387e-06   # -0.1%  N2 < 0.2233 and lam2 > 0.001131
        - 0.001213982 * max(0.0, Q.sj2_dr - 0.2179769) * max(0.0, Q.lam2 - 0.001130645) / 2.486958e-05   # -0.1%  sj2_dr > 0.218 and lam2 > 0.001131
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 9.616;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.615955 * (0.1401806
        + 0.1053485 * max(0.0, 0.03556091 - Q.e2) * max(0.0, 0.0003061234 - Q.lam2) / 3.480488e-06   # +10.5%  e2 < 0.03556 and lam2 < 0.0003061
        + 0.1049788 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +10.5%  LHA < 0.2161
        - 0.07582376 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # -7.6%  dr_0 < 0.06413
        - 0.06951855 * max(0.0, 0.03556091 - Q.e2) / 0.01371972   # -7.0%  e2 < 0.03556
        - 0.06583081 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -6.6%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.0620313 * max(0.0, 0.03556091 - Q.e2) * max(0.0, 0.0753896 - Q.z_7) / 0.0004290247   # +6.2%  e2 < 0.03556 and z_7 < 0.07539
        - 0.05688291 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.09543973 - Q.z_6) / 0.001514041   # -5.7%  LHA < 0.2161 and z_6 < 0.09544
        - 0.05571659 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # -5.6%  centroid_offset < 0.03776
        + 0.04585026 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +4.6%  lam1 < 0.002464
        + 0.04411363 * max(0.0, 0.002464291 - Q.lam1) * max(0.0, 0.01837778 - Q.centroid_offset) / 8.036566e-06   # +4.4%  lam1 < 0.002464 and centroid_offset < 0.01838
        - 0.03608071 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -3.6%  sj3_dr_max < 0.3012
        + 0.03079019 * max(0.0, 0.06081235 - Q.z_6) * max(0.0, 0.0005116989 - Q.e3) / 4.735652e-06   # +3.1%  z_6 < 0.06081 and e3 < 0.0005117
        - 0.02909614 * max(0.0, 0.06081235 - Q.z_6) / 0.009654642   # -2.9%  z_6 < 0.06081
        - 0.02702542 * max(0.0, 0.0262518 - Q.tau1) / 0.005366404   # -2.7%  tau1 < 0.02625
        - 0.02545373 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -2.5%  sum_pt < 739.5
        + 0.02389788 * max(0.0, 0.0008722282 - Q.lam1) / 0.0001882079   # +2.4%  lam1 < 0.0008722
        + 0.0232474 * max(0.0, 0.06081235 - Q.z_6) * max(0.0, 28.39396 - Q.pt1_dr01) / 0.2343338   # +2.3%  z_6 < 0.06081 and pt1_dr01 < 28.39
        + 0.01686069 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +1.7%  lam1 < 0.00484
        - 0.01562211 * max(0.0, Q.log_sum_pt - 6.804164) * max(0.0, 5.334511e-05 - Q.e3) / 6.116454e-07   # -1.6%  log_sum_pt > 6.804 and e3 < 5.335e-05
        - 0.01484611 * max(0.0, 0.02054282 - Q.sum_z_dr) / 0.002592388   # -1.5%  sum_z_dr < 0.02054
        + 0.01438074 * max(0.0, 0.02054282 - Q.sum_z_dr) * max(0.0, Q.sum_pt_top5 - 716.8828) / 0.2273844   # +1.4%  sum_z_dr < 0.02054 and sum_pt_top5 > 716.9
        - 0.01277925 * max(0.0, Q.n_pt_above_50 - 6.0) / 0.2884353   # -1.3%  n_pt_above_50 > 6
        + 0.01051528 * max(0.0, 7.12483e-05 - Q.lam1) / 3.380156e-06   # +1.1%  lam1 < 7.125e-05
        + 0.009346927 * max(0.0, 0.02886576 - Q.z_6) / 0.0008381782   # +0.9%  z_6 < 0.02887
        + 0.009119166 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +0.9%  z_7 < 0.02807
        - 0.00642431 * max(0.0, Q.z_top5_slots - 0.908903) / 0.002542271   # -0.6%  z_top5_slots > 0.9089
        - 0.00447696 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.4%  log_sum_pt > 6.896
        + 0.00394194 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # +0.4%  z_5 < 0.02818
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 23.56;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.55574 * (0.04277423
        - 0.1078543 * max(0.0, Q.sj3_dr_max - 0.03464708) / 0.1367546   # -10.8%  sj3_dr_max > 0.03465
        - 0.08979741 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # -9.0%  sum_z_dr > 0.08724
        - 0.0890257 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # -8.9%  max_dr < 0.1773
        - 0.08824878 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -8.8%  lam2 < 0.001131
        + 0.05747847 * max(0.0, Q.LHA - 0.251526) / 0.03924325   # +5.7%  LHA > 0.2515
        - 0.05339954 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -5.3%  pt_6 < 39.75
        + 0.05053645 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # +5.1%  C2 < 0.03579
        + 0.04725699 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +4.7%  sj3_dr_max > 0.179
        + 0.04668786 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +4.7%  lam2 < 0.003408
        + 0.04472808 * max(0.0, 39.75 - Q.pt_6) * max(0.0, 711.875 - Q.sum_pt_top3) / 684.1746   # +4.5%  pt_6 < 39.75 and sum_pt_top3 < 711.9
        + 0.04020781 * max(0.0, Q.LHA - 0.3255822) / 0.01245304   # +4.0%  LHA > 0.3256
        + 0.03409533 * max(0.0, 0.04355037 - Q.z_6) / 0.003302426   # +3.4%  z_6 < 0.04355
        + 0.02607948 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, Q.sj2_zsoft - 0.05966518) / 0.005829511   # +2.6%  sj2_dr > 0.1592 and sj2_zsoft > 0.05967
        - 0.02540255 * max(0.0, Q.psi_0p2 - 0.8990266) / 0.08559522   # -2.5%  psi_0p2 > 0.899
        - 0.02464388 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -2.5%  centroid_offset < 0.01838
        + 0.02200759 * max(0.0, Q.sj3_dr_max - 0.1789613) * max(0.0, 1.59265 - Q.D2_b2) / 0.03741161   # +2.2%  sj3_dr_max > 0.179 and D2_b2 < 1.593
        + 0.01876003 * max(0.0, 0.08082334 - Q.dr_0) / 0.0377444   # +1.9%  dr_0 < 0.08082
        + 0.01606226 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.01426135 - Q.mean_phi2) / 9.496308e-05   # +1.6%  centroid_offset > 0.008092 and mean_phi2 < 0.01426
        + 0.01446131 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.001460719   # +1.4%  centroid_offset > 0.008092 and sj3_dr_min < 0.209
        - 0.01332987 * max(0.0, Q.sum_z_dr - 0.1019409) / 0.005400003   # -1.3%  sum_z_dr > 0.1019
        - 0.01310391 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -1.3%  sj2_dr > 0.1592
        + 0.01306377 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +1.3%  sj2_dr > 0.1683
        - 0.01258287 * max(0.0, 39.75 - Q.pt_6) * max(0.0, 0.7861046 - Q.z_top3_slots) / 0.3107099   # -1.3%  pt_6 < 39.75 and z_top3_slots < 0.7861
        + 0.008783975 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_dr_0p05_0p1 - 0.0) / 0.01198963   # +0.9%  centroid_offset < 0.01838 and n_dr_0p05_0p1 > 0
        + 0.008392877 * max(0.0, Q.centroid_offset - 0.02076709) / 0.004751537   # +0.8%  centroid_offset > 0.02077
        - 0.007444659 * max(0.0, 39.75 - Q.pt_6) * max(0.0, 4.721224 - Q.D2_b2) / 13.45505   # -0.7%  pt_6 < 39.75 and D2_b2 < 4.721
        + 0.005597475 * max(0.0, 6.423044 - Q.log_sum_pt) * max(0.0, 38.53125 - Q.pt_7) / 0.3227003   # +0.6%  log_sum_pt < 6.423 and pt_7 < 38.53
        - 0.00481715 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.1115136 - Q.planar_flow) / 0.0002206557   # -0.5%  centroid_offset < 0.01838 and planar_flow < 0.1115
        - 0.004646871 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 5.0 - Q.n_dr_0_0p05) / 0.01797539   # -0.5%  centroid_offset > 0.02077 and n_dr_0_0p05 < 5
        + 0.004510031 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.N2 - 0.198361) / 0.0004989479   # +0.5%  centroid_offset > 0.008092 and N2 > 0.1984
        + 0.004501621 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 0.002094451   # +0.5%  lam2 < 0.003408 and n_dr_0p1_0p2 > 1
        - 0.001335942 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, Q.mean_eta - 0.02644207) / 7.861676e-05   # -0.1%  sj2_dr > 0.1592 and mean_eta > 0.02644
        + 0.001155122 * max(0.0, Q.sum_z_dr - 0.1484084) / 0.0008957407   # +0.1%  sum_z_dr > 0.1484
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 35.52;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.52334 * (0.02319347
        - 0.07929091 * max(0.0, Q.LHA - 0.3467135) / 0.008898884   # -7.9%  LHA > 0.3467
        + 0.0734945 * max(0.0, 0.08865369 - Q.tau1) / 0.03771145   # +7.3%  tau1 < 0.08865
        - 0.06475958 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # -6.5%  lam1 > 0.01643
        - 0.05529434 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -5.5%  e3 < 8.148e-05
        + 0.05352342 * max(0.0, Q.sum_z_dr - 0.0479157) * max(0.0, 2.357246 - Q.D2) / 0.03555018   # +5.4%  sum_z_dr > 0.04792 and D2 < 2.357
        + 0.05057314 * max(0.0, Q.sum_z_dr - 0.1019409) / 0.005400003   # +5.1%  sum_z_dr > 0.1019
        - 0.04062924 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # -4.1%  lam1 < 0.00484
        - 0.03914577 * max(0.0, Q.sum_z_dr - 0.1245537) / 0.002632136   # -3.9%  sum_z_dr > 0.1246
        - 0.03670845 * max(0.0, Q.sum_z_dr - 0.08065885) / 0.009330634   # -3.7%  sum_z_dr > 0.08066
        + 0.03572431 * max(0.0, Q.LHA - 0.1329373) / 0.1175814   # +3.6%  LHA > 0.1329
        + 0.03448649 * max(0.0, 0.1594012 - Q.sj3_dr_max) / 0.04422261   # +3.4%  sj3_dr_max < 0.1594
        - 0.03364795 * max(0.0, Q.sum_z_dr - 0.0479157) / 0.02294968   # -3.4%  sum_z_dr > 0.04792
        + 0.02945781 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # +2.9%  lam1 > 0.012
        - 0.02866522 * max(0.0, 0.02685622 - Q.centroid_offset) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.01150938   # -2.9%  centroid_offset < 0.02686 and n_dr_0p2_0p4 < 1
        + 0.02518049 * max(0.0, 0.02685622 - Q.centroid_offset) / 0.01292674   # +2.5%  centroid_offset < 0.02686
        + 0.02339898 * max(0.0, 0.04990367 - Q.centroid_offset) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.06026392   # +2.3%  centroid_offset < 0.0499 and n_dr_0p2_0p4 < 2
        - 0.02289795 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.lam1 - 0.008375572) / 0.0001274849   # -2.3%  planar_flow < 0.195 and lam1 > 0.008376
        + 0.02252241 * max(0.0, Q.lam1 - 0.002464291) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0003749324   # +2.3%  lam1 > 0.002464 and planar_flow < 0.195
        - 0.02099316 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # -2.1%  sj3_dr_max < 0.2134
        + 0.02027103 * max(0.0, Q.lam1 - 0.01643375) * max(0.0, 3.885568 - Q.D2) / 0.002114372   # +2.0%  lam1 > 0.01643 and D2 < 3.886
        - 0.0200098 * max(0.0, Q.lam1 - 0.002464291) * max(0.0, Q.D2 - 0.2568137) / 0.002599874   # -2.0%  lam1 > 0.002464 and D2 > 0.2568
        + 0.01848326 * max(0.0, 0.1950135 - Q.planar_flow) / 0.07573236   # +1.8%  planar_flow < 0.195
        - 0.017986 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 2.843757 - Q.D2) / 0.004089948   # -1.8%  lam1 > 0.00733 and D2 < 2.844
        - 0.01756114 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # -1.8%  lam1 < 0.003377
        + 0.01609389 * max(0.0, Q.tau1 - 0.06345984) / 0.02203215   # +1.6%  tau1 > 0.06346
        + 0.01387398 * max(0.0, Q.N2 - 0.1333619) / 0.1019708   # +1.4%  N2 > 0.1334
        - 0.01051373 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.001397205   # -1.1%  z_dr_0p1_0p2 < 0.1586 and centroid_offset < 0.02686
        + 0.01015648 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) / 0.09532154   # +1.0%  z_dr_0p1_0p2 < 0.1586
        - 0.009908987 * max(0.0, 0.06108601 - Q.sum_z_dr) / 0.01868685   # -1.0%  sum_z_dr < 0.06109
        - 0.009470243 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 0.2507612 - Q.max_dr) / 0.008139731   # -0.9%  planar_flow < 0.195 and max_dr < 0.2508
        + 0.009207784 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.46415) / 0.01159499   # +0.9%  planar_flow < 0.195 and log_sum_pt > 6.464
        + 0.007718411 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # +0.8%  lam1 > 0.00733
        - 0.007693912 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 43.5 - Q.pt_7) / 0.0006101229   # -0.8%  e3 < 8.148e-05 and pt_7 < 43.5
        - 0.006925043 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 56.53125 - Q.pt_6) / 1.191827   # -0.7%  planar_flow < 0.195 and pt_6 < 56.53
        + 0.006065104 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +0.6%  tau1 > 0.1136
        - 0.005284957 * max(0.0, 0.04656688 - Q.max_dr) / 0.005138022   # -0.5%  max_dr < 0.04657
        - 0.004714665 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 3.916009 - Q.D3) / 0.1558838   # -0.5%  planar_flow < 0.195 and D3 < 3.916
        - 0.004663932 * max(0.0, 0.0006570502 - Q.sum_z_dr2_top5) / 0.0001395474   # -0.5%  sum_z_dr2_top5 < 0.0006571
        + 0.00419422 * max(0.0, 4.192959e-06 - Q.e3) / 1.214837e-06   # +0.4%  e3 < 4.193e-06
        - 0.004075794 * max(0.0, Q.lam1 - 0.002464291) * max(0.0, 0.1879486 - Q.sj3_dr_max) / 1.34726e-05   # -0.4%  lam1 > 0.002464 and sj3_dr_max < 0.1879
        + 0.003509124 * max(0.0, Q.lam1 - 0.002464291) / 0.004201699   # +0.4%  lam1 > 0.002464
        + 0.001224384 * max(0.0, Q.sum_z_dr - 0.08065885) * max(0.0, 0.09384951 - Q.sj2_zsoft) / 5.444145e-06   # +0.1%  sum_z_dr > 0.08066 and sj2_zsoft < 0.09385
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 20.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.78039 * (-0.04856776
        - 0.1396766 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # -14.0%  sj3_dr_max < 0.107
        + 0.1360479 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, 0.0153634 - Q.sum_z_dr2_top3) / 0.0003596312   # +13.6%  sj3_dr_max < 0.107 and sum_z_dr2_top3 < 0.01536
        - 0.08919103 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, 0.0153634 - Q.sum_z_dr2_top3) / 0.0005547224   # -8.9%  sj3_dr_max < 0.1426 and sum_z_dr2_top3 < 0.01536
        + 0.06977284 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.0001209282   # +7.0%  lam1 < 0.005433 and z_dr_0p2_0p4 < 0.05644
        + 0.06305137 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +6.3%  sj3_dr_max < 0.1426
        + 0.05269211 * max(0.0, 0.0262518 - Q.tau1) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0001164763   # +5.3%  tau1 < 0.02625 and centroid_offset < 0.03117
        - 0.05190036 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -5.2%  centroid_offset > 0.00679
        - 0.0457825 * max(0.0, 0.1767241 - Q.LHA) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.00104409   # -4.6%  LHA < 0.1767 and z_dr_0p2_0p4 < 0.05644
        + 0.03212563 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +3.2%  sj3_dr_max < 0.1986
        - 0.02588236 * max(0.0, 0.0262518 - Q.tau1) / 0.005366404   # -2.6%  tau1 < 0.02625
        - 0.02587103 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -2.6%  log_sum_pt > 6.701
        - 0.0214519 * max(0.0, 0.251526 - Q.LHA) / 0.04800428   # -2.1%  LHA < 0.2515
        + 0.02070682 * max(0.0, 0.03360421 - Q.sum_z_dr) * max(0.0, Q.pt_7 - 15.55391) / 0.1111381   # +2.1%  sum_z_dr < 0.0336 and pt_7 > 15.55
        - 0.02038931 * max(0.0, 0.251526 - Q.LHA) * max(0.0, Q.z_7 - 0.01685855) / 0.001243859   # -2.0%  LHA < 0.2515 and z_7 > 0.01686
        - 0.01904238 * max(0.0, 0.1594012 - Q.sj3_dr_max) / 0.04422261   # -1.9%  sj3_dr_max < 0.1594
        - 0.01896885 * max(0.0, 0.03360421 - Q.sum_z_dr) / 0.006496997   # -1.9%  sum_z_dr < 0.0336
        + 0.01834437 * max(0.0, 9.651826e-05 - Q.lam2) / 4.237938e-05   # +1.8%  lam2 < 9.652e-05
        - 0.01817312 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.01627885) / 7.113e-05   # -1.8%  sj3_dr_max < 0.107 and centroid_offset > 0.01628
        + 0.01773773 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 0.03117077 - Q.centroid_offset) / 4.498827e-05   # +1.8%  lam1 < 0.005433 and centroid_offset < 0.03117
        + 0.01698664 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55592   # +1.7%  sum_pt_top5 > 658.1
        + 0.01522177 * max(0.0, 0.1767241 - Q.LHA) / 0.01861764   # +1.5%  LHA < 0.1767
        + 0.01227128 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.5925897 - Q.planar_flow) / 0.0009146444   # +1.2%  lam1 < 0.00733 and planar_flow < 0.5926
        - 0.0105122 * max(0.0, 0.03360421 - Q.sum_z_dr) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0001637288   # -1.1%  sum_z_dr < 0.0336 and centroid_offset < 0.03117
        - 0.009699555 * max(0.0, 0.019014 - Q.tau1) / 0.003025453   # -1.0%  tau1 < 0.01901
        - 0.009232348 * max(0.0, 0.005433361 - Q.lam1) * max(0.0, 38.53125 - Q.pt_7) / 0.01790589   # -0.9%  lam1 < 0.005433 and pt_7 < 38.53
        + 0.009100587 * max(0.0, 0.0002758826 - Q.lam1) / 3.586469e-05   # +0.9%  lam1 < 0.0002759
        + 0.008433439 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.02355416 - Q.centroid_offset) / 4.507403e-05   # +0.8%  lam1 < 0.00733 and centroid_offset < 0.02355
        + 0.007212362 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 0.2752975   # +0.7%  sj3_dr_max < 0.1986 and n_dr_0p05_0p1 < 5
        + 0.006687756 * max(0.0, 0.1594012 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.003343241) / 0.00037024   # +0.7%  sj3_dr_max < 0.1594 and centroid_offset > 0.003343
        - 0.004410719 * max(0.0, 0.019014 - Q.tau1) * max(0.0, Q.pt_7 - 20.125) / 0.03986084   # -0.4%  tau1 < 0.01901 and pt_7 > 20.12
        + 0.003423145 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 33.21875 - Q.pt_7) / 0.3351491   # +0.3%  log_sum_pt > 6.701 and pt_7 < 33.22
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 18.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.24029 * (-0.2927069
        + 0.1668177 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 0.001130645 - Q.lam2) / 2.659012e-06   # +16.7%  lam1 < 0.005954 and lam2 < 0.001131
        + 0.1296704 * max(0.0, 0.03117077 - Q.centroid_offset) / 0.01649145   # +13.0%  centroid_offset < 0.03117
        + 0.06830215 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +6.8%  sum_pt < 988.4
        - 0.06603857 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -6.6%  sj3_dr_max < 0.1426
        + 0.06405899 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, 0.00733008 - Q.lam1) / 4.507403e-05   # +6.4%  centroid_offset < 0.02355 and lam1 < 0.00733
        + 0.05588864 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # +5.6%  lam1 < 0.003377
        + 0.05564512 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +5.6%  sj3_dr_max < 0.2134
        - 0.04777614 * max(0.0, 0.02355416 - Q.centroid_offset) / 0.0103615   # -4.8%  centroid_offset < 0.02355
        - 0.04097515 * max(0.0, 0.0285317 - Q.e2) / 0.009532881   # -4.1%  e2 < 0.02853
        + 0.04051988 * max(0.0, 813.4156 - Q.sum_pt) / 129.0758   # +4.1%  sum_pt < 813.4
        + 0.0372163 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +3.7%  lam2 < 0.0001948
        - 0.0360513 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, 0.08543881 - Q.sj3_dr_min) / 0.0001709388   # -3.6%  lam1 < 0.005954 and sj3_dr_min < 0.08544
        + 0.0357127 * max(0.0, 0.01627885 - Q.centroid_offset) * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.0005664198   # +3.6%  centroid_offset < 0.01628 and sj3_dr_min < 0.1278
        + 0.02654362 * max(0.0, Q.lam2 - 0.000537286) / 0.0003825703   # +2.7%  lam2 > 0.0005373
        - 0.02397103 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -2.4%  sj2_dr < 0.1295
        + 0.02073403 * max(0.0, 0.1027585 - Q.max_dr) / 0.02411847   # +2.1%  max_dr < 0.1028
        - 0.0204593 * max(0.0, 1.960701e-05 - Q.e3) / 8.488392e-06   # -2.0%  e3 < 1.961e-05
        - 0.01167096 * max(0.0, 0.05464922 - Q.sum_z_dr) / 0.01532978   # -1.2%  sum_z_dr < 0.05465
        - 0.00996078 * max(0.0, Q.lam2 - 0.000537286) * max(0.0, 0.9979797 - Q.eccentricity) / 0.000146617   # -1.0%  lam2 > 0.0005373 and eccentricity < 0.998
        + 0.008544045 * max(0.0, 988.4078 - Q.sum_pt) * max(0.0, 0.05753583 - Q.z_5) / 0.5190201   # +0.9%  sum_pt < 988.4 and z_5 < 0.05754
        - 0.008227717 * max(0.0, 0.01627885 - Q.centroid_offset) * max(0.0, 0.1830092 - Q.D2_b2) / 0.0002309693   # -0.8%  centroid_offset < 0.01628 and D2_b2 < 0.183
        + 0.007709549 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +0.8%  lam1 < 0.005954
        - 0.007586786 * max(0.0, 0.04081947 - Q.sum_z_dr) / 0.009176315   # -0.8%  sum_z_dr < 0.04082
        + 0.005134763 * max(0.0, 0.0001413912 - Q.lam1) / 1.223988e-05   # +0.5%  lam1 < 0.0001414
        - 0.003177122 * max(0.0, Q.pt1_dr01 - 22.38578) / 0.8276113   # -0.3%  pt1_dr01 > 22.39
        - 0.001607191 * max(0.0, 4.192959e-06 - Q.e3) * max(0.0, Q.n_dr_0p05_0p1 - 3.0) / 6.634517e-08   # -0.2%  e3 < 4.193e-06 and n_dr_0p05_0p1 > 3
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 13.16;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.15818 * (0.2633951
        + 0.1939284 * Q.lam1 / 0.00591636   # +19.4%  lam1
        - 0.1508686 * max(0.0, 0.1484084 - Q.sum_z_dr) / 0.0905999   # -15.1%  sum_z_dr < 0.1484
        - 0.1081761 * max(0.0, Q.LHA - 0.1967397) / 0.07109091   # -10.8%  LHA > 0.1967
        - 0.07036189 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -7.0%  sj2_dr > 0.1592
        + 0.06692363 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # +6.7%  sj2_dr > 0.1295
        + 0.05877899 * max(0.0, Q.lam2 - 0.0001947983) / 0.0004461515   # +5.9%  lam2 > 0.0001948
        - 0.05001387 * max(0.0, Q.lam1 - 0.00483998) / 0.002931468   # -5.0%  lam1 > 0.00484
        + 0.0473719 * max(0.0, Q.tau1 - 0.04369778) / 0.031989   # +4.7%  tau1 > 0.0437
        - 0.03458111 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # -3.5%  lam1 < 0.003377
        - 0.03186945 * max(0.0, 0.06810151 - Q.z_7) / 0.01877013   # -3.2%  z_7 < 0.0681
        - 0.0316771 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.6947818 - Q.planar_flow) / 0.0166918   # -3.2%  sj2_dr > 0.1683 and planar_flow < 0.6948
        + 0.02523876 * max(0.0, 0.002915531 - Q.sum_z_dr2_top3) / 0.001178811   # +2.5%  sum_z_dr2_top3 < 0.002916
        - 0.01917291 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.planar_flow - 0.06126205) / 0.0002819968   # -1.9%  lam2 > 0.0001948 and planar_flow > 0.06126
        + 0.01915044 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +1.9%  sj2_dr > 0.1683
        - 0.0185542 * max(0.0, Q.LHA - 0.3127275) * max(0.0, 0.6947818 - Q.planar_flow) / 0.005887138   # -1.9%  LHA > 0.3127 and planar_flow < 0.6948
        - 0.01589101 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -1.6%  lam2 > 0.003408
        - 0.01047251 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.eccentricity - 0.4829602) / 8.513077e-05   # -1.0%  lam2 > 0.0001948 and eccentricity > 0.483
        - 0.009216893 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 4.721224 - Q.D2_b2) / 0.5862342   # -0.9%  n_dr_0p2_0p4 > 1 and D2_b2 < 4.721
        + 0.008871064 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.9%  n_dr_0p2_0p4 > 1
        - 0.008595862 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, 6.538559 - Q.log_sum_pt) / 0.0001231869   # -0.9%  lam2 > 0.0001948 and log_sum_pt < 6.539
        + 0.006959245 * max(0.0, Q.centroid_offset - 0.02355416) / 0.003991673   # +0.7%  centroid_offset > 0.02355
        + 0.005637797 * max(0.0, Q.lam2 - 0.003408389) * max(0.0, 111.75 - Q.pt_2) / 0.006488771   # +0.6%  lam2 > 0.003408 and pt_2 < 111.8
        + 0.004619843 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +0.5%  sj2_dr > 0.3004
        + 0.003068406 * max(0.0, Q.lam2 - 0.003408389) * max(0.0, Q.pt2_over_pt0 - 0.1852611) / 6.009766e-05   # +0.3%  lam2 > 0.003408 and pt2_over_pt0 > 0.1853
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 15.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.43617 * (-0.02256768
        + 0.2522733 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +25.2%  lam1 < 0.012
        + 0.09110672 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +9.1%  planar_flow < 0.2534
        - 0.08623755 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # -8.6%  sum_z_dr < 0.0717
        + 0.06961007 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # +7.0%  sj2_dr > 0.1295
        - 0.06703704 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # -6.7%  sj3_dr_max > 0.179
        - 0.0640367 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, 0.02685622 - Q.centroid_offset) / 2.007754e-05   # -6.4%  lam1 < 0.003377 and centroid_offset < 0.02686
        - 0.06116688 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # -6.1%  lam1 < 0.01643
        - 0.0518872 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # -5.2%  sj2_dr > 0.1592
        + 0.04941175 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 0.001130645 - Q.lam2) / 1.136697e-05   # +4.9%  lam1 < 0.01643 and lam2 < 0.001131
        + 0.03477247 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +3.5%  centroid_offset < 0.01838
        - 0.03393131 * max(0.0, 45.75 - Q.pt_7) / 12.14632   # -3.4%  pt_7 < 45.75
        + 0.02924022 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +2.9%  lam1 < 0.008376
        - 0.02384949 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.003652679   # -2.4%  planar_flow < 0.2534 and centroid_offset < 0.0499
        - 0.02121223 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.centroid_offset - 0.01627885) / 2.281746e-05   # -2.1%  lam1 < 0.012 and centroid_offset > 0.01628
        - 0.01841778 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -1.8%  planar_flow < 0.2534 and sum_pt < 840
        + 0.01345906 * max(0.0, Q.sj3_dr_max - 0.1426152) / 0.06279029   # +1.3%  sj3_dr_max > 0.1426
        + 0.01098928 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # +1.1%  sj2_dr > 0.1779
        + 0.009095469 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.9%  sj2_dr > 0.2688
        - 0.006159059 * max(0.0, 0.003377388 - Q.lam1) * max(0.0, 0.2534037 - Q.planar_flow) / 4.073218e-05   # -0.6%  lam1 < 0.003377 and planar_flow < 0.2534
        + 0.006106378 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # +0.6%  sj3_dr_max > 0.2623
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 1.428;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.428358 * (-0.7686308
        + 0.3943269 * max(0.0, Q.sum_z_dr - 0.1245537) / 0.002632136   # +39.4%  sum_z_dr > 0.1246
        - 0.1649313 * max(0.0, Q.LHA - 0.3855647) / 0.004130434   # -16.5%  LHA > 0.3856
        - 0.1415968 * max(0.0, Q.sd_rg - 0.324646) * max(0.0, 6.896095 - Q.log_sum_pt) / 0.001091861   # -14.2%  sd_rg > 0.3246 and log_sum_pt < 6.896
        - 0.08841715 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -8.8%  e2 > 0.06344
        + 0.07964848 * max(0.0, Q.sd_rg - 0.324646) * max(0.0, 6.701242 - Q.log_sum_pt) / 0.0007568975   # +8.0%  sd_rg > 0.3246 and log_sum_pt < 6.701
        + 0.07777465 * max(0.0, Q.sd_rg - 0.324646) / 0.001748965   # +7.8%  sd_rg > 0.3246
        - 0.05330479 * max(0.0, Q.sum_z_dr - 0.1245537) * max(0.0, 62.25 - Q.pt_6) / 0.06207836   # -5.3%  sum_z_dr > 0.1246 and pt_6 < 62.25
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 23.92;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.91592 * (-0.02011601
        + 0.3365827 * max(0.0, 0.1484084 - Q.sum_z_dr) / 0.0905999   # +33.7%  sum_z_dr < 0.1484
        + 0.133764 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +13.4%  lam1 < 0.01643
        - 0.09984079 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -10.0%  e2 < 0.08001
        - 0.07250129 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -7.3%  sum_z_dr < 0.1484 and log_sum_pt < 6.804
        - 0.04197773 * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.09151624   # -4.2%  sj3_dr_min < 0.1278
        + 0.04023658 * max(0.0, Q.z_7 - 0.04624032) / 0.01204993   # +4.0%  z_7 > 0.04624
        - 0.0402187 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 1.399607e-06   # -4.0%  e3 < 8.148e-05 and centroid_offset < 0.03776
        + 0.03341149 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +3.3%  e3 < 8.148e-05
        - 0.03237135 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.0003061234 - Q.lam2) / 2.082734e-05   # -3.2%  sum_z_dr < 0.1484 and lam2 < 0.0003061
        - 0.03110322 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -3.1%  tau1 < 0.1136
        - 0.01775091 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -1.8%  sum_z_dr < 0.1484 and pt_7 < 38.53
        - 0.01682541 * max(0.0, Q.pt_7 - 31.85938) / 5.945918   # -1.7%  pt_7 > 31.86
        + 0.01655528 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 40.04062 - Q.pt_7) / 652.1541   # +1.7%  sum_pt_top5 > 687.4 and pt_7 < 40.04
        - 0.01645997 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -1.6%  log_sum_pt > 6.67
        - 0.01219606 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.z_dr_0_0p05 - 0.3658817) / 0.004662966   # -1.2%  lam1 < 0.01643 and z_dr_0_0p05 > 0.3659
        - 0.01213837 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.2%  sj3_dr_max > 0.2337
        + 0.0113911 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +1.1%  sum_pt_top5 > 687.4
        - 0.009749426 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 25.57812 - Q.pt_7) / 0.01870322   # -1.0%  lam1 < 0.01643 and pt_7 < 25.58
        - 0.006204784 * max(0.0, Q.z_7 - 0.0586137) / 0.00587069   # -0.6%  z_7 > 0.05861
        - 0.005637241 * max(0.0, 27.57812 - Q.pt_6) / 0.9875435   # -0.6%  pt_6 < 27.58
        + 0.004417608 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # +0.4%  sum_pt_top5 > 840
        - 0.003428414 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.3%  sum_pt > 988.4
        - 0.002182748 * max(0.0, Q.z_7 - 0.0586137) * max(0.0, Q.lam2 - 4.64158e-05) / 5.481848e-06   # -0.2%  z_7 > 0.05861 and lam2 > 4.642e-05
        - 0.002180756 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # -0.2%  pt_5 < 24.58
        + 0.0008741627 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.z_6 - 0.03932388) / 0.02599753   # +0.1%  sum_pt > 988.4 and z_6 > 0.03932
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 37.99;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 37.98994 * (-0.1204726
        - 0.1476116 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # -14.8%  lam2 < 0.003408
        + 0.08802191 * max(0.0, Q.sum_z_dr - 0.02689598) / 0.03613842   # +8.8%  sum_z_dr > 0.0269
        - 0.08277568 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -8.3%  lam1 < 0.006507
        - 0.08232295 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # -8.2%  lam1 < 0.002464
        + 0.06525147 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +6.5%  tau1 < 0.09539
        - 0.06031098 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # -6.0%  sum_z_dr > 0.08724
        + 0.05351015 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +5.4%  lam1 < 0.012
        + 0.04269921 * max(0.0, 0.1416226 - Q.tau1) / 0.08190733   # +4.3%  tau1 < 0.1416
        + 0.03436713 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.LHA - 0.1767241) / 0.0002170323   # +3.4%  lam2 < 0.003408 and LHA > 0.1767
        + 0.0341264 * max(0.0, Q.sd_rg - 0.1572969) / 0.02905066   # +3.4%  sd_rg > 0.1573
        + 0.03355209 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +3.4%  e2 < 0.04111
        + 0.03198572 * max(0.0, 0.2330919 - Q.sd_rg) / 0.1256101   # +3.2%  sd_rg < 0.2331
        - 0.02846907 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -2.8%  centroid_offset > 0.0499
        - 0.01787545 * max(0.0, Q.sd_rg - 0.1449048) / 0.03437807   # -1.8%  sd_rg > 0.1449
        - 0.01647589 * max(0.0, Q.sd_rg - 0.1876504) / 0.01921774   # -1.6%  sd_rg > 0.1877
        + 0.01617469 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.1644126   # +1.6%  z_dr_0p05_0p1 > 0.1639 and n_dr_0p2_0p4 < 1
        + 0.01607321 * max(0.0, Q.sum_z_dr - 0.06663269) / 0.01393206   # +1.6%  sum_z_dr > 0.06663
        + 0.01585929 * max(0.0, Q.sum_z_dr - 0.08065885) / 0.009330634   # +1.6%  sum_z_dr > 0.08066
        + 0.01539641 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.sd_rg - 0.1572969) / 6.238901e-05   # +1.5%  lam2 < 0.003408 and sd_rg > 0.1573
        + 0.01470801 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +1.5%  lam1 < 0.01643
        - 0.01410476 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -1.4%  tau1 < 0.1136
        - 0.01263377 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -1.3%  planar_flow < 0.1115 and lam1 < 0.00733
        - 0.009316355 * max(0.0, 0.03556091 - Q.e2) / 0.01371972   # -0.9%  e2 < 0.03556
        - 0.008854046 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 0.20552 - Q.z_dr_0p2_0p4) / 0.03609844   # -0.9%  z_dr_0p05_0p1 > 0.1639 and z_dr_0p2_0p4 < 0.2055
        + 0.007814717 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 0.04510668 - Q.z_dr_0p1_0p2) / 0.01057962   # +0.8%  z_dr_0p05_0p1 < 0.5883 and z_dr_0p1_0p2 < 0.04511
        - 0.00737078 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) / 0.1950558   # -0.7%  z_dr_0p05_0p1 > 0.1639
        - 0.006194621 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02685622) / 1.516228e-05   # -0.6%  lam1 < 0.01643 and centroid_offset > 0.02686
        - 0.005164823 * max(0.0, 0.01655442 - Q.e2) / 0.003887347   # -0.5%  e2 < 0.01655
        + 0.004921391 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00595415 - Q.lam1) / 3.161575e-05   # +0.5%  planar_flow < 0.1115 and lam1 < 0.005954
        - 0.004717715 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -0.5%  LHA > 0.3033
        + 0.004019868 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.2832687 - Q.sj2_zsoft) / 0.002541531   # +0.4%  planar_flow < 0.1115 and sj2_zsoft < 0.2833
        + 0.003684703 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # +0.4%  planar_flow < 0.1115 and lam1 < 0.01643
        + 0.003636532 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.centroid_offset - 0.02685622) / 6.831551e-06   # +0.4%  lam2 < 0.003408 and centroid_offset > 0.02686
        - 0.003546113 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.sd_rg - 0.2037854) / 2.928331e-05   # -0.4%  lam2 < 0.003408 and sd_rg > 0.2038
        - 0.003276075 * max(0.0, Q.z_dr_0p05_0p1 - 0.163898) * max(0.0, 6.670067 - Q.log_sum_pt) / 0.03946963   # -0.3%  z_dr_0p05_0p1 > 0.1639 and log_sum_pt < 6.67
        - 0.001989832 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, Q.z_dr_0p2_0p4 - 0.0) / 5.139675e-05   # -0.2%  lam2 < 0.003408 and z_dr_0p2_0p4 > 0
        + 0.001186626 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.1974628) / 1.23895e-05   # +0.1%  lam1 < 0.006507 and sj3_dr23 > 0.1975
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 22.76;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.75635 * (-0.09646605
        - 0.1305367 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # -13.1%  sum_z_dr < 0.0717
        + 0.1035877 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +10.4%  lam2 < 0.003408
        - 0.08126217 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -8.1%  lam1 < 0.00733
        + 0.07698694 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +7.7%  tau1 < 0.05357
        - 0.07535306 * max(0.0, 0.03360421 - Q.sum_z_dr) / 0.006496997   # -7.5%  sum_z_dr < 0.0336
        + 0.04905519 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +4.9%  lam1 < 0.005954
        - 0.04816909 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -4.8%  lam1 < 0.008376
        + 0.04718872 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +4.7%  lam1 < 0.01643
        + 0.03842548 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # +3.8%  tau1 > 0.1028
        - 0.03809462 * max(0.0, Q.tau1 - 0.09538712) / 0.01057268   # -3.8%  tau1 > 0.09539
        + 0.02863123 * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) / 0.004543329   # +2.9%  sum_z_dr2_top2 < 0.00764
        + 0.02740049 * max(0.0, Q.eccentricity - 0.9458207) * max(0.0, Q.sum_pt_top5 - 367.5938) / 5.10997   # +2.7%  eccentricity > 0.9458 and sum_pt_top5 > 367.6
        - 0.02605129 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -2.6%  lam1 < 0.006507
        - 0.02507308 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.z_dr_0p2_0p4 - 0.20552) / 0.000298346   # -2.5%  N2 < 0.2233 and z_dr_0p2_0p4 > 0.2055
        + 0.02450956 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +2.5%  lam1 < 0.012
        + 0.01785994 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.2931906) / 0.001536651   # +1.8%  N2 < 0.2233 and LHA > 0.2932
        - 0.01728789 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.lam1 - 0.008375572) / 0.0001169809   # -1.7%  N2 < 0.2233 and lam1 > 0.008376
        + 0.01675623 * max(0.0, Q.psi_0p1 - 0.7143804) / 0.1805043   # +1.7%  psi_0p1 > 0.7144
        - 0.0165002 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 0.0002261574   # -1.7%  lam1 < 0.008376 and D2 < 0.8757
        - 0.01477955 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 0.01849752 - Q.zdr_1) / 3.216563e-05   # -1.5%  lam2 < 0.003408 and zdr_1 < 0.0185
        - 0.01438053 * max(0.0, Q.eccentricity - 0.9458207) / 0.02147681   # -1.4%  eccentricity > 0.9458
        + 0.01345378 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +1.3%  tau1 > 0.1136
        - 0.01100708 * max(0.0, Q.tau1 - 0.1416226) / 0.003662393   # -1.1%  tau1 > 0.1416
        + 0.009955682 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # +1.0%  lam1 < 0.006507 and D2 < 0.8757
        - 0.00881585 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1872617 - Q.sj2_dr) / 0.0007896743   # -0.9%  N2 < 0.2233 and sj2_dr < 0.1873
        + 0.007281808 * max(0.0, 0.1515405 - Q.z_dr_0_0p05) / 0.0478325   # +0.7%  z_dr_0_0p05 < 0.1515
        - 0.007156703 * max(0.0, Q.eccentricity - 0.9458207) * max(0.0, Q.mean_phi - -0.01753483) / 0.0004008253   # -0.7%  eccentricity > 0.9458 and mean_phi > -0.01753
        - 0.006507454 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3255822) / 0.0007869707   # -0.7%  N2 < 0.2233 and LHA > 0.3256
        - 0.004629222 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) * max(0.0, 0.1950135 - Q.planar_flow) / 2.579988e-05   # -0.5%  sum_z_dr2_top3 < 0.002152 and planar_flow < 0.195
        + 0.004590921 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +0.5%  N2 < 0.2233
        - 0.003871173 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # -0.4%  z_dr_0p05_0p1 > 0.7509
        + 0.002460087 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.1974628) / 2.285306e-05   # +0.2%  lam1 < 0.008376 and sj3_dr23 > 0.1975
        - 0.002380595 * max(0.0, 0.2213841 - Q.D3) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 0.0005276367   # -0.2%  D3 < 0.2214 and n_dr_0p1_0p2 > 1
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9320628151260504, 0.747260294117647, 2.1016591386554624, 0.9405714285714286, 1.2382197478991597, 1.693045168067227, 0.9814048319327731, 1.8138186974789916, 0.25895063025210086, 3.175146638655462, 2.069841701680672, 2.0436877100840336, 0.07533466386554621, 3.4511512605042016, 0.4319631302521008, 0.3501953781512605]
T = [2.349799846540179, 1.4211116120667013, 3.2552306657037815, 2.468585409007353, 2.9158293690913863]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +38%, n9 +23%, n5 -14%, n1 +12%, n0 -6%, n6 +5% ...
            + 0.3843122 * h[2] / H_AVG[2]
            + 0.2322446 * h[9] / H_AVG[9]
            - 0.1350949 * h[5] / H_AVG[5]
            + 0.1242227 * h[1] / H_AVG[1]
            - 0.06197754 * h[0] / H_AVG[0]
            + 0.04568098 * h[6] / H_AVG[6]
            - 0.01646709 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +57%, n10 -18%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.567295 * h[9] / H_AVG[9]
            - 0.1820619 * h[10] / H_AVG[10]
            + 0.08632369 * h[6] / H_AVG[6]
            - 0.08168472 * h[4] / H_AVG[4]
            + 0.05584466 * h[5] / H_AVG[5]
            + 0.01540147 * h[15] / H_AVG[15]
            + 0.01138856 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +12%, n14 -10%, n0 +10%, n6 -9% ...
            + 0.2354312 * h[11] / H_AVG[11]
            - 0.1444708 * h[3] / H_AVG[3]
            + 0.1218878 * h[7] / H_AVG[7]
            - 0.09952362 * h[14] / H_AVG[14]
            + 0.09842516 * h[0] / H_AVG[0]
            - 0.09421422 * h[6] / H_AVG[6]
            + 0.07454436 * h[13] / H_AVG[13]
            - 0.07396076 * h[15] / H_AVG[15]
            - 0.0304812 * h[9] / H_AVG[9]
            - 0.01988727 * h[8] / H_AVG[8]
            - 0.00717365 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +34%, n3 -21%, n6 -15%, n13 +8%, n14 +7%, n9 -4% ...
            + 0.3444189 * h[7] / H_AVG[7]
            - 0.2143217 * h[3] / H_AVG[3]
            - 0.1490841 * h[6] / H_AVG[6]
            + 0.07645465 * h[13] / H_AVG[13]
            + 0.06561903 * h[14] / H_AVG[14]
            - 0.04019441 * h[9] / H_AVG[9]
            + 0.03918678 * h[4] / H_AVG[4]
            + 0.03783849 * h[1] / H_AVG[1]
            - 0.02216574 * h[15] / H_AVG[15]
            + 0.01071619 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -48%, n10 +27%, n5 -15%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4808341 * h[13] / H_AVG[13]
            + 0.2661989 * h[10] / H_AVG[10]
            - 0.1451598 * h[5] / H_AVG[5]
            + 0.0530818 * h[4] / H_AVG[4]
            + 0.02016089 * h[3] / H_AVG[3]
            + 0.01665161 * h[8] / H_AVG[8]
            - 0.01291822 * h[12] / H_AVG[12]
            + 0.004994627 * h[0] / H_AVG[0]
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
