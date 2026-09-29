"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.2%   (on for 89% of jets)
  neuron  9:  11.7%   (on for 72% of jets)
  neuron  7:  10.2%   (on for 64% of jets)
  neuron  3:   8.6%   (on for 32% of jets)
  neuron 10:   8.1%   (on for 82% of jets)
  neuron  2:   7.2%   (on for 89% of jets)
  neuron  5:   7.1%   (on for 68% of jets)
  neuron  6:   7.0%   (on for 49% of jets)
  neuron 11:   6.5%   (on for 75% of jets)
  neuron 14:   4.6%   (on for 33% of jets)
  neuron  0:   4.2%   (on for 44% of jets)
  neuron  1:   3.3%   (on for 61% of jets)
  neuron  4:   3.2%   (on for 47% of jets)
  neuron 15:   2.7%   (on for 34% of jets)
  neuron  8:   1.0%   (on for 37% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.6% (the network: 65.8%); same class as the network for 87.2% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_7               |Δη| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
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
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_6=pair_mass(0, 6),
        mass_top5=mass_of(5),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        pt_6=pt[6],
        pt_7=pt[7],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        abseta_7=abs(eta[7]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 48.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 48.09102 * (-0.2308123
        + 0.1506558 * max(0.0, 0.0186 - Q.sum_z_dr2) / 0.01293784   # +15.1%  sum_z_dr2 < 0.0186
        + 0.1482414 * max(0.0, Q.LHA - 0.148) / 0.1057727   # +14.8%  LHA > 0.148
        + 0.1092386 * max(0.0, 0.0129 - Q.sum_z_dr2) / 0.007947649   # +10.9%  sum_z_dr2 < 0.0129
        + 0.0858031 * max(0.0, Q.sj3_dr_max - 0.105) / 0.08650648   # +8.6%  sj3_dr_max > 0.105
        - 0.08184512 * max(0.0, 0.083 - Q.mass_over_sum_pt) / 0.03174206   # -8.2%  mass_over_sum_pt < 0.083
        - 0.07125681 * max(0.0, Q.sj3_dr_max - 0.0156) / 0.1529827   # -7.1%  sj3_dr_max > 0.0156
        + 0.06585383 * max(0.0, 0.054 - Q.sum_z_dr) / 0.01500937   # +6.6%  sum_z_dr < 0.054
        - 0.06373662 * max(0.0, 0.0165 - Q.sum_z_dr2_top5) / 0.01156664   # -6.4%  sum_z_dr2_top5 < 0.0165
        - 0.06337422 * max(0.0, Q.sum_z_dr - 0.0541) / 0.01966278   # -6.3%  sum_z_dr > 0.0541
        - 0.03805507 * max(0.0, 62.1 - Q.mass) / 25.59591   # -3.8%  mass < 62.1
        - 0.02649078 * max(0.0, Q.sj3_dr_max - 0.179) / 0.04318537   # -2.6%  sj3_dr_max > 0.179
        + 0.02250395 * max(0.0, Q.sum_zz_dr2 - 0.00697) / 0.00228803   # +2.3%  sum_zz_dr2 > 0.00697
        - 0.01503115 * max(0.0, 0.339 - Q.N2) / 0.1254971   # -1.5%  N2 < 0.339
        + 0.01379013 * max(0.0, Q.mass - 37.3) / 13.61769   # +1.4%  mass > 37.3
        + 0.009148442 * max(0.0, 42.2 - Q.mass_top5) / 19.72906   # +0.9%  mass_top5 < 42.2
        - 0.006986038 * max(0.0, Q.mass_over_sum_pt_sq - 0.0164) / 0.0008378198   # -0.7%  mass_over_sum_pt_sq > 0.0164
        - 0.006809374 * max(0.0, 0.159 - Q.LHA) / 0.01370166   # -0.7%  LHA < 0.159
        + 0.00672598 * max(0.0, 0.128 - Q.planar_flow) / 0.04110028   # +0.7%  planar_flow < 0.128
        - 0.006356468 * Q.lam2 / 0.0005288738   # -0.6%  lam2
        - 0.004212353 * max(0.0, Q.log_sum_pt - 6.64) * max(0.0, Q.z_7 - 0.0118) / 0.001077534   # -0.4%  log_sum_pt > 6.64 and z_7 > 0.0118
        - 0.002024882 * max(0.0, Q.sj3_pair_mass_min - 13.2) / 1.653288   # -0.2%  sj3_pair_mass_min > 13.2
        - 0.0018599 * max(0.0, Q.sum_pt - 913.0) / 10.82863   # -0.2%  sum_pt > 913
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 8.973;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.972771 * (0.1504552
        + 0.1851237 * max(0.0, Q.log_sum_pt - 6.45) / 0.1581974   # +18.5%  log_sum_pt > 6.45
        - 0.159984 * max(0.0, 0.00838 - Q.sum_z_dr2) / 0.004209678   # -16.0%  sum_z_dr2 < 0.00838
        - 0.1276603 * max(0.0, Q.sum_pt_top5 - 494.0) / 129.8715   # -12.8%  sum_pt_top5 > 494
        + 0.09286646 * max(0.0, Q.sum_pt_top5 - 393.0) / 208.3174   # +9.3%  sum_pt_top5 > 393
        + 0.08644524 * max(0.0, 0.0565 - Q.sum_z_dr) / 0.01626108   # +8.6%  sum_z_dr < 0.0565
        - 0.05746243 * max(0.0, Q.log_sum_pt - 6.31) * max(0.0, 0.824 - Q.z_dr_0p05_0p1) / 0.1642029   # -5.7%  log_sum_pt > 6.31 and z_dr_0p05_0p1 < 0.824
        - 0.04969328 * max(0.0, 0.00462 - Q.sum_zz_dr2) / 0.00185786   # -5.0%  sum_zz_dr2 < 0.00462
        - 0.03506932 * max(0.0, Q.log_sum_pt - 6.44) * max(0.0, 0.00846 - Q.zdr_6) / 0.0009989492   # -3.5%  log_sum_pt > 6.44 and zdr_6 < 0.00846
        - 0.02284433 * max(0.0, 0.0112 - Q.lam1) * max(0.0, 0.284 - Q.planar_flow) / 0.0006878421   # -2.3%  lam1 < 0.0112 and planar_flow < 0.284
        + 0.01998116 * max(0.0, 0.00703 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00444) / 3.162016e-05   # +2.0%  mass_over_sum_pt_sq < 0.00703 and centroid_offset > 0.00444
        - 0.01943405 * max(0.0, 0.00776 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.39) / 0.002232744   # -1.9%  lam1 < 0.00776 and n_pt_above_50 > 5.39
        + 0.01714063 * max(0.0, 0.00863 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.0155) / 1.292428e-05   # +1.7%  sum_z_dr2 < 0.00863 and centroid_offset > 0.0155
        - 0.01674477 * max(0.0, 0.0661 - Q.z_7) * max(0.0, 1.67 - Q.D2) / 0.007258306   # -1.7%  z_7 < 0.0661 and D2 < 1.67
        + 0.01362792 * max(0.0, Q.pt_7 - 27.5) * max(0.0, Q.sj2_dr - 0.139) / 0.3762467   # +1.4%  pt_7 > 27.5 and sj2_dr > 0.139
        + 0.01326859 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 1.65 - Q.D2) / 0.02762321   # +1.3%  log_sum_pt > 6.6 and D2 < 1.65
        - 0.013228 * max(0.0, Q.pt_7 - 30.0) * max(0.0, 52.2 - Q.pt_6) / 30.51203   # -1.3%  pt_7 > 30 and pt_6 < 52.2
        - 0.0126734 * max(0.0, Q.sj3_dr_min - 0.022) / 0.02886181   # -1.3%  sj3_dr_min > 0.022
        + 0.01248575 * max(0.0, Q.pt_7 - 41.4) / 1.915072   # +1.2%  pt_7 > 41.4
        - 0.008859563 * max(0.0, 0.0537 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.00935) / 0.001472126   # -0.9%  z_7 < 0.0537 and z_dr_0p05_0p1 > 0.00935
        - 0.00865964 * max(0.0, 0.000754 - Q.lam1) / 0.0001538633   # -0.9%  lam1 < 0.000754
        - 0.007107485 * max(0.0, 0.0146 - Q.centroid_offset) / 0.004428738   # -0.7%  centroid_offset < 0.0146
        + 0.005709256 * max(0.0, Q.pt_7 - 30.8) * max(0.0, Q.n_dr_0p1_0p2 - 0.533) / 6.443754   # +0.6%  pt_7 > 30.8 and n_dr_0p1_0p2 > 0.533
        + 0.005666471 * max(0.0, Q.log_sum_pt - 6.33) * max(0.0, Q.centroid_offset - 0.00512) / 0.001723524   # +0.6%  log_sum_pt > 6.33 and centroid_offset > 0.00512
        - 0.005192333 * max(0.0, 0.0951 - Q.sum_z_dr) * max(0.0, Q.pair_mass_0_6 - 5.5) / 0.08289967   # -0.5%  sum_z_dr < 0.0951 and pair_mass_0_6 > 5.5
        - 0.003071849 * max(0.0, Q.pt_7 - 55.9) / 0.2277934   # -0.3%  pt_7 > 55.9
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 9.783;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.783372 * (0.3362849
        - 0.3088395 * max(0.0, Q.log_sum_pt - 6.2) / 0.3575729   # -30.9%  log_sum_pt > 6.2
        + 0.1829714 * max(0.0, Q.pt_7 - 20.7) / 14.55348   # +18.3%  pt_7 > 20.7
        - 0.1221083 * max(0.0, Q.LHA - 0.13) / 0.1199428   # -12.2%  LHA > 0.13
        + 0.05646429 * max(0.0, 0.256 - Q.max_dr) / 0.1363978   # +5.6%  max_dr < 0.256
        + 0.05298503 * max(0.0, Q.sum_pt - 737.0) / 60.27585   # +5.3%  sum_pt > 737
        - 0.04550537 * max(0.0, Q.z_7 - 0.0518) / 0.008957664   # -4.6%  z_7 > 0.0518
        + 0.04412601 * max(0.0, 0.00763 - Q.lam1_plus_lam2) / 0.003627741   # +4.4%  lam1_plus_lam2 < 0.00763
        + 0.04089467 * max(0.0, 0.0529 - Q.z_7) / 0.009134424   # +4.1%  z_7 < 0.0529
        - 0.0324014 * max(0.0, Q.sum_pt - 559.0) * max(0.0, 0.00018 - Q.lam2) / 0.02297065   # -3.2%  sum_pt > 559 and lam2 < 0.00018
        + 0.02979801 * max(0.0, 0.00565 - Q.lam1) * max(0.0, Q.pt_6 - 16.7) / 0.05428771   # +3.0%  lam1 < 0.00565 and pt_6 > 16.7
        - 0.02102924 * max(0.0, Q.z_7 - 0.0316) * max(0.0, 0.00419 - Q.C2_b2) / 6.310948e-05   # -2.1%  z_7 > 0.0316 and C2_b2 < 0.00419
        + 0.01573512 * max(0.0, 495.0 - Q.sum_pt) / 6.606977   # +1.6%  sum_pt < 495
        - 0.01426991 * max(0.0, 53.7 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.0106) / 0.1568627   # -1.4%  sj3_pair_mass_max < 53.7 and centroid_offset > 0.0106
        + 0.01263753 * max(0.0, Q.log_sum_pt - 6.83) / 0.009226693   # +1.3%  log_sum_pt > 6.83
        - 0.01223126 * max(0.0, 0.023 - Q.z_7) / 0.0006957151   # -1.2%  z_7 < 0.023
        - 0.006434687 * max(0.0, 7.84e-05 - Q.sum_z_dr2) / 2.955537e-06   # -0.6%  sum_z_dr2 < 7.84e-05
        - 0.001568317 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, Q.pt_6 - 30.2) / 0.0566178   # -0.2%  log_sum_pt > 6.91 and pt_6 > 30.2
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 20.05;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.04949 * (0.05336793
        - 0.1622383 * max(0.0, 0.00831 - Q.sum_z_dr2) / 0.004154273   # -16.2%  sum_z_dr2 < 0.00831
        - 0.1317633 * max(0.0, 0.0124 - Q.mass_over_sum_pt_sq) / 0.007839129   # -13.2%  mass_over_sum_pt_sq < 0.0124
        + 0.112845 * max(0.0, Q.lam1_plus_lam2 - 0.00866) / 0.002218122   # +11.3%  lam1_plus_lam2 > 0.00866
        + 0.1115916 * max(0.0, 0.048 - Q.e2) / 0.02308931   # +11.2%  e2 < 0.048
        - 0.1010721 * max(0.0, Q.sum_z_dr2 - 0.01) / 0.001967422   # -10.1%  sum_z_dr2 > 0.01
        + 0.09706118 * max(0.0, 0.00789 - Q.lam1) / 0.003915548   # +9.7%  lam1 < 0.00789
        + 0.06211881 * max(0.0, Q.sj3_dr_max - 0.208) / 0.03226556   # +6.2%  sj3_dr_max > 0.208
        - 0.04550182 * max(0.0, Q.sj3_dr_max - 0.235) / 0.02479044   # -4.6%  sj3_dr_max > 0.235
        + 0.04324056 * max(0.0, Q.sum_z_dr - 0.0567) / 0.01836761   # +4.3%  sum_z_dr > 0.0567
        + 0.03631196 * max(0.0, 43.0 - Q.mass) / 13.35847   # +3.6%  mass < 43
        - 0.02161129 * max(0.0, Q.mass - 49.8) / 7.668944   # -2.2%  mass > 49.8
        - 0.01919895 * Q.centroid_offset / 0.01718434   # -1.9%  centroid_offset
        - 0.01676257 * max(0.0, 0.0287 - Q.centroid_offset) / 0.01442408   # -1.7%  centroid_offset < 0.0287
        - 0.01419087 * max(0.0, 6.28e-05 - Q.lam2) / 2.171906e-05   # -1.4%  lam2 < 6.28e-05
        + 0.01027345 * max(0.0, Q.sd_mass - 49.9) * max(0.0, 0.962 - Q.D2_b2) / 4.363929   # +1.0%  sd_mass > 49.9 and D2_b2 < 0.962
        + 0.009440132 * max(0.0, 0.0489 - Q.planar_flow) / 0.008803249   # +0.9%  planar_flow < 0.0489
        - 0.004778228 * max(0.0, Q.mass_over_sum_pt - 0.0741) * max(0.0, Q.pt_6 - 30.4) / 0.1265536   # -0.5%  mass_over_sum_pt > 0.0741 and pt_6 > 30.4
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 34.1;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.10112 * (0.3342999
        - 0.1361131 * max(0.0, 77.9 - Q.sd_mass) / 45.06417   # -13.6%  sd_mass < 77.9
        + 0.1124694 * Q.sj3_dr_max / 0.1682164   # +11.2%  sj3_dr_max
        - 0.09404131 * max(0.0, 0.00817 - Q.lam1_plus_lam2) / 0.004044028   # -9.4%  lam1_plus_lam2 < 0.00817
        - 0.05974284 * max(0.0, 0.000969 - Q.lam2) / 0.0007687916   # -6.0%  lam2 < 0.000969
        - 0.05230031 * max(0.0, 0.214 - Q.sj3_dr_max) / 0.07621791   # -5.2%  sj3_dr_max < 0.214
        - 0.04875803 * max(0.0, 0.0499 - Q.centroid_offset) / 0.03352225   # -4.9%  centroid_offset < 0.0499
        + 0.04283799 * max(0.0, 27.5 - Q.sd_mass) / 10.14461   # +4.3%  sd_mass < 27.5
        - 0.04001674 * max(0.0, Q.sum_z_dr - 0.0421) / 0.02629317   # -4.0%  sum_z_dr > 0.0421
        - 0.03909303 * max(0.0, Q.sj3_dr23 - 0.103) / 0.05978099   # -3.9%  sj3_dr23 > 0.103
        - 0.03871196 * max(0.0, Q.sj3_dr12 - 0.084) / 0.05665756   # -3.9%  sj3_dr12 > 0.084
        - 0.03698407 * max(0.0, Q.sj3_dr13 - 0.112) / 0.0527698   # -3.7%  sj3_dr13 > 0.112
        + 0.03179239 * max(0.0, Q.lam1 - 0.00741) / 0.002053326   # +3.2%  lam1 > 0.00741
        - 0.03065136 * max(0.0, Q.mass_over_sum_pt - 0.0874) / 0.008933725   # -3.1%  mass_over_sum_pt > 0.0874
        + 0.02928983 * max(0.0, 0.102 - Q.sj3_dr23) / 0.03713071   # +2.9%  sj3_dr23 < 0.102
        + 0.02776913 * max(0.0, 0.108 - Q.sj3_dr13) / 0.03880977   # +2.8%  sj3_dr13 < 0.108
        + 0.02299716 * max(0.0, 0.0899 - Q.sj3_dr12) / 0.03664622   # +2.3%  sj3_dr12 < 0.0899
        - 0.02133868 * max(0.0, 6.61 - Q.log_sum_pt) / 0.1360136   # -2.1%  log_sum_pt < 6.61
        + 0.02079872 * max(0.0, Q.max_dr - 0.127) / 0.03329857   # +2.1%  max_dr > 0.127
        - 0.0168143 * max(0.0, 0.000285 - Q.lam2) / 0.0001820274   # -1.7%  lam2 < 0.000285
        + 0.01226671 * max(0.0, 0.225 - Q.N2) / 0.05202841   # +1.2%  N2 < 0.225
        + 0.01080088 * max(0.0, Q.sum_pt - 708.0) / 74.25847   # +1.1%  sum_pt > 708
        - 0.01075132 * max(0.0, 0.25 - Q.N2) * max(0.0, 56.1 - Q.pt_7) / 1.323582   # -1.1%  N2 < 0.25 and pt_7 < 56.1
        + 0.008037424 * max(0.0, Q.lam2 - 0.00094) / 0.0003318222   # +0.8%  lam2 > 0.00094
        - 0.007788505 * max(0.0, Q.sum_pt_top2 - 262.0) / 125.2815   # -0.8%  sum_pt_top2 > 262
        - 0.007605857 * max(0.0, Q.sd_mass - 33.8) * max(0.0, 0.355 - Q.sd_zg) / 1.168325   # -0.8%  sd_mass > 33.8 and sd_zg < 0.355
        - 0.006638015 * max(0.0, Q.sd_mass - 42.8) * max(0.0, Q.centroid_offset - 0.00257) / 0.1796538   # -0.7%  sd_mass > 42.8 and centroid_offset > 0.00257
        - 0.006182892 * max(0.0, Q.sj3_pair_mass_min - 10.5) / 2.027342   # -0.6%  sj3_pair_mass_min > 10.5
        - 0.005680356 * max(0.0, Q.e3 - 2.03e-05) / 6.371924e-05   # -0.6%  e3 > 2.03e-05
        - 0.0051841 * max(0.0, Q.sj2_dr - 0.236) / 0.01425674   # -0.5%  sj2_dr > 0.236
        + 0.004072402 * max(0.0, Q.n_dr_0p1_0p2 - 1.31) / 0.7845959   # +0.4%  n_dr_0p1_0p2 > 1.31
        + 0.003472256 * max(0.0, Q.abseta_7 - 0.0225) / 0.03191586   # +0.3%  abseta_7 > 0.0225
        + 0.002564153 * max(0.0, 18.8 - Q.mass) / 3.429039   # +0.3%  mass < 18.8
        + 0.002310898 * max(0.0, 403.0 - Q.sum_pt_top5) / 9.271084   # +0.2%  sum_pt_top5 < 403
        - 0.002251409 * max(0.0, 23.6 - Q.pt_7) / 0.9792801   # -0.2%  pt_7 < 23.6
        - 0.001082358 * max(0.0, Q.mean_eta - 0.0245) / 0.000846551   # -0.1%  mean_eta > 0.0245
        + 0.0007900642 * max(0.0, 0.226 - Q.N2) * max(0.0, -0.0107 - Q.mean_phi) / 0.0001113309   # +0.1%  N2 < 0.226 and mean_phi < -0.0107
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 2.998;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.997603 * (0.1361088
        + 0.2290627 * max(0.0, 0.00199 - Q.sum_z_dr2) * max(0.0, 0.0201 - Q.centroid_offset) / 7.013677e-06   # +22.9%  sum_z_dr2 < 0.00199 and centroid_offset < 0.0201
        - 0.1502428 * max(0.0, 6.83 - Q.log_sum_pt) / 0.2962949   # -15.0%  log_sum_pt < 6.83
        + 0.1336232 * max(0.0, 0.00437 - Q.lam1_plus_lam2) / 0.001564646   # +13.4%  lam1_plus_lam2 < 0.00437
        + 0.1082308 * max(0.0, 0.0535 - Q.z_7) * max(0.0, 71.0 - Q.mass_top5) / 0.4556644   # +10.8%  z_7 < 0.0535 and mass_top5 < 71
        + 0.07132542 * max(0.0, Q.sum_pt_top5 - 387.0) * max(0.0, Q.centroid_offset - 0.00458) / 1.657405   # +7.1%  sum_pt_top5 > 387 and centroid_offset > 0.00458
        + 0.06443792 * max(0.0, Q.sum_pt_top5 - 529.0) * max(0.0, 42.0 - Q.pt_6) / 1170.662   # +6.4%  sum_pt_top5 > 529 and pt_6 < 42
        + 0.05160828 * max(0.0, 0.0295 - Q.z_7) / 0.001531694   # +5.2%  z_7 < 0.0295
        - 0.04743731 * max(0.0, 0.196 - Q.LHA) * max(0.0, 6.81 - Q.log_sum_pt) / 0.00293798   # -4.7%  LHA < 0.196 and log_sum_pt < 6.81
        - 0.04600319 * max(0.0, Q.pt_7 - 37.4) / 3.19952   # -4.6%  pt_7 > 37.4
        + 0.04155984 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, 0.023 - Q.dr_2) / 0.0004057978   # +4.2%  log_sum_pt > 6.67 and dr_2 < 0.023
        - 0.03860595 * max(0.0, Q.sum_pt - 793.0) / 38.19317   # -3.9%  sum_pt > 793
        - 0.01786261 * max(0.0, Q.log_sum_pt - 6.87) * max(0.0, 0.0421 - Q.dr_2) / 0.0001471017   # -1.8%  log_sum_pt > 6.87 and dr_2 < 0.0421
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 30.42;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 30.42142 * (-0.2103781
        + 0.1856473 * Q.pt_7 / 34.64819   # +18.6%  pt_7
        - 0.155924 * max(0.0, 0.00797 - Q.lam1_plus_lam2) / 0.003888057   # -15.6%  lam1_plus_lam2 < 0.00797
        + 0.1261521 * max(0.0, 0.00779 - Q.lam1) / 0.003837727   # +12.6%  lam1 < 0.00779
        - 0.1252033 * max(0.0, 0.0123 - Q.lam1_plus_lam2) / 0.007439185   # -12.5%  lam1_plus_lam2 < 0.0123
        + 0.09870196 * max(0.0, 6.88 - Q.log_sum_pt) / 0.3419879   # +9.9%  log_sum_pt < 6.88
        + 0.05145049 * max(0.0, 0.0038 - Q.lam2) / 0.003410015   # +5.1%  lam2 < 0.0038
        + 0.03922822 * max(0.0, 0.0518 - Q.z_7) / 0.008585455   # +3.9%  z_7 < 0.0518
        - 0.03706824 * max(0.0, Q.z_7 - 0.0508) / 0.009476206   # -3.7%  z_7 > 0.0508
        + 0.03658925 * max(0.0, 0.0957 - Q.tau1) * max(0.0, 0.0151 - Q.mean_phi2) / 0.0006115916   # +3.7%  tau1 < 0.0957 and mean_phi2 < 0.0151
        - 0.02687209 * max(0.0, 0.175 - Q.max_dr) / 0.0686964   # -2.7%  max_dr < 0.175
        + 0.02678309 * max(0.0, 0.00779 - Q.C2_b2) / 0.006172574   # +2.7%  C2_b2 < 0.00779
        + 0.01695131 * max(0.0, 0.00247 - Q.sum_zz_dr2) / 0.0008317466   # +1.7%  sum_zz_dr2 < 0.00247
        - 0.01494806 * max(0.0, Q.mass_over_sum_pt - 0.0519) / 0.02380844   # -1.5%  mass_over_sum_pt > 0.0519
        + 0.01429847 * max(0.0, 0.144 - Q.sj2_dr) / 0.04142664   # +1.4%  sj2_dr < 0.144
        - 0.0116565 * max(0.0, 0.0292 - Q.centroid_offset) / 0.01483713   # -1.2%  centroid_offset < 0.0292
        - 0.008634537 * max(0.0, 0.0176 - Q.lam1) * max(0.0, 0.3 - Q.planar_flow) / 0.00153611   # -0.9%  lam1 < 0.0176 and planar_flow < 0.3
        + 0.007053802 * max(0.0, Q.sj3_dr_max - 0.196) / 0.03630908   # +0.7%  sj3_dr_max > 0.196
        - 0.005666397 * Q.C2_b2 / 0.003447597   # -0.6%  C2_b2
        + 0.004444388 * max(0.0, 31.5 - Q.pt_6) / 1.728959   # +0.4%  pt_6 < 31.5
        + 0.002951343 * max(0.0, 520.0 - Q.sum_pt) / 9.592312   # +0.3%  sum_pt < 520
        + 0.001244783 * max(0.0, 0.0185 - Q.lam1_plus_lam2) * max(0.0, -0.0217 - Q.mean_eta) / 5.703023e-06   # +0.1%  lam1_plus_lam2 < 0.0185 and mean_eta < -0.0217
        + 0.001041647 * max(0.0, 0.0138 - Q.lam1_plus_lam2) * max(0.0, -0.0196 - Q.mean_phi) / 3.845676e-06   # +0.1%  lam1_plus_lam2 < 0.0138 and mean_phi < -0.0196
        + 0.0007917745 * max(0.0, 0.0123 - Q.lam1) * max(0.0, Q.mean_eta - 0.0238) / 2.189718e-06   # +0.1%  lam1 < 0.0123 and mean_eta > 0.0238
        + 0.0006968322 * max(0.0, 0.0122 - Q.lam1_plus_lam2) * max(0.0, Q.mean_phi - 0.0251) / 1.593881e-06   # +0.1%  lam1_plus_lam2 < 0.0122 and mean_phi > 0.0251
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 29.2;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.19683 * (0.3630531
        - 0.137665 * max(0.0, 0.0887 - Q.sum_z_dr) / 0.03756432   # -13.8%  sum_z_dr < 0.0887
        - 0.1259408 * max(0.0, Q.sum_z_dr - 0.0213) / 0.04018658   # -12.6%  sum_z_dr > 0.0213
        - 0.101013 * max(0.0, Q.sum_z_dr2 - 0.00757) / 0.002457716   # -10.1%  sum_z_dr2 > 0.00757
        - 0.09874154 * max(0.0, Q.lam1_plus_lam2 - 0.00496) / 0.003356158   # -9.9%  lam1_plus_lam2 > 0.00496
        + 0.0958763 * max(0.0, Q.mass_over_sum_pt - 0.0689) / 0.01504991   # +9.6%  mass_over_sum_pt > 0.0689
        + 0.08423179 * max(0.0, 0.0798 - Q.mass_over_sum_pt) / 0.02945271   # +8.4%  mass_over_sum_pt < 0.0798
        - 0.07002013 * max(0.0, 0.00634 - Q.lam1_plus_lam2) / 0.002704187   # -7.0%  lam1_plus_lam2 < 0.00634
        - 0.05206294 * max(0.0, 0.195 - Q.sj2_dr) / 0.0697281   # -5.2%  sj2_dr < 0.195
        + 0.04007423 * max(0.0, 0.148 - Q.sj2_dr) / 0.04317492   # +4.0%  sj2_dr < 0.148
        + 0.03439977 * max(0.0, Q.tau1 - 0.0953) / 0.01059456   # +3.4%  tau1 > 0.0953
        - 0.02799174 * max(0.0, 0.00402 - Q.sum_z_dr2) / 0.001394659   # -2.8%  sum_z_dr2 < 0.00402
        + 0.02532297 * max(0.0, Q.sum_z_dr2 - 0.0219) / 0.0004962083   # +2.5%  sum_z_dr2 > 0.0219
        - 0.0190269 * max(0.0, 0.0264 - Q.centroid_offset) * max(0.0, 0.00481 - Q.C2_b2) / 4.960046e-05   # -1.9%  centroid_offset < 0.0264 and C2_b2 < 0.00481
        + 0.0164016 * max(0.0, 0.000433 - Q.lam2) * max(0.0, 0.0396 - Q.tau21_b2) / 3.830998e-06   # +1.6%  lam2 < 0.000433 and tau21_b2 < 0.0396
        + 0.01461348 * max(0.0, Q.lam1_plus_lam2 - 0.00978) * max(0.0, 51.3 - Q.pt_6) / 0.0260163   # +1.5%  lam1_plus_lam2 > 0.00978 and pt_6 < 51.3
        - 0.01437719 * max(0.0, 0.0361 - Q.tau21_b2) / 0.009475583   # -1.4%  tau21_b2 < 0.0361
        - 0.009772151 * max(0.0, 45.1 - Q.pt_7) / 11.5982   # -1.0%  pt_7 < 45.1
        - 0.009710101 * max(0.0, Q.mass - 79.0) / 1.331005   # -1.0%  mass > 79
        + 0.008044696 * max(0.0, 0.233 - Q.planar_flow) * max(0.0, Q.sd_mass - 46.4) / 1.087406   # +0.8%  planar_flow < 0.233 and sd_mass > 46.4
        - 0.00333391 * max(0.0, Q.lam2 - 0.000262) / 0.0004307061   # -0.3%  lam2 > 0.000262
        + 0.003275937 * max(0.0, Q.sum_pt - 752.0) / 53.73425   # +0.3%  sum_pt > 752
        + 0.002888622 * max(0.0, Q.mass_top5 - 61.4) / 1.077121   # +0.3%  mass_top5 > 61.4
        - 0.002300365 * max(0.0, Q.mass_over_sum_pt - 0.0176) * max(0.0, 30.9 - Q.pt_6) / 0.0433312   # -0.2%  mass_over_sum_pt > 0.0176 and pt_6 < 30.9
        - 0.001996421 * max(0.0, Q.centroid_offset - 0.0426) * max(0.0, Q.n_pt_above_50 - 3.06) / 0.001583944   # -0.2%  centroid_offset > 0.0426 and n_pt_above_50 > 3.06
        - 0.0009183378 * max(0.0, 0.0638 - Q.tau1) * max(0.0, Q.n_dr_0p05_0p1 - 4.09) / 0.001367987   # -0.1%  tau1 < 0.0638 and n_dr_0p05_0p1 > 4.09
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 29.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.7817 * (-0.04062897
        + 0.1487115 * max(0.0, 0.00748 - Q.lam1_plus_lam2) / 0.003514985   # +14.9%  lam1_plus_lam2 < 0.00748
        - 0.101443 * max(0.0, 0.0606 - Q.sum_z_dr) / 0.01842162   # -10.1%  sum_z_dr < 0.0606
        + 0.08349308 * max(0.0, 0.00431 - Q.sum_z_dr2) / 0.001534917   # +8.3%  sum_z_dr2 < 0.00431
        - 0.05537584 * max(0.0, 0.00732 - Q.lam1) / 0.003479297   # -5.5%  lam1 < 0.00732
        - 0.054274 * max(0.0, 0.196 - Q.sj3_dr_max) * max(0.0, 0.00673 - Q.sum_z_dr2) / 0.0003665243   # -5.4%  sj3_dr_max < 0.196 and sum_z_dr2 < 0.00673
        + 0.04708699 * max(0.0, 0.0052 - Q.sum_z_dr2) / 0.002003329   # +4.7%  sum_z_dr2 < 0.0052
        - 0.04088043 * max(0.0, 0.00424 - Q.lam1) / 0.001541125   # -4.1%  lam1 < 0.00424
        + 0.0367335 * max(0.0, Q.mass_over_sum_pt - 0.0621) / 0.01829408   # +3.7%  mass_over_sum_pt > 0.0621
        + 0.03611194 * max(0.0, 0.0549 - Q.tau1) * max(0.0, 0.00353 - Q.lam1_plus_lam2) / 4.910846e-05   # +3.6%  tau1 < 0.0549 and lam1_plus_lam2 < 0.00353
        + 0.03376747 * max(0.0, 0.199 - Q.sj3_dr_max) * max(0.0, 0.00348 - Q.lam1_plus_lam2) / 0.0001736879   # +3.4%  sj3_dr_max < 0.199 and lam1_plus_lam2 < 0.00348
        - 0.0274609 * max(0.0, 0.00387 - Q.mass_over_sum_pt_sq) / 0.001468281   # -2.7%  mass_over_sum_pt_sq < 0.00387
        + 0.02462926 * max(0.0, 0.2 - Q.sj3_dr_max) / 0.06668192   # +2.5%  sj3_dr_max < 0.2
        - 0.02449384 * max(0.0, 6.84 - Q.log_sum_pt) / 0.3052168   # -2.4%  log_sum_pt < 6.84
        - 0.01856683 * max(0.0, 0.0011 - Q.lam2) / 0.0008861408   # -1.9%  lam2 < 0.0011
        - 0.01751192 * max(0.0, Q.pt_7 - 20.5) / 14.73262   # -1.8%  pt_7 > 20.5
        - 0.01742884 * max(0.0, 0.0074 - Q.sum_z_dr2_top5) / 0.003816622   # -1.7%  sum_z_dr2_top5 < 0.0074
        + 0.01583107 * max(0.0, 0.067 - Q.C2) / 0.04135756   # +1.6%  C2 < 0.067
        - 0.0156346 * max(0.0, 0.0326 - Q.mass_over_sum_pt) / 0.007023002   # -1.6%  mass_over_sum_pt < 0.0326
        + 0.0150521 * max(0.0, 48.4 - Q.pt_7) / 14.46056   # +1.5%  pt_7 < 48.4
        - 0.01460366 * max(0.0, Q.mass_over_sum_pt - 0.0851) / 0.009475422   # -1.5%  mass_over_sum_pt > 0.0851
        - 0.01266069 * max(0.0, 0.00526 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.00577) / 1.278159e-05   # -1.3%  sum_z_dr2 < 0.00526 and centroid_offset > 0.00577
        - 0.0123912 * max(0.0, 28.4 - Q.mass) / 6.859313   # -1.2%  mass < 28.4
        + 0.01221497 * max(0.0, 0.000166 - Q.lam2) / 9.071883e-05   # +1.2%  lam2 < 0.000166
        - 0.01210209 * max(0.0, 0.0527 - Q.z_7) / 0.009033102   # -1.2%  z_7 < 0.0527
        + 0.01199216 * max(0.0, 0.0329 - Q.mass_over_sum_pt) * max(0.0, 0.0269 - Q.centroid_offset) / 0.000125756   # +1.2%  mass_over_sum_pt < 0.0329 and centroid_offset < 0.0269
        + 0.009978252 * max(0.0, Q.z_7 - 0.0516) / 0.009060041   # +1.0%  z_7 > 0.0516
        - 0.00947128 * max(0.0, 30.5 - Q.mass) * max(0.0, 0.0268 - Q.centroid_offset) / 0.128214   # -0.9%  mass < 30.5 and centroid_offset < 0.0268
        + 0.008964245 * max(0.0, 0.0602 - Q.sum_z_dr) * max(0.0, Q.lam1_plus_lam2 - 0.000434) / 8.988905e-06   # +0.9%  sum_z_dr < 0.0602 and lam1_plus_lam2 > 0.000434
        - 0.007462244 * max(0.0, 0.00134 - Q.C2_b2) / 0.0007145927   # -0.7%  C2_b2 < 0.00134
        - 0.006850841 * max(0.0, 0.0054 - Q.sum_z_dr2) * max(0.0, 44.1 - Q.pt_7) / 0.02599104   # -0.7%  sum_z_dr2 < 0.0054 and pt_7 < 44.1
        - 0.005718755 * max(0.0, 2.91e-05 - Q.e3) / 1.468226e-05   # -0.6%  e3 < 2.91e-05
        - 0.005101159 * max(0.0, Q.sj2_dr - 0.145) / 0.04688926   # -0.5%  sj2_dr > 0.145
        - 0.004571742 * max(0.0, 0.00101 - Q.mean_phi) / 0.005919751   # -0.5%  mean_phi < 0.00101
        - 0.00451005 * max(0.0, Q.sum_pt - 718.0) / 69.23556   # -0.5%  sum_pt > 718
        - 0.004415311 * max(0.0, Q.sum_pt - 847.0) / 22.75008   # -0.4%  sum_pt > 847
        + 0.003455022 * max(0.0, 0.0954 - Q.dr_4) / 0.0432338   # +0.3%  dr_4 < 0.0954
        + 0.003451768 * max(0.0, 0.00853 - Q.mean_eta2) / 0.006011667   # +0.3%  mean_eta2 < 0.00853
        + 0.003365077 * max(0.0, 0.0113 - Q.zdr_2) / 0.005192628   # +0.3%  zdr_2 < 0.0113
        - 0.003222335 * max(0.0, 0.00033 - Q.sum_z_dr2_top5) / 5.244077e-05   # -0.3%  sum_z_dr2_top5 < 0.00033
        + 0.003210649 * max(0.0, 0.0898 - Q.dr_3) / 0.04051636   # +0.3%  dr_3 < 0.0898
        + 0.003025146 * max(0.0, 0.0138 - Q.zdr_1) / 0.006087433   # +0.3%  zdr_1 < 0.0138
        - 0.002984919 * max(0.0, 0.135 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.0142) / 0.0001216087   # -0.3%  sj3_dr_max < 0.135 and centroid_offset > 0.0142
        - 0.002587047 * max(0.0, Q.mass - 41.1) / 11.62091   # -0.3%  mass > 41.1
        + 0.002487018 * max(0.0, Q.N2 - 0.212) / 0.0544615   # +0.2%  N2 > 0.212
        + 0.002152844 * max(0.0, 0.02 - Q.zdr_0) / 0.008167561   # +0.2%  zdr_0 < 0.02
        + 0.001952022 * max(0.0, Q.sj2_dr - 0.218) / 0.01833897   # +0.2%  sj2_dr > 0.218
        + 0.001849964 * max(0.0, Q.centroid_offset - 0.0274) / 0.00314829   # +0.2%  centroid_offset > 0.0274
        + 0.001466577 * max(0.0, -0.0133 - Q.mean_phi) / 0.001804841   # +0.1%  mean_phi < -0.0133
        + 0.0014208 * max(0.0, Q.max_pair_mass - 24.2) / 2.337781   # +0.1%  max_pair_mass > 24.2
        + 0.001278181 * max(0.0, Q.log_sum_pt - 6.76) * max(0.0, 0.000634 - Q.sum_z_dr2) / 6.312837e-06   # +0.1%  log_sum_pt > 6.76 and sum_z_dr2 < 0.000634
        - 0.001181807 * max(0.0, Q.sum_z_dr2_top5 - 0.00926) / 0.001902498   # -0.1%  sum_z_dr2_top5 > 0.00926
        + 0.001010842 * max(0.0, Q.sum_pt - 963.0) / 5.902861   # +0.1%  sum_pt > 963
        - 0.0009091766 * max(0.0, -0.000531 - Q.mean_eta) / 0.005298792   # -0.1%  mean_eta < -0.000531
        - 0.0008674054 * max(0.0, Q.sum_z_dr2 - 0.0197) / 0.0006780265   # -0.1%  sum_z_dr2 > 0.0197
        + 0.0006256652 * max(0.0, Q.mass_over_sum_pt - 0.151) / 0.0009964371   # +0.1%  mass_over_sum_pt > 0.151
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 18.61;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.61495 * (0.05280701
        + 0.3608297 * max(0.0, 0.00766 - Q.sum_z_dr2) / 0.003650451   # +36.1%  sum_z_dr2 < 0.00766
        - 0.1921826 * max(0.0, 0.00718 - Q.lam1) / 0.003374971   # -19.2%  lam1 < 0.00718
        + 0.1185824 * max(0.0, 58.9 - Q.mass) * max(0.0, 0.0258 - Q.centroid_offset) / 0.3284831   # +11.9%  mass < 58.9 and centroid_offset < 0.0258
        - 0.1184615 * max(0.0, 0.0158 - Q.sum_z_dr2) / 0.01045097   # -11.8%  sum_z_dr2 < 0.0158
        - 0.07609233 * max(0.0, Q.sj3_dr_max - 0.153) / 0.05665821   # -7.6%  sj3_dr_max > 0.153
        - 0.06706409 * max(0.0, 28.4 - Q.mass) / 6.859313   # -6.7%  mass < 28.4
        + 0.03702262 * max(0.0, Q.sj3_dr_max - 0.235) / 0.02479044   # +3.7%  sj3_dr_max > 0.235
        + 0.01709307 * max(0.0, 46.2 - Q.mass) * max(0.0, 6.83 - Q.log_sum_pt) / 3.704152   # +1.7%  mass < 46.2 and log_sum_pt < 6.83
        + 0.007588156 * max(0.0, Q.lam2 - 0.000378) / 0.0004082462   # +0.8%  lam2 > 0.000378
        + 0.005083534 * max(0.0, Q.mass_over_sum_pt - 0.129) / 0.002734964   # +0.5%  mass_over_sum_pt > 0.129
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 19.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.58528 * (-0.06433404
        + 0.4136366 * Q.sum_z_dr / 0.05870426   # +41.4%  sum_z_dr
        - 0.2095185 * max(0.0, Q.LHA - 0.165) / 0.09304938   # -21.0%  LHA > 0.165
        + 0.1215095 * max(0.0, 0.00763 - Q.lam1_plus_lam2) / 0.003627741   # +12.2%  lam1_plus_lam2 < 0.00763
        - 0.09051883 * max(0.0, 0.00627 - Q.lam1) / 0.00272744   # -9.1%  lam1 < 0.00627
        - 0.0435271 * max(0.0, 0.0013 - Q.lam2) / 0.001066947   # -4.4%  lam2 < 0.0013
        - 0.02070424 * max(0.0, Q.LHA - 0.311) / 0.01577814   # -2.1%  LHA > 0.311
        - 0.01973368 * max(0.0, 0.00263 - Q.sum_z_dr2) / 0.0007919868   # -2.0%  sum_z_dr2 < 0.00263
        + 0.01867564 * max(0.0, 0.0207 - Q.sum_z_dr) / 0.002631422   # +1.9%  sum_z_dr < 0.0207
        - 0.01681532 * max(0.0, 0.0572 - Q.z_7) / 0.01147501   # -1.7%  z_7 < 0.0572
        + 0.01320304 * max(0.0, Q.lam2 - 0.000935) / 0.0003323717   # +1.3%  lam2 > 0.000935
        - 0.008649939 * max(0.0, 42.7 - Q.pt_7) * max(0.0, 6.56 - Q.log_sum_pt) / 0.9011247   # -0.9%  pt_7 < 42.7 and log_sum_pt < 6.56
        + 0.007966204 * max(0.0, 53.4 - Q.pt_7) * max(0.0, 1.02 - Q.D2) / 3.197137   # +0.8%  pt_7 < 53.4 and D2 < 1.02
        - 0.005005733 * max(0.0, Q.lam1 - 0.00759) * max(0.0, 0.386 - Q.D2_b2) / 0.0002866628   # -0.5%  lam1 > 0.00759 and D2_b2 < 0.386
        + 0.004176068 * max(0.0, Q.sj3_pair_mass_min - 16.3) / 1.314943   # +0.4%  sj3_pair_mass_min > 16.3
        - 0.004130557 * max(0.0, Q.lam2 - 0.00398) / 0.0001311152   # -0.4%  lam2 > 0.00398
        + 0.002228962 * max(0.0, Q.C2_b2 - 0.012) / 0.001431306   # +0.2%  C2_b2 > 0.012
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 30.82;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 30.81741 * (-0.099619
        + 0.2689356 * max(0.0, 0.00907 - Q.lam1_plus_lam2) / 0.004763162   # +26.9%  lam1_plus_lam2 < 0.00907
        - 0.2293403 * max(0.0, 0.00857 - Q.sum_zz_dr2) / 0.004619395   # -22.9%  sum_zz_dr2 < 0.00857
        + 0.147782 * max(0.0, 0.0126 - Q.sum_z_dr2) / 0.007693006   # +14.8%  sum_z_dr2 < 0.0126
        - 0.09944168 * max(0.0, 81.9 - Q.mass) / 42.68155   # -9.9%  mass < 81.9
        + 0.08065927 * max(0.0, Q.LHA - 0.12) / 0.1281294   # +8.1%  LHA > 0.12
        - 0.04048814 * max(0.0, 0.00479 - Q.sum_z_dr2) / 0.001779943   # -4.0%  sum_z_dr2 < 0.00479
        + 0.03032764 * max(0.0, 0.0489 - Q.tau1) / 0.01469527   # +3.0%  tau1 < 0.0489
        + 0.02413724 * max(0.0, Q.z_7 - 0.0102) / 0.04202528   # +2.4%  z_7 > 0.0102
        + 0.02168076 * max(0.0, 0.273 - Q.planar_flow) / 0.1214809   # +2.2%  planar_flow < 0.273
        - 0.01759863 * max(0.0, Q.sum_z_dr - 0.08) / 0.009498146   # -1.8%  sum_z_dr > 0.08
        - 0.01516431 * max(0.0, 0.0154 - Q.centroid_offset) * max(0.0, 13.9 - Q.sj3_pair_mass_min) / 0.04788163   # -1.5%  centroid_offset < 0.0154 and sj3_pair_mass_min < 13.9
        - 0.007707723 * max(0.0, Q.sj3_dr_max - 0.176) / 0.04456512   # -0.8%  sj3_dr_max > 0.176
        - 0.00766368 * max(0.0, Q.sj2_dr - 0.168) * max(0.0, 0.00143 - Q.lam2) / 2.890757e-05   # -0.8%  sj2_dr > 0.168 and lam2 < 0.00143
        + 0.006173423 * max(0.0, 0.0152 - Q.centroid_offset) * max(0.0, 0.309 - Q.D2_b2) / 0.0004153907   # +0.6%  centroid_offset < 0.0152 and D2_b2 < 0.309
        - 0.002899584 * max(0.0, 0.26 - Q.planar_flow) * max(0.0, 1.03e-05 - Q.e3) / 1.794331e-07   # -0.3%  planar_flow < 0.26 and e3 < 1.03e-05
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 6.286;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.28621 * (-0.5058692
        + 0.1859293 * max(0.0, 0.00777 - Q.sum_z_dr2) / 0.003734156   # +18.6%  sum_z_dr2 < 0.00777
        - 0.1588593 * max(0.0, 0.13 - Q.sum_z_dr) / 0.07342814   # -15.9%  sum_z_dr < 0.13
        + 0.06912408 * max(0.0, Q.mass - 49.4) / 7.829342   # +6.9%  mass > 49.4
        + 0.06771863 * max(0.0, Q.centroid_offset - 0.00369) / 0.01377649   # +6.8%  centroid_offset > 0.00369
        + 0.05430954 * max(0.0, 0.116 - Q.sj2_dr) / 0.03021249   # +5.4%  sj2_dr < 0.116
        - 0.05295561 * max(0.0, 0.0668 - Q.z_7) / 0.01780161   # -5.3%  z_7 < 0.0668
        + 0.05207834 * max(0.0, 2.84e-09 - Q.e4) / 2.020836e-09   # +5.2%  e4 < 2.84e-09
        + 0.04911532 * max(0.0, 0.343 - Q.N2) / 0.1286455   # +4.9%  N2 < 0.343
        - 0.0485416 * max(0.0, Q.lam1 - 0.00601) / 0.002460828   # -4.9%  lam1 > 0.00601
        - 0.04314469 * max(0.0, 0.217 - Q.sj2_dr) / 0.08555728   # -4.3%  sj2_dr < 0.217
        - 0.03490808 * max(0.0, 5.21e-05 - Q.e3) / 3.203497e-05   # -3.5%  e3 < 5.21e-05
        - 0.02864502 * max(0.0, Q.sd_mass - 23.0) / 19.11556   # -2.9%  sd_mass > 23
        - 0.02560103 * max(0.0, 0.00397 - Q.sum_z_dr2_top5) / 0.0015934   # -2.6%  sum_z_dr2_top5 < 0.00397
        + 0.02144022 * max(0.0, Q.sum_pt - 758.0) / 51.24628   # +2.1%  sum_pt > 758
        + 0.01511042 * max(0.0, Q.zdr_7 - 0.00379) / 0.001666443   # +1.5%  zdr_7 > 0.00379
        + 0.01501615 * max(0.0, Q.zdr_6 - 0.00434) / 0.001771006   # +1.5%  zdr_6 > 0.00434
        + 0.007991895 * max(0.0, Q.sd_rg - 0.23) / 0.01116416   # +0.8%  sd_rg > 0.23
        + 0.006567373 * max(0.0, Q.e3 - 0.00026) / 3.104051e-05   # +0.7%  e3 > 0.00026
        + 0.006375739 * max(0.0, Q.dr_5 - 0.152) / 0.006922148   # +0.6%  dr_5 > 0.152
        - 0.005929319 * max(0.0, Q.sj3_pair_mass_min - 11.6) / 1.863647   # -0.6%  sj3_pair_mass_min > 11.6
        + 0.005718595 * max(0.0, Q.dr_4 - 0.147) / 0.006477169   # +0.6%  dr_4 > 0.147
        + 0.005462869 * max(0.0, Q.sj3_dr_max - 0.298) / 0.01262527   # +0.5%  sj3_dr_max > 0.298
        + 0.005438958 * max(0.0, Q.dr_3 - 0.133) / 0.006865548   # +0.5%  dr_3 > 0.133
        + 0.004402595 * max(0.0, Q.sum_z_dr - 0.138) / 0.00151233   # +0.4%  sum_z_dr > 0.138
        + 0.004308044 * max(0.0, Q.dr_2 - 0.134) / 0.005352029   # +0.4%  dr_2 > 0.134
        + 0.004167694 * max(0.0, Q.sum_z_dr2_top2 - 0.0111) / 0.0014555   # +0.4%  sum_z_dr2_top2 > 0.0111
        + 0.003787786 * max(0.0, Q.mass_over_sum_pt - 0.15) / 0.001053576   # +0.4%  mass_over_sum_pt > 0.15
        - 0.003519707 * max(0.0, 6.06 - Q.log_sum_pt) / 0.005193807   # -0.4%  log_sum_pt < 6.06
        + 0.002843743 * max(0.0, Q.mean_phi - 0.0149) / 0.001610483   # +0.3%  mean_phi > 0.0149
        + 0.002164158 * max(0.0, Q.centroid_offset - 0.0425) / 0.001271435   # +0.2%  centroid_offset > 0.0425
        - 0.001978293 * max(0.0, Q.zdr_0 - 0.0431) / 0.0004303103   # -0.2%  zdr_0 > 0.0431
        + 0.001870167 * max(0.0, Q.mass - 102.0) / 0.2600943   # +0.2%  mass > 102
        + 0.001865192 * max(0.0, Q.mass - 76.9) * max(0.0, 6.56 - Q.log_sum_pt) / 0.1376172   # +0.2%  mass > 76.9 and log_sum_pt < 6.56
        + 0.001561622 * max(0.0, Q.LHA - 0.0973) / 0.1476193   # +0.2%  LHA > 0.0973
        + 0.00103842 * max(0.0, -0.0296 - Q.mean_eta) / 0.0005988739   # +0.1%  mean_eta < -0.0296
        - 0.0005105009 * max(0.0, Q.lam1_plus_lam2 - 0.0196) / 0.0006871767   # -0.1%  lam1_plus_lam2 > 0.0196
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 16.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.6712 * (0.7617927
        - 0.2038671 * Q.LHA / 0.242765   # -20.4%  LHA
        - 0.1602322 * max(0.0, 0.0694 - Q.e2) / 0.04200099   # -16.0%  e2 < 0.0694
        - 0.1329691 * max(0.0, 0.00729 - Q.lam1_plus_lam2) / 0.003374056   # -13.3%  lam1_plus_lam2 < 0.00729
        + 0.1139418 * max(0.0, 0.00654 - Q.lam1) / 0.002913416   # +11.4%  lam1 < 0.00654
        - 0.07445216 * max(0.0, 0.133 - Q.sum_z_dr) * max(0.0, 6.83 - Q.log_sum_pt) / 0.01791064   # -7.4%  sum_z_dr < 0.133 and log_sum_pt < 6.83
        - 0.05907461 * max(0.0, Q.lam1 - 0.00518) / 0.002782048   # -5.9%  lam1 > 0.00518
        + 0.04356923 * max(0.0, 0.00482 - Q.sum_zz_dr2) / 0.001968432   # +4.4%  sum_zz_dr2 < 0.00482
        - 0.04336273 * max(0.0, 0.144 - Q.sum_z_dr) * max(0.0, 39.0 - Q.pt_7) / 0.6693601   # -4.3%  sum_z_dr < 0.144 and pt_7 < 39
        + 0.03871661 * max(0.0, Q.sum_pt_top5 - 576.0) * max(0.0, 37.7 - Q.pt_7) / 1036.039   # +3.9%  sum_pt_top5 > 576 and pt_7 < 37.7
        - 0.03268241 * max(0.0, Q.LHA - 0.239) / 0.04578615   # -3.3%  LHA > 0.239
        - 0.02372522 * max(0.0, 0.000319 - Q.lam2) / 0.0002092741   # -2.4%  lam2 < 0.000319
        - 0.020365 * max(0.0, 28.8 - Q.pt_7) / 2.108752   # -2.0%  pt_7 < 28.8
        - 0.01879999 * max(0.0, 0.0215 - Q.centroid_offset) / 0.008853629   # -1.9%  centroid_offset < 0.0215
        + 0.01216239 * max(0.0, Q.tau1 - 0.153) / 0.00265048   # +1.2%  tau1 > 0.153
        - 0.006460383 * max(0.0, 23.9 - Q.pt_6) / 0.5551667   # -0.6%  pt_6 < 23.9
        - 0.005914341 * max(0.0, Q.lam2 - 0.00111) / 0.0003140101   # -0.6%  lam2 > 0.00111
        + 0.004693753 * max(0.0, 0.152 - Q.sum_z_dr) * max(0.0, Q.z_7 - 0.0608) / 0.0003854705   # +0.5%  sum_z_dr < 0.152 and z_7 > 0.0608
        - 0.004363935 * max(0.0, Q.mass - 74.9) / 1.728077   # -0.4%  mass > 74.9
        - 0.0006469871 * max(0.0, Q.sum_pt - 1060.0) * max(0.0, 56.9 - Q.pt_6) / 36.31668   # -0.1%  sum_pt > 1060 and pt_6 < 56.9
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 45.27;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 45.27035 * (0.1990265
        - 0.1407423 * max(0.0, 0.0169 - Q.sum_zz_dr2) / 0.01179898   # -14.1%  sum_zz_dr2 < 0.0169
        + 0.1322147 * max(0.0, 0.0829 - Q.mass_over_sum_pt) / 0.03166881   # +13.2%  mass_over_sum_pt < 0.0829
        - 0.1279912 * max(0.0, 0.00723 - Q.lam1_plus_lam2) / 0.003330003   # -12.8%  lam1_plus_lam2 < 0.00723
        - 0.1266758 * max(0.0, Q.lam1_plus_lam2 - 0.00335) / 0.00418588   # -12.7%  lam1_plus_lam2 > 0.00335
        - 0.04885152 * max(0.0, 75.3 - Q.sd_mass) / 42.69354   # -4.9%  sd_mass < 75.3
        + 0.03646994 * max(0.0, Q.sum_z_dr2 - 0.00937) / 0.002081976   # +3.6%  sum_z_dr2 > 0.00937
        - 0.03635781 * max(0.0, Q.mass_over_sum_pt - 0.0916) / 0.008068288   # -3.6%  mass_over_sum_pt > 0.0916
        + 0.03632854 * max(0.0, 60.4 - Q.sd_mass) / 30.06592   # +3.6%  sd_mass < 60.4
        - 0.03090026 * max(0.0, 60.6 - Q.mass) / 24.45569   # -3.1%  mass < 60.6
        + 0.02963018 * max(0.0, Q.sum_z_dr - 0.0534) / 0.02002043   # +3.0%  sum_z_dr > 0.0534
        + 0.02397209 * max(0.0, Q.lam1 - 0.00402) / 0.00332891   # +2.4%  lam1 > 0.00402
        - 0.0203044 * max(0.0, Q.sum_z_dr - 0.0905) / 0.007237695   # -2.0%  sum_z_dr > 0.0905
        + 0.01862729 * max(0.0, Q.sj2_dr - 0.121) * max(0.0, 0.142 - Q.D2_b2) / 0.002395636   # +1.9%  sj2_dr > 0.121 and D2_b2 < 0.142
        + 0.01811674 * max(0.0, Q.lam1_plus_lam2 - 0.0167) / 0.000988134   # +1.8%  lam1_plus_lam2 > 0.0167
        - 0.01537032 * max(0.0, 0.145 - Q.D2_b2) / 0.02973588   # -1.5%  D2_b2 < 0.145
        - 0.01378957 * max(0.0, 0.0045 - Q.lam1_plus_lam2) / 0.001629918   # -1.4%  lam1_plus_lam2 < 0.0045
        + 0.01365283 * max(0.0, Q.sj2_dr - 0.137) / 0.0515057   # +1.4%  sj2_dr > 0.137
        + 0.01219192 * max(0.0, Q.tau1 - 0.0969) / 0.01020208   # +1.2%  tau1 > 0.0969
        + 0.01029295 * max(0.0, Q.sj3_dr23 - 0.0687) / 0.07676529   # +1.0%  sj3_dr23 > 0.0687
        + 0.009579342 * max(0.0, Q.sj3_dr13 - 0.081) / 0.06818556   # +1.0%  sj3_dr13 > 0.081
        + 0.009518082 * max(0.0, 0.121 - Q.planar_flow) / 0.0377971   # +1.0%  planar_flow < 0.121
        + 0.009382825 * max(0.0, Q.z_7 - 0.0363) / 0.01879486   # +0.9%  z_7 > 0.0363
        - 0.008864567 * max(0.0, 0.239 - Q.LHA) / 0.04202116   # -0.9%  LHA < 0.239
        + 0.00846593 * max(0.0, Q.sj3_dr12 - 0.0649) / 0.06607855   # +0.8%  sj3_dr12 > 0.0649
        - 0.006943322 * max(0.0, Q.sj2_dr - 0.185) * max(0.0, 0.136 - Q.D2_b2) / 0.0008404455   # -0.7%  sj2_dr > 0.185 and D2_b2 < 0.136
        - 0.006781962 * max(0.0, Q.sj2_dr - 0.205) / 0.02177459   # -0.7%  sj2_dr > 0.205
        - 0.006153263 * max(0.0, Q.centroid_offset - 0.0259) / 0.003451801   # -0.6%  centroid_offset > 0.0259
        - 0.005816628 * max(0.0, Q.pt_7 - 28.7) / 8.028072   # -0.6%  pt_7 > 28.7
        - 0.005547338 * max(0.0, Q.z_dr_0p05_0p1 - 0.74) * max(0.0, 0.0243 - Q.C2_b2) / 0.0005668847   # -0.6%  z_dr_0p05_0p1 > 0.74 and C2_b2 < 0.0243
        - 0.005318121 * max(0.0, Q.sj3_dr_max - 0.26) / 0.01926026   # -0.5%  sj3_dr_max > 0.26
        + 0.005202764 * max(0.0, Q.z_dr_0p05_0p1 - 0.735) * max(0.0, 1.03 - Q.n_dr_0p2_0p4) / 0.02355309   # +0.5%  z_dr_0p05_0p1 > 0.735 and n_dr_0p2_0p4 < 1.03
        + 0.004754222 * max(0.0, Q.D2 - 0.404) / 1.126834   # +0.5%  D2 > 0.404
        - 0.003604248 * max(0.0, 0.021 - Q.centroid_offset) / 0.008498207   # -0.4%  centroid_offset < 0.021
        - 0.003499197 * max(0.0, Q.sum_pt - 743.0) / 57.60359   # -0.3%  sum_pt > 743
        - 0.00310437 * max(0.0, 0.107 - Q.planar_flow) * max(0.0, 0.00746 - Q.lam1) / 5.78337e-05   # -0.3%  planar_flow < 0.107 and lam1 < 0.00746
        - 0.001790164 * max(0.0, 0.117 - Q.planar_flow) * max(0.0, 673.0 - Q.sum_pt_top5) / 3.538923   # -0.2%  planar_flow < 0.117 and sum_pt_top5 < 673
        + 0.001519832 * max(0.0, 0.00703 - Q.mean_phi) / 0.01000048   # +0.2%  mean_phi < 0.00703
        + 0.0009706512 * max(0.0, 0.684 - Q.z_dr_0p05_0p1) * max(0.0, 632.0 - Q.sum_pt) / 13.77483   # +0.1%  z_dr_0p05_0p1 < 0.684 and sum_pt < 632
        - 0.0007028979 * max(0.0, Q.centroid_offset - 0.0509) * max(0.0, 0.00111 - Q.C2_b2) / 2.052931e-07   # -0.1%  centroid_offset > 0.0509 and C2_b2 < 0.00111
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 33.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.00936 * (-0.3514155
        + 0.3910717 * max(0.0, 0.0175 - Q.lam1_plus_lam2) / 0.0119528   # +39.1%  lam1_plus_lam2 < 0.0175
        - 0.1261007 * max(0.0, 0.00723 - Q.sum_z_dr2) / 0.003330003   # -12.6%  sum_z_dr2 < 0.00723
        - 0.05188287 * max(0.0, 0.192 - Q.sj2_dr) / 0.0676925   # -5.2%  sj2_dr < 0.192
        + 0.04773976 * max(0.0, 0.163 - Q.sj2_dr) / 0.05034692   # +4.8%  sj2_dr < 0.163
        - 0.04225723 * max(0.0, 0.00478 - Q.lam1_plus_lam2) / 0.001774661   # -4.2%  lam1_plus_lam2 < 0.00478
        + 0.04161442 * max(0.0, 0.0047 - Q.lam1) / 0.001779359   # +4.2%  lam1 < 0.0047
        + 0.03766907 * max(0.0, Q.mass_over_sum_pt - 0.0808) / 0.01062762   # +3.8%  mass_over_sum_pt > 0.0808
        - 0.03306481 * max(0.0, 0.0896 - Q.sum_z_dr) / 0.03829642   # -3.3%  sum_z_dr < 0.0896
        + 0.0311153 * max(0.0, Q.tau1 - 0.0407) / 0.03367528   # +3.1%  tau1 > 0.0407
        + 0.03073282 * max(0.0, 0.0439 - Q.e2) / 0.01973678   # +3.1%  e2 < 0.0439
        - 0.02270755 * max(0.0, Q.sum_z_dr - 0.0882) / 0.007664228   # -2.3%  sum_z_dr > 0.0882
        + 0.01770532 * max(0.0, Q.LHA - 0.32) / 0.01362334   # +1.8%  LHA > 0.32
        - 0.01668111 * max(0.0, Q.sum_z_dr2 - 0.018) / 0.0008445289   # -1.7%  sum_z_dr2 > 0.018
        - 0.01464464 * max(0.0, 0.0293 - Q.centroid_offset) / 0.01492007   # -1.5%  centroid_offset < 0.0293
        + 0.01454117 * max(0.0, Q.lam1 - 0.00706) / 0.002142834   # +1.5%  lam1 > 0.00706
        + 0.01255428 * max(0.0, 0.000915 - Q.lam2) / 0.0007207106   # +1.3%  lam2 < 0.000915
        + 0.01085283 * max(0.0, 0.266 - Q.N2) * max(0.0, Q.sum_pt_top5 - 389.0) / 15.24447   # +1.1%  N2 < 0.266 and sum_pt_top5 > 389
        + 0.01000002 * max(0.0, Q.sj3_dr_max - 0.168) / 0.04847198   # +1.0%  sj3_dr_max > 0.168
        + 0.009848123 * max(0.0, 0.637 - Q.z_dr_0p05_0p1) / 0.4094209   # +1.0%  z_dr_0p05_0p1 < 0.637
        - 0.006035696 * max(0.0, 0.0228 - Q.sum_z_dr) / 0.003172523   # -0.6%  sum_z_dr < 0.0228
        - 0.004798638 * max(0.0, 0.277 - Q.N2) * max(0.0, 0.62 - Q.z_dr_0p05_0p1) / 0.02111999   # -0.5%  N2 < 0.277 and z_dr_0p05_0p1 < 0.62
        - 0.004577711 * max(0.0, Q.sj3_dr_max - 0.25) / 0.02134284   # -0.5%  sj3_dr_max > 0.25
        - 0.004470849 * max(0.0, 0.00804 - Q.sum_z_dr2) * max(0.0, 0.856 - Q.D2) / 0.0001773796   # -0.4%  sum_z_dr2 < 0.00804 and D2 < 0.856
        + 0.004262797 * max(0.0, Q.log_sum_pt - 6.82) / 0.01042312   # +0.4%  log_sum_pt > 6.82
        - 0.003552426 * max(0.0, Q.mass - 79.4) / 1.29716   # -0.4%  mass > 79.4
        - 0.002882587 * max(0.0, Q.sum_pt_top5 - 799.0) / 13.42064   # -0.3%  sum_pt_top5 > 799
        - 0.002556238 * max(0.0, Q.sum_pt_top5 - 709.0) / 31.13645   # -0.3%  sum_pt_top5 > 709
        + 0.001881648 * max(0.0, Q.D2 - 1.29) / 0.544842   # +0.2%  D2 > 1.29
        + 0.001353336 * max(0.0, 0.0076 - Q.mean_phi) / 0.01043755   # +0.1%  mean_phi < 0.0076
        + 0.0008443033 * max(0.0, Q.mass_top5 - 70.6) / 0.5123145   # +0.1%  mass_top5 > 70.6
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0118262605042017, 0.7511642857142857, 2.092385294117647, 0.9553138655462184, 1.1962033613445378, 1.7658289915966388, 0.9396754201680673, 1.8396600840336135, 0.2542997899159664, 2.9881058823529414, 2.0264446428571428, 2.1751300420168067, 0.10073949579831933, 3.3262100840336135, 0.5139534663865546, 0.3655004201680672]
T = [2.3354251969537816, 1.3631185793067229, 3.3882681131827725, 2.4921454109768906, 2.875740116202731]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +38%, n9 +22%, n5 -14%, n1 +13%, n0 -7%, n6 +4% ...
            + 0.3849714 * h[2] / H_AVG[2]
            + 0.2199089 * h[9] / H_AVG[9]
            - 0.1417699 * h[5] / H_AVG[5]
            + 0.1256403 * h[1] / H_AVG[1]
            - 0.06769553 * h[0] / H_AVG[0]
            + 0.04400783 * h[6] / H_AVG[6]
            - 0.01600623 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -19%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5565904 * h[9] / H_AVG[9]
            - 0.185828 * h[10] / H_AVG[10]
            + 0.08616963 * h[6] / H_AVG[6]
            - 0.08227022 * h[4] / H_AVG[4]
            + 0.06072343 * h[5] / H_AVG[5]
            + 0.01675847 * h[15] / H_AVG[15]
            + 0.01165983 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +12%, n14 -11%, n0 +10%, n6 -9% ...
            + 0.2407347 * h[11] / H_AVG[11]
            - 0.1409738 * h[3] / H_AVG[3]
            + 0.1187703 * h[7] / H_AVG[7]
            - 0.1137646 * h[14] / H_AVG[14]
            + 0.1026528 * h[0] / H_AVG[0]
            - 0.08666627 * h[6] / H_AVG[6]
            - 0.07416224 * h[15] / H_AVG[15]
            + 0.06902469 * h[13] / H_AVG[13]
            - 0.0275593 * h[9] / H_AVG[9]
            - 0.01876326 * h[8] / H_AVG[8]
            - 0.006927989 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -22%, n6 -14%, n14 +8%, n13 +7%, n1 +4% ...
            + 0.3460234 * h[7] / H_AVG[7]
            - 0.2156231 * h[3] / H_AVG[3]
            - 0.1413956 * h[6] / H_AVG[6]
            + 0.077336 * h[14] / H_AVG[14]
            + 0.07299017 * h[13] / H_AVG[13]
            + 0.03767659 * h[1] / H_AVG[1]
            + 0.03749917 * h[4] / H_AVG[4]
            - 0.03746905 * h[9] / H_AVG[9]
            - 0.02291577 * h[15] / H_AVG[15]
            + 0.01107122 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +26%, n5 -15%, n4 +5%, n3 +2%, n12 -2% ...
            - 0.469887 * h[13] / H_AVG[13]
            + 0.2642508 * h[10] / H_AVG[10]
            - 0.1535108 * h[5] / H_AVG[5]
            + 0.05199546 * h[4] / H_AVG[4]
            + 0.02076235 * h[3] / H_AVG[3]
            - 0.0175154 * h[12] / H_AVG[12]
            + 0.0165805 * h[8] / H_AVG[8]
            + 0.00549764 * h[0] / H_AVG[0]
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
