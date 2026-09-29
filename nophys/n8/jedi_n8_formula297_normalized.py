"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  13.1%   (on for 90% of jets)
  neuron  9:  11.8%   (on for 65% of jets)
  neuron  7:  10.3%   (on for 60% of jets)
  neuron  5:   8.8%   (on for 77% of jets)
  neuron  3:   8.1%   (on for 22% of jets)
  neuron  6:   8.1%   (on for 34% of jets)
  neuron  2:   8.0%   (on for 97% of jets)
  neuron 10:   7.7%   (on for 70% of jets)
  neuron 11:   6.0%   (on for 78% of jets)
  neuron 14:   4.3%   (on for 25% of jets)
  neuron  0:   4.0%   (on for 42% of jets)
  neuron  4:   3.3%   (on for 57% of jets)
  neuron  1:   3.2%   (on for 58% of jets)
  neuron 15:   2.0%   (on for 22% of jets)
  neuron  8:   1.1%   (on for 24% of jets)
  neuron 12:   0.3%   (on for 4% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 87.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr01                   ΔR between particles 0 and 1
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
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
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
        max_dr=max(dr[i] for i in real),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        pt_6=pt[6],
        pt_7=pt[7],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sj2_dr=subjets(2)["dr"][0],
        dr01=math.sqrt(dist2(0, 1)),
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 23.22;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.22452 * (-0.1040896
        - 0.1525289 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) / 0.00156571   # -15.3%  lam1_plus_lam2 < 0.004372
        + 0.1153875 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +11.5%  lam1_plus_lam2 < 0.008678
        + 0.08348644 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +8.3%  sum_z_dr2 < 0.01324
        + 0.07302529 * max(0.0, 0.0245477 - Q.e2) / 0.007455821   # +7.3%  e2 < 0.02455
        - 0.06745295 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -6.7%  sum_z_dr < 0.08724
        - 0.06466129 * max(0.0, 56.92035 - Q.mass) / 21.78475   # -6.5%  mass < 56.92
        + 0.0646155 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +6.5%  sj3_dr_max < 0.3012
        - 0.05171453 * max(0.0, 0.001563465 - Q.C2_b2) / 0.0008800039   # -5.2%  C2_b2 < 0.001563
        + 0.04552329 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +4.6%  lam1 < 0.006507
        - 0.03167001 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -3.2%  sj3_dr_max > 0.2337
        + 0.03044213 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +3.0%  planar_flow < 0.1484
        + 0.02983476 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +3.0%  sum_pt_top5 > 687.4
        - 0.02882969 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -2.9%  lam1 < 0.005433
        + 0.02813013 * max(0.0, Q.n_dr_0_0p05 - 4.0) / 1.61275   # +2.8%  n_dr_0_0p05 > 4
        + 0.0261036 * max(0.0, Q.sj3_dr_max - 0.1070199) / 0.08518534   # +2.6%  sj3_dr_max > 0.107
        - 0.02577833 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -2.6%  log_sum_pt > 6.67
        + 0.02013752 * max(0.0, 21.78408 - Q.mass) / 4.431023   # +2.0%  mass < 21.78
        + 0.02004092 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, 1.002471 - Q.D2) / 0.001008123   # +2.0%  sum_z_dr2 < 0.01324 and D2 < 1.002
        - 0.01207372 * max(0.0, Q.sum_pt - 901.5938) / 12.40362   # -1.2%  sum_pt > 901.6
        - 0.0108199 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.01837778) / 2.056482e-05   # -1.1%  sum_z_dr2 < 0.01324 and centroid_offset > 0.01838
        - 0.01024735 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # -1.0%  lam1 < 0.006507 and D2 < 0.8757
        + 0.007496243 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2) / 2.36013e-06   # +0.7%  lam2 < 7.301e-05 and D2_b2 < 0.267
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 12.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.08977 * (-0.007785221
        - 0.1986605 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -19.9%  sum_z_dr2 < 0.008678
        + 0.1350711 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) / 0.002578466   # +13.5%  mass_over_sum_pt_sq < 0.005833
        + 0.100814 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +10.1%  log_sum_pt > 6.378
        + 0.09427544 * max(0.0, Q.sum_pt_top5 - 367.5938) / 230.8403   # +9.4%  sum_pt_top5 > 367.6
        - 0.09102652 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -9.1%  z_7 < 0.06165
        - 0.08787675 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 0.002544039   # -8.8%  lam1_plus_lam2 < 0.006097
        + 0.07786432 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # +7.8%  sum_zz_dr2 < 0.008169
        + 0.06074249 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +6.1%  log_sum_pt > 6.503
        + 0.05505068 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +5.5%  pt_7 > 34.53
        - 0.03875706 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -3.9%  sum_pt_top5 > 531.2
        - 0.0307644 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1) / 0.1196218   # -3.1%  log_sum_pt > 6.378 and z_dr_0p05_0p1 < 0.7509
        - 0.02909682 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # -2.9%  pt_7 > 34.53 and tau2 < 0.01713
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 3.783;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.78314 * (0.2078605
        + 0.3498067 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +35.0%  lam1 < 0.005954
        + 0.1766154 * max(0.0, 0.0003193707 - Q.sum_z_dr2) * max(0.0, 36.76827 - Q.mass_top2) / 0.001427268   # +17.7%  sum_z_dr2 < 0.0003194 and mass_top2 < 36.77
        - 0.1450256 * max(0.0, 0.0003193707 - Q.sum_z_dr2) / 4.035777e-05   # -14.5%  sum_z_dr2 < 0.0003194
        + 0.09311114 * max(0.0, 788.4484 - Q.sum_pt) / 112.1576   # +9.3%  sum_pt < 788.4
        - 0.07192311 * max(0.0, 0.03243272 - Q.z_7) / 0.002061024   # -7.2%  z_7 < 0.03243
        + 0.07147977 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # +7.1%  log_sum_pt < 6.464
        - 0.03419076 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # -3.4%  sum_z_dr < 0.007674
        + 0.03180811 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # +3.2%  log_sum_pt > 6.843
        - 0.02603943 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -2.6%  log_sum_pt > 6.896
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 17.31;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.30557 * (-0.2812364
        - 0.1996764 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # -20.0%  sum_z_dr2 > 0.008678
        + 0.1582248 * max(0.0, Q.sum_z_dr - 0.04081947) / 0.0270611   # +15.8%  sum_z_dr > 0.04082
        + 0.07986331 * max(0.0, Q.lam1_plus_lam2 - 0.007520088) / 0.002470136   # +8.0%  lam1_plus_lam2 > 0.00752
        - 0.07880334 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # -7.9%  tau1 > 0.05357
        + 0.06600525 * max(0.0, Q.sum_z_dr - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494) / 0.008781747   # +6.6%  sum_z_dr > 0.04082 and log_sum_pt > 6.08
        + 0.05195628 * max(0.0, Q.sj2_dr - 0.1872617) / 0.02732165   # +5.2%  sj2_dr > 0.1873
        - 0.04932894 * max(0.0, 0.004372139 - Q.sum_z_dr2) / 0.00156571   # -4.9%  sum_z_dr2 < 0.004372
        + 0.04606082 * max(0.0, Q.sum_z_dr - 0.07608178) / 0.01059033   # +4.6%  sum_z_dr > 0.07608
        + 0.0335143 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +3.4%  lam1 > 0.008376
        + 0.03132219 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # +3.1%  tau1 > 0.1028
        + 0.02992656 * max(0.0, Q.mass_over_sum_pt - 0.0681391) / 0.01539157   # +3.0%  mass_over_sum_pt > 0.06814
        + 0.02837278 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # +2.8%  e2 > 0.06344
        - 0.02593199 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr) / 0.0001196243   # -2.6%  mass_over_sum_pt > 0.06814 and sj2_dr < 0.218
        - 0.02534116 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -2.5%  lam1 > 0.012
        + 0.02375862 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50) / 0.02994202   # +2.4%  centroid_offset > 0.01096 and n_pt_above_50 < 8
        - 0.01889664 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # -1.9%  lam1_plus_lam2 > 0.01324
        - 0.01796173 * max(0.0, Q.mass - 64.61873) / 3.274818   # -1.8%  mass > 64.62
        - 0.01763624 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # -1.8%  lam2 > 0.001131
        + 0.01128328 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6) / 0.005634141   # +1.1%  lam2 > 0.001131 and pt_6 < 56.53
        + 0.006135336 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # +0.6%  lam1 > 0.01643
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 43.75;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 43.75485 * (-0.07232836
        + 0.1254416 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # +12.5%  sum_zz_dr2 < 0.01166
        - 0.1143526 * max(0.0, 0.01882765 - Q.sum_z_dr2) / 0.01314296   # -11.4%  sum_z_dr2 < 0.01883
        + 0.1058254 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # +10.6%  C2_b2 < 0.009033
        - 0.08531634 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -8.5%  lam2 < 0.0005373
        + 0.07523088 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +7.5%  N2 < 0.2233
        + 0.05432112 * max(0.0, 0.01716248 - Q.sum_zz_dr2) / 0.0120354   # +5.4%  sum_zz_dr2 < 0.01716
        - 0.05184213 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -5.2%  mass < 69.61
        + 0.04737609 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +4.7%  mass < 76.66
        + 0.04117271 * max(0.0, Q.sum_z_dr2 - 0.003562611) / 0.004066872   # +4.1%  sum_z_dr2 > 0.003563
        - 0.03852382 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # -3.9%  lam1_plus_lam2 < 0.01324
        - 0.03702786 * max(0.0, Q.sum_z_dr - 0.05464922) / 0.01938482   # -3.7%  sum_z_dr > 0.05465
        - 0.0315204 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -3.2%  sum_z_dr2 < 0.008678
        - 0.02813489 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -2.8%  mass_over_sum_pt > 0.09041
        - 0.0274612 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266) / 0.01381411   # -2.7%  N2 < 0.2233 and eccentricity > 0.7117
        + 0.02474229 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # +2.5%  sum_z_dr2 < 0.002635
        + 0.02208801 * max(0.0, Q.max_dr - 0.1117619) / 0.04033009   # +2.2%  max_dr > 0.1118
        - 0.01682921 * max(0.0, 76.6557 - Q.mass) * max(0.0, 0.875672 - Q.D2) / 2.2476   # -1.7%  mass < 76.66 and D2 < 0.8757
        - 0.01233089 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.2%  sj3_dr_max > 0.2337
        + 0.01203895 * max(0.0, Q.tau1 - 0.07283629) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.001837197   # +1.2%  tau1 > 0.07284 and sj3_dr_min < 0.209
        + 0.01146112 * max(0.0, Q.e2 - 0.009668065) / 0.02044666   # +1.1%  e2 > 0.009668
        + 0.008506561 * max(0.0, 0.01165737 - Q.sum_zz_dr2) * max(0.0, 0.875672 - Q.D2) / 0.0005867961   # +0.9%  sum_zz_dr2 < 0.01166 and D2 < 0.8757
        + 0.006593768 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +0.7%  tau1 > 0.1136
        + 0.006455195 * max(0.0, 1.002471 - Q.D2) * max(0.0, Q.M3 - 0.04581318) / 0.006342224   # +0.6%  D2 < 1.002 and M3 > 0.04581
        - 0.005629057 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.25 - Q.pt_6) / 1.048693   # -0.6%  N2 < 0.2233 and pt_6 < 62.25
        - 0.005316949 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, 0.02117036 - Q.dr01) / 0.0003970056   # -0.5%  sj3_dr_max > 0.169 and dr01 < 0.02117
        - 0.00446099 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 7.11886e-05   # -0.4%  N2 < 0.2233 and sum_zz_dr2 > 0.01166
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 15.06;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.05516 * (0.1119439
        + 0.1130635 * max(0.0, 0.005284669 - Q.sum_zz_dr2) / 0.00223735   # +11.3%  sum_zz_dr2 < 0.005285
        - 0.09473126 * max(0.0, 0.02383244 - Q.zdr_0) / 0.01109706   # -9.5%  zdr_0 < 0.02383
        + 0.07815938 * max(0.0, Q.log_sum_pt - 6.267538) / 0.2983143   # +7.8%  log_sum_pt > 6.268
        + 0.07478141 * max(0.0, 0.001653836 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 7.0272e-06   # +7.5%  sum_z_dr2 < 0.001654 and centroid_offset < 0.02355
        - 0.07449591 * max(0.0, 0.09041383 - Q.mass_over_sum_pt) / 0.03744004   # -7.4%  mass_over_sum_pt < 0.09041
        + 0.06215811 * max(0.0, 60.63098 - Q.mass) / 24.47895   # +6.2%  mass < 60.63
        - 0.05874528 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -5.9%  sj3_dr_max < 0.3012
        + 0.04411515 * max(0.0, 0.002915531 - Q.sum_z_dr2_top3) * max(0.0, 0.006646257 - Q.tau3) / 3.390069e-06   # +4.4%  sum_z_dr2_top3 < 0.002916 and tau3 < 0.006646
        - 0.04410066 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0004149607   # -4.4%  z_7 < 0.07149 and centroid_offset < 0.03117
        - 0.04407033 * max(0.0, 0.002915531 - Q.sum_z_dr2_top3) / 0.001178811   # -4.4%  sum_z_dr2_top3 < 0.002916
        - 0.04194857 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -4.2%  log_sum_pt > 6.701
        - 0.04153207 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -4.2%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.04061158 * max(0.0, 0.04939969 - Q.z_7) / 0.007457505   # +4.1%  z_7 < 0.0494
        - 0.03930504 * max(0.0, Q.pt_7 - 23.21641) / 12.35308   # -3.9%  pt_7 > 23.22
        + 0.03732036 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 42.18339 - Q.mass_top3) / 0.6375506   # +3.7%  z_7 < 0.07149 and mass_top3 < 42.18
        - 0.03623703 * max(0.0, 37.15625 - Q.pt_7) / 5.803748   # -3.6%  pt_7 < 37.16
        + 0.03283345 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +3.3%  tau1 < 0.09539
        + 0.01475889 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +1.5%  z_7 < 0.02807
        - 0.01385265 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -1.4%  log_sum_pt > 6.896
        + 0.01317939 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0002302115 - Q.mean_phi2) / 3.738179e-06   # +1.3%  log_sum_pt > 6.701 and mean_phi2 < 0.0002302
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 36.27;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 36.26623 * (0.02187334
        - 0.2131627 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -21.3%  sum_z_dr2 < 0.008678
        + 0.09920003 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +9.9%  tau1 < 0.1136
        - 0.09193213 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # -9.2%  lam1_plus_lam2 < 0.01324
        + 0.0774925 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +7.7%  e3 < 0.0005117
        + 0.06481423 * max(0.0, 0.1789613 - Q.sj3_dr_max) / 0.05394779   # +6.5%  sj3_dr_max < 0.179
        + 0.05757753 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +5.8%  lam1 < 0.012
        - 0.05690044 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # -5.7%  max_dr < 0.1452
        - 0.05159632 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -5.2%  lam2 < 0.001131
        + 0.04397738 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +4.4%  centroid_offset > 0.008092
        + 0.03469879 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # +3.5%  sum_zz_dr2 < 0.003013
        + 0.03288542 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt) / 1.016213   # +3.3%  pt_6 < 41.22 and log_sum_pt < 6.767
        - 0.03014691 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 1.341509   # -3.0%  mass < 60.63 and z_dr_0p2_0p4 < 0.05644
        - 0.02883798 * max(0.0, Q.centroid_offset - 0.01837778) / 0.005523694   # -2.9%  centroid_offset > 0.01838
        - 0.02649736 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757) / 0.07098489   # -2.6%  pt_6 < 41.22 and z_7 > 0.02321
        + 0.02243257 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +2.2%  lam1 < 0.00733
        + 0.01622238 * max(0.0, 45.7571 - Q.mass) / 14.82409   # +1.6%  mass < 45.76
        + 0.01486421 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +1.5%  sj3_dr_max > 0.1879
        - 0.0100921 * max(0.0, Q.sj3_pair_mass_min - 4.501727) / 3.480268   # -1.0%  sj3_pair_mass_min > 4.502
        + 0.007960497 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002658741   # +0.8%  lam2 < 0.003408 and n_dr_0_0p05 < 3
        - 0.005975856 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # -0.6%  sj3_dr_min > 0.1278
        + 0.005017554 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2) / 2.179362e-05   # +0.5%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        + 0.004514267 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308) / 0.05438016   # +0.5%  centroid_offset > 0.008092 and sj3_pair_mass_min > 6.811
        - 0.003200849 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 2.69383   # -0.3%  sj3_pair_mass_min > 4.502 and n_dr_0p2_0p4 > 1
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 39.77;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 39.76982 * (0.2109189
        - 0.09573853 * max(0.0, 8.10414e-05 - Q.sum_z_dr2_top2) / 7.505045e-06   # -9.6%  sum_z_dr2_top2 < 8.104e-05
        - 0.09328587 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # -9.3%  lam1_plus_lam2 > 0.00559
        - 0.07001308 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -7.0%  mass_over_sum_pt > 0.09041
        + 0.06622231 * max(0.0, Q.sum_z_dr2 - 0.01323868) / 0.001442684   # +6.6%  sum_z_dr2 > 0.01324
        + 0.06171963 * max(0.0, Q.sum_z_dr2 - 0.001653836) / 0.005220304   # +6.2%  sum_z_dr2 > 0.001654
        - 0.05621274 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -5.6%  sum_z_dr2 > 0.00752
        - 0.05531617 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -5.5%  sum_z_dr < 0.08724
        - 0.04932756 * max(0.0, Q.mass_over_sum_pt - 0.01109984) / 0.05101851   # -4.9%  mass_over_sum_pt > 0.0111
        + 0.04822029 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +4.8%  mass_over_sum_pt > 0.08475
        + 0.04765565 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +4.8%  sj2_dr < 0.1592
        - 0.04481035 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -4.5%  e3 < 8.148e-05
        - 0.0397553 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -4.0%  sj2_dr < 0.1873
        + 0.03854949 * max(0.0, Q.mass_over_sum_pt - 0.07269073) / 0.01344138   # +3.9%  mass_over_sum_pt > 0.07269
        - 0.03757753 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -3.8%  lam1 < 0.008376
        + 0.03078761 * max(0.0, Q.mass - 36.22941) / 14.20528   # +3.1%  mass > 36.23
        - 0.02797399 * max(0.0, Q.sum_z_dr2 - 0.004372139) / 0.003638805   # -2.8%  sum_z_dr2 > 0.004372
        + 0.02792337 * max(0.0, Q.lam1_plus_lam2 - 0.008678045) / 0.002214537   # +2.8%  lam1_plus_lam2 > 0.008678
        - 0.0258347 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # -2.6%  e2 > 0.05028
        - 0.01990827 * max(0.0, 0.0009641429 - Q.sum_z_dr2) / 0.0002052288   # -2.0%  sum_z_dr2 < 0.0009641
        + 0.01982808 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +2.0%  LHA < 0.3033
        - 0.018609 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -1.9%  centroid_offset < 0.02077
        + 0.01458874 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.5%  tau1 < 0.05357
        - 0.01014173 * max(0.0, Q.mass - 76.6557) / 1.546047   # -1.0%  mass > 76.66
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 13.05;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.04748 * (-0.05570268
        - 0.123345 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # -12.3%  LHA < 0.1967
        - 0.1080753 * max(0.0, 0.06108601 - Q.sum_z_dr) / 0.01868685   # -10.8%  sum_z_dr < 0.06109
        + 0.1002718 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +10.0%  sum_z_dr2 < 0.006679 and centroid_offset < 0.02355
        - 0.09028286 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -9.0%  sj3_dr_max < 0.1426
        + 0.08256337 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 4.808955e-05   # +8.3%  tau1 < 0.05357 and lam1_plus_lam2 < 0.003563
        - 0.07689405 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.02261167   # -7.7%  mass < 29.64 and lam1_plus_lam2 < 0.003563
        + 0.07247919 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0001901112   # +7.2%  sum_z_dr2 < 0.00502 and z_dr_0p2_0p4 < 0.101
        + 0.06810371 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.0003703114   # +6.8%  sj3_dr_max < 0.1986 and sum_z_dr2 < 0.006679
        + 0.06071932 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.0001947983 - Q.lam2) / 2.869243e-06   # +6.1%  sum_z_dr < 0.06109 and lam2 < 0.0001948
        + 0.04941014 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) / 0.00722828   # +4.9%  mass_over_sum_pt < 0.03319
        + 0.03355606 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +3.4%  sum_z_dr2 < 0.00502
        - 0.03347641 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 2.955458e-05 - Q.e3) / 8.40609e-07   # -3.3%  log_sum_pt > 6.701 and e3 < 2.955e-05
        - 0.03344641 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -3.3%  log_sum_pt > 6.701
        + 0.02867904 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +2.9%  sum_pt_top5 > 687.4
        - 0.02021687 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -2.0%  sum_z_dr2 < 0.00502 and centroid_offset > 0.00679
        + 0.01848041 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.0002365852   # +1.8%  log_sum_pt > 6.701 and lam1_plus_lam2 < 0.008678
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 35.37;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.37145 * (-0.08006978
        + 0.1271867 * max(0.0, 0.00609665 - Q.sum_z_dr2) / 0.002544039   # +12.7%  sum_z_dr2 < 0.006097
        + 0.06882653 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +6.9%  sum_pt < 988.4
        + 0.06697436 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # +6.7%  sum_z_dr2 < 0.003563
        - 0.06405477 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -6.4%  sj3_dr_max < 0.1426
        + 0.05626042 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.2973626   # +5.6%  mass < 53.33 and centroid_offset < 0.02686
        - 0.05050085 * max(0.0, 53.33237 - Q.mass) / 19.36004   # -5.1%  mass < 53.33
        - 0.04892091 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -4.9%  lam1 < 0.005954
        + 0.04876734 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +4.9%  sj3_dr_max < 0.2134
        - 0.04673344 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -4.7%  mass_over_sum_pt < 0.07269
        + 0.04162364 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +4.2%  max_dr < 0.1118
        + 0.04006763 * max(0.0, Q.lam1 - 0.005433361) / 0.00267714   # +4.0%  lam1 > 0.005433
        + 0.03813778 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +3.8%  centroid_offset < 0.01838
        + 0.03128921 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # +3.1%  tau1 < 0.0437
        - 0.02811412 * max(0.0, 0.05464922 - Q.sum_z_dr) / 0.01532978   # -2.8%  sum_z_dr < 0.05465
        - 0.0264951 * max(0.0, Q.lam1 - 0.003377388) / 0.003672352   # -2.6%  lam1 > 0.003377
        - 0.02642444 * max(0.0, Q.sum_z_dr - 0.0717028) / 0.01201981   # -2.6%  sum_z_dr > 0.0717
        - 0.02641538 * max(0.0, 2.371297e-05 - Q.e3) / 1.105328e-05   # -2.6%  e3 < 2.371e-05
        - 0.02580792 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) / 6.087676   # -2.6%  sj3_pair_mass_max < 24.23
        + 0.02099845 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +2.1%  e2 > 0.04447
        - 0.0179281 * max(0.0, 0.0284695 - Q.C3) / 0.008714909   # -1.8%  C3 < 0.02847
        + 0.01732258 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt) / 5.175436   # +1.7%  mass < 53.33 and log_sum_pt < 6.843
        - 0.01642861 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1) / 0.1931178   # -1.6%  centroid_offset < 0.01838 and pt_1 < 159.2
        - 0.01536455 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.008712838   # -1.5%  centroid_offset < 0.01838 and n_for_90pct > 5
        + 0.01138293 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd) / 0.0002178654   # +1.1%  centroid_offset < 0.01838 and z_2nd < 0.2056
        - 0.01124032 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266) / 0.0005312449   # -1.1%  lam1 > 0.005433 and eccentricity > 0.7117
        - 0.01018475 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -1.0%  e2 > 0.03556
        - 0.007270261 * Q.z_dr_0p2_0p4 / 0.02920532   # -0.7%  z_dr_0p2_0p4
        + 0.005112667 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.5%  lam2 > 0.001131
        + 0.004166259 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.4%  n_dr_0p2_0p4 > 1
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 9.184;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.184016 * (0.2053524
        - 0.1496693 * Q.e2 / 0.02863215   # -15.0%  e2
        + 0.1477709 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # +14.8%  tau1 > 0.05357
        + 0.1405817 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # +14.1%  sum_z_dr2 > 0.00752
        + 0.1195049 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +12.0%  mass < 76.66
        - 0.09900653 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -9.9%  lam1 < 0.004184
        - 0.08144976 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -8.1%  lam1 < 0.001504
        - 0.06795693 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # -6.8%  lam1 > 0.00733
        - 0.05802347 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -5.8%  LHA > 0.3033
        + 0.03748612 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # +3.7%  sj3_pair_mass_min > 11.05
        - 0.03049235 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -3.0%  e3 > 3.892e-05
        + 0.02437704 * max(0.0, Q.lam2 - 0.0003061234) / 0.0004216599   # +2.4%  lam2 > 0.0003061
        + 0.02037062 * max(0.0, Q.C2_b2 - 0.009032972) / 0.001703043   # +2.0%  C2_b2 > 0.009033
        - 0.01260279 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2) / 0.0002929492   # -1.3%  lam1 > 0.00733 and D2_b2 < 0.3809
        - 0.01070769 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -1.1%  lam2 > 0.003408
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 28.15;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.14973 * (-0.1309795
        + 0.2390933 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +23.9%  lam1_plus_lam2 < 0.008678
        - 0.1212184 * max(0.0, 0.169029 - Q.sj3_dr_max) / 0.04876366   # -12.1%  sj3_dr_max < 0.169
        + 0.09188584 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +9.2%  sum_z_dr2 < 0.01324
        + 0.08481419 * max(0.0, 0.2623172 - Q.sj3_dr_max) / 0.1129006   # +8.5%  sj3_dr_max < 0.2623
        - 0.07425823 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -7.4%  lam1 < 0.008376
        + 0.07295647 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +7.3%  centroid_offset < 0.03776
        - 0.06266988 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # -6.3%  sum_z_dr < 0.0717
        - 0.05854572 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -5.9%  sum_zz_dr2 < 0.008169
        + 0.03302851 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +3.3%  z_7 > 0.01686
        - 0.03136305 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # -3.1%  tau1 < 0.09539
        + 0.02718655 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # +2.7%  max_dr < 0.1452
        - 0.02530727 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt) / 3.513015   # -2.5%  centroid_offset < 0.03776 and sum_pt < 901.6
        - 0.02038952 * max(0.0, 0.02054282 - Q.sum_z_dr) / 0.002592388   # -2.0%  sum_z_dr < 0.02054
        + 0.01968438 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +2.0%  sj3_dr_max < 0.1426
        + 0.01266056 * max(0.0, 15.45403 - Q.mass) / 2.385786   # +1.3%  mass < 15.45
        - 0.01087225 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # -1.1%  centroid_offset > 0.0499 and D2 < 2.844
        + 0.01054278 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +1.1%  sj2_dr > 0.2688
        + 0.003523075 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.5502779 - Q.tau32) / 0.0001096908   # +0.4%  centroid_offset > 0.0499 and tau32 < 0.5503
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.4287;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.4287394 * (-3.255479
        + 0.5105466 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # +51.1%  sum_z_dr2 > 0.01883
        - 0.3062212 * max(0.0, Q.mass_over_sum_pt - 0.1309286) / 0.002541185   # -30.6%  mass_over_sum_pt > 0.1309
        + 0.1832322 * max(0.0, Q.mass - 88.15578) / 0.7232262   # +18.3%  mass > 88.16
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 10.22;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.22003 * (-0.04388818
        + 0.3955081 * max(0.0, 0.1484084 - Q.sum_z_dr) / 0.0905999   # +39.6%  sum_z_dr < 0.1484
        + 0.1716304 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +17.2%  lam1 < 0.01643
        - 0.09606646 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -9.6%  sum_z_dr < 0.1484 and pt_7 < 38.53
        - 0.09507849 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -9.5%  e2 < 0.08001
        + 0.0895096 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7) / 781.5808   # +9.0%  sum_pt_top5 > 658.1 and pt_7 < 40.04
        - 0.08830813 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -8.8%  sum_z_dr < 0.1484 and log_sum_pt < 6.804
        - 0.03266014 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # -3.3%  z_7 < 0.02807
        + 0.02496303 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.tau2 - 0.008780509) / 0.000269945   # +2.5%  sum_z_dr < 0.1484 and tau2 > 0.008781
        - 0.006275635 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6) / 104.494   # -0.6%  sum_pt > 988.4 and pt_6 < 62.25
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 45.55;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 45.54989 * (-0.06985216
        - 0.1190551 * max(0.0, Q.lam1_plus_lam2 - 0.006679471) / 0.002702003   # -11.9%  lam1_plus_lam2 > 0.006679
        + 0.0962873 * max(0.0, Q.sum_z_dr - 0.02689598) / 0.03613842   # +9.6%  sum_z_dr > 0.0269
        - 0.07463894 * max(0.0, Q.e2 - 0.01655442) / 0.01596508   # -7.5%  e2 > 0.01655
        - 0.07445453 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -7.4%  sum_z_dr2 > 0.00752
        + 0.05881207 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # +5.9%  sj2_dr > 0.1592
        + 0.05741479 * max(0.0, Q.sum_z_dr - 0.04081947) / 0.0270611   # +5.7%  sum_z_dr > 0.04082
        - 0.05427611 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # -5.4%  sum_z_dr > 0.08724
        + 0.05021622 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +5.0%  mass_over_sum_pt > 0.07992
        + 0.04915592 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +4.9%  mass_over_sum_pt > 0.08475
        + 0.03293384 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +3.3%  e2 > 0.05028
        + 0.03070829 * max(0.0, Q.lam1_plus_lam2 - 0.003562611) / 0.004066872   # +3.1%  lam1_plus_lam2 > 0.003563
        - 0.03052456 * max(0.0, Q.sj2_dr - 0.06154135) / 0.1002688   # -3.1%  sj2_dr > 0.06154
        - 0.02842221 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -2.8%  mass_over_sum_pt > 0.09041
        - 0.02539455 * max(0.0, 74.57663 - Q.sd_mass) / 42.04056   # -2.5%  sd_mass < 74.58
        + 0.01971271 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # +2.0%  lam1_plus_lam2 > 0.00559
        + 0.01842135 * max(0.0, Q.sum_z_dr - 0.08065885) / 0.009330634   # +1.8%  sum_z_dr > 0.08066
        + 0.01755293 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +1.8%  e2 > 0.04111
        - 0.01739022 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -1.7%  centroid_offset > 0.0499
        - 0.01710419 * max(0.0, Q.sum_zz_dr2 - 0.0030133) / 0.003940214   # -1.7%  sum_zz_dr2 > 0.003013
        + 0.01570613 * max(0.0, 49.91626 - Q.sd_mass) / 22.60916   # +1.6%  sd_mass < 49.92
        - 0.01504444 * max(0.0, Q.sj2_dr - 0.2001708) / 0.02317349   # -1.5%  sj2_dr > 0.2002
        + 0.01421976 * max(0.0, Q.sum_zz_dr2 - 0.002074109) / 0.004484017   # +1.4%  sum_zz_dr2 > 0.002074
        - 0.01210974 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # -1.2%  z_dr_0p05_0p1 > 0.7509
        - 0.01205446 * max(0.0, Q.sum_z_dr - 0.1019409) / 0.005400003   # -1.2%  sum_z_dr > 0.1019
        - 0.0116174 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -1.2%  e2 > 0.03556
        + 0.01092691 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067297   # +1.1%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        + 0.01029808 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # +1.0%  lam1_plus_lam2 > 0.01324
        - 0.009445925 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -0.9%  planar_flow < 0.1115 and lam1 < 0.00733
        + 0.008131968 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 3.351526e-05   # +0.8%  planar_flow < 0.1115 and lam1_plus_lam2 < 0.006097
        - 0.007969303 * max(0.0, Q.centroid_offset - 0.02685622) / 0.003254849   # -0.8%  centroid_offset > 0.02686
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 40.83;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 40.83069 * (-0.02511489
        + 0.1343815 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +13.4%  lam1_plus_lam2 < 0.01324
        - 0.09146798 * max(0.0, 0.007520088 - Q.sum_z_dr2) / 0.00354499   # -9.1%  sum_z_dr2 < 0.00752
        - 0.08004605 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -8.0%  sum_zz_dr2 < 0.008169
        + 0.07536083 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +7.5%  e2 < 0.04111
        + 0.0729229 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +7.3%  sj2_dr < 0.1592
        - 0.06856001 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 0.002544039   # -6.9%  lam1_plus_lam2 < 0.006097
        + 0.06497302 * max(0.0, 0.01882765 - Q.lam1_plus_lam2) / 0.01314296   # +6.5%  lam1_plus_lam2 < 0.01883
        + 0.06447569 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +6.4%  mass_over_sum_pt_sq < 0.007183
        - 0.0510512 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # -5.1%  sum_zz_dr2 < 0.01166
        - 0.04128874 * max(0.0, 0.1492731 - Q.sj2_dr) / 0.04374278   # -4.1%  sj2_dr < 0.1493
        - 0.03410665 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -3.4%  sum_z_dr < 0.1019
        - 0.03329694 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -3.3%  centroid_offset < 0.01838
        - 0.02733377 * max(0.0, 0.2001708 - Q.sj2_dr) / 0.07331403   # -2.7%  sj2_dr < 0.2002
        - 0.02595516 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # -2.6%  N2 < 0.2233
        + 0.02417609 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # +2.4%  lam1 < 0.005433
        - 0.0219542 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # -2.2%  mass_over_sum_pt < 0.1309
        + 0.01648489 * max(0.0, 0.006756161 - Q.sum_z_dr2_top3) / 0.003650633   # +1.6%  sum_z_dr2_top3 < 0.006756
        - 0.01625163 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -1.6%  lam1 < 0.008376
        + 0.01541619 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +1.5%  N2 < 0.2233 and sum_pt_top5 > 430.8
        - 0.01521506 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2512159 - Q.dr01) / 0.001381924   # -1.5%  centroid_offset < 0.01838 and dr01 < 0.2512
        + 0.01435298 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2) / 0.003993721   # +1.4%  mass_over_sum_pt < 0.1309 and D2 < 0.746
        - 0.006365379 * max(0.0, 0.007520088 - Q.sum_z_dr2) * max(0.0, 0.7459513 - Q.D2) / 9.127003e-05   # -0.6%  sum_z_dr2 < 0.00752 and D2 < 0.746
        + 0.004563122 * max(0.0, 0.0005124533 - Q.sum_z_dr2_top2) / 0.0001104529   # +0.5%  sum_z_dr2_top2 < 0.0005125
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0381031512605041, 0.7890733193277311, 2.4981823529411766, 0.9621460084033614, 1.3581567226890756, 2.3532460084033615, 1.170672268907563, 2.0014203781512605, 0.30194348739495797, 3.24991512605042, 2.0580047268907564, 2.13112006302521, 0.06924810924369748, 3.316179201680672, 0.5088736344537815, 0.29606953781512607]
T = [2.7141705783876047, 1.5037698004201678, 3.4608129218093486, 2.6798865365677518, 3.0446236114758403]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +40%, n9 +21%, n5 -16%, n1 +11%, n0 -6%, n6 +5% ...
            + 0.3954938 * h[2] / H_AVG[2]
            + 0.2058011 * h[9] / H_AVG[9]
            - 0.1625667 * h[5] / H_AVG[5]
            + 0.1135639 * h[1] / H_AVG[1]
            - 0.05976176 * h[0] / H_AVG[0]
            + 0.04717547 * h[6] / H_AVG[6]
            - 0.01563734 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +55%, n10 -17%, n6 +10%, n4 -8%, n5 +7%, n8 +1% ...
            + 0.5487368 * h[9] / H_AVG[9]
            - 0.1710705 * h[10] / H_AVG[10]
            + 0.09731146 * h[6] / H_AVG[6]
            - 0.084672 * h[4] / H_AVG[4]
            + 0.07335458 * h[5] / H_AVG[5]
            + 0.01254944 * h[8] / H_AVG[8]
            + 0.01230531 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +23%, n3 -14%, n7 +13%, n14 -11%, n6 -11%, n0 +10% ...
            + 0.2309197 * h[11] / H_AVG[11]
            - 0.1390058 * h[3] / H_AVG[3]
            + 0.1265052 * h[7] / H_AVG[7]
            - 0.1102791 * h[14] / H_AVG[14]
            - 0.1057078 * h[6] / H_AVG[6]
            + 0.103111 * h[0] / H_AVG[0]
            + 0.06737401 * h[13] / H_AVG[13]
            - 0.05881503 * h[15] / H_AVG[15]
            - 0.02934566 * h[9] / H_AVG[9]
            - 0.0218116 * h[8] / H_AVG[8]
            - 0.007125072 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -20%, n6 -16%, n14 +7%, n13 +7%, n4 +4% ...
            + 0.3500767 * h[7] / H_AVG[7]
            - 0.2019515 * h[3] / H_AVG[3]
            - 0.1638137 * h[6] / H_AVG[6]
            + 0.07120735 * h[14] / H_AVG[14]
            + 0.0676721 * h[13] / H_AVG[13]
            + 0.03959347 * h[4] / H_AVG[4]
            - 0.03789707 * h[9] / H_AVG[9]
            + 0.03680535 * h[1] / H_AVG[1]
            - 0.01726225 * h[15] / H_AVG[15]
            + 0.01372053 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -44%, n10 +25%, n5 -19%, n4 +6%, n3 +2%, n8 +2% ...
            - 0.4424842 * h[13] / H_AVG[13]
            + 0.2534802 * h[10] / H_AVG[10]
            - 0.1932296 * h[5] / H_AVG[5]
            + 0.05576045 * h[4] / H_AVG[4]
            + 0.01975092 * h[3] / H_AVG[3]
            + 0.01859488 * h[8] / H_AVG[8]
            - 0.0113722 * h[12] / H_AVG[12]
            + 0.005327543 * h[0] / H_AVG[0]
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
