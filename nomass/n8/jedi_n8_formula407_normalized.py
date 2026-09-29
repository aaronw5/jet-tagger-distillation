"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.9%   (on for 90% of jets)
  neuron  9:  11.7%   (on for 69% of jets)
  neuron  7:  10.0%   (on for 66% of jets)
  neuron  6:   8.3%   (on for 35% of jets)
  neuron 10:   7.9%   (on for 74% of jets)
  neuron  3:   7.9%   (on for 23% of jets)
  neuron  5:   6.9%   (on for 53% of jets)
  neuron  2:   6.6%   (on for 86% of jets)
  neuron 11:   6.4%   (on for 76% of jets)
  neuron 14:   4.5%   (on for 28% of jets)
  neuron  0:   3.7%   (on for 43% of jets)
  neuron  4:   3.4%   (on for 72% of jets)
  neuron  1:   3.2%   (on for 62% of jets)
  neuron 15:   3.1%   (on for 31% of jets)
  neuron  8:   1.3%   (on for 42% of jets)
  neuron 12:   0.3%   (on for 4% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.8% (the network: 65.8%); same class as the network for 87.6% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr12                   ΔR between particles 1 and 2
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
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
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        max_dr=max(dr[i] for i in real),
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_5=z[5] * dr[5],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 37.81;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 37.80527 * (0.1929744
        - 0.2408727 * Q.log_sum_pt / 6.542932   # -24.1%  log_sum_pt
        + 0.110914 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +11.1%  width < 0.008678
        + 0.06793004 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +6.8%  girth2 < 0.01324
        - 0.05052127 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -5.1%  girth < 0.08724
        + 0.04371156 * max(0.0, 0.1116471 - Q.sd_rg) / 0.04385005   # +4.4%  sd_rg < 0.1116
        - 0.03985472 * max(0.0, 0.1773029 - Q.sd_rg) / 0.08115541   # -4.0%  sd_rg < 0.1773
        - 0.03689256 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -3.7%  lam1 < 0.008376
        - 0.03633937 * max(0.0, 0.00718279 - Q.e2_sq) / 0.003528787   # -3.6%  e2_sq < 0.007183
        + 0.0343022 * max(0.0, 0.06345984 - Q.tau1) / 0.02211428   # +3.4%  tau1 < 0.06346
        + 0.03287078 * max(0.0, 0.01882765 - Q.girth2) / 0.01314296   # +3.3%  girth2 < 0.01883
        - 0.02968192 * max(0.0, 0.005590289 - Q.girth2) / 0.002229311   # -3.0%  girth2 < 0.00559
        - 0.02956443 * max(0.0, 0.003562611 - Q.width) / 0.001184249   # -3.0%  width < 0.003563
        + 0.02761232 * max(0.0, 631.275 - Q.sum_pt_top5) / 94.34106   # +2.8%  sum_pt_top5 < 631.3
        + 0.02589926 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +2.6%  e3 < 8.148e-05
        - 0.02530706 * max(0.0, 0.01357153 - Q.tau2) / 0.005299336   # -2.5%  tau2 < 0.01357
        - 0.02474078 * max(0.0, 788.4484 - Q.sum_pt) / 112.1576   # -2.5%  sum_pt < 788.4
        - 0.01789571 * max(0.0, 0.007929074 - Q.girth2_top3) / 0.004560262   # -1.8%  girth2_top3 < 0.007929
        - 0.01547414 * max(0.0, 0.004372139 - Q.width) / 0.00156571   # -1.5%  width < 0.004372
        + 0.01170258 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 1.129616 - Q.D2_b2) / 0.04715373   # +1.2%  planar_flow < 0.1484 and D2_b2 < 1.13
        + 0.01135507 * max(0.0, 0.01882765 - Q.girth2) * max(0.0, 0.09029177 - Q.M3) / 0.0001995224   # +1.1%  girth2 < 0.01883 and M3 < 0.09029
        + 0.01080342 * max(0.0, Q.n_dr_0_0p05 - 5.0) / 1.093146   # +1.1%  n_dr_0_0p05 > 5
        - 0.01013897 * max(0.0, 0.01882765 - Q.girth2) * max(0.0, Q.centroid_offset - 0.009480685) / 8.306781e-05   # -1.0%  girth2 < 0.01883 and centroid_offset > 0.009481
        - 0.0100892 * max(0.0, 0.06545715 - Q.sd_rg) / 0.02200467   # -1.0%  sd_rg < 0.06546
        - 0.009757798 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -1.0%  sum_pt < 739.5
        - 0.009399735 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.9%  centroid_offset > 0.0499
        + 0.00850319 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +0.9%  planar_flow < 0.1484
        - 0.00677838 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, 0.380911 - Q.D2_b2) / 12.71118   # -0.7%  sum_pt < 788.4 and D2_b2 < 0.3809
        - 0.005107234 * max(0.0, 0.380911 - Q.D2_b2) * max(0.0, Q.pt_5 - 24.57812) / 3.196263   # -0.5%  D2_b2 < 0.3809 and pt_5 > 24.58
        + 0.005101648 * max(0.0, Q.eccentricity - 0.9884745) / 0.001961982   # +0.5%  eccentricity > 0.9885
        + 0.004773005 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, Q.M2 - 0.02563286) / 4.402579   # +0.5%  sum_pt < 788.4 and M2 > 0.02563
        - 0.002529932 * max(0.0, 0.06345984 - Q.tau1) * max(0.0, Q.dr0_6 - 0.1755206) / 7.050329e-05   # -0.3%  tau1 < 0.06346 and dr0_6 > 0.1755
        + 0.001918268 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.dr_6 - 0.1695089) / 1.449123e-07   # +0.2%  e3 < 8.148e-05 and dr_6 > 0.1695
        - 0.0009648743 * max(0.0, 0.01517359 - Q.girth) * max(0.0, Q.dr0_6 - 0.1979161) / 5.112851e-07   # -0.1%  girth < 0.01517 and dr0_6 > 0.1979
        + 0.0006918468 * max(0.0, 0.004372139 - Q.width) * max(0.0, Q.dr0_6 - 0.1755206) / 1.823868e-06   # +0.1%  width < 0.004372 and dr0_6 > 0.1755
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 15.72;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.71757 * (0.03676688
        - 0.1261605 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # -12.6%  width < 0.008678
        + 0.09733797 * max(0.0, 0.005834489 - Q.e2_sq) / 0.002579453   # +9.7%  e2_sq < 0.005834
        + 0.08641778 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1442881 - Q.dr_0) / 0.0005125595   # +8.6%  lam1 < 0.008376 and dr_0 < 0.1443
        - 0.07288085 * max(0.0, 0.00609665 - Q.girth2) / 0.002544039   # -7.3%  girth2 < 0.006097
        - 0.06836852 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # -6.8%  girth < 0.1019
        + 0.06577914 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # +6.6%  tau1 < 0.1028
        + 0.06445478 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +6.4%  log_sum_pt > 6.378
        + 0.06310586 * max(0.0, Q.log_sum_pt - 6.572938) / 0.08631512   # +6.3%  log_sum_pt > 6.573
        - 0.05082702 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -5.1%  z_7 < 0.06165
        - 0.04769588 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.08082334 - Q.dr_0) / 0.01037756   # -4.8%  log_sum_pt > 6.378 and dr_0 < 0.08082
        - 0.04696733 * max(0.0, 0.05240025 - Q.z_7) / 0.008882498   # -4.7%  z_7 < 0.0524
        + 0.03568811 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.01403324 - Q.girth2_top2) / 0.0001680113   # +3.6%  z_7 < 0.06165 and girth2_top2 < 0.01403
        + 0.03326746 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +3.3%  pt_7 > 34.53
        - 0.03175774 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -3.2%  zdr_0 < 0.02118
        + 0.02905042 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.0001869378 - Q.e3) / 2.398316e-06   # +2.9%  z_7 < 0.06165 and e3 < 0.0001869
        - 0.02788923 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # -2.8%  tau1 < 0.07284
        + 0.0145559 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +1.5%  lam1 < 0.002464
        - 0.01413153 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.007929074 - Q.girth2_top3) / 0.0005408911   # -1.4%  log_sum_pt > 6.573 and girth2_top3 < 0.007929
        - 0.01387275 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # -1.4%  pt_7 > 34.53 and tau2 < 0.01713
        + 0.009791207 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.003060958   # +1.0%  sj2_dr > 0.1873 and sj3_dr_min < 0.209
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 19.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.23308 * (0.0328239
        + 0.117852 * Q.pt_7 / 34.64819   # +11.8%  pt_7
        - 0.09667545 * max(0.0, 0.0423228 - Q.C2) * max(0.0, 0.02415398 - Q.C2_b2) / 0.0004792616   # -9.7%  C2 < 0.04232 and C2_b2 < 0.02415
        - 0.08824639 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -8.8%  pt_7 < 53.44
        - 0.08799023 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # -8.8%  z_7 > 0.02321
        + 0.08475078 * max(0.0, 6.701242 - Q.log_sum_pt) / 0.1935491   # +8.5%  log_sum_pt < 6.701
        + 0.07528939 * max(0.0, 0.0423228 - Q.C2) / 0.02012665   # +7.5%  C2 < 0.04232
        + 0.06003801 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +6.0%  width < 0.01324
        + 0.04490132 * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.09151624   # +4.5%  sj3_dr_min < 0.1278
        - 0.04335772 * max(0.0, Q.LHA - 0.111565) / 0.1352185   # -4.3%  LHA > 0.1116
        - 0.04270071 * max(0.0, 0.04447357 - Q.e2) / 0.02019334   # -4.3%  e2 < 0.04447
        + 0.03854085 * max(0.0, Q.N2 - 0.071455) / 0.1508638   # +3.9%  N2 > 0.07145
        + 0.03545844 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.2507612 - Q.max_dr) / 0.00062784   # +3.5%  lam1 < 0.00733 and max_dr < 0.2508
        - 0.02923858 * max(0.0, Q.z_7 - 0.02807091) / 0.02541111   # -2.9%  z_7 > 0.02807
        + 0.02891257 * max(0.0, 0.02383244 - Q.zdr_0) / 0.01109706   # +2.9%  zdr_0 < 0.02383
        - 0.02851001 * max(0.0, 6.701242 - Q.log_sum_pt) * max(0.0, 1.129616 - Q.D2_b2) / 0.1081022   # -2.9%  log_sum_pt < 6.701 and D2_b2 < 1.13
        + 0.02298832 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 1.129616 - Q.D2_b2) / 0.07303201   # +2.3%  log_sum_pt < 6.606 and D2_b2 < 1.13
        - 0.01763046 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # -1.8%  log_sum_pt < 6.606
        + 0.01404626 * max(0.0, 615.875 - Q.sum_pt) / 30.07766   # +1.4%  sum_pt < 615.9
        - 0.01222304 * max(0.0, 0.0030133 - Q.e2_sq) / 0.001065944   # -1.2%  e2_sq < 0.003013
        - 0.007617742 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.8%  sum_pt > 988.4
        - 0.004976381 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # -0.5%  girth < 0.007674
        - 0.004652791 * max(0.0, 0.005576073 - Q.centroid_offset) * max(0.0, 0.01685631 - Q.C3) / 3.692635e-06   # -0.5%  centroid_offset < 0.005576 and C3 < 0.01686
        + 0.004416631 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 1.129616 - Q.D2_b2) / 3.947956   # +0.4%  sum_pt_top5 > 791.1 and D2_b2 < 1.13
        + 0.003839184 * max(0.0, Q.sum_pt - 937.0312) / 8.087955   # +0.4%  sum_pt > 937
        + 0.003517049 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.4%  log_sum_pt > 6.896
        - 0.001629653 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 1.59265 - Q.D2_b2) / 0.00192904   # -0.2%  log_sum_pt > 6.896 and D2_b2 < 1.593
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 17.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.23527 * (-0.1038925
        - 0.1294397 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # -12.9%  girth2 > 0.008678
        + 0.09386446 * max(0.0, Q.e2_sq - 0.00718279) / 0.002233568   # +9.4%  e2_sq > 0.007183
        + 0.09050156 * max(0.0, Q.width - 0.00609665) / 0.002892623   # +9.1%  width > 0.006097
        - 0.08859206 * max(0.0, Q.e2_sq - 0.006390125) / 0.002450953   # -8.9%  e2_sq > 0.00639
        + 0.0802897 * max(0.0, Q.girth - 0.06663269) / 0.01393206   # +8.0%  girth > 0.06663
        - 0.05878565 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 7.0 - Q.n_dr_0p05_0p1) / 0.1780556   # -5.9%  sj2_dr > 0.1683 and n_dr_0p05_0p1 < 7
        + 0.05577055 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # +5.6%  sj2_dr > 0.1779
        + 0.0479815 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +4.8%  lam1 > 0.008376
        - 0.04255766 * max(0.0, Q.tau1 - 0.0811449) / 0.01489378   # -4.3%  tau1 > 0.08114
        + 0.04227343 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.06297984 - Q.tau2) / 0.001263954   # +4.2%  sj2_dr > 0.1683 and tau2 < 0.06298
        + 0.03079761 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +3.1%  sj2_dr > 0.1683
        - 0.03075229 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.eccentricity - 0.9458207) / 0.0008867862   # -3.1%  sj2_dr > 0.1683 and eccentricity > 0.9458
        + 0.0307035 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +3.1%  tau1 > 0.1136
        + 0.02985449 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, Q.eccentricity - 0.9458207) / 0.000763705   # +3.0%  sj2_dr > 0.1779 and eccentricity > 0.9458
        - 0.02425746 * max(0.0, Q.width - 0.01323868) / 0.001442684   # -2.4%  width > 0.01324
        + 0.02153153 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # +2.2%  girth > 0.08724
        - 0.01686912 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 988.4078 - Q.sum_pt) / 12.22843   # -1.7%  sj2_dr > 0.1683 and sum_pt < 988.4
        - 0.01419241 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001864148 - Q.girth2_top2) / 7.226227e-06   # -1.4%  sj2_dr > 0.1779 and girth2_top2 < 0.001864
        - 0.0128573 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -1.3%  lam1 > 0.012
        - 0.01145537 * max(0.0, Q.width - 0.00609665) * max(0.0, Q.log_sum_pt - 6.192222) / 0.0004573947   # -1.1%  width > 0.006097 and log_sum_pt > 6.192
        + 0.009746037 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +1.0%  sj2_dr > 0.3004
        + 0.009299096 * max(0.0, Q.lam2 - 0.000537286) / 0.0003825703   # +0.9%  lam2 > 0.0005373
        - 0.008525673 * max(0.0, Q.sj2_dr - 0.3003793) * max(0.0, Q.z_dr_0p05_0p1 - 0.08878489) / 0.001005291   # -0.9%  sj2_dr > 0.3004 and z_dr_0p05_0p1 > 0.08878
        - 0.007216902 * max(0.0, Q.width - 0.00609665) * max(0.0, Q.n_pt_above_50 - 3.0) / 0.004753839   # -0.7%  width > 0.006097 and n_pt_above_50 > 3
        - 0.005514961 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # -0.6%  sj2_dr > 0.2688
        - 0.004966105 * max(0.0, Q.e2_sq - 0.01165737) / 0.001433269   # -0.5%  e2_sq > 0.01166
        - 0.001403846 * max(0.0, Q.sj2_dr - 0.3003793) * max(0.0, 0.01618699 - Q.D2_b2) / 2.371565e-06   # -0.1%  sj2_dr > 0.3004 and D2_b2 < 0.01619
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 29.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.58678 * (-0.02016235
        - 0.1269971 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -12.7%  lam2 < 0.001131
        + 0.1223515 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +12.2%  e3 < 0.0001869
        - 0.1219936 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -12.2%  girth2 < 0.008678
        + 0.1030545 * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.166579   # +10.3%  sj3_dr_min < 0.209
        + 0.06824221 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # +6.8%  e2_sq < 0.01166
        + 0.05231297 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +5.2%  N2 < 0.2233
        + 0.05026369 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # +5.0%  C2_b2 < 0.009033
        - 0.04549889 * max(0.0, 0.006679471 - Q.girth2) / 0.00293624   # -4.5%  girth2 < 0.006679
        - 0.04257779 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.4007947 - Q.planar_flow) / 0.01755893   # -4.3%  N2 < 0.2233 and planar_flow < 0.4008
        + 0.04211019 * max(0.0, 0.002635418 - Q.girth2) / 0.0007941346   # +4.2%  girth2 < 0.002635
        - 0.0377485 * max(0.0, 0.01165737 - Q.e2_sq) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.0003843383   # -3.8%  e2_sq < 0.01166 and z_dr_0p2_0p4 < 0.05644
        - 0.03484997 * max(0.0, 0.04447357 - Q.e2) / 0.02019334   # -3.5%  e2 < 0.04447
        + 0.03055835 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +3.1%  lam1 < 0.00484
        - 0.02360299 * max(0.0, Q.sj3_dr_max - 0.213399) / 0.03061279   # -2.4%  sj3_dr_max > 0.2134
        + 0.0203191 * max(0.0, 0.07608178 - Q.girth) / 0.02796784   # +2.0%  girth < 0.07608
        + 0.01818947 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +1.8%  sj3_dr_max > 0.179
        - 0.01710941 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # -1.7%  lam1 < 0.002464
        + 0.0147315 * max(0.0, 0.006299534 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.01096064) / 1.639932e-05   # +1.5%  girth2_top2 < 0.0063 and centroid_offset > 0.01096
        - 0.01252621 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, 813.4156 - Q.sum_pt) / 0.01550241   # -1.3%  e3 < 0.0001869 and sum_pt < 813.4
        - 0.006017262 * max(0.0, Q.C2 - 0.02384388) / 0.01155414   # -0.6%  C2 > 0.02384
        + 0.005696322 * max(0.0, Q.sj3_dr_max - 0.3456459) / 0.006655619   # +0.6%  sj3_dr_max > 0.3456
        - 0.003248565 * max(0.0, Q.sj3_dr_max - 0.3456459) * max(0.0, 1.129616 - Q.D2_b2) / 0.001618684   # -0.3%  sj3_dr_max > 0.3456 and D2_b2 < 1.13
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 15.92;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.92125 * (0.01597487
        + 0.09414506 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +9.4%  e3 < 0.0005117
        - 0.07654858 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -7.7%  sj3_dr_max < 0.3012
        - 0.07219106 * max(0.0, 0.04649465 - Q.dr_0) / 0.01415326   # -7.2%  dr_0 < 0.04649
        - 0.07107785 * max(0.0, 715.4688 - Q.sum_pt) / 69.91196   # -7.1%  sum_pt < 715.5
        - 0.07102186 * max(0.0, 0.03117077 - Q.centroid_offset) / 0.01649145   # -7.1%  centroid_offset < 0.03117
        - 0.06928056 * max(0.0, 430.75 - Q.sum_pt_top5) / 14.05246   # -6.9%  sum_pt_top5 < 430.8
        + 0.06639443 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +6.6%  LHA < 0.2161
        + 0.05003239 * max(0.0, 0.0009641429 - Q.girth2) / 0.0002052288   # +5.0%  girth2 < 0.0009641
        + 0.04867778 * max(0.0, 0.03556091 - Q.e2) / 0.01371972   # +4.9%  e2 < 0.03556
        - 0.04612087 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -4.6%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.0432496 * max(0.0, 0.002635418 - Q.girth2) / 0.0007941346   # +4.3%  girth2 < 0.002635
        + 0.03792814 * max(0.0, 0.03117077 - Q.centroid_offset) * max(0.0, 0.01364517 - Q.tau3) / 0.0001472753   # +3.8%  centroid_offset < 0.03117 and tau3 < 0.01365
        - 0.03646537 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.09543973 - Q.z_6) / 0.001514041   # -3.6%  LHA < 0.2161 and z_6 < 0.09544
        + 0.03150543 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +3.2%  girth2 < 0.00502
        + 0.02725884 * max(0.0, 0.002635418 - Q.girth2) * max(0.0, 0.01627885 - Q.centroid_offset) / 7.136253e-06   # +2.7%  girth2 < 0.002635 and centroid_offset < 0.01628
        + 0.02071385 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +2.1%  z_7 < 0.02807
        - 0.01930528 * max(0.0, 0.04649465 - Q.dr_0) * max(0.0, 2.080881 - Q.N3) / 0.008324961   # -1.9%  dr_0 < 0.04649 and N3 < 2.081
        + 0.01663577 * max(0.0, 0.06081235 - Q.z_6) / 0.009654642   # +1.7%  z_6 < 0.06081
        + 0.01419605 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 0.04678345 - Q.absphi_0) / 0.000232957   # +1.4%  z_7 < 0.0494 and absphi_0 < 0.04678
        + 0.01349162 * max(0.0, 715.4688 - Q.sum_pt) * max(0.0, 0.007798268 - Q.zdr_0) / 0.05114364   # +1.3%  sum_pt < 715.5 and zdr_0 < 0.007798
        + 0.0113573 * max(0.0, 0.03556091 - Q.e2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.001328529   # +1.1%  e2 < 0.03556 and planar_flow < 0.3221
        - 0.01088228 * max(0.0, 0.1767241 - Q.LHA) * max(0.0, 0.006292091 - Q.zdr_0) / 7.498285e-05   # -1.1%  LHA < 0.1767 and zdr_0 < 0.006292
        - 0.01040201 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # -1.0%  pt_7 > 34.53
        - 0.01012464 * max(0.0, Q.n_pt_above_50 - 6.0) / 0.2884353   # -1.0%  n_pt_above_50 > 6
        - 0.01000052 * max(0.0, 0.03556091 - Q.e2) * max(0.0, Q.log_sum_pt - 6.896095) / 0.0001063848   # -1.0%  e2 < 0.03556 and log_sum_pt > 6.896
        - 0.008291181 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.8%  log_sum_pt > 6.896
        - 0.007182115 * max(0.0, 20.125 - Q.pt_7) / 0.5468342   # -0.7%  pt_7 < 20.12
        + 0.003413225 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # +0.3%  z_6 < 0.0216
        + 0.002106335 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 5.507482e-05 - Q.mean_eta2) / 6.203155e-08   # +0.2%  log_sum_pt > 6.896 and mean_eta2 < 5.507e-05
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 35.4;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.40405 * (0.05752745
        - 0.2408127 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -24.1%  girth2 < 0.008678
        - 0.09802688 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # -9.8%  width < 0.01324
        + 0.08658584 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +8.7%  lam1 < 0.012
        + 0.06871521 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +6.9%  tau1 < 0.1136
        + 0.05898541 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +5.9%  lam2 < 0.003408
        - 0.05053582 * max(0.0, 0.1598486 - Q.max_dr) / 0.05771655   # -5.1%  max_dr < 0.1598
        + 0.04922662 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # +4.9%  e2_sq < 0.008169
        - 0.04159662 * max(0.0, Q.centroid_offset - 0.01627885) * max(0.0, 988.4078 - Q.sum_pt) / 2.602791   # -4.2%  centroid_offset > 0.01628 and sum_pt < 988.4
        - 0.03707585 * max(0.0, Q.sj3_dr_max - 0.07264452) / 0.1084421   # -3.7%  sj3_dr_max > 0.07264
        + 0.03446227 * max(0.0, 0.02689598 - Q.girth) / 0.004330137   # +3.4%  girth < 0.0269
        + 0.02998637 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +3.0%  lam1 < 0.008376
        + 0.02835048 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +2.8%  sj3_dr_max > 0.179
        + 0.02822844 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +2.8%  centroid_offset > 0.008092
        - 0.0247666 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -2.5%  lam2 < 0.001131
        - 0.01743187 * max(0.0, 0.01323868 - Q.width) * max(0.0, 0.3220738 - Q.planar_flow) / 0.001091057   # -1.7%  width < 0.01324 and planar_flow < 0.3221
        + 0.01528536 * max(0.0, Q.centroid_offset - 0.01627885) / 0.006307541   # +1.5%  centroid_offset > 0.01628
        + 0.01372099 * max(0.0, 6.327379 - Q.log_sum_pt) / 0.0331691   # +1.4%  log_sum_pt < 6.327
        + 0.01101695 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.01426135 - Q.mean_phi2) / 9.496308e-05   # +1.1%  centroid_offset > 0.008092 and mean_phi2 < 0.01426
        - 0.01095005 * max(0.0, Q.sj3_dr_min - 0.02022982) / 0.02964063   # -1.1%  sj3_dr_min > 0.02023
        + 0.01000137 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 6.0 - Q.n_dr_0p05_0p1) / 0.03827938   # +1.0%  centroid_offset > 0.008092 and n_dr_0p05_0p1 < 6
        + 0.009503449 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.1682655 - Q.sj2_dr) / 0.0003126334   # +1.0%  centroid_offset > 0.008092 and sj2_dr < 0.1683
        + 0.008580151 * max(0.0, 3.0 - Q.n_dr_0_0p05) / 1.029338   # +0.9%  n_dr_0_0p05 < 3
        - 0.006768425 * max(0.0, Q.tau2 - 0.01713288) / 0.005826319   # -0.7%  tau2 > 0.01713
        + 0.00622989 * max(0.0, 6.638339 - Q.log_sum_pt) * max(0.0, 0.0586137 - Q.z_7) / 0.0003525255   # +0.6%  log_sum_pt < 6.638 and z_7 < 0.05861
        - 0.005399178 * max(0.0, 6.327379 - Q.log_sum_pt) * max(0.0, Q.pt_7 - 25.57812) / 0.2533514   # -0.5%  log_sum_pt < 6.327 and pt_7 > 25.58
        - 0.002261266 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.2%  z_6 < 0.0216
        + 0.001782402 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.2%  sum_pt > 988.4
        + 0.001511885 * max(0.0, 6.327379 - Q.log_sum_pt) * max(0.0, 0.08051087 - Q.z_6) / 0.0001292892   # +0.2%  log_sum_pt < 6.327 and z_6 < 0.08051
        + 0.001495669 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 19.46875 - Q.pt_6) / 0.002700092   # +0.1%  lam1 < 0.012 and pt_6 < 19.47
        + 0.0007059889 * max(0.0, 0.02160287 - Q.z_6) * max(0.0, 6.842717 - Q.log_sum_pt) / 4.457061e-06   # +0.1%  z_6 < 0.0216 and log_sum_pt < 6.843
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 47.42;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 47.42381 * (0.2095535
        - 0.1519318 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -15.2%  e2_sq < 0.008169
        - 0.07956424 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -8.0%  girth2 > 0.00752
        - 0.06828918 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -6.8%  tau1 < 0.1136
        + 0.06396053 * max(0.0, 0.00718279 - Q.e2_sq) / 0.003528787   # +6.4%  e2_sq < 0.007183
        - 0.05901856 * max(0.0, Q.girth2 - 0.004372139) / 0.003638805   # -5.9%  girth2 > 0.004372
        - 0.05620586 * max(0.0, 0.005590289 - Q.width) / 0.002229311   # -5.6%  width < 0.00559
        + 0.05419054 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +5.4%  tau1 < 0.09539
        - 0.05377602 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -5.4%  girth < 0.08724
        - 0.04660309 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -4.7%  sj2_dr < 0.1873
        + 0.04546163 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # +4.5%  girth2 > 0.008678
        + 0.04431172 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +4.4%  sj2_dr < 0.1592
        + 0.04127904 * max(0.0, Q.girth2 - 0.01323868) / 0.001442684   # +4.1%  girth2 > 0.01324
        + 0.0378978 * max(0.0, 0.005284669 - Q.e2_sq) / 0.00223735   # +3.8%  e2_sq < 0.005285
        + 0.02985095 * max(0.0, 0.1416226 - Q.tau1) / 0.08190733   # +3.0%  tau1 < 0.1416
        + 0.02353413 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +2.4%  lam2 < 0.001131
        + 0.01620652 * max(0.0, Q.girth2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +1.6%  girth2 > 0.004372 and planar_flow < 0.195
        - 0.01598822 * max(0.0, 0.1950135 - Q.planar_flow) / 0.07573236   # -1.6%  planar_flow < 0.195
        - 0.01495341 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # -1.5%  e3 < 0.0001869
        + 0.0148329 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.423044) / 0.01391616   # +1.5%  planar_flow < 0.195 and log_sum_pt > 6.423
        + 0.01417668 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.4%  tau1 < 0.05357
        - 0.01059523 * max(0.0, 2.843757 - Q.D2) / 1.524286   # -1.1%  D2 < 2.844
        + 0.01011922 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # +1.0%  sj2_dr < 0.1295
        - 0.009014039 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.width - 0.007520088) / 0.0001455029   # -0.9%  planar_flow < 0.195 and width > 0.00752
        - 0.00881516 * max(0.0, 0.0009641429 - Q.girth2) / 0.0002052288   # -0.9%  girth2 < 0.0009641
        + 0.007019017 * max(0.0, 0.2931906 - Q.LHA) / 0.07167606   # +0.7%  LHA < 0.2932
        + 0.006065658 * max(0.0, 0.0005611231 - Q.girth2) / 9.465572e-05   # +0.6%  girth2 < 0.0005611
        - 0.00578326 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 0.05240025 - Q.z_7) / 0.000631283   # -0.6%  planar_flow < 0.195 and z_7 < 0.0524
        - 0.005575005 * max(0.0, Q.N2 - 0.198361) / 0.06172685   # -0.6%  N2 > 0.1984
        + 0.004502698 * max(0.0, Q.centroid_offset - 0.009480685) / 0.009662931   # +0.5%  centroid_offset > 0.009481
        - 0.0004778369 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 6.679047e-05   # -0.0%  tau1 < 0.05357 and z_dr_0p05_0p1 > 0.6748
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 12.49;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.4941 * (-0.03559525
        + 0.0937936 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.002635418 - Q.girth2) / 3.338443e-05   # +9.4%  tau1 < 0.05357 and girth2 < 0.002635
        + 0.07827151 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +7.8%  girth2 < 0.006679 and centroid_offset < 0.02355
        - 0.07487183 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, 0.20552 - Q.z_dr_0p2_0p4) / 0.005135396   # -7.5%  LHA < 0.1967 and z_dr_0p2_0p4 < 0.2055
        - 0.07033817 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -7.0%  sj3_dr_max < 0.1426
        - 0.07006682 * max(0.0, 0.05464922 - Q.girth) * max(0.0, 0.005590289 - Q.width) / 7.638845e-05   # -7.0%  girth < 0.05465 and width < 0.00559
        - 0.06095963 * max(0.0, 0.05464922 - Q.girth) / 0.01532978   # -6.1%  girth < 0.05465
        + 0.0577616 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +5.8%  sj3_dr_max < 0.1986
        + 0.05192374 * max(0.0, 0.05464922 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2) / 2.397672e-06   # +5.2%  girth < 0.05465 and lam2 < 0.0001948
        + 0.04147 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +4.1%  tau1 < 0.05357
        - 0.04129426 * max(0.0, 0.05464922 - Q.girth) * max(0.0, 0.002635418 - Q.width) / 3.204761e-05   # -4.1%  girth < 0.05465 and width < 0.002635
        + 0.03889132 * max(0.0, 0.006679471 - Q.girth2) / 0.00293624   # +3.9%  girth2 < 0.006679
        + 0.03419484 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.0003061234 - Q.lam2) / 1.676724e-05   # +3.4%  sj3_dr_max < 0.1986 and lam2 < 0.0003061
        - 0.0296204 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -3.0%  girth2 < 0.00502 and centroid_offset > 0.00679
        - 0.02892888 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 4.721224 - Q.D2_b2) / 0.008595177   # -2.9%  girth2 < 0.006679 and D2_b2 < 4.721
        - 0.02737499 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, 0.0001947983 - Q.lam2) / 4.231311e-06   # -2.7%  LHA < 0.1967 and lam2 < 0.0001948
        + 0.02593698 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.0001053955   # +2.6%  girth2 < 0.00502 and z_dr_0p2_0p4 < 0.05644
        + 0.02434789 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # +2.4%  LHA < 0.1967
        - 0.02298312 * max(0.0, Q.z_dr_0_0p05 - 0.9008535) * max(0.0, 0.0001947983 - Q.lam2) / 5.136077e-06   # -2.3%  z_dr_0_0p05 > 0.9009 and lam2 < 0.0001948
        + 0.02281194 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +2.3%  sj3_dr_max < 0.107
        - 0.01977482 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -2.0%  log_sum_pt > 6.701
        + 0.01961567 * max(0.0, 0.005440034 - Q.zdr_5) / 0.002136845   # +2.0%  zdr_5 < 0.00544
        - 0.01656776 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 1.762929e-06 - Q.e3) / 3.265426e-05   # -1.7%  sum_pt_top5 > 687.4 and e3 < 1.763e-06
        - 0.01536048 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, 0.0003061234 - Q.lam2) / 6.518054e-06   # -1.5%  sj3_dr_max < 0.107 and lam2 < 0.0003061
        + 0.01442524 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.4007947 - Q.planar_flow) / 0.0003918103   # +1.4%  girth2 < 0.006679 and planar_flow < 0.4008
        + 0.01265556 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.002635418 - Q.width) / 5.489096e-05   # +1.3%  log_sum_pt > 6.701 and width < 0.002635
        - 0.005758945 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.pt_7 - 15.55391) / 0.4785904   # -0.6%  log_sum_pt > 6.701 and pt_7 > 15.55
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 26.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.08624 * (-0.1808128
        + 0.1021548 * max(0.0, 0.006679471 - Q.width) / 0.00293624   # +10.2%  width < 0.006679
        + 0.09397228 * max(0.0, 6.896095 - Q.log_sum_pt) / 0.3571984   # +9.4%  log_sum_pt < 6.896
        + 0.08964357 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +9.0%  girth2 < 0.00502
        + 0.07278334 * max(0.0, 0.02355416 - Q.centroid_offset) / 0.0103615   # +7.3%  centroid_offset < 0.02355
        + 0.07135765 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # +7.1%  girth2 < 0.003563
        - 0.05200753 * max(0.0, 0.06663269 - Q.girth) / 0.02186049   # -5.2%  girth < 0.06663
        - 0.04348412 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -4.3%  sj3_dr_max < 0.1426
        - 0.04123578 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -4.1%  sj2_dr < 0.1295
        - 0.04108967 * max(0.0, 0.004641801 - Q.e2_sq) / 0.001869767   # -4.1%  e2_sq < 0.004642
        - 0.04047658 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -4.0%  centroid_offset < 0.01838
        + 0.03749883 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.01364517 - Q.tau3) / 6.201347e-05   # +3.7%  centroid_offset < 0.01838 and tau3 < 0.01365
        + 0.0360113 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +3.6%  sj3_dr_max < 0.1986
        - 0.03467264 * max(0.0, 0.0030133 - Q.e2_sq) / 0.001065944   # -3.5%  e2_sq < 0.003013
        + 0.0273835 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.06729223 - Q.C2) / 0.0003300761   # +2.7%  centroid_offset < 0.01838 and C2 < 0.06729
        + 0.02701617 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +2.7%  max_dr < 0.1118
        - 0.02356323 * max(0.0, 0.05464922 - Q.girth) / 0.01532978   # -2.4%  girth < 0.05465
        - 0.02106579 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # -2.1%  tau1 < 0.0437
        - 0.02030687 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, 0.380911 - Q.D2_b2) / 0.001264948   # -2.0%  centroid_offset < 0.02355 and D2_b2 < 0.3809
        + 0.02019449 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 0.0234802   # +2.0%  centroid_offset < 0.01838 and n_dr_0p05_0p1 < 5
        + 0.01313898 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, Q.tau21_b2 - 0.004811143) / 0.002193894   # +1.3%  centroid_offset < 0.02355 and tau21_b2 > 0.004811
        + 0.01271413 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +1.3%  lam2 > 0.001131
        + 0.01200725 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +1.2%  LHA < 0.2161
        + 0.01165193 * max(0.0, 0.1778793 - Q.sj2_dr) / 0.05866968   # +1.2%  sj2_dr < 0.1779
        - 0.009718396 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, Q.planar_flow - 0.1115136) / 0.07272605   # -1.0%  log_sum_pt < 6.896 and planar_flow > 0.1115
        - 0.008101508 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -0.8%  sj3_dr_max > 0.2623
        - 0.007364904 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 1.395055 - Q.N3) / 0.0006017981   # -0.7%  centroid_offset < 0.01838 and N3 < 1.395
        + 0.006918213 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +0.7%  e3 > 0.0001869
        - 0.00681761 * max(0.0, 0.004918231 - Q.zdr_0) / 0.0006323791   # -0.7%  zdr_0 < 0.004918
        + 0.006316699 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, 0.05753583 - Q.z_5) / 0.0005902159   # +0.6%  log_sum_pt < 6.896 and z_5 < 0.05754
        - 0.006078826 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, Q.planar_flow - 0.2534037) / 0.0001586905   # -0.6%  lam2 > 0.001131 and planar_flow > 0.2534
        + 0.003253387 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.3%  n_dr_0p2_0p4 > 1
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 9.078;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.078457 * (0.2637209
        - 0.09772543 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # -9.8%  lam1 < 0.003377
        + 0.09398448 * Q.lam1 / 0.00591636   # +9.4%  lam1
        - 0.08854825 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -8.9%  lam1 < 0.006507
        + 0.07960492 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # +8.0%  girth2 > 0.00752
        - 0.07301175 * max(0.0, Q.LHA - 0.1967397) / 0.07109091   # -7.3%  LHA > 0.1967
        + 0.0718158 * max(0.0, Q.tau1 - 0.04369778) / 0.031989   # +7.2%  tau1 > 0.0437
        + 0.06623319 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # +6.6%  sj2_dr > 0.1295
        - 0.05694296 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.6947818 - Q.planar_flow) / 0.0166918   # -5.7%  sj2_dr > 0.1683 and planar_flow < 0.6948
        - 0.05308311 * max(0.0, Q.lam1 - 0.00483998) / 0.002931468   # -5.3%  lam1 > 0.00484
        + 0.05129164 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.158545   # +5.1%  n_dr_0p05_0p1 < 5
        + 0.04943157 * max(0.0, Q.lam2 - 0.0001947983) / 0.0004461515   # +4.9%  lam2 > 0.0001948
        - 0.04303088 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # -4.3%  sj2_dr > 0.1683
        + 0.03933497 * max(0.0, 0.002915531 - Q.girth2_top3) / 0.001178811   # +3.9%  girth2_top3 < 0.002916
        - 0.03237764 * max(0.0, Q.LHA - 0.3127275) / 0.01533444   # -3.2%  LHA > 0.3127
        - 0.01848195 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.eccentricity - 0.4829602) / 8.513077e-05   # -1.8%  lam2 > 0.0001948 and eccentricity > 0.483
        + 0.01819608 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # +1.8%  e3 > 8.148e-05
        - 0.01806506 * max(0.0, Q.LHA - 0.3127275) * max(0.0, 0.6947818 - Q.planar_flow) / 0.005887138   # -1.8%  LHA > 0.3127 and planar_flow < 0.6948
        - 0.01488833 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.planar_flow - 0.06126205) / 0.0002819968   # -1.5%  lam2 > 0.0001948 and planar_flow > 0.06126
        - 0.01235183 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -1.2%  lam2 > 0.003408
        - 0.01112618 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, 6.538559 - Q.log_sum_pt) / 0.0001231869   # -1.1%  lam2 > 0.0001948 and log_sum_pt < 6.539
        + 0.01047396 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +1.0%  sj3_dr_min > 0.1278
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 33.84;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.83674 * (-0.06582694
        + 0.2217461 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +22.2%  width < 0.008678
        - 0.1130649 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # -11.3%  LHA < 0.3127
        + 0.1046951 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +10.5%  width < 0.01324
        - 0.1039291 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -10.4%  lam1 < 0.008376
        + 0.09860799 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +9.9%  LHA < 0.3033
        - 0.06064054 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -6.1%  girth < 0.0717
        + 0.04121778 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +4.1%  centroid_offset < 0.03776
        - 0.03995298 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -4.0%  e2_sq < 0.008169
        + 0.03726424 * max(0.0, Q.sj2_dr - 0.09419022) / 0.07816795   # +3.7%  sj2_dr > 0.09419
        + 0.02374944 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +2.4%  planar_flow < 0.2534
        - 0.02219165 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -2.2%  sj2_dr > 0.1779
        - 0.02065386 * max(0.0, 0.006390125 - Q.e2_sq) / 0.002953508   # -2.1%  e2_sq < 0.00639
        - 0.01766429 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # -1.8%  tau1 < 0.09539
        - 0.01711079 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.08543881 - Q.sj3_dr_min) / 0.0002784225   # -1.7%  centroid_offset < 0.01438 and sj3_dr_min < 0.08544
        + 0.01467579 * max(0.0, 0.03459477 - Q.tau1) * max(0.0, 0.002915531 - Q.girth2_top3) / 2.198359e-05   # +1.5%  tau1 < 0.03459 and girth2_top3 < 0.002916
        - 0.0123788 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # -1.2%  girth2 < 0.003563
        + 0.01101431 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +1.1%  lam1 < 0.00733
        - 0.010736 * max(0.0, 0.03459477 - Q.tau1) / 0.008474553   # -1.1%  tau1 < 0.03459
        - 0.006264828 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.6%  planar_flow < 0.2534 and sum_pt < 840
        + 0.006075254 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.05932655 - Q.tau21_b2) / 7.260318e-05   # +0.6%  centroid_offset < 0.01438 and tau21_b2 < 0.05933
        - 0.005348888 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 37.15625 - Q.pt_7) / 0.5537514   # -0.5%  planar_flow < 0.2534 and pt_7 < 37.16
        - 0.004187805 * max(0.0, 39.75 - Q.pt_6) * max(0.0, Q.girth2_top3 - 0.005011407) / 0.010038   # -0.4%  pt_6 < 39.75 and girth2_top3 > 0.005011
        - 0.004078086 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001130645 - Q.lam2) / 1.843299e-05   # -0.4%  sj2_dr > 0.1779 and lam2 < 0.001131
        + 0.002114171 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +0.2%  centroid_offset > 0.0499
        - 0.0006372798 * max(0.0, 0.0030133 - Q.e2_sq) * max(0.0, Q.girth2_top3 - 0.003952582) / 2.119623e-08   # -0.1%  e2_sq < 0.003013 and girth2_top3 > 0.003953
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.4341;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.4340671 * (-3.139874
        + 0.2893478 * max(0.0, Q.girth2 - 0.02530566) / 0.0002851418   # +28.9%  girth2 > 0.02531
        + 0.2796328 * max(0.0, Q.girth2 - 0.01882765) / 0.0007605369   # +28.0%  girth2 > 0.01883
        - 0.2034046 * max(0.0, Q.girth2 - 0.02530566) * max(0.0, 868.5094 - Q.sum_pt) / 0.1051521   # -20.3%  girth2 > 0.02531 and sum_pt < 868.5
        + 0.1257553 * max(0.0, Q.ptdr0_4 - 15.26525) / 0.1831584   # +12.6%  ptdr0_4 > 15.27
        - 0.1018595 * max(0.0, Q.ptdr0_4 - 15.26525) * max(0.0, 988.4078 - Q.sum_pt) / 57.72599   # -10.2%  ptdr0_4 > 15.27 and sum_pt < 988.4
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 20.34;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.34164 * (0.07826138
        + 0.3503458 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # +35.0%  girth < 0.1484
        + 0.07366232 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +7.4%  lam1 < 0.01643
        - 0.06025232 * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.09151624   # -6.0%  sj3_dr_min < 0.1278
        - 0.06006166 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # -6.0%  width < 0.008678
        - 0.05796603 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -5.8%  girth < 0.1484 and log_sum_pt < 6.804
        - 0.05282351 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -5.3%  tau1 < 0.1136
        + 0.05114076 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +5.1%  width < 0.01324
        - 0.04712174 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 1.399607e-06   # -4.7%  e3 < 8.148e-05 and centroid_offset < 0.03776
        - 0.04639892 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -4.6%  e2 < 0.08001
        + 0.04056779 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +4.1%  lam1 < 0.006507
        - 0.03595416 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.03585464 - Q.tau2) / 0.002456165   # -3.6%  girth < 0.1484 and tau2 < 0.03585
        - 0.03249038 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -3.2%  girth < 0.1484 and pt_7 < 38.53
        + 0.02156865 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 40.04062 - Q.pt_7) / 652.1541   # +2.2%  sum_pt_top5 > 687.4 and pt_7 < 40.04
        - 0.01777834 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.8%  sj3_dr_max > 0.2337
        - 0.01526286 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 25.57812 - Q.pt_7) / 0.01870322   # -1.5%  lam1 < 0.01643 and pt_7 < 25.58
        + 0.01401458 * max(0.0, Q.z_7 - 0.0586137) / 0.00587069   # +1.4%  z_7 > 0.05861
        - 0.01330574 * max(0.0, Q.pt_7 - 31.85938) / 5.945918   # -1.3%  pt_7 > 31.86
        - 0.00762202 * max(0.0, 6.46415 - Q.log_sum_pt) * max(0.0, 0.06809619 - Q.C3) / 0.002112398   # -0.8%  log_sum_pt < 6.464 and C3 < 0.0681
        - 0.001662429 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.sj3_pairmin_over_m - 0.2195798) / 0.4564276   # -0.2%  sum_pt > 988.4 and sj3_pairmin_over_m > 0.2196
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 37.52;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 37.51778 * (-0.01373592
        - 0.09677068 * max(0.0, Q.girth2 - 0.006679471) / 0.002702003   # -9.7%  girth2 > 0.006679
        - 0.09198913 * max(0.0, 0.233678 - Q.sj3_dr_max) / 0.09057682   # -9.2%  sj3_dr_max < 0.2337
        + 0.08848928 * max(0.0, 0.1594012 - Q.sj3_dr_max) / 0.04422261   # +8.8%  sj3_dr_max < 0.1594
        + 0.07321389 * max(0.0, Q.girth - 0.04081947) / 0.0270611   # +7.3%  girth > 0.04082
        - 0.06633627 * max(0.0, Q.tau1 - 0.03459477) / 0.03725748   # -6.6%  tau1 > 0.03459
        + 0.05726416 * max(0.0, Q.e2_sq - 0.006390125) / 0.002450953   # +5.7%  e2_sq > 0.00639
        + 0.05275051 * max(0.0, Q.e2_sq - 0.00718279) / 0.002233568   # +5.3%  e2_sq > 0.007183
        - 0.05054176 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -5.1%  girth2 > 0.00752
        - 0.04276565 * max(0.0, Q.e2_sq - 0.008168571) / 0.002013747   # -4.3%  e2_sq > 0.008169
        + 0.03829995 * max(0.0, Q.girth - 0.02689598) / 0.03613842   # +3.8%  girth > 0.0269
        - 0.03786249 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -3.8%  girth > 0.08724
        - 0.02896015 * max(0.0, Q.girth2 - 0.008678045) * max(0.0, Q.log_sum_pt - 6.267538) / 0.0002118474   # -2.9%  girth2 > 0.008678 and log_sum_pt > 6.268
        + 0.02803661 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # +2.8%  girth2 > 0.008678
        - 0.02722169 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # -2.7%  sj3_dr_max < 0.2134
        + 0.02701184 * max(0.0, Q.width - 0.005590289) / 0.003084256   # +2.7%  width > 0.00559
        - 0.02280869 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # -2.3%  girth > 0.1019
        + 0.02054246 * max(0.0, Q.e2_sq - 0.002074109) / 0.004484017   # +2.1%  e2_sq > 0.002074
        - 0.02025267 * max(0.0, Q.e2_sq - 0.01716248) / 0.0007604864   # -2.0%  e2_sq > 0.01716
        + 0.01821191 * max(0.0, Q.LHA - 0.2160559) / 0.05893777   # +1.8%  LHA > 0.2161
        + 0.01690074 * max(0.0, Q.width - 0.01323868) * max(0.0, Q.sum_pt - 488.9312) / 0.1069057   # +1.7%  width > 0.01324 and sum_pt > 488.9
        - 0.01340235 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -1.3%  e2 > 0.03556
        + 0.0119818 * max(0.0, Q.width - 0.01323868) / 0.001442684   # +1.2%  width > 0.01324
        + 0.01049266 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +1.0%  e2 > 0.04111
        - 0.01006271 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -1.0%  planar_flow < 0.1115 and lam1 < 0.00733
        + 0.008674128 * max(0.0, Q.e2_sq - 0.002074109) * max(0.0, Q.log_sum_pt - 6.267538) / 0.0007382066   # +0.9%  e2_sq > 0.002074 and log_sum_pt > 6.268
        + 0.007022269 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # +0.7%  planar_flow < 0.1115 and lam1 < 0.01643
        - 0.006492973 * max(0.0, Q.centroid_offset - 0.02355416) / 0.003991673   # -0.6%  centroid_offset > 0.02355
        + 0.006000757 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.width) / 3.351526e-05   # +0.6%  planar_flow < 0.1115 and width < 0.006097
        + 0.005775818 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +0.6%  LHA > 0.3033
        - 0.004457413 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.4%  centroid_offset > 0.0499
        + 0.004187109 * max(0.0, Q.tau1 - 0.09538712) / 0.01057268   # +0.4%  tau1 > 0.09539
        - 0.003206215 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.3%  planar_flow < 0.1115 and centroid_offset < 0.01838
        - 0.00201328 * max(0.0, Q.psi_0p1 - 0.8155839) * max(0.0, Q.centroid_offset - 0.03776099) / 4.931851e-05   # -0.2%  psi_0p1 > 0.8156 and centroid_offset > 0.03776
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 30.84;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 30.8371 * (-0.05544261
        + 0.1499008 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +15.0%  girth2 < 0.01324
        - 0.1097267 * max(0.0, 0.007520088 - Q.girth2) / 0.00354499   # -11.0%  girth2 < 0.00752
        - 0.08237006 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # -8.2%  e2_sq < 0.01166
        + 0.0759968 * max(0.0, 0.006390125 - Q.e2_sq) / 0.002953508   # +7.6%  e2_sq < 0.00639
        + 0.07374502 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +7.4%  e2 < 0.04111
        - 0.07305634 * max(0.0, 0.006679471 - Q.girth2) / 0.00293624   # -7.3%  girth2 < 0.006679
        + 0.05077125 * max(0.0, 0.01716248 - Q.e2_sq) / 0.0120354   # +5.1%  e2_sq < 0.01716
        - 0.03579187 * max(0.0, 0.06344108 - Q.e2) / 0.03657078   # -3.6%  e2 < 0.06344
        - 0.03506989 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -3.5%  lam1 < 0.008376
        + 0.03504387 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +3.5%  lam1 < 0.005954
        + 0.03122598 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sum_pt_top5 - 402.625) / 15.59919   # +3.1%  planar_flow < 0.195 and sum_pt_top5 > 402.6
        + 0.03117418 * max(0.0, 0.0811449 - Q.tau1) / 0.03266098   # +3.1%  tau1 < 0.08114
        - 0.02485979 * max(0.0, 0.005834489 - Q.e2_sq) / 0.002579453   # -2.5%  e2_sq < 0.005834
        + 0.02138658 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.2931906) / 0.001536651   # +2.1%  N2 < 0.2233 and LHA > 0.2932
        - 0.02133416 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # -2.1%  girth2_top3 < 0.002152
        - 0.01854526 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3855647) / 0.0002330351   # -1.9%  N2 < 0.2233 and LHA > 0.3856
        - 0.01838015 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.girth2 - 0.008678045) / 0.0001197728   # -1.8%  N2 < 0.2233 and girth2 > 0.008678
        - 0.01545844 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -1.5%  lam1 < 0.005433
        - 0.01525708 * max(0.0, 0.1950135 - Q.planar_flow) / 0.07573236   # -1.5%  planar_flow < 0.195
        + 0.01428245 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +1.4%  sj3_dr_max > 0.1879
        - 0.01355493 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1872617 - Q.sj2_dr) / 0.0007896743   # -1.4%  N2 < 0.2233 and sj2_dr < 0.1873
        - 0.0122526 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 0.07148865 - Q.z_7) / 0.001619322   # -1.2%  planar_flow < 0.195 and z_7 < 0.07149
        + 0.01109612 * max(0.0, 0.1515405 - Q.z_dr_0_0p05) / 0.0478325   # +1.1%  z_dr_0_0p05 < 0.1515
        - 0.007485139 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3255822) / 0.0007869707   # -0.7%  N2 < 0.2233 and LHA > 0.3256
        - 0.005999096 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -0.6%  sj3_dr_max > 0.2623
        - 0.00543153 * max(0.0, 0.002151568 - Q.girth2_top3) * max(0.0, Q.eccentricity - 0.9598562) / 4.406774e-06   # -0.5%  girth2_top3 < 0.002152 and eccentricity > 0.9599
        - 0.004006378 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.dr12 - 0.1587481) / 1.436546e-06   # -0.4%  girth2 < 0.006679 and dr12 > 0.1587
        + 0.003516549 * max(0.0, 0.006390125 - Q.e2_sq) * max(0.0, Q.dr12 - 0.1587481) / 1.354517e-06   # +0.4%  e2_sq < 0.00639 and dr12 > 0.1587
        + 0.003281027 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1591713 - Q.sj2_dr) / 0.0002571871   # +0.3%  N2 < 0.2233 and sj2_dr < 0.1592
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9687689075630252, 0.7926764705882353, 2.0547959033613443, 0.9449180672268908, 1.4049886554621849, 1.849476680672269, 1.2155884453781514, 1.952059243697479, 0.35699936974789914, 3.2161705882352942, 2.1276720588235296, 2.2881351890756303, 0.07003991596638655, 3.7826936974789915, 0.5383765756302521, 0.4681109243697479]
T = [2.42034658203125, 1.504494675682773, 3.6765140460871844, 2.722542161239496, 3.1487408219537816]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +36%, n9 +23%, n5 -14%, n1 +13%, n0 -6%, n6 +5% ...
            + 0.3647908 * h[2] / H_AVG[2]
            + 0.2283885 * h[9] / H_AVG[9]
            - 0.1432757 * h[5] / H_AVG[5]
            + 0.1279318 * h[1] / H_AVG[1]
            - 0.06254069 * h[0] / H_AVG[0]
            + 0.05493221 * h[6] / H_AVG[6]
            - 0.01814033 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +54%, n10 -18%, n6 +10%, n4 -9%, n5 +6%, n15 +2% ...
            + 0.5427775 * h[9] / H_AVG[9]
            - 0.1767763 * h[10] / H_AVG[10]
            + 0.1009964 * h[6] / H_AVG[6]
            - 0.08754945 * h[4] / H_AVG[4]
            + 0.05762348 * h[5] / H_AVG[5]
            + 0.01944635 * h[15] / H_AVG[15]
            + 0.01483053 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +23%, n3 -13%, n7 +12%, n14 -11%, n6 -10%, n0 +9% ...
            + 0.233387 * h[11] / H_AVG[11]
            - 0.1285073 * h[3] / H_AVG[3]
            + 0.1161462 * h[7] / H_AVG[7]
            - 0.1098275 * h[14] / H_AVG[14]
            - 0.1033238 * h[6] / H_AVG[6]
            + 0.09057882 * h[0] / H_AVG[0]
            - 0.08753571 * h[15] / H_AVG[15]
            + 0.07234316 * h[13] / H_AVG[13]
            - 0.02733713 * h[9] / H_AVG[9]
            - 0.02427567 * h[8] / H_AVG[8]
            - 0.00673767 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +34%, n3 -20%, n6 -17%, n13 +8%, n14 +7%, n4 +4% ...
            + 0.3360931 * h[7] / H_AVG[7]
            - 0.195228 * h[3] / H_AVG[3]
            - 0.1674338 * h[6] / H_AVG[6]
            + 0.07598268 * h[13] / H_AVG[13]
            + 0.0741554 * h[14] / H_AVG[14]
            + 0.040317 * h[4] / H_AVG[4]
            - 0.03691599 * h[9] / H_AVG[9]
            + 0.03639413 * h[1] / H_AVG[1]
            - 0.02686545 * h[15] / H_AVG[15]
            + 0.01061437 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -49%, n10 +25%, n5 -15%, n4 +6%, n8 +2%, n3 +2% ...
            - 0.4880425 * h[13] / H_AVG[13]
            + 0.2533956 * h[10] / H_AVG[10]
            - 0.1468426 * h[5] / H_AVG[5]
            + 0.05577581 * h[4] / H_AVG[4]
            + 0.02125846 * h[8] / H_AVG[8]
            + 0.01875587 * h[3] / H_AVG[3]
            - 0.01112189 * h[12] / H_AVG[12]
            + 0.004807323 * h[0] / H_AVG[0]
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
