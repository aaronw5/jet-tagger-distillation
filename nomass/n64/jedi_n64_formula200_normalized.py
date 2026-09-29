"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.6%   (on for 51% of jets)
  neuron  1:  11.6%   (on for 95% of jets)
  neuron  4:  10.9%   (on for 68% of jets)
  neuron  5:  10.6%   (on for 82% of jets)
  neuron  9:   8.1%   (on for 82% of jets)
  neuron  0:   7.9%   (on for 60% of jets)
  neuron 13:   6.7%   (on for 85% of jets)
  neuron 10:   6.4%   (on for 79% of jets)
  neuron  7:   4.9%   (on for 35% of jets)
  neuron 12:   4.1%   (on for 47% of jets)
  neuron  6:   3.8%   (on for 55% of jets)
  neuron  3:   3.6%   (on for 58% of jets)
  neuron 11:   2.7%   (on for 46% of jets)
  neuron 14:   2.3%   (on for 30% of jets)
  neuron 15:   2.1%   (on for 66% of jets)
  neuron  2:   0.7%   (on for 61% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.4% (the network: 81.1%); same class as the network for 91.8% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.n_particles            number of real particles (pT > 0)
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft7_dr0              ΔR between the hardest and the 7. softest real particle (0 if among the 15 hardest)
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.sum_z_dr2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.sum_z_dr2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_phi               pT-weighted mean Δφ
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.tau43                  N-subjettiness τ4/τ3
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
        z_top5=sum(zs[:5]),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        n_particles=len(real),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        soft3_pt=softp(3, 'pt'),
        soft1_z=softp(1, 'z'),
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj2_dr=subjets(2)["dr"][0],
        soft7_dr0=softp(7, 'dr0'),
        eta_1=eta[1],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        sum_z_dr2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        sum_z_dr2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    # scale S = 8.253;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.252595 * (-0.1781258
        + 0.323819 * max(0.0, 0.00788 - Q.sum_zz_dr2) / 0.00224567   # +32.4%  sum_zz_dr2 < 0.00788
        - 0.1310417 * max(0.0, 0.00618 - Q.sum_zz_dr2) / 0.001301364   # -13.1%  sum_zz_dr2 < 0.00618
        + 0.1196622 * max(0.0, Q.sum_pt_top40 - 862.0) / 163.7685   # +12.0%  sum_pt_top40 > 862
        - 0.0937555 * max(0.0, Q.sum_pt - 1000.0) / 59.06306   # -9.4%  sum_pt > 1000
        + 0.09126385 * max(0.0, 0.00693 - Q.sum_zz_dr2) / 0.00168493   # +9.1%  sum_zz_dr2 < 0.00693
        - 0.06348358 * max(0.0, 0.00588 - Q.sum_z_dr2_top50) * max(0.0, 1260.0 - Q.sum_pt) / 0.2482959   # -6.3%  sum_z_dr2_top50 < 0.00588 and sum_pt < 1260
        - 0.03518302 * max(0.0, 0.00812 - Q.sum_z_dr2) * max(0.0, 0.00585 - Q.sum_z_dr2_top15) / 1.001211e-05   # -3.5%  sum_z_dr2 < 0.00812 and sum_z_dr2_top15 < 0.00585
        + 0.0328316 * max(0.0, Q.psi_0p3 - 0.994) / 0.002957925   # +3.3%  psi_0p3 > 0.994
        - 0.03143715 * max(0.0, Q.sum_pt_top40 - 1020.0) / 37.59972   # -3.1%  sum_pt_top40 > 1020
        - 0.024722 * max(0.0, 0.0712 - Q.tau1) / 0.01369266   # -2.5%  tau1 < 0.0712
        + 0.02207729 * max(0.0, 0.299 - Q.tau21_b2) / 0.0839608   # +2.2%  tau21_b2 < 0.299
        + 0.01442007 * max(0.0, Q.sum_pt - 1120.0) / 19.83383   # +1.4%  sum_pt > 1120
        - 0.01366077 * max(0.0, 0.00313 - Q.sum_z_dr2_top40) / 0.0004222352   # -1.4%  sum_z_dr2_top40 < 0.00313
        - 0.002642308 * max(0.0, Q.sum_pt - 1170.0) * max(0.0, Q.soft7_dr0 - 0.256) / 0.2756751   # -0.3%  sum_pt > 1170 and soft7_dr0 > 0.256
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 12.99;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.99035 * (0.229401
        - 0.1512942 * max(0.0, Q.sum_pt_top50 - 932.0) / 110.4138   # -15.1%  sum_pt_top50 > 932
        + 0.1488589 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # +14.9%  log_sum_pt > 6.91
        + 0.1152991 * max(0.0, Q.sum_pt - 973.0) / 80.52558   # +11.5%  sum_pt > 973
        - 0.1089704 * max(0.0, 7.15 - Q.log_sum_pt) / 0.2103363   # -10.9%  log_sum_pt < 7.15
        + 0.07074787 * max(0.0, 1080.0 - Q.sum_pt_top50) / 68.58507   # +7.1%  sum_pt_top50 < 1080
        - 0.0545663 * max(0.0, Q.log_sum_pt - 6.99) / 0.0210337   # -5.5%  log_sum_pt > 6.99
        - 0.04299534 * max(0.0, Q.z_top5 - 0.487) / 0.1126058   # -4.3%  z_top5 > 0.487
        + 0.03306742 * max(0.0, Q.n_particles - 38.1) * max(0.0, Q.tau32 - 0.316) / 3.801394   # +3.3%  n_particles > 38.1 and tau32 > 0.316
        + 0.03261989 * max(0.0, Q.n_particles - 39.4) / 9.900559   # +3.3%  n_particles > 39.4
        - 0.03043167 * max(0.0, 0.00178 - Q.lam2) / 0.0009548748   # -3.0%  lam2 < 0.00178
        - 0.02993002 * max(0.0, Q.n_particles - 37.3) * max(0.0, 0.00219 - Q.soft1_z) / 0.01524712   # -3.0%  n_particles > 37.3 and soft1_z < 0.00219
        + 0.0218482 * max(0.0, 0.247 - Q.LHA) / 0.03411248   # +2.2%  LHA < 0.247
        - 0.02105505 * max(0.0, 0.0364 - Q.M3) * max(0.0, Q.psi_0p3 - 0.943) / 0.0005373527   # -2.1%  M3 < 0.0364 and psi_0p3 > 0.943
        - 0.02023631 * max(0.0, 7.42 - Q.n_dr_0p2_0p4) / 2.266179   # -2.0%  n_dr_0p2_0p4 < 7.42
        + 0.02020751 * max(0.0, 0.00706 - Q.sum_z_dr2_top30) / 0.002083355   # +2.0%  sum_z_dr2_top30 < 0.00706
        + 0.01678848 * max(0.0, Q.z_top30_slots - 0.958) * max(0.0, Q.pt1_dr01 - -4.86) / 0.1929985   # +1.7%  z_top30_slots > 0.958 and pt1_dr01 > -4.86
        - 0.01620774 * max(0.0, Q.n_particles - 37.5) * max(0.0, 0.959 - Q.tau43) / 1.422597   # -1.6%  n_particles > 37.5 and tau43 < 0.959
        - 0.01356751 * max(0.0, Q.z_dr_0_0p05 - 0.827) * max(0.0, 10.7 - Q.n_dr_0p05_0p1) / 0.1989241   # -1.4%  z_dr_0_0p05 > 0.827 and n_dr_0p05_0p1 < 10.7
        + 0.01124271 * max(0.0, Q.n_particles - 35.4) * max(0.0, 0.01 - Q.zdr_0) / 0.03884223   # +1.1%  n_particles > 35.4 and zdr_0 < 0.01
        - 0.008896054 * max(0.0, 6.99 - Q.n_dr_0p1_0p2) / 0.9711167   # -0.9%  n_dr_0p1_0p2 < 6.99
        + 0.008182142 * max(0.0, Q.n_dr_0_0p05 - 14.1) / 3.054279   # +0.8%  n_dr_0_0p05 > 14.1
        + 0.008081574 * max(0.0, 0.00119 - Q.sum_z_dr2_top20) / 0.000127406   # +0.8%  sum_z_dr2_top20 < 0.00119
        + 0.007952072 * max(0.0, 0.207 - Q.tau21) / 0.01022775   # +0.8%  tau21 < 0.207
        - 0.006953603 * max(0.0, Q.zdr_0 - 0.0126) / 0.002150709   # -0.7%  zdr_0 > 0.0126
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 1.26;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.259594 * (-0.5978116
        + 0.6222665 * max(0.0, 0.0192 - Q.sum_z_dr2_top15) / 0.0126829   # +62.2%  sum_z_dr2_top15 < 0.0192
        + 0.210399 * max(0.0, Q.log_sum_pt - 6.92) / 0.04530211   # +21.0%  log_sum_pt > 6.92
        - 0.1438266 * max(0.0, 0.00386 - Q.sum_z_dr2) / 0.0005523268   # -14.4%  sum_z_dr2 < 0.00386
        - 0.02350789 * max(0.0, Q.sum_pt_top15 - 1010.0) * max(0.0, Q.eta_1 - 0.0872) / 0.006194645   # -2.4%  sum_pt_top15 > 1010 and eta_1 > 0.0872
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 2.484;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.484397 * (-0.01895832
        - 0.2455038 * max(0.0, 0.378 - Q.LHA) / 0.1222302   # -24.6%  LHA < 0.378
        + 0.2383895 * max(0.0, 0.00921 - Q.sum_zz_dr2) / 0.003201374   # +23.8%  sum_zz_dr2 < 0.00921
        - 0.1795069 * max(0.0, 0.348 - Q.tau21) / 0.0527147   # -18.0%  tau21 < 0.348
        + 0.1305942 * max(0.0, 0.203 - Q.tau21_b2) / 0.03844168   # +13.1%  tau21_b2 < 0.203
        + 0.08507693 * max(0.0, 0.000631 - Q.lam2) / 0.000157735   # +8.5%  lam2 < 0.000631
        + 0.08489126 * max(0.0, 47.9 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 822.0) / 1573.908   # +8.5%  n_particles < 47.9 and sum_pt_top40 > 822
        + 0.0360375 * max(0.0, Q.psi_0p2 - 0.998) / 0.0002337636   # +3.6%  psi_0p2 > 0.998
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 5.823;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.822874 * (0.2146706
        - 0.147044 * max(0.0, 0.00742 - Q.sum_z_dr2_top20) / 0.002659063   # -14.7%  sum_z_dr2_top20 < 0.00742
        + 0.1395522 * max(0.0, 0.0106 - Q.sum_z_dr2_top20) / 0.005110659   # +14.0%  sum_z_dr2_top20 < 0.0106
        - 0.1310919 * max(0.0, 1060.0 - Q.sum_pt_top50) / 53.00916   # -13.1%  sum_pt_top50 < 1060
        + 0.1123999 * max(0.0, 1050.0 - Q.sum_pt_top40) / 55.46527   # +11.2%  sum_pt_top40 < 1050
        - 0.09931659 * max(0.0, 0.103 - Q.tau2) / 0.0697597   # -9.9%  tau2 < 0.103
        - 0.08958006 * max(0.0, 0.00615 - Q.sum_zz_dr2) / 0.001287934   # -9.0%  sum_zz_dr2 < 0.00615
        - 0.06627219 * max(0.0, Q.n_particles - 28.9) / 17.70159   # -6.6%  n_particles > 28.9
        + 0.05735447 * max(0.0, 23.4 - Q.n_dr_0p2_0p4) / 14.77734   # +5.7%  n_dr_0p2_0p4 < 23.4
        + 0.05468883 * max(0.0, 25.8 - Q.n_dr_0p1_0p2) / 14.02846   # +5.5%  n_dr_0p1_0p2 < 25.8
        + 0.04434255 * max(0.0, Q.sum_z_dr2_top50 - 0.00443) / 0.005435812   # +4.4%  sum_z_dr2_top50 > 0.00443
        + 0.03076368 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, Q.sum_pt_top40 - 847.0) / 0.2233579   # +3.1%  psi_0p3 > 0.997 and sum_pt_top40 > 847
        - 0.0153014 * max(0.0, Q.sj2_dr - 0.261) * max(0.0, 0.0333 - Q.C2_b2) / 0.0001552232   # -1.5%  sj2_dr > 0.261 and C2_b2 < 0.0333
        - 0.01229228 * max(0.0, 0.984 - Q.z_top50_slots) / 0.00334469   # -1.2%  z_top50_slots < 0.984
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 5.905;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.904626 * (-0.08180027
        + 0.4708704 * max(0.0, Q.sum_pt_top50 - 934.0) / 108.606   # +47.1%  sum_pt_top50 > 934
        - 0.2531862 * max(0.0, Q.log_sum_pt - 6.92) / 0.04530211   # -25.3%  log_sum_pt > 6.92
        + 0.07320778 * max(0.0, 0.0156 - Q.tau4) / 0.002604003   # +7.3%  tau4 < 0.0156
        - 0.04499763 * max(0.0, Q.sum_pt_top50 - 1050.0) / 31.40593   # -4.5%  sum_pt_top50 > 1050
        + 0.03992447 * max(0.0, 0.0312 - Q.M3) / 0.00869886   # +4.0%  M3 < 0.0312
        - 0.0392463 * max(0.0, 69.5 - Q.n_particles) * max(0.0, 2.43 - Q.D2) / 16.67157   # -3.9%  n_particles < 69.5 and D2 < 2.43
        + 0.03670792 * max(0.0, Q.log_sum_pt - 7.01) / 0.01747956   # +3.7%  log_sum_pt > 7.01
        - 0.0312962 * max(0.0, Q.sum_zz_dr2 - 0.00946) / 0.003219379   # -3.1%  sum_zz_dr2 > 0.00946
        - 0.01056317 * max(0.0, 0.000452 - Q.sum_z_dr2_top15) / 3.231687e-05   # -1.1%  sum_z_dr2_top15 < 0.000452
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 9.917;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.916524 * (0.1139512
        - 0.3001754 * max(0.0, 0.00988 - Q.lam1_plus_lam2) / 0.003702358   # -30.0%  lam1_plus_lam2 < 0.00988
        - 0.121765 * max(0.0, Q.e2 - 0.0143) / 0.01775713   # -12.2%  e2 > 0.0143
        + 0.1044281 * max(0.0, 0.00774 - Q.lam1) / 0.002576029   # +10.4%  lam1 < 0.00774
        + 0.08709049 * max(0.0, Q.psi_0p3 - 0.985) / 0.009671164   # +8.7%  psi_0p3 > 0.985
        + 0.07840991 * max(0.0, 0.00382 - Q.lam1) / 0.0006820647   # +7.8%  lam1 < 0.00382
        - 0.0730514 * max(0.0, 0.0136 - Q.sum_z_dr2) * max(0.0, Q.psi_0p3 - 0.985) / 7.530311e-05   # -7.3%  sum_z_dr2 < 0.0136 and psi_0p3 > 0.985
        + 0.06716256 * max(0.0, 0.00207 - Q.lam2) / 0.001191447   # +6.7%  lam2 < 0.00207
        + 0.0452749 * max(0.0, Q.log_sum_pt - 6.81) / 0.1394316   # +4.5%  log_sum_pt > 6.81
        + 0.04295566 * max(0.0, Q.LHA - 0.207) / 0.07539308   # +4.3%  LHA > 0.207
        - 0.03358487 * max(0.0, 0.0252 - Q.sum_zz_dr2) * max(0.0, 6.98 - Q.log_sum_pt) / 0.0008695695   # -3.4%  sum_zz_dr2 < 0.0252 and log_sum_pt < 6.98
        + 0.01619972 * max(0.0, Q.e2 - 0.0493) * max(0.0, 0.255 - Q.z_dr_0_0p05) / 0.0004026189   # +1.6%  e2 > 0.0493 and z_dr_0_0p05 < 0.255
        + 0.01498968 * max(0.0, 0.00961 - Q.sum_zz_dr2) * max(0.0, 1010.0 - Q.sum_pt) / 0.04889655   # +1.5%  sum_zz_dr2 < 0.00961 and sum_pt < 1010
        - 0.01491239 * max(0.0, Q.lam2 - 0.00104) / 0.0007742361   # -1.5%  lam2 > 0.00104
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 16.13;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.12594 * (-0.04396643
        + 0.2084604 * max(0.0, 0.00967 - Q.sum_zz_dr2) / 0.003546012   # +20.8%  sum_zz_dr2 < 0.00967
        - 0.1849252 * max(0.0, 0.00966 - Q.sum_zz_dr2) * max(0.0, 1170.0 - Q.sum_pt) / 0.4532056   # -18.5%  sum_zz_dr2 < 0.00966 and sum_pt < 1170
        - 0.1781315 * max(0.0, 0.0078 - Q.sum_z_dr2) / 0.002192777   # -17.8%  sum_z_dr2 < 0.0078
        - 0.1134909 * max(0.0, 0.00652 - Q.sum_z_dr2_top50) / 0.001512518   # -11.3%  sum_z_dr2_top50 < 0.00652
        + 0.1026927 * max(0.0, 0.0138 - Q.lam1_plus_lam2) / 0.006759253   # +10.3%  lam1_plus_lam2 < 0.0138
        - 0.05910311 * max(0.0, 0.0474 - Q.tau2) / 0.02032181   # -5.9%  tau2 < 0.0474
        + 0.03492855 * max(0.0, 0.24 - Q.tau21_b2) * max(0.0, 0.0146 - Q.sum_zz_dr2) / 0.0003564909   # +3.5%  tau21_b2 < 0.24 and sum_zz_dr2 < 0.0146
        + 0.02767093 * max(0.0, 0.0343 - Q.e2) / 0.008978264   # +2.8%  e2 < 0.0343
        + 0.02336237 * max(0.0, Q.psi_0p3 - 0.996) / 0.001712455   # +2.3%  psi_0p3 > 0.996
        + 0.02140455 * max(0.0, 0.0024 - Q.lam2) * max(0.0, 7.14 - Q.log_sum_pt) / 0.0002829249   # +2.1%  lam2 < 0.0024 and log_sum_pt < 7.14
        + 0.02054402 * max(0.0, 15.1 - Q.n_dr_0p2_0p4) / 7.563736   # +2.1%  n_dr_0p2_0p4 < 15.1
        - 0.01856987 * max(0.0, 0.23 - Q.tau21_b2) * max(0.0, 0.00822 - Q.sum_zz_dr2) / 6.07417e-05   # -1.9%  tau21_b2 < 0.23 and sum_zz_dr2 < 0.00822
        + 0.005823936 * max(0.0, 0.224 - Q.tau21_b2) * max(0.0, Q.sum_pt_top30 - 983.0) / 2.318924   # +0.6%  tau21_b2 < 0.224 and sum_pt_top30 > 983
        - 0.0008919486 * max(0.0, Q.sum_pt_top15 - 1080.0) * max(0.0, Q.mean_phi - 1.99e-05) / 0.0002140403   # -0.1%  sum_pt_top15 > 1080 and mean_phi > 1.99e-05
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 22.58;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.57981 * (0.07085975
        - 0.2459038 * max(0.0, 0.0254 - Q.sum_z_dr2) * max(0.0, 7.15 - Q.log_sum_pt) / 0.003365128   # -24.6%  sum_z_dr2 < 0.0254 and log_sum_pt < 7.15
        + 0.2115204 * max(0.0, 0.0259 - Q.sum_z_dr2) / 0.01705746   # +21.2%  sum_z_dr2 < 0.0259
        - 0.1125737 * max(0.0, 0.00979 - Q.sum_zz_dr2) / 0.003636472   # -11.3%  sum_zz_dr2 < 0.00979
        + 0.07719583 * max(0.0, 0.00758 - Q.sum_z_dr2_top30) * max(0.0, 1250.0 - Q.sum_pt) / 0.4736596   # +7.7%  sum_z_dr2_top30 < 0.00758 and sum_pt < 1250
        + 0.06724466 * max(0.0, 1240.0 - Q.sum_pt_top50) / 212.0631   # +6.7%  sum_pt_top50 < 1240
        + 0.05543532 * max(0.0, 0.122 - Q.tau1) / 0.04585052   # +5.5%  tau1 < 0.122
        - 0.04579939 * max(0.0, 0.00821 - Q.lam1_plus_lam2) / 0.002468118   # -4.6%  lam1_plus_lam2 < 0.00821
        + 0.03984856 * max(0.0, 1070.0 - Q.sum_pt) / 55.2008   # +4.0%  sum_pt < 1070
        - 0.03755775 * max(0.0, 0.00729 - Q.sum_z_dr2_top30) / 0.002225845   # -3.8%  sum_z_dr2_top30 < 0.00729
        + 0.02660124 * max(0.0, 0.00359 - Q.sum_z_dr2) / 0.000484396   # +2.7%  sum_z_dr2 < 0.00359
        - 0.02210662 * max(0.0, 0.0254 - Q.sum_z_dr2) * max(0.0, Q.z_top50_slots - 0.973) / 0.0003869484   # -2.2%  sum_z_dr2 < 0.0254 and z_top50_slots > 0.973
        - 0.01944805 * max(0.0, 18.4 - Q.n_dr_0p2_0p4) / 10.30829   # -1.9%  n_dr_0p2_0p4 < 18.4
        - 0.01196422 * max(0.0, 0.00748 - Q.sum_z_dr2_top30) * max(0.0, 1060.0 - Q.sum_pt_top40) / 0.1120954   # -1.2%  sum_z_dr2_top30 < 0.00748 and sum_pt_top40 < 1060
        - 0.01140998 * max(0.0, Q.psi_0p3 - 0.994) / 0.002957925   # -1.1%  psi_0p3 > 0.994
        - 0.008905994 * max(0.0, 0.00115 - Q.lam2) / 0.000477662   # -0.9%  lam2 < 0.00115
        + 0.0064845 * max(0.0, 0.0153 - Q.sum_z_dr2) * max(0.0, 988.0 - Q.sum_pt) / 0.07212748   # +0.6%  sum_z_dr2 < 0.0153 and sum_pt < 988
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 4.815;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.8148 * (-0.03198471
        + 0.2150518 * max(0.0, 1190.0 - Q.sum_pt_top30) / 204.2271   # +21.5%  sum_pt_top30 < 1190
        + 0.2119317 * max(0.0, 0.00686 - Q.lam1_plus_lam2) / 0.001645821   # +21.2%  lam1_plus_lam2 < 0.00686
        - 0.1931055 * max(0.0, 0.0118 - Q.sum_z_dr2_top20) / 0.006076892   # -19.3%  sum_z_dr2_top20 < 0.0118
        + 0.1074953 * max(0.0, 0.00565 - Q.sum_z_dr2_top40) / 0.001195309   # +10.7%  sum_z_dr2_top40 < 0.00565
        - 0.06459464 * max(0.0, 0.0602 - Q.tau1) / 0.009688795   # -6.5%  tau1 < 0.0602
        + 0.06408874 * max(0.0, 985.0 - Q.sum_pt) / 11.77765   # +6.4%  sum_pt < 985
        - 0.03969988 * max(0.0, Q.sum_z_dr2_top15 - 0.0217) / 0.0005954735   # -4.0%  sum_z_dr2_top15 > 0.0217
        - 0.02841614 * max(0.0, 973.0 - Q.sum_pt_top40) * max(0.0, 0.0368 - Q.tau3) / 0.1278673   # -2.8%  sum_pt_top40 < 973 and tau3 < 0.0368
        - 0.02099073 * max(0.0, Q.e3 - 0.000515) * max(0.0, Q.soft3_pt - 2.1) / 8.21676e-07   # -2.1%  e3 > 0.000515 and soft3_pt > 2.1
        - 0.01598118 * max(0.0, Q.e3 - 0.000531) / 7.733286e-06   # -1.6%  e3 > 0.000531
        + 0.0148994 * max(0.0, 2.51 - Q.pt_entropy) / 0.1058077   # +1.5%  pt_entropy < 2.51
        - 0.01393537 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.sum_pt - 1040.0) / 0.2683842   # -1.4%  lam1 < 0.012 and sum_pt > 1040
        + 0.009809552 * max(0.0, Q.sum_z_dr - 0.114) * max(0.0, 0.381 - Q.sj3_pairmin_over_m) / 0.0002970505   # +1.0%  sum_z_dr > 0.114 and sj3_pairmin_over_m < 0.381
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 12.37;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.36762 * (0.1600955
        - 0.2485989 * max(0.0, 0.121 - Q.sum_z_dr) / 0.05600323   # -24.9%  sum_z_dr < 0.121
        + 0.1185872 * max(0.0, 0.368 - Q.LHA) / 0.1136932   # +11.9%  LHA < 0.368
        - 0.1097258 * max(0.0, 0.00051 - Q.e3) / 0.000416272   # -11.0%  e3 < 0.00051
        + 0.06852717 * max(0.0, 696.0 - Q.sum_pt_top3) / 243.5397   # +6.9%  sum_pt_top3 < 696
        + 0.06510046 * max(0.0, 0.0893 - Q.z_dr_0p2_0p4) / 0.05669985   # +6.5%  z_dr_0p2_0p4 < 0.0893
        + 0.06135065 * max(0.0, 0.0259 - Q.tau4) / 0.009918454   # +6.1%  tau4 < 0.0259
        + 0.05883715 * max(0.0, Q.e2 - 0.0076) / 0.02339793   # +5.9%  e2 > 0.0076
        + 0.04320906 * max(0.0, 0.00608 - Q.sum_zz_dr2) / 0.001257396   # +4.3%  sum_zz_dr2 < 0.00608
        - 0.03551576 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -3.6%  n_dr_0p2_0p4 < 15
        - 0.03268433 * max(0.0, Q.z_dr_0_0p05 - 0.709) / 0.07699569   # -3.3%  z_dr_0_0p05 > 0.709
        + 0.03002994 * max(0.0, 0.082 - Q.M2) / 0.02086511   # +3.0%  M2 < 0.082
        - 0.02788428 * max(0.0, 0.0357 - Q.sum_z_dr) / 0.004165003   # -2.8%  sum_z_dr < 0.0357
        - 0.02728934 * max(0.0, 0.307 - Q.tau21_b2) / 0.0883519   # -2.7%  tau21_b2 < 0.307
        - 0.02229991 * max(0.0, Q.sum_z_dr2 - 0.014) / 0.002224168   # -2.2%  sum_z_dr2 > 0.014
        + 0.01979893 * max(0.0, Q.n_dr_0_0p05 - 5.86) / 7.652051   # +2.0%  n_dr_0_0p05 > 5.86
        - 0.0159912 * max(0.0, Q.e2 - 0.00978) * max(0.0, Q.log_sum_pt - 6.89) / 0.001130132   # -1.6%  e2 > 0.00978 and log_sum_pt > 6.89
        - 0.01164565 * max(0.0, Q.psi_0p3 - 0.999) / 0.0002647591   # -1.2%  psi_0p3 > 0.999
        - 0.002924257 * max(0.0, Q.sum_z_dr2 - 0.0289) / 0.0002232476   # -0.3%  sum_z_dr2 > 0.0289
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 8.457;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.456508 * (-0.04883813
        + 0.3229105 * max(0.0, 0.00826 - Q.sum_zz_dr2) / 0.002505225   # +32.3%  sum_zz_dr2 < 0.00826
        - 0.2141824 * max(0.0, 0.0081 - Q.sum_zz_dr2) * max(0.0, 1120.0 - Q.sum_pt) / 0.2030533   # -21.4%  sum_zz_dr2 < 0.0081 and sum_pt < 1120
        - 0.1316278 * max(0.0, 0.00632 - Q.sum_z_dr2_top50) / 0.001410788   # -13.2%  sum_z_dr2_top50 < 0.00632
        - 0.1146859 * max(0.0, 0.00476 - Q.sum_zz_dr2) / 0.0008015222   # -11.5%  sum_zz_dr2 < 0.00476
        + 0.1011414 * max(0.0, 0.00944 - Q.sum_zz_dr2) * max(0.0, 1110.0 - Q.sum_pt) / 0.2615606   # +10.1%  sum_zz_dr2 < 0.00944 and sum_pt < 1110
        + 0.03867567 * max(0.0, 10.4 - Q.n_dr_0p2_0p4) / 4.08316   # +3.9%  n_dr_0p2_0p4 < 10.4
        - 0.03426607 * max(0.0, Q.C2_b2 - 0.00676) / 0.009992113   # -3.4%  C2_b2 > 0.00676
        + 0.02782362 * max(0.0, Q.psi_0p2 - 0.993) / 0.001656977   # +2.8%  psi_0p2 > 0.993
        - 0.01468663 * max(0.0, 0.00606 - Q.lam1) * max(0.0, 3.37 - Q.D2_b2) / 0.0006677292   # -1.5%  lam1 < 0.00606 and D2_b2 < 3.37
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 6.334;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.334267 * (-0.01244027
        + 0.2634492 * max(0.0, 0.0064 - Q.sum_z_dr2) * max(0.0, Q.log_sum_pt - 6.82) / 0.0002125806   # +26.3%  sum_z_dr2 < 0.0064 and log_sum_pt > 6.82
        - 0.2516623 * max(0.0, 0.00757 - Q.sum_zz_dr2) * max(0.0, Q.log_sum_pt - 6.81) / 0.000325991   # -25.2%  sum_zz_dr2 < 0.00757 and log_sum_pt > 6.81
        + 0.1767166 * max(0.0, 0.00747 - Q.sum_zz_dr2) / 0.001991762   # +17.7%  sum_zz_dr2 < 0.00747
        + 0.0662846 * max(0.0, Q.psi_0p3 - 0.994) * max(0.0, 0.00759 - Q.sum_z_dr2) / 7.116345e-06   # +6.6%  psi_0p3 > 0.994 and sum_z_dr2 < 0.00759
        - 0.06176464 * max(0.0, 0.00552 - Q.sum_z_dr2_top30) / 0.001278542   # -6.2%  sum_z_dr2_top30 < 0.00552
        - 0.05697057 * max(0.0, Q.psi_0p3 - 0.994) / 0.002957925   # -5.7%  psi_0p3 > 0.994
        - 0.04781591 * max(0.0, 0.00273 - Q.lam1) / 0.0003858328   # -4.8%  lam1 < 0.00273
        - 0.04310134 * max(0.0, Q.sum_pt - 1080.0) / 26.76622   # -4.3%  sum_pt > 1080
        - 0.01923071 * max(0.0, Q.z_dr_0_0p05 - 0.893) / 0.01250641   # -1.9%  z_dr_0_0p05 > 0.893
        + 0.01033197 * max(0.0, Q.LHA - 0.382) / 0.005740832   # +1.0%  LHA > 0.382
        + 0.002672152 * max(0.0, 0.0886 - Q.psi_0p1) * max(0.0, Q.sum_pt_top50 - 965.0) / 0.135409   # +0.3%  psi_0p1 < 0.0886 and sum_pt_top50 > 965
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 9.496;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.495524 * (0.09288587
        + 0.2481583 * max(0.0, Q.log_sum_pt - 6.81) / 0.1394316   # +24.8%  log_sum_pt > 6.81
        - 0.1465026 * max(0.0, 1090.0 - Q.sum_pt) / 70.97545   # -14.7%  sum_pt < 1090
        - 0.09314044 * max(0.0, Q.sum_pt - 1020.0) / 46.54828   # -9.3%  sum_pt > 1020
        - 0.08128436 * max(0.0, Q.sum_z_dr2_top50 - 0.0134) / 0.002263453   # -8.1%  sum_z_dr2_top50 > 0.0134
        + 0.0726735 * max(0.0, Q.sum_z_dr2_top40 - 0.0126) * max(0.0, 1220.0 - Q.sum_pt_top50) / 0.561035   # +7.3%  sum_z_dr2_top40 > 0.0126 and sum_pt_top50 < 1220
        - 0.07200762 * max(0.0, Q.sum_z_dr2_top40 - 0.00894) / 0.003136468   # -7.2%  sum_z_dr2_top40 > 0.00894
        + 0.07090952 * max(0.0, 1070.0 - Q.sum_pt_top50) / 60.65974   # +7.1%  sum_pt_top50 < 1070
        + 0.05753349 * max(0.0, Q.sum_z_dr2_top50 - 0.00656) / 0.004107599   # +5.8%  sum_z_dr2_top50 > 0.00656
        - 0.03069739 * max(0.0, Q.lam1_plus_lam2 - 0.02) / 0.001194622   # -3.1%  lam1_plus_lam2 > 0.02
        + 0.03041858 * max(0.0, 1090.0 - Q.sum_pt) * max(0.0, Q.tau21 - 0.099) / 25.78932   # +3.0%  sum_pt < 1090 and tau21 > 0.099
        - 0.02361761 * max(0.0, 25.2 - Q.n_pt_above_5) / 2.705206   # -2.4%  n_pt_above_5 < 25.2
        + 0.02137874 * max(0.0, Q.e2 - 0.038) / 0.004202947   # +2.1%  e2 > 0.038
        - 0.02058942 * max(0.0, 0.122 - Q.sum_z_dr) * max(0.0, 0.0128 - Q.C2_b2) / 0.0003189353   # -2.1%  sum_z_dr < 0.122 and C2_b2 < 0.0128
        + 0.01977535 * max(0.0, Q.sum_pt_top10 - 873.0) * max(0.0, 0.0069 - Q.C2_b2) / 0.04591132   # +2.0%  sum_pt_top10 > 873 and C2_b2 < 0.0069
        - 0.006162562 * max(0.0, Q.sum_z_dr2_top40 - 0.028) / 0.0002312915   # -0.6%  sum_z_dr2_top40 > 0.028
        - 0.00515048 * max(0.0, 0.124 - Q.sum_z_dr) * max(0.0, 854.0 - Q.sum_pt_top40) / 0.1446938   # -0.5%  sum_z_dr < 0.124 and sum_pt_top40 < 854
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 8.377;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.377245 * (-0.104569
        - 0.2967686 * max(0.0, 0.00736 - Q.sum_zz_dr2) / 0.001927212   # -29.7%  sum_zz_dr2 < 0.00736
        - 0.2104442 * max(0.0, 0.0564 - Q.tau1) / 0.008475684   # -21.0%  tau1 < 0.0564
        + 0.1722444 * max(0.0, Q.psi_0p3 - 0.995) / 0.002316105   # +17.2%  psi_0p3 > 0.995
        + 0.09611149 * max(0.0, 0.014 - Q.sum_z_dr2_top30) * max(0.0, Q.sum_pt - 904.0) / 1.161832   # +9.6%  sum_z_dr2_top30 < 0.014 and sum_pt > 904
        - 0.07668861 * max(0.0, Q.psi_0p3 - 0.996) * max(0.0, 1210.0 - Q.sum_pt_top40) / 0.3059235   # -7.7%  psi_0p3 > 0.996 and sum_pt_top40 < 1210
        + 0.04639431 * max(0.0, 0.323 - Q.tau21_b2) / 0.09740765   # +4.6%  tau21_b2 < 0.323
        + 0.0333461 * max(0.0, 0.0353 - Q.M3) / 0.01168822   # +3.3%  M3 < 0.0353
        - 0.02842544 * max(0.0, 17.5 - Q.n_dr_0p1_0p2) / 6.983192   # -2.8%  n_dr_0p1_0p2 < 17.5
        - 0.02209348 * max(0.0, 0.342 - Q.tau21_b2) * max(0.0, 0.00885 - Q.sum_z_dr2_top40) / 0.000195029   # -2.2%  tau21_b2 < 0.342 and sum_z_dr2_top40 < 0.00885
        - 0.01748343 * max(0.0, Q.LHA - 0.326) / 0.01536862   # -1.7%  LHA > 0.326
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 2.924;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.92366 * (-0.1867522
        + 0.2989521 * max(0.0, 0.00958 - Q.sum_z_dr2_top5) / 0.005567097   # +29.9%  sum_z_dr2_top5 < 0.00958
        - 0.1697104 * max(0.0, Q.sum_pt_top30 - 988.0) / 43.52417   # -17.0%  sum_pt_top30 > 988
        + 0.1564826 * max(0.0, Q.sum_pt_top20 - 834.0) / 114.3755   # +15.6%  sum_pt_top20 > 834
        - 0.1366617 * max(0.0, 0.0697 - Q.tau1) / 0.01310008   # -13.7%  tau1 < 0.0697
        + 0.08631377 * max(0.0, 0.127 - Q.z_dr_0p1_0p2) / 0.05098023   # +8.6%  z_dr_0p1_0p2 < 0.127
        + 0.0477304 * max(0.0, Q.sum_pt_top50 - 1120.0) / 17.8678   # +4.8%  sum_pt_top50 > 1120
        + 0.04519319 * max(0.0, Q.sj2_dr - 0.272) / 0.01048647   # +4.5%  sj2_dr > 0.272
        - 0.03045294 * max(0.0, Q.sj3_dr_max - 0.369) / 0.01012902   # -3.0%  sj3_dr_max > 0.369
        - 0.02850299 * max(0.0, 0.00819 - Q.sum_z_dr2_top10) * max(0.0, Q.tau4 - 0.019) / 4.604037e-06   # -2.9%  sum_z_dr2_top10 < 0.00819 and tau4 > 0.019
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.7068613445378151, 2.8669506827731093, 0.27578214285714286, 0.44664611344537813, 0.8921509453781512, 1.0448508403361345, 0.5418072478991597, 0.5439821428571429, 1.324263655462185, 1.2662955882352942, 1.2641714285714285, 0.6456932773109244, 0.4940747899159664, 1.482670325630252, 0.341549737394958, 0.3824295168067227]
T = [4.148502468487395, 2.7224538307510504, 4.571304303440127, 4.617967020089286, 4.094519491859244]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +43%, n4 -19%, n9 +16%, n3 -8%, n5 -8%, n12 +3% ...
            + 0.4319255 * h[1] / H_AVG[1]
            - 0.188172 * h[4] / H_AVG[4]
            + 0.1573902 * h[9] / H_AVG[9]
            - 0.08074832 * h[3] / H_AVG[3]
            - 0.07870693 * h[5] / H_AVG[5]
            + 0.02791339 * h[12] / H_AVG[12]
            + 0.02040673 * h[6] / H_AVG[6]
            + 0.009975465 * h[8] / H_AVG[8]
            - 0.0047614 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -35%, n9 +26%, n1 -20%, n12 +6%, n11 -6%, n6 +4% ...
            - 0.3481824 * h[4] / H_AVG[4]
            + 0.2616358 * h[9] / H_AVG[9]
            - 0.1974517 * h[1] / H_AVG[1]
            + 0.06238424 * h[12] / H_AVG[12]
            - 0.05929332 * h[11] / H_AVG[11]
            + 0.04353438 * h[6] / H_AVG[6]
            + 0.01266239 * h[2] / H_AVG[2]
            + 0.007600356 * h[8] / H_AVG[8]
            - 0.007255469 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +13%, n0 +12%, n14 -10%, n11 +8%, n7 -7% ...
            - 0.2534792 * h[8] / H_AVG[8]
            + 0.1321405 * h[5] / H_AVG[5]
            + 0.1159726 * h[0] / H_AVG[0]
            - 0.1027345 * h[14] / H_AVG[14]
            + 0.08386674 * h[11] / H_AVG[11]
            - 0.07437458 * h[7] / H_AVG[7]
            + 0.06708739 * h[4] / H_AVG[4]
            - 0.06059587 * h[9] / H_AVG[9]
            - 0.04390823 * h[12] / H_AVG[12]
            + 0.04274659 * h[3] / H_AVG[3]
            - 0.01568601 * h[15] / H_AVG[15]
            + 0.007407722 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -27%, n0 -21%, n5 +16%, n6 -11%, n7 +11%, n15 +5% ...
            - 0.2688406 * h[8] / H_AVG[8]
            - 0.210468 * h[0] / H_AVG[0]
            + 0.1555522 * h[5] / H_AVG[5]
            - 0.1136595 * h[6] / H_AVG[6]
            + 0.1067534 * h[7] / H_AVG[7]
            + 0.04658253 * h[15] / H_AVG[15]
            + 0.04231465 * h[3] / H_AVG[3]
            + 0.03343427 * h[12] / H_AVG[12]
            - 0.02239477 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -33%, n10 +30%, n5 -12%, n8 +7%, n12 -5%, n4 +4% ...
            - 0.328163 * h[13] / H_AVG[13]
            + 0.303923 * h[10] / H_AVG[10]
            - 0.1196169 * h[5] / H_AVG[5]
            + 0.06822214 * h[8] / H_AVG[8]
            - 0.04525025 * h[12] / H_AVG[12]
            + 0.0408542 * h[4] / H_AVG[4]
            - 0.0373658 * h[7] / H_AVG[7]
            - 0.03502513 * h[15] / H_AVG[15]
            + 0.0215795 * h[0] / H_AVG[0]
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
