"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network (step 4; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  12.8%   (on for 44% of jets)
  neuron  1:  12.6%   (on for 95% of jets)
  neuron  5:  11.5%   (on for 84% of jets)
  neuron  4:   9.6%   (on for 66% of jets)
  neuron  0:   7.7%   (on for 61% of jets)
  neuron  9:   7.5%   (on for 68% of jets)
  neuron 13:   7.2%   (on for 84% of jets)
  neuron 10:   6.5%   (on for 77% of jets)
  neuron 12:   4.9%   (on for 38% of jets)
  neuron  7:   3.9%   (on for 33% of jets)
  neuron  6:   3.8%   (on for 53% of jets)
  neuron  3:   3.7%   (on for 64% of jets)
  neuron 11:   3.7%   (on for 74% of jets)
  neuron 15:   2.2%   (on for 65% of jets)
  neuron 14:   1.8%   (on for 25% of jets)
  neuron  2:   0.7%   (on for 63% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.8% (the network: 81.1%); same class as the network for 93.2% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.log_sum_pt             natural log of the total pT
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_pt               pT [GeV] of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_z                pT share of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau4                   N-subjettiness τ4 (β=1)
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
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        log_sum_pt=math.log(tot),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_11=pt[11],
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_6=pt[6],
        soft1_pt=softp(1, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft7_pt=softp(7, 'pt'),
        soft1_z=softp(1, 'z'),
        soft3_z=softp(3, 'z'),
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau4=tau_n(4),
    )


def neuron_0(Q):
    # scale S = 7.113;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.112821 * (-0.1085364
        + 0.3378218 * max(0.0, 0.00788 - Q.e2_sq) / 0.00224567   # +33.8%  e2_sq < 0.00788
        - 0.1308166 * max(0.0, 0.00618 - Q.e2_sq) / 0.001301364   # -13.1%  e2_sq < 0.00618
        - 0.09192407 * max(0.0, Q.sum_pt - 1010.0) / 52.30716   # -9.2%  sum_pt > 1010
        + 0.07670793 * max(0.0, 0.0069 - Q.e2_sq) / 0.001668532   # +7.7%  e2_sq < 0.0069
        + 0.060236 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +6.0%  n_dr_0p2_0p4 < 18
        + 0.05729711 * max(0.0, Q.sum_pt_top40 - 860.0) / 165.6683   # +5.7%  sum_pt_top40 > 860
        - 0.05202142 * max(0.0, Q.psi_0p2 - 0.909) / 0.05799672   # -5.2%  psi_0p2 > 0.909
        - 0.05047377 * max(0.0, 0.00585 - Q.girth2_top50) * max(0.0, 1260.0 - Q.sum_pt) / 0.2458979   # -5.0%  girth2_top50 < 0.00585 and sum_pt < 1260
        - 0.03505963 * max(0.0, 0.00795 - Q.girth2) * max(0.0, 0.00596 - Q.girth2_top15) / 9.935174e-06   # -3.5%  girth2 < 0.00795 and girth2_top15 < 0.00596
        + 0.03143888 * max(0.0, Q.psi_0p3 - 0.994) / 0.002957925   # +3.1%  psi_0p3 > 0.994
        - 0.02947062 * max(0.0, 0.0719 - Q.tau1) / 0.01397462   # -2.9%  tau1 < 0.0719
        - 0.02439692 * max(0.0, 0.00332 - Q.girth2_top40) / 0.0004702735   # -2.4%  girth2_top40 < 0.00332
        + 0.01466731 * max(0.0, Q.sum_pt - 1120.0) / 19.83383   # +1.5%  sum_pt > 1120
        - 0.007667854 * max(0.0, 18.4 - Q.n_dr_0p2_0p4) * max(0.0, Q.log_sum_pt - 6.93) / 0.4070155   # -0.8%  n_dr_0p2_0p4 < 18.4 and log_sum_pt > 6.93
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 17.65;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.65365 * (0.2039239
        + 0.1268073 * max(0.0, Q.sum_pt - 973.0) / 80.52558   # +12.7%  sum_pt > 973
        - 0.11663 * max(0.0, 2.27 - Q.soft1_pt) / 1.647156   # -11.7%  soft1_pt < 2.27
        - 0.09647673 * max(0.0, Q.sum_pt_top50 - 940.0) / 103.2222   # -9.6%  sum_pt_top50 > 940
        + 0.09162357 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # +9.2%  log_sum_pt > 6.91
        - 0.06857939 * max(0.0, Q.tau1 - 0.0251) / 0.06272934   # -6.9%  tau1 > 0.0251
        - 0.06264293 * max(0.0, 7.13 - Q.log_sum_pt) / 0.191328   # -6.3%  log_sum_pt < 7.13
        + 0.05672154 * max(0.0, 1080.0 - Q.sum_pt_top50) / 68.58507   # +5.7%  sum_pt_top50 < 1080
        + 0.05280277 * max(0.0, 0.00138 - Q.soft1_z) / 0.000839785   # +5.3%  soft1_z < 0.00138
        + 0.04074783 * max(0.0, Q.n_particles - 37.9) / 10.86628   # +4.1%  n_particles > 37.9
        - 0.03537474 * max(0.0, 0.00962 - Q.e2_sq) / 0.003508389   # -3.5%  e2_sq < 0.00962
        - 0.03145467 * max(0.0, Q.log_sum_pt - 6.99) / 0.0210337   # -3.1%  log_sum_pt > 6.99
        + 0.02682132 * max(0.0, Q.z_top30_slots - 0.957) / 0.0191698   # +2.7%  z_top30_slots > 0.957
        + 0.02469944 * max(0.0, 593.0 - Q.sum_pt_top2) / 234.4275   # +2.5%  sum_pt_top2 < 593
        - 0.02378013 * max(0.0, Q.sum_pt - 1050.0) / 34.41033   # -2.4%  sum_pt > 1050
        + 0.01791113 * max(0.0, 0.239 - Q.LHA) / 0.03099969   # +1.8%  LHA < 0.239
        - 0.01550532 * max(0.0, Q.z_top30_slots - 0.958) * max(0.0, 10.4 - Q.ptdr0_3) / 0.1396558   # -1.6%  z_top30_slots > 0.958 and ptdr0_3 < 10.4
        + 0.01485266 * max(0.0, 0.0048 - Q.girth2_top30) / 0.001008476   # +1.5%  girth2_top30 < 0.0048
        - 0.01291201 * max(0.0, 0.000789 - Q.lam2) / 0.0002440515   # -1.3%  lam2 < 0.000789
        - 0.01013719 * max(0.0, 0.0339 - Q.M3) * max(0.0, Q.psi_0p3 - 0.929) / 0.0005829262   # -1.0%  M3 < 0.0339 and psi_0p3 > 0.929
        - 0.009792106 * max(0.0, 28.6 - Q.pt_11) / 8.231732   # -1.0%  pt_11 < 28.6
        - 0.00926315 * max(0.0, Q.z_top5 - 0.656) * max(0.0, 27.7 - Q.pt1_dr01) / 0.8055584   # -0.9%  z_top5 > 0.656 and pt1_dr01 < 27.7
        - 0.009221291 * max(0.0, 6.75 - Q.n_dr_0p2_0p4) / 1.912919   # -0.9%  n_dr_0p2_0p4 < 6.75
        + 0.006646676 * max(0.0, Q.tau1 - 0.128) / 0.009167036   # +0.7%  tau1 > 0.128
        - 0.006158128 * max(0.0, Q.z_dr_0_0p05 - 0.802) * max(0.0, 10.4 - Q.n_dr_0p05_0p1) / 0.2358208   # -0.6%  z_dr_0_0p05 > 0.802 and n_dr_0p05_0p1 < 10.4
        - 0.005678188 * max(0.0, Q.psi_0p3 - 0.997) / 0.001157514   # -0.6%  psi_0p3 > 0.997
        - 0.005568093 * max(0.0, 7.04 - Q.n_dr_0p1_0p2) / 0.9879109   # -0.6%  n_dr_0p1_0p2 < 7.04
        + 0.005343239 * max(0.0, Q.n_particles - 37.3) * max(0.0, 0.01 - Q.zdr_0) / 0.03480725   # +0.5%  n_particles > 37.3 and zdr_0 < 0.01
        + 0.005188597 * max(0.0, 0.226 - Q.tau21) / 0.01413544   # +0.5%  tau21 < 0.226
        + 0.004944749 * max(0.0, Q.n_dr_0_0p05 - 15.2) / 2.645238   # +0.5%  n_dr_0_0p05 > 15.2
        + 0.004605353 * max(0.0, Q.z_top30_slots - 0.939) * max(0.0, Q.pt1_dr01 - -0.337) / 0.181882   # +0.5%  z_top30_slots > 0.939 and pt1_dr01 > -0.337
        - 0.001109718 * max(0.0, Q.n_dr_0_0p05 - 30.0) / 0.212479   # -0.1%  n_dr_0_0p05 > 30
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 2.824;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.823969 * (-0.1639536
        - 0.4211432 * max(0.0, Q.LHA - 0.371) / 0.00729629   # -42.1%  LHA > 0.371
        + 0.2310643 * max(0.0, 0.0118 - Q.lam1) / 0.005723846   # +23.1%  lam1 < 0.0118
        - 0.1415707 * max(0.0, Q.log_sum_pt - 7.06) * max(0.0, Q.girth2_top30 - 0.0119) / 1.315103e-05   # -14.2%  log_sum_pt > 7.06 and girth2_top30 > 0.0119
        - 0.103233 * max(0.0, 0.00695 - Q.girth2) / 0.001694924   # -10.3%  girth2 < 0.00695
        + 0.1029888 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # +10.3%  log_sum_pt > 6.91
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 1.491;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.490932 * (0.06908429
        + 0.2736224 * max(0.0, 45.5 - Q.n_particles) / 6.143863   # +27.4%  n_particles < 45.5
        - 0.2465135 * max(0.0, 24.5 - Q.n_dr_0_0p05) / 12.63007   # -24.7%  n_dr_0_0p05 < 24.5
        + 0.2230198 * max(0.0, 5.25 - Q.n_dr_0p2_0p4) / 1.213531   # +22.3%  n_dr_0p2_0p4 < 5.25
        - 0.1102699 * max(0.0, 44.8 - Q.n_particles) * max(0.0, Q.e2 - 0.0146) / 0.07086419   # -11.0%  n_particles < 44.8 and e2 > 0.0146
        + 0.07202973 * max(0.0, 0.21 - Q.tau21_b2) / 0.0413044   # +7.2%  tau21_b2 < 0.21
        - 0.05298044 * max(0.0, 4.97 - Q.n_dr_0p2_0p4) * max(0.0, 0.034 - Q.e2) / 0.007668956   # -5.3%  n_dr_0p2_0p4 < 4.97 and e2 < 0.034
        - 0.02156418 * max(0.0, 4.92 - Q.n_dr_0p2_0p4) * max(0.0, 996.0 - Q.sum_pt_top30) / 19.4853   # -2.2%  n_dr_0p2_0p4 < 4.92 and sum_pt_top30 < 996
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 7.208;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.207666 * (0.1162651
        + 0.1623196 * max(0.0, 0.00988 - Q.width) / 0.003702358   # +16.2%  width < 0.00988
        - 0.1244759 * max(0.0, 0.00798 - Q.girth2_top40) / 0.0024784   # -12.4%  girth2_top40 < 0.00798
        - 0.1065606 * max(0.0, 0.00779 - Q.girth2_top20) / 0.002920355   # -10.7%  girth2_top20 < 0.00779
        + 0.09787911 * max(0.0, 0.0111 - Q.girth2_top20) / 0.005511562   # +9.8%  girth2_top20 < 0.0111
        - 0.07948673 * max(0.0, Q.n_particles - 28.6) / 17.95968   # -7.9%  n_particles > 28.6
        - 0.07332489 * max(0.0, 1060.0 - Q.sum_pt_top50) / 53.00916   # -7.3%  sum_pt_top50 < 1060
        - 0.06711889 * max(0.0, 0.00604 - Q.e2_sq) / 0.001240437   # -6.7%  e2_sq < 0.00604
        - 0.05549247 * max(0.0, Q.psi_0p1 - 0.391) / 0.4052393   # -5.5%  psi_0p1 > 0.391
        + 0.05090879 * max(0.0, Q.girth2_top50 - 0.00463) / 0.005294857   # +5.1%  girth2_top50 > 0.00463
        + 0.04855745 * max(0.0, 1050.0 - Q.sum_pt_top40) / 55.46527   # +4.9%  sum_pt_top40 < 1050
        + 0.03630813 * max(0.0, 0.0146 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.978) / 0.0001328411   # +3.6%  girth2 < 0.0146 and psi_0p3 > 0.978
        + 0.03206402 * max(0.0, Q.n_particles - 29.7) * max(0.0, 2.35 - Q.soft1_pt) / 26.23232   # +3.2%  n_particles > 29.7 and soft1_pt < 2.35
        + 0.01673986 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, Q.sum_pt_top40 - 870.0) / 0.1971492   # +1.7%  psi_0p3 > 0.997 and sum_pt_top40 > 870
        + 0.01598136 * max(0.0, Q.n_dr_0_0p05 - 11.8) / 4.070259   # +1.6%  n_dr_0_0p05 > 11.8
        - 0.01477286 * max(0.0, Q.LHA - 0.37) / 0.007446002   # -1.5%  LHA > 0.37
        - 0.006674351 * max(0.0, Q.sj2_dr - 0.256) * max(0.0, 0.0336 - Q.C2_b2) / 0.0001742989   # -0.7%  sj2_dr > 0.256 and C2_b2 < 0.0336
        + 0.006262989 * max(0.0, Q.e2_sq - 0.0253) / 0.0005153143   # +0.6%  e2_sq > 0.0253
        - 0.00507199 * max(0.0, Q.lam1 - 0.0208) / 0.0004741532   # -0.5%  lam1 > 0.0208
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 8.852;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.851959 * (0.01434711
        + 0.2997633 * max(0.0, Q.sum_pt_top50 - 926.0) / 115.873   # +30.0%  sum_pt_top50 > 926
        - 0.1470014 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # -14.7%  log_sum_pt > 6.91
        - 0.1417349 * max(0.0, Q.e2_sq - 0.00797) / 0.003626102   # -14.2%  e2_sq > 0.00797
        + 0.1216516 * max(0.0, Q.width - 0.00702) / 0.004018117   # +12.2%  width > 0.00702
        + 0.05354813 * max(0.0, Q.n_pt_above_1 - 23.2) / 18.88469   # +5.4%  n_pt_above_1 > 23.2
        - 0.04733931 * max(0.0, 0.00951 - Q.girth2_top20) / 0.004245649   # -4.7%  girth2_top20 < 0.00951
        + 0.02884929 * max(0.0, 46.6 - Q.n_real_top50) / 6.702697   # +2.9%  n_real_top50 < 46.6
        - 0.02876968 * max(0.0, 0.377 - Q.N2) / 0.08632815   # -2.9%  N2 < 0.377
        - 0.02467632 * max(0.0, Q.n_particles - 45.3) / 6.559572   # -2.5%  n_particles > 45.3
        + 0.02447534 * max(0.0, 7.83 - Q.n_dr_0p2_0p4) / 2.490284   # +2.4%  n_dr_0p2_0p4 < 7.83
        - 0.02395252 * max(0.0, Q.sum_pt_top50 - 1010.0) * max(0.0, 6.64e-08 - Q.e4) / 3.077311e-06   # -2.4%  sum_pt_top50 > 1010 and e4 < 6.64e-08
        - 0.02360719 * max(0.0, Q.girth - 0.158) / 0.0005886476   # -2.4%  girth > 0.158
        - 0.01876195 * max(0.0, 73.5 - Q.n_particles) * max(0.0, 2.44 - Q.D2) / 19.22223   # -1.9%  n_particles < 73.5 and D2 < 2.44
        - 0.01586906 * max(0.0, Q.girth2 - 0.0293) * max(0.0, 0.00263 - Q.soft3_z) / 1.945599e-07   # -1.6%  girth2 > 0.0293 and soft3_z < 0.00263
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 9.254;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.254162 * (0.0296083
        + 0.2157868 * max(0.0, 0.0271 - Q.e2_sq) / 0.01815387   # +21.6%  e2_sq < 0.0271
        - 0.1711926 * max(0.0, 0.0135 - Q.girth2) / 0.006519524   # -17.1%  girth2 < 0.0135
        - 0.1237526 * max(0.0, 0.0101 - Q.width) / 0.00386901   # -12.4%  width < 0.0101
        + 0.08830132 * max(0.0, Q.psi_0p3 - 0.986) / 0.008872473   # +8.8%  psi_0p3 > 0.986
        - 0.0830922 * max(0.0, Q.e2 - 0.0125) / 0.01917578   # -8.3%  e2 > 0.0125
        + 0.07976451 * max(0.0, 0.00389 - Q.lam1) / 0.0007030035   # +8.0%  lam1 < 0.00389
        - 0.07151841 * max(0.0, 0.0141 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.985) / 7.974011e-05   # -7.2%  girth2 < 0.0141 and psi_0p3 > 0.985
        + 0.06026762 * max(0.0, Q.log_sum_pt - 6.81) / 0.1394316   # +6.0%  log_sum_pt > 6.81
        - 0.02596656 * max(0.0, 0.0242 - Q.e2_sq) * max(0.0, 7.0 - Q.log_sum_pt) / 0.001063269   # -2.6%  e2_sq < 0.0242 and log_sum_pt < 7
        + 0.02176296 * max(0.0, Q.e2 - 0.0476) * max(0.0, 0.459 - Q.z_dr_0_0p05) / 0.0008533811   # +2.2%  e2 > 0.0476 and z_dr_0_0p05 < 0.459
        + 0.01932704 * max(0.0, 0.24 - Q.LHA) / 0.03137817   # +1.9%  LHA < 0.24
        - 0.01607308 * max(0.0, Q.lam2 - 0.000746) / 0.0008853743   # -1.6%  lam2 > 0.000746
        + 0.01177475 * max(0.0, 0.0105 - Q.e2_sq) * max(0.0, 1010.0 - Q.sum_pt) / 0.05765369   # +1.2%  e2_sq < 0.0105 and sum_pt < 1010
        - 0.01141946 * max(0.0, Q.log_sum_pt - 6.8) * max(0.0, 0.00186 - Q.girth2_top3) / 0.0001046312   # -1.1%  log_sum_pt > 6.8 and girth2_top3 < 0.00186
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 21.35;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.34726 * (-0.04033305
        + 0.1553871 * max(0.0, 0.0139 - Q.width) / 0.006839357   # +15.5%  width < 0.0139
        - 0.1358226 * max(0.0, 0.00963 - Q.e2_sq) * max(0.0, 1170.0 - Q.sum_pt) / 0.4502235   # -13.6%  e2_sq < 0.00963 and sum_pt < 1170
        + 0.1155729 * max(0.0, 0.00973 - Q.e2_sq) / 0.003591214   # +11.6%  e2_sq < 0.00973
        - 0.08714934 * max(0.0, 0.00796 - Q.girth2) / 0.00229679   # -8.7%  girth2 < 0.00796
        - 0.08224056 * max(0.0, 0.0858 - Q.girth) / 0.02738862   # -8.2%  girth < 0.0858
        - 0.06503132 * max(0.0, 0.00637 - Q.girth2_top50) / 0.001435616   # -6.5%  girth2_top50 < 0.00637
        + 0.06101129 * max(0.0, 0.0565 - Q.girth) / 0.01076383   # +6.1%  girth < 0.0565
        - 0.05618378 * max(0.0, 0.245 - Q.LHA) / 0.03331583   # -5.6%  LHA < 0.245
        + 0.05349452 * max(0.0, 0.303 - Q.LHA) / 0.06274513   # +5.3%  LHA < 0.303
        - 0.0328377 * max(0.0, 0.00587 - Q.lam1) / 0.001442376   # -3.3%  lam1 < 0.00587
        + 0.02706218 * max(0.0, 0.00586 - Q.lam1) * max(0.0, 1160.0 - Q.sum_pt) / 0.1699128   # +2.7%  lam1 < 0.00586 and sum_pt < 1160
        + 0.01746483 * max(0.0, 0.0864 - Q.girth) * max(0.0, 1150.0 - Q.sum_pt_top50) / 3.186548   # +1.7%  girth < 0.0864 and sum_pt_top50 < 1150
        - 0.01624997 * max(0.0, Q.psi_0p2 - 0.907) / 0.05960348   # -1.6%  psi_0p2 > 0.907
        + 0.0160306 * max(0.0, 0.00243 - Q.lam2) * max(0.0, 7.13 - Q.log_sum_pt) / 0.0002737675   # +1.6%  lam2 < 0.00243 and log_sum_pt < 7.13
        + 0.01474868 * max(0.0, Q.psi_0p3 - 0.997) / 0.001157514   # +1.5%  psi_0p3 > 0.997
        - 0.01458829 * max(0.0, 0.0483 - Q.tau2) / 0.02104189   # -1.5%  tau2 < 0.0483
        + 0.01428681 * max(0.0, 0.237 - Q.tau21_b2) / 0.05304074   # +1.4%  tau21_b2 < 0.237
        - 0.008699486 * max(0.0, 0.234 - Q.tau21_b2) * max(0.0, 0.00808 - Q.e2_sq) / 5.767397e-05   # -0.9%  tau21_b2 < 0.234 and e2_sq < 0.00808
        - 0.008463748 * max(0.0, Q.max_dr - 0.228) / 0.1290556   # -0.8%  max_dr > 0.228
        - 0.005336284 * max(0.0, 0.00252 - Q.lam2) * max(0.0, Q.zdr_0 - 0.00222) / 1.054769e-05   # -0.5%  lam2 < 0.00252 and zdr_0 > 0.00222
        + 0.003945304 * max(0.0, 0.229 - Q.tau21_b2) * max(0.0, Q.sum_pt_top30 - 978.0) / 2.607475   # +0.4%  tau21_b2 < 0.229 and sum_pt_top30 > 978
        - 0.003125576 * max(0.0, 0.236 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.188) / 0.001853402   # -0.3%  tau21_b2 < 0.236 and sj2_dr > 0.188
        - 0.00253042 * max(0.0, 0.000995 - Q.mean_eta2) / 0.0001174294   # -0.3%  mean_eta2 < 0.000995
        - 0.002514137 * max(0.0, Q.psi_0p3 - 0.998) * max(0.0, 0.0363 - Q.dr_0) / 4.472494e-06   # -0.3%  psi_0p3 > 0.998 and dr_0 < 0.0363
        - 0.0002226391 * max(0.0, Q.sum_pt_top15 - 1080.0) * max(0.0, Q.mean_phi - 8.35e-05) / 0.000152821   # -0.0%  sum_pt_top15 > 1080 and mean_phi > 8.35e-05
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 21.34;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.34107 * (0.04667057
        - 0.2493015 * max(0.0, 0.0263 - Q.girth2) * max(0.0, 7.14 - Q.log_sum_pt) / 0.003367317   # -24.9%  girth2 < 0.0263 and log_sum_pt < 7.14
        + 0.1794214 * max(0.0, 0.0271 - Q.girth2) / 0.01814714   # +17.9%  girth2 < 0.0271
        + 0.1095285 * max(0.0, 1220.0 - Q.sum_pt_top50) / 193.1781   # +11.0%  sum_pt_top50 < 1220
        - 0.0963614 * max(0.0, 0.0102 - Q.e2_sq) / 0.003947132   # -9.6%  e2_sq < 0.0102
        + 0.08415177 * max(0.0, 0.00748 - Q.girth2_top30) * max(0.0, 1280.0 - Q.sum_pt) / 0.5251137   # +8.4%  girth2_top30 < 0.00748 and sum_pt < 1280
        + 0.0630308 * max(0.0, 0.12 - Q.tau1) / 0.04424819   # +6.3%  tau1 < 0.12
        - 0.06025488 * max(0.0, 0.00727 - Q.girth2_top30) / 0.00221326   # -6.0%  girth2_top30 < 0.00727
        - 0.05599855 * max(0.0, 0.00814 - Q.width) / 0.002419168   # -5.6%  width < 0.00814
        - 0.03438691 * max(0.0, 18.2 - Q.n_dr_0p2_0p4) / 10.1361   # -3.4%  n_dr_0p2_0p4 < 18.2
        + 0.01847958 * max(0.0, 1030.0 - Q.sum_pt) / 27.96979   # +1.8%  sum_pt < 1030
        + 0.01615195 * max(0.0, 0.00379 - Q.girth2) / 0.0005344186   # +1.6%  girth2 < 0.00379
        - 0.008822431 * max(0.0, 0.00735 - Q.girth2_top30) * max(0.0, 1050.0 - Q.sum_pt_top40) / 0.09320799   # -0.9%  girth2_top30 < 0.00735 and sum_pt_top40 < 1050
        + 0.008064723 * max(0.0, 0.0136 - Q.girth2) * max(0.0, 995.0 - Q.sum_pt) / 0.06670924   # +0.8%  girth2 < 0.0136 and sum_pt < 995
        + 0.007983406 * max(0.0, 17.4 - Q.n_dr_0p2_0p4) * max(0.0, Q.sj2_zsoft - 0.0986) / 1.607306   # +0.8%  n_dr_0p2_0p4 < 17.4 and sj2_zsoft > 0.0986
        - 0.006731605 * max(0.0, 0.137 - Q.tau1) * max(0.0, 0.544 - Q.planar_flow) / 0.006744585   # -0.7%  tau1 < 0.137 and planar_flow < 0.544
        + 0.001330599 * max(0.0, Q.sum_pt_top20 - 1060.0) / 11.54325   # +0.1%  sum_pt_top20 > 1060
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 3.964;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.963921 * (-0.2749802
        + 0.2985736 * max(0.0, 0.00673 - Q.width) / 0.001575929   # +29.9%  width < 0.00673
        + 0.2307895 * max(0.0, 1200.0 - Q.sum_pt_top30) / 213.7457   # +23.1%  sum_pt_top30 < 1200
        + 0.08846461 * max(0.0, Q.lam1 - 0.00749) / 0.002805334   # +8.8%  lam1 > 0.00749
        - 0.07644662 * max(0.0, Q.girth2 - 0.0258) / 0.0004720068   # -7.6%  girth2 > 0.0258
        - 0.05841728 * max(0.0, Q.e2_sq - 0.0199) / 0.0011998   # -5.8%  e2_sq > 0.0199
        + 0.04226925 * max(0.0, 956.0 - Q.sum_pt_top40) / 13.96267   # +4.2%  sum_pt_top40 < 956
        - 0.03514102 * max(0.0, Q.z_dr_0_0p05 - 0.857) / 0.02162985   # -3.5%  z_dr_0_0p05 > 0.857
        + 0.02895899 * max(0.0, Q.e2_sq - 0.0292) / 0.0002024535   # +2.9%  e2_sq > 0.0292
        + 0.02379004 * max(0.0, Q.LHA - 0.332) * max(0.0, 1070.0 - Q.sum_pt) / 1.130718   # +2.4%  LHA > 0.332 and sum_pt < 1070
        - 0.01872222 * max(0.0, 0.0107 - Q.lam1) * max(0.0, Q.sum_pt - 1070.0) / 0.177544   # -1.9%  lam1 < 0.0107 and sum_pt > 1070
        - 0.01816696 * max(0.0, 959.0 - Q.sum_pt_top40) * max(0.0, 0.0369 - Q.tau3) / 0.1008577   # -1.8%  sum_pt_top40 < 959 and tau3 < 0.0369
        - 0.0176605 * max(0.0, Q.e2_sq - 0.0293) * max(0.0, 3.79 - Q.soft7_pt) / 0.0003017449   # -1.8%  e2_sq > 0.0293 and soft7_pt < 3.79
        + 0.01727299 * max(0.0, 6.9 - Q.log_sum_pt) * max(0.0, 0.527 - Q.z_dr_0p1_0p2) / 0.00492581   # +1.7%  log_sum_pt < 6.9 and z_dr_0p1_0p2 < 0.527
        - 0.0134694 * max(0.0, 983.0 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 0.907) / 15.38664   # -1.3%  sum_pt_top40 < 983 and soft4_pt > 0.907
        - 0.01195698 * max(0.0, 0.00546 - Q.girth2_top40) * max(0.0, 1010.0 - Q.sum_pt) / 0.01858687   # -1.2%  girth2_top40 < 0.00546 and sum_pt < 1010
        + 0.0104439 * max(0.0, Q.e3 - 0.000524) / 8.069944e-06   # +1.0%  e3 > 0.000524
        + 0.009456098 * max(0.0, Q.z_dr_0_0p05 - 0.88) * max(0.0, Q.sum_pt_top40 - 1000.0) / 1.029759   # +0.9%  z_dr_0_0p05 > 0.88 and sum_pt_top40 > 1000
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 6.647;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.646735 * (0.1805398
        + 0.2262839 * max(0.0, Q.e2 - 0.00532) / 0.02557907   # +22.6%  e2 > 0.00532
        - 0.1207239 * max(0.0, 0.124 - Q.girth) / 0.0585708   # -12.1%  girth < 0.124
        + 0.1132986 * max(0.0, 0.00641 - Q.width) / 0.001410236   # +11.3%  width < 0.00641
        - 0.08807699 * max(0.0, Q.sum_pt_top3 - 288.0) / 183.5186   # -8.8%  sum_pt_top3 > 288
        + 0.05686574 * max(0.0, 0.0854 - Q.M2) / 0.02347649   # +5.7%  M2 < 0.0854
        - 0.05523385 * max(0.0, 0.0372 - Q.girth) / 0.004543623   # -5.5%  girth < 0.0372
        - 0.0539462 * max(0.0, Q.girth2 - 0.0137) / 0.00228386   # -5.4%  girth2 > 0.0137
        - 0.04963079 * max(0.0, 0.0101 - Q.girth2_top10) / 0.005399062   # -5.0%  girth2_top10 < 0.0101
        - 0.04816202 * max(0.0, 0.026 - Q.tau4) / 0.01000376   # -4.8%  tau4 < 0.026
        - 0.03290019 * max(0.0, 0.314 - Q.tau21_b2) / 0.09226955   # -3.3%  tau21_b2 < 0.314
        + 0.029908 * max(0.0, 0.0975 - Q.z_dr_0p2_0p4) / 0.06330909   # +3.0%  z_dr_0p2_0p4 < 0.0975
        + 0.02866053 * max(0.0, 0.0671 - Q.dr_0) / 0.02729211   # +2.9%  dr_0 < 0.0671
        - 0.01965707 * max(0.0, Q.e2 - 0.00673) * max(0.0, Q.log_sum_pt - 6.9) / 0.001126339   # -2.0%  e2 > 0.00673 and log_sum_pt > 6.9
        - 0.01792555 * max(0.0, 0.0888 - Q.M2) * max(0.0, 0.423 - Q.max_dr) / 0.002150656   # -1.8%  M2 < 0.0888 and max_dr < 0.423
        - 0.01624878 * max(0.0, 0.00529 - Q.girth2_top30) * max(0.0, Q.pt_6 - 26.4) / 0.01833639   # -1.6%  girth2_top30 < 0.00529 and pt_6 > 26.4
        + 0.01515146 * max(0.0, Q.girth2 - 0.0144) * max(0.0, 113.0 - Q.pt_3) / 0.1150945   # +1.5%  girth2 > 0.0144 and pt_3 < 113
        + 0.01371747 * max(0.0, Q.e2 - 0.00434) * max(0.0, Q.sj3_pairmin_over_m - 0.161) / 0.003589622   # +1.4%  e2 > 0.00434 and sj3_pairmin_over_m > 0.161
        - 0.01360885 * max(0.0, Q.e2_sq - 0.0259) / 0.0004545448   # -1.4%  e2_sq > 0.0259
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 6.386;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.38565 * (-0.07109691
        + 0.2174733 * max(0.0, 0.0084 - Q.e2_sq) / 0.002605456   # +21.7%  e2_sq < 0.0084
        - 0.1519493 * max(0.0, 0.00638 - Q.girth2_top30) / 0.001687469   # -15.2%  girth2_top30 < 0.00638
        + 0.1309262 * max(0.0, 0.0118 - Q.girth2_top30) / 0.005648981   # +13.1%  girth2_top30 < 0.0118
        - 0.1206876 * max(0.0, 0.0084 - Q.e2_sq) * max(0.0, 1120.0 - Q.sum_pt) / 0.2214565   # -12.1%  e2_sq < 0.0084 and sum_pt < 1120
        - 0.1002603 * max(0.0, 0.00606 - Q.lam1) / 0.001539008   # -10.0%  lam1 < 0.00606
        + 0.08330263 * max(0.0, 0.00615 - Q.lam1) * max(0.0, 1120.0 - Q.sum_pt) / 0.1360464   # +8.3%  lam1 < 0.00615 and sum_pt < 1120
        + 0.04296671 * max(0.0, 10.4 - Q.n_dr_0p2_0p4) * max(0.0, 21.9 - Q.n_dr_0p1_0p2) / 49.70478   # +4.3%  n_dr_0p2_0p4 < 10.4 and n_dr_0p1_0p2 < 21.9
        + 0.04144012 * max(0.0, 0.0241 - Q.e2) / 0.004141191   # +4.1%  e2 < 0.0241
        + 0.0330212 * max(0.0, 9.02 - Q.n_dr_0p2_0p4) / 3.190043   # +3.3%  n_dr_0p2_0p4 < 9.02
        - 0.02813348 * max(0.0, 9.75 - Q.n_dr_0p2_0p4) * max(0.0, 0.00618 - Q.girth2) / 0.006069276   # -2.8%  n_dr_0p2_0p4 < 9.75 and girth2 < 0.00618
        + 0.02403895 * max(0.0, 0.00458 - Q.e2_sq) / 0.0007488017   # +2.4%  e2_sq < 0.00458
        + 0.01588255 * max(0.0, Q.z_top30_slots - 0.968) * max(0.0, 1.11 - Q.D2_b2) / 0.004568486   # +1.6%  z_top30_slots > 0.968 and D2_b2 < 1.11
        - 0.009917716 * max(0.0, Q.C2_b2 - 0.013) / 0.007398489   # -1.0%  C2_b2 > 0.013
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 8.397;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.397361 * (-0.1007459
        - 0.2601982 * max(0.0, 0.00754 - Q.e2_sq) * max(0.0, Q.log_sum_pt - 6.81) / 0.0003232216   # -26.0%  e2_sq < 0.00754 and log_sum_pt > 6.81
        + 0.2555591 * max(0.0, 0.00637 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.82) / 0.0002103943   # +25.6%  girth2 < 0.00637 and log_sum_pt > 6.82
        + 0.1624951 * max(0.0, 0.00768 - Q.e2_sq) / 0.002118835   # +16.2%  e2_sq < 0.00768
        + 0.06875264 * max(0.0, Q.psi_0p3 - 0.989) / 0.0065384   # +6.9%  psi_0p3 > 0.989
        - 0.06516524 * max(0.0, Q.psi_0p3 - 0.994) / 0.002957925   # -6.5%  psi_0p3 > 0.994
        - 0.04839344 * max(0.0, 0.00272 - Q.lam1) / 0.0003833747   # -4.8%  lam1 < 0.00272
        - 0.03828228 * max(0.0, 0.00564 - Q.girth2_top30) / 0.001328389   # -3.8%  girth2_top30 < 0.00564
        - 0.03474327 * max(0.0, Q.sum_pt - 1080.0) / 26.76622   # -3.5%  sum_pt > 1080
        + 0.03473 * max(0.0, Q.psi_0p3 - 0.994) * max(0.0, 0.00767 - Q.girth2) / 7.291008e-06   # +3.5%  psi_0p3 > 0.994 and girth2 < 0.00767
        + 0.02361532 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +2.4%  n_dr_0p2_0p4 < 10
        + 0.004963181 * max(0.0, 6.87 - Q.log_sum_pt) / 0.009559088   # +0.5%  log_sum_pt < 6.87
        + 0.003102276 * max(0.0, 0.0784 - Q.psi_0p1) * max(0.0, Q.sum_pt_top50 - 948.0) / 0.1393098   # +0.3%  psi_0p1 < 0.0784 and sum_pt_top50 > 948
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 9.991;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.991183 * (0.1271121
        + 0.2958558 * max(0.0, Q.log_sum_pt - 6.81) / 0.1394316   # +29.6%  log_sum_pt > 6.81
        - 0.1342455 * max(0.0, 0.124 - Q.girth) / 0.0585708   # -13.4%  girth < 0.124
        - 0.1104655 * max(0.0, Q.sum_pt - 1010.0) / 52.30716   # -11.0%  sum_pt > 1010
        - 0.1074359 * max(0.0, Q.girth2_top50 - 0.0132) / 0.002303458   # -10.7%  girth2_top50 > 0.0132
        - 0.07058907 * max(0.0, 1080.0 - Q.sum_pt) / 62.97038   # -7.1%  sum_pt < 1080
        + 0.05875236 * max(0.0, Q.girth2_top40 - 0.0125) * max(0.0, 1190.0 - Q.sum_pt_top50) / 0.4974624   # +5.9%  girth2_top40 > 0.0125 and sum_pt_top50 < 1190
        + 0.05051344 * max(0.0, 1070.0 - Q.sum_pt_top50) / 60.65974   # +5.1%  sum_pt_top50 < 1070
        + 0.03902296 * max(0.0, 0.0355 - Q.e2) / 0.009698646   # +3.9%  e2 < 0.0355
        - 0.0364113 * max(0.0, 54.0 - Q.n_particles) / 11.024   # -3.6%  n_particles < 54
        - 0.02147901 * max(0.0, Q.e2_sq - 0.0292) / 0.0002024535   # -2.1%  e2_sq > 0.0292
        - 0.01844825 * max(0.0, Q.girth2_top40 - 0.0282) * max(0.0, Q.pt_6 - 20.4) / 0.0036355   # -1.8%  girth2_top40 > 0.0282 and pt_6 > 20.4
        + 0.01658309 * max(0.0, 1130.0 - Q.sum_pt) * max(0.0, Q.tau21 - 0.184) / 29.06748   # +1.7%  sum_pt < 1130 and tau21 > 0.184
        + 0.01348243 * max(0.0, Q.psi_0p3 - 0.998) / 0.0006668586   # +1.3%  psi_0p3 > 0.998
        - 0.01216114 * max(0.0, 800.0 - Q.sum_pt_top15) / 31.97478   # -1.2%  sum_pt_top15 < 800
        + 0.009623822 * max(0.0, Q.sum_pt_top10 - 881.0) / 25.03994   # +1.0%  sum_pt_top10 > 881
        - 0.004930514 * max(0.0, Q.log_sum_pt - 6.85) * max(0.0, Q.C2 - 0.0674) / 0.001195672   # -0.5%  log_sum_pt > 6.85 and C2 > 0.0674
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 21.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.18318 * (-0.08072444
        - 0.2038936 * max(0.0, 0.00181 - Q.girth2) / 0.0001244701   # -20.4%  girth2 < 0.00181
        + 0.2025571 * max(0.0, 0.0198 - Q.e2_sq) / 0.0117235   # +20.3%  e2_sq < 0.0198
        - 0.1510965 * max(0.0, 0.012 - Q.girth2_top30) / 0.005808899   # -15.1%  girth2_top30 < 0.012
        - 0.06771529 * max(0.0, 0.00838 - Q.girth2) / 0.002589214   # -6.8%  girth2 < 0.00838
        + 0.05272273 * max(0.0, 0.0123 - Q.girth2_top30) * max(0.0, Q.sum_pt - 904.0) / 0.9545596   # +5.3%  girth2_top30 < 0.0123 and sum_pt > 904
        + 0.04074313 * max(0.0, Q.psi_0p3 - 0.989) / 0.0065384   # +4.1%  psi_0p3 > 0.989
        + 0.0388842 * max(0.0, Q.psi_0p3 - 0.996) / 0.001712455   # +3.9%  psi_0p3 > 0.996
        - 0.03437351 * max(0.0, Q.psi_0p3 - 0.996) * max(0.0, 1220.0 - Q.sum_pt_top40) / 0.3221859   # -3.4%  psi_0p3 > 0.996 and sum_pt_top40 < 1220
        - 0.03378487 * max(0.0, 0.00605 - Q.e2_sq) / 0.001244645   # -3.4%  e2_sq < 0.00605
        + 0.03301902 * max(0.0, 0.0574 - Q.tau1) / 0.008787031   # +3.3%  tau1 < 0.0574
        - 0.0278524 * max(0.0, 0.00837 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.995) / 6.982276e-06   # -2.8%  girth2 < 0.00837 and psi_0p3 > 0.995
        + 0.02687983 * max(0.0, 0.0345 - Q.e2) / 0.009095851   # +2.7%  e2 < 0.0345
        - 0.02244692 * max(0.0, 0.0357 - Q.e2) * max(0.0, Q.sum_pt_top30 - 881.0) / 1.495273   # -2.2%  e2 < 0.0357 and sum_pt_top30 > 881
        - 0.01719891 * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 9.82015   # -1.7%  n_dr_0p1_0p2 < 21
        - 0.01145846 * max(0.0, Q.sum_pt - 1250.0) / 8.200224   # -1.1%  sum_pt > 1250
        + 0.009492833 * max(0.0, Q.sum_pt_top50 - 1250.0) / 7.259507   # +0.9%  sum_pt_top50 > 1250
        - 0.008819515 * max(0.0, Q.e2_sq - 0.00367) / 0.006125421   # -0.9%  e2_sq > 0.00367
        - 0.007964395 * max(0.0, 30.7 - Q.n_pt_above_5) / 5.642514   # -0.8%  n_pt_above_5 < 30.7
        - 0.007296696 * max(0.0, 0.138 - Q.z_2nd) / 0.02642174   # -0.7%  z_2nd < 0.138
        + 0.001800162 * max(0.0, 0.0142 - Q.e2_sq) * max(0.0, 0.992 - Q.z_top50_slots) / 1.455464e-05   # +0.2%  e2_sq < 0.0142 and z_top50_slots < 0.992
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 5.423;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.422862 * (-0.09404628
        + 0.3065558 * max(0.0, 0.0795 - Q.girth) / 0.02286671   # +30.7%  girth < 0.0795
        - 0.2378988 * max(0.0, 0.289 - Q.LHA) / 0.05420556   # -23.8%  LHA < 0.289
        + 0.105012 * max(0.0, 6.98 - Q.log_sum_pt) / 0.05852677   # +10.5%  log_sum_pt < 6.98
        - 0.09696746 * max(0.0, 0.0137 - Q.girth2_top30) / 0.007183622   # -9.7%  girth2_top30 < 0.0137
        + 0.07936857 * max(0.0, 0.00762 - Q.girth2_top5) / 0.003985229   # +7.9%  girth2_top5 < 0.00762
        + 0.05932147 * max(0.0, 0.126 - Q.z_dr_0p1_0p2) / 0.05042196   # +5.9%  z_dr_0p1_0p2 < 0.126
        - 0.04955008 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # -5.0%  sum_pt < 1000
        - 0.03783686 * max(0.0, 0.00408 - Q.girth2_top30) / 0.0007713687   # -3.8%  girth2_top30 < 0.00408
        + 0.02006124 * max(0.0, 20.8 - Q.n_pt_above_10) / 3.171701   # +2.0%  n_pt_above_10 < 20.8
        - 0.007427749 * max(0.0, 0.00522 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 10.2) / 0.003248359   # -0.7%  girth2_top5 < 0.00522 and n_dr_0p2_0p4 > 10.2
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6442299369747899, 2.9055588760504203, 0.2650766806722689, 0.4311856092436975, 0.7267267857142857, 1.0568578781512605, 0.5025996848739496, 0.4058609243697479, 1.1570207983193277, 1.0859153361344538, 1.1932813025210085, 0.8139301470588235, 0.5480367647058824, 1.4923592962184873, 0.24925241596638656, 0.370294012605042]
T = [3.9272218618697483, 2.499438786764706, 4.1857003463104, 4.219417190782563, 3.9418605403098734]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +46%, n4 -16%, n9 +14%, n5 -8%, n3 -8%, n12 +3% ...
            + 0.4624069 * h[1] / H_AVG[1]
            - 0.1619175 * h[4] / H_AVG[4]
            + 0.1425754 * h[9] / H_AVG[9]
            - 0.08409713 * h[5] / H_AVG[5]
            - 0.08234554 * h[3] / H_AVG[3]
            + 0.03270661 * h[12] / H_AVG[12]
            + 0.01999663 * h[6] / H_AVG[6]
            + 0.009206737 * h[8] / H_AVG[8]
            - 0.004747636 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -31%, n9 +24%, n1 -22%, n11 -8%, n12 +8%, n6 +4% ...
            - 0.3089282 * h[4] / H_AVG[4]
            + 0.2443858 * h[9] / H_AVG[9]
            - 0.2179658 * h[1] / H_AVG[1]
            - 0.08141129 * h[11] / H_AVG[11]
            + 0.07537198 * h[12] / H_AVG[12]
            + 0.04398735 * h[6] / H_AVG[6]
            + 0.01325681 * h[2] / H_AVG[2]
            - 0.007459683 * h[10] / H_AVG[10]
            + 0.007233004 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +15%, n11 +12%, n0 +12%, n14 -8%, n7 -6% ...
            - 0.2418695 * h[8] / H_AVG[8]
            + 0.1459722 * h[5] / H_AVG[5]
            + 0.1154576 * h[11] / H_AVG[11]
            + 0.1154341 * h[0] / H_AVG[0]
            - 0.08187927 * h[14] / H_AVG[14]
            - 0.0606023 * h[7] / H_AVG[7]
            + 0.05968233 * h[4] / H_AVG[4]
            - 0.05675131 * h[9] / H_AVG[9]
            - 0.05319061 * h[12] / H_AVG[12]
            + 0.04506861 * h[3] / H_AVG[3]
            - 0.01658746 * h[15] / H_AVG[15]
            + 0.007504713 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -21%, n5 +17%, n6 -12%, n7 +9%, n15 +5% ...
            - 0.2570751 * h[8] / H_AVG[8]
            - 0.209938 * h[0] / H_AVG[0]
            + 0.1722015 * h[5] / H_AVG[5]
            - 0.1153935 * h[6] / H_AVG[6]
            + 0.08717115 * h[7] / H_AVG[7]
            + 0.04936473 * h[15] / H_AVG[15]
            + 0.04470847 * h[3] / H_AVG[3]
            + 0.0405889 * h[12] / H_AVG[12]
            - 0.02355865 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +30%, n5 -13%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3430996 * h[13] / H_AVG[13]
            + 0.2979903 * h[10] / H_AVG[10]
            - 0.1256772 * h[5] / H_AVG[5]
            + 0.06191469 * h[8] / H_AVG[8]
            - 0.05213624 * h[12] / H_AVG[12]
            - 0.03522708 * h[15] / H_AVG[15]
            + 0.03456776 * h[4] / H_AVG[4]
            - 0.028958 * h[7] / H_AVG[7]
            + 0.02042912 * h[0] / H_AVG[0]
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
