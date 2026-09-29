"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  12.7%   (on for 53% of jets)
  neuron  1:  11.9%   (on for 94% of jets)
  neuron  5:  11.2%   (on for 80% of jets)
  neuron  4:  10.0%   (on for 70% of jets)
  neuron  0:   7.8%   (on for 62% of jets)
  neuron 13:   7.3%   (on for 89% of jets)
  neuron  9:   6.8%   (on for 69% of jets)
  neuron 10:   6.7%   (on for 82% of jets)
  neuron 12:   4.9%   (on for 55% of jets)
  neuron  7:   4.7%   (on for 36% of jets)
  neuron  6:   4.0%   (on for 67% of jets)
  neuron 11:   3.4%   (on for 70% of jets)
  neuron  3:   3.3%   (on for 50% of jets)
  neuron 15:   2.6%   (on for 69% of jets)
  neuron 14:   2.0%   (on for 39% of jets)
  neuron  2:   0.6%   (on for 40% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.1% (the network: 81.1%); same class as the network for 91.5% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_14                  pT of particle 14 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_pt              pT [GeV] of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_8                    pT of particle 8 / total pT
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_2               |Δφ| of particle 2
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
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
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
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
  Q.tau32                  N-subjettiness τ3/τ2
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_14=pt[14],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_6=pt[6],
        soft1_pt=softp(1, 'pt'),
        soft10_pt=softp(10, 'pt'),
        soft5_pt=softp(5, 'pt'),
        z_8=z[8],
        soft7_z=softp(7, 'z'),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sd_rg=softdrop("rg"),
        absphi_2=abs(phi[2]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
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
        n_pt_above_5=sum(1 for x in pt if x > 5),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
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
        tau32=tau(3) / max(tau(2), 1e-12),
    )


def neuron_0(Q):
    # scale S = 14.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.7756 * (-0.3248599
        + 0.3171535 * max(0.0, 0.00947 - Q.e2_sq) / 0.00339575   # +31.7%  e2_sq < 0.00947
        + 0.07903965 * max(0.0, 1210.0 - Q.sum_pt_top30) / 223.2999   # +7.9%  sum_pt_top30 < 1210
        + 0.06794115 * max(0.0, 0.0212 - Q.girth2_top15) / 0.01446501   # +6.8%  girth2_top15 < 0.0212
        + 0.06492042 * max(0.0, Q.sum_pt_top40 - 856.0) / 169.4767   # +6.5%  sum_pt_top40 > 856
        - 0.04165829 * max(0.0, 0.00582 - Q.e2_sq) / 0.001152671   # -4.2%  e2_sq < 0.00582
        + 0.04164538 * max(0.0, Q.girth2_top30 - 0.00746) / 0.00336249   # +4.2%  girth2_top30 > 0.00746
        - 0.03942926 * max(0.0, Q.LHA - 0.118) / 0.1467484   # -3.9%  LHA > 0.118
        - 0.03840092 * max(0.0, Q.psi_0p2 - 0.81) / 0.1429211   # -3.8%  psi_0p2 > 0.81
        - 0.03787911 * max(0.0, Q.sum_pt - 1010.0) / 52.30716   # -3.8%  sum_pt > 1010
        - 0.03462075 * max(0.0, 0.00819 - Q.girth2) * max(0.0, 0.00612 - Q.girth2_top15) / 1.076931e-05   # -3.5%  girth2 < 0.00819 and girth2_top15 < 0.00612
        + 0.03216959 * max(0.0, Q.girth2_top30 - 0.00336) / 0.005685706   # +3.2%  girth2_top30 > 0.00336
        - 0.03071471 * max(0.0, 0.00633 - Q.girth2_top50) * max(0.0, 1260.0 - Q.sum_pt) / 0.2890627   # -3.1%  girth2_top50 < 0.00633 and sum_pt < 1260
        - 0.02359683 * max(0.0, 0.0856 - Q.girth) / 0.02723885   # -2.4%  girth < 0.0856
        + 0.01935168 * max(0.0, 13.8 - Q.n_dr_0p2_0p4) / 6.543081   # +1.9%  n_dr_0p2_0p4 < 13.8
        - 0.0185719 * max(0.0, 0.0701 - Q.tau1) / 0.01325657   # -1.9%  tau1 < 0.0701
        + 0.01839138 * max(0.0, Q.psi_0p3 - 0.988) / 0.007304939   # +1.8%  psi_0p3 > 0.988
        - 0.01439147 * max(0.0, 0.00578 - Q.lam1) / 0.001398964   # -1.4%  lam1 < 0.00578
        - 0.01391151 * max(0.0, Q.lam1 - 0.0177) / 0.00083219   # -1.4%  lam1 > 0.0177
        + 0.01201868 * max(0.0, 60.2 - Q.n_particles) / 15.3089   # +1.2%  n_particles < 60.2
        + 0.01026583 * max(0.0, Q.sum_pt - 1140.0) / 17.19771   # +1.0%  sum_pt > 1140
        - 0.009617991 * max(0.0, 0.00455 - Q.e2_sq) / 0.0007401647   # -1.0%  e2_sq < 0.00455
        - 0.006893424 * max(0.0, Q.lam2 - 0.00264) / 0.0004467302   # -0.7%  lam2 > 0.00264
        + 0.00678058 * max(0.0, 0.0188 - Q.e2) / 0.002357345   # +0.7%  e2 < 0.0188
        - 0.006122701 * max(0.0, 17.2 - Q.n_dr_0p2_0p4) * max(0.0, Q.log_sum_pt - 6.91) / 0.4863795   # -0.6%  n_dr_0p2_0p4 < 17.2 and log_sum_pt > 6.91
        - 0.004094156 * max(0.0, 993.0 - Q.sum_pt_top50) / 16.75724   # -0.4%  sum_pt_top50 < 993
        - 0.003569928 * max(0.0, 0.0615 - Q.C2) / 0.01277187   # -0.4%  C2 < 0.0615
        - 0.0031832 * max(0.0, Q.n_for_90pct - 28.8) / 1.433954   # -0.3%  n_for_90pct > 28.8
        - 0.001880746 * max(0.0, Q.sum_pt_top20 - 1080.0) / 9.682632   # -0.2%  sum_pt_top20 > 1080
        - 0.001785231 * max(0.0, Q.e2 - 0.0483) / 0.001998323   # -0.2%  e2 > 0.0483
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 15.85;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.85217 * (0.1709545
        - 0.2166182 * max(0.0, Q.sum_pt_top50 - 932.0) / 110.4138   # -21.7%  sum_pt_top50 > 932
        + 0.1246521 * max(0.0, Q.sum_pt - 974.0) / 79.67769   # +12.5%  sum_pt > 974
        - 0.1011426 * max(0.0, 7.13 - Q.log_sum_pt) / 0.191328   # -10.1%  log_sum_pt < 7.13
        + 0.08960844 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # +9.0%  log_sum_pt > 6.91
        + 0.07941843 * max(0.0, Q.pt_entropy - 1.88) / 0.9835581   # +7.9%  pt_entropy > 1.88
        - 0.06930322 * max(0.0, 0.00976 - Q.e2_sq) / 0.003613837   # -6.9%  e2_sq < 0.00976
        + 0.05552565 * max(0.0, 0.0069 - Q.girth2_top30) / 0.001986912   # +5.6%  girth2_top30 < 0.0069
        + 0.04443408 * max(0.0, Q.z_top30_slots - 0.949) / 0.02412248   # +4.4%  z_top30_slots > 0.949
        + 0.04124639 * max(0.0, Q.n_real_top50 - 33.6) / 9.531263   # +4.1%  n_real_top50 > 33.6
        - 0.02952176 * max(0.0, Q.tau1 - 0.0273) / 0.06085617   # -3.0%  tau1 > 0.0273
        - 0.02816514 * max(0.0, Q.log_sum_pt - 6.98) / 0.02313361   # -2.8%  log_sum_pt > 6.98
        + 0.01969842 * max(0.0, Q.n_particles - 46.7) / 5.869597   # +2.0%  n_particles > 46.7
        - 0.01643639 * max(0.0, 0.000796 - Q.lam2) / 0.0002481451   # -1.6%  lam2 < 0.000796
        - 0.01448662 * max(0.0, 0.0366 - Q.M3) * max(0.0, Q.psi_0p3 - 0.941) / 0.000567023   # -1.4%  M3 < 0.0366 and psi_0p3 > 0.941
        + 0.01374532 * max(0.0, Q.n_particles - 29.9) * max(0.0, 0.0107 - Q.zdr_0) / 0.05904962   # +1.4%  n_particles > 29.9 and zdr_0 < 0.0107
        - 0.01345717 * max(0.0, 7.31 - Q.n_dr_0p2_0p4) / 2.206053   # -1.3%  n_dr_0p2_0p4 < 7.31
        + 0.009874916 * max(0.0, 0.00148 - Q.girth2_top20) / 0.0001890566   # +1.0%  girth2_top20 < 0.00148
        - 0.0095084 * max(0.0, Q.z_dr_0_0p05 - 0.759) * max(0.0, 11.3 - Q.n_dr_0p05_0p1) / 0.3667366   # -1.0%  z_dr_0_0p05 > 0.759 and n_dr_0p05_0p1 < 11.3
        + 0.007987587 * max(0.0, 0.25 - Q.tau21) / 0.01994025   # +0.8%  tau21 < 0.25
        + 0.006509429 * max(0.0, 930.0 - Q.sum_pt_top50) / 6.530922   # +0.7%  sum_pt_top50 < 930
        + 0.005166137 * max(0.0, Q.sum_pt_top20 - 992.0) / 24.08661   # +0.5%  sum_pt_top20 > 992
        + 0.003493616 * max(0.0, Q.z_top30_slots - 0.936) * max(0.0, Q.pt1_dr01 - 3.56) / 0.1331283   # +0.3%  z_top30_slots > 0.936 and pt1_dr01 > 3.56
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 6.769;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.769105 * (0.7416047
        - 0.7051109 * Q.z_top50_slots / 0.9923015   # -70.5%  z_top50_slots
        + 0.05354706 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922642   # +5.4%  log_sum_pt > 6.9
        - 0.0443548 * max(0.0, 6.98 - Q.log_sum_pt) / 0.05852677   # -4.4%  log_sum_pt < 6.98
        - 0.02845318 * max(0.0, 0.0072 - Q.girth2) / 0.00183431   # -2.8%  girth2 < 0.0072
        + 0.02818024 * max(0.0, 0.0117 - Q.lam1) / 0.00564364   # +2.8%  lam1 < 0.0117
        - 0.02097688 * max(0.0, 60.6 - Q.n_particles) / 15.60382   # -2.1%  n_particles < 60.6
        - 0.01715862 * max(0.0, Q.psi_0p3 - 0.988) / 0.007304939   # -1.7%  psi_0p3 > 0.988
        - 0.01265492 * max(0.0, Q.girth2_top15 - 0.00672) / 0.003103713   # -1.3%  girth2_top15 > 0.00672
        + 0.01078788 * max(0.0, 6.93 - Q.log_sum_pt) / 0.02518079   # +1.1%  log_sum_pt < 6.93
        + 0.01037052 * max(0.0, Q.e3 - 5.32e-05) / 6.749916e-05   # +1.0%  e3 > 5.32e-05
        - 0.008850591 * max(0.0, Q.log_sum_pt - 6.94) * max(0.0, Q.girth2_top30 - 0.00283) / 0.000139978   # -0.9%  log_sum_pt > 6.94 and girth2_top30 > 0.00283
        - 0.00729887 * max(0.0, Q.C2 - 0.048) / 0.02421903   # -0.7%  C2 > 0.048
        - 0.007218292 * max(0.0, Q.log_sum_pt - 7.07) / 0.01017945   # -0.7%  log_sum_pt > 7.07
        + 0.007151361 * max(0.0, Q.log_sum_pt - 6.92) * max(0.0, Q.z_top15_slots - 0.69) / 0.006143187   # +0.7%  log_sum_pt > 6.92 and z_top15_slots > 0.69
        - 0.007046364 * max(0.0, Q.sd_rg - 0.192) / 0.02373014   # -0.7%  sd_rg > 0.192
        + 0.005996194 * max(0.0, 0.418 - Q.psi_0p1) / 0.02899205   # +0.6%  psi_0p1 < 0.418
        - 0.00587074 * max(0.0, Q.log_sum_pt - 6.94) * max(0.0, 0.966 - Q.psi_0p1) / 0.004918275   # -0.6%  log_sum_pt > 6.94 and psi_0p1 < 0.966
        + 0.003705964 * max(0.0, Q.sd_rg - 0.28) / 0.006816865   # +0.4%  sd_rg > 0.28
        - 0.00350308 * max(0.0, Q.z_dr_0p1_0p2 - 0.429) / 0.02574671   # -0.4%  z_dr_0p1_0p2 > 0.429
        + 0.003394033 * max(0.0, Q.log_sum_pt - 7.0) * max(0.0, Q.girth2_top30 - 0.00847) / 3.523707e-05   # +0.3%  log_sum_pt > 7 and girth2_top30 > 0.00847
        + 0.002497218 * max(0.0, Q.sum_pt_top50 - 1020.0) * max(0.0, Q.z_dr_0p1_0p2 - 0.00203) / 4.913935   # +0.2%  sum_pt_top50 > 1020 and z_dr_0p1_0p2 > 0.00203
        - 0.002085565 * max(0.0, Q.n_dr_0p1_0p2 - 20.4) / 1.484481   # -0.2%  n_dr_0p1_0p2 > 20.4
        - 0.001983163 * max(0.0, Q.e2 - 0.0551) / 0.001167325   # -0.2%  e2 > 0.0551
        + 0.001159853 * max(0.0, Q.sj3_dr13 - 0.287) / 0.01113641   # +0.1%  sj3_dr13 > 0.287
        - 0.0006436764 * max(0.0, Q.LHA - 0.42) / 0.001807931   # -0.1%  LHA > 0.42
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 5.286;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.285882 * (0.09175385
        + 0.2040109 * max(0.0, 0.00969 - Q.girth2) / 0.003559001   # +20.4%  girth2 < 0.00969
        - 0.1235875 * max(0.0, 0.371 - Q.LHA) / 0.1162401   # -12.4%  LHA < 0.371
        - 0.07787813 * max(0.0, Q.e2 - 0.00732) / 0.02365831   # -7.8%  e2 > 0.00732
        - 0.07132774 * max(0.0, Q.z_top30_slots - 0.92) / 0.04440872   # -7.1%  z_top30_slots > 0.92
        - 0.06859233 * max(0.0, 8.6e-05 - Q.e3) / 4.143668e-05   # -6.9%  e3 < 8.6e-05
        + 0.05083379 * max(0.0, 46.5 - Q.n_real_top50) / 6.651024   # +5.1%  n_real_top50 < 46.5
        + 0.04450239 * max(0.0, 0.0313 - Q.e2) / 0.007328173   # +4.5%  e2 < 0.0313
        - 0.03976825 * max(0.0, Q.n_particles - 45.6) / 6.408849   # -4.0%  n_particles > 45.6
        - 0.03804962 * max(0.0, 0.00844 - Q.girth2_top30) / 0.003029002   # -3.8%  girth2_top30 < 0.00844
        + 0.03265504 * max(0.0, 0.0013 - Q.lam2) / 0.000585121   # +3.3%  lam2 < 0.0013
        - 0.03074175 * max(0.0, 0.00604 - Q.e2_sq) / 0.001240437   # -3.1%  e2_sq < 0.00604
        + 0.02951965 * max(0.0, 4.46 - Q.n_dr_0p2_0p4) / 0.9019502   # +3.0%  n_dr_0p2_0p4 < 4.46
        - 0.02623474 * max(0.0, Q.n_dr_0p1_0p2 - 5.17) / 7.879189   # -2.6%  n_dr_0p1_0p2 > 5.17
        - 0.02470158 * max(0.0, 15.9 - Q.n_dr_0_0p05) / 5.803095   # -2.5%  n_dr_0_0p05 < 15.9
        + 0.02447423 * max(0.0, 11.8 - Q.n_dr_0p2_0p4) / 5.053432   # +2.4%  n_dr_0p2_0p4 < 11.8
        - 0.02280777 * max(0.0, Q.n_for_90pct - 21.0) / 3.95276   # -2.3%  n_for_90pct > 21
        + 0.01390349 * max(0.0, 1060.0 - Q.sum_pt) / 47.7222   # +1.4%  sum_pt < 1060
        - 0.01197628 * max(0.0, 0.00412 - Q.girth2) / 0.0006206392   # -1.2%  girth2 < 0.00412
        + 0.0115435 * max(0.0, Q.psi_0p3 - 0.998) / 0.0006668586   # +1.2%  psi_0p3 > 0.998
        + 0.01114023 * max(0.0, 0.625 - Q.D2_b2) / 0.09621884   # +1.1%  D2_b2 < 0.625
        + 0.01069339 * max(0.0, 0.272 - Q.LHA) / 0.04521918   # +1.1%  LHA < 0.272
        - 0.01014829 * max(0.0, 0.0084 - Q.e2_sq) * max(0.0, 1050.0 - Q.sum_pt_top40) / 0.1008321   # -1.0%  e2_sq < 0.0084 and sum_pt_top40 < 1050
        + 0.007908654 * max(0.0, Q.LHA - 0.274) / 0.03426575   # +0.8%  LHA > 0.274
        - 0.005239868 * max(0.0, 4.19 - Q.n_dr_0p2_0p4) * max(0.0, 0.0325 - Q.e2) / 0.004726505   # -0.5%  n_dr_0p2_0p4 < 4.19 and e2 < 0.0325
        + 0.003912563 * max(0.0, Q.lam1 - 0.0227) / 0.0003162285   # +0.4%  lam1 > 0.0227
        + 0.002589747 * max(0.0, Q.girth2_top50 - 0.00732) / 0.0037711   # +0.3%  girth2_top50 > 0.00732
        + 0.001258588 * max(0.0, Q.sum_pt_top30 - 1120.0) / 11.79565   # +0.1%  sum_pt_top30 > 1120
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 4.463;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.463471 * (0.2215764
        - 0.1613157 * max(0.0, Q.n_particles - 24.5) / 21.62246   # -16.1%  n_particles > 24.5
        - 0.119296 * max(0.0, 0.00798 - Q.girth2_top20) / 0.003060196   # -11.9%  girth2_top20 < 0.00798
        - 0.1188981 * max(0.0, 0.0067 - Q.e2_sq) / 0.001560877   # -11.9%  e2_sq < 0.0067
        + 0.1088749 * max(0.0, 0.0108 - Q.girth2_top20) / 0.005270718   # +10.9%  girth2_top20 < 0.0108
        + 0.09934318 * max(0.0, 24.6 - Q.n_dr_0p2_0p4) / 15.89303   # +9.9%  n_dr_0p2_0p4 < 24.6
        + 0.05971442 * max(0.0, Q.e2_sq - 0.0108) / 0.002897104   # +6.0%  e2_sq > 0.0108
        - 0.05968985 * max(0.0, 0.0505 - Q.e2) / 0.02131391   # -6.0%  e2 < 0.0505
        + 0.05430681 * max(0.0, Q.n_dr_0_0p05 - 2.36) / 10.35884   # +5.4%  n_dr_0_0p05 > 2.36
        + 0.05333448 * max(0.0, 0.0113 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.978) / 8.688208e-05   # +5.3%  girth2 < 0.0113 and psi_0p3 > 0.978
        + 0.0442517 * max(0.0, Q.n_particles - 29.6) * max(0.0, 2.69 - Q.soft1_pt) / 31.75501   # +4.4%  n_particles > 29.6 and soft1_pt < 2.69
        - 0.02792722 * max(0.0, Q.girth2_top15 - 0.00559) / 0.003561496   # -2.8%  girth2_top15 > 0.00559
        - 0.01720933 * max(0.0, 6.96 - Q.log_sum_pt) / 0.04364396   # -1.7%  log_sum_pt < 6.96
        + 0.016016 * max(0.0, Q.psi_0p3 - 0.998) * max(0.0, Q.sum_pt_top40 - 797.0) / 0.1613701   # +1.6%  psi_0p3 > 0.998 and sum_pt_top40 > 797
        - 0.01509912 * max(0.0, 0.00322 - Q.e2_sq) / 0.0003964382   # -1.5%  e2_sq < 0.00322
        + 0.01363216 * max(0.0, 0.012 - Q.girth2) * max(0.0, Q.eccentricity - 0.856) / 0.0001576341   # +1.4%  girth2 < 0.012 and eccentricity > 0.856
        - 0.008772177 * max(0.0, Q.sj2_dr - 0.254) * max(0.0, 0.0423 - Q.C2_b2) / 0.0002738067   # -0.9%  sj2_dr > 0.254 and C2_b2 < 0.0423
        - 0.006838702 * max(0.0, Q.zdr_0 - 0.0113) / 0.002565072   # -0.7%  zdr_0 > 0.0113
        - 0.006709438 * max(0.0, 0.0455 - Q.e2) * max(0.0, 1060.0 - Q.sum_pt_top50) / 0.7839629   # -0.7%  e2 < 0.0455 and sum_pt_top50 < 1060
        - 0.006553379 * max(0.0, 22.4 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 12.7) / 33.89434   # -0.7%  n_dr_0p2_0p4 < 22.4 and n_dr_0p1_0p2 > 12.7
        + 0.002217358 * max(0.0, Q.sum_pt_top30 - 1000.0) / 37.3476   # +0.2%  sum_pt_top30 > 1000
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.68;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.68295 * (-0.0634656
        + 0.1689535 * max(0.0, Q.sum_pt_top50 - 951.0) / 93.5193   # +16.9%  sum_pt_top50 > 951
        + 0.1349834 * max(0.0, 1080.0 - Q.sum_pt) / 62.97038   # +13.5%  sum_pt < 1080
        + 0.11116 * max(0.0, Q.girth - 0.0592) / 0.0219099   # +11.1%  girth > 0.0592
        - 0.08078072 * max(0.0, Q.LHA - 0.255) / 0.04448333   # -8.1%  LHA > 0.255
        - 0.05778465 * max(0.0, Q.e2_sq - 0.00783) / 0.003674468   # -5.8%  e2_sq > 0.00783
        - 0.05275792 * max(0.0, 6.91 - Q.log_sum_pt) / 0.01723579   # -5.3%  log_sum_pt < 6.91
        - 0.05273289 * max(0.0, 972.0 - Q.sum_pt) / 9.580663   # -5.3%  sum_pt < 972
        + 0.04856461 * max(0.0, 6.88 - Q.log_sum_pt) / 0.01092239   # +4.9%  log_sum_pt < 6.88
        - 0.0482615 * max(0.0, 0.0134 - Q.girth2_top30) / 0.006939103   # -4.8%  girth2_top30 < 0.0134
        - 0.04560008 * max(0.0, Q.sum_pt - 1070.0) / 28.99664   # -4.6%  sum_pt > 1070
        - 0.03625818 * max(0.0, Q.sum_pt_top30 - 936.0) / 77.78001   # -3.6%  sum_pt_top30 > 936
        + 0.02253531 * max(0.0, 52.1 - Q.n_particles) / 9.826273   # +2.3%  n_particles < 52.1
        - 0.02120089 * max(0.0, 825.0 - Q.sum_pt_top5) / 241.7163   # -2.1%  sum_pt_top5 < 825
        + 0.01747984 * max(0.0, 0.038 - Q.e2) / 0.01131735   # +1.7%  e2 < 0.038
        + 0.01429151 * max(0.0, 3.12 - Q.D2) / 1.024668   # +1.4%  D2 < 3.12
        - 0.01349216 * max(0.0, Q.sum_pt_top50 - 1020.0) / 42.77037   # -1.3%  sum_pt_top50 > 1020
        - 0.01274295 * max(0.0, 57.7 - Q.n_particles) * max(0.0, 2.66 - Q.D2) / 11.8376   # -1.3%  n_particles < 57.7 and D2 < 2.66
        + 0.01144105 * max(0.0, Q.sum_pt_top50 - 1020.0) * max(0.0, 6.89e-08 - Q.e4) / 2.842423e-06   # +1.1%  sum_pt_top50 > 1020 and e4 < 6.89e-08
        + 0.01117793 * max(0.0, 0.448 - Q.max_dr) / 0.1020626   # +1.1%  max_dr < 0.448
        + 0.00948785 * max(0.0, 9.37 - Q.n_dr_0p2_0p4) / 3.412736   # +0.9%  n_dr_0p2_0p4 < 9.37
        + 0.009106578 * max(0.0, 991.0 - Q.sum_pt_top40) / 22.89062   # +0.9%  sum_pt_top40 < 991
        + 0.008356501 * max(0.0, Q.tau2 - 0.0179) / 0.01796219   # +0.8%  tau2 > 0.0179
        - 0.006693924 * max(0.0, Q.girth2 - 0.0231) / 0.0007747657   # -0.7%  girth2 > 0.0231
        + 0.002797642 * max(0.0, 0.0254 - Q.z_8) / 0.002230379   # +0.3%  z_8 < 0.0254
        - 0.001358364 * max(0.0, Q.soft1_pt - 1.48) / 0.1043981   # -0.1%  soft1_pt > 1.48
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 6.826;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.826097 * (-0.0859935
        + 0.1652955 * max(0.0, 0.0571 - Q.e2) / 0.0271885   # +16.5%  e2 < 0.0571
        - 0.1624863 * max(0.0, Q.e2_sq - 0.00988) / 0.003115582   # -16.2%  e2_sq > 0.00988
        - 0.1564164 * max(0.0, 0.0132 - Q.girth2) / 0.006280669   # -15.6%  girth2 < 0.0132
        + 0.08392809 * max(0.0, Q.tau1 - 0.0671) / 0.03147809   # +8.4%  tau1 > 0.0671
        + 0.07526949 * max(0.0, Q.girth2_top30 - 0.008) / 0.003191285   # +7.5%  girth2_top30 > 0.008
        + 0.06835441 * max(0.0, Q.lam1 - 0.00379) / 0.004775782   # +6.8%  lam1 > 0.00379
        - 0.03321583 * max(0.0, 0.0217 - Q.e2_sq) * max(0.0, 7.01 - Q.log_sum_pt) / 0.001016746   # -3.3%  e2_sq < 0.0217 and log_sum_pt < 7.01
        + 0.03095009 * max(0.0, 0.0055 - Q.lam1) / 0.001272701   # +3.1%  lam1 < 0.0055
        + 0.02728641 * max(0.0, 0.00764 - Q.girth2_top40) / 0.002252233   # +2.7%  girth2_top40 < 0.00764
        + 0.02726013 * max(0.0, 0.31 - Q.z_dr_0p1_0p2) / 0.1789234   # +2.7%  z_dr_0p1_0p2 < 0.31
        + 0.02376577 * max(0.0, 0.00445 - Q.girth2) / 0.000711524   # +2.4%  girth2 < 0.00445
        - 0.0209587 * max(0.0, Q.n_pt_above_5 - 17.9) / 10.67658   # -2.1%  n_pt_above_5 > 17.9
        + 0.02018353 * max(0.0, 0.00798 - Q.e2_sq) / 0.002311656   # +2.0%  e2_sq < 0.00798
        - 0.01943121 * max(0.0, 0.0116 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.987) / 4.84085e-05   # -1.9%  girth2 < 0.0116 and psi_0p3 > 0.987
        + 0.01767465 * max(0.0, 0.00935 - Q.e2_sq) * max(0.0, 1020.0 - Q.sum_pt) / 0.05856742   # +1.8%  e2_sq < 0.00935 and sum_pt < 1020
        - 0.01535185 * max(0.0, Q.log_sum_pt - 6.84) * max(0.0, 0.00221 - Q.girth2_top3) / 0.000101741   # -1.5%  log_sum_pt > 6.84 and girth2_top3 < 0.00221
        + 0.0125018 * max(0.0, Q.sum_pt_top30 - 906.0) / 101.2319   # +1.3%  sum_pt_top30 > 906
        - 0.008746678 * max(0.0, 0.0192 - Q.e2) / 0.002477414   # -0.9%  e2 < 0.0192
        + 0.007062442 * max(0.0, Q.psi_0p3 - 0.99) * max(0.0, Q.n_dr_0p1_0p2 - 9.27) / 0.0238658   # +0.7%  psi_0p3 > 0.99 and n_dr_0p1_0p2 > 9.27
        + 0.006668851 * max(0.0, 0.00168 - Q.girth2_top2) * max(0.0, Q.M2 - 0.0304) / 2.571877e-05   # +0.7%  girth2_top2 < 0.00168 and M2 > 0.0304
        + 0.004390695 * max(0.0, Q.e2 - 0.04) * max(0.0, 27.2 - Q.pt_14) / 0.03464891   # +0.4%  e2 > 0.04 and pt_14 < 27.2
        - 0.003606479 * max(0.0, 0.00319 - Q.lam1) * max(0.0, 1020.0 - Q.sum_pt_top50) / 0.01061128   # -0.4%  lam1 < 0.00319 and sum_pt_top50 < 1020
        - 0.003347626 * max(0.0, Q.log_sum_pt - 6.88) * max(0.0, Q.dr_max_012 - 0.0707) / 0.003002789   # -0.3%  log_sum_pt > 6.88 and dr_max_012 > 0.0707
        + 0.003056553 * max(0.0, Q.z_top20_slots - 0.664) * max(0.0, Q.girth2_top3 - 0.00915) / 0.0002182461   # +0.3%  z_top20_slots > 0.664 and girth2_top3 > 0.00915
        + 0.001439033 * max(0.0, 0.000865 - Q.girth2_top40) / 2.168428e-05   # +0.1%  girth2_top40 < 0.000865
        + 0.001351505 * max(0.0, 0.0161 - Q.girth2) * max(0.0, Q.soft1_pt - 1.43) / 0.0004046275   # +0.1%  girth2 < 0.0161 and soft1_pt > 1.43
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 34.47;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.47247 * (0.09340788
        + 0.1684455 * max(0.0, 0.00947 - Q.e2_sq) / 0.00339575   # +16.8%  e2_sq < 0.00947
        - 0.1106846 * max(0.0, Q.girth - 0.0171) / 0.05284726   # -11.1%  girth > 0.0171
        - 0.09748074 * max(0.0, 0.00926 - Q.e2_sq) * max(0.0, 1190.0 - Q.sum_pt) / 0.4706445   # -9.7%  e2_sq < 0.00926 and sum_pt < 1190
        - 0.08170661 * max(0.0, 0.0861 - Q.girth) / 0.027614   # -8.2%  girth < 0.0861
        - 0.07902874 * max(0.0, 0.00807 - Q.girth2_top50) / 0.002432424   # -7.9%  girth2_top50 < 0.00807
        - 0.05894938 * max(0.0, 0.00647 - Q.e2_sq) / 0.001441227   # -5.9%  e2_sq < 0.00647
        + 0.05574473 * max(0.0, 0.00641 - Q.girth2_top50) / 0.001455802   # +5.6%  girth2_top50 < 0.00641
        - 0.04830938 * max(0.0, 0.00707 - Q.e2_sq) / 0.001762268   # -4.8%  e2_sq < 0.00707
        + 0.04633416 * max(0.0, 0.0252 - Q.girth2_top40) / 0.01676026   # +4.6%  girth2_top40 < 0.0252
        + 0.03768525 * max(0.0, 1170.0 - Q.sum_pt_top50) / 146.7914   # +3.8%  sum_pt_top50 < 1170
        - 0.03033821 * max(0.0, 0.00805 - Q.girth2_top20) / 0.003112598   # -3.0%  girth2_top20 < 0.00805
        - 0.02467877 * max(0.0, 0.0082 - Q.lam1) / 0.002913486   # -2.5%  lam1 < 0.0082
        + 0.02341453 * max(0.0, 0.00631 - Q.girth2_top20) / 0.001949654   # +2.3%  girth2_top20 < 0.00631
        - 0.01697379 * max(0.0, 1070.0 - Q.sum_pt) / 55.2008   # -1.7%  sum_pt < 1070
        + 0.01680975 * max(0.0, 0.00639 - Q.lam1) * max(0.0, 1120.0 - Q.sum_pt) / 0.1474487   # +1.7%  lam1 < 0.00639 and sum_pt < 1120
        + 0.01564374 * max(0.0, Q.LHA - 0.32) / 0.01679995   # +1.6%  LHA > 0.32
        + 0.01534769 * max(0.0, 0.0771 - Q.girth) * max(0.0, 1170.0 - Q.sum_pt_top50) / 2.784594   # +1.5%  girth < 0.0771 and sum_pt_top50 < 1170
        + 0.01442082 * max(0.0, 13.9 - Q.n_dr_0p2_0p4) / 6.619455   # +1.4%  n_dr_0p2_0p4 < 13.9
        + 0.01377349 * max(0.0, Q.width - 0.0124) / 0.002552722   # +1.4%  width > 0.0124
        + 0.01264986 * max(0.0, 0.0521 - Q.girth) / 0.009103798   # +1.3%  girth < 0.0521
        + 0.008271348 * max(0.0, Q.psi_0p3 - 0.996) * max(0.0, 2.52 - Q.soft1_pt) / 0.003327115   # +0.8%  psi_0p3 > 0.996 and soft1_pt < 2.52
        - 0.007505933 * max(0.0, Q.girth2_top15 - 0.00838) / 0.002640286   # -0.8%  girth2_top15 > 0.00838
        + 0.003941631 * max(0.0, 0.407 - Q.max_dr) / 0.06349427   # +0.4%  max_dr < 0.407
        - 0.003100127 * max(0.0, Q.LHA - 0.405) / 0.003079798   # -0.3%  LHA > 0.405
        + 0.002135349 * max(0.0, 0.178 - Q.tau21_b2) / 0.02886696   # +0.2%  tau21_b2 < 0.178
        - 0.002129827 * max(0.0, 0.993 - Q.z_top50_slots) / 0.005398557   # -0.2%  z_top50_slots < 0.993
        - 0.001472143 * max(0.0, 0.00782 - Q.girth2) * max(0.0, 0.484 - Q.z_dr_0_0p05) / 7.048388e-05   # -0.1%  girth2 < 0.00782 and z_dr_0_0p05 < 0.484
        - 0.001062079 * max(0.0, 14.8 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 14.9) / 10.34251   # -0.1%  n_dr_0p2_0p4 < 14.8 and n_dr_0p1_0p2 > 14.9
        - 0.0009132471 * max(0.0, 0.189 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.222) / 0.0006655788   # -0.1%  tau21_b2 < 0.189 and sj2_dr > 0.222
        + 0.0007310003 * max(0.0, Q.sd_rg - 0.302) / 0.004300236   # +0.1%  sd_rg > 0.302
        + 0.0003175169 * max(0.0, Q.log_sum_pt - 6.97) / 0.0255142   # +0.0%  log_sum_pt > 6.97
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 12.56;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.56126 * (0.2356451
        - 0.198257 * max(0.0, 0.0225 - Q.girth2) * max(0.0, 7.16 - Q.log_sum_pt) / 0.002971787   # -19.8%  girth2 < 0.0225 and log_sum_pt < 7.16
        + 0.1692017 * max(0.0, 0.0243 - Q.girth2) / 0.01562784   # +16.9%  girth2 < 0.0243
        - 0.1191644 * max(0.0, 0.0128 - Q.girth2) / 0.005963564   # -11.9%  girth2 < 0.0128
        + 0.09691541 * max(0.0, 1190.0 - Q.sum_pt_top50) / 165.1804   # +9.7%  sum_pt_top50 < 1190
        + 0.08463693 * max(0.0, 0.00769 - Q.girth2_top30) * max(0.0, 1270.0 - Q.sum_pt) / 0.5342445   # +8.5%  girth2_top30 < 0.00769 and sum_pt < 1270
        - 0.08378609 * max(0.0, 0.00944 - Q.e2_sq) / 0.003373265   # -8.4%  e2_sq < 0.00944
        - 0.06439594 * max(0.0, Q.tau1 - 0.0234) / 0.06419795   # -6.4%  tau1 > 0.0234
        - 0.04988596 * max(0.0, 0.0285 - Q.girth2) * max(0.0, Q.z_top50_slots - 0.972) / 0.0004676347   # -5.0%  girth2 < 0.0285 and z_top50_slots > 0.972
        - 0.04125844 * max(0.0, 0.00942 - Q.girth2_top30) * max(0.0, 1120.0 - Q.sum_pt_top40) / 0.3574193   # -4.1%  girth2_top30 < 0.00942 and sum_pt_top40 < 1120
        + 0.03652201 * max(0.0, 1040.0 - Q.sum_pt) / 33.98241   # +3.7%  sum_pt < 1040
        - 0.01945562 * max(0.0, 17.0 - Q.n_dr_0p2_0p4) / 9.118921   # -1.9%  n_dr_0p2_0p4 < 17
        + 0.015367 * max(0.0, Q.LHA - 0.334) / 0.01368999   # +1.5%  LHA > 0.334
        + 0.009601019 * max(0.0, 0.00363 - Q.girth2) / 0.000494266   # +1.0%  girth2 < 0.00363
        + 0.007537626 * max(0.0, 0.0188 - Q.girth2) * max(0.0, 983.0 - Q.sum_pt) / 0.09374463   # +0.8%  girth2 < 0.0188 and sum_pt < 983
        - 0.004014972 * max(0.0, 6.85 - Q.log_sum_pt) / 0.007460519   # -0.4%  log_sum_pt < 6.85
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 5.849;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.848513 * (0.1210564
        + 0.1846526 * max(0.0, 0.00703 - Q.width) / 0.001739039   # +18.5%  width < 0.00703
        + 0.1000752 * max(0.0, 0.00914 - Q.girth2_top50) / 0.003215884   # +10.0%  girth2_top50 < 0.00914
        - 0.09947661 * max(0.0, 0.00959 - Q.girth2) / 0.003483774   # -9.9%  girth2 < 0.00959
        - 0.09034921 * max(0.0, 0.0268 - Q.girth2_top15) / 0.01964344   # -9.0%  girth2_top15 < 0.0268
        + 0.0807043 * max(0.0, 0.00534 - Q.girth2_top40) / 0.001080092   # +8.1%  girth2_top40 < 0.00534
        - 0.07057013 * max(0.0, 0.00654 - Q.girth2_top50) / 0.00152299   # -7.1%  girth2_top50 < 0.00654
        + 0.0663847 * max(0.0, 999.0 - Q.sum_pt) / 14.99042   # +6.6%  sum_pt < 999
        - 0.05452021 * max(0.0, 6.91 - Q.log_sum_pt) / 0.01723579   # -5.5%  log_sum_pt < 6.91
        - 0.0540331 * max(0.0, Q.z_top30_slots - 0.881) / 0.07614778   # -5.4%  z_top30_slots > 0.881
        + 0.0353374 * max(0.0, Q.girth - 0.0981) * max(0.0, 3.05 - Q.soft5_pt) / 0.01245007   # +3.5%  girth > 0.0981 and soft5_pt < 3.05
        - 0.02401572 * max(0.0, Q.girth2 - 0.0247) / 0.0005876831   # -2.4%  girth2 > 0.0247
        - 0.02374464 * max(0.0, Q.LHA - 0.291) * max(0.0, 2.17 - Q.soft5_pt) / 0.02143068   # -2.4%  LHA > 0.291 and soft5_pt < 2.17
        - 0.02345535 * max(0.0, Q.girth - 0.12) / 0.004396761   # -2.3%  girth > 0.12
        + 0.02126174 * max(0.0, Q.LHA - 0.353) * max(0.0, 1080.0 - Q.sum_pt) / 0.9143352   # +2.1%  LHA > 0.353 and sum_pt < 1080
        - 0.01788695 * max(0.0, Q.z_dr_0_0p05 - 0.858) / 0.0213494   # -1.8%  z_dr_0_0p05 > 0.858
        - 0.01579229 * max(0.0, Q.log_sum_pt - 6.97) / 0.0255142   # -1.6%  log_sum_pt > 6.97
        - 0.01213142 * max(0.0, 0.00496 - Q.girth2_top40) * max(0.0, 1020.0 - Q.sum_pt) / 0.0192801   # -1.2%  girth2_top40 < 0.00496 and sum_pt < 1020
        + 0.006252473 * max(0.0, 0.00662 - Q.girth2_top15) * max(0.0, Q.n_dr_0p2_0p4 - 7.35) / 0.004312226   # +0.6%  girth2_top15 < 0.00662 and n_dr_0p2_0p4 > 7.35
        + 0.006111836 * max(0.0, 6.87 - Q.log_sum_pt) * max(0.0, 0.557 - Q.z_dr_0p1_0p2) / 0.003539124   # +0.6%  log_sum_pt < 6.87 and z_dr_0p1_0p2 < 0.557
        + 0.005349611 * max(0.0, 2.29 - Q.pt_entropy) / 0.05970853   # +0.5%  pt_entropy < 2.29
        + 0.004880392 * max(0.0, Q.n_dr_0p1_0p2 - 16.6) / 2.26532   # +0.5%  n_dr_0p1_0p2 > 16.6
        + 0.00301412 * max(0.0, 0.0094 - Q.lam1) * max(0.0, 865.0 - Q.sum_pt_top30) / 0.01506677   # +0.3%  lam1 < 0.0094 and sum_pt_top30 < 865
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 8.654;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.653562 * (0.1063146
        - 0.2737474 * max(0.0, 0.00956 - Q.e2_sq) / 0.00346329   # -27.4%  e2_sq < 0.00956
        + 0.2099573 * max(0.0, 0.00806 - Q.e2_sq) / 0.002365727   # +21.0%  e2_sq < 0.00806
        + 0.182549 * max(0.0, Q.e2 - 0.00465) / 0.02624084   # +18.3%  e2 > 0.00465
        - 0.0668888 * max(0.0, 0.991 - Q.psi_0p2) / 0.05122357   # -6.7%  psi_0p2 < 0.991
        - 0.04922121 * max(0.0, 13.2 - Q.n_dr_0p2_0p4) / 6.08484   # -4.9%  n_dr_0p2_0p4 < 13.2
        + 0.04284686 * max(0.0, Q.z_dr_0p2_0p4 - 0.0617) / 0.03142186   # +4.3%  z_dr_0p2_0p4 > 0.0617
        + 0.02088988 * max(0.0, 0.0655 - Q.C2) / 0.01506432   # +2.1%  C2 < 0.0655
        - 0.02060199 * max(0.0, Q.z_top5 - 0.535) / 0.08369981   # -2.1%  z_top5 > 0.535
        - 0.01956516 * max(0.0, 0.0387 - Q.girth) / 0.004936103   # -2.0%  girth < 0.0387
        - 0.01749894 * max(0.0, Q.e2_sq - 0.0252) / 0.0005257923   # -1.7%  e2_sq > 0.0252
        + 0.01566207 * max(0.0, 0.552 - Q.z_top5) / 0.05791996   # +1.6%  z_top5 < 0.552
        - 0.01524924 * max(0.0, Q.e2 - 0.00739) * max(0.0, Q.log_sum_pt - 6.9) / 0.00109058   # -1.5%  e2 > 0.00739 and log_sum_pt > 6.9
        + 0.01455228 * max(0.0, 0.0732 - Q.dr_0) / 0.03180027   # +1.5%  dr_0 < 0.0732
        - 0.01344684 * max(0.0, 0.258 - Q.tau21_b2) / 0.06289894   # -1.3%  tau21_b2 < 0.258
        + 0.01067984 * max(0.0, 14.1 - Q.n_dr_0p2_0p4) * max(0.0, Q.max_dr - 0.22) / 0.7701555   # +1.1%  n_dr_0p2_0p4 < 14.1 and max_dr > 0.22
        + 0.00794439 * max(0.0, 1020.0 - Q.sum_pt) * max(0.0, 0.551 - Q.z_dr_0p05_0p1) / 7.587999   # +0.8%  sum_pt < 1020 and z_dr_0p05_0p1 < 0.551
        - 0.005858414 * max(0.0, 1030.0 - Q.sum_pt) * max(0.0, Q.eccentricity - 0.606) / 4.737958   # -0.6%  sum_pt < 1030 and eccentricity > 0.606
        - 0.005051186 * max(0.0, 968.0 - Q.sum_pt) / 9.012525   # -0.5%  sum_pt < 968
        + 0.004227092 * max(0.0, 0.142 - Q.girth) * max(0.0, Q.pt1_dr01 - 9.54) / 0.1105118   # +0.4%  girth < 0.142 and pt1_dr01 > 9.54
        + 0.003024559 * max(0.0, 0.147 - Q.girth) * max(0.0, Q.ptdr0_2 - 10.1) / 0.06998183   # +0.3%  girth < 0.147 and ptdr0_2 > 10.1
        - 0.0005375553 * max(0.0, 0.0171 - Q.girth) / 0.0007026839   # -0.1%  girth < 0.0171
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 6.516;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.516072 * (-0.1856947
        + 0.2011023 * max(0.0, 0.0125 - Q.girth2_top30) / 0.006210414   # +20.1%  girth2_top30 < 0.0125
        + 0.1402843 * max(0.0, 0.00823 - Q.e2_sq) / 0.002483974   # +14.0%  e2_sq < 0.00823
        - 0.1024756 * max(0.0, 0.0422 - Q.e2) / 0.01439091   # -10.2%  e2 < 0.0422
        - 0.09943763 * max(0.0, 0.00642 - Q.girth2_top30) / 0.001709612   # -9.9%  girth2_top30 < 0.00642
        + 0.06601736 * max(0.0, 1130.0 - Q.sum_pt) / 104.6652   # +6.6%  sum_pt < 1130
        + 0.06263782 * max(0.0, 0.0259 - Q.e2) / 0.004847418   # +6.3%  e2 < 0.0259
        - 0.04948493 * max(0.0, Q.LHA - 0.308) / 0.02015296   # -4.9%  LHA > 0.308
        + 0.04394869 * max(0.0, 7.58 - Q.n_dr_0p2_0p4) * max(0.0, 22.4 - Q.n_dr_0p1_0p2) / 29.95532   # +4.4%  n_dr_0p2_0p4 < 7.58 and n_dr_0p1_0p2 < 22.4
        - 0.03843142 * max(0.0, 0.00914 - Q.e2_sq) * max(0.0, 1140.0 - Q.sum_pt) / 0.3210537   # -3.8%  e2_sq < 0.00914 and sum_pt < 1140
        - 0.03782143 * max(0.0, 0.00493 - Q.e2_sq) / 0.0008527584   # -3.8%  e2_sq < 0.00493
        - 0.02384774 * max(0.0, 8.1 - Q.n_dr_0p2_0p4) * max(0.0, 0.00612 - Q.girth2) / 0.004188506   # -2.4%  n_dr_0p2_0p4 < 8.1 and girth2 < 0.00612
        + 0.02270224 * max(0.0, 63.7 - Q.n_particles) / 17.9526   # +2.3%  n_particles < 63.7
        + 0.01934248 * max(0.0, 13.5 - Q.n_dr_0p2_0p4) * max(0.0, 1110.0 - Q.sum_pt_top50) / 522.975   # +1.9%  n_dr_0p2_0p4 < 13.5 and sum_pt_top50 < 1110
        - 0.01735826 * max(0.0, Q.n_dr_0p1_0p2 - 8.57) / 5.544493   # -1.7%  n_dr_0p1_0p2 > 8.57
        + 0.01588367 * max(0.0, Q.girth2_top15 - 0.00761) / 0.002827846   # +1.6%  girth2_top15 > 0.00761
        + 0.01358545 * max(0.0, 0.00124 - Q.girth2_top10) / 0.0002596006   # +1.4%  girth2_top10 < 0.00124
        + 0.01228743 * max(0.0, Q.girth - 0.132) / 0.002770442   # +1.2%  girth > 0.132
        + 0.008476525 * max(0.0, 0.00918 - Q.e2_sq) * max(0.0, Q.z_dr_0p1_0p2 - 0.0887) / 7.703438e-05   # +0.8%  e2_sq < 0.00918 and z_dr_0p1_0p2 > 0.0887
        + 0.008198944 * max(0.0, 0.00462 - Q.z_dr_0p2_0p4) / 0.0008801468   # +0.8%  z_dr_0p2_0p4 < 0.00462
        + 0.007233831 * max(0.0, Q.z_top30_slots - 0.977) * max(0.0, 1.27 - Q.D2_b2) / 0.00359818   # +0.7%  z_top30_slots > 0.977 and D2_b2 < 1.27
        - 0.006559196 * max(0.0, 916.0 - Q.sum_pt_top30) / 16.37555   # -0.7%  sum_pt_top30 < 916
        - 0.002882793 * max(0.0, Q.log_sum_pt - 6.98) / 0.02313361   # -0.3%  log_sum_pt > 6.98
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 11.06;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.06451 * (0.05730035
        - 0.2468329 * max(0.0, 0.00762 - Q.e2_sq) * max(0.0, Q.log_sum_pt - 6.81) / 0.0003306397   # -24.7%  e2_sq < 0.00762 and log_sum_pt > 6.81
        + 0.1945183 * max(0.0, 0.00645 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.81) / 0.0002299411   # +19.5%  girth2 < 0.00645 and log_sum_pt > 6.81
        + 0.1521131 * max(0.0, 0.00749 - Q.e2_sq) / 0.002003638   # +15.2%  e2_sq < 0.00749
        - 0.1053724 * max(0.0, 0.00634 - Q.e2_sq) / 0.001376498   # -10.5%  e2_sq < 0.00634
        - 0.09100559 * max(0.0, 0.0248 - Q.girth2_top40) / 0.01639954   # -9.1%  girth2_top40 < 0.0248
        + 0.05898966 * max(0.0, 0.0084 - Q.lam1) / 0.003064279   # +5.9%  lam1 < 0.0084
        - 0.02377646 * max(0.0, Q.z_top20_slots - 0.755) / 0.1399334   # -2.4%  z_top20_slots > 0.755
        - 0.02075854 * max(0.0, Q.girth2 - 0.00386) / 0.005996943   # -2.1%  girth2 > 0.00386
        + 0.0201321 * max(0.0, 0.00162 - Q.lam2) / 0.0008280734   # +2.0%  lam2 < 0.00162
        + 0.01517675 * max(0.0, Q.sum_pt_top20 - 840.0) / 109.7537   # +1.5%  sum_pt_top20 > 840
        + 0.01445069 * max(0.0, Q.psi_0p3 - 0.991) * max(0.0, 0.00902 - Q.girth2) / 1.852719e-05   # +1.4%  psi_0p3 > 0.991 and girth2 < 0.00902
        - 0.0119175 * max(0.0, Q.z_dr_0_0p05 - 0.888) / 0.01365023   # -1.2%  z_dr_0_0p05 > 0.888
        + 0.01070574 * max(0.0, Q.LHA - 0.366) / 0.008058073   # +1.1%  LHA > 0.366
        + 0.008937319 * max(0.0, 0.00609 - Q.girth2_top30) * max(0.0, 0.0803 - Q.z_dr_0p05_0p1) / 6.298536e-05   # +0.9%  girth2_top30 < 0.00609 and z_dr_0p05_0p1 < 0.0803
        + 0.008434317 * max(0.0, 0.0063 - Q.girth2) * max(0.0, 8.74 - Q.n_for_50pct) / 0.006811792   # +0.8%  girth2 < 0.0063 and n_for_50pct < 8.74
        - 0.006742405 * max(0.0, 960.0 - Q.sum_pt_top30) / 27.94059   # -0.7%  sum_pt_top30 < 960
        - 0.005411958 * max(0.0, Q.lam2 - 0.000736) / 0.0008897569   # -0.5%  lam2 > 0.000736
        + 0.003376874 * max(0.0, 0.00851 - Q.e2_sq) * max(0.0, Q.zdr_0 - 0.00435) / 5.644024e-06   # +0.3%  e2_sq < 0.00851 and zdr_0 > 0.00435
        + 0.001347411 * max(0.0, 6.81 - Q.log_sum_pt) / 0.004824738   # +0.1%  log_sum_pt < 6.81
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 11.69;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.69381 * (0.05455878
        + 0.2307022 * max(0.0, Q.log_sum_pt - 6.86) / 0.09302715   # +23.1%  log_sum_pt > 6.86
        - 0.1395982 * max(0.0, 1090.0 - Q.sum_pt) / 70.97545   # -14.0%  sum_pt < 1090
        + 0.101053 * max(0.0, 1090.0 - Q.sum_pt_top50) / 76.73338   # +10.1%  sum_pt_top50 < 1090
        + 0.0864394 * max(0.0, Q.girth2_top40 - 0.00976) * max(0.0, 1310.0 - Q.sum_pt_top50) / 0.9626723   # +8.6%  girth2_top40 > 0.00976 and sum_pt_top50 < 1310
        - 0.0788942 * max(0.0, Q.girth2_top40 - 0.0102) / 0.002838689   # -7.9%  girth2_top40 > 0.0102
        - 0.0632914 * max(0.0, Q.sum_pt - 1020.0) / 46.54828   # -6.3%  sum_pt > 1020
        + 0.05047134 * max(0.0, Q.girth2_top50 - 0.00599) / 0.004404495   # +5.0%  girth2_top50 > 0.00599
        - 0.04308238 * max(0.0, Q.e2_sq - 0.009) / 0.003336405   # -4.3%  e2_sq > 0.009
        - 0.03919072 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # -3.9%  log_sum_pt > 6.91
        - 0.02343355 * max(0.0, 957.0 - Q.sum_pt) / 7.6544   # -2.3%  sum_pt < 957
        + 0.02241125 * max(0.0, 1050.0 - Q.sum_pt) * max(0.0, Q.tau21 - 0.118) / 14.64094   # +2.2%  sum_pt < 1050 and tau21 > 0.118
        - 0.01834361 * max(0.0, Q.width - 0.0195) / 0.001269271   # -1.8%  width > 0.0195
        - 0.01820384 * max(0.0, 0.103 - Q.girth) / 0.04101585   # -1.8%  girth < 0.103
        + 0.01798972 * max(0.0, 1010.0 - Q.sum_pt_top50) / 22.47526   # +1.8%  sum_pt_top50 < 1010
        + 0.01089215 * max(0.0, 1030.0 - Q.sum_pt_top50) * max(0.0, Q.tau32 - 0.449) / 8.724019   # +1.1%  sum_pt_top50 < 1030 and tau32 > 0.449
        - 0.01003567 * max(0.0, Q.width - 0.0238) / 0.0006903247   # -1.0%  width > 0.0238
        - 0.008898399 * max(0.0, 25.1 - Q.n_pt_above_5) / 2.661284   # -0.9%  n_pt_above_5 < 25.1
        - 0.008562933 * max(0.0, 938.0 - Q.sum_pt_top30) / 21.48783   # -0.9%  sum_pt_top30 < 938
        - 0.005667059 * max(0.0, Q.girth2_top50 - 0.0104) * max(0.0, Q.soft7_z - 0.0013) / 2.117237e-06   # -0.6%  girth2_top50 > 0.0104 and soft7_z > 0.0013
        - 0.004315701 * max(0.0, Q.sum_pt_top10 - 847.0) * max(0.0, 0.0999 - Q.dr_max_012) / 2.548838   # -0.4%  sum_pt_top10 > 847 and dr_max_012 < 0.0999
        - 0.004026617 * max(0.0, Q.log_sum_pt - 6.8) * max(0.0, Q.C2 - 0.0763) / 0.001418268   # -0.4%  log_sum_pt > 6.8 and C2 > 0.0763
        + 0.00387739 * max(0.0, Q.e2 - 0.0499) / 0.001778097   # +0.4%  e2 > 0.0499
        - 0.003338313 * max(0.0, 0.0936 - Q.girth) * max(0.0, Q.z_dr_0p05_0p1 - 0.092) / 0.002710944   # -0.3%  girth < 0.0936 and z_dr_0p05_0p1 > 0.092
        + 0.002891202 * max(0.0, Q.girth2_top40 - 0.0285) * max(0.0, Q.pt_6 - 9.58) / 0.005542487   # +0.3%  girth2_top40 > 0.0285 and pt_6 > 9.58
        + 0.002875298 * max(0.0, 23.8 - Q.n_pt_above_5) * max(0.0, 5.73 - Q.D2) / 4.055873   # +0.3%  n_pt_above_5 < 23.8 and D2 < 5.73
        + 0.001036495 * max(0.0, Q.sum_pt_top15 - 1090.0) / 5.611379   # +0.1%  sum_pt_top15 > 1090
        - 0.0004779699 * max(0.0, Q.sum_pt - 1050.0) * max(0.0, Q.absphi_2 - 0.0223) / 0.5127788   # -0.0%  sum_pt > 1050 and absphi_2 > 0.0223
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 14.77;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.77103 * (0.01570642
        + 0.1185216 * max(0.0, 0.00962 - Q.e2_sq) / 0.003508389   # +11.9%  e2_sq < 0.00962
        - 0.1086328 * max(0.0, Q.e2_sq - 0.00364) / 0.006147961   # -10.9%  e2_sq > 0.00364
        - 0.08525038 * max(0.0, 0.00826 - Q.girth2) / 0.002503451   # -8.5%  girth2 < 0.00826
        - 0.08221951 * max(0.0, 0.00908 - Q.girth2_top50) / 0.003170931   # -8.2%  girth2_top50 < 0.00908
        - 0.068488 * max(0.0, 0.0115 - Q.girth2_top30) / 0.005409829   # -6.8%  girth2_top30 < 0.0115
        + 0.04986804 * max(0.0, 0.0117 - Q.girth2_top30) * max(0.0, Q.sum_pt - 905.0) / 0.8769073   # +5.0%  girth2_top30 < 0.0117 and sum_pt > 905
        + 0.04939216 * max(0.0, Q.psi_0p3 - 0.995) / 0.002316105   # +4.9%  psi_0p3 > 0.995
        - 0.04825157 * max(0.0, 0.00693 - Q.e2_sq) / 0.00168493   # -4.8%  e2_sq < 0.00693
        + 0.04575387 * max(0.0, 0.0543 - Q.e2) / 0.02466539   # +4.6%  e2 < 0.0543
        + 0.03557651 * max(0.0, Q.tau1 - 0.0459) / 0.04609663   # +3.6%  tau1 > 0.0459
        - 0.03075285 * max(0.0, Q.psi_0p3 - 0.996) * max(0.0, 1240.0 - Q.sum_pt_top40) / 0.3548838   # -3.1%  psi_0p3 > 0.996 and sum_pt_top40 < 1240
        + 0.03009473 * max(0.0, 0.00648 - Q.girth2_top50) / 0.001491712   # +3.0%  girth2_top50 < 0.00648
        + 0.02850995 * max(0.0, 0.0344 - Q.e2) / 0.009036936   # +2.9%  e2 < 0.0344
        + 0.02775856 * max(0.0, Q.lam1 - 0.00626) / 0.003280179   # +2.8%  lam1 > 0.00626
        - 0.02734033 * max(0.0, 0.00802 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.995) / 6.270882e-06   # -2.7%  girth2 < 0.00802 and psi_0p3 > 0.995
        + 0.02475274 * max(0.0, 1090.0 - Q.sum_pt_top30) / 112.1544   # +2.5%  sum_pt_top30 < 1090
        + 0.02379941 * max(0.0, Q.girth2_top20 - 0.00605) / 0.003590825   # +2.4%  girth2_top20 > 0.00605
        + 0.0169646 * max(0.0, Q.z_top30_slots - 0.921) / 0.04365585   # +1.7%  z_top30_slots > 0.921
        - 0.01656538 * max(0.0, 8.18e-05 - Q.e3) / 3.823245e-05   # -1.7%  e3 < 8.18e-05
        - 0.01413876 * max(0.0, Q.LHA - 0.323) / 0.01606492   # -1.4%  LHA > 0.323
        - 0.0114127 * max(0.0, 0.0317 - Q.e2) * max(0.0, Q.sum_pt_top30 - 859.0) / 1.327381   # -1.1%  e2 < 0.0317 and sum_pt_top30 > 859
        - 0.007315752 * max(0.0, 0.914 - Q.z_top30_slots) / 0.01164452   # -0.7%  z_top30_slots < 0.914
        + 0.007083739 * max(0.0, 0.00681 - Q.e2_sq) / 0.001619723   # +0.7%  e2_sq < 0.00681
        - 0.006947108 * max(0.0, 16.7 - Q.n_dr_0p1_0p2) / 6.37366   # -0.7%  n_dr_0p1_0p2 < 16.7
        + 0.006027907 * max(0.0, 0.416 - Q.max_dr) / 0.07180514   # +0.6%  max_dr < 0.416
        - 0.005441428 * max(0.0, Q.log_sum_pt - 7.06) / 0.01113234   # -0.5%  log_sum_pt > 7.06
        + 0.004372605 * max(0.0, Q.lam2 - 0.00314) / 0.0003844516   # +0.4%  lam2 > 0.00314
        + 0.003931158 * max(0.0, Q.psi_0p3 - 0.996) * max(0.0, 1030.0 - Q.sum_pt_top40) / 0.04207771   # +0.4%  psi_0p3 > 0.996 and sum_pt_top40 < 1030
        - 0.003614499 * max(0.0, Q.sj3_dr_max - 0.238) / 0.0503678   # -0.4%  sj3_dr_max > 0.238
        - 0.003531281 * max(0.0, 1.99 - Q.soft10_pt) / 0.3921854   # -0.4%  soft10_pt < 1.99
        - 0.002365909 * max(0.0, 963.0 - Q.sum_pt) / 8.360504   # -0.2%  sum_pt < 963
        + 0.002132296 * max(0.0, 0.0127 - Q.e2_sq) * max(0.0, 0.994 - Q.z_top50_slots) / 1.306897e-05   # +0.2%  e2_sq < 0.0127 and z_top50_slots < 0.994
        + 0.001826116 * max(0.0, 0.139 - Q.tau21_b2) / 0.01615186   # +0.2%  tau21_b2 < 0.139
        - 0.000835172 * max(0.0, 0.247 - Q.max_dr) / 0.005819032   # -0.1%  max_dr < 0.247
        - 0.0005305988 * max(0.0, Q.log_sum_pt - 7.0) / 0.01916257   # -0.1%  log_sum_pt > 7
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 8.596;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.595883 * (0.02524464
        + 0.1285548 * max(0.0, 0.0406 - Q.e2) / 0.01317094   # +12.9%  e2 < 0.0406
        + 0.1114929 * max(0.0, Q.LHA - 0.106) / 0.1576283   # +11.1%  LHA > 0.106
        - 0.1006091 * max(0.0, 0.000317 - Q.e3) / 0.0002382435   # -10.1%  e3 < 0.000317
        - 0.07831089 * max(0.0, Q.girth - 0.0599) / 0.02150643   # -7.8%  girth > 0.0599
        - 0.06199533 * max(0.0, 1070.0 - Q.sum_pt_top40) / 70.95933   # -6.2%  sum_pt_top40 < 1070
        + 0.05086097 * max(0.0, 6.98 - Q.log_sum_pt) / 0.05852677   # +5.1%  log_sum_pt < 6.98
        - 0.04371185 * max(0.0, Q.sum_pt_top50 - 1010.0) / 48.17204   # -4.4%  sum_pt_top50 > 1010
        + 0.03738895 * max(0.0, Q.sd_rg - 0.218) / 0.01727909   # +3.7%  sd_rg > 0.218
        + 0.03354822 * max(0.0, 0.00284 - Q.lam2) / 0.001848568   # +3.4%  lam2 < 0.00284
        + 0.02894279 * max(0.0, 0.00781 - Q.girth2_top5) / 0.004132704   # +2.9%  girth2_top5 < 0.00781
        + 0.02874831 * max(0.0, Q.girth2_top30 - 0.00634) / 0.003813535   # +2.9%  girth2_top30 > 0.00634
        - 0.02690792 * max(0.0, Q.sd_rg - 0.148) / 0.04122947   # -2.7%  sd_rg > 0.148
        - 0.02633725 * max(0.0, 0.00813 - Q.girth2_top50) / 0.002474229   # -2.6%  girth2_top50 < 0.00813
        - 0.02530927 * max(0.0, Q.e2 - 0.04) / 0.003613879   # -2.5%  e2 > 0.04
        - 0.02515462 * max(0.0, Q.psi_0p3 - 0.988) / 0.007304939   # -2.5%  psi_0p3 > 0.988
        - 0.02269589 * max(0.0, 0.0311 - Q.e2) / 0.0072256   # -2.3%  e2 < 0.0311
        - 0.02174127 * max(0.0, 5.59e-05 - Q.e3) * max(0.0, 21.3 - Q.n_dr_0p1_0p2) / 0.0002756422   # -2.2%  e3 < 5.59e-05 and n_dr_0p1_0p2 < 21.3
        - 0.01826906 * max(0.0, 0.54 - Q.z_dr_0p05_0p1) / 0.3172499   # -1.8%  z_dr_0p05_0p1 < 0.54
        + 0.01749842 * max(0.0, Q.sum_pt_top20 - 847.0) / 104.4544   # +1.7%  sum_pt_top20 > 847
        + 0.01596599 * max(0.0, Q.sum_pt_top50 - 1110.0) / 19.2485   # +1.6%  sum_pt_top50 > 1110
        + 0.01415057 * max(0.0, 0.129 - Q.z_dr_0p1_0p2) * max(0.0, 0.0567 - Q.C2_b2) / 0.002111747   # +1.4%  z_dr_0p1_0p2 < 0.129 and C2_b2 < 0.0567
        + 0.01378034 * max(0.0, 12.8 - Q.n_dr_0p1_0p2) / 3.701692   # +1.4%  n_dr_0p1_0p2 < 12.8
        - 0.01281004 * max(0.0, 0.00211 - Q.girth2_top5) / 0.0006925385   # -1.3%  girth2_top5 < 0.00211
        + 0.01276203 * max(0.0, 45.2 - Q.n_particles) / 5.994586   # +1.3%  n_particles < 45.2
        - 0.009771797 * max(0.0, Q.sd_rg - 0.282) / 0.006562283   # -1.0%  sd_rg > 0.282
        - 0.008187866 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # -0.8%  sum_pt < 1000
        - 0.007399811 * max(0.0, 0.105 - Q.z_dr_0p1_0p2) * max(0.0, Q.psi_0p3 - 0.979) / 0.0005679278   # -0.7%  z_dr_0p1_0p2 < 0.105 and psi_0p3 > 0.979
        + 0.00518758 * max(0.0, Q.e2 - 0.0551) / 0.001167325   # +0.5%  e2 > 0.0551
        - 0.003789475 * max(0.0, Q.psi_0p2 - 0.997) / 0.0004395935   # -0.4%  psi_0p2 > 0.997
        + 0.003619566 * max(0.0, 0.143 - Q.sd_rg) * max(0.0, 0.575 - Q.planar_flow) / 0.003327633   # +0.4%  sd_rg < 0.143 and planar_flow < 0.575
        - 0.002658665 * max(0.0, 0.00776 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 10.2) / 0.005771105   # -0.3%  girth2_top5 < 0.00776 and n_dr_0p2_0p4 > 10.2
        - 0.001838483 * max(0.0, 0.000989 - Q.girth2_top3) * max(0.0, Q.eccentricity - 0.788) / 5.664294e-06   # -0.2%  girth2_top3 < 0.000989 and eccentricity > 0.788
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6621992647058823, 2.775377836134454, 0.21233697478991598, 0.3846813025210084, 0.772153781512605, 1.0406120798319327, 0.5437964285714286, 0.4931079831932773, 1.166048424369748, 1.0007091386554623, 1.2577400210084033, 0.7651710084033614, 0.5518455882352942, 1.5290661239495797, 0.2740728991596639, 0.44231943277310926]
T = [3.8103368106617643, 2.4680545857405467, 4.2516491875656515, 4.361977954306723, 4.096606505219275]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +46%, n4 -18%, n9 +14%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.4552383 * h[1] / H_AVG[1]
            - 0.1773162 * h[4] / H_AVG[4]
            + 0.1354186 * h[9] / H_AVG[9]
            - 0.0853445 * h[5] / H_AVG[5]
            - 0.07571797 * h[3] / H_AVG[3]
            + 0.03394419 * h[12] / H_AVG[12]
            + 0.02229939 * h[6] / H_AVG[6]
            + 0.0095632 * h[8] / H_AVG[8]
            - 0.005157599 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +23%, n1 -21%, n11 -8%, n12 +8%, n6 +5% ...
            - 0.332413 * h[4] / H_AVG[4]
            + 0.2280739 * h[9] / H_AVG[9]
            - 0.2108476 * h[1] / H_AVG[1]
            - 0.0775075 * h[11] / H_AVG[11]
            + 0.07686091 * h[12] / H_AVG[12]
            + 0.04819807 * h[6] / H_AVG[6]
            + 0.01075427 * h[2] / H_AVG[2]
            - 0.007962623 * h[10] / H_AVG[10]
            + 0.007382133 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +14%, n0 +12%, n11 +11%, n14 -9%, n7 -7% ...
            - 0.2399757 * h[8] / H_AVG[8]
            + 0.1414989 * h[5] / H_AVG[5]
            + 0.1168134 * h[0] / H_AVG[0]
            + 0.1068574 * h[11] / H_AVG[11]
            - 0.08863625 * h[14] / H_AVG[14]
            - 0.07248775 * h[7] / H_AVG[7]
            + 0.06242939 * h[4] / H_AVG[4]
            - 0.05272948 * h[12] / H_AVG[12]
            - 0.05148711 * h[9] / H_AVG[9]
            + 0.03958419 * h[3] / H_AVG[3]
            - 0.01950652 * h[15] / H_AVG[15]
            + 0.007993904 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -25%, n0 -21%, n5 +16%, n6 -12%, n7 +10%, n15 +6% ...
            - 0.2506135 * h[8] / H_AVG[8]
            - 0.2087411 * h[0] / H_AVG[0]
            + 0.1640129 * h[5] / H_AVG[5]
            - 0.1207715 * h[6] / H_AVG[6]
            + 0.1024487 * h[7] / H_AVG[7]
            + 0.05703942 * h[15] / H_AVG[15]
            + 0.03953522 * h[12] / H_AVG[12]
            + 0.03858297 * h[3] / H_AVG[3]
            - 0.01825465 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +30%, n5 -12%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3382595 * h[13] / H_AVG[13]
            + 0.3022228 * h[10] / H_AVG[10]
            - 0.119071 * h[5] / H_AVG[5]
            + 0.06004075 * h[8] / H_AVG[8]
            - 0.05051549 * h[12] / H_AVG[12]
            - 0.04048956 * h[15] / H_AVG[15]
            + 0.03534116 * h[4] / H_AVG[4]
            - 0.03385403 * h[7] / H_AVG[7]
            + 0.02020573 * h[0] / H_AVG[0]
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
