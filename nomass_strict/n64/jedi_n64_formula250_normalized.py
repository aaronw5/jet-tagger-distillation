"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.2%   (on for 43% of jets)
  neuron  5:  11.7%   (on for 81% of jets)
  neuron  1:  11.3%   (on for 95% of jets)
  neuron  4:  10.4%   (on for 69% of jets)
  neuron  0:   7.6%   (on for 59% of jets)
  neuron  9:   6.7%   (on for 63% of jets)
  neuron 10:   6.7%   (on for 84% of jets)
  neuron 13:   6.7%   (on for 86% of jets)
  neuron 12:   5.6%   (on for 42% of jets)
  neuron  7:   5.1%   (on for 55% of jets)
  neuron  6:   4.3%   (on for 53% of jets)
  neuron  3:   3.3%   (on for 88% of jets)
  neuron 11:   2.6%   (on for 59% of jets)
  neuron 14:   2.2%   (on for 38% of jets)
  neuron 15:   1.7%   (on for 73% of jets)
  neuron  2:   0.8%   (on for 100% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.7% (the network: 81.1%); same class as the network for 90.7% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_0                   pT of particle 0 [GeV]
  Q.ptdr0_10               pT10 · ΔR(0, 10) [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_0                    pT of particle 0 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.absphi_1               |Δφ| of particle 1
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_0=pt[0],
        ptdr0_10=pt[10] * math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        pt_11=pt[11],
        soft1_pt=softp(1, 'pt'),
        soft6_pt=softp(6, 'pt'),
        soft9_pt=softp(9, 'pt'),
        z_0=z[0],
        soft1_z=softp(1, 'z'),
        soft5_z=softp(5, 'z'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        sd_rg=softdrop("rg"),
        sd_zg=softdrop("zg"),
        absphi_1=abs(phi[1]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        eta_1=eta[1],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    # scale S = 13.62;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.62436 * (-0.07926977
        + 0.2017497 * max(0.0, 0.0819 - Q.sum_z_dr) / 0.02454206   # +20.2%  sum_z_dr < 0.0819
        + 0.1766188 * max(0.0, 0.00782 - Q.sum_z_dr2_top15) / 0.003212708   # +17.7%  sum_z_dr2_top15 < 0.00782
        - 0.1491507 * max(0.0, 0.0101 - Q.sum_z_dr2_top15) / 0.00500513   # -14.9%  sum_z_dr2_top15 < 0.0101
        + 0.08166258 * max(0.0, 17.8 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_pt_top50 - 882.0) / 1612.465   # +8.2%  n_dr_0p2_0p4 < 17.8 and sum_pt_top50 > 882
        - 0.07875618 * max(0.0, 0.31 - Q.LHA) / 0.06748444   # -7.9%  LHA < 0.31
        - 0.0394738 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.log_sum_pt - 6.92) / 0.4519372   # -3.9%  n_dr_0p2_0p4 < 18 and log_sum_pt > 6.92
        - 0.03796892 * max(0.0, 0.00542 - Q.sum_z_dr2_top15) / 0.001685024   # -3.8%  sum_z_dr2_top15 < 0.00542
        - 0.02328948 * max(0.0, 0.0446 - Q.tau1) / 0.00520171   # -2.3%  tau1 < 0.0446
        - 0.02059875 * max(0.0, Q.sj2_dr - 0.152) / 0.05933294   # -2.1%  sj2_dr > 0.152
        - 0.02014432 * max(0.0, 0.0733 - Q.dr_0) / 0.03187613   # -2.0%  dr_0 < 0.0733
        - 0.01914631 * max(0.0, Q.sum_pt_top40 - 997.0) / 49.97247   # -1.9%  sum_pt_top40 > 997
        - 0.01820025 * max(0.0, 2.86e-05 - Q.e3) / 6.390896e-06   # -1.8%  e3 < 2.86e-05
        - 0.01811508 * max(0.0, 0.161 - Q.sj3_dr_max) / 0.009872257   # -1.8%  sj3_dr_max < 0.161
        + 0.01767947 * max(0.0, 0.211 - Q.sj3_dr_max) / 0.02609659   # +1.8%  sj3_dr_max < 0.211
        - 0.01560152 * max(0.0, 0.0512 - Q.sum_z_dr) / 0.0087835   # -1.6%  sum_z_dr < 0.0512
        + 0.015165 * max(0.0, Q.psi_0p3 - 0.989) / 0.0065384   # +1.5%  psi_0p3 > 0.989
        + 0.01288933 * max(0.0, Q.psi_0p3 - 0.994) * max(0.0, 24.0 - Q.n_dr_0p1_0p2) / 0.03955154   # +1.3%  psi_0p3 > 0.994 and n_dr_0p1_0p2 < 24
        + 0.01202334 * max(0.0, 5.06e-05 - Q.e3) / 1.663049e-05   # +1.2%  e3 < 5.06e-05
        + 0.01003045 * max(0.0, 0.00081 - Q.lam2) / 0.0002563949   # +1.0%  lam2 < 0.00081
        - 0.008582265 * max(0.0, 17.5 - Q.n_dr_0p2_0p4) * max(0.0, Q.tau21_b2 - 0.229) / 1.40877   # -0.9%  n_dr_0p2_0p4 < 17.5 and tau21_b2 > 0.229
        - 0.008402322 * max(0.0, 19.3 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_z_dr2_top10 - 0.00496) / 0.01524318   # -0.8%  n_dr_0p2_0p4 < 19.3 and sum_z_dr2_top10 > 0.00496
        + 0.004695951 * max(0.0, 0.119 - Q.sj3_dr_max) / 0.00444301   # +0.5%  sj3_dr_max < 0.119
        + 0.004036236 * max(0.0, Q.sum_pt_top40 - 1030.0) * max(0.0, 0.166 - Q.sj2_dr) / 1.264164   # +0.4%  sum_pt_top40 > 1030 and sj2_dr < 0.166
        + 0.003345386 * max(0.0, Q.sum_pt_top2 - 403.0) / 52.1496   # +0.3%  sum_pt_top2 > 403
        + 0.002673904 * max(0.0, 0.0866 - Q.sum_z_dr) * max(0.0, 0.494 - Q.z_dr_0_0p05) / 0.0006809389   # +0.3%  sum_z_dr < 0.0866 and z_dr_0_0p05 < 0.494
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 14.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.28824 * (0.1553725
        + 0.1386026 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # +13.9%  log_sum_pt > 6.91
        - 0.1367492 * max(0.0, Q.sum_pt_top50 - 950.0) / 94.39159   # -13.7%  sum_pt_top50 > 950
        + 0.1361188 * max(0.0, Q.sum_pt - 972.0) / 81.3765   # +13.6%  sum_pt > 972
        - 0.1308452 * max(0.0, 7.14 - Q.log_sum_pt) / 0.2008108   # -13.1%  log_sum_pt < 7.14
        + 0.09398175 * max(0.0, 1090.0 - Q.sum_pt_top50) / 76.73338   # +9.4%  sum_pt_top50 < 1090
        - 0.05711741 * max(0.0, Q.log_sum_pt - 6.99) / 0.0210337   # -5.7%  log_sum_pt > 6.99
        + 0.03214219 * max(0.0, 0.0757 - Q.tau1) / 0.01556798   # +3.2%  tau1 < 0.0757
        + 0.03205177 * max(0.0, Q.n_particles - 37.3) * max(0.0, Q.tau32 - 0.337) / 3.784822   # +3.2%  n_particles > 37.3 and tau32 > 0.337
        - 0.03083355 * max(0.0, Q.n_particles - 38.6) * max(0.0, 0.00227 - Q.soft1_z) / 0.01468524   # -3.1%  n_particles > 38.6 and soft1_z < 0.00227
        - 0.02958023 * max(0.0, 0.00172 - Q.lam2) / 0.0009069734   # -3.0%  lam2 < 0.00172
        + 0.02543003 * max(0.0, Q.n_particles - 39.4) / 9.900559   # +2.5%  n_particles > 39.4
        - 0.01725448 * max(0.0, 6.87 - Q.n_dr_0p2_0p4) / 1.97229   # -1.7%  n_dr_0p2_0p4 < 6.87
        - 0.01643028 * max(0.0, 29.5 - Q.pt_11) / 8.994632   # -1.6%  pt_11 < 29.5
        + 0.01619326 * max(0.0, 345.0 - Q.pt_0) / 129.2588   # +1.6%  pt_0 < 345
        - 0.01508848 * max(0.0, Q.n_particles - 36.6) * max(0.0, 0.966 - Q.tau43) / 1.573634   # -1.5%  n_particles > 36.6 and tau43 < 0.966
        - 0.01295233 * max(0.0, 0.0347 - Q.M3) * max(0.0, Q.psi_0p3 - 0.941) / 0.0004974893   # -1.3%  M3 < 0.0347 and psi_0p3 > 0.941
        + 0.01200537 * max(0.0, Q.n_dr_0_0p05 - 14.8) / 2.789198   # +1.2%  n_dr_0_0p05 > 14.8
        - 0.01188911 * max(0.0, Q.z_top5 - 0.595) * max(0.0, 25.1 - Q.pt1_dr01) / 1.179684   # -1.2%  z_top5 > 0.595 and pt1_dr01 < 25.1
        - 0.01143039 * max(0.0, Q.z_dr_0_0p05 - 0.783) * max(0.0, 8.96 - Q.n_dr_0p05_0p1) / 0.2115546   # -1.1%  z_dr_0_0p05 > 0.783 and n_dr_0p05_0p1 < 8.96
        + 0.0107765 * max(0.0, Q.n_particles - 38.8) * max(0.0, 0.013 - Q.zdr_0) / 0.05219568   # +1.1%  n_particles > 38.8 and zdr_0 < 0.013
        + 0.01019388 * max(0.0, 2.43 - Q.soft9_pt) / 0.690297   # +1.0%  soft9_pt < 2.43
        + 0.007357747 * max(0.0, 0.202 - Q.tau21) / 0.009303477   # +0.7%  tau21 < 0.202
        + 0.007157101 * max(0.0, 0.000777 - Q.sum_z_dr2_top15) / 8.815724e-05   # +0.7%  sum_z_dr2_top15 < 0.000777
        - 0.006530153 * max(0.0, 6.86 - Q.n_dr_0p1_0p2) / 0.9330442   # -0.7%  n_dr_0p1_0p2 < 6.86
        - 0.001288208 * max(0.0, Q.n_dr_0_0p05 - 30.2) / 0.2040602   # -0.1%  n_dr_0_0p05 > 30.2
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 0.2092;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.2092245 * (0.6404603
        + 0.9161043 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, 0.192 - Q.dr_max_012) / 0.006821049   # +91.6%  log_sum_pt > 6.91 and dr_max_012 < 0.192
        - 0.05072551 * max(0.0, Q.sum_pt_top20 - 1140.0) * max(0.0, Q.dr_max_012 - 0.122) / 0.05414806   # -5.1%  sum_pt_top20 > 1140 and dr_max_012 > 0.122
        - 0.03317024 * max(0.0, Q.sum_pt_top20 - 1130.0) * max(0.0, Q.eta_1 - 0.0876) / 0.00345275   # -3.3%  sum_pt_top20 > 1130 and eta_1 > 0.0876
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 0.7621;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.7621078 * (0.02742394
        + 0.617722 * max(0.0, 0.0171 - Q.tau4) / 0.003436283   # +61.8%  tau4 < 0.0171
        - 0.2610078 * max(0.0, 0.338 - Q.tau21) / 0.04875393   # -26.1%  tau21 < 0.338
        + 0.1212702 * max(0.0, 4.01 - Q.n_dr_0p2_0p4) / 0.7334996   # +12.1%  n_dr_0p2_0p4 < 4.01
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 6.553;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.552637 * (0.1420802
        + 0.1569327 * max(0.0, 0.0104 - Q.sum_z_dr2_top15) / 0.005246545   # +15.7%  sum_z_dr2_top15 < 0.0104
        - 0.1211101 * max(0.0, 0.00603 - Q.sum_z_dr2_top15) / 0.00201419   # -12.1%  sum_z_dr2_top15 < 0.00603
        + 0.1131689 * max(0.0, Q.sj2_dr - 0.117) / 0.0862273   # +11.3%  sj2_dr > 0.117
        - 0.08620501 * max(0.0, 1040.0 - Q.sum_pt_top40) / 48.2795   # -8.6%  sum_pt_top40 < 1040
        - 0.06988096 * max(0.0, 0.0471 - Q.e2) / 0.01838974   # -7.0%  e2 < 0.0471
        - 0.06564205 * max(0.0, Q.sj2_dr - 0.182) / 0.04135851   # -6.6%  sj2_dr > 0.182
        + 0.05781908 * max(0.0, 1010.0 - Q.sum_pt_top30) / 50.11474   # +5.8%  sum_pt_top30 < 1010
        - 0.05458684 * max(0.0, 0.0562 - Q.sum_z_dr) / 0.01064547   # -5.5%  sum_z_dr < 0.0562
        - 0.04989256 * max(0.0, 0.0557 - Q.z_dr_0p2_0p4) / 0.03113598   # -5.0%  z_dr_0p2_0p4 < 0.0557
        + 0.03755661 * max(0.0, 0.0969 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.997) / 4.27991e-05   # +3.8%  sum_z_dr < 0.0969 and psi_0p3 > 0.997
        + 0.03635777 * max(0.0, Q.n_particles - 29.3) * max(0.0, Q.tau21 - 0.11) / 6.527102   # +3.6%  n_particles > 29.3 and tau21 > 0.11
        - 0.03543753 * max(0.0, Q.n_particles - 46.3) / 6.062905   # -3.5%  n_particles > 46.3
        + 0.03527329 * max(0.0, 46.3 - Q.n_real_top50) / 6.547678   # +3.5%  n_real_top50 < 46.3
        - 0.02086083 * max(0.0, 0.0587 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.997) / 1.178392e-05   # -2.1%  sum_z_dr < 0.0587 and psi_0p3 > 0.997
        + 0.02012893 * max(0.0, 0.00214 - Q.sum_z_dr2_top15) / 0.0004338736   # +2.0%  sum_z_dr2_top15 < 0.00214
        - 0.01243503 * max(0.0, Q.n_particles - 28.4) * max(0.0, Q.soft1_pt - 0.363) / 9.82898   # -1.2%  n_particles > 28.4 and soft1_pt > 0.363
        - 0.00970926 * max(0.0, 3.33 - Q.pt_entropy) * max(0.0, Q.zdr_0 - -0.00141) / 0.005731644   # -1.0%  pt_entropy < 3.33 and zdr_0 > -0.00141
        + 0.006513315 * max(0.0, Q.n_particles - 28.8) * max(0.0, Q.sj3_dr_max - 0.367) / 0.2807855   # +0.7%  n_particles > 28.8 and sj3_dr_max > 0.367
        + 0.004255612 * max(0.0, 0.00851 - Q.sum_z_dr2_top15) * max(0.0, Q.sum_pt_top40 - 1100.0) / 0.1028984   # +0.4%  sum_z_dr2_top15 < 0.00851 and sum_pt_top40 > 1100
        + 0.003985685 * max(0.0, Q.n_particles - 27.3) * max(0.0, 0.286 - Q.max_dr) / 0.1231922   # +0.4%  n_particles > 27.3 and max_dr < 0.286
        + 0.002247888 * max(0.0, Q.n_dr_0p2_0p4 - 24.8) / 0.3362921   # +0.2%  n_dr_0p2_0p4 > 24.8
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 3.955;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.954547 * (0.3262067
        - 0.1895592 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # -19.0%  sum_pt < 1000
        + 0.1600653 * max(0.0, 7.12 - Q.log_sum_pt) / 0.1818924   # +16.0%  log_sum_pt < 7.12
        - 0.09912009 * max(0.0, Q.z_top20_slots - 0.84) / 0.07037255   # -9.9%  z_top20_slots > 0.84
        - 0.07695327 * max(0.0, 0.107 - Q.tau1) / 0.03407787   # -7.7%  tau1 < 0.107
        + 0.06936382 * max(0.0, 11.4 - Q.n_dr_0p2_0p4) / 4.770478   # +6.9%  n_dr_0p2_0p4 < 11.4
        + 0.06746235 * max(0.0, 41.2 - Q.n_particles) / 4.188117   # +6.7%  n_particles < 41.2
        - 0.065262 * max(0.0, Q.n_particles - 43.7) / 7.394891   # -6.5%  n_particles > 43.7
        + 0.06377626 * max(0.0, Q.z_dr_0_0p05 - 0.734) / 0.06602258   # +6.4%  z_dr_0_0p05 > 0.734
        + 0.0512421 * max(0.0, 903.0 - Q.sum_pt_top30) / 13.8794   # +5.1%  sum_pt_top30 < 903
        - 0.04793545 * max(0.0, 0.985 - Q.z_top50_slots) / 0.003536623   # -4.8%  z_top50_slots < 0.985
        - 0.03892588 * max(0.0, 53.8 - Q.n_particles) * max(0.0, 0.0943 - Q.C2) / 0.4721909   # -3.9%  n_particles < 53.8 and C2 < 0.0943
        - 0.02718985 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, Q.absphi_1 - 0.00021) / 0.7680253   # -2.7%  sum_pt < 1010 and absphi_1 > 0.00021
        - 0.02308173 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, Q.ptdr0_10 - 3.85) / 9.449046   # -2.3%  sum_pt < 1000 and ptdr0_10 > 3.85
        + 0.01059233 * max(0.0, 0.149 - Q.sum_z_dr) * max(0.0, Q.n_dr_0p1_0p2 - 12.2) / 0.1183273   # +1.1%  sum_z_dr < 0.149 and n_dr_0p1_0p2 > 12.2
        - 0.009470363 * max(0.0, 0.185 - Q.sum_z_dr) * max(0.0, 0.961 - Q.psi_0p3) / 0.0002355409   # -0.9%  sum_z_dr < 0.185 and psi_0p3 < 0.961
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 7.327;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.327425 * (0.2156283
        - 0.3057263 * max(0.0, 0.0952 - Q.sum_z_dr) / 0.03473157   # -30.6%  sum_z_dr < 0.0952
        + 0.1535887 * max(0.0, 0.00637 - Q.lam2) / 0.005092352   # +15.4%  lam2 < 0.00637
        - 0.1088247 * max(0.0, Q.psi_0p2 - 0.906) / 0.06040949   # -10.9%  psi_0p2 > 0.906
        + 0.09458057 * max(0.0, 0.0532 - Q.tau1) / 0.007516616   # +9.5%  tau1 < 0.0532
        - 0.0630701 * max(0.0, 0.000368 - Q.e3) * max(0.0, 7.02 - Q.log_sum_pt) / 2.445193e-05   # -6.3%  e3 < 0.000368 and log_sum_pt < 7.02
        - 0.05616001 * max(0.0, Q.e3 - 0.000115) / 5.18926e-05   # -5.6%  e3 > 0.000115
        + 0.04493438 * max(0.0, 0.0336 - Q.e2) / 0.008574305   # +4.5%  e2 < 0.0336
        + 0.03980316 * max(0.0, 0.394 - Q.sj2_zsoft) / 0.1602498   # +4.0%  sj2_zsoft < 0.394
        - 0.03261723 * max(0.0, Q.psi_0p3 - 0.994) / 0.002957925   # -3.3%  psi_0p3 > 0.994
        + 0.02588457 * max(0.0, Q.psi_0p2 - 0.896) * max(0.0, 0.135 - Q.sj2_dr) / 0.001364513   # +2.6%  psi_0p2 > 0.896 and sj2_dr < 0.135
        - 0.0197466 * max(0.0, 0.101 - Q.sum_z_dr) * max(0.0, Q.eccentricity - 0.631) / 0.005378876   # -2.0%  sum_z_dr < 0.101 and eccentricity > 0.631
        + 0.01630841 * max(0.0, Q.e2 - 0.0573) / 0.0009559891   # +1.6%  e2 > 0.0573
        + 0.01073492 * max(0.0, Q.n_dr_0p1_0p2 - 12.2) * max(0.0, Q.psi_0p3 - 0.992) / 0.01177535   # +1.1%  n_dr_0p1_0p2 > 12.2 and psi_0p3 > 0.992
        + 0.01029135 * max(0.0, 0.0711 - Q.sum_z_dr) * max(0.0, 993.0 - Q.sum_pt) / 0.2112301   # +1.0%  sum_z_dr < 0.0711 and sum_pt < 993
        + 0.009343655 * max(0.0, Q.sum_pt_top50 - 1050.0) / 31.40593   # +0.9%  sum_pt_top50 > 1050
        + 0.005994006 * max(0.0, 0.0445 - Q.sj2_zsoft) / 0.00214247   # +0.6%  sj2_zsoft < 0.0445
        + 0.002391312 * max(0.0, 0.0962 - Q.sum_z_dr) * max(0.0, 0.963 - Q.psi_0p3) / 1.448113e-05   # +0.2%  sum_z_dr < 0.0962 and psi_0p3 < 0.963
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 26.7;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.7019 * (0.01464316
        - 0.1408766 * max(0.0, 0.108 - Q.tau1) / 0.03483031   # -14.1%  tau1 < 0.108
        - 0.1235558 * max(0.0, 0.081 - Q.sum_z_dr) / 0.02390706   # -12.4%  sum_z_dr < 0.081
        + 0.1215284 * max(0.0, 0.0983 - Q.sum_z_dr) / 0.03721374   # +12.2%  sum_z_dr < 0.0983
        + 0.1039523 * max(0.0, 0.107 - Q.tau1) * max(0.0, 0.286 - Q.z_dr_0p1_0p2) / 0.00809249   # +10.4%  tau1 < 0.107 and z_dr_0p1_0p2 < 0.286
        + 0.09070332 * max(0.0, 0.304 - Q.LHA) / 0.06340185   # +9.1%  LHA < 0.304
        - 0.09034214 * max(0.0, 0.331 - Q.LHA) * max(0.0, 0.256 - Q.z_dr_0p1_0p2) / 0.01723076   # -9.0%  LHA < 0.331 and z_dr_0p1_0p2 < 0.256
        + 0.05808834 * max(0.0, Q.psi_0p3 - 0.997) / 0.001157514   # +5.8%  psi_0p3 > 0.997
        + 0.05094006 * max(0.0, 0.237 - Q.tau21_b2) * max(0.0, 0.0101 - Q.sum_z_dr2_top15) / 0.0001796825   # +5.1%  tau21_b2 < 0.237 and sum_z_dr2_top15 < 0.0101
        + 0.03432305 * max(0.0, 0.0366 - Q.e2) / 0.01039105   # +3.4%  e2 < 0.0366
        - 0.03087869 * max(0.0, 20.2 - Q.n_dr_0p2_0p4) * max(0.0, 8.01e-05 - Q.e3) / 0.0005218479   # -3.1%  n_dr_0p2_0p4 < 20.2 and e3 < 8.01e-05
        - 0.03050304 * max(0.0, 0.235 - Q.tau21_b2) * max(0.0, 0.00802 - Q.sum_z_dr2_top15) / 8.600727e-05   # -3.1%  tau21_b2 < 0.235 and sum_z_dr2_top15 < 0.00802
        - 0.01998101 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 0.00685 - Q.mean_eta2) / 4.680094e-06   # -2.0%  psi_0p3 > 0.997 and mean_eta2 < 0.00685
        - 0.01910774 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 0.00686 - Q.mean_phi2) / 4.680853e-06   # -1.9%  psi_0p3 > 0.997 and mean_phi2 < 0.00686
        - 0.01736857 * max(0.0, 0.239 - Q.tau21_b2) * max(0.0, 1250.0 - Q.sum_pt_top50) / 11.6234   # -1.7%  tau21_b2 < 0.239 and sum_pt_top50 < 1250
        - 0.0167836 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 7.07 - Q.log_sum_pt) / 0.000144101   # -1.7%  psi_0p3 > 0.997 and log_sum_pt < 7.07
        + 0.01559938 * max(0.0, 15.9 - Q.n_dr_0p2_0p4) * max(0.0, 0.113 - Q.M2) / 0.3929556   # +1.6%  n_dr_0p2_0p4 < 15.9 and M2 < 0.113
        - 0.007249137 * max(0.0, Q.psi_0p1 - 0.854) / 0.04419308   # -0.7%  psi_0p1 > 0.854
        - 0.006935335 * max(0.0, 0.347 - Q.tau21) / 0.0523126   # -0.7%  tau21 < 0.347
        - 0.005698181 * max(0.0, 3.44e-05 - Q.e3) / 8.547879e-06   # -0.6%  e3 < 3.44e-05
        + 0.005158675 * max(0.0, 0.223 - Q.tau21_b2) * max(0.0, 0.0712 - Q.M2) / 0.001299494   # +0.5%  tau21_b2 < 0.223 and M2 < 0.0712
        + 0.00504007 * max(0.0, 0.236 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.187) / 0.001879601   # +0.5%  tau21_b2 < 0.236 and sj2_dr > 0.187
        - 0.004218511 * max(0.0, Q.psi_0p3 - 0.998) * max(0.0, Q.sum_z_dr2_top10 - 0.00775) / 4.713065e-07   # -0.4%  psi_0p3 > 0.998 and sum_z_dr2_top10 > 0.00775
        - 0.001167978 * max(0.0, 0.257 - Q.tau21_b2) * max(0.0, 36.3 - Q.n_real_top40) / 0.2136112   # -0.1%  tau21_b2 < 0.257 and n_real_top40 < 36.3
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 17.48;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.47718 * (0.180235
        + 0.2865776 * max(0.0, Q.sum_z_dr - 0.0422) / 0.03295111   # +28.7%  sum_z_dr > 0.0422
        - 0.1333303 * max(0.0, Q.LHA - 0.208) / 0.07468708   # -13.3%  LHA > 0.208
        - 0.0972421 * max(0.0, 7.15 - Q.log_sum_pt) / 0.2103363   # -9.7%  log_sum_pt < 7.15
        - 0.08131922 * max(0.0, 0.00997 - Q.sum_z_dr2_top15) / 0.004900795   # -8.1%  sum_z_dr2_top15 < 0.00997
        - 0.05762563 * max(0.0, 17.8 - Q.n_dr_0p2_0p4) * max(0.0, 7.14 - Q.log_sum_pt) / 1.903844   # -5.8%  n_dr_0p2_0p4 < 17.8 and log_sum_pt < 7.14
        - 0.05171819 * max(0.0, Q.sum_z_dr - 0.0785) / 0.01293116   # -5.2%  sum_z_dr > 0.0785
        - 0.04254753 * max(0.0, Q.e2 - 0.0211) / 0.01286524   # -4.3%  e2 > 0.0211
        - 0.04006697 * max(0.0, Q.psi_0p3 - 0.977) / 0.01632302   # -4.0%  psi_0p3 > 0.977
        + 0.03072132 * max(0.0, 1040.0 - Q.sum_pt) / 33.98241   # +3.1%  sum_pt < 1040
        - 0.03033979 * max(0.0, Q.sum_z_dr - 0.0954) / 0.00873565   # -3.0%  sum_z_dr > 0.0954
        + 0.02611882 * max(0.0, 0.00793 - Q.sum_z_dr2_top5) / 0.004226697   # +2.6%  sum_z_dr2_top5 < 0.00793
        - 0.02431744 * max(0.0, 16.7 - Q.n_dr_0p2_0p4) * max(0.0, 4.1e-05 - Q.e3) / 0.0001214287   # -2.4%  n_dr_0p2_0p4 < 16.7 and e3 < 4.1e-05
        + 0.0226295 * max(0.0, Q.sum_z_dr - 0.0678) * max(0.0, Q.e2 - 0.0403) / 0.0002746526   # +2.3%  sum_z_dr > 0.0678 and e2 > 0.0403
        - 0.01699326 * max(0.0, 7.84 - Q.n_dr_0p2_0p4) / 2.49575   # -1.7%  n_dr_0p2_0p4 < 7.84
        + 0.01190633 * max(0.0, Q.sum_z_dr - 0.0785) * max(0.0, 0.0814 - Q.tau2) / 0.0002125526   # +1.2%  sum_z_dr > 0.0785 and tau2 < 0.0814
        + 0.01100126 * max(0.0, 999.0 - Q.sum_pt_top40) / 25.73909   # +1.1%  sum_pt_top40 < 999
        + 0.01062696 * max(0.0, 23.1 - Q.n_dr_0p2_0p4) * max(0.0, 0.843 - Q.psi_0p1) / 1.02049   # +1.1%  n_dr_0p2_0p4 < 23.1 and psi_0p1 < 0.843
        + 0.01049683 * max(0.0, 0.0238 - Q.sum_z_dr2_top15) * max(0.0, Q.tau4 - 0.0136) / 5.879968e-05   # +1.0%  sum_z_dr2_top15 < 0.0238 and tau4 > 0.0136
        - 0.009653274 * max(0.0, Q.sum_z_dr - 0.0747) * max(0.0, Q.sum_pt_top20 - 804.0) / 0.7335304   # -1.0%  sum_z_dr > 0.0747 and sum_pt_top20 > 804
        + 0.002465066 * max(0.0, 0.0078 - Q.sum_z_dr2_top5) * max(0.0, Q.sj3_dr23 - 0.254) / 0.0001001916   # +0.2%  sum_z_dr2_top5 < 0.0078 and sj3_dr23 > 0.254
        - 0.002302685 * max(0.0, 995.0 - Q.sum_pt) * max(0.0, Q.z_0 - 0.125) / 1.173307   # -0.2%  sum_pt < 995 and z_0 > 0.125
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 4.097;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.097264 * (-0.145951
        - 0.2022246 * max(0.0, Q.sum_pt_top20 - 741.0) / 193.5905   # -20.2%  sum_pt_top20 > 741
        + 0.1542825 * max(0.0, 0.0847 - Q.tau1) / 0.01981618   # +15.4%  tau1 < 0.0847
        + 0.1421377 * max(0.0, 0.00667 - Q.sum_z_dr2_top15) * max(0.0, 0.0609 - Q.z_dr_0p2_0p4) / 0.0001049325   # +14.2%  sum_z_dr2_top15 < 0.00667 and z_dr_0p2_0p4 < 0.0609
        + 0.09793571 * max(0.0, Q.e2 - 0.0235) / 0.01130334   # +9.8%  e2 > 0.0235
        + 0.06215268 * max(0.0, 2.31 - Q.D2) / 0.532753   # +6.2%  D2 < 2.31
        + 0.06096401 * max(0.0, 0.00603 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.978) / 3.348333e-05   # +6.1%  sum_z_dr2_top15 < 0.00603 and psi_0p3 > 0.978
        + 0.04659414 * max(0.0, Q.z_dr_0_0p05 - 0.806) / 0.03780367   # +4.7%  z_dr_0_0p05 > 0.806
        + 0.03581048 * max(0.0, 978.0 - Q.sum_pt_top50) / 13.21847   # +3.6%  sum_pt_top50 < 978
        - 0.03429644 * max(0.0, 0.000831 - Q.sum_z_dr2_top15) / 9.895887e-05   # -3.4%  sum_z_dr2_top15 < 0.000831
        - 0.0341077 * max(0.0, Q.sum_z_dr2_top15 - 0.0209) / 0.0006751123   # -3.4%  sum_z_dr2_top15 > 0.0209
        - 0.02923685 * max(0.0, Q.sum_z_dr - 0.139) / 0.001993196   # -2.9%  sum_z_dr > 0.139
        + 0.02427744 * max(0.0, Q.LHA - 0.331) * max(0.0, 0.00839 - Q.lam2) / 5.376815e-05   # +2.4%  LHA > 0.331 and lam2 < 0.00839
        + 0.02417252 * max(0.0, 6.8 - Q.log_sum_pt) * max(0.0, 0.0873 - Q.dr_13) / 0.0001090762   # +2.4%  log_sum_pt < 6.8 and dr_13 < 0.0873
        + 0.01536604 * max(0.0, 0.00639 - Q.sum_z_dr2_top15) * max(0.0, Q.n_dr_0p2_0p4 - 7.28) / 0.00411495   # +1.5%  sum_z_dr2_top15 < 0.00639 and n_dr_0p2_0p4 > 7.28
        + 0.01308799 * max(0.0, Q.z_dr_0p1_0p2 - 0.454) * max(0.0, Q.soft5_z - -0.000252) / 4.031952e-05   # +1.3%  z_dr_0p1_0p2 > 0.454 and soft5_z > -0.000252
        - 0.01167668 * max(0.0, 0.0608 - Q.sum_z_dr) * max(0.0, 1010.0 - Q.sum_pt) / 0.2126331   # -1.2%  sum_z_dr < 0.0608 and sum_pt < 1010
        + 0.009977884 * max(0.0, Q.z_top5_slots - 0.772) / 0.009145868   # +1.0%  z_top5_slots > 0.772
        + 0.001698693 * max(0.0, Q.sum_z_dr - 0.145) * max(0.0, Q.soft6_pt - 4.33) / 7.009056e-05   # +0.2%  sum_z_dr > 0.145 and soft6_pt > 4.33
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 6.396;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.395783 * (0.855251
        - 0.2784887 * max(0.0, 0.119 - Q.sum_z_dr) / 0.05430346   # -27.8%  sum_z_dr < 0.119
        - 0.1656851 * max(0.0, Q.LHA - 0.182) / 0.09377754   # -16.6%  LHA > 0.182
        - 0.1066375 * max(0.0, 0.0226 - Q.sum_z_dr2_top5) / 0.0171796   # -10.7%  sum_z_dr2_top5 < 0.0226
        - 0.0758991 * max(0.0, Q.sum_pt_top5 - 394.0) / 207.4505   # -7.6%  sum_pt_top5 > 394
        - 0.05659718 * max(0.0, Q.sj2_dr - 0.0894) / 0.1093605   # -5.7%  sj2_dr > 0.0894
        + 0.04474585 * max(0.0, Q.e2 - 0.0307) / 0.007208684   # +4.5%  e2 > 0.0307
        + 0.04159584 * max(0.0, 2.49 - Q.D2) / 0.6319191   # +4.2%  D2 < 2.49
        - 0.0389457 * max(0.0, 0.217 - Q.tau21_b2) / 0.04424303   # -3.9%  tau21_b2 < 0.217
        - 0.03531864 * max(0.0, 0.126 - Q.z_dr_0p1_0p2) / 0.05042196   # -3.5%  z_dr_0p1_0p2 < 0.126
        + 0.02960799 * max(0.0, 0.0563 - Q.C2) / 0.01001938   # +3.0%  C2 < 0.0563
        + 0.02676047 * max(0.0, Q.psi_0p1 - 0.914) / 0.01850315   # +2.7%  psi_0p1 > 0.914
        - 0.0221054 * max(0.0, Q.psi_0p3 - 0.999) / 0.0002647591   # -2.2%  psi_0p3 > 0.999
        + 0.01486982 * max(0.0, 992.0 - Q.sum_pt) / 13.2457   # +1.5%  sum_pt < 992
        - 0.01462236 * max(0.0, 0.000246 - Q.e3) * max(0.0, 0.861 - Q.tau32) / 1.72231e-05   # -1.5%  e3 < 0.000246 and tau32 < 0.861
        - 0.01261994 * max(0.0, Q.tau1 - 0.178) / 0.002053801   # -1.3%  tau1 > 0.178
        - 0.01026498 * max(0.0, 6.85 - Q.log_sum_pt) / 0.007460519   # -1.0%  log_sum_pt < 6.85
        + 0.007543349 * max(0.0, Q.n_dr_0p1_0p2 - 20.4) / 1.484481   # +0.8%  n_dr_0p1_0p2 > 20.4
        - 0.007033626 * max(0.0, Q.sd_rg - 0.306) / 0.003911787   # -0.7%  sd_rg > 0.306
        + 0.006970937 * max(0.0, 0.452 - Q.tau32) / 0.01334868   # +0.7%  tau32 < 0.452
        - 0.003687494 * max(0.0, Q.e2 - 0.0291) * max(0.0, Q.log_sum_pt - 6.92) / 0.0002087116   # -0.4%  e2 > 0.0291 and log_sum_pt > 6.92
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 3.05;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.050096 * (-0.06753885
        + 0.1973477 * max(0.0, 0.0734 - Q.z_dr_0p2_0p4) / 0.04425952   # +19.7%  z_dr_0p2_0p4 < 0.0734
        + 0.1967319 * max(0.0, 9.35 - Q.n_dr_0p2_0p4) * max(0.0, 32.9 - Q.n_dr_0p1_0p2) / 78.03008   # +19.7%  n_dr_0p2_0p4 < 9.35 and n_dr_0p1_0p2 < 32.9
        - 0.1887079 * max(0.0, 0.00592 - Q.sum_z_dr2_top15) / 0.001951109   # -18.9%  sum_z_dr2_top15 < 0.00592
        + 0.1095816 * max(0.0, Q.z_dr_0_0p05 - 0.814) / 0.03503506   # +11.0%  z_dr_0_0p05 > 0.814
        - 0.1033024 * max(0.0, Q.log_sum_pt - 6.89) / 0.0671817   # -10.3%  log_sum_pt > 6.89
        - 0.08476436 * max(0.0, 9.5 - Q.n_dr_0p2_0p4) * max(0.0, 0.00566 - Q.sum_z_dr2_top10) / 0.007044672   # -8.5%  n_dr_0p2_0p4 < 9.5 and sum_z_dr2_top10 < 0.00566
        + 0.06660975 * max(0.0, 0.00155 - Q.sum_z_dr2_top10) / 0.0003533324   # +6.7%  sum_z_dr2_top10 < 0.00155
        + 0.03434274 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.322 - Q.sd_zg) / 0.005134738   # +3.4%  log_sum_pt > 6.9 and sd_zg < 0.322
        - 0.01861167 * max(0.0, 0.145 - Q.sj3_dr_max) / 0.007362825   # -1.9%  sj3_dr_max < 0.145
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 5.378;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.377564 * (-0.2101323
        + 0.1770427 * max(0.0, 0.0624 - Q.sum_z_dr) / 0.01325987   # +17.7%  sum_z_dr < 0.0624
        - 0.1636546 * max(0.0, Q.log_sum_pt - 6.84) * max(0.0, 0.0118 - Q.mean_eta2) / 0.000921532   # -16.4%  log_sum_pt > 6.84 and mean_eta2 < 0.0118
        - 0.08864348 * max(0.0, 0.194 - Q.LHA) / 0.01672582   # -8.9%  LHA < 0.194
        - 0.0886241 * max(0.0, 0.264 - Q.sj3_dr_max) * max(0.0, Q.sum_z_dr2_top10 - 0.00471) / 3.55658e-05   # -8.9%  sj3_dr_max < 0.264 and sum_z_dr2_top10 > 0.00471
        + 0.08527381 * max(0.0, 3.91 - Q.D2) / 1.592241   # +8.5%  D2 < 3.91
        + 0.06846936 * max(0.0, Q.psi_0p2 - 0.971) * max(0.0, 21.9 - Q.n_dr_0p1_0p2) / 0.1622019   # +6.8%  psi_0p2 > 0.971 and n_dr_0p1_0p2 < 21.9
        + 0.06347694 * max(0.0, 0.00582 - Q.mean_eta2) / 0.002730811   # +6.3%  mean_eta2 < 0.00582
        + 0.0542475 * max(0.0, 0.0452 - Q.sum_z_dr) * max(0.0, Q.log_sum_pt - 6.79) / 0.001176288   # +5.4%  sum_z_dr < 0.0452 and log_sum_pt > 6.79
        + 0.05060841 * max(0.0, 0.261 - Q.sj3_dr_max) / 0.05233653   # +5.1%  sj3_dr_max < 0.261
        + 0.05008803 * max(0.0, Q.psi_0p3 - 0.99) * max(0.0, 0.00525 - Q.sum_z_dr2_top15) / 9.417887e-06   # +5.0%  psi_0p3 > 0.99 and sum_z_dr2_top15 < 0.00525
        + 0.03299175 * max(0.0, 0.241 - Q.sj3_dr_max) * max(0.0, Q.eccentricity - 0.514) / 0.009192499   # +3.3%  sj3_dr_max < 0.241 and eccentricity > 0.514
        - 0.02850551 * max(0.0, 0.144 - Q.sj3_dr_max) / 0.00723067   # -2.9%  sj3_dr_max < 0.144
        - 0.02736038 * max(0.0, Q.eccentricity - 0.893) / 0.0212006   # -2.7%  eccentricity > 0.893
        + 0.01348779 * max(0.0, Q.LHA - 0.364) * max(0.0, Q.log_sum_pt - 6.85) / 0.0005849312   # +1.3%  LHA > 0.364 and log_sum_pt > 6.85
        + 0.007525641 * max(0.0, Q.LHA - 0.388) * max(0.0, 0.00434 - Q.lam2) / 4.046962e-06   # +0.8%  LHA > 0.388 and lam2 < 0.00434
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 7.322;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.322398 * (0.3059107
        + 0.2260886 * max(0.0, 0.000513 - Q.e3) / 0.0004191167   # +22.6%  e3 < 0.000513
        - 0.1001844 * max(0.0, Q.sum_z_dr - 0.0863) / 0.01070934   # -10.0%  sum_z_dr > 0.0863
        - 0.09584979 * max(0.0, Q.sum_z_dr - 0.0974) / 0.008335515   # -9.6%  sum_z_dr > 0.0974
        - 0.08472412 * max(0.0, 6.98 - Q.log_sum_pt) / 0.05852677   # -8.5%  log_sum_pt < 6.98
        - 0.08024372 * max(0.0, 0.000188 - Q.e3) / 0.0001252828   # -8.0%  e3 < 0.000188
        - 0.07641289 * max(0.0, Q.psi_0p2 - 0.905) / 0.06121725   # -7.6%  psi_0p2 > 0.905
        - 0.07058078 * max(0.0, 32.6 - Q.n_dr_0p1_0p2) / 20.34727   # -7.1%  n_dr_0p1_0p2 < 32.6
        + 0.06636288 * max(0.0, Q.LHA - 0.324) / 0.01582852   # +6.6%  LHA > 0.324
        + 0.05408429 * max(0.0, Q.tau4 - 0.0144) / 0.004692259   # +5.4%  tau4 > 0.0144
        + 0.03580369 * max(0.0, Q.sum_z_dr - 0.0956) * max(0.0, 1170.0 - Q.sum_pt) / 1.489596   # +3.6%  sum_z_dr > 0.0956 and sum_pt < 1170
        + 0.0333917 * max(0.0, Q.e2 - 0.0393) / 0.003808525   # +3.3%  e2 > 0.0393
        - 0.0254795 * max(0.0, Q.sum_pt - 1180.0) / 13.04692   # -2.5%  sum_pt > 1180
        - 0.02085807 * max(0.0, 6.92 - Q.log_sum_pt) / 0.02069527   # -2.1%  log_sum_pt < 6.92
        + 0.01897873 * max(0.0, Q.sum_pt_top30 - 1110.0) / 12.74953   # +1.9%  sum_pt_top30 > 1110
        - 0.01095684 * max(0.0, Q.LHA - 0.433) / 0.00101301   # -1.1%  LHA > 0.433
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 7.829;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.828906 * (0.2273626
        - 0.2193279 * max(0.0, 0.153 - Q.tau1) / 0.07154574   # -21.9%  tau1 < 0.153
        - 0.1773537 * max(0.0, Q.sum_z_dr - 0.0837) / 0.01138103   # -17.7%  sum_z_dr > 0.0837
        + 0.1047019 * max(0.0, Q.sd_rg - 0.157) / 0.03675792   # +10.5%  sd_rg > 0.157
        + 0.1004197 * max(0.0, 0.339 - Q.tau21_b2) / 0.1068175   # +10.0%  tau21_b2 < 0.339
        - 0.0635521 * max(0.0, 7.69e-05 - Q.e3) / 3.455163e-05   # -6.4%  e3 < 7.69e-05
        - 0.05260822 * max(0.0, 0.337 - Q.tau21_b2) * max(0.0, 1150.0 - Q.sum_pt_top50) / 12.95172   # -5.3%  tau21_b2 < 0.337 and sum_pt_top50 < 1150
        + 0.05193029 * max(0.0, 0.151 - Q.sd_rg) / 0.05044136   # +5.2%  sd_rg < 0.151
        - 0.04581739 * max(0.0, Q.sd_rg - 0.206) / 0.02003911   # -4.6%  sd_rg > 0.206
        + 0.0425383 * max(0.0, Q.sum_z_dr - 0.121) / 0.00424781   # +4.3%  sum_z_dr > 0.121
        - 0.04172816 * max(0.0, 17.9 - Q.n_dr_0p1_0p2) / 7.292095   # -4.2%  n_dr_0p1_0p2 < 17.9
        + 0.02955736 * max(0.0, 0.111 - Q.z_dr_0p1_0p2) / 0.0422266   # +3.0%  z_dr_0p1_0p2 < 0.111
        - 0.02602741 * max(0.0, 3.34e-05 - Q.e3) / 8.150646e-06   # -2.6%  e3 < 3.34e-05
        - 0.01990949 * max(0.0, 0.338 - Q.tau21_b2) * max(0.0, 0.00786 - Q.sum_z_dr2_top15) / 0.0001783404   # -2.0%  tau21_b2 < 0.338 and sum_z_dr2_top15 < 0.00786
        + 0.01699015 * max(0.0, Q.sum_z_dr - 0.0732) * max(0.0, 0.389 - Q.max_dr) / 0.000418284   # +1.7%  sum_z_dr > 0.0732 and max_dr < 0.389
        - 0.007537828 * max(0.0, Q.sum_z_dr - 0.119) * max(0.0, 0.355 - Q.max_dr) / 4.098121e-05   # -0.8%  sum_z_dr > 0.119 and max_dr < 0.355
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 1.119;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.119321 * (0.04842222
        + 0.3239321 * max(0.0, 0.00687 - Q.sum_z_dr2_top5) / 0.003420603   # +32.4%  sum_z_dr2_top5 < 0.00687
        + 0.2250929 * max(0.0, 0.169 - Q.sd_rg) / 0.0602754   # +22.5%  sd_rg < 0.169
        - 0.185259 * max(0.0, 0.0713 - Q.tau1) / 0.01373273   # -18.5%  tau1 < 0.0713
        - 0.1710095 * max(0.0, Q.n_dr_0p2_0p4 - 9.09) / 3.200911   # -17.1%  n_dr_0p2_0p4 > 9.09
        - 0.09470649 * max(0.0, Q.log_sum_pt - 6.92) / 0.04530211   # -9.5%  log_sum_pt > 6.92
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.695309768907563, 2.8668403886554623, 0.31762163865546217, 0.42173061974789916, 0.8709539915966387, 1.1827044117647059, 0.6338335084033614, 0.5832515756302521, 1.3142387605042016, 1.0690186974789917, 1.358101050420168, 0.6320171218487395, 0.6896436974789916, 1.5129251575630251, 0.3345581407563025, 0.3151888655462185]
T = [4.1139278623949584, 2.6794226020877105, 4.651627754070378, 4.840279657956932, 4.330653388425683]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +44%, n4 -19%, n9 +13%, n5 -9%, n3 -8%, n12 +4% ...
            + 0.4355388 * h[1] / H_AVG[1]
            - 0.185245 * h[4] / H_AVG[4]
            + 0.133987 * h[9] / H_AVG[9]
            - 0.08983996 * h[5] / H_AVG[5]
            - 0.07688466 * h[3] / H_AVG[3]
            + 0.03928976 * h[12] / H_AVG[12]
            + 0.02407346 * h[6] / H_AVG[6]
            + 0.009983151 * h[8] / H_AVG[8]
            - 0.005158167 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -35%, n9 +22%, n1 -20%, n12 +9%, n11 -6%, n6 +5% ...
            - 0.3453687 * h[4] / H_AVG[4]
            + 0.2244226 * h[9] / H_AVG[9]
            - 0.2006151 * h[1] / H_AVG[1]
            + 0.08847616 * h[12] / H_AVG[12]
            - 0.05896953 * h[11] / H_AVG[11]
            + 0.05174663 * h[6] / H_AVG[6]
            + 0.01481763 * h[2] / H_AVG[2]
            - 0.007919739 * h[10] / H_AVG[10]
            + 0.007663957 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +15%, n0 +11%, n14 -10%, n11 +8%, n7 -8% ...
            - 0.2472165 * h[8] / H_AVG[8]
            + 0.1469918 * h[5] / H_AVG[5]
            + 0.1121075 * h[0] / H_AVG[0]
            - 0.09889386 * h[14] / H_AVG[14]
            + 0.08067287 * h[11] / H_AVG[11]
            - 0.0783666 * h[7] / H_AVG[7]
            + 0.06436251 * h[4] / H_AVG[4]
            - 0.06023005 * h[12] / H_AVG[12]
            - 0.05027226 * h[9] / H_AVG[9]
            + 0.03966507 * h[3] / H_AVG[3]
            - 0.01270478 * h[15] / H_AVG[15]
            + 0.008516286 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -25%, n0 -20%, n5 +17%, n6 -13%, n7 +11%, n12 +4% ...
            - 0.2545512 * h[8] / H_AVG[8]
            - 0.1975198 * h[0] / H_AVG[0]
            + 0.1679881 * h[5] / H_AVG[5]
            - 0.1268576 * h[6] / H_AVG[6]
            + 0.1092027 * h[7] / H_AVG[7]
            + 0.04452504 * h[12] / H_AVG[12]
            + 0.03811911 * h[3] / H_AVG[3]
            + 0.03662882 * h[15] / H_AVG[15]
            - 0.02460769 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -32%, n10 +31%, n5 -13%, n8 +6%, n12 -6%, n7 -4% ...
            - 0.3166008 * h[13] / H_AVG[13]
            + 0.3087019 * h[10] / H_AVG[10]
            - 0.1280159 * h[5] / H_AVG[5]
            + 0.06401395 * h[8] / H_AVG[8]
            - 0.05971764 * h[12] / H_AVG[12]
            - 0.0378787 * h[7] / H_AVG[7]
            + 0.03770883 * h[4] / H_AVG[4]
            - 0.02729284 * h[15] / H_AVG[15]
            + 0.02006942 * h[0] / H_AVG[0]
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
