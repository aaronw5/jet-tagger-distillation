"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  12.7%   (on for 55% of jets)
  neuron  1:  11.8%   (on for 95% of jets)
  neuron  5:  11.3%   (on for 80% of jets)
  neuron  4:  10.0%   (on for 69% of jets)
  neuron  0:   7.8%   (on for 65% of jets)
  neuron 13:   7.4%   (on for 89% of jets)
  neuron  9:   6.8%   (on for 70% of jets)
  neuron 10:   6.8%   (on for 81% of jets)
  neuron 12:   4.9%   (on for 58% of jets)
  neuron  7:   4.7%   (on for 43% of jets)
  neuron  6:   4.0%   (on for 70% of jets)
  neuron 11:   3.3%   (on for 72% of jets)
  neuron  3:   3.3%   (on for 51% of jets)
  neuron 15:   2.6%   (on for 68% of jets)
  neuron 14:   2.0%   (on for 44% of jets)
  neuron  2:   0.6%   (on for 39% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 79.0% (the network: 81.1%); same class as the network for 90.1% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
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
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.z_7                    pT of particle 7 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.z_1st                  largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_1               |Δφ| of particle 1
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
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
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau32                  N-subjettiness τ3/τ2
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
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
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_2=pt[2],
        pt_3=pt[3],
        soft1_pt=softp(1, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        z_7=z[7],
        soft1_z=softp(1, 'z'),
        soft5_z=softp(5, 'z'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        z_1st=zs[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        absphi_1=abs(phi[1]),
        sj2_dr=subjets(2)["dr"][0],
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
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
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
    )


def neuron_0(Q):
    # scale S = 41.43;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 41.42723 * (0.0203248
        - 0.1318362 * max(0.0, 0.0973 - Q.sum_z_dr) / 0.03641073   # -13.2%  sum_z_dr < 0.0973
        + 0.1126993 * max(0.0, 0.0859 - Q.sum_z_dr) / 0.02746365   # +11.3%  sum_z_dr < 0.0859
        - 0.09218554 * max(0.0, 0.314 - Q.LHA) / 0.07033133   # -9.2%  LHA < 0.314
        + 0.09075295 * max(0.0, 0.0824 - Q.sum_z_dr) / 0.0248983   # +9.1%  sum_z_dr < 0.0824
        + 0.06451175 * max(0.0, Q.sum_z_dr - 0.0876) / 0.010399   # +6.5%  sum_z_dr > 0.0876
        + 0.06199907 * max(0.0, 0.272 - Q.LHA) / 0.04521918   # +6.2%  LHA < 0.272
        - 0.05802035 * max(0.0, 0.0652 - Q.sum_z_dr) / 0.01456741   # -5.8%  sum_z_dr < 0.0652
        - 0.05184727 * max(0.0, Q.mean_eta2 - 0.000317) / 0.004339169   # -5.2%  mean_eta2 > 0.000317
        - 0.04743612 * max(0.0, Q.LHA - 0.325) / 0.0155964   # -4.7%  LHA > 0.325
        + 0.04552894 * max(0.0, Q.sj2_dr - 0.137) / 0.07037828   # +4.6%  sj2_dr > 0.137
        - 0.04172114 * max(0.0, Q.sj2_dr - 0.148) / 0.06217234   # -4.2%  sj2_dr > 0.148
        + 0.03348956 * max(0.0, 0.00615 - Q.mean_phi2) / 0.002983612   # +3.3%  mean_phi2 < 0.00615
        + 0.02179448 * max(0.0, 0.00779 - Q.sum_z_dr2_top15) / 0.003190406   # +2.2%  sum_z_dr2_top15 < 0.00779
        + 0.01829674 * max(0.0, 18.3 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_pt_top50 - 890.0) / 1602.502   # +1.8%  n_dr_0p2_0p4 < 18.3 and sum_pt_top50 > 890
        - 0.01790714 * max(0.0, Q.mean_phi2 - 0.00574) / 0.00158853   # -1.8%  mean_phi2 > 0.00574
        - 0.01662052 * max(0.0, 0.00545 - Q.sum_z_dr2_top15) / 0.001700104   # -1.7%  sum_z_dr2_top15 < 0.00545
        - 0.01345457 * max(0.0, 0.0527 - Q.sum_z_dr) / 0.009320829   # -1.3%  sum_z_dr < 0.0527
        - 0.009703656 * max(0.0, 17.5 - Q.n_dr_0p2_0p4) * max(0.0, Q.log_sum_pt - 6.91) / 0.4999945   # -1.0%  n_dr_0p2_0p4 < 17.5 and log_sum_pt > 6.91
        - 0.009575242 * max(0.0, 0.0371 - Q.sum_z_dr) / 0.004517947   # -1.0%  sum_z_dr < 0.0371
        + 0.006982176 * max(0.0, 1070.0 - Q.sum_pt) / 55.2008   # +0.7%  sum_pt < 1070
        + 0.006877645 * max(0.0, 0.201 - Q.LHA) / 0.01862234   # +0.7%  LHA < 0.201
        - 0.005838746 * max(0.0, 2.8e-05 - Q.e3) / 6.186268e-06   # -0.6%  e3 < 2.8e-05
        - 0.005564446 * max(0.0, Q.sum_pt_top50 - 1050.0) / 31.40593   # -0.6%  sum_pt_top50 > 1050
        + 0.005349226 * max(0.0, 0.305 - Q.tau21_b2) / 0.08724551   # +0.5%  tau21_b2 < 0.305
        + 0.004708462 * max(0.0, 0.019 - Q.e2) / 0.002417082   # +0.5%  e2 < 0.019
        - 0.003426745 * max(0.0, 0.448 - Q.tau21) / 0.0992731   # -0.3%  tau21 < 0.448
        + 0.003258756 * max(0.0, Q.log_sum_pt - 7.02) / 0.01595759   # +0.3%  log_sum_pt > 7.02
        - 0.003212777 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # -0.3%  sum_pt < 1010
        + 0.002379491 * max(0.0, Q.sum_z_dr2_top10 - 0.00923) / 0.002225185   # +0.2%  sum_z_dr2_top10 > 0.00923
        + 0.002281618 * max(0.0, Q.psi_0p3 - 0.996) * max(0.0, 28.5 - Q.n_dr_0p1_0p2) / 0.03029523   # +0.2%  psi_0p3 > 0.996 and n_dr_0p1_0p2 < 28.5
        - 0.002123456 * max(0.0, Q.sd_rg - 0.167) / 0.0323415   # -0.2%  sd_rg > 0.167
        - 0.002056446 * max(0.0, Q.psi_0p1 - 0.909) / 0.02033242   # -0.2%  psi_0p1 > 0.909
        + 0.001855741 * max(0.0, Q.sd_rg - 0.313) / 0.003285394   # +0.2%  sd_rg > 0.313
        + 0.001410394 * max(0.0, Q.sum_pt_top40 - 971.0) * max(0.0, 0.0453 - Q.dr_5) / 0.9953783   # +0.1%  sum_pt_top40 > 971 and dr_5 < 0.0453
        + 0.001052112 * max(0.0, Q.z_dr_0p05_0p1 - 0.41) / 0.0783922   # +0.1%  z_dr_0p05_0p1 > 0.41
        - 0.001016935 * max(0.0, Q.e2 - 0.0495) / 0.001831686   # -0.1%  e2 > 0.0495
        - 0.0006297411 * max(0.0, Q.psi_0p3 - 0.995) * max(0.0, 0.0289 - Q.dr_5) / 9.734488e-06   # -0.1%  psi_0p3 > 0.995 and dr_5 < 0.0289
        + 0.0005945142 * max(0.0, Q.sum_pt_top40 - 1000.0) * max(0.0, 0.163 - Q.sj2_dr) / 1.588973   # +0.1%  sum_pt_top40 > 1000 and sj2_dr < 0.163
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 9.415;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.414722 * (-0.04907208
        + 0.2224646 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # +22.2%  log_sum_pt > 6.91
        - 0.1364024 * max(0.0, Q.sum_pt_top50 - 919.0) / 122.3039   # -13.6%  sum_pt_top50 > 919
        + 0.108649 * max(0.0, Q.pt_entropy - 1.88) / 0.9835581   # +10.9%  pt_entropy > 1.88
        + 0.09061668 * max(0.0, Q.n_real_top50 - 34.6) / 8.786106   # +9.1%  n_real_top50 > 34.6
        + 0.08256629 * max(0.0, Q.z_top30_slots - 0.936) / 0.0327991   # +8.3%  z_top30_slots > 0.936
        + 0.05058515 * max(0.0, Q.n_particles - 48.6) / 4.992088   # +5.1%  n_particles > 48.6
        - 0.04301373 * max(0.0, 4.26 - Q.D2_b2) / 2.165574   # -4.3%  D2_b2 < 4.26
        - 0.04152182 * max(0.0, Q.log_sum_pt - 7.0) / 0.01916257   # -4.2%  log_sum_pt > 7
        + 0.03913642 * max(0.0, 0.075 - Q.C2) / 0.02105478   # +3.9%  C2 < 0.075
        - 0.03734669 * max(0.0, 7.65 - Q.n_dr_0p2_0p4) / 2.391896   # -3.7%  n_dr_0p2_0p4 < 7.65
        - 0.03082601 * max(0.0, Q.n_particles - 37.3) * max(0.0, 0.00237 - Q.soft1_z) / 0.01707167   # -3.1%  n_particles > 37.3 and soft1_z < 0.00237
        + 0.02785604 * max(0.0, Q.n_particles - 30.0) * max(0.0, 0.0111 - Q.zdr_0) / 0.06304252   # +2.8%  n_particles > 30 and zdr_0 < 0.0111
        + 0.0230008 * max(0.0, 0.0013 - Q.sum_z_dr2_top10) / 0.0002772678   # +2.3%  sum_z_dr2_top10 < 0.0013
        - 0.02252025 * max(0.0, 0.0412 - Q.M3) * max(0.0, Q.psi_0p3 - 0.956) / 0.0005287328   # -2.3%  M3 < 0.0412 and psi_0p3 > 0.956
        + 0.01773885 * max(0.0, Q.n_particles - 43.1) * max(0.0, Q.tau32 - 0.293) / 2.854809   # +1.8%  n_particles > 43.1 and tau32 > 0.293
        - 0.01756034 * max(0.0, Q.z_dr_0_0p05 - 0.757) * max(0.0, 12.7 - Q.n_dr_0p05_0p1) / 0.4468263   # -1.8%  z_dr_0_0p05 > 0.757 and n_dr_0p05_0p1 < 12.7
        + 0.00539574 * max(0.0, Q.z_top30_slots - 0.953) * max(0.0, Q.pt1_dr01 - 6.23) / 0.07525836   # +0.5%  z_top30_slots > 0.953 and pt1_dr01 > 6.23
        + 0.00279917 * max(0.0, 919.0 - Q.sum_pt_top50) / 5.607109   # +0.3%  sum_pt_top50 < 919
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 10.11;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.10502 * (0.2147447
        - 0.1103 * max(0.0, Q.tau1 - 0.0489) / 0.04388128   # -11.0%  tau1 > 0.0489
        + 0.1057944 * max(0.0, 0.139 - Q.sum_z_dr) / 0.07174862   # +10.6%  sum_z_dr < 0.139
        - 0.07762206 * max(0.0, 0.319 - Q.LHA) / 0.07399742   # -7.8%  LHA < 0.319
        - 0.06241612 * max(0.0, 0.012 - Q.mean_phi2) / 0.007983751   # -6.2%  mean_phi2 < 0.012
        - 0.06226989 * max(0.0, 0.0119 - Q.mean_eta2) / 0.00789509   # -6.2%  mean_eta2 < 0.0119
        - 0.05879597 * max(0.0, Q.z_top5_slots - 0.445) / 0.1421375   # -5.9%  z_top5_slots > 0.445
        + 0.05494078 * max(0.0, 0.00944 - Q.sum_z_dr2_top15) / 0.004477241   # +5.5%  sum_z_dr2_top15 < 0.00944
        - 0.04246668 * max(0.0, 0.00756 - Q.sum_z_dr2_top15) / 0.00302202   # -4.2%  sum_z_dr2_top15 < 0.00756
        - 0.04009908 * max(0.0, Q.sum_z_dr - 0.0623) / 0.02015931   # -4.0%  sum_z_dr > 0.0623
        + 0.03393888 * max(0.0, Q.e2 - 0.017) / 0.0157318   # +3.4%  e2 > 0.017
        + 0.03344543 * max(0.0, Q.LHA - 0.27) / 0.03630149   # +3.3%  LHA > 0.27
        + 0.03006738 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922642   # +3.0%  log_sum_pt > 6.9
        + 0.02875231 * max(0.0, Q.sum_pt_top5 - 580.0) / 79.38326   # +2.9%  sum_pt_top5 > 580
        - 0.02803906 * max(0.0, Q.z_top50_slots - 0.958) / 0.03489351   # -2.8%  z_top50_slots > 0.958
        - 0.02716477 * max(0.0, 577.0 - Q.sum_pt_top5) / 64.43676   # -2.7%  sum_pt_top5 < 577
        - 0.01977104 * max(0.0, Q.sum_pt - 1070.0) / 28.99664   # -2.0%  sum_pt > 1070
        + 0.01805776 * max(0.0, Q.LHA - 0.319) / 0.01705365   # +1.8%  LHA > 0.319
        + 0.01591877 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, 0.0204 - Q.sum_z_dr2_top15) / 0.0007963345   # +1.6%  log_sum_pt > 6.91 and sum_z_dr2_top15 < 0.0204
        + 0.01553897 * max(0.0, 0.199 - Q.LHA) / 0.01806923   # +1.6%  LHA < 0.199
        - 0.01544791 * max(0.0, 0.303 - Q.sd_rg) / 0.1669535   # -1.5%  sd_rg < 0.303
        + 0.01226696 * max(0.0, 1050.0 - Q.sum_pt_top30) / 78.4544   # +1.2%  sum_pt_top30 < 1050
        - 0.01200077 * max(0.0, Q.sd_rg - 0.188) / 0.02490103   # -1.2%  sd_rg > 0.188
        - 0.01094177 * max(0.0, 1140.0 - Q.sum_pt) / 113.4019   # -1.1%  sum_pt < 1140
        - 0.008929214 * max(0.0, Q.psi_0p3 - 0.989) / 0.0065384   # -0.9%  psi_0p3 > 0.989
        + 0.008222531 * max(0.0, 0.444 - Q.z_top5_slots) / 0.01810215   # +0.8%  z_top5_slots < 0.444
        + 0.00721045 * max(0.0, Q.sum_pt_top50 - 1050.0) / 31.40593   # +0.7%  sum_pt_top50 > 1050
        - 0.007143002 * max(0.0, 12.1 - Q.n_dr_0p2_0p4) / 5.268628   # -0.7%  n_dr_0p2_0p4 < 12.1
        - 0.006502515 * max(0.0, 47.9 - Q.n_real_top50) / 7.391233   # -0.7%  n_real_top50 < 47.9
        + 0.005866351 * max(0.0, Q.mean_eta2 - 0.0116) / 0.0006751665   # +0.6%  mean_eta2 > 0.0116
        + 0.005764708 * max(0.0, Q.mean_phi2 - 0.0115) / 0.0006901956   # +0.6%  mean_phi2 > 0.0115
        + 0.005307504 * max(0.0, Q.sd_rg - 0.262) / 0.009327382   # +0.5%  sd_rg > 0.262
        + 0.005001361 * max(0.0, 0.506 - Q.psi_0p1) / 0.03948349   # +0.5%  psi_0p1 < 0.506
        - 0.0041458 * max(0.0, Q.z_dr_0p1_0p2 - 0.365) / 0.03298693   # -0.4%  z_dr_0p1_0p2 > 0.365
        - 0.003414431 * max(0.0, Q.LHA - 0.41) / 0.002613856   # -0.3%  LHA > 0.41
        - 0.003032171 * max(0.0, Q.log_sum_pt - 7.07) / 0.01017945   # -0.3%  log_sum_pt > 7.07
        + 0.002927365 * max(0.0, Q.lam2 - 0.00021) / 0.00121234   # +0.3%  lam2 > 0.00021
        - 0.002346147 * max(0.0, 924.0 - Q.sum_pt_top30) / 18.09761   # -0.2%  sum_pt_top30 < 924
        - 0.00199365 * max(0.0, Q.sum_z_dr2_top10 - 0.00768) / 0.002556583   # -0.2%  sum_z_dr2_top10 > 0.00768
        - 0.001703216 * max(0.0, Q.log_sum_pt - 6.95) * max(0.0, Q.C2 - 0.0545) / 0.0005092021   # -0.2%  log_sum_pt > 6.95 and C2 > 0.0545
        - 0.001515244 * max(0.0, Q.C2 - 0.112) / 0.003780637   # -0.2%  C2 > 0.112
        - 0.001423824 * max(0.0, Q.e2 - 0.0544) / 0.001240325   # -0.1%  e2 > 0.0544
        + 0.0009236579 * max(0.0, 0.973 - Q.z_top50_slots) / 0.001725247   # +0.1%  z_top50_slots < 0.973
        + 0.0004506386 * max(0.0, Q.log_sum_pt - 7.02) * max(0.0, Q.C2 - 0.0902) / 7.465104e-05   # +0.0%  log_sum_pt > 7.02 and C2 > 0.0902
        - 0.0001194919 * max(0.0, 993.0 - Q.sum_pt) / 13.47621   # -0.0%  sum_pt < 993
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 4.211;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.210664 * (-0.3562383
        + 0.2024765 * max(0.0, Q.LHA - 0.201) / 0.07967857   # +20.2%  LHA > 0.201
        + 0.1185068 * max(0.0, 0.0875 - Q.sum_z_dr) / 0.02867771   # +11.9%  sum_z_dr < 0.0875
        - 0.0563726 * max(0.0, Q.n_dr_0p1_0p2 - 1.34) / 11.24958   # -5.6%  n_dr_0p1_0p2 > 1.34
        + 0.04879029 * max(0.0, 0.0294 - Q.tau4) / 0.0130025   # +4.9%  tau4 < 0.0294
        - 0.04824146 * max(0.0, 0.309 - Q.N2) / 0.04779496   # -4.8%  N2 < 0.309
        - 0.04815637 * max(0.0, Q.mean_phi2 - 0.0022) / 0.003004005   # -4.8%  mean_phi2 > 0.0022
        + 0.04151802 * max(0.0, 0.0333 - Q.e2) / 0.008404732   # +4.2%  e2 < 0.0333
        + 0.04073537 * max(0.0, 0.00668 - Q.mean_eta2) / 0.003396494   # +4.1%  mean_eta2 < 0.00668
        + 0.04045254 * max(0.0, 0.00156 - Q.lam2) / 0.0007813397   # +4.0%  lam2 < 0.00156
        + 0.04012632 * max(0.0, 11.5 - Q.n_dr_0p2_0p4) / 4.841217   # +4.0%  n_dr_0p2_0p4 < 11.5
        - 0.03860355 * max(0.0, 0.0469 - Q.tau1) / 0.005784575   # -3.9%  tau1 < 0.0469
        + 0.03636629 * max(0.0, 4.5 - Q.n_dr_0p2_0p4) / 0.9169235   # +3.6%  n_dr_0p2_0p4 < 4.5
        - 0.02777864 * max(0.0, Q.n_particles - 39.9) / 9.587421   # -2.8%  n_particles > 39.9
        - 0.02249094 * max(0.0, Q.mean_eta2 - 0.00579) / 0.001567911   # -2.2%  mean_eta2 > 0.00579
        - 0.02068331 * max(0.0, Q.n_real_top40 - 34.2) / 3.940746   # -2.1%  n_real_top40 > 34.2
        - 0.0194996 * max(0.0, 13.7 - Q.n_dr_0_0p05) / 4.414314   # -1.9%  n_dr_0_0p05 < 13.7
        - 0.01943216 * max(0.0, 0.997 - Q.z_top50_slots) / 0.006598572   # -1.9%  z_top50_slots < 0.997
        + 0.01757879 * max(0.0, Q.sum_z_dr - 0.124) / 0.003815381   # +1.8%  sum_z_dr > 0.124
        + 0.01447902 * max(0.0, 0.724 - Q.D2_b2) / 0.126749   # +1.4%  D2_b2 < 0.724
        + 0.01360152 * max(0.0, 52.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 845.0) / 1859.462   # +1.4%  n_particles < 52 and sum_pt_top40 > 845
        + 0.01198665 * max(0.0, Q.lam2 - 0.000429) / 0.001051495   # +1.2%  lam2 > 0.000429
        + 0.01061771 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, Q.eccentricity - 0.87) / 5.588453e-05   # +1.1%  psi_0p3 > 0.997 and eccentricity > 0.87
        + 0.01059604 * max(0.0, 33.3 - Q.n_real_top40) / 1.628335   # +1.1%  n_real_top40 < 33.3
        - 0.01008994 * max(0.0, Q.e2 - 0.0377) / 0.004300136   # -1.0%  e2 > 0.0377
        + 0.009185676 * max(0.0, Q.psi_0p3 - 0.998) / 0.0006668586   # +0.9%  psi_0p3 > 0.998
        + 0.00858713 * max(0.0, 0.999 - Q.psi_0p3) / 0.01217425   # +0.9%  psi_0p3 < 0.999
        + 0.006972924 * max(0.0, 0.329 - Q.tau21_b2) / 0.1008957   # +0.7%  tau21_b2 < 0.329
        - 0.006927503 * max(0.0, 4.3 - Q.n_dr_0p2_0p4) * max(0.0, 0.0344 - Q.e2) / 0.005730725   # -0.7%  n_dr_0p2_0p4 < 4.3 and e2 < 0.0344
        + 0.005245518 * max(0.0, 0.358 - Q.tau21) * max(0.0, 1050.0 - Q.sum_pt) / 1.871789   # +0.5%  tau21 < 0.358 and sum_pt < 1050
        - 0.00390084 * max(0.0, 14.5 - Q.n_dr_0p1_0p2) * max(0.0, 1020.0 - Q.sum_pt_top40) / 112.5009   # -0.4%  n_dr_0p1_0p2 < 14.5 and sum_pt_top40 < 1020
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 5.995;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.994603 * (0.2051845
        - 0.1269105 * max(0.0, 0.00764 - Q.sum_z_dr2_top15) / 0.003080073   # -12.7%  sum_z_dr2_top15 < 0.00764
        + 0.1244137 * max(0.0, 0.0105 - Q.sum_z_dr2_top15) / 0.00532722   # +12.4%  sum_z_dr2_top15 < 0.0105
        + 0.09383993 * max(0.0, Q.sum_z_dr - 0.0941) / 0.009000531   # +9.4%  sum_z_dr > 0.0941
        - 0.09136507 * max(0.0, 0.0552 - Q.sum_z_dr) / 0.0102565   # -9.1%  sum_z_dr < 0.0552
        - 0.07262442 * max(0.0, Q.LHA - 0.332) / 0.01408915   # -7.3%  LHA > 0.332
        + 0.06970852 * max(0.0, 0.242 - Q.LHA) / 0.03214423   # +7.0%  LHA < 0.242
        + 0.05819896 * max(0.0, 16.6 - Q.n_dr_0p2_0p4) / 8.787901   # +5.8%  n_dr_0p2_0p4 < 16.6
        - 0.04749214 * max(0.0, Q.n_particles - 37.9) / 10.86628   # -4.7%  n_particles > 37.9
        - 0.04333138 * max(0.0, 0.179 - Q.sj2_dr) / 0.02915314   # -4.3%  sj2_dr < 0.179
        - 0.04047912 * max(0.0, 0.044 - Q.sum_z_dr) / 0.006436505   # -4.0%  sum_z_dr < 0.044
        - 0.03608867 * max(0.0, 0.394 - Q.sj2_zsoft) / 0.1602498   # -3.6%  sj2_zsoft < 0.394
        + 0.03560798 * max(0.0, 0.1 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.997) / 4.620253e-05   # +3.6%  sum_z_dr < 0.1 and psi_0p3 > 0.997
        - 0.02504937 * max(0.0, 0.0505 - Q.z_dr_0p2_0p4) / 0.02745174   # -2.5%  z_dr_0p2_0p4 < 0.0505
        + 0.0231658 * max(0.0, Q.n_dr_0_0p05 - 8.57) / 5.859485   # +2.3%  n_dr_0_0p05 > 8.57
        - 0.01883519 * max(0.0, 1060.0 - Q.sum_pt_top50) / 53.00916   # -1.9%  sum_pt_top50 < 1060
        - 0.01689083 * max(0.0, 0.0614 - Q.sum_z_dr) * max(0.0, 1070.0 - Q.sum_pt_top40) / 0.6888017   # -1.7%  sum_z_dr < 0.0614 and sum_pt_top40 < 1070
        + 0.01507615 * max(0.0, 37.0 - Q.n_pt_above_1) / 3.96384   # +1.5%  n_pt_above_1 < 37
        - 0.01376335 * max(0.0, Q.n_particles - 34.4) * max(0.0, Q.soft1_pt - 0.0851) / 11.39584   # -1.4%  n_particles > 34.4 and soft1_pt > 0.0851
        - 0.01221955 * max(0.0, 0.0627 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.998) / 7.111783e-06   # -1.2%  sum_z_dr < 0.0627 and psi_0p3 > 0.998
        - 0.01092958 * max(0.0, Q.sum_z_dr2_top15 - 0.000577) * max(0.0, 0.444 - Q.sj3_pairmin_over_m) / 0.000974978   # -1.1%  sum_z_dr2_top15 > 0.000577 and sj3_pairmin_over_m < 0.444
        + 0.008334047 * max(0.0, Q.tau21_b2 - 0.542) / 0.0362024   # +0.8%  tau21_b2 > 0.542
        - 0.004974006 * max(0.0, Q.zdr_0 - 0.00912) / 0.003419403   # -0.5%  zdr_0 > 0.00912
        + 0.004275016 * max(0.0, Q.n_particles - 20.6) * max(0.0, Q.lam2 - 0.00199) / 0.02190344   # +0.4%  n_particles > 20.6 and lam2 > 0.00199
        - 0.003375266 * max(0.0, Q.e2 - 0.0568) / 0.001001653   # -0.3%  e2 > 0.0568
        + 0.003051464 * max(0.0, 0.00519 - Q.sum_z_dr2_top15) * max(0.0, Q.sum_pt_top40 - 1070.0) / 0.06486636   # +0.3%  sum_z_dr2_top15 < 0.00519 and sum_pt_top40 > 1070
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 13.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.29184 * (0.4152925
        - 0.3098514 * max(0.0, 0.156 - Q.sum_z_dr) / 0.08744149   # -31.0%  sum_z_dr < 0.156
        - 0.1194105 * max(0.0, Q.LHA - 0.192) / 0.08626003   # -11.9%  LHA > 0.192
        + 0.0961423 * max(0.0, 7.06 - Q.log_sum_pt) / 0.1265255   # +9.6%  log_sum_pt < 7.06
        - 0.05771739 * max(0.0, 0.11 - Q.tau1) / 0.03635877   # -5.8%  tau1 < 0.11
        - 0.04746231 * max(0.0, 1070.0 - Q.sum_pt_top50) / 60.65974   # -4.7%  sum_pt_top50 < 1070
        + 0.04685411 * max(0.0, 1080.0 - Q.sum_pt) / 62.97038   # +4.7%  sum_pt < 1080
        - 0.03813409 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # -3.8%  sum_pt < 1000
        + 0.03103254 * max(0.0, 0.00587 - Q.mean_eta2) / 0.002768318   # +3.1%  mean_eta2 < 0.00587
        + 0.02905204 * max(0.0, 0.00588 - Q.mean_phi2) / 0.002778093   # +2.9%  mean_phi2 < 0.00588
        + 0.02186911 * max(0.0, 0.0409 - Q.e2) / 0.01339542   # +2.2%  e2 < 0.0409
        - 0.01966781 * max(0.0, 0.0609 - Q.z_dr_0p2_0p4) / 0.03490271   # -2.0%  z_dr_0p2_0p4 < 0.0609
        - 0.01866018 * max(0.0, 838.0 - Q.sum_pt_top5) / 253.3483   # -1.9%  sum_pt_top5 < 838
        - 0.01726096 * max(0.0, Q.mean_eta2 - 0.00607) / 0.001499542   # -1.7%  mean_eta2 > 0.00607
        - 0.01675763 * max(0.0, Q.mean_phi2 - 0.00608) / 0.001504998   # -1.7%  mean_phi2 > 0.00608
        + 0.01481949 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) / 4.217948   # +1.5%  n_dr_0p2_0p4 < 10.6
        + 0.01457389 * max(0.0, 59.8 - Q.n_particles) / 15.01657   # +1.5%  n_particles < 59.8
        + 0.0142151 * max(0.0, 0.0369 - Q.sum_z_dr) / 0.004466778   # +1.4%  sum_z_dr < 0.0369
        + 0.01335713 * max(0.0, Q.sd_rg - 0.157) / 0.03675792   # +1.3%  sd_rg > 0.157
        - 0.00953381 * max(0.0, 60.5 - Q.n_particles) * max(0.0, 0.0891 - Q.C2) / 0.5949382   # -1.0%  n_particles < 60.5 and C2 < 0.0891
        - 0.008786648 * max(0.0, Q.sd_rg - 0.198) / 0.02207763   # -0.9%  sd_rg > 0.198
        + 0.007978023 * max(0.0, 0.443 - Q.max_dr) / 0.09728676   # +0.8%  max_dr < 0.443
        + 0.007558835 * max(0.0, Q.sum_pt_top50 - 1060.0) / 28.70594   # +0.8%  sum_pt_top50 > 1060
        - 0.006335901 * max(0.0, Q.sum_pt - 1160.0) / 14.95839   # -0.6%  sum_pt > 1160
        + 0.006251406 * max(0.0, 0.034 - Q.M3) / 0.01070782   # +0.6%  M3 < 0.034
        + 0.005612444 * max(0.0, Q.n_dr_0p1_0p2 - 8.32) / 5.694632   # +0.6%  n_dr_0p1_0p2 > 8.32
        - 0.00538558 * max(0.0, Q.soft1_pt - 1.72) / 0.08218627   # -0.5%  soft1_pt > 1.72
        + 0.003791704 * max(0.0, Q.z_dr_0_0p05 - 0.887) / 0.01388394   # +0.4%  z_dr_0_0p05 > 0.887
        + 0.003766815 * max(0.0, Q.soft1_z - 0.00198) / 5.292589e-05   # +0.4%  soft1_z > 0.00198
        - 0.002991833 * max(0.0, Q.sum_z_dr2_top15 - 0.0182) / 0.0009843305   # -0.3%  sum_z_dr2_top15 > 0.0182
        + 0.00184556 * max(0.0, 13.2 - Q.n_pt_above_10) / 0.4476437   # +0.2%  n_pt_above_10 < 13.2
        + 0.001481144 * max(0.0, 5.53 - Q.n_dr_0p1_0p2) / 0.568992   # +0.1%  n_dr_0p1_0p2 < 5.53
        + 0.001206173 * max(0.0, Q.sum_z_dr2_top2 - 0.0142) / 0.001113351   # +0.1%  sum_z_dr2_top2 > 0.0142
        - 0.0006361432 * max(0.0, 1080.0 - Q.sum_pt) * max(0.0, 6.87e-08 - Q.e4) / 3.774782e-06   # -0.1%  sum_pt < 1080 and e4 < 6.87e-08
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 12.04;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.04197 * (0.1818639
        - 0.2620909 * max(0.0, 0.0975 - Q.sum_z_dr) / 0.03657116   # -26.2%  sum_z_dr < 0.0975
        + 0.1686834 * max(0.0, 0.335 - Q.LHA) / 0.0864375   # +16.9%  LHA < 0.335
        + 0.07184148 * max(0.0, 0.0494 - Q.sum_z_dr) / 0.008161445   # +7.2%  sum_z_dr < 0.0494
        - 0.05729443 * max(0.0, Q.sum_z_dr - 0.0926) / 0.009310904   # -5.7%  sum_z_dr > 0.0926
        - 0.05467624 * max(0.0, Q.sj2_dr - 0.143) / 0.06584099   # -5.5%  sj2_dr > 0.143
        + 0.05218339 * max(0.0, Q.LHA - 0.328) / 0.01492615   # +5.2%  LHA > 0.328
        - 0.0500508 * max(0.0, 0.228 - Q.LHA) / 0.02702737   # -5.0%  LHA < 0.228
        - 0.04415552 * max(0.0, Q.e2 - 0.0129) / 0.0188553   # -4.4%  e2 > 0.0129
        + 0.03730076 * max(0.0, Q.sj2_dr - 0.18) / 0.04237497   # +3.7%  sj2_dr > 0.18
        + 0.02095261 * max(0.0, 0.251 - Q.z_dr_0p1_0p2) / 0.1320999   # +2.1%  z_dr_0p1_0p2 < 0.251
        - 0.01866373 * max(0.0, 0.182 - Q.sj2_dr) / 0.03061963   # -1.9%  sj2_dr < 0.182
        + 0.01721156 * max(0.0, 0.00469 - Q.sum_z_dr2_top15) / 0.001345852   # +1.7%  sum_z_dr2_top15 < 0.00469
        - 0.015516 * max(0.0, 981.0 - Q.sum_pt_top15) / 128.8574   # -1.6%  sum_pt_top15 < 981
        - 0.01402259 * max(0.0, 0.000304 - Q.e3) * max(0.0, 7.01 - Q.log_sum_pt) / 1.748029e-05   # -1.4%  e3 < 0.000304 and log_sum_pt < 7.01
        + 0.01224652 * max(0.0, Q.psi_0p2 - 0.912) * max(0.0, 0.154 - Q.sj2_dr) / 0.001531383   # +1.2%  psi_0p2 > 0.912 and sj2_dr < 0.154
        - 0.01167036 * max(0.0, 0.00208 - Q.sum_z_dr2_top3) / 0.0007435668   # -1.2%  sum_z_dr2_top3 < 0.00208
        + 0.009732987 * max(0.0, Q.sum_z_dr2_top15 - 0.00384) / 0.004542805   # +1.0%  sum_z_dr2_top15 > 0.00384
        - 0.009136165 * max(0.0, Q.lam2 - 0.000547) / 0.0009822986   # -0.9%  lam2 > 0.000547
        - 0.008493499 * max(0.0, 0.00448 - Q.sum_z_dr2_top15) * max(0.0, Q.sum_pt_top40 - 892.0) / 0.2104496   # -0.8%  sum_z_dr2_top15 < 0.00448 and sum_pt_top40 > 892
        + 0.007219333 * max(0.0, 1060.0 - Q.sum_pt_top50) / 53.00916   # +0.7%  sum_pt_top50 < 1060
        + 0.007107344 * max(0.0, 0.176 - Q.dr_max_012) / 0.1043737   # +0.7%  dr_max_012 < 0.176
        - 0.006773892 * max(0.0, 0.324 - Q.LHA) * max(0.0, 0.98 - Q.psi_0p3) / 0.0001108302   # -0.7%  LHA < 0.324 and psi_0p3 < 0.98
        + 0.006380567 * max(0.0, 0.0802 - Q.sum_z_dr) * max(0.0, 1010.0 - Q.sum_pt) / 0.3729835   # +0.6%  sum_z_dr < 0.0802 and sum_pt < 1010
        + 0.006333368 * max(0.0, 0.0952 - Q.sum_z_dr) * max(0.0, 0.979 - Q.psi_0p3) / 3.9722e-05   # +0.6%  sum_z_dr < 0.0952 and psi_0p3 < 0.979
        - 0.005312918 * max(0.0, 0.105 - Q.sum_z_dr) * max(0.0, Q.eccentricity - 0.667) / 0.004998282   # -0.5%  sum_z_dr < 0.105 and eccentricity > 0.667
        - 0.00514392 * max(0.0, 0.0767 - Q.sum_z_dr) * max(0.0, 0.117 - Q.M2) / 0.0008964247   # -0.5%  sum_z_dr < 0.0767 and M2 < 0.117
        + 0.0033161 * max(0.0, Q.e2 - 0.0567) / 0.001010946   # +0.3%  e2 > 0.0567
        + 0.003106927 * max(0.0, Q.n_dr_0p1_0p2 - 13.5) * max(0.0, Q.psi_0p3 - 0.985) / 0.02413776   # +0.3%  n_dr_0p1_0p2 > 13.5 and psi_0p3 > 0.985
        - 0.002544125 * max(0.0, Q.n_dr_0p2_0p4 - 14.1) / 1.730864   # -0.3%  n_dr_0p2_0p4 > 14.1
        - 0.002051219 * max(0.0, Q.z_dr_0p05_0p1 - 0.763) / 0.009321026   # -0.2%  z_dr_0p05_0p1 > 0.763
        - 0.00199873 * max(0.0, 0.0548 - Q.tau1) * max(0.0, 1020.0 - Q.sum_pt) / 0.1683123   # -0.2%  tau1 < 0.0548 and sum_pt < 1020
        + 0.001654372 * max(0.0, Q.sum_pt_top15 - 989.0) / 15.32454   # +0.2%  sum_pt_top15 > 989
        + 0.001019556 * max(0.0, Q.e2 - 0.0456) * max(0.0, Q.psi_0p3 - 0.99) / 5.959932e-06   # +0.1%  e2 > 0.0456 and psi_0p3 > 0.99
        - 0.0009926177 * max(0.0, Q.n_dr_0p1_0p2 - 19.8) * max(0.0, Q.sum_pt - 967.0) / 112.7649   # -0.1%  n_dr_0p1_0p2 > 19.8 and sum_pt > 967
        - 0.0008907247 * max(0.0, Q.psi_0p2 - 0.997) / 0.0004395935   # -0.1%  psi_0p2 > 0.997
        + 0.000779645 * max(0.0, 0.000159 - Q.lam2) / 4.326481e-06   # +0.1%  lam2 < 0.000159
        + 0.0007457399 * max(0.0, Q.psi_0p2 - 0.877) * max(0.0, 0.972 - Q.z_top50_slots) / 6.108966e-05   # +0.1%  psi_0p2 > 0.877 and z_top50_slots < 0.972
        + 0.0007059952 * max(0.0, Q.zdr_0 - 0.02) / 0.0007523518   # +0.1%  zdr_0 > 0.02
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 16.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.58612 * (0.0195344
        - 0.1150679 * max(0.0, 0.328 - Q.LHA) / 0.08086992   # -11.5%  LHA < 0.328
        + 0.1102652 * max(0.0, Q.psi_0p3 - 0.997) / 0.001157514   # +11.0%  psi_0p3 > 0.997
        + 0.09718495 * max(0.0, 0.306 - Q.LHA) / 0.06473581   # +9.7%  LHA < 0.306
        - 0.08910424 * max(0.0, 0.0809 - Q.sum_z_dr) / 0.023837   # -8.9%  sum_z_dr < 0.0809
        + 0.06828904 * max(0.0, 20.9 - Q.n_dr_0p2_0p4) / 12.50166   # +6.8%  n_dr_0p2_0p4 < 20.9
        + 0.06127591 * max(0.0, 0.313 - Q.LHA) / 0.06961164   # +6.1%  LHA < 0.313
        - 0.04979277 * max(0.0, 0.239 - Q.tau21_b2) * max(0.0, 0.00798 - Q.sum_z_dr2_top15) / 8.767187e-05   # -5.0%  tau21_b2 < 0.239 and sum_z_dr2_top15 < 0.00798
        + 0.04930861 * max(0.0, 0.24 - Q.tau21_b2) * max(0.0, 0.00998 - Q.sum_z_dr2_top15) / 0.0001789581   # +4.9%  tau21_b2 < 0.24 and sum_z_dr2_top15 < 0.00998
        - 0.04288701 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 0.00703 - Q.mean_eta2) / 4.872119e-06   # -4.3%  psi_0p3 > 0.997 and mean_eta2 < 0.00703
        - 0.0399194 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 0.00697 - Q.mean_phi2) / 4.797885e-06   # -4.0%  psi_0p3 > 0.997 and mean_phi2 < 0.00697
        - 0.02903417 * max(0.0, 20.1 - Q.n_dr_0p2_0p4) * max(0.0, 8.95e-05 - Q.e3) / 0.0006189774   # -2.9%  n_dr_0p2_0p4 < 20.1 and e3 < 8.95e-05
        + 0.02753492 * max(0.0, 0.03 - Q.e2) / 0.006676865   # +2.8%  e2 < 0.03
        - 0.02684605 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 7.07 - Q.log_sum_pt) / 0.000144101   # -2.7%  psi_0p3 > 0.997 and log_sum_pt < 7.07
        + 0.02673731 * max(0.0, 0.253 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.153) / 0.003634987   # +2.7%  tau21_b2 < 0.253 and sj2_dr > 0.153
        - 0.01914906 * max(0.0, 0.247 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.184) / 0.002175402   # -1.9%  tau21_b2 < 0.247 and sj2_dr > 0.184
        + 0.01830107 * max(0.0, Q.sum_pt_top30 - 1010.0) / 32.92233   # +1.8%  sum_pt_top30 > 1010
        - 0.0175257 * max(0.0, Q.tau1 - 0.102) / 0.01490684   # -1.8%  tau1 > 0.102
        - 0.01703881 * max(0.0, 0.0465 - Q.sum_z_dr) / 0.007209381   # -1.7%  sum_z_dr < 0.0465
        - 0.0158028 * max(0.0, Q.psi_0p3 - 0.995) * max(0.0, Q.sum_z_dr2_top10 - 0.00872) / 1.783042e-06   # -1.6%  psi_0p3 > 0.995 and sum_z_dr2_top10 > 0.00872
        - 0.01541462 * max(0.0, Q.sum_pt - 1050.0) / 34.41033   # -1.5%  sum_pt > 1050
        - 0.01363906 * max(0.0, 0.0754 - Q.z_dr_0p2_0p4) / 0.04579335   # -1.4%  z_dr_0p2_0p4 < 0.0754
        - 0.01204168 * max(0.0, Q.sd_rg - 0.203) / 0.02078301   # -1.2%  sd_rg > 0.203
        + 0.008877677 * max(0.0, Q.sd_rg - 0.151) / 0.03968902   # +0.9%  sd_rg > 0.151
        - 0.008802904 * max(0.0, Q.z_top30_slots - 0.972) / 0.01081526   # -0.9%  z_top30_slots > 0.972
        - 0.007423384 * max(0.0, 998.0 - Q.sum_pt_top30) / 43.50713   # -0.7%  sum_pt_top30 < 998
        + 0.00615747 * max(0.0, 0.22 - Q.tau21_b2) * max(0.0, 0.00595 - Q.sum_z_dr2_top15) / 1.873918e-05   # +0.6%  tau21_b2 < 0.22 and sum_z_dr2_top15 < 0.00595
        + 0.003610487 * max(0.0, Q.sd_rg - 0.3) / 0.004502555   # +0.4%  sd_rg > 0.3
        - 0.001867196 * max(0.0, 18.2 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 21.8) / 6.617425   # -0.2%  n_dr_0p2_0p4 < 18.2 and n_dr_0p1_0p2 > 21.8
        - 0.0006239908 * max(0.0, 22.1 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_z_dr2_top5 - 0.0105) / 0.007187215   # -0.1%  n_dr_0p2_0p4 < 22.1 and sum_z_dr2_top5 > 0.0105
        - 0.0004766336 * max(0.0, Q.n_dr_0p1_0p2 - 24.9) * max(0.0, Q.pt_2 - 78.4) / 4.227542   # -0.0%  n_dr_0p1_0p2 > 24.9 and pt_2 > 78.4
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 8.564;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.563658 * (0.4016975
        - 0.1551762 * max(0.0, Q.sum_z_dr - 0.0945) / 0.008918633   # -15.5%  sum_z_dr > 0.0945
        - 0.1451141 * max(0.0, 0.0112 - Q.mean_phi2) / 0.007267294   # -14.5%  mean_phi2 < 0.0112
        - 0.1235805 * max(0.0, 19.4 - Q.n_dr_0p2_0p4) / 11.1753   # -12.4%  n_dr_0p2_0p4 < 19.4
        - 0.1145203 * max(0.0, 7.03 - Q.log_sum_pt) / 0.09997075   # -11.5%  log_sum_pt < 7.03
        - 0.09483239 * max(0.0, Q.z_top50_slots - 0.968) / 0.02553812   # -9.5%  z_top50_slots > 0.968
        + 0.08730065 * max(0.0, 1040.0 - Q.sum_pt) / 33.98241   # +8.7%  sum_pt < 1040
        + 0.08500025 * max(0.0, Q.LHA - 0.33) / 0.01450026   # +8.5%  LHA > 0.33
        + 0.08475507 * max(0.0, Q.mean_eta2 - 0.000529) / 0.004147505   # +8.5%  mean_eta2 > 0.000529
        + 0.05711131 * max(0.0, Q.sum_z_dr - 0.0832) * max(0.0, 1150.0 - Q.sum_pt) / 1.740505   # +5.7%  sum_z_dr > 0.0832 and sum_pt < 1150
        + 0.01418488 * max(0.0, Q.mean_phi2 - 0.0115) / 0.0006901956   # +1.4%  mean_phi2 > 0.0115
        + 0.01328737 * max(0.0, 14.4 - Q.n_dr_0p2_0p4) * max(0.0, 3.5e-05 - Q.e3) / 7.388865e-05   # +1.3%  n_dr_0p2_0p4 < 14.4 and e3 < 3.5e-05
        - 0.01321057 * max(0.0, Q.sum_z_dr - 0.0907) * max(0.0, 1000.0 - Q.sum_pt_top50) / 0.4237108   # -1.3%  sum_z_dr > 0.0907 and sum_pt_top50 < 1000
        + 0.01192636 * max(0.0, 21.6 - Q.n_dr_0p2_0p4) * max(0.0, 0.784 - Q.psi_0p1) / 0.6079362   # +1.2%  n_dr_0p2_0p4 < 21.6 and psi_0p1 < 0.784
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 3.271;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.271407 * (0.0004065529
        + 0.1457524 * max(0.0, 0.00576 - Q.sum_z_dr2_top15) * max(0.0, 0.0588 - Q.z_dr_0p2_0p4) / 7.829484e-05   # +14.6%  sum_z_dr2_top15 < 0.00576 and z_dr_0p2_0p4 < 0.0588
        + 0.1170651 * max(0.0, 0.0611 - Q.sum_z_dr) / 0.01268104   # +11.7%  sum_z_dr < 0.0611
        - 0.1094141 * max(0.0, Q.sum_z_dr - 0.133) / 0.002651392   # -10.9%  sum_z_dr > 0.133
        + 0.08778342 * max(0.0, Q.LHA - 0.389) / 0.004842754   # +8.8%  LHA > 0.389
        - 0.08307946 * max(0.0, Q.z_top50_slots - 0.972) / 0.02191829   # -8.3%  z_top50_slots > 0.972
        + 0.08180458 * max(0.0, 0.00763 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.984) / 3.475534e-05   # +8.2%  sum_z_dr2_top15 < 0.00763 and psi_0p3 > 0.984
        + 0.04904916 * max(0.0, Q.sum_z_dr - 0.084) / 0.01129998   # +4.9%  sum_z_dr > 0.084
        + 0.04713538 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # +4.7%  sum_pt < 1000
        - 0.04173179 * max(0.0, Q.sum_pt - 1010.0) / 52.30716   # -4.2%  sum_pt > 1010
        - 0.03916486 * max(0.0, Q.z_top40_slots - 0.98) * max(0.0, 0.0153 - Q.C3) / 9.780473e-05   # -3.9%  z_top40_slots > 0.98 and C3 < 0.0153
        + 0.03588821 * max(0.0, Q.psi_0p1 - 0.89) / 0.02782108   # +3.6%  psi_0p1 > 0.89
        + 0.02899136 * max(0.0, Q.z_dr_0p1_0p2 - 0.448) * max(0.0, Q.soft5_z - 0.000325) / 2.765089e-05   # +2.9%  z_dr_0p1_0p2 > 0.448 and soft5_z > 0.000325
        - 0.02555395 * max(0.0, Q.z_dr_0p1_0p2 - 0.458) * max(0.0, Q.soft5_pt - 0.392) / 0.02533254   # -2.6%  z_dr_0p1_0p2 > 0.458 and soft5_pt > 0.392
        + 0.02434036 * max(0.0, Q.psi_0p2 - 0.989) / 0.003304035   # +2.4%  psi_0p2 > 0.989
        - 0.02305751 * max(0.0, 0.0462 - Q.sum_z_dr) * max(0.0, 1050.0 - Q.sum_pt) / 0.2674841   # -2.3%  sum_z_dr < 0.0462 and sum_pt < 1050
        - 0.02205584 * max(0.0, 0.000849 - Q.sum_z_dr2_top15) / 0.0001026367   # -2.2%  sum_z_dr2_top15 < 0.000849
        + 0.01524958 * max(0.0, 0.00378 - Q.sum_z_dr2_top15) * max(0.0, Q.n_dr_0p2_0p4 - 4.04) / 0.003303815   # +1.5%  sum_z_dr2_top15 < 0.00378 and n_dr_0p2_0p4 > 4.04
        - 0.009835465 * max(0.0, 1000.0 - Q.sum_pt_top50) * max(0.0, Q.sum_pt_top20 - 736.0) / 893.7725   # -1.0%  sum_pt_top50 < 1000 and sum_pt_top20 > 736
        - 0.008386399 * max(0.0, Q.sum_z_dr - 0.123) * max(0.0, Q.soft6_pt - 1.51) / 0.002406608   # -0.8%  sum_z_dr > 0.123 and soft6_pt > 1.51
        + 0.004661086 * max(0.0, 1010.0 - Q.sum_pt_top50) * max(0.0, Q.e3 - 0.000128) / 0.002335116   # +0.5%  sum_pt_top50 < 1010 and e3 > 0.000128
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 10.38;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.37526 * (0.4423984
        - 0.3108958 * max(0.0, 0.119 - Q.sum_z_dr) / 0.05430346   # -31.1%  sum_z_dr < 0.119
        + 0.1055413 * max(0.0, 0.317 - Q.LHA) / 0.07251783   # +10.6%  LHA < 0.317
        - 0.08111529 * max(0.0, Q.sum_z_dr - 0.102) / 0.00744772   # -8.1%  sum_z_dr > 0.102
        - 0.06134475 * max(0.0, 0.0416 - Q.e2) / 0.01392709   # -6.1%  e2 < 0.0416
        + 0.04928113 * max(0.0, 0.103 - Q.sum_z_dr) * max(0.0, Q.psi_0p2 - 0.955) / 0.001186322   # +4.9%  sum_z_dr < 0.103 and psi_0p2 > 0.955
        - 0.04768079 * max(0.0, 0.000361 - Q.e3) / 0.0002779217   # -4.8%  e3 < 0.000361
        + 0.04392508 * Q.n_real_top50 / 41.43038   # +4.4%  n_real_top50
        + 0.03656495 * max(0.0, Q.sum_z_dr - 0.105) * max(0.0, 1200.0 - Q.sum_pt_top50) / 1.535915   # +3.7%  sum_z_dr > 0.105 and sum_pt_top50 < 1200
        - 0.03081739 * max(0.0, Q.sum_pt_top2 - 184.0) / 189.1944   # -3.1%  sum_pt_top2 > 184
        + 0.02376922 * max(0.0, Q.LHA - 0.351) / 0.01053897   # +2.4%  LHA > 0.351
        - 0.02085981 * max(0.0, 1070.0 - Q.sum_pt_top40) / 70.95933   # -2.1%  sum_pt_top40 < 1070
        + 0.02083692 * max(0.0, 0.0667 - Q.C2) / 0.01578019   # +2.1%  C2 < 0.0667
        - 0.02020096 * max(0.0, Q.sum_z_dr - 0.146) / 0.001352195   # -2.0%  sum_z_dr > 0.146
        - 0.01897902 * max(0.0, 10.2 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_real_top40 - 22.0) / 48.26283   # -1.9%  n_dr_0p2_0p4 < 10.2 and n_real_top40 > 22
        - 0.01830287 * max(0.0, 0.0415 - Q.tau1) / 0.004457679   # -1.8%  tau1 < 0.0415
        - 0.01419723 * max(0.0, 0.257 - Q.tau21_b2) / 0.06241527   # -1.4%  tau21_b2 < 0.257
        + 0.01383476 * max(0.0, Q.LHA - 0.414) / 0.002271191   # +1.4%  LHA > 0.414
        - 0.01298104 * max(0.0, 989.0 - Q.sum_pt) / 12.58708   # -1.3%  sum_pt < 989
        + 0.01200081 * max(0.0, 1090.0 - Q.sum_pt_top40) * max(0.0, Q.n_dr_0_0p05 - -0.0017) / 972.7466   # +1.2%  sum_pt_top40 < 1090 and n_dr_0_0p05 > -0.0017
        + 0.01194934 * max(0.0, 0.125 - Q.sum_z_dr) * max(0.0, 1000.0 - Q.sum_pt_top50) / 0.8156419   # +1.2%  sum_z_dr < 0.125 and sum_pt_top50 < 1000
        - 0.01036964 * max(0.0, Q.psi_0p3 - 0.998) * max(0.0, Q.eccentricity - 0.474) / 0.000247898   # -1.0%  psi_0p3 > 0.998 and eccentricity > 0.474
        + 0.00857968 * max(0.0, Q.e2 - 0.0473) / 0.002144974   # +0.9%  e2 > 0.0473
        - 0.004852707 * max(0.0, 0.000395 - Q.e3) * max(0.0, 0.832 - Q.tau32) / 2.678091e-05   # -0.5%  e3 < 0.000395 and tau32 < 0.832
        - 0.004078172 * max(0.0, Q.LHA - 0.153) * max(0.0, 0.997 - Q.psi_0p3) / 0.00208434   # -0.4%  LHA > 0.153 and psi_0p3 < 0.997
        - 0.003974354 * max(0.0, Q.e2 - 0.027) * max(0.0, Q.soft1_pt - 1.11) / 0.00162984   # -0.4%  e2 > 0.027 and soft1_pt > 1.11
        + 0.003450133 * max(0.0, Q.n_dr_0p1_0p2 - 21.1) / 1.371496   # +0.3%  n_dr_0p1_0p2 > 21.1
        + 0.002934917 * max(0.0, 0.135 - Q.sum_z_dr) * max(0.0, Q.pt1_dr01 - 7.58) / 0.1136214   # +0.3%  sum_z_dr < 0.135 and pt1_dr01 > 7.58
        - 0.00261948 * max(0.0, 0.018 - Q.sum_z_dr) / 0.0008112774   # -0.3%  sum_z_dr < 0.018
        - 0.002552627 * max(0.0, Q.e2 - 0.0266) * max(0.0, Q.log_sum_pt - 6.94) / 0.0001777462   # -0.3%  e2 > 0.0266 and log_sum_pt > 6.94
        - 0.001509797 * max(0.0, 948.0 - Q.sum_pt_top50) * max(0.0, 0.175 - Q.z_dr_0p05_0p1) / 0.5439076   # -0.2%  sum_pt_top50 < 948 and z_dr_0p05_0p1 < 0.175
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 17.35;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.35241 * (-0.06742578
        + 0.230814 * max(0.0, 0.0973 - Q.sum_z_dr) / 0.03641073   # +23.1%  sum_z_dr < 0.0973
        - 0.1638663 * max(0.0, 0.338 - Q.LHA) / 0.08885863   # -16.4%  LHA < 0.338
        + 0.09997792 * max(0.0, 0.00801 - Q.sum_z_dr2_top15) / 0.003355625   # +10.0%  sum_z_dr2_top15 < 0.00801
        - 0.07940158 * max(0.0, 0.0684 - Q.sum_z_dr) / 0.01617147   # -7.9%  sum_z_dr < 0.0684
        + 0.07753493 * max(0.0, 0.28 - Q.LHA) / 0.04928272   # +7.8%  LHA < 0.28
        - 0.05602939 * max(0.0, 0.00613 - Q.sum_z_dr2_top15) / 0.002073017   # -5.6%  sum_z_dr2_top15 < 0.00613
        - 0.04221796 * max(0.0, 0.00823 - Q.sum_z_dr2_top10) / 0.003896721   # -4.2%  sum_z_dr2_top10 < 0.00823
        - 0.03234068 * max(0.0, Q.psi_0p1 - 0.647) / 0.1839964   # -3.2%  psi_0p1 > 0.647
        + 0.02726337 * max(0.0, 0.00497 - Q.sum_z_dr2_top10) / 0.001732913   # +2.7%  sum_z_dr2_top10 < 0.00497
        + 0.02299596 * max(0.0, Q.sum_z_dr2_top15 - 0.00102) / 0.006498948   # +2.3%  sum_z_dr2_top15 > 0.00102
        + 0.02232421 * max(0.0, 16.5 - Q.n_dr_0p2_0p4) / 8.705146   # +2.2%  n_dr_0p2_0p4 < 16.5
        + 0.02059368 * max(0.0, 1.84 - Q.D2) / 0.3080604   # +2.1%  D2 < 1.84
        - 0.0185388 * Q.n_dr_0p1_0p2 / 12.56613   # -1.9%  n_dr_0p1_0p2
        + 0.0147353 * max(0.0, 9.25 - Q.n_dr_0p2_0p4) * max(0.0, 33.1 - Q.n_dr_0p1_0p2) / 77.24864   # +1.5%  n_dr_0p2_0p4 < 9.25 and n_dr_0p1_0p2 < 33.1
        + 0.01201617 * max(0.0, 0.0983 - Q.z_dr_0p2_0p4) / 0.06395997   # +1.2%  z_dr_0p2_0p4 < 0.0983
        - 0.01200875 * max(0.0, Q.log_sum_pt - 6.86) / 0.09302715   # -1.2%  log_sum_pt > 6.86
        - 0.01117283 * max(0.0, 9.76 - Q.n_dr_0p2_0p4) * max(0.0, 0.0051 - Q.sum_z_dr2_top10) / 0.006294659   # -1.1%  n_dr_0p2_0p4 < 9.76 and sum_z_dr2_top10 < 0.0051
        - 0.009259614 * max(0.0, Q.pt_entropy - 2.25) / 0.6505128   # -0.9%  pt_entropy > 2.25
        - 0.00900694 * max(0.0, 1.83 - Q.D2) * max(0.0, Q.n_real_top50 - 19.0) / 5.724987   # -0.9%  D2 < 1.83 and n_real_top50 > 19
        + 0.006634659 * max(0.0, 19.5 - Q.n_dr_0p1_0p2) * max(0.0, 0.668 - Q.planar_flow) / 1.951311   # +0.7%  n_dr_0p1_0p2 < 19.5 and planar_flow < 0.668
        - 0.005095462 * max(0.0, 0.128 - Q.z_dr_0_0p05) / 0.0313541   # -0.5%  z_dr_0_0p05 < 0.128
        + 0.003964213 * max(0.0, Q.log_sum_pt - 6.88) * max(0.0, Q.M2 - 0.0374) / 0.002347736   # +0.4%  log_sum_pt > 6.88 and M2 > 0.0374
        - 0.003649212 * max(0.0, Q.sum_z_dr - 0.104) / 0.007075154   # -0.4%  sum_z_dr > 0.104
        - 0.003626326 * max(0.0, 9.23 - Q.n_dr_0p1_0p2) / 1.823928   # -0.4%  n_dr_0p1_0p2 < 9.23
        - 0.003296955 * max(0.0, 1.85 - Q.D2) * max(0.0, 0.0485 - Q.z_7) / 0.004028882   # -0.3%  D2 < 1.85 and z_7 < 0.0485
        - 0.002893937 * max(0.0, 0.0759 - Q.sum_z_dr) * max(0.0, 3.87 - Q.D2) / 0.0129092   # -0.3%  sum_z_dr < 0.0759 and D2 < 3.87
        + 0.0025022 * max(0.0, 0.000934 - Q.sum_z_dr2_top10) / 0.0001736769   # +0.3%  sum_z_dr2_top10 < 0.000934
        + 0.002283155 * max(0.0, Q.z_dr_0p05_0p1 - 0.691) / 0.01617071   # +0.2%  z_dr_0p05_0p1 > 0.691
        + 0.002168768 * max(0.0, Q.psi_0p2 - 0.996) / 0.0006905202   # +0.2%  psi_0p2 > 0.996
        - 0.001786697 * max(0.0, 0.19 - Q.N2) / 0.007693178   # -0.2%  N2 < 0.19
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 5.785;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.785048 * (-0.1884168
        + 0.195109 * max(0.0, Q.sum_z_dr - 0.0882) / 0.01026104   # +19.5%  sum_z_dr > 0.0882
        + 0.1609895 * max(0.0, 0.00613 - Q.mean_eta2) / 0.002966026   # +16.1%  mean_eta2 < 0.00613
        - 0.1277146 * max(0.0, Q.log_sum_pt - 6.8) * max(0.0, 0.0115 - Q.mean_eta2) / 0.001178365   # -12.8%  log_sum_pt > 6.8 and mean_eta2 < 0.0115
        - 0.09487133 * max(0.0, Q.LHA - 0.313) / 0.01866786   # -9.5%  LHA > 0.313
        - 0.06374564 * max(0.0, Q.mean_phi2 - 0.00279) / 0.002691764   # -6.4%  mean_phi2 > 0.00279
        + 0.05626317 * max(0.0, 0.0048 - Q.mean_phi2) / 0.002009168   # +5.6%  mean_phi2 < 0.0048
        + 0.04877415 * max(0.0, Q.sum_pt - 917.0) / 131.2376   # +4.9%  sum_pt > 917
        + 0.04333522 * max(0.0, 27.6 - Q.n_dr_0p1_0p2) / 15.66852   # +4.3%  n_dr_0p1_0p2 < 27.6
        - 0.03937215 * max(0.0, Q.mean_eta2 - 0.0062) / 0.001469482   # -3.9%  mean_eta2 > 0.0062
        + 0.02773275 * max(0.0, Q.psi_0p3 - 0.988) * max(0.0, 0.00578 - Q.sum_z_dr2_top15) / 1.419781e-05   # +2.8%  psi_0p3 > 0.988 and sum_z_dr2_top15 < 0.00578
        + 0.0256426 * max(0.0, 14.3 - Q.n_dr_0p2_0p4) / 6.931946   # +2.6%  n_dr_0p2_0p4 < 14.3
        + 0.01414468 * max(0.0, 0.191 - Q.sj3_dr_max) / 0.01802371   # +1.4%  sj3_dr_max < 0.191
        + 0.01407554 * max(0.0, Q.sum_z_dr2_top15 - 0.00441) / 0.004197304   # +1.4%  sum_z_dr2_top15 > 0.00441
        + 0.01316391 * max(0.0, 0.0388 - Q.sum_z_dr) * max(0.0, Q.log_sum_pt - 6.81) / 0.0007539986   # +1.3%  sum_z_dr < 0.0388 and log_sum_pt > 6.81
        - 0.01262223 * max(0.0, Q.lam2 - 0.000534) / 0.0009894341   # -1.3%  lam2 > 0.000534
        + 0.01092892 * max(0.0, 0.00884 - Q.sum_z_dr2_top15) / 0.004001541   # +1.1%  sum_z_dr2_top15 < 0.00884
        - 0.01068571 * max(0.0, 10.6 - Q.n_dr_0p1_0p2) / 2.472693   # -1.1%  n_dr_0p1_0p2 < 10.6
        + 0.009144621 * max(0.0, Q.psi_0p3 - 0.98) * max(0.0, Q.z_1st - 0.137) / 0.001477712   # +0.9%  psi_0p3 > 0.98 and z_1st > 0.137
        - 0.00909944 * max(0.0, 0.159 - Q.sj3_dr_max) / 0.009501931   # -0.9%  sj3_dr_max < 0.159
        - 0.005849333 * max(0.0, 0.0255 - Q.sum_z_dr) / 0.001967365   # -0.6%  sum_z_dr < 0.0255
        - 0.005368825 * max(0.0, 0.994 - Q.z_top50_slots) / 0.005678046   # -0.5%  z_top50_slots < 0.994
        - 0.0042359 * max(0.0, Q.n_dr_0_0p05 - 17.5) / 1.929518   # -0.4%  n_dr_0_0p05 > 17.5
        - 0.003652017 * max(0.0, Q.sum_z_dr2_top15 - 0.023) / 0.0004779886   # -0.4%  sum_z_dr2_top15 > 0.023
        + 0.003478789 * max(0.0, 0.00454 - Q.sum_z_dr2_top15) / 0.001281845   # +0.3%  sum_z_dr2_top15 < 0.00454
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 11.08;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.08462 * (0.2147118
        + 0.139887 * max(0.0, Q.sum_z_dr - 0.0364) / 0.03718452   # +14.0%  sum_z_dr > 0.0364
        - 0.1068667 * max(0.0, Q.LHA - 0.19) / 0.08774644   # -10.7%  LHA > 0.19
        - 0.08814331 * max(0.0, Q.sum_z_dr - 0.0922) / 0.009394572   # -8.8%  sum_z_dr > 0.0922
        + 0.06695114 * max(0.0, 1060.0 - Q.sum_pt_top50) / 53.00916   # +6.7%  sum_pt_top50 < 1060
        - 0.06378495 * max(0.0, 6.96 - Q.log_sum_pt) / 0.04364396   # -6.4%  log_sum_pt < 6.96
        - 0.05718288 * max(0.0, 1120.0 - Q.sum_pt) / 96.03799   # -5.7%  sum_pt < 1120
        + 0.05055408 * max(0.0, Q.LHA - 0.327) / 0.01514522   # +5.1%  LHA > 0.327
        + 0.03937763 * max(0.0, Q.sum_z_dr2_top15 - 0.00567) * max(0.0, 1290.0 - Q.sum_pt_top40) / 1.170204   # +3.9%  sum_z_dr2_top15 > 0.00567 and sum_pt_top40 < 1290
        + 0.03588377 * max(0.0, Q.sum_z_dr - 0.0884) * max(0.0, 1240.0 - Q.sum_pt) / 2.425354   # +3.6%  sum_z_dr > 0.0884 and sum_pt < 1240
        - 0.03381845 * max(0.0, 0.0855 - Q.sum_z_dr) / 0.02716411   # -3.4%  sum_z_dr < 0.0855
        - 0.03275637 * max(0.0, Q.sum_z_dr2_top15 - 0.00774) / 0.002793015   # -3.3%  sum_z_dr2_top15 > 0.00774
        - 0.02395662 * max(0.0, 956.0 - Q.sum_pt) / 7.544039   # -2.4%  sum_pt < 956
        - 0.02147049 * max(0.0, Q.sum_z_dr - 0.135) / 0.002421082   # -2.1%  sum_z_dr > 0.135
        + 0.02117258 * max(0.0, 0.00531 - Q.mean_phi2) / 0.002361068   # +2.1%  mean_phi2 < 0.00531
        + 0.02081412 * max(0.0, 6.86 - Q.log_sum_pt) / 0.008420316   # +2.1%  log_sum_pt < 6.86
        + 0.02011904 * max(0.0, 0.00534 - Q.mean_eta2) / 0.002380064   # +2.0%  mean_eta2 < 0.00534
        - 0.01990308 * max(0.0, 6.91 - Q.log_sum_pt) / 0.01723579   # -2.0%  log_sum_pt < 6.91
        - 0.01909275 * max(0.0, Q.sum_z_dr - 0.116) / 0.005015069   # -1.9%  sum_z_dr > 0.116
        + 0.0162159 * max(0.0, Q.LHA - 0.398) / 0.003800151   # +1.6%  LHA > 0.398
        - 0.0158502 * max(0.0, Q.psi_0p2 - 0.905) / 0.06121725   # -1.6%  psi_0p2 > 0.905
        + 0.01430236 * max(0.0, Q.n_particles - 29.5) / 17.19482   # +1.4%  n_particles > 29.5
        - 0.01376718 * max(0.0, Q.mean_eta2 - 0.00575) / 0.001578118   # -1.4%  mean_eta2 > 0.00575
        - 0.01327746 * max(0.0, Q.mean_phi2 - 0.00567) / 0.001606722   # -1.3%  mean_phi2 > 0.00567
        - 0.007500099 * max(0.0, 22.1 - Q.n_pt_above_5) / 1.571565   # -0.8%  n_pt_above_5 < 22.1
        + 0.007111278 * max(0.0, 1020.0 - Q.sum_pt_top50) * max(0.0, 0.000834 - Q.sum_z_dr2_top2) / 0.004988977   # +0.7%  sum_pt_top50 < 1020 and sum_z_dr2_top2 < 0.000834
        - 0.006753416 * max(0.0, 1060.0 - Q.sum_pt) * max(0.0, 0.743 - Q.planar_flow) / 12.07405   # -0.7%  sum_pt < 1060 and planar_flow < 0.743
        - 0.006122419 * max(0.0, Q.log_sum_pt - 7.11) / 0.007113701   # -0.6%  log_sum_pt > 7.11
        + 0.00526369 * max(0.0, Q.log_sum_pt - 7.07) * max(0.0, Q.psi_0p3 - 0.955) / 0.000376426   # +0.5%  log_sum_pt > 7.07 and psi_0p3 > 0.955
        - 0.005145 * max(0.0, Q.n_dr_0p05_0p1 - 8.29) / 5.002666   # -0.5%  n_dr_0p05_0p1 > 8.29
        + 0.005075789 * max(0.0, Q.e2 - 0.052) / 0.001512452   # +0.5%  e2 > 0.052
        + 0.004432771 * max(0.0, Q.sum_z_dr2_top5 - 0.00648) / 0.002519774   # +0.4%  sum_z_dr2_top5 > 0.00648
        + 0.004208542 * max(0.0, 6.96 - Q.log_sum_pt) * max(0.0, 0.00981 - Q.C3) / 0.0001815179   # +0.4%  log_sum_pt < 6.96 and C3 < 0.00981
        - 0.004070724 * max(0.0, Q.sum_z_dr2_top15 - 0.00476) * max(0.0, Q.soft6_pt - 1.17) / 0.003090579   # -0.4%  sum_z_dr2_top15 > 0.00476 and soft6_pt > 1.17
        - 0.003933971 * max(0.0, Q.C2 - 0.103) / 0.005172787   # -0.4%  C2 > 0.103
        - 0.00292772 * max(0.0, Q.sum_z_dr - 0.146) / 0.001352195   # -0.3%  sum_z_dr > 0.146
        + 0.002306571 * max(0.0, 22.5 - Q.n_pt_above_5) * max(0.0, 3.38 - Q.D2) / 1.223324   # +0.2%  n_pt_above_5 < 22.5 and D2 < 3.38
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 24.76;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.75821 * (0.189836
        - 0.173286 * max(0.0, 0.334 - Q.LHA) / 0.08563376   # -17.3%  LHA < 0.334
        - 0.1462608 * max(0.0, Q.LHA - 0.112) / 0.1521493   # -14.6%  LHA > 0.112
        + 0.07818016 * max(0.0, 0.306 - Q.LHA) / 0.06473581   # +7.8%  LHA < 0.306
        - 0.07069832 * max(0.0, 0.0668 - Q.sum_z_dr) / 0.01535407   # -7.1%  sum_z_dr < 0.0668
        + 0.05328835 * max(0.0, 0.0117 - Q.mean_eta2) / 0.007715345   # +5.3%  mean_eta2 < 0.0117
        - 0.04920806 * max(0.0, Q.sum_z_dr - 0.0813) / 0.01206241   # -4.9%  sum_z_dr > 0.0813
        + 0.04226564 * max(0.0, Q.tau1 - 0.0837) / 0.0220764   # +4.2%  tau1 > 0.0837
        + 0.03418783 * max(0.0, Q.sum_z_dr - 0.0984) / 0.008138745   # +3.4%  sum_z_dr > 0.0984
        + 0.03107491 * max(0.0, 0.271 - Q.LHA) / 0.04473018   # +3.1%  LHA < 0.271
        + 0.03061971 * max(0.0, Q.sj3_dr_max - 0.172) / 0.08929201   # +3.1%  sj3_dr_max > 0.172
        - 0.02632416 * max(0.0, Q.sj3_dr_max - 0.201) / 0.06985411   # -2.6%  sj3_dr_max > 0.201
        - 0.02292131 * max(0.0, 7.04 - Q.log_sum_pt) / 0.1087147   # -2.3%  log_sum_pt < 7.04
        + 0.01938037 * max(0.0, 0.00569 - Q.mean_phi2) / 0.002636391   # +1.9%  mean_phi2 < 0.00569
        + 0.01568042 * max(0.0, 0.0352 - Q.e2) / 0.009515173   # +1.6%  e2 < 0.0352
        + 0.01414515 * max(0.0, Q.sd_rg - 0.158) / 0.03629105   # +1.4%  sd_rg > 0.158
        + 0.01185107 * max(0.0, 1050.0 - Q.sum_pt_top40) / 55.46527   # +1.2%  sum_pt_top40 < 1050
        - 0.01126204 * max(0.0, Q.psi_0p1 - 0.811) / 0.06718747   # -1.1%  psi_0p1 > 0.811
        - 0.01105427 * max(0.0, Q.mean_phi2 - 0.00605) / 0.001512066   # -1.1%  mean_phi2 > 0.00605
        - 0.01054764 * max(0.0, 8.68e-05 - Q.e3) / 4.205163e-05   # -1.1%  e3 < 8.68e-05
        + 0.01053176 * max(0.0, 0.15 - Q.z_dr_0p1_0p2) / 0.06422353   # +1.1%  z_dr_0p1_0p2 < 0.15
        - 0.01015371 * max(0.0, 0.333 - Q.tau21_b2) * max(0.0, 0.00729 - Q.sum_z_dr2_top15) / 0.0001323094   # -1.0%  tau21_b2 < 0.333 and sum_z_dr2_top15 < 0.00729
        + 0.009993923 * max(0.0, 0.336 - Q.tau21_b2) * max(0.0, 0.0196 - Q.sum_z_dr2_top10) / 0.001295454   # +1.0%  tau21_b2 < 0.336 and sum_z_dr2_top10 < 0.0196
        + 0.008677586 * max(0.0, Q.e2 - 0.0266) / 0.009422872   # +0.9%  e2 > 0.0266
        + 0.008520045 * max(0.0, Q.psi_0p3 - 0.993) / 0.003630655   # +0.9%  psi_0p3 > 0.993
        - 0.007729882 * max(0.0, 0.316 - Q.sj2_dr) / 0.1267404   # -0.8%  sj2_dr < 0.316
        - 0.007318316 * max(0.0, 2.21 - Q.D2) / 0.4806058   # -0.7%  D2 < 2.21
        - 0.007265505 * max(0.0, 3.56 - Q.D2) / 1.332451   # -0.7%  D2 < 3.56
        - 0.006681938 * max(0.0, Q.sd_rg - 0.203) / 0.02078301   # -0.7%  sd_rg > 0.203
        - 0.005599683 * max(0.0, Q.z_dr_0p2_0p4 - 0.071) / 0.02906459   # -0.6%  z_dr_0p2_0p4 > 0.071
        + 0.005347976 * max(0.0, 0.158 - Q.sd_rg) / 0.05404339   # +0.5%  sd_rg < 0.158
        - 0.005011769 * max(0.0, 1.42 - Q.soft1_pt) / 0.8616834   # -0.5%  soft1_pt < 1.42
        - 0.004769694 * max(0.0, Q.mean_eta2 - 0.0108) / 0.000761865   # -0.5%  mean_eta2 > 0.0108
        - 0.004164674 * max(0.0, 0.0402 - Q.sum_z_dr) / 0.00534248   # -0.4%  sum_z_dr < 0.0402
        + 0.003854145 * max(0.0, 0.341 - Q.tau21_b2) * max(0.0, 0.00601 - Q.sum_z_dr2_top15) / 6.491274e-05   # +0.4%  tau21_b2 < 0.341 and sum_z_dr2_top15 < 0.00601
        + 0.003760532 * max(0.0, Q.z_dr_0p2_0p4 - 0.156) / 0.01452481   # +0.4%  z_dr_0p2_0p4 > 0.156
        - 0.003484651 * max(0.0, Q.n_dr_0p2_0p4 - 12.4) / 2.146112   # -0.3%  n_dr_0p2_0p4 > 12.4
        + 0.003462482 * max(0.0, 0.352 - Q.tau21_b2) * max(0.0, 2.34 - Q.soft1_pt) / 0.2036219   # +0.3%  tau21_b2 < 0.352 and soft1_pt < 2.34
        - 0.003242329 * max(0.0, 1.15 - Q.D2) / 0.08267174   # -0.3%  D2 < 1.15
        + 0.003116189 * max(0.0, 1.5 - Q.D2_b2) / 0.4408643   # +0.3%  D2_b2 < 1.5
        - 0.003087151 * max(0.0, 0.984 - Q.z_top40_slots) / 0.0136243   # -0.3%  z_top40_slots < 0.984
        + 0.00305455 * max(0.0, Q.tau21_b2 - 0.173) / 0.1813553   # +0.3%  tau21_b2 > 0.173
        + 0.003052087 * max(0.0, 0.219 - Q.N2) / 0.01436582   # +0.3%  N2 < 0.219
        + 0.002775063 * max(0.0, 11.9 - Q.n_dr_0p05_0p1) / 3.578416   # +0.3%  n_dr_0p05_0p1 < 11.9
        - 0.002767263 * max(0.0, Q.psi_0p3 - 0.993) * max(0.0, 0.055 - Q.dr_6) / 5.229959e-05   # -0.3%  psi_0p3 > 0.993 and dr_6 < 0.055
        + 0.002747868 * max(0.0, 0.0567 - Q.dr_6) / 0.0158954   # +0.3%  dr_6 < 0.0567
        - 0.001853989 * max(0.0, Q.sd_rg - 0.15) * max(0.0, Q.n_pt_above_50 - 4.09) / 0.04026443   # -0.2%  sd_rg > 0.15 and n_pt_above_50 > 4.09
        - 0.001545392 * max(0.0, Q.sum_z_dr2_top10 - -0.000483) * max(0.0, 36.5 - Q.n_real_top40) / 0.008446167   # -0.2%  sum_z_dr2_top10 > -0.000483 and n_real_top40 < 36.5
        + 0.001253366 * max(0.0, 0.0701 - Q.C2) * max(0.0, Q.pt_3 - 56.1) / 0.3988574   # +0.1%  C2 < 0.0701 and pt_3 > 56.1
        - 0.001235334 * max(0.0, 888.0 - Q.sum_pt_top15) / 68.11728   # -0.1%  sum_pt_top15 < 888
        - 0.0008663265 * max(0.0, 0.41 - Q.tau21_b2) * max(0.0, Q.n_pt_above_10 - 21.1) / 0.2356999   # -0.1%  tau21_b2 < 0.41 and n_pt_above_10 > 21.1
        - 0.000604545 * max(0.0, Q.n_dr_0p1_0p2 - 24.8) / 0.8962545   # -0.1%  n_dr_0p1_0p2 > 24.8
        + 0.0001460718 * max(0.0, Q.sum_z_dr - 0.0829) * max(0.0, 37.7 - Q.n_real_top40) / 0.001375086   # +0.0%  sum_z_dr > 0.0829 and n_real_top40 < 37.7
        + 8.799335e-05 * max(0.0, Q.sum_pt_top30 - 1070.0) / 17.71185   # +0.0%  sum_pt_top30 > 1070
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 7.586;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.585908 * (0.08463061
        + 0.1650375 * max(0.0, 0.0435 - Q.e2) / 0.01541822   # +16.5%  e2 < 0.0435
        - 0.09331148 * max(0.0, 0.000318 - Q.e3) / 0.0002391393   # -9.3%  e3 < 0.000318
        + 0.07572434 * max(0.0, 0.196 - Q.sd_rg) / 0.07836805   # +7.6%  sd_rg < 0.196
        + 0.05363776 * max(0.0, 0.00344 - Q.lam2) / 0.00237948   # +5.4%  lam2 < 0.00344
        - 0.05313907 * max(0.0, 0.161 - Q.sd_rg) / 0.05567791   # -5.3%  sd_rg < 0.161
        - 0.05057212 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # -5.1%  log_sum_pt > 6.91
        - 0.04911705 * max(0.0, Q.n_real_top50 - 24.4) / 17.33011   # -4.9%  n_real_top50 > 24.4
        - 0.04016889 * max(0.0, 0.0744 - Q.tau1) / 0.01501071   # -4.0%  tau1 < 0.0744
        - 0.03625689 * max(0.0, 0.0292 - Q.e2) / 0.006293855   # -3.6%  e2 < 0.0292
        + 0.02781557 * max(0.0, Q.psi_0p1 - 0.728) / 0.121268   # +2.8%  psi_0p1 > 0.728
        + 0.02685086 * max(0.0, Q.e3 - 1.52e-05) / 8.973047e-05   # +2.7%  e3 > 1.52e-05
        - 0.02644196 * max(0.0, Q.psi_0p3 - 0.992) * max(0.0, 0.399 - Q.tau21_b2) / 0.0007626853   # -2.6%  psi_0p3 > 0.992 and tau21_b2 < 0.399
        + 0.02325449 * max(0.0, 0.347 - Q.tau21_b2) / 0.1116496   # +2.3%  tau21_b2 < 0.347
        - 0.02137104 * max(0.0, Q.e2 - 0.043) / 0.002905354   # -2.1%  e2 > 0.043
        - 0.02116376 * max(0.0, 1020.0 - Q.sum_pt_top50) / 27.07359   # -2.1%  sum_pt_top50 < 1020
        - 0.01960856 * max(0.0, Q.LHA - 0.296) / 0.02438504   # -2.0%  LHA > 0.296
        + 0.01917107 * max(0.0, Q.log_sum_pt - 7.01) / 0.01747956   # +1.9%  log_sum_pt > 7.01
        - 0.01883363 * max(0.0, 0.951 - Q.z_top20_slots) / 0.07107968   # -1.9%  z_top20_slots < 0.951
        + 0.01803972 * max(0.0, 0.0588 - Q.z_dr_0p1_0p2) / 0.01670911   # +1.8%  z_dr_0p1_0p2 < 0.0588
        + 0.01785771 * max(0.0, Q.sd_rg - 0.224) / 0.01601264   # +1.8%  sd_rg > 0.224
        - 0.01741329 * max(0.0, 0.623 - Q.z_dr_0p05_0p1) / 0.3828858   # -1.7%  z_dr_0p05_0p1 < 0.623
        - 0.01661179 * max(0.0, Q.psi_0p1 - 0.923) / 0.01536775   # -1.7%  psi_0p1 > 0.923
        + 0.01603583 * max(0.0, 0.00771 - Q.sum_z_dr2_top5) / 0.004054878   # +1.6%  sum_z_dr2_top5 < 0.00771
        - 0.0144206 * max(0.0, 0.0267 - Q.z_dr_0p2_0p4) / 0.0118263   # -1.4%  z_dr_0p2_0p4 < 0.0267
        - 0.01405206 * max(0.0, 0.0016 - Q.sum_z_dr2_top3) / 0.0005174641   # -1.4%  sum_z_dr2_top3 < 0.0016
        + 0.0101019 * max(0.0, Q.sj3_dr_max - 0.24) / 0.04944004   # +1.0%  sj3_dr_max > 0.24
        + 0.008436592 * max(0.0, Q.sj2_dr - 0.227) / 0.02285686   # +0.8%  sj2_dr > 0.227
        - 0.008321241 * max(0.0, Q.sd_rg - 0.28) / 0.006816865   # -0.8%  sd_rg > 0.28
        - 0.008242873 * max(0.0, Q.sj3_dr_max - 0.33) / 0.0179683   # -0.8%  sj3_dr_max > 0.33
        + 0.007227309 * max(0.0, Q.sum_z_dr2_top15 - 0.00608) / 0.00334303   # +0.7%  sum_z_dr2_top15 > 0.00608
        - 0.005037565 * max(0.0, 0.0534 - Q.z_dr_0p1_0p2) * max(0.0, 0.0334 - Q.absphi_1) / 0.0003184542   # -0.5%  z_dr_0p1_0p2 < 0.0534 and absphi_1 < 0.0334
        - 0.004688375 * max(0.0, Q.sj2_dr - 0.195) * max(0.0, Q.sj3_pairmin_over_m - 0.147) / 0.005663309   # -0.5%  sj2_dr > 0.195 and sj3_pairmin_over_m > 0.147
        - 0.004246472 * max(0.0, 0.133 - Q.z_dr_0p1_0p2) * max(0.0, 1.35 - Q.soft5_pt) / 0.02206393   # -0.4%  z_dr_0p1_0p2 < 0.133 and soft5_pt < 1.35
        + 0.00250567 * max(0.0, Q.n_dr_0_0p05 - 19.5) / 1.439983   # +0.3%  n_dr_0_0p05 > 19.5
        + 0.002449933 * max(0.0, 909.0 - Q.sum_pt) / 4.014031   # +0.2%  sum_pt < 909
        - 0.001752094 * max(0.0, 0.289 - Q.max_dr) / 0.01176215   # -0.2%  max_dr < 0.289
        + 0.001082893 * max(0.0, 0.000361 - Q.soft5_z) / 9.2717e-06   # +0.1%  soft5_z < 0.000361
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6594200630252101, 2.7687805147058824, 0.21400483193277312, 0.38426843487394957, 0.7695552521008403, 1.0444288865546218, 0.5437656512605042, 0.491338025210084, 1.169656512605042, 1.0008830882352941, 1.2650605042016807, 0.7516471638655462, 0.555001680672269, 1.5410298844537815, 0.2754823004201681, 0.440284768907563]
T = [3.8058746011685916, 2.4622309512867644, 4.247593292082458, 4.3628159466911764, 4.116293083639706]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +45%, n4 -18%, n9 +14%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.4546886 * h[1] / H_AVG[1]
            - 0.1769267 * h[4] / H_AVG[4]
            + 0.135601 * h[9] / H_AVG[9]
            - 0.08575796 * h[5] / H_AVG[5]
            - 0.07572539 * h[3] / H_AVG[3]
            + 0.03417835 * h[12] / H_AVG[12]
            + 0.02232427 * h[6] / H_AVG[6]
            + 0.009604038 * h[8] / H_AVG[8]
            - 0.0051937 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +23%, n1 -21%, n12 +8%, n11 -8%, n6 +5% ...
            - 0.3320779 * h[4] / H_AVG[4]
            + 0.2286531 * h[9] / H_AVG[9]
            - 0.2108439 * h[1] / H_AVG[1]
            + 0.07748332 * h[12] / H_AVG[12]
            - 0.0763177 * h[11] / H_AVG[11]
            + 0.04830933 * h[6] / H_AVG[6]
            + 0.01086438 * h[2] / H_AVG[2]
            - 0.008027911 * h[10] / H_AVG[10]
            + 0.007422489 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +14%, n0 +12%, n11 +11%, n14 -9%, n7 -7% ...
            - 0.2409481 * h[8] / H_AVG[8]
            + 0.1421535 * h[5] / H_AVG[5]
            + 0.1164342 * h[0] / H_AVG[0]
            + 0.105069 * h[11] / H_AVG[11]
            - 0.08917713 * h[14] / H_AVG[14]
            - 0.07229653 * h[7] / H_AVG[7]
            + 0.06227871 * h[4] / H_AVG[4]
            - 0.05308169 * h[12] / H_AVG[12]
            - 0.05154523 * h[9] / H_AVG[9]
            + 0.03957946 * h[3] / H_AVG[3]
            - 0.01943533 * h[15] / H_AVG[15]
            + 0.008001085 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -25%, n0 -21%, n5 +16%, n6 -12%, n7 +10%, n15 +6% ...
            - 0.2513406 * h[8] / H_AVG[8]
            - 0.2078251 * h[0] / H_AVG[0]
            + 0.1645829 * h[5] / H_AVG[5]
            - 0.1207415 * h[6] / H_AVG[6]
            + 0.1020614 * h[7] / H_AVG[7]
            + 0.05676613 * h[15] / H_AVG[15]
            + 0.03975369 * h[12] / H_AVG[12]
            + 0.03853416 * h[3] / H_AVG[3]
            - 0.0183945 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +30%, n5 -12%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3392757 * h[13] / H_AVG[13]
            + 0.302528 * h[10] / H_AVG[10]
            - 0.1189361 * h[5] / H_AVG[5]
            + 0.0599385 * h[8] / H_AVG[8]
            - 0.05056142 * h[12] / H_AVG[12]
            - 0.04011055 * h[15] / H_AVG[15]
            + 0.03505377 * h[4] / H_AVG[4]
            - 0.03357118 * h[7] / H_AVG[7]
            + 0.02002469 * h[0] / H_AVG[0]
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
