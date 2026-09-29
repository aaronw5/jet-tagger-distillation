"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  12.8%   (on for 40% of jets)
  neuron  1:  12.6%   (on for 97% of jets)
  neuron  5:  11.7%   (on for 81% of jets)
  neuron  4:  10.3%   (on for 69% of jets)
  neuron  0:   7.7%   (on for 62% of jets)
  neuron  9:   7.0%   (on for 66% of jets)
  neuron 13:   6.9%   (on for 86% of jets)
  neuron 10:   6.1%   (on for 82% of jets)
  neuron 12:   6.1%   (on for 41% of jets)
  neuron  7:   4.2%   (on for 44% of jets)
  neuron  3:   4.0%   (on for 59% of jets)
  neuron  6:   3.9%   (on for 44% of jets)
  neuron 11:   2.6%   (on for 50% of jets)
  neuron 14:   1.8%   (on for 31% of jets)
  neuron 15:   1.7%   (on for 50% of jets)
  neuron  2:   0.6%   (on for 79% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.4% (the network: 81.1%); same class as the network for 92.1% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
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
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.ptdr0_10               pT10 · ΔR(0, 10) [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.z_3                    pT of particle 3 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_12                 ΔR between particle 12 and the hardest particle
  Q.dr1_11                 ΔR between particle 11 and the 2nd-hardest particle
  Q.dr1_13                 ΔR between particle 13 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
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
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
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
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        ptdr0_10=pt[10] * math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        pt_11=pt[11],
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        z_3=z[3],
        soft1_z=softp(1, 'z'),
        soft5_z=softp(5, 'z'),
        soft7_z=softp(7, 'z'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        sd_zg=softdrop("zg"),
        sj2_dr=subjets(2)["dr"][0],
        dr0_12=math.sqrt(dist2(0, 12)) if pt[12] > 0 else 0.0,
        dr1_11=math.sqrt(dist2(1, 11)) if pt[11] > 0 else 0.0,
        dr1_13=math.sqrt(dist2(1, 13)) if pt[13] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
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
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
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
    )


def neuron_0(Q):
    # scale S = 27.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.23184 * (-0.05237031
        - 0.1839135 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # -18.4%  LHA < 0.3098
        + 0.1786671 * max(0.0, 0.08068193 - Q.sum_z_dr) / 0.02368455   # +17.9%  sum_z_dr < 0.08068
        + 0.07762233 * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) / 0.00326329   # +7.8%  sum_z_dr2_top15 < 0.007888
        - 0.06270861 * max(0.0, Q.sj2_dr - 0.1512157) / 0.05988199   # -6.3%  sj2_dr > 0.1512
        - 0.05353615 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # -5.4%  sum_z_dr < 0.0975
        - 0.05333991 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # -5.3%  sum_z_dr2_top15 < 0.009962
        + 0.04742814 * max(0.0, 0.3203321 - Q.LHA) / 0.07499257   # +4.7%  LHA < 0.3203
        + 0.04574986 * max(0.0, Q.sj2_dr - 0.1411617) / 0.06721719   # +4.6%  sj2_dr > 0.1412
        + 0.04502887 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +4.5%  LHA < 0.3332
        + 0.0320255 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_pt_top50 - 889.8503) / 1563.946   # +3.2%  n_dr_0p2_0p4 < 18 and sum_pt_top50 > 889.9
        - 0.0283294 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # -2.8%  dr_0 < 0.06413
        + 0.02376366 * max(0.0, 0.05775119 - Q.dr_0) / 0.02089184   # +2.4%  dr_0 < 0.05775
        - 0.0208133 * max(0.0, 0.04466492 - Q.tau1) / 0.005217805   # -2.1%  tau1 < 0.04466
        - 0.02062269 * max(0.0, 0.005383629 - Q.sum_z_dr2_top15) / 0.001666876   # -2.1%  sum_z_dr2_top15 < 0.005384
        + 0.02046855 * max(0.0, 0.08589404 - Q.sum_z_dr) / 0.02745918   # +2.0%  sum_z_dr < 0.08589
        - 0.01594648 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.log_sum_pt - 6.920349) / 0.4496771   # -1.6%  n_dr_0p2_0p4 < 18 and log_sum_pt > 6.92
        + 0.01356777 * max(0.0, Q.sj2_dr - 0.1745007) / 0.04529905   # +1.4%  sj2_dr > 0.1745
        + 0.01110293 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +1.1%  psi_0p3 > 0.9897
        + 0.01077412 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # +1.1%  LHA < 0.187
        - 0.009971283 * max(0.0, Q.sum_pt_top40 - 1001.523) / 47.19068   # -1.0%  sum_pt_top40 > 1002
        + 0.009222937 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +0.9%  n_dr_0p2_0p4 < 18
        - 0.008359824 * max(0.0, 0.05048381 - Q.sum_z_dr) / 0.008533025   # -0.8%  sum_z_dr < 0.05048
        + 0.005500368 * max(0.0, 0.4947602 - Q.z_dr_0_0p05) / 0.1759565   # +0.6%  z_dr_0_0p05 < 0.4948
        - 0.004262206 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # -0.4%  e2 < 0.0303
        - 0.00415772 * max(0.0, 2.883342e-05 - Q.e3) / 6.471346e-06   # -0.4%  e3 < 2.883e-05
        + 0.004035979 * max(0.0, 0.2125209 - Q.sj3_dr_max) / 0.0267777   # +0.4%  sj3_dr_max < 0.2125
        - 0.002669834 * max(0.0, 0.1623049 - Q.sj3_dr_max) / 0.01012502   # -0.3%  sj3_dr_max < 0.1623
        + 0.002304093 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +0.2%  e2 < 0.01879
        + 0.002072624 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # +0.2%  log_sum_pt > 7.017
        + 0.002034258 * max(0.0, Q.sum_pt_top40 - 1001.523) * max(0.0, 0.1512157 - Q.sj2_dr) / 1.305424   # +0.2%  sum_pt_top40 > 1002 and sj2_dr < 0.1512
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 14.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.293 * (0.07493454
        + 0.1385167 * max(0.0, Q.sum_pt - 972.0419) / 81.34075   # +13.9%  sum_pt > 972
        - 0.1377387 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -13.8%  log_sum_pt < 7.139
        + 0.1203272 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +12.0%  log_sum_pt > 6.91
        + 0.07284382 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # +7.3%  sum_pt_top50 < 1079
        - 0.05876826 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # -5.9%  sum_pt_top50 > 934.2
        + 0.04898474 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +4.9%  n_particles > 38
        + 0.04535648 * max(0.0, 1225.842 - Q.sum_pt_top40) / 211.117   # +4.5%  sum_pt_top40 < 1226
        - 0.04427866 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -4.4%  sum_pt_top50 > 959.1
        + 0.03696512 * max(0.0, Q.z_top30_slots - 0.9341838) / 0.03406818   # +3.7%  z_top30_slots > 0.9342
        - 0.03533255 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -3.5%  sum_pt > 1053
        + 0.03241383 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +3.2%  tau1 < 0.07709
        + 0.03118544 * max(0.0, 605.875 - Q.sum_pt_top2) / 245.9717   # +3.1%  sum_pt_top2 < 605.9
        - 0.0310069 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -3.1%  log_sum_pt > 6.989
        - 0.02334753 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 17.94219 - Q.ptdr0_3) / 0.4842322   # -2.3%  z_top30_slots > 0.9342 and ptdr0_3 < 17.94
        - 0.01631947 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.002181998 - Q.soft1_z) / 0.01445558   # -1.6%  n_particles > 38 and soft1_z < 0.002182
        - 0.01557597 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # -1.6%  lam2 < 0.001776
        - 0.01452913 * max(0.0, 0.03457336 - Q.M3) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.0006031662   # -1.5%  M3 < 0.03457 and psi_0p3 > 0.9299
        - 0.01234968 * max(0.0, 29.04688 - Q.pt_11) / 8.607263   # -1.2%  pt_11 < 29.05
        - 0.01043573 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -1.0%  n_dr_0p2_0p4 < 7
        + 0.00984188 * max(0.0, Q.n_particles - 38.0) * max(0.0, Q.tau32 - 0.3293142) / 3.693221   # +1.0%  n_particles > 38 and tau32 > 0.3293
        + 0.009726871 * max(0.0, 0.002197765 - Q.sum_z_dr2_top15) / 0.0004509993   # +1.0%  sum_z_dr2_top15 < 0.002198
        - 0.009609189 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) * max(0.0, 10.0 - Q.n_dr_0p05_0p1) / 0.2065903   # -1.0%  z_dr_0_0p05 > 0.8103 and n_dr_0p05_0p1 < 10
        - 0.009207883 * max(0.0, Q.z_top5 - 0.6551948) * max(0.0, 28.39396 - Q.pt1_dr01) / 0.8332473   # -0.9%  z_top5 > 0.6552 and pt1_dr01 < 28.39
        + 0.008026997 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.009970338 - Q.zdr_0) / 0.03317992   # +0.8%  n_particles > 38 and zdr_0 < 0.00997
        - 0.006171308 * max(0.0, 7.0 - Q.n_dr_0p1_0p2) / 0.9740454   # -0.6%  n_dr_0p1_0p2 < 7
        + 0.006089908 * max(0.0, Q.n_dr_0_0p05 - 15.0) / 2.713461   # +0.6%  n_dr_0_0p05 > 15
        - 0.005488043 * max(0.0, Q.tau1 - 0.1219132) / 0.01033631   # -0.5%  tau1 > 0.1219
        + 0.005210183 * max(0.0, 0.0006570502 - Q.sum_z_dr2_top5) / 0.0001395474   # +0.5%  sum_z_dr2_top5 < 0.0006571
        + 0.004351877 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.pt1_dr01 - 5.351077) / 0.1255531   # +0.4%  z_top30_slots > 0.9342 and pt1_dr01 > 5.351
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 2.172;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.171693 * (0.02172006
        - 0.2136629 * max(0.0, Q.sum_pt_top50 - 997.0189) / 56.58644   # -21.4%  sum_pt_top50 > 997
        + 0.1914641 * max(0.0, Q.log_sum_pt - 6.930088) / 0.0397436   # +19.1%  log_sum_pt > 6.93
        + 0.1591536 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.sum_z_dr2_top15) / 0.0009227558   # +15.9%  log_sum_pt > 6.903 and sum_z_dr2_top15 < 0.02147
        - 0.1106131 * max(0.0, Q.sum_pt_top50 - 1038.855) / 34.95069   # -11.1%  sum_pt_top50 > 1039
        - 0.08282863 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.sum_z_dr2_top15 - 0.009962397) / 1.329441e-05   # -8.3%  log_sum_pt > 7.063 and sum_z_dr2_top15 > 0.009962
        - 0.07594749 * max(0.0, 0.03577037 - Q.sum_z_dr) / 0.004182456   # -7.6%  sum_z_dr < 0.03577
        + 0.03815633 * max(0.0, Q.sum_pt_top50 - 997.0189) * max(0.0, 0.006794973 - Q.zdr_4) / 0.2427126   # +3.8%  sum_pt_top50 > 997 and zdr_4 < 0.006795
        + 0.03349292 * max(0.0, Q.sum_pt_top15 - 985.0781) / 16.07962   # +3.3%  sum_pt_top15 > 985.1
        + 0.02916811 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.8065577 - Q.tau21) / 0.01865613   # +2.9%  log_sum_pt > 6.903 and tau21 < 0.8066
        + 0.02666982 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # +2.7%  LHA < 0.187
        - 0.01868477 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.dr_max_012 - 0.1206357) / 0.06050554   # -1.9%  sum_pt_top20 > 1129 and dr_max_012 > 0.1206
        - 0.01328849 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.zdr_3 - 0.00453462) / 5.033547e-05   # -1.3%  log_sum_pt > 6.903 and zdr_3 > 0.004535
        - 0.006869759 * max(0.0, Q.sum_pt_top15 - 1082.548) / 5.972549   # -0.7%  sum_pt_top15 > 1083
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 1.393;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.393326 * (-0.1657925
        + 0.2592017 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +25.9%  n_dr_0p2_0p4 < 5
        + 0.1799261 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +18.0%  lam2 < 0.0006155
        + 0.1598084 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +16.0%  n_particles < 46
        + 0.1249001 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # +12.5%  n_dr_0p1_0p2 < 17
        - 0.09745931 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.e2 - 0.01036127) / 0.09655312   # -9.7%  n_particles < 46 and e2 > 0.01036
        - 0.07755465 * max(0.0, 0.347196 - Q.tau21) / 0.05239131   # -7.8%  tau21 < 0.3472
        - 0.05619802 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 30.51645   # -5.6%  n_dr_0p2_0p4 < 8 and n_dr_0p05_0p1 > 2
        - 0.04495174 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03029714 - Q.e2) / 0.006086253   # -4.5%  n_dr_0p2_0p4 < 5 and e2 < 0.0303
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 6.332;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.331815 * (0.2329217
        + 0.1612052 * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.004894699   # +16.1%  sum_z_dr2_top15 < 0.009962
        + 0.1269306 * max(0.0, Q.sj2_dr - 0.1219342) / 0.08223311   # +12.7%  sj2_dr > 0.1219
        - 0.09089926 * max(0.0, 1041.263 - Q.sum_pt_top40) / 49.16087   # -9.1%  sum_pt_top40 < 1041
        - 0.0767395 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -7.7%  e2 < 0.04359
        - 0.07106272 * max(0.0, Q.sj2_dr - 0.1825048) / 0.04110551   # -7.1%  sj2_dr > 0.1825
        - 0.07105932 * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) / 0.00326329   # -7.1%  sum_z_dr2_top15 < 0.007888
        - 0.06289321 * max(0.0, Q.n_particles - 26.0) / 20.24921   # -6.3%  n_particles > 26
        + 0.06137585 * max(0.0, 1011.524 - Q.sum_pt_top30) / 51.02618   # +6.1%  sum_pt_top30 < 1012
        + 0.04929218 * max(0.0, 0.09749958 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.9973959) / 3.542375e-05   # +4.9%  sum_z_dr < 0.0975 and psi_0p3 > 0.9974
        - 0.04829949 * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.02836816   # -4.8%  z_dr_0p2_0p4 < 0.0518
        - 0.0465751 * max(0.0, 0.07374472 - Q.sum_z_dr) / 0.01915593   # -4.7%  sum_z_dr < 0.07374
        - 0.04328587 * max(0.0, 0.005788041 - Q.sum_z_dr2_top15) / 0.001877672   # -4.3%  sum_z_dr2_top15 < 0.005788
        - 0.02864081 * max(0.0, 0.05660088 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.9973959) / 8.756067e-06   # -2.9%  sum_z_dr < 0.0566 and psi_0p3 > 0.9974
        + 0.01534998 * max(0.0, Q.n_dr_0_0p05 - 14.0) / 3.092148   # +1.5%  n_dr_0_0p05 > 14
        - 0.01282448 * max(0.0, 0.004169954 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 8.516876e-07   # -1.3%  sum_z_dr2_top15 < 0.00417 and psi_0p3 > 0.9974
        - 0.01153095 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -1.2%  psi_0p1 > 0.8976
        - 0.01123799 * max(0.0, Q.n_particles - 26.0) * max(0.0, Q.soft1_pt - 0.4909668) / 9.262955   # -1.1%  n_particles > 26 and soft1_pt > 0.491
        + 0.01079752 * max(0.0, 0.002197765 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.9973959) / 3.575644e-07   # +1.1%  sum_z_dr2_top15 < 0.002198 and psi_0p3 > 0.9974
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 8.548;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.547662 * (-0.2317338
        + 0.2977322 * max(0.0, 0.1564779 - Q.sum_z_dr) / 0.08789501   # +29.8%  sum_z_dr < 0.1565
        + 0.1138181 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +11.4%  n_particles < 64
        - 0.06838539 * max(0.0, 58.0 - Q.n_pt_above_1) / 17.32351   # -6.8%  n_pt_above_1 < 58
        - 0.0662746 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -6.6%  tau1 < 0.1073
        + 0.05120821 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # +5.1%  sum_pt < 1085
        + 0.04992121 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +5.0%  n_dr_0p2_0p4 < 11
        - 0.04240225 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 7.684973e-07   # -4.2%  sum_pt < 1002 and e4 < 5.851e-08
        - 0.03766422 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -3.8%  sum_pt < 1002
        + 0.02958852 * max(0.0, Q.LHA - 0.1632346) / 0.1084458   # +3.0%  LHA > 0.1632
        - 0.0249917 * max(0.0, Q.LHA - 0.4331369) / 0.001006142   # -2.5%  LHA > 0.4331
        - 0.02404945 * max(0.0, Q.z_top20_slots - 0.8281581) / 0.07910948   # -2.4%  z_top20_slots > 0.8282
        - 0.02125545 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 0.07996447 - Q.C2) / 0.5514341   # -2.1%  n_particles < 64 and C2 < 0.07996
        + 0.01836272 * max(0.0, Q.z_dr_0_0p05 - 0.7128619) / 0.07526795   # +1.8%  z_dr_0_0p05 > 0.7129
        + 0.01731639 * max(0.0, Q.LHA - 0.302389) / 0.02201444   # +1.7%  LHA > 0.3024
        + 0.01711947 * max(0.0, 1013.042 - Q.sum_pt_top40) / 31.91616   # +1.7%  sum_pt_top40 < 1013
        - 0.01687503 * max(0.0, 907.9372 - Q.sum_pt) / 3.961002   # -1.7%  sum_pt < 907.9
        - 0.01532504 * max(0.0, 0.985099 - Q.z_top50_slots) / 0.003556075   # -1.5%  z_top50_slots < 0.9851
        + 0.01437065 * max(0.0, Q.sum_z_dr2_top10 - 0.005337976) / 0.003351803   # +1.4%  sum_z_dr2_top10 > 0.005338
        - 0.01376076 * max(0.0, 907.9372 - Q.sum_pt) * max(0.0, 0.2535773 - Q.dr_11) / 0.6391318   # -1.4%  sum_pt < 907.9 and dr_11 < 0.2536
        + 0.01216587 * max(0.0, 934.2416 - Q.sum_pt_top50) / 6.932864   # +1.2%  sum_pt_top50 < 934.2
        - 0.01185846 * max(0.0, Q.C2 - 0.1235569) / 0.002419231   # -1.2%  C2 > 0.1236
        + 0.009092962 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 3.38926e-06   # +0.9%  sum_pt < 1085 and e4 < 5.851e-08
        + 0.007799248 * max(0.0, 907.9372 - Q.sum_pt) * max(0.0, 0.1569963 - Q.dr0_12) / 0.3024992   # +0.8%  sum_pt < 907.9 and dr0_12 < 0.157
        - 0.005551638 * max(0.0, 0.0007894752 - Q.sum_z_dr2_top15) / 9.062109e-05   # -0.6%  sum_z_dr2_top15 < 0.0007895
        - 0.005052105 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.ptdr0_10 - 3.719859) / 10.46372   # -0.5%  sum_pt < 1002 and ptdr0_10 > 3.72
        + 0.004196574 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, 0.199671 - Q.dr_11) / 1.707103   # +0.4%  sum_pt < 1002 and dr_11 < 0.1997
        + 0.003861694 * max(0.0, 1002.379 - Q.sum_pt) * max(0.0, Q.dr1_13 - 0.1778153) / 0.5300108   # +0.4%  sum_pt < 1002 and dr1_13 > 0.1778
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 13.19;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.1917 * (0.01858446
        - 0.2827215 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # -28.3%  sum_z_dr < 0.0975
        + 0.1080781 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +10.8%  LHA < 0.3332
        + 0.07102639 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # +7.1%  tau1 < 0.05445
        - 0.06527397 * max(0.0, Q.e3 - 0.0001086251) / 5.31284e-05   # -6.5%  e3 > 0.0001086
        + 0.06447279 * max(0.0, Q.e3 - 5.13841e-05) / 6.821576e-05   # +6.4%  e3 > 5.138e-05
        + 0.06110735 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +6.1%  e3 < 0.0003372
        - 0.04063197 * max(0.0, 0.08589404 - Q.sum_z_dr) / 0.02745918   # -4.1%  sum_z_dr < 0.08589
        + 0.04013702 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # +4.0%  e2 < 0.04359
        + 0.03600802 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +3.6%  sum_pt > 907.9
        - 0.02526687 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -2.5%  psi_0p2 > 0.9087
        - 0.02525378 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, 7.017258 - Q.log_sum_pt) / 2.140029e-05   # -2.5%  e3 < 0.0003372 and log_sum_pt < 7.017
        + 0.02409195 * max(0.0, 0.006427167 - Q.lam2) / 0.005146703   # +2.4%  lam2 < 0.006427
        - 0.02401316 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # -2.4%  z_dr_0p2_0p4 < 0.06849
        + 0.02150874 * max(0.0, 0.003270031 - Q.sum_z_dr2_top15) / 0.0007969965   # +2.2%  sum_z_dr2_top15 < 0.00327
        + 0.01704889 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # +1.7%  tau2 < 0.0795
        + 0.01620851 * max(0.0, Q.z_dr_0_0p05 - 0.3289237) / 0.280304   # +1.6%  z_dr_0_0p05 > 0.3289
        - 0.01256587 * max(0.0, Q.n_dr_0p1_0p2 - 15.0) / 2.705171   # -1.3%  n_dr_0p1_0p2 > 15
        + 0.01069617 * max(0.0, Q.n_dr_0p1_0p2 - 15.0) * max(0.0, Q.psi_0p3 - 0.9853273) / 0.01929542   # +1.1%  n_dr_0p1_0p2 > 15 and psi_0p3 > 0.9853
        + 0.01065332 * max(0.0, 0.3676068 - Q.sj2_zsoft) / 0.1397909   # +1.1%  sj2_zsoft < 0.3676
        - 0.00962458 * max(0.0, 0.003270031 - Q.sum_z_dr2_top15) * max(0.0, Q.sum_pt_top40 - 858.8262) / 0.1598282   # -1.0%  sum_z_dr2_top15 < 0.00327 and sum_pt_top40 > 858.8
        + 0.007761084 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # +0.8%  e2 > 0.05557
        - 0.007480498 * max(0.0, Q.C2 - 0.1088881) / 0.004226185   # -0.7%  C2 > 0.1089
        - 0.005759549 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) / 1.539217   # -0.6%  n_dr_0p2_0p4 > 15
        - 0.005355397 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.03720487   # -0.5%  z_dr_0p1_0p2 > 0.334
        - 0.003480982 * max(0.0, 0.001319197 - Q.sum_z_dr2_top15) / 0.0002099631   # -0.3%  sum_z_dr2_top15 < 0.001319
        + 0.002549661 * max(0.0, 0.05444509 - Q.tau1) * max(0.0, 972.0419 - Q.sum_pt) / 0.0682246   # +0.3%  tau1 < 0.05445 and sum_pt < 972
        + 0.00122391 * max(0.0, 0.09749958 - Q.sum_z_dr) * max(0.0, 0.9638082 - Q.psi_0p3) / 1.616441e-05   # +0.1%  sum_z_dr < 0.0975 and psi_0p3 < 0.9638
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 26.97;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.96941 * (-0.0118599
        - 0.108944 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -10.9%  tau1 < 0.1073
        + 0.0987158 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +9.9%  psi_0p3 > 0.9974
        - 0.08095918 * max(0.0, 0.08068193 - Q.sum_z_dr) / 0.02368455   # -8.1%  sum_z_dr < 0.08068
        + 0.07124854 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # +7.1%  LHA < 0.3024
        + 0.07061735 * max(0.0, 0.1072713 - Q.tau1) * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) / 0.008148309   # +7.1%  tau1 < 0.1073 and z_dr_0p1_0p2 < 0.2865
        + 0.06566738 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # +6.6%  sum_z_dr < 0.0975
        - 0.05332926 * max(0.0, 0.3332345 - Q.LHA) * max(0.0, 0.250441 - Q.z_dr_0p1_0p2) / 0.01705837   # -5.3%  LHA < 0.3332 and z_dr_0p1_0p2 < 0.2504
        + 0.051807 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063   # +5.2%  n_dr_0p2_0p4 < 21
        - 0.04307334 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.006802603 - Q.mean_eta2) / 3.803697e-06   # -4.3%  psi_0p3 > 0.9974 and mean_eta2 < 0.006803
        - 0.0406392 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.006808102 - Q.mean_phi2) / 3.800578e-06   # -4.1%  psi_0p3 > 0.9974 and mean_phi2 < 0.006808
        + 0.03340056 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.009962397 - Q.sum_z_dr2_top15) / 0.0001707165   # +3.3%  tau21_b2 < 0.2352 and sum_z_dr2_top15 < 0.009962
        - 0.03031983 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -3.0%  n_dr_0p2_0p4 < 15
        + 0.02975288 * max(0.0, 0.1219132 - Q.tau1) / 0.04578087   # +3.0%  tau1 < 0.1219
        - 0.02544188 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1245.697 - Q.sum_pt_top50) / 11.02846   # -2.5%  tau21_b2 < 0.2352 and sum_pt_top50 < 1246
        - 0.02333681 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 7.062574 - Q.log_sum_pt) / 0.0001122434   # -2.3%  psi_0p3 > 0.9974 and log_sum_pt < 7.063
        - 0.02081628 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3) / 0.0005360215   # -2.1%  n_dr_0p2_0p4 < 21 and e3 < 7.876e-05
        + 0.01968847 * max(0.0, 0.03680582 - Q.e2) / 0.01052402   # +2.0%  e2 < 0.03681
        - 0.01957328 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) / 8.087416e-05   # -2.0%  tau21_b2 < 0.2352 and sum_z_dr2_top15 < 0.007888
        + 0.01777182 * max(0.0, 0.05557149 - Q.e2) / 0.02580562   # +1.8%  e2 < 0.05557
        - 0.01289025 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # -1.3%  LHA < 0.3332
        - 0.01129182 * max(0.0, 0.04755309 - Q.e2) / 0.01877457   # -1.1%  e2 < 0.04755
        + 0.01021459 * max(0.0, 6.567534e-05 - Q.e3) / 2.643172e-05   # +1.0%  e3 < 6.568e-05
        - 0.009292863 * max(0.0, Q.psi_0p1 - 0.8509811) / 0.04568859   # -0.9%  psi_0p1 > 0.851
        + 0.009080966 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1115.723 - Q.sum_pt) / 4.512378   # +0.9%  tau21_b2 < 0.2352 and sum_pt < 1116
        + 0.008332397 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.1134943 - Q.M2) / 0.3618294   # +0.8%  n_dr_0p2_0p4 < 15 and M2 < 0.1135
        - 0.008088688 * max(0.0, 5.727594e-05 - Q.e3) / 2.076531e-05   # -0.8%  e3 < 5.728e-05
        - 0.006255477 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.sum_z_dr2_top10 - 0.007678544) / 7.055851e-07   # -0.6%  psi_0p3 > 0.9974 and sum_z_dr2_top10 > 0.007679
        - 0.006029874 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.sum_z_dr2_top10 - 0.01414829) / 3.726039e-07   # -0.6%  psi_0p3 > 0.9974 and sum_z_dr2_top10 > 0.01415
        - 0.004573518 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 3.376709e-05 - Q.e3) / 0.0001233727   # -0.5%  n_dr_0p2_0p4 < 21 and e3 < 3.377e-05
        + 0.004242668 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 0.003687605 - Q.lam2) / 2.993211e-06   # +0.4%  psi_0p3 > 0.9974 and lam2 < 0.003688
        + 0.003670789 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.001696484   # +0.4%  tau21_b2 < 0.2352 and sj2_dr > 0.1938
        - 0.0009332076 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 972.0419 - Q.sum_pt) / 0.2148301   # -0.1%  tau21_b2 < 0.2352 and sum_pt < 972
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 25.42;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.4236 * (0.08902127
        - 0.1513654 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -15.1%  log_sum_pt < 7.139
        + 0.1140896 * max(0.0, Q.sum_z_dr - 0.04362872) / 0.03194126   # +11.4%  sum_z_dr > 0.04363
        + 0.09534062 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # +9.5%  sum_pt < 1261
        + 0.09155735 * max(0.0, Q.sum_z_dr - 0.03577037) / 0.03765667   # +9.2%  sum_z_dr > 0.03577
        - 0.08768578 * max(0.0, Q.LHA - 0.2091025) / 0.07391154   # -8.8%  LHA > 0.2091
        - 0.04841566 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # -4.8%  sum_z_dr > 0.0975
        - 0.03845775 * max(0.0, Q.sum_z_dr - 0.07374472) / 0.01465579   # -3.8%  sum_z_dr > 0.07374
        + 0.03066982 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +3.1%  LHA > 0.3332
        - 0.02834355 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.139296 - Q.log_sum_pt) / 1.930358   # -2.8%  n_dr_0p2_0p4 < 18 and log_sum_pt < 7.139
        + 0.02749011 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, 1191.938 - Q.sum_pt_top30) / 4.563404   # +2.7%  sum_z_dr > 0.07032 and sum_pt_top30 < 1192
        - 0.02467847 * max(0.0, Q.z_top50_slots - 0.9586536) / 0.03427112   # -2.5%  z_top50_slots > 0.9587
        - 0.02351325 * max(0.0, Q.psi_0p3 - 0.9777125) / 0.01571542   # -2.4%  psi_0p3 > 0.9777
        - 0.02327818 * max(0.0, 0.00727763 - Q.sum_z_dr2_top15) / 0.00282147   # -2.3%  sum_z_dr2_top15 < 0.007278
        - 0.02220859 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 3.793233e-05 - Q.e3) / 0.0001192169   # -2.2%  n_dr_0p2_0p4 < 18 and e3 < 3.793e-05
        - 0.02136334 * max(0.0, Q.e2 - 0.02210818) / 0.01219843   # -2.1%  e2 > 0.02211
        - 0.02074509 * max(0.0, Q.sum_z_dr - 0.07031778) / 0.01612201   # -2.1%  sum_z_dr > 0.07032
        + 0.01459056 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, 1115.723 - Q.sum_pt) / 1.878598   # +1.5%  sum_z_dr > 0.07032 and sum_pt < 1116
        - 0.01440404 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 4.450169 - Q.D2) / 457.6573   # -1.4%  sum_pt < 1261 and D2 < 4.45
        - 0.01239312 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # -1.2%  n_dr_0p2_0p4 < 8
        + 0.01086037 * max(0.0, 1042.609 - Q.sum_pt) / 35.65948   # +1.1%  sum_pt < 1043
        + 0.01078294 * max(0.0, 0.08082334 - Q.dr_0) / 0.0377444   # +1.1%  dr_0 < 0.08082
        - 0.009607589 * max(0.0, Q.LHA - 0.3203321) / 0.01671666   # -1.0%  LHA > 0.3203
        - 0.009224027 * max(0.0, 1027.303 - Q.sum_pt_top30) / 61.36161   # -0.9%  sum_pt_top30 < 1027
        + 0.007969408 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, Q.e2 - 0.04358622) / 0.000222409   # +0.8%  sum_z_dr > 0.07032 and e2 > 0.04359
        + 0.007611147 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # +0.8%  sum_pt < 1002
        + 0.007567318 * max(0.0, 1042.609 - Q.sum_pt) * max(0.0, Q.max_dr - 0.1939977) / 6.226413   # +0.8%  sum_pt < 1043 and max_dr > 0.194
        + 0.006209635 * max(0.0, 3.793233e-05 - Q.e3) / 1.004564e-05   # +0.6%  e3 < 3.793e-05
        - 0.005522313 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, 976.277 - Q.sum_pt_top50) / 0.4391148   # -0.6%  sum_z_dr > 0.07032 and sum_pt_top50 < 976.3
        + 0.004990825 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.8252004 - Q.psi_0p1) / 0.5392896   # +0.5%  n_dr_0p2_0p4 < 18 and psi_0p1 < 0.8252
        + 0.004605098 * max(0.0, Q.e2 - 0.04358622) / 0.002786698   # +0.5%  e2 > 0.04359
        - 0.004564191 * max(0.0, Q.LHA - 0.3098384) / 0.0195892   # -0.5%  LHA > 0.3098
        + 0.004159367 * max(0.0, Q.sum_z_dr - 0.07031778) * max(0.0, 0.0795038 - Q.tau2) / 0.0003075621   # +0.4%  sum_z_dr > 0.07032 and tau2 < 0.0795
        + 0.003884176 * max(0.0, 0.02675364 - Q.sum_z_dr2_top15) * max(0.0, Q.tau4 - 0.01517184) / 5.673266e-05   # +0.4%  sum_z_dr2_top15 < 0.02675 and tau4 > 0.01517
        + 0.003664306 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +0.4%  LHA > 0.372
        + 0.003170909 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # +0.3%  sum_pt_top40 < 1007
        + 0.003061816 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.3%  sj2_dr > 0.2232
        + 0.001954355 * max(0.0, Q.psi_0p3 - 0.9777125) * max(0.0, Q.n_dr_0p1_0p2 - 17.0) / 0.02691594   # +0.2%  psi_0p3 > 0.9777 and n_dr_0p1_0p2 > 17
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 8.197;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.196617 * (-0.08004832
        + 0.1277271 * max(0.0, 0.09591084 - Q.tau1) / 0.02629125   # +12.8%  tau1 < 0.09591
        - 0.0870887 * max(0.0, Q.sum_pt_top20 - 750.7313) / 184.752   # -8.7%  sum_pt_top20 > 750.7
        + 0.07839494 * max(0.0, Q.e2 - 0.02515919) / 0.01027642   # +7.8%  e2 > 0.02516
        + 0.07716477 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # +7.7%  tau1 < 0.1073
        + 0.07626495 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.000105955   # +7.6%  sum_z_dr2_top15 < 0.006143 and z_dr_0p2_0p4 < 0.06849
        - 0.07399818 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -7.4%  sum_z_dr > 0.1207
        - 0.05583808 * max(0.0, Q.z_top40_slots - 0.9674996) / 0.02046045   # -5.6%  z_top40_slots > 0.9675
        + 0.03521449 * max(0.0, 2.178951 - Q.D2) / 0.4648731   # +3.5%  D2 < 2.179
        - 0.03144265 * max(0.0, 0.004855289 - Q.sum_z_dr2_top15) / 0.001418414   # -3.1%  sum_z_dr2_top15 < 0.004855
        + 0.03123447 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # +3.1%  sum_z_dr > 0.0975
        + 0.03069355 * max(0.0, 0.05660088 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.0003340726   # +3.1%  sum_z_dr < 0.0566 and psi_0p3 > 0.9638
        + 0.0301396 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +3.0%  LHA > 0.372
        - 0.028845 * max(0.0, 0.002412891 - Q.sum_z_dr2_top2) / 0.0009487944   # -2.9%  sum_z_dr2_top2 < 0.002413
        - 0.02808759 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # -2.8%  LHA < 0.2454
        + 0.02171882 * max(0.0, 0.05660088 - Q.sum_z_dr) / 0.01080382   # +2.2%  sum_z_dr < 0.0566
        - 0.02091513 * max(0.0, Q.sum_z_dr - 0.1402186) / 0.001871547   # -2.1%  sum_z_dr > 0.1402
        + 0.02079401 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +2.1%  LHA > 0.4042
        + 0.02042529 * max(0.0, 988.4554 - Q.sum_pt_top50) / 15.57124   # +2.0%  sum_pt_top50 < 988.5
        + 0.01873603 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.09061548 - Q.dr_13) / 0.0001299587   # +1.9%  log_sum_pt < 6.811 and dr_13 < 0.09062
        - 0.01853913 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) / 0.002080651   # -1.9%  sum_z_dr2_top15 < 0.006143
        + 0.01238782 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, Q.soft5_pt - 0.5297852) / 0.02338419   # +1.2%  z_dr_0p1_0p2 > 0.4479 and soft5_pt > 0.5298
        - 0.01121572 * max(0.0, Q.sum_z_dr2_top15 - 0.02146578) / 0.0006182335   # -1.1%  sum_z_dr2_top15 > 0.02147
        - 0.01112147 * max(0.0, 0.0007894752 - Q.sum_z_dr2_top15) / 9.062109e-05   # -1.1%  sum_z_dr2_top15 < 0.0007895
        + 0.01053878 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) * max(0.0, Q.n_dr_0p2_0p4 - 7.0) / 0.004066923   # +1.1%  sum_z_dr2_top15 < 0.006143 and n_dr_0p2_0p4 > 7
        - 0.01048168 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, Q.soft5_z - 0.0004140594) / 2.555122e-05   # -1.0%  z_dr_0p1_0p2 > 0.4479 and soft5_z > 0.0004141
        + 0.00794552 * max(0.0, 889.8503 - Q.sum_pt_top50) / 3.808547   # +0.8%  sum_pt_top50 < 889.9
        - 0.007891894 * max(0.0, 988.4554 - Q.sum_pt_top50) * max(0.0, 0.1312677 - Q.dr_13) / 0.7213696   # -0.8%  sum_pt_top50 < 988.5 and dr_13 < 0.1313
        - 0.006911214 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.7%  log_sum_pt < 6.811
        - 0.0042553 * max(0.0, Q.sum_z_dr - 0.1402186) * max(0.0, Q.soft6_pt - 4.250195) / 9.168767e-05   # -0.4%  sum_z_dr > 0.1402 and soft6_pt > 4.25
        - 0.002839815 * max(0.0, 0.05660088 - Q.sum_z_dr) * max(0.0, 1007.788 - Q.sum_pt) / 0.1758936   # -0.3%  sum_z_dr < 0.0566 and sum_pt < 1008
        + 0.00114828 * max(0.0, Q.sum_z_dr - 0.1564779) * max(0.0, Q.soft6_pt - 1.931641) / 0.0004276585   # +0.1%  sum_z_dr > 0.1565 and soft6_pt > 1.932
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 8.497;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.496787 * (0.6988592
        - 0.2809366 * max(0.0, 0.1207452 - Q.sum_z_dr) / 0.05578614   # -28.1%  sum_z_dr < 0.1207
        - 0.2216777 * max(0.0, Q.LHA - 0.1870291) / 0.08997036   # -22.2%  LHA > 0.187
        - 0.07186217 * max(0.0, 0.01414829 - Q.sum_z_dr2_top10) / 0.00877908   # -7.2%  sum_z_dr2_top10 < 0.01415
        - 0.06721969 * max(0.0, Q.sum_pt_top5 - 402.625) / 200.0399   # -6.7%  sum_pt_top5 > 402.6
        + 0.04569804 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # +4.6%  e2 > 0.0303
        - 0.0322865 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # -3.2%  sum_z_dr > 0.0975
        - 0.02790404 * max(0.0, 0.2018786 - Q.tau21_b2) / 0.03799024   # -2.8%  tau21_b2 < 0.2019
        - 0.02789077 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # -2.8%  z_dr_0p1_0p2 < 0.1203
        + 0.02629072 * max(0.0, 0.1207452 - Q.sum_z_dr) * max(0.0, Q.psi_0p2 - 0.948102) / 0.001922283   # +2.6%  sum_z_dr < 0.1207 and psi_0p2 > 0.9481
        + 0.02623446 * max(0.0, 2.410481 - Q.D2) / 0.587301   # +2.6%  D2 < 2.41
        + 0.02096089 * max(0.0, 0.05602756 - Q.C2) / 0.009882647   # +2.1%  C2 < 0.05603
        + 0.02068458 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +2.1%  LHA > 0.3332
        + 0.02046097 * max(0.0, Q.psi_0p1 - 0.9184255) / 0.01693566   # +2.0%  psi_0p1 > 0.9184
        + 0.01882893 * max(0.0, Q.LHA - 0.1870291) * max(0.0, Q.sj3_pairmin_over_m - 0.1389615) / 0.01420104   # +1.9%  LHA > 0.187 and sj3_pairmin_over_m > 0.139
        + 0.01728753 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +1.7%  sum_pt < 986.1
        - 0.01121524 * max(0.0, Q.sj2_dr - 0.09395198) * max(0.0, Q.M2 - 0.04260132) / 0.001979695   # -1.1%  sj2_dr > 0.09395 and M2 > 0.0426
        + 0.01070613 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, 114.75 - Q.pt_3) / 0.04784342   # +1.1%  tau1 > 0.1954 and pt_3 < 114.8
        - 0.01026582 * max(0.0, 6.856375 - Q.log_sum_pt) / 0.008053459   # -1.0%  log_sum_pt < 6.856
        - 0.00994702 * max(0.0, Q.psi_0p3 - 0.9989733) / 0.0002740091   # -1.0%  psi_0p3 > 0.999
        - 0.00843495 * max(0.0, Q.tau1 - 0.1953848) / 0.000822026   # -0.8%  tau1 > 0.1954
        - 0.005995585 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, 0.1082864 - Q.z_3) / 4.224064e-05   # -0.6%  tau1 > 0.1954 and z_3 < 0.1083
        + 0.005643441 * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 1.386277   # +0.6%  n_dr_0p1_0p2 > 21
        - 0.005251537 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # -0.5%  sum_pt_top5 > 791.1
        - 0.003982002 * max(0.0, Q.e2 - 0.03029714) * max(0.0, Q.log_sum_pt - 6.910131) / 0.0002282547   # -0.4%  e2 > 0.0303 and log_sum_pt > 6.91
        - 0.001845024 * max(0.0, 889.8503 - Q.sum_pt_top50) / 3.808547   # -0.2%  sum_pt_top50 < 889.9
        + 0.0004896507 * max(0.0, Q.tau1 - 0.1953848) * max(0.0, Q.sum_pt - 1167.447) / 0.008922112   # +0.0%  tau1 > 0.1954 and sum_pt > 1167
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 15.69;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.68767 * (-0.04139308
        - 0.1577157 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # -15.8%  LHA < 0.3098
        + 0.1470015 * max(0.0, 0.08589404 - Q.sum_z_dr) / 0.02745918   # +14.7%  sum_z_dr < 0.08589
        - 0.1053347 * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) / 0.002080651   # -10.5%  sum_z_dr2_top15 < 0.006143
        + 0.1021053 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +10.2%  LHA < 0.3332
        - 0.07456289 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # -7.5%  e2 < 0.04083
        + 0.06332421 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +6.3%  e2 < 0.02516
        + 0.04719554 * max(0.0, 0.007887677 - Q.sum_z_dr2_top15) / 0.00326329   # +4.7%  sum_z_dr2_top15 < 0.007888
        - 0.04226389 * max(0.0, 0.09749958 - Q.sum_z_dr) / 0.03657082   # -4.2%  sum_z_dr < 0.0975
        + 0.04109537 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4) / 0.04053661   # +4.1%  z_dr_0p2_0p4 < 0.06849
        - 0.0379685 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # -3.8%  sum_pt > 986.1
        + 0.02913818 * max(0.0, Q.log_sum_pt - 6.893714) * max(0.0, Q.z_top50_slots - 0.9704436) / 0.001405037   # +2.9%  log_sum_pt > 6.894 and z_top50_slots > 0.9704
        + 0.02366527 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 33.0 - Q.n_dr_0p1_0p2) / 87.77824   # +2.4%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 33
        - 0.02024155 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # -2.0%  log_sum_pt > 6.894
        + 0.01874342 * max(0.0, 0.1894436 - Q.sj3_dr_max) / 0.01747403   # +1.9%  sj3_dr_max < 0.1894
        + 0.01766205 * max(0.0, 19.0 - Q.n_dr_0p1_0p2) / 8.166005   # +1.8%  n_dr_0p1_0p2 < 19
        - 0.01666789 * max(0.0, 0.1623049 - Q.sj3_dr_max) / 0.01012502   # -1.7%  sj3_dr_max < 0.1623
        + 0.01286856 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) / 0.03630126   # +1.3%  z_dr_0_0p05 > 0.8103
        - 0.01114352 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005337976 - Q.sum_z_dr2_top10) / 0.007083386   # -1.1%  n_dr_0p2_0p4 < 10 and sum_z_dr2_top10 < 0.005338
        + 0.01011784 * max(0.0, 0.00130722 - Q.sum_z_dr2_top10) / 0.0002794102   # +1.0%  sum_z_dr2_top10 < 0.001307
        - 0.006362435 * max(0.0, 0.076787 - Q.sum_z_dr) / 0.02105267   # -0.6%  sum_z_dr < 0.07679
        - 0.006085343 * max(0.0, 1.409617 - Q.D2) * max(0.0, Q.n_real_top50 - 22.0) / 2.18239   # -0.6%  D2 < 1.41 and n_real_top50 > 22
        + 0.004598771 * max(0.0, 1.409617 - Q.D2) / 0.1519846   # +0.5%  D2 < 1.41
        + 0.004137683 * max(0.0, Q.log_sum_pt - 6.893714) * max(0.0, 0.3213081 - Q.sd_zg) / 0.005513471   # +0.4%  log_sum_pt > 6.894 and sd_zg < 0.3213
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 11.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.08881 * (0.01188279
        + 0.1548807 * max(0.0, 0.06171014 - Q.sum_z_dr) / 0.01295056   # +15.5%  sum_z_dr < 0.06171
        - 0.1068283 * max(0.0, Q.log_sum_pt - 6.811175) / 0.1383141   # -10.7%  log_sum_pt > 6.811
        - 0.06057658 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -6.1%  psi_0p3 > 0.9897
        - 0.06022831 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # -6.0%  LHA < 0.2454
        - 0.05308982 * max(0.0, 1167.447 - Q.sum_pt) / 137.8628   # -5.3%  sum_pt < 1167
        - 0.05217205 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, 0.01204531 - Q.mean_eta2) / 0.001166867   # -5.2%  log_sum_pt > 6.811 and mean_eta2 < 0.01205
        - 0.04413054 * max(0.0, 0.2284021 - Q.LHA) / 0.02716657   # -4.4%  LHA < 0.2284
        + 0.0395521 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.006615185 - Q.sum_z_dr2_top15) / 1.491736e-05   # +4.0%  psi_0p3 > 0.9897 and sum_z_dr2_top15 < 0.006615
        + 0.03523664 * max(0.0, Q.psi_0p2 - 0.9734513) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 0.1343004   # +3.5%  psi_0p2 > 0.9735 and n_dr_0p1_0p2 < 21
        + 0.03496963 * max(0.0, 0.05048381 - Q.sum_z_dr) * max(0.0, Q.log_sum_pt - 6.811175) / 0.001311076   # +3.5%  sum_z_dr < 0.05048 and log_sum_pt > 6.811
        + 0.03192376 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +3.2%  n_dr_0p2_0p4 < 10
        + 0.03141594 * max(0.0, 4.450169 - Q.D2) / 2.013555   # +3.1%  D2 < 4.45
        + 0.03041978 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 7.139296 - Q.log_sum_pt) / 0.00115378   # +3.0%  psi_0p3 > 0.9897 and log_sum_pt < 7.139
        + 0.02794291 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +2.8%  LHA > 0.372
        + 0.02709887 * max(0.0, 0.1999777 - Q.sj3_dr_max) / 0.02143   # +2.7%  sj3_dr_max < 0.2
        - 0.02599445 * max(0.0, 0.1048824 - Q.sj3_dr12) / 0.02123626   # -2.6%  sj3_dr12 < 0.1049
        + 0.02570968 * max(0.0, 0.005850286 - Q.mean_eta2) / 0.00275351   # +2.6%  mean_eta2 < 0.00585
        - 0.02372916 * max(0.0, 0.2628766 - Q.sj3_dr_max) * max(0.0, Q.sum_z_dr2_top10 - 0.004752876) / 3.407988e-05   # -2.4%  sj3_dr_max < 0.2629 and sum_z_dr2_top10 > 0.004753
        + 0.02288677 * max(0.0, 0.1840219 - Q.sj3_dr12) / 0.05790731   # +2.3%  sj3_dr12 < 0.184
        - 0.02125895 * max(0.0, 0.1506299 - Q.sj3_dr_max) / 0.008148644   # -2.1%  sj3_dr_max < 0.1506
        - 0.01604501 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # -1.6%  LHA > 0.4042
        - 0.01400537 * max(0.0, Q.eccentricity - 0.8903081) / 0.02227339   # -1.4%  eccentricity > 0.8903
        - 0.01296878 * max(0.0, 0.1870291 - Q.LHA) / 0.01494324   # -1.3%  LHA < 0.187
        - 0.01141644 * max(0.0, Q.e2 - 0.03029714) / 0.007411026   # -1.1%  e2 > 0.0303
        + 0.008685532 * max(0.0, 0.2628766 - Q.sj3_dr_max) * max(0.0, Q.eccentricity - 0.5245966) / 0.01227931   # +0.9%  sj3_dr_max < 0.2629 and eccentricity > 0.5246
        + 0.007407783 * max(0.0, 0.06069672 - Q.sj3_dr12) / 0.008100502   # +0.7%  sj3_dr12 < 0.0607
        - 0.006995424 * max(0.0, 846.1934 - Q.sum_pt_top20) / 22.83716   # -0.7%  sum_pt_top20 < 846.2
        + 0.005383728 * max(0.0, Q.LHA - 0.3719813) * max(0.0, Q.log_sum_pt - 6.856375) / 0.0004589561   # +0.5%  LHA > 0.372 and log_sum_pt > 6.856
        + 0.004390305 * max(0.0, Q.LHA - 0.3719813) * max(0.0, 0.003687605 - Q.lam2) / 4.619984e-06   # +0.4%  LHA > 0.372 and lam2 < 0.003688
        + 0.002656695 * max(0.0, Q.e2 - 0.06524004) / 0.0004075092   # +0.3%  e2 > 0.06524
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 34.47;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.47389 * (0.06347615
        - 0.4134869 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # -41.3%  log_sum_pt < 6.989
        + 0.3683101 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # +36.8%  sum_pt < 1085
        + 0.03334898 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # +3.3%  e3 < 0.0005178
        - 0.02284952 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # -2.3%  sum_z_dr > 0.0975
        - 0.01867952 * max(0.0, Q.sum_z_dr - 0.08589404) / 0.01080971   # -1.9%  sum_z_dr > 0.08589
        - 0.01797046 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -1.8%  psi_0p2 > 0.9087
        + 0.01788897 * max(0.0, Q.LHA - 0.3203321) / 0.01671666   # +1.8%  LHA > 0.3203
        + 0.01752796 * max(0.0, 1115.723 - Q.sum_pt) / 92.38491   # +1.8%  sum_pt < 1116
        - 0.01210645 * max(0.0, 26.0 - Q.n_dr_0p1_0p2) / 14.20854   # -1.2%  n_dr_0p1_0p2 < 26
        - 0.01082588 * max(0.0, 0.0001841806 - Q.e3) * max(0.0, 41.4375 - Q.pt_9) / 0.00190888   # -1.1%  e3 < 0.0001842 and pt_9 < 41.44
        + 0.01013193 * max(0.0, 41.4375 - Q.pt_9) / 15.12338   # +1.0%  pt_9 < 41.44
        + 0.008546407 * max(0.0, 1048.098 - Q.sum_pt_top50) / 44.36839   # +0.9%  sum_pt_top50 < 1048
        + 0.007266504 * max(0.0, Q.sum_z_dr - 0.09749958) * max(0.0, 1167.447 - Q.sum_pt) / 1.406455   # +0.7%  sum_z_dr > 0.0975 and sum_pt < 1167
        + 0.005593397 * max(0.0, Q.sum_z_dr2_top15 - 0.009962397) * max(0.0, 1225.842 - Q.sum_pt_top40) / 0.6413342   # +0.6%  sum_z_dr2_top15 > 0.009962 and sum_pt_top40 < 1226
        - 0.004685606 * max(0.0, Q.sum_z_dr2_top15 - 0.009962397) / 0.002311908   # -0.5%  sum_z_dr2_top15 > 0.009962
        + 0.00425636 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 6.811175 - Q.log_sum_pt) / 1.755002   # +0.4%  sum_pt < 1085 and log_sum_pt < 6.811
        + 0.004180392 * max(0.0, Q.e2 - 0.04082832) / 0.003398869   # +0.4%  e2 > 0.04083
        - 0.003841352 * max(0.0, Q.sum_pt - 1167.447) / 14.21193   # -0.4%  sum_pt > 1167
        + 0.002903594 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # +0.3%  sum_pt_top50 < 959.1
        + 0.002526734 * max(0.0, Q.sum_z_dr - 0.1564779) / 0.0006617163   # +0.3%  sum_z_dr > 0.1565
        - 0.002190963 * max(0.0, Q.sum_z_dr - 0.1564779) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.000215543   # -0.2%  sum_z_dr > 0.1565 and n_pt_above_50 > 5
        + 0.002104633 * max(0.0, Q.sum_pt_top30 - 1111.245) / 12.62573   # +0.2%  sum_pt_top30 > 1111
        - 0.00201808 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -0.2%  sum_z_dr > 0.1207
        + 0.001824221 * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 1.386277   # +0.2%  n_dr_0p1_0p2 > 21
        - 0.001774677 * max(0.0, Q.LHA - 0.4331369) / 0.001006142   # -0.2%  LHA > 0.4331
        - 0.001745493 * max(0.0, 23.0 - Q.n_pt_above_5) / 1.859217   # -0.2%  n_pt_above_5 < 23
        - 0.0006555428 * max(0.0, Q.sum_z_dr2_top15 - 0.009962397) * max(0.0, Q.soft7_z - 0.003627839) / 1.996498e-07   # -0.1%  sum_z_dr2_top15 > 0.009962 and soft7_z > 0.003628
        - 0.0005708374 * max(0.0, Q.sum_z_dr - 0.1402186) / 0.001871547   # -0.1%  sum_z_dr > 0.1402
        + 0.0001884847 * max(0.0, Q.sum_z_dr - 0.1402186) * max(0.0, Q.soft6_pt - 4.250195) / 9.168767e-05   # +0.0%  sum_z_dr > 0.1402 and soft6_pt > 4.25
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 23.35;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.34872 * (0.1111628
        - 0.1269493 * max(0.0, Q.sum_z_dr - 0.076787) / 0.01351025   # -12.7%  sum_z_dr > 0.07679
        - 0.1226842 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # -12.3%  tau1 < 0.1507
        - 0.1047477 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # -10.5%  LHA < 0.3332
        + 0.08435619 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # +8.4%  sd_rg < 0.3017
        + 0.07885964 * max(0.0, Q.sum_z_dr - 0.08068193) / 0.0122472   # +7.9%  sum_z_dr > 0.08068
        + 0.07044872 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # +7.0%  LHA < 0.3024
        - 0.06118697 * max(0.0, 0.1751567 - Q.tau1) / 0.09101058   # -6.1%  tau1 < 0.1752
        - 0.04987804 * max(0.0, Q.sum_z_dr - 0.08589404) / 0.01080971   # -5.0%  sum_z_dr > 0.08589
        + 0.04507062 * max(0.0, Q.sd_rg - 0.1596365) / 0.03553981   # +4.5%  sd_rg > 0.1596
        - 0.03298111 * max(0.0, 7.876005e-05 - Q.e3) / 3.594075e-05   # -3.3%  e3 < 7.876e-05
        + 0.02991638 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +3.0%  tau21_b2 < 0.3425
        - 0.02938742 * max(0.0, Q.sd_rg - 0.2042612) / 0.02046744   # -2.9%  sd_rg > 0.2043
        + 0.02280739 * max(0.0, Q.sd_rg - 0.15159) / 0.03939137   # +2.3%  sd_rg > 0.1516
        + 0.01978948 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +2.0%  e2 < 0.03481
        - 0.01919833 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 1156.659 - Q.sum_pt_top50) / 14.04661   # -1.9%  tau21_b2 < 0.3425 and sum_pt_top50 < 1157
        - 0.0115871 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # -1.2%  n_dr_0p1_0p2 < 17
        - 0.01055027 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -1.1%  sum_z_dr > 0.1207
        - 0.0101759 * max(0.0, 3.376709e-05 - Q.e3) / 8.295151e-06   # -1.0%  e3 < 3.377e-05
        + 0.009930955 * max(0.0, Q.z_dr_0_0p05 - 0.6283153) / 0.1153015   # +1.0%  z_dr_0_0p05 > 0.6283
        - 0.008708217 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # -0.9%  sum_z_dr > 0.0975
        + 0.008690904 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +0.9%  z_dr_0p1_0p2 < 0.1203
        + 0.008448182 * max(0.0, 0.05347848 - Q.dr_6) / 0.01424699   # +0.8%  dr_6 < 0.05348
        + 0.007109467 * max(0.0, Q.sum_z_dr2_top10 - 0.0001295334) / 0.006638188   # +0.7%  sum_z_dr2_top10 > 0.0001295
        + 0.00695836 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +0.7%  psi_0p3 > 0.9924
        + 0.00670664 * max(0.0, Q.sum_z_dr - 0.076787) * max(0.0, 0.4021783 - Q.max_dr) / 0.0004881945   # +0.7%  sum_z_dr > 0.07679 and max_dr < 0.4022
        - 0.00652941 * max(0.0, 0.2941033 - Q.LHA) / 0.05718399   # -0.7%  LHA < 0.2941
        - 0.006343117 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.05347848 - Q.dr_6) / 5.520862e-05   # -0.6%  psi_0p3 > 0.9924 and dr_6 < 0.05348
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 6.669;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.669006 * (-0.1157947
        + 0.1762165 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +17.6%  sum_pt > 907.9
        - 0.07776414 * max(0.0, Q.sum_pt_top40 - 1001.523) / 47.19068   # -7.8%  sum_pt_top40 > 1002
        - 0.07671855 * max(0.0, Q.sj2_dr - 0.1512157) / 0.05988199   # -7.7%  sj2_dr > 0.1512
        + 0.06835607 * max(0.0, Q.sum_pt_top30 - 886.3438) / 117.6357   # +6.8%  sum_pt_top30 > 886.3
        + 0.06757784 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # +6.8%  sum_z_dr2_top5 < 0.00833
        + 0.06614412 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +6.6%  sj2_dr > 0.2232
        - 0.05524408 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) / 3.639534   # -5.5%  n_dr_0p2_0p4 > 8
        - 0.05217436 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # -5.2%  tau1 < 0.07709
        - 0.05183968 * max(0.0, Q.log_sum_pt - 6.915514) / 0.04811413   # -5.2%  log_sum_pt > 6.916
        + 0.04907694 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +4.9%  z_dr_0p1_0p2 < 0.1203
        + 0.04305926 * max(0.0, 0.08068193 - Q.sum_z_dr) / 0.02368455   # +4.3%  sum_z_dr < 0.08068
        - 0.04236598 * max(0.0, Q.psi_0p1 - 0.707925) / 0.1360654   # -4.2%  psi_0p1 > 0.7079
        - 0.0335929 * max(0.0, 0.02647293 - Q.z_dr_0p2_0p4) / 0.01168948   # -3.4%  z_dr_0p2_0p4 < 0.02647
        + 0.02630348 * max(0.0, Q.psi_0p1 - 0.707925) * max(0.0, 0.1828389 - Q.dr_max_012) / 0.01897164   # +2.6%  psi_0p1 > 0.7079 and dr_max_012 < 0.1828
        + 0.02513764 * max(0.0, 0.007678544 - Q.sum_z_dr2_top10) / 0.00347358   # +2.5%  sum_z_dr2_top10 < 0.007679
        - 0.0202823 * max(0.0, 0.001155057 - Q.sum_z_dr2_top3) / 0.000329166   # -2.0%  sum_z_dr2_top3 < 0.001155
        - 0.01674706 * max(0.0, Q.sj2_dr - 0.2780918) / 0.009227968   # -1.7%  sj2_dr > 0.2781
        - 0.01660331 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.4299592 - Q.tau21_b2) / 0.001167326   # -1.7%  psi_0p3 > 0.9897 and tau21_b2 < 0.43
        + 0.01629384 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # +1.6%  log_sum_pt > 7.017
        - 0.005667105 * max(0.0, 0.5297852 - Q.soft5_pt) / 0.01859865   # -0.6%  soft5_pt < 0.5298
        + 0.004796701 * max(0.0, 0.0004140594 - Q.soft5_z) / 1.144764e-05   # +0.5%  soft5_z < 0.0004141
        + 0.004037841 * max(0.0, Q.psi_0p1 - 0.707925) * max(0.0, Q.dr1_11 - 0.2195171) / 0.000707107   # +0.4%  psi_0p1 > 0.7079 and dr1_11 > 0.2195
        - 0.004000257 * max(0.0, 0.08068193 - Q.sum_z_dr) * max(0.0, Q.dr1_11 - 0.1911348) / 0.0001794349   # -0.4%  sum_z_dr < 0.08068 and dr1_11 > 0.1911
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6825215861344538, 3.0775236869747897, 0.2240279411764706, 0.4905985819327731, 0.8305100840336135, 1.1364714285714286, 0.544603781512605, 0.4549508403361345, 1.2251704831932773, 1.0690913865546219, 1.1910849789915967, 0.6205046218487394, 0.7197281512605042, 1.5201491071428572, 0.2569828256302521, 0.30010892857142857]
T = [4.235173038668592, 2.6482388294380255, 4.363744370404412, 4.5006464055934865, 4.092694453289129]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +45%, n4 -17%, n9 +13%, n3 -9%, n5 -8%, n12 +4% ...
            + 0.4541614 * h[1] / H_AVG[1]
            - 0.171586 * h[4] / H_AVG[4]
            + 0.13016 * h[9] / H_AVG[9]
            - 0.08687932 * h[3] / H_AVG[3]
            - 0.08385663 * h[5] / H_AVG[5]
            + 0.03982984 * h[12] / H_AVG[12]
            + 0.02009229 * h[6] / H_AVG[6]
            + 0.009040145 * h[8] / H_AVG[8]
            - 0.004394319 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +23%, n1 -22%, n12 +9%, n11 -6%, n6 +4% ...
            - 0.333209 * h[4] / H_AVG[4]
            + 0.2270807 * h[9] / H_AVG[9]
            - 0.2178941 * h[1] / H_AVG[1]
            + 0.09342305 * h[12] / H_AVG[12]
            - 0.0585771 * h[11] / H_AVG[11]
            + 0.0449854 * h[6] / H_AVG[6]
            + 0.01057438 * h[2] / H_AVG[2]
            + 0.007228687 * h[8] / H_AVG[8]
            - 0.007027577 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +15%, n0 +12%, n11 +8%, n14 -8%, n12 -7% ...
            - 0.2456661 * h[8] / H_AVG[8]
            + 0.1505639 * h[5] / H_AVG[5]
            + 0.1173055 * h[0] / H_AVG[0]
            + 0.08442855 * h[11] / H_AVG[11]
            - 0.08097435 * h[14] / H_AVG[14]
            - 0.06700428 * h[12] / H_AVG[12]
            + 0.06542268 * h[4] / H_AVG[4]
            - 0.06516062 * h[7] / H_AVG[7]
            - 0.05359245 * h[9] / H_AVG[9]
            + 0.0491864 * h[3] / H_AVG[3]
            - 0.01289499 * h[15] / H_AVG[15]
            + 0.007800122 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -21%, n5 +17%, n6 -12%, n7 +9%, n12 +5% ...
            - 0.2552072 * h[8] / H_AVG[8]
            - 0.2085183 * h[0] / H_AVG[0]
            + 0.1736026 * h[5] / H_AVG[5]
            - 0.1172243 * h[6] / H_AVG[6]
            + 0.09160889 * h[7] / H_AVG[7]
            + 0.04997394 * h[12] / H_AVG[12]
            + 0.04769023 * h[3] / H_AVG[3]
            + 0.03750823 * h[15] / H_AVG[15]
            - 0.01866631 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +29%, n5 -13%, n12 -7%, n8 +6%, n4 +4% ...
            - 0.3366084 * h[13] / H_AVG[13]
            + 0.2864798 * h[10] / H_AVG[10]
            - 0.1301639 * h[5] / H_AVG[5]
            - 0.0659463 * h[12] / H_AVG[12]
            + 0.0631453 * h[8] / H_AVG[8]
            + 0.03804844 * h[4] / H_AVG[4]
            - 0.03126423 * h[7] / H_AVG[7]
            - 0.02749798 * h[15] / H_AVG[15]
            + 0.02084573 * h[0] / H_AVG[0]
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
