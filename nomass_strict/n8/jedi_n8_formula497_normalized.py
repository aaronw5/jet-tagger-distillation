"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.0%   (on for 89% of jets)
  neuron  9:  11.8%   (on for 72% of jets)
  neuron  7:  10.2%   (on for 64% of jets)
  neuron  3:   8.6%   (on for 31% of jets)
  neuron 10:   8.2%   (on for 82% of jets)
  neuron  2:   7.2%   (on for 89% of jets)
  neuron  5:   7.1%   (on for 72% of jets)
  neuron  6:   7.1%   (on for 52% of jets)
  neuron 11:   6.5%   (on for 72% of jets)
  neuron 14:   4.7%   (on for 33% of jets)
  neuron  0:   4.3%   (on for 45% of jets)
  neuron  1:   3.2%   (on for 61% of jets)
  neuron  4:   3.1%   (on for 47% of jets)
  neuron 15:   2.7%   (on for 34% of jets)
  neuron  8:   1.0%   (on for 37% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.6% (the network: 65.8%); same class as the network for 87.4% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.ptdr0_7                pT7 · ΔR(0, 7) [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.phi_0                  Δφ of particle 0
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
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.centroid_offset        distance of the pT centroid from the jet axis
  Q.pt_dispersion          √(Σ pTᵢ²) / Σ pTᵢ
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
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_4=pt[4],
        pt_5=pt[5],
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        ptdr0_7=pt[7] * math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        z_top3_slots=sum(pt[:3]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        phi_0=phi[0],
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
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 15;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.99857 * (-0.2206877
        + 0.1834989 * max(0.0, 0.0121 - Q.lam1) / 0.007398442   # +18.3%  lam1 < 0.0121
        + 0.1366844 * max(0.0, Q.LHA - 0.12) / 0.1281294   # +13.7%  LHA > 0.12
        + 0.1137316 * max(0.0, Q.sj3_dr_max - 0.145) / 0.06136014   # +11.4%  sj3_dr_max > 0.145
        + 0.1075808 * max(0.0, 0.000501 - Q.e3) / 0.0004408629   # +10.8%  e3 < 0.000501
        - 0.0939887 * max(0.0, Q.sj3_dr_max - 0.17) / 0.04746452   # -9.4%  sj3_dr_max > 0.17
        - 0.06579769 * max(0.0, 0.00626 - Q.lam1) * max(0.0, 1.47 - Q.n_dr_0p2_0p4) / 0.003854966   # -6.6%  lam1 < 0.00626 and n_dr_0p2_0p4 < 1.47
        - 0.0387183 * max(0.0, 745.0 - Q.sum_pt) * max(0.0, 0.000468 - Q.e3) / 0.02932925   # -3.9%  sum_pt < 745 and e3 < 0.000468
        - 0.03596222 * max(0.0, 0.00391 - Q.lam1) / 0.001379493   # -3.6%  lam1 < 0.00391
        - 0.03356909 * max(0.0, Q.centroid_offset - 0.00875) / 0.01011021   # -3.4%  centroid_offset > 0.00875
        - 0.02819669 * max(0.0, 0.0363 - Q.C2) / 0.01543467   # -2.8%  C2 < 0.0363
        - 0.02705474 * max(0.0, Q.sum_z_dr - 0.0816) / 0.00909826   # -2.7%  sum_z_dr > 0.0816
        + 0.02563569 * max(0.0, 875.0 - Q.sum_pt) * max(0.0, Q.centroid_offset - 0.0158) / 1.941913   # +2.6%  sum_pt < 875 and centroid_offset > 0.0158
        - 0.02418947 * Q.lam2 / 0.0005288738   # -2.4%  lam2
        + 0.01564562 * max(0.0, 0.106 - Q.planar_flow) / 0.03095804   # +1.6%  planar_flow < 0.106
        + 0.01395914 * max(0.0, 0.00567 - Q.lam1) * max(0.0, 725.0 - Q.sum_pt) / 0.1046836   # +1.4%  lam1 < 0.00567 and sum_pt < 725
        - 0.01224445 * max(0.0, 0.0172 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0179) / 3.725137e-05   # -1.2%  lam1 < 0.0172 and centroid_offset > 0.0179
        - 0.00968222 * max(0.0, 0.00764 - Q.sum_z_dr2_top3) * max(0.0, 0.0325 - Q.phi_0) / 0.0001455105   # -1.0%  sum_z_dr2_top3 < 0.00764 and phi_0 < 0.0325
        - 0.00786464 * max(0.0, Q.n_dr_0p2_0p4 - 0.397) / 0.2693113   # -0.8%  n_dr_0p2_0p4 > 0.397
        - 0.007105084 * max(0.0, 0.00554 - Q.lam1) * max(0.0, 1.33 - Q.D2) / 0.0001846899   # -0.7%  lam1 < 0.00554 and D2 < 1.33
        - 0.005887729 * max(0.0, Q.centroid_offset - 0.0261) / 0.003409557   # -0.6%  centroid_offset > 0.0261
        - 0.005673699 * max(0.0, Q.sum_pt - 887.0) / 14.6973   # -0.6%  sum_pt > 887
        + 0.004762964 * max(0.0, Q.zdr_0 - 0.0148) / 0.004960948   # +0.5%  zdr_0 > 0.0148
        + 0.002566205 * max(0.0, 0.266 - Q.planar_flow) * max(0.0, 0.0311 - Q.z_7) / 0.0001539576   # +0.3%  planar_flow < 0.266 and z_7 < 0.0311
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 18.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.56884 * (0.2003356
        - 0.1374055 * max(0.0, 726.0 - Q.sum_pt_top5) / 159.4663   # -13.7%  sum_pt_top5 < 726
        - 0.08259302 * max(0.0, 0.00892 - Q.lam1) * max(0.0, 0.000937 - Q.lam2) / 4.025346e-06   # -8.3%  lam1 < 0.00892 and lam2 < 0.000937
        - 0.07275449 * max(0.0, 0.107 - Q.sum_z_dr) / 0.05297908   # -7.3%  sum_z_dr < 0.107
        - 0.05828546 * max(0.0, 0.0629 - Q.z_7) / 0.01505276   # -5.8%  z_7 < 0.0629
        + 0.05556478 * max(0.0, 0.0635 - Q.sum_z_dr) / 0.02003444   # +5.6%  sum_z_dr < 0.0635
        + 0.04866818 * max(0.0, 6.56 - Q.log_sum_pt) / 0.1099406   # +4.9%  log_sum_pt < 6.56
        + 0.04755422 * Q.z_5 / 0.07008149   # +4.8%  z_5
        + 0.04594812 * max(0.0, 0.1 - Q.tau1) / 0.0461191   # +4.6%  tau1 < 0.1
        - 0.03877499 * max(0.0, Q.log_sum_pt - 6.58) * max(0.0, 0.00794 - Q.sum_z_dr2_top3) / 0.000521744   # -3.9%  log_sum_pt > 6.58 and sum_z_dr2_top3 < 0.00794
        + 0.03776522 * max(0.0, 0.0655 - Q.z_7) * max(0.0, 0.00859 - Q.sum_z_dr2_top3) / 0.0001057702   # +3.8%  z_7 < 0.0655 and sum_z_dr2_top3 < 0.00859
        + 0.03077597 * max(0.0, Q.log_sum_pt - 6.72) / 0.02992011   # +3.1%  log_sum_pt > 6.72
        - 0.02891738 * max(0.0, 0.00537 - Q.lam1) / 0.002156475   # -2.9%  lam1 < 0.00537
        - 0.02885737 * max(0.0, Q.sum_pt - 839.0) / 24.69345   # -2.9%  sum_pt > 839
        - 0.02480638 * max(0.0, Q.log_sum_pt - 6.32) * max(0.0, 0.00352 - Q.sum_z_dr2_top2) / 0.0005406406   # -2.5%  log_sum_pt > 6.32 and sum_z_dr2_top2 < 0.00352
        + 0.02275851 * max(0.0, Q.sum_pt_top5 - 733.0) / 25.30534   # +2.3%  sum_pt_top5 > 733
        + 0.02116311 * max(0.0, 0.00117 - Q.lam2) / 0.0009492135   # +2.1%  lam2 < 0.00117
        - 0.01983185 * max(0.0, 556.0 - Q.sum_pt_top5) / 54.47551   # -2.0%  sum_pt_top5 < 556
        - 0.01973349 * max(0.0, 0.0104 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.76) / 0.00252709   # -2.0%  lam1 < 0.0104 and n_pt_above_50 > 5.76
        + 0.01916621 * max(0.0, Q.pt_7 - 33.7) / 4.902126   # +1.9%  pt_7 > 33.7
        - 0.0165915 * max(0.0, Q.pt_7 - 32.6) * max(0.0, 8.02e-05 - Q.e3) / 0.0003176134   # -1.7%  pt_7 > 32.6 and e3 < 8.02e-05
        - 0.01536242 * max(0.0, Q.log_sum_pt - 6.29) * max(0.0, 0.104 - Q.dr_2) / 0.01794103   # -1.5%  log_sum_pt > 6.29 and dr_2 < 0.104
        + 0.01485989 * max(0.0, Q.n_pt_above_50 - 5.82) * max(0.0, 8.49e-05 - Q.e3) / 2.378715e-05   # +1.5%  n_pt_above_50 > 5.82 and e3 < 8.49e-05
        + 0.01389934 * max(0.0, 0.159 - Q.sj3_dr_max) / 0.04404345   # +1.4%  sj3_dr_max < 0.159
        - 0.01319936 * max(0.0, 0.0467 - Q.z_6) / 0.004147156   # -1.3%  z_6 < 0.0467
        + 0.01144354 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.123 - Q.dr_max_012) / 0.006498267   # +1.1%  log_sum_pt > 6.6 and dr_max_012 < 0.123
        + 0.0113596 * max(0.0, 499.0 - Q.sum_pt) / 7.031153   # +1.1%  sum_pt < 499
        + 0.01106865 * max(0.0, Q.pt_6 - 37.0) / 6.760919   # +1.1%  pt_6 > 37
        - 0.01021746 * max(0.0, 6.21 - Q.log_sum_pt) / 0.01567986   # -1.0%  log_sum_pt < 6.21
        - 0.009488297 * max(0.0, 0.0196 - Q.centroid_offset) / 0.007529346   # -0.9%  centroid_offset < 0.0196
        + 0.00889407 * max(0.0, 0.0507 - Q.z_6) * max(0.0, 0.00348 - Q.sum_z_dr2_top2) / 1.35371e-05   # +0.9%  z_6 < 0.0507 and sum_z_dr2_top2 < 0.00348
        + 0.008046639 * max(0.0, Q.pt_7 - 42.0) / 1.766156   # +0.8%  pt_7 > 42
        + 0.004825833 * max(0.0, 0.00913 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0198) / 9.719103e-06   # +0.5%  lam1 < 0.00913 and centroid_offset > 0.0198
        - 0.004145437 * max(0.0, 0.000662 - Q.lam1) / 0.0001282933   # -0.4%  lam1 < 0.000662
        - 0.003374531 * max(0.0, Q.sj3_dr_min - 0.0229) / 0.02848233   # -0.3%  sj3_dr_min > 0.0229
        - 0.001899197 * max(0.0, Q.pt_7 - 53.6) / 0.3235403   # -0.2%  pt_7 > 53.6
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 14.21;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.20968 * (0.2118274
        + 0.1966506 * max(0.0, 6.72 - Q.log_sum_pt) / 0.2069883   # +19.7%  log_sum_pt < 6.72
        - 0.1244751 * max(0.0, Q.z_7 - 0.0247) / 0.02834536   # -12.4%  z_7 > 0.0247
        - 0.09705896 * max(0.0, 53.7 - Q.pt_7) / 19.37046   # -9.7%  pt_7 < 53.7
        - 0.09221725 * max(0.0, 840.0 - Q.sum_pt) / 148.4006   # -9.2%  sum_pt < 840
        + 0.07349318 * max(0.0, Q.pt_7 - 19.4) / 15.72763   # +7.3%  pt_7 > 19.4
        - 0.07312828 * max(0.0, Q.LHA - 0.12) / 0.1281294   # -7.3%  LHA > 0.12
        - 0.05925913 * max(0.0, Q.log_sum_pt - 6.08) * max(0.0, 9.53e-05 - Q.mean_phi2) / 7.725259e-06   # -5.9%  log_sum_pt > 6.08 and mean_phi2 < 9.53e-05
        + 0.05864269 * max(0.0, 0.00716 - Q.lam1) * max(0.0, 0.255 - Q.max_dr) / 0.0006218611   # +5.9%  lam1 < 0.00716 and max_dr < 0.255
        + 0.0577935 * max(0.0, 0.0121 - Q.lam1) / 0.007398442   # +5.8%  lam1 < 0.0121
        + 0.05583073 * max(0.0, 9.69e-05 - Q.mean_phi2) / 1.196586e-05   # +5.6%  mean_phi2 < 9.69e-05
        - 0.02320754 * max(0.0, 0.0643 - Q.C2) / 0.03893409   # -2.3%  C2 < 0.0643
        - 0.02013241 * max(0.0, Q.log_sum_pt - 6.1) * max(0.0, 0.00022 - Q.lam2) / 7.028873e-05   # -2.0%  log_sum_pt > 6.1 and lam2 < 0.00022
        - 0.01648073 * max(0.0, 0.00752 - Q.lam1) * max(0.0, 62.6 - Q.pt_6) / 0.08423953   # -1.6%  lam1 < 0.00752 and pt_6 < 62.6
        + 0.01346849 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.107 - Q.absphi_0) / 0.0003716173   # +1.3%  log_sum_pt > 6.9 and absphi_0 < 0.107
        - 0.009669651 * max(0.0, Q.sum_pt - 988.0) * max(0.0, 0.104 - Q.absphi_0) / 0.4138634   # -1.0%  sum_pt > 988 and absphi_0 < 0.104
        - 0.008930076 * max(0.0, 0.116 - Q.sum_z_dr) * max(0.0, Q.centroid_offset - 0.0173) / 0.0001580243   # -0.9%  sum_z_dr < 0.116 and centroid_offset > 0.0173
        + 0.007563324 * max(0.0, Q.sum_pt - 867.0) * max(0.0, 4.31 - Q.D2_b2) / 40.10165   # +0.8%  sum_pt > 867 and D2_b2 < 4.31
        - 0.004332618 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 9.61e-05 - Q.mean_phi2) / 6.095556e-07   # -0.4%  log_sum_pt < 6.7 and mean_phi2 < 9.61e-05
        - 0.002734446 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, 4.72 - Q.D2_b2) / 0.008520966   # -0.3%  log_sum_pt > 6.91 and D2_b2 < 4.72
        + 0.002558186 * max(0.0, Q.sum_pt - 927.0) * max(0.0, 9e-05 - Q.mean_phi2) / 0.0003054706   # +0.3%  sum_pt > 927 and mean_phi2 < 9e-05
        - 0.002373169 * max(0.0, 3.25e-05 - Q.mean_eta2) / 1.399252e-06   # -0.2%  mean_eta2 < 3.25e-05
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 8.922;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.921586 * (-0.09392949
        - 0.1836893 * max(0.0, 0.0767 - Q.sum_z_dr) / 0.02840207   # -18.4%  sum_z_dr < 0.0767
        + 0.1069017 * max(0.0, Q.lam1 - 0.00541) / 0.002686571   # +10.7%  lam1 > 0.00541
        + 0.0764092 * max(0.0, Q.sj3_dr_max - 0.185) / 0.04057686   # +7.6%  sj3_dr_max > 0.185
        - 0.06550537 * max(0.0, Q.lam1 - 0.0109) / 0.001398114   # -6.6%  lam1 > 0.0109
        + 0.05839657 * max(0.0, Q.sj2_dr - 0.181) / 0.02960171   # +5.8%  sj2_dr > 0.181
        + 0.04970443 * max(0.0, 0.107 - Q.tau1) / 0.05174356   # +5.0%  tau1 < 0.107
        + 0.04838799 * max(0.0, 0.186 - Q.sj3_dr_max) / 0.05794599   # +4.8%  sj3_dr_max < 0.186
        + 0.04626231 * max(0.0, Q.LHA - 0.302) / 0.0183437   # +4.6%  LHA > 0.302
        + 0.03999997 * max(0.0, Q.LHA - 0.331) * max(0.0, Q.eccentricity - 0.66) / 0.002347784   # +4.0%  LHA > 0.331 and eccentricity > 0.66
        - 0.0386568 * max(0.0, Q.LHA - 0.324) / 0.01277333   # -3.9%  LHA > 0.324
        + 0.0385817 * max(0.0, 0.00766 - Q.lam1) / 0.00373735   # +3.9%  lam1 < 0.00766
        - 0.03494795 * max(0.0, Q.sj3_dr_max - 0.269) / 0.01751636   # -3.5%  sj3_dr_max > 0.269
        + 0.02918127 * max(0.0, Q.sj2_dr - 0.136) * max(0.0, Q.eccentricity - 0.936) / 0.001771042   # +2.9%  sj2_dr > 0.136 and eccentricity > 0.936
        - 0.02665112 * max(0.0, Q.eccentricity - 0.926) / 0.03297785   # -2.7%  eccentricity > 0.926
        + 0.02620725 * max(0.0, Q.planar_flow - 0.0924) / 0.2145048   # +2.6%  planar_flow > 0.0924
        - 0.0243638 * max(0.0, Q.sj2_dr - 0.228) / 0.01598263   # -2.4%  sj2_dr > 0.228
        - 0.0215976 * max(0.0, Q.sj2_dr - 0.148) * max(0.0, Q.log_sum_pt - 6.11) / 0.01566544   # -2.2%  sj2_dr > 0.148 and log_sum_pt > 6.11
        - 0.02086998 * max(0.0, Q.sum_z_dr - 0.11) * max(0.0, Q.eccentricity - 0.686) / 0.0007538191   # -2.1%  sum_z_dr > 0.11 and eccentricity > 0.686
        + 0.02013657 * max(0.0, Q.tau1 - 0.0939) / 0.01095428   # +2.0%  tau1 > 0.0939
        - 0.01680117 * max(0.0, Q.sum_z_dr - 0.0444) * max(0.0, Q.n_pt_above_50 - 2.21) / 0.0660322   # -1.7%  sum_z_dr > 0.0444 and n_pt_above_50 > 2.21
        - 0.01344003 * max(0.0, 1.23 - Q.N3) * max(0.0, Q.log_sum_pt - 6.1) / 0.01600886   # -1.3%  N3 < 1.23 and log_sum_pt > 6.1
        + 0.007140712 * max(0.0, 1.22 - Q.N3) * max(0.0, 0.0817 - Q.z_7) / 0.001034196   # +0.7%  N3 < 1.22 and z_7 < 0.0817
        + 0.006167232 * max(0.0, Q.sum_z_dr - 0.132) / 0.001965053   # +0.6%  sum_z_dr > 0.132
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 39.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 39.64239 * (-0.07920814
        + 0.1435437 * max(0.0, 0.2 - Q.sj3_dr_min) / 0.1580671   # +14.4%  sj3_dr_min < 0.2
        - 0.1323714 * max(0.0, 0.21 - Q.sj3_dr_min) * max(0.0, 0.00328 - Q.lam2) / 0.0005247519   # -13.2%  sj3_dr_min < 0.21 and lam2 < 0.00328
        - 0.127612 * max(0.0, 0.00827 - Q.lam1) / 0.004215706   # -12.8%  lam1 < 0.00827
        + 0.08923701 * max(0.0, 0.00345 - Q.lam2) / 0.003076147   # +8.9%  lam2 < 0.00345
        + 0.07959985 * max(0.0, 0.247 - Q.max_dr) / 0.1282735   # +8.0%  max_dr < 0.247
        + 0.05872462 * max(0.0, 0.0761 - Q.sum_z_dr) / 0.02798058   # +5.9%  sum_z_dr < 0.0761
        + 0.05109303 * max(0.0, Q.max_dr - 0.113) / 0.0397147   # +5.1%  max_dr > 0.113
        - 0.02664002 * max(0.0, 0.00593 - Q.lam1) / 0.002502545   # -2.7%  lam1 < 0.00593
        - 0.02488301 * max(0.0, 0.142 - Q.sj3_dr_max) / 0.03694464   # -2.5%  sj3_dr_max < 0.142
        - 0.0246194 * max(0.0, 6.59 - Q.log_sum_pt) / 0.1251246   # -2.5%  log_sum_pt < 6.59
        + 0.01937966 * max(0.0, 0.0343 - Q.C2) / 0.01394294   # +1.9%  C2 < 0.0343
        - 0.01850467 * max(0.0, 0.000416 - Q.lam2) / 0.0002888069   # -1.9%  lam2 < 0.000416
        + 0.0180903 * max(0.0, Q.planar_flow - 0.0231) / 0.2607792   # +1.8%  planar_flow > 0.0231
        - 0.01752151 * max(0.0, 0.0149 - Q.sum_z_dr) / 0.001335759   # -1.8%  sum_z_dr < 0.0149
        - 0.01546288 * max(0.0, Q.sj2_dr - 0.163) / 0.03737718   # -1.5%  sj2_dr > 0.163
        + 0.015225 * max(0.0, Q.sum_pt - 722.0) / 67.28601   # +1.5%  sum_pt > 722
        + 0.01365547 * max(0.0, 0.564 - Q.pt_dispersion) / 0.1282785   # +1.4%  pt_dispersion < 0.564
        - 0.01259965 * max(0.0, Q.log_sum_pt - 6.3) * max(0.0, 6.6 - Q.n_dr_0p05_0p1) / 1.349947   # -1.3%  log_sum_pt > 6.3 and n_dr_0p05_0p1 < 6.6
        - 0.01243974 * max(0.0, Q.C2 - 0.0338) / 0.008302037   # -1.2%  C2 > 0.0338
        + 0.01153858 * max(0.0, 0.0818 - Q.dr_0) * max(0.0, Q.centroid_offset - 0.00691) / 0.0002394853   # +1.2%  dr_0 < 0.0818 and centroid_offset > 0.00691
        - 0.00976752 * max(0.0, 0.0662 - Q.z_7) / 0.01736358   # -1.0%  z_7 < 0.0662
        - 0.008769571 * max(0.0, Q.sj3_dr_max - 0.241) * max(0.0, Q.log_sum_pt - 6.11) / 0.006485948   # -0.9%  sj3_dr_max > 0.241 and log_sum_pt > 6.11
        + 0.008251583 * max(0.0, Q.D2 - 0.937) / 0.7384029   # +0.8%  D2 > 0.937
        - 0.00815682 * max(0.0, 0.0161 - Q.zdr_0) / 0.005536915   # -0.8%  zdr_0 < 0.0161
        + 0.007668187 * max(0.0, 0.0773 - Q.sj3_dr_max) / 0.01427161   # +0.8%  sj3_dr_max < 0.0773
        + 0.007614454 * Q.n_dr_0p1_0p2 / 1.372069   # +0.8%  n_dr_0p1_0p2
        - 0.007147008 * max(0.0, 0.00378 - Q.lam1) / 0.001317788   # -0.7%  lam1 < 0.00378
        - 0.006063599 * max(0.0, 0.259 - Q.N2) * max(0.0, 866.0 - Q.sum_pt) / 11.44646   # -0.6%  N2 < 0.259 and sum_pt < 866
        - 0.004872363 * max(0.0, Q.e3 - 3.15e-05) / 5.998514e-05   # -0.5%  e3 > 3.15e-05
        - 0.003699041 * max(0.0, Q.max_dr - 0.247) / 0.004970809   # -0.4%  max_dr > 0.247
        - 0.00281158 * max(0.0, 24.3 - Q.pt_7) / 1.092723   # -0.3%  pt_7 < 24.3
        + 0.002240992 * max(0.0, Q.eccentricity - 0.72) * max(0.0, Q.n_pt_above_50 - 6.13) / 0.05047629   # +0.2%  eccentricity > 0.72 and n_pt_above_50 > 6.13
        + 0.001710254 * max(0.0, 365.0 - Q.sum_pt_top5) / 4.741157   # +0.2%  sum_pt_top5 < 365
        - 0.001493073 * max(0.0, Q.sj2_dr - 0.22) * max(0.0, 0.00183 - Q.C2_b2) / 7.716947e-06   # -0.1%  sj2_dr > 0.22 and C2_b2 < 0.00183
        + 0.00131125 * max(0.0, Q.sj3_dr_max - 0.215) * max(0.0, 0.0638 - Q.sj2_zsoft) / 9.572945e-05   # +0.1%  sj3_dr_max > 0.215 and sj2_zsoft < 0.0638
        + 0.001155453 * max(0.0, -0.0226 - Q.mean_eta) / 0.000972504   # +0.1%  mean_eta < -0.0226
        - 0.0008997434 * max(0.0, Q.max_dr - 0.106) * max(0.0, Q.n_pt_above_50 - 6.19) / 0.005599369   # -0.1%  max_dr > 0.106 and n_pt_above_50 > 6.19
        + 0.0008290844 * max(0.0, 0.00822 - Q.lam1) * max(0.0, Q.lam2 - 0.000406) / 9.959664e-08   # +0.1%  lam1 < 0.00822 and lam2 > 0.000406
        + 0.0007743815 * max(0.0, 0.000141 - Q.lam1) / 1.218188e-05   # +0.1%  lam1 < 0.000141
        + 0.0007536778 * max(0.0, 0.29 - Q.N2) * max(0.0, -0.0194 - Q.mean_phi) / 0.0001067057   # +0.1%  N2 < 0.29 and mean_phi < -0.0194
        - 0.0006606375 * max(0.0, Q.sum_pt_top3 - 789.0) / 4.372162   # -0.1%  sum_pt_top3 > 789
        - 0.0006081303 * max(0.0, 0.0759 - Q.dr_0) * max(0.0, Q.centroid_offset - 0.0285) / 2.008978e-05   # -0.1%  dr_0 < 0.0759 and centroid_offset > 0.0285
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 3.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.179706 * (0.3207843
        + 0.2489925 * max(0.0, 0.00242 - Q.lam1) * max(0.0, 0.0214 - Q.centroid_offset) / 9.822866e-06   # +24.9%  lam1 < 0.00242 and centroid_offset < 0.0214
        - 0.1953385 * max(0.0, 0.0383 - Q.centroid_offset) / 0.02275161   # -19.5%  centroid_offset < 0.0383
        + 0.1891319 * max(0.0, 0.0361 - Q.e2) * max(0.0, 0.0713 - Q.z_7) / 0.0003879897   # +18.9%  e2 < 0.0361 and z_7 < 0.0713
        - 0.1448278 * max(0.0, 767.0 - Q.sum_pt) / 98.61023   # -14.5%  sum_pt < 767
        + 0.0634162 * max(0.0, 0.0537 - Q.z_6) * max(0.0, 36.2 - Q.pt1_dr01) / 0.2089585   # +6.3%  z_6 < 0.0537 and pt1_dr01 < 36.2
        + 0.05973196 * max(0.0, 0.0295 - Q.z_7) / 0.001531694   # +6.0%  z_7 < 0.0295
        - 0.03851433 * max(0.0, 0.177 - Q.LHA) * max(0.0, 6.86 - Q.log_sum_pt) / 0.002605623   # -3.9%  LHA < 0.177 and log_sum_pt < 6.86
        + 0.02264714 * max(0.0, 0.0114 - Q.sum_z_dr) * max(0.0, Q.sum_pt_top5 - 598.0) / 0.1469617   # +2.3%  sum_z_dr < 0.0114 and sum_pt_top5 > 598
        - 0.01840206 * max(0.0, Q.log_sum_pt - 6.83) * max(0.0, 3.49e-05 - Q.e3) / 2.840444e-07   # -1.8%  log_sum_pt > 6.83 and e3 < 3.49e-05
        + 0.01341161 * max(0.0, 673.0 - Q.sum_pt) * max(0.0, 0.00688 - Q.mean_eta2) / 0.1627671   # +1.3%  sum_pt < 673 and mean_eta2 < 0.00688
        - 0.005586022 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, 0.0118 - Q.zdr_0) / 2.283022e-05   # -0.6%  log_sum_pt > 6.91 and zdr_0 < 0.0118
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 25.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.77548 * (-0.009039599
        + 0.1368802 * max(0.0, 0.321 - Q.LHA) / 0.09164031   # +13.7%  LHA < 0.321
        - 0.1158489 * max(0.0, 0.0575 - Q.sum_z_dr) / 0.01677562   # -11.6%  sum_z_dr < 0.0575
        + 0.1054576 * max(0.0, Q.LHA - 0.257) / 0.03653523   # +10.5%  LHA > 0.257
        - 0.1048325 * max(0.0, Q.LHA - 0.141) / 0.1111978   # -10.5%  LHA > 0.141
        - 0.09731456 * max(0.0, 0.0816 - Q.sum_z_dr) / 0.031994   # -9.7%  sum_z_dr < 0.0816
        + 0.0703232 * max(0.0, 0.00398 - Q.lam2) / 0.003582241   # +7.0%  lam2 < 0.00398
        + 0.05007125 * max(0.0, 0.185 - Q.sj3_dr_max) / 0.05736046   # +5.0%  sj3_dr_max < 0.185
        - 0.04942657 * max(0.0, 0.17 - Q.max_dr) / 0.06499968   # -4.9%  max_dr < 0.17
        + 0.04099259 * max(0.0, 0.0605 - Q.tau1) / 0.02051658   # +4.1%  tau1 < 0.0605
        + 0.03534575 * max(0.0, 6.62 - Q.log_sum_pt) / 0.141688   # +3.5%  log_sum_pt < 6.62
        - 0.03500517 * max(0.0, Q.z_7 - 0.0388) / 0.01696006   # -3.5%  z_7 > 0.0388
        - 0.02437407 * max(0.0, Q.tau1 - 0.0615) / 0.02292896   # -2.4%  tau1 > 0.0615
        - 0.0240846 * max(0.0, Q.sum_z_dr - 0.0957) / 0.006360575   # -2.4%  sum_z_dr > 0.0957
        + 0.02047931 * max(0.0, Q.pt_7 - 28.5) / 8.171268   # +2.0%  pt_7 > 28.5
        - 0.0163465 * max(0.0, Q.eccentricity - 0.864) / 0.07417938   # -1.6%  eccentricity > 0.864
        + 0.01186594 * max(0.0, Q.sj2_dr - 0.16) * max(0.0, 0.108 - Q.C2) / 0.002025498   # +1.2%  sj2_dr > 0.16 and C2 < 0.108
        + 0.01177743 * max(0.0, Q.tau1 - 0.101) / 0.009283456   # +1.2%  tau1 > 0.101
        + 0.008820337 * max(0.0, Q.centroid_offset - 0.0271) / 0.003206607   # +0.9%  centroid_offset > 0.0271
        - 0.00707136 * max(0.0, Q.centroid_offset - 0.0249) * max(0.0, 5.27 - Q.n_dr_0_0p05) / 0.01557844   # -0.7%  centroid_offset > 0.0249 and n_dr_0_0p05 < 5.27
        + 0.006717349 * max(0.0, Q.centroid_offset - 0.0101) * max(0.0, 0.0132 - Q.mean_phi2) / 7.367782e-05   # +0.7%  centroid_offset > 0.0101 and mean_phi2 < 0.0132
        - 0.005415275 * max(0.0, Q.sum_pt_top5 - 765.0) / 18.83688   # -0.5%  sum_pt_top5 > 765
        + 0.004504939 * max(0.0, 0.0396 - Q.z_7) / 0.003819637   # +0.5%  z_7 < 0.0396
        + 0.003742 * max(0.0, 34.3 - Q.pt_6) * max(0.0, 728.0 - Q.sum_pt_top3) / 275.5767   # +0.4%  pt_6 < 34.3 and sum_pt_top3 < 728
        - 0.003375273 * max(0.0, Q.lam2 - 0.00088) / 0.0003385186   # -0.3%  lam2 > 0.00088
        + 0.00325047 * max(0.0, Q.sj3_dr_max - 0.176) / 0.04456512   # +0.3%  sj3_dr_max > 0.176
        + 0.002314063 * max(0.0, Q.centroid_offset - -0.000127) * max(0.0, 0.918 - Q.eccentricity) / 0.00114046   # +0.2%  centroid_offset > -0.000127 and eccentricity < 0.918
        + 0.001411324 * max(0.0, Q.sum_pt - 979.0) / 4.896035   # +0.1%  sum_pt > 979
        + 0.001293378 * max(0.0, Q.centroid_offset - 0.0174) * max(0.0, Q.pt_2 - 66.7) / 0.08251844   # +0.1%  centroid_offset > 0.0174 and pt_2 > 66.7
        + 0.001066748 * max(0.0, 26.4 - Q.pt_5) / 0.3856372   # +0.1%  pt_5 < 26.4
        - 0.000591359 * max(0.0, Q.sj2_dr - 0.108) * max(0.0, Q.mean_eta - 0.0256) / 0.0001163554   # -0.1%  sj2_dr > 0.108 and mean_eta > 0.0256
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 27.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.41343 * (0.02309087
        - 0.09064117 * max(0.0, Q.lam1 - 0.00734) / 0.002070655   # -9.1%  lam1 > 0.00734
        - 0.07538595 * max(0.0, Q.sum_z_dr - 0.0818) * max(0.0, 2.88 - Q.D2) / 0.01828839   # -7.5%  sum_z_dr > 0.0818 and D2 < 2.88
        + 0.06729572 * max(0.0, 0.0848 - Q.tau1) / 0.03507237   # +6.7%  tau1 < 0.0848
        + 0.06372948 * max(0.0, 0.0511 - Q.centroid_offset) / 0.03466357   # +6.4%  centroid_offset < 0.0511
        + 0.06260191 * max(0.0, Q.LHA - 0.317) / 0.01430111   # +6.3%  LHA > 0.317
        - 0.05895485 * max(0.0, Q.sum_z_dr - 0.0886) / 0.007587581   # -5.9%  sum_z_dr > 0.0886
        + 0.05790293 * max(0.0, Q.tau1 - 0.0573) / 0.02491865   # +5.8%  tau1 > 0.0573
        - 0.05539297 * max(0.0, 0.00461 - Q.lam1) / 0.001731484   # -5.5%  lam1 < 0.00461
        + 0.05024865 * max(0.0, Q.lam1 - 0.00798) * max(0.0, 100.0 - Q.pt_4) / 0.09434849   # +5.0%  lam1 > 0.00798 and pt_4 < 100
        + 0.04845266 * max(0.0, Q.lam1 - 0.00309) * max(0.0, 0.252 - Q.planar_flow) / 0.0004743763   # +4.8%  lam1 > 0.00309 and planar_flow < 0.252
        - 0.04710018 * max(0.0, 0.0315 - Q.centroid_offset) * max(0.0, 1.14 - Q.n_dr_0p2_0p4) / 0.017034   # -4.7%  centroid_offset < 0.0315 and n_dr_0p2_0p4 < 1.14
        + 0.03909928 * max(0.0, 0.165 - Q.sj3_dr_max) / 0.04680547   # +3.9%  sj3_dr_max < 0.165
        - 0.03759015 * max(0.0, 0.0556 - Q.sum_z_dr) / 0.01580483   # -3.8%  sum_z_dr < 0.0556
        + 0.03267319 * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.8069227   # +3.3%  n_dr_0p2_0p4 < 1
        - 0.03266673 * max(0.0, Q.lam1 - 0.00308) * max(0.0, Q.D2 - -0.125) / 0.003778511   # -3.3%  lam1 > 0.00308 and D2 > -0.125
        + 0.03111078 * max(0.0, Q.tau1 - 0.103) / 0.008874644   # +3.1%  tau1 > 0.103
        - 0.02204747 * max(0.0, 0.262 - Q.planar_flow) * max(0.0, Q.lam1 - 0.008) / 0.0002048803   # -2.2%  planar_flow < 0.262 and lam1 > 0.008
        + 0.01793587 * max(0.0, Q.sum_z_dr - 0.128) / 0.002308375   # +1.8%  sum_z_dr > 0.128
        - 0.01683022 * max(0.0, 0.000344 - Q.lam2) / 0.0002295393   # -1.7%  lam2 < 0.000344
        - 0.01444997 * max(0.0, 0.0954 - Q.max_dr) / 0.0209589   # -1.4%  max_dr < 0.0954
        - 0.01146827 * max(0.0, Q.lam2 - 0.0011) / 0.0003150146   # -1.1%  lam2 > 0.0011
        + 0.01055539 * max(0.0, Q.C2 - 0.0303) / 0.009304163   # +1.1%  C2 > 0.0303
        - 0.009658084 * max(0.0, 0.000137 - Q.e3) * max(0.0, 37.3 - Q.pt_7) / 0.0006602524   # -1.0%  e3 < 0.000137 and pt_7 < 37.3
        + 0.008305807 * max(0.0, 0.197 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.46) / 0.01198372   # +0.8%  planar_flow < 0.197 and log_sum_pt > 6.46
        - 0.007148017 * max(0.0, Q.lam1 - 0.00186) * max(0.0, 0.191 - Q.sj3_dr_max) / 2.007702e-05   # -0.7%  lam1 > 0.00186 and sj3_dr_max < 0.191
        + 0.00696603 * max(0.0, Q.sum_z_dr2_top3 - 0.0111) / 0.001491897   # +0.7%  sum_z_dr2_top3 > 0.0111
        - 0.006543302 * max(0.0, Q.sj3_dr_max - 0.225) / 0.02734365   # -0.7%  sj3_dr_max > 0.225
        - 0.006145656 * max(0.0, 0.188 - Q.planar_flow) * max(0.0, 0.237 - Q.max_dr) / 0.00682079   # -0.6%  planar_flow < 0.188 and max_dr < 0.237
        - 0.003558464 * max(0.0, Q.n_dr_0p2_0p4 - 1.04) / 0.1491586   # -0.4%  n_dr_0p2_0p4 > 1.04
        - 0.002230389 * max(0.0, -0.0118 - Q.mean_eta) / 0.002152909   # -0.2%  mean_eta < -0.0118
        - 0.001875666 * max(0.0, Q.zdr_0 - 0.0244) / 0.002160439   # -0.2%  zdr_0 > 0.0244
        + 0.001494019 * max(0.0, 0.00806 - Q.lam1) * max(0.0, -0.00825 - Q.mean_eta) / 6.040736e-06   # +0.1%  lam1 < 0.00806 and mean_eta < -0.00825
        - 0.001194171 * max(0.0, Q.zdr_1 - 0.0246) / 0.0005919771   # -0.1%  zdr_1 > 0.0246
        - 0.0007465895 * max(0.0, Q.zdr_2 - 0.0197) / 0.0005142357   # -0.1%  zdr_2 > 0.0197
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 29.58;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.57579 * (0.09196711
        - 0.1386574 * max(0.0, Q.sum_z_dr - 0.0218) / 0.03981459   # -13.9%  sum_z_dr > 0.0218
        + 0.07117826 * max(0.0, Q.LHA - 0.154) / 0.1012093   # +7.1%  LHA > 0.154
        + 0.05871978 * max(0.0, Q.lam1 - 0.0045) / 0.003090185   # +5.9%  lam1 > 0.0045
        - 0.04816042 * max(0.0, Q.sum_pt - 714.0) / 71.21911   # -4.8%  sum_pt > 714
        - 0.04423089 * max(0.0, 0.313 - Q.LHA) / 0.08550088   # -4.4%  LHA < 0.313
        + 0.04392695 * max(0.0, Q.log_sum_pt - 6.57) / 0.08778205   # +4.4%  log_sum_pt > 6.57
        + 0.04099681 * max(0.0, 0.0073 - Q.lam1) / 0.003464323   # +4.1%  lam1 < 0.0073
        - 0.03991965 * max(0.0, Q.lam1 - 0.00773) / 0.001977647   # -4.0%  lam1 > 0.00773
        - 0.0398444 * max(0.0, 0.016 - Q.lam1) / 0.01081128   # -4.0%  lam1 < 0.016
        + 0.02812863 * max(0.0, Q.sum_z_dr - 0.0828) / 0.008812779   # +2.8%  sum_z_dr > 0.0828
        - 0.02051271 * max(0.0, 6.73 - Q.log_sum_pt) / 0.2143744   # -2.1%  log_sum_pt < 6.73
        + 0.0191259 * max(0.0, 0.16 - Q.sj3_dr_max) * max(0.0, 0.0018 - Q.sum_z_dr2_top2) / 6.264269e-05   # +1.9%  sj3_dr_max < 0.16 and sum_z_dr2_top2 < 0.0018
        - 0.01766491 * max(0.0, Q.LHA - 0.257) / 0.03653523   # -1.8%  LHA > 0.257
        + 0.01741765 * max(0.0, 0.00696 - Q.lam1) * max(0.0, 0.0237 - Q.centroid_offset) / 4.222466e-05   # +1.7%  lam1 < 0.00696 and centroid_offset < 0.0237
        - 0.01663373 * max(0.0, 0.0338 - Q.sum_z_dr) * max(0.0, 0.0314 - Q.centroid_offset) / 0.0001667647   # -1.7%  sum_z_dr < 0.0338 and centroid_offset < 0.0314
        - 0.01639363 * max(0.0, 0.203 - Q.sj3_dr_max) * max(0.0, 0.00185 - Q.sum_z_dr2_top2) / 8.863884e-05   # -1.6%  sj3_dr_max < 0.203 and sum_z_dr2_top2 < 0.00185
        - 0.01598661 * max(0.0, 0.132 - Q.sj3_dr_max) / 0.0330641   # -1.6%  sj3_dr_max < 0.132
        - 0.01503106 * max(0.0, 0.00858 - Q.mean_eta2) / 0.006056615   # -1.5%  mean_eta2 < 0.00858
        + 0.01478697 * max(0.0, Q.z_6 - 0.043) / 0.02015374   # +1.5%  z_6 > 0.043
        + 0.01255549 * max(0.0, 0.000203 - Q.lam2) / 0.0001182606   # +1.3%  lam2 < 0.000203
        - 0.01210765 * max(0.0, Q.eccentricity - 0.811) / 0.1136804   # -1.2%  eccentricity > 0.811
        - 0.01177153 * max(0.0, Q.sj3_dr_max - 0.165) / 0.05002187   # -1.2%  sj3_dr_max > 0.165
        - 0.01157943 * max(0.0, Q.sj2_dr - 0.161) / 0.03835059   # -1.2%  sj2_dr > 0.161
        - 0.01005436 * max(0.0, 0.0259 - Q.tau1) / 0.005244544   # -1.0%  tau1 < 0.0259
        + 0.01001416 * max(0.0, 0.0274 - Q.tau1) * max(0.0, 0.0314 - Q.centroid_offset) / 0.0001260326   # +1.0%  tau1 < 0.0274 and centroid_offset < 0.0314
        - 0.009291574 * max(0.0, Q.pt_6 - 32.0) / 10.14043   # -0.9%  pt_6 > 32
        + 0.008917193 * max(0.0, 0.116 - Q.max_dr) / 0.03056003   # +0.9%  max_dr < 0.116
        - 0.008840025 * max(0.0, 0.0172 - Q.tau2) / 0.008119587   # -0.9%  tau2 < 0.0172
        - 0.008834853 * max(0.0, Q.sum_pt_top3 - 436.0) / 83.21584   # -0.9%  sum_pt_top3 > 436
        - 0.008652281 * max(0.0, 0.00715 - Q.sum_z_dr2_top3) / 0.003949044   # -0.9%  sum_z_dr2_top3 < 0.00715
        + 0.008598731 * max(0.0, 446.0 - Q.sum_pt_top3) / 60.26404   # +0.9%  sum_pt_top3 < 446
        + 0.007992914 * max(0.0, 0.00814 - Q.lam1) * max(0.0, 0.535 - Q.planar_flow) / 0.0009648846   # +0.8%  lam1 < 0.00814 and planar_flow < 0.535
        + 0.007299101 * max(0.0, Q.mean_phi2 - 0.00063) / 0.002789104   # +0.7%  mean_phi2 > 0.00063
        + 0.006675106 * max(0.0, Q.sum_pt - 937.0) / 8.091046   # +0.7%  sum_pt > 937
        + 0.006270681 * max(0.0, Q.z_top3_slots - 0.619) / 0.04945609   # +0.6%  z_top3_slots > 0.619
        + 0.006259778 * max(0.0, 41.6 - Q.pt_7) / 8.816089   # +0.6%  pt_7 < 41.6
        + 0.006147524 * max(0.0, Q.sj2_dr - 0.205) / 0.02177459   # +0.6%  sj2_dr > 0.205
        + 0.006115122 * max(0.0, 0.0945 - Q.M2) / 0.0401019   # +0.6%  M2 < 0.0945
        + 0.005965216 * max(0.0, 0.0371 - Q.tau1) / 0.009485268   # +0.6%  tau1 < 0.0371
        - 0.005892737 * max(0.0, 0.142 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.00405) / 0.0002871208   # -0.6%  sj3_dr_max < 0.142 and centroid_offset > 0.00405
        + 0.005779266 * max(0.0, 3.93 - Q.n_dr_0p1_0p2) / 2.770281   # +0.6%  n_dr_0p1_0p2 < 3.93
        - 0.005565456 * max(0.0, Q.log_sum_pt - 6.84) / 0.008148651   # -0.6%  log_sum_pt > 6.84
        - 0.005469124 * max(0.0, 0.00574 - Q.lam1) * max(0.0, 39.6 - Q.pt_7) / 0.02103428   # -0.5%  lam1 < 0.00574 and pt_7 < 39.6
        + 0.00471786 * max(0.0, 1.01 - Q.n_dr_0p2_0p4) / 0.8159909   # +0.5%  n_dr_0p2_0p4 < 1.01
        + 0.004653145 * max(0.0, 0.0199 - Q.zdr_0) / 0.00809532   # +0.5%  zdr_0 < 0.0199
        + 0.004450594 * max(0.0, Q.sj3_dr_max - 0.255) / 0.02028194   # +0.4%  sj3_dr_max > 0.255
        - 0.004337898 * max(0.0, 0.616 - Q.z_top3_slots) / 0.03544109   # -0.4%  z_top3_slots < 0.616
        + 0.00433194 * max(0.0, 0.0937 - Q.tau1) / 0.04132921   # +0.4%  tau1 < 0.0937
        - 0.004210757 * max(0.0, Q.psi_0p2 - 0.9) / 0.08471869   # -0.4%  psi_0p2 > 0.9
        - 0.004160597 * max(0.0, 0.00468 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0104) / 7.031597e-06   # -0.4%  lam1 < 0.00468 and centroid_offset > 0.0104
        + 0.003639568 * max(0.0, Q.z_7 - 0.0367) / 0.01849537   # +0.4%  z_7 > 0.0367
        - 0.003459301 * max(0.0, Q.absphi_0 - 0.0193) / 0.01926771   # -0.3%  absphi_0 > 0.0193
        + 0.003416421 * max(0.0, 0.014 - Q.zdr_1) / 0.006237244   # +0.3%  zdr_1 < 0.014
        + 0.003327054 * max(0.0, 0.00632 - Q.zdr_6) / 0.002946116   # +0.3%  zdr_6 < 0.00632
        + 0.003241635 * max(0.0, 0.107 - Q.dr_7) / 0.04722852   # +0.3%  dr_7 < 0.107
        + 0.003196878 * max(0.0, 0.0117 - Q.zdr_2) / 0.005497103   # +0.3%  zdr_2 < 0.0117
        + 0.003185876 * max(0.0, Q.phi_0 - -0.0047) / 0.01911254   # +0.3%  phi_0 > -0.0047
        + 0.0030678 * max(0.0, 0.0568 - Q.sj3_dr_max) / 0.008401166   # +0.3%  sj3_dr_max < 0.0568
        - 0.002835753 * max(0.0, 0.16 - Q.dr0_7) / 0.0928789   # -0.3%  dr0_7 < 0.16
        + 0.002651691 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.0027 - Q.sum_z_dr2_top5) / 6.428349e-05   # +0.3%  log_sum_pt > 6.7 and sum_z_dr2_top5 < 0.0027
        - 0.00250765 * max(0.0, 0.00363 - Q.mean_phi) / 0.007552517   # -0.3%  mean_phi < 0.00363
        + 0.00245116 * max(0.0, Q.N2 - 0.215) / 0.05291605   # +0.2%  N2 > 0.215
        - 0.002440124 * max(0.0, 5.57 - Q.ptdr0_6) / 3.325741   # -0.2%  ptdr0_6 < 5.57
        - 0.002402956 * max(0.0, 0.0491 - Q.max_dr) / 0.005777993   # -0.2%  max_dr < 0.0491
        + 0.002302171 * max(0.0, 0.00715 - Q.zdr_5) / 0.003337672   # +0.2%  zdr_5 < 0.00715
        + 0.002242969 * max(0.0, 0.000483 - Q.lam1) / 8.199947e-05   # +0.2%  lam1 < 0.000483
        + 0.002215808 * max(0.0, -0.0208 - Q.phi_0) / 0.009217198   # +0.2%  phi_0 < -0.0208
        + 0.002154142 * max(0.0, 0.137 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.0129) / 0.0001376035   # +0.2%  sj3_dr_max < 0.137 and centroid_offset > 0.0129
        - 0.002016825 * max(0.0, Q.C2_b2 - 0.00173) / 0.002723707   # -0.2%  C2_b2 > 0.00173
        - 0.001960779 * max(0.0, 0.0455 - Q.z_6) * max(0.0, 0.0703 - Q.absphi_1) / 0.0002056439   # -0.2%  z_6 < 0.0455 and absphi_1 < 0.0703
        - 0.001799281 * max(0.0, 0.00575 - Q.lam1) * max(0.0, 6.59 - Q.log_sum_pt) / 0.0001762091   # -0.2%  lam1 < 0.00575 and log_sum_pt < 6.59
        + 0.001751969 * max(0.0, Q.mean_eta2 - 0.00919) / 0.0006373415   # +0.2%  mean_eta2 > 0.00919
        - 0.001748016 * max(0.0, 0.159 - Q.sj2_dr) / 0.04831679   # -0.2%  sj2_dr < 0.159
        - 0.001395604 * max(0.0, Q.sum_z_dr2_top5 - 0.00906) / 0.001937845   # -0.1%  sum_z_dr2_top5 > 0.00906
        - 0.0009995936 * max(0.0, Q.ptdr0_7 - 4.76) / 0.9068641   # -0.1%  ptdr0_7 > 4.76
        - 0.0008175102 * max(0.0, Q.ptdr0_6 - 5.55) / 0.9632872   # -0.1%  ptdr0_6 > 5.55
        - 0.0007621044 * max(0.0, 0.000664 - Q.mean_eta) / 0.005900481   # -0.1%  mean_eta < 0.000664
        + 0.0006346828 * max(0.0, Q.sj3_dr_max - 0.351) / 0.006134393   # +0.1%  sj3_dr_max > 0.351
        + 0.0004603099 * max(0.0, Q.centroid_offset - 0.0335) / 0.002178244   # +0.0%  centroid_offset > 0.0335
        + 8.589112e-05 * max(0.0, Q.tau1 - 0.154) / 0.002571151   # +0.0%  tau1 > 0.154
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 16.33;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.33475 * (-0.0752996
        + 0.1593034 * max(0.0, 0.0267 - Q.centroid_offset) * max(0.0, 0.00719 - Q.lam1) / 5.321432e-05   # +15.9%  centroid_offset < 0.0267 and lam1 < 0.00719
        + 0.1588733 * max(0.0, 0.00564 - Q.lam1) * max(0.0, 0.00108 - Q.lam2) / 2.337977e-06   # +15.9%  lam1 < 0.00564 and lam2 < 0.00108
        + 0.1340877 * max(0.0, 0.00805 - Q.lam1) / 0.004041124   # +13.4%  lam1 < 0.00805
        - 0.1019447 * max(0.0, Q.sj3_dr_max - 0.163) / 0.05108099   # -10.2%  sj3_dr_max > 0.163
        - 0.09001685 * max(0.0, 0.00724 - Q.lam1) / 0.003419541   # -9.0%  lam1 < 0.00724
        + 0.08699671 * max(0.0, 963.0 - Q.sum_pt) / 252.8593   # +8.7%  sum_pt < 963
        - 0.07196317 * max(0.0, 0.0836 - Q.tau1) / 0.03427114   # -7.2%  tau1 < 0.0836
        - 0.07045441 * max(0.0, 0.224 - Q.sj3_dr_max) / 0.08339529   # -7.0%  sj3_dr_max < 0.224
        + 0.05536444 * max(0.0, Q.sj3_dr_max - 0.242) / 0.02312952   # +5.5%  sj3_dr_max > 0.242
        - 0.03862992 * max(0.0, 0.042 - Q.sum_z_dr) / 0.00964847   # -3.9%  sum_z_dr < 0.042
        + 0.007680429 * max(0.0, 0.000135 - Q.lam1) / 1.130251e-05   # +0.8%  lam1 < 0.000135
        - 0.007038743 * max(0.0, 910.0 - Q.sum_pt) * max(0.0, 0.0135 - Q.centroid_offset) / 0.479067   # -0.7%  sum_pt < 910 and centroid_offset < 0.0135
        + 0.006791991 * max(0.0, Q.sum_z_dr - 0.1) / 0.005689511   # +0.7%  sum_z_dr > 0.1
        + 0.006730925 * max(0.0, Q.lam2 - 0.00118) / 0.0003071172   # +0.7%  lam2 > 0.00118
        + 0.004123354 * max(0.0, 994.0 - Q.sum_pt) * max(0.0, 0.0548 - Q.z_5) / 0.4290061   # +0.4%  sum_pt < 994 and z_5 < 0.0548
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 17.69;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.69092 * (-0.3465055
        + 0.2102008 * max(0.0, 4.17 - Q.n_dr_0p05_0p1) / 2.512598   # +21.0%  n_dr_0p05_0p1 < 4.17
        + 0.2069589 * max(0.0, 3.91 - Q.n_dr_0p1_0p2) / 2.752851   # +20.7%  n_dr_0p1_0p2 < 3.91
        + 0.1452125 * max(0.0, 4.83 - Q.n_dr_0_0p05) / 1.861553   # +14.5%  n_dr_0_0p05 < 4.83
        - 0.09086942 * max(0.0, Q.n_dr_0_0p05 - 4.76) / 1.217851   # -9.1%  n_dr_0_0p05 > 4.76
        + 0.06791027 * max(0.0, 0.986 - Q.n_dr_0p2_0p4) / 0.7956258   # +6.8%  n_dr_0p2_0p4 < 0.986
        + 0.05453229 * max(0.0, Q.lam1 - 0.00291) / 0.003937658   # +5.5%  lam1 > 0.00291
        - 0.03781382 * max(0.0, 0.00279 - Q.lam1) / 0.0008813718   # -3.8%  lam1 < 0.00279
        - 0.0367726 * max(0.0, Q.n_dr_0p05_0p1 - 4.0) / 0.512237   # -3.7%  n_dr_0p05_0p1 > 4
        + 0.02508291 * max(0.0, Q.lam2 - 0.000183) / 0.0004491293   # +2.5%  lam2 > 0.000183
        - 0.02305215 * max(0.0, 0.0652 - Q.z_7) / 0.01664546   # -2.3%  z_7 < 0.0652
        - 0.01887945 * max(0.0, Q.n_dr_0p1_0p2 - 3.79) / 0.2303412   # -1.9%  n_dr_0p1_0p2 > 3.79
        + 0.01833523 * max(0.0, 0.307 - Q.D2_b2) / 0.09111436   # +1.8%  D2_b2 < 0.307
        - 0.01660373 * max(0.0, Q.LHA - 0.311) * max(0.0, 0.791 - Q.planar_flow) / 0.007343382   # -1.7%  LHA > 0.311 and planar_flow < 0.791
        - 0.01199133 * max(0.0, Q.sj2_dr - 0.161) * max(0.0, 0.453 - Q.D2_b2) / 0.007240191   # -1.2%  sj2_dr > 0.161 and D2_b2 < 0.453
        - 0.009642085 * max(0.0, Q.n_dr_0p2_0p4 - 1.12) * max(0.0, 4.59 - Q.D2_b2) / 0.5248533   # -1.0%  n_dr_0p2_0p4 > 1.12 and D2_b2 < 4.59
        - 0.007587152 * max(0.0, Q.pt_7 - 22.6) * max(0.0, 0.262 - Q.D2_b2) / 1.065267   # -0.8%  pt_7 > 22.6 and D2_b2 < 0.262
        - 0.007560277 * max(0.0, Q.lam2 - -0.000475) * max(0.0, 6.61 - Q.log_sum_pt) / 0.0002358876   # -0.8%  lam2 > -0.000475 and log_sum_pt < 6.61
        - 0.006725145 * max(0.0, 44.0 - Q.pt_6) * max(0.0, 6.56 - Q.log_sum_pt) / 0.7299018   # -0.7%  pt_6 < 44 and log_sum_pt < 6.56
        - 0.004269942 * max(0.0, Q.lam2 - 0.00352) / 0.0001516851   # -0.4%  lam2 > 0.00352
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 18.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.64395 * (-0.0227956
        + 0.3089273 * max(0.0, 0.0116 - Q.lam1) / 0.006972913   # +30.9%  lam1 < 0.0116
        - 0.1227592 * max(0.0, 0.138 - Q.tau1) / 0.07865005   # -12.3%  tau1 < 0.138
        + 0.09966724 * max(0.0, Q.sj2_dr - 0.122) / 0.0605274   # +10.0%  sj2_dr > 0.122
        + 0.09846895 * max(0.0, 0.0161 - Q.lam1) * max(0.0, 0.00167 - Q.lam2) / 1.684266e-05   # +9.8%  lam1 < 0.0161 and lam2 < 0.00167
        - 0.09327173 * max(0.0, 0.0745 - Q.sum_z_dr) / 0.02687718   # -9.3%  sum_z_dr < 0.0745
        - 0.08179642 * max(0.0, Q.sj2_dr - 0.156) / 0.04088494   # -8.2%  sj2_dr > 0.156
        - 0.05031475 * max(0.0, 0.00434 - Q.lam1) * max(0.0, 0.0266 - Q.centroid_offset) / 2.680188e-05   # -5.0%  lam1 < 0.00434 and centroid_offset < 0.0266
        - 0.03699437 * max(0.0, 6.78 - Q.log_sum_pt) / 0.2535739   # -3.7%  log_sum_pt < 6.78
        + 0.02849139 * max(0.0, 0.235 - Q.planar_flow) / 0.0985514   # +2.8%  planar_flow < 0.235
        - 0.02746272 * max(0.0, 0.0714 - Q.z_7) / 0.0213339   # -2.7%  z_7 < 0.0714
        - 0.01436438 * max(0.0, 0.0142 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0161) / 3.11768e-05   # -1.4%  lam1 < 0.0142 and centroid_offset > 0.0161
        + 0.01321459 * max(0.0, 0.0163 - Q.centroid_offset) / 0.005414774   # +1.3%  centroid_offset < 0.0163
        + 0.007360701 * max(0.0, Q.sum_pt_top5 - 474.0) * max(0.0, 0.318 - Q.D2_b2) / 12.47569   # +0.7%  sum_pt_top5 > 474 and D2_b2 < 0.318
        - 0.007260997 * max(0.0, Q.lam1 - 0.0116) / 0.001289273   # -0.7%  lam1 > 0.0116
        - 0.005995066 * max(0.0, 0.00769 - Q.lam1) * max(0.0, 0.378 - Q.D2_b2) / 0.000198882   # -0.6%  lam1 < 0.00769 and D2_b2 < 0.378
        - 0.003650169 * max(0.0, 0.00308 - Q.lam1) * max(0.0, 0.24 - Q.planar_flow) / 3.038106e-05   # -0.4%  lam1 < 0.00308 and planar_flow < 0.24
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 17.49;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.49416 * (0.2600869
        - 0.2742836 * max(0.0, 0.122 - Q.sum_z_dr) / 0.06618427   # -27.4%  sum_z_dr < 0.122
        - 0.1835809 * max(0.0, Q.LHA - 0.105) / 0.1408594   # -18.4%  LHA > 0.105
        - 0.07773643 * max(0.0, 0.00455 - Q.lam1) / 0.001699916   # -7.8%  lam1 < 0.00455
        - 0.04869224 * max(0.0, 0.0236 - Q.lam1) / 0.01789558   # -4.9%  lam1 < 0.0236
        + 0.04678074 * max(0.0, 0.151 - Q.sj3_dr_min) / 0.1125708   # +4.7%  sj3_dr_min < 0.151
        + 0.04014927 * max(0.0, 0.0675 - Q.C2) / 0.04180819   # +4.0%  C2 < 0.0675
        + 0.03448732 * max(0.0, 0.000194 - Q.e3) / 0.0001558983   # +3.4%  e3 < 0.000194
        - 0.02948501 * max(0.0, 0.0371 - Q.centroid_offset) / 0.02167291   # -2.9%  centroid_offset < 0.0371
        - 0.0283737 * max(0.0, Q.z_dr_0_0p05 - 0.782) / 0.08528762   # -2.8%  z_dr_0_0p05 > 0.782
        + 0.02782088 * max(0.0, Q.max_dr - 0.0391) / 0.08801137   # +2.8%  max_dr > 0.0391
        - 0.02308012 * max(0.0, 0.209 - Q.sj2_dr) / 0.0796385   # -2.3%  sj2_dr < 0.209
        - 0.02153812 * max(0.0, 0.0917 - Q.tau21_b2) / 0.03488807   # -2.2%  tau21_b2 < 0.0917
        + 0.02101614 * max(0.0, Q.log_sum_pt - 6.68) / 0.04192242   # +2.1%  log_sum_pt > 6.68
        + 0.01820684 * max(0.0, 3.59e-09 - Q.e4) / 2.632341e-09   # +1.8%  e4 < 3.59e-09
        - 0.01755542 * max(0.0, 40.6 - Q.pt_7) / 8.082034   # -1.8%  pt_7 < 40.6
        + 0.01412793 * max(0.0, Q.sum_z_dr - 0.123) / 0.002786428   # +1.4%  sum_z_dr > 0.123
        + 0.01150617 * max(0.0, 0.124 - Q.sj2_dr) / 0.03327118   # +1.2%  sj2_dr < 0.124
        + 0.009313497 * max(0.0, 0.077 - Q.sd_rg) / 0.02724611   # +0.9%  sd_rg < 0.077
        + 0.008083245 * max(0.0, Q.n_dr_0_0p05 - 5.69) / 0.7769755   # +0.8%  n_dr_0_0p05 > 5.69
        - 0.007763649 * max(0.0, 0.0151 - Q.tau21_b2) / 0.002286507   # -0.8%  tau21_b2 < 0.0151
        + 0.006128604 * max(0.0, Q.n_dr_0p2_0p4 - 0.994) / 0.1540442   # +0.6%  n_dr_0p2_0p4 > 0.994
        - 0.005887872 * max(0.0, Q.sd_rg - 0.321) * max(0.0, 6.94 - Q.log_sum_pt) / 0.001289153   # -0.6%  sd_rg > 0.321 and log_sum_pt < 6.94
        - 0.005675031 * max(0.0, 0.161 - Q.sd_rg) / 0.07041126   # -0.6%  sd_rg < 0.161
        + 0.005276311 * max(0.0, Q.centroid_offset - 0.0348) / 0.002015384   # +0.5%  centroid_offset > 0.0348
        - 0.004021556 * max(0.0, Q.sum_z_dr2_top3 - 0.00556) / 0.002605693   # -0.4%  sum_z_dr2_top3 > 0.00556
        - 0.003799178 * max(0.0, Q.sum_z_dr - 0.11) * max(0.0, 65.7 - Q.pt_6) / 0.1149886   # -0.4%  sum_z_dr > 0.11 and pt_6 < 65.7
        + 0.003603476 * max(0.0, Q.sd_rg - 0.316) / 0.002211922   # +0.4%  sd_rg > 0.316
        + 0.003170101 * max(0.0, Q.sd_rg - 0.323) * max(0.0, 6.77 - Q.log_sum_pt) / 0.0009136448   # +0.3%  sd_rg > 0.323 and log_sum_pt < 6.77
        - 0.00315003 * max(0.0, Q.centroid_offset - 0.0246) * max(0.0, 733.0 - Q.sum_pt) / 0.7020014   # -0.3%  centroid_offset > 0.0246 and sum_pt < 733
        + 0.002707973 * max(0.0, Q.lam2 - 0.000244) / 0.0004346211   # +0.3%  lam2 > 0.000244
        + 0.00253597 * max(0.0, Q.sj3_dr_max - 0.256) / 0.0200745   # +0.3%  sj3_dr_max > 0.256
        - 0.001629733 * max(0.0, 0.917 - Q.psi_0p2) / 0.01575183   # -0.2%  psi_0p2 < 0.917
        - 0.001498131 * max(0.0, Q.sd_rg - 0.181) / 0.02096682   # -0.1%  sd_rg > 0.181
        + 0.001123961 * max(0.0, Q.sum_z_dr - 0.155) / 0.0006087541   # +0.1%  sum_z_dr > 0.155
        - 0.001081574 * max(0.0, Q.n_dr_0p2_0p4 - 1.6) * max(0.0, 0.00409 - Q.lam2) / 0.000210939   # -0.1%  n_dr_0p2_0p4 > 1.6 and lam2 < 0.00409
        - 0.0008721259 * max(0.0, Q.n_dr_0p2_0p4 - 1.53) * max(0.0, 635.0 - Q.sum_pt) / 8.87041   # -0.1%  n_dr_0p2_0p4 > 1.53 and sum_pt < 635
        - 0.0008655919 * max(0.0, -0.0185 - Q.mean_phi) / 0.001211424   # -0.1%  mean_phi < -0.0185
        + 0.0006296844 * max(0.0, Q.sum_z_dr - 0.107) * max(0.0, Q.log_sum_pt - 6.52) / 5.591775e-05   # +0.1%  sum_z_dr > 0.107 and log_sum_pt > 6.52
        + 0.000612026 * max(0.0, Q.dr_2 - 0.135) / 0.005248469   # +0.1%  dr_2 > 0.135
        - 0.0004648545 * max(0.0, Q.dr0_7 - 0.158) / 0.02064019   # -0.0%  dr0_7 > 0.158
        - 0.0004351072 * max(0.0, Q.zdr_0 - 0.046) / 0.0003353231   # -0.0%  zdr_0 > 0.046
        + 0.0004324744 * max(0.0, Q.dr_1 - 0.13) / 0.004323299   # +0.0%  dr_1 > 0.13
        + 0.0002949597 * max(0.0, Q.ptdr0_7 - 9.41) / 0.1823346   # +0.0%  ptdr0_7 > 9.41
        + 0.0002755393 * max(0.0, Q.ptdr0_2 - 17.3) / 0.6219777   # +0.0%  ptdr0_2 > 17.3
        + 0.0001065302 * max(0.0, Q.ptdr0_5 - 13.1) / 0.16066   # +0.0%  ptdr0_5 > 13.1
        + 7.425795e-05 * max(0.0, Q.pt1_dr01 - 19.6) / 1.149628   # +0.0%  pt1_dr01 > 19.6
        - 6.610084e-05 * max(0.0, Q.sum_z_dr - 0.144) * max(0.0, Q.log_sum_pt - 6.56) / 6.842476e-06   # -0.0%  sum_z_dr > 0.144 and log_sum_pt > 6.56
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 12.73;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.73494 * (0.4389498
        - 0.1617646 * max(0.0, 0.162 - Q.sum_z_dr) * max(0.0, 6.86 - Q.log_sum_pt) / 0.02833649   # -16.2%  sum_z_dr < 0.162 and log_sum_pt < 6.86
        + 0.1574583 * max(0.0, 0.377 - Q.LHA) / 0.1392516   # +15.7%  LHA < 0.377
        - 0.115352 * max(0.0, 0.126 - Q.tau1) / 0.0680093   # -11.5%  tau1 < 0.126
        + 0.09065931 * max(0.0, 0.00547 - Q.mean_phi2) / 0.003366009   # +9.1%  mean_phi2 < 0.00547
        + 0.07704008 * max(0.0, 0.00513 - Q.mean_eta2) / 0.003094956   # +7.7%  mean_eta2 < 0.00513
        - 0.06626339 * max(0.0, 0.000711 - Q.lam2) / 0.0005409362   # -6.6%  lam2 < 0.000711
        - 0.05656409 * max(0.0, 0.153 - Q.sum_z_dr) * max(0.0, 34.9 - Q.pt_7) / 0.4967865   # -5.7%  sum_z_dr < 0.153 and pt_7 < 34.9
        + 0.0565003 * max(0.0, Q.z_7 - 0.0441) / 0.01337413   # +5.7%  z_7 > 0.0441
        - 0.05086231 * max(0.0, 8.02e-05 - Q.e3) * max(0.0, 0.0338 - Q.centroid_offset) / 1.167078e-06   # -5.1%  e3 < 8.02e-05 and centroid_offset < 0.0338
        - 0.04152262 * max(0.0, Q.LHA - 0.229) / 0.05133865   # -4.2%  LHA > 0.229
        - 0.03541406 * max(0.0, Q.pt_7 - 28.1) / 8.461463   # -3.5%  pt_7 > 28.1
        - 0.03248287 * max(0.0, Q.mean_eta2 - 0.00496) / 0.001220258   # -3.2%  mean_eta2 > 0.00496
        - 0.02880464 * max(0.0, Q.mean_phi2 - 0.00545) / 0.001125231   # -2.9%  mean_phi2 > 0.00545
        + 0.02002022 * max(0.0, Q.sum_pt_top5 - 658.0) * max(0.0, 33.5 - Q.pt_7) / 523.5244   # +2.0%  sum_pt_top5 > 658 and pt_7 < 33.5
        - 0.009291171 * max(0.0, 25.3 - Q.pt_6) / 0.6960148   # -0.9%  pt_6 < 25.3
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 27.13;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.12808 * (0.04128563
        - 0.1554276 * max(0.0, 0.0809 - Q.sum_z_dr) / 0.03146606   # -15.5%  sum_z_dr < 0.0809
        - 0.09094225 * max(0.0, 0.00693 - Q.lam1) / 0.003191577   # -9.1%  lam1 < 0.00693
        + 0.08793604 * max(0.0, 0.0117 - Q.lam1) / 0.0070578   # +8.8%  lam1 < 0.0117
        + 0.06697738 * max(0.0, 0.0981 - Q.tau1) / 0.04464295   # +6.7%  tau1 < 0.0981
        - 0.05319923 * max(0.0, Q.sum_z_dr - 0.0875) / 0.007801044   # -5.3%  sum_z_dr > 0.0875
        - 0.04275293 * max(0.0, Q.sum_z_dr - 0.0277) / 0.03557684   # -4.3%  sum_z_dr > 0.0277
        - 0.0374563 * max(0.0, 0.121 - Q.sj3_dr_min) / 0.08538803   # -3.7%  sj3_dr_min < 0.121
        + 0.03743096 * max(0.0, 0.00349 - Q.lam2) * max(0.0, Q.LHA - 0.171) / 0.0002350533   # +3.7%  lam2 < 0.00349 and LHA > 0.171
        + 0.02946507 * max(0.0, 0.253 - Q.LHA) / 0.04873969   # +2.9%  LHA < 0.253
        + 0.02754298 * max(0.0, Q.sum_z_dr - 0.0809) / 0.009270326   # +2.8%  sum_z_dr > 0.0809
        + 0.02743866 * max(0.0, Q.LHA - 0.318) / 0.01407104   # +2.7%  LHA > 0.318
        + 0.02541244 * max(0.0, 0.00353 - Q.lam2) * max(0.0, Q.sd_rg - 0.159) / 6.324686e-05   # +2.5%  lam2 < 0.00353 and sd_rg > 0.159
        + 0.02459971 * max(0.0, 0.25 - Q.sd_rg) / 0.1401981   # +2.5%  sd_rg < 0.25
        - 0.02401834 * max(0.0, 0.198 - Q.sj3_dr23) / 0.09754063   # -2.4%  sj3_dr23 < 0.198
        - 0.02077415 * max(0.0, 0.113 - Q.planar_flow) * max(0.0, 0.00746 - Q.lam1) / 6.360755e-05   # -2.1%  planar_flow < 0.113 and lam1 < 0.00746
        + 0.02022842 * max(0.0, Q.z_dr_0p05_0p1 - 0.132) * max(0.0, 1.29 - Q.n_dr_0p2_0p4) / 0.2305706   # +2.0%  z_dr_0p05_0p1 > 0.132 and n_dr_0p2_0p4 < 1.29
        - 0.01954475 * max(0.0, Q.z_dr_0p05_0p1 - 0.166) * max(0.0, 0.228 - Q.z_dr_0p2_0p4) / 0.04016755   # -2.0%  z_dr_0p05_0p1 > 0.166 and z_dr_0p2_0p4 < 0.228
        + 0.01788179 * max(0.0, 0.112 - Q.sj3_dr23) / 0.04255252   # +1.8%  sj3_dr23 < 0.112
        - 0.01630011 * Q.eccentricity / 0.8791067   # -1.6%  eccentricity
        - 0.01627311 * max(0.0, 0.00338 - Q.lam2) * max(0.0, Q.sd_rg - 0.191) / 3.503637e-05   # -1.6%  lam2 < 0.00338 and sd_rg > 0.191
        - 0.01545352 * max(0.0, Q.sj3_dr_max - 0.23) / 0.02603877   # -1.5%  sj3_dr_max > 0.23
        + 0.01542543 * max(0.0, 0.13 - Q.planar_flow) / 0.04205653   # +1.5%  planar_flow < 0.13
        - 0.01397354 * max(0.0, Q.sum_z_dr - 0.0557) / 0.01885947   # -1.4%  sum_z_dr > 0.0557
        + 0.01272966 * max(0.0, Q.sj3_dr13 - 0.0963) / 0.06037262   # +1.3%  sj3_dr13 > 0.0963
        + 0.009930665 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 0.00664 - Q.lam1) / 4.026904e-05   # +1.0%  planar_flow < 0.106 and lam1 < 0.00664
        + 0.008401392 * max(0.0, 0.00757 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0197) / 6.885609e-06   # +0.8%  lam1 < 0.00757 and centroid_offset > 0.0197
        - 0.008129347 * max(0.0, 0.00372 - Q.lam1) / 0.00128967   # -0.8%  lam1 < 0.00372
        + 0.007910595 * max(0.0, 0.0963 - Q.sj3_dr13) / 0.03281335   # +0.8%  sj3_dr13 < 0.0963
        - 0.007797016 * max(0.0, 0.0153 - Q.lam1) * max(0.0, Q.centroid_offset - 0.024) / 1.705791e-05   # -0.8%  lam1 < 0.0153 and centroid_offset > 0.024
        + 0.007649884 * max(0.0, 0.698 - Q.z_dr_0p05_0p1) * max(0.0, 0.0428 - Q.z_dr_0p1_0p2) / 0.0120655   # +0.8%  z_dr_0p05_0p1 < 0.698 and z_dr_0p1_0p2 < 0.0428
        + 0.007234042 * max(0.0, Q.e3 - 3.14e-05) / 6.001397e-05   # +0.7%  e3 > 3.14e-05
        + 0.006619434 * max(0.0, Q.max_dr - 0.159) / 0.02181926   # +0.7%  max_dr > 0.159
        - 0.0047066 * max(0.0, Q.sj3_dr13 - 0.191) / 0.02243955   # -0.5%  sj3_dr13 > 0.191
        + 0.004528725 * max(0.0, 0.136 - Q.planar_flow) * max(0.0, 0.304 - Q.sj2_zsoft) / 0.004028053   # +0.5%  planar_flow < 0.136 and sj2_zsoft < 0.304
        - 0.004330007 * max(0.0, 0.00782 - Q.lam1) * max(0.0, 0.987 - Q.D2) / 0.0002431983   # -0.4%  lam1 < 0.00782 and D2 < 0.987
        - 0.004237839 * max(0.0, 0.152 - Q.planar_flow) * max(0.0, 810.0 - Q.sum_pt) / 5.500692   # -0.4%  planar_flow < 0.152 and sum_pt < 810
        - 0.003688785 * max(0.0, Q.z_dr_0p05_0p1 - -0.0637) * max(0.0, 6.7 - Q.log_sum_pt) / 0.07581035   # -0.4%  z_dr_0p05_0p1 > -0.0637 and log_sum_pt < 6.7
        - 0.003420053 * max(0.0, 0.0036 - Q.lam2) * max(0.0, Q.centroid_offset - 0.0196) / 1.230497e-05   # -0.3%  lam2 < 0.0036 and centroid_offset > 0.0196
        - 0.002873798 * max(0.0, Q.pt_7 - 32.9) / 5.339769   # -0.3%  pt_7 > 32.9
        - 0.002872755 * max(0.0, Q.mean_phi - -0.0107) / 0.0129241   # -0.3%  mean_phi > -0.0107
        + 0.002586869 * max(0.0, Q.z_7 - 0.0494) / 0.01022985   # +0.3%  z_7 > 0.0494
        + 0.002519022 * max(0.0, 0.00753 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.186) / 2.190264e-05   # +0.3%  lam1 < 0.00753 and sj3_dr23 > 0.186
        - 0.001190062 * max(0.0, Q.log_sum_pt - 6.86) / 0.006330214   # -0.1%  log_sum_pt > 6.86
        - 0.000188745 * max(0.0, 0.0529 - Q.tau1) / 0.01662432   # -0.0%  tau1 < 0.0529
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 21.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.17589 * (-0.003730657
        + 0.1682169 * max(0.0, 0.00353 - Q.lam2) / 0.003152339   # +16.8%  lam2 < 0.00353
        - 0.147751 * max(0.0, 0.0892 - Q.sum_z_dr) / 0.03797039   # -14.8%  sum_z_dr < 0.0892
        - 0.07468932 * max(0.0, 0.0626 - Q.sum_z_dr) / 0.01952609   # -7.5%  sum_z_dr < 0.0626
        - 0.06511723 * max(0.0, Q.eccentricity - 0.619) / 0.2763358   # -6.5%  eccentricity > 0.619
        + 0.06328735 * max(0.0, 0.267 - Q.LHA) / 0.0560739   # +6.3%  LHA < 0.267
        - 0.06154086 * max(0.0, Q.sum_z_dr - 0.0818) / 0.00904988   # -6.2%  sum_z_dr > 0.0818
        + 0.04063607 * max(0.0, Q.LHA - 0.314) / 0.01501754   # +4.1%  LHA > 0.314
        + 0.03726693 * max(0.0, 0.0566 - Q.tau1) / 0.01848151   # +3.7%  tau1 < 0.0566
        - 0.03369038 * max(0.0, 0.000228 - Q.lam1) / 2.671999e-05   # -3.4%  lam1 < 0.000228
        - 0.02855477 * max(0.0, Q.e2 - 0.0254) / 0.01111531   # -2.9%  e2 > 0.0254
        + 0.02303723 * max(0.0, 0.103 - Q.dr_0) / 0.05646225   # +2.3%  dr_0 < 0.103
        + 0.0203799 * max(0.0, Q.tau1 - 0.0979) / 0.009966805   # +2.0%  tau1 > 0.0979
        + 0.02005709 * max(0.0, 0.189 - Q.sj3_dr_max) / 0.05973653   # +2.0%  sj3_dr_max < 0.189
        - 0.01930618 * max(0.0, 0.00738 - Q.lam1) / 0.003524358   # -1.9%  lam1 < 0.00738
        + 0.01758812 * max(0.0, Q.sj3_dr_max - 0.175) / 0.04503558   # +1.8%  sj3_dr_max > 0.175
        - 0.01479118 * max(0.0, 0.00821 - Q.lam1) * max(0.0, 1.06 - Q.D2) / 0.0003503539   # -1.5%  lam1 < 0.00821 and D2 < 1.06
        - 0.01443391 * max(0.0, Q.sj3_dr_max - 0.238) / 0.02406701   # -1.4%  sj3_dr_max > 0.238
        + 0.01322026 * max(0.0, 0.0145 - Q.lam1) * max(0.0, 1.07 - Q.D2) / 0.001413893   # +1.3%  lam1 < 0.0145 and D2 < 1.07
        - 0.01292312 * max(0.0, 0.25 - Q.N2) * max(0.0, 0.186 - Q.sj2_dr) / 0.001040527   # -1.3%  N2 < 0.25 and sj2_dr < 0.186
        + 0.01248247 * max(0.0, 0.0886 - Q.dr_1) / 0.04242817   # +1.2%  dr_1 < 0.0886
        + 0.0122623 * max(0.0, Q.max_dr - 0.0417) / 0.08598186   # +1.2%  max_dr > 0.0417
        - 0.01163098 * max(0.0, 0.000634 - Q.lam1) / 0.0001207335   # -1.2%  lam1 < 0.000634
        - 0.01064592 * max(0.0, 0.00328 - Q.lam2) * max(0.0, 0.017 - Q.zdr_1) / 2.706325e-05   # -1.1%  lam2 < 0.00328 and zdr_1 < 0.017
        + 0.009297081 * max(0.0, 0.248 - Q.N2) * max(0.0, 0.162 - Q.sj2_dr) / 0.0004252138   # +0.9%  N2 < 0.248 and sj2_dr < 0.162
        + 0.008807817 * max(0.0, Q.eccentricity - 0.894) * max(0.0, Q.sum_pt_top5 - 393.0) / 11.10199   # +0.9%  eccentricity > 0.894 and sum_pt_top5 > 393
        - 0.008607576 * max(0.0, 0.0207 - Q.zdr_0) / 0.008679671   # -0.9%  zdr_0 < 0.0207
        - 0.00853627 * max(0.0, 0.219 - Q.N2) * max(0.0, Q.lam1 - 0.00775) / 0.0001213175   # -0.9%  N2 < 0.219 and lam1 > 0.00775
        - 0.005385622 * max(0.0, 0.0212 - Q.centroid_offset) / 0.008639799   # -0.5%  centroid_offset < 0.0212
        + 0.005113843 * max(0.0, 0.0582 - Q.max_dr) / 0.008266427   # +0.5%  max_dr < 0.0582
        - 0.00432064 * max(0.0, Q.centroid_offset - 0.0262) / 0.003388645   # -0.4%  centroid_offset > 0.0262
        + 0.004106889 * max(0.0, 0.21 - Q.N2) * max(0.0, Q.sum_pt_top5 - 428.0) / 7.834868   # +0.4%  N2 < 0.21 and sum_pt_top5 > 428
        + 0.003976934 * max(0.0, Q.log_sum_pt - 6.81) / 0.01174548   # +0.4%  log_sum_pt > 6.81
        + 0.003629885 * max(0.0, Q.lam2 - 0.000866) / 0.0003401153   # +0.4%  lam2 > 0.000866
        + 0.003491318 * max(0.0, Q.tau1 - 0.103) * max(0.0, 0.65 - Q.D2_b2) / 0.002423993   # +0.3%  tau1 > 0.103 and D2_b2 < 0.65
        - 0.003474732 * max(0.0, Q.sum_pt_top5 - 761.0) / 19.56929   # -0.3%  sum_pt_top5 > 761
        + 0.002005419 * max(0.0, 0.00818 - Q.mean_phi) / 0.01088886   # +0.2%  mean_phi < 0.00818
        + 0.001444247 * max(0.0, Q.zdr_0 - 0.0192) / 0.003401917   # +0.1%  zdr_0 > 0.0192
        + 0.001346655 * max(0.0, 0.00977 - Q.lam1) * max(0.0, Q.sj3_dr23 - 0.193) / 3.490407e-05   # +0.1%  lam1 < 0.00977 and sj3_dr23 > 0.193
        - 0.0009553787 * max(0.0, Q.z_dr_0p05_0p1 - 0.687) * max(0.0, Q.sj3_dr23 - 0.193) / 0.000520077   # -0.1%  z_dr_0p05_0p1 > 0.687 and sj3_dr23 > 0.193
        - 0.0007717629 * max(0.0, 0.236 - Q.N2) * max(0.0, 0.0306 - Q.z_7) / 5.596838e-05   # -0.1%  N2 < 0.236 and z_7 < 0.0306
        - 0.0007703514 * max(0.0, 0.0142 - Q.lam1) * max(0.0, Q.centroid_offset - 0.053) / 1.073216e-06   # -0.1%  lam1 < 0.0142 and centroid_offset > 0.053
        + 0.0004480285 * max(0.0, 480.0 - Q.sum_pt) / 5.184373   # +0.0%  sum_pt < 480
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0314050420168068, 0.7409443277310924, 2.1012595588235294, 0.9544485294117647, 1.1781025210084033, 1.767086974789916, 0.956585294117647, 1.8446615546218488, 0.2546626050420168, 3.008581092436975, 2.0369502100840338, 2.1513235294117647, 0.1007546218487395, 3.296339705882353, 0.5165247899159664, 0.37501008403361347]
T = [2.3433442858784135, 1.3707233373818277, 3.3987939223345593, 2.4991287798713233, 2.8659241793592436]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +39%, n9 +22%, n5 -14%, n1 +12%, n0 -7%, n6 +4% ...
            + 0.3852976 * h[2] / H_AVG[2]
            + 0.2206675 * h[9] / H_AVG[9]
            - 0.1413914 * h[5] / H_AVG[5]
            + 0.1235121 * h[1] / H_AVG[1]
            - 0.06877224 * h[0] / H_AVG[0]
            + 0.04464838 * h[6] / H_AVG[6]
            - 0.01571075 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -19%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5572952 * h[9] / H_AVG[9]
            - 0.185755 * h[10] / H_AVG[10]
            + 0.08723362 * h[6] / H_AVG[6]
            - 0.08057579 * h[4] / H_AVG[4]
            + 0.06042956 * h[5] / H_AVG[5]
            + 0.0170991 * h[15] / H_AVG[15]
            + 0.01161169 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +12%, n14 -11%, n0 +10%, n6 -9% ...
            + 0.2373625 * h[11] / H_AVG[11]
            - 0.1404099 * h[3] / H_AVG[3]
            + 0.1187244 * h[7] / H_AVG[7]
            - 0.1139797 * h[14] / H_AVG[14]
            + 0.1043151 * h[0] / H_AVG[0]
            - 0.08795264 * h[6] / H_AVG[6]
            - 0.07585615 * h[15] / H_AVG[15]
            + 0.06819298 * h[13] / H_AVG[13]
            - 0.02766221 * h[9] / H_AVG[9]
            - 0.01873184 * h[8] / H_AVG[8]
            - 0.006812567 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -21%, n6 -14%, n14 +8%, n13 +7%, n9 -4% ...
            + 0.3459946 * h[7] / H_AVG[7]
            - 0.2148258 * h[3] / H_AVG[3]
            - 0.1435378 * h[6] / H_AVG[6]
            + 0.07750573 * h[14] / H_AVG[14]
            + 0.07213257 * h[13] / H_AVG[13]
            - 0.03762037 * h[9] / H_AVG[9]
            + 0.03706013 * h[1] / H_AVG[1]
            + 0.03682854 * h[4] / H_AVG[4]
            - 0.0234463 * h[15] / H_AVG[15]
            + 0.01104814 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +27%, n5 -15%, n4 +5%, n3 +2%, n12 -2% ...
            - 0.4672622 * h[13] / H_AVG[13]
            + 0.2665305 * h[10] / H_AVG[10]
            - 0.1541463 * h[5] / H_AVG[5]
            + 0.05138406 * h[4] / H_AVG[4]
            + 0.02081459 * h[3] / H_AVG[3]
            - 0.01757803 * h[12] / H_AVG[12]
            + 0.01666103 * h[8] / H_AVG[8]
            + 0.005623214 * h[0] / H_AVG[0]
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
