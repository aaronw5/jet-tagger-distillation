"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  9:  11.8%   (on for 71% of jets)
  neuron  7:  10.1%   (on for 64% of jets)
  neuron  3:   8.6%   (on for 29% of jets)
  neuron 10:   8.1%   (on for 83% of jets)
  neuron  2:   7.2%   (on for 89% of jets)
  neuron  5:   7.1%   (on for 65% of jets)
  neuron  6:   7.1%   (on for 51% of jets)
  neuron 11:   6.4%   (on for 72% of jets)
  neuron 14:   4.7%   (on for 33% of jets)
  neuron  0:   4.2%   (on for 44% of jets)
  neuron  1:   3.3%   (on for 62% of jets)
  neuron  4:   3.1%   (on for 47% of jets)
  neuron 15:   2.7%   (on for 33% of jets)
  neuron  8:   1.0%   (on for 36% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.6% (the network: 65.8%); same class as the network for 87.6% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_6               |Δη| of particle 6
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
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
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
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
        z_top5=sum(zs[:5]),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_6=pair_mass(0, 6),
        mass_top5=mass_of(5),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_for_90pct=ncum(0.9),
        pt_0=pt[0],
        pt_1=pt[1],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_6=abs(eta[6]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
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
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
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
    # scale S = 37.97;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 37.97243 * (-0.08927529
        - 0.2542207 * max(0.0, 0.129 - Q.mass_over_sum_pt) / 0.07046263   # -25.4%  mass_over_sum_pt < 0.129
        + 0.2038 * max(0.0, 0.0191 - Q.sum_z_dr2) / 0.0133889   # +20.4%  sum_z_dr2 < 0.0191
        + 0.1790192 * max(0.0, 0.0127 - Q.sum_z_dr2) / 0.007777796   # +17.9%  sum_z_dr2 < 0.0127
        + 0.05450064 * max(0.0, Q.sj3_dr_max - 0.118) / 0.07809517   # +5.5%  sj3_dr_max > 0.118
        - 0.0509146 * max(0.0, 0.0774 - Q.sum_z_dr) / 0.02889912   # -5.1%  sum_z_dr < 0.0774
        - 0.04648115 * max(0.0, 0.0163 - Q.sum_z_dr2_top5) / 0.01138711   # -4.6%  sum_z_dr2_top5 < 0.0163
        - 0.04129916 * max(0.0, 61.6 - Q.mass) / 25.21269   # -4.1%  mass < 61.6
        + 0.0292123 * max(0.0, 0.028 - Q.e2) / 0.009243851   # +2.9%  e2 < 0.028
        - 0.02473707 * max(0.0, Q.sj3_dr_max - 0.177) / 0.04409986   # -2.5%  sj3_dr_max > 0.177
        + 0.0244983 * max(0.0, 0.000955 - Q.lam2) / 0.0007563091   # +2.4%  lam2 < 0.000955
        + 0.01897533 * max(0.0, Q.mass - 34.9) / 14.94895   # +1.9%  mass > 34.9
        + 0.01290357 * max(0.0, Q.N2 - 0.194) / 0.06413349   # +1.3%  N2 > 0.194
        + 0.01102866 * max(0.0, 0.162 - Q.max_dr) / 0.05923412   # +1.1%  max_dr < 0.162
        - 0.008600949 * max(0.0, Q.sj3_dr_max - 0.222) / 0.02815508   # -0.9%  sj3_dr_max > 0.222
        - 0.007766221 * max(0.0, Q.e2 - 0.0281) / 0.009830078   # -0.8%  e2 > 0.0281
        + 0.006455878 * max(0.0, 6.47e-05 - Q.lam2) * max(0.0, 0.0942 - Q.dr_2) / 1.361919e-06   # +0.6%  lam2 < 6.47e-05 and dr_2 < 0.0942
        - 0.005632668 * max(0.0, Q.log_sum_pt - 6.66) * max(0.0, Q.z_7 - 0.014) / 0.0008071174   # -0.6%  log_sum_pt > 6.66 and z_7 > 0.014
        - 0.005003446 * max(0.0, 0.208 - Q.N2) / 0.04367656   # -0.5%  N2 < 0.208
        - 0.003664531 * max(0.0, Q.sj3_pair_mass_min - 12.8) / 1.703197   # -0.4%  sj3_pair_mass_min > 12.8
        - 0.002320169 * max(0.0, Q.dr_4 - 0.0459) / 0.0336269   # -0.2%  dr_4 > 0.0459
        - 0.002263183 * max(0.0, Q.sum_pt - 921.0) / 9.832789   # -0.2%  sum_pt > 921
        - 0.001962152 * max(0.0, 0.0134 - Q.sum_z_dr2) * max(0.0, -0.00046 - Q.mean_phi) / 3.17054e-05   # -0.2%  sum_z_dr2 < 0.0134 and mean_phi < -0.00046
        - 0.001824546 * Q.lam2 / 0.0005288738   # -0.2%  lam2
        - 0.001219982 * max(0.0, Q.log_sum_pt - 6.36) * max(0.0, Q.mean_eta - 0.000739) / 0.0007051094   # -0.1%  log_sum_pt > 6.36 and mean_eta > 0.000739
        - 0.001091909 * max(0.0, -0.0056 - Q.mean_eta) / 0.00348424   # -0.1%  mean_eta < -0.0056
        + 0.0006037013 * max(0.0, Q.sum_pt - 803.0) * max(0.0, Q.pt_7 - 36.5) / 121.2911   # +0.1%  sum_pt > 803 and pt_7 > 36.5
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 9.371;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.370617 * (0.08782773
        + 0.1525277 * max(0.0, Q.log_sum_pt - 6.47) / 0.1451044   # +15.3%  log_sum_pt > 6.47
        - 0.1461195 * max(0.0, 0.00864 - Q.sum_z_dr2) / 0.00441687   # -14.6%  sum_z_dr2 < 0.00864
        - 0.1229188 * max(0.0, Q.sum_pt_top5 - 502.0) / 124.5216   # -12.3%  sum_pt_top5 > 502
        + 0.1220102 * max(0.0, Q.sum_pt_top5 - 375.0) / 224.1786   # +12.2%  sum_pt_top5 > 375
        + 0.05853365 * max(0.0, 0.0582 - Q.sum_z_dr) / 0.01714051   # +5.9%  sum_z_dr < 0.0582
        + 0.05135367 * max(0.0, 48.2 - Q.mass) / 16.20255   # +5.1%  mass < 48.2
        - 0.04725011 * max(0.0, 0.00414 - Q.sum_zz_dr2) / 0.001604213   # -4.7%  sum_zz_dr2 < 0.00414
        - 0.04581854 * max(0.0, Q.log_sum_pt - 6.28) * max(0.0, 0.787 - Q.z_dr_0p05_0p1) / 0.1690347   # -4.6%  log_sum_pt > 6.28 and z_dr_0p05_0p1 < 0.787
        - 0.03285871 * max(0.0, 0.00795 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 4.72) / 0.003718676   # -3.3%  lam1 < 0.00795 and n_pt_above_50 > 4.72
        - 0.03041035 * max(0.0, Q.log_sum_pt - 6.46) * max(0.0, 0.00812 - Q.zdr_6) / 0.0008768115   # -3.0%  log_sum_pt > 6.46 and zdr_6 < 0.00812
        - 0.02087528 * max(0.0, 0.0115 - Q.lam1) * max(0.0, 0.297 - Q.planar_flow) / 0.000770135   # -2.1%  lam1 < 0.0115 and planar_flow < 0.297
        - 0.01951617 * max(0.0, 0.000777 - Q.lam1) / 0.0001604198   # -2.0%  lam1 < 0.000777
        + 0.01692635 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00376) / 4.18497e-05   # +1.7%  mass_over_sum_pt_sq < 0.0081 and centroid_offset > 0.00376
        + 0.01656207 * max(0.0, Q.n_pt_above_50 - 5.28) / 0.5790926   # +1.7%  n_pt_above_50 > 5.28
        + 0.01275821 * max(0.0, Q.sj3_dr_max - 0.165) / 0.05002187   # +1.3%  sj3_dr_max > 0.165
        + 0.01225825 * max(0.0, Q.pt_7 - 41.8) / 1.81465   # +1.2%  pt_7 > 41.8
        - 0.0116877 * max(0.0, 0.0636 - Q.z_7) * max(0.0, 1.34 - Q.D2) / 0.004086602   # -1.2%  z_7 < 0.0636 and D2 < 1.34
        + 0.009718973 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 1.32 - Q.D2) / 0.01754774   # +1.0%  log_sum_pt > 6.6 and D2 < 1.32
        + 0.009292938 * max(0.0, 0.00866 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.0199) / 8.138371e-06   # +0.9%  sum_z_dr2 < 0.00866 and centroid_offset > 0.0199
        + 0.009210586 * max(0.0, Q.pt_7 - 26.8) * max(0.0, Q.sj2_dr - 0.142) / 0.3870353   # +0.9%  pt_7 > 26.8 and sj2_dr > 0.142
        - 0.008075743 * max(0.0, Q.sj3_dr_min - 0.00774) / 0.03709544   # -0.8%  sj3_dr_min > 0.00774
        + 0.007883162 * max(0.0, Q.log_sum_pt - 6.35) * max(0.0, Q.centroid_offset - 0.00653) / 0.001378173   # +0.8%  log_sum_pt > 6.35 and centroid_offset > 0.00653
        - 0.007804075 * max(0.0, Q.sj3_dr_max - 0.105) * max(0.0, Q.sj3_pair_mass_min - 1.87) / 1.010069   # -0.8%  sj3_dr_max > 0.105 and sj3_pair_mass_min > 1.87
        - 0.006479276 * max(0.0, 0.018 - Q.centroid_offset) / 0.006472794   # -0.6%  centroid_offset < 0.018
        - 0.005751978 * max(0.0, 0.0509 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.0245) / 0.001132344   # -0.6%  z_7 < 0.0509 and z_dr_0p05_0p1 > 0.0245
        - 0.004822358 * max(0.0, 0.0984 - Q.sum_z_dr) * max(0.0, Q.pair_mass_0_6 - 5.49) / 0.09184649   # -0.5%  sum_z_dr < 0.0984 and pair_mass_0_6 > 5.49
        + 0.004456985 * max(0.0, Q.pt_7 - 32.2) * max(0.0, Q.n_dr_0p1_0p2 - 0.624) / 5.233672   # +0.4%  pt_7 > 32.2 and n_dr_0p1_0p2 > 0.624
        - 0.003466414 * max(0.0, Q.pt_7 - 54.2) / 0.2952949   # -0.3%  pt_7 > 54.2
        - 0.002652239 * max(0.0, 0.0645 - Q.z_7) * max(0.0, Q.centroid_offset - 0.0194) / 3.944939e-05   # -0.3%  z_7 < 0.0645 and centroid_offset > 0.0194
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 9.668;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.668494 * (0.06154009
        + 0.2190663 * max(0.0, 792.0 - Q.sum_pt) / 114.4887   # +21.9%  sum_pt < 792
        - 0.1481158 * max(0.0, 719.0 - Q.sum_pt_top5) / 154.1504   # -14.8%  sum_pt_top5 < 719
        + 0.1196808 * max(0.0, 0.0118 - Q.lam1) / 0.007142795   # +12.0%  lam1 < 0.0118
        + 0.09845058 * max(0.0, Q.pt_7 - 21.8) / 13.57873   # +9.8%  pt_7 > 21.8
        - 0.09586397 * max(0.0, Q.LHA - 0.116) / 0.1314695   # -9.6%  LHA > 0.116
        + 0.05977238 * max(0.0, 0.00532 - Q.sum_z_dr2) / 0.002071358   # +6.0%  sum_z_dr2 < 0.00532
        - 0.04682741 * max(0.0, 15.8 - Q.sj3_pair_mass_min) / 10.31322   # -4.7%  sj3_pair_mass_min < 15.8
        - 0.04245309 * max(0.0, Q.z_7 - 0.0382) / 0.01739227   # -4.2%  z_7 > 0.0382
        + 0.03551267 * max(0.0, 849.0 - Q.sum_pt) * max(0.0, 0.0243 - Q.dr_5) / 0.2488073   # +3.6%  sum_pt < 849 and dr_5 < 0.0243
        - 0.02841649 * max(0.0, 806.0 - Q.sum_pt) * max(0.0, 0.0246 - Q.dr_5) / 0.1881813   # -2.8%  sum_pt < 806 and dr_5 < 0.0246
        - 0.02354297 * max(0.0, 70.9 - Q.sj3_pair_mass_max) * max(0.0, 0.0596 - Q.z_7) / 0.5281324   # -2.4%  sj3_pair_mass_max < 70.9 and z_7 < 0.0596
        + 0.02141916 * max(0.0, Q.pt_6 - 34.2) / 8.55748   # +2.1%  pt_6 > 34.2
        - 0.0114107 * max(0.0, 0.0112 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0168) / 1.918684e-05   # -1.1%  lam1 < 0.0112 and centroid_offset > 0.0168
        + 0.0107879 * max(0.0, Q.log_sum_pt - 6.84) / 0.008148651   # +1.1%  log_sum_pt > 6.84
        + 0.01015326 * max(0.0, 558.0 - Q.sum_pt) * max(0.0, 4.13 - Q.D2_b2) / 52.77783   # +1.0%  sum_pt < 558 and D2_b2 < 4.13
        - 0.008390306 * max(0.0, 0.0231 - Q.z_7) / 0.0007054054   # -0.8%  z_7 < 0.0231
        - 0.007447181 * max(0.0, 0.0213 - Q.dr_5) / 0.002482863   # -0.7%  dr_5 < 0.0213
        - 0.006036197 * max(0.0, 0.00574 - Q.lam1) * max(0.0, Q.max_dr - 0.0998) / 3.071628e-05   # -0.6%  lam1 < 0.00574 and max_dr > 0.0998
        - 0.00533761 * max(0.0, 0.00797 - Q.sum_z_dr) / 0.0002619627   # -0.5%  sum_z_dr < 0.00797
        - 0.001315217 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, Q.pt_6 - 32.7) / 0.06055319   # -0.1%  log_sum_pt > 6.89 and pt_6 > 32.7
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 61.82;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 61.82263 * (-0.3493866
        + 0.4646352 * max(0.0, 0.0172 - Q.mass_over_sum_pt_sq) / 0.01206932   # +46.5%  mass_over_sum_pt_sq < 0.0172
        - 0.1481466 * max(0.0, 0.00754 - Q.mass_over_sum_pt_sq) / 0.003800338   # -14.8%  mass_over_sum_pt_sq < 0.00754
        + 0.07957628 * max(0.0, Q.mass_over_sum_pt - 0.088) / 0.008800742   # +8.0%  mass_over_sum_pt > 0.088
        - 0.07639988 * max(0.0, 0.00905 - Q.lam1_plus_lam2) / 0.004746976   # -7.6%  lam1_plus_lam2 < 0.00905
        - 0.04998451 * max(0.0, 0.246 - Q.sj3_dr_max) / 0.1000056   # -5.0%  sj3_dr_max < 0.246
        + 0.03797759 * max(0.0, 0.195 - Q.sj3_dr_max) / 0.06345606   # +3.8%  sj3_dr_max < 0.195
        + 0.02949657 * max(0.0, 0.0493 - Q.e2) / 0.02418508   # +2.9%  e2 < 0.0493
        - 0.02235192 * max(0.0, Q.mass_over_sum_pt - 0.132) / 0.002437133   # -2.2%  mass_over_sum_pt > 0.132
        + 0.02094632 * max(0.0, 0.0086 - Q.lam1) / 0.004480819   # +2.1%  lam1 < 0.0086
        + 0.01703084 * max(0.0, 0.0059 - Q.lam1) / 0.002483234   # +1.7%  lam1 < 0.0059
        - 0.01255408 * max(0.0, Q.log_sum_pt - 6.13) / 0.4218077   # -1.3%  log_sum_pt > 6.13
        + 0.01203716 * max(0.0, Q.sum_z_dr - 0.0576) / 0.01793178   # +1.2%  sum_z_dr > 0.0576
        + 0.005933382 * max(0.0, 0.355 - Q.tau21_b2) / 0.2072414   # +0.6%  tau21_b2 < 0.355
        - 0.005172144 * max(0.0, Q.centroid_offset - 0.00899) / 0.009961232   # -0.5%  centroid_offset > 0.00899
        - 0.004116268 * max(0.0, 0.0248 - Q.centroid_offset) / 0.01131015   # -0.4%  centroid_offset < 0.0248
        - 0.003798546 * max(0.0, 4.83e-05 - Q.lam2) / 1.389563e-05   # -0.4%  lam2 < 4.83e-05
        + 0.002435525 * max(0.0, 0.0368 - Q.planar_flow) / 0.005246362   # +0.2%  planar_flow < 0.0368
        + 0.002312248 * max(0.0, Q.centroid_offset - 0.00578) * max(0.0, 7.78 - Q.n_pt_above_50) / 0.03751948   # +0.2%  centroid_offset > 0.00578 and n_pt_above_50 < 7.78
        + 0.001792151 * max(0.0, Q.sd_mass - 55.6) * max(0.0, 0.752 - Q.D2_b2) / 2.2704   # +0.2%  sd_mass > 55.6 and D2_b2 < 0.752
        - 0.001637452 * max(0.0, Q.mass - 67.1) / 2.811988   # -0.2%  mass > 67.1
        - 0.001236715 * max(0.0, Q.mass_over_sum_pt - 0.0794) * max(0.0, Q.pt_6 - 29.8) / 0.1117792   # -0.1%  mass_over_sum_pt > 0.0794 and pt_6 > 29.8
        + 0.0004285493 * max(0.0, Q.log_sum_pt - 6.87) / 0.005577693   # +0.0%  log_sum_pt > 6.87
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 84.97;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 84.96861 * (0.02671575
        + 0.2418236 * Q.mass_over_sum_pt_sq / 0.005887511   # +24.2%  mass_over_sum_pt_sq
        - 0.2130791 * max(0.0, Q.sum_zz_dr2 - 7.11e-05) / 0.005821555   # -21.3%  sum_zz_dr2 > 7.11e-05
        - 0.08089859 * max(0.0, 0.0188 - Q.sum_z_dr2) / 0.01311802   # -8.1%  sum_z_dr2 < 0.0188
        + 0.05839998 * max(0.0, 0.128 - Q.sum_z_dr) / 0.07160411   # +5.8%  sum_z_dr < 0.128
        + 0.04472793 * max(0.0, 0.00783 - Q.sum_zz_dr2) / 0.004025922   # +4.5%  sum_zz_dr2 < 0.00783
        - 0.03629002 * max(0.0, 0.00808 - Q.sum_z_dr2) / 0.003973599   # -3.6%  sum_z_dr2 < 0.00808
        + 0.03354493 * max(0.0, Q.e2 - 0.00829) / 0.02143057   # +3.4%  e2 > 0.00829
        - 0.02407831 * max(0.0, Q.mass_over_sum_pt - 0.091) / 0.008183604   # -2.4%  mass_over_sum_pt > 0.091
        + 0.02158838 * max(0.0, 42.5 - Q.mass) / 13.10239   # +2.2%  mass < 42.5
        + 0.01903215 * max(0.0, 0.229 - Q.N2) / 0.05408479   # +1.9%  N2 < 0.229
        - 0.01872618 * max(0.0, 0.342 - Q.sj3_dr_max) / 0.180811   # -1.9%  sj3_dr_max < 0.342
        - 0.01669913 * max(0.0, Q.mass - 40.7) / 11.82418   # -1.7%  mass > 40.7
        - 0.01317849 * max(0.0, 0.223 - Q.N2) * max(0.0, Q.eccentricity - 0.695) / 0.01461825   # -1.3%  N2 < 0.223 and eccentricity > 0.695
        - 0.01239928 * max(0.0, 6.59 - Q.log_sum_pt) / 0.1251246   # -1.2%  log_sum_pt < 6.59
        - 0.01237069 * max(0.0, 0.00098 - Q.lam2) / 0.0007786077   # -1.2%  lam2 < 0.00098
        + 0.01136406 * max(0.0, 1.09 - Q.D2) / 0.2240345   # +1.1%  D2 < 1.09
        + 0.008163673 * max(0.0, Q.max_dr - 0.123) / 0.03503313   # +0.8%  max_dr > 0.123
        - 0.008077284 * max(0.0, Q.tau1 - 0.0658) / 0.02098825   # -0.8%  tau1 > 0.0658
        + 0.007986063 * max(0.0, Q.sum_pt - 711.0) / 72.72933   # +0.8%  sum_pt > 711
        - 0.007954845 * max(0.0, 0.186 - Q.dr_7) / 0.1139818   # -0.8%  dr_7 < 0.186
        - 0.00775973 * max(0.0, 0.173 - Q.dr_5) / 0.1070347   # -0.8%  dr_5 < 0.173
        - 0.007740128 * max(0.0, 0.18 - Q.dr_6) / 0.1110926   # -0.8%  dr_6 < 0.18
        - 0.007349584 * max(0.0, 0.00016 - Q.lam1) / 1.508415e-05   # -0.7%  lam1 < 0.00016
        + 0.006321055 * max(0.0, Q.sd_mass - 31.4) / 14.83678   # +0.6%  sd_mass > 31.4
        - 0.005889774 * max(0.0, Q.sj3_dr_max - 0.232) / 0.02553295   # -0.6%  sj3_dr_max > 0.232
        - 0.005721825 * max(0.0, 0.0235 - Q.centroid_offset) * max(0.0, 0.585 - Q.z_dr_0p05_0p1) / 0.004191169   # -0.6%  centroid_offset < 0.0235 and z_dr_0p05_0p1 < 0.585
        - 0.005460572 * max(0.0, 76.4 - Q.mass) * max(0.0, 1.1 - Q.D2) / 4.034585   # -0.5%  mass < 76.4 and D2 < 1.1
        + 0.00514411 * max(0.0, Q.mass_top5 - 14.6) / 16.68274   # +0.5%  mass_top5 > 14.6
        - 0.004876998 * max(0.0, 0.000267 - Q.lam2) / 0.0001677699   # -0.5%  lam2 < 0.000267
        - 0.004719919 * max(0.0, Q.sd_rg - 0.212) / 0.0141213   # -0.5%  sd_rg > 0.212
        - 0.004715713 * max(0.0, Q.C2_b2 - 0.000674) / 0.003035512   # -0.5%  C2_b2 > 0.000674
        - 0.003916275 * max(0.0, 0.00438 - Q.sum_z_dr2) / 0.001569625   # -0.4%  sum_z_dr2 < 0.00438
        + 0.003796321 * max(0.0, Q.mean_eta - -0.0171) / 0.0185384   # +0.4%  mean_eta > -0.0171
        + 0.003761662 * max(0.0, Q.e2 - 0.0456) / 0.004134841   # +0.4%  e2 > 0.0456
        - 0.003272357 * max(0.0, 0.0683 - Q.z_6) / 0.01383322   # -0.3%  z_6 < 0.0683
        + 0.002959324 * max(0.0, 0.602 - Q.z_dr_0p05_0p1) / 0.382724   # +0.3%  z_dr_0p05_0p1 < 0.602
        + 0.002664575 * max(0.0, Q.tau1 - 0.0633) * max(0.0, 0.212 - Q.sj3_dr_min) / 0.002455588   # +0.3%  tau1 > 0.0633 and sj3_dr_min < 0.212
        - 0.002564789 * max(0.0, 1.0 - Q.D2) * max(0.0, 54.6 - Q.pt_7) / 3.27217   # -0.3%  D2 < 1 and pt_7 < 54.6
        + 0.002487097 * max(0.0, Q.sj3_dr_max - 0.34) / 0.007237162   # +0.2%  sj3_dr_max > 0.34
        + 0.001997672 * max(0.0, -0.000197 - Q.mean_eta) / 0.005457859   # +0.2%  mean_eta < -0.000197
        + 0.001779591 * max(0.0, Q.sd_rg - 0.284) / 0.004624139   # +0.2%  sd_rg > 0.284
        - 0.001731011 * max(0.0, 14.2 - Q.mass) / 2.020352   # -0.2%  mass < 14.2
        - 0.001710049 * max(0.0, Q.sj2_dr - 0.286) / 0.006156799   # -0.2%  sj2_dr > 0.286
        + 0.00170521 * max(0.0, Q.sd_rg - 0.209) * max(0.0, 181.0 - Q.pt_0) / 0.6411033   # +0.2%  sd_rg > 0.209 and pt_0 < 181
        - 0.001683177 * max(0.0, 4.19e-05 - Q.lam2) / 1.075318e-05   # -0.2%  lam2 < 4.19e-05
        - 0.001288887 * max(0.0, 24.9 - Q.pt_7) / 1.198194   # -0.1%  pt_7 < 24.9
        + 0.001284459 * max(0.0, -0.011 - Q.mean_phi) / 0.002161162   # +0.1%  mean_phi < -0.011
        - 0.001057975 * max(0.0, Q.sd_mass - 46.7) * max(0.0, 171.0 - Q.pt_0) / 133.3749   # -0.1%  sd_mass > 46.7 and pt_0 < 171
        + 0.0009456796 * max(0.0, 0.0362 - Q.planar_flow) / 0.005085638   # +0.1%  planar_flow < 0.0362
        - 0.0008391997 * max(0.0, Q.sj3_pair_mass_min - 11.3) / 1.906568   # -0.1%  sj3_pair_mass_min > 11.3
        - 0.0006189155 * max(0.0, 0.014 - Q.sum_z_dr) / 0.00116346   # -0.1%  sum_z_dr < 0.014
        + 0.0005866551 * max(0.0, Q.pt_7 - 40.6) / 2.130225   # +0.1%  pt_7 > 40.6
        - 0.0005431329 * max(0.0, 0.00119 - Q.lam2) * max(0.0, -0.0134 - Q.mean_phi) / 1.131109e-06   # -0.1%  lam2 < 0.00119 and mean_phi < -0.0134
        - 0.0004029543 * max(0.0, Q.sum_pt_top3 - 738.0) / 7.608549   # -0.0%  sum_pt_top3 > 738
        - 0.0003209599 * max(0.0, 0.0705 - Q.C2) * max(0.0, Q.mean_eta - 0.0232) / 2.572785e-05   # -0.0%  C2 < 0.0705 and mean_eta > 0.0232
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 6.217;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.21729 * (0.1262608
        + 0.1632237 * max(0.0, Q.pt_7 - 24.3) / 11.44091   # +16.3%  pt_7 > 24.3
        + 0.1335893 * max(0.0, 37.3 - Q.pt_7) / 5.890521   # +13.4%  pt_7 < 37.3
        - 0.1279916 * max(0.0, 6.7 - Q.log_sum_pt) / 0.1926782   # -12.8%  log_sum_pt < 6.7
        - 0.1139717 * max(0.0, 0.0873 - Q.z_7) * max(0.0, 0.0388 - Q.centroid_offset) / 0.0009015209   # -11.4%  z_7 < 0.0873 and centroid_offset < 0.0388
        - 0.09072963 * max(0.0, Q.pt_7 - 37.1) / 3.318191   # -9.1%  pt_7 > 37.1
        - 0.07007804 * max(0.0, Q.sum_zz_dr2 - 7.74e-06) / 0.005879831   # -7.0%  sum_zz_dr2 > 7.74e-06
        + 0.06402072 * max(0.0, 0.00503 - Q.sum_zz_dr2) * max(0.0, 0.0171 - Q.centroid_offset) / 1.638006e-05   # +6.4%  sum_zz_dr2 < 0.00503 and centroid_offset < 0.0171
        + 0.06055283 * max(0.0, 0.00116 - Q.sum_z_dr2) * max(0.0, 0.0252 - Q.centroid_offset) / 4.986418e-06   # +6.1%  sum_z_dr2 < 0.00116 and centroid_offset < 0.0252
        + 0.04857997 * max(0.0, 0.0464 - Q.z_7) * max(0.0, 84.8 - Q.mass_top5) / 0.3887204   # +4.9%  z_7 < 0.0464 and mass_top5 < 84.8
        + 0.02851788 * max(0.0, 0.00275 - Q.lam1) / 0.0008648971   # +2.9%  lam1 < 0.00275
        + 0.02510144 * max(0.0, Q.log_sum_pt - 6.66) * max(0.0, 0.00294 - Q.zdr_3) / 6.785344e-05   # +2.5%  log_sum_pt > 6.66 and zdr_3 < 0.00294
        + 0.02208881 * max(0.0, Q.mass - 58.0) / 4.869948   # +2.2%  mass > 58
        - 0.02156809 * max(0.0, 0.179 - Q.LHA) * max(0.0, 6.83 - Q.log_sum_pt) / 0.002315977   # -2.2%  LHA < 0.179 and log_sum_pt < 6.83
        - 0.01015004 * max(0.0, Q.max_pair_mass - 16.9) * max(0.0, 0.249 - Q.dr_max_012) / 0.163911   # -1.0%  max_pair_mass > 16.9 and dr_max_012 < 0.249
        - 0.009926448 * max(0.0, Q.log_sum_pt - 6.87) * max(0.0, 0.0045 - Q.zdr_3) / 1.686219e-05   # -1.0%  log_sum_pt > 6.87 and zdr_3 < 0.0045
        + 0.009909788 * max(0.0, 0.0267 - Q.z_6) * max(0.0, 0.209 - Q.abseta_6) / 0.0001055   # +1.0%  z_6 < 0.0267 and abseta_6 < 0.209
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 24.37;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.37433 * (0.3745743
        - 0.1753262 * max(0.0, 0.0124 - Q.lam1_plus_lam2) / 0.007523698   # -17.5%  lam1_plus_lam2 < 0.0124
        - 0.1561073 * Q.mass_over_sum_pt / 0.06127233   # -15.6%  mass_over_sum_pt
        - 0.1242889 * max(0.0, 0.00801 - Q.sum_z_dr2) / 0.003919092   # -12.4%  sum_z_dr2 < 0.00801
        + 0.09391121 * max(0.0, 0.00764 - Q.lam1) / 0.003721989   # +9.4%  lam1 < 0.00764
        - 0.06202604 * max(0.0, Q.z_7 - 0.0342) / 0.02040274   # -6.2%  z_7 > 0.0342
        - 0.05242207 * max(0.0, 0.175 - Q.max_dr) / 0.0686964   # -5.2%  max_dr < 0.175
        + 0.05059501 * max(0.0, 0.099 - Q.tau1) / 0.04533895   # +5.1%  tau1 < 0.099
        - 0.03600737 * max(0.0, 0.0631 - Q.z_6) / 0.01083525   # -3.6%  z_6 < 0.0631
        + 0.03123087 * max(0.0, Q.pt_7 - 29.4) / 7.536946   # +3.1%  pt_7 > 29.4
        + 0.03060274 * max(0.0, Q.z_6 - 0.061) / 0.008734444   # +3.1%  z_6 > 0.061
        - 0.03039154 * max(0.0, 0.12 - Q.tau1) / 0.06277742   # -3.0%  tau1 < 0.12
        + 0.03030651 * max(0.0, 42.9 - Q.pt_6) * max(0.0, 1010.0 - Q.sum_pt) / 1571.704   # +3.0%  pt_6 < 42.9 and sum_pt < 1010
        + 0.0260953 * max(0.0, 0.174 - Q.sj3_dr_max) / 0.05129481   # +2.6%  sj3_dr_max < 0.174
        - 0.01832327 * max(0.0, Q.pt_6 - 42.8) / 3.88363   # -1.8%  pt_6 > 42.8
        + 0.01482525 * max(0.0, 0.496 - Q.D2_b2) / 0.1780076   # +1.5%  D2_b2 < 0.496
        - 0.01182082 * max(0.0, 0.0161 - Q.lam1) * max(0.0, 0.243 - Q.planar_flow) / 0.0009935335   # -1.2%  lam1 < 0.0161 and planar_flow < 0.243
        + 0.01125364 * max(0.0, Q.sj3_dr_max - 0.187) / 0.03975363   # +1.1%  sj3_dr_max > 0.187
        + 0.008626362 * max(0.0, 0.00273 - Q.lam2) * max(0.0, 2.79 - Q.n_dr_0_0p05) / 0.001894251   # +0.9%  lam2 < 0.00273 and n_dr_0_0p05 < 2.79
        - 0.006943279 * max(0.0, Q.C2_b2 - 0.000142) / 0.003318388   # -0.7%  C2_b2 > 0.000142
        + 0.006302616 * max(0.0, Q.log_sum_pt - 6.67) / 0.04531624   # +0.6%  log_sum_pt > 6.67
        + 0.006005451 * max(0.0, Q.lam1 - 0.0132) / 0.001060716   # +0.6%  lam1 > 0.0132
        + 0.005018136 * max(0.0, 0.0331 - Q.z_7) / 0.002195937   # +0.5%  z_7 < 0.0331
        + 0.004253473 * max(0.0, Q.centroid_offset - 0.00912) * max(0.0, 0.0117 - Q.mean_phi2) / 6.688746e-05   # +0.4%  centroid_offset > 0.00912 and mean_phi2 < 0.0117
        + 0.00157596 * max(0.0, Q.centroid_offset - 0.0118) * max(0.0, Q.sum_pt_top5 - 544.0) / 0.241591   # +0.2%  centroid_offset > 0.0118 and sum_pt_top5 > 544
        + 0.001249256 * max(0.0, 24.7 - Q.pt_5) / 0.289998   # +0.1%  pt_5 < 24.7
        + 0.001030024 * max(0.0, 0.0128 - Q.lam1_plus_lam2) * max(0.0, -0.0223 - Q.mean_eta) / 2.59361e-06   # +0.1%  lam1_plus_lam2 < 0.0128 and mean_eta < -0.0223
        - 0.0009623384 * max(0.0, Q.centroid_offset - 0.0435) / 0.001196753   # -0.1%  centroid_offset > 0.0435
        + 0.0008478651 * max(0.0, 0.0125 - Q.lam1_plus_lam2) * max(0.0, -0.0203 - Q.mean_phi) / 2.939708e-06   # +0.1%  lam1_plus_lam2 < 0.0125 and mean_phi < -0.0203
        + 0.0006417847 * max(0.0, 0.0101 - Q.lam1) * max(0.0, Q.mean_eta - 0.0245) / 1.337015e-06   # +0.1%  lam1 < 0.0101 and mean_eta > 0.0245
        + 0.0005266112 * max(0.0, 0.0103 - Q.lam1_plus_lam2) * max(0.0, Q.mean_phi - 0.0265) / 9.438086e-07   # +0.1%  lam1_plus_lam2 < 0.0103 and mean_phi > 0.0265
        + 0.0004827999 * max(0.0, 0.0207 - Q.centroid_offset) / 0.008287272   # +0.0%  centroid_offset < 0.0207
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 35.73;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.73332 * (0.3582091
        - 0.1595753 * max(0.0, 0.00729 - Q.sum_z_dr2) / 0.003374056   # -16.0%  sum_z_dr2 < 0.00729
        - 0.1358035 * max(0.0, Q.lam1_plus_lam2 - 0.00493) / 0.003369938   # -13.6%  lam1_plus_lam2 > 0.00493
        + 0.1083778 * max(0.0, 0.0823 - Q.mass_over_sum_pt) / 0.03123145   # +10.8%  mass_over_sum_pt < 0.0823
        - 0.1053822 * max(0.0, 0.0879 - Q.sum_z_dr) / 0.03691818   # -10.5%  sum_z_dr < 0.0879
        - 0.1003926 * max(0.0, Q.sum_z_dr - 0.0187) / 0.04215465   # -10.0%  sum_z_dr > 0.0187
        + 0.08591478 * max(0.0, Q.mass_over_sum_pt - 0.0684) / 0.01527373   # +8.6%  mass_over_sum_pt > 0.0684
        - 0.08413194 * max(0.0, Q.sum_z_dr2 - 0.0019) / 0.005061134   # -8.4%  sum_z_dr2 > 0.0019
        + 0.02382309 * max(0.0, Q.mass - 27.9) / 19.08695   # +2.4%  mass > 27.9
        + 0.0226424 * max(0.0, Q.tau1 - 0.0922) / 0.01141168   # +2.3%  tau1 > 0.0922
        + 0.01882334 * max(0.0, Q.lam1_plus_lam2 - 0.0223) / 0.0004670975   # +1.9%  lam1_plus_lam2 > 0.0223
        - 0.01867458 * max(0.0, Q.sd_mass - 23.3) / 18.95751   # -1.9%  sd_mass > 23.3
        + 0.01752787 * max(0.0, Q.sd_rg - 0.113) / 0.04970865   # +1.8%  sd_rg > 0.113
        + 0.01662217 * max(0.0, Q.sj3_dr_max - 0.155) / 0.05551078   # +1.7%  sj3_dr_max > 0.155
        + 0.01311979 * max(0.0, Q.lam1_plus_lam2 - 0.0103) * max(0.0, 55.0 - Q.pt_6) / 0.031464   # +1.3%  lam1_plus_lam2 > 0.0103 and pt_6 < 55
        - 0.01257119 * max(0.0, 0.0243 - Q.centroid_offset) * max(0.0, 0.00677 - Q.C2_b2) / 6.362754e-05   # -1.3%  centroid_offset < 0.0243 and C2_b2 < 0.00677
        - 0.01193066 * max(0.0, 44.1 - Q.mass) / 13.93209   # -1.2%  mass < 44.1
        - 0.0113437 * max(0.0, Q.sd_rg - 0.192) / 0.01817705   # -1.1%  sd_rg > 0.192
        - 0.01057412 * max(0.0, Q.sj3_dr_max - 0.243) / 0.02289991   # -1.1%  sj3_dr_max > 0.243
        + 0.009898564 * max(0.0, 0.196 - Q.planar_flow) * max(0.0, Q.sd_mass - 42.1) / 1.081677   # +1.0%  planar_flow < 0.196 and sd_mass > 42.1
        - 0.007432379 * max(0.0, Q.mass - 78.4) / 1.383248   # -0.7%  mass > 78.4
        - 0.006731494 * max(0.0, 50.7 - Q.pt_6) / 12.02693   # -0.7%  pt_6 < 50.7
        - 0.00407714 * max(0.0, Q.lam2 - 0.000165) / 0.0004538621   # -0.4%  lam2 > 0.000165
        - 0.003908341 * max(0.0, 29.7 - Q.pt_7) / 2.383242   # -0.4%  pt_7 < 29.7
        + 0.003365974 * max(0.0, Q.e3 - 5.65e-05) / 5.417901e-05   # +0.3%  e3 > 5.65e-05
        + 0.003289838 * max(0.0, 643.0 - Q.sum_pt) / 38.9261   # +0.3%  sum_pt < 643
        + 0.001804091 * max(0.0, Q.mass_top5 - 61.7) / 1.05165   # +0.2%  mass_top5 > 61.7
        - 0.001631479 * max(0.0, Q.centroid_offset - 0.0409) * max(0.0, Q.n_pt_above_50 - 3.13) / 0.001714651   # -0.2%  centroid_offset > 0.0409 and n_pt_above_50 > 3.13
        - 0.0006295954 * max(0.0, 0.0556 - Q.tau1) * max(0.0, Q.n_dr_0p05_0p1 - 4.13) / 0.0009182666   # -0.1%  tau1 < 0.0556 and n_dr_0p05_0p1 > 4.13
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 36.58;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 36.58453 * (0.1623637
        + 0.1610687 * max(0.0, 0.00747 - Q.lam1_plus_lam2) / 0.003507514   # +16.1%  lam1_plus_lam2 < 0.00747
        - 0.139224 * max(0.0, 0.0168 - Q.sum_zz_dr2) / 0.01170907   # -13.9%  sum_zz_dr2 < 0.0168
        - 0.08839276 * max(0.0, 0.0609 - Q.sum_z_dr) / 0.0185851   # -8.8%  sum_z_dr < 0.0609
        - 0.08654369 * max(0.0, 0.00732 - Q.lam1) / 0.003479297   # -8.7%  lam1 < 0.00732
        - 0.05028609 * Q.sum_pt_top5 / 593.4494   # -5.0%  sum_pt_top5
        - 0.04818639 * max(0.0, 0.197 - Q.sj3_dr_max) * max(0.0, 0.00664 - Q.sum_z_dr2) / 0.0003634797   # -4.8%  sj3_dr_max < 0.197 and sum_z_dr2 < 0.00664
        + 0.03474924 * max(0.0, 0.00427 - Q.lam1_plus_lam2) / 0.001515238   # +3.5%  lam1_plus_lam2 < 0.00427
        + 0.03199605 * max(0.0, 0.00527 - Q.sum_z_dr2) / 0.002042863   # +3.2%  sum_z_dr2 < 0.00527
        - 0.03078487 * max(0.0, 6.84 - Q.log_sum_pt) / 0.3052168   # -3.1%  log_sum_pt < 6.84
        + 0.0291609 * max(0.0, 0.199 - Q.sj3_dr_max) * max(0.0, 0.00352 - Q.lam1_plus_lam2) / 0.0001760458   # +2.9%  sj3_dr_max < 0.199 and lam1_plus_lam2 < 0.00352
        + 0.02735467 * max(0.0, 0.055 - Q.tau1) * max(0.0, 0.00352 - Q.lam1_plus_lam2) / 4.905675e-05   # +2.7%  tau1 < 0.055 and lam1_plus_lam2 < 0.00352
        - 0.01996688 * max(0.0, Q.mass_over_sum_pt - 0.0857) / 0.009329232   # -2.0%  mass_over_sum_pt > 0.0857
        - 0.01662537 * max(0.0, Q.pt_7 - 20.1) / 15.09259   # -1.7%  pt_7 > 20.1
        + 0.01429529 * max(0.0, 0.345 - Q.sj3_dr_max) / 0.1835041   # +1.4%  sj3_dr_max < 0.345
        - 0.0136971 * max(0.0, 29.7 - Q.mass) / 7.369144   # -1.4%  mass < 29.7
        - 0.01363684 * max(0.0, 0.0011 - Q.lam2) / 0.0008861408   # -1.4%  lam2 < 0.0011
        + 0.01189374 * max(0.0, 0.199 - Q.sj3_dr_max) / 0.06602837   # +1.2%  sj3_dr_max < 0.199
        + 0.0111561 * max(0.0, 48.3 - Q.pt_7) / 14.37115   # +1.1%  pt_7 < 48.3
        + 0.01112134 * max(0.0, 0.0663 - Q.C2) / 0.04072764   # +1.1%  C2 < 0.0663
        - 0.01014865 * max(0.0, 0.00696 - Q.sum_z_dr2_top3) / 0.003804134   # -1.0%  sum_z_dr2_top3 < 0.00696
        - 0.00938547 * max(0.0, 0.0528 - Q.z_7) / 0.009083678   # -0.9%  z_7 < 0.0528
        + 0.008779184 * max(0.0, 0.201 - Q.sj3_dr_max) * max(0.0, 0.000172 - Q.lam2) / 8.610786e-06   # +0.9%  sj3_dr_max < 0.201 and lam2 < 0.000172
        - 0.008612311 * max(0.0, 21.6 - Q.mass) * max(0.0, 0.000203 - Q.lam2) / 0.000743107   # -0.9%  mass < 21.6 and lam2 < 0.000203
        - 0.008477558 * max(0.0, 0.00527 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.00673) / 1.174801e-05   # -0.8%  sum_z_dr2 < 0.00527 and centroid_offset > 0.00673
        + 0.00825443 * max(0.0, 0.0601 - Q.sum_z_dr) * max(0.0, Q.lam1_plus_lam2 - 0.000381) / 9.291829e-06   # +0.8%  sum_z_dr < 0.0601 and lam1_plus_lam2 > 0.000381
        + 0.008069315 * max(0.0, Q.z_7 - 0.0515) / 0.009111485   # +0.8%  z_7 > 0.0515
        + 0.007261197 * max(0.0, Q.mass_over_sum_pt - 0.132) / 0.002437133   # +0.7%  mass_over_sum_pt > 0.132
        - 0.006615666 * max(0.0, 0.00142 - Q.C2_b2) / 0.0007732621   # -0.7%  C2_b2 < 0.00142
        - 0.005672498 * max(0.0, 0.00531 - Q.sum_z_dr2) * max(0.0, 44.0 - Q.pt_7) / 0.02521576   # -0.6%  sum_z_dr2 < 0.00531 and pt_7 < 44
        + 0.005402789 * max(0.0, 0.000161 - Q.lam2) / 8.707422e-05   # +0.5%  lam2 < 0.000161
        + 0.004532931 * max(0.0, 0.167 - Q.dr_max_012) / 0.09755009   # +0.5%  dr_max_012 < 0.167
        - 0.004425106 * max(0.0, Q.sum_pt - 838.0) / 24.9446   # -0.4%  sum_pt > 838
        - 0.004357589 * max(0.0, Q.mass - 42.2) / 11.07086   # -0.4%  mass > 42.2
        + 0.004018997 * max(0.0, 68.8 - Q.pt_5) / 21.94524   # +0.4%  pt_5 < 68.8
        + 0.003984885 * max(0.0, Q.sd_mass - 23.9) / 18.6426   # +0.4%  sd_mass > 23.9
        - 0.003695108 * max(0.0, 3.1e-05 - Q.e3) / 1.601704e-05   # -0.4%  e3 < 3.1e-05
        - 0.003560262 * max(0.0, Q.sj2_dr - 0.142) / 0.04860094   # -0.4%  sj2_dr > 0.142
        - 0.003480483 * max(0.0, 0.00199 - Q.mean_phi) / 0.006496523   # -0.3%  mean_phi < 0.00199
        + 0.003177995 * max(0.0, 0.0142 - Q.zdr_1) / 0.006388213   # +0.3%  zdr_1 < 0.0142
        - 0.003018054 * max(0.0, 0.132 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.0149) / 0.0001110806   # -0.3%  sj3_dr_max < 0.132 and centroid_offset > 0.0149
        - 0.003007797 * max(0.0, Q.zdr_2 - 0.00454) / 0.004384018   # -0.3%  zdr_2 > 0.00454
        + 0.002775305 * max(0.0, 0.0205 - Q.zdr_0) / 0.008532203   # +0.3%  zdr_0 < 0.0205
        + 0.002490965 * max(0.0, 0.109 - Q.dr_7) / 0.04873305   # +0.2%  dr_7 < 0.109
        + 0.00245206 * max(0.0, Q.sj2_dr - 0.185) / 0.02812146   # +0.2%  sj2_dr > 0.185
        + 0.002421309 * max(0.0, 0.00714 - Q.zdr_5) / 0.003330167   # +0.2%  zdr_5 < 0.00714
        + 0.002215735 * max(0.0, 0.105 - Q.dr_6) / 0.04740446   # +0.2%  dr_6 < 0.105
        + 0.002174253 * max(0.0, 0.0095 - Q.zdr_3) / 0.004419113   # +0.2%  zdr_3 < 0.0095
        + 0.001931089 * max(0.0, 0.0959 - Q.dr_4) / 0.04360987   # +0.2%  dr_4 < 0.0959
        + 0.001446874 * max(0.0, Q.centroid_offset - 0.0287) / 0.002908417   # +0.1%  centroid_offset > 0.0287
        + 0.001434532 * max(0.0, 0.0116 - Q.centroid_offset) / 0.002883608   # +0.1%  centroid_offset < 0.0116
        - 0.001401534 * max(0.0, Q.sum_z_dr2_top5 - 0.0094) / 0.001878185   # -0.1%  sum_z_dr2_top5 > 0.0094
        + 0.001324376 * max(0.0, Q.zdr_2 - 0.0116) / 0.00165364   # +0.1%  zdr_2 > 0.0116
        + 0.001318316 * max(0.0, 0.0326 - Q.mass_over_sum_pt) * max(0.0, 0.0267 - Q.centroid_offset) / 0.0001227226   # +0.1%  mass_over_sum_pt < 0.0326 and centroid_offset < 0.0267
        - 0.001309396 * max(0.0, Q.lam1_plus_lam2 - 0.0185) / 0.0007931065   # -0.1%  lam1_plus_lam2 > 0.0185
        + 0.001224602 * max(0.0, Q.sum_pt - 953.0) / 6.656982   # +0.1%  sum_pt > 953
        + 0.001111991 * max(0.0, Q.max_pair_mass - 22.4) / 2.74876   # +0.1%  max_pair_mass > 22.4
        + 0.001026102 * max(0.0, Q.log_sum_pt - 6.75) * max(0.0, 0.000544 - Q.sum_z_dr2) / 5.679194e-06   # +0.1%  log_sum_pt > 6.75 and sum_z_dr2 < 0.000544
        + 0.0008344303 * max(0.0, -0.0161 - Q.mean_phi) / 0.001453678   # +0.1%  mean_phi < -0.0161
        - 0.0007394435 * max(0.0, -0.000608 - Q.mean_eta) / 0.005263073   # -0.1%  mean_eta < -0.000608
        + 0.0006735166 * max(0.0, Q.sd_mass - 59.2) / 4.086284   # +0.1%  sd_mass > 59.2
        + 0.0005926109 * max(0.0, Q.sum_z_dr - 0.124) / 0.002686542   # +0.1%  sum_z_dr > 0.124
        + 0.0003257445 * max(0.0, Q.dr_3 - 0.123) / 0.008218765   # +0.0%  dr_3 > 0.123
        - 0.0003007859 * max(0.0, 36.3 - Q.mass) * max(0.0, 0.543 - Q.D2_b2) / 0.2136721   # -0.0%  mass < 36.3 and D2_b2 < 0.543
        + 0.0002509074 * max(0.0, Q.dr_4 - 0.151) / 0.006039033   # +0.0%  dr_4 > 0.151
        - 0.0001458282 * max(0.0, Q.mass - 76.0) / 1.611799   # -0.0%  mass > 76
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 24.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.00641 * (0.03490735
        + 0.2563558 * max(0.0, 0.00725 - Q.lam1_plus_lam2) / 0.003344664   # +25.6%  lam1_plus_lam2 < 0.00725
        + 0.139122 * max(0.0, 0.00511 - Q.sum_z_dr2) / 0.001953111   # +13.9%  sum_z_dr2 < 0.00511
        - 0.1335366 * max(0.0, 0.00643 - Q.lam1) / 0.002836933   # -13.4%  lam1 < 0.00643
        - 0.1217212 * max(0.0, 0.00467 - Q.sum_zz_dr2) / 0.001885219   # -12.2%  sum_zz_dr2 < 0.00467
        - 0.09790739 * max(0.0, 0.0146 - Q.sum_zz_dr2) / 0.009752717   # -9.8%  sum_zz_dr2 < 0.0146
        + 0.07320211 * max(0.0, 58.8 - Q.mass) * max(0.0, 0.0232 - Q.centroid_offset) / 0.2780569   # +7.3%  mass < 58.8 and centroid_offset < 0.0232
        - 0.0508067 * max(0.0, Q.sj3_dr_max - 0.167) / 0.04898339   # -5.1%  sj3_dr_max > 0.167
        - 0.04422137 * max(0.0, 27.5 - Q.mass) / 6.512861   # -4.4%  mass < 27.5
        + 0.02567416 * max(0.0, Q.sj3_dr_max - 0.248) / 0.02177895   # +2.6%  sj3_dr_max > 0.248
        + 0.02549781 * max(0.0, 48.1 - Q.mass) * max(0.0, 6.87 - Q.log_sum_pt) / 4.567991   # +2.5%  mass < 48.1 and log_sum_pt < 6.87
        - 0.0138701 * max(0.0, 0.0497 - Q.sum_z_dr) * max(0.0, Q.z_6 - 0.0359) / 0.0002430448   # -1.4%  sum_z_dr < 0.0497 and z_6 > 0.0359
        + 0.007870744 * max(0.0, 0.158 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 34.5) / 0.4313888   # +0.8%  sj3_dr_max < 0.158 and pt_6 > 34.5
        + 0.006122764 * max(0.0, Q.lam2 - 0.000321) / 0.0004187623   # +0.6%  lam2 > 0.000321
        + 0.004091266 * max(0.0, Q.mass_over_sum_pt - 0.128) / 0.00283863   # +0.4%  mass_over_sum_pt > 0.128
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 21.04;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.03754 * (-0.3802727
        + 0.3906635 * Q.sum_z_dr / 0.05870426   # +39.1%  sum_z_dr
        + 0.2023463 * max(0.0, 0.316 - Q.LHA) / 0.08777045   # +20.2%  LHA < 0.316
        + 0.08644292 * max(0.0, 0.00752 - Q.sum_z_dr2) / 0.003544924   # +8.6%  sum_z_dr2 < 0.00752
        - 0.08152524 * max(0.0, 0.00636 - Q.lam1) / 0.002788764   # -8.2%  lam1 < 0.00636
        - 0.05223421 * max(0.0, Q.LHA - 0.316) / 0.01453544   # -5.2%  LHA > 0.316
        - 0.04246438 * max(0.0, 0.00111 - Q.lam2) / 0.0008951363   # -4.2%  lam2 < 0.00111
        + 0.02169911 * max(0.0, 0.0594 - Q.C2) / 0.03458302   # +2.2%  C2 < 0.0594
        - 0.02120405 * max(0.0, 0.181 - Q.LHA) / 0.01991433   # -2.1%  LHA < 0.181
        - 0.01999681 * max(0.0, 0.0029 - Q.sum_z_dr2) / 0.0009008214   # -2.0%  sum_z_dr2 < 0.0029
        + 0.01412773 * max(0.0, Q.sj3_pair_mass_min - 14.3) / 1.524168   # +1.4%  sj3_pair_mass_min > 14.3
        + 0.01242062 * max(0.0, Q.lam2 - 0.000892) / 0.0003371604   # +1.2%  lam2 > 0.000892
        - 0.01121508 * max(0.0, Q.mass - 44.7) / 9.871866   # -1.1%  mass > 44.7
        - 0.01106772 * max(0.0, Q.z_top5 - 0.802) / 0.03501317   # -1.1%  z_top5 > 0.802
        - 0.009086395 * max(0.0, 43.8 - Q.pt_7) * max(0.0, 6.59 - Q.log_sum_pt) / 1.124443   # -0.9%  pt_7 < 43.8 and log_sum_pt < 6.59
        - 0.007353219 * max(0.0, Q.sj3_pair_mass_min - 4.79) * max(0.0, Q.sj3_pairmin_over_m - 0.208) / 0.5408867   # -0.7%  sj3_pair_mass_min > 4.79 and sj3_pairmin_over_m > 0.208
        + 0.005329834 * max(0.0, 52.7 - Q.pt_7) * max(0.0, 0.795 - Q.D2) / 1.751978   # +0.5%  pt_7 < 52.7 and D2 < 0.795
        - 0.003858128 * max(0.0, Q.lam2 - 0.00383) / 0.0001375687   # -0.4%  lam2 > 0.00383
        - 0.003742438 * max(0.0, Q.lam1 - 0.00427) * max(0.0, 0.227 - Q.D2_b2) / 0.0002633167   # -0.4%  lam1 > 0.00427 and D2_b2 < 0.227
        + 0.002839834 * max(0.0, Q.sj3_dr_min - 0.14) / 0.007053496   # +0.3%  sj3_dr_min > 0.14
        + 0.0003825615 * max(0.0, Q.sum_pt - 1060.0) / 2.186998   # +0.0%  sum_pt > 1060
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 25.47;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.47015 * (0.1268151
        + 0.2453989 * max(0.0, 0.00908 - Q.lam1_plus_lam2) / 0.004771257   # +24.5%  lam1_plus_lam2 < 0.00908
        - 0.1820597 * max(0.0, 0.00848 - Q.sum_zz_dr2) / 0.004546166   # -18.2%  sum_zz_dr2 < 0.00848
        + 0.107653 * max(0.0, 0.0129 - Q.sum_z_dr2) / 0.007947649   # +10.8%  sum_z_dr2 < 0.0129
        - 0.1050893 * max(0.0, 0.076 - Q.sum_z_dr) / 0.02791073   # -10.5%  sum_z_dr < 0.076
        - 0.08505119 * max(0.0, 82.5 - Q.mass) / 43.23885   # -8.5%  mass < 82.5
        - 0.04640843 * max(0.0, 0.161 - Q.sj3_dr_max) / 0.04494409   # -4.6%  sj3_dr_max < 0.161
        - 0.03969696 * max(0.0, Q.lam1_plus_lam2 - 0.00389) / 0.003888798   # -4.0%  lam1_plus_lam2 > 0.00389
        + 0.0369197 * max(0.0, 0.128 - Q.sj3_dr_max) / 0.03155538   # +3.7%  sj3_dr_max < 0.128
        - 0.02451309 * max(0.0, Q.centroid_offset - 0.0045) / 0.01311664   # -2.5%  centroid_offset > 0.0045
        - 0.02354779 * max(0.0, 0.0169 - Q.centroid_offset) * max(0.0, 15.4 - Q.sj3_pair_mass_min) / 0.06462993   # -2.4%  centroid_offset < 0.0169 and sj3_pair_mass_min < 15.4
        + 0.02294436 * max(0.0, 0.286 - Q.planar_flow) / 0.1295779   # +2.3%  planar_flow < 0.286
        + 0.02148852 * max(0.0, Q.z_7 - 0.0164) / 0.03600762   # +2.1%  z_7 > 0.0164
        + 0.02052487 * max(0.0, 0.0459 - Q.tau1) / 0.01330208   # +2.1%  tau1 < 0.0459
        - 0.01583615 * max(0.0, Q.sj3_dr_max - 0.173) / 0.04599193   # -1.6%  sj3_dr_max > 0.173
        - 0.008061664 * max(0.0, 0.0348 - Q.centroid_offset) * max(0.0, 845.0 - Q.sum_pt) / 2.231867   # -0.8%  centroid_offset < 0.0348 and sum_pt < 845
        + 0.007980692 * max(0.0, 0.016 - Q.centroid_offset) * max(0.0, 0.334 - Q.D2_b2) / 0.0005107272   # +0.8%  centroid_offset < 0.016 and D2_b2 < 0.334
        - 0.00365607 * max(0.0, 32.2 - Q.pt_6) / 1.900421   # -0.4%  pt_6 < 32.2
        - 0.003169569 * max(0.0, 0.271 - Q.planar_flow) * max(0.0, 9.89e-06 - Q.e3) / 1.806027e-07   # -0.3%  planar_flow < 0.271 and e3 < 9.89e-06
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 12.65;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.65396 * (-0.4591447
        + 0.1138121 * max(0.0, 0.311 - Q.sj3_dr_max) / 0.1535367   # +11.4%  sj3_dr_max < 0.311
        - 0.1047201 * max(0.0, 61.2 - Q.mass) / 24.90835   # -10.5%  mass < 61.2
        - 0.08091002 * max(0.0, Q.tau1 - 0.0509) / 0.02812726   # -8.1%  tau1 > 0.0509
        - 0.0627191 * max(0.0, 0.242 - Q.max_dr) / 0.1238136   # -6.3%  max_dr < 0.242
        + 0.05880476 * max(0.0, 0.342 - Q.N2) / 0.1278545   # +5.9%  N2 < 0.342
        + 0.05462812 * max(0.0, Q.e2 - 0.025) / 0.01131362   # +5.5%  e2 > 0.025
        + 0.05425688 * max(0.0, 0.00793 - Q.lam1_plus_lam2) / 0.003857104   # +5.4%  lam1_plus_lam2 < 0.00793
        + 0.04926989 * max(0.0, 0.0298 - Q.e2) / 0.01023743   # +4.9%  e2 < 0.0298
        + 0.04592207 * max(0.0, 0.000277 - Q.e3) / 0.0002315124   # +4.6%  e3 < 0.000277
        - 0.04175915 * max(0.0, 0.0672 - Q.z_7) / 0.01809653   # -4.2%  z_7 < 0.0672
        - 0.03899331 * max(0.0, Q.z_dr_0_0p05 - 0.622) / 0.1684027   # -3.9%  z_dr_0_0p05 > 0.622
        - 0.0333784 * max(0.0, 0.525 - Q.planar_flow) / 0.297443   # -3.3%  planar_flow < 0.525
        + 0.03120335 * max(0.0, Q.centroid_offset - 0.00201) / 0.01524502   # +3.1%  centroid_offset > 0.00201
        + 0.03087528 * max(0.0, Q.sum_z_dr2_top5 - 0.00293) / 0.003998921   # +3.1%  sum_z_dr2_top5 > 0.00293
        + 0.02965188 * max(0.0, 0.115 - Q.sj3_dr_max) / 0.02680099   # +3.0%  sj3_dr_max < 0.115
        + 0.02807915 * max(0.0, 3.18 - Q.D3) / 1.870066   # +2.8%  D3 < 3.18
        + 0.02026841 * max(0.0, Q.mass - 63.8) / 3.442627   # +2.0%  mass > 63.8
        + 0.01758364 * max(0.0, Q.log_sum_pt - 6.67) / 0.04531624   # +1.8%  log_sum_pt > 6.67
        - 0.01552639 * max(0.0, 0.00413 - Q.sum_z_dr2_top5) / 0.001679234   # -1.6%  sum_z_dr2_top5 < 0.00413
        + 0.009098976 * max(0.0, Q.dr_3 - 0.0177) / 0.04837735   # +0.9%  dr_3 > 0.0177
        + 0.006651096 * max(0.0, Q.centroid_offset - 0.0246) / 0.003740565   # +0.7%  centroid_offset > 0.0246
        - 0.006198241 * max(0.0, Q.sj3_pair_mass_min - 8.44) / 2.391229   # -0.6%  sj3_pair_mass_min > 8.44
        + 0.005854539 * max(0.0, Q.mass_over_sum_pt_sq - 0.0189) / 0.0006023017   # +0.6%  mass_over_sum_pt_sq > 0.0189
        + 0.005729404 * max(0.0, Q.zdr_6 - 0.00489) / 0.001569257   # +0.6%  zdr_6 > 0.00489
        - 0.005261578 * max(0.0, Q.lam1 - 0.0175) / 0.0005840334   # -0.5%  lam1 > 0.0175
        + 0.005246477 * max(0.0, Q.zdr_7 - 0.00431) / 0.001472034   # +0.5%  zdr_7 > 0.00431
        + 0.005188124 * max(0.0, 53.5 - Q.pt_4) / 5.050025   # +0.5%  pt_4 < 53.5
        + 0.00513964 * max(0.0, Q.C2 - 0.0394) / 0.006940961   # +0.5%  C2 > 0.0394
        + 0.004971353 * max(0.0, Q.sum_z_dr - 0.125) / 0.002588778   # +0.5%  sum_z_dr > 0.125
        - 0.00455013 * max(0.0, Q.lam1_plus_lam2 - 0.0182) / 0.0008237077   # -0.5%  lam1_plus_lam2 > 0.0182
        + 0.003666006 * max(0.0, Q.dr_5 - 0.151) / 0.007039378   # +0.4%  dr_5 > 0.151
        + 0.003412527 * max(0.0, Q.n_dr_0p05_0p1 - 5.7) / 0.1853304   # +0.3%  n_dr_0p05_0p1 > 5.7
        + 0.003263628 * max(0.0, Q.dr_7 - 0.156) / 0.008462668   # +0.3%  dr_7 > 0.156
        + 0.002918415 * max(0.0, Q.dr_6 - 0.152) / 0.00790782   # +0.3%  dr_6 > 0.152
        + 0.002719588 * max(0.0, Q.zdr_5 - 0.00686) / 0.001313495   # +0.3%  zdr_5 > 0.00686
        + 0.002054416 * max(0.0, Q.dr_4 - 0.142) / 0.007064268   # +0.2%  dr_4 > 0.142
        - 0.001692705 * max(0.0, 417.0 - Q.sum_pt) / 1.563462   # -0.2%  sum_pt < 417
        + 0.001377409 * max(0.0, Q.dr_2 - 0.122) / 0.00675569   # +0.1%  dr_2 > 0.122
        - 0.001207834 * max(0.0, -0.0216 - Q.mean_phi) / 0.0009612503   # -0.1%  mean_phi < -0.0216
        - 0.0008805962 * max(0.0, Q.zdr_0 - 0.0443) / 0.0003882589   # -0.1%  zdr_0 > 0.0443
        - 0.00055529 * max(0.0, Q.mean_eta - 0.0279) / 0.0006692017   # -0.1%  mean_eta > 0.0279
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 22.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.43751 * (-0.1074094
        + 0.298748 * max(0.0, 0.146 - Q.sum_z_dr) / 0.08831571   # +29.9%  sum_z_dr < 0.146
        + 0.133192 * max(0.0, 0.0141 - Q.lam1_plus_lam2) / 0.008974464   # +13.3%  lam1_plus_lam2 < 0.0141
        + 0.1325692 * max(0.0, 0.0167 - Q.sum_zz_dr2) / 0.01161924   # +13.3%  sum_zz_dr2 < 0.0167
        - 0.1249605 * max(0.0, 0.0666 - Q.e2) / 0.03943465   # -12.5%  e2 < 0.0666
        - 0.0619592 * max(0.0, 0.00751 - Q.lam1_plus_lam2) / 0.003537431   # -6.2%  lam1_plus_lam2 < 0.00751
        - 0.05894492 * max(0.0, 0.139 - Q.sum_z_dr) * max(0.0, 6.84 - Q.log_sum_pt) / 0.02013055   # -5.9%  sum_z_dr < 0.139 and log_sum_pt < 6.84
        - 0.03694735 * max(0.0, 0.00076 - Q.lam2) / 0.0005838074   # -3.7%  lam2 < 0.00076
        - 0.03124537 * max(0.0, 0.14 - Q.sum_z_dr) * max(0.0, 38.9 - Q.pt_7) / 0.637335   # -3.1%  sum_z_dr < 0.14 and pt_7 < 38.9
        + 0.02824954 * max(0.0, Q.sum_pt_top5 - 579.0) * max(0.0, 37.7 - Q.pt_7) / 1022.338   # +2.8%  sum_pt_top5 > 579 and pt_7 < 37.7
        + 0.02773671 * max(0.0, Q.mass_over_sum_pt - 0.0802) / 0.01080456   # +2.8%  mass_over_sum_pt > 0.0802
        - 0.02770991 * max(0.0, 7.89e-05 - Q.e3) * max(0.0, 0.0314 - Q.centroid_offset) / 1.024286e-06   # -2.8%  e3 < 7.89e-05 and centroid_offset < 0.0314
        - 0.01483145 * max(0.0, 28.7 - Q.pt_7) / 2.079881   # -1.5%  pt_7 < 28.7
        - 0.008943348 * max(0.0, Q.sum_z_dr2 - 0.0157) / 0.001108655   # -0.9%  sum_z_dr2 > 0.0157
        - 0.004781691 * max(0.0, 24.1 - Q.pt_6) / 0.5737394   # -0.5%  pt_6 < 24.1
        + 0.003550626 * max(0.0, 0.141 - Q.sum_z_dr) * max(0.0, Q.z_7 - 0.0604) / 0.0003463792   # +0.4%  sum_z_dr < 0.141 and z_7 > 0.0604
        - 0.003395968 * max(0.0, Q.mass - 75.5) / 1.663692   # -0.3%  mass > 75.5
        - 0.001757063 * max(0.0, Q.sj3_dr23 - 0.311) / 0.007300763   # -0.2%  sj3_dr23 > 0.311
        - 0.0004770897 * max(0.0, Q.sum_pt - 1060.0) * max(0.0, 56.8 - Q.pt_6) / 36.16455   # -0.0%  sum_pt > 1060 and pt_6 < 56.8
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 63.94;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 63.93746 * (0.2314762
        - 0.1892566 * max(0.0, Q.sum_z_dr2 - 0.000857) / 0.005762184   # -18.9%  sum_z_dr2 > 0.000857
        - 0.1067154 * max(0.0, 0.00725 - Q.sum_z_dr2) / 0.003344664   # -10.7%  sum_z_dr2 < 0.00725
        + 0.07969819 * max(0.0, 0.0624 - Q.e2) / 0.03563426   # +8.0%  e2 < 0.0624
        - 0.05251578 * max(0.0, 74.5 - Q.sd_mass) / 41.97157   # -5.3%  sd_mass < 74.5
        - 0.04631124 * max(0.0, Q.sj2_dr - 0.0391) / 0.1165757   # -4.6%  sj2_dr > 0.0391
        + 0.03564235 * max(0.0, 59.0 - Q.sd_mass) / 28.9934   # +3.6%  sd_mass < 59
        - 0.03208149 * max(0.0, Q.sum_pt_top5 - 368.0) / 230.4729   # -3.2%  sum_pt_top5 > 368
        + 0.03168909 * max(0.0, Q.sj2_dr - 0.132) / 0.0544656   # +3.2%  sj2_dr > 0.132
        + 0.03149326 * max(0.0, Q.mass_over_sum_pt - 0.0828) / 0.01006799   # +3.1%  mass_over_sum_pt > 0.0828
        - 0.03134605 * max(0.0, 59.8 - Q.mass) / 23.85937   # -3.1%  mass < 59.8
        + 0.02919307 * max(0.0, Q.log_sum_pt - 6.2) / 0.3575729   # +2.9%  log_sum_pt > 6.2
        + 0.02866298 * max(0.0, Q.sum_z_dr2 - 0.0092) / 0.002113769   # +2.9%  sum_z_dr2 > 0.0092
        - 0.02700303 * max(0.0, 0.0808 - Q.sum_z_dr) / 0.03139101   # -2.7%  sum_z_dr < 0.0808
        - 0.02231032 * max(0.0, Q.sum_z_dr - 0.0886) / 0.007587581   # -2.2%  sum_z_dr > 0.0886
        - 0.02140031 * max(0.0, Q.mass_over_sum_pt - 0.0922) / 0.007955126   # -2.1%  mass_over_sum_pt > 0.0922
        - 0.01774401 * max(0.0, 0.241 - Q.sj3_dr_max) / 0.09614464   # -1.8%  sj3_dr_max < 0.241
        - 0.01656941 * max(0.0, 0.00449 - Q.sum_z_dr2) / 0.001624855   # -1.7%  sum_z_dr2 < 0.00449
        + 0.01610499 * max(0.0, Q.e2 - 0.0172) / 0.0155781   # +1.6%  e2 > 0.0172
        - 0.0151142 * max(0.0, Q.sd_mass - 4.58) / 30.0113   # -1.5%  sd_mass > 4.58
        + 0.01461221 * max(0.0, Q.LHA - 0.312) / 0.0155194   # +1.5%  LHA > 0.312
        + 0.01338848 * max(0.0, 0.00429 - Q.sum_zz_dr2) / 0.001681779   # +1.3%  sum_zz_dr2 < 0.00429
        + 0.01322436 * max(0.0, Q.lam1 - 0.00715) / 0.002119127   # +1.3%  lam1 > 0.00715
        + 0.01288006 * max(0.0, Q.tau1 - 0.0531) / 0.0270006   # +1.3%  tau1 > 0.0531
        + 0.01094663 * max(0.0, Q.lam1_plus_lam2 - 0.0126) / 0.00153824   # +1.1%  lam1_plus_lam2 > 0.0126
        - 0.01018315 * max(0.0, 0.188 - Q.sd_rg) / 0.08894599   # -1.0%  sd_rg < 0.188
        + 0.00878062 * max(0.0, Q.e2 - 0.0431) / 0.004639757   # +0.9%  e2 > 0.0431
        + 0.00819244 * max(0.0, Q.sj2_dr - 0.128) * max(0.0, 0.109 - Q.D2_b2) / 0.001446972   # +0.8%  sj2_dr > 0.128 and D2_b2 < 0.109
        + 0.007125674 * max(0.0, Q.z_7 - 0.0231) / 0.02977761   # +0.7%  z_7 > 0.0231
        - 0.005939235 * max(0.0, 0.114 - Q.D2_b2) / 0.02019892   # -0.6%  D2_b2 < 0.114
        + 0.005681007 * max(0.0, 0.12 - Q.planar_flow) / 0.03733085   # +0.6%  planar_flow < 0.12
        + 0.004556069 * max(0.0, Q.lam1_plus_lam2 - 0.0186) / 0.0007830739   # +0.5%  lam1_plus_lam2 > 0.0186
        - 0.0042671 * max(0.0, Q.sj2_dr - 0.209) / 0.02066875   # -0.4%  sj2_dr > 0.209
        - 0.004236928 * max(0.0, 0.0224 - Q.centroid_offset) / 0.009505207   # -0.4%  centroid_offset < 0.0224
        - 0.003880264 * max(0.0, Q.z_dr_0p05_0p1 - 0.745) * max(0.0, 0.0244 - Q.C2_b2) / 0.0005513204   # -0.4%  z_dr_0p05_0p1 > 0.745 and C2_b2 < 0.0244
        - 0.003830282 * max(0.0, Q.e2 - 0.0618) / 0.001928335   # -0.4%  e2 > 0.0618
        + 0.003504247 * Q.D2 / 1.513869   # +0.4%  D2
        - 0.003434279 * max(0.0, Q.centroid_offset - 0.0264) / 0.003347242   # -0.3%  centroid_offset > 0.0264
        + 0.003369373 * max(0.0, Q.z_dr_0p05_0p1 - 0.754) * max(0.0, 1.03 - Q.n_dr_0p2_0p4) / 0.02091545   # +0.3%  z_dr_0p05_0p1 > 0.754 and n_dr_0p2_0p4 < 1.03
        - 0.003197599 * max(0.0, Q.sj2_dr - 0.191) * max(0.0, 0.106 - Q.D2_b2) / 0.0005175857   # -0.3%  sj2_dr > 0.191 and D2_b2 < 0.106
        - 0.003153029 * max(0.0, 1.19 - Q.D2) / 0.2702368   # -0.3%  D2 < 1.19
        - 0.002851676 * max(0.0, Q.pt_7 - 33.8) / 4.849174   # -0.3%  pt_7 > 33.8
        + 0.002850165 * max(0.0, 4.89 - Q.n_dr_0_0p05) / 1.890377   # +0.3%  n_dr_0_0p05 < 4.89
        - 0.002390696 * max(0.0, Q.pt_5 - 37.3) / 12.2284   # -0.2%  pt_5 > 37.3
        - 0.002351591 * max(0.0, 0.115 - Q.planar_flow) * max(0.0, 0.00711 - Q.lam1) / 5.71691e-05   # -0.2%  planar_flow < 0.115 and lam1 < 0.00711
        + 0.002227255 * max(0.0, 0.0406 - Q.sj3_dr_max) / 0.004436294   # +0.2%  sj3_dr_max < 0.0406
        + 0.001664883 * max(0.0, Q.centroid_offset - 0.0258) * max(0.0, 161.0 - Q.pt_1) / 0.22843   # +0.2%  centroid_offset > 0.0258 and pt_1 < 161
        + 0.001391272 * max(0.0, 34.8 - Q.pt_7) / 4.492647   # +0.1%  pt_7 < 34.8
        + 0.001311388 * max(0.0, 0.674 - Q.z_dr_0p05_0p1) * max(0.0, 687.0 - Q.sum_pt) / 21.49918   # +0.1%  z_dr_0p05_0p1 < 0.674 and sum_pt < 687
        + 0.001188631 * max(0.0, Q.mass_top5 - 56.7) / 1.560535   # +0.1%  mass_top5 > 56.7
        + 0.001125604 * max(0.0, 0.00646 - Q.mean_phi) / 0.009570251   # +0.1%  mean_phi < 0.00646
        - 0.0009520969 * max(0.0, 0.121 - Q.planar_flow) * max(0.0, 665.0 - Q.sum_pt_top5) / 3.539224   # -0.1%  planar_flow < 0.121 and sum_pt_top5 < 665
        - 0.0004598941 * max(0.0, Q.centroid_offset - 0.0554) * max(0.0, 0.00154 - Q.C2_b2) / 2.579338e-07   # -0.0%  centroid_offset > 0.0554 and C2_b2 < 0.00154
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 26.96;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.95557 * (-0.1098103
        + 0.1274 * max(0.0, 0.0189 - Q.lam1_plus_lam2) / 0.01320823   # +12.7%  lam1_plus_lam2 < 0.0189
        - 0.1050142 * max(0.0, Q.lam1_plus_lam2 - 0.00687) / 0.002645532   # -10.5%  lam1_plus_lam2 > 0.00687
        - 0.08506104 * max(0.0, 0.0975 - Q.sum_z_dr) / 0.04487023   # -8.5%  sum_z_dr < 0.0975
        + 0.07257914 * max(0.0, Q.lam1 - 0.00425) / 0.003212499   # +7.3%  lam1 > 0.00425
        + 0.05985381 * max(0.0, 0.0794 - Q.mass_over_sum_pt) / 0.02917529   # +6.0%  mass_over_sum_pt < 0.0794
        - 0.04624669 * max(0.0, 0.191 - Q.sj2_dr) / 0.06702183   # -4.6%  sj2_dr < 0.191
        - 0.04506179 * max(0.0, Q.sum_z_dr - 0.0842) / 0.00849417   # -4.5%  sum_z_dr > 0.0842
        + 0.04404472 * max(0.0, 0.165 - Q.sj2_dr) / 0.05139613   # +4.4%  sj2_dr < 0.165
        + 0.03806569 * max(0.0, Q.lam1 - 0.00647) / 0.002310996   # +3.8%  lam1 > 0.00647
        - 0.03650588 * max(0.0, 0.00433 - Q.lam1_plus_lam2) / 0.001544799   # -3.7%  lam1_plus_lam2 < 0.00433
        - 0.03602748 * max(0.0, 0.00847 - Q.sum_zz_dr2) / 0.004538043   # -3.6%  sum_zz_dr2 < 0.00847
        + 0.02811882 * max(0.0, Q.LHA - 0.317) / 0.01430111   # +2.8%  LHA > 0.317
        - 0.02405846 * max(0.0, 68.3 - Q.mass) / 30.59007   # -2.4%  mass < 68.3
        + 0.02254447 * max(0.0, 0.00114 - Q.lam2) / 0.0009221532   # +2.3%  lam2 < 0.00114
        + 0.02028526 * max(0.0, 6.85 - Q.log_sum_pt) / 0.3142533   # +2.0%  log_sum_pt < 6.85
        + 0.01724616 * Q.lam2 / 0.0005288738   # +1.7%  lam2
        + 0.01675411 * max(0.0, 0.00768 - Q.sum_z_dr2_top2) / 0.004575648   # +1.7%  sum_z_dr2_top2 < 0.00768
        + 0.0157288 * max(0.0, 0.061 - Q.tau1) / 0.02078327   # +1.6%  tau1 < 0.061
        + 0.01482818 * max(0.0, Q.sj3_dr_max - 0.171) / 0.04696851   # +1.5%  sj3_dr_max > 0.171
        - 0.01458162 * max(0.0, 0.133 - Q.mass_over_sum_pt) * max(0.0, 0.0138 - Q.tau21_b2) / 8.953438e-05   # -1.5%  mass_over_sum_pt < 0.133 and tau21_b2 < 0.0138
        + 0.0144943 * max(0.0, 0.0139 - Q.lam1_plus_lam2) * max(0.0, 0.014 - Q.tau21_b2) / 1.198473e-05   # +1.4%  lam1_plus_lam2 < 0.0139 and tau21_b2 < 0.014
        - 0.01427965 * max(0.0, 691.0 - Q.sum_pt_top5) / 133.6514   # -1.4%  sum_pt_top5 < 691
        + 0.01384482 * max(0.0, Q.tau1 - 0.101) / 0.009283456   # +1.4%  tau1 > 0.101
        - 0.0133314 * max(0.0, 0.00772 - Q.sum_z_dr2) * max(0.0, 0.782 - Q.D2) / 0.0001170539   # -1.3%  sum_z_dr2 < 0.00772 and D2 < 0.782
        - 0.00859904 * max(0.0, 0.0207 - Q.centroid_offset) * max(0.0, 0.263 - Q.dr01) / 0.001796838   # -0.9%  centroid_offset < 0.0207 and dr01 < 0.263
        - 0.00824305 * max(0.0, Q.sj3_dr_max - 0.247) / 0.02199962   # -0.8%  sj3_dr_max > 0.247
        + 0.007742027 * max(0.0, 0.00711 - Q.mass_over_sum_pt_sq) * max(0.0, 0.769 - Q.D2) / 8.659368e-05   # +0.8%  mass_over_sum_pt_sq < 0.00711 and D2 < 0.769
        + 0.006976343 * max(0.0, 0.23 - Q.N2) * max(0.0, Q.sum_pt_top5 - 420.0) / 9.845619   # +0.7%  N2 < 0.23 and sum_pt_top5 > 420
        - 0.006909704 * max(0.0, Q.z_dr_0p05_0p1 - 0.723) / 0.02735022   # -0.7%  z_dr_0p05_0p1 > 0.723
        - 0.006462603 * max(0.0, 0.024 - Q.sum_z_dr) / 0.003498055   # -0.6%  sum_z_dr < 0.024
        + 0.006182183 * max(0.0, Q.z_dr_0p05_0p1 - 0.706) * max(0.0, 1.99 - Q.n_dr_0p2_0p4) / 0.05573387   # +0.6%  z_dr_0p05_0p1 > 0.706 and n_dr_0p2_0p4 < 1.99
        + 0.003982783 * max(0.0, Q.log_sum_pt - 6.82) / 0.01042312   # +0.4%  log_sum_pt > 6.82
        - 0.00370685 * max(0.0, Q.mass - 78.5) / 1.374419   # -0.4%  mass > 78.5
        - 0.002466065 * max(0.0, Q.sum_pt_top5 - 810.0) / 11.95579   # -0.2%  sum_pt_top5 > 810
        + 0.002425597 * max(0.0, 0.000286 - Q.lam2) * max(0.0, 0.0995 - Q.D2_b2) / 3.534235e-06   # +0.2%  lam2 < 0.000286 and D2_b2 < 0.0995
        - 0.002277666 * max(0.0, 0.202 - Q.N2) * max(0.0, 0.686 - Q.z_dr_0p05_0p1) / 0.01096353   # -0.2%  N2 < 0.202 and z_dr_0p05_0p1 < 0.686
        - 0.002113139 * max(0.0, Q.centroid_offset - 0.0459) / 0.001033772   # -0.2%  centroid_offset > 0.0459
        + 0.001946509 * max(0.0, Q.D2 - 1.64) / 0.4099161   # +0.2%  D2 > 1.64
        + 0.001710103 * max(0.0, 0.007 - Q.mean_phi) / 0.00997766   # +0.2%  mean_phi < 0.007
        + 0.001253068 * max(0.0, Q.mass_top5 - 63.4) / 0.9178577   # +0.1%  mass_top5 > 63.4
        + 0.0005203706 * max(0.0, Q.zdr_0 - 0.0427) / 0.0004452979   # +0.1%  zdr_0 > 0.0427
        - 0.0003249237 * max(0.0, 0.197 - Q.N2) * max(0.0, 5.77 - Q.n_for_90pct) / 0.003161915   # -0.0%  N2 < 0.197 and n_for_90pct < 5.77
        - 0.0002014575 * max(0.0, 0.0239 - Q.tau1) * max(0.0, Q.zdr_3 - 0.00443) / 2.715201e-07   # -0.0%  tau1 < 0.0239 and zdr_3 > 0.00443
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0229714285714286, 0.748173949579832, 2.1000061974789914, 0.9546689075630252, 1.1777319327731093, 1.770180462184874, 0.9561115546218487, 1.8348495798319329, 0.25944747899159665, 3.000528571428571, 2.0151745798319327, 2.1356338235294117, 0.10178319327731092, 3.2897306722689077, 0.5149353991596639, 0.3728655462184874]
T = [2.3434446568080354, 1.3661728663340333, 3.3858663964023106, 2.4938546349789914, 2.857093894432773]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +39%, n9 +22%, n5 -14%, n1 +12%, n0 -7%, n6 +4% ...
            + 0.3850513 * h[2] / H_AVG[2]
            + 0.2200674 * h[9] / H_AVG[9]
            - 0.1416329 * h[5] / H_AVG[5]
            + 0.1247119 * h[1] / H_AVG[1]
            - 0.06820698 * h[0] / H_AVG[0]
            + 0.04462435 * h[6] / H_AVG[6]
            - 0.01570514 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -18%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5576549 * h[9] / H_AVG[9]
            - 0.1843814 * h[10] / H_AVG[10]
            + 0.08748084 * h[6] / H_AVG[6]
            - 0.08081874 * h[4] / H_AVG[4]
            + 0.06073698 * h[5] / H_AVG[5]
            + 0.01705794 * h[15] / H_AVG[15]
            + 0.01186926 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +12%, n14 -11%, n0 +10%, n6 -9% ...
            + 0.2365311 * h[11] / H_AVG[11]
            - 0.1409785 * h[3] / H_AVG[3]
            + 0.1185438 * h[7] / H_AVG[7]
            - 0.1140628 * h[14] / H_AVG[14]
            + 0.1038571 * h[0] / H_AVG[0]
            - 0.08824473 * h[6] / H_AVG[6]
            - 0.07571033 * h[15] / H_AVG[15]
            + 0.0683161 * h[13] / H_AVG[13]
            - 0.02769351 * h[9] / H_AVG[9]
            - 0.01915665 * h[8] / H_AVG[8]
            - 0.006905304 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +34%, n3 -22%, n6 -14%, n14 +8%, n13 +7%, n9 -4% ...
            + 0.3448821 * h[7] / H_AVG[7]
            - 0.2153298 * h[3] / H_AVG[3]
            - 0.1437701 * h[6] / H_AVG[6]
            + 0.07743065 * h[14] / H_AVG[14]
            + 0.07214019 * h[13] / H_AVG[13]
            - 0.03759903 * h[9] / H_AVG[9]
            + 0.03750088 * h[1] / H_AVG[1]
            + 0.03689482 * h[4] / H_AVG[4]
            - 0.02336152 * h[15] / H_AVG[15]
            + 0.01109089 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +26%, n5 -15%, n4 +5%, n3 +2%, n12 -2% ...
            - 0.4677666 * h[13] / H_AVG[13]
            + 0.2644962 * h[10] / H_AVG[10]
            - 0.1548934 * h[5] / H_AVG[5]
            + 0.05152666 * h[4] / H_AVG[4]
            + 0.02088374 * h[3] / H_AVG[3]
            - 0.01781236 * h[12] / H_AVG[12]
            + 0.01702653 * h[8] / H_AVG[8]
            + 0.005594471 * h[0] / H_AVG[0]
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
