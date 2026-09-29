"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network (step 4; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.2%   (on for 86% of jets)
  neuron  9:  11.8%   (on for 66% of jets)
  neuron  7:  10.6%   (on for 59% of jets)
  neuron  6:   8.5%   (on for 36% of jets)
  neuron  3:   8.2%   (on for 23% of jets)
  neuron 10:   7.8%   (on for 75% of jets)
  neuron  2:   7.3%   (on for 89% of jets)
  neuron  5:   6.9%   (on for 52% of jets)
  neuron 11:   6.3%   (on for 74% of jets)
  neuron  0:   3.8%   (on for 41% of jets)
  neuron 14:   3.8%   (on for 26% of jets)
  neuron  1:   3.5%   (on for 66% of jets)
  neuron  4:   3.2%   (on for 52% of jets)
  neuron 15:   2.6%   (on for 29% of jets)
  neuron  8:   1.0%   (on for 32% of jets)
  neuron 12:   0.4%   (on for 8% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 89.9% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
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
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.pt1_over_pt0           pT1 / pT0
  Q.pt1_dr01               pT1 · ΔR01
  Q.z_3rd                  3rd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_6               |Δη| of particle 6
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_5                  ΔR between particle 5 and the hardest particle
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_5                  Δη of particle 5
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
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
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        z_3rd=zs[2],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        abseta_0=abs(eta[0]),
        abseta_6=abs(eta[6]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_5=math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_5=eta[5],
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 17.91;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.9054 * (-0.06031699
        + 0.3368074 * max(0.0, 0.0128 - Q.sum_z_dr2) / 0.007862676   # +33.7%  sum_z_dr2 < 0.0128
        - 0.1630446 * max(0.0, 0.083 - Q.sum_z_dr) / 0.03306206   # -16.3%  sum_z_dr < 0.083
        - 0.09088156 * max(0.0, 0.00559 - Q.sum_z_dr2) / 0.002229139   # -9.1%  sum_z_dr2 < 0.00559
        + 0.06240051 * max(0.0, 0.0316 - Q.e2) / 0.01127453   # +6.2%  e2 < 0.0316
        - 0.05959769 * max(0.0, 0.00374 - Q.lam1_plus_lam2) / 0.001264361   # -6.0%  lam1_plus_lam2 < 0.00374
        - 0.04541052 * max(0.0, 0.0135 - Q.tau2) / 0.005245766   # -4.5%  tau2 < 0.0135
        - 0.04222365 * max(0.0, 0.178 - Q.sd_rg) / 0.08164488   # -4.2%  sd_rg < 0.178
        + 0.0377961 * max(0.0, 0.113 - Q.sd_rg) / 0.04452331   # +3.8%  sd_rg < 0.113
        + 0.0208305 * max(0.0, Q.n_dr_0_0p05 - 5.06) / 1.065653   # +2.1%  n_dr_0_0p05 > 5.06
        + 0.0204313 * max(0.0, 0.171 - Q.planar_flow) * max(0.0, 1.21 - Q.D2_b2) / 0.06169152   # +2.0%  planar_flow < 0.171 and D2_b2 < 1.21
        - 0.01791567 * max(0.0, 0.0183 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.0105) / 7.29062e-05   # -1.8%  sum_z_dr2 < 0.0183 and centroid_offset > 0.0105
        + 0.01666129 * max(0.0, 0.145 - Q.planar_flow) / 0.04939192   # +1.7%  planar_flow < 0.145
        - 0.01541044 * max(0.0, 0.0153 - Q.sum_z_dr) / 0.001415026   # -1.5%  sum_z_dr < 0.0153
        - 0.01450396 * max(0.0, Q.centroid_offset - 0.05) / 0.0008015409   # -1.5%  centroid_offset > 0.05
        - 0.01419643 * max(0.0, 783.0 - Q.sum_pt) / 108.6294   # -1.4%  sum_pt < 783
        - 0.01299524 * max(0.0, 747.0 - Q.sum_pt) * max(0.0, 1.07 - Q.D2_b2) / 44.23671   # -1.3%  sum_pt < 747 and D2_b2 < 1.07
        - 0.006030624 * max(0.0, 0.262 - Q.D2_b2) * max(0.0, Q.pt_5 - 16.1) / 2.522915   # -0.6%  D2_b2 < 0.262 and pt_5 > 16.1
        + 0.005742429 * max(0.0, 0.0189 - Q.sum_z_dr2) * max(0.0, 0.0889 - Q.M3) / 0.0001886615   # +0.6%  sum_z_dr2 < 0.0189 and M3 < 0.0889
        - 0.004047893 * max(0.0, 0.00505 - Q.sum_z_dr2) * max(0.0, 0.094 - Q.M3) / 2.85351e-05   # -0.4%  sum_z_dr2 < 0.00505 and M3 < 0.094
        - 0.00306127 * max(0.0, 8e-05 - Q.e3) * max(0.0, Q.dr0_7 - 0.241) / 1.022636e-07   # -0.3%  e3 < 8e-05 and dr0_7 > 0.241
        - 0.002542864 * max(0.0, 0.0631 - Q.tau1) * max(0.0, Q.dr0_6 - 0.177) / 6.80583e-05   # -0.3%  tau1 < 0.0631 and dr0_6 > 0.177
        + 0.001956111 * max(0.0, 0.0036 - Q.lam1_plus_lam2) * max(0.0, Q.dr0_7 - 0.179) / 1.71691e-06   # +0.2%  lam1_plus_lam2 < 0.0036 and dr0_7 > 0.179
        - 0.001892394 * max(0.0, 0.0629 - Q.tau1) * max(0.0, Q.dr_5 - 0.163) / 3.504042e-05   # -0.2%  tau1 < 0.0629 and dr_5 > 0.163
        + 0.001870253 * max(0.0, 0.00438 - Q.lam1_plus_lam2) * max(0.0, Q.dr0_6 - 0.177) / 1.790783e-06   # +0.2%  lam1_plus_lam2 < 0.00438 and dr0_6 > 0.177
        + 0.001749277 * max(0.0, 0.00558 - Q.sum_z_dr2) * max(0.0, Q.dr0_5 - 0.172) / 2.677052e-06   # +0.2%  sum_z_dr2 < 0.00558 and dr0_5 > 0.172
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 15.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.63548 * (0.08122554
        - 0.1388554 * max(0.0, 0.00868 - Q.lam1_plus_lam2) / 0.004448915   # -13.9%  lam1_plus_lam2 < 0.00868
        + 0.1195902 * max(0.0, 0.00807 - Q.lam1) * max(0.0, 0.147 - Q.dr_0) / 0.0004973006   # +12.0%  lam1 < 0.00807 and dr_0 < 0.147
        + 0.09756094 * max(0.0, 0.00594 - Q.sum_zz_dr2) / 0.002648284   # +9.8%  sum_zz_dr2 < 0.00594
        + 0.08720889 * max(0.0, Q.log_sum_pt - 6.38) / 0.2078586   # +8.7%  log_sum_pt > 6.38
        - 0.06870136 * max(0.0, 0.0616 - Q.z_7) / 0.01418994   # -6.9%  z_7 < 0.0616
        - 0.06725988 * max(0.0, 0.102 - Q.sum_z_dr) / 0.04868705   # -6.7%  sum_z_dr < 0.102
        - 0.05416033 * max(0.0, Q.log_sum_pt - 6.37) * max(0.0, 0.0793 - Q.dr_0) / 0.0103271   # -5.4%  log_sum_pt > 6.37 and dr_0 < 0.0793
        - 0.05225874 * max(0.0, 0.00625 - Q.sum_z_dr2) / 0.002644305   # -5.2%  sum_z_dr2 < 0.00625
        - 0.05045289 * max(0.0, 0.344 - Q.sj3_dr_max) / 0.1826053   # -5.0%  sj3_dr_max < 0.344
        + 0.04137729 * max(0.0, Q.log_sum_pt - 6.57) / 0.08778205   # +4.1%  log_sum_pt > 6.57
        + 0.03842265 * max(0.0, 0.0615 - Q.z_7) * max(0.0, 0.000187 - Q.e3) / 2.383954e-06   # +3.8%  z_7 < 0.0615 and e3 < 0.000187
        + 0.02690947 * max(0.0, 0.061 - Q.z_7) * max(0.0, 0.0409 - Q.zdr_0) / 0.0003932172   # +2.7%  z_7 < 0.061 and zdr_0 < 0.0409
        + 0.02552332 * max(0.0, Q.pt_7 - 37.8) / 3.04633   # +2.6%  pt_7 > 37.8
        - 0.0252146 * max(0.0, 0.052 - Q.z_7) / 0.00868375   # -2.5%  z_7 < 0.052
        - 0.02313411 * max(0.0, Q.log_sum_pt - 6.58) * max(0.0, 0.00784 - Q.sum_z_dr2_top3) / 0.0005137967   # -2.3%  log_sum_pt > 6.58 and sum_z_dr2_top3 < 0.00784
        - 0.02228925 * max(0.0, 0.0204 - Q.zdr_0) / 0.008458814   # -2.2%  zdr_0 < 0.0204
        + 0.01760558 * max(0.0, 0.159 - Q.sj3_dr_max) / 0.04404345   # +1.8%  sj3_dr_max < 0.159
        + 0.01377811 * max(0.0, 0.00342 - Q.lam2) * max(0.0, 0.291 - Q.tau21_b2) / 0.0005092845   # +1.4%  lam2 < 0.00342 and tau21_b2 < 0.291
        - 0.0117784 * max(0.0, Q.pt_7 - 34.7) * max(0.0, 3.08e-05 - Q.e3) / 7.425843e-05   # -1.2%  pt_7 > 34.7 and e3 < 3.08e-05
        + 0.008537194 * max(0.0, Q.log_sum_pt - 6.65) * max(0.0, 0.139 - Q.dr01) / 0.006151295   # +0.9%  log_sum_pt > 6.65 and dr01 < 0.139
        - 0.003653134 * max(0.0, 0.0522 - Q.z_7) * max(0.0, 0.121 - Q.D2_b2) / 0.0001446038   # -0.4%  z_7 < 0.0522 and D2_b2 < 0.121
        + 0.00344567 * max(0.0, 0.00857 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0202) / 8.26299e-06   # +0.3%  lam1 < 0.00857 and centroid_offset > 0.0202
        - 0.002282562 * max(0.0, Q.pt_7 - 53.7) / 0.3186513   # -0.2%  pt_7 > 53.7
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 11.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.40632 * (0.0460271
        + 0.1271963 * max(0.0, 0.0681 - Q.z_7) / 0.01876899   # +12.7%  z_7 < 0.0681
        - 0.111516 * max(0.0, Q.LHA - 0.118) / 0.1297946   # -11.2%  LHA > 0.118
        + 0.0875444 * max(0.0, 0.0128 - Q.lam1_plus_lam2) / 0.007862676   # +8.8%  lam1_plus_lam2 < 0.0128
        + 0.08078897 * max(0.0, 0.00721 - Q.lam1) * max(0.0, 0.256 - Q.max_dr) / 0.000631168   # +8.1%  lam1 < 0.00721 and max_dr < 0.256
        - 0.07808402 * max(0.0, 53.3 - Q.pt_7) / 18.99044   # -7.8%  pt_7 < 53.3
        - 0.07027287 * max(0.0, 0.00727 - Q.sum_zz_dr2) / 0.003594418   # -7.0%  sum_zz_dr2 < 0.00727
        + 0.06969806 * max(0.0, 831.0 - Q.sum_pt) / 141.711   # +7.0%  sum_pt < 831
        + 0.05421446 * max(0.0, 766.0 - Q.sum_pt) / 98.00122   # +5.4%  sum_pt < 766
        - 0.03873866 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 1.15 - Q.D2_b2) / 0.1104664   # -3.9%  log_sum_pt < 6.7 and D2_b2 < 1.15
        + 0.0387239 * max(0.0, Q.pt_7 - 34.4) / 4.539542   # +3.9%  pt_7 > 34.4
        - 0.03634605 * max(0.0, 34.5 - Q.pt_7) / 4.341097   # -3.6%  pt_7 < 34.5
        + 0.03468373 * max(0.0, 6.61 - Q.log_sum_pt) * max(0.0, 1.15 - Q.D2_b2) / 0.07637334   # +3.5%  log_sum_pt < 6.61 and D2_b2 < 1.15
        + 0.02266877 * max(0.0, 610.0 - Q.sum_pt) / 28.35168   # +2.3%  sum_pt < 610
        - 0.02053707 * max(0.0, 0.0404 - Q.C2) * max(0.0, 0.024 - Q.C2_b2) / 0.0004403243   # -2.1%  C2 < 0.0404 and C2_b2 < 0.024
        + 0.01948833 * max(0.0, 0.023 - Q.zdr_0) / 0.01043616   # +1.9%  zdr_0 < 0.023
        - 0.01898202 * max(0.0, Q.z_7 - 0.0679) / 0.002890722   # -1.9%  z_7 > 0.0679
        + 0.01748987 * max(0.0, 0.00173 - Q.lam1_plus_lam2) / 0.0004554682   # +1.7%  lam1_plus_lam2 < 0.00173
        - 0.01732355 * max(0.0, 0.00714 - Q.lam1) * max(0.0, 60.7 - Q.pt_6) / 0.07185382   # -1.7%  lam1 < 0.00714 and pt_6 < 60.7
        - 0.01703965 * max(0.0, 0.00122 - Q.sum_zz_dr2) / 0.0003495679   # -1.7%  sum_zz_dr2 < 0.00122
        + 0.01093028 * max(0.0, Q.log_sum_pt - 6.84) / 0.008148651   # +1.1%  log_sum_pt > 6.84
        - 0.007441741 * max(0.0, Q.sum_pt_top5 - 782.0) * max(0.0, 0.0148 - Q.abseta_0) / 0.1343084   # -0.7%  sum_pt_top5 > 782 and abseta_0 < 0.0148
        - 0.00596569 * max(0.0, 0.00582 - Q.centroid_offset) * max(0.0, 0.0163 - Q.C3) / 3.738824e-06   # -0.6%  centroid_offset < 0.00582 and C3 < 0.0163
        - 0.005412184 * max(0.0, Q.sum_pt - 981.0) / 4.785514   # -0.5%  sum_pt > 981
        - 0.004230556 * max(0.0, 0.0225 - Q.z_7) / 0.00064859   # -0.4%  z_7 < 0.0225
        + 0.002803637 * max(0.0, Q.sum_pt - 976.0) * max(0.0, 0.0147 - Q.abseta_0) / 0.04230052   # +0.3%  sum_pt > 976 and abseta_0 < 0.0147
        - 0.001879223 * max(0.0, Q.sum_pt - 954.0) * max(0.0, Q.n_pt_above_50 - 4.97) / 6.141842   # -0.2%  sum_pt > 954 and n_pt_above_50 > 4.97
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 11.5;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.49816 * (-0.2435173
        + 0.1962051 * max(0.0, Q.sum_z_dr - 0.066) / 0.01418866   # +19.6%  sum_z_dr > 0.066
        - 0.1632325 * max(0.0, Q.sum_z_dr2 - 0.00875) / 0.00220032   # -16.3%  sum_z_dr2 > 0.00875
        + 0.1324765 * max(0.0, Q.sj2_dr - 0.178) / 0.03077243   # +13.2%  sj2_dr > 0.178
        + 0.08253675 * max(0.0, Q.sum_zz_dr2 - 0.00748) / 0.002161778   # +8.3%  sum_zz_dr2 > 0.00748
        - 0.07819608 * max(0.0, Q.tau1 - 0.0801) / 0.01526504   # -7.8%  tau1 > 0.0801
        + 0.07195032 * max(0.0, Q.lam1_plus_lam2 - 0.00584) / 0.002986629   # +7.2%  lam1_plus_lam2 > 0.00584
        - 0.06760203 * max(0.0, Q.sj2_dr - 0.17) * max(0.0, 7.1 - Q.n_dr_0p05_0p1) / 0.1778716   # -6.8%  sj2_dr > 0.17 and n_dr_0p05_0p1 < 7.1
        - 0.03321055 * max(0.0, Q.sj2_dr - 0.268) / 0.008581126   # -3.3%  sj2_dr > 0.268
        - 0.03259558 * max(0.0, Q.sj2_dr - 0.165) * max(0.0, 991.0 - Q.sum_pt) / 12.79144   # -3.3%  sj2_dr > 0.165 and sum_pt < 991
        + 0.03088239 * max(0.0, Q.max_dr - 0.105) / 0.04383834   # +3.1%  max_dr > 0.105
        + 0.02602853 * max(0.0, Q.sj2_dr - 0.18) * max(0.0, 0.0651 - Q.tau2) / 0.001104355   # +2.6%  sj2_dr > 0.18 and tau2 < 0.0651
        - 0.01957276 * max(0.0, Q.lam1_plus_lam2 - 0.0061) * max(0.0, Q.log_sum_pt - 6.15) / 0.000544917   # -2.0%  lam1_plus_lam2 > 0.0061 and log_sum_pt > 6.15
        - 0.01875714 * max(0.0, Q.sj2_dr - 0.179) * max(0.0, 0.00187 - Q.sum_z_dr2_top2) / 7.165203e-06   # -1.9%  sj2_dr > 0.179 and sum_z_dr2_top2 < 0.00187
        + 0.01256225 * max(0.0, Q.sj2_dr - 0.157) * max(0.0, Q.n_dr_0p1_0p2 - 0.955) / 0.08160609   # +1.3%  sj2_dr > 0.157 and n_dr_0p1_0p2 > 0.955
        + 0.01224352 * max(0.0, Q.sj2_dr - 0.189) * max(0.0, Q.eccentricity - 0.941) / 0.0007256594   # +1.2%  sj2_dr > 0.189 and eccentricity > 0.941
        + 0.01121144 * max(0.0, Q.lam1_plus_lam2 - 0.02) / 0.0006510652   # +1.1%  lam1_plus_lam2 > 0.02
        + 0.006143518 * max(0.0, Q.sj2_dr - 0.269) * max(0.0, 0.00767 - Q.sum_z_dr2_top2) / 1.545714e-05   # +0.6%  sj2_dr > 0.269 and sum_z_dr2_top2 < 0.00767
        - 0.004593133 * max(0.0, Q.sj2_dr - 0.302) * max(0.0, Q.z_dr_0p05_0p1 - 0.116) / 0.0009152959   # -0.5%  sj2_dr > 0.302 and z_dr_0p05_0p1 > 0.116
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 25.61;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.61075 * (-0.02983122
        - 0.1945596 * max(0.0, 0.00873 - Q.sum_z_dr2) / 0.004489025   # -19.5%  sum_z_dr2 < 0.00873
        + 0.1301614 * max(0.0, 0.000205 - Q.e3) / 0.0001658473   # +13.0%  e3 < 0.000205
        + 0.1131742 * max(0.0, 0.21 - Q.sj3_dr_min) / 0.1675419   # +11.3%  sj3_dr_min < 0.21
        + 0.09458755 * max(0.0, 0.0111 - Q.sum_zz_dr2) / 0.00672905   # +9.5%  sum_zz_dr2 < 0.0111
        - 0.07523807 * max(0.0, 0.00117 - Q.lam2) / 0.0009492135   # -7.5%  lam2 < 0.00117
        - 0.07036088 * max(0.0, Q.sum_zz_dr2 - 0.00334) / 0.003761994   # -7.0%  sum_zz_dr2 > 0.00334
        + 0.06022764 * max(0.0, Q.e2 - 0.0212) / 0.0132972   # +6.0%  e2 > 0.0212
        - 0.05857297 * max(0.0, 0.312 - Q.sj3_dr_max) * max(0.0, 6.74 - Q.log_sum_pt) / 0.02688347   # -5.9%  sj3_dr_max < 0.312 and log_sum_pt < 6.74
        + 0.03905722 * max(0.0, 0.238 - Q.N2) / 0.05884028   # +3.9%  N2 < 0.238
        + 0.03320205 * max(0.0, 0.0026 - Q.sum_z_dr2) / 0.0007801188   # +3.3%  sum_z_dr2 < 0.0026
        - 0.03114512 * max(0.0, 0.228 - Q.N2) * max(0.0, 0.452 - Q.planar_flow) / 0.02099078   # -3.1%  N2 < 0.228 and planar_flow < 0.452
        + 0.0234411 * max(0.0, Q.max_dr - 0.0975) / 0.04802752   # +2.3%  max_dr > 0.0975
        - 0.02180614 * max(0.0, 0.0134 - Q.tau2) / 0.005171034   # -2.2%  tau2 < 0.0134
        - 0.01642736 * max(0.0, 0.000974 - Q.lam2) * max(0.0, 0.217 - Q.dr_max_012) / 0.0001175187   # -1.6%  lam2 < 0.000974 and dr_max_012 < 0.217
        - 0.01271617 * max(0.0, Q.sj2_dr - 0.219) / 0.01809281   # -1.3%  sj2_dr > 0.219
        + 0.01174882 * max(0.0, 0.767 - Q.D2) * max(0.0, Q.pt_6 - 13.8) / 3.073505   # +1.2%  D2 < 0.767 and pt_6 > 13.8
        + 0.009320987 * max(0.0, 0.00575 - Q.sum_z_dr2_top2) * max(0.0, Q.centroid_offset - 0.0132) / 1.126026e-05   # +0.9%  sum_z_dr2_top2 < 0.00575 and centroid_offset > 0.0132
        - 0.004252751 * max(0.0, Q.C2 - 0.0724) / 0.002378082   # -0.4%  C2 > 0.0724
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 9.014;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.014222 * (0.1032812
        + 0.1366336 * max(0.0, 0.211 - Q.LHA) / 0.03026157   # +13.7%  LHA < 0.211
        - 0.1223236 * max(0.0, 0.288 - Q.sj3_dr_max) / 0.1339796   # -12.2%  sj3_dr_max < 0.288
        - 0.08786727 * max(0.0, 0.217 - Q.LHA) * max(0.0, 6.81 - Q.log_sum_pt) / 0.004258361   # -8.8%  LHA < 0.217 and log_sum_pt < 6.81
        - 0.08773811 * max(0.0, 712.0 - Q.sum_pt) / 68.18024   # -8.8%  sum_pt < 712
        + 0.08478676 * max(0.0, 0.164 - Q.max_dr) / 0.06065767   # +8.5%  max_dr < 0.164
        + 0.07735722 * max(0.0, 0.00282 - Q.sum_z_dr2) * max(0.0, 1.96 - Q.n_dr_0p2_0p4) / 0.001688414   # +7.7%  sum_z_dr2 < 0.00282 and n_dr_0p2_0p4 < 1.96
        + 0.06195421 * max(0.0, 0.00244 - Q.sum_z_dr2) * max(0.0, 0.0186 - Q.centroid_offset) / 7.955398e-06   # +6.2%  sum_z_dr2 < 0.00244 and centroid_offset < 0.0186
        - 0.06194364 * max(0.0, 427.0 - Q.sum_pt_top5) / 13.32634   # -6.2%  sum_pt_top5 < 427
        - 0.04092861 * max(0.0, Q.pt_7 - 36.1) / 3.73799   # -4.1%  pt_7 > 36.1
        - 0.03478608 * max(0.0, 0.212 - Q.LHA) * max(0.0, 0.098 - Q.z_6) / 0.001522182   # -3.5%  LHA < 0.212 and z_6 < 0.098
        - 0.03388108 * max(0.0, 0.0284 - Q.centroid_offset) * max(0.0, 0.0962 - Q.z_5) / 0.0004407093   # -3.4%  centroid_offset < 0.0284 and z_5 < 0.0962
        + 0.02719719 * max(0.0, 0.054 - Q.z_6) / 0.006625986   # +2.7%  z_6 < 0.054
        + 0.02334695 * max(0.0, 710.0 - Q.sum_pt) * max(0.0, 0.035 - Q.dr_0) / 0.2328037   # +2.3%  sum_pt < 710 and dr_0 < 0.035
        + 0.02302588 * max(0.0, 0.0321 - Q.z_7) / 0.001995773   # +2.3%  z_7 < 0.0321
        - 0.0209753 * max(0.0, 0.174 - Q.LHA) * max(0.0, 0.00587 - Q.zdr_0) / 6.519862e-05   # -2.1%  LHA < 0.174 and zdr_0 < 0.00587
        + 0.01794348 * max(0.0, 0.037 - Q.e2) * max(0.0, Q.zdr_0 - 0.00385) / 5.285834e-05   # +1.8%  e2 < 0.037 and zdr_0 > 0.00385
        - 0.01587979 * max(0.0, 0.0261 - Q.centroid_offset) * max(0.0, Q.pt_5 - 34.6) / 0.1942251   # -1.6%  centroid_offset < 0.0261 and pt_5 > 34.6
        - 0.01111063 * max(0.0, 0.302 - Q.sj3_dr_max) * max(0.0, Q.pt1_dr01 - 13.7) / 0.1447308   # -1.1%  sj3_dr_max < 0.302 and pt1_dr01 > 13.7
        + 0.01072375 * max(0.0, 0.0421 - Q.e2) * max(0.0, 0.326 - Q.planar_flow) / 0.002018085   # +1.1%  e2 < 0.0421 and planar_flow < 0.326
        - 0.009394055 * max(0.0, Q.log_sum_pt - 6.9) / 0.003849095   # -0.9%  log_sum_pt > 6.9
        + 0.007150202 * max(0.0, 0.00762 - Q.sum_z_dr) / 0.0002261526   # +0.7%  sum_z_dr < 0.00762
        + 0.003052595 * max(0.0, 22.8 - Q.pt_5) * max(0.0, 0.181 - Q.abseta_6) / 0.02697723   # +0.3%  pt_5 < 22.8 and abseta_6 < 0.181
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 27.58;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.57878 * (0.05257665
        - 0.2399287 * max(0.0, 0.00867 - Q.sum_z_dr2) / 0.0044409   # -24.0%  sum_z_dr2 < 0.00867
        + 0.08527012 * max(0.0, 0.104 - Q.tau1) / 0.04930076   # +8.5%  tau1 < 0.104
        - 0.07408288 * max(0.0, Q.centroid_offset - 0.0162) * max(0.0, 993.0 - Q.sum_pt) / 2.643099   # -7.4%  centroid_offset > 0.0162 and sum_pt < 993
        + 0.0665299 * max(0.0, 0.0121 - Q.lam1) / 0.007398442   # +6.7%  lam1 < 0.0121
        - 0.04806563 * max(0.0, Q.sj3_dr_max - 0.0735) * max(0.0, Q.eccentricity - 0.9) / 0.006342544   # -4.8%  sj3_dr_max > 0.0735 and eccentricity > 0.9
        - 0.04591645 * max(0.0, 0.16 - Q.max_dr) / 0.05782283   # -4.6%  max_dr < 0.16
        - 0.03869555 * max(0.0, Q.sj3_dr_max - 0.072) / 0.1088955   # -3.9%  sj3_dr_max > 0.072
        + 0.03409136 * max(0.0, Q.sj3_dr_max - 0.18) / 0.04273628   # +3.4%  sj3_dr_max > 0.18
        + 0.03405571 * max(0.0, Q.centroid_offset - 0.0165) / 0.006219966   # +3.4%  centroid_offset > 0.0165
        + 0.03119444 * max(0.0, 0.0268 - Q.sum_z_dr) / 0.004301523   # +3.1%  sum_z_dr < 0.0268
        + 0.03114885 * max(0.0, Q.centroid_offset - 0.00732) / 0.01104174   # +3.1%  centroid_offset > 0.00732
        + 0.02629508 * max(0.0, 0.0387 - Q.C3) / 0.01507664   # +2.6%  C3 < 0.0387
        + 0.02329423 * max(0.0, Q.sj3_dr_max - 0.178) * max(0.0, Q.eccentricity - 0.901) / 0.002134307   # +2.3%  sj3_dr_max > 0.178 and eccentricity > 0.901
        + 0.02286463 * max(0.0, 0.00998 - Q.sum_z_dr2_top2) / 0.006494117   # +2.3%  sum_z_dr2_top2 < 0.00998
        - 0.0218884 * max(0.0, 1.67 - Q.D3) / 0.6042598   # -2.2%  D3 < 1.67
        + 0.02084256 * max(0.0, Q.centroid_offset - 0.00811) * max(0.0, 714.0 - Q.sum_pt) / 1.368601   # +2.1%  centroid_offset > 0.00811 and sum_pt < 714
        - 0.01944835 * max(0.0, 0.0135 - Q.lam1_plus_lam2) * max(0.0, 0.335 - Q.planar_flow) / 0.001191915   # -1.9%  lam1_plus_lam2 < 0.0135 and planar_flow < 0.335
        - 0.01711232 * max(0.0, Q.sj3_dr_min - 0.0185) / 0.03044754   # -1.7%  sj3_dr_min > 0.0185
        + 0.01583531 * max(0.0, Q.centroid_offset - 0.00856) * max(0.0, 0.0148 - Q.mean_phi2) / 9.661913e-05   # +1.6%  centroid_offset > 0.00856 and mean_phi2 < 0.0148
        + 0.01226443 * max(0.0, 2.95 - Q.n_dr_0_0p05) / 1.009666   # +1.2%  n_dr_0_0p05 < 2.95
        + 0.0120742 * max(0.0, Q.LHA - 0.267) * max(0.0, Q.eccentricity - 0.892) / 0.001849953   # +1.2%  LHA > 0.267 and eccentricity > 0.892
        + 0.01172725 * max(0.0, 6.33 - Q.log_sum_pt) / 0.03368993   # +1.2%  log_sum_pt < 6.33
        + 0.00836294 * max(0.0, 29.8 - Q.pt_6) / 1.364732   # +0.8%  pt_6 < 29.8
        + 0.008161679 * max(0.0, 6.65 - Q.log_sum_pt) / 0.1596377   # +0.8%  log_sum_pt < 6.65
        + 0.007946761 * max(0.0, Q.centroid_offset - 0.00932) * max(0.0, 0.174 - Q.sj2_dr) / 0.0003035485   # +0.8%  centroid_offset > 0.00932 and sj2_dr < 0.174
        + 0.007391865 * max(0.0, Q.centroid_offset - 0.00848) * max(0.0, 6.16 - Q.n_dr_0p05_0p1) / 0.03883022   # +0.7%  centroid_offset > 0.00848 and n_dr_0p05_0p1 < 6.16
        - 0.006691352 * max(0.0, Q.sum_zz_dr2 - 0.0116) / 0.001441714   # -0.7%  sum_zz_dr2 > 0.0116
        + 0.005303372 * max(0.0, 6.64 - Q.log_sum_pt) * max(0.0, 0.0586 - Q.z_7) / 0.000356733   # +0.5%  log_sum_pt < 6.64 and z_7 < 0.0586
        - 0.005303226 * max(0.0, 0.0346 - Q.z_6) / 0.00154279   # -0.5%  z_6 < 0.0346
        - 0.005222943 * max(0.0, 0.0216 - Q.z_6) / 0.0003104362   # -0.5%  z_6 < 0.0216
        - 0.005038158 * max(0.0, 6.33 - Q.log_sum_pt) * max(0.0, Q.pt_7 - 25.5) / 0.2606872   # -0.5%  log_sum_pt < 6.33 and pt_7 > 25.5
        + 0.002817387 * max(0.0, 0.0117 - Q.lam1) * max(0.0, 19.6 - Q.pt_6) / 0.002688585   # +0.3%  lam1 < 0.0117 and pt_6 < 19.6
        + 0.001923459 * max(0.0, Q.sum_pt - 991.0) / 4.277955   # +0.2%  sum_pt > 991
        + 0.0009368676 * max(0.0, Q.sum_pt - 985.0) * max(0.0, Q.dr01 - 0.141) / 0.01279092   # +0.1%  sum_pt > 985 and dr01 > 0.141
        - 0.0008301917 * max(0.0, Q.eccentricity - 0.905) * max(0.0, Q.mean_eta - 0.0178) / 5.767173e-05   # -0.1%  eccentricity > 0.905 and mean_eta > 0.0178
        + 0.0007377751 * max(0.0, 0.0216 - Q.z_6) * max(0.0, 6.84 - Q.log_sum_pt) / 4.265606e-06   # +0.1%  z_6 < 0.0216 and log_sum_pt < 6.84
        - 0.0007057033 * max(0.0, Q.sum_pt - 985.0) * max(0.0, Q.dr01 - 0.166) / 0.008846563   # -0.1%  sum_pt > 985 and dr01 > 0.166
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 29.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.64339 * (0.2621158
        - 0.2462854 * max(0.0, 0.0082 - Q.sum_zz_dr2) / 0.004319962   # -24.6%  sum_zz_dr2 < 0.0082
        - 0.08215506 * max(0.0, 0.0876 - Q.sum_z_dr) / 0.03667703   # -8.2%  sum_z_dr < 0.0876
        - 0.08040025 * max(0.0, Q.sum_z_dr2 - 0.00745) / 0.002487825   # -8.0%  sum_z_dr2 > 0.00745
        + 0.07879745 * max(0.0, 0.00714 - Q.sum_zz_dr2) / 0.003496742   # +7.9%  sum_zz_dr2 < 0.00714
        - 0.06003688 * max(0.0, 0.00562 - Q.lam1_plus_lam2) / 0.002247092   # -6.0%  lam1_plus_lam2 < 0.00562
        + 0.05792458 * max(0.0, 0.095 - Q.tau1) / 0.04229264   # +5.8%  tau1 < 0.095
        - 0.05780163 * max(0.0, Q.sum_z_dr2 - 0.00442) / 0.003614845   # -5.8%  sum_z_dr2 > 0.00442
        + 0.04042238 * max(0.0, 0.159 - Q.sj2_dr) / 0.04831679   # +4.0%  sj2_dr < 0.159
        - 0.03805571 * max(0.0, 0.186 - Q.sj2_dr) / 0.06373448   # -3.8%  sj2_dr < 0.186
        + 0.03556161 * max(0.0, 0.294 - Q.LHA) / 0.07220321   # +3.6%  LHA < 0.294
        + 0.03033557 * max(0.0, 0.00535 - Q.sum_zz_dr2) / 0.00227658   # +3.0%  sum_zz_dr2 < 0.00535
        + 0.02361241 * max(0.0, 0.204 - Q.max_dr) / 0.09137755   # +2.4%  max_dr < 0.204
        - 0.020204 * max(0.0, 0.213 - Q.LHA) / 0.03103187   # -2.0%  LHA < 0.213
        - 0.01858219 * max(0.0, 0.0208 - Q.centroid_offset) * max(0.0, 4.67 - Q.D2_b2) / 0.02782017   # -1.9%  centroid_offset < 0.0208 and D2_b2 < 4.67
        + 0.0174424 * max(0.0, 0.0215 - Q.centroid_offset) / 0.008853629   # +1.7%  centroid_offset < 0.0215
        + 0.01689408 * max(0.0, Q.sum_z_dr2 - 0.00448) * max(0.0, 0.205 - Q.planar_flow) / 0.0002766839   # +1.7%  sum_z_dr2 > 0.00448 and planar_flow < 0.205
        + 0.01627589 * max(0.0, 0.196 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.42) / 0.01419037   # +1.6%  planar_flow < 0.196 and log_sum_pt > 6.42
        + 0.01470637 * max(0.0, 0.142 - Q.sj3_dr_max) / 0.03694464   # +1.5%  sj3_dr_max < 0.142
        + 0.01467397 * max(0.0, Q.sum_z_dr2 - 0.0136) * max(0.0, Q.eccentricity - 0.934) / 2.718664e-05   # +1.5%  sum_z_dr2 > 0.0136 and eccentricity > 0.934
        + 0.01277584 * max(0.0, 0.0509 - Q.tau1) / 0.01564956   # +1.3%  tau1 < 0.0509
        - 0.01134166 * max(0.0, 0.000461 - Q.lam2) / 0.0003264128   # -1.1%  lam2 < 0.000461
        - 0.0054272 * max(0.0, 0.0119 - Q.sum_zz_dr2) * max(0.0, 48.7 - Q.pt_7) / 0.1132962   # -0.5%  sum_zz_dr2 < 0.0119 and pt_7 < 48.7
        - 0.004724756 * max(0.0, 0.214 - Q.sj3_dr_max) * max(0.0, Q.eccentricity - 0.945) / 0.0008437217   # -0.5%  sj3_dr_max < 0.214 and eccentricity > 0.945
        - 0.003371444 * max(0.0, 0.198 - Q.planar_flow) * max(0.0, 0.0524 - Q.z_7) / 0.000644781   # -0.3%  planar_flow < 0.198 and z_7 < 0.0524
        + 0.002564758 * max(0.0, 0.00569 - Q.lam1_plus_lam2) * max(0.0, -0.00458 - Q.mean_phi) / 4.446089e-06   # +0.3%  lam1_plus_lam2 < 0.00569 and mean_phi < -0.00458
        + 0.002221374 * max(0.0, 0.00568 - Q.lam1_plus_lam2) * max(0.0, -0.00679 - Q.mean_eta) / 3.484077e-06   # +0.2%  lam1_plus_lam2 < 0.00568 and mean_eta < -0.00679
        + 0.001766393 * max(0.0, 0.00569 - Q.lam1_plus_lam2) * max(0.0, Q.mean_phi - 0.00912) / 2.605068e-06   # +0.2%  lam1_plus_lam2 < 0.00569 and mean_phi > 0.00912
        + 0.001629819 * max(0.0, 0.00569 - Q.lam1_plus_lam2) * max(0.0, Q.mean_eta - 0.00949) / 2.556262e-06   # +0.2%  lam1_plus_lam2 < 0.00569 and mean_eta > 0.00949
        - 0.001626531 * max(0.0, 0.207 - Q.planar_flow) * max(0.0, 33.6 - Q.pt_6) / 0.1456673   # -0.2%  planar_flow < 0.207 and pt_6 < 33.6
        + 0.001349953 * max(0.0, 0.00738 - Q.sum_zz_dr2) * max(0.0, Q.sj3_dr13 - 0.201) / 1.594311e-05   # +0.1%  sum_zz_dr2 < 0.00738 and sj3_dr13 > 0.201
        - 0.001032404 * max(0.0, 0.2 - Q.max_dr) * max(0.0, Q.n_dr_0p1_0p2 - 4.12) / 0.004175166   # -0.1%  max_dr < 0.2 and n_dr_0p1_0p2 > 4.12
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 4.813;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.813245 * (-0.1518726
        + 0.3097298 * max(0.0, 0.00539 - Q.sum_z_dr2) / 0.002111623   # +31.0%  sum_z_dr2 < 0.00539
        - 0.1633137 * max(0.0, 0.201 - Q.LHA) * max(0.0, 0.198 - Q.z_dr_0p2_0p4) / 0.005240459   # -16.3%  LHA < 0.201 and z_dr_0p2_0p4 < 0.198
        + 0.1412544 * max(0.0, 0.223 - Q.sj3_dr_max) * max(0.0, 0.000253 - Q.lam2) / 1.654239e-05   # +14.1%  sj3_dr_max < 0.223 and lam2 < 0.000253
        - 0.1399745 * max(0.0, 0.129 - Q.sj3_dr_max) / 0.0319304   # -14.0%  sj3_dr_max < 0.129
        + 0.093224 * max(0.0, 0.00632 - Q.sum_z_dr2) * max(0.0, 0.0222 - Q.centroid_offset) / 3.275255e-05   # +9.3%  sum_z_dr2 < 0.00632 and centroid_offset < 0.0222
        - 0.05992907 * max(0.0, 0.00528 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.00735) / 1.113719e-05   # -6.0%  sum_z_dr2 < 0.00528 and centroid_offset > 0.00735
        - 0.0461987 * max(0.0, 0.00543 - Q.sum_z_dr2) * max(0.0, 50.2 - Q.pt_7) / 0.03730967   # -4.6%  sum_z_dr2 < 0.00543 and pt_7 < 50.2
        - 0.01762787 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, Q.pt_7 - 19.2) / 0.4636463   # -1.8%  log_sum_pt > 6.68 and pt_7 > 19.2
        - 0.01562769 * max(0.0, 0.00427 - Q.sum_z_dr2) * max(0.0, 6.52 - Q.log_sum_pt) / 6.776569e-05   # -1.6%  sum_z_dr2 < 0.00427 and log_sum_pt < 6.52
        + 0.009915621 * max(0.0, 0.194 - Q.sj3_dr_max) * max(0.0, Q.sum_z_dr2_top3 - 0.00121) / 2.927995e-05   # +1.0%  sj3_dr_max < 0.194 and sum_z_dr2_top3 > 0.00121
        - 0.003204639 * max(0.0, 0.188 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0171) / 2.337078e-06   # -0.3%  LHA < 0.188 and centroid_offset > 0.0171
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 21.04;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.03849 * (-0.2295792
        + 0.2041802 * max(0.0, 0.00652 - Q.lam1_plus_lam2) / 0.002826081   # +20.4%  lam1_plus_lam2 < 0.00652
        + 0.1090322 * max(0.0, 0.0036 - Q.sum_z_dr2) / 0.001200981   # +10.9%  sum_z_dr2 < 0.0036
        + 0.1085918 * max(0.0, 6.9 - Q.log_sum_pt) / 0.3609173   # +10.9%  log_sum_pt < 6.9
        - 0.05890709 * max(0.0, 0.0541 - Q.sum_z_dr) / 0.01505852   # -5.9%  sum_z_dr < 0.0541
        + 0.05313972 * max(0.0, 0.198 - Q.sj3_dr_max) / 0.06537891   # +5.3%  sj3_dr_max < 0.198
        - 0.05233505 * max(0.0, 0.00293 - Q.sum_zz_dr2) / 0.001029019   # -5.2%  sum_zz_dr2 < 0.00293
        - 0.05162788 * max(0.0, 0.142 - Q.sj3_dr_max) / 0.03694464   # -5.2%  sj3_dr_max < 0.142
        + 0.04827255 * max(0.0, 0.0242 - Q.centroid_offset) / 0.01085023   # +4.8%  centroid_offset < 0.0242
        - 0.03868324 * max(0.0, 0.126 - Q.sj2_dr) / 0.03405175   # -3.9%  sj2_dr < 0.126
        + 0.03023505 * max(0.0, 0.000178 - Q.lam2) / 9.954614e-05   # +3.0%  lam2 < 0.000178
        + 0.02858861 * max(0.0, 0.0186 - Q.centroid_offset) * max(0.0, 0.0139 - Q.tau3) / 6.502282e-05   # +2.9%  centroid_offset < 0.0186 and tau3 < 0.0139
        - 0.02545894 * max(0.0, 0.0662 - Q.dr_0) / 0.02664764   # -2.5%  dr_0 < 0.0662
        + 0.02119202 * max(0.0, 0.114 - Q.max_dr) / 0.02952636   # +2.1%  max_dr < 0.114
        - 0.01751592 * max(0.0, 0.005 - Q.lam1) * max(0.0, 0.18 - Q.dr1_7) / 0.0002813042   # -1.8%  lam1 < 0.005 and dr1_7 < 0.18
        - 0.01518625 * max(0.0, 0.0188 - Q.centroid_offset) * max(0.0, Q.pt1_over_pt0 - 0.151) / 0.003290378   # -1.5%  centroid_offset < 0.0188 and pt1_over_pt0 > 0.151
        + 0.0150297 * max(0.0, 0.0183 - Q.centroid_offset) * max(0.0, 5.08 - Q.n_dr_0p05_0p1) / 0.02377459   # +1.5%  centroid_offset < 0.0183 and n_dr_0p05_0p1 < 5.08
        + 0.01320555 * max(0.0, 0.0224 - Q.centroid_offset) * max(0.0, 0.0834 - Q.dr_7) / 0.0003709276   # +1.3%  centroid_offset < 0.0224 and dr_7 < 0.0834
        - 0.01172649 * max(0.0, 6.95 - Q.log_sum_pt) * max(0.0, 0.789 - Q.D2_b2) / 0.1393828   # -1.2%  log_sum_pt < 6.95 and D2_b2 < 0.789
        - 0.01133469 * max(0.0, 0.00729 - Q.lam1_plus_lam2) * max(0.0, 0.826 - Q.D3) / 0.0002947647   # -1.1%  lam1_plus_lam2 < 0.00729 and D3 < 0.826
        + 0.01028758 * max(0.0, 0.0193 - Q.centroid_offset) * max(0.0, Q.pt_1 - 93.5) / 0.4068329   # +1.0%  centroid_offset < 0.0193 and pt_1 > 93.5
        + 0.008462634 * max(0.0, 0.0223 - Q.centroid_offset) * max(0.0, Q.tau21_b2 - 0.00316) / 0.002034755   # +0.8%  centroid_offset < 0.0223 and tau21_b2 > 0.00316
        - 0.007830031 * max(0.0, 0.0705 - Q.sum_z_dr) * max(0.0, 0.00149 - Q.mean_phi) / 9.747456e-05   # -0.8%  sum_z_dr < 0.0705 and mean_phi < 0.00149
        - 0.007223081 * max(0.0, 6.84 - Q.log_sum_pt) * max(0.0, Q.planar_flow - 0.0775) / 0.06907395   # -0.7%  log_sum_pt < 6.84 and planar_flow > 0.0775
        + 0.006648971 * max(0.0, Q.lam2 - 0.0009) / 0.0003362603   # +0.7%  lam2 > 0.0009
        + 0.006024465 * max(0.0, Q.e3 - 0.000257) / 3.129522e-05   # +0.6%  e3 > 0.000257
        + 0.005684328 * max(0.0, 0.0042 - Q.sum_z_dr2) * max(0.0, 0.802 - Q.D3) / 9.270517e-05   # +0.6%  sum_z_dr2 < 0.0042 and D3 < 0.802
        + 0.005037903 * max(0.0, 6.9 - Q.log_sum_pt) * max(0.0, 0.0572 - Q.z_5) / 0.0005888326   # +0.5%  log_sum_pt < 6.9 and z_5 < 0.0572
        - 0.00498607 * max(0.0, 0.00641 - Q.lam1_plus_lam2) * max(0.0, -0.00924 - Q.mean_eta) / 3.319601e-06   # -0.5%  lam1_plus_lam2 < 0.00641 and mean_eta < -0.00924
        - 0.004869047 * max(0.0, 0.0716 - Q.sum_z_dr) * max(0.0, Q.mean_phi - 0.00423) / 4.415404e-05   # -0.5%  sum_z_dr < 0.0716 and mean_phi > 0.00423
        - 0.004810355 * max(0.0, 0.0163 - Q.centroid_offset) * max(0.0, 0.0457 - Q.M2) / 4.729093e-05   # -0.5%  centroid_offset < 0.0163 and M2 < 0.0457
        - 0.004375823 * max(0.0, 0.0183 - Q.centroid_offset) * max(0.0, 0.0768 - Q.mratio_min_012) / 0.0001022897   # -0.4%  centroid_offset < 0.0183 and mratio_min_012 < 0.0768
        - 0.003920895 * max(0.0, 0.216 - Q.LHA) * max(0.0, 0.00318 - Q.mean_eta) / 0.0001334785   # -0.4%  LHA < 0.216 and mean_eta < 0.00318
        - 0.003814625 * max(0.0, 0.00558 - Q.lam1_plus_lam2) * max(0.0, Q.mean_eta - 0.0093) / 2.515798e-06   # -0.4%  lam1_plus_lam2 < 0.00558 and mean_eta > 0.0093
        + 0.001781269 * max(0.0, 0.13 - Q.sj2_dr) * max(0.0, -0.00798 - Q.mean_eta) / 5.300595e-05   # +0.2%  sj2_dr < 0.13 and mean_eta < -0.00798
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 20.43;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.43364 * (0.8270676
        - 0.5168993 * max(0.0, 0.0162 - Q.lam1) / 0.01099077   # -51.7%  lam1 < 0.0162
        - 0.08065076 * max(0.0, 0.122 - Q.sum_z_dr) / 0.06618427   # -8.1%  sum_z_dr < 0.122
        - 0.0733758 * max(0.0, Q.sum_z_dr2 - 0.00285) / 0.004475626   # -7.3%  sum_z_dr2 > 0.00285
        - 0.07055155 * max(0.0, Q.LHA - 0.193) / 0.07355229   # -7.1%  LHA > 0.193
        - 0.0653338 * max(0.0, Q.lam1 - 0.00482) / 0.002940545   # -6.5%  lam1 > 0.00482
        + 0.03495244 * max(0.0, Q.lam1 - 0.0162) / 0.0007071343   # +3.5%  lam1 > 0.0162
        + 0.02399721 * max(0.0, Q.tau1 - 0.0492) / 0.02901481   # +2.4%  tau1 > 0.0492
        - 0.01792107 * max(0.0, Q.sj2_dr - 0.17) * max(0.0, 0.676 - Q.planar_flow) / 0.01571642   # -1.8%  sj2_dr > 0.17 and planar_flow < 0.676
        + 0.01742927 * max(0.0, 0.00296 - Q.sum_z_dr2_top3) / 0.001203187   # +1.7%  sum_z_dr2_top3 < 0.00296
        + 0.01636872 * max(0.0, Q.lam2 - 0.00024) / 0.0004355112   # +1.6%  lam2 > 0.00024
        - 0.0150343 * max(0.0, 0.0679 - Q.z_7) / 0.01861851   # -1.5%  z_7 < 0.0679
        + 0.01083602 * max(0.0, 4.74 - Q.n_dr_0p05_0p1) / 2.9562   # +1.1%  n_dr_0p05_0p1 < 4.74
        - 0.01063369 * max(0.0, Q.LHA - 0.311) * max(0.0, 0.838 - Q.planar_flow) / 0.007988422   # -1.1%  LHA > 0.311 and planar_flow < 0.838
        + 0.006964569 * max(0.0, Q.e3 - 8.06e-05) / 5.010968e-05   # +0.7%  e3 > 8.06e-05
        + 0.006876013 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.7%  n_dr_0p2_0p4 > 1
        - 0.006320523 * max(0.0, Q.log_sum_pt - 6.67) / 0.04531624   # -0.6%  log_sum_pt > 6.67
        - 0.005962144 * max(0.0, Q.n_dr_0p2_0p4 - 1.02) * max(0.0, 4.62 - Q.D2_b2) / 0.5640199   # -0.6%  n_dr_0p2_0p4 > 1.02 and D2_b2 < 4.62
        - 0.003870105 * max(0.0, 49.1 - Q.pt_6) * max(0.0, 6.48 - Q.log_sum_pt) / 0.8796477   # -0.4%  pt_6 < 49.1 and log_sum_pt < 6.48
        + 0.003599162 * max(0.0, Q.sj2_dr - 0.299) / 0.004714358   # +0.4%  sj2_dr > 0.299
        + 0.003194937 * max(0.0, Q.centroid_offset - 0.023) / 0.004131911   # +0.3%  centroid_offset > 0.023
        - 0.002819477 * max(0.0, Q.lam2 - 0.000588) * max(0.0, 6.54 - Q.log_sum_pt) / 0.0001060998   # -0.3%  lam2 > 0.000588 and log_sum_pt < 6.54
        + 0.00264604 * max(0.0, 48.7 - Q.pt_6) * max(0.0, 0.127 - Q.D2_b2) / 0.1826629   # +0.3%  pt_6 < 48.7 and D2_b2 < 0.127
        - 0.001984411 * max(0.0, Q.sum_z_dr2 - 0.0247) / 0.0003167871   # -0.2%  sum_z_dr2 > 0.0247
        + 0.001778666 * max(0.0, Q.sum_pt - 989.0) / 4.3736   # +0.2%  sum_pt > 989
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 26.89;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.89454 * (-0.07622364
        + 0.2384714 * max(0.0, 0.00861 - Q.lam1_plus_lam2) / 0.004392863   # +23.8%  lam1_plus_lam2 < 0.00861
        - 0.1513379 * max(0.0, 0.00843 - Q.lam1) / 0.004343825   # -15.1%  lam1 < 0.00843
        + 0.1257391 * max(0.0, 0.0133 - Q.lam1_plus_lam2) / 0.008288473   # +12.6%  lam1_plus_lam2 < 0.0133
        + 0.07174947 * max(0.0, 0.0391 - Q.centroid_offset) / 0.02347529   # +7.2%  centroid_offset < 0.0391
        - 0.07115216 * max(0.0, 0.071 - Q.sum_z_dr) / 0.02456489   # -7.1%  sum_z_dr < 0.071
        - 0.05286429 * max(0.0, 0.00642 - Q.sum_zz_dr2) / 0.002974395   # -5.3%  sum_zz_dr2 < 0.00642
        - 0.02938437 * max(0.0, Q.sj2_dr - 0.175) / 0.03199511   # -2.9%  sj2_dr > 0.175
        + 0.02849733 * max(0.0, 0.264 - Q.planar_flow) / 0.115949   # +2.8%  planar_flow < 0.264
        + 0.02538368 * max(0.0, Q.sj2_dr - 0.0899) / 0.0809825   # +2.5%  sj2_dr > 0.0899
        - 0.02407388 * max(0.0, 0.039 - Q.centroid_offset) * max(0.0, 53.6 - Q.pt_7) / 0.4591886   # -2.4%  centroid_offset < 0.039 and pt_7 < 53.6
        - 0.01869754 * max(0.0, 0.0142 - Q.centroid_offset) * max(0.0, 0.09 - Q.sj3_dr_min) / 0.0002906716   # -1.9%  centroid_offset < 0.0142 and sj3_dr_min < 0.09
        - 0.01849471 * max(0.0, 0.037 - Q.centroid_offset) * max(0.0, 120.0 - Q.pt_2) / 0.5872571   # -1.8%  centroid_offset < 0.037 and pt_2 < 120
        - 0.01837847 * max(0.0, Q.centroid_offset - 0.0502) * max(0.0, Q.M3 - 0.0444) / 2.167898e-05   # -1.8%  centroid_offset > 0.0502 and M3 > 0.0444
        + 0.01444878 * max(0.0, 0.0384 - Q.centroid_offset) * max(0.0, 0.156 - Q.z_3rd) / 0.0006653997   # +1.4%  centroid_offset < 0.0384 and z_3rd < 0.156
        + 0.01136383 * max(0.0, Q.log_sum_pt - 6.57) * max(0.0, 0.0594 - Q.z_7) / 0.002387696   # +1.1%  log_sum_pt > 6.57 and z_7 < 0.0594
        + 0.0113413 * max(0.0, Q.LHA - 0.252) / 0.03900501   # +1.1%  LHA > 0.252
        - 0.01034111 * max(0.0, 0.249 - Q.planar_flow) * max(0.0, 862.0 - Q.sum_pt) / 16.07627   # -1.0%  planar_flow < 0.249 and sum_pt < 862
        - 0.008895618 * max(0.0, Q.centroid_offset - 0.0483) * max(0.0, 3.12 - Q.D2) / 0.001746303   # -0.9%  centroid_offset > 0.0483 and D2 < 3.12
        - 0.00762768 * max(0.0, 0.00162 - Q.mean_phi2) / 0.0006838099   # -0.8%  mean_phi2 < 0.00162
        - 0.007622007 * max(0.0, 0.0345 - Q.sum_z_dr) / 0.006810312   # -0.8%  sum_z_dr < 0.0345
        + 0.007513993 * max(0.0, 0.0144 - Q.centroid_offset) * max(0.0, 0.263 - Q.tau21_b2) / 0.0005077523   # +0.8%  centroid_offset < 0.0144 and tau21_b2 < 0.263
        + 0.007209399 * max(0.0, 0.0061 - Q.sum_zz_dr2) * max(0.0, 121.0 - Q.pt_2) / 0.07515252   # +0.7%  sum_zz_dr2 < 0.0061 and pt_2 < 121
        + 0.006051283 * max(0.0, 0.0141 - Q.centroid_offset) * max(0.0, 0.0608 - Q.tau21_b2) / 7.201172e-05   # +0.6%  centroid_offset < 0.0141 and tau21_b2 < 0.0608
        - 0.005768782 * max(0.0, 0.014 - Q.centroid_offset) * max(0.0, 1.38 - Q.D2) / 0.001463667   # -0.6%  centroid_offset < 0.014 and D2 < 1.38
        - 0.00529445 * max(0.0, 0.266 - Q.planar_flow) * max(0.0, 36.9 - Q.pt_7) / 0.5764851   # -0.5%  planar_flow < 0.266 and pt_7 < 36.9
        + 0.005102465 * max(0.0, 0.00143 - Q.mean_phi2) * max(0.0, Q.M2 - 0.0154) / 3.909643e-05   # +0.5%  mean_phi2 < 0.00143 and M2 > 0.0154
        + 0.005048158 * max(0.0, Q.sj2_dr - 0.269) / 0.008432789   # +0.5%  sj2_dr > 0.269
        - 0.005003668 * max(0.0, 0.0061 - Q.sum_zz_dr2) * max(0.0, 0.147 - Q.z_3rd) / 6.288381e-05   # -0.5%  sum_zz_dr2 < 0.0061 and z_3rd < 0.147
        - 0.003655108 * max(0.0, Q.log_sum_pt - 6.73) / 0.02730624   # -0.4%  log_sum_pt > 6.73
        - 0.001740804 * max(0.0, 0.239 - Q.planar_flow) * max(0.0, 9.55e-06 - Q.e3) / 1.349225e-07   # -0.2%  planar_flow < 0.239 and e3 < 9.55e-06
        + 0.001078727 * max(0.0, 0.00556 - Q.lam1_plus_lam2) * max(0.0, Q.sj3_dr23 - 0.146) / 1.726896e-05   # +0.1%  lam1_plus_lam2 < 0.00556 and sj3_dr23 > 0.146
        - 0.000668443 * max(0.0, 0.00346 - Q.sum_zz_dr2) * max(0.0, Q.sum_z_dr2_top3 - 0.000792) / 2.575569e-07   # -0.1%  sum_zz_dr2 < 0.00346 and sum_z_dr2_top3 > 0.000792
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.6509;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.6509028 * (-0.3088019
        + 0.4252466 * max(0.0, Q.sum_z_dr2 - 0.0185) / 0.0007931065   # +42.5%  sum_z_dr2 > 0.0185
        - 0.2259954 * max(0.0, Q.e2 - 0.0634) / 0.001765919   # -22.6%  e2 > 0.0634
        - 0.07292498 * max(0.0, Q.sum_z_dr2 - 0.0189) * max(0.0, 52.7 - Q.pt_7) / 0.01371881   # -7.3%  sum_z_dr2 > 0.0189 and pt_7 < 52.7
        + 0.06972285 * max(0.0, Q.sum_z_dr2 - 0.0253) / 0.0002854264   # +7.0%  sum_z_dr2 > 0.0253
        - 0.06416826 * max(0.0, Q.sum_z_dr2 - 0.019) * max(0.0, 96.3 - Q.pt_2) / 0.02067688   # -6.4%  sum_z_dr2 > 0.019 and pt_2 < 96.3
        - 0.06047032 * max(0.0, Q.sum_z_dr2 - 0.0187) * max(0.0, 0.798 - Q.planar_flow) / 0.0003200024   # -6.0%  sum_z_dr2 > 0.0187 and planar_flow < 0.798
        + 0.04485217 * max(0.0, Q.lam1_plus_lam2 - 0.0163) / 0.001035262   # +4.5%  lam1_plus_lam2 > 0.0163
        + 0.03661941 * max(0.0, Q.lam1_plus_lam2 - 0.0122) * max(0.0, Q.log_sum_pt - 6.34) / 8.927219e-05   # +3.7%  lam1_plus_lam2 > 0.0122 and log_sum_pt > 6.34
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 20.86;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.86262 * (0.03570981
        + 0.3172374 * max(0.0, 0.149 - Q.sum_z_dr) / 0.09116256   # +31.7%  sum_z_dr < 0.149
        + 0.1108704 * max(0.0, 0.0137 - Q.lam1_plus_lam2) / 0.00863077   # +11.1%  lam1_plus_lam2 < 0.0137
        - 0.09782168 * max(0.0, 0.116 - Q.tau1) / 0.05932605   # -9.8%  tau1 < 0.116
        - 0.07651527 * max(0.0, 0.00934 - Q.C2_b2) / 0.007565445   # -7.7%  C2_b2 < 0.00934
        + 0.06850447 * max(0.0, Q.sum_pt_top5 - 494.0) * max(0.0, 3.04e-08 - Q.e4) / 3.831589e-06   # +6.9%  sum_pt_top5 > 494 and e4 < 3.04e-08
        - 0.06541822 * max(0.0, Q.sum_pt_top5 - 492.0) / 131.2303   # -6.5%  sum_pt_top5 > 492
        - 0.05315426 * max(0.0, 0.145 - Q.sum_z_dr) * max(0.0, 6.8 - Q.log_sum_pt) / 0.01885947   # -5.3%  sum_z_dr < 0.145 and log_sum_pt < 6.8
        - 0.04538213 * max(0.0, 9.3e-05 - Q.e3) * max(0.0, 0.0387 - Q.centroid_offset) / 1.696756e-06   # -4.5%  e3 < 9.3e-05 and centroid_offset < 0.0387
        + 0.02930423 * max(0.0, Q.z_7 - 0.0465) / 0.01189422   # +2.9%  z_7 > 0.0465
        - 0.02209988 * max(0.0, 0.15 - Q.sum_z_dr) * max(0.0, 38.3 - Q.pt_7) / 0.6672378   # -2.2%  sum_z_dr < 0.15 and pt_7 < 38.3
        + 0.020744 * max(0.0, Q.sum_pt_top5 - 687.0) * max(0.0, 38.7 - Q.pt_7) / 609.541   # +2.1%  sum_pt_top5 > 687 and pt_7 < 38.7
        - 0.01740396 * max(0.0, Q.pt_7 - 31.8) / 5.981747   # -1.7%  pt_7 > 31.8
        - 0.01291048 * max(0.0, 50.5 - Q.pt_6) / 11.86548   # -1.3%  pt_6 < 50.5
        - 0.01144786 * max(0.0, 6.48 - Q.log_sum_pt) / 0.07581975   # -1.1%  log_sum_pt < 6.48
        - 0.0104089 * max(0.0, 0.149 - Q.sum_z_dr) * max(0.0, 0.000283 - Q.lam2) / 1.904885e-05   # -1.0%  sum_z_dr < 0.149 and lam2 < 0.000283
        - 0.009811632 * max(0.0, 0.0175 - Q.lam1) * max(0.0, 25.5 - Q.pt_7) / 0.01987343   # -1.0%  lam1 < 0.0175 and pt_7 < 25.5
        - 0.009166723 * max(0.0, Q.sj3_dr_max - 0.232) / 0.02553295   # -0.9%  sj3_dr_max > 0.232
        + 0.005855712 * max(0.0, 0.0822 - Q.e2) * max(0.0, 59.8 - Q.pt_4) / 0.4426285   # +0.6%  e2 < 0.0822 and pt_4 < 59.8
        - 0.005109605 * max(0.0, 27.7 - Q.pt_6) / 1.005658   # -0.5%  pt_6 < 27.7
        + 0.004471861 * max(0.0, 0.0299 - Q.M2) / 0.003068905   # +0.4%  M2 < 0.0299
        - 0.003374492 * max(0.0, Q.sj3_dr_max - 0.225) * max(0.0, 0.418 - Q.tau32) / 0.003868172   # -0.3%  sj3_dr_max > 0.225 and tau32 < 0.418
        - 0.00298687 * max(0.0, 25.3 - Q.pt_5) / 0.3212058   # -0.3%  pt_5 < 25.3
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 32.92;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 32.91618 * (-0.02682571
        + 0.09558685 * max(0.0, Q.sum_z_dr - 0.0411) / 0.02689191   # +9.6%  sum_z_dr > 0.0411
        - 0.09407598 * max(0.0, Q.sum_zz_dr2 - 0.0117) / 0.001427015   # -9.4%  sum_zz_dr2 > 0.0117
        - 0.07387875 * max(0.0, 0.233 - Q.sj3_dr_max) / 0.09006689   # -7.4%  sj3_dr_max < 0.233
        - 0.06638295 * max(0.0, Q.sum_z_dr2 - 0.00748) / 0.002480219   # -6.6%  sum_z_dr2 > 0.00748
        - 0.06176447 * max(0.0, Q.sum_z_dr2 - 0.00671) / 0.002692782   # -6.2%  sum_z_dr2 > 0.00671
        + 0.06143171 * max(0.0, Q.sum_zz_dr2 - 0.00715) / 0.002241793   # +6.1%  sum_zz_dr2 > 0.00715
        - 0.06030902 * max(0.0, Q.tau1 - 0.0345) / 0.03731471   # -6.0%  tau1 > 0.0345
        - 0.05901706 * max(0.0, Q.sum_z_dr2 - 0.00868) * max(0.0, Q.log_sum_pt - 6.27) / 0.0002088834   # -5.9%  sum_z_dr2 > 0.00868 and log_sum_pt > 6.27
        - 0.05880036 * max(0.0, Q.sum_z_dr - 0.102) / 0.005391318   # -5.9%  sum_z_dr > 0.102
        - 0.04907004 * max(0.0, Q.sum_z_dr - 0.0873) / 0.007840768   # -4.9%  sum_z_dr > 0.0873
        + 0.04531392 * max(0.0, Q.sum_z_dr - 0.0263) / 0.03655787   # +4.5%  sum_z_dr > 0.0263
        + 0.03272895 * max(0.0, Q.sum_zz_dr2 - 0.00815) / 0.002017438   # +3.3%  sum_zz_dr2 > 0.00815
        + 0.02746917 * max(0.0, Q.lam1_plus_lam2 - 0.00556) / 0.003096507   # +2.7%  lam1_plus_lam2 > 0.00556
        - 0.02584312 * max(0.0, 4.96 - Q.n_dr_0p05_0p1) / 3.127415   # -2.6%  n_dr_0p05_0p1 < 4.96
        + 0.01735118 * max(0.0, Q.sum_zz_dr2 - 0.00817) * max(0.0, Q.sum_pt - 493.0) / 0.1641191   # +1.7%  sum_zz_dr2 > 0.00817 and sum_pt > 493
        + 0.01454663 * max(0.0, 0.159 - Q.sj2_dr) / 0.04831679   # +1.5%  sj2_dr < 0.159
        + 0.01280796 * max(0.0, 0.572 - Q.z_dr_0p05_0p1) / 0.3603326   # +1.3%  z_dr_0p05_0p1 < 0.572
        - 0.01279915 * max(0.0, Q.centroid_offset - 0.0376) / 0.001705665   # -1.3%  centroid_offset > 0.0376
        + 0.01277745 * max(0.0, Q.n_dr_0_0p05 - 5.02) / 1.083982   # +1.3%  n_dr_0_0p05 > 5.02
        - 0.011656 * max(0.0, Q.lam1_plus_lam2 - 0.0129) / 0.001492883   # -1.2%  lam1_plus_lam2 > 0.0129
        - 0.01081183 * max(0.0, Q.centroid_offset - 0.05) / 0.0008015409   # -1.1%  centroid_offset > 0.05
        + 0.01018458 * max(0.0, Q.LHA - 0.304) / 0.01773743   # +1.0%  LHA > 0.304
        + 0.009047897 * max(0.0, Q.sum_zz_dr2 - 0.00217) * max(0.0, Q.log_sum_pt - 6.26) / 0.0007464215   # +0.9%  sum_zz_dr2 > 0.00217 and log_sum_pt > 6.26
        + 0.008646981 * max(0.0, Q.psi_0p1 - 0.635) * max(0.0, Q.eccentricity - 0.971) / 0.001790098   # +0.9%  psi_0p1 > 0.635 and eccentricity > 0.971
        + 0.007420306 * max(0.0, Q.n_dr_0p1_0p2 - 1.09) / 0.8600286   # +0.7%  n_dr_0p1_0p2 > 1.09
        - 0.00676146 * max(0.0, Q.z_dr_0p05_0p1 - 0.751) * max(0.0, Q.dr0_5 - 0.29) / 2.771624e-05   # -0.7%  z_dr_0p05_0p1 > 0.751 and dr0_5 > 0.29
        - 0.006295631 * max(0.0, 0.0356 - Q.C2) / 0.0149085   # -0.6%  C2 < 0.0356
        + 0.006142478 * max(0.0, 1.95e-05 - Q.e3) / 8.424454e-06   # +0.6%  e3 < 1.95e-05
        + 0.005950872 * max(0.0, Q.tau1 - 0.0969) / 0.01020208   # +0.6%  tau1 > 0.0969
        + 0.005906349 * max(0.0, 0.154 - Q.z_dr_0p1_0p2) / 0.09213954   # +0.6%  z_dr_0p1_0p2 < 0.154
        - 0.005821332 * max(0.0, Q.psi_0p1 - 0.975) / 0.01101241   # -0.6%  psi_0p1 > 0.975
        - 0.005644365 * max(0.0, Q.centroid_offset - 0.0234) / 0.004030172   # -0.6%  centroid_offset > 0.0234
        - 0.005349299 * max(0.0, 0.112 - Q.planar_flow) * max(0.0, 0.00725 - Q.lam1) / 5.773065e-05   # -0.5%  planar_flow < 0.112 and lam1 < 0.00725
        - 0.00361395 * max(0.0, Q.sum_zz_dr2 - 0.00232) * max(0.0, 0.188 - Q.sj2_dr) / 1.888213e-05   # -0.4%  sum_zz_dr2 > 0.00232 and sj2_dr < 0.188
        - 0.002610201 * max(0.0, 0.109 - Q.planar_flow) * max(0.0, 0.0185 - Q.centroid_offset) / 0.0002164178   # -0.3%  planar_flow < 0.109 and centroid_offset < 0.0185
        - 0.002295542 * max(0.0, 0.111 - Q.planar_flow) * max(0.0, Q.z_dr_0p05_0p1 - 0.676) / 0.002202929   # -0.2%  planar_flow < 0.111 and z_dr_0p05_0p1 > 0.676
        - 0.002012413 * max(0.0, Q.centroid_offset - 0.0491) * max(0.0, Q.eta_5 - 0.107) / 7.774761e-06   # -0.2%  centroid_offset > 0.0491 and eta_5 > 0.107
        + 0.001535909 * max(0.0, Q.psi_0p1 - 0.809) * max(0.0, Q.centroid_offset - 0.0375) / 5.310531e-05   # +0.2%  psi_0p1 > 0.809 and centroid_offset > 0.0375
        - 0.0003379023 * max(0.0, Q.lam1_plus_lam2 - 0.0133) * max(0.0, Q.sum_pt - 488.0) / 0.1069467   # -0.0%  lam1_plus_lam2 > 0.0133 and sum_pt > 488
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 25.89;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.88573 * (-0.0409492
        - 0.1905082 * max(0.0, 0.00749 - Q.sum_z_dr2) / 0.003522461   # -19.1%  sum_z_dr2 < 0.00749
        + 0.1492551 * max(0.0, 0.102 - Q.tau1) / 0.04769848   # +14.9%  tau1 < 0.102
        - 0.1359906 * max(0.0, 0.114 - Q.tau1) / 0.057614   # -13.6%  tau1 < 0.114
        + 0.08945902 * max(0.0, 0.0136 - Q.sum_z_dr2) / 0.008545064   # +8.9%  sum_z_dr2 < 0.0136
        + 0.0636413 * max(0.0, 0.0822 - Q.tau1) / 0.03334821   # +6.4%  tau1 < 0.0822
        + 0.04688116 * max(0.0, 0.00642 - Q.sum_zz_dr2) / 0.002974395   # +4.7%  sum_zz_dr2 < 0.00642
        + 0.04201001 * max(0.0, Q.sj3_dr_max - 0.185) / 0.04057686   # +4.2%  sj3_dr_max > 0.185
        - 0.03204259 * max(0.0, 0.226 - Q.N2) * max(0.0, Q.sum_z_dr2 - 0.00886) / 0.000120559   # -3.2%  N2 < 0.226 and sum_z_dr2 > 0.00886
        + 0.0301521 * max(0.0, 0.223 - Q.N2) / 0.05101367   # +3.0%  N2 < 0.223
        - 0.02753709 * max(0.0, 0.00268 - Q.lam1_plus_lam2) / 0.0008118654   # -2.8%  lam1_plus_lam2 < 0.00268
        + 0.0211692 * max(0.0, 0.198 - Q.planar_flow) * max(0.0, Q.sum_pt_top5 - 414.0) / 15.13757   # +2.1%  planar_flow < 0.198 and sum_pt_top5 > 414
        - 0.01771059 * max(0.0, Q.max_dr - 0.12) / 0.03638505   # -1.8%  max_dr > 0.12
        - 0.01443684 * max(0.0, 0.22 - Q.N2) * max(0.0, 8.24e-05 - Q.e3) / 2.613344e-06   # -1.4%  N2 < 0.22 and e3 < 8.24e-05
        - 0.01435377 * max(0.0, 0.22 - Q.N2) * max(0.0, Q.LHA - 0.409) / 0.0001161118   # -1.4%  N2 < 0.22 and LHA > 0.409
        - 0.01429437 * max(0.0, Q.sj3_dr_max - 0.268) / 0.01770432   # -1.4%  sj3_dr_max > 0.268
        + 0.01319538 * max(0.0, 0.233 - Q.N2) * max(0.0, Q.LHA - 0.327) / 0.0008371867   # +1.3%  N2 < 0.233 and LHA > 0.327
        - 0.01285811 * max(0.0, 0.00216 - Q.sum_z_dr2_top3) / 0.0007831567   # -1.3%  sum_z_dr2_top3 < 0.00216
        - 0.01074998 * max(0.0, 0.199 - Q.planar_flow) * max(0.0, 0.0715 - Q.z_7) / 0.001666294   # -1.1%  planar_flow < 0.199 and z_7 < 0.0715
        - 0.00997314 * max(0.0, Q.z_dr_0p05_0p1 - 0.752) / 0.02264579   # -1.0%  z_dr_0p05_0p1 > 0.752
        + 0.009837836 * max(0.0, Q.z_dr_0p05_0p1 - 0.743) * max(0.0, 0.0548 - Q.z_dr_0p2_0p4) / 0.001195585   # +1.0%  z_dr_0p05_0p1 > 0.743 and z_dr_0p2_0p4 < 0.0548
        - 0.009249435 * max(0.0, 0.221 - Q.N2) * max(0.0, 0.187 - Q.sj2_dr) / 0.0007600902   # -0.9%  N2 < 0.221 and sj2_dr < 0.187
        + 0.009179928 * max(0.0, 0.232 - Q.N2) * max(0.0, Q.sum_pt_top5 - 408.0) / 10.60845   # +0.9%  N2 < 0.232 and sum_pt_top5 > 408
        - 0.008067426 * max(0.0, 0.227 - Q.N2) * max(0.0, 0.647 - Q.z_dr_0p05_0p1) / 0.01356047   # -0.8%  N2 < 0.227 and z_dr_0p05_0p1 < 0.647
        + 0.007579722 * max(0.0, 0.223 - Q.N2) * max(0.0, Q.LHA - 0.292) / 0.001569653   # +0.8%  N2 < 0.223 and LHA > 0.292
        - 0.006705799 * max(0.0, 0.207 - Q.planar_flow) * max(0.0, Q.mean_phi - -0.0197) / 0.001701809   # -0.7%  planar_flow < 0.207 and mean_phi > -0.0197
        - 0.0066335 * max(0.0, 0.00642 - Q.sum_z_dr2) * max(0.0, Q.pt_entropy - 1.81) / 0.0002224262   # -0.7%  sum_z_dr2 < 0.00642 and pt_entropy > 1.81
        - 0.004046562 * max(0.0, 0.00207 - Q.sum_z_dr2_top3) * max(0.0, Q.eccentricity - 0.957) / 4.594221e-06   # -0.4%  sum_z_dr2_top3 < 0.00207 and eccentricity > 0.957
        - 0.001463471 * max(0.0, 0.219 - Q.N2) * max(0.0, 25.7 - Q.pt_7) / 0.03714022   # -0.1%  N2 < 0.219 and pt_7 < 25.7
        - 0.0007483125 * max(0.0, 0.0153 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.0501) / 1.354589e-06   # -0.1%  sum_z_dr2 < 0.0153 and centroid_offset > 0.0501
        - 0.0002695026 * max(0.0, 0.00726 - Q.sum_z_dr2) * max(0.0, Q.dr12 - 0.171) / 1.386933e-06   # -0.0%  sum_z_dr2 < 0.00726 and dr12 > 0.171
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9169002100840337, 0.7867594537815126, 2.0973301470588237, 0.8982077731092437, 1.201713025210084, 1.7033306722689077, 1.1343548319327732, 1.8997731092436976, 0.2594531512605042, 2.9859798319327733, 1.9333490546218488, 2.0899317226890757, 0.08908949579831933, 3.3137100840336133, 0.42002174369747897, 0.3612716386554622]
T = [2.3460034934676997, 1.3729214482668068, 3.2972211725315126, 2.528474120273109, 2.8109041787027307]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +38%, n9 +22%, n5 -14%, n1 +13%, n0 -6%, n6 +5% ...
            + 0.3841412 * h[2] / H_AVG[2]
            + 0.2187615 * h[9] / H_AVG[9]
            - 0.1361356 * h[5] / H_AVG[5]
            + 0.1310006 * h[1] / H_AVG[1]
            - 0.06106796 * h[0] / H_AVG[0]
            + 0.05288571 * h[6] / H_AVG[6]
            - 0.01600745 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +55%, n10 -18%, n6 +10%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5522231 * h[9] / H_AVG[9]
            - 0.1760251 * h[10] / H_AVG[10]
            + 0.1032793 * h[6] / H_AVG[6]
            - 0.08205903 * h[4] / H_AVG[4]
            + 0.058156 * h[5] / H_AVG[5]
            + 0.0164463 * h[15] / H_AVG[15]
            + 0.01181118 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +13%, n6 -11%, n0 +10%, n14 -10% ...
            + 0.2376924 * h[11] / H_AVG[11]
            - 0.1362068 * h[3] / H_AVG[3]
            + 0.1260381 * h[7] / H_AVG[7]
            - 0.1075105 * h[6] / H_AVG[6]
            + 0.09559093 * h[0] / H_AVG[0]
            - 0.09553994 * h[14] / H_AVG[14]
            - 0.07532836 * h[15] / H_AVG[15]
            + 0.07066412 * h[13] / H_AVG[13]
            - 0.02830015 * h[9] / H_AVG[9]
            - 0.01967211 * h[8] / H_AVG[8]
            - 0.007456653 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -20%, n6 -17%, n13 +7%, n14 +6%, n1 +4% ...
            + 0.3521961 * h[7] / H_AVG[7]
            - 0.1998209 * h[3] / H_AVG[3]
            - 0.1682371 * h[6] / H_AVG[6]
            + 0.0716711 * h[13] / H_AVG[13]
            + 0.06229376 * h[14] / H_AVG[14]
            + 0.03889497 * h[1] / H_AVG[1]
            + 0.03713063 * h[4] / H_AVG[4]
            - 0.03690442 * h[9] / H_AVG[9]
            - 0.0223252 * h[15] / H_AVG[15]
            + 0.01052593 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -48%, n10 +26%, n5 -15%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4789188 * h[13] / H_AVG[13]
            + 0.2579262 * h[10] / H_AVG[10]
            - 0.1514931 * h[5] / H_AVG[5]
            + 0.05343979 * h[4] / H_AVG[4]
            + 0.0199715 * h[3] / H_AVG[3]
            + 0.0173067 * h[8] / H_AVG[8]
            - 0.01584712 * h[12] / H_AVG[12]
            + 0.005096782 * h[0] / H_AVG[0]
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
