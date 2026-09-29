"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  13.9%   (on for 87% of jets)
  neuron  9:  12.0%   (on for 70% of jets)
  neuron  7:  10.6%   (on for 61% of jets)
  neuron  3:   8.6%   (on for 22% of jets)
  neuron 10:   7.9%   (on for 83% of jets)
  neuron  6:   7.8%   (on for 36% of jets)
  neuron  2:   7.1%   (on for 90% of jets)
  neuron  5:   7.1%   (on for 53% of jets)
  neuron 11:   6.4%   (on for 76% of jets)
  neuron  0:   3.7%   (on for 40% of jets)
  neuron 14:   3.6%   (on for 27% of jets)
  neuron  1:   3.4%   (on for 60% of jets)
  neuron  4:   3.3%   (on for 55% of jets)
  neuron 15:   2.8%   (on for 26% of jets)
  neuron  8:   1.3%   (on for 58% of jets)
  neuron 12:   0.4%   (on for 9% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 89.9% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
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
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z1                 pT share of subjet 1 of 3 (by pT)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr12                   ΔR between particles 1 and 2
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
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
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z1=subjets(3)["z"][0],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        zdr_6=z[6] * dr[6],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        abseta_0=abs(eta[0]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 33.36;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.36428 * (0.2209309
        - 0.2575833 * Q.log_sum_pt / 6.542932   # -25.8%  log_sum_pt
        + 0.1278666 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +12.8%  lam1_plus_lam2 < 0.008678
        + 0.0903789 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +9.0%  sum_z_dr2 < 0.01324
        - 0.06021178 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -6.0%  sum_z_dr < 0.08724
        + 0.05758735 * max(0.0, 0.01882765 - Q.sum_z_dr2) / 0.01314296   # +5.8%  sum_z_dr2 < 0.01883
        - 0.04296813 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -4.3%  lam1 < 0.008376
        - 0.03837473 * max(0.0, 0.005590289 - Q.sum_z_dr2) / 0.002229311   # -3.8%  sum_z_dr2 < 0.00559
        + 0.03752204 * max(0.0, 0.1116471 - Q.sd_rg) / 0.04385005   # +3.8%  sd_rg < 0.1116
        - 0.03402904 * max(0.0, 0.00718279 - Q.sum_zz_dr2) / 0.003528787   # -3.4%  sum_zz_dr2 < 0.007183
        - 0.03103004 * max(0.0, 0.1773029 - Q.sd_rg) / 0.08115541   # -3.1%  sd_rg < 0.1773
        - 0.03101373 * max(0.0, 788.4484 - Q.sum_pt) / 112.1576   # -3.1%  sum_pt < 788.4
        - 0.02915617 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) / 0.00156571   # -2.9%  lam1_plus_lam2 < 0.004372
        + 0.02213582 * max(0.0, 0.06345984 - Q.tau1) / 0.02211428   # +2.2%  tau1 < 0.06346
        - 0.0169607 * max(0.0, 0.1027642 - Q.tau1) / 0.04830819   # -1.7%  tau1 < 0.1028
        + 0.01693596 * max(0.0, 631.275 - Q.sum_pt_top5) / 94.34106   # +1.7%  sum_pt_top5 < 631.3
        - 0.01690256 * max(0.0, 0.01357153 - Q.tau2) / 0.005299336   # -1.7%  tau2 < 0.01357
        + 0.01296652 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 1.129616 - Q.D2_b2) / 0.04715373   # +1.3%  planar_flow < 0.1484 and D2_b2 < 1.13
        - 0.01206205 * max(0.0, 0.01517359 - Q.sum_z_dr) / 0.001389803   # -1.2%  sum_z_dr < 0.01517
        - 0.01165221 * max(0.0, 0.007929074 - Q.sum_z_dr2_top3) / 0.004560262   # -1.2%  sum_z_dr2_top3 < 0.007929
        - 0.01106078 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.001184249   # -1.1%  lam1_plus_lam2 < 0.003563
        - 0.009038955 * max(0.0, 0.06545715 - Q.sd_rg) / 0.02200467   # -0.9%  sd_rg < 0.06546
        - 0.006381434 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -0.6%  sum_pt < 739.5
        - 0.005688718 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, 0.380911 - Q.D2_b2) / 12.71118   # -0.6%  sum_pt < 788.4 and D2_b2 < 0.3809
        - 0.005306033 * max(0.0, 0.01882765 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.009480685) / 8.306781e-05   # -0.5%  sum_z_dr2 < 0.01883 and centroid_offset > 0.009481
        + 0.004625068 * max(0.0, 0.01882765 - Q.sum_z_dr2) * max(0.0, 0.09029177 - Q.M3) / 0.0001995224   # +0.5%  sum_z_dr2 < 0.01883 and M3 < 0.09029
        - 0.003329558 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.3%  centroid_offset > 0.0499
        + 0.003082418 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +0.3%  planar_flow < 0.1484
        - 0.001203439 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.dr0_7 - 0.2405707) / 1.062493e-07   # -0.1%  e3 < 8.148e-05 and dr0_7 > 0.2406
        - 0.001085683 * max(0.0, 0.06345984 - Q.tau1) * max(0.0, Q.dr0_6 - 0.1755206) / 7.050329e-05   # -0.1%  tau1 < 0.06346 and dr0_6 > 0.1755
        + 0.0009658054 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) * max(0.0, Q.dr0_7 - 0.1786203) / 1.674303e-06   # +0.1%  lam1_plus_lam2 < 0.003563 and dr0_7 > 0.1786
        + 0.0008945048 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) * max(0.0, Q.dr0_6 - 0.1755206) / 1.823868e-06   # +0.1%  lam1_plus_lam2 < 0.004372 and dr0_6 > 0.1755
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 16.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.18148 * (0.07332496
        - 0.1467948 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # -14.7%  lam1_plus_lam2 < 0.008678
        + 0.1259235 * max(0.0, 0.005834489 - Q.sum_zz_dr2) / 0.002579453   # +12.6%  sum_zz_dr2 < 0.005834
        + 0.0981988 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1442881 - Q.dr_0) / 0.0005125595   # +9.8%  lam1 < 0.008376 and dr_0 < 0.1443
        - 0.09784141 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -9.8%  sum_z_dr < 0.1019
        - 0.09027314 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -9.0%  z_7 < 0.06165
        + 0.08844182 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +8.8%  log_sum_pt > 6.378
        + 0.06024736 * max(0.0, Q.log_sum_pt - 6.572938) / 0.08631512   # +6.0%  log_sum_pt > 6.573
        + 0.05915195 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.0001869378 - Q.e3) / 2.398316e-06   # +5.9%  z_7 < 0.06165 and e3 < 0.0001869
        - 0.04609672 * max(0.0, 0.00609665 - Q.sum_z_dr2) / 0.002544039   # -4.6%  sum_z_dr2 < 0.006097
        - 0.04127556 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.08082334 - Q.dr_0) / 0.01037756   # -4.1%  log_sum_pt > 6.378 and dr_0 < 0.08082
        - 0.0263424 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # -2.6%  tau1 < 0.07284
        + 0.0195164 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.01403324 - Q.sum_z_dr2_top2) / 0.0001680113   # +2.0%  z_7 < 0.06165 and sum_z_dr2_top2 < 0.01403
        + 0.01920596 * max(0.0, Q.pt_7 - 38.53125) / 2.781159   # +1.9%  pt_7 > 38.53
        - 0.01896846 * max(0.0, 0.05240025 - Q.z_7) / 0.008882498   # -1.9%  z_7 < 0.0524
        - 0.01275699 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.007929074 - Q.sum_z_dr2_top3) / 0.0005408911   # -1.3%  log_sum_pt > 6.573 and sum_z_dr2_top3 < 0.007929
        - 0.01185609 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -1.2%  centroid_offset < 0.02077
        - 0.01010583 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 0.008654951 - Q.zdr_6) / 0.0005705688   # -1.0%  log_sum_pt > 6.573 and zdr_6 < 0.008655
        - 0.00830729 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 2.955458e-05 - Q.e3) / 7.137805e-05   # -0.8%  pt_7 > 34.53 and e3 < 2.955e-05
        + 0.005307309 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.003060958   # +0.5%  sj2_dr > 0.1873 and sj3_dr_min < 0.209
        - 0.004525893 * max(0.0, 0.05240025 - Q.z_7) * max(0.0, Q.dr0_6 - 0.003538987) / 0.0006137886   # -0.5%  z_7 < 0.0524 and dr0_6 > 0.003539
        - 0.003613265 * max(0.0, 0.05240025 - Q.z_7) * max(0.0, 0.1236856 - Q.D2_b2) / 0.000152235   # -0.4%  z_7 < 0.0524 and D2_b2 < 0.1237
        + 0.003040729 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # +0.3%  lam1 < 0.008376 and centroid_offset > 0.02077
        - 0.002208374 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.2%  pt_7 > 53.44
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 16.47;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.46981 * (0.05640383
        + 0.1819277 * Q.pt_7 / 34.64819   # +18.2%  pt_7
        - 0.1061841 * max(0.0, 0.0423228 - Q.C2) * max(0.0, 0.02415398 - Q.C2_b2) / 0.0004792616   # -10.6%  C2 < 0.04232 and C2_b2 < 0.02415
        - 0.09367681 * max(0.0, Q.LHA - 0.111565) / 0.1352185   # -9.4%  LHA > 0.1116
        + 0.09126823 * max(0.0, 6.701242 - Q.log_sum_pt) / 0.1935491   # +9.1%  log_sum_pt < 6.701
        + 0.0888106 * max(0.0, 0.0423228 - Q.C2) / 0.02012665   # +8.9%  C2 < 0.04232
        - 0.07621457 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # -7.6%  z_7 > 0.02321
        + 0.06713622 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +6.7%  lam1_plus_lam2 < 0.01324
        + 0.04691471 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 0.2507612 - Q.max_dr) / 0.00062784   # +4.7%  lam1 < 0.00733 and max_dr < 0.2508
        - 0.04615558 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -4.6%  pt_7 < 53.44
        - 0.04538008 * max(0.0, 0.00718279 - Q.sum_zz_dr2) / 0.003528787   # -4.5%  sum_zz_dr2 < 0.007183
        - 0.03236078 * max(0.0, Q.z_7 - 0.02807091) / 0.02541111   # -3.2%  z_7 > 0.02807
        + 0.02332214 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # +2.3%  lam1 < 0.003377
        - 0.01883224 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # -1.9%  sum_zz_dr2 < 0.003013
        - 0.01811578 * max(0.0, 6.701242 - Q.log_sum_pt) * max(0.0, 1.129616 - Q.D2_b2) / 0.1081022   # -1.8%  log_sum_pt < 6.701 and D2_b2 < 1.13
        + 0.01585262 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 1.129616 - Q.D2_b2) / 0.07303201   # +1.6%  log_sum_pt < 6.606 and D2_b2 < 1.13
        - 0.009545136 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 62.25 - Q.pt_6) / 0.07984241   # -1.0%  lam1 < 0.00733 and pt_6 < 62.25
        + 0.009227498 * max(0.0, 615.875 - Q.sum_pt) / 30.07766   # +0.9%  sum_pt < 615.9
        - 0.006066756 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.6%  sum_pt > 988.4
        - 0.005693415 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # -0.6%  log_sum_pt < 6.606
        - 0.0047409 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # -0.5%  sum_z_dr < 0.007674
        + 0.004061269 * max(0.0, Q.sum_pt - 937.0312) / 8.087955   # +0.4%  sum_pt > 937
        - 0.002391495 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 0.01425934 - Q.abseta_0) / 0.1176352   # -0.2%  sum_pt_top5 > 791.1 and abseta_0 < 0.01426
        + 0.002000703 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.2%  log_sum_pt > 6.896
        + 0.001907323 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 1.129616 - Q.D2_b2) / 3.947956   # +0.2%  sum_pt_top5 > 791.1 and D2_b2 < 1.13
        + 0.001145687 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.01425934 - Q.abseta_0) / 0.03515712   # +0.1%  sum_pt > 988.4 and abseta_0 < 0.01426
        - 0.001067664 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 1.59265 - Q.D2_b2) / 0.00192904   # -0.1%  log_sum_pt > 6.896 and D2_b2 < 1.593
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 17.84;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.83569 * (-0.1844277
        + 0.148727 * max(0.0, Q.sum_z_dr - 0.06663269) / 0.01393206   # +14.9%  sum_z_dr > 0.06663
        - 0.1399523 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # -14.0%  sum_z_dr2 > 0.008678
        + 0.074818 * max(0.0, Q.sum_zz_dr2 - 0.00718279) / 0.002233568   # +7.5%  sum_zz_dr2 > 0.007183
        + 0.07195435 * max(0.0, Q.lam1_plus_lam2 - 0.00609665) / 0.002892623   # +7.2%  lam1_plus_lam2 > 0.006097
        - 0.06779578 * max(0.0, Q.tau1 - 0.0811449) / 0.01489378   # -6.8%  tau1 > 0.08114
        - 0.04570078 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 7.0 - Q.n_dr_0p05_0p1) / 0.1780556   # -4.6%  sj2_dr > 0.1683 and n_dr_0p05_0p1 < 7
        + 0.04426706 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # +4.4%  sj2_dr > 0.1683
        + 0.0437375 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # +4.4%  sj2_dr > 0.1779
        - 0.04034198 * max(0.0, Q.sum_zz_dr2 - 0.006390125) / 0.002450953   # -4.0%  sum_zz_dr2 > 0.00639
        + 0.03803122 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, Q.eccentricity - 0.9458207) / 0.000763705   # +3.8%  sj2_dr > 0.1779 and eccentricity > 0.9458
        + 0.03331825 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +3.3%  lam1 > 0.008376
        - 0.03227767 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, Q.eccentricity - 0.9458207) / 0.0008867862   # -3.2%  sj2_dr > 0.1683 and eccentricity > 0.9458
        - 0.03040203 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # -3.0%  lam1_plus_lam2 > 0.01324
        - 0.02316882 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # -2.3%  sj2_dr > 0.2688
        + 0.02278079 * max(0.0, Q.max_dr - 0.1117619) / 0.04033009   # +2.3%  max_dr > 0.1118
        + 0.01963458 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.06297984 - Q.tau2) / 0.001263954   # +2.0%  sj2_dr > 0.1683 and tau2 < 0.06298
        - 0.01605383 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # -1.6%  sum_z_dr > 0.08724
        - 0.01316663 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 988.4078 - Q.sum_pt) / 12.22843   # -1.3%  sj2_dr > 0.1683 and sum_pt < 988.4
        + 0.01146638 * max(0.0, Q.lam1_plus_lam2 - 0.01882765) / 0.0007605369   # +1.1%  lam1_plus_lam2 > 0.01883
        - 0.01125595 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001864148 - Q.sum_z_dr2_top2) / 7.226227e-06   # -1.1%  sj2_dr > 0.1779 and sum_z_dr2_top2 < 0.001864
        + 0.01079771 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +1.1%  tau1 > 0.1136
        + 0.01035516 * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 0.001433269   # +1.0%  sum_zz_dr2 > 0.01166
        - 0.009338524 * max(0.0, Q.lam1_plus_lam2 - 0.00609665) * max(0.0, Q.log_sum_pt - 6.192222) / 0.0004573947   # -0.9%  lam1_plus_lam2 > 0.006097 and log_sum_pt > 6.192
        + 0.006606534 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 0.07848552   # +0.7%  sj2_dr > 0.1592 and n_dr_0p1_0p2 > 1
        + 0.006353525 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # +0.6%  lam1 > 0.012
        + 0.005613552 * max(0.0, Q.lam2 - 0.000537286) / 0.0003825703   # +0.6%  lam2 > 0.0005373
        - 0.004122165 * max(0.0, Q.lam1_plus_lam2 - 0.00609665) * max(0.0, Q.pt_6 - 31.90625) / 0.02357333   # -0.4%  lam1_plus_lam2 > 0.006097 and pt_6 > 31.91
        - 0.003991161 * max(0.0, Q.sj2_dr - 0.3003793) * max(0.0, Q.z_dr_0p05_0p1 - 0.08878489) / 0.001005291   # -0.4%  sj2_dr > 0.3004 and z_dr_0p05_0p1 > 0.08878
        + 0.003425376 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +0.3%  sj2_dr > 0.3004
        + 0.002911157 * max(0.0, Q.sum_z_dr - 0.08723651) * max(0.0, 0.0003061234 - Q.lam2) / 3.733963e-07   # +0.3%  sum_z_dr > 0.08724 and lam2 < 0.0003061
        + 0.002866698 * max(0.0, 1.138757 - Q.N3) / 0.04383098   # +0.3%  N3 < 1.139
        + 0.002604253 * max(0.0, Q.sj2_dr - 0.2687922) * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) / 1.542054e-05   # +0.3%  sj2_dr > 0.2688 and sum_z_dr2_top2 < 0.00764
        - 0.002163322 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9884745) / 1.823692e-06   # -0.2%  lam1_plus_lam2 > 0.01324 and eccentricity > 0.9885
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 30.35;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 30.35058 * (-0.07242596
        - 0.1618096 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -16.2%  sum_z_dr2 < 0.008678
        + 0.1324044 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +13.2%  e3 < 0.0001869
        - 0.1043259 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -10.4%  lam2 < 0.001131
        + 0.1036596 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # +10.4%  sum_zz_dr2 < 0.01166
        + 0.08898096 * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.166579   # +8.9%  sj3_dr_min < 0.209
        + 0.07465347 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # +7.5%  C2_b2 < 0.009033
        - 0.0334435 * max(0.0, 0.04447357 - Q.e2) / 0.02019334   # -3.3%  e2 < 0.04447
        + 0.03171013 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +3.2%  sj3_dr_max > 0.179
        - 0.02785419 * max(0.0, 0.01165737 - Q.sum_zz_dr2) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.0003843383   # -2.8%  sum_zz_dr2 < 0.01166 and z_dr_0p2_0p4 < 0.05644
        + 0.02390977 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # +2.4%  sum_z_dr2 < 0.002635
        - 0.02343256 * max(0.0, 0.006299534 - Q.sum_z_dr2_top2) / 0.003504831   # -2.3%  sum_z_dr2_top2 < 0.0063
        - 0.02266499 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, 813.4156 - Q.sum_pt) / 0.01550241   # -2.3%  e3 < 0.0001869 and sum_pt < 813.4
        - 0.02255607 * max(0.0, Q.sj3_dr_max - 0.213399) / 0.03061279   # -2.3%  sj3_dr_max > 0.2134
        + 0.02022891 * max(0.0, 0.07608178 - Q.sum_z_dr) / 0.02796784   # +2.0%  sum_z_dr < 0.07608
        + 0.01902031 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +1.9%  N2 < 0.2233
        - 0.01432526 * max(0.0, 0.01357153 - Q.tau2) / 0.005299336   # -1.4%  tau2 < 0.01357
        - 0.01190353 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.4007947 - Q.planar_flow) / 0.01755893   # -1.2%  N2 < 0.2233 and planar_flow < 0.4008
        + 0.01131788 * max(0.0, Q.max_dr - 0.0931108) / 0.05062176   # +1.1%  max_dr > 0.09311
        + 0.01109417 * max(0.0, 0.006299534 - Q.sum_z_dr2_top2) * max(0.0, Q.centroid_offset - 0.01096064) / 1.639932e-05   # +1.1%  sum_z_dr2_top2 < 0.0063 and centroid_offset > 0.01096
        + 0.009396241 * max(0.0, 0.7459513 - Q.D2) / 0.09626874   # +0.9%  D2 < 0.746
        + 0.009323427 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +0.9%  lam1 < 0.00484
        + 0.008449103 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) / 0.003638607   # +0.8%  sum_z_dr2_top5 < 0.007164
        - 0.006855973 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1626038 - Q.abseta_7) / 0.005313035   # -0.7%  N2 < 0.2233 and abseta_7 < 0.1626
        - 0.006085386 * max(0.0, Q.C2 - 0.02384388) / 0.01155414   # -0.6%  C2 > 0.02384
        - 0.004449429 * max(0.0, Q.sj2_dr - 0.2414351) / 0.01315894   # -0.4%  sj2_dr > 0.2414
        - 0.003984875 * max(0.0, Q.sj2_dr - 0.2414351) * max(0.0, 1.129616 - Q.D2_b2) / 0.007762894   # -0.4%  sj2_dr > 0.2414 and D2_b2 < 1.13
        + 0.002862723 * max(0.0, Q.sj2_dr - 0.2414351) * max(0.0, 0.1989187 - Q.sj3_z3) / 0.001224215   # +0.3%  sj2_dr > 0.2414 and sj3_z3 < 0.1989
        - 0.00262531 * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.00293624   # -0.3%  sum_z_dr2 < 0.006679
        + 0.002276437 * max(0.0, Q.max_dr - 0.0931108) * max(0.0, 0.04019165 - Q.absphi_0) / 0.0006099482   # +0.2%  max_dr > 0.09311 and absphi_0 < 0.04019
        - 0.001841018 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.008074527   # -0.2%  N2 < 0.2233 and n_dr_0p2_0p4 > 1
        - 0.0008969439 * max(0.0, Q.sj2_dr - 0.2414351) * max(0.0, 0.0008333816 - Q.C2_b2) / 1.323442e-06   # -0.1%  sj2_dr > 0.2414 and C2_b2 < 0.0008334
        - 0.0008308712 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 1.0 - Q.psi_0p3) / 5.287696e-07   # -0.1%  sum_z_dr2 < 0.006679 and psi_0p3 < 1
        - 0.000827127 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 24.42188 - Q.pt_6) / 0.01371831   # -0.1%  N2 < 0.2233 and pt_6 < 24.42
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 14.3;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.29508 * (-0.06477802
        + 0.1545046 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +15.5%  e3 < 0.0005117
        - 0.1343082 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -13.4%  sj3_dr_max < 0.3012
        - 0.09512509 * max(0.0, 715.4688 - Q.sum_pt) / 69.91196   # -9.5%  sum_pt < 715.5
        + 0.06816759 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +6.8%  LHA < 0.2161
        - 0.05282157 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -5.3%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.05055275 * max(0.0, 0.03556091 - Q.e2) / 0.01371972   # +5.1%  e2 < 0.03556
        + 0.04633603 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # +4.6%  sum_z_dr2 < 0.002635
        - 0.04136759 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.09543973 - Q.z_6) / 0.001514041   # -4.1%  LHA < 0.2161 and z_6 < 0.09544
        + 0.03995649 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # +4.0%  max_dr < 0.1773
        + 0.03855802 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +3.9%  sum_z_dr2 < 0.00502
        + 0.03671279 * max(0.0, 0.002635418 - Q.sum_z_dr2) * max(0.0, 0.01627885 - Q.centroid_offset) / 7.136253e-06   # +3.7%  sum_z_dr2 < 0.002635 and centroid_offset < 0.01628
        - 0.03372117 * max(0.0, 0.04649465 - Q.dr_0) / 0.01415326   # -3.4%  dr_0 < 0.04649
        - 0.02425116 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # -2.4%  pt_7 > 34.53
        + 0.02268229 * max(0.0, 0.0009641429 - Q.sum_z_dr2) / 0.0002052288   # +2.3%  sum_z_dr2 < 0.0009641
        + 0.02247883 * max(0.0, 715.4688 - Q.sum_pt) * max(0.0, 0.0361727 - Q.dr_0) / 0.2640582   # +2.2%  sum_pt < 715.5 and dr_0 < 0.03617
        - 0.01856487 * max(0.0, 430.75 - Q.sum_pt_top5) / 14.05246   # -1.9%  sum_pt_top5 < 430.8
        - 0.01671023 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -1.7%  log_sum_pt > 6.896
        + 0.01576621 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +1.6%  z_7 < 0.02807
        + 0.01552856 * max(0.0, 0.06081235 - Q.z_6) / 0.009654642   # +1.6%  z_6 < 0.06081
        - 0.01229102 * max(0.0, 0.002085238 - Q.mean_eta2) / 0.0009518454   # -1.2%  mean_eta2 < 0.002085
        + 0.01193971 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 0.04678345 - Q.absphi_0) / 0.000232957   # +1.2%  z_7 < 0.0494 and absphi_0 < 0.04678
        - 0.01040997 * max(0.0, 0.002635418 - Q.sum_z_dr2) * max(0.0, Q.pt_6 - 27.57812) / 0.01123489   # -1.0%  sum_z_dr2 < 0.002635 and pt_6 > 27.58
        + 0.008666454 * max(0.0, 0.03556091 - Q.e2) * max(0.0, Q.zdr_0 - 0.004918231) / 3.97457e-05   # +0.9%  e2 < 0.03556 and zdr_0 > 0.004918
        - 0.007892485 * max(0.0, 20.125 - Q.pt_7) / 0.5468342   # -0.8%  pt_7 < 20.12
        - 0.007283492 * max(0.0, 0.1767241 - Q.LHA) * max(0.0, 0.006292091 - Q.zdr_0) / 7.498285e-05   # -0.7%  LHA < 0.1767 and zdr_0 < 0.006292
        + 0.00531549 * max(0.0, 0.03556091 - Q.e2) * max(0.0, Q.log_sum_pt - 6.896095) / 0.0001063848   # +0.5%  e2 < 0.03556 and log_sum_pt > 6.896
        + 0.005092142 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # +0.5%  z_6 < 0.0216
        - 0.001327614 * max(0.0, 715.4688 - Q.sum_pt) * max(0.0, 0.05464886 - Q.sj3_z3) / 0.0225657   # -0.1%  sum_pt < 715.5 and sj3_z3 < 0.05465
        + 0.0009928578 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 5.507482e-05 - Q.mean_eta2) / 6.203155e-08   # +0.1%  log_sum_pt > 6.896 and mean_eta2 < 5.507e-05
        + 0.0003985016 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, Q.zdr_0 - 0.01216943) / 7.562341e-06   # +0.0%  log_sum_pt > 6.896 and zdr_0 > 0.01217
        - 0.0002762172 * max(0.0, 0.03556091 - Q.e2) * max(0.0, Q.n_dr_0p1_0p2 - 6.0) / 1.282152e-05   # -0.0%  e2 < 0.03556 and n_dr_0p1_0p2 > 6
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 40.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 40.66718 * (0.02426289
        - 0.200542 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -20.1%  sum_z_dr2 < 0.008678
        + 0.1032655 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +10.3%  tau1 < 0.1136
        + 0.09590954 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +9.6%  lam1 < 0.012
        - 0.07391573 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # -7.4%  lam1_plus_lam2 < 0.01324
        - 0.06352369 * max(0.0, Q.centroid_offset - 0.01627885) * max(0.0, 988.4078 - Q.sum_pt) / 2.602791   # -6.4%  centroid_offset > 0.01628 and sum_pt < 988.4
        + 0.04330348 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +4.3%  lam2 < 0.003408
        - 0.04039143 * max(0.0, Q.sj3_dr_max - 0.07264452) / 0.1084421   # -4.0%  sj3_dr_max > 0.07264
        - 0.03877181 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -3.9%  lam2 < 0.001131
        + 0.0307543 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # +3.1%  sj3_dr_max > 0.179
        - 0.02997335 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -3.0%  lam1 < 0.00733
        + 0.02886095 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # +2.9%  sum_zz_dr2 < 0.008169
        + 0.02812998 * max(0.0, Q.centroid_offset - 0.01627885) / 0.006307541   # +2.8%  centroid_offset > 0.01628
        + 0.0249994 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +2.5%  lam1 < 0.008376
        + 0.02437593 * max(0.0, 0.02689598 - Q.sum_z_dr) / 0.004330137   # +2.4%  sum_z_dr < 0.0269
        - 0.02394709 * max(0.0, 0.1598486 - Q.max_dr) / 0.05771655   # -2.4%  max_dr < 0.1598
        + 0.01958642 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 715.4688 - Q.sum_pt) / 1.381606   # +2.0%  centroid_offset > 0.008092 and sum_pt < 715.5
        + 0.0182553 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +1.8%  centroid_offset > 0.008092
        - 0.01539003 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.001091057   # -1.5%  lam1_plus_lam2 < 0.01324 and planar_flow < 0.3221
        - 0.01182201 * max(0.0, Q.sj3_dr_min - 0.02022982) / 0.02964063   # -1.2%  sj3_dr_min > 0.02023
        + 0.00908229 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 6.0 - Q.n_dr_0p05_0p1) / 0.03827938   # +0.9%  centroid_offset > 0.008092 and n_dr_0p05_0p1 < 6
        + 0.009016054 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.01426135 - Q.mean_phi2) / 9.496308e-05   # +0.9%  centroid_offset > 0.008092 and mean_phi2 < 0.01426
        + 0.008070548 * max(0.0, 6.327379 - Q.log_sum_pt) / 0.0331691   # +0.8%  log_sum_pt < 6.327
        + 0.007212882 * max(0.0, 1.332146 - Q.D2) / 0.3438169   # +0.7%  D2 < 1.332
        + 0.006138647 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 0.1682655 - Q.sj2_dr) / 0.0003126334   # +0.6%  centroid_offset > 0.008092 and sj2_dr < 0.1683
        + 0.005680897 * max(0.0, 3.0 - Q.n_dr_0_0p05) / 1.029338   # +0.6%  n_dr_0_0p05 < 3
        + 0.005045916 * max(0.0, 29.90625 - Q.pt_6) / 1.385454   # +0.5%  pt_6 < 29.91
        + 0.004688523 * max(0.0, 6.638339 - Q.log_sum_pt) * max(0.0, 0.0586137 - Q.z_7) / 0.0003525255   # +0.5%  log_sum_pt < 6.638 and z_7 < 0.05861
        - 0.004035178 * max(0.0, Q.tau2 - 0.01713288) / 0.005826319   # -0.4%  tau2 > 0.01713
        + 0.003741671 * max(0.0, 6.638339 - Q.log_sum_pt) / 0.1524936   # +0.4%  log_sum_pt < 6.638
        - 0.00321819 * max(0.0, 6.327379 - Q.log_sum_pt) * max(0.0, Q.pt_7 - 25.57812) / 0.2533514   # -0.3%  log_sum_pt < 6.327 and pt_7 > 25.58
        - 0.003205657 * max(0.0, Q.sj3_dr_max - 0.1789613) * max(0.0, 6.0 - Q.n_dr_0p05_0p1) / 0.1786734   # -0.3%  sj3_dr_max > 0.179 and n_dr_0p05_0p1 < 6
        - 0.002811123 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.3%  z_6 < 0.0216
        - 0.002181081 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, 38.53125 - Q.pt_7) / 0.05994081   # -0.2%  centroid_offset > 0.008092 and pt_7 < 38.53
        + 0.002007069 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_dr_min - 0.08543881) / 0.0003349792   # +0.2%  centroid_offset > 0.008092 and sj3_dr_min > 0.08544
        - 0.001990326 * max(0.0, 0.03448406 - Q.z_6) / 0.001525592   # -0.2%  z_6 < 0.03448
        + 0.001713434 * max(0.0, Q.centroid_offset - 0.02076709) / 0.004751537   # +0.2%  centroid_offset > 0.02077
        + 0.001691166 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.2%  sum_pt > 988.4
        + 0.001179087 * max(0.0, 6.327379 - Q.log_sum_pt) * max(0.0, 0.08051087 - Q.z_6) / 0.0001292892   # +0.1%  log_sum_pt < 6.327 and z_6 < 0.08051
        + 0.001064983 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 19.46875 - Q.pt_6) / 0.002700092   # +0.1%  lam1 < 0.012 and pt_6 < 19.47
        + 0.0005073775 * max(0.0, 0.02160287 - Q.z_6) * max(0.0, 6.842717 - Q.log_sum_pt) / 4.457061e-06   # +0.1%  z_6 < 0.0216 and log_sum_pt < 6.843
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 36.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 36.57387 * (0.2790053
        - 0.2034653 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -20.3%  sum_zz_dr2 < 0.008169
        - 0.08184253 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -8.2%  sum_z_dr2 > 0.00752
        - 0.0648667 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -6.5%  tau1 < 0.1136
        + 0.06346068 * max(0.0, Q.sum_z_dr2 - 0.01323868) / 0.001442684   # +6.3%  sum_z_dr2 > 0.01324
        - 0.06340523 * max(0.0, Q.sum_z_dr2 - 0.004372139) / 0.003638805   # -6.3%  sum_z_dr2 > 0.004372
        + 0.06221404 * max(0.0, 0.1416226 - Q.tau1) / 0.08190733   # +6.2%  tau1 < 0.1416
        + 0.06067179 * max(0.0, 0.00718279 - Q.sum_zz_dr2) / 0.003528787   # +6.1%  sum_zz_dr2 < 0.007183
        + 0.05995369 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +6.0%  tau1 < 0.09539
        + 0.0479556 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +4.8%  sj2_dr < 0.1592
        - 0.04750407 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -4.8%  sum_z_dr < 0.08724
        - 0.04447758 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -4.4%  sj2_dr < 0.1873
        - 0.03467323 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # -3.5%  sum_zz_dr2 < 0.01166
        - 0.03015605 * max(0.0, 0.005590289 - Q.lam1_plus_lam2) / 0.002229311   # -3.0%  lam1_plus_lam2 < 0.00559
        - 0.02486155 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # -2.5%  centroid_offset < 0.03776
        + 0.01596774 * max(0.0, Q.sum_z_dr2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +1.6%  sum_z_dr2 > 0.004372 and planar_flow < 0.195
        + 0.01298618 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.423044) / 0.01391616   # +1.3%  planar_flow < 0.195 and log_sum_pt > 6.423
        + 0.0115885 * max(0.0, 0.2931906 - Q.LHA) / 0.07167606   # +1.2%  LHA < 0.2932
        + 0.01140471 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # +1.1%  lam1 < 0.004184
        - 0.01095049 * max(0.0, 0.1950135 - Q.planar_flow) / 0.07573236   # -1.1%  planar_flow < 0.195
        + 0.01039512 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.0%  tau1 < 0.05357
        - 0.009478566 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # -0.9%  sum_z_dr2 > 0.008678
        + 0.008337645 * max(0.0, 0.005284669 - Q.sum_zz_dr2) / 0.00223735   # +0.8%  sum_zz_dr2 < 0.005285
        - 0.005023899 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 0.05240025 - Q.z_7) / 0.000631283   # -0.5%  planar_flow < 0.195 and z_7 < 0.0524
        - 0.00456505 * max(0.0, 0.01165737 - Q.sum_zz_dr2) * max(0.0, 48.71875 - Q.pt_7) / 0.1103404   # -0.5%  sum_zz_dr2 < 0.01166 and pt_7 < 48.72
        + 0.002272547 * max(0.0, 0.0005611231 - Q.sum_z_dr2) / 9.465572e-05   # +0.2%  sum_z_dr2 < 0.0005611
        + 0.002064916 * max(0.0, 0.005590289 - Q.lam1_plus_lam2) * max(0.0, -0.00469376 - Q.mean_phi) / 4.24156e-06   # +0.2%  lam1_plus_lam2 < 0.00559 and mean_phi < -0.004694
        + 0.001797602 * max(0.0, 0.005590289 - Q.lam1_plus_lam2) * max(0.0, -0.006779839 - Q.mean_eta) / 3.381615e-06   # +0.2%  lam1_plus_lam2 < 0.00559 and mean_eta < -0.00678
        + 0.001627856 * max(0.0, 0.005590289 - Q.lam1_plus_lam2) * max(0.0, Q.mean_eta - 0.009367547) / 2.50486e-06   # +0.2%  lam1_plus_lam2 < 0.00559 and mean_eta > 0.009368
        + 0.001442178 * max(0.0, 0.005590289 - Q.lam1_plus_lam2) * max(0.0, Q.mean_phi - 0.009050008) / 2.537131e-06   # +0.1%  lam1_plus_lam2 < 0.00559 and mean_phi > 0.00905
        - 0.0003565697 * max(0.0, Q.sum_z_dr2 - 0.004372139) * max(0.0, Q.sj3_z1 - 0.8740683) / 7.896779e-07   # -0.0%  sum_z_dr2 > 0.004372 and sj3_z1 > 0.8741
        + 0.0002324225 * max(0.0, Q.sum_z_dr2 - 0.008678045) * max(0.0, Q.sj3_z1 - 0.8740683) / 2.104095e-07   # +0.0%  sum_z_dr2 > 0.008678 and sj3_z1 > 0.8741
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 19.76;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.76261 * (0.003168565
        + 0.07938639 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.002635418 - Q.sum_z_dr2) / 3.338443e-05   # +7.9%  tau1 < 0.05357 and sum_z_dr2 < 0.002635
        - 0.06834796 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, 0.005590289 - Q.lam1_plus_lam2) / 7.638845e-05   # -6.8%  sum_z_dr < 0.05465 and lam1_plus_lam2 < 0.00559
        - 0.05928287 * max(0.0, 0.05464922 - Q.sum_z_dr) / 0.01532978   # -5.9%  sum_z_dr < 0.05465
        - 0.05174145 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -5.2%  sj3_dr_max < 0.1426
        - 0.04963026 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, 0.002635418 - Q.lam1_plus_lam2) / 3.204761e-05   # -5.0%  sum_z_dr < 0.05465 and lam1_plus_lam2 < 0.002635
        - 0.04846986 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, 0.20552 - Q.z_dr_0p2_0p4) / 0.005135396   # -4.8%  LHA < 0.1967 and z_dr_0p2_0p4 < 0.2055
        + 0.04386175 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, 0.0001947983 - Q.lam2) / 2.397672e-06   # +4.4%  sum_z_dr < 0.05465 and lam2 < 0.0001948
        + 0.04014766 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +4.0%  sum_z_dr2 < 0.006679 and centroid_offset < 0.02355
        + 0.0395546 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +4.0%  tau1 < 0.05357
        - 0.03808088 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.0003061234 - Q.lam2) / 4.443043e-06   # -3.8%  tau1 < 0.05357 and lam2 < 0.0003061
        + 0.03379059 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.0003061234 - Q.lam2) / 1.676724e-05   # +3.4%  sj3_dr_max < 0.1986 and lam2 < 0.0003061
        + 0.03238605 * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.00293624   # +3.2%  sum_z_dr2 < 0.006679
        + 0.03146289 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +3.1%  sum_pt_top5 > 687.4
        - 0.03118472 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -3.1%  log_sum_pt > 6.701
        + 0.03034035 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +3.0%  sj3_dr_max < 0.107
        - 0.026755 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, 0.0003061234 - Q.lam2) / 6.518054e-06   # -2.7%  sj3_dr_max < 0.107 and lam2 < 0.0003061
        + 0.02460576 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +2.5%  sum_z_dr2 < 0.00502
        - 0.02312988 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.01002369 - Q.sum_z_dr2_top3) / 0.3268181   # -2.3%  sum_pt_top5 > 687.4 and sum_z_dr2_top3 < 0.01002
        + 0.0223583 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.0001053955   # +2.2%  sum_z_dr2 < 0.00502 and z_dr_0p2_0p4 < 0.05644
        - 0.02195775 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -2.2%  sum_z_dr2 < 0.00502 and centroid_offset > 0.00679
        + 0.01944607 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # +1.9%  LHA < 0.1967
        + 0.01759612 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +1.8%  sj3_dr_max < 0.1986
        + 0.01747588 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 5.290325e-05   # +1.7%  log_sum_pt > 6.701 and sum_z_dr2_top3 < 0.002152
        - 0.01520558 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, 0.0001947983 - Q.lam2) / 4.231311e-06   # -1.5%  LHA < 0.1967 and lam2 < 0.0001948
        - 0.01511588 * max(0.0, Q.z_dr_0_0p05 - 0.9008535) * max(0.0, 0.0001947983 - Q.lam2) / 5.136077e-06   # -1.5%  z_dr_0_0p05 > 0.9009 and lam2 < 0.0001948
        - 0.01337006 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 43.5 - Q.pt_7) / 0.02255472   # -1.3%  sum_z_dr2 < 0.00502 and pt_7 < 43.5
        + 0.01294245 * max(0.0, Q.z_dr_0_0p05 - 0.9008535) / 0.03206403   # +1.3%  z_dr_0_0p05 > 0.9009
        - 0.01148072 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, Q.lam1_plus_lam2 - 0.002635418) / 2.147931e-05   # -1.1%  sj3_dr_max < 0.1986 and lam1_plus_lam2 > 0.002635
        + 0.009661129 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 43.5 - Q.pt_7) / 0.2124887   # +1.0%  tau1 < 0.05357 and pt_7 < 43.5
        - 0.008831441 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, Q.lam1_plus_lam2 - 0.001653836) / 1.467156e-06   # -0.9%  sum_z_dr2 < 0.006679 and lam1_plus_lam2 > 0.001654
        + 0.008559623 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.pt_7 - 15.55391) / 0.4153091   # +0.9%  LHA < 0.1967 and pt_7 > 15.55
        + 0.007495009 * max(0.0, 0.0003193707 - Q.sum_z_dr2) / 4.035777e-05   # +0.7%  sum_z_dr2 < 0.0003194
        + 0.00700101 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, Q.sum_z_dr2_top3 - 0.001155057) / 3.499346e-05   # +0.7%  sj3_dr_max < 0.1986 and sum_z_dr2_top3 > 0.001155
        + 0.006110937 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 0.4007947 - Q.planar_flow) / 0.0003918103   # +0.6%  sum_z_dr2 < 0.006679 and planar_flow < 0.4008
        + 0.006031786 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.002635418 - Q.lam1_plus_lam2) / 5.489096e-05   # +0.6%  log_sum_pt > 6.701 and lam1_plus_lam2 < 0.002635
        + 0.005728396 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # +0.6%  lam1 < 0.001504
        - 0.004674907 * max(0.0, 0.01357518 - Q.tau1) / 0.001531397   # -0.5%  tau1 < 0.01358
        - 0.004640907 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.0009641429 - Q.sum_z_dr2) / 0.01750133   # -0.5%  sum_pt_top5 > 687.4 and sum_z_dr2 < 0.0009641
        - 0.003770405 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, Q.pt_7 - 31.85938) / 0.01797499   # -0.4%  sum_z_dr2 < 0.006679 and pt_7 > 31.86
        + 0.003568615 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.psi_0p2 - 0.9435576) / 0.001795607   # +0.4%  log_sum_pt > 6.701 and psi_0p2 > 0.9436
        - 0.002622385 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 6.502799 - Q.log_sum_pt) / 8.085113e-05   # -0.3%  sum_z_dr2 < 0.00502 and log_sum_pt < 6.503
        - 0.002195739 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.pt_7 - 15.55391) / 0.4785904   # -0.2%  log_sum_pt > 6.701 and pt_7 > 15.55
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 25.11;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.11035 * (-0.1885255
        + 0.127184 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) / 0.00293624   # +12.7%  lam1_plus_lam2 < 0.006679
        + 0.1020202 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +10.2%  sum_z_dr2 < 0.00502
        + 0.08499332 * max(0.0, 6.896095 - Q.log_sum_pt) / 0.3571984   # +8.5%  log_sum_pt < 6.896
        - 0.07646945 * max(0.0, 0.004641801 - Q.sum_zz_dr2) / 0.001869767   # -7.6%  sum_zz_dr2 < 0.004642
        + 0.07484095 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # +7.5%  sum_z_dr2 < 0.003563
        + 0.06542582 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +6.5%  sj3_dr_max < 0.1986
        - 0.05866838 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -5.9%  sj3_dr_max < 0.1426
        - 0.0517534 * max(0.0, 0.06663269 - Q.sum_z_dr) / 0.02186049   # -5.2%  sum_z_dr < 0.06663
        - 0.04511738 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # -4.5%  sum_zz_dr2 < 0.003013
        + 0.0342629 * max(0.0, 0.02355416 - Q.centroid_offset) / 0.0103615   # +3.4%  centroid_offset < 0.02355
        + 0.02913074 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.06729223 - Q.C2) / 0.0003300761   # +2.9%  centroid_offset < 0.01838 and C2 < 0.06729
        - 0.02646327 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -2.6%  sj2_dr < 0.1295
        + 0.02397682 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 0.0234802   # +2.4%  centroid_offset < 0.01838 and n_dr_0p05_0p1 < 5
        - 0.02358354 * max(0.0, 0.05464922 - Q.sum_z_dr) / 0.01532978   # -2.4%  sum_z_dr < 0.05465
        + 0.01844562 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +1.8%  max_dr < 0.1118
        + 0.01769661 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +1.8%  lam2 > 0.001131
        - 0.01769531 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -1.8%  centroid_offset < 0.01838
        + 0.01711317 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.01364517 - Q.tau3) / 6.201347e-05   # +1.7%  centroid_offset < 0.01838 and tau3 < 0.01365
        + 0.0154935 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +1.5%  lam2 < 0.0001948
        + 0.0136931 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +1.4%  LHA < 0.2161
        - 0.01074516 * max(0.0, 0.001101266 - Q.sum_zz_dr2) / 0.000308004   # -1.1%  sum_zz_dr2 < 0.001101
        - 0.01057395 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, 0.380911 - Q.D2_b2) / 0.001264948   # -1.1%  centroid_offset < 0.02355 and D2_b2 < 0.3809
        + 0.009063855 * max(0.0, 0.02355416 - Q.centroid_offset) * max(0.0, Q.tau21_b2 - 0.004811143) / 0.002193894   # +0.9%  centroid_offset < 0.02355 and tau21_b2 > 0.004811
        - 0.008356408 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, Q.planar_flow - 0.1115136) / 0.07272605   # -0.8%  log_sum_pt < 6.896 and planar_flow > 0.1115
        - 0.007268015 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, Q.planar_flow - 0.2534037) / 0.0001586905   # -0.7%  lam2 > 0.001131 and planar_flow > 0.2534
        + 0.005216409 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, 0.05753583 - Q.z_5) / 0.0005902159   # +0.5%  log_sum_pt < 6.896 and z_5 < 0.05754
        - 0.004456788 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -0.4%  sj3_dr_max > 0.2623
        + 0.004030566 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.4%  n_dr_0p2_0p4 > 1
        - 0.003598946 * max(0.0, 0.004918231 - Q.zdr_0) / 0.0006323791   # -0.4%  zdr_0 < 0.004918
        + 0.003330764 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +0.3%  e3 > 0.0001869
        - 0.002290341 * max(0.0, 6.896095 - Q.log_sum_pt) * max(0.0, Q.z_dr_0p2_0p4 - 0.0) / 0.0166461   # -0.2%  log_sum_pt < 6.896 and z_dr_0p2_0p4 > 0
        - 0.002190926 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) * max(0.0, -0.006779839 - Q.mean_eta) / 4.770336e-06   # -0.2%  lam1_plus_lam2 < 0.006679 and mean_eta < -0.00678
        - 0.001320981 * max(0.0, Q.lam1_plus_lam2 - 0.02530566) / 0.0002851417   # -0.1%  lam1_plus_lam2 > 0.02531
        + 0.001246777 * max(0.0, 0.1294903 - Q.sj2_dr) * max(0.0, -0.006779839 - Q.mean_eta) / 5.903935e-05   # +0.1%  sj2_dr < 0.1295 and mean_eta < -0.00678
        - 0.001200812 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.04435703 - Q.M2) / 5.61399e-05   # -0.1%  centroid_offset < 0.01838 and M2 < 0.04436
        + 0.001081747 * max(0.0, Q.e3 - 0.0001869378) * max(0.0, 33.21875 - Q.pt_7) / 8.091322e-05   # +0.1%  e3 > 0.0001869 and pt_7 < 33.22
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 9.781;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.780896 * (0.2613686
        + 0.1271404 * Q.lam1 / 0.00591636   # +12.7%  lam1
        - 0.1263289 * max(0.0, Q.LHA - 0.1967397) / 0.07109091   # -12.6%  LHA > 0.1967
        - 0.09940868 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # -9.9%  lam1 < 0.006507
        + 0.08251652 * max(0.0, Q.tau1 - 0.04369778) / 0.031989   # +8.3%  tau1 > 0.0437
        - 0.05842869 * max(0.0, 0.003377388 - Q.lam1) / 0.00113338   # -5.8%  lam1 < 0.003377
        + 0.05842336 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # +5.8%  sj2_dr > 0.1295
        - 0.05705824 * max(0.0, Q.lam1 - 0.00483998) / 0.002931468   # -5.7%  lam1 > 0.00484
        + 0.05386148 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # +5.4%  sum_z_dr2 > 0.00752
        - 0.04295547 * max(0.0, Q.sj2_dr - 0.1682655) * max(0.0, 0.6947818 - Q.planar_flow) / 0.0166918   # -4.3%  sj2_dr > 0.1683 and planar_flow < 0.6948
        - 0.04055148 * max(0.0, Q.sj2_dr - 0.1682655) / 0.03492091   # -4.1%  sj2_dr > 0.1683
        + 0.03792139 * max(0.0, Q.lam2 - 0.0001947983) / 0.0004461515   # +3.8%  lam2 > 0.0001948
        + 0.03721589 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.158545   # +3.7%  n_dr_0p05_0p1 < 5
        + 0.03123139 * max(0.0, 0.002915531 - Q.sum_z_dr2_top3) / 0.001178811   # +3.1%  sum_z_dr2_top3 < 0.002916
        - 0.03080318 * max(0.0, 48.03125 - Q.pt_6) / 9.943776   # -3.1%  pt_6 < 48.03
        - 0.02174393 * max(0.0, Q.LHA - 0.3127275) * max(0.0, 0.6947818 - Q.planar_flow) / 0.005887138   # -2.2%  LHA > 0.3127 and planar_flow < 0.6948
        + 0.01402993 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +1.4%  n_dr_0p2_0p4 > 1
        - 0.0109118 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.eccentricity - 0.4829602) / 8.513077e-05   # -1.1%  lam2 > 0.0001948 and eccentricity > 0.483
        + 0.009529644 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # +1.0%  e3 > 8.148e-05
        - 0.009317924 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 4.721224 - Q.D2_b2) / 0.5862342   # -0.9%  n_dr_0p2_0p4 > 1 and D2_b2 < 4.721
        - 0.008740333 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, Q.planar_flow - 0.06126205) / 0.0002819968   # -0.9%  lam2 > 0.0001948 and planar_flow > 0.06126
        + 0.008732925 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +0.9%  sj2_dr > 0.3004
        - 0.008510457 * max(0.0, Q.LHA - 0.3127275) / 0.01533444   # -0.9%  LHA > 0.3127
        - 0.007349719 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -0.7%  lam2 > 0.003408
        + 0.006742193 * max(0.0, Q.centroid_offset - 0.02355416) / 0.003991673   # +0.7%  centroid_offset > 0.02355
        - 0.006626668 * max(0.0, Q.lam2 - 0.0001947983) * max(0.0, 6.538559 - Q.log_sum_pt) / 0.0001231869   # -0.7%  lam2 > 0.0001948 and log_sum_pt < 6.539
        - 0.003919429 * max(0.0, Q.sum_z_dr2 - 0.02530566) / 0.0002851418   # -0.4%  sum_z_dr2 > 0.02531
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 34.45;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.44982 * (-0.0650879
        + 0.2134047 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +21.3%  lam1_plus_lam2 < 0.008678
        + 0.123507 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +12.4%  lam1_plus_lam2 < 0.01324
        - 0.1226824 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # -12.3%  LHA < 0.3127
        - 0.1070417 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -10.7%  lam1 < 0.008376
        + 0.09271427 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +9.3%  LHA < 0.3033
        - 0.04604034 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -4.6%  sum_zz_dr2 < 0.008169
        - 0.04395133 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # -4.4%  sum_z_dr < 0.0717
        + 0.03578184 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +3.6%  centroid_offset < 0.03776
        - 0.02549236 * max(0.0, 0.006390125 - Q.sum_zz_dr2) / 0.002953508   # -2.5%  sum_zz_dr2 < 0.00639
        + 0.02472391 * max(0.0, Q.sj2_dr - 0.09419022) / 0.07816795   # +2.5%  sj2_dr > 0.09419
        + 0.02148158 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +2.1%  planar_flow < 0.2534
        - 0.02124537 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # -2.1%  centroid_offset > 0.0499 and D2 < 2.844
        - 0.01211573 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # -1.2%  sum_z_dr2 < 0.003563
        + 0.01190428 * max(0.0, 0.03459477 - Q.tau1) * max(0.0, 0.002915531 - Q.sum_z_dr2_top3) / 2.198359e-05   # +1.2%  tau1 < 0.03459 and sum_z_dr2_top3 < 0.002916
        - 0.01181473 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -1.2%  sj2_dr > 0.1779
        + 0.01068291 * max(0.0, 0.09538712 - Q.tau1) * max(0.0, 6.0 - Q.n_dr_0p05_0p1) / 0.2202217   # +1.1%  tau1 < 0.09539 and n_dr_0p05_0p1 < 6
        - 0.01053607 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.08543881 - Q.sj3_dr_min) / 0.0002784225   # -1.1%  centroid_offset < 0.01438 and sj3_dr_min < 0.08544
        - 0.009608637 * max(0.0, 0.03459477 - Q.tau1) / 0.008474553   # -1.0%  tau1 < 0.03459
        - 0.009460224 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 53.4375 - Q.pt_7) / 0.4342437   # -0.9%  centroid_offset < 0.03776 and pt_7 < 53.44
        - 0.00703722 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.7%  planar_flow < 0.2534 and sum_pt < 840
        + 0.005739044 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.05932655 - Q.tau21_b2) / 7.260318e-05   # +0.6%  centroid_offset < 0.01438 and tau21_b2 < 0.05933
        - 0.004662016 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001130645 - Q.lam2) / 1.843299e-05   # -0.5%  sj2_dr > 0.1779 and lam2 < 0.001131
        - 0.004633001 * max(0.0, 0.001570267 - Q.mean_phi2) / 0.0006563898   # -0.5%  mean_phi2 < 0.00157
        + 0.004562151 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +0.5%  centroid_offset > 0.0499
        + 0.004470301 * max(0.0, Q.log_sum_pt - 6.572938) / 0.08631512   # +0.4%  log_sum_pt > 6.573
        - 0.00373047 * max(0.0, 0.02689598 - Q.sum_z_dr) / 0.004330137   # -0.4%  sum_z_dr < 0.0269
        - 0.003008269 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 37.15625 - Q.pt_7) / 0.5537514   # -0.3%  planar_flow < 0.2534 and pt_7 < 37.16
        + 0.002364192 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.2%  sj2_dr > 0.2688
        - 0.002109286 * max(0.0, Q.log_sum_pt - 6.572938) * max(0.0, 1.002471 - Q.D2) / 0.01268009   # -0.2%  log_sum_pt > 6.573 and D2 < 1.002
        - 0.001949278 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 1.050302e-05 - Q.e3) / 1.771164e-07   # -0.2%  planar_flow < 0.2534 and e3 < 1.05e-05
        - 0.001545265 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.M3 - 0.03843235) / 2.574583e-05   # -0.2%  centroid_offset > 0.0499 and M3 > 0.03843
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.5102;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5102346 * (-0.4467616
        + 0.3863686 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # +38.6%  sum_z_dr2 > 0.01883
        - 0.2454196 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -24.5%  e2 > 0.06344
        - 0.146714 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, 53.4375 - Q.pt_7) / 0.01439677   # -14.7%  sum_z_dr2 > 0.01883 and pt_7 < 53.44
        + 0.08909684 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # +8.9%  lam1_plus_lam2 > 0.01324
        + 0.0718283 * max(0.0, Q.sum_z_dr2 - 0.02530566) / 0.0002851418   # +7.2%  sum_z_dr2 > 0.02531
        + 0.06057259 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) * max(0.0, Q.log_sum_pt - 6.327379) / 8.505384e-05   # +6.1%  lam1_plus_lam2 > 0.01324 and log_sum_pt > 6.327
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 20.03;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.03431 * (0.02323757
        + 0.428115 * max(0.0, 0.1484084 - Q.sum_z_dr) / 0.0905999   # +42.8%  sum_z_dr < 0.1484
        - 0.07742506 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -7.7%  sum_z_dr < 0.1484 and log_sum_pt < 6.804
        - 0.06947043 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # -6.9%  tau1 < 0.1136
        - 0.06568731 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -6.6%  e2 < 0.08001
        - 0.05370282 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 1.399607e-06   # -5.4%  e3 < 8.148e-05 and centroid_offset < 0.03776
        + 0.04692055 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +4.7%  lam1_plus_lam2 < 0.01324
        + 0.03894375 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +3.9%  lam1 < 0.01643
        - 0.02744609 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.03585464 - Q.tau2) / 0.002456165   # -2.7%  sum_z_dr < 0.1484 and tau2 < 0.03585
        - 0.02617247 * max(0.0, 50.25 - Q.pt_6) / 11.66482   # -2.6%  pt_6 < 50.25
        - 0.02497126 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -2.5%  sum_z_dr < 0.1484 and pt_7 < 38.53
        + 0.02453421 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 40.04062 - Q.pt_7) / 652.1541   # +2.5%  sum_pt_top5 > 687.4 and pt_7 < 40.04
        - 0.02233578 * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.09151624   # -2.2%  sj3_dr_min < 0.1278
        - 0.02165679 * max(0.0, Q.sum_pt_top5 - 482.2812) / 137.9506   # -2.2%  sum_pt_top5 > 482.3
        - 0.01473776 * max(0.0, Q.pt_7 - 31.85938) / 5.945918   # -1.5%  pt_7 > 31.86
        - 0.01174779 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.2%  sj3_dr_max > 0.2337
        + 0.01154386 * max(0.0, Q.z_7 - 0.0586137) / 0.00587069   # +1.2%  z_7 > 0.05861
        - 0.007961434 * max(0.0, 0.01643375 - Q.lam1) * max(0.0, 25.57812 - Q.pt_7) / 0.01870322   # -0.8%  lam1 < 0.01643 and pt_7 < 25.58
        + 0.007404571 * max(0.0, 0.08000524 - Q.e2) * max(0.0, 60.03125 - Q.pt_4) / 0.4322666   # +0.7%  e2 < 0.08001 and pt_4 < 60.03
        - 0.006904976 * max(0.0, 27.57812 - Q.pt_6) / 0.9875435   # -0.7%  pt_6 < 27.58
        + 0.006136078 * max(0.0, 0.0294431 - Q.M2) / 0.002929467   # +0.6%  M2 < 0.02944
        + 0.003578062 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.08525808 - Q.dr_2) / 2.250996   # +0.4%  sum_pt_top5 > 687.4 and dr_2 < 0.08526
        - 0.002603928 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # -0.3%  pt_5 < 24.58
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 36.98;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 36.98396 * (-0.01825277
        - 0.1172227 * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 0.001433269   # -11.7%  sum_zz_dr2 > 0.01166
        - 0.08700779 * max(0.0, Q.tau1 - 0.03459477) / 0.03725748   # -8.7%  tau1 > 0.03459
        - 0.07931883 * max(0.0, Q.sum_z_dr2 - 0.006679471) / 0.002702003   # -7.9%  sum_z_dr2 > 0.006679
        + 0.07548189 * max(0.0, Q.sum_z_dr - 0.04081947) / 0.0270611   # +7.5%  sum_z_dr > 0.04082
        - 0.07482764 * max(0.0, 0.233678 - Q.sj3_dr_max) / 0.09057682   # -7.5%  sj3_dr_max < 0.2337
        - 0.06918602 * max(0.0, Q.sum_z_dr - 0.1019409) / 0.005400003   # -6.9%  sum_z_dr > 0.1019
        - 0.05886038 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # -5.9%  lam1_plus_lam2 > 0.01324
        + 0.0548019 * max(0.0, Q.sum_zz_dr2 - 0.00718279) / 0.002233568   # +5.5%  sum_zz_dr2 > 0.007183
        - 0.049851 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -5.0%  sum_z_dr2 > 0.00752
        + 0.04940606 * max(0.0, Q.sum_z_dr - 0.02689598) / 0.03613842   # +4.9%  sum_z_dr > 0.0269
        - 0.04397297 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -4.4%  centroid_offset > 0.0499
        + 0.04105229 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # +4.1%  lam1_plus_lam2 > 0.00559
        - 0.03299812 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # -3.3%  sum_z_dr > 0.08724
        - 0.03152354 * max(0.0, Q.sum_z_dr2 - 0.008678045) * max(0.0, Q.log_sum_pt - 6.267538) / 0.0002118474   # -3.2%  sum_z_dr2 > 0.008678 and log_sum_pt > 6.268
        + 0.02352441 * max(0.0, Q.sum_zz_dr2 - 0.006390125) / 0.002450953   # +2.4%  sum_zz_dr2 > 0.00639
        - 0.01896784 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) * max(0.0, Q.sum_pt - 488.9312) / 0.1069057   # -1.9%  lam1_plus_lam2 > 0.01324 and sum_pt > 488.9
        + 0.0172752 * max(0.0, Q.sum_zz_dr2 - 0.002074109) / 0.004484017   # +1.7%  sum_zz_dr2 > 0.002074
        + 0.01505966 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +1.5%  e2 > 0.05028
        + 0.01237991 * max(0.0, Q.LHA - 0.2160559) / 0.05893777   # +1.2%  LHA > 0.2161
        - 0.009779526 * max(0.0, Q.centroid_offset - 0.02355416) / 0.003991673   # -1.0%  centroid_offset > 0.02355
        + 0.008064955 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +0.8%  LHA > 0.3033
        - 0.007851146 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # -0.8%  sj2_dr < 0.1592
        + 0.006110651 * max(0.0, Q.tau1 - 0.09538712) / 0.01057268   # +0.6%  tau1 > 0.09539
        + 0.00584486 * max(0.0, Q.sum_zz_dr2 - 0.002074109) * max(0.0, Q.log_sum_pt - 6.267538) / 0.0007382066   # +0.6%  sum_zz_dr2 > 0.002074 and log_sum_pt > 6.268
        - 0.00557138 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -0.6%  e2 > 0.03556
        + 0.002704262 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # +0.3%  sum_z_dr2 > 0.008678
        + 0.001355028 * max(0.0, Q.sum_zz_dr2 - 0.01716248) / 0.0007604864   # +0.1%  sum_zz_dr2 > 0.01716
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 35.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.23315 * (-0.05064712
        + 0.1449944 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +14.5%  sum_z_dr2 < 0.01324
        - 0.09835834 * max(0.0, 0.007520088 - Q.sum_z_dr2) / 0.00354499   # -9.8%  sum_z_dr2 < 0.00752
        + 0.07771256 * max(0.0, 0.006390125 - Q.sum_zz_dr2) / 0.002953508   # +7.8%  sum_zz_dr2 < 0.00639
        - 0.07163773 * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.00293624   # -7.2%  sum_z_dr2 < 0.006679
        - 0.0714208 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # -7.1%  sum_zz_dr2 < 0.01166
        - 0.06615013 * max(0.0, 0.06344108 - Q.e2) / 0.03657078   # -6.6%  e2 < 0.06344
        + 0.06474322 * max(0.0, 0.0811449 - Q.tau1) / 0.03266098   # +6.5%  tau1 < 0.08114
        + 0.0509105 * max(0.0, 0.01716248 - Q.sum_zz_dr2) / 0.0120354   # +5.1%  sum_zz_dr2 < 0.01716
        + 0.03841837 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +3.8%  N2 < 0.2233
        + 0.03383267 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +3.4%  e2 < 0.04111
        - 0.02995833 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # -3.0%  lam2 < 0.003408
        - 0.0269451 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.0001197728   # -2.7%  N2 < 0.2233 and sum_z_dr2 > 0.008678
        - 0.01967738 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # -2.0%  sum_z_dr2_top3 < 0.002152
        - 0.01754639 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 8.147744e-05 - Q.e3) / 2.656078e-06   # -1.8%  N2 < 0.2233 and e3 < 8.148e-05
        - 0.0162967 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -1.6%  sj3_dr_max > 0.2623
        - 0.01541626 * max(0.0, 0.002635418 - Q.lam1_plus_lam2) / 0.0007941347   # -1.5%  lam1_plus_lam2 < 0.002635
        - 0.0148745 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1872617 - Q.sj2_dr) / 0.0007896743   # -1.5%  N2 < 0.2233 and sj2_dr < 0.1873
        - 0.01336588 * max(0.0, 0.005834489 - Q.sum_zz_dr2) / 0.002579453   # -1.3%  sum_zz_dr2 < 0.005834
        + 0.01332135 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +1.3%  sj3_dr_max > 0.1879
        + 0.0126206 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sum_pt_top5 - 402.625) / 15.59919   # +1.3%  planar_flow < 0.195 and sum_pt_top5 > 402.6
        + 0.01210831 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # +1.2%  sj3_dr_max > 0.2337
        + 0.01082754 * max(0.0, 0.08082334 - Q.dr_0) / 0.0377444   # +1.1%  dr_0 < 0.08082
        - 0.010602 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -1.1%  lam1 < 0.005433
        + 0.01024148 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.2931906) / 0.001536651   # +1.0%  N2 < 0.2233 and LHA > 0.2932
        + 0.008625411 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3255822) / 0.0007869707   # +0.9%  N2 < 0.2233 and LHA > 0.3256
        + 0.005943746 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +0.6%  lam1 < 0.005954
        - 0.005701453 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 0.07148865 - Q.z_7) / 0.001619322   # -0.6%  planar_flow < 0.195 and z_7 < 0.07149
        + 0.005213573 * max(0.0, 0.1515405 - Q.z_dr_0_0p05) / 0.0478325   # +0.5%  z_dr_0_0p05 < 0.1515
        - 0.004882398 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.0114178   # -0.5%  N2 < 0.2233 and z_dr_0p05_0p1 < 0.5883
        - 0.00479963 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, Q.pt_entropy - 1.797618) / 0.0002596149   # -0.5%  sum_z_dr2 < 0.006679 and pt_entropy > 1.798
        + 0.004590561 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1591713 - Q.sj2_dr) / 0.0002571871   # +0.5%  N2 < 0.2233 and sj2_dr < 0.1592
        - 0.004385665 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.mean_phi - -0.01753483) / 0.001413063   # -0.4%  planar_flow < 0.195 and mean_phi > -0.01753
        + 0.003649628 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +0.4%  N2 < 0.2233 and sum_pt_top5 > 430.8
        - 0.0024111 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) * max(0.0, Q.eccentricity - 0.9598562) / 4.406774e-06   # -0.2%  sum_z_dr2_top3 < 0.002152 and eccentricity > 0.9599
        - 0.002318428 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.4235925) / 7.321856e-05   # -0.2%  N2 < 0.2233 and LHA > 0.4236
        - 0.001465923 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, Q.dr12 - 0.1587481) / 1.436546e-06   # -0.1%  sum_z_dr2 < 0.006679 and dr12 > 0.1587
        - 0.001396724 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 25.57812 - Q.pt_7) / 0.037942   # -0.1%  N2 < 0.2233 and pt_7 < 25.58
        + 0.001211168 * max(0.0, 0.006390125 - Q.sum_zz_dr2) * max(0.0, Q.dr12 - 0.1587481) / 1.354517e-06   # +0.1%  sum_zz_dr2 < 0.00639 and dr12 > 0.1587
        - 0.000783222 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.LHA - 0.3855647) / 0.0002330351   # -0.1%  N2 < 0.2233 and LHA > 0.3856
        - 0.0006407725 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.04990367) / 8.981184e-07   # -0.1%  sum_z_dr2 < 0.01324 and centroid_offset > 0.0499
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.8983233193277311, 0.7896720588235294, 2.096946113445378, 0.9683955882352941, 1.280420168067227, 1.7977728991596638, 1.070792857142857, 1.9430134453781513, 0.32466848739495796, 3.105170588235294, 1.9945779411764706, 2.1488621848739498, 0.09124768907563026, 3.3138682773109243, 0.40240262605042015, 0.38968508403361346]
T = [2.377774912191439, 1.4205506630777311, 3.364074496126576, 2.57394245831145, 2.88478167345063]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +38%, n9 +22%, n5 -14%, n1 +13%, n0 -6%, n6 +5% ...
            + 0.378939 * h[2] / H_AVG[2]
            + 0.224454 * h[9] / H_AVG[9]
            - 0.1417638 * h[5] / H_AVG[5]
            + 0.1297287 * h[1] / H_AVG[1]
            - 0.05903125 * h[0] / H_AVG[0]
            + 0.04925528 * h[6] / H_AVG[6]
            - 0.01682797 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -18%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5550117 * h[9] / H_AVG[9]
            - 0.175511 * h[10] / H_AVG[10]
            + 0.0942234 * h[6] / H_AVG[6]
            - 0.08450201 * h[4] / H_AVG[4]
            + 0.05932249 * h[5] / H_AVG[5]
            + 0.01714498 * h[15] / H_AVG[15]
            + 0.01428445 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +13%, n6 -10%, n0 +9%, n14 -9% ...
            + 0.2395379 * h[11] / H_AVG[11]
            - 0.1439319 * h[3] / H_AVG[3]
            + 0.1263451 * h[7] / H_AVG[7]
            - 0.09946949 * h[6] / H_AVG[6]
            + 0.09179304 * h[0] / H_AVG[0]
            - 0.08971322 * h[14] / H_AVG[14]
            - 0.0796381 * h[15] / H_AVG[15]
            + 0.06926314 * h[13] / H_AVG[13]
            - 0.02884496 * h[9] / H_AVG[9]
            - 0.02412762 * h[8] / H_AVG[8]
            - 0.007335525 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -21%, n6 -16%, n13 +7%, n14 +6%, n4 +4% ...
            + 0.3538492 * h[7] / H_AVG[7]
            - 0.2116296 * h[3] / H_AVG[3]
            - 0.1560048 * h[6] / H_AVG[6]
            + 0.0704084 * h[13] / H_AVG[13]
            + 0.0586264 * h[14] / H_AVG[14]
            + 0.03886366 * h[4] / H_AVG[4]
            + 0.03834935 * h[1] / H_AVG[1]
            - 0.03769959 * h[9] / H_AVG[9]
            - 0.02365565 * h[15] / H_AVG[15]
            + 0.0109133 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +26%, n5 -16%, n4 +6%, n8 +2%, n3 +2% ...
            - 0.4666762 * h[13] / H_AVG[13]
            + 0.2592802 * h[10] / H_AVG[10]
            - 0.155798 * h[5] / H_AVG[5]
            + 0.05548168 * h[4] / H_AVG[4]
            + 0.02110224 * h[8] / H_AVG[8]
            + 0.02098069 * h[3] / H_AVG[3]
            - 0.01581535 * h[12] / H_AVG[12]
            + 0.004865637 * h[0] / H_AVG[0]
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
