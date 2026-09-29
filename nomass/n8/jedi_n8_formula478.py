"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no mass observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

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
    z = 7.371198
    if Q.planar_flow < 0.1484197:
        z += -2.012434 * Q.planar_flow + 0.2986849
    if Q.lam1_plus_lam2 < 0.003562611:
        z += -26.3447 * Q.lam1_plus_lam2 + 4.497943
    if 0.003562611 <= Q.lam1_plus_lam2 < 0.004372139:
        z += -337.9638 * Q.lam1_plus_lam2 + 5.608121
    if 0.004372139 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -959.263 * Q.lam1_plus_lam2 + 8.324527
    if Q.sum_z_dr2 < 0.005590289:
        z += 62.01184 * Q.sum_z_dr2 + 4.388741
    if 0.005590289 <= Q.sum_z_dr2 < 0.01323868:
        z += -512.3114 * Q.sum_z_dr2 + 7.599373
    if 0.01323868 <= Q.sum_z_dr2 < 0.01882765:
        z += -146.1894 * Q.sum_z_dr2 + 2.752403
    if Q.sum_pt < 739.5:
        z += 0.01180319 * Q.sum_pt - 9.180049
    if 739.5 <= Q.sum_pt < 788.4484:
        z += 0.009225868 * Q.sum_pt - 7.274121
    if Q.sum_z_dr2_top3 < 0.007929074:
        z += 85.25117 * Q.sum_z_dr2_top3 - 0.6759628
    if Q.tau1 < 0.06345984:
        z += -21.68277 * Q.tau1 + 0.9155742
    if 0.06345984 <= Q.tau1 < 0.1027642:
        z += 11.71399 * Q.tau1 - 1.203779
    z += -1.313491 * Q.log_sum_pt
    if Q.tau2 < 0.01357153:
        z += 106.4174 * Q.tau2 - 1.444247
    if Q.sd_rg < 0.06545715:
        z += -2.087341 * Q.sd_rg + 0.02851878
    if 0.06545715 <= Q.sd_rg < 0.1116471:
        z += -15.79253 * Q.sd_rg + 0.9256213
    if 0.1116471 <= Q.sd_rg < 0.1773029:
        z += 12.75694 * Q.sd_rg - 2.261843
    if Q.centroid_offset >= 0.04990367:
        z += -137.7583 * Q.centroid_offset + 6.874645
    if Q.lam1 < 0.008375572:
        z += 333.3842 * Q.lam1 - 2.792284
    if Q.sum_pt_top5 < 631.275:
        z += -0.005989503 * Q.sum_pt_top5 + 3.781024
    if Q.sum_z_dr < 0.01517359:
        z += 344.7792 * Q.sum_z_dr - 9.210269
    if 0.01517359 <= Q.sum_z_dr < 0.08723651:
        z += 55.21189 * Q.sum_z_dr - 4.816493
    if Q.sum_zz_dr2 < 0.00718279:
        z += 321.7407 * Q.sum_zz_dr2 - 2.310996
    if Q.sum_z_dr2 < 0.01882765 and Q.M3 < 0.09029177:
        z += 773.407 * (0.01882765 - Q.sum_z_dr2) * (0.09029177 - Q.M3)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.dr0_6 > 0.1755206:
        z += 16363.31 * (0.004372139 - Q.lam1_plus_lam2) * (Q.dr0_6 - 0.1755206)
    if Q.planar_flow < 0.1484197 and Q.D2_b2 < 1.129616:
        z += 9.174643 * (0.1484197 - Q.planar_flow) * (1.129616 - Q.D2_b2)
    if Q.sum_pt < 788.4484 and Q.D2_b2 < 0.380911:
        z += -0.01493173 * (788.4484 - Q.sum_pt) * (0.380911 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.01882765 and Q.centroid_offset > 0.009480685:
        z += -2131.174 * (0.01882765 - Q.sum_z_dr2) * (Q.centroid_offset - 0.009480685)
    if Q.tau1 < 0.06345984 and Q.dr0_6 > 0.1755206:
        z += -513.7778 * (0.06345984 - Q.tau1) * (Q.dr0_6 - 0.1755206)
    if Q.lam1_plus_lam2 < 0.003562611 and Q.dr0_7 > 0.1786203:
        z += 19245.86 * (0.003562611 - Q.lam1_plus_lam2) * (Q.dr0_7 - 0.1786203)
    if Q.e3 < 8.147744e-05 and Q.dr0_7 > 0.2405707:
        z += -377902.2 * (8.147744e-05 - Q.e3) * (Q.dr0_7 - 0.2405707)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.186506
    if 6.377723 <= Q.log_sum_pt < 6.572938:
        z += 6.8289 * Q.log_sum_pt - 43.55283
    if Q.log_sum_pt >= 6.572938:
        z += 18.12346 * Q.log_sum_pt - 117.7913
    if Q.sum_zz_dr2 < 0.005834489:
        z += -789.9457 * Q.sum_zz_dr2 + 4.60893
    if Q.z_7 < 0.05240025:
        z += 137.2844 * Q.z_7 - 8.143461
    if 0.05240025 <= Q.z_7 < 0.06164517:
        z += 102.7291 * Q.z_7 - 6.332753
    if Q.lam1_plus_lam2 < 0.008678045:
        z += 534.1063 * Q.lam1_plus_lam2 - 4.634998
    if 38.53125 <= Q.pt_7 < 53.4375:
        z += 0.1117451 * Q.pt_7 - 4.305678
    if Q.pt_7 >= 53.4375:
        z += 0.003991611 * Q.pt_7 + 1.452399
    if Q.centroid_offset < 0.02076709:
        z += 23.01924 * Q.centroid_offset - 0.4780426
    if Q.sum_z_dr2 < 0.00609665:
        z += 293.2003 * Q.sum_z_dr2 - 1.787539
    if Q.tau1 < 0.07283629:
        z += 15.50797 * Q.tau1 - 1.129543
    if Q.sum_z_dr < 0.1019409:
        z += 32.55195 * Q.sum_z_dr - 3.318376
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += 6618.998 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.pt_7 > 34.53125 and Q.e3 < 2.955458e-05:
        z += -1883.271 * (Q.pt_7 - 34.53125) * (2.955458e-05 - Q.e3)
    if Q.z_7 < 0.06164517 and Q.e3 < 0.0001869378:
        z += 399099.2 * (0.06164517 - Q.z_7) * (0.0001869378 - Q.e3)
    if Q.log_sum_pt > 6.572938 and Q.zdr_6 < 0.008654951:
        z += -286.6038 * (Q.log_sum_pt - 6.572938) * (0.008654951 - Q.zdr_6)
    if Q.z_7 < 0.05240025 and Q.dr0_6 > 0.003538987:
        z += -119.3173 * (0.05240025 - Q.z_7) * (Q.dr0_6 - 0.003538987)
    if Q.z_7 < 0.05240025 and Q.D2_b2 < 0.1236856:
        z += -384.0638 * (0.05240025 - Q.z_7) * (0.1236856 - Q.D2_b2)
    if Q.sj2_dr > 0.1872617 and Q.sj3_dr_min < 0.2089872:
        z += 28.05661 * (Q.sj2_dr - 0.1872617) * (0.2089872 - Q.sj3_dr_min)
    if Q.z_7 < 0.06164517 and Q.sum_z_dr2_top2 < 0.01403324:
        z += 1879.659 * (0.06164517 - Q.z_7) * (0.01403324 - Q.sum_z_dr2_top2)
    if Q.log_sum_pt > 6.377723 and Q.dr_0 < 0.08082334:
        z += -64.35999 * (Q.log_sum_pt - 6.377723) * (0.08082334 - Q.dr_0)
    if Q.lam1 < 0.008375572 and Q.dr_0 < 0.1442881:
        z += 3100.131 * (0.008375572 - Q.lam1) * (0.1442881 - Q.dr_0)
    if Q.log_sum_pt > 6.572938 and Q.sum_z_dr2_top3 < 0.007929074:
        z += -381.6423 * (Q.log_sum_pt - 6.572938) * (0.007929074 - Q.sum_z_dr2_top3)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.9289603
    if Q.pt_7 < 53.4375:
        z += 0.1262343 * Q.pt_7 - 2.124465
    if Q.pt_7 >= 53.4375:
        z += 0.08647821 * Q.pt_7
    if Q.log_sum_pt < 6.605974:
        z += -7.065389 * Q.log_sum_pt + 47.41367
    if 6.605974 <= Q.log_sum_pt < 6.701242:
        z += -7.766351 * Q.log_sum_pt + 52.0442
    if Q.log_sum_pt >= 6.896095:
        z += 8.166552 * Q.log_sum_pt - 56.31732
    if 0.02320757 <= Q.z_7 < 0.02807091:
        z += -42.29162 * Q.z_7 + 0.9814856
    if Q.z_7 >= 0.02807091:
        z += -63.26575 * Q.z_7 + 1.570249
    if Q.sum_z_dr < 0.007673833:
        z += 337.2612 * Q.sum_z_dr - 2.588086
    if Q.LHA >= 0.111565:
        z += -11.40997 * Q.LHA + 1.272954
    if Q.sum_pt < 615.875:
        z += -0.005052758 * Q.sum_pt + 3.111867
    if 937.0312 <= Q.sum_pt < 988.4078:
        z += 0.008270116 * Q.sum_pt - 7.749357
    if Q.sum_pt >= 988.4078:
        z += -0.01442581 * Q.sum_pt + 14.68347
    if Q.C2 < 0.0423228:
        z += -72.67446 * Q.C2 + 3.075787
    if Q.sum_zz_dr2 < 0.0030133:
        z += 502.7766 * Q.sum_zz_dr2 - 2.398119
    if 0.0030133 <= Q.sum_zz_dr2 < 0.00718279:
        z += 211.8011 * Q.sum_zz_dr2 - 1.521323
    if Q.lam1 < 0.003377388:
        z += -338.9078 * Q.lam1 + 1.144623
    if Q.lam1_plus_lam2 < 0.01323868:
        z += -134.2525 * Q.lam1_plus_lam2 + 1.777326
    if Q.lam1 < 0.00733008 and Q.max_dr < 0.2507612:
        z += 1230.69 * (0.00733008 - Q.lam1) * (0.2507612 - Q.max_dr)
    if Q.lam1 < 0.00733008 and Q.pt_6 < 62.25:
        z += -1.968961 * (0.00733008 - Q.lam1) * (62.25 - Q.pt_6)
    if Q.sum_pt_top5 > 791.125 and Q.D2_b2 < 1.129616:
        z += 0.007956838 * (Q.sum_pt_top5 - 791.125) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 1.59265:
        z += -9.115527 * (Q.log_sum_pt - 6.896095) * (1.59265 - Q.D2_b2)
    if Q.log_sum_pt < 6.605974 and Q.D2_b2 < 1.129616:
        z += 3.575003 * (6.605974 - Q.log_sum_pt) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt < 6.701242 and Q.D2_b2 < 1.129616:
        z += -2.760012 * (6.701242 - Q.log_sum_pt) * (1.129616 - Q.D2_b2)
    if Q.sum_pt_top5 > 791.125 and Q.abseta_0 < 0.01425934:
        z += -0.3348273 * (Q.sum_pt_top5 - 791.125) * (0.01425934 - Q.abseta_0)
    if Q.sum_pt > 988.4078 and Q.abseta_0 < 0.01425934:
        z += 0.5367119 * (Q.sum_pt - 988.4078) * (0.01425934 - Q.abseta_0)
    if Q.C2 < 0.0423228 and Q.C2_b2 < 0.02415398:
        z += -3649.013 * (0.0423228 - Q.C2) * (0.02415398 - Q.C2_b2)
    return max(0.0, z)


def neuron_3(Q):
    z = -3.289395
    if 0.06663269 <= Q.sum_z_dr < 0.08723651:
        z += 190.3989 * Q.sum_z_dr - 12.68679
    if Q.sum_z_dr >= 0.08723651:
        z += 153.9395 * Q.sum_z_dr - 9.506202
    if 0.1682655 <= Q.sj2_dr < 0.1778793:
        z += 22.60919 * Q.sj2_dr - 3.804348
    if 0.1778793 <= Q.sj2_dr < 0.2687922:
        z += 47.91979 * Q.sj2_dr - 8.306579
    if 0.2687922 <= Q.sj2_dr < 0.3003793:
        z += -0.9055042 * Q.sj2_dr + 4.817276
    if Q.sj2_dr >= 0.3003793:
        z += 12.44633 * Q.sj2_dr + 0.80666
    if 0.00609665 <= Q.lam1_plus_lam2 < 0.01323868:
        z += 443.665 * Q.lam1_plus_lam2 - 2.70487
    if 0.01323868 <= Q.lam1_plus_lam2 < 0.01882765:
        z += 67.80908 * Q.lam1_plus_lam2 + 2.270964
    if Q.lam1_plus_lam2 >= 0.01882765:
        z += 336.7123 * Q.lam1_plus_lam2 - 2.791853
    if 0.006390125 <= Q.sum_zz_dr2 < 0.00718279:
        z += -293.5703 * Q.sum_zz_dr2 + 1.875951
    if 0.00718279 <= Q.sum_zz_dr2 < 0.01165737:
        z += 303.8734 * Q.sum_zz_dr2 - 2.415362
    if Q.sum_zz_dr2 >= 0.01165737:
        z += 432.7337 * Q.sum_zz_dr2 - 3.917535
    if Q.sum_z_dr2 >= 0.008678045:
        z += -1127.164 * Q.sum_z_dr2 + 9.781578
    if 0.0811449 <= Q.tau1 < 0.1136369:
        z += -81.1872 * Q.tau1 + 6.587928
    if Q.tau1 >= 0.1136369:
        z += -53.85176 * Q.tau1 + 3.481613
    if Q.N3 < 1.138757:
        z += -1.166516 * Q.N3 + 1.328379
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += 322.8004 * Q.lam1 - 2.703638
    if Q.lam1 >= 0.01200373:
        z += 415.011 * Q.lam1 - 3.810508
    if Q.lam2 >= 0.000537286:
        z += 261.7076 * Q.lam2 - 0.1406119
    if Q.max_dr >= 0.1117619:
        z += 10.07464 * Q.max_dr - 1.125961
    if Q.sj2_dr > 0.1682655 and Q.sum_pt < 988.4078:
        z += -0.0192041 * (Q.sj2_dr - 0.1682655) * (988.4078 - Q.sum_pt)
    if Q.sj2_dr > 0.1682655 and Q.tau2 < 0.06297984:
        z += 277.0642 * (Q.sj2_dr - 0.1682655) * (0.06297984 - Q.tau2)
    if Q.sj2_dr > 0.1682655 and Q.n_dr_0p05_0p1 < 7.0:
        z += -4.577812 * (Q.sj2_dr - 0.1682655) * (7.0 - Q.n_dr_0p05_0p1)
    if Q.lam1_plus_lam2 > 0.00609665 and Q.pt_6 > 31.90625:
        z += -3.118849 * (Q.lam1_plus_lam2 - 0.00609665) * (Q.pt_6 - 31.90625)
    if Q.sum_z_dr > 0.08723651 and Q.lam2 < 0.0003061234:
        z += 139054.7 * (Q.sum_z_dr - 0.08723651) * (0.0003061234 - Q.lam2)
    if Q.lam1_plus_lam2 > 0.00609665 and Q.log_sum_pt > 6.192222:
        z += -364.1473 * (Q.lam1_plus_lam2 - 0.00609665) * (Q.log_sum_pt - 6.192222)
    if Q.sj2_dr > 0.1591713 and Q.n_dr_0p1_0p2 > 1.0:
        z += 1.501323 * (Q.sj2_dr - 0.1591713) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.sj2_dr > 0.3003793 and Q.z_dr_0p05_0p1 > 0.08878489:
        z += -70.81048 * (Q.sj2_dr - 0.3003793) * (Q.z_dr_0p05_0p1 - 0.08878489)
    if Q.lam1_plus_lam2 > 0.01323868 and Q.eccentricity > 0.9884745:
        z += -21157.27 * (Q.lam1_plus_lam2 - 0.01323868) * (Q.eccentricity - 0.9884745)
    if Q.sj2_dr > 0.1778793 and Q.eccentricity > 0.9458207:
        z += 888.1873 * (Q.sj2_dr - 0.1778793) * (Q.eccentricity - 0.9458207)
    if Q.sj2_dr > 0.1682655 and Q.eccentricity > 0.9458207:
        z += -649.1921 * (Q.sj2_dr - 0.1682655) * (Q.eccentricity - 0.9458207)
    if Q.sj2_dr > 0.1778793 and Q.sum_z_dr2_top2 < 0.001864148:
        z += -27781.82 * (Q.sj2_dr - 0.1778793) * (0.001864148 - Q.sum_z_dr2_top2)
    if Q.sj2_dr > 0.2687922 and Q.sum_z_dr2_top2 < 0.007639643:
        z += 3012.128 * (Q.sj2_dr - 0.2687922) * (0.007639643 - Q.sum_z_dr2_top2)
    return max(0.0, z)


def neuron_4(Q):
    z = -2.19817
    if Q.N2 < 0.2233283:
        z += -11.27944 * Q.N2 + 2.519018
    if Q.sum_z_dr2 < 0.002635418:
        z += 217.5995 * Q.sum_z_dr2 - 7.35582
    if 0.002635418 <= Q.sum_z_dr2 < 0.006679471:
        z += 1131.393 * Q.sum_z_dr2 - 9.764049
    if 0.006679471 <= Q.sum_z_dr2 < 0.008678045:
        z += 1104.257 * Q.sum_z_dr2 - 9.58279
    if Q.e3 < 0.0001869378:
        z += -26875.57 * Q.e3 + 5.024061
    if Q.lam1 < 0.00483998:
        z += -152.538 * Q.lam1 + 0.7382809
    if Q.sj2_dr >= 0.2414351:
        z += -10.26243 * Q.sj2_dr + 2.477712
    if Q.sum_z_dr2_top2 < 0.006299534:
        z += 202.9176 * Q.sum_z_dr2_top2 - 1.278286
    if Q.sj3_dr_min < 0.2089872:
        z += -16.21227 * Q.sj3_dr_min + 3.388159
    if Q.D2 < 0.7459513:
        z += -2.962347 * Q.D2 + 2.209766
    if Q.lam2 < 0.001130645:
        z += 3465.329 * Q.lam2 - 3.918056
    if Q.max_dr >= 0.0931108:
        z += 6.785706 * Q.max_dr - 0.6318225
    if 0.1789613 <= Q.sj3_dr_max < 0.213399:
        z += 22.27679 * Q.sj3_dr_max - 3.986685
    if Q.sj3_dr_max >= 0.213399:
        z += -0.08608246 * Q.sj3_dr_max + 0.7855299
    if Q.sum_z_dr2_top5 < 0.007164202:
        z += -70.4762 * Q.sum_z_dr2_top5 + 0.5049057
    if Q.sum_z_dr < 0.07608178:
        z += -21.95232 * Q.sum_z_dr + 1.670172
    if Q.tau2 < 0.01357153:
        z += 82.0442 * Q.tau2 - 1.113466
    if Q.e2 < 0.04447357:
        z += 50.26557 * Q.e2 - 2.235489
    if Q.sum_zz_dr2 < 0.01165737:
        z += -436.7758 * Q.sum_zz_dr2 + 5.091659
    if Q.C2_b2 < 0.009032972:
        z += -310.8735 * Q.C2_b2 + 2.808112
    if Q.C2 >= 0.02384388:
        z += -15.98518 * Q.C2 + 0.3811488
    if Q.sj2_dr > 0.2414351 and Q.C2_b2 < 0.0008333816:
        z += -20569.67 * (Q.sj2_dr - 0.2414351) * (0.0008333816 - Q.C2_b2)
    if Q.N2 < 0.2233283 and Q.planar_flow < 0.4007947:
        z += -20.57522 * (0.2233283 - Q.N2) * (0.4007947 - Q.planar_flow)
    if Q.e3 < 0.0001869378 and Q.sum_pt < 813.4156:
        z += -44.37347 * (0.0001869378 - Q.e3) * (813.4156 - Q.sum_pt)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1626038:
        z += -39.16458 * (0.2233283 - Q.N2) * (0.1626038 - Q.abseta_7)
    if Q.sj2_dr > 0.2414351 and Q.sj3_z3 < 0.1989187:
        z += 70.97226 * (Q.sj2_dr - 0.2414351) * (0.1989187 - Q.sj3_z3)
    if Q.N2 < 0.2233283 and Q.pt_6 < 24.42188:
        z += -1.829947 * (0.2233283 - Q.N2) * (24.42188 - Q.pt_6)
    if Q.sum_z_dr2_top2 < 0.006299534 and Q.centroid_offset > 0.01096064:
        z += 20532.23 * (0.006299534 - Q.sum_z_dr2_top2) * (Q.centroid_offset - 0.01096064)
    if Q.sj2_dr > 0.2414351 and Q.D2_b2 < 1.129616:
        z += -15.57966 * (Q.sj2_dr - 0.2414351) * (1.129616 - Q.D2_b2)
    if Q.max_dr > 0.0931108 and Q.absphi_0 < 0.04019165:
        z += 113.2739 * (Q.max_dr - 0.0931108) * (0.04019165 - Q.absphi_0)
    if Q.N2 < 0.2233283 and Q.n_dr_0p2_0p4 > 1.0:
        z += -6.920029 * (0.2233283 - Q.N2) * (Q.n_dr_0p2_0p4 - 1.0)
    if Q.sum_z_dr2 < 0.006679471 and Q.psi_0p3 < 1.0:
        z += -47690.76 * (0.006679471 - Q.sum_z_dr2) * (1.0 - Q.psi_0p3)
    if Q.sum_zz_dr2 < 0.01165737 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -2199.601 * (0.01165737 - Q.sum_zz_dr2) * (0.05643852 - Q.z_dr_0p2_0p4)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.9260067
    if Q.LHA < 0.2160559:
        z += -30.23586 * Q.LHA + 6.532635
    if Q.e2 < 0.03556091:
        z += -52.67274 * Q.e2 + 1.873091
    if Q.log_sum_pt >= 6.896095:
        z += -59.20201 * Q.log_sum_pt + 408.2627
    if Q.sum_z_dr2 < 0.0009641429:
        z += -2703.584 * Q.sum_z_dr2 + 5.175035
    if 0.0009641429 <= Q.sum_z_dr2 < 0.002635418:
        z += -1123.665 * Q.sum_z_dr2 + 3.651766
    if 0.002635418 <= Q.sum_z_dr2 < 0.005019719:
        z += -289.578 * Q.sum_z_dr2 + 1.4536
    if Q.sum_pt < 715.4688:
        z += 0.01945047 * Q.sum_pt - 13.9162
    if Q.z_7 < 0.02807091:
        z += -172.0695 * Q.z_7 + 4.830148
    if Q.sum_pt_top5 < 430.75:
        z += 0.01888539 * Q.sum_pt_top5 - 8.134884
    if Q.dr_0 < 0.04649465:
        z += 34.05906 * Q.dr_0 - 1.583564
    if Q.sj3_dr_max < 0.3012016:
        z += 13.22899 * Q.sj3_dr_max - 3.984593
    if Q.max_dr < 0.177305:
        z += -8.11066 * Q.max_dr + 1.43806
    if Q.e3 < 0.0005116989:
        z += -4896.9 * Q.e3 + 2.505739
    if Q.z_6 < 0.02160287:
        z += -257.3695 * Q.z_6 + 6.461436
    if 0.02160287 <= Q.z_6 < 0.06081235:
        z += -22.99226 * Q.z_6 + 1.398213
    if Q.pt_7 < 20.125:
        z += 0.2063215 * Q.pt_7 - 4.152221
    if Q.pt_7 >= 34.53125:
        z += -0.07749182 * Q.pt_7 + 2.675889
    if Q.mean_eta2 < 0.002085238:
        z += 184.5899 * Q.mean_eta2 - 0.3849139
    if Q.LHA < 0.2160559 and Q.z_6 < 0.09543973:
        z += -390.5792 * (0.2160559 - Q.LHA) * (0.09543973 - Q.z_6)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -185.3663 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.e2 < 0.03556091 and Q.zdr_0 > 0.004918231:
        z += 3117.007 * (0.03556091 - Q.e2) * (Q.zdr_0 - 0.004918231)
    if Q.e2 < 0.03556091 and Q.log_sum_pt > 6.896095:
        z += 714.2499 * (0.03556091 - Q.e2) * (Q.log_sum_pt - 6.896095)
    if Q.sum_z_dr2 < 0.002635418 and Q.centroid_offset < 0.01627885:
        z += 73541.7 * (0.002635418 - Q.sum_z_dr2) * (0.01627885 - Q.centroid_offset)
    if Q.LHA < 0.1767241 and Q.zdr_0 < 0.006292091:
        z += -1388.558 * (0.1767241 - Q.LHA) * (0.006292091 - Q.zdr_0)
    if Q.sum_z_dr2 < 0.002635418 and Q.pt_6 > 27.57812:
        z += -13.24546 * (0.002635418 - Q.sum_z_dr2) * (Q.pt_6 - 27.57812)
    if Q.sum_pt < 715.4688 and Q.sj3_z3 < 0.05464886:
        z += -0.8410264 * (715.4688 - Q.sum_pt) * (0.05464886 - Q.sj3_z3)
    if Q.log_sum_pt > 6.896095 and Q.zdr_0 > 0.01216943:
        z += 753.2867 * (Q.log_sum_pt - 6.896095) * (Q.zdr_0 - 0.01216943)
    if Q.z_7 < 0.04939969 and Q.absphi_0 < 0.04678345:
        z += 732.6635 * (0.04939969 - Q.z_7) * (0.04678345 - Q.absphi_0)
    if Q.sum_pt < 715.4688 and Q.dr_0 < 0.0361727:
        z += 1.216916 * (715.4688 - Q.sum_pt) * (0.0361727 - Q.dr_0)
    if Q.e2 < 0.03556091 and Q.n_dr_0p1_0p2 > 6.0:
        z += -307.9625 * (0.03556091 - Q.e2) * (Q.n_dr_0p1_0p2 - 6.0)
    if Q.log_sum_pt > 6.896095 and Q.mean_eta2 < 5.507482e-05:
        z += 228802.6 * (Q.log_sum_pt - 6.896095) * (5.507482e-05 - Q.mean_eta2)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.9867033
    if 0.00809236 <= Q.centroid_offset < 0.01627885:
        z += 70.50824 * Q.centroid_offset - 0.5705781
    if 0.01627885 <= Q.centroid_offset < 0.02076709:
        z += 251.8732 * Q.centroid_offset - 3.52299
    if Q.centroid_offset >= 0.02076709:
        z += 266.538 * Q.centroid_offset - 3.827536
    if Q.lam1_plus_lam2 < 0.01323868:
        z += 364.9707 * Q.lam1_plus_lam2 - 4.831729
    if Q.tau1 < 0.1136369:
        z += -73.28433 * Q.tau1 + 8.327804
    if Q.log_sum_pt < 6.327379:
        z += -10.89278 * Q.log_sum_pt + 69.23302
    if 6.327379 <= Q.log_sum_pt < 6.638339:
        z += -0.9978335 * Q.log_sum_pt + 6.623957
    if Q.tau2 >= 0.01713288:
        z += -28.16518 * Q.tau2 + 0.4825506
    if Q.lam2 < 0.001130645:
        z += 1145.676 * Q.lam2 + 0.02561488
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -579.9454 * Q.lam2 + 1.976679
    if Q.lam1 < 0.00733008:
        z += -419.952 * Q.lam1 + 5.817018
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += -769.5312 * Q.lam1 + 8.379462
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -533.1078 * Q.lam1 + 6.399281
    if Q.z_6 < 0.02160287:
        z += 421.1441 * Q.z_6 - 9.781339
    if 0.02160287 <= Q.z_6 < 0.03448406:
        z += 53.05543 * Q.z_6 - 1.829567
    if Q.sj3_dr_min >= 0.02022982:
        z += -16.21989 * Q.sj3_dr_min + 0.3281254
    if Q.sum_z_dr2 < 0.008678045:
        z += 1833.784 * Q.sum_z_dr2 - 15.91366
    if 0.07264452 <= Q.sj3_dr_max < 0.1789613:
        z += -15.1473 * Q.sj3_dr_max + 1.100368
    if Q.sj3_dr_max >= 0.1789613:
        z += 13.80196 * Q.sj3_dr_max - 4.08043
    if Q.sum_zz_dr2 < 0.008168571:
        z += -273.2857 * Q.sum_zz_dr2 + 2.232354
    if Q.sum_pt >= 988.4078:
        z += 0.01562187 * Q.sum_pt - 15.44078
    if Q.max_dr < 0.1598486:
        z += 16.87316 * Q.max_dr - 2.697151
    if Q.sum_z_dr < 0.02689598:
        z += -228.9304 * Q.sum_z_dr + 6.157309
    if Q.D2 < 1.332146:
        z += -0.8531506 * Q.D2 + 1.136522
    if Q.pt_6 < 29.90625:
        z += -0.1481126 * Q.pt_6 + 4.429492
    if Q.n_dr_0_0p05 < 3.0:
        z += -0.2244414 * Q.n_dr_0_0p05 + 0.6733243
    if Q.centroid_offset > 0.00809236 and Q.sj3_dr_min > 0.08543881:
        z += 243.6624 * (Q.centroid_offset - 0.00809236) * (Q.sj3_dr_min - 0.08543881)
    if Q.centroid_offset > 0.00809236 and Q.sj2_dr < 0.1682655:
        z += 798.5117 * (Q.centroid_offset - 0.00809236) * (0.1682655 - Q.sj2_dr)
    if Q.log_sum_pt < 6.327379 and Q.z_6 < 0.08051087:
        z += 370.8752 * (6.327379 - Q.log_sum_pt) * (0.08051087 - Q.z_6)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.planar_flow < 0.3220738:
        z += -573.6354 * (0.01323868 - Q.lam1_plus_lam2) * (0.3220738 - Q.planar_flow)
    if Q.centroid_offset > 0.00809236 and Q.pt_7 < 38.53125:
        z += -1.479767 * (Q.centroid_offset - 0.00809236) * (38.53125 - Q.pt_7)
    if Q.centroid_offset > 0.00809236 and Q.n_dr_0p05_0p1 < 6.0:
        z += 9.648827 * (Q.centroid_offset - 0.00809236) * (6.0 - Q.n_dr_0p05_0p1)
    if Q.log_sum_pt < 6.327379 and Q.pt_7 > 25.57812:
        z += -0.5165739 * (6.327379 - Q.log_sum_pt) * (Q.pt_7 - 25.57812)
    if Q.lam1 < 0.01200373 and Q.pt_6 < 19.46875:
        z += 16.04014 * (0.01200373 - Q.lam1) * (19.46875 - Q.pt_6)
    if Q.z_6 < 0.02160287 and Q.log_sum_pt < 6.842717:
        z += 4629.421 * (0.02160287 - Q.z_6) * (6.842717 - Q.log_sum_pt)
    if Q.log_sum_pt < 6.638339 and Q.z_7 < 0.0586137:
        z += 540.8658 * (6.638339 - Q.log_sum_pt) * (0.0586137 - Q.z_7)
    if Q.centroid_offset > 0.00809236 and Q.mean_phi2 < 0.01426135:
        z += 3861.053 * (Q.centroid_offset - 0.00809236) * (0.01426135 - Q.mean_phi2)
    if Q.centroid_offset > 0.01627885 and Q.sum_pt < 988.4078:
        z += -0.9925227 * (Q.centroid_offset - 0.01627885) * (988.4078 - Q.sum_pt)
    if Q.centroid_offset > 0.00809236 and Q.sum_pt < 715.4688:
        z += 0.5765209 * (Q.centroid_offset - 0.00809236) * (715.4688 - Q.sum_pt)
    if Q.sj3_dr_max > 0.1789613 and Q.n_dr_0p05_0p1 < 6.0:
        z += -0.7296273 * (Q.sj3_dr_max - 0.1789613) * (6.0 - Q.n_dr_0p05_0p1)
    return max(0.0, z)


def neuron_7(Q):
    z = 10.20431
    if Q.planar_flow < 0.1950135:
        z += 5.288383 * Q.planar_flow - 1.031306
    if Q.sum_z_dr2 < 0.0005611231:
        z += -878.0857 * Q.sum_z_dr2 + 0.4927142
    if 0.004372139 <= Q.sum_z_dr2 < 0.007520088:
        z += -637.2902 * Q.sum_z_dr2 + 2.786322
    if 0.007520088 <= Q.sum_z_dr2 < 0.008678045:
        z += -1849.085 * Q.sum_z_dr2 + 11.89913
    if 0.008678045 <= Q.sum_z_dr2 < 0.01323868:
        z += -2005.627 * Q.sum_z_dr2 + 13.2576
    if Q.sum_z_dr2 >= 0.01323868:
        z += -396.8175 * Q.sum_z_dr2 - 8.040902
    if Q.sum_zz_dr2 < 0.005284669:
        z += 1143.632 * Q.sum_zz_dr2 - 10.96901
    if 0.005284669 <= Q.sum_zz_dr2 < 0.00718279:
        z += 1279.927 * Q.sum_zz_dr2 - 11.68928
    if 0.00718279 <= Q.sum_zz_dr2 < 0.008168571:
        z += 1908.755 * Q.sum_zz_dr2 - 16.20602
    if 0.008168571 <= Q.sum_zz_dr2 < 0.01165737:
        z += 176.0546 * Q.sum_zz_dr2 - 2.052335
    if Q.tau1 < 0.05356915:
        z += -60.2978 * Q.tau1 + 5.342814
    if 0.05356915 <= Q.tau1 < 0.09538712:
        z += -37.87425 * Q.tau1 + 4.141603
    if 0.09538712 <= Q.tau1 < 0.1136369:
        z += 13.62013 * Q.tau1 - 0.7702972
    if 0.1136369 <= Q.tau1 < 0.1416226:
        z += -27.78028 * Q.tau1 + 3.934316
    if Q.lam1_plus_lam2 < 0.005590289:
        z += 494.7375 * Q.lam1_plus_lam2 - 2.765725
    if Q.sj2_dr < 0.1591713:
        z += -11.037 * Q.sj2_dr + 1.048905
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 25.19967 * Q.sj2_dr - 4.718933
    if Q.centroid_offset < 0.03776099:
        z += 40.83725 * Q.centroid_offset - 1.542055
    if Q.sum_z_dr < 0.08723651:
        z += 47.74976 * Q.sum_z_dr - 4.165523
    if Q.LHA < 0.2931906:
        z += -5.913222 * Q.LHA + 1.733701
    if Q.lam1 < 0.004183811:
        z += -275.6729 * Q.lam1 + 1.153363
    if Q.planar_flow < 0.1950135 and Q.log_sum_pt > 6.423044:
        z += 34.12973 * (0.1950135 - Q.planar_flow) * (Q.log_sum_pt - 6.423044)
    if Q.planar_flow < 0.1950135 and Q.z_7 < 0.05240025:
        z += -291.0635 * (0.1950135 - Q.planar_flow) * (0.05240025 - Q.z_7)
    if Q.lam1_plus_lam2 < 0.005590289 and Q.mean_phi < -0.00469376:
        z += 17805.23 * (0.005590289 - Q.lam1_plus_lam2) * (-0.00469376 - Q.mean_phi)
    if Q.sum_z_dr2 > 0.004372139 and Q.sj3_z1 > 0.8740683:
        z += -16514.5 * (Q.sum_z_dr2 - 0.004372139) * (Q.sj3_z1 - 0.8740683)
    if Q.sum_z_dr2 > 0.008678045 and Q.sj3_z1 > 0.8740683:
        z += 40400.22 * (Q.sum_z_dr2 - 0.008678045) * (Q.sj3_z1 - 0.8740683)
    if Q.sum_zz_dr2 < 0.01165737 and Q.pt_7 < 48.71875:
        z += -1.513149 * (0.01165737 - Q.sum_zz_dr2) * (48.71875 - Q.pt_7)
    if Q.lam1_plus_lam2 < 0.005590289 and Q.mean_eta < -0.006779839:
        z += 19441.97 * (0.005590289 - Q.lam1_plus_lam2) * (-0.006779839 - Q.mean_eta)
    if Q.lam1_plus_lam2 < 0.005590289 and Q.mean_eta > 0.009367547:
        z += 23768.59 * (0.005590289 - Q.lam1_plus_lam2) * (Q.mean_eta - 0.009367547)
    if Q.lam1_plus_lam2 < 0.005590289 and Q.mean_phi > 0.009050008:
        z += 20789.63 * (0.005590289 - Q.lam1_plus_lam2) * (Q.mean_phi - 0.009050008)
    if Q.sum_z_dr2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 2213.768 * (Q.sum_z_dr2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    return max(0.0, z)


def neuron_8(Q):
    z = 0.06261912
    if Q.sum_z_dr2 < 0.0003193707:
        z += -4143.648 * Q.sum_z_dr2 + 3.91053
    if 0.0003193707 <= Q.sum_z_dr2 < 0.005019719:
        z += -473.4504 * Q.sum_z_dr2 + 2.738376
    if 0.005019719 <= Q.sum_z_dr2 < 0.006679471:
        z += -217.9771 * Q.sum_z_dr2 + 1.455972
    if Q.tau1 < 0.01357518:
        z += 14.22476 * Q.tau1 + 1.650808
    if 0.01357518 <= Q.tau1 < 0.05356915:
        z += -46.10474 * Q.tau1 + 2.469792
    if Q.LHA < 0.1967397:
        z += -15.33195 * Q.LHA + 3.016403
    if Q.log_sum_pt >= 6.701242:
        z += -17.48904 * Q.log_sum_pt + 117.1983
    if Q.sj3_dr_max < 0.1070199:
        z += -2.785247 * Q.sj3_dr_max - 0.1964072
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 22.20986 * Q.sj3_dr_max - 2.871383
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += -5.286025 * Q.sj3_dr_max + 1.049948
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.01673897 * Q.sum_pt_top5 - 11.50699
    if Q.sum_z_dr < 0.05464922:
        z += 76.42539 * Q.sum_z_dr - 4.176588
    if Q.lam1 < 0.001503553:
        z += -288.3413 * Q.lam1 + 0.4335365
    if Q.z_dr_0_0p05 >= 0.9008535:
        z += 7.977057 * Q.z_dr_0_0p05 - 7.18616
    if Q.sum_z_dr2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 20432.81 * (0.006679471 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.sum_z_dr2 < 0.005019719 and Q.log_sum_pt < 6.502799:
        z += -640.9951 * (0.005019719 - Q.sum_z_dr2) * (6.502799 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.006679471 and Q.planar_flow < 0.4007947:
        z += 308.231 * (0.006679471 - Q.sum_z_dr2) * (0.4007947 - Q.planar_flow)
    if Q.sum_z_dr2 < 0.005019719 and Q.pt_7 < 43.5:
        z += -11.71494 * (0.005019719 - Q.sum_z_dr2) * (43.5 - Q.pt_7)
    if Q.sum_z_dr2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -40770.71 * (0.005019719 - Q.sum_z_dr2) * (Q.centroid_offset - 0.006789738)
    if Q.LHA < 0.1967397 and Q.z_dr_0p2_0p4 < 0.20552:
        z += -186.5272 * (0.1967397 - Q.LHA) * (0.20552 - Q.z_dr_0p2_0p4)
    if Q.sum_z_dr2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += 4192.383 * (0.005019719 - Q.sum_z_dr2) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt > 6.701242 and Q.psi_0p2 > 0.9435576:
        z += 39.27649 * (Q.log_sum_pt - 6.701242) * (Q.psi_0p2 - 0.9435576)
    if Q.sj3_dr_max < 0.1986272 and Q.lam1_plus_lam2 > 0.002635418:
        z += -10563.14 * (0.1986272 - Q.sj3_dr_max) * (Q.lam1_plus_lam2 - 0.002635418)
    if Q.tau1 < 0.05356915 and Q.sum_z_dr2 < 0.002635418:
        z += 46994.43 * (0.05356915 - Q.tau1) * (0.002635418 - Q.sum_z_dr2)
    if Q.sum_pt_top5 > 687.4375 and Q.sum_z_dr2_top3 < 0.01002369:
        z += -1.398659 * (Q.sum_pt_top5 - 687.4375) * (0.01002369 - Q.sum_z_dr2_top3)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2_top3 < 0.002151568:
        z += 6528.313 * (Q.log_sum_pt - 6.701242) * (0.002151568 - Q.sum_z_dr2_top3)
    if Q.sum_z_dr < 0.05464922 and Q.lam2 < 0.0001947983:
        z += 361526.8 * (0.05464922 - Q.sum_z_dr) * (0.0001947983 - Q.lam2)
    if Q.tau1 < 0.05356915 and Q.lam2 < 0.0003061234:
        z += -169383.4 * (0.05356915 - Q.tau1) * (0.0003061234 - Q.lam2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2_top3 > 0.001155057:
        z += 3953.832 * (0.1986272 - Q.sj3_dr_max) * (Q.sum_z_dr2_top3 - 0.001155057)
    if Q.LHA < 0.1967397 and Q.lam2 < 0.0001947983:
        z += -71018.66 * (0.1967397 - Q.LHA) * (0.0001947983 - Q.lam2)
    if Q.tau1 < 0.05356915 and Q.pt_7 < 43.5:
        z += 0.8985381 * (0.05356915 - Q.tau1) * (43.5 - Q.pt_7)
    if Q.LHA < 0.1967397 and Q.pt_7 > 15.55391:
        z += 0.4073124 * (0.1967397 - Q.LHA) * (Q.pt_7 - 15.55391)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 > 15.55391:
        z += -0.09066947 * (Q.log_sum_pt - 6.701242) * (Q.pt_7 - 15.55391)
    if Q.z_dr_0_0p05 > 0.9008535 and Q.lam2 < 0.0001947983:
        z += -58162.93 * (Q.z_dr_0_0p05 - 0.9008535) * (0.0001947983 - Q.lam2)
    if Q.sum_z_dr2 < 0.006679471 and Q.pt_7 > 31.85938:
        z += -4.145374 * (0.006679471 - Q.sum_z_dr2) * (Q.pt_7 - 31.85938)
    if Q.log_sum_pt > 6.701242 and Q.lam1_plus_lam2 < 0.002635418:
        z += 2171.648 * (Q.log_sum_pt - 6.701242) * (0.002635418 - Q.lam1_plus_lam2)
    if Q.sum_pt_top5 > 687.4375 and Q.sum_z_dr2 < 0.0009641429:
        z += -5.240542 * (Q.sum_pt_top5 - 687.4375) * (0.0009641429 - Q.sum_z_dr2)
    if Q.sum_z_dr < 0.05464922 and Q.lam1_plus_lam2 < 0.002635418:
        z += -30605.21 * (0.05464922 - Q.sum_z_dr) * (0.002635418 - Q.lam1_plus_lam2)
    if Q.sum_z_dr2 < 0.006679471 and Q.lam1_plus_lam2 > 0.001653836:
        z += -118959.6 * (0.006679471 - Q.sum_z_dr2) * (Q.lam1_plus_lam2 - 0.001653836)
    if Q.sum_z_dr < 0.05464922 and Q.lam1_plus_lam2 < 0.005590289:
        z += -17682.44 * (0.05464922 - Q.sum_z_dr) * (0.005590289 - Q.lam1_plus_lam2)
    if Q.sj3_dr_max < 0.1986272 and Q.lam2 < 0.0003061234:
        z += 39827.09 * (0.1986272 - Q.sj3_dr_max) * (0.0003061234 - Q.lam2)
    if Q.sj3_dr_max < 0.1070199 and Q.lam2 < 0.0003061234:
        z += -81120.65 * (0.1070199 - Q.sj3_dr_max) * (0.0003061234 - Q.lam2)
    return max(0.0, z)


def neuron_9(Q):
    z = -4.73394
    if Q.sum_z_dr < 0.05464922:
        z += 98.07736 * Q.sum_z_dr - 6.072236
    if 0.05464922 <= Q.sum_z_dr < 0.06663269:
        z += 59.44726 * Q.sum_z_dr - 3.961131
    if Q.sum_z_dr2 < 0.003562611:
        z += -2932.768 * Q.sum_z_dr2 + 12.40939
    if 0.003562611 <= Q.sum_z_dr2 < 0.005019719:
        z += -1345.871 * Q.sum_z_dr2 + 6.755893
    if Q.LHA < 0.2160559:
        z += -10.66872 * Q.LHA + 2.30504
    if Q.e3 >= 0.0001869378:
        z += 2218.519 * Q.e3 - 0.414725
    if Q.centroid_offset < 0.01837778:
        z += -16.8842 * Q.centroid_offset + 0.7401082
    if 0.01837778 <= Q.centroid_offset < 0.02355416:
        z += -83.03371 * Q.centroid_offset + 1.955789
    if Q.log_sum_pt < 6.896095:
        z += -5.974864 * Q.log_sum_pt + 41.20323
    if Q.lam1_plus_lam2 < 0.006679471:
        z += -1087.662 * Q.lam1_plus_lam2 + 7.265005
    if Q.lam1_plus_lam2 >= 0.02530566:
        z += -116.3291 * Q.lam1_plus_lam2 + 2.943786
    if Q.lam2 < 0.0001947983:
        z += -3471.281 * Q.lam2 + 0.6761996
    if Q.lam2 >= 0.001130645:
        z += 1424.474 * Q.lam2 - 1.610574
    if Q.sj2_dr < 0.1294903:
        z += 18.75515 * Q.sj2_dr - 2.42861
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.6619907 * Q.n_dr_0p2_0p4 - 0.6619907
    if Q.sj3_dr_max < 0.1426152:
        z += 14.6404 * Q.sj3_dr_max - 0.6891573
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += -24.97295 * Q.sj3_dr_max + 4.960307
    if Q.sj3_dr_max >= 0.2623172:
        z += -5.952806 * Q.sj3_dr_max + 1.561524
    if Q.sum_zz_dr2 < 0.001101266:
        z += 2965.796 * Q.sum_zz_dr2 - 8.934277
    if 0.001101266 <= Q.sum_zz_dr2 < 0.0030133:
        z += 2089.786 * Q.sum_zz_dr2 - 7.969556
    if 0.0030133 <= Q.sum_zz_dr2 < 0.004641801:
        z += 1026.959 * Q.sum_zz_dr2 - 4.766941
    if Q.max_dr < 0.1117619:
        z += -16.31208 * Q.max_dr + 1.823069
    if Q.zdr_0 < 0.004918231:
        z += 142.906 * Q.zdr_0 - 0.7028449
    if Q.centroid_offset < 0.01837778 and Q.M2 < 0.04435703:
        z += -537.101 * (0.01837778 - Q.centroid_offset) * (0.04435703 - Q.M2)
    if Q.centroid_offset < 0.01837778 and Q.tau3 < 0.01364517:
        z += 6929.423 * (0.01837778 - Q.centroid_offset) * (0.01364517 - Q.tau3)
    if Q.centroid_offset < 0.01837778 and Q.C2 < 0.06729223:
        z += 2216.104 * (0.01837778 - Q.centroid_offset) * (0.06729223 - Q.C2)
    if Q.lam2 > 0.001130645 and Q.planar_flow > 0.2534037:
        z += -1150.052 * (Q.lam2 - 0.001130645) * (Q.planar_flow - 0.2534037)
    if Q.log_sum_pt < 6.896095 and Q.z_5 < 0.05753583:
        z += 221.9287 * (6.896095 - Q.log_sum_pt) * (0.05753583 - Q.z_5)
    if Q.centroid_offset < 0.01837778 and Q.n_dr_0p05_0p1 < 5.0:
        z += 25.64145 * (0.01837778 - Q.centroid_offset) * (5.0 - Q.n_dr_0p05_0p1)
    if Q.centroid_offset < 0.02355416 and Q.tau21_b2 > 0.004811143:
        z += 103.7409 * (0.02355416 - Q.centroid_offset) * (Q.tau21_b2 - 0.004811143)
    if Q.centroid_offset < 0.02355416 and Q.D2_b2 < 0.380911:
        z += -209.9024 * (0.02355416 - Q.centroid_offset) * (0.380911 - Q.D2_b2)
    if Q.log_sum_pt < 6.896095 and Q.planar_flow > 0.1115136:
        z += -2.885243 * (6.896095 - Q.log_sum_pt) * (Q.planar_flow - 0.1115136)
    if Q.lam1_plus_lam2 < 0.006679471 and Q.mean_eta < -0.006779839:
        z += -11532.71 * (0.006679471 - Q.lam1_plus_lam2) * (-0.006779839 - Q.mean_eta)
    if Q.e3 > 0.0001869378 and Q.pt_7 < 33.21875:
        z += 335.7059 * (Q.e3 - 0.0001869378) * (33.21875 - Q.pt_7)
    if Q.log_sum_pt < 6.896095 and Q.z_dr_0p2_0p4 > 0.0:
        z += -3.454938 * (6.896095 - Q.log_sum_pt) * (Q.z_dr_0p2_0p4 - 0.0)
    if Q.sj2_dr < 0.1294903 and Q.mean_eta < -0.006779839:
        z += 530.2736 * (0.1294903 - Q.sj2_dr) * (-0.006779839 - Q.mean_eta)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.556419
    if Q.lam1 < 0.003377388:
        z += 1050.848 * Q.lam1 - 3.891984
    if 0.003377388 <= Q.lam1 < 0.00483998:
        z += 546.6168 * Q.lam1 - 2.189001
    if 0.00483998 <= Q.lam1 < 0.006506576:
        z += 356.2409 * Q.lam1 - 1.267585
    if Q.lam1 >= 0.006506576:
        z += 19.81195 * Q.lam1 + 0.9214153
    if 0.0001947983 <= Q.lam2 < 0.003408389:
        z += 831.3434 * Q.lam2 - 0.1619443
    if Q.lam2 >= 0.003408389:
        z += 373.5528 * Q.lam2 + 1.398384
    if Q.pt_6 < 48.03125:
        z += 0.03029862 * Q.pt_6 - 1.455281
    if Q.e3 >= 8.147744e-05:
        z += 1864.979 * Q.e3 - 0.1519538
    if 0.1967397 <= Q.LHA < 0.3127275:
        z += -17.3807 * Q.LHA + 3.419475
    if Q.LHA >= 0.3127275:
        z += -22.809 * Q.LHA + 5.117052
    if Q.centroid_offset >= 0.02355416:
        z += 16.52056 * Q.centroid_offset - 0.3891279
    if Q.n_dr_0p05_0p1 < 5.0:
        z += -0.1152444 * Q.n_dr_0p05_0p1 + 0.5762222
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.8975678 * Q.n_dr_0p2_0p4 - 0.8975678
    if Q.tau1 >= 0.04369778:
        z += 25.23009 * Q.tau1 - 1.102499
    if 0.1294903 <= Q.sj2_dr < 0.1682655:
        z += 10.20957 * Q.sj2_dr - 1.32204
    if 0.1682655 <= Q.sj2_dr < 0.3003793:
        z += -1.148379 * Q.sj2_dr + 0.5891111
    if Q.sj2_dr >= 0.3003793:
        z += 17.51891 * Q.sj2_dr - 5.018157
    if Q.sum_z_dr2_top3 < 0.002915531:
        z += -259.1347 * Q.sum_z_dr2_top3 + 0.7555153
    if 0.007520088 <= Q.sum_z_dr2 < 0.02530566:
        z += 213.273 * Q.sum_z_dr2 - 1.603832
    if Q.sum_z_dr2 >= 0.02530566:
        z += 78.8293 * Q.sum_z_dr2 + 1.798356
    if Q.lam2 > 0.0001947983 and Q.log_sum_pt < 6.538559:
        z += -526.1496 * (Q.lam2 - 0.0001947983) * (6.538559 - Q.log_sum_pt)
    if Q.lam2 > 0.0001947983 and Q.eccentricity > 0.4829602:
        z += -1253.685 * (Q.lam2 - 0.0001947983) * (Q.eccentricity - 0.4829602)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.D2_b2 < 4.721224:
        z += -0.1554629 * (Q.n_dr_0p2_0p4 - 1.0) * (4.721224 - Q.D2_b2)
    if Q.lam2 > 0.0001947983 and Q.planar_flow > 0.06126205:
        z += -303.1534 * (Q.lam2 - 0.0001947983) * (Q.planar_flow - 0.06126205)
    if Q.sj2_dr > 0.1682655 and Q.planar_flow < 0.6947818:
        z += -25.17062 * (Q.sj2_dr - 0.1682655) * (0.6947818 - Q.planar_flow)
    if Q.LHA > 0.3127275 and Q.planar_flow < 0.6947818:
        z += -36.12538 * (Q.LHA - 0.3127275) * (0.6947818 - Q.planar_flow)
    return max(0.0, z)


def neuron_11(Q):
    z = -2.242266
    if Q.planar_flow < 0.2534037:
        z += -6.757399 * Q.planar_flow + 1.71235
    if 0.09419022 <= Q.sj2_dr < 0.1778793:
        z += 10.89621 * Q.sj2_dr - 1.026316
    if 0.1778793 <= Q.sj2_dr < 0.2687922:
        z += -2.309722 * Q.sj2_dr + 1.322745
    if Q.sj2_dr >= 0.2687922:
        z += 7.3135 * Q.sj2_dr - 1.263901
    if Q.lam1_plus_lam2 < 0.008678045:
        z += -2169.667 * Q.lam1_plus_lam2 + 21.18449
    if 0.008678045 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -516.6016 * Q.lam1_plus_lam2 + 6.839121
    if Q.LHA < 0.3033137:
        z += 8.857117 * Q.LHA - 3.152927
    if 0.3033137 <= Q.LHA < 0.3127275:
        z += 49.54911 * Q.LHA - 15.49537
    if Q.centroid_offset < 0.03776099:
        z += -55.36138 * Q.centroid_offset + 2.090501
    if Q.centroid_offset >= 0.04990367:
        z += 194.8974 * Q.centroid_offset - 9.726094
    if Q.sum_z_dr < 0.02689598:
        z += 90.19915 * Q.sum_z_dr - 5.137712
    if 0.02689598 <= Q.sum_z_dr < 0.0717028:
        z += 60.52019 * Q.sum_z_dr - 4.339467
    if Q.tau1 < 0.03459477:
        z += 39.05997 * Q.tau1 - 1.351271
    if Q.log_sum_pt >= 6.572938:
        z += 1.784172 * Q.log_sum_pt - 11.72725
    if Q.sum_zz_dr2 < 0.006390125:
        z += 666.6511 * Q.sum_zz_dr2 - 4.916777
    if 0.006390125 <= Q.sum_zz_dr2 < 0.008168571:
        z += 369.3073 * Q.sum_zz_dr2 - 3.016713
    if Q.mean_phi2 < 0.001570267:
        z += 243.1574 * Q.mean_phi2 - 0.3818222
    if Q.sum_z_dr2 < 0.003562611:
        z += 352.4465 * Q.sum_z_dr2 - 1.25563
    if Q.lam1 < 0.008375572:
        z += 857.5451 * Q.lam1 - 7.182431
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.01661525 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.pt_7 < 37.15625:
        z += -0.1871496 * (0.2534037 - Q.planar_flow) * (37.15625 - Q.pt_7)
    if Q.tau1 < 0.09538712 and Q.n_dr_0p05_0p1 < 6.0:
        z += 1.671154 * (0.09538712 - Q.tau1) * (6.0 - Q.n_dr_0p05_0p1)
    if Q.planar_flow < 0.2534037 and Q.e3 < 1.050302e-05:
        z += -379142.0 * (0.2534037 - Q.planar_flow) * (1.050302e-05 - Q.e3)
    if Q.sj2_dr > 0.1778793 and Q.lam2 < 0.001130645:
        z += -8712.945 * (Q.sj2_dr - 0.1778793) * (0.001130645 - Q.lam2)
    if Q.log_sum_pt > 6.572938 and Q.D2 < 1.002471:
        z += -5.730601 * (Q.log_sum_pt - 6.572938) * (1.002471 - Q.D2)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += -537.0956 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.centroid_offset < 0.01437952 and Q.tau21_b2 < 0.05932655:
        z += 2723.145 * (0.01437952 - Q.centroid_offset) * (0.05932655 - Q.tau21_b2)
    if Q.centroid_offset > 0.04990367 and Q.M3 > 0.03843235:
        z += -2067.678 * (Q.centroid_offset - 0.04990367) * (Q.M3 - 0.03843235)
    if Q.centroid_offset < 0.01437952 and Q.sj3_dr_min < 0.08543881:
        z += -1303.65 * (0.01437952 - Q.centroid_offset) * (0.08543881 - Q.sj3_dr_min)
    if Q.tau1 < 0.03459477 and Q.sum_z_dr2_top3 < 0.002915531:
        z += 18654.84 * (0.03459477 - Q.tau1) * (0.002915531 - Q.sum_z_dr2_top3)
    if Q.centroid_offset < 0.03776099 and Q.pt_7 < 53.4375:
        z += -0.7505072 * (0.03776099 - Q.centroid_offset) * (53.4375 - Q.pt_7)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.2279532
    if 0.01882765 <= Q.sum_z_dr2 < 0.02530566:
        z += 259.2098 * Q.sum_z_dr2 - 4.880313
    if Q.sum_z_dr2 >= 0.02530566:
        z += 387.7399 * Q.sum_z_dr2 - 8.132851
    if Q.e2 >= 0.06344108:
        z += -71.07346 * Q.e2 + 4.508977
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += 31.51092 * Q.lam1_plus_lam2 - 0.4171629
    if Q.sum_z_dr2 > 0.01882765 and Q.pt_7 < 53.4375:
        z += -5.199678 * (Q.sum_z_dr2 - 0.01882765) * (53.4375 - Q.pt_7)
    if Q.lam1_plus_lam2 > 0.01323868 and Q.log_sum_pt > 6.327379:
        z += 363.3725 * (Q.lam1_plus_lam2 - 0.01323868) * (Q.log_sum_pt - 6.327379)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.4655485
    if Q.sum_z_dr < 0.1484084:
        z += -94.66884 * Q.sum_z_dr + 14.04965
    if Q.lam1 < 0.01643375:
        z += -69.65487 * Q.lam1 + 1.144691
    if Q.z_7 >= 0.0586137:
        z += 39.39455 * Q.z_7 - 2.309061
    if Q.pt_6 < 27.57812:
        z += 0.1850325 * Q.pt_6 - 6.121976
    if 27.57812 <= Q.pt_6 < 50.25:
        z += 0.04495117 * Q.pt_6 - 2.258796
    if Q.sj3_dr_max >= 0.233678:
        z += -9.37116 * Q.sj3_dr_max + 2.189834
    if Q.pt_5 < 24.57812:
        z += 0.1837051 * Q.pt_5 - 4.515126
    if Q.e2 < 0.08000524:
        z += 25.33973 * Q.e2 - 2.027311
    if Q.sj3_dr_min < 0.1278212:
        z += 4.889645 * Q.sj3_dr_min - 0.6250001
    if Q.M2 < 0.0294431:
        z += -41.96397 * Q.M2 + 1.235549
    if Q.pt_7 >= 31.85938:
        z += -0.04965774 * Q.pt_7 + 1.582065
    if Q.tau1 < 0.1136369:
        z += 24.28768 * Q.tau1 - 2.759977
    if Q.lam1_plus_lam2 < 0.01323868:
        z += -114.1339 * Q.lam1_plus_lam2 + 1.510981
    if Q.sum_pt_top5 >= 482.2812:
        z += -0.003145174 * Q.sum_pt_top5 + 1.516859
    if Q.sum_z_dr < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -77.62641 * (0.1484084 - Q.sum_z_dr) * (6.804164 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.1484084 and Q.pt_7 < 38.53125:
        z += -0.7457773 * (0.1484084 - Q.sum_z_dr) * (38.53125 - Q.pt_7)
    if Q.sum_z_dr < 0.1484084 and Q.tau2 < 0.03585464:
        z += -223.8707 * (0.1484084 - Q.sum_z_dr) * (0.03585464 - Q.tau2)
    if Q.sum_pt_top5 > 687.4375 and Q.pt_7 < 40.04062:
        z += 0.0007536958 * (Q.sum_pt_top5 - 687.4375) * (40.04062 - Q.pt_7)
    if Q.lam1 < 0.01643375 and Q.pt_7 < 25.57812:
        z += -8.528039 * (0.01643375 - Q.lam1) * (25.57812 - Q.pt_7)
    if Q.e3 < 8.147744e-05 and Q.centroid_offset < 0.03776099:
        z += -768714.8 * (8.147744e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.e2 < 0.08000524 and Q.pt_4 < 60.03125:
        z += 0.3431804 * (0.08000524 - Q.e2) * (60.03125 - Q.pt_4)
    if Q.sum_pt_top5 > 687.4375 and Q.dr_2 < 0.08525808:
        z += 0.03184545 * (Q.sum_pt_top5 - 687.4375) * (0.08525808 - Q.dr_2)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.6750596
    if 0.006679471 <= Q.sum_z_dr2 < 0.007520088:
        z += -1085.685 * Q.sum_z_dr2 + 7.251805
    if 0.007520088 <= Q.sum_z_dr2 < 0.008678045:
        z += -1832.076 * Q.sum_z_dr2 + 12.86473
    if Q.sum_z_dr2 >= 0.008678045:
        z += -1786.914 * Q.sum_z_dr2 + 12.47281
    if 0.002074109 <= Q.sum_zz_dr2 < 0.006390125:
        z += 142.4851 * Q.sum_zz_dr2 - 0.2955296
    if 0.006390125 <= Q.sum_zz_dr2 < 0.00718279:
        z += 497.4595 * Q.sum_zz_dr2 - 2.563861
    if 0.00718279 <= Q.sum_zz_dr2 < 0.01165737:
        z += 1404.883 * Q.sum_zz_dr2 - 9.081691
    if 0.01165737 <= Q.sum_zz_dr2 < 0.01716248:
        z += -1619.923 * Q.sum_zz_dr2 + 26.17961
    if Q.sum_zz_dr2 >= 0.01716248:
        z += -1554.026 * Q.sum_zz_dr2 + 25.04864
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.01323868:
        z += 492.2667 * Q.lam1_plus_lam2 - 2.751913
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += -1016.651 * Q.lam1_plus_lam2 + 17.22415
    if 0.03556091 <= Q.e2 < 0.05028464:
        z += -30.34204 * Q.e2 + 1.07899
    if Q.e2 >= 0.05028464:
        z += 134.9774 * Q.e2 - 7.234039
    if 0.02355416 <= Q.centroid_offset < 0.04990367:
        z += -90.61003 * Q.centroid_offset + 2.134243
    if Q.centroid_offset >= 0.04990367:
        z += -2107.344 * Q.centroid_offset + 102.7767
    if 0.2160559 <= Q.LHA < 0.3033137:
        z += 7.768503 * Q.LHA - 1.678431
    if Q.LHA >= 0.3033137:
        z += 24.39184 * Q.LHA - 6.720516
    if 0.03459477 <= Q.tau1 < 0.09538712:
        z += -86.36903 * Q.tau1 + 2.987917
    if Q.tau1 >= 0.09538712:
        z += -64.99355 * Q.tau1 + 0.9489709
    if 0.02689598 <= Q.sum_z_dr < 0.04081947:
        z += 50.56203 * Q.sum_z_dr - 1.359915
    if 0.04081947 <= Q.sum_z_dr < 0.08723651:
        z += 153.7219 * Q.sum_z_dr - 5.570846
    if 0.08723651 <= Q.sum_z_dr < 0.1019409:
        z += -1.67524 * Q.sum_z_dr + 7.985457
    if Q.sum_z_dr >= 0.1019409:
        z += -475.5219 * Q.sum_z_dr + 56.28982
    if Q.sj2_dr < 0.1591713:
        z += 5.999078 * Q.sj2_dr - 0.9548811
    if Q.sj3_dr_max < 0.233678:
        z += 30.55332 * Q.sj3_dr_max - 7.139638
    if Q.sum_z_dr2 > 0.008678045 and Q.log_sum_pt > 6.267538:
        z += -5503.328 * (Q.sum_z_dr2 - 0.008678045) * (Q.log_sum_pt - 6.267538)
    if Q.sum_zz_dr2 > 0.002074109 and Q.log_sum_pt > 6.267538:
        z += 292.826 * (Q.sum_zz_dr2 - 0.002074109) * (Q.log_sum_pt - 6.267538)
    if Q.lam1_plus_lam2 > 0.01323868 and Q.sum_pt > 488.9312:
        z += -6.561913 * (Q.lam1_plus_lam2 - 0.01323868) * (Q.sum_pt - 488.9312)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.784458
    if Q.N2 < 0.2233283:
        z += -26.44803 * Q.N2 + 5.906593
    if Q.lam2 < 0.003408389:
        z += 347.6077 * Q.lam2 - 1.184782
    if Q.sum_z_dr2 < 0.006679471:
        z += 1216.911 * Q.sum_z_dr2 - 4.881616
    if 0.006679471 <= Q.sum_z_dr2 < 0.007520088:
        z += 357.3007 * Q.sum_z_dr2 + 0.8601282
    if 0.007520088 <= Q.sum_z_dr2 < 0.01323868:
        z += -620.2687 * Q.sum_z_dr2 + 8.211536
    if Q.sum_zz_dr2 < 0.005834489:
        z += -544.1767 * Q.sum_zz_dr2 + 3.34419
    if 0.005834489 <= Q.sum_zz_dr2 < 0.006390125:
        z += -726.7434 * Q.sum_zz_dr2 + 4.409373
    if 0.006390125 <= Q.sum_zz_dr2 < 0.01165737:
        z += 200.3096 * Q.sum_zz_dr2 - 1.514611
    if 0.01165737 <= Q.sum_zz_dr2 < 0.01716248:
        z += -149.0385 * Q.sum_zz_dr2 + 2.55787
    if Q.e2 < 0.04110972:
        z += -4.058056 * Q.e2 - 1.256365
    if 0.04110972 <= Q.e2 < 0.06344108:
        z += 63.73059 * Q.e2 - 4.043137
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 890.0273 * Q.sum_z_dr2_top3 - 1.914954
    if Q.lam1 < 0.005433361:
        z += 87.08241 * Q.lam1 - 0.4298398
    if 0.005433361 <= Q.lam1 < 0.00595415:
        z += -83.16286 * Q.lam1 + 0.4951641
    if 0.1879486 <= Q.sj3_dr_max < 0.233678:
        z += 11.92124 * Q.sj3_dr_max - 2.240581
    if 0.233678 <= Q.sj3_dr_max < 0.2623172:
        z += 28.90751 * Q.sj3_dr_max - 6.209896
    if Q.sj3_dr_max >= 0.2623172:
        z += -1.634547 * Q.sj3_dr_max + 1.801811
    if Q.dr_0 < 0.08082334:
        z += -10.10715 * Q.dr_0 + 0.8168936
    if Q.z_dr_0_0p05 < 0.1515405:
        z += -3.840288 * Q.z_dr_0_0p05 + 0.581959
    if Q.lam1_plus_lam2 < 0.002635418:
        z += 683.969 * Q.lam1_plus_lam2 - 1.802544
    if Q.tau1 < 0.0811449:
        z += -69.84198 * Q.tau1 + 5.667321
    if Q.N2 < 0.2233283 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -15.06615 * (0.2233283 - Q.N2) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1872617:
        z += -663.6603 * (0.2233283 - Q.N2) * (0.1872617 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.sum_z_dr2 > 0.008678045:
        z += -7926.346 * (0.2233283 - Q.N2) * (Q.sum_z_dr2 - 0.008678045)
    if Q.N2 < 0.2233283 and Q.LHA > 0.2931906:
        z += 234.8223 * (0.2233283 - Q.N2) * (Q.LHA - 0.2931906)
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1591713:
        z += 628.8804 * (0.2233283 - Q.N2) * (0.1591713 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.LHA > 0.4235925:
        z += -1115.64 * (0.2233283 - Q.N2) * (Q.LHA - 0.4235925)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3255822:
        z += 386.1648 * (0.2233283 - Q.N2) * (Q.LHA - 0.3255822)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3855647:
        z += -118.4173 * (0.2233283 - Q.N2) * (Q.LHA - 0.3855647)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.01462716 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
    if Q.N2 < 0.2233283 and Q.e3 < 8.147744e-05:
        z += -232754.8 * (0.2233283 - Q.N2) * (8.147744e-05 - Q.e3)
    if Q.sum_z_dr2_top3 < 0.002151568 and Q.eccentricity > 0.9598562:
        z += -19277.29 * (0.002151568 - Q.sum_z_dr2_top3) * (Q.eccentricity - 0.9598562)
    if Q.planar_flow < 0.1950135 and Q.mean_phi > -0.01753483:
        z += -109.3517 * (0.1950135 - Q.planar_flow) * (Q.mean_phi - -0.01753483)
    if Q.sum_z_dr2 < 0.006679471 and Q.dr12 > 0.1587481:
        z += -35953.64 * (0.006679471 - Q.sum_z_dr2) * (Q.dr12 - 0.1587481)
    if Q.sum_zz_dr2 < 0.006390125 and Q.dr12 > 0.1587481:
        z += 31504.42 * (0.006390125 - Q.sum_zz_dr2) * (Q.dr12 - 0.1587481)
    if Q.sum_z_dr2 < 0.006679471 and Q.pt_entropy > 1.797618:
        z += -651.3728 * (0.006679471 - Q.sum_z_dr2) * (Q.pt_entropy - 1.797618)
    if Q.N2 < 0.2233283 and Q.pt_7 < 25.57812:
        z += -1.297006 * (0.2233283 - Q.N2) * (25.57812 - Q.pt_7)
    if Q.planar_flow < 0.1950135 and Q.z_7 < 0.07148865:
        z += -124.052 * (0.1950135 - Q.planar_flow) * (0.07148865 - Q.z_7)
    if Q.sum_z_dr2 < 0.01323868 and Q.centroid_offset > 0.04990367:
        z += -25137.48 * (0.01323868 - Q.sum_z_dr2) * (Q.centroid_offset - 0.04990367)
    if Q.planar_flow < 0.1950135 and Q.sum_pt_top5 > 402.625:
        z += 0.02850555 * (0.1950135 - Q.planar_flow) * (Q.sum_pt_top5 - 402.625)
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    return [B[c] + sum(h[j] * W[j][c] for j in range(len(h))) for c in range(5)]


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
