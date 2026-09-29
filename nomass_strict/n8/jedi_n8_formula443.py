"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.3% (the network: 65.8%); same class as the network for 89.5% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
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
  Q.girth                  pT-weighted mean ΔR
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        pt_2=pt[2],
        pt_5=pt[5],
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        z_top3_slots=sum(pt[:3]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        abseta_0=abs(eta[0]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
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
        girth=sum(z[i] * dr[i] for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -1.775728
    if Q.planar_flow < 0.1484197:
        z += -5.190878 * Q.planar_flow + 0.7704287
    if Q.lam1 < 0.004183811:
        z += 671.2434 * Q.lam1 + 0.4586386
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += -174.8085 * Q.lam1 + 3.99836
    if 0.00595415 <= Q.lam1 < 0.01200373:
        z += -422.759 * Q.lam1 + 5.474695
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += -90.29543 * Q.lam1 + 1.483893
    if Q.sum_pt < 763.825:
        z += 0.003845825 * Q.sum_pt - 2.937538
    if Q.tau1 < 0.05356915:
        z += -68.17223 * Q.tau1 + 3.651928
    if Q.girth2_top3 < 0.007929074:
        z += 65.85129 * Q.girth2_top3 - 0.5221397
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 24.16647 * Q.sj3_dr_max - 3.446505
    if 0.169029 <= Q.sj3_dr_max < 0.1986272:
        z += 5.018288 * Q.sj3_dr_max - 0.2099065
    if Q.sj3_dr_max >= 0.1986272:
        z += -13.85985 * Q.sj3_dr_max + 3.539805
    if Q.centroid_offset >= 0.006789738:
        z += -66.24236 * Q.centroid_offset + 0.4497683
    if Q.girth < 0.08065885:
        z += 57.87207 * Q.girth - 4.667895
    if Q.log_sum_pt >= 6.842717:
        z += -29.21842 * Q.log_sum_pt + 199.9334
    if Q.sum_pt_top5 >= 791.125:
        z += 0.01285608 * Q.sum_pt_top5 - 10.17076
    if Q.tau2 < 0.01713288:
        z += 129.2997 * Q.tau2 - 2.215276
    if Q.e3 < 8.147744e-05:
        z += -12864.09 * Q.e3 + 2.656416
    if 8.147744e-05 <= Q.e3 < 0.0005116989:
        z += -3738.267 * Q.e3 + 1.912867
    if 0.1546891 <= Q.LHA < 0.3033137:
        z += 5.867685 * Q.LHA - 0.907667
    if Q.LHA >= 0.3033137:
        z += -23.98413 * Q.LHA + 8.146799
    if Q.lam1 < 0.01643375 and Q.centroid_offset > 0.01837778:
        z += -3205.845 * (0.01643375 - Q.lam1) * (Q.centroid_offset - 0.01837778)
    if Q.sum_pt < 763.825 and Q.centroid_offset > 0.01096064:
        z += 0.2635974 * (763.825 - Q.sum_pt) * (Q.centroid_offset - 0.01096064)
    if Q.lam1 < 0.004183811 and Q.sum_pt < 763.825:
        z += 2.792217 * (0.004183811 - Q.lam1) * (763.825 - Q.sum_pt)
    if Q.sum_pt < 763.825 and Q.e3 < 0.0005116989:
        z += -11.02624 * (763.825 - Q.sum_pt) * (0.0005116989 - Q.e3)
    if Q.lam1 < 0.004183811 and Q.D2 < 0.7459513:
        z += -6725.221 * (0.004183811 - Q.lam1) * (0.7459513 - Q.D2)
    if Q.sj3_dr_max > 0.1426152 and Q.D2 > 0.2568137:
        z += 1.327653 * (Q.sj3_dr_max - 0.1426152) * (Q.D2 - 0.2568137)
    if Q.log_sum_pt > 6.842717 and Q.dr_2 < 0.03843804:
        z += 312.346 * (Q.log_sum_pt - 6.842717) * (0.03843804 - Q.dr_2)
    if Q.tau2 < 0.01713288 and Q.z_dr_0p05_0p1 < 0.2919447:
        z += 129.9101 * (0.01713288 - Q.tau2) * (0.2919447 - Q.z_dr_0p05_0p1)
    if Q.planar_flow < 0.04505724 and Q.tau21_b2 < 0.05932655:
        z += 367.628 * (0.04505724 - Q.planar_flow) * (0.05932655 - Q.tau21_b2)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.27682
    if Q.lam1 < 0.002464291:
        z += -923.9581 * Q.lam1 + 2.155429
    if 0.002464291 <= Q.lam1 < 0.005433361:
        z += -208.8348 * Q.lam1 + 0.393157
    if 0.005433361 <= Q.lam1 < 0.008375572:
        z += 252.0273 * Q.lam1 - 2.110873
    if Q.pt_7 >= 34.53125:
        z += 0.1006074 * Q.pt_7 - 3.474099
    if 6.377723 <= Q.log_sum_pt < 6.572938:
        z += 8.951631 * Q.log_sum_pt - 57.09102
    if Q.log_sum_pt >= 6.572938:
        z += 16.20739 * Q.log_sum_pt - 104.7827
    if Q.z_7 < 0.06473447:
        z += 92.25953 * Q.z_7 - 5.972372
    if Q.tau1 < 0.1027642:
        z += -7.327364 * Q.tau1 + 0.7529909
    if Q.sj2_dr < 0.1591713:
        z += -4.165994 * Q.sj2_dr + 0.6631066
    if Q.max_dr < 0.1117619:
        z += 10.23676 * Q.max_dr - 1.14408
    if Q.sum_pt >= 988.4078:
        z += -0.007064956 * Q.sum_pt + 6.983058
    if Q.centroid_offset < 0.02076709:
        z += 26.54515 * Q.centroid_offset - 0.5512655
    if Q.girth < 0.06108601:
        z += 38.44265 * Q.girth - 2.348308
    if Q.sum_pt_top5 >= 839.9547:
        z += 0.01401222 * Q.sum_pt_top5 - 11.76963
    if Q.zdr_0 < 0.0211821:
        z += 36.17612 * Q.zdr_0 - 0.7662864
    if Q.z_6 < 0.04737278:
        z += 123.9423 * Q.z_6 - 5.871489
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += 11558.11 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.log_sum_pt > 6.377723 and Q.e3 < 1.340118e-05:
        z += 168685.4 * (Q.log_sum_pt - 6.377723) * (1.340118e-05 - Q.e3)
    if Q.z_7 < 0.06473447 and Q.e3 < 0.0005116989:
        z += 114234.2 * (0.06473447 - Q.z_7) * (0.0005116989 - Q.e3)
    if Q.lam1 < 0.008375572 and Q.lam2 < 0.001130645:
        z += -194636.1 * (0.008375572 - Q.lam1) * (0.001130645 - Q.lam2)
    if Q.z_7 < 0.06473447 and Q.D2_b2 < 0.716559:
        z += -36.26324 * (0.06473447 - Q.z_7) * (0.716559 - Q.D2_b2)
    if Q.lam1 < 0.005433361 and Q.tau21_b2 > 0.04019753:
        z += -162.8234 * (0.005433361 - Q.lam1) * (Q.tau21_b2 - 0.04019753)
    if Q.z_7 < 0.06473447 and Q.girth2_top2 < 0.005180665:
        z += -2778.621 * (0.06473447 - Q.z_7) * (0.005180665 - Q.girth2_top2)
    if Q.z_6 < 0.04737278 and Q.girth2_top2 < 0.004007842:
        z += 15155.72 * (0.04737278 - Q.z_6) * (0.004007842 - Q.girth2_top2)
    if Q.log_sum_pt > 6.572938 and Q.girth2_top3 < 0.006756161:
        z += -1883.984 * (Q.log_sum_pt - 6.572938) * (0.006756161 - Q.girth2_top3)
    if Q.log_sum_pt > 6.572938 and Q.dr_max_012 < 0.1206357:
        z += 51.73827 * (Q.log_sum_pt - 6.572938) * (0.1206357 - Q.dr_max_012)
    if Q.lam1 < 0.01200373 and Q.dr_0 > 0.005517012:
        z += -1632.153 * (0.01200373 - Q.lam1) * (Q.dr_0 - 0.005517012)
    if Q.pt_7 > 34.53125 and Q.e3 < 8.147744e-05:
        z += -958.4165 * (Q.pt_7 - 34.53125) * (8.147744e-05 - Q.e3)
    if Q.sum_pt > 988.4078 and Q.girth2_top2 < 0.0095303:
        z += -2.556029 * (Q.sum_pt - 988.4078) * (0.0095303 - Q.girth2_top2)
    if Q.lam1 < 0.005433361 and Q.dr_max_012 < 0.1550922:
        z += -2094.4 * (0.005433361 - Q.lam1) * (0.1550922 - Q.dr_max_012)
    return max(0.0, z)


def neuron_2(Q):
    z = 5.690071
    if Q.pt_7 < 20.125:
        z += 0.1395632 * Q.pt_7 - 7.561412
    if 20.125 <= Q.pt_7 < 23.21641:
        z += 0.2612477 * Q.pt_7 - 10.01031
    if 23.21641 <= Q.pt_7 < 53.4375:
        z += 0.1733778 * Q.pt_7 - 7.970288
    if Q.pt_7 >= 53.4375:
        z += 0.07005355 * Q.pt_7 - 2.4489
    if Q.log_sum_pt < 6.080494:
        z += -10.45321 * Q.log_sum_pt + 69.5673
    if 6.080494 <= Q.log_sum_pt < 6.605974:
        z += -11.69392 * Q.log_sum_pt + 77.11142
    if 6.605974 <= Q.log_sum_pt < 6.701242:
        z += -6.632386 * Q.log_sum_pt + 43.67506
    if 6.701242 <= Q.log_sum_pt < 6.896095:
        z += -1.240708 * Q.log_sum_pt + 7.544121
    if Q.log_sum_pt >= 6.896095:
        z += 28.92817 * Q.log_sum_pt - 200.5033
    if Q.z_7 < 0.02320757:
        z += -49.68348 * Q.z_7 + 2.912133
    if 0.02320757 <= Q.z_7 < 0.02807091:
        z += -157.1818 * Q.z_7 + 5.406908
    if 0.02807091 <= Q.z_7 < 0.0586137:
        z += -130.3609 * Q.z_7 + 4.654021
    if Q.z_7 >= 0.0586137:
        z += -80.67746 * Q.z_7 + 1.741888
    if Q.LHA >= 0.111565:
        z += -9.832933 * Q.LHA + 1.097012
    if Q.sum_pt >= 988.4078:
        z += -0.02752486 * Q.sum_pt + 27.20578
    if Q.girth < 0.1019409:
        z += -11.43752 * Q.girth + 1.165951
    if Q.mean_phi2 < 8.836697e-05:
        z += -60451.97 * Q.mean_phi2 + 5.341958
    if Q.lam1 < 0.00733008 and Q.max_dr < 0.2507612:
        z += 661.8376 * (0.00733008 - Q.lam1) * (0.2507612 - Q.max_dr)
    if Q.lam1 < 0.00733008 and Q.pt_6 < 62.25:
        z += -2.287117 * (0.00733008 - Q.lam1) * (62.25 - Q.pt_6)
    if Q.log_sum_pt < 6.701242 and Q.mean_phi2 < 8.836697e-05:
        z += -104574.3 * (6.701242 - Q.log_sum_pt) * (8.836697e-05 - Q.mean_phi2)
    if Q.log_sum_pt > 6.080494 and Q.mean_phi2 < 8.836697e-05:
        z += -95512.26 * (Q.log_sum_pt - 6.080494) * (8.836697e-05 - Q.mean_phi2)
    if Q.log_sum_pt > 6.896095 and Q.absphi_0 < 0.1056549:
        z += 103.3703 * (Q.log_sum_pt - 6.896095) * (0.1056549 - Q.absphi_0)
    if Q.sum_pt > 937.0312 and Q.mean_phi2 < 8.836697e-05:
        z += 116.0926 * (Q.sum_pt - 937.0312) * (8.836697e-05 - Q.mean_phi2)
    if Q.z_7 < 0.0586137 and Q.abseta_0 < 0.1057739:
        z += -284.1499 * (0.0586137 - Q.z_7) * (0.1057739 - Q.abseta_0)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.7225232
    if 0.06663269 <= Q.girth < 0.1019409:
        z += 42.10459 * Q.girth - 2.805542
    if 0.1019409 <= Q.girth < 0.1245537:
        z += -98.83184 * Q.girth + 11.56165
    if Q.girth >= 0.1245537:
        z += 55.54428 * Q.girth - 7.666476
    if Q.sj2_dr >= 0.1682655:
        z += 24.69453 * Q.sj2_dr - 4.155238
    if 0.1876504 <= Q.sd_rg < 0.2787955:
        z += 12.29705 * Q.sd_rg - 2.307547
    if Q.sd_rg >= 0.2787955:
        z += -18.12258 * Q.sd_rg + 6.173311
    if Q.max_dr >= 0.2507612:
        z += -15.80555 * Q.max_dr + 3.963419
    if 0.2931906 <= Q.LHA < 0.3255822:
        z += 18.88291 * Q.LHA - 5.536292
    if Q.LHA >= 0.3255822:
        z += -2.771948 * Q.LHA + 1.514145
    if Q.tau1 >= 0.1136369:
        z += 23.0884 * Q.tau1 - 2.623694
    if Q.sj3_dr_max >= 0.213399:
        z += 15.35583 * Q.sj3_dr_max - 3.276919
    if Q.mean_eta < -0.006779839:
        z += -32.73939 * Q.mean_eta - 0.2219678
    if 0.00595415 <= Q.lam1 < 0.01200373:
        z += -260.7077 * Q.lam1 + 1.552293
    if Q.lam1 >= 0.01200373:
        z += -387.2824 * Q.lam1 + 3.071661
    if Q.lam2 < 0.003408389:
        z += 1087.137 * Q.lam2 - 3.705386
    if Q.sj2_dr > 0.1682655 and Q.log_sum_pt > 6.080494:
        z += -19.31228 * (Q.sj2_dr - 0.1682655) * (Q.log_sum_pt - 6.080494)
    if Q.sj2_dr > 0.1682655 and Q.C2 < 0.09482124:
        z += 223.7174 * (Q.sj2_dr - 0.1682655) * (0.09482124 - Q.C2)
    if Q.sj2_dr > 0.1682655 and Q.n_dr_0_0p05 > 2.0:
        z += -5.580613 * (Q.sj2_dr - 0.1682655) * (Q.n_dr_0_0p05 - 2.0)
    if Q.sj3_dr_max > 0.213399 and Q.n_dr_0p05_0p1 > 4.0:
        z += 9.216686 * (Q.sj3_dr_max - 0.213399) * (Q.n_dr_0p05_0p1 - 4.0)
    if Q.sj3_dr_max > 0.2623172 and Q.n_dr_0p05_0p1 > 4.0:
        z += -21.98531 * (Q.sj3_dr_max - 0.2623172) * (Q.n_dr_0p05_0p1 - 4.0)
    if Q.mean_eta < -0.006779839 and Q.eta_0 < -0.03967285:
        z += -427.4155 * (-0.006779839 - Q.mean_eta) * (-0.03967285 - Q.eta_0)
    if Q.girth > 0.1245537 and Q.eccentricity > 0.7792127:
        z += -220.2848 * (Q.girth - 0.1245537) * (Q.eccentricity - 0.7792127)
    if Q.girth > 0.06663269 and Q.eccentricity > 0.7117266:
        z += 221.8232 * (Q.girth - 0.06663269) * (Q.eccentricity - 0.7117266)
    if Q.sj2_dr > 0.1682655 and Q.eccentricity > 0.9458207:
        z += 363.6396 * (Q.sj2_dr - 0.1682655) * (Q.eccentricity - 0.9458207)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.651114
    if Q.N2 < 0.2233283:
        z += -9.752797 * Q.N2 + 2.178075
    if Q.lam1 < 0.002464291:
        z += 436.5112 * Q.lam1 - 5.869465
    if 0.002464291 <= Q.lam1 < 0.006506576:
        z += 965.9156 * Q.lam1 - 7.174071
    if 0.006506576 <= Q.lam1 < 0.00733008:
        z += 503.7879 * Q.lam1 - 4.167202
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 453.7545 * Q.lam1 - 3.800454
    if Q.e3 < 0.0001869378:
        z += -28833.0 * Q.e3 + 5.389978
    if 0.169029 <= Q.sj3_dr_max < 0.233678:
        z += 22.42849 * Q.sj3_dr_max - 3.791066
    if Q.sj3_dr_max >= 0.233678:
        z += 0.6267281 * Q.sj3_dr_max + 1.303526
    if Q.lam2 < 0.000537286:
        z += 2999.78 * Q.lam2 - 1.61174
    if Q.sum_pt < 763.825:
        z += 0.01016894 * Q.sum_pt - 7.767293
    if Q.pt_7 < 25.57812:
        z += 0.2156971 * Q.pt_7 - 5.517128
    if Q.sj2_dr >= 0.2179769:
        z += -21.35201 * Q.sj2_dr + 4.654246
    if Q.C2 >= 0.01867771:
        z += -25.93117 * Q.C2 + 0.484335
    if Q.LHA < 0.3127275:
        z += -21.01192 * Q.LHA + 6.571005
    if Q.girth2_top3 < 0.01002369:
        z += 92.90618 * Q.girth2_top3 - 0.9312627
    if Q.sj3_dr_min < 0.2089872:
        z += -31.14712 * Q.sj3_dr_min + 6.50935
    if Q.N2 < 0.2233283 and Q.planar_flow > 0.08366273:
        z += 31.75316 * (0.2233283 - Q.N2) * (Q.planar_flow - 0.08366273)
    if Q.N2 < 0.2233283 and Q.sj2_zsoft < 0.2832687:
        z += -61.58624 * (0.2233283 - Q.N2) * (0.2832687 - Q.sj2_zsoft)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1626038:
        z += -55.28769 * (0.2233283 - Q.N2) * (0.1626038 - Q.abseta_7)
    if Q.sj3_dr_max > 0.233678 and Q.log_sum_pt > 6.267538:
        z += -69.15436 * (Q.sj3_dr_max - 0.233678) * (Q.log_sum_pt - 6.267538)
    if Q.sum_pt < 763.825 and Q.M3 < 0.107953:
        z += -0.04336205 * (763.825 - Q.sum_pt) * (0.107953 - Q.M3)
    if Q.sj3_dr_max > 0.233678 and Q.sj2_zsoft < 0.05966518:
        z += 1007.753 * (Q.sj3_dr_max - 0.233678) * (0.05966518 - Q.sj2_zsoft)
    if Q.log_sum_pt > 6.327379 and Q.ptdr0_6 < 7.998907:
        z += -0.1959752 * (Q.log_sum_pt - 6.327379) * (7.998907 - Q.ptdr0_6)
    if Q.sj2_dr > 0.2179769 and Q.C2_b2 < 0.0006435798:
        z += -41634.93 * (Q.sj2_dr - 0.2179769) * (0.0006435798 - Q.C2_b2)
    if Q.max_dr > 0.121681 and Q.C2_b2 < 0.009032972:
        z += 2134.115 * (Q.max_dr - 0.121681) * (0.009032972 - Q.C2_b2)
    if Q.N2 < 0.2233283 and Q.lam2 > 0.001130645:
        z += -15107.97 * (0.2233283 - Q.N2) * (Q.lam2 - 0.001130645)
    if Q.dr_0 < 0.08082334 and Q.centroid_offset > 0.009480685:
        z += 1453.736 * (0.08082334 - Q.dr_0) * (Q.centroid_offset - 0.009480685)
    if Q.log_sum_pt > 6.327379 and Q.n_dr_0p05_0p1 < 7.0:
        z += -0.2399809 * (Q.log_sum_pt - 6.327379) * (7.0 - Q.n_dr_0p05_0p1)
    if Q.log_sum_pt > 6.327379 and Q.zdr_2 < 0.009799324:
        z += -319.3334 * (Q.log_sum_pt - 6.327379) * (0.009799324 - Q.zdr_2)
    if Q.sj3_dr_min < 0.2089872 and Q.lam2 < 0.003408389:
        z += -9980.838 * (0.2089872 - Q.sj3_dr_min) * (0.003408389 - Q.lam2)
    if Q.sj2_dr > 0.2179769 and Q.lam2 > 0.001130645:
        z += -1428.583 * (Q.sj2_dr - 0.2179769) * (Q.lam2 - 0.001130645)
    if Q.pt_7 < 25.57812 and Q.eccentricity > 0.9458207:
        z += 2.37998 * (25.57812 - Q.pt_7) * (Q.eccentricity - 0.9458207)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.34797
    if Q.LHA < 0.2160559:
        z += -31.32218 * Q.LHA + 6.76734
    if Q.e2 < 0.03556091:
        z += 48.72454 * Q.e2 - 1.732689
    if Q.log_sum_pt >= 6.896095:
        z += -10.66948 * Q.log_sum_pt + 73.57773
    if Q.z_6 < 0.02886576:
        z += -78.25258 * Q.z_6 + 1.333022
    if 0.02886576 <= Q.z_6 < 0.06081235:
        z += 28.97955 * Q.z_6 - 1.762314
    if Q.lam1 < 7.12483e-05:
        z += -31810.67 * Q.lam1 + 5.068694
    if 7.12483e-05 <= Q.lam1 < 0.0008722282:
        z += -1896.543 * Q.lam1 + 2.937364
    if 0.0008722282 <= Q.lam1 < 0.002464291:
        z += -675.5475 * Q.lam1 + 1.872377
    if 0.002464291 <= Q.lam1 < 0.00483998:
        z += -87.39835 * Q.lam1 + 0.4230063
    if Q.tau1 < 0.0262518:
        z += 48.42632 * Q.tau1 - 1.271278
    if Q.girth < 0.02054282:
        z += 55.06875 * Q.girth - 1.131267
    if Q.sum_pt < 739.5:
        z += 0.002962868 * Q.sum_pt - 2.191041
    if Q.n_pt_above_50 >= 6.0:
        z += -0.426039 * Q.n_pt_above_50 + 2.556234
    if Q.z_7 < 0.02807091:
        z += -66.94801 * Q.z_7 + 1.879292
    if Q.z_5 < 0.02818362:
        z += -99.63943 * Q.z_5 + 2.8082
    if Q.z_top5_slots >= 0.908903:
        z += -24.29948 * Q.z_top5_slots + 22.08587
    if Q.dr_0 < 0.06413297:
        z += 28.94524 * Q.dr_0 - 1.856344
    if Q.sj3_dr_max < 0.3012016:
        z += 2.390591 * Q.sj3_dr_max - 0.7200498
    if Q.centroid_offset < 0.03776099:
        z += 24.06214 * Q.centroid_offset - 0.9086102
    if Q.LHA < 0.2160559 and Q.z_6 < 0.09543973:
        z += -361.2739 * (0.2160559 - Q.LHA) * (0.09543973 - Q.z_6)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -155.4013 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.e2 < 0.03556091 and Q.z_7 < 0.0753896:
        z += 1390.34 * (0.03556091 - Q.e2) * (0.0753896 - Q.z_7)
    if Q.lam1 < 0.002464291 and Q.centroid_offset < 0.01837778:
        z += 52783.07 * (0.002464291 - Q.lam1) * (0.01837778 - Q.centroid_offset)
    if Q.girth < 0.02054282 and Q.sum_pt_top5 > 716.8828:
        z += 0.6081532 * (0.02054282 - Q.girth) * (Q.sum_pt_top5 - 716.8828)
    if Q.z_6 < 0.06081235 and Q.pt1_dr01 < 28.39396:
        z += 0.9539639 * (0.06081235 - Q.z_6) * (28.39396 - Q.pt1_dr01)
    if Q.log_sum_pt > 6.804164 and Q.e3 < 5.334511e-05:
        z += -245602.3 * (Q.log_sum_pt - 6.804164) * (5.334511e-05 - Q.e3)
    if Q.e2 < 0.03556091 and Q.lam2 < 0.0003061234:
        z += 291058.6 * (0.03556091 - Q.e2) * (0.0003061234 - Q.lam2)
    if Q.z_6 < 0.06081235 and Q.e3 < 0.0005116989:
        z += 62520.88 * (0.06081235 - Q.z_6) * (0.0005116989 - Q.e3)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.007579
    if 0.1591713 <= Q.sj2_dr < 0.1682655:
        z += -7.862102 * Q.sj2_dr + 1.251421
    if Q.sj2_dr >= 0.1682655:
        z += 0.9500051 * Q.sj2_dr - 0.231353
    if Q.centroid_offset < 0.01837778:
        z += 86.42148 * Q.centroid_offset - 1.588235
    if Q.centroid_offset >= 0.02076709:
        z += 41.60769 * Q.centroid_offset - 0.8640706
    if 0.08723651 <= Q.girth < 0.1019409:
        z += -269.3401 * Q.girth + 23.49629
    if 0.1019409 <= Q.girth < 0.1484084:
        z += -327.4872 * Q.girth + 29.42387
    if Q.girth >= 0.1484084:
        z += -297.1104 * Q.girth + 24.91569
    if Q.pt_6 < 39.75:
        z += 0.2668573 * Q.pt_6 - 10.60758
    if Q.lam2 < 0.001130645:
        z += 1912.872 * Q.lam2 - 1.337832
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -362.1772 * Q.lam2 + 1.234441
    if 0.251526 <= Q.LHA < 0.3255822:
        z += 34.50143 * Q.LHA - 8.678007
    if Q.LHA >= 0.3255822:
        z += 110.5571 * Q.LHA - 33.44039
    if Q.psi_0p2 >= 0.8990266:
        z += -6.990764 * Q.psi_0p2 + 6.284883
    if Q.C2 < 0.03578649:
        z += -79.10716 * Q.C2 + 2.830967
    if 0.03464708 <= Q.sj3_dr_max < 0.1789613:
        z += -18.57771 * Q.sj3_dr_max + 0.6436635
    if Q.sj3_dr_max >= 0.1789613:
        z += 7.188498 * Q.sj3_dr_max - 3.967492
    if Q.z_6 < 0.04355037:
        z += -243.1972 * Q.z_6 + 10.59133
    if Q.max_dr < 0.177305:
        z += 29.77794 * Q.max_dr - 5.279777
    if Q.dr_0 < 0.08082334:
        z += -11.70787 * Q.dr_0 + 0.9462692
    if Q.log_sum_pt < 6.423044 and Q.pt_7 < 38.53125:
        z += 0.4085918 * (6.423044 - Q.log_sum_pt) * (38.53125 - Q.pt_7)
    if Q.centroid_offset > 0.00809236 and Q.N2 > 0.198361:
        z += 212.9223 * (Q.centroid_offset - 0.00809236) * (Q.N2 - 0.198361)
    if Q.sj2_dr > 0.1591713 and Q.sj2_zsoft > 0.05966518:
        z += 105.3813 * (Q.sj2_dr - 0.1591713) * (Q.sj2_zsoft - 0.05966518)
    if Q.centroid_offset < 0.01837778 and Q.planar_flow < 0.1115136:
        z += -514.2471 * (0.01837778 - Q.centroid_offset) * (0.1115136 - Q.planar_flow)
    if Q.pt_6 < 39.75 and Q.sum_pt_top3 < 711.875:
        z += 0.001539962 * (39.75 - Q.pt_6) * (711.875 - Q.sum_pt_top3)
    if Q.centroid_offset > 0.00809236 and Q.sj3_dr_min < 0.2089872:
        z += 233.205 * (Q.centroid_offset - 0.00809236) * (0.2089872 - Q.sj3_dr_min)
    if Q.centroid_offset > 0.00809236 and Q.mean_phi2 < 0.01426135:
        z += 3984.27 * (Q.centroid_offset - 0.00809236) * (0.01426135 - Q.mean_phi2)
    if Q.sj2_dr > 0.1591713 and Q.mean_eta > 0.02644207:
        z += -400.2851 * (Q.sj2_dr - 0.1591713) * (Q.mean_eta - 0.02644207)
    if Q.centroid_offset > 0.02076709 and Q.n_dr_0_0p05 < 5.0:
        z += -6.089464 * (Q.centroid_offset - 0.02076709) * (5.0 - Q.n_dr_0_0p05)
    if Q.sj3_dr_max > 0.1789613 and Q.D2_b2 < 1.59265:
        z += 13.8568 * (Q.sj3_dr_max - 0.1789613) * (1.59265 - Q.D2_b2)
    if Q.pt_6 < 39.75 and Q.D2_b2 < 4.721224:
        z += -0.01303336 * (39.75 - Q.pt_6) * (4.721224 - Q.D2_b2)
    if Q.pt_6 < 39.75 and Q.z_top3_slots < 0.7861046:
        z += -0.9539409 * (39.75 - Q.pt_6) * (0.7861046 - Q.z_top3_slots)
    if Q.lam2 < 0.003408389 and Q.n_dr_0p1_0p2 > 1.0:
        z += 50.62857 * (0.003408389 - Q.lam2) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.centroid_offset < 0.01837778 and Q.n_dr_0p05_0p1 > 0.0:
        z += 17.25766 * (0.01837778 - Q.centroid_offset) * (Q.n_dr_0p05_0p1 - 0.0)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.8239093
    if Q.planar_flow < 0.1950135:
        z += -8.669836 * Q.planar_flow + 1.690735
    if Q.girth < 0.0479157:
        z += 18.83679 * Q.girth - 1.150665
    if 0.0479157 <= Q.girth < 0.06108601:
        z += -33.24616 * Q.girth + 1.344927
    if 0.06108601 <= Q.girth < 0.08065885:
        z += -52.08296 * Q.girth + 2.495591
    if 0.08065885 <= Q.girth < 0.1019409:
        z += -191.8384 * Q.girth + 13.7681
    if 0.1019409 <= Q.girth < 0.1245537:
        z += 140.8516 * Q.girth - 20.14662
    if Q.girth >= 0.1245537:
        z += -387.4602 * Q.girth + 45.65658
    if Q.tau1 < 0.06345984:
        z += -69.23016 * Q.tau1 + 6.13751
    if 0.06345984 <= Q.tau1 < 0.08865369:
        z += -43.28133 * Q.tau1 + 4.490801
    if 0.08865369 <= Q.tau1 < 0.1136369:
        z += 25.94883 * Q.tau1 - 1.646709
    if Q.tau1 >= 0.1136369:
        z += 56.53016 * Q.tau1 - 5.121876
    if Q.z_dr_0p1_0p2 < 0.1585582:
        z += -3.785 * Q.z_dr_0p1_0p2 + 0.6001428
    if Q.lam1 < 0.002464291:
        z += 1328.431 * Q.lam1 - 5.624546
    if 0.002464291 <= Q.lam1 < 0.003377388:
        z += 1358.099 * Q.lam1 - 5.697656
    if 0.003377388 <= Q.lam1 < 0.00483998:
        z += 807.6829 * Q.lam1 - 3.838688
    if 0.00483998 <= Q.lam1 < 0.00733008:
        z += 29.66795 * Q.lam1 - 0.07311047
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += 161.9237 * Q.lam1 - 1.042555
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 1013.434 * Q.lam1 - 11.26386
    if Q.lam1 >= 0.01643375:
        z += -2351.27 * Q.lam1 + 44.03086
    if 0.1329373 <= Q.LHA < 0.3467135:
        z += 10.79292 * Q.LHA - 1.434782
    if Q.LHA >= 0.3467135:
        z += -305.7274 * Q.LHA + 108.3071
    if Q.girth2_top5 < 0.0006570502:
        z += 1187.255 * Q.girth2_top5 - 0.7800864
    if Q.e3 < 4.192959e-06:
        z += -87781.92 * Q.e3 - 2.326248
    if 4.192959e-06 <= Q.e3 < 8.147744e-05:
        z += 34862.29 * Q.e3 - 2.84049
    if Q.centroid_offset < 0.02685622:
        z += -69.19729 * Q.centroid_offset + 1.858378
    if Q.N2 >= 0.1333619:
        z += 4.83325 * Q.N2 - 0.6445712
    if Q.sj3_dr_max < 0.1594012:
        z += -17.86351 * Q.sj3_dr_max + 2.316181
    if 0.1594012 <= Q.sj3_dr_max < 0.213399:
        z += 9.838955 * Q.sj3_dr_max - 2.099623
    if Q.max_dr < 0.04656688:
        z += 36.53922 * Q.max_dr - 1.701518
    if Q.planar_flow < 0.1950135 and Q.lam1 > 0.008375572:
        z += -6380.452 * (0.1950135 - Q.planar_flow) * (Q.lam1 - 0.008375572)
    if Q.z_dr_0p1_0p2 < 0.1585582 and Q.centroid_offset < 0.02685622:
        z += -267.3069 * (0.1585582 - Q.z_dr_0p1_0p2) * (0.02685622 - Q.centroid_offset)
    if Q.planar_flow < 0.1950135 and Q.pt_6 < 56.53125:
        z += -0.2064063 * (0.1950135 - Q.planar_flow) * (56.53125 - Q.pt_6)
    if Q.planar_flow < 0.1950135 and Q.log_sum_pt > 6.46415:
        z += 28.20971 * (0.1950135 - Q.planar_flow) * (Q.log_sum_pt - 6.46415)
    if Q.lam1 > 0.002464291 and Q.sj3_dr_max < 0.1879486:
        z += -10746.69 * (Q.lam1 - 0.002464291) * (0.1879486 - Q.sj3_dr_max)
    if Q.planar_flow < 0.1950135 and Q.max_dr < 0.2507612:
        z += -41.32994 * (0.1950135 - Q.planar_flow) * (0.2507612 - Q.max_dr)
    if Q.girth > 0.08065885 and Q.sj2_zsoft < 0.09384951:
        z += 7989.17 * (Q.girth - 0.08065885) * (0.09384951 - Q.sj2_zsoft)
    if Q.planar_flow < 0.1950135 and Q.D3 < 3.916009:
        z += -1.074394 * (0.1950135 - Q.planar_flow) * (3.916009 - Q.D3)
    if Q.lam1 > 0.00733008 and Q.D2 < 2.843757:
        z += -156.2179 * (Q.lam1 - 0.00733008) * (2.843757 - Q.D2)
    if Q.lam1 > 0.002464291 and Q.planar_flow < 0.1950135:
        z += 2133.907 * (Q.lam1 - 0.002464291) * (0.1950135 - Q.planar_flow)
    if Q.lam1 > 0.002464291 and Q.D2 > 0.2568137:
        z += -273.4036 * (Q.lam1 - 0.002464291) * (Q.D2 - 0.2568137)
    if Q.lam1 > 0.01643375 and Q.D2 < 3.885568:
        z += 340.5713 * (Q.lam1 - 0.01643375) * (3.885568 - Q.D2)
    if Q.girth > 0.0479157 and Q.D2 < 2.357246:
        z += 53.483 * (Q.girth - 0.0479157) * (2.357246 - Q.D2)
    if Q.e3 < 8.147744e-05 and Q.pt_7 < 43.5:
        z += -447.9645 * (8.147744e-05 - Q.e3) * (43.5 - Q.pt_7)
    if Q.centroid_offset < 0.02685622 and Q.n_dr_0p2_0p4 < 1.0:
        z += -88.47433 * (0.02685622 - Q.centroid_offset) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.centroid_offset < 0.04990367 and Q.n_dr_0p2_0p4 < 2.0:
        z += 13.79283 * (0.04990367 - Q.centroid_offset) * (2.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_8(Q):
    z = -1.009257
    if Q.tau1 < 0.019014:
        z += 166.8462 * Q.tau1 - 3.897819
    if 0.019014 <= Q.tau1 < 0.0262518:
        z += 100.2246 * Q.tau1 - 2.631076
    if Q.LHA < 0.1767241:
        z += -7.703804 * Q.LHA + 0.6668203
    if 0.1767241 <= Q.LHA < 0.251526:
        z += 9.286232 * Q.LHA - 2.335729
    if Q.sum_pt_top5 >= 658.125:
        z += 0.007582045 * Q.sum_pt_top5 - 4.989933
    if Q.log_sum_pt >= 6.701242:
        z += -15.25623 * Q.log_sum_pt + 102.2357
    if Q.lam1 < 0.0002758826:
        z += -5272.979 * Q.lam1 + 1.454723
    if Q.sj3_dr_max < 0.1070199:
        z += 84.5636 * Q.sj3_dr_max - 7.33501
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += -36.43139 * Q.sj3_dr_max + 5.613867
    if 0.1426152 <= Q.sj3_dr_max < 0.1594012:
        z += -1.199738 * Q.sj3_dr_max + 0.5892989
    if 0.1594012 <= Q.sj3_dr_max < 0.1986272:
        z += -10.14783 * Q.sj3_dr_max + 2.015635
    if Q.girth < 0.03360421:
        z += 60.67111 * Q.girth - 2.038805
    if Q.lam2 < 9.651826e-05:
        z += -8995.016 * Q.lam2 + 0.8681832
    if Q.centroid_offset >= 0.006789738:
        z += -94.54847 * Q.centroid_offset + 0.6419594
    if Q.lam1 < 0.00733008 and Q.centroid_offset < 0.02355416:
        z += 3888.052 * (0.00733008 - Q.lam1) * (0.02355416 - Q.centroid_offset)
    if Q.lam1 < 0.00733008 and Q.planar_flow < 0.5925897:
        z += 278.799 * (0.00733008 - Q.lam1) * (0.5925897 - Q.planar_flow)
    if Q.LHA < 0.251526 and Q.z_7 > 0.01685855:
        z += -340.6316 * (0.251526 - Q.LHA) * (Q.z_7 - 0.01685855)
    if Q.sj3_dr_max < 0.1986272 and Q.n_dr_0p05_0p1 < 5.0:
        z += 0.5444136 * (0.1986272 - Q.sj3_dr_max) * (5.0 - Q.n_dr_0p05_0p1)
    if Q.LHA < 0.1767241 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -911.2035 * (0.1767241 - Q.LHA) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.lam1 < 0.005433361 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += 11989.82 * (0.005433361 - Q.lam1) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.tau1 < 0.0262518 and Q.centroid_offset < 0.03117077:
        z += 9400.735 * (0.0262518 - Q.tau1) * (0.03117077 - Q.centroid_offset)
    if Q.lam1 < 0.005433361 and Q.pt_7 < 38.53125:
        z += -10.71445 * (0.005433361 - Q.lam1) * (38.53125 - Q.pt_7)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 < 33.21875:
        z += 0.2122467 * (Q.log_sum_pt - 6.701242) * (33.21875 - Q.pt_7)
    if Q.girth < 0.03360421 and Q.pt_7 > 15.55391:
        z += 3.871722 * (0.03360421 - Q.girth) * (Q.pt_7 - 15.55391)
    if Q.tau1 < 0.019014 and Q.pt_7 > 20.125:
        z += -2.299411 * (0.019014 - Q.tau1) * (Q.pt_7 - 20.125)
    if Q.girth < 0.03360421 and Q.centroid_offset < 0.03117077:
        z += -1334.203 * (0.03360421 - Q.girth) * (0.03117077 - Q.centroid_offset)
    if Q.lam1 < 0.005433361 and Q.centroid_offset < 0.03117077:
        z += 8193.178 * (0.005433361 - Q.lam1) * (0.03117077 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.1070199 and Q.centroid_offset > 0.01627885:
        z += -5309.215 * (0.1070199 - Q.sj3_dr_max) * (Q.centroid_offset - 0.01627885)
    if Q.sj3_dr_max < 0.1594012 and Q.centroid_offset > 0.003343241:
        z += 375.3624 * (0.1594012 - Q.sj3_dr_max) * (Q.centroid_offset - 0.003343241)
    if Q.sj3_dr_max < 0.1070199 and Q.girth2_top3 < 0.0153634:
        z += 7861.189 * (0.1070199 - Q.sj3_dr_max) * (0.0153634 - Q.girth2_top3)
    if Q.sj3_dr_max < 0.1426152 and Q.girth2_top3 < 0.0153634:
        z += -3341.175 * (0.1426152 - Q.sj3_dr_max) * (0.0153634 - Q.girth2_top3)
    return max(0.0, z)


def neuron_9(Q):
    z = -5.339058
    if Q.girth < 0.04081947:
        z += 28.96749 * Q.girth - 1.374489
    if 0.04081947 <= Q.girth < 0.05464922:
        z += 13.8868 * Q.girth - 0.7589029
    if Q.lam1 < 0.0001413912:
        z += -8607.301 * Q.lam1 + 4.452242
    if 0.0001413912 <= Q.lam1 < 0.003377388:
        z += -955.2999 * Q.lam1 + 3.370316
    if 0.003377388 <= Q.lam1 < 0.00595415:
        z += -55.84424 * Q.lam1 + 0.332505
    if Q.sum_pt < 813.4156:
        z += -0.0102275 * Q.sum_pt + 9.106923
    if 813.4156 <= Q.sum_pt < 988.4078:
        z += -0.004501448 * Q.sum_pt + 4.449266
    if Q.lam2 < 0.0001947983:
        z += -6056.927 * Q.lam2 + 1.179879
    if Q.lam2 >= 0.000537286:
        z += 1265.554 * Q.lam2 - 0.6799644
    if Q.e2 < 0.0285317:
        z += 78.40217 * Q.e2 - 2.236947
    if Q.sj2_dr < 0.1294903:
        z += 12.34078 * Q.sj2_dr - 1.598011
    if Q.centroid_offset < 0.02355416:
        z += -59.31664 * Q.centroid_offset + 2.489538
    if 0.02355416 <= Q.centroid_offset < 0.03117077:
        z += -143.4214 * Q.centroid_offset + 4.470554
    if Q.sj3_dr_max < 0.1426152:
        z += 18.99914 * Q.sj3_dr_max - 1.761693
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -13.3911 * Q.sj3_dr_max + 2.857646
    if Q.e3 < 1.960701e-05:
        z += 43963.99 * Q.e3 - 0.8620024
    if Q.max_dr < 0.1027585:
        z += -15.68071 * Q.max_dr + 1.611327
    if Q.pt1_dr01 >= 22.38578:
        z += -0.07002275 * Q.pt1_dr01 + 1.567514
    if Q.lam1 < 0.00595415 and Q.lam2 < 0.001130645:
        z += 1144336.0 * (0.00595415 - Q.lam1) * (0.001130645 - Q.lam2)
    if Q.centroid_offset < 0.01627885 and Q.D2_b2 < 0.1830092:
        z += -649.7657 * (0.01627885 - Q.centroid_offset) * (0.1830092 - Q.D2_b2)
    if Q.e3 < 4.192959e-06 and Q.n_dr_0p05_0p1 > 3.0:
        z += -441865.4 * (4.192959e-06 - Q.e3) * (Q.n_dr_0p05_0p1 - 3.0)
    if Q.sum_pt < 988.4078 and Q.z_5 < 0.05753583:
        z += 0.3002694 * (988.4078 - Q.sum_pt) * (0.05753583 - Q.z_5)
    if Q.centroid_offset < 0.01627885 and Q.sj3_dr_min < 0.1278212:
        z += 1150.048 * (0.01627885 - Q.centroid_offset) * (0.1278212 - Q.sj3_dr_min)
    if Q.lam1 < 0.00595415 and Q.sj3_dr_min < 0.08543881:
        z += -3846.911 * (0.00595415 - Q.lam1) * (0.08543881 - Q.sj3_dr_min)
    if Q.centroid_offset < 0.02355416 and Q.lam1 < 0.00733008:
        z += 25923.01 * (0.02355416 - Q.centroid_offset) * (0.00733008 - Q.lam1)
    if Q.lam2 > 0.000537286 and Q.eccentricity < 0.9979797:
        z += -1239.198 * (Q.lam2 - 0.000537286) * (0.9979797 - Q.eccentricity)
    return max(0.0, z)


def neuron_10(Q):
    z = 3.465799
    if Q.lam1 < 0.003377388:
        z += 832.7787 * Q.lam1 - 1.355939
    if 0.003377388 <= Q.lam1 < 0.00483998:
        z += 431.303 * Q.lam1
    if Q.lam1 >= 0.00483998:
        z += 206.811 * Q.lam1 + 1.086537
    if 0.0001947983 <= Q.lam2 < 0.003408389:
        z += 1733.546 * Q.lam2 - 0.3376919
    if Q.lam2 >= 0.003408389:
        z += 401.9742 * Q.lam2 + 4.200824
    if Q.centroid_offset >= 0.02355416:
        z += 22.9405 * Q.centroid_offset - 0.5403442
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.763492 * Q.n_dr_0p2_0p4 - 0.763492
    if Q.z_7 < 0.06810151:
        z += 22.34102 * Q.z_7 - 1.521457
    if Q.tau1 >= 0.04369778:
        z += 19.48569 * Q.tau1 - 0.8514816
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += 15.73321 * Q.sj2_dr - 2.037298
    if 0.1591713 <= Q.sj2_dr < 0.1682655:
        z += -7.848432 * Q.sj2_dr + 1.716222
    if 0.1682655 <= Q.sj2_dr < 0.3003793:
        z += -0.6325598 * Q.sj2_dr + 0.5020399
    if Q.sj2_dr >= 0.3003793:
        z += 12.65257 * Q.sj2_dr - 3.488539
    if Q.girth2_top3 < 0.002915531:
        z += -281.7212 * Q.girth2_top3 + 0.8213668
    if Q.LHA >= 0.1967397:
        z += -20.02225 * Q.LHA + 3.939172
    if Q.girth < 0.1484084:
        z += 21.91124 * Q.girth - 3.251813
    if Q.lam2 > 0.0001947983 and Q.log_sum_pt < 6.538559:
        z += -918.1647 * (Q.lam2 - 0.0001947983) * (6.538559 - Q.log_sum_pt)
    if Q.lam2 > 0.0001947983 and Q.eccentricity > 0.4829602:
        z += -1618.676 * (Q.lam2 - 0.0001947983) * (Q.eccentricity - 0.4829602)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.D2_b2 < 4.721224:
        z += -0.2068755 * (Q.n_dr_0p2_0p4 - 1.0) * (4.721224 - Q.D2_b2)
    if Q.lam2 > 0.0001947983 and Q.planar_flow > 0.06126205:
        z += -894.6222 * (Q.lam2 - 0.0001947983) * (Q.planar_flow - 0.06126205)
    if Q.sj2_dr > 0.1682655 and Q.planar_flow < 0.6947818:
        z += -24.97112 * (Q.sj2_dr - 0.1682655) * (0.6947818 - Q.planar_flow)
    if Q.LHA > 0.3127275 and Q.planar_flow < 0.6947818:
        z += -41.46997 * (Q.LHA - 0.3127275) * (0.6947818 - Q.planar_flow)
    if Q.lam2 > 0.003408389 and Q.pt_2 < 111.75:
        z += 11.43254 * (Q.lam2 - 0.003408389) * (111.75 - Q.pt_2)
    if Q.lam2 > 0.003408389 and Q.pt2_over_pt0 > 0.1852611:
        z += 671.817 * (Q.lam2 - 0.003408389) * (Q.pt2_over_pt0 - 0.1852611)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.3483586
    if Q.planar_flow < 0.2534037:
        z += -12.84151 * Q.planar_flow + 3.254087
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += 19.1979 * Q.sj2_dr - 2.485942
    if 0.1591713 <= Q.sj2_dr < 0.1778793:
        z += -1.202589 * Q.sj2_dr + 0.7612306
    if 0.1778793 <= Q.sj2_dr < 0.2687922:
        z += 4.301268 * Q.sj2_dr - 0.2177916
    if Q.sj2_dr >= 0.2687922:
        z += 20.89009 * Q.sj2_dr - 4.676737
    if Q.pt_7 < 45.75:
        z += 0.04312166 * Q.pt_7 - 1.972816
    if Q.lam1 < 0.008375572:
        z += -552.925 * Q.lam1 + 5.882914
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -447.9618 * Q.lam1 + 5.003787
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 84.29374 * Q.lam1 - 1.385262
    if Q.girth < 0.0717028:
        z += 53.20805 * Q.girth - 3.815166
    if Q.centroid_offset < 0.01837778:
        z += -79.90812 * Q.centroid_offset + 1.468534
    if 0.1426152 <= Q.sj3_dr_max < 0.1789613:
        z += 3.308733 * Q.sj3_dr_max - 0.4718755
    if 0.1789613 <= Q.sj3_dr_max < 0.2623172:
        z += -20.64328 * Q.sj3_dr_max + 3.814609
    if Q.sj3_dr_max >= 0.2623172:
        z += -15.62944 * Q.sj3_dr_max + 2.499393
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.01948478 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.centroid_offset < 0.04990367:
        z += -100.7876 * (0.2534037 - Q.planar_flow) * (0.04990367 - Q.centroid_offset)
    if Q.lam1 < 0.003377388 and Q.centroid_offset < 0.02685622:
        z += -49233.19 * (0.003377388 - Q.lam1) * (0.02685622 - Q.centroid_offset)
    if Q.lam1 < 0.01200373 and Q.centroid_offset > 0.01627885:
        z += -14350.22 * (0.01200373 - Q.lam1) * (Q.centroid_offset - 0.01627885)
    if Q.lam1 < 0.01643375 and Q.lam2 < 0.001130645:
        z += 67100.4 * (0.01643375 - Q.lam1) * (0.001130645 - Q.lam2)
    if Q.lam1 < 0.003377388 and Q.planar_flow < 0.2534037:
        z += -2334.083 * (0.003377388 - Q.lam1) * (0.2534037 - Q.planar_flow)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.09788
    if Q.girth >= 0.1245537:
        z += 213.9858 * Q.girth - 26.65273
    if Q.sd_rg >= 0.324646:
        z += 63.51757 * Q.sd_rg - 20.62073
    if Q.LHA >= 0.3855647:
        z += -57.03537 * Q.LHA + 21.99083
    if Q.e2 >= 0.06344108:
        z += -71.68063 * Q.e2 + 4.547496
    if Q.sd_rg > 0.324646 and Q.log_sum_pt < 6.896095:
        z += -185.235 * (Q.sd_rg - 0.324646) * (6.896095 - Q.log_sum_pt)
    if Q.girth > 0.1245537 and Q.pt_6 < 62.25:
        z += -1.226487 * (Q.girth - 0.1245537) * (62.25 - Q.pt_6)
    if Q.sd_rg > 0.324646 and Q.log_sum_pt < 6.701242:
        z += 150.3064 * (Q.sd_rg - 0.324646) * (6.701242 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_13(Q):
    z = -0.481093
    if Q.girth < 0.1484084:
        z += -88.84872 * Q.girth + 13.1859
    if Q.lam1 < 0.01643375:
        z += -285.6049 * Q.lam1 + 4.69356
    if 687.4375 <= Q.sum_pt_top5 < 839.9547:
        z += 0.007333962 * Q.sum_pt_top5 - 5.041641
    if Q.sum_pt_top5 >= 839.9547:
        z += 0.01961058 * Q.sum_pt_top5 - 15.35344
    if 0.04624032 <= Q.z_7 < 0.0586137:
        z += 79.85899 * Q.z_7 - 3.692705
    if Q.z_7 >= 0.0586137:
        z += 54.58204 * Q.z_7 - 2.21113
    if Q.pt_6 < 27.57812:
        z += 0.1365204 * Q.pt_6 - 3.764976
    if Q.sum_pt >= 988.4078:
        z += -0.01862444 * Q.sum_pt + 18.40854
    if Q.sj3_dr_max >= 0.233678:
        z += -11.55874 * Q.sj3_dr_max + 2.701022
    if Q.pt_5 < 24.57812:
        z += 0.1836589 * Q.pt_5 - 4.513992
    if Q.e3 < 8.147744e-05:
        z += -14182.23 * Q.e3 + 1.155532
    if Q.e2 < 0.08000524:
        z += 45.97709 * Q.e2 - 3.678408
    if Q.sj3_dr_min < 0.1278212:
        z += 10.97003 * Q.sj3_dr_min - 1.402202
    if Q.pt_7 >= 31.85938:
        z += -0.06767586 * Q.pt_7 + 2.156111
    if Q.tau1 < 0.1136369:
        z += 12.98089 * Q.tau1 - 1.475108
    if Q.log_sum_pt >= 6.670067:
        z += -8.691312 * Q.log_sum_pt + 57.97163
    if Q.girth < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -86.77339 * (0.1484084 - Q.girth) * (6.804164 - Q.log_sum_pt)
    if Q.girth < 0.1484084 and Q.pt_7 < 38.53125:
        z += -0.6328521 * (0.1484084 - Q.girth) * (38.53125 - Q.pt_7)
    if Q.girth < 0.1484084 and Q.lam2 < 0.0003061234:
        z += -37171.85 * (0.1484084 - Q.girth) * (0.0003061234 - Q.lam2)
    if Q.lam1 < 0.01643375 and Q.z_dr_0_0p05 > 0.3658817:
        z += -62.55245 * (0.01643375 - Q.lam1) * (Q.z_dr_0_0p05 - 0.3658817)
    if Q.sum_pt > 988.4078 and Q.z_6 > 0.03932388:
        z += 0.8041689 * (Q.sum_pt - 988.4078) * (Q.z_6 - 0.03932388)
    if Q.sum_pt_top5 > 687.4375 and Q.pt_7 < 40.04062:
        z += 0.0006071185 * (Q.sum_pt_top5 - 687.4375) * (40.04062 - Q.pt_7)
    if Q.lam1 < 0.01643375 and Q.pt_7 < 25.57812:
        z += -12.46665 * (0.01643375 - Q.lam1) * (25.57812 - Q.pt_7)
    if Q.e3 < 8.147744e-05 and Q.centroid_offset < 0.03776099:
        z += -687241.0 * (8.147744e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.z_7 > 0.0586137 and Q.lam2 > 4.64158e-05:
        z += -9522.78 * (Q.z_7 - 0.0586137) * (Q.lam2 - 4.64158e-05)
    return max(0.0, z)


def neuron_14(Q):
    z = -4.576746
    if Q.lam2 < 0.003408389:
        z += 1846.755 * Q.lam2 - 6.294458
    if Q.lam1 < 0.002464291:
        z += 4932.332 * Q.lam1 - 13.20563
    if 0.002464291 <= Q.lam1 < 0.006506576:
        z += 760.3462 * Q.lam1 - 2.924646
    if 0.006506576 <= Q.lam1 < 0.01200373:
        z += -327.7364 * Q.lam1 + 4.155045
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += -49.88407 * Q.lam1 + 0.8197823
    if Q.sd_rg < 0.1449048:
        z += -9.67387 * Q.sd_rg + 2.254901
    if 0.1449048 <= Q.sd_rg < 0.1572969:
        z += -29.42738 * Q.sd_rg + 5.117278
    if 0.1572969 <= Q.sd_rg < 0.1876504:
        z += 15.20018 * Q.sd_rg - 1.902496
    if 0.1876504 <= Q.sd_rg < 0.2330919:
        z += -17.36963 * Q.sd_rg + 4.209241
    if Q.sd_rg >= 0.2330919:
        z += -7.695755 * Q.sd_rg + 1.95434
    if 0.02689598 <= Q.girth < 0.06663269:
        z += 92.53163 * Q.girth - 2.488729
    if 0.06663269 <= Q.girth < 0.08065885:
        z += 136.3601 * Q.girth - 5.409135
    if 0.08065885 <= Q.girth < 0.08723651:
        z += 200.9316 * Q.girth - 10.6174
    if Q.girth >= 0.08723651:
        z += -90.81465 * Q.girth + 14.83352
    if Q.z_dr_0p05_0p1 >= 0.163898:
        z += -1.435566 * Q.z_dr_0p05_0p1 + 0.2352865
    if Q.e2 < 0.01655442:
        z += 3.784796 * Q.e2 + 1.226963
    if 0.01655442 <= Q.e2 < 0.03556091:
        z += -46.68955 * Q.e2 + 2.062536
    if 0.03556091 <= Q.e2 < 0.04110972:
        z += -72.48655 * Q.e2 + 2.979901
    if Q.tau1 < 0.09538712:
        z += -68.66842 * Q.tau1 + 7.29511
    if 0.09538712 <= Q.tau1 < 0.1136369:
        z += -10.45384 * Q.tau1 + 1.742187
    if 0.1136369 <= Q.tau1 < 0.1416226:
        z += -19.80458 * Q.tau1 + 2.804777
    if Q.centroid_offset >= 0.04990367:
        z += -1341.193 * Q.centroid_offset + 66.93046
    if Q.LHA >= 0.3033137:
        z += -9.988562 * Q.LHA + 3.029668
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -8117.468 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += 437.2769 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00595415:
        z += 5913.613 * (0.1115136 - Q.planar_flow) * (0.00595415 - Q.lam1)
    if Q.lam2 < 0.003408389 and Q.sd_rg > 0.2037854:
        z += -4600.458 * (0.003408389 - Q.lam2) * (Q.sd_rg - 0.2037854)
    if Q.planar_flow < 0.1115136 and Q.sj2_zsoft < 0.2832687:
        z += 60.08763 * (0.1115136 - Q.planar_flow) * (0.2832687 - Q.sj2_zsoft)
    if Q.lam2 < 0.003408389 and Q.centroid_offset > 0.02685622:
        z += 20222.58 * (0.003408389 - Q.lam2) * (Q.centroid_offset - 0.02685622)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.z_dr_0p1_0p2 < 0.04510668:
        z += 28.06156 * (0.5882598 - Q.z_dr_0p05_0p1) * (0.04510668 - Q.z_dr_0p1_0p2)
    if Q.lam2 < 0.003408389 and Q.LHA > 0.1767241:
        z += 6015.718 * (0.003408389 - Q.lam2) * (Q.LHA - 0.1767241)
    if Q.lam2 < 0.003408389 and Q.sd_rg > 0.1572969:
        z += 9375.187 * (0.003408389 - Q.lam2) * (Q.sd_rg - 0.1572969)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.n_dr_0p2_0p4 < 1.0:
        z += 3.737399 * (Q.z_dr_0p05_0p1 - 0.163898) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.log_sum_pt < 6.670067:
        z += -3.153257 * (Q.z_dr_0p05_0p1 - 0.163898) * (6.670067 - Q.log_sum_pt)
    if Q.lam1 < 0.01643375 and Q.centroid_offset > 0.02685622:
        z += -15520.97 * (0.01643375 - Q.lam1) * (Q.centroid_offset - 0.02685622)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.z_dr_0p2_0p4 < 0.20552:
        z += -9.317984 * (Q.z_dr_0p05_0p1 - 0.163898) * (0.20552 - Q.z_dr_0p2_0p4)
    if Q.lam1 < 0.006506576 and Q.sj3_dr23 > 0.1974628:
        z += 3638.554 * (0.006506576 - Q.lam1) * (Q.sj3_dr23 - 0.1974628)
    if Q.lam2 < 0.003408389 and Q.z_dr_0p2_0p4 > 0.0:
        z += -1470.786 * (0.003408389 - Q.lam2) * (Q.z_dr_0p2_0p4 - 0.0)
    return max(0.0, z)


def neuron_15(Q):
    z = -2.195215
    if Q.N2 < 0.2233283:
        z += -2.041292 * Q.N2 + 0.4558783
    if Q.lam2 < 0.003408389:
        z += -776.3024 * Q.lam2 + 2.64594
    if Q.lam1 < 0.00595415:
        z += 374.9702 * Q.lam1 - 2.227053
    if 0.00595415 <= Q.lam1 < 0.006506576:
        z += 818.2777 * Q.lam1 - 4.866572
    if 0.006506576 <= Q.lam1 < 0.00733008:
        z += 613.151 * Q.lam1 - 3.5319
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 82.80738 * Q.lam1 + 0.3555615
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -172.1032 * Q.lam1 + 2.490583
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += -95.86943 * Q.lam1 + 1.575494
    if Q.tau1 < 0.05356915:
        z += -103.3294 * Q.tau1 + 5.535266
    if 0.09538712 <= Q.tau1 < 0.1027642:
        z += -81.99383 * Q.tau1 + 7.821155
    if 0.1027642 <= Q.tau1 < 0.1136369:
        z += 16.01743 * Q.tau1 - 2.250896
    if 0.1136369 <= Q.tau1 < 0.1416226:
        z += 59.47357 * Q.tau1 - 7.189117
    if Q.tau1 >= 0.1416226:
        z += -8.919094 * Q.tau1 + 2.496832
    if Q.girth2_top2 < 0.007639643:
        z += -143.4064 * Q.girth2_top2 + 1.095573
    if Q.eccentricity >= 0.9458207:
        z += -15.23729 * Q.eccentricity + 14.41174
    if Q.girth < 0.03360421:
        z += 382.6656 * Q.girth - 17.38279
    if 0.03360421 <= Q.girth < 0.0717028:
        z += 118.7344 * Q.girth - 8.513588
    if Q.z_dr_0_0p05 < 0.1515405:
        z += -3.464325 * Q.z_dr_0_0p05 + 0.5249854
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += -3.861527 * Q.z_dr_0p05_0p1 + 2.899657
    if Q.psi_0p1 >= 0.7143804:
        z += 2.112474 * Q.psi_0p1 - 1.50911
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1872617:
        z += -254.0497 * (0.2233283 - Q.N2) * (0.1872617 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.lam1 > 0.008375572:
        z += -3363.021 * (0.2233283 - Q.N2) * (Q.lam1 - 0.008375572)
    if Q.N2 < 0.2233283 and Q.LHA > 0.2931906:
        z += 264.4889 * (0.2233283 - Q.N2) * (Q.LHA - 0.2931906)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3255822:
        z += -188.172 * (0.2233283 - Q.N2) * (Q.LHA - 0.3255822)
    if Q.girth2_top3 < 0.002151568 and Q.planar_flow < 0.1950135:
        z += -4083.126 * (0.002151568 - Q.girth2_top3) * (0.1950135 - Q.planar_flow)
    if Q.lam2 < 0.003408389 and Q.zdr_1 < 0.01849752:
        z += -10456.15 * (0.003408389 - Q.lam2) * (0.01849752 - Q.zdr_1)
    if Q.eccentricity > 0.9458207 and Q.mean_phi > -0.01753483:
        z += -406.3127 * (Q.eccentricity - 0.9458207) * (Q.mean_phi - -0.01753483)
    if Q.D3 < 0.2213841 and Q.n_dr_0p1_0p2 > 1.0:
        z += -102.6722 * (0.2213841 - Q.D3) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.N2 < 0.2233283 and Q.z_dr_0p2_0p4 > 0.20552:
        z += -1912.449 * (0.2233283 - Q.N2) * (Q.z_dr_0p2_0p4 - 0.20552)
    if Q.eccentricity > 0.9458207 and Q.sum_pt_top5 > 367.5938:
        z += 0.1220232 * (Q.eccentricity - 0.9458207) * (Q.sum_pt_top5 - 367.5938)
    if Q.lam1 < 0.008375572 and Q.sj3_dr23 > 0.1974628:
        z += 2449.676 * (0.008375572 - Q.lam1) * (Q.sj3_dr23 - 0.1974628)
    if Q.lam1 < 0.008375572 and Q.D2 < 0.875672:
        z += -1660.278 * (0.008375572 - Q.lam1) * (0.875672 - Q.D2)
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += 2845.385 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
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
