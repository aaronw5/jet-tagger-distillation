"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no mass observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.3% (the network: 65.8%); same class as the network for 87.2% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr12                   ΔR between particles 1 and 2
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
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
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        max_dr=max(dr[i] for i in real),
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -2.0
    if Q.centroid_offset >= 0.0503:
        z += -415.0 * Q.centroid_offset + 20.8745
    if Q.eccentricity >= 0.988:
        z += 98.9 * Q.eccentricity - 97.7132
    if Q.girth < 0.0838:
        z += 58.3 * Q.girth - 4.88554
    if Q.girth2 < 0.00555:
        z += -54.0 * Q.girth2 + 5.0044
    if 0.00555 <= Q.girth2 < 0.0127:
        z += -658.0 * Q.girth2 + 8.3566
    if Q.n_dr_0_0p05 >= 5.27:
        z += 0.474 * Q.n_dr_0_0p05 - 2.49798
    if Q.sd_rg < 0.119:
        z += -8.3 * Q.sd_rg - 0.1659
    if 0.119 <= Q.sd_rg < 0.175:
        z += 20.6 * Q.sd_rg - 3.605
    if Q.sum_pt_top5 < 603.0:
        z += -0.00335 * Q.sum_pt_top5 + 2.02005
    if Q.tau1 < 0.0632:
        z += -54.3 * Q.tau1 + 3.43176
    if Q.tau2 < 0.0128:
        z += 190.0 * Q.tau2 - 2.432
    if Q.width < 0.00369:
        z += 868.0 * Q.width - 3.20292
    if Q.girth < 0.015 and Q.dr0_6 > 0.199:
        z += -65600.0 * (0.015 - Q.girth) * (Q.dr0_6 - 0.199)
    if Q.girth2 < 0.02 and Q.M3 < 0.0907:
        z += 2070.0 * (0.02 - Q.girth2) * (0.0907 - Q.M3)
    if Q.girth2 < 0.0208 and Q.centroid_offset > 0.00894:
        z += -4310.0 * (0.0208 - Q.girth2) * (Q.centroid_offset - 0.00894)
    if Q.planar_flow < 0.149 and Q.D2_b2 < 1.48:
        z += 11.4 * (0.149 - Q.planar_flow) * (1.48 - Q.D2_b2)
    if Q.sum_pt < 763.0 and Q.D2_b2 < 0.395:
        z += -0.0258 * (763.0 - Q.sum_pt) * (0.395 - Q.D2_b2)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.457
    if Q.e2_sq < 0.00723:
        z += -621.0 * Q.e2_sq + 4.48983
    if 6.37 <= Q.log_sum_pt < 6.57:
        z += 4.96 * Q.log_sum_pt - 31.5952
    if Q.log_sum_pt >= 6.57:
        z += 13.89 * Q.log_sum_pt - 90.2653
    if Q.pt_7 >= 35.0:
        z += 0.11 * Q.pt_7 - 3.85
    if Q.width < 0.00902:
        z += 631.0 * Q.width - 5.69162
    if Q.z_7 < 0.0529:
        z += 66.9 * Q.z_7 - 3.53901
    if Q.log_sum_pt > 6.34 and Q.dr_0 < 0.0739:
        z += -76.1 * (Q.log_sum_pt - 6.34) * (0.0739 - Q.dr_0)
    if Q.pt_7 > 35.3 and Q.tau2 < 0.0186:
        z += -5.54 * (Q.pt_7 - 35.3) * (0.0186 - Q.tau2)
    if Q.sj2_dr > 0.19 and Q.sj3_dr_min < 0.201:
        z += 51.8 * (Q.sj2_dr - 0.19) * (0.201 - Q.sj3_dr_min)
    return max(0.0, z)


def neuron_2(Q):
    z = 3.17
    if Q.LHA >= 0.113:
        z += -6.99 * Q.LHA + 0.78987
    if Q.N2 >= 0.194:
        z += 4.81 * Q.N2 - 0.93314
    if Q.e2 < 0.0395:
        z += 30.7 * Q.e2 - 1.21265
    if Q.girth < 0.00765:
        z += 413.0 * Q.girth - 3.15945
    if Q.log_sum_pt < 6.7:
        z += -4.13 * Q.log_sum_pt + 27.671
    if Q.log_sum_pt >= 6.83:
        z += 12.6 * Q.log_sum_pt - 86.058
    if Q.pt_7 < 56.1:
        z += 0.0973 * Q.pt_7 - 5.45853
    if Q.sum_pt < 618.0:
        z += -0.00918 * Q.sum_pt + 5.67324
    if Q.sum_pt >= 987.0:
        z += -0.0218 * Q.sum_pt + 21.5166
    if Q.width < 0.0141:
        z += -139.0 * Q.width + 1.9599
    if Q.z_7 >= 0.039:
        z += -45.7 * Q.z_7 + 1.7823
    if Q.zdr_0 < 0.0236:
        z += -48.2 * Q.zdr_0 + 1.13752
    if Q.C2 < 0.0362 and Q.C2_b2 < 0.0256:
        z += -1060.0 * (0.0362 - Q.C2) * (0.0256 - Q.C2_b2)
    if Q.centroid_offset < 0.00539 and Q.C3 < 0.0187:
        z += -20200.0 * (0.00539 - Q.centroid_offset) * (0.0187 - Q.C3)
    if Q.lam1 < 0.00702 and Q.max_dr < 0.21:
        z += 1080.0 * (0.00702 - Q.lam1) * (0.21 - Q.max_dr)
    if Q.sum_pt_top5 > 703.0 and Q.D2_b2 < 0.999:
        z += 0.00683 * (Q.sum_pt_top5 - 703.0) * (0.999 - Q.D2_b2)
    return max(0.0, z)


def neuron_3(Q):
    z = -1.91
    if Q.girth >= 0.0648:
        z += 97.4 * Q.girth - 6.31152
    if Q.sj2_dr >= 0.18:
        z += 48.0 * Q.sj2_dr - 8.64
    if Q.width >= 0.0136:
        z += -272.0 * Q.width + 3.6992
    if Q.sj2_dr > 0.13 and Q.girth2_top2 < 0.00188:
        z += -20900.0 * (Q.sj2_dr - 0.13) * (0.00188 - Q.girth2_top2)
    if Q.sj2_dr > 0.173 and Q.n_dr_0p05_0p1 < 7.02:
        z += -5.71 * (Q.sj2_dr - 0.173) * (7.02 - Q.n_dr_0p05_0p1)
    if Q.sj2_dr > 0.181 and Q.tau2 < 0.063:
        z += 579.0 * (Q.sj2_dr - 0.181) * (0.063 - Q.tau2)
    if Q.sj2_dr > 0.291 and Q.z_dr_0p05_0p1 > 0.0713:
        z += -124.0 * (Q.sj2_dr - 0.291) * (Q.z_dr_0p05_0p1 - 0.0713)
    if Q.width > 0.00674 and Q.log_sum_pt > 6.08:
        z += -393.0 * (Q.width - 0.00674) * (Q.log_sum_pt - 6.08)
    return max(0.0, z)


def neuron_4(Q):
    z = -1.12
    if Q.N2 < 0.23:
        z += -38.3 * Q.N2 + 8.809
    if Q.e3 < 0.000188:
        z += -25400.0 * Q.e3 + 4.7752
    if Q.girth2 < 0.00263:
        z += -578.0 * Q.girth2 - 3.25106
    if 0.00263 <= Q.girth2 < 0.00823:
        z += 852.0 * Q.girth2 - 7.01196
    if Q.lam2 < 0.00114:
        z += 3680.0 * Q.lam2 - 4.1952
    if Q.sj3_dr_min < 0.206:
        z += -19.8 * Q.sj3_dr_min + 4.0788
    if Q.N2 < 0.221 and Q.planar_flow < 0.425:
        z += -77.1 * (0.221 - Q.N2) * (0.425 - Q.planar_flow)
    if Q.girth2_top2 < 0.00564 and Q.centroid_offset > 0.0116:
        z += 27000.0 * (0.00564 - Q.girth2_top2) * (Q.centroid_offset - 0.0116)
    if Q.sj3_dr_max > 0.317 and Q.D2_b2 < 1.35:
        z += -31.2 * (Q.sj3_dr_max - 0.317) * (1.35 - Q.D2_b2)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.53
    if Q.dr_0 < 0.0481:
        z += 103.0 * Q.dr_0 - 4.9543
    if Q.e2 < 0.0352:
        z += -138.0 * Q.e2 + 4.8576
    if Q.girth2 < 0.00102:
        z += -4370.0 * Q.girth2 + 6.354
    if 0.00102 <= Q.girth2 < 0.00276:
        z += -1090.0 * Q.girth2 + 3.0084
    if Q.n_pt_above_50 >= 6.29:
        z += -0.719 * Q.n_pt_above_50 + 4.52251
    if Q.sj3_dr_max < 0.297:
        z += 9.85 * Q.sj3_dr_max - 2.92545
    if Q.sum_pt < 721.0:
        z += 0.016 * Q.sum_pt - 11.536
    if Q.sum_pt_top5 < 431.0:
        z += 0.0727 * Q.sum_pt_top5 - 31.3337
    if Q.z_7 < 0.0295:
        z += -160.0 * Q.z_7 + 4.72
    if Q.LHA < 0.22 and Q.log_sum_pt < 6.8:
        z += -161.0 * (0.22 - Q.LHA) * (6.8 - Q.log_sum_pt)
    if Q.e2 < 0.0415 and Q.log_sum_pt > 6.9:
        z += -1800.0 * (0.0415 - Q.e2) * (Q.log_sum_pt - 6.9)
    if Q.girth2 < 0.0031 and Q.centroid_offset < 0.0166:
        z += 39000.0 * (0.0031 - Q.girth2) * (0.0166 - Q.centroid_offset)
    if Q.sum_pt < 716.0 and Q.zdr_0 < 0.00797:
        z += 4.01 * (716.0 - Q.sum_pt) * (0.00797 - Q.zdr_0)
    if Q.z_7 < 0.0515 and Q.absphi_0 < 0.0683:
        z += 807.0 * (0.0515 - Q.z_7) * (0.0683 - Q.absphi_0)
    return max(0.0, z)


def neuron_6(Q):
    z = 2.3
    if Q.centroid_offset >= 0.00758:
        z += 181.0 * Q.centroid_offset - 1.37198
    if Q.girth < 0.0271:
        z += -271.0 * Q.girth + 7.3441
    if Q.girth2 < 0.00872:
        z += 1340.0 * Q.girth2 - 11.6848
    if Q.log_sum_pt < 6.36:
        z += -18.3 * Q.log_sum_pt + 116.388
    if Q.max_dr < 0.155:
        z += 25.2 * Q.max_dr - 3.906
    if Q.sj3_dr_max >= 0.173:
        z += 10.1 * Q.sj3_dr_max - 1.7473
    if Q.sj3_dr_min >= 0.0182:
        z += -13.0 * Q.sj3_dr_min + 0.2366
    if Q.sum_pt >= 1010.0:
        z += 0.0148 * Q.sum_pt - 14.948
    if Q.tau1 < 0.111:
        z += -42.9 * Q.tau1 + 4.7619
    if Q.tau2 >= 0.0189:
        z += -43.9 * Q.tau2 + 0.82971
    if Q.centroid_offset > 0.013 and Q.mean_phi2 < 0.0143:
        z += 5260.0 * (Q.centroid_offset - 0.013) * (0.0143 - Q.mean_phi2)
    if Q.centroid_offset > 0.0139 and Q.n_dr_0p05_0p1 < 4.55:
        z += 9.69 * (Q.centroid_offset - 0.0139) * (4.55 - Q.n_dr_0p05_0p1)
    if Q.centroid_offset > 0.00723 and Q.sj2_dr < 0.169:
        z += 1220.0 * (Q.centroid_offset - 0.00723) * (0.169 - Q.sj2_dr)
    if Q.centroid_offset > 0.0149 and Q.sum_pt < 969.0:
        z += -0.556 * (Q.centroid_offset - 0.0149) * (969.0 - Q.sum_pt)
    if Q.log_sum_pt < 6.36 and Q.pt_7 > 21.4:
        z += -0.795 * (6.36 - Q.log_sum_pt) * (Q.pt_7 - 21.4)
    if Q.log_sum_pt < 6.64 and Q.z_7 < 0.0591:
        z += 629.0 * (6.64 - Q.log_sum_pt) * (0.0591 - Q.z_7)
    if Q.width < 0.0151 and Q.planar_flow < 0.369:
        z += -585.0 * (0.0151 - Q.width) * (0.369 - Q.planar_flow)
    if Q.z_6 < 0.0224 and Q.log_sum_pt < 6.83:
        z += 5250.0 * (0.0224 - Q.z_6) * (6.83 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_7(Q):
    z = 5.66
    if Q.e2_sq < 0.00532:
        z += -745.0 * Q.e2_sq + 2.4265
    if 0.00532 <= Q.e2_sq < 0.00814:
        z += 545.0 * Q.e2_sq - 4.4363
    if Q.girth < 0.0908:
        z += 41.1 * Q.girth - 3.73188
    if 0.00766 <= Q.girth2 < 0.013:
        z += -1230.0 * Q.girth2 + 9.4218
    if Q.girth2 >= 0.013:
        z += 50.0 * Q.girth2 - 7.2182
    if Q.lam2 < 0.00112:
        z += -1760.0 * Q.lam2 + 1.9712
    if Q.planar_flow < 0.238:
        z += 6.87 * Q.planar_flow - 1.63506
    if Q.sj2_dr < 0.158:
        z += -11.5 * Q.sj2_dr + 0.6548
    if 0.158 <= Q.sj2_dr < 0.184:
        z += 44.7 * Q.sj2_dr - 8.2248
    if Q.tau1 < 0.0644:
        z += -48.2 * Q.tau1 + 3.10408
    if Q.width < 0.00545:
        z += 1830.0 * Q.width - 9.9735
    if Q.planar_flow < 0.196 and Q.log_sum_pt > 6.42:
        z += 46.2 * (0.196 - Q.planar_flow) * (Q.log_sum_pt - 6.42)
    if Q.planar_flow < 0.183 and Q.z_7 < 0.0523:
        z += -449.0 * (0.183 - Q.planar_flow) * (0.0523 - Q.z_7)
    if Q.tau1 < 0.057 and Q.z_dr_0p05_0p1 > 0.655:
        z += -347.0 * (0.057 - Q.tau1) * (Q.z_dr_0p05_0p1 - 0.655)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.295
    if Q.log_sum_pt >= 6.68:
        z += -5.72 * Q.log_sum_pt + 38.2096
    if Q.sj3_dr_max < 0.104:
        z += 1.5 * Q.sj3_dr_max - 0.156
    if Q.LHA < 0.196 and Q.z_dr_0p2_0p4 < 0.239:
        z += -89.8 * (0.196 - Q.LHA) * (0.239 - Q.z_dr_0p2_0p4)
    if Q.girth2 < 0.00708 and Q.centroid_offset < 0.0236:
        z += 24800.0 * (0.00708 - Q.girth2) * (0.0236 - Q.centroid_offset)
    if Q.girth2 < 0.00901 and Q.planar_flow < 0.434:
        z += 290.0 * (0.00901 - Q.girth2) * (0.434 - Q.planar_flow)
    return max(0.0, z)


def neuron_9(Q):
    z = -4.2
    if Q.centroid_offset < 0.0229:
        z += -122.0 * Q.centroid_offset + 2.7938
    if Q.e3 >= 0.000242:
        z += 7780.0 * Q.e3 - 1.88276
    if Q.girth2 < 0.00382:
        z += -1300.0 * Q.girth2 + 4.966
    if Q.log_sum_pt < 6.88:
        z += -5.99 * Q.log_sum_pt + 41.2112
    if Q.max_dr < 0.125:
        z += -19.8 * Q.max_dr + 2.475
    if Q.sj2_dr < 0.13:
        z += 42.9 * Q.sj2_dr - 5.577
    if Q.tau1 < 0.0565:
        z += 89.7 * Q.tau1 - 5.06805
    if Q.width < 0.0067:
        z += -1080.0 * Q.width + 7.236
    if Q.centroid_offset < 0.0197 and Q.C2 < 0.075:
        z += 3920.0 * (0.0197 - Q.centroid_offset) * (0.075 - Q.C2)
    if Q.centroid_offset < 0.0218 and Q.D2_b2 < 0.387:
        z += -466.0 * (0.0218 - Q.centroid_offset) * (0.387 - Q.D2_b2)
    if Q.centroid_offset < 0.0181 and Q.N3 < 1.47:
        z += -276.0 * (0.0181 - Q.centroid_offset) * (1.47 - Q.N3)
    if Q.centroid_offset < 0.0255 and Q.tau21_b2 > 0.0724:
        z += 154.0 * (0.0255 - Q.centroid_offset) * (Q.tau21_b2 - 0.0724)
    if Q.log_sum_pt < 7.03 and Q.planar_flow > 0.116:
        z += -2.05 * (7.03 - Q.log_sum_pt) * (Q.planar_flow - 0.116)
    if Q.log_sum_pt < 6.93 and Q.z_5 < 0.0581:
        z += 240.0 * (6.93 - Q.log_sum_pt) * (0.0581 - Q.z_5)
    return max(0.0, z)


def neuron_10(Q):
    z = 4.38
    if Q.LHA >= 0.184:
        z += -16.9 * Q.LHA + 3.1096
    if Q.e3 >= 7.84e-05:
        z += 3340.0 * Q.e3 - 0.261856
    if Q.girth2 >= 0.0076:
        z += 277.0 * Q.girth2 - 2.1052
    if Q.lam1 < 0.00345:
        z += 1319.0 * Q.lam1 - 5.81005
    if 0.00345 <= Q.lam1 < 0.0062:
        z += 458.0 * Q.lam1 - 2.8396
    if 0.00011 <= Q.lam2 < 0.00326:
        z += 608.0 * Q.lam2 - 0.06688
    if Q.lam2 >= 0.00326:
        z += -282.0 * Q.lam2 + 2.83452
    if Q.n_dr_0p05_0p1 < 4.78:
        z += -0.165 * Q.n_dr_0p05_0p1 + 0.7887
    if Q.sj3_dr_min >= 0.129:
        z += 11.7 * Q.sj3_dr_min - 1.5093
    if Q.tau1 >= 0.0475:
        z += 20.7 * Q.tau1 - 0.98325
    if Q.LHA > 0.317 and Q.planar_flow < 0.915:
        z += -35.6 * (Q.LHA - 0.317) * (0.915 - Q.planar_flow)
    if Q.lam2 > 0.000186 and Q.eccentricity > 0.48:
        z += -1410.0 * (Q.lam2 - 0.000186) * (Q.eccentricity - 0.48)
    if Q.lam2 > 5.13e-06 and Q.log_sum_pt < 6.55:
        z += -788.0 * (Q.lam2 - 5.13e-06) * (6.55 - Q.log_sum_pt)
    if Q.sj2_dr > 0.177 and Q.planar_flow < 0.681:
        z += -32.8 * (Q.sj2_dr - 0.177) * (0.681 - Q.planar_flow)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.406
    if Q.e2_sq < 0.00816:
        z += 2090.0 * Q.e2_sq - 17.0544
    if Q.girth < 0.0726:
        z += 101.0 * Q.girth - 7.3326
    if Q.planar_flow < 0.262:
        z += -7.88 * Q.planar_flow + 2.06456
    if 0.0847 <= Q.sj2_dr < 0.173:
        z += 16.8 * Q.sj2_dr - 1.42296
    if Q.sj2_dr >= 0.173:
        z += -15.6 * Q.sj2_dr + 4.18224
    if Q.width < 0.00867:
        z += -2780.0 * Q.width + 25.3116
    if 0.00867 <= Q.width < 0.0127:
        z += -300.0 * Q.width + 3.81
    if Q.centroid_offset < 0.014 and Q.sj3_dr_min < 0.0728:
        z += -1480.0 * (0.014 - Q.centroid_offset) * (0.0728 - Q.sj3_dr_min)
    if Q.centroid_offset < 0.0146 and Q.tau21_b2 < 0.0608:
        z += 2440.0 * (0.0146 - Q.centroid_offset) * (0.0608 - Q.tau21_b2)
    if Q.planar_flow < 0.249 and Q.pt_7 < 37.2:
        z += -0.336 * (0.249 - Q.planar_flow) * (37.2 - Q.pt_7)
    if Q.planar_flow < 0.246 and Q.sum_pt < 851.0:
        z += -0.0136 * (0.246 - Q.planar_flow) * (851.0 - Q.sum_pt)
    if Q.pt_6 < 40.0 and Q.girth2_top3 > 0.0049:
        z += -11.7 * (40.0 - Q.pt_6) * (Q.girth2_top3 - 0.0049)
    return max(0.0, z)


def neuron_12(Q):
    z = 0.00181
    if 0.021 <= Q.girth2 < 0.0251:
        z += 13.3 * Q.girth2 - 0.2793
    if Q.girth2 >= 0.0251:
        z += 614.3 * Q.girth2 - 15.3644
    if Q.girth2 > 0.0239 and Q.sum_pt < 916.0:
        z += -0.797 * (Q.girth2 - 0.0239) * (916.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.46
    if Q.girth < 0.148:
        z += -76.6 * Q.girth + 11.3368
    if Q.lam1 < 0.0148:
        z += -199.0 * Q.lam1 + 2.9452
    if Q.pt_7 >= 32.1:
        z += -0.0459 * Q.pt_7 + 1.47339
    if Q.sj3_dr_max >= 0.233:
        z += -14.1 * Q.sj3_dr_max + 3.2853
    if Q.sj3_dr_min < 0.124:
        z += 17.2 * Q.sj3_dr_min - 2.1328
    if Q.tau1 < 0.121:
        z += 28.4 * Q.tau1 - 3.4364
    if Q.z_7 >= 0.0583:
        z += 49.8 * Q.z_7 - 2.90334
    if Q.e3 < 7.94e-05 and Q.centroid_offset < 0.0419:
        z += -761000.0 * (7.94e-05 - Q.e3) * (0.0419 - Q.centroid_offset)
    if Q.girth < 0.15 and Q.log_sum_pt < 6.8:
        z += -55.3 * (0.15 - Q.girth) * (6.8 - Q.log_sum_pt)
    if Q.girth < 0.15 and Q.pt_7 < 38.4:
        z += -0.962 * (0.15 - Q.girth) * (38.4 - Q.pt_7)
    if Q.lam1 < 0.0178 and Q.pt_7 < 25.6:
        z += -15.2 * (0.0178 - Q.lam1) * (25.6 - Q.pt_7)
    if Q.log_sum_pt < 6.47 and Q.C3 < 0.0667:
        z += -76.2 * (6.47 - Q.log_sum_pt) * (0.0667 - Q.C3)
    if Q.sum_pt > 994.0 and Q.sj3_pairmin_over_m > 0.225:
        z += -0.0759 * (Q.sum_pt - 994.0) * (Q.sj3_pairmin_over_m - 0.225)
    if Q.sum_pt_top5 > 687.0 and Q.pt_7 < 40.4:
        z += 0.000663 * (Q.sum_pt_top5 - 687.0) * (40.4 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.0658
    if 0.0234 <= Q.centroid_offset < 0.0504:
        z += -67.6 * Q.centroid_offset + 1.58184
    if Q.centroid_offset >= 0.0504:
        z += -263.6 * Q.centroid_offset + 11.46024
    if 0.0064 <= Q.e2_sq < 0.0174:
        z += 1230.0 * Q.e2_sq - 7.872
    if Q.e2_sq >= 0.0174:
        z += 80.0 * Q.e2_sq + 12.138
    if 0.04 <= Q.girth < 0.0883:
        z += 184.0 * Q.girth - 7.36
    if 0.0883 <= Q.girth < 0.103:
        z += 17.0 * Q.girth + 7.3861
    if Q.girth >= 0.103:
        z += -208.0 * Q.girth + 30.5611
    if Q.girth2 >= 0.00676:
        z += -1230.0 * Q.girth2 + 8.3148
    if Q.sj3_dr_max < 0.159:
        z += -22.4 * Q.sj3_dr_max - 0.1176
    if 0.159 <= Q.sj3_dr_max < 0.231:
        z += 51.1 * Q.sj3_dr_max - 11.8041
    if Q.tau1 >= 0.0356:
        z += -68.2 * Q.tau1 + 2.42792
    if Q.e2_sq > 0.00234 and Q.log_sum_pt > 6.22:
        z += 510.0 * (Q.e2_sq - 0.00234) * (Q.log_sum_pt - 6.22)
    if Q.girth2 > 0.00864 and Q.log_sum_pt > 6.25:
        z += -5870.0 * (Q.girth2 - 0.00864) * (Q.log_sum_pt - 6.25)
    if Q.planar_flow < 0.111 and Q.centroid_offset < 0.0181:
        z += -553.0 * (0.111 - Q.planar_flow) * (0.0181 - Q.centroid_offset)
    if Q.planar_flow < 0.115 and Q.lam1 < 0.00755:
        z += -5980.0 * (0.115 - Q.planar_flow) * (0.00755 - Q.lam1)
    if Q.planar_flow < 0.119 and Q.lam1 < 0.0159:
        z += 1060.0 * (0.119 - Q.planar_flow) * (0.0159 - Q.lam1)
    if Q.planar_flow < 0.104 and Q.width < 0.00596:
        z += 5990.0 * (0.104 - Q.planar_flow) * (0.00596 - Q.width)
    if Q.psi_0p1 > 0.819 and Q.centroid_offset > 0.0379:
        z += -1670.0 * (Q.psi_0p1 - 0.819) * (Q.centroid_offset - 0.0379)
    if Q.width > 0.0129 and Q.sum_pt > 483.0:
        z += 5.59 * (Q.width - 0.0129) * (Q.sum_pt - 483.0)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.46
    if Q.girth2 < 0.00756:
        z += 1080.0 * Q.girth2 - 8.1648
    if Q.girth2_top3 < 0.00231:
        z += 557.0 * Q.girth2_top3 - 1.28667
    if Q.planar_flow < 0.216:
        z += 4.85 * Q.planar_flow - 1.0476
    if Q.tau1 < 0.0908:
        z += -86.3 * Q.tau1 + 7.83604
    if Q.z_dr_0_0p05 < 0.137:
        z += -7.3 * Q.z_dr_0_0p05 + 1.0001
    if Q.N2 < 0.217 and Q.LHA > 0.386:
        z += -6780.0 * (0.217 - Q.N2) * (Q.LHA - 0.386)
    if Q.N2 < 0.238 and Q.LHA > 0.289:
        z += 285.0 * (0.238 - Q.N2) * (Q.LHA - 0.289)
    if Q.N2 < 0.247 and Q.girth2 > 0.00871:
        z += -6320.0 * (0.247 - Q.N2) * (Q.girth2 - 0.00871)
    if Q.N2 < 0.219 and Q.sj2_dr < 0.19:
        z += -380.0 * (0.219 - Q.N2) * (0.19 - Q.sj2_dr)
    if Q.e2_sq < 0.00638 and Q.dr12 > 0.157:
        z += 75000.0 * (0.00638 - Q.e2_sq) * (Q.dr12 - 0.157)
    if Q.girth2 < 0.00668 and Q.dr12 > 0.156:
        z += -78600.0 * (0.00668 - Q.girth2) * (Q.dr12 - 0.156)
    if Q.girth2_top3 < 0.00214 and Q.eccentricity > 0.959:
        z += -38700.0 * (0.00214 - Q.girth2_top3) * (Q.eccentricity - 0.959)
    if Q.planar_flow < 0.195 and Q.sum_pt_top5 > 401.0:
        z += 0.0601 * (0.195 - Q.planar_flow) * (Q.sum_pt_top5 - 401.0)
    if Q.planar_flow < 0.182 and Q.z_7 < 0.0723:
        z += -231.0 * (0.182 - Q.planar_flow) * (0.0723 - Q.z_7)
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
