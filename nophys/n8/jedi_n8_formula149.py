"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no W/Z/H/t mass values offered as thresholds), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.4% (the network: 65.8%); same class as the network for 86.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
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
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top2=mass_of(2),
        max_dr=max(dr[i] for i in real),
        pt_6=pt[6],
        pt_7=pt[7],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sj2_dr=subjets(2)["dr"][0],
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = 0.00189
    if Q.C2_b2 < 0.0017:
        z += 1010.0 * Q.C2_b2 - 1.717
    if Q.e2 < 0.026:
        z += -238.0 * Q.e2 + 6.188
    if Q.mass < 23.0:
        z += -0.0825 * Q.mass - 0.165
    if 23.0 <= Q.mass < 56.0:
        z += 0.0625 * Q.mass - 3.5
    if Q.planar_flow < 0.12:
        z += -18.3 * Q.planar_flow + 2.196
    if Q.sj3_dr_max >= 0.23:
        z += -29.5 * Q.sj3_dr_max + 6.785
    if Q.sum_pt >= 910.0:
        z += -0.0168 * Q.sum_pt + 15.288
    if Q.lam1_plus_lam2 < 0.0044:
        z += 1581.0 * Q.lam1_plus_lam2 - 3.6608
    if 0.0044 <= Q.lam1_plus_lam2 < 0.0088:
        z += -749.0 * Q.lam1_plus_lam2 + 6.5912
    if Q.sum_z_dr2 < 0.013 and Q.D2 < 1.0:
        z += 595.0 * (0.013 - Q.sum_z_dr2) * (1.0 - Q.D2)
    if Q.sum_z_dr2 < 0.015 and Q.centroid_offset > 0.019:
        z += -10800.0 * (0.015 - Q.sum_z_dr2) * (Q.centroid_offset - 0.019)
    if Q.lam1 < 0.0065 and Q.D2 < 0.88:
        z += -3150.0 * (0.0065 - Q.lam1) * (0.88 - Q.D2)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.088
    if Q.log_sum_pt >= 6.4:
        z += 10.2 * Q.log_sum_pt - 65.28
    if Q.mass_over_sum_pt_sq < 0.0065:
        z += -822.0 * Q.mass_over_sum_pt_sq + 5.343
    if Q.pt_7 >= 34.0:
        z += 0.147 * Q.pt_7 - 4.998
    if Q.lam1_plus_lam2 < 0.0074:
        z += 990.0 * Q.lam1_plus_lam2 - 7.326
    if Q.z_7 < 0.061:
        z += 74.0 * Q.z_7 - 4.514
    if Q.pt_7 > 34.0 and Q.tau2 < 0.017:
        z += -8.75 * (Q.pt_7 - 34.0) * (0.017 - Q.tau2)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.828
    if Q.sum_z_dr < 0.0077:
        z += 541.0 * Q.sum_z_dr - 4.1657
    if Q.lam1 < 0.006:
        z += -519.0 * Q.lam1 + 3.114
    if Q.log_sum_pt < 6.5:
        z += -4.11 * Q.log_sum_pt + 26.715
    if Q.sum_pt < 780.0:
        z += -0.00235 * Q.sum_pt + 1.833
    if Q.z_7 < 0.033:
        z += 126.0 * Q.z_7 - 4.158
    if Q.sum_z_dr2 < 0.00033 and Q.mass_top2 < 35.0:
        z += 87.3 * (0.00033 - Q.sum_z_dr2) * (35.0 - Q.mass_top2)
    return max(0.0, z)


def neuron_3(Q):
    z = -5.31
    if Q.centroid_offset >= 0.011:
        z += 46.5 * Q.centroid_offset - 0.5115
    if Q.sum_z_dr >= 0.036:
        z += 117.0 * Q.sum_z_dr - 4.212
    if Q.sum_z_dr2 >= 0.0093:
        z += -823.0 * Q.sum_z_dr2 + 7.6539
    if Q.mass >= 66.0:
        z += -0.0831 * Q.mass + 5.4846
    if Q.sj2_dr >= 0.2:
        z += 34.8 * Q.sj2_dr - 6.96
    if Q.tau1 >= 0.11:
        z += 89.8 * Q.tau1 - 9.878
    if Q.sum_z_dr > 0.033 and Q.log_sum_pt > 6.2:
        z += 91.0 * (Q.sum_z_dr - 0.033) * (Q.log_sum_pt - 6.2)
    if Q.mass_over_sum_pt > 0.062 and Q.sj2_dr < 0.22:
        z += -2830.0 * (Q.mass_over_sum_pt - 0.062) * (0.22 - Q.sj2_dr)
    return max(0.0, z)


def neuron_4(Q):
    z = -1.45
    if Q.C2_b2 < 0.009:
        z += -799.0 * Q.C2_b2 + 7.191
    if Q.N2 < 0.21:
        z += -36.0 * Q.N2 + 7.56
    if Q.lam2 < 0.00066:
        z += 9090.0 * Q.lam2 - 5.9994
    if Q.mass_over_sum_pt >= 0.091:
        z += -71.4 * Q.mass_over_sum_pt + 6.4974
    if Q.mass < 68.0 and Q.D2 < 0.96:
        z += -0.201 * (68.0 - Q.mass) * (0.96 - Q.D2)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.99
    if Q.sum_zz_dr2 < 0.0055:
        z += -497.0 * Q.sum_zz_dr2 + 2.7335
    if 6.2 <= Q.log_sum_pt < 6.7:
        z += 4.73 * Q.log_sum_pt - 29.326
    if 6.7 <= Q.log_sum_pt < 6.9:
        z += -7.37 * Q.log_sum_pt + 51.744
    if Q.log_sum_pt >= 6.9:
        z += -50.37 * Q.log_sum_pt + 348.444
    if Q.pt_7 >= 22.0:
        z += -0.0978 * Q.pt_7 + 2.1516
    if Q.z_7 < 0.03:
        z += -180.0 * Q.z_7 + 5.4
    if Q.zdr_0 < 0.022:
        z += 120.0 * Q.zdr_0 - 2.64
    if Q.LHA < 0.22 and Q.log_sum_pt < 6.8:
        z += -187.0 * (0.22 - Q.LHA) * (6.8 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.0016 and Q.centroid_offset < 0.023:
        z += 188000.0 * (0.0016 - Q.sum_z_dr2) * (0.023 - Q.centroid_offset)
    if Q.sum_z_dr2_top3 < 0.003 and Q.tau3 < 0.0066:
        z += 205000.0 * (0.003 - Q.sum_z_dr2_top3) * (0.0066 - Q.tau3)
    if Q.z_7 < 0.083 and Q.centroid_offset < 0.034:
        z += -1760.0 * (0.083 - Q.z_7) * (0.034 - Q.centroid_offset)
    return max(0.0, z)


def neuron_6(Q):
    z = 3.82
    if 0.0082 <= Q.centroid_offset < 0.019:
        z += 154.0 * Q.centroid_offset - 1.2628
    if Q.centroid_offset >= 0.019:
        z += 9.0 * Q.centroid_offset + 1.4922
    if Q.sum_zz_dr2 < 0.0031:
        z += -1090.0 * Q.sum_zz_dr2 + 3.379
    if Q.sum_z_dr2 < 0.0087:
        z += 1830.0 * Q.sum_z_dr2 - 15.921
    if Q.lam2 < 0.0012:
        z += 2230.0 * Q.lam2 - 2.676
    if Q.max_dr < 0.14:
        z += 42.6 * Q.max_dr - 5.964
    if Q.sj3_dr_max < 0.18:
        z += -42.6 * Q.sj3_dr_max + 7.668
    if Q.sj3_dr_max >= 0.18:
        z += 12.1 * Q.sj3_dr_max - 2.178
    if Q.sj3_dr_min >= 0.14:
        z += -39.1 * Q.sj3_dr_min + 5.474
    if Q.sj3_pair_mass_min >= 3.2:
        z += -0.124 * Q.sj3_pair_mass_min + 0.3968
    if Q.tau1 < 0.11:
        z += -67.1 * Q.tau1 + 7.381
    if Q.pt_6 < 40.0 and Q.log_sum_pt < 6.8:
        z += 1.03 * (40.0 - Q.pt_6) * (6.8 - Q.log_sum_pt)
    if Q.pt_6 < 41.0 and Q.z_7 > 0.024:
        z += -11.4 * (41.0 - Q.pt_6) * (Q.z_7 - 0.024)
    return max(0.0, z)


def neuron_7(Q):
    z = 7.71
    if Q.centroid_offset < 0.023:
        z += 103.0 * Q.centroid_offset - 2.369
    if Q.e2 >= 0.05:
        z += -262.0 * Q.e2 + 13.1
    if Q.e3 < 8.1e-05:
        z += 31400.0 * Q.e3 - 2.5434
    if Q.sum_z_dr2 < 0.00097:
        z += 3450.0 * Q.sum_z_dr2 - 3.3465
    if Q.sum_z_dr2 >= 0.013:
        z += 1850.0 * Q.sum_z_dr2 - 24.05
    if Q.lam1 < 0.0081:
        z += 775.0 * Q.lam1 - 6.2775
    if 37.0 <= Q.mass < 77.0:
        z += 0.0836 * Q.mass - 3.0932
    if Q.mass >= 77.0:
        z += -0.1864 * Q.mass + 17.6968
    if 0.075 <= Q.mass_over_sum_pt < 0.091:
        z += 184.0 * Q.mass_over_sum_pt - 13.8
    if Q.mass_over_sum_pt >= 0.091:
        z += -32.0 * Q.mass_over_sum_pt + 5.856
    if Q.sj2_dr < 0.15:
        z += -17.5 * Q.sj2_dr + 2.625
    if Q.tau1 < 0.051:
        z += -43.7 * Q.tau1 + 2.2287
    if Q.lam1_plus_lam2 >= 0.0056:
        z += -1550.0 * Q.lam1_plus_lam2 + 8.68
    return max(0.0, z)


def neuron_8(Q):
    z = -0.357
    if Q.LHA < 0.19:
        z += 44.5 * Q.LHA - 8.455
    if Q.sum_z_dr2 < 0.0057 and Q.centroid_offset < 0.027:
        z += 38400.0 * (0.0057 - Q.sum_z_dr2) * (0.027 - Q.centroid_offset)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.756
    if Q.e2 >= 0.042:
        z += 121.0 * Q.e2 - 5.082
    if Q.sum_z_dr >= 0.061:
        z += -43.2 * Q.sum_z_dr + 2.6352
    if Q.sum_z_dr2 < 0.0038:
        z += -2850.0 * Q.sum_z_dr2 + 10.83
    if Q.lam2 >= 0.0013:
        z += 788.0 * Q.lam2 - 1.0244
    if Q.mass_over_sum_pt < 0.062:
        z += 178.0 * Q.mass_over_sum_pt - 11.036
    if Q.max_dr < 0.11:
        z += -44.4 * Q.max_dr + 4.884
    if Q.sj3_dr_max < 0.14:
        z += 43.4 * Q.sj3_dr_max - 4.072
    if 0.14 <= Q.sj3_dr_max < 0.2:
        z += -33.4 * Q.sj3_dr_max + 6.68
    if Q.z_7 < 0.05:
        z += -33.9 * Q.z_7 + 1.695
    if Q.mass < 58.0 and Q.centroid_offset < 0.022:
        z += 6.8 * (58.0 - Q.mass) * (0.022 - Q.centroid_offset)
    if Q.mass < 52.0 and Q.log_sum_pt < 6.8:
        z += 0.162 * (52.0 - Q.mass) * (6.8 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_10(Q):
    z = 0.935
    if Q.C2_b2 >= 0.0075:
        z += 111.0 * Q.C2_b2 - 0.8325
    if Q.LHA >= 0.3:
        z += -28.4 * Q.LHA + 8.52
    if Q.e3 >= 7.5e-08:
        z += -6830.0 * Q.e3 + 0.00051225
    if Q.sum_z_dr2 >= 0.0073:
        z += 250.0 * Q.sum_z_dr2 - 1.825
    if Q.lam1 < 0.0015:
        z += 2091.0 * Q.lam1 - 4.5972
    if 0.0015 <= Q.lam1 < 0.0042:
        z += 541.0 * Q.lam1 - 2.2722
    if Q.lam2 >= 0.00019:
        z += 488.0 * Q.lam2 - 0.09272
    if Q.mass < 77.0:
        z += -0.0313 * Q.mass + 2.4101
    if Q.sj3_pair_mass_min >= 11.0:
        z += 0.175 * Q.sj3_pair_mass_min - 1.925
    if Q.tau1 >= 0.059:
        z += 35.2 * Q.tau1 - 2.0768
    if Q.lam1 > 0.0071 and Q.D2_b2 < 0.38:
        z += -491.0 * (Q.lam1 - 0.0071) * (0.38 - Q.D2_b2)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.217
    if Q.centroid_offset < 0.0091:
        z += -102.0 * Q.centroid_offset + 0.9282
    if Q.centroid_offset >= 0.0094:
        z += -121.0 * Q.centroid_offset + 1.1374
    if Q.sum_z_dr < 0.021:
        z += 296.4 * Q.sum_z_dr - 10.2738
    if 0.021 <= Q.sum_z_dr < 0.072:
        z += 79.4 * Q.sum_z_dr - 5.7168
    if Q.mass < 15.0:
        z += -0.158 * Q.mass + 2.37
    if Q.max_dr < 0.15:
        z += -17.8 * Q.max_dr + 2.67
    if Q.sj2_dr >= 0.27:
        z += 32.3 * Q.sj2_dr - 8.721
    if Q.sj3_dr_max < 0.17:
        z += 37.2 * Q.sj3_dr_max - 4.299
    if 0.17 <= Q.sj3_dr_max < 0.26:
        z += -22.5 * Q.sj3_dr_max + 5.85
    if Q.lam1_plus_lam2 < 0.0094:
        z += -782.0 * Q.lam1_plus_lam2 + 7.3508
    if Q.z_7 >= 0.019:
        z += 27.1 * Q.z_7 - 0.5149
    if Q.centroid_offset > 0.05 and Q.D2 < 2.9:
        z += -195.0 * (Q.centroid_offset - 0.05) * (2.9 - Q.D2)
    if Q.centroid_offset < 0.04 and Q.sum_pt < 900.0:
        z += -0.195 * (0.04 - Q.centroid_offset) * (900.0 - Q.sum_pt)
    if Q.centroid_offset > 0.047 and Q.tau32 < 0.7:
        z += 828.0 * (Q.centroid_offset - 0.047) * (0.7 - Q.tau32)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.18
    if Q.sum_z_dr2 >= 0.02:
        z += 145.0 * Q.sum_z_dr2 - 2.9
    if Q.mass >= 91.0:
        z += 0.105 * Q.mass - 9.555
    return max(0.0, z)


def neuron_13(Q):
    z = -0.581
    if Q.sum_z_dr < 0.15:
        z += -37.0 * Q.sum_z_dr + 5.55
    if Q.lam1 < 0.016:
        z += -142.0 * Q.lam1 + 2.272
    if Q.z_7 < 0.028:
        z += 262.0 * Q.z_7 - 7.336
    if Q.sum_z_dr < 0.15 and Q.log_sum_pt < 6.8:
        z += -45.0 * (0.15 - Q.sum_z_dr) * (6.8 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.15 and Q.pt_7 < 39.0:
        z += -1.43 * (0.15 - Q.sum_z_dr) * (39.0 - Q.pt_7)
    if Q.sum_z_dr < 0.15 and Q.tau2 > 0.0083:
        z += 985.0 * (0.15 - Q.sum_z_dr) * (Q.tau2 - 0.0083)
    if Q.sum_pt > 990.0 and Q.pt_6 < 62.0:
        z += -0.00063 * (Q.sum_pt - 990.0) * (62.0 - Q.pt_6)
    if Q.sum_pt_top5 > 660.0 and Q.pt_7 < 40.0:
        z += 0.0012 * (Q.sum_pt_top5 - 660.0) * (40.0 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = -3.3
    if 0.027 <= Q.centroid_offset < 0.05:
        z += -91.3 * Q.centroid_offset + 2.4651
    if Q.centroid_offset >= 0.05:
        z += -1074.3 * Q.centroid_offset + 51.6151
    if 0.017 <= Q.e2 < 0.05:
        z += -185.0 * Q.e2 + 3.145
    if Q.e2 >= 0.05:
        z += 142.0 * Q.e2 - 13.205
    if 0.027 <= Q.sum_z_dr < 0.041:
        z += 138.0 * Q.sum_z_dr - 3.726
    if 0.041 <= Q.sum_z_dr < 0.081:
        z += 225.5 * Q.sum_z_dr - 7.3135
    if 0.081 <= Q.sum_z_dr < 0.087:
        z += 401.5 * Q.sum_z_dr - 21.5695
    if Q.sum_z_dr >= 0.087:
        z += -38.5 * Q.sum_z_dr + 16.7105
    if Q.mass_over_sum_pt >= 0.08:
        z += 294.0 * Q.mass_over_sum_pt - 23.52
    if Q.sd_mass < 50.0:
        z += -0.0018 * Q.sd_mass - 0.5288
    if 50.0 <= Q.sd_mass < 76.0:
        z += 0.0238 * Q.sd_mass - 1.8088
    if 0.062 <= Q.sj2_dr < 0.16:
        z += -16.3 * Q.sj2_dr + 1.0106
    if 0.16 <= Q.sj2_dr < 0.2:
        z += 63.3 * Q.sj2_dr - 11.7254
    if Q.sj2_dr >= 0.2:
        z += 27.4 * Q.sj2_dr - 4.5454
    if Q.lam1_plus_lam2 >= 0.0068:
        z += -2310.0 * Q.lam1_plus_lam2 + 15.708
    if Q.z_dr_0p05_0p1 >= 0.75:
        z += -25.4 * Q.z_dr_0p05_0p1 + 19.05
    if Q.planar_flow < 0.11 and Q.lam1 < 0.0075:
        z += -5110.0 * (0.11 - Q.planar_flow) * (0.0075 - Q.lam1)
    if Q.planar_flow < 0.1 and Q.lam1_plus_lam2 < 0.006:
        z += 8910.0 * (0.1 - Q.planar_flow) * (0.006 - Q.lam1_plus_lam2)
    if Q.z_dr_0p05_0p1 > 0.75 and Q.n_dr_0p2_0p4 < 1.0:
        z += 24.7 * (Q.z_dr_0p05_0p1 - 0.75) * (1.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.0482
    if Q.N2 < 0.23:
        z += 18.3 * Q.N2 - 4.209
    if Q.centroid_offset < 0.018:
        z += 236.0 * Q.centroid_offset - 4.248
    if Q.e2 < 0.043:
        z += -144.0 * Q.e2 + 6.192
    if Q.sum_z_dr2 < 0.0074:
        z += 1470.0 * Q.sum_z_dr2 - 10.878
    if Q.lam1_plus_lam2 < 0.013:
        z += -281.0 * Q.lam1_plus_lam2 + 3.653
    if Q.N2 < 0.23 and Q.sum_pt_top5 > 410.0:
        z += 0.0615 * (0.23 - Q.N2) * (Q.sum_pt_top5 - 410.0)
    if Q.sum_z_dr2 < 0.0083 and Q.D2 < 0.72:
        z += -2340.0 * (0.0083 - Q.sum_z_dr2) * (0.72 - Q.D2)
    if Q.mass_over_sum_pt < 0.15 and Q.D2 < 0.71:
        z += 116.0 * (0.15 - Q.mass_over_sum_pt) * (0.71 - Q.D2)
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
