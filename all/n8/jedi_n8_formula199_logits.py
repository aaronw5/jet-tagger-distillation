"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; all observables), as if-statements, with each class score (logit) written out as a formula.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 87.2% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
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
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        max_dr=max(dr[i] for i in real),
        n_for_90pct=ncum(0.9),
        pt_6=pt[6],
        pt_7=pt[7],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr23=subjets(3)["dr"][2],
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -0.0143
    if Q.C2_b2 < 0.0016:
        z += 1510.0 * Q.C2_b2 - 2.416
    if Q.e2 < 0.025:
        z += -197.0 * Q.e2 + 4.925
    if Q.girth < 0.088:
        z += 60.8 * Q.girth - 5.3504
    if Q.log_sum_pt >= 6.7:
        z += -17.3 * Q.log_sum_pt + 115.91
    if Q.mass < 18.0:
        z += -0.1303 * Q.mass + 0.8368
    if 18.0 <= Q.mass < 56.0:
        z += 0.0397 * Q.mass - 2.2232
    if Q.n_dr_0_0p05 >= 4.0:
        z += 0.338 * Q.n_dr_0_0p05 - 1.352
    if Q.planar_flow < 0.15:
        z += -16.3 * Q.planar_flow + 2.445
    if Q.sj3_dr_max >= 0.23:
        z += -27.8 * Q.sj3_dr_max + 6.394
    if Q.sum_pt >= 900.0:
        z += -0.0179 * Q.sum_pt + 16.11
    if Q.sum_pt_top5 >= 690.0:
        z += 0.0163 * Q.sum_pt_top5 - 11.247
    if Q.width < 0.0044:
        z += 1140.0 * Q.width - 0.264
    if 0.0044 <= Q.width < 0.0088:
        z += -1080.0 * Q.width + 9.504
    if Q.girth2 < 0.013 and Q.D2 < 1.0:
        z += 455.0 * (0.013 - Q.girth2) * (1.0 - Q.D2)
    if Q.girth2 < 0.013 and Q.centroid_offset > 0.018:
        z += -12700.0 * (0.013 - Q.girth2) * (Q.centroid_offset - 0.018)
    if Q.lam1 < 0.0065 and Q.D2 < 0.87:
        z += -3060.0 * (0.0065 - Q.lam1) * (0.87 - Q.D2)
    if Q.lam2 < 7.1e-05 and Q.D2_b2 < 0.28:
        z += 75500.0 * (7.1e-05 - Q.lam2) * (0.28 - Q.D2_b2)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.0983
    if Q.girth < 0.076:
        z += 49.3 * Q.girth - 3.7468
    if Q.girth2 < 0.0088:
        z += 333.0 * Q.girth2 - 2.9304
    if 6.3 <= Q.log_sum_pt < 6.6:
        z += 7.09 * Q.log_sum_pt - 44.667
    if Q.log_sum_pt >= 6.6:
        z += 12.97 * Q.log_sum_pt - 83.475
    if Q.mass_over_sum_pt_sq < 0.0064:
        z += -584.0 * Q.mass_over_sum_pt_sq + 3.7376
    if Q.pt_7 >= 34.0:
        z += 0.125 * Q.pt_7 - 4.25
    if Q.sj3_dr_max >= 0.19:
        z += 6.69 * Q.sj3_dr_max - 1.2711
    if Q.sj3_dr_min >= 0.041:
        z += -10.5 * Q.sj3_dr_min + 0.4305
    if Q.z_7 < 0.06:
        z += 79.9 * Q.z_7 - 4.794
    if Q.pt_7 > 33.0 and Q.tau2 < 0.017:
        z += -7.43 * (Q.pt_7 - 33.0) * (0.017 - Q.tau2)
    return max(0.0, z)


def neuron_2(Q):
    z = 1.76
    if Q.LHA >= 0.13:
        z += -8.76 * Q.LHA + 1.1388
    if Q.N2 >= 0.12:
        z += 9.81 * Q.N2 - 1.1772
    if Q.lam1 < 0.006:
        z += -312.0 * Q.lam1 + 1.872
    if Q.log_sum_pt < 6.5:
        z += -2.77 * Q.log_sum_pt + 18.005
    if Q.mass < 36.0:
        z += 0.0504 * Q.mass - 1.8144
    if Q.pt_7 < 53.0:
        z += 0.0531 * Q.pt_7 - 2.8143
    if Q.sum_pt < 810.0:
        z += -0.0055 * Q.sum_pt + 4.455
    if Q.sum_pt_top5 >= 750.0:
        z += -0.00806 * Q.sum_pt_top5 + 6.045
    if Q.pt_7 < 53.0 and Q.D2_b2 < 1.1:
        z += 0.0373 * (53.0 - Q.pt_7) * (1.1 - Q.D2_b2)
    return max(0.0, z)


def neuron_3(Q):
    z = -5.26
    if Q.e2 >= 0.064:
        z += 322.0 * Q.e2 - 20.608
    if 0.042 <= Q.girth < 0.078:
        z += 97.1 * Q.girth - 4.0782
    if Q.girth >= 0.078:
        z += 194.4 * Q.girth - 11.6676
    if Q.girth2 < 0.0044:
        z += 872.0 * Q.girth2 - 3.8368
    if Q.girth2 >= 0.0089:
        z += -1670.0 * Q.girth2 + 14.863
    if Q.lam1 >= 0.0081:
        z += 578.0 * Q.lam1 - 4.6818
    if Q.mass >= 65.0:
        z += -0.0864 * Q.mass + 5.616
    if Q.sj2_dr >= 0.18:
        z += 37.7 * Q.sj2_dr - 6.786
    if Q.centroid_offset > 0.01 and Q.n_pt_above_50 < 8.4:
        z += 12.6 * (Q.centroid_offset - 0.01) * (8.4 - Q.n_pt_above_50)
    if Q.girth > 0.038 and Q.log_sum_pt > 6.1:
        z += 113.0 * (Q.girth - 0.038) * (Q.log_sum_pt - 6.1)
    if Q.lam1 > 0.016 and Q.n_for_90pct < 6.0:
        z += 3750.0 * (Q.lam1 - 0.016) * (6.0 - Q.n_for_90pct)
    if Q.mass_over_sum_pt > 0.066 and Q.sj2_dr < 0.22:
        z += -4280.0 * (Q.mass_over_sum_pt - 0.066) * (0.22 - Q.sj2_dr)
    return max(0.0, z)


def neuron_4(Q):
    z = 3.48
    if Q.N2 < 0.21:
        z += -56.3 * Q.N2 + 11.823
    if Q.centroid_offset < 0.056:
        z += 57.5 * Q.centroid_offset - 3.22
    if Q.girth2_top2 < 0.0082:
        z += -414.0 * Q.girth2_top2 + 3.3948
    if Q.lam2 < 0.00054:
        z += 11000.0 * Q.lam2 - 5.94
    if Q.mass >= 76.0:
        z += -0.223 * Q.mass + 16.948
    if Q.mass_over_sum_pt >= 0.086:
        z += -121.0 * Q.mass_over_sum_pt + 10.406
    if Q.max_dr >= 0.15:
        z += 24.9 * Q.max_dr - 3.735
    if Q.sd_mass >= 43.0:
        z += 0.166 * Q.sd_mass - 7.138
    if Q.sj3_dr_max < 0.22:
        z += 22.7 * Q.sj3_dr_max - 4.994
    if Q.width < 0.0017:
        z += -4080.0 * Q.width + 6.936
    if Q.N2 < 0.22 and Q.pt_7 < 53.0:
        z += -0.72 * (0.22 - Q.N2) * (53.0 - Q.pt_7)
    if Q.girth2_top2 < 0.0093 and Q.C2_b2 > 0.00095:
        z += -61500.0 * (0.0093 - Q.girth2_top2) * (Q.C2_b2 - 0.00095)
    if Q.sd_mass > 43.0 and Q.sd_zg < 0.29:
        z += -0.966 * (Q.sd_mass - 43.0) * (0.29 - Q.sd_zg)
    if Q.width > 0.005 and Q.sj3_z3 < 0.11:
        z += -5930.0 * (Q.width - 0.005) * (0.11 - Q.sj3_z3)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.0645
    if Q.e2_sq < 0.0055:
        z += -443.0 * Q.e2_sq + 2.4365
    if Q.girth < 0.0074:
        z += -1050.0 * Q.girth + 7.77
    if 6.7 <= Q.log_sum_pt < 6.9:
        z += -12.8 * Q.log_sum_pt + 85.76
    if Q.log_sum_pt >= 6.9:
        z += -56.1 * Q.log_sum_pt + 384.53
    if Q.z_7 < 0.029:
        z += -277.7 * Q.z_7 + 9.6689
    if 0.029 <= Q.z_7 < 0.057:
        z += -57.7 * Q.z_7 + 3.2889
    if Q.LHA < 0.22 and Q.log_sum_pt < 6.8:
        z += -186.0 * (0.22 - Q.LHA) * (6.8 - Q.log_sum_pt)
    if Q.girth2 < 0.0017 and Q.centroid_offset < 0.03:
        z += 124000.0 * (0.0017 - Q.girth2) * (0.03 - Q.centroid_offset)
    if Q.sum_pt_top5 > 350.0 and Q.sj2_dr > 0.098:
        z += 0.0319 * (Q.sum_pt_top5 - 350.0) * (Q.sj2_dr - 0.098)
    if Q.z_7 < 0.075 and Q.centroid_offset < 0.025:
        z += -2100.0 * (0.075 - Q.z_7) * (0.025 - Q.centroid_offset)
    return max(0.0, z)


def neuron_6(Q):
    z = 3.03
    if 0.0079 <= Q.centroid_offset < 0.019:
        z += 150.0 * Q.centroid_offset - 1.185
    if Q.centroid_offset >= 0.019:
        z += 13.0 * Q.centroid_offset + 1.418
    if Q.e2_sq < 0.0032:
        z += -1150.0 * Q.e2_sq + 3.68
    if Q.girth2 < 0.0087:
        z += 1620.0 * Q.girth2 - 14.094
    if Q.lam2 < 0.0012:
        z += 2270.0 * Q.lam2 - 2.724
    if Q.max_dr < 0.14:
        z += 41.3 * Q.max_dr - 5.782
    if Q.sj3_dr_max < 0.18:
        z += -46.0 * Q.sj3_dr_max + 8.28
    if Q.sj3_dr_max >= 0.18:
        z += 9.18 * Q.sj3_dr_max - 1.6524
    if Q.sj3_dr_min >= 0.14:
        z += -28.4 * Q.sj3_dr_min + 3.976
    if Q.sj3_pair_mass_min >= 4.0:
        z += -0.101 * Q.sj3_pair_mass_min + 0.404
    if Q.tau1 < 0.11:
        z += -71.3 * Q.tau1 + 7.843
    if Q.lam2 < 0.0028 and Q.n_dr_0_0p05 < 3.1:
        z += 96.0 * (0.0028 - Q.lam2) * (3.1 - Q.n_dr_0_0p05)
    if Q.mass < 64.0 and Q.z_dr_0p2_0p4 < 0.052:
        z += -0.798 * (64.0 - Q.mass) * (0.052 - Q.z_dr_0p2_0p4)
    if Q.pt_6 < 41.0 and Q.log_sum_pt < 6.8:
        z += 1.02 * (41.0 - Q.pt_6) * (6.8 - Q.log_sum_pt)
    if Q.pt_6 < 41.0 and Q.z_7 > 0.023:
        z += -12.0 * (41.0 - Q.pt_6) * (Q.z_7 - 0.023)
    return max(0.0, z)


def neuron_7(Q):
    z = 6.92
    if Q.e2 >= 0.05:
        z += -183.0 * Q.e2 + 9.15
    if Q.e3 < 8.1e-05:
        z += 38200.0 * Q.e3 - 3.0942
    if Q.girth < 0.086:
        z += 82.6 * Q.girth - 7.1036
    if Q.girth2 < 0.001:
        z += 2720.0 * Q.girth2 - 2.72
    if Q.girth2 >= 0.013:
        z += 1660.0 * Q.girth2 - 21.58
    if 36.0 <= Q.mass < 77.0:
        z += 0.0537 * Q.mass - 1.9332
    if Q.mass >= 77.0:
        z += -0.1593 * Q.mass + 14.4678
    if 0.075 <= Q.mass_over_sum_pt < 0.091:
        z += 159.0 * Q.mass_over_sum_pt - 11.925
    if Q.mass_over_sum_pt >= 0.091:
        z += -39.0 * Q.mass_over_sum_pt + 6.093
    if Q.sj2_dr < 0.16:
        z += -18.9 * Q.sj2_dr + 1.674
    if 0.16 <= Q.sj2_dr < 0.19:
        z += 45.0 * Q.sj2_dr - 8.55
    if Q.tau1 < 0.059:
        z += -54.4 * Q.tau1 + 3.2096
    if Q.width >= 0.0056:
        z += -1330.0 * Q.width + 7.448
    if Q.girth2 > 0.0032 and Q.planar_flow < 0.21:
        z += 1570.0 * (Q.girth2 - 0.0032) * (0.21 - Q.planar_flow)
    if Q.planar_flow < 0.23 and Q.width > 0.008:
        z += -2500.0 * (0.23 - Q.planar_flow) * (Q.width - 0.008)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.681
    if Q.LHA < 0.21:
        z += 60.8 * Q.LHA - 12.768
    if Q.log_sum_pt >= 6.7:
        z += -5.19 * Q.log_sum_pt + 34.773
    if Q.sj3_dr_max < 0.13:
        z += 22.7 * Q.sj3_dr_max - 2.951
    if Q.girth2 < 0.0063 and Q.centroid_offset < 0.025:
        z += 48600.0 * (0.0063 - Q.girth2) * (0.025 - Q.centroid_offset)
    if Q.tau1 < 0.058 and Q.width < 0.0032:
        z += 26900.0 * (0.058 - Q.tau1) * (0.0032 - Q.width)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.0914
    if Q.C3 < 0.028:
        z += 81.1 * Q.C3 - 2.2708
    if Q.centroid_offset < 0.019:
        z += -173.0 * Q.centroid_offset + 3.287
    if Q.e2 >= 0.047:
        z += 100.0 * Q.e2 - 4.7
    if Q.girth >= 0.069:
        z += -68.5 * Q.girth + 4.7265
    if Q.girth2 < 0.0035:
        z += -2166.0 * Q.girth2 + 9.4674
    if 0.0035 <= Q.girth2 < 0.0059:
        z += -786.0 * Q.girth2 + 4.6374
    if Q.lam2 >= 0.00083:
        z += 784.0 * Q.lam2 - 0.65072
    if 6.4 <= Q.log_sum_pt < 6.9:
        z += -7.62 * Q.log_sum_pt + 48.768
    if Q.log_sum_pt >= 6.9:
        z += 2.02 * Q.log_sum_pt - 17.748
    if Q.mass < 55.0:
        z += 0.102 * Q.mass - 5.61
    if Q.max_dr < 0.11:
        z += -55.8 * Q.max_dr + 6.138
    if Q.sj3_dr_max < 0.14:
        z += 54.3 * Q.sj3_dr_max - 6.216
    if 0.14 <= Q.sj3_dr_max < 0.21:
        z += -19.8 * Q.sj3_dr_max + 4.158
    if Q.sum_pt < 570.0:
        z += -0.0224 * Q.sum_pt + 12.768
    if Q.centroid_offset < 0.02 and Q.n_for_90pct > 5.5:
        z += -65.5 * (0.02 - Q.centroid_offset) * (Q.n_for_90pct - 5.5)
    if Q.girth2 < 0.0086 and Q.planar_flow < 0.33:
        z += 627.0 * (0.0086 - Q.girth2) * (0.33 - Q.planar_flow)
    if Q.mass < 54.0 and Q.centroid_offset < 0.027:
        z += 6.06 * (54.0 - Q.mass) * (0.027 - Q.centroid_offset)
    return max(0.0, z)


def neuron_10(Q):
    z = 0.34
    if Q.C2_b2 >= 0.0093:
        z += 128.0 * Q.C2_b2 - 1.1904
    if Q.LHA >= 0.3:
        z += -33.2 * Q.LHA + 9.96
    if Q.e3 < 7.5e-05:
        z += -25100.0 * Q.e3 + 2.008
    if 7.5e-05 <= Q.e3 < 8e-05:
        z += -30710.0 * Q.e3 + 2.42875
    if Q.e3 >= 8e-05:
        z += -5610.0 * Q.e3 + 0.42075
    if Q.girth2 >= 0.0075:
        z += 340.0 * Q.girth2 - 2.55
    if Q.lam1 < 0.0015:
        z += 2449.0 * Q.lam1 - 5.8424
    if 0.0015 <= Q.lam1 < 0.0056:
        z += 529.0 * Q.lam1 - 2.9624
    if 0.00034 <= Q.lam2 < 0.0034:
        z += 871.0 * Q.lam2 - 0.29614
    if Q.lam2 >= 0.0034:
        z += 228.0 * Q.lam2 + 1.89006
    if Q.mass < 75.0:
        z += -0.03 * Q.mass + 2.25
    if Q.pt_7 < 46.0:
        z += 0.0398 * Q.pt_7 - 1.8308
    if Q.sj3_pair_mass_min >= 11.0:
        z += 0.101 * Q.sj3_pair_mass_min - 1.111
    if Q.tau1 >= 0.057:
        z += 28.0 * Q.tau1 - 1.596
    if Q.e3 < 8.1e-05 and Q.sj3_dr23 > 0.18:
        z += -141000.0 * (8.1e-05 - Q.e3) * (Q.sj3_dr23 - 0.18)
    if Q.lam1 > 0.0074 and Q.D2_b2 < 0.38:
        z += -554.0 * (Q.lam1 - 0.0074) * (0.38 - Q.D2_b2)
    if Q.pt_7 < 45.0 and Q.D2 < 0.98:
        z += 0.129 * (45.0 - Q.pt_7) * (0.98 - Q.D2)
    if Q.zdr_0 < 0.022 and Q.z_dr_0p05_0p1 < 0.24:
        z += 204.0 * (0.022 - Q.zdr_0) * (0.24 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.835
    if Q.centroid_offset < 0.01:
        z += -137.0 * Q.centroid_offset + 1.37
    if Q.centroid_offset >= 0.01:
        z += -155.0 * Q.centroid_offset + 1.55
    if Q.girth < 0.021:
        z += 334.0 * Q.girth - 10.635
    if 0.021 <= Q.girth < 0.072:
        z += 71.0 * Q.girth - 5.112
    if Q.mass < 16.0:
        z += -0.182 * Q.mass + 2.912
    if Q.max_dr < 0.15:
        z += -15.0 * Q.max_dr + 2.25
    if Q.sj2_dr >= 0.27:
        z += 23.6 * Q.sj2_dr - 6.372
    if Q.sj3_dr_max < 0.17:
        z += 36.5 * Q.sj3_dr_max - 4.594
    if 0.17 <= Q.sj3_dr_max < 0.26:
        z += -17.9 * Q.sj3_dr_max + 4.654
    if Q.width < 0.0094:
        z += -708.0 * Q.width + 6.6552
    if Q.z_7 >= 0.018:
        z += 35.7 * Q.z_7 - 0.6426
    if Q.centroid_offset > 0.05 and Q.D2 < 2.8:
        z += -212.0 * (Q.centroid_offset - 0.05) * (2.8 - Q.D2)
    if Q.centroid_offset < 0.041 and Q.sum_pt < 900.0:
        z += -0.247 * (0.041 - Q.centroid_offset) * (900.0 - Q.sum_pt)
    if Q.centroid_offset > 0.047 and Q.tau32 < 0.64:
        z += 1160.0 * (Q.centroid_offset - 0.047) * (0.64 - Q.tau32)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.37
    if Q.e2 >= 0.063:
        z += -79.3 * Q.e2 + 4.9959
    if Q.girth2 >= 0.019:
        z += 347.0 * Q.girth2 - 6.593
    if Q.mass >= 91.0:
        z += 0.0953 * Q.mass - 8.6723
    return max(0.0, z)


def neuron_13(Q):
    z = 0.0376
    if Q.e2 < 0.08:
        z += 41.0 * Q.e2 - 3.28
    if Q.girth < 0.15:
        z += -52.6 * Q.girth + 7.89
    if Q.lam1 < 0.016:
        z += -245.0 * Q.lam1 + 3.92
    if Q.z_7 < 0.028:
        z += 306.0 * Q.z_7 - 8.568
    if Q.e3 < 5.8e-05 and Q.D3 > 0.16:
        z += 9000.0 * (5.8e-05 - Q.e3) * (Q.D3 - 0.16)
    if Q.e3 < 6.1e-05 and Q.centroid_offset < 0.038:
        z += -799000.0 * (6.1e-05 - Q.e3) * (0.038 - Q.centroid_offset)
    if Q.girth < 0.15 and Q.log_sum_pt < 6.8:
        z += -50.1 * (0.15 - Q.girth) * (6.8 - Q.log_sum_pt)
    if Q.girth < 0.15 and Q.pt_7 < 38.0:
        z += -1.48 * (0.15 - Q.girth) * (38.0 - Q.pt_7)
    if Q.sum_pt > 990.0 and Q.pt_6 < 62.0:
        z += -0.000625 * (Q.sum_pt - 990.0) * (62.0 - Q.pt_6)
    if Q.sum_pt_top5 > 660.0 and Q.pt_7 < 41.0:
        z += 0.0011 * (Q.sum_pt_top5 - 660.0) * (41.0 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = -1.75
    if 0.026 <= Q.centroid_offset < 0.05:
        z += -107.0 * Q.centroid_offset + 2.782
    if Q.centroid_offset >= 0.05:
        z += -1217.0 * Q.centroid_offset + 58.282
    if 0.017 <= Q.e2 < 0.042:
        z += -218.0 * Q.e2 + 3.706
    if 0.042 <= Q.e2 < 0.05:
        z += -80.0 * Q.e2 - 2.09
    if Q.e2 >= 0.05:
        z += 293.0 * Q.e2 - 20.74
    if Q.e3 < 5.1e-05:
        z += 25100.0 * Q.e3 - 1.2801
    if 0.027 <= Q.girth < 0.042:
        z += 109.0 * Q.girth - 2.943
    if 0.042 <= Q.girth < 0.089:
        z += 174.0 * Q.girth - 5.673
    if 0.089 <= Q.girth < 0.094:
        z += -83.0 * Q.girth + 17.2
    if Q.girth >= 0.094:
        z += -130.4 * Q.girth + 21.6556
    if Q.girth2 >= 0.0075:
        z += -2990.0 * Q.girth2 + 22.425
    if Q.mass_over_sum_pt >= 0.085:
        z += 327.0 * Q.mass_over_sum_pt - 27.795
    if Q.sd_mass < 50.0:
        z += -0.0053 * Q.sd_mass - 1.0425
    if 50.0 <= Q.sd_mass < 75.0:
        z += 0.0523 * Q.sd_mass - 3.9225
    if 0.16 <= Q.sj2_dr < 0.2:
        z += 41.1 * Q.sj2_dr - 6.576
    if Q.sj2_dr >= 0.2:
        z += 21.1 * Q.sj2_dr - 2.576
    if 0.0033 <= Q.width < 0.014:
        z += 270.0 * Q.width - 0.891
    if Q.width >= 0.014:
        z += 1166.0 * Q.width - 13.435
    if Q.planar_flow < 0.1 and Q.centroid_offset < 0.018:
        z += -792.0 * (0.1 - Q.planar_flow) * (0.018 - Q.centroid_offset)
    if Q.planar_flow < 0.11 and Q.lam1 < 0.0072:
        z += -9930.0 * (0.11 - Q.planar_flow) * (0.0072 - Q.lam1)
    if Q.planar_flow < 0.11 and Q.lam1 < 0.016:
        z += 1300.0 * (0.11 - Q.planar_flow) * (0.016 - Q.lam1)
    if Q.planar_flow < 0.11 and Q.width < 0.0061:
        z += 9130.0 * (0.11 - Q.planar_flow) * (0.0061 - Q.width)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.101
    if Q.e2 < 0.04:
        z += -203.0 * Q.e2 + 8.12
    if Q.girth < 0.1:
        z += 97.9 * Q.girth - 9.79
    if Q.girth2 < 0.008:
        z += 1050.0 * Q.girth2 - 8.4
    if Q.girth2_top2 < 0.0005:
        z += 6550.0 * Q.girth2_top2 - 3.275
    if Q.sj2_dr < 0.16:
        z += -3.4 * Q.sj2_dr - 1.26
    if 0.16 <= Q.sj2_dr < 0.2:
        z += 45.1 * Q.sj2_dr - 9.02
    if Q.width < 0.014:
        z += -560.0 * Q.width + 7.84
    if Q.mass_over_sum_pt < 0.15 and Q.D2 < 0.73:
        z += 56.4 * (0.15 - Q.mass_over_sum_pt) * (0.73 - Q.D2)
    if Q.width < 0.0061 and Q.D2 < 0.72:
        z += -7420.0 * (0.0061 - Q.width) * (0.72 - Q.D2)
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


def logit_g(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return -0.4375 - 0.15625 * h0 + 0.390625 * h1 + 0.4296875 * h2 - 0.03125 * h4 - 0.1875 * h5 + 0.109375 * h6 + 0.171875 * h9


def logit_q(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 0.03125 - 0.09375 * h4 + 0.046875 * h5 + 0.125 * h6 + 0.0625 * h8 + 0.2539062 * h9 - 0.125 * h10 + 0.0625 * h15


def logit_W(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return -0.125 + 0.34375 * h0 - 0.03125 * h1 - 0.5 * h3 - 0.3125 * h6 + 0.21875 * h7 - 0.25 * h8 - 0.03125 * h9 + 0.375 * h11 + 0.0703125 * h13 - 0.75 * h14 - 0.6875 * h15


def logit_Z(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return -0.09375 + 0.125 * h1 - 0.5625 * h3 + 0.078125 * h4 + 0.015625 * h5 - 0.375 * h6 + 0.46875 * h7 - 0.03125 * h9 + 0.0546875 * h13 + 0.375 * h14 - 0.15625 * h15


def logit_t(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 1.34375 + 0.015625 * h0 + 0.0625 * h3 + 0.125 * h4 - 0.25 * h5 + 0.1875 * h8 + 0.375 * h10 - 0.5 * h12 - 0.40625 * h13


def logits(h):
    h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15 = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    return [logit_g(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15), logit_q(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15), logit_W(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15), logit_Z(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15), logit_t(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15)]


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
