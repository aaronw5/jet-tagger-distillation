"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; all observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.8% (the network: 65.8%); same class as the network for 87.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
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
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
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
        mass_top3=mass_of(3),
        sj2_mass1=subjets(2)["mass"][0],
        max_dr=max(dr[i] for i in real),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        pt_6=pt[6],
        pt_7=pt[7],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
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
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -2.547365
    if Q.planar_flow < 0.1484197:
        z += -17.12106 * Q.planar_flow + 2.541102
    if Q.width < 0.004372139:
        z += 1505.216 * Q.width - 4.019823
    if 0.004372139 <= Q.width < 0.008678045:
        z += -594.8094 * Q.width + 5.161783
    if Q.mass < 21.78408:
        z += -0.0649402 * Q.mass - 1.257102
    if 21.78408 <= Q.mass < 29.6447:
        z += 0.1937544 * Q.mass - 6.892527
    if 29.6447 <= Q.mass < 56.92035:
        z += 0.04211574 * Q.mass - 2.397243
    if Q.girth2 < 0.01323868:
        z += -290.5179 * Q.girth2 + 3.846072
    if Q.lam1 < 0.005433361:
        z += 87.68066 * Q.lam1 - 0.1158353
    if 0.005433361 <= Q.lam1 < 0.006506576:
        z += -335.9675 * Q.lam1 + 2.185998
    if Q.sum_pt >= 901.5938:
        z += -0.02317622 * Q.sum_pt + 20.89553
    if Q.C2_b2 < 0.001563465:
        z += 1656.548 * Q.C2_b2 - 2.589956
    if Q.sj3_dr_max < 0.1070199:
        z += -10.18314 * Q.sj3_dr_max + 3.067178
    if 0.1070199 <= Q.sj3_dr_max < 0.233678:
        z += -5.824914 * Q.sj3_dr_max + 2.600761
    if 0.233678 <= Q.sj3_dr_max < 0.3012016:
        z += -33.64275 * Q.sj3_dr_max + 9.101176
    if Q.sj3_dr_max >= 0.3012016:
        z += -23.4596 * Q.sj3_dr_max + 6.033998
    if Q.girth < 0.08723651:
        z += 47.34599 * Q.girth - 4.130299
    if Q.e2 < 0.0245477:
        z += -228.7248 * Q.e2 + 5.614668
    if Q.n_dr_0_0p05 >= 4.0:
        z += 0.360254 * Q.n_dr_0_0p05 - 1.441016
    if Q.log_sum_pt >= 6.670067:
        z += -14.56131 * Q.log_sum_pt + 97.12488
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.01695238 * Q.sum_pt_top5 - 11.6537
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += -3140.132 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.girth2 < 0.01323868 and Q.D2 < 1.002471:
        z += 435.0238 * (0.01323868 - Q.girth2) * (1.002471 - Q.D2)
    if Q.girth2 < 0.01323868 and Q.centroid_offset > 0.01837778:
        z += -12398.28 * (0.01323868 - Q.girth2) * (Q.centroid_offset - 0.01837778)
    if Q.lam2 < 7.300726e-05 and Q.D2_b2 < 0.2669656:
        z += 84155.38 * (7.300726e-05 - Q.lam2) * (0.2669656 - Q.D2_b2)
    return max(0.0, z)


def neuron_1(Q):
    z = -0.1230816
    if Q.pt_7 >= 34.53125:
        z += 0.1324637 * Q.pt_7 - 4.574136
    if 6.377723 <= Q.log_sum_pt < 6.502799:
        z += 4.910623 * Q.log_sum_pt - 31.31859
    if 6.502799 <= Q.log_sum_pt < 6.605974:
        z += 10.52663 * Q.log_sum_pt - 67.83838
    if Q.log_sum_pt >= 6.605974:
        z += 14.57885 * Q.log_sum_pt - 94.60719
    if Q.mass_over_sum_pt_sq < 0.005832932:
        z += -892.9637 * Q.mass_over_sum_pt_sq + 5.208596
    if Q.z_7 < 0.06164517:
        z += 75.00406 * Q.z_7 - 4.623638
    if Q.girth2 < 0.008678045:
        z += 584.8421 * Q.girth2 - 5.075286
    if Q.sj3_dr_min >= 0.03628191:
        z += -10.67569 * Q.sj3_dr_min + 0.3873343
    if Q.sj3_dr_max >= 0.169029:
        z += 7.170598 * Q.sj3_dr_max - 1.212039
    if Q.e2_sq < 0.008168571:
        z += -277.1018 * Q.e2_sq + 2.263526
    if 367.5938 <= Q.sum_pt_top5 < 531.1875:
        z += 0.005023588 * Q.sum_pt_top5 - 1.84664
    if Q.sum_pt_top5 >= 531.1875:
        z += -0.0004896973 * Q.sum_pt_top5 + 1.081949
    if Q.lam1 < 0.00595415:
        z += -519.7603 * Q.lam1 + 3.094731
    if Q.width < 0.00609665:
        z += 575.5693 * Q.width - 3.509045
    if Q.girth < 0.0717028:
        z += 64.09914 * Q.girth - 4.596088
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += -8.175155 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.log_sum_pt > 6.377723 and Q.z_dr_0p05_0p1 < 0.7509095:
        z += -2.146648 * (Q.log_sum_pt - 6.377723) * (0.7509095 - Q.z_dr_0p05_0p1)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.centroid_offset > 0.00809236:
        z += -7969.009 * (0.005832932 - Q.mass_over_sum_pt_sq) * (Q.centroid_offset - 0.00809236)
    return max(0.0, z)


def neuron_2(Q):
    z = 1.713223
    if Q.log_sum_pt < 6.46415:
        z += -2.684133 * Q.log_sum_pt + 17.35064
    if Q.lam1 < 0.00595415:
        z += -311.5759 * Q.lam1 + 1.855169
    if Q.LHA >= 0.1329373:
        z += -8.787485 * Q.LHA + 1.168184
    if Q.pt_7 < 53.4375:
        z += 0.05292109 * Q.pt_7 - 2.827971
    if Q.mass < 36.22941:
        z += 0.04941354 * Q.mass - 1.790223
    if Q.N2 >= 0.1150852:
        z += 9.748331 * Q.N2 - 1.121889
    if Q.sum_pt < 813.4156:
        z += -0.005845624 * Q.sum_pt + 4.754922
    if Q.sum_pt_top5 >= 752.1:
        z += -0.007983031 * Q.sum_pt_top5 + 6.004037
    if Q.pt_7 < 53.4375 and Q.D2_b2 < 1.129616:
        z += 0.03575489 * (53.4375 - Q.pt_7) * (1.129616 - Q.D2_b2)
    return max(0.0, z)


def neuron_3(Q):
    z = -4.505646
    if Q.mass_over_sum_pt >= 0.0681391:
        z += 25.31239 * Q.mass_over_sum_pt - 1.724763
    if 0.05356915 <= Q.tau1 < 0.1027642:
        z += -26.71264 * Q.tau1 + 1.430974
    if Q.tau1 >= 0.1027642:
        z += 27.16826 * Q.tau1 - 4.106056
    if 0.04081947 <= Q.girth < 0.07608178:
        z += 73.33973 * Q.girth - 2.993689
    if Q.girth >= 0.07608178:
        z += 146.3599 * Q.girth - 8.549194
    if 0.007520088 <= Q.width < 0.01323868:
        z += 481.726 * Q.width - 3.622622
    if Q.width >= 0.01323868:
        z += 331.1221 * Q.width - 1.628826
    if Q.e2 >= 0.06344108:
        z += 303.5563 * Q.e2 - 19.25794
    if Q.girth2 < 0.004372139:
        z += 833.7736 * Q.girth2 - 3.645375
    if Q.girth2 >= 0.008678045:
        z += -1590.494 * Q.girth2 + 13.80238
    if Q.sj2_dr >= 0.1872617:
        z += 49.59021 * Q.sj2_dr - 9.286349
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += 371.8363 * Q.lam1 - 3.114342
    if Q.lam1 >= 0.01200373:
        z += 123.6518 * Q.lam1 - 0.1352034
    if Q.lam2 >= 0.001130645:
        z += -829.1944 * Q.lam2 + 0.9375243
    if Q.mass >= 64.61873:
        z += -0.05985048 * Q.mass + 3.867462
    if Q.sd_mass >= 62.55:
        z += 0.05348794 * Q.sd_mass - 3.345671
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj3_pair_mass_min > 5.744224:
        z += -3.078744 * (Q.mass_over_sum_pt - 0.0681391) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.girth > 0.04081947 and Q.log_sum_pt > 6.080494:
        z += 135.1412 * (Q.girth - 0.04081947) * (Q.log_sum_pt - 6.080494)
    if Q.e2 > 0.06344108 and Q.sj2_mass1 > 16.86126:
        z += 5.768425 * (Q.e2 - 0.06344108) * (Q.sj2_mass1 - 16.86126)
    if Q.sj2_dr > 0.1872617 and Q.sj2_mass1 > 2.250113:
        z += -0.4136194 * (Q.sj2_dr - 0.1872617) * (Q.sj2_mass1 - 2.250113)
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj2_dr < 0.2179769:
        z += -4159.167 * (Q.mass_over_sum_pt - 0.0681391) * (0.2179769 - Q.sj2_dr)
    if Q.width > 0.007520088 and Q.sj3_pairmin_over_m > 0.07708997:
        z += 767.6512 * (Q.width - 0.007520088) * (Q.sj3_pairmin_over_m - 0.07708997)
    if Q.centroid_offset > 0.01096064 and Q.n_pt_above_50 < 8.0:
        z += 12.96784 * (Q.centroid_offset - 0.01096064) * (8.0 - Q.n_pt_above_50)
    if Q.lam2 > 0.001130645 and Q.pt_6 < 56.53125:
        z += 27.12811 * (Q.lam2 - 0.001130645) * (56.53125 - Q.pt_6)
    if Q.sd_mass > 62.55 and Q.D2_b2 < 0.9206502:
        z += -0.1051442 * (Q.sd_mass - 62.55) * (0.9206502 - Q.D2_b2)
    if Q.lam1 > 0.01643375 and Q.n_for_90pct < 6.0:
        z += 3897.237 * (Q.lam1 - 0.01643375) * (6.0 - Q.n_for_90pct)
    if Q.max_dr > 0.1027585 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += -11.09927 * (Q.max_dr - 0.1027585) * (0.8460335 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_4(Q):
    z = -1.984515
    if Q.N2 < 0.2233283:
        z += -61.9689 * Q.N2 + 13.83941
    if Q.lam2 < 0.000537286:
        z += 10570.73 * Q.lam2 - 5.679507
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -149.41 * Q.mass_over_sum_pt + 13.50873
    if Q.width < 0.001653836:
        z += -3246.117 * Q.width + 5.368546
    if Q.width >= 0.003562611:
        z += 376.1682 * Q.width - 1.340141
    if Q.e2 < 0.04447357:
        z += 85.26767 * Q.e2 - 4.287654
    if 0.04447357 <= Q.e2 < 0.05028464:
        z += 158.791 * Q.e2 - 7.557499
    if Q.e2 >= 0.05028464:
        z += 73.52333 * Q.e2 - 3.269845
    if Q.sum_pt < 739.5:
        z += 0.003755188 * Q.sum_pt - 2.776961
    if Q.max_dr < 0.1598486:
        z += 1.867846 * Q.max_dr - 0.3311784
    if 0.1598486 <= Q.max_dr < 0.177305:
        z += 32.76727 * Q.max_dr - 5.27041
    if Q.max_dr >= 0.177305:
        z += 30.89943 * Q.max_dr - 4.939231
    if Q.C2 < 0.06729223:
        z += -79.57784 * Q.C2 + 5.35497
    if Q.girth >= 0.08723651:
        z += -42.64309 * Q.girth + 3.720034
    if 44.82259 <= Q.sd_mass < 74.57663:
        z += 0.1290954 * Q.sd_mass - 5.786392
    if Q.sd_mass >= 74.57663:
        z += 0.2210287 * Q.sd_mass - 12.64246
    if Q.e3 < 0.0001869378:
        z += -14224.94 * Q.e3 + 2.65918
    if Q.sj3_dr_max < 0.233678:
        z += 35.53963 * Q.sj3_dr_max - 7.670636
    if 0.233678 <= Q.sj3_dr_max < 0.3456459:
        z += -5.664068 * Q.sj3_dr_max + 1.957762
    if Q.girth2 < 0.008678045:
        z += 197.5795 * Q.girth2 - 1.714604
    if Q.centroid_offset < 0.04990367:
        z += 73.89394 * Q.centroid_offset - 3.687579
    if Q.girth2_top2 < 0.007639643:
        z += -390.8946 * Q.girth2_top2 + 2.986295
    if Q.mass >= 76.6557:
        z += -0.2999585 * Q.mass + 22.99353
    if Q.e2_sq < 0.01165737:
        z += -3214.91 * Q.e2_sq + 37.47741
    if Q.mass_over_sum_pt_sq < 0.0116609:
        z += 2661.375 * Q.mass_over_sum_pt_sq - 31.03403
    if Q.LHA < 0.3467135:
        z += -8.220409 * Q.LHA + 2.850127
    if Q.N2 < 0.2233283 and Q.mass < 62.55:
        z += -0.7834277 * (0.2233283 - Q.N2) * (62.55 - Q.mass)
    if Q.N2 < 0.2233283 and Q.pt_7 < 53.4375:
        z += -0.7519606 * (0.2233283 - Q.N2) * (53.4375 - Q.pt_7)
    if Q.N2 < 0.2233283 and Q.eccentricity > 0.7117266:
        z += -73.94671 * (0.2233283 - Q.N2) * (Q.eccentricity - 0.7117266)
    if Q.sd_mass > 44.82259 and Q.centroid_offset > 0.001308549:
        z += -1.730639 * (Q.sd_mass - 44.82259) * (Q.centroid_offset - 0.001308549)
    if Q.sd_mass > 44.82259 and Q.sd_zg < 0.275762:
        z += -0.9066542 * (Q.sd_mass - 44.82259) * (0.275762 - Q.sd_zg)
    if Q.width > 0.003562611 and Q.sj3_z3 < 0.1057566:
        z += -5836.058 * (Q.width - 0.003562611) * (0.1057566 - Q.sj3_z3)
    if Q.girth2_top2 < 0.007639643 and Q.C2_b2 > 0.0006435798:
        z += -56676.59 * (0.007639643 - Q.girth2_top2) * (Q.C2_b2 - 0.0006435798)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.7058495
    if Q.LHA < 0.2160559:
        z += -17.35268 * Q.LHA + 3.749148
    if Q.z_7 < 0.02807091:
        z += -319.7937 * Q.z_7 + 10.67512
    if 0.02807091 <= Q.z_7 < 0.04939969:
        z += -79.62085 * Q.z_7 + 3.933246
    if 6.701242 <= Q.log_sum_pt < 6.896095:
        z += -18.88884 * Q.log_sum_pt + 126.5787
    if Q.log_sum_pt >= 6.896095:
        z += -77.10687 * Q.log_sum_pt + 528.0558
    if Q.e2_sq < 0.002074109:
        z += 265.0399 * Q.e2_sq + 1.514676
    if 0.002074109 <= Q.e2_sq < 0.005284669:
        z += -643.0026 * Q.e2_sq + 3.398055
    if Q.girth2 < 0.001653836:
        z += -1925.216 * Q.girth2 + 3.183992
    if Q.girth < 0.007673833:
        z += -996.9643 * Q.girth + 7.650538
    if Q.pt_7 < 37.15625:
        z += 0.0638411 * Q.pt_7 - 2.372096
    if Q.zdr_0 < 0.0211821:
        z += 154.6689 * Q.zdr_0 - 3.276213
    if Q.width < 0.003562611:
        z += -385.7445 * Q.width + 1.374258
    if Q.girth2_top3 < 0.002151568:
        z += 615.4584 * Q.girth2_top3 - 1.324201
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -159.9139 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.e2_sq < 0.005284669 and Q.centroid_offset < 0.01437952:
        z += -13172.7 * (0.005284669 - Q.e2_sq) * (0.01437952 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.centroid_offset < 0.03117077:
        z += -1771.377 * (0.07148865 - Q.z_7) * (0.03117077 - Q.centroid_offset)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.05744392:
        z += 1540.288 * (Q.log_sum_pt - 6.896095) * (0.05744392 - Q.D2_b2)
    if Q.z_7 < 0.07148865 and Q.mass_top3 < 40.2:
        z += 0.5976955 * (0.07148865 - Q.z_7) * (40.2 - Q.mass_top3)
    if Q.girth2 < 0.001653836 and Q.centroid_offset < 0.02355416:
        z += 151167.2 * (0.001653836 - Q.girth2) * (0.02355416 - Q.centroid_offset)
    if Q.LHA < 0.2160559 and Q.lam1 < 0.001503553:
        z += -13584.06 * (0.2160559 - Q.LHA) * (0.001503553 - Q.lam1)
    if Q.log_sum_pt > 6.701242 and Q.dr_2 < 0.01341502:
        z += 1026.925 * (Q.log_sum_pt - 6.701242) * (0.01341502 - Q.dr_2)
    if Q.sum_pt_top5 > 430.75 and Q.pt1_dr01 < 28.39396:
        z += 0.0001369071 * (Q.sum_pt_top5 - 430.75) * (28.39396 - Q.pt1_dr01)
    if Q.sum_pt_top5 > 430.75 and Q.sj2_dr > 0.1682655:
        z += 0.03659241 * (Q.sum_pt_top5 - 430.75) * (Q.sj2_dr - 0.1682655)
    if Q.zdr_0 < 0.0211821 and Q.sj3_dr13 > 0.04995258:
        z += 693.5255 * (0.0211821 - Q.zdr_0) * (Q.sj3_dr13 - 0.04995258)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.5173415
    if 0.00809236 <= Q.centroid_offset < 0.01837778:
        z += 153.643 * Q.centroid_offset - 1.243335
    if Q.centroid_offset >= 0.01837778:
        z += -27.77963 * Q.centroid_offset + 2.090811
    if Q.width < 0.01323868:
        z += 354.7936 * Q.width - 4.696997
    if Q.pt_6 < 39.75:
        z += 0.02021047 * Q.pt_6 - 0.6881945
    if 39.75 <= Q.pt_6 < 41.21875:
        z += -0.07841492 * Q.pt_6 + 3.232165
    if Q.tau1 < 0.1136369:
        z += -53.59186 * Q.tau1 + 6.090013
    if Q.sj3_dr_min >= 0.1278212:
        z += -17.47437 * Q.sj3_dr_min + 2.233594
    if Q.lam1 < 0.00733008:
        z += -435.0072 * Q.lam1 + 4.565081
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += -294.5117 * Q.lam1 + 3.535237
    if Q.sj3_pair_mass_min >= 4.501727:
        z += -0.09517011 * Q.sj3_pair_mass_min + 0.4284298
    if Q.e2_sq < 0.0030133:
        z += -1311.96 * Q.e2_sq + 3.953329
    if Q.lam2 < 0.001130645:
        z += 1879.342 * Q.lam2 - 2.124868
    if Q.sj3_dr_max < 0.1789613:
        z += -43.21113 * Q.sj3_dr_max + 7.733122
    if Q.sj3_dr_max >= 0.1879486:
        z += 13.03165 * Q.sj3_dr_max - 2.449281
    if Q.max_dr < 0.1452311:
        z += 41.4775 * Q.max_dr - 6.023824
    if Q.e3 < 0.0005116989:
        z += -5599.613 * Q.e3 + 2.865316
    if Q.mass < 45.595:
        z += -0.02568978 * Q.mass + 1.171326
    if Q.girth2 < 0.008678045:
        z += 1487.827 * Q.girth2 - 12.91143
    if Q.centroid_offset > 0.00809236 and Q.sj3_pair_mass_min > 6.811308:
        z += 2.134053 * (Q.centroid_offset - 0.00809236) * (Q.sj3_pair_mass_min - 6.811308)
    if Q.pt_6 < 41.21875 and Q.log_sum_pt < 6.766778:
        z += 1.196886 * (41.21875 - Q.pt_6) * (6.766778 - Q.log_sum_pt)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.008921136:
        z += 8726.583 * (Q.centroid_offset - 0.01837778) * (0.008921136 - Q.mean_phi2)
    if Q.pt_6 < 41.21875 and Q.z_7 > 0.02320757:
        z += -13.67302 * (41.21875 - Q.pt_6) * (Q.z_7 - 0.02320757)
    if Q.mass < 60.63098 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -1.093805 * (60.63098 - Q.mass) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.lam2 < 0.003408389 and Q.n_dr_0_0p05 < 3.0:
        z += 88.03266 * (0.003408389 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.sj3_pair_mass_min > 4.501727 and Q.n_dr_0p2_0p4 > 1.0:
        z += -0.0335275 * (Q.sj3_pair_mass_min - 4.501727) * (Q.n_dr_0p2_0p4 - 1.0)
    return max(0.0, z)


def neuron_7(Q):
    z = 8.568775
    if Q.girth2 < 0.0009641429:
        z += 2626.53 * Q.girth2 - 2.53235
    if 0.001653836 <= Q.girth2 < 0.004372139:
        z += 490.7249 * Q.girth2 - 0.8115786
    if 0.004372139 <= Q.girth2 < 0.007520088:
        z += 118.3871 * Q.girth2 + 0.816334
    if 0.007520088 <= Q.girth2 < 0.01323868:
        z += -858.5552 * Q.girth2 + 8.163026
    if Q.girth2 >= 0.01323868:
        z += 938.0216 * Q.girth2 - 15.62127
    if 0.01109984 <= Q.mass_over_sum_pt < 0.07269073:
        z += -48.20671 * Q.mass_over_sum_pt + 0.5350869
    if 0.07269073 <= Q.mass_over_sum_pt < 0.08475161:
        z += 72.6985 * Q.mass_over_sum_pt - 8.253602
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 338.0735 * Q.mass_over_sum_pt - 30.74456
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -32.52916 * Q.mass_over_sum_pt + 2.763044
    if Q.tau1 < 0.05356915:
        z += -42.93539 * Q.tau1 + 2.300013
    if Q.girth < 0.04081947:
        z += 132.7547 * Q.girth - 9.093796
    if 0.04081947 <= Q.girth < 0.08723651:
        z += 79.16962 * Q.girth - 6.906481
    if 36.22941 <= Q.mass < 76.6557:
        z += 0.05135435 * Q.mass - 1.860538
    if Q.mass >= 76.6557:
        z += -0.1610443 * Q.mass + 14.42103
    if 0.005590289 <= Q.width < 0.008678045:
        z += -1182.647 * Q.width + 6.61134
    if Q.width >= 0.008678045:
        z += -482.8449 * Q.width + 0.5384231
    if Q.girth2_top2 < 8.10414e-05:
        z += 371446.3 * Q.girth2_top2 - 30.10253
    if Q.sj2_dr < 0.1294903:
        z += -15.71053 * Q.sj2_dr + 1.195904
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -4.902435 * Q.sj2_dr - 0.20364
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 35.02856 * Q.sj2_dr - 6.559508
    if Q.LHA < 0.3033137:
        z += -11.07262 * Q.LHA + 3.358479
    if Q.centroid_offset >= 0.03776099:
        z += -48.65684 * Q.centroid_offset + 1.837331
    if Q.lam1 < 0.008375572:
        z += 179.2791 * Q.lam1 - 1.501565
    if Q.e3 < 8.147744e-05:
        z += 34560.39 * Q.e3 - 2.815892
    if Q.e2 >= 0.05028464:
        z += -207.0337 * Q.e2 + 10.41062
    if Q.planar_flow < 0.1950135 and Q.width > 0.007520088:
        z += -2721.445 * (0.1950135 - Q.planar_flow) * (Q.width - 0.007520088)
    if Q.girth2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 2021.835 * (Q.girth2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.5655652
    if Q.girth2 < 0.005019719:
        z += -284.2185 * Q.girth2 + 1.426697
    if Q.LHA < 0.1967397:
        z += 57.23899 * Q.LHA - 11.26118
    if Q.log_sum_pt >= 6.701242:
        z += -16.61397 * Q.log_sum_pt + 111.3343
    if Q.girth < 0.06108601:
        z += 66.44382 * Q.girth - 4.058788
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.009398292 * Q.sum_pt_top5 - 6.460738
    if Q.z_dr_0_0p05 >= 0.8477313:
        z += -4.433283 * Q.z_dr_0_0p05 + 3.758233
    if Q.width < 0.003562611:
        z += -233.7561 * Q.width + 0.8327823
    if Q.mass_over_sum_pt < 0.03319429:
        z += -118.4842 * Q.mass_over_sum_pt + 3.932998
    if Q.sj3_dr_max < 0.1426152:
        z += 27.84765 * Q.sj3_dr_max - 3.971497
    if Q.girth2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 31808.03 * (0.006679471 - Q.girth2) * (0.02355416 - Q.centroid_offset)
    if Q.girth2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 4965.175 * (0.005019719 - Q.girth2) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.girth < 0.06108601 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -336.3577 * (0.06108601 - Q.girth) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.girth < 0.06108601 and Q.lam2 < 0.0001947983:
        z += 236714.0 * (0.06108601 - Q.girth) * (0.0001947983 - Q.lam2)
    if Q.mass < 21.78408 and Q.pt_7 < 48.71875:
        z += -0.003727224 * (21.78408 - Q.mass) * (48.71875 - Q.pt_7)
    if Q.girth2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -25643.27 * (0.005019719 - Q.girth2) * (Q.centroid_offset - 0.006789738)
    if Q.mass < 29.6447 and Q.width < 0.003562611:
        z += -41.76267 * (29.6447 - Q.mass) * (0.003562611 - Q.width)
    if Q.sj3_dr_max < 0.1986272 and Q.girth2 < 0.006679471:
        z += 1633.608 * (0.1986272 - Q.sj3_dr_max) * (0.006679471 - Q.girth2)
    if Q.tau1 < 0.05356915 and Q.width < 0.003562611:
        z += 21672.79 * (0.05356915 - Q.tau1) * (0.003562611 - Q.width)
    return max(0.0, z)


def neuron_9(Q):
    z = -3.212293
    if Q.girth < 0.05464922:
        z += 57.93995 * Q.girth - 3.166373
    if Q.girth >= 0.0717028:
        z += -93.23307 * Q.girth + 6.685072
    if Q.tau1 < 0.04369778:
        z += -97.89552 * Q.tau1 + 4.277817
    if Q.mass < 53.33237:
        z += 0.09790827 * Q.mass - 5.221681
    if Q.e3 < 2.371297e-05:
        z += 83266.02 * Q.e3 - 1.974485
    if 8.147744e-05 <= Q.e3 < 0.0001869378:
        z += 10562.2 * Q.e3 - 0.860581
    if Q.e3 >= 0.0001869378:
        z += -95.5127 * Q.e3 + 1.131749
    if Q.girth2 < 0.003562611:
        z += -3911.047 * Q.girth2 + 17.8791
    if 0.003562611 <= Q.girth2 < 0.00609665:
        z += -1557.023 * Q.girth2 + 9.492626
    if Q.sj3_dr_max < 0.1426152:
        z += 39.53739 * Q.sj3_dr_max - 4.305711
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -18.83085 * Q.sj3_dr_max + 4.018485
    if Q.lam2 >= 0.001130645:
        z += 551.8829 * Q.lam2 - 0.6239835
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 1.028209 * Q.n_dr_0p2_0p4 - 1.028209
    if Q.lam1 < 0.003377388:
        z += 815.4236 * Q.lam1 - 4.855155
    if 0.003377388 <= Q.lam1 < 0.005433361:
        z += 585.8942 * Q.lam1 - 4.079944
    if 0.005433361 <= Q.lam1 < 0.00595415:
        z += 1151.418 * Q.lam1 - 7.15264
    if Q.lam1 >= 0.00595415:
        z += 335.9945 * Q.lam1 - 2.297486
    if Q.centroid_offset < 0.01837778:
        z += -265.1134 * Q.centroid_offset + 4.872195
    if 0.03556091 <= Q.e2 < 0.04447357:
        z += -82.83437 * Q.e2 + 2.945666
    if Q.e2 >= 0.04447357:
        z += 86.55103 * Q.e2 - 4.587508
    if Q.sj3_pair_mass_max < 24.23013:
        z += 0.08341188 * Q.sj3_pair_mass_max - 2.021081
    if Q.C3 < 0.0284695:
        z += 69.61066 * Q.C3 - 1.981781
    if Q.zdr_0 < 0.006292091:
        z += 269.305 * Q.zdr_0 - 1.694492
    if Q.mass_over_sum_pt < 0.07269073:
        z += 53.34438 * Q.mass_over_sum_pt - 3.877642
    if Q.sum_pt < 559.6875:
        z += -0.01986689 * Q.sum_pt + 14.99015
    if 559.6875 <= Q.sum_pt < 988.4078:
        z += -0.009028949 * Q.sum_pt + 8.924284
    z += -8.507298 * Q.z_dr_0p2_0p4
    if Q.log_sum_pt >= 6.896095:
        z += 20.67153 * Q.log_sum_pt - 142.5529
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.01073602 * Q.sum_pt_top5 + 9.017768
    if Q.max_dr < 0.1117619:
        z += -47.1219 * Q.max_dr + 5.266431
    if Q.mass < 53.33237 and Q.centroid_offset < 0.02685622:
        z += 6.756995 * (53.33237 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass < 53.33237 and Q.log_sum_pt < 6.842717:
        z += 0.06960289 * (53.33237 - Q.mass) * (6.842717 - Q.log_sum_pt)
    if Q.girth2 < 0.00609665 and Q.planar_flow < 0.3220738:
        z += 2131.607 * (0.00609665 - Q.girth2) * (0.3220738 - Q.planar_flow)
    if Q.lam1 > 0.005433361 and Q.eccentricity > 0.7117266:
        z += -802.3769 * (Q.lam1 - 0.005433361) * (Q.eccentricity - 0.7117266)
    if Q.tau1 < 0.04369778 and Q.eccentricity > 0.9031255:
        z += -866.67 * (0.04369778 - Q.tau1) * (Q.eccentricity - 0.9031255)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.716559:
        z += -55.42017 * (Q.log_sum_pt - 6.896095) * (0.716559 - Q.D2_b2)
    if Q.centroid_offset < 0.01837778 and Q.n_for_90pct > 5.0:
        z += -56.22345 * (0.01837778 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 < 159.25:
        z += -2.795783 * (0.01837778 - Q.centroid_offset) * (159.25 - Q.pt_1)
    if Q.centroid_offset < 0.01837778 and Q.z_2nd < 0.2055511:
        z += 1587.651 * (0.01837778 - Q.centroid_offset) * (0.2055511 - Q.z_2nd)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.525997
    z += -36.4558 * Q.e2
    if Q.sj3_pair_mass_min >= 11.051:
        z += 0.1035518 * Q.sj3_pair_mass_min - 1.144351
    if Q.lam1 < 0.001503553:
        z += 2573.025 * Q.lam1 - 6.152087
    if 0.001503553 <= Q.lam1 < 0.004183811:
        z += 644.9202 * Q.lam1 - 3.253078
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += 313.4167 * Q.lam1 - 1.86613
    if Q.lam1 >= 0.00733008:
        z += -149.677 * Q.lam1 + 1.097144
    if 0.0003061234 <= Q.lam2 < 0.003408389:
        z += 746.1591 * Q.lam2 - 0.2284167
    if Q.lam2 >= 0.003408389:
        z += 49.55206 * Q.lam2 + 2.145891
    if Q.pt_7 < 45.75:
        z += 0.04274978 * Q.pt_7 - 1.955802
    if Q.e3 < 3.892127e-05:
        z += -17558.84 * Q.e3 + 1.430649
    if 3.892127e-05 <= Q.e3 < 8.147744e-05:
        z += -22320.98 * Q.e3 + 1.615998
    if Q.e3 >= 8.147744e-05:
        z += -4762.144 * Q.e3 + 0.1853487
    if Q.LHA >= 0.3033137:
        z += -34.71547 * Q.LHA + 10.52968
    if Q.tau1 >= 0.05356915:
        z += 39.6598 * Q.tau1 - 2.124542
    if Q.girth2 >= 0.007520088:
        z += 487.1833 * Q.girth2 - 3.663662
    if Q.mass < 76.6557:
        z += -0.02752135 * Q.mass + 2.109668
    if Q.C2_b2 >= 0.009032972:
        z += 126.0312 * Q.C2_b2 - 1.138436
    if Q.pt_7 < 45.75 and Q.D2 < 1.002471:
        z += 0.1273066 * (45.75 - Q.pt_7) * (1.002471 - Q.D2)
    if Q.zdr_0 < 0.0211821 and Q.z_dr_0p05_0p1 < 0.2919447:
        z += 182.2886 * (0.0211821 - Q.zdr_0) * (0.2919447 - Q.z_dr_0p05_0p1)
    if Q.e3 < 8.147744e-05 and Q.sj3_dr23 > 0.1797097:
        z += -146055.7 * (8.147744e-05 - Q.e3) * (Q.sj3_dr23 - 0.1797097)
    if Q.lam1 > 0.00733008 and Q.D2_b2 < 0.380911:
        z += -524.1245 * (Q.lam1 - 0.00733008) * (0.380911 - Q.D2_b2)
    return max(0.0, z)


def neuron_11(Q):
    z = -3.966889
    if Q.girth2 < 0.01323868:
        z += -284.4231 * Q.girth2 + 3.765385
    if Q.tau1 < 0.09538712:
        z += 12.77784 * Q.tau1 - 1.218841
    if Q.mass < 15.45403:
        z += -0.1677523 * Q.mass + 2.313034
    if 15.45403 <= Q.mass < 69.61135:
        z += 0.005159328 * Q.mass - 0.3591478
    if Q.centroid_offset < 0.03776099:
        z += -129.0277 * Q.centroid_offset + 4.872213
    if Q.sj3_dr_max < 0.1426152:
        z += 32.01374 * Q.sj3_dr_max - 4.183014
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 47.06729 * Q.sj3_dr_max - 6.329879
    if 0.169029 <= Q.sj3_dr_max < 0.2623172:
        z += -17.42834 * Q.sj3_dr_max + 4.571755
    if Q.width < 0.008678045:
        z += -1501.963 * Q.width + 13.0341
    if Q.girth < 0.02054282:
        z += 328.8148 * Q.girth - 10.10969
    if 0.02054282 <= Q.girth < 0.0717028:
        z += 65.5769 * Q.girth - 4.702047
    if Q.e2_sq < 0.008168571:
        z += 415.8805 * Q.e2_sq - 3.397149
    if Q.z_7 >= 0.01685855:
        z += 34.47866 * Q.z_7 - 0.5812601
    if Q.sj2_dr >= 0.2687922:
        z += 25.381 * Q.sj2_dr - 6.822214
    if Q.lam1 < 0.008375572:
        z += 530.8262 * Q.lam1 - 4.445973
    if Q.max_dr < 0.1452311:
        z += -13.78469 * Q.max_dr + 2.001966
    if Q.centroid_offset < 0.03776099 and Q.sum_pt < 901.5938:
        z += -0.2518099 * (0.03776099 - Q.centroid_offset) * (901.5938 - Q.sum_pt)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += -235.6676 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.centroid_offset > 0.04990367 and Q.tau32 < 0.5502779:
        z += 1141.619 * (Q.centroid_offset - 0.04990367) * (0.5502779 - Q.tau32)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.43861
    if Q.girth2 >= 0.01882765:
        z += 346.9805 * Q.girth2 - 6.532828
    if Q.mass >= 91.19:
        z += 0.09597781 * Q.mass - 8.752216
    if Q.e2 >= 0.06344108:
        z += -79.84167 * Q.e2 + 5.065242
    return max(0.0, z)


def neuron_13(Q):
    z = 0.1209103
    if Q.girth < 0.1484084:
        z += -51.75967 * Q.girth + 7.681571
    if Q.lam1 < 0.006506576:
        z += -453.6043 * Q.lam1 + 5.226045
    if 0.006506576 <= Q.lam1 < 0.01643375:
        z += -229.1322 * Q.lam1 + 3.765501
    if Q.e2 < 0.08000524:
        z += 39.23735 * Q.e2 - 3.139193
    if Q.z_7 < 0.02807091:
        z += 271.9938 * Q.z_7 - 7.635116
    if Q.width < 0.007520088:
        z += 144.2247 * Q.width - 1.084582
    if Q.girth < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -51.65779 * (0.1484084 - Q.girth) * (6.804164 - Q.log_sum_pt)
    if Q.girth < 0.1484084 and Q.pt_7 < 38.53125:
        z += -1.390619 * (0.1484084 - Q.girth) * (38.53125 - Q.pt_7)
    if Q.e3 < 5.334511e-05 and Q.centroid_offset < 0.03776099:
        z += -959087.9 * (5.334511e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 40.04062:
        z += 0.001175084 * (Q.sum_pt_top5 - 658.125) * (40.04062 - Q.pt_7)
    if Q.pt_6 < 31.90625 and Q.z_7 < 0.0586137:
        z += -1.59117 * (31.90625 - Q.pt_6) * (0.0586137 - Q.z_7)
    if Q.e3 < 5.334511e-05 and Q.D3 > 0.2213841:
        z += 10907.64 * (5.334511e-05 - Q.e3) * (Q.D3 - 0.2213841)
    if Q.sum_pt > 988.4078 and Q.pt_6 < 62.25:
        z += -0.0006380717 * (Q.sum_pt - 988.4078) * (62.25 - Q.pt_6)
    return max(0.0, z)


def neuron_14(Q):
    z = -1.410004
    if Q.psi_0p1 >= 0.9761279:
        z += -27.56896 * Q.psi_0p1 + 26.91084
    if 0.007520088 <= Q.girth2 < 0.008678045:
        z += -1211.825 * Q.girth2 + 9.113034
    if Q.girth2 >= 0.008678045:
        z += -1172.253 * Q.girth2 + 8.769619
    if 0.002074109 <= Q.e2_sq < 0.0030133:
        z += 260.1843 * Q.e2_sq - 0.5396507
    if 0.0030133 <= Q.e2_sq < 0.01165737:
        z += -32.03888 * Q.e2_sq + 0.3409056
    if Q.e2_sq >= 0.01165737:
        z += 259.1897 * Q.e2_sq - 3.054055
    if 0.003562611 <= Q.width < 0.006679471:
        z += 406.8015 * Q.width - 1.449276
    if 0.006679471 <= Q.width < 0.01323868:
        z += -1396.655 * Q.width + 10.59686
    if Q.width >= 0.01323868:
        z += -791.9214 * Q.width + 2.590991
    if 0.01655442 <= Q.e2 < 0.03556091:
        z += -192.4897 * Q.e2 + 3.186555
    if 0.03556091 <= Q.e2 < 0.04110972:
        z += -247.3458 * Q.e2 + 5.137288
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += -83.35678 * Q.e2 - 1.604253
    if Q.e2 >= 0.05028464:
        z += 341.6371 * Q.e2 - 22.97492
    if 0.02685622 <= Q.centroid_offset < 0.04990367:
        z += -109.0461 * Q.centroid_offset + 2.928566
    if Q.centroid_offset >= 0.04990367:
        z += -1219.946 * Q.centroid_offset + 58.36656
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 185.8474 * Q.mass_over_sum_pt - 14.85362
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 401.9515 * Q.mass_over_sum_pt - 33.16879
    if Q.mass_over_sum_pt >= 0.09041383:
        z += 259.1203 * Q.mass_over_sum_pt - 20.25487
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += -11.13433 * Q.z_dr_0p05_0p1 + 8.360875
    if Q.e3 < 5.334511e-05:
        z += 20895.89 * Q.e3 - 1.114694
    if Q.sd_mass < 49.91626:
        z += -0.00780303 * Q.sd_mass - 0.940606
    if 49.91626 <= Q.sd_mass < 74.57663:
        z += 0.05393691 * Q.sd_mass - 4.022433
    if 0.02689598 <= Q.girth < 0.04081947:
        z += 90.44431 * Q.girth - 2.432588
    if 0.04081947 <= Q.girth < 0.08065885:
        z += 162.7062 * Q.girth - 5.38228
    if 0.08065885 <= Q.girth < 0.08723651:
        z += 242.4558 * Q.girth - 11.8148
    if 0.08723651 <= Q.girth < 0.1019409:
        z += -75.21127 * Q.girth + 15.89737
    if Q.girth >= 0.1019409:
        z += -206.7702 * Q.girth + 29.30861
    if 0.06154135 <= Q.sj2_dr < 0.1294903:
        z += -4.554884 * Q.sj2_dr + 0.2803137
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -11.52158 * Q.sj2_dr + 1.182433
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 42.80022 * Q.sj2_dr - 7.464038
    if Q.sj2_dr >= 0.2001708:
        z += 20.9357 * Q.sj2_dr - 3.087401
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -10286.64 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += 1410.604 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.width < 0.00609665:
        z += 9784.525 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.width)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -838.7508 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 1.0:
        z += 13.24309 * (Q.z_dr_0p05_0p1 - 0.7509095) * (1.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.3878572
    if Q.girth2 < 0.007520088:
        z += 859.4324 * Q.girth2 - 6.463008
    if Q.mass_over_sum_pt < 0.1309286:
        z += 30.25677 * Q.mass_over_sum_pt - 3.961477
    if Q.width < 0.00609665:
        z += -592.011 * Q.width + 11.81536
    if 0.00609665 <= Q.width < 0.01323868:
        z += -959.687 * Q.width + 14.05695
    if 0.01323868 <= Q.width < 0.01882765:
        z += -241.8992 * Q.width + 4.554395
    if Q.e2 < 0.04110972:
        z += -230.7089 * Q.e2 + 9.484378
    if Q.e2_sq < 0.008168571:
        z += 1074.341 * Q.e2_sq - 9.890667
    if 0.008168571 <= Q.e2_sq < 0.01165737:
        z += 319.5466 * Q.e2_sq - 3.725074
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -942.3414 * Q.mass_over_sum_pt_sq + 6.768684
    if Q.lam1 < 0.00483998:
        z += -158.8951 * Q.lam1 + 0.7690491
    if Q.girth < 0.1019409:
        z += 111.7434 * Q.girth - 11.39122
    if Q.sj2_dr < 0.1492731:
        z += -6.573044 * Q.sj2_dr - 0.3185192
    if 0.1492731 <= Q.sj2_dr < 0.1591713:
        z += -20.09675 * Q.sj2_dr + 1.700207
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 36.55215 * Q.sj2_dr - 7.316672
    if Q.girth2_top2 < 0.0005124533:
        z += 4810.125 * Q.girth2_top2 - 2.464964
    if Q.girth2 < 0.007520088 and Q.D2 < 0.7459513:
        z += -1700.759 * (0.007520088 - Q.girth2) * (0.7459513 - Q.D2)
    if Q.width < 0.00609665 and Q.D2 < 0.7459513:
        z += -4818.035 * (0.00609665 - Q.width) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.D2 < 0.7459513:
        z += 100.9499 * (0.1309286 - Q.mass_over_sum_pt) * (0.7459513 - Q.D2)
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
