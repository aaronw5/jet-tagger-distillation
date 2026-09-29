"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 87.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr01                   ΔR between particles 0 and 1
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
        max_dr=max(dr[i] for i in real),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        pt_6=pt[6],
        pt_7=pt[7],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sj2_dr=subjets(2)["dr"][0],
        dr01=math.sqrt(dist2(0, 1)),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-0.4375
        - 0.15625 * grid(0, max(0.0, -2.417431
            + 13.83471 * max(0.0, 0.1484197 - Q.planar_flow)
            - 2262.493 * max(0.0, 0.004372139 - Q.width)
            + 0.1055477 * max(0.0, 21.78408 - Q.mass)
            + 235.418 * max(0.0, 0.01323868 - Q.girth2)
            + 365.8228 * max(0.0, 0.006506576 - Q.lam1)
            - 2988.999 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2)
            - 0.02260682 * max(0.0, Q.sum_pt - 901.5938)
            + 461.6906 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 1.002471 - Q.D2)
            - 0.0689348 * max(0.0, 56.92035 - Q.mass)
            + 602.5658 * max(0.0, 0.008678045 - Q.width)
            - 1364.818 * max(0.0, 0.001563465 - Q.C2_b2)
            - 29.28583 * max(0.0, Q.sj3_dr_max - 0.233678)
            + 7.116758 * max(0.0, Q.sj3_dr_max - 0.1070199)
            + 10.34001 * max(0.0, 0.3012016 - Q.sj3_dr_max)
            - 12219.26 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.centroid_offset - 0.01837778)
            - 43.05435 * max(0.0, 0.08723651 - Q.girth)
            + 227.4702 * max(0.0, 0.0245477 - Q.e2)
            - 305.1563 * max(0.0, 0.005433361 - Q.lam1)
            + 0.40509 * max(0.0, Q.n_dr_0_0p05 - 4.0)
            - 13.21815 * max(0.0, Q.log_sum_pt - 6.670067)
            + 0.01865327 * max(0.0, Q.sum_pt_top5 - 687.4375)
            + 73765.7 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2)
        ))
        + 0.390625 * grid(1, max(0.0, -0.09412154
            + 0.1487708 * max(0.0, Q.pt_7 - 34.53125)
            + 5.815856 * max(0.0, Q.log_sum_pt - 6.377723)
            + 633.314 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq)
            - 77.3932 * max(0.0, 0.06164517 - Q.z_7)
            - 540.0432 * max(0.0, 0.008678045 - Q.girth2)
            - 9.214913 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2)
            - 3.109254 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1)
            + 219.1891 * max(0.0, 0.008168571 - Q.e2_sq)
            - 0.004414408 * max(0.0, Q.sum_pt_top5 - 531.1875)
            - 417.6075 * max(0.0, 0.00609665 - Q.width)
            + 5.885901 * max(0.0, Q.log_sum_pt - 6.502799)
            + 0.004937477 * max(0.0, Q.sum_pt_top5 - 367.5938)
        ))
        + 0.4296875 * grid(2, max(0.0, 0.7863654
            + 15.2788 * max(0.0, Q.log_sum_pt - 6.842717)
            + 3.857629 * max(0.0, 6.46415 - Q.log_sum_pt)
            + 525.5309 * max(0.0, 0.00595415 - Q.lam1)
            - 132.0194 * max(0.0, 0.03243272 - Q.z_7)
            - 558.6993 * max(0.0, 0.007673833 - Q.girth)
            + 0.003140693 * max(0.0, 788.4484 - Q.sum_pt)
            - 13594.71 * max(0.0, 0.0003193707 - Q.girth2)
            + 468.1396 * max(0.0, 0.0003193707 - Q.girth2) * max(0.0, 36.76827 - Q.mass_top2)
            - 24.41469 * max(0.0, Q.log_sum_pt - 6.896095)
        ))
        - 0.03125 * grid(4, max(0.0, -3.164716
            + 64.31693 * max(0.0, 0.2233283 - Q.N2)
            - 2741.871 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.e2_sq - 0.01165737)
            - 9547.75 * max(0.0, 0.000537286 - Q.lam2)
            - 148.3439 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            + 442.9708 * max(0.0, Q.girth2 - 0.003562611)
            - 0.2348624 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.25 - Q.pt_6)
            - 86.98067 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266)
            - 310.1108 * max(0.0, 0.008678045 - Q.girth2)
            - 21.48242 * max(0.0, Q.sj3_dr_max - 0.233678)
            + 761.9908 * max(0.0, 0.01165737 - Q.e2_sq)
            + 0.05472227 * max(0.0, 76.6557 - Q.mass)
            + 286.7206 * max(0.0, Q.tau1 - 0.07283629) * max(0.0, 0.2089872 - Q.sj3_dr_min)
            - 0.3276204 * max(0.0, 76.6557 - Q.mass) * max(0.0, 0.875672 - Q.D2)
            + 634.2974 * max(0.0, 0.01165737 - Q.e2_sq) * max(0.0, 0.875672 - Q.D2)
            + 23.96368 * max(0.0, Q.max_dr - 0.1117619)
            + 1363.239 * max(0.0, 0.002635418 - Q.girth2)
            - 585.9924 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, 0.02117036 - Q.dr01)
            - 204.6598 * max(0.0, 0.01323868 - Q.width)
            + 24.52623 * max(0.0, Q.e2 - 0.009668065)
            - 83.57821 * max(0.0, Q.girth - 0.05464922)
            + 635.306 * max(0.0, 0.009032972 - Q.C2_b2)
            + 197.4851 * max(0.0, 0.01716248 - Q.e2_sq)
            + 40.95097 * max(0.0, Q.tau1 - 0.1136369)
            - 380.6967 * max(0.0, 0.01882765 - Q.girth2)
            + 44.53423 * max(0.0, 1.002471 - Q.D2) * max(0.0, Q.M3 - 0.04581318)
            - 0.07156271 * max(0.0, 69.61135 - Q.mass)
        ))
        - 0.1875 * grid(5, max(0.0, 1.685334
            + 81.98639 * max(0.0, 0.04939969 - Q.z_7)
            - 153.4978 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 51.68755 * max(0.0, Q.log_sum_pt - 6.896095)
            + 760.8062 * max(0.0, 0.005284669 - Q.e2_sq)
            - 1600.013 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset)
            + 0.8812857 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 42.18339 - Q.mass_top3)
            - 0.09400035 * max(0.0, 37.15625 - Q.pt_7)
            + 160212.7 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            + 169.6404 * max(0.0, 0.02807091 - Q.z_7)
            - 128.5201 * max(0.0, 0.02383244 - Q.zdr_0)
            - 6.093912 * max(0.0, 0.3012016 - Q.sj3_dr_max)
            - 17.92183 * max(0.0, Q.log_sum_pt - 6.701242)
            + 3.944505 * max(0.0, Q.log_sum_pt - 6.267538)
            + 0.03822879 * max(0.0, 60.63098 - Q.mass)
            - 562.8433 * max(0.0, 0.002915531 - Q.girth2_top3)
            - 0.04790254 * max(0.0, Q.pt_7 - 23.21641)
            - 29.95585 * max(0.0, 0.09041383 - Q.mass_over_sum_pt)
            + 11.60847 * max(0.0, 0.09538712 - Q.tau1)
            + 53078.77 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0002302115 - Q.mean_phi2)
            + 195913.6 * max(0.0, 0.002915531 - Q.girth2_top3) * max(0.0, 0.006646257 - Q.tau3)
        ))
        + 0.109375 * grid(6, max(0.0, 0.7932636
            + 151.4742 * max(0.0, Q.centroid_offset - 0.00809236)
            - 404.8059 * max(0.0, 0.01323868 - Q.width)
            + 3.010573 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308)
            + 1.173602 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt)
            + 62.78069 * max(0.0, 0.1136369 - Q.tau1)
            - 26.39771 * max(0.0, Q.sj3_dr_min - 0.1278212)
            - 189.338 * max(0.0, Q.centroid_offset - 0.01837778)
            + 285.4071 * max(0.0, 0.01200373 - Q.lam1)
            + 8349.59 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2)
            - 0.105165 * max(0.0, Q.sj3_pair_mass_min - 4.501727)
            - 13.53752 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757)
            + 1180.545 * max(0.0, 0.0030133 - Q.e2_sq)
            - 2047.89 * max(0.0, 0.001130645 - Q.lam2)
            - 0.814989 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4)
            + 43.57116 * max(0.0, 0.1789613 - Q.sj3_dr_max)
            - 43.15722 * max(0.0, 0.1452311 - Q.max_dr)
            + 108.5842 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05)
            + 6230.968 * max(0.0, 0.0005116989 - Q.e3)
            + 0.03968707 * max(0.0, 45.7571 - Q.mass)
            - 1738.251 * max(0.0, 0.008678045 - Q.girth2)
            + 233.3178 * max(0.0, 0.00733008 - Q.lam1)
            + 13.69197 * max(0.0, Q.sj3_dr_max - 0.1879486)
            - 0.04309208 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        ))
        + 0.171875 * grid(9, max(0.0, -2.832185
            - 64.86963 * max(0.0, 0.05464922 - Q.girth)
            + 89.9129 * max(0.0, 0.04369778 - Q.tau1)
            - 0.09226678 * max(0.0, 53.33237 - Q.mass)
            + 6.692211 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset)
            + 1768.36 * max(0.0, 0.00609665 - Q.girth2)
            + 22.75828 * max(0.0, 0.213399 - Q.sj3_dr_max)
            + 579.7119 * max(0.0, Q.lam2 - 0.001130645)
            + 0.1183909 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt)
            + 0.9639006 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
            - 60.9241 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            - 255.1962 * max(0.0, Q.lam1 - 0.003377388)
            + 200.828 * max(0.0, 0.01837778 - Q.centroid_offset)
            - 53.04834 * max(0.0, Q.e2 - 0.03556091)
            - 0.1499527 * max(0.0, 24.23013 - Q.sj3_pair_mass_max)
            + 529.3897 * max(0.0, Q.lam1 - 0.005433361)
            - 72.7653 * max(0.0, 0.0284695 - Q.C3)
            - 748.4049 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266)
            - 84531.47 * max(0.0, 2.371297e-05 - Q.e3)
            + 2000.407 * max(0.0, 0.003562611 - Q.girth2)
            - 66.49413 * max(0.0, 0.07269073 - Q.mass_over_sum_pt)
            + 0.008796196 * max(0.0, 988.4078 - Q.sum_pt)
            + 170.6707 * max(0.0, Q.e2 - 0.04447357)
            - 687.1715 * max(0.0, 0.00595415 - Q.lam1)
            - 8.805234 * Q.z_dr_0p2_0p4
            - 62.37537 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0)
            - 3.009064 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1)
            + 1848.072 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd)
            - 77.76084 * max(0.0, Q.girth - 0.0717028)
            + 51.85093 * max(0.0, 0.1117619 - Q.max_dr)
        ))
    )


def logit_q(Q):
    return (0.03125
        - 0.09375 * grid(4, max(0.0, -3.164716
            + 64.31693 * max(0.0, 0.2233283 - Q.N2)
            - 2741.871 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.e2_sq - 0.01165737)
            - 9547.75 * max(0.0, 0.000537286 - Q.lam2)
            - 148.3439 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            + 442.9708 * max(0.0, Q.girth2 - 0.003562611)
            - 0.2348624 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.25 - Q.pt_6)
            - 86.98067 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266)
            - 310.1108 * max(0.0, 0.008678045 - Q.girth2)
            - 21.48242 * max(0.0, Q.sj3_dr_max - 0.233678)
            + 761.9908 * max(0.0, 0.01165737 - Q.e2_sq)
            + 0.05472227 * max(0.0, 76.6557 - Q.mass)
            + 286.7206 * max(0.0, Q.tau1 - 0.07283629) * max(0.0, 0.2089872 - Q.sj3_dr_min)
            - 0.3276204 * max(0.0, 76.6557 - Q.mass) * max(0.0, 0.875672 - Q.D2)
            + 634.2974 * max(0.0, 0.01165737 - Q.e2_sq) * max(0.0, 0.875672 - Q.D2)
            + 23.96368 * max(0.0, Q.max_dr - 0.1117619)
            + 1363.239 * max(0.0, 0.002635418 - Q.girth2)
            - 585.9924 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, 0.02117036 - Q.dr01)
            - 204.6598 * max(0.0, 0.01323868 - Q.width)
            + 24.52623 * max(0.0, Q.e2 - 0.009668065)
            - 83.57821 * max(0.0, Q.girth - 0.05464922)
            + 635.306 * max(0.0, 0.009032972 - Q.C2_b2)
            + 197.4851 * max(0.0, 0.01716248 - Q.e2_sq)
            + 40.95097 * max(0.0, Q.tau1 - 0.1136369)
            - 380.6967 * max(0.0, 0.01882765 - Q.girth2)
            + 44.53423 * max(0.0, 1.002471 - Q.D2) * max(0.0, Q.M3 - 0.04581318)
            - 0.07156271 * max(0.0, 69.61135 - Q.mass)
        ))
        + 0.046875 * grid(5, max(0.0, 1.685334
            + 81.98639 * max(0.0, 0.04939969 - Q.z_7)
            - 153.4978 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 51.68755 * max(0.0, Q.log_sum_pt - 6.896095)
            + 760.8062 * max(0.0, 0.005284669 - Q.e2_sq)
            - 1600.013 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset)
            + 0.8812857 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 42.18339 - Q.mass_top3)
            - 0.09400035 * max(0.0, 37.15625 - Q.pt_7)
            + 160212.7 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            + 169.6404 * max(0.0, 0.02807091 - Q.z_7)
            - 128.5201 * max(0.0, 0.02383244 - Q.zdr_0)
            - 6.093912 * max(0.0, 0.3012016 - Q.sj3_dr_max)
            - 17.92183 * max(0.0, Q.log_sum_pt - 6.701242)
            + 3.944505 * max(0.0, Q.log_sum_pt - 6.267538)
            + 0.03822879 * max(0.0, 60.63098 - Q.mass)
            - 562.8433 * max(0.0, 0.002915531 - Q.girth2_top3)
            - 0.04790254 * max(0.0, Q.pt_7 - 23.21641)
            - 29.95585 * max(0.0, 0.09041383 - Q.mass_over_sum_pt)
            + 11.60847 * max(0.0, 0.09538712 - Q.tau1)
            + 53078.77 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0002302115 - Q.mean_phi2)
            + 195913.6 * max(0.0, 0.002915531 - Q.girth2_top3) * max(0.0, 0.006646257 - Q.tau3)
        ))
        + 0.125 * grid(6, max(0.0, 0.7932636
            + 151.4742 * max(0.0, Q.centroid_offset - 0.00809236)
            - 404.8059 * max(0.0, 0.01323868 - Q.width)
            + 3.010573 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308)
            + 1.173602 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt)
            + 62.78069 * max(0.0, 0.1136369 - Q.tau1)
            - 26.39771 * max(0.0, Q.sj3_dr_min - 0.1278212)
            - 189.338 * max(0.0, Q.centroid_offset - 0.01837778)
            + 285.4071 * max(0.0, 0.01200373 - Q.lam1)
            + 8349.59 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2)
            - 0.105165 * max(0.0, Q.sj3_pair_mass_min - 4.501727)
            - 13.53752 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757)
            + 1180.545 * max(0.0, 0.0030133 - Q.e2_sq)
            - 2047.89 * max(0.0, 0.001130645 - Q.lam2)
            - 0.814989 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4)
            + 43.57116 * max(0.0, 0.1789613 - Q.sj3_dr_max)
            - 43.15722 * max(0.0, 0.1452311 - Q.max_dr)
            + 108.5842 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05)
            + 6230.968 * max(0.0, 0.0005116989 - Q.e3)
            + 0.03968707 * max(0.0, 45.7571 - Q.mass)
            - 1738.251 * max(0.0, 0.008678045 - Q.girth2)
            + 233.3178 * max(0.0, 0.00733008 - Q.lam1)
            + 13.69197 * max(0.0, Q.sj3_dr_max - 0.1879486)
            - 0.04309208 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        ))
        + 0.0625 * grid(8, max(0.0, -0.7267793
            + 230.018 * max(0.0, 0.005019719 - Q.girth2)
            + 33692.16 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            - 64.20506 * max(0.0, 0.1967397 - Q.LHA)
            - 12.38385 * max(0.0, Q.log_sum_pt - 6.701242)
            - 75.46003 * max(0.0, 0.06108601 - Q.girth)
            + 0.01007342 * max(0.0, Q.sum_pt_top5 - 687.4375)
            + 4974.301 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4)
            + 276112.5 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2)
            - 519602.5 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 2.955458e-05 - Q.e3)
            - 24783.14 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738)
            + 89.18823 * max(0.0, 0.03319429 - Q.mass_over_sum_pt)
            - 31.67499 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            + 1019.179 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.008678045 - Q.width)
            - 44.36972 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.003562611 - Q.width)
            + 2399.552 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.girth2)
            + 22400.78 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.width)
        ))
        + 0.2539062 * grid(9, max(0.0, -2.832185
            - 64.86963 * max(0.0, 0.05464922 - Q.girth)
            + 89.9129 * max(0.0, 0.04369778 - Q.tau1)
            - 0.09226678 * max(0.0, 53.33237 - Q.mass)
            + 6.692211 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset)
            + 1768.36 * max(0.0, 0.00609665 - Q.girth2)
            + 22.75828 * max(0.0, 0.213399 - Q.sj3_dr_max)
            + 579.7119 * max(0.0, Q.lam2 - 0.001130645)
            + 0.1183909 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt)
            + 0.9639006 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
            - 60.9241 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            - 255.1962 * max(0.0, Q.lam1 - 0.003377388)
            + 200.828 * max(0.0, 0.01837778 - Q.centroid_offset)
            - 53.04834 * max(0.0, Q.e2 - 0.03556091)
            - 0.1499527 * max(0.0, 24.23013 - Q.sj3_pair_mass_max)
            + 529.3897 * max(0.0, Q.lam1 - 0.005433361)
            - 72.7653 * max(0.0, 0.0284695 - Q.C3)
            - 748.4049 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266)
            - 84531.47 * max(0.0, 2.371297e-05 - Q.e3)
            + 2000.407 * max(0.0, 0.003562611 - Q.girth2)
            - 66.49413 * max(0.0, 0.07269073 - Q.mass_over_sum_pt)
            + 0.008796196 * max(0.0, 988.4078 - Q.sum_pt)
            + 170.6707 * max(0.0, Q.e2 - 0.04447357)
            - 687.1715 * max(0.0, 0.00595415 - Q.lam1)
            - 8.805234 * Q.z_dr_0p2_0p4
            - 62.37537 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0)
            - 3.009064 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1)
            + 1848.072 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd)
            - 77.76084 * max(0.0, Q.girth - 0.0717028)
            + 51.85093 * max(0.0, 0.1117619 - Q.max_dr)
        ))
        - 0.125 * grid(10, max(0.0, 1.88596
            - 48.00774 * Q.e2
            + 0.1771744 * max(0.0, Q.sj3_pair_mass_min - 11.051)
            - 600.9459 * max(0.0, 0.004183811 - Q.lam1)
            + 530.9472 * max(0.0, Q.lam2 - 0.0003061234)
            - 626.2477 * max(0.0, Q.lam2 - 0.003408389)
            - 1905.25 * max(0.0, 0.001503553 - Q.lam1)
            - 29.69881 * max(0.0, Q.LHA - 0.3033137)
            + 50.7083 * max(0.0, Q.tau1 - 0.05356915)
            + 522.6854 * max(0.0, Q.girth2 - 0.007520088)
            - 301.0504 * max(0.0, Q.lam1 - 0.00733008)
            - 395.1002 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2)
            - 4828.053 * max(0.0, Q.e3 - 3.892127e-05)
            + 0.02897324 * max(0.0, 76.6557 - Q.mass)
            + 109.8528 * max(0.0, Q.C2_b2 - 0.009032972)
        ))
        + 0.0625 * grid(15, max(0.0, -1.025458
            - 20.70682 * max(0.0, 0.2233283 - Q.N2)
            - 1053.515 * max(0.0, 0.007520088 - Q.girth2)
            - 12.41601 * max(0.0, 0.1309286 - Q.mass_over_sum_pt)
            - 1100.358 * max(0.0, 0.00609665 - Q.width)
            + 666.1981 * max(0.0, 0.01323868 - Q.width)
            - 2847.625 * max(0.0, 0.007520088 - Q.girth2) * max(0.0, 0.7459513 - Q.D2)
            + 146.7409 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2)
            + 174.9853 * max(0.0, 0.04110972 - Q.e2)
            - 289.3842 * max(0.0, 0.01165737 - Q.e2_sq)
            + 746.0186 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq)
            - 761.0077 * max(0.0, 0.008168571 - Q.e2_sq)
            + 201.8491 * max(0.0, 0.01882765 - Q.width)
            - 28.63268 * max(0.0, 0.1019409 - Q.girth)
            + 61.51609 * max(0.0, 0.1591713 - Q.sj2_dr)
            - 15.22296 * max(0.0, 0.2001708 - Q.sj2_dr)
            + 0.07160176 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75)
            - 38.54002 * max(0.0, 0.1492731 - Q.sj2_dr)
            + 1686.831 * max(0.0, 0.0005124533 - Q.girth2_top2)
            + 184.3761 * max(0.0, 0.006756161 - Q.girth2_top3)
            + 449.8922 * max(0.0, 0.005433361 - Q.lam1)
            - 154.3123 * max(0.0, 0.008375572 - Q.lam1)
            - 202.3983 * max(0.0, 0.01837778 - Q.centroid_offset)
            - 449.5484 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2512159 - Q.dr01)
        ))
    )


def logit_W(Q):
    return (-0.125
        + 0.34375 * grid(0, max(0.0, -2.417431
            + 13.83471 * max(0.0, 0.1484197 - Q.planar_flow)
            - 2262.493 * max(0.0, 0.004372139 - Q.width)
            + 0.1055477 * max(0.0, 21.78408 - Q.mass)
            + 235.418 * max(0.0, 0.01323868 - Q.girth2)
            + 365.8228 * max(0.0, 0.006506576 - Q.lam1)
            - 2988.999 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2)
            - 0.02260682 * max(0.0, Q.sum_pt - 901.5938)
            + 461.6906 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 1.002471 - Q.D2)
            - 0.0689348 * max(0.0, 56.92035 - Q.mass)
            + 602.5658 * max(0.0, 0.008678045 - Q.width)
            - 1364.818 * max(0.0, 0.001563465 - Q.C2_b2)
            - 29.28583 * max(0.0, Q.sj3_dr_max - 0.233678)
            + 7.116758 * max(0.0, Q.sj3_dr_max - 0.1070199)
            + 10.34001 * max(0.0, 0.3012016 - Q.sj3_dr_max)
            - 12219.26 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.centroid_offset - 0.01837778)
            - 43.05435 * max(0.0, 0.08723651 - Q.girth)
            + 227.4702 * max(0.0, 0.0245477 - Q.e2)
            - 305.1563 * max(0.0, 0.005433361 - Q.lam1)
            + 0.40509 * max(0.0, Q.n_dr_0_0p05 - 4.0)
            - 13.21815 * max(0.0, Q.log_sum_pt - 6.670067)
            + 0.01865327 * max(0.0, Q.sum_pt_top5 - 687.4375)
            + 73765.7 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2)
        ))
        - 0.03125 * grid(1, max(0.0, -0.09412154
            + 0.1487708 * max(0.0, Q.pt_7 - 34.53125)
            + 5.815856 * max(0.0, Q.log_sum_pt - 6.377723)
            + 633.314 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq)
            - 77.3932 * max(0.0, 0.06164517 - Q.z_7)
            - 540.0432 * max(0.0, 0.008678045 - Q.girth2)
            - 9.214913 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2)
            - 3.109254 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1)
            + 219.1891 * max(0.0, 0.008168571 - Q.e2_sq)
            - 0.004414408 * max(0.0, Q.sum_pt_top5 - 531.1875)
            - 417.6075 * max(0.0, 0.00609665 - Q.width)
            + 5.885901 * max(0.0, Q.log_sum_pt - 6.502799)
            + 0.004937477 * max(0.0, Q.sum_pt_top5 - 367.5938)
        ))
        - 0.5 * grid(3, max(0.0, -4.866957
            + 33.64804 * max(0.0, Q.mass_over_sum_pt - 0.0681391)
            - 50.95515 * max(0.0, Q.tau1 - 0.05356915)
            + 155.2936 * max(0.0, Q.lam1 - 0.01643375)
            + 75.26761 * max(0.0, Q.girth - 0.07608178)
            - 226.6729 * max(0.0, Q.width - 0.01323868)
            + 559.5157 * max(0.0, Q.width - 0.007520088)
            + 278.6867 * max(0.0, Q.e2 - 0.06344108)
            + 101.1847 * max(0.0, Q.girth - 0.04081947)
            + 60.75642 * max(0.0, Q.tau1 - 0.1027642)
            + 130.0719 * max(0.0, Q.girth - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494)
            - 1560.378 * max(0.0, Q.girth2 - 0.008678045)
            + 32.90918 * max(0.0, Q.sj2_dr - 0.1872617)
            - 3751.477 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr)
            + 13.73175 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50)
            + 315.049 * max(0.0, Q.lam1 - 0.008375572)
            - 978.3712 * max(0.0, Q.lam2 - 0.001130645)
            + 34.65721 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6)
            - 356.8522 * max(0.0, Q.lam1 - 0.01200373)
            - 0.09491761 * max(0.0, Q.mass - 64.61873)
            - 545.2256 * max(0.0, 0.004372139 - Q.girth2)
        ))
        - 0.3125 * grid(6, max(0.0, 0.7932636
            + 151.4742 * max(0.0, Q.centroid_offset - 0.00809236)
            - 404.8059 * max(0.0, 0.01323868 - Q.width)
            + 3.010573 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308)
            + 1.173602 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt)
            + 62.78069 * max(0.0, 0.1136369 - Q.tau1)
            - 26.39771 * max(0.0, Q.sj3_dr_min - 0.1278212)
            - 189.338 * max(0.0, Q.centroid_offset - 0.01837778)
            + 285.4071 * max(0.0, 0.01200373 - Q.lam1)
            + 8349.59 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2)
            - 0.105165 * max(0.0, Q.sj3_pair_mass_min - 4.501727)
            - 13.53752 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757)
            + 1180.545 * max(0.0, 0.0030133 - Q.e2_sq)
            - 2047.89 * max(0.0, 0.001130645 - Q.lam2)
            - 0.814989 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4)
            + 43.57116 * max(0.0, 0.1789613 - Q.sj3_dr_max)
            - 43.15722 * max(0.0, 0.1452311 - Q.max_dr)
            + 108.5842 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05)
            + 6230.968 * max(0.0, 0.0005116989 - Q.e3)
            + 0.03968707 * max(0.0, 45.7571 - Q.mass)
            - 1738.251 * max(0.0, 0.008678045 - Q.girth2)
            + 233.3178 * max(0.0, 0.00733008 - Q.lam1)
            + 13.69197 * max(0.0, Q.sj3_dr_max - 0.1879486)
            - 0.04309208 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        ))
        + 0.21875 * grid(7, max(0.0, 8.388207
            - 905.0393 * max(0.0, Q.girth2 - 0.007520088)
            + 1825.521 * max(0.0, Q.girth2 - 0.01323868)
            - 305.7379 * max(0.0, Q.girth2 - 0.004372139)
            + 114.0587 * max(0.0, Q.mass_over_sum_pt - 0.07269073)
            + 470.1983 * max(0.0, Q.girth2 - 0.001653836)
            + 34.21965 * max(0.0, 0.05356915 - Q.tau1)
            - 0.2608814 * max(0.0, Q.mass - 76.6557)
            - 335.5297 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            + 501.4627 * max(0.0, Q.width - 0.008678045)
            + 200.5578 * max(0.0, Q.mass_over_sum_pt - 0.08475161)
            - 88.799 * max(0.0, 0.02076709 - Q.centroid_offset)
            + 0.08619455 * max(0.0, Q.mass - 36.22941)
            - 507325.9 * max(0.0, 8.10414e-05 - Q.girth2_top2)
            - 3857.88 * max(0.0, 0.0009641429 - Q.girth2)
            - 24.49241 * max(0.0, 0.1872617 - Q.sj2_dr)
            - 60.46098 * max(0.0, 0.08723651 - Q.girth)
            + 39.1567 * max(0.0, 0.1591713 - Q.sj2_dr)
            + 10.04638 * max(0.0, 0.3033137 - Q.LHA)
            - 1202.871 * max(0.0, Q.width - 0.005590289)
            - 38.45169 * max(0.0, Q.mass_over_sum_pt - 0.01109984)
            - 347.5352 * max(0.0, 0.008375572 - Q.lam1)
            - 31629.58 * max(0.0, 8.147744e-05 - Q.e3)
            - 304.9666 * max(0.0, Q.e2 - 0.05028464)
        ))
        - 0.25 * grid(8, max(0.0, -0.7267793
            + 230.018 * max(0.0, 0.005019719 - Q.girth2)
            + 33692.16 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            - 64.20506 * max(0.0, 0.1967397 - Q.LHA)
            - 12.38385 * max(0.0, Q.log_sum_pt - 6.701242)
            - 75.46003 * max(0.0, 0.06108601 - Q.girth)
            + 0.01007342 * max(0.0, Q.sum_pt_top5 - 687.4375)
            + 4974.301 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4)
            + 276112.5 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2)
            - 519602.5 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 2.955458e-05 - Q.e3)
            - 24783.14 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738)
            + 89.18823 * max(0.0, 0.03319429 - Q.mass_over_sum_pt)
            - 31.67499 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            + 1019.179 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.008678045 - Q.width)
            - 44.36972 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.003562611 - Q.width)
            + 2399.552 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.girth2)
            + 22400.78 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.width)
        ))
        - 0.03125 * grid(9, max(0.0, -2.832185
            - 64.86963 * max(0.0, 0.05464922 - Q.girth)
            + 89.9129 * max(0.0, 0.04369778 - Q.tau1)
            - 0.09226678 * max(0.0, 53.33237 - Q.mass)
            + 6.692211 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset)
            + 1768.36 * max(0.0, 0.00609665 - Q.girth2)
            + 22.75828 * max(0.0, 0.213399 - Q.sj3_dr_max)
            + 579.7119 * max(0.0, Q.lam2 - 0.001130645)
            + 0.1183909 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt)
            + 0.9639006 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
            - 60.9241 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            - 255.1962 * max(0.0, Q.lam1 - 0.003377388)
            + 200.828 * max(0.0, 0.01837778 - Q.centroid_offset)
            - 53.04834 * max(0.0, Q.e2 - 0.03556091)
            - 0.1499527 * max(0.0, 24.23013 - Q.sj3_pair_mass_max)
            + 529.3897 * max(0.0, Q.lam1 - 0.005433361)
            - 72.7653 * max(0.0, 0.0284695 - Q.C3)
            - 748.4049 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266)
            - 84531.47 * max(0.0, 2.371297e-05 - Q.e3)
            + 2000.407 * max(0.0, 0.003562611 - Q.girth2)
            - 66.49413 * max(0.0, 0.07269073 - Q.mass_over_sum_pt)
            + 0.008796196 * max(0.0, 988.4078 - Q.sum_pt)
            + 170.6707 * max(0.0, Q.e2 - 0.04447357)
            - 687.1715 * max(0.0, 0.00595415 - Q.lam1)
            - 8.805234 * Q.z_dr_0p2_0p4
            - 62.37537 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0)
            - 3.009064 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1)
            + 1848.072 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd)
            - 77.76084 * max(0.0, Q.girth - 0.0717028)
            + 51.85093 * max(0.0, 0.1117619 - Q.max_dr)
        ))
        + 0.375 * grid(11, max(0.0, -3.687039
            + 314.0508 * max(0.0, 0.01323868 - Q.girth2)
            - 20.73316 * max(0.0, 0.09538712 - Q.tau1)
            + 0.1493811 * max(0.0, 15.45403 - Q.mass)
            + 92.23491 * max(0.0, 0.03776099 - Q.centroid_offset)
            - 0.2027867 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt)
            - 69.97559 * max(0.0, 0.169029 - Q.sj3_dr_max)
            + 1513.354 * max(0.0, 0.008678045 - Q.width)
            - 221.4018 * max(0.0, 0.02054282 - Q.girth)
            - 224.5919 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2)
            - 383.7354 * max(0.0, 0.008168571 - Q.e2_sq)
            - 70.51385 * max(0.0, 0.0717028 - Q.girth)
            + 26.13782 * max(0.0, Q.z_7 - 0.01685855)
            + 35.06553 * max(0.0, Q.sj2_dr - 0.2687922)
            + 14.89981 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            + 21.14689 * max(0.0, 0.2623172 - Q.sj3_dr_max)
            + 904.1195 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.5502779 - Q.tau32)
            - 486.1113 * max(0.0, 0.008375572 - Q.lam1)
            + 16.0053 * max(0.0, 0.1452311 - Q.max_dr)
        ))
        + 0.0703125 * grid(13, max(0.0, -0.4485388
            + 44.61491 * max(0.0, 0.1484084 - Q.girth)
            - 45.16549 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 1.463587 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
            + 156.5979 * max(0.0, 0.01643375 - Q.lam1)
            + 0.001170437 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7)
            - 18.71031 * max(0.0, 0.08000524 - Q.e2)
            - 254.8358 * max(0.0, 0.02807091 - Q.z_7)
            + 945.0926 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.tau2 - 0.008780509)
            - 0.0006137886 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6)
        ))
        - 0.75 * grid(14, max(0.0, -3.181758
            - 7276.977 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1)
            + 11051.99 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.width)
            - 1372.959 * max(0.0, Q.girth2 - 0.007520088)
            + 144.4483 * max(0.0, Q.e2_sq - 0.002074109)
            + 325.1417 * max(0.0, Q.width - 0.01323868)
            + 291.1275 * max(0.0, Q.width - 0.005590289)
            - 77.92285 * max(0.0, Q.e2 - 0.03556091)
            + 445.2716 * max(0.0, Q.e2 - 0.05028464)
            - 982.2945 * max(0.0, Q.centroid_offset - 0.04990367)
            + 156.5575 * max(0.0, Q.e2 - 0.04110972)
            - 156.0067 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            - 24.17888 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095)
            + 24.07586 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
            - 212.952 * max(0.0, Q.e2 - 0.01655442)
            - 0.02751436 * max(0.0, 74.57663 - Q.sd_mass)
            + 0.03164259 * max(0.0, 49.91626 - Q.sd_mass)
            + 234.1635 * max(0.0, Q.mass_over_sum_pt - 0.08475161)
            - 197.7289 * max(0.0, Q.e2_sq - 0.0030133)
            - 101.6813 * max(0.0, Q.girth - 0.1019409)
            + 96.64194 * max(0.0, Q.girth - 0.04081947)
            - 111.5262 * max(0.0, Q.centroid_offset - 0.02685622)
            - 314.8012 * max(0.0, Q.girth - 0.08723651)
            + 89.92855 * max(0.0, Q.girth - 0.08065885)
            - 2007.011 * max(0.0, Q.width - 0.006679471)
            - 13.86663 * max(0.0, Q.sj2_dr - 0.06154135)
            + 68.23302 * max(0.0, Q.sj2_dr - 0.1591713)
            - 29.57141 * max(0.0, Q.sj2_dr - 0.2001708)
            + 121.3632 * max(0.0, Q.girth - 0.02689598)
            + 343.9398 * max(0.0, Q.width - 0.003562611)
            + 210.088 * max(0.0, Q.mass_over_sum_pt - 0.07992374)
        ))
        - 0.6875 * grid(15, max(0.0, -1.025458
            - 20.70682 * max(0.0, 0.2233283 - Q.N2)
            - 1053.515 * max(0.0, 0.007520088 - Q.girth2)
            - 12.41601 * max(0.0, 0.1309286 - Q.mass_over_sum_pt)
            - 1100.358 * max(0.0, 0.00609665 - Q.width)
            + 666.1981 * max(0.0, 0.01323868 - Q.width)
            - 2847.625 * max(0.0, 0.007520088 - Q.girth2) * max(0.0, 0.7459513 - Q.D2)
            + 146.7409 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2)
            + 174.9853 * max(0.0, 0.04110972 - Q.e2)
            - 289.3842 * max(0.0, 0.01165737 - Q.e2_sq)
            + 746.0186 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq)
            - 761.0077 * max(0.0, 0.008168571 - Q.e2_sq)
            + 201.8491 * max(0.0, 0.01882765 - Q.width)
            - 28.63268 * max(0.0, 0.1019409 - Q.girth)
            + 61.51609 * max(0.0, 0.1591713 - Q.sj2_dr)
            - 15.22296 * max(0.0, 0.2001708 - Q.sj2_dr)
            + 0.07160176 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75)
            - 38.54002 * max(0.0, 0.1492731 - Q.sj2_dr)
            + 1686.831 * max(0.0, 0.0005124533 - Q.girth2_top2)
            + 184.3761 * max(0.0, 0.006756161 - Q.girth2_top3)
            + 449.8922 * max(0.0, 0.005433361 - Q.lam1)
            - 154.3123 * max(0.0, 0.008375572 - Q.lam1)
            - 202.3983 * max(0.0, 0.01837778 - Q.centroid_offset)
            - 449.5484 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2512159 - Q.dr01)
        ))
    )


def logit_Z(Q):
    return (-0.09375
        + 0.125 * grid(1, max(0.0, -0.09412154
            + 0.1487708 * max(0.0, Q.pt_7 - 34.53125)
            + 5.815856 * max(0.0, Q.log_sum_pt - 6.377723)
            + 633.314 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq)
            - 77.3932 * max(0.0, 0.06164517 - Q.z_7)
            - 540.0432 * max(0.0, 0.008678045 - Q.girth2)
            - 9.214913 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2)
            - 3.109254 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1)
            + 219.1891 * max(0.0, 0.008168571 - Q.e2_sq)
            - 0.004414408 * max(0.0, Q.sum_pt_top5 - 531.1875)
            - 417.6075 * max(0.0, 0.00609665 - Q.width)
            + 5.885901 * max(0.0, Q.log_sum_pt - 6.502799)
            + 0.004937477 * max(0.0, Q.sum_pt_top5 - 367.5938)
        ))
        - 0.5625 * grid(3, max(0.0, -4.866957
            + 33.64804 * max(0.0, Q.mass_over_sum_pt - 0.0681391)
            - 50.95515 * max(0.0, Q.tau1 - 0.05356915)
            + 155.2936 * max(0.0, Q.lam1 - 0.01643375)
            + 75.26761 * max(0.0, Q.girth - 0.07608178)
            - 226.6729 * max(0.0, Q.width - 0.01323868)
            + 559.5157 * max(0.0, Q.width - 0.007520088)
            + 278.6867 * max(0.0, Q.e2 - 0.06344108)
            + 101.1847 * max(0.0, Q.girth - 0.04081947)
            + 60.75642 * max(0.0, Q.tau1 - 0.1027642)
            + 130.0719 * max(0.0, Q.girth - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494)
            - 1560.378 * max(0.0, Q.girth2 - 0.008678045)
            + 32.90918 * max(0.0, Q.sj2_dr - 0.1872617)
            - 3751.477 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr)
            + 13.73175 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50)
            + 315.049 * max(0.0, Q.lam1 - 0.008375572)
            - 978.3712 * max(0.0, Q.lam2 - 0.001130645)
            + 34.65721 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6)
            - 356.8522 * max(0.0, Q.lam1 - 0.01200373)
            - 0.09491761 * max(0.0, Q.mass - 64.61873)
            - 545.2256 * max(0.0, 0.004372139 - Q.girth2)
        ))
        + 0.078125 * grid(4, max(0.0, -3.164716
            + 64.31693 * max(0.0, 0.2233283 - Q.N2)
            - 2741.871 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.e2_sq - 0.01165737)
            - 9547.75 * max(0.0, 0.000537286 - Q.lam2)
            - 148.3439 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            + 442.9708 * max(0.0, Q.girth2 - 0.003562611)
            - 0.2348624 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.25 - Q.pt_6)
            - 86.98067 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266)
            - 310.1108 * max(0.0, 0.008678045 - Q.girth2)
            - 21.48242 * max(0.0, Q.sj3_dr_max - 0.233678)
            + 761.9908 * max(0.0, 0.01165737 - Q.e2_sq)
            + 0.05472227 * max(0.0, 76.6557 - Q.mass)
            + 286.7206 * max(0.0, Q.tau1 - 0.07283629) * max(0.0, 0.2089872 - Q.sj3_dr_min)
            - 0.3276204 * max(0.0, 76.6557 - Q.mass) * max(0.0, 0.875672 - Q.D2)
            + 634.2974 * max(0.0, 0.01165737 - Q.e2_sq) * max(0.0, 0.875672 - Q.D2)
            + 23.96368 * max(0.0, Q.max_dr - 0.1117619)
            + 1363.239 * max(0.0, 0.002635418 - Q.girth2)
            - 585.9924 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, 0.02117036 - Q.dr01)
            - 204.6598 * max(0.0, 0.01323868 - Q.width)
            + 24.52623 * max(0.0, Q.e2 - 0.009668065)
            - 83.57821 * max(0.0, Q.girth - 0.05464922)
            + 635.306 * max(0.0, 0.009032972 - Q.C2_b2)
            + 197.4851 * max(0.0, 0.01716248 - Q.e2_sq)
            + 40.95097 * max(0.0, Q.tau1 - 0.1136369)
            - 380.6967 * max(0.0, 0.01882765 - Q.girth2)
            + 44.53423 * max(0.0, 1.002471 - Q.D2) * max(0.0, Q.M3 - 0.04581318)
            - 0.07156271 * max(0.0, 69.61135 - Q.mass)
        ))
        + 0.015625 * grid(5, max(0.0, 1.685334
            + 81.98639 * max(0.0, 0.04939969 - Q.z_7)
            - 153.4978 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 51.68755 * max(0.0, Q.log_sum_pt - 6.896095)
            + 760.8062 * max(0.0, 0.005284669 - Q.e2_sq)
            - 1600.013 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset)
            + 0.8812857 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 42.18339 - Q.mass_top3)
            - 0.09400035 * max(0.0, 37.15625 - Q.pt_7)
            + 160212.7 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            + 169.6404 * max(0.0, 0.02807091 - Q.z_7)
            - 128.5201 * max(0.0, 0.02383244 - Q.zdr_0)
            - 6.093912 * max(0.0, 0.3012016 - Q.sj3_dr_max)
            - 17.92183 * max(0.0, Q.log_sum_pt - 6.701242)
            + 3.944505 * max(0.0, Q.log_sum_pt - 6.267538)
            + 0.03822879 * max(0.0, 60.63098 - Q.mass)
            - 562.8433 * max(0.0, 0.002915531 - Q.girth2_top3)
            - 0.04790254 * max(0.0, Q.pt_7 - 23.21641)
            - 29.95585 * max(0.0, 0.09041383 - Q.mass_over_sum_pt)
            + 11.60847 * max(0.0, 0.09538712 - Q.tau1)
            + 53078.77 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0002302115 - Q.mean_phi2)
            + 195913.6 * max(0.0, 0.002915531 - Q.girth2_top3) * max(0.0, 0.006646257 - Q.tau3)
        ))
        - 0.375 * grid(6, max(0.0, 0.7932636
            + 151.4742 * max(0.0, Q.centroid_offset - 0.00809236)
            - 404.8059 * max(0.0, 0.01323868 - Q.width)
            + 3.010573 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308)
            + 1.173602 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt)
            + 62.78069 * max(0.0, 0.1136369 - Q.tau1)
            - 26.39771 * max(0.0, Q.sj3_dr_min - 0.1278212)
            - 189.338 * max(0.0, Q.centroid_offset - 0.01837778)
            + 285.4071 * max(0.0, 0.01200373 - Q.lam1)
            + 8349.59 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2)
            - 0.105165 * max(0.0, Q.sj3_pair_mass_min - 4.501727)
            - 13.53752 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757)
            + 1180.545 * max(0.0, 0.0030133 - Q.e2_sq)
            - 2047.89 * max(0.0, 0.001130645 - Q.lam2)
            - 0.814989 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4)
            + 43.57116 * max(0.0, 0.1789613 - Q.sj3_dr_max)
            - 43.15722 * max(0.0, 0.1452311 - Q.max_dr)
            + 108.5842 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05)
            + 6230.968 * max(0.0, 0.0005116989 - Q.e3)
            + 0.03968707 * max(0.0, 45.7571 - Q.mass)
            - 1738.251 * max(0.0, 0.008678045 - Q.girth2)
            + 233.3178 * max(0.0, 0.00733008 - Q.lam1)
            + 13.69197 * max(0.0, Q.sj3_dr_max - 0.1879486)
            - 0.04309208 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        ))
        + 0.46875 * grid(7, max(0.0, 8.388207
            - 905.0393 * max(0.0, Q.girth2 - 0.007520088)
            + 1825.521 * max(0.0, Q.girth2 - 0.01323868)
            - 305.7379 * max(0.0, Q.girth2 - 0.004372139)
            + 114.0587 * max(0.0, Q.mass_over_sum_pt - 0.07269073)
            + 470.1983 * max(0.0, Q.girth2 - 0.001653836)
            + 34.21965 * max(0.0, 0.05356915 - Q.tau1)
            - 0.2608814 * max(0.0, Q.mass - 76.6557)
            - 335.5297 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            + 501.4627 * max(0.0, Q.width - 0.008678045)
            + 200.5578 * max(0.0, Q.mass_over_sum_pt - 0.08475161)
            - 88.799 * max(0.0, 0.02076709 - Q.centroid_offset)
            + 0.08619455 * max(0.0, Q.mass - 36.22941)
            - 507325.9 * max(0.0, 8.10414e-05 - Q.girth2_top2)
            - 3857.88 * max(0.0, 0.0009641429 - Q.girth2)
            - 24.49241 * max(0.0, 0.1872617 - Q.sj2_dr)
            - 60.46098 * max(0.0, 0.08723651 - Q.girth)
            + 39.1567 * max(0.0, 0.1591713 - Q.sj2_dr)
            + 10.04638 * max(0.0, 0.3033137 - Q.LHA)
            - 1202.871 * max(0.0, Q.width - 0.005590289)
            - 38.45169 * max(0.0, Q.mass_over_sum_pt - 0.01109984)
            - 347.5352 * max(0.0, 0.008375572 - Q.lam1)
            - 31629.58 * max(0.0, 8.147744e-05 - Q.e3)
            - 304.9666 * max(0.0, Q.e2 - 0.05028464)
        ))
        - 0.03125 * grid(9, max(0.0, -2.832185
            - 64.86963 * max(0.0, 0.05464922 - Q.girth)
            + 89.9129 * max(0.0, 0.04369778 - Q.tau1)
            - 0.09226678 * max(0.0, 53.33237 - Q.mass)
            + 6.692211 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset)
            + 1768.36 * max(0.0, 0.00609665 - Q.girth2)
            + 22.75828 * max(0.0, 0.213399 - Q.sj3_dr_max)
            + 579.7119 * max(0.0, Q.lam2 - 0.001130645)
            + 0.1183909 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt)
            + 0.9639006 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
            - 60.9241 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            - 255.1962 * max(0.0, Q.lam1 - 0.003377388)
            + 200.828 * max(0.0, 0.01837778 - Q.centroid_offset)
            - 53.04834 * max(0.0, Q.e2 - 0.03556091)
            - 0.1499527 * max(0.0, 24.23013 - Q.sj3_pair_mass_max)
            + 529.3897 * max(0.0, Q.lam1 - 0.005433361)
            - 72.7653 * max(0.0, 0.0284695 - Q.C3)
            - 748.4049 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266)
            - 84531.47 * max(0.0, 2.371297e-05 - Q.e3)
            + 2000.407 * max(0.0, 0.003562611 - Q.girth2)
            - 66.49413 * max(0.0, 0.07269073 - Q.mass_over_sum_pt)
            + 0.008796196 * max(0.0, 988.4078 - Q.sum_pt)
            + 170.6707 * max(0.0, Q.e2 - 0.04447357)
            - 687.1715 * max(0.0, 0.00595415 - Q.lam1)
            - 8.805234 * Q.z_dr_0p2_0p4
            - 62.37537 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0)
            - 3.009064 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1)
            + 1848.072 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd)
            - 77.76084 * max(0.0, Q.girth - 0.0717028)
            + 51.85093 * max(0.0, 0.1117619 - Q.max_dr)
        ))
        + 0.0546875 * grid(13, max(0.0, -0.4485388
            + 44.61491 * max(0.0, 0.1484084 - Q.girth)
            - 45.16549 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 1.463587 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
            + 156.5979 * max(0.0, 0.01643375 - Q.lam1)
            + 0.001170437 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7)
            - 18.71031 * max(0.0, 0.08000524 - Q.e2)
            - 254.8358 * max(0.0, 0.02807091 - Q.z_7)
            + 945.0926 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.tau2 - 0.008780509)
            - 0.0006137886 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6)
        ))
        + 0.375 * grid(14, max(0.0, -3.181758
            - 7276.977 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1)
            + 11051.99 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.width)
            - 1372.959 * max(0.0, Q.girth2 - 0.007520088)
            + 144.4483 * max(0.0, Q.e2_sq - 0.002074109)
            + 325.1417 * max(0.0, Q.width - 0.01323868)
            + 291.1275 * max(0.0, Q.width - 0.005590289)
            - 77.92285 * max(0.0, Q.e2 - 0.03556091)
            + 445.2716 * max(0.0, Q.e2 - 0.05028464)
            - 982.2945 * max(0.0, Q.centroid_offset - 0.04990367)
            + 156.5575 * max(0.0, Q.e2 - 0.04110972)
            - 156.0067 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            - 24.17888 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095)
            + 24.07586 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
            - 212.952 * max(0.0, Q.e2 - 0.01655442)
            - 0.02751436 * max(0.0, 74.57663 - Q.sd_mass)
            + 0.03164259 * max(0.0, 49.91626 - Q.sd_mass)
            + 234.1635 * max(0.0, Q.mass_over_sum_pt - 0.08475161)
            - 197.7289 * max(0.0, Q.e2_sq - 0.0030133)
            - 101.6813 * max(0.0, Q.girth - 0.1019409)
            + 96.64194 * max(0.0, Q.girth - 0.04081947)
            - 111.5262 * max(0.0, Q.centroid_offset - 0.02685622)
            - 314.8012 * max(0.0, Q.girth - 0.08723651)
            + 89.92855 * max(0.0, Q.girth - 0.08065885)
            - 2007.011 * max(0.0, Q.width - 0.006679471)
            - 13.86663 * max(0.0, Q.sj2_dr - 0.06154135)
            + 68.23302 * max(0.0, Q.sj2_dr - 0.1591713)
            - 29.57141 * max(0.0, Q.sj2_dr - 0.2001708)
            + 121.3632 * max(0.0, Q.girth - 0.02689598)
            + 343.9398 * max(0.0, Q.width - 0.003562611)
            + 210.088 * max(0.0, Q.mass_over_sum_pt - 0.07992374)
        ))
        - 0.15625 * grid(15, max(0.0, -1.025458
            - 20.70682 * max(0.0, 0.2233283 - Q.N2)
            - 1053.515 * max(0.0, 0.007520088 - Q.girth2)
            - 12.41601 * max(0.0, 0.1309286 - Q.mass_over_sum_pt)
            - 1100.358 * max(0.0, 0.00609665 - Q.width)
            + 666.1981 * max(0.0, 0.01323868 - Q.width)
            - 2847.625 * max(0.0, 0.007520088 - Q.girth2) * max(0.0, 0.7459513 - Q.D2)
            + 146.7409 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2)
            + 174.9853 * max(0.0, 0.04110972 - Q.e2)
            - 289.3842 * max(0.0, 0.01165737 - Q.e2_sq)
            + 746.0186 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq)
            - 761.0077 * max(0.0, 0.008168571 - Q.e2_sq)
            + 201.8491 * max(0.0, 0.01882765 - Q.width)
            - 28.63268 * max(0.0, 0.1019409 - Q.girth)
            + 61.51609 * max(0.0, 0.1591713 - Q.sj2_dr)
            - 15.22296 * max(0.0, 0.2001708 - Q.sj2_dr)
            + 0.07160176 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75)
            - 38.54002 * max(0.0, 0.1492731 - Q.sj2_dr)
            + 1686.831 * max(0.0, 0.0005124533 - Q.girth2_top2)
            + 184.3761 * max(0.0, 0.006756161 - Q.girth2_top3)
            + 449.8922 * max(0.0, 0.005433361 - Q.lam1)
            - 154.3123 * max(0.0, 0.008375572 - Q.lam1)
            - 202.3983 * max(0.0, 0.01837778 - Q.centroid_offset)
            - 449.5484 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2512159 - Q.dr01)
        ))
    )


def logit_t(Q):
    return (1.34375
        + 0.015625 * grid(0, max(0.0, -2.417431
            + 13.83471 * max(0.0, 0.1484197 - Q.planar_flow)
            - 2262.493 * max(0.0, 0.004372139 - Q.width)
            + 0.1055477 * max(0.0, 21.78408 - Q.mass)
            + 235.418 * max(0.0, 0.01323868 - Q.girth2)
            + 365.8228 * max(0.0, 0.006506576 - Q.lam1)
            - 2988.999 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2)
            - 0.02260682 * max(0.0, Q.sum_pt - 901.5938)
            + 461.6906 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 1.002471 - Q.D2)
            - 0.0689348 * max(0.0, 56.92035 - Q.mass)
            + 602.5658 * max(0.0, 0.008678045 - Q.width)
            - 1364.818 * max(0.0, 0.001563465 - Q.C2_b2)
            - 29.28583 * max(0.0, Q.sj3_dr_max - 0.233678)
            + 7.116758 * max(0.0, Q.sj3_dr_max - 0.1070199)
            + 10.34001 * max(0.0, 0.3012016 - Q.sj3_dr_max)
            - 12219.26 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.centroid_offset - 0.01837778)
            - 43.05435 * max(0.0, 0.08723651 - Q.girth)
            + 227.4702 * max(0.0, 0.0245477 - Q.e2)
            - 305.1563 * max(0.0, 0.005433361 - Q.lam1)
            + 0.40509 * max(0.0, Q.n_dr_0_0p05 - 4.0)
            - 13.21815 * max(0.0, Q.log_sum_pt - 6.670067)
            + 0.01865327 * max(0.0, Q.sum_pt_top5 - 687.4375)
            + 73765.7 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2)
        ))
        + 0.0625 * grid(3, max(0.0, -4.866957
            + 33.64804 * max(0.0, Q.mass_over_sum_pt - 0.0681391)
            - 50.95515 * max(0.0, Q.tau1 - 0.05356915)
            + 155.2936 * max(0.0, Q.lam1 - 0.01643375)
            + 75.26761 * max(0.0, Q.girth - 0.07608178)
            - 226.6729 * max(0.0, Q.width - 0.01323868)
            + 559.5157 * max(0.0, Q.width - 0.007520088)
            + 278.6867 * max(0.0, Q.e2 - 0.06344108)
            + 101.1847 * max(0.0, Q.girth - 0.04081947)
            + 60.75642 * max(0.0, Q.tau1 - 0.1027642)
            + 130.0719 * max(0.0, Q.girth - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494)
            - 1560.378 * max(0.0, Q.girth2 - 0.008678045)
            + 32.90918 * max(0.0, Q.sj2_dr - 0.1872617)
            - 3751.477 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr)
            + 13.73175 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50)
            + 315.049 * max(0.0, Q.lam1 - 0.008375572)
            - 978.3712 * max(0.0, Q.lam2 - 0.001130645)
            + 34.65721 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6)
            - 356.8522 * max(0.0, Q.lam1 - 0.01200373)
            - 0.09491761 * max(0.0, Q.mass - 64.61873)
            - 545.2256 * max(0.0, 0.004372139 - Q.girth2)
        ))
        + 0.125 * grid(4, max(0.0, -3.164716
            + 64.31693 * max(0.0, 0.2233283 - Q.N2)
            - 2741.871 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.e2_sq - 0.01165737)
            - 9547.75 * max(0.0, 0.000537286 - Q.lam2)
            - 148.3439 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            + 442.9708 * max(0.0, Q.girth2 - 0.003562611)
            - 0.2348624 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.25 - Q.pt_6)
            - 86.98067 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266)
            - 310.1108 * max(0.0, 0.008678045 - Q.girth2)
            - 21.48242 * max(0.0, Q.sj3_dr_max - 0.233678)
            + 761.9908 * max(0.0, 0.01165737 - Q.e2_sq)
            + 0.05472227 * max(0.0, 76.6557 - Q.mass)
            + 286.7206 * max(0.0, Q.tau1 - 0.07283629) * max(0.0, 0.2089872 - Q.sj3_dr_min)
            - 0.3276204 * max(0.0, 76.6557 - Q.mass) * max(0.0, 0.875672 - Q.D2)
            + 634.2974 * max(0.0, 0.01165737 - Q.e2_sq) * max(0.0, 0.875672 - Q.D2)
            + 23.96368 * max(0.0, Q.max_dr - 0.1117619)
            + 1363.239 * max(0.0, 0.002635418 - Q.girth2)
            - 585.9924 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, 0.02117036 - Q.dr01)
            - 204.6598 * max(0.0, 0.01323868 - Q.width)
            + 24.52623 * max(0.0, Q.e2 - 0.009668065)
            - 83.57821 * max(0.0, Q.girth - 0.05464922)
            + 635.306 * max(0.0, 0.009032972 - Q.C2_b2)
            + 197.4851 * max(0.0, 0.01716248 - Q.e2_sq)
            + 40.95097 * max(0.0, Q.tau1 - 0.1136369)
            - 380.6967 * max(0.0, 0.01882765 - Q.girth2)
            + 44.53423 * max(0.0, 1.002471 - Q.D2) * max(0.0, Q.M3 - 0.04581318)
            - 0.07156271 * max(0.0, 69.61135 - Q.mass)
        ))
        - 0.25 * grid(5, max(0.0, 1.685334
            + 81.98639 * max(0.0, 0.04939969 - Q.z_7)
            - 153.4978 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 51.68755 * max(0.0, Q.log_sum_pt - 6.896095)
            + 760.8062 * max(0.0, 0.005284669 - Q.e2_sq)
            - 1600.013 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset)
            + 0.8812857 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 42.18339 - Q.mass_top3)
            - 0.09400035 * max(0.0, 37.15625 - Q.pt_7)
            + 160212.7 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            + 169.6404 * max(0.0, 0.02807091 - Q.z_7)
            - 128.5201 * max(0.0, 0.02383244 - Q.zdr_0)
            - 6.093912 * max(0.0, 0.3012016 - Q.sj3_dr_max)
            - 17.92183 * max(0.0, Q.log_sum_pt - 6.701242)
            + 3.944505 * max(0.0, Q.log_sum_pt - 6.267538)
            + 0.03822879 * max(0.0, 60.63098 - Q.mass)
            - 562.8433 * max(0.0, 0.002915531 - Q.girth2_top3)
            - 0.04790254 * max(0.0, Q.pt_7 - 23.21641)
            - 29.95585 * max(0.0, 0.09041383 - Q.mass_over_sum_pt)
            + 11.60847 * max(0.0, 0.09538712 - Q.tau1)
            + 53078.77 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0002302115 - Q.mean_phi2)
            + 195913.6 * max(0.0, 0.002915531 - Q.girth2_top3) * max(0.0, 0.006646257 - Q.tau3)
        ))
        + 0.1875 * grid(8, max(0.0, -0.7267793
            + 230.018 * max(0.0, 0.005019719 - Q.girth2)
            + 33692.16 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            - 64.20506 * max(0.0, 0.1967397 - Q.LHA)
            - 12.38385 * max(0.0, Q.log_sum_pt - 6.701242)
            - 75.46003 * max(0.0, 0.06108601 - Q.girth)
            + 0.01007342 * max(0.0, Q.sum_pt_top5 - 687.4375)
            + 4974.301 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4)
            + 276112.5 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2)
            - 519602.5 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 2.955458e-05 - Q.e3)
            - 24783.14 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738)
            + 89.18823 * max(0.0, 0.03319429 - Q.mass_over_sum_pt)
            - 31.67499 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            + 1019.179 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.008678045 - Q.width)
            - 44.36972 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.003562611 - Q.width)
            + 2399.552 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.girth2)
            + 22400.78 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.width)
        ))
        + 0.375 * grid(10, max(0.0, 1.88596
            - 48.00774 * Q.e2
            + 0.1771744 * max(0.0, Q.sj3_pair_mass_min - 11.051)
            - 600.9459 * max(0.0, 0.004183811 - Q.lam1)
            + 530.9472 * max(0.0, Q.lam2 - 0.0003061234)
            - 626.2477 * max(0.0, Q.lam2 - 0.003408389)
            - 1905.25 * max(0.0, 0.001503553 - Q.lam1)
            - 29.69881 * max(0.0, Q.LHA - 0.3033137)
            + 50.7083 * max(0.0, Q.tau1 - 0.05356915)
            + 522.6854 * max(0.0, Q.girth2 - 0.007520088)
            - 301.0504 * max(0.0, Q.lam1 - 0.00733008)
            - 395.1002 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2)
            - 4828.053 * max(0.0, Q.e3 - 3.892127e-05)
            + 0.02897324 * max(0.0, 76.6557 - Q.mass)
            + 109.8528 * max(0.0, Q.C2_b2 - 0.009032972)
        ))
        - 0.5 * grid(12, max(0.0, -1.395752
            + 287.8117 * max(0.0, Q.girth2 - 0.01882765)
            + 0.1086228 * max(0.0, Q.mass - 88.15578)
            - 51.66452 * max(0.0, Q.mass_over_sum_pt - 0.1309286)
        ))
        - 0.40625 * grid(13, max(0.0, -0.4485388
            + 44.61491 * max(0.0, 0.1484084 - Q.girth)
            - 45.16549 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 1.463587 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
            + 156.5979 * max(0.0, 0.01643375 - Q.lam1)
            + 0.001170437 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7)
            - 18.71031 * max(0.0, 0.08000524 - Q.e2)
            - 254.8358 * max(0.0, 0.02807091 - Q.z_7)
            + 945.0926 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.tau2 - 0.008780509)
            - 0.0006137886 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6)
        ))
    )


def logits(Q):
    return [logit_g(Q), logit_q(Q), logit_W(Q), logit_Z(Q), logit_t(Q)]


def classify(pt, eta, phi):
    s = logits(quantities(pt, eta, phi))
    m = max(s)
    e = [math.exp(x - m) for x in s]
    p = [x / sum(e) for x in e]
    return CLASSES[s.index(m)], s, p


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 0
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
