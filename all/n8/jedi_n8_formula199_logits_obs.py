"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; all observables), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

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


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-0.4375
        - 0.15625 * grid(0, max(0.0, -0.0143
            - 1510.0 * max(0.0, 0.0016 - Q.C2_b2)
            + 197.0 * max(0.0, 0.025 - Q.e2)
            - 60.8 * max(0.0, 0.088 - Q.girth)
            - 17.3 * max(0.0, Q.log_sum_pt - 6.7)
            + 0.17 * max(0.0, 18.0 - Q.mass)
            - 0.0397 * max(0.0, 56.0 - Q.mass)
            + 0.338 * max(0.0, Q.n_dr_0_0p05 - 4.0)
            + 16.3 * max(0.0, 0.15 - Q.planar_flow)
            - 27.8 * max(0.0, Q.sj3_dr_max - 0.23)
            - 0.0179 * max(0.0, Q.sum_pt - 900.0)
            + 0.0163 * max(0.0, Q.sum_pt_top5 - 690.0)
            - 2220.0 * max(0.0, 0.0044 - Q.width)
            + 1080.0 * max(0.0, 0.0088 - Q.width)
            + 455.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, 1.0 - Q.D2)
            - 12700.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018)
            - 3060.0 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.87 - Q.D2)
            + 75500.0 * max(0.0, 7.1e-05 - Q.lam2) * max(0.0, 0.28 - Q.D2_b2)
        ))
        + 0.390625 * grid(1, max(0.0, 0.0983
            - 49.3 * max(0.0, 0.076 - Q.girth)
            - 333.0 * max(0.0, 0.0088 - Q.girth2)
            + 7.09 * max(0.0, Q.log_sum_pt - 6.3)
            + 5.88 * max(0.0, Q.log_sum_pt - 6.6)
            + 584.0 * max(0.0, 0.0064 - Q.mass_over_sum_pt_sq)
            + 0.125 * max(0.0, Q.pt_7 - 34.0)
            + 6.69 * max(0.0, Q.sj3_dr_max - 0.19)
            - 10.5 * max(0.0, Q.sj3_dr_min - 0.041)
            - 79.9 * max(0.0, 0.06 - Q.z_7)
            - 7.43 * max(0.0, Q.pt_7 - 33.0) * max(0.0, 0.017 - Q.tau2)
        ))
        + 0.4296875 * grid(2, max(0.0, 1.76
            - 8.76 * max(0.0, Q.LHA - 0.13)
            + 9.81 * max(0.0, Q.N2 - 0.12)
            + 312.0 * max(0.0, 0.006 - Q.lam1)
            + 2.77 * max(0.0, 6.5 - Q.log_sum_pt)
            - 0.0504 * max(0.0, 36.0 - Q.mass)
            - 0.0531 * max(0.0, 53.0 - Q.pt_7)
            + 0.0055 * max(0.0, 810.0 - Q.sum_pt)
            - 0.00806 * max(0.0, Q.sum_pt_top5 - 750.0)
            + 0.0373 * max(0.0, 53.0 - Q.pt_7) * max(0.0, 1.1 - Q.D2_b2)
        ))
        - 0.03125 * grid(4, max(0.0, 3.48
            + 56.3 * max(0.0, 0.21 - Q.N2)
            - 57.5 * max(0.0, 0.056 - Q.centroid_offset)
            + 414.0 * max(0.0, 0.0082 - Q.girth2_top2)
            - 11000.0 * max(0.0, 0.00054 - Q.lam2)
            - 0.223 * max(0.0, Q.mass - 76.0)
            - 121.0 * max(0.0, Q.mass_over_sum_pt - 0.086)
            + 24.9 * max(0.0, Q.max_dr - 0.15)
            + 0.166 * max(0.0, Q.sd_mass - 43.0)
            - 22.7 * max(0.0, 0.22 - Q.sj3_dr_max)
            + 4080.0 * max(0.0, 0.0017 - Q.width)
            - 0.72 * max(0.0, 0.22 - Q.N2) * max(0.0, 53.0 - Q.pt_7)
            - 61500.0 * max(0.0, 0.0093 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.00095)
            - 0.966 * max(0.0, Q.sd_mass - 43.0) * max(0.0, 0.29 - Q.sd_zg)
            - 5930.0 * max(0.0, Q.width - 0.005) * max(0.0, 0.11 - Q.sj3_z3)
        ))
        - 0.1875 * grid(5, max(0.0, 0.0645
            + 443.0 * max(0.0, 0.0055 - Q.e2_sq)
            + 1050.0 * max(0.0, 0.0074 - Q.girth)
            - 12.8 * max(0.0, Q.log_sum_pt - 6.7)
            - 43.3 * max(0.0, Q.log_sum_pt - 6.9)
            + 220.0 * max(0.0, 0.029 - Q.z_7)
            + 57.7 * max(0.0, 0.057 - Q.z_7)
            - 186.0 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
            + 124000.0 * max(0.0, 0.0017 - Q.girth2) * max(0.0, 0.03 - Q.centroid_offset)
            + 0.0319 * max(0.0, Q.sum_pt_top5 - 350.0) * max(0.0, Q.sj2_dr - 0.098)
            - 2100.0 * max(0.0, 0.075 - Q.z_7) * max(0.0, 0.025 - Q.centroid_offset)
        ))
        + 0.109375 * grid(6, max(0.0, 3.03
            + 150.0 * max(0.0, Q.centroid_offset - 0.0079)
            - 137.0 * max(0.0, Q.centroid_offset - 0.019)
            + 1150.0 * max(0.0, 0.0032 - Q.e2_sq)
            - 1620.0 * max(0.0, 0.0087 - Q.girth2)
            - 2270.0 * max(0.0, 0.0012 - Q.lam2)
            - 41.3 * max(0.0, 0.14 - Q.max_dr)
            + 9.18 * max(0.0, Q.sj3_dr_max - 0.18)
            + 46.0 * max(0.0, 0.18 - Q.sj3_dr_max)
            - 28.4 * max(0.0, Q.sj3_dr_min - 0.14)
            - 0.101 * max(0.0, Q.sj3_pair_mass_min - 4.0)
            + 71.3 * max(0.0, 0.11 - Q.tau1)
            + 96.0 * max(0.0, 0.0028 - Q.lam2) * max(0.0, 3.1 - Q.n_dr_0_0p05)
            - 0.798 * max(0.0, 64.0 - Q.mass) * max(0.0, 0.052 - Q.z_dr_0p2_0p4)
            + 1.02 * max(0.0, 41.0 - Q.pt_6) * max(0.0, 6.8 - Q.log_sum_pt)
            - 12.0 * max(0.0, 41.0 - Q.pt_6) * max(0.0, Q.z_7 - 0.023)
        ))
        + 0.171875 * grid(9, max(0.0, 0.0914
            - 81.1 * max(0.0, 0.028 - Q.C3)
            + 173.0 * max(0.0, 0.019 - Q.centroid_offset)
            + 100.0 * max(0.0, Q.e2 - 0.047)
            - 68.5 * max(0.0, Q.girth - 0.069)
            + 1380.0 * max(0.0, 0.0035 - Q.girth2)
            + 786.0 * max(0.0, 0.0059 - Q.girth2)
            + 784.0 * max(0.0, Q.lam2 - 0.00083)
            - 7.62 * max(0.0, Q.log_sum_pt - 6.4)
            + 9.64 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.102 * max(0.0, 55.0 - Q.mass)
            + 55.8 * max(0.0, 0.11 - Q.max_dr)
            - 74.1 * max(0.0, 0.14 - Q.sj3_dr_max)
            + 19.8 * max(0.0, 0.21 - Q.sj3_dr_max)
            + 0.0224 * max(0.0, 570.0 - Q.sum_pt)
            - 65.5 * max(0.0, 0.02 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.5)
            + 627.0 * max(0.0, 0.0086 - Q.girth2) * max(0.0, 0.33 - Q.planar_flow)
            + 6.06 * max(0.0, 54.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        ))
    )


def logit_q(Q):
    return (0.03125
        - 0.09375 * grid(4, max(0.0, 3.48
            + 56.3 * max(0.0, 0.21 - Q.N2)
            - 57.5 * max(0.0, 0.056 - Q.centroid_offset)
            + 414.0 * max(0.0, 0.0082 - Q.girth2_top2)
            - 11000.0 * max(0.0, 0.00054 - Q.lam2)
            - 0.223 * max(0.0, Q.mass - 76.0)
            - 121.0 * max(0.0, Q.mass_over_sum_pt - 0.086)
            + 24.9 * max(0.0, Q.max_dr - 0.15)
            + 0.166 * max(0.0, Q.sd_mass - 43.0)
            - 22.7 * max(0.0, 0.22 - Q.sj3_dr_max)
            + 4080.0 * max(0.0, 0.0017 - Q.width)
            - 0.72 * max(0.0, 0.22 - Q.N2) * max(0.0, 53.0 - Q.pt_7)
            - 61500.0 * max(0.0, 0.0093 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.00095)
            - 0.966 * max(0.0, Q.sd_mass - 43.0) * max(0.0, 0.29 - Q.sd_zg)
            - 5930.0 * max(0.0, Q.width - 0.005) * max(0.0, 0.11 - Q.sj3_z3)
        ))
        + 0.046875 * grid(5, max(0.0, 0.0645
            + 443.0 * max(0.0, 0.0055 - Q.e2_sq)
            + 1050.0 * max(0.0, 0.0074 - Q.girth)
            - 12.8 * max(0.0, Q.log_sum_pt - 6.7)
            - 43.3 * max(0.0, Q.log_sum_pt - 6.9)
            + 220.0 * max(0.0, 0.029 - Q.z_7)
            + 57.7 * max(0.0, 0.057 - Q.z_7)
            - 186.0 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
            + 124000.0 * max(0.0, 0.0017 - Q.girth2) * max(0.0, 0.03 - Q.centroid_offset)
            + 0.0319 * max(0.0, Q.sum_pt_top5 - 350.0) * max(0.0, Q.sj2_dr - 0.098)
            - 2100.0 * max(0.0, 0.075 - Q.z_7) * max(0.0, 0.025 - Q.centroid_offset)
        ))
        + 0.125 * grid(6, max(0.0, 3.03
            + 150.0 * max(0.0, Q.centroid_offset - 0.0079)
            - 137.0 * max(0.0, Q.centroid_offset - 0.019)
            + 1150.0 * max(0.0, 0.0032 - Q.e2_sq)
            - 1620.0 * max(0.0, 0.0087 - Q.girth2)
            - 2270.0 * max(0.0, 0.0012 - Q.lam2)
            - 41.3 * max(0.0, 0.14 - Q.max_dr)
            + 9.18 * max(0.0, Q.sj3_dr_max - 0.18)
            + 46.0 * max(0.0, 0.18 - Q.sj3_dr_max)
            - 28.4 * max(0.0, Q.sj3_dr_min - 0.14)
            - 0.101 * max(0.0, Q.sj3_pair_mass_min - 4.0)
            + 71.3 * max(0.0, 0.11 - Q.tau1)
            + 96.0 * max(0.0, 0.0028 - Q.lam2) * max(0.0, 3.1 - Q.n_dr_0_0p05)
            - 0.798 * max(0.0, 64.0 - Q.mass) * max(0.0, 0.052 - Q.z_dr_0p2_0p4)
            + 1.02 * max(0.0, 41.0 - Q.pt_6) * max(0.0, 6.8 - Q.log_sum_pt)
            - 12.0 * max(0.0, 41.0 - Q.pt_6) * max(0.0, Q.z_7 - 0.023)
        ))
        + 0.0625 * grid(8, max(0.0, -0.681
            - 60.8 * max(0.0, 0.21 - Q.LHA)
            - 5.19 * max(0.0, Q.log_sum_pt - 6.7)
            - 22.7 * max(0.0, 0.13 - Q.sj3_dr_max)
            + 48600.0 * max(0.0, 0.0063 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
            + 26900.0 * max(0.0, 0.058 - Q.tau1) * max(0.0, 0.0032 - Q.width)
        ))
        + 0.2539062 * grid(9, max(0.0, 0.0914
            - 81.1 * max(0.0, 0.028 - Q.C3)
            + 173.0 * max(0.0, 0.019 - Q.centroid_offset)
            + 100.0 * max(0.0, Q.e2 - 0.047)
            - 68.5 * max(0.0, Q.girth - 0.069)
            + 1380.0 * max(0.0, 0.0035 - Q.girth2)
            + 786.0 * max(0.0, 0.0059 - Q.girth2)
            + 784.0 * max(0.0, Q.lam2 - 0.00083)
            - 7.62 * max(0.0, Q.log_sum_pt - 6.4)
            + 9.64 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.102 * max(0.0, 55.0 - Q.mass)
            + 55.8 * max(0.0, 0.11 - Q.max_dr)
            - 74.1 * max(0.0, 0.14 - Q.sj3_dr_max)
            + 19.8 * max(0.0, 0.21 - Q.sj3_dr_max)
            + 0.0224 * max(0.0, 570.0 - Q.sum_pt)
            - 65.5 * max(0.0, 0.02 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.5)
            + 627.0 * max(0.0, 0.0086 - Q.girth2) * max(0.0, 0.33 - Q.planar_flow)
            + 6.06 * max(0.0, 54.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        ))
        - 0.125 * grid(10, max(0.0, 0.34
            + 128.0 * max(0.0, Q.C2_b2 - 0.0093)
            - 33.2 * max(0.0, Q.LHA - 0.3)
            - 5610.0 * max(0.0, Q.e3 - 7.5e-05)
            + 25100.0 * max(0.0, 8e-05 - Q.e3)
            + 340.0 * max(0.0, Q.girth2 - 0.0075)
            - 1920.0 * max(0.0, 0.0015 - Q.lam1)
            - 529.0 * max(0.0, 0.0056 - Q.lam1)
            + 871.0 * max(0.0, Q.lam2 - 0.00034)
            - 643.0 * max(0.0, Q.lam2 - 0.0034)
            + 0.03 * max(0.0, 75.0 - Q.mass)
            - 0.0398 * max(0.0, 46.0 - Q.pt_7)
            + 0.101 * max(0.0, Q.sj3_pair_mass_min - 11.0)
            + 28.0 * max(0.0, Q.tau1 - 0.057)
            - 141000.0 * max(0.0, 8.1e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.18)
            - 554.0 * max(0.0, Q.lam1 - 0.0074) * max(0.0, 0.38 - Q.D2_b2)
            + 0.129 * max(0.0, 45.0 - Q.pt_7) * max(0.0, 0.98 - Q.D2)
            + 204.0 * max(0.0, 0.022 - Q.zdr_0) * max(0.0, 0.24 - Q.z_dr_0p05_0p1)
        ))
        + 0.0625 * grid(15, max(0.0, -0.101
            + 203.0 * max(0.0, 0.04 - Q.e2)
            - 97.9 * max(0.0, 0.1 - Q.girth)
            - 1050.0 * max(0.0, 0.008 - Q.girth2)
            - 6550.0 * max(0.0, 0.0005 - Q.girth2_top2)
            + 48.5 * max(0.0, 0.16 - Q.sj2_dr)
            - 45.1 * max(0.0, 0.2 - Q.sj2_dr)
            + 560.0 * max(0.0, 0.014 - Q.width)
            + 56.4 * max(0.0, 0.15 - Q.mass_over_sum_pt) * max(0.0, 0.73 - Q.D2)
            - 7420.0 * max(0.0, 0.0061 - Q.width) * max(0.0, 0.72 - Q.D2)
        ))
    )


def logit_W(Q):
    return (-0.125
        + 0.34375 * grid(0, max(0.0, -0.0143
            - 1510.0 * max(0.0, 0.0016 - Q.C2_b2)
            + 197.0 * max(0.0, 0.025 - Q.e2)
            - 60.8 * max(0.0, 0.088 - Q.girth)
            - 17.3 * max(0.0, Q.log_sum_pt - 6.7)
            + 0.17 * max(0.0, 18.0 - Q.mass)
            - 0.0397 * max(0.0, 56.0 - Q.mass)
            + 0.338 * max(0.0, Q.n_dr_0_0p05 - 4.0)
            + 16.3 * max(0.0, 0.15 - Q.planar_flow)
            - 27.8 * max(0.0, Q.sj3_dr_max - 0.23)
            - 0.0179 * max(0.0, Q.sum_pt - 900.0)
            + 0.0163 * max(0.0, Q.sum_pt_top5 - 690.0)
            - 2220.0 * max(0.0, 0.0044 - Q.width)
            + 1080.0 * max(0.0, 0.0088 - Q.width)
            + 455.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, 1.0 - Q.D2)
            - 12700.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018)
            - 3060.0 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.87 - Q.D2)
            + 75500.0 * max(0.0, 7.1e-05 - Q.lam2) * max(0.0, 0.28 - Q.D2_b2)
        ))
        - 0.03125 * grid(1, max(0.0, 0.0983
            - 49.3 * max(0.0, 0.076 - Q.girth)
            - 333.0 * max(0.0, 0.0088 - Q.girth2)
            + 7.09 * max(0.0, Q.log_sum_pt - 6.3)
            + 5.88 * max(0.0, Q.log_sum_pt - 6.6)
            + 584.0 * max(0.0, 0.0064 - Q.mass_over_sum_pt_sq)
            + 0.125 * max(0.0, Q.pt_7 - 34.0)
            + 6.69 * max(0.0, Q.sj3_dr_max - 0.19)
            - 10.5 * max(0.0, Q.sj3_dr_min - 0.041)
            - 79.9 * max(0.0, 0.06 - Q.z_7)
            - 7.43 * max(0.0, Q.pt_7 - 33.0) * max(0.0, 0.017 - Q.tau2)
        ))
        - 0.5 * grid(3, max(0.0, -5.26
            + 322.0 * max(0.0, Q.e2 - 0.064)
            + 97.1 * max(0.0, Q.girth - 0.042)
            + 97.3 * max(0.0, Q.girth - 0.078)
            - 1670.0 * max(0.0, Q.girth2 - 0.0089)
            - 872.0 * max(0.0, 0.0044 - Q.girth2)
            + 578.0 * max(0.0, Q.lam1 - 0.0081)
            - 0.0864 * max(0.0, Q.mass - 65.0)
            + 37.7 * max(0.0, Q.sj2_dr - 0.18)
            + 12.6 * max(0.0, Q.centroid_offset - 0.01) * max(0.0, 8.4 - Q.n_pt_above_50)
            + 113.0 * max(0.0, Q.girth - 0.038) * max(0.0, Q.log_sum_pt - 6.1)
            + 3750.0 * max(0.0, Q.lam1 - 0.016) * max(0.0, 6.0 - Q.n_for_90pct)
            - 4280.0 * max(0.0, Q.mass_over_sum_pt - 0.066) * max(0.0, 0.22 - Q.sj2_dr)
        ))
        - 0.3125 * grid(6, max(0.0, 3.03
            + 150.0 * max(0.0, Q.centroid_offset - 0.0079)
            - 137.0 * max(0.0, Q.centroid_offset - 0.019)
            + 1150.0 * max(0.0, 0.0032 - Q.e2_sq)
            - 1620.0 * max(0.0, 0.0087 - Q.girth2)
            - 2270.0 * max(0.0, 0.0012 - Q.lam2)
            - 41.3 * max(0.0, 0.14 - Q.max_dr)
            + 9.18 * max(0.0, Q.sj3_dr_max - 0.18)
            + 46.0 * max(0.0, 0.18 - Q.sj3_dr_max)
            - 28.4 * max(0.0, Q.sj3_dr_min - 0.14)
            - 0.101 * max(0.0, Q.sj3_pair_mass_min - 4.0)
            + 71.3 * max(0.0, 0.11 - Q.tau1)
            + 96.0 * max(0.0, 0.0028 - Q.lam2) * max(0.0, 3.1 - Q.n_dr_0_0p05)
            - 0.798 * max(0.0, 64.0 - Q.mass) * max(0.0, 0.052 - Q.z_dr_0p2_0p4)
            + 1.02 * max(0.0, 41.0 - Q.pt_6) * max(0.0, 6.8 - Q.log_sum_pt)
            - 12.0 * max(0.0, 41.0 - Q.pt_6) * max(0.0, Q.z_7 - 0.023)
        ))
        + 0.21875 * grid(7, max(0.0, 6.92
            - 183.0 * max(0.0, Q.e2 - 0.05)
            - 38200.0 * max(0.0, 8.1e-05 - Q.e3)
            - 82.6 * max(0.0, 0.086 - Q.girth)
            + 1660.0 * max(0.0, Q.girth2 - 0.013)
            - 2720.0 * max(0.0, 0.001 - Q.girth2)
            + 0.0537 * max(0.0, Q.mass - 36.0)
            - 0.213 * max(0.0, Q.mass - 77.0)
            + 159.0 * max(0.0, Q.mass_over_sum_pt - 0.075)
            - 198.0 * max(0.0, Q.mass_over_sum_pt - 0.091)
            + 63.9 * max(0.0, 0.16 - Q.sj2_dr)
            - 45.0 * max(0.0, 0.19 - Q.sj2_dr)
            + 54.4 * max(0.0, 0.059 - Q.tau1)
            - 1330.0 * max(0.0, Q.width - 0.0056)
            + 1570.0 * max(0.0, Q.girth2 - 0.0032) * max(0.0, 0.21 - Q.planar_flow)
            - 2500.0 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.width - 0.008)
        ))
        - 0.25 * grid(8, max(0.0, -0.681
            - 60.8 * max(0.0, 0.21 - Q.LHA)
            - 5.19 * max(0.0, Q.log_sum_pt - 6.7)
            - 22.7 * max(0.0, 0.13 - Q.sj3_dr_max)
            + 48600.0 * max(0.0, 0.0063 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
            + 26900.0 * max(0.0, 0.058 - Q.tau1) * max(0.0, 0.0032 - Q.width)
        ))
        - 0.03125 * grid(9, max(0.0, 0.0914
            - 81.1 * max(0.0, 0.028 - Q.C3)
            + 173.0 * max(0.0, 0.019 - Q.centroid_offset)
            + 100.0 * max(0.0, Q.e2 - 0.047)
            - 68.5 * max(0.0, Q.girth - 0.069)
            + 1380.0 * max(0.0, 0.0035 - Q.girth2)
            + 786.0 * max(0.0, 0.0059 - Q.girth2)
            + 784.0 * max(0.0, Q.lam2 - 0.00083)
            - 7.62 * max(0.0, Q.log_sum_pt - 6.4)
            + 9.64 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.102 * max(0.0, 55.0 - Q.mass)
            + 55.8 * max(0.0, 0.11 - Q.max_dr)
            - 74.1 * max(0.0, 0.14 - Q.sj3_dr_max)
            + 19.8 * max(0.0, 0.21 - Q.sj3_dr_max)
            + 0.0224 * max(0.0, 570.0 - Q.sum_pt)
            - 65.5 * max(0.0, 0.02 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.5)
            + 627.0 * max(0.0, 0.0086 - Q.girth2) * max(0.0, 0.33 - Q.planar_flow)
            + 6.06 * max(0.0, 54.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        ))
        + 0.375 * grid(11, max(0.0, 0.835
            - 155.0 * max(0.0, Q.centroid_offset - 0.01)
            + 137.0 * max(0.0, 0.01 - Q.centroid_offset)
            - 263.0 * max(0.0, 0.021 - Q.girth)
            - 71.0 * max(0.0, 0.072 - Q.girth)
            + 0.182 * max(0.0, 16.0 - Q.mass)
            + 15.0 * max(0.0, 0.15 - Q.max_dr)
            + 23.6 * max(0.0, Q.sj2_dr - 0.27)
            - 54.4 * max(0.0, 0.17 - Q.sj3_dr_max)
            + 17.9 * max(0.0, 0.26 - Q.sj3_dr_max)
            + 708.0 * max(0.0, 0.0094 - Q.width)
            + 35.7 * max(0.0, Q.z_7 - 0.018)
            - 212.0 * max(0.0, Q.centroid_offset - 0.05) * max(0.0, 2.8 - Q.D2)
            - 0.247 * max(0.0, 0.041 - Q.centroid_offset) * max(0.0, 900.0 - Q.sum_pt)
            + 1160.0 * max(0.0, Q.centroid_offset - 0.047) * max(0.0, 0.64 - Q.tau32)
        ))
        + 0.0703125 * grid(13, max(0.0, 0.0376
            - 41.0 * max(0.0, 0.08 - Q.e2)
            + 52.6 * max(0.0, 0.15 - Q.girth)
            + 245.0 * max(0.0, 0.016 - Q.lam1)
            - 306.0 * max(0.0, 0.028 - Q.z_7)
            + 9000.0 * max(0.0, 5.8e-05 - Q.e3) * max(0.0, Q.D3 - 0.16)
            - 799000.0 * max(0.0, 6.1e-05 - Q.e3) * max(0.0, 0.038 - Q.centroid_offset)
            - 50.1 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
            - 1.48 * max(0.0, 0.15 - Q.girth) * max(0.0, 38.0 - Q.pt_7)
            - 0.000625 * max(0.0, Q.sum_pt - 990.0) * max(0.0, 62.0 - Q.pt_6)
            + 0.0011 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 41.0 - Q.pt_7)
        ))
        - 0.75 * grid(14, max(0.0, -1.75
            - 107.0 * max(0.0, Q.centroid_offset - 0.026)
            - 1110.0 * max(0.0, Q.centroid_offset - 0.05)
            - 218.0 * max(0.0, Q.e2 - 0.017)
            + 138.0 * max(0.0, Q.e2 - 0.042)
            + 373.0 * max(0.0, Q.e2 - 0.05)
            - 25100.0 * max(0.0, 5.1e-05 - Q.e3)
            + 109.0 * max(0.0, Q.girth - 0.027)
            + 65.0 * max(0.0, Q.girth - 0.042)
            - 257.0 * max(0.0, Q.girth - 0.089)
            - 47.4 * max(0.0, Q.girth - 0.094)
            - 2990.0 * max(0.0, Q.girth2 - 0.0075)
            + 327.0 * max(0.0, Q.mass_over_sum_pt - 0.085)
            + 0.0576 * max(0.0, 50.0 - Q.sd_mass)
            - 0.0523 * max(0.0, 75.0 - Q.sd_mass)
            + 41.1 * max(0.0, Q.sj2_dr - 0.16)
            - 20.0 * max(0.0, Q.sj2_dr - 0.2)
            + 270.0 * max(0.0, Q.width - 0.0033)
            + 896.0 * max(0.0, Q.width - 0.014)
            - 792.0 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 0.018 - Q.centroid_offset)
            - 9930.0 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.0072 - Q.lam1)
            + 1300.0 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.016 - Q.lam1)
            + 9130.0 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.0061 - Q.width)
        ))
        - 0.6875 * grid(15, max(0.0, -0.101
            + 203.0 * max(0.0, 0.04 - Q.e2)
            - 97.9 * max(0.0, 0.1 - Q.girth)
            - 1050.0 * max(0.0, 0.008 - Q.girth2)
            - 6550.0 * max(0.0, 0.0005 - Q.girth2_top2)
            + 48.5 * max(0.0, 0.16 - Q.sj2_dr)
            - 45.1 * max(0.0, 0.2 - Q.sj2_dr)
            + 560.0 * max(0.0, 0.014 - Q.width)
            + 56.4 * max(0.0, 0.15 - Q.mass_over_sum_pt) * max(0.0, 0.73 - Q.D2)
            - 7420.0 * max(0.0, 0.0061 - Q.width) * max(0.0, 0.72 - Q.D2)
        ))
    )


def logit_Z(Q):
    return (-0.09375
        + 0.125 * grid(1, max(0.0, 0.0983
            - 49.3 * max(0.0, 0.076 - Q.girth)
            - 333.0 * max(0.0, 0.0088 - Q.girth2)
            + 7.09 * max(0.0, Q.log_sum_pt - 6.3)
            + 5.88 * max(0.0, Q.log_sum_pt - 6.6)
            + 584.0 * max(0.0, 0.0064 - Q.mass_over_sum_pt_sq)
            + 0.125 * max(0.0, Q.pt_7 - 34.0)
            + 6.69 * max(0.0, Q.sj3_dr_max - 0.19)
            - 10.5 * max(0.0, Q.sj3_dr_min - 0.041)
            - 79.9 * max(0.0, 0.06 - Q.z_7)
            - 7.43 * max(0.0, Q.pt_7 - 33.0) * max(0.0, 0.017 - Q.tau2)
        ))
        - 0.5625 * grid(3, max(0.0, -5.26
            + 322.0 * max(0.0, Q.e2 - 0.064)
            + 97.1 * max(0.0, Q.girth - 0.042)
            + 97.3 * max(0.0, Q.girth - 0.078)
            - 1670.0 * max(0.0, Q.girth2 - 0.0089)
            - 872.0 * max(0.0, 0.0044 - Q.girth2)
            + 578.0 * max(0.0, Q.lam1 - 0.0081)
            - 0.0864 * max(0.0, Q.mass - 65.0)
            + 37.7 * max(0.0, Q.sj2_dr - 0.18)
            + 12.6 * max(0.0, Q.centroid_offset - 0.01) * max(0.0, 8.4 - Q.n_pt_above_50)
            + 113.0 * max(0.0, Q.girth - 0.038) * max(0.0, Q.log_sum_pt - 6.1)
            + 3750.0 * max(0.0, Q.lam1 - 0.016) * max(0.0, 6.0 - Q.n_for_90pct)
            - 4280.0 * max(0.0, Q.mass_over_sum_pt - 0.066) * max(0.0, 0.22 - Q.sj2_dr)
        ))
        + 0.078125 * grid(4, max(0.0, 3.48
            + 56.3 * max(0.0, 0.21 - Q.N2)
            - 57.5 * max(0.0, 0.056 - Q.centroid_offset)
            + 414.0 * max(0.0, 0.0082 - Q.girth2_top2)
            - 11000.0 * max(0.0, 0.00054 - Q.lam2)
            - 0.223 * max(0.0, Q.mass - 76.0)
            - 121.0 * max(0.0, Q.mass_over_sum_pt - 0.086)
            + 24.9 * max(0.0, Q.max_dr - 0.15)
            + 0.166 * max(0.0, Q.sd_mass - 43.0)
            - 22.7 * max(0.0, 0.22 - Q.sj3_dr_max)
            + 4080.0 * max(0.0, 0.0017 - Q.width)
            - 0.72 * max(0.0, 0.22 - Q.N2) * max(0.0, 53.0 - Q.pt_7)
            - 61500.0 * max(0.0, 0.0093 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.00095)
            - 0.966 * max(0.0, Q.sd_mass - 43.0) * max(0.0, 0.29 - Q.sd_zg)
            - 5930.0 * max(0.0, Q.width - 0.005) * max(0.0, 0.11 - Q.sj3_z3)
        ))
        + 0.015625 * grid(5, max(0.0, 0.0645
            + 443.0 * max(0.0, 0.0055 - Q.e2_sq)
            + 1050.0 * max(0.0, 0.0074 - Q.girth)
            - 12.8 * max(0.0, Q.log_sum_pt - 6.7)
            - 43.3 * max(0.0, Q.log_sum_pt - 6.9)
            + 220.0 * max(0.0, 0.029 - Q.z_7)
            + 57.7 * max(0.0, 0.057 - Q.z_7)
            - 186.0 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
            + 124000.0 * max(0.0, 0.0017 - Q.girth2) * max(0.0, 0.03 - Q.centroid_offset)
            + 0.0319 * max(0.0, Q.sum_pt_top5 - 350.0) * max(0.0, Q.sj2_dr - 0.098)
            - 2100.0 * max(0.0, 0.075 - Q.z_7) * max(0.0, 0.025 - Q.centroid_offset)
        ))
        - 0.375 * grid(6, max(0.0, 3.03
            + 150.0 * max(0.0, Q.centroid_offset - 0.0079)
            - 137.0 * max(0.0, Q.centroid_offset - 0.019)
            + 1150.0 * max(0.0, 0.0032 - Q.e2_sq)
            - 1620.0 * max(0.0, 0.0087 - Q.girth2)
            - 2270.0 * max(0.0, 0.0012 - Q.lam2)
            - 41.3 * max(0.0, 0.14 - Q.max_dr)
            + 9.18 * max(0.0, Q.sj3_dr_max - 0.18)
            + 46.0 * max(0.0, 0.18 - Q.sj3_dr_max)
            - 28.4 * max(0.0, Q.sj3_dr_min - 0.14)
            - 0.101 * max(0.0, Q.sj3_pair_mass_min - 4.0)
            + 71.3 * max(0.0, 0.11 - Q.tau1)
            + 96.0 * max(0.0, 0.0028 - Q.lam2) * max(0.0, 3.1 - Q.n_dr_0_0p05)
            - 0.798 * max(0.0, 64.0 - Q.mass) * max(0.0, 0.052 - Q.z_dr_0p2_0p4)
            + 1.02 * max(0.0, 41.0 - Q.pt_6) * max(0.0, 6.8 - Q.log_sum_pt)
            - 12.0 * max(0.0, 41.0 - Q.pt_6) * max(0.0, Q.z_7 - 0.023)
        ))
        + 0.46875 * grid(7, max(0.0, 6.92
            - 183.0 * max(0.0, Q.e2 - 0.05)
            - 38200.0 * max(0.0, 8.1e-05 - Q.e3)
            - 82.6 * max(0.0, 0.086 - Q.girth)
            + 1660.0 * max(0.0, Q.girth2 - 0.013)
            - 2720.0 * max(0.0, 0.001 - Q.girth2)
            + 0.0537 * max(0.0, Q.mass - 36.0)
            - 0.213 * max(0.0, Q.mass - 77.0)
            + 159.0 * max(0.0, Q.mass_over_sum_pt - 0.075)
            - 198.0 * max(0.0, Q.mass_over_sum_pt - 0.091)
            + 63.9 * max(0.0, 0.16 - Q.sj2_dr)
            - 45.0 * max(0.0, 0.19 - Q.sj2_dr)
            + 54.4 * max(0.0, 0.059 - Q.tau1)
            - 1330.0 * max(0.0, Q.width - 0.0056)
            + 1570.0 * max(0.0, Q.girth2 - 0.0032) * max(0.0, 0.21 - Q.planar_flow)
            - 2500.0 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.width - 0.008)
        ))
        - 0.03125 * grid(9, max(0.0, 0.0914
            - 81.1 * max(0.0, 0.028 - Q.C3)
            + 173.0 * max(0.0, 0.019 - Q.centroid_offset)
            + 100.0 * max(0.0, Q.e2 - 0.047)
            - 68.5 * max(0.0, Q.girth - 0.069)
            + 1380.0 * max(0.0, 0.0035 - Q.girth2)
            + 786.0 * max(0.0, 0.0059 - Q.girth2)
            + 784.0 * max(0.0, Q.lam2 - 0.00083)
            - 7.62 * max(0.0, Q.log_sum_pt - 6.4)
            + 9.64 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.102 * max(0.0, 55.0 - Q.mass)
            + 55.8 * max(0.0, 0.11 - Q.max_dr)
            - 74.1 * max(0.0, 0.14 - Q.sj3_dr_max)
            + 19.8 * max(0.0, 0.21 - Q.sj3_dr_max)
            + 0.0224 * max(0.0, 570.0 - Q.sum_pt)
            - 65.5 * max(0.0, 0.02 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.5)
            + 627.0 * max(0.0, 0.0086 - Q.girth2) * max(0.0, 0.33 - Q.planar_flow)
            + 6.06 * max(0.0, 54.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        ))
        + 0.0546875 * grid(13, max(0.0, 0.0376
            - 41.0 * max(0.0, 0.08 - Q.e2)
            + 52.6 * max(0.0, 0.15 - Q.girth)
            + 245.0 * max(0.0, 0.016 - Q.lam1)
            - 306.0 * max(0.0, 0.028 - Q.z_7)
            + 9000.0 * max(0.0, 5.8e-05 - Q.e3) * max(0.0, Q.D3 - 0.16)
            - 799000.0 * max(0.0, 6.1e-05 - Q.e3) * max(0.0, 0.038 - Q.centroid_offset)
            - 50.1 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
            - 1.48 * max(0.0, 0.15 - Q.girth) * max(0.0, 38.0 - Q.pt_7)
            - 0.000625 * max(0.0, Q.sum_pt - 990.0) * max(0.0, 62.0 - Q.pt_6)
            + 0.0011 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 41.0 - Q.pt_7)
        ))
        + 0.375 * grid(14, max(0.0, -1.75
            - 107.0 * max(0.0, Q.centroid_offset - 0.026)
            - 1110.0 * max(0.0, Q.centroid_offset - 0.05)
            - 218.0 * max(0.0, Q.e2 - 0.017)
            + 138.0 * max(0.0, Q.e2 - 0.042)
            + 373.0 * max(0.0, Q.e2 - 0.05)
            - 25100.0 * max(0.0, 5.1e-05 - Q.e3)
            + 109.0 * max(0.0, Q.girth - 0.027)
            + 65.0 * max(0.0, Q.girth - 0.042)
            - 257.0 * max(0.0, Q.girth - 0.089)
            - 47.4 * max(0.0, Q.girth - 0.094)
            - 2990.0 * max(0.0, Q.girth2 - 0.0075)
            + 327.0 * max(0.0, Q.mass_over_sum_pt - 0.085)
            + 0.0576 * max(0.0, 50.0 - Q.sd_mass)
            - 0.0523 * max(0.0, 75.0 - Q.sd_mass)
            + 41.1 * max(0.0, Q.sj2_dr - 0.16)
            - 20.0 * max(0.0, Q.sj2_dr - 0.2)
            + 270.0 * max(0.0, Q.width - 0.0033)
            + 896.0 * max(0.0, Q.width - 0.014)
            - 792.0 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 0.018 - Q.centroid_offset)
            - 9930.0 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.0072 - Q.lam1)
            + 1300.0 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.016 - Q.lam1)
            + 9130.0 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.0061 - Q.width)
        ))
        - 0.15625 * grid(15, max(0.0, -0.101
            + 203.0 * max(0.0, 0.04 - Q.e2)
            - 97.9 * max(0.0, 0.1 - Q.girth)
            - 1050.0 * max(0.0, 0.008 - Q.girth2)
            - 6550.0 * max(0.0, 0.0005 - Q.girth2_top2)
            + 48.5 * max(0.0, 0.16 - Q.sj2_dr)
            - 45.1 * max(0.0, 0.2 - Q.sj2_dr)
            + 560.0 * max(0.0, 0.014 - Q.width)
            + 56.4 * max(0.0, 0.15 - Q.mass_over_sum_pt) * max(0.0, 0.73 - Q.D2)
            - 7420.0 * max(0.0, 0.0061 - Q.width) * max(0.0, 0.72 - Q.D2)
        ))
    )


def logit_t(Q):
    return (1.34375
        + 0.015625 * grid(0, max(0.0, -0.0143
            - 1510.0 * max(0.0, 0.0016 - Q.C2_b2)
            + 197.0 * max(0.0, 0.025 - Q.e2)
            - 60.8 * max(0.0, 0.088 - Q.girth)
            - 17.3 * max(0.0, Q.log_sum_pt - 6.7)
            + 0.17 * max(0.0, 18.0 - Q.mass)
            - 0.0397 * max(0.0, 56.0 - Q.mass)
            + 0.338 * max(0.0, Q.n_dr_0_0p05 - 4.0)
            + 16.3 * max(0.0, 0.15 - Q.planar_flow)
            - 27.8 * max(0.0, Q.sj3_dr_max - 0.23)
            - 0.0179 * max(0.0, Q.sum_pt - 900.0)
            + 0.0163 * max(0.0, Q.sum_pt_top5 - 690.0)
            - 2220.0 * max(0.0, 0.0044 - Q.width)
            + 1080.0 * max(0.0, 0.0088 - Q.width)
            + 455.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, 1.0 - Q.D2)
            - 12700.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018)
            - 3060.0 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.87 - Q.D2)
            + 75500.0 * max(0.0, 7.1e-05 - Q.lam2) * max(0.0, 0.28 - Q.D2_b2)
        ))
        + 0.0625 * grid(3, max(0.0, -5.26
            + 322.0 * max(0.0, Q.e2 - 0.064)
            + 97.1 * max(0.0, Q.girth - 0.042)
            + 97.3 * max(0.0, Q.girth - 0.078)
            - 1670.0 * max(0.0, Q.girth2 - 0.0089)
            - 872.0 * max(0.0, 0.0044 - Q.girth2)
            + 578.0 * max(0.0, Q.lam1 - 0.0081)
            - 0.0864 * max(0.0, Q.mass - 65.0)
            + 37.7 * max(0.0, Q.sj2_dr - 0.18)
            + 12.6 * max(0.0, Q.centroid_offset - 0.01) * max(0.0, 8.4 - Q.n_pt_above_50)
            + 113.0 * max(0.0, Q.girth - 0.038) * max(0.0, Q.log_sum_pt - 6.1)
            + 3750.0 * max(0.0, Q.lam1 - 0.016) * max(0.0, 6.0 - Q.n_for_90pct)
            - 4280.0 * max(0.0, Q.mass_over_sum_pt - 0.066) * max(0.0, 0.22 - Q.sj2_dr)
        ))
        + 0.125 * grid(4, max(0.0, 3.48
            + 56.3 * max(0.0, 0.21 - Q.N2)
            - 57.5 * max(0.0, 0.056 - Q.centroid_offset)
            + 414.0 * max(0.0, 0.0082 - Q.girth2_top2)
            - 11000.0 * max(0.0, 0.00054 - Q.lam2)
            - 0.223 * max(0.0, Q.mass - 76.0)
            - 121.0 * max(0.0, Q.mass_over_sum_pt - 0.086)
            + 24.9 * max(0.0, Q.max_dr - 0.15)
            + 0.166 * max(0.0, Q.sd_mass - 43.0)
            - 22.7 * max(0.0, 0.22 - Q.sj3_dr_max)
            + 4080.0 * max(0.0, 0.0017 - Q.width)
            - 0.72 * max(0.0, 0.22 - Q.N2) * max(0.0, 53.0 - Q.pt_7)
            - 61500.0 * max(0.0, 0.0093 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.00095)
            - 0.966 * max(0.0, Q.sd_mass - 43.0) * max(0.0, 0.29 - Q.sd_zg)
            - 5930.0 * max(0.0, Q.width - 0.005) * max(0.0, 0.11 - Q.sj3_z3)
        ))
        - 0.25 * grid(5, max(0.0, 0.0645
            + 443.0 * max(0.0, 0.0055 - Q.e2_sq)
            + 1050.0 * max(0.0, 0.0074 - Q.girth)
            - 12.8 * max(0.0, Q.log_sum_pt - 6.7)
            - 43.3 * max(0.0, Q.log_sum_pt - 6.9)
            + 220.0 * max(0.0, 0.029 - Q.z_7)
            + 57.7 * max(0.0, 0.057 - Q.z_7)
            - 186.0 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
            + 124000.0 * max(0.0, 0.0017 - Q.girth2) * max(0.0, 0.03 - Q.centroid_offset)
            + 0.0319 * max(0.0, Q.sum_pt_top5 - 350.0) * max(0.0, Q.sj2_dr - 0.098)
            - 2100.0 * max(0.0, 0.075 - Q.z_7) * max(0.0, 0.025 - Q.centroid_offset)
        ))
        + 0.1875 * grid(8, max(0.0, -0.681
            - 60.8 * max(0.0, 0.21 - Q.LHA)
            - 5.19 * max(0.0, Q.log_sum_pt - 6.7)
            - 22.7 * max(0.0, 0.13 - Q.sj3_dr_max)
            + 48600.0 * max(0.0, 0.0063 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
            + 26900.0 * max(0.0, 0.058 - Q.tau1) * max(0.0, 0.0032 - Q.width)
        ))
        + 0.375 * grid(10, max(0.0, 0.34
            + 128.0 * max(0.0, Q.C2_b2 - 0.0093)
            - 33.2 * max(0.0, Q.LHA - 0.3)
            - 5610.0 * max(0.0, Q.e3 - 7.5e-05)
            + 25100.0 * max(0.0, 8e-05 - Q.e3)
            + 340.0 * max(0.0, Q.girth2 - 0.0075)
            - 1920.0 * max(0.0, 0.0015 - Q.lam1)
            - 529.0 * max(0.0, 0.0056 - Q.lam1)
            + 871.0 * max(0.0, Q.lam2 - 0.00034)
            - 643.0 * max(0.0, Q.lam2 - 0.0034)
            + 0.03 * max(0.0, 75.0 - Q.mass)
            - 0.0398 * max(0.0, 46.0 - Q.pt_7)
            + 0.101 * max(0.0, Q.sj3_pair_mass_min - 11.0)
            + 28.0 * max(0.0, Q.tau1 - 0.057)
            - 141000.0 * max(0.0, 8.1e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.18)
            - 554.0 * max(0.0, Q.lam1 - 0.0074) * max(0.0, 0.38 - Q.D2_b2)
            + 0.129 * max(0.0, 45.0 - Q.pt_7) * max(0.0, 0.98 - Q.D2)
            + 204.0 * max(0.0, 0.022 - Q.zdr_0) * max(0.0, 0.24 - Q.z_dr_0p05_0p1)
        ))
        - 0.5 * grid(12, max(0.0, -1.37
            - 79.3 * max(0.0, Q.e2 - 0.063)
            + 347.0 * max(0.0, Q.girth2 - 0.019)
            + 0.0953 * max(0.0, Q.mass - 91.0)
        ))
        - 0.40625 * grid(13, max(0.0, 0.0376
            - 41.0 * max(0.0, 0.08 - Q.e2)
            + 52.6 * max(0.0, 0.15 - Q.girth)
            + 245.0 * max(0.0, 0.016 - Q.lam1)
            - 306.0 * max(0.0, 0.028 - Q.z_7)
            + 9000.0 * max(0.0, 5.8e-05 - Q.e3) * max(0.0, Q.D3 - 0.16)
            - 799000.0 * max(0.0, 6.1e-05 - Q.e3) * max(0.0, 0.038 - Q.centroid_offset)
            - 50.1 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
            - 1.48 * max(0.0, 0.15 - Q.girth) * max(0.0, 38.0 - Q.pt_7)
            - 0.000625 * max(0.0, Q.sum_pt - 990.0) * max(0.0, 62.0 - Q.pt_6)
            + 0.0011 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 41.0 - Q.pt_7)
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
