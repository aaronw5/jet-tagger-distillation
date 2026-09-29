"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; all observables), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

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


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-0.4375
        - 0.15625 * grid(0, max(0.0, -2.547365
            + 17.12106 * max(0.0, 0.1484197 - Q.planar_flow)
            - 2100.026 * max(0.0, 0.004372139 - Q.width)
            + 0.2586946 * max(0.0, 21.78408 - Q.mass)
            + 290.5179 * max(0.0, 0.01323868 - Q.girth2)
            + 335.9675 * max(0.0, 0.006506576 - Q.lam1)
            - 3140.132 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2)
            - 0.02317622 * max(0.0, Q.sum_pt - 901.5938)
            + 435.0238 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 1.002471 - Q.D2)
            - 0.04211574 * max(0.0, 56.92035 - Q.mass)
            + 594.8094 * max(0.0, 0.008678045 - Q.width)
            - 1656.548 * max(0.0, 0.001563465 - Q.C2_b2)
            - 27.81783 * max(0.0, Q.sj3_dr_max - 0.233678)
            + 4.358228 * max(0.0, Q.sj3_dr_max - 0.1070199)
            + 10.18314 * max(0.0, 0.3012016 - Q.sj3_dr_max)
            - 0.1516387 * max(0.0, 29.6447 - Q.mass)
            - 12398.28 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.centroid_offset - 0.01837778)
            - 47.34599 * max(0.0, 0.08723651 - Q.girth)
            + 228.7248 * max(0.0, 0.0245477 - Q.e2)
            - 423.6481 * max(0.0, 0.005433361 - Q.lam1)
            + 0.360254 * max(0.0, Q.n_dr_0_0p05 - 4.0)
            - 14.56131 * max(0.0, Q.log_sum_pt - 6.670067)
            + 0.01695238 * max(0.0, Q.sum_pt_top5 - 687.4375)
            + 84155.38 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2)
        ))
        + 0.390625 * grid(1, max(0.0, -0.1230816
            + 0.1324637 * max(0.0, Q.pt_7 - 34.53125)
            + 4.910623 * max(0.0, Q.log_sum_pt - 6.377723)
            + 892.9637 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq)
            - 75.00406 * max(0.0, 0.06164517 - Q.z_7)
            - 584.8421 * max(0.0, 0.008678045 - Q.girth2)
            - 8.175155 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2)
            - 2.146648 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1)
            - 10.67569 * max(0.0, Q.sj3_dr_min - 0.03628191)
            + 7.170598 * max(0.0, Q.sj3_dr_max - 0.169029)
            + 4.052212 * max(0.0, Q.log_sum_pt - 6.605974)
            + 277.1018 * max(0.0, 0.008168571 - Q.e2_sq)
            - 0.005513285 * max(0.0, Q.sum_pt_top5 - 531.1875)
            + 519.7603 * max(0.0, 0.00595415 - Q.lam1)
            - 575.5693 * max(0.0, 0.00609665 - Q.width)
            + 5.616011 * max(0.0, Q.log_sum_pt - 6.502799)
            + 0.005023588 * max(0.0, Q.sum_pt_top5 - 367.5938)
            - 7969.009 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236)
            - 64.09914 * max(0.0, 0.0717028 - Q.girth)
        ))
        + 0.4296875 * grid(2, max(0.0, 1.713223
            + 2.684133 * max(0.0, 6.46415 - Q.log_sum_pt)
            + 311.5759 * max(0.0, 0.00595415 - Q.lam1)
            - 8.787485 * max(0.0, Q.LHA - 0.1329373)
            - 0.05292109 * max(0.0, 53.4375 - Q.pt_7)
            - 0.04941354 * max(0.0, 36.22941 - Q.mass)
            + 9.748331 * max(0.0, Q.N2 - 0.1150852)
            + 0.005845624 * max(0.0, 813.4156 - Q.sum_pt)
            + 0.03575489 * max(0.0, 53.4375 - Q.pt_7) * max(0.0, 1.129616 - Q.D2_b2)
            - 0.007983031 * max(0.0, Q.sum_pt_top5 - 752.1)
        ))
        - 0.03125 * grid(4, max(0.0, -1.984515
            + 61.9689 * max(0.0, 0.2233283 - Q.N2)
            - 0.7834277 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.55 - Q.mass)
            - 10570.73 * max(0.0, 0.000537286 - Q.lam2)
            - 149.41 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            + 376.1682 * max(0.0, Q.width - 0.003562611)
            + 73.52333 * max(0.0, Q.e2 - 0.04447357)
            - 0.003755188 * max(0.0, 739.5 - Q.sum_pt)
            - 0.7519606 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7)
            - 73.94671 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266)
            - 1.867846 * max(0.0, 0.177305 - Q.max_dr)
            + 79.57784 * max(0.0, 0.06729223 - Q.C2)
            - 42.64309 * max(0.0, Q.girth - 0.08723651)
            + 0.1290954 * max(0.0, Q.sd_mass - 44.82259)
            + 0.09193323 * max(0.0, Q.sd_mass - 74.57663)
            + 14224.94 * max(0.0, 0.0001869378 - Q.e3)
            + 5.664068 * max(0.0, 0.3456459 - Q.sj3_dr_max)
            + 30.89943 * max(0.0, Q.max_dr - 0.1598486)
            - 197.5795 * max(0.0, 0.008678045 - Q.girth2)
            - 1.730639 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.centroid_offset - 0.001308549)
            - 73.89394 * max(0.0, 0.04990367 - Q.centroid_offset)
            + 390.8946 * max(0.0, 0.007639643 - Q.girth2_top2)
            - 0.9066542 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.275762 - Q.sd_zg)
            - 5836.058 * max(0.0, Q.width - 0.003562611) * max(0.0, 0.1057566 - Q.sj3_z3)
            - 0.2999585 * max(0.0, Q.mass - 76.6557)
            + 3246.117 * max(0.0, 0.001653836 - Q.width)
            + 3214.91 * max(0.0, 0.01165737 - Q.e2_sq)
            - 85.26767 * max(0.0, 0.05028464 - Q.e2)
            - 56676.59 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.0006435798)
            - 41.2037 * max(0.0, 0.233678 - Q.sj3_dr_max)
            - 2661.375 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq)
            + 8.220409 * max(0.0, 0.3467135 - Q.LHA)
        ))
        - 0.1875 * grid(5, max(0.0, 0.7058495
            + 17.35268 * max(0.0, 0.2160559 - Q.LHA)
            + 79.62085 * max(0.0, 0.04939969 - Q.z_7)
            - 159.9139 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 58.21803 * max(0.0, Q.log_sum_pt - 6.896095)
            + 643.0026 * max(0.0, 0.005284669 - Q.e2_sq)
            - 13172.7 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, 0.01437952 - Q.centroid_offset)
            - 1771.377 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset)
            + 1925.216 * max(0.0, 0.001653836 - Q.girth2)
            + 1540.288 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.05744392 - Q.D2_b2)
            + 0.5976955 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 40.2 - Q.mass_top3)
            + 996.9643 * max(0.0, 0.007673833 - Q.girth)
            - 0.0638411 * max(0.0, 37.15625 - Q.pt_7)
            + 151167.2 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            + 240.1729 * max(0.0, 0.02807091 - Q.z_7)
            - 13584.06 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.001503553 - Q.lam1)
            - 154.6689 * max(0.0, 0.0211821 - Q.zdr_0)
            - 18.88884 * max(0.0, Q.log_sum_pt - 6.701242)
            + 1026.925 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01341502 - Q.dr_2)
            + 0.0001369071 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 28.39396 - Q.pt1_dr01)
            + 0.03659241 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.sj2_dr - 0.1682655)
            + 385.7445 * max(0.0, 0.003562611 - Q.width)
            - 908.0424 * max(0.0, 0.002074109 - Q.e2_sq)
            - 615.4584 * max(0.0, 0.002151568 - Q.girth2_top3)
            + 693.5255 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.sj3_dr13 - 0.04995258)
        ))
        + 0.109375 * grid(6, max(0.0, 0.5173415
            + 153.643 * max(0.0, Q.centroid_offset - 0.00809236)
            - 354.7936 * max(0.0, 0.01323868 - Q.width)
            + 2.134053 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308)
            + 0.07841492 * max(0.0, 41.21875 - Q.pt_6)
            + 1.196886 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt)
            + 53.59186 * max(0.0, 0.1136369 - Q.tau1)
            - 17.47437 * max(0.0, Q.sj3_dr_min - 0.1278212)
            - 181.4227 * max(0.0, Q.centroid_offset - 0.01837778)
            + 294.5117 * max(0.0, 0.01200373 - Q.lam1)
            + 8726.583 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2)
            - 0.09517011 * max(0.0, Q.sj3_pair_mass_min - 4.501727)
            - 13.67302 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757)
            + 1311.96 * max(0.0, 0.0030133 - Q.e2_sq)
            - 1879.342 * max(0.0, 0.001130645 - Q.lam2)
            - 1.093805 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4)
            + 43.21113 * max(0.0, 0.1789613 - Q.sj3_dr_max)
            - 41.4775 * max(0.0, 0.1452311 - Q.max_dr)
            + 88.03266 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05)
            + 5599.613 * max(0.0, 0.0005116989 - Q.e3)
            + 0.02568978 * max(0.0, 45.595 - Q.mass)
            - 1487.827 * max(0.0, 0.008678045 - Q.girth2)
            + 140.4956 * max(0.0, 0.00733008 - Q.lam1)
            - 0.09862539 * max(0.0, 39.75 - Q.pt_6)
            + 13.03165 * max(0.0, Q.sj3_dr_max - 0.1879486)
            - 0.0335275 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        ))
        + 0.171875 * grid(9, max(0.0, -3.212293
            - 57.93995 * max(0.0, 0.05464922 - Q.girth)
            + 97.89552 * max(0.0, 0.04369778 - Q.tau1)
            - 0.09790827 * max(0.0, 53.33237 - Q.mass)
            - 10657.71 * max(0.0, Q.e3 - 0.0001869378)
            + 6.756995 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset)
            + 1557.023 * max(0.0, 0.00609665 - Q.girth2)
            + 18.83085 * max(0.0, 0.213399 - Q.sj3_dr_max)
            + 551.8829 * max(0.0, Q.lam2 - 0.001130645)
            + 0.06960289 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt)
            + 2131.607 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.3220738 - Q.planar_flow)
            + 1.028209 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
            - 58.36824 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            - 229.5294 * max(0.0, Q.lam1 - 0.003377388)
            + 265.1134 * max(0.0, 0.01837778 - Q.centroid_offset)
            - 82.83437 * max(0.0, Q.e2 - 0.03556091)
            - 0.08341188 * max(0.0, 24.23013 - Q.sj3_pair_mass_max)
            + 565.524 * max(0.0, Q.lam1 - 0.005433361)
            - 69.61066 * max(0.0, 0.0284695 - Q.C3)
            - 802.3769 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266)
            - 866.67 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255)
            - 83266.02 * max(0.0, 2.371297e-05 - Q.e3)
            - 269.305 * max(0.0, 0.006292091 - Q.zdr_0)
            + 2354.024 * max(0.0, 0.003562611 - Q.girth2)
            - 53.34438 * max(0.0, 0.07269073 - Q.mass_over_sum_pt)
            + 0.009028949 * max(0.0, 988.4078 - Q.sum_pt)
            + 169.3854 * max(0.0, Q.e2 - 0.04447357)
            - 815.4236 * max(0.0, 0.00595415 - Q.lam1)
            - 8.507298 * Q.z_dr_0p2_0p4
            + 20.67153 * max(0.0, Q.log_sum_pt - 6.896095)
            - 55.42017 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2)
            - 0.01073602 * max(0.0, Q.sum_pt_top5 - 839.9547)
            - 56.22345 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0)
            - 2.795783 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1)
            + 1587.651 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd)
            + 0.01083795 * max(0.0, 559.6875 - Q.sum_pt)
            - 93.23307 * max(0.0, Q.girth - 0.0717028)
            + 10562.2 * max(0.0, Q.e3 - 8.147744e-05)
            + 47.1219 * max(0.0, 0.1117619 - Q.max_dr)
        ))
    )


def logit_q(Q):
    return (0.03125
        - 0.09375 * grid(4, max(0.0, -1.984515
            + 61.9689 * max(0.0, 0.2233283 - Q.N2)
            - 0.7834277 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.55 - Q.mass)
            - 10570.73 * max(0.0, 0.000537286 - Q.lam2)
            - 149.41 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            + 376.1682 * max(0.0, Q.width - 0.003562611)
            + 73.52333 * max(0.0, Q.e2 - 0.04447357)
            - 0.003755188 * max(0.0, 739.5 - Q.sum_pt)
            - 0.7519606 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7)
            - 73.94671 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266)
            - 1.867846 * max(0.0, 0.177305 - Q.max_dr)
            + 79.57784 * max(0.0, 0.06729223 - Q.C2)
            - 42.64309 * max(0.0, Q.girth - 0.08723651)
            + 0.1290954 * max(0.0, Q.sd_mass - 44.82259)
            + 0.09193323 * max(0.0, Q.sd_mass - 74.57663)
            + 14224.94 * max(0.0, 0.0001869378 - Q.e3)
            + 5.664068 * max(0.0, 0.3456459 - Q.sj3_dr_max)
            + 30.89943 * max(0.0, Q.max_dr - 0.1598486)
            - 197.5795 * max(0.0, 0.008678045 - Q.girth2)
            - 1.730639 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.centroid_offset - 0.001308549)
            - 73.89394 * max(0.0, 0.04990367 - Q.centroid_offset)
            + 390.8946 * max(0.0, 0.007639643 - Q.girth2_top2)
            - 0.9066542 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.275762 - Q.sd_zg)
            - 5836.058 * max(0.0, Q.width - 0.003562611) * max(0.0, 0.1057566 - Q.sj3_z3)
            - 0.2999585 * max(0.0, Q.mass - 76.6557)
            + 3246.117 * max(0.0, 0.001653836 - Q.width)
            + 3214.91 * max(0.0, 0.01165737 - Q.e2_sq)
            - 85.26767 * max(0.0, 0.05028464 - Q.e2)
            - 56676.59 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.0006435798)
            - 41.2037 * max(0.0, 0.233678 - Q.sj3_dr_max)
            - 2661.375 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq)
            + 8.220409 * max(0.0, 0.3467135 - Q.LHA)
        ))
        + 0.046875 * grid(5, max(0.0, 0.7058495
            + 17.35268 * max(0.0, 0.2160559 - Q.LHA)
            + 79.62085 * max(0.0, 0.04939969 - Q.z_7)
            - 159.9139 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 58.21803 * max(0.0, Q.log_sum_pt - 6.896095)
            + 643.0026 * max(0.0, 0.005284669 - Q.e2_sq)
            - 13172.7 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, 0.01437952 - Q.centroid_offset)
            - 1771.377 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset)
            + 1925.216 * max(0.0, 0.001653836 - Q.girth2)
            + 1540.288 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.05744392 - Q.D2_b2)
            + 0.5976955 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 40.2 - Q.mass_top3)
            + 996.9643 * max(0.0, 0.007673833 - Q.girth)
            - 0.0638411 * max(0.0, 37.15625 - Q.pt_7)
            + 151167.2 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            + 240.1729 * max(0.0, 0.02807091 - Q.z_7)
            - 13584.06 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.001503553 - Q.lam1)
            - 154.6689 * max(0.0, 0.0211821 - Q.zdr_0)
            - 18.88884 * max(0.0, Q.log_sum_pt - 6.701242)
            + 1026.925 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01341502 - Q.dr_2)
            + 0.0001369071 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 28.39396 - Q.pt1_dr01)
            + 0.03659241 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.sj2_dr - 0.1682655)
            + 385.7445 * max(0.0, 0.003562611 - Q.width)
            - 908.0424 * max(0.0, 0.002074109 - Q.e2_sq)
            - 615.4584 * max(0.0, 0.002151568 - Q.girth2_top3)
            + 693.5255 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.sj3_dr13 - 0.04995258)
        ))
        + 0.125 * grid(6, max(0.0, 0.5173415
            + 153.643 * max(0.0, Q.centroid_offset - 0.00809236)
            - 354.7936 * max(0.0, 0.01323868 - Q.width)
            + 2.134053 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308)
            + 0.07841492 * max(0.0, 41.21875 - Q.pt_6)
            + 1.196886 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt)
            + 53.59186 * max(0.0, 0.1136369 - Q.tau1)
            - 17.47437 * max(0.0, Q.sj3_dr_min - 0.1278212)
            - 181.4227 * max(0.0, Q.centroid_offset - 0.01837778)
            + 294.5117 * max(0.0, 0.01200373 - Q.lam1)
            + 8726.583 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2)
            - 0.09517011 * max(0.0, Q.sj3_pair_mass_min - 4.501727)
            - 13.67302 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757)
            + 1311.96 * max(0.0, 0.0030133 - Q.e2_sq)
            - 1879.342 * max(0.0, 0.001130645 - Q.lam2)
            - 1.093805 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4)
            + 43.21113 * max(0.0, 0.1789613 - Q.sj3_dr_max)
            - 41.4775 * max(0.0, 0.1452311 - Q.max_dr)
            + 88.03266 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05)
            + 5599.613 * max(0.0, 0.0005116989 - Q.e3)
            + 0.02568978 * max(0.0, 45.595 - Q.mass)
            - 1487.827 * max(0.0, 0.008678045 - Q.girth2)
            + 140.4956 * max(0.0, 0.00733008 - Q.lam1)
            - 0.09862539 * max(0.0, 39.75 - Q.pt_6)
            + 13.03165 * max(0.0, Q.sj3_dr_max - 0.1879486)
            - 0.0335275 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        ))
        + 0.0625 * grid(8, max(0.0, -0.5655652
            + 284.2185 * max(0.0, 0.005019719 - Q.girth2)
            + 31808.03 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            - 57.23899 * max(0.0, 0.1967397 - Q.LHA)
            - 16.61397 * max(0.0, Q.log_sum_pt - 6.701242)
            - 66.44382 * max(0.0, 0.06108601 - Q.girth)
            + 0.009398292 * max(0.0, Q.sum_pt_top5 - 687.4375)
            + 4965.175 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4)
            - 336.3577 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4)
            + 236714.0 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2)
            - 4.433283 * max(0.0, Q.z_dr_0_0p05 - 0.8477313)
            + 233.7561 * max(0.0, 0.003562611 - Q.width)
            - 0.003727224 * max(0.0, 21.78408 - Q.mass) * max(0.0, 48.71875 - Q.pt_7)
            - 25643.27 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738)
            + 118.4842 * max(0.0, 0.03319429 - Q.mass_over_sum_pt)
            - 27.84765 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            - 41.76267 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.003562611 - Q.width)
            + 1633.608 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.girth2)
            + 21672.79 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.width)
        ))
        + 0.2539062 * grid(9, max(0.0, -3.212293
            - 57.93995 * max(0.0, 0.05464922 - Q.girth)
            + 97.89552 * max(0.0, 0.04369778 - Q.tau1)
            - 0.09790827 * max(0.0, 53.33237 - Q.mass)
            - 10657.71 * max(0.0, Q.e3 - 0.0001869378)
            + 6.756995 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset)
            + 1557.023 * max(0.0, 0.00609665 - Q.girth2)
            + 18.83085 * max(0.0, 0.213399 - Q.sj3_dr_max)
            + 551.8829 * max(0.0, Q.lam2 - 0.001130645)
            + 0.06960289 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt)
            + 2131.607 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.3220738 - Q.planar_flow)
            + 1.028209 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
            - 58.36824 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            - 229.5294 * max(0.0, Q.lam1 - 0.003377388)
            + 265.1134 * max(0.0, 0.01837778 - Q.centroid_offset)
            - 82.83437 * max(0.0, Q.e2 - 0.03556091)
            - 0.08341188 * max(0.0, 24.23013 - Q.sj3_pair_mass_max)
            + 565.524 * max(0.0, Q.lam1 - 0.005433361)
            - 69.61066 * max(0.0, 0.0284695 - Q.C3)
            - 802.3769 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266)
            - 866.67 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255)
            - 83266.02 * max(0.0, 2.371297e-05 - Q.e3)
            - 269.305 * max(0.0, 0.006292091 - Q.zdr_0)
            + 2354.024 * max(0.0, 0.003562611 - Q.girth2)
            - 53.34438 * max(0.0, 0.07269073 - Q.mass_over_sum_pt)
            + 0.009028949 * max(0.0, 988.4078 - Q.sum_pt)
            + 169.3854 * max(0.0, Q.e2 - 0.04447357)
            - 815.4236 * max(0.0, 0.00595415 - Q.lam1)
            - 8.507298 * Q.z_dr_0p2_0p4
            + 20.67153 * max(0.0, Q.log_sum_pt - 6.896095)
            - 55.42017 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2)
            - 0.01073602 * max(0.0, Q.sum_pt_top5 - 839.9547)
            - 56.22345 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0)
            - 2.795783 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1)
            + 1587.651 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd)
            + 0.01083795 * max(0.0, 559.6875 - Q.sum_pt)
            - 93.23307 * max(0.0, Q.girth - 0.0717028)
            + 10562.2 * max(0.0, Q.e3 - 8.147744e-05)
            + 47.1219 * max(0.0, 0.1117619 - Q.max_dr)
        ))
        - 0.125 * grid(10, max(0.0, 1.525997
            - 36.4558 * Q.e2
            + 0.1035518 * max(0.0, Q.sj3_pair_mass_min - 11.051)
            - 331.5034 * max(0.0, 0.004183811 - Q.lam1)
            + 746.1591 * max(0.0, Q.lam2 - 0.0003061234)
            - 0.04274978 * max(0.0, 45.75 - Q.pt_7)
            + 17558.84 * max(0.0, 8.147744e-05 - Q.e3)
            - 696.6071 * max(0.0, Q.lam2 - 0.003408389)
            + 0.1273066 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.002471 - Q.D2)
            - 313.4167 * max(0.0, 0.00595415 - Q.lam1)
            - 1928.105 * max(0.0, 0.001503553 - Q.lam1)
            + 182.2886 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1)
            - 34.71547 * max(0.0, Q.LHA - 0.3033137)
            + 39.6598 * max(0.0, Q.tau1 - 0.05356915)
            + 487.1833 * max(0.0, Q.girth2 - 0.007520088)
            - 149.677 * max(0.0, Q.lam1 - 0.00733008)
            - 146055.7 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1797097)
            - 524.1245 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2)
            - 4762.144 * max(0.0, Q.e3 - 3.892127e-05)
            + 0.02752135 * max(0.0, 76.6557 - Q.mass)
            + 126.0312 * max(0.0, Q.C2_b2 - 0.009032972)
        ))
        + 0.0625 * grid(15, max(0.0, -0.3878572
            - 859.4324 * max(0.0, 0.007520088 - Q.girth2)
            - 30.25677 * max(0.0, 0.1309286 - Q.mass_over_sum_pt)
            - 367.6759 * max(0.0, 0.00609665 - Q.width)
            + 717.7877 * max(0.0, 0.01323868 - Q.width)
            - 1700.759 * max(0.0, 0.007520088 - Q.girth2) * max(0.0, 0.7459513 - Q.D2)
            - 4818.035 * max(0.0, 0.00609665 - Q.width) * max(0.0, 0.7459513 - Q.D2)
            + 100.9499 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2)
            + 230.7089 * max(0.0, 0.04110972 - Q.e2)
            - 319.5466 * max(0.0, 0.01165737 - Q.e2_sq)
            + 942.3414 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq)
            - 754.7946 * max(0.0, 0.008168571 - Q.e2_sq)
            + 158.8951 * max(0.0, 0.00483998 - Q.lam1)
            + 241.8992 * max(0.0, 0.01882765 - Q.width)
            - 111.7434 * max(0.0, 0.1019409 - Q.girth)
            + 56.6489 * max(0.0, 0.1591713 - Q.sj2_dr)
            - 36.55215 * max(0.0, 0.2001708 - Q.sj2_dr)
            - 13.52371 * max(0.0, 0.1492731 - Q.sj2_dr)
            - 4810.125 * max(0.0, 0.0005124533 - Q.girth2_top2)
        ))
    )


def logit_W(Q):
    return (-0.125
        + 0.34375 * grid(0, max(0.0, -2.547365
            + 17.12106 * max(0.0, 0.1484197 - Q.planar_flow)
            - 2100.026 * max(0.0, 0.004372139 - Q.width)
            + 0.2586946 * max(0.0, 21.78408 - Q.mass)
            + 290.5179 * max(0.0, 0.01323868 - Q.girth2)
            + 335.9675 * max(0.0, 0.006506576 - Q.lam1)
            - 3140.132 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2)
            - 0.02317622 * max(0.0, Q.sum_pt - 901.5938)
            + 435.0238 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 1.002471 - Q.D2)
            - 0.04211574 * max(0.0, 56.92035 - Q.mass)
            + 594.8094 * max(0.0, 0.008678045 - Q.width)
            - 1656.548 * max(0.0, 0.001563465 - Q.C2_b2)
            - 27.81783 * max(0.0, Q.sj3_dr_max - 0.233678)
            + 4.358228 * max(0.0, Q.sj3_dr_max - 0.1070199)
            + 10.18314 * max(0.0, 0.3012016 - Q.sj3_dr_max)
            - 0.1516387 * max(0.0, 29.6447 - Q.mass)
            - 12398.28 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.centroid_offset - 0.01837778)
            - 47.34599 * max(0.0, 0.08723651 - Q.girth)
            + 228.7248 * max(0.0, 0.0245477 - Q.e2)
            - 423.6481 * max(0.0, 0.005433361 - Q.lam1)
            + 0.360254 * max(0.0, Q.n_dr_0_0p05 - 4.0)
            - 14.56131 * max(0.0, Q.log_sum_pt - 6.670067)
            + 0.01695238 * max(0.0, Q.sum_pt_top5 - 687.4375)
            + 84155.38 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2)
        ))
        - 0.03125 * grid(1, max(0.0, -0.1230816
            + 0.1324637 * max(0.0, Q.pt_7 - 34.53125)
            + 4.910623 * max(0.0, Q.log_sum_pt - 6.377723)
            + 892.9637 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq)
            - 75.00406 * max(0.0, 0.06164517 - Q.z_7)
            - 584.8421 * max(0.0, 0.008678045 - Q.girth2)
            - 8.175155 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2)
            - 2.146648 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1)
            - 10.67569 * max(0.0, Q.sj3_dr_min - 0.03628191)
            + 7.170598 * max(0.0, Q.sj3_dr_max - 0.169029)
            + 4.052212 * max(0.0, Q.log_sum_pt - 6.605974)
            + 277.1018 * max(0.0, 0.008168571 - Q.e2_sq)
            - 0.005513285 * max(0.0, Q.sum_pt_top5 - 531.1875)
            + 519.7603 * max(0.0, 0.00595415 - Q.lam1)
            - 575.5693 * max(0.0, 0.00609665 - Q.width)
            + 5.616011 * max(0.0, Q.log_sum_pt - 6.502799)
            + 0.005023588 * max(0.0, Q.sum_pt_top5 - 367.5938)
            - 7969.009 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236)
            - 64.09914 * max(0.0, 0.0717028 - Q.girth)
        ))
        - 0.5 * grid(3, max(0.0, -4.505646
            + 25.31239 * max(0.0, Q.mass_over_sum_pt - 0.0681391)
            - 3.078744 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.sj3_pair_mass_min - 5.744224)
            - 26.71264 * max(0.0, Q.tau1 - 0.05356915)
            + 73.02017 * max(0.0, Q.girth - 0.07608178)
            - 150.6039 * max(0.0, Q.width - 0.01323868)
            + 481.726 * max(0.0, Q.width - 0.007520088)
            + 303.5563 * max(0.0, Q.e2 - 0.06344108)
            + 73.33973 * max(0.0, Q.girth - 0.04081947)
            + 53.8809 * max(0.0, Q.tau1 - 0.1027642)
            + 135.1412 * max(0.0, Q.girth - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494)
            - 1590.494 * max(0.0, Q.girth2 - 0.008678045)
            + 5.768425 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sj2_mass1 - 16.86126)
            + 49.59021 * max(0.0, Q.sj2_dr - 0.1872617)
            - 0.4136194 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113)
            - 4159.167 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr)
            + 767.6512 * max(0.0, Q.width - 0.007520088) * max(0.0, Q.sj3_pairmin_over_m - 0.07708997)
            + 12.96784 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50)
            + 371.8363 * max(0.0, Q.lam1 - 0.008375572)
            - 829.1944 * max(0.0, Q.lam2 - 0.001130645)
            + 27.12811 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6)
            - 248.1844 * max(0.0, Q.lam1 - 0.01200373)
            - 0.05985048 * max(0.0, Q.mass - 64.61873)
            + 0.05348794 * max(0.0, Q.sd_mass - 62.55)
            - 833.7736 * max(0.0, 0.004372139 - Q.girth2)
            - 0.1051442 * max(0.0, Q.sd_mass - 62.55) * max(0.0, 0.9206502 - Q.D2_b2)
            + 3897.237 * max(0.0, Q.lam1 - 0.01643375) * max(0.0, 6.0 - Q.n_for_90pct)
            - 11.09927 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1)
        ))
        - 0.3125 * grid(6, max(0.0, 0.5173415
            + 153.643 * max(0.0, Q.centroid_offset - 0.00809236)
            - 354.7936 * max(0.0, 0.01323868 - Q.width)
            + 2.134053 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308)
            + 0.07841492 * max(0.0, 41.21875 - Q.pt_6)
            + 1.196886 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt)
            + 53.59186 * max(0.0, 0.1136369 - Q.tau1)
            - 17.47437 * max(0.0, Q.sj3_dr_min - 0.1278212)
            - 181.4227 * max(0.0, Q.centroid_offset - 0.01837778)
            + 294.5117 * max(0.0, 0.01200373 - Q.lam1)
            + 8726.583 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2)
            - 0.09517011 * max(0.0, Q.sj3_pair_mass_min - 4.501727)
            - 13.67302 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757)
            + 1311.96 * max(0.0, 0.0030133 - Q.e2_sq)
            - 1879.342 * max(0.0, 0.001130645 - Q.lam2)
            - 1.093805 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4)
            + 43.21113 * max(0.0, 0.1789613 - Q.sj3_dr_max)
            - 41.4775 * max(0.0, 0.1452311 - Q.max_dr)
            + 88.03266 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05)
            + 5599.613 * max(0.0, 0.0005116989 - Q.e3)
            + 0.02568978 * max(0.0, 45.595 - Q.mass)
            - 1487.827 * max(0.0, 0.008678045 - Q.girth2)
            + 140.4956 * max(0.0, 0.00733008 - Q.lam1)
            - 0.09862539 * max(0.0, 39.75 - Q.pt_6)
            + 13.03165 * max(0.0, Q.sj3_dr_max - 0.1879486)
            - 0.0335275 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        ))
        + 0.21875 * grid(7, max(0.0, 8.568775
            - 2721.445 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.width - 0.007520088)
            - 976.9423 * max(0.0, Q.girth2 - 0.007520088)
            + 1796.577 * max(0.0, Q.girth2 - 0.01323868)
            - 372.3378 * max(0.0, Q.girth2 - 0.004372139)
            + 120.9052 * max(0.0, Q.mass_over_sum_pt - 0.07269073)
            + 490.7249 * max(0.0, Q.girth2 - 0.001653836)
            + 42.93539 * max(0.0, 0.05356915 - Q.tau1)
            - 53.58508 * max(0.0, 0.04081947 - Q.girth)
            - 0.2123986 * max(0.0, Q.mass - 76.6557)
            - 370.6026 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            + 699.8024 * max(0.0, Q.width - 0.008678045)
            + 265.375 * max(0.0, Q.mass_over_sum_pt - 0.08475161)
            + 0.05135435 * max(0.0, Q.mass - 36.22941)
            - 371446.3 * max(0.0, 8.10414e-05 - Q.girth2_top2)
            + 10.8081 * max(0.0, 0.1294903 - Q.sj2_dr)
            - 2626.53 * max(0.0, 0.0009641429 - Q.girth2)
            - 35.02856 * max(0.0, 0.1872617 - Q.sj2_dr)
            - 79.16962 * max(0.0, 0.08723651 - Q.girth)
            + 39.931 * max(0.0, 0.1591713 - Q.sj2_dr)
            + 11.07262 * max(0.0, 0.3033137 - Q.LHA)
            - 1182.647 * max(0.0, Q.width - 0.005590289)
            - 48.65684 * max(0.0, Q.centroid_offset - 0.03776099)
            - 48.20671 * max(0.0, Q.mass_over_sum_pt - 0.01109984)
            - 179.2791 * max(0.0, 0.008375572 - Q.lam1)
            + 2021.835 * max(0.0, Q.girth2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow)
            - 34560.39 * max(0.0, 8.147744e-05 - Q.e3)
            - 207.0337 * max(0.0, Q.e2 - 0.05028464)
        ))
        - 0.25 * grid(8, max(0.0, -0.5655652
            + 284.2185 * max(0.0, 0.005019719 - Q.girth2)
            + 31808.03 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            - 57.23899 * max(0.0, 0.1967397 - Q.LHA)
            - 16.61397 * max(0.0, Q.log_sum_pt - 6.701242)
            - 66.44382 * max(0.0, 0.06108601 - Q.girth)
            + 0.009398292 * max(0.0, Q.sum_pt_top5 - 687.4375)
            + 4965.175 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4)
            - 336.3577 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4)
            + 236714.0 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2)
            - 4.433283 * max(0.0, Q.z_dr_0_0p05 - 0.8477313)
            + 233.7561 * max(0.0, 0.003562611 - Q.width)
            - 0.003727224 * max(0.0, 21.78408 - Q.mass) * max(0.0, 48.71875 - Q.pt_7)
            - 25643.27 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738)
            + 118.4842 * max(0.0, 0.03319429 - Q.mass_over_sum_pt)
            - 27.84765 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            - 41.76267 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.003562611 - Q.width)
            + 1633.608 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.girth2)
            + 21672.79 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.width)
        ))
        - 0.03125 * grid(9, max(0.0, -3.212293
            - 57.93995 * max(0.0, 0.05464922 - Q.girth)
            + 97.89552 * max(0.0, 0.04369778 - Q.tau1)
            - 0.09790827 * max(0.0, 53.33237 - Q.mass)
            - 10657.71 * max(0.0, Q.e3 - 0.0001869378)
            + 6.756995 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset)
            + 1557.023 * max(0.0, 0.00609665 - Q.girth2)
            + 18.83085 * max(0.0, 0.213399 - Q.sj3_dr_max)
            + 551.8829 * max(0.0, Q.lam2 - 0.001130645)
            + 0.06960289 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt)
            + 2131.607 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.3220738 - Q.planar_flow)
            + 1.028209 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
            - 58.36824 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            - 229.5294 * max(0.0, Q.lam1 - 0.003377388)
            + 265.1134 * max(0.0, 0.01837778 - Q.centroid_offset)
            - 82.83437 * max(0.0, Q.e2 - 0.03556091)
            - 0.08341188 * max(0.0, 24.23013 - Q.sj3_pair_mass_max)
            + 565.524 * max(0.0, Q.lam1 - 0.005433361)
            - 69.61066 * max(0.0, 0.0284695 - Q.C3)
            - 802.3769 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266)
            - 866.67 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255)
            - 83266.02 * max(0.0, 2.371297e-05 - Q.e3)
            - 269.305 * max(0.0, 0.006292091 - Q.zdr_0)
            + 2354.024 * max(0.0, 0.003562611 - Q.girth2)
            - 53.34438 * max(0.0, 0.07269073 - Q.mass_over_sum_pt)
            + 0.009028949 * max(0.0, 988.4078 - Q.sum_pt)
            + 169.3854 * max(0.0, Q.e2 - 0.04447357)
            - 815.4236 * max(0.0, 0.00595415 - Q.lam1)
            - 8.507298 * Q.z_dr_0p2_0p4
            + 20.67153 * max(0.0, Q.log_sum_pt - 6.896095)
            - 55.42017 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2)
            - 0.01073602 * max(0.0, Q.sum_pt_top5 - 839.9547)
            - 56.22345 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0)
            - 2.795783 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1)
            + 1587.651 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd)
            + 0.01083795 * max(0.0, 559.6875 - Q.sum_pt)
            - 93.23307 * max(0.0, Q.girth - 0.0717028)
            + 10562.2 * max(0.0, Q.e3 - 8.147744e-05)
            + 47.1219 * max(0.0, 0.1117619 - Q.max_dr)
        ))
        + 0.375 * grid(11, max(0.0, -3.966889
            + 284.4231 * max(0.0, 0.01323868 - Q.girth2)
            - 12.77784 * max(0.0, 0.09538712 - Q.tau1)
            + 0.1729116 * max(0.0, 15.45403 - Q.mass)
            + 129.0277 * max(0.0, 0.03776099 - Q.centroid_offset)
            - 0.2518099 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt)
            - 64.49563 * max(0.0, 0.169029 - Q.sj3_dr_max)
            + 1501.963 * max(0.0, 0.008678045 - Q.width)
            - 263.2379 * max(0.0, 0.02054282 - Q.girth)
            - 235.6676 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2)
            - 415.8805 * max(0.0, 0.008168571 - Q.e2_sq)
            - 65.5769 * max(0.0, 0.0717028 - Q.girth)
            + 34.47866 * max(0.0, Q.z_7 - 0.01685855)
            + 25.381 * max(0.0, Q.sj2_dr - 0.2687922)
            - 0.005159328 * max(0.0, 69.61135 - Q.mass)
            + 15.05355 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            + 17.42834 * max(0.0, 0.2623172 - Q.sj3_dr_max)
            + 1141.619 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.5502779 - Q.tau32)
            - 530.8262 * max(0.0, 0.008375572 - Q.lam1)
            + 13.78469 * max(0.0, 0.1452311 - Q.max_dr)
        ))
        + 0.0703125 * grid(13, max(0.0, 0.1209103
            + 51.75967 * max(0.0, 0.1484084 - Q.girth)
            - 51.65779 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 1.390619 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
            + 229.1322 * max(0.0, 0.01643375 - Q.lam1)
            - 959087.9 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset)
            + 0.001175084 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7)
            - 1.59117 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7)
            - 39.23735 * max(0.0, 0.08000524 - Q.e2)
            - 271.9938 * max(0.0, 0.02807091 - Q.z_7)
            + 10907.64 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.D3 - 0.2213841)
            - 144.2247 * max(0.0, 0.007520088 - Q.width)
            + 224.4721 * max(0.0, 0.006506576 - Q.lam1)
            - 0.0006380717 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6)
        ))
        - 0.75 * grid(14, max(0.0, -1.410004
            - 27.56896 * max(0.0, Q.psi_0p1 - 0.9761279)
            - 10286.64 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1)
            + 1410.604 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1)
            + 9784.525 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.width)
            - 1211.825 * max(0.0, Q.girth2 - 0.007520088)
            + 260.1843 * max(0.0, Q.e2_sq - 0.002074109)
            + 604.7333 * max(0.0, Q.width - 0.01323868)
            - 838.7508 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset)
            + 291.2286 * max(0.0, Q.e2_sq - 0.01165737)
            - 54.85611 * max(0.0, Q.e2 - 0.03556091)
            + 424.9939 * max(0.0, Q.e2 - 0.05028464)
            - 1110.9 * max(0.0, Q.centroid_offset - 0.04990367)
            + 163.989 * max(0.0, Q.e2 - 0.04110972)
            - 142.8312 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            - 11.13433 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095)
            + 13.24309 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
            - 192.4897 * max(0.0, Q.e2 - 0.01655442)
            - 20895.89 * max(0.0, 5.334511e-05 - Q.e3)
            - 0.05393691 * max(0.0, 74.57663 - Q.sd_mass)
            + 0.06173994 * max(0.0, 49.91626 - Q.sd_mass)
            + 216.1041 * max(0.0, Q.mass_over_sum_pt - 0.08475161)
            + 39.57292 * max(0.0, Q.girth2 - 0.008678045)
            - 292.2232 * max(0.0, Q.e2_sq - 0.0030133)
            - 131.5589 * max(0.0, Q.girth - 0.1019409)
            + 72.26187 * max(0.0, Q.girth - 0.04081947)
            - 109.0461 * max(0.0, Q.centroid_offset - 0.02685622)
            - 317.6671 * max(0.0, Q.girth - 0.08723651)
            + 79.74967 * max(0.0, Q.girth - 0.08065885)
            - 1803.456 * max(0.0, Q.width - 0.006679471)
            - 4.554884 * max(0.0, Q.sj2_dr - 0.06154135)
            + 54.3218 * max(0.0, Q.sj2_dr - 0.1591713)
            - 21.86451 * max(0.0, Q.sj2_dr - 0.2001708)
            - 6.966696 * max(0.0, Q.sj2_dr - 0.1294903)
            + 90.44431 * max(0.0, Q.girth - 0.02689598)
            + 406.8015 * max(0.0, Q.width - 0.003562611)
            + 185.8474 * max(0.0, Q.mass_over_sum_pt - 0.07992374)
        ))
        - 0.6875 * grid(15, max(0.0, -0.3878572
            - 859.4324 * max(0.0, 0.007520088 - Q.girth2)
            - 30.25677 * max(0.0, 0.1309286 - Q.mass_over_sum_pt)
            - 367.6759 * max(0.0, 0.00609665 - Q.width)
            + 717.7877 * max(0.0, 0.01323868 - Q.width)
            - 1700.759 * max(0.0, 0.007520088 - Q.girth2) * max(0.0, 0.7459513 - Q.D2)
            - 4818.035 * max(0.0, 0.00609665 - Q.width) * max(0.0, 0.7459513 - Q.D2)
            + 100.9499 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2)
            + 230.7089 * max(0.0, 0.04110972 - Q.e2)
            - 319.5466 * max(0.0, 0.01165737 - Q.e2_sq)
            + 942.3414 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq)
            - 754.7946 * max(0.0, 0.008168571 - Q.e2_sq)
            + 158.8951 * max(0.0, 0.00483998 - Q.lam1)
            + 241.8992 * max(0.0, 0.01882765 - Q.width)
            - 111.7434 * max(0.0, 0.1019409 - Q.girth)
            + 56.6489 * max(0.0, 0.1591713 - Q.sj2_dr)
            - 36.55215 * max(0.0, 0.2001708 - Q.sj2_dr)
            - 13.52371 * max(0.0, 0.1492731 - Q.sj2_dr)
            - 4810.125 * max(0.0, 0.0005124533 - Q.girth2_top2)
        ))
    )


def logit_Z(Q):
    return (-0.09375
        + 0.125 * grid(1, max(0.0, -0.1230816
            + 0.1324637 * max(0.0, Q.pt_7 - 34.53125)
            + 4.910623 * max(0.0, Q.log_sum_pt - 6.377723)
            + 892.9637 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq)
            - 75.00406 * max(0.0, 0.06164517 - Q.z_7)
            - 584.8421 * max(0.0, 0.008678045 - Q.girth2)
            - 8.175155 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2)
            - 2.146648 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1)
            - 10.67569 * max(0.0, Q.sj3_dr_min - 0.03628191)
            + 7.170598 * max(0.0, Q.sj3_dr_max - 0.169029)
            + 4.052212 * max(0.0, Q.log_sum_pt - 6.605974)
            + 277.1018 * max(0.0, 0.008168571 - Q.e2_sq)
            - 0.005513285 * max(0.0, Q.sum_pt_top5 - 531.1875)
            + 519.7603 * max(0.0, 0.00595415 - Q.lam1)
            - 575.5693 * max(0.0, 0.00609665 - Q.width)
            + 5.616011 * max(0.0, Q.log_sum_pt - 6.502799)
            + 0.005023588 * max(0.0, Q.sum_pt_top5 - 367.5938)
            - 7969.009 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236)
            - 64.09914 * max(0.0, 0.0717028 - Q.girth)
        ))
        - 0.5625 * grid(3, max(0.0, -4.505646
            + 25.31239 * max(0.0, Q.mass_over_sum_pt - 0.0681391)
            - 3.078744 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.sj3_pair_mass_min - 5.744224)
            - 26.71264 * max(0.0, Q.tau1 - 0.05356915)
            + 73.02017 * max(0.0, Q.girth - 0.07608178)
            - 150.6039 * max(0.0, Q.width - 0.01323868)
            + 481.726 * max(0.0, Q.width - 0.007520088)
            + 303.5563 * max(0.0, Q.e2 - 0.06344108)
            + 73.33973 * max(0.0, Q.girth - 0.04081947)
            + 53.8809 * max(0.0, Q.tau1 - 0.1027642)
            + 135.1412 * max(0.0, Q.girth - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494)
            - 1590.494 * max(0.0, Q.girth2 - 0.008678045)
            + 5.768425 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sj2_mass1 - 16.86126)
            + 49.59021 * max(0.0, Q.sj2_dr - 0.1872617)
            - 0.4136194 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113)
            - 4159.167 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr)
            + 767.6512 * max(0.0, Q.width - 0.007520088) * max(0.0, Q.sj3_pairmin_over_m - 0.07708997)
            + 12.96784 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50)
            + 371.8363 * max(0.0, Q.lam1 - 0.008375572)
            - 829.1944 * max(0.0, Q.lam2 - 0.001130645)
            + 27.12811 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6)
            - 248.1844 * max(0.0, Q.lam1 - 0.01200373)
            - 0.05985048 * max(0.0, Q.mass - 64.61873)
            + 0.05348794 * max(0.0, Q.sd_mass - 62.55)
            - 833.7736 * max(0.0, 0.004372139 - Q.girth2)
            - 0.1051442 * max(0.0, Q.sd_mass - 62.55) * max(0.0, 0.9206502 - Q.D2_b2)
            + 3897.237 * max(0.0, Q.lam1 - 0.01643375) * max(0.0, 6.0 - Q.n_for_90pct)
            - 11.09927 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1)
        ))
        + 0.078125 * grid(4, max(0.0, -1.984515
            + 61.9689 * max(0.0, 0.2233283 - Q.N2)
            - 0.7834277 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.55 - Q.mass)
            - 10570.73 * max(0.0, 0.000537286 - Q.lam2)
            - 149.41 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            + 376.1682 * max(0.0, Q.width - 0.003562611)
            + 73.52333 * max(0.0, Q.e2 - 0.04447357)
            - 0.003755188 * max(0.0, 739.5 - Q.sum_pt)
            - 0.7519606 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7)
            - 73.94671 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266)
            - 1.867846 * max(0.0, 0.177305 - Q.max_dr)
            + 79.57784 * max(0.0, 0.06729223 - Q.C2)
            - 42.64309 * max(0.0, Q.girth - 0.08723651)
            + 0.1290954 * max(0.0, Q.sd_mass - 44.82259)
            + 0.09193323 * max(0.0, Q.sd_mass - 74.57663)
            + 14224.94 * max(0.0, 0.0001869378 - Q.e3)
            + 5.664068 * max(0.0, 0.3456459 - Q.sj3_dr_max)
            + 30.89943 * max(0.0, Q.max_dr - 0.1598486)
            - 197.5795 * max(0.0, 0.008678045 - Q.girth2)
            - 1.730639 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.centroid_offset - 0.001308549)
            - 73.89394 * max(0.0, 0.04990367 - Q.centroid_offset)
            + 390.8946 * max(0.0, 0.007639643 - Q.girth2_top2)
            - 0.9066542 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.275762 - Q.sd_zg)
            - 5836.058 * max(0.0, Q.width - 0.003562611) * max(0.0, 0.1057566 - Q.sj3_z3)
            - 0.2999585 * max(0.0, Q.mass - 76.6557)
            + 3246.117 * max(0.0, 0.001653836 - Q.width)
            + 3214.91 * max(0.0, 0.01165737 - Q.e2_sq)
            - 85.26767 * max(0.0, 0.05028464 - Q.e2)
            - 56676.59 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.0006435798)
            - 41.2037 * max(0.0, 0.233678 - Q.sj3_dr_max)
            - 2661.375 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq)
            + 8.220409 * max(0.0, 0.3467135 - Q.LHA)
        ))
        + 0.015625 * grid(5, max(0.0, 0.7058495
            + 17.35268 * max(0.0, 0.2160559 - Q.LHA)
            + 79.62085 * max(0.0, 0.04939969 - Q.z_7)
            - 159.9139 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 58.21803 * max(0.0, Q.log_sum_pt - 6.896095)
            + 643.0026 * max(0.0, 0.005284669 - Q.e2_sq)
            - 13172.7 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, 0.01437952 - Q.centroid_offset)
            - 1771.377 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset)
            + 1925.216 * max(0.0, 0.001653836 - Q.girth2)
            + 1540.288 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.05744392 - Q.D2_b2)
            + 0.5976955 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 40.2 - Q.mass_top3)
            + 996.9643 * max(0.0, 0.007673833 - Q.girth)
            - 0.0638411 * max(0.0, 37.15625 - Q.pt_7)
            + 151167.2 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            + 240.1729 * max(0.0, 0.02807091 - Q.z_7)
            - 13584.06 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.001503553 - Q.lam1)
            - 154.6689 * max(0.0, 0.0211821 - Q.zdr_0)
            - 18.88884 * max(0.0, Q.log_sum_pt - 6.701242)
            + 1026.925 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01341502 - Q.dr_2)
            + 0.0001369071 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 28.39396 - Q.pt1_dr01)
            + 0.03659241 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.sj2_dr - 0.1682655)
            + 385.7445 * max(0.0, 0.003562611 - Q.width)
            - 908.0424 * max(0.0, 0.002074109 - Q.e2_sq)
            - 615.4584 * max(0.0, 0.002151568 - Q.girth2_top3)
            + 693.5255 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.sj3_dr13 - 0.04995258)
        ))
        - 0.375 * grid(6, max(0.0, 0.5173415
            + 153.643 * max(0.0, Q.centroid_offset - 0.00809236)
            - 354.7936 * max(0.0, 0.01323868 - Q.width)
            + 2.134053 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308)
            + 0.07841492 * max(0.0, 41.21875 - Q.pt_6)
            + 1.196886 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt)
            + 53.59186 * max(0.0, 0.1136369 - Q.tau1)
            - 17.47437 * max(0.0, Q.sj3_dr_min - 0.1278212)
            - 181.4227 * max(0.0, Q.centroid_offset - 0.01837778)
            + 294.5117 * max(0.0, 0.01200373 - Q.lam1)
            + 8726.583 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2)
            - 0.09517011 * max(0.0, Q.sj3_pair_mass_min - 4.501727)
            - 13.67302 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757)
            + 1311.96 * max(0.0, 0.0030133 - Q.e2_sq)
            - 1879.342 * max(0.0, 0.001130645 - Q.lam2)
            - 1.093805 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4)
            + 43.21113 * max(0.0, 0.1789613 - Q.sj3_dr_max)
            - 41.4775 * max(0.0, 0.1452311 - Q.max_dr)
            + 88.03266 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05)
            + 5599.613 * max(0.0, 0.0005116989 - Q.e3)
            + 0.02568978 * max(0.0, 45.595 - Q.mass)
            - 1487.827 * max(0.0, 0.008678045 - Q.girth2)
            + 140.4956 * max(0.0, 0.00733008 - Q.lam1)
            - 0.09862539 * max(0.0, 39.75 - Q.pt_6)
            + 13.03165 * max(0.0, Q.sj3_dr_max - 0.1879486)
            - 0.0335275 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        ))
        + 0.46875 * grid(7, max(0.0, 8.568775
            - 2721.445 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.width - 0.007520088)
            - 976.9423 * max(0.0, Q.girth2 - 0.007520088)
            + 1796.577 * max(0.0, Q.girth2 - 0.01323868)
            - 372.3378 * max(0.0, Q.girth2 - 0.004372139)
            + 120.9052 * max(0.0, Q.mass_over_sum_pt - 0.07269073)
            + 490.7249 * max(0.0, Q.girth2 - 0.001653836)
            + 42.93539 * max(0.0, 0.05356915 - Q.tau1)
            - 53.58508 * max(0.0, 0.04081947 - Q.girth)
            - 0.2123986 * max(0.0, Q.mass - 76.6557)
            - 370.6026 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            + 699.8024 * max(0.0, Q.width - 0.008678045)
            + 265.375 * max(0.0, Q.mass_over_sum_pt - 0.08475161)
            + 0.05135435 * max(0.0, Q.mass - 36.22941)
            - 371446.3 * max(0.0, 8.10414e-05 - Q.girth2_top2)
            + 10.8081 * max(0.0, 0.1294903 - Q.sj2_dr)
            - 2626.53 * max(0.0, 0.0009641429 - Q.girth2)
            - 35.02856 * max(0.0, 0.1872617 - Q.sj2_dr)
            - 79.16962 * max(0.0, 0.08723651 - Q.girth)
            + 39.931 * max(0.0, 0.1591713 - Q.sj2_dr)
            + 11.07262 * max(0.0, 0.3033137 - Q.LHA)
            - 1182.647 * max(0.0, Q.width - 0.005590289)
            - 48.65684 * max(0.0, Q.centroid_offset - 0.03776099)
            - 48.20671 * max(0.0, Q.mass_over_sum_pt - 0.01109984)
            - 179.2791 * max(0.0, 0.008375572 - Q.lam1)
            + 2021.835 * max(0.0, Q.girth2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow)
            - 34560.39 * max(0.0, 8.147744e-05 - Q.e3)
            - 207.0337 * max(0.0, Q.e2 - 0.05028464)
        ))
        - 0.03125 * grid(9, max(0.0, -3.212293
            - 57.93995 * max(0.0, 0.05464922 - Q.girth)
            + 97.89552 * max(0.0, 0.04369778 - Q.tau1)
            - 0.09790827 * max(0.0, 53.33237 - Q.mass)
            - 10657.71 * max(0.0, Q.e3 - 0.0001869378)
            + 6.756995 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset)
            + 1557.023 * max(0.0, 0.00609665 - Q.girth2)
            + 18.83085 * max(0.0, 0.213399 - Q.sj3_dr_max)
            + 551.8829 * max(0.0, Q.lam2 - 0.001130645)
            + 0.06960289 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt)
            + 2131.607 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.3220738 - Q.planar_flow)
            + 1.028209 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
            - 58.36824 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            - 229.5294 * max(0.0, Q.lam1 - 0.003377388)
            + 265.1134 * max(0.0, 0.01837778 - Q.centroid_offset)
            - 82.83437 * max(0.0, Q.e2 - 0.03556091)
            - 0.08341188 * max(0.0, 24.23013 - Q.sj3_pair_mass_max)
            + 565.524 * max(0.0, Q.lam1 - 0.005433361)
            - 69.61066 * max(0.0, 0.0284695 - Q.C3)
            - 802.3769 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266)
            - 866.67 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255)
            - 83266.02 * max(0.0, 2.371297e-05 - Q.e3)
            - 269.305 * max(0.0, 0.006292091 - Q.zdr_0)
            + 2354.024 * max(0.0, 0.003562611 - Q.girth2)
            - 53.34438 * max(0.0, 0.07269073 - Q.mass_over_sum_pt)
            + 0.009028949 * max(0.0, 988.4078 - Q.sum_pt)
            + 169.3854 * max(0.0, Q.e2 - 0.04447357)
            - 815.4236 * max(0.0, 0.00595415 - Q.lam1)
            - 8.507298 * Q.z_dr_0p2_0p4
            + 20.67153 * max(0.0, Q.log_sum_pt - 6.896095)
            - 55.42017 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2)
            - 0.01073602 * max(0.0, Q.sum_pt_top5 - 839.9547)
            - 56.22345 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0)
            - 2.795783 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1)
            + 1587.651 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd)
            + 0.01083795 * max(0.0, 559.6875 - Q.sum_pt)
            - 93.23307 * max(0.0, Q.girth - 0.0717028)
            + 10562.2 * max(0.0, Q.e3 - 8.147744e-05)
            + 47.1219 * max(0.0, 0.1117619 - Q.max_dr)
        ))
        + 0.0546875 * grid(13, max(0.0, 0.1209103
            + 51.75967 * max(0.0, 0.1484084 - Q.girth)
            - 51.65779 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 1.390619 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
            + 229.1322 * max(0.0, 0.01643375 - Q.lam1)
            - 959087.9 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset)
            + 0.001175084 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7)
            - 1.59117 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7)
            - 39.23735 * max(0.0, 0.08000524 - Q.e2)
            - 271.9938 * max(0.0, 0.02807091 - Q.z_7)
            + 10907.64 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.D3 - 0.2213841)
            - 144.2247 * max(0.0, 0.007520088 - Q.width)
            + 224.4721 * max(0.0, 0.006506576 - Q.lam1)
            - 0.0006380717 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6)
        ))
        + 0.375 * grid(14, max(0.0, -1.410004
            - 27.56896 * max(0.0, Q.psi_0p1 - 0.9761279)
            - 10286.64 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1)
            + 1410.604 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1)
            + 9784.525 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.width)
            - 1211.825 * max(0.0, Q.girth2 - 0.007520088)
            + 260.1843 * max(0.0, Q.e2_sq - 0.002074109)
            + 604.7333 * max(0.0, Q.width - 0.01323868)
            - 838.7508 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset)
            + 291.2286 * max(0.0, Q.e2_sq - 0.01165737)
            - 54.85611 * max(0.0, Q.e2 - 0.03556091)
            + 424.9939 * max(0.0, Q.e2 - 0.05028464)
            - 1110.9 * max(0.0, Q.centroid_offset - 0.04990367)
            + 163.989 * max(0.0, Q.e2 - 0.04110972)
            - 142.8312 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            - 11.13433 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095)
            + 13.24309 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
            - 192.4897 * max(0.0, Q.e2 - 0.01655442)
            - 20895.89 * max(0.0, 5.334511e-05 - Q.e3)
            - 0.05393691 * max(0.0, 74.57663 - Q.sd_mass)
            + 0.06173994 * max(0.0, 49.91626 - Q.sd_mass)
            + 216.1041 * max(0.0, Q.mass_over_sum_pt - 0.08475161)
            + 39.57292 * max(0.0, Q.girth2 - 0.008678045)
            - 292.2232 * max(0.0, Q.e2_sq - 0.0030133)
            - 131.5589 * max(0.0, Q.girth - 0.1019409)
            + 72.26187 * max(0.0, Q.girth - 0.04081947)
            - 109.0461 * max(0.0, Q.centroid_offset - 0.02685622)
            - 317.6671 * max(0.0, Q.girth - 0.08723651)
            + 79.74967 * max(0.0, Q.girth - 0.08065885)
            - 1803.456 * max(0.0, Q.width - 0.006679471)
            - 4.554884 * max(0.0, Q.sj2_dr - 0.06154135)
            + 54.3218 * max(0.0, Q.sj2_dr - 0.1591713)
            - 21.86451 * max(0.0, Q.sj2_dr - 0.2001708)
            - 6.966696 * max(0.0, Q.sj2_dr - 0.1294903)
            + 90.44431 * max(0.0, Q.girth - 0.02689598)
            + 406.8015 * max(0.0, Q.width - 0.003562611)
            + 185.8474 * max(0.0, Q.mass_over_sum_pt - 0.07992374)
        ))
        - 0.15625 * grid(15, max(0.0, -0.3878572
            - 859.4324 * max(0.0, 0.007520088 - Q.girth2)
            - 30.25677 * max(0.0, 0.1309286 - Q.mass_over_sum_pt)
            - 367.6759 * max(0.0, 0.00609665 - Q.width)
            + 717.7877 * max(0.0, 0.01323868 - Q.width)
            - 1700.759 * max(0.0, 0.007520088 - Q.girth2) * max(0.0, 0.7459513 - Q.D2)
            - 4818.035 * max(0.0, 0.00609665 - Q.width) * max(0.0, 0.7459513 - Q.D2)
            + 100.9499 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2)
            + 230.7089 * max(0.0, 0.04110972 - Q.e2)
            - 319.5466 * max(0.0, 0.01165737 - Q.e2_sq)
            + 942.3414 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq)
            - 754.7946 * max(0.0, 0.008168571 - Q.e2_sq)
            + 158.8951 * max(0.0, 0.00483998 - Q.lam1)
            + 241.8992 * max(0.0, 0.01882765 - Q.width)
            - 111.7434 * max(0.0, 0.1019409 - Q.girth)
            + 56.6489 * max(0.0, 0.1591713 - Q.sj2_dr)
            - 36.55215 * max(0.0, 0.2001708 - Q.sj2_dr)
            - 13.52371 * max(0.0, 0.1492731 - Q.sj2_dr)
            - 4810.125 * max(0.0, 0.0005124533 - Q.girth2_top2)
        ))
    )


def logit_t(Q):
    return (1.34375
        + 0.015625 * grid(0, max(0.0, -2.547365
            + 17.12106 * max(0.0, 0.1484197 - Q.planar_flow)
            - 2100.026 * max(0.0, 0.004372139 - Q.width)
            + 0.2586946 * max(0.0, 21.78408 - Q.mass)
            + 290.5179 * max(0.0, 0.01323868 - Q.girth2)
            + 335.9675 * max(0.0, 0.006506576 - Q.lam1)
            - 3140.132 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2)
            - 0.02317622 * max(0.0, Q.sum_pt - 901.5938)
            + 435.0238 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 1.002471 - Q.D2)
            - 0.04211574 * max(0.0, 56.92035 - Q.mass)
            + 594.8094 * max(0.0, 0.008678045 - Q.width)
            - 1656.548 * max(0.0, 0.001563465 - Q.C2_b2)
            - 27.81783 * max(0.0, Q.sj3_dr_max - 0.233678)
            + 4.358228 * max(0.0, Q.sj3_dr_max - 0.1070199)
            + 10.18314 * max(0.0, 0.3012016 - Q.sj3_dr_max)
            - 0.1516387 * max(0.0, 29.6447 - Q.mass)
            - 12398.28 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.centroid_offset - 0.01837778)
            - 47.34599 * max(0.0, 0.08723651 - Q.girth)
            + 228.7248 * max(0.0, 0.0245477 - Q.e2)
            - 423.6481 * max(0.0, 0.005433361 - Q.lam1)
            + 0.360254 * max(0.0, Q.n_dr_0_0p05 - 4.0)
            - 14.56131 * max(0.0, Q.log_sum_pt - 6.670067)
            + 0.01695238 * max(0.0, Q.sum_pt_top5 - 687.4375)
            + 84155.38 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2)
        ))
        + 0.0625 * grid(3, max(0.0, -4.505646
            + 25.31239 * max(0.0, Q.mass_over_sum_pt - 0.0681391)
            - 3.078744 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.sj3_pair_mass_min - 5.744224)
            - 26.71264 * max(0.0, Q.tau1 - 0.05356915)
            + 73.02017 * max(0.0, Q.girth - 0.07608178)
            - 150.6039 * max(0.0, Q.width - 0.01323868)
            + 481.726 * max(0.0, Q.width - 0.007520088)
            + 303.5563 * max(0.0, Q.e2 - 0.06344108)
            + 73.33973 * max(0.0, Q.girth - 0.04081947)
            + 53.8809 * max(0.0, Q.tau1 - 0.1027642)
            + 135.1412 * max(0.0, Q.girth - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494)
            - 1590.494 * max(0.0, Q.girth2 - 0.008678045)
            + 5.768425 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sj2_mass1 - 16.86126)
            + 49.59021 * max(0.0, Q.sj2_dr - 0.1872617)
            - 0.4136194 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113)
            - 4159.167 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr)
            + 767.6512 * max(0.0, Q.width - 0.007520088) * max(0.0, Q.sj3_pairmin_over_m - 0.07708997)
            + 12.96784 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50)
            + 371.8363 * max(0.0, Q.lam1 - 0.008375572)
            - 829.1944 * max(0.0, Q.lam2 - 0.001130645)
            + 27.12811 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6)
            - 248.1844 * max(0.0, Q.lam1 - 0.01200373)
            - 0.05985048 * max(0.0, Q.mass - 64.61873)
            + 0.05348794 * max(0.0, Q.sd_mass - 62.55)
            - 833.7736 * max(0.0, 0.004372139 - Q.girth2)
            - 0.1051442 * max(0.0, Q.sd_mass - 62.55) * max(0.0, 0.9206502 - Q.D2_b2)
            + 3897.237 * max(0.0, Q.lam1 - 0.01643375) * max(0.0, 6.0 - Q.n_for_90pct)
            - 11.09927 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1)
        ))
        + 0.125 * grid(4, max(0.0, -1.984515
            + 61.9689 * max(0.0, 0.2233283 - Q.N2)
            - 0.7834277 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.55 - Q.mass)
            - 10570.73 * max(0.0, 0.000537286 - Q.lam2)
            - 149.41 * max(0.0, Q.mass_over_sum_pt - 0.09041383)
            + 376.1682 * max(0.0, Q.width - 0.003562611)
            + 73.52333 * max(0.0, Q.e2 - 0.04447357)
            - 0.003755188 * max(0.0, 739.5 - Q.sum_pt)
            - 0.7519606 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7)
            - 73.94671 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266)
            - 1.867846 * max(0.0, 0.177305 - Q.max_dr)
            + 79.57784 * max(0.0, 0.06729223 - Q.C2)
            - 42.64309 * max(0.0, Q.girth - 0.08723651)
            + 0.1290954 * max(0.0, Q.sd_mass - 44.82259)
            + 0.09193323 * max(0.0, Q.sd_mass - 74.57663)
            + 14224.94 * max(0.0, 0.0001869378 - Q.e3)
            + 5.664068 * max(0.0, 0.3456459 - Q.sj3_dr_max)
            + 30.89943 * max(0.0, Q.max_dr - 0.1598486)
            - 197.5795 * max(0.0, 0.008678045 - Q.girth2)
            - 1.730639 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.centroid_offset - 0.001308549)
            - 73.89394 * max(0.0, 0.04990367 - Q.centroid_offset)
            + 390.8946 * max(0.0, 0.007639643 - Q.girth2_top2)
            - 0.9066542 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.275762 - Q.sd_zg)
            - 5836.058 * max(0.0, Q.width - 0.003562611) * max(0.0, 0.1057566 - Q.sj3_z3)
            - 0.2999585 * max(0.0, Q.mass - 76.6557)
            + 3246.117 * max(0.0, 0.001653836 - Q.width)
            + 3214.91 * max(0.0, 0.01165737 - Q.e2_sq)
            - 85.26767 * max(0.0, 0.05028464 - Q.e2)
            - 56676.59 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.0006435798)
            - 41.2037 * max(0.0, 0.233678 - Q.sj3_dr_max)
            - 2661.375 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq)
            + 8.220409 * max(0.0, 0.3467135 - Q.LHA)
        ))
        - 0.25 * grid(5, max(0.0, 0.7058495
            + 17.35268 * max(0.0, 0.2160559 - Q.LHA)
            + 79.62085 * max(0.0, 0.04939969 - Q.z_7)
            - 159.9139 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 58.21803 * max(0.0, Q.log_sum_pt - 6.896095)
            + 643.0026 * max(0.0, 0.005284669 - Q.e2_sq)
            - 13172.7 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, 0.01437952 - Q.centroid_offset)
            - 1771.377 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset)
            + 1925.216 * max(0.0, 0.001653836 - Q.girth2)
            + 1540.288 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.05744392 - Q.D2_b2)
            + 0.5976955 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 40.2 - Q.mass_top3)
            + 996.9643 * max(0.0, 0.007673833 - Q.girth)
            - 0.0638411 * max(0.0, 37.15625 - Q.pt_7)
            + 151167.2 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            + 240.1729 * max(0.0, 0.02807091 - Q.z_7)
            - 13584.06 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.001503553 - Q.lam1)
            - 154.6689 * max(0.0, 0.0211821 - Q.zdr_0)
            - 18.88884 * max(0.0, Q.log_sum_pt - 6.701242)
            + 1026.925 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01341502 - Q.dr_2)
            + 0.0001369071 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 28.39396 - Q.pt1_dr01)
            + 0.03659241 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.sj2_dr - 0.1682655)
            + 385.7445 * max(0.0, 0.003562611 - Q.width)
            - 908.0424 * max(0.0, 0.002074109 - Q.e2_sq)
            - 615.4584 * max(0.0, 0.002151568 - Q.girth2_top3)
            + 693.5255 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.sj3_dr13 - 0.04995258)
        ))
        + 0.1875 * grid(8, max(0.0, -0.5655652
            + 284.2185 * max(0.0, 0.005019719 - Q.girth2)
            + 31808.03 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset)
            - 57.23899 * max(0.0, 0.1967397 - Q.LHA)
            - 16.61397 * max(0.0, Q.log_sum_pt - 6.701242)
            - 66.44382 * max(0.0, 0.06108601 - Q.girth)
            + 0.009398292 * max(0.0, Q.sum_pt_top5 - 687.4375)
            + 4965.175 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4)
            - 336.3577 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4)
            + 236714.0 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2)
            - 4.433283 * max(0.0, Q.z_dr_0_0p05 - 0.8477313)
            + 233.7561 * max(0.0, 0.003562611 - Q.width)
            - 0.003727224 * max(0.0, 21.78408 - Q.mass) * max(0.0, 48.71875 - Q.pt_7)
            - 25643.27 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738)
            + 118.4842 * max(0.0, 0.03319429 - Q.mass_over_sum_pt)
            - 27.84765 * max(0.0, 0.1426152 - Q.sj3_dr_max)
            - 41.76267 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.003562611 - Q.width)
            + 1633.608 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.girth2)
            + 21672.79 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.width)
        ))
        + 0.375 * grid(10, max(0.0, 1.525997
            - 36.4558 * Q.e2
            + 0.1035518 * max(0.0, Q.sj3_pair_mass_min - 11.051)
            - 331.5034 * max(0.0, 0.004183811 - Q.lam1)
            + 746.1591 * max(0.0, Q.lam2 - 0.0003061234)
            - 0.04274978 * max(0.0, 45.75 - Q.pt_7)
            + 17558.84 * max(0.0, 8.147744e-05 - Q.e3)
            - 696.6071 * max(0.0, Q.lam2 - 0.003408389)
            + 0.1273066 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.002471 - Q.D2)
            - 313.4167 * max(0.0, 0.00595415 - Q.lam1)
            - 1928.105 * max(0.0, 0.001503553 - Q.lam1)
            + 182.2886 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1)
            - 34.71547 * max(0.0, Q.LHA - 0.3033137)
            + 39.6598 * max(0.0, Q.tau1 - 0.05356915)
            + 487.1833 * max(0.0, Q.girth2 - 0.007520088)
            - 149.677 * max(0.0, Q.lam1 - 0.00733008)
            - 146055.7 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1797097)
            - 524.1245 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2)
            - 4762.144 * max(0.0, Q.e3 - 3.892127e-05)
            + 0.02752135 * max(0.0, 76.6557 - Q.mass)
            + 126.0312 * max(0.0, Q.C2_b2 - 0.009032972)
        ))
        - 0.5 * grid(12, max(0.0, -1.43861
            + 346.9805 * max(0.0, Q.girth2 - 0.01882765)
            + 0.09597781 * max(0.0, Q.mass - 91.19)
            - 79.84167 * max(0.0, Q.e2 - 0.06344108)
        ))
        - 0.40625 * grid(13, max(0.0, 0.1209103
            + 51.75967 * max(0.0, 0.1484084 - Q.girth)
            - 51.65779 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt)
            - 1.390619 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
            + 229.1322 * max(0.0, 0.01643375 - Q.lam1)
            - 959087.9 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset)
            + 0.001175084 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7)
            - 1.59117 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7)
            - 39.23735 * max(0.0, 0.08000524 - Q.e2)
            - 271.9938 * max(0.0, 0.02807091 - Q.z_7)
            + 10907.64 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.D3 - 0.2213841)
            - 144.2247 * max(0.0, 0.007520088 - Q.width)
            + 224.4721 * max(0.0, 0.006506576 - Q.lam1)
            - 0.0006380717 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6)
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
