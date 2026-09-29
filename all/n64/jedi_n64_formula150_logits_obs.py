"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; all observables), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.6% (the network: 81.1%); same class as the network for 91.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.tau43                  N-subjettiness τ4/τ3
"""
import math
from types import SimpleNamespace

CLASSES = ['g', 'q', 'W', 'Z', 't']
W = [[0.0, 0.0, 0.75, -1.375, 0.125], [0.625, -0.1875, 0.0, 0.0, 0.0], [0.0, 0.125, 0.0, -0.375, 0.0], [-0.75, 0.0, 0.4375, 0.4375, 0.0], [-0.875, -1.0625, 0.34375, 0.0, 0.1875], [-0.3125, 0.0, 0.578125, 0.6875, -0.46875], [0.15625, 0.21875, 0.0625, -0.96875, 0.0], [0.0, 0.0, -0.625, 0.90625, -0.28125], [0.03125, 0.015625, -0.875, -0.9375, 0.2109375], [0.515625, 0.5625, -0.21875, 0.0, 0.0], [-0.015625, -0.015625, 0.0, 0.0, 0.984375], [0.0, -0.25, 0.59375, 0.0, 0.0], [0.234375, 0.34375, -0.40625, 0.3125, -0.375], [0.0, 0.0, 0.0, 0.0, -0.90625], [0.0, 0.0, -1.375, 0.0, 0.0], [0.0, 0.0, -0.1875, 0.5625, -0.375]]
B = [-1.078125, 1.359375, 0.09375, 0.984375, 0.78125]
INT_BITS = [2, 4, 3, 2, 2, 2, 3, 3, 4, 3, 3, 3, 3, 2, 2, 3]
FRAC_BITS = [5, 5, 3, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 5, 5, 4]


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
        z_top5=sum(zs[:5]),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        z_top2_slots=sum(pt[:2]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        eta_1=eta[1],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-1.078125
        + 0.625 * grid(1, max(0.0, 3.08
            - 46.1 * max(0.0, 0.0312 - Q.M3)
            + 1590.0 * max(0.0, 0.000549 - Q.girth2_top3)
            + 20.3 * max(0.0, Q.log_sum_pt - 6.9)
            - 18.4 * max(0.0, Q.log_sum_pt - 6.99)
            - 10.9 * max(0.0, 7.16 - Q.log_sum_pt)
            - 0.11 * max(0.0, 6.23 - Q.n_dr_0p2_0p4)
            + 0.118 * max(0.0, Q.n_particles - 38.5)
            - 0.0252 * max(0.0, 31.5 - Q.sj3_mass1)
            + 0.019 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.00447 * max(0.0, 699.0 - Q.sum_pt_top2)
            + 0.00307 * max(0.0, 51.1 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 27.7)
            + 0.414 * max(0.0, Q.n_particles - 35.1) * max(0.0, 0.127 - Q.dr_0)
            - 0.0425 * max(0.0, Q.n_particles - 37.8) * max(0.0, 2.32 - Q.soft1_pt)
            - 0.0029 * max(0.0, 32.3 - Q.sj3_mass1) * max(0.0, 18.5 - Q.sj3_mass2)
            - 0.0075 * max(0.0, 706.0 - Q.sum_pt_top2) * max(0.0, 0.959 - Q.tau43)
            + 0.538 * max(0.0, Q.z_top30_slots - 0.947) * max(0.0, Q.max_pair_mass - 2.55)
        ))
        - 0.75 * grid(3, max(0.0, 0.0448
            - 621.0 * max(0.0, 0.00144 - Q.lam1)
            - 0.0391 * max(0.0, 51.5 - Q.n_particles)
            + 0.0385 * max(0.0, 26.6 - Q.sj2_mass1)
            - 5.86 * max(0.0, 0.384 - Q.tau21)
            + 146.0 * max(0.0, 0.0166 - Q.tau4)
            + 615.0 * max(0.0, 2.43 - Q.D2) * max(0.0, Q.psi_0p3 - 0.999)
            + 0.000126 * max(0.0, 53.2 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 762.0)
        ))
        - 0.875 * grid(4, max(0.0, 1.01
            - 171.0 * max(0.0, 0.00506 - Q.girth2_top15)
            + 0.123 * max(0.0, 73.7 - Q.mass)
            - 0.146 * max(0.0, 78.2 - Q.mass)
            - 0.0754 * max(0.0, 87.4 - Q.mass)
            + 0.0569 * max(0.0, 104.0 - Q.mass)
            + 0.0476 * max(0.0, 59.7 - Q.mass_top30)
            - 0.0248 * max(0.0, Q.n_particles - 22.4)
            + 0.0282 * max(0.0, Q.sj3_pair_mass_min - 33.1)
        ))
        - 0.3125 * grid(5, max(0.0, 0.0908
            - 180.0 * max(0.0, Q.girth2_top30 - 0.00755)
            - 26.0 * max(0.0, Q.log_sum_pt - 6.91)
            + 0.018 * max(0.0, Q.mass - 70.6)
            - 0.133 * max(0.0, Q.mass_top50 - 154.0)
            + 0.036 * max(0.0, 57.5 - Q.n_particles)
            + 0.0472 * max(0.0, Q.sd_mass - 71.9)
            - 0.0607 * max(0.0, Q.sd_mass - 88.0)
            - 0.0117 * max(0.0, Q.sum_pt_top40 - 1020.0)
            + 0.0287 * max(0.0, Q.sum_pt_top50 - 934.0)
            - 2.56 * max(0.0, 0.524 - Q.tau21)
            - 9.04 * max(0.0, Q.z_top30_slots - 0.898)
        ))
        + 0.15625 * grid(6, max(0.0, 1.15
            - 36.6 * max(0.0, Q.LHA - 0.37)
            - 34.4 * max(0.0, Q.e2 - 0.0274)
            + 286.0 * max(0.0, Q.girth2_top20 - 0.00754)
            + 0.133 * max(0.0, 92.6 - Q.mass)
            - 0.143 * max(0.0, 101.0 - Q.mass)
            + 0.0107 * max(0.0, 92.1 - Q.mass_top15)
            + 0.0375 * max(0.0, 72.6 - Q.mass_top50)
            - 12.2 * max(0.0, Q.sj3_dr_min - 0.119)
            + 3630.0 * max(0.0, Q.e2 - 0.0572) * max(0.0, Q.zdr_0 - 0.00437)
            - 6610.0 * max(0.0, Q.girth2_top20 - 0.00729) * max(0.0, Q.C2_b2 - -0.000102)
            + 2.21 * max(0.0, Q.sj3_dr_min - 0.112) * max(0.0, 7.8 - Q.sj3_mass2)
        ))
        + 0.03125 * grid(8, max(0.0, 0.659
            + 0.063 * max(0.0, Q.mass - 73.3)
            - 0.0714 * max(0.0, Q.mass - 104.0)
            - 0.0851 * max(0.0, 14.9 - Q.n_dr_0p2_0p4)
            + 0.07 * max(0.0, Q.n_particles - 50.1)
            - 118.0 * max(0.0, Q.psi_0p3 - 0.994)
            + 0.00662 * max(0.0, 1030.0 - Q.sum_pt)
            + 0.129 * max(0.0, Q.mass_over_sum_pt - 0.0758) * max(0.0, 1120.0 - Q.sum_pt)
            - 0.0676 * max(0.0, Q.n_particles - 50.0) * max(0.0, 1.05 - Q.soft1_pt)
        ))
        + 0.515625 * grid(9, max(0.0, 1.28
            + 343.0 * max(0.0, 0.00598 - Q.girth2_top40)
            - 0.0809 * max(0.0, 64.7 - Q.mass)
            + 0.115 * max(0.0, 85.1 - Q.mass)
            - 0.0308 * max(0.0, 136.0 - Q.mass)
            - 3.22 * max(0.0, Q.sj3_dr13 - 0.161)
            + 0.0162 * max(0.0, 984.0 - Q.sum_pt)
            + 6.83 * max(0.0, Q.z_top5 - 0.805)
            - 0.000222 * max(0.0, 84.5 - Q.mass_top40) * max(0.0, 1030.0 - Q.sum_pt)
            + 0.00693 * max(0.0, 960.0 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.92)
        ))
        - 0.015625 * grid(10, max(0.0, 1.62
            + 41.0 * max(0.0, Q.e2 - 0.0101)
            + 8130.0 * max(0.0, 0.000117 - Q.e3)
            - 0.157 * max(0.0, Q.mass - 148.0)
            + 0.124 * max(0.0, 89.3 - Q.mass)
            - 0.0667 * max(0.0, 102.0 - Q.mass)
            - 0.0235 * max(0.0, 70.6 - Q.mass_top15)
            - 0.00941 * max(0.0, 62.6 - Q.mass_top5)
            + 0.105 * max(0.0, Q.mass_top50 - 138.0)
            - 0.116 * max(0.0, Q.mass_top50 - 173.0)
            - 0.0511 * max(0.0, 11.9 - Q.n_dr_0p2_0p4)
            - 0.0103 * max(0.0, 67.4 - Q.sj2_mass1)
            - 0.00409 * max(0.0, 967.0 - Q.sum_pt_top50)
            - 10.8 * max(0.0, Q.z_dr_0_0p05 - 0.769)
            + 8.57 * max(0.0, 0.0949 - Q.z_dr_0p2_0p4)
            - 3.16 * max(0.0, Q.z_top2_slots - 0.298)
            + 3.02 * max(0.0, 0.319 - Q.z_top2_slots)
            - 16500.0 * max(0.0, 0.0201 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.998)
        ))
        + 0.234375 * grid(12, max(0.0, -0.402
            - 0.0599 * max(0.0, Q.mass - 162.0)
            - 0.107 * max(0.0, 60.9 - Q.mass)
            + 0.115 * max(0.0, 82.3 - Q.mass)
            - 3.83 * max(0.0, 92.5 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            - 18.5 * max(0.0, Q.sd_mass - 128.0) * max(0.0, Q.lam2 - 0.000638)
            + 0.379 * max(0.0, Q.sd_mass - 128.0) * max(0.0, 0.422 - Q.sd_zg)
            + 0.00116 * max(0.0, Q.sj3_pair_mass_max - 122.0) * max(0.0, 80.6 - Q.sj3_pair_mass_min)
        ))
    )


def logit_q(Q):
    return (1.359375
        - 0.1875 * grid(1, max(0.0, 3.08
            - 46.1 * max(0.0, 0.0312 - Q.M3)
            + 1590.0 * max(0.0, 0.000549 - Q.girth2_top3)
            + 20.3 * max(0.0, Q.log_sum_pt - 6.9)
            - 18.4 * max(0.0, Q.log_sum_pt - 6.99)
            - 10.9 * max(0.0, 7.16 - Q.log_sum_pt)
            - 0.11 * max(0.0, 6.23 - Q.n_dr_0p2_0p4)
            + 0.118 * max(0.0, Q.n_particles - 38.5)
            - 0.0252 * max(0.0, 31.5 - Q.sj3_mass1)
            + 0.019 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.00447 * max(0.0, 699.0 - Q.sum_pt_top2)
            + 0.00307 * max(0.0, 51.1 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 27.7)
            + 0.414 * max(0.0, Q.n_particles - 35.1) * max(0.0, 0.127 - Q.dr_0)
            - 0.0425 * max(0.0, Q.n_particles - 37.8) * max(0.0, 2.32 - Q.soft1_pt)
            - 0.0029 * max(0.0, 32.3 - Q.sj3_mass1) * max(0.0, 18.5 - Q.sj3_mass2)
            - 0.0075 * max(0.0, 706.0 - Q.sum_pt_top2) * max(0.0, 0.959 - Q.tau43)
            + 0.538 * max(0.0, Q.z_top30_slots - 0.947) * max(0.0, Q.max_pair_mass - 2.55)
        ))
        + 0.125 * grid(2, max(0.0, 0.0765
            + 200.0 * max(0.0, Q.log_sum_pt - 6.93) * max(0.0, 0.0271 - Q.girth2_top15)
            - 6.94 * max(0.0, Q.sum_pt_top20 - 1130.0) * max(0.0, Q.eta_1 - 0.0866)
        ))
        - 1.0625 * grid(4, max(0.0, 1.01
            - 171.0 * max(0.0, 0.00506 - Q.girth2_top15)
            + 0.123 * max(0.0, 73.7 - Q.mass)
            - 0.146 * max(0.0, 78.2 - Q.mass)
            - 0.0754 * max(0.0, 87.4 - Q.mass)
            + 0.0569 * max(0.0, 104.0 - Q.mass)
            + 0.0476 * max(0.0, 59.7 - Q.mass_top30)
            - 0.0248 * max(0.0, Q.n_particles - 22.4)
            + 0.0282 * max(0.0, Q.sj3_pair_mass_min - 33.1)
        ))
        + 0.21875 * grid(6, max(0.0, 1.15
            - 36.6 * max(0.0, Q.LHA - 0.37)
            - 34.4 * max(0.0, Q.e2 - 0.0274)
            + 286.0 * max(0.0, Q.girth2_top20 - 0.00754)
            + 0.133 * max(0.0, 92.6 - Q.mass)
            - 0.143 * max(0.0, 101.0 - Q.mass)
            + 0.0107 * max(0.0, 92.1 - Q.mass_top15)
            + 0.0375 * max(0.0, 72.6 - Q.mass_top50)
            - 12.2 * max(0.0, Q.sj3_dr_min - 0.119)
            + 3630.0 * max(0.0, Q.e2 - 0.0572) * max(0.0, Q.zdr_0 - 0.00437)
            - 6610.0 * max(0.0, Q.girth2_top20 - 0.00729) * max(0.0, Q.C2_b2 - -0.000102)
            + 2.21 * max(0.0, Q.sj3_dr_min - 0.112) * max(0.0, 7.8 - Q.sj3_mass2)
        ))
        + 0.015625 * grid(8, max(0.0, 0.659
            + 0.063 * max(0.0, Q.mass - 73.3)
            - 0.0714 * max(0.0, Q.mass - 104.0)
            - 0.0851 * max(0.0, 14.9 - Q.n_dr_0p2_0p4)
            + 0.07 * max(0.0, Q.n_particles - 50.1)
            - 118.0 * max(0.0, Q.psi_0p3 - 0.994)
            + 0.00662 * max(0.0, 1030.0 - Q.sum_pt)
            + 0.129 * max(0.0, Q.mass_over_sum_pt - 0.0758) * max(0.0, 1120.0 - Q.sum_pt)
            - 0.0676 * max(0.0, Q.n_particles - 50.0) * max(0.0, 1.05 - Q.soft1_pt)
        ))
        + 0.5625 * grid(9, max(0.0, 1.28
            + 343.0 * max(0.0, 0.00598 - Q.girth2_top40)
            - 0.0809 * max(0.0, 64.7 - Q.mass)
            + 0.115 * max(0.0, 85.1 - Q.mass)
            - 0.0308 * max(0.0, 136.0 - Q.mass)
            - 3.22 * max(0.0, Q.sj3_dr13 - 0.161)
            + 0.0162 * max(0.0, 984.0 - Q.sum_pt)
            + 6.83 * max(0.0, Q.z_top5 - 0.805)
            - 0.000222 * max(0.0, 84.5 - Q.mass_top40) * max(0.0, 1030.0 - Q.sum_pt)
            + 0.00693 * max(0.0, 960.0 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.92)
        ))
        - 0.015625 * grid(10, max(0.0, 1.62
            + 41.0 * max(0.0, Q.e2 - 0.0101)
            + 8130.0 * max(0.0, 0.000117 - Q.e3)
            - 0.157 * max(0.0, Q.mass - 148.0)
            + 0.124 * max(0.0, 89.3 - Q.mass)
            - 0.0667 * max(0.0, 102.0 - Q.mass)
            - 0.0235 * max(0.0, 70.6 - Q.mass_top15)
            - 0.00941 * max(0.0, 62.6 - Q.mass_top5)
            + 0.105 * max(0.0, Q.mass_top50 - 138.0)
            - 0.116 * max(0.0, Q.mass_top50 - 173.0)
            - 0.0511 * max(0.0, 11.9 - Q.n_dr_0p2_0p4)
            - 0.0103 * max(0.0, 67.4 - Q.sj2_mass1)
            - 0.00409 * max(0.0, 967.0 - Q.sum_pt_top50)
            - 10.8 * max(0.0, Q.z_dr_0_0p05 - 0.769)
            + 8.57 * max(0.0, 0.0949 - Q.z_dr_0p2_0p4)
            - 3.16 * max(0.0, Q.z_top2_slots - 0.298)
            + 3.02 * max(0.0, 0.319 - Q.z_top2_slots)
            - 16500.0 * max(0.0, 0.0201 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.998)
        ))
        - 0.25 * grid(11, max(0.0, -0.605
            + 86.3 * max(0.0, 0.0242 - Q.e2)
            - 304.0 * max(0.0, 0.00632 - Q.lam1)
            - 0.0995 * max(0.0, 81.2 - Q.mass)
            + 0.0887 * max(0.0, 93.7 - Q.mass)
            + 150.0 * max(0.0, Q.psi_0p2 - 0.994)
            + 80.3 * max(0.0, Q.psi_0p3 - 0.99)
            + 0.115 * max(0.0, 115.0 - Q.mass) * max(0.0, 0.351 - Q.planar_flow)
        ))
        + 0.34375 * grid(12, max(0.0, -0.402
            - 0.0599 * max(0.0, Q.mass - 162.0)
            - 0.107 * max(0.0, 60.9 - Q.mass)
            + 0.115 * max(0.0, 82.3 - Q.mass)
            - 3.83 * max(0.0, 92.5 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            - 18.5 * max(0.0, Q.sd_mass - 128.0) * max(0.0, Q.lam2 - 0.000638)
            + 0.379 * max(0.0, Q.sd_mass - 128.0) * max(0.0, 0.422 - Q.sd_zg)
            + 0.00116 * max(0.0, Q.sj3_pair_mass_max - 122.0) * max(0.0, 80.6 - Q.sj3_pair_mass_min)
        ))
    )


def logit_W(Q):
    return (0.09375
        + 0.75 * grid(0, max(0.0, 1.58
            - 662.0 * max(0.0, 0.00614 - Q.e2_sq)
            - 204.0 * max(0.0, 0.00639 - Q.girth2_top20)
            - 0.161 * max(0.0, Q.mass - 78.9)
            - 0.103 * max(0.0, Q.mass - 88.9)
            + 239.0 * max(0.0, 0.00498 - Q.mass_over_sum_pt_sq)
            + 621.0 * max(0.0, 0.00768 - Q.mass_over_sum_pt_sq)
            - 0.0101 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.0176 * max(0.0, 852.0 - Q.sum_pt_top40)
            - 26.2 * max(0.0, 0.0695 - Q.tau1)
            - 21.4 * max(0.0, 0.994 - Q.z_top50_slots)
        ))
        + 0.4375 * grid(3, max(0.0, 0.0448
            - 621.0 * max(0.0, 0.00144 - Q.lam1)
            - 0.0391 * max(0.0, 51.5 - Q.n_particles)
            + 0.0385 * max(0.0, 26.6 - Q.sj2_mass1)
            - 5.86 * max(0.0, 0.384 - Q.tau21)
            + 146.0 * max(0.0, 0.0166 - Q.tau4)
            + 615.0 * max(0.0, 2.43 - Q.D2) * max(0.0, Q.psi_0p3 - 0.999)
            + 0.000126 * max(0.0, 53.2 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 762.0)
        ))
        + 0.34375 * grid(4, max(0.0, 1.01
            - 171.0 * max(0.0, 0.00506 - Q.girth2_top15)
            + 0.123 * max(0.0, 73.7 - Q.mass)
            - 0.146 * max(0.0, 78.2 - Q.mass)
            - 0.0754 * max(0.0, 87.4 - Q.mass)
            + 0.0569 * max(0.0, 104.0 - Q.mass)
            + 0.0476 * max(0.0, 59.7 - Q.mass_top30)
            - 0.0248 * max(0.0, Q.n_particles - 22.4)
            + 0.0282 * max(0.0, Q.sj3_pair_mass_min - 33.1)
        ))
        + 0.578125 * grid(5, max(0.0, 0.0908
            - 180.0 * max(0.0, Q.girth2_top30 - 0.00755)
            - 26.0 * max(0.0, Q.log_sum_pt - 6.91)
            + 0.018 * max(0.0, Q.mass - 70.6)
            - 0.133 * max(0.0, Q.mass_top50 - 154.0)
            + 0.036 * max(0.0, 57.5 - Q.n_particles)
            + 0.0472 * max(0.0, Q.sd_mass - 71.9)
            - 0.0607 * max(0.0, Q.sd_mass - 88.0)
            - 0.0117 * max(0.0, Q.sum_pt_top40 - 1020.0)
            + 0.0287 * max(0.0, Q.sum_pt_top50 - 934.0)
            - 2.56 * max(0.0, 0.524 - Q.tau21)
            - 9.04 * max(0.0, Q.z_top30_slots - 0.898)
        ))
        + 0.0625 * grid(6, max(0.0, 1.15
            - 36.6 * max(0.0, Q.LHA - 0.37)
            - 34.4 * max(0.0, Q.e2 - 0.0274)
            + 286.0 * max(0.0, Q.girth2_top20 - 0.00754)
            + 0.133 * max(0.0, 92.6 - Q.mass)
            - 0.143 * max(0.0, 101.0 - Q.mass)
            + 0.0107 * max(0.0, 92.1 - Q.mass_top15)
            + 0.0375 * max(0.0, 72.6 - Q.mass_top50)
            - 12.2 * max(0.0, Q.sj3_dr_min - 0.119)
            + 3630.0 * max(0.0, Q.e2 - 0.0572) * max(0.0, Q.zdr_0 - 0.00437)
            - 6610.0 * max(0.0, Q.girth2_top20 - 0.00729) * max(0.0, Q.C2_b2 - -0.000102)
            + 2.21 * max(0.0, Q.sj3_dr_min - 0.112) * max(0.0, 7.8 - Q.sj3_mass2)
        ))
        - 0.625 * grid(7, max(0.0, 0.241
            + 0.0354 * max(0.0, 69.8 - Q.sd_mass)
            - 0.0263 * max(0.0, 85.7 - Q.sd_mass)
            - 17.4 * max(0.0, 0.106 - Q.tau1)
            - 4.83 * max(0.0, 0.394 - Q.tau21)
            + 8.91 * max(0.0, 0.248 - Q.tau21_b2)
            - 136.0 * max(0.0, 83.1 - Q.mass) * max(0.0, Q.psi_0p3 - 0.998)
            - 14.7 * max(0.0, 91.4 - Q.mass) * max(0.0, Q.psi_0p3 - 0.975)
            - 6.9 * max(0.0, 93.8 - Q.mass) * max(0.0, Q.psi_0p3 - 0.967)
            + 26.7 * max(0.0, 98.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997)
            + 12.4 * max(0.0, 102.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.977)
        ))
        - 0.875 * grid(8, max(0.0, 0.659
            + 0.063 * max(0.0, Q.mass - 73.3)
            - 0.0714 * max(0.0, Q.mass - 104.0)
            - 0.0851 * max(0.0, 14.9 - Q.n_dr_0p2_0p4)
            + 0.07 * max(0.0, Q.n_particles - 50.1)
            - 118.0 * max(0.0, Q.psi_0p3 - 0.994)
            + 0.00662 * max(0.0, 1030.0 - Q.sum_pt)
            + 0.129 * max(0.0, Q.mass_over_sum_pt - 0.0758) * max(0.0, 1120.0 - Q.sum_pt)
            - 0.0676 * max(0.0, Q.n_particles - 50.0) * max(0.0, 1.05 - Q.soft1_pt)
        ))
        - 0.21875 * grid(9, max(0.0, 1.28
            + 343.0 * max(0.0, 0.00598 - Q.girth2_top40)
            - 0.0809 * max(0.0, 64.7 - Q.mass)
            + 0.115 * max(0.0, 85.1 - Q.mass)
            - 0.0308 * max(0.0, 136.0 - Q.mass)
            - 3.22 * max(0.0, Q.sj3_dr13 - 0.161)
            + 0.0162 * max(0.0, 984.0 - Q.sum_pt)
            + 6.83 * max(0.0, Q.z_top5 - 0.805)
            - 0.000222 * max(0.0, 84.5 - Q.mass_top40) * max(0.0, 1030.0 - Q.sum_pt)
            + 0.00693 * max(0.0, 960.0 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.92)
        ))
        + 0.59375 * grid(11, max(0.0, -0.605
            + 86.3 * max(0.0, 0.0242 - Q.e2)
            - 304.0 * max(0.0, 0.00632 - Q.lam1)
            - 0.0995 * max(0.0, 81.2 - Q.mass)
            + 0.0887 * max(0.0, 93.7 - Q.mass)
            + 150.0 * max(0.0, Q.psi_0p2 - 0.994)
            + 80.3 * max(0.0, Q.psi_0p3 - 0.99)
            + 0.115 * max(0.0, 115.0 - Q.mass) * max(0.0, 0.351 - Q.planar_flow)
        ))
        - 0.40625 * grid(12, max(0.0, -0.402
            - 0.0599 * max(0.0, Q.mass - 162.0)
            - 0.107 * max(0.0, 60.9 - Q.mass)
            + 0.115 * max(0.0, 82.3 - Q.mass)
            - 3.83 * max(0.0, 92.5 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            - 18.5 * max(0.0, Q.sd_mass - 128.0) * max(0.0, Q.lam2 - 0.000638)
            + 0.379 * max(0.0, Q.sd_mass - 128.0) * max(0.0, 0.422 - Q.sd_zg)
            + 0.00116 * max(0.0, Q.sj3_pair_mass_max - 122.0) * max(0.0, 80.6 - Q.sj3_pair_mass_min)
        ))
        - 1.375 * grid(14, max(0.0, -0.714
            + 76.3 * max(0.0, 0.006 - Q.girth2_top5)
            + 20.0 * max(0.0, 6.94 - Q.log_sum_pt)
            - 0.124 * max(0.0, 82.7 - Q.mass)
            - 0.117 * max(0.0, 90.8 - Q.mass)
            - 70.9 * max(0.0, 0.0775 - Q.mass_over_sum_pt)
            + 38.8 * max(0.0, 0.118 - Q.mass_over_sum_pt)
            + 174.0 * max(0.0, Q.psi_0p3 - 0.993)
            - 0.0264 * max(0.0, 973.0 - Q.sum_pt_top50)
            + 61.5 * max(0.0, 0.0657 - Q.tau1)
            - 8.51 * max(0.0, 0.455 - Q.N2) * max(0.0, Q.max_dr - 0.244)
            - 676.0 * max(0.0, Q.psi_0p3 - 0.993) * max(0.0, 0.235 - Q.sd_rg)
            - 1470.0 * max(0.0, 0.331 - Q.tau21_b2) * max(0.0, 0.00769 - Q.girth2_top40)
        ))
        - 0.1875 * grid(15, max(0.0, -0.268
            + 151.0 * max(0.0, 0.0102 - Q.girth2_top10)
            - 57.5 * max(0.0, Q.psi_0p3 - 0.989)
            - 22.4 * max(0.0, 0.0674 - Q.tau1)
            + 6.5 * max(0.0, 0.142 - Q.z_dr_0p1_0p2)
            + 392.0 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0397 - Q.C2_b2)
            - 0.433 * max(0.0, 0.139 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.49)
        ))
    )


def logit_Z(Q):
    return (0.984375
        - 1.375 * grid(0, max(0.0, 1.58
            - 662.0 * max(0.0, 0.00614 - Q.e2_sq)
            - 204.0 * max(0.0, 0.00639 - Q.girth2_top20)
            - 0.161 * max(0.0, Q.mass - 78.9)
            - 0.103 * max(0.0, Q.mass - 88.9)
            + 239.0 * max(0.0, 0.00498 - Q.mass_over_sum_pt_sq)
            + 621.0 * max(0.0, 0.00768 - Q.mass_over_sum_pt_sq)
            - 0.0101 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.0176 * max(0.0, 852.0 - Q.sum_pt_top40)
            - 26.2 * max(0.0, 0.0695 - Q.tau1)
            - 21.4 * max(0.0, 0.994 - Q.z_top50_slots)
        ))
        - 0.375 * grid(2, max(0.0, 0.0765
            + 200.0 * max(0.0, Q.log_sum_pt - 6.93) * max(0.0, 0.0271 - Q.girth2_top15)
            - 6.94 * max(0.0, Q.sum_pt_top20 - 1130.0) * max(0.0, Q.eta_1 - 0.0866)
        ))
        + 0.4375 * grid(3, max(0.0, 0.0448
            - 621.0 * max(0.0, 0.00144 - Q.lam1)
            - 0.0391 * max(0.0, 51.5 - Q.n_particles)
            + 0.0385 * max(0.0, 26.6 - Q.sj2_mass1)
            - 5.86 * max(0.0, 0.384 - Q.tau21)
            + 146.0 * max(0.0, 0.0166 - Q.tau4)
            + 615.0 * max(0.0, 2.43 - Q.D2) * max(0.0, Q.psi_0p3 - 0.999)
            + 0.000126 * max(0.0, 53.2 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 762.0)
        ))
        + 0.6875 * grid(5, max(0.0, 0.0908
            - 180.0 * max(0.0, Q.girth2_top30 - 0.00755)
            - 26.0 * max(0.0, Q.log_sum_pt - 6.91)
            + 0.018 * max(0.0, Q.mass - 70.6)
            - 0.133 * max(0.0, Q.mass_top50 - 154.0)
            + 0.036 * max(0.0, 57.5 - Q.n_particles)
            + 0.0472 * max(0.0, Q.sd_mass - 71.9)
            - 0.0607 * max(0.0, Q.sd_mass - 88.0)
            - 0.0117 * max(0.0, Q.sum_pt_top40 - 1020.0)
            + 0.0287 * max(0.0, Q.sum_pt_top50 - 934.0)
            - 2.56 * max(0.0, 0.524 - Q.tau21)
            - 9.04 * max(0.0, Q.z_top30_slots - 0.898)
        ))
        - 0.96875 * grid(6, max(0.0, 1.15
            - 36.6 * max(0.0, Q.LHA - 0.37)
            - 34.4 * max(0.0, Q.e2 - 0.0274)
            + 286.0 * max(0.0, Q.girth2_top20 - 0.00754)
            + 0.133 * max(0.0, 92.6 - Q.mass)
            - 0.143 * max(0.0, 101.0 - Q.mass)
            + 0.0107 * max(0.0, 92.1 - Q.mass_top15)
            + 0.0375 * max(0.0, 72.6 - Q.mass_top50)
            - 12.2 * max(0.0, Q.sj3_dr_min - 0.119)
            + 3630.0 * max(0.0, Q.e2 - 0.0572) * max(0.0, Q.zdr_0 - 0.00437)
            - 6610.0 * max(0.0, Q.girth2_top20 - 0.00729) * max(0.0, Q.C2_b2 - -0.000102)
            + 2.21 * max(0.0, Q.sj3_dr_min - 0.112) * max(0.0, 7.8 - Q.sj3_mass2)
        ))
        + 0.90625 * grid(7, max(0.0, 0.241
            + 0.0354 * max(0.0, 69.8 - Q.sd_mass)
            - 0.0263 * max(0.0, 85.7 - Q.sd_mass)
            - 17.4 * max(0.0, 0.106 - Q.tau1)
            - 4.83 * max(0.0, 0.394 - Q.tau21)
            + 8.91 * max(0.0, 0.248 - Q.tau21_b2)
            - 136.0 * max(0.0, 83.1 - Q.mass) * max(0.0, Q.psi_0p3 - 0.998)
            - 14.7 * max(0.0, 91.4 - Q.mass) * max(0.0, Q.psi_0p3 - 0.975)
            - 6.9 * max(0.0, 93.8 - Q.mass) * max(0.0, Q.psi_0p3 - 0.967)
            + 26.7 * max(0.0, 98.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997)
            + 12.4 * max(0.0, 102.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.977)
        ))
        - 0.9375 * grid(8, max(0.0, 0.659
            + 0.063 * max(0.0, Q.mass - 73.3)
            - 0.0714 * max(0.0, Q.mass - 104.0)
            - 0.0851 * max(0.0, 14.9 - Q.n_dr_0p2_0p4)
            + 0.07 * max(0.0, Q.n_particles - 50.1)
            - 118.0 * max(0.0, Q.psi_0p3 - 0.994)
            + 0.00662 * max(0.0, 1030.0 - Q.sum_pt)
            + 0.129 * max(0.0, Q.mass_over_sum_pt - 0.0758) * max(0.0, 1120.0 - Q.sum_pt)
            - 0.0676 * max(0.0, Q.n_particles - 50.0) * max(0.0, 1.05 - Q.soft1_pt)
        ))
        + 0.3125 * grid(12, max(0.0, -0.402
            - 0.0599 * max(0.0, Q.mass - 162.0)
            - 0.107 * max(0.0, 60.9 - Q.mass)
            + 0.115 * max(0.0, 82.3 - Q.mass)
            - 3.83 * max(0.0, 92.5 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            - 18.5 * max(0.0, Q.sd_mass - 128.0) * max(0.0, Q.lam2 - 0.000638)
            + 0.379 * max(0.0, Q.sd_mass - 128.0) * max(0.0, 0.422 - Q.sd_zg)
            + 0.00116 * max(0.0, Q.sj3_pair_mass_max - 122.0) * max(0.0, 80.6 - Q.sj3_pair_mass_min)
        ))
        + 0.5625 * grid(15, max(0.0, -0.268
            + 151.0 * max(0.0, 0.0102 - Q.girth2_top10)
            - 57.5 * max(0.0, Q.psi_0p3 - 0.989)
            - 22.4 * max(0.0, 0.0674 - Q.tau1)
            + 6.5 * max(0.0, 0.142 - Q.z_dr_0p1_0p2)
            + 392.0 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0397 - Q.C2_b2)
            - 0.433 * max(0.0, 0.139 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.49)
        ))
    )


def logit_t(Q):
    return (0.78125
        + 0.125 * grid(0, max(0.0, 1.58
            - 662.0 * max(0.0, 0.00614 - Q.e2_sq)
            - 204.0 * max(0.0, 0.00639 - Q.girth2_top20)
            - 0.161 * max(0.0, Q.mass - 78.9)
            - 0.103 * max(0.0, Q.mass - 88.9)
            + 239.0 * max(0.0, 0.00498 - Q.mass_over_sum_pt_sq)
            + 621.0 * max(0.0, 0.00768 - Q.mass_over_sum_pt_sq)
            - 0.0101 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.0176 * max(0.0, 852.0 - Q.sum_pt_top40)
            - 26.2 * max(0.0, 0.0695 - Q.tau1)
            - 21.4 * max(0.0, 0.994 - Q.z_top50_slots)
        ))
        + 0.1875 * grid(4, max(0.0, 1.01
            - 171.0 * max(0.0, 0.00506 - Q.girth2_top15)
            + 0.123 * max(0.0, 73.7 - Q.mass)
            - 0.146 * max(0.0, 78.2 - Q.mass)
            - 0.0754 * max(0.0, 87.4 - Q.mass)
            + 0.0569 * max(0.0, 104.0 - Q.mass)
            + 0.0476 * max(0.0, 59.7 - Q.mass_top30)
            - 0.0248 * max(0.0, Q.n_particles - 22.4)
            + 0.0282 * max(0.0, Q.sj3_pair_mass_min - 33.1)
        ))
        - 0.46875 * grid(5, max(0.0, 0.0908
            - 180.0 * max(0.0, Q.girth2_top30 - 0.00755)
            - 26.0 * max(0.0, Q.log_sum_pt - 6.91)
            + 0.018 * max(0.0, Q.mass - 70.6)
            - 0.133 * max(0.0, Q.mass_top50 - 154.0)
            + 0.036 * max(0.0, 57.5 - Q.n_particles)
            + 0.0472 * max(0.0, Q.sd_mass - 71.9)
            - 0.0607 * max(0.0, Q.sd_mass - 88.0)
            - 0.0117 * max(0.0, Q.sum_pt_top40 - 1020.0)
            + 0.0287 * max(0.0, Q.sum_pt_top50 - 934.0)
            - 2.56 * max(0.0, 0.524 - Q.tau21)
            - 9.04 * max(0.0, Q.z_top30_slots - 0.898)
        ))
        - 0.28125 * grid(7, max(0.0, 0.241
            + 0.0354 * max(0.0, 69.8 - Q.sd_mass)
            - 0.0263 * max(0.0, 85.7 - Q.sd_mass)
            - 17.4 * max(0.0, 0.106 - Q.tau1)
            - 4.83 * max(0.0, 0.394 - Q.tau21)
            + 8.91 * max(0.0, 0.248 - Q.tau21_b2)
            - 136.0 * max(0.0, 83.1 - Q.mass) * max(0.0, Q.psi_0p3 - 0.998)
            - 14.7 * max(0.0, 91.4 - Q.mass) * max(0.0, Q.psi_0p3 - 0.975)
            - 6.9 * max(0.0, 93.8 - Q.mass) * max(0.0, Q.psi_0p3 - 0.967)
            + 26.7 * max(0.0, 98.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997)
            + 12.4 * max(0.0, 102.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.977)
        ))
        + 0.2109375 * grid(8, max(0.0, 0.659
            + 0.063 * max(0.0, Q.mass - 73.3)
            - 0.0714 * max(0.0, Q.mass - 104.0)
            - 0.0851 * max(0.0, 14.9 - Q.n_dr_0p2_0p4)
            + 0.07 * max(0.0, Q.n_particles - 50.1)
            - 118.0 * max(0.0, Q.psi_0p3 - 0.994)
            + 0.00662 * max(0.0, 1030.0 - Q.sum_pt)
            + 0.129 * max(0.0, Q.mass_over_sum_pt - 0.0758) * max(0.0, 1120.0 - Q.sum_pt)
            - 0.0676 * max(0.0, Q.n_particles - 50.0) * max(0.0, 1.05 - Q.soft1_pt)
        ))
        + 0.984375 * grid(10, max(0.0, 1.62
            + 41.0 * max(0.0, Q.e2 - 0.0101)
            + 8130.0 * max(0.0, 0.000117 - Q.e3)
            - 0.157 * max(0.0, Q.mass - 148.0)
            + 0.124 * max(0.0, 89.3 - Q.mass)
            - 0.0667 * max(0.0, 102.0 - Q.mass)
            - 0.0235 * max(0.0, 70.6 - Q.mass_top15)
            - 0.00941 * max(0.0, 62.6 - Q.mass_top5)
            + 0.105 * max(0.0, Q.mass_top50 - 138.0)
            - 0.116 * max(0.0, Q.mass_top50 - 173.0)
            - 0.0511 * max(0.0, 11.9 - Q.n_dr_0p2_0p4)
            - 0.0103 * max(0.0, 67.4 - Q.sj2_mass1)
            - 0.00409 * max(0.0, 967.0 - Q.sum_pt_top50)
            - 10.8 * max(0.0, Q.z_dr_0_0p05 - 0.769)
            + 8.57 * max(0.0, 0.0949 - Q.z_dr_0p2_0p4)
            - 3.16 * max(0.0, Q.z_top2_slots - 0.298)
            + 3.02 * max(0.0, 0.319 - Q.z_top2_slots)
            - 16500.0 * max(0.0, 0.0201 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.998)
        ))
        - 0.375 * grid(12, max(0.0, -0.402
            - 0.0599 * max(0.0, Q.mass - 162.0)
            - 0.107 * max(0.0, 60.9 - Q.mass)
            + 0.115 * max(0.0, 82.3 - Q.mass)
            - 3.83 * max(0.0, 92.5 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            - 18.5 * max(0.0, Q.sd_mass - 128.0) * max(0.0, Q.lam2 - 0.000638)
            + 0.379 * max(0.0, Q.sd_mass - 128.0) * max(0.0, 0.422 - Q.sd_zg)
            + 0.00116 * max(0.0, Q.sj3_pair_mass_max - 122.0) * max(0.0, 80.6 - Q.sj3_pair_mass_min)
        ))
        - 0.90625 * grid(13, max(0.0, 2.58
            - 0.12 * max(0.0, Q.mass - 142.0)
            - 0.158 * max(0.0, Q.mass - 160.0)
            + 0.192 * max(0.0, Q.mass - 172.0)
            - 0.0205 * max(0.0, Q.mass_top50 - 99.2)
            + 0.13 * max(0.0, Q.mass_top50 - 137.0)
            - 0.0202 * max(0.0, 1010.0 - Q.sum_pt)
            - 0.0102 * max(0.0, 1090.0 - Q.sum_pt)
            + 0.00816 * max(0.0, 1020.0 - Q.sum_pt_top40)
            + 1620.0 * max(0.0, Q.log_sum_pt - 7.14) * max(0.0, Q.sd_zg - 0.446)
        ))
        - 0.375 * grid(15, max(0.0, -0.268
            + 151.0 * max(0.0, 0.0102 - Q.girth2_top10)
            - 57.5 * max(0.0, Q.psi_0p3 - 0.989)
            - 22.4 * max(0.0, 0.0674 - Q.tau1)
            + 6.5 * max(0.0, 0.142 - Q.z_dr_0p1_0p2)
            + 392.0 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0397 - Q.C2_b2)
            - 0.433 * max(0.0, 0.139 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.49)
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
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
