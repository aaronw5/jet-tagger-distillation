"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no W/Z/H/t mass values offered as thresholds), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.5% (the network: 81.1%); same class as the network for 91.7% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
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
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top20=mass_of(20),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        soft1_pt=softp(1, 'pt'),
        soft5_pt=softp(5, 'pt'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_mass=softdrop("mass"),
        sj2_dr=subjets(2)["dr"][0],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
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
        + 0.625 * grid(1, max(0.0, 3.56
            - 35.9 * max(0.0, 0.0323 - Q.M3)
            + 1190.0 * max(0.0, 0.000745 - Q.girth2_top15)
            + 37.9 * max(0.0, Q.log_sum_pt - 6.9)
            - 22.8 * max(0.0, Q.log_sum_pt - 6.99)
            - 8.51 * max(0.0, 7.14 - Q.log_sum_pt)
            + 0.126 * max(0.0, Q.n_particles - 37.8)
            - 0.0349 * max(0.0, 33.9 - Q.sj3_mass1)
            + 0.0158 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.00311 * max(0.0, 701.0 - Q.sum_pt_top2)
            + 0.00656 * max(0.0, Q.sum_pt_top30 - 1180.0)
            - 0.0143 * max(0.0, Q.sum_pt_top50 - 959.0)
            - 53.2 * max(0.0, Q.zdr_0 - 0.00156)
            + 0.0035 * max(0.0, 56.7 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.2)
            - 0.0315 * max(0.0, Q.n_particles - 38.3) * max(0.0, 2.31 - Q.soft1_pt)
            - 0.00257 * max(0.0, 33.4 - Q.sj3_mass1) * max(0.0, 18.4 - Q.sj3_mass2)
            - 0.00754 * max(0.0, 703.0 - Q.sum_pt_top2) * max(0.0, 0.965 - Q.tau43)
            + 0.815 * max(0.0, Q.z_top30_slots - 0.944) * max(0.0, Q.max_pair_mass - 7.47)
        ))
        - 0.75 * grid(3, max(0.0, 0.0826
            + 0.127 * max(0.0, 4.64 - Q.n_dr_0p2_0p4)
            + 161.0 * max(0.0, 0.0155 - Q.tau4)
            - 1440.0 * max(0.0, 0.317 - Q.tau21) * max(0.0, Q.lam1 - 0.00429)
        ))
        - 0.875 * grid(4, max(0.0, 1.28
            - 173.0 * max(0.0, 0.00523 - Q.girth2_top15)
            + 0.047 * max(0.0, 69.8 - Q.mass_top40)
            - 0.0344 * max(0.0, Q.n_particles - 22.5)
            + 1.91 * max(0.0, 0.324 - Q.planar_flow)
            + 0.0238 * max(0.0, Q.sj3_pair_mass_min - 33.4)
            - 36.4 * max(0.0, 80.7 - Q.mass) * max(0.0, 0.0041 - Q.lam2)
            + 12.0 * max(0.0, 104.0 - Q.mass) * max(0.0, 0.00334 - Q.lam2)
            - 0.00338 * max(0.0, 91.8 - Q.mass_top40) * max(0.0, 9.1 - Q.D2)
            + 0.00834 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.22 - Q.soft1_pt)
            - 289.0 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0309 - Q.C2_b2)
        ))
        - 0.3125 * grid(5, max(0.0, 2.33
            - 129.0 * max(0.0, Q.girth2_top30 - 0.00764)
            - 37.5 * max(0.0, Q.log_sum_pt - 6.92)
            + 0.0258 * max(0.0, 70.3 - Q.mass)
            - 0.0214 * max(0.0, 159.0 - Q.mass_top40)
            - 0.402 * max(0.0, Q.mass_top50 - 157.0)
            - 4.11 * max(0.0, Q.max_dr - 0.267)
            + 5.44 * max(0.0, Q.max_dr - 0.435)
            + 0.0338 * max(0.0, 54.1 - Q.n_particles)
            + 0.0253 * max(0.0, Q.sum_pt_top50 - 917.0)
            - 3.37 * max(0.0, 0.469 - Q.tau21)
            - 9.3 * max(0.0, Q.z_top30_slots - 0.895)
        ))
        + 0.15625 * grid(6, max(0.0, -0.458
            - 0.35 * max(0.0, 2.23 - Q.D2)
            + 0.0898 * max(0.0, Q.mass - 83.8)
            - 0.0971 * max(0.0, Q.mass - 106.0)
            + 0.0359 * max(0.0, 93.8 - Q.mass)
        ))
        + 0.03125 * grid(8, max(0.0, 2.08
            + 9.91 * max(0.0, Q.log_sum_pt - 6.96)
            - 0.0823 * max(0.0, Q.mass - 106.0)
            + 83.4 * max(0.0, Q.mass_over_sum_pt - 0.0766)
            - 0.12 * max(0.0, 16.9 - Q.n_dr_0p2_0p4)
            - 0.00768 * max(0.0, Q.sum_pt_top40 - 887.0)
            + 0.0756 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.sj3_pair_mass_max - 37.0)
        ))
        + 0.515625 * grid(9, max(0.0, 0.174
            - 5600.0 * max(0.0, Q.e3 - 0.000341)
            + 0.0167 * max(0.0, Q.mass - 110.0)
            - 0.0721 * max(0.0, Q.mass - 168.0)
            - 0.0803 * max(0.0, 64.1 - Q.mass)
            + 0.141 * max(0.0, 81.7 - Q.mass)
            + 0.0121 * max(0.0, 960.0 - Q.sum_pt_top50)
            - 21.0 * max(0.0, Q.z_top40_slots - 0.961)
            - 0.00016 * max(0.0, 95.5 - Q.mass_top40) * max(0.0, 1050.0 - Q.sum_pt)
            + 0.00423 * max(0.0, 1010.0 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.98)
        ))
        - 0.015625 * grid(10, max(0.0, 1.4
            + 0.512 * max(0.0, 3.84 - Q.D2)
            - 0.0624 * max(0.0, Q.mass - 149.0)
            - 0.286 * max(0.0, Q.mass - 172.0)
            + 0.0569 * max(0.0, 80.8 - Q.mass)
            + 0.132 * max(0.0, Q.mass_top50 - 156.0)
            - 0.0511 * max(0.0, 82.7 - Q.sj2_mass1)
            - 0.00483 * max(0.0, 980.0 - Q.sum_pt)
            + 0.00292 * max(0.0, 880.0 - Q.sum_pt_top5)
            + 2.51 * max(0.0, 0.604 - Q.tau21_b2)
            - 9.99 * max(0.0, Q.z_dr_0_0p05 - 0.768)
            - 1.81 * max(0.0, 3.83 - Q.D2) * max(0.0, Q.sj2_dr - 0.196)
            + 2.67 * max(0.0, Q.mass - 173.0) * max(0.0, Q.M2 - 0.0406)
            + 0.000955 * max(0.0, 81.5 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.9)
        ))
        + 0.234375 * grid(12, max(0.0, -0.127
            - 870.0 * max(0.0, 0.00289 - Q.e2_sq)
            + 0.0819 * max(0.0, 82.6 - Q.mass)
            - 6.07 * max(0.0, 83.6 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            + 0.00173 * max(0.0, Q.sj3_pair_mass_max - 126.0) * max(0.0, 52.1 - Q.sj2_mass1)
        ))
    )


def logit_q(Q):
    return (1.359375
        - 0.1875 * grid(1, max(0.0, 3.56
            - 35.9 * max(0.0, 0.0323 - Q.M3)
            + 1190.0 * max(0.0, 0.000745 - Q.girth2_top15)
            + 37.9 * max(0.0, Q.log_sum_pt - 6.9)
            - 22.8 * max(0.0, Q.log_sum_pt - 6.99)
            - 8.51 * max(0.0, 7.14 - Q.log_sum_pt)
            + 0.126 * max(0.0, Q.n_particles - 37.8)
            - 0.0349 * max(0.0, 33.9 - Q.sj3_mass1)
            + 0.0158 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.00311 * max(0.0, 701.0 - Q.sum_pt_top2)
            + 0.00656 * max(0.0, Q.sum_pt_top30 - 1180.0)
            - 0.0143 * max(0.0, Q.sum_pt_top50 - 959.0)
            - 53.2 * max(0.0, Q.zdr_0 - 0.00156)
            + 0.0035 * max(0.0, 56.7 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.2)
            - 0.0315 * max(0.0, Q.n_particles - 38.3) * max(0.0, 2.31 - Q.soft1_pt)
            - 0.00257 * max(0.0, 33.4 - Q.sj3_mass1) * max(0.0, 18.4 - Q.sj3_mass2)
            - 0.00754 * max(0.0, 703.0 - Q.sum_pt_top2) * max(0.0, 0.965 - Q.tau43)
            + 0.815 * max(0.0, Q.z_top30_slots - 0.944) * max(0.0, Q.max_pair_mass - 7.47)
        ))
        + 0.125 * grid(2, max(0.0, 1.22
            - 0.00446 * max(0.0, 1270.0 - Q.sum_pt)
        ))
        - 1.0625 * grid(4, max(0.0, 1.28
            - 173.0 * max(0.0, 0.00523 - Q.girth2_top15)
            + 0.047 * max(0.0, 69.8 - Q.mass_top40)
            - 0.0344 * max(0.0, Q.n_particles - 22.5)
            + 1.91 * max(0.0, 0.324 - Q.planar_flow)
            + 0.0238 * max(0.0, Q.sj3_pair_mass_min - 33.4)
            - 36.4 * max(0.0, 80.7 - Q.mass) * max(0.0, 0.0041 - Q.lam2)
            + 12.0 * max(0.0, 104.0 - Q.mass) * max(0.0, 0.00334 - Q.lam2)
            - 0.00338 * max(0.0, 91.8 - Q.mass_top40) * max(0.0, 9.1 - Q.D2)
            + 0.00834 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.22 - Q.soft1_pt)
            - 289.0 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0309 - Q.C2_b2)
        ))
        + 0.21875 * grid(6, max(0.0, -0.458
            - 0.35 * max(0.0, 2.23 - Q.D2)
            + 0.0898 * max(0.0, Q.mass - 83.8)
            - 0.0971 * max(0.0, Q.mass - 106.0)
            + 0.0359 * max(0.0, 93.8 - Q.mass)
        ))
        + 0.015625 * grid(8, max(0.0, 2.08
            + 9.91 * max(0.0, Q.log_sum_pt - 6.96)
            - 0.0823 * max(0.0, Q.mass - 106.0)
            + 83.4 * max(0.0, Q.mass_over_sum_pt - 0.0766)
            - 0.12 * max(0.0, 16.9 - Q.n_dr_0p2_0p4)
            - 0.00768 * max(0.0, Q.sum_pt_top40 - 887.0)
            + 0.0756 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.sj3_pair_mass_max - 37.0)
        ))
        + 0.5625 * grid(9, max(0.0, 0.174
            - 5600.0 * max(0.0, Q.e3 - 0.000341)
            + 0.0167 * max(0.0, Q.mass - 110.0)
            - 0.0721 * max(0.0, Q.mass - 168.0)
            - 0.0803 * max(0.0, 64.1 - Q.mass)
            + 0.141 * max(0.0, 81.7 - Q.mass)
            + 0.0121 * max(0.0, 960.0 - Q.sum_pt_top50)
            - 21.0 * max(0.0, Q.z_top40_slots - 0.961)
            - 0.00016 * max(0.0, 95.5 - Q.mass_top40) * max(0.0, 1050.0 - Q.sum_pt)
            + 0.00423 * max(0.0, 1010.0 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.98)
        ))
        - 0.015625 * grid(10, max(0.0, 1.4
            + 0.512 * max(0.0, 3.84 - Q.D2)
            - 0.0624 * max(0.0, Q.mass - 149.0)
            - 0.286 * max(0.0, Q.mass - 172.0)
            + 0.0569 * max(0.0, 80.8 - Q.mass)
            + 0.132 * max(0.0, Q.mass_top50 - 156.0)
            - 0.0511 * max(0.0, 82.7 - Q.sj2_mass1)
            - 0.00483 * max(0.0, 980.0 - Q.sum_pt)
            + 0.00292 * max(0.0, 880.0 - Q.sum_pt_top5)
            + 2.51 * max(0.0, 0.604 - Q.tau21_b2)
            - 9.99 * max(0.0, Q.z_dr_0_0p05 - 0.768)
            - 1.81 * max(0.0, 3.83 - Q.D2) * max(0.0, Q.sj2_dr - 0.196)
            + 2.67 * max(0.0, Q.mass - 173.0) * max(0.0, Q.M2 - 0.0406)
            + 0.000955 * max(0.0, 81.5 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.9)
        ))
        - 0.25 * grid(11, max(0.0, 0.0886
            - 0.0154 * max(0.0, 79.4 - Q.mass)
            + 0.0504 * max(0.0, 15.7 - Q.n_dr_0p1_0p2)
            + 0.0989 * max(0.0, 10.3 - Q.n_dr_0p2_0p4)
            + 119.0 * max(0.0, Q.psi_0p2 - 0.992)
            - 0.0708 * max(0.0, 78.5 - Q.mass) * max(0.0, 3.87 - Q.D2)
        ))
        + 0.34375 * grid(12, max(0.0, -0.127
            - 870.0 * max(0.0, 0.00289 - Q.e2_sq)
            + 0.0819 * max(0.0, 82.6 - Q.mass)
            - 6.07 * max(0.0, 83.6 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            + 0.00173 * max(0.0, Q.sj3_pair_mass_max - 126.0) * max(0.0, 52.1 - Q.sj2_mass1)
        ))
    )


def logit_W(Q):
    return (0.09375
        + 0.75 * grid(0, max(0.0, 1.51
            - 629.0 * max(0.0, 0.00619 - Q.e2_sq)
            - 183.0 * max(0.0, 0.0061 - Q.girth2_top20)
            - 0.162 * max(0.0, Q.mass - 78.3)
            - 0.182 * max(0.0, Q.mass - 87.4)
            + 659.0 * max(0.0, 0.00777 - Q.mass_over_sum_pt_sq)
            + 182.0 * max(0.0, Q.psi_0p3 - 0.996)
            - 0.0101 * max(0.0, 1010.0 - Q.sum_pt)
            - 18.3 * max(0.0, 0.0693 - Q.tau1)
            - 5.38 * max(0.0, 0.0912 - Q.z_dr_0p2_0p4)
            - 26.1 * max(0.0, 0.992 - Q.z_top50_slots)
        ))
        + 0.4375 * grid(3, max(0.0, 0.0826
            + 0.127 * max(0.0, 4.64 - Q.n_dr_0p2_0p4)
            + 161.0 * max(0.0, 0.0155 - Q.tau4)
            - 1440.0 * max(0.0, 0.317 - Q.tau21) * max(0.0, Q.lam1 - 0.00429)
        ))
        + 0.34375 * grid(4, max(0.0, 1.28
            - 173.0 * max(0.0, 0.00523 - Q.girth2_top15)
            + 0.047 * max(0.0, 69.8 - Q.mass_top40)
            - 0.0344 * max(0.0, Q.n_particles - 22.5)
            + 1.91 * max(0.0, 0.324 - Q.planar_flow)
            + 0.0238 * max(0.0, Q.sj3_pair_mass_min - 33.4)
            - 36.4 * max(0.0, 80.7 - Q.mass) * max(0.0, 0.0041 - Q.lam2)
            + 12.0 * max(0.0, 104.0 - Q.mass) * max(0.0, 0.00334 - Q.lam2)
            - 0.00338 * max(0.0, 91.8 - Q.mass_top40) * max(0.0, 9.1 - Q.D2)
            + 0.00834 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.22 - Q.soft1_pt)
            - 289.0 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0309 - Q.C2_b2)
        ))
        + 0.578125 * grid(5, max(0.0, 2.33
            - 129.0 * max(0.0, Q.girth2_top30 - 0.00764)
            - 37.5 * max(0.0, Q.log_sum_pt - 6.92)
            + 0.0258 * max(0.0, 70.3 - Q.mass)
            - 0.0214 * max(0.0, 159.0 - Q.mass_top40)
            - 0.402 * max(0.0, Q.mass_top50 - 157.0)
            - 4.11 * max(0.0, Q.max_dr - 0.267)
            + 5.44 * max(0.0, Q.max_dr - 0.435)
            + 0.0338 * max(0.0, 54.1 - Q.n_particles)
            + 0.0253 * max(0.0, Q.sum_pt_top50 - 917.0)
            - 3.37 * max(0.0, 0.469 - Q.tau21)
            - 9.3 * max(0.0, Q.z_top30_slots - 0.895)
        ))
        + 0.0625 * grid(6, max(0.0, -0.458
            - 0.35 * max(0.0, 2.23 - Q.D2)
            + 0.0898 * max(0.0, Q.mass - 83.8)
            - 0.0971 * max(0.0, Q.mass - 106.0)
            + 0.0359 * max(0.0, 93.8 - Q.mass)
        ))
        - 0.625 * grid(7, max(0.0, 0.175
            - 600.0 * max(0.0, Q.psi_0p3 - 0.998)
            - 4.04 * max(0.0, 0.379 - Q.tau21)
            + 6.49 * max(0.0, 0.249 - Q.tau21_b2)
            - 9.28 * max(0.0, Q.z_top20_slots - 0.918)
            - 8.19 * max(0.0, 82.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3)
            - 31.4 * max(0.0, 82.8 - Q.mass) * max(0.0, Q.psi_0p3 - 0.995)
            - 78.5 * max(0.0, 90.9 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997)
            - 11.7 * max(0.0, 92.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.965)
            + 89.9 * max(0.0, 101.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997)
            + 3.72 * max(0.0, 104.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.966)
            + 0.00331 * max(0.0, 121.0 - Q.mass) * max(0.0, Q.sd_mass - 75.4)
        ))
        - 0.875 * grid(8, max(0.0, 2.08
            + 9.91 * max(0.0, Q.log_sum_pt - 6.96)
            - 0.0823 * max(0.0, Q.mass - 106.0)
            + 83.4 * max(0.0, Q.mass_over_sum_pt - 0.0766)
            - 0.12 * max(0.0, 16.9 - Q.n_dr_0p2_0p4)
            - 0.00768 * max(0.0, Q.sum_pt_top40 - 887.0)
            + 0.0756 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.sj3_pair_mass_max - 37.0)
        ))
        - 0.21875 * grid(9, max(0.0, 0.174
            - 5600.0 * max(0.0, Q.e3 - 0.000341)
            + 0.0167 * max(0.0, Q.mass - 110.0)
            - 0.0721 * max(0.0, Q.mass - 168.0)
            - 0.0803 * max(0.0, 64.1 - Q.mass)
            + 0.141 * max(0.0, 81.7 - Q.mass)
            + 0.0121 * max(0.0, 960.0 - Q.sum_pt_top50)
            - 21.0 * max(0.0, Q.z_top40_slots - 0.961)
            - 0.00016 * max(0.0, 95.5 - Q.mass_top40) * max(0.0, 1050.0 - Q.sum_pt)
            + 0.00423 * max(0.0, 1010.0 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.98)
        ))
        + 0.59375 * grid(11, max(0.0, 0.0886
            - 0.0154 * max(0.0, 79.4 - Q.mass)
            + 0.0504 * max(0.0, 15.7 - Q.n_dr_0p1_0p2)
            + 0.0989 * max(0.0, 10.3 - Q.n_dr_0p2_0p4)
            + 119.0 * max(0.0, Q.psi_0p2 - 0.992)
            - 0.0708 * max(0.0, 78.5 - Q.mass) * max(0.0, 3.87 - Q.D2)
        ))
        - 0.40625 * grid(12, max(0.0, -0.127
            - 870.0 * max(0.0, 0.00289 - Q.e2_sq)
            + 0.0819 * max(0.0, 82.6 - Q.mass)
            - 6.07 * max(0.0, 83.6 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            + 0.00173 * max(0.0, Q.sj3_pair_mass_max - 126.0) * max(0.0, 52.1 - Q.sj2_mass1)
        ))
        - 1.375 * grid(14, max(0.0, 0.652
            - 0.144 * max(0.0, 83.6 - Q.mass)
            + 3.67 * max(0.0, 0.303 - Q.tau21_b2)
            - 20.7 * max(0.0, 0.42 - Q.N2) * max(0.0, Q.max_dr - 0.25)
            - 2550.0 * max(0.0, 0.29 - Q.tau21_b2) * max(0.0, 0.00776 - Q.girth2_top40)
        ))
        - 0.1875 * grid(15, max(0.0, -0.444
            - 0.0266 * max(0.0, 33.4 - Q.sd_mass)
            + 0.0271 * max(0.0, 94.5 - Q.sd_mass)
            - 16.7 * max(0.0, 0.0618 - Q.tau1)
            + 257.0 * max(0.0, Q.sj2_dr - 0.202) * max(0.0, 0.0389 - Q.C2_b2)
        ))
    )


def logit_Z(Q):
    return (0.984375
        - 1.375 * grid(0, max(0.0, 1.51
            - 629.0 * max(0.0, 0.00619 - Q.e2_sq)
            - 183.0 * max(0.0, 0.0061 - Q.girth2_top20)
            - 0.162 * max(0.0, Q.mass - 78.3)
            - 0.182 * max(0.0, Q.mass - 87.4)
            + 659.0 * max(0.0, 0.00777 - Q.mass_over_sum_pt_sq)
            + 182.0 * max(0.0, Q.psi_0p3 - 0.996)
            - 0.0101 * max(0.0, 1010.0 - Q.sum_pt)
            - 18.3 * max(0.0, 0.0693 - Q.tau1)
            - 5.38 * max(0.0, 0.0912 - Q.z_dr_0p2_0p4)
            - 26.1 * max(0.0, 0.992 - Q.z_top50_slots)
        ))
        - 0.375 * grid(2, max(0.0, 1.22
            - 0.00446 * max(0.0, 1270.0 - Q.sum_pt)
        ))
        + 0.4375 * grid(3, max(0.0, 0.0826
            + 0.127 * max(0.0, 4.64 - Q.n_dr_0p2_0p4)
            + 161.0 * max(0.0, 0.0155 - Q.tau4)
            - 1440.0 * max(0.0, 0.317 - Q.tau21) * max(0.0, Q.lam1 - 0.00429)
        ))
        + 0.6875 * grid(5, max(0.0, 2.33
            - 129.0 * max(0.0, Q.girth2_top30 - 0.00764)
            - 37.5 * max(0.0, Q.log_sum_pt - 6.92)
            + 0.0258 * max(0.0, 70.3 - Q.mass)
            - 0.0214 * max(0.0, 159.0 - Q.mass_top40)
            - 0.402 * max(0.0, Q.mass_top50 - 157.0)
            - 4.11 * max(0.0, Q.max_dr - 0.267)
            + 5.44 * max(0.0, Q.max_dr - 0.435)
            + 0.0338 * max(0.0, 54.1 - Q.n_particles)
            + 0.0253 * max(0.0, Q.sum_pt_top50 - 917.0)
            - 3.37 * max(0.0, 0.469 - Q.tau21)
            - 9.3 * max(0.0, Q.z_top30_slots - 0.895)
        ))
        - 0.96875 * grid(6, max(0.0, -0.458
            - 0.35 * max(0.0, 2.23 - Q.D2)
            + 0.0898 * max(0.0, Q.mass - 83.8)
            - 0.0971 * max(0.0, Q.mass - 106.0)
            + 0.0359 * max(0.0, 93.8 - Q.mass)
        ))
        + 0.90625 * grid(7, max(0.0, 0.175
            - 600.0 * max(0.0, Q.psi_0p3 - 0.998)
            - 4.04 * max(0.0, 0.379 - Q.tau21)
            + 6.49 * max(0.0, 0.249 - Q.tau21_b2)
            - 9.28 * max(0.0, Q.z_top20_slots - 0.918)
            - 8.19 * max(0.0, 82.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3)
            - 31.4 * max(0.0, 82.8 - Q.mass) * max(0.0, Q.psi_0p3 - 0.995)
            - 78.5 * max(0.0, 90.9 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997)
            - 11.7 * max(0.0, 92.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.965)
            + 89.9 * max(0.0, 101.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997)
            + 3.72 * max(0.0, 104.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.966)
            + 0.00331 * max(0.0, 121.0 - Q.mass) * max(0.0, Q.sd_mass - 75.4)
        ))
        - 0.9375 * grid(8, max(0.0, 2.08
            + 9.91 * max(0.0, Q.log_sum_pt - 6.96)
            - 0.0823 * max(0.0, Q.mass - 106.0)
            + 83.4 * max(0.0, Q.mass_over_sum_pt - 0.0766)
            - 0.12 * max(0.0, 16.9 - Q.n_dr_0p2_0p4)
            - 0.00768 * max(0.0, Q.sum_pt_top40 - 887.0)
            + 0.0756 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.sj3_pair_mass_max - 37.0)
        ))
        + 0.3125 * grid(12, max(0.0, -0.127
            - 870.0 * max(0.0, 0.00289 - Q.e2_sq)
            + 0.0819 * max(0.0, 82.6 - Q.mass)
            - 6.07 * max(0.0, 83.6 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            + 0.00173 * max(0.0, Q.sj3_pair_mass_max - 126.0) * max(0.0, 52.1 - Q.sj2_mass1)
        ))
        + 0.5625 * grid(15, max(0.0, -0.444
            - 0.0266 * max(0.0, 33.4 - Q.sd_mass)
            + 0.0271 * max(0.0, 94.5 - Q.sd_mass)
            - 16.7 * max(0.0, 0.0618 - Q.tau1)
            + 257.0 * max(0.0, Q.sj2_dr - 0.202) * max(0.0, 0.0389 - Q.C2_b2)
        ))
    )


def logit_t(Q):
    return (0.78125
        + 0.125 * grid(0, max(0.0, 1.51
            - 629.0 * max(0.0, 0.00619 - Q.e2_sq)
            - 183.0 * max(0.0, 0.0061 - Q.girth2_top20)
            - 0.162 * max(0.0, Q.mass - 78.3)
            - 0.182 * max(0.0, Q.mass - 87.4)
            + 659.0 * max(0.0, 0.00777 - Q.mass_over_sum_pt_sq)
            + 182.0 * max(0.0, Q.psi_0p3 - 0.996)
            - 0.0101 * max(0.0, 1010.0 - Q.sum_pt)
            - 18.3 * max(0.0, 0.0693 - Q.tau1)
            - 5.38 * max(0.0, 0.0912 - Q.z_dr_0p2_0p4)
            - 26.1 * max(0.0, 0.992 - Q.z_top50_slots)
        ))
        + 0.1875 * grid(4, max(0.0, 1.28
            - 173.0 * max(0.0, 0.00523 - Q.girth2_top15)
            + 0.047 * max(0.0, 69.8 - Q.mass_top40)
            - 0.0344 * max(0.0, Q.n_particles - 22.5)
            + 1.91 * max(0.0, 0.324 - Q.planar_flow)
            + 0.0238 * max(0.0, Q.sj3_pair_mass_min - 33.4)
            - 36.4 * max(0.0, 80.7 - Q.mass) * max(0.0, 0.0041 - Q.lam2)
            + 12.0 * max(0.0, 104.0 - Q.mass) * max(0.0, 0.00334 - Q.lam2)
            - 0.00338 * max(0.0, 91.8 - Q.mass_top40) * max(0.0, 9.1 - Q.D2)
            + 0.00834 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.22 - Q.soft1_pt)
            - 289.0 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0309 - Q.C2_b2)
        ))
        - 0.46875 * grid(5, max(0.0, 2.33
            - 129.0 * max(0.0, Q.girth2_top30 - 0.00764)
            - 37.5 * max(0.0, Q.log_sum_pt - 6.92)
            + 0.0258 * max(0.0, 70.3 - Q.mass)
            - 0.0214 * max(0.0, 159.0 - Q.mass_top40)
            - 0.402 * max(0.0, Q.mass_top50 - 157.0)
            - 4.11 * max(0.0, Q.max_dr - 0.267)
            + 5.44 * max(0.0, Q.max_dr - 0.435)
            + 0.0338 * max(0.0, 54.1 - Q.n_particles)
            + 0.0253 * max(0.0, Q.sum_pt_top50 - 917.0)
            - 3.37 * max(0.0, 0.469 - Q.tau21)
            - 9.3 * max(0.0, Q.z_top30_slots - 0.895)
        ))
        - 0.28125 * grid(7, max(0.0, 0.175
            - 600.0 * max(0.0, Q.psi_0p3 - 0.998)
            - 4.04 * max(0.0, 0.379 - Q.tau21)
            + 6.49 * max(0.0, 0.249 - Q.tau21_b2)
            - 9.28 * max(0.0, Q.z_top20_slots - 0.918)
            - 8.19 * max(0.0, 82.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3)
            - 31.4 * max(0.0, 82.8 - Q.mass) * max(0.0, Q.psi_0p3 - 0.995)
            - 78.5 * max(0.0, 90.9 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997)
            - 11.7 * max(0.0, 92.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.965)
            + 89.9 * max(0.0, 101.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997)
            + 3.72 * max(0.0, 104.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.966)
            + 0.00331 * max(0.0, 121.0 - Q.mass) * max(0.0, Q.sd_mass - 75.4)
        ))
        + 0.2109375 * grid(8, max(0.0, 2.08
            + 9.91 * max(0.0, Q.log_sum_pt - 6.96)
            - 0.0823 * max(0.0, Q.mass - 106.0)
            + 83.4 * max(0.0, Q.mass_over_sum_pt - 0.0766)
            - 0.12 * max(0.0, 16.9 - Q.n_dr_0p2_0p4)
            - 0.00768 * max(0.0, Q.sum_pt_top40 - 887.0)
            + 0.0756 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.sj3_pair_mass_max - 37.0)
        ))
        + 0.984375 * grid(10, max(0.0, 1.4
            + 0.512 * max(0.0, 3.84 - Q.D2)
            - 0.0624 * max(0.0, Q.mass - 149.0)
            - 0.286 * max(0.0, Q.mass - 172.0)
            + 0.0569 * max(0.0, 80.8 - Q.mass)
            + 0.132 * max(0.0, Q.mass_top50 - 156.0)
            - 0.0511 * max(0.0, 82.7 - Q.sj2_mass1)
            - 0.00483 * max(0.0, 980.0 - Q.sum_pt)
            + 0.00292 * max(0.0, 880.0 - Q.sum_pt_top5)
            + 2.51 * max(0.0, 0.604 - Q.tau21_b2)
            - 9.99 * max(0.0, Q.z_dr_0_0p05 - 0.768)
            - 1.81 * max(0.0, 3.83 - Q.D2) * max(0.0, Q.sj2_dr - 0.196)
            + 2.67 * max(0.0, Q.mass - 173.0) * max(0.0, Q.M2 - 0.0406)
            + 0.000955 * max(0.0, 81.5 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.9)
        ))
        - 0.375 * grid(12, max(0.0, -0.127
            - 870.0 * max(0.0, 0.00289 - Q.e2_sq)
            + 0.0819 * max(0.0, 82.6 - Q.mass)
            - 6.07 * max(0.0, 83.6 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            + 0.00173 * max(0.0, Q.sj3_pair_mass_max - 126.0) * max(0.0, 52.1 - Q.sj2_mass1)
        ))
        - 0.90625 * grid(13, max(0.0, 2.34
            + 14.1 * max(0.0, 6.82 - Q.log_sum_pt)
            - 0.203 * max(0.0, Q.mass - 143.0)
            + 0.0774 * max(0.0, Q.mass - 174.0)
            + 0.146 * max(0.0, Q.mass_top50 - 141.0)
            - 0.0248 * max(0.0, 1010.0 - Q.sum_pt)
            - 0.0081 * max(0.0, 1100.0 - Q.sum_pt)
            + 0.0185 * max(0.0, 1020.0 - Q.sum_pt_top40) * max(0.0, 0.65 - Q.D3)
        ))
        - 0.375 * grid(15, max(0.0, -0.444
            - 0.0266 * max(0.0, 33.4 - Q.sd_mass)
            + 0.0271 * max(0.0, 94.5 - Q.sd_mass)
            - 16.7 * max(0.0, 0.0618 - Q.tau1)
            + 257.0 * max(0.0, Q.sj2_dr - 0.202) * max(0.0, 0.0389 - Q.C2_b2)
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
