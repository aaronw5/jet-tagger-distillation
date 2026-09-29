"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network (step 4; no W/Z/H/t mass values offered as thresholds), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.1% (the network: 81.1%); same class as the network for 93.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_pt               pT [GeV] of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.z_1                    pT of particle 1 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft3_z                pT share of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top10=mass_of(10),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_particles=len(real),
        n_for_50pct=ncum(0.5),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_0=pt[0],
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        soft1_pt=softp(1, 'pt'),
        soft2_pt=softp(2, 'pt'),
        soft3_pt=softp(3, 'pt'),
        z_1=z[1],
        z_7=z[7],
        z_9=z[9],
        soft3_z=softp(3, 'z'),
        soft4_z=softp(4, 'z'),
        z_top10_slots=sum(pt[:10]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_2=z[2] * dr[2],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_mass=softdrop("mass"),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        eta_0=eta[0],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
    )


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-1.078125
        + 0.625 * grid(1, max(0.0, 2.34
            + 721.0 * max(0.0, 0.00083 - Q.girth2_top15)
            + 172.0 * max(0.0, 0.0033 - Q.girth2_top15)
            + 760.0 * max(0.0, 0.001 - Q.girth2_top5)
            + 95.2 * max(0.0, 0.0087 - Q.girth2_top5)
            + 34.8 * max(0.0, Q.log_sum_pt - 6.9)
            - 23.4 * max(0.0, Q.log_sum_pt - 7.0)
            - 6.04 * max(0.0, 7.1 - Q.log_sum_pt)
            - 0.0099 * max(0.0, 51.0 - Q.m012)
            + 0.0379 * max(0.0, 88.0 - Q.mass)
            - 0.0298 * max(0.0, 100.0 - Q.mass)
            - 0.0275 * max(0.0, 48.0 - Q.mass_top20)
            - 0.0272 * max(0.0, 17.0 - Q.mass_top5)
            - 0.0485 * max(0.0, 7.6 - Q.n_dr_0p1_0p2)
            - 0.0998 * max(0.0, 7.4 - Q.n_dr_0p2_0p4)
            + 0.0334 * max(0.0, Q.n_particles - 38.0)
            - 0.0234 * max(0.0, 32.0 - Q.sj2_mass1)
            + 0.888 * max(0.0, 1.6 - Q.soft1_pt)
            - 0.959 * max(0.0, 2.3 - Q.soft1_pt)
            + 0.00626 * max(0.0, 1000.0 - Q.sum_pt)
            + 0.00226 * max(0.0, 700.0 - Q.sum_pt_top2)
            + 0.00196 * max(0.0, Q.sum_pt_top30 - 1200.0)
            + 0.00713 * max(0.0, 1100.0 - Q.sum_pt_top40)
            - 0.00663 * max(0.0, Q.sum_pt_top50 - 960.0)
            + 6.9 * max(0.0, 0.078 - Q.tau2)
            - 30.2 * max(0.0, 0.032 - Q.z_7)
            - 3.07 * max(0.0, Q.z_dr_0_0p05 - 0.85)
            - 18.6 * max(0.0, Q.z_top50_slots - 0.96)
            - 28.6 * max(0.0, Q.zdr_0 - 0.00098)
            - 476.0 * max(0.0, 0.03 - Q.M3) * max(0.0, Q.M2 - 0.046)
            + 0.00372 * max(0.0, 48.0 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 30.0)
            + 0.2 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.15 - Q.dr_1)
            + 1.81 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.013 - Q.zdr_2)
            - 0.00167 * max(0.0, 31.0 - Q.sj3_mass1) * max(0.0, 21.0 - Q.sj3_mass2)
            + 39.3 * max(0.0, 0.17 - Q.tau1) * max(0.0, Q.eta_0 - -0.016)
            + 172.0 * max(0.0, Q.z_top30_slots - 0.93) * max(0.0, 0.075 - Q.C2)
            + 0.408 * max(0.0, Q.z_top30_slots - 0.93) * max(0.0, Q.max_pair_mass - 14.0)
            + 0.74 * max(0.0, Q.z_top30_slots - 0.93) * max(0.0, Q.ptdr0_3 - 6.5)
            + 1.28 * max(0.0, Q.z_top30_slots - 0.96) * max(0.0, Q.ptdr0_4 - 3.7)
        ))
        - 0.75 * grid(3, max(0.0, -0.0429
            + 1060.0 * max(0.0, 0.00061 - Q.lam2)
            - 0.0212 * max(0.0, 56.0 - Q.mass)
            + 0.0364 * max(0.0, 46.0 - Q.n_particles)
            - 2.01 * max(0.0, 0.38 - Q.tau21)
            + 0.802 * max(0.0, 14.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.95)
            - 0.000709 * max(0.0, 6.6 - Q.n_dr_0p2_0p4) * max(0.0, 1000.0 - Q.sum_pt_top30)
            + 2.92 * max(0.0, 6.4 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.97)
            - 312.0 * max(0.0, 0.38 - Q.tau21) * max(0.0, Q.lam1 - 0.0036)
        ))
        - 0.875 * grid(4, max(0.0, 0.0131
            + 0.138 * max(0.0, 6.5 - Q.D2)
            - 8.68 * max(0.0, Q.LHA - 0.37)
            + 12.7 * max(0.0, Q.e2 - 0.038)
            + 85.2 * max(0.0, Q.e2_sq - 0.014)
            - 28.6 * max(0.0, Q.girth2_top15 - 0.0072)
            - 89.8 * max(0.0, 0.0054 - Q.girth2_top15)
            + 53.7 * max(0.0, Q.lam2 - 0.00044)
            + 0.0134 * max(0.0, Q.mass - 140.0)
            - 0.0578 * max(0.0, 74.0 - Q.mass)
            - 0.0639 * max(0.0, 87.0 - Q.mass)
            + 0.0426 * max(0.0, 100.0 - Q.mass)
            - 0.0163 * max(0.0, Q.mass_top20 - 100.0)
            + 0.0264 * max(0.0, Q.n_dr_0_0p05 - 6.8)
            + 0.0244 * max(0.0, 25.0 - Q.n_dr_0p2_0p4)
            - 0.0257 * max(0.0, Q.n_particles - 21.0)
            + 0.00629 * max(0.0, Q.sj3_pair_mass_min - 34.0)
            - 0.00279 * max(0.0, 1000.0 - Q.sum_pt)
            - 18.7 * max(0.0, 0.072 - Q.tau1)
            - 17.3 * max(0.0, 80.0 - Q.mass) * max(0.0, 0.0037 - Q.lam2)
            + 10.5 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.004 - Q.lam2)
            - 3.83 * max(0.0, 86.0 - Q.mass) * max(0.0, Q.zdr_0 - 0.0069)
            + 3.09e-05 * max(0.0, 120.0 - Q.mass_top30) * max(0.0, 1000.0 - Q.sum_pt_top40)
            - 0.0105 * max(0.0, 89.0 - Q.mass_top40) * max(0.0, 6.8 - Q.D2)
            + 0.00368 * max(0.0, Q.n_particles - 24.0) * max(0.0, 2.1 - Q.soft1_pt)
        ))
        - 0.3125 * grid(5, max(0.0, -0.119
            + 0.387 * max(0.0, 2.0 - Q.D2)
            + 16.8 * max(0.0, 0.041 - Q.e2)
            - 52.8 * max(0.0, Q.girth2_top50 - 0.016)
            - 19.4 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.0849 * max(0.0, Q.mass - 160.0)
            - 18.9 * max(0.0, Q.mass_over_sum_pt - 0.092)
            + 0.0144 * max(0.0, Q.mass_top40 - 59.0)
            - 0.567 * max(0.0, Q.max_dr - 0.33)
            + 1.69 * max(0.0, 0.42 - Q.max_dr)
            - 0.0202 * max(0.0, 22.0 - Q.n_dr_0p1_0p2)
            + 0.0369 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 0.0204 * max(0.0, Q.n_particles - 46.0)
            + 0.0321 * max(0.0, 46.0 - Q.n_particles)
            - 0.91 * max(0.0, 0.71 - Q.psi_0p1)
            + 0.0318 * max(0.0, Q.sd_mass - 70.0)
            - 0.0375 * max(0.0, Q.sd_mass - 83.0)
            + 0.0186 * max(0.0, 23.0 - Q.sj3_mass1)
            + 0.361 * max(0.0, Q.soft3_pt - 0.38)
            - 490.0 * max(0.0, Q.soft3_z - 0.00083)
            + 0.0157 * max(0.0, Q.sum_pt - 920.0)
            - 0.0229 * max(0.0, Q.sum_pt - 990.0)
            + 0.019 * max(0.0, Q.sum_pt_top50 - 950.0)
            - 1.12 * max(0.0, 0.49 - Q.tau21)
            - 10.7 * max(0.0, 0.036 - Q.z_dr_0p2_0p4)
            - 6.19 * max(0.0, Q.z_top30_slots - 0.9)
            - 0.0196 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.2 - Q.D2)
        ))
        + 0.15625 * grid(6, max(0.0, 1.27
            - 0.216 * max(0.0, 2.8 - Q.D2)
            - 24.5 * max(0.0, Q.e2 - 0.027)
            - 0.0211 * max(0.0, Q.mass - 140.0)
            + 0.093 * max(0.0, 84.0 - Q.mass)
            - 0.0571 * max(0.0, 120.0 - Q.mass)
            + 0.0406 * max(0.0, Q.mass_top10 - 95.0)
            - 0.0274 * max(0.0, Q.sj3_mass1 - 22.0)
            + 0.00838 * max(0.0, 76.0 - Q.sj3_pair_mass_min)
            + 3900.0 * max(0.0, Q.e2 - 0.022) * max(0.0, Q.psi_0p3 - 0.99)
            + 3.63e-05 * max(0.0, 150.0 - Q.mass) * max(0.0, 990.0 - Q.sum_pt_top50)
        ))
        + 0.03125 * grid(8, max(0.0, -1.41
            + 39.7 * max(0.0, Q.e2 - 0.047)
            + 387.0 * max(0.0, Q.e2_sq - 0.008)
            - 466.0 * max(0.0, Q.e2_sq - 0.0098)
            + 2990.0 * max(0.0, 0.00035 - Q.e3)
            + 272.0 * max(0.0, 0.0082 - Q.girth2_top15)
            - 140.0 * max(0.0, 0.017 - Q.girth2_top15)
            + 10.1 * max(0.0, 6.9 - Q.log_sum_pt)
            + 0.0171 * max(0.0, Q.mass - 65.0)
            + 0.0695 * max(0.0, Q.mass - 81.0)
            - 0.0376 * max(0.0, Q.mass - 100.0)
            - 0.0408 * max(0.0, Q.mass - 120.0)
            + 233.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.00055)
            - 0.0651 * max(0.0, Q.mass_top50 - 81.0)
            + 0.0324 * max(0.0, Q.n_dr_0p2_0p4 - 8.1)
            - 0.0234 * max(0.0, 16.0 - Q.n_dr_0p2_0p4)
            + 0.0296 * max(0.0, Q.n_pt_above_1 - 55.0)
            + 4.27 * max(0.0, Q.sj2_dr - 0.23)
            - 163.0 * max(0.0, 0.0094 - Q.girth2_top15) * max(0.0, 0.61 - Q.tau21_b2)
            + 0.044 * max(0.0, Q.log_sum_pt - 7.0) * max(0.0, Q.sj3_pair_mass_max - 16.0)
            - 1.37 * max(0.0, Q.n_dr_0p2_0p4 - 8.9) * max(0.0, Q.zdr_0 - 0.0039)
        ))
        + 0.515625 * grid(9, max(0.0, 0.851
            + 13.8 * max(0.0, 0.21 - Q.LHA)
            + 33.8 * max(0.0, Q.girth - 0.093)
            - 50.7 * max(0.0, 0.047 - Q.girth)
            - 122.0 * max(0.0, Q.girth2_top30 - 0.024)
            + 283.0 * max(0.0, 0.0061 - Q.girth2_top40)
            - 0.039 * max(0.0, Q.mass - 140.0)
            - 0.0931 * max(0.0, Q.mass - 160.0)
            + 0.13 * max(0.0, Q.mass - 170.0)
            - 0.0537 * max(0.0, 64.0 - Q.mass)
            + 0.0704 * max(0.0, 84.0 - Q.mass)
            + 0.0323 * max(0.0, 130.0 - Q.mass_top40)
            - 0.0326 * max(0.0, 160.0 - Q.mass_top40)
            - 26.0 * max(0.0, Q.psi_0p3 - 0.99)
            + 0.00893 * max(0.0, 960.0 - Q.sum_pt_top50)
            + 1.22 * max(0.0, 0.29 - Q.z_dr_0p1_0p2)
            + 0.000696 * max(0.0, 88.0 - Q.mass) * max(0.0, Q.n_particles - 40.0)
            + 0.000726 * max(0.0, 110.0 - Q.mass_top30) * max(0.0, Q.n_dr_0p2_0p4 - 11.0)
            - 0.0003 * max(0.0, 72.0 - Q.mass_top40) * max(0.0, 1000.0 - Q.sum_pt)
            + 0.000433 * max(0.0, 73.0 - Q.mass_top40) * max(0.0, 940.0 - Q.sum_pt_top50)
            - 0.000715 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 19.0 - Q.n_pt_above_10)
        ))
        - 0.015625 * grid(10, max(0.0, 0.67
            + 0.297 * max(0.0, 3.8 - Q.D2)
            - 5.67 * max(0.0, Q.M2 - 0.055)
            + 6.11 * max(0.0, 0.064 - Q.dr_0)
            - 27.1 * max(0.0, 0.062 - Q.e2)
            - 47.8 * max(0.0, 0.005 - Q.girth2_top20)
            + 125.0 * max(0.0, 0.018 - Q.girth2_top30)
            - 3.91 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.0497 * max(0.0, Q.mass - 61.0)
            + 0.077 * max(0.0, Q.mass - 81.0)
            + 0.0191 * max(0.0, Q.mass - 88.0)
            - 0.0729 * max(0.0, Q.mass - 140.0)
            - 0.0852 * max(0.0, Q.mass - 160.0)
            - 318.0 * max(0.0, Q.mass_over_sum_pt - 0.17)
            + 18.6 * max(0.0, 0.078 - Q.mass_over_sum_pt)
            + 747.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
            - 0.00737 * max(0.0, 66.0 - Q.mass_top10)
            + 0.00873 * max(0.0, Q.mass_top5 - 15.0)
            + 0.0401 * max(0.0, Q.mass_top50 - 140.0)
            - 0.0147 * max(0.0, 78.0 - Q.sj2_mass1)
            - 0.00171 * max(0.0, 990.0 - Q.sum_pt)
            + 0.0017 * max(0.0, 850.0 - Q.sum_pt_top5)
            - 43.2 * max(0.0, 0.056 - Q.tau1)
            - 3.55 * max(0.0, 3.6 - Q.D2) * max(0.0, Q.sj2_dr - 0.2)
            + 0.146 * max(0.0, Q.mass - 90.0) * max(0.0, Q.psi_0p3 - 0.98)
            + 0.00409 * max(0.0, Q.mass - 59.0) * max(0.0, 2.7 - Q.soft2_pt)
            + 0.000774 * max(0.0, 110.0 - Q.mass_top40) * max(0.0, Q.pt1_dr01 - 13.0)
            + 28.4 * max(0.0, Q.mass_top40 - 160.0) * max(0.0, Q.soft4_z - 0.0011)
            + 2.53e-05 * max(0.0, 89.0 - Q.sj2_mass1) * max(0.0, 530.0 - Q.sum_pt_top2)
        ))
        + 0.234375 * grid(12, max(0.0, -0.427
            - 0.0674 * max(0.0, 56.0 - Q.mass)
            + 0.101 * max(0.0, 84.0 - Q.mass)
            - 0.00243 * max(0.0, Q.sum_pt - 1200.0)
            - 0.0068 * max(0.0, 970.0 - Q.sum_pt)
            + 1.4 * max(0.0, Q.z_dr_0p1_0p2 - 0.4)
            - 2.72 * max(0.0, 90.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3)
            + 0.0785 * max(0.0, 90.0 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.4)
            - 0.209 * max(0.0, 88.0 - Q.mass) * max(0.0, 0.075 - Q.z_dr_0p1_0p2)
            + 0.00124 * max(0.0, Q.sd_mass - 130.0) * max(0.0, 37.0 - Q.sj3_mass1)
            + 4.5e-05 * max(0.0, Q.sj3_pair_mass_max - 130.0) * max(0.0, 530.0 - Q.pt_0)
        ))
    )


def logit_q(Q):
    return (1.359375
        - 0.1875 * grid(1, max(0.0, 2.34
            + 721.0 * max(0.0, 0.00083 - Q.girth2_top15)
            + 172.0 * max(0.0, 0.0033 - Q.girth2_top15)
            + 760.0 * max(0.0, 0.001 - Q.girth2_top5)
            + 95.2 * max(0.0, 0.0087 - Q.girth2_top5)
            + 34.8 * max(0.0, Q.log_sum_pt - 6.9)
            - 23.4 * max(0.0, Q.log_sum_pt - 7.0)
            - 6.04 * max(0.0, 7.1 - Q.log_sum_pt)
            - 0.0099 * max(0.0, 51.0 - Q.m012)
            + 0.0379 * max(0.0, 88.0 - Q.mass)
            - 0.0298 * max(0.0, 100.0 - Q.mass)
            - 0.0275 * max(0.0, 48.0 - Q.mass_top20)
            - 0.0272 * max(0.0, 17.0 - Q.mass_top5)
            - 0.0485 * max(0.0, 7.6 - Q.n_dr_0p1_0p2)
            - 0.0998 * max(0.0, 7.4 - Q.n_dr_0p2_0p4)
            + 0.0334 * max(0.0, Q.n_particles - 38.0)
            - 0.0234 * max(0.0, 32.0 - Q.sj2_mass1)
            + 0.888 * max(0.0, 1.6 - Q.soft1_pt)
            - 0.959 * max(0.0, 2.3 - Q.soft1_pt)
            + 0.00626 * max(0.0, 1000.0 - Q.sum_pt)
            + 0.00226 * max(0.0, 700.0 - Q.sum_pt_top2)
            + 0.00196 * max(0.0, Q.sum_pt_top30 - 1200.0)
            + 0.00713 * max(0.0, 1100.0 - Q.sum_pt_top40)
            - 0.00663 * max(0.0, Q.sum_pt_top50 - 960.0)
            + 6.9 * max(0.0, 0.078 - Q.tau2)
            - 30.2 * max(0.0, 0.032 - Q.z_7)
            - 3.07 * max(0.0, Q.z_dr_0_0p05 - 0.85)
            - 18.6 * max(0.0, Q.z_top50_slots - 0.96)
            - 28.6 * max(0.0, Q.zdr_0 - 0.00098)
            - 476.0 * max(0.0, 0.03 - Q.M3) * max(0.0, Q.M2 - 0.046)
            + 0.00372 * max(0.0, 48.0 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 30.0)
            + 0.2 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.15 - Q.dr_1)
            + 1.81 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.013 - Q.zdr_2)
            - 0.00167 * max(0.0, 31.0 - Q.sj3_mass1) * max(0.0, 21.0 - Q.sj3_mass2)
            + 39.3 * max(0.0, 0.17 - Q.tau1) * max(0.0, Q.eta_0 - -0.016)
            + 172.0 * max(0.0, Q.z_top30_slots - 0.93) * max(0.0, 0.075 - Q.C2)
            + 0.408 * max(0.0, Q.z_top30_slots - 0.93) * max(0.0, Q.max_pair_mass - 14.0)
            + 0.74 * max(0.0, Q.z_top30_slots - 0.93) * max(0.0, Q.ptdr0_3 - 6.5)
            + 1.28 * max(0.0, Q.z_top30_slots - 0.96) * max(0.0, Q.ptdr0_4 - 3.7)
        ))
        + 0.125 * grid(2, max(0.0, -0.0281
            - 6.25 * max(0.0, Q.log_sum_pt - 7.1)
            - 0.0111 * max(0.0, 87.0 - Q.mass)
            + 320.0 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.025 - Q.girth2_top15)
        ))
        - 1.0625 * grid(4, max(0.0, 0.0131
            + 0.138 * max(0.0, 6.5 - Q.D2)
            - 8.68 * max(0.0, Q.LHA - 0.37)
            + 12.7 * max(0.0, Q.e2 - 0.038)
            + 85.2 * max(0.0, Q.e2_sq - 0.014)
            - 28.6 * max(0.0, Q.girth2_top15 - 0.0072)
            - 89.8 * max(0.0, 0.0054 - Q.girth2_top15)
            + 53.7 * max(0.0, Q.lam2 - 0.00044)
            + 0.0134 * max(0.0, Q.mass - 140.0)
            - 0.0578 * max(0.0, 74.0 - Q.mass)
            - 0.0639 * max(0.0, 87.0 - Q.mass)
            + 0.0426 * max(0.0, 100.0 - Q.mass)
            - 0.0163 * max(0.0, Q.mass_top20 - 100.0)
            + 0.0264 * max(0.0, Q.n_dr_0_0p05 - 6.8)
            + 0.0244 * max(0.0, 25.0 - Q.n_dr_0p2_0p4)
            - 0.0257 * max(0.0, Q.n_particles - 21.0)
            + 0.00629 * max(0.0, Q.sj3_pair_mass_min - 34.0)
            - 0.00279 * max(0.0, 1000.0 - Q.sum_pt)
            - 18.7 * max(0.0, 0.072 - Q.tau1)
            - 17.3 * max(0.0, 80.0 - Q.mass) * max(0.0, 0.0037 - Q.lam2)
            + 10.5 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.004 - Q.lam2)
            - 3.83 * max(0.0, 86.0 - Q.mass) * max(0.0, Q.zdr_0 - 0.0069)
            + 3.09e-05 * max(0.0, 120.0 - Q.mass_top30) * max(0.0, 1000.0 - Q.sum_pt_top40)
            - 0.0105 * max(0.0, 89.0 - Q.mass_top40) * max(0.0, 6.8 - Q.D2)
            + 0.00368 * max(0.0, Q.n_particles - 24.0) * max(0.0, 2.1 - Q.soft1_pt)
        ))
        + 0.21875 * grid(6, max(0.0, 1.27
            - 0.216 * max(0.0, 2.8 - Q.D2)
            - 24.5 * max(0.0, Q.e2 - 0.027)
            - 0.0211 * max(0.0, Q.mass - 140.0)
            + 0.093 * max(0.0, 84.0 - Q.mass)
            - 0.0571 * max(0.0, 120.0 - Q.mass)
            + 0.0406 * max(0.0, Q.mass_top10 - 95.0)
            - 0.0274 * max(0.0, Q.sj3_mass1 - 22.0)
            + 0.00838 * max(0.0, 76.0 - Q.sj3_pair_mass_min)
            + 3900.0 * max(0.0, Q.e2 - 0.022) * max(0.0, Q.psi_0p3 - 0.99)
            + 3.63e-05 * max(0.0, 150.0 - Q.mass) * max(0.0, 990.0 - Q.sum_pt_top50)
        ))
        + 0.015625 * grid(8, max(0.0, -1.41
            + 39.7 * max(0.0, Q.e2 - 0.047)
            + 387.0 * max(0.0, Q.e2_sq - 0.008)
            - 466.0 * max(0.0, Q.e2_sq - 0.0098)
            + 2990.0 * max(0.0, 0.00035 - Q.e3)
            + 272.0 * max(0.0, 0.0082 - Q.girth2_top15)
            - 140.0 * max(0.0, 0.017 - Q.girth2_top15)
            + 10.1 * max(0.0, 6.9 - Q.log_sum_pt)
            + 0.0171 * max(0.0, Q.mass - 65.0)
            + 0.0695 * max(0.0, Q.mass - 81.0)
            - 0.0376 * max(0.0, Q.mass - 100.0)
            - 0.0408 * max(0.0, Q.mass - 120.0)
            + 233.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.00055)
            - 0.0651 * max(0.0, Q.mass_top50 - 81.0)
            + 0.0324 * max(0.0, Q.n_dr_0p2_0p4 - 8.1)
            - 0.0234 * max(0.0, 16.0 - Q.n_dr_0p2_0p4)
            + 0.0296 * max(0.0, Q.n_pt_above_1 - 55.0)
            + 4.27 * max(0.0, Q.sj2_dr - 0.23)
            - 163.0 * max(0.0, 0.0094 - Q.girth2_top15) * max(0.0, 0.61 - Q.tau21_b2)
            + 0.044 * max(0.0, Q.log_sum_pt - 7.0) * max(0.0, Q.sj3_pair_mass_max - 16.0)
            - 1.37 * max(0.0, Q.n_dr_0p2_0p4 - 8.9) * max(0.0, Q.zdr_0 - 0.0039)
        ))
        + 0.5625 * grid(9, max(0.0, 0.851
            + 13.8 * max(0.0, 0.21 - Q.LHA)
            + 33.8 * max(0.0, Q.girth - 0.093)
            - 50.7 * max(0.0, 0.047 - Q.girth)
            - 122.0 * max(0.0, Q.girth2_top30 - 0.024)
            + 283.0 * max(0.0, 0.0061 - Q.girth2_top40)
            - 0.039 * max(0.0, Q.mass - 140.0)
            - 0.0931 * max(0.0, Q.mass - 160.0)
            + 0.13 * max(0.0, Q.mass - 170.0)
            - 0.0537 * max(0.0, 64.0 - Q.mass)
            + 0.0704 * max(0.0, 84.0 - Q.mass)
            + 0.0323 * max(0.0, 130.0 - Q.mass_top40)
            - 0.0326 * max(0.0, 160.0 - Q.mass_top40)
            - 26.0 * max(0.0, Q.psi_0p3 - 0.99)
            + 0.00893 * max(0.0, 960.0 - Q.sum_pt_top50)
            + 1.22 * max(0.0, 0.29 - Q.z_dr_0p1_0p2)
            + 0.000696 * max(0.0, 88.0 - Q.mass) * max(0.0, Q.n_particles - 40.0)
            + 0.000726 * max(0.0, 110.0 - Q.mass_top30) * max(0.0, Q.n_dr_0p2_0p4 - 11.0)
            - 0.0003 * max(0.0, 72.0 - Q.mass_top40) * max(0.0, 1000.0 - Q.sum_pt)
            + 0.000433 * max(0.0, 73.0 - Q.mass_top40) * max(0.0, 940.0 - Q.sum_pt_top50)
            - 0.000715 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 19.0 - Q.n_pt_above_10)
        ))
        - 0.015625 * grid(10, max(0.0, 0.67
            + 0.297 * max(0.0, 3.8 - Q.D2)
            - 5.67 * max(0.0, Q.M2 - 0.055)
            + 6.11 * max(0.0, 0.064 - Q.dr_0)
            - 27.1 * max(0.0, 0.062 - Q.e2)
            - 47.8 * max(0.0, 0.005 - Q.girth2_top20)
            + 125.0 * max(0.0, 0.018 - Q.girth2_top30)
            - 3.91 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.0497 * max(0.0, Q.mass - 61.0)
            + 0.077 * max(0.0, Q.mass - 81.0)
            + 0.0191 * max(0.0, Q.mass - 88.0)
            - 0.0729 * max(0.0, Q.mass - 140.0)
            - 0.0852 * max(0.0, Q.mass - 160.0)
            - 318.0 * max(0.0, Q.mass_over_sum_pt - 0.17)
            + 18.6 * max(0.0, 0.078 - Q.mass_over_sum_pt)
            + 747.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
            - 0.00737 * max(0.0, 66.0 - Q.mass_top10)
            + 0.00873 * max(0.0, Q.mass_top5 - 15.0)
            + 0.0401 * max(0.0, Q.mass_top50 - 140.0)
            - 0.0147 * max(0.0, 78.0 - Q.sj2_mass1)
            - 0.00171 * max(0.0, 990.0 - Q.sum_pt)
            + 0.0017 * max(0.0, 850.0 - Q.sum_pt_top5)
            - 43.2 * max(0.0, 0.056 - Q.tau1)
            - 3.55 * max(0.0, 3.6 - Q.D2) * max(0.0, Q.sj2_dr - 0.2)
            + 0.146 * max(0.0, Q.mass - 90.0) * max(0.0, Q.psi_0p3 - 0.98)
            + 0.00409 * max(0.0, Q.mass - 59.0) * max(0.0, 2.7 - Q.soft2_pt)
            + 0.000774 * max(0.0, 110.0 - Q.mass_top40) * max(0.0, Q.pt1_dr01 - 13.0)
            + 28.4 * max(0.0, Q.mass_top40 - 160.0) * max(0.0, Q.soft4_z - 0.0011)
            + 2.53e-05 * max(0.0, 89.0 - Q.sj2_mass1) * max(0.0, 530.0 - Q.sum_pt_top2)
        ))
        - 0.25 * grid(11, max(0.0, -0.551
            + 6.03 * max(0.0, 0.24 - Q.LHA)
            + 224.0 * max(0.0, 0.0043 - Q.girth2_top10)
            - 85.9 * max(0.0, 0.0058 - Q.girth2_top2)
            - 161.0 * max(0.0, 0.0065 - Q.lam1)
            - 0.0565 * max(0.0, 81.0 - Q.mass)
            + 0.0564 * max(0.0, 96.0 - Q.mass)
            + 0.0379 * max(0.0, 17.0 - Q.n_dr_0p1_0p2)
            + 0.143 * max(0.0, 9.0 - Q.n_dr_0p2_0p4)
            + 45.6 * max(0.0, Q.psi_0p3 - 0.99)
            + 4.2 * max(0.0, 0.15 - Q.tau21_b2)
            - 1.55 * max(0.0, Q.z_top10_slots - 0.75)
            - 0.00729 * max(0.0, 84.0 - Q.mass) * max(0.0, 3.6 - Q.D2)
            + 0.00118 * max(0.0, 95.0 - Q.mass) * max(0.0, Q.mass_top10 - 38.0)
            - 28.5 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0061 - Q.girth2)
            - 0.0027 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 36.0)
        ))
        + 0.34375 * grid(12, max(0.0, -0.427
            - 0.0674 * max(0.0, 56.0 - Q.mass)
            + 0.101 * max(0.0, 84.0 - Q.mass)
            - 0.00243 * max(0.0, Q.sum_pt - 1200.0)
            - 0.0068 * max(0.0, 970.0 - Q.sum_pt)
            + 1.4 * max(0.0, Q.z_dr_0p1_0p2 - 0.4)
            - 2.72 * max(0.0, 90.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3)
            + 0.0785 * max(0.0, 90.0 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.4)
            - 0.209 * max(0.0, 88.0 - Q.mass) * max(0.0, 0.075 - Q.z_dr_0p1_0p2)
            + 0.00124 * max(0.0, Q.sd_mass - 130.0) * max(0.0, 37.0 - Q.sj3_mass1)
            + 4.5e-05 * max(0.0, Q.sj3_pair_mass_max - 130.0) * max(0.0, 530.0 - Q.pt_0)
        ))
    )


def logit_W(Q):
    return (0.09375
        + 0.75 * grid(0, max(0.0, 1.12
            - 313.0 * max(0.0, 0.0061 - Q.e2_sq)
            - 94.5 * max(0.0, 0.0073 - Q.girth2_top20)
            - 0.146 * max(0.0, Q.mass - 78.0)
            + 0.0821 * max(0.0, Q.mass - 87.0)
            - 0.286 * max(0.0, Q.mass - 93.0)
            - 0.0115 * max(0.0, 100.0 - Q.mass)
            + 447.0 * max(0.0, 0.0077 - Q.mass_over_sum_pt_sq)
            + 0.0453 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.0082 * max(0.0, 980.0 - Q.sum_pt)
            + 0.0026 * max(0.0, 950.0 - Q.sum_pt_top30)
            - 0.0014 * max(0.0, 1100.0 - Q.sum_pt_top40)
            + 0.00151 * max(0.0, 1200.0 - Q.sum_pt_top50)
            - 19.3 * max(0.0, 0.07 - Q.tau1)
            - 3.85 * max(0.0, 0.091 - Q.z_dr_0p2_0p4)
            - 17.1 * max(0.0, 0.99 - Q.z_top50_slots)
            + 0.0885 * max(0.0, 7.0 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 960.0)
            + 2.66 * max(0.0, Q.mass - 78.0) * max(0.0, 0.07 - Q.sj2_zsoft)
            - 1.71 * max(0.0, Q.mass_top50 - 72.0) * max(0.0, 0.069 - Q.sj2_zsoft)
        ))
        + 0.4375 * grid(3, max(0.0, -0.0429
            + 1060.0 * max(0.0, 0.00061 - Q.lam2)
            - 0.0212 * max(0.0, 56.0 - Q.mass)
            + 0.0364 * max(0.0, 46.0 - Q.n_particles)
            - 2.01 * max(0.0, 0.38 - Q.tau21)
            + 0.802 * max(0.0, 14.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.95)
            - 0.000709 * max(0.0, 6.6 - Q.n_dr_0p2_0p4) * max(0.0, 1000.0 - Q.sum_pt_top30)
            + 2.92 * max(0.0, 6.4 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.97)
            - 312.0 * max(0.0, 0.38 - Q.tau21) * max(0.0, Q.lam1 - 0.0036)
        ))
        + 0.34375 * grid(4, max(0.0, 0.0131
            + 0.138 * max(0.0, 6.5 - Q.D2)
            - 8.68 * max(0.0, Q.LHA - 0.37)
            + 12.7 * max(0.0, Q.e2 - 0.038)
            + 85.2 * max(0.0, Q.e2_sq - 0.014)
            - 28.6 * max(0.0, Q.girth2_top15 - 0.0072)
            - 89.8 * max(0.0, 0.0054 - Q.girth2_top15)
            + 53.7 * max(0.0, Q.lam2 - 0.00044)
            + 0.0134 * max(0.0, Q.mass - 140.0)
            - 0.0578 * max(0.0, 74.0 - Q.mass)
            - 0.0639 * max(0.0, 87.0 - Q.mass)
            + 0.0426 * max(0.0, 100.0 - Q.mass)
            - 0.0163 * max(0.0, Q.mass_top20 - 100.0)
            + 0.0264 * max(0.0, Q.n_dr_0_0p05 - 6.8)
            + 0.0244 * max(0.0, 25.0 - Q.n_dr_0p2_0p4)
            - 0.0257 * max(0.0, Q.n_particles - 21.0)
            + 0.00629 * max(0.0, Q.sj3_pair_mass_min - 34.0)
            - 0.00279 * max(0.0, 1000.0 - Q.sum_pt)
            - 18.7 * max(0.0, 0.072 - Q.tau1)
            - 17.3 * max(0.0, 80.0 - Q.mass) * max(0.0, 0.0037 - Q.lam2)
            + 10.5 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.004 - Q.lam2)
            - 3.83 * max(0.0, 86.0 - Q.mass) * max(0.0, Q.zdr_0 - 0.0069)
            + 3.09e-05 * max(0.0, 120.0 - Q.mass_top30) * max(0.0, 1000.0 - Q.sum_pt_top40)
            - 0.0105 * max(0.0, 89.0 - Q.mass_top40) * max(0.0, 6.8 - Q.D2)
            + 0.00368 * max(0.0, Q.n_particles - 24.0) * max(0.0, 2.1 - Q.soft1_pt)
        ))
        + 0.578125 * grid(5, max(0.0, -0.119
            + 0.387 * max(0.0, 2.0 - Q.D2)
            + 16.8 * max(0.0, 0.041 - Q.e2)
            - 52.8 * max(0.0, Q.girth2_top50 - 0.016)
            - 19.4 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.0849 * max(0.0, Q.mass - 160.0)
            - 18.9 * max(0.0, Q.mass_over_sum_pt - 0.092)
            + 0.0144 * max(0.0, Q.mass_top40 - 59.0)
            - 0.567 * max(0.0, Q.max_dr - 0.33)
            + 1.69 * max(0.0, 0.42 - Q.max_dr)
            - 0.0202 * max(0.0, 22.0 - Q.n_dr_0p1_0p2)
            + 0.0369 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 0.0204 * max(0.0, Q.n_particles - 46.0)
            + 0.0321 * max(0.0, 46.0 - Q.n_particles)
            - 0.91 * max(0.0, 0.71 - Q.psi_0p1)
            + 0.0318 * max(0.0, Q.sd_mass - 70.0)
            - 0.0375 * max(0.0, Q.sd_mass - 83.0)
            + 0.0186 * max(0.0, 23.0 - Q.sj3_mass1)
            + 0.361 * max(0.0, Q.soft3_pt - 0.38)
            - 490.0 * max(0.0, Q.soft3_z - 0.00083)
            + 0.0157 * max(0.0, Q.sum_pt - 920.0)
            - 0.0229 * max(0.0, Q.sum_pt - 990.0)
            + 0.019 * max(0.0, Q.sum_pt_top50 - 950.0)
            - 1.12 * max(0.0, 0.49 - Q.tau21)
            - 10.7 * max(0.0, 0.036 - Q.z_dr_0p2_0p4)
            - 6.19 * max(0.0, Q.z_top30_slots - 0.9)
            - 0.0196 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.2 - Q.D2)
        ))
        + 0.0625 * grid(6, max(0.0, 1.27
            - 0.216 * max(0.0, 2.8 - Q.D2)
            - 24.5 * max(0.0, Q.e2 - 0.027)
            - 0.0211 * max(0.0, Q.mass - 140.0)
            + 0.093 * max(0.0, 84.0 - Q.mass)
            - 0.0571 * max(0.0, 120.0 - Q.mass)
            + 0.0406 * max(0.0, Q.mass_top10 - 95.0)
            - 0.0274 * max(0.0, Q.sj3_mass1 - 22.0)
            + 0.00838 * max(0.0, 76.0 - Q.sj3_pair_mass_min)
            + 3900.0 * max(0.0, Q.e2 - 0.022) * max(0.0, Q.psi_0p3 - 0.99)
            + 3.63e-05 * max(0.0, 150.0 - Q.mass) * max(0.0, 990.0 - Q.sum_pt_top50)
        ))
        - 0.625 * grid(7, max(0.0, -0.555
            + 13.9 * max(0.0, 0.31 - Q.LHA)
            + 320.0 * max(0.0, 0.0068 - Q.girth2_top30)
            - 530.0 * max(0.0, 0.0082 - Q.girth2_top30)
            + 194.0 * max(0.0, 0.012 - Q.girth2_top30)
            - 2740.0 * max(0.0, 0.0034 - Q.girth2_top50)
            + 0.107 * max(0.0, 81.0 - Q.mass)
            + 0.263 * max(0.0, 91.0 - Q.mass)
            - 0.285 * max(0.0, 93.0 - Q.mass)
            - 28.9 * max(0.0, 0.11 - Q.tau1)
            + 3.47 * max(0.0, 0.23 - Q.tau21_b2)
            - 10.7 * max(0.0, Q.z_top20_slots - 0.92)
            + 2490.0 * max(0.0, 0.0076 - Q.girth2) * max(0.0, 0.12 - Q.M2)
            - 8.38 * max(0.0, 91.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.96)
            + 5.64 * max(0.0, 100.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.96)
            - 0.00264 * max(0.0, 91.0 - Q.mass) * max(0.0, Q.sd_mass - 54.0)
            + 0.00239 * max(0.0, 120.0 - Q.mass) * max(0.0, Q.sd_mass - 76.0)
            - 1810.0 * max(0.0, 6.2 - Q.n_dr_0p2_0p4) * max(0.0, 6.7e-05 - Q.e3)
            + 0.00298 * max(0.0, 6.1 - Q.n_dr_0p2_0p4) * max(0.0, Q.pt_4 - 60.0)
            + 0.0741 * max(0.0, 6.3 - Q.n_dr_0p2_0p4) * max(0.0, 2.3 - Q.soft1_pt)
            - 22.2 * max(0.0, 0.25 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.19)
        ))
        - 0.875 * grid(8, max(0.0, -1.41
            + 39.7 * max(0.0, Q.e2 - 0.047)
            + 387.0 * max(0.0, Q.e2_sq - 0.008)
            - 466.0 * max(0.0, Q.e2_sq - 0.0098)
            + 2990.0 * max(0.0, 0.00035 - Q.e3)
            + 272.0 * max(0.0, 0.0082 - Q.girth2_top15)
            - 140.0 * max(0.0, 0.017 - Q.girth2_top15)
            + 10.1 * max(0.0, 6.9 - Q.log_sum_pt)
            + 0.0171 * max(0.0, Q.mass - 65.0)
            + 0.0695 * max(0.0, Q.mass - 81.0)
            - 0.0376 * max(0.0, Q.mass - 100.0)
            - 0.0408 * max(0.0, Q.mass - 120.0)
            + 233.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.00055)
            - 0.0651 * max(0.0, Q.mass_top50 - 81.0)
            + 0.0324 * max(0.0, Q.n_dr_0p2_0p4 - 8.1)
            - 0.0234 * max(0.0, 16.0 - Q.n_dr_0p2_0p4)
            + 0.0296 * max(0.0, Q.n_pt_above_1 - 55.0)
            + 4.27 * max(0.0, Q.sj2_dr - 0.23)
            - 163.0 * max(0.0, 0.0094 - Q.girth2_top15) * max(0.0, 0.61 - Q.tau21_b2)
            + 0.044 * max(0.0, Q.log_sum_pt - 7.0) * max(0.0, Q.sj3_pair_mass_max - 16.0)
            - 1.37 * max(0.0, Q.n_dr_0p2_0p4 - 8.9) * max(0.0, Q.zdr_0 - 0.0039)
        ))
        - 0.21875 * grid(9, max(0.0, 0.851
            + 13.8 * max(0.0, 0.21 - Q.LHA)
            + 33.8 * max(0.0, Q.girth - 0.093)
            - 50.7 * max(0.0, 0.047 - Q.girth)
            - 122.0 * max(0.0, Q.girth2_top30 - 0.024)
            + 283.0 * max(0.0, 0.0061 - Q.girth2_top40)
            - 0.039 * max(0.0, Q.mass - 140.0)
            - 0.0931 * max(0.0, Q.mass - 160.0)
            + 0.13 * max(0.0, Q.mass - 170.0)
            - 0.0537 * max(0.0, 64.0 - Q.mass)
            + 0.0704 * max(0.0, 84.0 - Q.mass)
            + 0.0323 * max(0.0, 130.0 - Q.mass_top40)
            - 0.0326 * max(0.0, 160.0 - Q.mass_top40)
            - 26.0 * max(0.0, Q.psi_0p3 - 0.99)
            + 0.00893 * max(0.0, 960.0 - Q.sum_pt_top50)
            + 1.22 * max(0.0, 0.29 - Q.z_dr_0p1_0p2)
            + 0.000696 * max(0.0, 88.0 - Q.mass) * max(0.0, Q.n_particles - 40.0)
            + 0.000726 * max(0.0, 110.0 - Q.mass_top30) * max(0.0, Q.n_dr_0p2_0p4 - 11.0)
            - 0.0003 * max(0.0, 72.0 - Q.mass_top40) * max(0.0, 1000.0 - Q.sum_pt)
            + 0.000433 * max(0.0, 73.0 - Q.mass_top40) * max(0.0, 940.0 - Q.sum_pt_top50)
            - 0.000715 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 19.0 - Q.n_pt_above_10)
        ))
        + 0.59375 * grid(11, max(0.0, -0.551
            + 6.03 * max(0.0, 0.24 - Q.LHA)
            + 224.0 * max(0.0, 0.0043 - Q.girth2_top10)
            - 85.9 * max(0.0, 0.0058 - Q.girth2_top2)
            - 161.0 * max(0.0, 0.0065 - Q.lam1)
            - 0.0565 * max(0.0, 81.0 - Q.mass)
            + 0.0564 * max(0.0, 96.0 - Q.mass)
            + 0.0379 * max(0.0, 17.0 - Q.n_dr_0p1_0p2)
            + 0.143 * max(0.0, 9.0 - Q.n_dr_0p2_0p4)
            + 45.6 * max(0.0, Q.psi_0p3 - 0.99)
            + 4.2 * max(0.0, 0.15 - Q.tau21_b2)
            - 1.55 * max(0.0, Q.z_top10_slots - 0.75)
            - 0.00729 * max(0.0, 84.0 - Q.mass) * max(0.0, 3.6 - Q.D2)
            + 0.00118 * max(0.0, 95.0 - Q.mass) * max(0.0, Q.mass_top10 - 38.0)
            - 28.5 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0061 - Q.girth2)
            - 0.0027 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 36.0)
        ))
        - 0.40625 * grid(12, max(0.0, -0.427
            - 0.0674 * max(0.0, 56.0 - Q.mass)
            + 0.101 * max(0.0, 84.0 - Q.mass)
            - 0.00243 * max(0.0, Q.sum_pt - 1200.0)
            - 0.0068 * max(0.0, 970.0 - Q.sum_pt)
            + 1.4 * max(0.0, Q.z_dr_0p1_0p2 - 0.4)
            - 2.72 * max(0.0, 90.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3)
            + 0.0785 * max(0.0, 90.0 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.4)
            - 0.209 * max(0.0, 88.0 - Q.mass) * max(0.0, 0.075 - Q.z_dr_0p1_0p2)
            + 0.00124 * max(0.0, Q.sd_mass - 130.0) * max(0.0, 37.0 - Q.sj3_mass1)
            + 4.5e-05 * max(0.0, Q.sj3_pair_mass_max - 130.0) * max(0.0, 530.0 - Q.pt_0)
        ))
        - 1.375 * grid(14, max(0.0, 0.24
            - 7.59 * max(0.0, Q.e2 - 0.018)
            + 8.03 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.0816 * max(0.0, 83.0 - Q.mass)
            - 0.0685 * max(0.0, 91.0 - Q.mass)
            - 64.3 * max(0.0, 0.091 - Q.mass_over_sum_pt)
            + 26.2 * max(0.0, 0.12 - Q.mass_over_sum_pt)
            - 0.0127 * max(0.0, Q.mass_top50 - 130.0)
            - 0.0215 * max(0.0, 20.0 - Q.n_dr_0p1_0p2)
            + 0.0217 * max(0.0, 24.0 - Q.n_pt_above_10)
            - 0.00635 * max(0.0, Q.sum_pt_top50 - 950.0)
            + 1.87 * max(0.0, 0.36 - Q.tau21_b2)
            - 13.8 * max(0.0, 0.39 - Q.N2) * max(0.0, Q.max_dr - 0.25)
            + 0.0147 * max(0.0, 83.0 - Q.mass) * max(0.0, Q.n_for_50pct - 4.3)
            + 68.5 * max(0.0, Q.psi_0p3 - 0.99) * max(0.0, 1.1 - Q.N3)
            + 0.464 * max(0.0, Q.psi_0p3 - 0.99) * max(0.0, Q.pt_3 - 63.0)
            - 0.0163 * max(0.0, 0.34 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -13.0)
            - 19.1 * max(0.0, 0.38 - Q.tau21_b2) * max(0.0, 0.12 - Q.z_1)
        ))
        - 0.1875 * grid(15, max(0.0, 0.227
            + 93.7 * max(0.0, 0.009 - Q.girth2_top5)
            - 372.0 * max(0.0, Q.lam1 - 0.017)
            + 234.0 * max(0.0, 0.002 - Q.lam2)
            + 5.95 * max(0.0, 7.0 - Q.log_sum_pt)
            - 0.0398 * max(0.0, Q.n_dr_0p2_0p4 - 8.6)
            - 4.07 * max(0.0, Q.psi_0p2 - 0.86)
            - 44.4 * max(0.0, Q.psi_0p3 - 0.99)
            - 0.0276 * max(0.0, 43.0 - Q.sd_mass)
            + 0.0206 * max(0.0, 76.0 - Q.sd_mass)
            + 4.54 * max(0.0, Q.sj2_dr - 0.23)
            - 3.0 * max(0.0, Q.sj3_dr_max - 0.36)
            - 0.0135 * max(0.0, 1000.0 - Q.sum_pt)
            - 21.0 * max(0.0, 0.069 - Q.tau1)
            - 0.676 * max(0.0, 0.97 - Q.z_dr_0_0p05)
            + 5.92 * max(0.0, 0.12 - Q.z_dr_0p1_0p2)
            - 3.51 * max(0.0, 0.0054 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.0)
            - 103.0 * max(0.0, 0.0094 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.08)
        ))
    )


def logit_Z(Q):
    return (0.984375
        - 1.375 * grid(0, max(0.0, 1.12
            - 313.0 * max(0.0, 0.0061 - Q.e2_sq)
            - 94.5 * max(0.0, 0.0073 - Q.girth2_top20)
            - 0.146 * max(0.0, Q.mass - 78.0)
            + 0.0821 * max(0.0, Q.mass - 87.0)
            - 0.286 * max(0.0, Q.mass - 93.0)
            - 0.0115 * max(0.0, 100.0 - Q.mass)
            + 447.0 * max(0.0, 0.0077 - Q.mass_over_sum_pt_sq)
            + 0.0453 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.0082 * max(0.0, 980.0 - Q.sum_pt)
            + 0.0026 * max(0.0, 950.0 - Q.sum_pt_top30)
            - 0.0014 * max(0.0, 1100.0 - Q.sum_pt_top40)
            + 0.00151 * max(0.0, 1200.0 - Q.sum_pt_top50)
            - 19.3 * max(0.0, 0.07 - Q.tau1)
            - 3.85 * max(0.0, 0.091 - Q.z_dr_0p2_0p4)
            - 17.1 * max(0.0, 0.99 - Q.z_top50_slots)
            + 0.0885 * max(0.0, 7.0 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 960.0)
            + 2.66 * max(0.0, Q.mass - 78.0) * max(0.0, 0.07 - Q.sj2_zsoft)
            - 1.71 * max(0.0, Q.mass_top50 - 72.0) * max(0.0, 0.069 - Q.sj2_zsoft)
        ))
        - 0.375 * grid(2, max(0.0, -0.0281
            - 6.25 * max(0.0, Q.log_sum_pt - 7.1)
            - 0.0111 * max(0.0, 87.0 - Q.mass)
            + 320.0 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.025 - Q.girth2_top15)
        ))
        + 0.4375 * grid(3, max(0.0, -0.0429
            + 1060.0 * max(0.0, 0.00061 - Q.lam2)
            - 0.0212 * max(0.0, 56.0 - Q.mass)
            + 0.0364 * max(0.0, 46.0 - Q.n_particles)
            - 2.01 * max(0.0, 0.38 - Q.tau21)
            + 0.802 * max(0.0, 14.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.95)
            - 0.000709 * max(0.0, 6.6 - Q.n_dr_0p2_0p4) * max(0.0, 1000.0 - Q.sum_pt_top30)
            + 2.92 * max(0.0, 6.4 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.97)
            - 312.0 * max(0.0, 0.38 - Q.tau21) * max(0.0, Q.lam1 - 0.0036)
        ))
        + 0.6875 * grid(5, max(0.0, -0.119
            + 0.387 * max(0.0, 2.0 - Q.D2)
            + 16.8 * max(0.0, 0.041 - Q.e2)
            - 52.8 * max(0.0, Q.girth2_top50 - 0.016)
            - 19.4 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.0849 * max(0.0, Q.mass - 160.0)
            - 18.9 * max(0.0, Q.mass_over_sum_pt - 0.092)
            + 0.0144 * max(0.0, Q.mass_top40 - 59.0)
            - 0.567 * max(0.0, Q.max_dr - 0.33)
            + 1.69 * max(0.0, 0.42 - Q.max_dr)
            - 0.0202 * max(0.0, 22.0 - Q.n_dr_0p1_0p2)
            + 0.0369 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 0.0204 * max(0.0, Q.n_particles - 46.0)
            + 0.0321 * max(0.0, 46.0 - Q.n_particles)
            - 0.91 * max(0.0, 0.71 - Q.psi_0p1)
            + 0.0318 * max(0.0, Q.sd_mass - 70.0)
            - 0.0375 * max(0.0, Q.sd_mass - 83.0)
            + 0.0186 * max(0.0, 23.0 - Q.sj3_mass1)
            + 0.361 * max(0.0, Q.soft3_pt - 0.38)
            - 490.0 * max(0.0, Q.soft3_z - 0.00083)
            + 0.0157 * max(0.0, Q.sum_pt - 920.0)
            - 0.0229 * max(0.0, Q.sum_pt - 990.0)
            + 0.019 * max(0.0, Q.sum_pt_top50 - 950.0)
            - 1.12 * max(0.0, 0.49 - Q.tau21)
            - 10.7 * max(0.0, 0.036 - Q.z_dr_0p2_0p4)
            - 6.19 * max(0.0, Q.z_top30_slots - 0.9)
            - 0.0196 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.2 - Q.D2)
        ))
        - 0.96875 * grid(6, max(0.0, 1.27
            - 0.216 * max(0.0, 2.8 - Q.D2)
            - 24.5 * max(0.0, Q.e2 - 0.027)
            - 0.0211 * max(0.0, Q.mass - 140.0)
            + 0.093 * max(0.0, 84.0 - Q.mass)
            - 0.0571 * max(0.0, 120.0 - Q.mass)
            + 0.0406 * max(0.0, Q.mass_top10 - 95.0)
            - 0.0274 * max(0.0, Q.sj3_mass1 - 22.0)
            + 0.00838 * max(0.0, 76.0 - Q.sj3_pair_mass_min)
            + 3900.0 * max(0.0, Q.e2 - 0.022) * max(0.0, Q.psi_0p3 - 0.99)
            + 3.63e-05 * max(0.0, 150.0 - Q.mass) * max(0.0, 990.0 - Q.sum_pt_top50)
        ))
        + 0.90625 * grid(7, max(0.0, -0.555
            + 13.9 * max(0.0, 0.31 - Q.LHA)
            + 320.0 * max(0.0, 0.0068 - Q.girth2_top30)
            - 530.0 * max(0.0, 0.0082 - Q.girth2_top30)
            + 194.0 * max(0.0, 0.012 - Q.girth2_top30)
            - 2740.0 * max(0.0, 0.0034 - Q.girth2_top50)
            + 0.107 * max(0.0, 81.0 - Q.mass)
            + 0.263 * max(0.0, 91.0 - Q.mass)
            - 0.285 * max(0.0, 93.0 - Q.mass)
            - 28.9 * max(0.0, 0.11 - Q.tau1)
            + 3.47 * max(0.0, 0.23 - Q.tau21_b2)
            - 10.7 * max(0.0, Q.z_top20_slots - 0.92)
            + 2490.0 * max(0.0, 0.0076 - Q.girth2) * max(0.0, 0.12 - Q.M2)
            - 8.38 * max(0.0, 91.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.96)
            + 5.64 * max(0.0, 100.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.96)
            - 0.00264 * max(0.0, 91.0 - Q.mass) * max(0.0, Q.sd_mass - 54.0)
            + 0.00239 * max(0.0, 120.0 - Q.mass) * max(0.0, Q.sd_mass - 76.0)
            - 1810.0 * max(0.0, 6.2 - Q.n_dr_0p2_0p4) * max(0.0, 6.7e-05 - Q.e3)
            + 0.00298 * max(0.0, 6.1 - Q.n_dr_0p2_0p4) * max(0.0, Q.pt_4 - 60.0)
            + 0.0741 * max(0.0, 6.3 - Q.n_dr_0p2_0p4) * max(0.0, 2.3 - Q.soft1_pt)
            - 22.2 * max(0.0, 0.25 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.19)
        ))
        - 0.9375 * grid(8, max(0.0, -1.41
            + 39.7 * max(0.0, Q.e2 - 0.047)
            + 387.0 * max(0.0, Q.e2_sq - 0.008)
            - 466.0 * max(0.0, Q.e2_sq - 0.0098)
            + 2990.0 * max(0.0, 0.00035 - Q.e3)
            + 272.0 * max(0.0, 0.0082 - Q.girth2_top15)
            - 140.0 * max(0.0, 0.017 - Q.girth2_top15)
            + 10.1 * max(0.0, 6.9 - Q.log_sum_pt)
            + 0.0171 * max(0.0, Q.mass - 65.0)
            + 0.0695 * max(0.0, Q.mass - 81.0)
            - 0.0376 * max(0.0, Q.mass - 100.0)
            - 0.0408 * max(0.0, Q.mass - 120.0)
            + 233.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.00055)
            - 0.0651 * max(0.0, Q.mass_top50 - 81.0)
            + 0.0324 * max(0.0, Q.n_dr_0p2_0p4 - 8.1)
            - 0.0234 * max(0.0, 16.0 - Q.n_dr_0p2_0p4)
            + 0.0296 * max(0.0, Q.n_pt_above_1 - 55.0)
            + 4.27 * max(0.0, Q.sj2_dr - 0.23)
            - 163.0 * max(0.0, 0.0094 - Q.girth2_top15) * max(0.0, 0.61 - Q.tau21_b2)
            + 0.044 * max(0.0, Q.log_sum_pt - 7.0) * max(0.0, Q.sj3_pair_mass_max - 16.0)
            - 1.37 * max(0.0, Q.n_dr_0p2_0p4 - 8.9) * max(0.0, Q.zdr_0 - 0.0039)
        ))
        + 0.3125 * grid(12, max(0.0, -0.427
            - 0.0674 * max(0.0, 56.0 - Q.mass)
            + 0.101 * max(0.0, 84.0 - Q.mass)
            - 0.00243 * max(0.0, Q.sum_pt - 1200.0)
            - 0.0068 * max(0.0, 970.0 - Q.sum_pt)
            + 1.4 * max(0.0, Q.z_dr_0p1_0p2 - 0.4)
            - 2.72 * max(0.0, 90.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3)
            + 0.0785 * max(0.0, 90.0 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.4)
            - 0.209 * max(0.0, 88.0 - Q.mass) * max(0.0, 0.075 - Q.z_dr_0p1_0p2)
            + 0.00124 * max(0.0, Q.sd_mass - 130.0) * max(0.0, 37.0 - Q.sj3_mass1)
            + 4.5e-05 * max(0.0, Q.sj3_pair_mass_max - 130.0) * max(0.0, 530.0 - Q.pt_0)
        ))
        + 0.5625 * grid(15, max(0.0, 0.227
            + 93.7 * max(0.0, 0.009 - Q.girth2_top5)
            - 372.0 * max(0.0, Q.lam1 - 0.017)
            + 234.0 * max(0.0, 0.002 - Q.lam2)
            + 5.95 * max(0.0, 7.0 - Q.log_sum_pt)
            - 0.0398 * max(0.0, Q.n_dr_0p2_0p4 - 8.6)
            - 4.07 * max(0.0, Q.psi_0p2 - 0.86)
            - 44.4 * max(0.0, Q.psi_0p3 - 0.99)
            - 0.0276 * max(0.0, 43.0 - Q.sd_mass)
            + 0.0206 * max(0.0, 76.0 - Q.sd_mass)
            + 4.54 * max(0.0, Q.sj2_dr - 0.23)
            - 3.0 * max(0.0, Q.sj3_dr_max - 0.36)
            - 0.0135 * max(0.0, 1000.0 - Q.sum_pt)
            - 21.0 * max(0.0, 0.069 - Q.tau1)
            - 0.676 * max(0.0, 0.97 - Q.z_dr_0_0p05)
            + 5.92 * max(0.0, 0.12 - Q.z_dr_0p1_0p2)
            - 3.51 * max(0.0, 0.0054 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.0)
            - 103.0 * max(0.0, 0.0094 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.08)
        ))
    )


def logit_t(Q):
    return (0.78125
        + 0.125 * grid(0, max(0.0, 1.12
            - 313.0 * max(0.0, 0.0061 - Q.e2_sq)
            - 94.5 * max(0.0, 0.0073 - Q.girth2_top20)
            - 0.146 * max(0.0, Q.mass - 78.0)
            + 0.0821 * max(0.0, Q.mass - 87.0)
            - 0.286 * max(0.0, Q.mass - 93.0)
            - 0.0115 * max(0.0, 100.0 - Q.mass)
            + 447.0 * max(0.0, 0.0077 - Q.mass_over_sum_pt_sq)
            + 0.0453 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.0082 * max(0.0, 980.0 - Q.sum_pt)
            + 0.0026 * max(0.0, 950.0 - Q.sum_pt_top30)
            - 0.0014 * max(0.0, 1100.0 - Q.sum_pt_top40)
            + 0.00151 * max(0.0, 1200.0 - Q.sum_pt_top50)
            - 19.3 * max(0.0, 0.07 - Q.tau1)
            - 3.85 * max(0.0, 0.091 - Q.z_dr_0p2_0p4)
            - 17.1 * max(0.0, 0.99 - Q.z_top50_slots)
            + 0.0885 * max(0.0, 7.0 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 960.0)
            + 2.66 * max(0.0, Q.mass - 78.0) * max(0.0, 0.07 - Q.sj2_zsoft)
            - 1.71 * max(0.0, Q.mass_top50 - 72.0) * max(0.0, 0.069 - Q.sj2_zsoft)
        ))
        + 0.1875 * grid(4, max(0.0, 0.0131
            + 0.138 * max(0.0, 6.5 - Q.D2)
            - 8.68 * max(0.0, Q.LHA - 0.37)
            + 12.7 * max(0.0, Q.e2 - 0.038)
            + 85.2 * max(0.0, Q.e2_sq - 0.014)
            - 28.6 * max(0.0, Q.girth2_top15 - 0.0072)
            - 89.8 * max(0.0, 0.0054 - Q.girth2_top15)
            + 53.7 * max(0.0, Q.lam2 - 0.00044)
            + 0.0134 * max(0.0, Q.mass - 140.0)
            - 0.0578 * max(0.0, 74.0 - Q.mass)
            - 0.0639 * max(0.0, 87.0 - Q.mass)
            + 0.0426 * max(0.0, 100.0 - Q.mass)
            - 0.0163 * max(0.0, Q.mass_top20 - 100.0)
            + 0.0264 * max(0.0, Q.n_dr_0_0p05 - 6.8)
            + 0.0244 * max(0.0, 25.0 - Q.n_dr_0p2_0p4)
            - 0.0257 * max(0.0, Q.n_particles - 21.0)
            + 0.00629 * max(0.0, Q.sj3_pair_mass_min - 34.0)
            - 0.00279 * max(0.0, 1000.0 - Q.sum_pt)
            - 18.7 * max(0.0, 0.072 - Q.tau1)
            - 17.3 * max(0.0, 80.0 - Q.mass) * max(0.0, 0.0037 - Q.lam2)
            + 10.5 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.004 - Q.lam2)
            - 3.83 * max(0.0, 86.0 - Q.mass) * max(0.0, Q.zdr_0 - 0.0069)
            + 3.09e-05 * max(0.0, 120.0 - Q.mass_top30) * max(0.0, 1000.0 - Q.sum_pt_top40)
            - 0.0105 * max(0.0, 89.0 - Q.mass_top40) * max(0.0, 6.8 - Q.D2)
            + 0.00368 * max(0.0, Q.n_particles - 24.0) * max(0.0, 2.1 - Q.soft1_pt)
        ))
        - 0.46875 * grid(5, max(0.0, -0.119
            + 0.387 * max(0.0, 2.0 - Q.D2)
            + 16.8 * max(0.0, 0.041 - Q.e2)
            - 52.8 * max(0.0, Q.girth2_top50 - 0.016)
            - 19.4 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.0849 * max(0.0, Q.mass - 160.0)
            - 18.9 * max(0.0, Q.mass_over_sum_pt - 0.092)
            + 0.0144 * max(0.0, Q.mass_top40 - 59.0)
            - 0.567 * max(0.0, Q.max_dr - 0.33)
            + 1.69 * max(0.0, 0.42 - Q.max_dr)
            - 0.0202 * max(0.0, 22.0 - Q.n_dr_0p1_0p2)
            + 0.0369 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 0.0204 * max(0.0, Q.n_particles - 46.0)
            + 0.0321 * max(0.0, 46.0 - Q.n_particles)
            - 0.91 * max(0.0, 0.71 - Q.psi_0p1)
            + 0.0318 * max(0.0, Q.sd_mass - 70.0)
            - 0.0375 * max(0.0, Q.sd_mass - 83.0)
            + 0.0186 * max(0.0, 23.0 - Q.sj3_mass1)
            + 0.361 * max(0.0, Q.soft3_pt - 0.38)
            - 490.0 * max(0.0, Q.soft3_z - 0.00083)
            + 0.0157 * max(0.0, Q.sum_pt - 920.0)
            - 0.0229 * max(0.0, Q.sum_pt - 990.0)
            + 0.019 * max(0.0, Q.sum_pt_top50 - 950.0)
            - 1.12 * max(0.0, 0.49 - Q.tau21)
            - 10.7 * max(0.0, 0.036 - Q.z_dr_0p2_0p4)
            - 6.19 * max(0.0, Q.z_top30_slots - 0.9)
            - 0.0196 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.2 - Q.D2)
        ))
        - 0.28125 * grid(7, max(0.0, -0.555
            + 13.9 * max(0.0, 0.31 - Q.LHA)
            + 320.0 * max(0.0, 0.0068 - Q.girth2_top30)
            - 530.0 * max(0.0, 0.0082 - Q.girth2_top30)
            + 194.0 * max(0.0, 0.012 - Q.girth2_top30)
            - 2740.0 * max(0.0, 0.0034 - Q.girth2_top50)
            + 0.107 * max(0.0, 81.0 - Q.mass)
            + 0.263 * max(0.0, 91.0 - Q.mass)
            - 0.285 * max(0.0, 93.0 - Q.mass)
            - 28.9 * max(0.0, 0.11 - Q.tau1)
            + 3.47 * max(0.0, 0.23 - Q.tau21_b2)
            - 10.7 * max(0.0, Q.z_top20_slots - 0.92)
            + 2490.0 * max(0.0, 0.0076 - Q.girth2) * max(0.0, 0.12 - Q.M2)
            - 8.38 * max(0.0, 91.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.96)
            + 5.64 * max(0.0, 100.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.96)
            - 0.00264 * max(0.0, 91.0 - Q.mass) * max(0.0, Q.sd_mass - 54.0)
            + 0.00239 * max(0.0, 120.0 - Q.mass) * max(0.0, Q.sd_mass - 76.0)
            - 1810.0 * max(0.0, 6.2 - Q.n_dr_0p2_0p4) * max(0.0, 6.7e-05 - Q.e3)
            + 0.00298 * max(0.0, 6.1 - Q.n_dr_0p2_0p4) * max(0.0, Q.pt_4 - 60.0)
            + 0.0741 * max(0.0, 6.3 - Q.n_dr_0p2_0p4) * max(0.0, 2.3 - Q.soft1_pt)
            - 22.2 * max(0.0, 0.25 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.19)
        ))
        + 0.2109375 * grid(8, max(0.0, -1.41
            + 39.7 * max(0.0, Q.e2 - 0.047)
            + 387.0 * max(0.0, Q.e2_sq - 0.008)
            - 466.0 * max(0.0, Q.e2_sq - 0.0098)
            + 2990.0 * max(0.0, 0.00035 - Q.e3)
            + 272.0 * max(0.0, 0.0082 - Q.girth2_top15)
            - 140.0 * max(0.0, 0.017 - Q.girth2_top15)
            + 10.1 * max(0.0, 6.9 - Q.log_sum_pt)
            + 0.0171 * max(0.0, Q.mass - 65.0)
            + 0.0695 * max(0.0, Q.mass - 81.0)
            - 0.0376 * max(0.0, Q.mass - 100.0)
            - 0.0408 * max(0.0, Q.mass - 120.0)
            + 233.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.00055)
            - 0.0651 * max(0.0, Q.mass_top50 - 81.0)
            + 0.0324 * max(0.0, Q.n_dr_0p2_0p4 - 8.1)
            - 0.0234 * max(0.0, 16.0 - Q.n_dr_0p2_0p4)
            + 0.0296 * max(0.0, Q.n_pt_above_1 - 55.0)
            + 4.27 * max(0.0, Q.sj2_dr - 0.23)
            - 163.0 * max(0.0, 0.0094 - Q.girth2_top15) * max(0.0, 0.61 - Q.tau21_b2)
            + 0.044 * max(0.0, Q.log_sum_pt - 7.0) * max(0.0, Q.sj3_pair_mass_max - 16.0)
            - 1.37 * max(0.0, Q.n_dr_0p2_0p4 - 8.9) * max(0.0, Q.zdr_0 - 0.0039)
        ))
        + 0.984375 * grid(10, max(0.0, 0.67
            + 0.297 * max(0.0, 3.8 - Q.D2)
            - 5.67 * max(0.0, Q.M2 - 0.055)
            + 6.11 * max(0.0, 0.064 - Q.dr_0)
            - 27.1 * max(0.0, 0.062 - Q.e2)
            - 47.8 * max(0.0, 0.005 - Q.girth2_top20)
            + 125.0 * max(0.0, 0.018 - Q.girth2_top30)
            - 3.91 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.0497 * max(0.0, Q.mass - 61.0)
            + 0.077 * max(0.0, Q.mass - 81.0)
            + 0.0191 * max(0.0, Q.mass - 88.0)
            - 0.0729 * max(0.0, Q.mass - 140.0)
            - 0.0852 * max(0.0, Q.mass - 160.0)
            - 318.0 * max(0.0, Q.mass_over_sum_pt - 0.17)
            + 18.6 * max(0.0, 0.078 - Q.mass_over_sum_pt)
            + 747.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
            - 0.00737 * max(0.0, 66.0 - Q.mass_top10)
            + 0.00873 * max(0.0, Q.mass_top5 - 15.0)
            + 0.0401 * max(0.0, Q.mass_top50 - 140.0)
            - 0.0147 * max(0.0, 78.0 - Q.sj2_mass1)
            - 0.00171 * max(0.0, 990.0 - Q.sum_pt)
            + 0.0017 * max(0.0, 850.0 - Q.sum_pt_top5)
            - 43.2 * max(0.0, 0.056 - Q.tau1)
            - 3.55 * max(0.0, 3.6 - Q.D2) * max(0.0, Q.sj2_dr - 0.2)
            + 0.146 * max(0.0, Q.mass - 90.0) * max(0.0, Q.psi_0p3 - 0.98)
            + 0.00409 * max(0.0, Q.mass - 59.0) * max(0.0, 2.7 - Q.soft2_pt)
            + 0.000774 * max(0.0, 110.0 - Q.mass_top40) * max(0.0, Q.pt1_dr01 - 13.0)
            + 28.4 * max(0.0, Q.mass_top40 - 160.0) * max(0.0, Q.soft4_z - 0.0011)
            + 2.53e-05 * max(0.0, 89.0 - Q.sj2_mass1) * max(0.0, 530.0 - Q.sum_pt_top2)
        ))
        - 0.375 * grid(12, max(0.0, -0.427
            - 0.0674 * max(0.0, 56.0 - Q.mass)
            + 0.101 * max(0.0, 84.0 - Q.mass)
            - 0.00243 * max(0.0, Q.sum_pt - 1200.0)
            - 0.0068 * max(0.0, 970.0 - Q.sum_pt)
            + 1.4 * max(0.0, Q.z_dr_0p1_0p2 - 0.4)
            - 2.72 * max(0.0, 90.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3)
            + 0.0785 * max(0.0, 90.0 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.4)
            - 0.209 * max(0.0, 88.0 - Q.mass) * max(0.0, 0.075 - Q.z_dr_0p1_0p2)
            + 0.00124 * max(0.0, Q.sd_mass - 130.0) * max(0.0, 37.0 - Q.sj3_mass1)
            + 4.5e-05 * max(0.0, Q.sj3_pair_mass_max - 130.0) * max(0.0, 530.0 - Q.pt_0)
        ))
        - 0.90625 * grid(13, max(0.0, 1.86
            + 0.0195 * max(0.0, Q.mass - 77.0)
            - 0.147 * max(0.0, Q.mass - 140.0)
            + 103.0 * max(0.0, Q.mass_over_sum_pt - 0.17)
            + 0.0722 * max(0.0, Q.mass_top40 - 150.0)
            - 0.016 * max(0.0, Q.mass_top50 - 97.0)
            + 0.0829 * max(0.0, Q.mass_top50 - 140.0)
            - 0.128 * max(0.0, Q.mass_top50 - 160.0)
            + 17.4 * max(0.0, Q.psi_0p3 - 0.98)
            - 0.0314 * max(0.0, 1000.0 - Q.sum_pt)
            - 0.00984 * max(0.0, 1100.0 - Q.sum_pt)
            - 0.00215 * max(0.0, 880.0 - Q.sum_pt_top20)
            + 0.0042 * max(0.0, 1100.0 - Q.sum_pt_top40)
            + 0.0139 * max(0.0, 1000.0 - Q.sum_pt_top50)
            + 1.27 * max(0.0, 990.0 - Q.sum_pt) * max(0.0, 0.018 - Q.z_9)
        ))
        - 0.375 * grid(15, max(0.0, 0.227
            + 93.7 * max(0.0, 0.009 - Q.girth2_top5)
            - 372.0 * max(0.0, Q.lam1 - 0.017)
            + 234.0 * max(0.0, 0.002 - Q.lam2)
            + 5.95 * max(0.0, 7.0 - Q.log_sum_pt)
            - 0.0398 * max(0.0, Q.n_dr_0p2_0p4 - 8.6)
            - 4.07 * max(0.0, Q.psi_0p2 - 0.86)
            - 44.4 * max(0.0, Q.psi_0p3 - 0.99)
            - 0.0276 * max(0.0, 43.0 - Q.sd_mass)
            + 0.0206 * max(0.0, 76.0 - Q.sd_mass)
            + 4.54 * max(0.0, Q.sj2_dr - 0.23)
            - 3.0 * max(0.0, Q.sj3_dr_max - 0.36)
            - 0.0135 * max(0.0, 1000.0 - Q.sum_pt)
            - 21.0 * max(0.0, 0.069 - Q.tau1)
            - 0.676 * max(0.0, 0.97 - Q.z_dr_0_0p05)
            + 5.92 * max(0.0, 0.12 - Q.z_dr_0p1_0p2)
            - 3.51 * max(0.0, 0.0054 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.0)
            - 103.0 * max(0.0, 0.0094 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.08)
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
