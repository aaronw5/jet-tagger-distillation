"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network (step 4; all observables), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 89.8% of jets.

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
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
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
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        phi_0=phi[0],
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
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
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-0.4375
        - 0.15625 * grid(0, max(0.0, -0.464
            - 930.0 * max(0.0, 0.00151 - Q.C2_b2)
            - 50.7 * max(0.0, Q.centroid_offset - 0.00783)
            - 347.0 * max(0.0, Q.centroid_offset - 0.0499)
            + 71.7 * max(0.0, 0.0243 - Q.e2)
            + 81.8 * max(0.0, 0.0354 - Q.e2)
            - 47.9 * max(0.0, 0.0751 - Q.girth)
            + 497.0 * max(0.0, 0.013 - Q.girth2)
            - 0.103 * max(0.0, 29.6 - Q.mass)
            - 0.0679 * max(0.0, 64.7 - Q.mass)
            + 6.63 * max(0.0, 0.159 - Q.planar_flow)
            + 11.9 * max(0.0, Q.sj3_dr_max - 0.11)
            - 27.2 * max(0.0, Q.sj3_dr_max - 0.182)
            - 0.0134 * max(0.0, Q.sum_pt - 906.0)
            - 1120.0 * max(0.0, 0.00444 - Q.width)
            - 1540.0 * max(0.0, Q.centroid_offset - 0.0203) * max(0.0, 0.0609 - Q.z_7)
            + 172.0 * max(0.0, 0.0133 - Q.girth2) * max(0.0, 1.02 - Q.D2)
            + 818.0 * max(0.0, 0.0134 - Q.girth2) * max(0.0, Q.phi_0 - -0.00668)
            - 1580.0 * max(0.0, 0.00649 - Q.lam1) * max(0.0, 0.843 - Q.D2)
            + 55000.0 * max(0.0, 8.01e-05 - Q.lam2) * max(0.0, 0.257 - Q.D2_b2)
            + 49.3 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, 0.0746 - Q.dr_4)
            - 458.0 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, Q.z_7 - 0.0178)
            - 661.0 * max(0.0, 0.142 - Q.planar_flow) * max(0.0, 0.023 - Q.dr_2)
            + 510.0 * max(0.0, 0.165 - Q.planar_flow) * max(0.0, 0.0237 - Q.z_7)
            - 190.0 * max(0.0, 0.245 - Q.sj3_dr_max) * max(0.0, 0.058 - Q.z_7)
            + 0.000467 * max(0.0, Q.sum_pt - 915.0) * max(0.0, Q.pt_7 - 35.4)
            + 8830.0 * max(0.0, 0.00414 - Q.width) * max(0.0, 0.0364 - Q.C3)
        ))
        + 0.390625 * grid(1, max(0.0, 1.28
            - 29.9 * max(0.0, 0.0991 - Q.girth)
            - 196.0 * max(0.0, 0.00871 - Q.girth2)
            - 736.0 * max(0.0, 0.00085 - Q.lam1)
            + 10.3 * max(0.0, Q.log_sum_pt - 6.5)
            + 5.23 * max(0.0, Q.log_sum_pt - 6.62)
            + 323.0 * max(0.0, 0.00591 - Q.mass_over_sum_pt_sq)
            - 0.113 * max(0.0, Q.pt_7 - 53.5)
            + 0.0053 * max(0.0, Q.sum_pt_top5 - 364.0)
            - 0.0129 * max(0.0, Q.sum_pt_top5 - 528.0)
            + 21.6 * max(0.0, Q.z_7 - 0.0586)
            - 35.5 * max(0.0, 0.0619 - Q.z_7)
            + 6070.0 * max(0.0, 0.00939 - Q.girth2) * max(0.0, Q.centroid_offset - 0.0216)
            - 46.4 * max(0.0, 0.00871 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.25)
            - 596.0 * max(0.0, 0.00962 - Q.lam1) * max(0.0, 0.15 - Q.planar_flow)
            + 3.11 * max(0.0, Q.log_sum_pt - 6.48) * max(0.0, 1.4 - Q.D2)
            + 82.0 * max(0.0, Q.log_sum_pt - 6.37) * max(0.0, Q.centroid_offset - 0.0126)
            - 0.00445 * max(0.0, Q.pt_7 - 33.4) * max(0.0, 53.4 - Q.pt_6)
            + 0.477 * max(0.0, Q.pt_7 - 33.5) * max(0.0, Q.sj2_dr - 0.143)
            - 0.0846 * max(0.0, Q.sj3_dr_max - 0.104) * max(0.0, Q.sj3_pair_mass_min - 2.6)
            - 36.3 * max(0.0, 0.0607 - Q.z_7) * max(0.0, 1.52 - Q.D2)
        ))
        + 0.4296875 * grid(2, max(0.0, 4.22
            - 12.2 * max(0.0, Q.LHA - 0.131)
            - 144.0 * max(0.0, 0.00765 - Q.girth)
            + 1810.0 * max(0.0, 0.000201 - Q.lam2)
            + 4.06 * max(0.0, 6.48 - Q.log_sum_pt)
            + 0.0269 * max(0.0, Q.m012 - 42.1)
            - 0.0236 * max(0.0, 35.5 - Q.mass)
            + 24.2 * max(0.0, 0.105 - Q.mass_over_sum_pt)
            - 2.21 * max(0.0, 0.236 - Q.max_dr)
            - 0.933 * max(0.0, 0.502 - Q.planar_flow)
            + 0.0976 * max(0.0, Q.pt_7 - 43.7)
            - 0.0939 * max(0.0, 52.4 - Q.pt_7)
            + 0.00368 * max(0.0, Q.sum_pt - 923.0)
            + 0.00878 * max(0.0, 807.0 - Q.sum_pt)
            - 43.8 * max(0.0, Q.z_7 - 0.0366)
            - 2060.0 * max(0.0, 0.00609 - Q.lam1) * max(0.0, Q.max_dr - 0.0769)
            + 3.28 * max(0.0, 0.00496 - Q.lam1) * max(0.0, Q.pt_6 - 17.3)
            - 9.83 * max(0.0, Q.log_sum_pt - 6.85) * max(0.0, 1.31 - Q.D2_b2)
            + 1.45 * max(0.0, 6.33 - Q.log_sum_pt) * max(0.0, 1.14 - Q.D2_b2)
            - 0.173 * max(0.0, Q.log_sum_pt - 6.95) * max(0.0, Q.pt_6 - 40.0)
            - 0.73 * max(0.0, 59.4 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.0112)
            - 12.6 * max(0.0, Q.sum_pt - 538.0) * max(0.0, 0.000194 - Q.lam2)
            + 0.00862 * max(0.0, Q.sum_pt_top5 - 745.0) * max(0.0, 1.1 - Q.D2_b2)
            - 319.0 * max(0.0, Q.z_7 - 0.0548) * max(0.0, 0.0684 - Q.sj3_dr_min)
        ))
        - 0.03125 * grid(4, max(0.0, 0.0436
            + 76.8 * max(0.0, 0.0651 - Q.C2)
            + 13.2 * max(0.0, 0.351 - Q.LHA)
            + 53.9 * max(0.0, 0.224 - Q.N2)
            + 648.0 * max(0.0, 0.0116 - Q.e2_sq)
            - 969.0 * max(0.0, 0.00919 - Q.girth2)
            + 220.0 * max(0.0, 0.00779 - Q.girth2_top2)
            - 2750.0 * max(0.0, 0.000876 - Q.lam2)
            - 0.0739 * max(0.0, Q.mass - 45.1)
            - 64.2 * max(0.0, Q.mass_over_sum_pt - 0.0903)
            - 21.7 * max(0.0, 0.237 - Q.sj3_dr_max)
            - 0.00773 * max(0.0, 753.0 - Q.sum_pt)
            + 313.0 * max(0.0, Q.width - 0.00339)
            - 1740.0 * max(0.0, 0.219 - Q.N2) * max(0.0, Q.e2_sq - 0.0116)
            - 136.0 * max(0.0, 0.224 - Q.N2) * max(0.0, Q.eccentricity - 0.727)
            - 0.848 * max(0.0, 0.219 - Q.N2) * max(0.0, 63.2 - Q.mass)
            - 0.332 * max(0.0, 0.242 - Q.N2) * max(0.0, 52.9 - Q.pt_7)
            - 19900.0 * max(0.0, 0.0076 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.000612)
            - 17300.0 * max(0.0, 0.000453 - Q.lam2) * max(0.0, 0.173 - Q.dr01)
            - 45400.0 * max(0.0, Q.max_dr - 0.164) * max(0.0, 0.000596 - Q.C2_b2)
            - 0.321 * max(0.0, Q.sd_mass - 37.0) * max(0.0, 0.276 - Q.sd_zg)
            + 0.00274 * max(0.0, Q.sd_mass - 42.1) * max(0.0, 26.6 - Q.sj3_pair_mass_min)
        ))
        - 0.1875 * grid(5, max(0.0, -0.444
            + 240.0 * max(0.0, 0.00764 - Q.girth)
            - 22.2 * max(0.0, Q.log_sum_pt - 6.83)
            - 0.132 * max(0.0, 35.4 - Q.pt_7)
            + 569.0 * max(0.0, 0.0039 - Q.width)
            + 132.0 * max(0.0, 0.0286 - Q.z_7)
            + 69.9 * max(0.0, 0.0473 - Q.z_7)
            + 61.0 * max(0.0, 0.0702 - Q.z_7)
            - 45.2 * max(0.0, 0.021 - Q.zdr_0)
            - 11700.0 * max(0.0, 0.215 - Q.LHA) * max(0.0, 0.00121 - Q.lam1)
            - 67.1 * max(0.0, 0.215 - Q.LHA) * max(0.0, 6.78 - Q.log_sum_pt)
            + 135000.0 * max(0.0, 0.00167 - Q.girth2) * max(0.0, 0.024 - Q.centroid_offset)
            + 981.0 * max(0.0, Q.log_sum_pt - 6.73) * max(0.0, 0.0147 - Q.dr_2)
            - 924.0 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.0244 - Q.dr_2)
            + 84600.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 8.48e-05 - Q.mean_eta2)
            - 902.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, Q.mean_phi - 0.00334)
            + 0.000174 * max(0.0, Q.sum_pt_top5 - 530.0) * max(0.0, 43.2 - Q.pt_6)
            + 0.019 * max(0.0, Q.sum_pt_top5 - 438.0) * max(0.0, Q.sj2_dr - 0.176)
            - 1450.0 * max(0.0, 0.0687 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
            + 0.354 * max(0.0, 0.0708 - Q.z_7) * max(0.0, 45.4 - Q.mass_top3)
        ))
        + 0.109375 * grid(6, max(0.0, 2.65
            + 106.0 * max(0.0, Q.centroid_offset - 0.00762)
            - 102.0 * max(0.0, Q.centroid_offset - 0.0193)
            + 547.0 * max(0.0, 0.00301 - Q.e2_sq)
            - 1020.0 * max(0.0, 0.00867 - Q.girth2)
            + 298.0 * max(0.0, 0.0118 - Q.lam1)
            - 1090.0 * max(0.0, 0.00109 - Q.lam2)
            + 0.0803 * max(0.0, 22.5 - Q.mass)
            - 34.4 * max(0.0, 0.144 - Q.max_dr)
            + 0.513 * max(0.0, 19.1 - Q.pt_6)
            + 8.14 * max(0.0, Q.sj3_dr_max - 0.191)
            + 30.4 * max(0.0, 0.181 - Q.sj3_dr_max)
            - 9.6 * max(0.0, 0.29 - Q.sj3_dr_max)
            - 17.8 * max(0.0, Q.sj3_dr_min - 0.0225)
            - 0.0953 * max(0.0, Q.sj3_pair_mass_min - 4.35)
            + 0.0162 * max(0.0, Q.sum_pt - 975.0)
            - 437.0 * max(0.0, 0.0221 - Q.z_6)
            + 5830.0 * max(0.0, Q.centroid_offset - 0.0213) * max(0.0, 0.01 - Q.mean_phi2)
            + 65.4 * max(0.0, Q.centroid_offset - 0.0103) * max(0.0, Q.psi_0p1 - 0.335)
            + 1.84 * max(0.0, Q.centroid_offset - 0.017) * max(0.0, Q.pt_4 - 66.3)
            - 1140.0 * max(0.0, 0.0118 - Q.lam1) * max(0.0, 0.254 - Q.planar_flow)
            + 57.4 * max(0.0, 0.0032 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05)
            - 0.707 * max(0.0, 64.5 - Q.mass) * max(0.0, 0.0534 - Q.z_dr_0p2_0p4)
            - 1210.0 * max(0.0, Q.mean_eta - 0.0349) * max(0.0, Q.M3 - 0.0492)
            + 0.589 * max(0.0, 42.5 - Q.pt_6) * max(0.0, 6.76 - Q.log_sum_pt)
            - 6.53 * max(0.0, 41.5 - Q.pt_6) * max(0.0, Q.z_7 - 0.0235)
            + 176.0 * max(0.0, 0.171 - Q.sj2_dr) * max(0.0, Q.eccentricity - 0.927)
            - 3.68 * max(0.0, Q.sum_pt - 953.0) * max(0.0, 0.00311 - Q.mean_phi2)
            + 0.00255 * max(0.0, 589.0 - Q.sum_pt) * max(0.0, 1.97 - Q.n_dr_0p2_0p4)
            + 0.00863 * max(0.0, 625.0 - Q.sum_pt) * max(0.0, 24.6 - Q.pt_5)
            + 1560.0 * max(0.0, 0.111 - Q.tau1) * max(0.0, 0.0157 - Q.mean_phi2)
        ))
        + 0.171875 * grid(9, max(0.0, -5.15
            - 0.17 * max(0.0, Q.D2 - 2.16)
            + 12.7 * max(0.0, 0.0796 - Q.M3)
            + 182.0 * max(0.0, 0.0185 - Q.centroid_offset)
            + 43.6 * max(0.0, Q.e2 - 0.0456)
            - 1790.0 * max(0.0, 0.00327 - Q.e2_sq)
            + 279.0 * max(0.0, Q.girth2 - 0.0134)
            + 2880.0 * max(0.0, 0.00365 - Q.girth2)
            + 865.0 * max(0.0, 0.00601 - Q.girth2)
            + 5410.0 * max(0.0, 0.000315 - Q.lam2)
            - 0.0971 * max(0.0, 30.1 - Q.mass)
            - 0.062 * max(0.0, 52.0 - Q.mass)
            + 18.6 * max(0.0, 0.107 - Q.max_dr)
            - 55.0 * max(0.0, 0.141 - Q.sj3_dr_max)
            + 21.9 * max(0.0, 0.215 - Q.sj3_dr_max)
            + 0.0077 * max(0.0, 986.0 - Q.sum_pt)
            + 7810.0 * max(0.0, 0.000176 - Q.width)
            - 161.0 * max(0.0, 0.00631 - Q.zdr_0)
            - 33.6 * max(0.0, 0.017 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0)
            - 2.01 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, 164.0 - Q.pt_1)
            + 1090.0 * max(0.0, 0.0191 - Q.centroid_offset) * max(0.0, 0.205 - Q.z_2nd)
            + 3990.0 * max(0.0, 0.0528 - Q.girth) * max(0.0, 0.0275 - Q.tau21_b2)
            - 1380.0 * max(0.0, 0.0549 - Q.girth) * max(0.0, Q.z_6 - 0.0338)
            - 9500.0 * max(0.0, 0.00578 - Q.girth2) * max(0.0, 0.00437 - Q.mean_phi)
            - 859.0 * max(0.0, Q.lam1 - 0.00485) * max(0.0, Q.eccentricity - 0.638)
            - 27.2 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.746 - Q.D2_b2)
            + 4.38 * max(0.0, 54.9 - Q.mass) * max(0.0, 0.0267 - Q.centroid_offset)
            + 0.129 * max(0.0, 55.0 - Q.mass) * max(0.0, 6.84 - Q.log_sum_pt)
            + 0.581 * max(0.0, 0.15 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 32.9)
            + 3360.0 * max(0.0, 0.046 - Q.tau1) * max(0.0, -0.0119 - Q.mean_phi)
            - 4610.0 * max(0.0, 0.0457 - Q.tau1) * max(0.0, 0.0286 - Q.tau21_b2)
        ))
    )


def logit_q(Q):
    return (0.03125
        - 0.09375 * grid(4, max(0.0, 0.0436
            + 76.8 * max(0.0, 0.0651 - Q.C2)
            + 13.2 * max(0.0, 0.351 - Q.LHA)
            + 53.9 * max(0.0, 0.224 - Q.N2)
            + 648.0 * max(0.0, 0.0116 - Q.e2_sq)
            - 969.0 * max(0.0, 0.00919 - Q.girth2)
            + 220.0 * max(0.0, 0.00779 - Q.girth2_top2)
            - 2750.0 * max(0.0, 0.000876 - Q.lam2)
            - 0.0739 * max(0.0, Q.mass - 45.1)
            - 64.2 * max(0.0, Q.mass_over_sum_pt - 0.0903)
            - 21.7 * max(0.0, 0.237 - Q.sj3_dr_max)
            - 0.00773 * max(0.0, 753.0 - Q.sum_pt)
            + 313.0 * max(0.0, Q.width - 0.00339)
            - 1740.0 * max(0.0, 0.219 - Q.N2) * max(0.0, Q.e2_sq - 0.0116)
            - 136.0 * max(0.0, 0.224 - Q.N2) * max(0.0, Q.eccentricity - 0.727)
            - 0.848 * max(0.0, 0.219 - Q.N2) * max(0.0, 63.2 - Q.mass)
            - 0.332 * max(0.0, 0.242 - Q.N2) * max(0.0, 52.9 - Q.pt_7)
            - 19900.0 * max(0.0, 0.0076 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.000612)
            - 17300.0 * max(0.0, 0.000453 - Q.lam2) * max(0.0, 0.173 - Q.dr01)
            - 45400.0 * max(0.0, Q.max_dr - 0.164) * max(0.0, 0.000596 - Q.C2_b2)
            - 0.321 * max(0.0, Q.sd_mass - 37.0) * max(0.0, 0.276 - Q.sd_zg)
            + 0.00274 * max(0.0, Q.sd_mass - 42.1) * max(0.0, 26.6 - Q.sj3_pair_mass_min)
        ))
        + 0.046875 * grid(5, max(0.0, -0.444
            + 240.0 * max(0.0, 0.00764 - Q.girth)
            - 22.2 * max(0.0, Q.log_sum_pt - 6.83)
            - 0.132 * max(0.0, 35.4 - Q.pt_7)
            + 569.0 * max(0.0, 0.0039 - Q.width)
            + 132.0 * max(0.0, 0.0286 - Q.z_7)
            + 69.9 * max(0.0, 0.0473 - Q.z_7)
            + 61.0 * max(0.0, 0.0702 - Q.z_7)
            - 45.2 * max(0.0, 0.021 - Q.zdr_0)
            - 11700.0 * max(0.0, 0.215 - Q.LHA) * max(0.0, 0.00121 - Q.lam1)
            - 67.1 * max(0.0, 0.215 - Q.LHA) * max(0.0, 6.78 - Q.log_sum_pt)
            + 135000.0 * max(0.0, 0.00167 - Q.girth2) * max(0.0, 0.024 - Q.centroid_offset)
            + 981.0 * max(0.0, Q.log_sum_pt - 6.73) * max(0.0, 0.0147 - Q.dr_2)
            - 924.0 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.0244 - Q.dr_2)
            + 84600.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 8.48e-05 - Q.mean_eta2)
            - 902.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, Q.mean_phi - 0.00334)
            + 0.000174 * max(0.0, Q.sum_pt_top5 - 530.0) * max(0.0, 43.2 - Q.pt_6)
            + 0.019 * max(0.0, Q.sum_pt_top5 - 438.0) * max(0.0, Q.sj2_dr - 0.176)
            - 1450.0 * max(0.0, 0.0687 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
            + 0.354 * max(0.0, 0.0708 - Q.z_7) * max(0.0, 45.4 - Q.mass_top3)
        ))
        + 0.125 * grid(6, max(0.0, 2.65
            + 106.0 * max(0.0, Q.centroid_offset - 0.00762)
            - 102.0 * max(0.0, Q.centroid_offset - 0.0193)
            + 547.0 * max(0.0, 0.00301 - Q.e2_sq)
            - 1020.0 * max(0.0, 0.00867 - Q.girth2)
            + 298.0 * max(0.0, 0.0118 - Q.lam1)
            - 1090.0 * max(0.0, 0.00109 - Q.lam2)
            + 0.0803 * max(0.0, 22.5 - Q.mass)
            - 34.4 * max(0.0, 0.144 - Q.max_dr)
            + 0.513 * max(0.0, 19.1 - Q.pt_6)
            + 8.14 * max(0.0, Q.sj3_dr_max - 0.191)
            + 30.4 * max(0.0, 0.181 - Q.sj3_dr_max)
            - 9.6 * max(0.0, 0.29 - Q.sj3_dr_max)
            - 17.8 * max(0.0, Q.sj3_dr_min - 0.0225)
            - 0.0953 * max(0.0, Q.sj3_pair_mass_min - 4.35)
            + 0.0162 * max(0.0, Q.sum_pt - 975.0)
            - 437.0 * max(0.0, 0.0221 - Q.z_6)
            + 5830.0 * max(0.0, Q.centroid_offset - 0.0213) * max(0.0, 0.01 - Q.mean_phi2)
            + 65.4 * max(0.0, Q.centroid_offset - 0.0103) * max(0.0, Q.psi_0p1 - 0.335)
            + 1.84 * max(0.0, Q.centroid_offset - 0.017) * max(0.0, Q.pt_4 - 66.3)
            - 1140.0 * max(0.0, 0.0118 - Q.lam1) * max(0.0, 0.254 - Q.planar_flow)
            + 57.4 * max(0.0, 0.0032 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05)
            - 0.707 * max(0.0, 64.5 - Q.mass) * max(0.0, 0.0534 - Q.z_dr_0p2_0p4)
            - 1210.0 * max(0.0, Q.mean_eta - 0.0349) * max(0.0, Q.M3 - 0.0492)
            + 0.589 * max(0.0, 42.5 - Q.pt_6) * max(0.0, 6.76 - Q.log_sum_pt)
            - 6.53 * max(0.0, 41.5 - Q.pt_6) * max(0.0, Q.z_7 - 0.0235)
            + 176.0 * max(0.0, 0.171 - Q.sj2_dr) * max(0.0, Q.eccentricity - 0.927)
            - 3.68 * max(0.0, Q.sum_pt - 953.0) * max(0.0, 0.00311 - Q.mean_phi2)
            + 0.00255 * max(0.0, 589.0 - Q.sum_pt) * max(0.0, 1.97 - Q.n_dr_0p2_0p4)
            + 0.00863 * max(0.0, 625.0 - Q.sum_pt) * max(0.0, 24.6 - Q.pt_5)
            + 1560.0 * max(0.0, 0.111 - Q.tau1) * max(0.0, 0.0157 - Q.mean_phi2)
        ))
        + 0.0625 * grid(8, max(0.0, -0.632
            - 218.0 * max(0.0, 0.0595 - Q.girth)
            + 1110.0 * max(0.0, 0.00491 - Q.girth2)
            - 5.72 * max(0.0, Q.log_sum_pt - 6.7)
            + 822.0 * max(0.0, 0.00355 - Q.width)
            + 42000.0 * max(0.0, 0.0585 - Q.girth) * max(0.0, Q.width - 0.000656)
            - 24400.0 * max(0.0, 0.00535 - Q.girth2) * max(0.0, Q.centroid_offset - 0.00661)
            + 21200.0 * max(0.0, 0.00662 - Q.girth2) * max(0.0, 0.0259 - Q.centroid_offset)
            + 1120.0 * max(0.0, 0.000136 - Q.lam2) * max(0.0, 5.27 - Q.D2_b2)
            - 6.02 * max(0.0, 31.3 - Q.mass) * max(0.0, 0.0291 - Q.centroid_offset)
            - 720.0 * max(0.0, 0.142 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.0059)
            - 0.00955 * max(0.0, Q.sum_pt_top5 - 679.0) * max(0.0, 8.0 - Q.n_pt_above_10)
            + 30000.0 * max(0.0, 0.0617 - Q.tau1) * max(0.0, 0.00304 - Q.width)
        ))
        + 0.2539062 * grid(9, max(0.0, -5.15
            - 0.17 * max(0.0, Q.D2 - 2.16)
            + 12.7 * max(0.0, 0.0796 - Q.M3)
            + 182.0 * max(0.0, 0.0185 - Q.centroid_offset)
            + 43.6 * max(0.0, Q.e2 - 0.0456)
            - 1790.0 * max(0.0, 0.00327 - Q.e2_sq)
            + 279.0 * max(0.0, Q.girth2 - 0.0134)
            + 2880.0 * max(0.0, 0.00365 - Q.girth2)
            + 865.0 * max(0.0, 0.00601 - Q.girth2)
            + 5410.0 * max(0.0, 0.000315 - Q.lam2)
            - 0.0971 * max(0.0, 30.1 - Q.mass)
            - 0.062 * max(0.0, 52.0 - Q.mass)
            + 18.6 * max(0.0, 0.107 - Q.max_dr)
            - 55.0 * max(0.0, 0.141 - Q.sj3_dr_max)
            + 21.9 * max(0.0, 0.215 - Q.sj3_dr_max)
            + 0.0077 * max(0.0, 986.0 - Q.sum_pt)
            + 7810.0 * max(0.0, 0.000176 - Q.width)
            - 161.0 * max(0.0, 0.00631 - Q.zdr_0)
            - 33.6 * max(0.0, 0.017 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0)
            - 2.01 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, 164.0 - Q.pt_1)
            + 1090.0 * max(0.0, 0.0191 - Q.centroid_offset) * max(0.0, 0.205 - Q.z_2nd)
            + 3990.0 * max(0.0, 0.0528 - Q.girth) * max(0.0, 0.0275 - Q.tau21_b2)
            - 1380.0 * max(0.0, 0.0549 - Q.girth) * max(0.0, Q.z_6 - 0.0338)
            - 9500.0 * max(0.0, 0.00578 - Q.girth2) * max(0.0, 0.00437 - Q.mean_phi)
            - 859.0 * max(0.0, Q.lam1 - 0.00485) * max(0.0, Q.eccentricity - 0.638)
            - 27.2 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.746 - Q.D2_b2)
            + 4.38 * max(0.0, 54.9 - Q.mass) * max(0.0, 0.0267 - Q.centroid_offset)
            + 0.129 * max(0.0, 55.0 - Q.mass) * max(0.0, 6.84 - Q.log_sum_pt)
            + 0.581 * max(0.0, 0.15 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 32.9)
            + 3360.0 * max(0.0, 0.046 - Q.tau1) * max(0.0, -0.0119 - Q.mean_phi)
            - 4610.0 * max(0.0, 0.0457 - Q.tau1) * max(0.0, 0.0286 - Q.tau21_b2)
        ))
        - 0.125 * grid(10, max(0.0, 1.59
            + 36.9 * max(0.0, Q.C2_b2 - 0.00763)
            - 29.3 * max(0.0, Q.LHA - 0.305)
            - 2620.0 * max(0.0, Q.e3 - 5.22e-05)
            + 274.0 * max(0.0, Q.girth2 - 0.00749)
            - 90.8 * max(0.0, Q.girth2 - 0.0258)
            + 379.0 * max(0.0, 0.00215 - Q.girth2_top3)
            - 1040.0 * max(0.0, 0.00144 - Q.lam1)
            - 328.0 * max(0.0, 0.00404 - Q.lam1)
            - 361.0 * max(0.0, 0.00598 - Q.lam1)
            + 1290.0 * max(0.0, Q.lam2 - 0.000278)
            + 0.0276 * max(0.0, 79.4 - Q.mass)
            - 0.0122 * max(0.0, Q.mass_top5 - 51.0)
            - 0.0151 * max(0.0, 43.9 - Q.pt_7)
            - 4.57 * max(0.0, Q.sj3_dr_max - 0.184)
            + 0.123 * max(0.0, Q.sj3_pair_mass_min - 15.8)
            - 0.00569 * max(0.0, Q.sum_pt - 988.0)
            + 16.3 * max(0.0, Q.tau1 - 0.057)
            - 241.0 * max(0.0, Q.lam1 - 0.00707) * max(0.0, 0.36 - Q.D2_b2)
            - 702.0 * max(0.0, Q.lam2 - 0.000257) * max(0.0, Q.planar_flow - 0.055)
            + 0.0613 * max(0.0, 45.6 - Q.pt_7) * max(0.0, 0.985 - Q.D2)
            - 0.0904 * max(0.0, 48.0 - Q.pt_7) * max(0.0, 6.56 - Q.log_sum_pt)
            - 0.191 * max(0.0, Q.sj3_pair_mass_min - 6.68) * max(0.0, Q.sj3_pairmin_over_m - 0.245)
            + 1610.0 * max(0.0, 0.0205 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.0125)
            + 115.0 * max(0.0, 0.0242 - Q.zdr_0) * max(0.0, 0.248 - Q.z_dr_0p05_0p1)
        ))
        + 0.0625 * grid(15, max(0.0, 0.00133
            + 156.0 * max(0.0, 0.0409 - Q.e2)
            - 1130.0 * max(0.0, 0.00818 - Q.e2_sq)
            - 91.6 * max(0.0, 0.102 - Q.girth)
            - 546.0 * max(0.0, 0.00739 - Q.girth2)
            - 10700.0 * max(0.0, 0.000513 - Q.girth2_top2)
            + 655.0 * max(0.0, 0.00708 - Q.mass_over_sum_pt_sq)
            + 44.0 * max(0.0, 0.159 - Q.sj2_dr)
            - 35.7 * max(0.0, 0.199 - Q.sj2_dr)
            - 0.049 * max(0.0, Q.sj3_pair_mass_max - 81.1)
            + 710.0 * max(0.0, 0.0133 - Q.width)
            + 1.25 * max(0.0, 0.229 - Q.N2) * max(0.0, 4.0 - Q.n_dr_0p1_0p2)
            + 0.0181 * max(0.0, 0.218 - Q.N2) * max(0.0, Q.sum_pt_top5 - 498.0)
            - 13.9 * max(0.0, 0.223 - Q.N2) * max(0.0, 0.548 - Q.z_dr_0p05_0p1)
            - 1140.0 * max(0.0, 0.00752 - Q.girth2) * max(0.0, 0.731 - Q.D2)
            + 31.3 * max(0.0, 0.133 - Q.mass_over_sum_pt) * max(0.0, 0.711 - Q.D2)
        ))
    )


def logit_W(Q):
    return (-0.125
        + 0.34375 * grid(0, max(0.0, -0.464
            - 930.0 * max(0.0, 0.00151 - Q.C2_b2)
            - 50.7 * max(0.0, Q.centroid_offset - 0.00783)
            - 347.0 * max(0.0, Q.centroid_offset - 0.0499)
            + 71.7 * max(0.0, 0.0243 - Q.e2)
            + 81.8 * max(0.0, 0.0354 - Q.e2)
            - 47.9 * max(0.0, 0.0751 - Q.girth)
            + 497.0 * max(0.0, 0.013 - Q.girth2)
            - 0.103 * max(0.0, 29.6 - Q.mass)
            - 0.0679 * max(0.0, 64.7 - Q.mass)
            + 6.63 * max(0.0, 0.159 - Q.planar_flow)
            + 11.9 * max(0.0, Q.sj3_dr_max - 0.11)
            - 27.2 * max(0.0, Q.sj3_dr_max - 0.182)
            - 0.0134 * max(0.0, Q.sum_pt - 906.0)
            - 1120.0 * max(0.0, 0.00444 - Q.width)
            - 1540.0 * max(0.0, Q.centroid_offset - 0.0203) * max(0.0, 0.0609 - Q.z_7)
            + 172.0 * max(0.0, 0.0133 - Q.girth2) * max(0.0, 1.02 - Q.D2)
            + 818.0 * max(0.0, 0.0134 - Q.girth2) * max(0.0, Q.phi_0 - -0.00668)
            - 1580.0 * max(0.0, 0.00649 - Q.lam1) * max(0.0, 0.843 - Q.D2)
            + 55000.0 * max(0.0, 8.01e-05 - Q.lam2) * max(0.0, 0.257 - Q.D2_b2)
            + 49.3 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, 0.0746 - Q.dr_4)
            - 458.0 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, Q.z_7 - 0.0178)
            - 661.0 * max(0.0, 0.142 - Q.planar_flow) * max(0.0, 0.023 - Q.dr_2)
            + 510.0 * max(0.0, 0.165 - Q.planar_flow) * max(0.0, 0.0237 - Q.z_7)
            - 190.0 * max(0.0, 0.245 - Q.sj3_dr_max) * max(0.0, 0.058 - Q.z_7)
            + 0.000467 * max(0.0, Q.sum_pt - 915.0) * max(0.0, Q.pt_7 - 35.4)
            + 8830.0 * max(0.0, 0.00414 - Q.width) * max(0.0, 0.0364 - Q.C3)
        ))
        - 0.03125 * grid(1, max(0.0, 1.28
            - 29.9 * max(0.0, 0.0991 - Q.girth)
            - 196.0 * max(0.0, 0.00871 - Q.girth2)
            - 736.0 * max(0.0, 0.00085 - Q.lam1)
            + 10.3 * max(0.0, Q.log_sum_pt - 6.5)
            + 5.23 * max(0.0, Q.log_sum_pt - 6.62)
            + 323.0 * max(0.0, 0.00591 - Q.mass_over_sum_pt_sq)
            - 0.113 * max(0.0, Q.pt_7 - 53.5)
            + 0.0053 * max(0.0, Q.sum_pt_top5 - 364.0)
            - 0.0129 * max(0.0, Q.sum_pt_top5 - 528.0)
            + 21.6 * max(0.0, Q.z_7 - 0.0586)
            - 35.5 * max(0.0, 0.0619 - Q.z_7)
            + 6070.0 * max(0.0, 0.00939 - Q.girth2) * max(0.0, Q.centroid_offset - 0.0216)
            - 46.4 * max(0.0, 0.00871 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.25)
            - 596.0 * max(0.0, 0.00962 - Q.lam1) * max(0.0, 0.15 - Q.planar_flow)
            + 3.11 * max(0.0, Q.log_sum_pt - 6.48) * max(0.0, 1.4 - Q.D2)
            + 82.0 * max(0.0, Q.log_sum_pt - 6.37) * max(0.0, Q.centroid_offset - 0.0126)
            - 0.00445 * max(0.0, Q.pt_7 - 33.4) * max(0.0, 53.4 - Q.pt_6)
            + 0.477 * max(0.0, Q.pt_7 - 33.5) * max(0.0, Q.sj2_dr - 0.143)
            - 0.0846 * max(0.0, Q.sj3_dr_max - 0.104) * max(0.0, Q.sj3_pair_mass_min - 2.6)
            - 36.3 * max(0.0, 0.0607 - Q.z_7) * max(0.0, 1.52 - Q.D2)
        ))
        - 0.5 * grid(3, max(0.0, -4.48
            + 28.4 * max(0.0, Q.centroid_offset - 0.00928)
            + 98.5 * max(0.0, Q.e2 - 0.046)
            + 171.0 * max(0.0, Q.girth - 0.0409)
            - 142.0 * max(0.0, Q.girth - 0.103)
            - 1380.0 * max(0.0, Q.girth2 - 0.00879)
            + 478.0 * max(0.0, Q.girth2 - 0.0187)
            + 566.0 * max(0.0, Q.lam1 - 0.00828)
            - 0.0994 * max(0.0, Q.mass - 65.1)
            - 288.0 * max(0.0, 0.0118 - Q.mass_over_sum_pt_sq)
            + 0.643 * max(0.0, 1.04 - Q.n_dr_0_0p05)
            + 0.0673 * max(0.0, Q.sd_mass - 63.8)
            - 0.0809 * max(0.0, Q.sd_mass - 86.0)
            + 47.1 * max(0.0, Q.sj2_dr - 0.188)
            - 58.7 * max(0.0, Q.sj2_dr - 0.269)
            + 36.7 * max(0.0, Q.sj3_dr_max - 0.196)
            - 19.5 * max(0.0, Q.sj3_dr_max - 0.297)
            - 65.8 * max(0.0, Q.tau1 - 0.0469)
            + 103.0 * max(0.0, Q.tau1 - 0.114)
            - 0.898 * max(0.0, 0.836 - Q.z_dr_0_0p05)
            + 63.1 * max(0.0, Q.girth - 0.0307) * max(0.0, Q.log_sum_pt - 6.11)
            + 16.7 * max(0.0, Q.lam2 - 0.000776) * max(0.0, 64.0 - Q.pt_6)
            - 203.0 * max(0.0, Q.sj2_dr - 0.214) * max(0.0, 0.0549 - Q.dr_3)
            - 0.302 * max(0.0, Q.sj2_dr - 0.206) * max(0.0, Q.sj2_mass1 - -0.0277)
            - 33.0 * max(0.0, Q.sj2_dr - 0.199) * max(0.0, 1.02 - Q.z_dr_0p05_0p1)
            + 60.9 * max(0.0, Q.sj2_dr - 0.268) * max(0.0, 1.03 - Q.z_dr_0p05_0p1)
        ))
        - 0.3125 * grid(6, max(0.0, 2.65
            + 106.0 * max(0.0, Q.centroid_offset - 0.00762)
            - 102.0 * max(0.0, Q.centroid_offset - 0.0193)
            + 547.0 * max(0.0, 0.00301 - Q.e2_sq)
            - 1020.0 * max(0.0, 0.00867 - Q.girth2)
            + 298.0 * max(0.0, 0.0118 - Q.lam1)
            - 1090.0 * max(0.0, 0.00109 - Q.lam2)
            + 0.0803 * max(0.0, 22.5 - Q.mass)
            - 34.4 * max(0.0, 0.144 - Q.max_dr)
            + 0.513 * max(0.0, 19.1 - Q.pt_6)
            + 8.14 * max(0.0, Q.sj3_dr_max - 0.191)
            + 30.4 * max(0.0, 0.181 - Q.sj3_dr_max)
            - 9.6 * max(0.0, 0.29 - Q.sj3_dr_max)
            - 17.8 * max(0.0, Q.sj3_dr_min - 0.0225)
            - 0.0953 * max(0.0, Q.sj3_pair_mass_min - 4.35)
            + 0.0162 * max(0.0, Q.sum_pt - 975.0)
            - 437.0 * max(0.0, 0.0221 - Q.z_6)
            + 5830.0 * max(0.0, Q.centroid_offset - 0.0213) * max(0.0, 0.01 - Q.mean_phi2)
            + 65.4 * max(0.0, Q.centroid_offset - 0.0103) * max(0.0, Q.psi_0p1 - 0.335)
            + 1.84 * max(0.0, Q.centroid_offset - 0.017) * max(0.0, Q.pt_4 - 66.3)
            - 1140.0 * max(0.0, 0.0118 - Q.lam1) * max(0.0, 0.254 - Q.planar_flow)
            + 57.4 * max(0.0, 0.0032 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05)
            - 0.707 * max(0.0, 64.5 - Q.mass) * max(0.0, 0.0534 - Q.z_dr_0p2_0p4)
            - 1210.0 * max(0.0, Q.mean_eta - 0.0349) * max(0.0, Q.M3 - 0.0492)
            + 0.589 * max(0.0, 42.5 - Q.pt_6) * max(0.0, 6.76 - Q.log_sum_pt)
            - 6.53 * max(0.0, 41.5 - Q.pt_6) * max(0.0, Q.z_7 - 0.0235)
            + 176.0 * max(0.0, 0.171 - Q.sj2_dr) * max(0.0, Q.eccentricity - 0.927)
            - 3.68 * max(0.0, Q.sum_pt - 953.0) * max(0.0, 0.00311 - Q.mean_phi2)
            + 0.00255 * max(0.0, 589.0 - Q.sum_pt) * max(0.0, 1.97 - Q.n_dr_0p2_0p4)
            + 0.00863 * max(0.0, 625.0 - Q.sum_pt) * max(0.0, 24.6 - Q.pt_5)
            + 1560.0 * max(0.0, 0.111 - Q.tau1) * max(0.0, 0.0157 - Q.mean_phi2)
        ))
        + 0.21875 * grid(7, max(0.0, 11.6
            - 10.8 * max(0.0, 0.0586 - Q.D2_b2)
            - 118.0 * max(0.0, Q.centroid_offset - 0.0384)
            - 61.0 * max(0.0, 0.0884 - Q.girth)
            - 238.0 * max(0.0, Q.girth2 - 0.00439)
            + 617.0 * max(0.0, Q.girth2 - 0.0134)
            - 1260.0 * max(0.0, 0.00104 - Q.girth2)
            + 603.0 * max(0.0, 0.00109 - Q.girth2_top2)
            - 1780.0 * max(0.0, 0.000271 - Q.lam2)
            + 0.0576 * max(0.0, Q.mass - 36.6)
            - 0.18 * max(0.0, Q.mass - 76.8)
            - 124.0 * max(0.0, Q.mass_over_sum_pt - 0.0108)
            + 158.0 * max(0.0, Q.mass_over_sum_pt - 0.0803)
            - 273.0 * max(0.0, Q.mass_over_sum_pt - 0.0909)
            - 494.0 * max(0.0, Q.mass_over_sum_pt - 0.108)
            + 0.0721 * max(0.0, Q.mass_top5 - 53.0)
            - 18.2 * max(0.0, 0.109 - Q.max_dr)
            + 9.5 * max(0.0, 0.178 - Q.max_dr)
            - 0.0511 * max(0.0, 29.4 - Q.pt_7)
            + 33.0 * max(0.0, Q.sd_rg - 0.283)
            + 28.1 * max(0.0, 0.156 - Q.sj2_dr)
            - 21.4 * max(0.0, 0.187 - Q.sj2_dr)
            + 23.0 * max(0.0, Q.sj3_dr_max - 0.139)
            - 19.4 * max(0.0, Q.sj3_dr_max - 0.261)
            - 24.2 * max(0.0, 0.0407 - Q.tau21_b2)
            - 1120.0 * max(0.0, 0.00664 - Q.width)
            + 15.2 * max(0.0, Q.z_7 - 0.0342)
            + 2.49 * max(0.0, 0.139 - Q.z_dr_0p1_0p2)
            - 17200.0 * max(0.0, 0.02 - Q.centroid_offset) * max(0.0, 0.00426 - Q.C2_b2)
            - 10.0 * max(0.0, Q.centroid_offset - 0.0258) * max(0.0, Q.n_pt_above_50 - 3.71)
            + 2880.0 * max(0.0, 0.0216 - Q.centroid_offset) * max(0.0, 0.0279 - Q.tau21_b2)
            + 4120.0 * max(0.0, 0.0893 - Q.girth) * max(0.0, 0.00429 - Q.C2_b2)
            - 2520.0 * max(0.0, Q.girth2 - 0.0145) * max(0.0, Q.eccentricity - 0.925)
            + 1040.0 * max(0.0, Q.girth2 - 0.00447) * max(0.0, 0.197 - Q.planar_flow)
            + 127.0 * max(0.0, Q.girth2 - 0.0131) * max(0.0, 41.3 - Q.pt_6)
            - 90500.0 * max(0.0, 0.00107 - Q.girth2_top2) * max(0.0, 0.0257 - Q.tau21_b2)
            + 125000.0 * max(0.0, 0.0003 - Q.lam2) * max(0.0, 0.0408 - Q.tau21_b2)
            - 9.31 * max(0.0, Q.mass - 76.2) * max(0.0, Q.zdr_6 - 0.00612)
            + 17.2 * max(0.0, Q.mass_over_sum_pt - 0.0911) * max(0.0, 40.8 - Q.pt_6)
            - 0.224 * max(0.0, 0.216 - Q.planar_flow) * max(0.0, 35.6 - Q.pt_6)
            - 0.947 * max(0.0, Q.sj3_dr_max - 0.132) * max(0.0, 20.0 - Q.pt_6)
            - 111.0 * max(0.0, Q.width - 0.00864) * max(0.0, 40.8 - Q.pt_6)
        ))
        - 0.25 * grid(8, max(0.0, -0.632
            - 218.0 * max(0.0, 0.0595 - Q.girth)
            + 1110.0 * max(0.0, 0.00491 - Q.girth2)
            - 5.72 * max(0.0, Q.log_sum_pt - 6.7)
            + 822.0 * max(0.0, 0.00355 - Q.width)
            + 42000.0 * max(0.0, 0.0585 - Q.girth) * max(0.0, Q.width - 0.000656)
            - 24400.0 * max(0.0, 0.00535 - Q.girth2) * max(0.0, Q.centroid_offset - 0.00661)
            + 21200.0 * max(0.0, 0.00662 - Q.girth2) * max(0.0, 0.0259 - Q.centroid_offset)
            + 1120.0 * max(0.0, 0.000136 - Q.lam2) * max(0.0, 5.27 - Q.D2_b2)
            - 6.02 * max(0.0, 31.3 - Q.mass) * max(0.0, 0.0291 - Q.centroid_offset)
            - 720.0 * max(0.0, 0.142 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.0059)
            - 0.00955 * max(0.0, Q.sum_pt_top5 - 679.0) * max(0.0, 8.0 - Q.n_pt_above_10)
            + 30000.0 * max(0.0, 0.0617 - Q.tau1) * max(0.0, 0.00304 - Q.width)
        ))
        - 0.03125 * grid(9, max(0.0, -5.15
            - 0.17 * max(0.0, Q.D2 - 2.16)
            + 12.7 * max(0.0, 0.0796 - Q.M3)
            + 182.0 * max(0.0, 0.0185 - Q.centroid_offset)
            + 43.6 * max(0.0, Q.e2 - 0.0456)
            - 1790.0 * max(0.0, 0.00327 - Q.e2_sq)
            + 279.0 * max(0.0, Q.girth2 - 0.0134)
            + 2880.0 * max(0.0, 0.00365 - Q.girth2)
            + 865.0 * max(0.0, 0.00601 - Q.girth2)
            + 5410.0 * max(0.0, 0.000315 - Q.lam2)
            - 0.0971 * max(0.0, 30.1 - Q.mass)
            - 0.062 * max(0.0, 52.0 - Q.mass)
            + 18.6 * max(0.0, 0.107 - Q.max_dr)
            - 55.0 * max(0.0, 0.141 - Q.sj3_dr_max)
            + 21.9 * max(0.0, 0.215 - Q.sj3_dr_max)
            + 0.0077 * max(0.0, 986.0 - Q.sum_pt)
            + 7810.0 * max(0.0, 0.000176 - Q.width)
            - 161.0 * max(0.0, 0.00631 - Q.zdr_0)
            - 33.6 * max(0.0, 0.017 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0)
            - 2.01 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, 164.0 - Q.pt_1)
            + 1090.0 * max(0.0, 0.0191 - Q.centroid_offset) * max(0.0, 0.205 - Q.z_2nd)
            + 3990.0 * max(0.0, 0.0528 - Q.girth) * max(0.0, 0.0275 - Q.tau21_b2)
            - 1380.0 * max(0.0, 0.0549 - Q.girth) * max(0.0, Q.z_6 - 0.0338)
            - 9500.0 * max(0.0, 0.00578 - Q.girth2) * max(0.0, 0.00437 - Q.mean_phi)
            - 859.0 * max(0.0, Q.lam1 - 0.00485) * max(0.0, Q.eccentricity - 0.638)
            - 27.2 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.746 - Q.D2_b2)
            + 4.38 * max(0.0, 54.9 - Q.mass) * max(0.0, 0.0267 - Q.centroid_offset)
            + 0.129 * max(0.0, 55.0 - Q.mass) * max(0.0, 6.84 - Q.log_sum_pt)
            + 0.581 * max(0.0, 0.15 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 32.9)
            + 3360.0 * max(0.0, 0.046 - Q.tau1) * max(0.0, -0.0119 - Q.mean_phi)
            - 4610.0 * max(0.0, 0.0457 - Q.tau1) * max(0.0, 0.0286 - Q.tau21_b2)
        ))
        + 0.375 * grid(11, max(0.0, -1.44
            - 624.0 * max(0.0, Q.centroid_offset - 0.05)
            + 65.3 * max(0.0, 0.0377 - Q.centroid_offset)
            - 1160.0 * max(0.0, 0.00814 - Q.e2_sq)
            - 75.7 * max(0.0, 0.0231 - Q.girth)
            - 67.1 * max(0.0, 0.0737 - Q.girth)
            + 137.0 * max(0.0, 0.0134 - Q.girth2)
            - 708.0 * max(0.0, 0.00838 - Q.lam1)
            + 0.125 * max(0.0, 15.7 - Q.mass)
            - 0.04 * max(0.0, 70.5 - Q.mass)
            - 0.351 * max(0.0, Q.n_dr_0p1_0p2 - 2.92)
            + 5.84 * max(0.0, 0.242 - Q.planar_flow)
            - 0.0691 * max(0.0, 25.4 - Q.pt_6)
            - 0.036 * max(0.0, Q.pt_7 - 29.6)
            - 27.7 * max(0.0, 0.167 - Q.sj3_dr_max)
            + 15.1 * max(0.0, 0.257 - Q.sj3_dr_max)
            - 0.00349 * max(0.0, Q.sum_pt - 971.0)
            + 2510.0 * max(0.0, 0.00872 - Q.width)
            + 35.8 * max(0.0, Q.z_7 - 0.0169)
            + 2.49 * max(0.0, Q.z_dr_0p05_0p1 - 0.826)
            + 213.0 * max(0.0, 0.0138 - Q.centroid_offset) * max(0.0, 0.571 - Q.D2_b2)
            - 328.0 * max(0.0, 0.0384 - Q.centroid_offset) * max(0.0, 0.0324 - Q.absphi_0)
            - 1070.0 * max(0.0, 0.0383 - Q.centroid_offset) * max(0.0, -0.00159 - Q.mean_phi)
            - 8.45 * max(0.0, 0.0137 - Q.centroid_offset) * max(0.0, 17.2 - Q.sj3_pair_mass_min)
            - 0.193 * max(0.0, 0.0377 - Q.centroid_offset) * max(0.0, 892.0 - Q.sum_pt)
            - 86100.0 * max(0.0, Q.centroid_offset - 0.05) * max(0.0, Q.tau3 - 0.000494)
            + 1720.0 * max(0.0, Q.centroid_offset - 0.0494) * max(0.0, 0.619 - Q.tau32)
            - 57100.0 * max(0.0, Q.centroid_offset - 0.0492) * max(0.0, Q.zdr_6 - 0.00714)
            + 4770.0 * max(0.0, 0.00431 - Q.girth2) * max(0.0, Q.sj3_dr13 - 0.162)
            - 446000.0 * max(0.0, 0.233 - Q.planar_flow) * max(0.0, 9.53e-06 - Q.e3)
            - 0.183 * max(0.0, 0.268 - Q.planar_flow) * max(0.0, 36.7 - Q.pt_7)
            - 0.0134 * max(0.0, 0.241 - Q.planar_flow) * max(0.0, 835.0 - Q.sum_pt)
            + 91.8 * max(0.0, 0.264 - Q.sj3_dr_max) * max(0.0, -0.023 - Q.phi_0)
        ))
        + 0.0703125 * grid(13, max(0.0, 0.714
            - 55.3 * max(0.0, 0.079 - Q.e2)
            + 19700.0 * max(0.0, 5.05e-05 - Q.e3)
            + 58.6 * max(0.0, 0.148 - Q.girth)
            + 253.0 * max(0.0, 0.0161 - Q.lam1)
            - 0.0244 * max(0.0, Q.mass - 49.1)
            - 0.0571 * max(0.0, Q.pt_7 - 48.4)
            - 0.00746 * max(0.0, Q.sum_pt_top5 - 674.0)
            + 0.0137 * max(0.0, Q.sum_pt_top5 - 794.0)
            - 167.0 * max(0.0, 0.0283 - Q.z_5)
            - 189.0 * max(0.0, 0.028 - Q.z_7)
            + 2100.0 * max(0.0, 6.94e-05 - Q.e3) * max(0.0, Q.D3 - 0.0598)
            - 1180000.0 * max(0.0, 5.38e-05 - Q.e3) * max(0.0, 0.0377 - Q.centroid_offset)
            + 125.0 * max(0.0, 0.15 - Q.girth) * max(0.0, 0.0746 - Q.M3)
            - 8780.0 * max(0.0, 0.148 - Q.girth) * max(0.0, 0.000536 - Q.lam2)
            - 75.4 * max(0.0, 0.148 - Q.girth) * max(0.0, 6.81 - Q.log_sum_pt)
            - 0.987 * max(0.0, 0.145 - Q.girth) * max(0.0, 38.2 - Q.pt_7)
            - 1.58 * max(0.0, 0.152 - Q.girth) * max(0.0, Q.sj2_mass1 - 30.4)
            + 281.0 * max(0.0, 0.142 - Q.girth) * max(0.0, Q.tau2 - 0.00886)
            + 392.0 * max(0.0, 0.153 - Q.girth) * max(0.0, Q.z_7 - 0.0616)
            - 2.22 * max(0.0, 31.9 - Q.pt_6) * max(0.0, 0.0624 - Q.z_7)
            - 0.000466 * max(0.0, Q.sum_pt - 1000.0) * max(0.0, 70.0 - Q.pt_6)
            + 0.00108 * max(0.0, Q.sum_pt_top5 - 667.0) * max(0.0, 38.1 - Q.pt_7)
        ))
        - 0.75 * grid(14, max(0.0, -0.689
            - 146.0 * max(0.0, 0.00403 - Q.C2_b2)
            - 84.3 * max(0.0, Q.centroid_offset - 0.0269)
            - 979.0 * max(0.0, Q.centroid_offset - 0.0497)
            - 147.0 * max(0.0, Q.e2 - 0.0167)
            + 258.0 * max(0.0, Q.e2 - 0.0499)
            + 1130.0 * max(0.0, Q.e2_sq - 0.0117)
            - 23600.0 * max(0.0, 7.93e-05 - Q.e3)
            + 84.3 * max(0.0, Q.girth - 0.0272)
            + 70.1 * max(0.0, Q.girth - 0.041)
            + 93.5 * max(0.0, Q.girth - 0.0802)
            - 193.0 * max(0.0, Q.girth - 0.0868)
            - 78.8 * max(0.0, Q.girth - 0.102)
            - 1570.0 * max(0.0, Q.girth2 - 0.00856)
            + 1010.0 * max(0.0, 0.000504 - Q.lam2)
            - 0.0427 * max(0.0, Q.mass - 68.9)
            + 157.0 * max(0.0, Q.mass_over_sum_pt - 0.0804)
            + 173.0 * max(0.0, Q.mass_over_sum_pt - 0.0906)
            + 0.192 * max(0.0, 5.06 - Q.n_dr_0_0p05)
            - 22.5 * max(0.0, Q.psi_0p1 - 0.974)
            + 0.0682 * max(0.0, 49.6 - Q.sd_mass)
            - 0.0178 * max(0.0, 74.8 - Q.sd_mass)
            - 32.8 * max(0.0, Q.sd_rg - 0.234)
            - 9.92 * max(0.0, 0.325 - Q.sd_rg)
            - 6.65 * max(0.0, Q.sj2_dr - 0.0606)
            + 31.8 * max(0.0, Q.sj2_dr - 0.158)
            - 18.9 * max(0.0, Q.sj2_dr - 0.197)
            - 0.00585 * max(0.0, 717.0 - Q.sum_pt)
            + 339.0 * max(0.0, Q.width - 0.00353)
            - 1480.0 * max(0.0, Q.width - 0.0067)
            - 1260.0 * max(0.0, Q.width - 0.0131)
            + 1.5 * max(0.0, Q.z_dr_0_0p05 - 0.152)
            - 2.29 * max(0.0, 0.163 - Q.z_dr_0p05_0p1)
            + 4.72 * max(0.0, 0.0677 - Q.z_dr_0p1_0p2)
            + 1160000.0 * max(0.0, Q.centroid_offset - 0.0499) * max(0.0, 0.000872 - Q.C2_b2)
            + 0.547 * max(0.0, Q.centroid_offset - 0.0267) * max(0.0, 152.0 - Q.pt_1)
            + 218000.0 * max(0.0, 5.56e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.162)
            - 400.0 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 0.0179 - Q.centroid_offset)
            - 4200.0 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.00746 - Q.lam1)
            + 987.0 * max(0.0, 0.115 - Q.planar_flow) * max(0.0, 0.0165 - Q.lam1)
            - 98.8 * max(0.0, 0.109 - Q.planar_flow) * max(0.0, Q.mean_eta - -0.00684)
            - 0.0177 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 652.0 - Q.sum_pt_top5)
            + 3620.0 * max(0.0, 0.112 - Q.planar_flow) * max(0.0, 0.00601 - Q.width)
            - 168.0 * max(0.0, Q.sj2_dr - 0.13) * max(0.0, 0.0886 - Q.D2_b2)
            + 378.0 * max(0.0, Q.sj2_dr - 0.159) * max(0.0, 0.0881 - Q.D2_b2)
            - 264.0 * max(0.0, Q.sj2_dr - 0.198) * max(0.0, 0.0872 - Q.D2_b2)
            - 569.0 * max(0.0, Q.sj2_dr - 0.164) * max(0.0, 0.0332 - Q.dr_2)
            + 289.0 * max(0.0, Q.sj2_dr - 0.197) * max(0.0, 0.0517 - Q.dr_2)
            - 74.7 * max(0.0, Q.sj2_dr - 0.203) * max(0.0, 0.105 - Q.z_5)
            + 0.00254 * max(0.0, 0.627 - Q.z_dr_0p05_0p1) * max(0.0, 714.0 - Q.sum_pt)
            - 3190.0 * max(0.0, Q.z_dr_0p05_0p1 - 0.745) * max(0.0, Q.zdr_5 - 0.0154)
        ))
        - 0.6875 * grid(15, max(0.0, 0.00133
            + 156.0 * max(0.0, 0.0409 - Q.e2)
            - 1130.0 * max(0.0, 0.00818 - Q.e2_sq)
            - 91.6 * max(0.0, 0.102 - Q.girth)
            - 546.0 * max(0.0, 0.00739 - Q.girth2)
            - 10700.0 * max(0.0, 0.000513 - Q.girth2_top2)
            + 655.0 * max(0.0, 0.00708 - Q.mass_over_sum_pt_sq)
            + 44.0 * max(0.0, 0.159 - Q.sj2_dr)
            - 35.7 * max(0.0, 0.199 - Q.sj2_dr)
            - 0.049 * max(0.0, Q.sj3_pair_mass_max - 81.1)
            + 710.0 * max(0.0, 0.0133 - Q.width)
            + 1.25 * max(0.0, 0.229 - Q.N2) * max(0.0, 4.0 - Q.n_dr_0p1_0p2)
            + 0.0181 * max(0.0, 0.218 - Q.N2) * max(0.0, Q.sum_pt_top5 - 498.0)
            - 13.9 * max(0.0, 0.223 - Q.N2) * max(0.0, 0.548 - Q.z_dr_0p05_0p1)
            - 1140.0 * max(0.0, 0.00752 - Q.girth2) * max(0.0, 0.731 - Q.D2)
            + 31.3 * max(0.0, 0.133 - Q.mass_over_sum_pt) * max(0.0, 0.711 - Q.D2)
        ))
    )


def logit_Z(Q):
    return (-0.09375
        + 0.125 * grid(1, max(0.0, 1.28
            - 29.9 * max(0.0, 0.0991 - Q.girth)
            - 196.0 * max(0.0, 0.00871 - Q.girth2)
            - 736.0 * max(0.0, 0.00085 - Q.lam1)
            + 10.3 * max(0.0, Q.log_sum_pt - 6.5)
            + 5.23 * max(0.0, Q.log_sum_pt - 6.62)
            + 323.0 * max(0.0, 0.00591 - Q.mass_over_sum_pt_sq)
            - 0.113 * max(0.0, Q.pt_7 - 53.5)
            + 0.0053 * max(0.0, Q.sum_pt_top5 - 364.0)
            - 0.0129 * max(0.0, Q.sum_pt_top5 - 528.0)
            + 21.6 * max(0.0, Q.z_7 - 0.0586)
            - 35.5 * max(0.0, 0.0619 - Q.z_7)
            + 6070.0 * max(0.0, 0.00939 - Q.girth2) * max(0.0, Q.centroid_offset - 0.0216)
            - 46.4 * max(0.0, 0.00871 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.25)
            - 596.0 * max(0.0, 0.00962 - Q.lam1) * max(0.0, 0.15 - Q.planar_flow)
            + 3.11 * max(0.0, Q.log_sum_pt - 6.48) * max(0.0, 1.4 - Q.D2)
            + 82.0 * max(0.0, Q.log_sum_pt - 6.37) * max(0.0, Q.centroid_offset - 0.0126)
            - 0.00445 * max(0.0, Q.pt_7 - 33.4) * max(0.0, 53.4 - Q.pt_6)
            + 0.477 * max(0.0, Q.pt_7 - 33.5) * max(0.0, Q.sj2_dr - 0.143)
            - 0.0846 * max(0.0, Q.sj3_dr_max - 0.104) * max(0.0, Q.sj3_pair_mass_min - 2.6)
            - 36.3 * max(0.0, 0.0607 - Q.z_7) * max(0.0, 1.52 - Q.D2)
        ))
        - 0.5625 * grid(3, max(0.0, -4.48
            + 28.4 * max(0.0, Q.centroid_offset - 0.00928)
            + 98.5 * max(0.0, Q.e2 - 0.046)
            + 171.0 * max(0.0, Q.girth - 0.0409)
            - 142.0 * max(0.0, Q.girth - 0.103)
            - 1380.0 * max(0.0, Q.girth2 - 0.00879)
            + 478.0 * max(0.0, Q.girth2 - 0.0187)
            + 566.0 * max(0.0, Q.lam1 - 0.00828)
            - 0.0994 * max(0.0, Q.mass - 65.1)
            - 288.0 * max(0.0, 0.0118 - Q.mass_over_sum_pt_sq)
            + 0.643 * max(0.0, 1.04 - Q.n_dr_0_0p05)
            + 0.0673 * max(0.0, Q.sd_mass - 63.8)
            - 0.0809 * max(0.0, Q.sd_mass - 86.0)
            + 47.1 * max(0.0, Q.sj2_dr - 0.188)
            - 58.7 * max(0.0, Q.sj2_dr - 0.269)
            + 36.7 * max(0.0, Q.sj3_dr_max - 0.196)
            - 19.5 * max(0.0, Q.sj3_dr_max - 0.297)
            - 65.8 * max(0.0, Q.tau1 - 0.0469)
            + 103.0 * max(0.0, Q.tau1 - 0.114)
            - 0.898 * max(0.0, 0.836 - Q.z_dr_0_0p05)
            + 63.1 * max(0.0, Q.girth - 0.0307) * max(0.0, Q.log_sum_pt - 6.11)
            + 16.7 * max(0.0, Q.lam2 - 0.000776) * max(0.0, 64.0 - Q.pt_6)
            - 203.0 * max(0.0, Q.sj2_dr - 0.214) * max(0.0, 0.0549 - Q.dr_3)
            - 0.302 * max(0.0, Q.sj2_dr - 0.206) * max(0.0, Q.sj2_mass1 - -0.0277)
            - 33.0 * max(0.0, Q.sj2_dr - 0.199) * max(0.0, 1.02 - Q.z_dr_0p05_0p1)
            + 60.9 * max(0.0, Q.sj2_dr - 0.268) * max(0.0, 1.03 - Q.z_dr_0p05_0p1)
        ))
        + 0.078125 * grid(4, max(0.0, 0.0436
            + 76.8 * max(0.0, 0.0651 - Q.C2)
            + 13.2 * max(0.0, 0.351 - Q.LHA)
            + 53.9 * max(0.0, 0.224 - Q.N2)
            + 648.0 * max(0.0, 0.0116 - Q.e2_sq)
            - 969.0 * max(0.0, 0.00919 - Q.girth2)
            + 220.0 * max(0.0, 0.00779 - Q.girth2_top2)
            - 2750.0 * max(0.0, 0.000876 - Q.lam2)
            - 0.0739 * max(0.0, Q.mass - 45.1)
            - 64.2 * max(0.0, Q.mass_over_sum_pt - 0.0903)
            - 21.7 * max(0.0, 0.237 - Q.sj3_dr_max)
            - 0.00773 * max(0.0, 753.0 - Q.sum_pt)
            + 313.0 * max(0.0, Q.width - 0.00339)
            - 1740.0 * max(0.0, 0.219 - Q.N2) * max(0.0, Q.e2_sq - 0.0116)
            - 136.0 * max(0.0, 0.224 - Q.N2) * max(0.0, Q.eccentricity - 0.727)
            - 0.848 * max(0.0, 0.219 - Q.N2) * max(0.0, 63.2 - Q.mass)
            - 0.332 * max(0.0, 0.242 - Q.N2) * max(0.0, 52.9 - Q.pt_7)
            - 19900.0 * max(0.0, 0.0076 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.000612)
            - 17300.0 * max(0.0, 0.000453 - Q.lam2) * max(0.0, 0.173 - Q.dr01)
            - 45400.0 * max(0.0, Q.max_dr - 0.164) * max(0.0, 0.000596 - Q.C2_b2)
            - 0.321 * max(0.0, Q.sd_mass - 37.0) * max(0.0, 0.276 - Q.sd_zg)
            + 0.00274 * max(0.0, Q.sd_mass - 42.1) * max(0.0, 26.6 - Q.sj3_pair_mass_min)
        ))
        + 0.015625 * grid(5, max(0.0, -0.444
            + 240.0 * max(0.0, 0.00764 - Q.girth)
            - 22.2 * max(0.0, Q.log_sum_pt - 6.83)
            - 0.132 * max(0.0, 35.4 - Q.pt_7)
            + 569.0 * max(0.0, 0.0039 - Q.width)
            + 132.0 * max(0.0, 0.0286 - Q.z_7)
            + 69.9 * max(0.0, 0.0473 - Q.z_7)
            + 61.0 * max(0.0, 0.0702 - Q.z_7)
            - 45.2 * max(0.0, 0.021 - Q.zdr_0)
            - 11700.0 * max(0.0, 0.215 - Q.LHA) * max(0.0, 0.00121 - Q.lam1)
            - 67.1 * max(0.0, 0.215 - Q.LHA) * max(0.0, 6.78 - Q.log_sum_pt)
            + 135000.0 * max(0.0, 0.00167 - Q.girth2) * max(0.0, 0.024 - Q.centroid_offset)
            + 981.0 * max(0.0, Q.log_sum_pt - 6.73) * max(0.0, 0.0147 - Q.dr_2)
            - 924.0 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.0244 - Q.dr_2)
            + 84600.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 8.48e-05 - Q.mean_eta2)
            - 902.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, Q.mean_phi - 0.00334)
            + 0.000174 * max(0.0, Q.sum_pt_top5 - 530.0) * max(0.0, 43.2 - Q.pt_6)
            + 0.019 * max(0.0, Q.sum_pt_top5 - 438.0) * max(0.0, Q.sj2_dr - 0.176)
            - 1450.0 * max(0.0, 0.0687 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
            + 0.354 * max(0.0, 0.0708 - Q.z_7) * max(0.0, 45.4 - Q.mass_top3)
        ))
        - 0.375 * grid(6, max(0.0, 2.65
            + 106.0 * max(0.0, Q.centroid_offset - 0.00762)
            - 102.0 * max(0.0, Q.centroid_offset - 0.0193)
            + 547.0 * max(0.0, 0.00301 - Q.e2_sq)
            - 1020.0 * max(0.0, 0.00867 - Q.girth2)
            + 298.0 * max(0.0, 0.0118 - Q.lam1)
            - 1090.0 * max(0.0, 0.00109 - Q.lam2)
            + 0.0803 * max(0.0, 22.5 - Q.mass)
            - 34.4 * max(0.0, 0.144 - Q.max_dr)
            + 0.513 * max(0.0, 19.1 - Q.pt_6)
            + 8.14 * max(0.0, Q.sj3_dr_max - 0.191)
            + 30.4 * max(0.0, 0.181 - Q.sj3_dr_max)
            - 9.6 * max(0.0, 0.29 - Q.sj3_dr_max)
            - 17.8 * max(0.0, Q.sj3_dr_min - 0.0225)
            - 0.0953 * max(0.0, Q.sj3_pair_mass_min - 4.35)
            + 0.0162 * max(0.0, Q.sum_pt - 975.0)
            - 437.0 * max(0.0, 0.0221 - Q.z_6)
            + 5830.0 * max(0.0, Q.centroid_offset - 0.0213) * max(0.0, 0.01 - Q.mean_phi2)
            + 65.4 * max(0.0, Q.centroid_offset - 0.0103) * max(0.0, Q.psi_0p1 - 0.335)
            + 1.84 * max(0.0, Q.centroid_offset - 0.017) * max(0.0, Q.pt_4 - 66.3)
            - 1140.0 * max(0.0, 0.0118 - Q.lam1) * max(0.0, 0.254 - Q.planar_flow)
            + 57.4 * max(0.0, 0.0032 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05)
            - 0.707 * max(0.0, 64.5 - Q.mass) * max(0.0, 0.0534 - Q.z_dr_0p2_0p4)
            - 1210.0 * max(0.0, Q.mean_eta - 0.0349) * max(0.0, Q.M3 - 0.0492)
            + 0.589 * max(0.0, 42.5 - Q.pt_6) * max(0.0, 6.76 - Q.log_sum_pt)
            - 6.53 * max(0.0, 41.5 - Q.pt_6) * max(0.0, Q.z_7 - 0.0235)
            + 176.0 * max(0.0, 0.171 - Q.sj2_dr) * max(0.0, Q.eccentricity - 0.927)
            - 3.68 * max(0.0, Q.sum_pt - 953.0) * max(0.0, 0.00311 - Q.mean_phi2)
            + 0.00255 * max(0.0, 589.0 - Q.sum_pt) * max(0.0, 1.97 - Q.n_dr_0p2_0p4)
            + 0.00863 * max(0.0, 625.0 - Q.sum_pt) * max(0.0, 24.6 - Q.pt_5)
            + 1560.0 * max(0.0, 0.111 - Q.tau1) * max(0.0, 0.0157 - Q.mean_phi2)
        ))
        + 0.46875 * grid(7, max(0.0, 11.6
            - 10.8 * max(0.0, 0.0586 - Q.D2_b2)
            - 118.0 * max(0.0, Q.centroid_offset - 0.0384)
            - 61.0 * max(0.0, 0.0884 - Q.girth)
            - 238.0 * max(0.0, Q.girth2 - 0.00439)
            + 617.0 * max(0.0, Q.girth2 - 0.0134)
            - 1260.0 * max(0.0, 0.00104 - Q.girth2)
            + 603.0 * max(0.0, 0.00109 - Q.girth2_top2)
            - 1780.0 * max(0.0, 0.000271 - Q.lam2)
            + 0.0576 * max(0.0, Q.mass - 36.6)
            - 0.18 * max(0.0, Q.mass - 76.8)
            - 124.0 * max(0.0, Q.mass_over_sum_pt - 0.0108)
            + 158.0 * max(0.0, Q.mass_over_sum_pt - 0.0803)
            - 273.0 * max(0.0, Q.mass_over_sum_pt - 0.0909)
            - 494.0 * max(0.0, Q.mass_over_sum_pt - 0.108)
            + 0.0721 * max(0.0, Q.mass_top5 - 53.0)
            - 18.2 * max(0.0, 0.109 - Q.max_dr)
            + 9.5 * max(0.0, 0.178 - Q.max_dr)
            - 0.0511 * max(0.0, 29.4 - Q.pt_7)
            + 33.0 * max(0.0, Q.sd_rg - 0.283)
            + 28.1 * max(0.0, 0.156 - Q.sj2_dr)
            - 21.4 * max(0.0, 0.187 - Q.sj2_dr)
            + 23.0 * max(0.0, Q.sj3_dr_max - 0.139)
            - 19.4 * max(0.0, Q.sj3_dr_max - 0.261)
            - 24.2 * max(0.0, 0.0407 - Q.tau21_b2)
            - 1120.0 * max(0.0, 0.00664 - Q.width)
            + 15.2 * max(0.0, Q.z_7 - 0.0342)
            + 2.49 * max(0.0, 0.139 - Q.z_dr_0p1_0p2)
            - 17200.0 * max(0.0, 0.02 - Q.centroid_offset) * max(0.0, 0.00426 - Q.C2_b2)
            - 10.0 * max(0.0, Q.centroid_offset - 0.0258) * max(0.0, Q.n_pt_above_50 - 3.71)
            + 2880.0 * max(0.0, 0.0216 - Q.centroid_offset) * max(0.0, 0.0279 - Q.tau21_b2)
            + 4120.0 * max(0.0, 0.0893 - Q.girth) * max(0.0, 0.00429 - Q.C2_b2)
            - 2520.0 * max(0.0, Q.girth2 - 0.0145) * max(0.0, Q.eccentricity - 0.925)
            + 1040.0 * max(0.0, Q.girth2 - 0.00447) * max(0.0, 0.197 - Q.planar_flow)
            + 127.0 * max(0.0, Q.girth2 - 0.0131) * max(0.0, 41.3 - Q.pt_6)
            - 90500.0 * max(0.0, 0.00107 - Q.girth2_top2) * max(0.0, 0.0257 - Q.tau21_b2)
            + 125000.0 * max(0.0, 0.0003 - Q.lam2) * max(0.0, 0.0408 - Q.tau21_b2)
            - 9.31 * max(0.0, Q.mass - 76.2) * max(0.0, Q.zdr_6 - 0.00612)
            + 17.2 * max(0.0, Q.mass_over_sum_pt - 0.0911) * max(0.0, 40.8 - Q.pt_6)
            - 0.224 * max(0.0, 0.216 - Q.planar_flow) * max(0.0, 35.6 - Q.pt_6)
            - 0.947 * max(0.0, Q.sj3_dr_max - 0.132) * max(0.0, 20.0 - Q.pt_6)
            - 111.0 * max(0.0, Q.width - 0.00864) * max(0.0, 40.8 - Q.pt_6)
        ))
        - 0.03125 * grid(9, max(0.0, -5.15
            - 0.17 * max(0.0, Q.D2 - 2.16)
            + 12.7 * max(0.0, 0.0796 - Q.M3)
            + 182.0 * max(0.0, 0.0185 - Q.centroid_offset)
            + 43.6 * max(0.0, Q.e2 - 0.0456)
            - 1790.0 * max(0.0, 0.00327 - Q.e2_sq)
            + 279.0 * max(0.0, Q.girth2 - 0.0134)
            + 2880.0 * max(0.0, 0.00365 - Q.girth2)
            + 865.0 * max(0.0, 0.00601 - Q.girth2)
            + 5410.0 * max(0.0, 0.000315 - Q.lam2)
            - 0.0971 * max(0.0, 30.1 - Q.mass)
            - 0.062 * max(0.0, 52.0 - Q.mass)
            + 18.6 * max(0.0, 0.107 - Q.max_dr)
            - 55.0 * max(0.0, 0.141 - Q.sj3_dr_max)
            + 21.9 * max(0.0, 0.215 - Q.sj3_dr_max)
            + 0.0077 * max(0.0, 986.0 - Q.sum_pt)
            + 7810.0 * max(0.0, 0.000176 - Q.width)
            - 161.0 * max(0.0, 0.00631 - Q.zdr_0)
            - 33.6 * max(0.0, 0.017 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0)
            - 2.01 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, 164.0 - Q.pt_1)
            + 1090.0 * max(0.0, 0.0191 - Q.centroid_offset) * max(0.0, 0.205 - Q.z_2nd)
            + 3990.0 * max(0.0, 0.0528 - Q.girth) * max(0.0, 0.0275 - Q.tau21_b2)
            - 1380.0 * max(0.0, 0.0549 - Q.girth) * max(0.0, Q.z_6 - 0.0338)
            - 9500.0 * max(0.0, 0.00578 - Q.girth2) * max(0.0, 0.00437 - Q.mean_phi)
            - 859.0 * max(0.0, Q.lam1 - 0.00485) * max(0.0, Q.eccentricity - 0.638)
            - 27.2 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.746 - Q.D2_b2)
            + 4.38 * max(0.0, 54.9 - Q.mass) * max(0.0, 0.0267 - Q.centroid_offset)
            + 0.129 * max(0.0, 55.0 - Q.mass) * max(0.0, 6.84 - Q.log_sum_pt)
            + 0.581 * max(0.0, 0.15 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 32.9)
            + 3360.0 * max(0.0, 0.046 - Q.tau1) * max(0.0, -0.0119 - Q.mean_phi)
            - 4610.0 * max(0.0, 0.0457 - Q.tau1) * max(0.0, 0.0286 - Q.tau21_b2)
        ))
        + 0.0546875 * grid(13, max(0.0, 0.714
            - 55.3 * max(0.0, 0.079 - Q.e2)
            + 19700.0 * max(0.0, 5.05e-05 - Q.e3)
            + 58.6 * max(0.0, 0.148 - Q.girth)
            + 253.0 * max(0.0, 0.0161 - Q.lam1)
            - 0.0244 * max(0.0, Q.mass - 49.1)
            - 0.0571 * max(0.0, Q.pt_7 - 48.4)
            - 0.00746 * max(0.0, Q.sum_pt_top5 - 674.0)
            + 0.0137 * max(0.0, Q.sum_pt_top5 - 794.0)
            - 167.0 * max(0.0, 0.0283 - Q.z_5)
            - 189.0 * max(0.0, 0.028 - Q.z_7)
            + 2100.0 * max(0.0, 6.94e-05 - Q.e3) * max(0.0, Q.D3 - 0.0598)
            - 1180000.0 * max(0.0, 5.38e-05 - Q.e3) * max(0.0, 0.0377 - Q.centroid_offset)
            + 125.0 * max(0.0, 0.15 - Q.girth) * max(0.0, 0.0746 - Q.M3)
            - 8780.0 * max(0.0, 0.148 - Q.girth) * max(0.0, 0.000536 - Q.lam2)
            - 75.4 * max(0.0, 0.148 - Q.girth) * max(0.0, 6.81 - Q.log_sum_pt)
            - 0.987 * max(0.0, 0.145 - Q.girth) * max(0.0, 38.2 - Q.pt_7)
            - 1.58 * max(0.0, 0.152 - Q.girth) * max(0.0, Q.sj2_mass1 - 30.4)
            + 281.0 * max(0.0, 0.142 - Q.girth) * max(0.0, Q.tau2 - 0.00886)
            + 392.0 * max(0.0, 0.153 - Q.girth) * max(0.0, Q.z_7 - 0.0616)
            - 2.22 * max(0.0, 31.9 - Q.pt_6) * max(0.0, 0.0624 - Q.z_7)
            - 0.000466 * max(0.0, Q.sum_pt - 1000.0) * max(0.0, 70.0 - Q.pt_6)
            + 0.00108 * max(0.0, Q.sum_pt_top5 - 667.0) * max(0.0, 38.1 - Q.pt_7)
        ))
        + 0.375 * grid(14, max(0.0, -0.689
            - 146.0 * max(0.0, 0.00403 - Q.C2_b2)
            - 84.3 * max(0.0, Q.centroid_offset - 0.0269)
            - 979.0 * max(0.0, Q.centroid_offset - 0.0497)
            - 147.0 * max(0.0, Q.e2 - 0.0167)
            + 258.0 * max(0.0, Q.e2 - 0.0499)
            + 1130.0 * max(0.0, Q.e2_sq - 0.0117)
            - 23600.0 * max(0.0, 7.93e-05 - Q.e3)
            + 84.3 * max(0.0, Q.girth - 0.0272)
            + 70.1 * max(0.0, Q.girth - 0.041)
            + 93.5 * max(0.0, Q.girth - 0.0802)
            - 193.0 * max(0.0, Q.girth - 0.0868)
            - 78.8 * max(0.0, Q.girth - 0.102)
            - 1570.0 * max(0.0, Q.girth2 - 0.00856)
            + 1010.0 * max(0.0, 0.000504 - Q.lam2)
            - 0.0427 * max(0.0, Q.mass - 68.9)
            + 157.0 * max(0.0, Q.mass_over_sum_pt - 0.0804)
            + 173.0 * max(0.0, Q.mass_over_sum_pt - 0.0906)
            + 0.192 * max(0.0, 5.06 - Q.n_dr_0_0p05)
            - 22.5 * max(0.0, Q.psi_0p1 - 0.974)
            + 0.0682 * max(0.0, 49.6 - Q.sd_mass)
            - 0.0178 * max(0.0, 74.8 - Q.sd_mass)
            - 32.8 * max(0.0, Q.sd_rg - 0.234)
            - 9.92 * max(0.0, 0.325 - Q.sd_rg)
            - 6.65 * max(0.0, Q.sj2_dr - 0.0606)
            + 31.8 * max(0.0, Q.sj2_dr - 0.158)
            - 18.9 * max(0.0, Q.sj2_dr - 0.197)
            - 0.00585 * max(0.0, 717.0 - Q.sum_pt)
            + 339.0 * max(0.0, Q.width - 0.00353)
            - 1480.0 * max(0.0, Q.width - 0.0067)
            - 1260.0 * max(0.0, Q.width - 0.0131)
            + 1.5 * max(0.0, Q.z_dr_0_0p05 - 0.152)
            - 2.29 * max(0.0, 0.163 - Q.z_dr_0p05_0p1)
            + 4.72 * max(0.0, 0.0677 - Q.z_dr_0p1_0p2)
            + 1160000.0 * max(0.0, Q.centroid_offset - 0.0499) * max(0.0, 0.000872 - Q.C2_b2)
            + 0.547 * max(0.0, Q.centroid_offset - 0.0267) * max(0.0, 152.0 - Q.pt_1)
            + 218000.0 * max(0.0, 5.56e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.162)
            - 400.0 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 0.0179 - Q.centroid_offset)
            - 4200.0 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.00746 - Q.lam1)
            + 987.0 * max(0.0, 0.115 - Q.planar_flow) * max(0.0, 0.0165 - Q.lam1)
            - 98.8 * max(0.0, 0.109 - Q.planar_flow) * max(0.0, Q.mean_eta - -0.00684)
            - 0.0177 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 652.0 - Q.sum_pt_top5)
            + 3620.0 * max(0.0, 0.112 - Q.planar_flow) * max(0.0, 0.00601 - Q.width)
            - 168.0 * max(0.0, Q.sj2_dr - 0.13) * max(0.0, 0.0886 - Q.D2_b2)
            + 378.0 * max(0.0, Q.sj2_dr - 0.159) * max(0.0, 0.0881 - Q.D2_b2)
            - 264.0 * max(0.0, Q.sj2_dr - 0.198) * max(0.0, 0.0872 - Q.D2_b2)
            - 569.0 * max(0.0, Q.sj2_dr - 0.164) * max(0.0, 0.0332 - Q.dr_2)
            + 289.0 * max(0.0, Q.sj2_dr - 0.197) * max(0.0, 0.0517 - Q.dr_2)
            - 74.7 * max(0.0, Q.sj2_dr - 0.203) * max(0.0, 0.105 - Q.z_5)
            + 0.00254 * max(0.0, 0.627 - Q.z_dr_0p05_0p1) * max(0.0, 714.0 - Q.sum_pt)
            - 3190.0 * max(0.0, Q.z_dr_0p05_0p1 - 0.745) * max(0.0, Q.zdr_5 - 0.0154)
        ))
        - 0.15625 * grid(15, max(0.0, 0.00133
            + 156.0 * max(0.0, 0.0409 - Q.e2)
            - 1130.0 * max(0.0, 0.00818 - Q.e2_sq)
            - 91.6 * max(0.0, 0.102 - Q.girth)
            - 546.0 * max(0.0, 0.00739 - Q.girth2)
            - 10700.0 * max(0.0, 0.000513 - Q.girth2_top2)
            + 655.0 * max(0.0, 0.00708 - Q.mass_over_sum_pt_sq)
            + 44.0 * max(0.0, 0.159 - Q.sj2_dr)
            - 35.7 * max(0.0, 0.199 - Q.sj2_dr)
            - 0.049 * max(0.0, Q.sj3_pair_mass_max - 81.1)
            + 710.0 * max(0.0, 0.0133 - Q.width)
            + 1.25 * max(0.0, 0.229 - Q.N2) * max(0.0, 4.0 - Q.n_dr_0p1_0p2)
            + 0.0181 * max(0.0, 0.218 - Q.N2) * max(0.0, Q.sum_pt_top5 - 498.0)
            - 13.9 * max(0.0, 0.223 - Q.N2) * max(0.0, 0.548 - Q.z_dr_0p05_0p1)
            - 1140.0 * max(0.0, 0.00752 - Q.girth2) * max(0.0, 0.731 - Q.D2)
            + 31.3 * max(0.0, 0.133 - Q.mass_over_sum_pt) * max(0.0, 0.711 - Q.D2)
        ))
    )


def logit_t(Q):
    return (1.34375
        + 0.015625 * grid(0, max(0.0, -0.464
            - 930.0 * max(0.0, 0.00151 - Q.C2_b2)
            - 50.7 * max(0.0, Q.centroid_offset - 0.00783)
            - 347.0 * max(0.0, Q.centroid_offset - 0.0499)
            + 71.7 * max(0.0, 0.0243 - Q.e2)
            + 81.8 * max(0.0, 0.0354 - Q.e2)
            - 47.9 * max(0.0, 0.0751 - Q.girth)
            + 497.0 * max(0.0, 0.013 - Q.girth2)
            - 0.103 * max(0.0, 29.6 - Q.mass)
            - 0.0679 * max(0.0, 64.7 - Q.mass)
            + 6.63 * max(0.0, 0.159 - Q.planar_flow)
            + 11.9 * max(0.0, Q.sj3_dr_max - 0.11)
            - 27.2 * max(0.0, Q.sj3_dr_max - 0.182)
            - 0.0134 * max(0.0, Q.sum_pt - 906.0)
            - 1120.0 * max(0.0, 0.00444 - Q.width)
            - 1540.0 * max(0.0, Q.centroid_offset - 0.0203) * max(0.0, 0.0609 - Q.z_7)
            + 172.0 * max(0.0, 0.0133 - Q.girth2) * max(0.0, 1.02 - Q.D2)
            + 818.0 * max(0.0, 0.0134 - Q.girth2) * max(0.0, Q.phi_0 - -0.00668)
            - 1580.0 * max(0.0, 0.00649 - Q.lam1) * max(0.0, 0.843 - Q.D2)
            + 55000.0 * max(0.0, 8.01e-05 - Q.lam2) * max(0.0, 0.257 - Q.D2_b2)
            + 49.3 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, 0.0746 - Q.dr_4)
            - 458.0 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, Q.z_7 - 0.0178)
            - 661.0 * max(0.0, 0.142 - Q.planar_flow) * max(0.0, 0.023 - Q.dr_2)
            + 510.0 * max(0.0, 0.165 - Q.planar_flow) * max(0.0, 0.0237 - Q.z_7)
            - 190.0 * max(0.0, 0.245 - Q.sj3_dr_max) * max(0.0, 0.058 - Q.z_7)
            + 0.000467 * max(0.0, Q.sum_pt - 915.0) * max(0.0, Q.pt_7 - 35.4)
            + 8830.0 * max(0.0, 0.00414 - Q.width) * max(0.0, 0.0364 - Q.C3)
        ))
        + 0.0625 * grid(3, max(0.0, -4.48
            + 28.4 * max(0.0, Q.centroid_offset - 0.00928)
            + 98.5 * max(0.0, Q.e2 - 0.046)
            + 171.0 * max(0.0, Q.girth - 0.0409)
            - 142.0 * max(0.0, Q.girth - 0.103)
            - 1380.0 * max(0.0, Q.girth2 - 0.00879)
            + 478.0 * max(0.0, Q.girth2 - 0.0187)
            + 566.0 * max(0.0, Q.lam1 - 0.00828)
            - 0.0994 * max(0.0, Q.mass - 65.1)
            - 288.0 * max(0.0, 0.0118 - Q.mass_over_sum_pt_sq)
            + 0.643 * max(0.0, 1.04 - Q.n_dr_0_0p05)
            + 0.0673 * max(0.0, Q.sd_mass - 63.8)
            - 0.0809 * max(0.0, Q.sd_mass - 86.0)
            + 47.1 * max(0.0, Q.sj2_dr - 0.188)
            - 58.7 * max(0.0, Q.sj2_dr - 0.269)
            + 36.7 * max(0.0, Q.sj3_dr_max - 0.196)
            - 19.5 * max(0.0, Q.sj3_dr_max - 0.297)
            - 65.8 * max(0.0, Q.tau1 - 0.0469)
            + 103.0 * max(0.0, Q.tau1 - 0.114)
            - 0.898 * max(0.0, 0.836 - Q.z_dr_0_0p05)
            + 63.1 * max(0.0, Q.girth - 0.0307) * max(0.0, Q.log_sum_pt - 6.11)
            + 16.7 * max(0.0, Q.lam2 - 0.000776) * max(0.0, 64.0 - Q.pt_6)
            - 203.0 * max(0.0, Q.sj2_dr - 0.214) * max(0.0, 0.0549 - Q.dr_3)
            - 0.302 * max(0.0, Q.sj2_dr - 0.206) * max(0.0, Q.sj2_mass1 - -0.0277)
            - 33.0 * max(0.0, Q.sj2_dr - 0.199) * max(0.0, 1.02 - Q.z_dr_0p05_0p1)
            + 60.9 * max(0.0, Q.sj2_dr - 0.268) * max(0.0, 1.03 - Q.z_dr_0p05_0p1)
        ))
        + 0.125 * grid(4, max(0.0, 0.0436
            + 76.8 * max(0.0, 0.0651 - Q.C2)
            + 13.2 * max(0.0, 0.351 - Q.LHA)
            + 53.9 * max(0.0, 0.224 - Q.N2)
            + 648.0 * max(0.0, 0.0116 - Q.e2_sq)
            - 969.0 * max(0.0, 0.00919 - Q.girth2)
            + 220.0 * max(0.0, 0.00779 - Q.girth2_top2)
            - 2750.0 * max(0.0, 0.000876 - Q.lam2)
            - 0.0739 * max(0.0, Q.mass - 45.1)
            - 64.2 * max(0.0, Q.mass_over_sum_pt - 0.0903)
            - 21.7 * max(0.0, 0.237 - Q.sj3_dr_max)
            - 0.00773 * max(0.0, 753.0 - Q.sum_pt)
            + 313.0 * max(0.0, Q.width - 0.00339)
            - 1740.0 * max(0.0, 0.219 - Q.N2) * max(0.0, Q.e2_sq - 0.0116)
            - 136.0 * max(0.0, 0.224 - Q.N2) * max(0.0, Q.eccentricity - 0.727)
            - 0.848 * max(0.0, 0.219 - Q.N2) * max(0.0, 63.2 - Q.mass)
            - 0.332 * max(0.0, 0.242 - Q.N2) * max(0.0, 52.9 - Q.pt_7)
            - 19900.0 * max(0.0, 0.0076 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.000612)
            - 17300.0 * max(0.0, 0.000453 - Q.lam2) * max(0.0, 0.173 - Q.dr01)
            - 45400.0 * max(0.0, Q.max_dr - 0.164) * max(0.0, 0.000596 - Q.C2_b2)
            - 0.321 * max(0.0, Q.sd_mass - 37.0) * max(0.0, 0.276 - Q.sd_zg)
            + 0.00274 * max(0.0, Q.sd_mass - 42.1) * max(0.0, 26.6 - Q.sj3_pair_mass_min)
        ))
        - 0.25 * grid(5, max(0.0, -0.444
            + 240.0 * max(0.0, 0.00764 - Q.girth)
            - 22.2 * max(0.0, Q.log_sum_pt - 6.83)
            - 0.132 * max(0.0, 35.4 - Q.pt_7)
            + 569.0 * max(0.0, 0.0039 - Q.width)
            + 132.0 * max(0.0, 0.0286 - Q.z_7)
            + 69.9 * max(0.0, 0.0473 - Q.z_7)
            + 61.0 * max(0.0, 0.0702 - Q.z_7)
            - 45.2 * max(0.0, 0.021 - Q.zdr_0)
            - 11700.0 * max(0.0, 0.215 - Q.LHA) * max(0.0, 0.00121 - Q.lam1)
            - 67.1 * max(0.0, 0.215 - Q.LHA) * max(0.0, 6.78 - Q.log_sum_pt)
            + 135000.0 * max(0.0, 0.00167 - Q.girth2) * max(0.0, 0.024 - Q.centroid_offset)
            + 981.0 * max(0.0, Q.log_sum_pt - 6.73) * max(0.0, 0.0147 - Q.dr_2)
            - 924.0 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.0244 - Q.dr_2)
            + 84600.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 8.48e-05 - Q.mean_eta2)
            - 902.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, Q.mean_phi - 0.00334)
            + 0.000174 * max(0.0, Q.sum_pt_top5 - 530.0) * max(0.0, 43.2 - Q.pt_6)
            + 0.019 * max(0.0, Q.sum_pt_top5 - 438.0) * max(0.0, Q.sj2_dr - 0.176)
            - 1450.0 * max(0.0, 0.0687 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
            + 0.354 * max(0.0, 0.0708 - Q.z_7) * max(0.0, 45.4 - Q.mass_top3)
        ))
        + 0.1875 * grid(8, max(0.0, -0.632
            - 218.0 * max(0.0, 0.0595 - Q.girth)
            + 1110.0 * max(0.0, 0.00491 - Q.girth2)
            - 5.72 * max(0.0, Q.log_sum_pt - 6.7)
            + 822.0 * max(0.0, 0.00355 - Q.width)
            + 42000.0 * max(0.0, 0.0585 - Q.girth) * max(0.0, Q.width - 0.000656)
            - 24400.0 * max(0.0, 0.00535 - Q.girth2) * max(0.0, Q.centroid_offset - 0.00661)
            + 21200.0 * max(0.0, 0.00662 - Q.girth2) * max(0.0, 0.0259 - Q.centroid_offset)
            + 1120.0 * max(0.0, 0.000136 - Q.lam2) * max(0.0, 5.27 - Q.D2_b2)
            - 6.02 * max(0.0, 31.3 - Q.mass) * max(0.0, 0.0291 - Q.centroid_offset)
            - 720.0 * max(0.0, 0.142 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.0059)
            - 0.00955 * max(0.0, Q.sum_pt_top5 - 679.0) * max(0.0, 8.0 - Q.n_pt_above_10)
            + 30000.0 * max(0.0, 0.0617 - Q.tau1) * max(0.0, 0.00304 - Q.width)
        ))
        + 0.375 * grid(10, max(0.0, 1.59
            + 36.9 * max(0.0, Q.C2_b2 - 0.00763)
            - 29.3 * max(0.0, Q.LHA - 0.305)
            - 2620.0 * max(0.0, Q.e3 - 5.22e-05)
            + 274.0 * max(0.0, Q.girth2 - 0.00749)
            - 90.8 * max(0.0, Q.girth2 - 0.0258)
            + 379.0 * max(0.0, 0.00215 - Q.girth2_top3)
            - 1040.0 * max(0.0, 0.00144 - Q.lam1)
            - 328.0 * max(0.0, 0.00404 - Q.lam1)
            - 361.0 * max(0.0, 0.00598 - Q.lam1)
            + 1290.0 * max(0.0, Q.lam2 - 0.000278)
            + 0.0276 * max(0.0, 79.4 - Q.mass)
            - 0.0122 * max(0.0, Q.mass_top5 - 51.0)
            - 0.0151 * max(0.0, 43.9 - Q.pt_7)
            - 4.57 * max(0.0, Q.sj3_dr_max - 0.184)
            + 0.123 * max(0.0, Q.sj3_pair_mass_min - 15.8)
            - 0.00569 * max(0.0, Q.sum_pt - 988.0)
            + 16.3 * max(0.0, Q.tau1 - 0.057)
            - 241.0 * max(0.0, Q.lam1 - 0.00707) * max(0.0, 0.36 - Q.D2_b2)
            - 702.0 * max(0.0, Q.lam2 - 0.000257) * max(0.0, Q.planar_flow - 0.055)
            + 0.0613 * max(0.0, 45.6 - Q.pt_7) * max(0.0, 0.985 - Q.D2)
            - 0.0904 * max(0.0, 48.0 - Q.pt_7) * max(0.0, 6.56 - Q.log_sum_pt)
            - 0.191 * max(0.0, Q.sj3_pair_mass_min - 6.68) * max(0.0, Q.sj3_pairmin_over_m - 0.245)
            + 1610.0 * max(0.0, 0.0205 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.0125)
            + 115.0 * max(0.0, 0.0242 - Q.zdr_0) * max(0.0, 0.248 - Q.z_dr_0p05_0p1)
        ))
        - 0.5 * grid(12, max(0.0, -0.611
            + 22.4 * max(0.0, Q.centroid_offset - 0.0499)
            - 32.5 * max(0.0, Q.e2 - 0.0634)
            + 260.0 * max(0.0, Q.girth2 - 0.0188)
            - 32.5 * max(0.0, Q.girth2_top2 - 0.014)
            + 0.0546 * max(0.0, Q.mass - 91.2)
            - 29.0 * max(0.0, Q.zdr_0 - 0.0398)
            + 5030.0 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.lam2 - 0.000537)
            - 2.27 * max(0.0, Q.girth2 - 0.0188) * max(0.0, 53.4 - Q.pt_7)
            + 0.59 * max(0.0, Q.mass - 69.6) * max(0.0, Q.centroid_offset - 0.00948)
        ))
        - 0.40625 * grid(13, max(0.0, 0.714
            - 55.3 * max(0.0, 0.079 - Q.e2)
            + 19700.0 * max(0.0, 5.05e-05 - Q.e3)
            + 58.6 * max(0.0, 0.148 - Q.girth)
            + 253.0 * max(0.0, 0.0161 - Q.lam1)
            - 0.0244 * max(0.0, Q.mass - 49.1)
            - 0.0571 * max(0.0, Q.pt_7 - 48.4)
            - 0.00746 * max(0.0, Q.sum_pt_top5 - 674.0)
            + 0.0137 * max(0.0, Q.sum_pt_top5 - 794.0)
            - 167.0 * max(0.0, 0.0283 - Q.z_5)
            - 189.0 * max(0.0, 0.028 - Q.z_7)
            + 2100.0 * max(0.0, 6.94e-05 - Q.e3) * max(0.0, Q.D3 - 0.0598)
            - 1180000.0 * max(0.0, 5.38e-05 - Q.e3) * max(0.0, 0.0377 - Q.centroid_offset)
            + 125.0 * max(0.0, 0.15 - Q.girth) * max(0.0, 0.0746 - Q.M3)
            - 8780.0 * max(0.0, 0.148 - Q.girth) * max(0.0, 0.000536 - Q.lam2)
            - 75.4 * max(0.0, 0.148 - Q.girth) * max(0.0, 6.81 - Q.log_sum_pt)
            - 0.987 * max(0.0, 0.145 - Q.girth) * max(0.0, 38.2 - Q.pt_7)
            - 1.58 * max(0.0, 0.152 - Q.girth) * max(0.0, Q.sj2_mass1 - 30.4)
            + 281.0 * max(0.0, 0.142 - Q.girth) * max(0.0, Q.tau2 - 0.00886)
            + 392.0 * max(0.0, 0.153 - Q.girth) * max(0.0, Q.z_7 - 0.0616)
            - 2.22 * max(0.0, 31.9 - Q.pt_6) * max(0.0, 0.0624 - Q.z_7)
            - 0.000466 * max(0.0, Q.sum_pt - 1000.0) * max(0.0, 70.0 - Q.pt_6)
            + 0.00108 * max(0.0, Q.sum_pt_top5 - 667.0) * max(0.0, 38.1 - Q.pt_7)
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
