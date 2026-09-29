"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network (step 4; no W/Z/H/t mass values offered as thresholds), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 90.0% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_3                    pT of particle 3 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_6               |Δη| of particle 6
  Q.absphi_1               |Δφ| of particle 1
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr1_3                  ΔR between particle 3 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.phi_0                  Δφ of particle 0
  Q.phi_1                  Δφ of particle 1
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
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
  Q.tau4                   N-subjettiness τ4 (β=1)
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
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top2=mass_of(2),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_4=pt[4],
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        pt_7=pt[7],
        z_3=z[3],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_3=z[3] * dr[3],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_0=abs(eta[0]),
        abseta_6=abs(eta[6]),
        absphi_1=abs(phi[1]),
        sj2_dr=subjets(2)["dr"][0],
        dr1_3=math.sqrt(dist2(1, 3)) if pt[3] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        phi_0=phi[0],
        phi_1=phi[1],
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
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
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-0.4375
        - 0.15625 * grid(0, max(0.0, -0.483
            - 775.0 * max(0.0, 0.00158 - Q.C2_b2)
            - 45.5 * max(0.0, Q.centroid_offset - 0.00711)
            - 627.0 * max(0.0, Q.centroid_offset - 0.0499)
            + 98.2 * max(0.0, 0.0248 - Q.e2)
            + 78.2 * max(0.0, 0.0355 - Q.e2)
            - 52.2 * max(0.0, 0.0753 - Q.girth)
            + 292.0 * max(0.0, 0.0132 - Q.girth2)
            + 12600.0 * max(0.0, 0.000264 - Q.lam1)
            - 9140.0 * max(0.0, 7.82e-05 - Q.lam2)
            + 2.12 * max(0.0, Q.log_sum_pt - 6.36)
            - 6.38 * max(0.0, 6.11 - Q.log_sum_pt)
            - 0.0318 * max(0.0, 59.3 - Q.mass)
            - 49.0 * max(0.0, 0.0847 - Q.mass_over_sum_pt)
            + 10.9 * max(0.0, 0.15 - Q.planar_flow)
            + 10.2 * max(0.0, Q.sj3_dr_max - 0.11)
            - 26.2 * max(0.0, Q.sj3_dr_max - 0.183)
            - 0.0137 * max(0.0, Q.sum_pt - 906.0)
            - 1090.0 * max(0.0, 0.00449 - Q.width)
            + 372.0 * max(0.0, 0.00866 - Q.width)
            + 4.35 * max(0.0, Q.centroid_offset - 0.00817) * max(0.0, 6.81 - Q.n_pt_above_50)
            - 1550.0 * max(0.0, Q.centroid_offset - 0.0198) * max(0.0, 0.0628 - Q.z_7)
            + 152.0 * max(0.0, 0.0132 - Q.girth2) * max(0.0, 1.04 - Q.D2)
            + 593.0 * max(0.0, 0.0141 - Q.girth2) * max(0.0, Q.phi_0 - 0.000879)
            - 1290.0 * max(0.0, 0.00631 - Q.lam1) * max(0.0, 0.837 - Q.D2)
            + 56800.0 * max(0.0, 7.64e-05 - Q.lam2) * max(0.0, 0.266 - Q.D2_b2)
            + 61.5 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, 0.0755 - Q.dr_4)
            - 45.0 * max(0.0, Q.log_sum_pt - 6.38) * max(0.0, Q.mean_eta - 0.000559)
            - 438.0 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, Q.z_7 - 0.018)
            - 0.183 * max(0.0, 63.7 - Q.mass) * max(0.0, 0.163 - Q.D2_b2)
            - 666.0 * max(0.0, 0.149 - Q.planar_flow) * max(0.0, 0.0225 - Q.dr_2)
            - 209.0 * max(0.0, 0.274 - Q.sj3_dr_max) * max(0.0, 0.0577 - Q.z_7)
            + 11300.0 * max(0.0, 0.00432 - Q.width) * max(0.0, 0.0369 - Q.C3)
        ))
        + 0.390625 * grid(1, max(0.0, 1.04
            - 29.8 * max(0.0, 0.0962 - Q.girth)
            + 7.34 * max(0.0, Q.log_sum_pt - 6.39)
            + 7.1 * max(0.0, Q.log_sum_pt - 6.59)
            + 578.0 * max(0.0, 0.00605 - Q.mass_over_sum_pt_sq)
            - 0.0776 * max(0.0, Q.pt_7 - 55.0)
            + 3.07 * max(0.0, Q.sj3_dr_max - 0.171)
            - 0.00622 * max(0.0, Q.sum_pt_top5 - 524.0)
            - 448.0 * max(0.0, 0.00669 - Q.width)
            + 9.04 * max(0.0, Q.z_7 - 0.0481)
            - 31.9 * max(0.0, 0.0475 - Q.z_7)
            - 23.7 * max(0.0, 0.00874 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 4.82)
            - 358.0 * max(0.0, 0.0103 - Q.lam1) * max(0.0, 0.145 - Q.planar_flow)
            + 2.12 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 1.49 - Q.D2)
            + 86.7 * max(0.0, Q.log_sum_pt - 6.32) * max(0.0, Q.centroid_offset - 0.0126)
            - 3.48 * max(0.0, Q.log_sum_pt - 6.38) * max(0.0, 0.74 - Q.z_dr_0p05_0p1)
            - 23500.0 * max(0.0, 0.00606 - Q.mass_over_sum_pt_sq) * max(0.0, 0.0392 - Q.D2_b2)
            + 0.00832 * max(0.0, Q.pt_7 - 35.0) * max(0.0, Q.n_dr_0p1_0p2 - 0.551)
            - 0.00323 * max(0.0, Q.pt_7 - 32.5) * max(0.0, 53.1 - Q.pt_6)
            + 0.426 * max(0.0, Q.pt_7 - 33.3) * max(0.0, Q.sj2_dr - 0.136)
            - 0.113 * max(0.0, Q.sj3_dr_max - 0.139) * max(0.0, Q.sj3_pair_mass_min - 1.38)
            - 24.8 * max(0.0, 0.0626 - Q.z_7) * max(0.0, 1.62 - Q.D2)
            - 45.7 * max(0.0, 0.0609 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.09)
        ))
        + 0.4296875 * grid(2, max(0.0, 4.11
            - 10.2 * max(0.0, Q.LHA - 0.112)
            - 86.4 * max(0.0, 0.00464 - Q.centroid_offset)
            - 18.3 * max(0.0, 0.0219 - Q.dr_5)
            - 186.0 * max(0.0, 0.0078 - Q.girth)
            + 146.0 * max(0.0, 0.00431 - Q.girth2)
            + 104.0 * max(0.0, 0.0118 - Q.lam1)
            + 21.2 * max(0.0, Q.log_sum_pt - 6.85)
            + 4.48 * max(0.0, 6.59 - Q.log_sum_pt)
            + 0.0392 * max(0.0, Q.m012 - 42.8)
            + 0.0201 * max(0.0, Q.pt_6 - 27.2)
            + 0.0601 * max(0.0, Q.pt_7 - 43.7)
            - 0.0969 * max(0.0, 53.3 - Q.pt_7)
            - 0.0167 * max(0.0, Q.sj3_pair_mass_max - 28.4)
            + 0.0111 * max(0.0, 798.0 - Q.sum_pt)
            - 0.00524 * max(0.0, Q.sum_pt_top5 - 608.0)
            - 0.00779 * max(0.0, Q.sum_pt_top5 - 835.0)
            - 0.00706 * max(0.0, 718.0 - Q.sum_pt_top5)
            - 46.6 * max(0.0, Q.z_7 - 0.0532)
            + 58.6 * max(0.0, 0.0534 - Q.z_7)
            + 4.75 * max(0.0, 0.00794 - Q.girth) * max(0.0, 72.3 - Q.pt_4)
            - 1870.0 * max(0.0, 0.00596 - Q.lam1) * max(0.0, Q.max_dr - 0.0963)
            - 6.21 * max(0.0, Q.log_sum_pt - 6.88) * max(0.0, 2.99 - Q.D2_b2)
            - 0.149 * max(0.0, Q.log_sum_pt - 6.85) * max(0.0, Q.pt_6 - 43.4)
            - 0.0157 * max(0.0, Q.pt_6 - 31.4) * max(0.0, 0.781 - Q.planar_flow)
            - 0.39 * max(0.0, 67.5 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.0124)
            - 0.409 * max(0.0, 66.3 - Q.sj3_pair_mass_max) * max(0.0, 0.065 - Q.z_7)
            + 0.001 * max(0.0, 588.0 - Q.sum_pt) * max(0.0, 4.15 - Q.D2_b2)
            - 0.828 * max(0.0, 793.0 - Q.sum_pt) * max(0.0, 0.0223 - Q.dr_5)
            + 0.765 * max(0.0, 839.0 - Q.sum_pt) * max(0.0, 0.0223 - Q.dr_5)
            + 0.00577 * max(0.0, Q.sum_pt_top5 - 842.0) * max(0.0, 2.32 - Q.D2_b2)
            - 11.9 * max(0.0, 0.0726 - Q.z_7) * max(0.0, 0.64 - Q.planar_flow)
            - 132.0 * max(0.0, Q.z_7 - 0.0396) * max(0.0, 0.0633 - Q.sj3_dr_min)
        ))
        - 0.03125 * grid(4, max(0.0, -4.63
            + 287.0 * max(0.0, 0.00913 - Q.C2_b2)
            + 48.2 * max(0.0, 0.221 - Q.N2)
            + 494.0 * max(0.0, 0.0119 - Q.e2_sq)
            + 335.0 * max(0.0, 0.0174 - Q.e2_sq)
            - 118000.0 * max(0.0, 1.38e-05 - Q.e3)
            - 43.9 * max(0.0, Q.girth - 0.0543)
            + 340.0 * max(0.0, Q.girth2 - 0.00353)
            + 1250.0 * max(0.0, 0.00251 - Q.girth2)
            - 908.0 * max(0.0, 0.00892 - Q.girth2)
            + 192.0 * max(0.0, 0.00812 - Q.girth2_top5)
            - 3510.0 * max(0.0, 0.000532 - Q.lam2)
            - 1550.0 * max(0.0, 0.00115 - Q.lam2)
            + 20.9 * max(0.0, Q.max_dr - 0.239)
            + 27.1 * max(0.0, Q.sj3_dr_max - 0.162)
            - 41.9 * max(0.0, Q.sj3_dr_max - 0.232)
            - 0.00445 * max(0.0, 762.0 - Q.sum_pt)
            - 0.0917 * max(0.0, 1.1 - Q.D2) * max(0.0, 51.6 - Q.pt_7)
            - 2860.0 * max(0.0, 0.222 - Q.N2) * max(0.0, Q.e2_sq - 0.0128)
            - 94.8 * max(0.0, 0.221 - Q.N2) * max(0.0, Q.eccentricity - 0.696)
            - 0.586 * max(0.0, 0.212 - Q.N2) * max(0.0, 53.1 - Q.mass)
            - 116.0 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, 0.621 - Q.z_dr_0p05_0p1)
            + 592.0 * max(0.0, 0.0112 - Q.e2_sq) * max(0.0, 1.05 - Q.D2)
            - 0.165 * max(0.0, 68.1 - Q.mass) * max(0.0, 0.985 - Q.D2)
            + 292.0 * max(0.0, Q.tau1 - 0.0711) * max(0.0, 0.208 - Q.sj3_dr_min)
        ))
        - 0.1875 * grid(5, max(0.0, 0.948
            - 190.0 * max(0.0, 0.00987 - Q.C3)
            + 410.0 * max(0.0, 0.00258 - Q.girth2)
            + 3.37 * max(0.0, Q.log_sum_pt - 6.26)
            - 12.6 * max(0.0, Q.log_sum_pt - 6.77)
            - 23.1 * max(0.0, 0.0896 - Q.mass_over_sum_pt)
            - 0.0783 * max(0.0, Q.pt_7 - 22.9)
            - 0.0616 * max(0.0, 37.6 - Q.pt_7)
            - 13.4 * max(0.0, 0.109 - Q.sj3_dr_max)
            + 10.5 * max(0.0, 0.186 - Q.sj3_dr_max)
            + 0.00733 * max(0.0, Q.sum_pt_top5 - 714.0)
            + 22.0 * max(0.0, 0.0968 - Q.tau1)
            + 54.3 * max(0.0, 0.0475 - Q.z_6)
            + 165.0 * max(0.0, 0.0283 - Q.z_7)
            - 33.5 * max(0.0, 0.0239 - Q.zdr_0)
            - 52.8 * max(0.0, 0.209 - Q.LHA) * max(0.0, 6.82 - Q.log_sum_pt)
            + 103000.0 * max(0.0, 0.00172 - Q.girth2) * max(0.0, 0.0234 - Q.centroid_offset)
            - 4950.0 * max(0.0, 0.00272 - Q.girth2_top3) * max(0.0, 0.0266 - Q.phi_1)
            + 542.0 * max(0.0, Q.log_sum_pt - 6.27) * max(0.0, 0.0113 - Q.C3)
            - 1.33 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 4.61 - Q.D2_b2)
            + 232000.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 8.69e-05 - Q.mean_eta2)
            - 1750.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.00424 - Q.mean_eta2)
            - 675.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, Q.mean_phi - 0.000708)
            + 0.139 * max(0.0, Q.log_sum_pt - 6.26) * max(0.0, 7.76 - Q.ptdr0_5)
            - 0.231 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, Q.sj3_pair_mass_max - 21.9)
            + 6310.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.00175 - Q.zdr_3)
            - 9410.0 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.00263 - Q.zdr_3)
            - 0.653 * max(0.0, Q.max_pair_mass - 20.4) * max(0.0, 0.248 - Q.dr_max_012)
            - 2.56 * max(0.0, 36.1 - Q.pt_7) * max(0.0, 0.0221 - Q.C3)
            - 173.0 * max(0.0, Q.sum_pt_top5 - 720.0) * max(0.0, 8.92e-05 - Q.mean_eta2)
            + 10.1 * max(0.0, 0.104 - Q.tau1) * max(0.0, 0.464 - Q.planar_flow)
            + 1040.0 * max(0.0, 0.022 - Q.z_6) * max(0.0, 0.167 - Q.abseta_6)
            - 27400.0 * max(0.0, 0.0463 - Q.z_6) * max(0.0, 0.00185 - Q.zdr_3)
            - 912.0 * max(0.0, 0.0738 - Q.z_7) * max(0.0, 0.0304 - Q.centroid_offset)
            + 0.813 * max(0.0, 0.0497 - Q.z_7) * max(0.0, 72.9 - Q.mass_top5)
            - 1070.0 * max(0.0, 0.0237 - Q.zdr_0) * max(0.0, 0.0218 - Q.abseta_0)
        ))
        + 0.109375 * grid(6, max(0.0, 3.15
            + 41.8 * max(0.0, Q.centroid_offset - 0.00779)
            - 60.5 * max(0.0, Q.centroid_offset - 0.0187)
            - 1070.0 * max(0.0, 0.00869 - Q.girth2)
            - 1200.0 * max(0.0, 0.00114 - Q.lam2)
            + 65.1 * max(0.0, 0.0894 - Q.mass_over_sum_pt)
            - 30.5 * max(0.0, 0.145 - Q.max_dr)
            - 68.8 * max(0.0, Q.mean_eta - 0.0325)
            + 0.267 * max(0.0, 19.1 - Q.pt_6)
            - 0.145 * max(0.0, 40.2 - Q.pt_6)
            + 10.3 * max(0.0, Q.sj3_dr_max - 0.189)
            + 39.8 * max(0.0, 0.18 - Q.sj3_dr_max)
            - 9.26 * max(0.0, 0.293 - Q.sj3_dr_max)
            - 14.6 * max(0.0, Q.sj3_dr_min - 0.0235)
            - 0.0504 * max(0.0, Q.sj3_pair_mass_min - 3.99)
            + 0.00969 * max(0.0, 484.0 - Q.sum_pt)
            + 21.7 * max(0.0, 0.118 - Q.tau1)
            + 6800.0 * max(0.0, Q.centroid_offset - 0.0183) * max(0.0, 0.00887 - Q.mean_phi2)
            + 48.5 * max(0.0, Q.centroid_offset - 0.00885) * max(0.0, Q.psi_0p1 - 0.406)
            + 1.29 * max(0.0, Q.centroid_offset - 0.0185) * max(0.0, Q.sum_pt_top5 - 655.0)
            + 106.0 * max(0.0, 0.00761 - Q.lam1) * max(0.0, Q.mass_top5 - 66.8)
            - 544.0 * max(0.0, 0.0132 - Q.lam1) * max(0.0, 0.265 - Q.planar_flow)
            + 79.6 * max(0.0, 0.00333 - Q.lam2) * max(0.0, 2.87 - Q.n_dr_0_0p05)
            - 0.453 * max(0.0, 66.5 - Q.mass) * max(0.0, 0.0612 - Q.z_dr_0p2_0p4)
            - 153.0 * max(0.0, 0.146 - Q.max_dr) * max(0.0, Q.mean_phi - 0.00126)
            + 2.46 * max(0.0, 40.8 - Q.pt_6) * max(0.0, Q.M3 - 0.078)
            + 0.000846 * max(0.0, 41.1 - Q.pt_6) * max(0.0, 995.0 - Q.sum_pt)
            - 6.04 * max(0.0, 41.1 - Q.pt_6) * max(0.0, Q.z_7 - 0.0225)
            + 0.00273 * max(0.0, 601.0 - Q.sum_pt) * max(0.0, 2.19 - Q.n_dr_0p2_0p4)
            + 10.5 * max(0.0, 617.0 - Q.sum_pt) * max(0.0, 0.0507 - Q.z_3)
            + 0.119 * max(0.0, 761.0 - Q.sum_pt) * max(0.0, 0.0766 - Q.z_7)
        ))
        + 0.171875 * grid(9, max(0.0, -3.66
            - 0.422 * max(0.0, Q.D2 - 2.2)
            - 21.3 * max(0.0, 0.0235 - Q.absphi_1)
            - 318.0 * max(0.0, 0.00228 - Q.centroid_offset)
            + 198.0 * max(0.0, 0.0179 - Q.centroid_offset)
            - 60300.0 * max(0.0, 2.39e-05 - Q.e3)
            + 1220.0 * max(0.0, 0.00364 - Q.girth2)
            + 1520.0 * max(0.0, 0.00598 - Q.girth2)
            + 254.0 * max(0.0, Q.lam1 - 0.00845)
            + 527.0 * max(0.0, Q.lam2 - 0.00153)
            + 2830.0 * max(0.0, 0.000325 - Q.lam2)
            - 0.012 * max(0.0, Q.mass - 45.5)
            - 0.129 * max(0.0, 30.4 - Q.mass)
            - 0.0768 * max(0.0, 51.6 - Q.mass)
            - 75.1 * max(0.0, 0.0724 - Q.mass_over_sum_pt)
            + 27.2 * max(0.0, 0.111 - Q.max_dr)
            + 0.0448 * max(0.0, 36.2 - Q.pt_4)
            - 46.8 * max(0.0, 0.143 - Q.sj3_dr_max)
            + 22.9 * max(0.0, 0.214 - Q.sj3_dr_max)
            + 0.00493 * max(0.0, 991.0 - Q.sum_pt)
            + 43.3 * max(0.0, 0.0418 - Q.tau1)
            + 8680.0 * max(0.0, 0.000168 - Q.width)
            - 154.0 * max(0.0, 0.0064 - Q.zdr_0)
            + 5000.0 * max(0.0, 0.0185 - Q.centroid_offset) * max(0.0, Q.mean_eta - 0.000187)
            - 32.6 * max(0.0, 0.0185 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.02)
            - 2.07 * max(0.0, 0.0179 - Q.centroid_offset) * max(0.0, 164.0 - Q.pt_1)
            + 1080.0 * max(0.0, 0.0191 - Q.centroid_offset) * max(0.0, 0.205 - Q.z_2nd)
            + 3460.0 * max(0.0, 0.0552 - Q.girth) * max(0.0, 0.0289 - Q.tau21_b2)
            - 1090.0 * max(0.0, 0.0575 - Q.girth) * max(0.0, Q.z_6 - 0.0334)
            - 52500.0 * max(0.0, 0.00585 - Q.girth2) * max(0.0, Q.mean_phi - 0.0255)
            - 6850.0 * max(0.0, 0.00594 - Q.girth2) * max(0.0, 0.00379 - Q.mean_phi)
            + 333.0 * max(0.0, 0.00677 - Q.girth2) * max(0.0, 0.296 - Q.planar_flow)
            - 724.0 * max(0.0, Q.lam1 - 0.00596) * max(0.0, Q.eccentricity - 0.708)
            - 25.9 * max(0.0, Q.log_sum_pt - 6.87) * max(0.0, 0.754 - Q.D2_b2)
            + 0.0098 * max(0.0, 53.3 - Q.mass) * max(0.0, Q.D2 - 2.81)
            + 4.8 * max(0.0, 54.9 - Q.mass) * max(0.0, 0.0266 - Q.centroid_offset)
            - 0.472 * max(0.0, 54.4 - Q.mass) * max(0.0, Q.dr1_7 - 0.209)
            + 0.146 * max(0.0, 52.7 - Q.mass) * max(0.0, 6.85 - Q.log_sum_pt)
            - 1.99 * max(0.0, 0.0736 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top2 - 1.83)
            + 929.0 * max(0.0, 0.0643 - Q.mass_over_sum_pt) * max(0.0, -0.00767 - Q.phi_0)
            - 0.071 * max(0.0, Q.sd_mass - 44.8) * max(0.0, Q.n_dr_0p05_0p1 - 6.78)
            + 0.497 * max(0.0, 0.146 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.3)
            + 4710.0 * max(0.0, 0.0446 - Q.tau1) * max(0.0, Q.mean_phi - 0.0255)
            - 4250.0 * max(0.0, 0.0447 - Q.tau1) * max(0.0, 0.0309 - Q.tau21_b2)
        ))
    )


def logit_q(Q):
    return (0.03125
        - 0.09375 * grid(4, max(0.0, -4.63
            + 287.0 * max(0.0, 0.00913 - Q.C2_b2)
            + 48.2 * max(0.0, 0.221 - Q.N2)
            + 494.0 * max(0.0, 0.0119 - Q.e2_sq)
            + 335.0 * max(0.0, 0.0174 - Q.e2_sq)
            - 118000.0 * max(0.0, 1.38e-05 - Q.e3)
            - 43.9 * max(0.0, Q.girth - 0.0543)
            + 340.0 * max(0.0, Q.girth2 - 0.00353)
            + 1250.0 * max(0.0, 0.00251 - Q.girth2)
            - 908.0 * max(0.0, 0.00892 - Q.girth2)
            + 192.0 * max(0.0, 0.00812 - Q.girth2_top5)
            - 3510.0 * max(0.0, 0.000532 - Q.lam2)
            - 1550.0 * max(0.0, 0.00115 - Q.lam2)
            + 20.9 * max(0.0, Q.max_dr - 0.239)
            + 27.1 * max(0.0, Q.sj3_dr_max - 0.162)
            - 41.9 * max(0.0, Q.sj3_dr_max - 0.232)
            - 0.00445 * max(0.0, 762.0 - Q.sum_pt)
            - 0.0917 * max(0.0, 1.1 - Q.D2) * max(0.0, 51.6 - Q.pt_7)
            - 2860.0 * max(0.0, 0.222 - Q.N2) * max(0.0, Q.e2_sq - 0.0128)
            - 94.8 * max(0.0, 0.221 - Q.N2) * max(0.0, Q.eccentricity - 0.696)
            - 0.586 * max(0.0, 0.212 - Q.N2) * max(0.0, 53.1 - Q.mass)
            - 116.0 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, 0.621 - Q.z_dr_0p05_0p1)
            + 592.0 * max(0.0, 0.0112 - Q.e2_sq) * max(0.0, 1.05 - Q.D2)
            - 0.165 * max(0.0, 68.1 - Q.mass) * max(0.0, 0.985 - Q.D2)
            + 292.0 * max(0.0, Q.tau1 - 0.0711) * max(0.0, 0.208 - Q.sj3_dr_min)
        ))
        + 0.046875 * grid(5, max(0.0, 0.948
            - 190.0 * max(0.0, 0.00987 - Q.C3)
            + 410.0 * max(0.0, 0.00258 - Q.girth2)
            + 3.37 * max(0.0, Q.log_sum_pt - 6.26)
            - 12.6 * max(0.0, Q.log_sum_pt - 6.77)
            - 23.1 * max(0.0, 0.0896 - Q.mass_over_sum_pt)
            - 0.0783 * max(0.0, Q.pt_7 - 22.9)
            - 0.0616 * max(0.0, 37.6 - Q.pt_7)
            - 13.4 * max(0.0, 0.109 - Q.sj3_dr_max)
            + 10.5 * max(0.0, 0.186 - Q.sj3_dr_max)
            + 0.00733 * max(0.0, Q.sum_pt_top5 - 714.0)
            + 22.0 * max(0.0, 0.0968 - Q.tau1)
            + 54.3 * max(0.0, 0.0475 - Q.z_6)
            + 165.0 * max(0.0, 0.0283 - Q.z_7)
            - 33.5 * max(0.0, 0.0239 - Q.zdr_0)
            - 52.8 * max(0.0, 0.209 - Q.LHA) * max(0.0, 6.82 - Q.log_sum_pt)
            + 103000.0 * max(0.0, 0.00172 - Q.girth2) * max(0.0, 0.0234 - Q.centroid_offset)
            - 4950.0 * max(0.0, 0.00272 - Q.girth2_top3) * max(0.0, 0.0266 - Q.phi_1)
            + 542.0 * max(0.0, Q.log_sum_pt - 6.27) * max(0.0, 0.0113 - Q.C3)
            - 1.33 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 4.61 - Q.D2_b2)
            + 232000.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 8.69e-05 - Q.mean_eta2)
            - 1750.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.00424 - Q.mean_eta2)
            - 675.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, Q.mean_phi - 0.000708)
            + 0.139 * max(0.0, Q.log_sum_pt - 6.26) * max(0.0, 7.76 - Q.ptdr0_5)
            - 0.231 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, Q.sj3_pair_mass_max - 21.9)
            + 6310.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.00175 - Q.zdr_3)
            - 9410.0 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.00263 - Q.zdr_3)
            - 0.653 * max(0.0, Q.max_pair_mass - 20.4) * max(0.0, 0.248 - Q.dr_max_012)
            - 2.56 * max(0.0, 36.1 - Q.pt_7) * max(0.0, 0.0221 - Q.C3)
            - 173.0 * max(0.0, Q.sum_pt_top5 - 720.0) * max(0.0, 8.92e-05 - Q.mean_eta2)
            + 10.1 * max(0.0, 0.104 - Q.tau1) * max(0.0, 0.464 - Q.planar_flow)
            + 1040.0 * max(0.0, 0.022 - Q.z_6) * max(0.0, 0.167 - Q.abseta_6)
            - 27400.0 * max(0.0, 0.0463 - Q.z_6) * max(0.0, 0.00185 - Q.zdr_3)
            - 912.0 * max(0.0, 0.0738 - Q.z_7) * max(0.0, 0.0304 - Q.centroid_offset)
            + 0.813 * max(0.0, 0.0497 - Q.z_7) * max(0.0, 72.9 - Q.mass_top5)
            - 1070.0 * max(0.0, 0.0237 - Q.zdr_0) * max(0.0, 0.0218 - Q.abseta_0)
        ))
        + 0.125 * grid(6, max(0.0, 3.15
            + 41.8 * max(0.0, Q.centroid_offset - 0.00779)
            - 60.5 * max(0.0, Q.centroid_offset - 0.0187)
            - 1070.0 * max(0.0, 0.00869 - Q.girth2)
            - 1200.0 * max(0.0, 0.00114 - Q.lam2)
            + 65.1 * max(0.0, 0.0894 - Q.mass_over_sum_pt)
            - 30.5 * max(0.0, 0.145 - Q.max_dr)
            - 68.8 * max(0.0, Q.mean_eta - 0.0325)
            + 0.267 * max(0.0, 19.1 - Q.pt_6)
            - 0.145 * max(0.0, 40.2 - Q.pt_6)
            + 10.3 * max(0.0, Q.sj3_dr_max - 0.189)
            + 39.8 * max(0.0, 0.18 - Q.sj3_dr_max)
            - 9.26 * max(0.0, 0.293 - Q.sj3_dr_max)
            - 14.6 * max(0.0, Q.sj3_dr_min - 0.0235)
            - 0.0504 * max(0.0, Q.sj3_pair_mass_min - 3.99)
            + 0.00969 * max(0.0, 484.0 - Q.sum_pt)
            + 21.7 * max(0.0, 0.118 - Q.tau1)
            + 6800.0 * max(0.0, Q.centroid_offset - 0.0183) * max(0.0, 0.00887 - Q.mean_phi2)
            + 48.5 * max(0.0, Q.centroid_offset - 0.00885) * max(0.0, Q.psi_0p1 - 0.406)
            + 1.29 * max(0.0, Q.centroid_offset - 0.0185) * max(0.0, Q.sum_pt_top5 - 655.0)
            + 106.0 * max(0.0, 0.00761 - Q.lam1) * max(0.0, Q.mass_top5 - 66.8)
            - 544.0 * max(0.0, 0.0132 - Q.lam1) * max(0.0, 0.265 - Q.planar_flow)
            + 79.6 * max(0.0, 0.00333 - Q.lam2) * max(0.0, 2.87 - Q.n_dr_0_0p05)
            - 0.453 * max(0.0, 66.5 - Q.mass) * max(0.0, 0.0612 - Q.z_dr_0p2_0p4)
            - 153.0 * max(0.0, 0.146 - Q.max_dr) * max(0.0, Q.mean_phi - 0.00126)
            + 2.46 * max(0.0, 40.8 - Q.pt_6) * max(0.0, Q.M3 - 0.078)
            + 0.000846 * max(0.0, 41.1 - Q.pt_6) * max(0.0, 995.0 - Q.sum_pt)
            - 6.04 * max(0.0, 41.1 - Q.pt_6) * max(0.0, Q.z_7 - 0.0225)
            + 0.00273 * max(0.0, 601.0 - Q.sum_pt) * max(0.0, 2.19 - Q.n_dr_0p2_0p4)
            + 10.5 * max(0.0, 617.0 - Q.sum_pt) * max(0.0, 0.0507 - Q.z_3)
            + 0.119 * max(0.0, 761.0 - Q.sum_pt) * max(0.0, 0.0766 - Q.z_7)
        ))
        + 0.0625 * grid(8, max(0.0, -0.548
            - 213.0 * max(0.0, 0.0594 - Q.girth)
            + 922.0 * max(0.0, 0.00495 - Q.girth2)
            + 994.0 * max(0.0, 0.00351 - Q.width)
            + 40500.0 * max(0.0, 0.0584 - Q.girth) * max(0.0, Q.width - 0.000544)
            - 20500.0 * max(0.0, 0.00539 - Q.girth2) * max(0.0, Q.centroid_offset - 0.0057)
            + 20600.0 * max(0.0, 0.00643 - Q.girth2) * max(0.0, 0.0258 - Q.centroid_offset)
            + 995.0 * max(0.0, 0.00014 - Q.lam2) * max(0.0, 5.69 - Q.D2_b2)
            - 119000.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 3.53e-05 - Q.e3)
            - 1.85 * max(0.0, Q.log_sum_pt - 6.72) * max(0.0, Q.n_dr_0p05_0p1 - 0.999)
            - 4.83 * max(0.0, 31.2 - Q.mass) * max(0.0, 0.0297 - Q.centroid_offset)
            - 753.0 * max(0.0, 0.143 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.00799)
            - 0.00681 * max(0.0, Q.sum_pt_top5 - 670.0) * max(0.0, 8.0 - Q.n_pt_above_10)
            + 27000.0 * max(0.0, 0.0605 - Q.tau1) * max(0.0, 0.00306 - Q.width)
        ))
        + 0.2539062 * grid(9, max(0.0, -3.66
            - 0.422 * max(0.0, Q.D2 - 2.2)
            - 21.3 * max(0.0, 0.0235 - Q.absphi_1)
            - 318.0 * max(0.0, 0.00228 - Q.centroid_offset)
            + 198.0 * max(0.0, 0.0179 - Q.centroid_offset)
            - 60300.0 * max(0.0, 2.39e-05 - Q.e3)
            + 1220.0 * max(0.0, 0.00364 - Q.girth2)
            + 1520.0 * max(0.0, 0.00598 - Q.girth2)
            + 254.0 * max(0.0, Q.lam1 - 0.00845)
            + 527.0 * max(0.0, Q.lam2 - 0.00153)
            + 2830.0 * max(0.0, 0.000325 - Q.lam2)
            - 0.012 * max(0.0, Q.mass - 45.5)
            - 0.129 * max(0.0, 30.4 - Q.mass)
            - 0.0768 * max(0.0, 51.6 - Q.mass)
            - 75.1 * max(0.0, 0.0724 - Q.mass_over_sum_pt)
            + 27.2 * max(0.0, 0.111 - Q.max_dr)
            + 0.0448 * max(0.0, 36.2 - Q.pt_4)
            - 46.8 * max(0.0, 0.143 - Q.sj3_dr_max)
            + 22.9 * max(0.0, 0.214 - Q.sj3_dr_max)
            + 0.00493 * max(0.0, 991.0 - Q.sum_pt)
            + 43.3 * max(0.0, 0.0418 - Q.tau1)
            + 8680.0 * max(0.0, 0.000168 - Q.width)
            - 154.0 * max(0.0, 0.0064 - Q.zdr_0)
            + 5000.0 * max(0.0, 0.0185 - Q.centroid_offset) * max(0.0, Q.mean_eta - 0.000187)
            - 32.6 * max(0.0, 0.0185 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.02)
            - 2.07 * max(0.0, 0.0179 - Q.centroid_offset) * max(0.0, 164.0 - Q.pt_1)
            + 1080.0 * max(0.0, 0.0191 - Q.centroid_offset) * max(0.0, 0.205 - Q.z_2nd)
            + 3460.0 * max(0.0, 0.0552 - Q.girth) * max(0.0, 0.0289 - Q.tau21_b2)
            - 1090.0 * max(0.0, 0.0575 - Q.girth) * max(0.0, Q.z_6 - 0.0334)
            - 52500.0 * max(0.0, 0.00585 - Q.girth2) * max(0.0, Q.mean_phi - 0.0255)
            - 6850.0 * max(0.0, 0.00594 - Q.girth2) * max(0.0, 0.00379 - Q.mean_phi)
            + 333.0 * max(0.0, 0.00677 - Q.girth2) * max(0.0, 0.296 - Q.planar_flow)
            - 724.0 * max(0.0, Q.lam1 - 0.00596) * max(0.0, Q.eccentricity - 0.708)
            - 25.9 * max(0.0, Q.log_sum_pt - 6.87) * max(0.0, 0.754 - Q.D2_b2)
            + 0.0098 * max(0.0, 53.3 - Q.mass) * max(0.0, Q.D2 - 2.81)
            + 4.8 * max(0.0, 54.9 - Q.mass) * max(0.0, 0.0266 - Q.centroid_offset)
            - 0.472 * max(0.0, 54.4 - Q.mass) * max(0.0, Q.dr1_7 - 0.209)
            + 0.146 * max(0.0, 52.7 - Q.mass) * max(0.0, 6.85 - Q.log_sum_pt)
            - 1.99 * max(0.0, 0.0736 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top2 - 1.83)
            + 929.0 * max(0.0, 0.0643 - Q.mass_over_sum_pt) * max(0.0, -0.00767 - Q.phi_0)
            - 0.071 * max(0.0, Q.sd_mass - 44.8) * max(0.0, Q.n_dr_0p05_0p1 - 6.78)
            + 0.497 * max(0.0, 0.146 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.3)
            + 4710.0 * max(0.0, 0.0446 - Q.tau1) * max(0.0, Q.mean_phi - 0.0255)
            - 4250.0 * max(0.0, 0.0447 - Q.tau1) * max(0.0, 0.0309 - Q.tau21_b2)
        ))
        - 0.125 * grid(10, max(0.0, 1.48
            - 23.1 * max(0.0, Q.LHA - 0.304)
            - 34.2 * max(0.0, 0.0255 - Q.M2)
            + 13.7 * max(0.0, 0.0229 - Q.centroid_offset)
            + 144.0 * max(0.0, Q.girth2 - 0.00732)
            - 84.3 * max(0.0, Q.girth2 - 0.0253)
            + 262.0 * max(0.0, 0.00217 - Q.girth2_top3)
            - 886.0 * max(0.0, 0.00165 - Q.lam1)
            - 618.0 * max(0.0, 0.00435 - Q.lam1)
            + 1280.0 * max(0.0, Q.lam2 - 0.000304)
            + 0.0257 * max(0.0, 76.0 - Q.mass)
            + 0.0157 * max(0.0, Q.mass_top5 - 22.8)
            - 0.0223 * max(0.0, Q.mass_top5 - 45.4)
            - 0.0238 * max(0.0, 45.0 - Q.pt_7)
            + 13.8 * max(0.0, Q.sj3_dr_min - 0.127)
            + 0.128 * max(0.0, Q.sj3_pair_mass_min - 15.2)
            - 0.00683 * max(0.0, Q.sum_pt - 986.0)
            + 7.68 * max(0.0, Q.tau1 - 0.0557)
            - 29.8 * max(0.0, 0.0208 - Q.zdr_0)
            - 85100.0 * max(0.0, 7.95e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.177)
            - 182.0 * max(0.0, Q.lam1 - 0.00686) * max(0.0, 0.383 - Q.D2_b2)
            - 351.0 * max(0.0, 0.00619 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.215)
            - 695.0 * max(0.0, Q.lam2 - 0.00023) * max(0.0, Q.planar_flow - 0.0514)
            - 24.1 * max(0.0, Q.lam2 - 0.000412) * max(0.0, 31.9 - Q.pt_6)
            + 0.0757 * max(0.0, 45.6 - Q.pt_7) * max(0.0, 1.0 - Q.D2)
            - 0.0706 * max(0.0, 44.7 - Q.pt_7) * max(0.0, 6.54 - Q.log_sum_pt)
            - 22.2 * max(0.0, Q.sj3_dr_min - 0.134) * max(0.0, 0.43 - Q.z_dr_0_0p05)
            - 0.26 * max(0.0, Q.sj3_pair_mass_min - 9.6) * max(0.0, Q.sj3_pairmin_over_m - 0.292)
            + 14.4 * max(0.0, Q.tau1 - 0.0451) * max(0.0, 0.973 - Q.D2)
            + 1680.0 * max(0.0, 0.0209 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.0107)
            + 175.0 * max(0.0, 0.022 - Q.zdr_0) * max(0.0, Q.dr_7 - 0.0771)
            + 134.0 * max(0.0, 0.0213 - Q.zdr_0) * max(0.0, 0.26 - Q.z_dr_0p05_0p1)
        ))
        + 0.0625 * grid(15, max(0.0, -1.75
            - 6.37 * max(0.0, 0.228 - Q.N2)
            + 84.6 * max(0.0, 0.0416 - Q.e2)
            - 517.0 * max(0.0, 0.00834 - Q.e2_sq)
            - 81.0 * max(0.0, 0.0992 - Q.girth)
            - 698.0 * max(0.0, 0.00754 - Q.girth2)
            + 4000.0 * max(0.0, 0.000524 - Q.girth2_top2)
            + 182.0 * max(0.0, 0.00752 - Q.girth2_top2)
            + 137.0 * max(0.0, 0.00686 - Q.girth2_top3)
            - 616.0 * max(0.0, 0.00498 - Q.lam1)
            - 1990.0 * max(0.0, 0.000329 - Q.lam2)
            - 5.64 * max(0.0, 6.09 - Q.log_sum_pt)
            + 0.0563 * max(0.0, Q.mass - 45.8)
            - 0.0485 * max(0.0, Q.mass - 77.3)
            + 26.3 * max(0.0, 0.163 - Q.sj2_dr)
            - 19.8 * max(0.0, 0.2 - Q.sj2_dr)
            - 0.0189 * max(0.0, Q.sj3_pair_mass_max - 38.7)
            + 93.5 * max(0.0, 0.0622 - Q.tau1)
            - 19.5 * max(0.0, Q.z_dr_0p05_0p1 - 0.735)
            + 0.906 * max(0.0, 0.23 - Q.N2) * max(0.0, 4.02 - Q.n_dr_0p1_0p2)
            - 3.45 * max(0.0, 0.228 - Q.N2) * max(0.0, 5.9 - Q.n_for_90pct)
            + 0.0336 * max(0.0, 0.217 - Q.N2) * max(0.0, 717.0 - Q.sum_pt_top5)
            + 0.0383 * max(0.0, 0.226 - Q.N2) * max(0.0, Q.sum_pt_top5 - 426.0)
            - 7.63 * max(0.0, 0.229 - Q.N2) * max(0.0, 0.56 - Q.z_dr_0p05_0p1)
            - 441.0 * max(0.0, 0.0179 - Q.centroid_offset) * max(0.0, 0.286 - Q.dr01)
            - 15600.0 * max(0.0, 0.0199 - Q.centroid_offset) * max(0.0, 0.0123 - Q.zdr_0)
            + 100.0 * max(0.0, 0.065 - Q.e2) * max(0.0, Q.dr12 - 0.0982)
            + 411000.0 * max(0.0, 0.0106 - Q.e2_sq) * max(0.0, Q.lam2 - -1.14e-06)
            - 126.0 * max(0.0, 0.1 - Q.girth) * max(0.0, Q.dr1_3 - 0.155)
            - 1230.0 * max(0.0, 0.00758 - Q.girth2) * max(0.0, 0.771 - Q.D2)
            - 11400.0 * max(0.0, 0.00111 - Q.lam2) * max(0.0, Q.mean_phi - -0.0179)
            - 0.00426 * max(0.0, Q.mass - 46.7) * max(0.0, 23.5 - Q.pt_7)
            - 168000.0 * max(0.0, 0.0267 - Q.tau1) * max(0.0, Q.zdr_3 - 0.00824)
            + 2140.0 * max(0.0, 0.00611 - Q.width) * max(0.0, 0.713 - Q.D2)
            + 181.0 * max(0.0, 0.0182 - Q.width) * max(0.0, Q.dr1_3 - 0.0627)
            + 138000.0 * max(0.0, 0.0187 - Q.width) * max(0.0, 0.00343 - Q.lam2)
            + 4.3 * max(0.0, 0.0187 - Q.width) * max(0.0, Q.ptdr0_2 - 14.7)
            + 8.82 * max(0.0, Q.z_dr_0p05_0p1 - 0.741) * max(0.0, 2.03 - Q.n_dr_0p2_0p4)
        ))
    )


def logit_W(Q):
    return (-0.125
        + 0.34375 * grid(0, max(0.0, -0.483
            - 775.0 * max(0.0, 0.00158 - Q.C2_b2)
            - 45.5 * max(0.0, Q.centroid_offset - 0.00711)
            - 627.0 * max(0.0, Q.centroid_offset - 0.0499)
            + 98.2 * max(0.0, 0.0248 - Q.e2)
            + 78.2 * max(0.0, 0.0355 - Q.e2)
            - 52.2 * max(0.0, 0.0753 - Q.girth)
            + 292.0 * max(0.0, 0.0132 - Q.girth2)
            + 12600.0 * max(0.0, 0.000264 - Q.lam1)
            - 9140.0 * max(0.0, 7.82e-05 - Q.lam2)
            + 2.12 * max(0.0, Q.log_sum_pt - 6.36)
            - 6.38 * max(0.0, 6.11 - Q.log_sum_pt)
            - 0.0318 * max(0.0, 59.3 - Q.mass)
            - 49.0 * max(0.0, 0.0847 - Q.mass_over_sum_pt)
            + 10.9 * max(0.0, 0.15 - Q.planar_flow)
            + 10.2 * max(0.0, Q.sj3_dr_max - 0.11)
            - 26.2 * max(0.0, Q.sj3_dr_max - 0.183)
            - 0.0137 * max(0.0, Q.sum_pt - 906.0)
            - 1090.0 * max(0.0, 0.00449 - Q.width)
            + 372.0 * max(0.0, 0.00866 - Q.width)
            + 4.35 * max(0.0, Q.centroid_offset - 0.00817) * max(0.0, 6.81 - Q.n_pt_above_50)
            - 1550.0 * max(0.0, Q.centroid_offset - 0.0198) * max(0.0, 0.0628 - Q.z_7)
            + 152.0 * max(0.0, 0.0132 - Q.girth2) * max(0.0, 1.04 - Q.D2)
            + 593.0 * max(0.0, 0.0141 - Q.girth2) * max(0.0, Q.phi_0 - 0.000879)
            - 1290.0 * max(0.0, 0.00631 - Q.lam1) * max(0.0, 0.837 - Q.D2)
            + 56800.0 * max(0.0, 7.64e-05 - Q.lam2) * max(0.0, 0.266 - Q.D2_b2)
            + 61.5 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, 0.0755 - Q.dr_4)
            - 45.0 * max(0.0, Q.log_sum_pt - 6.38) * max(0.0, Q.mean_eta - 0.000559)
            - 438.0 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, Q.z_7 - 0.018)
            - 0.183 * max(0.0, 63.7 - Q.mass) * max(0.0, 0.163 - Q.D2_b2)
            - 666.0 * max(0.0, 0.149 - Q.planar_flow) * max(0.0, 0.0225 - Q.dr_2)
            - 209.0 * max(0.0, 0.274 - Q.sj3_dr_max) * max(0.0, 0.0577 - Q.z_7)
            + 11300.0 * max(0.0, 0.00432 - Q.width) * max(0.0, 0.0369 - Q.C3)
        ))
        - 0.03125 * grid(1, max(0.0, 1.04
            - 29.8 * max(0.0, 0.0962 - Q.girth)
            + 7.34 * max(0.0, Q.log_sum_pt - 6.39)
            + 7.1 * max(0.0, Q.log_sum_pt - 6.59)
            + 578.0 * max(0.0, 0.00605 - Q.mass_over_sum_pt_sq)
            - 0.0776 * max(0.0, Q.pt_7 - 55.0)
            + 3.07 * max(0.0, Q.sj3_dr_max - 0.171)
            - 0.00622 * max(0.0, Q.sum_pt_top5 - 524.0)
            - 448.0 * max(0.0, 0.00669 - Q.width)
            + 9.04 * max(0.0, Q.z_7 - 0.0481)
            - 31.9 * max(0.0, 0.0475 - Q.z_7)
            - 23.7 * max(0.0, 0.00874 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 4.82)
            - 358.0 * max(0.0, 0.0103 - Q.lam1) * max(0.0, 0.145 - Q.planar_flow)
            + 2.12 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 1.49 - Q.D2)
            + 86.7 * max(0.0, Q.log_sum_pt - 6.32) * max(0.0, Q.centroid_offset - 0.0126)
            - 3.48 * max(0.0, Q.log_sum_pt - 6.38) * max(0.0, 0.74 - Q.z_dr_0p05_0p1)
            - 23500.0 * max(0.0, 0.00606 - Q.mass_over_sum_pt_sq) * max(0.0, 0.0392 - Q.D2_b2)
            + 0.00832 * max(0.0, Q.pt_7 - 35.0) * max(0.0, Q.n_dr_0p1_0p2 - 0.551)
            - 0.00323 * max(0.0, Q.pt_7 - 32.5) * max(0.0, 53.1 - Q.pt_6)
            + 0.426 * max(0.0, Q.pt_7 - 33.3) * max(0.0, Q.sj2_dr - 0.136)
            - 0.113 * max(0.0, Q.sj3_dr_max - 0.139) * max(0.0, Q.sj3_pair_mass_min - 1.38)
            - 24.8 * max(0.0, 0.0626 - Q.z_7) * max(0.0, 1.62 - Q.D2)
            - 45.7 * max(0.0, 0.0609 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.09)
        ))
        - 0.5 * grid(3, max(0.0, -4.33
            + 158.0 * max(0.0, Q.e2 - 0.0628)
            + 66.8 * max(0.0, Q.girth - 0.0382)
            + 89.6 * max(0.0, Q.girth - 0.077)
            - 1180.0 * max(0.0, Q.girth2 - 0.0087)
            - 157.0 * max(0.0, 0.00529 - Q.girth2)
            + 670.0 * max(0.0, Q.lam1 - 0.0081)
            + 396.0 * max(0.0, Q.lam1 - 0.016)
            - 0.0696 * max(0.0, Q.mass - 65.4)
            + 58.5 * max(0.0, Q.mass_over_sum_pt - 0.0707)
            + 11.8 * max(0.0, Q.max_dr - 0.103)
            + 41.4 * max(0.0, Q.mean_eta - 0.018)
            + 25.5 * max(0.0, -0.00445 - Q.mean_eta)
            + 24.7 * max(0.0, Q.sj2_dr - 0.188)
            - 38.7 * max(0.0, Q.tau1 - 0.053)
            - 638.0 * max(0.0, Q.width - 0.0133)
            + 7.86 * max(0.0, 0.069 - Q.z_dr_0_0p05)
            + 554.0 * max(0.0, Q.centroid_offset - 0.0107) * max(0.0, 0.0822 - Q.abseta_0)
            + 79.9 * max(0.0, Q.girth - 0.0415) * max(0.0, Q.log_sum_pt - 6.12)
            + 27.6 * max(0.0, Q.lam2 - 0.000916) * max(0.0, 56.4 - Q.pt_6)
            + 10200.0 * max(0.0, Q.lam2 - 0.00162) * max(0.0, Q.z_6 - 0.0476)
            - 1350.0 * max(0.0, Q.mass_over_sum_pt - 0.0659) * max(0.0, 0.219 - Q.sj2_dr)
            - 731.0 * max(0.0, Q.max_dr - 0.112) * max(0.0, 0.0104 - Q.zdr_1)
            - 344.0 * max(0.0, Q.sj2_dr - 0.186) * max(0.0, 0.0524 - Q.dr_3)
            - 0.459 * max(0.0, Q.sj2_dr - 0.187) * max(0.0, Q.sj2_mass1 - 5.18)
            + 369.0 * max(0.0, Q.width - 0.00678) * max(0.0, Q.sj3_pairmin_over_m - 0.0341)
        ))
        - 0.3125 * grid(6, max(0.0, 3.15
            + 41.8 * max(0.0, Q.centroid_offset - 0.00779)
            - 60.5 * max(0.0, Q.centroid_offset - 0.0187)
            - 1070.0 * max(0.0, 0.00869 - Q.girth2)
            - 1200.0 * max(0.0, 0.00114 - Q.lam2)
            + 65.1 * max(0.0, 0.0894 - Q.mass_over_sum_pt)
            - 30.5 * max(0.0, 0.145 - Q.max_dr)
            - 68.8 * max(0.0, Q.mean_eta - 0.0325)
            + 0.267 * max(0.0, 19.1 - Q.pt_6)
            - 0.145 * max(0.0, 40.2 - Q.pt_6)
            + 10.3 * max(0.0, Q.sj3_dr_max - 0.189)
            + 39.8 * max(0.0, 0.18 - Q.sj3_dr_max)
            - 9.26 * max(0.0, 0.293 - Q.sj3_dr_max)
            - 14.6 * max(0.0, Q.sj3_dr_min - 0.0235)
            - 0.0504 * max(0.0, Q.sj3_pair_mass_min - 3.99)
            + 0.00969 * max(0.0, 484.0 - Q.sum_pt)
            + 21.7 * max(0.0, 0.118 - Q.tau1)
            + 6800.0 * max(0.0, Q.centroid_offset - 0.0183) * max(0.0, 0.00887 - Q.mean_phi2)
            + 48.5 * max(0.0, Q.centroid_offset - 0.00885) * max(0.0, Q.psi_0p1 - 0.406)
            + 1.29 * max(0.0, Q.centroid_offset - 0.0185) * max(0.0, Q.sum_pt_top5 - 655.0)
            + 106.0 * max(0.0, 0.00761 - Q.lam1) * max(0.0, Q.mass_top5 - 66.8)
            - 544.0 * max(0.0, 0.0132 - Q.lam1) * max(0.0, 0.265 - Q.planar_flow)
            + 79.6 * max(0.0, 0.00333 - Q.lam2) * max(0.0, 2.87 - Q.n_dr_0_0p05)
            - 0.453 * max(0.0, 66.5 - Q.mass) * max(0.0, 0.0612 - Q.z_dr_0p2_0p4)
            - 153.0 * max(0.0, 0.146 - Q.max_dr) * max(0.0, Q.mean_phi - 0.00126)
            + 2.46 * max(0.0, 40.8 - Q.pt_6) * max(0.0, Q.M3 - 0.078)
            + 0.000846 * max(0.0, 41.1 - Q.pt_6) * max(0.0, 995.0 - Q.sum_pt)
            - 6.04 * max(0.0, 41.1 - Q.pt_6) * max(0.0, Q.z_7 - 0.0225)
            + 0.00273 * max(0.0, 601.0 - Q.sum_pt) * max(0.0, 2.19 - Q.n_dr_0p2_0p4)
            + 10.5 * max(0.0, 617.0 - Q.sum_pt) * max(0.0, 0.0507 - Q.z_3)
            + 0.119 * max(0.0, 761.0 - Q.sum_pt) * max(0.0, 0.0766 - Q.z_7)
        ))
        + 0.21875 * grid(7, max(0.0, 9.14
            - 17.9 * max(0.0, 0.0568 - Q.D2_b2)
            + 11.0 * max(0.0, 0.303 - Q.LHA)
            - 76.7 * max(0.0, Q.centroid_offset - 0.0381)
            - 242.0 * max(0.0, Q.e2 - 0.0503)
            - 78.9 * max(0.0, 0.0868 - Q.girth)
            - 407.0 * max(0.0, Q.girth2 - 0.00462)
            + 199.0 * max(0.0, Q.girth2 - 0.0135)
            - 1320.0 * max(0.0, 0.00107 - Q.girth2)
            - 3100.0 * max(0.0, 0.000261 - Q.lam2)
            + 0.0467 * max(0.0, Q.mass - 35.9)
            - 0.227 * max(0.0, Q.mass - 75.9)
            - 71.3 * max(0.0, Q.mass_over_sum_pt - 0.0113)
            + 116.0 * max(0.0, Q.mass_over_sum_pt - 0.0732)
            + 184.0 * max(0.0, Q.mass_over_sum_pt - 0.0842)
            - 283.0 * max(0.0, Q.mass_over_sum_pt - 0.0902)
            - 213.0 * max(0.0, Q.mass_over_sum_pt - 0.108)
            + 0.0709 * max(0.0, Q.mass_top5 - 53.4)
            - 14.8 * max(0.0, 0.111 - Q.max_dr)
            + 9.95 * max(0.0, 0.174 - Q.max_dr)
            - 0.0459 * max(0.0, 29.0 - Q.pt_7)
            - 21.7 * max(0.0, Q.sd_rg - 0.204)
            + 40.7 * max(0.0, Q.sd_rg - 0.279)
            + 30.6 * max(0.0, 0.158 - Q.sj2_dr)
            - 21.7 * max(0.0, 0.187 - Q.sj2_dr)
            + 19.2 * max(0.0, Q.sj3_dr_max - 0.145)
            - 19.6 * max(0.0, Q.sj3_dr_max - 0.265)
            + 32.1 * max(0.0, 0.0549 - Q.tau1)
            - 25.7 * max(0.0, 0.0413 - Q.tau21_b2)
            - 683.0 * max(0.0, Q.width - 0.00559)
            - 992.0 * max(0.0, 0.00675 - Q.width)
            + 11.7 * max(0.0, Q.z_7 - 0.0333)
            + 1.54 * max(0.0, 0.144 - Q.z_dr_0p1_0p2)
            - 19500.0 * max(0.0, 0.0204 - Q.centroid_offset) * max(0.0, 0.0042 - Q.C2_b2)
            - 15.8 * max(0.0, Q.centroid_offset - 0.0297) * max(0.0, Q.n_pt_above_50 - 3.89)
            - 28.6 * max(0.0, Q.centroid_offset - 0.0378) * max(0.0, Q.pt_4 - 81.4)
            + 2540.0 * max(0.0, 0.0203 - Q.centroid_offset) * max(0.0, 0.0272 - Q.tau21_b2)
            + 3690.0 * max(0.0, 0.0892 - Q.girth) * max(0.0, 0.00397 - Q.C2_b2)
            - 25600.0 * max(0.0, Q.girth2 - 0.0133) * max(0.0, 0.0732 - Q.dr_6)
            + 23200.0 * max(0.0, Q.girth2 - 0.0132) * max(0.0, Q.eccentricity - 0.943)
            + 881.0 * max(0.0, Q.girth2 - 0.00467) * max(0.0, 0.181 - Q.planar_flow)
            + 54.3 * max(0.0, Q.girth2 - 0.0135) * max(0.0, 39.2 - Q.pt_6)
            - 104000.0 * max(0.0, 0.00102 - Q.girth2_top2) * max(0.0, 0.0266 - Q.tau21_b2)
            + 146000.0 * max(0.0, 0.000311 - Q.lam2) * max(0.0, 0.04 - Q.tau21_b2)
            - 10.4 * max(0.0, Q.mass - 76.8) * max(0.0, Q.zdr_6 - 0.00582)
            - 0.937 * max(0.0, Q.mass_over_sum_pt - 0.0155) * max(0.0, 35.0 - Q.pt_6)
            + 17.9 * max(0.0, Q.mass_over_sum_pt - 0.0904) * max(0.0, 37.4 - Q.pt_6)
            + 0.127 * max(0.0, 0.197 - Q.planar_flow) * max(0.0, Q.sd_mass - 39.0)
            + 596.0 * max(0.0, Q.sd_rg - 0.204) * max(0.0, 0.0702 - Q.dr_6)
            - 0.825 * max(0.0, Q.sj3_dr_max - 0.132) * max(0.0, 19.3 - Q.pt_6)
            + 2730.0 * max(0.0, 0.00672 - Q.width) * max(0.0, 0.00613 - Q.mean_phi)
            - 121.0 * max(0.0, Q.width - 0.00861) * max(0.0, 38.5 - Q.pt_6)
        ))
        - 0.25 * grid(8, max(0.0, -0.548
            - 213.0 * max(0.0, 0.0594 - Q.girth)
            + 922.0 * max(0.0, 0.00495 - Q.girth2)
            + 994.0 * max(0.0, 0.00351 - Q.width)
            + 40500.0 * max(0.0, 0.0584 - Q.girth) * max(0.0, Q.width - 0.000544)
            - 20500.0 * max(0.0, 0.00539 - Q.girth2) * max(0.0, Q.centroid_offset - 0.0057)
            + 20600.0 * max(0.0, 0.00643 - Q.girth2) * max(0.0, 0.0258 - Q.centroid_offset)
            + 995.0 * max(0.0, 0.00014 - Q.lam2) * max(0.0, 5.69 - Q.D2_b2)
            - 119000.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 3.53e-05 - Q.e3)
            - 1.85 * max(0.0, Q.log_sum_pt - 6.72) * max(0.0, Q.n_dr_0p05_0p1 - 0.999)
            - 4.83 * max(0.0, 31.2 - Q.mass) * max(0.0, 0.0297 - Q.centroid_offset)
            - 753.0 * max(0.0, 0.143 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.00799)
            - 0.00681 * max(0.0, Q.sum_pt_top5 - 670.0) * max(0.0, 8.0 - Q.n_pt_above_10)
            + 27000.0 * max(0.0, 0.0605 - Q.tau1) * max(0.0, 0.00306 - Q.width)
        ))
        - 0.03125 * grid(9, max(0.0, -3.66
            - 0.422 * max(0.0, Q.D2 - 2.2)
            - 21.3 * max(0.0, 0.0235 - Q.absphi_1)
            - 318.0 * max(0.0, 0.00228 - Q.centroid_offset)
            + 198.0 * max(0.0, 0.0179 - Q.centroid_offset)
            - 60300.0 * max(0.0, 2.39e-05 - Q.e3)
            + 1220.0 * max(0.0, 0.00364 - Q.girth2)
            + 1520.0 * max(0.0, 0.00598 - Q.girth2)
            + 254.0 * max(0.0, Q.lam1 - 0.00845)
            + 527.0 * max(0.0, Q.lam2 - 0.00153)
            + 2830.0 * max(0.0, 0.000325 - Q.lam2)
            - 0.012 * max(0.0, Q.mass - 45.5)
            - 0.129 * max(0.0, 30.4 - Q.mass)
            - 0.0768 * max(0.0, 51.6 - Q.mass)
            - 75.1 * max(0.0, 0.0724 - Q.mass_over_sum_pt)
            + 27.2 * max(0.0, 0.111 - Q.max_dr)
            + 0.0448 * max(0.0, 36.2 - Q.pt_4)
            - 46.8 * max(0.0, 0.143 - Q.sj3_dr_max)
            + 22.9 * max(0.0, 0.214 - Q.sj3_dr_max)
            + 0.00493 * max(0.0, 991.0 - Q.sum_pt)
            + 43.3 * max(0.0, 0.0418 - Q.tau1)
            + 8680.0 * max(0.0, 0.000168 - Q.width)
            - 154.0 * max(0.0, 0.0064 - Q.zdr_0)
            + 5000.0 * max(0.0, 0.0185 - Q.centroid_offset) * max(0.0, Q.mean_eta - 0.000187)
            - 32.6 * max(0.0, 0.0185 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.02)
            - 2.07 * max(0.0, 0.0179 - Q.centroid_offset) * max(0.0, 164.0 - Q.pt_1)
            + 1080.0 * max(0.0, 0.0191 - Q.centroid_offset) * max(0.0, 0.205 - Q.z_2nd)
            + 3460.0 * max(0.0, 0.0552 - Q.girth) * max(0.0, 0.0289 - Q.tau21_b2)
            - 1090.0 * max(0.0, 0.0575 - Q.girth) * max(0.0, Q.z_6 - 0.0334)
            - 52500.0 * max(0.0, 0.00585 - Q.girth2) * max(0.0, Q.mean_phi - 0.0255)
            - 6850.0 * max(0.0, 0.00594 - Q.girth2) * max(0.0, 0.00379 - Q.mean_phi)
            + 333.0 * max(0.0, 0.00677 - Q.girth2) * max(0.0, 0.296 - Q.planar_flow)
            - 724.0 * max(0.0, Q.lam1 - 0.00596) * max(0.0, Q.eccentricity - 0.708)
            - 25.9 * max(0.0, Q.log_sum_pt - 6.87) * max(0.0, 0.754 - Q.D2_b2)
            + 0.0098 * max(0.0, 53.3 - Q.mass) * max(0.0, Q.D2 - 2.81)
            + 4.8 * max(0.0, 54.9 - Q.mass) * max(0.0, 0.0266 - Q.centroid_offset)
            - 0.472 * max(0.0, 54.4 - Q.mass) * max(0.0, Q.dr1_7 - 0.209)
            + 0.146 * max(0.0, 52.7 - Q.mass) * max(0.0, 6.85 - Q.log_sum_pt)
            - 1.99 * max(0.0, 0.0736 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top2 - 1.83)
            + 929.0 * max(0.0, 0.0643 - Q.mass_over_sum_pt) * max(0.0, -0.00767 - Q.phi_0)
            - 0.071 * max(0.0, Q.sd_mass - 44.8) * max(0.0, Q.n_dr_0p05_0p1 - 6.78)
            + 0.497 * max(0.0, 0.146 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.3)
            + 4710.0 * max(0.0, 0.0446 - Q.tau1) * max(0.0, Q.mean_phi - 0.0255)
            - 4250.0 * max(0.0, 0.0447 - Q.tau1) * max(0.0, 0.0309 - Q.tau21_b2)
        ))
        + 0.375 * grid(11, max(0.0, -0.741
            - 62.0 * max(0.0, Q.centroid_offset - 0.0231)
            + 334.0 * max(0.0, Q.centroid_offset - 0.0467)
            + 66.0 * max(0.0, 0.0237 - Q.centroid_offset)
            + 58.3 * max(0.0, 0.0167 - Q.e2)
            - 973.0 * max(0.0, 0.00806 - Q.e2_sq)
            + 52.8 * max(0.0, Q.eccentricity - 0.988)
            - 66.4 * max(0.0, 0.0263 - Q.girth)
            - 74.4 * max(0.0, 0.0722 - Q.girth)
            + 305.0 * max(0.0, 0.0132 - Q.girth2)
            - 480.0 * max(0.0, 0.0084 - Q.lam1)
            + 0.0955 * max(0.0, 15.8 - Q.mass)
            - 0.0459 * max(0.0, 68.2 - Q.mass)
            + 5.96 * max(0.0, 0.143 - Q.max_dr)
            - 0.0296 * max(0.0, Q.max_pair_mass - 32.4)
            - 0.358 * max(0.0, Q.n_dr_0p1_0p2 - 3.01)
            + 6.09 * max(0.0, 0.248 - Q.planar_flow)
            - 0.0484 * max(0.0, 24.2 - Q.pt_6)
            - 0.0502 * max(0.0, Q.pt_7 - 28.6)
            + 14.4 * max(0.0, Q.sj2_dr - 0.266)
            - 24.8 * max(0.0, 0.169 - Q.sj3_dr_max)
            + 8.87 * max(0.0, 0.263 - Q.sj3_dr_max)
            - 0.00334 * max(0.0, 501.0 - Q.sum_pt_top5)
            + 1930.0 * max(0.0, 0.00865 - Q.width)
            + 45.7 * max(0.0, Q.z_7 - 0.0167)
            - 81.1 * max(0.0, Q.centroid_offset - 0.0481) * max(0.0, 3.22 - Q.D2)
            + 182.0 * max(0.0, 0.0137 - Q.centroid_offset) * max(0.0, 0.592 - Q.D2_b2)
            - 556.0 * max(0.0, 0.042 - Q.centroid_offset) * max(0.0, 0.0004 - Q.mean_phi)
            - 10.1 * max(0.0, 0.014 - Q.centroid_offset) * max(0.0, 16.6 - Q.sj3_pair_mass_min)
            - 0.167 * max(0.0, 0.0392 - Q.centroid_offset) * max(0.0, 895.0 - Q.sum_pt)
            - 86200.0 * max(0.0, Q.centroid_offset - 0.0488) * max(0.0, Q.tau4 - 0.00201)
            - 7570.0 * max(0.0, Q.centroid_offset - 0.0504) * max(0.0, Q.zdr_7 - 0.00649)
            - 4.5 * max(0.0, 0.00865 - Q.e2_sq) * max(0.0, Q.mass_top2 - 9.42)
            - 18.4 * max(0.0, Q.eccentricity - 0.988) * max(0.0, 2.9 - Q.sj2_mass2)
            - 432000.0 * max(0.0, 0.247 - Q.planar_flow) * max(0.0, 9.97e-06 - Q.e3)
            - 0.17 * max(0.0, 0.255 - Q.planar_flow) * max(0.0, 37.0 - Q.pt_7)
            - 0.142 * max(0.0, 0.252 - Q.planar_flow) * max(0.0, Q.sj3_pair_mass_min - 1.69)
            - 0.0156 * max(0.0, 0.245 - Q.planar_flow) * max(0.0, 847.0 - Q.sum_pt)
            + 177.0 * max(0.0, Q.sj2_dr - 0.177) * max(0.0, 0.0469 - Q.dr_7)
            + 91.1 * max(0.0, 0.267 - Q.sj3_dr_max) * max(0.0, -0.0184 - Q.phi_0)
        ))
        + 0.0703125 * grid(13, max(0.0, 0.402
            - 47.7 * max(0.0, 0.0799 - Q.e2)
            + 11000.0 * max(0.0, 5.1e-05 - Q.e3)
            + 62.6 * max(0.0, 0.149 - Q.girth)
            + 184.0 * max(0.0, 0.016 - Q.lam1)
            - 0.0171 * max(0.0, Q.mass - 50.0)
            - 0.0455 * max(0.0, Q.pt_7 - 48.6)
            - 2.74 * max(0.0, Q.sj3_dr23 - 0.2)
            - 192.0 * max(0.0, 0.0276 - Q.z_5)
            - 132.0 * max(0.0, 0.0273 - Q.z_7)
            + 3150.0 * max(0.0, 6.01e-05 - Q.e3) * max(0.0, Q.D3 - 0.207)
            - 914000.0 * max(0.0, 5.3e-05 - Q.e3) * max(0.0, 0.0378 - Q.centroid_offset)
            + 108.0 * max(0.0, 0.149 - Q.girth) * max(0.0, 0.0747 - Q.M3)
            - 9030.0 * max(0.0, 0.142 - Q.girth) * max(0.0, 0.000592 - Q.lam2)
            - 64.3 * max(0.0, 0.149 - Q.girth) * max(0.0, 6.81 - Q.log_sum_pt)
            - 0.998 * max(0.0, 0.145 - Q.girth) * max(0.0, 38.3 - Q.pt_7)
            - 1.61 * max(0.0, 0.146 - Q.girth) * max(0.0, Q.sj2_mass1 - 31.7)
            + 288.0 * max(0.0, 0.144 - Q.girth) * max(0.0, Q.tau2 - 0.00941)
            + 349.0 * max(0.0, 0.152 - Q.girth) * max(0.0, Q.z_7 - 0.0616)
            - 4.15 * max(0.0, 31.8 - Q.pt_6) * max(0.0, 0.0579 - Q.z_7)
            - 0.000761 * max(0.0, Q.sum_pt - 995.0) * max(0.0, 54.8 - Q.pt_6)
            + 0.000988 * max(0.0, Q.sum_pt_top5 - 789.0) * max(0.0, 35.9 - Q.pt_6)
            + 0.000721 * max(0.0, Q.sum_pt_top5 - 662.0) * max(0.0, 40.4 - Q.pt_7)
        ))
        - 0.75 * grid(14, max(0.0, -0.173
            - 146.0 * max(0.0, 0.00406 - Q.C2_b2)
            + 26.7 * max(0.0, Q.LHA - 0.304)
            - 71.3 * max(0.0, Q.centroid_offset - 0.0381)
            - 740.0 * max(0.0, Q.centroid_offset - 0.0498)
            - 97.6 * max(0.0, Q.e2 - 0.0167)
            + 319.0 * max(0.0, Q.e2 - 0.0501)
            - 792.0 * max(0.0, Q.e2_sq - 0.0116)
            - 17300.0 * max(0.0, 7.91e-05 - Q.e3)
            + 54.3 * max(0.0, Q.girth - 0.027)
            + 54.4 * max(0.0, Q.girth - 0.0412)
            + 58.2 * max(0.0, Q.girth - 0.0807)
            - 267.0 * max(0.0, Q.girth - 0.0873)
            - 154.0 * max(0.0, Q.girth - 0.102)
            - 1070.0 * max(0.0, Q.girth2 - 0.00754)
            + 1180.0 * max(0.0, 0.000504 - Q.lam2)
            - 0.0265 * max(0.0, Q.mass - 71.0)
            + 141.0 * max(0.0, Q.mass_over_sum_pt - 0.0798)
            + 163.0 * max(0.0, Q.mass_over_sum_pt - 0.0848)
            + 0.194 * max(0.0, 5.06 - Q.n_dr_0_0p05)
            - 13.2 * max(0.0, Q.psi_0p1 - 0.973)
            + 0.0452 * max(0.0, 49.2 - Q.sd_mass)
            - 17.1 * max(0.0, Q.sd_rg - 0.234)
            + 18.9 * max(0.0, Q.sd_rg - 0.279)
            - 9.48 * max(0.0, 0.325 - Q.sd_rg)
            - 9.74 * max(0.0, Q.sj2_dr - 0.0614)
            + 28.4 * max(0.0, Q.sj2_dr - 0.159)
            - 12.3 * max(0.0, Q.sj2_dr - 0.199)
            - 0.00528 * max(0.0, 724.0 - Q.sum_pt)
            + 324.0 * max(0.0, Q.width - 0.00348)
            - 1310.0 * max(0.0, Q.width - 0.00668)
            - 99.8 * max(0.0, Q.width - 0.0119)
            + 1.4 * max(0.0, Q.z_dr_0_0p05 - 0.152)
            - 2.28 * max(0.0, 0.163 - Q.z_dr_0p05_0p1)
            + 908000.0 * max(0.0, Q.centroid_offset - 0.0497) * max(0.0, 0.000829 - Q.C2_b2)
            - 2030.0 * max(0.0, Q.e2_sq - 0.00194) * max(0.0, 0.184 - Q.sj2_dr)
            + 153000.0 * max(0.0, 5.6e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.164)
            - 78.9 * max(0.0, Q.girth - 0.027) * max(0.0, Q.D2_b2 - 4.77)
            - 303.0 * max(0.0, 0.109 - Q.planar_flow) * max(0.0, 0.0183 - Q.centroid_offset)
            - 2710.0 * max(0.0, 0.105 - Q.planar_flow) * max(0.0, 0.0074 - Q.lam1)
            + 1070.0 * max(0.0, 0.115 - Q.planar_flow) * max(0.0, 0.0164 - Q.lam1)
            - 120.0 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, Q.mean_eta - -0.00694)
            - 0.0193 * max(0.0, 0.095 - Q.planar_flow) * max(0.0, 667.0 - Q.sum_pt_top5)
            + 37.5 * max(0.0, Q.psi_0p1 - 0.821) * max(0.0, 0.0838 - Q.D2_b2)
            - 59.5 * max(0.0, Q.sd_rg - 0.278) * max(0.0, 0.263 - Q.D2_b2)
            - 265.0 * max(0.0, Q.sj2_dr - 0.13) * max(0.0, 0.087 - Q.D2_b2)
            + 659.0 * max(0.0, Q.sj2_dr - 0.159) * max(0.0, 0.0866 - Q.D2_b2)
            - 516.0 * max(0.0, Q.sj2_dr - 0.2) * max(0.0, 0.0864 - Q.D2_b2)
            - 529.0 * max(0.0, Q.sj2_dr - 0.16) * max(0.0, 0.033 - Q.dr_2)
            + 271.0 * max(0.0, Q.sj2_dr - 0.198) * max(0.0, 0.0508 - Q.dr_2)
            - 47.0 * max(0.0, Q.sj2_dr - 0.199) * max(0.0, 0.106 - Q.z_5)
            - 321.0 * max(0.0, Q.z_dr_0p05_0p1 - 0.75) * max(0.0, 0.024 - Q.C2_b2)
            - 4.77 * max(0.0, 0.579 - Q.z_dr_0p05_0p1) * max(0.0, 0.184 - Q.D2_b2)
            + 3.61 * max(0.0, 0.589 - Q.z_dr_0p05_0p1) * max(0.0, 0.315 - Q.D3)
            + 6.91 * max(0.0, Q.z_dr_0p05_0p1 - 0.752) * max(0.0, 1.01 - Q.n_dr_0p2_0p4)
            + 0.0025 * max(0.0, 0.601 - Q.z_dr_0p05_0p1) * max(0.0, 720.0 - Q.sum_pt)
            - 3970.0 * max(0.0, Q.z_dr_0p05_0p1 - 0.754) * max(0.0, Q.zdr_5 - 0.0156)
        ))
        - 0.6875 * grid(15, max(0.0, -1.75
            - 6.37 * max(0.0, 0.228 - Q.N2)
            + 84.6 * max(0.0, 0.0416 - Q.e2)
            - 517.0 * max(0.0, 0.00834 - Q.e2_sq)
            - 81.0 * max(0.0, 0.0992 - Q.girth)
            - 698.0 * max(0.0, 0.00754 - Q.girth2)
            + 4000.0 * max(0.0, 0.000524 - Q.girth2_top2)
            + 182.0 * max(0.0, 0.00752 - Q.girth2_top2)
            + 137.0 * max(0.0, 0.00686 - Q.girth2_top3)
            - 616.0 * max(0.0, 0.00498 - Q.lam1)
            - 1990.0 * max(0.0, 0.000329 - Q.lam2)
            - 5.64 * max(0.0, 6.09 - Q.log_sum_pt)
            + 0.0563 * max(0.0, Q.mass - 45.8)
            - 0.0485 * max(0.0, Q.mass - 77.3)
            + 26.3 * max(0.0, 0.163 - Q.sj2_dr)
            - 19.8 * max(0.0, 0.2 - Q.sj2_dr)
            - 0.0189 * max(0.0, Q.sj3_pair_mass_max - 38.7)
            + 93.5 * max(0.0, 0.0622 - Q.tau1)
            - 19.5 * max(0.0, Q.z_dr_0p05_0p1 - 0.735)
            + 0.906 * max(0.0, 0.23 - Q.N2) * max(0.0, 4.02 - Q.n_dr_0p1_0p2)
            - 3.45 * max(0.0, 0.228 - Q.N2) * max(0.0, 5.9 - Q.n_for_90pct)
            + 0.0336 * max(0.0, 0.217 - Q.N2) * max(0.0, 717.0 - Q.sum_pt_top5)
            + 0.0383 * max(0.0, 0.226 - Q.N2) * max(0.0, Q.sum_pt_top5 - 426.0)
            - 7.63 * max(0.0, 0.229 - Q.N2) * max(0.0, 0.56 - Q.z_dr_0p05_0p1)
            - 441.0 * max(0.0, 0.0179 - Q.centroid_offset) * max(0.0, 0.286 - Q.dr01)
            - 15600.0 * max(0.0, 0.0199 - Q.centroid_offset) * max(0.0, 0.0123 - Q.zdr_0)
            + 100.0 * max(0.0, 0.065 - Q.e2) * max(0.0, Q.dr12 - 0.0982)
            + 411000.0 * max(0.0, 0.0106 - Q.e2_sq) * max(0.0, Q.lam2 - -1.14e-06)
            - 126.0 * max(0.0, 0.1 - Q.girth) * max(0.0, Q.dr1_3 - 0.155)
            - 1230.0 * max(0.0, 0.00758 - Q.girth2) * max(0.0, 0.771 - Q.D2)
            - 11400.0 * max(0.0, 0.00111 - Q.lam2) * max(0.0, Q.mean_phi - -0.0179)
            - 0.00426 * max(0.0, Q.mass - 46.7) * max(0.0, 23.5 - Q.pt_7)
            - 168000.0 * max(0.0, 0.0267 - Q.tau1) * max(0.0, Q.zdr_3 - 0.00824)
            + 2140.0 * max(0.0, 0.00611 - Q.width) * max(0.0, 0.713 - Q.D2)
            + 181.0 * max(0.0, 0.0182 - Q.width) * max(0.0, Q.dr1_3 - 0.0627)
            + 138000.0 * max(0.0, 0.0187 - Q.width) * max(0.0, 0.00343 - Q.lam2)
            + 4.3 * max(0.0, 0.0187 - Q.width) * max(0.0, Q.ptdr0_2 - 14.7)
            + 8.82 * max(0.0, Q.z_dr_0p05_0p1 - 0.741) * max(0.0, 2.03 - Q.n_dr_0p2_0p4)
        ))
    )


def logit_Z(Q):
    return (-0.09375
        + 0.125 * grid(1, max(0.0, 1.04
            - 29.8 * max(0.0, 0.0962 - Q.girth)
            + 7.34 * max(0.0, Q.log_sum_pt - 6.39)
            + 7.1 * max(0.0, Q.log_sum_pt - 6.59)
            + 578.0 * max(0.0, 0.00605 - Q.mass_over_sum_pt_sq)
            - 0.0776 * max(0.0, Q.pt_7 - 55.0)
            + 3.07 * max(0.0, Q.sj3_dr_max - 0.171)
            - 0.00622 * max(0.0, Q.sum_pt_top5 - 524.0)
            - 448.0 * max(0.0, 0.00669 - Q.width)
            + 9.04 * max(0.0, Q.z_7 - 0.0481)
            - 31.9 * max(0.0, 0.0475 - Q.z_7)
            - 23.7 * max(0.0, 0.00874 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 4.82)
            - 358.0 * max(0.0, 0.0103 - Q.lam1) * max(0.0, 0.145 - Q.planar_flow)
            + 2.12 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 1.49 - Q.D2)
            + 86.7 * max(0.0, Q.log_sum_pt - 6.32) * max(0.0, Q.centroid_offset - 0.0126)
            - 3.48 * max(0.0, Q.log_sum_pt - 6.38) * max(0.0, 0.74 - Q.z_dr_0p05_0p1)
            - 23500.0 * max(0.0, 0.00606 - Q.mass_over_sum_pt_sq) * max(0.0, 0.0392 - Q.D2_b2)
            + 0.00832 * max(0.0, Q.pt_7 - 35.0) * max(0.0, Q.n_dr_0p1_0p2 - 0.551)
            - 0.00323 * max(0.0, Q.pt_7 - 32.5) * max(0.0, 53.1 - Q.pt_6)
            + 0.426 * max(0.0, Q.pt_7 - 33.3) * max(0.0, Q.sj2_dr - 0.136)
            - 0.113 * max(0.0, Q.sj3_dr_max - 0.139) * max(0.0, Q.sj3_pair_mass_min - 1.38)
            - 24.8 * max(0.0, 0.0626 - Q.z_7) * max(0.0, 1.62 - Q.D2)
            - 45.7 * max(0.0, 0.0609 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.09)
        ))
        - 0.5625 * grid(3, max(0.0, -4.33
            + 158.0 * max(0.0, Q.e2 - 0.0628)
            + 66.8 * max(0.0, Q.girth - 0.0382)
            + 89.6 * max(0.0, Q.girth - 0.077)
            - 1180.0 * max(0.0, Q.girth2 - 0.0087)
            - 157.0 * max(0.0, 0.00529 - Q.girth2)
            + 670.0 * max(0.0, Q.lam1 - 0.0081)
            + 396.0 * max(0.0, Q.lam1 - 0.016)
            - 0.0696 * max(0.0, Q.mass - 65.4)
            + 58.5 * max(0.0, Q.mass_over_sum_pt - 0.0707)
            + 11.8 * max(0.0, Q.max_dr - 0.103)
            + 41.4 * max(0.0, Q.mean_eta - 0.018)
            + 25.5 * max(0.0, -0.00445 - Q.mean_eta)
            + 24.7 * max(0.0, Q.sj2_dr - 0.188)
            - 38.7 * max(0.0, Q.tau1 - 0.053)
            - 638.0 * max(0.0, Q.width - 0.0133)
            + 7.86 * max(0.0, 0.069 - Q.z_dr_0_0p05)
            + 554.0 * max(0.0, Q.centroid_offset - 0.0107) * max(0.0, 0.0822 - Q.abseta_0)
            + 79.9 * max(0.0, Q.girth - 0.0415) * max(0.0, Q.log_sum_pt - 6.12)
            + 27.6 * max(0.0, Q.lam2 - 0.000916) * max(0.0, 56.4 - Q.pt_6)
            + 10200.0 * max(0.0, Q.lam2 - 0.00162) * max(0.0, Q.z_6 - 0.0476)
            - 1350.0 * max(0.0, Q.mass_over_sum_pt - 0.0659) * max(0.0, 0.219 - Q.sj2_dr)
            - 731.0 * max(0.0, Q.max_dr - 0.112) * max(0.0, 0.0104 - Q.zdr_1)
            - 344.0 * max(0.0, Q.sj2_dr - 0.186) * max(0.0, 0.0524 - Q.dr_3)
            - 0.459 * max(0.0, Q.sj2_dr - 0.187) * max(0.0, Q.sj2_mass1 - 5.18)
            + 369.0 * max(0.0, Q.width - 0.00678) * max(0.0, Q.sj3_pairmin_over_m - 0.0341)
        ))
        + 0.078125 * grid(4, max(0.0, -4.63
            + 287.0 * max(0.0, 0.00913 - Q.C2_b2)
            + 48.2 * max(0.0, 0.221 - Q.N2)
            + 494.0 * max(0.0, 0.0119 - Q.e2_sq)
            + 335.0 * max(0.0, 0.0174 - Q.e2_sq)
            - 118000.0 * max(0.0, 1.38e-05 - Q.e3)
            - 43.9 * max(0.0, Q.girth - 0.0543)
            + 340.0 * max(0.0, Q.girth2 - 0.00353)
            + 1250.0 * max(0.0, 0.00251 - Q.girth2)
            - 908.0 * max(0.0, 0.00892 - Q.girth2)
            + 192.0 * max(0.0, 0.00812 - Q.girth2_top5)
            - 3510.0 * max(0.0, 0.000532 - Q.lam2)
            - 1550.0 * max(0.0, 0.00115 - Q.lam2)
            + 20.9 * max(0.0, Q.max_dr - 0.239)
            + 27.1 * max(0.0, Q.sj3_dr_max - 0.162)
            - 41.9 * max(0.0, Q.sj3_dr_max - 0.232)
            - 0.00445 * max(0.0, 762.0 - Q.sum_pt)
            - 0.0917 * max(0.0, 1.1 - Q.D2) * max(0.0, 51.6 - Q.pt_7)
            - 2860.0 * max(0.0, 0.222 - Q.N2) * max(0.0, Q.e2_sq - 0.0128)
            - 94.8 * max(0.0, 0.221 - Q.N2) * max(0.0, Q.eccentricity - 0.696)
            - 0.586 * max(0.0, 0.212 - Q.N2) * max(0.0, 53.1 - Q.mass)
            - 116.0 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, 0.621 - Q.z_dr_0p05_0p1)
            + 592.0 * max(0.0, 0.0112 - Q.e2_sq) * max(0.0, 1.05 - Q.D2)
            - 0.165 * max(0.0, 68.1 - Q.mass) * max(0.0, 0.985 - Q.D2)
            + 292.0 * max(0.0, Q.tau1 - 0.0711) * max(0.0, 0.208 - Q.sj3_dr_min)
        ))
        + 0.015625 * grid(5, max(0.0, 0.948
            - 190.0 * max(0.0, 0.00987 - Q.C3)
            + 410.0 * max(0.0, 0.00258 - Q.girth2)
            + 3.37 * max(0.0, Q.log_sum_pt - 6.26)
            - 12.6 * max(0.0, Q.log_sum_pt - 6.77)
            - 23.1 * max(0.0, 0.0896 - Q.mass_over_sum_pt)
            - 0.0783 * max(0.0, Q.pt_7 - 22.9)
            - 0.0616 * max(0.0, 37.6 - Q.pt_7)
            - 13.4 * max(0.0, 0.109 - Q.sj3_dr_max)
            + 10.5 * max(0.0, 0.186 - Q.sj3_dr_max)
            + 0.00733 * max(0.0, Q.sum_pt_top5 - 714.0)
            + 22.0 * max(0.0, 0.0968 - Q.tau1)
            + 54.3 * max(0.0, 0.0475 - Q.z_6)
            + 165.0 * max(0.0, 0.0283 - Q.z_7)
            - 33.5 * max(0.0, 0.0239 - Q.zdr_0)
            - 52.8 * max(0.0, 0.209 - Q.LHA) * max(0.0, 6.82 - Q.log_sum_pt)
            + 103000.0 * max(0.0, 0.00172 - Q.girth2) * max(0.0, 0.0234 - Q.centroid_offset)
            - 4950.0 * max(0.0, 0.00272 - Q.girth2_top3) * max(0.0, 0.0266 - Q.phi_1)
            + 542.0 * max(0.0, Q.log_sum_pt - 6.27) * max(0.0, 0.0113 - Q.C3)
            - 1.33 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 4.61 - Q.D2_b2)
            + 232000.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 8.69e-05 - Q.mean_eta2)
            - 1750.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.00424 - Q.mean_eta2)
            - 675.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, Q.mean_phi - 0.000708)
            + 0.139 * max(0.0, Q.log_sum_pt - 6.26) * max(0.0, 7.76 - Q.ptdr0_5)
            - 0.231 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, Q.sj3_pair_mass_max - 21.9)
            + 6310.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.00175 - Q.zdr_3)
            - 9410.0 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.00263 - Q.zdr_3)
            - 0.653 * max(0.0, Q.max_pair_mass - 20.4) * max(0.0, 0.248 - Q.dr_max_012)
            - 2.56 * max(0.0, 36.1 - Q.pt_7) * max(0.0, 0.0221 - Q.C3)
            - 173.0 * max(0.0, Q.sum_pt_top5 - 720.0) * max(0.0, 8.92e-05 - Q.mean_eta2)
            + 10.1 * max(0.0, 0.104 - Q.tau1) * max(0.0, 0.464 - Q.planar_flow)
            + 1040.0 * max(0.0, 0.022 - Q.z_6) * max(0.0, 0.167 - Q.abseta_6)
            - 27400.0 * max(0.0, 0.0463 - Q.z_6) * max(0.0, 0.00185 - Q.zdr_3)
            - 912.0 * max(0.0, 0.0738 - Q.z_7) * max(0.0, 0.0304 - Q.centroid_offset)
            + 0.813 * max(0.0, 0.0497 - Q.z_7) * max(0.0, 72.9 - Q.mass_top5)
            - 1070.0 * max(0.0, 0.0237 - Q.zdr_0) * max(0.0, 0.0218 - Q.abseta_0)
        ))
        - 0.375 * grid(6, max(0.0, 3.15
            + 41.8 * max(0.0, Q.centroid_offset - 0.00779)
            - 60.5 * max(0.0, Q.centroid_offset - 0.0187)
            - 1070.0 * max(0.0, 0.00869 - Q.girth2)
            - 1200.0 * max(0.0, 0.00114 - Q.lam2)
            + 65.1 * max(0.0, 0.0894 - Q.mass_over_sum_pt)
            - 30.5 * max(0.0, 0.145 - Q.max_dr)
            - 68.8 * max(0.0, Q.mean_eta - 0.0325)
            + 0.267 * max(0.0, 19.1 - Q.pt_6)
            - 0.145 * max(0.0, 40.2 - Q.pt_6)
            + 10.3 * max(0.0, Q.sj3_dr_max - 0.189)
            + 39.8 * max(0.0, 0.18 - Q.sj3_dr_max)
            - 9.26 * max(0.0, 0.293 - Q.sj3_dr_max)
            - 14.6 * max(0.0, Q.sj3_dr_min - 0.0235)
            - 0.0504 * max(0.0, Q.sj3_pair_mass_min - 3.99)
            + 0.00969 * max(0.0, 484.0 - Q.sum_pt)
            + 21.7 * max(0.0, 0.118 - Q.tau1)
            + 6800.0 * max(0.0, Q.centroid_offset - 0.0183) * max(0.0, 0.00887 - Q.mean_phi2)
            + 48.5 * max(0.0, Q.centroid_offset - 0.00885) * max(0.0, Q.psi_0p1 - 0.406)
            + 1.29 * max(0.0, Q.centroid_offset - 0.0185) * max(0.0, Q.sum_pt_top5 - 655.0)
            + 106.0 * max(0.0, 0.00761 - Q.lam1) * max(0.0, Q.mass_top5 - 66.8)
            - 544.0 * max(0.0, 0.0132 - Q.lam1) * max(0.0, 0.265 - Q.planar_flow)
            + 79.6 * max(0.0, 0.00333 - Q.lam2) * max(0.0, 2.87 - Q.n_dr_0_0p05)
            - 0.453 * max(0.0, 66.5 - Q.mass) * max(0.0, 0.0612 - Q.z_dr_0p2_0p4)
            - 153.0 * max(0.0, 0.146 - Q.max_dr) * max(0.0, Q.mean_phi - 0.00126)
            + 2.46 * max(0.0, 40.8 - Q.pt_6) * max(0.0, Q.M3 - 0.078)
            + 0.000846 * max(0.0, 41.1 - Q.pt_6) * max(0.0, 995.0 - Q.sum_pt)
            - 6.04 * max(0.0, 41.1 - Q.pt_6) * max(0.0, Q.z_7 - 0.0225)
            + 0.00273 * max(0.0, 601.0 - Q.sum_pt) * max(0.0, 2.19 - Q.n_dr_0p2_0p4)
            + 10.5 * max(0.0, 617.0 - Q.sum_pt) * max(0.0, 0.0507 - Q.z_3)
            + 0.119 * max(0.0, 761.0 - Q.sum_pt) * max(0.0, 0.0766 - Q.z_7)
        ))
        + 0.46875 * grid(7, max(0.0, 9.14
            - 17.9 * max(0.0, 0.0568 - Q.D2_b2)
            + 11.0 * max(0.0, 0.303 - Q.LHA)
            - 76.7 * max(0.0, Q.centroid_offset - 0.0381)
            - 242.0 * max(0.0, Q.e2 - 0.0503)
            - 78.9 * max(0.0, 0.0868 - Q.girth)
            - 407.0 * max(0.0, Q.girth2 - 0.00462)
            + 199.0 * max(0.0, Q.girth2 - 0.0135)
            - 1320.0 * max(0.0, 0.00107 - Q.girth2)
            - 3100.0 * max(0.0, 0.000261 - Q.lam2)
            + 0.0467 * max(0.0, Q.mass - 35.9)
            - 0.227 * max(0.0, Q.mass - 75.9)
            - 71.3 * max(0.0, Q.mass_over_sum_pt - 0.0113)
            + 116.0 * max(0.0, Q.mass_over_sum_pt - 0.0732)
            + 184.0 * max(0.0, Q.mass_over_sum_pt - 0.0842)
            - 283.0 * max(0.0, Q.mass_over_sum_pt - 0.0902)
            - 213.0 * max(0.0, Q.mass_over_sum_pt - 0.108)
            + 0.0709 * max(0.0, Q.mass_top5 - 53.4)
            - 14.8 * max(0.0, 0.111 - Q.max_dr)
            + 9.95 * max(0.0, 0.174 - Q.max_dr)
            - 0.0459 * max(0.0, 29.0 - Q.pt_7)
            - 21.7 * max(0.0, Q.sd_rg - 0.204)
            + 40.7 * max(0.0, Q.sd_rg - 0.279)
            + 30.6 * max(0.0, 0.158 - Q.sj2_dr)
            - 21.7 * max(0.0, 0.187 - Q.sj2_dr)
            + 19.2 * max(0.0, Q.sj3_dr_max - 0.145)
            - 19.6 * max(0.0, Q.sj3_dr_max - 0.265)
            + 32.1 * max(0.0, 0.0549 - Q.tau1)
            - 25.7 * max(0.0, 0.0413 - Q.tau21_b2)
            - 683.0 * max(0.0, Q.width - 0.00559)
            - 992.0 * max(0.0, 0.00675 - Q.width)
            + 11.7 * max(0.0, Q.z_7 - 0.0333)
            + 1.54 * max(0.0, 0.144 - Q.z_dr_0p1_0p2)
            - 19500.0 * max(0.0, 0.0204 - Q.centroid_offset) * max(0.0, 0.0042 - Q.C2_b2)
            - 15.8 * max(0.0, Q.centroid_offset - 0.0297) * max(0.0, Q.n_pt_above_50 - 3.89)
            - 28.6 * max(0.0, Q.centroid_offset - 0.0378) * max(0.0, Q.pt_4 - 81.4)
            + 2540.0 * max(0.0, 0.0203 - Q.centroid_offset) * max(0.0, 0.0272 - Q.tau21_b2)
            + 3690.0 * max(0.0, 0.0892 - Q.girth) * max(0.0, 0.00397 - Q.C2_b2)
            - 25600.0 * max(0.0, Q.girth2 - 0.0133) * max(0.0, 0.0732 - Q.dr_6)
            + 23200.0 * max(0.0, Q.girth2 - 0.0132) * max(0.0, Q.eccentricity - 0.943)
            + 881.0 * max(0.0, Q.girth2 - 0.00467) * max(0.0, 0.181 - Q.planar_flow)
            + 54.3 * max(0.0, Q.girth2 - 0.0135) * max(0.0, 39.2 - Q.pt_6)
            - 104000.0 * max(0.0, 0.00102 - Q.girth2_top2) * max(0.0, 0.0266 - Q.tau21_b2)
            + 146000.0 * max(0.0, 0.000311 - Q.lam2) * max(0.0, 0.04 - Q.tau21_b2)
            - 10.4 * max(0.0, Q.mass - 76.8) * max(0.0, Q.zdr_6 - 0.00582)
            - 0.937 * max(0.0, Q.mass_over_sum_pt - 0.0155) * max(0.0, 35.0 - Q.pt_6)
            + 17.9 * max(0.0, Q.mass_over_sum_pt - 0.0904) * max(0.0, 37.4 - Q.pt_6)
            + 0.127 * max(0.0, 0.197 - Q.planar_flow) * max(0.0, Q.sd_mass - 39.0)
            + 596.0 * max(0.0, Q.sd_rg - 0.204) * max(0.0, 0.0702 - Q.dr_6)
            - 0.825 * max(0.0, Q.sj3_dr_max - 0.132) * max(0.0, 19.3 - Q.pt_6)
            + 2730.0 * max(0.0, 0.00672 - Q.width) * max(0.0, 0.00613 - Q.mean_phi)
            - 121.0 * max(0.0, Q.width - 0.00861) * max(0.0, 38.5 - Q.pt_6)
        ))
        - 0.03125 * grid(9, max(0.0, -3.66
            - 0.422 * max(0.0, Q.D2 - 2.2)
            - 21.3 * max(0.0, 0.0235 - Q.absphi_1)
            - 318.0 * max(0.0, 0.00228 - Q.centroid_offset)
            + 198.0 * max(0.0, 0.0179 - Q.centroid_offset)
            - 60300.0 * max(0.0, 2.39e-05 - Q.e3)
            + 1220.0 * max(0.0, 0.00364 - Q.girth2)
            + 1520.0 * max(0.0, 0.00598 - Q.girth2)
            + 254.0 * max(0.0, Q.lam1 - 0.00845)
            + 527.0 * max(0.0, Q.lam2 - 0.00153)
            + 2830.0 * max(0.0, 0.000325 - Q.lam2)
            - 0.012 * max(0.0, Q.mass - 45.5)
            - 0.129 * max(0.0, 30.4 - Q.mass)
            - 0.0768 * max(0.0, 51.6 - Q.mass)
            - 75.1 * max(0.0, 0.0724 - Q.mass_over_sum_pt)
            + 27.2 * max(0.0, 0.111 - Q.max_dr)
            + 0.0448 * max(0.0, 36.2 - Q.pt_4)
            - 46.8 * max(0.0, 0.143 - Q.sj3_dr_max)
            + 22.9 * max(0.0, 0.214 - Q.sj3_dr_max)
            + 0.00493 * max(0.0, 991.0 - Q.sum_pt)
            + 43.3 * max(0.0, 0.0418 - Q.tau1)
            + 8680.0 * max(0.0, 0.000168 - Q.width)
            - 154.0 * max(0.0, 0.0064 - Q.zdr_0)
            + 5000.0 * max(0.0, 0.0185 - Q.centroid_offset) * max(0.0, Q.mean_eta - 0.000187)
            - 32.6 * max(0.0, 0.0185 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.02)
            - 2.07 * max(0.0, 0.0179 - Q.centroid_offset) * max(0.0, 164.0 - Q.pt_1)
            + 1080.0 * max(0.0, 0.0191 - Q.centroid_offset) * max(0.0, 0.205 - Q.z_2nd)
            + 3460.0 * max(0.0, 0.0552 - Q.girth) * max(0.0, 0.0289 - Q.tau21_b2)
            - 1090.0 * max(0.0, 0.0575 - Q.girth) * max(0.0, Q.z_6 - 0.0334)
            - 52500.0 * max(0.0, 0.00585 - Q.girth2) * max(0.0, Q.mean_phi - 0.0255)
            - 6850.0 * max(0.0, 0.00594 - Q.girth2) * max(0.0, 0.00379 - Q.mean_phi)
            + 333.0 * max(0.0, 0.00677 - Q.girth2) * max(0.0, 0.296 - Q.planar_flow)
            - 724.0 * max(0.0, Q.lam1 - 0.00596) * max(0.0, Q.eccentricity - 0.708)
            - 25.9 * max(0.0, Q.log_sum_pt - 6.87) * max(0.0, 0.754 - Q.D2_b2)
            + 0.0098 * max(0.0, 53.3 - Q.mass) * max(0.0, Q.D2 - 2.81)
            + 4.8 * max(0.0, 54.9 - Q.mass) * max(0.0, 0.0266 - Q.centroid_offset)
            - 0.472 * max(0.0, 54.4 - Q.mass) * max(0.0, Q.dr1_7 - 0.209)
            + 0.146 * max(0.0, 52.7 - Q.mass) * max(0.0, 6.85 - Q.log_sum_pt)
            - 1.99 * max(0.0, 0.0736 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top2 - 1.83)
            + 929.0 * max(0.0, 0.0643 - Q.mass_over_sum_pt) * max(0.0, -0.00767 - Q.phi_0)
            - 0.071 * max(0.0, Q.sd_mass - 44.8) * max(0.0, Q.n_dr_0p05_0p1 - 6.78)
            + 0.497 * max(0.0, 0.146 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.3)
            + 4710.0 * max(0.0, 0.0446 - Q.tau1) * max(0.0, Q.mean_phi - 0.0255)
            - 4250.0 * max(0.0, 0.0447 - Q.tau1) * max(0.0, 0.0309 - Q.tau21_b2)
        ))
        + 0.0546875 * grid(13, max(0.0, 0.402
            - 47.7 * max(0.0, 0.0799 - Q.e2)
            + 11000.0 * max(0.0, 5.1e-05 - Q.e3)
            + 62.6 * max(0.0, 0.149 - Q.girth)
            + 184.0 * max(0.0, 0.016 - Q.lam1)
            - 0.0171 * max(0.0, Q.mass - 50.0)
            - 0.0455 * max(0.0, Q.pt_7 - 48.6)
            - 2.74 * max(0.0, Q.sj3_dr23 - 0.2)
            - 192.0 * max(0.0, 0.0276 - Q.z_5)
            - 132.0 * max(0.0, 0.0273 - Q.z_7)
            + 3150.0 * max(0.0, 6.01e-05 - Q.e3) * max(0.0, Q.D3 - 0.207)
            - 914000.0 * max(0.0, 5.3e-05 - Q.e3) * max(0.0, 0.0378 - Q.centroid_offset)
            + 108.0 * max(0.0, 0.149 - Q.girth) * max(0.0, 0.0747 - Q.M3)
            - 9030.0 * max(0.0, 0.142 - Q.girth) * max(0.0, 0.000592 - Q.lam2)
            - 64.3 * max(0.0, 0.149 - Q.girth) * max(0.0, 6.81 - Q.log_sum_pt)
            - 0.998 * max(0.0, 0.145 - Q.girth) * max(0.0, 38.3 - Q.pt_7)
            - 1.61 * max(0.0, 0.146 - Q.girth) * max(0.0, Q.sj2_mass1 - 31.7)
            + 288.0 * max(0.0, 0.144 - Q.girth) * max(0.0, Q.tau2 - 0.00941)
            + 349.0 * max(0.0, 0.152 - Q.girth) * max(0.0, Q.z_7 - 0.0616)
            - 4.15 * max(0.0, 31.8 - Q.pt_6) * max(0.0, 0.0579 - Q.z_7)
            - 0.000761 * max(0.0, Q.sum_pt - 995.0) * max(0.0, 54.8 - Q.pt_6)
            + 0.000988 * max(0.0, Q.sum_pt_top5 - 789.0) * max(0.0, 35.9 - Q.pt_6)
            + 0.000721 * max(0.0, Q.sum_pt_top5 - 662.0) * max(0.0, 40.4 - Q.pt_7)
        ))
        + 0.375 * grid(14, max(0.0, -0.173
            - 146.0 * max(0.0, 0.00406 - Q.C2_b2)
            + 26.7 * max(0.0, Q.LHA - 0.304)
            - 71.3 * max(0.0, Q.centroid_offset - 0.0381)
            - 740.0 * max(0.0, Q.centroid_offset - 0.0498)
            - 97.6 * max(0.0, Q.e2 - 0.0167)
            + 319.0 * max(0.0, Q.e2 - 0.0501)
            - 792.0 * max(0.0, Q.e2_sq - 0.0116)
            - 17300.0 * max(0.0, 7.91e-05 - Q.e3)
            + 54.3 * max(0.0, Q.girth - 0.027)
            + 54.4 * max(0.0, Q.girth - 0.0412)
            + 58.2 * max(0.0, Q.girth - 0.0807)
            - 267.0 * max(0.0, Q.girth - 0.0873)
            - 154.0 * max(0.0, Q.girth - 0.102)
            - 1070.0 * max(0.0, Q.girth2 - 0.00754)
            + 1180.0 * max(0.0, 0.000504 - Q.lam2)
            - 0.0265 * max(0.0, Q.mass - 71.0)
            + 141.0 * max(0.0, Q.mass_over_sum_pt - 0.0798)
            + 163.0 * max(0.0, Q.mass_over_sum_pt - 0.0848)
            + 0.194 * max(0.0, 5.06 - Q.n_dr_0_0p05)
            - 13.2 * max(0.0, Q.psi_0p1 - 0.973)
            + 0.0452 * max(0.0, 49.2 - Q.sd_mass)
            - 17.1 * max(0.0, Q.sd_rg - 0.234)
            + 18.9 * max(0.0, Q.sd_rg - 0.279)
            - 9.48 * max(0.0, 0.325 - Q.sd_rg)
            - 9.74 * max(0.0, Q.sj2_dr - 0.0614)
            + 28.4 * max(0.0, Q.sj2_dr - 0.159)
            - 12.3 * max(0.0, Q.sj2_dr - 0.199)
            - 0.00528 * max(0.0, 724.0 - Q.sum_pt)
            + 324.0 * max(0.0, Q.width - 0.00348)
            - 1310.0 * max(0.0, Q.width - 0.00668)
            - 99.8 * max(0.0, Q.width - 0.0119)
            + 1.4 * max(0.0, Q.z_dr_0_0p05 - 0.152)
            - 2.28 * max(0.0, 0.163 - Q.z_dr_0p05_0p1)
            + 908000.0 * max(0.0, Q.centroid_offset - 0.0497) * max(0.0, 0.000829 - Q.C2_b2)
            - 2030.0 * max(0.0, Q.e2_sq - 0.00194) * max(0.0, 0.184 - Q.sj2_dr)
            + 153000.0 * max(0.0, 5.6e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.164)
            - 78.9 * max(0.0, Q.girth - 0.027) * max(0.0, Q.D2_b2 - 4.77)
            - 303.0 * max(0.0, 0.109 - Q.planar_flow) * max(0.0, 0.0183 - Q.centroid_offset)
            - 2710.0 * max(0.0, 0.105 - Q.planar_flow) * max(0.0, 0.0074 - Q.lam1)
            + 1070.0 * max(0.0, 0.115 - Q.planar_flow) * max(0.0, 0.0164 - Q.lam1)
            - 120.0 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, Q.mean_eta - -0.00694)
            - 0.0193 * max(0.0, 0.095 - Q.planar_flow) * max(0.0, 667.0 - Q.sum_pt_top5)
            + 37.5 * max(0.0, Q.psi_0p1 - 0.821) * max(0.0, 0.0838 - Q.D2_b2)
            - 59.5 * max(0.0, Q.sd_rg - 0.278) * max(0.0, 0.263 - Q.D2_b2)
            - 265.0 * max(0.0, Q.sj2_dr - 0.13) * max(0.0, 0.087 - Q.D2_b2)
            + 659.0 * max(0.0, Q.sj2_dr - 0.159) * max(0.0, 0.0866 - Q.D2_b2)
            - 516.0 * max(0.0, Q.sj2_dr - 0.2) * max(0.0, 0.0864 - Q.D2_b2)
            - 529.0 * max(0.0, Q.sj2_dr - 0.16) * max(0.0, 0.033 - Q.dr_2)
            + 271.0 * max(0.0, Q.sj2_dr - 0.198) * max(0.0, 0.0508 - Q.dr_2)
            - 47.0 * max(0.0, Q.sj2_dr - 0.199) * max(0.0, 0.106 - Q.z_5)
            - 321.0 * max(0.0, Q.z_dr_0p05_0p1 - 0.75) * max(0.0, 0.024 - Q.C2_b2)
            - 4.77 * max(0.0, 0.579 - Q.z_dr_0p05_0p1) * max(0.0, 0.184 - Q.D2_b2)
            + 3.61 * max(0.0, 0.589 - Q.z_dr_0p05_0p1) * max(0.0, 0.315 - Q.D3)
            + 6.91 * max(0.0, Q.z_dr_0p05_0p1 - 0.752) * max(0.0, 1.01 - Q.n_dr_0p2_0p4)
            + 0.0025 * max(0.0, 0.601 - Q.z_dr_0p05_0p1) * max(0.0, 720.0 - Q.sum_pt)
            - 3970.0 * max(0.0, Q.z_dr_0p05_0p1 - 0.754) * max(0.0, Q.zdr_5 - 0.0156)
        ))
        - 0.15625 * grid(15, max(0.0, -1.75
            - 6.37 * max(0.0, 0.228 - Q.N2)
            + 84.6 * max(0.0, 0.0416 - Q.e2)
            - 517.0 * max(0.0, 0.00834 - Q.e2_sq)
            - 81.0 * max(0.0, 0.0992 - Q.girth)
            - 698.0 * max(0.0, 0.00754 - Q.girth2)
            + 4000.0 * max(0.0, 0.000524 - Q.girth2_top2)
            + 182.0 * max(0.0, 0.00752 - Q.girth2_top2)
            + 137.0 * max(0.0, 0.00686 - Q.girth2_top3)
            - 616.0 * max(0.0, 0.00498 - Q.lam1)
            - 1990.0 * max(0.0, 0.000329 - Q.lam2)
            - 5.64 * max(0.0, 6.09 - Q.log_sum_pt)
            + 0.0563 * max(0.0, Q.mass - 45.8)
            - 0.0485 * max(0.0, Q.mass - 77.3)
            + 26.3 * max(0.0, 0.163 - Q.sj2_dr)
            - 19.8 * max(0.0, 0.2 - Q.sj2_dr)
            - 0.0189 * max(0.0, Q.sj3_pair_mass_max - 38.7)
            + 93.5 * max(0.0, 0.0622 - Q.tau1)
            - 19.5 * max(0.0, Q.z_dr_0p05_0p1 - 0.735)
            + 0.906 * max(0.0, 0.23 - Q.N2) * max(0.0, 4.02 - Q.n_dr_0p1_0p2)
            - 3.45 * max(0.0, 0.228 - Q.N2) * max(0.0, 5.9 - Q.n_for_90pct)
            + 0.0336 * max(0.0, 0.217 - Q.N2) * max(0.0, 717.0 - Q.sum_pt_top5)
            + 0.0383 * max(0.0, 0.226 - Q.N2) * max(0.0, Q.sum_pt_top5 - 426.0)
            - 7.63 * max(0.0, 0.229 - Q.N2) * max(0.0, 0.56 - Q.z_dr_0p05_0p1)
            - 441.0 * max(0.0, 0.0179 - Q.centroid_offset) * max(0.0, 0.286 - Q.dr01)
            - 15600.0 * max(0.0, 0.0199 - Q.centroid_offset) * max(0.0, 0.0123 - Q.zdr_0)
            + 100.0 * max(0.0, 0.065 - Q.e2) * max(0.0, Q.dr12 - 0.0982)
            + 411000.0 * max(0.0, 0.0106 - Q.e2_sq) * max(0.0, Q.lam2 - -1.14e-06)
            - 126.0 * max(0.0, 0.1 - Q.girth) * max(0.0, Q.dr1_3 - 0.155)
            - 1230.0 * max(0.0, 0.00758 - Q.girth2) * max(0.0, 0.771 - Q.D2)
            - 11400.0 * max(0.0, 0.00111 - Q.lam2) * max(0.0, Q.mean_phi - -0.0179)
            - 0.00426 * max(0.0, Q.mass - 46.7) * max(0.0, 23.5 - Q.pt_7)
            - 168000.0 * max(0.0, 0.0267 - Q.tau1) * max(0.0, Q.zdr_3 - 0.00824)
            + 2140.0 * max(0.0, 0.00611 - Q.width) * max(0.0, 0.713 - Q.D2)
            + 181.0 * max(0.0, 0.0182 - Q.width) * max(0.0, Q.dr1_3 - 0.0627)
            + 138000.0 * max(0.0, 0.0187 - Q.width) * max(0.0, 0.00343 - Q.lam2)
            + 4.3 * max(0.0, 0.0187 - Q.width) * max(0.0, Q.ptdr0_2 - 14.7)
            + 8.82 * max(0.0, Q.z_dr_0p05_0p1 - 0.741) * max(0.0, 2.03 - Q.n_dr_0p2_0p4)
        ))
    )


def logit_t(Q):
    return (1.34375
        + 0.015625 * grid(0, max(0.0, -0.483
            - 775.0 * max(0.0, 0.00158 - Q.C2_b2)
            - 45.5 * max(0.0, Q.centroid_offset - 0.00711)
            - 627.0 * max(0.0, Q.centroid_offset - 0.0499)
            + 98.2 * max(0.0, 0.0248 - Q.e2)
            + 78.2 * max(0.0, 0.0355 - Q.e2)
            - 52.2 * max(0.0, 0.0753 - Q.girth)
            + 292.0 * max(0.0, 0.0132 - Q.girth2)
            + 12600.0 * max(0.0, 0.000264 - Q.lam1)
            - 9140.0 * max(0.0, 7.82e-05 - Q.lam2)
            + 2.12 * max(0.0, Q.log_sum_pt - 6.36)
            - 6.38 * max(0.0, 6.11 - Q.log_sum_pt)
            - 0.0318 * max(0.0, 59.3 - Q.mass)
            - 49.0 * max(0.0, 0.0847 - Q.mass_over_sum_pt)
            + 10.9 * max(0.0, 0.15 - Q.planar_flow)
            + 10.2 * max(0.0, Q.sj3_dr_max - 0.11)
            - 26.2 * max(0.0, Q.sj3_dr_max - 0.183)
            - 0.0137 * max(0.0, Q.sum_pt - 906.0)
            - 1090.0 * max(0.0, 0.00449 - Q.width)
            + 372.0 * max(0.0, 0.00866 - Q.width)
            + 4.35 * max(0.0, Q.centroid_offset - 0.00817) * max(0.0, 6.81 - Q.n_pt_above_50)
            - 1550.0 * max(0.0, Q.centroid_offset - 0.0198) * max(0.0, 0.0628 - Q.z_7)
            + 152.0 * max(0.0, 0.0132 - Q.girth2) * max(0.0, 1.04 - Q.D2)
            + 593.0 * max(0.0, 0.0141 - Q.girth2) * max(0.0, Q.phi_0 - 0.000879)
            - 1290.0 * max(0.0, 0.00631 - Q.lam1) * max(0.0, 0.837 - Q.D2)
            + 56800.0 * max(0.0, 7.64e-05 - Q.lam2) * max(0.0, 0.266 - Q.D2_b2)
            + 61.5 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, 0.0755 - Q.dr_4)
            - 45.0 * max(0.0, Q.log_sum_pt - 6.38) * max(0.0, Q.mean_eta - 0.000559)
            - 438.0 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, Q.z_7 - 0.018)
            - 0.183 * max(0.0, 63.7 - Q.mass) * max(0.0, 0.163 - Q.D2_b2)
            - 666.0 * max(0.0, 0.149 - Q.planar_flow) * max(0.0, 0.0225 - Q.dr_2)
            - 209.0 * max(0.0, 0.274 - Q.sj3_dr_max) * max(0.0, 0.0577 - Q.z_7)
            + 11300.0 * max(0.0, 0.00432 - Q.width) * max(0.0, 0.0369 - Q.C3)
        ))
        + 0.0625 * grid(3, max(0.0, -4.33
            + 158.0 * max(0.0, Q.e2 - 0.0628)
            + 66.8 * max(0.0, Q.girth - 0.0382)
            + 89.6 * max(0.0, Q.girth - 0.077)
            - 1180.0 * max(0.0, Q.girth2 - 0.0087)
            - 157.0 * max(0.0, 0.00529 - Q.girth2)
            + 670.0 * max(0.0, Q.lam1 - 0.0081)
            + 396.0 * max(0.0, Q.lam1 - 0.016)
            - 0.0696 * max(0.0, Q.mass - 65.4)
            + 58.5 * max(0.0, Q.mass_over_sum_pt - 0.0707)
            + 11.8 * max(0.0, Q.max_dr - 0.103)
            + 41.4 * max(0.0, Q.mean_eta - 0.018)
            + 25.5 * max(0.0, -0.00445 - Q.mean_eta)
            + 24.7 * max(0.0, Q.sj2_dr - 0.188)
            - 38.7 * max(0.0, Q.tau1 - 0.053)
            - 638.0 * max(0.0, Q.width - 0.0133)
            + 7.86 * max(0.0, 0.069 - Q.z_dr_0_0p05)
            + 554.0 * max(0.0, Q.centroid_offset - 0.0107) * max(0.0, 0.0822 - Q.abseta_0)
            + 79.9 * max(0.0, Q.girth - 0.0415) * max(0.0, Q.log_sum_pt - 6.12)
            + 27.6 * max(0.0, Q.lam2 - 0.000916) * max(0.0, 56.4 - Q.pt_6)
            + 10200.0 * max(0.0, Q.lam2 - 0.00162) * max(0.0, Q.z_6 - 0.0476)
            - 1350.0 * max(0.0, Q.mass_over_sum_pt - 0.0659) * max(0.0, 0.219 - Q.sj2_dr)
            - 731.0 * max(0.0, Q.max_dr - 0.112) * max(0.0, 0.0104 - Q.zdr_1)
            - 344.0 * max(0.0, Q.sj2_dr - 0.186) * max(0.0, 0.0524 - Q.dr_3)
            - 0.459 * max(0.0, Q.sj2_dr - 0.187) * max(0.0, Q.sj2_mass1 - 5.18)
            + 369.0 * max(0.0, Q.width - 0.00678) * max(0.0, Q.sj3_pairmin_over_m - 0.0341)
        ))
        + 0.125 * grid(4, max(0.0, -4.63
            + 287.0 * max(0.0, 0.00913 - Q.C2_b2)
            + 48.2 * max(0.0, 0.221 - Q.N2)
            + 494.0 * max(0.0, 0.0119 - Q.e2_sq)
            + 335.0 * max(0.0, 0.0174 - Q.e2_sq)
            - 118000.0 * max(0.0, 1.38e-05 - Q.e3)
            - 43.9 * max(0.0, Q.girth - 0.0543)
            + 340.0 * max(0.0, Q.girth2 - 0.00353)
            + 1250.0 * max(0.0, 0.00251 - Q.girth2)
            - 908.0 * max(0.0, 0.00892 - Q.girth2)
            + 192.0 * max(0.0, 0.00812 - Q.girth2_top5)
            - 3510.0 * max(0.0, 0.000532 - Q.lam2)
            - 1550.0 * max(0.0, 0.00115 - Q.lam2)
            + 20.9 * max(0.0, Q.max_dr - 0.239)
            + 27.1 * max(0.0, Q.sj3_dr_max - 0.162)
            - 41.9 * max(0.0, Q.sj3_dr_max - 0.232)
            - 0.00445 * max(0.0, 762.0 - Q.sum_pt)
            - 0.0917 * max(0.0, 1.1 - Q.D2) * max(0.0, 51.6 - Q.pt_7)
            - 2860.0 * max(0.0, 0.222 - Q.N2) * max(0.0, Q.e2_sq - 0.0128)
            - 94.8 * max(0.0, 0.221 - Q.N2) * max(0.0, Q.eccentricity - 0.696)
            - 0.586 * max(0.0, 0.212 - Q.N2) * max(0.0, 53.1 - Q.mass)
            - 116.0 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, 0.621 - Q.z_dr_0p05_0p1)
            + 592.0 * max(0.0, 0.0112 - Q.e2_sq) * max(0.0, 1.05 - Q.D2)
            - 0.165 * max(0.0, 68.1 - Q.mass) * max(0.0, 0.985 - Q.D2)
            + 292.0 * max(0.0, Q.tau1 - 0.0711) * max(0.0, 0.208 - Q.sj3_dr_min)
        ))
        - 0.25 * grid(5, max(0.0, 0.948
            - 190.0 * max(0.0, 0.00987 - Q.C3)
            + 410.0 * max(0.0, 0.00258 - Q.girth2)
            + 3.37 * max(0.0, Q.log_sum_pt - 6.26)
            - 12.6 * max(0.0, Q.log_sum_pt - 6.77)
            - 23.1 * max(0.0, 0.0896 - Q.mass_over_sum_pt)
            - 0.0783 * max(0.0, Q.pt_7 - 22.9)
            - 0.0616 * max(0.0, 37.6 - Q.pt_7)
            - 13.4 * max(0.0, 0.109 - Q.sj3_dr_max)
            + 10.5 * max(0.0, 0.186 - Q.sj3_dr_max)
            + 0.00733 * max(0.0, Q.sum_pt_top5 - 714.0)
            + 22.0 * max(0.0, 0.0968 - Q.tau1)
            + 54.3 * max(0.0, 0.0475 - Q.z_6)
            + 165.0 * max(0.0, 0.0283 - Q.z_7)
            - 33.5 * max(0.0, 0.0239 - Q.zdr_0)
            - 52.8 * max(0.0, 0.209 - Q.LHA) * max(0.0, 6.82 - Q.log_sum_pt)
            + 103000.0 * max(0.0, 0.00172 - Q.girth2) * max(0.0, 0.0234 - Q.centroid_offset)
            - 4950.0 * max(0.0, 0.00272 - Q.girth2_top3) * max(0.0, 0.0266 - Q.phi_1)
            + 542.0 * max(0.0, Q.log_sum_pt - 6.27) * max(0.0, 0.0113 - Q.C3)
            - 1.33 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 4.61 - Q.D2_b2)
            + 232000.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 8.69e-05 - Q.mean_eta2)
            - 1750.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.00424 - Q.mean_eta2)
            - 675.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, Q.mean_phi - 0.000708)
            + 0.139 * max(0.0, Q.log_sum_pt - 6.26) * max(0.0, 7.76 - Q.ptdr0_5)
            - 0.231 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, Q.sj3_pair_mass_max - 21.9)
            + 6310.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.00175 - Q.zdr_3)
            - 9410.0 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.00263 - Q.zdr_3)
            - 0.653 * max(0.0, Q.max_pair_mass - 20.4) * max(0.0, 0.248 - Q.dr_max_012)
            - 2.56 * max(0.0, 36.1 - Q.pt_7) * max(0.0, 0.0221 - Q.C3)
            - 173.0 * max(0.0, Q.sum_pt_top5 - 720.0) * max(0.0, 8.92e-05 - Q.mean_eta2)
            + 10.1 * max(0.0, 0.104 - Q.tau1) * max(0.0, 0.464 - Q.planar_flow)
            + 1040.0 * max(0.0, 0.022 - Q.z_6) * max(0.0, 0.167 - Q.abseta_6)
            - 27400.0 * max(0.0, 0.0463 - Q.z_6) * max(0.0, 0.00185 - Q.zdr_3)
            - 912.0 * max(0.0, 0.0738 - Q.z_7) * max(0.0, 0.0304 - Q.centroid_offset)
            + 0.813 * max(0.0, 0.0497 - Q.z_7) * max(0.0, 72.9 - Q.mass_top5)
            - 1070.0 * max(0.0, 0.0237 - Q.zdr_0) * max(0.0, 0.0218 - Q.abseta_0)
        ))
        + 0.1875 * grid(8, max(0.0, -0.548
            - 213.0 * max(0.0, 0.0594 - Q.girth)
            + 922.0 * max(0.0, 0.00495 - Q.girth2)
            + 994.0 * max(0.0, 0.00351 - Q.width)
            + 40500.0 * max(0.0, 0.0584 - Q.girth) * max(0.0, Q.width - 0.000544)
            - 20500.0 * max(0.0, 0.00539 - Q.girth2) * max(0.0, Q.centroid_offset - 0.0057)
            + 20600.0 * max(0.0, 0.00643 - Q.girth2) * max(0.0, 0.0258 - Q.centroid_offset)
            + 995.0 * max(0.0, 0.00014 - Q.lam2) * max(0.0, 5.69 - Q.D2_b2)
            - 119000.0 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 3.53e-05 - Q.e3)
            - 1.85 * max(0.0, Q.log_sum_pt - 6.72) * max(0.0, Q.n_dr_0p05_0p1 - 0.999)
            - 4.83 * max(0.0, 31.2 - Q.mass) * max(0.0, 0.0297 - Q.centroid_offset)
            - 753.0 * max(0.0, 0.143 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.00799)
            - 0.00681 * max(0.0, Q.sum_pt_top5 - 670.0) * max(0.0, 8.0 - Q.n_pt_above_10)
            + 27000.0 * max(0.0, 0.0605 - Q.tau1) * max(0.0, 0.00306 - Q.width)
        ))
        + 0.375 * grid(10, max(0.0, 1.48
            - 23.1 * max(0.0, Q.LHA - 0.304)
            - 34.2 * max(0.0, 0.0255 - Q.M2)
            + 13.7 * max(0.0, 0.0229 - Q.centroid_offset)
            + 144.0 * max(0.0, Q.girth2 - 0.00732)
            - 84.3 * max(0.0, Q.girth2 - 0.0253)
            + 262.0 * max(0.0, 0.00217 - Q.girth2_top3)
            - 886.0 * max(0.0, 0.00165 - Q.lam1)
            - 618.0 * max(0.0, 0.00435 - Q.lam1)
            + 1280.0 * max(0.0, Q.lam2 - 0.000304)
            + 0.0257 * max(0.0, 76.0 - Q.mass)
            + 0.0157 * max(0.0, Q.mass_top5 - 22.8)
            - 0.0223 * max(0.0, Q.mass_top5 - 45.4)
            - 0.0238 * max(0.0, 45.0 - Q.pt_7)
            + 13.8 * max(0.0, Q.sj3_dr_min - 0.127)
            + 0.128 * max(0.0, Q.sj3_pair_mass_min - 15.2)
            - 0.00683 * max(0.0, Q.sum_pt - 986.0)
            + 7.68 * max(0.0, Q.tau1 - 0.0557)
            - 29.8 * max(0.0, 0.0208 - Q.zdr_0)
            - 85100.0 * max(0.0, 7.95e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.177)
            - 182.0 * max(0.0, Q.lam1 - 0.00686) * max(0.0, 0.383 - Q.D2_b2)
            - 351.0 * max(0.0, 0.00619 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.215)
            - 695.0 * max(0.0, Q.lam2 - 0.00023) * max(0.0, Q.planar_flow - 0.0514)
            - 24.1 * max(0.0, Q.lam2 - 0.000412) * max(0.0, 31.9 - Q.pt_6)
            + 0.0757 * max(0.0, 45.6 - Q.pt_7) * max(0.0, 1.0 - Q.D2)
            - 0.0706 * max(0.0, 44.7 - Q.pt_7) * max(0.0, 6.54 - Q.log_sum_pt)
            - 22.2 * max(0.0, Q.sj3_dr_min - 0.134) * max(0.0, 0.43 - Q.z_dr_0_0p05)
            - 0.26 * max(0.0, Q.sj3_pair_mass_min - 9.6) * max(0.0, Q.sj3_pairmin_over_m - 0.292)
            + 14.4 * max(0.0, Q.tau1 - 0.0451) * max(0.0, 0.973 - Q.D2)
            + 1680.0 * max(0.0, 0.0209 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.0107)
            + 175.0 * max(0.0, 0.022 - Q.zdr_0) * max(0.0, Q.dr_7 - 0.0771)
            + 134.0 * max(0.0, 0.0213 - Q.zdr_0) * max(0.0, 0.26 - Q.z_dr_0p05_0p1)
        ))
        - 0.5 * grid(12, max(0.0, -0.443
            + 352.0 * max(0.0, Q.girth2 - 0.0188)
            - 29.6 * max(0.0, Q.girth2_top2 - 0.014)
            + 0.0667 * max(0.0, Q.mass - 88.2)
            - 53.1 * max(0.0, Q.mass_over_sum_pt - 0.131)
            - 24.9 * max(0.0, Q.zdr_0 - 0.0398)
            + 185.0 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.abseta_6 - 0.0284)
            + 5340.0 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.lam2 - 0.000537)
            - 2.84 * max(0.0, Q.girth2 - 0.0188) * max(0.0, 53.4 - Q.pt_7)
        ))
        - 0.40625 * grid(13, max(0.0, 0.402
            - 47.7 * max(0.0, 0.0799 - Q.e2)
            + 11000.0 * max(0.0, 5.1e-05 - Q.e3)
            + 62.6 * max(0.0, 0.149 - Q.girth)
            + 184.0 * max(0.0, 0.016 - Q.lam1)
            - 0.0171 * max(0.0, Q.mass - 50.0)
            - 0.0455 * max(0.0, Q.pt_7 - 48.6)
            - 2.74 * max(0.0, Q.sj3_dr23 - 0.2)
            - 192.0 * max(0.0, 0.0276 - Q.z_5)
            - 132.0 * max(0.0, 0.0273 - Q.z_7)
            + 3150.0 * max(0.0, 6.01e-05 - Q.e3) * max(0.0, Q.D3 - 0.207)
            - 914000.0 * max(0.0, 5.3e-05 - Q.e3) * max(0.0, 0.0378 - Q.centroid_offset)
            + 108.0 * max(0.0, 0.149 - Q.girth) * max(0.0, 0.0747 - Q.M3)
            - 9030.0 * max(0.0, 0.142 - Q.girth) * max(0.0, 0.000592 - Q.lam2)
            - 64.3 * max(0.0, 0.149 - Q.girth) * max(0.0, 6.81 - Q.log_sum_pt)
            - 0.998 * max(0.0, 0.145 - Q.girth) * max(0.0, 38.3 - Q.pt_7)
            - 1.61 * max(0.0, 0.146 - Q.girth) * max(0.0, Q.sj2_mass1 - 31.7)
            + 288.0 * max(0.0, 0.144 - Q.girth) * max(0.0, Q.tau2 - 0.00941)
            + 349.0 * max(0.0, 0.152 - Q.girth) * max(0.0, Q.z_7 - 0.0616)
            - 4.15 * max(0.0, 31.8 - Q.pt_6) * max(0.0, 0.0579 - Q.z_7)
            - 0.000761 * max(0.0, Q.sum_pt - 995.0) * max(0.0, 54.8 - Q.pt_6)
            + 0.000988 * max(0.0, Q.sum_pt_top5 - 789.0) * max(0.0, 35.9 - Q.pt_6)
            + 0.000721 * max(0.0, Q.sum_pt_top5 - 662.0) * max(0.0, 40.4 - Q.pt_7)
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
