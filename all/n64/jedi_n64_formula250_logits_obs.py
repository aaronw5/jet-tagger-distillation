"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network (step 4; all observables), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.2% (the network: 81.1%); same class as the network for 93.7% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.pt2_over_pt0           pT2 / pT0
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
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
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.pt_dispersion          √(Σ pTᵢ²) / Σ pTᵢ
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
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top20=mass_of(20),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_90pct=ncum(0.9),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft7_z=softp(7, 'z'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        z_2nd=zs[1],
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
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
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-1.078125
        + 0.625 * grid(1, max(0.0, 1.06
            + 10.9 * max(0.0, 0.258 - Q.LHA)
            - 16.4 * max(0.0, 0.032 - Q.M3)
            + 638.0 * max(0.0, 0.000986 - Q.girth2_top15)
            + 166.0 * max(0.0, 0.00509 - Q.lam1)
            + 19.4 * max(0.0, Q.log_sum_pt - 6.89)
            + 30.0 * max(0.0, Q.log_sum_pt - 6.91)
            - 23.2 * max(0.0, Q.log_sum_pt - 6.97)
            - 10.4 * max(0.0, Q.log_sum_pt - 7.07)
            - 0.0133 * max(0.0, 122.0 - Q.mass)
            - 0.0271 * max(0.0, 49.0 - Q.mass_top20)
            - 0.0558 * max(0.0, 8.0 - Q.n_dr_0p1_0p2)
            - 0.0777 * max(0.0, 6.57 - Q.n_dr_0p2_0p4)
            - 0.0714 * max(0.0, Q.n_for_90pct - 10.4)
            + 0.0527 * max(0.0, Q.n_particles - 38.4)
            - 164.0 * max(0.0, Q.psi_0p3 - 0.998)
            - 0.0177 * max(0.0, 31.6 - Q.pt_9)
            + 1.69 * max(0.0, Q.pt_entropy - 2.0)
            + 1.26 * max(0.0, 1.5 - Q.soft1_pt)
            - 1.3 * max(0.0, 2.27 - Q.soft1_pt)
            + 0.0035 * max(0.0, Q.sum_pt_top30 - 1190.0)
            + 0.00749 * max(0.0, 1070.0 - Q.sum_pt_top40)
            - 0.0114 * max(0.0, Q.sum_pt_top50 - 953.0)
            + 12.0 * max(0.0, 0.196 - Q.tau1)
            - 6.1 * max(0.0, Q.z_dr_0_0p05 - 0.875)
            - 25.4 * max(0.0, Q.z_top50_slots - 0.96)
            + 0.00282 * max(0.0, 50.6 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.8)
            + 0.305 * max(0.0, Q.n_particles - 39.4) * max(0.0, 0.111 - Q.dr_0)
            - 0.00204 * max(0.0, 29.5 - Q.sj3_mass1) * max(0.0, 21.0 - Q.sj3_mass2)
            - 175.0 * max(0.0, Q.z_top20_slots - 0.887) * max(0.0, 0.0414 - Q.dr_2)
            + 200.0 * max(0.0, Q.z_top30_slots - 0.939) * max(0.0, 0.075 - Q.C2)
            + 0.565 * max(0.0, Q.z_top30_slots - 0.942) * max(0.0, Q.max_pair_mass - 9.45)
            + 0.582 * max(0.0, Q.z_top30_slots - 0.914) * max(0.0, Q.ptdr0_3 - 6.21)
        ))
        - 0.75 * grid(3, max(0.0, -0.0649
            + 79.4 * max(0.0, 0.00669 - Q.girth2_top40)
            + 1150.0 * max(0.0, 0.000625 - Q.lam2)
            - 0.0241 * max(0.0, 69.2 - Q.mass_top50)
            + 0.0306 * max(0.0, 46.1 - Q.n_particles)
            - 2.0 * max(0.0, 0.375 - Q.tau21)
            + 0.592 * max(0.0, 11.2 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.934)
            + 3.7 * max(0.0, 7.15 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.98)
            - 214.0 * max(0.0, 0.406 - Q.tau21) * max(0.0, Q.lam1 - 0.00572)
        ))
        - 0.875 * grid(4, max(0.0, 1.11
            - 2.55 * max(0.0, Q.C2 - 0.0725)
            - 23.2 * max(0.0, Q.e2 - 0.0568)
            + 173.0 * max(0.0, Q.e2_sq - 0.0103)
            - 15.4 * max(0.0, Q.girth - 0.121)
            - 71.7 * max(0.0, Q.girth2_top15 - 0.00752)
            + 78.5 * max(0.0, Q.girth2_top15 - 0.016)
            - 93.2 * max(0.0, 0.00531 - Q.girth2_top15)
            - 70.8 * max(0.0, Q.lam1 - 0.00794)
            - 0.0511 * max(0.0, 78.4 - Q.mass)
            - 0.0818 * max(0.0, 86.2 - Q.mass)
            + 0.0279 * max(0.0, 119.0 - Q.mass)
            + 0.061 * max(0.0, 67.5 - Q.mass_top40)
            - 0.0135 * max(0.0, 166.0 - Q.mass_top40)
            + 0.0309 * max(0.0, 24.2 - Q.n_dr_0p2_0p4)
            - 0.0255 * max(0.0, Q.n_particles - 23.1)
            + 182.0 * max(0.0, Q.psi_0p3 - 0.997)
            + 0.00874 * max(0.0, 101.0 - Q.mass) * max(0.0, 3.79 - Q.D2_b2)
            - 0.0186 * max(0.0, 88.5 - Q.mass_top40) * max(0.0, 3.21 - Q.D2_b2)
            - 0.00112 * max(0.0, 22.7 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 12.1)
            + 0.000656 * max(0.0, Q.n_particles - 27.6) * max(0.0, Q.n_dr_0_0p05 - 10.7)
            + 0.00517 * max(0.0, Q.n_particles - 20.4) * max(0.0, 2.44 - Q.soft1_pt)
            - 11.5 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 13.2 - Q.n_dr_0_0p05)
            - 0.194 * max(0.0, Q.sj2_dr - 0.232) * max(0.0, 20.6 - Q.D2_b2)
        ))
        - 0.3125 * grid(5, max(0.0, -0.375
            - 108.0 * max(0.0, Q.girth2_top50 - 0.0191)
            - 23.0 * max(0.0, Q.log_sum_pt - 6.91)
            + 4.88 * max(0.0, Q.log_sum_pt - 6.99)
            + 0.0337 * max(0.0, Q.mass - 73.6)
            - 0.022 * max(0.0, Q.mass - 92.4)
            - 0.177 * max(0.0, Q.mass - 171.0)
            - 23.0 * max(0.0, Q.mass_over_sum_pt - 0.0781)
            - 0.277 * max(0.0, Q.mass_top50 - 158.0)
            + 2.32 * max(0.0, 0.459 - Q.max_dr)
            - 0.0129 * max(0.0, 22.6 - Q.n_dr_0p1_0p2)
            + 0.0302 * max(0.0, 12.4 - Q.n_dr_0p2_0p4)
            + 0.0303 * max(0.0, 47.5 - Q.n_particles)
            + 0.0286 * max(0.0, Q.sd_mass - 72.1)
            - 0.0345 * max(0.0, Q.sd_mass - 86.4)
            + 0.0107 * max(0.0, Q.sum_pt - 908.0)
            - 0.0157 * max(0.0, Q.sum_pt - 990.0)
            + 0.0172 * max(0.0, Q.sum_pt_top50 - 943.0)
            - 0.695 * max(0.0, 0.561 - Q.tau21)
            - 0.0157 * max(0.0, 56.1 - Q.n_particles) * max(0.0, 2.7 - Q.D2)
        ))
        + 0.15625 * grid(6, max(0.0, 0.882
            - 31.7 * max(0.0, Q.e2 - 0.0294)
            + 69.1 * max(0.0, Q.e2 - 0.053)
            + 198.0 * max(0.0, Q.girth2_top20 - 0.00733)
            - 223.0 * max(0.0, Q.girth2_top30 - 0.0275)
            + 0.0999 * max(0.0, 93.9 - Q.mass)
            - 0.105 * max(0.0, 102.0 - Q.mass)
            - 61.0 * max(0.0, Q.mass_over_sum_pt - 0.121)
            + 0.0307 * max(0.0, 73.3 - Q.mass_top50)
            - 0.119 * max(0.0, Q.sj3_pair_mass_min - 77.9)
            + 15.2 * max(0.0, 0.0533 - Q.tau1)
            + 11.9 * max(0.0, 0.931 - Q.z_top40_slots)
            - 2330.0 * max(0.0, Q.girth2_top20 - 0.0035) * max(0.0, Q.C2_b2 - 0.00379)
            + 8.43e-05 * max(0.0, 102.0 - Q.mass) * max(0.0, 1010.0 - Q.sum_pt)
        ))
        + 0.03125 * grid(8, max(0.0, 1.3
            - 202.0 * max(0.0, Q.girth2_top20 - 0.0059)
            + 268.0 * max(0.0, Q.girth2_top20 - 0.00788)
            + 165.0 * max(0.0, 0.00754 - Q.girth2_top30)
            + 0.0435 * max(0.0, Q.mass - 71.1)
            - 0.0408 * max(0.0, Q.mass - 103.0)
            - 0.0209 * max(0.0, Q.mass - 127.0)
            + 0.047 * max(0.0, 8.6 - Q.n_dr_0p2_0p4)
            - 0.0677 * max(0.0, 17.9 - Q.n_dr_0p2_0p4)
            + 4.23 * max(0.0, Q.sj2_dr - 0.23)
            + 0.0195 * max(0.0, 1030.0 - Q.sum_pt)
            + 0.0014 * max(0.0, Q.sum_pt_top30 - 987.0)
            - 0.00795 * max(0.0, 1030.0 - Q.sum_pt_top40)
            - 214.0 * max(0.0, 0.00987 - Q.width)
            - 35.8 * max(0.0, Q.z_top50_slots - 0.972)
            - 810.0 * max(0.0, Q.girth2_top40 - 0.00555) * max(0.0, 6.99 - Q.log_sum_pt)
            + 0.305 * max(0.0, Q.mass_over_sum_pt - 0.0785) * max(0.0, 1100.0 - Q.sum_pt)
            + 0.207 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.313)
            + 0.403 * max(0.0, 0.0848 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.4)
        ))
        + 0.515625 * grid(9, max(0.0, -1.13
            + 23.3 * max(0.0, Q.girth - 0.0976)
            + 281.0 * max(0.0, 0.00617 - Q.girth2_top40)
            - 0.07 * max(0.0, 62.9 - Q.mass)
            + 0.103 * max(0.0, 85.5 - Q.mass)
            - 0.0621 * max(0.0, 147.0 - Q.mass)
            + 0.0491 * max(0.0, 178.0 - Q.mass)
            + 0.00819 * max(0.0, 989.0 - Q.sum_pt)
            - 8.68 * max(0.0, Q.z_top40_slots - 0.954)
            + 0.00132 * max(0.0, 116.0 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 8.57)
            - 0.000231 * max(0.0, 97.9 - Q.mass_top40) * max(0.0, 1020.0 - Q.sum_pt)
            + 0.000185 * max(0.0, 100.0 - Q.mass_top40) * max(0.0, 961.0 - Q.sum_pt_top40)
        ))
        - 0.015625 * grid(10, max(0.0, 0.309
            + 0.473 * max(0.0, 3.04 - Q.D2)
            + 7.76 * max(0.0, 0.0664 - Q.dr_0)
            + 29.9 * max(0.0, Q.e2 - 0.0128)
            - 236.0 * max(0.0, 0.00533 - Q.girth2_top20)
            + 0.0141 * max(0.0, Q.mass - 78.8)
            - 0.0869 * max(0.0, Q.mass - 145.0)
            - 0.0481 * max(0.0, Q.mass - 164.0)
            + 0.0998 * max(0.0, 87.9 - Q.mass)
            - 0.0326 * max(0.0, 99.6 - Q.mass)
            - 31.1 * max(0.0, Q.mass_over_sum_pt - 0.16)
            + 0.0112 * max(0.0, Q.mass_top5 - 17.5)
            + 0.0434 * max(0.0, Q.mass_top50 - 138.0)
            - 0.055 * max(0.0, 11.1 - Q.n_dr_0p2_0p4)
            - 2.54 * max(0.0, Q.pt_dispersion - 0.27)
            - 0.0219 * max(0.0, 65.0 - Q.sj2_mass1)
            - 0.0107 * max(0.0, 984.0 - Q.sum_pt)
            - 0.0034 * max(0.0, Q.sum_pt_top10 - 941.0)
            + 0.00394 * max(0.0, 941.0 - Q.sum_pt_top15)
            - 82.7 * max(0.0, 0.047 - Q.tau1)
            + 14.2 * max(0.0, 0.0485 - Q.tau2)
            + 4.97 * max(0.0, 0.0859 - Q.z_dr_0p2_0p4)
            - 3.21 * max(0.0, 2.99 - Q.D2) * max(0.0, Q.sj2_dr - 0.192)
            - 17300.0 * max(0.0, 0.0234 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.998)
            - 7.19 * max(0.0, Q.mass_top50 - 133.0) * max(0.0, Q.soft5_z - 0.00163)
            + 0.000322 * max(0.0, 63.3 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 7.52)
            - 0.877 * max(0.0, Q.sum_pt_top10 - 943.0) * max(0.0, -0.078 - Q.eta_0)
        ))
        + 0.234375 * grid(12, max(0.0, -0.151
            - 172.0 * max(0.0, 0.00586 - Q.girth2_top15)
            - 0.0956 * max(0.0, 59.7 - Q.mass)
            + 0.119 * max(0.0, 83.4 - Q.mass)
            + 0.0144 * max(0.0, Q.sd_mass - 124.0)
            - 3400.0 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0036 - Q.lam2)
            - 3.47 * max(0.0, 90.0 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            + 0.000437 * max(0.0, Q.sj3_pair_mass_max - 123.0) * max(0.0, 65.0 - Q.sj3_pair_mass_min)
        ))
    )


def logit_q(Q):
    return (1.359375
        - 0.1875 * grid(1, max(0.0, 1.06
            + 10.9 * max(0.0, 0.258 - Q.LHA)
            - 16.4 * max(0.0, 0.032 - Q.M3)
            + 638.0 * max(0.0, 0.000986 - Q.girth2_top15)
            + 166.0 * max(0.0, 0.00509 - Q.lam1)
            + 19.4 * max(0.0, Q.log_sum_pt - 6.89)
            + 30.0 * max(0.0, Q.log_sum_pt - 6.91)
            - 23.2 * max(0.0, Q.log_sum_pt - 6.97)
            - 10.4 * max(0.0, Q.log_sum_pt - 7.07)
            - 0.0133 * max(0.0, 122.0 - Q.mass)
            - 0.0271 * max(0.0, 49.0 - Q.mass_top20)
            - 0.0558 * max(0.0, 8.0 - Q.n_dr_0p1_0p2)
            - 0.0777 * max(0.0, 6.57 - Q.n_dr_0p2_0p4)
            - 0.0714 * max(0.0, Q.n_for_90pct - 10.4)
            + 0.0527 * max(0.0, Q.n_particles - 38.4)
            - 164.0 * max(0.0, Q.psi_0p3 - 0.998)
            - 0.0177 * max(0.0, 31.6 - Q.pt_9)
            + 1.69 * max(0.0, Q.pt_entropy - 2.0)
            + 1.26 * max(0.0, 1.5 - Q.soft1_pt)
            - 1.3 * max(0.0, 2.27 - Q.soft1_pt)
            + 0.0035 * max(0.0, Q.sum_pt_top30 - 1190.0)
            + 0.00749 * max(0.0, 1070.0 - Q.sum_pt_top40)
            - 0.0114 * max(0.0, Q.sum_pt_top50 - 953.0)
            + 12.0 * max(0.0, 0.196 - Q.tau1)
            - 6.1 * max(0.0, Q.z_dr_0_0p05 - 0.875)
            - 25.4 * max(0.0, Q.z_top50_slots - 0.96)
            + 0.00282 * max(0.0, 50.6 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.8)
            + 0.305 * max(0.0, Q.n_particles - 39.4) * max(0.0, 0.111 - Q.dr_0)
            - 0.00204 * max(0.0, 29.5 - Q.sj3_mass1) * max(0.0, 21.0 - Q.sj3_mass2)
            - 175.0 * max(0.0, Q.z_top20_slots - 0.887) * max(0.0, 0.0414 - Q.dr_2)
            + 200.0 * max(0.0, Q.z_top30_slots - 0.939) * max(0.0, 0.075 - Q.C2)
            + 0.565 * max(0.0, Q.z_top30_slots - 0.942) * max(0.0, Q.max_pair_mass - 9.45)
            + 0.582 * max(0.0, Q.z_top30_slots - 0.914) * max(0.0, Q.ptdr0_3 - 6.21)
        ))
        + 0.125 * grid(2, max(0.0, 0.00944
            + 206.0 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, 0.0272 - Q.girth2_top15)
        ))
        - 1.0625 * grid(4, max(0.0, 1.11
            - 2.55 * max(0.0, Q.C2 - 0.0725)
            - 23.2 * max(0.0, Q.e2 - 0.0568)
            + 173.0 * max(0.0, Q.e2_sq - 0.0103)
            - 15.4 * max(0.0, Q.girth - 0.121)
            - 71.7 * max(0.0, Q.girth2_top15 - 0.00752)
            + 78.5 * max(0.0, Q.girth2_top15 - 0.016)
            - 93.2 * max(0.0, 0.00531 - Q.girth2_top15)
            - 70.8 * max(0.0, Q.lam1 - 0.00794)
            - 0.0511 * max(0.0, 78.4 - Q.mass)
            - 0.0818 * max(0.0, 86.2 - Q.mass)
            + 0.0279 * max(0.0, 119.0 - Q.mass)
            + 0.061 * max(0.0, 67.5 - Q.mass_top40)
            - 0.0135 * max(0.0, 166.0 - Q.mass_top40)
            + 0.0309 * max(0.0, 24.2 - Q.n_dr_0p2_0p4)
            - 0.0255 * max(0.0, Q.n_particles - 23.1)
            + 182.0 * max(0.0, Q.psi_0p3 - 0.997)
            + 0.00874 * max(0.0, 101.0 - Q.mass) * max(0.0, 3.79 - Q.D2_b2)
            - 0.0186 * max(0.0, 88.5 - Q.mass_top40) * max(0.0, 3.21 - Q.D2_b2)
            - 0.00112 * max(0.0, 22.7 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 12.1)
            + 0.000656 * max(0.0, Q.n_particles - 27.6) * max(0.0, Q.n_dr_0_0p05 - 10.7)
            + 0.00517 * max(0.0, Q.n_particles - 20.4) * max(0.0, 2.44 - Q.soft1_pt)
            - 11.5 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 13.2 - Q.n_dr_0_0p05)
            - 0.194 * max(0.0, Q.sj2_dr - 0.232) * max(0.0, 20.6 - Q.D2_b2)
        ))
        + 0.21875 * grid(6, max(0.0, 0.882
            - 31.7 * max(0.0, Q.e2 - 0.0294)
            + 69.1 * max(0.0, Q.e2 - 0.053)
            + 198.0 * max(0.0, Q.girth2_top20 - 0.00733)
            - 223.0 * max(0.0, Q.girth2_top30 - 0.0275)
            + 0.0999 * max(0.0, 93.9 - Q.mass)
            - 0.105 * max(0.0, 102.0 - Q.mass)
            - 61.0 * max(0.0, Q.mass_over_sum_pt - 0.121)
            + 0.0307 * max(0.0, 73.3 - Q.mass_top50)
            - 0.119 * max(0.0, Q.sj3_pair_mass_min - 77.9)
            + 15.2 * max(0.0, 0.0533 - Q.tau1)
            + 11.9 * max(0.0, 0.931 - Q.z_top40_slots)
            - 2330.0 * max(0.0, Q.girth2_top20 - 0.0035) * max(0.0, Q.C2_b2 - 0.00379)
            + 8.43e-05 * max(0.0, 102.0 - Q.mass) * max(0.0, 1010.0 - Q.sum_pt)
        ))
        + 0.015625 * grid(8, max(0.0, 1.3
            - 202.0 * max(0.0, Q.girth2_top20 - 0.0059)
            + 268.0 * max(0.0, Q.girth2_top20 - 0.00788)
            + 165.0 * max(0.0, 0.00754 - Q.girth2_top30)
            + 0.0435 * max(0.0, Q.mass - 71.1)
            - 0.0408 * max(0.0, Q.mass - 103.0)
            - 0.0209 * max(0.0, Q.mass - 127.0)
            + 0.047 * max(0.0, 8.6 - Q.n_dr_0p2_0p4)
            - 0.0677 * max(0.0, 17.9 - Q.n_dr_0p2_0p4)
            + 4.23 * max(0.0, Q.sj2_dr - 0.23)
            + 0.0195 * max(0.0, 1030.0 - Q.sum_pt)
            + 0.0014 * max(0.0, Q.sum_pt_top30 - 987.0)
            - 0.00795 * max(0.0, 1030.0 - Q.sum_pt_top40)
            - 214.0 * max(0.0, 0.00987 - Q.width)
            - 35.8 * max(0.0, Q.z_top50_slots - 0.972)
            - 810.0 * max(0.0, Q.girth2_top40 - 0.00555) * max(0.0, 6.99 - Q.log_sum_pt)
            + 0.305 * max(0.0, Q.mass_over_sum_pt - 0.0785) * max(0.0, 1100.0 - Q.sum_pt)
            + 0.207 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.313)
            + 0.403 * max(0.0, 0.0848 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.4)
        ))
        + 0.5625 * grid(9, max(0.0, -1.13
            + 23.3 * max(0.0, Q.girth - 0.0976)
            + 281.0 * max(0.0, 0.00617 - Q.girth2_top40)
            - 0.07 * max(0.0, 62.9 - Q.mass)
            + 0.103 * max(0.0, 85.5 - Q.mass)
            - 0.0621 * max(0.0, 147.0 - Q.mass)
            + 0.0491 * max(0.0, 178.0 - Q.mass)
            + 0.00819 * max(0.0, 989.0 - Q.sum_pt)
            - 8.68 * max(0.0, Q.z_top40_slots - 0.954)
            + 0.00132 * max(0.0, 116.0 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 8.57)
            - 0.000231 * max(0.0, 97.9 - Q.mass_top40) * max(0.0, 1020.0 - Q.sum_pt)
            + 0.000185 * max(0.0, 100.0 - Q.mass_top40) * max(0.0, 961.0 - Q.sum_pt_top40)
        ))
        - 0.015625 * grid(10, max(0.0, 0.309
            + 0.473 * max(0.0, 3.04 - Q.D2)
            + 7.76 * max(0.0, 0.0664 - Q.dr_0)
            + 29.9 * max(0.0, Q.e2 - 0.0128)
            - 236.0 * max(0.0, 0.00533 - Q.girth2_top20)
            + 0.0141 * max(0.0, Q.mass - 78.8)
            - 0.0869 * max(0.0, Q.mass - 145.0)
            - 0.0481 * max(0.0, Q.mass - 164.0)
            + 0.0998 * max(0.0, 87.9 - Q.mass)
            - 0.0326 * max(0.0, 99.6 - Q.mass)
            - 31.1 * max(0.0, Q.mass_over_sum_pt - 0.16)
            + 0.0112 * max(0.0, Q.mass_top5 - 17.5)
            + 0.0434 * max(0.0, Q.mass_top50 - 138.0)
            - 0.055 * max(0.0, 11.1 - Q.n_dr_0p2_0p4)
            - 2.54 * max(0.0, Q.pt_dispersion - 0.27)
            - 0.0219 * max(0.0, 65.0 - Q.sj2_mass1)
            - 0.0107 * max(0.0, 984.0 - Q.sum_pt)
            - 0.0034 * max(0.0, Q.sum_pt_top10 - 941.0)
            + 0.00394 * max(0.0, 941.0 - Q.sum_pt_top15)
            - 82.7 * max(0.0, 0.047 - Q.tau1)
            + 14.2 * max(0.0, 0.0485 - Q.tau2)
            + 4.97 * max(0.0, 0.0859 - Q.z_dr_0p2_0p4)
            - 3.21 * max(0.0, 2.99 - Q.D2) * max(0.0, Q.sj2_dr - 0.192)
            - 17300.0 * max(0.0, 0.0234 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.998)
            - 7.19 * max(0.0, Q.mass_top50 - 133.0) * max(0.0, Q.soft5_z - 0.00163)
            + 0.000322 * max(0.0, 63.3 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 7.52)
            - 0.877 * max(0.0, Q.sum_pt_top10 - 943.0) * max(0.0, -0.078 - Q.eta_0)
        ))
        - 0.25 * grid(11, max(0.0, -0.575
            + 96.2 * max(0.0, 0.0266 - Q.e2)
            - 402.0 * max(0.0, 0.00666 - Q.girth2_top30)
            - 0.0787 * max(0.0, 80.9 - Q.mass_top50)
            + 0.0866 * max(0.0, 94.7 - Q.mass_top50)
            + 0.0549 * max(0.0, 16.7 - Q.n_dr_0p1_0p2)
            + 0.184 * max(0.0, 7.5 - Q.n_dr_0p2_0p4)
            + 64.2 * max(0.0, Q.psi_0p3 - 0.99)
            - 1.56 * max(0.0, Q.z_top5_slots - 0.552)
            - 2.65e-05 * max(0.0, 119.0 - Q.mass) * max(0.0, 872.0 - Q.sum_pt_top10)
            - 28.5 * max(0.0, 9.67 - Q.n_dr_0p2_0p4) * max(0.0, 0.0062 - Q.girth2)
            - 0.0051 * max(0.0, 8.65 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 36.1)
        ))
        + 0.34375 * grid(12, max(0.0, -0.151
            - 172.0 * max(0.0, 0.00586 - Q.girth2_top15)
            - 0.0956 * max(0.0, 59.7 - Q.mass)
            + 0.119 * max(0.0, 83.4 - Q.mass)
            + 0.0144 * max(0.0, Q.sd_mass - 124.0)
            - 3400.0 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0036 - Q.lam2)
            - 3.47 * max(0.0, 90.0 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            + 0.000437 * max(0.0, Q.sj3_pair_mass_max - 123.0) * max(0.0, 65.0 - Q.sj3_pair_mass_min)
        ))
    )


def logit_W(Q):
    return (0.09375
        + 0.75 * grid(0, max(0.0, 0.798
            - 315.0 * max(0.0, 0.00614 - Q.e2_sq)
            - 73.9 * max(0.0, 0.00652 - Q.girth2_top20)
            - 0.156 * max(0.0, Q.mass - 78.3)
            - 0.0352 * max(0.0, Q.mass - 93.8)
            - 0.0161 * max(0.0, 99.1 - Q.mass)
            + 501.0 * max(0.0, 0.00766 - Q.mass_over_sum_pt_sq)
            + 0.0474 * max(0.0, Q.mass_top50 - 82.0)
            + 0.0494 * max(0.0, 15.3 - Q.n_dr_0p2_0p4)
            + 55.8 * max(0.0, Q.psi_0p3 - 0.996)
            - 0.00875 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.00302 * max(0.0, 1160.0 - Q.sum_pt_top50)
            - 18.4 * max(0.0, 0.0699 - Q.tau1)
            - 4.77 * max(0.0, 0.0913 - Q.z_dr_0p2_0p4)
            - 22.1 * max(0.0, 0.991 - Q.z_top50_slots)
            + 0.0566 * max(0.0, 7.0 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 963.0)
        ))
        + 0.4375 * grid(3, max(0.0, -0.0649
            + 79.4 * max(0.0, 0.00669 - Q.girth2_top40)
            + 1150.0 * max(0.0, 0.000625 - Q.lam2)
            - 0.0241 * max(0.0, 69.2 - Q.mass_top50)
            + 0.0306 * max(0.0, 46.1 - Q.n_particles)
            - 2.0 * max(0.0, 0.375 - Q.tau21)
            + 0.592 * max(0.0, 11.2 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.934)
            + 3.7 * max(0.0, 7.15 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.98)
            - 214.0 * max(0.0, 0.406 - Q.tau21) * max(0.0, Q.lam1 - 0.00572)
        ))
        + 0.34375 * grid(4, max(0.0, 1.11
            - 2.55 * max(0.0, Q.C2 - 0.0725)
            - 23.2 * max(0.0, Q.e2 - 0.0568)
            + 173.0 * max(0.0, Q.e2_sq - 0.0103)
            - 15.4 * max(0.0, Q.girth - 0.121)
            - 71.7 * max(0.0, Q.girth2_top15 - 0.00752)
            + 78.5 * max(0.0, Q.girth2_top15 - 0.016)
            - 93.2 * max(0.0, 0.00531 - Q.girth2_top15)
            - 70.8 * max(0.0, Q.lam1 - 0.00794)
            - 0.0511 * max(0.0, 78.4 - Q.mass)
            - 0.0818 * max(0.0, 86.2 - Q.mass)
            + 0.0279 * max(0.0, 119.0 - Q.mass)
            + 0.061 * max(0.0, 67.5 - Q.mass_top40)
            - 0.0135 * max(0.0, 166.0 - Q.mass_top40)
            + 0.0309 * max(0.0, 24.2 - Q.n_dr_0p2_0p4)
            - 0.0255 * max(0.0, Q.n_particles - 23.1)
            + 182.0 * max(0.0, Q.psi_0p3 - 0.997)
            + 0.00874 * max(0.0, 101.0 - Q.mass) * max(0.0, 3.79 - Q.D2_b2)
            - 0.0186 * max(0.0, 88.5 - Q.mass_top40) * max(0.0, 3.21 - Q.D2_b2)
            - 0.00112 * max(0.0, 22.7 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 12.1)
            + 0.000656 * max(0.0, Q.n_particles - 27.6) * max(0.0, Q.n_dr_0_0p05 - 10.7)
            + 0.00517 * max(0.0, Q.n_particles - 20.4) * max(0.0, 2.44 - Q.soft1_pt)
            - 11.5 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 13.2 - Q.n_dr_0_0p05)
            - 0.194 * max(0.0, Q.sj2_dr - 0.232) * max(0.0, 20.6 - Q.D2_b2)
        ))
        + 0.578125 * grid(5, max(0.0, -0.375
            - 108.0 * max(0.0, Q.girth2_top50 - 0.0191)
            - 23.0 * max(0.0, Q.log_sum_pt - 6.91)
            + 4.88 * max(0.0, Q.log_sum_pt - 6.99)
            + 0.0337 * max(0.0, Q.mass - 73.6)
            - 0.022 * max(0.0, Q.mass - 92.4)
            - 0.177 * max(0.0, Q.mass - 171.0)
            - 23.0 * max(0.0, Q.mass_over_sum_pt - 0.0781)
            - 0.277 * max(0.0, Q.mass_top50 - 158.0)
            + 2.32 * max(0.0, 0.459 - Q.max_dr)
            - 0.0129 * max(0.0, 22.6 - Q.n_dr_0p1_0p2)
            + 0.0302 * max(0.0, 12.4 - Q.n_dr_0p2_0p4)
            + 0.0303 * max(0.0, 47.5 - Q.n_particles)
            + 0.0286 * max(0.0, Q.sd_mass - 72.1)
            - 0.0345 * max(0.0, Q.sd_mass - 86.4)
            + 0.0107 * max(0.0, Q.sum_pt - 908.0)
            - 0.0157 * max(0.0, Q.sum_pt - 990.0)
            + 0.0172 * max(0.0, Q.sum_pt_top50 - 943.0)
            - 0.695 * max(0.0, 0.561 - Q.tau21)
            - 0.0157 * max(0.0, 56.1 - Q.n_particles) * max(0.0, 2.7 - Q.D2)
        ))
        + 0.0625 * grid(6, max(0.0, 0.882
            - 31.7 * max(0.0, Q.e2 - 0.0294)
            + 69.1 * max(0.0, Q.e2 - 0.053)
            + 198.0 * max(0.0, Q.girth2_top20 - 0.00733)
            - 223.0 * max(0.0, Q.girth2_top30 - 0.0275)
            + 0.0999 * max(0.0, 93.9 - Q.mass)
            - 0.105 * max(0.0, 102.0 - Q.mass)
            - 61.0 * max(0.0, Q.mass_over_sum_pt - 0.121)
            + 0.0307 * max(0.0, 73.3 - Q.mass_top50)
            - 0.119 * max(0.0, Q.sj3_pair_mass_min - 77.9)
            + 15.2 * max(0.0, 0.0533 - Q.tau1)
            + 11.9 * max(0.0, 0.931 - Q.z_top40_slots)
            - 2330.0 * max(0.0, Q.girth2_top20 - 0.0035) * max(0.0, Q.C2_b2 - 0.00379)
            + 8.43e-05 * max(0.0, 102.0 - Q.mass) * max(0.0, 1010.0 - Q.sum_pt)
        ))
        - 0.625 * grid(7, max(0.0, 0.0182
            + 0.0156 * max(0.0, 121.0 - Q.mass)
            + 0.0606 * max(0.0, 8.15 - Q.n_dr_0p2_0p4)
            - 476.0 * max(0.0, Q.psi_0p3 - 0.998)
            - 0.0854 * max(0.0, Q.sd_mass - 97.2)
            + 0.0268 * max(0.0, 69.8 - Q.sd_mass)
            - 0.0198 * max(0.0, 87.2 - Q.sd_mass)
            + 24.1 * max(0.0, 0.0957 - Q.tau1)
            - 33.8 * max(0.0, 0.107 - Q.tau1)
            + 6.45 * max(0.0, 0.234 - Q.tau21_b2)
            - 30200.0 * max(0.0, 0.00796 - Q.lam1) * max(0.0, Q.zdr_0 - 0.0152)
            + 7.71 * max(0.0, 82.3 - Q.mass) * max(0.0, Q.psi_0p3 - 0.978)
            - 17.3 * max(0.0, 91.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.978)
            + 9.53 * max(0.0, 91.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.964)
            - 64.8 * max(0.0, 91.5 - Q.mass) * max(0.0, Q.psi_0p3 - 0.998)
            - 8.46 * max(0.0, 93.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.964)
            + 62.7 * max(0.0, 101.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.998)
            + 7.44 * max(0.0, 102.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.978)
            - 0.0988 * max(0.0, 93.2 - Q.mass) * max(0.0, 0.503 - Q.z_dr_0_0p05)
            + 6910.0 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, Q.z_top50_slots - 0.979)
            - 1320.0 * max(0.0, 0.23 - Q.tau21_b2) * max(0.0, 0.00808 - Q.mass_over_sum_pt_sq)
            - 0.0184 * max(0.0, 0.231 - Q.tau21_b2) * max(0.0, 1270.0 - Q.sum_pt)
        ))
        - 0.875 * grid(8, max(0.0, 1.3
            - 202.0 * max(0.0, Q.girth2_top20 - 0.0059)
            + 268.0 * max(0.0, Q.girth2_top20 - 0.00788)
            + 165.0 * max(0.0, 0.00754 - Q.girth2_top30)
            + 0.0435 * max(0.0, Q.mass - 71.1)
            - 0.0408 * max(0.0, Q.mass - 103.0)
            - 0.0209 * max(0.0, Q.mass - 127.0)
            + 0.047 * max(0.0, 8.6 - Q.n_dr_0p2_0p4)
            - 0.0677 * max(0.0, 17.9 - Q.n_dr_0p2_0p4)
            + 4.23 * max(0.0, Q.sj2_dr - 0.23)
            + 0.0195 * max(0.0, 1030.0 - Q.sum_pt)
            + 0.0014 * max(0.0, Q.sum_pt_top30 - 987.0)
            - 0.00795 * max(0.0, 1030.0 - Q.sum_pt_top40)
            - 214.0 * max(0.0, 0.00987 - Q.width)
            - 35.8 * max(0.0, Q.z_top50_slots - 0.972)
            - 810.0 * max(0.0, Q.girth2_top40 - 0.00555) * max(0.0, 6.99 - Q.log_sum_pt)
            + 0.305 * max(0.0, Q.mass_over_sum_pt - 0.0785) * max(0.0, 1100.0 - Q.sum_pt)
            + 0.207 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.313)
            + 0.403 * max(0.0, 0.0848 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.4)
        ))
        - 0.21875 * grid(9, max(0.0, -1.13
            + 23.3 * max(0.0, Q.girth - 0.0976)
            + 281.0 * max(0.0, 0.00617 - Q.girth2_top40)
            - 0.07 * max(0.0, 62.9 - Q.mass)
            + 0.103 * max(0.0, 85.5 - Q.mass)
            - 0.0621 * max(0.0, 147.0 - Q.mass)
            + 0.0491 * max(0.0, 178.0 - Q.mass)
            + 0.00819 * max(0.0, 989.0 - Q.sum_pt)
            - 8.68 * max(0.0, Q.z_top40_slots - 0.954)
            + 0.00132 * max(0.0, 116.0 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 8.57)
            - 0.000231 * max(0.0, 97.9 - Q.mass_top40) * max(0.0, 1020.0 - Q.sum_pt)
            + 0.000185 * max(0.0, 100.0 - Q.mass_top40) * max(0.0, 961.0 - Q.sum_pt_top40)
        ))
        + 0.59375 * grid(11, max(0.0, -0.575
            + 96.2 * max(0.0, 0.0266 - Q.e2)
            - 402.0 * max(0.0, 0.00666 - Q.girth2_top30)
            - 0.0787 * max(0.0, 80.9 - Q.mass_top50)
            + 0.0866 * max(0.0, 94.7 - Q.mass_top50)
            + 0.0549 * max(0.0, 16.7 - Q.n_dr_0p1_0p2)
            + 0.184 * max(0.0, 7.5 - Q.n_dr_0p2_0p4)
            + 64.2 * max(0.0, Q.psi_0p3 - 0.99)
            - 1.56 * max(0.0, Q.z_top5_slots - 0.552)
            - 2.65e-05 * max(0.0, 119.0 - Q.mass) * max(0.0, 872.0 - Q.sum_pt_top10)
            - 28.5 * max(0.0, 9.67 - Q.n_dr_0p2_0p4) * max(0.0, 0.0062 - Q.girth2)
            - 0.0051 * max(0.0, 8.65 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 36.1)
        ))
        - 0.40625 * grid(12, max(0.0, -0.151
            - 172.0 * max(0.0, 0.00586 - Q.girth2_top15)
            - 0.0956 * max(0.0, 59.7 - Q.mass)
            + 0.119 * max(0.0, 83.4 - Q.mass)
            + 0.0144 * max(0.0, Q.sd_mass - 124.0)
            - 3400.0 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0036 - Q.lam2)
            - 3.47 * max(0.0, 90.0 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            + 0.000437 * max(0.0, Q.sj3_pair_mass_max - 123.0) * max(0.0, 65.0 - Q.sj3_pair_mass_min)
        ))
        - 1.375 * grid(14, max(0.0, -0.384
            + 59.7 * max(0.0, 0.0302 - Q.e2)
            - 359.0 * max(0.0, 0.00154 - Q.girth2_top3)
            - 3370.0 * max(0.0, 0.00262 - Q.girth2_top30)
            + 78.0 * max(0.0, 0.0067 - Q.girth2_top5)
            + 5.69 * max(0.0, 6.94 - Q.log_sum_pt)
            - 0.137 * max(0.0, 82.8 - Q.mass)
            - 0.11 * max(0.0, 92.0 - Q.mass)
            + 17.6 * max(0.0, 0.144 - Q.mass_over_sum_pt)
            + 0.0691 * max(0.0, 78.4 - Q.mass_top50)
            + 152.0 * max(0.0, Q.psi_0p3 - 0.993)
            - 7.57 * max(0.0, 0.484 - Q.N2) * max(0.0, Q.max_dr - 0.24)
            - 7.52 * max(0.0, 0.00654 - Q.girth2_top5) * max(0.0, Q.n_pt_above_10 - 13.0)
            - 405.0 * max(0.0, 0.0905 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p2 - 0.914)
            + 30.8 * max(0.0, Q.psi_0p2 - 0.978) * max(0.0, Q.pt2_over_pt0 - 0.121)
            - 420.0 * max(0.0, Q.psi_0p3 - 0.993) * max(0.0, 0.188 - Q.sd_rg)
            - 939.0 * max(0.0, Q.psi_0p3 - 0.993) * max(0.0, 0.124 - Q.z_2nd)
            - 0.22 * max(0.0, 0.322 - Q.tau21_b2) * max(0.0, 30.6 - Q.n_real_top40)
            - 0.0104 * max(0.0, 0.432 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -12.9)
        ))
        - 0.1875 * grid(15, max(0.0, 0.322
            - 50.4 * max(0.0, Q.e2 - 0.0294)
            + 305.0 * max(0.0, 0.00181 - Q.lam2)
            + 7.67 * max(0.0, 7.03 - Q.log_sum_pt)
            - 4.66 * max(0.0, 0.294 - Q.max_dr)
            + 0.0487 * max(0.0, 13.6 - Q.n_dr_0p1_0p2)
            - 0.047 * max(0.0, Q.n_dr_0p2_0p4 - 8.79)
            - 4.81 * max(0.0, Q.psi_0p2 - 0.857)
            - 47.9 * max(0.0, Q.psi_0p3 - 0.989)
            - 0.0361 * max(0.0, 44.2 - Q.sd_mass)
            + 0.0292 * max(0.0, 73.6 - Q.sd_mass)
            + 5.7 * max(0.0, Q.sd_rg - 0.203)
            - 0.0123 * max(0.0, 1020.0 - Q.sum_pt_top40)
            - 15.5 * max(0.0, 0.0678 - Q.tau1)
            + 147.0 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0493 - Q.C2_b2)
        ))
    )


def logit_Z(Q):
    return (0.984375
        - 1.375 * grid(0, max(0.0, 0.798
            - 315.0 * max(0.0, 0.00614 - Q.e2_sq)
            - 73.9 * max(0.0, 0.00652 - Q.girth2_top20)
            - 0.156 * max(0.0, Q.mass - 78.3)
            - 0.0352 * max(0.0, Q.mass - 93.8)
            - 0.0161 * max(0.0, 99.1 - Q.mass)
            + 501.0 * max(0.0, 0.00766 - Q.mass_over_sum_pt_sq)
            + 0.0474 * max(0.0, Q.mass_top50 - 82.0)
            + 0.0494 * max(0.0, 15.3 - Q.n_dr_0p2_0p4)
            + 55.8 * max(0.0, Q.psi_0p3 - 0.996)
            - 0.00875 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.00302 * max(0.0, 1160.0 - Q.sum_pt_top50)
            - 18.4 * max(0.0, 0.0699 - Q.tau1)
            - 4.77 * max(0.0, 0.0913 - Q.z_dr_0p2_0p4)
            - 22.1 * max(0.0, 0.991 - Q.z_top50_slots)
            + 0.0566 * max(0.0, 7.0 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 963.0)
        ))
        - 0.375 * grid(2, max(0.0, 0.00944
            + 206.0 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, 0.0272 - Q.girth2_top15)
        ))
        + 0.4375 * grid(3, max(0.0, -0.0649
            + 79.4 * max(0.0, 0.00669 - Q.girth2_top40)
            + 1150.0 * max(0.0, 0.000625 - Q.lam2)
            - 0.0241 * max(0.0, 69.2 - Q.mass_top50)
            + 0.0306 * max(0.0, 46.1 - Q.n_particles)
            - 2.0 * max(0.0, 0.375 - Q.tau21)
            + 0.592 * max(0.0, 11.2 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.934)
            + 3.7 * max(0.0, 7.15 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.98)
            - 214.0 * max(0.0, 0.406 - Q.tau21) * max(0.0, Q.lam1 - 0.00572)
        ))
        + 0.6875 * grid(5, max(0.0, -0.375
            - 108.0 * max(0.0, Q.girth2_top50 - 0.0191)
            - 23.0 * max(0.0, Q.log_sum_pt - 6.91)
            + 4.88 * max(0.0, Q.log_sum_pt - 6.99)
            + 0.0337 * max(0.0, Q.mass - 73.6)
            - 0.022 * max(0.0, Q.mass - 92.4)
            - 0.177 * max(0.0, Q.mass - 171.0)
            - 23.0 * max(0.0, Q.mass_over_sum_pt - 0.0781)
            - 0.277 * max(0.0, Q.mass_top50 - 158.0)
            + 2.32 * max(0.0, 0.459 - Q.max_dr)
            - 0.0129 * max(0.0, 22.6 - Q.n_dr_0p1_0p2)
            + 0.0302 * max(0.0, 12.4 - Q.n_dr_0p2_0p4)
            + 0.0303 * max(0.0, 47.5 - Q.n_particles)
            + 0.0286 * max(0.0, Q.sd_mass - 72.1)
            - 0.0345 * max(0.0, Q.sd_mass - 86.4)
            + 0.0107 * max(0.0, Q.sum_pt - 908.0)
            - 0.0157 * max(0.0, Q.sum_pt - 990.0)
            + 0.0172 * max(0.0, Q.sum_pt_top50 - 943.0)
            - 0.695 * max(0.0, 0.561 - Q.tau21)
            - 0.0157 * max(0.0, 56.1 - Q.n_particles) * max(0.0, 2.7 - Q.D2)
        ))
        - 0.96875 * grid(6, max(0.0, 0.882
            - 31.7 * max(0.0, Q.e2 - 0.0294)
            + 69.1 * max(0.0, Q.e2 - 0.053)
            + 198.0 * max(0.0, Q.girth2_top20 - 0.00733)
            - 223.0 * max(0.0, Q.girth2_top30 - 0.0275)
            + 0.0999 * max(0.0, 93.9 - Q.mass)
            - 0.105 * max(0.0, 102.0 - Q.mass)
            - 61.0 * max(0.0, Q.mass_over_sum_pt - 0.121)
            + 0.0307 * max(0.0, 73.3 - Q.mass_top50)
            - 0.119 * max(0.0, Q.sj3_pair_mass_min - 77.9)
            + 15.2 * max(0.0, 0.0533 - Q.tau1)
            + 11.9 * max(0.0, 0.931 - Q.z_top40_slots)
            - 2330.0 * max(0.0, Q.girth2_top20 - 0.0035) * max(0.0, Q.C2_b2 - 0.00379)
            + 8.43e-05 * max(0.0, 102.0 - Q.mass) * max(0.0, 1010.0 - Q.sum_pt)
        ))
        + 0.90625 * grid(7, max(0.0, 0.0182
            + 0.0156 * max(0.0, 121.0 - Q.mass)
            + 0.0606 * max(0.0, 8.15 - Q.n_dr_0p2_0p4)
            - 476.0 * max(0.0, Q.psi_0p3 - 0.998)
            - 0.0854 * max(0.0, Q.sd_mass - 97.2)
            + 0.0268 * max(0.0, 69.8 - Q.sd_mass)
            - 0.0198 * max(0.0, 87.2 - Q.sd_mass)
            + 24.1 * max(0.0, 0.0957 - Q.tau1)
            - 33.8 * max(0.0, 0.107 - Q.tau1)
            + 6.45 * max(0.0, 0.234 - Q.tau21_b2)
            - 30200.0 * max(0.0, 0.00796 - Q.lam1) * max(0.0, Q.zdr_0 - 0.0152)
            + 7.71 * max(0.0, 82.3 - Q.mass) * max(0.0, Q.psi_0p3 - 0.978)
            - 17.3 * max(0.0, 91.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.978)
            + 9.53 * max(0.0, 91.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.964)
            - 64.8 * max(0.0, 91.5 - Q.mass) * max(0.0, Q.psi_0p3 - 0.998)
            - 8.46 * max(0.0, 93.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.964)
            + 62.7 * max(0.0, 101.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.998)
            + 7.44 * max(0.0, 102.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.978)
            - 0.0988 * max(0.0, 93.2 - Q.mass) * max(0.0, 0.503 - Q.z_dr_0_0p05)
            + 6910.0 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, Q.z_top50_slots - 0.979)
            - 1320.0 * max(0.0, 0.23 - Q.tau21_b2) * max(0.0, 0.00808 - Q.mass_over_sum_pt_sq)
            - 0.0184 * max(0.0, 0.231 - Q.tau21_b2) * max(0.0, 1270.0 - Q.sum_pt)
        ))
        - 0.9375 * grid(8, max(0.0, 1.3
            - 202.0 * max(0.0, Q.girth2_top20 - 0.0059)
            + 268.0 * max(0.0, Q.girth2_top20 - 0.00788)
            + 165.0 * max(0.0, 0.00754 - Q.girth2_top30)
            + 0.0435 * max(0.0, Q.mass - 71.1)
            - 0.0408 * max(0.0, Q.mass - 103.0)
            - 0.0209 * max(0.0, Q.mass - 127.0)
            + 0.047 * max(0.0, 8.6 - Q.n_dr_0p2_0p4)
            - 0.0677 * max(0.0, 17.9 - Q.n_dr_0p2_0p4)
            + 4.23 * max(0.0, Q.sj2_dr - 0.23)
            + 0.0195 * max(0.0, 1030.0 - Q.sum_pt)
            + 0.0014 * max(0.0, Q.sum_pt_top30 - 987.0)
            - 0.00795 * max(0.0, 1030.0 - Q.sum_pt_top40)
            - 214.0 * max(0.0, 0.00987 - Q.width)
            - 35.8 * max(0.0, Q.z_top50_slots - 0.972)
            - 810.0 * max(0.0, Q.girth2_top40 - 0.00555) * max(0.0, 6.99 - Q.log_sum_pt)
            + 0.305 * max(0.0, Q.mass_over_sum_pt - 0.0785) * max(0.0, 1100.0 - Q.sum_pt)
            + 0.207 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.313)
            + 0.403 * max(0.0, 0.0848 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.4)
        ))
        + 0.3125 * grid(12, max(0.0, -0.151
            - 172.0 * max(0.0, 0.00586 - Q.girth2_top15)
            - 0.0956 * max(0.0, 59.7 - Q.mass)
            + 0.119 * max(0.0, 83.4 - Q.mass)
            + 0.0144 * max(0.0, Q.sd_mass - 124.0)
            - 3400.0 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0036 - Q.lam2)
            - 3.47 * max(0.0, 90.0 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            + 0.000437 * max(0.0, Q.sj3_pair_mass_max - 123.0) * max(0.0, 65.0 - Q.sj3_pair_mass_min)
        ))
        + 0.5625 * grid(15, max(0.0, 0.322
            - 50.4 * max(0.0, Q.e2 - 0.0294)
            + 305.0 * max(0.0, 0.00181 - Q.lam2)
            + 7.67 * max(0.0, 7.03 - Q.log_sum_pt)
            - 4.66 * max(0.0, 0.294 - Q.max_dr)
            + 0.0487 * max(0.0, 13.6 - Q.n_dr_0p1_0p2)
            - 0.047 * max(0.0, Q.n_dr_0p2_0p4 - 8.79)
            - 4.81 * max(0.0, Q.psi_0p2 - 0.857)
            - 47.9 * max(0.0, Q.psi_0p3 - 0.989)
            - 0.0361 * max(0.0, 44.2 - Q.sd_mass)
            + 0.0292 * max(0.0, 73.6 - Q.sd_mass)
            + 5.7 * max(0.0, Q.sd_rg - 0.203)
            - 0.0123 * max(0.0, 1020.0 - Q.sum_pt_top40)
            - 15.5 * max(0.0, 0.0678 - Q.tau1)
            + 147.0 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0493 - Q.C2_b2)
        ))
    )


def logit_t(Q):
    return (0.78125
        + 0.125 * grid(0, max(0.0, 0.798
            - 315.0 * max(0.0, 0.00614 - Q.e2_sq)
            - 73.9 * max(0.0, 0.00652 - Q.girth2_top20)
            - 0.156 * max(0.0, Q.mass - 78.3)
            - 0.0352 * max(0.0, Q.mass - 93.8)
            - 0.0161 * max(0.0, 99.1 - Q.mass)
            + 501.0 * max(0.0, 0.00766 - Q.mass_over_sum_pt_sq)
            + 0.0474 * max(0.0, Q.mass_top50 - 82.0)
            + 0.0494 * max(0.0, 15.3 - Q.n_dr_0p2_0p4)
            + 55.8 * max(0.0, Q.psi_0p3 - 0.996)
            - 0.00875 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.00302 * max(0.0, 1160.0 - Q.sum_pt_top50)
            - 18.4 * max(0.0, 0.0699 - Q.tau1)
            - 4.77 * max(0.0, 0.0913 - Q.z_dr_0p2_0p4)
            - 22.1 * max(0.0, 0.991 - Q.z_top50_slots)
            + 0.0566 * max(0.0, 7.0 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 963.0)
        ))
        + 0.1875 * grid(4, max(0.0, 1.11
            - 2.55 * max(0.0, Q.C2 - 0.0725)
            - 23.2 * max(0.0, Q.e2 - 0.0568)
            + 173.0 * max(0.0, Q.e2_sq - 0.0103)
            - 15.4 * max(0.0, Q.girth - 0.121)
            - 71.7 * max(0.0, Q.girth2_top15 - 0.00752)
            + 78.5 * max(0.0, Q.girth2_top15 - 0.016)
            - 93.2 * max(0.0, 0.00531 - Q.girth2_top15)
            - 70.8 * max(0.0, Q.lam1 - 0.00794)
            - 0.0511 * max(0.0, 78.4 - Q.mass)
            - 0.0818 * max(0.0, 86.2 - Q.mass)
            + 0.0279 * max(0.0, 119.0 - Q.mass)
            + 0.061 * max(0.0, 67.5 - Q.mass_top40)
            - 0.0135 * max(0.0, 166.0 - Q.mass_top40)
            + 0.0309 * max(0.0, 24.2 - Q.n_dr_0p2_0p4)
            - 0.0255 * max(0.0, Q.n_particles - 23.1)
            + 182.0 * max(0.0, Q.psi_0p3 - 0.997)
            + 0.00874 * max(0.0, 101.0 - Q.mass) * max(0.0, 3.79 - Q.D2_b2)
            - 0.0186 * max(0.0, 88.5 - Q.mass_top40) * max(0.0, 3.21 - Q.D2_b2)
            - 0.00112 * max(0.0, 22.7 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 12.1)
            + 0.000656 * max(0.0, Q.n_particles - 27.6) * max(0.0, Q.n_dr_0_0p05 - 10.7)
            + 0.00517 * max(0.0, Q.n_particles - 20.4) * max(0.0, 2.44 - Q.soft1_pt)
            - 11.5 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 13.2 - Q.n_dr_0_0p05)
            - 0.194 * max(0.0, Q.sj2_dr - 0.232) * max(0.0, 20.6 - Q.D2_b2)
        ))
        - 0.46875 * grid(5, max(0.0, -0.375
            - 108.0 * max(0.0, Q.girth2_top50 - 0.0191)
            - 23.0 * max(0.0, Q.log_sum_pt - 6.91)
            + 4.88 * max(0.0, Q.log_sum_pt - 6.99)
            + 0.0337 * max(0.0, Q.mass - 73.6)
            - 0.022 * max(0.0, Q.mass - 92.4)
            - 0.177 * max(0.0, Q.mass - 171.0)
            - 23.0 * max(0.0, Q.mass_over_sum_pt - 0.0781)
            - 0.277 * max(0.0, Q.mass_top50 - 158.0)
            + 2.32 * max(0.0, 0.459 - Q.max_dr)
            - 0.0129 * max(0.0, 22.6 - Q.n_dr_0p1_0p2)
            + 0.0302 * max(0.0, 12.4 - Q.n_dr_0p2_0p4)
            + 0.0303 * max(0.0, 47.5 - Q.n_particles)
            + 0.0286 * max(0.0, Q.sd_mass - 72.1)
            - 0.0345 * max(0.0, Q.sd_mass - 86.4)
            + 0.0107 * max(0.0, Q.sum_pt - 908.0)
            - 0.0157 * max(0.0, Q.sum_pt - 990.0)
            + 0.0172 * max(0.0, Q.sum_pt_top50 - 943.0)
            - 0.695 * max(0.0, 0.561 - Q.tau21)
            - 0.0157 * max(0.0, 56.1 - Q.n_particles) * max(0.0, 2.7 - Q.D2)
        ))
        - 0.28125 * grid(7, max(0.0, 0.0182
            + 0.0156 * max(0.0, 121.0 - Q.mass)
            + 0.0606 * max(0.0, 8.15 - Q.n_dr_0p2_0p4)
            - 476.0 * max(0.0, Q.psi_0p3 - 0.998)
            - 0.0854 * max(0.0, Q.sd_mass - 97.2)
            + 0.0268 * max(0.0, 69.8 - Q.sd_mass)
            - 0.0198 * max(0.0, 87.2 - Q.sd_mass)
            + 24.1 * max(0.0, 0.0957 - Q.tau1)
            - 33.8 * max(0.0, 0.107 - Q.tau1)
            + 6.45 * max(0.0, 0.234 - Q.tau21_b2)
            - 30200.0 * max(0.0, 0.00796 - Q.lam1) * max(0.0, Q.zdr_0 - 0.0152)
            + 7.71 * max(0.0, 82.3 - Q.mass) * max(0.0, Q.psi_0p3 - 0.978)
            - 17.3 * max(0.0, 91.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.978)
            + 9.53 * max(0.0, 91.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.964)
            - 64.8 * max(0.0, 91.5 - Q.mass) * max(0.0, Q.psi_0p3 - 0.998)
            - 8.46 * max(0.0, 93.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.964)
            + 62.7 * max(0.0, 101.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.998)
            + 7.44 * max(0.0, 102.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.978)
            - 0.0988 * max(0.0, 93.2 - Q.mass) * max(0.0, 0.503 - Q.z_dr_0_0p05)
            + 6910.0 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, Q.z_top50_slots - 0.979)
            - 1320.0 * max(0.0, 0.23 - Q.tau21_b2) * max(0.0, 0.00808 - Q.mass_over_sum_pt_sq)
            - 0.0184 * max(0.0, 0.231 - Q.tau21_b2) * max(0.0, 1270.0 - Q.sum_pt)
        ))
        + 0.2109375 * grid(8, max(0.0, 1.3
            - 202.0 * max(0.0, Q.girth2_top20 - 0.0059)
            + 268.0 * max(0.0, Q.girth2_top20 - 0.00788)
            + 165.0 * max(0.0, 0.00754 - Q.girth2_top30)
            + 0.0435 * max(0.0, Q.mass - 71.1)
            - 0.0408 * max(0.0, Q.mass - 103.0)
            - 0.0209 * max(0.0, Q.mass - 127.0)
            + 0.047 * max(0.0, 8.6 - Q.n_dr_0p2_0p4)
            - 0.0677 * max(0.0, 17.9 - Q.n_dr_0p2_0p4)
            + 4.23 * max(0.0, Q.sj2_dr - 0.23)
            + 0.0195 * max(0.0, 1030.0 - Q.sum_pt)
            + 0.0014 * max(0.0, Q.sum_pt_top30 - 987.0)
            - 0.00795 * max(0.0, 1030.0 - Q.sum_pt_top40)
            - 214.0 * max(0.0, 0.00987 - Q.width)
            - 35.8 * max(0.0, Q.z_top50_slots - 0.972)
            - 810.0 * max(0.0, Q.girth2_top40 - 0.00555) * max(0.0, 6.99 - Q.log_sum_pt)
            + 0.305 * max(0.0, Q.mass_over_sum_pt - 0.0785) * max(0.0, 1100.0 - Q.sum_pt)
            + 0.207 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.313)
            + 0.403 * max(0.0, 0.0848 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.4)
        ))
        + 0.984375 * grid(10, max(0.0, 0.309
            + 0.473 * max(0.0, 3.04 - Q.D2)
            + 7.76 * max(0.0, 0.0664 - Q.dr_0)
            + 29.9 * max(0.0, Q.e2 - 0.0128)
            - 236.0 * max(0.0, 0.00533 - Q.girth2_top20)
            + 0.0141 * max(0.0, Q.mass - 78.8)
            - 0.0869 * max(0.0, Q.mass - 145.0)
            - 0.0481 * max(0.0, Q.mass - 164.0)
            + 0.0998 * max(0.0, 87.9 - Q.mass)
            - 0.0326 * max(0.0, 99.6 - Q.mass)
            - 31.1 * max(0.0, Q.mass_over_sum_pt - 0.16)
            + 0.0112 * max(0.0, Q.mass_top5 - 17.5)
            + 0.0434 * max(0.0, Q.mass_top50 - 138.0)
            - 0.055 * max(0.0, 11.1 - Q.n_dr_0p2_0p4)
            - 2.54 * max(0.0, Q.pt_dispersion - 0.27)
            - 0.0219 * max(0.0, 65.0 - Q.sj2_mass1)
            - 0.0107 * max(0.0, 984.0 - Q.sum_pt)
            - 0.0034 * max(0.0, Q.sum_pt_top10 - 941.0)
            + 0.00394 * max(0.0, 941.0 - Q.sum_pt_top15)
            - 82.7 * max(0.0, 0.047 - Q.tau1)
            + 14.2 * max(0.0, 0.0485 - Q.tau2)
            + 4.97 * max(0.0, 0.0859 - Q.z_dr_0p2_0p4)
            - 3.21 * max(0.0, 2.99 - Q.D2) * max(0.0, Q.sj2_dr - 0.192)
            - 17300.0 * max(0.0, 0.0234 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.998)
            - 7.19 * max(0.0, Q.mass_top50 - 133.0) * max(0.0, Q.soft5_z - 0.00163)
            + 0.000322 * max(0.0, 63.3 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 7.52)
            - 0.877 * max(0.0, Q.sum_pt_top10 - 943.0) * max(0.0, -0.078 - Q.eta_0)
        ))
        - 0.375 * grid(12, max(0.0, -0.151
            - 172.0 * max(0.0, 0.00586 - Q.girth2_top15)
            - 0.0956 * max(0.0, 59.7 - Q.mass)
            + 0.119 * max(0.0, 83.4 - Q.mass)
            + 0.0144 * max(0.0, Q.sd_mass - 124.0)
            - 3400.0 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0036 - Q.lam2)
            - 3.47 * max(0.0, 90.0 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3)
            + 0.000437 * max(0.0, Q.sj3_pair_mass_max - 123.0) * max(0.0, 65.0 - Q.sj3_pair_mass_min)
        ))
        - 0.90625 * grid(13, max(0.0, 1.97
            - 24.1 * max(0.0, 6.92 - Q.log_sum_pt)
            + 0.015 * max(0.0, Q.mass - 63.1)
            - 0.0148 * max(0.0, Q.mass - 101.0)
            - 0.0998 * max(0.0, Q.mass - 161.0)
            + 0.0121 * max(0.0, Q.mass_top5 - 40.1)
            + 0.00809 * max(0.0, 1020.0 - Q.sum_pt_top40)
            + 1140.0 * max(0.0, Q.log_sum_pt - 7.14) * max(0.0, Q.sd_zg - 0.446)
            - 0.0225 * max(0.0, Q.mass - 142.0) * max(0.0, Q.D2 - 0.0302)
            + 0.0811 * max(0.0, Q.mass - 169.0) * max(0.0, Q.D2 - 0.281)
            - 329.0 * max(0.0, Q.mass - 173.0) * max(0.0, Q.soft5_z - 0.000296)
            - 27.5 * max(0.0, Q.mass_top50 - 134.0) * max(0.0, Q.soft6_z - 0.00094)
            - 7.59 * max(0.0, Q.mass_top50 - 90.5) * max(0.0, 0.00189 - Q.soft7_z)
            - 0.0078 * max(0.0, 1090.0 - Q.sum_pt) * max(0.0, 1.0 - Q.tau21_b2)
        ))
        - 0.375 * grid(15, max(0.0, 0.322
            - 50.4 * max(0.0, Q.e2 - 0.0294)
            + 305.0 * max(0.0, 0.00181 - Q.lam2)
            + 7.67 * max(0.0, 7.03 - Q.log_sum_pt)
            - 4.66 * max(0.0, 0.294 - Q.max_dr)
            + 0.0487 * max(0.0, 13.6 - Q.n_dr_0p1_0p2)
            - 0.047 * max(0.0, Q.n_dr_0p2_0p4 - 8.79)
            - 4.81 * max(0.0, Q.psi_0p2 - 0.857)
            - 47.9 * max(0.0, Q.psi_0p3 - 0.989)
            - 0.0361 * max(0.0, 44.2 - Q.sd_mass)
            + 0.0292 * max(0.0, 73.6 - Q.sd_mass)
            + 5.7 * max(0.0, Q.sd_rg - 0.203)
            - 0.0123 * max(0.0, 1020.0 - Q.sum_pt_top40)
            - 15.5 * max(0.0, 0.0678 - Q.tau1)
            + 147.0 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0493 - Q.C2_b2)
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
