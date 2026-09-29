"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; all observables), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.8% (the network: 81.1%); same class as the network for 92.3% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_11                 pT share × ΔR of particle 11 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
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
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
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
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
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
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_90pct=ncum(0.9),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_11=pt[11],
        pt_2=pt[2],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        z_11=z[11],
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top2_slots=sum(pt[:2]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        zdr_11=z[11] * dr[11],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
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
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
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
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-1.078125
        + 0.625 * grid(1, max(0.0, 1.48714
            + 0.09328952 * max(0.0, Q.n_particles - 38.0)
            + 21.0509 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.01273634 * max(0.0, Q.sum_pt_top50 - 959.0957)
            - 20.74087 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.00253462 * max(0.0, 689.25 - Q.sum_pt_top2)
            - 0.01626576 * max(0.0, 47.88842 - Q.mass_top20)
            - 0.02671432 * max(0.0, 32.50209 - Q.sj3_mass1)
            + 0.5237017 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793)
            + 0.005140426 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.3875843 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0)
            + 0.003762992 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0)
            - 0.03179651 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt)
            - 0.1279552 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            + 160.675 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2)
            + 1862.292 * max(0.0, 0.0005522528 - Q.girth2_top3)
            - 0.002737062 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2)
            - 44.64202 * max(0.0, 0.03187688 - Q.M3)
            - 0.02125498 * max(0.0, 31.35938 - Q.pt_9)
            + 22.84232 * max(0.0, Q.log_sum_pt - 6.893714)
            + 171.814 * max(0.0, 0.004673423 - Q.lam1)
            - 0.01121205 * max(0.0, 120.6 - Q.mass)
            + 54.39146 * max(0.0, 0.02412652 - Q.girth2_top30)
            - 0.007804931 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 0.9624339 - Q.tau43)
            - 5.901248 * max(0.0, 7.139296 - Q.log_sum_pt)
            - 2.742065 * max(0.0, Q.log_sum_pt - 6.959294)
            + 0.009289864 * max(0.0, 1017.435 - Q.sum_pt)
            + 0.529206 * max(0.0, Q.pt_entropy - 2.07371)
        ))
        - 0.75 * grid(3, max(0.0, 0.06251741
            - 0.0393809 * max(0.0, 46.0 - Q.n_particles)
            + 0.0001463702 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732)
            - 6.400079 * max(0.0, 0.3861957 - Q.tau21)
            + 24.27948 * max(0.0, 0.08665515 - Q.mass_over_sum_pt)
            + 480.666 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 0.01364892 * max(0.0, 79.21004 - Q.mass_top50)
            - 147.3656 * max(0.0, 0.009614971 - Q.girth2)
            + 99.26788 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 0.01474629 * max(0.0, 92.85979 - Q.mass)
            - 0.00500493 * max(0.0, 79.21004 - Q.mass_top50) * max(0.0, Q.mass_top3 - 8.921413)
            + 0.0439486 * max(0.0, 27.56535 - Q.sj2_mass1)
            - 0.02451443 * max(0.0, 80.4 - Q.mass_top40)
            - 731.6249 * max(0.0, 0.001260456 - Q.lam1)
            + 150.2475 * max(0.0, 0.01626937 - Q.tau4)
        ))
        - 0.875 * grid(4, max(0.0, 1.367567
            - 0.02142609 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.03053927 * max(0.0, 60.43821 - Q.mass_top30)
            + 0.0201289 * max(0.0, 120.6 - Q.mass)
            - 0.02839266 * max(0.0, 86.4 - Q.mass)
            + 0.02532736 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.02472351 * max(0.0, Q.n_particles - 22.0)
            - 282.8053 * max(0.0, 0.004855289 - Q.girth2_top15)
            - 38.14918 * max(0.0, Q.girth2_top15 - 0.00727763)
            + 0.01708885 * max(0.0, 57.87349 - Q.mass_top15)
            + 0.05217597 * max(0.0, 101.0497 - Q.mass)
            - 0.03281671 * max(0.0, 80.78464 - Q.mass)
            + 0.08842908 * max(0.0, 74.25181 - Q.mass)
            - 0.008418597 * max(0.0, 163.2541 - Q.mass_top40)
            - 0.02998102 * max(0.0, 92.85979 - Q.mass)
            + 0.04484549 * max(0.0, 67.72643 - Q.mass_top40)
            - 0.1146276 * max(0.0, 78.26182 - Q.mass)
            - 13.37402 * max(0.0, Q.girth - 0.1207452)
            - 149.4077 * max(0.0, Q.lam1 - 0.008241985)
            + 55.23399 * max(0.0, Q.girth2_top15 - 0.01563836)
            + 167.0418 * max(0.0, Q.e2_sq - 0.009606007)
            - 143.5591 * max(0.0, Q.lam2 - 0.001776308)
        ))
        - 0.3125 * grid(5, max(0.0, 0.6380697
            + 0.0381413 * max(0.0, 64.0 - Q.n_particles)
            - 10.20386 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 0.006261253 * max(0.0, Q.sum_pt - 907.9372)
            - 15.76866 * max(0.0, Q.log_sum_pt - 6.920349)
            + 167.2861 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.01647099 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.02016727 * max(0.0, Q.sum_pt_top50 - 934.2416)
            - 0.005608218 * max(0.0, Q.sum_pt_top20 - 1129.275)
            + 29743.12 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4)
            - 15.12077 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.01244614 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 9.362286 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.005879184 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 2.518499 * max(0.0, Q.max_dr - 0.2404747)
            - 14.75085 * max(0.0, Q.z_top30_slots - 0.9048492)
            - 0.02569512 * max(0.0, Q.mass_top40 - 120.6)
            + 0.07803723 * max(0.0, Q.mass - 172.8)
            - 115.5639 * max(0.0, Q.girth2_top30 - 0.006363916)
            - 55.14108 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2)
            - 0.005410795 * max(0.0, 150.0144 - Q.mass_top40)
            - 511.5796 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            - 2.370548 * max(0.0, 0.5494307 - Q.tau21)
            + 0.05009839 * max(0.0, Q.sd_mass - 69.65633)
            - 0.1307292 * max(0.0, Q.mass_top50 - 157.5448)
            - 0.004752506 * max(0.0, Q.sum_pt - 986.0565)
            + 84.69973 * max(0.0, 0.01375115 - Q.z_11)
            + 0.01644698 * max(0.0, Q.mass - 74.25181)
            - 0.07463258 * max(0.0, Q.sd_mass - 86.4)
            + 0.02046421 * max(0.0, Q.mass_top10 - 71.781)
            - 0.06378087 * max(0.0, 14.14062 - Q.pt_11)
            + 3.843215 * max(0.0, Q.max_dr - 0.4357228)
            + 0.402659 * max(0.0, 1.976207 - Q.D2)
        ))
        + 0.15625 * grid(6, max(0.0, 1.528864
            + 0.03260267 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.0372454 * max(0.0, 120.6 - Q.mass)
            + 0.03564298 * max(0.0, 86.4 - Q.mass)
            + 0.0001010919 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt)
            - 0.08586285 * max(0.0, 101.0497 - Q.mass)
            + 0.06746218 * max(0.0, 92.85979 - Q.mass)
            - 6.208754 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 17.12618 * max(0.0, 0.06310829 - Q.tau1)
            - 13.78118 * max(0.0, Q.sj3_dr_min - 0.1204829)
            + 231.1054 * max(0.0, Q.girth2_top30 - 0.008376291)
            + 2203.189 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832)
            + 3022.732 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707)
            + 2.275593 * max(0.0, Q.sj3_dr_min - 0.1204829) * max(0.0, 7.863702 - Q.sj3_mass2)
            + 116.9532 * max(0.0, Q.girth2_top30 - 0.02809026)
            + 164.6479 * max(0.0, Q.girth2_top20 - 0.008031209)
            + 313.3979 * max(0.0, 0.003811746 - Q.lam1)
            - 43.46545 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722)
            - 1.351723 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2)
            - 337.6895 * max(0.0, 0.004573744 - Q.girth2_top50)
            + 0.01124433 * max(0.0, 91.19 - Q.mass_top15)
            - 36.05405 * max(0.0, Q.LHA - 0.3719813)
            - 1.050915 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.orientation_deg - 26.6454)
            + 0.007861364 * max(0.0, 172.8 - Q.mass_top50)
            - 7945.985 * max(0.0, Q.girth2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529)
            - 40.531 * max(0.0, Q.e2 - 0.02793599)
            - 31.71488 * max(0.0, 0.01807679 - Q.girth2_top30)
            - 26.50031 * max(0.0, Q.mass_over_sum_pt - 0.09795415)
        ))
        + 0.03125 * grid(8, max(0.0, 2.120228
            - 14.22432 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 17.78255 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.0724109 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.03722501 * max(0.0, Q.mass - 125.1)
            + 0.05697973 * max(0.0, Q.mass - 87.36377)
            + 0.4111008 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt)
            + 0.05891433 * max(0.0, Q.n_particles - 51.0)
            + 0.009781456 * max(0.0, 1017.435 - Q.sum_pt)
            - 974.6301 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.06636952 * max(0.0, Q.n_particles - 51.0) * max(0.0, 1.091797 - Q.soft1_pt)
            - 0.07660802 * max(0.0, Q.mass - 101.0497)
            + 239.2091 * max(0.0, Q.girth2_top20 - 0.008031209)
            - 12.99823 * max(0.0, 6.930088 - Q.log_sum_pt)
            - 195.7614 * max(0.0, Q.girth2_top20 - 0.006043209)
            + 0.01750729 * max(0.0, Q.mass - 64.48544)
            + 132.4598 * max(0.0, 0.007463985 - Q.girth2_top30)
            - 648.1557 * max(0.0, 0.009614971 - Q.width)
            + 175.6781 * max(0.0, 0.007877041 - Q.girth2)
            - 0.0584343 * max(0.0, Q.n_for_90pct - 7.0)
            - 107.4047 * max(0.0, Q.psi_0p3 - 0.9943058)
            + 189.9806 * max(0.0, 0.007671243 - Q.lam1)
            + 12.06935 * max(0.0, 0.0795038 - Q.tau2)
            - 1112.627 * max(0.0, Q.e3 - 5.13841e-05)
            - 0.03764843 * max(0.0, 39.0 - Q.n_for_90pct)
            - 0.04839883 * max(0.0, Q.mass_top50 - 82.04491)
            + 0.05231694 * max(0.0, Q.mass_top50 - 117.0487)
            + 0.04775024 * max(0.0, Q.mass - 74.25181)
            - 0.02550469 * max(0.0, Q.mass_top20 - 119.2969)
            + 0.0172691 * max(0.0, 1028.184 - Q.sum_pt)
            - 0.006395085 * max(0.0, 1024.942 - Q.sum_pt_top40)
            + 4.235687 * max(0.0, 0.8316924 - Q.z_top15_slots)
            + 2.532682 * max(0.0, 0.3628388 - Q.psi_0p1)
            - 0.3927259 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 0.3571579 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 132.7549 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 4925.494 * max(0.0, Q.e3 - 0.0003372339)
            - 26.55848 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
        ))
        + 0.515625 * grid(9, max(0.0, -0.6800824
            + 0.03173081 * max(0.0, 80.89043 - Q.mass_top40)
            + 0.006757492 * max(0.0, 956.2133 - Q.sum_pt_top40)
            + 315.7675 * max(0.0, 0.006259772 - Q.girth2_top40)
            + 7.232401 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177)
            - 0.003253904 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258)
            - 0.0002556704 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt)
            + 0.07821117 * max(0.0, 172.8 - Q.mass)
            - 0.1584773 * max(0.0, 160.8 - Q.mass)
            + 0.06618742 * max(0.0, 92.85979 - Q.mass)
            - 12.42092 * max(0.0, Q.girth - 0.09749958)
            - 0.09681596 * max(0.0, 143.7876 - Q.mass)
            + 6.542096 * max(0.0, Q.z_top5 - 0.7963975)
            + 0.0001718745 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40)
            + 0.01028863 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047)
            + 0.05485804 * max(0.0, 82.85409 - Q.mass)
            - 0.0878961 * max(0.0, 64.48544 - Q.mass)
            - 0.02411342 * max(0.0, 163.2541 - Q.mass_top40)
            + 75.21263 * max(0.0, Q.girth2_top30 - 0.0008564881)
            - 84.10431 * max(0.0, Q.girth - 0.1402186)
            - 0.09432579 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, 154.25 - Q.pt_2)
            + 0.127614 * max(0.0, 162.8363 - Q.mass)
            + 0.05001859 * max(0.0, 138.8977 - Q.mass_top50)
            + 0.01394976 * max(0.0, 73.33139 - Q.mass_top30)
            + 67.9024 * max(0.0, Q.LHA - 0.404204)
            - 3.076444 * max(0.0, Q.sj3_dr13 - 0.1654269)
            - 0.04130594 * max(0.0, 92.16545 - Q.mass_top50)
            - 3833.846 * max(0.0, Q.e3 - 0.0003372339)
            + 0.02278801 * max(0.0, 986.0565 - Q.sum_pt)
            - 13.1781 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
        - 0.015625 * grid(10, max(0.0, 0.1516575
            - 9.156365 * max(0.0, 0.1207452 - Q.girth)
            - 0.04117765 * max(0.0, Q.mass - 162.8363)
            - 0.01079364 * max(0.0, 65.20727 - Q.sj2_mass1)
            + 0.1971708 * max(0.0, 2.975532 - Q.D2)
            + 3.47822 * max(0.0, 0.5760704 - Q.z_top2_slots)
            - 0.008408502 * max(0.0, 59.40777 - Q.mass_top5)
            + 75.61714 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109)
            - 0.003694212 * max(0.0, 959.0957 - Q.sum_pt_top50)
            - 552.6863 * max(0.0, 0.002575211 - Q.girth2)
            - 0.04266331 * max(0.0, 101.0497 - Q.mass)
            + 7.608833 * max(0.0, 0.06413297 - Q.dr_0)
            - 8.225958 * max(0.0, Q.z_dr_0_0p05 - 0.7674734)
            - 0.1112269 * max(0.0, Q.mass_top50 - 172.8)
            - 22285.17 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.9985421)
            + 44.35472 * max(0.0, Q.e2 - 0.01256572)
            + 5259.916 * max(0.0, 0.0001086251 - Q.e3)
            - 0.03858257 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 0.1152859 * max(0.0, Q.mass - 143.7876)
            + 0.08793941 * max(0.0, Q.mass_top50 - 136.785)
            - 49.35608 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897)
            + 0.06253669 * max(0.0, Q.mass - 78.26182)
            + 7.189171 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 0.02327332 * max(0.0, 72.18744 - Q.mass_top15)
            + 89.15488 * max(0.0, 0.01655983 - Q.girth2_top20)
            - 0.04186768 * max(0.0, Q.mass - 64.48544)
            + 0.08707199 * max(0.0, 89.74183 - Q.mass)
            - 0.0108149 * max(0.0, Q.mass_top30 - 78.53034)
        ))
        + 0.234375 * grid(12, max(0.0, -0.5687879
            + 0.4328124 * max(0.0, Q.sd_mass - 125.1) * max(0.0, 0.4199841 - Q.sd_zg)
            - 5.354943 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            - 50.029 * max(0.0, 0.05077291 - Q.mass_over_sum_pt)
            + 0.001064185 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            + 0.04677431 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926)
            + 464.2041 * max(0.0, 0.0007431905 - Q.girth2_top10)
            + 0.07865754 * max(0.0, 82.85409 - Q.mass)
            - 0.02207065 * max(0.0, 74.78616 - Q.mass_top40)
            + 0.03257782 * max(0.0, 74.25181 - Q.mass)
            - 0.08413784 * max(0.0, 62.55 - Q.mass)
            + 13.10186 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 18.94421 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            - 18.48827 * max(0.0, Q.sd_mass - 125.1) * max(0.0, Q.lam2 - 0.0001679609)
            - 0.06440303 * max(0.0, Q.mass - 160.8)
        ))
    )


def logit_q(Q):
    return (1.359375
        - 0.1875 * grid(1, max(0.0, 1.48714
            + 0.09328952 * max(0.0, Q.n_particles - 38.0)
            + 21.0509 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.01273634 * max(0.0, Q.sum_pt_top50 - 959.0957)
            - 20.74087 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.00253462 * max(0.0, 689.25 - Q.sum_pt_top2)
            - 0.01626576 * max(0.0, 47.88842 - Q.mass_top20)
            - 0.02671432 * max(0.0, 32.50209 - Q.sj3_mass1)
            + 0.5237017 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793)
            + 0.005140426 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.3875843 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0)
            + 0.003762992 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0)
            - 0.03179651 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt)
            - 0.1279552 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            + 160.675 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2)
            + 1862.292 * max(0.0, 0.0005522528 - Q.girth2_top3)
            - 0.002737062 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2)
            - 44.64202 * max(0.0, 0.03187688 - Q.M3)
            - 0.02125498 * max(0.0, 31.35938 - Q.pt_9)
            + 22.84232 * max(0.0, Q.log_sum_pt - 6.893714)
            + 171.814 * max(0.0, 0.004673423 - Q.lam1)
            - 0.01121205 * max(0.0, 120.6 - Q.mass)
            + 54.39146 * max(0.0, 0.02412652 - Q.girth2_top30)
            - 0.007804931 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 0.9624339 - Q.tau43)
            - 5.901248 * max(0.0, 7.139296 - Q.log_sum_pt)
            - 2.742065 * max(0.0, Q.log_sum_pt - 6.959294)
            + 0.009289864 * max(0.0, 1017.435 - Q.sum_pt)
            + 0.529206 * max(0.0, Q.pt_entropy - 2.07371)
        ))
        + 0.125 * grid(2, max(0.0, 0.3078427
            + 0.009503978 * max(0.0, Q.sum_pt - 1017.435)
            + 647.1664 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15)
            - 0.003541556 * max(0.0, Q.sum_pt_top50 - 988.4554)
            - 0.4691034 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15)
            + 2.314562 * max(0.0, Q.z_top20_slots - 0.7516206)
            + 0.03207423 * max(0.0, 91.19 - Q.mass)
            - 0.005482459 * max(0.0, Q.sum_pt - 1115.723)
            + 175.842 * max(0.0, 0.004763596 - Q.girth2_top30)
            - 0.004625244 * max(0.0, 1260.541 - Q.sum_pt)
            - 0.07258039 * max(0.0, 92.85979 - Q.mass)
            + 74.57761 * max(0.0, 0.01174405 - Q.lam1)
            - 0.01053969 * max(0.0, 972.0419 - Q.sum_pt)
            - 7.395631 * max(0.0, Q.log_sum_pt - 6.959294)
            + 0.01051345 * max(0.0, 1048.098 - Q.sum_pt_top50)
            + 0.01188559 * max(0.0, 92.16545 - Q.mass_top50)
            - 12.724 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.08734131)
            + 0.004063689 * max(0.0, Q.sum_pt_top40 - 1069.671)
        ))
        - 1.0625 * grid(4, max(0.0, 1.367567
            - 0.02142609 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.03053927 * max(0.0, 60.43821 - Q.mass_top30)
            + 0.0201289 * max(0.0, 120.6 - Q.mass)
            - 0.02839266 * max(0.0, 86.4 - Q.mass)
            + 0.02532736 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.02472351 * max(0.0, Q.n_particles - 22.0)
            - 282.8053 * max(0.0, 0.004855289 - Q.girth2_top15)
            - 38.14918 * max(0.0, Q.girth2_top15 - 0.00727763)
            + 0.01708885 * max(0.0, 57.87349 - Q.mass_top15)
            + 0.05217597 * max(0.0, 101.0497 - Q.mass)
            - 0.03281671 * max(0.0, 80.78464 - Q.mass)
            + 0.08842908 * max(0.0, 74.25181 - Q.mass)
            - 0.008418597 * max(0.0, 163.2541 - Q.mass_top40)
            - 0.02998102 * max(0.0, 92.85979 - Q.mass)
            + 0.04484549 * max(0.0, 67.72643 - Q.mass_top40)
            - 0.1146276 * max(0.0, 78.26182 - Q.mass)
            - 13.37402 * max(0.0, Q.girth - 0.1207452)
            - 149.4077 * max(0.0, Q.lam1 - 0.008241985)
            + 55.23399 * max(0.0, Q.girth2_top15 - 0.01563836)
            + 167.0418 * max(0.0, Q.e2_sq - 0.009606007)
            - 143.5591 * max(0.0, Q.lam2 - 0.001776308)
        ))
        + 0.21875 * grid(6, max(0.0, 1.528864
            + 0.03260267 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.0372454 * max(0.0, 120.6 - Q.mass)
            + 0.03564298 * max(0.0, 86.4 - Q.mass)
            + 0.0001010919 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt)
            - 0.08586285 * max(0.0, 101.0497 - Q.mass)
            + 0.06746218 * max(0.0, 92.85979 - Q.mass)
            - 6.208754 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 17.12618 * max(0.0, 0.06310829 - Q.tau1)
            - 13.78118 * max(0.0, Q.sj3_dr_min - 0.1204829)
            + 231.1054 * max(0.0, Q.girth2_top30 - 0.008376291)
            + 2203.189 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832)
            + 3022.732 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707)
            + 2.275593 * max(0.0, Q.sj3_dr_min - 0.1204829) * max(0.0, 7.863702 - Q.sj3_mass2)
            + 116.9532 * max(0.0, Q.girth2_top30 - 0.02809026)
            + 164.6479 * max(0.0, Q.girth2_top20 - 0.008031209)
            + 313.3979 * max(0.0, 0.003811746 - Q.lam1)
            - 43.46545 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722)
            - 1.351723 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2)
            - 337.6895 * max(0.0, 0.004573744 - Q.girth2_top50)
            + 0.01124433 * max(0.0, 91.19 - Q.mass_top15)
            - 36.05405 * max(0.0, Q.LHA - 0.3719813)
            - 1.050915 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.orientation_deg - 26.6454)
            + 0.007861364 * max(0.0, 172.8 - Q.mass_top50)
            - 7945.985 * max(0.0, Q.girth2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529)
            - 40.531 * max(0.0, Q.e2 - 0.02793599)
            - 31.71488 * max(0.0, 0.01807679 - Q.girth2_top30)
            - 26.50031 * max(0.0, Q.mass_over_sum_pt - 0.09795415)
        ))
        + 0.015625 * grid(8, max(0.0, 2.120228
            - 14.22432 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 17.78255 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.0724109 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.03722501 * max(0.0, Q.mass - 125.1)
            + 0.05697973 * max(0.0, Q.mass - 87.36377)
            + 0.4111008 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt)
            + 0.05891433 * max(0.0, Q.n_particles - 51.0)
            + 0.009781456 * max(0.0, 1017.435 - Q.sum_pt)
            - 974.6301 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.06636952 * max(0.0, Q.n_particles - 51.0) * max(0.0, 1.091797 - Q.soft1_pt)
            - 0.07660802 * max(0.0, Q.mass - 101.0497)
            + 239.2091 * max(0.0, Q.girth2_top20 - 0.008031209)
            - 12.99823 * max(0.0, 6.930088 - Q.log_sum_pt)
            - 195.7614 * max(0.0, Q.girth2_top20 - 0.006043209)
            + 0.01750729 * max(0.0, Q.mass - 64.48544)
            + 132.4598 * max(0.0, 0.007463985 - Q.girth2_top30)
            - 648.1557 * max(0.0, 0.009614971 - Q.width)
            + 175.6781 * max(0.0, 0.007877041 - Q.girth2)
            - 0.0584343 * max(0.0, Q.n_for_90pct - 7.0)
            - 107.4047 * max(0.0, Q.psi_0p3 - 0.9943058)
            + 189.9806 * max(0.0, 0.007671243 - Q.lam1)
            + 12.06935 * max(0.0, 0.0795038 - Q.tau2)
            - 1112.627 * max(0.0, Q.e3 - 5.13841e-05)
            - 0.03764843 * max(0.0, 39.0 - Q.n_for_90pct)
            - 0.04839883 * max(0.0, Q.mass_top50 - 82.04491)
            + 0.05231694 * max(0.0, Q.mass_top50 - 117.0487)
            + 0.04775024 * max(0.0, Q.mass - 74.25181)
            - 0.02550469 * max(0.0, Q.mass_top20 - 119.2969)
            + 0.0172691 * max(0.0, 1028.184 - Q.sum_pt)
            - 0.006395085 * max(0.0, 1024.942 - Q.sum_pt_top40)
            + 4.235687 * max(0.0, 0.8316924 - Q.z_top15_slots)
            + 2.532682 * max(0.0, 0.3628388 - Q.psi_0p1)
            - 0.3927259 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 0.3571579 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 132.7549 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 4925.494 * max(0.0, Q.e3 - 0.0003372339)
            - 26.55848 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
        ))
        + 0.5625 * grid(9, max(0.0, -0.6800824
            + 0.03173081 * max(0.0, 80.89043 - Q.mass_top40)
            + 0.006757492 * max(0.0, 956.2133 - Q.sum_pt_top40)
            + 315.7675 * max(0.0, 0.006259772 - Q.girth2_top40)
            + 7.232401 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177)
            - 0.003253904 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258)
            - 0.0002556704 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt)
            + 0.07821117 * max(0.0, 172.8 - Q.mass)
            - 0.1584773 * max(0.0, 160.8 - Q.mass)
            + 0.06618742 * max(0.0, 92.85979 - Q.mass)
            - 12.42092 * max(0.0, Q.girth - 0.09749958)
            - 0.09681596 * max(0.0, 143.7876 - Q.mass)
            + 6.542096 * max(0.0, Q.z_top5 - 0.7963975)
            + 0.0001718745 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40)
            + 0.01028863 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047)
            + 0.05485804 * max(0.0, 82.85409 - Q.mass)
            - 0.0878961 * max(0.0, 64.48544 - Q.mass)
            - 0.02411342 * max(0.0, 163.2541 - Q.mass_top40)
            + 75.21263 * max(0.0, Q.girth2_top30 - 0.0008564881)
            - 84.10431 * max(0.0, Q.girth - 0.1402186)
            - 0.09432579 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, 154.25 - Q.pt_2)
            + 0.127614 * max(0.0, 162.8363 - Q.mass)
            + 0.05001859 * max(0.0, 138.8977 - Q.mass_top50)
            + 0.01394976 * max(0.0, 73.33139 - Q.mass_top30)
            + 67.9024 * max(0.0, Q.LHA - 0.404204)
            - 3.076444 * max(0.0, Q.sj3_dr13 - 0.1654269)
            - 0.04130594 * max(0.0, 92.16545 - Q.mass_top50)
            - 3833.846 * max(0.0, Q.e3 - 0.0003372339)
            + 0.02278801 * max(0.0, 986.0565 - Q.sum_pt)
            - 13.1781 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
        - 0.015625 * grid(10, max(0.0, 0.1516575
            - 9.156365 * max(0.0, 0.1207452 - Q.girth)
            - 0.04117765 * max(0.0, Q.mass - 162.8363)
            - 0.01079364 * max(0.0, 65.20727 - Q.sj2_mass1)
            + 0.1971708 * max(0.0, 2.975532 - Q.D2)
            + 3.47822 * max(0.0, 0.5760704 - Q.z_top2_slots)
            - 0.008408502 * max(0.0, 59.40777 - Q.mass_top5)
            + 75.61714 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109)
            - 0.003694212 * max(0.0, 959.0957 - Q.sum_pt_top50)
            - 552.6863 * max(0.0, 0.002575211 - Q.girth2)
            - 0.04266331 * max(0.0, 101.0497 - Q.mass)
            + 7.608833 * max(0.0, 0.06413297 - Q.dr_0)
            - 8.225958 * max(0.0, Q.z_dr_0_0p05 - 0.7674734)
            - 0.1112269 * max(0.0, Q.mass_top50 - 172.8)
            - 22285.17 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.9985421)
            + 44.35472 * max(0.0, Q.e2 - 0.01256572)
            + 5259.916 * max(0.0, 0.0001086251 - Q.e3)
            - 0.03858257 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 0.1152859 * max(0.0, Q.mass - 143.7876)
            + 0.08793941 * max(0.0, Q.mass_top50 - 136.785)
            - 49.35608 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897)
            + 0.06253669 * max(0.0, Q.mass - 78.26182)
            + 7.189171 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 0.02327332 * max(0.0, 72.18744 - Q.mass_top15)
            + 89.15488 * max(0.0, 0.01655983 - Q.girth2_top20)
            - 0.04186768 * max(0.0, Q.mass - 64.48544)
            + 0.08707199 * max(0.0, 89.74183 - Q.mass)
            - 0.0108149 * max(0.0, Q.mass_top30 - 78.53034)
        ))
        - 0.25 * grid(11, max(0.0, -0.6431159
            - 33.04289 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2)
            + 0.007455982 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 122.2319 * max(0.0, Q.psi_0p2 - 0.9935324)
            + 0.01069702 * max(0.0, 101.0497 - Q.mass)
            + 0.129551 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.3563114 - Q.planar_flow)
            - 0.07224298 * max(0.0, 80.4 - Q.mass)
            + 0.07976845 * max(0.0, 92.85979 - Q.mass)
            - 120.9615 * max(0.0, 0.00625621 - Q.girth2_top10)
            + 80.29568 * max(0.0, 0.02515919 - Q.e2)
            - 106.264 * max(0.0, 0.006929741 - Q.girth2_top30)
            + 161.4944 * max(0.0, 0.00406126 - Q.girth2_top10)
            - 0.02270075 * max(0.0, 83.32554 - Q.mass_top40)
            + 21.60515 * max(0.0, 0.08589404 - Q.girth)
            + 222.1106 * max(0.0, 0.00818374 - Q.e2_sq)
            - 40.08382 * max(0.0, 0.076787 - Q.girth)
            - 206.561 * max(0.0, 0.006716737 - Q.lam1)
            - 22559.03 * max(0.0, 3.793233e-05 - Q.e3)
            + 0.02105189 * max(0.0, 67.72643 - Q.mass_top40)
            - 7.908947 * max(0.0, Q.psi_0p2 - 0.9313699)
            + 56.30672 * max(0.0, 0.03029714 - Q.e2)
            - 0.02686474 * max(0.0, 80.35535 - Q.mass_top50)
            + 89.62396 * max(0.0, Q.psi_0p3 - 0.9896594)
        ))
        + 0.34375 * grid(12, max(0.0, -0.5687879
            + 0.4328124 * max(0.0, Q.sd_mass - 125.1) * max(0.0, 0.4199841 - Q.sd_zg)
            - 5.354943 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            - 50.029 * max(0.0, 0.05077291 - Q.mass_over_sum_pt)
            + 0.001064185 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            + 0.04677431 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926)
            + 464.2041 * max(0.0, 0.0007431905 - Q.girth2_top10)
            + 0.07865754 * max(0.0, 82.85409 - Q.mass)
            - 0.02207065 * max(0.0, 74.78616 - Q.mass_top40)
            + 0.03257782 * max(0.0, 74.25181 - Q.mass)
            - 0.08413784 * max(0.0, 62.55 - Q.mass)
            + 13.10186 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 18.94421 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            - 18.48827 * max(0.0, Q.sd_mass - 125.1) * max(0.0, Q.lam2 - 0.0001679609)
            - 0.06440303 * max(0.0, Q.mass - 160.8)
        ))
    )


def logit_W(Q):
    return (0.09375
        + 0.75 * grid(0, max(0.0, 1.348244
            - 0.145521 * max(0.0, Q.mass - 78.26182)
            - 0.08541639 * max(0.0, Q.mass - 92.85979)
            + 132.4016 * max(0.0, 0.005312783 - Q.girth2_top20)
            - 0.008300073 * max(0.0, 1012.673 - Q.sum_pt)
            - 0.04610753 * max(0.0, Q.mass - 91.19)
            - 0.01718703 * max(0.0, Q.mass - 74.25181)
            + 0.009118849 * max(0.0, 80.4 - Q.mass_top30)
            - 0.2622419 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3)
            + 123.2004 * max(0.0, 0.007856958 - Q.girth2_top30)
            - 151.5123 * max(0.0, 0.005913555 - Q.lam1)
            + 442.6132 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            - 27.28144 * max(0.0, 0.0705748 - Q.tau1)
            - 584.1328 * max(0.0, 0.00616708 - Q.e2_sq)
            - 0.05509232 * max(0.0, Q.mass_top50 - 82.04491)
            - 30.09608 * max(0.0, 0.9906378 - Q.z_top50_slots)
            - 0.02437571 * max(0.0, 101.0497 - Q.mass)
            + 208.2458 * max(0.0, 0.004754444 - Q.mass_over_sum_pt_sq)
            - 277.9319 * max(0.0, 0.006363916 - Q.girth2_top30)
            + 532.5071 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq)
            - 0.003375368 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.003710437 * max(0.0, 846.1934 - Q.sum_pt_top20)
            - 252.1068 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.0333566 * max(0.0, Q.mass_top50 - 71.79516)
            - 0.09426288 * max(0.0, Q.mass - 89.74183)
            + 0.01283502 * max(0.0, 858.8262 - Q.sum_pt_top40)
            + 0.004181412 * max(0.0, 1156.659 - Q.sum_pt_top50)
        ))
        + 0.4375 * grid(3, max(0.0, 0.06251741
            - 0.0393809 * max(0.0, 46.0 - Q.n_particles)
            + 0.0001463702 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732)
            - 6.400079 * max(0.0, 0.3861957 - Q.tau21)
            + 24.27948 * max(0.0, 0.08665515 - Q.mass_over_sum_pt)
            + 480.666 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 0.01364892 * max(0.0, 79.21004 - Q.mass_top50)
            - 147.3656 * max(0.0, 0.009614971 - Q.girth2)
            + 99.26788 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 0.01474629 * max(0.0, 92.85979 - Q.mass)
            - 0.00500493 * max(0.0, 79.21004 - Q.mass_top50) * max(0.0, Q.mass_top3 - 8.921413)
            + 0.0439486 * max(0.0, 27.56535 - Q.sj2_mass1)
            - 0.02451443 * max(0.0, 80.4 - Q.mass_top40)
            - 731.6249 * max(0.0, 0.001260456 - Q.lam1)
            + 150.2475 * max(0.0, 0.01626937 - Q.tau4)
        ))
        + 0.34375 * grid(4, max(0.0, 1.367567
            - 0.02142609 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.03053927 * max(0.0, 60.43821 - Q.mass_top30)
            + 0.0201289 * max(0.0, 120.6 - Q.mass)
            - 0.02839266 * max(0.0, 86.4 - Q.mass)
            + 0.02532736 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.02472351 * max(0.0, Q.n_particles - 22.0)
            - 282.8053 * max(0.0, 0.004855289 - Q.girth2_top15)
            - 38.14918 * max(0.0, Q.girth2_top15 - 0.00727763)
            + 0.01708885 * max(0.0, 57.87349 - Q.mass_top15)
            + 0.05217597 * max(0.0, 101.0497 - Q.mass)
            - 0.03281671 * max(0.0, 80.78464 - Q.mass)
            + 0.08842908 * max(0.0, 74.25181 - Q.mass)
            - 0.008418597 * max(0.0, 163.2541 - Q.mass_top40)
            - 0.02998102 * max(0.0, 92.85979 - Q.mass)
            + 0.04484549 * max(0.0, 67.72643 - Q.mass_top40)
            - 0.1146276 * max(0.0, 78.26182 - Q.mass)
            - 13.37402 * max(0.0, Q.girth - 0.1207452)
            - 149.4077 * max(0.0, Q.lam1 - 0.008241985)
            + 55.23399 * max(0.0, Q.girth2_top15 - 0.01563836)
            + 167.0418 * max(0.0, Q.e2_sq - 0.009606007)
            - 143.5591 * max(0.0, Q.lam2 - 0.001776308)
        ))
        + 0.578125 * grid(5, max(0.0, 0.6380697
            + 0.0381413 * max(0.0, 64.0 - Q.n_particles)
            - 10.20386 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 0.006261253 * max(0.0, Q.sum_pt - 907.9372)
            - 15.76866 * max(0.0, Q.log_sum_pt - 6.920349)
            + 167.2861 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.01647099 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.02016727 * max(0.0, Q.sum_pt_top50 - 934.2416)
            - 0.005608218 * max(0.0, Q.sum_pt_top20 - 1129.275)
            + 29743.12 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4)
            - 15.12077 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.01244614 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 9.362286 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.005879184 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 2.518499 * max(0.0, Q.max_dr - 0.2404747)
            - 14.75085 * max(0.0, Q.z_top30_slots - 0.9048492)
            - 0.02569512 * max(0.0, Q.mass_top40 - 120.6)
            + 0.07803723 * max(0.0, Q.mass - 172.8)
            - 115.5639 * max(0.0, Q.girth2_top30 - 0.006363916)
            - 55.14108 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2)
            - 0.005410795 * max(0.0, 150.0144 - Q.mass_top40)
            - 511.5796 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            - 2.370548 * max(0.0, 0.5494307 - Q.tau21)
            + 0.05009839 * max(0.0, Q.sd_mass - 69.65633)
            - 0.1307292 * max(0.0, Q.mass_top50 - 157.5448)
            - 0.004752506 * max(0.0, Q.sum_pt - 986.0565)
            + 84.69973 * max(0.0, 0.01375115 - Q.z_11)
            + 0.01644698 * max(0.0, Q.mass - 74.25181)
            - 0.07463258 * max(0.0, Q.sd_mass - 86.4)
            + 0.02046421 * max(0.0, Q.mass_top10 - 71.781)
            - 0.06378087 * max(0.0, 14.14062 - Q.pt_11)
            + 3.843215 * max(0.0, Q.max_dr - 0.4357228)
            + 0.402659 * max(0.0, 1.976207 - Q.D2)
        ))
        + 0.0625 * grid(6, max(0.0, 1.528864
            + 0.03260267 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.0372454 * max(0.0, 120.6 - Q.mass)
            + 0.03564298 * max(0.0, 86.4 - Q.mass)
            + 0.0001010919 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt)
            - 0.08586285 * max(0.0, 101.0497 - Q.mass)
            + 0.06746218 * max(0.0, 92.85979 - Q.mass)
            - 6.208754 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 17.12618 * max(0.0, 0.06310829 - Q.tau1)
            - 13.78118 * max(0.0, Q.sj3_dr_min - 0.1204829)
            + 231.1054 * max(0.0, Q.girth2_top30 - 0.008376291)
            + 2203.189 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832)
            + 3022.732 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707)
            + 2.275593 * max(0.0, Q.sj3_dr_min - 0.1204829) * max(0.0, 7.863702 - Q.sj3_mass2)
            + 116.9532 * max(0.0, Q.girth2_top30 - 0.02809026)
            + 164.6479 * max(0.0, Q.girth2_top20 - 0.008031209)
            + 313.3979 * max(0.0, 0.003811746 - Q.lam1)
            - 43.46545 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722)
            - 1.351723 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2)
            - 337.6895 * max(0.0, 0.004573744 - Q.girth2_top50)
            + 0.01124433 * max(0.0, 91.19 - Q.mass_top15)
            - 36.05405 * max(0.0, Q.LHA - 0.3719813)
            - 1.050915 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.orientation_deg - 26.6454)
            + 0.007861364 * max(0.0, 172.8 - Q.mass_top50)
            - 7945.985 * max(0.0, Q.girth2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529)
            - 40.531 * max(0.0, Q.e2 - 0.02793599)
            - 31.71488 * max(0.0, 0.01807679 - Q.girth2_top30)
            - 26.50031 * max(0.0, Q.mass_over_sum_pt - 0.09795415)
        ))
        - 0.625 * grid(7, max(0.0, 0.02822393
            + 15.81855 * max(0.0, 0.2352054 - Q.tau21_b2)
            - 2245.457 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            + 8.27917 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            - 55.89039 * max(0.0, 0.006403325 - Q.girth2)
            - 0.2412638 * max(0.0, 82.85409 - Q.mass)
            - 0.06299596 * max(0.0, 101.0497 - Q.mass)
            + 0.07985738 * max(0.0, 6.0 - Q.n_dr_0p2_0p4)
            - 1634.034 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 7.807953 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            + 10.95938 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            + 4.81802 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            - 19.21488 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            - 109.9719 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            - 66.66523 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            + 123.7156 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            - 0.02163195 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt)
            + 0.04028611 * max(0.0, 120.6 - Q.mass)
            - 8.922787 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 6.339537 * max(0.0, 0.3861957 - Q.tau21)
            + 152371.2 * max(0.0, 0.006403325 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.9980008)
            - 817.5024 * max(0.0, 0.00363788 - Q.e2_sq)
            + 0.182892 * max(0.0, 78.26182 - Q.mass)
            - 317.4453 * max(0.0, 0.008190222 - Q.girth2)
            + 408.5902 * max(0.0, 0.009614971 - Q.girth2)
            - 42.6427 * max(0.0, 0.1072713 - Q.tau1)
            + 27.09727 * max(0.0, 0.09591084 - Q.tau1)
            - 353.4245 * max(0.0, 0.008241985 - Q.lam1)
            + 346.7997 * max(0.0, 0.006189818 - Q.lam1)
            + 548.1077 * max(0.0, Q.psi_0p3 - 0.9973959)
            + 0.04255483 * max(0.0, 69.65633 - Q.sd_mass)
            - 0.03352977 * max(0.0, 86.4 - Q.sd_mass)
        ))
        - 0.875 * grid(8, max(0.0, 2.120228
            - 14.22432 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 17.78255 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.0724109 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.03722501 * max(0.0, Q.mass - 125.1)
            + 0.05697973 * max(0.0, Q.mass - 87.36377)
            + 0.4111008 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt)
            + 0.05891433 * max(0.0, Q.n_particles - 51.0)
            + 0.009781456 * max(0.0, 1017.435 - Q.sum_pt)
            - 974.6301 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.06636952 * max(0.0, Q.n_particles - 51.0) * max(0.0, 1.091797 - Q.soft1_pt)
            - 0.07660802 * max(0.0, Q.mass - 101.0497)
            + 239.2091 * max(0.0, Q.girth2_top20 - 0.008031209)
            - 12.99823 * max(0.0, 6.930088 - Q.log_sum_pt)
            - 195.7614 * max(0.0, Q.girth2_top20 - 0.006043209)
            + 0.01750729 * max(0.0, Q.mass - 64.48544)
            + 132.4598 * max(0.0, 0.007463985 - Q.girth2_top30)
            - 648.1557 * max(0.0, 0.009614971 - Q.width)
            + 175.6781 * max(0.0, 0.007877041 - Q.girth2)
            - 0.0584343 * max(0.0, Q.n_for_90pct - 7.0)
            - 107.4047 * max(0.0, Q.psi_0p3 - 0.9943058)
            + 189.9806 * max(0.0, 0.007671243 - Q.lam1)
            + 12.06935 * max(0.0, 0.0795038 - Q.tau2)
            - 1112.627 * max(0.0, Q.e3 - 5.13841e-05)
            - 0.03764843 * max(0.0, 39.0 - Q.n_for_90pct)
            - 0.04839883 * max(0.0, Q.mass_top50 - 82.04491)
            + 0.05231694 * max(0.0, Q.mass_top50 - 117.0487)
            + 0.04775024 * max(0.0, Q.mass - 74.25181)
            - 0.02550469 * max(0.0, Q.mass_top20 - 119.2969)
            + 0.0172691 * max(0.0, 1028.184 - Q.sum_pt)
            - 0.006395085 * max(0.0, 1024.942 - Q.sum_pt_top40)
            + 4.235687 * max(0.0, 0.8316924 - Q.z_top15_slots)
            + 2.532682 * max(0.0, 0.3628388 - Q.psi_0p1)
            - 0.3927259 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 0.3571579 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 132.7549 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 4925.494 * max(0.0, Q.e3 - 0.0003372339)
            - 26.55848 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
        ))
        - 0.21875 * grid(9, max(0.0, -0.6800824
            + 0.03173081 * max(0.0, 80.89043 - Q.mass_top40)
            + 0.006757492 * max(0.0, 956.2133 - Q.sum_pt_top40)
            + 315.7675 * max(0.0, 0.006259772 - Q.girth2_top40)
            + 7.232401 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177)
            - 0.003253904 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258)
            - 0.0002556704 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt)
            + 0.07821117 * max(0.0, 172.8 - Q.mass)
            - 0.1584773 * max(0.0, 160.8 - Q.mass)
            + 0.06618742 * max(0.0, 92.85979 - Q.mass)
            - 12.42092 * max(0.0, Q.girth - 0.09749958)
            - 0.09681596 * max(0.0, 143.7876 - Q.mass)
            + 6.542096 * max(0.0, Q.z_top5 - 0.7963975)
            + 0.0001718745 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40)
            + 0.01028863 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047)
            + 0.05485804 * max(0.0, 82.85409 - Q.mass)
            - 0.0878961 * max(0.0, 64.48544 - Q.mass)
            - 0.02411342 * max(0.0, 163.2541 - Q.mass_top40)
            + 75.21263 * max(0.0, Q.girth2_top30 - 0.0008564881)
            - 84.10431 * max(0.0, Q.girth - 0.1402186)
            - 0.09432579 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, 154.25 - Q.pt_2)
            + 0.127614 * max(0.0, 162.8363 - Q.mass)
            + 0.05001859 * max(0.0, 138.8977 - Q.mass_top50)
            + 0.01394976 * max(0.0, 73.33139 - Q.mass_top30)
            + 67.9024 * max(0.0, Q.LHA - 0.404204)
            - 3.076444 * max(0.0, Q.sj3_dr13 - 0.1654269)
            - 0.04130594 * max(0.0, 92.16545 - Q.mass_top50)
            - 3833.846 * max(0.0, Q.e3 - 0.0003372339)
            + 0.02278801 * max(0.0, 986.0565 - Q.sum_pt)
            - 13.1781 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
        + 0.59375 * grid(11, max(0.0, -0.6431159
            - 33.04289 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2)
            + 0.007455982 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 122.2319 * max(0.0, Q.psi_0p2 - 0.9935324)
            + 0.01069702 * max(0.0, 101.0497 - Q.mass)
            + 0.129551 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.3563114 - Q.planar_flow)
            - 0.07224298 * max(0.0, 80.4 - Q.mass)
            + 0.07976845 * max(0.0, 92.85979 - Q.mass)
            - 120.9615 * max(0.0, 0.00625621 - Q.girth2_top10)
            + 80.29568 * max(0.0, 0.02515919 - Q.e2)
            - 106.264 * max(0.0, 0.006929741 - Q.girth2_top30)
            + 161.4944 * max(0.0, 0.00406126 - Q.girth2_top10)
            - 0.02270075 * max(0.0, 83.32554 - Q.mass_top40)
            + 21.60515 * max(0.0, 0.08589404 - Q.girth)
            + 222.1106 * max(0.0, 0.00818374 - Q.e2_sq)
            - 40.08382 * max(0.0, 0.076787 - Q.girth)
            - 206.561 * max(0.0, 0.006716737 - Q.lam1)
            - 22559.03 * max(0.0, 3.793233e-05 - Q.e3)
            + 0.02105189 * max(0.0, 67.72643 - Q.mass_top40)
            - 7.908947 * max(0.0, Q.psi_0p2 - 0.9313699)
            + 56.30672 * max(0.0, 0.03029714 - Q.e2)
            - 0.02686474 * max(0.0, 80.35535 - Q.mass_top50)
            + 89.62396 * max(0.0, Q.psi_0p3 - 0.9896594)
        ))
        - 0.40625 * grid(12, max(0.0, -0.5687879
            + 0.4328124 * max(0.0, Q.sd_mass - 125.1) * max(0.0, 0.4199841 - Q.sd_zg)
            - 5.354943 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            - 50.029 * max(0.0, 0.05077291 - Q.mass_over_sum_pt)
            + 0.001064185 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            + 0.04677431 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926)
            + 464.2041 * max(0.0, 0.0007431905 - Q.girth2_top10)
            + 0.07865754 * max(0.0, 82.85409 - Q.mass)
            - 0.02207065 * max(0.0, 74.78616 - Q.mass_top40)
            + 0.03257782 * max(0.0, 74.25181 - Q.mass)
            - 0.08413784 * max(0.0, 62.55 - Q.mass)
            + 13.10186 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 18.94421 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            - 18.48827 * max(0.0, Q.sd_mass - 125.1) * max(0.0, Q.lam2 - 0.0001679609)
            - 0.06440303 * max(0.0, Q.mass - 160.8)
        ))
        - 1.375 * grid(14, max(0.0, -0.6994699
            - 1660.737 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.007709916 - Q.girth2_top40)
            - 0.1328738 * max(0.0, 91.19 - Q.mass)
            - 0.1219697 * max(0.0, 82.85409 - Q.mass)
            + 163.3773 * max(0.0, Q.psi_0p3 - 0.9924477)
            - 10.17149 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2404747)
            + 40.43768 * max(0.0, 0.03029714 - Q.e2)
            - 63.21893 * max(0.0, 0.09046749 - Q.mass_over_sum_pt)
            + 53.30122 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            - 916.5762 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1881908 - Q.sd_rg)
            - 86.9298 * max(0.0, 0.01083435 - Q.girth2_top20)
            + 40.01744 * max(0.0, 0.08873143 - Q.mass_over_sum_pt)
            - 0.02436746 * max(0.0, 80.4 - Q.mass)
            - 10.77636 * max(0.0, 0.076787 - Q.girth)
            + 129.0502 * max(0.0, 0.007164202 - Q.girth2_top5)
            + 37.35547 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            + 109.5338 * max(0.0, 0.006374178 - Q.girth2_top20)
            - 3.523758 * max(0.0, 0.3017146 - Q.sd_rg)
            - 75.44558 * max(0.0, 0.07852883 - Q.mass_over_sum_pt)
            + 12.16646 * max(0.0, 0.140939 - Q.mass_over_sum_pt)
            - 75.38352 * max(0.0, 0.01215787 - Q.girth2_top30)
            + 49.67054 * max(0.0, 0.06310829 - Q.tau1)
            + 21.51449 * max(0.0, 6.941997 - Q.log_sum_pt)
            - 0.03000849 * max(0.0, 976.277 - Q.sum_pt_top50)
            - 2.046417 * max(0.0, 6.98945 - Q.log_sum_pt)
            - 7.44323 * max(0.0, 0.007164202 - Q.girth2_top5) * max(0.0, Q.n_pt_above_10 - 13.0)
            - 267.7451 * max(0.0, 0.076787 - Q.girth) * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4)
            + 5.066188 * max(0.0, 0.1778185 - Q.sd_rg)
            + 0.004099871 * max(0.0, 1024.942 - Q.sum_pt_top40)
        ))
        - 0.1875 * grid(15, max(0.0, -0.4554918
            + 0.6593533 * max(0.0, 0.8459004 - Q.z_dr_0_0p05)
            + 10.74733 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2)
            + 91.84845 * max(0.0, 0.008329695 - Q.girth2_top5)
            + 0.008982757 * max(0.0, 986.0565 - Q.sum_pt)
            + 6.495682 * max(0.0, 7.017258 - Q.log_sum_pt)
            - 72.05958 * max(0.0, Q.psi_0p3 - 0.9896594)
            - 0.004182898 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 1260.541 - Q.sum_pt)
            - 0.4257766 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0)
            - 0.006464618 * max(0.0, 1002.379 - Q.sum_pt)
            - 288.0632 * max(0.0, 0.002270363 - Q.girth2_top5)
            + 390.9258 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2)
            + 100.6571 * max(0.0, 0.007678544 - Q.girth2_top10)
            - 3.407665 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677)
            - 16.10477 * max(0.0, 0.0705748 - Q.tau1)
            + 0.01923916 * max(0.0, 79.18312 - Q.sd_mass)
            - 0.02277408 * max(0.0, 45.595 - Q.sd_mass)
            - 0.006371291 * max(0.0, 935.8189 - Q.sum_pt_top40)
            - 0.005431578 * max(0.0, 1018.698 - Q.sum_pt_top40)
            - 9.187314 * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            + 0.005659686 * max(0.0, 933.1875 - Q.sum_pt_top30)
            - 5.053077 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
    )


def logit_Z(Q):
    return (0.984375
        - 1.375 * grid(0, max(0.0, 1.348244
            - 0.145521 * max(0.0, Q.mass - 78.26182)
            - 0.08541639 * max(0.0, Q.mass - 92.85979)
            + 132.4016 * max(0.0, 0.005312783 - Q.girth2_top20)
            - 0.008300073 * max(0.0, 1012.673 - Q.sum_pt)
            - 0.04610753 * max(0.0, Q.mass - 91.19)
            - 0.01718703 * max(0.0, Q.mass - 74.25181)
            + 0.009118849 * max(0.0, 80.4 - Q.mass_top30)
            - 0.2622419 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3)
            + 123.2004 * max(0.0, 0.007856958 - Q.girth2_top30)
            - 151.5123 * max(0.0, 0.005913555 - Q.lam1)
            + 442.6132 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            - 27.28144 * max(0.0, 0.0705748 - Q.tau1)
            - 584.1328 * max(0.0, 0.00616708 - Q.e2_sq)
            - 0.05509232 * max(0.0, Q.mass_top50 - 82.04491)
            - 30.09608 * max(0.0, 0.9906378 - Q.z_top50_slots)
            - 0.02437571 * max(0.0, 101.0497 - Q.mass)
            + 208.2458 * max(0.0, 0.004754444 - Q.mass_over_sum_pt_sq)
            - 277.9319 * max(0.0, 0.006363916 - Q.girth2_top30)
            + 532.5071 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq)
            - 0.003375368 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.003710437 * max(0.0, 846.1934 - Q.sum_pt_top20)
            - 252.1068 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.0333566 * max(0.0, Q.mass_top50 - 71.79516)
            - 0.09426288 * max(0.0, Q.mass - 89.74183)
            + 0.01283502 * max(0.0, 858.8262 - Q.sum_pt_top40)
            + 0.004181412 * max(0.0, 1156.659 - Q.sum_pt_top50)
        ))
        - 0.375 * grid(2, max(0.0, 0.3078427
            + 0.009503978 * max(0.0, Q.sum_pt - 1017.435)
            + 647.1664 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15)
            - 0.003541556 * max(0.0, Q.sum_pt_top50 - 988.4554)
            - 0.4691034 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15)
            + 2.314562 * max(0.0, Q.z_top20_slots - 0.7516206)
            + 0.03207423 * max(0.0, 91.19 - Q.mass)
            - 0.005482459 * max(0.0, Q.sum_pt - 1115.723)
            + 175.842 * max(0.0, 0.004763596 - Q.girth2_top30)
            - 0.004625244 * max(0.0, 1260.541 - Q.sum_pt)
            - 0.07258039 * max(0.0, 92.85979 - Q.mass)
            + 74.57761 * max(0.0, 0.01174405 - Q.lam1)
            - 0.01053969 * max(0.0, 972.0419 - Q.sum_pt)
            - 7.395631 * max(0.0, Q.log_sum_pt - 6.959294)
            + 0.01051345 * max(0.0, 1048.098 - Q.sum_pt_top50)
            + 0.01188559 * max(0.0, 92.16545 - Q.mass_top50)
            - 12.724 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.08734131)
            + 0.004063689 * max(0.0, Q.sum_pt_top40 - 1069.671)
        ))
        + 0.4375 * grid(3, max(0.0, 0.06251741
            - 0.0393809 * max(0.0, 46.0 - Q.n_particles)
            + 0.0001463702 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732)
            - 6.400079 * max(0.0, 0.3861957 - Q.tau21)
            + 24.27948 * max(0.0, 0.08665515 - Q.mass_over_sum_pt)
            + 480.666 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 0.01364892 * max(0.0, 79.21004 - Q.mass_top50)
            - 147.3656 * max(0.0, 0.009614971 - Q.girth2)
            + 99.26788 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 0.01474629 * max(0.0, 92.85979 - Q.mass)
            - 0.00500493 * max(0.0, 79.21004 - Q.mass_top50) * max(0.0, Q.mass_top3 - 8.921413)
            + 0.0439486 * max(0.0, 27.56535 - Q.sj2_mass1)
            - 0.02451443 * max(0.0, 80.4 - Q.mass_top40)
            - 731.6249 * max(0.0, 0.001260456 - Q.lam1)
            + 150.2475 * max(0.0, 0.01626937 - Q.tau4)
        ))
        + 0.6875 * grid(5, max(0.0, 0.6380697
            + 0.0381413 * max(0.0, 64.0 - Q.n_particles)
            - 10.20386 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 0.006261253 * max(0.0, Q.sum_pt - 907.9372)
            - 15.76866 * max(0.0, Q.log_sum_pt - 6.920349)
            + 167.2861 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.01647099 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.02016727 * max(0.0, Q.sum_pt_top50 - 934.2416)
            - 0.005608218 * max(0.0, Q.sum_pt_top20 - 1129.275)
            + 29743.12 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4)
            - 15.12077 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.01244614 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 9.362286 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.005879184 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 2.518499 * max(0.0, Q.max_dr - 0.2404747)
            - 14.75085 * max(0.0, Q.z_top30_slots - 0.9048492)
            - 0.02569512 * max(0.0, Q.mass_top40 - 120.6)
            + 0.07803723 * max(0.0, Q.mass - 172.8)
            - 115.5639 * max(0.0, Q.girth2_top30 - 0.006363916)
            - 55.14108 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2)
            - 0.005410795 * max(0.0, 150.0144 - Q.mass_top40)
            - 511.5796 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            - 2.370548 * max(0.0, 0.5494307 - Q.tau21)
            + 0.05009839 * max(0.0, Q.sd_mass - 69.65633)
            - 0.1307292 * max(0.0, Q.mass_top50 - 157.5448)
            - 0.004752506 * max(0.0, Q.sum_pt - 986.0565)
            + 84.69973 * max(0.0, 0.01375115 - Q.z_11)
            + 0.01644698 * max(0.0, Q.mass - 74.25181)
            - 0.07463258 * max(0.0, Q.sd_mass - 86.4)
            + 0.02046421 * max(0.0, Q.mass_top10 - 71.781)
            - 0.06378087 * max(0.0, 14.14062 - Q.pt_11)
            + 3.843215 * max(0.0, Q.max_dr - 0.4357228)
            + 0.402659 * max(0.0, 1.976207 - Q.D2)
        ))
        - 0.96875 * grid(6, max(0.0, 1.528864
            + 0.03260267 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.0372454 * max(0.0, 120.6 - Q.mass)
            + 0.03564298 * max(0.0, 86.4 - Q.mass)
            + 0.0001010919 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt)
            - 0.08586285 * max(0.0, 101.0497 - Q.mass)
            + 0.06746218 * max(0.0, 92.85979 - Q.mass)
            - 6.208754 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 17.12618 * max(0.0, 0.06310829 - Q.tau1)
            - 13.78118 * max(0.0, Q.sj3_dr_min - 0.1204829)
            + 231.1054 * max(0.0, Q.girth2_top30 - 0.008376291)
            + 2203.189 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832)
            + 3022.732 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707)
            + 2.275593 * max(0.0, Q.sj3_dr_min - 0.1204829) * max(0.0, 7.863702 - Q.sj3_mass2)
            + 116.9532 * max(0.0, Q.girth2_top30 - 0.02809026)
            + 164.6479 * max(0.0, Q.girth2_top20 - 0.008031209)
            + 313.3979 * max(0.0, 0.003811746 - Q.lam1)
            - 43.46545 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722)
            - 1.351723 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2)
            - 337.6895 * max(0.0, 0.004573744 - Q.girth2_top50)
            + 0.01124433 * max(0.0, 91.19 - Q.mass_top15)
            - 36.05405 * max(0.0, Q.LHA - 0.3719813)
            - 1.050915 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.orientation_deg - 26.6454)
            + 0.007861364 * max(0.0, 172.8 - Q.mass_top50)
            - 7945.985 * max(0.0, Q.girth2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529)
            - 40.531 * max(0.0, Q.e2 - 0.02793599)
            - 31.71488 * max(0.0, 0.01807679 - Q.girth2_top30)
            - 26.50031 * max(0.0, Q.mass_over_sum_pt - 0.09795415)
        ))
        + 0.90625 * grid(7, max(0.0, 0.02822393
            + 15.81855 * max(0.0, 0.2352054 - Q.tau21_b2)
            - 2245.457 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            + 8.27917 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            - 55.89039 * max(0.0, 0.006403325 - Q.girth2)
            - 0.2412638 * max(0.0, 82.85409 - Q.mass)
            - 0.06299596 * max(0.0, 101.0497 - Q.mass)
            + 0.07985738 * max(0.0, 6.0 - Q.n_dr_0p2_0p4)
            - 1634.034 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 7.807953 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            + 10.95938 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            + 4.81802 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            - 19.21488 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            - 109.9719 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            - 66.66523 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            + 123.7156 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            - 0.02163195 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt)
            + 0.04028611 * max(0.0, 120.6 - Q.mass)
            - 8.922787 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 6.339537 * max(0.0, 0.3861957 - Q.tau21)
            + 152371.2 * max(0.0, 0.006403325 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.9980008)
            - 817.5024 * max(0.0, 0.00363788 - Q.e2_sq)
            + 0.182892 * max(0.0, 78.26182 - Q.mass)
            - 317.4453 * max(0.0, 0.008190222 - Q.girth2)
            + 408.5902 * max(0.0, 0.009614971 - Q.girth2)
            - 42.6427 * max(0.0, 0.1072713 - Q.tau1)
            + 27.09727 * max(0.0, 0.09591084 - Q.tau1)
            - 353.4245 * max(0.0, 0.008241985 - Q.lam1)
            + 346.7997 * max(0.0, 0.006189818 - Q.lam1)
            + 548.1077 * max(0.0, Q.psi_0p3 - 0.9973959)
            + 0.04255483 * max(0.0, 69.65633 - Q.sd_mass)
            - 0.03352977 * max(0.0, 86.4 - Q.sd_mass)
        ))
        - 0.9375 * grid(8, max(0.0, 2.120228
            - 14.22432 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 17.78255 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.0724109 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.03722501 * max(0.0, Q.mass - 125.1)
            + 0.05697973 * max(0.0, Q.mass - 87.36377)
            + 0.4111008 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt)
            + 0.05891433 * max(0.0, Q.n_particles - 51.0)
            + 0.009781456 * max(0.0, 1017.435 - Q.sum_pt)
            - 974.6301 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.06636952 * max(0.0, Q.n_particles - 51.0) * max(0.0, 1.091797 - Q.soft1_pt)
            - 0.07660802 * max(0.0, Q.mass - 101.0497)
            + 239.2091 * max(0.0, Q.girth2_top20 - 0.008031209)
            - 12.99823 * max(0.0, 6.930088 - Q.log_sum_pt)
            - 195.7614 * max(0.0, Q.girth2_top20 - 0.006043209)
            + 0.01750729 * max(0.0, Q.mass - 64.48544)
            + 132.4598 * max(0.0, 0.007463985 - Q.girth2_top30)
            - 648.1557 * max(0.0, 0.009614971 - Q.width)
            + 175.6781 * max(0.0, 0.007877041 - Q.girth2)
            - 0.0584343 * max(0.0, Q.n_for_90pct - 7.0)
            - 107.4047 * max(0.0, Q.psi_0p3 - 0.9943058)
            + 189.9806 * max(0.0, 0.007671243 - Q.lam1)
            + 12.06935 * max(0.0, 0.0795038 - Q.tau2)
            - 1112.627 * max(0.0, Q.e3 - 5.13841e-05)
            - 0.03764843 * max(0.0, 39.0 - Q.n_for_90pct)
            - 0.04839883 * max(0.0, Q.mass_top50 - 82.04491)
            + 0.05231694 * max(0.0, Q.mass_top50 - 117.0487)
            + 0.04775024 * max(0.0, Q.mass - 74.25181)
            - 0.02550469 * max(0.0, Q.mass_top20 - 119.2969)
            + 0.0172691 * max(0.0, 1028.184 - Q.sum_pt)
            - 0.006395085 * max(0.0, 1024.942 - Q.sum_pt_top40)
            + 4.235687 * max(0.0, 0.8316924 - Q.z_top15_slots)
            + 2.532682 * max(0.0, 0.3628388 - Q.psi_0p1)
            - 0.3927259 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 0.3571579 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 132.7549 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 4925.494 * max(0.0, Q.e3 - 0.0003372339)
            - 26.55848 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
        ))
        + 0.3125 * grid(12, max(0.0, -0.5687879
            + 0.4328124 * max(0.0, Q.sd_mass - 125.1) * max(0.0, 0.4199841 - Q.sd_zg)
            - 5.354943 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            - 50.029 * max(0.0, 0.05077291 - Q.mass_over_sum_pt)
            + 0.001064185 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            + 0.04677431 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926)
            + 464.2041 * max(0.0, 0.0007431905 - Q.girth2_top10)
            + 0.07865754 * max(0.0, 82.85409 - Q.mass)
            - 0.02207065 * max(0.0, 74.78616 - Q.mass_top40)
            + 0.03257782 * max(0.0, 74.25181 - Q.mass)
            - 0.08413784 * max(0.0, 62.55 - Q.mass)
            + 13.10186 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 18.94421 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            - 18.48827 * max(0.0, Q.sd_mass - 125.1) * max(0.0, Q.lam2 - 0.0001679609)
            - 0.06440303 * max(0.0, Q.mass - 160.8)
        ))
        + 0.5625 * grid(15, max(0.0, -0.4554918
            + 0.6593533 * max(0.0, 0.8459004 - Q.z_dr_0_0p05)
            + 10.74733 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2)
            + 91.84845 * max(0.0, 0.008329695 - Q.girth2_top5)
            + 0.008982757 * max(0.0, 986.0565 - Q.sum_pt)
            + 6.495682 * max(0.0, 7.017258 - Q.log_sum_pt)
            - 72.05958 * max(0.0, Q.psi_0p3 - 0.9896594)
            - 0.004182898 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 1260.541 - Q.sum_pt)
            - 0.4257766 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0)
            - 0.006464618 * max(0.0, 1002.379 - Q.sum_pt)
            - 288.0632 * max(0.0, 0.002270363 - Q.girth2_top5)
            + 390.9258 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2)
            + 100.6571 * max(0.0, 0.007678544 - Q.girth2_top10)
            - 3.407665 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677)
            - 16.10477 * max(0.0, 0.0705748 - Q.tau1)
            + 0.01923916 * max(0.0, 79.18312 - Q.sd_mass)
            - 0.02277408 * max(0.0, 45.595 - Q.sd_mass)
            - 0.006371291 * max(0.0, 935.8189 - Q.sum_pt_top40)
            - 0.005431578 * max(0.0, 1018.698 - Q.sum_pt_top40)
            - 9.187314 * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            + 0.005659686 * max(0.0, 933.1875 - Q.sum_pt_top30)
            - 5.053077 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
    )


def logit_t(Q):
    return (0.78125
        + 0.125 * grid(0, max(0.0, 1.348244
            - 0.145521 * max(0.0, Q.mass - 78.26182)
            - 0.08541639 * max(0.0, Q.mass - 92.85979)
            + 132.4016 * max(0.0, 0.005312783 - Q.girth2_top20)
            - 0.008300073 * max(0.0, 1012.673 - Q.sum_pt)
            - 0.04610753 * max(0.0, Q.mass - 91.19)
            - 0.01718703 * max(0.0, Q.mass - 74.25181)
            + 0.009118849 * max(0.0, 80.4 - Q.mass_top30)
            - 0.2622419 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3)
            + 123.2004 * max(0.0, 0.007856958 - Q.girth2_top30)
            - 151.5123 * max(0.0, 0.005913555 - Q.lam1)
            + 442.6132 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            - 27.28144 * max(0.0, 0.0705748 - Q.tau1)
            - 584.1328 * max(0.0, 0.00616708 - Q.e2_sq)
            - 0.05509232 * max(0.0, Q.mass_top50 - 82.04491)
            - 30.09608 * max(0.0, 0.9906378 - Q.z_top50_slots)
            - 0.02437571 * max(0.0, 101.0497 - Q.mass)
            + 208.2458 * max(0.0, 0.004754444 - Q.mass_over_sum_pt_sq)
            - 277.9319 * max(0.0, 0.006363916 - Q.girth2_top30)
            + 532.5071 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq)
            - 0.003375368 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.003710437 * max(0.0, 846.1934 - Q.sum_pt_top20)
            - 252.1068 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.0333566 * max(0.0, Q.mass_top50 - 71.79516)
            - 0.09426288 * max(0.0, Q.mass - 89.74183)
            + 0.01283502 * max(0.0, 858.8262 - Q.sum_pt_top40)
            + 0.004181412 * max(0.0, 1156.659 - Q.sum_pt_top50)
        ))
        + 0.1875 * grid(4, max(0.0, 1.367567
            - 0.02142609 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.03053927 * max(0.0, 60.43821 - Q.mass_top30)
            + 0.0201289 * max(0.0, 120.6 - Q.mass)
            - 0.02839266 * max(0.0, 86.4 - Q.mass)
            + 0.02532736 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.02472351 * max(0.0, Q.n_particles - 22.0)
            - 282.8053 * max(0.0, 0.004855289 - Q.girth2_top15)
            - 38.14918 * max(0.0, Q.girth2_top15 - 0.00727763)
            + 0.01708885 * max(0.0, 57.87349 - Q.mass_top15)
            + 0.05217597 * max(0.0, 101.0497 - Q.mass)
            - 0.03281671 * max(0.0, 80.78464 - Q.mass)
            + 0.08842908 * max(0.0, 74.25181 - Q.mass)
            - 0.008418597 * max(0.0, 163.2541 - Q.mass_top40)
            - 0.02998102 * max(0.0, 92.85979 - Q.mass)
            + 0.04484549 * max(0.0, 67.72643 - Q.mass_top40)
            - 0.1146276 * max(0.0, 78.26182 - Q.mass)
            - 13.37402 * max(0.0, Q.girth - 0.1207452)
            - 149.4077 * max(0.0, Q.lam1 - 0.008241985)
            + 55.23399 * max(0.0, Q.girth2_top15 - 0.01563836)
            + 167.0418 * max(0.0, Q.e2_sq - 0.009606007)
            - 143.5591 * max(0.0, Q.lam2 - 0.001776308)
        ))
        - 0.46875 * grid(5, max(0.0, 0.6380697
            + 0.0381413 * max(0.0, 64.0 - Q.n_particles)
            - 10.20386 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 0.006261253 * max(0.0, Q.sum_pt - 907.9372)
            - 15.76866 * max(0.0, Q.log_sum_pt - 6.920349)
            + 167.2861 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.01647099 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.02016727 * max(0.0, Q.sum_pt_top50 - 934.2416)
            - 0.005608218 * max(0.0, Q.sum_pt_top20 - 1129.275)
            + 29743.12 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4)
            - 15.12077 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.01244614 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 9.362286 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.005879184 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 2.518499 * max(0.0, Q.max_dr - 0.2404747)
            - 14.75085 * max(0.0, Q.z_top30_slots - 0.9048492)
            - 0.02569512 * max(0.0, Q.mass_top40 - 120.6)
            + 0.07803723 * max(0.0, Q.mass - 172.8)
            - 115.5639 * max(0.0, Q.girth2_top30 - 0.006363916)
            - 55.14108 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2)
            - 0.005410795 * max(0.0, 150.0144 - Q.mass_top40)
            - 511.5796 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            - 2.370548 * max(0.0, 0.5494307 - Q.tau21)
            + 0.05009839 * max(0.0, Q.sd_mass - 69.65633)
            - 0.1307292 * max(0.0, Q.mass_top50 - 157.5448)
            - 0.004752506 * max(0.0, Q.sum_pt - 986.0565)
            + 84.69973 * max(0.0, 0.01375115 - Q.z_11)
            + 0.01644698 * max(0.0, Q.mass - 74.25181)
            - 0.07463258 * max(0.0, Q.sd_mass - 86.4)
            + 0.02046421 * max(0.0, Q.mass_top10 - 71.781)
            - 0.06378087 * max(0.0, 14.14062 - Q.pt_11)
            + 3.843215 * max(0.0, Q.max_dr - 0.4357228)
            + 0.402659 * max(0.0, 1.976207 - Q.D2)
        ))
        - 0.28125 * grid(7, max(0.0, 0.02822393
            + 15.81855 * max(0.0, 0.2352054 - Q.tau21_b2)
            - 2245.457 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            + 8.27917 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            - 55.89039 * max(0.0, 0.006403325 - Q.girth2)
            - 0.2412638 * max(0.0, 82.85409 - Q.mass)
            - 0.06299596 * max(0.0, 101.0497 - Q.mass)
            + 0.07985738 * max(0.0, 6.0 - Q.n_dr_0p2_0p4)
            - 1634.034 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 7.807953 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            + 10.95938 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            + 4.81802 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            - 19.21488 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            - 109.9719 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            - 66.66523 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            + 123.7156 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            - 0.02163195 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt)
            + 0.04028611 * max(0.0, 120.6 - Q.mass)
            - 8.922787 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 6.339537 * max(0.0, 0.3861957 - Q.tau21)
            + 152371.2 * max(0.0, 0.006403325 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.9980008)
            - 817.5024 * max(0.0, 0.00363788 - Q.e2_sq)
            + 0.182892 * max(0.0, 78.26182 - Q.mass)
            - 317.4453 * max(0.0, 0.008190222 - Q.girth2)
            + 408.5902 * max(0.0, 0.009614971 - Q.girth2)
            - 42.6427 * max(0.0, 0.1072713 - Q.tau1)
            + 27.09727 * max(0.0, 0.09591084 - Q.tau1)
            - 353.4245 * max(0.0, 0.008241985 - Q.lam1)
            + 346.7997 * max(0.0, 0.006189818 - Q.lam1)
            + 548.1077 * max(0.0, Q.psi_0p3 - 0.9973959)
            + 0.04255483 * max(0.0, 69.65633 - Q.sd_mass)
            - 0.03352977 * max(0.0, 86.4 - Q.sd_mass)
        ))
        + 0.2109375 * grid(8, max(0.0, 2.120228
            - 14.22432 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 17.78255 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.0724109 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.03722501 * max(0.0, Q.mass - 125.1)
            + 0.05697973 * max(0.0, Q.mass - 87.36377)
            + 0.4111008 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt)
            + 0.05891433 * max(0.0, Q.n_particles - 51.0)
            + 0.009781456 * max(0.0, 1017.435 - Q.sum_pt)
            - 974.6301 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.06636952 * max(0.0, Q.n_particles - 51.0) * max(0.0, 1.091797 - Q.soft1_pt)
            - 0.07660802 * max(0.0, Q.mass - 101.0497)
            + 239.2091 * max(0.0, Q.girth2_top20 - 0.008031209)
            - 12.99823 * max(0.0, 6.930088 - Q.log_sum_pt)
            - 195.7614 * max(0.0, Q.girth2_top20 - 0.006043209)
            + 0.01750729 * max(0.0, Q.mass - 64.48544)
            + 132.4598 * max(0.0, 0.007463985 - Q.girth2_top30)
            - 648.1557 * max(0.0, 0.009614971 - Q.width)
            + 175.6781 * max(0.0, 0.007877041 - Q.girth2)
            - 0.0584343 * max(0.0, Q.n_for_90pct - 7.0)
            - 107.4047 * max(0.0, Q.psi_0p3 - 0.9943058)
            + 189.9806 * max(0.0, 0.007671243 - Q.lam1)
            + 12.06935 * max(0.0, 0.0795038 - Q.tau2)
            - 1112.627 * max(0.0, Q.e3 - 5.13841e-05)
            - 0.03764843 * max(0.0, 39.0 - Q.n_for_90pct)
            - 0.04839883 * max(0.0, Q.mass_top50 - 82.04491)
            + 0.05231694 * max(0.0, Q.mass_top50 - 117.0487)
            + 0.04775024 * max(0.0, Q.mass - 74.25181)
            - 0.02550469 * max(0.0, Q.mass_top20 - 119.2969)
            + 0.0172691 * max(0.0, 1028.184 - Q.sum_pt)
            - 0.006395085 * max(0.0, 1024.942 - Q.sum_pt_top40)
            + 4.235687 * max(0.0, 0.8316924 - Q.z_top15_slots)
            + 2.532682 * max(0.0, 0.3628388 - Q.psi_0p1)
            - 0.3927259 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 0.3571579 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 132.7549 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 4925.494 * max(0.0, Q.e3 - 0.0003372339)
            - 26.55848 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
        ))
        + 0.984375 * grid(10, max(0.0, 0.1516575
            - 9.156365 * max(0.0, 0.1207452 - Q.girth)
            - 0.04117765 * max(0.0, Q.mass - 162.8363)
            - 0.01079364 * max(0.0, 65.20727 - Q.sj2_mass1)
            + 0.1971708 * max(0.0, 2.975532 - Q.D2)
            + 3.47822 * max(0.0, 0.5760704 - Q.z_top2_slots)
            - 0.008408502 * max(0.0, 59.40777 - Q.mass_top5)
            + 75.61714 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109)
            - 0.003694212 * max(0.0, 959.0957 - Q.sum_pt_top50)
            - 552.6863 * max(0.0, 0.002575211 - Q.girth2)
            - 0.04266331 * max(0.0, 101.0497 - Q.mass)
            + 7.608833 * max(0.0, 0.06413297 - Q.dr_0)
            - 8.225958 * max(0.0, Q.z_dr_0_0p05 - 0.7674734)
            - 0.1112269 * max(0.0, Q.mass_top50 - 172.8)
            - 22285.17 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.9985421)
            + 44.35472 * max(0.0, Q.e2 - 0.01256572)
            + 5259.916 * max(0.0, 0.0001086251 - Q.e3)
            - 0.03858257 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 0.1152859 * max(0.0, Q.mass - 143.7876)
            + 0.08793941 * max(0.0, Q.mass_top50 - 136.785)
            - 49.35608 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897)
            + 0.06253669 * max(0.0, Q.mass - 78.26182)
            + 7.189171 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 0.02327332 * max(0.0, 72.18744 - Q.mass_top15)
            + 89.15488 * max(0.0, 0.01655983 - Q.girth2_top20)
            - 0.04186768 * max(0.0, Q.mass - 64.48544)
            + 0.08707199 * max(0.0, 89.74183 - Q.mass)
            - 0.0108149 * max(0.0, Q.mass_top30 - 78.53034)
        ))
        - 0.375 * grid(12, max(0.0, -0.5687879
            + 0.4328124 * max(0.0, Q.sd_mass - 125.1) * max(0.0, 0.4199841 - Q.sd_zg)
            - 5.354943 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            - 50.029 * max(0.0, 0.05077291 - Q.mass_over_sum_pt)
            + 0.001064185 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            + 0.04677431 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926)
            + 464.2041 * max(0.0, 0.0007431905 - Q.girth2_top10)
            + 0.07865754 * max(0.0, 82.85409 - Q.mass)
            - 0.02207065 * max(0.0, 74.78616 - Q.mass_top40)
            + 0.03257782 * max(0.0, 74.25181 - Q.mass)
            - 0.08413784 * max(0.0, 62.55 - Q.mass)
            + 13.10186 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 18.94421 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            - 18.48827 * max(0.0, Q.sd_mass - 125.1) * max(0.0, Q.lam2 - 0.0001679609)
            - 0.06440303 * max(0.0, Q.mass - 160.8)
        ))
        - 0.90625 * grid(13, max(0.0, 2.342864
            - 0.008430008 * max(0.0, 1085.125 - Q.sum_pt)
            - 0.04095423 * max(0.0, Q.mass - 136.785)
            + 0.1371896 * max(0.0, Q.mass - 172.8)
            + 0.0103348 * max(0.0, Q.mass - 74.25181)
            - 0.02034397 * max(0.0, 1007.788 - Q.sum_pt)
            + 229.6855 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            + 0.007036842 * max(0.0, 1053.047 - Q.sum_pt_top40)
            - 0.004984864 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.8310045 - Q.tau21_b2)
            - 0.2166997 * max(0.0, Q.mass - 160.8)
            + 0.1255019 * max(0.0, Q.mass_top50 - 136.785)
            + 66.34904 * max(0.0, 0.02550569 - Q.girth2_top50)
            - 6.774786 * max(0.0, 0.08286256 - Q.tau1)
            - 0.02938187 * max(0.0, Q.mass_top50 - 92.16545)
            - 0.07534643 * max(0.0, Q.mass - 143.7876)
            - 584.77 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            - 9.559906 * max(0.0, 6.910131 - Q.log_sum_pt)
            + 71.43535 * max(0.0, 0.007820315 - Q.girth2_top50)
            - 0.01118256 * max(0.0, 160.8 - Q.mass_top40)
            - 164.6668 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.1312677 - Q.dr_13)
            + 0.01108942 * max(0.0, 997.0189 - Q.sum_pt_top50)
            - 0.005400036 * max(0.0, 966.0633 - Q.sum_pt_top30)
            + 0.08388918 * max(0.0, Q.mass_top50 - 168.9698)
            - 20.76268 * max(0.0, 0.02580396 - Q.mass_over_sum_pt_sq)
            + 1592.06 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.sd_zg - 0.4462823)
            + 0.03927978 * max(0.0, Q.mass - 162.8363)
            + 4232.593 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.005041702 - Q.zdr_11)
            + 0.005001009 * max(0.0, 1007.44 - Q.sum_pt_top40)
        ))
        - 0.375 * grid(15, max(0.0, -0.4554918
            + 0.6593533 * max(0.0, 0.8459004 - Q.z_dr_0_0p05)
            + 10.74733 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2)
            + 91.84845 * max(0.0, 0.008329695 - Q.girth2_top5)
            + 0.008982757 * max(0.0, 986.0565 - Q.sum_pt)
            + 6.495682 * max(0.0, 7.017258 - Q.log_sum_pt)
            - 72.05958 * max(0.0, Q.psi_0p3 - 0.9896594)
            - 0.004182898 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 1260.541 - Q.sum_pt)
            - 0.4257766 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0)
            - 0.006464618 * max(0.0, 1002.379 - Q.sum_pt)
            - 288.0632 * max(0.0, 0.002270363 - Q.girth2_top5)
            + 390.9258 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2)
            + 100.6571 * max(0.0, 0.007678544 - Q.girth2_top10)
            - 3.407665 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677)
            - 16.10477 * max(0.0, 0.0705748 - Q.tau1)
            + 0.01923916 * max(0.0, 79.18312 - Q.sd_mass)
            - 0.02277408 * max(0.0, 45.595 - Q.sd_mass)
            - 0.006371291 * max(0.0, 935.8189 - Q.sum_pt_top40)
            - 0.005431578 * max(0.0, 1018.698 - Q.sum_pt_top40)
            - 9.187314 * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            + 0.005659686 * max(0.0, 933.1875 - Q.sum_pt_top30)
            - 5.053077 * max(0.0, 6.903423 - Q.log_sum_pt)
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
