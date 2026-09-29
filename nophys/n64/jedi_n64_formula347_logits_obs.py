"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.8% (the network: 81.1%); same class as the network for 92.5% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_11                  pT of particle 11 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
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
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_11=pt[11],
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft5_pt=softp(5, 'pt'),
        z_11=z[11],
        z_top10_slots=sum(pt[:10]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        zdr_2=z[2] * dr[2],
        zdr_5=z[5] * dr[5],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_mass=softdrop("mass"),
        sj2_dr=subjets(2)["dr"][0],
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
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
        + 0.625 * grid(1, max(0.0, 1.699144
            + 0.1110718 * max(0.0, Q.n_particles - 38.0)
            + 21.37111 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.01530065 * max(0.0, Q.sum_pt_top50 - 959.0957)
            - 24.24301 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.003618716 * max(0.0, 689.25 - Q.sum_pt_top2)
            - 0.04804845 * max(0.0, 47.88842 - Q.mass_top20)
            - 0.03191178 * max(0.0, 32.50209 - Q.sj3_mass1)
            + 0.5826408 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793)
            + 0.0032811 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.004765498 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0)
            - 0.03159573 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt)
            - 0.08479318 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            + 269.9196 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2)
            - 0.002724569 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2)
            - 29.0832 * max(0.0, 0.03187688 - Q.M3)
            + 24.05734 * max(0.0, Q.log_sum_pt - 6.893714)
            - 0.03220738 * max(0.0, 101.0497 - Q.mass)
            + 0.03357584 * max(0.0, 89.74183 - Q.mass)
            + 2.302253 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.01395978 - Q.zdr_2)
            - 0.007497506 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 0.9624339 - Q.tau43)
            - 7.734848 * max(0.0, 7.139296 - Q.log_sum_pt)
            + 1444.61 * max(0.0, 0.0007894752 - Q.girth2_top15)
            - 3.712832 * max(0.0, Q.log_sum_pt - 6.959294)
            + 0.01240808 * max(0.0, 1017.435 - Q.sum_pt)
            + 0.005714293 * max(0.0, Q.sum_pt_top30 - 1191.938)
            + 0.01638019 * max(0.0, 80.24626 - Q.mass_top30)
            + 9.509696 * max(0.0, 0.1751567 - Q.tau1)
            + 0.008045675 * max(0.0, Q.mass_top10 - 31.33272)
            - 43.4562 * max(0.0, Q.zdr_0 - 0.001901263)
        ))
        - 0.75 * grid(3, max(0.0, 0.1813042
            + 0.2345929 * max(0.0, 5.0 - Q.n_dr_0p2_0p4)
            - 0.04334046 * max(0.0, 46.0 - Q.n_particles)
            + 8.458651e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732)
            - 4.145237 * max(0.0, 0.3861957 - Q.tau21)
            + 10.96933 * max(0.0, 0.08665515 - Q.mass_over_sum_pt)
            + 279.3914 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 26.2678 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5)
            - 1164.553 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243)
            - 0.01614239 * max(0.0, 79.21004 - Q.mass_top50)
            + 0.02264908 * max(0.0, 87.36377 - Q.mass)
            - 133.4898 * max(0.0, 0.009614971 - Q.girth2)
            + 0.08496657 * max(0.0, 10.0 - Q.n_dr_0p1_0p2)
            - 229.8354 * max(0.0, 0.006259772 - Q.girth2_top40)
            - 8.498907 * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            + 184.1081 * max(0.0, 0.01626937 - Q.tau4)
            + 170.9969 * max(0.0, 0.008124776 - Q.girth2_top50)
        ))
        - 0.875 * grid(4, max(0.0, 1.320231
            + 1.815251 * max(0.0, 0.3036026 - Q.planar_flow)
            + 0.02854843 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.01838809 * max(0.0, 121.3913 - Q.mass)
            - 0.0281917 * max(0.0, 87.36377 - Q.mass)
            - 0.01492449 * max(0.0, 121.737 - Q.mass_top30)
            + 0.02031191 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.0295923 * max(0.0, Q.n_particles - 22.0)
            - 248.5671 * max(0.0, 0.004855289 - Q.girth2_top15)
            - 0.00464297 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 6.916121 - Q.D2)
            + 0.01542349 * max(0.0, 57.87349 - Q.mass_top15)
            + 0.03164212 * max(0.0, 67.72643 - Q.mass_top40)
            - 0.04345001 * max(0.0, 79.65241 - Q.mass)
            + 0.007693702 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt)
            - 34.44664 * max(0.0, 79.65241 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            + 0.07679006 * max(0.0, 74.25181 - Q.mass)
            - 0.03272052 * max(0.0, 94.64253 - Q.mass_top40)
            - 283.6493 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.0329485 - Q.C2_b2)
            + 14.8246 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
        ))
        - 0.3125 * grid(5, max(0.0, 1.587376
            + 0.03551522 * max(0.0, 64.0 - Q.n_particles)
            - 10.32355 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 0.008465681 * max(0.0, Q.sum_pt - 907.9372)
            - 14.57031 * max(0.0, Q.log_sum_pt - 6.920349)
            + 226.7243 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.01557447 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.02182161 * max(0.0, Q.sum_pt_top50 - 934.2416)
            - 17.02726 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.01354899 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 5.554297 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.004782319 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 4.054803 * max(0.0, Q.max_dr - 0.2404747)
            - 13.21984 * max(0.0, Q.z_top30_slots - 0.9048492)
            - 0.02606257 * max(0.0, Q.mass_top40 - 111.2487)
            - 92.67976 * max(0.0, Q.girth2_top30 - 0.006363916)
            - 0.01584143 * max(0.0, 150.0144 - Q.mass_top40)
            - 0.2978184 * max(0.0, Q.mass_top50 - 157.5448)
            - 485.1822 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 0.03047514 * max(0.0, Q.sd_mass - 69.65633)
            - 2.165932 * max(0.0, 0.5100475 - Q.tau21)
            - 0.006146357 * max(0.0, Q.sum_pt - 986.0565)
            + 156.1481 * max(0.0, 0.01375115 - Q.z_11)
            - 0.1408581 * max(0.0, 14.14062 - Q.pt_11)
            - 0.03696409 * max(0.0, Q.sd_mass - 83.30647)
            + 0.01843801 * max(0.0, 64.48544 - Q.mass)
            + 0.02141708 * max(0.0, Q.mass - 101.0497)
            + 5.396198 * max(0.0, Q.max_dr - 0.4357228)
        ))
        + 0.15625 * grid(6, max(0.0, 0.5705026
            + 0.05212258 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.03720253 * max(0.0, 121.3913 - Q.mass)
            + 2796.072 * max(0.0, 0.0003372339 - Q.e3)
            + 0.03012143 * max(0.0, 82.85409 - Q.mass)
            + 7.163951e-05 * max(0.0, 121.3913 - Q.mass) * max(0.0, 1003.544 - Q.sum_pt_top50)
            + 0.007722004 * max(0.0, 172.4888 - Q.mass)
            + 0.03779809 * max(0.0, 89.74183 - Q.mass)
            - 0.08445418 * max(0.0, 101.0497 - Q.mass)
            + 0.05069144 * max(0.0, 92.85979 - Q.mass)
            - 0.4640216 * max(0.0, 2.178951 - Q.D2)
            + 241.3345 * max(0.0, 0.007259287 - Q.lam1)
            - 46.95896 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            + 200.8755 * max(0.0, 0.02580859 - Q.e2_sq)
            - 63.29214 * max(0.0, 0.01649354 - Q.lam1)
            - 78.40662 * max(0.0, 0.02550569 - Q.girth2_top50)
            + 111.3142 * max(0.0, 0.008376291 - Q.girth2_top30)
            - 81.29808 * max(0.0, 0.02412652 - Q.girth2_top30)
        ))
        + 0.03125 * grid(8, max(0.0, -1.123515
            + 62.43621 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 0.003108768 * max(0.0, 1001.523 - Q.sum_pt_top40)
            + 43.62621 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.09117773 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.038689 * max(0.0, Q.mass - 121.3913)
            + 0.0227803 * max(0.0, Q.mass - 89.74183)
            + 3.973755 * max(0.0, Q.log_sum_pt - 6.959294)
            - 0.004201743 * max(0.0, Q.sum_pt_top40 - 858.8262)
            - 0.118754 * max(0.0, Q.mass - 101.0497)
            + 158.411 * max(0.0, Q.girth2_top40 - 0.008031986)
            + 0.06151402 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435)
            + 0.02949538 * max(0.0, Q.mass - 64.48544)
            + 240.4989 * max(0.0, 0.007887677 - Q.girth2_top15)
            - 0.06021091 * max(0.0, Q.mass_top50 - 80.35535)
            - 520.5222 * max(0.0, Q.e2_sq - 0.009606007)
            + 0.04492959 * max(0.0, Q.n_dr_0p2_0p4 - 8.0)
            + 44.87561 * max(0.0, Q.girth2_top15 - 0.001319197)
            - 0.00987398 * max(0.0, 1003.544 - Q.sum_pt_top50)
            - 13.75792 * max(0.0, Q.tau1 - 0.06310829)
            + 341.8495 * max(0.0, Q.e2_sq - 0.007872294)
            - 163.7658 * max(0.0, Q.girth2_top40 - 0.006026828)
            + 7.677561 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4)
            + 0.03520039 * max(0.0, Q.mass_top40 - 111.2487)
            + 16.80617 * max(0.0, Q.LHA - 0.3332345)
            + 0.05480542 * max(0.0, Q.mass - 80.78464)
            - 0.02767922 * max(0.0, Q.mass - 143.7876)
            + 0.03713345 * max(0.0, Q.mass_top50 - 97.93004)
            + 4192.388 * max(0.0, 0.0003372339 - Q.e3)
            - 47.41452 * max(0.0, 0.02737453 - Q.girth2_top20)
            + 9.75284 * max(0.0, 6.935549 - Q.log_sum_pt)
            + 99.76002 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266)
            - 126.1674 * max(0.0, 0.005383629 - Q.girth2_top15)
        ))
        + 0.515625 * grid(9, max(0.0, 0.8206395
            + 0.01775765 * max(0.0, 80.89043 - Q.mass_top40)
            + 411.9231 * max(0.0, 0.006259772 - Q.girth2_top40)
            - 0.0001695353 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt)
            + 0.007765967 * max(0.0, 121.737 - Q.mass_top30)
            - 0.02276211 * max(0.0, Q.mass - 143.7876)
            - 0.08533186 * max(0.0, Q.mass - 172.4888)
            + 0.01829782 * max(0.0, Q.mass - 121.3913)
            + 0.04986 * max(0.0, 87.36377 - Q.mass)
            - 0.01204777 * max(0.0, 152.6883 - Q.mass_top30)
            - 19.80164 * max(0.0, Q.z_top40_slots - 0.9574183)
            + 0.01182259 * max(0.0, 959.0957 - Q.sum_pt_top50)
            + 0.02130183 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.894531)
            - 92.11723 * max(0.0, 0.01083435 - Q.girth2_top20)
            - 0.01273443 * max(0.0, 89.6788 - Q.mass_top40)
            - 0.08549108 * max(0.0, 64.48544 - Q.mass)
            + 0.06270789 * max(0.0, 79.65241 - Q.mass)
            - 50.80126 * max(0.0, 0.05444509 - Q.tau1)
            - 4883.59 * max(0.0, Q.e3 - 0.0003372339)
            + 19.71374 * max(0.0, 0.2091025 - Q.LHA)
            - 0.02446064 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft5_pt - 2.894531)
        ))
        - 0.015625 * grid(10, max(0.0, -3.927037
            - 0.04428965 * max(0.0, 80.3008 - Q.sj2_mass1)
            + 0.03904127 * max(0.0, 87.27603 - Q.mass_top40)
            + 0.4760902 * max(0.0, 3.814159 - Q.D2)
            + 0.009703165 * max(0.0, 111.2487 - Q.mass_top40)
            - 2.506262 * max(0.0, 3.814159 - Q.D2) * max(0.0, Q.sj2_dr - 0.1937688)
            + 0.07892971 * max(0.0, Q.mass_top50 - 157.5448)
            - 8.426499 * max(0.0, Q.z_dr_0_0p05 - 0.7674734)
            - 6.278313 * max(0.0, Q.log_sum_pt - 6.903423)
            + 0.08179433 * max(0.0, Q.mass - 82.85409)
            - 0.07422982 * max(0.0, Q.mass - 143.7876)
            - 0.06287764 * max(0.0, Q.mass - 64.48544)
            + 0.003245351 * Q.sum_pt
            - 0.01138367 * max(0.0, 75.26407 - Q.mass_top15)
            - 360.7204 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            + 0.003063681 * max(0.0, 839.9547 - Q.sum_pt_top5)
            - 0.2432757 * max(0.0, Q.mass - 172.4888)
            + 48.49059 * max(0.0, 0.01807679 - Q.girth2_top30)
            + 0.03895951 * max(0.0, 132.4278 - Q.mass_top40)
            - 0.08366244 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.03746091 * max(0.0, Q.mass_top40 - 67.72643)
            + 837.5802 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 2.654827 * max(0.0, Q.mass - 172.4888) * max(0.0, Q.M2 - 0.04260132)
            + 0.03023939 * max(0.0, 74.25181 - Q.mass)
            + 0.0008305667 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.91979)
            + 1.801406 * max(0.0, 0.6133424 - Q.tau21_b2)
        ))
        + 0.234375 * grid(12, max(0.0, -0.3564799
            + 0.0211521 * max(0.0, 87.36377 - Q.mass)
            - 0.05023374 * max(0.0, Q.sd_mass - 133.2575)
            - 5.702853 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3)
            + 0.05076254 * max(0.0, 82.85409 - Q.mass)
            - 590.416 * max(0.0, 0.00363788 - Q.e2_sq)
            - 0.009460028 * max(0.0, Q.sj3_pair_mass_max - 122.7494)
            + 0.001487776 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 65.20727 - Q.sj2_mass1)
            - 0.03205357 * max(0.0, 77.93668 - Q.mass_top40)
            - 0.04164589 * max(0.0, 53.87362 - Q.mass)
            + 0.003350507 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1)
            - 0.005710475 * max(0.0, 972.0419 - Q.sum_pt)
            - 16.22844 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            + 0.07374941 * max(0.0, 80.78464 - Q.mass)
        ))
    )


def logit_q(Q):
    return (1.359375
        - 0.1875 * grid(1, max(0.0, 1.699144
            + 0.1110718 * max(0.0, Q.n_particles - 38.0)
            + 21.37111 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.01530065 * max(0.0, Q.sum_pt_top50 - 959.0957)
            - 24.24301 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.003618716 * max(0.0, 689.25 - Q.sum_pt_top2)
            - 0.04804845 * max(0.0, 47.88842 - Q.mass_top20)
            - 0.03191178 * max(0.0, 32.50209 - Q.sj3_mass1)
            + 0.5826408 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793)
            + 0.0032811 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.004765498 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0)
            - 0.03159573 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt)
            - 0.08479318 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            + 269.9196 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2)
            - 0.002724569 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2)
            - 29.0832 * max(0.0, 0.03187688 - Q.M3)
            + 24.05734 * max(0.0, Q.log_sum_pt - 6.893714)
            - 0.03220738 * max(0.0, 101.0497 - Q.mass)
            + 0.03357584 * max(0.0, 89.74183 - Q.mass)
            + 2.302253 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.01395978 - Q.zdr_2)
            - 0.007497506 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 0.9624339 - Q.tau43)
            - 7.734848 * max(0.0, 7.139296 - Q.log_sum_pt)
            + 1444.61 * max(0.0, 0.0007894752 - Q.girth2_top15)
            - 3.712832 * max(0.0, Q.log_sum_pt - 6.959294)
            + 0.01240808 * max(0.0, 1017.435 - Q.sum_pt)
            + 0.005714293 * max(0.0, Q.sum_pt_top30 - 1191.938)
            + 0.01638019 * max(0.0, 80.24626 - Q.mass_top30)
            + 9.509696 * max(0.0, 0.1751567 - Q.tau1)
            + 0.008045675 * max(0.0, Q.mass_top10 - 31.33272)
            - 43.4562 * max(0.0, Q.zdr_0 - 0.001901263)
        ))
        + 0.125 * grid(2, max(0.0, 0.1748484
            - 0.0282944 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.mass_top40 - 77.93668)
            + 0.008190333 * max(0.0, Q.sum_pt - 1017.435)
            + 654.9003 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15)
            - 50.88179 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135)
            - 0.4347747 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15)
            + 0.005464577 * max(0.0, Q.sum_pt_top50 - 1156.659)
            + 0.03823619 * max(0.0, 91.03469 - Q.mass)
            - 0.009834525 * max(0.0, 996.8867 - Q.sum_pt_top30)
            - 0.003338323 * max(0.0, Q.sum_pt - 1115.723)
            - 0.003162768 * max(0.0, 1260.541 - Q.sum_pt)
            - 0.06000166 * max(0.0, 92.85979 - Q.mass)
            + 0.009692423 * max(0.0, 121.3913 - Q.mass)
            - 192.8288 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.dr_3 - 0.1042479)
            - 0.0005093501 * max(0.0, 1041.263 - Q.sum_pt_top40) * max(0.0, 26.08605 - Q.sj3_pair_mass_min)
            - 0.006842264 * max(0.0, Q.sum_pt - 1052.889)
            + 0.004269679 * max(0.0, 926.0902 - Q.sum_pt_top20)
            + 0.005339832 * max(0.0, 1053.047 - Q.sum_pt_top40)
            + 0.008010576 * max(0.0, 143.7876 - Q.mass)
        ))
        - 1.0625 * grid(4, max(0.0, 1.320231
            + 1.815251 * max(0.0, 0.3036026 - Q.planar_flow)
            + 0.02854843 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.01838809 * max(0.0, 121.3913 - Q.mass)
            - 0.0281917 * max(0.0, 87.36377 - Q.mass)
            - 0.01492449 * max(0.0, 121.737 - Q.mass_top30)
            + 0.02031191 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.0295923 * max(0.0, Q.n_particles - 22.0)
            - 248.5671 * max(0.0, 0.004855289 - Q.girth2_top15)
            - 0.00464297 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 6.916121 - Q.D2)
            + 0.01542349 * max(0.0, 57.87349 - Q.mass_top15)
            + 0.03164212 * max(0.0, 67.72643 - Q.mass_top40)
            - 0.04345001 * max(0.0, 79.65241 - Q.mass)
            + 0.007693702 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt)
            - 34.44664 * max(0.0, 79.65241 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            + 0.07679006 * max(0.0, 74.25181 - Q.mass)
            - 0.03272052 * max(0.0, 94.64253 - Q.mass_top40)
            - 283.6493 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.0329485 - Q.C2_b2)
            + 14.8246 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
        ))
        + 0.21875 * grid(6, max(0.0, 0.5705026
            + 0.05212258 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.03720253 * max(0.0, 121.3913 - Q.mass)
            + 2796.072 * max(0.0, 0.0003372339 - Q.e3)
            + 0.03012143 * max(0.0, 82.85409 - Q.mass)
            + 7.163951e-05 * max(0.0, 121.3913 - Q.mass) * max(0.0, 1003.544 - Q.sum_pt_top50)
            + 0.007722004 * max(0.0, 172.4888 - Q.mass)
            + 0.03779809 * max(0.0, 89.74183 - Q.mass)
            - 0.08445418 * max(0.0, 101.0497 - Q.mass)
            + 0.05069144 * max(0.0, 92.85979 - Q.mass)
            - 0.4640216 * max(0.0, 2.178951 - Q.D2)
            + 241.3345 * max(0.0, 0.007259287 - Q.lam1)
            - 46.95896 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            + 200.8755 * max(0.0, 0.02580859 - Q.e2_sq)
            - 63.29214 * max(0.0, 0.01649354 - Q.lam1)
            - 78.40662 * max(0.0, 0.02550569 - Q.girth2_top50)
            + 111.3142 * max(0.0, 0.008376291 - Q.girth2_top30)
            - 81.29808 * max(0.0, 0.02412652 - Q.girth2_top30)
        ))
        + 0.015625 * grid(8, max(0.0, -1.123515
            + 62.43621 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 0.003108768 * max(0.0, 1001.523 - Q.sum_pt_top40)
            + 43.62621 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.09117773 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.038689 * max(0.0, Q.mass - 121.3913)
            + 0.0227803 * max(0.0, Q.mass - 89.74183)
            + 3.973755 * max(0.0, Q.log_sum_pt - 6.959294)
            - 0.004201743 * max(0.0, Q.sum_pt_top40 - 858.8262)
            - 0.118754 * max(0.0, Q.mass - 101.0497)
            + 158.411 * max(0.0, Q.girth2_top40 - 0.008031986)
            + 0.06151402 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435)
            + 0.02949538 * max(0.0, Q.mass - 64.48544)
            + 240.4989 * max(0.0, 0.007887677 - Q.girth2_top15)
            - 0.06021091 * max(0.0, Q.mass_top50 - 80.35535)
            - 520.5222 * max(0.0, Q.e2_sq - 0.009606007)
            + 0.04492959 * max(0.0, Q.n_dr_0p2_0p4 - 8.0)
            + 44.87561 * max(0.0, Q.girth2_top15 - 0.001319197)
            - 0.00987398 * max(0.0, 1003.544 - Q.sum_pt_top50)
            - 13.75792 * max(0.0, Q.tau1 - 0.06310829)
            + 341.8495 * max(0.0, Q.e2_sq - 0.007872294)
            - 163.7658 * max(0.0, Q.girth2_top40 - 0.006026828)
            + 7.677561 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4)
            + 0.03520039 * max(0.0, Q.mass_top40 - 111.2487)
            + 16.80617 * max(0.0, Q.LHA - 0.3332345)
            + 0.05480542 * max(0.0, Q.mass - 80.78464)
            - 0.02767922 * max(0.0, Q.mass - 143.7876)
            + 0.03713345 * max(0.0, Q.mass_top50 - 97.93004)
            + 4192.388 * max(0.0, 0.0003372339 - Q.e3)
            - 47.41452 * max(0.0, 0.02737453 - Q.girth2_top20)
            + 9.75284 * max(0.0, 6.935549 - Q.log_sum_pt)
            + 99.76002 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266)
            - 126.1674 * max(0.0, 0.005383629 - Q.girth2_top15)
        ))
        + 0.5625 * grid(9, max(0.0, 0.8206395
            + 0.01775765 * max(0.0, 80.89043 - Q.mass_top40)
            + 411.9231 * max(0.0, 0.006259772 - Q.girth2_top40)
            - 0.0001695353 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt)
            + 0.007765967 * max(0.0, 121.737 - Q.mass_top30)
            - 0.02276211 * max(0.0, Q.mass - 143.7876)
            - 0.08533186 * max(0.0, Q.mass - 172.4888)
            + 0.01829782 * max(0.0, Q.mass - 121.3913)
            + 0.04986 * max(0.0, 87.36377 - Q.mass)
            - 0.01204777 * max(0.0, 152.6883 - Q.mass_top30)
            - 19.80164 * max(0.0, Q.z_top40_slots - 0.9574183)
            + 0.01182259 * max(0.0, 959.0957 - Q.sum_pt_top50)
            + 0.02130183 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.894531)
            - 92.11723 * max(0.0, 0.01083435 - Q.girth2_top20)
            - 0.01273443 * max(0.0, 89.6788 - Q.mass_top40)
            - 0.08549108 * max(0.0, 64.48544 - Q.mass)
            + 0.06270789 * max(0.0, 79.65241 - Q.mass)
            - 50.80126 * max(0.0, 0.05444509 - Q.tau1)
            - 4883.59 * max(0.0, Q.e3 - 0.0003372339)
            + 19.71374 * max(0.0, 0.2091025 - Q.LHA)
            - 0.02446064 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft5_pt - 2.894531)
        ))
        - 0.015625 * grid(10, max(0.0, -3.927037
            - 0.04428965 * max(0.0, 80.3008 - Q.sj2_mass1)
            + 0.03904127 * max(0.0, 87.27603 - Q.mass_top40)
            + 0.4760902 * max(0.0, 3.814159 - Q.D2)
            + 0.009703165 * max(0.0, 111.2487 - Q.mass_top40)
            - 2.506262 * max(0.0, 3.814159 - Q.D2) * max(0.0, Q.sj2_dr - 0.1937688)
            + 0.07892971 * max(0.0, Q.mass_top50 - 157.5448)
            - 8.426499 * max(0.0, Q.z_dr_0_0p05 - 0.7674734)
            - 6.278313 * max(0.0, Q.log_sum_pt - 6.903423)
            + 0.08179433 * max(0.0, Q.mass - 82.85409)
            - 0.07422982 * max(0.0, Q.mass - 143.7876)
            - 0.06287764 * max(0.0, Q.mass - 64.48544)
            + 0.003245351 * Q.sum_pt
            - 0.01138367 * max(0.0, 75.26407 - Q.mass_top15)
            - 360.7204 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            + 0.003063681 * max(0.0, 839.9547 - Q.sum_pt_top5)
            - 0.2432757 * max(0.0, Q.mass - 172.4888)
            + 48.49059 * max(0.0, 0.01807679 - Q.girth2_top30)
            + 0.03895951 * max(0.0, 132.4278 - Q.mass_top40)
            - 0.08366244 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.03746091 * max(0.0, Q.mass_top40 - 67.72643)
            + 837.5802 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 2.654827 * max(0.0, Q.mass - 172.4888) * max(0.0, Q.M2 - 0.04260132)
            + 0.03023939 * max(0.0, 74.25181 - Q.mass)
            + 0.0008305667 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.91979)
            + 1.801406 * max(0.0, 0.6133424 - Q.tau21_b2)
        ))
        - 0.25 * grid(11, max(0.0, -0.3048289
            + 0.09017519 * max(0.0, 10.0 - Q.n_dr_0p2_0p4)
            + 103.6906 * max(0.0, Q.psi_0p2 - 0.9935324)
            - 0.06464016 * max(0.0, 79.65241 - Q.mass)
            + 0.07707272 * max(0.0, 92.85979 - Q.mass)
            - 0.0657739 * max(0.0, 79.65241 - Q.mass) * max(0.0, 3.814159 - Q.D2)
            - 34.47731 * max(0.0, 0.05048381 - Q.girth)
            - 40.87882 * max(0.0, 0.07374472 - Q.girth)
            + 137.7768 * max(0.0, 0.003213724 - Q.girth2_top10)
            - 252.4746 * max(0.0, 0.006716737 - Q.lam1)
            - 118.473 * max(0.0, 0.006929741 - Q.girth2_top30)
            + 324.5386 * max(0.0, 0.00818374 - Q.e2_sq)
            + 3.27305 * max(0.0, 0.2845608 - Q.LHA)
            + 0.0663157 * max(0.0, 15.0 - Q.n_dr_0p1_0p2)
            - 0.04018418 * max(0.0, 85.8667 - Q.mass_top50)
            + 13.43382 * max(0.0, 0.1632346 - Q.LHA)
            - 2.178899 * max(0.0, Q.z_top10_slots - 0.7271951)
            + 0.01732716 * max(0.0, 64.48544 - Q.mass)
            + 59.92109 * max(0.0, 0.02793599 - Q.e2)
            + 8.829669 * max(0.0, 0.1072713 - Q.tau1)
        ))
        + 0.34375 * grid(12, max(0.0, -0.3564799
            + 0.0211521 * max(0.0, 87.36377 - Q.mass)
            - 0.05023374 * max(0.0, Q.sd_mass - 133.2575)
            - 5.702853 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3)
            + 0.05076254 * max(0.0, 82.85409 - Q.mass)
            - 590.416 * max(0.0, 0.00363788 - Q.e2_sq)
            - 0.009460028 * max(0.0, Q.sj3_pair_mass_max - 122.7494)
            + 0.001487776 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 65.20727 - Q.sj2_mass1)
            - 0.03205357 * max(0.0, 77.93668 - Q.mass_top40)
            - 0.04164589 * max(0.0, 53.87362 - Q.mass)
            + 0.003350507 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1)
            - 0.005710475 * max(0.0, 972.0419 - Q.sum_pt)
            - 16.22844 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            + 0.07374941 * max(0.0, 80.78464 - Q.mass)
        ))
    )


def logit_W(Q):
    return (0.09375
        + 0.75 * grid(0, max(0.0, 1.290131
            - 0.1559339 * max(0.0, Q.mass - 78.26182)
            - 0.06463391 * max(0.0, Q.mass - 92.85979)
            - 0.01011209 * max(0.0, 1012.673 - Q.sum_pt)
            + 168.3064 * max(0.0, Q.psi_0p3 - 0.9956185)
            + 0.04576459 * max(0.0, Q.mass - 91.03469)
            - 0.02439389 * max(0.0, Q.mass - 74.25181)
            + 70.95295 * max(0.0, 0.007538019 - Q.girth2_top20)
            + 0.01468743 * max(0.0, 80.24626 - Q.mass_top30)
            - 0.273688 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3)
            + 146.5153 * max(0.0, 0.007856958 - Q.girth2_top30)
            - 138.0553 * max(0.0, 0.005913555 - Q.lam1)
            + 456.5516 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            - 17.6065 * max(0.0, 0.0705748 - Q.tau1)
            - 581.9955 * max(0.0, 0.00616708 - Q.e2_sq)
            - 0.01802056 * max(0.0, Q.mass_top50 - 82.04491)
            - 5.359746 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 28.94799 * max(0.0, 0.9906378 - Q.z_top50_slots)
            - 0.02652359 * max(0.0, 101.0497 - Q.mass)
            - 282.2777 * max(0.0, 0.006363916 - Q.girth2_top30)
            + 516.5536 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq)
            - 226.3936 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.02931703 * max(0.0, Q.mass_top50 - 71.79516)
            - 0.1726541 * max(0.0, Q.mass - 87.36377)
            + 0.002732019 * max(0.0, 1156.659 - Q.sum_pt_top50)
        ))
        + 0.4375 * grid(3, max(0.0, 0.1813042
            + 0.2345929 * max(0.0, 5.0 - Q.n_dr_0p2_0p4)
            - 0.04334046 * max(0.0, 46.0 - Q.n_particles)
            + 8.458651e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732)
            - 4.145237 * max(0.0, 0.3861957 - Q.tau21)
            + 10.96933 * max(0.0, 0.08665515 - Q.mass_over_sum_pt)
            + 279.3914 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 26.2678 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5)
            - 1164.553 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243)
            - 0.01614239 * max(0.0, 79.21004 - Q.mass_top50)
            + 0.02264908 * max(0.0, 87.36377 - Q.mass)
            - 133.4898 * max(0.0, 0.009614971 - Q.girth2)
            + 0.08496657 * max(0.0, 10.0 - Q.n_dr_0p1_0p2)
            - 229.8354 * max(0.0, 0.006259772 - Q.girth2_top40)
            - 8.498907 * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            + 184.1081 * max(0.0, 0.01626937 - Q.tau4)
            + 170.9969 * max(0.0, 0.008124776 - Q.girth2_top50)
        ))
        + 0.34375 * grid(4, max(0.0, 1.320231
            + 1.815251 * max(0.0, 0.3036026 - Q.planar_flow)
            + 0.02854843 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.01838809 * max(0.0, 121.3913 - Q.mass)
            - 0.0281917 * max(0.0, 87.36377 - Q.mass)
            - 0.01492449 * max(0.0, 121.737 - Q.mass_top30)
            + 0.02031191 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.0295923 * max(0.0, Q.n_particles - 22.0)
            - 248.5671 * max(0.0, 0.004855289 - Q.girth2_top15)
            - 0.00464297 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 6.916121 - Q.D2)
            + 0.01542349 * max(0.0, 57.87349 - Q.mass_top15)
            + 0.03164212 * max(0.0, 67.72643 - Q.mass_top40)
            - 0.04345001 * max(0.0, 79.65241 - Q.mass)
            + 0.007693702 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt)
            - 34.44664 * max(0.0, 79.65241 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            + 0.07679006 * max(0.0, 74.25181 - Q.mass)
            - 0.03272052 * max(0.0, 94.64253 - Q.mass_top40)
            - 283.6493 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.0329485 - Q.C2_b2)
            + 14.8246 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
        ))
        + 0.578125 * grid(5, max(0.0, 1.587376
            + 0.03551522 * max(0.0, 64.0 - Q.n_particles)
            - 10.32355 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 0.008465681 * max(0.0, Q.sum_pt - 907.9372)
            - 14.57031 * max(0.0, Q.log_sum_pt - 6.920349)
            + 226.7243 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.01557447 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.02182161 * max(0.0, Q.sum_pt_top50 - 934.2416)
            - 17.02726 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.01354899 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 5.554297 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.004782319 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 4.054803 * max(0.0, Q.max_dr - 0.2404747)
            - 13.21984 * max(0.0, Q.z_top30_slots - 0.9048492)
            - 0.02606257 * max(0.0, Q.mass_top40 - 111.2487)
            - 92.67976 * max(0.0, Q.girth2_top30 - 0.006363916)
            - 0.01584143 * max(0.0, 150.0144 - Q.mass_top40)
            - 0.2978184 * max(0.0, Q.mass_top50 - 157.5448)
            - 485.1822 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 0.03047514 * max(0.0, Q.sd_mass - 69.65633)
            - 2.165932 * max(0.0, 0.5100475 - Q.tau21)
            - 0.006146357 * max(0.0, Q.sum_pt - 986.0565)
            + 156.1481 * max(0.0, 0.01375115 - Q.z_11)
            - 0.1408581 * max(0.0, 14.14062 - Q.pt_11)
            - 0.03696409 * max(0.0, Q.sd_mass - 83.30647)
            + 0.01843801 * max(0.0, 64.48544 - Q.mass)
            + 0.02141708 * max(0.0, Q.mass - 101.0497)
            + 5.396198 * max(0.0, Q.max_dr - 0.4357228)
        ))
        + 0.0625 * grid(6, max(0.0, 0.5705026
            + 0.05212258 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.03720253 * max(0.0, 121.3913 - Q.mass)
            + 2796.072 * max(0.0, 0.0003372339 - Q.e3)
            + 0.03012143 * max(0.0, 82.85409 - Q.mass)
            + 7.163951e-05 * max(0.0, 121.3913 - Q.mass) * max(0.0, 1003.544 - Q.sum_pt_top50)
            + 0.007722004 * max(0.0, 172.4888 - Q.mass)
            + 0.03779809 * max(0.0, 89.74183 - Q.mass)
            - 0.08445418 * max(0.0, 101.0497 - Q.mass)
            + 0.05069144 * max(0.0, 92.85979 - Q.mass)
            - 0.4640216 * max(0.0, 2.178951 - Q.D2)
            + 241.3345 * max(0.0, 0.007259287 - Q.lam1)
            - 46.95896 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            + 200.8755 * max(0.0, 0.02580859 - Q.e2_sq)
            - 63.29214 * max(0.0, 0.01649354 - Q.lam1)
            - 78.40662 * max(0.0, 0.02550569 - Q.girth2_top50)
            + 111.3142 * max(0.0, 0.008376291 - Q.girth2_top30)
            - 81.29808 * max(0.0, 0.02412652 - Q.girth2_top30)
        ))
        - 0.625 * grid(7, max(0.0, 0.2129564
            + 9.321959 * max(0.0, 0.2352054 - Q.tau21_b2)
            - 168.6935 * max(0.0, 0.007877041 - Q.girth2)
            + 13.3668 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            + 0.1101537 * max(0.0, 91.03469 - Q.mass)
            + 0.01382997 * max(0.0, 101.0497 - Q.mass)
            - 1118.301 * max(0.0, Q.psi_0p3 - 0.9980008)
            - 1.052441 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9299135)
            + 3.606845 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 8.315009 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9995915 - Q.psi_0p3)
            - 6.606819 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 0.1950215 * max(0.0, 92.85979 - Q.mass)
            - 0.01339084 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt)
            + 3.74815 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.9973959 - Q.psi_0p3)
            - 99.57103 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            + 9467.25 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 106.2355 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            + 41.82194 * max(0.0, 0.08786745 - Q.tau1)
            - 49.2305 * max(0.0, 0.1072713 - Q.tau1)
            - 8.185602 * max(0.0, 0.2601462 - Q.LHA)
            + 27.66883 * max(0.0, 0.3098384 - Q.LHA)
            + 110.2737 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 4.812168 * max(0.0, 0.3861957 - Q.tau21)
            - 9.65816 * max(0.0, 0.2845608 - Q.LHA)
            + 0.02034264 * max(0.0, 121.3913 - Q.mass)
            - 9.230698 * max(0.0, 0.3332345 - Q.LHA)
            - 0.07539474 * max(0.0, 73.35236 - Q.mass_top20)
            + 0.003507411 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.sd_mass - 76.29481)
            + 0.06780563 * max(0.0, 70.42121 - Q.mass_top20)
            + 0.1576101 * max(0.0, 78.26182 - Q.mass)
            - 10.26318 * max(0.0, Q.z_top20_slots - 0.9102775)
            - 1310.111 * max(0.0, 0.003418057 - Q.girth2_top50)
            - 386.8683 * max(0.0, 0.008241985 - Q.lam1)
            + 466.9206 * max(0.0, 0.006189818 - Q.lam1)
            + 444.8661 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq)
            - 386.6692 * max(0.0, 0.008190222 - Q.width)
        ))
        - 0.875 * grid(8, max(0.0, -1.123515
            + 62.43621 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 0.003108768 * max(0.0, 1001.523 - Q.sum_pt_top40)
            + 43.62621 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.09117773 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.038689 * max(0.0, Q.mass - 121.3913)
            + 0.0227803 * max(0.0, Q.mass - 89.74183)
            + 3.973755 * max(0.0, Q.log_sum_pt - 6.959294)
            - 0.004201743 * max(0.0, Q.sum_pt_top40 - 858.8262)
            - 0.118754 * max(0.0, Q.mass - 101.0497)
            + 158.411 * max(0.0, Q.girth2_top40 - 0.008031986)
            + 0.06151402 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435)
            + 0.02949538 * max(0.0, Q.mass - 64.48544)
            + 240.4989 * max(0.0, 0.007887677 - Q.girth2_top15)
            - 0.06021091 * max(0.0, Q.mass_top50 - 80.35535)
            - 520.5222 * max(0.0, Q.e2_sq - 0.009606007)
            + 0.04492959 * max(0.0, Q.n_dr_0p2_0p4 - 8.0)
            + 44.87561 * max(0.0, Q.girth2_top15 - 0.001319197)
            - 0.00987398 * max(0.0, 1003.544 - Q.sum_pt_top50)
            - 13.75792 * max(0.0, Q.tau1 - 0.06310829)
            + 341.8495 * max(0.0, Q.e2_sq - 0.007872294)
            - 163.7658 * max(0.0, Q.girth2_top40 - 0.006026828)
            + 7.677561 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4)
            + 0.03520039 * max(0.0, Q.mass_top40 - 111.2487)
            + 16.80617 * max(0.0, Q.LHA - 0.3332345)
            + 0.05480542 * max(0.0, Q.mass - 80.78464)
            - 0.02767922 * max(0.0, Q.mass - 143.7876)
            + 0.03713345 * max(0.0, Q.mass_top50 - 97.93004)
            + 4192.388 * max(0.0, 0.0003372339 - Q.e3)
            - 47.41452 * max(0.0, 0.02737453 - Q.girth2_top20)
            + 9.75284 * max(0.0, 6.935549 - Q.log_sum_pt)
            + 99.76002 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266)
            - 126.1674 * max(0.0, 0.005383629 - Q.girth2_top15)
        ))
        - 0.21875 * grid(9, max(0.0, 0.8206395
            + 0.01775765 * max(0.0, 80.89043 - Q.mass_top40)
            + 411.9231 * max(0.0, 0.006259772 - Q.girth2_top40)
            - 0.0001695353 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt)
            + 0.007765967 * max(0.0, 121.737 - Q.mass_top30)
            - 0.02276211 * max(0.0, Q.mass - 143.7876)
            - 0.08533186 * max(0.0, Q.mass - 172.4888)
            + 0.01829782 * max(0.0, Q.mass - 121.3913)
            + 0.04986 * max(0.0, 87.36377 - Q.mass)
            - 0.01204777 * max(0.0, 152.6883 - Q.mass_top30)
            - 19.80164 * max(0.0, Q.z_top40_slots - 0.9574183)
            + 0.01182259 * max(0.0, 959.0957 - Q.sum_pt_top50)
            + 0.02130183 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.894531)
            - 92.11723 * max(0.0, 0.01083435 - Q.girth2_top20)
            - 0.01273443 * max(0.0, 89.6788 - Q.mass_top40)
            - 0.08549108 * max(0.0, 64.48544 - Q.mass)
            + 0.06270789 * max(0.0, 79.65241 - Q.mass)
            - 50.80126 * max(0.0, 0.05444509 - Q.tau1)
            - 4883.59 * max(0.0, Q.e3 - 0.0003372339)
            + 19.71374 * max(0.0, 0.2091025 - Q.LHA)
            - 0.02446064 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft5_pt - 2.894531)
        ))
        + 0.59375 * grid(11, max(0.0, -0.3048289
            + 0.09017519 * max(0.0, 10.0 - Q.n_dr_0p2_0p4)
            + 103.6906 * max(0.0, Q.psi_0p2 - 0.9935324)
            - 0.06464016 * max(0.0, 79.65241 - Q.mass)
            + 0.07707272 * max(0.0, 92.85979 - Q.mass)
            - 0.0657739 * max(0.0, 79.65241 - Q.mass) * max(0.0, 3.814159 - Q.D2)
            - 34.47731 * max(0.0, 0.05048381 - Q.girth)
            - 40.87882 * max(0.0, 0.07374472 - Q.girth)
            + 137.7768 * max(0.0, 0.003213724 - Q.girth2_top10)
            - 252.4746 * max(0.0, 0.006716737 - Q.lam1)
            - 118.473 * max(0.0, 0.006929741 - Q.girth2_top30)
            + 324.5386 * max(0.0, 0.00818374 - Q.e2_sq)
            + 3.27305 * max(0.0, 0.2845608 - Q.LHA)
            + 0.0663157 * max(0.0, 15.0 - Q.n_dr_0p1_0p2)
            - 0.04018418 * max(0.0, 85.8667 - Q.mass_top50)
            + 13.43382 * max(0.0, 0.1632346 - Q.LHA)
            - 2.178899 * max(0.0, Q.z_top10_slots - 0.7271951)
            + 0.01732716 * max(0.0, 64.48544 - Q.mass)
            + 59.92109 * max(0.0, 0.02793599 - Q.e2)
            + 8.829669 * max(0.0, 0.1072713 - Q.tau1)
        ))
        - 0.40625 * grid(12, max(0.0, -0.3564799
            + 0.0211521 * max(0.0, 87.36377 - Q.mass)
            - 0.05023374 * max(0.0, Q.sd_mass - 133.2575)
            - 5.702853 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3)
            + 0.05076254 * max(0.0, 82.85409 - Q.mass)
            - 590.416 * max(0.0, 0.00363788 - Q.e2_sq)
            - 0.009460028 * max(0.0, Q.sj3_pair_mass_max - 122.7494)
            + 0.001487776 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 65.20727 - Q.sj2_mass1)
            - 0.03205357 * max(0.0, 77.93668 - Q.mass_top40)
            - 0.04164589 * max(0.0, 53.87362 - Q.mass)
            + 0.003350507 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1)
            - 0.005710475 * max(0.0, 972.0419 - Q.sum_pt)
            - 16.22844 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            + 0.07374941 * max(0.0, 80.78464 - Q.mass)
        ))
        - 1.375 * grid(14, max(0.0, -0.09159952
            + 1.872814 * max(0.0, 0.342495 - Q.tau21_b2)
            - 2503.318 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.007709916 - Q.girth2_top40)
            - 0.06707092 * max(0.0, 91.03469 - Q.mass)
            - 0.1707647 * max(0.0, 82.85409 - Q.mass)
            - 20.53072 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2738063)
            - 105.4495 * max(0.0, 0.09046749 - Q.mass_over_sum_pt)
            + 45.96461 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            + 41.77168 * max(0.0, 0.08873143 - Q.mass_over_sum_pt)
            - 80.13932 * max(0.0, 0.01083435 - Q.girth2_top20)
            + 125.0817 * max(0.0, 0.01986381 - Q.mass_over_sum_pt_sq)
            - 0.03808386 * max(0.0, 79.65241 - Q.mass)
            - 18.53314 * max(0.0, Q.psi_0p1 - 0.9538343)
            + 26.82068 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            + 132.7743 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.01691783 - Q.C2_b2)
            - 145.5217 * max(0.0, 0.01292642 - Q.girth2_top40)
            + 0.07385614 * max(0.0, 67.72643 - Q.mass_top40)
            - 224.7088 * max(0.0, 0.01256572 - Q.e2)
            - 14.54324 * max(0.0, 0.04142826 - Q.tau2)
            + 104.6997 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 1.059188 - Q.N3)
            + 32.74407 * max(0.0, 0.03680582 - Q.e2)
            - 53.37889 * max(0.0, 0.01897915 - Q.girth2_top40)
            + 27.49739 * max(0.0, 0.0705748 - Q.tau1)
        ))
        - 0.1875 * grid(15, max(0.0, -0.956537
            + 5.010468 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2)
            + 159.9133 * max(0.0, 0.008329695 - Q.girth2_top5)
            - 0.09615909 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 58.0 - Q.n_particles)
            + 0.00647135 * max(0.0, 986.0565 - Q.sum_pt)
            + 5.932479 * max(0.0, 7.017258 - Q.log_sum_pt)
            - 213.8386 * max(0.0, 0.002270363 - Q.girth2_top5)
            + 283.84 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2)
            + 103.211 * max(0.0, 0.007678544 - Q.girth2_top10)
            - 28.87719 * max(0.0, 0.0705748 - Q.tau1)
            - 4.092722 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.n_real_top40 - 22.0)
            + 0.0217793 * max(0.0, 79.18312 - Q.sd_mass)
            - 0.01864383 * max(0.0, 40.97891 - Q.sd_mass)
            - 0.003186645 * max(0.0, 1018.832 - Q.sum_pt_top30)
            - 10.04477 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
    )


def logit_Z(Q):
    return (0.984375
        - 1.375 * grid(0, max(0.0, 1.290131
            - 0.1559339 * max(0.0, Q.mass - 78.26182)
            - 0.06463391 * max(0.0, Q.mass - 92.85979)
            - 0.01011209 * max(0.0, 1012.673 - Q.sum_pt)
            + 168.3064 * max(0.0, Q.psi_0p3 - 0.9956185)
            + 0.04576459 * max(0.0, Q.mass - 91.03469)
            - 0.02439389 * max(0.0, Q.mass - 74.25181)
            + 70.95295 * max(0.0, 0.007538019 - Q.girth2_top20)
            + 0.01468743 * max(0.0, 80.24626 - Q.mass_top30)
            - 0.273688 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3)
            + 146.5153 * max(0.0, 0.007856958 - Q.girth2_top30)
            - 138.0553 * max(0.0, 0.005913555 - Q.lam1)
            + 456.5516 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            - 17.6065 * max(0.0, 0.0705748 - Q.tau1)
            - 581.9955 * max(0.0, 0.00616708 - Q.e2_sq)
            - 0.01802056 * max(0.0, Q.mass_top50 - 82.04491)
            - 5.359746 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 28.94799 * max(0.0, 0.9906378 - Q.z_top50_slots)
            - 0.02652359 * max(0.0, 101.0497 - Q.mass)
            - 282.2777 * max(0.0, 0.006363916 - Q.girth2_top30)
            + 516.5536 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq)
            - 226.3936 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.02931703 * max(0.0, Q.mass_top50 - 71.79516)
            - 0.1726541 * max(0.0, Q.mass - 87.36377)
            + 0.002732019 * max(0.0, 1156.659 - Q.sum_pt_top50)
        ))
        - 0.375 * grid(2, max(0.0, 0.1748484
            - 0.0282944 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.mass_top40 - 77.93668)
            + 0.008190333 * max(0.0, Q.sum_pt - 1017.435)
            + 654.9003 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15)
            - 50.88179 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135)
            - 0.4347747 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15)
            + 0.005464577 * max(0.0, Q.sum_pt_top50 - 1156.659)
            + 0.03823619 * max(0.0, 91.03469 - Q.mass)
            - 0.009834525 * max(0.0, 996.8867 - Q.sum_pt_top30)
            - 0.003338323 * max(0.0, Q.sum_pt - 1115.723)
            - 0.003162768 * max(0.0, 1260.541 - Q.sum_pt)
            - 0.06000166 * max(0.0, 92.85979 - Q.mass)
            + 0.009692423 * max(0.0, 121.3913 - Q.mass)
            - 192.8288 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.dr_3 - 0.1042479)
            - 0.0005093501 * max(0.0, 1041.263 - Q.sum_pt_top40) * max(0.0, 26.08605 - Q.sj3_pair_mass_min)
            - 0.006842264 * max(0.0, Q.sum_pt - 1052.889)
            + 0.004269679 * max(0.0, 926.0902 - Q.sum_pt_top20)
            + 0.005339832 * max(0.0, 1053.047 - Q.sum_pt_top40)
            + 0.008010576 * max(0.0, 143.7876 - Q.mass)
        ))
        + 0.4375 * grid(3, max(0.0, 0.1813042
            + 0.2345929 * max(0.0, 5.0 - Q.n_dr_0p2_0p4)
            - 0.04334046 * max(0.0, 46.0 - Q.n_particles)
            + 8.458651e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732)
            - 4.145237 * max(0.0, 0.3861957 - Q.tau21)
            + 10.96933 * max(0.0, 0.08665515 - Q.mass_over_sum_pt)
            + 279.3914 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 26.2678 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5)
            - 1164.553 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243)
            - 0.01614239 * max(0.0, 79.21004 - Q.mass_top50)
            + 0.02264908 * max(0.0, 87.36377 - Q.mass)
            - 133.4898 * max(0.0, 0.009614971 - Q.girth2)
            + 0.08496657 * max(0.0, 10.0 - Q.n_dr_0p1_0p2)
            - 229.8354 * max(0.0, 0.006259772 - Q.girth2_top40)
            - 8.498907 * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            + 184.1081 * max(0.0, 0.01626937 - Q.tau4)
            + 170.9969 * max(0.0, 0.008124776 - Q.girth2_top50)
        ))
        + 0.6875 * grid(5, max(0.0, 1.587376
            + 0.03551522 * max(0.0, 64.0 - Q.n_particles)
            - 10.32355 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 0.008465681 * max(0.0, Q.sum_pt - 907.9372)
            - 14.57031 * max(0.0, Q.log_sum_pt - 6.920349)
            + 226.7243 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.01557447 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.02182161 * max(0.0, Q.sum_pt_top50 - 934.2416)
            - 17.02726 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.01354899 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 5.554297 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.004782319 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 4.054803 * max(0.0, Q.max_dr - 0.2404747)
            - 13.21984 * max(0.0, Q.z_top30_slots - 0.9048492)
            - 0.02606257 * max(0.0, Q.mass_top40 - 111.2487)
            - 92.67976 * max(0.0, Q.girth2_top30 - 0.006363916)
            - 0.01584143 * max(0.0, 150.0144 - Q.mass_top40)
            - 0.2978184 * max(0.0, Q.mass_top50 - 157.5448)
            - 485.1822 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 0.03047514 * max(0.0, Q.sd_mass - 69.65633)
            - 2.165932 * max(0.0, 0.5100475 - Q.tau21)
            - 0.006146357 * max(0.0, Q.sum_pt - 986.0565)
            + 156.1481 * max(0.0, 0.01375115 - Q.z_11)
            - 0.1408581 * max(0.0, 14.14062 - Q.pt_11)
            - 0.03696409 * max(0.0, Q.sd_mass - 83.30647)
            + 0.01843801 * max(0.0, 64.48544 - Q.mass)
            + 0.02141708 * max(0.0, Q.mass - 101.0497)
            + 5.396198 * max(0.0, Q.max_dr - 0.4357228)
        ))
        - 0.96875 * grid(6, max(0.0, 0.5705026
            + 0.05212258 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.03720253 * max(0.0, 121.3913 - Q.mass)
            + 2796.072 * max(0.0, 0.0003372339 - Q.e3)
            + 0.03012143 * max(0.0, 82.85409 - Q.mass)
            + 7.163951e-05 * max(0.0, 121.3913 - Q.mass) * max(0.0, 1003.544 - Q.sum_pt_top50)
            + 0.007722004 * max(0.0, 172.4888 - Q.mass)
            + 0.03779809 * max(0.0, 89.74183 - Q.mass)
            - 0.08445418 * max(0.0, 101.0497 - Q.mass)
            + 0.05069144 * max(0.0, 92.85979 - Q.mass)
            - 0.4640216 * max(0.0, 2.178951 - Q.D2)
            + 241.3345 * max(0.0, 0.007259287 - Q.lam1)
            - 46.95896 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            + 200.8755 * max(0.0, 0.02580859 - Q.e2_sq)
            - 63.29214 * max(0.0, 0.01649354 - Q.lam1)
            - 78.40662 * max(0.0, 0.02550569 - Q.girth2_top50)
            + 111.3142 * max(0.0, 0.008376291 - Q.girth2_top30)
            - 81.29808 * max(0.0, 0.02412652 - Q.girth2_top30)
        ))
        + 0.90625 * grid(7, max(0.0, 0.2129564
            + 9.321959 * max(0.0, 0.2352054 - Q.tau21_b2)
            - 168.6935 * max(0.0, 0.007877041 - Q.girth2)
            + 13.3668 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            + 0.1101537 * max(0.0, 91.03469 - Q.mass)
            + 0.01382997 * max(0.0, 101.0497 - Q.mass)
            - 1118.301 * max(0.0, Q.psi_0p3 - 0.9980008)
            - 1.052441 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9299135)
            + 3.606845 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 8.315009 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9995915 - Q.psi_0p3)
            - 6.606819 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 0.1950215 * max(0.0, 92.85979 - Q.mass)
            - 0.01339084 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt)
            + 3.74815 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.9973959 - Q.psi_0p3)
            - 99.57103 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            + 9467.25 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 106.2355 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            + 41.82194 * max(0.0, 0.08786745 - Q.tau1)
            - 49.2305 * max(0.0, 0.1072713 - Q.tau1)
            - 8.185602 * max(0.0, 0.2601462 - Q.LHA)
            + 27.66883 * max(0.0, 0.3098384 - Q.LHA)
            + 110.2737 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 4.812168 * max(0.0, 0.3861957 - Q.tau21)
            - 9.65816 * max(0.0, 0.2845608 - Q.LHA)
            + 0.02034264 * max(0.0, 121.3913 - Q.mass)
            - 9.230698 * max(0.0, 0.3332345 - Q.LHA)
            - 0.07539474 * max(0.0, 73.35236 - Q.mass_top20)
            + 0.003507411 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.sd_mass - 76.29481)
            + 0.06780563 * max(0.0, 70.42121 - Q.mass_top20)
            + 0.1576101 * max(0.0, 78.26182 - Q.mass)
            - 10.26318 * max(0.0, Q.z_top20_slots - 0.9102775)
            - 1310.111 * max(0.0, 0.003418057 - Q.girth2_top50)
            - 386.8683 * max(0.0, 0.008241985 - Q.lam1)
            + 466.9206 * max(0.0, 0.006189818 - Q.lam1)
            + 444.8661 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq)
            - 386.6692 * max(0.0, 0.008190222 - Q.width)
        ))
        - 0.9375 * grid(8, max(0.0, -1.123515
            + 62.43621 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 0.003108768 * max(0.0, 1001.523 - Q.sum_pt_top40)
            + 43.62621 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.09117773 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.038689 * max(0.0, Q.mass - 121.3913)
            + 0.0227803 * max(0.0, Q.mass - 89.74183)
            + 3.973755 * max(0.0, Q.log_sum_pt - 6.959294)
            - 0.004201743 * max(0.0, Q.sum_pt_top40 - 858.8262)
            - 0.118754 * max(0.0, Q.mass - 101.0497)
            + 158.411 * max(0.0, Q.girth2_top40 - 0.008031986)
            + 0.06151402 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435)
            + 0.02949538 * max(0.0, Q.mass - 64.48544)
            + 240.4989 * max(0.0, 0.007887677 - Q.girth2_top15)
            - 0.06021091 * max(0.0, Q.mass_top50 - 80.35535)
            - 520.5222 * max(0.0, Q.e2_sq - 0.009606007)
            + 0.04492959 * max(0.0, Q.n_dr_0p2_0p4 - 8.0)
            + 44.87561 * max(0.0, Q.girth2_top15 - 0.001319197)
            - 0.00987398 * max(0.0, 1003.544 - Q.sum_pt_top50)
            - 13.75792 * max(0.0, Q.tau1 - 0.06310829)
            + 341.8495 * max(0.0, Q.e2_sq - 0.007872294)
            - 163.7658 * max(0.0, Q.girth2_top40 - 0.006026828)
            + 7.677561 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4)
            + 0.03520039 * max(0.0, Q.mass_top40 - 111.2487)
            + 16.80617 * max(0.0, Q.LHA - 0.3332345)
            + 0.05480542 * max(0.0, Q.mass - 80.78464)
            - 0.02767922 * max(0.0, Q.mass - 143.7876)
            + 0.03713345 * max(0.0, Q.mass_top50 - 97.93004)
            + 4192.388 * max(0.0, 0.0003372339 - Q.e3)
            - 47.41452 * max(0.0, 0.02737453 - Q.girth2_top20)
            + 9.75284 * max(0.0, 6.935549 - Q.log_sum_pt)
            + 99.76002 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266)
            - 126.1674 * max(0.0, 0.005383629 - Q.girth2_top15)
        ))
        + 0.3125 * grid(12, max(0.0, -0.3564799
            + 0.0211521 * max(0.0, 87.36377 - Q.mass)
            - 0.05023374 * max(0.0, Q.sd_mass - 133.2575)
            - 5.702853 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3)
            + 0.05076254 * max(0.0, 82.85409 - Q.mass)
            - 590.416 * max(0.0, 0.00363788 - Q.e2_sq)
            - 0.009460028 * max(0.0, Q.sj3_pair_mass_max - 122.7494)
            + 0.001487776 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 65.20727 - Q.sj2_mass1)
            - 0.03205357 * max(0.0, 77.93668 - Q.mass_top40)
            - 0.04164589 * max(0.0, 53.87362 - Q.mass)
            + 0.003350507 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1)
            - 0.005710475 * max(0.0, 972.0419 - Q.sum_pt)
            - 16.22844 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            + 0.07374941 * max(0.0, 80.78464 - Q.mass)
        ))
        + 0.5625 * grid(15, max(0.0, -0.956537
            + 5.010468 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2)
            + 159.9133 * max(0.0, 0.008329695 - Q.girth2_top5)
            - 0.09615909 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 58.0 - Q.n_particles)
            + 0.00647135 * max(0.0, 986.0565 - Q.sum_pt)
            + 5.932479 * max(0.0, 7.017258 - Q.log_sum_pt)
            - 213.8386 * max(0.0, 0.002270363 - Q.girth2_top5)
            + 283.84 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2)
            + 103.211 * max(0.0, 0.007678544 - Q.girth2_top10)
            - 28.87719 * max(0.0, 0.0705748 - Q.tau1)
            - 4.092722 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.n_real_top40 - 22.0)
            + 0.0217793 * max(0.0, 79.18312 - Q.sd_mass)
            - 0.01864383 * max(0.0, 40.97891 - Q.sd_mass)
            - 0.003186645 * max(0.0, 1018.832 - Q.sum_pt_top30)
            - 10.04477 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
    )


def logit_t(Q):
    return (0.78125
        + 0.125 * grid(0, max(0.0, 1.290131
            - 0.1559339 * max(0.0, Q.mass - 78.26182)
            - 0.06463391 * max(0.0, Q.mass - 92.85979)
            - 0.01011209 * max(0.0, 1012.673 - Q.sum_pt)
            + 168.3064 * max(0.0, Q.psi_0p3 - 0.9956185)
            + 0.04576459 * max(0.0, Q.mass - 91.03469)
            - 0.02439389 * max(0.0, Q.mass - 74.25181)
            + 70.95295 * max(0.0, 0.007538019 - Q.girth2_top20)
            + 0.01468743 * max(0.0, 80.24626 - Q.mass_top30)
            - 0.273688 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3)
            + 146.5153 * max(0.0, 0.007856958 - Q.girth2_top30)
            - 138.0553 * max(0.0, 0.005913555 - Q.lam1)
            + 456.5516 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            - 17.6065 * max(0.0, 0.0705748 - Q.tau1)
            - 581.9955 * max(0.0, 0.00616708 - Q.e2_sq)
            - 0.01802056 * max(0.0, Q.mass_top50 - 82.04491)
            - 5.359746 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 28.94799 * max(0.0, 0.9906378 - Q.z_top50_slots)
            - 0.02652359 * max(0.0, 101.0497 - Q.mass)
            - 282.2777 * max(0.0, 0.006363916 - Q.girth2_top30)
            + 516.5536 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq)
            - 226.3936 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.02931703 * max(0.0, Q.mass_top50 - 71.79516)
            - 0.1726541 * max(0.0, Q.mass - 87.36377)
            + 0.002732019 * max(0.0, 1156.659 - Q.sum_pt_top50)
        ))
        + 0.1875 * grid(4, max(0.0, 1.320231
            + 1.815251 * max(0.0, 0.3036026 - Q.planar_flow)
            + 0.02854843 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.01838809 * max(0.0, 121.3913 - Q.mass)
            - 0.0281917 * max(0.0, 87.36377 - Q.mass)
            - 0.01492449 * max(0.0, 121.737 - Q.mass_top30)
            + 0.02031191 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.0295923 * max(0.0, Q.n_particles - 22.0)
            - 248.5671 * max(0.0, 0.004855289 - Q.girth2_top15)
            - 0.00464297 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 6.916121 - Q.D2)
            + 0.01542349 * max(0.0, 57.87349 - Q.mass_top15)
            + 0.03164212 * max(0.0, 67.72643 - Q.mass_top40)
            - 0.04345001 * max(0.0, 79.65241 - Q.mass)
            + 0.007693702 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt)
            - 34.44664 * max(0.0, 79.65241 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            + 0.07679006 * max(0.0, 74.25181 - Q.mass)
            - 0.03272052 * max(0.0, 94.64253 - Q.mass_top40)
            - 283.6493 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.0329485 - Q.C2_b2)
            + 14.8246 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
        ))
        - 0.46875 * grid(5, max(0.0, 1.587376
            + 0.03551522 * max(0.0, 64.0 - Q.n_particles)
            - 10.32355 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 0.008465681 * max(0.0, Q.sum_pt - 907.9372)
            - 14.57031 * max(0.0, Q.log_sum_pt - 6.920349)
            + 226.7243 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.01557447 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.02182161 * max(0.0, Q.sum_pt_top50 - 934.2416)
            - 17.02726 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.01354899 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 5.554297 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.004782319 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 4.054803 * max(0.0, Q.max_dr - 0.2404747)
            - 13.21984 * max(0.0, Q.z_top30_slots - 0.9048492)
            - 0.02606257 * max(0.0, Q.mass_top40 - 111.2487)
            - 92.67976 * max(0.0, Q.girth2_top30 - 0.006363916)
            - 0.01584143 * max(0.0, 150.0144 - Q.mass_top40)
            - 0.2978184 * max(0.0, Q.mass_top50 - 157.5448)
            - 485.1822 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 0.03047514 * max(0.0, Q.sd_mass - 69.65633)
            - 2.165932 * max(0.0, 0.5100475 - Q.tau21)
            - 0.006146357 * max(0.0, Q.sum_pt - 986.0565)
            + 156.1481 * max(0.0, 0.01375115 - Q.z_11)
            - 0.1408581 * max(0.0, 14.14062 - Q.pt_11)
            - 0.03696409 * max(0.0, Q.sd_mass - 83.30647)
            + 0.01843801 * max(0.0, 64.48544 - Q.mass)
            + 0.02141708 * max(0.0, Q.mass - 101.0497)
            + 5.396198 * max(0.0, Q.max_dr - 0.4357228)
        ))
        - 0.28125 * grid(7, max(0.0, 0.2129564
            + 9.321959 * max(0.0, 0.2352054 - Q.tau21_b2)
            - 168.6935 * max(0.0, 0.007877041 - Q.girth2)
            + 13.3668 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            + 0.1101537 * max(0.0, 91.03469 - Q.mass)
            + 0.01382997 * max(0.0, 101.0497 - Q.mass)
            - 1118.301 * max(0.0, Q.psi_0p3 - 0.9980008)
            - 1.052441 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9299135)
            + 3.606845 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 8.315009 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9995915 - Q.psi_0p3)
            - 6.606819 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 0.1950215 * max(0.0, 92.85979 - Q.mass)
            - 0.01339084 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt)
            + 3.74815 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.9973959 - Q.psi_0p3)
            - 99.57103 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            + 9467.25 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 106.2355 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            + 41.82194 * max(0.0, 0.08786745 - Q.tau1)
            - 49.2305 * max(0.0, 0.1072713 - Q.tau1)
            - 8.185602 * max(0.0, 0.2601462 - Q.LHA)
            + 27.66883 * max(0.0, 0.3098384 - Q.LHA)
            + 110.2737 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 4.812168 * max(0.0, 0.3861957 - Q.tau21)
            - 9.65816 * max(0.0, 0.2845608 - Q.LHA)
            + 0.02034264 * max(0.0, 121.3913 - Q.mass)
            - 9.230698 * max(0.0, 0.3332345 - Q.LHA)
            - 0.07539474 * max(0.0, 73.35236 - Q.mass_top20)
            + 0.003507411 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.sd_mass - 76.29481)
            + 0.06780563 * max(0.0, 70.42121 - Q.mass_top20)
            + 0.1576101 * max(0.0, 78.26182 - Q.mass)
            - 10.26318 * max(0.0, Q.z_top20_slots - 0.9102775)
            - 1310.111 * max(0.0, 0.003418057 - Q.girth2_top50)
            - 386.8683 * max(0.0, 0.008241985 - Q.lam1)
            + 466.9206 * max(0.0, 0.006189818 - Q.lam1)
            + 444.8661 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq)
            - 386.6692 * max(0.0, 0.008190222 - Q.width)
        ))
        + 0.2109375 * grid(8, max(0.0, -1.123515
            + 62.43621 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 0.003108768 * max(0.0, 1001.523 - Q.sum_pt_top40)
            + 43.62621 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.09117773 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.038689 * max(0.0, Q.mass - 121.3913)
            + 0.0227803 * max(0.0, Q.mass - 89.74183)
            + 3.973755 * max(0.0, Q.log_sum_pt - 6.959294)
            - 0.004201743 * max(0.0, Q.sum_pt_top40 - 858.8262)
            - 0.118754 * max(0.0, Q.mass - 101.0497)
            + 158.411 * max(0.0, Q.girth2_top40 - 0.008031986)
            + 0.06151402 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435)
            + 0.02949538 * max(0.0, Q.mass - 64.48544)
            + 240.4989 * max(0.0, 0.007887677 - Q.girth2_top15)
            - 0.06021091 * max(0.0, Q.mass_top50 - 80.35535)
            - 520.5222 * max(0.0, Q.e2_sq - 0.009606007)
            + 0.04492959 * max(0.0, Q.n_dr_0p2_0p4 - 8.0)
            + 44.87561 * max(0.0, Q.girth2_top15 - 0.001319197)
            - 0.00987398 * max(0.0, 1003.544 - Q.sum_pt_top50)
            - 13.75792 * max(0.0, Q.tau1 - 0.06310829)
            + 341.8495 * max(0.0, Q.e2_sq - 0.007872294)
            - 163.7658 * max(0.0, Q.girth2_top40 - 0.006026828)
            + 7.677561 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4)
            + 0.03520039 * max(0.0, Q.mass_top40 - 111.2487)
            + 16.80617 * max(0.0, Q.LHA - 0.3332345)
            + 0.05480542 * max(0.0, Q.mass - 80.78464)
            - 0.02767922 * max(0.0, Q.mass - 143.7876)
            + 0.03713345 * max(0.0, Q.mass_top50 - 97.93004)
            + 4192.388 * max(0.0, 0.0003372339 - Q.e3)
            - 47.41452 * max(0.0, 0.02737453 - Q.girth2_top20)
            + 9.75284 * max(0.0, 6.935549 - Q.log_sum_pt)
            + 99.76002 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266)
            - 126.1674 * max(0.0, 0.005383629 - Q.girth2_top15)
        ))
        + 0.984375 * grid(10, max(0.0, -3.927037
            - 0.04428965 * max(0.0, 80.3008 - Q.sj2_mass1)
            + 0.03904127 * max(0.0, 87.27603 - Q.mass_top40)
            + 0.4760902 * max(0.0, 3.814159 - Q.D2)
            + 0.009703165 * max(0.0, 111.2487 - Q.mass_top40)
            - 2.506262 * max(0.0, 3.814159 - Q.D2) * max(0.0, Q.sj2_dr - 0.1937688)
            + 0.07892971 * max(0.0, Q.mass_top50 - 157.5448)
            - 8.426499 * max(0.0, Q.z_dr_0_0p05 - 0.7674734)
            - 6.278313 * max(0.0, Q.log_sum_pt - 6.903423)
            + 0.08179433 * max(0.0, Q.mass - 82.85409)
            - 0.07422982 * max(0.0, Q.mass - 143.7876)
            - 0.06287764 * max(0.0, Q.mass - 64.48544)
            + 0.003245351 * Q.sum_pt
            - 0.01138367 * max(0.0, 75.26407 - Q.mass_top15)
            - 360.7204 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            + 0.003063681 * max(0.0, 839.9547 - Q.sum_pt_top5)
            - 0.2432757 * max(0.0, Q.mass - 172.4888)
            + 48.49059 * max(0.0, 0.01807679 - Q.girth2_top30)
            + 0.03895951 * max(0.0, 132.4278 - Q.mass_top40)
            - 0.08366244 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.03746091 * max(0.0, Q.mass_top40 - 67.72643)
            + 837.5802 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 2.654827 * max(0.0, Q.mass - 172.4888) * max(0.0, Q.M2 - 0.04260132)
            + 0.03023939 * max(0.0, 74.25181 - Q.mass)
            + 0.0008305667 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.91979)
            + 1.801406 * max(0.0, 0.6133424 - Q.tau21_b2)
        ))
        - 0.375 * grid(12, max(0.0, -0.3564799
            + 0.0211521 * max(0.0, 87.36377 - Q.mass)
            - 0.05023374 * max(0.0, Q.sd_mass - 133.2575)
            - 5.702853 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3)
            + 0.05076254 * max(0.0, 82.85409 - Q.mass)
            - 590.416 * max(0.0, 0.00363788 - Q.e2_sq)
            - 0.009460028 * max(0.0, Q.sj3_pair_mass_max - 122.7494)
            + 0.001487776 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 65.20727 - Q.sj2_mass1)
            - 0.03205357 * max(0.0, 77.93668 - Q.mass_top40)
            - 0.04164589 * max(0.0, 53.87362 - Q.mass)
            + 0.003350507 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1)
            - 0.005710475 * max(0.0, 972.0419 - Q.sum_pt)
            - 16.22844 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            + 0.07374941 * max(0.0, 80.78464 - Q.mass)
        ))
        - 0.90625 * grid(13, max(0.0, 1.762228
            - 0.006507519 * max(0.0, 1085.125 - Q.sum_pt)
            - 0.1627033 * max(0.0, Q.mass - 143.7876)
            + 0.1196382 * max(0.0, Q.mass - 172.4888)
            + 0.0232466 * max(0.0, Q.mass - 64.48544)
            - 0.0345782 * max(0.0, 1012.673 - Q.sum_pt)
            + 8.640333 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 0.003602301 * max(0.0, 1053.047 - Q.sum_pt_top40)
            + 0.1028272 * max(0.0, Q.mass_top50 - 138.8977)
            - 0.120368 * max(0.0, Q.mass_top50 - 157.5448)
            + 6.504316 * max(0.0, 6.879399 - Q.log_sum_pt)
            + 0.007708099 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.5260785 - Q.D3)
            - 13.59186 * max(0.0, Q.mass_over_sum_pt - 0.07435617)
            - 0.03938065 * max(0.0, Q.mass_top50 - 97.93004)
            - 6.271736 * max(0.0, Q.mass_over_sum_pt - 0.06030419)
            + 0.01045673 * max(0.0, 1008.935 - Q.sum_pt_top50)
            + 0.1006837 * max(0.0, Q.mass_top40 - 150.0144)
            + 0.005318985 * max(0.0, Q.mass_top50 - 97.93004) * max(0.0, 2.873047 - Q.soft3_pt)
            + 0.01209646 * max(0.0, Q.mass - 78.26182)
        ))
        - 0.375 * grid(15, max(0.0, -0.956537
            + 5.010468 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2)
            + 159.9133 * max(0.0, 0.008329695 - Q.girth2_top5)
            - 0.09615909 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 58.0 - Q.n_particles)
            + 0.00647135 * max(0.0, 986.0565 - Q.sum_pt)
            + 5.932479 * max(0.0, 7.017258 - Q.log_sum_pt)
            - 213.8386 * max(0.0, 0.002270363 - Q.girth2_top5)
            + 283.84 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2)
            + 103.211 * max(0.0, 0.007678544 - Q.girth2_top10)
            - 28.87719 * max(0.0, 0.0705748 - Q.tau1)
            - 4.092722 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.n_real_top40 - 22.0)
            + 0.0217793 * max(0.0, 79.18312 - Q.sd_mass)
            - 0.01864383 * max(0.0, 40.97891 - Q.sd_mass)
            - 0.003186645 * max(0.0, 1018.832 - Q.sum_pt_top30)
            - 10.04477 * max(0.0, 6.903423 - Q.log_sum_pt)
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
