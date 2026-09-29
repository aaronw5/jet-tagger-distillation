"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network (step 4; no W/Z/H/t mass values offered as thresholds), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

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


def neuron_0(Q):
    z = 1.12
    if Q.e2_sq < 0.0061:
        z += 313.0 * Q.e2_sq - 1.9093
    if Q.girth2_top20 < 0.0073:
        z += 94.5 * Q.girth2_top20 - 0.68985
    if Q.mass < 78.0:
        z += 0.0115 * Q.mass - 1.15
    if 78.0 <= Q.mass < 87.0:
        z += -0.1345 * Q.mass + 10.238
    if 87.0 <= Q.mass < 93.0:
        z += -0.0524 * Q.mass + 3.0953
    if 93.0 <= Q.mass < 100.0:
        z += -0.3384 * Q.mass + 29.6933
    if Q.mass >= 100.0:
        z += -0.3499 * Q.mass + 30.8433
    if Q.mass_over_sum_pt_sq < 0.0077:
        z += -447.0 * Q.mass_over_sum_pt_sq + 3.4419
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.0453 * Q.n_dr_0p2_0p4 + 0.6795
    if Q.sum_pt < 980.0:
        z += 0.0082 * Q.sum_pt - 8.036
    if Q.sum_pt_top30 < 950.0:
        z += -0.0026 * Q.sum_pt_top30 + 2.47
    if Q.sum_pt_top40 < 1100.0:
        z += 0.0014 * Q.sum_pt_top40 - 1.54
    if Q.sum_pt_top50 < 1200.0:
        z += -0.00151 * Q.sum_pt_top50 + 1.812
    if Q.tau1 < 0.07:
        z += 19.3 * Q.tau1 - 1.351
    if Q.z_dr_0p2_0p4 < 0.091:
        z += 3.85 * Q.z_dr_0p2_0p4 - 0.35035
    if Q.z_top50_slots < 0.99:
        z += 17.1 * Q.z_top50_slots - 16.929
    if Q.log_sum_pt < 7.0 and Q.sum_pt_top50 > 960.0:
        z += 0.0885 * (7.0 - Q.log_sum_pt) * (Q.sum_pt_top50 - 960.0)
    if Q.mass > 78.0 and Q.sj2_zsoft < 0.07:
        z += 2.66 * (Q.mass - 78.0) * (0.07 - Q.sj2_zsoft)
    if Q.mass_top50 > 72.0 and Q.sj2_zsoft < 0.069:
        z += -1.71 * (Q.mass_top50 - 72.0) * (0.069 - Q.sj2_zsoft)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.34
    if Q.girth2_top15 < 0.00083:
        z += -893.0 * Q.girth2_top15 + 1.16603
    if 0.00083 <= Q.girth2_top15 < 0.0033:
        z += -172.0 * Q.girth2_top15 + 0.5676
    if Q.girth2_top5 < 0.001:
        z += -855.2 * Q.girth2_top5 + 1.58824
    if 0.001 <= Q.girth2_top5 < 0.0087:
        z += -95.2 * Q.girth2_top5 + 0.82824
    if Q.log_sum_pt < 6.9:
        z += 6.04 * Q.log_sum_pt - 42.884
    if 6.9 <= Q.log_sum_pt < 7.0:
        z += 40.84 * Q.log_sum_pt - 283.004
    if 7.0 <= Q.log_sum_pt < 7.1:
        z += 17.44 * Q.log_sum_pt - 119.204
    if Q.log_sum_pt >= 7.1:
        z += 11.4 * Q.log_sum_pt - 76.32
    if Q.m012 < 51.0:
        z += 0.0099 * Q.m012 - 0.5049
    if Q.mass < 88.0:
        z += -0.0081 * Q.mass + 0.3552
    if 88.0 <= Q.mass < 100.0:
        z += 0.0298 * Q.mass - 2.98
    if Q.mass_top20 < 48.0:
        z += 0.0275 * Q.mass_top20 - 1.32
    if Q.mass_top5 < 17.0:
        z += 0.0272 * Q.mass_top5 - 0.4624
    if Q.n_dr_0p1_0p2 < 7.6:
        z += 0.0485 * Q.n_dr_0p1_0p2 - 0.3686
    if Q.n_dr_0p2_0p4 < 7.4:
        z += 0.0998 * Q.n_dr_0p2_0p4 - 0.73852
    if Q.n_particles >= 38.0:
        z += 0.0334 * Q.n_particles - 1.2692
    if Q.sj2_mass1 < 32.0:
        z += 0.0234 * Q.sj2_mass1 - 0.7488
    if Q.soft1_pt < 1.6:
        z += 0.071 * Q.soft1_pt - 0.7849
    if 1.6 <= Q.soft1_pt < 2.3:
        z += 0.959 * Q.soft1_pt - 2.2057
    if Q.sum_pt < 1000.0:
        z += -0.00626 * Q.sum_pt + 6.26
    if Q.sum_pt_top2 < 700.0:
        z += -0.00226 * Q.sum_pt_top2 + 1.582
    if Q.sum_pt_top30 >= 1200.0:
        z += 0.00196 * Q.sum_pt_top30 - 2.352
    if Q.sum_pt_top40 < 1100.0:
        z += -0.00713 * Q.sum_pt_top40 + 7.843
    if Q.sum_pt_top50 >= 960.0:
        z += -0.00663 * Q.sum_pt_top50 + 6.3648
    if Q.tau2 < 0.078:
        z += -6.9 * Q.tau2 + 0.5382
    if Q.z_7 < 0.032:
        z += 30.2 * Q.z_7 - 0.9664
    if Q.z_dr_0_0p05 >= 0.85:
        z += -3.07 * Q.z_dr_0_0p05 + 2.6095
    if Q.z_top50_slots >= 0.96:
        z += -18.6 * Q.z_top50_slots + 17.856
    if Q.zdr_0 >= 0.00098:
        z += -28.6 * Q.zdr_0 + 0.028028
    if Q.M3 < 0.03 and Q.M2 > 0.046:
        z += -476.0 * (0.03 - Q.M3) * (Q.M2 - 0.046)
    if Q.mass_top20 < 48.0 and Q.n_real_top40 > 30.0:
        z += 0.00372 * (48.0 - Q.mass_top20) * (Q.n_real_top40 - 30.0)
    if Q.n_particles > 38.0 and Q.dr_1 < 0.15:
        z += 0.2 * (Q.n_particles - 38.0) * (0.15 - Q.dr_1)
    if Q.n_particles > 38.0 and Q.zdr_2 < 0.013:
        z += 1.81 * (Q.n_particles - 38.0) * (0.013 - Q.zdr_2)
    if Q.sj3_mass1 < 31.0 and Q.sj3_mass2 < 21.0:
        z += -0.00167 * (31.0 - Q.sj3_mass1) * (21.0 - Q.sj3_mass2)
    if Q.tau1 < 0.17 and Q.eta_0 > -0.016:
        z += 39.3 * (0.17 - Q.tau1) * (Q.eta_0 - -0.016)
    if Q.z_top30_slots > 0.93 and Q.C2 < 0.075:
        z += 172.0 * (Q.z_top30_slots - 0.93) * (0.075 - Q.C2)
    if Q.z_top30_slots > 0.93 and Q.max_pair_mass > 14.0:
        z += 0.408 * (Q.z_top30_slots - 0.93) * (Q.max_pair_mass - 14.0)
    if Q.z_top30_slots > 0.93 and Q.ptdr0_3 > 6.5:
        z += 0.74 * (Q.z_top30_slots - 0.93) * (Q.ptdr0_3 - 6.5)
    if Q.z_top30_slots > 0.96 and Q.ptdr0_4 > 3.7:
        z += 1.28 * (Q.z_top30_slots - 0.96) * (Q.ptdr0_4 - 3.7)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.0281
    if Q.log_sum_pt >= 7.1:
        z += -6.25 * Q.log_sum_pt + 44.375
    if Q.mass < 87.0:
        z += 0.0111 * Q.mass - 0.9657
    if Q.log_sum_pt > 6.9 and Q.girth2_top15 < 0.025:
        z += 320.0 * (Q.log_sum_pt - 6.9) * (0.025 - Q.girth2_top15)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.0429
    if Q.lam2 < 0.00061:
        z += -1060.0 * Q.lam2 + 0.6466
    if Q.mass < 56.0:
        z += 0.0212 * Q.mass - 1.1872
    if Q.n_particles < 46.0:
        z += -0.0364 * Q.n_particles + 1.6744
    if Q.tau21 < 0.38:
        z += 2.01 * Q.tau21 - 0.7638
    if Q.n_dr_0p1_0p2 < 14.0 and Q.psi_0p2 > 0.95:
        z += 0.802 * (14.0 - Q.n_dr_0p1_0p2) * (Q.psi_0p2 - 0.95)
    if Q.n_dr_0p2_0p4 < 6.6 and Q.sum_pt_top30 < 1000.0:
        z += -0.000709 * (6.6 - Q.n_dr_0p2_0p4) * (1000.0 - Q.sum_pt_top30)
    if Q.n_dr_0p2_0p4 < 6.4 and Q.z_top50_slots > 0.97:
        z += 2.92 * (6.4 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.97)
    if Q.tau21 < 0.38 and Q.lam1 > 0.0036:
        z += -312.0 * (0.38 - Q.tau21) * (Q.lam1 - 0.0036)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.0131
    if Q.D2 < 6.5:
        z += -0.138 * Q.D2 + 0.897
    if Q.LHA >= 0.37:
        z += -8.68 * Q.LHA + 3.2116
    if Q.e2 >= 0.038:
        z += 12.7 * Q.e2 - 0.4826
    if Q.e2_sq >= 0.014:
        z += 85.2 * Q.e2_sq - 1.1928
    if Q.girth2_top15 < 0.0054:
        z += 89.8 * Q.girth2_top15 - 0.48492
    if Q.girth2_top15 >= 0.0072:
        z += -28.6 * Q.girth2_top15 + 0.20592
    if Q.lam2 >= 0.00044:
        z += 53.7 * Q.lam2 - 0.023628
    if Q.mass < 74.0:
        z += 0.0791 * Q.mass - 5.5765
    if 74.0 <= Q.mass < 87.0:
        z += 0.0213 * Q.mass - 1.2993
    if 87.0 <= Q.mass < 100.0:
        z += -0.0426 * Q.mass + 4.26
    if Q.mass >= 140.0:
        z += 0.0134 * Q.mass - 1.876
    if Q.mass_top20 >= 100.0:
        z += -0.0163 * Q.mass_top20 + 1.63
    if Q.n_dr_0_0p05 >= 6.8:
        z += 0.0264 * Q.n_dr_0_0p05 - 0.17952
    if Q.n_dr_0p2_0p4 < 25.0:
        z += -0.0244 * Q.n_dr_0p2_0p4 + 0.61
    if Q.n_particles >= 21.0:
        z += -0.0257 * Q.n_particles + 0.5397
    if Q.sj3_pair_mass_min >= 34.0:
        z += 0.00629 * Q.sj3_pair_mass_min - 0.21386
    if Q.sum_pt < 1000.0:
        z += 0.00279 * Q.sum_pt - 2.79
    if Q.tau1 < 0.072:
        z += 18.7 * Q.tau1 - 1.3464
    if Q.mass < 80.0 and Q.lam2 < 0.0037:
        z += -17.3 * (80.0 - Q.mass) * (0.0037 - Q.lam2)
    if Q.mass < 100.0 and Q.lam2 < 0.004:
        z += 10.5 * (100.0 - Q.mass) * (0.004 - Q.lam2)
    if Q.mass < 86.0 and Q.zdr_0 > 0.0069:
        z += -3.83 * (86.0 - Q.mass) * (Q.zdr_0 - 0.0069)
    if Q.mass_top30 < 120.0 and Q.sum_pt_top40 < 1000.0:
        z += 3.09e-05 * (120.0 - Q.mass_top30) * (1000.0 - Q.sum_pt_top40)
    if Q.mass_top40 < 89.0 and Q.D2 < 6.8:
        z += -0.0105 * (89.0 - Q.mass_top40) * (6.8 - Q.D2)
    if Q.n_particles > 24.0 and Q.soft1_pt < 2.1:
        z += 0.00368 * (Q.n_particles - 24.0) * (2.1 - Q.soft1_pt)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.119
    if Q.D2 < 2.0:
        z += -0.387 * Q.D2 + 0.774
    if Q.e2 < 0.041:
        z += -16.8 * Q.e2 + 0.6888
    if Q.girth2_top50 >= 0.016:
        z += -52.8 * Q.girth2_top50 + 0.8448
    if Q.log_sum_pt >= 6.9:
        z += -19.4 * Q.log_sum_pt + 133.86
    if Q.mass >= 160.0:
        z += -0.0849 * Q.mass + 13.584
    if Q.mass_over_sum_pt >= 0.092:
        z += -18.9 * Q.mass_over_sum_pt + 1.7388
    if Q.mass_top40 >= 59.0:
        z += 0.0144 * Q.mass_top40 - 0.8496
    if Q.max_dr < 0.33:
        z += -1.69 * Q.max_dr + 0.7098
    if 0.33 <= Q.max_dr < 0.42:
        z += -2.257 * Q.max_dr + 0.89691
    if Q.max_dr >= 0.42:
        z += -0.567 * Q.max_dr + 0.18711
    if Q.n_dr_0p1_0p2 < 22.0:
        z += 0.0202 * Q.n_dr_0p1_0p2 - 0.4444
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.0369 * Q.n_dr_0p2_0p4 + 0.4059
    if Q.n_particles < 46.0:
        z += -0.0321 * Q.n_particles + 1.4766
    if Q.n_particles >= 46.0:
        z += -0.0204 * Q.n_particles + 0.9384
    if Q.psi_0p1 < 0.71:
        z += 0.91 * Q.psi_0p1 - 0.6461
    if 70.0 <= Q.sd_mass < 83.0:
        z += 0.0318 * Q.sd_mass - 2.226
    if Q.sd_mass >= 83.0:
        z += -0.0057 * Q.sd_mass + 0.8865
    if Q.sj3_mass1 < 23.0:
        z += -0.0186 * Q.sj3_mass1 + 0.4278
    if Q.soft3_pt >= 0.38:
        z += 0.361 * Q.soft3_pt - 0.13718
    if Q.soft3_z >= 0.00083:
        z += -490.0 * Q.soft3_z + 0.4067
    if 920.0 <= Q.sum_pt < 990.0:
        z += 0.0157 * Q.sum_pt - 14.444
    if Q.sum_pt >= 990.0:
        z += -0.0072 * Q.sum_pt + 8.227
    if Q.sum_pt_top50 >= 950.0:
        z += 0.019 * Q.sum_pt_top50 - 18.05
    if Q.tau21 < 0.49:
        z += 1.12 * Q.tau21 - 0.5488
    if Q.z_dr_0p2_0p4 < 0.036:
        z += 10.7 * Q.z_dr_0p2_0p4 - 0.3852
    if Q.z_top30_slots >= 0.9:
        z += -6.19 * Q.z_top30_slots + 5.571
    if Q.n_particles < 64.0 and Q.D2 < 2.2:
        z += -0.0196 * (64.0 - Q.n_particles) * (2.2 - Q.D2)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.27
    if Q.D2 < 2.8:
        z += 0.216 * Q.D2 - 0.6048
    if Q.e2 >= 0.027:
        z += -24.5 * Q.e2 + 0.6615
    if Q.mass < 84.0:
        z += -0.0359 * Q.mass + 0.96
    if 84.0 <= Q.mass < 120.0:
        z += 0.0571 * Q.mass - 6.852
    if Q.mass >= 140.0:
        z += -0.0211 * Q.mass + 2.954
    if Q.mass_top10 >= 95.0:
        z += 0.0406 * Q.mass_top10 - 3.857
    if Q.sj3_mass1 >= 22.0:
        z += -0.0274 * Q.sj3_mass1 + 0.6028
    if Q.sj3_pair_mass_min < 76.0:
        z += -0.00838 * Q.sj3_pair_mass_min + 0.63688
    if Q.e2 > 0.022 and Q.psi_0p3 > 0.99:
        z += 3900.0 * (Q.e2 - 0.022) * (Q.psi_0p3 - 0.99)
    if Q.mass < 150.0 and Q.sum_pt_top50 < 990.0:
        z += 3.63e-05 * (150.0 - Q.mass) * (990.0 - Q.sum_pt_top50)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.555
    if Q.LHA < 0.31:
        z += -13.9 * Q.LHA + 4.309
    if Q.girth2_top30 < 0.0068:
        z += 16.0 * Q.girth2_top30 + 0.158
    if 0.0068 <= Q.girth2_top30 < 0.0082:
        z += 336.0 * Q.girth2_top30 - 2.018
    if 0.0082 <= Q.girth2_top30 < 0.012:
        z += -194.0 * Q.girth2_top30 + 2.328
    if Q.girth2_top50 < 0.0034:
        z += 2740.0 * Q.girth2_top50 - 9.316
    if Q.mass < 81.0:
        z += -0.085 * Q.mass + 6.095
    if 81.0 <= Q.mass < 91.0:
        z += 0.022 * Q.mass - 2.572
    if 91.0 <= Q.mass < 93.0:
        z += 0.285 * Q.mass - 26.505
    if Q.tau1 < 0.11:
        z += 28.9 * Q.tau1 - 3.179
    if Q.tau21_b2 < 0.23:
        z += -3.47 * Q.tau21_b2 + 0.7981
    if Q.z_top20_slots >= 0.92:
        z += -10.7 * Q.z_top20_slots + 9.844
    if Q.girth2 < 0.0076 and Q.M2 < 0.12:
        z += 2490.0 * (0.0076 - Q.girth2) * (0.12 - Q.M2)
    if Q.mass < 91.0 and Q.psi_0p3 > 0.96:
        z += -8.38 * (91.0 - Q.mass) * (Q.psi_0p3 - 0.96)
    if Q.mass < 100.0 and Q.psi_0p3 > 0.96:
        z += 5.64 * (100.0 - Q.mass) * (Q.psi_0p3 - 0.96)
    if Q.mass < 91.0 and Q.sd_mass > 54.0:
        z += -0.00264 * (91.0 - Q.mass) * (Q.sd_mass - 54.0)
    if Q.mass < 120.0 and Q.sd_mass > 76.0:
        z += 0.00239 * (120.0 - Q.mass) * (Q.sd_mass - 76.0)
    if Q.n_dr_0p2_0p4 < 6.2 and Q.e3 < 6.7e-05:
        z += -1810.0 * (6.2 - Q.n_dr_0p2_0p4) * (6.7e-05 - Q.e3)
    if Q.n_dr_0p2_0p4 < 6.1 and Q.pt_4 > 60.0:
        z += 0.00298 * (6.1 - Q.n_dr_0p2_0p4) * (Q.pt_4 - 60.0)
    if Q.n_dr_0p2_0p4 < 6.3 and Q.soft1_pt < 2.3:
        z += 0.0741 * (6.3 - Q.n_dr_0p2_0p4) * (2.3 - Q.soft1_pt)
    if Q.tau21_b2 < 0.25 and Q.sj2_dr > 0.19:
        z += -22.2 * (0.25 - Q.tau21_b2) * (Q.sj2_dr - 0.19)
    return max(0.0, z)


def neuron_8(Q):
    z = -1.41
    if Q.e2 >= 0.047:
        z += 39.7 * Q.e2 - 1.8659
    if 0.008 <= Q.e2_sq < 0.0098:
        z += 387.0 * Q.e2_sq - 3.096
    if Q.e2_sq >= 0.0098:
        z += -79.0 * Q.e2_sq + 1.4708
    if Q.e3 < 0.00035:
        z += -2990.0 * Q.e3 + 1.0465
    if Q.girth2_top15 < 0.0082:
        z += -132.0 * Q.girth2_top15 - 0.1496
    if 0.0082 <= Q.girth2_top15 < 0.017:
        z += 140.0 * Q.girth2_top15 - 2.38
    if Q.log_sum_pt < 6.9:
        z += -10.1 * Q.log_sum_pt + 69.69
    if 65.0 <= Q.mass < 81.0:
        z += 0.0171 * Q.mass - 1.1115
    if 81.0 <= Q.mass < 100.0:
        z += 0.0866 * Q.mass - 6.741
    if 100.0 <= Q.mass < 120.0:
        z += 0.049 * Q.mass - 2.981
    if Q.mass >= 120.0:
        z += 0.0082 * Q.mass + 1.915
    if Q.mass_over_sum_pt_sq >= 0.00055:
        z += 233.0 * Q.mass_over_sum_pt_sq - 0.12815
    if Q.mass_top50 >= 81.0:
        z += -0.0651 * Q.mass_top50 + 5.2731
    if Q.n_dr_0p2_0p4 < 8.1:
        z += 0.0234 * Q.n_dr_0p2_0p4 - 0.3744
    if 8.1 <= Q.n_dr_0p2_0p4 < 16.0:
        z += 0.0558 * Q.n_dr_0p2_0p4 - 0.63684
    if Q.n_dr_0p2_0p4 >= 16.0:
        z += 0.0324 * Q.n_dr_0p2_0p4 - 0.26244
    if Q.n_pt_above_1 >= 55.0:
        z += 0.0296 * Q.n_pt_above_1 - 1.628
    if Q.sj2_dr >= 0.23:
        z += 4.27 * Q.sj2_dr - 0.9821
    if Q.girth2_top15 < 0.0094 and Q.tau21_b2 < 0.61:
        z += -163.0 * (0.0094 - Q.girth2_top15) * (0.61 - Q.tau21_b2)
    if Q.log_sum_pt > 7.0 and Q.sj3_pair_mass_max > 16.0:
        z += 0.044 * (Q.log_sum_pt - 7.0) * (Q.sj3_pair_mass_max - 16.0)
    if Q.n_dr_0p2_0p4 > 8.9 and Q.zdr_0 > 0.0039:
        z += -1.37 * (Q.n_dr_0p2_0p4 - 8.9) * (Q.zdr_0 - 0.0039)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.851
    if Q.LHA < 0.21:
        z += -13.8 * Q.LHA + 2.898
    if Q.girth < 0.047:
        z += 50.7 * Q.girth - 2.3829
    if Q.girth >= 0.093:
        z += 33.8 * Q.girth - 3.1434
    if Q.girth2_top30 >= 0.024:
        z += -122.0 * Q.girth2_top30 + 2.928
    if Q.girth2_top40 < 0.0061:
        z += -283.0 * Q.girth2_top40 + 1.7263
    if Q.mass < 64.0:
        z += -0.0167 * Q.mass + 2.4768
    if 64.0 <= Q.mass < 84.0:
        z += -0.0704 * Q.mass + 5.9136
    if 140.0 <= Q.mass < 160.0:
        z += -0.039 * Q.mass + 5.46
    if 160.0 <= Q.mass < 170.0:
        z += -0.1321 * Q.mass + 20.356
    if Q.mass >= 170.0:
        z += -0.0021 * Q.mass - 1.744
    if Q.mass_top40 < 130.0:
        z += 0.0003 * Q.mass_top40 - 1.017
    if 130.0 <= Q.mass_top40 < 160.0:
        z += 0.0326 * Q.mass_top40 - 5.216
    if Q.psi_0p3 >= 0.99:
        z += -26.0 * Q.psi_0p3 + 25.74
    if Q.sum_pt_top50 < 960.0:
        z += -0.00893 * Q.sum_pt_top50 + 8.5728
    if Q.z_dr_0p1_0p2 < 0.29:
        z += -1.22 * Q.z_dr_0p1_0p2 + 0.3538
    if Q.mass < 88.0 and Q.n_particles > 40.0:
        z += 0.000696 * (88.0 - Q.mass) * (Q.n_particles - 40.0)
    if Q.mass_top30 < 110.0 and Q.n_dr_0p2_0p4 > 11.0:
        z += 0.000726 * (110.0 - Q.mass_top30) * (Q.n_dr_0p2_0p4 - 11.0)
    if Q.mass_top40 < 72.0 and Q.sum_pt < 1000.0:
        z += -0.0003 * (72.0 - Q.mass_top40) * (1000.0 - Q.sum_pt)
    if Q.mass_top40 < 73.0 and Q.sum_pt_top50 < 940.0:
        z += 0.000433 * (73.0 - Q.mass_top40) * (940.0 - Q.sum_pt_top50)
    if Q.sum_pt_top40 < 930.0 and Q.n_pt_above_10 < 19.0:
        z += -0.000715 * (930.0 - Q.sum_pt_top40) * (19.0 - Q.n_pt_above_10)
    return max(0.0, z)


def neuron_10(Q):
    z = 0.67
    if Q.D2 < 3.8:
        z += -0.297 * Q.D2 + 1.1286
    if Q.M2 >= 0.055:
        z += -5.67 * Q.M2 + 0.31185
    if Q.dr_0 < 0.064:
        z += -6.11 * Q.dr_0 + 0.39104
    if Q.e2 < 0.062:
        z += 27.1 * Q.e2 - 1.6802
    if Q.girth2_top20 < 0.005:
        z += 47.8 * Q.girth2_top20 - 0.239
    if Q.girth2_top30 < 0.018:
        z += -125.0 * Q.girth2_top30 + 2.25
    if Q.log_sum_pt >= 6.9:
        z += -3.91 * Q.log_sum_pt + 26.979
    if 61.0 <= Q.mass < 81.0:
        z += -0.0497 * Q.mass + 3.0317
    if 81.0 <= Q.mass < 88.0:
        z += 0.0273 * Q.mass - 3.2053
    if 88.0 <= Q.mass < 140.0:
        z += 0.0464 * Q.mass - 4.8861
    if 140.0 <= Q.mass < 160.0:
        z += -0.0265 * Q.mass + 5.3199
    if Q.mass >= 160.0:
        z += -0.1117 * Q.mass + 18.9519
    if Q.mass_over_sum_pt < 0.078:
        z += -18.6 * Q.mass_over_sum_pt + 1.4508
    if Q.mass_over_sum_pt >= 0.17:
        z += -318.0 * Q.mass_over_sum_pt + 54.06
    if Q.mass_over_sum_pt_sq >= 0.029:
        z += 747.0 * Q.mass_over_sum_pt_sq - 21.663
    if Q.mass_top10 < 66.0:
        z += 0.00737 * Q.mass_top10 - 0.48642
    if Q.mass_top5 >= 15.0:
        z += 0.00873 * Q.mass_top5 - 0.13095
    if Q.mass_top50 >= 140.0:
        z += 0.0401 * Q.mass_top50 - 5.614
    if Q.sj2_mass1 < 78.0:
        z += 0.0147 * Q.sj2_mass1 - 1.1466
    if Q.sum_pt < 990.0:
        z += 0.00171 * Q.sum_pt - 1.6929
    if Q.sum_pt_top5 < 850.0:
        z += -0.0017 * Q.sum_pt_top5 + 1.445
    if Q.tau1 < 0.056:
        z += 43.2 * Q.tau1 - 2.4192
    if Q.D2 < 3.6 and Q.sj2_dr > 0.2:
        z += -3.55 * (3.6 - Q.D2) * (Q.sj2_dr - 0.2)
    if Q.mass > 90.0 and Q.psi_0p3 > 0.98:
        z += 0.146 * (Q.mass - 90.0) * (Q.psi_0p3 - 0.98)
    if Q.mass > 59.0 and Q.soft2_pt < 2.7:
        z += 0.00409 * (Q.mass - 59.0) * (2.7 - Q.soft2_pt)
    if Q.mass_top40 < 110.0 and Q.pt1_dr01 > 13.0:
        z += 0.000774 * (110.0 - Q.mass_top40) * (Q.pt1_dr01 - 13.0)
    if Q.mass_top40 > 160.0 and Q.soft4_z > 0.0011:
        z += 28.4 * (Q.mass_top40 - 160.0) * (Q.soft4_z - 0.0011)
    if Q.sj2_mass1 < 89.0 and Q.sum_pt_top2 < 530.0:
        z += 2.53e-05 * (89.0 - Q.sj2_mass1) * (530.0 - Q.sum_pt_top2)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.551
    if Q.LHA < 0.24:
        z += -6.03 * Q.LHA + 1.4472
    if Q.girth2_top10 < 0.0043:
        z += -224.0 * Q.girth2_top10 + 0.9632
    if Q.girth2_top2 < 0.0058:
        z += 85.9 * Q.girth2_top2 - 0.49822
    if Q.lam1 < 0.0065:
        z += 161.0 * Q.lam1 - 1.0465
    if Q.mass < 81.0:
        z += 0.0001 * Q.mass + 0.8379
    if 81.0 <= Q.mass < 96.0:
        z += -0.0564 * Q.mass + 5.4144
    if Q.n_dr_0p1_0p2 < 17.0:
        z += -0.0379 * Q.n_dr_0p1_0p2 + 0.6443
    if Q.n_dr_0p2_0p4 < 9.0:
        z += -0.143 * Q.n_dr_0p2_0p4 + 1.287
    if Q.psi_0p3 >= 0.99:
        z += 45.6 * Q.psi_0p3 - 45.144
    if Q.tau21_b2 < 0.15:
        z += -4.2 * Q.tau21_b2 + 0.63
    if Q.z_top10_slots >= 0.75:
        z += -1.55 * Q.z_top10_slots + 1.1625
    if Q.mass < 84.0 and Q.D2 < 3.6:
        z += -0.00729 * (84.0 - Q.mass) * (3.6 - Q.D2)
    if Q.mass < 95.0 and Q.mass_top10 > 38.0:
        z += 0.00118 * (95.0 - Q.mass) * (Q.mass_top10 - 38.0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2 < 0.0061:
        z += -28.5 * (10.0 - Q.n_dr_0p2_0p4) * (0.0061 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_particles > 36.0:
        z += -0.0027 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_particles - 36.0)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.427
    if Q.mass < 56.0:
        z += -0.0336 * Q.mass + 4.7096
    if 56.0 <= Q.mass < 84.0:
        z += -0.101 * Q.mass + 8.484
    if Q.sum_pt < 970.0:
        z += 0.0068 * Q.sum_pt - 6.596
    if Q.sum_pt >= 1200.0:
        z += -0.00243 * Q.sum_pt + 2.916
    if Q.z_dr_0p1_0p2 >= 0.4:
        z += 1.4 * Q.z_dr_0p1_0p2 - 0.56
    if Q.mass < 90.0 and Q.psi_0p3 < 1.0:
        z += -2.72 * (90.0 - Q.mass) * (1.0 - Q.psi_0p3)
    if Q.mass < 90.0 and Q.z_dr_0p05_0p1 > 0.4:
        z += 0.0785 * (90.0 - Q.mass) * (Q.z_dr_0p05_0p1 - 0.4)
    if Q.mass < 88.0 and Q.z_dr_0p1_0p2 < 0.075:
        z += -0.209 * (88.0 - Q.mass) * (0.075 - Q.z_dr_0p1_0p2)
    if Q.sd_mass > 130.0 and Q.sj3_mass1 < 37.0:
        z += 0.00124 * (Q.sd_mass - 130.0) * (37.0 - Q.sj3_mass1)
    if Q.sj3_pair_mass_max > 130.0 and Q.pt_0 < 530.0:
        z += 4.5e-05 * (Q.sj3_pair_mass_max - 130.0) * (530.0 - Q.pt_0)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.86
    if 77.0 <= Q.mass < 140.0:
        z += 0.0195 * Q.mass - 1.5015
    if Q.mass >= 140.0:
        z += -0.1275 * Q.mass + 19.0785
    if Q.mass_over_sum_pt >= 0.17:
        z += 103.0 * Q.mass_over_sum_pt - 17.51
    if Q.mass_top40 >= 150.0:
        z += 0.0722 * Q.mass_top40 - 10.83
    if 97.0 <= Q.mass_top50 < 140.0:
        z += -0.016 * Q.mass_top50 + 1.552
    if 140.0 <= Q.mass_top50 < 160.0:
        z += 0.0669 * Q.mass_top50 - 10.054
    if Q.mass_top50 >= 160.0:
        z += -0.0611 * Q.mass_top50 + 10.426
    if Q.psi_0p3 >= 0.98:
        z += 17.4 * Q.psi_0p3 - 17.052
    if Q.sum_pt < 1000.0:
        z += 0.04124 * Q.sum_pt - 42.224
    if 1000.0 <= Q.sum_pt < 1100.0:
        z += 0.00984 * Q.sum_pt - 10.824
    if Q.sum_pt_top20 < 880.0:
        z += 0.00215 * Q.sum_pt_top20 - 1.892
    if Q.sum_pt_top40 < 1100.0:
        z += -0.0042 * Q.sum_pt_top40 + 4.62
    if Q.sum_pt_top50 < 1000.0:
        z += -0.0139 * Q.sum_pt_top50 + 13.9
    if Q.sum_pt < 990.0 and Q.z_9 < 0.018:
        z += 1.27 * (990.0 - Q.sum_pt) * (0.018 - Q.z_9)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.24
    if Q.e2 >= 0.018:
        z += -7.59 * Q.e2 + 0.13662
    if Q.log_sum_pt >= 6.9:
        z += 8.03 * Q.log_sum_pt - 55.407
    if Q.mass < 83.0:
        z += 0.1501 * Q.mass - 13.0063
    if 83.0 <= Q.mass < 91.0:
        z += 0.0685 * Q.mass - 6.2335
    if Q.mass_over_sum_pt < 0.091:
        z += 38.1 * Q.mass_over_sum_pt - 2.7073
    if 0.091 <= Q.mass_over_sum_pt < 0.12:
        z += -26.2 * Q.mass_over_sum_pt + 3.144
    if Q.mass_top50 >= 130.0:
        z += -0.0127 * Q.mass_top50 + 1.651
    if Q.n_dr_0p1_0p2 < 20.0:
        z += 0.0215 * Q.n_dr_0p1_0p2 - 0.43
    if Q.n_pt_above_10 < 24.0:
        z += -0.0217 * Q.n_pt_above_10 + 0.5208
    if Q.sum_pt_top50 >= 950.0:
        z += -0.00635 * Q.sum_pt_top50 + 6.0325
    if Q.tau21_b2 < 0.36:
        z += -1.87 * Q.tau21_b2 + 0.6732
    if Q.N2 < 0.39 and Q.max_dr > 0.25:
        z += -13.8 * (0.39 - Q.N2) * (Q.max_dr - 0.25)
    if Q.mass < 83.0 and Q.n_for_50pct > 4.3:
        z += 0.0147 * (83.0 - Q.mass) * (Q.n_for_50pct - 4.3)
    if Q.psi_0p3 > 0.99 and Q.N3 < 1.1:
        z += 68.5 * (Q.psi_0p3 - 0.99) * (1.1 - Q.N3)
    if Q.psi_0p3 > 0.99 and Q.pt_3 > 63.0:
        z += 0.464 * (Q.psi_0p3 - 0.99) * (Q.pt_3 - 63.0)
    if Q.tau21_b2 < 0.34 and Q.orientation_deg > -13.0:
        z += -0.0163 * (0.34 - Q.tau21_b2) * (Q.orientation_deg - -13.0)
    if Q.tau21_b2 < 0.38 and Q.z_1 < 0.12:
        z += -19.1 * (0.38 - Q.tau21_b2) * (0.12 - Q.z_1)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.227
    if Q.girth2_top5 < 0.009:
        z += -93.7 * Q.girth2_top5 + 0.8433
    if Q.lam1 >= 0.017:
        z += -372.0 * Q.lam1 + 6.324
    if Q.lam2 < 0.002:
        z += -234.0 * Q.lam2 + 0.468
    if Q.log_sum_pt < 7.0:
        z += -5.95 * Q.log_sum_pt + 41.65
    if Q.n_dr_0p2_0p4 >= 8.6:
        z += -0.0398 * Q.n_dr_0p2_0p4 + 0.34228
    if Q.psi_0p2 >= 0.86:
        z += -4.07 * Q.psi_0p2 + 3.5002
    if Q.psi_0p3 >= 0.99:
        z += -44.4 * Q.psi_0p3 + 43.956
    if Q.sd_mass < 43.0:
        z += 0.007 * Q.sd_mass + 0.3788
    if 43.0 <= Q.sd_mass < 76.0:
        z += -0.0206 * Q.sd_mass + 1.5656
    if Q.sj2_dr >= 0.23:
        z += 4.54 * Q.sj2_dr - 1.0442
    if Q.sj3_dr_max >= 0.36:
        z += -3.0 * Q.sj3_dr_max + 1.08
    if Q.sum_pt < 1000.0:
        z += 0.0135 * Q.sum_pt - 13.5
    if Q.tau1 < 0.069:
        z += 21.0 * Q.tau1 - 1.449
    if Q.z_dr_0_0p05 < 0.97:
        z += 0.676 * Q.z_dr_0_0p05 - 0.65572
    if Q.z_dr_0p1_0p2 < 0.12:
        z += -5.92 * Q.z_dr_0p1_0p2 + 0.7104
    if Q.girth2_top5 < 0.0054 and Q.sj3_mass1 > 5.0:
        z += -3.51 * (0.0054 - Q.girth2_top5) * (Q.sj3_mass1 - 5.0)
    if Q.girth2_top5 < 0.0094 and Q.sj3_pairmin_over_m > 0.08:
        z += -103.0 * (0.0094 - Q.girth2_top5) * (Q.sj3_pairmin_over_m - 0.08)
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    return [B[c] + sum(h[j] * W[j][c] for j in range(len(h))) for c in range(5)]


def classify(pt, eta, phi):
    s = logits(jet_layer_4(quantities(pt, eta, phi)))
    m = max(s)
    e = [math.exp(x - m) for x in s]
    p = [x / sum(e) for x in e]
    return CLASSES[s.index(m)], s, p


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:64] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:64] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:64] + [0.0] * 56
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
