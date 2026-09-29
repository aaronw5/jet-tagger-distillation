"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; no W/Z/H/t mass values offered as thresholds), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.6% (the network: 65.8%); same class as the network for 87.6% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
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
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_6               |Δη| of particle 6
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        z_top5=sum(zs[:5]),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
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
        pair_mass_0_6=pair_mass(0, 6),
        mass_top5=mass_of(5),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_for_90pct=ncum(0.9),
        pt_0=pt[0],
        pt_1=pt[1],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_6=abs(eta[6]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -3.39
    if Q.N2 < 0.194:
        z += 4.35 * Q.N2 - 0.9048
    if 0.194 <= Q.N2 < 0.208:
        z += 11.99 * Q.N2 - 2.38696
    if Q.N2 >= 0.208:
        z += 7.64 * Q.N2 - 1.48216
    if Q.dr_4 >= 0.0459:
        z += -2.62 * Q.dr_4 + 0.120258
    if Q.e2 < 0.028:
        z += -120.0 * Q.e2 + 3.36
    if Q.e2 >= 0.0281:
        z += -30.0 * Q.e2 + 0.843
    if Q.girth < 0.0774:
        z += 66.9 * Q.girth - 5.17806
    if Q.girth2 < 0.0127:
        z += -1452.0 * Q.girth2 + 22.1396
    if 0.0127 <= Q.girth2 < 0.0191:
        z += -578.0 * Q.girth2 + 11.0398
    if Q.girth2_top5 < 0.0163:
        z += 155.0 * Q.girth2_top5 - 2.5265
    if Q.lam2 < 0.000955:
        z += -1361.0 * Q.lam2 + 1.17465
    if Q.lam2 >= 0.000955:
        z += -131.0 * Q.lam2
    if Q.mass < 34.9:
        z += 0.0622 * Q.mass - 3.83152
    if 34.9 <= Q.mass < 61.6:
        z += 0.1104 * Q.mass - 5.5137
    if Q.mass >= 61.6:
        z += 0.0482 * Q.mass - 1.68218
    if Q.mass_over_sum_pt < 0.129:
        z += 137.0 * Q.mass_over_sum_pt - 17.673
    if Q.max_dr < 0.162:
        z += -7.07 * Q.max_dr + 1.14534
    if Q.mean_eta < -0.0056:
        z += 11.9 * Q.mean_eta + 0.06664
    if 0.118 <= Q.sj3_dr_max < 0.177:
        z += 26.5 * Q.sj3_dr_max - 3.127
    if 0.177 <= Q.sj3_dr_max < 0.222:
        z += 5.2 * Q.sj3_dr_max + 0.6431
    if Q.sj3_dr_max >= 0.222:
        z += -6.4 * Q.sj3_dr_max + 3.2183
    if Q.sj3_pair_mass_min >= 12.8:
        z += -0.0817 * Q.sj3_pair_mass_min + 1.04576
    if Q.sum_pt >= 921.0:
        z += -0.00874 * Q.sum_pt + 8.04954
    if Q.girth2 < 0.0134 and Q.mean_phi < -0.00046:
        z += -2350.0 * (0.0134 - Q.girth2) * (-0.00046 - Q.mean_phi)
    if Q.lam2 < 6.47e-05 and Q.dr_2 < 0.0942:
        z += 180000.0 * (6.47e-05 - Q.lam2) * (0.0942 - Q.dr_2)
    if Q.log_sum_pt > 6.36 and Q.mean_eta > 0.000739:
        z += -65.7 * (Q.log_sum_pt - 6.36) * (Q.mean_eta - 0.000739)
    if Q.log_sum_pt > 6.66 and Q.z_7 > 0.014:
        z += -265.0 * (Q.log_sum_pt - 6.66) * (Q.z_7 - 0.014)
    if Q.sum_pt > 803.0 and Q.pt_7 > 36.5:
        z += 0.000189 * (Q.sum_pt - 803.0) * (Q.pt_7 - 36.5)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.823
    if Q.centroid_offset < 0.018:
        z += 9.38 * Q.centroid_offset - 0.16884
    if Q.e2_sq < 0.00414:
        z += 276.0 * Q.e2_sq - 1.14264
    if Q.girth < 0.0582:
        z += -32.0 * Q.girth + 1.8624
    if Q.girth2 < 0.00864:
        z += 310.0 * Q.girth2 - 2.6784
    if Q.lam1 < 0.000777:
        z += 1140.0 * Q.lam1 - 0.88578
    if Q.log_sum_pt >= 6.47:
        z += 9.85 * Q.log_sum_pt - 63.7295
    if Q.mass < 48.2:
        z += -0.0297 * Q.mass + 1.43154
    if Q.n_pt_above_50 >= 5.28:
        z += 0.268 * Q.n_pt_above_50 - 1.41504
    if 41.8 <= Q.pt_7 < 54.2:
        z += 0.0633 * Q.pt_7 - 2.64594
    if Q.pt_7 >= 54.2:
        z += -0.0467 * Q.pt_7 + 3.31606
    if Q.sj3_dr_max >= 0.165:
        z += 2.39 * Q.sj3_dr_max - 0.39435
    if Q.sj3_dr_min >= 0.00774:
        z += -2.04 * Q.sj3_dr_min + 0.0157896
    if 375.0 <= Q.sum_pt_top5 < 502.0:
        z += 0.0051 * Q.sum_pt_top5 - 1.9125
    if Q.sum_pt_top5 >= 502.0:
        z += -0.00415 * Q.sum_pt_top5 + 2.731
    if Q.girth < 0.0984 and Q.pair_mass_0_6 > 5.49:
        z += -0.492 * (0.0984 - Q.girth) * (Q.pair_mass_0_6 - 5.49)
    if Q.girth2 < 0.00866 and Q.centroid_offset > 0.0199:
        z += 10700.0 * (0.00866 - Q.girth2) * (Q.centroid_offset - 0.0199)
    if Q.lam1 < 0.00795 and Q.n_pt_above_50 > 4.72:
        z += -82.8 * (0.00795 - Q.lam1) * (Q.n_pt_above_50 - 4.72)
    if Q.lam1 < 0.0115 and Q.planar_flow < 0.297:
        z += -254.0 * (0.0115 - Q.lam1) * (0.297 - Q.planar_flow)
    if Q.log_sum_pt > 6.6 and Q.D2 < 1.32:
        z += 5.19 * (Q.log_sum_pt - 6.6) * (1.32 - Q.D2)
    if Q.log_sum_pt > 6.35 and Q.centroid_offset > 0.00653:
        z += 53.6 * (Q.log_sum_pt - 6.35) * (Q.centroid_offset - 0.00653)
    if Q.log_sum_pt > 6.28 and Q.z_dr_0p05_0p1 < 0.787:
        z += -2.54 * (Q.log_sum_pt - 6.28) * (0.787 - Q.z_dr_0p05_0p1)
    if Q.log_sum_pt > 6.46 and Q.zdr_6 < 0.00812:
        z += -325.0 * (Q.log_sum_pt - 6.46) * (0.00812 - Q.zdr_6)
    if Q.mass_over_sum_pt_sq < 0.0081 and Q.centroid_offset > 0.00376:
        z += 3790.0 * (0.0081 - Q.mass_over_sum_pt_sq) * (Q.centroid_offset - 0.00376)
    if Q.pt_7 > 32.2 and Q.n_dr_0p1_0p2 > 0.624:
        z += 0.00798 * (Q.pt_7 - 32.2) * (Q.n_dr_0p1_0p2 - 0.624)
    if Q.pt_7 > 26.8 and Q.sj2_dr > 0.142:
        z += 0.223 * (Q.pt_7 - 26.8) * (Q.sj2_dr - 0.142)
    if Q.sj3_dr_max > 0.105 and Q.sj3_pair_mass_min > 1.87:
        z += -0.0724 * (Q.sj3_dr_max - 0.105) * (Q.sj3_pair_mass_min - 1.87)
    if Q.z_7 < 0.0636 and Q.D2 < 1.34:
        z += -26.8 * (0.0636 - Q.z_7) * (1.34 - Q.D2)
    if Q.z_7 < 0.0645 and Q.centroid_offset > 0.0194:
        z += -630.0 * (0.0645 - Q.z_7) * (Q.centroid_offset - 0.0194)
    if Q.z_7 < 0.0509 and Q.z_dr_0p05_0p1 > 0.0245:
        z += -47.6 * (0.0509 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.0245)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.595
    if Q.LHA >= 0.116:
        z += -7.05 * Q.LHA + 0.8178
    if Q.dr_5 < 0.0213:
        z += 29.0 * Q.dr_5 - 0.6177
    if Q.girth < 0.00797:
        z += 197.0 * Q.girth - 1.57009
    if Q.girth2 < 0.00532:
        z += -279.0 * Q.girth2 + 1.48428
    if Q.lam1 < 0.0118:
        z += -162.0 * Q.lam1 + 1.9116
    if Q.log_sum_pt >= 6.84:
        z += 12.8 * Q.log_sum_pt - 87.552
    if Q.pt_6 >= 34.2:
        z += 0.0242 * Q.pt_6 - 0.82764
    if Q.pt_7 >= 21.8:
        z += 0.0701 * Q.pt_7 - 1.52818
    if Q.sj3_pair_mass_min < 15.8:
        z += 0.0439 * Q.sj3_pair_mass_min - 0.69362
    if Q.sum_pt < 792.0:
        z += -0.0185 * Q.sum_pt + 14.652
    if Q.sum_pt_top5 < 719.0:
        z += 0.00929 * Q.sum_pt_top5 - 6.67951
    if Q.z_7 < 0.0231:
        z += 115.0 * Q.z_7 - 2.6565
    if Q.z_7 >= 0.0382:
        z += -23.6 * Q.z_7 + 0.90152
    if Q.lam1 < 0.0112 and Q.centroid_offset > 0.0168:
        z += -5750.0 * (0.0112 - Q.lam1) * (Q.centroid_offset - 0.0168)
    if Q.lam1 < 0.00574 and Q.max_dr > 0.0998:
        z += -1900.0 * (0.00574 - Q.lam1) * (Q.max_dr - 0.0998)
    if Q.log_sum_pt > 6.89 and Q.pt_6 > 32.7:
        z += -0.21 * (Q.log_sum_pt - 6.89) * (Q.pt_6 - 32.7)
    if Q.sj3_pair_mass_max < 70.9 and Q.z_7 < 0.0596:
        z += -0.431 * (70.9 - Q.sj3_pair_mass_max) * (0.0596 - Q.z_7)
    if Q.sum_pt < 558.0 and Q.D2_b2 < 4.13:
        z += 0.00186 * (558.0 - Q.sum_pt) * (4.13 - Q.D2_b2)
    if Q.sum_pt < 806.0 and Q.dr_5 < 0.0246:
        z += -1.46 * (806.0 - Q.sum_pt) * (0.0246 - Q.dr_5)
    if Q.sum_pt < 849.0 and Q.dr_5 < 0.0243:
        z += 1.38 * (849.0 - Q.sum_pt) * (0.0243 - Q.dr_5)
    return max(0.0, z)


def neuron_3(Q):
    z = -21.6
    if Q.centroid_offset < 0.00899:
        z += 22.5 * Q.centroid_offset - 0.558
    if 0.00899 <= Q.centroid_offset < 0.0248:
        z += -9.6 * Q.centroid_offset - 0.269421
    if Q.centroid_offset >= 0.0248:
        z += -32.1 * Q.centroid_offset + 0.288579
    if Q.e2 < 0.0493:
        z += -75.4 * Q.e2 + 3.71722
    if Q.girth >= 0.0576:
        z += 41.5 * Q.girth - 2.3904
    if Q.lam1 < 0.0059:
        z += -713.0 * Q.lam1 + 4.987
    if 0.0059 <= Q.lam1 < 0.0086:
        z += -289.0 * Q.lam1 + 2.4854
    if Q.lam2 < 4.83e-05:
        z += 16900.0 * Q.lam2 - 0.81627
    if 6.13 <= Q.log_sum_pt < 6.87:
        z += -1.84 * Q.log_sum_pt + 11.2792
    if Q.log_sum_pt >= 6.87:
        z += 2.91 * Q.log_sum_pt - 21.3533
    if Q.mass >= 67.1:
        z += -0.036 * Q.mass + 2.4156
    if 0.088 <= Q.mass_over_sum_pt < 0.132:
        z += 559.0 * Q.mass_over_sum_pt - 49.192
    if Q.mass_over_sum_pt >= 0.132:
        z += -8.0 * Q.mass_over_sum_pt + 25.652
    if Q.mass_over_sum_pt_sq < 0.00754:
        z += 30.0 * Q.mass_over_sum_pt_sq + 22.7646
    if 0.00754 <= Q.mass_over_sum_pt_sq < 0.0172:
        z += -2380.0 * Q.mass_over_sum_pt_sq + 40.936
    if Q.planar_flow < 0.0368:
        z += -28.7 * Q.planar_flow + 1.05616
    if Q.sj3_dr_max < 0.195:
        z += -6.1 * Q.sj3_dr_max - 0.3864
    if 0.195 <= Q.sj3_dr_max < 0.246:
        z += 30.9 * Q.sj3_dr_max - 7.6014
    if Q.tau21_b2 < 0.355:
        z += -1.77 * Q.tau21_b2 + 0.62835
    if Q.width < 0.00905:
        z += 995.0 * Q.width - 9.00475
    if Q.centroid_offset > 0.00578 and Q.n_pt_above_50 < 7.78:
        z += 3.81 * (Q.centroid_offset - 0.00578) * (7.78 - Q.n_pt_above_50)
    if Q.mass_over_sum_pt > 0.0794 and Q.pt_6 > 29.8:
        z += -0.684 * (Q.mass_over_sum_pt - 0.0794) * (Q.pt_6 - 29.8)
    if Q.sd_mass > 55.6 and Q.D2_b2 < 0.752:
        z += 0.0488 * (Q.sd_mass - 55.6) * (0.752 - Q.D2_b2)
    return max(0.0, z)


def neuron_4(Q):
    z = 2.27
    if Q.C2_b2 >= 0.000674:
        z += -132.0 * Q.C2_b2 + 0.088968
    if Q.D2 < 1.09:
        z += -4.31 * Q.D2 + 4.6979
    if Q.N2 < 0.229:
        z += -29.9 * Q.N2 + 6.8471
    if Q.dr_5 < 0.173:
        z += 6.16 * Q.dr_5 - 1.06568
    if Q.dr_6 < 0.18:
        z += 5.92 * Q.dr_6 - 1.0656
    if Q.dr_7 < 0.186:
        z += 5.93 * Q.dr_7 - 1.10298
    if 0.00829 <= Q.e2 < 0.0456:
        z += 133.0 * Q.e2 - 1.10257
    if Q.e2 >= 0.0456:
        z += 210.3 * Q.e2 - 4.62745
    if Q.e2_sq < 7.11e-05:
        z += -944.0 * Q.e2_sq + 7.39152
    if 7.11e-05 <= Q.e2_sq < 0.00783:
        z += -4054.0 * Q.e2_sq + 7.612641
    if Q.e2_sq >= 0.00783:
        z += -3110.0 * Q.e2_sq + 0.221121
    if Q.girth < 0.014:
        z += -24.1 * Q.girth + 8.2376
    if 0.014 <= Q.girth < 0.128:
        z += -69.3 * Q.girth + 8.8704
    if Q.girth2 < 0.00438:
        z += 1512.0 * Q.girth2 - 17.04984
    if 0.00438 <= Q.girth2 < 0.00808:
        z += 1300.0 * Q.girth2 - 16.12128
    if 0.00808 <= Q.girth2 < 0.0188:
        z += 524.0 * Q.girth2 - 9.8512
    if Q.lam1 < 0.00016:
        z += 41400.0 * Q.lam1 - 6.624
    if Q.lam2 < 4.19e-05:
        z += 17120.0 * Q.lam2 - 2.53976
    if 4.19e-05 <= Q.lam2 < 0.000267:
        z += 3820.0 * Q.lam2 - 1.98249
    if 0.000267 <= Q.lam2 < 0.00098:
        z += 1350.0 * Q.lam2 - 1.323
    if Q.log_sum_pt < 6.59:
        z += 8.42 * Q.log_sum_pt - 55.4878
    if Q.mass < 14.2:
        z += -0.0672 * Q.mass + 4.91624
    if 14.2 <= Q.mass < 40.7:
        z += -0.14 * Q.mass + 5.95
    if 40.7 <= Q.mass < 42.5:
        z += -0.26 * Q.mass + 10.834
    if Q.mass >= 42.5:
        z += -0.12 * Q.mass + 4.884
    if Q.mass_over_sum_pt >= 0.091:
        z += -250.0 * Q.mass_over_sum_pt + 22.75
    z += 3490.0 * Q.mass_over_sum_pt_sq
    if Q.mass_top5 >= 14.6:
        z += 0.0262 * Q.mass_top5 - 0.38252
    if Q.max_dr >= 0.123:
        z += 19.8 * Q.max_dr - 2.4354
    if Q.mean_eta < -0.0171:
        z += -31.1 * Q.mean_eta - 0.0061267
    if -0.0171 <= Q.mean_eta < -0.000197:
        z += -13.7 * Q.mean_eta + 0.2914133
    if Q.mean_eta >= -0.000197:
        z += 17.4 * Q.mean_eta + 0.29754
    if Q.mean_phi < -0.011:
        z += -50.5 * Q.mean_phi - 0.5555
    if Q.planar_flow < 0.0362:
        z += -15.8 * Q.planar_flow + 0.57196
    if Q.pt_7 < 24.9:
        z += 0.0914 * Q.pt_7 - 2.27586
    if Q.pt_7 >= 40.6:
        z += 0.0234 * Q.pt_7 - 0.95004
    if Q.sd_mass >= 31.4:
        z += 0.0362 * Q.sd_mass - 1.13668
    if 0.212 <= Q.sd_rg < 0.284:
        z += -28.4 * Q.sd_rg + 6.0208
    if Q.sd_rg >= 0.284:
        z += 4.3 * Q.sd_rg - 3.266
    if Q.sj2_dr >= 0.286:
        z += -23.6 * Q.sj2_dr + 6.7496
    if Q.sj3_dr_max < 0.232:
        z += 8.8 * Q.sj3_dr_max - 3.0096
    if 0.232 <= Q.sj3_dr_max < 0.34:
        z += -10.8 * Q.sj3_dr_max + 1.5376
    if 0.34 <= Q.sj3_dr_max < 0.342:
        z += 18.4 * Q.sj3_dr_max - 8.3904
    if Q.sj3_dr_max >= 0.342:
        z += 9.6 * Q.sj3_dr_max - 5.3808
    if Q.sj3_pair_mass_min >= 11.3:
        z += -0.0374 * Q.sj3_pair_mass_min + 0.42262
    if Q.sum_pt >= 711.0:
        z += 0.00933 * Q.sum_pt - 6.63363
    if Q.sum_pt_top3 >= 738.0:
        z += -0.0045 * Q.sum_pt_top3 + 3.321
    if Q.tau1 >= 0.0658:
        z += -32.7 * Q.tau1 + 2.15166
    if Q.z_6 < 0.0683:
        z += 20.1 * Q.z_6 - 1.37283
    if Q.z_dr_0p05_0p1 < 0.602:
        z += -0.657 * Q.z_dr_0p05_0p1 + 0.395514
    if Q.C2 < 0.0705 and Q.mean_eta > 0.0232:
        z += -1060.0 * (0.0705 - Q.C2) * (Q.mean_eta - 0.0232)
    if Q.D2 < 1.0 and Q.pt_7 < 54.6:
        z += -0.0666 * (1.0 - Q.D2) * (54.6 - Q.pt_7)
    if Q.N2 < 0.223 and Q.eccentricity > 0.695:
        z += -76.6 * (0.223 - Q.N2) * (Q.eccentricity - 0.695)
    if Q.centroid_offset < 0.0235 and Q.z_dr_0p05_0p1 < 0.585:
        z += -116.0 * (0.0235 - Q.centroid_offset) * (0.585 - Q.z_dr_0p05_0p1)
    if Q.lam2 < 0.00119 and Q.mean_phi < -0.0134:
        z += -40800.0 * (0.00119 - Q.lam2) * (-0.0134 - Q.mean_phi)
    if Q.mass < 76.4 and Q.D2 < 1.1:
        z += -0.115 * (76.4 - Q.mass) * (1.1 - Q.D2)
    if Q.sd_mass > 46.7 and Q.pt_0 < 171.0:
        z += -0.000674 * (Q.sd_mass - 46.7) * (171.0 - Q.pt_0)
    if Q.sd_rg > 0.209 and Q.pt_0 < 181.0:
        z += 0.226 * (Q.sd_rg - 0.209) * (181.0 - Q.pt_0)
    if Q.tau1 > 0.0633 and Q.sj3_dr_min < 0.212:
        z += 92.2 * (Q.tau1 - 0.0633) * (0.212 - Q.sj3_dr_min)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.785
    if Q.e2_sq >= 7.74e-06:
        z += -74.1 * Q.e2_sq + 0.000573534
    if Q.lam1 < 0.00275:
        z += -205.0 * Q.lam1 + 0.56375
    if Q.log_sum_pt < 6.7:
        z += 4.13 * Q.log_sum_pt - 27.671
    if Q.mass >= 58.0:
        z += 0.0282 * Q.mass - 1.6356
    if Q.pt_7 < 24.3:
        z += -0.141 * Q.pt_7 + 5.2593
    if 24.3 <= Q.pt_7 < 37.1:
        z += -0.0523 * Q.pt_7 + 3.10389
    if 37.1 <= Q.pt_7 < 37.3:
        z += -0.2223 * Q.pt_7 + 9.41089
    if Q.pt_7 >= 37.3:
        z += -0.0813 * Q.pt_7 + 4.15159
    if Q.LHA < 0.179 and Q.log_sum_pt < 6.83:
        z += -57.9 * (0.179 - Q.LHA) * (6.83 - Q.log_sum_pt)
    if Q.e2_sq < 0.00503 and Q.centroid_offset < 0.0171:
        z += 24300.0 * (0.00503 - Q.e2_sq) * (0.0171 - Q.centroid_offset)
    if Q.girth2 < 0.00116 and Q.centroid_offset < 0.0252:
        z += 75500.0 * (0.00116 - Q.girth2) * (0.0252 - Q.centroid_offset)
    if Q.log_sum_pt > 6.66 and Q.zdr_3 < 0.00294:
        z += 2300.0 * (Q.log_sum_pt - 6.66) * (0.00294 - Q.zdr_3)
    if Q.log_sum_pt > 6.87 and Q.zdr_3 < 0.0045:
        z += -3660.0 * (Q.log_sum_pt - 6.87) * (0.0045 - Q.zdr_3)
    if Q.max_pair_mass > 16.9 and Q.dr_max_012 < 0.249:
        z += -0.385 * (Q.max_pair_mass - 16.9) * (0.249 - Q.dr_max_012)
    if Q.z_6 < 0.0267 and Q.abseta_6 < 0.209:
        z += 584.0 * (0.0267 - Q.z_6) * (0.209 - Q.abseta_6)
    if Q.z_7 < 0.0873 and Q.centroid_offset < 0.0388:
        z += -786.0 * (0.0873 - Q.z_7) * (0.0388 - Q.centroid_offset)
    if Q.z_7 < 0.0464 and Q.mass_top5 < 84.8:
        z += 0.777 * (0.0464 - Q.z_7) * (84.8 - Q.mass_top5)
    return max(0.0, z)


def neuron_6(Q):
    z = 9.13
    if Q.C2_b2 >= 0.000142:
        z += -51.0 * Q.C2_b2 + 0.007242
    if Q.D2_b2 < 0.496:
        z += -2.03 * Q.D2_b2 + 1.00688
    if Q.centroid_offset < 0.0207:
        z += -1.42 * Q.centroid_offset + 0.029394
    if Q.centroid_offset >= 0.0435:
        z += -19.6 * Q.centroid_offset + 0.8526
    if Q.girth2 < 0.00801:
        z += 773.0 * Q.girth2 - 6.19173
    if Q.lam1 < 0.00764:
        z += -615.0 * Q.lam1 + 4.6986
    if Q.lam1 >= 0.0132:
        z += 138.0 * Q.lam1 - 1.8216
    if Q.log_sum_pt >= 6.67:
        z += 3.39 * Q.log_sum_pt - 22.6113
    z += -62.1 * Q.mass_over_sum_pt
    if Q.max_dr < 0.175:
        z += 18.6 * Q.max_dr - 3.255
    if Q.pt_5 < 24.7:
        z += -0.105 * Q.pt_5 + 2.5935
    if Q.pt_6 >= 42.8:
        z += -0.115 * Q.pt_6 + 4.922
    if Q.pt_7 >= 29.4:
        z += 0.101 * Q.pt_7 - 2.9694
    if Q.sj3_dr_max < 0.174:
        z += -12.4 * Q.sj3_dr_max + 2.1576
    if Q.sj3_dr_max >= 0.187:
        z += 6.9 * Q.sj3_dr_max - 1.2903
    if Q.tau1 < 0.099:
        z += -15.4 * Q.tau1 + 1.2768
    if 0.099 <= Q.tau1 < 0.12:
        z += 11.8 * Q.tau1 - 1.416
    if Q.width < 0.0124:
        z += 568.0 * Q.width - 7.0432
    if Q.z_6 < 0.061:
        z += 81.0 * Q.z_6 - 5.1111
    if 0.061 <= Q.z_6 < 0.0631:
        z += 166.4 * Q.z_6 - 10.3205
    if Q.z_6 >= 0.0631:
        z += 85.4 * Q.z_6 - 5.2094
    if Q.z_7 < 0.0331:
        z += -55.7 * Q.z_7 + 1.84367
    if Q.z_7 >= 0.0342:
        z += -74.1 * Q.z_7 + 2.53422
    if Q.centroid_offset > 0.00912 and Q.mean_phi2 < 0.0117:
        z += 1550.0 * (Q.centroid_offset - 0.00912) * (0.0117 - Q.mean_phi2)
    if Q.centroid_offset > 0.0118 and Q.sum_pt_top5 > 544.0:
        z += 0.159 * (Q.centroid_offset - 0.0118) * (Q.sum_pt_top5 - 544.0)
    if Q.lam1 < 0.0101 and Q.mean_eta > 0.0245:
        z += 11700.0 * (0.0101 - Q.lam1) * (Q.mean_eta - 0.0245)
    if Q.lam1 < 0.0161 and Q.planar_flow < 0.243:
        z += -290.0 * (0.0161 - Q.lam1) * (0.243 - Q.planar_flow)
    if Q.lam2 < 0.00273 and Q.n_dr_0_0p05 < 2.79:
        z += 111.0 * (0.00273 - Q.lam2) * (2.79 - Q.n_dr_0_0p05)
    if Q.pt_6 < 42.9 and Q.sum_pt < 1010.0:
        z += 0.00047 * (42.9 - Q.pt_6) * (1010.0 - Q.sum_pt)
    if Q.width < 0.0128 and Q.mean_eta < -0.0223:
        z += 9680.0 * (0.0128 - Q.width) * (-0.0223 - Q.mean_eta)
    if Q.width < 0.0103 and Q.mean_phi > 0.0265:
        z += 13600.0 * (0.0103 - Q.width) * (Q.mean_phi - 0.0265)
    if Q.width < 0.0125 and Q.mean_phi < -0.0203:
        z += 7030.0 * (0.0125 - Q.width) * (-0.0203 - Q.mean_phi)
    return max(0.0, z)


def neuron_7(Q):
    z = 12.8
    if Q.e3 >= 5.65e-05:
        z += 2220.0 * Q.e3 - 0.12543
    if Q.girth < 0.0187:
        z += 102.0 * Q.girth - 8.9658
    if 0.0187 <= Q.girth < 0.0879:
        z += 16.9 * Q.girth - 7.37443
    if Q.girth >= 0.0879:
        z += -85.1 * Q.girth + 1.59137
    if Q.girth2 < 0.0019:
        z += 1690.0 * Q.girth2 - 12.3201
    if 0.0019 <= Q.girth2 < 0.00729:
        z += 1096.0 * Q.girth2 - 11.1915
    if Q.girth2 >= 0.00729:
        z += -594.0 * Q.girth2 + 1.1286
    if Q.lam2 >= 0.000165:
        z += -321.0 * Q.lam2 + 0.052965
    if Q.mass < 27.9:
        z += 0.0306 * Q.mass - 1.34946
    if 27.9 <= Q.mass < 44.1:
        z += 0.0752 * Q.mass - 2.5938
    if 44.1 <= Q.mass < 78.4:
        z += 0.0446 * Q.mass - 1.24434
    if Q.mass >= 78.4:
        z += -0.1474 * Q.mass + 13.80846
    if Q.mass_over_sum_pt < 0.0684:
        z += -124.0 * Q.mass_over_sum_pt + 10.2052
    if 0.0684 <= Q.mass_over_sum_pt < 0.0823:
        z += 77.0 * Q.mass_over_sum_pt - 3.5432
    if Q.mass_over_sum_pt >= 0.0823:
        z += 201.0 * Q.mass_over_sum_pt - 13.7484
    if Q.mass_top5 >= 61.7:
        z += 0.0613 * Q.mass_top5 - 3.78221
    if Q.pt_6 < 50.7:
        z += 0.02 * Q.pt_6 - 1.014
    if Q.pt_7 < 29.7:
        z += 0.0586 * Q.pt_7 - 1.74042
    if Q.sd_mass >= 23.3:
        z += -0.0352 * Q.sd_mass + 0.82016
    if 0.113 <= Q.sd_rg < 0.192:
        z += 12.6 * Q.sd_rg - 1.4238
    if Q.sd_rg >= 0.192:
        z += -9.7 * Q.sd_rg + 2.8578
    if 0.155 <= Q.sj3_dr_max < 0.243:
        z += 10.7 * Q.sj3_dr_max - 1.6585
    if Q.sj3_dr_max >= 0.243:
        z += -5.8 * Q.sj3_dr_max + 2.351
    if Q.sum_pt < 643.0:
        z += -0.00302 * Q.sum_pt + 1.94186
    if Q.tau1 >= 0.0922:
        z += 70.9 * Q.tau1 - 6.53698
    if 0.00493 <= Q.width < 0.0223:
        z += -1440.0 * Q.width + 7.0992
    if Q.width >= 0.0223:
        z += -25.0128
    if Q.centroid_offset < 0.0243 and Q.C2_b2 < 0.00677:
        z += -7060.0 * (0.0243 - Q.centroid_offset) * (0.00677 - Q.C2_b2)
    if Q.centroid_offset > 0.0409 and Q.n_pt_above_50 > 3.13:
        z += -34.0 * (Q.centroid_offset - 0.0409) * (Q.n_pt_above_50 - 3.13)
    if Q.planar_flow < 0.196 and Q.sd_mass > 42.1:
        z += 0.327 * (0.196 - Q.planar_flow) * (Q.sd_mass - 42.1)
    if Q.tau1 < 0.0556 and Q.n_dr_0p05_0p1 > 4.13:
        z += -24.5 * (0.0556 - Q.tau1) * (Q.n_dr_0p05_0p1 - 4.13)
    if Q.width > 0.0103 and Q.pt_6 < 55.0:
        z += 14.9 * (Q.width - 0.0103) * (55.0 - Q.pt_6)
    return max(0.0, z)


def neuron_8(Q):
    z = 5.94
    if Q.C2 < 0.0663:
        z += -9.99 * Q.C2 + 0.662337
    if Q.C2_b2 < 0.00142:
        z += 313.0 * Q.C2_b2 - 0.44446
    if Q.centroid_offset < 0.0116:
        z += -18.2 * Q.centroid_offset + 0.21112
    if Q.centroid_offset >= 0.0287:
        z += 18.2 * Q.centroid_offset - 0.52234
    if Q.dr_3 >= 0.123:
        z += 1.45 * Q.dr_3 - 0.17835
    if Q.dr_4 < 0.0959:
        z += -1.62 * Q.dr_4 + 0.155358
    if Q.dr_4 >= 0.151:
        z += 1.52 * Q.dr_4 - 0.22952
    if Q.dr_6 < 0.105:
        z += -1.71 * Q.dr_6 + 0.17955
    if Q.dr_7 < 0.109:
        z += -1.87 * Q.dr_7 + 0.20383
    if Q.dr_max_012 < 0.167:
        z += -1.7 * Q.dr_max_012 + 0.2839
    if Q.e2_sq < 0.0168:
        z += 435.0 * Q.e2_sq - 7.308
    if Q.e3 < 3.1e-05:
        z += 8440.0 * Q.e3 - 0.26164
    if Q.girth < 0.0609:
        z += 174.0 * Q.girth - 10.5966
    if Q.girth >= 0.124:
        z += 8.07 * Q.girth - 1.00068
    if Q.girth2 < 0.00527:
        z += -573.0 * Q.girth2 + 3.01971
    if Q.girth2_top3 < 0.00696:
        z += 97.6 * Q.girth2_top3 - 0.679296
    if Q.girth2_top5 >= 0.0094:
        z += -27.3 * Q.girth2_top5 + 0.25662
    if Q.lam1 < 0.00732:
        z += 910.0 * Q.lam1 - 6.6612
    if Q.lam2 < 0.000161:
        z += -1707.0 * Q.lam2 - 0.25383
    if 0.000161 <= Q.lam2 < 0.0011:
        z += 563.0 * Q.lam2 - 0.6193
    if Q.log_sum_pt < 6.84:
        z += 3.69 * Q.log_sum_pt - 25.2396
    if Q.mass < 29.7:
        z += 0.068 * Q.mass - 2.0196
    if 42.2 <= Q.mass < 76.0:
        z += -0.0144 * Q.mass + 0.60768
    if Q.mass >= 76.0:
        z += -0.01771 * Q.mass + 0.85924
    if 0.0857 <= Q.mass_over_sum_pt < 0.132:
        z += -78.3 * Q.mass_over_sum_pt + 6.71031
    if Q.mass_over_sum_pt >= 0.132:
        z += 30.7 * Q.mass_over_sum_pt - 7.67769
    if Q.max_pair_mass >= 22.4:
        z += 0.0148 * Q.max_pair_mass - 0.33152
    if Q.mean_eta < -0.000608:
        z += 5.14 * Q.mean_eta + 0.00312512
    if Q.mean_phi < -0.0161:
        z += -1.4 * Q.mean_phi - 0.377104
    if -0.0161 <= Q.mean_phi < 0.00199:
        z += 19.6 * Q.mean_phi - 0.039004
    if Q.pt_5 < 68.8:
        z += -0.0067 * Q.pt_5 + 0.46096
    if Q.pt_7 < 20.1:
        z += -0.0284 * Q.pt_7 + 1.37172
    if 20.1 <= Q.pt_7 < 48.3:
        z += -0.0687 * Q.pt_7 + 2.18175
    if Q.pt_7 >= 48.3:
        z += -0.0403 * Q.pt_7 + 0.81003
    if 23.9 <= Q.sd_mass < 59.2:
        z += 0.00782 * Q.sd_mass - 0.186898
    if Q.sd_mass >= 59.2:
        z += 0.01385 * Q.sd_mass - 0.543874
    if 0.142 <= Q.sj2_dr < 0.185:
        z += -2.68 * Q.sj2_dr + 0.38056
    if Q.sj2_dr >= 0.185:
        z += 0.51 * Q.sj2_dr - 0.20959
    if Q.sj3_dr_max < 0.199:
        z += -9.44 * Q.sj3_dr_max + 2.29466
    if 0.199 <= Q.sj3_dr_max < 0.345:
        z += -2.85 * Q.sj3_dr_max + 0.98325
    if 838.0 <= Q.sum_pt < 953.0:
        z += -0.00649 * Q.sum_pt + 5.43862
    if Q.sum_pt >= 953.0:
        z += 0.00024 * Q.sum_pt - 0.97507
    z += -0.0031 * Q.sum_pt_top5
    if Q.width < 0.00427:
        z += -2519.0 * Q.width + 16.13213
    if 0.00427 <= Q.width < 0.00747:
        z += -1680.0 * Q.width + 12.5496
    if Q.width >= 0.0185:
        z += -60.4 * Q.width + 1.1174
    if Q.z_7 < 0.0515:
        z += 37.8 * Q.z_7 - 1.99584
    if 0.0515 <= Q.z_7 < 0.0528:
        z += 70.2 * Q.z_7 - 3.66444
    if Q.z_7 >= 0.0528:
        z += 32.4 * Q.z_7 - 1.6686
    if Q.zdr_0 < 0.0205:
        z += -11.9 * Q.zdr_0 + 0.24395
    if Q.zdr_1 < 0.0142:
        z += -18.2 * Q.zdr_1 + 0.25844
    if 0.00454 <= Q.zdr_2 < 0.0116:
        z += -25.1 * Q.zdr_2 + 0.113954
    if Q.zdr_2 >= 0.0116:
        z += 4.2 * Q.zdr_2 - 0.225926
    if Q.zdr_3 < 0.0095:
        z += -18.0 * Q.zdr_3 + 0.171
    if Q.zdr_5 < 0.00714:
        z += -26.6 * Q.zdr_5 + 0.189924
    if Q.girth < 0.0601 and Q.width > 0.000381:
        z += 32500.0 * (0.0601 - Q.girth) * (Q.width - 0.000381)
    if Q.girth2 < 0.00527 and Q.centroid_offset > 0.00673:
        z += -26400.0 * (0.00527 - Q.girth2) * (Q.centroid_offset - 0.00673)
    if Q.girth2 < 0.00531 and Q.pt_7 < 44.0:
        z += -8.23 * (0.00531 - Q.girth2) * (44.0 - Q.pt_7)
    if Q.log_sum_pt > 6.75 and Q.girth2 < 0.000544:
        z += 6610.0 * (Q.log_sum_pt - 6.75) * (0.000544 - Q.girth2)
    if Q.mass < 36.3 and Q.D2_b2 < 0.543:
        z += -0.0515 * (36.3 - Q.mass) * (0.543 - Q.D2_b2)
    if Q.mass < 21.6 and Q.lam2 < 0.000203:
        z += -424.0 * (21.6 - Q.mass) * (0.000203 - Q.lam2)
    if Q.mass_over_sum_pt < 0.0326 and Q.centroid_offset < 0.0267:
        z += 393.0 * (0.0326 - Q.mass_over_sum_pt) * (0.0267 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.132 and Q.centroid_offset > 0.0149:
        z += -994.0 * (0.132 - Q.sj3_dr_max) * (Q.centroid_offset - 0.0149)
    if Q.sj3_dr_max < 0.197 and Q.girth2 < 0.00664:
        z += -4850.0 * (0.197 - Q.sj3_dr_max) * (0.00664 - Q.girth2)
    if Q.sj3_dr_max < 0.201 and Q.lam2 < 0.000172:
        z += 37300.0 * (0.201 - Q.sj3_dr_max) * (0.000172 - Q.lam2)
    if Q.sj3_dr_max < 0.199 and Q.width < 0.00352:
        z += 6060.0 * (0.199 - Q.sj3_dr_max) * (0.00352 - Q.width)
    if Q.tau1 < 0.055 and Q.width < 0.00352:
        z += 20400.0 * (0.055 - Q.tau1) * (0.00352 - Q.width)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.838
    if Q.e2_sq < 0.00467:
        z += 1791.0 * Q.e2_sq - 10.7571
    if 0.00467 <= Q.e2_sq < 0.0146:
        z += 241.0 * Q.e2_sq - 3.5186
    if Q.girth2 < 0.00511:
        z += -1710.0 * Q.girth2 + 8.7381
    if Q.lam1 < 0.00643:
        z += 1130.0 * Q.lam1 - 7.2659
    if Q.lam2 >= 0.000321:
        z += 351.0 * Q.lam2 - 0.112671
    if Q.mass < 27.5:
        z += 0.163 * Q.mass - 4.4825
    if Q.mass_over_sum_pt >= 0.128:
        z += 34.6 * Q.mass_over_sum_pt - 4.4288
    if 0.167 <= Q.sj3_dr_max < 0.248:
        z += -24.9 * Q.sj3_dr_max + 4.1583
    if Q.sj3_dr_max >= 0.248:
        z += 3.4 * Q.sj3_dr_max - 2.8601
    if Q.width < 0.00725:
        z += -1840.0 * Q.width + 13.34
    if Q.girth < 0.0497 and Q.z_6 > 0.0359:
        z += -1370.0 * (0.0497 - Q.girth) * (Q.z_6 - 0.0359)
    if Q.mass < 58.8 and Q.centroid_offset < 0.0232:
        z += 6.32 * (58.8 - Q.mass) * (0.0232 - Q.centroid_offset)
    if Q.mass < 48.1 and Q.log_sum_pt < 6.87:
        z += 0.134 * (48.1 - Q.mass) * (6.87 - Q.log_sum_pt)
    if Q.sj3_dr_max < 0.158 and Q.pt_6 > 34.5:
        z += 0.438 * (0.158 - Q.sj3_dr_max) * (Q.pt_6 - 34.5)
    return max(0.0, z)


def neuron_10(Q):
    z = -8.0
    if Q.C2 < 0.0594:
        z += -13.2 * Q.C2 + 0.78408
    if Q.LHA < 0.181:
        z += -26.1 * Q.LHA + 11.2716
    if 0.181 <= Q.LHA < 0.316:
        z += -48.5 * Q.LHA + 15.326
    if Q.LHA >= 0.316:
        z += -75.6 * Q.LHA + 23.8896
    z += 140.0 * Q.girth
    if Q.girth2 < 0.0029:
        z += -46.0 * Q.girth2 + 2.50346
    if 0.0029 <= Q.girth2 < 0.00752:
        z += -513.0 * Q.girth2 + 3.85776
    if Q.lam1 < 0.00636:
        z += 615.0 * Q.lam1 - 3.9114
    if Q.lam2 < 0.000892:
        z += 998.0 * Q.lam2 - 1.10778
    if 0.000892 <= Q.lam2 < 0.00111:
        z += 1773.0 * Q.lam2 - 1.79908
    if 0.00111 <= Q.lam2 < 0.00383:
        z += 775.0 * Q.lam2 - 0.6913
    if Q.lam2 >= 0.00383:
        z += 185.0 * Q.lam2 + 1.5684
    if Q.mass >= 44.7:
        z += -0.0239 * Q.mass + 1.06833
    if Q.sj3_dr_min >= 0.14:
        z += 8.47 * Q.sj3_dr_min - 1.1858
    if Q.sj3_pair_mass_min >= 14.3:
        z += 0.195 * Q.sj3_pair_mass_min - 2.7885
    if Q.sum_pt >= 1060.0:
        z += 0.00368 * Q.sum_pt - 3.9008
    if Q.z_top5 >= 0.802:
        z += -6.65 * Q.z_top5 + 5.3333
    if Q.lam1 > 0.00427 and Q.D2_b2 < 0.227:
        z += -299.0 * (Q.lam1 - 0.00427) * (0.227 - Q.D2_b2)
    if Q.pt_7 < 52.7 and Q.D2 < 0.795:
        z += 0.064 * (52.7 - Q.pt_7) * (0.795 - Q.D2)
    if Q.pt_7 < 43.8 and Q.log_sum_pt < 6.59:
        z += -0.17 * (43.8 - Q.pt_7) * (6.59 - Q.log_sum_pt)
    if Q.sj3_pair_mass_min > 4.79 and Q.sj3_pairmin_over_m > 0.208:
        z += -0.286 * (Q.sj3_pair_mass_min - 4.79) * (Q.sj3_pairmin_over_m - 0.208)
    return max(0.0, z)


def neuron_11(Q):
    z = 3.23
    if Q.centroid_offset >= 0.0045:
        z += -47.6 * Q.centroid_offset + 0.2142
    if Q.e2_sq < 0.00848:
        z += 1020.0 * Q.e2_sq - 8.6496
    if Q.girth < 0.076:
        z += 95.9 * Q.girth - 7.2884
    if Q.girth2 < 0.0129:
        z += -345.0 * Q.girth2 + 4.4505
    if Q.mass < 82.5:
        z += 0.0501 * Q.mass - 4.13325
    if Q.planar_flow < 0.286:
        z += -4.51 * Q.planar_flow + 1.28986
    if Q.pt_6 < 32.2:
        z += 0.049 * Q.pt_6 - 1.5778
    if Q.sj3_dr_max < 0.128:
        z += -3.5 * Q.sj3_dr_max - 0.4199
    if 0.128 <= Q.sj3_dr_max < 0.161:
        z += 26.3 * Q.sj3_dr_max - 4.2343
    if Q.sj3_dr_max >= 0.173:
        z += -8.77 * Q.sj3_dr_max + 1.51721
    if Q.tau1 < 0.0459:
        z += -39.3 * Q.tau1 + 1.80387
    if Q.width < 0.00389:
        z += -1310.0 * Q.width + 11.8948
    if 0.00389 <= Q.width < 0.00908:
        z += -1570.0 * Q.width + 12.9062
    if Q.width >= 0.00908:
        z += -260.0 * Q.width + 1.0114
    if Q.z_7 >= 0.0164:
        z += 15.2 * Q.z_7 - 0.24928
    if Q.centroid_offset < 0.016 and Q.D2_b2 < 0.334:
        z += 398.0 * (0.016 - Q.centroid_offset) * (0.334 - Q.D2_b2)
    if Q.centroid_offset < 0.0169 and Q.sj3_pair_mass_min < 15.4:
        z += -9.28 * (0.0169 - Q.centroid_offset) * (15.4 - Q.sj3_pair_mass_min)
    if Q.centroid_offset < 0.0348 and Q.sum_pt < 845.0:
        z += -0.092 * (0.0348 - Q.centroid_offset) * (845.0 - Q.sum_pt)
    if Q.planar_flow < 0.271 and Q.e3 < 9.89e-06:
        z += -447000.0 * (0.271 - Q.planar_flow) * (9.89e-06 - Q.e3)
    return max(0.0, z)


def neuron_12(Q):
    z = -5.81
    if Q.C2 >= 0.0394:
        z += 9.37 * Q.C2 - 0.369178
    if Q.D3 < 3.18:
        z += -0.19 * Q.D3 + 0.6042
    if Q.N2 < 0.342:
        z += -5.82 * Q.N2 + 1.99044
    if 0.00201 <= Q.centroid_offset < 0.0246:
        z += 25.9 * Q.centroid_offset - 0.052059
    if Q.centroid_offset >= 0.0246:
        z += 48.4 * Q.centroid_offset - 0.605559
    if Q.dr_2 >= 0.122:
        z += 2.58 * Q.dr_2 - 0.31476
    if Q.dr_3 >= 0.0177:
        z += 2.38 * Q.dr_3 - 0.042126
    if Q.dr_4 >= 0.142:
        z += 3.68 * Q.dr_4 - 0.52256
    if Q.dr_5 >= 0.151:
        z += 6.59 * Q.dr_5 - 0.99509
    if Q.dr_6 >= 0.152:
        z += 4.67 * Q.dr_6 - 0.70984
    if Q.dr_7 >= 0.156:
        z += 4.88 * Q.dr_7 - 0.76128
    if Q.e2 < 0.025:
        z += -60.9 * Q.e2 + 1.81482
    if 0.025 <= Q.e2 < 0.0298:
        z += 0.2 * Q.e2 + 0.28732
    if Q.e2 >= 0.0298:
        z += 61.1 * Q.e2 - 1.5275
    if Q.e3 < 0.000277:
        z += -2510.0 * Q.e3 + 0.69527
    if Q.girth >= 0.125:
        z += 24.3 * Q.girth - 3.0375
    if Q.girth2_top5 < 0.00293:
        z += 117.0 * Q.girth2_top5 - 0.48321
    if 0.00293 <= Q.girth2_top5 < 0.00413:
        z += 214.7 * Q.girth2_top5 - 0.769471
    if Q.girth2_top5 >= 0.00413:
        z += 97.7 * Q.girth2_top5 - 0.286261
    if Q.lam1 >= 0.0175:
        z += -114.0 * Q.lam1 + 1.995
    if Q.log_sum_pt >= 6.67:
        z += 4.91 * Q.log_sum_pt - 32.7497
    if Q.mass < 61.2:
        z += 0.0532 * Q.mass - 3.25584
    if Q.mass >= 63.8:
        z += 0.0745 * Q.mass - 4.7531
    if Q.mass_over_sum_pt_sq >= 0.0189:
        z += 123.0 * Q.mass_over_sum_pt_sq - 2.3247
    if Q.max_dr < 0.242:
        z += 6.41 * Q.max_dr - 1.55122
    if Q.mean_eta >= 0.0279:
        z += -10.5 * Q.mean_eta + 0.29295
    if Q.mean_phi < -0.0216:
        z += 15.9 * Q.mean_phi + 0.34344
    if Q.n_dr_0p05_0p1 >= 5.7:
        z += 0.233 * Q.n_dr_0p05_0p1 - 1.3281
    if Q.planar_flow < 0.525:
        z += 1.42 * Q.planar_flow - 0.7455
    if Q.pt_4 < 53.5:
        z += -0.013 * Q.pt_4 + 0.6955
    if Q.sj3_dr_max < 0.115:
        z += -23.38 * Q.sj3_dr_max + 4.52718
    if 0.115 <= Q.sj3_dr_max < 0.311:
        z += -9.38 * Q.sj3_dr_max + 2.91718
    if Q.sj3_pair_mass_min >= 8.44:
        z += -0.0328 * Q.sj3_pair_mass_min + 0.276832
    if Q.sum_pt < 417.0:
        z += 0.0137 * Q.sum_pt - 5.7129
    if Q.tau1 >= 0.0509:
        z += -36.4 * Q.tau1 + 1.85276
    if Q.width < 0.00793:
        z += -178.0 * Q.width + 1.41154
    if Q.width >= 0.0182:
        z += -69.9 * Q.width + 1.27218
    if Q.z_7 < 0.0672:
        z += 29.2 * Q.z_7 - 1.96224
    if Q.z_dr_0_0p05 >= 0.622:
        z += -2.93 * Q.z_dr_0_0p05 + 1.82246
    if Q.zdr_0 >= 0.0443:
        z += -28.7 * Q.zdr_0 + 1.27141
    if Q.zdr_5 >= 0.00686:
        z += 26.2 * Q.zdr_5 - 0.179732
    if Q.zdr_6 >= 0.00489:
        z += 46.2 * Q.zdr_6 - 0.225918
    if Q.zdr_7 >= 0.00431:
        z += 45.1 * Q.zdr_7 - 0.194381
    return max(0.0, z)


def neuron_13(Q):
    z = -2.41
    if Q.e2 < 0.0666:
        z += 71.1 * Q.e2 - 4.73526
    if Q.e2_sq < 0.0167:
        z += -256.0 * Q.e2_sq + 4.2752
    if Q.girth < 0.146:
        z += -75.9 * Q.girth + 11.0814
    if Q.girth2 >= 0.0157:
        z += -181.0 * Q.girth2 + 2.8417
    if Q.lam2 < 0.00076:
        z += 1420.0 * Q.lam2 - 1.0792
    if Q.mass >= 75.5:
        z += -0.0458 * Q.mass + 3.4579
    if Q.mass_over_sum_pt >= 0.0802:
        z += 57.6 * Q.mass_over_sum_pt - 4.61952
    if Q.pt_6 < 24.1:
        z += 0.187 * Q.pt_6 - 4.5067
    if Q.pt_7 < 28.7:
        z += 0.16 * Q.pt_7 - 4.592
    if Q.sj3_dr23 >= 0.311:
        z += -5.4 * Q.sj3_dr23 + 1.6794
    if Q.width < 0.00751:
        z += 60.0 * Q.width + 1.74387
    if 0.00751 <= Q.width < 0.0141:
        z += -333.0 * Q.width + 4.6953
    if Q.e3 < 7.89e-05 and Q.centroid_offset < 0.0314:
        z += -607000.0 * (7.89e-05 - Q.e3) * (0.0314 - Q.centroid_offset)
    if Q.girth < 0.139 and Q.log_sum_pt < 6.84:
        z += -65.7 * (0.139 - Q.girth) * (6.84 - Q.log_sum_pt)
    if Q.girth < 0.14 and Q.pt_7 < 38.9:
        z += -1.1 * (0.14 - Q.girth) * (38.9 - Q.pt_7)
    if Q.girth < 0.141 and Q.z_7 > 0.0604:
        z += 230.0 * (0.141 - Q.girth) * (Q.z_7 - 0.0604)
    if Q.sum_pt > 1060.0 and Q.pt_6 < 56.8:
        z += -0.000296 * (Q.sum_pt - 1060.0) * (56.8 - Q.pt_6)
    if Q.sum_pt_top5 > 579.0 and Q.pt_7 < 37.7:
        z += 0.00062 * (Q.sum_pt_top5 - 579.0) * (37.7 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = 14.8
    if Q.D2 < 1.19:
        z += 0.894 * Q.D2 - 0.88774
    if Q.D2 >= 1.19:
        z += 0.148 * Q.D2
    if Q.D2_b2 < 0.114:
        z += 18.8 * Q.D2_b2 - 2.1432
    if Q.LHA >= 0.312:
        z += 60.2 * Q.LHA - 18.7824
    if Q.centroid_offset < 0.0224:
        z += 28.5 * Q.centroid_offset - 0.6384
    if Q.centroid_offset >= 0.0264:
        z += -65.6 * Q.centroid_offset + 1.73184
    if Q.e2 < 0.0172:
        z += -143.0 * Q.e2 + 8.9232
    if 0.0172 <= Q.e2 < 0.0431:
        z += -76.9 * Q.e2 + 7.78628
    if 0.0431 <= Q.e2 < 0.0618:
        z += 44.1 * Q.e2 + 2.57118
    if 0.0618 <= Q.e2 < 0.0624:
        z += -82.9 * Q.e2 + 10.41978
    if Q.e2 >= 0.0624:
        z += 60.1 * Q.e2 + 1.49658
    if Q.e2_sq < 0.00429:
        z += -509.0 * Q.e2_sq + 2.18361
    if Q.girth < 0.0808:
        z += 55.0 * Q.girth - 4.444
    if Q.girth >= 0.0886:
        z += -188.0 * Q.girth + 16.6568
    if Q.girth2 < 0.000857:
        z += 2692.0 * Q.girth2 - 17.71748
    if 0.000857 <= Q.girth2 < 0.00449:
        z += 592.0 * Q.girth2 - 15.91778
    if 0.00449 <= Q.girth2 < 0.00725:
        z += -60.0 * Q.girth2 - 12.9903
    if 0.00725 <= Q.girth2 < 0.0092:
        z += -2100.0 * Q.girth2 + 1.7997
    if Q.girth2 >= 0.0092:
        z += -1233.0 * Q.girth2 - 6.1767
    if Q.lam1 >= 0.00715:
        z += 399.0 * Q.lam1 - 2.85285
    if Q.log_sum_pt >= 6.2:
        z += 5.22 * Q.log_sum_pt - 32.364
    if Q.mass < 59.8:
        z += 0.084 * Q.mass - 5.0232
    if 0.0828 <= Q.mass_over_sum_pt < 0.0922:
        z += 200.0 * Q.mass_over_sum_pt - 16.56
    if Q.mass_over_sum_pt >= 0.0922:
        z += 28.0 * Q.mass_over_sum_pt - 0.7016
    if Q.mass_top5 >= 56.7:
        z += 0.0487 * Q.mass_top5 - 2.76129
    if Q.mean_phi < 0.00646:
        z += -7.52 * Q.mean_phi + 0.0485792
    if Q.n_dr_0_0p05 < 4.89:
        z += -0.0964 * Q.n_dr_0_0p05 + 0.471396
    if Q.planar_flow < 0.12:
        z += -9.73 * Q.planar_flow + 1.1676
    if Q.pt_5 >= 37.3:
        z += -0.0125 * Q.pt_5 + 0.46625
    if Q.pt_7 < 33.8:
        z += -0.0198 * Q.pt_7 + 0.68904
    if 33.8 <= Q.pt_7 < 34.8:
        z += -0.0574 * Q.pt_7 + 1.95992
    if Q.pt_7 >= 34.8:
        z += -0.0376 * Q.pt_7 + 1.27088
    if Q.sd_mass < 4.58:
        z += 0.0014 * Q.sd_mass - 1.3226
    if 4.58 <= Q.sd_mass < 59.0:
        z += -0.0308 * Q.sd_mass - 1.175124
    if 59.0 <= Q.sd_mass < 74.5:
        z += 0.0478 * Q.sd_mass - 5.812524
    if Q.sd_mass >= 74.5:
        z += -0.0322 * Q.sd_mass + 0.147476
    if Q.sd_rg < 0.188:
        z += 7.32 * Q.sd_rg - 1.37616
    if 0.0391 <= Q.sj2_dr < 0.132:
        z += -25.4 * Q.sj2_dr + 0.99314
    if 0.132 <= Q.sj2_dr < 0.209:
        z += 11.8 * Q.sj2_dr - 3.91726
    if Q.sj2_dr >= 0.209:
        z += -1.4 * Q.sj2_dr - 1.15846
    if Q.sj3_dr_max < 0.0406:
        z += -20.3 * Q.sj3_dr_max - 1.54054
    if 0.0406 <= Q.sj3_dr_max < 0.241:
        z += 11.8 * Q.sj3_dr_max - 2.8438
    if Q.sum_pt_top5 >= 368.0:
        z += -0.0089 * Q.sum_pt_top5 + 3.2752
    if Q.tau1 >= 0.0531:
        z += 30.5 * Q.tau1 - 1.61955
    if 0.0126 <= Q.width < 0.0186:
        z += 455.0 * Q.width - 5.733
    if Q.width >= 0.0186:
        z += 827.0 * Q.width - 12.6522
    if Q.z_7 >= 0.0231:
        z += 15.3 * Q.z_7 - 0.35343
    if Q.centroid_offset > 0.0554 and Q.C2_b2 < 0.00154:
        z += -114000.0 * (Q.centroid_offset - 0.0554) * (0.00154 - Q.C2_b2)
    if Q.centroid_offset > 0.0258 and Q.pt_1 < 161.0:
        z += 0.466 * (Q.centroid_offset - 0.0258) * (161.0 - Q.pt_1)
    if Q.planar_flow < 0.115 and Q.lam1 < 0.00711:
        z += -2630.0 * (0.115 - Q.planar_flow) * (0.00711 - Q.lam1)
    if Q.planar_flow < 0.121 and Q.sum_pt_top5 < 665.0:
        z += -0.0172 * (0.121 - Q.planar_flow) * (665.0 - Q.sum_pt_top5)
    if Q.sj2_dr > 0.128 and Q.D2_b2 < 0.109:
        z += 362.0 * (Q.sj2_dr - 0.128) * (0.109 - Q.D2_b2)
    if Q.sj2_dr > 0.191 and Q.D2_b2 < 0.106:
        z += -395.0 * (Q.sj2_dr - 0.191) * (0.106 - Q.D2_b2)
    if Q.z_dr_0p05_0p1 > 0.745 and Q.C2_b2 < 0.0244:
        z += -450.0 * (Q.z_dr_0p05_0p1 - 0.745) * (0.0244 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 > 0.754 and Q.n_dr_0p2_0p4 < 1.03:
        z += 10.3 * (Q.z_dr_0p05_0p1 - 0.754) * (1.03 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p05_0p1 < 0.674 and Q.sum_pt < 687.0:
        z += 0.0039 * (0.674 - Q.z_dr_0p05_0p1) * (687.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_15(Q):
    z = -2.96
    if Q.D2 >= 1.64:
        z += 0.128 * Q.D2 - 0.20992
    if Q.LHA >= 0.317:
        z += 53.0 * Q.LHA - 16.801
    if Q.centroid_offset >= 0.0459:
        z += -55.1 * Q.centroid_offset + 2.52909
    if Q.e2_sq < 0.00847:
        z += 214.0 * Q.e2_sq - 1.81258
    if Q.girth < 0.024:
        z += 100.9 * Q.girth - 6.17745
    if 0.024 <= Q.girth < 0.0842:
        z += 51.1 * Q.girth - 4.98225
    if 0.0842 <= Q.girth < 0.0975:
        z += -91.9 * Q.girth + 7.05835
    if Q.girth >= 0.0975:
        z += -143.0 * Q.girth + 12.0406
    if Q.girth2_top2 < 0.00768:
        z += -98.7 * Q.girth2_top2 + 0.758016
    if 0.00425 <= Q.lam1 < 0.00647:
        z += 609.0 * Q.lam1 - 2.58825
    if Q.lam1 >= 0.00647:
        z += 1053.0 * Q.lam1 - 5.46093
    if Q.lam2 < 0.00114:
        z += 220.0 * Q.lam2 + 0.75126
    if Q.lam2 >= 0.00114:
        z += 879.0 * Q.lam2
    if Q.log_sum_pt < 6.82:
        z += -1.74 * Q.log_sum_pt + 11.919
    if 6.82 <= Q.log_sum_pt < 6.85:
        z += 8.56 * Q.log_sum_pt - 58.327
    if Q.log_sum_pt >= 6.85:
        z += 10.3 * Q.log_sum_pt - 70.246
    if Q.mass < 68.3:
        z += 0.0212 * Q.mass - 1.44796
    if Q.mass >= 78.5:
        z += -0.0727 * Q.mass + 5.70695
    if Q.mass_over_sum_pt < 0.0794:
        z += -55.3 * Q.mass_over_sum_pt + 4.39082
    if Q.mass_top5 >= 63.4:
        z += 0.0368 * Q.mass_top5 - 2.33312
    if Q.mean_phi < 0.007:
        z += -4.62 * Q.mean_phi + 0.03234
    if Q.sj2_dr < 0.165:
        z += -4.5 * Q.sj2_dr + 0.2589
    if 0.165 <= Q.sj2_dr < 0.191:
        z += 18.6 * Q.sj2_dr - 3.5526
    if 0.171 <= Q.sj3_dr_max < 0.247:
        z += 8.51 * Q.sj3_dr_max - 1.45521
    if Q.sj3_dr_max >= 0.247:
        z += -1.59 * Q.sj3_dr_max + 1.03949
    if Q.sum_pt_top5 < 691.0:
        z += 0.00288 * Q.sum_pt_top5 - 1.99008
    if Q.sum_pt_top5 >= 810.0:
        z += -0.00556 * Q.sum_pt_top5 + 4.5036
    if Q.tau1 < 0.061:
        z += -20.4 * Q.tau1 + 1.2444
    if Q.tau1 >= 0.101:
        z += 40.2 * Q.tau1 - 4.0602
    if Q.width < 0.00433:
        z += 377.0 * Q.width + 2.15579
    if 0.00433 <= Q.width < 0.00687:
        z += -260.0 * Q.width + 4.914
    if 0.00687 <= Q.width < 0.0189:
        z += -1330.0 * Q.width + 12.2649
    if Q.width >= 0.0189:
        z += -1070.0 * Q.width + 7.3509
    if Q.z_dr_0p05_0p1 >= 0.723:
        z += -6.81 * Q.z_dr_0p05_0p1 + 4.92363
    if Q.zdr_0 >= 0.0427:
        z += 31.5 * Q.zdr_0 - 1.34505
    if Q.N2 < 0.197 and Q.n_for_90pct < 5.77:
        z += -2.77 * (0.197 - Q.N2) * (5.77 - Q.n_for_90pct)
    if Q.N2 < 0.23 and Q.sum_pt_top5 > 420.0:
        z += 0.0191 * (0.23 - Q.N2) * (Q.sum_pt_top5 - 420.0)
    if Q.N2 < 0.202 and Q.z_dr_0p05_0p1 < 0.686:
        z += -5.6 * (0.202 - Q.N2) * (0.686 - Q.z_dr_0p05_0p1)
    if Q.centroid_offset < 0.0207 and Q.dr01 < 0.263:
        z += -129.0 * (0.0207 - Q.centroid_offset) * (0.263 - Q.dr01)
    if Q.girth2 < 0.00772 and Q.D2 < 0.782:
        z += -3070.0 * (0.00772 - Q.girth2) * (0.782 - Q.D2)
    if Q.lam2 < 0.000286 and Q.D2_b2 < 0.0995:
        z += 18500.0 * (0.000286 - Q.lam2) * (0.0995 - Q.D2_b2)
    if Q.mass_over_sum_pt < 0.133 and Q.tau21_b2 < 0.0138:
        z += -4390.0 * (0.133 - Q.mass_over_sum_pt) * (0.0138 - Q.tau21_b2)
    if Q.mass_over_sum_pt_sq < 0.00711 and Q.D2 < 0.769:
        z += 2410.0 * (0.00711 - Q.mass_over_sum_pt_sq) * (0.769 - Q.D2)
    if Q.tau1 < 0.0239 and Q.zdr_3 > 0.00443:
        z += -20000.0 * (0.0239 - Q.tau1) * (Q.zdr_3 - 0.00443)
    if Q.width < 0.0139 and Q.tau21_b2 < 0.014:
        z += 32600.0 * (0.0139 - Q.width) * (0.014 - Q.tau21_b2)
    if Q.z_dr_0p05_0p1 > 0.706 and Q.n_dr_0p2_0p4 < 1.99:
        z += 2.99 * (Q.z_dr_0p05_0p1 - 0.706) * (1.99 - Q.n_dr_0p2_0p4)
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
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:8] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:8] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:8] + [0.0] * 0
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
