"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no W/Z/H/t mass values offered as thresholds), as if-statements, with each class score (logit) written out as a formula.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

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


def neuron_0(Q):
    z = 1.51
    if Q.e2_sq < 0.00619:
        z += 629.0 * Q.e2_sq - 3.89351
    if Q.girth2_top20 < 0.0061:
        z += 183.0 * Q.girth2_top20 - 1.1163
    if 78.3 <= Q.mass < 87.4:
        z += -0.162 * Q.mass + 12.6846
    if Q.mass >= 87.4:
        z += -0.344 * Q.mass + 28.5914
    if Q.mass_over_sum_pt_sq < 0.00777:
        z += -659.0 * Q.mass_over_sum_pt_sq + 5.12043
    if Q.psi_0p3 >= 0.996:
        z += 182.0 * Q.psi_0p3 - 181.272
    if Q.sum_pt < 1010.0:
        z += 0.0101 * Q.sum_pt - 10.201
    if Q.tau1 < 0.0693:
        z += 18.3 * Q.tau1 - 1.26819
    if Q.z_dr_0p2_0p4 < 0.0912:
        z += 5.38 * Q.z_dr_0p2_0p4 - 0.490656
    if Q.z_top50_slots < 0.992:
        z += 26.1 * Q.z_top50_slots - 25.8912
    return max(0.0, z)


def neuron_1(Q):
    z = 3.56
    if Q.M3 < 0.0323:
        z += 35.9 * Q.M3 - 1.15957
    if Q.girth2_top15 < 0.000745:
        z += -1190.0 * Q.girth2_top15 + 0.88655
    if Q.log_sum_pt < 6.9:
        z += 8.51 * Q.log_sum_pt - 60.7614
    if 6.9 <= Q.log_sum_pt < 6.99:
        z += 46.41 * Q.log_sum_pt - 322.2714
    if 6.99 <= Q.log_sum_pt < 7.14:
        z += 23.61 * Q.log_sum_pt - 162.8994
    if Q.log_sum_pt >= 7.14:
        z += 15.1 * Q.log_sum_pt - 102.138
    if Q.n_particles >= 37.8:
        z += 0.126 * Q.n_particles - 4.7628
    if Q.sj3_mass1 < 33.9:
        z += 0.0349 * Q.sj3_mass1 - 1.18311
    if Q.sum_pt < 1010.0:
        z += -0.0158 * Q.sum_pt + 15.958
    if Q.sum_pt_top2 < 701.0:
        z += -0.00311 * Q.sum_pt_top2 + 2.18011
    if Q.sum_pt_top30 >= 1180.0:
        z += 0.00656 * Q.sum_pt_top30 - 7.7408
    if Q.sum_pt_top50 >= 959.0:
        z += -0.0143 * Q.sum_pt_top50 + 13.7137
    if Q.zdr_0 >= 0.00156:
        z += -53.2 * Q.zdr_0 + 0.082992
    if Q.mass_top20 < 56.7 and Q.n_real_top40 > 29.2:
        z += 0.0035 * (56.7 - Q.mass_top20) * (Q.n_real_top40 - 29.2)
    if Q.n_particles > 38.3 and Q.soft1_pt < 2.31:
        z += -0.0315 * (Q.n_particles - 38.3) * (2.31 - Q.soft1_pt)
    if Q.sj3_mass1 < 33.4 and Q.sj3_mass2 < 18.4:
        z += -0.00257 * (33.4 - Q.sj3_mass1) * (18.4 - Q.sj3_mass2)
    if Q.sum_pt_top2 < 703.0 and Q.tau43 < 0.965:
        z += -0.00754 * (703.0 - Q.sum_pt_top2) * (0.965 - Q.tau43)
    if Q.z_top30_slots > 0.944 and Q.max_pair_mass > 7.47:
        z += 0.815 * (Q.z_top30_slots - 0.944) * (Q.max_pair_mass - 7.47)
    return max(0.0, z)


def neuron_2(Q):
    z = 1.22
    if Q.sum_pt < 1270.0:
        z += 0.00446 * Q.sum_pt - 5.6642
    return max(0.0, z)


def neuron_3(Q):
    z = 0.0826
    if Q.n_dr_0p2_0p4 < 4.64:
        z += -0.127 * Q.n_dr_0p2_0p4 + 0.58928
    if Q.tau4 < 0.0155:
        z += -161.0 * Q.tau4 + 2.4955
    if Q.tau21 < 0.317 and Q.lam1 > 0.00429:
        z += -1440.0 * (0.317 - Q.tau21) * (Q.lam1 - 0.00429)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.28
    if Q.girth2_top15 < 0.00523:
        z += 173.0 * Q.girth2_top15 - 0.90479
    if Q.mass_top40 < 69.8:
        z += -0.047 * Q.mass_top40 + 3.2806
    if Q.n_particles >= 22.5:
        z += -0.0344 * Q.n_particles + 0.774
    if Q.planar_flow < 0.324:
        z += -1.91 * Q.planar_flow + 0.61884
    if Q.sj3_pair_mass_min >= 33.4:
        z += 0.0238 * Q.sj3_pair_mass_min - 0.79492
    if Q.mass < 80.7 and Q.lam2 < 0.0041:
        z += -36.4 * (80.7 - Q.mass) * (0.0041 - Q.lam2)
    if Q.mass < 104.0 and Q.lam2 < 0.00334:
        z += 12.0 * (104.0 - Q.mass) * (0.00334 - Q.lam2)
    if Q.mass_top40 < 91.8 and Q.D2 < 9.1:
        z += -0.00338 * (91.8 - Q.mass_top40) * (9.1 - Q.D2)
    if Q.n_particles > 22.0 and Q.soft1_pt < 2.22:
        z += 0.00834 * (Q.n_particles - 22.0) * (2.22 - Q.soft1_pt)
    if Q.sj2_dr > 0.225 and Q.C2_b2 < 0.0309:
        z += -289.0 * (Q.sj2_dr - 0.225) * (0.0309 - Q.C2_b2)
    return max(0.0, z)


def neuron_5(Q):
    z = 2.33
    if Q.girth2_top30 >= 0.00764:
        z += -129.0 * Q.girth2_top30 + 0.98556
    if Q.log_sum_pt >= 6.92:
        z += -37.5 * Q.log_sum_pt + 259.5
    if Q.mass < 70.3:
        z += -0.0258 * Q.mass + 1.81374
    if Q.mass_top40 < 159.0:
        z += 0.0214 * Q.mass_top40 - 3.4026
    if Q.mass_top50 >= 157.0:
        z += -0.402 * Q.mass_top50 + 63.114
    if 0.267 <= Q.max_dr < 0.435:
        z += -4.11 * Q.max_dr + 1.09737
    if Q.max_dr >= 0.435:
        z += 1.33 * Q.max_dr - 1.26903
    if Q.n_particles < 54.1:
        z += -0.0338 * Q.n_particles + 1.82858
    if Q.sum_pt_top50 >= 917.0:
        z += 0.0253 * Q.sum_pt_top50 - 23.2001
    if Q.tau21 < 0.469:
        z += 3.37 * Q.tau21 - 1.58053
    if Q.z_top30_slots >= 0.895:
        z += -9.3 * Q.z_top30_slots + 8.3235
    return max(0.0, z)


def neuron_6(Q):
    z = -0.458
    if Q.D2 < 2.23:
        z += 0.35 * Q.D2 - 0.7805
    if Q.mass < 83.8:
        z += -0.0359 * Q.mass + 3.36742
    if 83.8 <= Q.mass < 93.8:
        z += 0.0539 * Q.mass - 4.15782
    if 93.8 <= Q.mass < 106.0:
        z += 0.0898 * Q.mass - 7.52524
    if Q.mass >= 106.0:
        z += -0.0073 * Q.mass + 2.76736
    return max(0.0, z)


def neuron_7(Q):
    z = 0.175
    if Q.psi_0p3 >= 0.998:
        z += -600.0 * Q.psi_0p3 + 598.8
    if Q.tau21 < 0.379:
        z += 4.04 * Q.tau21 - 1.53116
    if Q.tau21_b2 < 0.249:
        z += -6.49 * Q.tau21_b2 + 1.61601
    if Q.z_top20_slots >= 0.918:
        z += -9.28 * Q.z_top20_slots + 8.51904
    if Q.mass < 82.0 and Q.psi_0p3 < 1.0:
        z += -8.19 * (82.0 - Q.mass) * (1.0 - Q.psi_0p3)
    if Q.mass < 82.8 and Q.psi_0p3 > 0.995:
        z += -31.4 * (82.8 - Q.mass) * (Q.psi_0p3 - 0.995)
    if Q.mass < 90.9 and Q.psi_0p3 > 0.997:
        z += -78.5 * (90.9 - Q.mass) * (Q.psi_0p3 - 0.997)
    if Q.mass < 92.0 and Q.psi_0p3 > 0.965:
        z += -11.7 * (92.0 - Q.mass) * (Q.psi_0p3 - 0.965)
    if Q.mass < 101.0 and Q.psi_0p3 > 0.997:
        z += 89.9 * (101.0 - Q.mass) * (Q.psi_0p3 - 0.997)
    if Q.mass < 104.0 and Q.psi_0p3 > 0.966:
        z += 3.72 * (104.0 - Q.mass) * (Q.psi_0p3 - 0.966)
    if Q.mass < 121.0 and Q.sd_mass > 75.4:
        z += 0.00331 * (121.0 - Q.mass) * (Q.sd_mass - 75.4)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.08
    if Q.log_sum_pt >= 6.96:
        z += 9.91 * Q.log_sum_pt - 68.9736
    if Q.mass >= 106.0:
        z += -0.0823 * Q.mass + 8.7238
    if Q.mass_over_sum_pt >= 0.0766:
        z += 83.4 * Q.mass_over_sum_pt - 6.38844
    if Q.n_dr_0p2_0p4 < 16.9:
        z += 0.12 * Q.n_dr_0p2_0p4 - 2.028
    if Q.sum_pt_top40 >= 887.0:
        z += -0.00768 * Q.sum_pt_top40 + 6.81216
    if Q.log_sum_pt > 6.9 and Q.sj3_pair_mass_max > 37.0:
        z += 0.0756 * (Q.log_sum_pt - 6.9) * (Q.sj3_pair_mass_max - 37.0)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.174
    if Q.e3 >= 0.000341:
        z += -5600.0 * Q.e3 + 1.9096
    if Q.mass < 64.1:
        z += -0.0607 * Q.mass + 6.37247
    if 64.1 <= Q.mass < 81.7:
        z += -0.141 * Q.mass + 11.5197
    if 110.0 <= Q.mass < 168.0:
        z += 0.0167 * Q.mass - 1.837
    if Q.mass >= 168.0:
        z += -0.0554 * Q.mass + 10.2758
    if Q.sum_pt_top50 < 960.0:
        z += -0.0121 * Q.sum_pt_top50 + 11.616
    if Q.z_top40_slots >= 0.961:
        z += -21.0 * Q.z_top40_slots + 20.181
    if Q.mass_top40 < 95.5 and Q.sum_pt < 1050.0:
        z += -0.00016 * (95.5 - Q.mass_top40) * (1050.0 - Q.sum_pt)
    if Q.sum_pt_top40 < 1010.0 and Q.soft5_pt > 2.98:
        z += 0.00423 * (1010.0 - Q.sum_pt_top40) * (Q.soft5_pt - 2.98)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.4
    if Q.D2 < 3.84:
        z += -0.512 * Q.D2 + 1.96608
    if Q.mass < 80.8:
        z += -0.0569 * Q.mass + 4.59752
    if 149.0 <= Q.mass < 172.0:
        z += -0.0624 * Q.mass + 9.2976
    if Q.mass >= 172.0:
        z += -0.3484 * Q.mass + 58.4896
    if Q.mass_top50 >= 156.0:
        z += 0.132 * Q.mass_top50 - 20.592
    if Q.sj2_mass1 < 82.7:
        z += 0.0511 * Q.sj2_mass1 - 4.22597
    if Q.sum_pt < 980.0:
        z += 0.00483 * Q.sum_pt - 4.7334
    if Q.sum_pt_top5 < 880.0:
        z += -0.00292 * Q.sum_pt_top5 + 2.5696
    if Q.tau21_b2 < 0.604:
        z += -2.51 * Q.tau21_b2 + 1.51604
    if Q.z_dr_0_0p05 >= 0.768:
        z += -9.99 * Q.z_dr_0_0p05 + 7.67232
    if Q.D2 < 3.83 and Q.sj2_dr > 0.196:
        z += -1.81 * (3.83 - Q.D2) * (Q.sj2_dr - 0.196)
    if Q.mass > 173.0 and Q.M2 > 0.0406:
        z += 2.67 * (Q.mass - 173.0) * (Q.M2 - 0.0406)
    if Q.sj2_mass1 < 81.5 and Q.sj2_mass2 > 11.9:
        z += 0.000955 * (81.5 - Q.sj2_mass1) * (Q.sj2_mass2 - 11.9)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.0886
    if Q.mass < 79.4:
        z += 0.0154 * Q.mass - 1.22276
    if Q.n_dr_0p1_0p2 < 15.7:
        z += -0.0504 * Q.n_dr_0p1_0p2 + 0.79128
    if Q.n_dr_0p2_0p4 < 10.3:
        z += -0.0989 * Q.n_dr_0p2_0p4 + 1.01867
    if Q.psi_0p2 >= 0.992:
        z += 119.0 * Q.psi_0p2 - 118.048
    if Q.mass < 78.5 and Q.D2 < 3.87:
        z += -0.0708 * (78.5 - Q.mass) * (3.87 - Q.D2)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.127
    if Q.e2_sq < 0.00289:
        z += 870.0 * Q.e2_sq - 2.5143
    if Q.mass < 82.6:
        z += -0.0819 * Q.mass + 6.76494
    if Q.mass < 83.6 and Q.psi_0p3 < 0.999:
        z += -6.07 * (83.6 - Q.mass) * (0.999 - Q.psi_0p3)
    if Q.sj3_pair_mass_max > 126.0 and Q.sj2_mass1 < 52.1:
        z += 0.00173 * (Q.sj3_pair_mass_max - 126.0) * (52.1 - Q.sj2_mass1)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.34
    if Q.log_sum_pt < 6.82:
        z += -14.1 * Q.log_sum_pt + 96.162
    if 143.0 <= Q.mass < 174.0:
        z += -0.203 * Q.mass + 29.029
    if Q.mass >= 174.0:
        z += -0.1256 * Q.mass + 15.5614
    if Q.mass_top50 >= 141.0:
        z += 0.146 * Q.mass_top50 - 20.586
    if Q.sum_pt < 1010.0:
        z += 0.0329 * Q.sum_pt - 33.958
    if 1010.0 <= Q.sum_pt < 1100.0:
        z += 0.0081 * Q.sum_pt - 8.91
    if Q.sum_pt_top40 < 1020.0 and Q.D3 < 0.65:
        z += 0.0185 * (1020.0 - Q.sum_pt_top40) * (0.65 - Q.D3)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.652
    if Q.mass < 83.6:
        z += 0.144 * Q.mass - 12.0384
    if Q.tau21_b2 < 0.303:
        z += -3.67 * Q.tau21_b2 + 1.11201
    if Q.N2 < 0.42 and Q.max_dr > 0.25:
        z += -20.7 * (0.42 - Q.N2) * (Q.max_dr - 0.25)
    if Q.tau21_b2 < 0.29 and Q.girth2_top40 < 0.00776:
        z += -2550.0 * (0.29 - Q.tau21_b2) * (0.00776 - Q.girth2_top40)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.444
    if Q.sd_mass < 33.4:
        z += -0.0005 * Q.sd_mass + 1.67251
    if 33.4 <= Q.sd_mass < 94.5:
        z += -0.0271 * Q.sd_mass + 2.56095
    if Q.tau1 < 0.0618:
        z += 16.7 * Q.tau1 - 1.03206
    if Q.sj2_dr > 0.202 and Q.C2_b2 < 0.0389:
        z += 257.0 * (Q.sj2_dr - 0.202) * (0.0389 - Q.C2_b2)
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


def logit_g(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return -1.078125 + 0.625 * h1 - 0.75 * h3 - 0.875 * h4 - 0.3125 * h5 + 0.15625 * h6 + 0.03125 * h8 + 0.515625 * h9 - 0.015625 * h10 + 0.234375 * h12


def logit_q(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 1.359375 - 0.1875 * h1 + 0.125 * h2 - 1.0625 * h4 + 0.21875 * h6 + 0.015625 * h8 + 0.5625 * h9 - 0.015625 * h10 - 0.25 * h11 + 0.34375 * h12


def logit_W(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 0.09375 + 0.75 * h0 + 0.4375 * h3 + 0.34375 * h4 + 0.578125 * h5 + 0.0625 * h6 - 0.625 * h7 - 0.875 * h8 - 0.21875 * h9 + 0.59375 * h11 - 0.40625 * h12 - 1.375 * h14 - 0.1875 * h15


def logit_Z(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 0.984375 - 1.375 * h0 - 0.375 * h2 + 0.4375 * h3 + 0.6875 * h5 - 0.96875 * h6 + 0.90625 * h7 - 0.9375 * h8 + 0.3125 * h12 + 0.5625 * h15


def logit_t(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 0.78125 + 0.125 * h0 + 0.1875 * h4 - 0.46875 * h5 - 0.28125 * h7 + 0.2109375 * h8 + 0.984375 * h10 - 0.375 * h12 - 0.90625 * h13 - 0.375 * h15


def logits(h):
    h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15 = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    return [logit_g(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15), logit_q(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15), logit_W(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15), logit_Z(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15), logit_t(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15)]


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
