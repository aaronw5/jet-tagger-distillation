"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; all observables), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.9% (the network: 81.1%); same class as the network for 93.0% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
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
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        e3=ecf('e3'),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_4=z[4] * dr[4],
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
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
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -12.2
    if Q.D2 < 2.36:
        z += 0.135 * Q.D2 - 0.3186
    if Q.e2 < 0.0184:
        z += -38.8 * Q.e2 + 0.71392
    if Q.e2 >= 0.0511:
        z += 8.71 * Q.e2 - 0.445081
    if Q.e2_sq < 0.00607:
        z += 298.0 * Q.e2_sq - 1.80886
    if Q.girth >= 0.0792:
        z += -3.69 * Q.girth + 0.292248
    if Q.girth2_top20 < 0.000725:
        z += 152.0 * Q.girth2_top20 - 0.9348
    if 0.000725 <= Q.girth2_top20 < 0.00615:
        z += 95.6 * Q.girth2_top20 - 0.89391
    if Q.girth2_top20 >= 0.00615:
        z += -56.4 * Q.girth2_top20 + 0.04089
    if Q.lam1 < 0.00829:
        z += 133.0 * Q.lam1 - 1.10257
    if Q.log_sum_pt < 7.03:
        z += -6.48 * Q.log_sum_pt + 45.5544
    if Q.mass < 79.0:
        z += 0.0254 * Q.mass - 2.05486
    if 79.0 <= Q.mass < 80.9:
        z += -0.0896 * Q.mass + 7.03014
    if 80.9 <= Q.mass < 89.8:
        z += -0.115 * Q.mass + 9.085
    if Q.mass >= 89.8:
        z += 0.051 * Q.mass - 5.8218
    if Q.mass_over_sum_pt >= 0.109:
        z += 39.2 * Q.mass_over_sum_pt - 4.2728
    if Q.mass_over_sum_pt_sq < 0.00932:
        z += -535.0 * Q.mass_over_sum_pt_sq + 4.9862
    if Q.mass_over_sum_pt_sq >= 0.0104:
        z += -24.4 * Q.mass_over_sum_pt_sq + 0.25376
    if Q.mass_top50 >= 82.4:
        z += -0.052 * Q.mass_top50 + 4.2848
    if Q.n_dr_0p2_0p4 < 14.8:
        z += -0.0482 * Q.n_dr_0p2_0p4 + 0.71336
    if Q.psi_0p2 >= 0.926:
        z += -5.09 * Q.psi_0p2 + 4.71334
    if Q.psi_0p3 >= 0.988:
        z += 42.0 * Q.psi_0p3 - 41.496
    if Q.pt_entropy < 2.6:
        z += -0.234 * Q.pt_entropy + 0.6084
    if Q.sum_pt < 1010.0:
        z += 0.0149 * Q.sum_pt - 15.049
    if Q.sum_pt_top40 < 878.0:
        z += -0.00683 * Q.sum_pt_top40 + 5.99674
    if Q.tau1 < 0.0734:
        z += 16.4 * Q.tau1 - 1.20376
    z += 12.5 * Q.z_top50_slots
    return max(0.0, z)


def neuron_1(Q):
    z = -0.273
    if Q.D2 < 1.57:
        z += -0.818 * Q.D2 + 1.28426
    if Q.dr_0 < 0.117:
        z += 7.9 * Q.dr_0 - 0.9243
    if Q.girth2_top10 < 0.000756:
        z += -1040.0 * Q.girth2_top10 + 0.78624
    if Q.lam2 < 0.00108:
        z += 574.0 * Q.lam2 - 0.61992
    if 6.91 <= Q.log_sum_pt < 6.97:
        z += 39.0 * Q.log_sum_pt - 269.49
    if 6.97 <= Q.log_sum_pt < 7.07:
        z += 22.9 * Q.log_sum_pt - 157.273
    if Q.log_sum_pt >= 7.07:
        z += -5.0 * Q.log_sum_pt + 39.98
    if Q.mass < 119.0:
        z += 0.0252 * Q.mass - 2.9988
    if Q.mass_over_sum_pt < 0.0734:
        z += -47.3 * Q.mass_over_sum_pt + 3.47182
    if Q.n_dr_0_0p05 < 14.8:
        z += -0.0724 * Q.n_dr_0_0p05 + 1.07152
    if Q.n_dr_0_0p05 >= 24.1:
        z += -0.0752 * Q.n_dr_0_0p05 + 1.81232
    if Q.n_dr_0p05_0p1 < 15.2:
        z += -0.0554 * Q.n_dr_0p05_0p1 + 0.92518
    if 15.2 <= Q.n_dr_0p05_0p1 < 16.7:
        z += -0.1103 * Q.n_dr_0p05_0p1 + 1.75966
    if Q.n_dr_0p05_0p1 >= 16.7:
        z += -0.0549 * Q.n_dr_0p05_0p1 + 0.83448
    if Q.n_dr_0p1_0p2 >= 6.88:
        z += -0.0651 * Q.n_dr_0p1_0p2 + 0.447888
    if Q.n_dr_0p2_0p4 >= 10.4:
        z += -0.0622 * Q.n_dr_0p2_0p4 + 0.64688
    if Q.n_particles >= 23.4:
        z += 0.0532 * Q.n_particles - 1.24488
    if Q.psi_0p3 >= 0.998:
        z += -192.0 * Q.psi_0p3 + 191.616
    if 1.88 <= Q.pt_entropy < 3.08:
        z += 1.69 * Q.pt_entropy - 3.1772
    if Q.pt_entropy >= 3.08:
        z += 3.33 * Q.pt_entropy - 8.2284
    if Q.sum_pt < 957.0:
        z += -0.0379 * Q.sum_pt + 39.3375
    if 957.0 <= Q.sum_pt < 1170.0:
        z += -0.0144 * Q.sum_pt + 16.848
    if Q.sum_pt_top15 < 893.0:
        z += 0.00873 * Q.sum_pt_top15 - 7.81335
    if 893.0 <= Q.sum_pt_top15 < 895.0:
        z += 0.01623 * Q.sum_pt_top15 - 14.51085
    if Q.sum_pt_top15 >= 895.0:
        z += 0.0075 * Q.sum_pt_top15 - 6.6975
    if Q.sum_pt_top50 < 936.0:
        z += 0.0252 * Q.sum_pt_top50 - 23.5872
    if Q.tau1 >= 0.0284:
        z += -9.0 * Q.tau1 + 0.2556
    if Q.z_dr_0_0p05 >= 0.864:
        z += -7.06 * Q.z_dr_0_0p05 + 6.09984
    if Q.z_top50_slots < 0.956:
        z += -35.2 * Q.z_top50_slots + 34.0384
    if 0.956 <= Q.z_top50_slots < 0.967:
        z += -89.4 * Q.z_top50_slots + 85.8536
    if Q.z_top50_slots >= 0.967:
        z += -54.2 * Q.z_top50_slots + 51.8152
    if Q.mass_top20 < 69.4 and Q.n_real_top40 > 29.0:
        z += 0.00249 * (69.4 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 36.5 and Q.dr_0 < 0.111:
        z += 0.699 * (Q.n_particles - 36.5) * (0.111 - Q.dr_0)
    return max(0.0, z)


def neuron_2(Q):
    z = 2.1
    if 0.29 <= Q.LHA < 0.427:
        z += 0.878 * Q.LHA - 0.25462
    if Q.LHA >= 0.427:
        z += -8.922 * Q.LHA + 3.92998
    if Q.e2 >= 0.0602:
        z += -4.85 * Q.e2 + 0.29197
    if Q.e3 >= 0.00037:
        z += -370.0 * Q.e3 + 0.1369
    if Q.girth2 >= 0.0089:
        z += 142.0 * Q.girth2 - 1.2638
    if Q.girth2_top15 < 0.0211:
        z += -36.5 * Q.girth2_top15 + 0.77015
    if Q.girth2_top30 >= 0.00765:
        z += -59.2 * Q.girth2_top30 + 0.45288
    if Q.lam2 < 0.00145:
        z += 76.7 * Q.lam2 - 0.111215
    if 6.9 <= Q.log_sum_pt < 6.96:
        z += -0.0346 * Q.log_sum_pt + 0.23874
    if Q.log_sum_pt >= 6.96:
        z += -4.1846 * Q.log_sum_pt + 29.12274
    if Q.mass < 64.5:
        z += 0.0436 * Q.mass - 4.08968
    if 64.5 <= Q.mass < 93.8:
        z += 0.0317 * Q.mass - 3.32213
    if Q.mass >= 93.8:
        z += -0.0119 * Q.mass + 0.76755
    if Q.mass_over_sum_pt < 0.0612:
        z += -46.1 * Q.mass_over_sum_pt + 2.82132
    if Q.mass_over_sum_pt >= 0.0612:
        z += -23.6 * Q.mass_over_sum_pt + 1.44432
    if Q.mass_top15 >= 31.5:
        z += 0.00623 * Q.mass_top15 - 0.196245
    if Q.mass_top20 >= 109.0:
        z += 0.00518 * Q.mass_top20 - 0.56462
    if Q.mass_top30 >= 88.3:
        z += 0.0114 * Q.mass_top30 - 1.00662
    if Q.n_dr_0p1_0p2 >= 21.6:
        z += -0.00789 * Q.n_dr_0p1_0p2 + 0.170424
    if Q.n_particles < 50.2:
        z += 0.0106 * Q.n_particles - 0.53212
    if Q.psi_0p2 >= 0.816:
        z += -0.0781 * Q.psi_0p2 + 0.0637296
    if 0.961 <= Q.psi_0p3 < 0.993:
        z += -5.64 * Q.psi_0p3 + 5.42004
    if Q.psi_0p3 >= 0.993:
        z += -24.84 * Q.psi_0p3 + 24.48564
    if Q.sd_mass >= 85.3:
        z += -0.00927 * Q.sd_mass + 0.790731
    if Q.sd_rg >= 0.264:
        z += 2.7 * Q.sd_rg - 0.7128
    if Q.sj2_dr >= 0.174:
        z += -0.482 * Q.sj2_dr + 0.083868
    if Q.sum_pt < 963.0:
        z += 0.00514 * Q.sum_pt - 6.5907
    if 963.0 <= Q.sum_pt < 1020.0:
        z += 0.01464 * Q.sum_pt - 15.7392
    if 1020.0 <= Q.sum_pt < 1030.0:
        z += 0.02704 * Q.sum_pt - 28.3872
    if 1030.0 <= Q.sum_pt < 1110.0:
        z += 0.01504 * Q.sum_pt - 16.0272
    if 1110.0 <= Q.sum_pt < 1280.0:
        z += 0.00974 * Q.sum_pt - 10.1442
    if Q.sum_pt >= 1280.0:
        z += 0.0071 * Q.sum_pt - 6.765
    if Q.sum_pt_top15 >= 1040.0:
        z += 0.00155 * Q.sum_pt_top15 - 1.612
    if 987.0 <= Q.sum_pt_top40 < 1060.0:
        z += -0.000936 * Q.sum_pt_top40 + 0.923832
    if Q.sum_pt_top40 >= 1060.0:
        z += 0.001264 * Q.sum_pt_top40 - 1.408168
    if Q.sum_pt_top50 < 933.0:
        z += -0.0041 * Q.sum_pt_top50 + 4.223
    if 933.0 <= Q.sum_pt_top50 < 1030.0:
        z += -0.01105 * Q.sum_pt_top50 + 10.70735
    if Q.sum_pt_top50 >= 1030.0:
        z += -0.00695 * Q.sum_pt_top50 + 6.48435
    if Q.z_dr_0p1_0p2 < 0.324:
        z += 0.0846 * Q.z_dr_0p1_0p2 - 0.0366318
    if 0.324 <= Q.z_dr_0p1_0p2 < 0.433:
        z += -0.0394 * Q.z_dr_0p1_0p2 + 0.0035442
    if Q.z_dr_0p1_0p2 >= 0.433:
        z += -0.124 * Q.z_dr_0p1_0p2 + 0.040176
    if Q.z_top40_slots < 0.997:
        z += 2.28 * Q.z_top40_slots - 2.27316
    if Q.z_top5_slots >= 0.498:
        z += -0.255 * Q.z_top5_slots + 0.12699
    if Q.log_sum_pt > 6.9 and Q.girth2_top15 < 0.0209:
        z += 248.0 * (Q.log_sum_pt - 6.9) * (0.0209 - Q.girth2_top15)
    if Q.sum_pt_top50 > 995.0 and Q.girth2_top15 < 0.0218:
        z += -0.15 * (Q.sum_pt_top50 - 995.0) * (0.0218 - Q.girth2_top15)
    return max(0.0, z)


def neuron_3(Q):
    z = 1.2
    if Q.D2 < 1.45:
        z += 0.552 * Q.D2 - 0.8004
    if Q.D2_b2 < 0.533:
        z += -0.618 * Q.D2_b2 + 0.329394
    if Q.LHA >= 0.398:
        z += 11.5 * Q.LHA - 4.577
    if Q.e2 < 0.0287:
        z += -32.2 * Q.e2 + 0.92414
    if Q.girth2 < 0.00974:
        z += -256.0 * Q.girth2 + 2.49344
    if Q.girth2_top10 < 0.00851:
        z += 112.0 * Q.girth2_top10 - 0.95312
    if Q.lam1 >= 0.00748:
        z += 80.4 * Q.lam1 - 0.601392
    if Q.lam2 < 0.00121:
        z += -156.0 * Q.lam2 + 0.18876
    if Q.mass < 66.2:
        z += 0.0193 * Q.mass - 1.76981
    if 66.2 <= Q.mass < 79.6:
        z += 0.01216 * Q.mass - 1.297142
    if 79.6 <= Q.mass < 91.7:
        z += -0.01294 * Q.mass + 0.700818
    if 91.7 <= Q.mass < 95.4:
        z += -0.03224 * Q.mass + 2.470628
    if 95.4 <= Q.mass < 160.0:
        z += 0.01396 * Q.mass - 1.936852
    if Q.mass >= 160.0:
        z += 0.02396 * Q.mass - 3.536852
    if Q.mass_over_sum_pt >= 0.1:
        z += -30.5 * Q.mass_over_sum_pt + 3.05
    if Q.mass_top10 >= 20.6:
        z += -0.0118 * Q.mass_top10 + 0.24308
    if Q.mass_top30 >= 46.8:
        z += 0.0108 * Q.mass_top30 - 0.50544
    if Q.mass_top50 >= 95.1:
        z += -0.0237 * Q.mass_top50 + 2.25387
    if Q.n_dr_0p05_0p1 >= 9.06:
        z += -0.011 * Q.n_dr_0p05_0p1 + 0.09966
    if Q.n_dr_0p1_0p2 < 25.2:
        z += -0.0262 * Q.n_dr_0p1_0p2 + 0.66024
    if Q.n_dr_0p2_0p4 < 12.6:
        z += -0.0361 * Q.n_dr_0p2_0p4 + 0.45486
    if Q.n_particles >= 24.4:
        z += -0.0245 * Q.n_particles + 0.5978
    if Q.n_pt_above_1 < 24.1:
        z += -0.0257 * Q.n_pt_above_1 + 0.61937
    if 0.966 <= Q.psi_0p3 < 0.997:
        z += -7.62 * Q.psi_0p3 + 7.36092
    if Q.psi_0p3 >= 0.997:
        z += 54.68 * Q.psi_0p3 - 54.75218
    if Q.tau1 < 0.173:
        z += 12.8 * Q.tau1 - 2.2144
    if Q.z_top50_slots < 0.99:
        z += 17.8 * Q.z_top50_slots - 17.622
    if Q.n_dr_0p2_0p4 < 4.05 and Q.eccentricity > 0.75:
        z += 0.445 * (4.05 - Q.n_dr_0p2_0p4) * (Q.eccentricity - 0.75)
    if Q.n_dr_0p2_0p4 < 5.39 and Q.psi_0p3 > 0.993:
        z += 11.2 * (5.39 - Q.n_dr_0p2_0p4) * (Q.psi_0p3 - 0.993)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.138
    if Q.girth >= 0.126:
        z += -15.3 * Q.girth + 1.9278
    if 0.00761 <= Q.girth2_top15 < 0.0161:
        z += -62.6 * Q.girth2_top15 + 0.476386
    if Q.girth2_top15 >= 0.0161:
        z += 4.0 * Q.girth2_top15 - 0.595874
    if Q.girth2_top30 < 0.0072:
        z += 84.4 * Q.girth2_top30 - 0.60768
    if Q.girth2_top50 >= 0.00885:
        z += 115.0 * Q.girth2_top50 - 1.01775
    if Q.lam1 < 0.00751:
        z += 93.4 * Q.lam1 - 0.701434
    if Q.lam1 >= 0.0191:
        z += -35.2 * Q.lam1 + 0.67232
    if Q.log_sum_pt < 6.98:
        z += 1.85 * Q.log_sum_pt - 12.913
    if Q.mass < 80.0:
        z += 0.0786 * Q.mass - 5.52656
    if 80.0 <= Q.mass < 86.8:
        z += 0.0118 * Q.mass - 0.18256
    if 86.8 <= Q.mass < 112.0:
        z += -0.0334 * Q.mass + 3.7408
    if Q.mass >= 154.0:
        z += 0.00954 * Q.mass - 1.46916
    if Q.mass_top15 >= 102.0:
        z += -0.0167 * Q.mass_top15 + 1.7034
    if Q.n_dr_0_0p05 >= 14.1:
        z += 0.0186 * Q.n_dr_0_0p05 - 0.26226
    if Q.n_dr_0p2_0p4 < 21.2:
        z += -0.0344 * Q.n_dr_0p2_0p4 + 0.72928
    if Q.n_particles < 48.7:
        z += -0.031 * Q.n_particles + 1.5283
    if 48.7 <= Q.n_particles < 49.3:
        z += -0.0491 * Q.n_particles + 2.40977
    if Q.n_particles >= 49.3:
        z += -0.0181 * Q.n_particles + 0.88147
    if Q.psi_0p3 >= 0.997:
        z += 134.0 * Q.psi_0p3 - 133.598
    if Q.pt_entropy < 2.77:
        z += 0.208 * Q.pt_entropy - 0.57616
    if Q.sj2_dr >= 0.245:
        z += -2.25 * Q.sj2_dr + 0.55125
    if Q.sj3_pair_mass_min >= 27.5:
        z += 0.00641 * Q.sj3_pair_mass_min - 0.176275
    if Q.z_top50_slots < 0.988:
        z += 15.9 * Q.z_top50_slots - 15.7092
    if Q.mass < 110.0 and Q.psi_0p3 < 1.0:
        z += -0.321 * (110.0 - Q.mass) * (1.0 - Q.psi_0p3)
    if Q.n_dr_0p2_0p4 < 20.4 and Q.n_dr_0p1_0p2 > 12.0:
        z += -0.000912 * (20.4 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 12.0)
    if Q.psi_0p3 > 0.996 and Q.n_dr_0_0p05 < 14.2:
        z += -7.63 * (Q.psi_0p3 - 0.996) * (14.2 - Q.n_dr_0_0p05)
    return max(0.0, z)


def neuron_5(Q):
    z = 2.96
    if Q.e2 < 0.0362:
        z += -10.5 * Q.e2 + 0.3801
    if Q.girth2_top15 < 0.000928:
        z += 277.0 * Q.girth2_top15 - 0.257056
    if Q.log_sum_pt < 6.86:
        z += -12.8 * Q.log_sum_pt + 87.808
    if 6.91 <= Q.log_sum_pt < 7.05:
        z += -34.4 * Q.log_sum_pt + 237.704
    if Q.log_sum_pt >= 7.05:
        z += -14.7 * Q.log_sum_pt + 98.819
    if Q.mass < 65.8:
        z += -0.0301 * Q.mass + 1.98058
    if 91.5 <= Q.mass < 150.0:
        z += -0.0214 * Q.mass + 1.9581
    if Q.mass >= 150.0:
        z += -0.0494 * Q.mass + 6.1581
    if Q.mass_over_sum_pt_sq < 0.0258:
        z += -81.0 * Q.mass_over_sum_pt_sq + 2.0898
    if Q.mass_top40 >= 112.0:
        z += -0.0127 * Q.mass_top40 + 1.4224
    if Q.mass_top5 >= 55.6:
        z += 0.00547 * Q.mass_top5 - 0.304132
    if Q.mass_top50 < 94.9:
        z += 0.0284 * Q.mass_top50 - 2.69516
    if Q.mass_top50 >= 97.3:
        z += 0.0331 * Q.mass_top50 - 3.22063
    if Q.n_dr_0p1_0p2 >= 7.0:
        z += 0.00908 * Q.n_dr_0p1_0p2 - 0.06356
    if Q.n_particles < 63.0:
        z += -0.0261 * Q.n_particles + 1.6443
    if Q.sd_mass >= 65.3:
        z += 0.00673 * Q.sd_mass - 0.439469
    if Q.sum_pt < 968.0:
        z += 0.03 * Q.sum_pt - 31.8624
    if 968.0 <= Q.sum_pt < 1160.0:
        z += 0.0147 * Q.sum_pt - 17.052
    if Q.sum_pt_top15 >= 740.0:
        z += -0.00265 * Q.sum_pt_top15 + 1.961
    if Q.sum_pt_top5 < 835.0:
        z += 0.00132 * Q.sum_pt_top5 - 1.1022
    if Q.sum_pt_top50 >= 968.0:
        z += 0.0103 * Q.sum_pt_top50 - 9.9704
    if Q.width >= 0.0247:
        z += -90.1 * Q.width + 2.22547
    if Q.z_top50_slots >= 0.989:
        z += -33.4 * Q.z_top50_slots + 33.0326
    if Q.z_top5_slots >= 0.772:
        z += 1.61 * Q.z_top5_slots - 1.24292
    if Q.n_particles < 61.5 and Q.D2 < 2.57:
        z += -0.0114 * (61.5 - Q.n_particles) * (2.57 - Q.D2)
    if Q.psi_0p3 > 0.997 and Q.D2 < 3.83:
        z += 51.4 * (Q.psi_0p3 - 0.997) * (3.83 - Q.D2)
    return max(0.0, z)


def neuron_6(Q):
    z = 13.5
    if Q.LHA < 0.203:
        z += 2.39 * Q.LHA - 1.00911
    if 0.203 <= Q.LHA < 0.27:
        z += 7.82 * Q.LHA - 2.1114
    if Q.LHA >= 0.4:
        z += 30.1 * Q.LHA - 12.04
    if Q.e2 < 0.0152:
        z += -7.4 * Q.e2 + 0.64532
    if 0.0152 <= Q.e2 < 0.0287:
        z += -34.6 * Q.e2 + 1.05876
    if 0.0287 <= Q.e2 < 0.0306:
        z += -102.4 * Q.e2 + 3.00462
    if Q.e2 >= 0.0306:
        z += -67.8 * Q.e2 + 1.94586
    if Q.e2_sq >= 0.00656:
        z += -538.0 * Q.e2_sq + 3.52928
    if Q.e3 >= 1.71e-05:
        z += 2740.0 * Q.e3 - 0.046854
    if Q.eccentricity >= 0.929:
        z += 5.2 * Q.eccentricity - 4.8308
    if Q.girth < 0.0627:
        z += -45.3 * Q.girth + 2.84031
    if Q.girth2_top20 >= 0.00768:
        z += 77.0 * Q.girth2_top20 - 0.59136
    if Q.girth2_top50 < 0.00419:
        z += 116.0 * Q.girth2_top50 - 0.48604
    if Q.lam1 >= 0.00674:
        z += 202.0 * Q.lam1 - 1.36148
    if Q.lam2 < 0.000618:
        z += 442.0 * Q.lam2 - 0.273156
    if Q.lam2 >= 0.00381:
        z += 141.0 * Q.lam2 - 0.53721
    if Q.log_sum_pt < 6.8:
        z += 3.71 * Q.log_sum_pt - 25.228
    if 6.97 <= Q.log_sum_pt < 7.05:
        z += 9.35 * Q.log_sum_pt - 65.1695
    if Q.log_sum_pt >= 7.05:
        z += -0.33 * Q.log_sum_pt + 3.0745
    if Q.mass < 84.4:
        z += -0.0163 * Q.mass - 3.728
    if 84.4 <= Q.mass < 102.0:
        z += 0.0477 * Q.mass - 9.1296
    if 102.0 <= Q.mass < 120.0:
        z += 0.0049 * Q.mass - 4.764
    if Q.mass >= 120.0:
        z += -0.0348 * Q.mass
    if 0.0796 <= Q.mass_over_sum_pt < 0.102:
        z += 76.9 * Q.mass_over_sum_pt - 6.12124
    if Q.mass_over_sum_pt >= 0.102:
        z += 49.6 * Q.mass_over_sum_pt - 3.33664
    if Q.mass_top10 >= 98.2:
        z += 0.0124 * Q.mass_top10 - 1.21768
    if Q.mass_top15 < 89.1:
        z += -0.00524 * Q.mass_top15 + 0.466884
    if Q.mass_top30 >= 80.1:
        z += 0.0239 * Q.mass_top30 - 1.91439
    if Q.mass_top40 < 75.5:
        z += -0.0154 * Q.mass_top40 + 1.1627
    if Q.pt_dispersion < 0.382:
        z += -2.77 * Q.pt_dispersion + 1.05814
    if 2.46 <= Q.pt_entropy < 2.93:
        z += -0.643 * Q.pt_entropy + 1.58178
    if Q.pt_entropy >= 2.93:
        z += -1.083 * Q.pt_entropy + 2.87098
    if Q.sum_pt < 1060.0:
        z += 0.00696 * Q.sum_pt - 7.3776
    if Q.sum_pt >= 1160.0:
        z += 0.00679 * Q.sum_pt - 7.8764
    z += -0.00628 * Q.sum_pt_top50
    if Q.tau1 < 0.172:
        z += 24.9 * Q.tau1 - 4.2828
    if Q.z_dr_0p1_0p2 < 0.263:
        z += -0.925 * Q.z_dr_0p1_0p2 + 0.243275
    if Q.e2 > 0.0489 and Q.D2_b2 < 1.6:
        z += 27.7 * (Q.e2 - 0.0489) * (1.6 - Q.D2_b2)
    if Q.mass < 122.0 and Q.sum_pt < 1000.0:
        z += 8.54e-05 * (122.0 - Q.mass) * (1000.0 - Q.sum_pt)
    if Q.mass_over_sum_pt > 0.0295 and Q.z_dr_0p05_0p1 < 0.694:
        z += -3.47 * (Q.mass_over_sum_pt - 0.0295) * (0.694 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_7(Q):
    z = -4.73
    if 0.32 <= Q.LHA < 0.395:
        z += 29.2 * Q.LHA - 9.344
    if Q.LHA >= 0.395:
        z += -7.3 * Q.LHA + 5.0735
    if Q.e2 >= 0.0342:
        z += 36.0 * Q.e2 - 1.2312
    if Q.girth >= 0.0894:
        z += -58.1 * Q.girth + 5.19414
    if Q.girth2 < 0.00836:
        z += 425.0 * Q.girth2 - 3.553
    if Q.girth2_top15 >= 0.00865:
        z += -94.7 * Q.girth2_top15 + 0.819155
    if Q.girth2_top30 < 0.00816:
        z += 282.0 * Q.girth2_top30 - 2.30112
    if Q.girth2_top50 < 0.00944:
        z += -904.0 * Q.girth2_top50 + 12.30782
    if 0.00944 <= Q.girth2_top50 < 0.0251:
        z += -241.0 * Q.girth2_top50 + 6.0491
    if Q.lam1 < 0.00761:
        z += 316.0 * Q.lam1 - 2.607
    if 0.00761 <= Q.lam1 < 0.00825:
        z += 549.0 * Q.lam1 - 4.38013
    if Q.lam1 >= 0.00825:
        z += 233.0 * Q.lam1 - 1.77313
    if Q.lam2 >= 0.000777:
        z += 245.0 * Q.lam2 - 0.190365
    if Q.log_sum_pt >= 6.96:
        z += 1.17 * Q.log_sum_pt - 8.1432
    if Q.mass < 49.5:
        z += -0.1596 * Q.mass + 13.2207
    if 49.5 <= Q.mass < 82.0:
        z += -0.1154 * Q.mass + 11.0328
    if 82.0 <= Q.mass < 100.0:
        z += -0.0785 * Q.mass + 8.007
    if 100.0 <= Q.mass < 102.0:
        z += -0.1437 * Q.mass + 14.527
    if 102.0 <= Q.mass < 125.0:
        z += -0.0652 * Q.mass + 6.52
    if Q.mass >= 125.0:
        z += -0.0477 * Q.mass + 4.3325
    if Q.mass_top30 < 80.7:
        z += -0.0407 * Q.mass_top30 + 3.28449
    if Q.mass_top50 < 83.6:
        z += 0.104 * Q.mass_top50 - 10.036
    if 83.6 <= Q.mass_top50 < 96.5:
        z += 0.1493 * Q.mass_top50 - 13.82308
    if Q.mass_top50 >= 96.5:
        z += 0.0453 * Q.mass_top50 - 3.78708
    if Q.n_dr_0p2_0p4 < 6.54:
        z += -0.112 * Q.n_dr_0p2_0p4 + 0.73248
    if Q.n_particles < 47.5:
        z += -0.012 * Q.n_particles + 0.57
    if Q.psi_0p2 >= 0.806:
        z += -2.54 * Q.psi_0p2 + 2.04724
    if Q.psi_0p3 >= 0.964:
        z += -10.5 * Q.psi_0p3 + 10.122
    if Q.pt_entropy < 2.9:
        z += 0.338 * Q.pt_entropy - 0.9802
    if Q.sd_mass >= 92.8:
        z += -0.0189 * Q.sd_mass + 1.75392
    if Q.sd_rg < 0.158:
        z += -1.8 * Q.sd_rg + 1.1466
    if 0.158 <= Q.sd_rg < 0.214:
        z += 6.06 * Q.sd_rg - 0.09528
    if 0.214 <= Q.sd_rg < 0.268:
        z += -6.24 * Q.sd_rg + 2.53692
    if Q.sd_rg >= 0.268:
        z += 7.86 * Q.sd_rg - 1.24188
    if Q.sum_pt < 1090.0:
        z += 0.00208 * Q.sum_pt - 2.2672
    if Q.tau1 < 0.0859:
        z += 4.2 * Q.tau1 - 0.80057
    if 0.0859 <= Q.tau1 < 0.108:
        z += 19.9 * Q.tau1 - 2.1492
    if Q.mass < 90.4 and Q.psi_0p3 > 0.997:
        z += -74.8 * (90.4 - Q.mass) * (Q.psi_0p3 - 0.997)
    if Q.mass < 93.0 and Q.psi_0p3 > 0.979:
        z += -6.29 * (93.0 - Q.mass) * (Q.psi_0p3 - 0.979)
    if Q.mass < 95.3 and Q.psi_0p3 > 0.967:
        z += -2.15 * (95.3 - Q.mass) * (Q.psi_0p3 - 0.967)
    if Q.mass < 99.7 and Q.psi_0p3 > 0.997:
        z += 41.0 * (99.7 - Q.mass) * (Q.psi_0p3 - 0.997)
    if Q.mass < 103.0 and Q.psi_0p3 > 0.977:
        z += 5.82 * (103.0 - Q.mass) * (Q.psi_0p3 - 0.977)
    if Q.mass < 91.7 and Q.z_dr_0_0p05 < 0.577:
        z += -0.123 * (91.7 - Q.mass) * (0.577 - Q.z_dr_0_0p05)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.51
    if Q.e2 < 0.0443:
        z += -22.3 * Q.e2 + 0.98789
    if Q.girth2_top20 >= 0.0074:
        z += 45.7 * Q.girth2_top20 - 0.33818
    if Q.log_sum_pt < 7.06:
        z += 15.7 * Q.log_sum_pt - 110.842
    if Q.mass < 83.1:
        z += 0.0107 * Q.mass - 2.44275
    if 83.1 <= Q.mass < 102.0:
        z += 0.0822 * Q.mass - 8.3844
    if Q.mass >= 124.0:
        z += -0.0604 * Q.mass + 7.4896
    if Q.mass_top50 >= 112.0:
        z += 0.0503 * Q.mass_top50 - 5.6336
    if Q.n_dr_0p2_0p4 < 18.7:
        z += 0.0662 * Q.n_dr_0p2_0p4 - 1.23794
    if Q.sum_pt < 1030.0:
        z += -0.0292 * Q.sum_pt + 31.658
    if 1030.0 <= Q.sum_pt < 1170.0:
        z += -0.0113 * Q.sum_pt + 13.221
    if Q.z_top50_slots >= 0.97:
        z += -28.6 * Q.z_top50_slots + 27.742
    if Q.girth2_top40 > 0.00536 and Q.log_sum_pt < 6.98:
        z += -746.0 * (Q.girth2_top40 - 0.00536) * (6.98 - Q.log_sum_pt)
    if Q.mass_over_sum_pt > 0.0747 and Q.sum_pt < 1100.0:
        z += 0.247 * (Q.mass_over_sum_pt - 0.0747) * (1100.0 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 20.9 and Q.z_dr_0p1_0p2 > 0.228:
        z += 0.136 * (20.9 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p1_0p2 - 0.228)
    return max(0.0, z)


def neuron_9(Q):
    z = 9.91
    if Q.girth < 0.0419:
        z += 28.1 * Q.girth - 1.17739
    if Q.girth >= 0.104:
        z += 30.9 * Q.girth - 3.2136
    if Q.girth2_top40 < 0.00887:
        z += -140.0 * Q.girth2_top40 + 1.2418
    if Q.mass < 65.1:
        z += -0.0494 * Q.mass - 5.51458
    if 65.1 <= Q.mass < 69.5:
        z += -0.1684 * Q.mass + 2.23232
    if 69.5 <= Q.mass < 88.3:
        z += -0.073 * Q.mass - 4.39798
    if 88.3 <= Q.mass < 143.0:
        z += 0.0124 * Q.mass - 11.9388
    if 143.0 <= Q.mass < 158.0:
        z += -0.0593 * Q.mass - 1.6857
    if Q.mass >= 158.0:
        z += -0.119 * Q.mass + 7.7469
    if Q.mass_over_sum_pt >= 0.159:
        z += 180.0 * Q.mass_over_sum_pt - 28.62
    if Q.mass_over_sum_pt_sq >= 0.0255:
        z += -646.0 * Q.mass_over_sum_pt_sq + 16.473
    if Q.mass_top30 >= 125.0:
        z += 0.0302 * Q.mass_top30 - 3.775
    if Q.n_particles >= 40.4:
        z += 0.0274 * Q.n_particles - 1.10696
    if Q.pt_entropy < 2.36:
        z += -0.547 * Q.pt_entropy + 1.29092
    if Q.sum_pt < 994.0:
        z += -0.01 * Q.sum_pt + 9.94
    if Q.mass_top40 < 110.0 and Q.n_dr_0p2_0p4 > 9.63:
        z += 0.000885 * (110.0 - Q.mass_top40) * (Q.n_dr_0p2_0p4 - 9.63)
    if Q.mass_top40 < 96.8 and Q.sum_pt < 1020.0:
        z += -0.000272 * (96.8 - Q.mass_top40) * (1020.0 - Q.sum_pt)
    if Q.mass_top40 < 101.0 and Q.sum_pt_top40 < 918.0:
        z += 0.00019 * (101.0 - Q.mass_top40) * (918.0 - Q.sum_pt_top40)
    return max(0.0, z)


def neuron_10(Q):
    z = 0.577
    if Q.LHA < 0.276:
        z += -19.5 * Q.LHA + 5.382
    if Q.LHA >= 0.276:
        z += -27.6 * Q.LHA + 7.6176
    if Q.e2 >= 0.0131:
        z += 63.5 * Q.e2 - 0.83185
    if Q.e3 < 0.000137:
        z += -5860.0 * Q.e3 + 0.80282
    if Q.girth < 0.0208:
        z += 177.1 * Q.girth - 9.85209
    if 0.0208 <= Q.girth < 0.0389:
        z += 137.5 * Q.girth - 9.02841
    if 0.0389 <= Q.girth < 0.0881:
        z += 73.3 * Q.girth - 6.53103
    if 0.0881 <= Q.girth < 0.0891:
        z += 133.8 * Q.girth - 11.86108
    if Q.girth >= 0.0891:
        z += 60.5 * Q.girth - 5.33005
    if Q.girth2_top50 >= 0.00838:
        z += -69.7 * Q.girth2_top50 + 0.584086
    if Q.mass < 59.9:
        z += -0.0449 * Q.mass + 5.29783
    if 59.9 <= Q.mass < 78.2:
        z += -0.0858 * Q.mass + 7.74774
    if 78.2 <= Q.mass < 90.3:
        z += -0.0291 * Q.mass + 3.3138
    if 90.3 <= Q.mass < 98.1:
        z += 0.0567 * Q.mass - 4.43394
    if 98.1 <= Q.mass < 147.0:
        z += 0.0171 * Q.mass - 0.54918
    if 147.0 <= Q.mass < 159.0:
        z += -0.1209 * Q.mass + 19.73682
    if Q.mass >= 159.0:
        z += -0.1893 * Q.mass + 30.61242
    if Q.mass_over_sum_pt_sq >= 0.0273:
        z += -118.0 * Q.mass_over_sum_pt_sq + 3.2214
    if 18.7 <= Q.mass_top10 < 75.1:
        z += 0.0131 * Q.mass_top10 - 0.24497
    if Q.mass_top10 >= 75.1:
        z += -0.0066 * Q.mass_top10 + 1.2345
    if Q.mass_top20 >= 74.7:
        z += -0.0209 * Q.mass_top20 + 1.56123
    if Q.mass_top5 >= 25.7:
        z += 0.0137 * Q.mass_top5 - 0.35209
    if Q.mass_top50 >= 142.0:
        z += 0.118 * Q.mass_top50 - 16.756
    if Q.n_dr_0_0p05 < 11.9:
        z += -0.0454 * Q.n_dr_0_0p05 + 0.54026
    if Q.n_dr_0_0p05 >= 12.0:
        z += -0.0403 * Q.n_dr_0_0p05 + 0.4836
    if Q.n_dr_0p05_0p1 >= 4.41:
        z += -0.0388 * Q.n_dr_0p05_0p1 + 0.171108
    if Q.n_dr_0p1_0p2 >= 6.11:
        z += -0.0319 * Q.n_dr_0p1_0p2 + 0.194909
    if Q.n_dr_0p2_0p4 >= 9.37:
        z += -0.0337 * Q.n_dr_0p2_0p4 + 0.315769
    if Q.n_particles >= 33.1:
        z += 0.0457 * Q.n_particles - 1.51267
    if Q.pt_entropy < 2.46:
        z += 0.524 * Q.pt_entropy - 1.28904
    if Q.sum_pt < 973.0:
        z += 0.0035 * Q.sum_pt - 3.4055
    if Q.z_top5_slots >= 0.339:
        z += -2.54 * Q.z_top5_slots + 0.86106
    if Q.D2 < 2.48 and Q.sj2_dr > 0.205:
        z += -2.37 * (2.48 - Q.D2) * (Q.sj2_dr - 0.205)
    if Q.girth2_top10 < 0.0204 and Q.psi_0p3 > 0.998:
        z += -23000.0 * (0.0204 - Q.girth2_top10) * (Q.psi_0p3 - 0.998)
    if Q.girth2_top30 < 0.00789 and Q.psi_0p3 > 0.998:
        z += 56000.0 * (0.00789 - Q.girth2_top30) * (Q.psi_0p3 - 0.998)
    if Q.sum_pt < 1000.0 and Q.dr_3 > 0.00491:
        z += -0.0244 * (1000.0 - Q.sum_pt) * (Q.dr_3 - 0.00491)
    return max(0.0, z)


def neuron_11(Q):
    z = 1.98
    if Q.D2 < 1.59:
        z += -0.35 * Q.D2 + 0.5565
    if Q.LHA >= 0.176:
        z += -14.3 * Q.LHA + 2.5168
    if Q.e2 < 0.0255:
        z += -54.5 * Q.e2 + 0.57597
    if 0.0255 <= Q.e2 < 0.0552:
        z += 27.4 * Q.e2 - 1.51248
    if Q.e2_sq < 0.00821:
        z += -177.0 * Q.e2_sq + 1.45317
    if Q.e2_sq >= 0.00885:
        z += 237.0 * Q.e2_sq - 2.09745
    if Q.girth < 0.0328:
        z += 6.8 * Q.girth - 2.12384
    if 0.0328 <= Q.girth < 0.0768:
        z += 43.2 * Q.girth - 3.31776
    if Q.girth2_top10 < 0.00394:
        z += -61.0 * Q.girth2_top10 - 0.22386
    if 0.00394 <= Q.girth2_top10 < 0.00816:
        z += 110.0 * Q.girth2_top10 - 0.8976
    if Q.girth2_top15 >= 0.00856:
        z += 57.5 * Q.girth2_top15 - 0.4922
    if Q.girth2_top30 < 0.0066:
        z += 189.0 * Q.girth2_top30 - 1.2474
    if Q.mass < 79.8:
        z += -0.0219 * Q.mass + 3.38062
    if 79.8 <= Q.mass < 94.2:
        z += -0.0898 * Q.mass + 8.79904
    if 94.2 <= Q.mass < 100.0:
        z += -0.0586 * Q.mass + 5.86
    if Q.mass_over_sum_pt >= 0.097:
        z += -81.4 * Q.mass_over_sum_pt + 7.8958
    if Q.mass_top50 < 82.7:
        z += 0.0404 * Q.mass_top50 - 3.34108
    if Q.n_dr_0_0p05 >= 18.0:
        z += -0.0158 * Q.n_dr_0_0p05 + 0.2844
    if Q.n_dr_0p05_0p1 >= 14.1:
        z += -0.0145 * Q.n_dr_0p05_0p1 + 0.20445
    if Q.n_dr_0p1_0p2 >= 7.42:
        z += -0.0291 * Q.n_dr_0p1_0p2 + 0.215922
    if Q.n_dr_0p2_0p4 < 5.79:
        z += -0.0944 * Q.n_dr_0p2_0p4 + 0.964595
    if 5.79 <= Q.n_dr_0p2_0p4 < 15.1:
        z += -0.0449 * Q.n_dr_0p2_0p4 + 0.67799
    if Q.psi_0p2 >= 0.995:
        z += 42.4 * Q.psi_0p2 - 42.188
    if Q.sum_pt_top5 >= 762.0:
        z += -0.00089 * Q.sum_pt_top5 + 0.67818
    if Q.mass < 90.1 and Q.zdr_4 < 0.00447:
        z += -7.13 * (90.1 - Q.mass) * (0.00447 - Q.zdr_4)
    if Q.mass_over_sum_pt_sq < 0.00909 and Q.z_dr_0p1_0p2 > 0.0755:
        z += 478.0 * (0.00909 - Q.mass_over_sum_pt_sq) * (Q.z_dr_0p1_0p2 - 0.0755)
    if Q.mass_top50 < 79.9 and Q.zdr_4 < 0.00411:
        z += 10.7 * (79.9 - Q.mass_top50) * (0.00411 - Q.zdr_4)
    if Q.n_dr_0p2_0p4 < 9.41 and Q.girth2 < 0.0059:
        z += -29.7 * (9.41 - Q.n_dr_0p2_0p4) * (0.0059 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 9.12 and Q.n_dr_0p1_0p2 < 20.8:
        z += 0.00549 * (9.12 - Q.n_dr_0p2_0p4) * (20.8 - Q.n_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 9.72 and Q.n_real_top40 > 25.2:
        z += -0.00238 * (9.72 - Q.n_dr_0p2_0p4) * (Q.n_real_top40 - 25.2)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.604
    if Q.LHA >= 0.368:
        z += 15.7 * Q.LHA - 5.7776
    if Q.girth2_top10 < 0.00732:
        z += 57.2 * Q.girth2_top10 - 0.418704
    if Q.girth2_top20 >= 0.018:
        z += -56.4 * Q.girth2_top20 + 1.0152
    if Q.lam2 >= 0.000289:
        z += -90.7 * Q.lam2 + 0.0262123
    if Q.mass < 60.2:
        z += -0.0546 * Q.mass + 6.26071
    if 60.2 <= Q.mass < 81.7:
        z += -0.0971 * Q.mass + 8.81921
    if 81.7 <= Q.mass < 85.5:
        z += -0.0187 * Q.mass + 2.41393
    if 85.5 <= Q.mass < 99.7:
        z += -0.0574 * Q.mass + 5.72278
    if 115.0 <= Q.mass < 183.0:
        z += 0.0168 * Q.mass - 1.932
    if Q.mass >= 183.0:
        z += 0.0029 * Q.mass + 0.6117
    if Q.mass_over_sum_pt < 0.0719:
        z += 18.5 * Q.mass_over_sum_pt - 1.33015
    if Q.n_dr_0p2_0p4 < 21.1:
        z += -0.0162 * Q.n_dr_0p2_0p4 + 0.34182
    if Q.pt_entropy >= 2.17:
        z += -0.326 * Q.pt_entropy + 0.70742
    if 65.6 <= Q.sd_mass < 83.1:
        z += -0.0185 * Q.sd_mass + 1.2136
    if Q.sd_mass >= 83.1:
        z += 0.0074 * Q.sd_mass - 0.93869
    if Q.sum_pt_top50 < 990.0:
        z += 0.00166 * Q.sum_pt_top50 - 1.6434
    if Q.girth2_top10 < 0.000882 and Q.sum_pt_top40 < 1060.0:
        z += 2.56 * (0.000882 - Q.girth2_top10) * (1060.0 - Q.sum_pt_top40)
    if Q.mass < 90.0 and Q.psi_0p3 < 1.0:
        z += -1.2 * (90.0 - Q.mass) * (1.0 - Q.psi_0p3)
    if Q.mass < 79.7 and Q.z_dr_0p1_0p2 < 0.0842:
        z += -0.206 * (79.7 - Q.mass) * (0.0842 - Q.z_dr_0p1_0p2)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.56
    if Q.LHA >= 0.179:
        z += -6.79 * Q.LHA + 1.21541
    if Q.girth < 0.0859:
        z += 34.4 * Q.girth - 3.01688
    if 0.0859 <= Q.girth < 0.0877:
        z += 62.1 * Q.girth - 5.39631
    if Q.girth >= 0.0877:
        z += 27.7 * Q.girth - 2.37943
    if Q.girth2_top15 < 0.000844:
        z += -513.0 * Q.girth2_top15 + 0.432972
    if Q.girth2_top15 >= 0.000862:
        z += -49.2 * Q.girth2_top15 + 0.0424104
    if Q.girth2_top40 < 0.0081:
        z += -64.9 * Q.girth2_top40 + 0.52569
    if Q.log_sum_pt < 6.86:
        z += 28.3 * Q.log_sum_pt - 195.033
    if 6.86 <= Q.log_sum_pt < 6.91:
        z += 22.98 * Q.log_sum_pt - 158.5378
    if Q.log_sum_pt >= 6.91:
        z += 5.08 * Q.log_sum_pt - 34.8488
    if 73.5 <= Q.mass < 100.0:
        z += 0.0277 * Q.mass - 2.03595
    if 100.0 <= Q.mass < 145.0:
        z += 0.0016 * Q.mass + 0.57405
    if 145.0 <= Q.mass < 154.0:
        z += -0.0738 * Q.mass + 11.50705
    if Q.mass >= 154.0:
        z += -0.2308 * Q.mass + 35.68505
    if Q.mass_over_sum_pt >= 0.169:
        z += 101.0 * Q.mass_over_sum_pt - 17.069
    if Q.mass_top10 < 65.7:
        z += 0.0017 * Q.mass_top10 - 0.23779
    if 65.7 <= Q.mass_top10 < 72.2:
        z += 0.0194 * Q.mass_top10 - 1.40068
    if Q.mass_top15 < 77.8:
        z += 0.0082 * Q.mass_top15 - 0.63796
    if Q.mass_top5 >= 50.1:
        z += 0.0116 * Q.mass_top5 - 0.58116
    if Q.mass_top50 >= 147.0:
        z += 0.0949 * Q.mass_top50 - 13.9503
    if Q.n_dr_0_0p05 >= 26.7:
        z += -0.0258 * Q.n_dr_0_0p05 + 0.68886
    if Q.n_dr_0p05_0p1 >= 6.5:
        z += -0.0111 * Q.n_dr_0p05_0p1 + 0.07215
    if Q.n_dr_0p2_0p4 >= 12.5:
        z += -0.0145 * Q.n_dr_0p2_0p4 + 0.18125
    if Q.n_particles >= 21.0:
        z += 0.0114 * Q.n_particles - 0.2394
    if Q.pt_entropy >= 2.11:
        z += 0.202 * Q.pt_entropy - 0.42622
    if Q.sum_pt < 951.0:
        z += 0.00209 * Q.sum_pt - 3.2421
    if 951.0 <= Q.sum_pt < 1110.0:
        z += 0.00789 * Q.sum_pt - 8.7579
    if 964.0 <= Q.sum_pt_top15 < 1000.0:
        z += -0.002 * Q.sum_pt_top15 + 1.928
    if Q.sum_pt_top15 >= 1000.0:
        z += 0.00013 * Q.sum_pt_top15 - 0.202
    if Q.sum_pt_top40 < 986.0:
        z += -0.00488 * Q.sum_pt_top40 + 5.1728
    if 986.0 <= Q.sum_pt_top40 < 1060.0:
        z += -0.00998 * Q.sum_pt_top40 + 10.2014
    if Q.sum_pt_top40 >= 1060.0:
        z += -0.0051 * Q.sum_pt_top40 + 5.0286
    if Q.sum_pt_top50 < 1010.0:
        z += -0.0151 * Q.sum_pt_top50 + 15.251
    if Q.tau1 < 0.0893:
        z += -10.7 * Q.tau1 + 0.95551
    if Q.z_dr_0p1_0p2 >= 0.286:
        z += -0.678 * Q.z_dr_0p1_0p2 + 0.193908
    if Q.z_top40_slots >= 0.933:
        z += 6.37 * Q.z_top40_slots - 5.94321
    if Q.z_top50_slots < 0.971:
        z += 10.9 * Q.z_top50_slots - 10.5839
    if Q.z_top5_slots >= 0.752:
        z += -2.4 * Q.z_top5_slots + 1.8048
    if Q.girth2_top50 < 0.0244 and Q.psi_0p3 > 0.963:
        z += 536.0 * (0.0244 - Q.girth2_top50) * (Q.psi_0p3 - 0.963)
    if Q.mass > 133.0 and Q.D2 > 1.01:
        z += -0.0135 * (Q.mass - 133.0) * (Q.D2 - 1.01)
    if Q.mass > 156.0 and Q.D2 > 1.33:
        z += 0.0296 * (Q.mass - 156.0) * (Q.D2 - 1.33)
    if Q.mass > 48.1 and Q.sum_pt < 1020.0:
        z += -0.000168 * (Q.mass - 48.1) * (1020.0 - Q.sum_pt)
    if Q.mass > 131.0 and Q.sum_pt < 1010.0:
        z += 0.000311 * (Q.mass - 131.0) * (1010.0 - Q.sum_pt)
    if Q.sum_pt < 1040.0 and Q.D2 < 4.15:
        z += -0.00196 * (1040.0 - Q.sum_pt) * (4.15 - Q.D2)
    return max(0.0, z)


def neuron_14(Q):
    z = 2.59
    if Q.LHA >= 0.325:
        z += -9.57 * Q.LHA + 3.11025
    if Q.e2 >= 0.0152:
        z += -17.9 * Q.e2 + 0.27208
    if Q.e2_sq < 0.00955:
        z += -360.0 * Q.e2_sq + 3.438
    if Q.e3 < 0.000109:
        z += 2440.0 * Q.e3 - 0.26596
    if Q.girth >= 0.0339:
        z += 16.2 * Q.girth - 0.54918
    if Q.girth2_top20 < 0.0056:
        z += -74.5 * Q.girth2_top20 + 0.4172
    if Q.girth2_top40 < 0.00873:
        z += 224.0 * Q.girth2_top40 - 1.95552
    if Q.lam1 < 0.00728:
        z += 33.0 * Q.lam1 - 0.55525
    if 0.00728 <= Q.lam1 < 0.00806:
        z += 289.0 * Q.lam1 - 2.41893
    if 0.00806 <= Q.lam1 < 0.00837:
        z += 435.0 * Q.lam1 - 3.59569
    if Q.lam1 >= 0.00837:
        z += 146.0 * Q.lam1 - 1.17676
    if Q.lam2 >= 0.00294:
        z += 172.0 * Q.lam2 - 0.50568
    if Q.log_sum_pt < 6.86:
        z += -7.7 * Q.log_sum_pt + 49.542
    if 6.86 <= Q.log_sum_pt < 7.06:
        z += 16.4 * Q.log_sum_pt - 115.784
    if Q.mass < 84.2:
        z += 0.11065 * Q.mass - 10.399
    if 84.2 <= Q.mass < 91.9:
        z += 0.08485 * Q.mass - 8.22664
    if 91.9 <= Q.mass < 122.0:
        z += -0.01815 * Q.mass + 1.23906
    if Q.mass >= 122.0:
        z += -0.0258 * Q.mass + 2.17236
    if Q.mass_over_sum_pt < 0.0793:
        z += -23.9 * Q.mass_over_sum_pt + 1.26388
    if 0.0793 <= Q.mass_over_sum_pt < 0.0807:
        z += 61.3 * Q.mass_over_sum_pt - 5.49248
    if 0.0807 <= Q.mass_over_sum_pt < 0.0896:
        z += 11.7 * Q.mass_over_sum_pt - 1.48976
    if Q.mass_over_sum_pt >= 0.0896:
        z += -49.6 * Q.mass_over_sum_pt + 4.00272
    if Q.mass_top10 < 86.1:
        z += -0.00384 * Q.mass_top10 + 0.330624
    if Q.mass_top20 >= 85.4:
        z += 0.0108 * Q.mass_top20 - 0.92232
    if Q.mass_top30 < 50.9:
        z += -0.0114 * Q.mass_top30 + 0.9405
    if 50.9 <= Q.mass_top30 < 82.5:
        z += 0.0083 * Q.mass_top30 - 0.06223
    if Q.mass_top30 >= 82.5:
        z += 0.0197 * Q.mass_top30 - 1.00273
    if Q.mass_top40 < 81.3:
        z += -0.0327 * Q.mass_top40 + 2.65851
    if Q.mass_top5 < 66.4:
        z += -0.00412 * Q.mass_top5 + 0.273568
    if Q.n_dr_0_0p05 < 12.0:
        z += 0.145 * Q.n_dr_0_0p05 - 1.74
    if Q.n_dr_0_0p05 >= 12.2:
        z += 0.142 * Q.n_dr_0_0p05 - 1.7324
    if Q.n_dr_0p05_0p1 < 16.9:
        z += 0.139 * Q.n_dr_0p05_0p1 - 2.3769
    if 16.9 <= Q.n_dr_0p05_0p1 < 17.1:
        z += 0.277 * Q.n_dr_0p05_0p1 - 4.7091
    if Q.n_dr_0p05_0p1 >= 17.1:
        z += 0.138 * Q.n_dr_0p05_0p1 - 2.3322
    if Q.n_dr_0p1_0p2 < 10.1:
        z += 0.151 * Q.n_dr_0p1_0p2 - 1.5704
    if 10.1 <= Q.n_dr_0p1_0p2 < 10.4:
        z += 0.287 * Q.n_dr_0p1_0p2 - 2.944
    if Q.n_dr_0p1_0p2 >= 10.4:
        z += 0.136 * Q.n_dr_0p1_0p2 - 1.3736
    if Q.n_dr_0p2_0p4 < 21.1:
        z += 0.12 * Q.n_dr_0p2_0p4 - 2.532
    if Q.n_dr_0p2_0p4 >= 21.4:
        z += 0.136 * Q.n_dr_0p2_0p4 - 2.9104
    if Q.n_particles < 46.0:
        z += -0.134 * Q.n_particles + 6.164
    if Q.n_particles >= 46.2:
        z += -0.157 * Q.n_particles + 7.2534
    if 0.965 <= Q.psi_0p3 < 0.992:
        z += 7.1 * Q.psi_0p3 - 6.8515
    if Q.psi_0p3 >= 0.992:
        z += 81.3 * Q.psi_0p3 - 80.4579
    if Q.sd_mass < 63.7:
        z += -0.00498 * Q.sd_mass + 0.317226
    if Q.sum_pt < 953.0:
        z += 0.0264 * Q.sum_pt - 25.1592
    if Q.sum_pt >= 1170.0:
        z += 0.0127 * Q.sum_pt - 14.859
    if 884.0 <= Q.sum_pt_top15 < 1050.0:
        z += -0.00132 * Q.sum_pt_top15 + 1.16688
    if Q.sum_pt_top15 >= 1050.0:
        z += 9e-05 * Q.sum_pt_top15 - 0.31362
    if Q.sum_pt_top40 < 971.0:
        z += 0.0018 * Q.sum_pt_top40 - 1.38677
    if 971.0 <= Q.sum_pt_top40 < 1050.0:
        z += -0.00457 * Q.sum_pt_top40 + 4.7985
    if Q.sum_pt_top5 < 734.0:
        z += 0.00126 * Q.sum_pt_top5 - 0.92484
    if Q.sum_pt_top50 < 1020.0:
        z += -0.02026 * Q.sum_pt_top50 + 22.4332
    if 1020.0 <= Q.sum_pt_top50 < 1150.0:
        z += -0.0136 * Q.sum_pt_top50 + 15.64
    if Q.sum_pt_top50 >= 1150.0:
        z += -0.0139 * Q.sum_pt_top50 + 15.985
    if Q.z_top50_slots < 0.972:
        z += 21.5 * Q.z_top50_slots - 20.898
    if Q.mass_over_sum_pt < 0.091 and Q.psi_0p2 > 0.914:
        z += -349.0 * (0.091 - Q.mass_over_sum_pt) * (Q.psi_0p2 - 0.914)
    if Q.psi_0p3 > 0.993 and Q.sd_rg < 0.185:
        z += -411.0 * (Q.psi_0p3 - 0.993) * (0.185 - Q.sd_rg)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.25
    if Q.LHA < 0.102:
        z += 11.7 * Q.LHA - 4.7242
    if 0.102 <= Q.LHA < 0.178:
        z += 18.2 * Q.LHA - 5.3872
    if 0.178 <= Q.LHA < 0.296:
        z += 27.89 * Q.LHA - 7.11202
    if Q.LHA >= 0.296:
        z += 9.69 * Q.LHA - 1.72482
    if 0.0299 <= Q.e2 < 0.0423:
        z += -26.1 * Q.e2 + 0.78039
    if Q.e2 >= 0.0423:
        z += -37.1 * Q.e2 + 1.24569
    if Q.e2_sq < 0.0013:
        z += -361.0 * Q.e2_sq + 0.4693
    if Q.e3 >= 4.09e-06:
        z += 2010.0 * Q.e3 - 0.0082209
    if Q.girth < 0.0827:
        z += -86.1 * Q.girth + 7.12047
    if Q.girth >= 0.0857:
        z += -42.9 * Q.girth + 3.67653
    if Q.girth2_top10 < 0.00282:
        z += 21.1 * Q.girth2_top10 + 0.254967
    if 0.00282 <= Q.girth2_top10 < 0.00895:
        z += -51.3 * Q.girth2_top10 + 0.459135
    if Q.lam2 < 0.00229:
        z += -213.0 * Q.lam2 + 0.48777
    if Q.log_sum_pt < 6.91:
        z += -2.72 * Q.log_sum_pt + 20.0624
    if 6.91 <= Q.log_sum_pt < 6.97:
        z += -15.52 * Q.log_sum_pt + 108.5104
    if 6.97 <= Q.log_sum_pt < 7.01:
        z += -8.4 * Q.log_sum_pt + 58.884
    if Q.mass_top30 >= 76.7:
        z += 0.0128 * Q.mass_top30 - 0.98176
    if Q.mass_top50 < 86.8:
        z += 0.0127 * Q.mass_top50 - 1.10236
    if Q.n_dr_0p1_0p2 >= 16.7:
        z += 0.0164 * Q.n_dr_0p1_0p2 - 0.27388
    if Q.n_particles < 34.6:
        z += -0.0329 * Q.n_particles + 1.18111
    if 34.6 <= Q.n_particles < 35.9:
        z += -0.0605 * Q.n_particles + 2.13607
    if Q.n_particles >= 35.9:
        z += -0.0276 * Q.n_particles + 0.95496
    if Q.n_pt_above_1 < 54.7:
        z += 0.0162 * Q.n_pt_above_1 - 0.88614
    if Q.psi_0p2 >= 0.996:
        z += -58.8 * Q.psi_0p2 + 58.5648
    if Q.psi_0p3 >= 0.99:
        z += -33.2 * Q.psi_0p3 + 32.868
    if 52.3 <= Q.sd_mass < 86.7:
        z += -0.0193 * Q.sd_mass + 1.00939
    if Q.sd_mass >= 86.7:
        z += -0.0006 * Q.sd_mass - 0.6119
    if Q.sd_rg < 0.166:
        z += -0.06 * Q.sd_rg + 0.20964
    if 0.166 <= Q.sd_rg < 0.205:
        z += -5.12 * Q.sd_rg + 1.0496
    if 0.217 <= Q.sd_rg < 0.279:
        z += 14.4 * Q.sd_rg - 3.1248
    if Q.sd_rg >= 0.279:
        z += 0.7 * Q.sd_rg + 0.6975
    if Q.sum_pt_top40 < 893.0:
        z += -0.0043 * Q.sum_pt_top40 + 3.8399
    if Q.sum_pt_top50 < 1060.0:
        z += 0.00948 * Q.sum_pt_top50 - 10.0488
    if Q.width < 0.00832:
        z += 135.0 * Q.width - 1.1232
    if Q.z_dr_0_0p05 >= 0.907:
        z += -4.8 * Q.z_dr_0_0p05 + 4.3536
    if Q.z_dr_0p1_0p2 < 0.441:
        z += -1.34 * Q.z_dr_0p1_0p2 + 0.59094
    if Q.z_top50_slots >= 0.973:
        z += -13.9 * Q.z_top50_slots + 13.5247
    if Q.girth2_top20 < 0.00303 and Q.sum_pt > 1120.0:
        z += 0.262 * (0.00303 - Q.girth2_top20) * (Q.sum_pt - 1120.0)
    if Q.sj2_dr > 0.225 and Q.C2_b2 < 0.0393:
        z += 137.0 * (Q.sj2_dr - 0.225) * (0.0393 - Q.C2_b2)
    if Q.z_dr_0_0p05 < 0.897 and Q.sum_pt < 1000.0:
        z += 0.00343 * (0.897 - Q.z_dr_0_0p05) * (1000.0 - Q.sum_pt)
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
