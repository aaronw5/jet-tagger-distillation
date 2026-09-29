"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network (step 4; all observables, tuned for agreement), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 79.2% (the network: 81.1%); same class as the network for 91.1% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_7                   pT of particle 7 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_z                pT share of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.soft10_abseta          |Δη| of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_9                   ΔR of particle 9 from the jet axis
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
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
        z_top5=sum(zs[:5]),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_7=pt[7],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft5_z=softp(5, 'z'),
        soft9_z=softp(9, 'z'),
        z_top2_slots=sum(pt[:2]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        soft10_abseta=softp(10, 'abseta'),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_9=dr[9] if pt[9] > 0 else 0.0,
        eta_1=eta[1],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
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
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = 0.988
    if Q.e2_sq < 0.0062:
        z += 351.0 * Q.e2_sq - 2.1762
    if Q.girth2_top30 < 0.0063:
        z += 115.0 * Q.girth2_top30 - 0.7245
    if Q.lam2 < 0.00069:
        z += -381.0 * Q.lam2 + 0.26289
    if 79.0 <= Q.mass < 90.0:
        z += -0.119 * Q.mass + 9.401
    if Q.mass >= 90.0:
        z += 0.054 * Q.mass - 6.169
    if Q.mass_over_sum_pt_sq < 0.0076:
        z += -488.0 * Q.mass_over_sum_pt_sq + 3.7088
    if Q.mass_top50 >= 82.0:
        z += -0.0516 * Q.mass_top50 + 4.2312
    if Q.n_dr_0p2_0p4 < 14.0:
        z += -0.0443 * Q.n_dr_0p2_0p4 + 0.6202
    if Q.psi_0p3 >= 0.99:
        z += 25.6 * Q.psi_0p3 - 25.344
    if Q.sum_pt < 980.0:
        z += 0.00623 * Q.sum_pt - 6.1054
    if Q.tau1 < 0.07:
        z += 16.5 * Q.tau1 - 1.155
    if Q.z_dr_0p2_0p4 < 0.084:
        z += 4.36 * Q.z_dr_0p2_0p4 - 0.36624
    z += 10.3 * Q.z_top50_slots - 10.3
    if Q.log_sum_pt < 7.0 and Q.sum_pt_top50 > 960.0:
        z += 0.109 * (7.0 - Q.log_sum_pt) * (Q.sum_pt_top50 - 960.0)
    return max(0.0, z)


def neuron_1(Q):
    z = 4.86
    if Q.D3 < 0.12:
        z += -2.46 * Q.D3 + 0.2952
    if Q.girth2_top15 < 0.00096:
        z += -755.0 * Q.girth2_top15 + 0.7248
    if Q.girth2_top5 < 0.00065:
        z += -1510.0 * Q.girth2_top5 + 0.9815
    if Q.lam1 < 0.0044:
        z += -307.0 * Q.lam1 + 1.3508
    if Q.log_sum_pt < 6.9:
        z += 8.53 * Q.log_sum_pt - 60.563
    if 6.9 <= Q.log_sum_pt < 7.0:
        z += 31.33 * Q.log_sum_pt - 217.883
    if 7.0 <= Q.log_sum_pt < 7.1:
        z += 22.17 * Q.log_sum_pt - 153.763
    if Q.log_sum_pt >= 7.1:
        z += 13.64 * Q.log_sum_pt - 93.2
    if Q.mass < 120.0:
        z += 0.0241 * Q.mass - 2.892
    if Q.n_dr_0p1_0p2 < 7.4:
        z += 0.0883 * Q.n_dr_0p1_0p2 - 0.65342
    if Q.n_dr_0p2_0p4 < 7.5:
        z += 0.182 * Q.n_dr_0p2_0p4 - 1.365
    if Q.psi_0p1 < 0.34:
        z += -2.31 * Q.psi_0p1 + 0.7854
    if Q.pt_9 < 30.0:
        z += 0.0564 * Q.pt_9 - 1.692
    if Q.sj2_mass1 < 30.0:
        z += 0.0359 * Q.sj2_mass1 - 1.077
    if Q.sj3_mass1 < 31.0:
        z += 0.0199 * Q.sj3_mass1 - 0.6169
    if Q.soft1_pt < 1.5:
        z += 0.143 * Q.soft1_pt - 1.1265
    if 1.5 <= Q.soft1_pt < 2.3:
        z += 1.14 * Q.soft1_pt - 2.622
    if Q.sum_pt < 1000.0:
        z += -0.0191 * Q.sum_pt + 19.1
    if Q.sum_pt_top2 < 710.0:
        z += -0.0039 * Q.sum_pt_top2 + 2.769
    if Q.sum_pt_top30 >= 1200.0:
        z += 0.00637 * Q.sum_pt_top30 - 7.644
    if Q.tau1 < 0.19:
        z += -9.06 * Q.tau1 + 1.7214
    if Q.z_top50_slots >= 0.96:
        z += -51.4 * Q.z_top50_slots + 49.344
    if Q.girth2_top20 < 0.0074 and Q.eta_1 > -0.047:
        z += 1170.0 * (0.0074 - Q.girth2_top20) * (Q.eta_1 - -0.047)
    if Q.log_sum_pt < 7.2 and Q.pt1_dr01 < 6.1:
        z += -0.204 * (7.2 - Q.log_sum_pt) * (6.1 - Q.pt1_dr01)
    if Q.mass < 120.0 and Q.dr_2 < 0.038:
        z += 0.289 * (120.0 - Q.mass) * (0.038 - Q.dr_2)
    if Q.mass_top20 < 49.0 and Q.n_real_top40 > 29.0:
        z += 0.00407 * (49.0 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.12:
        z += 0.378 * (Q.n_particles - 38.0) * (0.12 - Q.dr_0)
    if Q.n_particles > 39.0 and Q.dr_1 < 0.16:
        z += 0.289 * (Q.n_particles - 39.0) * (0.16 - Q.dr_1)
    if Q.n_particles > 38.0 and Q.dr_9 < 0.17:
        z += 0.151 * (Q.n_particles - 38.0) * (0.17 - Q.dr_9)
    if Q.n_particles > 37.0 and Q.mass_top15 < 63.0:
        z += 0.00108 * (Q.n_particles - 37.0) * (63.0 - Q.mass_top15)
    if Q.n_particles > 39.0 and Q.sj3_dr12 < 0.26:
        z += 0.131 * (Q.n_particles - 39.0) * (0.26 - Q.sj3_dr12)
    if Q.sj3_mass1 < 33.0 and Q.sj3_mass2 < 20.0:
        z += -0.00313 * (33.0 - Q.sj3_mass1) * (20.0 - Q.sj3_mass2)
    if Q.soft1_pt < 1.5 and Q.soft10_abseta < 0.13:
        z += -2.6 * (1.5 - Q.soft1_pt) * (0.13 - Q.soft10_abseta)
    if Q.z_top20_slots > 0.89 and Q.dr_2 < 0.04:
        z += -400.0 * (Q.z_top20_slots - 0.89) * (0.04 - Q.dr_2)
    if Q.z_top30_slots > 0.92 and Q.C2 < 0.07:
        z += 221.0 * (Q.z_top30_slots - 0.92) * (0.07 - Q.C2)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.12
    return max(0.0, z)


def neuron_3(Q):
    z = -0.0507
    if Q.n_dr_0p1_0p2 < 9.4:
        z += -0.0567 * Q.n_dr_0p1_0p2 + 0.53298
    if Q.n_particles < 47.0:
        z += -0.0523 * Q.n_particles + 2.4581
    if Q.sj2_mass1 < 27.0:
        z += -0.0225 * Q.sj2_mass1 + 0.6075
    if Q.tau21 < 0.34:
        z += 3.37 * Q.tau21 - 1.1458
    if Q.zdr_0 < 0.003:
        z += 206.0 * Q.zdr_0 - 0.618
    if Q.girth2 < 0.009 and Q.sj3_mass1 < 30.0:
        z += 3.11 * (0.009 - Q.girth2) * (30.0 - Q.sj3_mass1)
    if Q.mass_top50 < 78.0 and Q.mass_top3 > 9.1:
        z += -0.00439 * (78.0 - Q.mass_top50) * (Q.mass_top3 - 9.1)
    if Q.n_dr_0p2_0p4 < 9.4 and Q.sum_pt_top40 > 1000.0:
        z += 0.000179 * (9.4 - Q.n_dr_0p2_0p4) * (Q.sum_pt_top40 - 1000.0)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.z_top50_slots > 0.97:
        z += 4.8 * (6.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.97)
    return max(0.0, z)


def neuron_4(Q):
    z = 3.48
    if Q.e2_sq >= 0.011:
        z += 40.2 * Q.e2_sq - 0.4422
    if Q.girth2_top15 < 0.0055:
        z += 109.0 * Q.girth2_top15 - 0.5995
    if Q.lam2 >= 0.0012:
        z += 92.6 * Q.lam2 - 0.11112
    if Q.mass < 85.0:
        z += 0.0611 * Q.mass - 4.322
    if 85.0 <= Q.mass < 120.0:
        z += -0.0249 * Q.mass + 2.988
    if Q.n_dr_0p1_0p2 < 28.0:
        z += -0.0142 * Q.n_dr_0p1_0p2 + 0.3976
    if Q.n_dr_0p2_0p4 < 23.0:
        z += -0.0297 * Q.n_dr_0p2_0p4 + 0.6831
    if Q.n_particles >= 25.0:
        z += -0.024 * Q.n_particles + 0.6
    if Q.sj3_pair_mass_min >= 33.0:
        z += 0.0176 * Q.sj3_pair_mass_min - 0.5808
    if Q.sum_pt < 1100.0:
        z += 0.00245 * Q.sum_pt - 2.695
    z += -0.00271 * Q.sum_pt_top30
    if Q.mass < 80.0 and Q.D2_b2 < 4.1:
        z += -0.0218 * (80.0 - Q.mass) * (4.1 - Q.D2_b2)
    if Q.mass < 100.0 and Q.D2_b2 < 4.1:
        z += 0.0159 * (100.0 - Q.mass) * (4.1 - Q.D2_b2)
    if Q.mass_top40 < 87.0 and Q.D2_b2 < 3.3:
        z += -0.026 * (87.0 - Q.mass_top40) * (3.3 - Q.D2_b2)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.sj3_dr13 > 0.14:
        z += 0.101 * (15.0 - Q.n_dr_0p2_0p4) * (Q.sj3_dr13 - 0.14)
    if Q.n_dr_0p2_0p4 < 20.0 and Q.z_top50_slots < 1.0:
        z += -1.14 * (20.0 - Q.n_dr_0p2_0p4) * (1.0 - Q.z_top50_slots)
    if Q.n_particles > 14.0 and Q.n_dr_0_0p05 > 9.6:
        z += 0.000594 * (Q.n_particles - 14.0) * (Q.n_dr_0_0p05 - 9.6)
    if Q.n_particles > 13.0 and Q.pt_7 < 34.0:
        z += -0.0005 * (Q.n_particles - 13.0) * (34.0 - Q.pt_7)
    if Q.n_particles > 23.0 and Q.soft1_pt < 2.4:
        z += 0.00678 * (Q.n_particles - 23.0) * (2.4 - Q.soft1_pt)
    if Q.sj2_dr > 0.23 and Q.D2_b2 < 20.0:
        z += -0.2 * (Q.sj2_dr - 0.23) * (20.0 - Q.D2_b2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.566
    if Q.girth2_top30 >= 0.008:
        z += -65.1 * Q.girth2_top30 + 0.5208
    if Q.log_sum_pt >= 6.9:
        z += -32.2 * Q.log_sum_pt + 222.18
    if 56.0 <= Q.mass < 93.0:
        z += 0.0147 * Q.mass - 0.8232
    if Q.mass >= 93.0:
        z += -0.0022 * Q.mass + 0.7485
    if Q.max_dr >= 0.19:
        z += -1.33 * Q.max_dr + 0.2527
    if Q.n_dr_0p2_0p4 < 13.0:
        z += -0.0524 * Q.n_dr_0p2_0p4 + 0.6812
    if Q.pt_entropy < 2.8:
        z += -0.771 * Q.pt_entropy + 2.1588
    if Q.sum_pt_top50 >= 930.0:
        z += 0.0234 * Q.sum_pt_top50 - 21.762
    if Q.n_particles < 75.0 and Q.D2 < 2.9:
        z += -0.00642 * (75.0 - Q.n_particles) * (2.9 - Q.D2)
    if Q.z_top30_slots > 0.9 and Q.sj3_mass1 > 8.0:
        z += -0.267 * (Q.z_top30_slots - 0.9) * (Q.sj3_mass1 - 8.0)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.808
    if Q.mass < 93.0:
        z += 0.007 * Q.mass - 1.547
    if 93.0 <= Q.mass < 100.0:
        z += 0.128 * Q.mass - 12.8
    if Q.mass_top15 < 94.0:
        z += -0.00781 * Q.mass_top15 + 0.73414
    if Q.mass_top50 < 75.0:
        z += -0.048 * Q.mass_top50 + 3.6
    if Q.tau1 < 0.072:
        z += -18.2 * Q.tau1 + 1.3104
    if Q.girth2_top30 > 0.0081 and Q.D2_b2 > 1.3:
        z += -19.0 * (Q.girth2_top30 - 0.0081) * (Q.D2_b2 - 1.3)
    if Q.girth2_top30 > 0.0085 and Q.n_real_top50 < 48.0:
        z += 19.0 * (Q.girth2_top30 - 0.0085) * (48.0 - Q.n_real_top50)
    if Q.mass_over_sum_pt > 0.15 and Q.D2_b2 < 15.0:
        z += -0.505 * (Q.mass_over_sum_pt - 0.15) * (15.0 - Q.D2_b2)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.232
    if Q.girth2 < 0.0083:
        z += 54.0 * Q.girth2 - 0.0306
    if 0.0083 <= Q.girth2 < 0.0099:
        z += -261.0 * Q.girth2 + 2.5839
    if Q.mass < 100.0:
        z += 0.0299 * Q.mass - 2.114
    if 100.0 <= Q.mass < 120.0:
        z += -0.0438 * Q.mass + 5.256
    if 74.0 <= Q.mass_top20 < 86.0:
        z += 0.041 * Q.mass_top20 - 3.034
    if Q.mass_top20 >= 86.0:
        z += -0.0012 * Q.mass_top20 + 0.5952
    if Q.n_dr_0p2_0p4 < 6.5:
        z += -0.164 * Q.n_dr_0p2_0p4 + 1.066
    if Q.tau1 < 0.096:
        z += -10.7 * Q.tau1 + 0.463
    if 0.096 <= Q.tau1 < 0.11:
        z += 40.3 * Q.tau1 - 4.433
    if Q.lam1 < 0.0078 and Q.zdr_0 > 0.015:
        z += -21200.0 * (0.0078 - Q.lam1) * (Q.zdr_0 - 0.015)
    if Q.mass < 83.0 and Q.psi_0p3 > 0.98:
        z += 12.1 * (83.0 - Q.mass) * (Q.psi_0p3 - 0.98)
    if Q.mass < 92.0 and Q.psi_0p3 > 0.98:
        z += -22.8 * (92.0 - Q.mass) * (Q.psi_0p3 - 0.98)
    if Q.mass < 100.0 and Q.psi_0p3 > 0.98:
        z += 12.0 * (100.0 - Q.mass) * (Q.psi_0p3 - 0.98)
    if Q.mass < 92.0 and Q.z_dr_0_0p05 < 0.47:
        z += -0.0856 * (92.0 - Q.mass) * (0.47 - Q.z_dr_0_0p05)
    return max(0.0, z)


def neuron_8(Q):
    z = 1.34
    if Q.girth2 < 0.0079:
        z += -643.0 * Q.girth2 + 5.0797
    if Q.girth2_top20 >= 0.0081:
        z += 142.0 * Q.girth2_top20 - 1.1502
    if 69.0 <= Q.mass < 120.0:
        z += 0.034 * Q.mass - 2.346
    if Q.mass >= 120.0:
        z += -0.0055 * Q.mass + 2.394
    if Q.n_dr_0p2_0p4 >= 3.0:
        z += 0.0595 * Q.n_dr_0p2_0p4 - 0.1785
    if Q.psi_0p3 >= 0.99:
        z += -38.0 * Q.psi_0p3 + 37.62
    if Q.sum_pt < 1000.0:
        z += -0.0259 * Q.sum_pt + 25.9
    if Q.width < 0.0096:
        z += 705.0 * Q.width - 6.768
    if Q.z_top50_slots >= 0.97:
        z += -19.5 * Q.z_top50_slots + 18.915
    if Q.tau2 < 0.084 and Q.sj3_mass1 > 13.0:
        z += 0.428 * (0.084 - Q.tau2) * (Q.sj3_mass1 - 13.0)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.66
    if Q.girth2_top40 < 0.0061:
        z += -309.0 * Q.girth2_top40 + 1.8849
    if Q.mass < 65.0:
        z += -0.0534 * Q.mass + 7.2539
    if 65.0 <= Q.mass < 87.0:
        z += -0.0794 * Q.mass + 8.9439
    if 87.0 <= Q.mass < 150.0:
        z += 0.0043 * Q.mass + 1.662
    if 150.0 <= Q.mass < 180.0:
        z += -0.0769 * Q.mass + 13.842
    if Q.mass_top30 < 76.0:
        z += -0.0212 * Q.mass_top30 + 1.6112
    if Q.sum_pt_top40 < 950.0:
        z += -0.0119 * Q.sum_pt_top40 + 11.305
    if Q.z_top5 >= 0.82:
        z += 7.63 * Q.z_top5 - 6.2566
    if Q.mass_top40 < 120.0 and Q.n_dr_0p2_0p4 > 10.0:
        z += 0.00159 * (120.0 - Q.mass_top40) * (Q.n_dr_0p2_0p4 - 10.0)
    if Q.sum_pt_top40 < 950.0 and Q.lam2 < 0.0018:
        z += -7.39 * (950.0 - Q.sum_pt_top40) * (0.0018 - Q.lam2)
    return max(0.0, z)


def neuron_10(Q):
    z = -0.195
    if Q.D2 < 3.4:
        z += -0.505 * Q.D2 + 1.717
    if Q.e2 >= 0.012:
        z += 46.9 * Q.e2 - 0.5628
    if Q.mass < 79.0:
        z += -0.0437 * Q.mass + 3.3872
    if 79.0 <= Q.mass < 88.0:
        z += -0.0222 * Q.mass + 1.6887
    if 88.0 <= Q.mass < 100.0:
        z += 0.0597 * Q.mass - 5.5185
    if 100.0 <= Q.mass < 160.0:
        z += 0.0215 * Q.mass - 1.6985
    if Q.mass >= 160.0:
        z += -0.1065 * Q.mass + 18.7815
    if Q.mass_over_sum_pt >= 0.16:
        z += -36.6 * Q.mass_over_sum_pt + 5.856
    if Q.mass_top5 >= 14.0:
        z += 0.0107 * Q.mass_top5 - 0.1498
    if Q.mass_top50 >= 180.0:
        z += 0.0639 * Q.mass_top50 - 11.502
    if Q.pt_dispersion >= 0.26:
        z += -2.32 * Q.pt_dispersion + 0.6032
    if Q.sj2_mass1 < 69.0:
        z += 0.0111 * Q.sj2_mass1 - 0.7659
    if Q.sj3_dr_min >= 0.11:
        z += 2.7 * Q.sj3_dr_min - 0.297
    if Q.sum_pt_top10 >= 930.0:
        z += -0.00495 * Q.sum_pt_top10 + 4.6035
    if Q.sum_pt_top15 < 960.0:
        z += -0.00232 * Q.sum_pt_top15 + 2.2272
    if Q.sum_pt_top50 < 950.0:
        z += 0.00509 * Q.sum_pt_top50 - 4.8355
    if Q.tau1 < 0.049:
        z += 67.5 * Q.tau1 - 3.3075
    if Q.D2 < 3.3 and Q.sj2_dr > 0.2:
        z += -3.23 * (3.3 - Q.D2) * (Q.sj2_dr - 0.2)
    if Q.girth2_top10 < 0.019 and Q.max_dr < 0.44:
        z += -391.0 * (0.019 - Q.girth2_top10) * (0.44 - Q.max_dr)
    if Q.mass > 76.0 and Q.lam2 < 0.00061:
        z += -51.7 * (Q.mass - 76.0) * (0.00061 - Q.lam2)
    if Q.mass < 100.0 and Q.pt1_dr01 > 9.9:
        z += 0.00195 * (100.0 - Q.mass) * (Q.pt1_dr01 - 9.9)
    if Q.mass > 160.0 and Q.soft5_z > 0.0015:
        z += 40.3 * (Q.mass - 160.0) * (Q.soft5_z - 0.0015)
    if Q.mass_top50 > 130.0 and Q.soft5_z > 0.0017:
        z += -34.2 * (Q.mass_top50 - 130.0) * (Q.soft5_z - 0.0017)
    if Q.z_top2_slots < 0.58 and Q.soft9_z < 0.0031:
        z += 866.0 * (0.58 - Q.z_top2_slots) * (0.0031 - Q.soft9_z)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.094
    if Q.mass < 81.0:
        z += 0.0127 * Q.mass - 0.3105
    if 81.0 <= Q.mass < 95.0:
        z += -0.0513 * Q.mass + 4.8735
    if Q.mass < 120.0 and Q.planar_flow < 0.41:
        z += 0.0599 * (120.0 - Q.mass) * (0.41 - Q.planar_flow)
    if Q.mass_over_sum_pt_sq < 0.0097 and Q.z_dr_0p1_0p2 > 0.15:
        z += 1490.0 * (0.0097 - Q.mass_over_sum_pt_sq) * (Q.z_dr_0p1_0p2 - 0.15)
    if Q.n_dr_0p2_0p4 < 9.2 and Q.girth2 < 0.0064:
        z += -51.5 * (9.2 - Q.n_dr_0p2_0p4) * (0.0064 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 9.1 and Q.n_dr_0p1_0p2 < 27.0:
        z += 0.00966 * (9.1 - Q.n_dr_0p2_0p4) * (27.0 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.187
    if Q.mass < 81.0:
        z += -0.0908 * Q.mass + 7.3548
    if Q.sd_mass >= 130.0:
        z += 0.0377 * Q.sd_mass - 4.901
    if Q.log_sum_pt < 6.8 and Q.lam2 < 0.0036:
        z += -2110.0 * (6.8 - Q.log_sum_pt) * (0.0036 - Q.lam2)
    if Q.mass < 84.0 and Q.psi_0p3 < 1.0:
        z += -1.77 * (84.0 - Q.mass) * (1.0 - Q.psi_0p3)
    if Q.mass < 89.0 and Q.tau21 < 0.53:
        z += 0.0896 * (89.0 - Q.mass) * (0.53 - Q.tau21)
    if Q.sj3_pair_mass_max > 130.0 and Q.sj3_pair_mass_min < 88.0:
        z += 0.000522 * (Q.sj3_pair_mass_max - 130.0) * (88.0 - Q.sj3_pair_mass_min)
    if Q.z_dr_0p1_0p2 > 0.6 and Q.n_dr_0p4_up > -0.26:
        z += 2.85 * (Q.z_dr_0p1_0p2 - 0.6) * (Q.n_dr_0p4_up - -0.26)
    return max(0.0, z)


def neuron_13(Q):
    z = 3.32
    if Q.log_sum_pt < 6.8:
        z += -18.7 * Q.log_sum_pt + 124.95
    if 6.8 <= Q.log_sum_pt < 6.9:
        z += 22.1 * Q.log_sum_pt - 152.49
    if Q.mass >= 150.0:
        z += -0.126 * Q.mass + 18.9
    if Q.n_pt_above_1 >= 53.0:
        z += 0.087 * Q.n_pt_above_1 - 4.611
    if Q.sum_pt < 1100.0:
        z += 0.015 * Q.sum_pt - 16.5
    if Q.sum_pt_top30 >= 1100.0:
        z += 0.00468 * Q.sum_pt_top30 - 5.148
    if Q.girth2_top50 < 0.032 and Q.psi_0p3 > 0.96:
        z += 856.0 * (0.032 - Q.girth2_top50) * (Q.psi_0p3 - 0.96)
    if Q.log_sum_pt > 7.1 and Q.sd_zg > 0.45:
        z += 370.0 * (Q.log_sum_pt - 7.1) * (Q.sd_zg - 0.45)
    if Q.mass > 140.0 and Q.D2 > -0.44:
        z += -0.0231 * (Q.mass - 140.0) * (Q.D2 - -0.44)
    if Q.sum_pt < 1000.0 and Q.D2 < 3.0:
        z += -0.00652 * (1000.0 - Q.sum_pt) * (3.0 - Q.D2)
    if Q.sum_pt_top30 > 1100.0 and Q.mratio_min_012 < 0.012:
        z += 1.72 * (Q.sum_pt_top30 - 1100.0) * (0.012 - Q.mratio_min_012)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.952
    if Q.mass < 91.0:
        z += 0.0484 * Q.mass - 4.4044
    if Q.max_pair_mass >= 19.0:
        z += -0.0209 * Q.max_pair_mass + 0.3971
    if Q.tau21_b2 < 0.39:
        z += -1.63 * Q.tau21_b2 + 0.6357
    if Q.mass_over_sum_pt < 0.089 and Q.psi_0p2 > 0.89:
        z += -224.0 * (0.089 - Q.mass_over_sum_pt) * (Q.psi_0p2 - 0.89)
    if Q.tau21_b2 < 0.38 and Q.girth2_top40 < 0.008:
        z += -906.0 * (0.38 - Q.tau21_b2) * (0.008 - Q.girth2_top40)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.0613
    if Q.D2 < 2.3:
        z += 0.312 * Q.D2 - 0.7176
    if Q.e2 >= 0.03:
        z += -31.7 * Q.e2 + 0.951
    if Q.girth2_top10 < 0.0082:
        z += -89.9 * Q.girth2_top10 + 0.73718
    if Q.log_sum_pt < 7.0:
        z += -4.69 * Q.log_sum_pt + 32.83
    if Q.psi_0p1 >= 0.93:
        z += -14.4 * Q.psi_0p1 + 13.392
    if Q.sum_pt < 1000.0:
        z += 0.0132 * Q.sum_pt - 13.2
    if Q.tau1 < 0.072:
        z += 28.2 * Q.tau1 - 2.0304
    if Q.z_dr_0p1_0p2 < 0.092:
        z += -7.42 * Q.z_dr_0p1_0p2 + 0.68264
    if Q.sj2_dr > 0.23 and Q.C2_b2 < 0.049:
        z += 232.0 * (Q.sj2_dr - 0.23) * (0.049 - Q.C2_b2)
    if Q.z_dr_0_0p05 < 0.98 and Q.sum_pt < 960.0:
        z += 0.0133 * (0.98 - Q.z_dr_0_0p05) * (960.0 - Q.sum_pt)
    if Q.z_dr_0p1_0p2 < 0.26 and Q.n_particles < 63.0:
        z += 0.0898 * (0.26 - Q.z_dr_0p1_0p2) * (63.0 - Q.n_particles)
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
