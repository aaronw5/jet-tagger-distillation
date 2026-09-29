"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no mass observables or exact equivalents), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.7% (the network: 81.1%); same class as the network for 90.7% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_0                   pT of particle 0 [GeV]
  Q.ptdr0_10               pT10 · ΔR(0, 10) [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_0                    pT of particle 0 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.absphi_1               |Δφ| of particle 1
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau32                  N-subjettiness τ3/τ2
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
        z_top5=sum(zs[:5]),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_0=pt[0],
        ptdr0_10=pt[10] * math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        pt_11=pt[11],
        soft1_pt=softp(1, 'pt'),
        soft6_pt=softp(6, 'pt'),
        soft9_pt=softp(9, 'pt'),
        z_0=z[0],
        soft1_z=softp(1, 'z'),
        soft5_z=softp(5, 'z'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        sd_rg=softdrop("rg"),
        sd_zg=softdrop("zg"),
        absphi_1=abs(phi[1]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        eta_1=eta[1],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    z = -1.08
    if Q.LHA < 0.31:
        z += 15.9 * Q.LHA - 4.929
    if Q.dr_0 < 0.0733:
        z += 8.61 * Q.dr_0 - 0.631113
    if Q.e3 < 2.86e-05:
        z += 28950.0 * Q.e3 - 0.61127
    if 2.86e-05 <= Q.e3 < 5.06e-05:
        z += -9850.0 * Q.e3 + 0.49841
    if Q.girth < 0.0512:
        z += -87.8 * Q.girth + 7.93376
    if 0.0512 <= Q.girth < 0.0819:
        z += -112.0 * Q.girth + 9.1728
    if Q.girth2_top15 < 0.00542:
        z += -36.0 * Q.girth2_top15 + 0.09264
    if 0.00542 <= Q.girth2_top15 < 0.00782:
        z += -343.0 * Q.girth2_top15 + 1.75658
    if 0.00782 <= Q.girth2_top15 < 0.0101:
        z += 406.0 * Q.girth2_top15 - 4.1006
    if Q.lam2 < 0.00081:
        z += -533.0 * Q.lam2 + 0.43173
    if Q.psi_0p3 >= 0.989:
        z += 31.6 * Q.psi_0p3 - 31.2524
    if Q.sj2_dr >= 0.152:
        z += -4.73 * Q.sj2_dr + 0.71896
    if Q.sj3_dr_max < 0.119:
        z += 1.37 * Q.sj3_dr_max - 0.36387
    if 0.119 <= Q.sj3_dr_max < 0.161:
        z += 15.77 * Q.sj3_dr_max - 2.07747
    if 0.161 <= Q.sj3_dr_max < 0.211:
        z += -9.23 * Q.sj3_dr_max + 1.94753
    if Q.sum_pt_top2 >= 403.0:
        z += 0.000874 * Q.sum_pt_top2 - 0.352222
    if Q.sum_pt_top40 >= 997.0:
        z += -0.00522 * Q.sum_pt_top40 + 5.20434
    if Q.tau1 < 0.0446:
        z += 61.0 * Q.tau1 - 2.7206
    if Q.girth < 0.0866 and Q.z_dr_0_0p05 < 0.494:
        z += 53.5 * (0.0866 - Q.girth) * (0.494 - Q.z_dr_0_0p05)
    if Q.n_dr_0p2_0p4 < 19.3 and Q.girth2_top10 > 0.00496:
        z += -7.51 * (19.3 - Q.n_dr_0p2_0p4) * (Q.girth2_top10 - 0.00496)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.log_sum_pt > 6.92:
        z += -1.19 * (18.0 - Q.n_dr_0p2_0p4) * (Q.log_sum_pt - 6.92)
    if Q.n_dr_0p2_0p4 < 17.8 and Q.sum_pt_top50 > 882.0:
        z += 0.00069 * (17.8 - Q.n_dr_0p2_0p4) * (Q.sum_pt_top50 - 882.0)
    if Q.n_dr_0p2_0p4 < 17.5 and Q.tau21_b2 > 0.229:
        z += -0.083 * (17.5 - Q.n_dr_0p2_0p4) * (Q.tau21_b2 - 0.229)
    if Q.psi_0p3 > 0.994 and Q.n_dr_0p1_0p2 < 24.0:
        z += 4.44 * (Q.psi_0p3 - 0.994) * (24.0 - Q.n_dr_0p1_0p2)
    if Q.sum_pt_top40 > 1030.0 and Q.sj2_dr < 0.166:
        z += 0.0435 * (Q.sum_pt_top40 - 1030.0) * (0.166 - Q.sj2_dr)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.22
    if Q.girth2_top15 < 0.000777:
        z += -1160.0 * Q.girth2_top15 + 0.90132
    if Q.lam2 < 0.00172:
        z += 466.0 * Q.lam2 - 0.80152
    if Q.log_sum_pt < 6.91:
        z += 9.31 * Q.log_sum_pt - 66.4734
    if 6.91 <= Q.log_sum_pt < 6.99:
        z += 47.51 * Q.log_sum_pt - 330.4354
    if 6.99 <= Q.log_sum_pt < 7.14:
        z += 8.71 * Q.log_sum_pt - 59.2234
    if Q.log_sum_pt >= 7.14:
        z += -0.6 * Q.log_sum_pt + 7.25
    if 14.8 <= Q.n_dr_0_0p05 < 30.2:
        z += 0.0615 * Q.n_dr_0_0p05 - 0.9102
    if Q.n_dr_0_0p05 >= 30.2:
        z += -0.0287 * Q.n_dr_0_0p05 + 1.81384
    if Q.n_dr_0p1_0p2 < 6.86:
        z += 0.1 * Q.n_dr_0p1_0p2 - 0.686
    if Q.n_dr_0p2_0p4 < 6.87:
        z += 0.125 * Q.n_dr_0p2_0p4 - 0.85875
    if Q.n_particles >= 39.4:
        z += 0.0367 * Q.n_particles - 1.44598
    if Q.pt_0 < 345.0:
        z += -0.00179 * Q.pt_0 + 0.61755
    if Q.pt_11 < 29.5:
        z += 0.0261 * Q.pt_11 - 0.76995
    if Q.soft9_pt < 2.43:
        z += -0.211 * Q.soft9_pt + 0.51273
    if Q.sum_pt >= 972.0:
        z += 0.0239 * Q.sum_pt - 23.2308
    if Q.sum_pt_top50 < 950.0:
        z += -0.0175 * Q.sum_pt_top50 + 19.075
    if 950.0 <= Q.sum_pt_top50 < 1090.0:
        z += -0.0382 * Q.sum_pt_top50 + 38.74
    if Q.sum_pt_top50 >= 1090.0:
        z += -0.0207 * Q.sum_pt_top50 + 19.665
    if Q.tau1 < 0.0757:
        z += -29.5 * Q.tau1 + 2.23315
    if Q.tau21 < 0.202:
        z += -11.3 * Q.tau21 + 2.2826
    if Q.M3 < 0.0347 and Q.psi_0p3 > 0.941:
        z += -372.0 * (0.0347 - Q.M3) * (Q.psi_0p3 - 0.941)
    if Q.n_particles > 38.6 and Q.soft1_z < 0.00227:
        z += -30.0 * (Q.n_particles - 38.6) * (0.00227 - Q.soft1_z)
    if Q.n_particles > 37.3 and Q.tau32 > 0.337:
        z += 0.121 * (Q.n_particles - 37.3) * (Q.tau32 - 0.337)
    if Q.n_particles > 36.6 and Q.tau43 < 0.966:
        z += -0.137 * (Q.n_particles - 36.6) * (0.966 - Q.tau43)
    if Q.n_particles > 38.8 and Q.zdr_0 < 0.013:
        z += 2.95 * (Q.n_particles - 38.8) * (0.013 - Q.zdr_0)
    if Q.z_dr_0_0p05 > 0.783 and Q.n_dr_0p05_0p1 < 8.96:
        z += -0.772 * (Q.z_dr_0_0p05 - 0.783) * (8.96 - Q.n_dr_0p05_0p1)
    if Q.z_top5 > 0.595 and Q.pt1_dr01 < 25.1:
        z += -0.144 * (Q.z_top5 - 0.595) * (25.1 - Q.pt1_dr01)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.134
    if Q.log_sum_pt > 6.91 and Q.dr_max_012 < 0.192:
        z += 28.1 * (Q.log_sum_pt - 6.91) * (0.192 - Q.dr_max_012)
    if Q.sum_pt_top20 > 1140.0 and Q.dr_max_012 > 0.122:
        z += -0.196 * (Q.sum_pt_top20 - 1140.0) * (Q.dr_max_012 - 0.122)
    if Q.sum_pt_top20 > 1130.0 and Q.eta_1 > 0.0876:
        z += -2.01 * (Q.sum_pt_top20 - 1130.0) * (Q.eta_1 - 0.0876)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.0209
    if Q.n_dr_0p2_0p4 < 4.01:
        z += -0.126 * Q.n_dr_0p2_0p4 + 0.50526
    if Q.tau21 < 0.338:
        z += 4.08 * Q.tau21 - 1.37904
    if Q.tau4 < 0.0171:
        z += -137.0 * Q.tau4 + 2.3427
    return max(0.0, z)


def neuron_4(Q):
    z = 0.931
    if Q.e2 < 0.0471:
        z += 24.9 * Q.e2 - 1.17279
    if Q.girth < 0.0562:
        z += 33.6 * Q.girth - 1.88832
    if Q.girth2_top15 < 0.00214:
        z += -106.0 * Q.girth2_top15 + 0.31314
    if 0.00214 <= Q.girth2_top15 < 0.00603:
        z += 198.0 * Q.girth2_top15 - 0.33742
    if 0.00603 <= Q.girth2_top15 < 0.0104:
        z += -196.0 * Q.girth2_top15 + 2.0384
    if Q.n_dr_0p2_0p4 >= 24.8:
        z += 0.0438 * Q.n_dr_0p2_0p4 - 1.08624
    if Q.n_particles >= 46.3:
        z += -0.0383 * Q.n_particles + 1.77329
    if Q.n_real_top50 < 46.3:
        z += -0.0353 * Q.n_real_top50 + 1.63439
    if 0.117 <= Q.sj2_dr < 0.182:
        z += 8.6 * Q.sj2_dr - 1.0062
    if Q.sj2_dr >= 0.182:
        z += -1.8 * Q.sj2_dr + 0.8866
    if Q.sum_pt_top30 < 1010.0:
        z += -0.00756 * Q.sum_pt_top30 + 7.6356
    if Q.sum_pt_top40 < 1040.0:
        z += 0.0117 * Q.sum_pt_top40 - 12.168
    if Q.z_dr_0p2_0p4 < 0.0557:
        z += 10.5 * Q.z_dr_0p2_0p4 - 0.58485
    if Q.girth < 0.0587 and Q.psi_0p3 > 0.997:
        z += -11600.0 * (0.0587 - Q.girth) * (Q.psi_0p3 - 0.997)
    if Q.girth < 0.0969 and Q.psi_0p3 > 0.997:
        z += 5750.0 * (0.0969 - Q.girth) * (Q.psi_0p3 - 0.997)
    if Q.girth2_top15 < 0.00851 and Q.sum_pt_top40 > 1100.0:
        z += 0.271 * (0.00851 - Q.girth2_top15) * (Q.sum_pt_top40 - 1100.0)
    if Q.n_particles > 27.3 and Q.max_dr < 0.286:
        z += 0.212 * (Q.n_particles - 27.3) * (0.286 - Q.max_dr)
    if Q.n_particles > 28.8 and Q.sj3_dr_max > 0.367:
        z += 0.152 * (Q.n_particles - 28.8) * (Q.sj3_dr_max - 0.367)
    if Q.n_particles > 28.4 and Q.soft1_pt > 0.363:
        z += -0.00829 * (Q.n_particles - 28.4) * (Q.soft1_pt - 0.363)
    if Q.n_particles > 29.3 and Q.tau21 > 0.11:
        z += 0.0365 * (Q.n_particles - 29.3) * (Q.tau21 - 0.11)
    if Q.pt_entropy < 3.33 and Q.zdr_0 > -0.00141:
        z += -11.1 * (3.33 - Q.pt_entropy) * (Q.zdr_0 - -0.00141)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.29
    if Q.log_sum_pt < 7.12:
        z += -3.48 * Q.log_sum_pt + 24.7776
    if Q.n_dr_0p2_0p4 < 11.4:
        z += -0.0575 * Q.n_dr_0p2_0p4 + 0.6555
    if Q.n_particles < 41.2:
        z += -0.0637 * Q.n_particles + 2.62444
    if Q.n_particles >= 43.7:
        z += -0.0349 * Q.n_particles + 1.52513
    if Q.sum_pt < 1000.0:
        z += 0.0491 * Q.sum_pt - 49.1
    if Q.sum_pt_top30 < 903.0:
        z += -0.0146 * Q.sum_pt_top30 + 13.1838
    if Q.tau1 < 0.107:
        z += 8.93 * Q.tau1 - 0.95551
    if Q.z_dr_0_0p05 >= 0.734:
        z += 3.82 * Q.z_dr_0_0p05 - 2.80388
    if Q.z_top20_slots >= 0.84:
        z += -5.57 * Q.z_top20_slots + 4.6788
    if Q.z_top50_slots < 0.985:
        z += 53.6 * Q.z_top50_slots - 52.796
    if Q.girth < 0.149 and Q.n_dr_0p1_0p2 > 12.2:
        z += 0.354 * (0.149 - Q.girth) * (Q.n_dr_0p1_0p2 - 12.2)
    if Q.girth < 0.185 and Q.psi_0p3 < 0.961:
        z += -159.0 * (0.185 - Q.girth) * (0.961 - Q.psi_0p3)
    if Q.n_particles < 53.8 and Q.C2 < 0.0943:
        z += -0.326 * (53.8 - Q.n_particles) * (0.0943 - Q.C2)
    if Q.sum_pt < 1010.0 and Q.absphi_1 > 0.00021:
        z += -0.14 * (1010.0 - Q.sum_pt) * (Q.absphi_1 - 0.00021)
    if Q.sum_pt < 1000.0 and Q.ptdr0_10 > 3.85:
        z += -0.00966 * (1000.0 - Q.sum_pt) * (Q.ptdr0_10 - 3.85)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.58
    if Q.e2 < 0.0336:
        z += -38.4 * Q.e2 + 1.29024
    if Q.e2 >= 0.0573:
        z += 125.0 * Q.e2 - 7.1625
    if Q.e3 >= 0.000115:
        z += -7930.0 * Q.e3 + 0.91195
    if Q.girth < 0.0952:
        z += 64.5 * Q.girth - 6.1404
    if Q.lam2 < 0.00637:
        z += -221.0 * Q.lam2 + 1.40777
    if Q.psi_0p2 >= 0.906:
        z += -13.2 * Q.psi_0p2 + 11.9592
    if Q.psi_0p3 >= 0.994:
        z += -80.8 * Q.psi_0p3 + 80.3152
    if Q.sj2_zsoft < 0.0445:
        z += -22.32 * Q.sj2_zsoft + 1.62933
    if 0.0445 <= Q.sj2_zsoft < 0.394:
        z += -1.82 * Q.sj2_zsoft + 0.71708
    if Q.sum_pt_top50 >= 1050.0:
        z += 0.00218 * Q.sum_pt_top50 - 2.289
    if Q.tau1 < 0.0532:
        z += -92.2 * Q.tau1 + 4.90504
    if Q.e3 < 0.000368 and Q.log_sum_pt < 7.02:
        z += -18900.0 * (0.000368 - Q.e3) * (7.02 - Q.log_sum_pt)
    if Q.girth < 0.101 and Q.eccentricity > 0.631:
        z += -26.9 * (0.101 - Q.girth) * (Q.eccentricity - 0.631)
    if Q.girth < 0.0962 and Q.psi_0p3 < 0.963:
        z += 1210.0 * (0.0962 - Q.girth) * (0.963 - Q.psi_0p3)
    if Q.girth < 0.0711 and Q.sum_pt < 993.0:
        z += 0.357 * (0.0711 - Q.girth) * (993.0 - Q.sum_pt)
    if Q.n_dr_0p1_0p2 > 12.2 and Q.psi_0p3 > 0.992:
        z += 6.68 * (Q.n_dr_0p1_0p2 - 12.2) * (Q.psi_0p3 - 0.992)
    if Q.psi_0p2 > 0.896 and Q.sj2_dr < 0.135:
        z += 139.0 * (Q.psi_0p2 - 0.896) * (0.135 - Q.sj2_dr)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.391
    if Q.LHA < 0.304:
        z += -38.2 * Q.LHA + 11.6128
    if Q.e2 < 0.0366:
        z += -88.2 * Q.e2 + 3.22812
    if Q.e3 < 3.44e-05:
        z += 17800.0 * Q.e3 - 0.61232
    if Q.girth < 0.081:
        z += 50.8 * Q.girth - 2.60624
    if 0.081 <= Q.girth < 0.0983:
        z += -87.2 * Q.girth + 8.57176
    if Q.psi_0p1 >= 0.854:
        z += -4.38 * Q.psi_0p1 + 3.74052
    if Q.psi_0p3 >= 0.997:
        z += 1340.0 * Q.psi_0p3 - 1335.98
    if Q.tau1 < 0.108:
        z += 108.0 * Q.tau1 - 11.664
    if Q.tau21 < 0.347:
        z += 3.54 * Q.tau21 - 1.22838
    if Q.LHA < 0.331 and Q.z_dr_0p1_0p2 < 0.256:
        z += -140.0 * (0.331 - Q.LHA) * (0.256 - Q.z_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 15.9 and Q.M2 < 0.113:
        z += 1.06 * (15.9 - Q.n_dr_0p2_0p4) * (0.113 - Q.M2)
    if Q.n_dr_0p2_0p4 < 20.2 and Q.e3 < 8.01e-05:
        z += -1580.0 * (20.2 - Q.n_dr_0p2_0p4) * (8.01e-05 - Q.e3)
    if Q.psi_0p3 > 0.998 and Q.girth2_top10 > 0.00775:
        z += -239000.0 * (Q.psi_0p3 - 0.998) * (Q.girth2_top10 - 0.00775)
    if Q.psi_0p3 > 0.997 and Q.log_sum_pt < 7.07:
        z += -3110.0 * (Q.psi_0p3 - 0.997) * (7.07 - Q.log_sum_pt)
    if Q.psi_0p3 > 0.997 and Q.mean_eta2 < 0.00685:
        z += -114000.0 * (Q.psi_0p3 - 0.997) * (0.00685 - Q.mean_eta2)
    if Q.psi_0p3 > 0.997 and Q.mean_phi2 < 0.00686:
        z += -109000.0 * (Q.psi_0p3 - 0.997) * (0.00686 - Q.mean_phi2)
    if Q.tau1 < 0.107 and Q.z_dr_0p1_0p2 < 0.286:
        z += 343.0 * (0.107 - Q.tau1) * (0.286 - Q.z_dr_0p1_0p2)
    if Q.tau21_b2 < 0.223 and Q.M2 < 0.0712:
        z += 106.0 * (0.223 - Q.tau21_b2) * (0.0712 - Q.M2)
    if Q.tau21_b2 < 0.235 and Q.girth2_top15 < 0.00802:
        z += -9470.0 * (0.235 - Q.tau21_b2) * (0.00802 - Q.girth2_top15)
    if Q.tau21_b2 < 0.237 and Q.girth2_top15 < 0.0101:
        z += 7570.0 * (0.237 - Q.tau21_b2) * (0.0101 - Q.girth2_top15)
    if Q.tau21_b2 < 0.257 and Q.n_real_top40 < 36.3:
        z += -0.146 * (0.257 - Q.tau21_b2) * (36.3 - Q.n_real_top40)
    if Q.tau21_b2 < 0.236 and Q.sj2_dr > 0.187:
        z += 71.6 * (0.236 - Q.tau21_b2) * (Q.sj2_dr - 0.187)
    if Q.tau21_b2 < 0.239 and Q.sum_pt_top50 < 1250.0:
        z += -0.0399 * (0.239 - Q.tau21_b2) * (1250.0 - Q.sum_pt_top50)
    return max(0.0, z)


def neuron_8(Q):
    z = 3.15
    if Q.LHA >= 0.208:
        z += -31.2 * Q.LHA + 6.4896
    if Q.e2 >= 0.0211:
        z += -57.8 * Q.e2 + 1.21958
    if 0.0422 <= Q.girth < 0.0785:
        z += 152.0 * Q.girth - 6.4144
    if 0.0785 <= Q.girth < 0.0954:
        z += 82.1 * Q.girth - 0.92725
    if Q.girth >= 0.0954:
        z += 21.4 * Q.girth + 4.86353
    if Q.girth2_top15 < 0.00997:
        z += 290.0 * Q.girth2_top15 - 2.8913
    if Q.girth2_top5 < 0.00793:
        z += -108.0 * Q.girth2_top5 + 0.85644
    if Q.log_sum_pt < 7.15:
        z += 8.08 * Q.log_sum_pt - 57.772
    if Q.n_dr_0p2_0p4 < 7.84:
        z += 0.119 * Q.n_dr_0p2_0p4 - 0.93296
    if Q.psi_0p3 >= 0.977:
        z += -42.9 * Q.psi_0p3 + 41.9133
    if Q.sum_pt < 1040.0:
        z += -0.0158 * Q.sum_pt + 16.432
    if Q.sum_pt_top40 < 999.0:
        z += -0.00747 * Q.sum_pt_top40 + 7.46253
    if Q.girth > 0.0678 and Q.e2 > 0.0403:
        z += 1440.0 * (Q.girth - 0.0678) * (Q.e2 - 0.0403)
    if Q.girth > 0.0747 and Q.sum_pt_top20 > 804.0:
        z += -0.23 * (Q.girth - 0.0747) * (Q.sum_pt_top20 - 804.0)
    if Q.girth > 0.0785 and Q.tau2 < 0.0814:
        z += 979.0 * (Q.girth - 0.0785) * (0.0814 - Q.tau2)
    if Q.girth2_top15 < 0.0238 and Q.tau4 > 0.0136:
        z += 3120.0 * (0.0238 - Q.girth2_top15) * (Q.tau4 - 0.0136)
    if Q.girth2_top5 < 0.0078 and Q.sj3_dr23 > 0.254:
        z += 430.0 * (0.0078 - Q.girth2_top5) * (Q.sj3_dr23 - 0.254)
    if Q.n_dr_0p2_0p4 < 16.7 and Q.e3 < 4.1e-05:
        z += -3500.0 * (16.7 - Q.n_dr_0p2_0p4) * (4.1e-05 - Q.e3)
    if Q.n_dr_0p2_0p4 < 17.8 and Q.log_sum_pt < 7.14:
        z += -0.529 * (17.8 - Q.n_dr_0p2_0p4) * (7.14 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 < 23.1 and Q.psi_0p1 < 0.843:
        z += 0.182 * (23.1 - Q.n_dr_0p2_0p4) * (0.843 - Q.psi_0p1)
    if Q.sum_pt < 995.0 and Q.z_0 > 0.125:
        z += -0.0343 * (995.0 - Q.sum_pt) * (Q.z_0 - 0.125)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.598
    if Q.D2 < 2.31:
        z += -0.478 * Q.D2 + 1.10418
    if Q.e2 >= 0.0235:
        z += 35.5 * Q.e2 - 0.83425
    if Q.girth >= 0.139:
        z += -60.1 * Q.girth + 8.3539
    if Q.girth2_top15 < 0.000831:
        z += 1420.0 * Q.girth2_top15 - 1.18002
    if Q.girth2_top15 >= 0.0209:
        z += -207.0 * Q.girth2_top15 + 4.3263
    if Q.sum_pt_top20 >= 741.0:
        z += -0.00428 * Q.sum_pt_top20 + 3.17148
    if Q.sum_pt_top50 < 978.0:
        z += -0.0111 * Q.sum_pt_top50 + 10.8558
    if Q.tau1 < 0.0847:
        z += -31.9 * Q.tau1 + 2.70193
    if Q.z_dr_0_0p05 >= 0.806:
        z += 5.05 * Q.z_dr_0_0p05 - 4.0703
    if Q.z_top5_slots >= 0.772:
        z += 4.47 * Q.z_top5_slots - 3.45084
    if Q.LHA > 0.331 and Q.lam2 < 0.00839:
        z += 1850.0 * (Q.LHA - 0.331) * (0.00839 - Q.lam2)
    if Q.girth > 0.145 and Q.soft6_pt > 4.33:
        z += 99.3 * (Q.girth - 0.145) * (Q.soft6_pt - 4.33)
    if Q.girth < 0.0608 and Q.sum_pt < 1010.0:
        z += -0.225 * (0.0608 - Q.girth) * (1010.0 - Q.sum_pt)
    if Q.girth2_top15 < 0.00639 and Q.n_dr_0p2_0p4 > 7.28:
        z += 15.3 * (0.00639 - Q.girth2_top15) * (Q.n_dr_0p2_0p4 - 7.28)
    if Q.girth2_top15 < 0.00603 and Q.psi_0p3 > 0.978:
        z += 7460.0 * (0.00603 - Q.girth2_top15) * (Q.psi_0p3 - 0.978)
    if Q.girth2_top15 < 0.00667 and Q.z_dr_0p2_0p4 < 0.0609:
        z += 5550.0 * (0.00667 - Q.girth2_top15) * (0.0609 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt < 6.8 and Q.dr_13 < 0.0873:
        z += 908.0 * (6.8 - Q.log_sum_pt) * (0.0873 - Q.dr_13)
    if Q.z_dr_0p1_0p2 > 0.454 and Q.soft5_z > -0.000252:
        z += 1330.0 * (Q.z_dr_0p1_0p2 - 0.454) * (Q.soft5_z - -0.000252)
    return max(0.0, z)


def neuron_10(Q):
    z = 5.47
    if Q.C2 < 0.0563:
        z += -18.9 * Q.C2 + 1.06407
    if Q.D2 < 2.49:
        z += -0.421 * Q.D2 + 1.04829
    if Q.LHA >= 0.182:
        z += -11.3 * Q.LHA + 2.0566
    if Q.e2 >= 0.0307:
        z += 39.7 * Q.e2 - 1.21879
    if Q.girth < 0.119:
        z += 32.8 * Q.girth - 3.9032
    if Q.girth2_top5 < 0.0226:
        z += 39.7 * Q.girth2_top5 - 0.89722
    if Q.log_sum_pt < 6.85:
        z += 8.8 * Q.log_sum_pt - 60.28
    if Q.n_dr_0p1_0p2 >= 20.4:
        z += 0.0325 * Q.n_dr_0p1_0p2 - 0.663
    if Q.psi_0p1 >= 0.914:
        z += 9.25 * Q.psi_0p1 - 8.4545
    if Q.psi_0p3 >= 0.999:
        z += -534.0 * Q.psi_0p3 + 533.466
    if Q.sd_rg >= 0.306:
        z += -11.5 * Q.sd_rg + 3.519
    if Q.sj2_dr >= 0.0894:
        z += -3.31 * Q.sj2_dr + 0.295914
    if Q.sum_pt < 992.0:
        z += -0.00718 * Q.sum_pt + 7.12256
    if Q.sum_pt_top5 >= 394.0:
        z += -0.00234 * Q.sum_pt_top5 + 0.92196
    if Q.tau1 >= 0.178:
        z += -39.3 * Q.tau1 + 6.9954
    if Q.tau21_b2 < 0.217:
        z += 5.63 * Q.tau21_b2 - 1.22171
    if Q.tau32 < 0.452:
        z += -3.34 * Q.tau32 + 1.50968
    if Q.z_dr_0p1_0p2 < 0.126:
        z += 4.48 * Q.z_dr_0p1_0p2 - 0.56448
    if Q.e2 > 0.0291 and Q.log_sum_pt > 6.92:
        z += -113.0 * (Q.e2 - 0.0291) * (Q.log_sum_pt - 6.92)
    if Q.e3 < 0.000246 and Q.tau32 < 0.861:
        z += -5430.0 * (0.000246 - Q.e3) * (0.861 - Q.tau32)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.206
    if Q.girth2_top10 < 0.00155:
        z += -575.0 * Q.girth2_top10 + 0.89125
    if Q.girth2_top15 < 0.00592:
        z += 295.0 * Q.girth2_top15 - 1.7464
    if Q.log_sum_pt >= 6.89:
        z += -4.69 * Q.log_sum_pt + 32.3141
    if Q.sj3_dr_max < 0.145:
        z += 7.71 * Q.sj3_dr_max - 1.11795
    if Q.z_dr_0_0p05 >= 0.814:
        z += 9.54 * Q.z_dr_0_0p05 - 7.76556
    if Q.z_dr_0p2_0p4 < 0.0734:
        z += -13.6 * Q.z_dr_0p2_0p4 + 0.99824
    if Q.log_sum_pt > 6.9 and Q.sd_zg < 0.322:
        z += 20.4 * (Q.log_sum_pt - 6.9) * (0.322 - Q.sd_zg)
    if Q.n_dr_0p2_0p4 < 9.5 and Q.girth2_top10 < 0.00566:
        z += -36.7 * (9.5 - Q.n_dr_0p2_0p4) * (0.00566 - Q.girth2_top10)
    if Q.n_dr_0p2_0p4 < 9.35 and Q.n_dr_0p1_0p2 < 32.9:
        z += 0.00769 * (9.35 - Q.n_dr_0p2_0p4) * (32.9 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.13
    if Q.D2 < 3.91:
        z += -0.288 * Q.D2 + 1.12608
    if Q.LHA < 0.194:
        z += 28.5 * Q.LHA - 5.529
    if Q.eccentricity >= 0.893:
        z += -6.94 * Q.eccentricity + 6.19742
    if Q.girth < 0.0624:
        z += -71.8 * Q.girth + 4.48032
    if Q.mean_eta2 < 0.00582:
        z += -125.0 * Q.mean_eta2 + 0.7275
    if Q.sj3_dr_max < 0.144:
        z += 16.0 * Q.sj3_dr_max - 1.6956
    if 0.144 <= Q.sj3_dr_max < 0.261:
        z += -5.2 * Q.sj3_dr_max + 1.3572
    if Q.LHA > 0.388 and Q.lam2 < 0.00434:
        z += 10000.0 * (Q.LHA - 0.388) * (0.00434 - Q.lam2)
    if Q.LHA > 0.364 and Q.log_sum_pt > 6.85:
        z += 124.0 * (Q.LHA - 0.364) * (Q.log_sum_pt - 6.85)
    if Q.girth < 0.0452 and Q.log_sum_pt > 6.79:
        z += 248.0 * (0.0452 - Q.girth) * (Q.log_sum_pt - 6.79)
    if Q.log_sum_pt > 6.84 and Q.mean_eta2 < 0.0118:
        z += -955.0 * (Q.log_sum_pt - 6.84) * (0.0118 - Q.mean_eta2)
    if Q.psi_0p2 > 0.971 and Q.n_dr_0p1_0p2 < 21.9:
        z += 2.27 * (Q.psi_0p2 - 0.971) * (21.9 - Q.n_dr_0p1_0p2)
    if Q.psi_0p3 > 0.99 and Q.girth2_top15 < 0.00525:
        z += 28600.0 * (Q.psi_0p3 - 0.99) * (0.00525 - Q.girth2_top15)
    if Q.sj3_dr_max < 0.241 and Q.eccentricity > 0.514:
        z += 19.3 * (0.241 - Q.sj3_dr_max) * (Q.eccentricity - 0.514)
    if Q.sj3_dr_max < 0.264 and Q.girth2_top10 > 0.00471:
        z += -13400.0 * (0.264 - Q.sj3_dr_max) * (Q.girth2_top10 - 0.00471)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.24
    if 0.324 <= Q.LHA < 0.433:
        z += 30.7 * Q.LHA - 9.9468
    if Q.LHA >= 0.433:
        z += -48.5 * Q.LHA + 24.3468
    if Q.e2 >= 0.0393:
        z += 64.2 * Q.e2 - 2.52306
    if Q.e3 < 0.000188:
        z += 740.0 * Q.e3 + 1.14463
    if 0.000188 <= Q.e3 < 0.000513:
        z += -3950.0 * Q.e3 + 2.02635
    if 0.0863 <= Q.girth < 0.0974:
        z += -68.5 * Q.girth + 5.91155
    if Q.girth >= 0.0974:
        z += -152.7 * Q.girth + 14.11263
    if Q.log_sum_pt < 6.92:
        z += 17.98 * Q.log_sum_pt - 125.0576
    if 6.92 <= Q.log_sum_pt < 6.98:
        z += 10.6 * Q.log_sum_pt - 73.988
    if Q.n_dr_0p1_0p2 < 32.6:
        z += 0.0254 * Q.n_dr_0p1_0p2 - 0.82804
    if Q.psi_0p2 >= 0.905:
        z += -9.14 * Q.psi_0p2 + 8.2717
    if Q.sum_pt >= 1180.0:
        z += -0.0143 * Q.sum_pt + 16.874
    if Q.sum_pt_top30 >= 1110.0:
        z += 0.0109 * Q.sum_pt_top30 - 12.099
    if Q.tau4 >= 0.0144:
        z += 84.4 * Q.tau4 - 1.21536
    if Q.girth > 0.0956 and Q.sum_pt < 1170.0:
        z += 0.176 * (Q.girth - 0.0956) * (1170.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_14(Q):
    z = 1.78
    if Q.e3 < 3.34e-05:
        z += 39400.0 * Q.e3 - 1.94236
    if 3.34e-05 <= Q.e3 < 7.69e-05:
        z += 14400.0 * Q.e3 - 1.10736
    if 0.0837 <= Q.girth < 0.121:
        z += -122.0 * Q.girth + 10.2114
    if Q.girth >= 0.121:
        z += -43.6 * Q.girth + 0.725
    if Q.n_dr_0p1_0p2 < 17.9:
        z += 0.0448 * Q.n_dr_0p1_0p2 - 0.80192
    if Q.sd_rg < 0.151:
        z += -8.06 * Q.sd_rg + 1.21706
    if 0.157 <= Q.sd_rg < 0.206:
        z += 22.3 * Q.sd_rg - 3.5011
    if Q.sd_rg >= 0.206:
        z += 4.4 * Q.sd_rg + 0.1863
    if Q.tau1 < 0.153:
        z += 24.0 * Q.tau1 - 3.672
    if Q.tau21_b2 < 0.339:
        z += -7.36 * Q.tau21_b2 + 2.49504
    if Q.z_dr_0p1_0p2 < 0.111:
        z += -5.48 * Q.z_dr_0p1_0p2 + 0.60828
    if Q.girth > 0.0732 and Q.max_dr < 0.389:
        z += 318.0 * (Q.girth - 0.0732) * (0.389 - Q.max_dr)
    if Q.girth > 0.119 and Q.max_dr < 0.355:
        z += -1440.0 * (Q.girth - 0.119) * (0.355 - Q.max_dr)
    if Q.tau21_b2 < 0.338 and Q.girth2_top15 < 0.00786:
        z += -874.0 * (0.338 - Q.tau21_b2) * (0.00786 - Q.girth2_top15)
    if Q.tau21_b2 < 0.337 and Q.sum_pt_top50 < 1150.0:
        z += -0.0318 * (0.337 - Q.tau21_b2) * (1150.0 - Q.sum_pt_top50)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.0542
    if Q.girth2_top5 < 0.00687:
        z += -106.0 * Q.girth2_top5 + 0.72822
    if Q.log_sum_pt >= 6.92:
        z += -2.34 * Q.log_sum_pt + 16.1928
    if Q.n_dr_0p2_0p4 >= 9.09:
        z += -0.0598 * Q.n_dr_0p2_0p4 + 0.543582
    if Q.sd_rg < 0.169:
        z += -4.18 * Q.sd_rg + 0.70642
    if Q.tau1 < 0.0713:
        z += 15.1 * Q.tau1 - 1.07663
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
