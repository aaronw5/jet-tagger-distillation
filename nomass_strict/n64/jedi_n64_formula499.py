"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; no mass observables or exact equivalents), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 79.0% (the network: 81.1%); same class as the network for 90.1% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.z_7                    pT of particle 7 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.z_1st                  largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_1               |Δφ| of particle 1
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau4                   N-subjettiness τ4 (β=1)
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_2=pt[2],
        pt_3=pt[3],
        soft1_pt=softp(1, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        z_7=z[7],
        soft1_z=softp(1, 'z'),
        soft5_z=softp(5, 'z'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        z_1st=zs[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        absphi_1=abs(phi[1]),
        sj2_dr=subjets(2)["dr"][0],
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam2=lam2,
        tau1=tau_n(1),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
    )


def neuron_0(Q):
    z = 0.842
    if Q.LHA < 0.201:
        z += -17.8 * Q.LHA + 1.4747
    if 0.201 <= Q.LHA < 0.272:
        z += -2.5 * Q.LHA - 1.6006
    if 0.272 <= Q.LHA < 0.314:
        z += 54.3 * Q.LHA - 17.0502
    if Q.LHA >= 0.325:
        z += -126.0 * Q.LHA + 40.95
    if Q.e2 < 0.019:
        z += -80.7 * Q.e2 + 1.5333
    if Q.e2 >= 0.0495:
        z += -23.0 * Q.e2 + 1.1385
    if Q.e3 < 2.8e-05:
        z += 39100.0 * Q.e3 - 1.0948
    if Q.sum_z_dr < 0.0371:
        z += 141.6 * Q.sum_z_dr - 4.71644
    if 0.0371 <= Q.sum_z_dr < 0.0527:
        z += 53.8 * Q.sum_z_dr - 1.45906
    if 0.0527 <= Q.sum_z_dr < 0.0652:
        z += -6.0 * Q.sum_z_dr + 1.6924
    if 0.0652 <= Q.sum_z_dr < 0.0824:
        z += -171.0 * Q.sum_z_dr + 12.4504
    if 0.0824 <= Q.sum_z_dr < 0.0859:
        z += -20.0 * Q.sum_z_dr + 0.008
    if 0.0859 <= Q.sum_z_dr < 0.0876:
        z += 150.0 * Q.sum_z_dr - 14.595
    if 0.0876 <= Q.sum_z_dr < 0.0973:
        z += 407.0 * Q.sum_z_dr - 37.1082
    if Q.sum_z_dr >= 0.0973:
        z += 257.0 * Q.sum_z_dr - 22.5132
    if Q.sum_z_dr2_top10 >= 0.00923:
        z += 44.3 * Q.sum_z_dr2_top10 - 0.408889
    if Q.sum_z_dr2_top15 < 0.00545:
        z += 122.0 * Q.sum_z_dr2_top15 - 0.00268
    if 0.00545 <= Q.sum_z_dr2_top15 < 0.00779:
        z += -283.0 * Q.sum_z_dr2_top15 + 2.20457
    if Q.log_sum_pt >= 7.02:
        z += 8.46 * Q.log_sum_pt - 59.3892
    if Q.mean_eta2 >= 0.000317:
        z += -495.0 * Q.mean_eta2 + 0.156915
    if Q.mean_phi2 < 0.00574:
        z += -465.0 * Q.mean_phi2 + 2.85975
    if 0.00574 <= Q.mean_phi2 < 0.00615:
        z += -932.0 * Q.mean_phi2 + 5.54033
    if Q.mean_phi2 >= 0.00615:
        z += -467.0 * Q.mean_phi2 + 2.68058
    if Q.psi_0p1 >= 0.909:
        z += -4.19 * Q.psi_0p1 + 3.80871
    if 0.167 <= Q.sd_rg < 0.313:
        z += -2.72 * Q.sd_rg + 0.45424
    if Q.sd_rg >= 0.313:
        z += 20.68 * Q.sd_rg - 6.86996
    if 0.137 <= Q.sj2_dr < 0.148:
        z += 26.8 * Q.sj2_dr - 3.6716
    if Q.sj2_dr >= 0.148:
        z += -1.0 * Q.sj2_dr + 0.4428
    if Q.sum_pt < 1010.0:
        z += 0.00195 * Q.sum_pt - 1.6551
    if 1010.0 <= Q.sum_pt < 1070.0:
        z += -0.00524 * Q.sum_pt + 5.6068
    if Q.sum_pt_top50 >= 1050.0:
        z += -0.00734 * Q.sum_pt_top50 + 7.707
    if Q.tau21 < 0.448:
        z += 1.43 * Q.tau21 - 0.64064
    if Q.tau21_b2 < 0.305:
        z += -2.54 * Q.tau21_b2 + 0.7747
    if Q.z_dr_0p05_0p1 >= 0.41:
        z += 0.556 * Q.z_dr_0p05_0p1 - 0.22796
    if Q.n_dr_0p2_0p4 < 17.5 and Q.log_sum_pt > 6.91:
        z += -0.804 * (17.5 - Q.n_dr_0p2_0p4) * (Q.log_sum_pt - 6.91)
    if Q.n_dr_0p2_0p4 < 18.3 and Q.sum_pt_top50 > 890.0:
        z += 0.000473 * (18.3 - Q.n_dr_0p2_0p4) * (Q.sum_pt_top50 - 890.0)
    if Q.psi_0p3 > 0.995 and Q.dr_5 < 0.0289:
        z += -2680.0 * (Q.psi_0p3 - 0.995) * (0.0289 - Q.dr_5)
    if Q.psi_0p3 > 0.996 and Q.n_dr_0p1_0p2 < 28.5:
        z += 3.12 * (Q.psi_0p3 - 0.996) * (28.5 - Q.n_dr_0p1_0p2)
    if Q.sum_pt_top40 > 971.0 and Q.dr_5 < 0.0453:
        z += 0.0587 * (Q.sum_pt_top40 - 971.0) * (0.0453 - Q.dr_5)
    if Q.sum_pt_top40 > 1000.0 and Q.sj2_dr < 0.163:
        z += 0.0155 * (Q.sum_pt_top40 - 1000.0) * (0.163 - Q.sj2_dr)
    return max(0.0, z)


def neuron_1(Q):
    z = -0.462
    if Q.C2 < 0.075:
        z += -17.5 * Q.C2 + 1.3125
    if Q.D2_b2 < 4.26:
        z += 0.187 * Q.D2_b2 - 0.79662
    if Q.sum_z_dr2_top10 < 0.0013:
        z += -781.0 * Q.sum_z_dr2_top10 + 1.0153
    if 6.91 <= Q.log_sum_pt < 7.0:
        z += 40.4 * Q.log_sum_pt - 279.164
    if Q.log_sum_pt >= 7.0:
        z += 20.0 * Q.log_sum_pt - 136.364
    if Q.n_dr_0p2_0p4 < 7.65:
        z += 0.147 * Q.n_dr_0p2_0p4 - 1.12455
    if Q.n_particles >= 48.6:
        z += 0.0954 * Q.n_particles - 4.63644
    if Q.n_real_top50 >= 34.6:
        z += 0.0971 * Q.n_real_top50 - 3.35966
    if Q.pt_entropy >= 1.88:
        z += 1.04 * Q.pt_entropy - 1.9552
    if Q.sum_pt_top50 < 919.0:
        z += -0.0047 * Q.sum_pt_top50 + 4.3193
    if Q.sum_pt_top50 >= 919.0:
        z += -0.0105 * Q.sum_pt_top50 + 9.6495
    if Q.z_top30_slots >= 0.936:
        z += 23.7 * Q.z_top30_slots - 22.1832
    if Q.M3 < 0.0412 and Q.psi_0p3 > 0.956:
        z += -401.0 * (0.0412 - Q.M3) * (Q.psi_0p3 - 0.956)
    if Q.n_particles > 37.3 and Q.soft1_z < 0.00237:
        z += -17.0 * (Q.n_particles - 37.3) * (0.00237 - Q.soft1_z)
    if Q.n_particles > 43.1 and Q.tau32 > 0.293:
        z += 0.0585 * (Q.n_particles - 43.1) * (Q.tau32 - 0.293)
    if Q.n_particles > 30.0 and Q.zdr_0 < 0.0111:
        z += 4.16 * (Q.n_particles - 30.0) * (0.0111 - Q.zdr_0)
    if Q.z_dr_0_0p05 > 0.757 and Q.n_dr_0p05_0p1 < 12.7:
        z += -0.37 * (Q.z_dr_0_0p05 - 0.757) * (12.7 - Q.n_dr_0p05_0p1)
    if Q.z_top30_slots > 0.953 and Q.pt1_dr01 > 6.23:
        z += 0.675 * (Q.z_top30_slots - 0.953) * (Q.pt1_dr01 - 6.23)
    return max(0.0, z)


def neuron_2(Q):
    z = 2.17
    if Q.C2 >= 0.112:
        z += -4.05 * Q.C2 + 0.4536
    if Q.LHA < 0.199:
        z += 1.91 * Q.LHA - 1.65209
    if 0.199 <= Q.LHA < 0.27:
        z += 10.6 * Q.LHA - 3.3814
    if 0.27 <= Q.LHA < 0.319:
        z += 19.91 * Q.LHA - 5.8951
    if 0.319 <= Q.LHA < 0.41:
        z += 20.01 * Q.LHA - 5.927
    if Q.LHA >= 0.41:
        z += 6.81 * Q.LHA - 0.515
    if 0.017 <= Q.e2 < 0.0544:
        z += 21.8 * Q.e2 - 0.3706
    if Q.e2 >= 0.0544:
        z += 10.2 * Q.e2 + 0.26044
    if Q.sum_z_dr < 0.0623:
        z += -14.9 * Q.sum_z_dr + 2.0711
    if 0.0623 <= Q.sum_z_dr < 0.139:
        z += -35.0 * Q.sum_z_dr + 3.32333
    if Q.sum_z_dr >= 0.139:
        z += -20.1 * Q.sum_z_dr + 1.25223
    if Q.sum_z_dr2_top10 >= 0.00768:
        z += -7.88 * Q.sum_z_dr2_top10 + 0.0605184
    if Q.sum_z_dr2_top15 < 0.00756:
        z += 18.0 * Q.sum_z_dr2_top15 + 0.09704
    if 0.00756 <= Q.sum_z_dr2_top15 < 0.00944:
        z += -124.0 * Q.sum_z_dr2_top15 + 1.17056
    if Q.lam2 >= 0.00021:
        z += 24.4 * Q.lam2 - 0.005124
    if 6.9 <= Q.log_sum_pt < 7.07:
        z += 5.13 * Q.log_sum_pt - 35.397
    if Q.log_sum_pt >= 7.07:
        z += 2.12 * Q.log_sum_pt - 14.1163
    if Q.mean_eta2 < 0.0116:
        z += 79.7 * Q.mean_eta2 - 0.94843
    if 0.0116 <= Q.mean_eta2 < 0.0119:
        z += 167.5 * Q.mean_eta2 - 1.96691
    if Q.mean_eta2 >= 0.0119:
        z += 87.8 * Q.mean_eta2 - 1.01848
    if Q.mean_phi2 < 0.0115:
        z += 79.0 * Q.mean_phi2 - 0.948
    if 0.0115 <= Q.mean_phi2 < 0.012:
        z += 163.4 * Q.mean_phi2 - 1.9186
    if Q.mean_phi2 >= 0.012:
        z += 84.4 * Q.mean_phi2 - 0.9706
    if Q.n_dr_0p2_0p4 < 12.1:
        z += 0.0137 * Q.n_dr_0p2_0p4 - 0.16577
    if Q.n_real_top50 < 47.9:
        z += 0.00889 * Q.n_real_top50 - 0.425831
    if Q.psi_0p1 < 0.506:
        z += -1.28 * Q.psi_0p1 + 0.64768
    if Q.psi_0p3 >= 0.989:
        z += -13.8 * Q.psi_0p3 + 13.6482
    if Q.sd_rg < 0.188:
        z += 0.935 * Q.sd_rg - 0.283305
    if 0.188 <= Q.sd_rg < 0.262:
        z += -3.935 * Q.sd_rg + 0.632255
    if 0.262 <= Q.sd_rg < 0.303:
        z += 1.815 * Q.sd_rg - 0.874245
    if Q.sd_rg >= 0.303:
        z += 0.88 * Q.sd_rg - 0.59094
    if Q.sum_pt < 993.0:
        z += 0.0010646 * Q.sum_pt - 1.200473
    if 993.0 <= Q.sum_pt < 1070.0:
        z += 0.000975 * Q.sum_pt - 1.1115
    if 1070.0 <= Q.sum_pt < 1140.0:
        z += -0.005915 * Q.sum_pt + 6.2608
    if Q.sum_pt >= 1140.0:
        z += -0.00689 * Q.sum_pt + 7.3723
    if Q.sum_pt_top30 < 924.0:
        z += -0.00027 * Q.sum_pt_top30 + 0.44856
    if 924.0 <= Q.sum_pt_top30 < 1050.0:
        z += -0.00158 * Q.sum_pt_top30 + 1.659
    if Q.sum_pt_top5 < 577.0:
        z += 0.00426 * Q.sum_pt_top5 - 2.45802
    if Q.sum_pt_top5 >= 580.0:
        z += 0.00366 * Q.sum_pt_top5 - 2.1228
    if Q.sum_pt_top50 >= 1050.0:
        z += 0.00232 * Q.sum_pt_top50 - 2.436
    if Q.tau1 >= 0.0489:
        z += -25.4 * Q.tau1 + 1.24206
    if Q.z_dr_0p1_0p2 >= 0.365:
        z += -1.27 * Q.z_dr_0p1_0p2 + 0.46355
    if Q.z_top50_slots < 0.958:
        z += -5.41 * Q.z_top50_slots + 5.26393
    if 0.958 <= Q.z_top50_slots < 0.973:
        z += -13.53 * Q.z_top50_slots + 13.04289
    if Q.z_top50_slots >= 0.973:
        z += -8.12 * Q.z_top50_slots + 7.77896
    if Q.z_top5_slots < 0.444:
        z += -4.59 * Q.z_top5_slots + 2.03796
    if Q.z_top5_slots >= 0.445:
        z += -4.18 * Q.z_top5_slots + 1.8601
    if Q.log_sum_pt > 6.95 and Q.C2 > 0.0545:
        z += -33.8 * (Q.log_sum_pt - 6.95) * (Q.C2 - 0.0545)
    if Q.log_sum_pt > 7.02 and Q.C2 > 0.0902:
        z += 61.0 * (Q.log_sum_pt - 7.02) * (Q.C2 - 0.0902)
    if Q.log_sum_pt > 6.91 and Q.sum_z_dr2_top15 < 0.0204:
        z += 202.0 * (Q.log_sum_pt - 6.91) * (0.0204 - Q.sum_z_dr2_top15)
    return max(0.0, z)


def neuron_3(Q):
    z = -1.5
    if Q.D2_b2 < 0.724:
        z += -0.481 * Q.D2_b2 + 0.348244
    if Q.LHA >= 0.201:
        z += 10.7 * Q.LHA - 2.1507
    if Q.N2 < 0.309:
        z += 4.25 * Q.N2 - 1.31325
    if Q.e2 < 0.0333:
        z += -20.8 * Q.e2 + 0.69264
    if Q.e2 >= 0.0377:
        z += -9.88 * Q.e2 + 0.372476
    if Q.sum_z_dr < 0.0875:
        z += -17.4 * Q.sum_z_dr + 1.5225
    if Q.sum_z_dr >= 0.124:
        z += 19.4 * Q.sum_z_dr - 2.4056
    if Q.lam2 < 0.000429:
        z += -218.0 * Q.lam2 + 0.34008
    if 0.000429 <= Q.lam2 < 0.00156:
        z += -170.0 * Q.lam2 + 0.319488
    if Q.lam2 >= 0.00156:
        z += 48.0 * Q.lam2 - 0.020592
    if Q.mean_eta2 < 0.00579:
        z += -50.5 * Q.mean_eta2 + 0.33734
    if 0.00579 <= Q.mean_eta2 < 0.00668:
        z += -110.9 * Q.mean_eta2 + 0.687056
    if Q.mean_eta2 >= 0.00668:
        z += -60.4 * Q.mean_eta2 + 0.349716
    if Q.mean_phi2 >= 0.0022:
        z += -67.5 * Q.mean_phi2 + 0.1485
    if Q.n_dr_0_0p05 < 13.7:
        z += 0.0186 * Q.n_dr_0_0p05 - 0.25482
    if Q.n_dr_0p1_0p2 >= 1.34:
        z += -0.0211 * Q.n_dr_0p1_0p2 + 0.028274
    if Q.n_dr_0p2_0p4 < 4.5:
        z += -0.2019 * Q.n_dr_0p2_0p4 + 1.15285
    if 4.5 <= Q.n_dr_0p2_0p4 < 11.5:
        z += -0.0349 * Q.n_dr_0p2_0p4 + 0.40135
    if Q.n_particles >= 39.9:
        z += -0.0122 * Q.n_particles + 0.48678
    if Q.n_real_top40 < 33.3:
        z += -0.0274 * Q.n_real_top40 + 0.91242
    if Q.n_real_top40 >= 34.2:
        z += -0.0221 * Q.n_real_top40 + 0.75582
    if Q.psi_0p3 < 0.998:
        z += -2.97 * Q.psi_0p3 + 2.96703
    if 0.998 <= Q.psi_0p3 < 0.999:
        z += 55.03 * Q.psi_0p3 - 54.91697
    if Q.psi_0p3 >= 0.999:
        z += 58.0 * Q.psi_0p3 - 57.884
    if Q.tau1 < 0.0469:
        z += 28.1 * Q.tau1 - 1.31789
    if Q.tau21_b2 < 0.329:
        z += -0.291 * Q.tau21_b2 + 0.095739
    if Q.tau4 < 0.0294:
        z += -15.8 * Q.tau4 + 0.46452
    if Q.z_top50_slots < 0.997:
        z += 12.4 * Q.z_top50_slots - 12.3628
    if Q.n_dr_0p1_0p2 < 14.5 and Q.sum_pt_top40 < 1020.0:
        z += -0.000146 * (14.5 - Q.n_dr_0p1_0p2) * (1020.0 - Q.sum_pt_top40)
    if Q.n_dr_0p2_0p4 < 4.3 and Q.e2 < 0.0344:
        z += -5.09 * (4.3 - Q.n_dr_0p2_0p4) * (0.0344 - Q.e2)
    if Q.n_particles < 52.0 and Q.sum_pt_top40 > 845.0:
        z += 3.08e-05 * (52.0 - Q.n_particles) * (Q.sum_pt_top40 - 845.0)
    if Q.psi_0p3 > 0.997 and Q.eccentricity > 0.87:
        z += 800.0 * (Q.psi_0p3 - 0.997) * (Q.eccentricity - 0.87)
    if Q.tau21 < 0.358 and Q.sum_pt < 1050.0:
        z += 0.0118 * (0.358 - Q.tau21) * (1050.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.23
    if Q.LHA < 0.242:
        z += -13.0 * Q.LHA + 3.146
    if Q.LHA >= 0.332:
        z += -30.9 * Q.LHA + 10.2588
    if Q.e2 >= 0.0568:
        z += -20.2 * Q.e2 + 1.14736
    if Q.sum_z_dr < 0.044:
        z += 91.1 * Q.sum_z_dr - 4.60648
    if 0.044 <= Q.sum_z_dr < 0.0552:
        z += 53.4 * Q.sum_z_dr - 2.94768
    if Q.sum_z_dr >= 0.0941:
        z += 62.5 * Q.sum_z_dr - 5.88125
    if Q.sum_z_dr2_top15 < 0.00764:
        z += 107.0 * Q.sum_z_dr2_top15 - 0.41708
    if 0.00764 <= Q.sum_z_dr2_top15 < 0.0105:
        z += -140.0 * Q.sum_z_dr2_top15 + 1.47
    if Q.n_dr_0_0p05 >= 8.57:
        z += 0.0237 * Q.n_dr_0_0p05 - 0.203109
    if Q.n_dr_0p2_0p4 < 16.6:
        z += -0.0397 * Q.n_dr_0p2_0p4 + 0.65902
    if Q.n_particles >= 37.9:
        z += -0.0262 * Q.n_particles + 0.99298
    if Q.n_pt_above_1 < 37.0:
        z += -0.0228 * Q.n_pt_above_1 + 0.8436
    if Q.sj2_dr < 0.179:
        z += 8.91 * Q.sj2_dr - 1.59489
    if Q.sj2_zsoft < 0.394:
        z += 1.35 * Q.sj2_zsoft - 0.5319
    if Q.sum_pt_top50 < 1060.0:
        z += 0.00213 * Q.sum_pt_top50 - 2.2578
    if Q.tau21_b2 >= 0.542:
        z += 1.38 * Q.tau21_b2 - 0.74796
    if Q.z_dr_0p2_0p4 < 0.0505:
        z += 5.47 * Q.z_dr_0p2_0p4 - 0.276235
    if Q.zdr_0 >= 0.00912:
        z += -8.72 * Q.zdr_0 + 0.0795264
    if Q.sum_z_dr < 0.0627 and Q.psi_0p3 > 0.998:
        z += -10300.0 * (0.0627 - Q.sum_z_dr) * (Q.psi_0p3 - 0.998)
    if Q.sum_z_dr < 0.1 and Q.psi_0p3 > 0.997:
        z += 4620.0 * (0.1 - Q.sum_z_dr) * (Q.psi_0p3 - 0.997)
    if Q.sum_z_dr < 0.0614 and Q.sum_pt_top40 < 1070.0:
        z += -0.147 * (0.0614 - Q.sum_z_dr) * (1070.0 - Q.sum_pt_top40)
    if Q.sum_z_dr2_top15 > 0.000577 and Q.sj3_pairmin_over_m < 0.444:
        z += -67.2 * (Q.sum_z_dr2_top15 - 0.000577) * (0.444 - Q.sj3_pairmin_over_m)
    if Q.sum_z_dr2_top15 < 0.00519 and Q.sum_pt_top40 > 1070.0:
        z += 0.282 * (0.00519 - Q.sum_z_dr2_top15) * (Q.sum_pt_top40 - 1070.0)
    if Q.n_particles > 20.6 and Q.lam2 > 0.00199:
        z += 1.17 * (Q.n_particles - 20.6) * (Q.lam2 - 0.00199)
    if Q.n_particles > 34.4 and Q.soft1_pt > 0.0851:
        z += -0.00724 * (Q.n_particles - 34.4) * (Q.soft1_pt - 0.0851)
    return max(0.0, z)


def neuron_5(Q):
    z = 5.52
    if Q.LHA >= 0.192:
        z += -18.4 * Q.LHA + 3.5328
    if Q.M3 < 0.034:
        z += -7.76 * Q.M3 + 0.26384
    if Q.e2 < 0.0409:
        z += -21.7 * Q.e2 + 0.88753
    if Q.sum_z_dr < 0.0369:
        z += 4.8 * Q.sum_z_dr - 5.78673
    if 0.0369 <= Q.sum_z_dr < 0.156:
        z += 47.1 * Q.sum_z_dr - 7.3476
    if Q.sum_z_dr2_top15 >= 0.0182:
        z += -40.4 * Q.sum_z_dr2_top15 + 0.73528
    if Q.sum_z_dr2_top2 >= 0.0142:
        z += 14.4 * Q.sum_z_dr2_top2 - 0.20448
    if Q.log_sum_pt < 7.06:
        z += -10.1 * Q.log_sum_pt + 71.306
    if Q.max_dr < 0.443:
        z += -1.09 * Q.max_dr + 0.48287
    if Q.mean_eta2 < 0.00587:
        z += -149.0 * Q.mean_eta2 + 0.87463
    if Q.mean_eta2 >= 0.00607:
        z += -153.0 * Q.mean_eta2 + 0.92871
    if Q.mean_phi2 < 0.00588:
        z += -139.0 * Q.mean_phi2 + 0.81732
    if Q.mean_phi2 >= 0.00608:
        z += -148.0 * Q.mean_phi2 + 0.89984
    if Q.n_dr_0p1_0p2 < 5.53:
        z += -0.0346 * Q.n_dr_0p1_0p2 + 0.191338
    if Q.n_dr_0p1_0p2 >= 8.32:
        z += 0.0131 * Q.n_dr_0p1_0p2 - 0.108992
    if Q.n_dr_0p2_0p4 < 10.6:
        z += -0.0467 * Q.n_dr_0p2_0p4 + 0.49502
    if Q.n_particles < 59.8:
        z += -0.0129 * Q.n_particles + 0.77142
    if Q.n_pt_above_10 < 13.2:
        z += -0.0548 * Q.n_pt_above_10 + 0.72336
    if 0.157 <= Q.sd_rg < 0.198:
        z += 4.83 * Q.sd_rg - 0.75831
    if Q.sd_rg >= 0.198:
        z += -0.46 * Q.sd_rg + 0.28911
    if Q.soft1_pt >= 1.72:
        z += -0.871 * Q.soft1_pt + 1.49812
    if Q.soft1_z >= 0.00198:
        z += 946.0 * Q.soft1_z - 1.87308
    if Q.sum_pt < 1000.0:
        z += 0.02331 * Q.sum_pt - 22.5188
    if 1000.0 <= Q.sum_pt < 1080.0:
        z += -0.00989 * Q.sum_pt + 10.6812
    if Q.sum_pt >= 1160.0:
        z += -0.00563 * Q.sum_pt + 6.5308
    if Q.sum_pt_top5 < 838.0:
        z += 0.000979 * Q.sum_pt_top5 - 0.820402
    if Q.sum_pt_top50 < 1060.0:
        z += 0.0104 * Q.sum_pt_top50 - 11.128
    if 1060.0 <= Q.sum_pt_top50 < 1070.0:
        z += 0.0139 * Q.sum_pt_top50 - 14.838
    if Q.sum_pt_top50 >= 1070.0:
        z += 0.0035 * Q.sum_pt_top50 - 3.71
    if Q.tau1 < 0.11:
        z += 21.1 * Q.tau1 - 2.321
    if Q.z_dr_0_0p05 >= 0.887:
        z += 3.63 * Q.z_dr_0_0p05 - 3.21981
    if Q.z_dr_0p2_0p4 < 0.0609:
        z += 7.49 * Q.z_dr_0p2_0p4 - 0.456141
    if Q.n_particles < 60.5 and Q.C2 < 0.0891:
        z += -0.213 * (60.5 - Q.n_particles) * (0.0891 - Q.C2)
    if Q.sum_pt < 1080.0 and Q.e4 < 6.87e-08:
        z += -2240.0 * (1080.0 - Q.sum_pt) * (6.87e-08 - Q.e4)
    return max(0.0, z)


def neuron_6(Q):
    z = 2.19
    if Q.LHA < 0.228:
        z += -1.2 * Q.LHA + 2.7881
    if 0.228 <= Q.LHA < 0.328:
        z += -23.5 * Q.LHA + 7.8725
    if 0.328 <= Q.LHA < 0.335:
        z += 18.6 * Q.LHA - 5.9363
    if Q.LHA >= 0.335:
        z += 42.1 * Q.LHA - 13.8088
    if Q.dr_max_012 < 0.176:
        z += -0.82 * Q.dr_max_012 + 0.14432
    if 0.0129 <= Q.e2 < 0.0567:
        z += -28.2 * Q.e2 + 0.36378
    if Q.e2 >= 0.0567:
        z += 11.3 * Q.e2 - 1.87587
    if Q.sum_z_dr < 0.0494:
        z += -19.7 * Q.sum_z_dr - 3.17785
    if 0.0494 <= Q.sum_z_dr < 0.0926:
        z += 86.3 * Q.sum_z_dr - 8.41425
    if 0.0926 <= Q.sum_z_dr < 0.0975:
        z += 12.2 * Q.sum_z_dr - 1.55259
    if Q.sum_z_dr >= 0.0975:
        z += -74.1 * Q.sum_z_dr + 6.86166
    if Q.sum_z_dr2_top15 < 0.00384:
        z += -154.0 * Q.sum_z_dr2_top15 + 0.72226
    if 0.00384 <= Q.sum_z_dr2_top15 < 0.00469:
        z += -128.2 * Q.sum_z_dr2_top15 + 0.623188
    if Q.sum_z_dr2_top15 >= 0.00469:
        z += 25.8 * Q.sum_z_dr2_top15 - 0.099072
    if Q.sum_z_dr2_top3 < 0.00208:
        z += 189.0 * Q.sum_z_dr2_top3 - 0.39312
    if Q.lam2 < 0.000159:
        z += -2170.0 * Q.lam2 + 0.34503
    if Q.lam2 >= 0.000547:
        z += -112.0 * Q.lam2 + 0.061264
    if Q.n_dr_0p2_0p4 >= 14.1:
        z += -0.0177 * Q.n_dr_0p2_0p4 + 0.24957
    if Q.psi_0p2 >= 0.997:
        z += -24.4 * Q.psi_0p2 + 24.3268
    if Q.sj2_dr < 0.143:
        z += 7.34 * Q.sj2_dr - 1.33588
    if 0.143 <= Q.sj2_dr < 0.18:
        z += -2.66 * Q.sj2_dr + 0.09412
    if 0.18 <= Q.sj2_dr < 0.182:
        z += 7.94 * Q.sj2_dr - 1.81388
    if Q.sj2_dr >= 0.182:
        z += 0.6 * Q.sj2_dr - 0.478
    if Q.sum_pt_top15 < 981.0:
        z += 0.00145 * Q.sum_pt_top15 - 1.42245
    if Q.sum_pt_top15 >= 989.0:
        z += 0.0013 * Q.sum_pt_top15 - 1.2857
    if Q.sum_pt_top50 < 1060.0:
        z += -0.00164 * Q.sum_pt_top50 + 1.7384
    if Q.z_dr_0p05_0p1 >= 0.763:
        z += -2.65 * Q.z_dr_0p05_0p1 + 2.02195
    if Q.z_dr_0p1_0p2 < 0.251:
        z += -1.91 * Q.z_dr_0p1_0p2 + 0.47941
    if Q.zdr_0 >= 0.02:
        z += 11.3 * Q.zdr_0 - 0.226
    if Q.LHA < 0.324 and Q.psi_0p3 < 0.98:
        z += -736.0 * (0.324 - Q.LHA) * (0.98 - Q.psi_0p3)
    if Q.e2 > 0.0456 and Q.psi_0p3 > 0.99:
        z += 2060.0 * (Q.e2 - 0.0456) * (Q.psi_0p3 - 0.99)
    if Q.e3 < 0.000304 and Q.log_sum_pt < 7.01:
        z += -9660.0 * (0.000304 - Q.e3) * (7.01 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.0767 and Q.M2 < 0.117:
        z += -69.1 * (0.0767 - Q.sum_z_dr) * (0.117 - Q.M2)
    if Q.sum_z_dr < 0.105 and Q.eccentricity > 0.667:
        z += -12.8 * (0.105 - Q.sum_z_dr) * (Q.eccentricity - 0.667)
    if Q.sum_z_dr < 0.0952 and Q.psi_0p3 < 0.979:
        z += 1920.0 * (0.0952 - Q.sum_z_dr) * (0.979 - Q.psi_0p3)
    if Q.sum_z_dr < 0.0802 and Q.sum_pt < 1010.0:
        z += 0.206 * (0.0802 - Q.sum_z_dr) * (1010.0 - Q.sum_pt)
    if Q.sum_z_dr2_top15 < 0.00448 and Q.sum_pt_top40 > 892.0:
        z += -0.486 * (0.00448 - Q.sum_z_dr2_top15) * (Q.sum_pt_top40 - 892.0)
    if Q.n_dr_0p1_0p2 > 13.5 and Q.psi_0p3 > 0.985:
        z += 1.55 * (Q.n_dr_0p1_0p2 - 13.5) * (Q.psi_0p3 - 0.985)
    if Q.n_dr_0p1_0p2 > 19.8 and Q.sum_pt > 967.0:
        z += -0.000106 * (Q.n_dr_0p1_0p2 - 19.8) * (Q.sum_pt - 967.0)
    if Q.psi_0p2 > 0.912 and Q.sj2_dr < 0.154:
        z += 96.3 * (Q.psi_0p2 - 0.912) * (0.154 - Q.sj2_dr)
    if Q.psi_0p2 > 0.877 and Q.z_top50_slots < 0.972:
        z += 147.0 * (Q.psi_0p2 - 0.877) * (0.972 - Q.z_top50_slots)
    if Q.tau1 < 0.0548 and Q.sum_pt < 1020.0:
        z += -0.143 * (0.0548 - Q.tau1) * (1020.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.324
    if Q.LHA < 0.306:
        z += -15.9 * Q.LHA + 4.4484
    if 0.306 <= Q.LHA < 0.313:
        z += 9.0 * Q.LHA - 3.171
    if 0.313 <= Q.LHA < 0.328:
        z += 23.6 * Q.LHA - 7.7408
    if Q.e2 < 0.03:
        z += -68.4 * Q.e2 + 2.052
    if Q.sum_z_dr < 0.0465:
        z += 101.2 * Q.sum_z_dr - 6.8386
    if 0.0465 <= Q.sum_z_dr < 0.0809:
        z += 62.0 * Q.sum_z_dr - 5.0158
    if Q.n_dr_0p2_0p4 < 20.9:
        z += -0.0906 * Q.n_dr_0p2_0p4 + 1.89354
    if Q.psi_0p3 >= 0.997:
        z += 1580.0 * Q.psi_0p3 - 1575.26
    if 0.151 <= Q.sd_rg < 0.203:
        z += 3.71 * Q.sd_rg - 0.56021
    if 0.203 <= Q.sd_rg < 0.3:
        z += -5.9 * Q.sd_rg + 1.39062
    if Q.sd_rg >= 0.3:
        z += 7.4 * Q.sd_rg - 2.59938
    if Q.sum_pt >= 1050.0:
        z += -0.00743 * Q.sum_pt + 7.8015
    if Q.sum_pt_top30 < 998.0:
        z += 0.00283 * Q.sum_pt_top30 - 2.82434
    if Q.sum_pt_top30 >= 1010.0:
        z += 0.00922 * Q.sum_pt_top30 - 9.3122
    if Q.tau1 >= 0.102:
        z += -19.5 * Q.tau1 + 1.989
    if Q.z_dr_0p2_0p4 < 0.0754:
        z += 4.94 * Q.z_dr_0p2_0p4 - 0.372476
    if Q.z_top30_slots >= 0.972:
        z += -13.5 * Q.z_top30_slots + 13.122
    if Q.n_dr_0p1_0p2 > 24.9 and Q.pt_2 > 78.4:
        z += -0.00187 * (Q.n_dr_0p1_0p2 - 24.9) * (Q.pt_2 - 78.4)
    if Q.n_dr_0p2_0p4 < 20.1 and Q.e3 < 8.95e-05:
        z += -778.0 * (20.1 - Q.n_dr_0p2_0p4) * (8.95e-05 - Q.e3)
    if Q.n_dr_0p2_0p4 < 22.1 and Q.sum_z_dr2_top5 > 0.0105:
        z += -1.44 * (22.1 - Q.n_dr_0p2_0p4) * (Q.sum_z_dr2_top5 - 0.0105)
    if Q.n_dr_0p2_0p4 < 18.2 and Q.n_dr_0p1_0p2 > 21.8:
        z += -0.00468 * (18.2 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 21.8)
    if Q.psi_0p3 > 0.995 and Q.sum_z_dr2_top10 > 0.00872:
        z += -147000.0 * (Q.psi_0p3 - 0.995) * (Q.sum_z_dr2_top10 - 0.00872)
    if Q.psi_0p3 > 0.997 and Q.log_sum_pt < 7.07:
        z += -3090.0 * (Q.psi_0p3 - 0.997) * (7.07 - Q.log_sum_pt)
    if Q.psi_0p3 > 0.997 and Q.mean_eta2 < 0.00703:
        z += -146000.0 * (Q.psi_0p3 - 0.997) * (0.00703 - Q.mean_eta2)
    if Q.psi_0p3 > 0.997 and Q.mean_phi2 < 0.00697:
        z += -138000.0 * (Q.psi_0p3 - 0.997) * (0.00697 - Q.mean_phi2)
    if Q.tau21_b2 < 0.22 and Q.sum_z_dr2_top15 < 0.00595:
        z += 5450.0 * (0.22 - Q.tau21_b2) * (0.00595 - Q.sum_z_dr2_top15)
    if Q.tau21_b2 < 0.239 and Q.sum_z_dr2_top15 < 0.00798:
        z += -9420.0 * (0.239 - Q.tau21_b2) * (0.00798 - Q.sum_z_dr2_top15)
    if Q.tau21_b2 < 0.24 and Q.sum_z_dr2_top15 < 0.00998:
        z += 4570.0 * (0.24 - Q.tau21_b2) * (0.00998 - Q.sum_z_dr2_top15)
    if Q.tau21_b2 < 0.247 and Q.sj2_dr > 0.184:
        z += -146.0 * (0.247 - Q.tau21_b2) * (Q.sj2_dr - 0.184)
    if Q.tau21_b2 < 0.253 and Q.sj2_dr > 0.153:
        z += 122.0 * (0.253 - Q.tau21_b2) * (Q.sj2_dr - 0.153)
    return max(0.0, z)


def neuron_8(Q):
    z = 3.44
    if Q.LHA >= 0.33:
        z += 50.2 * Q.LHA - 16.566
    if Q.sum_z_dr >= 0.0945:
        z += -149.0 * Q.sum_z_dr + 14.0805
    if Q.log_sum_pt < 7.03:
        z += 9.81 * Q.log_sum_pt - 68.9643
    if Q.mean_eta2 >= 0.000529:
        z += 175.0 * Q.mean_eta2 - 0.092575
    if Q.mean_phi2 < 0.0112:
        z += 171.0 * Q.mean_phi2 - 1.9152
    if Q.mean_phi2 >= 0.0115:
        z += 176.0 * Q.mean_phi2 - 2.024
    if Q.n_dr_0p2_0p4 < 19.4:
        z += 0.0947 * Q.n_dr_0p2_0p4 - 1.83718
    if Q.sum_pt < 1040.0:
        z += -0.022 * Q.sum_pt + 22.88
    if Q.z_top50_slots >= 0.968:
        z += -31.8 * Q.z_top50_slots + 30.7824
    if Q.sum_z_dr > 0.0832 and Q.sum_pt < 1150.0:
        z += 0.281 * (Q.sum_z_dr - 0.0832) * (1150.0 - Q.sum_pt)
    if Q.sum_z_dr > 0.0907 and Q.sum_pt_top50 < 1000.0:
        z += -0.267 * (Q.sum_z_dr - 0.0907) * (1000.0 - Q.sum_pt_top50)
    if Q.n_dr_0p2_0p4 < 14.4 and Q.e3 < 3.5e-05:
        z += 1540.0 * (14.4 - Q.n_dr_0p2_0p4) * (3.5e-05 - Q.e3)
    if Q.n_dr_0p2_0p4 < 21.6 and Q.psi_0p1 < 0.784:
        z += 0.168 * (21.6 - Q.n_dr_0p2_0p4) * (0.784 - Q.psi_0p1)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.00133
    if Q.LHA >= 0.389:
        z += 59.3 * Q.LHA - 23.0677
    if Q.sum_z_dr < 0.0611:
        z += -30.2 * Q.sum_z_dr + 1.84522
    if 0.084 <= Q.sum_z_dr < 0.133:
        z += 14.2 * Q.sum_z_dr - 1.1928
    if Q.sum_z_dr >= 0.133:
        z += -120.8 * Q.sum_z_dr + 16.7622
    if Q.sum_z_dr2_top15 < 0.000849:
        z += 703.0 * Q.sum_z_dr2_top15 - 0.596847
    if Q.psi_0p1 >= 0.89:
        z += 4.22 * Q.psi_0p1 - 3.7558
    if Q.psi_0p2 >= 0.989:
        z += 24.1 * Q.psi_0p2 - 23.8349
    if Q.sum_pt < 1000.0:
        z += -0.0101 * Q.sum_pt + 10.1
    if Q.sum_pt >= 1010.0:
        z += -0.00261 * Q.sum_pt + 2.6361
    if Q.z_top50_slots >= 0.972:
        z += -12.4 * Q.z_top50_slots + 12.0528
    if Q.sum_z_dr > 0.123 and Q.soft6_pt > 1.51:
        z += -11.4 * (Q.sum_z_dr - 0.123) * (Q.soft6_pt - 1.51)
    if Q.sum_z_dr < 0.0462 and Q.sum_pt < 1050.0:
        z += -0.282 * (0.0462 - Q.sum_z_dr) * (1050.0 - Q.sum_pt)
    if Q.sum_z_dr2_top15 < 0.00378 and Q.n_dr_0p2_0p4 > 4.04:
        z += 15.1 * (0.00378 - Q.sum_z_dr2_top15) * (Q.n_dr_0p2_0p4 - 4.04)
    if Q.sum_z_dr2_top15 < 0.00763 and Q.psi_0p3 > 0.984:
        z += 7700.0 * (0.00763 - Q.sum_z_dr2_top15) * (Q.psi_0p3 - 0.984)
    if Q.sum_z_dr2_top15 < 0.00576 and Q.z_dr_0p2_0p4 < 0.0588:
        z += 6090.0 * (0.00576 - Q.sum_z_dr2_top15) * (0.0588 - Q.z_dr_0p2_0p4)
    if Q.sum_pt_top50 < 1010.0 and Q.e3 > 0.000128:
        z += 6.53 * (1010.0 - Q.sum_pt_top50) * (Q.e3 - 0.000128)
    if Q.sum_pt_top50 < 1000.0 and Q.sum_pt_top20 > 736.0:
        z += -3.6e-05 * (1000.0 - Q.sum_pt_top50) * (Q.sum_pt_top20 - 736.0)
    if Q.z_dr_0p1_0p2 > 0.458 and Q.soft5_pt > 0.392:
        z += -3.3 * (Q.z_dr_0p1_0p2 - 0.458) * (Q.soft5_pt - 0.392)
    if Q.z_dr_0p1_0p2 > 0.448 and Q.soft5_z > 0.000325:
        z += 3430.0 * (Q.z_dr_0p1_0p2 - 0.448) * (Q.soft5_z - 0.000325)
    if Q.z_top40_slots > 0.98 and Q.C3 < 0.0153:
        z += -1310.0 * (Q.z_top40_slots - 0.98) * (0.0153 - Q.C3)
    return max(0.0, z)


def neuron_10(Q):
    z = 4.59
    if Q.C2 < 0.0667:
        z += -13.7 * Q.C2 + 0.91379
    if Q.LHA < 0.317:
        z += -15.1 * Q.LHA + 4.7867
    if 0.351 <= Q.LHA < 0.414:
        z += 23.4 * Q.LHA - 8.2134
    if Q.LHA >= 0.414:
        z += 86.6 * Q.LHA - 34.3782
    if Q.e2 < 0.0416:
        z += 45.7 * Q.e2 - 1.90112
    if Q.e2 >= 0.0473:
        z += 41.5 * Q.e2 - 1.96295
    if Q.e3 < 0.000361:
        z += 1780.0 * Q.e3 - 0.64258
    if Q.sum_z_dr < 0.018:
        z += 92.9 * Q.sum_z_dr - 7.6716
    if 0.018 <= Q.sum_z_dr < 0.102:
        z += 59.4 * Q.sum_z_dr - 7.0686
    if 0.102 <= Q.sum_z_dr < 0.119:
        z += -53.6 * Q.sum_z_dr + 4.4574
    if 0.119 <= Q.sum_z_dr < 0.146:
        z += -113.0 * Q.sum_z_dr + 11.526
    if Q.sum_z_dr >= 0.146:
        z += -268.0 * Q.sum_z_dr + 34.156
    if Q.n_dr_0p1_0p2 >= 21.1:
        z += 0.0261 * Q.n_dr_0p1_0p2 - 0.55071
    z += 0.011 * Q.n_real_top50
    if Q.sum_pt < 989.0:
        z += 0.0107 * Q.sum_pt - 10.5823
    if Q.sum_pt_top2 >= 184.0:
        z += -0.00169 * Q.sum_pt_top2 + 0.31096
    if Q.sum_pt_top40 < 1070.0:
        z += 0.00305 * Q.sum_pt_top40 - 3.2635
    if Q.tau1 < 0.0415:
        z += 42.6 * Q.tau1 - 1.7679
    if Q.tau21_b2 < 0.257:
        z += 2.36 * Q.tau21_b2 - 0.60652
    if Q.LHA > 0.153 and Q.psi_0p3 < 0.997:
        z += -20.3 * (Q.LHA - 0.153) * (0.997 - Q.psi_0p3)
    if Q.e2 > 0.0266 and Q.log_sum_pt > 6.94:
        z += -149.0 * (Q.e2 - 0.0266) * (Q.log_sum_pt - 6.94)
    if Q.e2 > 0.027 and Q.soft1_pt > 1.11:
        z += -25.3 * (Q.e2 - 0.027) * (Q.soft1_pt - 1.11)
    if Q.e3 < 0.000395 and Q.tau32 < 0.832:
        z += -1880.0 * (0.000395 - Q.e3) * (0.832 - Q.tau32)
    if Q.sum_z_dr < 0.103 and Q.psi_0p2 > 0.955:
        z += 431.0 * (0.103 - Q.sum_z_dr) * (Q.psi_0p2 - 0.955)
    if Q.sum_z_dr < 0.135 and Q.pt1_dr01 > 7.58:
        z += 0.268 * (0.135 - Q.sum_z_dr) * (Q.pt1_dr01 - 7.58)
    if Q.sum_z_dr > 0.105 and Q.sum_pt_top50 < 1200.0:
        z += 0.247 * (Q.sum_z_dr - 0.105) * (1200.0 - Q.sum_pt_top50)
    if Q.sum_z_dr < 0.125 and Q.sum_pt_top50 < 1000.0:
        z += 0.152 * (0.125 - Q.sum_z_dr) * (1000.0 - Q.sum_pt_top50)
    if Q.n_dr_0p2_0p4 < 10.2 and Q.n_real_top40 > 22.0:
        z += -0.00408 * (10.2 - Q.n_dr_0p2_0p4) * (Q.n_real_top40 - 22.0)
    if Q.psi_0p3 > 0.998 and Q.eccentricity > 0.474:
        z += -434.0 * (Q.psi_0p3 - 0.998) * (Q.eccentricity - 0.474)
    if Q.sum_pt_top40 < 1090.0 and Q.n_dr_0_0p05 > -0.0017:
        z += 0.000128 * (1090.0 - Q.sum_pt_top40) * (Q.n_dr_0_0p05 - -0.0017)
    if Q.sum_pt_top50 < 948.0 and Q.z_dr_0p05_0p1 < 0.175:
        z += -0.0288 * (948.0 - Q.sum_pt_top50) * (0.175 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_11(Q):
    z = -1.17
    if Q.D2 < 1.84:
        z += -1.16 * Q.D2 + 2.1344
    if Q.LHA < 0.28:
        z += 4.7 * Q.LHA - 3.172
    if 0.28 <= Q.LHA < 0.338:
        z += 32.0 * Q.LHA - 10.816
    if Q.N2 < 0.19:
        z += 4.03 * Q.N2 - 0.7657
    if Q.sum_z_dr < 0.0684:
        z += -24.8 * Q.sum_z_dr + 4.87532
    if 0.0684 <= Q.sum_z_dr < 0.0973:
        z += -110.0 * Q.sum_z_dr + 10.703
    if Q.sum_z_dr >= 0.104:
        z += -8.95 * Q.sum_z_dr + 0.9308
    if Q.sum_z_dr2_top10 < 0.000934:
        z += -335.0 * Q.sum_z_dr2_top10 + 0.04307
    if 0.000934 <= Q.sum_z_dr2_top10 < 0.00497:
        z += -85.0 * Q.sum_z_dr2_top10 - 0.19043
    if 0.00497 <= Q.sum_z_dr2_top10 < 0.00823:
        z += 188.0 * Q.sum_z_dr2_top10 - 1.54724
    if Q.sum_z_dr2_top15 < 0.00102:
        z += -48.0 * Q.sum_z_dr2_top15 + 1.2662
    if 0.00102 <= Q.sum_z_dr2_top15 < 0.00613:
        z += 13.4 * Q.sum_z_dr2_top15 + 1.203572
    if 0.00613 <= Q.sum_z_dr2_top15 < 0.00801:
        z += -455.6 * Q.sum_z_dr2_top15 + 4.078542
    if Q.sum_z_dr2_top15 >= 0.00801:
        z += 61.4 * Q.sum_z_dr2_top15 - 0.062628
    if Q.log_sum_pt >= 6.86:
        z += -2.24 * Q.log_sum_pt + 15.3664
    if Q.n_dr_0p1_0p2 < 9.23:
        z += 0.0089 * Q.n_dr_0p1_0p2 - 0.318435
    if Q.n_dr_0p1_0p2 >= 9.23:
        z += -0.0256 * Q.n_dr_0p1_0p2
    if Q.n_dr_0p2_0p4 < 16.5:
        z += -0.0445 * Q.n_dr_0p2_0p4 + 0.73425
    if Q.psi_0p1 >= 0.647:
        z += -3.05 * Q.psi_0p1 + 1.97335
    if Q.psi_0p2 >= 0.996:
        z += 54.5 * Q.psi_0p2 - 54.282
    if Q.pt_entropy >= 2.25:
        z += -0.247 * Q.pt_entropy + 0.55575
    if Q.z_dr_0_0p05 < 0.128:
        z += 2.82 * Q.z_dr_0_0p05 - 0.36096
    if Q.z_dr_0p05_0p1 >= 0.691:
        z += 2.45 * Q.z_dr_0p05_0p1 - 1.69295
    if Q.z_dr_0p2_0p4 < 0.0983:
        z += -3.26 * Q.z_dr_0p2_0p4 + 0.320458
    if Q.D2 < 1.83 and Q.n_real_top50 > 19.0:
        z += -0.0273 * (1.83 - Q.D2) * (Q.n_real_top50 - 19.0)
    if Q.D2 < 1.85 and Q.z_7 < 0.0485:
        z += -14.2 * (1.85 - Q.D2) * (0.0485 - Q.z_7)
    if Q.sum_z_dr < 0.0759 and Q.D2 < 3.87:
        z += -3.89 * (0.0759 - Q.sum_z_dr) * (3.87 - Q.D2)
    if Q.log_sum_pt > 6.88 and Q.M2 > 0.0374:
        z += 29.3 * (Q.log_sum_pt - 6.88) * (Q.M2 - 0.0374)
    if Q.n_dr_0p1_0p2 < 19.5 and Q.planar_flow < 0.668:
        z += 0.059 * (19.5 - Q.n_dr_0p1_0p2) * (0.668 - Q.planar_flow)
    if Q.n_dr_0p2_0p4 < 9.76 and Q.sum_z_dr2_top10 < 0.0051:
        z += -30.8 * (9.76 - Q.n_dr_0p2_0p4) * (0.0051 - Q.sum_z_dr2_top10)
    if Q.n_dr_0p2_0p4 < 9.25 and Q.n_dr_0p1_0p2 < 33.1:
        z += 0.00331 * (9.25 - Q.n_dr_0p2_0p4) * (33.1 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.09
    if Q.LHA >= 0.313:
        z += -29.4 * Q.LHA + 9.2022
    if Q.sum_z_dr < 0.0255:
        z += 17.2 * Q.sum_z_dr - 0.4386
    if Q.sum_z_dr >= 0.0882:
        z += 110.0 * Q.sum_z_dr - 9.702
    if Q.sum_z_dr2_top15 < 0.00441:
        z += -31.5 * Q.sum_z_dr2_top15 + 0.21095
    if 0.00441 <= Q.sum_z_dr2_top15 < 0.00454:
        z += -12.1 * Q.sum_z_dr2_top15 + 0.125396
    if 0.00454 <= Q.sum_z_dr2_top15 < 0.00884:
        z += 3.6 * Q.sum_z_dr2_top15 + 0.054118
    if 0.00884 <= Q.sum_z_dr2_top15 < 0.023:
        z += 19.4 * Q.sum_z_dr2_top15 - 0.085554
    if Q.sum_z_dr2_top15 >= 0.023:
        z += -24.8 * Q.sum_z_dr2_top15 + 0.931046
    if Q.lam2 >= 0.000534:
        z += -73.8 * Q.lam2 + 0.0394092
    if Q.mean_eta2 < 0.00613:
        z += -314.0 * Q.mean_eta2 + 1.92482
    if Q.mean_eta2 >= 0.0062:
        z += -155.0 * Q.mean_eta2 + 0.961
    if Q.mean_phi2 < 0.00279:
        z += -162.0 * Q.mean_phi2 + 0.7776
    if 0.00279 <= Q.mean_phi2 < 0.0048:
        z += -299.0 * Q.mean_phi2 + 1.15983
    if Q.mean_phi2 >= 0.0048:
        z += -137.0 * Q.mean_phi2 + 0.38223
    if Q.n_dr_0_0p05 >= 17.5:
        z += -0.0127 * Q.n_dr_0_0p05 + 0.22225
    if Q.n_dr_0p1_0p2 < 10.6:
        z += 0.009 * Q.n_dr_0p1_0p2 + 0.1766
    if 10.6 <= Q.n_dr_0p1_0p2 < 27.6:
        z += -0.016 * Q.n_dr_0p1_0p2 + 0.4416
    if Q.n_dr_0p2_0p4 < 14.3:
        z += -0.0214 * Q.n_dr_0p2_0p4 + 0.30602
    if Q.sj3_dr_max < 0.159:
        z += 1.0 * Q.sj3_dr_max - 0.01372
    if 0.159 <= Q.sj3_dr_max < 0.191:
        z += -4.54 * Q.sj3_dr_max + 0.86714
    if Q.sum_pt >= 917.0:
        z += 0.00215 * Q.sum_pt - 1.97155
    if Q.z_top50_slots < 0.994:
        z += 5.47 * Q.z_top50_slots - 5.43718
    if Q.sum_z_dr < 0.0388 and Q.log_sum_pt > 6.81:
        z += 101.0 * (0.0388 - Q.sum_z_dr) * (Q.log_sum_pt - 6.81)
    if Q.log_sum_pt > 6.8 and Q.mean_eta2 < 0.0115:
        z += -627.0 * (Q.log_sum_pt - 6.8) * (0.0115 - Q.mean_eta2)
    if Q.psi_0p3 > 0.988 and Q.sum_z_dr2_top15 < 0.00578:
        z += 11300.0 * (Q.psi_0p3 - 0.988) * (0.00578 - Q.sum_z_dr2_top15)
    if Q.psi_0p3 > 0.98 and Q.z_1st > 0.137:
        z += 35.8 * (Q.psi_0p3 - 0.98) * (Q.z_1st - 0.137)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.38
    if Q.C2 >= 0.103:
        z += -8.43 * Q.C2 + 0.86829
    if 0.19 <= Q.LHA < 0.327:
        z += -13.5 * Q.LHA + 2.565
    if 0.327 <= Q.LHA < 0.398:
        z += 23.5 * Q.LHA - 9.534
    if Q.LHA >= 0.398:
        z += 70.8 * Q.LHA - 28.3594
    if Q.e2 >= 0.052:
        z += 37.2 * Q.e2 - 1.9344
    if Q.sum_z_dr < 0.0364:
        z += 13.8 * Q.sum_z_dr - 1.1799
    if 0.0364 <= Q.sum_z_dr < 0.0855:
        z += 55.5 * Q.sum_z_dr - 2.69778
    if 0.0855 <= Q.sum_z_dr < 0.0922:
        z += 41.7 * Q.sum_z_dr - 1.51788
    if 0.0922 <= Q.sum_z_dr < 0.116:
        z += -62.3 * Q.sum_z_dr + 8.07092
    if 0.116 <= Q.sum_z_dr < 0.135:
        z += -104.5 * Q.sum_z_dr + 12.96612
    if 0.135 <= Q.sum_z_dr < 0.146:
        z += -202.8 * Q.sum_z_dr + 26.23662
    if Q.sum_z_dr >= 0.146:
        z += -226.8 * Q.sum_z_dr + 29.74062
    if Q.sum_z_dr2_top15 >= 0.00774:
        z += -130.0 * Q.sum_z_dr2_top15 + 1.0062
    if Q.sum_z_dr2_top5 >= 0.00648:
        z += 19.5 * Q.sum_z_dr2_top5 - 0.12636
    if Q.log_sum_pt < 6.86:
        z += 1.6 * Q.log_sum_pt - 13.236
    if 6.86 <= Q.log_sum_pt < 6.91:
        z += 29.0 * Q.log_sum_pt - 201.2
    if 6.91 <= Q.log_sum_pt < 6.96:
        z += 16.2 * Q.log_sum_pt - 112.752
    if Q.log_sum_pt >= 7.11:
        z += -9.54 * Q.log_sum_pt + 67.8294
    if Q.mean_eta2 < 0.00534:
        z += -93.7 * Q.mean_eta2 + 0.500358
    if Q.mean_eta2 >= 0.00575:
        z += -96.7 * Q.mean_eta2 + 0.556025
    if Q.mean_phi2 < 0.00531:
        z += -99.4 * Q.mean_phi2 + 0.527814
    if Q.mean_phi2 >= 0.00567:
        z += -91.6 * Q.mean_phi2 + 0.519372
    if Q.n_dr_0p05_0p1 >= 8.29:
        z += -0.0114 * Q.n_dr_0p05_0p1 + 0.094506
    if Q.n_particles >= 29.5:
        z += 0.00922 * Q.n_particles - 0.27199
    if Q.n_pt_above_5 < 22.1:
        z += 0.0529 * Q.n_pt_above_5 - 1.16909
    if Q.psi_0p2 >= 0.905:
        z += -2.87 * Q.psi_0p2 + 2.59735
    if Q.sum_pt < 956.0:
        z += 0.0418 * Q.sum_pt - 41.0432
    if 956.0 <= Q.sum_pt < 1120.0:
        z += 0.0066 * Q.sum_pt - 7.392
    if Q.sum_pt_top50 < 1060.0:
        z += -0.014 * Q.sum_pt_top50 + 14.84
    if Q.sum_z_dr > 0.0884 and Q.sum_pt < 1240.0:
        z += 0.164 * (Q.sum_z_dr - 0.0884) * (1240.0 - Q.sum_pt)
    if Q.sum_z_dr2_top15 > 0.00476 and Q.soft6_pt > 1.17:
        z += -14.6 * (Q.sum_z_dr2_top15 - 0.00476) * (Q.soft6_pt - 1.17)
    if Q.sum_z_dr2_top15 > 0.00567 and Q.sum_pt_top40 < 1290.0:
        z += 0.373 * (Q.sum_z_dr2_top15 - 0.00567) * (1290.0 - Q.sum_pt_top40)
    if Q.log_sum_pt < 6.96 and Q.C3 < 0.00981:
        z += 257.0 * (6.96 - Q.log_sum_pt) * (0.00981 - Q.C3)
    if Q.log_sum_pt > 7.07 and Q.psi_0p3 > 0.955:
        z += 155.0 * (Q.log_sum_pt - 7.07) * (Q.psi_0p3 - 0.955)
    if Q.n_pt_above_5 < 22.5 and Q.D2 < 3.38:
        z += 0.0209 * (22.5 - Q.n_pt_above_5) * (3.38 - Q.D2)
    if Q.sum_pt < 1060.0 and Q.planar_flow < 0.743:
        z += -0.0062 * (1060.0 - Q.sum_pt) * (0.743 - Q.planar_flow)
    if Q.sum_pt_top50 < 1020.0 and Q.sum_z_dr2_top2 < 0.000834:
        z += 15.8 * (1020.0 - Q.sum_pt_top50) * (0.000834 - Q.sum_z_dr2_top2)
    return max(0.0, z)


def neuron_14(Q):
    z = 4.7
    if Q.D2 < 1.15:
        z += 1.483 * Q.D2 - 2.43042
    if 1.15 <= Q.D2 < 2.21:
        z += 0.512 * Q.D2 - 1.31377
    if 2.21 <= Q.D2 < 3.56:
        z += 0.135 * Q.D2 - 0.4806
    if Q.D2_b2 < 1.5:
        z += -0.175 * Q.D2_b2 + 0.2625
    if Q.LHA < 0.112:
        z += 3.0 * Q.LHA - 2.9228
    if 0.112 <= Q.LHA < 0.271:
        z += -20.8 * Q.LHA - 0.2572
    if 0.271 <= Q.LHA < 0.306:
        z += -3.6 * Q.LHA - 4.9184
    if 0.306 <= Q.LHA < 0.334:
        z += 26.3 * Q.LHA - 14.0678
    if Q.LHA >= 0.334:
        z += -23.8 * Q.LHA + 2.6656
    if Q.N2 < 0.219:
        z += -5.26 * Q.N2 + 1.15194
    if Q.dr_6 < 0.0567:
        z += -4.28 * Q.dr_6 + 0.242676
    if Q.e2 < 0.0266:
        z += -40.8 * Q.e2 + 1.43616
    if 0.0266 <= Q.e2 < 0.0352:
        z += -18.0 * Q.e2 + 0.82968
    if Q.e2 >= 0.0352:
        z += 22.8 * Q.e2 - 0.60648
    if Q.e3 < 8.68e-05:
        z += 6210.0 * Q.e3 - 0.539028
    if Q.sum_z_dr < 0.0402:
        z += 133.3 * Q.sum_z_dr - 8.39106
    if 0.0402 <= Q.sum_z_dr < 0.0668:
        z += 114.0 * Q.sum_z_dr - 7.6152
    if 0.0813 <= Q.sum_z_dr < 0.0984:
        z += -101.0 * Q.sum_z_dr + 8.2113
    if Q.sum_z_dr >= 0.0984:
        z += 3.0 * Q.sum_z_dr - 2.0223
    if Q.log_sum_pt < 7.04:
        z += 5.22 * Q.log_sum_pt - 36.7488
    if Q.mean_eta2 < 0.0108:
        z += -171.0 * Q.mean_eta2 + 2.0007
    if 0.0108 <= Q.mean_eta2 < 0.0117:
        z += -326.0 * Q.mean_eta2 + 3.6747
    if Q.mean_eta2 >= 0.0117:
        z += -155.0 * Q.mean_eta2 + 1.674
    if Q.mean_phi2 < 0.00569:
        z += -182.0 * Q.mean_phi2 + 1.03558
    if Q.mean_phi2 >= 0.00605:
        z += -181.0 * Q.mean_phi2 + 1.09505
    if Q.n_dr_0p05_0p1 < 11.9:
        z += -0.0192 * Q.n_dr_0p05_0p1 + 0.22848
    if Q.n_dr_0p1_0p2 >= 24.8:
        z += -0.0167 * Q.n_dr_0p1_0p2 + 0.41416
    if Q.n_dr_0p2_0p4 >= 12.4:
        z += -0.0402 * Q.n_dr_0p2_0p4 + 0.49848
    if Q.psi_0p1 >= 0.811:
        z += -4.15 * Q.psi_0p1 + 3.36565
    if Q.psi_0p3 >= 0.993:
        z += 58.1 * Q.psi_0p3 - 57.6933
    if Q.sd_rg < 0.158:
        z += -2.45 * Q.sd_rg + 0.3871
    if 0.158 <= Q.sd_rg < 0.203:
        z += 9.65 * Q.sd_rg - 1.5247
    if Q.sd_rg >= 0.203:
        z += 1.69 * Q.sd_rg + 0.09118
    if Q.sj2_dr < 0.316:
        z += 1.51 * Q.sj2_dr - 0.47716
    if 0.172 <= Q.sj3_dr_max < 0.201:
        z += 8.49 * Q.sj3_dr_max - 1.46028
    if Q.sj3_dr_max >= 0.201:
        z += -0.84 * Q.sj3_dr_max + 0.41505
    if Q.soft1_pt < 1.42:
        z += 0.144 * Q.soft1_pt - 0.20448
    if Q.sum_pt_top15 < 888.0:
        z += 0.000449 * Q.sum_pt_top15 - 0.398712
    if Q.sum_pt_top30 >= 1070.0:
        z += 0.000123 * Q.sum_pt_top30 - 0.13161
    if Q.sum_pt_top40 < 1050.0:
        z += -0.00529 * Q.sum_pt_top40 + 5.5545
    if Q.tau1 >= 0.0837:
        z += 47.4 * Q.tau1 - 3.96738
    if Q.tau21_b2 >= 0.173:
        z += 0.417 * Q.tau21_b2 - 0.072141
    if Q.z_dr_0p1_0p2 < 0.15:
        z += -4.06 * Q.z_dr_0p1_0p2 + 0.609
    if 0.071 <= Q.z_dr_0p2_0p4 < 0.156:
        z += -4.77 * Q.z_dr_0p2_0p4 + 0.33867
    if Q.z_dr_0p2_0p4 >= 0.156:
        z += 1.64 * Q.z_dr_0p2_0p4 - 0.66129
    if Q.z_top40_slots < 0.984:
        z += 5.61 * Q.z_top40_slots - 5.52024
    if Q.C2 < 0.0701 and Q.pt_3 > 56.1:
        z += 0.0778 * (0.0701 - Q.C2) * (Q.pt_3 - 56.1)
    if Q.sum_z_dr > 0.0829 and Q.n_real_top40 < 37.7:
        z += 2.63 * (Q.sum_z_dr - 0.0829) * (37.7 - Q.n_real_top40)
    if Q.sum_z_dr2_top10 > -0.000483 and Q.n_real_top40 < 36.5:
        z += -4.53 * (Q.sum_z_dr2_top10 - -0.000483) * (36.5 - Q.n_real_top40)
    if Q.psi_0p3 > 0.993 and Q.dr_6 < 0.055:
        z += -1310.0 * (Q.psi_0p3 - 0.993) * (0.055 - Q.dr_6)
    if Q.sd_rg > 0.15 and Q.n_pt_above_50 > 4.09:
        z += -1.14 * (Q.sd_rg - 0.15) * (Q.n_pt_above_50 - 4.09)
    if Q.tau21_b2 < 0.336 and Q.sum_z_dr2_top10 < 0.0196:
        z += 191.0 * (0.336 - Q.tau21_b2) * (0.0196 - Q.sum_z_dr2_top10)
    if Q.tau21_b2 < 0.333 and Q.sum_z_dr2_top15 < 0.00729:
        z += -1900.0 * (0.333 - Q.tau21_b2) * (0.00729 - Q.sum_z_dr2_top15)
    if Q.tau21_b2 < 0.341 and Q.sum_z_dr2_top15 < 0.00601:
        z += 1470.0 * (0.341 - Q.tau21_b2) * (0.00601 - Q.sum_z_dr2_top15)
    if Q.tau21_b2 < 0.41 and Q.n_pt_above_10 > 21.1:
        z += -0.091 * (0.41 - Q.tau21_b2) * (Q.n_pt_above_10 - 21.1)
    if Q.tau21_b2 < 0.352 and Q.soft1_pt < 2.34:
        z += 0.421 * (0.352 - Q.tau21_b2) * (2.34 - Q.soft1_pt)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.642
    if Q.LHA >= 0.296:
        z += -6.1 * Q.LHA + 1.8056
    if Q.e2 < 0.0292:
        z += -37.5 * Q.e2 + 2.25616
    if 0.0292 <= Q.e2 < 0.043:
        z += -81.2 * Q.e2 + 3.5322
    if 0.043 <= Q.e2 < 0.0435:
        z += -137.0 * Q.e2 + 5.9316
    if Q.e2 >= 0.0435:
        z += -55.8 * Q.e2 + 2.3994
    if Q.e3 < 1.52e-05:
        z += 2960.0 * Q.e3 - 0.94128
    if 1.52e-05 <= Q.e3 < 0.000318:
        z += 5230.0 * Q.e3 - 0.975784
    if Q.e3 >= 0.000318:
        z += 2270.0 * Q.e3 - 0.034504
    if Q.sum_z_dr2_top15 >= 0.00608:
        z += 16.4 * Q.sum_z_dr2_top15 - 0.099712
    if Q.sum_z_dr2_top3 < 0.0016:
        z += 206.0 * Q.sum_z_dr2_top3 - 0.3296
    if Q.sum_z_dr2_top5 < 0.00771:
        z += -30.0 * Q.sum_z_dr2_top5 + 0.2313
    if Q.lam2 < 0.00344:
        z += -171.0 * Q.lam2 + 0.58824
    if 6.91 <= Q.log_sum_pt < 7.01:
        z += -7.4 * Q.log_sum_pt + 51.134
    if Q.log_sum_pt >= 7.01:
        z += 0.92 * Q.log_sum_pt - 7.1892
    if Q.max_dr < 0.289:
        z += 1.13 * Q.max_dr - 0.32657
    if Q.n_dr_0_0p05 >= 19.5:
        z += 0.0132 * Q.n_dr_0_0p05 - 0.2574
    if Q.n_real_top50 >= 24.4:
        z += -0.0215 * Q.n_real_top50 + 0.5246
    if 0.728 <= Q.psi_0p1 < 0.923:
        z += 1.74 * Q.psi_0p1 - 1.26672
    if Q.psi_0p1 >= 0.923:
        z += -6.46 * Q.psi_0p1 + 6.30188
    if Q.sd_rg < 0.161:
        z += -0.09 * Q.sd_rg + 0.27104
    if 0.161 <= Q.sd_rg < 0.196:
        z += -7.33 * Q.sd_rg + 1.43668
    if 0.224 <= Q.sd_rg < 0.28:
        z += 8.46 * Q.sd_rg - 1.89504
    if Q.sd_rg >= 0.28:
        z += -0.8 * Q.sd_rg + 0.69776
    if Q.sj2_dr >= 0.227:
        z += 2.8 * Q.sj2_dr - 0.6356
    if 0.24 <= Q.sj3_dr_max < 0.33:
        z += 1.55 * Q.sj3_dr_max - 0.372
    if Q.sj3_dr_max >= 0.33:
        z += -1.93 * Q.sj3_dr_max + 0.7764
    if Q.soft5_z < 0.000361:
        z += -886.0 * Q.soft5_z + 0.319846
    if Q.sum_pt < 909.0:
        z += -0.00463 * Q.sum_pt + 4.20867
    if Q.sum_pt_top50 < 1020.0:
        z += 0.00593 * Q.sum_pt_top50 - 6.0486
    if Q.tau1 < 0.0744:
        z += 20.3 * Q.tau1 - 1.51032
    if Q.tau21_b2 < 0.347:
        z += -1.58 * Q.tau21_b2 + 0.54826
    if Q.z_dr_0p05_0p1 < 0.623:
        z += 0.345 * Q.z_dr_0p05_0p1 - 0.214935
    if Q.z_dr_0p1_0p2 < 0.0588:
        z += -8.19 * Q.z_dr_0p1_0p2 + 0.481572
    if Q.z_dr_0p2_0p4 < 0.0267:
        z += 9.25 * Q.z_dr_0p2_0p4 - 0.246975
    if Q.z_top20_slots < 0.951:
        z += 2.01 * Q.z_top20_slots - 1.91151
    if Q.psi_0p3 > 0.992 and Q.tau21_b2 < 0.399:
        z += -263.0 * (Q.psi_0p3 - 0.992) * (0.399 - Q.tau21_b2)
    if Q.sj2_dr > 0.195 and Q.sj3_pairmin_over_m > 0.147:
        z += -6.28 * (Q.sj2_dr - 0.195) * (Q.sj3_pairmin_over_m - 0.147)
    if Q.z_dr_0p1_0p2 < 0.0534 and Q.absphi_1 < 0.0334:
        z += -120.0 * (0.0534 - Q.z_dr_0p1_0p2) * (0.0334 - Q.absphi_1)
    if Q.z_dr_0p1_0p2 < 0.133 and Q.soft5_pt < 1.35:
        z += -1.46 * (0.133 - Q.z_dr_0p1_0p2) * (1.35 - Q.soft5_pt)
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
