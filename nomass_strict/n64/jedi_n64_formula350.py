"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network (step 4; no mass observables or exact equivalents), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.5% (the network: 81.1%); same class as the network for 92.4% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
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
  Q.pt_1                   pT of particle 1 [GeV]
  Q.ptdr0_10               pT10 · ΔR(0, 10) [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_8                   pT of particle 8 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.absphi_1               |Δφ| of particle 1
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_12                 ΔR between particle 12 and the hardest particle
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.soft6_dr               ΔR from the jet axis of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau4                   N-subjettiness τ4 (β=1)
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
        z_top5=sum(zs[:5]),
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
        pt_1=pt[1],
        ptdr0_10=pt[10] * math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_8=pt[8],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft6_pt=softp(6, 'pt'),
        soft9_pt=softp(9, 'pt'),
        soft1_z=softp(1, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        sd_zg=softdrop("zg"),
        absphi_1=abs(phi[1]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_12=math.sqrt(dist2(0, 12)) if pt[12] > 0 else 0.0,
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        soft6_dr=softp(6, 'dr'),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau4=tau_n(4),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -0.924
    if Q.LHA < 0.187:
        z += 12.2 * Q.LHA - 6.0756
    if 0.187 <= Q.LHA < 0.309:
        z += 31.1 * Q.LHA - 9.6099
    if Q.dr_0 < 0.0718:
        z += 6.71 * Q.dr_0 - 0.481778
    if Q.e3 < 2.89e-05:
        z += 25150.0 * Q.e3 - 0.59192
    if 2.89e-05 <= Q.e3 < 5.12e-05:
        z += -6050.0 * Q.e3 + 0.30976
    if Q.girth < 0.0504:
        z += -118.8 * Q.girth + 10.80772
    if 0.0504 <= Q.girth < 0.0817:
        z += -154.0 * Q.girth + 12.5818
    if Q.girth2_top15 < 0.00544:
        z += -32.0 * Q.girth2_top15 + 0.18068
    if 0.00544 <= Q.girth2_top15 < 0.00784:
        z += -342.0 * Q.girth2_top15 + 1.86708
    if 0.00784 <= Q.girth2_top15 < 0.0102:
        z += 345.0 * Q.girth2_top15 - 3.519
    if 6.89 <= Q.log_sum_pt < 7.02:
        z += -7.27 * Q.log_sum_pt + 50.0903
    if Q.log_sum_pt >= 7.02:
        z += -4.79 * Q.log_sum_pt + 32.6807
    if Q.n_dr_0p2_0p4 < 17.9:
        z += -0.0299 * Q.n_dr_0p2_0p4 + 0.53521
    if Q.psi_0p3 >= 0.99:
        z += 50.0 * Q.psi_0p3 - 49.5
    if Q.sj2_dr < 0.153:
        z += 3.8 * Q.sj2_dr - 0.703
    if 0.153 <= Q.sj2_dr < 0.175:
        z += -14.4 * Q.sj2_dr + 2.0816
    if 0.175 <= Q.sj2_dr < 0.185:
        z += 1.0 * Q.sj2_dr - 0.6134
    if Q.sj2_dr >= 0.185:
        z += -2.8 * Q.sj2_dr + 0.0896
    if Q.sj3_dr_max < 0.166:
        z += 2.81 * Q.sj3_dr_max - 0.32102
    if 0.166 <= Q.sj3_dr_max < 0.214:
        z += -3.03 * Q.sj3_dr_max + 0.64842
    if Q.sum_pt_top2 >= 409.0:
        z += 0.0009 * Q.sum_pt_top2 - 0.3681
    if Q.tau1 < 0.0446:
        z += 92.6 * Q.tau1 - 4.12996
    if Q.z_dr_0_0p05 < 0.485:
        z += -0.779 * Q.z_dr_0_0p05 + 0.377815
    if Q.n_dr_0p2_0p4 < 17.6 and Q.girth2_top10 > 0.00488:
        z += -3.24 * (17.6 - Q.n_dr_0p2_0p4) * (Q.girth2_top10 - 0.00488)
    if Q.n_dr_0p2_0p4 < 18.1 and Q.log_sum_pt > 6.92:
        z += -0.937 * (18.1 - Q.n_dr_0p2_0p4) * (Q.log_sum_pt - 6.92)
    if Q.n_dr_0p2_0p4 < 17.9 and Q.sum_pt_top50 > 890.0:
        z += 0.000555 * (17.9 - Q.n_dr_0p2_0p4) * (Q.sum_pt_top50 - 890.0)
    if Q.n_dr_0p2_0p4 < 17.7 and Q.tau21_b2 > 0.212:
        z += -0.0421 * (17.7 - Q.n_dr_0p2_0p4) * (Q.tau21_b2 - 0.212)
    if Q.psi_0p3 > 0.996 and Q.n_dr_0p1_0p2 < 26.2:
        z += 2.04 * (Q.psi_0p3 - 0.996) * (26.2 - Q.n_dr_0p1_0p2)
    if Q.sum_pt_top40 > 999.0 and Q.sj2_dr < 0.142:
        z += 0.0449 * (Q.sum_pt_top40 - 999.0) * (0.142 - Q.sj2_dr)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.12
    if Q.C2 < 0.0585:
        z += -9.09 * Q.C2 + 0.531765
    if Q.girth2_top15 < 0.00194:
        z += -279.0 * Q.girth2_top15 + 0.54126
    if Q.girth2_top3 < 0.00682:
        z += -94.0 * Q.girth2_top3 + 0.64108
    if Q.lam2 < 0.00168:
        z += 311.0 * Q.lam2 - 0.52248
    if Q.log_sum_pt < 6.91:
        z += 7.71 * Q.log_sum_pt - 54.9723
    if 6.91 <= Q.log_sum_pt < 6.99:
        z += 37.31 * Q.log_sum_pt - 259.5083
    if 6.99 <= Q.log_sum_pt < 7.13:
        z += 14.71 * Q.log_sum_pt - 101.5343
    if Q.log_sum_pt >= 7.13:
        z += 7.0 * Q.log_sum_pt - 46.562
    if 15.2 <= Q.n_dr_0_0p05 < 29.9:
        z += 0.0371 * Q.n_dr_0_0p05 - 0.56392
    if Q.n_dr_0_0p05 >= 29.9:
        z += -0.0505 * Q.n_dr_0_0p05 + 2.05532
    if Q.n_dr_0p1_0p2 < 7.29:
        z += 0.0693 * Q.n_dr_0p1_0p2 - 0.505197
    if Q.n_particles >= 38.9:
        z += 0.0831 * Q.n_particles - 3.23259
    if Q.psi_0p3 >= 0.997:
        z += -116.0 * Q.psi_0p3 + 115.652
    if 973.0 <= Q.sum_pt < 1050.0:
        z += 0.0251 * Q.sum_pt - 24.4223
    if Q.sum_pt >= 1050.0:
        z += 0.0136 * Q.sum_pt - 12.3473
    if Q.sum_pt_top2 < 573.0:
        z += -0.00214 * Q.sum_pt_top2 + 1.22622
    if Q.sum_pt_top50 < 941.0:
        z += -0.0143 * Q.sum_pt_top50 + 15.444
    if 941.0 <= Q.sum_pt_top50 < 1080.0:
        z += -0.03 * Q.sum_pt_top50 + 30.2177
    if Q.sum_pt_top50 >= 1080.0:
        z += -0.0157 * Q.sum_pt_top50 + 14.7737
    if Q.tau1 < 0.0747:
        z += -32.4 * Q.tau1 + 2.42028
    if Q.tau4 < 0.0237:
        z += 46.6 * Q.tau4 - 1.10442
    if Q.zdr_0 >= 0.0129:
        z += -32.8 * Q.zdr_0 + 0.42312
    if Q.M3 < 0.0334 and Q.psi_0p3 > 0.932:
        z += -330.0 * (0.0334 - Q.M3) * (Q.psi_0p3 - 0.932)
    if Q.N2 < 0.354 and Q.pt2_over_pt0 < 0.875:
        z += 5.32 * (0.354 - Q.N2) * (0.875 - Q.pt2_over_pt0)
    if Q.n_particles > 39.1 and Q.soft1_z < 0.0024:
        z += -21.5 * (Q.n_particles - 39.1) * (0.0024 - Q.soft1_z)
    if Q.n_particles > 38.0 and Q.zdr_0 < 0.0115:
        z += 2.99 * (Q.n_particles - 38.0) * (0.0115 - Q.zdr_0)
    if Q.soft9_pt < 2.4 and Q.dr_11 < 0.14:
        z += 1.49 * (2.4 - Q.soft9_pt) * (0.14 - Q.dr_11)
    if Q.z_dr_0_0p05 > 0.811 and Q.n_dr_0p05_0p1 < 10.7:
        z += -0.494 * (Q.z_dr_0_0p05 - 0.811) * (10.7 - Q.n_dr_0p05_0p1)
    if Q.z_top30_slots > 0.934 and Q.pt1_dr01 > 6.04:
        z += 0.598 * (Q.z_top30_slots - 0.934) * (Q.pt1_dr01 - 6.04)
    if Q.z_top30_slots > 0.933 and Q.ptdr0_2 > 6.97:
        z += 0.698 * (Q.z_top30_slots - 0.933) * (Q.ptdr0_2 - 6.97)
    if Q.z_top30_slots > 0.938 and Q.ptdr0_4 > 4.73:
        z += 0.9 * (Q.z_top30_slots - 0.938) * (Q.ptdr0_4 - 4.73)
    if Q.z_top5 > 0.645 and Q.pt1_dr01 < 27.4:
        z += -0.166 * (Q.z_top5 - 0.645) * (27.4 - Q.pt1_dr01)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.037
    if Q.log_sum_pt >= 6.92:
        z += 5.15 * Q.log_sum_pt - 35.638
    if Q.sd_rg >= 0.274:
        z += -11.1 * Q.sd_rg + 3.0414
    if Q.log_sum_pt > 7.05 and Q.girth2_top15 > 0.00976:
        z += -9540.0 * (Q.log_sum_pt - 7.05) * (Q.girth2_top15 - 0.00976)
    if Q.sum_pt_top15 > 1080.0 and Q.C2 > 0.143:
        z += -41.1 * (Q.sum_pt_top15 - 1080.0) * (Q.C2 - 0.143)
    if Q.sum_pt_top20 > 1120.0 and Q.dr_max_012 > 0.121:
        z += -0.505 * (Q.sum_pt_top20 - 1120.0) * (Q.dr_max_012 - 0.121)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.282
    if Q.lam2 < 0.000636:
        z += -904.0 * Q.lam2 + 0.574944
    if Q.n_dr_0p1_0p2 < 14.0:
        z += -0.0314 * Q.n_dr_0p1_0p2 + 0.4396
    if Q.n_dr_0p2_0p4 < 5.48:
        z += -0.142 * Q.n_dr_0p2_0p4 + 0.77816
    if Q.n_particles < 47.6:
        z += -0.0368 * Q.n_particles + 1.75168
    if Q.n_dr_0p1_0p2 < 10.5 and Q.sum_pt_top40 < 1010.0:
        z += -0.000427 * (10.5 - Q.n_dr_0p1_0p2) * (1010.0 - Q.sum_pt_top40)
    if Q.n_particles < 47.7 and Q.e2 > 0.0133:
        z += -1.42 * (47.7 - Q.n_particles) * (Q.e2 - 0.0133)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.71
    if Q.LHA >= 0.334:
        z += -34.0 * Q.LHA + 11.356
    if Q.e2 < 0.0326:
        z += -0.2 * Q.e2 - 0.32152
    if 0.0326 <= Q.e2 < 0.0444:
        z += 27.8 * Q.e2 - 1.23432
    if Q.e2 >= 0.0559:
        z += -35.6 * Q.e2 + 1.99004
    if Q.girth < 0.0758:
        z += 14.9 * Q.girth - 1.12942
    if Q.girth >= 0.0967:
        z += 56.8 * Q.girth - 5.49256
    if Q.girth2_top10 < 0.00675:
        z += 79.1 * Q.girth2_top10 - 0.533925
    if Q.girth2_top15 < 0.0076:
        z += 54.0 * Q.girth2_top15 + 0.0565
    if 0.0076 <= Q.girth2_top15 < 0.0099:
        z += -203.0 * Q.girth2_top15 + 2.0097
    if Q.n_pt_above_1 >= 28.1:
        z += -0.0176 * Q.n_pt_above_1 + 0.49456
    if Q.pt_entropy < 3.57:
        z += 0.424 * Q.pt_entropy - 1.51368
    if 0.122 <= Q.sj2_dr < 0.182:
        z += 12.3 * Q.sj2_dr - 1.5006
    if Q.sj2_dr >= 0.182:
        z += -2.6 * Q.sj2_dr + 1.2112
    if 1.59 <= Q.soft1_pt < 2.28:
        z += -0.893 * Q.soft1_pt + 1.41987
    if Q.soft1_pt >= 2.28:
        z += 0.227 * Q.soft1_pt - 1.13373
    if Q.sum_pt_top50 < 1040.0:
        z += 0.00767 * Q.sum_pt_top50 - 7.9768
    if Q.z_dr_0p2_0p4 < 0.0475:
        z += 9.77 * Q.z_dr_0p2_0p4 - 0.464075
    if Q.girth < 0.0579 and Q.psi_0p3 > 0.997:
        z += -13300.0 * (0.0579 - Q.girth) * (Q.psi_0p3 - 0.997)
    if Q.girth < 0.0999 and Q.psi_0p3 > 0.997:
        z += 5080.0 * (0.0999 - Q.girth) * (Q.psi_0p3 - 0.997)
    if Q.girth2_top15 > 0.00558 and Q.sj3_pairmin_over_m < 0.41:
        z += -118.0 * (Q.girth2_top15 - 0.00558) * (0.41 - Q.sj3_pairmin_over_m)
    if Q.girth2_top15 < 0.0077 and Q.sum_pt_top40 > 1110.0:
        z += 0.201 * (0.0077 - Q.girth2_top15) * (Q.sum_pt_top40 - 1110.0)
    if Q.n_particles > 24.2 and Q.lam2 > 0.00389:
        z += 1.94 * (Q.n_particles - 24.2) * (Q.lam2 - 0.00389)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.2
    if Q.C2 >= 0.123:
        z += -18.8 * Q.C2 + 2.3124
    if Q.LHA >= 0.433:
        z += -56.3 * Q.LHA + 24.3779
    if Q.M2 >= 0.0621:
        z += 5.55 * Q.M2 - 0.344655
    if Q.girth2_top10 >= 0.00961:
        z += -146.0 * Q.girth2_top10 + 1.40306
    if Q.girth2_top15 < 0.00071:
        z += 513.0 * Q.girth2_top15 - 0.36423
    if Q.log_sum_pt < 6.89:
        z += 13.7 * Q.log_sum_pt - 94.393
    if Q.n_dr_0p1_0p2 >= 7.08:
        z += -0.0271 * Q.n_dr_0p1_0p2 + 0.191868
    if Q.n_dr_0p2_0p4 < 10.8:
        z += -0.0895 * Q.n_dr_0p2_0p4 + 0.9666
    if Q.n_particles >= 46.1:
        z += -0.0411 * Q.n_particles + 1.89471
    if Q.n_real_top50 < 46.7:
        z += -0.0452 * Q.n_real_top50 + 2.11084
    if Q.sum_pt < 909.0:
        z += 0.07415 * Q.sum_pt - 65.9272
    if 909.0 <= Q.sum_pt < 1090.0:
        z += -0.00815 * Q.sum_pt + 8.8835
    if Q.sum_pt_top40 < 1010.0:
        z += -0.00651 * Q.sum_pt_top40 + 6.5751
    if Q.sum_pt_top50 < 947.0:
        z += -0.0184 * Q.sum_pt_top50 + 17.4248
    if Q.z_dr_0p2_0p4 < 0.0611:
        z += 9.21 * Q.z_dr_0p2_0p4 - 0.562731
    if Q.z_top30_slots >= 0.951:
        z += -11.1 * Q.z_top30_slots + 10.5561
    if Q.z_top50_slots < 0.984:
        z += 38.2 * Q.z_top50_slots - 37.5888
    if Q.girth < 0.158 and Q.n_dr_0p1_0p2 > 11.1:
        z += 0.692 * (0.158 - Q.girth) * (Q.n_dr_0p1_0p2 - 11.1)
    if Q.n_particles < 62.4 and Q.C2 < 0.0793:
        z += -0.218 * (62.4 - Q.n_particles) * (0.0793 - Q.C2)
    if Q.psi_0p3 > 0.997 and Q.mean_eta < -0.00325:
        z += -197000.0 * (Q.psi_0p3 - 0.997) * (-0.00325 - Q.mean_eta)
    if Q.sum_pt < 1000.0 and Q.absphi_1 > 0.00176:
        z += -0.156 * (1000.0 - Q.sum_pt) * (Q.absphi_1 - 0.00176)
    if Q.sum_pt < 914.0 and Q.dr0_12 < 0.14:
        z += 0.25 * (914.0 - Q.sum_pt) * (0.14 - Q.dr0_12)
    if Q.sum_pt < 1000.0 and Q.e4 < 6.07e-08:
        z += -582000.0 * (1000.0 - Q.sum_pt) * (6.07e-08 - Q.e4)
    if Q.sum_pt < 1010.0 and Q.ptdr0_10 > 3.46:
        z += -0.00295 * (1010.0 - Q.sum_pt) * (Q.ptdr0_10 - 3.46)
    if Q.sum_pt_top50 < 931.0 and Q.soft6_dr < 0.0369:
        z += 0.957 * (931.0 - Q.sum_pt_top50) * (0.0369 - Q.soft6_dr)
    return max(0.0, z)


def neuron_6(Q):
    z = 2.51
    if Q.C2 >= 0.111:
        z += -13.9 * Q.C2 + 1.5429
    if Q.e2 < 0.0473:
        z += -28.2 * Q.e2 + 1.33386
    if Q.e2 >= 0.0547:
        z += 125.0 * Q.e2 - 6.8375
    if Q.e3 >= 0.000116:
        z += -8710.0 * Q.e3 + 1.01036
    if Q.girth < 0.0955:
        z += 47.1 * Q.girth - 4.49805
    if Q.girth2_top15 < 0.0014:
        z += 146.0 * Q.girth2_top15 - 0.49748
    if 0.0014 <= Q.girth2_top15 < 0.00318:
        z += -474.0 * Q.girth2_top15 + 0.37052
    if 0.00318 <= Q.girth2_top15 < 0.00878:
        z += 203.0 * Q.girth2_top15 - 1.78234
    if Q.n_dr_0p2_0p4 >= 15.1:
        z += -0.055 * Q.n_dr_0p2_0p4 + 0.8305
    if Q.sj2_dr < 0.191:
        z += 11.1 * Q.sj2_dr - 2.1201
    if Q.sj2_zsoft < 0.0441:
        z += -13.9 * Q.sj2_zsoft + 0.61299
    if Q.sum_pt_top50 >= 1070.0:
        z += 0.0033 * Q.sum_pt_top50 - 3.531
    if Q.tau1 < 0.0565:
        z += -98.2 * Q.tau1 + 5.5483
    if Q.z_dr_0p1_0p2 >= 0.324:
        z += -2.92 * Q.z_dr_0p1_0p2 + 0.94608
    if Q.z_dr_0p2_0p4 < 0.0764:
        z += 16.2 * Q.z_dr_0p2_0p4 - 1.23768
    if Q.z_dr_0p2_0p4 >= 0.264:
        z += -24.8 * Q.z_dr_0p2_0p4 + 6.5472
    if Q.LHA < 0.333 and Q.psi_0p3 < 0.966:
        z += -2290.0 * (0.333 - Q.LHA) * (0.966 - Q.psi_0p3)
    if Q.e3 < 0.000324 and Q.log_sum_pt < 7.02:
        z += -28900.0 * (0.000324 - Q.e3) * (7.02 - Q.log_sum_pt)
    if Q.girth < 0.0988 and Q.psi_0p3 < 0.966:
        z += 6330.0 * (0.0988 - Q.girth) * (0.966 - Q.psi_0p3)
    if Q.girth2_top15 < 0.00298 and Q.sum_pt_top40 > 852.0:
        z += -0.776 * (0.00298 - Q.girth2_top15) * (Q.sum_pt_top40 - 852.0)
    if Q.n_dr_0p1_0p2 > 12.9 and Q.psi_0p3 > 0.99:
        z += 5.86 * (Q.n_dr_0p1_0p2 - 12.9) * (Q.psi_0p3 - 0.99)
    if Q.psi_0p2 > 0.907 and Q.sj2_dr < 0.154:
        z += 235.0 * (Q.psi_0p2 - 0.907) * (0.154 - Q.sj2_dr)
    if Q.tau1 < 0.0536 and Q.sum_pt < 971.0:
        z += 0.799 * (0.0536 - Q.tau1) * (971.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.576
    if Q.LHA < 0.302:
        z += -22.1 * Q.LHA + 6.6742
    if Q.e2 < 0.0375:
        z += -38.1 * Q.e2 + 1.42875
    if Q.girth < 0.0435:
        z += 33.2 * Q.girth - 0.74835
    if 0.0435 <= Q.girth < 0.081:
        z += 8.1 * Q.girth + 0.3435
    if 0.081 <= Q.girth < 0.0978:
        z += -59.5 * Q.girth + 5.8191
    if Q.n_dr_0p2_0p4 < 7.23:
        z += 0.0577 * Q.n_dr_0p2_0p4 - 0.85973
    if 7.23 <= Q.n_dr_0p2_0p4 < 14.9:
        z += 0.0241 * Q.n_dr_0p2_0p4 - 0.616802
    if Q.n_dr_0p2_0p4 >= 14.9:
        z += -0.0336 * Q.n_dr_0p2_0p4 + 0.242928
    if Q.psi_0p1 >= 0.854:
        z += -4.43 * Q.psi_0p1 + 3.78322
    if Q.psi_0p3 >= 0.997:
        z += 1990.0 * Q.psi_0p3 - 1984.03
    if Q.tau1 < 0.107:
        z += 80.0 * Q.tau1 - 8.56
    if Q.tau21_b2 < 0.234:
        z += 5.82 * Q.tau21_b2 - 1.36188
    if Q.LHA < 0.333 and Q.z_dr_0p1_0p2 < 0.253:
        z += -77.5 * (0.333 - Q.LHA) * (0.253 - Q.z_dr_0p1_0p2)
    if Q.e2 < 0.0475 and Q.tau21_b2 < 0.235:
        z += -350.0 * (0.0475 - Q.e2) * (0.235 - Q.tau21_b2)
    if Q.n_dr_0p2_0p4 < 15.4 and Q.M2 < 0.114:
        z += 0.906 * (15.4 - Q.n_dr_0p2_0p4) * (0.114 - Q.M2)
    if Q.n_dr_0p2_0p4 < 20.4 and Q.e3 < 8e-05:
        z += -701.0 * (20.4 - Q.n_dr_0p2_0p4) * (8e-05 - Q.e3)
    if Q.n_dr_0p2_0p4 < 14.9 and Q.max_dr < 0.434:
        z += 0.0854 * (14.9 - Q.n_dr_0p2_0p4) * (0.434 - Q.max_dr)
    if Q.n_dr_0p2_0p4 < 14.9 and Q.psi_0p3 > 0.996:
        z += 14.0 * (14.9 - Q.n_dr_0p2_0p4) * (Q.psi_0p3 - 0.996)
    if Q.psi_0p3 > 0.997 and Q.girth2_top10 > 0.00774:
        z += -176000.0 * (Q.psi_0p3 - 0.997) * (Q.girth2_top10 - 0.00774)
    if Q.psi_0p3 > 0.997 and Q.log_sum_pt < 7.06:
        z += -4100.0 * (Q.psi_0p3 - 0.997) * (7.06 - Q.log_sum_pt)
    if Q.psi_0p3 > 0.997 and Q.mean_eta2 < 0.00681:
        z += -227000.0 * (Q.psi_0p3 - 0.997) * (0.00681 - Q.mean_eta2)
    if Q.psi_0p3 > 0.997 and Q.mean_phi2 < 0.00681:
        z += -213000.0 * (Q.psi_0p3 - 0.997) * (0.00681 - Q.mean_phi2)
    if Q.tau1 < 0.107 and Q.z_dr_0p1_0p2 < 0.287:
        z += 216.0 * (0.107 - Q.tau1) * (0.287 - Q.z_dr_0p1_0p2)
    if Q.tau21_b2 < 0.235 and Q.girth2_top15 < 0.00789:
        z += -6320.0 * (0.235 - Q.tau21_b2) * (0.00789 - Q.girth2_top15)
    if Q.tau21_b2 < 0.237 and Q.girth2_top15 < 0.00996:
        z += 6200.0 * (0.237 - Q.tau21_b2) * (0.00996 - Q.girth2_top15)
    if Q.tau21_b2 < 0.243 and Q.n_dr_0p05_0p1 > 2.11:
        z += 0.141 * (0.243 - Q.tau21_b2) * (Q.n_dr_0p05_0p1 - 2.11)
    if Q.tau21_b2 < 0.234 and Q.sj2_dr > 0.15:
        z += 208.0 * (0.234 - Q.tau21_b2) * (Q.sj2_dr - 0.15)
    if Q.tau21_b2 < 0.235 and Q.sj2_dr > 0.193:
        z += -141.0 * (0.235 - Q.tau21_b2) * (Q.sj2_dr - 0.193)
    if Q.tau21_b2 < 0.234 and Q.sum_pt < 972.0:
        z += -0.091 * (0.234 - Q.tau21_b2) * (972.0 - Q.sum_pt)
    if Q.tau21_b2 < 0.236 and Q.sum_pt < 1110.0:
        z += 0.0584 * (0.236 - Q.tau21_b2) * (1110.0 - Q.sum_pt)
    if Q.tau21_b2 < 0.236 and Q.sum_pt_top50 < 1250.0:
        z += -0.0712 * (0.236 - Q.tau21_b2) * (1250.0 - Q.sum_pt_top50)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.37
    if Q.LHA >= 0.21:
        z += -43.8 * Q.LHA + 9.198
    if Q.e2 >= 0.0213:
        z += -45.3 * Q.e2 + 0.96489
    if 0.0353 <= Q.girth < 0.0438:
        z += 67.9 * Q.girth - 2.39687
    if 0.0438 <= Q.girth < 0.0747:
        z += 186.9 * Q.girth - 7.60907
    if 0.0747 <= Q.girth < 0.0965:
        z += 101.8 * Q.girth - 1.2521
    if Q.girth >= 0.0965:
        z += -1.2 * Q.girth + 8.6874
    if Q.girth2_top15 < 0.00717:
        z += 135.0 * Q.girth2_top15 - 0.96795
    if Q.log_sum_pt < 7.14:
        z += 5.55 * Q.log_sum_pt - 39.627
    if Q.psi_0p3 >= 0.978:
        z += -32.0 * Q.psi_0p3 + 31.296
    if Q.sj2_dr >= 0.213:
        z += 5.55 * Q.sj2_dr - 1.18215
    if Q.sj2_zsoft < 0.0454:
        z += 29.0 * Q.sj2_zsoft - 1.3166
    if Q.sum_pt < 1010.0:
        z += -0.0158 * Q.sum_pt + 15.958
    if Q.z_top50_slots >= 0.957:
        z += -22.9 * Q.z_top50_slots + 21.9153
    if Q.dr_0 < 0.0798 and Q.n_dr_0p4_up > -0.251:
        z += 6.52 * (0.0798 - Q.dr_0) * (Q.n_dr_0p4_up - -0.251)
    if Q.girth > 0.0787 and Q.e2 > 0.0433:
        z += 1190.0 * (Q.girth - 0.0787) * (Q.e2 - 0.0433)
    if Q.girth > 0.0683 and Q.sum_pt < 1120.0:
        z += 0.195 * (Q.girth - 0.0683) * (1120.0 - Q.sum_pt)
    if Q.girth > 0.0757 and Q.sum_pt_top30 < 1200.0:
        z += 0.0842 * (Q.girth - 0.0757) * (1200.0 - Q.sum_pt_top30)
    if Q.girth > 0.0733 and Q.sum_pt_top50 < 965.0:
        z += -0.283 * (Q.girth - 0.0733) * (965.0 - Q.sum_pt_top50)
    if Q.girth > 0.0736 and Q.tau2 < 0.0816:
        z += 344.0 * (Q.girth - 0.0736) * (0.0816 - Q.tau2)
    if Q.girth2_top15 < 0.0117 and Q.max_dr < 0.328:
        z += -657.0 * (0.0117 - Q.girth2_top15) * (0.328 - Q.max_dr)
    if Q.girth2_top5 < 0.00679 and Q.sj3_dr23 > 0.264:
        z += 463.0 * (0.00679 - Q.girth2_top5) * (Q.sj3_dr23 - 0.264)
    if Q.n_dr_0p2_0p4 < 16.0 and Q.e3 < 4.44e-05:
        z += -2810.0 * (16.0 - Q.n_dr_0p2_0p4) * (4.44e-05 - Q.e3)
    if Q.n_dr_0p2_0p4 < 17.6 and Q.log_sum_pt < 7.17:
        z += -0.511 * (17.6 - Q.n_dr_0p2_0p4) * (7.17 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 < 20.5 and Q.psi_0p1 < 0.793:
        z += 0.138 * (20.5 - Q.n_dr_0p2_0p4) * (0.793 - Q.psi_0p1)
    if Q.psi_0p3 > 0.971 and Q.n_dr_0p1_0p2 > 16.7:
        z += 1.87 * (Q.psi_0p3 - 0.971) * (Q.n_dr_0p1_0p2 - 16.7)
    if Q.sum_pt < 1050.0 and Q.max_dr > 0.189:
        z += 0.0184 * (1050.0 - Q.sum_pt) * (Q.max_dr - 0.189)
    if Q.sum_pt < 1260.0 and Q.planar_flow < 0.786:
        z += -0.00306 * (1260.0 - Q.sum_pt) * (0.786 - Q.planar_flow)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.03
    if Q.D2 < 2.18:
        z += -0.496 * Q.D2 + 1.08128
    if Q.e2 >= 0.0261:
        z += 43.5 * Q.e2 - 1.13535
    if 0.0989 <= Q.girth < 0.122:
        z += 38.0 * Q.girth - 3.7582
    if Q.girth >= 0.122:
        z += -29.0 * Q.girth + 4.4158
    if Q.girth2_top15 < 0.00599:
        z += 195.0 * Q.girth2_top15 - 1.16805
    if Q.girth2_top15 >= 0.0201:
        z += -102.0 * Q.girth2_top15 + 2.0502
    if Q.sj2_dr < 0.182:
        z += 2.04 * Q.sj2_dr + 0.12324
    if 0.182 <= Q.sj2_dr < 0.26:
        z += -6.34 * Q.sj2_dr + 1.6484
    if Q.sum_pt_top20 >= 759.0:
        z += -0.00255 * Q.sum_pt_top20 + 1.93545
    if Q.sum_pt_top50 < 895.0:
        z += -0.0151 * Q.sum_pt_top50 + 13.5145
    if Q.tau1 < 0.0966:
        z += -53.4 * Q.tau1 + 5.15844
    if Q.z_top40_slots >= 0.967:
        z += -18.6 * Q.z_top40_slots + 17.9862
    if Q.LHA > 0.375 and Q.lam2 < 0.00652:
        z += 4810.0 * (Q.LHA - 0.375) * (0.00652 - Q.lam2)
    if Q.girth < 0.0566 and Q.sum_pt < 1010.0:
        z += 0.467 * (0.0566 - Q.girth) * (1010.0 - Q.sum_pt)
    if Q.girth2_top15 < 0.00632 and Q.n_dr_0p2_0p4 > 7.2:
        z += 18.7 * (0.00632 - Q.girth2_top15) * (Q.n_dr_0p2_0p4 - 7.2)
    if Q.girth2_top15 < 0.00592 and Q.psi_0p3 > 0.976:
        z += 4540.0 * (0.00592 - Q.girth2_top15) * (Q.psi_0p3 - 0.976)
    if Q.girth2_top15 < 0.00675 and Q.z_dr_0p2_0p4 < 0.0696:
        z += 3050.0 * (0.00675 - Q.girth2_top15) * (0.0696 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt < 6.81 and Q.dr_13 < 0.0896:
        z += 299.0 * (6.81 - Q.log_sum_pt) * (0.0896 - Q.dr_13)
    if Q.sum_pt_top50 < 983.0 and Q.C3 < 0.00143:
        z += 9.92 * (983.0 - Q.sum_pt_top50) * (0.00143 - Q.C3)
    if Q.sum_pt_top50 < 988.0 and Q.sum_pt_top20 > 787.0:
        z += -0.000259 * (988.0 - Q.sum_pt_top50) * (Q.sum_pt_top20 - 787.0)
    if Q.z_dr_0p1_0p2 > 0.454 and Q.soft5_z > -0.000103:
        z += 1160.0 * (Q.z_dr_0p1_0p2 - 0.454) * (Q.soft5_z - -0.000103)
    return max(0.0, z)


def neuron_10(Q):
    z = 6.13
    if Q.D2 < 2.44:
        z += -0.61 * Q.D2 + 1.4884
    if Q.D2_b2 < 1.91:
        z += 0.285 * Q.D2_b2 - 0.54435
    if 0.196 <= Q.LHA < 0.334:
        z += -18.3 * Q.LHA + 3.5868
    if Q.LHA >= 0.334:
        z += 25.0 * Q.LHA - 10.8754
    if Q.M3 < 0.0295:
        z += 14.1 * Q.M3 - 0.41595
    if Q.e2 < 0.0294:
        z += 119.0 * Q.e2 - 4.9623
    if 0.0294 <= Q.e2 < 0.0416:
        z += 78.2 * Q.e2 - 3.76278
    if 0.0416 <= Q.e2 < 0.0417:
        z += 198.2 * Q.e2 - 8.75478
    if Q.e2 >= 0.0417:
        z += 79.2 * Q.e2 - 3.79248
    if Q.e3 < 5.77e-05:
        z += -18800.0 * Q.e3 + 1.08476
    if Q.girth < 0.0984:
        z += 22.1 * Q.girth - 2.652
    if 0.0984 <= Q.girth < 0.12:
        z += -52.6 * Q.girth + 4.69848
    if 0.12 <= Q.girth < 0.139:
        z += -74.7 * Q.girth + 7.35048
    if Q.girth >= 0.139:
        z += -125.5 * Q.girth + 14.41168
    if Q.girth2_top2 < 0.00427:
        z += -145.0 * Q.girth2_top2 + 0.61915
    if Q.girth2_top5 < 0.0242:
        z += 24.7 * Q.girth2_top5 - 0.59774
    if Q.log_sum_pt < 6.84:
        z += 7.28 * Q.log_sum_pt - 49.7952
    if Q.n_dr_0p1_0p2 >= 21.4:
        z += 0.0327 * Q.n_dr_0p1_0p2 - 0.69978
    if Q.n_dr_0p2_0p4 < 19.7:
        z += -0.0284 * Q.n_dr_0p2_0p4 + 0.55948
    if Q.psi_0p1 >= 0.916:
        z += 9.24 * Q.psi_0p1 - 8.46384
    if Q.pt_entropy >= 2.07:
        z += -0.469 * Q.pt_entropy + 0.97083
    if Q.sum_pt_top20 < 889.0:
        z += -0.00385 * Q.sum_pt_top20 + 3.42265
    if 402.0 <= Q.sum_pt_top5 < 793.0:
        z += -0.00416 * Q.sum_pt_top5 + 1.67232
    if Q.sum_pt_top5 >= 793.0:
        z += -0.00791 * Q.sum_pt_top5 + 4.64607
    if Q.tau1 >= 0.197:
        z += -32.4 * Q.tau1 + 6.3828
    if Q.tau21_b2 < 0.203:
        z += 4.01 * Q.tau21_b2 - 0.81403
    if Q.z_dr_0p1_0p2 < 0.12:
        z += 4.07 * Q.z_dr_0p1_0p2 - 0.4884
    if Q.e2 > 0.0278 and Q.soft1_pt > 1.12:
        z += -13.2 * (Q.e2 - 0.0278) * (Q.soft1_pt - 1.12)
    if Q.girth < 0.119 and Q.pt1_dr01 > 6.63:
        z += 0.467 * (0.119 - Q.girth) * (Q.pt1_dr01 - 6.63)
    if Q.girth > 0.0975 and Q.sum_pt_top50 < 1160.0:
        z += 0.128 * (Q.girth - 0.0975) * (1160.0 - Q.sum_pt_top50)
    if Q.girth < 0.109 and Q.sum_pt_top50 < 1010.0:
        z += 0.0658 * (0.109 - Q.girth) * (1010.0 - Q.sum_pt_top50)
    if Q.psi_0p3 > 0.998 and Q.eccentricity > 0.526:
        z += -550.0 * (Q.psi_0p3 - 0.998) * (Q.eccentricity - 0.526)
    if Q.sj2_dr > 0.111 and Q.M2 > 0.041:
        z += -53.2 * (Q.sj2_dr - 0.111) * (Q.M2 - 0.041)
    if Q.tau1 > 0.195 and Q.sum_pt > 1170.0:
        z += -5.44 * (Q.tau1 - 0.195) * (Q.sum_pt - 1170.0)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.302
    if Q.LHA < 0.307:
        z += 19.0 * Q.LHA - 5.833
    if Q.dr_0 < 0.0624:
        z += -13.4 * Q.dr_0 + 0.83616
    if Q.e2 < 0.0245:
        z += -88.6 * Q.e2 + 1.16144
    if 0.0245 <= Q.e2 < 0.0434:
        z += 53.4 * Q.e2 - 2.31756
    if Q.e3 < 3.66e-05:
        z += 21500.0 * Q.e3 - 0.7869
    if Q.girth < 0.0876:
        z += -55.6 * Q.girth + 4.87056
    if Q.girth2_top10 < 0.00469:
        z += -211.0 * Q.girth2_top10 + 0.98959
    if Q.girth2_top15 < 0.00615:
        z += 212.0 * Q.girth2_top15 - 0.6958
    if 0.00615 <= Q.girth2_top15 < 0.00805:
        z += -320.0 * Q.girth2_top15 + 2.576
    if Q.psi_0p2 >= 0.995:
        z += 82.7 * Q.psi_0p2 - 82.2865
    if Q.psi_0p3 >= 0.99:
        z += 33.2 * Q.psi_0p3 - 32.868
    if Q.sj3_dr_max < 0.171:
        z += 6.46 * Q.sj3_dr_max - 1.10466
    if Q.girth < 0.0926 and Q.tau4 > 0.00981:
        z += -808.0 * (0.0926 - Q.girth) * (Q.tau4 - 0.00981)
    if Q.n_dr_0p1_0p2 < 20.6 and Q.planar_flow < 0.562:
        z += 0.0768 * (20.6 - Q.n_dr_0p1_0p2) * (0.562 - Q.planar_flow)
    if Q.n_dr_0p2_0p4 < 9.9 and Q.girth2_top10 < 0.00609:
        z += -24.2 * (9.9 - Q.n_dr_0p2_0p4) * (0.00609 - Q.girth2_top10)
    if Q.n_dr_0p2_0p4 < 9.68 and Q.n_dr_0p1_0p2 < 34.6:
        z += 0.00498 * (9.68 - Q.n_dr_0p2_0p4) * (34.6 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.647
    if Q.LHA < 0.185:
        z += 41.0 * Q.LHA - 8.9597
    if 0.185 <= Q.LHA < 0.244:
        z += 23.3 * Q.LHA - 5.6852
    if Q.M2 < 0.0589:
        z += 30.1 * Q.M2 - 1.77289
    if Q.e2 < 0.0252:
        z += 189.0 * Q.e2 - 4.7628
    if Q.e2 >= 0.0575:
        z += 52.0 * Q.e2 - 2.99
    if Q.e3 < 4.57e-05:
        z += -48500.0 * Q.e3 + 2.21645
    if Q.girth < 0.0609:
        z += -106.0 * Q.girth + 6.4554
    if Q.girth2_top15 < 0.00213:
        z += 477.0 * Q.girth2_top15 + 0.50427
    if 0.00213 <= Q.girth2_top15 < 0.00582:
        z += -412.0 * Q.girth2_top15 + 2.39784
    if Q.n_dr_0p2_0p4 < 10.1:
        z += -0.132 * Q.n_dr_0p2_0p4 + 1.3332
    if Q.pt_dispersion >= 0.324:
        z += 2.37 * Q.pt_dispersion - 0.76788
    if Q.sj3_dr_max < 0.154:
        z += 14.4 * Q.sj3_dr_max - 1.6876
    if 0.154 <= Q.sj3_dr_max < 0.204:
        z += -10.6 * Q.sj3_dr_max + 2.1624
    if Q.sum_pt >= 906.0:
        z += -0.0153 * Q.sum_pt + 13.8618
    if Q.LHA > 0.314 and Q.log_sum_pt > 6.85:
        z += 109.0 * (Q.LHA - 0.314) * (Q.log_sum_pt - 6.85)
    if Q.girth < 0.0507 and Q.log_sum_pt > 6.81:
        z += 459.0 * (0.0507 - Q.girth) * (Q.log_sum_pt - 6.81)
    if Q.psi_0p2 > 0.972 and Q.n_dr_0p1_0p2 < 21.7:
        z += 2.97 * (Q.psi_0p2 - 0.972) * (21.7 - Q.n_dr_0p1_0p2)
    if Q.psi_0p3 > 0.989 and Q.C3 < 0.00762:
        z += -5940.0 * (Q.psi_0p3 - 0.989) * (0.00762 - Q.C3)
    if Q.psi_0p3 > 0.989 and Q.girth2_top15 < 0.00655:
        z += 26200.0 * (Q.psi_0p3 - 0.989) * (0.00655 - Q.girth2_top15)
    if Q.sj3_dr_max < 0.264 and Q.girth2_top10 > 0.00487:
        z += -8320.0 * (0.264 - Q.sj3_dr_max) * (Q.girth2_top10 - 0.00487)
    if Q.sum_pt < 1160.0 and Q.M2 < 0.0594:
        z += 0.129 * (1160.0 - Q.sum_pt) * (0.0594 - Q.M2)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.12
    if Q.LHA >= 0.32:
        z += 35.4 * Q.LHA - 11.328
    if Q.e2 >= 0.0411:
        z += 36.1 * Q.e2 - 1.48371
    if Q.e3 < 0.000199:
        z += 1930.0 * Q.e3 + 0.41605
    if 0.000199 <= Q.e3 < 0.000531:
        z += -2410.0 * Q.e3 + 1.27971
    if 0.086 <= Q.girth < 0.097:
        z += -62.3 * Q.girth + 5.3578
    if 0.097 <= Q.girth < 0.125:
        z += -167.3 * Q.girth + 15.5428
    if 0.125 <= Q.girth < 0.156:
        z += -200.6 * Q.girth + 19.7053
    if Q.girth >= 0.156:
        z += -343.6 * Q.girth + 42.0133
    if Q.girth2_top5 >= 0.0053:
        z += 30.8 * Q.girth2_top5 - 0.16324
    if Q.log_sum_pt < 6.9:
        z += 45.6 * Q.log_sum_pt - 316.734
    if 6.9 <= Q.log_sum_pt < 6.96:
        z += 34.9 * Q.log_sum_pt - 242.904
    if Q.n_dr_0p1_0p2 >= 17.4:
        z += 0.0453 * Q.n_dr_0p1_0p2 - 0.78822
    if Q.sum_pt >= 1150.0:
        z += -0.00506 * Q.sum_pt + 5.819
    if Q.sum_pt_top15 < 712.0:
        z += 0.00403 * Q.sum_pt_top15 - 2.86936
    if Q.sum_pt_top50 < 957.0:
        z += -0.0367 * Q.sum_pt_top50 + 37.3818
    if 957.0 <= Q.sum_pt_top50 < 1050.0:
        z += -0.0243 * Q.sum_pt_top50 + 25.515
    if Q.e3 < 0.000565 and Q.psi_0p3 < 0.992:
        z += -15600.0 * (0.000565 - Q.e3) * (0.992 - Q.psi_0p3)
    if Q.e3 < 0.000183 and Q.pt_9 < 41.1:
        z += -120.0 * (0.000183 - Q.e3) * (41.1 - Q.pt_9)
    if Q.girth > 0.157 and Q.n_pt_above_50 > 4.84:
        z += 56.8 * (Q.girth - 0.157) * (Q.n_pt_above_50 - 4.84)
    if Q.girth > 0.139 and Q.soft6_pt > 3.21:
        z += -156.0 * (Q.girth - 0.139) * (Q.soft6_pt - 3.21)
    if Q.girth > 0.141 and Q.soft6_pt > 4.23:
        z += -375.0 * (Q.girth - 0.141) * (Q.soft6_pt - 4.23)
    if Q.girth > 0.138 and Q.soft6_z > 0.00296:
        z += 61800.0 * (Q.girth - 0.138) * (Q.soft6_z - 0.00296)
    if Q.girth > 0.0965 and Q.sum_pt < 1180.0:
        z += 0.377 * (Q.girth - 0.0965) * (1180.0 - Q.sum_pt)
    if Q.girth2_top15 > 0.00359 and Q.soft6_pt > 1.66:
        z += -24.5 * (Q.girth2_top15 - 0.00359) * (Q.soft6_pt - 1.66)
    if Q.sum_pt < 1180.0 and Q.log_sum_pt < 6.81:
        z += 0.0189 * (1180.0 - Q.sum_pt) * (6.81 - Q.log_sum_pt)
    if Q.sum_pt < 1070.0 and Q.sd_zg > 0.152:
        z += -0.0116 * (1070.0 - Q.sum_pt) * (Q.sd_zg - 0.152)
    return max(0.0, z)


def neuron_14(Q):
    z = 3.18
    if Q.C2 < 0.0773:
        z += 25.5 * Q.C2 - 1.97115
    if Q.LHA < 0.26:
        z += -29.4 * Q.LHA + 7.644
    if Q.girth < 0.0702:
        z += 130.0 * Q.girth - 9.126
    if 0.0873 <= Q.girth < 0.121:
        z += -130.0 * Q.girth + 11.349
    if Q.girth >= 0.121:
        z += -1190.0 * Q.girth + 139.609
    if Q.n_dr_0p2_0p4 >= 11.0:
        z += -0.0541 * Q.n_dr_0p2_0p4 + 0.5951
    if Q.sd_rg < 0.145:
        z += -6.35 * Q.sd_rg + 0.92075
    if 0.158 <= Q.sd_rg < 0.201:
        z += 16.4 * Q.sd_rg - 2.5912
    if Q.sd_rg >= 0.201:
        z += -2.2 * Q.sd_rg + 1.1474
    if Q.sj3_dr_max < 0.262:
        z += 6.18 * Q.sj3_dr_max - 2.3484
    if 0.262 <= Q.sj3_dr_max < 0.38:
        z += -4.52 * Q.sj3_dr_max + 0.455
    if Q.sj3_dr_max >= 0.38:
        z += -10.7 * Q.sj3_dr_max + 2.8034
    if Q.tau1 < 0.152:
        z += 24.9 * Q.tau1 - 3.7848
    if Q.tau21_b2 < 0.346:
        z += -7.54 * Q.tau21_b2 + 2.60884
    if Q.z_dr_0_0p05 >= 0.617:
        z += 3.83 * Q.z_dr_0_0p05 - 2.36311
    if Q.C2 < 0.0718 and Q.pt_3 > 66.9:
        z += 0.157 * (0.0718 - Q.C2) * (Q.pt_3 - 66.9)
    if Q.girth > 0.0753 and Q.max_dr < 0.386:
        z += 182.0 * (Q.girth - 0.0753) * (0.386 - Q.max_dr)
    if Q.girth > 0.12 and Q.max_dr < 0.398:
        z += -8390.0 * (Q.girth - 0.12) * (0.398 - Q.max_dr)
    if Q.girth > 0.0854 and Q.n_real_top40 < 36.2:
        z += 28.3 * (Q.girth - 0.0854) * (36.2 - Q.n_real_top40)
    if Q.girth2_top10 > -0.000908 and Q.n_real_top40 < 36.8:
        z += -5.79 * (Q.girth2_top10 - -0.000908) * (36.8 - Q.n_real_top40)
    if Q.psi_0p3 > 0.956 and Q.pt_1 < 184.0:
        z += -0.059 * (Q.psi_0p3 - 0.956) * (184.0 - Q.pt_1)
    if Q.psi_0p3 > 0.995 and Q.pt_8 > 9.3:
        z += 2.17 * (Q.psi_0p3 - 0.995) * (Q.pt_8 - 9.3)
    if Q.tau21_b2 < 0.315 and Q.girth2_top15 < 0.00759:
        z += -591.0 * (0.315 - Q.tau21_b2) * (0.00759 - Q.girth2_top15)
    if Q.tau21_b2 < 0.342 and Q.sum_pt_top50 < 1150.0:
        z += -0.033 * (0.342 - Q.tau21_b2) * (1150.0 - Q.sum_pt_top50)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.479
    if Q.e3 < 2.89e-05:
        z += 20600.0 * Q.e3 - 0.59534
    if Q.girth2_top3 < 0.00107:
        z += 580.0 * Q.girth2_top3 - 0.6206
    if Q.girth2_top5 < 0.00819:
        z += -137.0 * Q.girth2_top5 + 1.12203
    if 6.92 <= Q.log_sum_pt < 7.02:
        z += -8.82 * Q.log_sum_pt + 61.0344
    if Q.log_sum_pt >= 7.02:
        z += -1.1 * Q.log_sum_pt + 6.84
    if Q.n_dr_0p2_0p4 >= 7.11:
        z += -0.06 * Q.n_dr_0p2_0p4 + 0.4266
    if Q.sd_rg < 0.187:
        z += -3.08 * Q.sd_rg - 0.03034
    if 0.187 <= Q.sd_rg < 0.281:
        z += 6.45 * Q.sd_rg - 1.81245
    if 0.137 <= Q.sj2_dr < 0.216:
        z += -5.44 * Q.sj2_dr + 0.74528
    if Q.sj2_dr >= 0.216:
        z += 5.66 * Q.sj2_dr - 1.65232
    if Q.sum_pt_top30 >= 838.0:
        z += 0.0035 * Q.sum_pt_top30 - 2.933
    if Q.tau1 >= 0.186:
        z += -31.3 * Q.tau1 + 5.8218
    if Q.z_dr_0p1_0p2 < 0.0897:
        z += -4.51 * Q.z_dr_0p1_0p2 + 0.404547
    if Q.z_dr_0p2_0p4 < 0.0324:
        z += 21.4 * Q.z_dr_0p2_0p4 - 0.69336
    if Q.sj2_dr > 0.212 and Q.dr_4 < 0.0292:
        z += -243.0 * (Q.sj2_dr - 0.212) * (0.0292 - Q.dr_4)
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
