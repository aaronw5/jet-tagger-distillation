"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no mass observables), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.2% (the network: 81.1%); same class as the network for 93.4% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_14                  pT of particle 14 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_pt              pT [GeV] of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_1                    pT of particle 1 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_z               pT share of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft5_dr0              ΔR between the hardest and the 5. softest real particle (0 if among the 15 hardest)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.soft5_dr               ΔR from the jet axis of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta2              pT-weighted mean Δη²
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
  Q.tau3                   N-subjettiness τ3 (β=1)
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
        z_top5=sum(zs[:5]),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_90pct=ncum(0.9),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_11=pt[11],
        pt_14=pt[14],
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_6=pt[6],
        soft1_pt=softp(1, 'pt'),
        soft10_pt=softp(10, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft9_pt=softp(9, 'pt'),
        z_1=z[1],
        z_6=z[6],
        soft1_z=softp(1, 'z'),
        soft10_z=softp(10, 'z'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        soft5_dr0=softp(5, 'dr0'),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        soft5_dr=softp(5, 'dr'),
        eta_1=eta[1],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
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
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
    )


def neuron_0(Q):
    z = -1.280414
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.03522007 * Q.n_dr_0p2_0p4 + 0.6339613
    if Q.girth2_top40 < 0.003113918:
        z += 288.6496 * Q.girth2_top40 - 0.1576941
    if 0.003113918 <= Q.girth2_top40 < 0.008840538:
        z += -129.4196 * Q.girth2_top40 + 1.144139
    if Q.girth2 < 0.008190222:
        z += -143.8339 * Q.girth2 + 1.178032
    if Q.e2_sq < 0.00616708:
        z += -464.1493 * Q.e2_sq + 3.694524
    if 0.00616708 <= Q.e2_sq < 0.006936725:
        z += -1011.101 * Q.e2_sq + 7.06762
    if 0.006936725 <= Q.e2_sq < 0.007872294:
        z += -468.027 * Q.e2_sq + 3.300464
    if 0.007872294 <= Q.e2_sq < 0.009606007:
        z += 221.4794 * Q.e2_sq - 2.127533
    if 1002.379 <= Q.sum_pt < 1115.723:
        z += -0.01149419 * Q.sum_pt + 11.52153
    if 1115.723 <= Q.sum_pt < 1167.447:
        z += -0.005301136 * Q.sum_pt + 4.611798
    if Q.sum_pt >= 1167.447:
        z += -0.0073532 * Q.sum_pt + 7.007473
    if 858.8262 <= Q.sum_pt_top40 < 1024.942:
        z += 0.005354839 * Q.sum_pt_top40 - 4.598876
    if Q.sum_pt_top40 >= 1024.942:
        z += 0.002368875 * Q.sum_pt_top40 - 1.538435
    if Q.psi_0p3 >= 0.9943058:
        z += 83.74023 * Q.psi_0p3 - 83.2634
    if Q.lam1 < 0.005913555:
        z += 161.2314 * Q.lam1 - 0.9534507
    if Q.tau1 < 0.0705748:
        z += 19.0985 * Q.tau1 - 1.347873
    if Q.psi_0p2 >= 0.9087063:
        z += -4.286581 * Q.psi_0p2 + 3.895243
    if Q.e2 < 0.01879315:
        z += -28.16457 * Q.e2 + 0.5293009
    if Q.sum_pt_top30 < 1073.473:
        z += -0.0003213774 * Q.sum_pt_top30 + 0.5424645
    if 1073.473 <= Q.sum_pt_top30 < 1191.938:
        z += -0.00166695 * Q.sum_pt_top30 + 1.986901
    if Q.girth2_top30 < 0.006363916:
        z += 101.3587 * Q.girth2_top30 - 0.4969726
    if 0.006363916 <= Q.girth2_top30 < 0.008376291:
        z += -73.57757 * Q.girth2_top30 + 0.6163071
    if Q.girth2_top50 < 0.008124776:
        z += -190.5342 * Q.girth2_top50 + 1.548047
    if Q.girth2 < 0.008190222 and Q.girth2_top15 < 0.006142802:
        z += -19573.84 * (0.008190222 - Q.girth2) * (0.006142802 - Q.girth2_top15)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.log_sum_pt > 6.915514:
        z += -0.1538483 * (18.0 - Q.n_dr_0p2_0p4) * (Q.log_sum_pt - 6.915514)
    if Q.girth2_top50 < 0.005852839 and Q.sum_pt < 1260.541:
        z += -1.568748 * (0.005852839 - Q.girth2_top50) * (1260.541 - Q.sum_pt)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.090839
    if Q.n_particles >= 38.0:
        z += 0.07494003 * Q.n_particles - 2.847721
    if Q.log_sum_pt < 6.910131:
        z += 8.799349 * Q.log_sum_pt - 62.82116
    if 6.910131 <= Q.log_sum_pt < 6.98945:
        z += 41.74911 * Q.log_sum_pt - 290.5083
    if 6.98945 <= Q.log_sum_pt < 7.139296:
        z += 18.13917 * Q.log_sum_pt - 125.4878
    if Q.log_sum_pt >= 7.139296:
        z += 9.339825 * Q.log_sum_pt - 62.66669
    if Q.sum_pt_top50 < 934.2416:
        z += -0.01335465 * Q.sum_pt_top50 + 14.40959
    if 934.2416 <= Q.sum_pt_top50 < 1078.994:
        z += -0.02662153 * Q.sum_pt_top50 + 26.80406
    if Q.sum_pt_top50 >= 1078.994:
        z += -0.01326688 * Q.sum_pt_top50 + 12.39447
    if Q.girth2_top15 < 0.002197765:
        z += -207.5173 * Q.girth2_top15 + 0.4560742
    if Q.psi_0p3 >= 0.9966167:
        z += -130.5255 * Q.psi_0p3 + 130.0839
    if Q.tau21 < 0.2140276:
        z += -5.966698 * Q.tau21 + 1.277038
    if Q.lam2 < 0.0007143144:
        z += 863.4247 * Q.lam2 - 0.6167568
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.07452324 * Q.n_dr_0p2_0p4 - 0.5216627
    if Q.pt_11 < 29.04688:
        z += 0.01743152 * Q.pt_11 - 0.5063312
    if 972.0419 <= Q.sum_pt < 1052.889:
        z += 0.02271044 * Q.sum_pt - 22.0755
    if Q.sum_pt >= 1052.889:
        z += 0.009794354 * Q.sum_pt - 8.476289
    if Q.girth2_top5 < 0.0006570502:
        z += -693.4711 * Q.girth2_top5 + 0.4556453
    if Q.sum_pt_top2 < 605.875:
        z += -0.001450911 * Q.sum_pt_top2 + 0.8790709
    if 15.0 <= Q.n_dr_0_0p05 < 30.0:
        z += 0.04915266 * Q.n_dr_0_0p05 - 0.73729
    if Q.n_dr_0_0p05 >= 30.0:
        z += -0.04676526 * Q.n_dr_0_0p05 + 2.140248
    if Q.n_dr_0p1_0p2 < 7.0:
        z += 0.08310739 * Q.n_dr_0p1_0p2 - 0.5817517
    if Q.soft9_pt < 2.5:
        z += -0.1643109 * Q.soft9_pt + 0.4107772
    if Q.tau1 >= 0.1219132:
        z += -6.021176 * Q.tau1 + 0.7340611
    if Q.girth2_top30 < 0.004763596:
        z += -243.6967 * Q.girth2_top30 + 1.54889
    if 0.004763596 <= Q.girth2_top30 < 0.007463985:
        z += -143.6893 * Q.girth2_top30 + 1.072495
    if Q.e2_sq < 0.009606007:
        z += 92.26755 * Q.e2_sq - 0.8863227
    if Q.sum_pt_top40 < 1225.842:
        z += -0.003700064 * Q.sum_pt_top40 + 4.535694
    if Q.e2 < 0.02210818:
        z += 57.65485 * Q.e2 - 1.274644
    if Q.girth2_top20 < 0.001153063:
        z += -436.4973 * Q.girth2_top20 + 0.5033088
    if Q.girth2 < 0.001784491:
        z += 389.0371 * Q.girth2 - 0.6942332
    if Q.z_top30_slots >= 0.9564984:
        z += 18.16402 * Q.z_top30_slots - 17.37386
    if Q.LHA < 0.2454112:
        z += -8.386306 * Q.LHA + 2.058094
    if Q.zdr_0 >= 0.01240028:
        z += -18.82267 * Q.zdr_0 + 0.2334063
    if Q.n_particles > 38.0 and Q.zdr_0 < 0.009970338:
        z += 3.487009 * (Q.n_particles - 38.0) * (0.009970338 - Q.zdr_0)
    if Q.n_particles > 38.0 and Q.soft1_z < 0.002181998:
        z += -26.60028 * (Q.n_particles - 38.0) * (0.002181998 - Q.soft1_z)
    if Q.z_top5 > 0.6551948 and Q.pt1_dr01 < 28.39396:
        z += -0.1692261 * (Q.z_top5 - 0.6551948) * (28.39396 - Q.pt1_dr01)
    if Q.girth2_top15 < 0.002197765 and Q.psi_0p3 > 0.9966167:
        z += 75615.48 * (0.002197765 - Q.girth2_top15) * (Q.psi_0p3 - 0.9966167)
    if Q.z_top30_slots > 0.9341838 and Q.pt1_dr01 > 5.351077:
        z += 0.4321889 * (Q.z_top30_slots - 0.9341838) * (Q.pt1_dr01 - 5.351077)
    if Q.n_particles > 38.0 and Q.tau32 > 0.3293142:
        z += 0.02149977 * (Q.n_particles - 38.0) * (Q.tau32 - 0.3293142)
    if Q.M3 < 0.03457336 and Q.psi_0p3 > 0.9299135:
        z += -279.3354 * (0.03457336 - Q.M3) * (Q.psi_0p3 - 0.9299135)
    if Q.z_dr_0_0p05 > 0.8103116 and Q.n_dr_0p05_0p1 < 10.0:
        z += -0.4186745 * (Q.z_dr_0_0p05 - 0.8103116) * (10.0 - Q.n_dr_0p05_0p1)
    if Q.z_top30_slots > 0.9564984 and Q.ptdr0_3 < 10.17512:
        z += -1.700912 * (Q.z_top30_slots - 0.9564984) * (10.17512 - Q.ptdr0_3)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.6369432
    if 6.930088 <= Q.log_sum_pt < 7.017258:
        z += 8.811402 * Q.log_sum_pt - 61.06379
    if 7.017258 <= Q.log_sum_pt < 7.062574:
        z += 3.148297 * Q.log_sum_pt - 21.32432
    if Q.log_sum_pt >= 7.062574:
        z += -5.097057 * Q.log_sum_pt + 36.9091
    if Q.girth2 < 0.003638856:
        z += 719.2401 * Q.girth2 - 4.187436
    if 0.003638856 <= Q.girth2 < 0.006941794:
        z += 475.4025 * Q.girth2 - 3.300146
    if Q.girth2_top15 < 0.01563836:
        z += -37.91886 * Q.girth2_top15 + 0.5929886
    if 1007.788 <= Q.sum_pt < 1085.125:
        z += 0.006634569 * Q.sum_pt - 6.686242
    if Q.sum_pt >= 1085.125:
        z += 0.001855652 * Q.sum_pt - 1.50052
    if Q.girth2_top50 < 0.00634935:
        z += -395.8882 * Q.girth2_top50 + 2.513633
    if Q.sum_pt_top50 >= 1003.544:
        z += -0.005569454 * Q.sum_pt_top50 + 5.589193
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.008377022 * Q.sum_pt_top40 - 8.960659
    if 800.732 <= Q.sum_pt_top30 < 1011.524:
        z += -0.00162814 * Q.sum_pt_top30 + 1.303704
    if Q.sum_pt_top30 >= 1011.524:
        z += -0.005983152 * Q.sum_pt_top30 + 5.708903
    if Q.z_top30_slots >= 0.9203881:
        z += 5.141624 * Q.z_top30_slots - 4.73229
    if Q.LHA >= 0.3719813:
        z += -175.7084 * Q.LHA + 65.36025
    if Q.lam1 < 0.01174405:
        z += -67.36192 * Q.lam1 + 0.7911018
    if Q.log_sum_pt > 6.903423 and Q.girth2_top15 < 0.02146578:
        z += 196.2624 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.girth2_top15)
    if Q.log_sum_pt > 7.062574 and Q.girth2_top30 > 0.01215787:
        z += -20158.11 * (Q.log_sum_pt - 7.062574) * (Q.girth2_top30 - 0.01215787)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p1 < 0.8747961:
        z += -25.58815 * (Q.log_sum_pt - 6.903423) * (0.8747961 - Q.psi_0p1)
    if Q.sum_pt_top30 > 1111.245 and Q.z_dr_0p1_0p2 < 0.1203437:
        z += 0.03259707 * (Q.sum_pt_top30 - 1111.245) * (0.1203437 - Q.z_dr_0p1_0p2)
    if Q.sum_pt_top15 > 1003.329 and Q.lam2 > 0.001163277:
        z += -8.428369 * (Q.sum_pt_top15 - 1003.329) * (Q.lam2 - 0.001163277)
    if Q.sum_pt_top50 > 1003.544 and Q.z_dr_0p1_0p2 > 0.003353111:
        z += 0.02343551 * (Q.sum_pt_top50 - 1003.544) * (Q.z_dr_0p1_0p2 - 0.003353111)
    if Q.sum_pt_top15 > 1003.329 and Q.eta_1 > 0.08734131:
        z += -0.6303558 * (Q.sum_pt_top15 - 1003.329) * (Q.eta_1 - 0.08734131)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.48439
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.2174915 * Q.n_dr_0p2_0p4 + 1.087458
    if Q.n_particles < 46.0:
        z += -0.05263779 * Q.n_particles + 2.421338
    if Q.tau21 < 0.347196:
        z += 5.313172 * Q.tau21 - 1.844712
    if Q.D2 < 1.788105:
        z += -0.2971011 * Q.D2 + 0.5312481
    if Q.e2_sq < 0.007511864:
        z += -199.1208 * Q.e2_sq + 1.495768
    if Q.LHA < 0.3719813:
        z += 4.554845 * Q.LHA - 1.694317
    if Q.girth2 < 0.009614971:
        z += 40.92977 * Q.girth2 - 0.3935385
    if Q.n_pt_above_1 < 40.0:
        z += -0.01557746 * Q.n_pt_above_1 + 0.6230983
    if Q.z_dr_0p2_0p4 < 0.02647293:
        z += 15.40533 * Q.z_dr_0p2_0p4 - 0.4078241
    if Q.girth2_top40 < 0.006259772:
        z += 58.38828 * Q.girth2_top40 - 0.08952124
    if 0.006259772 <= Q.girth2_top40 < 0.008840538:
        z += -106.9357 * Q.girth2_top40 + 0.9453693
    if Q.n_dr_0_0p05 < 25.0:
        z += 0.02651482 * Q.n_dr_0_0p05 - 0.6628705
    if Q.tau21_b2 < 0.2018786:
        z += -5.329538 * Q.tau21_b2 + 1.07592
    if Q.n_dr_0p2_0p4 < 5.0 and Q.sum_pt_top30 < 988.4375:
        z += -0.001237731 * (5.0 - Q.n_dr_0p2_0p4) * (988.4375 - Q.sum_pt_top30)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.e2 < 0.03029714:
        z += -10.60937 * (5.0 - Q.n_dr_0p2_0p4) * (0.03029714 - Q.e2)
    if Q.n_particles < 46.0 and Q.e2 > 0.01036127:
        z += -1.446914 * (46.0 - Q.n_particles) * (Q.e2 - 0.01036127)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.z_top50_slots > 0.985099:
        z += 4.343233 * (8.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.985099)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.3114405
    if Q.e2 < 0.03263075:
        z += -3.819244 * Q.e2 - 0.3273722
    if 0.03263075 <= Q.e2 < 0.04358622:
        z += 41.25768 * Q.e2 - 1.798266
    if Q.e2_sq < 0.00592208:
        z += 358.4383 * Q.e2_sq - 2.1227
    if Q.psi_0p3 >= 0.9973959:
        z += 105.1659 * Q.psi_0p3 - 104.892
    if Q.n_particles >= 29.0:
        z += -0.01802707 * Q.n_particles + 0.5227852
    if Q.n_dr_0_0p05 >= 10.0:
        z += 0.02083536 * Q.n_dr_0_0p05 - 0.2083536
    if Q.sum_pt_top50 < 1061.184:
        z += 0.01059496 * Q.sum_pt_top50 - 11.2432
    if Q.girth2 < 0.01397874:
        z += -36.18039 * Q.girth2 + 0.5057564
    if Q.girth2_top20 < 0.007538019:
        z += 57.20147 * Q.girth2_top20 - 0.03120438
    if 0.007538019 <= Q.girth2_top20 < 0.01083435:
        z += -121.3412 * Q.girth2_top20 + 1.314654
    if 0.00242543 <= Q.girth2_top50 < 0.004573744:
        z += -107.6116 * Q.girth2_top50 + 0.2610044
    if Q.girth2_top50 >= 0.004573744:
        z += 65.30979 * Q.girth2_top50 - 0.5298936
    if 0.002197765 <= Q.girth2_top15 < 0.004169954:
        z += 125.1663 * Q.girth2_top15 - 0.275086
    if 0.004169954 <= Q.girth2_top15 < 0.007887677:
        z += 65.09157 * Q.girth2_top15 - 0.02457725
    if Q.girth2_top15 >= 0.007887677:
        z += -12.10915 * Q.girth2_top15 + 0.5843571
    if Q.n_dr_0p2_0p4 < 26.0:
        z += -0.01353414 * Q.n_dr_0p2_0p4 + 0.3518877
    if Q.lam1 < 0.006716737:
        z += 149.9889 * Q.lam1 - 1.007436
    if Q.width < 0.009614971:
        z += -268.7002 * Q.width + 2.583545
    if 0.1870291 <= Q.LHA < 0.2454112:
        z += 2.54491 * Q.LHA - 0.4759723
    if 0.2454112 <= Q.LHA < 0.3332345:
        z += -4.056242 * Q.LHA + 1.144025
    if Q.LHA >= 0.3332345:
        z += -11.6347 * Q.LHA + 3.669428
    if Q.girth >= 0.076787:
        z += 11.15754 * Q.girth - 0.8567538
    if Q.girth2_top40 < 0.008031986:
        z += 249.8394 * Q.girth2_top40 - 2.006706
    if Q.sum_pt_top40 < 1053.047:
        z += -0.007351409 * Q.sum_pt_top40 + 7.741382
    if Q.soft10_pt >= 1.603516:
        z += 0.09735924 * Q.soft10_pt - 0.1561171
    if Q.soft10_z >= 0.0007540821:
        z += -88.64215 * Q.soft10_z + 0.06684346
    if Q.lam2 < 0.0005358203:
        z += 757.1044 * Q.lam2 - 0.4056719
    if Q.girth2 < 0.01397874 and Q.eccentricity > 0.8903081:
        z += 594.7365 * (0.01397874 - Q.girth2) * (Q.eccentricity - 0.8903081)
    if Q.n_particles > 29.0 and Q.soft1_pt < 2.275391:
        z += 0.009701986 * (Q.n_particles - 29.0) * (2.275391 - Q.soft1_pt)
    if Q.psi_0p3 > 0.9973959 and Q.sum_pt_top40 > 858.8262:
        z += 0.4311803 * (Q.psi_0p3 - 0.9973959) * (Q.sum_pt_top40 - 858.8262)
    if Q.n_particles > 29.0 and Q.eccentricity > 0.3346888:
        z += -0.02100373 * (Q.n_particles - 29.0) * (Q.eccentricity - 0.3346888)
    if Q.sj2_dr > 0.2595052 and Q.C2_b2 < 0.0329485:
        z += -356.9314 * (Q.sj2_dr - 0.2595052) * (0.0329485 - Q.C2_b2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.3443158
    z += -0.04648173 * Q.n_particles + 2.974831
    if Q.e2_sq >= 0.007872294:
        z += -259.2604 * Q.e2_sq + 2.040974
    if Q.sum_pt >= 907.9372:
        z += 0.004372127 * Q.sum_pt - 3.969617
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -28.36928 * Q.log_sum_pt + 196.0354
    if 6.920349 <= Q.log_sum_pt < 6.959294:
        z += -34.35869 * Q.log_sum_pt + 237.4842
    if Q.log_sum_pt >= 6.959294:
        z += -23.2126 * Q.log_sum_pt + 159.9153
    if Q.girth2 >= 0.02928196:
        z += 102.6179 * Q.girth2 - 3.004852
    if Q.width >= 0.006941794:
        z += 170.3053 * Q.width - 1.182224
    if 934.2416 <= Q.sum_pt_top50 < 1048.098:
        z += 0.01680415 * Q.sum_pt_top50 - 15.69913
    if Q.sum_pt_top50 >= 1048.098:
        z += 0.006681501 * Q.sum_pt_top50 - 5.089605
    if Q.n_pt_above_1 >= 26.0:
        z += 0.02226096 * Q.n_pt_above_1 - 0.5787849
    if Q.sum_pt_top20 >= 1017.778:
        z += 0.005052379 * Q.sum_pt_top20 - 5.142201
    if Q.sum_pt_top5 < 531.1875:
        z += 0.0007941072 * Q.sum_pt_top5 - 1.169934
    if 531.1875 <= Q.sum_pt_top5 < 579.875:
        z += -0.0007479858 * Q.sum_pt_top5 - 0.3507932
    if 579.875 <= Q.sum_pt_top5 < 902.4062:
        z += 0.0006575414 * Q.sum_pt_top5 - 1.165823
    if Q.sum_pt_top5 >= 902.4062:
        z += -0.001542093 * Q.sum_pt_top5 + 0.8191405
    if Q.n_for_90pct >= 11.0:
        z += 0.02757077 * Q.n_for_90pct - 0.3032785
    if Q.girth2_top15 < 0.005383629:
        z += -19.18028 * Q.girth2_top15 - 0.4671546
    if 0.005383629 <= Q.girth2_top15 < 0.01563836:
        z += 55.62452 * Q.girth2_top15 - 0.8698759
    if Q.sum_pt_top30 >= 911.9328:
        z += 0.003694669 * Q.sum_pt_top30 - 3.36929
    if Q.z_top30_slots >= 0.9203881:
        z += -5.801709 * Q.z_top30_slots + 5.339824
    if Q.n_dr_0p2_0p4 < 8.0:
        z += -0.06923257 * Q.n_dr_0p2_0p4 + 0.5538605
    if Q.girth2_top30 < 0.01215787:
        z += 44.64937 * Q.girth2_top30 - 0.5428414
    if Q.tau4 < 0.01626937:
        z += -39.5092 * Q.tau4 + 0.6427896
    if Q.sum_pt_top15 >= 967.7705:
        z += -0.003512707 * Q.sum_pt_top15 + 3.399494
    if Q.girth2_top20 < 0.008031209:
        z += 40.68884 * Q.girth2_top20 - 0.3267806
    if Q.n_particles < 64.0 and Q.D2 < 2.410481:
        z += -0.018967 * (64.0 - Q.n_particles) * (2.410481 - Q.D2)
    if Q.n_pt_above_1 > 26.0 and Q.n_dr_0_0p05 < 30.0:
        z += -0.0003196268 * (Q.n_pt_above_1 - 26.0) * (30.0 - Q.n_dr_0_0p05)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 20953.9 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.7072238
    if Q.lam1 < 0.003811746:
        z += -1103.969 * Q.lam1 + 4.739667
    if 0.003811746 <= Q.lam1 < 0.006189818:
        z += -137.7432 * Q.lam1 + 1.056662
    if 0.006189818 <= Q.lam1 < 0.007671243:
        z += -184.6392 * Q.lam1 + 1.34694
    if Q.lam1 >= 0.007671243:
        z += -46.89602 * Q.lam1 + 0.2902778
    if Q.girth2 < 0.01397874:
        z += 150.2881 * Q.girth2 - 2.100839
    if Q.e3 < 0.0003372339:
        z += -1203.886 * Q.e3 + 0.4059912
    if Q.psi_0p3 >= 0.9853273:
        z += 71.03677 * Q.psi_0p3 - 69.99447
    if Q.e2_sq < 0.007511864:
        z += -143.9134 * Q.e2_sq + 2.88957
    if 0.007511864 <= Q.e2_sq < 0.02580859:
        z += -98.84351 * Q.e2_sq + 2.551011
    if 0.01541561 <= Q.e2 < 0.04755309:
        z += -68.51546 * Q.e2 + 1.056208
    if Q.e2 >= 0.04755309:
        z += -94.36802 * Q.e2 + 2.285577
    if Q.lam2 < 0.0009731947:
        z += -283.3541 * Q.lam2 + 0.5033243
    if 0.0009731947 <= Q.lam2 < 0.001776308:
        z += -461.6036 * Q.lam2 + 0.6767958
    if Q.lam2 >= 0.001776308:
        z += -178.2495 * Q.lam2 + 0.1734715
    if Q.width < 0.008190222:
        z += 571.7137 * Q.width - 5.629614
    if 0.008190222 <= Q.width < 0.009614971:
        z += 664.7849 * Q.width - 6.391888
    if Q.girth2_top2 < 0.002412891:
        z += -100.339 * Q.girth2_top2 + 0.2421071
    if Q.girth2_top20 < 0.00287991:
        z += -132.2136 * Q.girth2_top20 - 1.061934
    if 0.00287991 <= Q.girth2_top20 < 0.01083435:
        z += 114.9565 * Q.girth2_top20 - 1.773761
    if 0.01083435 <= Q.girth2_top20 < 0.02265114:
        z += 44.70608 * Q.girth2_top20 - 1.012643
    if Q.LHA >= 0.2091025:
        z += 6.424307 * Q.LHA - 1.343339
    if Q.girth2_top40 < 0.008840538:
        z += -101.9757 * Q.girth2_top40 + 0.9015203
    if Q.girth2_top30 >= 0.008376291:
        z += 17.88667 * Q.girth2_top30 - 0.1498239
    if Q.log_sum_pt >= 6.811175:
        z += 1.576047 * Q.log_sum_pt - 10.73473
    if Q.sum_pt_top20 >= 695.3094:
        z += 0.002064444 * Q.sum_pt_top20 - 1.435427
    if Q.girth2 < 0.01397874 and Q.psi_0p3 > 0.9853273:
        z += -7879.986 * (0.01397874 - Q.girth2) * (Q.psi_0p3 - 0.9853273)
    if Q.lam1 < 0.003811746 and Q.sum_pt_top50 < 1008.935:
        z += -5.134892 * (0.003811746 - Q.lam1) * (1008.935 - Q.sum_pt_top50)
    if Q.e2 > 0.04755309 and Q.pt_14 < 26.14062:
        z += 4.286503 * (Q.e2 - 0.04755309) * (26.14062 - Q.pt_14)
    if Q.e3 < 0.0003372339 and Q.N2 < 0.3384815:
        z += -10086.67 * (0.0003372339 - Q.e3) * (0.3384815 - Q.N2)
    if Q.e2 > 0.04755309 and Q.z_dr_0_0p05 < 0.3289237:
        z += 244.0399 * (Q.e2 - 0.04755309) * (0.3289237 - Q.z_dr_0_0p05)
    if Q.psi_0p3 > 0.9853273 and Q.n_dr_0p1_0p2 > 15.0:
        z += 2.052033 * (Q.psi_0p3 - 0.9853273) * (Q.n_dr_0p1_0p2 - 15.0)
    if Q.e2_sq < 0.007511864 and Q.sum_pt < 1007.788:
        z += 5.860332 * (0.007511864 - Q.e2_sq) * (1007.788 - Q.sum_pt)
    if Q.e2_sq < 0.02580859 and Q.log_sum_pt < 6.98945:
        z += -286.9285 * (0.02580859 - Q.e2_sq) * (6.98945 - Q.log_sum_pt)
    if Q.sum_pt_top20 > 695.3094 and Q.D2_b2 < 1.36316:
        z += 0.000802413 * (Q.sum_pt_top20 - 695.3094) * (1.36316 - Q.D2_b2)
    if Q.log_sum_pt > 6.811175 and Q.dr_max_012 > 0.1686705:
        z += -14.81581 * (Q.log_sum_pt - 6.811175) * (Q.dr_max_012 - 0.1686705)
    if Q.log_sum_pt > 6.811175 and Q.girth2_top3 < 0.001592178:
        z += -1136.946 * (Q.log_sum_pt - 6.811175) * (0.001592178 - Q.girth2_top3)
    if Q.z_top20_slots > 0.7111557 and Q.girth2_top3 > 0.01002369:
        z += 274.2772 * (Q.z_top20_slots - 0.7111557) * (Q.girth2_top3 - 0.01002369)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.3994141
    if Q.girth2 < 0.007877041:
        z += 806.8046 * Q.girth2 - 6.355233
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.02403464 * Q.n_dr_0p2_0p4 + 0.3605197
    if Q.lam1 < 0.005913555:
        z += 644.2767 * Q.lam1 - 4.198224
    if 0.005913555 <= Q.lam1 < 0.007671243:
        z += 35.37067 * Q.lam1 - 0.5974251
    if 0.007671243 <= Q.lam1 < 0.008241985:
        z += 571.3402 * Q.lam1 - 4.708978
    if Q.lam2 < 0.002396991:
        z += 122.8628 * Q.lam2 - 0.294501
    if Q.tau1 < 0.07708632:
        z += -26.18088 * Q.tau1 + 2.018188
    if Q.e2_sq < 0.006936725:
        z += -608.0588 * Q.e2_sq + 7.141542
    if 0.006936725 <= Q.e2_sq < 0.007511864:
        z += -1046.009 * Q.e2_sq + 10.17948
    if 0.007511864 <= Q.e2_sq < 0.009606007:
        z += -1108.808 * Q.e2_sq + 10.65122
    if Q.girth < 0.05660088:
        z += 33.44077 * Q.girth - 3.703739
    if 0.05660088 <= Q.girth < 0.08589404:
        z += 61.82201 * Q.girth - 5.310143
    if Q.width < 0.01397874:
        z += -359.9561 * Q.width + 5.031734
    if Q.girth2_top50 < 0.00634935:
        z += 726.2427 * Q.girth2_top50 - 5.23494
    if 0.00634935 <= Q.girth2_top50 < 0.007820315:
        z += 153.027 * Q.girth2_top50 - 1.595393
    if 0.007820315 <= Q.girth2_top50 < 0.008124776:
        z += 1309.44 * Q.girth2_top50 - 10.63891
    if Q.tau2 < 0.04828819:
        z += 18.27887 * Q.tau2 - 0.8826537
    if Q.LHA < 0.2601462:
        z += 3.419655 * Q.LHA - 0.07551896
    if 0.2601462 <= Q.LHA < 0.302389:
        z += -19.2717 * Q.LHA + 5.827549
    if Q.e2 < 0.03480688:
        z += -37.80416 * Q.e2 + 1.315845
    if Q.psi_0p3 >= 0.9966167:
        z += 174.8501 * Q.psi_0p3 - 174.2586
    if 0.2404747 <= Q.max_dr < 0.2982:
        z += -3.306801 * Q.max_dr + 0.7952021
    if Q.max_dr >= 0.2982:
        z += -1.414802 * Q.max_dr + 0.231008
    if Q.mean_eta2 < 0.0009793444:
        z += 464.7023 * Q.mean_eta2 - 0.4551036
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1937688:
        z += -19.23673 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1937688)
    if Q.tau21_b2 < 0.2352054 and Q.e2_sq < 0.007872294:
        z += -3236.813 * (0.2352054 - Q.tau21_b2) * (0.007872294 - Q.e2_sq)
    if Q.tau21_b2 < 0.2352054 and Q.e2_sq < 0.01396296:
        z += 711.1328 * (0.2352054 - Q.tau21_b2) * (0.01396296 - Q.e2_sq)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt_top30 > 978.0762:
        z += 0.02947017 * (0.2352054 - Q.tau21_b2) * (Q.sum_pt_top30 - 978.0762)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.sum_pt_top40 < 1225.842:
        z += -7.987221e-05 * (15.0 - Q.n_dr_0p2_0p4) * (1225.842 - Q.sum_pt_top40)
    if Q.lam1 < 0.005913555 and Q.sum_pt_top40 < 1139.734:
        z += 1.592598 * (0.005913555 - Q.lam1) * (1139.734 - Q.sum_pt_top40)
    if Q.psi_0p3 > 0.9980008 and Q.dr_0 < 0.0361727:
        z += -8115.812 * (Q.psi_0p3 - 0.9980008) * (0.0361727 - Q.dr_0)
    if Q.e2_sq < 0.009606007 and Q.sum_pt < 1167.447:
        z += -6.585104 * (0.009606007 - Q.e2_sq) * (1167.447 - Q.sum_pt)
    if Q.lam1 < 0.005913555 and Q.sum_pt < 1167.447:
        z += 1.972116 * (0.005913555 - Q.lam1) * (1167.447 - Q.sum_pt)
    if Q.lam2 < 0.002396991 and Q.log_sum_pt < 7.139296:
        z += 1335.874 * (0.002396991 - Q.lam2) * (7.139296 - Q.log_sum_pt)
    if Q.psi_0p3 > 0.9980008 and Q.n_pt_above_10 > 20.0:
        z += -23.76791 * (Q.psi_0p3 - 0.9980008) * (Q.n_pt_above_10 - 20.0)
    if Q.lam2 < 0.002396991 and Q.zdr_0 > 0.002524869:
        z += -12147.24 * (0.002396991 - Q.lam2) * (Q.zdr_0 - 0.002524869)
    if Q.girth < 0.08589404 and Q.sum_pt_top50 < 1156.659:
        z += 0.1021656 * (0.08589404 - Q.girth) * (1156.659 - Q.sum_pt_top50)
    if Q.psi_0p3 > 0.9966167 and Q.soft1_pt < 2.275391:
        z += 45.703 * (Q.psi_0p3 - 0.9966167) * (2.275391 - Q.soft1_pt)
    return max(0.0, z)


def neuron_8(Q):
    z = 1.212959
    if Q.girth2 < 0.003638856:
        z += -763.2057 * Q.girth2 + 5.469664
    if 0.003638856 <= Q.girth2 < 0.01397874:
        z += 33.16547 * Q.girth2 + 2.571784
    if 0.01397874 <= Q.girth2 < 0.0258669:
        z += -255.3294 * Q.girth2 + 6.60458
    if Q.sum_pt_top40 < 1007.44:
        z += 0.006963182 * Q.sum_pt_top40 - 7.014989
    if Q.n_dr_0p2_0p4 < 18.0:
        z += 0.04566218 * Q.n_dr_0p2_0p4 - 0.8219193
    if Q.e2_sq < 0.002574843:
        z += 1222.993 * Q.e2_sq - 7.173958
    if 0.002574843 <= Q.e2_sq < 0.009606007:
        z += 572.4432 * Q.e2_sq - 5.498894
    if Q.girth2_top30 < 0.007463985:
        z += 578.2334 * Q.girth2_top30 - 4.315925
    if Q.sum_pt < 1028.184:
        z += -0.0163594 * Q.sum_pt + 16.95038
    if 1028.184 <= Q.sum_pt < 1085.125:
        z += -0.002281426 * Q.sum_pt + 2.475632
    if Q.girth2_top50 < 0.01351431:
        z += -107.0048 * Q.girth2_top50 + 1.446096
    if Q.sj2_dr >= 0.2232169:
        z += 2.748506 * Q.sj2_dr - 0.613513
    if Q.sum_pt_top50 < 1008.935:
        z += -0.0162678 * Q.sum_pt_top50 + 18.53826
    if 1008.935 <= Q.sum_pt_top50 < 1156.659:
        z += -0.01080826 * Q.sum_pt_top50 + 13.02994
    if 1156.659 <= Q.sum_pt_top50 < 1245.697:
        z += -0.005935268 * Q.sum_pt_top50 + 7.393544
    if Q.width < 0.008190222:
        z += 511.9983 * Q.width - 4.193379
    if Q.log_sum_pt < 6.811175:
        z += 4.271585 * Q.log_sum_pt - 29.09452
    if Q.tau1 < 0.1219132:
        z += -33.47641 * Q.tau1 + 4.081218
    if 0.005788041 <= Q.girth2_top15 < 0.007887677:
        z += -137.2223 * Q.girth2_top15 + 0.7942481
    if Q.girth2_top15 >= 0.007887677:
        z += 19.39037 * Q.girth2_top15 - 0.4410618
    if Q.girth2_top40 < 0.008840538:
        z += -181.6401 * Q.girth2_top40 + 1.605796
    if 926.0902 <= Q.sum_pt_top20 < 1017.778:
        z += -0.001746196 * Q.sum_pt_top20 + 1.617135
    if Q.sum_pt_top20 >= 1017.778:
        z += 0.002690186 * Q.sum_pt_top20 - 2.898118
    if Q.girth2 < 0.0258669 and Q.log_sum_pt < 7.139296:
        z += -1532.364 * (0.0258669 - Q.girth2) * (7.139296 - Q.log_sum_pt)
    if Q.girth2_top30 < 0.007463985 and Q.sum_pt < 1260.541:
        z += 4.385956 * (0.007463985 - Q.girth2_top30) * (1260.541 - Q.sum_pt)
    if Q.girth2_top30 < 0.007463985 and Q.sum_pt_top40 < 1095.686:
        z += -2.819332 * (0.007463985 - Q.girth2_top30) * (1095.686 - Q.sum_pt_top40)
    if Q.girth2 < 0.01397874 and Q.sum_pt < 986.0565:
        z += 2.734773 * (0.01397874 - Q.girth2) * (986.0565 - Q.sum_pt)
    if Q.girth2 < 0.0258669 and Q.z_top50_slots > 0.9704436:
        z += -708.486 * (0.0258669 - Q.girth2) * (Q.z_top50_slots - 0.9704436)
    if Q.tau1 < 0.1219132 and Q.planar_flow < 0.5364935:
        z += -23.62852 * (0.1219132 - Q.tau1) * (0.5364935 - Q.planar_flow)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.sj2_zsoft > 0.1181474:
        z += 0.1040963 * (18.0 - Q.n_dr_0p2_0p4) * (Q.sj2_zsoft - 0.1181474)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.5285594
    if Q.girth2_top40 < 0.005712208:
        z += -136.2596 * Q.girth2_top40 + 0.7783433
    if Q.sum_pt_top40 < 972.4111:
        z += -0.008737303 * Q.sum_pt_top40 + 8.496251
    if Q.girth2 >= 0.0258669:
        z += -523.4471 * Q.girth2 + 13.53995
    if Q.lam1 < 0.004673423:
        z += -234.7832 * Q.lam1 + 1.704359
    if 0.004673423 <= Q.lam1 < 0.006716737:
        z += -33.68042 * Q.lam1 + 0.7645202
    if 0.006716737 <= Q.lam1 < 0.007259287:
        z += -196.3362 * Q.lam1 + 1.857037
    if Q.lam1 >= 0.007259287:
        z += 38.44693 * Q.lam1 + 0.152678
    if 0.01989454 <= Q.e2_sq < 0.0292152:
        z += -151.9587 * Q.e2_sq + 3.023149
    if Q.e2_sq >= 0.0292152:
        z += 598.8953 * Q.e2_sq - 18.9132
    if Q.sum_pt_top30 < 1191.938:
        z += -0.003955224 * Q.sum_pt_top30 + 4.714381
    if Q.width < 0.006941794:
        z += -538.1354 * Q.width + 3.735625
    if Q.girth2_top15 < 0.004169954:
        z += 112.2504 * Q.girth2_top15 - 1.152893
    if 0.004169954 <= Q.girth2_top15 < 0.02146578:
        z += 30.32342 * Q.girth2_top15 - 0.8112618
    if 0.02146578 <= Q.girth2_top15 < 0.02675364:
        z += -61.78697 * Q.girth2_top15 + 1.16596
    if Q.girth2_top15 >= 0.02675364:
        z += -92.11039 * Q.girth2_top15 + 1.977221
    if Q.n_pt_above_1 < 26.0:
        z += -0.04605181 * Q.n_pt_above_1 + 1.197347
    if Q.girth >= 0.1207452:
        z += -11.1022 * Q.girth + 1.340537
    if Q.e3 >= 0.0005178279:
        z += 4562.032 * Q.e3 - 2.362347
    if Q.girth2_top20 < 0.01083435:
        z += 66.29205 * Q.girth2_top20 - 0.7182314
    if Q.tau1 < 0.06310829:
        z += 10.81083 * Q.tau1 - 0.6822531
    if Q.girth2_top40 < 0.005712208 and Q.sum_pt < 1007.788:
        z += -3.196259 * (0.005712208 - Q.girth2_top40) * (1007.788 - Q.sum_pt)
    if Q.girth2_top40 < 0.005712208 and Q.log_sum_pt > 7.017258:
        z += 3545.253 * (0.005712208 - Q.girth2_top40) * (Q.log_sum_pt - 7.017258)
    if Q.sum_pt_top40 < 972.4111 and Q.soft4_pt > 1.106445:
        z += -0.004045569 * (972.4111 - Q.sum_pt_top40) * (Q.soft4_pt - 1.106445)
    if Q.LHA > 0.3332345 and Q.sum_pt < 1042.609:
        z += 0.09291425 * (Q.LHA - 0.3332345) * (1042.609 - Q.sum_pt)
    if Q.lam1 < 0.007259287 and Q.sum_pt > 1085.125:
        z += -2.276065 * (0.007259287 - Q.lam1) * (Q.sum_pt - 1085.125)
    if Q.e2_sq > 0.0292152 and Q.sum_pt < 1115.723:
        z += -1.633307 * (Q.e2_sq - 0.0292152) * (1115.723 - Q.sum_pt)
    if Q.z_dr_0_0p05 > 0.878906 and Q.sum_pt_top40 > 1013.042:
        z += 0.02384802 * (Q.z_dr_0_0p05 - 0.878906) * (Q.sum_pt_top40 - 1013.042)
    if Q.lam1 < 0.007259287 and Q.n_particles > 36.0:
        z += 3.85475 * (0.007259287 - Q.lam1) * (Q.n_particles - 36.0)
    if Q.sum_pt_top40 < 972.4111 and Q.tau3 < 0.0369869:
        z += -0.5535105 * (972.4111 - Q.sum_pt_top40) * (0.0369869 - Q.tau3)
    if Q.girth2_top15 < 0.004169954 and Q.n_dr_0p2_0p4 > 8.0:
        z += 21.26751 * (0.004169954 - Q.girth2_top15) * (Q.n_dr_0p2_0p4 - 8.0)
    if Q.log_sum_pt < 6.903423 and Q.z_dr_0p1_0p2 < 0.4479367:
        z += 21.5924 * (6.903423 - Q.log_sum_pt) * (0.4479367 - Q.z_dr_0p1_0p2)
    if Q.log_sum_pt < 6.903423 and Q.soft4_pt > 1.789258:
        z += 5.364608 * (6.903423 - Q.log_sum_pt) * (Q.soft4_pt - 1.789258)
    if Q.girth > 0.1207452 and Q.sj3_pairmin_over_m < 0.4606099:
        z += 100.3783 * (Q.girth - 0.1207452) * (0.4606099 - Q.sj3_pairmin_over_m)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.769947
    if Q.girth < 0.03577037:
        z += 146.8651 * Q.girth - 10.59747
    if 0.03577037 <= Q.girth < 0.1207452:
        z += 62.88989 * Q.girth - 7.593651
    if 0.01397874 <= Q.girth2 < 0.02928196:
        z += -137.0983 * Q.girth2 + 1.916462
    if Q.girth2 >= 0.02928196:
        z += -208.7324 * Q.girth2 + 4.014049
    if Q.girth2_top30 < 0.005402331:
        z += 98.74514 * Q.girth2_top30 - 0.5334539
    if Q.sum_pt_top3 < 309.625:
        z += -0.0008216826 * Q.sum_pt_top3 + 0.5393396
    if 309.625 <= Q.sum_pt_top3 < 656.3844:
        z += -0.002951981 * Q.sum_pt_top3 + 1.198933
    if Q.sum_pt_top3 >= 656.3844:
        z += -0.002130298 * Q.sum_pt_top3 + 0.6595936
    if Q.e2 >= 0.006720044:
        z += 31.99691 * Q.e2 - 0.2150206
    if Q.e2_sq < 0.00592208:
        z += -276.5104 * Q.e2_sq + 1.486938
    if 0.00592208 <= Q.e2_sq < 0.00818374:
        z += -41.70961 * Q.e2_sq + 0.09642862
    if 0.00818374 <= Q.e2_sq < 0.009606007:
        z += 172.1983 * Q.e2_sq - 1.654138
    if Q.e2_sq >= 0.02580859:
        z += -208.2462 * Q.e2_sq + 5.374542
    if Q.psi_0p3 >= 0.9985421:
        z += -255.5547 * Q.psi_0p3 + 255.1821
    if Q.z_dr_0_0p05 >= 0.7128619:
        z += -1.410108 * Q.z_dr_0_0p05 + 1.005212
    if Q.M2 < 0.08222447:
        z += -8.582362 * Q.M2 + 0.7056801
    if Q.width < 0.006403325:
        z += -248.0285 * Q.width + 1.588207
    if Q.n_dr_0_0p05 >= 6.0:
        z += 0.01652345 * Q.n_dr_0_0p05 - 0.09914068
    if Q.tau21_b2 < 0.3062621:
        z += 2.285322 * Q.tau21_b2 - 0.6999077
    if Q.dr_0 < 0.06413297:
        z += -8.846019 * Q.dr_0 + 0.5673215
    if Q.D2 < 2.410481:
        z += -0.3058233 * Q.D2 + 0.7371812
    if Q.e3 < 0.0005178279:
        z += 996.0394 * Q.e3 - 0.5157769
    if Q.girth2_top10 < 0.002247756:
        z += 226.9274 * Q.girth2_top10 - 1.323129
    if 0.002247756 <= Q.girth2_top10 < 0.007678544:
        z += 100.4842 * Q.girth2_top10 - 1.038916
    if 0.007678544 <= Q.girth2_top10 < 0.01976735:
        z += 22.11496 * Q.girth2_top10 - 0.4371542
    if Q.n_dr_0p2_0p4 < 15.0:
        z += 0.04299195 * Q.n_dr_0p2_0p4 - 0.6448792
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -7.450574 * Q.z_dr_0p2_0p4 + 0.6796837
    if Q.LHA < 0.3719813:
        z += -8.807123 * Q.LHA + 3.276085
    if Q.girth2_top40 >= 0.02497133:
        z += 116.2119 * Q.girth2_top40 - 2.901965
    if 0.005718442 <= Q.girth2_top20 < 0.008031209:
        z += -195.8126 * Q.girth2_top20 + 1.119743
    if Q.girth2_top20 >= 0.008031209:
        z += -18.74928 * Q.girth2_top20 - 0.3022894
    if Q.tau1 < 0.1507173:
        z += -17.13594 * Q.tau1 + 2.582682
    if Q.e2 > 0.006720044 and Q.log_sum_pt > 6.893714:
        z += -129.9993 * (Q.e2 - 0.006720044) * (Q.log_sum_pt - 6.893714)
    if Q.e2 > 0.006720044 and Q.sj3_pairmin_over_m > 0.1765064:
        z += 23.03901 * (Q.e2 - 0.006720044) * (Q.sj3_pairmin_over_m - 0.1765064)
    if Q.girth2 > 0.01397874 and Q.pt_3 < 114.75:
        z += 0.6538593 * (Q.girth2 - 0.01397874) * (114.75 - Q.pt_3)
    return max(0.0, z)


def neuron_11(Q):
    z = 1.000478
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.09156531 * Q.n_dr_0p2_0p4 + 0.9156531
    if Q.e2_sq < 0.00818374:
        z += -558.2957 * Q.e2_sq + 4.357424
    if 0.00818374 <= Q.e2_sq < 0.009606007:
        z += 148.722 * Q.e2_sq - 1.428625
    if Q.lam1 < 0.005913555:
        z += 259.3529 * Q.lam1 - 1.533698
    if Q.girth2_top50 < 0.00634935:
        z += 261.9372 * Q.girth2_top50 - 1.663131
    if Q.e2 < 0.02210818:
        z += -53.3587 * Q.e2 + 1.021703
    if 0.02210818 <= Q.e2 < 0.02515919:
        z += -75.69069 * Q.e2 + 1.515423
    if 0.02515919 <= Q.e2 < 0.03875945:
        z += 28.59458 * Q.e2 - 1.10831
    if Q.z_top30_slots >= 0.9734886:
        z += -21.85225 * Q.z_top30_slots + 21.27291
    if Q.D2 < 4.450169:
        z += 0.1136093 * Q.D2 - 0.5055804
    if Q.C2_b2 >= 0.002289486:
        z += -6.40149 * Q.C2_b2 + 0.01465612
    if Q.girth2_top30 < 0.005809485:
        z += 196.6729 * Q.girth2_top30 - 0.4379416
    if 0.005809485 <= Q.girth2_top30 < 0.006363916:
        z += 483.4729 * Q.girth2_top30 - 2.104102
    if 0.006363916 <= Q.girth2_top30 < 0.01215787:
        z += -167.8782 * Q.girth2_top30 + 2.041042
    if Q.n_real_top50 >= 22.0:
        z += -0.03382726 * Q.n_real_top50 + 0.7441998
    if Q.girth < 0.07374472:
        z += 18.06478 * Q.girth - 1.332182
    if Q.LHA < 0.3332345:
        z += -2.778764 * Q.LHA + 0.9259801
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2 < 0.005532208:
        z += -18.93516 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.003420488 * (10.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.sum_pt_top50 < 1156.659:
        z += -0.0003704985 * (10.0 - Q.n_dr_0p2_0p4) * (1156.659 - Q.sum_pt_top50)
    if Q.lam1 < 0.005913555 and Q.sum_pt < 1115.723:
        z += 2.923866 * (0.005913555 - Q.lam1) * (1115.723 - Q.sum_pt)
    if Q.e2_sq < 0.009606007 and Q.sum_pt < 1115.723:
        z += 0.891349 * (0.009606007 - Q.e2_sq) * (1115.723 - Q.sum_pt)
    if Q.z_top30_slots > 0.9734886 and Q.D2_b2 < 1.08774:
        z += 25.79397 * (Q.z_top30_slots - 0.9734886) * (1.08774 - Q.D2_b2)
    if Q.e2_sq < 0.00818374 and Q.sum_pt < 1115.723:
        z += -3.484286 * (0.00818374 - Q.e2_sq) * (1115.723 - Q.sum_pt)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.9136016
    if Q.girth2_top30 < 0.005809485:
        z += 285.7672 * Q.girth2_top30 - 1.66016
    if Q.psi_0p1 < 0.08133662:
        z += 15.36421 * Q.psi_0p1 - 1.249673
    if Q.e2_sq < 0.007511864:
        z += -753.9428 * Q.e2_sq + 5.663516
    if Q.z_dr_0p1_0p2 < 0.03502997:
        z += 11.90093 * Q.z_dr_0p1_0p2 - 0.4168892
    if Q.sum_pt >= 1085.125:
        z += -0.01213002 * Q.sum_pt + 13.16259
    if 0.9896594 <= Q.psi_0p3 < 0.9943058:
        z += 60.1745 * Q.psi_0p3 - 59.55226
    if Q.psi_0p3 >= 0.9943058:
        z += -118.506 * Q.psi_0p3 + 118.1108
    if Q.LHA >= 0.3719813:
        z += 7.4703 * Q.LHA - 2.778812
    if Q.lam1 < 0.002752094:
        z += 1137.469 * Q.lam1 - 3.13042
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.05445037 * Q.n_dr_0p2_0p4 + 0.5445037
    if Q.log_sum_pt < 6.879399:
        z += -4.335038 * Q.log_sum_pt + 29.82245
    if Q.girth2 < 0.006403325 and Q.log_sum_pt > 6.811175:
        z += 8747.501 * (0.006403325 - Q.girth2) * (Q.log_sum_pt - 6.811175)
    if Q.e2_sq < 0.007511864 and Q.log_sum_pt > 6.811175:
        z += -5662.049 * (0.007511864 - Q.e2_sq) * (Q.log_sum_pt - 6.811175)
    if Q.psi_0p1 < 0.08133662 and Q.sum_pt_top50 > 889.8503:
        z += 0.1934197 * (0.08133662 - Q.psi_0p1) * (Q.sum_pt_top50 - 889.8503)
    if Q.sum_pt > 1085.125 and Q.lam1 < 0.001868041:
        z += 4.596581 * (Q.sum_pt - 1085.125) * (0.001868041 - Q.lam1)
    if Q.psi_0p3 > 0.9943058 and Q.girth2 < 0.00751625:
        z += 46105.1 * (Q.psi_0p3 - 0.9943058) * (0.00751625 - Q.girth2)
    if Q.sum_pt > 1085.125 and Q.z_dr_0p2_0p4 < 0.019523:
        z += -0.2102461 * (Q.sum_pt - 1085.125) * (0.019523 - Q.z_dr_0p2_0p4)
    if Q.psi_0p1 > 0.9371031 and Q.absphi_0 < 0.05444336:
        z += -94.29931 * (Q.psi_0p1 - 0.9371031) * (0.05444336 - Q.absphi_0)
    if Q.psi_0p1 > 0.9371031 and Q.dr_max_012 > 0.06112084:
        z += 147.4273 * (Q.psi_0p1 - 0.9371031) * (Q.dr_max_012 - 0.06112084)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.8507885
    if Q.sum_pt < 1017.435:
        z += 0.01892152 * Q.sum_pt - 20.53222
    if 1017.435 <= Q.sum_pt < 1085.125:
        z += -0.001099175 * Q.sum_pt - 0.1624638
    if Q.sum_pt >= 1085.125:
        z += -0.0200207 * Q.sum_pt + 20.36975
    if 0.008840538 <= Q.girth2_top40 < 0.01292642:
        z += -178.776 * Q.girth2_top40 + 1.580476
    if 0.01292642 <= Q.girth2_top40 < 0.02862386:
        z += -321.233 * Q.girth2_top40 + 3.421935
    if Q.girth2_top40 >= 0.02862386:
        z += 732.5747 * Q.girth2_top40 - 26.74211
    if Q.n_pt_above_5 < 25.0:
        z += 0.04690687 * Q.n_pt_above_5 - 1.172672
    if Q.girth < 0.1207452:
        z += 9.132299 * Q.girth - 1.102681
    if Q.sum_pt_top40 >= 972.4111:
        z += -0.00577916 * Q.sum_pt_top40 + 5.61972
    if Q.e2 < 0.03480688:
        z += -27.19427 * Q.e2 + 0.9465474
    if Q.e2 >= 0.03680582:
        z += 34.23886 * Q.e2 - 1.260189
    if Q.sum_pt_top50 < 889.8503:
        z += -0.01201251 * Q.sum_pt_top50 + 12.96143
    if 889.8503 <= Q.sum_pt_top50 < 976.277:
        z += -0.01901036 * Q.sum_pt_top50 + 19.18846
    if 976.277 <= Q.sum_pt_top50 < 1013.916:
        z += -0.009230693 * Q.sum_pt_top50 + 9.640803
    if 1013.916 <= Q.sum_pt_top50 < 1078.994:
        z += -0.002417963 * Q.sum_pt_top50 + 2.733269
    if Q.sum_pt_top50 >= 1078.994:
        z += 0.009594545 * Q.sum_pt_top50 - 10.22816
    if 6.811175 <= Q.log_sum_pt < 6.893714:
        z += 27.74861 * Q.log_sum_pt - 189.0006
    if Q.log_sum_pt >= 6.893714:
        z += 13.70184 * Q.log_sum_pt - 92.16622
    if Q.width >= 0.01991191:
        z += -178.2822 * Q.width + 3.54994
    if Q.n_particles < 54.0:
        z += 0.02443225 * Q.n_particles - 1.319341
    if Q.e2_sq >= 0.0292152:
        z += -1757.525 * Q.e2_sq + 51.34645
    if 0.005852839 <= Q.girth2_top50 < 0.01351431:
        z += 78.69245 * Q.girth2_top50 - 0.4605743
    if Q.girth2_top50 >= 0.01351431:
        z += -197.8507 * Q.girth2_top50 + 3.276717
    if Q.sum_pt_top20 >= 1064.139:
        z += 0.002105083 * Q.sum_pt_top20 - 2.240101
    if Q.sum_pt_top10 >= 867.9266:
        z += 0.001918253 * Q.sum_pt_top10 - 1.664902
    if Q.psi_0p3 >= 0.9980008:
        z += 178.6531 * Q.psi_0p3 - 178.2959
    if Q.sum_pt_top15 < 795.6047:
        z += 0.004079798 * Q.sum_pt_top15 - 3.245907
    if Q.girth2_top40 > 0.01292642 and Q.sum_pt_top50 < 1245.697:
        z += 1.426494 * (Q.girth2_top40 - 0.01292642) * (1245.697 - Q.sum_pt_top50)
    if Q.n_pt_above_5 < 25.0 and Q.D2 < 5.378975:
        z += 0.007441197 * (25.0 - Q.n_pt_above_5) * (5.378975 - Q.D2)
    if Q.sum_pt < 1085.125 and Q.tau21 > 0.1295048:
        z += 0.009767705 * (1085.125 - Q.sum_pt) * (Q.tau21 - 0.1295048)
    if Q.girth2_top40 > 0.02862386 and Q.pt_6 > 19.46875:
        z += -43.86219 * (Q.girth2_top40 - 0.02862386) * (Q.pt_6 - 19.46875)
    if Q.girth2_top40 > 0.02862386 and Q.z_6 > 0.01906139:
        z += -51553.19 * (Q.girth2_top40 - 0.02862386) * (Q.z_6 - 0.01906139)
    if Q.girth2_top40 > 0.02862386 and Q.pt_6 < 62.25:
        z += -66.7951 * (Q.girth2_top40 - 0.02862386) * (62.25 - Q.pt_6)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.5728275
    if Q.tau21_b2 < 0.342495:
        z += -2.200716 * Q.tau21_b2 + 0.7537343
    if Q.n_dr_0p1_0p2 < 17.0:
        z += 0.0400856 * Q.n_dr_0p1_0p2 - 0.6814551
    if 0.9896594 <= Q.psi_0p3 < 0.9956185:
        z += 213.517 * Q.psi_0p3 - 211.3091
    if Q.psi_0p3 >= 0.9956185:
        z += 434.4151 * Q.psi_0p3 - 431.2394
    if Q.LHA < 0.2091025:
        z += 20.20521 * Q.LHA - 4.22496
    if Q.LHA >= 0.3098384:
        z += -3.78187 * Q.LHA + 1.171768
    if Q.width < 0.00751625:
        z += -329.4647 * Q.width + 2.476339
    if Q.e2_sq < 0.00616708:
        z += 294.0743 * Q.e2_sq + 1.876611
    if 0.00616708 <= Q.e2_sq < 0.009606007:
        z += -406.024 * Q.e2_sq + 6.194173
    if 0.009606007 <= Q.e2_sq < 0.01396296:
        z += -315.3222 * Q.e2_sq + 5.322891
    if 0.01396296 <= Q.e2_sq < 0.01989454:
        z += -155.112 * Q.e2_sq + 3.085881
    if Q.tau1 < 0.05444509:
        z += 101.158 * Q.tau1 - 5.630203
    if 0.05444509 <= Q.tau1 < 0.0705748:
        z += -18.11333 * Q.tau1 + 0.8635346
    if 0.0705748 <= Q.tau1 < 0.1008497:
        z += 13.70146 * Q.tau1 - 1.381788
    if Q.girth2 < 0.008190222:
        z += 1075.276 * Q.girth2 - 8.806752
    if Q.girth2_top30 < 0.008376291:
        z += 452.0593 * Q.girth2_top30 - 4.349678
    if 0.008376291 <= Q.girth2_top30 < 0.01215787:
        z += 148.9053 * Q.girth2_top30 - 1.810372
    if Q.N2 < 0.3572263:
        z += 2.296133 * Q.N2 - 0.8202391
    if Q.girth2_top3 < 0.001155057:
        z += 358.2264 * Q.girth2_top3 - 0.4137718
    if Q.n_pt_above_10 >= 28.0:
        z += -0.08884712 * Q.n_pt_above_10 + 2.487719
    if Q.e2 < 0.01879315:
        z += -143.004 * Q.e2 + 4.019773
    if 0.01879315 <= Q.e2 < 0.03480688:
        z += -83.19606 * Q.e2 + 2.895795
    if Q.girth2_top50 < 0.01951641:
        z += 88.05518 * Q.girth2_top50 - 1.718521
    if Q.girth2_top20 < 0.006374178:
        z += 50.414 * Q.girth2_top20 - 1.305169
    if 0.006374178 <= Q.girth2_top20 < 0.01083435:
        z += 220.579 * Q.girth2_top20 - 2.389831
    if Q.log_sum_pt >= 7.062574:
        z += -4.329853 * Q.log_sum_pt + 30.57991
    if Q.tau21_b2 < 0.342495 and Q.lam1 < 0.02048524:
        z += -83.33684 * (0.342495 - Q.tau21_b2) * (0.02048524 - Q.lam1)
    if Q.e2_sq < 0.01396296 and Q.z_top50_slots < 0.9906378:
        z += 2744.098 * (0.01396296 - Q.e2_sq) * (0.9906378 - Q.z_top50_slots)
    if Q.psi_0p3 > 0.9956185 and Q.sum_pt_top40 < 1225.842:
        z += -1.673011 * (Q.psi_0p3 - 0.9956185) * (1225.842 - Q.sum_pt_top40)
    if Q.psi_0p3 > 0.9896594 and Q.z_1 < 0.1438306:
        z += -464.8103 * (Q.psi_0p3 - 0.9896594) * (0.1438306 - Q.z_1)
    if Q.girth2_top30 < 0.01215787 and Q.sum_pt > 907.9372:
        z += 1.455396 * (0.01215787 - Q.girth2_top30) * (Q.sum_pt - 907.9372)
    if Q.e2 < 0.03480688 and Q.sum_pt_top30 > 886.3438:
        z += -0.4005014 * (0.03480688 - Q.e2) * (Q.sum_pt_top30 - 886.3438)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.07443579
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -5.673331 * Q.z_dr_0p1_0p2 + 0.6827495
    if Q.psi_0p1 >= 0.8976117:
        z += -6.275508 * Q.psi_0p1 + 5.63297
    if Q.girth2_top5 < 0.002270363:
        z += 100.1213 * Q.girth2_top5 + 0.4348273
    if 0.002270363 <= Q.girth2_top5 < 0.008329695:
        z += -109.2759 * Q.girth2_top5 + 0.9102348
    if Q.n_dr_0p1_0p2 < 10.0:
        z += -0.04212372 * Q.n_dr_0p1_0p2 + 0.4212372
    if Q.log_sum_pt >= 7.139296:
        z += -12.92497 * Q.log_sum_pt + 92.27517
    if Q.sum_pt_top30 >= 988.4375:
        z += -0.009634195 * Q.sum_pt_top30 + 9.5228
    if Q.sum_pt_top50 >= 1156.659:
        z += 0.01512018 * Q.sum_pt_top50 - 17.4889
    if Q.tau1 < 0.06310829:
        z += 31.27895 * Q.tau1 - 2.258961
    if 0.06310829 <= Q.tau1 < 0.0705748:
        z += 38.17048 * Q.tau1 - 2.693874
    if Q.girth < 0.08068193:
        z += -7.163282 * Q.girth + 0.5779475
    if Q.sum_pt_top20 >= 869.693:
        z += 0.003102306 * Q.sum_pt_top20 - 2.698054
    if Q.sj2_dr >= 0.2232169:
        z += 3.976566 * Q.sj2_dr - 0.8876368
    if Q.e2 < 0.04082832:
        z += -29.85009 * Q.e2 + 1.218729
    if Q.z_top30_slots < 0.9860575:
        z += 5.347004 * Q.z_top30_slots - 5.272454
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.sum_pt < 986.0565:
        z += -0.07594191 * (0.1203437 - Q.z_dr_0p1_0p2) * (986.0565 - Q.sum_pt)
    if Q.girth2_top5 < 0.008329695 and Q.n_dr_0p2_0p4 > 10.0:
        z += -8.782103 * (0.008329695 - Q.girth2_top5) * (Q.n_dr_0p2_0p4 - 10.0)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.lam2 < 0.002396991:
        z += 1319.316 * (0.1203437 - Q.z_dr_0p1_0p2) * (0.002396991 - Q.lam2)
    if Q.sd_rg < 0.1778185 and Q.soft5_dr > 0.04139378:
        z += -21.3306 * (0.1778185 - Q.sd_rg) * (Q.soft5_dr - 0.04139378)
    if Q.sd_rg < 0.1778185 and Q.soft5_dr0 > 0.03721986:
        z += 20.38487 * (0.1778185 - Q.sd_rg) * (Q.soft5_dr0 - 0.03721986)
    if Q.n_dr_0p2_0p4 < 26.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += -0.001199139 * (26.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
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
