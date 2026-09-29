"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; no mass observables), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.1% (the network: 81.1%); same class as the network for 91.5% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_14                  pT of particle 14 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_pt              pT [GeV] of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_8                    pT of particle 8 / total pT
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_2               |Δφ| of particle 2
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
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
  Q.n_pt_above_5           number of particles with pT > 5 GeV
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
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau32                  N-subjettiness τ3/τ2
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
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_14=pt[14],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_6=pt[6],
        soft1_pt=softp(1, 'pt'),
        soft10_pt=softp(10, 'pt'),
        soft5_pt=softp(5, 'pt'),
        z_8=z[8],
        soft7_z=softp(7, 'z'),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sd_rg=softdrop("rg"),
        absphi_2=abs(phi[2]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top20=sum(pt[:20]),
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
        n_pt_above_5=sum(1 for x in pt if x > 5),
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
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
    )


def neuron_0(Q):
    z = -4.8
    if Q.C2 < 0.0615:
        z += 4.13 * Q.C2 - 0.253995
    if Q.LHA >= 0.118:
        z += -3.97 * Q.LHA + 0.46846
    if Q.e2 < 0.0188:
        z += -42.5 * Q.e2 + 0.799
    if Q.e2 >= 0.0483:
        z += -13.2 * Q.e2 + 0.63756
    if Q.e2_sq < 0.00455:
        z += -654.0 * Q.e2_sq + 9.08712
    if 0.00455 <= Q.e2_sq < 0.00582:
        z += -846.0 * Q.e2_sq + 9.96072
    if 0.00582 <= Q.e2_sq < 0.00947:
        z += -1380.0 * Q.e2_sq + 13.0686
    if Q.girth < 0.0856:
        z += 12.8 * Q.girth - 1.09568
    if Q.girth2_top15 < 0.0212:
        z += -69.4 * Q.girth2_top15 + 1.47128
    if 0.00336 <= Q.girth2_top30 < 0.00746:
        z += 83.6 * Q.girth2_top30 - 0.280896
    if Q.girth2_top30 >= 0.00746:
        z += 266.6 * Q.girth2_top30 - 1.646076
    if Q.lam1 < 0.00578:
        z += 152.0 * Q.lam1 - 0.87856
    if Q.lam1 >= 0.0177:
        z += -247.0 * Q.lam1 + 4.3719
    if Q.lam2 >= 0.00264:
        z += -228.0 * Q.lam2 + 0.60192
    if Q.n_dr_0p2_0p4 < 13.8:
        z += -0.0437 * Q.n_dr_0p2_0p4 + 0.60306
    if Q.n_for_90pct >= 28.8:
        z += -0.0328 * Q.n_for_90pct + 0.94464
    if Q.n_particles < 60.2:
        z += -0.0116 * Q.n_particles + 0.69832
    if Q.psi_0p2 >= 0.81:
        z += -3.97 * Q.psi_0p2 + 3.2157
    if Q.psi_0p3 >= 0.988:
        z += 37.2 * Q.psi_0p3 - 36.7536
    if 1010.0 <= Q.sum_pt < 1140.0:
        z += -0.0107 * Q.sum_pt + 10.807
    if Q.sum_pt >= 1140.0:
        z += -0.00188 * Q.sum_pt + 0.7522
    if Q.sum_pt_top20 >= 1080.0:
        z += -0.00287 * Q.sum_pt_top20 + 3.0996
    if Q.sum_pt_top30 < 1210.0:
        z += -0.00523 * Q.sum_pt_top30 + 6.3283
    if Q.sum_pt_top40 >= 856.0:
        z += 0.00566 * Q.sum_pt_top40 - 4.84496
    if Q.sum_pt_top50 < 993.0:
        z += 0.00361 * Q.sum_pt_top50 - 3.58473
    if Q.tau1 < 0.0701:
        z += 20.7 * Q.tau1 - 1.45107
    if Q.girth2 < 0.00819 and Q.girth2_top15 < 0.00612:
        z += -47500.0 * (0.00819 - Q.girth2) * (0.00612 - Q.girth2_top15)
    if Q.girth2_top50 < 0.00633 and Q.sum_pt < 1260.0:
        z += -1.57 * (0.00633 - Q.girth2_top50) * (1260.0 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 17.2 and Q.log_sum_pt > 6.91:
        z += -0.186 * (17.2 - Q.n_dr_0p2_0p4) * (Q.log_sum_pt - 6.91)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.71
    if Q.e2_sq < 0.00976:
        z += 304.0 * Q.e2_sq - 2.96704
    if Q.girth2_top20 < 0.00148:
        z += -828.0 * Q.girth2_top20 + 1.22544
    if Q.girth2_top30 < 0.0069:
        z += -443.0 * Q.girth2_top30 + 3.0567
    if Q.lam2 < 0.000796:
        z += 1050.0 * Q.lam2 - 0.8358
    if Q.log_sum_pt < 6.91:
        z += 8.38 * Q.log_sum_pt - 59.7494
    if 6.91 <= Q.log_sum_pt < 6.98:
        z += 35.78 * Q.log_sum_pt - 249.0834
    if 6.98 <= Q.log_sum_pt < 7.13:
        z += 16.48 * Q.log_sum_pt - 114.3694
    if Q.log_sum_pt >= 7.13:
        z += 8.1 * Q.log_sum_pt - 54.62
    if Q.n_dr_0p2_0p4 < 7.31:
        z += 0.0967 * Q.n_dr_0p2_0p4 - 0.706877
    if Q.n_particles >= 46.7:
        z += 0.0532 * Q.n_particles - 2.48444
    if Q.n_real_top50 >= 33.6:
        z += 0.0686 * Q.n_real_top50 - 2.30496
    if Q.pt_entropy >= 1.88:
        z += 1.28 * Q.pt_entropy - 2.4064
    if Q.sum_pt >= 974.0:
        z += 0.0248 * Q.sum_pt - 24.1552
    if Q.sum_pt_top20 >= 992.0:
        z += 0.0034 * Q.sum_pt_top20 - 3.3728
    if Q.sum_pt_top50 < 930.0:
        z += -0.0158 * Q.sum_pt_top50 + 14.694
    if Q.sum_pt_top50 >= 932.0:
        z += -0.0311 * Q.sum_pt_top50 + 28.9852
    if Q.tau1 >= 0.0273:
        z += -7.69 * Q.tau1 + 0.209937
    if Q.tau21 < 0.25:
        z += -6.35 * Q.tau21 + 1.5875
    if Q.z_top30_slots >= 0.949:
        z += 29.2 * Q.z_top30_slots - 27.7108
    if Q.M3 < 0.0366 and Q.psi_0p3 > 0.941:
        z += -405.0 * (0.0366 - Q.M3) * (Q.psi_0p3 - 0.941)
    if Q.n_particles > 29.9 and Q.zdr_0 < 0.0107:
        z += 3.69 * (Q.n_particles - 29.9) * (0.0107 - Q.zdr_0)
    if Q.z_dr_0_0p05 > 0.759 and Q.n_dr_0p05_0p1 < 11.3:
        z += -0.411 * (Q.z_dr_0_0p05 - 0.759) * (11.3 - Q.n_dr_0p05_0p1)
    if Q.z_top30_slots > 0.936 and Q.pt1_dr01 > 3.56:
        z += 0.416 * (Q.z_top30_slots - 0.936) * (Q.pt1_dr01 - 3.56)
    return max(0.0, z)


def neuron_2(Q):
    z = 5.02
    if Q.C2 >= 0.048:
        z += -2.04 * Q.C2 + 0.09792
    if Q.LHA >= 0.42:
        z += -2.41 * Q.LHA + 1.0122
    if Q.e2 >= 0.0551:
        z += -11.5 * Q.e2 + 0.63365
    if Q.e3 >= 5.32e-05:
        z += 1040.0 * Q.e3 - 0.055328
    if Q.girth2 < 0.0072:
        z += 105.0 * Q.girth2 - 0.756
    if Q.girth2_top15 >= 0.00672:
        z += -27.6 * Q.girth2_top15 + 0.185472
    if Q.lam1 < 0.0117:
        z += -33.8 * Q.lam1 + 0.39546
    if Q.log_sum_pt < 6.9:
        z += 2.23 * Q.log_sum_pt - 15.7104
    if 6.9 <= Q.log_sum_pt < 6.93:
        z += 8.35 * Q.log_sum_pt - 57.9384
    if 6.93 <= Q.log_sum_pt < 6.98:
        z += 11.25 * Q.log_sum_pt - 78.0354
    if 6.98 <= Q.log_sum_pt < 7.07:
        z += 6.12 * Q.log_sum_pt - 42.228
    if Q.log_sum_pt >= 7.07:
        z += 1.32 * Q.log_sum_pt - 8.292
    if Q.n_dr_0p1_0p2 >= 20.4:
        z += -0.00951 * Q.n_dr_0p1_0p2 + 0.194004
    if Q.n_particles < 60.6:
        z += 0.0091 * Q.n_particles - 0.55146
    if Q.psi_0p1 < 0.418:
        z += -1.4 * Q.psi_0p1 + 0.5852
    if Q.psi_0p3 >= 0.988:
        z += -15.9 * Q.psi_0p3 + 15.7092
    if 0.192 <= Q.sd_rg < 0.28:
        z += -2.01 * Q.sd_rg + 0.38592
    if Q.sd_rg >= 0.28:
        z += 1.67 * Q.sd_rg - 0.64448
    if Q.sj3_dr13 >= 0.287:
        z += 0.705 * Q.sj3_dr13 - 0.202335
    if Q.z_dr_0p1_0p2 >= 0.429:
        z += -0.921 * Q.z_dr_0p1_0p2 + 0.395109
    z += -4.81 * Q.z_top50_slots
    if Q.log_sum_pt > 6.94 and Q.girth2_top30 > 0.00283:
        z += -428.0 * (Q.log_sum_pt - 6.94) * (Q.girth2_top30 - 0.00283)
    if Q.log_sum_pt > 7.0 and Q.girth2_top30 > 0.00847:
        z += 652.0 * (Q.log_sum_pt - 7.0) * (Q.girth2_top30 - 0.00847)
    if Q.log_sum_pt > 6.94 and Q.psi_0p1 < 0.966:
        z += -8.08 * (Q.log_sum_pt - 6.94) * (0.966 - Q.psi_0p1)
    if Q.log_sum_pt > 6.92 and Q.z_top15_slots > 0.69:
        z += 7.88 * (Q.log_sum_pt - 6.92) * (Q.z_top15_slots - 0.69)
    if Q.sum_pt_top50 > 1020.0 and Q.z_dr_0p1_0p2 > 0.00203:
        z += 0.00344 * (Q.sum_pt_top50 - 1020.0) * (Q.z_dr_0p1_0p2 - 0.00203)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.485
    if Q.D2_b2 < 0.625:
        z += -0.612 * Q.D2_b2 + 0.3825
    if Q.LHA < 0.272:
        z += 4.37 * Q.LHA - 1.74502
    if 0.272 <= Q.LHA < 0.274:
        z += 5.62 * Q.LHA - 2.08502
    if 0.274 <= Q.LHA < 0.371:
        z += 6.84 * Q.LHA - 2.4193
    if Q.LHA >= 0.371:
        z += 1.22 * Q.LHA - 0.33428
    if Q.e2 < 0.00732:
        z += -32.1 * Q.e2 + 1.00473
    if 0.00732 <= Q.e2 < 0.0313:
        z += -49.5 * Q.e2 + 1.132098
    if Q.e2 >= 0.0313:
        z += -17.4 * Q.e2 + 0.127368
    if Q.e2_sq < 0.00604:
        z += 131.0 * Q.e2_sq - 0.79124
    if Q.e3 < 8.6e-05:
        z += 8750.0 * Q.e3 - 0.7525
    if Q.girth2 < 0.00412:
        z += -201.0 * Q.girth2 + 2.51583
    if 0.00412 <= Q.girth2 < 0.00969:
        z += -303.0 * Q.girth2 + 2.93607
    if Q.girth2_top30 < 0.00844:
        z += 66.4 * Q.girth2_top30 - 0.560416
    if Q.girth2_top50 >= 0.00732:
        z += 3.63 * Q.girth2_top50 - 0.0265716
    if Q.lam1 >= 0.0227:
        z += 65.4 * Q.lam1 - 1.48458
    if Q.lam2 < 0.0013:
        z += -295.0 * Q.lam2 + 0.3835
    if Q.n_dr_0_0p05 < 15.9:
        z += 0.0225 * Q.n_dr_0_0p05 - 0.35775
    if Q.n_dr_0p1_0p2 >= 5.17:
        z += -0.0176 * Q.n_dr_0p1_0p2 + 0.090992
    if Q.n_dr_0p2_0p4 < 4.46:
        z += -0.1986 * Q.n_dr_0p2_0p4 + 1.07366
    if 4.46 <= Q.n_dr_0p2_0p4 < 11.8:
        z += -0.0256 * Q.n_dr_0p2_0p4 + 0.30208
    if Q.n_for_90pct >= 21.0:
        z += -0.0305 * Q.n_for_90pct + 0.6405
    if Q.n_particles >= 45.6:
        z += -0.0328 * Q.n_particles + 1.49568
    if Q.n_real_top50 < 46.5:
        z += -0.0404 * Q.n_real_top50 + 1.8786
    if Q.psi_0p3 >= 0.998:
        z += 91.5 * Q.psi_0p3 - 91.317
    if Q.sum_pt < 1060.0:
        z += -0.00154 * Q.sum_pt + 1.6324
    if Q.sum_pt_top30 >= 1120.0:
        z += 0.000564 * Q.sum_pt_top30 - 0.63168
    if Q.z_top30_slots >= 0.92:
        z += -8.49 * Q.z_top30_slots + 7.8108
    if Q.e2_sq < 0.0084 and Q.sum_pt_top40 < 1050.0:
        z += -0.532 * (0.0084 - Q.e2_sq) * (1050.0 - Q.sum_pt_top40)
    if Q.n_dr_0p2_0p4 < 4.19 and Q.e2 < 0.0325:
        z += -5.86 * (4.19 - Q.n_dr_0p2_0p4) * (0.0325 - Q.e2)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.989
    if Q.e2 < 0.0505:
        z += 12.5 * Q.e2 - 0.63125
    if Q.e2_sq < 0.00322:
        z += 510.0 * Q.e2_sq - 2.8254
    if 0.00322 <= Q.e2_sq < 0.0067:
        z += 340.0 * Q.e2_sq - 2.278
    if Q.e2_sq >= 0.0108:
        z += 92.0 * Q.e2_sq - 0.9936
    if Q.girth2_top15 >= 0.00559:
        z += -35.0 * Q.girth2_top15 + 0.19565
    if Q.girth2_top20 < 0.00798:
        z += 81.8 * Q.girth2_top20 - 0.39276
    if 0.00798 <= Q.girth2_top20 < 0.0108:
        z += -92.2 * Q.girth2_top20 + 0.99576
    if Q.log_sum_pt < 6.96:
        z += 1.76 * Q.log_sum_pt - 12.2496
    if Q.n_dr_0_0p05 >= 2.36:
        z += 0.0234 * Q.n_dr_0_0p05 - 0.055224
    if Q.n_dr_0p2_0p4 < 24.6:
        z += -0.0279 * Q.n_dr_0p2_0p4 + 0.68634
    if Q.n_particles >= 24.5:
        z += -0.0333 * Q.n_particles + 0.81585
    if Q.sum_pt_top30 >= 1000.0:
        z += 0.000265 * Q.sum_pt_top30 - 0.265
    if Q.zdr_0 >= 0.0113:
        z += -11.9 * Q.zdr_0 + 0.13447
    if Q.e2 < 0.0455 and Q.sum_pt_top50 < 1060.0:
        z += -0.0382 * (0.0455 - Q.e2) * (1060.0 - Q.sum_pt_top50)
    if Q.girth2 < 0.012 and Q.eccentricity > 0.856:
        z += 386.0 * (0.012 - Q.girth2) * (Q.eccentricity - 0.856)
    if Q.girth2 < 0.0113 and Q.psi_0p3 > 0.978:
        z += 2740.0 * (0.0113 - Q.girth2) * (Q.psi_0p3 - 0.978)
    if Q.n_dr_0p2_0p4 < 22.4 and Q.n_dr_0p1_0p2 > 12.7:
        z += -0.000863 * (22.4 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 12.7)
    if Q.n_particles > 29.6 and Q.soft1_pt < 2.69:
        z += 0.00622 * (Q.n_particles - 29.6) * (2.69 - Q.soft1_pt)
    if Q.psi_0p3 > 0.998 and Q.sum_pt_top40 > 797.0:
        z += 0.443 * (Q.psi_0p3 - 0.998) * (Q.sum_pt_top40 - 797.0)
    if Q.sj2_dr > 0.254 and Q.C2_b2 < 0.0423:
        z += -143.0 * (Q.sj2_dr - 0.254) * (0.0423 - Q.C2_b2)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.678
    if Q.D2 < 3.12:
        z += -0.149 * Q.D2 + 0.46488
    if Q.LHA >= 0.255:
        z += -19.4 * Q.LHA + 4.947
    if Q.e2 < 0.038:
        z += -16.5 * Q.e2 + 0.627
    if Q.e2_sq >= 0.00783:
        z += -168.0 * Q.e2_sq + 1.31544
    if Q.girth >= 0.0592:
        z += 54.2 * Q.girth - 3.20864
    if Q.girth2 >= 0.0231:
        z += -92.3 * Q.girth2 + 2.13213
    if Q.girth2_top30 < 0.0134:
        z += 74.3 * Q.girth2_top30 - 0.99562
    if Q.log_sum_pt < 6.88:
        z += -14.8 * Q.log_sum_pt + 100.843
    if 6.88 <= Q.log_sum_pt < 6.91:
        z += 32.7 * Q.log_sum_pt - 225.957
    if Q.max_dr < 0.448:
        z += -1.17 * Q.max_dr + 0.52416
    if Q.n_dr_0p2_0p4 < 9.37:
        z += -0.0297 * Q.n_dr_0p2_0p4 + 0.278289
    if Q.n_particles < 52.1:
        z += -0.0245 * Q.n_particles + 1.27645
    if Q.soft1_pt >= 1.48:
        z += -0.139 * Q.soft1_pt + 0.20572
    if Q.sum_pt < 972.0:
        z += 0.0359 * Q.sum_pt - 32.4216
    if 972.0 <= Q.sum_pt < 1070.0:
        z += -0.0229 * Q.sum_pt + 24.732
    if 1070.0 <= Q.sum_pt < 1080.0:
        z += -0.0397 * Q.sum_pt + 42.708
    if Q.sum_pt >= 1080.0:
        z += -0.0168 * Q.sum_pt + 17.976
    if Q.sum_pt_top30 >= 936.0:
        z += -0.00498 * Q.sum_pt_top30 + 4.66128
    if Q.sum_pt_top40 < 991.0:
        z += -0.00425 * Q.sum_pt_top40 + 4.21175
    if Q.sum_pt_top5 < 825.0:
        z += 0.000937 * Q.sum_pt_top5 - 0.773025
    if 951.0 <= Q.sum_pt_top50 < 1020.0:
        z += 0.0193 * Q.sum_pt_top50 - 18.3543
    if Q.sum_pt_top50 >= 1020.0:
        z += 0.01593 * Q.sum_pt_top50 - 14.9169
    if Q.tau2 >= 0.0179:
        z += 4.97 * Q.tau2 - 0.088963
    if Q.z_8 < 0.0254:
        z += -13.4 * Q.z_8 + 0.34036
    if Q.n_particles < 57.7 and Q.D2 < 2.66:
        z += -0.0115 * (57.7 - Q.n_particles) * (2.66 - Q.D2)
    if Q.sum_pt_top50 > 1020.0 and Q.e4 < 6.89e-08:
        z += 43000.0 * (Q.sum_pt_top50 - 1020.0) * (6.89e-08 - Q.e4)
    return max(0.0, z)


def neuron_6(Q):
    z = -0.587
    if Q.e2 < 0.0192:
        z += -17.4 * Q.e2 + 1.90693
    if 0.0192 <= Q.e2 < 0.0571:
        z += -41.5 * Q.e2 + 2.36965
    if Q.e2_sq < 0.00798:
        z += -59.6 * Q.e2_sq + 0.475608
    if Q.e2_sq >= 0.00988:
        z += -356.0 * Q.e2_sq + 3.51728
    if Q.girth2 < 0.00445:
        z += -58.0 * Q.girth2 - 1.2294
    if 0.00445 <= Q.girth2 < 0.0132:
        z += 170.0 * Q.girth2 - 2.244
    if Q.girth2_top30 >= 0.008:
        z += 161.0 * Q.girth2_top30 - 1.288
    if Q.girth2_top40 < 0.000865:
        z += -535.7 * Q.girth2_top40 + 1.023673
    if 0.000865 <= Q.girth2_top40 < 0.00764:
        z += -82.7 * Q.girth2_top40 + 0.631828
    if Q.lam1 < 0.00379:
        z += -166.0 * Q.lam1 + 0.913
    if 0.00379 <= Q.lam1 < 0.0055:
        z += -68.3 * Q.lam1 + 0.542717
    if Q.lam1 >= 0.0055:
        z += 97.7 * Q.lam1 - 0.370283
    if Q.n_pt_above_5 >= 17.9:
        z += -0.0134 * Q.n_pt_above_5 + 0.23986
    if Q.sum_pt_top30 >= 906.0:
        z += 0.000843 * Q.sum_pt_top30 - 0.763758
    if Q.tau1 >= 0.0671:
        z += 18.2 * Q.tau1 - 1.22122
    if Q.z_dr_0p1_0p2 < 0.31:
        z += -1.04 * Q.z_dr_0p1_0p2 + 0.3224
    if Q.e2 > 0.04 and Q.pt_14 < 27.2:
        z += 0.865 * (Q.e2 - 0.04) * (27.2 - Q.pt_14)
    if Q.e2_sq < 0.0217 and Q.log_sum_pt < 7.01:
        z += -223.0 * (0.0217 - Q.e2_sq) * (7.01 - Q.log_sum_pt)
    if Q.e2_sq < 0.00935 and Q.sum_pt < 1020.0:
        z += 2.06 * (0.00935 - Q.e2_sq) * (1020.0 - Q.sum_pt)
    if Q.girth2 < 0.0116 and Q.psi_0p3 > 0.987:
        z += -2740.0 * (0.0116 - Q.girth2) * (Q.psi_0p3 - 0.987)
    if Q.girth2 < 0.0161 and Q.soft1_pt > 1.43:
        z += 22.8 * (0.0161 - Q.girth2) * (Q.soft1_pt - 1.43)
    if Q.girth2_top2 < 0.00168 and Q.M2 > 0.0304:
        z += 1770.0 * (0.00168 - Q.girth2_top2) * (Q.M2 - 0.0304)
    if Q.lam1 < 0.00319 and Q.sum_pt_top50 < 1020.0:
        z += -2.32 * (0.00319 - Q.lam1) * (1020.0 - Q.sum_pt_top50)
    if Q.log_sum_pt > 6.88 and Q.dr_max_012 > 0.0707:
        z += -7.61 * (Q.log_sum_pt - 6.88) * (Q.dr_max_012 - 0.0707)
    if Q.log_sum_pt > 6.84 and Q.girth2_top3 < 0.00221:
        z += -1030.0 * (Q.log_sum_pt - 6.84) * (0.00221 - Q.girth2_top3)
    if Q.psi_0p3 > 0.99 and Q.n_dr_0p1_0p2 > 9.27:
        z += 2.02 * (Q.psi_0p3 - 0.99) * (Q.n_dr_0p1_0p2 - 9.27)
    if Q.z_top20_slots > 0.664 and Q.girth2_top3 > 0.00915:
        z += 95.6 * (Q.z_top20_slots - 0.664) * (Q.girth2_top3 - 0.00915)
    return max(0.0, z)


def neuron_7(Q):
    z = 3.22
    if 0.32 <= Q.LHA < 0.405:
        z += 32.1 * Q.LHA - 10.272
    if Q.LHA >= 0.405:
        z += -2.6 * Q.LHA + 3.7815
    if Q.e2_sq < 0.00647:
        z += 645.0 * Q.e2_sq + 0.38985
    if 0.00647 <= Q.e2_sq < 0.00707:
        z += -765.0 * Q.e2_sq + 9.51255
    if 0.00707 <= Q.e2_sq < 0.00947:
        z += -1710.0 * Q.e2_sq + 16.1937
    if Q.girth < 0.0171:
        z += 54.1 * Q.girth - 6.28661
    if 0.0171 <= Q.girth < 0.0521:
        z += -18.1 * Q.girth - 5.05199
    if 0.0521 <= Q.girth < 0.0861:
        z += 29.8 * Q.girth - 7.54758
    if Q.girth >= 0.0861:
        z += -72.2 * Q.girth + 1.23462
    if Q.girth2_top15 >= 0.00838:
        z += -98.0 * Q.girth2_top15 + 0.82124
    if Q.girth2_top20 < 0.00631:
        z += -78.0 * Q.girth2_top20 - 0.09246
    if 0.00631 <= Q.girth2_top20 < 0.00805:
        z += 336.0 * Q.girth2_top20 - 2.7048
    if Q.girth2_top40 < 0.0252:
        z += -95.3 * Q.girth2_top40 + 2.40156
    if Q.girth2_top50 < 0.00641:
        z += -200.0 * Q.girth2_top50 - 0.5772
    if 0.00641 <= Q.girth2_top50 < 0.00807:
        z += 1120.0 * Q.girth2_top50 - 9.0384
    if Q.lam1 < 0.0082:
        z += 292.0 * Q.lam1 - 2.3944
    if Q.log_sum_pt >= 6.97:
        z += 0.429 * Q.log_sum_pt - 2.99013
    if Q.max_dr < 0.407:
        z += -2.14 * Q.max_dr + 0.87098
    if Q.n_dr_0p2_0p4 < 13.9:
        z += -0.0751 * Q.n_dr_0p2_0p4 + 1.04389
    if Q.sd_rg >= 0.302:
        z += 5.86 * Q.sd_rg - 1.76972
    if Q.sum_pt < 1070.0:
        z += 0.0106 * Q.sum_pt - 11.342
    if Q.sum_pt_top50 < 1170.0:
        z += -0.00885 * Q.sum_pt_top50 + 10.3545
    if Q.tau21_b2 < 0.178:
        z += -2.55 * Q.tau21_b2 + 0.4539
    if Q.width >= 0.0124:
        z += 186.0 * Q.width - 2.3064
    if Q.z_top50_slots < 0.993:
        z += 13.6 * Q.z_top50_slots - 13.5048
    if Q.e2_sq < 0.00926 and Q.sum_pt < 1190.0:
        z += -7.14 * (0.00926 - Q.e2_sq) * (1190.0 - Q.sum_pt)
    if Q.girth < 0.0771 and Q.sum_pt_top50 < 1170.0:
        z += 0.19 * (0.0771 - Q.girth) * (1170.0 - Q.sum_pt_top50)
    if Q.girth2 < 0.00782 and Q.z_dr_0_0p05 < 0.484:
        z += -720.0 * (0.00782 - Q.girth2) * (0.484 - Q.z_dr_0_0p05)
    if Q.lam1 < 0.00639 and Q.sum_pt < 1120.0:
        z += 3.93 * (0.00639 - Q.lam1) * (1120.0 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 14.8 and Q.n_dr_0p1_0p2 > 14.9:
        z += -0.00354 * (14.8 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 14.9)
    if Q.psi_0p3 > 0.996 and Q.soft1_pt < 2.52:
        z += 85.7 * (Q.psi_0p3 - 0.996) * (2.52 - Q.soft1_pt)
    if Q.tau21_b2 < 0.189 and Q.sj2_dr > 0.222:
        z += -47.3 * (0.189 - Q.tau21_b2) * (Q.sj2_dr - 0.222)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.96
    if Q.LHA >= 0.334:
        z += 14.1 * Q.LHA - 4.7094
    if Q.e2_sq < 0.00944:
        z += 312.0 * Q.e2_sq - 2.94528
    if Q.girth2 < 0.00363:
        z += -129.0 * Q.girth2 + 0.97772
    if 0.00363 <= Q.girth2 < 0.0128:
        z += 115.0 * Q.girth2 + 0.092
    if 0.0128 <= Q.girth2 < 0.0243:
        z += -136.0 * Q.girth2 + 3.3048
    if Q.log_sum_pt < 6.85:
        z += 6.76 * Q.log_sum_pt - 46.306
    if Q.n_dr_0p2_0p4 < 17.0:
        z += 0.0268 * Q.n_dr_0p2_0p4 - 0.4556
    if Q.sum_pt < 1040.0:
        z += -0.0135 * Q.sum_pt + 14.04
    if Q.sum_pt_top50 < 1190.0:
        z += -0.00737 * Q.sum_pt_top50 + 8.7703
    if Q.tau1 >= 0.0234:
        z += -12.6 * Q.tau1 + 0.29484
    if Q.girth2 < 0.0225 and Q.log_sum_pt < 7.16:
        z += -838.0 * (0.0225 - Q.girth2) * (7.16 - Q.log_sum_pt)
    if Q.girth2 < 0.0188 and Q.sum_pt < 983.0:
        z += 1.01 * (0.0188 - Q.girth2) * (983.0 - Q.sum_pt)
    if Q.girth2 < 0.0285 and Q.z_top50_slots > 0.972:
        z += -1340.0 * (0.0285 - Q.girth2) * (Q.z_top50_slots - 0.972)
    if Q.girth2_top30 < 0.00769 and Q.sum_pt < 1270.0:
        z += 1.99 * (0.00769 - Q.girth2_top30) * (1270.0 - Q.sum_pt)
    if Q.girth2_top30 < 0.00942 and Q.sum_pt_top40 < 1120.0:
        z += -1.45 * (0.00942 - Q.girth2_top30) * (1120.0 - Q.sum_pt_top40)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.708
    if Q.girth >= 0.12:
        z += -31.2 * Q.girth + 3.744
    if Q.girth2 < 0.00959:
        z += 167.0 * Q.girth2 - 1.60153
    if Q.girth2 >= 0.0247:
        z += -239.0 * Q.girth2 + 5.9033
    if Q.girth2_top15 < 0.0268:
        z += 26.9 * Q.girth2_top15 - 0.72092
    if Q.girth2_top40 < 0.00534:
        z += -437.0 * Q.girth2_top40 + 2.33358
    if Q.girth2_top50 < 0.00654:
        z += 89.0 * Q.girth2_top50 - 0.10886
    if 0.00654 <= Q.girth2_top50 < 0.00914:
        z += -182.0 * Q.girth2_top50 + 1.66348
    if Q.log_sum_pt < 6.91:
        z += 18.5 * Q.log_sum_pt - 127.835
    if Q.log_sum_pt >= 6.97:
        z += -3.62 * Q.log_sum_pt + 25.2314
    if Q.n_dr_0p1_0p2 >= 16.6:
        z += 0.0126 * Q.n_dr_0p1_0p2 - 0.20916
    if Q.pt_entropy < 2.29:
        z += -0.524 * Q.pt_entropy + 1.19996
    if Q.sum_pt < 999.0:
        z += -0.0259 * Q.sum_pt + 25.8741
    if Q.width < 0.00703:
        z += -621.0 * Q.width + 4.36563
    if Q.z_dr_0_0p05 >= 0.858:
        z += -4.9 * Q.z_dr_0_0p05 + 4.2042
    if Q.z_top30_slots >= 0.881:
        z += -4.15 * Q.z_top30_slots + 3.65615
    if Q.LHA > 0.291 and Q.soft5_pt < 2.17:
        z += -6.48 * (Q.LHA - 0.291) * (2.17 - Q.soft5_pt)
    if Q.LHA > 0.353 and Q.sum_pt < 1080.0:
        z += 0.136 * (Q.LHA - 0.353) * (1080.0 - Q.sum_pt)
    if Q.girth > 0.0981 and Q.soft5_pt < 3.05:
        z += 16.6 * (Q.girth - 0.0981) * (3.05 - Q.soft5_pt)
    if Q.girth2_top15 < 0.00662 and Q.n_dr_0p2_0p4 > 7.35:
        z += 8.48 * (0.00662 - Q.girth2_top15) * (Q.n_dr_0p2_0p4 - 7.35)
    if Q.girth2_top40 < 0.00496 and Q.sum_pt < 1020.0:
        z += -3.68 * (0.00496 - Q.girth2_top40) * (1020.0 - Q.sum_pt)
    if Q.lam1 < 0.0094 and Q.sum_pt_top30 < 865.0:
        z += 1.17 * (0.0094 - Q.lam1) * (865.0 - Q.sum_pt_top30)
    if Q.log_sum_pt < 6.87 and Q.z_dr_0p1_0p2 < 0.557:
        z += 10.1 * (6.87 - Q.log_sum_pt) * (0.557 - Q.z_dr_0p1_0p2)
    return max(0.0, z)


def neuron_10(Q):
    z = 0.92
    if Q.C2 < 0.0655:
        z += -12.0 * Q.C2 + 0.786
    if Q.dr_0 < 0.0732:
        z += -3.96 * Q.dr_0 + 0.289872
    if Q.e2 >= 0.00465:
        z += 60.2 * Q.e2 - 0.27993
    if Q.e2_sq < 0.00806:
        z += -84.0 * Q.e2_sq - 0.34896
    if 0.00806 <= Q.e2_sq < 0.00956:
        z += 684.0 * Q.e2_sq - 6.53904
    if Q.e2_sq >= 0.0252:
        z += -288.0 * Q.e2_sq + 7.2576
    if Q.girth < 0.0171:
        z += 40.92 * Q.girth - 1.440612
    if 0.0171 <= Q.girth < 0.0387:
        z += 34.3 * Q.girth - 1.32741
    if Q.n_dr_0p2_0p4 < 13.2:
        z += 0.07 * Q.n_dr_0p2_0p4 - 0.924
    if Q.psi_0p2 < 0.991:
        z += 11.3 * Q.psi_0p2 - 11.1983
    if Q.sum_pt < 968.0:
        z += 0.00485 * Q.sum_pt - 4.6948
    if Q.tau21_b2 < 0.258:
        z += 1.85 * Q.tau21_b2 - 0.4773
    if Q.z_dr_0p2_0p4 >= 0.0617:
        z += 11.8 * Q.z_dr_0p2_0p4 - 0.72806
    if Q.z_top5 < 0.535:
        z += -2.34 * Q.z_top5 + 1.29168
    if 0.535 <= Q.z_top5 < 0.552:
        z += -4.47 * Q.z_top5 + 2.43123
    if Q.z_top5 >= 0.552:
        z += -2.13 * Q.z_top5 + 1.13955
    if Q.e2 > 0.00739 and Q.log_sum_pt > 6.9:
        z += -121.0 * (Q.e2 - 0.00739) * (Q.log_sum_pt - 6.9)
    if Q.girth < 0.142 and Q.pt1_dr01 > 9.54:
        z += 0.331 * (0.142 - Q.girth) * (Q.pt1_dr01 - 9.54)
    if Q.girth < 0.147 and Q.ptdr0_2 > 10.1:
        z += 0.374 * (0.147 - Q.girth) * (Q.ptdr0_2 - 10.1)
    if Q.n_dr_0p2_0p4 < 14.1 and Q.max_dr > 0.22:
        z += 0.12 * (14.1 - Q.n_dr_0p2_0p4) * (Q.max_dr - 0.22)
    if Q.sum_pt < 1030.0 and Q.eccentricity > 0.606:
        z += -0.0107 * (1030.0 - Q.sum_pt) * (Q.eccentricity - 0.606)
    if Q.sum_pt < 1020.0 and Q.z_dr_0p05_0p1 < 0.551:
        z += 0.00906 * (1020.0 - Q.sum_pt) * (0.551 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_11(Q):
    z = -1.21
    if Q.LHA >= 0.308:
        z += -16.0 * Q.LHA + 4.928
    if Q.e2 < 0.0259:
        z += -37.8 * Q.e2 + 0.2227
    if 0.0259 <= Q.e2 < 0.0422:
        z += 46.4 * Q.e2 - 1.95808
    if Q.e2_sq < 0.00493:
        z += -79.0 * Q.e2_sq + 1.60387
    if 0.00493 <= Q.e2_sq < 0.00823:
        z += -368.0 * Q.e2_sq + 3.02864
    if Q.girth >= 0.132:
        z += 28.9 * Q.girth - 3.8148
    if Q.girth2_top10 < 0.00124:
        z += -341.0 * Q.girth2_top10 + 0.42284
    if Q.girth2_top15 >= 0.00761:
        z += 36.6 * Q.girth2_top15 - 0.278526
    if Q.girth2_top30 < 0.00642:
        z += 168.0 * Q.girth2_top30 + 0.20432
    if 0.00642 <= Q.girth2_top30 < 0.0125:
        z += -211.0 * Q.girth2_top30 + 2.6375
    if Q.log_sum_pt >= 6.98:
        z += -0.812 * Q.log_sum_pt + 5.66776
    if Q.n_dr_0p1_0p2 >= 8.57:
        z += -0.0204 * Q.n_dr_0p1_0p2 + 0.174828
    if Q.n_particles < 63.7:
        z += -0.00824 * Q.n_particles + 0.524888
    if Q.sum_pt < 1130.0:
        z += -0.00411 * Q.sum_pt + 4.6443
    if Q.sum_pt_top30 < 916.0:
        z += 0.00261 * Q.sum_pt_top30 - 2.39076
    if Q.z_dr_0p2_0p4 < 0.00462:
        z += -60.7 * Q.z_dr_0p2_0p4 + 0.280434
    if Q.e2_sq < 0.00914 and Q.sum_pt < 1140.0:
        z += -0.78 * (0.00914 - Q.e2_sq) * (1140.0 - Q.sum_pt)
    if Q.e2_sq < 0.00918 and Q.z_dr_0p1_0p2 > 0.0887:
        z += 717.0 * (0.00918 - Q.e2_sq) * (Q.z_dr_0p1_0p2 - 0.0887)
    if Q.n_dr_0p2_0p4 < 8.1 and Q.girth2 < 0.00612:
        z += -37.1 * (8.1 - Q.n_dr_0p2_0p4) * (0.00612 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 7.58 and Q.n_dr_0p1_0p2 < 22.4:
        z += 0.00956 * (7.58 - Q.n_dr_0p2_0p4) * (22.4 - Q.n_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 13.5 and Q.sum_pt_top50 < 1110.0:
        z += 0.000241 * (13.5 - Q.n_dr_0p2_0p4) * (1110.0 - Q.sum_pt_top50)
    if Q.z_top30_slots > 0.977 and Q.D2_b2 < 1.27:
        z += 13.1 * (Q.z_top30_slots - 0.977) * (1.27 - Q.D2_b2)
    return max(0.0, z)


def neuron_12(Q):
    z = 0.634
    if Q.LHA >= 0.366:
        z += 14.7 * Q.LHA - 5.3802
    if Q.e2_sq < 0.00634:
        z += 7.0 * Q.e2_sq + 0.92162
    if 0.00634 <= Q.e2_sq < 0.00749:
        z += -840.0 * Q.e2_sq + 6.2916
    if Q.girth2 >= 0.00386:
        z += -38.3 * Q.girth2 + 0.147838
    if Q.girth2_top40 < 0.0248:
        z += 61.4 * Q.girth2_top40 - 1.52272
    if Q.lam1 < 0.0084:
        z += -213.0 * Q.lam1 + 1.7892
    if Q.lam2 < 0.000736:
        z += -269.0 * Q.lam2 + 0.43578
    if 0.000736 <= Q.lam2 < 0.00162:
        z += -336.3 * Q.lam2 + 0.4853128
    if Q.lam2 >= 0.00162:
        z += -67.3 * Q.lam2 + 0.0495328
    if Q.log_sum_pt < 6.81:
        z += -3.09 * Q.log_sum_pt + 21.0429
    if Q.sum_pt_top20 >= 840.0:
        z += 0.00153 * Q.sum_pt_top20 - 1.2852
    if Q.sum_pt_top30 < 960.0:
        z += 0.00267 * Q.sum_pt_top30 - 2.5632
    if Q.z_dr_0_0p05 >= 0.888:
        z += -9.66 * Q.z_dr_0_0p05 + 8.57808
    if Q.z_top20_slots >= 0.755:
        z += -1.88 * Q.z_top20_slots + 1.4194
    if Q.e2_sq < 0.00762 and Q.log_sum_pt > 6.81:
        z += -8260.0 * (0.00762 - Q.e2_sq) * (Q.log_sum_pt - 6.81)
    if Q.e2_sq < 0.00851 and Q.zdr_0 > 0.00435:
        z += 6620.0 * (0.00851 - Q.e2_sq) * (Q.zdr_0 - 0.00435)
    if Q.girth2 < 0.00645 and Q.log_sum_pt > 6.81:
        z += 9360.0 * (0.00645 - Q.girth2) * (Q.log_sum_pt - 6.81)
    if Q.girth2 < 0.0063 and Q.n_for_50pct < 8.74:
        z += 13.7 * (0.0063 - Q.girth2) * (8.74 - Q.n_for_50pct)
    if Q.girth2_top30 < 0.00609 and Q.z_dr_0p05_0p1 < 0.0803:
        z += 1570.0 * (0.00609 - Q.girth2_top30) * (0.0803 - Q.z_dr_0p05_0p1)
    if Q.psi_0p3 > 0.991 and Q.girth2 < 0.00902:
        z += 8630.0 * (Q.psi_0p3 - 0.991) * (0.00902 - Q.girth2)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.638
    if Q.e2 >= 0.0499:
        z += 25.5 * Q.e2 - 1.27245
    if Q.e2_sq >= 0.009:
        z += -151.0 * Q.e2_sq + 1.359
    if Q.girth < 0.103:
        z += 5.19 * Q.girth - 0.53457
    if Q.girth2_top40 >= 0.0102:
        z += -325.0 * Q.girth2_top40 + 3.315
    if Q.girth2_top50 >= 0.00599:
        z += 134.0 * Q.girth2_top50 - 0.80266
    if 6.86 <= Q.log_sum_pt < 6.91:
        z += 29.0 * Q.log_sum_pt - 198.94
    if Q.log_sum_pt >= 6.91:
        z += 20.16 * Q.log_sum_pt - 137.8556
    if Q.n_pt_above_5 < 25.1:
        z += 0.0391 * Q.n_pt_above_5 - 0.98141
    if Q.sum_pt < 957.0:
        z += 0.0588 * Q.sum_pt - 59.3306
    if 957.0 <= Q.sum_pt < 1020.0:
        z += 0.023 * Q.sum_pt - 25.07
    if 1020.0 <= Q.sum_pt < 1090.0:
        z += 0.0071 * Q.sum_pt - 8.852
    if Q.sum_pt >= 1090.0:
        z += -0.0159 * Q.sum_pt + 16.218
    if Q.sum_pt_top15 >= 1090.0:
        z += 0.00216 * Q.sum_pt_top15 - 2.3544
    if Q.sum_pt_top30 < 938.0:
        z += 0.00466 * Q.sum_pt_top30 - 4.37108
    if Q.sum_pt_top50 < 1010.0:
        z += -0.02476 * Q.sum_pt_top50 + 26.2396
    if 1010.0 <= Q.sum_pt_top50 < 1090.0:
        z += -0.0154 * Q.sum_pt_top50 + 16.786
    if 0.0195 <= Q.width < 0.0238:
        z += -169.0 * Q.width + 3.2955
    if Q.width >= 0.0238:
        z += -339.0 * Q.width + 7.3415
    if Q.girth < 0.0936 and Q.z_dr_0p05_0p1 > 0.092:
        z += -14.4 * (0.0936 - Q.girth) * (Q.z_dr_0p05_0p1 - 0.092)
    if Q.girth2_top40 > 0.0285 and Q.pt_6 > 9.58:
        z += 6.1 * (Q.girth2_top40 - 0.0285) * (Q.pt_6 - 9.58)
    if Q.girth2_top40 > 0.00976 and Q.sum_pt_top50 < 1310.0:
        z += 1.05 * (Q.girth2_top40 - 0.00976) * (1310.0 - Q.sum_pt_top50)
    if Q.girth2_top50 > 0.0104 and Q.soft7_z > 0.0013:
        z += -31300.0 * (Q.girth2_top50 - 0.0104) * (Q.soft7_z - 0.0013)
    if Q.log_sum_pt > 6.8 and Q.C2 > 0.0763:
        z += -33.2 * (Q.log_sum_pt - 6.8) * (Q.C2 - 0.0763)
    if Q.n_pt_above_5 < 23.8 and Q.D2 < 5.73:
        z += 0.00829 * (23.8 - Q.n_pt_above_5) * (5.73 - Q.D2)
    if Q.sum_pt > 1050.0 and Q.absphi_2 > 0.0223:
        z += -0.0109 * (Q.sum_pt - 1050.0) * (Q.absphi_2 - 0.0223)
    if Q.sum_pt < 1050.0 and Q.tau21 > 0.118:
        z += 0.0179 * (1050.0 - Q.sum_pt) * (Q.tau21 - 0.118)
    if Q.sum_pt_top10 > 847.0 and Q.dr_max_012 < 0.0999:
        z += -0.0198 * (Q.sum_pt_top10 - 847.0) * (0.0999 - Q.dr_max_012)
    if Q.sum_pt_top50 < 1030.0 and Q.tau32 > 0.449:
        z += 0.0146 * (1030.0 - Q.sum_pt_top50) * (Q.tau32 - 0.449)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.232
    if Q.LHA >= 0.323:
        z += -13.0 * Q.LHA + 4.199
    if Q.e2 < 0.0344:
        z += -74.0 * Q.e2 + 3.09086
    if 0.0344 <= Q.e2 < 0.0543:
        z += -27.4 * Q.e2 + 1.48782
    if Q.e2_sq < 0.00364:
        z += -140.6 * Q.e2_sq + 2.308916
    if 0.00364 <= Q.e2_sq < 0.00681:
        z += -401.6 * Q.e2_sq + 3.258956
    if 0.00681 <= Q.e2_sq < 0.00693:
        z += -337.0 * Q.e2_sq + 2.81903
    if 0.00693 <= Q.e2_sq < 0.00962:
        z += -760.0 * Q.e2_sq + 5.75042
    if Q.e2_sq >= 0.00962:
        z += -261.0 * Q.e2_sq + 0.95004
    if Q.e3 < 8.18e-05:
        z += 6400.0 * Q.e3 - 0.52352
    if Q.girth2 < 0.00826:
        z += 503.0 * Q.girth2 - 4.15478
    if Q.girth2_top20 >= 0.00605:
        z += 97.9 * Q.girth2_top20 - 0.592295
    if Q.girth2_top30 < 0.0115:
        z += 187.0 * Q.girth2_top30 - 2.1505
    if Q.girth2_top50 < 0.00648:
        z += 85.0 * Q.girth2_top50 - 1.5466
    if 0.00648 <= Q.girth2_top50 < 0.00908:
        z += 383.0 * Q.girth2_top50 - 3.47764
    if Q.lam1 >= 0.00626:
        z += 125.0 * Q.lam1 - 0.7825
    if Q.lam2 >= 0.00314:
        z += 168.0 * Q.lam2 - 0.52752
    if 7.0 <= Q.log_sum_pt < 7.06:
        z += -0.409 * Q.log_sum_pt + 2.863
    if Q.log_sum_pt >= 7.06:
        z += -7.629 * Q.log_sum_pt + 53.8362
    if Q.max_dr < 0.247:
        z += 0.88 * Q.max_dr - 0.0078
    if 0.247 <= Q.max_dr < 0.416:
        z += -1.24 * Q.max_dr + 0.51584
    if Q.n_dr_0p1_0p2 < 16.7:
        z += 0.0161 * Q.n_dr_0p1_0p2 - 0.26887
    if Q.psi_0p3 >= 0.995:
        z += 315.0 * Q.psi_0p3 - 313.425
    if Q.sj3_dr_max >= 0.238:
        z += -1.06 * Q.sj3_dr_max + 0.25228
    if Q.soft10_pt < 1.99:
        z += 0.133 * Q.soft10_pt - 0.26467
    if Q.sum_pt < 963.0:
        z += 0.00418 * Q.sum_pt - 4.02534
    if Q.sum_pt_top30 < 1090.0:
        z += -0.00326 * Q.sum_pt_top30 + 3.5534
    if Q.tau1 >= 0.0459:
        z += 11.4 * Q.tau1 - 0.52326
    if Q.tau21_b2 < 0.139:
        z += -1.67 * Q.tau21_b2 + 0.23213
    if Q.z_top30_slots < 0.914:
        z += 9.28 * Q.z_top30_slots - 8.48192
    if Q.z_top30_slots >= 0.921:
        z += 5.74 * Q.z_top30_slots - 5.28654
    if Q.e2 < 0.0317 and Q.sum_pt_top30 > 859.0:
        z += -0.127 * (0.0317 - Q.e2) * (Q.sum_pt_top30 - 859.0)
    if Q.e2_sq < 0.0127 and Q.z_top50_slots < 0.994:
        z += 2410.0 * (0.0127 - Q.e2_sq) * (0.994 - Q.z_top50_slots)
    if Q.girth2 < 0.00802 and Q.psi_0p3 > 0.995:
        z += -64400.0 * (0.00802 - Q.girth2) * (Q.psi_0p3 - 0.995)
    if Q.girth2_top30 < 0.0117 and Q.sum_pt > 905.0:
        z += 0.84 * (0.0117 - Q.girth2_top30) * (Q.sum_pt - 905.0)
    if Q.psi_0p3 > 0.996 and Q.sum_pt_top40 < 1030.0:
        z += 1.38 * (Q.psi_0p3 - 0.996) * (1030.0 - Q.sum_pt_top40)
    if Q.psi_0p3 > 0.996 and Q.sum_pt_top40 < 1240.0:
        z += -1.28 * (Q.psi_0p3 - 0.996) * (1240.0 - Q.sum_pt_top40)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.217
    if Q.LHA >= 0.106:
        z += 6.08 * Q.LHA - 0.64448
    if Q.e2 < 0.0311:
        z += -56.9 * Q.e2 + 2.56664
    if 0.0311 <= Q.e2 < 0.04:
        z += -83.9 * Q.e2 + 3.40634
    if 0.04 <= Q.e2 < 0.0406:
        z += -144.1 * Q.e2 + 5.81434
    if 0.0406 <= Q.e2 < 0.0551:
        z += -60.2 * Q.e2 + 2.408
    if Q.e2 >= 0.0551:
        z += -22.0 * Q.e2 + 0.30318
    if Q.e3 < 0.000317:
        z += 3630.0 * Q.e3 - 1.15071
    if Q.girth >= 0.0599:
        z += -31.3 * Q.girth + 1.87487
    if Q.girth2_top30 >= 0.00634:
        z += 64.8 * Q.girth2_top30 - 0.410832
    if Q.girth2_top5 < 0.00211:
        z += 98.8 * Q.girth2_top5 + 0.134672
    if 0.00211 <= Q.girth2_top5 < 0.00781:
        z += -60.2 * Q.girth2_top5 + 0.470162
    if Q.girth2_top50 < 0.00813:
        z += 91.5 * Q.girth2_top50 - 0.743895
    if Q.lam2 < 0.00284:
        z += -156.0 * Q.lam2 + 0.44304
    if Q.log_sum_pt < 6.98:
        z += -7.47 * Q.log_sum_pt + 52.1406
    if Q.n_dr_0p1_0p2 < 12.8:
        z += -0.032 * Q.n_dr_0p1_0p2 + 0.4096
    if Q.n_particles < 45.2:
        z += -0.0183 * Q.n_particles + 0.82716
    if Q.psi_0p2 >= 0.997:
        z += -74.1 * Q.psi_0p2 + 73.8777
    if Q.psi_0p3 >= 0.988:
        z += -29.6 * Q.psi_0p3 + 29.2448
    if 0.148 <= Q.sd_rg < 0.218:
        z += -5.61 * Q.sd_rg + 0.83028
    if 0.218 <= Q.sd_rg < 0.282:
        z += 12.99 * Q.sd_rg - 3.22452
    if Q.sd_rg >= 0.282:
        z += 0.19 * Q.sd_rg + 0.38508
    if Q.sum_pt < 1000.0:
        z += 0.00461 * Q.sum_pt - 4.61
    if Q.sum_pt_top20 >= 847.0:
        z += 0.00144 * Q.sum_pt_top20 - 1.21968
    if Q.sum_pt_top40 < 1070.0:
        z += 0.00751 * Q.sum_pt_top40 - 8.0357
    if 1010.0 <= Q.sum_pt_top50 < 1110.0:
        z += -0.0078 * Q.sum_pt_top50 + 7.878
    if Q.sum_pt_top50 >= 1110.0:
        z += -0.00067 * Q.sum_pt_top50 - 0.0363
    if Q.z_dr_0p05_0p1 < 0.54:
        z += 0.495 * Q.z_dr_0p05_0p1 - 0.2673
    if Q.e3 < 5.59e-05 and Q.n_dr_0p1_0p2 < 21.3:
        z += -678.0 * (5.59e-05 - Q.e3) * (21.3 - Q.n_dr_0p1_0p2)
    if Q.girth2_top3 < 0.000989 and Q.eccentricity > 0.788:
        z += -2790.0 * (0.000989 - Q.girth2_top3) * (Q.eccentricity - 0.788)
    if Q.girth2_top5 < 0.00776 and Q.n_dr_0p2_0p4 > 10.2:
        z += -3.96 * (0.00776 - Q.girth2_top5) * (Q.n_dr_0p2_0p4 - 10.2)
    if Q.sd_rg < 0.143 and Q.planar_flow < 0.575:
        z += 9.35 * (0.143 - Q.sd_rg) * (0.575 - Q.planar_flow)
    if Q.z_dr_0p1_0p2 < 0.129 and Q.C2_b2 < 0.0567:
        z += 57.6 * (0.129 - Q.z_dr_0p1_0p2) * (0.0567 - Q.C2_b2)
    if Q.z_dr_0p1_0p2 < 0.105 and Q.psi_0p3 > 0.979:
        z += -112.0 * (0.105 - Q.z_dr_0p1_0p2) * (Q.psi_0p3 - 0.979)
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
