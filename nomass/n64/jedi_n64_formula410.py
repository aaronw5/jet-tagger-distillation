"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no mass observables), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.6% (the network: 81.1%); same class as the network for 92.1% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.z_6                    pT of particle 6 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_z                pT share of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
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
  Q.soft7_dr0              ΔR between the hardest and the 7. softest real particle (0 if among the 15 hardest)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_8                   ΔR of particle 8 from the jet axis
  Q.soft5_dr               ΔR from the jet axis of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
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
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_phi               pT-weighted mean Δφ
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
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        pt_6=pt[6],
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        z_6=z[6],
        soft1_z=softp(1, 'z'),
        soft2_z=softp(2, 'z'),
        soft5_z=softp(5, 'z'),
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
        soft7_dr0=softp(7, 'dr0'),
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_8=dr[8] if pt[8] > 0 else 0.0,
        soft5_dr=softp(5, 'dr'),
        eta_1=eta[1],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
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
        n_pt_above_5=sum(1 for x in pt if x > 5),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
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
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    z = -1.244176
    if Q.girth2_top40 < 0.003113918:
        z += 264.6285 * Q.girth2_top40 - 0.3253222
    if 0.003113918 <= Q.girth2_top40 < 0.006259772:
        z += -42.76824 * Q.girth2_top40 + 0.6318861
    if 0.006259772 <= Q.girth2_top40 < 0.008840538:
        z += -141.108 * Q.girth2_top40 + 1.24747
    if Q.girth2 < 0.008190222:
        z += -176.878 * Q.girth2 + 1.44867
    if Q.e2_sq < 0.00616708:
        z += -548.8941 * Q.e2_sq + 4.190422
    if 0.00616708 <= Q.e2_sq < 0.006936725:
        z += -1190.205 * Q.e2_sq + 8.14544
    if 0.006936725 <= Q.e2_sq < 0.007872294:
        z += -480.206 * Q.e2_sq + 3.22037
    if 0.007872294 <= Q.e2_sq < 0.009606007:
        z += 322.9788 * Q.e2_sq - 3.102537
    if 858.8262 <= Q.sum_pt_top40 < 1024.942:
        z += 0.00601724 * Q.sum_pt_top40 - 5.167763
    if Q.sum_pt_top40 >= 1024.942:
        z += -0.0003819633 * Q.sum_pt_top40 + 1.391052
    if Q.psi_0p3 >= 0.9943058:
        z += 92.94341 * Q.psi_0p3 - 92.41417
    if Q.lam1 < 0.005913555:
        z += 110.2964 * Q.lam1 - 0.6522439
    if Q.tau1 < 0.0705748:
        z += 16.90764 * Q.tau1 - 1.193253
    if 1002.379 <= Q.sum_pt < 1115.723:
        z += -0.01280559 * Q.sum_pt + 12.83605
    if Q.sum_pt >= 1115.723:
        z += -0.006254509 * Q.sum_pt + 5.526858
    if Q.girth2_top30 < 0.006363916:
        z += 99.84943 * Q.girth2_top30 - 0.3813597
    if 0.006363916 <= Q.girth2_top30 < 0.008376291:
        z += -126.2557 * Q.girth2_top30 + 1.057554
    if Q.girth2_top50 < 0.008124776:
        z += -246.1815 * Q.girth2_top50 + 2.00017
    if Q.tau21_b2 < 0.2708235:
        z += -2.227567 * Q.tau21_b2 + 0.6032776
    if Q.girth2 < 0.008190222 and Q.girth2_top15 < 0.006142802:
        z += -20078.21 * (0.008190222 - Q.girth2) * (0.006142802 - Q.girth2_top15)
    if Q.girth2_top50 < 0.005852839 and Q.sum_pt < 1260.541:
        z += -2.279041 * (0.005852839 - Q.girth2_top50) * (1260.541 - Q.sum_pt)
    if Q.sum_pt > 1167.447 and Q.soft7_dr0 > 0.2576672:
        z += -0.07911974 * (Q.sum_pt - 1167.447) * (Q.soft7_dr0 - 0.2576672)
    if Q.log_sum_pt > 6.910131 and Q.e3 < 0.0001086251:
        z += -22474.45 * (Q.log_sum_pt - 6.910131) * (0.0001086251 - Q.e3)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.243648
    if Q.n_particles >= 38.0:
        z += 0.04705613 * Q.n_particles - 1.788133
    if Q.log_sum_pt < 6.910131:
        z += 8.506829 * Q.log_sum_pt - 60.73277
    if 6.910131 <= Q.log_sum_pt < 6.98945:
        z += 49.17498 * Q.log_sum_pt - 341.7551
    if 6.98945 <= Q.log_sum_pt < 7.139296:
        z += 19.03733 * Q.log_sum_pt - 131.1094
    if Q.log_sum_pt >= 7.139296:
        z += 10.5305 * Q.log_sum_pt - 70.37662
    if Q.sum_pt_top50 < 934.2416:
        z += -0.01250375 * Q.sum_pt_top50 + 13.49147
    if 934.2416 <= Q.sum_pt_top50 < 1078.994:
        z += -0.02999276 * Q.sum_pt_top50 + 29.83044
    if Q.sum_pt_top50 >= 1078.994:
        z += -0.01748902 * Q.sum_pt_top50 + 16.33897
    if 0.4670285 <= Q.z_top5 < 0.6551948:
        z += -4.044255 * Q.z_top5 + 1.888782
    if Q.z_top5 >= 0.6551948:
        z += 0.007111073 * Q.z_top5 - 0.7656519
    if Q.psi_0p3 >= 0.9966167:
        z += -153.9235 * Q.psi_0p3 + 153.4027
    if Q.tau21 < 0.2140276:
        z += -15.42558 * Q.tau21 + 3.3015
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.08529303 * Q.n_dr_0p2_0p4 - 0.5970512
    if 972.0419 <= Q.sum_pt < 1052.889:
        z += 0.01732927 * Q.sum_pt - 16.84477
    if Q.sum_pt >= 1052.889:
        z += 0.01408786 * Q.sum_pt - 13.43193
    if Q.n_dr_0_0p05 >= 15.0:
        z += 0.03493999 * Q.n_dr_0_0p05 - 0.5240999
    if Q.n_dr_0p1_0p2 < 7.0:
        z += 0.1101801 * Q.n_dr_0p1_0p2 - 0.7712609
    if Q.D2_b2 < 0.2773918:
        z += 4.262991 * Q.D2_b2 - 1.182519
    if Q.lam2 < 0.001776308:
        z += 318.9135 * Q.lam2 - 0.5664887
    if Q.e2_sq < 0.009606007:
        z += 105.7272 * Q.e2_sq - 1.015617
    if Q.sum_pt_top40 < 1225.842:
        z += -0.003349801 * Q.sum_pt_top40 + 4.106327
    if Q.girth2_top30 < 0.007463985:
        z += -248.6274 * Q.girth2_top30 + 1.855751
    if Q.e2 < 0.02210818:
        z += 58.89648 * Q.e2 - 1.302094
    if Q.girth2_top20 < 0.001153063:
        z += -988.0237 * Q.girth2_top20 + 1.139254
    if Q.girth2 < 0.001784491:
        z += 434.4817 * Q.girth2 - 0.7753287
    if Q.z_top30_slots >= 0.9564984:
        z += 23.67869 * Q.z_top30_slots - 22.64863
    if Q.LHA < 0.2454112:
        z += -13.90696 * Q.LHA + 3.412924
    if Q.zdr_0 >= 0.01240028:
        z += -45.58742 * Q.zdr_0 + 0.5652966
    if Q.n_particles > 38.0 and Q.zdr_0 < 0.009970338:
        z += 3.985883 * (Q.n_particles - 38.0) * (0.009970338 - Q.zdr_0)
    if Q.n_particles > 38.0 and Q.soft1_z < 0.002181998:
        z += -23.75054 * (Q.n_particles - 38.0) * (0.002181998 - Q.soft1_z)
    if Q.z_top5 > 0.6551948 and Q.pt1_dr01 < 28.39396:
        z += -0.2244428 * (Q.z_top5 - 0.6551948) * (28.39396 - Q.pt1_dr01)
    if Q.girth2_top15 < 0.002197765 and Q.psi_0p3 > 0.9966167:
        z += 126081.3 * (0.002197765 - Q.girth2_top15) * (Q.psi_0p3 - 0.9966167)
    if Q.z_top30_slots > 0.9341838 and Q.pt1_dr01 > 5.351077:
        z += 0.4475637 * (Q.z_top30_slots - 0.9341838) * (Q.pt1_dr01 - 5.351077)
    if Q.n_particles > 38.0 and Q.tau32 > 0.3293142:
        z += 0.1145475 * (Q.n_particles - 38.0) * (Q.tau32 - 0.3293142)
    if Q.M3 < 0.03457336 and Q.psi_0p3 > 0.9299135:
        z += -370.5331 * (0.03457336 - Q.M3) * (Q.psi_0p3 - 0.9299135)
    if Q.n_particles > 38.0 and Q.tau43 < 0.9624339:
        z += -0.153108 * (Q.n_particles - 38.0) * (0.9624339 - Q.tau43)
    if Q.z_dr_0_0p05 > 0.8103116 and Q.n_dr_0p05_0p1 < 10.0:
        z += -0.718922 * (Q.z_dr_0_0p05 - 0.8103116) * (10.0 - Q.n_dr_0p05_0p1)
    if Q.z_top30_slots > 0.9564984 and Q.ptdr0_3 < 10.17512:
        z += -1.462549 * (Q.z_top30_slots - 0.9564984) * (10.17512 - Q.ptdr0_3)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.6701581
    if 6.930088 <= Q.log_sum_pt < 7.062574:
        z += 10.8292 * Q.log_sum_pt - 75.04727
    if Q.log_sum_pt >= 7.062574:
        z += -0.01690102 * Q.log_sum_pt + 1.554085
    if Q.girth2 < 0.003638856:
        z += 1013.803 * Q.girth2 - 5.520105
    if 0.003638856 <= Q.girth2 < 0.006941794:
        z += 554.3613 * Q.girth2 - 3.848262
    if Q.girth2_top15 < 0.01563836:
        z += -77.65173 * Q.girth2_top15 + 1.214345
    if 1007.788 <= Q.sum_pt < 1085.125:
        z += 0.006139109 * Q.sum_pt - 6.186923
    if Q.sum_pt >= 1085.125:
        z += 0.00027757 * Q.sum_pt + 0.1735787
    if Q.girth2_top50 < 0.00634935:
        z += -586.1411 * Q.girth2_top50 + 3.721615
    if Q.sum_pt_top50 >= 1003.544:
        z += -0.006597779 * Q.sum_pt_top50 + 6.621163
    if Q.sum_pt_top20 >= 993.5664:
        z += 0.002610055 * Q.sum_pt_top20 - 2.593263
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.005372643 * Q.sum_pt_top40 - 5.746961
    if 800.732 <= Q.sum_pt_top30 < 1011.524:
        z += -0.001538473 * Q.sum_pt_top30 + 1.231904
    if Q.sum_pt_top30 >= 1011.524:
        z += -0.003902892 * Q.sum_pt_top30 + 3.623571
    if Q.z_top30_slots >= 0.9203881:
        z += 4.877331 * Q.z_top30_slots - 4.489038
    if Q.log_sum_pt > 6.903423 and Q.psi_0p1 < 0.8747961:
        z += -44.26139 * (Q.log_sum_pt - 6.903423) * (0.8747961 - Q.psi_0p1)
    if Q.sum_pt_top30 > 1111.245 and Q.z_dr_0p1_0p2 < 0.1203437:
        z += 0.03721369 * (Q.sum_pt_top30 - 1111.245) * (0.1203437 - Q.z_dr_0p1_0p2)
    if Q.sum_pt_top20 > 993.5664 and Q.dr_8 < 0.2319351:
        z += -0.01500013 * (Q.sum_pt_top20 - 993.5664) * (0.2319351 - Q.dr_8)
    if Q.sum_pt_top50 > 1003.544 and Q.z_dr_0p1_0p2 > 0.003353111:
        z += 0.0336685 * (Q.sum_pt_top50 - 1003.544) * (Q.z_dr_0p1_0p2 - 0.003353111)
    if Q.sum_pt_top50 > 1003.544 and Q.dr_4 < 0.1289397:
        z += 0.02481388 * (Q.sum_pt_top50 - 1003.544) * (0.1289397 - Q.dr_4)
    if Q.sum_pt_top15 > 1003.329 and Q.eta_1 > 0.08734131:
        z += -4.929738 * (Q.sum_pt_top15 - 1003.329) * (Q.eta_1 - 0.08734131)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.1209226
    if Q.lam2 < 0.0006154841:
        z += -1359.756 * Q.lam2 + 0.8369081
    if Q.tau21 < 0.347196:
        z += 10.50758 * Q.tau21 - 3.648189
    if Q.D2 < 1.788105:
        z += -0.8225018 * Q.D2 + 1.47072
    if Q.e2_sq < 0.007511864:
        z += -370.1218 * Q.e2_sq + 2.780305
    if Q.LHA < 0.3719813:
        z += 4.468407 * Q.LHA - 1.662164
    if Q.psi_0p2 >= 0.9976427:
        z += 294.6822 * Q.psi_0p2 - 293.9875
    if Q.girth2_top40 < 0.006259772:
        z += 208.7273 * Q.girth2_top40 - 1.104781
    if 0.006259772 <= Q.girth2_top40 < 0.008840538:
        z += -78.19556 * Q.girth2_top40 + 0.6912908
    if Q.tau21_b2 < 0.2018786:
        z += -9.212946 * Q.tau21_b2 + 1.859897
    if Q.n_particles < 46.0 and Q.sum_pt_top40 > 858.8262:
        z += 0.0001496675 * (46.0 - Q.n_particles) * (Q.sum_pt_top40 - 858.8262)
    if Q.D2 < 1.788105 and Q.soft5_z < 0.003582374:
        z += -202.0706 * (1.788105 - Q.D2) * (0.003582374 - Q.soft5_z)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.5037524
    if Q.e2 < 0.03263075:
        z += -26.61235 * Q.e2 + 0.8683811
    if Q.e2_sq < 0.00592208:
        z += 374.4352 * Q.e2_sq - 2.217435
    if Q.n_particles >= 29.0:
        z += -0.02035368 * Q.n_particles + 0.5902566
    if Q.sum_pt_top50 < 1061.184:
        z += 0.01353166 * Q.sum_pt_top50 - 14.35958
    if Q.girth2_top20 < 0.007538019:
        z += 76.54826 * Q.girth2_top20 - 0.06843392
    if 0.007538019 <= Q.girth2_top20 < 0.01083435:
        z += -154.289 * Q.girth2_top20 + 1.671622
    if Q.tau2 < 0.1011219:
        z += 10.18882 * Q.tau2 - 1.030313
    if 0.00242543 <= Q.girth2_top50 < 0.004573744:
        z += -118.5116 * Q.girth2_top50 + 0.2874417
    if Q.girth2_top50 >= 0.004573744:
        z += 49.49422 * Q.girth2_top50 - 0.4809739
    if 0.002197765 <= Q.girth2_top15 < 0.004169954:
        z += 135.8704 * Q.girth2_top15 - 0.2986111
    if 0.004169954 <= Q.girth2_top15 < 0.007887677:
        z += 104.6915 * Q.girth2_top15 - 0.1685965
    if Q.girth2_top15 >= 0.007887677:
        z += -34.44778 * Q.girth2_top15 + 0.9288889
    if Q.n_dr_0p2_0p4 < 26.0:
        z += -0.02218575 * Q.n_dr_0p2_0p4 + 0.5768296
    if Q.lam1 < 0.006716737:
        z += 164.8251 * Q.lam1 - 1.107087
    if Q.width < 0.009614971:
        z += -244.5062 * Q.width + 2.35092
    if 0.1870291 <= Q.LHA < 0.2454112:
        z += 6.153758 * Q.LHA - 1.150932
    if Q.LHA >= 0.2454112:
        z += -2.654126 * Q.LHA + 1.010622
    if Q.girth >= 0.076787:
        z += 13.69571 * Q.girth - 1.051653
    if Q.girth2_top40 < 0.008031986:
        z += 212.6356 * Q.girth2_top40 - 1.707886
    if Q.n_dr_0p1_0p2 < 26.0:
        z += -0.02288938 * Q.n_dr_0p1_0p2 + 0.5951238
    if Q.z_top50_slots < 0.985099:
        z += 23.06822 * Q.z_top50_slots - 22.72448
    if Q.sum_pt_top40 < 1053.047:
        z += -0.01088362 * Q.sum_pt_top40 + 11.46097
    if Q.psi_0p3 > 0.9973959 and Q.sum_pt_top40 > 858.8262:
        z += 0.892721 * (Q.psi_0p3 - 0.9973959) * (Q.sum_pt_top40 - 858.8262)
    if Q.sj2_dr > 0.2595052 and Q.C2_b2 < 0.0329485:
        z += -572.1375 * (Q.sj2_dr - 0.2595052) * (0.0329485 - Q.C2_b2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.2886304
    z += -0.02820985 * Q.n_particles + 1.80543
    if Q.e2_sq >= 0.007872294:
        z += -338.0414 * Q.e2_sq + 2.661162
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -24.04999 * Q.log_sum_pt + 166.1886
    if 6.920349 <= Q.log_sum_pt < 6.935549:
        z += -34.14145 * Q.log_sum_pt + 236.025
    if 6.935549 <= Q.log_sum_pt < 6.959294:
        z += -39.81966 * Q.log_sum_pt + 275.4065
    if 6.959294 <= Q.log_sum_pt < 7.017258:
        z += -33.1258 * Q.log_sum_pt + 228.822
    if Q.log_sum_pt >= 7.017258:
        z += -22.02537 * Q.log_sum_pt + 150.9274
    if Q.sum_pt >= 907.9372:
        z += 0.006753823 * Q.sum_pt - 6.132047
    if Q.girth2 >= 0.02928196:
        z += 116.6486 * Q.girth2 - 3.415698
    if Q.width >= 0.006941794:
        z += 246.991 * Q.width - 1.71456
    if 934.2416 <= Q.sum_pt_top50 < 1048.098:
        z += 0.01854411 * Q.sum_pt_top50 - 17.32467
    if Q.sum_pt_top50 >= 1048.098:
        z += 0.005589158 * Q.sum_pt_top50 - 3.746617
    if Q.n_pt_above_1 >= 26.0:
        z += 0.01795097 * Q.n_pt_above_1 - 0.4667253
    if Q.sum_pt_top20 >= 1017.778:
        z += 0.006642913 * Q.sum_pt_top20 - 6.761012
    if Q.sum_pt_top5 < 531.1875:
        z += 0.000508605 * Q.sum_pt_top5 - 1.146376
    if 531.1875 <= Q.sum_pt_top5 < 579.875:
        z += -0.002797542 * Q.sum_pt_top5 + 0.6098078
    if 579.875 <= Q.sum_pt_top5 < 902.4062:
        z += -0.0006662514 * Q.sum_pt_top5 - 0.6260744
    if Q.sum_pt_top5 >= 902.4062:
        z += -0.003306147 * Q.sum_pt_top5 + 1.756184
    if Q.girth2_top15 < 0.0004816552:
        z += 1980.597 * Q.girth2_top15 - 1.382145
    if 0.0004816552 <= Q.girth2_top15 < 0.005383629:
        z += -21.73272 * Q.girth2_top15 - 0.4177128
    if 0.005383629 <= Q.girth2_top15 < 0.01563836:
        z += 52.14315 * Q.girth2_top15 - 0.8154331
    if 911.9328 <= Q.sum_pt_top30 < 1052.08:
        z += 0.007731927 * Q.sum_pt_top30 - 7.050998
    if Q.sum_pt_top30 >= 1052.08:
        z += 0.00392001 * Q.sum_pt_top30 - 3.040557
    if Q.z_top30_slots >= 0.9203881:
        z += -12.45683 * Q.z_top30_slots + 11.46512
    if Q.M3 < 0.03187688:
        z += -26.52981 * Q.M3 + 0.8456876
    if Q.tau4 < 0.01626937:
        z += -145.9046 * Q.tau4 + 2.373775
    if Q.sum_pt_top15 >= 967.7705:
        z += -0.004903328 * Q.sum_pt_top15 + 4.745296
    if Q.sum_pt_top10 >= 822.975:
        z += 0.001935832 * Q.sum_pt_top10 - 1.593141
    if Q.n_particles < 64.0 and Q.D2 < 2.410481:
        z += -0.01778976 * (64.0 - Q.n_particles) * (2.410481 - Q.D2)
    if Q.n_pt_above_1 > 26.0 and Q.n_dr_0_0p05 < 30.0:
        z += -0.0004580559 * (Q.n_pt_above_1 - 26.0) * (30.0 - Q.n_dr_0_0p05)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.6284117
    if Q.lam1 < 0.003811746:
        z += -1360.743 * Q.lam1 + 6.368019
    if 0.003811746 <= Q.lam1 < 0.006189818:
        z += -306.0531 * Q.lam1 + 2.347807
    if 0.006189818 <= Q.lam1 < 0.007671243:
        z += -352.1878 * Q.lam1 + 2.633373
    if Q.lam1 >= 0.007671243:
        z += -46.13469 * Q.lam1 + 0.2855653
    if Q.girth2 < 0.01397874:
        z += 149.972 * Q.girth2 - 2.09642
    if Q.e3 < 7.876005e-05:
        z += 6126.013 * Q.e3 + 0.3782134
    if 7.876005e-05 <= Q.e3 < 0.0003372339:
        z += -3329.925 * Q.e3 + 1.122963
    if Q.psi_0p3 >= 0.9853273:
        z += 89.03646 * Q.psi_0p3 - 87.73006
    if Q.e2_sq < 0.007511864:
        z += -240.1771 * Q.e2_sq + 3.729401
    if 0.007511864 <= Q.e2_sq < 0.02580859:
        z += -105.2223 * Q.e2_sq + 2.715638
    if Q.lam2 < 0.0009731947:
        z += -405.1895 * Q.lam2 + 0.7197415
    if 0.0009731947 <= Q.lam2 < 0.001776308:
        z += -718.5637 * Q.lam2 + 1.024716
    if Q.lam2 >= 0.001776308:
        z += -313.3743 * Q.lam2 + 0.3049742
    if Q.width < 0.009614971:
        z += 773.0939 * Q.width - 7.433275
    if Q.girth2_top20 < 0.00287991:
        z += -211.1545 * Q.girth2_top20 - 0.2116558
    if 0.00287991 <= Q.girth2_top20 < 0.02265114:
        z += 41.46236 * Q.girth2_top20 - 0.9391696
    if Q.e2 >= 0.01541561:
        z += -90.36873 * Q.e2 + 1.393089
    if Q.LHA >= 0.2091025:
        z += 7.238356 * Q.LHA - 1.513558
    if Q.girth2_top30 >= 0.008376291:
        z += 73.48302 * Q.girth2_top30 - 0.6155152
    if Q.log_sum_pt >= 6.811175:
        z += 2.521864 * Q.log_sum_pt - 17.17686
    if Q.girth2 < 0.01397874 and Q.psi_0p3 > 0.9853273:
        z += -9339.318 * (0.01397874 - Q.girth2) * (Q.psi_0p3 - 0.9853273)
    if Q.lam1 < 0.003811746 and Q.sum_pt_top50 < 1008.935:
        z += -9.290886 * (0.003811746 - Q.lam1) * (1008.935 - Q.sum_pt_top50)
    if Q.e3 < 0.0003372339 and Q.N2 < 0.3384815:
        z += -6948.239 * (0.0003372339 - Q.e3) * (0.3384815 - Q.N2)
    if Q.e2 > 0.04755309 and Q.z_dr_0_0p05 < 0.3289237:
        z += 355.6651 * (Q.e2 - 0.04755309) * (0.3289237 - Q.z_dr_0_0p05)
    if Q.e2_sq < 0.007511864 and Q.sum_pt < 1007.788:
        z += 8.269983 * (0.007511864 - Q.e2_sq) * (1007.788 - Q.sum_pt)
    if Q.e2_sq < 0.02580859 and Q.log_sum_pt < 6.98945:
        z += -381.5609 * (0.02580859 - Q.e2_sq) * (6.98945 - Q.log_sum_pt)
    if Q.lam2 > 0.003687605 and Q.D2_b2 < 2.459011:
        z += 269.4821 * (Q.lam2 - 0.003687605) * (2.459011 - Q.D2_b2)
    if Q.sum_pt_top20 > 695.3094 and Q.D2_b2 < 1.36316:
        z += 0.00159406 * (Q.sum_pt_top20 - 695.3094) * (1.36316 - Q.D2_b2)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.3445991
    if Q.girth2 < 0.007877041:
        z += 788.0132 * Q.girth2 - 6.207213
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.07074797 * Q.n_dr_0p2_0p4 + 1.06122
    if Q.lam1 < 0.005913555:
        z += 161.2815 * Q.lam1 - 1.100744
    if 0.005913555 <= Q.lam1 < 0.007671243:
        z += -86.04657 * Q.lam1 + 0.3618438
    if 0.007671243 <= Q.lam1 < 0.008241985:
        z += 522.548 * Q.lam1 - 4.306833
    if 0.9966167 <= Q.psi_0p3 < 0.9980008:
        z += 283.5286 * Q.psi_0p3 - 282.5693
    if Q.psi_0p3 >= 0.9980008:
        z += -48.70547 * Q.psi_0p3 + 49.00053
    if Q.tau1 < 0.07708632:
        z += -17.8425 * Q.tau1 + 1.375412
    if Q.e2_sq < 0.006936725:
        z += -385.3435 * Q.e2_sq + 6.398906
    if 0.006936725 <= Q.e2_sq < 0.007511864:
        z += -1077.14 * Q.e2_sq + 11.19771
    if 0.007511864 <= Q.e2_sq < 0.009606007:
        z += -1483.365 * Q.e2_sq + 14.24921
    if Q.girth < 0.05660088:
        z += 46.81264 * Q.girth - 4.772291
    if 0.05660088 <= Q.girth < 0.08589404:
        z += 72.46243 * Q.girth - 6.224091
    if Q.width < 0.01397874:
        z += -293.9283 * Q.width + 4.108748
    if Q.girth2_top50 < 0.00634935:
        z += 1080.047 * Q.girth2_top50 - 7.819111
    if 0.00634935 <= Q.girth2_top50 < 0.007820315:
        z += 405.5353 * Q.girth2_top50 - 3.536402
    if 0.007820315 <= Q.girth2_top50 < 0.008124776:
        z += 1198.8 * Q.girth2_top50 - 9.73998
    if Q.tau2 < 0.04828819:
        z += 38.12706 * Q.tau2 - 1.841087
    if Q.LHA < 0.2601462:
        z += -2.606121 * Q.LHA + 1.564191
    if 0.2601462 <= Q.LHA < 0.302389:
        z += -20.97914 * Q.LHA + 6.343861
    if Q.e3 < 0.0001086251:
        z += 4717.936 * Q.e3 - 0.5124862
    if Q.e2 < 0.03480688:
        z += -70.57723 * Q.e2 + 2.456573
    if 0.2404747 <= Q.max_dr < 0.2982:
        z += -4.322962 * Q.max_dr + 1.039563
    if Q.max_dr >= 0.2982:
        z += -1.007359 * Q.max_dr + 0.05084999
    if Q.tau21_b2 < 0.2352054 and Q.e2_sq < 0.007872294:
        z += -4754.986 * (0.2352054 - Q.tau21_b2) * (0.007872294 - Q.e2_sq)
    if Q.tau21_b2 < 0.2352054 and Q.e2_sq < 0.01396296:
        z += 1439.051 * (0.2352054 - Q.tau21_b2) * (0.01396296 - Q.e2_sq)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt_top30 > 978.0762:
        z += 0.03965203 * (0.2352054 - Q.tau21_b2) * (Q.sum_pt_top30 - 978.0762)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.sum_pt_top40 < 1225.842:
        z += -0.000139947 * (15.0 - Q.n_dr_0p2_0p4) * (1225.842 - Q.sum_pt_top40)
    if Q.e2_sq < 0.009606007 and Q.sum_pt < 1167.447:
        z += -7.673933 * (0.009606007 - Q.e2_sq) * (1167.447 - Q.sum_pt)
    if Q.lam1 < 0.005913555 and Q.sum_pt < 1167.447:
        z += 1.257515 * (0.005913555 - Q.lam1) * (1167.447 - Q.sum_pt)
    if Q.lam2 < 0.002396991 and Q.log_sum_pt < 7.139296:
        z += 1711.954 * (0.002396991 - Q.lam2) * (7.139296 - Q.log_sum_pt)
    if Q.lam2 < 0.002396991 and Q.zdr_0 > 0.002524869:
        z += -11187.71 * (0.002396991 - Q.lam2) * (Q.zdr_0 - 0.002524869)
    if Q.girth < 0.08589404 and Q.sum_pt_top50 < 1156.659:
        z += 0.1038043 * (0.08589404 - Q.girth) * (1156.659 - Q.sum_pt_top50)
    if Q.sum_pt_top15 > 1082.548 and Q.mean_phi > 2.298159e-05:
        z += -64.72404 * (Q.sum_pt_top15 - 1082.548) * (Q.mean_phi - 2.298159e-05)
    if Q.psi_0p3 > 0.9966167 and Q.soft1_pt < 2.275391:
        z += 72.02871 * (Q.psi_0p3 - 0.9966167) * (2.275391 - Q.soft1_pt)
    return max(0.0, z)


def neuron_8(Q):
    z = 1.531522
    if Q.girth2 < 0.003638856:
        z += -832.5582 * Q.girth2 + 6.214276
    if 0.003638856 <= Q.girth2 < 0.01397874:
        z += 16.93073 * Q.girth2 + 3.123109
    if 0.01397874 <= Q.girth2 < 0.0258669:
        z += -282.6156 * Q.girth2 + 7.31039
    if Q.sum_pt_top40 < 1007.44:
        z += 0.004905803 * Q.sum_pt_top40 - 4.942302
    if Q.n_dr_0p2_0p4 < 18.0:
        z += 0.0436985 * Q.n_dr_0p2_0p4 - 0.786573
    if Q.e2_sq < 0.002574843:
        z += 594.2432 * Q.e2_sq - 7.431562
    if 0.002574843 <= Q.e2_sq < 0.009606007:
        z += 839.3317 * Q.e2_sq - 8.062626
    if Q.girth2_top30 < 0.007463985:
        z += 478.8906 * Q.girth2_top30 - 3.574432
    if Q.sum_pt < 1028.184:
        z += -0.0199413 * Q.sum_pt + 20.90569
    if 1028.184 <= Q.sum_pt < 1085.125:
        z += -0.00706645 * Q.sum_pt + 7.667981
    if Q.girth2_top50 < 0.01351431:
        z += -181.1827 * Q.girth2_top50 + 2.44856
    if Q.sum_pt_top50 < 1156.659:
        z += -0.01042218 * Q.sum_pt_top50 + 12.53977
    if 1156.659 <= Q.sum_pt_top50 < 1245.697:
        z += -0.005445539 * Q.sum_pt_top50 + 6.78349
    if Q.width < 0.008190222:
        z += 392.7333 * Q.width - 3.216573
    if Q.psi_0p3 >= 0.9943058:
        z += -90.43474 * Q.psi_0p3 + 89.91978
    if Q.tau1 < 0.1219132:
        z += -31.76015 * Q.tau1 + 3.871982
    if Q.D2 < 1.788105:
        z += 0.2297658 * Q.D2 - 0.4108454
    if 0.005788041 <= Q.girth2_top15 < 0.007887677:
        z += -168.9269 * Q.girth2_top15 + 0.9777556
    if Q.girth2_top15 >= 0.007887677:
        z += 20.30862 * Q.girth2_top15 - 0.5148729
    if Q.girth2_top40 < 0.008840538:
        z += -208.4547 * Q.girth2_top40 + 1.842852
    if Q.sum_pt_top20 >= 1017.778:
        z += 0.002742795 * Q.sum_pt_top20 - 2.791557
    if Q.girth2 < 0.0258669 and Q.log_sum_pt < 7.139296:
        z += -1561.935 * (0.0258669 - Q.girth2) * (7.139296 - Q.log_sum_pt)
    if Q.girth2_top30 < 0.007463985 and Q.sum_pt < 1260.541:
        z += 3.821053 * (0.007463985 - Q.girth2_top30) * (1260.541 - Q.sum_pt)
    if Q.girth2_top30 < 0.007463985 and Q.sum_pt_top40 < 1095.686:
        z += -2.246205 * (0.007463985 - Q.girth2_top30) * (1095.686 - Q.sum_pt_top40)
    if Q.girth2 < 0.01397874 and Q.sum_pt < 986.0565:
        z += 1.944848 * (0.01397874 - Q.girth2) * (986.0565 - Q.sum_pt)
    if Q.girth2 < 0.0258669 and Q.z_top50_slots > 0.9704436:
        z += -1003.966 * (0.0258669 - Q.girth2) * (Q.z_top50_slots - 0.9704436)
    if Q.tau1 < 0.1219132 and Q.planar_flow < 0.5364935:
        z += -21.56024 * (0.1219132 - Q.tau1) * (0.5364935 - Q.planar_flow)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.sj2_zsoft > 0.1181474:
        z += 0.08459573 * (18.0 - Q.n_dr_0p2_0p4) * (Q.sj2_zsoft - 0.1181474)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.5635076
    if Q.girth2_top40 < 0.005712208:
        z += -378.3418 * Q.girth2_top40 + 2.161167
    if Q.girth2 >= 0.0258669:
        z += -334.5331 * Q.girth2 + 8.653334
    if Q.lam1 < 0.004673423:
        z += -106.8902 * Q.lam1 + 0.7759465
    if 0.004673423 <= Q.lam1 < 0.006716737:
        z += 91.70226 * Q.lam1 - 0.15216
    if 0.006716737 <= Q.lam1 < 0.007259287:
        z += -70.10561 * Q.lam1 + 0.9346609
    if Q.lam1 >= 0.007259287:
        z += 36.78456 * Q.lam1 + 0.1587144
    if 0.01989454 <= Q.e2_sq < 0.0292152:
        z += 162.5899 * Q.e2_sq - 3.234652
    if Q.e2_sq >= 0.0292152:
        z += 454.4074 * Q.e2_sq - 11.76016
    if Q.sum_pt_top30 < 1191.938:
        z += -0.004912826 * Q.sum_pt_top30 + 5.855783
    if Q.width < 0.006941794:
        z += -598.229 * Q.width + 4.152782
    if Q.log_sum_pt < 6.903423:
        z += 7.060063 * Q.log_sum_pt - 48.7386
    if Q.sum_pt < 986.0565:
        z += -0.03522966 * Q.sum_pt + 34.73843
    if Q.girth >= 0.1207452:
        z += -59.96156 * Q.girth + 7.240069
    if Q.girth2_top15 < 0.001319197:
        z += 457.5147 * Q.girth2_top15 - 1.101749
    if 0.001319197 <= Q.girth2_top15 < 0.02146578:
        z += 19.5875 * Q.girth2_top15 - 0.5240369
    if 0.02146578 <= Q.girth2_top15 < 0.02675364:
        z += -283.5481 * Q.girth2_top15 + 5.983004
    if Q.girth2_top15 >= 0.02675364:
        z += -303.1356 * Q.girth2_top15 + 6.507041
    if Q.e3 >= 0.0005178279:
        z += -9361.406 * Q.e3 + 4.847597
    if Q.girth2_top20 < 0.01083435:
        z += 92.3514 * Q.girth2_top20 - 1.000568
    if Q.pt_entropy < 2.485823:
        z += -0.5538948 * Q.pt_entropy + 1.376884
    if Q.tau1 < 0.06310829:
        z += 25.22317 * Q.tau1 - 1.591791
    if Q.girth2_top40 < 0.005712208 and Q.log_sum_pt > 7.017258:
        z += 4486.58 * (0.005712208 - Q.girth2_top40) * (Q.log_sum_pt - 7.017258)
    if Q.lam1 < 0.007259287 and Q.sum_pt > 1085.125:
        z += -2.622984 * (0.007259287 - Q.lam1) * (Q.sum_pt - 1085.125)
    if Q.sum_pt_top40 < 972.4111 and Q.tau3 < 0.0369869:
        z += -1.086892 * (972.4111 - Q.sum_pt_top40) * (0.0369869 - Q.tau3)
    if Q.e3 > 0.0005178279 and Q.soft3_pt > 2.144727:
        z += -139052.2 * (Q.e3 - 0.0005178279) * (Q.soft3_pt - 2.144727)
    if Q.girth > 0.1207452 and Q.sj3_pairmin_over_m < 0.4606099:
        z += 218.5768 * (Q.girth - 0.1207452) * (0.4606099 - Q.sj3_pairmin_over_m)
    return max(0.0, z)


def neuron_10(Q):
    z = 3.044487
    if Q.girth < 0.03577037:
        z += 152.4877 * Q.girth - 11.13898
    if 0.03577037 <= Q.girth < 0.1207452:
        z += 66.89553 * Q.girth - 8.077313
    if 0.01397874 <= Q.girth2 < 0.02928196:
        z += -171.836 * Q.girth2 + 2.402051
    if Q.girth2 >= 0.02928196:
        z += -329.1517 * Q.girth2 + 7.008564
    if Q.girth2_top30 < 0.005402331:
        z += -157.101 * Q.girth2_top30 + 0.8487116
    if Q.e2 >= 0.006720044:
        z += 46.20219 * Q.e2 - 0.3104807
    if Q.psi_0p3 >= 0.9985421:
        z += -376.3823 * Q.psi_0p3 + 375.8336
    if Q.z_dr_0_0p05 >= 0.7128619:
        z += -4.392839 * Q.z_dr_0_0p05 + 3.131488
    if Q.M2 < 0.08222447:
        z += -21.14979 * Q.M2 + 1.73903
    if Q.width < 0.006403325:
        z += 206.4373 * Q.width - 1.321885
    if Q.n_dr_0_0p05 >= 6.0:
        z += 0.02407947 * Q.n_dr_0_0p05 - 0.1444768
    if Q.sum_pt_top3 < 656.3844:
        z += -0.002681777 * Q.sum_pt_top3 + 1.760276
    if Q.tau21_b2 < 0.3062621:
        z += 3.789469 * Q.tau21_b2 - 1.160571
    if Q.dr_0 < 0.06413297:
        z += -7.716965 * Q.dr_0 + 0.4949119
    if Q.e3 < 0.0005178279:
        z += 3485.128 * Q.e3 - 1.804696
    if Q.tau4 < 0.0259543:
        z += -77.56644 * Q.tau4 + 2.013182
    if Q.pt_entropy < 3.539367:
        z += 0.5532575 * Q.pt_entropy - 1.958181
    if Q.n_dr_0p2_0p4 < 15.0:
        z += 0.05361186 * Q.n_dr_0p2_0p4 - 0.8041778
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -11.29619 * Q.z_dr_0p2_0p4 + 1.030502
    if Q.girth2_top10 < 0.007678544:
        z += 107.4891 * Q.girth2_top10 - 1.154817
    if 0.007678544 <= Q.girth2_top10 < 0.01976735:
        z += 27.25306 * Q.girth2_top10 - 0.5387209
    if Q.LHA < 0.3719813:
        z += -12.35237 * Q.LHA + 4.594849
    if 0.005718442 <= Q.girth2_top20 < 0.008031209:
        z += -253.592 * Q.girth2_top20 + 1.450151
    if Q.girth2_top20 >= 0.008031209:
        z += -51.94777 * Q.girth2_top20 - 0.1692959
    if Q.e2_sq < 0.00592208:
        z += -610.5974 * Q.e2_sq + 3.678693
    if 0.00592208 <= Q.e2_sq < 0.00818374:
        z += -112.4908 * Q.e2_sq + 0.7288662
    if 0.00818374 <= Q.e2_sq < 0.009606007:
        z += 134.8054 * Q.e2_sq - 1.294942
    if Q.tau1 < 0.1507173:
        z += -10.98066 * Q.tau1 + 1.654975
    if Q.e2 > 0.006720044 and Q.log_sum_pt > 6.893714:
        z += -170.1872 * (Q.e2 - 0.006720044) * (Q.log_sum_pt - 6.893714)
    if Q.girth2 > 0.02928196 and Q.sj3_dr13 < 0.03287546:
        z += -424633.3 * (Q.girth2 - 0.02928196) * (0.03287546 - Q.sj3_dr13)
    if Q.girth2 > 0.01397874 and Q.pt_3 < 114.75:
        z += 0.9830154 * (Q.girth2 - 0.01397874) * (114.75 - Q.pt_3)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.7168729
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.1171087 * Q.n_dr_0p2_0p4 + 1.171087
    if Q.psi_0p2 >= 0.9935324:
        z += 102.3379 * Q.psi_0p2 - 101.6761
    if Q.girth2_top10 < 0.0007431905:
        z += 373.9181 * Q.girth2_top10 - 0.08942423
    if 0.0007431905 <= Q.girth2_top10 < 0.00130722:
        z += -334.1457 * Q.girth2_top10 + 0.4368021
    if Q.lam1 < 0.005913555:
        z += -300.5671 * Q.lam1 + 1.77742
    if Q.e2_sq < 0.004754578:
        z += 177.2351 * Q.e2_sq + 2.685575
    if 0.004754578 <= Q.e2_sq < 0.00818374:
        z += -1028.897 * Q.e2_sq + 8.420222
    if Q.girth2_top50 < 0.00634935:
        z += 701.363 * Q.girth2_top50 - 4.453199
    if Q.e2 < 0.02210818:
        z += 57.29647 * Q.e2 - 1.342525
    if 0.02210818 <= Q.e2 < 0.02515919:
        z += -56.61303 * Q.e2 + 1.175807
    if 0.02515919 <= Q.e2 < 0.03875945:
        z += 18.27401 * Q.e2 - 0.7082907
    if Q.z_top30_slots >= 0.9734886:
        z += -21.30107 * Q.z_top30_slots + 20.73635
    if Q.D2 < 4.450169:
        z += 0.4579081 * Q.D2 - 2.037768
    if Q.C2_b2 >= 0.002289486:
        z += -34.55635 * Q.C2_b2 + 0.07911628
    if Q.girth2_top30 < 0.005809485:
        z += 117.3122 * Q.girth2_top30 + 0.3258888
    if 0.005809485 <= Q.girth2_top30 < 0.006363916:
        z += 232.705 * Q.girth2_top30 - 0.344484
    if 0.006363916 <= Q.girth2_top30 < 0.01215787:
        z += -196.1407 * Q.girth2_top30 + 2.384654
    if Q.n_real_top50 >= 22.0:
        z += -0.02539742 * Q.n_real_top50 + 0.5587433
    if Q.sum_pt_top15 >= 840.7969:
        z += -0.001841497 * Q.sum_pt_top15 + 1.548325
    if Q.girth < 0.07374472:
        z += 23.50392 * Q.girth - 1.73329
    if Q.n_dr_0p2_0p4 < 10.0 and Q.sum_pt_top50 < 1156.659:
        z += -0.0005259442 * (10.0 - Q.n_dr_0p2_0p4) * (1156.659 - Q.sum_pt_top50)
    if Q.lam1 < 0.005913555 and Q.D2_b2 < 3.014827:
        z += -178.0093 * (0.005913555 - Q.lam1) * (3.014827 - Q.D2_b2)
    if Q.lam1 < 0.005913555 and Q.sum_pt < 1115.723:
        z += -4.753717 * (0.005913555 - Q.lam1) * (1115.723 - Q.sum_pt)
    if Q.e2_sq < 0.009606007 and Q.sum_pt < 1115.723:
        z += 3.433832 * (0.009606007 - Q.e2_sq) * (1115.723 - Q.sum_pt)
    if Q.z_top30_slots > 0.9734886 and Q.D2_b2 < 1.08774:
        z += 40.47427 * (Q.z_top30_slots - 0.9734886) * (1.08774 - Q.D2_b2)
    if Q.e2_sq < 0.00818374 and Q.sum_pt < 1115.723:
        z += -7.299248 * (0.00818374 - Q.e2_sq) * (1115.723 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.soft2_z < 0.0008024071:
        z += 156.0711 * (10.0 - Q.n_dr_0p2_0p4) * (0.0008024071 - Q.soft2_z)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.3531858
    if Q.girth2_top30 < 0.005809485:
        z += 325.3068 * Q.girth2_top30 - 1.889865
    if Q.psi_0p1 < 0.08133662:
        z += 17.16582 * Q.psi_0p1 - 1.39621
    if Q.e2_sq < 0.007511864:
        z += -678.5312 * Q.e2_sq + 5.097035
    if Q.sum_pt >= 1085.125:
        z += -0.01184812 * Q.sum_pt + 12.85669
    if Q.psi_0p3 >= 0.9943058:
        z += -135.8406 * Q.psi_0p3 + 135.0671
    if Q.LHA >= 0.3719813:
        z += 16.38764 * Q.LHA - 6.095894
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -6.757235 * Q.z_dr_0_0p05 + 5.938974
    if Q.lam1 < 0.002752094:
        z += 680.1094 * Q.lam1 - 1.871725
    if Q.girth2 < 0.006403325 and Q.log_sum_pt > 6.811175:
        z += 6122.764 * (0.006403325 - Q.girth2) * (Q.log_sum_pt - 6.811175)
    if Q.e2_sq < 0.007511864 and Q.log_sum_pt > 6.811175:
        z += -3856.398 * (0.007511864 - Q.e2_sq) * (Q.log_sum_pt - 6.811175)
    if Q.psi_0p1 < 0.08133662 and Q.sum_pt_top50 > 889.8503:
        z += 0.1601908 * (0.08133662 - Q.psi_0p1) * (Q.sum_pt_top50 - 889.8503)
    if Q.sum_pt > 1085.125 and Q.lam1 < 0.001868041:
        z += 4.841577 * (Q.sum_pt - 1085.125) * (0.001868041 - Q.lam1)
    if Q.psi_0p3 > 0.9943058 and Q.girth2 < 0.00751625:
        z += 62326.03 * (Q.psi_0p3 - 0.9943058) * (0.00751625 - Q.girth2)
    if Q.psi_0p1 > 0.9371031 and Q.absphi_0 < 0.05444336:
        z += -312.0497 * (Q.psi_0p1 - 0.9371031) * (0.05444336 - Q.absphi_0)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.7286193
    if Q.sum_pt < 1017.435:
        z += 0.01624397 * Q.sum_pt - 17.62674
    if 1017.435 <= Q.sum_pt < 1085.125:
        z += -0.008093454 * Q.sum_pt + 7.135002
    if 1085.125 <= Q.sum_pt < 1167.447:
        z += -0.02433742 * Q.sum_pt + 24.76174
    if Q.sum_pt >= 1167.447:
        z += -0.02680424 * Q.sum_pt + 27.64162
    if 0.008840538 <= Q.girth2_top40 < 0.01292642:
        z += -179.6383 * Q.girth2_top40 + 1.588099
    if 0.01292642 <= Q.girth2_top40 < 0.02862386:
        z += -326.9324 * Q.girth2_top40 + 3.492086
    if Q.girth2_top40 >= 0.02862386:
        z += 1460.0 * Q.girth2_top40 - 47.65682
    if Q.n_pt_above_5 < 25.0:
        z += 0.0903003 * Q.n_pt_above_5 - 2.257507
    if Q.sum_pt_top40 >= 972.4111:
        z += -0.005398742 * Q.sum_pt_top40 + 5.249797
    if Q.e2 < 0.03480688:
        z += -16.54486 * Q.e2 + 0.5758747
    if Q.e2 >= 0.03680582:
        z += 47.16279 * Q.e2 - 1.735865
    if Q.sum_pt_top50 < 889.8503:
        z += -0.007330143 * Q.sum_pt_top50 + 7.909182
    if 889.8503 <= Q.sum_pt_top50 < 976.277:
        z += -0.01340586 * Q.sum_pt_top50 + 13.31566
    if 976.277 <= Q.sum_pt_top50 < 1013.916:
        z += -0.004315955 * Q.sum_pt_top50 + 4.441395
    if 1013.916 <= Q.sum_pt_top50 < 1078.994:
        z += 0.004751072 * Q.sum_pt_top50 - 4.751806
    if Q.sum_pt_top50 >= 1078.994:
        z += 0.01208122 * Q.sum_pt_top50 - 12.66099
    if 6.811175 <= Q.log_sum_pt < 6.893714:
        z += 22.88243 * Q.log_sum_pt - 155.8562
    if Q.log_sum_pt >= 6.893714:
        z += 16.4366 * Q.log_sum_pt - 111.4205
    if Q.width >= 0.01991191:
        z += -250.6764 * Q.width + 4.991446
    if 0.005852839 <= Q.girth2_top50 < 0.01351431:
        z += 125.6034 * Q.girth2_top50 - 0.7351366
    if Q.girth2_top50 >= 0.01351431:
        z += -150.2854 * Q.girth2_top50 + 2.993311
    if Q.sum_pt_top20 >= 1064.139:
        z += 0.00503252 * Q.sum_pt_top20 - 5.355301
    if Q.girth2_top40 > 0.01292642 and Q.sum_pt_top50 < 1245.697:
        z += 1.335522 * (Q.girth2_top40 - 0.01292642) * (1245.697 - Q.sum_pt_top50)
    if Q.girth < 0.1207452 and Q.sum_pt_top40 < 858.8262:
        z += -0.2741142 * (0.1207452 - Q.girth) * (858.8262 - Q.sum_pt_top40)
    if Q.sum_pt < 1085.125 and Q.tau21 > 0.1295048:
        z += 0.01022395 * (1085.125 - Q.sum_pt) * (Q.tau21 - 0.1295048)
    if Q.girth2_top40 > 0.02862386 and Q.pt_6 > 19.46875:
        z += -67.85086 * (Q.girth2_top40 - 0.02862386) * (Q.pt_6 - 19.46875)
    if Q.girth2_top40 > 0.02862386 and Q.z_6 > 0.01906139:
        z += -48395.43 * (Q.girth2_top40 - 0.02862386) * (Q.z_6 - 0.01906139)
    if Q.girth2_top40 > 0.02862386 and Q.pt_6 < 62.25:
        z += -34.1767 * (Q.girth2_top40 - 0.02862386) * (62.25 - Q.pt_6)
    if Q.sum_pt_top20 > 1064.139 and Q.M2 > 0.05568888:
        z += -0.2303546 * (Q.sum_pt_top20 - 1064.139) * (Q.M2 - 0.05568888)
    if Q.sum_pt_top10 > 867.9266 and Q.C2_b2 < 0.006580753:
        z += 3.954924 * (Q.sum_pt_top10 - 867.9266) * (0.006580753 - Q.C2_b2)
    if Q.girth < 0.1207452 and Q.C2_b2 < 0.01330402:
        z += -597.3895 * (0.1207452 - Q.girth) * (0.01330402 - Q.C2_b2)
    if Q.e2_sq > 0.0292152 and Q.pt_4 > 90.625:
        z += -138.8738 * (Q.e2_sq - 0.0292152) * (Q.pt_4 - 90.625)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.6048099
    if Q.tau21_b2 < 0.342495:
        z += -6.001898 * Q.tau21_b2 + 2.05562
    if Q.n_dr_0p1_0p2 < 17.0:
        z += 0.04647837 * Q.n_dr_0p1_0p2 - 0.7901323
    if 0.9896594 <= Q.psi_0p3 < 0.9956185:
        z += 109.0318 * Q.psi_0p3 - 107.9044
    if Q.psi_0p3 >= 0.9956185:
        z += 655.4909 * Q.psi_0p3 - 651.9691
    if Q.LHA < 0.2091025:
        z += 10.67936 * Q.LHA - 2.233081
    if Q.LHA >= 0.3098384:
        z += -13.19103 * Q.LHA + 4.087087
    if Q.width < 0.00751625:
        z += 112.3824 * Q.width - 0.8446944
    if Q.e2_sq < 0.00616708:
        z += 734.2193 * Q.e2_sq - 1.442536
    if 0.00616708 <= Q.e2_sq < 0.006936725:
        z += 173.821 * Q.e2_sq + 2.013485
    if 0.006936725 <= Q.e2_sq < 0.009606007:
        z += -551.4065 * Q.e2_sq + 7.044188
    if 0.009606007 <= Q.e2_sq < 0.01396296:
        z += -230.7841 * Q.e2_sq + 3.964287
    if 0.01396296 <= Q.e2_sq < 0.01989454:
        z += -125.0692 * Q.e2_sq + 2.488193
    if Q.girth2 < 0.008190222:
        z += 625.806 * Q.girth2 - 5.12549
    if Q.girth2_top30 < 0.006088416:
        z += 377.972 * Q.girth2_top30 - 3.628668
    if 0.006088416 <= Q.girth2_top30 < 0.008376291:
        z += 284.2624 * Q.girth2_top30 - 3.058125
    if 0.008376291 <= Q.girth2_top30 < 0.01215787:
        z += 179.0415 * Q.girth2_top30 - 2.176764
    if Q.girth2_top3 < 0.001155057:
        z += -445.5543 * Q.girth2_top3 + 0.5146404
    if Q.e2 < 0.03480688:
        z += -73.36827 * Q.e2 + 2.55372
    if Q.tau1 < 0.05444509:
        z += 161.8664 * Q.tau1 - 9.682434
    if 0.05444509 <= Q.tau1 < 0.1008497:
        z += 18.73954 * Q.tau1 - 1.889877
    if Q.M3 < 0.03787151:
        z += -20.83572 * Q.M3 + 0.78908
    if Q.tau21 < 0.5936969:
        z += 1.382929 * Q.tau21 - 0.8210408
    if Q.girth2_top50 < 0.01951641:
        z += 117.9509 * Q.girth2_top50 - 2.301979
    if Q.girth2_top20 < 0.006374178:
        z += -29.62683 * Q.girth2_top20 - 0.5699059
    if 0.006374178 <= Q.girth2_top20 < 0.01083435:
        z += 170.1172 * Q.girth2_top20 - 1.84311
    if Q.log_sum_pt >= 7.062574:
        z += -7.224354 * Q.log_sum_pt + 51.02254
    if Q.sum_pt_top30 >= 1052.08:
        z += 0.003951237 * Q.sum_pt_top30 - 4.157017
    if Q.tau21_b2 < 0.342495 and Q.girth2_top40 < 0.007709916:
        z += -2674.167 * (0.342495 - Q.tau21_b2) * (0.007709916 - Q.girth2_top40)
    if Q.tau21_b2 < 0.342495 and Q.girth2_top40 < 0.006259772:
        z += 2404.439 * (0.342495 - Q.tau21_b2) * (0.006259772 - Q.girth2_top40)
    if Q.psi_0p3 > 0.9956185 and Q.sum_pt_top40 < 1225.842:
        z += -2.064797 * (Q.psi_0p3 - 0.9956185) * (1225.842 - Q.sum_pt_top40)
    if Q.girth2_top30 < 0.01215787 and Q.sum_pt > 907.9372:
        z += 1.428887 * (0.01215787 - Q.girth2_top30) * (Q.sum_pt - 907.9372)
    if Q.e2 < 0.03480688 and Q.sum_pt_top30 > 886.3438:
        z += -0.3004728 * (0.03480688 - Q.e2) * (Q.sum_pt_top30 - 886.3438)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.1577932
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -6.413539 * Q.z_dr_0p1_0p2 + 0.7718289
    if Q.psi_0p1 >= 0.8976117:
        z += -3.939851 * Q.psi_0p1 + 3.536456
    if Q.girth2_top5 < 0.002270363:
        z += 3.245804 * Q.girth2_top5 + 1.12311
    if 0.002270363 <= Q.girth2_top5 < 0.008329695:
        z += -186.5683 * Q.girth2_top5 + 1.554057
    if Q.log_sum_pt >= 7.139296:
        z += -20.11085 * Q.log_sum_pt + 143.5773
    if Q.sum_pt_top30 >= 988.4375:
        z += -0.01248664 * Q.sum_pt_top30 + 12.34226
    if Q.sum_pt_top50 >= 1156.659:
        z += 0.02030123 * Q.sum_pt_top50 - 23.48162
    if Q.tau1 < 0.06310829:
        z += 30.07871 * Q.tau1 - 2.251364
    if 0.06310829 <= Q.tau1 < 0.0705748:
        z += 47.29759 * Q.tau1 - 3.338018
    if Q.sum_pt_top20 >= 869.693:
        z += 0.003483148 * Q.sum_pt_top20 - 3.02927
    if Q.sj3_dr_max >= 0.3715619:
        z += -8.740375 * Q.sj3_dr_max + 3.24759
    if Q.sj2_dr >= 0.2780918:
        z += 12.92991 * Q.sj2_dr - 3.595702
    if Q.e2 < 0.04082832:
        z += -31.01027 * Q.e2 + 1.266097
    if Q.z_top30_slots < 0.9860575:
        z += 4.240128 * Q.z_top30_slots - 4.18101
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.sum_pt < 986.0565:
        z += 0.122368 * (0.1203437 - Q.z_dr_0p1_0p2) * (986.0565 - Q.sum_pt)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.sum_pt < 995.6769:
        z += -0.1500379 * (0.1203437 - Q.z_dr_0p1_0p2) * (995.6769 - Q.sum_pt)
    if Q.girth2_top5 < 0.008329695 and Q.psi_0p3 > 0.9299135:
        z += -866.4205 * (0.008329695 - Q.girth2_top5) * (Q.psi_0p3 - 0.9299135)
    if Q.girth2_top10 < 0.007678544 and Q.tau4 > 0.01873799:
        z += -21585.3 * (0.007678544 - Q.girth2_top10) * (Q.tau4 - 0.01873799)
    if Q.sd_rg < 0.1778185 and Q.soft5_dr > 0.04139378:
        z += -21.14177 * (0.1778185 - Q.sd_rg) * (Q.soft5_dr - 0.04139378)
    if Q.sd_rg < 0.1778185 and Q.soft5_dr0 > 0.03721986:
        z += 23.07728 * (0.1778185 - Q.sd_rg) * (Q.soft5_dr0 - 0.03721986)
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
