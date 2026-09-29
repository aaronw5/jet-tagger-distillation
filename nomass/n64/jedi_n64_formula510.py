"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; no mass observables), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.1% (the network: 81.1%); same class as the network for 93.5% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.C2                     energy correlation ratio e3/e2²
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
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_14                  pT of particle 14 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_pt               pT [GeV] of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.z_6                    pT of particle 6 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_z                pT share of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.sj3_z2                 pT share of subjet 2 of 3 (by pT)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft2_dr0              ΔR between the hardest and the 2. softest real particle (0 if among the 15 hardest)
  Q.soft5_dr0              ΔR between the hardest and the 5. softest real particle (0 if among the 15 hardest)
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.soft2_dr               ΔR from the jet axis of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_dr               ΔR from the jet axis of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.dr01                   ΔR between particles 0 and 1
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
        C2=e3 / max(e2 ** 2, 1e-12),
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
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_90pct=ncum(0.9),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_11=pt[11],
        pt_14=pt[14],
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        soft1_pt=softp(1, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft7_pt=softp(7, 'pt'),
        z_6=z[6],
        soft1_z=softp(1, 'z'),
        soft3_z=softp(3, 'z'),
        soft7_z=softp(7, 'z'),
        sj3_z2=subjets(3)["z"][1],
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        soft2_dr0=softp(2, 'dr0'),
        soft5_dr0=softp(5, 'dr0'),
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        soft2_dr=softp(2, 'dr'),
        soft5_dr=softp(5, 'dr'),
        dr01=math.sqrt(dist2(0, 1)),
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
        tau4=tau_n(4),
    )


def neuron_0(Q):
    z = -1.429295
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.04166162 * Q.n_dr_0p2_0p4 + 0.7499091
    if Q.girth2_top40 < 0.003113918:
        z += 227.2658 * Q.girth2_top40 - 0.09159513
    if 0.003113918 <= Q.girth2_top40 < 0.008840538:
        z += -107.5839 * Q.girth2_top40 + 0.9510992
    if Q.girth2 < 0.008190222:
        z += -210.1597 * Q.girth2 + 1.721254
    if Q.e2_sq < 0.00616708:
        z += -392.4122 * Q.e2_sq + 3.321708
    if 0.00616708 <= Q.e2_sq < 0.006936725:
        z += -962.0127 * Q.e2_sq + 6.834479
    if 0.006936725 <= Q.e2_sq < 0.007872294:
        z += -543.1398 * Q.e2_sq + 3.928873
    if 0.007872294 <= Q.e2_sq < 0.009606007:
        z += 200.0809 * Q.e2_sq - 1.921978
    if 1002.379 <= Q.sum_pt < 1115.723:
        z += -0.01248236 * Q.sum_pt + 12.51205
    if 1115.723 <= Q.sum_pt < 1167.447:
        z += -0.005725734 * Q.sum_pt + 4.973528
    if Q.sum_pt >= 1167.447:
        z += -0.008288951 * Q.sum_pt + 7.965947
    if 858.8262 <= Q.sum_pt_top40 < 1024.942:
        z += 0.005558898 * Q.sum_pt_top40 - 4.774127
    if Q.sum_pt_top40 >= 1024.942:
        z += 0.003895444 * Q.sum_pt_top40 - 3.069183
    if Q.psi_0p3 >= 0.9943058:
        z += 76.44701 * Q.psi_0p3 - 76.01171
    if Q.lam1 < 0.005913555:
        z += 120.3972 * Q.lam1 - 0.7119751
    if Q.tau1 < 0.0705748:
        z += 18.09551 * Q.tau1 - 1.277087
    if Q.psi_0p2 >= 0.9087063:
        z += -5.998082 * Q.psi_0p2 + 5.450495
    if Q.sum_pt_top30 < 1191.938:
        z += -0.002712918 * Q.sum_pt_top30 + 3.23363
    if Q.girth2_top30 < 0.00375223:
        z += 217.1979 * Q.girth2_top30 - 1.071959
    if 0.00375223 <= Q.girth2_top30 < 0.006363916:
        z += 98.39736 * Q.girth2_top30 - 0.6261925
    if Q.girth2_top50 < 0.008124776:
        z += -202.6602 * Q.girth2_top50 + 1.646569
    if Q.girth2 < 0.008190222 and Q.girth2_top15 < 0.006142802:
        z += -14920.25 * (0.008190222 - Q.girth2) * (0.006142802 - Q.girth2_top15)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.log_sum_pt > 6.915514:
        z += -0.1622179 * (18.0 - Q.n_dr_0p2_0p4) * (Q.log_sum_pt - 6.915514)
    if Q.girth2_top50 < 0.005852839 and Q.sum_pt < 1260.541:
        z += -1.634594 * (0.005852839 - Q.girth2_top50) * (1260.541 - Q.sum_pt)
    if Q.sum_pt_top30 < 1073.473 and Q.D2_b2 < 43.66338:
        z += -3.804559e-05 * (1073.473 - Q.sum_pt_top30) * (43.66338 - Q.D2_b2)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.877187
    if Q.n_particles >= 38.0:
        z += 0.09530739 * Q.n_particles - 3.621681
    if Q.log_sum_pt < 6.910131:
        z += 6.814936 * Q.log_sum_pt - 48.65385
    if 6.910131 <= Q.log_sum_pt < 6.98945:
        z += 39.87886 * Q.log_sum_pt - 277.1299
    if 6.98945 <= Q.log_sum_pt < 7.139296:
        z += 17.05713 * Q.log_sum_pt - 117.6185
    if Q.log_sum_pt >= 7.139296:
        z += 10.24219 * Q.log_sum_pt - 68.96469
    if Q.sum_pt_top50 < 934.2416:
        z += -0.01439534 * Q.sum_pt_top50 + 15.53249
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += -0.02459339 * Q.sum_pt_top50 + 25.05993
    if 959.0957 <= Q.sum_pt_top50 < 1078.994:
        z += -0.02982595 * Q.sum_pt_top50 + 30.07846
    if Q.sum_pt_top50 >= 1078.994:
        z += -0.01543061 * Q.sum_pt_top50 + 14.54597
    if Q.girth2_top15 < 0.002197765:
        z += -210.628 * Q.girth2_top15 + 0.4629108
    if Q.z_top5 >= 0.6551948:
        z += 2.543782 * Q.z_top5 - 1.666673
    if Q.psi_0p3 >= 0.9966167:
        z += -73.68755 * Q.psi_0p3 + 73.43824
    if Q.tau21 < 0.2140276:
        z += -7.155497 * Q.tau21 + 1.928602
    if 0.2140276 <= Q.tau21 < 0.5100475:
        z += -1.341561 * Q.tau21 + 0.6842599
    if Q.lam2 < 0.0007143144:
        z += 695.0899 * Q.lam2 - 0.4965127
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.07544541 * Q.n_dr_0p2_0p4 - 0.5281178
    if Q.pt_11 < 29.04688:
        z += 0.01957093 * Q.pt_11 - 0.5684745
    if Q.sum_pt < 972.0419:
        z += 0.003497597 * Q.sum_pt - 4.083258
    if 972.0419 <= Q.sum_pt < 1052.889:
        z += 0.02872634 * Q.sum_pt - 28.60666
    if 1052.889 <= Q.sum_pt < 1167.447:
        z += 0.01485513 * Q.sum_pt - 14.00181
    if Q.sum_pt >= 1167.447:
        z += 0.01135754 * Q.sum_pt - 9.918549
    if Q.girth2_top5 < 0.0006570502:
        z += -643.5392 * Q.girth2_top5 + 0.4228376
    if Q.sum_pt_top2 < 605.875:
        z += -0.001823493 * Q.sum_pt_top2 + 1.104809
    if 15.0 <= Q.n_dr_0_0p05 < 30.0:
        z += 0.03280369 * Q.n_dr_0_0p05 - 0.4920553
    if Q.n_dr_0_0p05 >= 30.0:
        z += -0.06461446 * Q.n_dr_0_0p05 + 2.430489
    if Q.n_dr_0p1_0p2 < 7.0:
        z += 0.09355393 * Q.n_dr_0p1_0p2 - 0.6548775
    if Q.tau1 < 0.02607472:
        z += -12.48093 * Q.tau1 + 2.438584
    if 0.02607472 <= Q.tau1 < 0.1219132:
        z += -30.79964 * Q.tau1 + 2.916239
    if 0.1219132 <= Q.tau1 < 0.1953848:
        z += -17.23765 * Q.tau1 + 1.262853
    if Q.tau1 >= 0.1953848:
        z += -4.756723 * Q.tau1 - 1.175731
    if Q.girth2_top30 < 0.004763596:
        z += -249.7592 * Q.girth2_top30 + 1.417311
    if 0.004763596 <= Q.girth2_top30 < 0.007463985:
        z += -84.269 * Q.girth2_top30 + 0.6289826
    if Q.e2_sq < 0.009606007:
        z += 185.3061 * Q.e2_sq - 1.780052
    if Q.sum_pt_top40 < 1225.842:
        z += -0.004912728 * Q.sum_pt_top40 + 6.02223
    if Q.e2 < 0.01036127:
        z += 21.52622 * Q.e2 - 0.4759055
    if 0.01036127 <= Q.e2 < 0.02210818:
        z += 47.90917 * Q.e2 - 0.7492666
    if Q.e2 >= 0.02210818:
        z += 26.38296 * Q.e2 - 0.273361
    if Q.z_top30_slots >= 0.9564984:
        z += 18.25141 * Q.z_top30_slots - 17.45744
    if Q.LHA < 0.2454112:
        z += -6.254213 * Q.LHA + 1.534854
    if Q.zdr_0 >= 0.01240028:
        z += -19.41198 * Q.zdr_0 + 0.2407139
    if Q.soft2_dr < 0.3845054:
        z += 1.145816 * Q.soft2_dr - 0.4405726
    if Q.soft2_dr0 < 0.4086381:
        z += -0.9293616 * Q.soft2_dr0 + 0.3797726
    if Q.tau4 < 0.02017767:
        z += 28.65054 * Q.tau4 - 0.5781012
    if Q.girth2 < 0.01397874:
        z += 32.67836 * Q.girth2 - 0.4568023
    if Q.soft1_z < 0.001452174:
        z += -1081.517 * Q.soft1_z + 1.570551
    if Q.soft1_pt < 2.275391:
        z += 0.8962801 * Q.soft1_pt - 2.039387
    if Q.tau21_b2 < 0.7058597:
        z += 0.6098643 * Q.tau21_b2 - 0.4304786
    if Q.n_particles > 38.0 and Q.zdr_0 < 0.009970338:
        z += 1.896814 * (Q.n_particles - 38.0) * (0.009970338 - Q.zdr_0)
    if Q.n_particles > 38.0 and Q.soft1_z < 0.002181998:
        z += -17.53072 * (Q.n_particles - 38.0) * (0.002181998 - Q.soft1_z)
    if Q.z_top5 > 0.6551948 and Q.pt1_dr01 < 28.39396:
        z += -0.2624334 * (Q.z_top5 - 0.6551948) * (28.39396 - Q.pt1_dr01)
    if Q.z_top30_slots > 0.9341838 and Q.pt1_dr01 > 5.351077:
        z += 0.4164276 * (Q.z_top30_slots - 0.9341838) * (Q.pt1_dr01 - 5.351077)
    if Q.M3 < 0.03457336 and Q.psi_0p3 > 0.9299135:
        z += -286.5671 * (0.03457336 - Q.M3) * (Q.psi_0p3 - 0.9299135)
    if Q.z_dr_0_0p05 > 0.8103116 and Q.n_dr_0p05_0p1 < 10.0:
        z += -0.4147933 * (Q.z_dr_0_0p05 - 0.8103116) * (10.0 - Q.n_dr_0p05_0p1)
    if Q.z_top30_slots > 0.9564984 and Q.ptdr0_3 < 10.17512:
        z += -1.714058 * (Q.z_top30_slots - 0.9564984) * (10.17512 - Q.ptdr0_3)
    if Q.z_top30_slots > 0.9564984 and Q.ptdr0_5 > 0.4637912:
        z += 1.109202 * (Q.z_top30_slots - 0.9564984) * (Q.ptdr0_5 - 0.4637912)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.8276832
    if 6.930088 <= Q.log_sum_pt < 7.017258:
        z += 9.429903 * Q.log_sum_pt - 65.35006
    if 7.017258 <= Q.log_sum_pt < 7.062574:
        z += 0.5783224 * Q.log_sum_pt - 3.236236
    if Q.log_sum_pt >= 7.062574:
        z += -6.648414 * Q.log_sum_pt + 47.80313
    if Q.girth2 < 0.003638856:
        z += 792.3793 * Q.girth2 - 4.698954
    if 0.003638856 <= Q.girth2 < 0.006941794:
        z += 549.6923 * Q.girth2 - 3.815851
    if Q.girth2_top15 < 0.01563836:
        z += -42.12336 * Q.girth2_top15 + 0.6587401
    if 1007.788 <= Q.sum_pt < 1085.125:
        z += 0.008050145 * Q.sum_pt - 8.112843
    if Q.sum_pt >= 1085.125:
        z += 0.003357302 * Q.sum_pt - 3.020522
    if Q.girth2_top50 < 0.00634935:
        z += -470.9539 * Q.girth2_top50 + 2.990251
    if Q.sum_pt_top50 >= 1003.544:
        z += -0.00777744 * Q.sum_pt_top50 + 7.805005
    if Q.girth2_top40 < 0.008840538:
        z += 47.47056 * Q.girth2_top40 - 0.4196653
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.01218449 * Q.sum_pt_top40 - 13.0334
    if 800.732 <= Q.sum_pt_top30 < 1011.524:
        z += -0.00207418 * Q.sum_pt_top30 + 1.660863
    if Q.sum_pt_top30 >= 1011.524:
        z += -0.007405567 * Q.sum_pt_top30 + 7.053688
    if Q.z_top30_slots >= 0.9203881:
        z += 6.437678 * Q.z_top30_slots - 5.925162
    if Q.LHA >= 0.3719813:
        z += -218.3378 * Q.LHA + 81.21756
    if Q.lam1 < 0.01174405:
        z += -116.1563 * Q.lam1 + 1.364145
    if Q.log_sum_pt > 6.903423 and Q.girth2_top15 < 0.02146578:
        z += 232.3819 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.girth2_top15)
    if Q.log_sum_pt > 7.062574 and Q.girth2_top30 > 0.01215787:
        z += -28342.68 * (Q.log_sum_pt - 7.062574) * (Q.girth2_top30 - 0.01215787)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p1 < 0.8747961:
        z += -15.43714 * (Q.log_sum_pt - 6.903423) * (0.8747961 - Q.psi_0p1)
    if Q.sum_pt_top30 > 1111.245 and Q.z_dr_0p1_0p2 < 0.1203437:
        z += 0.02132104 * (Q.sum_pt_top30 - 1111.245) * (0.1203437 - Q.z_dr_0p1_0p2)
    if Q.sum_pt_top15 > 1003.329 and Q.lam2 > 0.001163277:
        z += -5.842872 * (Q.sum_pt_top15 - 1003.329) * (Q.lam2 - 0.001163277)
    if Q.sum_pt_top50 > 1003.544 and Q.z_dr_0p1_0p2 > 0.003353111:
        z += 0.01631779 * (Q.sum_pt_top50 - 1003.544) * (Q.z_dr_0p1_0p2 - 0.003353111)
    if Q.sum_pt_top15 > 1003.329 and Q.eta_1 > 0.08734131:
        z += -0.7243729 * (Q.sum_pt_top15 - 1003.329) * (Q.eta_1 - 0.08734131)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.266503
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.2122705 * Q.n_dr_0p2_0p4 + 1.061352
    if Q.n_particles < 46.0:
        z += -0.07205436 * Q.n_particles + 3.314501
    if Q.tau21 < 0.347196:
        z += 4.542716 * Q.tau21 - 1.577213
    if Q.D2 < 1.788105:
        z += -0.2869428 * Q.D2 + 0.5130839
    if Q.e2_sq < 0.007511864:
        z += -149.8561 * Q.e2_sq + 1.125699
    if Q.LHA < 0.3719813:
        z += 2.859671 * Q.LHA - 1.063744
    if Q.girth2_top50 < 0.00634935:
        z += 43.30798 * Q.girth2_top50 - 0.2749775
    if Q.girth2 < 0.009614971:
        z += 66.31719 * Q.girth2 - 0.6376379
    if Q.z_dr_0p2_0p4 < 0.02647293:
        z += 10.92507 * Q.z_dr_0p2_0p4 - 0.2892186
    if Q.girth2_top40 < 0.006259772:
        z += -24.14194 * Q.girth2_top40 + 0.4559751
    if 0.006259772 <= Q.girth2_top40 < 0.008840538:
        z += -118.1246 * Q.girth2_top40 + 1.044285
    if Q.n_dr_0_0p05 < 25.0:
        z += 0.02833469 * Q.n_dr_0_0p05 - 0.7083673
    if Q.tau21_b2 < 0.2018786:
        z += -6.513724 * Q.tau21_b2 + 1.314982
    if Q.n_dr_0p2_0p4 < 5.0 and Q.sum_pt_top30 < 988.4375:
        z += -0.001544625 * (5.0 - Q.n_dr_0p2_0p4) * (988.4375 - Q.sum_pt_top30)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.e2 < 0.03029714:
        z += -11.23748 * (5.0 - Q.n_dr_0p2_0p4) * (0.03029714 - Q.e2)
    if Q.n_particles < 46.0 and Q.e2 > 0.01036127:
        z += -1.727225 * (46.0 - Q.n_particles) * (Q.e2 - 0.01036127)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.z_top50_slots > 0.985099:
        z += 3.684611 * (8.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.985099)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.06927184
    if Q.e2 < 0.03263075:
        z += -4.366413 * Q.e2 - 0.2210002
    if 0.03263075 <= Q.e2 < 0.04358622:
        z += 33.17792 * Q.e2 - 1.4461
    if Q.e2_sq < 0.00592208:
        z += 377.2182 * Q.e2_sq - 2.233916
    if Q.e2_sq >= 0.02580859:
        z += 77.14285 * Q.e2_sq - 1.990948
    if Q.n_particles >= 29.0:
        z += -0.02810716 * Q.n_particles + 0.8151077
    if Q.n_dr_0_0p05 >= 10.0:
        z += 0.02257615 * Q.n_dr_0_0p05 - 0.2257615
    if Q.sum_pt_top50 < 1061.184:
        z += 0.0107372 * Q.sum_pt_top50 - 11.39414
    if Q.girth2 < 0.01397874:
        z += -41.54728 * Q.girth2 + 0.5807787
    if Q.girth2_top20 < 0.007538019:
        z += 84.48224 * Q.girth2_top20 - 0.2389622
    if 0.007538019 <= Q.girth2_top20 < 0.01083435:
        z += -120.6997 * Q.girth2_top20 + 1.307703
    if 0.00242543 <= Q.girth2_top50 < 0.004573744:
        z += -104.1181 * Q.girth2_top50 + 0.2525313
    if Q.girth2_top50 >= 0.004573744:
        z += 60.01907 * Q.girth2_top50 - 0.4981902
    if Q.girth2_top15 < 0.002197765:
        z += -55.70132 * Q.girth2_top15 + 0.8710771
    if 0.002197765 <= Q.girth2_top15 < 0.004169954:
        z += 85.38986 * Q.girth2_top15 + 0.5609919
    if 0.004169954 <= Q.girth2_top15 < 0.007887677:
        z += 39.86115 * Q.girth2_top15 + 0.7508445
    if 0.007887677 <= Q.girth2_top15 < 0.01563836:
        z += -53.53625 * Q.girth2_top15 + 1.487533
    if Q.girth2_top15 >= 0.01563836:
        z += 2.165077 * Q.girth2_top15 + 0.6164559
    if Q.lam1 < 0.006716737:
        z += 110.7783 * Q.lam1 - 0.7440687
    if Q.lam1 >= 0.02048524:
        z += -79.08809 * Q.lam1 + 1.620138
    if Q.width < 0.009614971:
        z += -323.337 * Q.width + 3.108876
    if 0.1870291 <= Q.LHA < 0.2454112:
        z += 2.320769 * Q.LHA - 0.4340513
    if 0.2454112 <= Q.LHA < 0.2845608:
        z += -1.116507 * Q.LHA + 0.4094948
    if 0.2845608 <= Q.LHA < 0.3719813:
        z += -6.089052 * Q.LHA + 1.824486
    if Q.LHA >= 0.3719813:
        z += -22.51505 * Q.LHA + 7.934648
    if Q.girth >= 0.076787:
        z += 17.77995 * Q.girth - 1.365269
    if Q.girth2_top40 < 0.008031986:
        z += 344.9697 * Q.girth2_top40 - 2.770792
    if Q.sum_pt_top40 < 1053.047:
        z += -0.006997215 * Q.sum_pt_top40 + 7.368399
    if Q.psi_0p1 >= 0.3628388:
        z += -0.8538718 * Q.psi_0p1 + 0.3098178
    if Q.girth2 < 0.01397874 and Q.psi_0p3 > 0.9777125:
        z += 1565.781 * (0.01397874 - Q.girth2) * (Q.psi_0p3 - 0.9777125)
    if Q.n_particles > 29.0 and Q.soft1_pt < 2.275391:
        z += 0.008276195 * (Q.n_particles - 29.0) * (2.275391 - Q.soft1_pt)
    if Q.psi_0p3 > 0.9973959 and Q.sum_pt_top40 > 858.8262:
        z += 0.6697973 * (Q.psi_0p3 - 0.9973959) * (Q.sum_pt_top40 - 858.8262)
    if Q.sj2_dr > 0.2595052 and Q.C2_b2 < 0.0329485:
        z += -279.0571 * (Q.sj2_dr - 0.2595052) * (0.0329485 - Q.C2_b2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.9528859
    z += -0.03934027 * Q.n_particles + 2.517777
    if 0.007872294 <= Q.e2_sq < 0.009606007:
        z += -366.1219 * Q.e2_sq + 2.882219
    if Q.e2_sq >= 0.009606007:
        z += -252.195 * Q.e2_sq + 1.787837
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -29.43704 * Q.log_sum_pt + 203.4138
    if 6.920349 <= Q.log_sum_pt < 6.935549:
        z += -33.87296 * Q.log_sum_pt + 234.1119
    if 6.935549 <= Q.log_sum_pt < 6.959294:
        z += -26.56779 * Q.log_sum_pt + 183.4465
    if Q.log_sum_pt >= 6.959294:
        z += -21.62072 * Q.log_sum_pt + 149.0184
    if Q.sum_pt >= 907.9372:
        z += 0.001425555 * Q.sum_pt - 1.294315
    if Q.girth2 >= 0.02928196:
        z += 230.7605 * Q.girth2 - 6.75712
    if Q.width >= 0.006941794:
        z += 161.9423 * Q.width - 1.12417
    if 934.2416 <= Q.sum_pt_top50 < 1024.689:
        z += 0.01560186 * Q.sum_pt_top50 - 14.57591
    if 1024.689 <= Q.sum_pt_top50 < 1048.098:
        z += 0.02007083 * Q.sum_pt_top50 - 19.15521
    if Q.sum_pt_top50 >= 1048.098:
        z += 0.01329594 * Q.sum_pt_top50 - 12.05446
    if Q.n_pt_above_1 >= 26.0:
        z += 0.01911462 * Q.n_pt_above_1 - 0.4969802
    if Q.sum_pt_top20 >= 1017.778:
        z += 0.007227657 * Q.sum_pt_top20 - 7.356152
    if Q.sum_pt_top5 < 531.1875:
        z += 0.0002650005 * Q.sum_pt_top5 - 0.6839422
    if 531.1875 <= Q.sum_pt_top5 < 579.875:
        z += -0.0007953029 * Q.sum_pt_top5 - 0.1207223
    if 579.875 <= Q.sum_pt_top5 < 902.4062:
        z += 0.0005838011 * Q.sum_pt_top5 - 0.9204302
    if Q.sum_pt_top5 >= 902.4062:
        z += -0.001060303 * Q.sum_pt_top5 + 0.5632199
    if Q.n_for_90pct >= 11.0:
        z += 0.01491065 * Q.n_for_90pct - 0.1640171
    if Q.girth2_top15 < 0.01563836:
        z += 25.63259 * Q.girth2_top15 - 0.4008516
    if Q.sum_pt_top30 >= 911.9328:
        z += 0.003316675 * Q.sum_pt_top30 - 3.024585
    if Q.z_top30_slots >= 0.9203881:
        z += -3.416373 * Q.z_top30_slots + 3.144389
    if Q.n_dr_0p2_0p4 < 8.0:
        z += -0.08862679 * Q.n_dr_0p2_0p4 + 0.7090144
    if Q.girth2_top30 < 0.01215787:
        z += 30.86684 * Q.girth2_top30 - 0.3752751
    if Q.pt_entropy >= 2.903111:
        z += -0.3829942 * Q.pt_entropy + 1.111874
    if 951.1375 <= Q.sum_pt_top15 < 967.7705:
        z += -0.003519956 * Q.sum_pt_top15 + 3.347962
    if Q.sum_pt_top15 >= 967.7705:
        z += -0.005518537 * Q.sum_pt_top15 + 5.28213
    if Q.girth2_top20 < 0.008031209:
        z += 58.27165 * Q.girth2_top20 - 0.4679918
    if Q.tau2 >= 0.02164722:
        z += 7.29241 * Q.tau2 - 0.1578604
    if Q.e2 < 0.03875945:
        z += -16.04746 * Q.e2 + 0.6219908
    if Q.tau1 < 0.1072713:
        z += 13.0755 * Q.tau1 - 1.402625
    if Q.girth < 0.07374472:
        z += -13.17251 * Q.girth + 0.9714029
    if Q.girth >= 0.1564779:
        z += -236.9788 * Q.girth + 37.08194
    if Q.e3 < 0.0005178279:
        z += 1422.715 * Q.e3 - 0.7367213
    if Q.N2 < 0.3572263:
        z += 2.302573 * Q.N2 - 0.8225397
    if Q.n_particles < 64.0 and Q.D2 < 2.410481:
        z += -0.01277744 * (64.0 - Q.n_particles) * (2.410481 - Q.D2)
    if Q.n_pt_above_1 > 26.0 and Q.n_dr_0_0p05 < 30.0:
        z += -0.0004079712 * (Q.n_pt_above_1 - 26.0) * (30.0 - Q.n_dr_0_0p05)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 99588.02 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    if Q.e2_sq > 0.009606007 and Q.soft7_z < 0.003627839:
        z += -83051.95 * (Q.e2_sq - 0.009606007) * (0.003627839 - Q.soft7_z)
    if Q.girth2 > 0.02928196 and Q.soft3_z < 0.002741632:
        z += -815481.1 * (Q.girth2 - 0.02928196) * (0.002741632 - Q.soft3_z)
    if Q.e2_sq > 0.007872294 and Q.soft7_z < 0.003627839:
        z += 70586.73 * (Q.e2_sq - 0.007872294) * (0.003627839 - Q.soft7_z)
    if Q.sum_pt_top50 > 1024.689 and Q.e4 < 5.8505e-08:
        z += -165668.0 * (Q.sum_pt_top50 - 1024.689) * (5.8505e-08 - Q.e4)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.9453069
    if Q.lam1 < 0.003811746:
        z += -986.5342 * Q.lam1 + 4.360774
    if 0.003811746 <= Q.lam1 < 0.006189818:
        z += -155.5528 * Q.lam1 + 1.193283
    if 0.006189818 <= Q.lam1 < 0.007671243:
        z += -183.8681 * Q.lam1 + 1.36855
    if Q.lam1 >= 0.007671243:
        z += -28.3153 * Q.lam1 + 0.1752665
    if Q.girth2 < 0.01397874:
        z += 172.2285 * Q.girth2 - 2.407539
    if Q.psi_0p3 >= 0.9853273:
        z += 71.36096 * Q.psi_0p3 - 70.3139
    if Q.e2_sq < 0.007511864:
        z += -266.5398 * Q.e2_sq + 5.170328
    if 0.007511864 <= Q.e2_sq < 0.02580859:
        z += -173.1521 * Q.e2_sq + 4.468812
    if Q.lam2 < 0.0009731947:
        z += -250.0225 * Q.lam2 + 0.4441171
    if 0.0009731947 <= Q.lam2 < 0.001776308:
        z += -451.8783 * Q.lam2 + 0.6405621
    if Q.lam2 >= 0.001776308:
        z += -201.8558 * Q.lam2 + 0.196445
    if Q.width < 0.008190222:
        z += 662.3542 * Q.width - 6.511075
    if 0.008190222 <= Q.width < 0.009614971:
        z += 762.4126 * Q.width - 7.330575
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -3.001695 * Q.z_dr_0p1_0p2 + 0.361235
    if Q.girth2_top20 < 0.00287991:
        z += -117.037 * Q.girth2_top20 - 1.025451
    if 0.00287991 <= Q.girth2_top20 < 0.01083435:
        z += 115.6165 * Q.girth2_top20 - 1.695472
    if 0.01083435 <= Q.girth2_top20 < 0.02265114:
        z += 37.47572 * Q.girth2_top20 - 0.8488677
    if Q.e2 >= 0.01541561:
        z += -59.60213 * Q.e2 + 0.9188032
    if Q.LHA < 0.2091025:
        z += -4.969285 * Q.LHA + 1.292741
    if 0.2091025 <= Q.LHA < 0.2601462:
        z += -0.001920223 * Q.LHA + 0.2540521
    if Q.LHA >= 0.2601462:
        z += 4.967365 * Q.LHA - 1.038689
    if Q.girth2_top40 < 0.008840538:
        z += -97.9136 * Q.girth2_top40 - 0.4669612
    if 0.008840538 <= Q.girth2_top40 < 0.02497133:
        z += 82.61033 * Q.girth2_top40 - 2.06289
    if Q.girth2_top30 >= 0.008376291:
        z += -54.22561 * Q.girth2_top30 + 0.4542095
    if Q.log_sum_pt >= 6.811175:
        z += 2.605705 * Q.log_sum_pt - 17.74791
    if Q.sum_pt_top20 >= 695.3094:
        z += 0.0008284604 * Q.sum_pt_top20 - 0.5760363
    z += 0.02039423 * Q.n_dr_0p1_0p2
    if Q.girth2 < 0.01397874 and Q.psi_0p3 > 0.9853273:
        z += -6853.643 * (0.01397874 - Q.girth2) * (Q.psi_0p3 - 0.9853273)
    if Q.lam1 < 0.003811746 and Q.sum_pt_top50 < 1008.935:
        z += -5.985262 * (0.003811746 - Q.lam1) * (1008.935 - Q.sum_pt_top50)
    if Q.e2 > 0.04755309 and Q.pt_14 < 26.14062:
        z += 4.235493 * (Q.e2 - 0.04755309) * (26.14062 - Q.pt_14)
    if Q.e3 < 0.0003372339 and Q.N2 < 0.3384815:
        z += -11941.08 * (0.0003372339 - Q.e3) * (0.3384815 - Q.N2)
    if Q.e2 > 0.04755309 and Q.z_dr_0_0p05 < 0.3289237:
        z += 197.8356 * (Q.e2 - 0.04755309) * (0.3289237 - Q.z_dr_0_0p05)
    if Q.e2_sq < 0.007511864 and Q.sum_pt < 1007.788:
        z += 5.766861 * (0.007511864 - Q.e2_sq) * (1007.788 - Q.sum_pt)
    if Q.e2_sq < 0.02580859 and Q.log_sum_pt < 6.98945:
        z += -245.5509 * (0.02580859 - Q.e2_sq) * (6.98945 - Q.log_sum_pt)
    if Q.sum_pt_top20 > 695.3094 and Q.D2_b2 < 1.36316:
        z += 0.001247523 * (Q.sum_pt_top20 - 695.3094) * (1.36316 - Q.D2_b2)
    if Q.log_sum_pt > 6.811175 and Q.girth2_top3 < 0.001592178:
        z += -1025.176 * (Q.log_sum_pt - 6.811175) * (0.001592178 - Q.girth2_top3)
    if Q.z_top20_slots > 0.7111557 and Q.girth2_top3 > 0.01002369:
        z += 328.488 * (Q.z_top20_slots - 0.7111557) * (Q.girth2_top3 - 0.01002369)
    if Q.z_top20_slots > 0.7111557 and Q.dr01 > 0.1662967:
        z += -26.89622 * (Q.z_top20_slots - 0.7111557) * (Q.dr01 - 0.1662967)
    if Q.e2 > 0.04755309 and Q.C2_b2 > 0.02704832:
        z += 1984.703 * (Q.e2 - 0.04755309) * (Q.C2_b2 - 0.02704832)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.7528688
    if Q.tau21_b2 < 0.2352054:
        z += -3.191628 * Q.tau21_b2 + 0.7506882
    if Q.girth2 < 0.007877041:
        z += 937.6538 * Q.girth2 - 7.385938
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.02080364 * Q.n_dr_0p2_0p4 + 0.3120546
    if Q.lam1 < 0.005913555:
        z += 567.5161 * Q.lam1 - 3.755153
    if 0.005913555 <= Q.lam1 < 0.007671243:
        z += 57.22897 * Q.lam1 - 0.7375426
    if 0.007671243 <= Q.lam1 < 0.008241985:
        z += 523.0471 * Q.lam1 - 4.310947
    if Q.lam2 < 0.002396991:
        z += 139.6704 * Q.lam2 - 0.3347887
    if Q.tau1 < 0.07708632:
        z += -10.03796 * Q.tau1 + 0.7737891
    if Q.e2_sq < 0.006936725:
        z += -750.4932 * Q.e2_sq + 8.128055
    if 0.006936725 <= Q.e2_sq < 0.009606007:
        z += -1094.71 * Q.e2_sq + 10.51579
    if Q.girth < 0.05048381:
        z += -41.38493 * Q.girth + 0.4596652
    if 0.05048381 <= Q.girth < 0.05660088:
        z += -24.97044 * Q.girth - 0.3690009
    if 0.05660088 <= Q.girth < 0.08589404:
        z += 60.84523 * Q.girth - 5.226243
    if Q.width < 0.01397874:
        z += -328.8007 * Q.width + 4.596221
    if Q.girth2_top50 < 0.00634935:
        z += 832.2175 * Q.girth2_top50 - 5.456502
    if 0.00634935 <= Q.girth2_top50 < 0.007820315:
        z += -64.86804 * Q.girth2_top50 + 0.2394078
    if 0.007820315 <= Q.girth2_top50 < 0.008124776:
        z += 879.8507 * Q.girth2_top50 - 7.14859
    if Q.tau2 < 0.04828819:
        z += 11.27064 * Q.tau2 - 0.544239
    if Q.LHA < 0.2454112:
        z += 16.28607 * Q.LHA - 3.145132
    if 0.2454112 <= Q.LHA < 0.2601462:
        z += -11.76897 * Q.LHA + 3.739889
    if 0.2601462 <= Q.LHA < 0.302389:
        z += -16.05565 * Q.LHA + 4.855051
    if Q.e2 < 0.03480688:
        z += -16.18578 * Q.e2 + 0.5633764
    if Q.psi_0p3 >= 0.9966167:
        z += 132.3283 * Q.psi_0p3 - 131.8806
    if 0.2404747 <= Q.max_dr < 0.2982:
        z += -2.977597 * Q.max_dr + 0.7160367
    if Q.max_dr >= 0.2982:
        z += -1.438406 * Q.max_dr + 0.2570499
    if Q.mean_eta2 < 0.0009793444:
        z += 466.2276 * Q.mean_eta2 - 0.4565974
    if Q.girth2_top30 < 0.00375223:
        z += 109.8764 * Q.girth2_top30 + 0.491101
    if 0.00375223 <= Q.girth2_top30 < 0.006363916:
        z += -92.14635 * Q.girth2_top30 + 1.249137
    if 0.006363916 <= Q.girth2_top30 < 0.008376291:
        z += 75.72285 * Q.girth2_top30 + 0.1808314
    if 0.008376291 <= Q.girth2_top30 < 0.01215787:
        z += -215.5468 * Q.girth2_top30 + 2.620591
    if Q.psi_0p2 >= 0.9087063:
        z += -7.084135 * Q.psi_0p2 + 6.437398
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1937688:
        z += -36.12357 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1937688)
    if Q.tau21_b2 < 0.2352054 and Q.e2_sq < 0.007872294:
        z += -3038.467 * (0.2352054 - Q.tau21_b2) * (0.007872294 - Q.e2_sq)
    if Q.tau21_b2 < 0.2352054 and Q.e2_sq < 0.01396296:
        z += 230.3729 * (0.2352054 - Q.tau21_b2) * (0.01396296 - Q.e2_sq)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt_top30 > 978.0762:
        z += 0.02853208 * (0.2352054 - Q.tau21_b2) * (Q.sum_pt_top30 - 978.0762)
    if Q.psi_0p3 > 0.9980008 and Q.dr_0 < 0.0361727:
        z += -10767.42 * (Q.psi_0p3 - 0.9980008) * (0.0361727 - Q.dr_0)
    if Q.e2_sq < 0.009606007 and Q.sum_pt < 1167.447:
        z += -6.756675 * (0.009606007 - Q.e2_sq) * (1167.447 - Q.sum_pt)
    if Q.lam1 < 0.005913555 and Q.sum_pt < 1167.447:
        z += 3.186049 * (0.005913555 - Q.lam1) * (1167.447 - Q.sum_pt)
    if Q.lam2 < 0.002396991 and Q.log_sum_pt < 7.139296:
        z += 1348.522 * (0.002396991 - Q.lam2) * (7.139296 - Q.log_sum_pt)
    if Q.lam2 < 0.002396991 and Q.zdr_0 > 0.002524869:
        z += -10806.49 * (0.002396991 - Q.lam2) * (Q.zdr_0 - 0.002524869)
    if Q.girth < 0.08589404 and Q.sum_pt_top50 < 1156.659:
        z += 0.1174925 * (0.08589404 - Q.girth) * (1156.659 - Q.sum_pt_top50)
    if Q.sum_pt_top15 > 1082.548 and Q.mean_phi > 2.298159e-05:
        z += -69.48254 * (Q.sum_pt_top15 - 1082.548) * (Q.mean_phi - 2.298159e-05)
    if Q.psi_0p3 > 0.9966167 and Q.soft1_pt < 2.275391:
        z += 55.84187 * (Q.psi_0p3 - 0.9966167) * (2.275391 - Q.soft1_pt)
    if Q.sum_pt_top15 > 1082.548 and Q.mean_phi > -3.355129e-05:
        z += 40.64831 * (Q.sum_pt_top15 - 1082.548) * (Q.mean_phi - -3.355129e-05)
    return max(0.0, z)


def neuron_8(Q):
    z = 1.042206
    if Q.girth2 < 0.003638856:
        z += -905.5091 * Q.girth2 + 6.738367
    if 0.003638856 <= Q.girth2 < 0.01397874:
        z += -7.21048 * Q.girth2 + 3.469588
    if 0.01397874 <= Q.girth2 < 0.0258669:
        z += -283.3739 * Q.girth2 + 7.330005
    if Q.sum_pt_top40 < 1007.44:
        z += 0.00578122 * Q.sum_pt_top40 - 5.824232
    if Q.n_dr_0p2_0p4 < 18.0:
        z += 0.06826857 * Q.n_dr_0p2_0p4 - 1.228834
    if Q.e2_sq < 0.002574843:
        z += 930.2069 * Q.e2_sq - 5.479075
    if 0.002574843 <= Q.e2_sq < 0.009606007:
        z += 438.6098 * Q.e2_sq - 4.213289
    if Q.girth2_top30 < 0.007463985:
        z += 694.5333 * Q.girth2_top30 - 5.183986
    if Q.girth2_top50 < 0.01351431:
        z += -83.26046 * Q.girth2_top50 + 1.125208
    if Q.sum_pt_top50 < 1156.659:
        z += -0.01373405 * Q.sum_pt_top50 + 16.43096
    if 1156.659 <= Q.sum_pt_top50 < 1245.697:
        z += -0.006124854 * Q.sum_pt_top50 + 7.62971
    if Q.width < 0.008190222:
        z += 476.2668 * Q.width - 3.900731
    if Q.log_sum_pt < 6.811175:
        z += 4.881365 * Q.log_sum_pt - 33.24783
    if Q.tau1 < 0.1219132:
        z += -31.61824 * Q.tau1 + 3.854682
    if 0.005788041 <= Q.girth2_top15 < 0.007887677:
        z += -151.4694 * Q.girth2_top15 + 0.8767109
    if Q.girth2_top15 >= 0.007887677:
        z += 20.40355 * Q.girth2_top15 - 0.4789672
    if Q.girth2_top40 < 0.008840538:
        z += -82.27393 * Q.girth2_top40 + 0.7273459
    if Q.sum_pt < 1028.184:
        z += -0.01979114 * Q.sum_pt + 20.34893
    if Q.sum_pt_top20 >= 1017.778:
        z += 0.002873867 * Q.sum_pt_top20 - 2.924959
    if Q.girth2 < 0.0258669 and Q.log_sum_pt < 7.139296:
        z += -1617.458 * (0.0258669 - Q.girth2) * (7.139296 - Q.log_sum_pt)
    if Q.girth2_top30 < 0.007463985 and Q.sum_pt < 1260.541:
        z += 4.843924 * (0.007463985 - Q.girth2_top30) * (1260.541 - Q.sum_pt)
    if Q.girth2_top30 < 0.007463985 and Q.sum_pt_top40 < 1095.686:
        z += -3.299757 * (0.007463985 - Q.girth2_top30) * (1095.686 - Q.sum_pt_top40)
    if Q.girth2 < 0.01397874 and Q.sum_pt < 986.0565:
        z += 2.649632 * (0.01397874 - Q.girth2) * (986.0565 - Q.sum_pt)
    if Q.girth2 < 0.0258669 and Q.z_top50_slots > 0.9704436:
        z += -931.8813 * (0.0258669 - Q.girth2) * (Q.z_top50_slots - 0.9704436)
    if Q.tau1 < 0.1219132 and Q.planar_flow < 0.5364935:
        z += -26.78524 * (0.1219132 - Q.tau1) * (0.5364935 - Q.planar_flow)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.sj2_zsoft > 0.1181474:
        z += 0.1127041 * (18.0 - Q.n_dr_0p2_0p4) * (Q.sj2_zsoft - 0.1181474)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.7087165
    if Q.girth2_top40 < 0.005712208:
        z += -250.8824 * Q.girth2_top40 + 1.433092
    if Q.sum_pt_top40 < 972.4111:
        z += -0.008257322 * Q.sum_pt_top40 + 8.029512
    if Q.girth2 >= 0.0258669:
        z += -617.0937 * Q.girth2 + 15.9623
    if Q.LHA >= 0.3332345:
        z += 5.015996 * Q.LHA - 1.671503
    if Q.lam1 < 0.004673423:
        z += -260.0898 * Q.lam1 + 1.888066
    if 0.004673423 <= Q.lam1 < 0.006716737:
        z += -90.30403 * Q.lam1 + 1.094586
    if 0.006716737 <= Q.lam1 < 0.007259287:
        z += -183.6593 * Q.lam1 + 1.721628
    if Q.lam1 >= 0.007259287:
        z += 76.43052 * Q.lam1 - 0.1664382
    if 0.01989454 <= Q.e2_sq < 0.0292152:
        z += -159.4305 * Q.e2_sq + 3.171796
    if Q.e2_sq >= 0.0292152:
        z += 419.7314 * Q.e2_sq - 13.74853
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -5.268234 * Q.z_dr_0_0p05 + 4.630283
    if Q.sum_pt_top30 < 1191.938:
        z += -0.003612689 * Q.sum_pt_top30 + 4.306101
    if Q.width < 0.006941794:
        z += -510.378 * Q.width + 3.542939
    if Q.girth2_top15 < 0.004169954:
        z += 148.1359 * Q.girth2_top15 - 1.188057
    if 0.004169954 <= Q.girth2_top15 < 0.02675364:
        z += 25.25439 * Q.girth2_top15 - 0.6756467
    if Q.sum_pt < 986.0565:
        z += -0.005726425 * Q.sum_pt + 5.646579
    if Q.girth >= 0.1207452:
        z += -23.71551 * Q.girth + 2.863533
    if Q.e3 >= 0.0005178279:
        z += 5645.965 * Q.e3 - 2.923638
    if Q.girth2_top20 < 0.01083435:
        z += 42.44669 * Q.girth2_top20 - 0.4598825
    if Q.tau1 < 0.06310829:
        z += 10.71501 * Q.tau1 - 0.6762057
    if Q.girth2_top40 < 0.005712208 and Q.sum_pt < 1007.788:
        z += -2.689249 * (0.005712208 - Q.girth2_top40) * (1007.788 - Q.sum_pt)
    if Q.girth2_top40 < 0.005712208 and Q.log_sum_pt > 7.017258:
        z += 4031.333 * (0.005712208 - Q.girth2_top40) * (Q.log_sum_pt - 7.017258)
    if Q.sum_pt_top40 < 972.4111 and Q.soft4_pt > 1.106445:
        z += -0.00310288 * (972.4111 - Q.sum_pt_top40) * (Q.soft4_pt - 1.106445)
    if Q.LHA > 0.3332345 and Q.sum_pt < 1042.609:
        z += 0.07071688 * (Q.LHA - 0.3332345) * (1042.609 - Q.sum_pt)
    if Q.lam1 < 0.007259287 and Q.sum_pt > 1085.125:
        z += -2.697803 * (0.007259287 - Q.lam1) * (Q.sum_pt - 1085.125)
    if Q.z_dr_0_0p05 > 0.878906 and Q.sum_pt_top40 > 1013.042:
        z += 0.03868923 * (Q.z_dr_0_0p05 - 0.878906) * (Q.sum_pt_top40 - 1013.042)
    if Q.sum_pt_top40 < 972.4111 and Q.tau3 < 0.0369869:
        z += -0.6980432 * (972.4111 - Q.sum_pt_top40) * (0.0369869 - Q.tau3)
    if Q.girth2_top15 < 0.004169954 and Q.n_dr_0p2_0p4 > 8.0:
        z += 24.02242 * (0.004169954 - Q.girth2_top15) * (Q.n_dr_0p2_0p4 - 8.0)
    if Q.log_sum_pt < 6.903423 and Q.z_dr_0p1_0p2 < 0.4479367:
        z += 13.68587 * (6.903423 - Q.log_sum_pt) * (0.4479367 - Q.z_dr_0p1_0p2)
    if Q.girth > 0.1207452 and Q.sj3_pairmin_over_m < 0.4606099:
        z += 84.56539 * (Q.girth - 0.1207452) * (0.4606099 - Q.sj3_pairmin_over_m)
    if Q.e2_sq > 0.0292152 and Q.soft7_pt < 3.779492:
        z += -235.0272 * (Q.e2_sq - 0.0292152) * (3.779492 - Q.soft7_pt)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.620965
    if Q.girth < 0.03577037:
        z += 139.8029 * Q.girth - 9.464266
    if 0.03577037 <= Q.girth < 0.1207452:
        z += 52.52692 * Q.girth - 6.342372
    if Q.girth2_top30 < 0.005402331:
        z += 71.69109 * Q.girth2_top30 - 0.387299
    if Q.sum_pt_top3 < 309.625:
        z += -0.0007673121 * Q.sum_pt_top3 + 0.5036517
    if 309.625 <= Q.sum_pt_top3 < 656.3844:
        z += -0.003176538 * Q.sum_pt_top3 + 1.249608
    if Q.sum_pt_top3 >= 656.3844:
        z += -0.002409226 * Q.sum_pt_top3 + 0.7459566
    if Q.e2 >= 0.006720044:
        z += 38.40393 * Q.e2 - 0.2580761
    if Q.e2_sq < 0.00592208:
        z += -234.1927 * Q.e2_sq + 1.007649
    if 0.00592208 <= Q.e2_sq < 0.00818374:
        z += -14.74762 * Q.e2_sq - 0.291922
    if 0.00818374 <= Q.e2_sq < 0.009606007:
        z += 290.109 * Q.e2_sq - 2.78679
    if Q.e2_sq >= 0.02580859:
        z += -304.7543 * Q.e2_sq + 7.865278
    if Q.z_dr_0_0p05 >= 0.7128619:
        z += -1.296784 * Q.z_dr_0_0p05 + 0.924428
    if Q.M2 < 0.08222447:
        z += -12.71821 * Q.M2 + 1.045748
    if Q.width < 0.006403325:
        z += -344.8176 * Q.width + 2.207979
    if Q.tau21_b2 < 0.3062621:
        z += 2.285075 * Q.tau21_b2 - 0.699832
    if Q.girth2 >= 0.01397874:
        z += -155.421 * Q.girth2 + 2.17259
    if Q.dr_0 < 0.06413297:
        z += -5.821622 * Q.dr_0 + 0.373358
    if Q.D2 < 2.410481:
        z += -0.3097054 * Q.D2 + 0.7465391
    if Q.e3 < 0.0005178279:
        z += 1234.651 * Q.e3 - 0.6393369
    if Q.tau4 < 0.0259543:
        z += 25.0165 * Q.tau4 - 0.6492858
    if Q.girth2_top10 < 0.002247756:
        z += 213.9789 * Q.girth2_top10 - 1.407837
    if 0.002247756 <= Q.girth2_top10 < 0.007678544:
        z += 104.0243 * Q.girth2_top10 - 1.160686
    if 0.007678544 <= Q.girth2_top10 < 0.01976735:
        z += 29.93934 * Q.girth2_top10 - 0.5918216
    if Q.n_dr_0p2_0p4 < 15.0:
        z += 0.02723024 * Q.n_dr_0p2_0p4 - 0.4084537
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -5.729156 * Q.z_dr_0p2_0p4 + 0.5226462
    if Q.LHA < 0.3719813:
        z += -7.152036 * Q.LHA + 2.660424
    if Q.girth2_top40 >= 0.02497133:
        z += 116.6017 * Q.girth2_top40 - 2.9117
    if 0.005718442 <= Q.girth2_top20 < 0.008031209:
        z += -171.3951 * Q.girth2_top20 + 0.9801132
    if Q.girth2_top20 >= 0.008031209:
        z += -3.867188 * Q.girth2_top20 - 0.3653388
    if Q.tau1 < 0.1507173:
        z += -22.19906 * Q.tau1 + 3.345783
    if Q.e2 > 0.006720044 and Q.log_sum_pt > 6.893714:
        z += -117.9254 * (Q.e2 - 0.006720044) * (Q.log_sum_pt - 6.893714)
    if Q.e2 > 0.006720044 and Q.sj3_pairmin_over_m > 0.1765064:
        z += 26.32949 * (Q.e2 - 0.006720044) * (Q.sj3_pairmin_over_m - 0.1765064)
    if Q.M2 < 0.08222447 and Q.max_dr < 0.4357228:
        z += -52.36031 * (0.08222447 - Q.M2) * (0.4357228 - Q.max_dr)
    if Q.girth2 > 0.01397874 and Q.pt_3 < 114.75:
        z += 0.7366813 * (Q.girth2 - 0.01397874) * (114.75 - Q.pt_3)
    if Q.girth2_top30 < 0.005402331 and Q.pt_6 > 33.6875:
        z += -6.800769 * (0.005402331 - Q.girth2_top30) * (Q.pt_6 - 33.6875)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.789953
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.120573 * Q.n_dr_0p2_0p4 + 1.20573
    if Q.lam1 < 0.005913555:
        z += 304.6021 * Q.lam1 - 1.801281
    if Q.e2_sq < 0.004754578:
        z += -973.3397 * Q.e2_sq + 7.085494
    if 0.004754578 <= Q.e2_sq < 0.00818374:
        z += -716.6982 * Q.e2_sq + 5.865272
    if Q.girth2_top50 < 0.00634935:
        z += 266.93 * Q.girth2_top50 - 1.694832
    if Q.z_top30_slots >= 0.9734886:
        z += -12.48148 * Q.z_top30_slots + 12.15058
    if Q.e2 < 0.02515919:
        z += -60.50631 * Q.e2 + 1.138479
    if 0.02515919 <= Q.e2 < 0.03875945:
        z += 28.22084 * Q.e2 - 1.093824
    if Q.D2 < 4.450169:
        z += 0.08603627 * Q.D2 - 0.3828759
    if Q.C2_b2 >= 0.002289486:
        z += -8.928555 * Q.C2_b2 + 0.0204418
    if Q.girth2_top30 < 0.005809485:
        z += 303.4264 * Q.girth2_top30 - 0.9566985
    if 0.005809485 <= Q.girth2_top30 < 0.006363916:
        z += 477.9878 * Q.girth2_top30 - 1.97081
    if 0.006363916 <= Q.girth2_top30 < 0.01215787:
        z += -184.8588 * Q.girth2_top30 + 2.24749
    if Q.girth2_top5 < 0.00327978:
        z += -69.3201 * Q.girth2_top5 - 0.1050136
    if 0.00327978 <= Q.girth2_top5 < 0.007164202:
        z += 85.56441 * Q.girth2_top5 - 0.6130007
    if Q.n_real_top50 >= 22.0:
        z += -0.02620555 * Q.n_real_top50 + 0.5765222
    if Q.LHA < 0.3098384:
        z += -0.9577065 * Q.LHA + 0.5189548
    if 0.3098384 <= Q.LHA < 0.3332345:
        z += -9.498167 * Q.LHA + 3.165117
    if Q.tau1 < 0.1008497:
        z += 13.98676 * Q.tau1 - 1.31878
    if 0.1008497 <= Q.tau1 < 0.1072713:
        z += -14.29268 * Q.tau1 + 1.533194
    if Q.psi_0p2 >= 0.9313699:
        z += -7.456738 * Q.psi_0p2 + 6.944981
    if Q.girth < 0.09749958:
        z += 7.099604 * Q.girth - 0.6922084
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2 < 0.005532208:
        z += -26.84717 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.002627806 * (10.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.sum_pt_top50 < 1156.659:
        z += -0.0003928224 * (10.0 - Q.n_dr_0p2_0p4) * (1156.659 - Q.sum_pt_top50)
    if Q.lam1 < 0.005913555 and Q.sum_pt < 1115.723:
        z += 3.405365 * (0.005913555 - Q.lam1) * (1115.723 - Q.sum_pt)
    if Q.e2_sq < 0.009606007 and Q.sum_pt < 1115.723:
        z += 0.5094452 * (0.009606007 - Q.e2_sq) * (1115.723 - Q.sum_pt)
    if Q.z_top30_slots > 0.9734886 and Q.D2_b2 < 1.08774:
        z += 23.57222 * (Q.z_top30_slots - 0.9734886) * (1.08774 - Q.D2_b2)
    if Q.e2_sq < 0.00818374 and Q.sum_pt < 1115.723:
        z += -3.410032 * (0.00818374 - Q.e2_sq) * (1115.723 - Q.sum_pt)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.9574029
    if Q.girth2_top30 < 0.005809485:
        z += 255.1604 * Q.girth2_top30 - 1.482351
    if Q.psi_0p1 < 0.08133662:
        z += 11.19635 * Q.psi_0p1 - 0.910673
    if Q.e2_sq < 0.007511864:
        z += -690.3917 * Q.e2_sq + 5.186129
    if Q.sum_pt >= 1085.125:
        z += -0.01096792 * Q.sum_pt + 11.90156
    if 0.9896594 <= Q.psi_0p3 < 0.9943058:
        z += 86.60423 * Q.psi_0p3 - 85.70869
    if Q.psi_0p3 >= 0.9943058:
        z += -97.28886 * Q.psi_0p3 + 97.13727
    if Q.lam1 < 0.002752094:
        z += 977.1779 * Q.lam1 - 2.689285
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.06136539 * Q.n_dr_0p2_0p4 + 0.6136539
    if Q.log_sum_pt < 6.879399:
        z += -4.938553 * Q.log_sum_pt + 33.97428
    if Q.girth2 < 0.006403325 and Q.log_sum_pt > 6.811175:
        z += 9044.99 * (0.006403325 - Q.girth2) * (Q.log_sum_pt - 6.811175)
    if Q.e2_sq < 0.007511864 and Q.log_sum_pt > 6.811175:
        z += -6091.415 * (0.007511864 - Q.e2_sq) * (Q.log_sum_pt - 6.811175)
    if Q.psi_0p1 < 0.08133662 and Q.sum_pt_top50 > 889.8503:
        z += 0.188758 * (0.08133662 - Q.psi_0p1) * (Q.sum_pt_top50 - 889.8503)
    if Q.sum_pt > 1085.125 and Q.lam1 < 0.001868041:
        z += 3.8991 * (Q.sum_pt - 1085.125) * (0.001868041 - Q.lam1)
    if Q.psi_0p3 > 0.9943058 and Q.girth2 < 0.00751625:
        z += 39708.33 * (Q.psi_0p3 - 0.9943058) * (0.00751625 - Q.girth2)
    if Q.sum_pt > 1085.125 and Q.z_dr_0p2_0p4 < 0.019523:
        z += -0.137722 * (Q.sum_pt - 1085.125) * (0.019523 - Q.z_dr_0p2_0p4)
    if Q.psi_0p1 > 0.9371031 and Q.absphi_0 < 0.05444336:
        z += -180.6325 * (Q.psi_0p1 - 0.9371031) * (0.05444336 - Q.absphi_0)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.8263043
    if Q.sum_pt < 1017.435:
        z += 0.01693795 * Q.sum_pt - 18.3798
    if 1017.435 <= Q.sum_pt < 1085.125:
        z += -0.007298267 * Q.sum_pt + 6.278975
    if Q.sum_pt >= 1085.125:
        z += -0.02423622 * Q.sum_pt + 24.65877
    if 0.008840538 <= Q.girth2_top40 < 0.01292642:
        z += -173.3183 * Q.girth2_top40 + 1.532227
    if 0.01292642 <= Q.girth2_top40 < 0.02862386:
        z += -366.3647 * Q.girth2_top40 + 4.027627
    if Q.girth2_top40 >= 0.02862386:
        z += 1712.125 * Q.girth2_top40 - 55.46677
    if Q.n_pt_above_5 < 25.0:
        z += 0.02429174 * Q.n_pt_above_5 - 0.6072934
    if Q.girth < 0.1207452:
        z += 12.59098 * Q.girth - 1.5203
    if Q.sum_pt_top40 >= 972.4111:
        z += -0.008593139 * Q.sum_pt_top40 + 8.356064
    if Q.e2 < 0.03480688:
        z += -28.32603 * Q.e2 + 0.9859408
    if Q.e2 >= 0.03680582:
        z += 23.51088 * Q.e2 - 0.8653371
    if Q.sum_pt_top50 < 889.8503:
        z += -0.01229965 * Q.sum_pt_top50 + 13.27125
    if 889.8503 <= Q.sum_pt_top50 < 976.277:
        z += -0.01886982 * Q.sum_pt_top50 + 19.11772
    if 976.277 <= Q.sum_pt_top50 < 1013.916:
        z += -0.004312797 * Q.sum_pt_top50 + 4.906031
    if 1013.916 <= Q.sum_pt_top50 < 1078.994:
        z += 0.004505471 * Q.sum_pt_top50 - 4.03495
    if Q.sum_pt_top50 >= 1078.994:
        z += 0.01680512 * Q.sum_pt_top50 - 17.3062
    if 6.811175 <= Q.log_sum_pt < 6.893714:
        z += 27.56009 * Q.log_sum_pt - 187.7166
    if Q.log_sum_pt >= 6.893714:
        z += 14.65982 * Q.log_sum_pt - 98.78584
    if Q.width >= 0.01991191:
        z += -168.3125 * Q.width + 3.351423
    if Q.n_particles < 54.0:
        z += 0.02752354 * Q.n_particles - 1.486271
    if Q.e2_sq >= 0.0292152:
        z += -1020.479 * Q.e2_sq + 29.81351
    if 0.005852839 <= Q.girth2_top50 < 0.01351431:
        z += 94.75779 * Q.girth2_top50 - 0.5546021
    if Q.girth2_top50 >= 0.01351431:
        z += -149.4408 * Q.girth2_top50 + 2.745573
    if Q.sum_pt_top10 >= 867.9266:
        z += 0.00349488 * Q.sum_pt_top10 - 3.033299
    if Q.psi_0p3 >= 0.9980008:
        z += 223.2027 * Q.psi_0p3 - 222.7564
    if Q.sum_pt_top15 < 795.6047:
        z += 0.00357781 * Q.sum_pt_top15 - 2.846522
    if Q.girth2_top40 > 0.01292642 and Q.sum_pt_top50 < 1245.697:
        z += 1.386284 * (Q.girth2_top40 - 0.01292642) * (1245.697 - Q.sum_pt_top50)
    if Q.sum_pt < 1085.125 and Q.tau21 > 0.1295048:
        z += 0.00662668 * (1085.125 - Q.sum_pt) * (Q.tau21 - 0.1295048)
    if Q.girth2_top40 > 0.02862386 and Q.pt_6 > 19.46875:
        z += -80.29738 * (Q.girth2_top40 - 0.02862386) * (Q.pt_6 - 19.46875)
    if Q.girth2_top40 > 0.02862386 and Q.z_6 > 0.01906139:
        z += -44579.83 * (Q.girth2_top40 - 0.02862386) * (Q.z_6 - 0.01906139)
    if Q.girth2_top40 > 0.02862386 and Q.pt_6 < 62.25:
        z += -33.80071 * (Q.girth2_top40 - 0.02862386) * (62.25 - Q.pt_6)
    if Q.log_sum_pt > 6.811175 and Q.C2 > 0.06655881:
        z += -32.31798 * (Q.log_sum_pt - 6.811175) * (Q.C2 - 0.06655881)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.01677902
    if Q.tau21_b2 < 0.342495:
        z += -2.197915 * Q.tau21_b2 + 0.7527747
    if 0.9896594 <= Q.psi_0p3 < 0.9956185:
        z += 134.1496 * Q.psi_0p3 - 132.7625
    if Q.psi_0p3 >= 0.9956185:
        z += 564.3562 * Q.psi_0p3 - 561.084
    if Q.LHA < 0.1632346:
        z += -3.667713 * Q.LHA - 0.1972626
    if 0.1632346 <= Q.LHA < 0.2091025:
        z += 17.35333 * Q.LHA - 3.628625
    if Q.LHA >= 0.3098384:
        z += 5.430693 * Q.LHA - 1.682637
    if Q.e2_sq < 0.00363788:
        z += 246.7455 * Q.e2_sq + 2.89988
    if 0.00363788 <= Q.e2_sq < 0.00616708:
        z += 117.1945 * Q.e2_sq + 3.371171
    if 0.00616708 <= Q.e2_sq < 0.009606007:
        z += -616.0866 * Q.e2_sq + 7.893375
    if 0.009606007 <= Q.e2_sq < 0.01396296:
        z += -519.786 * Q.e2_sq + 6.96831
    if 0.01396296 <= Q.e2_sq < 0.01989454:
        z += -306.263 * Q.e2_sq + 3.986896
    if Q.e2_sq >= 0.01989454:
        z += -129.5509 * Q.e2_sq + 0.4712907
    if Q.tau1 < 0.05444509:
        z += -117.3145 * Q.tau1 + 6.095331
    if 0.05444509 <= Q.tau1 < 0.0705748:
        z += -12.61251 * Q.tau1 + 0.3948234
    if 0.0705748 <= Q.tau1 < 0.1008497:
        z += 16.36015 * Q.tau1 - 1.649917
    if Q.girth2 < 0.001784491:
        z += 19717.78 * Q.girth2 - 39.6397
    if 0.001784491 <= Q.girth2 < 0.008190222:
        z += 695.2374 * Q.girth2 - 5.694148
    if Q.girth2_top30 < 0.008376291:
        z += 444.7684 * Q.girth2_top30 - 4.340893
    if 0.008376291 <= Q.girth2_top30 < 0.01215787:
        z += 162.7318 * Q.girth2_top30 - 1.978472
    if Q.N2 < 0.3572263:
        z += 2.075435 * Q.N2 - 0.7414001
    if Q.soft7_z < 0.001595561:
        z += 67.01805 * Q.soft7_z - 0.1069314
    if Q.e2 < 0.03480688:
        z += -75.1913 * Q.e2 + 2.617174
    if Q.girth2_top20 < 0.006374178:
        z += 78.62768 * Q.girth2_top20 - 1.385856
    if 0.006374178 <= Q.girth2_top20 < 0.01083435:
        z += 198.3486 * Q.girth2_top20 - 2.148979
    if Q.log_sum_pt >= 7.062574:
        z += -4.379347 * Q.log_sum_pt + 30.92946
    if Q.sum_pt >= 1260.541:
        z += -0.02504633 * Q.sum_pt + 31.57193
    if Q.sum_pt_top50 >= 1245.697:
        z += 0.02552555 * Q.sum_pt_top50 - 31.7971
    if Q.girth2_top50 < 0.007820315:
        z += -96.37195 * Q.girth2_top50 + 0.5692447
    if 0.007820315 <= Q.girth2_top50 < 0.008124776:
        z += 605.7063 * Q.girth2_top50 - 4.921228
    if 0.1666442 <= Q.sj2_dr < 0.2412757:
        z += 2.964171 * Q.sj2_dr - 0.4939618
    if Q.sj2_dr >= 0.2412757:
        z += -1.256974 * Q.sj2_dr + 0.5244978
    if Q.girth2_top40 < 0.01292642:
        z += 147.6896 * Q.girth2_top40 - 1.909098
    if Q.z_2nd < 0.1361635:
        z += 6.828571 * Q.z_2nd - 0.9298025
    if Q.n_pt_above_5 < 28.0:
        z += 0.02740751 * Q.n_pt_above_5 - 0.7674102
    if Q.n_dr_0p1_0p2 < 21.0:
        z += 0.04544815 * Q.n_dr_0p1_0p2 - 0.9544111
    if Q.tau21_b2 < 0.342495 and Q.girth2_top40 < 0.007709916:
        z += 584.3956 * (0.342495 - Q.tau21_b2) * (0.007709916 - Q.girth2_top40)
    if Q.tau21_b2 < 0.342495 and Q.lam1 < 0.02048524:
        z += -155.5125 * (0.342495 - Q.tau21_b2) * (0.02048524 - Q.lam1)
    if Q.e2_sq < 0.01396296 and Q.z_top50_slots < 0.9906378:
        z += 2724.117 * (0.01396296 - Q.e2_sq) * (0.9906378 - Q.z_top50_slots)
    if Q.psi_0p3 > 0.9956185 and Q.sum_pt_top40 < 1225.842:
        z += -2.168266 * (Q.psi_0p3 - 0.9956185) * (1225.842 - Q.sum_pt_top40)
    if Q.girth2_top30 < 0.01215787 and Q.sum_pt > 907.9372:
        z += 1.278744 * (0.01215787 - Q.girth2_top30) * (Q.sum_pt - 907.9372)
    if Q.e2 < 0.03480688 and Q.sum_pt_top30 > 886.3438:
        z += -0.3778932 * (0.03480688 - Q.e2) * (Q.sum_pt_top30 - 886.3438)
    if Q.girth2 < 0.008190222 and Q.psi_0p3 > 0.9956185:
        z += -89138.04 * (0.008190222 - Q.girth2) * (Q.psi_0p3 - 0.9956185)
    if Q.N2 < 0.3572263 and Q.dr0_7 > 0.1786203:
        z += 23.48809 * (0.3572263 - Q.N2) * (Q.dr0_7 - 0.1786203)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.4839343
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -4.610252 * Q.z_dr_0p1_0p2 + 0.5548146
    if Q.psi_0p1 >= 0.8976117:
        z += -4.827565 * Q.psi_0p1 + 4.333279
    if Q.girth2_top5 < 0.002270363:
        z += 57.62312 * Q.girth2_top5 + 0.6030645
    if 0.002270363 <= Q.girth2_top5 < 0.008329695:
        z += -121.1173 * Q.girth2_top5 + 1.00887
    if Q.log_sum_pt < 6.97212:
        z += -6.582512 * Q.log_sum_pt + 45.89407
    if Q.log_sum_pt >= 7.139296:
        z += -16.53364 * Q.log_sum_pt + 118.0385
    if Q.sum_pt_top30 >= 988.4375:
        z += -0.001917436 * Q.sum_pt_top30 + 1.895265
    if 1003.544 <= Q.sum_pt_top50 < 1107.226:
        z += -0.002860485 * Q.sum_pt_top50 + 2.870623
    if 1107.226 <= Q.sum_pt_top50 < 1156.659:
        z += 0.003161433 * Q.sum_pt_top50 - 3.797
    if Q.sum_pt_top50 >= 1156.659:
        z += 0.02327539 * Q.sum_pt_top50 - 27.06201
    if Q.tau1 < 0.0705748:
        z += 24.36086 * Q.tau1 - 1.719263
    if Q.girth < 0.08068193:
        z += -34.96071 * Q.girth + 3.09583
    if 0.08068193 <= Q.girth < 0.1207452:
        z += -6.867438 * Q.girth + 0.82921
    if Q.D2 < 1.976207:
        z += -0.4880224 * Q.D2 + 0.9644335
    if Q.girth2_top30 < 0.005402331:
        z += 346.2058 * Q.girth2_top30 - 2.591365
    if 0.005402331 <= Q.girth2_top30 < 0.008376291:
        z += -12.58954 * Q.girth2_top30 - 0.6530339
    if 0.008376291 <= Q.girth2_top30 < 0.01215787:
        z += 200.5741 * Q.girth2_top30 - 2.438555
    if Q.z_dr_0p2_0p4 < 0.02647293:
        z += 15.23616 * Q.z_dr_0p2_0p4 - 0.4033459
    if 0.1745007 <= Q.sj2_dr < 0.2232169:
        z += -5.266106 * Q.sj2_dr + 0.9189392
    if 0.2232169 <= Q.sj2_dr < 0.2780918:
        z += 7.51394 * Q.sj2_dr - 1.933783
    if Q.sj2_dr >= 0.2780918:
        z += 1.603023 * Q.sj2_dr - 0.2900057
    if Q.e2 < 0.04082832:
        z += -44.26087 * Q.e2 + 1.807097
    if Q.z_top30_slots < 0.9860575:
        z += 3.75466 * Q.z_top30_slots - 3.702311
    if Q.lam1 < 0.006189818:
        z += -169.5315 * Q.lam1 + 1.049369
    if Q.LHA < 0.2941033:
        z += 12.84064 * Q.LHA - 3.776475
    if Q.n_pt_above_10 < 24.0:
        z += -0.03791625 * Q.n_pt_above_10 + 0.9099901
    if Q.girth2_top50 < 0.006805717:
        z += -98.0441 * Q.girth2_top50 + 0.6672604
    if Q.sum_pt_top40 >= 1095.686:
        z += -0.01039518 * Q.sum_pt_top40 + 11.38986
    if Q.sum_pt < 1002.379:
        z += 0.01455396 * Q.sum_pt - 14.58858
    if Q.girth2_top5 < 0.008329695 and Q.n_dr_0p2_0p4 > 10.0:
        z += -8.821661 * (0.008329695 - Q.girth2_top5) * (Q.n_dr_0p2_0p4 - 10.0)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.lam2 < 0.002396991:
        z += 1819.585 * (0.1203437 - Q.z_dr_0p1_0p2) * (0.002396991 - Q.lam2)
    if Q.sd_rg < 0.1778185 and Q.soft5_dr > 0.04139378:
        z += -23.78508 * (0.1778185 - Q.sd_rg) * (Q.soft5_dr - 0.04139378)
    if Q.sd_rg < 0.1778185 and Q.soft5_dr0 > 0.03721986:
        z += 23.27818 * (0.1778185 - Q.sd_rg) * (Q.soft5_dr0 - 0.03721986)
    if Q.D2 < 1.976207 and Q.sj3_z2 > 0.08173536:
        z += -0.9447067 * (1.976207 - Q.D2) * (Q.sj3_z2 - 0.08173536)
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
