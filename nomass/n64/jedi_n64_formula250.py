"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network (step 4; no mass observables), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.8% (the network: 81.1%); same class as the network for 93.2% of jets.

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
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_pt               pT [GeV] of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_z                pT share of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.sum_z_dr2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.sum_z_dr2_top50           pT-weighted mean ΔR² of the 50 hardest particles
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
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
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
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_11=pt[11],
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_6=pt[6],
        soft1_pt=softp(1, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft7_pt=softp(7, 'pt'),
        soft1_z=softp(1, 'z'),
        soft3_z=softp(3, 'z'),
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        sum_z_dr2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        sum_z_dr2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
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
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau4=tau_n(4),
    )


def neuron_0(Q):
    z = -0.772
    if Q.sum_zz_dr2 < 0.00618:
        z += -682.0 * Q.sum_zz_dr2 + 6.2692
    if 0.00618 <= Q.sum_zz_dr2 < 0.0069:
        z += -1397.0 * Q.sum_zz_dr2 + 10.6879
    if 0.0069 <= Q.sum_zz_dr2 < 0.00788:
        z += -1070.0 * Q.sum_zz_dr2 + 8.4316
    if Q.sum_z_dr2_top40 < 0.00332:
        z += 369.0 * Q.sum_z_dr2_top40 - 1.22508
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.043 * Q.n_dr_0p2_0p4 + 0.774
    if Q.psi_0p2 >= 0.909:
        z += -6.38 * Q.psi_0p2 + 5.79942
    if Q.psi_0p3 >= 0.994:
        z += 75.6 * Q.psi_0p3 - 75.1464
    if 1010.0 <= Q.sum_pt < 1120.0:
        z += -0.0125 * Q.sum_pt + 12.625
    if Q.sum_pt >= 1120.0:
        z += -0.00724 * Q.sum_pt + 6.7338
    if Q.sum_pt_top40 >= 860.0:
        z += 0.00246 * Q.sum_pt_top40 - 2.1156
    if Q.tau1 < 0.0719:
        z += 15.0 * Q.tau1 - 1.0785
    if Q.sum_z_dr2 < 0.00795 and Q.sum_z_dr2_top15 < 0.00596:
        z += -25100.0 * (0.00795 - Q.sum_z_dr2) * (0.00596 - Q.sum_z_dr2_top15)
    if Q.sum_z_dr2_top50 < 0.00585 and Q.sum_pt < 1260.0:
        z += -1.46 * (0.00585 - Q.sum_z_dr2_top50) * (1260.0 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 18.4 and Q.log_sum_pt > 6.93:
        z += -0.134 * (18.4 - Q.n_dr_0p2_0p4) * (Q.log_sum_pt - 6.93)
    return max(0.0, z)


def neuron_1(Q):
    z = 3.6
    if Q.LHA < 0.239:
        z += -10.2 * Q.LHA + 2.4378
    if Q.sum_zz_dr2 < 0.00962:
        z += 178.0 * Q.sum_zz_dr2 - 1.71236
    if Q.sum_z_dr2_top30 < 0.0048:
        z += -260.0 * Q.sum_z_dr2_top30 + 1.248
    if Q.lam2 < 0.000789:
        z += 934.0 * Q.lam2 - 0.736926
    if Q.log_sum_pt < 6.91:
        z += 5.78 * Q.log_sum_pt - 41.2114
    if 6.91 <= Q.log_sum_pt < 6.99:
        z += 36.98 * Q.log_sum_pt - 256.8034
    if 6.99 <= Q.log_sum_pt < 7.13:
        z += 10.58 * Q.log_sum_pt - 72.2674
    if Q.log_sum_pt >= 7.13:
        z += 4.8 * Q.log_sum_pt - 31.056
    if 15.2 <= Q.n_dr_0_0p05 < 30.0:
        z += 0.033 * Q.n_dr_0_0p05 - 0.5016
    if Q.n_dr_0_0p05 >= 30.0:
        z += -0.0592 * Q.n_dr_0_0p05 + 2.2644
    if Q.n_dr_0p1_0p2 < 7.04:
        z += 0.0995 * Q.n_dr_0p1_0p2 - 0.70048
    if Q.n_dr_0p2_0p4 < 6.75:
        z += 0.0851 * Q.n_dr_0p2_0p4 - 0.574425
    if Q.n_particles >= 37.9:
        z += 0.0662 * Q.n_particles - 2.50898
    if Q.psi_0p3 >= 0.997:
        z += -86.6 * Q.psi_0p3 + 86.3402
    if Q.pt_11 < 28.6:
        z += 0.021 * Q.pt_11 - 0.6006
    if Q.soft1_pt < 2.27:
        z += 1.25 * Q.soft1_pt - 2.8375
    if Q.soft1_z < 0.00138:
        z += -1110.0 * Q.soft1_z + 1.5318
    if 973.0 <= Q.sum_pt < 1050.0:
        z += 0.0278 * Q.sum_pt - 27.0494
    if Q.sum_pt >= 1050.0:
        z += 0.0156 * Q.sum_pt - 14.2394
    if Q.sum_pt_top2 < 593.0:
        z += -0.00186 * Q.sum_pt_top2 + 1.10298
    if Q.sum_pt_top50 < 940.0:
        z += -0.0146 * Q.sum_pt_top50 + 15.768
    if 940.0 <= Q.sum_pt_top50 < 1080.0:
        z += -0.0311 * Q.sum_pt_top50 + 31.278
    if Q.sum_pt_top50 >= 1080.0:
        z += -0.0165 * Q.sum_pt_top50 + 15.51
    if 0.0251 <= Q.tau1 < 0.128:
        z += -19.3 * Q.tau1 + 0.48443
    if Q.tau1 >= 0.128:
        z += -6.5 * Q.tau1 - 1.15397
    if Q.tau21 < 0.226:
        z += -6.48 * Q.tau21 + 1.46448
    if Q.z_top30_slots >= 0.957:
        z += 24.7 * Q.z_top30_slots - 23.6379
    if Q.M3 < 0.0339 and Q.psi_0p3 > 0.929:
        z += -307.0 * (0.0339 - Q.M3) * (Q.psi_0p3 - 0.929)
    if Q.n_particles > 37.3 and Q.zdr_0 < 0.01:
        z += 2.71 * (Q.n_particles - 37.3) * (0.01 - Q.zdr_0)
    if Q.z_dr_0_0p05 > 0.802 and Q.n_dr_0p05_0p1 < 10.4:
        z += -0.461 * (Q.z_dr_0_0p05 - 0.802) * (10.4 - Q.n_dr_0p05_0p1)
    if Q.z_top30_slots > 0.939 and Q.pt1_dr01 > -0.337:
        z += 0.447 * (Q.z_top30_slots - 0.939) * (Q.pt1_dr01 - -0.337)
    if Q.z_top30_slots > 0.958 and Q.ptdr0_3 < 10.4:
        z += -1.96 * (Q.z_top30_slots - 0.958) * (10.4 - Q.ptdr0_3)
    if Q.z_top5 > 0.656 and Q.pt1_dr01 < 27.7:
        z += -0.203 * (Q.z_top5 - 0.656) * (27.7 - Q.pt1_dr01)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.463
    if Q.LHA >= 0.371:
        z += -163.0 * Q.LHA + 60.473
    if Q.sum_z_dr2 < 0.00695:
        z += 172.0 * Q.sum_z_dr2 - 1.1954
    if Q.lam1 < 0.0118:
        z += -114.0 * Q.lam1 + 1.3452
    if Q.log_sum_pt >= 6.91:
        z += 5.61 * Q.log_sum_pt - 38.7651
    if Q.log_sum_pt > 7.06 and Q.sum_z_dr2_top30 > 0.0119:
        z += -30400.0 * (Q.log_sum_pt - 7.06) * (Q.sum_z_dr2_top30 - 0.0119)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.103
    if Q.n_dr_0_0p05 < 24.5:
        z += 0.0291 * Q.n_dr_0_0p05 - 0.71295
    if Q.n_dr_0p2_0p4 < 5.25:
        z += -0.274 * Q.n_dr_0p2_0p4 + 1.4385
    if Q.n_particles < 45.5:
        z += -0.0664 * Q.n_particles + 3.0212
    if Q.tau21_b2 < 0.21:
        z += -2.6 * Q.tau21_b2 + 0.546
    if Q.n_dr_0p2_0p4 < 4.97 and Q.e2 < 0.034:
        z += -10.3 * (4.97 - Q.n_dr_0p2_0p4) * (0.034 - Q.e2)
    if Q.n_dr_0p2_0p4 < 4.92 and Q.sum_pt_top30 < 996.0:
        z += -0.00165 * (4.92 - Q.n_dr_0p2_0p4) * (996.0 - Q.sum_pt_top30)
    if Q.n_particles < 44.8 and Q.e2 > 0.0146:
        z += -2.32 * (44.8 - Q.n_particles) * (Q.e2 - 0.0146)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.838
    if Q.LHA >= 0.37:
        z += -14.3 * Q.LHA + 5.291
    if Q.sum_zz_dr2 < 0.00604:
        z += 390.0 * Q.sum_zz_dr2 - 2.3556
    if Q.sum_zz_dr2 >= 0.0253:
        z += 87.6 * Q.sum_zz_dr2 - 2.21628
    if Q.sum_z_dr2_top20 < 0.00779:
        z += 135.0 * Q.sum_z_dr2_top20 - 0.62797
    if 0.00779 <= Q.sum_z_dr2_top20 < 0.0111:
        z += -128.0 * Q.sum_z_dr2_top20 + 1.4208
    if Q.sum_z_dr2_top40 < 0.00798:
        z += 362.0 * Q.sum_z_dr2_top40 - 2.88876
    if Q.sum_z_dr2_top50 >= 0.00463:
        z += 69.3 * Q.sum_z_dr2_top50 - 0.320859
    if Q.lam1 >= 0.0208:
        z += -77.1 * Q.lam1 + 1.60368
    if Q.n_dr_0_0p05 >= 11.8:
        z += 0.0283 * Q.n_dr_0_0p05 - 0.33394
    if Q.n_particles >= 28.6:
        z += -0.0319 * Q.n_particles + 0.91234
    if Q.psi_0p1 >= 0.391:
        z += -0.987 * Q.psi_0p1 + 0.385917
    if Q.sum_pt_top40 < 1050.0:
        z += -0.00631 * Q.sum_pt_top40 + 6.6255
    if Q.sum_pt_top50 < 1060.0:
        z += 0.00997 * Q.sum_pt_top50 - 10.5682
    if Q.lam1_plus_lam2 < 0.00988:
        z += -316.0 * Q.lam1_plus_lam2 + 3.12208
    if Q.sum_z_dr2 < 0.0146 and Q.psi_0p3 > 0.978:
        z += 1970.0 * (0.0146 - Q.sum_z_dr2) * (Q.psi_0p3 - 0.978)
    if Q.n_particles > 29.7 and Q.soft1_pt < 2.35:
        z += 0.00881 * (Q.n_particles - 29.7) * (2.35 - Q.soft1_pt)
    if Q.psi_0p3 > 0.997 and Q.sum_pt_top40 > 870.0:
        z += 0.612 * (Q.psi_0p3 - 0.997) * (Q.sum_pt_top40 - 870.0)
    if Q.sj2_dr > 0.256 and Q.C2_b2 < 0.0336:
        z += -276.0 * (Q.sj2_dr - 0.256) * (0.0336 - Q.C2_b2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.127
    if Q.N2 < 0.377:
        z += 2.95 * Q.N2 - 1.11215
    if Q.sum_zz_dr2 >= 0.00797:
        z += -346.0 * Q.sum_zz_dr2 + 2.75762
    if Q.sum_z_dr >= 0.158:
        z += -355.0 * Q.sum_z_dr + 56.09
    if Q.sum_z_dr2_top20 < 0.00951:
        z += 98.7 * Q.sum_z_dr2_top20 - 0.938637
    if Q.log_sum_pt >= 6.91:
        z += -25.1 * Q.log_sum_pt + 173.441
    if Q.n_dr_0p2_0p4 < 7.83:
        z += -0.087 * Q.n_dr_0p2_0p4 + 0.68121
    if Q.n_particles >= 45.3:
        z += -0.0333 * Q.n_particles + 1.50849
    if Q.n_pt_above_1 >= 23.2:
        z += 0.0251 * Q.n_pt_above_1 - 0.58232
    if Q.n_real_top50 < 46.6:
        z += -0.0381 * Q.n_real_top50 + 1.77546
    if Q.sum_pt_top50 >= 926.0:
        z += 0.0229 * Q.sum_pt_top50 - 21.2054
    if Q.lam1_plus_lam2 >= 0.00702:
        z += 268.0 * Q.lam1_plus_lam2 - 1.88136
    if Q.sum_z_dr2 > 0.0293 and Q.soft3_z < 0.00263:
        z += -722000.0 * (Q.sum_z_dr2 - 0.0293) * (0.00263 - Q.soft3_z)
    if Q.n_particles < 73.5 and Q.D2 < 2.44:
        z += -0.00864 * (73.5 - Q.n_particles) * (2.44 - Q.D2)
    if Q.sum_pt_top50 > 1010.0 and Q.e4 < 6.64e-08:
        z += -68900.0 * (Q.sum_pt_top50 - 1010.0) * (6.64e-08 - Q.e4)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.274
    if Q.LHA < 0.24:
        z += -5.7 * Q.LHA + 1.368
    if Q.e2 >= 0.0125:
        z += -40.1 * Q.e2 + 0.50125
    if Q.sum_zz_dr2 < 0.0271:
        z += -110.0 * Q.sum_zz_dr2 + 2.981
    if Q.sum_z_dr2 < 0.0135:
        z += 243.0 * Q.sum_z_dr2 - 3.2805
    if Q.lam1 < 0.00389:
        z += -1050.0 * Q.lam1 + 4.0845
    if Q.lam2 >= 0.000746:
        z += -168.0 * Q.lam2 + 0.125328
    if Q.log_sum_pt >= 6.81:
        z += 4.0 * Q.log_sum_pt - 27.24
    if Q.psi_0p3 >= 0.986:
        z += 92.1 * Q.psi_0p3 - 90.8106
    if Q.lam1_plus_lam2 < 0.0101:
        z += 296.0 * Q.lam1_plus_lam2 - 2.9896
    if Q.e2 > 0.0476 and Q.z_dr_0_0p05 < 0.459:
        z += 236.0 * (Q.e2 - 0.0476) * (0.459 - Q.z_dr_0_0p05)
    if Q.sum_zz_dr2 < 0.0242 and Q.log_sum_pt < 7.0:
        z += -226.0 * (0.0242 - Q.sum_zz_dr2) * (7.0 - Q.log_sum_pt)
    if Q.sum_zz_dr2 < 0.0105 and Q.sum_pt < 1010.0:
        z += 1.89 * (0.0105 - Q.sum_zz_dr2) * (1010.0 - Q.sum_pt)
    if Q.sum_z_dr2 < 0.0141 and Q.psi_0p3 > 0.985:
        z += -8300.0 * (0.0141 - Q.sum_z_dr2) * (Q.psi_0p3 - 0.985)
    if Q.log_sum_pt > 6.8 and Q.sum_z_dr2_top3 < 0.00186:
        z += -1010.0 * (Q.log_sum_pt - 6.8) * (0.00186 - Q.sum_z_dr2_top3)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.861
    if Q.LHA < 0.245:
        z += 17.8 * Q.LHA - 3.3054
    if 0.245 <= Q.LHA < 0.303:
        z += -18.2 * Q.LHA + 5.5146
    if Q.sum_zz_dr2 < 0.00973:
        z += -687.0 * Q.sum_zz_dr2 + 6.68451
    if Q.sum_z_dr < 0.0565:
        z += -56.9 * Q.sum_z_dr + 1.33672
    if 0.0565 <= Q.sum_z_dr < 0.0858:
        z += 64.1 * Q.sum_z_dr - 5.49978
    if Q.sum_z_dr2 < 0.00796:
        z += 810.0 * Q.sum_z_dr2 - 6.4476
    if Q.sum_z_dr2_top50 < 0.00637:
        z += 967.0 * Q.sum_z_dr2_top50 - 6.15979
    if Q.lam1 < 0.00587:
        z += 486.0 * Q.lam1 - 2.85282
    if Q.max_dr >= 0.228:
        z += -1.4 * Q.max_dr + 0.3192
    if Q.mean_eta2 < 0.000995:
        z += 460.0 * Q.mean_eta2 - 0.4577
    if Q.psi_0p2 >= 0.907:
        z += -5.82 * Q.psi_0p2 + 5.27874
    if Q.psi_0p3 >= 0.997:
        z += 272.0 * Q.psi_0p3 - 271.184
    if Q.tau2 < 0.0483:
        z += 14.8 * Q.tau2 - 0.71484
    if Q.tau21_b2 < 0.237:
        z += -5.75 * Q.tau21_b2 + 1.36275
    if Q.lam1_plus_lam2 < 0.0139:
        z += -485.0 * Q.lam1_plus_lam2 + 6.7415
    if Q.sum_zz_dr2 < 0.00963 and Q.sum_pt < 1170.0:
        z += -6.44 * (0.00963 - Q.sum_zz_dr2) * (1170.0 - Q.sum_pt)
    if Q.sum_z_dr < 0.0864 and Q.sum_pt_top50 < 1150.0:
        z += 0.117 * (0.0864 - Q.sum_z_dr) * (1150.0 - Q.sum_pt_top50)
    if Q.lam1 < 0.00586 and Q.sum_pt < 1160.0:
        z += 3.4 * (0.00586 - Q.lam1) * (1160.0 - Q.sum_pt)
    if Q.lam2 < 0.00243 and Q.log_sum_pt < 7.13:
        z += 1250.0 * (0.00243 - Q.lam2) * (7.13 - Q.log_sum_pt)
    if Q.lam2 < 0.00252 and Q.zdr_0 > 0.00222:
        z += -10800.0 * (0.00252 - Q.lam2) * (Q.zdr_0 - 0.00222)
    if Q.psi_0p3 > 0.998 and Q.dr_0 < 0.0363:
        z += -12000.0 * (Q.psi_0p3 - 0.998) * (0.0363 - Q.dr_0)
    if Q.sum_pt_top15 > 1080.0 and Q.mean_phi > 8.35e-05:
        z += -31.1 * (Q.sum_pt_top15 - 1080.0) * (Q.mean_phi - 8.35e-05)
    if Q.tau21_b2 < 0.234 and Q.sum_zz_dr2 < 0.00808:
        z += -3220.0 * (0.234 - Q.tau21_b2) * (0.00808 - Q.sum_zz_dr2)
    if Q.tau21_b2 < 0.236 and Q.sj2_dr > 0.188:
        z += -36.0 * (0.236 - Q.tau21_b2) * (Q.sj2_dr - 0.188)
    if Q.tau21_b2 < 0.229 and Q.sum_pt_top30 > 978.0:
        z += 0.0323 * (0.229 - Q.tau21_b2) * (Q.sum_pt_top30 - 978.0)
    return max(0.0, z)


def neuron_8(Q):
    z = 0.996
    if Q.sum_zz_dr2 < 0.0102:
        z += 521.0 * Q.sum_zz_dr2 - 5.3142
    if Q.sum_z_dr2 < 0.00379:
        z += -856.0 * Q.sum_z_dr2 + 8.16265
    if 0.00379 <= Q.sum_z_dr2 < 0.0271:
        z += -211.0 * Q.sum_z_dr2 + 5.7181
    if Q.sum_z_dr2_top30 < 0.00727:
        z += 581.0 * Q.sum_z_dr2_top30 - 4.22387
    if Q.n_dr_0p2_0p4 < 18.2:
        z += 0.0724 * Q.n_dr_0p2_0p4 - 1.31768
    if Q.sum_pt < 1030.0:
        z += -0.0141 * Q.sum_pt + 14.523
    if Q.sum_pt_top20 >= 1060.0:
        z += 0.00246 * Q.sum_pt_top20 - 2.6076
    if Q.sum_pt_top50 < 1220.0:
        z += -0.0121 * Q.sum_pt_top50 + 14.762
    if Q.tau1 < 0.12:
        z += -30.4 * Q.tau1 + 3.648
    if Q.lam1_plus_lam2 < 0.00814:
        z += 494.0 * Q.lam1_plus_lam2 - 4.02116
    if Q.sum_z_dr2 < 0.0263 and Q.log_sum_pt < 7.14:
        z += -1580.0 * (0.0263 - Q.sum_z_dr2) * (7.14 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.0136 and Q.sum_pt < 995.0:
        z += 2.58 * (0.0136 - Q.sum_z_dr2) * (995.0 - Q.sum_pt)
    if Q.sum_z_dr2_top30 < 0.00748 and Q.sum_pt < 1280.0:
        z += 3.42 * (0.00748 - Q.sum_z_dr2_top30) * (1280.0 - Q.sum_pt)
    if Q.sum_z_dr2_top30 < 0.00735 and Q.sum_pt_top40 < 1050.0:
        z += -2.02 * (0.00735 - Q.sum_z_dr2_top30) * (1050.0 - Q.sum_pt_top40)
    if Q.n_dr_0p2_0p4 < 17.4 and Q.sj2_zsoft > 0.0986:
        z += 0.106 * (17.4 - Q.n_dr_0p2_0p4) * (Q.sj2_zsoft - 0.0986)
    if Q.tau1 < 0.137 and Q.planar_flow < 0.544:
        z += -21.3 * (0.137 - Q.tau1) * (0.544 - Q.planar_flow)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.09
    if 0.0199 <= Q.sum_zz_dr2 < 0.0292:
        z += -193.0 * Q.sum_zz_dr2 + 3.8407
    if Q.sum_zz_dr2 >= 0.0292:
        z += 374.0 * Q.sum_zz_dr2 - 12.7157
    if Q.e3 >= 0.000524:
        z += 5130.0 * Q.e3 - 2.68812
    if Q.sum_z_dr2 >= 0.0258:
        z += -642.0 * Q.sum_z_dr2 + 16.5636
    if Q.lam1 >= 0.00749:
        z += 125.0 * Q.lam1 - 0.93625
    if Q.sum_pt_top30 < 1200.0:
        z += -0.00428 * Q.sum_pt_top30 + 5.136
    if Q.sum_pt_top40 < 956.0:
        z += -0.012 * Q.sum_pt_top40 + 11.472
    if Q.lam1_plus_lam2 < 0.00673:
        z += -751.0 * Q.lam1_plus_lam2 + 5.05423
    if Q.z_dr_0_0p05 >= 0.857:
        z += -6.44 * Q.z_dr_0_0p05 + 5.51908
    if Q.LHA > 0.332 and Q.sum_pt < 1070.0:
        z += 0.0834 * (Q.LHA - 0.332) * (1070.0 - Q.sum_pt)
    if Q.sum_zz_dr2 > 0.0293 and Q.soft7_pt < 3.79:
        z += -232.0 * (Q.sum_zz_dr2 - 0.0293) * (3.79 - Q.soft7_pt)
    if Q.sum_z_dr2_top40 < 0.00546 and Q.sum_pt < 1010.0:
        z += -2.55 * (0.00546 - Q.sum_z_dr2_top40) * (1010.0 - Q.sum_pt)
    if Q.lam1 < 0.0107 and Q.sum_pt > 1070.0:
        z += -0.418 * (0.0107 - Q.lam1) * (Q.sum_pt - 1070.0)
    if Q.log_sum_pt < 6.9 and Q.z_dr_0p1_0p2 < 0.527:
        z += 13.9 * (6.9 - Q.log_sum_pt) * (0.527 - Q.z_dr_0p1_0p2)
    if Q.sum_pt_top40 < 983.0 and Q.soft4_pt > 0.907:
        z += -0.00347 * (983.0 - Q.sum_pt_top40) * (Q.soft4_pt - 0.907)
    if Q.sum_pt_top40 < 959.0 and Q.tau3 < 0.0369:
        z += -0.714 * (959.0 - Q.sum_pt_top40) * (0.0369 - Q.tau3)
    if Q.z_dr_0_0p05 > 0.88 and Q.sum_pt_top40 > 1000.0:
        z += 0.0364 * (Q.z_dr_0_0p05 - 0.88) * (Q.sum_pt_top40 - 1000.0)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.2
    if Q.M2 < 0.0854:
        z += -16.1 * Q.M2 + 1.37494
    if Q.dr_0 < 0.0671:
        z += -6.98 * Q.dr_0 + 0.468358
    if Q.e2 >= 0.00532:
        z += 58.8 * Q.e2 - 0.312816
    if Q.sum_zz_dr2 >= 0.0259:
        z += -199.0 * Q.sum_zz_dr2 + 5.1541
    if Q.sum_z_dr < 0.0372:
        z += 94.5 * Q.sum_z_dr - 4.70456
    if 0.0372 <= Q.sum_z_dr < 0.124:
        z += 13.7 * Q.sum_z_dr - 1.6988
    if Q.sum_z_dr2 >= 0.0137:
        z += -157.0 * Q.sum_z_dr2 + 2.1509
    if Q.sum_z_dr2_top10 < 0.0101:
        z += 61.1 * Q.sum_z_dr2_top10 - 0.61711
    if Q.sum_pt_top3 >= 288.0:
        z += -0.00319 * Q.sum_pt_top3 + 0.91872
    if Q.tau21_b2 < 0.314:
        z += 2.37 * Q.tau21_b2 - 0.74418
    if Q.tau4 < 0.026:
        z += 32.0 * Q.tau4 - 0.832
    if Q.lam1_plus_lam2 < 0.00641:
        z += -534.0 * Q.lam1_plus_lam2 + 3.42294
    if Q.z_dr_0p2_0p4 < 0.0975:
        z += -3.14 * Q.z_dr_0p2_0p4 + 0.30615
    if Q.M2 < 0.0888 and Q.max_dr < 0.423:
        z += -55.4 * (0.0888 - Q.M2) * (0.423 - Q.max_dr)
    if Q.e2 > 0.00673 and Q.log_sum_pt > 6.9:
        z += -116.0 * (Q.e2 - 0.00673) * (Q.log_sum_pt - 6.9)
    if Q.e2 > 0.00434 and Q.sj3_pairmin_over_m > 0.161:
        z += 25.4 * (Q.e2 - 0.00434) * (Q.sj3_pairmin_over_m - 0.161)
    if Q.sum_z_dr2 > 0.0144 and Q.pt_3 < 113.0:
        z += 0.875 * (Q.sum_z_dr2 - 0.0144) * (113.0 - Q.pt_3)
    if Q.sum_z_dr2_top30 < 0.00529 and Q.pt_6 > 26.4:
        z += -5.89 * (0.00529 - Q.sum_z_dr2_top30) * (Q.pt_6 - 26.4)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.454
    if Q.C2_b2 >= 0.013:
        z += -8.56 * Q.C2_b2 + 0.11128
    if Q.e2 < 0.0241:
        z += -63.9 * Q.e2 + 1.53999
    if Q.sum_zz_dr2 < 0.00458:
        z += -738.0 * Q.sum_zz_dr2 + 5.4161
    if 0.00458 <= Q.sum_zz_dr2 < 0.0084:
        z += -533.0 * Q.sum_zz_dr2 + 4.4772
    if Q.sum_z_dr2_top30 < 0.00638:
        z += 427.0 * Q.sum_z_dr2_top30 - 1.9221
    if 0.00638 <= Q.sum_z_dr2_top30 < 0.0118:
        z += -148.0 * Q.sum_z_dr2_top30 + 1.7464
    if Q.lam1 < 0.00606:
        z += 416.0 * Q.lam1 - 2.52096
    if Q.n_dr_0p2_0p4 < 9.02:
        z += -0.0661 * Q.n_dr_0p2_0p4 + 0.596222
    if Q.sum_zz_dr2 < 0.0084 and Q.sum_pt < 1120.0:
        z += -3.48 * (0.0084 - Q.sum_zz_dr2) * (1120.0 - Q.sum_pt)
    if Q.lam1 < 0.00615 and Q.sum_pt < 1120.0:
        z += 3.91 * (0.00615 - Q.lam1) * (1120.0 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 9.75 and Q.sum_z_dr2 < 0.00618:
        z += -29.6 * (9.75 - Q.n_dr_0p2_0p4) * (0.00618 - Q.sum_z_dr2)
    if Q.n_dr_0p2_0p4 < 10.4 and Q.n_dr_0p1_0p2 < 21.9:
        z += 0.00552 * (10.4 - Q.n_dr_0p2_0p4) * (21.9 - Q.n_dr_0p1_0p2)
    if Q.z_top30_slots > 0.968 and Q.D2_b2 < 1.11:
        z += 22.2 * (Q.z_top30_slots - 0.968) * (1.11 - Q.D2_b2)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.846
    if Q.sum_zz_dr2 < 0.00768:
        z += -644.0 * Q.sum_zz_dr2 + 4.94592
    if Q.sum_z_dr2_top30 < 0.00564:
        z += 242.0 * Q.sum_z_dr2_top30 - 1.36488
    if Q.lam1 < 0.00272:
        z += 1060.0 * Q.lam1 - 2.8832
    if Q.log_sum_pt < 6.87:
        z += -4.36 * Q.log_sum_pt + 29.9532
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.052 * Q.n_dr_0p2_0p4 + 0.52
    if 0.989 <= Q.psi_0p3 < 0.994:
        z += 88.3 * Q.psi_0p3 - 87.3287
    if Q.psi_0p3 >= 0.994:
        z += -96.7 * Q.psi_0p3 + 96.5613
    if Q.sum_pt >= 1080.0:
        z += -0.0109 * Q.sum_pt + 11.772
    if Q.sum_zz_dr2 < 0.00754 and Q.log_sum_pt > 6.81:
        z += -6760.0 * (0.00754 - Q.sum_zz_dr2) * (Q.log_sum_pt - 6.81)
    if Q.sum_z_dr2 < 0.00637 and Q.log_sum_pt > 6.82:
        z += 10200.0 * (0.00637 - Q.sum_z_dr2) * (Q.log_sum_pt - 6.82)
    if Q.psi_0p1 < 0.0784 and Q.sum_pt_top50 > 948.0:
        z += 0.187 * (0.0784 - Q.psi_0p1) * (Q.sum_pt_top50 - 948.0)
    if Q.psi_0p3 > 0.994 and Q.sum_z_dr2 < 0.00767:
        z += 40000.0 * (Q.psi_0p3 - 0.994) * (0.00767 - Q.sum_z_dr2)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.27
    if Q.e2 < 0.0355:
        z += -40.2 * Q.e2 + 1.4271
    if Q.sum_zz_dr2 >= 0.0292:
        z += -1060.0 * Q.sum_zz_dr2 + 30.952
    if Q.sum_z_dr < 0.124:
        z += 22.9 * Q.sum_z_dr - 2.8396
    if Q.sum_z_dr2_top50 >= 0.0132:
        z += -466.0 * Q.sum_z_dr2_top50 + 6.1512
    if Q.log_sum_pt >= 6.81:
        z += 21.2 * Q.log_sum_pt - 144.372
    if Q.n_particles < 54.0:
        z += 0.033 * Q.n_particles - 1.782
    if Q.psi_0p3 >= 0.998:
        z += 202.0 * Q.psi_0p3 - 201.596
    if Q.sum_pt < 1010.0:
        z += 0.0112 * Q.sum_pt - 12.096
    if 1010.0 <= Q.sum_pt < 1080.0:
        z += -0.0099 * Q.sum_pt + 9.215
    if Q.sum_pt >= 1080.0:
        z += -0.0211 * Q.sum_pt + 21.311
    if Q.sum_pt_top10 >= 881.0:
        z += 0.00384 * Q.sum_pt_top10 - 3.38304
    if Q.sum_pt_top15 < 800.0:
        z += 0.0038 * Q.sum_pt_top15 - 3.04
    if Q.sum_pt_top50 < 1070.0:
        z += -0.00832 * Q.sum_pt_top50 + 8.9024
    if Q.sum_z_dr2_top40 > 0.0282 and Q.pt_6 > 20.4:
        z += -50.7 * (Q.sum_z_dr2_top40 - 0.0282) * (Q.pt_6 - 20.4)
    if Q.sum_z_dr2_top40 > 0.0125 and Q.sum_pt_top50 < 1190.0:
        z += 1.18 * (Q.sum_z_dr2_top40 - 0.0125) * (1190.0 - Q.sum_pt_top50)
    if Q.log_sum_pt > 6.85 and Q.C2 > 0.0674:
        z += -41.2 * (Q.log_sum_pt - 6.85) * (Q.C2 - 0.0674)
    if Q.sum_pt < 1130.0 and Q.tau21 > 0.184:
        z += 0.0057 * (1130.0 - Q.sum_pt) * (Q.tau21 - 0.184)
    return max(0.0, z)


def neuron_14(Q):
    z = -1.71
    if Q.e2 < 0.0345:
        z += -62.6 * Q.e2 + 2.1597
    if Q.sum_zz_dr2 < 0.00367:
        z += 209.0 * Q.sum_zz_dr2 + 3.76805
    if 0.00367 <= Q.sum_zz_dr2 < 0.00605:
        z += 178.5 * Q.sum_zz_dr2 + 3.879985
    if 0.00605 <= Q.sum_zz_dr2 < 0.0198:
        z += -396.5 * Q.sum_zz_dr2 + 7.358735
    if Q.sum_zz_dr2 >= 0.0198:
        z += -30.5 * Q.sum_zz_dr2 + 0.111935
    if Q.sum_z_dr2 < 0.00181:
        z += 35254.0 * Q.sum_z_dr2 - 67.44952
    if 0.00181 <= Q.sum_z_dr2 < 0.00838:
        z += 554.0 * Q.sum_z_dr2 - 4.64252
    if Q.sum_z_dr2_top30 < 0.012:
        z += 551.0 * Q.sum_z_dr2_top30 - 6.612
    if Q.n_dr_0p1_0p2 < 21.0:
        z += 0.0371 * Q.n_dr_0p1_0p2 - 0.7791
    if Q.n_pt_above_5 < 30.7:
        z += 0.0299 * Q.n_pt_above_5 - 0.91793
    if 0.989 <= Q.psi_0p3 < 0.996:
        z += 132.0 * Q.psi_0p3 - 130.548
    if Q.psi_0p3 >= 0.996:
        z += 613.0 * Q.psi_0p3 - 609.624
    if Q.sum_pt >= 1250.0:
        z += -0.0296 * Q.sum_pt + 37.0
    if Q.sum_pt_top50 >= 1250.0:
        z += 0.0277 * Q.sum_pt_top50 - 34.625
    if Q.tau1 < 0.0574:
        z += -79.6 * Q.tau1 + 4.56904
    if Q.z_2nd < 0.138:
        z += 5.85 * Q.z_2nd - 0.8073
    if Q.e2 < 0.0357 and Q.sum_pt_top30 > 881.0:
        z += -0.318 * (0.0357 - Q.e2) * (Q.sum_pt_top30 - 881.0)
    if Q.sum_zz_dr2 < 0.0142 and Q.z_top50_slots < 0.992:
        z += 2620.0 * (0.0142 - Q.sum_zz_dr2) * (0.992 - Q.z_top50_slots)
    if Q.sum_z_dr2 < 0.00837 and Q.psi_0p3 > 0.995:
        z += -84500.0 * (0.00837 - Q.sum_z_dr2) * (Q.psi_0p3 - 0.995)
    if Q.sum_z_dr2_top30 < 0.0123 and Q.sum_pt > 904.0:
        z += 1.17 * (0.0123 - Q.sum_z_dr2_top30) * (Q.sum_pt - 904.0)
    if Q.psi_0p3 > 0.996 and Q.sum_pt_top40 < 1220.0:
        z += -2.26 * (Q.psi_0p3 - 0.996) * (1220.0 - Q.sum_pt_top40)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.51
    if Q.LHA < 0.289:
        z += 23.8 * Q.LHA - 6.8782
    if Q.sum_z_dr < 0.0795:
        z += -72.7 * Q.sum_z_dr + 5.77965
    if Q.sum_z_dr2_top30 < 0.00408:
        z += 339.2 * Q.sum_z_dr2_top30 - 2.08812
    if 0.00408 <= Q.sum_z_dr2_top30 < 0.0137:
        z += 73.2 * Q.sum_z_dr2_top30 - 1.00284
    if Q.sum_z_dr2_top5 < 0.00762:
        z += -108.0 * Q.sum_z_dr2_top5 + 0.82296
    if Q.log_sum_pt < 6.98:
        z += -9.73 * Q.log_sum_pt + 67.9154
    if Q.n_pt_above_10 < 20.8:
        z += -0.0343 * Q.n_pt_above_10 + 0.71344
    if Q.sum_pt < 1000.0:
        z += 0.0176 * Q.sum_pt - 17.6
    if Q.z_dr_0p1_0p2 < 0.126:
        z += -6.38 * Q.z_dr_0p1_0p2 + 0.80388
    if Q.sum_z_dr2_top5 < 0.00522 and Q.n_dr_0p2_0p4 > 10.2:
        z += -12.4 * (0.00522 - Q.sum_z_dr2_top5) * (Q.n_dr_0p2_0p4 - 10.2)
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
