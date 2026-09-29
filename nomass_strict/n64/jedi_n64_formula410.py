"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.4% (the network: 81.1%); same class as the network for 92.1% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
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
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.ptdr0_10               pT10 · ΔR(0, 10) [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.z_3                    pT of particle 3 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_12                 ΔR between particle 12 and the hardest particle
  Q.dr1_11                 ΔR between particle 11 and the 2nd-hardest particle
  Q.dr1_13                 ΔR between particle 13 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
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
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
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
  Q.tau2                   N-subjettiness τ2 (β=1)
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
        z_top5=sum(zs[:5]),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
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
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        ptdr0_10=pt[10] * math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        pt_11=pt[11],
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        z_3=z[3],
        soft1_z=softp(1, 'z'),
        soft5_z=softp(5, 'z'),
        soft7_z=softp(7, 'z'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        sd_zg=softdrop("zg"),
        sj2_dr=subjets(2)["dr"][0],
        dr0_12=math.sqrt(dist2(0, 12)) if pt[12] > 0 else 0.0,
        dr1_11=math.sqrt(dist2(1, 11)) if pt[11] > 0 else 0.0,
        dr1_13=math.sqrt(dist2(1, 13)) if pt[13] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
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
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
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
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
    )


def neuron_0(Q):
    z = -1.42614
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.02520674 * Q.n_dr_0p2_0p4 + 0.4537214
    if Q.sum_z_dr2_top15 < 0.005383629:
        z += -14.07825 * Q.sum_z_dr2_top15 + 0.3390025
    if 0.005383629 <= Q.sum_z_dr2_top15 < 0.007887677:
        z += -350.9922 * Q.sum_z_dr2_top15 + 2.152823
    if 0.007887677 <= Q.sum_z_dr2_top15 < 0.009962397:
        z += 296.7587 * Q.sum_z_dr2_top15 - 2.956428
    if Q.tau1 < 0.04466492:
        z += 108.6251 * Q.tau1 - 4.85173
    if Q.sum_pt_top40 >= 1001.523:
        z += -0.005754026 * Q.sum_pt_top40 + 5.76279
    if Q.sum_z_dr < 0.05048381:
        z += -159.1817 * Q.sum_z_dr + 13.08412
    if 0.05048381 <= Q.sum_z_dr < 0.08068193:
        z += -185.8608 * Q.sum_z_dr + 14.43098
    if 0.08068193 <= Q.sum_z_dr < 0.08589404:
        z += 19.56569 * Q.sum_z_dr - 2.143228
    if 0.08589404 <= Q.sum_z_dr < 0.09749958:
        z += 39.86478 * Q.sum_z_dr - 3.8868
    if Q.sj3_dr_max < 0.1623049:
        z += 3.076251 * Q.sj3_dr_max - 0.2931823
    if 0.1623049 <= Q.sj3_dr_max < 0.2125209:
        z += -4.104428 * Q.sj3_dr_max + 0.8722769
    if Q.log_sum_pt >= 7.017258:
        z += 3.449986 * Q.log_sum_pt - 24.20944
    if 0.1411617 <= Q.sj2_dr < 0.1512157:
        z += 18.53474 * Q.sj2_dr - 2.616395
    if 0.1512157 <= Q.sj2_dr < 0.1745007:
        z += -9.982538 * Q.sj2_dr + 1.695864
    if Q.sj2_dr >= 0.1745007:
        z += -1.82618 * Q.sj2_dr + 0.2725735
    if Q.z_dr_0_0p05 < 0.4947602:
        z += -0.8512626 * Q.z_dr_0_0p05 + 0.4211709
    if Q.dr_0 < 0.05775119:
        z += -0.348917 * Q.dr_0 - 0.1752996
    if 0.05775119 <= Q.dr_0 < 0.06413297:
        z += 30.62626 * Q.dr_0 - 1.964153
    if Q.LHA < 0.1870291:
        z += 23.05936 * Q.LHA - 9.037776
    if 0.1870291 <= Q.LHA < 0.3098384:
        z += 42.6936 * Q.LHA - 12.70995
    if 0.3098384 <= Q.LHA < 0.3203321:
        z += -31.64518 * Q.LHA + 10.32306
    if 0.3203321 <= Q.LHA < 0.3332345:
        z += -14.42273 * Q.LHA + 4.806152
    if Q.e3 < 2.883342e-05:
        z += 17495.96 * Q.e3 - 0.5044683
    if Q.e2 < 0.01879315:
        z += -9.62739 * Q.e2 - 0.01478056
    if 0.01879315 <= Q.e2 < 0.03029714:
        z += 17.0123 * Q.e2 - 0.5154242
    if Q.psi_0p3 >= 0.9896594:
        z += 50.0564 * Q.psi_0p3 - 49.53879
    if Q.n_dr_0p2_0p4 < 18.0 and Q.log_sum_pt > 6.920349:
        z += -0.9656978 * (18.0 - Q.n_dr_0p2_0p4) * (Q.log_sum_pt - 6.920349)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.sum_pt_top50 > 889.8503:
        z += 0.0005576365 * (18.0 - Q.n_dr_0p2_0p4) * (Q.sum_pt_top50 - 889.8503)
    if Q.sum_pt_top40 > 1001.523 and Q.sj2_dr < 0.1512157:
        z += 0.04243572 * (Q.sum_pt_top40 - 1001.523) * (0.1512157 - Q.sj2_dr)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.071039
    if Q.n_particles >= 38.0:
        z += 0.06483201 * Q.n_particles - 2.463616
    if Q.log_sum_pt < 6.910131:
        z += 9.836518 * Q.log_sum_pt - 70.22582
    if 6.910131 <= Q.log_sum_pt < 6.98945:
        z += 43.06951 * Q.log_sum_pt - 299.8702
    if 6.98945 <= Q.log_sum_pt < 7.139296:
        z += 22.10811 * Q.log_sum_pt - 153.3615
    if Q.log_sum_pt >= 7.139296:
        z += 12.27159 * Q.log_sum_pt - 83.13567
    if Q.sum_pt_top50 < 934.2416:
        z += -0.01536151 * Q.sum_pt_top50 + 16.57498
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += -0.02311121 * Q.sum_pt_top50 + 23.81507
    if 959.0957 <= Q.sum_pt_top50 < 1078.994:
        z += -0.03042438 * Q.sum_pt_top50 + 30.8291
    if Q.sum_pt_top50 >= 1078.994:
        z += -0.01506287 * Q.sum_pt_top50 + 14.25412
    if Q.sum_z_dr2_top15 < 0.002197765:
        z += -308.2624 * Q.sum_z_dr2_top15 + 0.6774883
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.07323838 * Q.n_dr_0p2_0p4 - 0.5126687
    if Q.pt_11 < 29.04688:
        z += 0.02050756 * Q.pt_11 - 0.5956807
    if 972.0419 <= Q.sum_pt < 1052.889:
        z += 0.02433981 * Q.sum_pt - 23.65932
    if Q.sum_pt >= 1052.889:
        z += 0.009278902 * Q.sum_pt - 7.801845
    if Q.sum_z_dr2_top5 < 0.0006570502:
        z += -533.6476 * Q.sum_z_dr2_top5 + 0.3506332
    if Q.sum_pt_top2 < 605.875:
        z += -0.001812132 * Q.sum_pt_top2 + 1.097926
    if Q.z_top30_slots >= 0.9341838:
        z += 15.50838 * Q.z_top30_slots - 14.48768
    if Q.n_dr_0_0p05 >= 15.0:
        z += 0.03207824 * Q.n_dr_0_0p05 - 0.4811736
    if Q.n_dr_0p1_0p2 < 7.0:
        z += 0.09055687 * Q.n_dr_0p1_0p2 - 0.6338981
    if Q.tau1 < 0.07708632:
        z += -28.63897 * Q.tau1 + 2.207673
    if Q.tau1 >= 0.1219132:
        z += -7.588839 * Q.tau1 + 0.92518
    if Q.lam2 < 0.001776308:
        z += 233.8727 * Q.lam2 - 0.4154301
    if Q.sum_pt_top40 < 1225.842:
        z += -0.003070715 * Q.sum_pt_top40 + 3.764212
    if Q.n_particles > 38.0 and Q.zdr_0 < 0.009970338:
        z += 3.45781 * (Q.n_particles - 38.0) * (0.009970338 - Q.zdr_0)
    if Q.n_particles > 38.0 and Q.soft1_z < 0.002181998:
        z += -16.13593 * (Q.n_particles - 38.0) * (0.002181998 - Q.soft1_z)
    if Q.z_top5 > 0.6551948 and Q.pt1_dr01 < 28.39396:
        z += -0.1579462 * (Q.z_top5 - 0.6551948) * (28.39396 - Q.pt1_dr01)
    if Q.z_top30_slots > 0.9341838 and Q.pt1_dr01 > 5.351077:
        z += 0.4954187 * (Q.z_top30_slots - 0.9341838) * (Q.pt1_dr01 - 5.351077)
    if Q.n_particles > 38.0 and Q.tau32 > 0.3293142:
        z += 0.0380887 * (Q.n_particles - 38.0) * (Q.tau32 - 0.3293142)
    if Q.M3 < 0.03457336 and Q.psi_0p3 > 0.9299135:
        z += -344.2912 * (0.03457336 - Q.M3) * (Q.psi_0p3 - 0.9299135)
    if Q.z_dr_0_0p05 > 0.8103116 and Q.n_dr_0p05_0p1 < 10.0:
        z += -0.664814 * (Q.z_dr_0_0p05 - 0.8103116) * (10.0 - Q.n_dr_0p05_0p1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 < 17.94219:
        z += -0.6891449 * (Q.z_top30_slots - 0.9341838) * (17.94219 - Q.ptdr0_3)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.0471693
    if Q.log_sum_pt >= 6.930088:
        z += 10.46209 * Q.log_sum_pt - 72.50321
    if Q.sum_z_dr < 0.03577037:
        z += 39.43488 * Q.sum_z_dr - 1.4106
    if 997.0189 <= Q.sum_pt_top50 < 1038.855:
        z += -0.008200026 * Q.sum_pt_top50 + 8.175581
    if Q.sum_pt_top50 >= 1038.855:
        z += -0.01507307 * Q.sum_pt_top50 + 15.31568
    if 985.0781 <= Q.sum_pt_top15 < 1082.548:
        z += 0.004523512 * Q.sum_pt_top15 - 4.456012
    if Q.sum_pt_top15 >= 1082.548:
        z += 0.002025582 * Q.sum_pt_top15 - 1.751884
    if Q.LHA < 0.1870291:
        z += -3.875911 * Q.LHA + 0.7249081
    if Q.log_sum_pt > 6.903423 and Q.sum_z_dr2_top15 < 0.02146578:
        z += 374.5659 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.sum_z_dr2_top15)
    if Q.log_sum_pt > 6.903423 and Q.tau21 < 0.8065577:
        z += 3.395355 * (Q.log_sum_pt - 6.903423) * (0.8065577 - Q.tau21)
    if Q.log_sum_pt > 7.062574 and Q.sum_z_dr2_top15 > 0.009962397:
        z += -13530.37 * (Q.log_sum_pt - 7.062574) * (Q.sum_z_dr2_top15 - 0.009962397)
    if Q.sum_pt_top50 > 997.0189 and Q.zdr_4 < 0.006794973:
        z += 0.3414072 * (Q.sum_pt_top50 - 997.0189) * (0.006794973 - Q.zdr_4)
    if Q.sum_pt_top20 > 1129.275 and Q.dr_max_012 > 0.1206357:
        z += -0.6706423 * (Q.sum_pt_top20 - 1129.275) * (Q.dr_max_012 - 0.1206357)
    if Q.log_sum_pt > 6.903423 and Q.zdr_3 > 0.00453462:
        z += -573.3236 * (Q.log_sum_pt - 6.903423) * (Q.zdr_3 - 0.00453462)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.231003
    if Q.lam2 < 0.0006154841:
        z += -1671.857 * Q.lam2 + 1.029001
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.327104 * Q.n_dr_0p2_0p4 + 1.63552
    if Q.n_particles < 46.0:
        z += -0.0348314 * Q.n_particles + 1.602244
    if Q.tau21 < 0.347196:
        z += 2.062535 * Q.tau21 - 0.7161039
    if Q.n_dr_0p1_0p2 < 17.0:
        z += -0.0263794 * Q.n_dr_0p1_0p2 + 0.4484498
    if Q.n_dr_0p2_0p4 < 5.0 and Q.e2 < 0.03029714:
        z += -10.2908 * (5.0 - Q.n_dr_0p2_0p4) * (0.03029714 - Q.e2)
    if Q.n_particles < 46.0 and Q.e2 > 0.01036127:
        z += -1.406403 * (46.0 - Q.n_particles) * (Q.e2 - 0.01036127)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.n_dr_0p05_0p1 > 2.0:
        z += -0.002565901 * (8.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p05_0p1 - 2.0)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.474817
    if Q.sum_pt_top40 < 1041.263:
        z += 0.01170763 * Q.sum_pt_top40 - 12.19072
    if Q.psi_0p1 >= 0.8976117:
        z += -2.953425 * Q.psi_0p1 + 2.651029
    if Q.n_particles >= 26.0:
        z += -0.01966635 * Q.n_particles + 0.5113252
    if Q.n_dr_0_0p05 >= 14.0:
        z += 0.03143227 * Q.n_dr_0_0p05 - 0.4400517
    if Q.sum_z_dr2_top15 < 0.005788041:
        z += 75.30847 * Q.sum_z_dr2_top15 + 0.1451226
    if 0.005788041 <= Q.sum_z_dr2_top15 < 0.007887677:
        z += -70.65854 * Q.sum_z_dr2_top15 + 0.9899857
    if 0.007887677 <= Q.sum_z_dr2_top15 < 0.009962397:
        z += -208.5361 * Q.sum_z_dr2_top15 + 2.077519
    if Q.e2 < 0.04358622:
        z += 31.37407 * Q.e2 - 1.367477
    if Q.sum_pt_top30 < 1011.524:
        z += -0.007616101 * Q.sum_pt_top30 + 7.703868
    if Q.sum_z_dr < 0.07374472:
        z += 15.39496 * Q.sum_z_dr - 1.135297
    if 0.1219342 <= Q.sj2_dr < 0.1825048:
        z += 9.773448 * Q.sj2_dr - 1.191718
    if Q.sj2_dr >= 0.1825048:
        z += -1.172919 * Q.sj2_dr + 0.8060472
    if Q.z_dr_0p2_0p4 < 0.05180474:
        z += 10.78052 * Q.z_dr_0p2_0p4 - 0.5584819
    if Q.n_particles > 26.0 and Q.soft1_pt > 0.4909668:
        z += -0.007681879 * (Q.n_particles - 26.0) * (Q.soft1_pt - 0.4909668)
    if Q.sum_z_dr2_top15 < 0.002197765 and Q.psi_0p3 > 0.9973959:
        z += 191204.4 * (0.002197765 - Q.sum_z_dr2_top15) * (Q.psi_0p3 - 0.9973959)
    if Q.sum_z_dr < 0.09749958 and Q.psi_0p3 > 0.9973959:
        z += 8810.725 * (0.09749958 - Q.sum_z_dr) * (Q.psi_0p3 - 0.9973959)
    if Q.sum_z_dr < 0.05660088 and Q.psi_0p3 > 0.9973959:
        z += -20711.16 * (0.05660088 - Q.sum_z_dr) * (Q.psi_0p3 - 0.9973959)
    if Q.sum_z_dr2_top15 < 0.004169954 and Q.psi_0p3 > 0.9973959:
        z += -95342.73 * (0.004169954 - Q.sum_z_dr2_top15) * (Q.psi_0p3 - 0.9973959)
    return max(0.0, z)


def neuron_5(Q):
    z = -1.980782
    z += -0.05349965 * Q.n_particles + 3.423978
    if Q.sum_pt < 907.9372:
        z += 0.05006215 * Q.sum_pt - 46.20189
    if 907.9372 <= Q.sum_pt < 1002.379:
        z += 0.01364661 * Q.sum_pt - 13.13886
    if 1002.379 <= Q.sum_pt < 1085.125:
        z += -0.006528474 * Q.sum_pt + 7.08421
    if Q.sum_pt_top50 < 934.2416:
        z += -0.01499954 * Q.sum_pt_top50 + 14.0132
    if Q.sum_z_dr < 0.1564779:
        z += -28.95403 * Q.sum_z_dr + 4.530665
    if 0.1632346 <= Q.LHA < 0.302389:
        z += 2.332157 * Q.LHA - 0.3806888
    if 0.302389 <= Q.LHA < 0.4331369:
        z += 9.055681 * Q.LHA - 2.413809
    if Q.LHA >= 0.4331369:
        z += -203.2609 * Q.LHA + 89.54833
    if Q.z_top20_slots >= 0.8281581:
        z += -2.598508 * Q.z_top20_slots + 2.151976
    if Q.z_dr_0_0p05 >= 0.7128619:
        z += 2.085328 * Q.z_dr_0_0p05 - 1.486551
    if Q.tau1 < 0.1072713:
        z += 16.52489 * Q.tau1 - 1.772646
    if Q.sum_z_dr2_top15 < 0.0007894752:
        z += 523.6477 * Q.sum_z_dr2_top15 - 0.4134069
    if Q.n_pt_above_1 < 58.0:
        z += 0.0337423 * Q.n_pt_above_1 - 1.957054
    if Q.C2 >= 0.1235569:
        z += -41.89849 * Q.C2 + 5.17685
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.09508798 * Q.n_dr_0p2_0p4 + 1.045968
    if Q.z_top50_slots < 0.985099:
        z += 36.83648 * Q.z_top50_slots - 36.28758
    if Q.sum_pt_top40 < 1013.042:
        z += -0.00458487 * Q.sum_pt_top40 + 4.644668
    if Q.sum_z_dr2_top10 >= 0.005337976:
        z += 36.64757 * Q.sum_z_dr2_top10 - 0.1956239
    if Q.n_particles < 64.0 and Q.C2 < 0.07996447:
        z += -0.3294763 * (64.0 - Q.n_particles) * (0.07996447 - Q.C2)
    if Q.sum_pt < 1002.379 and Q.dr_11 < 0.199671:
        z += 0.02101273 * (1002.379 - Q.sum_pt) * (0.199671 - Q.dr_11)
    if Q.sum_pt < 1002.379 and Q.dr1_13 > 0.1778153:
        z += 0.06227884 * (1002.379 - Q.sum_pt) * (Q.dr1_13 - 0.1778153)
    if Q.sum_pt < 1002.379 and Q.ptdr0_10 > 3.719859:
        z += -0.004126993 * (1002.379 - Q.sum_pt) * (Q.ptdr0_10 - 3.719859)
    if Q.sum_pt < 907.9372 and Q.dr0_12 < 0.1569963:
        z += 0.2203818 * (907.9372 - Q.sum_pt) * (0.1569963 - Q.dr0_12)
    if Q.sum_pt < 1002.379 and Q.e4 < 5.8505e-08:
        z += -471621.9 * (1002.379 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.sum_pt < 1085.125 and Q.e4 < 5.8505e-08:
        z += 22932.31 * (1085.125 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.sum_pt < 907.9372 and Q.dr_11 < 0.2535773:
        z += -0.1840346 * (907.9372 - Q.sum_pt) * (0.2535773 - Q.dr_11)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.2451605
    if Q.tau1 < 0.05444509:
        z += -118.8552 * Q.tau1 + 6.471084
    if Q.n_dr_0p1_0p2 >= 15.0:
        z += -0.06127711 * Q.n_dr_0p1_0p2 + 0.9191567
    if Q.psi_0p2 >= 0.9087063:
        z += -5.723855 * Q.psi_0p2 + 5.201303
    if Q.C2 >= 0.1088881:
        z += -23.34978 * Q.C2 + 2.542513
    if Q.sum_z_dr2_top15 < 0.001319197:
        z += -137.3021 * Q.sum_z_dr2_top15 + 0.8756401
    if 0.001319197 <= Q.sum_z_dr2_top15 < 0.003270031:
        z += -356.0075 * Q.sum_z_dr2_top15 + 1.164156
    if Q.sum_z_dr < 0.08589404:
        z += 121.5023 * Q.sum_z_dr - 11.61989
    if 0.08589404 <= Q.sum_z_dr < 0.09749958:
        z += 101.9823 * Q.sum_z_dr - 9.94323
    if Q.e3 < 5.13841e-05:
        z += -3143.679 * Q.e3 + 1.060155
    if 5.13841e-05 <= Q.e3 < 0.0001086251:
        z += 9324.196 * Q.e3 + 0.4195047
    if 0.0001086251 <= Q.e3 < 0.0003372339:
        z += -6883.228 * Q.e3 + 2.180037
    if Q.e3 >= 0.0003372339:
        z += -3739.549 * Q.e3 + 1.119882
    if Q.sj2_zsoft < 0.3676068:
        z += -1.005326 * Q.sj2_zsoft + 0.3695645
    if Q.e2 < 0.04358622:
        z += -34.18767 * Q.e2 + 1.490111
    if Q.e2 >= 0.05557149:
        z += 91.43436 * Q.e2 - 5.081144
    if Q.z_dr_0_0p05 >= 0.3289237:
        z += 0.762807 * Q.z_dr_0_0p05 - 0.2509053
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += -0.04936161 * Q.n_dr_0p2_0p4 + 0.7404242
    if Q.tau2 < 0.0795038:
        z += -4.691187 * Q.tau2 + 0.3729672
    if Q.lam2 < 0.006427167:
        z += -61.75095 * Q.lam2 + 0.3968836
    if Q.LHA < 0.3332345:
        z += -16.76941 * Q.LHA + 5.588148
    if Q.sum_pt >= 907.9372:
        z += 0.003397282 * Q.sum_pt - 3.084519
    if Q.z_dr_0p1_0p2 >= 0.3340477:
        z += -1.898859 * Q.z_dr_0p1_0p2 + 0.6343094
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += 7.814523 * Q.z_dr_0p2_0p4 - 0.5352284
    if Q.e3 < 0.0003372339 and Q.log_sum_pt < 7.017258:
        z += -15567.09 * (0.0003372339 - Q.e3) * (7.017258 - Q.log_sum_pt)
    if Q.tau1 < 0.05444509 and Q.sum_pt < 972.0419:
        z += 0.4929946 * (0.05444509 - Q.tau1) * (972.0419 - Q.sum_pt)
    if Q.n_dr_0p1_0p2 > 15.0 and Q.psi_0p3 > 0.9853273:
        z += 7.312651 * (Q.n_dr_0p1_0p2 - 15.0) * (Q.psi_0p3 - 0.9853273)
    if Q.sum_z_dr2_top15 < 0.003270031 and Q.sum_pt_top40 > 858.8262:
        z += -0.7943816 * (0.003270031 - Q.sum_z_dr2_top15) * (Q.sum_pt_top40 - 858.8262)
    if Q.sum_z_dr < 0.09749958 and Q.psi_0p3 < 0.9638082:
        z += 998.8271 * (0.09749958 - Q.sum_z_dr) * (0.9638082 - Q.psi_0p3)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.3198544
    if Q.e3 < 5.727594e-05:
        z += 82.9873 * Q.e3 + 0.08278854
    if 5.727594e-05 <= Q.e3 < 6.567534e-05:
        z += -10422.38 * Q.e3 + 0.6844933
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.001694694 * Q.n_dr_0p2_0p4 + 0.6912509
    if 15.0 <= Q.n_dr_0p2_0p4 < 21.0:
        z += -0.1109717 * Q.n_dr_0p2_0p4 + 2.330407
    if Q.psi_0p3 >= 0.9973959:
        z += 2789.05 * Q.psi_0p3 - 2781.787
    if Q.psi_0p1 >= 0.8509811:
        z += -5.485463 * Q.psi_0p1 + 4.668025
    if Q.tau1 < 0.1072713:
        z += 68.18019 * Q.tau1 - 7.057139
    if 0.1072713 <= Q.tau1 < 0.1219132:
        z += -17.52736 * Q.tau1 + 2.136817
    if Q.LHA < 0.302389:
        z += -26.73088 * Q.LHA + 7.956998
    if 0.302389 <= Q.LHA < 0.3332345:
        z += 4.088955 * Q.LHA - 1.362581
    if Q.e2 < 0.03680582:
        z += -52.80746 * Q.e2 + 2.117836
    if 0.03680582 <= Q.e2 < 0.04755309:
        z += -2.352753 * Q.e2 + 0.2608088
    if 0.04755309 <= Q.e2 < 0.05557149:
        z += -18.5733 * Q.e2 + 1.032146
    if Q.sum_z_dr < 0.08068193:
        z += 43.76071 * Q.sum_z_dr - 2.716272
    if 0.08068193 <= Q.sum_z_dr < 0.09749958:
        z += -48.42687 * Q.sum_z_dr + 4.7216
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1937688:
        z += 58.35539 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1937688)
    if Q.tau21_b2 < 0.2352054 and Q.sum_z_dr2_top15 < 0.007887677:
        z += -6527.175 * (0.2352054 - Q.tau21_b2) * (0.007887677 - Q.sum_z_dr2_top15)
    if Q.tau21_b2 < 0.2352054 and Q.sum_z_dr2_top15 < 0.009962397:
        z += 5276.544 * (0.2352054 - Q.tau21_b2) * (0.009962397 - Q.sum_z_dr2_top15)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt_top50 < 1245.697:
        z += -0.06221657 * (0.2352054 - Q.tau21_b2) * (1245.697 - Q.sum_pt_top50)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.e3 < 7.876005e-05:
        z += -1047.351 * (21.0 - Q.n_dr_0p2_0p4) * (7.876005e-05 - Q.e3)
    if Q.psi_0p3 > 0.9973959 and Q.mean_eta2 < 0.006802603:
        z += -305403.6 * (Q.psi_0p3 - 0.9973959) * (0.006802603 - Q.mean_eta2)
    if Q.psi_0p3 > 0.9973959 and Q.log_sum_pt < 7.062574:
        z += -5607.277 * (Q.psi_0p3 - 0.9973959) * (7.062574 - Q.log_sum_pt)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 972.0419:
        z += -0.1171533 * (0.2352054 - Q.tau21_b2) * (972.0419 - Q.sum_pt)
    if Q.psi_0p3 > 0.9973959 and Q.mean_phi2 < 0.006808102:
        z += -288381.2 * (Q.psi_0p3 - 0.9973959) * (0.006808102 - Q.mean_phi2)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.e3 < 3.376709e-05:
        z += -999.7761 * (21.0 - Q.n_dr_0p2_0p4) * (3.376709e-05 - Q.e3)
    if Q.psi_0p3 > 0.9973959 and Q.sum_z_dr2_top10 > 0.007678544:
        z += -239101.6 * (Q.psi_0p3 - 0.9973959) * (Q.sum_z_dr2_top10 - 0.007678544)
    if Q.psi_0p3 > 0.9973959 and Q.sum_z_dr2_top10 > 0.01414829:
        z += -436447.7 * (Q.psi_0p3 - 0.9973959) * (Q.sum_z_dr2_top10 - 0.01414829)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 1115.723:
        z += 0.05427478 * (0.2352054 - Q.tau21_b2) * (1115.723 - Q.sum_pt)
    if Q.tau1 < 0.1072713 and Q.z_dr_0p1_0p2 < 0.2864926:
        z += 233.7305 * (0.1072713 - Q.tau1) * (0.2864926 - Q.z_dr_0p1_0p2)
    if Q.LHA < 0.3332345 and Q.z_dr_0p1_0p2 < 0.250441:
        z += -84.31394 * (0.3332345 - Q.LHA) * (0.250441 - Q.z_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.M2 < 0.1134943:
        z += 0.6210658 * (15.0 - Q.n_dr_0p2_0p4) * (0.1134943 - Q.M2)
    if Q.psi_0p3 > 0.9973959 and Q.lam2 < 0.003687605:
        z += 38227.25 * (Q.psi_0p3 - 0.9973959) * (0.003687605 - Q.lam2)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.263241
    if 0.03577037 <= Q.sum_z_dr < 0.04362872:
        z += 61.81422 * Q.sum_z_dr - 2.211117
    if 0.04362872 <= Q.sum_z_dr < 0.07031778:
        z += 152.6236 * Q.sum_z_dr - 6.173016
    if 0.07031778 <= Q.sum_z_dr < 0.07374472:
        z += 119.9097 * Q.sum_z_dr - 3.872642
    if 0.07374472 <= Q.sum_z_dr < 0.09749958:
        z += 53.1965 * Q.sum_z_dr + 1.047102
    if Q.sum_z_dr >= 0.09749958:
        z += -94.82259 * Q.sum_z_dr + 15.4789
    if Q.sum_pt_top40 < 1007.44:
        z += -0.002756182 * Q.sum_pt_top40 + 2.776688
    if Q.sum_pt < 1002.379:
        z += -0.03067087 * Q.sum_pt + 33.84391
    if 1002.379 <= Q.sum_pt < 1042.609:
        z += -0.01854463 * Q.sum_pt + 21.68883
    if 1042.609 <= Q.sum_pt < 1260.541:
        z += -0.01080168 * Q.sum_pt + 13.61595
    if 0.2091025 <= Q.LHA < 0.3098384:
        z += -30.16157 * Q.LHA + 6.30686
    if 0.3098384 <= Q.LHA < 0.3203321:
        z += -36.08515 * Q.LHA + 8.142212
    if 0.3203321 <= Q.LHA < 0.3332345:
        z += -50.6969 * Q.LHA + 12.82282
    if 0.3332345 <= Q.LHA < 0.3719813:
        z += 5.636086 * Q.LHA - 5.949271
    if Q.LHA >= 0.3719813:
        z += 18.66418 * Q.LHA - 10.79548
    if Q.psi_0p3 >= 0.9777125:
        z += -38.03852 * Q.psi_0p3 + 37.19074
    if Q.z_top50_slots >= 0.9586536:
        z += -18.30741 * Q.z_top50_slots + 17.55047
    if Q.e3 < 3.793233e-05:
        z += -15715.4 * Q.e3 + 0.5961218
    if Q.n_dr_0p2_0p4 < 8.0:
        z += 0.1219717 * Q.n_dr_0p2_0p4 - 0.9757733
    if Q.sum_pt_top30 < 1027.303:
        z += 0.003821738 * Q.sum_pt_top30 - 3.926084
    if Q.log_sum_pt < 7.139296:
        z += 19.22763 * Q.log_sum_pt - 137.2717
    if Q.sum_z_dr2_top15 < 0.00727763:
        z += 209.7542 * Q.sum_z_dr2_top15 - 1.526513
    if 0.02210818 <= Q.e2 < 0.04358622:
        z += -44.52482 * Q.e2 + 0.9843628
    if Q.e2 >= 0.04358622:
        z += -2.511597 * Q.e2 - 0.8468348
    if Q.sj2_dr >= 0.2232169:
        z += 3.222962 * Q.sj2_dr - 0.7194195
    if Q.dr_0 < 0.08082334:
        z += -7.263092 * Q.dr_0 + 0.5870273
    if Q.n_dr_0p2_0p4 < 18.0 and Q.log_sum_pt < 7.139296:
        z += -0.373296 * (18.0 - Q.n_dr_0p2_0p4) * (7.139296 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.psi_0p1 < 0.8252004:
        z += 0.2352812 * (18.0 - Q.n_dr_0p2_0p4) * (0.8252004 - Q.psi_0p1)
    if Q.sum_z_dr > 0.07031778 and Q.e2 > 0.04358622:
        z += 910.9838 * (Q.sum_z_dr - 0.07031778) * (Q.e2 - 0.04358622)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.e3 < 3.793233e-05:
        z += -4736.091 * (18.0 - Q.n_dr_0p2_0p4) * (3.793233e-05 - Q.e3)
    if Q.sum_z_dr > 0.07031778 and Q.tau2 < 0.0795038:
        z += 343.8203 * (Q.sum_z_dr - 0.07031778) * (0.0795038 - Q.tau2)
    if Q.psi_0p3 > 0.9777125 and Q.n_dr_0p1_0p2 > 17.0:
        z += 1.845996 * (Q.psi_0p3 - 0.9777125) * (Q.n_dr_0p1_0p2 - 17.0)
    if Q.sum_z_dr > 0.07031778 and Q.sum_pt_top30 < 1191.938:
        z += 0.1531527 * (Q.sum_z_dr - 0.07031778) * (1191.938 - Q.sum_pt_top30)
    if Q.sum_z_dr > 0.07031778 and Q.sum_pt_top50 < 976.277:
        z += -0.3197275 * (Q.sum_z_dr - 0.07031778) * (976.277 - Q.sum_pt_top50)
    if Q.sum_z_dr > 0.07031778 and Q.sum_pt < 1115.723:
        z += 0.1974583 * (Q.sum_z_dr - 0.07031778) * (1115.723 - Q.sum_pt)
    if Q.sum_pt < 1260.541 and Q.D2 < 4.450169:
        z += -0.0008001676 * (1260.541 - Q.sum_pt) * (4.450169 - Q.D2)
    if Q.sum_pt < 1042.609 and Q.max_dr > 0.1939977:
        z += 0.03089876 * (1042.609 - Q.sum_pt) * (Q.max_dr - 0.1939977)
    if Q.sum_z_dr2_top15 < 0.02675364 and Q.tau4 > 0.01517184:
        z += 1740.615 * (0.02675364 - Q.sum_z_dr2_top15) * (Q.tau4 - 0.01517184)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.6561254
    if Q.sum_pt_top50 < 889.8503:
        z += -0.02785183 * Q.sum_pt_top50 + 25.84414
    if 889.8503 <= Q.sum_pt_top50 < 988.4554:
        z += -0.01075177 * Q.sum_pt_top50 + 10.62764
    if Q.sum_z_dr < 0.05660088:
        z += -16.47759 * Q.sum_z_dr + 0.9326463
    if 0.09749958 <= Q.sum_z_dr < 0.1207452:
        z += 30.78674 * Q.sum_z_dr - 3.001694
    if 0.1207452 <= Q.sum_z_dr < 0.1402186:
        z += -110.7437 * Q.sum_z_dr + 14.08743
    if Q.sum_z_dr >= 0.1402186:
        z += -202.3435 * Q.sum_z_dr + 26.93142
    if Q.LHA < 0.2454112:
        z += 6.876725 * Q.LHA - 1.687625
    if 0.3719813 <= Q.LHA < 0.404204:
        z += 34.5481 * Q.LHA - 12.85124
    if Q.LHA >= 0.404204:
        z += 88.52402 * Q.LHA - 34.66853
    if Q.sum_pt_top20 >= 750.7313:
        z += -0.003863734 * Q.sum_pt_top20 + 2.900626
    if Q.sum_z_dr2_top15 < 0.0007894752:
        z += 1260.662 * Q.sum_z_dr2_top15 - 2.124987
    if 0.0007894752 <= Q.sum_z_dr2_top15 < 0.004855289:
        z += 254.7322 * Q.sum_z_dr2_top15 - 1.330831
    if 0.004855289 <= Q.sum_z_dr2_top15 < 0.006142802:
        z += 73.03395 * Q.sum_z_dr2_top15 - 0.4486331
    if Q.sum_z_dr2_top15 >= 0.02146578:
        z += -148.6994 * Q.sum_z_dr2_top15 + 3.191949
    if Q.D2 < 2.178951:
        z += -0.6209 * Q.D2 + 1.35291
    if Q.z_top40_slots >= 0.9674996:
        z += -22.36917 * Q.z_top40_slots + 21.64217
    if Q.tau1 < 0.09591084:
        z += -58.27054 * Q.tau1 + 5.798377
    if 0.09591084 <= Q.tau1 < 0.1072713:
        z += -18.45006 * Q.tau1 + 1.979161
    if Q.e2 >= 0.02515919:
        z += 62.5289 * Q.e2 - 1.573177
    if Q.log_sum_pt < 6.811175:
        z += 11.60256 * Q.log_sum_pt - 79.02708
    if Q.sum_z_dr2_top2 < 0.002412891:
        z += 249.1914 * Q.sum_z_dr2_top2 - 0.6012718
    if Q.sum_z_dr < 0.05660088 and Q.sum_pt < 1007.788:
        z += -0.132335 * (0.05660088 - Q.sum_z_dr) * (1007.788 - Q.sum_pt)
    if Q.sum_z_dr < 0.05660088 and Q.psi_0p3 > 0.9638082:
        z += 753.0797 * (0.05660088 - Q.sum_z_dr) * (Q.psi_0p3 - 0.9638082)
    if Q.sum_z_dr2_top15 < 0.006142802 and Q.z_dr_0p2_0p4 < 0.0684915:
        z += 5899.814 * (0.006142802 - Q.sum_z_dr2_top15) * (0.0684915 - Q.z_dr_0p2_0p4)
    if Q.sum_z_dr2_top15 < 0.006142802 and Q.n_dr_0p2_0p4 > 7.0:
        z += 21.24022 * (0.006142802 - Q.sum_z_dr2_top15) * (Q.n_dr_0p2_0p4 - 7.0)
    if Q.sum_z_dr > 0.1402186 and Q.soft6_pt > 4.250195:
        z += -380.4117 * (Q.sum_z_dr - 0.1402186) * (Q.soft6_pt - 4.250195)
    if Q.log_sum_pt < 6.811175 and Q.dr_13 < 0.09061548:
        z += 1181.699 * (6.811175 - Q.log_sum_pt) * (0.09061548 - Q.dr_13)
    if Q.sum_pt_top50 < 988.4554 and Q.dr_13 < 0.1312677:
        z += -0.08967225 * (988.4554 - Q.sum_pt_top50) * (0.1312677 - Q.dr_13)
    if Q.sum_z_dr > 0.1564779 and Q.soft6_pt > 1.931641:
        z += 22.00824 * (Q.sum_z_dr - 0.1564779) * (Q.soft6_pt - 1.931641)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.soft5_z > 0.0004140594:
        z += -3362.434 * (Q.z_dr_0p1_0p2 - 0.4479367) * (Q.soft5_z - 0.0004140594)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.soft5_pt > 0.5297852:
        z += 4.342172 * (Q.z_dr_0p1_0p2 - 0.4479367) * (Q.soft5_pt - 0.5297852)
    return max(0.0, z)


def neuron_10(Q):
    z = 5.938058
    if Q.sum_z_dr < 0.09749958:
        z += 42.78946 * Q.sum_z_dr - 5.166621
    if 0.09749958 <= Q.sum_z_dr < 0.1207452:
        z += 9.800354 * Q.sum_z_dr - 1.950197
    if Q.sum_z_dr >= 0.1207452:
        z += -32.98911 * Q.sum_z_dr + 3.216424
    if Q.tau1 >= 0.1953848:
        z += -87.18699 * Q.tau1 + 17.03502
    if Q.e2 >= 0.03029714:
        z += 52.39308 * Q.e2 - 1.587361
    if 402.625 <= Q.sum_pt_top5 < 791.125:
        z += -0.002855187 * Q.sum_pt_top5 + 1.14957
    if Q.sum_pt_top5 >= 791.125:
        z += -0.005921398 * Q.sum_pt_top5 + 3.575326
    if Q.psi_0p3 >= 0.9989733:
        z += -308.4485 * Q.psi_0p3 + 308.1318
    if Q.sum_pt_top50 < 889.8503:
        z += 0.004116209 * Q.sum_pt_top50 - 3.66281
    if 0.1870291 <= Q.LHA < 0.3332345:
        z += -20.93521 * Q.LHA + 3.915493
    if Q.LHA >= 0.3332345:
        z += -8.237777 * Q.LHA - 0.3157292
    if Q.n_dr_0p1_0p2 >= 21.0:
        z += 0.03458984 * Q.n_dr_0p1_0p2 - 0.7263867
    if Q.sum_z_dr2_top10 < 0.01414829:
        z += 69.55142 * Q.sum_z_dr2_top10 - 0.9840335
    if Q.C2 < 0.05602756:
        z += -18.02151 * Q.C2 + 1.009701
    if Q.tau21_b2 < 0.2018786:
        z += 6.240935 * Q.tau21_b2 - 1.259911
    if Q.log_sum_pt < 6.856375:
        z += 10.83093 * Q.log_sum_pt - 74.26094
    if Q.D2 < 2.410481:
        z += -0.3795475 * Q.D2 + 0.9148921
    if Q.sum_pt < 986.0565:
        z += -0.01225667 * Q.sum_pt + 12.08576
    if Q.psi_0p1 >= 0.9184255:
        z += 10.26547 * Q.psi_0p1 - 9.428072
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += 5.011037 * Q.z_dr_0p1_0p2 - 0.6030466
    if Q.sum_z_dr < 0.1207452 and Q.psi_0p2 > 0.948102:
        z += 116.209 * (0.1207452 - Q.sum_z_dr) * (Q.psi_0p2 - 0.948102)
    if Q.e2 > 0.03029714 and Q.log_sum_pt > 6.910131:
        z += -148.2301 * (Q.e2 - 0.03029714) * (Q.log_sum_pt - 6.910131)
    if Q.tau1 > 0.1953848 and Q.sum_pt > 1167.447:
        z += 0.4663086 * (Q.tau1 - 0.1953848) * (Q.sum_pt - 1167.447)
    if Q.sj2_dr > 0.09395198 and Q.M2 > 0.04260132:
        z += -48.13543 * (Q.sj2_dr - 0.09395198) * (Q.M2 - 0.04260132)
    if Q.tau1 > 0.1953848 and Q.pt_3 < 114.75:
        z += 1.901363 * (Q.tau1 - 0.1953848) * (114.75 - Q.pt_3)
    if Q.tau1 > 0.1953848 and Q.z_3 < 0.1082864:
        z += -1206.024 * (Q.tau1 - 0.1953848) * (0.1082864 - Q.z_3)
    if Q.LHA > 0.1870291 and Q.sj3_pairmin_over_m > 0.1389615:
        z += 11.26575 * (Q.LHA - 0.1870291) * (Q.sj3_pairmin_over_m - 0.1389615)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.6493608
    if Q.sum_z_dr < 0.076787:
        z += -61.11238 * Q.sum_z_dr + 5.081961
    if 0.076787 <= Q.sum_z_dr < 0.08589404:
        z += -65.85343 * Q.sum_z_dr + 5.446012
    if 0.08589404 <= Q.sum_z_dr < 0.09749958:
        z += 18.1298 * Q.sum_z_dr - 1.767648
    if Q.D2 < 1.409617:
        z += -0.4746796 * Q.D2 + 0.6691166
    if Q.sum_z_dr2_top10 < 0.00130722:
        z += -568.0725 * Q.sum_z_dr2_top10 + 0.7425959
    if Q.log_sum_pt >= 6.893714:
        z += -4.948132 * Q.log_sum_pt + 34.11101
    if Q.n_dr_0p1_0p2 < 19.0:
        z += -0.03393046 * Q.n_dr_0p1_0p2 + 0.6446787
    if Q.sum_z_dr2_top15 < 0.006142802:
        z += 567.3172 * Q.sum_z_dr2_top15 - 3.089033
    if 0.006142802 <= Q.sum_z_dr2_top15 < 0.007887677:
        z += -226.8838 * Q.sum_z_dr2_top15 + 1.789587
    if Q.LHA < 0.3098384:
        z += 17.88446 * Q.LHA - 5.100503
    if 0.3098384 <= Q.LHA < 0.3332345:
        z += -18.84022 * Q.LHA + 6.278211
    if Q.e2 < 0.02515919:
        z += -130.6563 * Q.e2 + 1.913424
    if 0.02515919 <= Q.e2 < 0.04082832:
        z += 87.67452 * Q.e2 - 3.579603
    if Q.sum_pt >= 986.0565:
        z += -0.008542825 * Q.sum_pt + 8.423709
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += -15.9039 * Q.z_dr_0p2_0p4 + 1.089282
    if Q.sj3_dr_max < 0.1623049:
        z += 8.997885 * Q.sj3_dr_max - 1.003728
    if 0.1623049 <= Q.sj3_dr_max < 0.1894436:
        z += -16.82729 * Q.sj3_dr_max + 3.187823
    if Q.z_dr_0_0p05 >= 0.8103116:
        z += 5.561177 * Q.z_dr_0_0p05 - 4.506286
    if Q.n_dr_0p2_0p4 < 10.0 and Q.sum_z_dr2_top10 < 0.005337976:
        z += -24.67969 * (10.0 - Q.n_dr_0p2_0p4) * (0.005337976 - Q.sum_z_dr2_top10)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 33.0:
        z += 0.00422944 * (10.0 - Q.n_dr_0p2_0p4) * (33.0 - Q.n_dr_0p1_0p2)
    if Q.D2 < 1.409617 and Q.n_real_top50 > 22.0:
        z += -0.04374325 * (1.409617 - Q.D2) * (Q.n_real_top50 - 22.0)
    if Q.log_sum_pt > 6.893714 and Q.z_top50_slots > 0.9704436:
        z += 325.3366 * (Q.log_sum_pt - 6.893714) * (Q.z_top50_slots - 0.9704436)
    if Q.log_sum_pt > 6.893714 and Q.sd_zg < 0.3213081:
        z += 11.77309 * (Q.log_sum_pt - 6.893714) * (0.3213081 - Q.sd_zg)
    return max(0.0, z)


def neuron_12(Q):
    z = 0.1317661
    if Q.psi_0p3 >= 0.9896594:
        z += -111.2077 * Q.psi_0p3 + 110.0578
    if Q.D2 < 4.450169:
        z += -0.1730102 * Q.D2 + 0.7699246
    if Q.LHA < 0.1870291:
        z += 47.58566 * Q.LHA - 10.80982
    if 0.1870291 <= Q.LHA < 0.2284021:
        z += 37.96202 * Q.LHA - 9.009918
    if 0.2284021 <= Q.LHA < 0.2454112:
        z += 19.94887 * Q.LHA - 4.895678
    if 0.3719813 <= Q.LHA < 0.404204:
        z += 43.332 * Q.LHA - 16.11869
    if Q.LHA >= 0.404204:
        z += -13.01258 * Q.LHA + 6.656011
    if Q.log_sum_pt >= 6.811175:
        z += -8.564563 * Q.log_sum_pt + 58.33474
    if Q.sum_z_dr < 0.06171014:
        z += -132.6153 * Q.sum_z_dr + 8.183709
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.09282518 * Q.n_dr_0p2_0p4 + 0.9282518
    if Q.sj3_dr12 < 0.06069672:
        z += -0.949821 * Q.sj3_dr12 - 0.001607674
    if 0.06069672 <= Q.sj3_dr12 < 0.1048824:
        z += 9.190726 * Q.sj3_dr12 - 0.6171056
    if 0.1048824 <= Q.sj3_dr12 < 0.1840219:
        z += -4.382643 * Q.sj3_dr12 + 0.8065021
    if Q.mean_eta2 < 0.005850286:
        z += -103.5369 * Q.mean_eta2 + 0.6057205
    if Q.sum_pt < 1167.447:
        z += 0.00427021 * Q.sum_pt - 4.985243
    if Q.eccentricity >= 0.8903081:
        z += -6.972577 * Q.eccentricity + 6.207742
    if Q.sj3_dr_max < 0.1506299:
        z += 14.90741 * Q.sj3_dr_max - 1.55354
    if 0.1506299 <= Q.sj3_dr_max < 0.1999777:
        z += -14.02213 * Q.sj3_dr_max + 2.804113
    if 0.03029714 <= Q.e2 < 0.06524004:
        z += -17.08195 * Q.e2 + 0.5175342
    if Q.e2 >= 0.06524004:
        z += 55.20991 * Q.e2 - 4.198789
    if Q.sum_pt_top20 < 846.1934:
        z += 0.003396698 * Q.sum_pt_top20 - 2.874264
    if Q.sum_z_dr < 0.05048381 and Q.log_sum_pt > 6.811175:
        z += 295.7659 * (0.05048381 - Q.sum_z_dr) * (Q.log_sum_pt - 6.811175)
    if Q.sj3_dr_max < 0.2628766 and Q.sum_z_dr2_top10 > 0.004752876:
        z += -7720.924 * (0.2628766 - Q.sj3_dr_max) * (Q.sum_z_dr2_top10 - 0.004752876)
    if Q.psi_0p3 > 0.9896594 and Q.sum_z_dr2_top15 < 0.006615185:
        z += 29401.03 * (Q.psi_0p3 - 0.9896594) * (0.006615185 - Q.sum_z_dr2_top15)
    if Q.psi_0p3 > 0.9896594 and Q.log_sum_pt < 7.139296:
        z += 292.36 * (Q.psi_0p3 - 0.9896594) * (7.139296 - Q.log_sum_pt)
    if Q.LHA > 0.3719813 and Q.lam2 < 0.003687605:
        z += 10537.54 * (Q.LHA - 0.3719813) * (0.003687605 - Q.lam2)
    if Q.LHA > 0.3719813 and Q.log_sum_pt > 6.856375:
        z += 130.0759 * (Q.LHA - 0.3719813) * (Q.log_sum_pt - 6.856375)
    if Q.psi_0p2 > 0.9734513 and Q.n_dr_0p1_0p2 < 21.0:
        z += 2.909391 * (Q.psi_0p2 - 0.9734513) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.log_sum_pt > 6.811175 and Q.mean_eta2 < 0.01204531:
        z += -495.7945 * (Q.log_sum_pt - 6.811175) * (0.01204531 - Q.mean_eta2)
    if Q.sj3_dr_max < 0.2628766 and Q.eccentricity > 0.5245966:
        z += 7.843458 * (0.2628766 - Q.sj3_dr_max) * (Q.eccentricity - 0.5245966)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.18827
    if Q.sum_pt < 1085.125:
        z += -0.1959183 * Q.sum_pt + 212.7959
    if 1085.125 <= Q.sum_pt < 1115.723:
        z += -0.006540644 * Q.sum_pt + 7.297545
    if Q.sum_pt >= 1167.447:
        z += -0.009317971 * Q.sum_pt + 10.87823
    if Q.sum_z_dr2_top15 >= 0.009962397:
        z += -69.86916 * Q.sum_z_dr2_top15 + 0.6960643
    if Q.n_pt_above_5 < 23.0:
        z += 0.0323652 * Q.n_pt_above_5 - 0.7443997
    if Q.n_dr_0p1_0p2 < 21.0:
        z += 0.02937363 * Q.n_dr_0p1_0p2 - 0.7637143
    if 21.0 <= Q.n_dr_0p1_0p2 < 26.0:
        z += 0.07473828 * Q.n_dr_0p1_0p2 - 1.716372
    if Q.n_dr_0p1_0p2 >= 26.0:
        z += 0.04536465 * Q.n_dr_0p1_0p2 - 0.9526576
    if Q.log_sum_pt < 6.98945:
        z += 216.0222 * Q.log_sum_pt - 1509.876
    if 0.08589404 <= Q.sum_z_dr < 0.09749958:
        z += -59.57195 * Q.sum_z_dr + 5.116876
    if 0.09749958 <= Q.sum_z_dr < 0.1207452:
        z += -154.2964 * Q.sum_z_dr + 14.35247
    if 0.1207452 <= Q.sum_z_dr < 0.1402186:
        z += -170.5303 * Q.sum_z_dr + 16.31264
    if 0.1402186 <= Q.sum_z_dr < 0.1564779:
        z += -181.0452 * Q.sum_z_dr + 17.78701
    if Q.sum_z_dr >= 0.1564779:
        z += -49.40818 * Q.sum_z_dr - 2.811262
    if 0.3203321 <= Q.LHA < 0.4331369:
        z += 36.89148 * Q.LHA - 11.81753
    if Q.LHA >= 0.4331369:
        z += -23.91506 * Q.LHA + 14.52003
    if Q.e2 >= 0.04082832:
        z += 42.40068 * Q.e2 - 1.731148
    if Q.sum_pt_top50 < 959.0957:
        z += -0.01671282 * Q.sum_pt_top50 + 16.62021
    if 959.0957 <= Q.sum_pt_top50 < 1048.098:
        z += -0.00664049 * Q.sum_pt_top50 + 6.959886
    if Q.e3 < 0.0005178279:
        z += -2713.408 * Q.e3 + 1.405079
    if Q.psi_0p2 >= 0.9087063:
        z += -10.63864 * Q.psi_0p2 + 9.667397
    if Q.sum_pt_top30 >= 1111.245:
        z += 0.00574659 * Q.sum_pt_top30 - 6.385869
    if Q.pt_9 < 41.4375:
        z += -0.02309584 * Q.pt_9 + 0.9570337
    if Q.sum_z_dr2_top15 > 0.009962397 and Q.sum_pt_top40 < 1225.842:
        z += 0.3006641 * (Q.sum_z_dr2_top15 - 0.009962397) * (1225.842 - Q.sum_pt_top40)
    if Q.sum_pt < 1085.125 and Q.log_sum_pt < 6.811175:
        z += 0.08360859 * (1085.125 - Q.sum_pt) * (6.811175 - Q.log_sum_pt)
    if Q.sum_z_dr > 0.09749958 and Q.sum_pt < 1167.447:
        z += 0.1781107 * (Q.sum_z_dr - 0.09749958) * (1167.447 - Q.sum_pt)
    if Q.sum_z_dr > 0.1402186 and Q.soft6_pt > 4.250195:
        z += 70.86886 * (Q.sum_z_dr - 0.1402186) * (Q.soft6_pt - 4.250195)
    if Q.sum_z_dr2_top15 > 0.009962397 and Q.soft7_z > 0.003627839:
        z += -113193.7 * (Q.sum_z_dr2_top15 - 0.009962397) * (Q.soft7_z - 0.003627839)
    if Q.sum_z_dr > 0.1564779 and Q.n_pt_above_50 > 5.0:
        z += -350.422 * (Q.sum_z_dr - 0.1564779) * (Q.n_pt_above_50 - 5.0)
    if Q.e3 < 0.0001841806 and Q.pt_9 < 41.4375:
        z += -195.5127 * (0.0001841806 - Q.e3) * (41.4375 - Q.pt_9)
    return max(0.0, z)


def neuron_14(Q):
    z = 2.595509
    if Q.tau21_b2 < 0.342495:
        z += -6.413137 * Q.tau21_b2 + 2.196467
    if Q.n_dr_0p1_0p2 < 17.0:
        z += 0.04100976 * Q.n_dr_0p1_0p2 - 0.6971659
    if 0.076787 <= Q.sum_z_dr < 0.08068193:
        z += -219.3965 * Q.sum_z_dr + 16.8468
    if 0.08068193 <= Q.sum_z_dr < 0.08589404:
        z += -69.05431 * Q.sum_z_dr + 4.716901
    if 0.08589404 <= Q.sum_z_dr < 0.09749958:
        z += -176.7897 * Q.sum_z_dr + 13.97073
    if 0.09749958 <= Q.sum_z_dr < 0.1207452:
        z += -201.2401 * Q.sum_z_dr + 16.35464
    if Q.sum_z_dr >= 0.1207452:
        z += -258.7207 * Q.sum_z_dr + 23.29513
    if Q.sum_z_dr2_top10 >= 0.0001295334:
        z += 25.00636 * Q.sum_z_dr2_top10 - 0.003239158
    if Q.e3 < 3.376709e-05:
        z += 50068.54 * Q.e3 - 2.654688
    if 3.376709e-05 <= Q.e3 < 7.876005e-05:
        z += 21426.0 * Q.e3 - 1.687512
    if Q.psi_0p3 >= 0.9924477:
        z += 40.48257 * Q.psi_0p3 - 40.17683
    if Q.e2 < 0.03480688:
        z += -49.80053 * Q.e2 + 1.733401
    if Q.tau1 < 0.1507173:
        z += 56.85628 * Q.tau1 - 8.952862
    if 0.1507173 <= Q.tau1 < 0.1751567:
        z += 15.69748 * Q.tau1 - 2.74952
    if Q.z_dr_0_0p05 >= 0.6283153:
        z += 2.011032 * Q.z_dr_0_0p05 - 1.263562
    if Q.dr_6 < 0.05347848:
        z += -13.84532 * Q.dr_6 + 0.7404268
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -4.29082 * Q.z_dr_0p1_0p2 + 0.516373
    if Q.LHA < 0.2941033:
        z += 5.049824 * Q.LHA - 2.392239
    if 0.2941033 <= Q.LHA < 0.302389:
        z += 2.38381 * Q.LHA - 1.608155
    if 0.302389 <= Q.LHA < 0.3332345:
        z += 28.76649 * Q.LHA - 9.585989
    if Q.sd_rg < 0.15159:
        z += -11.87973 * Q.sd_rg + 3.584289
    if 0.15159 <= Q.sd_rg < 0.1596365:
        z += 1.639044 * Q.sd_rg + 1.534977
    if 0.1596365 <= Q.sd_rg < 0.2042612:
        z += 31.24925 * Q.sd_rg - 3.191892
    if 0.2042612 <= Q.sd_rg < 0.3017146:
        z += -2.275135 * Q.sd_rg + 3.655839
    if Q.sd_rg >= 0.3017146:
        z += 9.6046 * Q.sd_rg + 0.07154959
    if Q.tau21_b2 < 0.342495 and Q.sum_pt_top50 < 1156.659:
        z += -0.03191207 * (0.342495 - Q.tau21_b2) * (1156.659 - Q.sum_pt_top50)
    if Q.psi_0p3 > 0.9924477 and Q.dr_6 < 0.05347848:
        z += -2682.618 * (Q.psi_0p3 - 0.9924477) * (0.05347848 - Q.dr_6)
    if Q.sum_z_dr > 0.076787 and Q.max_dr < 0.4021783:
        z += 320.7562 * (Q.sum_z_dr - 0.076787) * (0.4021783 - Q.max_dr)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.7722356
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -6.920714 * Q.z_dr_0p1_0p2 + 0.8328642
    if Q.sum_z_dr2_top5 < 0.008329695:
        z += -99.17749 * Q.sum_z_dr2_top5 + 0.8261183
    if 6.915514 <= Q.log_sum_pt < 7.017258:
        z += -7.185398 * Q.log_sum_pt + 49.69072
    if Q.log_sum_pt >= 7.017258:
        z += -0.5433183 * Q.log_sum_pt + 3.081532
    if Q.sum_pt >= 907.9372:
        z += 0.008405035 * Q.sum_pt - 7.631244
    if Q.n_dr_0p2_0p4 >= 8.0:
        z += -0.1012281 * Q.n_dr_0p2_0p4 + 0.8098247
    if Q.sum_z_dr2_top3 < 0.001155057:
        z += 410.9258 * Q.sum_z_dr2_top3 - 0.4746425
    if Q.sum_z_dr2_top10 < 0.007678544:
        z += -48.26233 * Q.sum_z_dr2_top10 + 0.3705844
    if 0.1512157 <= Q.sj2_dr < 0.2232169:
        z += -8.54408 * Q.sj2_dr + 1.291999
    if 0.2232169 <= Q.sj2_dr < 0.2780918:
        z += 9.719731 * Q.sj2_dr - 2.784792
    if Q.sj2_dr >= 0.2780918:
        z += -2.383284 * Q.sj2_dr + 0.5809564
    if Q.psi_0p1 >= 0.707925:
        z += -2.076494 * Q.psi_0p1 + 1.470002
    if Q.sum_pt_top30 >= 886.3438:
        z += 0.003875245 * Q.sum_pt_top30 - 3.434799
    if Q.tau1 < 0.07708632:
        z += 21.50908 * Q.tau1 - 1.658056
    if Q.sum_pt_top40 >= 1001.523:
        z += -0.01098966 * Q.sum_pt_top40 + 11.0064
    if Q.soft5_z < 0.0004140594:
        z += -2794.394 * Q.soft5_z + 1.157045
    if Q.sum_z_dr < 0.08068193:
        z += -12.12446 * Q.sum_z_dr + 0.9782252
    if Q.soft5_pt < 0.5297852:
        z += 2.032081 * Q.soft5_pt - 1.076566
    if Q.z_dr_0p2_0p4 < 0.02647293:
        z += 19.16521 * Q.z_dr_0p2_0p4 - 0.5073591
    if Q.psi_0p3 > 0.9896594 and Q.tau21_b2 < 0.4299592:
        z += -94.85575 * (Q.psi_0p3 - 0.9896594) * (0.4299592 - Q.tau21_b2)
    if Q.psi_0p1 > 0.707925 and Q.dr_max_012 < 0.1828389:
        z += 9.246332 * (Q.psi_0p1 - 0.707925) * (0.1828389 - Q.dr_max_012)
    if Q.sum_z_dr < 0.08068193 and Q.dr1_11 > 0.1911348:
        z += -148.6764 * (0.08068193 - Q.sum_z_dr) * (Q.dr1_11 - 0.1911348)
    if Q.psi_0p1 > 0.707925 and Q.dr1_11 > 0.2195171:
        z += 38.08248 * (Q.psi_0p1 - 0.707925) * (Q.dr1_11 - 0.2195171)
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
