"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.0% (the network: 65.8%); same class as the network for 88.6% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.ptdr0_7                pT7 · ΔR(0, 7) [GeV]
  Q.z_2                    pT of particle 2 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_nremoved            number of branches removed by soft drop
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_2               |Δη| of particle 2
  Q.abseta_3               |Δη| of particle 3
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_4                  ΔR between particle 4 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.centroid_offset        distance of the pT centroid from the jet axis
  Q.pt_dispersion          √(Σ pTᵢ²) / Σ pTᵢ
"""
import math
from types import SimpleNamespace

CLASSES = ['g', 'q', 'W', 'Z', 't']
W = [[-0.15625, 0.0, 0.34375, 0.0, 0.015625], [0.390625, 0.0, -0.03125, 0.125, 0.0], [0.4296875, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.5, -0.5625, 0.0625], [-0.03125, -0.09375, 0.0, 0.078125, 0.125], [-0.1875, 0.046875, 0.0, 0.015625, -0.25], [0.109375, 0.125, -0.3125, -0.375, 0.0], [0.0, 0.0, 0.21875, 0.46875, 0.0], [0.0, 0.0625, -0.25, 0.0, 0.1875], [0.171875, 0.25390625, -0.03125, -0.03125, 0.0], [0.0, -0.125, 0.0, 0.0, 0.375], [0.0, 0.0, 0.375, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -0.5], [0.0, 0.0, 0.0703125, 0.0546875, -0.40625], [0.0, 0.0, -0.75, 0.375, 0.0], [0.0, 0.0625, -0.6875, -0.15625, 0.0]]
B = [-0.4375, 0.03125, -0.125, -0.09375, 1.34375]
INT_BITS = [3, 5, 4, 4, 5, 5, 4, 4, 3, 4, 5, 4, 5, 4, 3, 3]
FRAC_BITS = [3, 3, 4, 3, 2, 3, 3, 3, 3, 2, 4, 4, 3, 3, 4, 3]


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
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        pt_1=pt[1],
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        pt_4=pt[4],
        pt_5=pt[5],
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        ptdr0_7=pt[7] * math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        z_2=z[2],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        z_top3_slots=sum(pt[:3]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_nremoved=softdrop("removed"),
        abseta_0=abs(eta[0]),
        abseta_2=abs(eta[2]),
        abseta_3=abs(eta[3]),
        abseta_4=abs(eta[4]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_4=math.sqrt(dist2(1, 4)) if pt[4] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        eta_0=eta[0],
        phi_0=phi[0],
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -1.83426
    if Q.planar_flow < 0.04505724:
        z += 13.17395 * Q.planar_flow + 0.2710243
    if 0.04505724 <= Q.planar_flow < 0.1484197:
        z += -8.364797 * Q.planar_flow + 1.241501
    if Q.lam1 < 0.004183811:
        z += 567.3792 * Q.lam1 + 0.4962114
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += -187.7386 * Q.lam1 + 3.655481
    if 0.00595415 <= Q.lam1 < 0.01200373:
        z += -359.8632 * Q.lam1 + 4.680337
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += -81.40771 * Q.lam1 + 1.337834
    if Q.sum_pt < 763.825:
        z += 0.006753675 * Q.sum_pt - 5.158626
    if Q.tau1 < 0.05356915:
        z += -28.49377 * Q.tau1 + 1.526387
    if Q.sum_z_dr2_top3 < 0.007929074:
        z += 69.61295 * Q.sum_z_dr2_top3 - 0.5519662
    if Q.sj3_dr_max < 0.1426152:
        z += 7.144762 * Q.sj3_dr_max - 2.469558
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 23.20144 * Q.sj3_dr_max - 4.759483
    if 0.169029 <= Q.sj3_dr_max < 0.1986272:
        z += 3.398809 * Q.sj3_dr_max - 1.412264
    if 0.1986272 <= Q.sj3_dr_max < 0.3456459:
        z += -4.52245 * Q.sj3_dr_max + 0.1611137
    if Q.sj3_dr_max >= 0.3456459:
        z += -11.66721 * Q.sj3_dr_max + 2.630671
    if Q.centroid_offset >= 0.006789738:
        z += -41.88072 * Q.centroid_offset + 0.2843591
    if Q.sum_z_dr < 0.08065885:
        z += 39.90657 * Q.sum_z_dr - 3.218818
    if 6.766778 <= Q.log_sum_pt < 6.842717:
        z += -10.79228 * Q.log_sum_pt + 73.02897
    if Q.log_sum_pt >= 6.842717:
        z += -22.9683 * Q.log_sum_pt + 156.346
    if Q.sum_pt_top5 >= 791.125:
        z += 0.01414342 * Q.sum_pt_top5 - 11.18921
    if Q.tau2 < 0.01713288:
        z += 50.07465 * Q.tau2 - 0.8579229
    if Q.e3 < 8.147744e-05:
        z += -17215.78 * Q.e3 + 4.59179
    if 8.147744e-05 <= Q.e3 < 0.0005116989:
        z += -7412.676 * Q.e3 + 3.793058
    if 0.1546891 <= Q.LHA < 0.3033137:
        z += 4.395849 * Q.LHA - 0.6799899
    if Q.LHA >= 0.3033137:
        z += -0.6945233 * Q.LHA + 0.8639899
    if Q.e2 < 0.02045966:
        z += -21.67403 * Q.e2 + 0.4434433
    if Q.e2 >= 0.06344108:
        z += -52.75632 * Q.e2 + 3.346917
    if Q.C2_b2 < 0.001563465:
        z += 522.2257 * Q.C2_b2 - 0.8164819
    if Q.pt_6 < 27.57812:
        z += 0.04195492 * Q.pt_6 - 1.157038
    if Q.n_dr_0p2_0p4 < 1.0:
        z += -0.1931638 * Q.n_dr_0p2_0p4 + 0.1931638
    if Q.lam1 < 0.01643375 and Q.centroid_offset > 0.01837778:
        z += -5737.711 * (0.01643375 - Q.lam1) * (Q.centroid_offset - 0.01837778)
    if Q.sum_pt < 763.825 and Q.centroid_offset > 0.01096064:
        z += 0.1390751 * (763.825 - Q.sum_pt) * (Q.centroid_offset - 0.01096064)
    if Q.lam1 < 0.004183811 and Q.sum_pt < 763.825:
        z += 2.901892 * (0.004183811 - Q.lam1) * (763.825 - Q.sum_pt)
    if Q.sum_pt < 763.825 and Q.e3 < 0.0005116989:
        z += -5.252614 * (763.825 - Q.sum_pt) * (0.0005116989 - Q.e3)
    if Q.lam1 < 0.00595415 and Q.D2 < 0.7459513:
        z += -913.4998 * (0.00595415 - Q.lam1) * (0.7459513 - Q.D2)
    if Q.sum_z_dr2_top3 < 0.007929074 and Q.phi_0 < 2.482534e-05:
        z += -1147.668 * (0.007929074 - Q.sum_z_dr2_top3) * (2.482534e-05 - Q.phi_0)
    if Q.sj3_dr_max > 0.1426152 and Q.D2 > 0.2568137:
        z += 0.3037389 * (Q.sj3_dr_max - 0.1426152) * (Q.D2 - 0.2568137)
    if Q.sum_pt_top5 > 791.125 and Q.eta_0 < 0.04037476:
        z += 0.04173533 * (Q.sum_pt_top5 - 791.125) * (0.04037476 - Q.eta_0)
    if Q.planar_flow < 0.1484197 and Q.z_7 < 0.02807091:
        z += 414.3186 * (0.1484197 - Q.planar_flow) * (0.02807091 - Q.z_7)
    if Q.tau2 < 0.01713288 and Q.z_dr_0p05_0p1 < 0.2919447:
        z += 100.1228 * (0.01713288 - Q.tau2) * (0.2919447 - Q.z_dr_0p05_0p1)
    if Q.sum_pt < 763.825 and Q.z_7 > 0.01685855:
        z += 0.03749556 * (763.825 - Q.sum_pt) * (Q.z_7 - 0.01685855)
    if Q.planar_flow < 0.04505724 and Q.tau21_b2 < 0.05932655:
        z += 561.3345 * (0.04505724 - Q.planar_flow) * (0.05932655 - Q.tau21_b2)
    if Q.log_sum_pt > 6.842717 and Q.tau21_b2 < 0.04019753:
        z += -101.4505 * (Q.log_sum_pt - 6.842717) * (0.04019753 - Q.tau21_b2)
    if Q.log_sum_pt > 6.842717 and Q.n_pt_above_50 > 5.0:
        z += 1.43981 * (Q.log_sum_pt - 6.842717) * (Q.n_pt_above_50 - 5.0)
    if Q.planar_flow < 0.1484197 and Q.dr_3 < 0.02358801:
        z += -393.1251 * (0.1484197 - Q.planar_flow) * (0.02358801 - Q.dr_3)
    if Q.lam1 < 0.00595415 and Q.n_dr_0p2_0p4 < 1.0:
        z += -184.7943 * (0.00595415 - Q.lam1) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.planar_flow < 0.1484197 and Q.C3 > 0.04857476:
        z += -210.0041 * (0.1484197 - Q.planar_flow) * (Q.C3 - 0.04857476)
    if Q.e3 < 8.147744e-05 and Q.D2_b2 < 0.05744392:
        z += -138800.4 * (8.147744e-05 - Q.e3) * (0.05744392 - Q.D2_b2)
    if Q.sum_pt < 763.825 and Q.D2_b2 < 0.03885671:
        z += 0.07596058 * (763.825 - Q.sum_pt) * (0.03885671 - Q.D2_b2)
    if Q.sum_pt < 763.825 and Q.D2_b2 < 0.380911:
        z += -0.004637691 * (763.825 - Q.sum_pt) * (0.380911 - Q.D2_b2)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.42198
    if Q.lam1 < 0.005433361:
        z += 139.6264 * Q.lam1 - 0.6077407
    if 0.005433361 <= Q.lam1 < 0.008375572:
        z += 36.84237 * Q.lam1 - 0.04927819
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -71.46822 * Q.lam1 + 0.8578849
    if 34.53125 <= Q.pt_7 < 41.65625:
        z += 0.04893417 * Q.pt_7 - 1.689758
    if 41.65625 <= Q.pt_7 < 53.4375:
        z += 0.119881 * Q.pt_7 - 4.645136
    if Q.pt_7 >= 53.4375:
        z += 0.01928623 * Q.pt_7 + 0.7303961
    if 6.377723 <= Q.log_sum_pt < 6.572938:
        z += 5.767262 * Q.log_sum_pt - 36.782
    if Q.log_sum_pt >= 6.572938:
        z += 8.880797 * Q.log_sum_pt - 57.24707
    if Q.z_7 < 0.04939969:
        z += 75.15299 * Q.z_7 - 5.361051
    if 0.04939969 <= Q.z_7 < 0.06473447:
        z += 50.94309 * Q.z_7 - 4.165089
    if 0.06473447 <= Q.z_7 < 0.08670959:
        z += -1.636135 * Q.z_7 - 0.7614005
    if Q.z_7 >= 0.08670959:
        z += -24.2099 * Q.z_7 + 1.195962
    if Q.tau1 < 0.07283629:
        z += -18.78664 * Q.tau1 + 1.741221
    if 0.07283629 <= Q.tau1 < 0.1027642:
        z += -12.45899 * Q.tau1 + 1.280339
    if Q.sj3_dr_min >= 0.02913153:
        z += -2.83529 * Q.sj3_dr_min + 0.08259634
    if Q.sj2_dr < 0.1591713:
        z += -3.289022 * Q.sj2_dr + 0.523518
    if Q.max_dr < 0.1117619:
        z += 7.480844 * Q.max_dr - 0.836073
    if 763.825 <= Q.sum_pt < 988.4078:
        z += 0.004780934 * Q.sum_pt - 3.651797
    if Q.sum_pt >= 988.4078:
        z += -0.006921203 * Q.sum_pt + 7.914687
    if Q.centroid_offset < 0.02076709:
        z += 25.55285 * Q.centroid_offset - 0.5306583
    if Q.sum_z_dr < 0.06108601:
        z += -19.61098 * Q.sum_z_dr + 0.3785369
    if 0.06108601 <= Q.sum_z_dr < 0.07608178:
        z += 9.822033 * Q.sum_z_dr - 1.419408
    if 0.07608178 <= Q.sum_z_dr < 0.1019409:
        z += 25.99198 * Q.sum_z_dr - 2.649647
    if Q.sum_pt_top5 >= 839.9547:
        z += 0.009589939 * Q.sum_pt_top5 - 8.055115
    if Q.zdr_0 < 0.0211821:
        z += 12.33269 * Q.zdr_0 - 0.2612324
    if Q.z_6 < 0.04737278:
        z += 70.43957 * Q.z_6 - 3.336918
    if Q.lam2 < 0.001130645:
        z += -738.9806 * Q.lam2 + 0.8355245
    if Q.planar_flow < 0.1950135:
        z += 2.42066 * Q.planar_flow - 0.4720614
    if Q.sj3_dr_max < 0.169029:
        z += -5.722099 * Q.sj3_dr_max + 0.4081314
    if 0.169029 <= Q.sj3_dr_max < 0.3456459:
        z += 3.165437 * Q.sj3_dr_max - 1.09412
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += 13167.83 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.pt_7 > 34.53125 and Q.e3 < 2.955458e-05:
        z += -711.571 * (Q.pt_7 - 34.53125) * (2.955458e-05 - Q.e3)
    if Q.log_sum_pt > 6.377723 and Q.e3 < 1.340118e-05:
        z += 36705.13 * (Q.log_sum_pt - 6.377723) * (1.340118e-05 - Q.e3)
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += 2.205953 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.z_7 < 0.06473447 and Q.e3 < 0.0005116989:
        z += 49979.48 * (0.06473447 - Q.z_7) * (0.0005116989 - Q.e3)
    if Q.pt_7 > 34.53125 and Q.n_dr_0_0p05 < 5.0:
        z += 0.004360749 * (Q.pt_7 - 34.53125) * (5.0 - Q.n_dr_0_0p05)
    if Q.lam1 < 0.008375572 and Q.lam2 < 0.001130645:
        z += -355813.8 * (0.008375572 - Q.lam1) * (0.001130645 - Q.lam2)
    if Q.z_7 < 0.06473447 and Q.D2_b2 < 0.716559:
        z += -32.02286 * (0.06473447 - Q.z_7) * (0.716559 - Q.D2_b2)
    if Q.log_sum_pt > 6.377723 and Q.D2_b2 < 0.716559:
        z += 2.391046 * (Q.log_sum_pt - 6.377723) * (0.716559 - Q.D2_b2)
    if Q.lam1 < 0.01200373 and Q.n_pt_above_50 > 6.0:
        z += -76.06849 * (0.01200373 - Q.lam1) * (Q.n_pt_above_50 - 6.0)
    if Q.log_sum_pt > 6.377723 and Q.zdr_5 < 0.0155217:
        z += -39.31197 * (Q.log_sum_pt - 6.377723) * (0.0155217 - Q.zdr_5)
    if Q.log_sum_pt > 6.377723 and Q.sum_z_dr2_top2 < 0.004007842:
        z += -713.0535 * (Q.log_sum_pt - 6.377723) * (0.004007842 - Q.sum_z_dr2_top2)
    if Q.z_6 < 0.04737278 and Q.sum_z_dr2_top2 < 0.004007842:
        z += 11428.7 * (0.04737278 - Q.z_6) * (0.004007842 - Q.sum_z_dr2_top2)
    if Q.log_sum_pt > 6.572938 and Q.sum_z_dr2_top3 < 0.006756161:
        z += -1458.787 * (Q.log_sum_pt - 6.572938) * (0.006756161 - Q.sum_z_dr2_top3)
    if Q.log_sum_pt > 6.572938 and Q.dr_max_012 < 0.1206357:
        z += 34.82871 * (Q.log_sum_pt - 6.572938) * (0.1206357 - Q.dr_max_012)
    if Q.lam1 < 0.01200373 and Q.dr_0 > 0.005517012:
        z += -594.8057 * (0.01200373 - Q.lam1) * (Q.dr_0 - 0.005517012)
    if Q.n_pt_above_50 > 6.0 and Q.e3 < 8.147744e-05:
        z += 6299.954 * (Q.n_pt_above_50 - 6.0) * (8.147744e-05 - Q.e3)
    if Q.pt_7 > 34.53125 and Q.e3 < 8.147744e-05:
        z += -834.2264 * (Q.pt_7 - 34.53125) * (8.147744e-05 - Q.e3)
    if Q.sum_pt > 988.4078 and Q.sum_z_dr2_top2 < 0.0095303:
        z += 0.7464111 * (Q.sum_pt - 988.4078) * (0.0095303 - Q.sum_z_dr2_top2)
    if Q.sum_pt_top5 > 839.9547 and Q.e3 < 3.892127e-05:
        z += -160.9245 * (Q.sum_pt_top5 - 839.9547) * (3.892127e-05 - Q.e3)
    if Q.lam1 < 0.005433361 and Q.dr_max_012 < 0.1550922:
        z += -511.0117 * (0.005433361 - Q.lam1) * (0.1550922 - Q.dr_max_012)
    if Q.z_7 < 0.06473447 and Q.sum_z_dr2_top3 < 0.007929074:
        z += 5134.338 * (0.06473447 - Q.z_7) * (0.007929074 - Q.sum_z_dr2_top3)
    if Q.log_sum_pt > 6.377723 and Q.zdr_6 < 0.01058212:
        z += -53.98867 * (Q.log_sum_pt - 6.377723) * (0.01058212 - Q.zdr_6)
    if Q.z_6 < 0.04737278 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 229.2938 * (0.04737278 - Q.z_6) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.sum_pt_top5 > 839.9547 and Q.n_dr_0p2_0p4 < 2.0:
        z += -0.00149138 * (Q.sum_pt_top5 - 839.9547) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.n_pt_above_50 > 6.0 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.1076637 * (Q.n_pt_above_50 - 6.0) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.log_sum_pt > 6.377723 and Q.dr_2 < 0.1147987:
        z += -15.68253 * (Q.log_sum_pt - 6.377723) * (0.1147987 - Q.dr_2)
    if Q.pt_7 > 34.53125 and Q.dr_3 < 0.1212677:
        z += -0.1486093 * (Q.pt_7 - 34.53125) * (0.1212677 - Q.dr_3)
    return max(0.0, z)


def neuron_2(Q):
    z = 6.93597
    if Q.pt_7 < 20.125:
        z += 0.1143322 * Q.pt_7 - 5.757473
    if 20.125 <= Q.pt_7 < 23.21641:
        z += 0.2525324 * Q.pt_7 - 8.538752
    if 23.21641 <= Q.pt_7 < 53.4375:
        z += 0.2101215 * Q.pt_7 - 7.554122
    if Q.pt_7 >= 53.4375:
        z += 0.1208051 * Q.pt_7 - 2.781279
    if Q.log_sum_pt < 6.080494:
        z += -6.749154 * Q.log_sum_pt + 44.65195
    if 6.080494 <= Q.log_sum_pt < 6.605974:
        z += -11.38715 * Q.log_sum_pt + 72.85328
    if 6.605974 <= Q.log_sum_pt < 6.701242:
        z += -5.343498 * Q.log_sum_pt + 32.92904
    if 6.701242 <= Q.log_sum_pt < 6.896095:
        z += -4.638 * Q.log_sum_pt + 28.20134
    if Q.log_sum_pt >= 6.896095:
        z += -7.13659 * Q.log_sum_pt + 45.43185
    if Q.z_7 < 0.02320757:
        z += -29.1334 * Q.z_7 + 1.707616
    if 0.02320757 <= Q.z_7 < 0.02807091:
        z += -140.6892 * Q.z_7 + 4.296556
    if 0.02807091 <= Q.z_7 < 0.0586137:
        z += -113.3296 * Q.z_7 + 3.528546
    if Q.z_7 >= 0.0586137:
        z += -84.19621 * Q.z_7 + 1.82093
    if Q.sum_z_dr < 0.007673833:
        z += 151.3091 * Q.sum_z_dr - 0.6701584
    if 0.007673833 <= Q.sum_z_dr < 0.01517359:
        z += 27.50051 * Q.sum_z_dr + 0.2799282
    if 0.01517359 <= Q.sum_z_dr < 0.1019409:
        z += -8.035394 * Q.sum_z_dr + 0.8191355
    if Q.LHA >= 0.111565:
        z += -8.815741 * Q.LHA + 0.9835285
    if Q.sum_pt < 641.7188:
        z += -0.003879971 * Q.sum_pt + 2.48985
    if 937.0312 <= Q.sum_pt < 988.4078:
        z += 0.01063183 * Q.sum_pt - 9.962356
    if Q.sum_pt >= 988.4078:
        z += 0.02546843 * Q.sum_pt - 24.62697
    if Q.C2 < 0.0423228:
        z += 14.27942 * Q.C2 - 0.604345
    if Q.sum_pt_top5 < 324.5922:
        z += -0.01140176 * Q.sum_pt_top5 + 2.786545
    if 324.5922 <= Q.sum_pt_top5 < 367.5938:
        z += -4.127156e-06 * Q.sum_pt_top5 - 0.913039
    if 367.5938 <= Q.sum_pt_top5 < 752.1:
        z += 0.002378521 * Q.sum_pt_top5 - 1.788885
    if Q.lam1 < 0.004183811:
        z += -267.3927 * Q.lam1 + 1.986702
    if 0.004183811 <= Q.lam1 < 0.01200373:
        z += -110.9963 * Q.lam1 + 1.332369
    if Q.mean_phi2 < 8.836697e-05:
        z += -64729.14 * Q.mean_phi2 + 5.719918
    if Q.e2 < 0.08000524:
        z += 11.20194 * Q.e2 - 0.8962141
    if Q.lam1 < 0.00733008 and Q.max_dr < 0.2507612:
        z += 1301.109 * (0.00733008 - Q.lam1) * (0.2507612 - Q.max_dr)
    if Q.lam1 < 0.00733008 and Q.pt_6 < 62.25:
        z += -3.838801 * (0.00733008 - Q.lam1) * (62.25 - Q.pt_6)
    if Q.pt_7 < 53.4375 and Q.pt1_dr01 > 17.82896:
        z += 0.0006144634 * (53.4375 - Q.pt_7) * (Q.pt1_dr01 - 17.82896)
    if Q.sum_pt_top5 < 752.1 and Q.D2_b2 > 0.1830092:
        z += -0.0006365911 * (752.1 - Q.sum_pt_top5) * (Q.D2_b2 - 0.1830092)
    if Q.log_sum_pt < 6.605974 and Q.D2_b2 > 0.01618699:
        z += -1.820971 * (6.605974 - Q.log_sum_pt) * (Q.D2_b2 - 0.01618699)
    if Q.z_7 < 0.0586137 and Q.absphi_0 < 0.07897949:
        z += -271.0544 * (0.0586137 - Q.z_7) * (0.07897949 - Q.absphi_0)
    if Q.log_sum_pt < 6.701242 and Q.mean_phi2 < 8.836697e-05:
        z += -95243.01 * (6.701242 - Q.log_sum_pt) * (8.836697e-05 - Q.mean_phi2)
    if Q.log_sum_pt > 6.080494 and Q.mean_phi2 < 8.836697e-05:
        z += -101707.5 * (Q.log_sum_pt - 6.080494) * (8.836697e-05 - Q.mean_phi2)
    if Q.log_sum_pt > 6.896095 and Q.absphi_0 < 0.1056549:
        z += 532.2308 * (Q.log_sum_pt - 6.896095) * (0.1056549 - Q.absphi_0)
    if Q.sum_pt > 937.0312 and Q.mean_phi2 < 8.836697e-05:
        z += 148.868 * (Q.sum_pt - 937.0312) * (8.836697e-05 - Q.mean_phi2)
    if Q.z_7 < 0.0586137 and Q.absphi_0 < 0.006721497:
        z += 1402.133 * (0.0586137 - Q.z_7) * (0.006721497 - Q.absphi_0)
    if Q.sum_pt > 868.5094 and Q.absphi_0 < 0.01452637:
        z += -0.1887115 * (Q.sum_pt - 868.5094) * (0.01452637 - Q.absphi_0)
    if Q.z_7 > 0.02320757 and Q.sj3_dr_min < 0.1278212:
        z += -108.0398 * (Q.z_7 - 0.02320757) * (0.1278212 - Q.sj3_dr_min)
    if Q.log_sum_pt < 6.701242 and Q.D2_b2 > 0.380911:
        z += 1.738958 * (6.701242 - Q.log_sum_pt) * (Q.D2_b2 - 0.380911)
    if Q.sum_pt > 988.4078 and Q.absphi_0 < 0.1056549:
        z += -0.5082162 * (Q.sum_pt - 988.4078) * (0.1056549 - Q.absphi_0)
    if Q.sum_pt > 868.5094 and Q.abseta_2 < 0.01558685:
        z += -0.1399871 * (Q.sum_pt - 868.5094) * (0.01558685 - Q.abseta_2)
    if Q.centroid_offset < 0.005576073 and Q.dr_7 < 0.03607145:
        z += -2856.618 * (0.005576073 - Q.centroid_offset) * (0.03607145 - Q.dr_7)
    if Q.log_sum_pt > 6.896095 and Q.n_pt_above_50 > 5.0:
        z += -3.546459 * (Q.log_sum_pt - 6.896095) * (Q.n_pt_above_50 - 5.0)
    if Q.sum_pt > 868.5094 and Q.D2_b2 < 4.721224:
        z += 0.00270417 * (Q.sum_pt - 868.5094) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 4.721224:
        z += -4.528013 * (Q.log_sum_pt - 6.896095) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt < 6.701242 and Q.lam2 < 0.0003061234:
        z += 6229.26 * (6.701242 - Q.log_sum_pt) * (0.0003061234 - Q.lam2)
    if Q.pt_7 < 53.4375 and Q.D2_b2 < 0.1236856:
        z += 0.1226297 * (53.4375 - Q.pt_7) * (0.1236856 - Q.D2_b2)
    if Q.log_sum_pt > 6.080494 and Q.lam2 < 0.0001947983:
        z += -4902.05 * (Q.log_sum_pt - 6.080494) * (0.0001947983 - Q.lam2)
    if Q.sj2_dr < 0.1294903 and Q.D2_b2 < 4.721224:
        z += -1.159379 * (0.1294903 - Q.sj2_dr) * (4.721224 - Q.D2_b2)
    if Q.sum_z_dr < 0.01517359 and Q.n_pt_above_50 < 7.0:
        z += 16.65223 * (0.01517359 - Q.sum_z_dr) * (7.0 - Q.n_pt_above_50)
    if Q.sum_pt > 937.0312 and Q.absphi_0 > 0.003417683:
        z += -0.1461764 * (Q.sum_pt - 937.0312) * (Q.absphi_0 - 0.003417683)
    if Q.log_sum_pt < 6.701242 and Q.absphi_0 < 0.1056549:
        z += 2.955171 * (6.701242 - Q.log_sum_pt) * (0.1056549 - Q.absphi_0)
    if Q.pt_7 > 20.125 and Q.sd_rg > 0.1667615:
        z += -0.08807488 * (Q.pt_7 - 20.125) * (Q.sd_rg - 0.1667615)
    if Q.sum_z_dr < 0.1019409 and Q.centroid_offset > 0.01837778:
        z += -753.735 * (0.1019409 - Q.sum_z_dr) * (Q.centroid_offset - 0.01837778)
    if Q.sum_pt < 641.7188 and Q.lam2 < 0.0003061234:
        z += -14.01718 * (641.7188 - Q.sum_pt) * (0.0003061234 - Q.lam2)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.4424923
    if 0.06663269 <= Q.sum_z_dr < 0.1019409:
        z += 32.87845 * Q.sum_z_dr - 2.190779
    if 0.1019409 <= Q.sum_z_dr < 0.1245537:
        z += -60.09017 * Q.sum_z_dr + 7.286528
    if Q.sum_z_dr >= 0.1245537:
        z += -7.8419 * Q.sum_z_dr + 0.7788102
    if Q.sj2_dr >= 0.1682655:
        z += 6.774813 * Q.sj2_dr - 1.139968
    if Q.N2 < 0.2654572:
        z += 1.382659 * Q.N2 - 0.3670369
    if 0.1876504 <= Q.sd_rg < 0.2787955:
        z += 1.093913 * Q.sd_rg - 0.2052733
    if Q.sd_rg >= 0.2787955:
        z += -4.934969 * Q.sd_rg + 1.475552
    if 0.2931906 <= Q.LHA < 0.3255822:
        z += -8.524812 * Q.LHA + 2.499395
    if Q.LHA >= 0.3255822:
        z += 13.1299 * Q.LHA - 4.550993
    if Q.tau1 >= 0.1136369:
        z += 20.28491 * Q.tau1 - 2.305114
    if 0.213399 <= Q.sj3_dr_max < 0.2623172:
        z += 15.05937 * Q.sj3_dr_max - 3.213655
    if Q.sj3_dr_max >= 0.2623172:
        z += 0.5064278 * Q.sj3_dr_max + 0.6038336
    if Q.mean_eta < -0.006779839:
        z += -11.73742 * Q.mean_eta - 0.07957778
    if 0.00595415 <= Q.lam1 < 0.01200373:
        z += 306.8574 * Q.lam1 - 1.827075
    if Q.lam1 >= 0.01200373:
        z += -80.05994 * Q.lam1 + 2.817375
    if Q.lam2 < 0.003408389:
        z += 267.61 * Q.lam2 - 0.912119
    if Q.sj2_dr > 0.1682655 and Q.log_sum_pt > 6.080494:
        z += -14.10669 * (Q.sj2_dr - 0.1682655) * (Q.log_sum_pt - 6.080494)
    if Q.sj2_dr > 0.1682655 and Q.C2 < 0.09482124:
        z += 34.02536 * (Q.sj2_dr - 0.1682655) * (0.09482124 - Q.C2)
    if Q.sum_z_dr > 0.06663269 and Q.n_pt_above_50 > 3.0:
        z += -3.11367 * (Q.sum_z_dr - 0.06663269) * (Q.n_pt_above_50 - 3.0)
    if Q.sd_rg > 0.1876504 and Q.lam2 < 0.000537286:
        z += 6277.029 * (Q.sd_rg - 0.1876504) * (0.000537286 - Q.lam2)
    if Q.sj2_dr > 0.1682655 and Q.n_dr_0_0p05 > 2.0:
        z += -1.021615 * (Q.sj2_dr - 0.1682655) * (Q.n_dr_0_0p05 - 2.0)
    if Q.N3 < 1.138757 and Q.log_sum_pt > 6.080494:
        z += -8.778383 * (1.138757 - Q.N3) * (Q.log_sum_pt - 6.080494)
    if Q.mean_eta < -0.006779839 and Q.log_sum_pt < 6.267538:
        z += 51.54001 * (-0.006779839 - Q.mean_eta) * (6.267538 - Q.log_sum_pt)
    if Q.sd_rg > 0.1876504 and Q.z_dr_0_0p05 < 0.1515405:
        z += 18.42488 * (Q.sd_rg - 0.1876504) * (0.1515405 - Q.z_dr_0_0p05)
    if Q.LHA > 0.3255822 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -125.4015 * (Q.LHA - 0.3255822) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.sj3_dr_max > 0.213399 and Q.n_dr_0p05_0p1 > 4.0:
        z += 8.22669 * (Q.sj3_dr_max - 0.213399) * (Q.n_dr_0p05_0p1 - 4.0)
    if Q.sj3_dr_max > 0.2623172 and Q.n_dr_0p05_0p1 > 4.0:
        z += -9.936808 * (Q.sj3_dr_max - 0.2623172) * (Q.n_dr_0p05_0p1 - 4.0)
    if Q.mean_eta < -0.006779839 and Q.eta_0 < -0.03967285:
        z += -169.2996 * (-0.006779839 - Q.mean_eta) * (-0.03967285 - Q.eta_0)
    if Q.LHA > 0.3255822 and Q.eccentricity > 0.7117266:
        z += 97.77332 * (Q.LHA - 0.3255822) * (Q.eccentricity - 0.7117266)
    if Q.sum_z_dr > 0.1245537 and Q.eccentricity > 0.7792127:
        z += -198.1694 * (Q.sum_z_dr - 0.1245537) * (Q.eccentricity - 0.7792127)
    if Q.sj2_dr > 0.1682655 and Q.eccentricity > 0.9458207:
        z += 139.7046 * (Q.sj2_dr - 0.1682655) * (Q.eccentricity - 0.9458207)
    if Q.N3 < 1.138757 and Q.z_7 < 0.08670959:
        z += 69.44087 * (1.138757 - Q.N3) * (0.08670959 - Q.z_7)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.943396
    if Q.N2 < 0.2233283:
        z += -19.1831 * Q.N2 + 4.284128
    if Q.lam1 < 0.002464291:
        z += 1203.01 * Q.lam1 - 12.28517
    if 0.002464291 <= Q.lam1 < 0.006506576:
        z += 1818.253 * Q.lam1 - 13.80131
    if 0.006506576 <= Q.lam1 < 0.00733008:
        z += 1088.882 * Q.lam1 - 9.055606
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 1027.277 * Q.lam1 - 8.604035
    if Q.lam1 >= 0.01200373:
        z += 38.26113 * Q.lam1 - 0.4592762
    if Q.e3 < 0.0001869378:
        z += -10030.67 * Q.e3 + 1.875112
    if Q.dr_0 < 0.08082334:
        z += 10.50326 * Q.dr_0 - 0.8489084
    if Q.lam2 < 0.000537286:
        z += 2155.665 * Q.lam2 + 1.529344
    if 0.000537286 <= Q.lam2 < 0.003408389:
        z += -936.0698 * Q.lam2 + 3.19049
    if Q.tau1 < 0.04369778:
        z += 9.921326 * Q.tau1 - 0.4335399
    if Q.sj3_dr_max < 0.07264452:
        z += 1.400982 * Q.sj3_dr_max - 2.208202
    if 0.07264452 <= Q.sj3_dr_max < 0.1426152:
        z += 30.10446 * Q.sj3_dr_max - 4.293353
    if 0.169029 <= Q.sj3_dr_max < 0.3012016:
        z += 11.94894 * Q.sj3_dr_max - 2.019718
    if 0.3012016 <= Q.sj3_dr_max < 0.3456459:
        z += -8.7809 * Q.sj3_dr_max + 4.224143
    if Q.sj3_dr_max >= 0.3456459:
        z += -2.942537 * Q.sj3_dr_max + 2.206137
    if Q.sum_pt < 763.825:
        z += 0.007513319 * Q.sum_pt - 6.902732
    if 763.825 <= Q.sum_pt < 988.4078:
        z += 0.00518237 * Q.sum_pt - 5.122295
    if Q.max_dr < 0.121681:
        z += -11.54368 * Q.max_dr + 2.894707
    if 0.121681 <= Q.max_dr < 0.2507612:
        z += 11.21237 * Q.max_dr + 0.1257289
    if Q.max_dr >= 0.2507612:
        z += 22.75605 * Q.max_dr - 2.768978
    if Q.log_sum_pt >= 6.327379:
        z += 5.714287 * Q.log_sum_pt - 36.15646
    if Q.pt_7 < 25.57812:
        z += 0.07614444 * Q.pt_7 - 1.947632
    if 0.2001708 <= Q.sj2_dr < 0.2179769:
        z += -23.92569 * Q.sj2_dr + 4.789225
    if Q.sj2_dr >= 0.2179769:
        z += -16.43187 * Q.sj2_dr + 3.155744
    if Q.C2 >= 0.01867771:
        z += -41.08223 * Q.C2 + 0.767322
    if Q.LHA < 0.1329373:
        z += -24.67239 * Q.LHA + 8.030837
    if 0.1329373 <= Q.LHA < 0.3127275:
        z += -26.42501 * Q.LHA + 8.263826
    if Q.sum_z_dr2_top3 < 0.01002369:
        z += -79.19546 * Q.sum_z_dr2_top3 + 0.7938307
    if Q.tau21 < 0.33555:
        z += 4.7877 * Q.tau21 - 1.606513
    if Q.sj3_dr_min < 0.2089872:
        z += -29.57756 * Q.sj3_dr_min + 6.181332
    if Q.eccentricity >= 0.7117266:
        z += -10.88505 * Q.eccentricity + 7.747183
    if Q.zdr_1 < 0.009612129:
        z += 41.11642 * Q.zdr_1 - 0.3952164
    if Q.D2 < 1.002471:
        z += -0.690725 * Q.D2 + 0.6924316
    if Q.zdr_0 < 0.01705377:
        z += 23.47784 * Q.zdr_0 - 0.4003857
    if Q.M2 < 0.04435703:
        z += 26.0996 * Q.M2 - 1.157701
    if Q.n_dr_0p2_0p4 < 1.0:
        z += 0.302982 * Q.n_dr_0p2_0p4 - 0.302982
    if Q.N2 < 0.2233283 and Q.planar_flow > 0.08366273:
        z += 13.74719 * (0.2233283 - Q.N2) * (Q.planar_flow - 0.08366273)
    if Q.N2 < 0.2233283 and Q.sum_pt < 840.0195:
        z += -0.0207253 * (0.2233283 - Q.N2) * (840.0195 - Q.sum_pt)
    if Q.N2 < 0.2233283 and Q.mean_phi < -0.02594505:
        z += 355.3669 * (0.2233283 - Q.N2) * (-0.02594505 - Q.mean_phi)
    if Q.N2 < 0.2233283 and Q.sj2_zsoft < 0.2832687:
        z += -34.59816 * (0.2233283 - Q.N2) * (0.2832687 - Q.sj2_zsoft)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1626038:
        z += -16.54596 * (0.2233283 - Q.N2) * (0.1626038 - Q.abseta_7)
    if Q.sj3_dr_max > 0.233678 and Q.log_sum_pt > 6.267538:
        z += -67.91066 * (Q.sj3_dr_max - 0.233678) * (Q.log_sum_pt - 6.267538)
    if Q.sum_pt < 763.825 and Q.dr_1 < 0.06108421:
        z += 0.05007676 * (763.825 - Q.sum_pt) * (0.06108421 - Q.dr_1)
    if Q.sj3_dr_max > 0.233678 and Q.sj2_zsoft < 0.05966518:
        z += 982.9642 * (Q.sj3_dr_max - 0.233678) * (0.05966518 - Q.sj2_zsoft)
    if Q.dr_0 < 0.08082334 and Q.centroid_offset > 0.02355416:
        z += -1341.798 * (0.08082334 - Q.dr_0) * (Q.centroid_offset - 0.02355416)
    if Q.log_sum_pt > 6.327379 and Q.zdr_5 < 0.006197061:
        z += -197.4858 * (Q.log_sum_pt - 6.327379) * (0.006197061 - Q.zdr_5)
    if Q.e3 < 0.0001869378 and Q.mean_eta < -0.01790907:
        z += 339202.2 * (0.0001869378 - Q.e3) * (-0.01790907 - Q.mean_eta)
    if Q.log_sum_pt > 6.327379 and Q.ptdr0_6 < 7.998907:
        z += -0.09768027 * (Q.log_sum_pt - 6.327379) * (7.998907 - Q.ptdr0_6)
    if Q.sj2_dr > 0.2179769 and Q.C2_b2 < 0.0006435798:
        z += -30811.62 * (Q.sj2_dr - 0.2179769) * (0.0006435798 - Q.C2_b2)
    if Q.max_dr > 0.121681 and Q.C2_b2 < 0.009032972:
        z += 2283.623 * (Q.max_dr - 0.121681) * (0.009032972 - Q.C2_b2)
    if Q.N2 < 0.2233283 and Q.lam2 > 0.001130645:
        z += -6237.686 * (0.2233283 - Q.N2) * (Q.lam2 - 0.001130645)
    if Q.sj2_dr > 0.2179769 and Q.C2_b2 < 0.009032972:
        z += -1405.034 * (Q.sj2_dr - 0.2179769) * (0.009032972 - Q.C2_b2)
    if Q.lam1 < 0.00733008 and Q.lam2 > 0.000537286:
        z += 520061.8 * (0.00733008 - Q.lam1) * (Q.lam2 - 0.000537286)
    if Q.dr_0 < 0.08082334 and Q.centroid_offset > 0.009480685:
        z += 2364.444 * (0.08082334 - Q.dr_0) * (Q.centroid_offset - 0.009480685)
    if Q.N2 < 0.2233283 and Q.n_dr_0p1_0p2 > 2.0:
        z += 1.393033 * (0.2233283 - Q.N2) * (Q.n_dr_0p1_0p2 - 2.0)
    if Q.sj3_dr_max > 0.3456459 and Q.sj3_z3 < 0.1167998:
        z += -406.2336 * (Q.sj3_dr_max - 0.3456459) * (0.1167998 - Q.sj3_z3)
    if Q.log_sum_pt > 6.327379 and Q.n_dr_0p05_0p1 < 7.0:
        z += -0.217186 * (Q.log_sum_pt - 6.327379) * (7.0 - Q.n_dr_0p05_0p1)
    if Q.log_sum_pt > 6.327379 and Q.ptdr0_7 < 8.336138:
        z += -0.0522285 * (Q.log_sum_pt - 6.327379) * (8.336138 - Q.ptdr0_7)
    if Q.log_sum_pt > 6.327379 and Q.zdr_2 < 0.009799324:
        z += -132.6771 * (Q.log_sum_pt - 6.327379) * (0.009799324 - Q.zdr_2)
    if Q.lam2 < 0.001130645 and Q.mean_phi < -0.02594505:
        z += -69002.58 * (0.001130645 - Q.lam2) * (-0.02594505 - Q.mean_phi)
    if Q.sj3_dr_min < 0.2089872 and Q.lam2 < 0.003408389:
        z += -7002.844 * (0.2089872 - Q.sj3_dr_min) * (0.003408389 - Q.lam2)
    if Q.sj3_dr_min < 0.2089872 and Q.mean_phi < -0.02594505:
        z += 340.1847 * (0.2089872 - Q.sj3_dr_min) * (-0.02594505 - Q.mean_phi)
    if Q.sj2_dr > 0.2179769 and Q.lam2 > 0.001130645:
        z += 1378.807 * (Q.sj2_dr - 0.2179769) * (Q.lam2 - 0.001130645)
    if Q.log_sum_pt > 6.327379 and Q.ptdr0_5 > 13.11362:
        z += -0.3583881 * (Q.log_sum_pt - 6.327379) * (Q.ptdr0_5 - 13.11362)
    if Q.N2 < 0.2233283 and Q.pt_7 < 53.4375:
        z += -0.2842398 * (0.2233283 - Q.N2) * (53.4375 - Q.pt_7)
    if Q.lam1 < 0.006506576 and Q.eccentricity > 0.7117266:
        z += 1417.623 * (0.006506576 - Q.lam1) * (Q.eccentricity - 0.7117266)
    if Q.max_dr > 0.121681 and Q.n_pt_above_50 > 6.0:
        z += -6.97233 * (Q.max_dr - 0.121681) * (Q.n_pt_above_50 - 6.0)
    if Q.eccentricity > 0.7117266 and Q.n_pt_above_50 > 6.0:
        z += 1.555823 * (Q.eccentricity - 0.7117266) * (Q.n_pt_above_50 - 6.0)
    if Q.log_sum_pt > 6.327379 and Q.phi_0 < -0.01452103:
        z += -19.13665 * (Q.log_sum_pt - 6.327379) * (-0.01452103 - Q.phi_0)
    if Q.e3 < 0.0001869378 and Q.mean_phi < -0.0008556753:
        z += 63447.27 * (0.0001869378 - Q.e3) * (-0.0008556753 - Q.mean_phi)
    if Q.lam1 > 0.01200373 and Q.C3 < 0.02460965:
        z += 9663.897 * (Q.lam1 - 0.01200373) * (0.02460965 - Q.C3)
    if Q.log_sum_pt > 6.327379 and Q.abseta_4 < 0.08575439:
        z += -8.926167 * (Q.log_sum_pt - 6.327379) * (0.08575439 - Q.abseta_4)
    if Q.D2 < 1.002471 and Q.zdr_5 < 0.0155217:
        z += -40.24248 * (1.002471 - Q.D2) * (0.0155217 - Q.zdr_5)
    if Q.log_sum_pt > 6.327379 and Q.dr1_4 > 0.1883455:
        z += -10.53125 * (Q.log_sum_pt - 6.327379) * (Q.dr1_4 - 0.1883455)
    if Q.max_dr > 0.121681 and Q.abseta_0 < 0.07861328:
        z += -130.0968 * (Q.max_dr - 0.121681) * (0.07861328 - Q.abseta_0)
    if Q.lam2 < 0.001130645 and Q.absphi_7 > 0.001937771:
        z += 2943.147 * (0.001130645 - Q.lam2) * (Q.absphi_7 - 0.001937771)
    if Q.sj2_dr > 0.2179769 and Q.D2_b2 < 0.5327104:
        z += -19.81243 * (Q.sj2_dr - 0.2179769) * (0.5327104 - Q.D2_b2)
    if Q.lam2 < 0.000537286 and Q.dr0_7 > 0.09118326:
        z += -2103.455 * (0.000537286 - Q.lam2) * (Q.dr0_7 - 0.09118326)
    if Q.pt_7 < 25.57812 and Q.D2_b2 < 3.032555:
        z += -0.00929885 * (25.57812 - Q.pt_7) * (3.032555 - Q.D2_b2)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.113021
    if Q.LHA < 0.2160559:
        z += -12.75433 * Q.LHA + 2.587733
    if 0.2160559 <= Q.LHA < 0.266913:
        z += 3.301684 * Q.LHA - 0.8812622
    if Q.z_7 < 0.02320757:
        z += -223.5609 * Q.z_7 + 6.408354
    if 0.02320757 <= Q.z_7 < 0.02807091:
        z += -105.892 * Q.z_7 + 3.677544
    if 0.02807091 <= Q.z_7 < 0.03629544:
        z += -59.02501 * Q.z_7 + 2.361945
    if 0.03629544 <= Q.z_7 < 0.04939969:
        z += -16.75838 * Q.z_7 + 0.8278586
    if Q.e2 < 0.03556091:
        z += 35.86061 * Q.e2 - 1.275236
    if Q.z_6 < 0.02160287:
        z += -84.02162 * Q.z_6 + 1.363615
    if 0.02160287 <= Q.z_6 < 0.02886576:
        z += -13.78046 * Q.z_6 - 0.1537963
    if 0.02886576 <= Q.z_6 < 0.06081235:
        z += 17.26568 * Q.z_6 - 1.049967
    if Q.lam1 < 7.12483e-05:
        z += -19360.88 * Q.lam1 + 3.521959
    if 7.12483e-05 <= Q.lam1 < 0.0008722282:
        z += -1076.991 * Q.lam1 + 2.219263
    if 0.0008722282 <= Q.lam1 < 0.002464291:
        z += -478.7982 * Q.lam1 + 1.697503
    if 0.002464291 <= Q.lam1 < 0.00483998:
        z += -217.8756 * Q.lam1 + 1.054513
    if Q.sum_z_dr < 0.02054282:
        z += 74.54131 * Q.sum_z_dr - 1.531289
    if Q.sum_pt < 739.5:
        z += 0.01127597 * Q.sum_pt - 8.338581
    if Q.centroid_offset < 0.03117077:
        z += 18.12891 * Q.centroid_offset - 0.7406504
    if 0.03117077 <= Q.centroid_offset < 0.03776099:
        z += 26.6392 * Q.centroid_offset - 1.005923
    if Q.log_sum_pt >= 6.804164:
        z += -3.957182 * Q.log_sum_pt + 26.92531
    if Q.z_5 < 0.02818362:
        z += 15.87749 * Q.z_5 - 0.4474851
    if Q.z_top5_slots >= 0.908903:
        z += -8.302767 * Q.z_top5_slots + 7.54641
    if Q.dr_0 < 0.06413297:
        z += 18.09746 * Q.dr_0 - 1.160644
    if Q.pt_7 < 20.125:
        z += 0.1411501 * Q.pt_7 - 2.840647
    if 37.15625 <= Q.pt_7 < 53.4375:
        z += -0.04471783 * Q.pt_7 + 1.661547
    if Q.pt_7 >= 53.4375:
        z += -0.009158719 * Q.pt_7 - 0.2386431
    if Q.sj3_dr_max < 0.1070199:
        z += 3.581611 * Q.sj3_dr_max - 1.060829
    if 0.1070199 <= Q.sj3_dr_max < 0.1986272:
        z += -1.990144 * Q.sj3_dr_max - 0.4645403
    if 0.1986272 <= Q.sj3_dr_max < 0.3012016:
        z += 8.38257 * Q.sj3_dr_max - 2.524843
    if Q.e3 < 0.0001869378:
        z += -3979.036 * Q.e3 + 0.7438324
    if Q.sum_pt_top5 < 579.875:
        z += -0.004214015 * Q.sum_pt_top5 + 2.443602
    if Q.z_top5 < 0.8233866:
        z += 2.423995 * Q.z_top5 - 1.995885
    if Q.pt_5 < 24.57812:
        z += -0.09900829 * Q.pt_5 + 2.433438
    if Q.LHA < 0.2160559 and Q.z_6 < 0.09543973:
        z += -85.98429 * (0.2160559 - Q.LHA) * (0.09543973 - Q.z_6)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -44.40995 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.e2 < 0.03556091 and Q.z_7 < 0.0753896:
        z += 843.7184 * (0.03556091 - Q.e2) * (0.0753896 - Q.z_7)
    if Q.e2 < 0.03556091 and Q.zdr_0 > 0.004918231:
        z += 1290.035 * (0.03556091 - Q.e2) * (Q.zdr_0 - 0.004918231)
    if Q.e2 < 0.03556091 and Q.log_sum_pt > 6.896095:
        z += -358.4955 * (0.03556091 - Q.e2) * (Q.log_sum_pt - 6.896095)
    if Q.lam1 < 0.002464291 and Q.centroid_offset < 0.01837778:
        z += 72406.86 * (0.002464291 - Q.lam1) * (0.01837778 - Q.centroid_offset)
    if Q.sum_z_dr < 0.02054282 and Q.sum_pt_top5 > 716.8828:
        z += 0.6496206 * (0.02054282 - Q.sum_z_dr) * (Q.sum_pt_top5 - 716.8828)
    if Q.sum_z_dr < 0.02054282 and Q.n_dr_0p2_0p4 > 0.0:
        z += -97.88427 * (0.02054282 - Q.sum_z_dr) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.log_sum_pt > 6.896095 and Q.zdr_0 < 0.007798268:
        z += -1431.326 * (Q.log_sum_pt - 6.896095) * (0.007798268 - Q.zdr_0)
    if Q.z_6 < 0.06081235 and Q.ptdr0_6 > 6.814768:
        z += 3.802616 * (0.06081235 - Q.z_6) * (Q.ptdr0_6 - 6.814768)
    if Q.sum_z_dr < 0.02054282 and Q.M3 < 0.09607851:
        z += -615.1614 * (0.02054282 - Q.sum_z_dr) * (0.09607851 - Q.M3)
    if Q.z_6 < 0.06081235 and Q.pt1_dr01 < 28.39396:
        z += 0.7584431 * (0.06081235 - Q.z_6) * (28.39396 - Q.pt1_dr01)
    if Q.log_sum_pt > 6.804164 and Q.e3 < 5.334511e-05:
        z += -163856.9 * (Q.log_sum_pt - 6.804164) * (5.334511e-05 - Q.e3)
    if Q.centroid_offset < 0.03117077 and Q.pt_2 < 105.9562:
        z += 0.4911326 * (0.03117077 - Q.centroid_offset) * (105.9562 - Q.pt_2)
    if Q.centroid_offset < 0.03117077 and Q.z_2 < 0.1364165:
        z += -293.8005 * (0.03117077 - Q.centroid_offset) * (0.1364165 - Q.z_2)
    if Q.e2 < 0.03556091 and Q.lam2 < 0.0003061234:
        z += 116247.3 * (0.03556091 - Q.e2) * (0.0003061234 - Q.lam2)
    if Q.z_6 < 0.06081235 and Q.e3 < 0.0005116989:
        z += 35866.72 * (0.06081235 - Q.z_6) * (0.0005116989 - Q.e3)
    if Q.sum_pt < 739.5 and Q.mean_eta2 < 0.008675069:
        z += 0.4533034 * (739.5 - Q.sum_pt) * (0.008675069 - Q.mean_eta2)
    if Q.sj3_dr_max < 0.3012016 and Q.ptdr0_2 > 11.87288:
        z += -0.2871894 * (0.3012016 - Q.sj3_dr_max) * (Q.ptdr0_2 - 11.87288)
    if Q.lam1 < 7.12483e-05 and Q.pt_5 < 73.75:
        z += -291.6231 * (7.12483e-05 - Q.lam1) * (73.75 - Q.pt_5)
    if Q.log_sum_pt > 6.804164 and Q.mean_phi < 0.01251955:
        z += 596.9517 * (Q.log_sum_pt - 6.804164) * (0.01251955 - Q.mean_phi)
    if Q.log_sum_pt > 6.896095 and Q.mean_phi > -0.01275329:
        z += 289.9139 * (Q.log_sum_pt - 6.896095) * (Q.mean_phi - -0.01275329)
    if Q.log_sum_pt > 6.804164 and Q.mean_phi < 0.00161889:
        z += -439.4984 * (Q.log_sum_pt - 6.804164) * (0.00161889 - Q.mean_phi)
    if Q.centroid_offset < 0.03117077 and Q.D3 > 0.5730377:
        z += -3.568854 * (0.03117077 - Q.centroid_offset) * (Q.D3 - 0.5730377)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.6158488
    if Q.centroid_offset < 0.00809236:
        z += -23.02713 * Q.centroid_offset + 0.4231876
    if 0.00809236 <= Q.centroid_offset < 0.01837778:
        z += -9.558409 * Q.centroid_offset + 0.3141938
    if 0.01837778 <= Q.centroid_offset < 0.02685622:
        z += 13.46872 * Q.centroid_offset - 0.1089938
    if Q.centroid_offset >= 0.02685622:
        z += 56.77727 * Q.centroid_offset - 1.272098
    if Q.log_sum_pt < 6.423044:
        z += -2.602126 * Q.log_sum_pt + 16.71357
    if 0.1591713 <= Q.sj2_dr < 0.1682655:
        z += -11.82704 * Q.sj2_dr + 1.882526
    if Q.sj2_dr >= 0.1682655:
        z += 5.582916 * Q.sj2_dr - 1.046971
    if Q.z_dr_0p1_0p2 >= 0.1585582:
        z += 0.1708072 * Q.z_dr_0p1_0p2 - 0.02708288
    if 0.08723651 <= Q.sum_z_dr < 0.1019409:
        z += -132.2826 * Q.sum_z_dr + 11.53987
    if 0.1019409 <= Q.sum_z_dr < 0.1484084:
        z += -154.573 * Q.sum_z_dr + 13.81218
    if Q.sum_z_dr >= 0.1484084:
        z += -100.5303 * Q.sum_z_dr + 5.791786
    if Q.pt_6 < 19.46875:
        z += -0.3779248 * Q.pt_6 + 6.739558
    if 19.46875 <= Q.pt_6 < 24.42188:
        z += -0.02224671 * Q.pt_6 - 0.1850503
    if 24.42188 <= Q.pt_6 < 31.90625:
        z += 0.02519149 * Q.pt_6 - 1.34358
    if 31.90625 <= Q.pt_6 < 39.75:
        z += 0.06882093 * Q.pt_6 - 2.735632
    if Q.lam2 < 5.794419e-05:
        z += 4319.003 * Q.lam2 + 0.3220342
    if 5.794419e-05 <= Q.lam2 < 0.001130645:
        z += 396.1529 * Q.lam2 + 0.5493406
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -437.823 * Q.lam2 + 1.492271
    if 0.251526 <= Q.LHA < 0.3255822:
        z += 25.15079 * Q.LHA - 6.326078
    if Q.LHA >= 0.3255822:
        z += 78.39787 * Q.LHA - 23.66238
    if Q.psi_0p2 >= 0.8990266:
        z += -5.188497 * Q.psi_0p2 + 4.664597
    if Q.C2 < 0.03578649:
        z += -37.18512 * Q.C2 + 1.330725
    if 0.03464708 <= Q.sj3_dr_max < 0.1789613:
        z += -13.96468 * Q.sj3_dr_max + 0.4838353
    if Q.sj3_dr_max >= 0.1789613:
        z += 2.278915 * Q.sj3_dr_max - 2.42314
    if Q.z_6 < 0.02160287:
        z += 279.0972 * Q.z_6 - 5.193058
    if 0.02160287 <= Q.z_6 < 0.04355037:
        z += -38.10201 * Q.z_6 + 1.659357
    if Q.e3 < 7.330536e-06:
        z += -62729.79 * Q.e3 + 0.459843
    if Q.max_dr < 0.177305:
        z += 19.46248 * Q.max_dr - 3.450795
    if Q.dr_0 < 0.08082334:
        z += -4.525296 * Q.dr_0 + 0.3657496
    if Q.sj3_pairmin_over_m < 0.07708997:
        z += 4.061665 * Q.sj3_pairmin_over_m - 0.3131136
    if 0.05356915 <= Q.tau1 < 0.1748444:
        z += -12.2782 * Q.tau1 + 0.6577329
    if Q.tau1 >= 0.1748444:
        z += -33.62751 * Q.tau1 + 4.390539
    if Q.pt_7 < 33.21875:
        z += -0.1024963 * Q.pt_7 + 3.4048
    if Q.z_7 < 0.03978449:
        z += 68.02136 * Q.z_7 - 2.706195
    if Q.sum_pt >= 840.0195:
        z += 0.00278734 * Q.sum_pt - 2.34142
    if Q.centroid_offset > 0.00809236 and Q.lam2 > 0.000537286:
        z += -3356.893 * (Q.centroid_offset - 0.00809236) * (Q.lam2 - 0.000537286)
    if Q.sj2_dr > 0.1591713 and Q.C2 < 0.09482124:
        z += 28.77513 * (Q.sj2_dr - 0.1591713) * (0.09482124 - Q.C2)
    if Q.sj2_dr > 0.1591713 and Q.n_dr_0p05_0p1 < 5.0:
        z += -0.6411816 * (Q.sj2_dr - 0.1591713) * (5.0 - Q.n_dr_0p05_0p1)
    if Q.sj2_dr > 0.1591713 and Q.sj2_zsoft > 0.05966518:
        z += 26.31275 * (Q.sj2_dr - 0.1591713) * (Q.sj2_zsoft - 0.05966518)
    if Q.centroid_offset > 0.00809236 and Q.eccentricity < 0.8319502:
        z += 61.53141 * (Q.centroid_offset - 0.00809236) * (0.8319502 - Q.eccentricity)
    if Q.centroid_offset < 0.01837778 and Q.planar_flow < 0.1115136:
        z += -227.1471 * (0.01837778 - Q.centroid_offset) * (0.1115136 - Q.planar_flow)
    if Q.pt_6 < 39.75 and Q.sum_pt_top3 < 711.875:
        z += 0.0005443578 * (39.75 - Q.pt_6) * (711.875 - Q.sum_pt_top3)
    if Q.centroid_offset > 0.00809236 and Q.sj3_dr_min < 0.2089872:
        z += 199.5447 * (Q.centroid_offset - 0.00809236) * (0.2089872 - Q.sj3_dr_min)
    if Q.centroid_offset > 0.00809236 and Q.mean_phi2 < 0.01426135:
        z += 1909.208 * (Q.centroid_offset - 0.00809236) * (0.01426135 - Q.mean_phi2)
    if Q.sj2_dr > 0.1591713 and Q.mean_eta > 0.02644207:
        z += -183.6942 * (Q.sj2_dr - 0.1591713) * (Q.mean_eta - 0.02644207)
    if Q.centroid_offset > 0.02076709 and Q.sj3_dr23 < 0.3661115:
        z += 77.92692 * (Q.centroid_offset - 0.02076709) * (0.3661115 - Q.sj3_dr23)
    if Q.sj3_dr_max > 0.1789613 and Q.n_dr_0p2_0p4 < 2.0:
        z += 1.34186 * (Q.sj3_dr_max - 0.1789613) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.centroid_offset > 0.02076709 and Q.n_dr_0_0p05 < 5.0:
        z += -10.24119 * (Q.centroid_offset - 0.02076709) * (5.0 - Q.n_dr_0_0p05)
    if Q.centroid_offset > 0.00809236 and Q.n_dr_0_0p05 > 6.0:
        z += -15.79448 * (Q.centroid_offset - 0.00809236) * (Q.n_dr_0_0p05 - 6.0)
    if Q.sj3_dr_max > 0.1789613 and Q.D2_b2 < 1.59265:
        z += 3.981068 * (Q.sj3_dr_max - 0.1789613) * (1.59265 - Q.D2_b2)
    if Q.lam2 < 0.001130645 and Q.C3 > 0.04165477:
        z += 11465.96 * (0.001130645 - Q.lam2) * (Q.C3 - 0.04165477)
    if Q.sum_z_dr > 0.1019409 and Q.pt_7 > 15.55391:
        z += -0.7856227 * (Q.sum_z_dr - 0.1019409) * (Q.pt_7 - 15.55391)
    if Q.pt_6 < 39.75 and Q.D2_b2 < 4.721224:
        z += -0.01037115 * (39.75 - Q.pt_6) * (4.721224 - Q.D2_b2)
    if Q.pt_6 < 39.75 and Q.z_top3_slots < 0.7861046:
        z += -0.3004273 * (39.75 - Q.pt_6) * (0.7861046 - Q.z_top3_slots)
    if Q.lam2 < 0.003408389 and Q.n_dr_0p1_0p2 > 1.0:
        z += 18.29595 * (0.003408389 - Q.lam2) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.centroid_offset < 0.01837778 and Q.n_dr_0p05_0p1 > 0.0:
        z += 1.132191 * (0.01837778 - Q.centroid_offset) * (Q.n_dr_0p05_0p1 - 0.0)
    if Q.sum_z_dr > 0.08723651 and Q.D2_b2 < 0.380911:
        z += 70.53213 * (Q.sum_z_dr - 0.08723651) * (0.380911 - Q.D2_b2)
    if Q.centroid_offset > 0.00809236 and Q.D3 > 0.8272948:
        z += -6.149772 * (Q.centroid_offset - 0.00809236) * (Q.D3 - 0.8272948)
    if Q.sj3_dr_max > 0.03464708 and Q.dr_min_012 < 0.01488897:
        z += -59.69736 * (Q.sj3_dr_max - 0.03464708) * (0.01488897 - Q.dr_min_012)
    if Q.centroid_offset > 0.02685622 and Q.pt_2 > 69.875:
        z += 0.7042421 * (Q.centroid_offset - 0.02685622) * (Q.pt_2 - 69.875)
    if Q.sum_z_dr > 0.1019409 and Q.D2_b2 < 0.2669656:
        z += -53.8587 * (Q.sum_z_dr - 0.1019409) * (0.2669656 - Q.D2_b2)
    if Q.centroid_offset > 0.02685622 and Q.D2_b2 < 0.380911:
        z += -114.1347 * (Q.centroid_offset - 0.02685622) * (0.380911 - Q.D2_b2)
    if Q.sum_pt > 840.0195 and Q.D2_b2 > 0.716559:
        z += -0.0008759812 * (Q.sum_pt - 840.0195) * (Q.D2_b2 - 0.716559)
    if Q.tau1 > 0.05356915 and Q.pt_dispersion < 0.6144369:
        z += 32.72538 * (Q.tau1 - 0.05356915) * (0.6144369 - Q.pt_dispersion)
    if Q.sj2_dr > 0.1682655 and Q.mean_phi > -0.01753483:
        z += -41.58373 * (Q.sj2_dr - 0.1682655) * (Q.mean_phi - -0.01753483)
    return max(0.0, z)


def neuron_7(Q):
    z = 2.163921
    if Q.sum_z_dr < 0.03360421:
        z += 18.94988 * Q.sum_z_dr - 2.183279
    if 0.03360421 <= Q.sum_z_dr < 0.0479157:
        z += 56.27297 * Q.sum_z_dr - 3.437492
    if 0.0479157 <= Q.sum_z_dr < 0.06108601:
        z += -20.37881 * Q.sum_z_dr + 0.2353325
    if 0.06108601 <= Q.sum_z_dr < 0.08065885:
        z += -76.65179 * Q.sum_z_dr + 3.672824
    if 0.08065885 <= Q.sum_z_dr < 0.08723651:
        z += -132.2649 * Q.sum_z_dr + 8.158514
    if 0.08723651 <= Q.sum_z_dr < 0.1019409:
        z += -263.4179 * Q.sum_z_dr + 19.59984
    if Q.sum_z_dr >= 0.1019409:
        z += -174.0905 * Q.sum_z_dr + 10.49372
    if Q.tau1 < 0.06345984:
        z += -48.00162 * Q.tau1 + 4.255521
    if 0.06345984 <= Q.tau1 < 0.08865369:
        z += -19.83489 * Q.tau1 + 2.468065
    if 0.08865369 <= Q.tau1 < 0.1027642:
        z += 28.16673 * Q.tau1 - 1.787456
    if 0.1027642 <= Q.tau1 < 0.1136369:
        z += 167.8884 * Q.tau1 - 16.14585
    if Q.tau1 >= 0.1136369:
        z += 130.5564 * Q.tau1 - 11.90356
    if Q.LHA < 0.2346184:
        z += -0.2671728 * Q.LHA - 1.735185
    if 0.2346184 <= Q.LHA < 0.3033137:
        z += 9.513839 * Q.LHA - 4.029991
    if 0.3033137 <= Q.LHA < 0.3467135:
        z += 24.06899 * Q.LHA - 8.444768
    if 0.3467135 <= Q.LHA < 0.4235925:
        z += -25.01539 * Q.LHA + 8.573449
    if Q.LHA >= 0.4235925:
        z += -34.52923 * Q.LHA + 12.60344
    if Q.z_dr_0p1_0p2 < 0.1585582:
        z += 2.187336 * Q.z_dr_0p1_0p2 - 0.3468201
    if Q.lam1 < 0.002464291:
        z += 953.7146 * Q.lam1 - 2.620903
    if 0.002464291 <= Q.lam1 < 0.003377388:
        z += 1156.634 * Q.lam1 - 3.120955
    if 0.003377388 <= Q.lam1 < 0.00483998:
        z += 454.9595 * Q.lam1 - 0.751129
    if 0.00483998 <= Q.lam1 < 0.00733008:
        z += -71.09259 * Q.lam1 + 1.794953
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += -561.787 * Q.lam1 + 5.391782
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -287.7753 * Q.lam1 + 3.096777
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 131.7517 * Q.lam1 - 1.93911
    if Q.lam1 >= 0.01643375:
        z += -1043.624 * Q.lam1 + 17.37672
    if 0.1333619 <= Q.N2 < 0.1754643:
        z += 4.918295 * Q.N2 - 0.6559131
    if Q.N2 >= 0.1754643:
        z += 2.318158 * Q.N2 - 0.1996819
    if Q.lam2 < 0.0003061234:
        z += 1969.631 * Q.lam2 + 0.7471686
    if 0.0003061234 <= Q.lam2 < 0.003408389:
        z += -435.204 * Q.lam2 + 1.483345
    if Q.centroid_offset < 0.02685622:
        z += -55.61462 * Q.centroid_offset + 2.470745
    if 0.02685622 <= Q.centroid_offset < 0.04990367:
        z += -42.39716 * Q.centroid_offset + 2.115774
    if Q.sum_z_dr2_top5 < 0.0006570502:
        z += 645.6288 * Q.sum_z_dr2_top5 - 0.4242105
    if Q.e3 < 4.192959e-06:
        z += 98926.88 * Q.e3 - 1.778342
    if 4.192959e-06 <= Q.e3 < 1.050302e-05:
        z += -14183.44 * Q.e3 - 1.304075
    if 1.050302e-05 <= Q.e3 < 8.147744e-05:
        z += 20472.79 * Q.e3 - 1.66807
    if Q.sj3_dr_max < 0.02481095:
        z += 4.865372 * Q.sj3_dr_max + 0.9611658
    if 0.02481095 <= Q.sj3_dr_max < 0.07264452:
        z += -13.43241 * Q.sj3_dr_max + 1.415151
    if 0.07264452 <= Q.sj3_dr_max < 0.1426152:
        z += -18.27189 * Q.sj3_dr_max + 1.766713
    if 0.1426152 <= Q.sj3_dr_max < 0.1594012:
        z += -24.55773 * Q.sj3_dr_max + 2.663169
    if 0.1594012 <= Q.sj3_dr_max < 0.213399:
        z += 10.55936 * Q.sj3_dr_max - 2.934536
    if Q.sj3_dr_max >= 0.213399:
        z += -4.839482 * Q.sj3_dr_max + 0.3515618
    if Q.ptdr0_6 >= 11.6057:
        z += -0.07687314 * Q.ptdr0_6 + 0.8921668
    if Q.D2 < 1.432482:
        z += 0.5227734 * Q.D2 - 0.7488636
    if Q.max_dr < 0.08050702:
        z += 10.75132 * Q.max_dr - 1.100066
    if 0.08050702 <= Q.max_dr < 0.0931108:
        z += 18.6063 * Q.max_dr - 1.732447
    if Q.psi_0p1 >= 0.7688952:
        z += 2.297964 * Q.psi_0p1 - 1.766893
    if Q.pt_5 < 33.02656:
        z += 0.03791199 * Q.pt_5 - 1.252103
    if Q.planar_flow < 0.1950135 and Q.lam1 > 0.008375572:
        z += -3874.269 * (0.1950135 - Q.planar_flow) * (Q.lam1 - 0.008375572)
    if Q.z_dr_0p1_0p2 < 0.1585582 and Q.centroid_offset < 0.02685622:
        z += -157.4209 * (0.1585582 - Q.z_dr_0p1_0p2) * (0.02685622 - Q.centroid_offset)
    if Q.planar_flow < 0.1950135 and Q.pt_6 < 56.53125:
        z += -0.05698377 * (0.1950135 - Q.planar_flow) * (56.53125 - Q.pt_6)
    if Q.planar_flow < 0.1950135 and Q.log_sum_pt > 6.46415:
        z += 19.44353 * (0.1950135 - Q.planar_flow) * (Q.log_sum_pt - 6.46415)
    if Q.planar_flow < 0.1950135 and Q.pt_7 < 29.04219:
        z += -0.2941374 * (0.1950135 - Q.planar_flow) * (29.04219 - Q.pt_7)
    if Q.lam1 > 0.002464291 and Q.sj3_dr_max < 0.1879486:
        z += -9136.629 * (Q.lam1 - 0.002464291) * (0.1879486 - Q.sj3_dr_max)
    if Q.planar_flow < 0.1950135 and Q.max_dr < 0.2507612:
        z += -17.78345 * (0.1950135 - Q.planar_flow) * (0.2507612 - Q.max_dr)
    if Q.sum_z_dr < 0.06108601 and Q.mean_phi < -0.00469376:
        z += 1879.055 * (0.06108601 - Q.sum_z_dr) * (-0.00469376 - Q.mean_phi)
    if Q.z_dr_0p1_0p2 < 0.1585582 and Q.pt1_dr01 > 17.82896:
        z += 0.2312932 * (0.1585582 - Q.z_dr_0p1_0p2) * (Q.pt1_dr01 - 17.82896)
    if Q.sum_z_dr > 0.08065885 and Q.sj2_zsoft < 0.09384951:
        z += -43362.56 * (Q.sum_z_dr - 0.08065885) * (0.09384951 - Q.sj2_zsoft)
    if Q.lam1 > 0.00733008 and Q.D2 < 2.843757:
        z += -471.7996 * (Q.lam1 - 0.00733008) * (2.843757 - Q.D2)
    if Q.lam1 > 0.002464291 and Q.planar_flow < 0.1950135:
        z += 2672.55 * (Q.lam1 - 0.002464291) * (0.1950135 - Q.planar_flow)
    if Q.lam1 > 0.002464291 and Q.D2 > 0.2568137:
        z += -147.1159 * (Q.lam1 - 0.002464291) * (Q.D2 - 0.2568137)
    if Q.lam1 > 0.01643375 and Q.D2 < 3.885568:
        z += 322.4758 * (Q.lam1 - 0.01643375) * (3.885568 - Q.D2)
    if Q.sum_z_dr > 0.0479157 and Q.D2 < 2.357246:
        z += 59.02991 * (Q.sum_z_dr - 0.0479157) * (2.357246 - Q.D2)
    if Q.e3 < 8.147744e-05 and Q.pt_7 < 43.5:
        z += -284.265 * (8.147744e-05 - Q.e3) * (43.5 - Q.pt_7)
    if Q.sum_z_dr > 0.08065885 and Q.D2 < 2.843757:
        z += -23.90886 * (Q.sum_z_dr - 0.08065885) * (2.843757 - Q.D2)
    if Q.lam1 < 0.003377388 and Q.mean_eta < -0.009391681:
        z += 14300.1 * (0.003377388 - Q.lam1) * (-0.009391681 - Q.mean_eta)
    if Q.sj3_dr_max < 0.1426152 and Q.D2 < 1.432482:
        z += 14.66524 * (0.1426152 - Q.sj3_dr_max) * (1.432482 - Q.D2)
    if Q.centroid_offset < 0.02685622 and Q.n_dr_0p2_0p4 < 1.0:
        z += -64.03282 * (0.02685622 - Q.centroid_offset) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.centroid_offset < 0.04990367 and Q.n_dr_0p2_0p4 < 2.0:
        z += 4.220497 * (0.04990367 - Q.centroid_offset) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.centroid_offset < 0.02685622 and Q.N3 < 1.808379:
        z += -12.53387 * (0.02685622 - Q.centroid_offset) * (1.808379 - Q.N3)
    if Q.sum_z_dr > 0.08065885 and Q.D2_b2 > 1.911889:
        z += -282.8307 * (Q.sum_z_dr - 0.08065885) * (Q.D2_b2 - 1.911889)
    if Q.lam1 < 0.00483998 and Q.mean_eta > 0.01271871:
        z += 18240.2 * (0.00483998 - Q.lam1) * (Q.mean_eta - 0.01271871)
    if Q.lam1 < 0.003377388 and Q.mean_phi > 0.01251955:
        z += 28554.95 * (0.003377388 - Q.lam1) * (Q.mean_phi - 0.01251955)
    if Q.centroid_offset < 0.04990367 and Q.tau4 > 0.002042374:
        z += -1952.517 * (0.04990367 - Q.centroid_offset) * (Q.tau4 - 0.002042374)
    if Q.z_dr_0p05_0p1 > 0.6747704 and Q.mean_eta < -0.02665591:
        z += -177.0509 * (Q.z_dr_0p05_0p1 - 0.6747704) * (-0.02665591 - Q.mean_eta)
    if Q.lam1 < 0.008375572 and Q.mean_eta < -0.004664942:
        z += 4113.345 * (0.008375572 - Q.lam1) * (-0.004664942 - Q.mean_eta)
    if Q.lam1 > 0.00733008 and Q.pt_4 < 90.625:
        z += 10.30692 * (Q.lam1 - 0.00733008) * (90.625 - Q.pt_4)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.8472343
    if Q.lam1 < 0.0002758826:
        z += -1743.905 * Q.lam1 + 1.23611
    if 0.0002758826 <= Q.lam1 < 0.003377388:
        z += -259.8884 * Q.lam1 + 0.8266953
    if 0.003377388 <= Q.lam1 < 0.00733008:
        z += 12.91495 * Q.lam1 - 0.09466762
    if Q.LHA < 0.1767241:
        z += -19.07123 * Q.LHA + 2.872278
    if 0.1767241 <= Q.LHA < 0.251526:
        z += 6.658511 * Q.LHA - 1.674789
    if Q.sum_pt_top5 >= 658.125:
        z += 0.003164089 * Q.sum_pt_top5 - 2.082366
    if Q.log_sum_pt >= 6.701242:
        z += -5.281885 * Q.log_sum_pt + 35.39519
    if Q.z_dr_0_0p05 >= 0.9008535:
        z += -1.007837 * Q.z_dr_0_0p05 + 0.9079135
    if Q.sj3_dr_max < 0.1070199:
        z += 52.76161 * Q.sj3_dr_max - 3.728939
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += -37.73202 * Q.sj3_dr_max + 5.955683
    if 0.1426152 <= Q.sj3_dr_max < 0.1594012:
        z += -15.82958 * Q.sj3_dr_max + 2.832064
    if 0.1594012 <= Q.sj3_dr_max < 0.1986272:
        z += -7.872589 * Q.sj3_dr_max + 1.56371
    if Q.sum_z_dr < 0.02054282:
        z += 91.56038 * Q.sum_z_dr - 2.421791
    if 0.02054282 <= Q.sum_z_dr < 0.03360421:
        z += 41.41082 * Q.sum_z_dr - 1.391578
    if Q.tau1 < 0.0262518:
        z += 28.74124 * Q.tau1 - 0.4696586
    if 0.0262518 <= Q.tau1 < 0.03459477:
        z += -34.14261 * Q.tau1 + 1.181156
    if Q.z_6 < 0.04355037:
        z += -23.62881 * Q.z_6 + 1.029043
    if Q.max_dr >= 0.290267:
        z += -222.6371 * Q.max_dr + 64.62422
    if Q.centroid_offset >= 0.006789738:
        z += -17.62526 * Q.centroid_offset + 0.1196709
    if Q.z_top5_slots >= 0.8919245:
        z += -6.920509 * Q.z_top5_slots + 6.172571
    if Q.lam2 < 0.0001947983:
        z += -4994.471 * Q.lam2 + 0.9729145
    if Q.tau2 < 0.01713288:
        z += 33.16376 * Q.tau2 - 0.5681907
    if Q.lam1 < 0.00733008 and Q.centroid_offset < 0.02355416:
        z += 6794.583 * (0.00733008 - Q.lam1) * (0.02355416 - Q.centroid_offset)
    if Q.lam1 < 0.00733008 and Q.planar_flow < 0.5925897:
        z += 50.40654 * (0.00733008 - Q.lam1) * (0.5925897 - Q.planar_flow)
    if Q.lam1 < 0.005433361 and Q.log_sum_pt < 6.502799:
        z += -437.2847 * (0.005433361 - Q.lam1) * (6.502799 - Q.log_sum_pt)
    if Q.lam1 < 0.003377388 and Q.centroid_offset > 0.006789738:
        z += -15760.57 * (0.003377388 - Q.lam1) * (Q.centroid_offset - 0.006789738)
    if Q.sj3_dr_max < 0.1986272 and Q.n_dr_0p05_0p1 < 5.0:
        z += 0.3883267 * (0.1986272 - Q.sj3_dr_max) * (5.0 - Q.n_dr_0p05_0p1)
    if Q.LHA < 0.1767241 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -496.4369 * (0.1767241 - Q.LHA) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.lam1 < 0.005433361 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += 4263.026 * (0.005433361 - Q.lam1) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.lam1 < 0.003377388 and Q.centroid_offset > 0.01627885:
        z += -11999.71 * (0.003377388 - Q.lam1) * (Q.centroid_offset - 0.01627885)
    if Q.tau1 < 0.0262518 and Q.centroid_offset < 0.03117077:
        z += 2361.839 * (0.0262518 - Q.tau1) * (0.03117077 - Q.centroid_offset)
    if Q.lam1 < 0.005433361 and Q.pt_7 < 38.53125:
        z += -7.04785 * (0.005433361 - Q.lam1) * (38.53125 - Q.pt_7)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 < 33.21875:
        z += 0.1134207 * (Q.log_sum_pt - 6.701242) * (33.21875 - Q.pt_7)
    if Q.sum_z_dr < 0.03360421 and Q.centroid_offset < 0.03117077:
        z += -2655.367 * (0.03360421 - Q.sum_z_dr) * (0.03117077 - Q.centroid_offset)
    if Q.lam1 < 0.005433361 and Q.centroid_offset < 0.03117077:
        z += 8020.634 * (0.005433361 - Q.lam1) * (0.03117077 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.1070199 and Q.centroid_offset > 0.01627885:
        z += -1121.016 * (0.1070199 - Q.sj3_dr_max) * (Q.centroid_offset - 0.01627885)
    if Q.sj3_dr_max < 0.1594012 and Q.centroid_offset > 0.003343241:
        z += -288.565 * (0.1594012 - Q.sj3_dr_max) * (Q.centroid_offset - 0.003343241)
    if Q.z_6 < 0.04355037 and Q.absphi_1 < 0.07141113:
        z += -373.9046 * (0.04355037 - Q.z_6) * (0.07141113 - Q.absphi_1)
    if Q.sj3_dr_max < 0.1070199 and Q.sum_z_dr2_top3 < 0.0153634:
        z += 4862.746 * (0.1070199 - Q.sj3_dr_max) * (0.0153634 - Q.sum_z_dr2_top3)
    if Q.sj3_dr_max < 0.1426152 and Q.sum_z_dr2_top3 < 0.0153634:
        z += -2257.891 * (0.1426152 - Q.sj3_dr_max) * (0.0153634 - Q.sum_z_dr2_top3)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2_top3 < 0.01002369:
        z += -276.7315 * (Q.log_sum_pt - 6.701242) * (0.01002369 - Q.sum_z_dr2_top3)
    if Q.lam1 < 0.00733008 and Q.sum_z_dr2_top2 < 0.001056655:
        z += -74219.41 * (0.00733008 - Q.lam1) * (0.001056655 - Q.sum_z_dr2_top2)
    if Q.sum_pt_top5 > 658.125 and Q.n_pt_above_10 < 8.0:
        z += -0.001420828 * (Q.sum_pt_top5 - 658.125) * (8.0 - Q.n_pt_above_10)
    if Q.tau1 < 0.03459477 and Q.sum_z_dr2_top2 < 0.001056655:
        z += 21969.2 * (0.03459477 - Q.tau1) * (0.001056655 - Q.sum_z_dr2_top2)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2_top3 < 0.002915531:
        z += 567.5154 * (Q.log_sum_pt - 6.701242) * (0.002915531 - Q.sum_z_dr2_top3)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2_top5 < 0.00633598:
        z += -982.8176 * (Q.log_sum_pt - 6.701242) * (0.00633598 - Q.sum_z_dr2_top5)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2_top5 < 0.00327978:
        z += 2231.2 * (Q.log_sum_pt - 6.701242) * (0.00327978 - Q.sum_z_dr2_top5)
    if Q.sum_z_dr < 0.03360421 and Q.psi_0p1 > 0.9761279:
        z += 792.0322 * (0.03360421 - Q.sum_z_dr) * (Q.psi_0p1 - 0.9761279)
    if Q.tau1 < 0.01357518 and Q.sum_z_dr2_top2 < 0.001056655:
        z += -35469.79 * (0.01357518 - Q.tau1) * (0.001056655 - Q.sum_z_dr2_top2)
    if Q.sj3_dr_max < 0.1070199 and Q.sum_z_dr2_top2 < 0.001864148:
        z += 12803.94 * (0.1070199 - Q.sj3_dr_max) * (0.001864148 - Q.sum_z_dr2_top2)
    if Q.lam1 < 0.005433361 and Q.pt1_dr01 < 5.351077:
        z += 13.2389 * (0.005433361 - Q.lam1) * (5.351077 - Q.pt1_dr01)
    if Q.sj3_dr_max < 0.1426152 and Q.pt1_dr01 < 22.38578:
        z += -0.2317341 * (0.1426152 - Q.sj3_dr_max) * (22.38578 - Q.pt1_dr01)
    if Q.sj3_dr_max < 0.1594012 and Q.sum_z_dr2_top2 < 0.001864148:
        z += -7820.239 * (0.1594012 - Q.sj3_dr_max) * (0.001864148 - Q.sum_z_dr2_top2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2_top2 < 0.001864148:
        z += 2285.244 * (0.1986272 - Q.sj3_dr_max) * (0.001864148 - Q.sum_z_dr2_top2)
    return max(0.0, z)


def neuron_9(Q):
    z = -4.009979
    if Q.sum_z_dr < 0.007673833:
        z += 404.7354 * Q.sum_z_dr - 10.30971
    if 0.007673833 <= Q.sum_z_dr < 0.02689598:
        z += 219.3487 * Q.sum_z_dr - 8.887081
    if 0.02689598 <= Q.sum_z_dr < 0.04081947:
        z += 147.3044 * Q.sum_z_dr - 6.949378
    if 0.04081947 <= Q.sum_z_dr < 0.05464922:
        z += 67.71556 * Q.sum_z_dr - 3.700603
    if Q.tau1 < 0.04369778:
        z += 17.07421 * Q.tau1 - 0.7461051
    if Q.e3 < 1.960701e-05:
        z += 37471.51 * Q.e3 - 0.7347042
    if 0.0001869378 <= Q.e3 < 0.0005116989:
        z += 2078.831 * Q.e3 - 0.3886121
    if Q.e3 >= 0.0005116989:
        z += 799.8051 * Q.e3 + 0.2658639
    if Q.centroid_offset < 0.002316125:
        z += 242.7971 * Q.centroid_offset + 1.233709
    if 0.002316125 <= Q.centroid_offset < 0.01627885:
        z += -102.4313 * Q.centroid_offset + 2.033301
    if 0.01627885 <= Q.centroid_offset < 0.03117077:
        z += -24.56619 * Q.centroid_offset + 0.7657471
    if Q.lam1 < 0.0001413912:
        z += -9589.034 * Q.lam1 + 7.455957
    if 0.0001413912 <= Q.lam1 < 0.0005049491:
        z += -2463.596 * Q.lam1 + 6.448483
    if 0.0005049491 <= Q.lam1 < 0.003377388:
        z += -1341.982 * Q.lam1 + 5.882125
    if 0.003377388 <= Q.lam1 < 0.00595415:
        z += -523.8084 * Q.lam1 + 3.118834
    if Q.lam1 >= 0.01643375:
        z += 64.05631 * Q.lam1 - 1.052685
    if Q.sum_pt < 813.4156:
        z += -0.001413612 * Q.sum_pt + 2.26533
    if 813.4156 <= Q.sum_pt < 988.4078:
        z += -0.006374435 * Q.sum_pt + 6.300542
    if Q.lam2 < 0.0001947983:
        z += -2751.419 * Q.lam2 + 0.5359717
    if Q.lam2 >= 0.000537286:
        z += 673.2201 * Q.lam2 - 0.3617117
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.6159914 * Q.n_dr_0p2_0p4 - 0.6159914
    if Q.sj2_dr < 0.1294903:
        z += 5.861151 * Q.sj2_dr - 0.5647212
    if 0.1294903 <= Q.sj2_dr < 0.1778793:
        z += -4.014157 * Q.sj2_dr + 0.7140354
    if Q.sj3_dr_min >= 0.1278212:
        z += 2.637887 * Q.sj3_dr_min - 0.3371778
    if Q.zdr_0 < 0.004918231:
        z += 56.24419 * Q.zdr_0 - 0.2766219
    if Q.C2_b2 < 0.009032972:
        z += 38.18456 * Q.C2_b2 - 0.3449201
    if Q.sj3_dr_max < 0.1426152:
        z += 12.31715 * Q.sj3_dr_max - 0.504307
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -17.69197 * Q.sj3_dr_max + 3.775448
    if Q.max_dr < 0.1027585:
        z += -6.293591 * Q.max_dr + 0.6467201
    if Q.dr_0 < 0.01675541:
        z += -41.47571 * Q.dr_0 + 0.6949424
    if Q.planar_flow < 0.5925897:
        z += 0.3548555 * Q.planar_flow - 0.2102837
    if Q.e2 < 0.007078158:
        z += 65.21852 * Q.e2 - 0.461627
    if Q.log_sum_pt < 6.638339:
        z += -0.9638822 * Q.log_sum_pt + 6.398577
    if Q.sum_z_dr < 0.05464922 and Q.max_dr > 0.1326115:
        z += -114.3664 * (0.05464922 - Q.sum_z_dr) * (Q.max_dr - 0.1326115)
    if Q.lam1 < 0.00595415 and Q.lam2 < 0.001130645:
        z += 916331.8 * (0.00595415 - Q.lam1) * (0.001130645 - Q.lam2)
    if Q.centroid_offset < 0.01627885 and Q.D2_b2 < 0.1830092:
        z += -377.4008 * (0.01627885 - Q.centroid_offset) * (0.1830092 - Q.D2_b2)
    if Q.e3 < 4.192959e-06 and Q.n_dr_0p05_0p1 > 3.0:
        z += -387405.1 * (4.192959e-06 - Q.e3) * (Q.n_dr_0p05_0p1 - 3.0)
    if Q.sum_z_dr < 0.05464922 and Q.mean_phi < 0.0006973656:
        z += -1292.553 * (0.05464922 - Q.sum_z_dr) * (0.0006973656 - Q.mean_phi)
    if Q.sum_pt < 988.4078 and Q.z_5 < 0.05753583:
        z += 0.1304562 * (988.4078 - Q.sum_pt) * (0.05753583 - Q.z_5)
    if Q.lam1 < 0.00595415 and Q.mean_eta < -0.006779839:
        z += -21653.64 * (0.00595415 - Q.lam1) * (-0.006779839 - Q.mean_eta)
    if Q.centroid_offset < 0.01627885 and Q.sj3_dr_min < 0.1278212:
        z += 671.7156 * (0.01627885 - Q.centroid_offset) * (0.1278212 - Q.sj3_dr_min)
    if Q.lam1 < 0.00595415 and Q.sj3_dr_min < 0.08543881:
        z += -2581.825 * (0.00595415 - Q.lam1) * (0.08543881 - Q.sj3_dr_min)
    if Q.centroid_offset < 0.02355416 and Q.lam1 < 0.00733008:
        z += 17612.33 * (0.02355416 - Q.centroid_offset) * (0.00733008 - Q.lam1)
    if Q.zdr_0 < 0.004918231 and Q.pt_1 < 182.125:
        z += -1.450639 * (0.004918231 - Q.zdr_0) * (182.125 - Q.pt_1)
    if Q.lam2 > 0.000537286 and Q.eccentricity < 0.9979797:
        z += -469.3521 * (Q.lam2 - 0.000537286) * (0.9979797 - Q.eccentricity)
    if Q.lam2 > 0.000537286 and Q.n_dr_0p05_0p1 > 1.0:
        z += -31.23712 * (Q.lam2 - 0.000537286) * (Q.n_dr_0p05_0p1 - 1.0)
    if Q.lam1 < 0.00595415 and Q.mean_eta > 0.009367547:
        z += -26839.9 * (0.00595415 - Q.lam1) * (Q.mean_eta - 0.009367547)
    if Q.lam1 < 0.00595415 and Q.mean_phi < -0.009352575:
        z += -17106.87 * (0.00595415 - Q.lam1) * (-0.009352575 - Q.mean_phi)
    if Q.lam1 < 0.00595415 and Q.mean_phi > 0.01251955:
        z += -25843.21 * (0.00595415 - Q.lam1) * (Q.mean_phi - 0.01251955)
    if Q.sum_pt < 813.4156 and Q.psi_0p2 > 0.79448:
        z += 0.007333569 * (813.4156 - Q.sum_pt) * (Q.psi_0p2 - 0.79448)
    if Q.sum_pt < 813.4156 and Q.e4 < 3.0374e-08:
        z += 93686.0 * (813.4156 - Q.sum_pt) * (3.0374e-08 - Q.e4)
    if Q.centroid_offset < 0.01627885 and Q.N3 < 1.808379:
        z += -27.65969 * (0.01627885 - Q.centroid_offset) * (1.808379 - Q.N3)
    if Q.e3 > 0.0001869378 and Q.pt_7 < 31.85938:
        z += -173.9453 * (Q.e3 - 0.0001869378) * (31.85938 - Q.pt_7)
    if Q.sum_pt < 813.4156 and Q.sum_z_dr2_top2 < 0.0003125151:
        z += -11.82135 * (813.4156 - Q.sum_pt) * (0.0003125151 - Q.sum_z_dr2_top2)
    if Q.sum_pt < 813.4156 and Q.sum_z_dr2_top2 < 0.0095303:
        z += 0.1084789 * (813.4156 - Q.sum_pt) * (0.0095303 - Q.sum_z_dr2_top2)
    if Q.planar_flow < 0.5925897 and Q.D2_b2 < 4.721224:
        z += 0.3056928 * (0.5925897 - Q.planar_flow) * (4.721224 - Q.D2_b2)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.D2_b2 < 1.911889:
        z += -0.2220528 * (Q.n_dr_0p2_0p4 - 1.0) * (1.911889 - Q.D2_b2)
    if Q.sum_z_dr < 0.007673833 and Q.absphi_7 < 0.08074951:
        z += 3965.803 * (0.007673833 - Q.sum_z_dr) * (0.08074951 - Q.absphi_7)
    if Q.tau1 < 0.04369778 and Q.centroid_offset < 0.03117077:
        z += 3872.744 * (0.04369778 - Q.tau1) * (0.03117077 - Q.centroid_offset)
    if Q.sum_pt < 988.4078 and Q.centroid_offset < 0.01627885:
        z += -0.2529599 * (988.4078 - Q.sum_pt) * (0.01627885 - Q.centroid_offset)
    return max(0.0, z)


def neuron_10(Q):
    z = 3.761169
    z += -34.28003 * Q.e2
    if Q.lam1 < 0.0008722282:
        z += 736.0228 * Q.lam1 - 1.014849
    if 0.0008722282 <= Q.lam1 < 0.003377388:
        z += 665.6621 * Q.lam1 - 0.9534789
    if 0.003377388 <= Q.lam1 < 0.00483998:
        z += 365.1786 * Q.lam1 + 0.06137058
    if Q.lam1 >= 0.00483998:
        z += 271.3686 * Q.lam1 + 0.5154091
    if 0.0001947983 <= Q.lam2 < 0.003408389:
        z += 1564.491 * Q.lam2 - 0.3047601
    if Q.lam2 >= 0.003408389:
        z += 734.7964 * Q.lam2 + 2.523161
    if Q.pt_6 < 48.03125:
        z += 0.01290999 * Q.pt_6 - 0.6200828
    if 8.147744e-05 <= Q.e3 < 0.0005116989:
        z += 2131.004 * Q.e3 - 0.1736287
    if Q.e3 >= 0.0005116989:
        z += -40.81519 * Q.e3 + 0.9376887
    if 0.1967397 <= Q.LHA < 0.3127275:
        z += -12.66657 * Q.LHA + 2.492018
    if Q.LHA >= 0.3127275:
        z += -20.84309 * Q.LHA + 5.049038
    if Q.centroid_offset >= 0.02355416:
        z += 7.490668 * Q.centroid_offset - 0.1764364
    if Q.D2 < 1.002471:
        z += -0.4160775 * Q.D2 + 0.4171056
    if Q.n_dr_0p05_0p1 < 5.0:
        z += -0.04796981 * Q.n_dr_0p05_0p1 + 0.239849
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 1.162046 * Q.n_dr_0p2_0p4 - 1.162046
    if Q.z_7 < 0.06810151:
        z += 21.21096 * Q.z_7 - 1.444499
    z += 4.511083 * Q.mean_phi
    if Q.tau1 >= 0.04369778:
        z += 14.64734 * Q.tau1 - 0.6400563
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += 9.852464 * Q.sj2_dr - 1.275798
    if 0.1591713 <= Q.sj2_dr < 0.1682655:
        z += -13.18363 * Q.sj2_dr + 2.390887
    if 0.1682655 <= Q.sj2_dr < 0.3003793:
        z += 1.670551 * Q.sj2_dr - 0.1085605
    if Q.sj2_dr >= 0.3003793:
        z += 9.136447 * Q.sj2_dr - 2.351161
    if Q.sum_z_dr2_top3 < 0.002915531:
        z += -147.3415 * Q.sum_z_dr2_top3 + 0.4295787
    if Q.sum_pt >= 988.4078:
        z += 0.002288764 * Q.sum_pt - 2.262232
    if Q.M3 < 0.04581318:
        z += -10.21105 * Q.M3 + 0.4678007
    if Q.sum_z_dr < 0.1484084:
        z += 23.77034 * Q.sum_z_dr - 3.527718
    if Q.pt_7 >= 20.125:
        z += 0.01145077 * Q.pt_7 - 0.2304468
    if Q.C2_b2 < 0.009032972:
        z += -40.4031 * Q.C2_b2 + 0.36496
    if Q.D2_b2 < 0.2669656:
        z += -2.364246 * Q.D2_b2 + 0.6311724
    if Q.log_sum_pt >= 6.670067:
        z += -3.38167 * Q.log_sum_pt + 22.55597
    if Q.lam2 > 0.0001947983 and Q.log_sum_pt < 6.538559:
        z += -855.8291 * (Q.lam2 - 0.0001947983) * (6.538559 - Q.log_sum_pt)
    if Q.lam2 > 0.0001947983 and Q.eccentricity > 0.4829602:
        z += -815.1451 * (Q.lam2 - 0.0001947983) * (Q.eccentricity - 0.4829602)
    if Q.pt_6 < 48.03125 and Q.log_sum_pt < 6.46415:
        z += -0.1135686 * (48.03125 - Q.pt_6) * (6.46415 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.D2_b2 < 4.721224:
        z += -0.2369644 * (Q.n_dr_0p2_0p4 - 1.0) * (4.721224 - Q.D2_b2)
    if Q.pt_6 < 48.03125 and Q.D2_b2 < 0.1236856:
        z += 0.1460446 * (48.03125 - Q.pt_6) * (0.1236856 - Q.D2_b2)
    if Q.pt_6 < 48.03125 and Q.zdr_0 < 0.01705377:
        z += 0.6515843 * (48.03125 - Q.pt_6) * (0.01705377 - Q.zdr_0)
    if Q.lam2 > 0.0001947983 and Q.planar_flow > 0.06126205:
        z += -584.4365 * (Q.lam2 - 0.0001947983) * (Q.planar_flow - 0.06126205)
    if Q.sj2_dr > 0.1682655 and Q.planar_flow < 0.6947818:
        z += -13.23344 * (Q.sj2_dr - 0.1682655) * (0.6947818 - Q.planar_flow)
    if Q.lam2 > 0.0001947983 and Q.zdr_7 < 0.01265416:
        z += -15608.04 * (Q.lam2 - 0.0001947983) * (0.01265416 - Q.zdr_7)
    if Q.sj2_dr > 0.1682655 and Q.mean_eta < 0.02644207:
        z += -46.90931 * (Q.sj2_dr - 0.1682655) * (0.02644207 - Q.mean_eta)
    if Q.LHA > 0.3127275 and Q.planar_flow < 0.6947818:
        z += -21.49035 * (Q.LHA - 0.3127275) * (0.6947818 - Q.planar_flow)
    if Q.sum_pt > 988.4078 and Q.abseta_3 < 0.04702759:
        z += 0.06769374 * (Q.sum_pt - 988.4078) * (0.04702759 - Q.abseta_3)
    if Q.LHA > 0.1967397 and Q.sj3_dr23 > 0.05931259:
        z += -2.853047 * (Q.LHA - 0.1967397) * (Q.sj3_dr23 - 0.05931259)
    if Q.lam2 > 0.003408389 and Q.pt_2 < 111.75:
        z += 6.985354 * (Q.lam2 - 0.003408389) * (111.75 - Q.pt_2)
    if Q.lam2 > 0.003408389 and Q.pt2_over_pt0 > 0.1852611:
        z += 344.1344 * (Q.lam2 - 0.003408389) * (Q.pt2_over_pt0 - 0.1852611)
    if Q.sj3_dr_min > 0.1278212 and Q.pt_3 < 45.05938:
        z += -0.4775693 * (Q.sj3_dr_min - 0.1278212) * (45.05938 - Q.pt_3)
    if Q.sj2_dr > 0.1591713 and Q.D2_b2 < 0.380911:
        z += -13.83895 * (Q.sj2_dr - 0.1591713) * (0.380911 - Q.D2_b2)
    if Q.lam1 < 0.003377388 and Q.dr_7 < 0.2229947:
        z += -1064.507 * (0.003377388 - Q.lam1) * (0.2229947 - Q.dr_7)
    if Q.M2 < 0.02563286 and Q.pt_2 < 118.5:
        z += 1.096793 * (0.02563286 - Q.M2) * (118.5 - Q.pt_2)
    if Q.M2 < 0.02563286 and Q.z_2 < 0.1818564:
        z += -865.7716 * (0.02563286 - Q.M2) * (0.1818564 - Q.z_2)
    if Q.pt_7 > 20.125 and Q.dr_5 < 0.07530049:
        z += 0.07670539 * (Q.pt_7 - 20.125) * (0.07530049 - Q.dr_5)
    if Q.lam1 > 0.00483998 and Q.sj3_pairmin_over_m < 0.5042684:
        z += -240.284 * (Q.lam1 - 0.00483998) * (0.5042684 - Q.sj3_pairmin_over_m)
    if Q.pt_7 > 20.125 and Q.D2_b2 < 0.2669656:
        z += -0.1061211 * (Q.pt_7 - 20.125) * (0.2669656 - Q.D2_b2)
    if Q.z_7 < 0.06810151 and Q.centroid_offset < 0.04990367:
        z += 395.7754 * (0.06810151 - Q.z_7) * (0.04990367 - Q.centroid_offset)
    if Q.log_sum_pt > 6.670067 and Q.n_dr_0p1_0p2 < 2.0:
        z += 0.9415158 * (Q.log_sum_pt - 6.670067) * (2.0 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.4571061
    if Q.planar_flow < 0.2534037:
        z += -10.74506 * Q.planar_flow + 2.722838
    if Q.e3 < 7.330536e-06:
        z += 125154.9 * Q.e3 - 0.9174525
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += 12.22437 * Q.sj2_dr - 1.582938
    if 0.1591713 <= Q.sj2_dr < 0.2414351:
        z += -6.544033 * Q.sj2_dr + 1.404454
    if 0.2414351 <= Q.sj2_dr < 0.2687922:
        z += -11.13484 * Q.sj2_dr + 2.512836
    if Q.sj2_dr >= 0.2687922:
        z += 6.24403 * Q.sj2_dr - 2.158468
    if Q.pt_7 < 45.75:
        z += 0.02821322 * Q.pt_7 - 1.290755
    if Q.lam1 < 0.003377388:
        z += -673.895 * Q.lam1 + 6.966859
    if 0.003377388 <= Q.lam1 < 0.01200373:
        z += -542.6453 * Q.lam1 + 6.523578
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += -2.215082 * Q.lam1 + 0.03640211
    if Q.sum_z_dr < 0.06108601:
        z += 78.34339 * Q.sum_z_dr - 5.319427
    if 0.06108601 <= Q.sum_z_dr < 0.0717028:
        z += 50.27335 * Q.sum_z_dr - 3.60474
    if Q.e2 < 0.04447357:
        z += 58.53816 * Q.e2 - 2.548247
    if 0.04447357 <= Q.e2 < 0.05028464:
        z += -9.491217 * Q.e2 + 0.4772624
    if Q.tau2 < 0.01133547:
        z += -42.23405 * Q.tau2 + 0.4787429
    if Q.lam2 < 0.000537286:
        z += 474.0291 * Q.lam2 - 0.2546892
    if Q.centroid_offset < 0.01837778:
        z += -85.21562 * Q.centroid_offset + 1.566074
    if Q.centroid_offset >= 0.02355416:
        z += -25.46165 * Q.centroid_offset + 0.5997278
    if Q.sum_z_dr2_top3 < 0.002915531:
        z += -94.48714 * Q.sum_z_dr2_top3 + 0.2754802
    if Q.n_dr_0p05_0p1 >= 6.0:
        z += 0.1377035 * Q.n_dr_0p05_0p1 - 0.826221
    if 0.1426152 <= Q.sj3_dr_max < 0.1789613:
        z += 10.63707 * Q.sj3_dr_max - 1.517008
    if Q.sj3_dr_max >= 0.1789613:
        z += -9.1435 * Q.sj3_dr_max + 2.02295
    if Q.mean_phi2 < 0.002127561:
        z += 152.8554 * Q.mean_phi2 - 0.3252092
    if Q.tau1 < 0.07283629:
        z += -25.56237 * Q.tau1 + 1.861868
    if Q.sum_pt_top5 >= 506.875:
        z += 0.002416177 * Q.sum_pt_top5 - 1.2247
    if Q.tau21 < 0.1357675:
        z += 6.914782 * Q.tau21 - 0.9388027
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.007756107 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.centroid_offset < 0.04990367:
        z += -36.82755 * (0.2534037 - Q.planar_flow) * (0.04990367 - Q.centroid_offset)
    if Q.lam1 < 0.01643375 and Q.centroid_offset > 0.01837778:
        z += -4111.116 * (0.01643375 - Q.lam1) * (Q.centroid_offset - 0.01837778)
    if Q.sj2_dr > 0.2414351 and Q.sum_pt < 739.5:
        z += 0.03093877 * (Q.sj2_dr - 0.2414351) * (739.5 - Q.sum_pt)
    if Q.lam1 < 0.01643375 and Q.log_sum_pt < 6.423044:
        z += -136.6166 * (0.01643375 - Q.lam1) * (6.423044 - Q.log_sum_pt)
    if Q.lam1 < 0.003377388 and Q.centroid_offset < 0.02685622:
        z += -41625.86 * (0.003377388 - Q.lam1) * (0.02685622 - Q.centroid_offset)
    if Q.centroid_offset < 0.01837778 and Q.sum_pt < 937.0312:
        z += -0.1690923 * (0.01837778 - Q.centroid_offset) * (937.0312 - Q.sum_pt)
    if Q.lam1 < 0.01200373 and Q.centroid_offset > 0.01627885:
        z += -5901.471 * (0.01200373 - Q.lam1) * (Q.centroid_offset - 0.01627885)
    if Q.lam1 < 0.01643375 and Q.lam2 < 0.001130645:
        z += 138185.6 * (0.01643375 - Q.lam1) * (0.001130645 - Q.lam2)
    if Q.lam1 < 0.003377388 and Q.planar_flow < 0.2534037:
        z += -2282.097 * (0.003377388 - Q.lam1) * (0.2534037 - Q.planar_flow)
    if Q.pt_7 < 45.75 and Q.D2_b2 < 1.345805:
        z += -0.005549085 * (45.75 - Q.pt_7) * (1.345805 - Q.D2_b2)
    if Q.pt_6 < 33.6875 and Q.D2_b2 < 4.721224:
        z += -0.01142033 * (33.6875 - Q.pt_6) * (4.721224 - Q.D2_b2)
    if Q.pt_7 < 45.75 and Q.centroid_offset > 0.01096064:
        z += 0.4918897 * (45.75 - Q.pt_7) * (Q.centroid_offset - 0.01096064)
    if Q.sj2_dr > 0.1778793 and Q.eccentricity > 0.927072:
        z += -36.01405 * (Q.sj2_dr - 0.1778793) * (Q.eccentricity - 0.927072)
    if Q.pt_7 < 45.75 and Q.n_dr_0p2_0p4 < 1.0:
        z += -0.01410478 * (45.75 - Q.pt_7) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.planar_flow < 0.2534037 and Q.D3 < 3.916009:
        z += -0.9732737 * (0.2534037 - Q.planar_flow) * (3.916009 - Q.D3)
    if Q.lam2 < 0.000537286 and Q.C3 > 0.04473419:
        z += -30124.89 * (0.000537286 - Q.lam2) * (Q.C3 - 0.04473419)
    if Q.lam1 < 0.01200373 and Q.dr1_7 > 0.2411619:
        z += -2021.521 * (0.01200373 - Q.lam1) * (Q.dr1_7 - 0.2411619)
    if Q.sum_z_dr < 0.06108601 and Q.dr1_7 > 0.2025074:
        z += 507.7323 * (0.06108601 - Q.sum_z_dr) * (Q.dr1_7 - 0.2025074)
    if Q.zdr_0 < 0.004918231 and Q.dr0_7 > 0.09118326:
        z += -2345.927 * (0.004918231 - Q.zdr_0) * (Q.dr0_7 - 0.09118326)
    if Q.e3 < 7.330536e-06 and Q.D2_b2 < 0.716559:
        z += 321834.1 * (7.330536e-06 - Q.e3) * (0.716559 - Q.D2_b2)
    if Q.lam1 < 0.00595415 and Q.D2_b2 < 0.716559:
        z += -612.5136 * (0.00595415 - Q.lam1) * (0.716559 - Q.D2_b2)
    if Q.lam1 < 0.008375572 and Q.D2_b2 < 0.716559:
        z += 185.6038 * (0.008375572 - Q.lam1) * (0.716559 - Q.D2_b2)
    if Q.lam1 < 0.01643375 and Q.D2_b2 < 0.380911:
        z += -110.8483 * (0.01643375 - Q.lam1) * (0.380911 - Q.D2_b2)
    if Q.sum_pt_top5 > 506.875 and Q.D2_b2 < 0.380911:
        z += 0.006596329 * (Q.sum_pt_top5 - 506.875) * (0.380911 - Q.D2_b2)
    if Q.sum_pt_top5 > 506.875 and Q.mean_phi > 0.01746433:
        z += -0.3057031 * (Q.sum_pt_top5 - 506.875) * (Q.mean_phi - 0.01746433)
    if Q.sum_pt_top5 > 506.875 and Q.e3 < 2.371297e-05:
        z += -148.8965 * (Q.sum_pt_top5 - 506.875) * (2.371297e-05 - Q.e3)
    if Q.pt_7 < 45.75 and Q.e3 < 1.050302e-05:
        z += 3172.087 * (45.75 - Q.pt_7) * (1.050302e-05 - Q.e3)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.3948339
    if 0.1245537 <= Q.sum_z_dr < 0.1484084:
        z += 134.1059 * Q.sum_z_dr - 16.70339
    if Q.sum_z_dr >= 0.1484084:
        z += 185.4496 * Q.sum_z_dr - 24.32323
    if Q.n_dr_0p2_0p4 >= 2.0:
        z += 0.6981567 * Q.n_dr_0p2_0p4 - 1.396313
    if Q.sd_rg >= 0.324646:
        z += 59.79136 * Q.sd_rg - 19.41103
    if Q.centroid_offset >= 0.03776099:
        z += 27.81455 * Q.centroid_offset - 1.050305
    if Q.zdr_0 >= 0.03981924:
        z += -31.56357 * Q.zdr_0 + 1.256838
    if Q.LHA >= 0.3855647:
        z += -55.45965 * Q.LHA + 21.38328
    if Q.e2 >= 0.06344108:
        z += -32.88763 * Q.e2 + 2.086427
    if Q.sum_z_dr > 0.1245537 and Q.log_sum_pt > 6.502799:
        z += 523.9494 * (Q.sum_z_dr - 0.1245537) * (Q.log_sum_pt - 6.502799)
    if Q.sum_z_dr > 0.1245537 and Q.C2_b2 > 0.009032972:
        z += 974.981 * (Q.sum_z_dr - 0.1245537) * (Q.C2_b2 - 0.009032972)
    if Q.sd_rg > 0.324646 and Q.log_sum_pt < 6.896095:
        z += -214.4191 * (Q.sd_rg - 0.324646) * (6.896095 - Q.log_sum_pt)
    if Q.sum_z_dr > 0.1245537 and Q.pt_6 < 62.25:
        z += -1.001827 * (Q.sum_z_dr - 0.1245537) * (62.25 - Q.pt_6)
    if Q.sum_z_dr > 0.1484084 and Q.log_sum_pt > 6.502799:
        z += -516.8504 * (Q.sum_z_dr - 0.1484084) * (Q.log_sum_pt - 6.502799)
    if Q.sum_z_dr > 0.1484084 and Q.sum_pt < 588.5859:
        z += -0.1049274 * (Q.sum_z_dr - 0.1484084) * (588.5859 - Q.sum_pt)
    if Q.sd_rg > 0.324646 and Q.log_sum_pt < 6.701242:
        z += 193.1392 * (Q.sd_rg - 0.324646) * (6.701242 - Q.log_sum_pt)
    if Q.centroid_offset > 0.03776099 and Q.sum_pt < 615.875:
        z += -0.07692887 * (Q.centroid_offset - 0.03776099) * (615.875 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 > 2.0 and Q.sum_pt < 667.0063:
        z += -0.00175822 * (Q.n_dr_0p2_0p4 - 2.0) * (667.0063 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 > 2.0 and Q.lam2 < 0.003408389:
        z += -129.4256 * (Q.n_dr_0p2_0p4 - 2.0) * (0.003408389 - Q.lam2)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.2984715
    if Q.sum_z_dr < 0.1484084:
        z += -82.15522 * Q.sum_z_dr + 12.19253
    if Q.lam1 < 0.01643375:
        z += -197.0212 * Q.lam1 + 3.237797
    if 687.4375 <= Q.sum_pt_top5 < 791.125:
        z += 0.002194349 * Q.sum_pt_top5 - 1.508478
    if 791.125 <= Q.sum_pt_top5 < 839.9547:
        z += 0.007815934 * Q.sum_pt_top5 - 5.955854
    if Q.sum_pt_top5 >= 839.9547:
        z += 0.009247013 * Q.sum_pt_top5 - 7.157896
    if 0.04624032 <= Q.z_7 < 0.0586137:
        z += 35.63189 * Q.z_7 - 1.64763
    if Q.z_7 >= 0.0586137:
        z += 49.67593 * Q.z_7 - 2.470803
    if Q.pt_6 < 27.57812:
        z += 0.09994289 * Q.pt_6 - 3.379252
    if 27.57812 <= Q.pt_6 < 50.25:
        z += 0.02747962 * Q.pt_6 - 1.380851
    if Q.sj3_dr_max >= 0.233678:
        z += -5.379931 * Q.sj3_dr_max + 1.257171
    if Q.pt_5 < 24.57812:
        z += 0.232587 * Q.pt_5 - 5.716551
    if Q.e3 < 8.147744e-05:
        z += -4736.604 * Q.e3 + 0.3859264
    if Q.e2 < 0.08000524:
        z += 18.74184 * Q.e2 - 1.499445
    if Q.z_5 < 0.02818362:
        z += -94.99788 * Q.z_5 + 2.677384
    if Q.sj3_dr_min < 0.1278212:
        z += 2.932298 * Q.sj3_dr_min - 0.3748098
    if Q.M2 < 0.0294431:
        z += -18.42939 * Q.M2 + 0.5426183
    if Q.pt_7 < 31.85938:
        z += 0.03051576 * Q.pt_7 - 1.48669
    if 31.85938 <= Q.pt_7 < 48.71875:
        z += -0.05992252 * Q.pt_7 + 1.394617
    if Q.pt_7 >= 48.71875:
        z += -0.09043828 * Q.pt_7 + 2.881307
    if Q.tau1 < 0.09538712:
        z += 16.15484 * Q.tau1 - 2.017642
    if 0.09538712 <= Q.tau1 < 0.1136369:
        z += 26.11973 * Q.tau1 - 2.968165
    if Q.pt_2 < 61.53125:
        z += 0.02263407 * Q.pt_2 - 1.392703
    if Q.z_top5 >= 0.8770155:
        z += -7.6767 * Q.z_top5 + 6.732585
    if Q.log_sum_pt >= 6.670067:
        z += -3.487963 * Q.log_sum_pt + 23.26495
    if Q.C2_b2 < 0.009032972:
        z += 75.00626 * Q.C2_b2 - 0.6775294
    if Q.z_dr_0_0p05 >= 0.6080732:
        z += 1.459976 * Q.z_dr_0_0p05 - 0.8877726
    if Q.sum_z_dr < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -81.98095 * (0.1484084 - Q.sum_z_dr) * (6.804164 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.1484084 and Q.pt_7 < 38.53125:
        z += -0.573726 * (0.1484084 - Q.sum_z_dr) * (38.53125 - Q.pt_7)
    if Q.sum_z_dr < 0.1484084 and Q.lam2 < 0.0003061234:
        z += -20410.2 * (0.1484084 - Q.sum_z_dr) * (0.0003061234 - Q.lam2)
    if Q.sj3_dr_max > 0.233678 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -33.44603 * (Q.sj3_dr_max - 0.233678) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.lam1 < 0.01643375 and Q.z_dr_0_0p05 > 0.3658817:
        z += -117.1259 * (0.01643375 - Q.lam1) * (Q.z_dr_0_0p05 - 0.3658817)
    if Q.sum_pt > 988.4078 and Q.z_6 > 0.03932388:
        z += 0.254214 * (Q.sum_pt - 988.4078) * (Q.z_6 - 0.03932388)
    if Q.sum_pt_top5 > 687.4375 and Q.pt_7 < 40.04062:
        z += 0.0004998436 * (Q.sum_pt_top5 - 687.4375) * (40.04062 - Q.pt_7)
    if Q.lam1 < 0.01643375 and Q.pt_7 < 25.57812:
        z += -8.059869 * (0.01643375 - Q.lam1) * (25.57812 - Q.pt_7)
    if Q.sum_pt > 988.4078 and Q.M3 < 0.07092092:
        z += -0.1652685 * (Q.sum_pt - 988.4078) * (0.07092092 - Q.M3)
    if Q.e3 < 8.147744e-05 and Q.centroid_offset < 0.03776099:
        z += -621105.0 * (8.147744e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.z_7 > 0.0586137 and Q.lam2 > 4.64158e-05:
        z += -3912.442 * (Q.z_7 - 0.0586137) * (Q.lam2 - 4.64158e-05)
    if Q.z_7 > 0.0586137 and Q.pt_4 < 31.125:
        z += -748.7341 * (Q.z_7 - 0.0586137) * (31.125 - Q.pt_4)
    if Q.pt_6 < 50.25 and Q.zdr_0 > 0.001113887:
        z += 0.5350939 * (50.25 - Q.pt_6) * (Q.zdr_0 - 0.001113887)
    if Q.e2 < 0.08000524 and Q.pt_4 < 60.03125:
        z += 0.133542 * (0.08000524 - Q.e2) * (60.03125 - Q.pt_4)
    if Q.sum_pt_top5 > 687.4375 and Q.dr_2 < 0.08525808:
        z += 0.01060156 * (Q.sum_pt_top5 - 687.4375) * (0.08525808 - Q.dr_2)
    if Q.tau1 < 0.1136369 and Q.pt_2 < 69.875:
        z += 0.2227996 * (0.1136369 - Q.tau1) * (69.875 - Q.pt_2)
    if Q.pt_7 > 31.85938 and Q.max_dr < 0.1598486:
        z += 0.1517268 * (Q.pt_7 - 31.85938) * (0.1598486 - Q.max_dr)
    if Q.log_sum_pt < 6.46415 and Q.pt_5 < 29.875:
        z += -1.455809 * (6.46415 - Q.log_sum_pt) * (29.875 - Q.pt_5)
    if Q.sum_pt_top5 > 791.125 and Q.zdr_1 > 0.007459436:
        z += -0.3677722 * (Q.sum_pt_top5 - 791.125) * (Q.zdr_1 - 0.007459436)
    if Q.pt_7 > 31.85938 and Q.zdr_1 < 0.01228369:
        z += 3.224476 * (Q.pt_7 - 31.85938) * (0.01228369 - Q.zdr_1)
    if Q.z_7 > 0.04624032 and Q.zdr_1 < 0.01084112:
        z += -2357.621 * (Q.z_7 - 0.04624032) * (0.01084112 - Q.zdr_1)
    if Q.log_sum_pt < 6.46415 and Q.zdr_1 < 0.01084112:
        z += 317.9257 * (6.46415 - Q.log_sum_pt) * (0.01084112 - Q.zdr_1)
    if Q.sum_pt > 988.4078 and Q.zdr_1 < 0.02788929:
        z += -0.3647214 * (Q.sum_pt - 988.4078) * (0.02788929 - Q.zdr_1)
    if Q.pt_7 > 31.85938 and Q.z_dr_0p05_0p1 < 0.7509095:
        z += 0.02172369 * (Q.pt_7 - 31.85938) * (0.7509095 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_14(Q):
    z = -4.567949
    if Q.planar_flow < 0.1115136:
        z += -19.14495 * Q.planar_flow + 2.134922
    if Q.z_dr_0p05_0p1 < 0.163898:
        z += -0.468978 * Q.z_dr_0p05_0p1 + 0.2758809
    if 0.163898 <= Q.z_dr_0p05_0p1 < 0.5882598:
        z += -1.78634 * Q.z_dr_0p05_0p1 + 0.4917939
    if Q.z_dr_0p05_0p1 >= 0.5882598:
        z += -1.317362 * Q.z_dr_0p05_0p1 + 0.215913
    if Q.psi_0p1 >= 0.9761279:
        z += -11.34767 * Q.psi_0p1 + 11.07678
    if Q.lam2 < 0.003408389:
        z += 407.9913 * Q.lam2 - 1.390593
    if Q.e3 < 5.334511e-05:
        z += 23833.72 * Q.e3 - 1.773913
    if 5.334511e-05 <= Q.e3 < 8.147744e-05:
        z += 17862.03 * Q.e3 - 1.455352
    if Q.lam1 < 0.002464291:
        z += 4116.168 * Q.lam1 - 7.377305
    if 0.002464291 <= Q.lam1 < 0.006506576:
        z += 287.8662 * Q.lam1 + 2.056745
    if 0.006506576 <= Q.lam1 < 0.00733008:
        z += -215.1185 * Q.lam1 + 5.329453
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += -619.9852 * Q.lam1 + 8.297158
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += -193.0068 * Q.lam1 + 3.171826
    if Q.sd_rg < 0.06545715:
        z += 3.507636 * Q.sd_rg + 0.7857546
    if 0.06545715 <= Q.sd_rg < 0.1116471:
        z += -12.33674 * Q.sd_rg + 1.822882
    if 0.1116471 <= Q.sd_rg < 0.1449048:
        z += -17.13084 * Q.sd_rg + 2.358129
    if 0.1449048 <= Q.sd_rg < 0.1572969:
        z += -20.5591 * Q.sd_rg + 2.854901
    if 0.1572969 <= Q.sd_rg < 0.1667615:
        z += -32.58889 * Q.sd_rg + 4.747149
    if 0.1667615 <= Q.sd_rg < 0.1773029:
        z += 4.462781 * Q.sd_rg - 1.431644
    if 0.1773029 <= Q.sd_rg < 0.1876504:
        z += -10.28424 * Q.sd_rg + 1.183046
    if 0.1876504 <= Q.sd_rg < 0.2330919:
        z += -40.49422 * Q.sd_rg + 6.851962
    if Q.sd_rg >= 0.2330919:
        z += -45.66804 * Q.sd_rg + 8.057935
    if 0.02689598 <= Q.sum_z_dr < 0.08065885:
        z += 64.62941 * Q.sum_z_dr - 1.738271
    if 0.08065885 <= Q.sum_z_dr < 0.08723651:
        z += -0.6331406 * Q.sum_z_dr + 3.525731
    if Q.sum_z_dr >= 0.08723651:
        z += -116.5492 * Q.sum_z_dr + 13.63785
    if Q.e2 < 0.01289969:
        z += -111.9185 * Q.e2 + 5.392718
    if 0.01289969 <= Q.e2 < 0.03556091:
        z += -147.4752 * Q.e2 + 5.851388
    if 0.03556091 <= Q.e2 < 0.04110972:
        z += -109.3993 * Q.e2 + 4.497374
    if Q.tau1 < 0.09538712:
        z += 0.2938137 * Q.tau1 - 1.307612
    if 0.09538712 <= Q.tau1 < 0.1136369:
        z += 33.41728 * Q.tau1 - 4.467164
    if 0.1136369 <= Q.tau1 < 0.1416226:
        z += 23.93104 * Q.tau1 - 3.389177
    if Q.centroid_offset >= 0.04990367:
        z += -1588.226 * Q.centroid_offset + 79.25829
    if Q.LHA >= 0.3033137:
        z += 25.93348 * Q.LHA - 7.865981
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 147.0287 * Q.sum_z_dr2_top3 - 0.3163423
    if Q.n_dr_0p05_0p1 < 5.0:
        z += 0.1035133 * Q.n_dr_0p05_0p1 - 0.4532713
    if 5.0 <= Q.n_dr_0p05_0p1 < 6.0:
        z += -0.06429538 * Q.n_dr_0p05_0p1 + 0.3857723
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -8877.88 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += 373.0061 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00595415:
        z += 4375.388 * (0.1115136 - Q.planar_flow) * (0.00595415 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.sum_pt < 763.825:
        z += -0.03477363 * (0.1115136 - Q.planar_flow) * (763.825 - Q.sum_pt)
    if Q.lam2 < 0.003408389 and Q.sd_rg > 0.2037854:
        z += -6471.316 * (0.003408389 - Q.lam2) * (Q.sd_rg - 0.2037854)
    if Q.planar_flow < 0.1115136 and Q.sj2_zsoft < 0.2832687:
        z += 16.7132 * (0.1115136 - Q.planar_flow) * (0.2832687 - Q.sj2_zsoft)
    if Q.lam2 < 0.003408389 and Q.centroid_offset > 0.02685622:
        z += -20108.62 * (0.003408389 - Q.lam2) * (Q.centroid_offset - 0.02685622)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.z_dr_0p1_0p2 < 0.04510668:
        z += 11.8028 * (0.5882598 - Q.z_dr_0p05_0p1) * (0.04510668 - Q.z_dr_0p1_0p2)
    if Q.lam2 < 0.003408389 and Q.LHA > 0.1767241:
        z += 3896.63 * (0.003408389 - Q.lam2) * (Q.LHA - 0.1767241)
    if Q.lam2 < 0.003408389 and Q.sd_rg > 0.1572969:
        z += 19139.21 * (0.003408389 - Q.lam2) * (Q.sd_rg - 0.1572969)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.n_dr_0p2_0p4 < 1.0:
        z += 3.881252 * (Q.z_dr_0p05_0p1 - 0.163898) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.log_sum_pt < 6.670067:
        z += -2.396101 * (Q.z_dr_0p05_0p1 - 0.163898) * (6.670067 - Q.log_sum_pt)
    if Q.lam1 < 0.01643375 and Q.centroid_offset > 0.02685622:
        z += -6759.763 * (0.01643375 - Q.lam1) * (Q.centroid_offset - 0.02685622)
    if Q.lam2 < 0.003408389 and Q.n_dr_0p2_0p4 > 1.0:
        z += -87.32877 * (0.003408389 - Q.lam2) * (Q.n_dr_0p2_0p4 - 1.0)
    if Q.lam1 < 0.006506576 and Q.centroid_offset > 0.03117077:
        z += 30799.54 * (0.006506576 - Q.lam1) * (Q.centroid_offset - 0.03117077)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.z_dr_0p2_0p4 < 0.20552:
        z += -9.950348 * (Q.z_dr_0p05_0p1 - 0.163898) * (0.20552 - Q.z_dr_0p2_0p4)
    if Q.sd_rg < 0.06545715 and Q.psi_0p3 < 1.0:
        z += -273.1732 * (0.06545715 - Q.sd_rg) * (1.0 - Q.psi_0p3)
    if Q.lam1 < 0.006506576 and Q.sj3_dr23 > 0.1974628:
        z += 3298.102 * (0.006506576 - Q.lam1) * (Q.sj3_dr23 - 0.1974628)
    if Q.lam2 < 0.003408389 and Q.z_dr_0p2_0p4 > 0.0:
        z += 1209.672 * (0.003408389 - Q.lam2) * (Q.z_dr_0p2_0p4 - 0.0)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.psi_0p3 < 1.0:
        z += -1343.197 * (Q.z_dr_0p05_0p1 - 0.163898) * (1.0 - Q.psi_0p3)
    if Q.tau1 < 0.1136369 and Q.D2 < 0.415246:
        z += -171.496 * (0.1136369 - Q.tau1) * (0.415246 - Q.D2)
    if Q.e2 < 0.04110972 and Q.D2 < 0.415246:
        z += 1058.616 * (0.04110972 - Q.e2) * (0.415246 - Q.D2)
    if Q.lam1 < 0.01643375 and Q.D2 < 0.415246:
        z += 368.952 * (0.01643375 - Q.lam1) * (0.415246 - Q.D2)
    if Q.planar_flow < 0.1115136 and Q.sd_nremoved < 1.0:
        z += -10.46144 * (0.1115136 - Q.planar_flow) * (1.0 - Q.sd_nremoved)
    if Q.lam1 < 0.00733008 and Q.D2 < 0.415246:
        z += -4058.831 * (0.00733008 - Q.lam1) * (0.415246 - Q.D2)
    return max(0.0, z)


def neuron_15(Q):
    z = -2.824271
    if Q.N2 < 0.2233283:
        z += 2.601444 * Q.N2 - 0.5809759
    if Q.zdr_0 < 0.009233892:
        z += 111.2518 * Q.zdr_0 - 1.238654
    if 0.009233892 <= Q.zdr_0 < 0.0211821:
        z += 17.69023 * Q.zdr_0 - 0.3747162
    if Q.lam2 < 0.003408389:
        z += -543.4883 * Q.lam2 + 1.85242
    if Q.lam1 < 0.002464291:
        z += -723.2277 * Q.lam1 + 1.548948
    if 0.002464291 <= Q.lam1 < 0.005433361:
        z += 340.9264 * Q.lam1 - 1.073438
    if 0.005433361 <= Q.lam1 < 0.006506576:
        z += 622.2549 * Q.lam1 - 2.601997
    if 0.006506576 <= Q.lam1 < 0.00733008:
        z += 562.1142 * Q.lam1 - 2.210687
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += -2.830162 * Q.lam1 + 1.9304
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -170.0896 * Q.lam1 + 3.331293
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += -291.1011 * Q.lam1 + 4.783883
    if Q.tau1 < 0.04369778:
        z += -12.69629 * Q.tau1 + 0.726254
    if 0.04369778 <= Q.tau1 < 0.05356915:
        z += -17.36884 * Q.tau1 + 0.9304342
    if 0.08865369 <= Q.tau1 < 0.09538712:
        z += 34.05485 * Q.tau1 - 3.019088
    if 0.09538712 <= Q.tau1 < 0.1027642:
        z += 20.00014 * Q.tau1 - 1.67845
    if 0.1027642 <= Q.tau1 < 0.1136369:
        z += 4.721909 * Q.tau1 - 0.1083941
    if 0.1136369 <= Q.tau1 < 0.1416226:
        z += 26.87323 * Q.tau1 - 2.625601
    if Q.tau1 >= 0.1416226:
        z += -89.29707 * Q.tau1 + 13.82674
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += -744.5307 * Q.sum_z_dr2_top3 + 1.601908
    if Q.sum_z_dr2_top2 < 0.007639643:
        z += -99.59494 * Q.sum_z_dr2_top2 + 0.7608698
    if Q.eccentricity >= 0.9458207:
        z += 3.740569 * Q.eccentricity - 3.537907
    if Q.D3 < 0.2213841:
        z += 7.559711 * Q.D3 - 1.6736
    if Q.sum_z_dr < 0.03360421:
        z += 201.1144 * Q.sum_z_dr - 12.52248
    if 0.03360421 <= Q.sum_z_dr < 0.0717028:
        z += 141.18 * Q.sum_z_dr - 10.50843
    if 0.0717028 <= Q.sum_z_dr < 0.08723651:
        z += 32.9819 * Q.sum_z_dr - 2.75032
    if 0.08723651 <= Q.sum_z_dr < 0.1019409:
        z += -8.630514 * Q.sum_z_dr + 0.8798026
    if Q.z_dr_0_0p05 < 0.1515405:
        z += -3.341355 * Q.z_dr_0_0p05 + 0.5063504
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += -3.200891 * Q.z_dr_0p05_0p1 + 2.403579
    if Q.psi_0p1 >= 0.7143804:
        z += 3.299296 * Q.psi_0p1 - 2.356952
    if Q.e2 < 0.032347:
        z += -109.6471 * Q.e2 + 4.129024
    if 0.032347 <= Q.e2 < 0.04110972:
        z += -66.44868 * Q.e2 + 2.731687
    if Q.LHA < 0.1329373:
        z += 6.409727 * Q.LHA + 0.2991896
    if 0.1329373 <= Q.LHA < 0.3033137:
        z += -7.214104 * Q.LHA + 2.110305
    if 0.3033137 <= Q.LHA < 0.3127275:
        z += 8.267943 * Q.LHA - 2.585613
    if 0.181053 <= Q.sj3_dr13 < 0.249546:
        z += 3.375463 * Q.sj3_dr13 - 0.6111376
    if 0.249546 <= Q.sj3_dr13 < 0.2871004:
        z += -3.72547 * Q.sj3_dr13 + 1.160871
    if Q.sj3_dr13 >= 0.2871004:
        z += -11.15159 * Q.sj3_dr13 + 3.292913
    if Q.e3 < 7.330536e-06:
        z += 66479.68 * Q.e3 - 0.4873317
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1872617:
        z += -364.4244 * (0.2233283 - Q.N2) * (0.1872617 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.lam1 > 0.008375572:
        z += -2217.73 * (0.2233283 - Q.N2) * (Q.lam1 - 0.008375572)
    if Q.N2 < 0.2233283 and Q.LHA > 0.2931906:
        z += 151.6758 * (0.2233283 - Q.N2) * (Q.LHA - 0.2931906)
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1591713:
        z += 582.3833 * (0.2233283 - Q.N2) * (0.1591713 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.LHA > 0.4235925:
        z += -2202.285 * (0.2233283 - Q.N2) * (Q.LHA - 0.4235925)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3255822:
        z += -225.3974 * (0.2233283 - Q.N2) * (Q.LHA - 0.3255822)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3855647:
        z += -189.0689 * (0.2233283 - Q.N2) * (Q.LHA - 0.3855647)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 402.625:
        z += 0.01202604 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 402.625)
    if Q.N2 < 0.2233283 and Q.z_7 < 0.02320757:
        z += -814.6414 * (0.2233283 - Q.N2) * (0.02320757 - Q.z_7)
    if Q.sum_z_dr2_top3 < 0.002151568 and Q.planar_flow < 0.1950135:
        z += -5454.635 * (0.002151568 - Q.sum_z_dr2_top3) * (0.1950135 - Q.planar_flow)
    if Q.lam2 < 0.003408389 and Q.zdr_1 < 0.01849752:
        z += -6191.954 * (0.003408389 - Q.lam2) * (0.01849752 - Q.zdr_1)
    if Q.eccentricity > 0.9458207 and Q.mean_phi > -0.01753483:
        z += -140.8948 * (Q.eccentricity - 0.9458207) * (Q.mean_phi - -0.01753483)
    if Q.sum_z_dr2_top2 < 0.007639643 and Q.D3 < 0.2213841:
        z += 1396.259 * (0.007639643 - Q.sum_z_dr2_top2) * (0.2213841 - Q.D3)
    if Q.eccentricity > 0.9458207 and Q.sum_pt_top5 > 367.5938:
        z += 0.0503517 * (Q.eccentricity - 0.9458207) * (Q.sum_pt_top5 - 367.5938)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.sj3_dr23 > 0.1797097:
        z += -85.28551 * (Q.z_dr_0p05_0p1 - 0.7509095) * (Q.sj3_dr23 - 0.1797097)
    if Q.lam1 < 0.008375572 and Q.sj3_dr23 > 0.1974628:
        z += 3211.878 * (0.008375572 - Q.lam1) * (Q.sj3_dr23 - 0.1974628)
    if Q.lam1 < 0.008375572 and Q.D2 < 0.875672:
        z += -995.622 * (0.008375572 - Q.lam1) * (0.875672 - Q.D2)
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += 896.03 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.lam1 < 0.01643375 and Q.D2 < 0.875672:
        z += 99.19271 * (0.01643375 - Q.lam1) * (0.875672 - Q.D2)
    if Q.lam1 < 0.00595415 and Q.sj3_dr23 > 0.1974628:
        z += -1611.408 * (0.00595415 - Q.lam1) * (Q.sj3_dr23 - 0.1974628)
    if Q.lam1 < 0.01643375 and Q.centroid_offset > 0.04990367:
        z += -7025.337 * (0.01643375 - Q.lam1) * (Q.centroid_offset - 0.04990367)
    if Q.lam1 < 0.01643375 and Q.C2_b2 > 0.009032972:
        z += -6420.992 * (0.01643375 - Q.lam1) * (Q.C2_b2 - 0.009032972)
    if Q.lam1 < 0.00733008 and Q.centroid_offset > 0.03776099:
        z += 42705.75 * (0.00733008 - Q.lam1) * (Q.centroid_offset - 0.03776099)
    if Q.lam1 < 0.01643375 and Q.centroid_offset > 0.03776099:
        z += -14984.6 * (0.01643375 - Q.lam1) * (Q.centroid_offset - 0.03776099)
    if Q.zdr_0 < 0.009233892 and Q.D2_b2 < 0.716559:
        z += 260.209 * (0.009233892 - Q.zdr_0) * (0.716559 - Q.D2_b2)
    if Q.tau1 > 0.1027642 and Q.D2_b2 < 0.5327104:
        z += 146.5941 * (Q.tau1 - 0.1027642) * (0.5327104 - Q.D2_b2)
    if Q.tau1 > 0.1416226 and Q.D2_b2 < 0.5327104:
        z += 230.0033 * (Q.tau1 - 0.1416226) * (0.5327104 - Q.D2_b2)
    if Q.tau1 > 0.08865369 and Q.D2_b2 < 0.5327104:
        z += -59.33326 * (Q.tau1 - 0.08865369) * (0.5327104 - Q.D2_b2)
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
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:8] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:8] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:8] + [0.0] * 0
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
