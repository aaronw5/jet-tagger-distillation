"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.7% (the network: 65.8%); same class as the network for 90.3% of jets.

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
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z1                 pT share of subjet 1 of 3 (by pT)
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_nremoved            number of branches removed by soft drop
  Q.abseta_0               |Δη| of particle 0
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
  Q.dr_2                   ΔR of particle 2 from the jet axis
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
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.mean_eta               pT-weighted mean Δη
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
        pt_3=pt[3],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z1=subjets(3)["z"][0],
        z_top3_slots=sum(pt[:3]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_7=z[7] * dr[7],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_nremoved=softdrop("removed"),
        abseta_0=abs(eta[0]),
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
        dr_2=dr[2] if pt[2] > 0 else 0.0,
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
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
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
    )


def neuron_0(Q):
    z = -1.531342
    if Q.planar_flow < 0.1484197:
        z += -3.327392 * Q.planar_flow + 0.4938506
    if Q.lam1 < 0.004183811:
        z += 604.6003 * Q.lam1 - 0.6255626
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += -139.6057 * Q.lam1 + 2.488055
    if 0.00595415 <= Q.lam1 < 0.01200373:
        z += -273.8739 * Q.lam1 + 3.287507
    if Q.sum_pt < 763.825:
        z += 0.006694192 * Q.sum_pt - 5.113191
    if Q.tau1 < 0.05356915:
        z += -34.23488 * Q.tau1 + 1.833934
    if Q.sum_z_dr2_top3 < 0.007929074:
        z += 58.97569 * Q.sum_z_dr2_top3 - 0.4676226
    if Q.sj3_dr_max < 0.1426152:
        z += 1.528347 * Q.sj3_dr_max - 0.5282667
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 22.1526 * Q.sj3_dr_max - 3.469599
    if 0.169029 <= Q.sj3_dr_max < 0.1986272:
        z += 5.324661 * Q.sj3_dr_max - 0.6251877
    if 0.1986272 <= Q.sj3_dr_max < 0.3456459:
        z += -12.39715 * Q.sj3_dr_max + 2.894846
    if Q.sj3_dr_max >= 0.3456459:
        z += -13.9255 * Q.sj3_dr_max + 3.423113
    if Q.centroid_offset >= 0.006789738:
        z += -74.16845 * Q.centroid_offset + 0.5035844
    if Q.sum_z_dr < 0.08065885:
        z += 26.97678 * Q.sum_z_dr - 2.175916
    if Q.e2 < 0.02045966:
        z += -52.91323 * Q.e2 + 1.082587
    if Q.e2 >= 0.04110972:
        z += 40.02513 * Q.e2 - 1.645422
    if 6.766778 <= Q.log_sum_pt < 6.842717:
        z += -9.490622 * Q.log_sum_pt + 64.22093
    if Q.log_sum_pt >= 6.842717:
        z += -35.65864 * Q.log_sum_pt + 243.2812
    if Q.sum_pt_top5 >= 791.125:
        z += 0.02037336 * Q.sum_pt_top5 - 16.11787
    if Q.tau2 < 0.01713288:
        z += 123.7991 * Q.tau2 - 2.121035
    if Q.e3 < 8.147744e-05:
        z += -17947.52 * Q.e3 + 3.515948
    if 8.147744e-05 <= Q.e3 < 0.0005116989:
        z += -4773.425 * Q.e3 + 2.442556
    if 0.1546891 <= Q.LHA < 0.3033137:
        z += 5.494792 * Q.LHA - 0.8499844
    if Q.LHA >= 0.3033137:
        z += -31.22118 * Q.LHA + 10.28647
    if Q.pt_6 < 27.57812:
        z += 0.07032252 * Q.pt_6 - 1.939363
    if Q.n_dr_0p2_0p4 < 1.0:
        z += -0.5375412 * Q.n_dr_0p2_0p4 + 0.5375412
    if Q.planar_flow < 0.1484197 and Q.centroid_offset < 0.04990367:
        z += 61.07003 * (0.1484197 - Q.planar_flow) * (0.04990367 - Q.centroid_offset)
    if Q.sum_pt < 763.825 and Q.centroid_offset > 0.01096064:
        z += 0.1801378 * (763.825 - Q.sum_pt) * (Q.centroid_offset - 0.01096064)
    if Q.lam1 < 0.004183811 and Q.sum_pt < 763.825:
        z += 2.19421 * (0.004183811 - Q.lam1) * (763.825 - Q.sum_pt)
    if Q.sum_pt < 763.825 and Q.e3 < 0.0005116989:
        z += -7.273711 * (763.825 - Q.sum_pt) * (0.0005116989 - Q.e3)
    if Q.lam1 < 0.004183811 and Q.D2 < 0.7459513:
        z += -6352.808 * (0.004183811 - Q.lam1) * (0.7459513 - Q.D2)
    if Q.sj3_dr_max > 0.1426152 and Q.D2 > 0.2568137:
        z += 1.45151 * (Q.sj3_dr_max - 0.1426152) * (Q.D2 - 0.2568137)
    if Q.log_sum_pt > 6.842717 and Q.dr_2 < 0.03843804:
        z += 129.1958 * (Q.log_sum_pt - 6.842717) * (0.03843804 - Q.dr_2)
    if Q.sum_pt_top5 > 791.125 and Q.eta_0 < 0.04037476:
        z += -0.05877886 * (Q.sum_pt_top5 - 791.125) * (0.04037476 - Q.eta_0)
    if Q.sum_pt < 763.825 and Q.z_7 > 0.01685855:
        z += 0.04513012 * (763.825 - Q.sum_pt) * (Q.z_7 - 0.01685855)
    if Q.sj3_dr_max > 0.169029 and Q.dr_7 < 0.06396627:
        z += 107.2235 * (Q.sj3_dr_max - 0.169029) * (0.06396627 - Q.dr_7)
    if Q.planar_flow < 0.04505724 and Q.tau21_b2 < 0.05932655:
        z += 261.2663 * (0.04505724 - Q.planar_flow) * (0.05932655 - Q.tau21_b2)
    if Q.log_sum_pt > 6.766778 and Q.z_dr_0p1_0p2 < 0.04510668:
        z += 119.3275 * (Q.log_sum_pt - 6.766778) * (0.04510668 - Q.z_dr_0p1_0p2)
    if Q.lam1 < 0.00595415 and Q.n_dr_0p2_0p4 < 1.0:
        z += -237.7917 * (0.00595415 - Q.lam1) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.planar_flow < 0.1484197 and Q.C3 > 0.04857476:
        z += 266.1713 * (0.1484197 - Q.planar_flow) * (Q.C3 - 0.04857476)
    if Q.sum_pt < 763.825 and Q.D2_b2 < 0.380911:
        z += -0.005509276 * (763.825 - Q.sum_pt) * (0.380911 - Q.D2_b2)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.409845
    if Q.lam1 < 0.002464291:
        z += -600.3692 * Q.lam1 + 2.852872
    if 0.002464291 <= Q.lam1 < 0.008375572:
        z += -120.7549 * Q.lam1 + 1.670962
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -181.7924 * Q.lam1 + 2.182186
    if 34.53125 <= Q.pt_7 < 41.65625:
        z += 0.03635443 * Q.pt_7 - 1.255364
    if 41.65625 <= Q.pt_7 < 53.4375:
        z += 0.1146282 * Q.pt_7 - 4.515956
    if Q.pt_7 >= 53.4375:
        z += -0.006683629 * Q.pt_7 + 1.966645
    if 6.377723 <= Q.log_sum_pt < 6.572938:
        z += 6.264421 * Q.log_sum_pt - 39.95274
    if Q.log_sum_pt >= 6.572938:
        z += 13.17707 * Q.log_sum_pt - 85.38915
    if Q.z_7 < 0.04939969:
        z += 106.9437 * Q.z_7 - 7.527682
    if 0.04939969 <= Q.z_7 < 0.06473447:
        z += 76.61849 * Q.z_7 - 6.029627
    if 0.06473447 <= Q.z_7 < 0.08670959:
        z += -2.805891 * Q.z_7 - 0.8881319
    if Q.z_7 >= 0.08670959:
        z += -30.32518 * Q.z_7 + 1.498055
    if Q.tau1 < 0.07283629:
        z += -29.56003 * Q.tau1 + 2.591242
    if 0.07283629 <= Q.tau1 < 0.1027642:
        z += -14.64179 * Q.tau1 + 1.504653
    if Q.sj3_dr_min >= 0.02913153:
        z += -2.395327 * Q.sj3_dr_min + 0.06977955
    if Q.e3 < 1.340118e-05:
        z += 57614.54 * Q.e3 - 0.7721031
    if Q.sj2_dr < 0.1591713:
        z += -5.06938 * Q.sj2_dr + 0.8068999
    if Q.max_dr < 0.1117619:
        z += 11.29117 * Q.max_dr - 1.261922
    if 763.825 <= Q.sum_pt < 988.4078:
        z += 0.002248973 * Q.sum_pt - 1.717822
    if Q.sum_pt >= 988.4078:
        z += -0.006825864 * Q.sum_pt + 7.251818
    if Q.sum_pt_top5 >= 839.9547:
        z += 0.009971807 * Q.sum_pt_top5 - 8.375866
    if Q.zdr_0 < 0.0211821:
        z += 28.35954 * Q.zdr_0 - 0.6007147
    if Q.z_6 < 0.04737278:
        z += 124.2993 * Q.z_6 - 5.888402
    if Q.sum_z_dr < 0.1019409:
        z += 38.74 * Q.sum_z_dr - 3.949192
    if Q.lam2 < 0.001130645:
        z += -574.3148 * Q.lam2 + 0.649346
    if Q.LHA >= 0.2160559:
        z += 1.922909 * Q.LHA - 0.4154557
    if Q.sj3_dr_max < 0.169029:
        z += 0.4785409 * Q.sj3_dr_max - 0.9121309
    if 0.169029 <= Q.sj3_dr_max < 0.3456459:
        z += 4.70648 * Q.sj3_dr_max - 1.626775
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += 11299.41 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.log_sum_pt > 6.377723 and Q.e3 < 1.340118e-05:
        z += 192590.5 * (Q.log_sum_pt - 6.377723) * (1.340118e-05 - Q.e3)
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += -2.864591 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.z_7 < 0.06473447 and Q.e3 < 0.0005116989:
        z += 91331.76 * (0.06473447 - Q.z_7) * (0.0005116989 - Q.e3)
    if Q.pt_7 > 34.53125 and Q.n_dr_0_0p05 < 5.0:
        z += 0.006776573 * (Q.pt_7 - 34.53125) * (5.0 - Q.n_dr_0_0p05)
    if Q.lam1 < 0.008375572 and Q.lam2 < 0.001130645:
        z += -295003.3 * (0.008375572 - Q.lam1) * (0.001130645 - Q.lam2)
    if Q.z_7 < 0.06473447 and Q.D2_b2 < 0.716559:
        z += -32.39339 * (0.06473447 - Q.z_7) * (0.716559 - Q.D2_b2)
    if Q.lam1 < 0.01200373 and Q.n_pt_above_50 > 6.0:
        z += -64.75536 * (0.01200373 - Q.lam1) * (Q.n_pt_above_50 - 6.0)
    if Q.sum_pt > 988.4078 and Q.zdr_7 < 0.01265416:
        z += 0.6062558 * (Q.sum_pt - 988.4078) * (0.01265416 - Q.zdr_7)
    if Q.log_sum_pt > 6.377723 and Q.sum_z_dr2_top2 < 0.004007842:
        z += -350.5563 * (Q.log_sum_pt - 6.377723) * (0.004007842 - Q.sum_z_dr2_top2)
    if Q.log_sum_pt > 6.572938 and Q.sum_z_dr2_top3 < 0.006756161:
        z += -1474.794 * (Q.log_sum_pt - 6.572938) * (0.006756161 - Q.sum_z_dr2_top3)
    if Q.log_sum_pt > 6.572938 and Q.dr_max_012 < 0.1206357:
        z += 42.63387 * (Q.log_sum_pt - 6.572938) * (0.1206357 - Q.dr_max_012)
    if Q.lam1 < 0.01200373 and Q.dr_0 > 0.005517012:
        z += -1963.06 * (0.01200373 - Q.lam1) * (Q.dr_0 - 0.005517012)
    if Q.n_pt_above_50 > 6.0 and Q.e3 < 8.147744e-05:
        z += 6373.845 * (Q.n_pt_above_50 - 6.0) * (8.147744e-05 - Q.e3)
    if Q.sum_pt > 988.4078 and Q.sum_z_dr2_top2 < 0.0095303:
        z += -1.445516 * (Q.sum_pt - 988.4078) * (0.0095303 - Q.sum_z_dr2_top2)
    if Q.lam1 < 0.005433361 and Q.dr_max_012 < 0.1550922:
        z += -913.2755 * (0.005433361 - Q.lam1) * (0.1550922 - Q.dr_max_012)
    if Q.z_7 < 0.06473447 and Q.sum_z_dr2_top3 < 0.007929074:
        z += 3586.868 * (0.06473447 - Q.z_7) * (0.007929074 - Q.sum_z_dr2_top3)
    if Q.z_6 < 0.04737278 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 849.5187 * (0.04737278 - Q.z_6) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.sum_pt_top5 > 839.9547 and Q.n_dr_0p2_0p4 < 2.0:
        z += -0.003481904 * (Q.sum_pt_top5 - 839.9547) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.log_sum_pt > 6.377723 and Q.dr_2 < 0.1147987:
        z += -12.61272 * (Q.log_sum_pt - 6.377723) * (0.1147987 - Q.dr_2)
    if Q.sj2_dr < 0.1591713 and Q.D2_b2 < 0.716559:
        z += 22.11037 * (0.1591713 - Q.sj2_dr) * (0.716559 - Q.D2_b2)
    return max(0.0, z)


def neuron_2(Q):
    z = 6.63557
    if Q.pt_7 < 20.125:
        z += 0.1047614 * Q.pt_7 - 4.741961
    if 20.125 <= Q.pt_7 < 23.21641:
        z += 0.2387093 * Q.pt_7 - 7.437663
    if 23.21641 <= Q.pt_7 < 53.4375:
        z += 0.1598964 * Q.pt_7 - 5.607909
    if Q.pt_7 >= 53.4375:
        z += 0.1053989 * Q.pt_7 - 2.695701
    if Q.log_sum_pt < 6.080494:
        z += -4.352543 * Q.log_sum_pt + 28.8399
    if 6.080494 <= Q.log_sum_pt < 6.605974:
        z += -9.307103 * Q.log_sum_pt + 58.96608
    if 6.605974 <= Q.log_sum_pt < 6.701242:
        z += -5.86899 * Q.log_sum_pt + 36.25399
    if 6.701242 <= Q.log_sum_pt < 6.896095:
        z += -4.95456 * Q.log_sum_pt + 30.12617
    if Q.log_sum_pt >= 6.896095:
        z += -15.13508 * Q.log_sum_pt + 100.332
    if Q.z_7 < 0.02320757:
        z += -45.69477 * Q.z_7 + 2.678339
    if 0.02320757 <= Q.z_7 < 0.02807091:
        z += -157.7235 * Q.z_7 + 5.278255
    if 0.02807091 <= Q.z_7 < 0.0586137:
        z += -118.9776 * Q.z_7 + 4.190621
    if Q.z_7 >= 0.0586137:
        z += -73.28284 * Q.z_7 + 1.512282
    if Q.LHA >= 0.111565:
        z += -11.18065 * Q.LHA + 1.24737
    if Q.e2 < 0.04447357:
        z += 22.95476 * Q.e2 - 1.02088
    if Q.sum_pt < 641.7188:
        z += -0.006205687 * Q.sum_pt + 3.982306
    if Q.sum_pt >= 988.4078:
        z += 0.01883918 * Q.sum_pt - 18.62079
    if Q.C2 < 0.0423228:
        z += 13.50804 * Q.C2 - 0.5716981
    if Q.sum_z_dr < 0.1019409:
        z += -13.41227 * Q.sum_z_dr + 1.367259
    if Q.sum_pt_top5 < 324.5922:
        z += -0.01362504 * Q.sum_pt_top5 + 3.893599
    if 324.5922 <= Q.sum_pt_top5 < 752.1:
        z += 0.001237366 * Q.sum_pt_top5 - 0.9306228
    if Q.mean_phi2 < 8.836697e-05:
        z += -53492.8 * Q.mean_phi2 + 4.726996
    if Q.C2_b2 < 0.004032342:
        z += -109.163 * Q.C2_b2 + 0.4401825
    if Q.lam1 < 0.01200373:
        z += -83.15985 * Q.lam1 + 0.9982281
    if Q.lam1 < 0.00733008 and Q.max_dr < 0.2507612:
        z += 978.9135 * (0.00733008 - Q.lam1) * (0.2507612 - Q.max_dr)
    if Q.lam1 < 0.00733008 and Q.pt_6 < 62.25:
        z += -3.398378 * (0.00733008 - Q.lam1) * (62.25 - Q.pt_6)
    if Q.log_sum_pt < 6.605974 and Q.D2_b2 > 0.01618699:
        z += -1.161958 * (6.605974 - Q.log_sum_pt) * (Q.D2_b2 - 0.01618699)
    if Q.z_7 < 0.0586137 and Q.absphi_0 < 0.07897949:
        z += -229.6909 * (0.0586137 - Q.z_7) * (0.07897949 - Q.absphi_0)
    if Q.log_sum_pt < 6.701242 and Q.mean_phi2 < 8.836697e-05:
        z += -84788.45 * (6.701242 - Q.log_sum_pt) * (8.836697e-05 - Q.mean_phi2)
    if Q.log_sum_pt > 6.080494 and Q.mean_phi2 < 8.836697e-05:
        z += -83953.42 * (Q.log_sum_pt - 6.080494) * (8.836697e-05 - Q.mean_phi2)
    if Q.log_sum_pt > 6.896095 and Q.absphi_0 < 0.1056549:
        z += 571.5647 * (Q.log_sum_pt - 6.896095) * (0.1056549 - Q.absphi_0)
    if Q.sum_pt > 937.0312 and Q.mean_phi2 < 8.836697e-05:
        z += 126.0757 * (Q.sum_pt - 937.0312) * (8.836697e-05 - Q.mean_phi2)
    if Q.z_7 < 0.0586137 and Q.abseta_0 < 0.1057739:
        z += -248.1648 * (0.0586137 - Q.z_7) * (0.1057739 - Q.abseta_0)
    if Q.z_7 > 0.02320757 and Q.sj3_dr_min < 0.1278212:
        z += -80.74989 * (Q.z_7 - 0.02320757) * (0.1278212 - Q.sj3_dr_min)
    if Q.log_sum_pt < 6.701242 and Q.D2_b2 > 0.380911:
        z += 0.8704158 * (6.701242 - Q.log_sum_pt) * (Q.D2_b2 - 0.380911)
    if Q.sum_pt > 988.4078 and Q.absphi_0 < 0.1056549:
        z += -0.4302307 * (Q.sum_pt - 988.4078) * (0.1056549 - Q.absphi_0)
    if Q.centroid_offset < 0.005576073 and Q.dr_7 < 0.03607145:
        z += -4719.495 * (0.005576073 - Q.centroid_offset) * (0.03607145 - Q.dr_7)
    if Q.log_sum_pt > 6.896095 and Q.n_pt_above_50 > 5.0:
        z += -2.928453 * (Q.log_sum_pt - 6.896095) * (Q.n_pt_above_50 - 5.0)
    if Q.sum_pt > 868.5094 and Q.D2_b2 < 4.721224:
        z += 0.001603548 * (Q.sum_pt - 868.5094) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 4.721224:
        z += -3.33892 * (Q.log_sum_pt - 6.896095) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.080494 and Q.lam2 < 0.0001947983:
        z += -2603.63 * (Q.log_sum_pt - 6.080494) * (0.0001947983 - Q.lam2)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.5616404
    if 0.06663269 <= Q.sum_z_dr < 0.1019409:
        z += 40.83496 * Q.sum_z_dr - 2.720943
    if 0.1019409 <= Q.sum_z_dr < 0.1245537:
        z += -90.43137 * Q.sum_z_dr + 10.66047
    if Q.sum_z_dr >= 0.1245537:
        z += 9.094307 * Q.sum_z_dr - 1.735827
    if Q.sj2_dr >= 0.1682655:
        z += 21.24145 * Q.sj2_dr - 3.574204
    if Q.N2 < 0.2654572:
        z += -1.983971 * Q.N2 + 0.5266593
    if Q.sd_rg >= 0.2787955:
        z += -27.70953 * Q.sd_rg + 7.725292
    if Q.max_dr >= 0.2507612:
        z += -9.460021 * Q.max_dr + 2.372206
    if 0.2931906 <= Q.LHA < 0.3255822:
        z += 25.85513 * Q.LHA - 7.580481
    if Q.LHA >= 0.3255822:
        z += -7.030806 * Q.LHA + 3.126594
    if Q.tau1 >= 0.1136369:
        z += 31.19677 * Q.tau1 - 3.545105
    if Q.sj3_dr_max >= 0.213399:
        z += 14.97997 * Q.sj3_dr_max - 3.19671
    if Q.sj3_dr23 >= 0.3661115:
        z += -9.652806 * Q.sj3_dr23 + 3.534003
    if Q.mean_eta < -0.006779839:
        z += -35.06504 * Q.mean_eta - 0.2377353
    if Q.lam1 >= 0.01200373:
        z += -220.0257 * Q.lam1 + 2.641128
    if Q.lam2 < 0.003408389:
        z += 1011.438 * Q.lam2 - 3.447373
    if Q.sj2_dr > 0.1682655 and Q.log_sum_pt > 6.080494:
        z += -22.23134 * (Q.sj2_dr - 0.1682655) * (Q.log_sum_pt - 6.080494)
    if Q.sj2_dr > 0.1682655 and Q.C2 < 0.09482124:
        z += 196.9533 * (Q.sj2_dr - 0.1682655) * (0.09482124 - Q.C2)
    if Q.sum_z_dr > 0.06663269 and Q.n_pt_above_50 > 3.0:
        z += -2.800026 * (Q.sum_z_dr - 0.06663269) * (Q.n_pt_above_50 - 3.0)
    if Q.sd_rg > 0.1876504 and Q.lam2 < 0.000537286:
        z += 14508.58 * (Q.sd_rg - 0.1876504) * (0.000537286 - Q.lam2)
    if Q.sj2_dr > 0.1682655 and Q.n_dr_0_0p05 > 2.0:
        z += -4.822762 * (Q.sj2_dr - 0.1682655) * (Q.n_dr_0_0p05 - 2.0)
    if Q.max_dr > 0.2507612 and Q.z_dr_0_0p05 < 0.6080732:
        z += -18.29825 * (Q.max_dr - 0.2507612) * (0.6080732 - Q.z_dr_0_0p05)
    if Q.sd_rg > 0.1876504 and Q.z_dr_0_0p05 < 0.1515405:
        z += 68.10986 * (Q.sd_rg - 0.1876504) * (0.1515405 - Q.z_dr_0_0p05)
    if Q.sj3_dr_max > 0.213399 and Q.n_dr_0p05_0p1 > 4.0:
        z += 10.73367 * (Q.sj3_dr_max - 0.213399) * (Q.n_dr_0p05_0p1 - 4.0)
    if Q.sj3_dr_max > 0.2623172 and Q.n_dr_0p05_0p1 > 4.0:
        z += -20.53426 * (Q.sj3_dr_max - 0.2623172) * (Q.n_dr_0p05_0p1 - 4.0)
    if Q.mean_eta < -0.006779839 and Q.eta_0 < -0.03967285:
        z += -388.178 * (-0.006779839 - Q.mean_eta) * (-0.03967285 - Q.eta_0)
    if Q.sum_z_dr > 0.06663269 and Q.eccentricity > 0.7117266:
        z += 73.42568 * (Q.sum_z_dr - 0.06663269) * (Q.eccentricity - 0.7117266)
    if Q.sj2_dr > 0.1682655 and Q.eccentricity > 0.9458207:
        z += 363.8594 * (Q.sj2_dr - 0.1682655) * (Q.eccentricity - 0.9458207)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.6591341
    if Q.N2 < 0.2233283:
        z += -28.91009 * Q.N2 + 6.45644
    if Q.lam1 < 0.006506576:
        z += 1922.463 * Q.lam1 - 13.59
    if 0.006506576 <= Q.lam1 < 0.008375572:
        z += 578.5721 * Q.lam1 - 4.845872
    if Q.lam1 >= 0.01200373:
        z += -27.47238 * Q.lam1 + 0.3297709
    if Q.e3 < 0.0001869378:
        z += -26231.51 * Q.e3 + 4.903661
    if Q.sj3_dr_max < 0.1426152:
        z += 25.0123 * Q.sj3_dr_max - 3.567134
    if 0.169029 <= Q.sj3_dr_max < 0.233678:
        z += 26.35912 * Q.sj3_dr_max - 4.455456
    if Q.sj3_dr_max >= 0.233678:
        z += 8.174885 * Q.sj3_dr_max - 0.2062012
    if Q.lam2 < 0.000537286:
        z += 3994.117 * Q.lam2 - 2.621958
    if 0.000537286 <= Q.lam2 < 0.001130645:
        z += 802.1705 * Q.lam2 - 0.9069698
    if Q.tau21_b2 < 0.003206041:
        z += 586.5712 * Q.tau21_b2 - 1.880571
    if Q.sum_pt < 763.825:
        z += 0.005047049 * Q.sum_pt - 3.855062
    if Q.max_dr >= 0.121681:
        z += 6.147015 * Q.max_dr - 0.7479747
    if Q.log_sum_pt >= 6.327379:
        z += 5.141326 * Q.log_sum_pt - 32.53112
    if Q.pt_7 < 25.57812:
        z += 0.2045101 * Q.pt_7 - 5.230985
    if 0.2001708 <= Q.sj2_dr < 0.2179769:
        z += -19.43291 * Q.sj2_dr + 3.889901
    if Q.sj2_dr >= 0.2179769:
        z += -15.49797 * Q.sj2_dr + 3.032175
    if Q.C2 >= 0.01867771:
        z += -6.642227 * Q.C2 + 0.1240616
    if Q.LHA < 0.1329373:
        z += 4.066283 * Q.LHA + 1.646996
    if 0.1329373 <= Q.LHA < 0.3127275:
        z += -12.16728 * Q.LHA + 3.805041
    if Q.sj3_dr_min < 0.2089872:
        z += -29.06677 * Q.sj3_dr_min + 6.074584
    if Q.eccentricity >= 0.7117266:
        z += -7.671405 * Q.eccentricity + 5.459943
    if Q.zdr_0 < 0.01705377:
        z += 63.33855 * Q.zdr_0 - 1.080161
    if Q.M2 < 0.04435703:
        z += 69.23695 * Q.M2 - 3.071146
    if Q.N2 < 0.2233283 and Q.sj2_zsoft < 0.2832687:
        z += -98.35616 * (0.2233283 - Q.N2) * (0.2832687 - Q.sj2_zsoft)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1626038:
        z += -39.80976 * (0.2233283 - Q.N2) * (0.1626038 - Q.abseta_7)
    if Q.sj3_dr_max > 0.233678 and Q.log_sum_pt > 6.267538:
        z += -82.86937 * (Q.sj3_dr_max - 0.233678) * (Q.log_sum_pt - 6.267538)
    if Q.sum_pt < 763.825 and Q.M3 < 0.107953:
        z += -0.06290323 * (763.825 - Q.sum_pt) * (0.107953 - Q.M3)
    if Q.sj3_dr_max > 0.233678 and Q.sj2_zsoft < 0.05966518:
        z += 761.7151 * (Q.sj3_dr_max - 0.233678) * (0.05966518 - Q.sj2_zsoft)
    if Q.dr_0 < 0.08082334 and Q.centroid_offset > 0.02355416:
        z += -1463.33 * (0.08082334 - Q.dr_0) * (Q.centroid_offset - 0.02355416)
    if Q.log_sum_pt > 6.327379 and Q.ptdr0_6 < 7.998907:
        z += -0.1712521 * (Q.log_sum_pt - 6.327379) * (7.998907 - Q.ptdr0_6)
    if Q.max_dr > 0.121681 and Q.C2_b2 < 0.009032972:
        z += 2302.259 * (Q.max_dr - 0.121681) * (0.009032972 - Q.C2_b2)
    if Q.N2 < 0.2233283 and Q.lam2 > 0.001130645:
        z += -9355.604 * (0.2233283 - Q.N2) * (Q.lam2 - 0.001130645)
    if Q.lam1 < 0.00733008 and Q.lam2 > 0.000537286:
        z += 846240.9 * (0.00733008 - Q.lam1) * (Q.lam2 - 0.000537286)
    if Q.dr_0 < 0.08082334 and Q.centroid_offset > 0.009480685:
        z += 2409.714 * (0.08082334 - Q.dr_0) * (Q.centroid_offset - 0.009480685)
    if Q.N2 < 0.2233283 and Q.n_dr_0p1_0p2 > 2.0:
        z += 1.768208 * (0.2233283 - Q.N2) * (Q.n_dr_0p1_0p2 - 2.0)
    if Q.log_sum_pt > 6.327379 and Q.zdr_2 < 0.009799324:
        z += -241.9404 * (Q.log_sum_pt - 6.327379) * (0.009799324 - Q.zdr_2)
    if Q.sj3_dr_min < 0.2089872 and Q.lam2 < 0.003408389:
        z += -6839.389 * (0.2089872 - Q.sj3_dr_min) * (0.003408389 - Q.lam2)
    if Q.sj3_dr_min < 0.2089872 and Q.mean_phi < -0.02594505:
        z += 240.5886 * (0.2089872 - Q.sj3_dr_min) * (-0.02594505 - Q.mean_phi)
    if Q.sj2_dr > 0.2179769 and Q.lam2 > 0.001130645:
        z += -3545.377 * (Q.sj2_dr - 0.2179769) * (Q.lam2 - 0.001130645)
    if Q.pt_7 < 25.57812 and Q.eccentricity > 0.9458207:
        z += 2.876151 * (25.57812 - Q.pt_7) * (Q.eccentricity - 0.9458207)
    if Q.N2 < 0.2233283 and Q.pt_7 < 53.4375:
        z += -0.145496 * (0.2233283 - Q.N2) * (53.4375 - Q.pt_7)
    if Q.lam1 < 0.006506576 and Q.eccentricity > 0.7117266:
        z += 3651.263 * (0.006506576 - Q.lam1) * (Q.eccentricity - 0.7117266)
    if Q.max_dr > 0.121681 and Q.n_pt_above_50 > 6.0:
        z += -7.068597 * (Q.max_dr - 0.121681) * (Q.n_pt_above_50 - 6.0)
    if Q.eccentricity > 0.7117266 and Q.n_pt_above_50 > 6.0:
        z += 1.290606 * (Q.eccentricity - 0.7117266) * (Q.n_pt_above_50 - 6.0)
    if Q.sj3_dr_max > 0.3456459 and Q.sj3_z1 > 0.7947645:
        z += -212.5456 * (Q.sj3_dr_max - 0.3456459) * (Q.sj3_z1 - 0.7947645)
    if Q.log_sum_pt > 6.327379 and Q.dr1_4 > 0.1883455:
        z += -16.26019 * (Q.log_sum_pt - 6.327379) * (Q.dr1_4 - 0.1883455)
    if Q.max_dr > 0.121681 and Q.abseta_0 < 0.07861328:
        z += -70.8035 * (Q.max_dr - 0.121681) * (0.07861328 - Q.abseta_0)
    if Q.lam2 < 0.001130645 and Q.absphi_7 > 0.001937771:
        z += 3227.4 * (0.001130645 - Q.lam2) * (Q.absphi_7 - 0.001937771)
    if Q.sj2_dr > 0.2179769 and Q.D2_b2 < 0.5327104:
        z += -27.91448 * (Q.sj2_dr - 0.2179769) * (0.5327104 - Q.D2_b2)
    if Q.lam2 < 0.000537286 and Q.dr0_7 > 0.09118326:
        z += -3354.748 * (0.000537286 - Q.lam2) * (Q.dr0_7 - 0.09118326)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.454121
    if Q.LHA < 0.2160559:
        z += -27.78753 * Q.LHA + 5.901047
    if 0.2160559 <= Q.LHA < 0.266913:
        z += 2.017662 * Q.LHA - 0.53854
    if Q.e2 < 0.03556091:
        z += 45.18177 * Q.e2 - 1.606705
    if 6.804164 <= Q.log_sum_pt < 6.896095:
        z += -10.5617 * Q.log_sum_pt + 71.86352
    if Q.log_sum_pt >= 6.896095:
        z += -21.14787 * Q.log_sum_pt + 144.8668
    if Q.z_6 < 0.02160287:
        z += -114.2896 * Q.z_6 + 1.173407
    if 0.02160287 <= Q.z_6 < 0.02886576:
        z += -17.6379 * Q.z_6 - 0.9145464
    if 0.02886576 <= Q.z_6 < 0.06081235:
        z += 44.56431 * Q.z_6 - 2.71006
    if Q.lam1 < 7.12483e-05:
        z += -36322.9 * Q.lam1 + 6.867788
    if 7.12483e-05 <= Q.lam1 < 0.0008722282:
        z += -3239.715 * Q.lam1 + 4.510668
    if 0.0008722282 <= Q.lam1 < 0.002464291:
        z += -877.4087 * Q.lam1 + 2.450197
    if 0.002464291 <= Q.lam1 < 0.00483998:
        z += -121.2309 * Q.lam1 + 0.586755
    if Q.tau1 < 0.0262518:
        z += 37.87709 * Q.tau1 - 0.9943419
    if Q.sum_z_dr < 0.02054282:
        z += 63.6225 * Q.sum_z_dr - 1.306985
    if Q.sum_pt < 739.5:
        z += 0.01533331 * Q.sum_pt - 11.33898
    if Q.n_pt_above_50 >= 6.0:
        z += -0.287566 * Q.n_pt_above_50 + 1.725396
    if Q.z_7 < 0.02807091:
        z += -148.0535 * Q.z_7 + 4.444664
    if 0.02807091 <= Q.z_7 < 0.03629544:
        z += -35.09818 * Q.z_7 + 1.273904
    if Q.z_top5_slots >= 0.908903:
        z += -23.77033 * Q.z_top5_slots + 21.60493
    if Q.dr_0 < 0.06413297:
        z += 28.04747 * Q.dr_0 - 1.798768
    if Q.pt_7 < 20.125:
        z += 0.1011107 * Q.pt_7 - 2.034853
    if Q.pt_7 >= 37.15625:
        z += -0.05426237 * Q.pt_7 + 2.016186
    if Q.sj3_dr_max < 0.1070199:
        z += 9.076913 * Q.sj3_dr_max - 1.45355
    if 0.1070199 <= Q.sj3_dr_max < 0.3012016:
        z += 2.48293 * Q.sj3_dr_max - 0.7478624
    if Q.centroid_offset < 0.03776099:
        z += 34.10953 * Q.centroid_offset - 1.28801
    if Q.e3 < 0.0001869378:
        z += -4220.081 * Q.e3 + 0.7888928
    if Q.sum_pt_top5 < 579.875:
        z += -0.01677413 * Q.sum_pt_top5 + 9.726899
    if Q.z_top5 < 0.8233866:
        z += 11.08291 * Q.z_top5 - 9.125522
    if Q.pt_5 < 24.57812:
        z += -0.12527 * Q.pt_5 + 3.078901
    if Q.LHA < 0.2160559 and Q.z_6 < 0.09543973:
        z += -361.8395 * (0.2160559 - Q.LHA) * (0.09543973 - Q.z_6)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -115.5149 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.e2 < 0.03556091 and Q.z_7 < 0.0753896:
        z += 744.1642 * (0.03556091 - Q.e2) * (0.0753896 - Q.z_7)
    if Q.e2 < 0.03556091 and Q.zdr_0 > 0.004918231:
        z += 1894.037 * (0.03556091 - Q.e2) * (Q.zdr_0 - 0.004918231)
    if Q.e2 < 0.03556091 and Q.planar_flow < 0.3220738:
        z += 60.24976 * (0.03556091 - Q.e2) * (0.3220738 - Q.planar_flow)
    if Q.lam1 < 0.002464291 and Q.centroid_offset < 0.01837778:
        z += 59527.83 * (0.002464291 - Q.lam1) * (0.01837778 - Q.centroid_offset)
    if Q.sum_z_dr < 0.02054282 and Q.sum_pt_top5 > 716.8828:
        z += 0.9274743 * (0.02054282 - Q.sum_z_dr) * (Q.sum_pt_top5 - 716.8828)
    if Q.z_6 < 0.02886576 and Q.phi_0 > 0.008850098:
        z += -5919.208 * (0.02886576 - Q.z_6) * (Q.phi_0 - 0.008850098)
    if Q.sum_z_dr < 0.02054282 and Q.M3 < 0.09607851:
        z += -1068.887 * (0.02054282 - Q.sum_z_dr) * (0.09607851 - Q.M3)
    if Q.z_6 < 0.06081235 and Q.pt1_dr01 < 28.39396:
        z += 0.9261437 * (0.06081235 - Q.z_6) * (28.39396 - Q.pt1_dr01)
    if Q.log_sum_pt > 6.804164 and Q.e3 < 5.334511e-05:
        z += -199800.3 * (Q.log_sum_pt - 6.804164) * (5.334511e-05 - Q.e3)
    if Q.e2 < 0.03556091 and Q.lam2 < 0.0003061234:
        z += 191621.2 * (0.03556091 - Q.e2) * (0.0003061234 - Q.lam2)
    if Q.z_6 < 0.06081235 and Q.e3 < 0.0005116989:
        z += 97548.46 * (0.06081235 - Q.z_6) * (0.0005116989 - Q.e3)
    if Q.lam1 < 7.12483e-05 and Q.pt_5 < 73.75:
        z += -575.1004 * (7.12483e-05 - Q.lam1) * (73.75 - Q.pt_5)
    if Q.log_sum_pt > 6.804164 and Q.mean_phi < 0.01251955:
        z += 289.97 * (Q.log_sum_pt - 6.804164) * (0.01251955 - Q.mean_phi)
    if Q.log_sum_pt > 6.896095 and Q.mean_phi > -0.01275329:
        z += 338.4392 * (Q.log_sum_pt - 6.896095) * (Q.mean_phi - -0.01275329)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.333628
    if Q.centroid_offset < 0.00809236:
        z += 107.6908 * Q.centroid_offset - 1.979117
    if 0.00809236 <= Q.centroid_offset < 0.01837778:
        z += 113.2852 * Q.centroid_offset - 2.024389
    if 0.01837778 <= Q.centroid_offset < 0.02076709:
        z += 5.594389 * Q.centroid_offset - 0.04527181
    if 0.02076709 <= Q.centroid_offset < 0.02685622:
        z += 26.5599 * Q.centroid_offset - 0.4806645
    if Q.centroid_offset >= 0.02685622:
        z += -11.85474 * Q.centroid_offset + 0.5510076
    if Q.log_sum_pt < 6.423044:
        z += -1.940836 * Q.log_sum_pt + 12.46607
    if 0.1591713 <= Q.sj2_dr < 0.1682655:
        z += -15.30104 * Q.sj2_dr + 2.435487
    if Q.sj2_dr >= 0.1682655:
        z += 1.231492 * Q.sj2_dr - 0.3463692
    if 0.08723651 <= Q.sum_z_dr < 0.1019409:
        z += -237.3641 * Q.sum_z_dr + 20.70682
    if 0.1019409 <= Q.sum_z_dr < 0.1484084:
        z += -279.5596 * Q.sum_z_dr + 25.00827
    if Q.sum_z_dr >= 0.1484084:
        z += -113.1428 * Q.sum_z_dr + 0.3106029
    if Q.pt_6 < 19.46875:
        z += -0.6652826 * Q.pt_6 + 11.46692
    if 19.46875 <= Q.pt_6 < 24.42188:
        z += -0.0145075 * Q.pt_6 - 1.202861
    if 24.42188 <= Q.pt_6 < 31.90625:
        z += 0.0643811 * Q.pt_6 - 3.129469
    if 31.90625 <= Q.pt_6 < 39.75:
        z += 0.1370912 * Q.pt_6 - 5.449375
    if Q.lam2 < 5.794419e-05:
        z += 20171.4 * Q.lam2 - 2.488769
    if 5.794419e-05 <= Q.lam2 < 0.001130645:
        z += 1944.885 * Q.lam2 - 1.432649
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -336.4405 * Q.lam2 + 1.14672
    if 0.251526 <= Q.LHA < 0.3255822:
        z += 38.59822 * Q.LHA - 9.708457
    if Q.LHA >= 0.3255822:
        z += 85.13391 * Q.LHA - 24.85965
    if Q.psi_0p2 >= 0.8990266:
        z += -7.026195 * Q.psi_0p2 + 6.316737
    if Q.C2 < 0.03578649:
        z += -98.82111 * Q.C2 + 3.53646
    if 0.03464708 <= Q.sj3_dr_max < 0.1789613:
        z += -12.96632 * Q.sj3_dr_max + 0.4492453
    if Q.sj3_dr_max >= 0.1789613:
        z += 10.83241 * Q.sj3_dr_max - 3.809809
    if Q.z_6 < 0.02160287:
        z += 545.9617 * Q.z_6 - 9.665113
    if 0.02160287 <= Q.z_6 < 0.04355037:
        z += -97.01466 * Q.z_6 + 4.225025
    if Q.e3 < 7.330536e-06:
        z += -67385.91 * Q.e3 + 0.4939748
    if Q.max_dr < 0.177305:
        z += 30.08162 * Q.max_dr - 5.333621
    if Q.dr_0 < 0.08082334:
        z += -6.377899 * Q.dr_0 + 0.5154831
    if 0.05356915 <= Q.tau1 < 0.1748444:
        z += -7.234602 * Q.tau1 + 0.3875515
    if Q.tau1 >= 0.1748444:
        z += -96.03133 * Q.tau1 + 15.91316
    if Q.pt_7 < 33.21875:
        z += -0.09748334 * Q.pt_7 + 3.238275
    if Q.z_7 < 0.03978449:
        z += 86.44211 * Q.z_7 - 3.439055
    if Q.sum_pt >= 840.0195:
        z += 0.01078466 * Q.sum_pt - 9.059322
    if Q.sj2_dr > 0.1591713 and Q.C2 < 0.09482124:
        z += 99.0826 * (Q.sj2_dr - 0.1591713) * (0.09482124 - Q.C2)
    if Q.log_sum_pt < 6.423044 and Q.pt_7 < 38.53125:
        z += 0.1336713 * (6.423044 - Q.log_sum_pt) * (38.53125 - Q.pt_7)
    if Q.centroid_offset > 0.00809236 and Q.N2 > 0.198361:
        z += 174.4364 * (Q.centroid_offset - 0.00809236) * (Q.N2 - 0.198361)
    if Q.sj2_dr > 0.1591713 and Q.sj2_zsoft > 0.05966518:
        z += 120.0782 * (Q.sj2_dr - 0.1591713) * (Q.sj2_zsoft - 0.05966518)
    if Q.centroid_offset < 0.01837778 and Q.planar_flow < 0.1115136:
        z += -657.9141 * (0.01837778 - Q.centroid_offset) * (0.1115136 - Q.planar_flow)
    if Q.pt_6 < 39.75 and Q.sum_pt_top3 < 711.875:
        z += 0.0008475144 * (39.75 - Q.pt_6) * (711.875 - Q.sum_pt_top3)
    if Q.centroid_offset > 0.00809236 and Q.sj3_dr_min < 0.2089872:
        z += 379.3298 * (Q.centroid_offset - 0.00809236) * (0.2089872 - Q.sj3_dr_min)
    if Q.centroid_offset > 0.00809236 and Q.mean_phi2 < 0.01426135:
        z += 2994.284 * (Q.centroid_offset - 0.00809236) * (0.01426135 - Q.mean_phi2)
    if Q.sj2_dr > 0.1591713 and Q.mean_eta > 0.02644207:
        z += -420.8193 * (Q.sj2_dr - 0.1591713) * (Q.mean_eta - 0.02644207)
    if Q.centroid_offset > 0.02076709 and Q.n_dr_0p1_0p2 > 0.0:
        z += 5.129985 * (Q.centroid_offset - 0.02076709) * (Q.n_dr_0p1_0p2 - 0.0)
    if Q.centroid_offset > 0.02076709 and Q.n_dr_0_0p05 < 5.0:
        z += -4.963295 * (Q.centroid_offset - 0.02076709) * (5.0 - Q.n_dr_0_0p05)
    if Q.sj3_dr_max > 0.1789613 and Q.D2_b2 < 1.59265:
        z += 10.06574 * (Q.sj3_dr_max - 0.1789613) * (1.59265 - Q.D2_b2)
    if Q.lam2 < 0.001130645 and Q.C3 > 0.04165477:
        z += 14913.01 * (0.001130645 - Q.lam2) * (Q.C3 - 0.04165477)
    if Q.pt_6 < 39.75 and Q.D2_b2 < 4.721224:
        z += -0.01667082 * (39.75 - Q.pt_6) * (4.721224 - Q.D2_b2)
    if Q.pt_6 < 39.75 and Q.z_top3_slots < 0.7861046:
        z += -0.3254647 * (39.75 - Q.pt_6) * (0.7861046 - Q.z_top3_slots)
    if Q.lam2 < 0.003408389 and Q.n_dr_0p1_0p2 > 1.0:
        z += 62.75803 * (0.003408389 - Q.lam2) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.centroid_offset < 0.01837778 and Q.n_dr_0p05_0p1 > 0.0:
        z += 23.15506 * (0.01837778 - Q.centroid_offset) * (Q.n_dr_0p05_0p1 - 0.0)
    if Q.sum_z_dr > 0.08723651 and Q.D2_b2 < 0.380911:
        z += -75.17817 * (Q.sum_z_dr - 0.08723651) * (0.380911 - Q.D2_b2)
    if Q.centroid_offset > 0.00809236 and Q.D3 > 0.8272948:
        z += -8.049837 * (Q.centroid_offset - 0.00809236) * (Q.D3 - 0.8272948)
    if Q.sj3_dr_max > 0.03464708 and Q.dr_min_012 < 0.01488897:
        z += -187.7826 * (Q.sj3_dr_max - 0.03464708) * (0.01488897 - Q.dr_min_012)
    if Q.centroid_offset > 0.02685622 and Q.pt_2 > 69.875:
        z += 1.713746 * (Q.centroid_offset - 0.02685622) * (Q.pt_2 - 69.875)
    if Q.centroid_offset > 0.02685622 and Q.D2_b2 < 0.380911:
        z += -73.54165 * (Q.centroid_offset - 0.02685622) * (0.380911 - Q.D2_b2)
    if Q.sum_pt > 840.0195 and Q.D2_b2 > 0.716559:
        z += -0.0007323816 * (Q.sum_pt - 840.0195) * (Q.D2_b2 - 0.716559)
    if Q.centroid_offset > 0.02685622 and Q.zdr_2 < 0.01498227:
        z += 3769.81 * (Q.centroid_offset - 0.02685622) * (0.01498227 - Q.zdr_2)
    return max(0.0, z)


def neuron_7(Q):
    z = 2.215177
    if Q.planar_flow < 0.1950135:
        z += -2.435697 * Q.planar_flow + 0.4749939
    if Q.tau1 < 0.06345984:
        z += -48.4791 * Q.tau1 + 4.297851
    if 0.06345984 <= Q.tau1 < 0.08865369:
        z += -30.62218 * Q.tau1 + 3.164654
    if 0.08865369 <= Q.tau1 < 0.1027642:
        z += 17.85691 * Q.tau1 - 1.133197
    if Q.tau1 >= 0.1027642:
        z += 121.6164 * Q.tau1 - 11.79596
    if Q.lam1 < 0.002464291:
        z += 1105.934 * Q.lam1 - 3.031882
    if 0.002464291 <= Q.lam1 < 0.003377388:
        z += 1478.217 * Q.lam1 - 3.949297
    if 0.003377388 <= Q.lam1 < 0.00483998:
        z += 849.2524 * Q.lam1 - 1.825039
    if 0.00483998 <= Q.lam1 < 0.00733008:
        z += -23.94379 * Q.lam1 + 2.401213
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += -457.0118 * Q.lam1 + 5.575637
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -60.78491 * Q.lam1 + 2.257009
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 437.6931 * Q.lam1 - 3.726584
    if Q.lam1 >= 0.01643375:
        z += -3119.417 * Q.lam1 + 54.73008
    if Q.sum_z_dr < 0.03360421:
        z += 64.8315 * Q.sum_z_dr - 2.581049
    if 0.03360421 <= Q.sum_z_dr < 0.0479157:
        z += 14.64377 * Q.sum_z_dr - 0.8945294
    if 0.0479157 <= Q.sum_z_dr < 0.06108601:
        z += -38.41593 * Q.sum_z_dr + 1.647863
    if 0.06108601 <= Q.sum_z_dr < 0.08723651:
        z += -53.05969 * Q.sum_z_dr + 2.542392
    if 0.08723651 <= Q.sum_z_dr < 0.1019409:
        z += -304.5286 * Q.sum_z_dr + 24.47966
    if Q.sum_z_dr >= 0.1019409:
        z += 94.76483 * Q.sum_z_dr - 16.22468
    if Q.lam2 < 0.0003061234:
        z += 1550.629 * Q.lam2 - 1.017382
    if 0.0003061234 <= Q.lam2 < 0.003408389:
        z += 174.936 * Q.lam2 - 0.5962498
    if Q.LHA < 0.1329373:
        z += 12.38166 * Q.LHA - 5.24478
    if 0.1329373 <= Q.LHA < 0.3467135:
        z += 16.63017 * Q.LHA - 5.809564
    if 0.3467135 <= Q.LHA < 0.4235925:
        z += -246.5101 * Q.LHA + 85.42473
    if Q.LHA >= 0.4235925:
        z += -258.8918 * Q.LHA + 90.66951
    if Q.centroid_offset < 0.02685622:
        z += -27.40015 * Q.centroid_offset + 0.8639319
    if 0.02685622 <= Q.centroid_offset < 0.04990367:
        z += -5.556675 * Q.centroid_offset + 0.2772985
    if Q.N2 >= 0.1333619:
        z += 4.266865 * Q.N2 - 0.5690371
    if Q.sj3_dr_max < 0.07264452:
        z += -25.86168 * Q.sj3_dr_max + 3.738211
    if 0.07264452 <= Q.sj3_dr_max < 0.1594012:
        z += -21.68772 * Q.sj3_dr_max + 3.434995
    if 0.1594012 <= Q.sj3_dr_max < 0.213399:
        z += 11.28853 * Q.sj3_dr_max - 1.821457
    if Q.sj3_dr_max >= 0.213399:
        z += 4.173967 * Q.sj3_dr_max - 0.3032158
    if Q.max_dr < 0.04656688:
        z += 33.15971 * Q.max_dr - 1.933861
    if 0.04656688 <= Q.max_dr < 0.08050702:
        z += 2.178793 * Q.max_dr - 0.4911765
    if 0.08050702 <= Q.max_dr < 0.0931108:
        z += 25.05347 * Q.max_dr - 2.332749
    if Q.e3 < 1.050302e-05:
        z += -75911.19 * Q.e3 - 0.04021862
    if 1.050302e-05 <= Q.e3 < 8.147744e-05:
        z += 11800.24 * Q.e3 - 0.9614536
    if Q.D2 < 1.432482:
        z += 0.6964248 * Q.D2 - 0.9976162
    if Q.n_dr_0_0p05 < 4.0:
        z += -0.07103679 * Q.n_dr_0_0p05 + 0.2841471
    if Q.planar_flow < 0.1950135 and Q.lam1 > 0.008375572:
        z += -6292.838 * (0.1950135 - Q.planar_flow) * (Q.lam1 - 0.008375572)
    if Q.planar_flow < 0.1950135 and Q.pt_6 < 56.53125:
        z += -0.1177043 * (0.1950135 - Q.planar_flow) * (56.53125 - Q.pt_6)
    if Q.planar_flow < 0.1950135 and Q.log_sum_pt > 6.46415:
        z += 28.9584 * (0.1950135 - Q.planar_flow) * (Q.log_sum_pt - 6.46415)
    if Q.planar_flow < 0.1950135 and Q.pt_7 < 29.04219:
        z += -0.3762147 * (0.1950135 - Q.planar_flow) * (29.04219 - Q.pt_7)
    if Q.lam1 > 0.002464291 and Q.sj3_dr_max < 0.1879486:
        z += -11421.21 * (Q.lam1 - 0.002464291) * (0.1879486 - Q.sj3_dr_max)
    if Q.planar_flow < 0.1950135 and Q.max_dr < 0.2507612:
        z += -44.42754 * (0.1950135 - Q.planar_flow) * (0.2507612 - Q.max_dr)
    if Q.sum_z_dr > 0.08065885 and Q.sj2_zsoft < 0.09384951:
        z += 5540.779 * (Q.sum_z_dr - 0.08065885) * (0.09384951 - Q.sj2_zsoft)
    if Q.lam1 > 0.00733008 and Q.D2 < 2.843757:
        z += -95.59605 * (Q.lam1 - 0.00733008) * (2.843757 - Q.D2)
    if Q.lam1 > 0.002464291 and Q.planar_flow < 0.1950135:
        z += 2459.285 * (Q.lam1 - 0.002464291) * (0.1950135 - Q.planar_flow)
    if Q.lam1 > 0.002464291 and Q.D2 > 0.2568137:
        z += -301.1805 * (Q.lam1 - 0.002464291) * (Q.D2 - 0.2568137)
    if Q.lam1 > 0.01643375 and Q.D2 < 3.885568:
        z += 580.6964 * (Q.lam1 - 0.01643375) * (3.885568 - Q.D2)
    if Q.sum_z_dr > 0.0479157 and Q.D2 < 2.357246:
        z += 51.41581 * (Q.sum_z_dr - 0.0479157) * (2.357246 - Q.D2)
    if Q.e3 < 8.147744e-05 and Q.pt_7 < 43.5:
        z += -332.8272 * (8.147744e-05 - Q.e3) * (43.5 - Q.pt_7)
    if Q.sum_z_dr > 0.08065885 and Q.D2 < 2.843757:
        z += -33.2448 * (Q.sum_z_dr - 0.08065885) * (2.843757 - Q.D2)
    if Q.sj3_dr_max < 0.1426152 and Q.D2 < 1.432482:
        z += 15.2584 * (0.1426152 - Q.sj3_dr_max) * (1.432482 - Q.D2)
    if Q.centroid_offset < 0.02685622 and Q.n_dr_0p2_0p4 < 1.0:
        z += -47.77116 * (0.02685622 - Q.centroid_offset) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.centroid_offset < 0.04990367 and Q.n_dr_0p2_0p4 < 2.0:
        z += 5.822149 * (0.04990367 - Q.centroid_offset) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.sum_z_dr > 0.08065885 and Q.D2_b2 > 1.911889:
        z += 60.22951 * (Q.sum_z_dr - 0.08065885) * (Q.D2_b2 - 1.911889)
    if Q.centroid_offset < 0.04990367 and Q.tau4 > 0.002042374:
        z += -1567.593 * (0.04990367 - Q.centroid_offset) * (Q.tau4 - 0.002042374)
    if Q.lam1 > 0.00733008 and Q.pt_4 < 90.625:
        z += 5.188363 * (Q.lam1 - 0.00733008) * (90.625 - Q.pt_4)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.7918924
    if Q.lam1 < 0.0002758826:
        z += -3040.803 * Q.lam1 + 1.828389
    if 0.0002758826 <= Q.lam1 < 0.003377388:
        z += -290.9678 * Q.lam1 + 1.069757
    if 0.003377388 <= Q.lam1 < 0.00733008:
        z += -22.0219 * Q.lam1 + 0.1614223
    if Q.LHA < 0.1767241:
        z += -30.31542 * Q.LHA + 5.150195
    if 0.1767241 <= Q.LHA < 0.251526:
        z += 2.770927 * Q.LHA - 0.6969602
    if Q.log_sum_pt >= 6.701242:
        z += -5.474082 * Q.log_sum_pt + 36.68315
    if Q.sj3_dr_max < 0.1070199:
        z += 62.97473 * Q.sj3_dr_max - 4.474008
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += -38.65482 * Q.sj3_dr_max + 6.40238
    if 0.1426152 <= Q.sj3_dr_max < 0.1594012:
        z += -18.46031 * Q.sj3_dr_max + 3.522336
    if 0.1594012 <= Q.sj3_dr_max < 0.1986272:
        z += -14.77951 * Q.sj3_dr_max + 2.935613
    if Q.sum_z_dr < 0.02054282:
        z += 105.2177 * Q.sum_z_dr - 2.926506
    if 0.02054282 <= Q.sum_z_dr < 0.03360421:
        z += 58.57242 * Q.sum_z_dr - 1.96828
    if Q.tau1 < 0.0262518:
        z += 87.04675 * Q.tau1 - 2.285134
    if Q.z_6 < 0.04355037:
        z += -44.13896 * Q.z_6 + 1.922268
    if Q.centroid_offset >= 0.006789738:
        z += -63.95623 * Q.centroid_offset + 0.4342461
    if Q.lam2 < 0.0001947983:
        z += -9616.1 * Q.lam2 + 1.8732
    if Q.tau2 < 0.01713288:
        z += 90.07706 * Q.tau2 - 1.543279
    if Q.lam1 < 0.00733008 and Q.centroid_offset < 0.02355416:
        z += 6753.53 * (0.00733008 - Q.lam1) * (0.02355416 - Q.centroid_offset)
    if Q.lam1 < 0.00733008 and Q.planar_flow < 0.5925897:
        z += 117.0437 * (0.00733008 - Q.lam1) * (0.5925897 - Q.planar_flow)
    if Q.lam1 < 0.005433361 and Q.log_sum_pt < 6.502799:
        z += -637.7435 * (0.005433361 - Q.lam1) * (6.502799 - Q.log_sum_pt)
    if Q.LHA < 0.251526 and Q.z_7 > 0.01685855:
        z += -190.7666 * (0.251526 - Q.LHA) * (Q.z_7 - 0.01685855)
    if Q.sj3_dr_max < 0.1986272 and Q.n_dr_0p05_0p1 < 5.0:
        z += 0.5390228 * (0.1986272 - Q.sj3_dr_max) * (5.0 - Q.n_dr_0p05_0p1)
    if Q.LHA < 0.1767241 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -904.4292 * (0.1767241 - Q.LHA) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt > 6.701242 and Q.centroid_offset > 0.006789738:
        z += -203.8321 * (Q.log_sum_pt - 6.701242) * (Q.centroid_offset - 0.006789738)
    if Q.lam1 < 0.005433361 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += 4166.694 * (0.005433361 - Q.lam1) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.lam1 < 0.003377388 and Q.centroid_offset > 0.01627885:
        z += -27756.6 * (0.003377388 - Q.lam1) * (Q.centroid_offset - 0.01627885)
    if Q.tau1 < 0.0262518 and Q.centroid_offset < 0.03117077:
        z += 5840.359 * (0.0262518 - Q.tau1) * (0.03117077 - Q.centroid_offset)
    if Q.lam1 < 0.005433361 and Q.pt_7 < 38.53125:
        z += -9.121419 * (0.005433361 - Q.lam1) * (38.53125 - Q.pt_7)
    if Q.sum_z_dr < 0.03360421 and Q.pt_7 > 15.55391:
        z += 2.463239 * (0.03360421 - Q.sum_z_dr) * (Q.pt_7 - 15.55391)
    if Q.tau1 < 0.019014 and Q.pt_7 > 20.125:
        z += -1.235652 * (0.019014 - Q.tau1) * (Q.pt_7 - 20.125)
    if Q.sum_z_dr < 0.03360421 and Q.centroid_offset < 0.03117077:
        z += -1670.259 * (0.03360421 - Q.sum_z_dr) * (0.03117077 - Q.centroid_offset)
    if Q.lam1 < 0.005433361 and Q.centroid_offset < 0.03117077:
        z += 7124.274 * (0.005433361 - Q.lam1) * (0.03117077 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.1070199 and Q.centroid_offset > 0.01627885:
        z += -3432.121 * (0.1070199 - Q.sj3_dr_max) * (Q.centroid_offset - 0.01627885)
    if Q.sj3_dr_max < 0.1594012 and Q.centroid_offset > 0.003343241:
        z += 321.9524 * (0.1594012 - Q.sj3_dr_max) * (Q.centroid_offset - 0.003343241)
    if Q.sj3_dr_max < 0.1070199 and Q.sum_z_dr2_top3 < 0.0153634:
        z += 4335.573 * (0.1070199 - Q.sj3_dr_max) * (0.0153634 - Q.sum_z_dr2_top3)
    if Q.sj3_dr_max < 0.1426152 and Q.sum_z_dr2_top3 < 0.0153634:
        z += -2470.964 * (0.1426152 - Q.sj3_dr_max) * (0.0153634 - Q.sum_z_dr2_top3)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2_top3 < 0.01002369:
        z += -426.2766 * (Q.log_sum_pt - 6.701242) * (0.01002369 - Q.sum_z_dr2_top3)
    if Q.lam1 < 0.00733008 and Q.sum_z_dr2_top2 < 0.001056655:
        z += -127029.4 * (0.00733008 - Q.lam1) * (0.001056655 - Q.sum_z_dr2_top2)
    if Q.tau1 < 0.03459477 and Q.sum_z_dr2_top2 < 0.001056655:
        z += 15502.63 * (0.03459477 - Q.tau1) * (0.001056655 - Q.sum_z_dr2_top2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2_top5 > 0.002270363:
        z += -2242.639 * (0.1986272 - Q.sj3_dr_max) * (Q.sum_z_dr2_top5 - 0.002270363)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2_top3 < 0.002915531:
        z += 1664.234 * (Q.log_sum_pt - 6.701242) * (0.002915531 - Q.sum_z_dr2_top3)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2_top5 < 0.00633598:
        z += -702.3293 * (Q.log_sum_pt - 6.701242) * (0.00633598 - Q.sum_z_dr2_top5)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2_top5 < 0.00327978:
        z += 1865.971 * (Q.log_sum_pt - 6.701242) * (0.00327978 - Q.sum_z_dr2_top5)
    if Q.sum_z_dr < 0.03360421 and Q.psi_0p1 > 0.9761279:
        z += 835.0616 * (0.03360421 - Q.sum_z_dr) * (Q.psi_0p1 - 0.9761279)
    if Q.tau1 < 0.01357518 and Q.sum_z_dr2_top2 < 0.001056655:
        z += -31532.71 * (0.01357518 - Q.tau1) * (0.001056655 - Q.sum_z_dr2_top2)
    if Q.sj3_dr_max < 0.1070199 and Q.sum_z_dr2_top2 < 0.001864148:
        z += 25335.71 * (0.1070199 - Q.sj3_dr_max) * (0.001864148 - Q.sum_z_dr2_top2)
    if Q.lam1 < 0.005433361 and Q.pt1_dr01 < 5.351077:
        z += 19.33669 * (0.005433361 - Q.lam1) * (5.351077 - Q.pt1_dr01)
    if Q.sj3_dr_max < 0.1426152 and Q.pt1_dr01 < 22.38578:
        z += -0.01079886 * (0.1426152 - Q.sj3_dr_max) * (22.38578 - Q.pt1_dr01)
    if Q.sj3_dr_max < 0.1594012 and Q.sum_z_dr2_top2 < 0.001864148:
        z += -11787.79 * (0.1594012 - Q.sj3_dr_max) * (0.001864148 - Q.sum_z_dr2_top2)
    return max(0.0, z)


def neuron_9(Q):
    z = -6.044573
    if Q.sum_z_dr < 0.02689598:
        z += 259.1058 * Q.sum_z_dr - 10.22014
    if 0.02689598 <= Q.sum_z_dr < 0.04081947:
        z += 172.06 * Q.sum_z_dr - 7.87896
    if 0.04081947 <= Q.sum_z_dr < 0.05464922:
        z += 61.86382 * Q.sum_z_dr - 3.38081
    if Q.e3 < 1.960701e-05:
        z += 57921.54 * Q.e3 - 1.135668
    if 0.0001869378 <= Q.e3 < 0.0005116989:
        z += 6837.862 * Q.e3 - 1.278255
    if Q.e3 >= 0.0005116989:
        z += 1459.045 * Q.e3 + 1.47408
    if Q.centroid_offset < 0.002316125:
        z += 336.6768 * Q.centroid_offset + 2.784881
    if 0.002316125 <= Q.centroid_offset < 0.01627885:
        z += -173.9268 * Q.centroid_offset + 3.967502
    if 0.01627885 <= Q.centroid_offset < 0.02355416:
        z += -96.83842 * Q.centroid_offset + 2.712593
    if 0.02355416 <= Q.centroid_offset < 0.03117077:
        z += -56.67157 * Q.centroid_offset + 1.766497
    if Q.lam1 < 0.0001413912:
        z += -8281.124 * Q.lam1 + 5.727216
    if 0.0001413912 <= Q.lam1 < 0.003377388:
        z += -1147.708 * Q.lam1 + 4.718613
    if 0.003377388 <= Q.lam1 < 0.00595415:
        z += -326.9049 * Q.lam1 + 1.946441
    if Q.lam1 >= 0.01643375:
        z += 146.9955 * Q.lam1 - 2.415687
    if Q.sum_pt < 813.4156:
        z += -0.0004792199 * Q.sum_pt + 1.809053
    if 813.4156 <= Q.sum_pt < 988.4078:
        z += -0.008110351 * Q.sum_pt + 8.016334
    if Q.lam2 < 0.0001947983:
        z += -3863.491 * Q.lam2 + 0.7526014
    if Q.lam2 >= 0.000537286:
        z += 734.845 * Q.lam2 - 0.3948219
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.5114031 * Q.n_dr_0p2_0p4 - 0.5114031
    if Q.e2 < 0.0285317:
        z += 12.93869 * Q.e2 - 0.3691628
    if Q.sj2_dr < 0.1294903:
        z += 8.427637 * Q.sj2_dr - 0.9557781
    if 0.1294903 <= Q.sj2_dr < 0.1778793:
        z += -2.800618 * Q.sj2_dr + 0.498172
    if Q.C2_b2 < 0.009032972:
        z += -89.07771 * Q.C2_b2 + 0.8046365
    if Q.sj3_dr_max < 0.1426152:
        z += 11.27843 * Q.sj3_dr_max - 0.7053946
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -12.7583 * Q.sj3_dr_max + 2.722608
    if Q.max_dr < 0.1027585:
        z += -9.351142 * Q.max_dr + 0.9609095
    if Q.absphi_1 < 0.02227783:
        z += 13.75882 * Q.absphi_1 - 0.3065167
    if Q.pt1_dr01 >= 22.38578:
        z += -0.07421747 * Q.pt1_dr01 + 1.661416
    if Q.planar_flow < 0.5925897:
        z += 0.974549 * Q.planar_flow - 0.5775077
    if Q.log_sum_pt < 6.638339:
        z += -2.492517 * Q.log_sum_pt + 16.54617
    if Q.pt_4 < 31.125:
        z += -0.05826508 * Q.pt_4 + 1.813501
    if Q.lam1 < 0.00595415 and Q.lam2 < 0.001130645:
        z += 1341394.0 * (0.00595415 - Q.lam1) * (0.001130645 - Q.lam2)
    if Q.centroid_offset < 0.01627885 and Q.D2_b2 < 0.1830092:
        z += -686.717 * (0.01627885 - Q.centroid_offset) * (0.1830092 - Q.D2_b2)
    if Q.e3 < 4.192959e-06 and Q.n_dr_0p05_0p1 > 3.0:
        z += -571839.4 * (4.192959e-06 - Q.e3) * (Q.n_dr_0p05_0p1 - 3.0)
    if Q.sum_z_dr < 0.05464922 and Q.mean_phi < 0.0006973656:
        z += -1845.981 * (0.05464922 - Q.sum_z_dr) * (0.0006973656 - Q.mean_phi)
    if Q.sum_pt < 988.4078 and Q.z_5 < 0.05753583:
        z += 0.2578582 * (988.4078 - Q.sum_pt) * (0.05753583 - Q.z_5)
    if Q.lam1 < 0.00595415 and Q.mean_eta < -0.006779839:
        z += -24698.55 * (0.00595415 - Q.lam1) * (-0.006779839 - Q.mean_eta)
    if Q.centroid_offset < 0.01627885 and Q.sj3_dr_min < 0.1278212:
        z += 1595.444 * (0.01627885 - Q.centroid_offset) * (0.1278212 - Q.sj3_dr_min)
    if Q.lam1 < 0.00595415 and Q.sj3_dr_min < 0.08543881:
        z += -4895.817 * (0.00595415 - Q.lam1) * (0.08543881 - Q.sj3_dr_min)
    if Q.centroid_offset < 0.02355416 and Q.lam1 < 0.00733008:
        z += 8119.835 * (0.02355416 - Q.centroid_offset) * (0.00733008 - Q.lam1)
    if Q.zdr_0 < 0.004918231 and Q.pt_1 < 182.125:
        z += -2.175506 * (0.004918231 - Q.zdr_0) * (182.125 - Q.pt_1)
    if Q.lam2 > 0.000537286 and Q.eccentricity < 0.9979797:
        z += -523.4797 * (Q.lam2 - 0.000537286) * (0.9979797 - Q.eccentricity)
    if Q.lam1 < 0.00595415 and Q.mean_eta > 0.009367547:
        z += -27086.2 * (0.00595415 - Q.lam1) * (Q.mean_eta - 0.009367547)
    if Q.lam1 < 0.00595415 and Q.mean_phi < -0.009352575:
        z += -9426.241 * (0.00595415 - Q.lam1) * (-0.009352575 - Q.mean_phi)
    if Q.sum_pt < 813.4156 and Q.absphi_0 > 0.02548218:
        z += -0.02111605 * (813.4156 - Q.sum_pt) * (Q.absphi_0 - 0.02548218)
    if Q.lam1 < 0.00595415 and Q.mean_phi > 0.01251955:
        z += -32830.11 * (0.00595415 - Q.lam1) * (Q.mean_phi - 0.01251955)
    if Q.sum_pt < 813.4156 and Q.e4 < 3.0374e-08:
        z += 260452.6 * (813.4156 - Q.sum_pt) * (3.0374e-08 - Q.e4)
    if Q.centroid_offset < 0.01627885 and Q.N3 < 1.808379:
        z += -44.54306 * (0.01627885 - Q.centroid_offset) * (1.808379 - Q.N3)
    if Q.sum_pt < 813.4156 and Q.sum_z_dr2_top2 < 0.0095303:
        z += 0.06437165 * (813.4156 - Q.sum_pt) * (0.0095303 - Q.sum_z_dr2_top2)
    if Q.planar_flow < 0.5925897 and Q.D2_b2 < 4.721224:
        z += 0.1541857 * (0.5925897 - Q.planar_flow) * (4.721224 - Q.D2_b2)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.D2_b2 < 1.911889:
        z += -0.3635394 * (Q.n_dr_0p2_0p4 - 1.0) * (1.911889 - Q.D2_b2)
    if Q.log_sum_pt < 6.638339 and Q.pt_3 < 39.3125:
        z += 0.1972943 * (6.638339 - Q.log_sum_pt) * (39.3125 - Q.pt_3)
    if Q.tau1 < 0.04369778 and Q.centroid_offset < 0.03117077:
        z += 4944.28 * (0.04369778 - Q.tau1) * (0.03117077 - Q.centroid_offset)
    if Q.sum_pt < 988.4078 and Q.centroid_offset < 0.01627885:
        z += -0.3777746 * (988.4078 - Q.sum_pt) * (0.01627885 - Q.centroid_offset)
    return max(0.0, z)


def neuron_10(Q):
    z = 3.660934
    z += -25.23737 * Q.e2
    if Q.lam1 < 0.0008722282:
        z += 697.8145 * Q.lam1 - 0.8058058
    if 0.0008722282 <= Q.lam1 < 0.003377388:
        z += 648.3731 * Q.lam1 - 0.7626816
    if 0.003377388 <= Q.lam1 < 0.00483998:
        z += 409.7846 * Q.lam1 + 0.0431242
    if Q.lam1 >= 0.00483998:
        z += 287.4374 * Q.lam1 + 0.6352822
    if 0.0001947983 <= Q.lam2 < 0.003408389:
        z += 1469.388 * Q.lam2 - 0.2862343
    if Q.lam2 >= 0.003408389:
        z += 946.2973 * Q.lam2 + 1.496662
    if Q.e3 >= 8.147744e-05:
        z += 1166.005 * Q.e3 - 0.09500308
    if 0.1967397 <= Q.LHA < 0.3127275:
        z += -14.62272 * Q.LHA + 2.876869
    if Q.LHA >= 0.3127275:
        z += -22.71887 * Q.LHA + 5.408758
    if Q.centroid_offset >= 0.02355416:
        z += 22.07472 * Q.centroid_offset - 0.5199516
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.7205257 * Q.n_dr_0p2_0p4 - 0.7205257
    if Q.z_7 < 0.06810151:
        z += 22.51045 * Q.z_7 - 1.532996
    if Q.tau1 >= 0.04369778:
        z += 14.56264 * Q.tau1 - 0.6363549
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += 18.07271 * Q.sj2_dr - 2.34024
    if 0.1591713 <= Q.sj2_dr < 0.1682655:
        z += -4.752615 * Q.sj2_dr + 1.292896
    if 0.1682655 <= Q.sj2_dr < 0.3003793:
        z += 0.6314816 * Q.sj2_dr + 0.386938
    if Q.sj2_dr >= 0.3003793:
        z += 10.33515 * Q.sj2_dr - 2.527843
    if Q.sum_z_dr2_top3 < 0.002915531:
        z += -199.5092 * Q.sum_z_dr2_top3 + 0.5816752
    if Q.sum_pt >= 988.4078:
        z += 0.006348338 * Q.sum_pt - 6.274747
    if Q.sum_z_dr < 0.1484084:
        z += 17.71926 * Q.sum_z_dr - 2.629687
    if Q.D2_b2 < 0.2669656:
        z += -2.264608 * Q.D2_b2 + 0.6045725
    if Q.log_sum_pt >= 6.670067:
        z += -3.895994 * Q.log_sum_pt + 25.98654
    if Q.lam2 > 0.0001947983 and Q.log_sum_pt < 6.538559:
        z += -955.1746 * (Q.lam2 - 0.0001947983) * (6.538559 - Q.log_sum_pt)
    if Q.lam2 > 0.0001947983 and Q.eccentricity > 0.4829602:
        z += -1270.078 * (Q.lam2 - 0.0001947983) * (Q.eccentricity - 0.4829602)
    if Q.pt_6 < 48.03125 and Q.log_sum_pt < 6.46415:
        z += -0.0661571 * (48.03125 - Q.pt_6) * (6.46415 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.D2_b2 < 4.721224:
        z += -0.1488989 * (Q.n_dr_0p2_0p4 - 1.0) * (4.721224 - Q.D2_b2)
    if Q.lam2 > 0.0001947983 and Q.planar_flow > 0.06126205:
        z += -950.2601 * (Q.lam2 - 0.0001947983) * (Q.planar_flow - 0.06126205)
    if Q.sj2_dr > 0.1682655 and Q.planar_flow < 0.6947818:
        z += -23.82163 * (Q.sj2_dr - 0.1682655) * (0.6947818 - Q.planar_flow)
    if Q.LHA > 0.3127275 and Q.planar_flow < 0.6947818:
        z += -23.28649 * (Q.LHA - 0.3127275) * (0.6947818 - Q.planar_flow)
    if Q.lam2 > 0.003408389 and Q.pt_2 < 111.75:
        z += 8.683213 * (Q.lam2 - 0.003408389) * (111.75 - Q.pt_2)
    if Q.lam2 > 0.003408389 and Q.pt2_over_pt0 > 0.1852611:
        z += 434.3091 * (Q.lam2 - 0.003408389) * (Q.pt2_over_pt0 - 0.1852611)
    if Q.lam1 < 0.003377388 and Q.dr_7 < 0.2229947:
        z += -1331.399 * (0.003377388 - Q.lam1) * (0.2229947 - Q.dr_7)
    if Q.lam1 > 0.00483998 and Q.sj3_pairmin_over_m < 0.5042684:
        z += -218.1634 * (Q.lam1 - 0.00483998) * (0.5042684 - Q.sj3_pairmin_over_m)
    if Q.pt_7 > 20.125 and Q.D2_b2 < 0.2669656:
        z += -0.1419385 * (Q.pt_7 - 20.125) * (0.2669656 - Q.D2_b2)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.1124623
    if Q.planar_flow < 0.2534037:
        z += -13.38143 * Q.planar_flow + 3.390904
    if Q.e3 < 7.330536e-06:
        z += 162758.4 * Q.e3 - 1.193106
    if Q.pt_7 < 45.75:
        z += 0.04098492 * Q.pt_7 - 1.87506
    if Q.lam1 < 0.00595415:
        z += -620.9609 * Q.lam1 + 5.738279
    if 0.00595415 <= Q.lam1 < 0.01200373:
        z += -396.7058 * Q.lam1 + 4.403031
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 81.01929 * Q.lam1 - 1.331451
    if Q.sum_z_dr < 0.06108601:
        z += 99.5704 * Q.sum_z_dr - 6.524732
    if 0.06108601 <= Q.sum_z_dr < 0.0717028:
        z += 41.66732 * Q.sum_z_dr - 2.987664
    if Q.e2 < 0.04447357:
        z += 31.84606 * Q.e2 - 1.233153
    if 0.04447357 <= Q.e2 < 0.05028464:
        z += -31.5182 * Q.e2 + 1.584881
    if Q.tau2 < 0.01133547:
        z += -40.23332 * Q.tau2 + 0.4560637
    if Q.centroid_offset < 0.01837778:
        z += -87.79581 * Q.centroid_offset + 1.613492
    if Q.centroid_offset >= 0.02355416:
        z += -51.40448 * Q.centroid_offset + 1.210789
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += 14.58407 * Q.sj2_dr - 1.888496
    if 0.1591713 <= Q.sj2_dr < 0.2687922:
        z += 3.733761 * Q.sj2_dr - 0.1614377
    if Q.sj2_dr >= 0.2687922:
        z += 18.7781 * Q.sj2_dr - 4.205238
    if 0.1426152 <= Q.sj3_dr_max < 0.1789613:
        z += -3.040751 * Q.sj3_dr_max + 0.4336573
    if 0.1789613 <= Q.sj3_dr_max < 0.2623172:
        z += -19.75991 * Q.sj3_dr_max + 3.42574
    if Q.sj3_dr_max >= 0.2623172:
        z += -13.20729 * Q.sj3_dr_max + 1.706876
    if Q.mean_phi2 < 0.002127561:
        z += 191.7601 * Q.mean_phi2 - 0.4079812
    if Q.tau1 < 0.07283629:
        z += -37.71484 * Q.tau1 + 2.747009
    if Q.tau21 < 0.1357675:
        z += 11.28471 * Q.tau21 - 1.532097
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.01590163 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.centroid_offset < 0.04990367:
        z += -77.96554 * (0.2534037 - Q.planar_flow) * (0.04990367 - Q.centroid_offset)
    if Q.lam1 < 0.01643375 and Q.centroid_offset > 0.01837778:
        z += 1618.83 * (0.01643375 - Q.lam1) * (Q.centroid_offset - 0.01837778)
    if Q.sj2_dr > 0.2414351 and Q.sum_pt < 739.5:
        z += 0.006872601 * (Q.sj2_dr - 0.2414351) * (739.5 - Q.sum_pt)
    if Q.lam1 < 0.003377388 and Q.centroid_offset < 0.02685622:
        z += -37823.78 * (0.003377388 - Q.lam1) * (0.02685622 - Q.centroid_offset)
    if Q.centroid_offset < 0.01837778 and Q.sum_pt < 937.0312:
        z += -0.1658342 * (0.01837778 - Q.centroid_offset) * (937.0312 - Q.sum_pt)
    if Q.lam1 < 0.01200373 and Q.centroid_offset > 0.01627885:
        z += -16116.1 * (0.01200373 - Q.lam1) * (Q.centroid_offset - 0.01627885)
    if Q.lam1 < 0.01643375 and Q.lam2 < 0.001130645:
        z += 136811.8 * (0.01643375 - Q.lam1) * (0.001130645 - Q.lam2)
    if Q.lam1 < 0.003377388 and Q.planar_flow < 0.2534037:
        z += -2858.017 * (0.003377388 - Q.lam1) * (0.2534037 - Q.planar_flow)
    if Q.tau2 < 0.01133547 and Q.pt1_dr01 > 22.38578:
        z += -9.525689 * (0.01133547 - Q.tau2) * (Q.pt1_dr01 - 22.38578)
    if Q.planar_flow < 0.2534037 and Q.D3 < 3.916009:
        z += -0.8134627 * (0.2534037 - Q.planar_flow) * (3.916009 - Q.D3)
    if Q.lam2 < 0.000537286 and Q.C3 > 0.04473419:
        z += -83086.67 * (0.000537286 - Q.lam2) * (Q.C3 - 0.04473419)
    if Q.lam1 < 0.01200373 and Q.dr1_7 > 0.2411619:
        z += -4145.41 * (0.01200373 - Q.lam1) * (Q.dr1_7 - 0.2411619)
    if Q.sum_z_dr < 0.06108601 and Q.dr1_7 > 0.2025074:
        z += 860.448 * (0.06108601 - Q.sum_z_dr) * (Q.dr1_7 - 0.2025074)
    if Q.zdr_0 < 0.004918231 and Q.dr0_7 > 0.09118326:
        z += -4041.012 * (0.004918231 - Q.zdr_0) * (Q.dr0_7 - 0.09118326)
    if Q.e3 < 7.330536e-06 and Q.D2_b2 < 0.716559:
        z += 443795.4 * (7.330536e-06 - Q.e3) * (0.716559 - Q.D2_b2)
    if Q.lam1 < 0.00595415 and Q.D2_b2 < 0.716559:
        z += -1039.164 * (0.00595415 - Q.lam1) * (0.716559 - Q.D2_b2)
    if Q.lam1 < 0.008375572 and Q.D2_b2 < 0.716559:
        z += 366.5133 * (0.008375572 - Q.lam1) * (0.716559 - Q.D2_b2)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.6892291
    if 0.1245537 <= Q.sum_z_dr < 0.1484084:
        z += 123.7413 * Q.sum_z_dr - 15.41245
    if Q.sum_z_dr >= 0.1484084:
        z += 158.4142 * Q.sum_z_dr - 20.5582
    if Q.n_dr_0p2_0p4 >= 2.0:
        z += 0.344421 * Q.n_dr_0p2_0p4 - 0.688842
    if Q.sd_rg >= 0.324646:
        z += 47.83123 * Q.sd_rg - 15.52822
    if Q.centroid_offset >= 0.03776099:
        z += 16.4741 * Q.centroid_offset - 0.6220784
    if Q.LHA >= 0.3855647:
        z += -34.92741 * Q.LHA + 13.46678
    if Q.e2 >= 0.06344108:
        z += -44.63586 * Q.e2 + 2.831747
    if Q.sum_z_dr > 0.1245537 and Q.log_sum_pt > 6.502799:
        z += 373.0777 * (Q.sum_z_dr - 0.1245537) * (Q.log_sum_pt - 6.502799)
    if Q.sum_z_dr > 0.1245537 and Q.C2_b2 > 0.009032972:
        z += 1078.809 * (Q.sum_z_dr - 0.1245537) * (Q.C2_b2 - 0.009032972)
    if Q.sd_rg > 0.324646 and Q.log_sum_pt < 6.896095:
        z += -171.5818 * (Q.sd_rg - 0.324646) * (6.896095 - Q.log_sum_pt)
    if Q.sum_z_dr > 0.1245537 and Q.pt_6 < 62.25:
        z += -1.131125 * (Q.sum_z_dr - 0.1245537) * (62.25 - Q.pt_6)
    if Q.sum_z_dr > 0.1484084 and Q.log_sum_pt > 6.502799:
        z += -487.3901 * (Q.sum_z_dr - 0.1484084) * (Q.log_sum_pt - 6.502799)
    if Q.sum_z_dr > 0.1484084 and Q.sum_pt < 588.5859:
        z += -0.166948 * (Q.sum_z_dr - 0.1484084) * (588.5859 - Q.sum_pt)
    if Q.sd_rg > 0.324646 and Q.log_sum_pt < 6.701242:
        z += 162.6646 * (Q.sd_rg - 0.324646) * (6.701242 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.093027
    if Q.sum_z_dr < 0.1484084:
        z += -81.01021 * Q.sum_z_dr + 12.0226
    if Q.lam1 < 0.01643375:
        z += -226.406 * Q.lam1 + 3.7207
    if 687.4375 <= Q.sum_pt_top5 < 791.125:
        z += 0.005644461 * Q.sum_pt_top5 - 3.880214
    if 791.125 <= Q.sum_pt_top5 < 839.9547:
        z += 0.01341713 * Q.sum_pt_top5 - 10.02937
    if Q.sum_pt_top5 >= 839.9547:
        z += 0.01590635 * Q.sum_pt_top5 - 12.1202
    if Q.pt_6 < 27.57812:
        z += 0.07721015 * Q.pt_6 - 2.654473
    if 27.57812 <= Q.pt_6 < 50.25:
        z += 0.02316361 * Q.pt_6 - 1.163971
    if Q.sum_pt >= 988.4078:
        z += -0.01197619 * Q.sum_pt + 11.83736
    if Q.sj3_dr_max >= 0.233678:
        z += -12.39643 * Q.sj3_dr_max + 2.896772
    if Q.pt_5 < 24.57812:
        z += 0.2066281 * Q.pt_5 - 5.078531
    if Q.e3 < 8.147744e-05:
        z += -7877.199 * Q.e3 + 0.641814
    if Q.e2 < 0.08000524:
        z += 20.67455 * Q.e2 - 1.654072
    if Q.sj3_dr_min < 0.1278212:
        z += 3.858227 * Q.sj3_dr_min - 0.4931631
    if Q.log_sum_pt < 6.46415:
        z += 1.588953 * Q.log_sum_pt - 10.27123
    if Q.log_sum_pt >= 6.670067:
        z += -7.138858 * Q.log_sum_pt + 47.61666
    if Q.pt_7 < 31.85938:
        z += 0.03544329 * Q.pt_7 - 1.726753
    if 31.85938 <= Q.pt_7 < 48.71875:
        z += -0.08271355 * Q.pt_7 + 2.03765
    if Q.pt_7 >= 48.71875:
        z += -0.1181568 * Q.pt_7 + 3.764403
    if Q.tau1 < 0.1136369:
        z += 22.39016 * Q.tau1 - 2.544348
    if Q.z_7 >= 0.04624032:
        z += 57.59603 * Q.z_7 - 2.663259
    if Q.z_top5 >= 0.8770155:
        z += -7.447598 * Q.z_top5 + 6.531659
    if Q.C2_b2 < 0.009032972:
        z += 186.3089 * Q.C2_b2 - 1.682923
    if Q.z_dr_0_0p05 >= 0.6080732:
        z += -1.20966 * Q.z_dr_0_0p05 + 0.735562
    if Q.sum_z_dr < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -78.63358 * (0.1484084 - Q.sum_z_dr) * (6.804164 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.1484084 and Q.pt_7 < 38.53125:
        z += -0.3748623 * (0.1484084 - Q.sum_z_dr) * (38.53125 - Q.pt_7)
    if Q.sum_z_dr < 0.1484084 and Q.lam2 < 0.0003061234:
        z += -11190.92 * (0.1484084 - Q.sum_z_dr) * (0.0003061234 - Q.lam2)
    if Q.sum_pt > 988.4078 and Q.z_6 > 0.03932388:
        z += 0.7039536 * (Q.sum_pt - 988.4078) * (Q.z_6 - 0.03932388)
    if Q.sum_pt_top5 > 687.4375 and Q.pt_7 < 40.04062:
        z += 0.0006148427 * (Q.sum_pt_top5 - 687.4375) * (40.04062 - Q.pt_7)
    if Q.lam1 < 0.01643375 and Q.pt_7 < 25.57812:
        z += -12.40205 * (0.01643375 - Q.lam1) * (25.57812 - Q.pt_7)
    if Q.e3 < 8.147744e-05 and Q.centroid_offset < 0.03776099:
        z += -723810.3 * (8.147744e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.z_7 > 0.0586137 and Q.lam2 > 4.64158e-05:
        z += -8625.679 * (Q.z_7 - 0.0586137) * (Q.lam2 - 4.64158e-05)
    if Q.log_sum_pt < 6.46415 and Q.pt_4 < 31.125:
        z += -0.2922173 * (6.46415 - Q.log_sum_pt) * (31.125 - Q.pt_4)
    if Q.e2 < 0.08000524 and Q.pt_4 < 60.03125:
        z += 0.3218781 * (0.08000524 - Q.e2) * (60.03125 - Q.pt_4)
    if Q.sum_pt > 988.4078 and Q.sj3_pairmin_over_m > 0.2195798:
        z += -0.03411305 * (Q.sum_pt - 988.4078) * (Q.sj3_pairmin_over_m - 0.2195798)
    if Q.pt_7 > 31.85938 and Q.max_dr < 0.1598486:
        z += 0.2814063 * (Q.pt_7 - 31.85938) * (0.1598486 - Q.max_dr)
    if Q.pt_7 > 31.85938 and Q.zdr_1 < 0.01228369:
        z += 1.751637 * (Q.pt_7 - 31.85938) * (0.01228369 - Q.zdr_1)
    if Q.z_7 > 0.04624032 and Q.max_dr > 0.08050702:
        z += 71.33181 * (Q.z_7 - 0.04624032) * (Q.max_dr - 0.08050702)
    if Q.log_sum_pt > 6.670067 and Q.C2_b2 > 0.0001052765:
        z += -447.3581 * (Q.log_sum_pt - 6.670067) * (Q.C2_b2 - 0.0001052765)
    return max(0.0, z)


def neuron_14(Q):
    z = -5.000432
    if Q.sd_rg < 0.06545715:
        z += -0.1415088 * Q.sd_rg + 1.453253
    if 0.06545715 <= Q.sd_rg < 0.1116471:
        z += -17.1076 * Q.sd_rg + 2.563805
    if 0.1116471 <= Q.sd_rg < 0.1449048:
        z += -15.97999 * Q.sd_rg + 2.437911
    if 0.1449048 <= Q.sd_rg < 0.1572969:
        z += -11.95415 * Q.sd_rg + 1.854547
    if 0.1572969 <= Q.sd_rg < 0.1667615:
        z += -2.049737 * Q.sd_rg + 0.2966141
    if 0.1667615 <= Q.sd_rg < 0.1773029:
        z += 31.99842 * Q.sd_rg - 5.381308
    if 0.1773029 <= Q.sd_rg < 0.1876504:
        z += 14.58402 * Q.sd_rg - 2.293684
    if 0.1876504 <= Q.sd_rg < 0.2330919:
        z += -12.9601 * Q.sd_rg + 2.874981
    if Q.sd_rg >= 0.2330919:
        z += -13.61386 * Q.sd_rg + 3.027367
    if Q.lam1 < 0.006506576:
        z += 609.6105 * Q.lam1 - 2.843499
    if 0.006506576 <= Q.lam1 < 0.00733008:
        z += -87.01906 * Q.lam1 + 1.689174
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += -224.9459 * Q.lam1 + 2.700189
    if 0.02689598 <= Q.sum_z_dr < 0.06663269:
        z += 94.65022 * Q.sum_z_dr - 2.54571
    if 0.06663269 <= Q.sum_z_dr < 0.08723651:
        z += 125.4291 * Q.sum_z_dr - 4.596588
    if Q.sum_z_dr >= 0.08723651:
        z += -85.80575 * Q.sum_z_dr + 13.8308
    if Q.tau1 < 0.09538712:
        z += -37.85058 * Q.tau1 + 4.027327
    if 0.09538712 <= Q.tau1 < 0.1136369:
        z += -5.587979 * Q.tau1 + 0.9498906
    if 0.1136369 <= Q.tau1 < 0.1416226:
        z += -11.2518 * Q.tau1 + 1.593509
    if Q.centroid_offset >= 0.04990367:
        z += -1273.227 * Q.centroid_offset + 63.53869
    if Q.e2 < 0.01289969:
        z += -5.730003 * Q.e2 + 2.096125
    if 0.01289969 <= Q.e2 < 0.04110972:
        z += -71.68409 * Q.e2 + 2.946913
    if Q.LHA >= 0.3033137:
        z += 10.14071 * Q.LHA - 3.075818
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 256.8401 * Q.sum_z_dr2_top3 - 0.5526089
    if Q.e3 < 8.147744e-05:
        z += 9621.036 * Q.e3 - 0.7838974
    if Q.n_dr_0p05_0p1 < 5.0:
        z += 0.1525832 * Q.n_dr_0p05_0p1 - 0.762916
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -6821.122 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += 938.3052 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00595415:
        z += 2892.785 * (0.1115136 - Q.planar_flow) * (0.00595415 - Q.lam1)
    if Q.lam2 < 0.003408389 and Q.sd_rg > 0.2037854:
        z += -4059.512 * (0.003408389 - Q.lam2) * (Q.sd_rg - 0.2037854)
    if Q.planar_flow < 0.1115136 and Q.sj2_zsoft < 0.2832687:
        z += 36.46486 * (0.1115136 - Q.planar_flow) * (0.2832687 - Q.sj2_zsoft)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01627885:
        z += -256.5586 * (0.1115136 - Q.planar_flow) * (0.01627885 - Q.centroid_offset)
    if Q.lam2 < 0.003408389 and Q.centroid_offset > 0.02685622:
        z += 13028.24 * (0.003408389 - Q.lam2) * (Q.centroid_offset - 0.02685622)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.z_dr_0p1_0p2 < 0.04510668:
        z += 21.67384 * (0.5882598 - Q.z_dr_0p05_0p1) * (0.04510668 - Q.z_dr_0p1_0p2)
    if Q.lam2 < 0.003408389 and Q.LHA > 0.1767241:
        z += -1020.32 * (0.003408389 - Q.lam2) * (Q.LHA - 0.1767241)
    if Q.lam2 < 0.003408389 and Q.sd_rg > 0.1572969:
        z += 9449.372 * (0.003408389 - Q.lam2) * (Q.sd_rg - 0.1572969)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.n_dr_0p2_0p4 < 1.0:
        z += 2.025981 * (Q.z_dr_0p05_0p1 - 0.163898) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.log_sum_pt < 6.670067:
        z += -3.083267 * (Q.z_dr_0p05_0p1 - 0.163898) * (6.670067 - Q.log_sum_pt)
    if Q.lam1 < 0.01643375 and Q.centroid_offset > 0.02685622:
        z += -12986.19 * (0.01643375 - Q.lam1) * (Q.centroid_offset - 0.02685622)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.z_dr_0p2_0p4 < 0.20552:
        z += -10.10203 * (Q.z_dr_0p05_0p1 - 0.163898) * (0.20552 - Q.z_dr_0p2_0p4)
    if Q.lam1 < 0.006506576 and Q.sj3_dr23 > 0.1974628:
        z += 2707.233 * (0.006506576 - Q.lam1) * (Q.sj3_dr23 - 0.1974628)
    if Q.lam2 < 0.003408389 and Q.z_dr_0p2_0p4 > 0.0:
        z += -1518.71 * (0.003408389 - Q.lam2) * (Q.z_dr_0p2_0p4 - 0.0)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.psi_0p3 < 1.0:
        z += -1470.874 * (Q.z_dr_0p05_0p1 - 0.163898) * (1.0 - Q.psi_0p3)
    if Q.tau1 < 0.1136369 and Q.D2 < 0.415246:
        z += 1.46173 * (0.1136369 - Q.tau1) * (0.415246 - Q.D2)
    if Q.lam1 < 0.01643375 and Q.D2 < 0.415246:
        z += 111.0414 * (0.01643375 - Q.lam1) * (0.415246 - Q.D2)
    if Q.planar_flow < 0.1115136 and Q.sd_nremoved < 1.0:
        z += -4.746702 * (0.1115136 - Q.planar_flow) * (1.0 - Q.sd_nremoved)
    if Q.e2 < 0.04110972 and Q.mean_eta < -0.02665591:
        z += 2427.9 * (0.04110972 - Q.e2) * (-0.02665591 - Q.mean_eta)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.4113381
    if Q.lam1 < 0.002464291:
        z += -733.8023 * Q.lam1 + 2.486718
    if 0.002464291 <= Q.lam1 < 0.005433361:
        z += 100.5958 * Q.lam1 + 0.4305175
    if 0.005433361 <= Q.lam1 < 0.00733008:
        z += 536.1202 * Q.lam1 - 1.935843
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += -65.13247 * Q.lam1 + 2.471387
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -278.0041 * Q.lam1 + 4.254308
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += -207.0471 * Q.lam1 + 3.40256
    if Q.tau1 < 0.05356915:
        z += -47.75542 * Q.tau1 + 2.558217
    if 0.09538712 <= Q.tau1 < 0.1027642:
        z += -88.27397 * Q.tau1 + 8.4202
    if 0.1027642 <= Q.tau1 < 0.1136369:
        z += -1.400459 * Q.tau1 - 0.5072896
    if 0.1136369 <= Q.tau1 < 0.1416226:
        z += 9.922627 * Q.tau1 - 1.79401
    if Q.tau1 >= 0.1416226:
        z += -58.2752 * Q.tau1 + 7.864346
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += -1249.927 * Q.sum_z_dr2_top3 + 2.689302
    if Q.sum_z_dr2_top2 < 0.007639643:
        z += -225.7216 * Q.sum_z_dr2_top2 + 1.724433
    if Q.eccentricity >= 0.9458207:
        z += -11.94504 * Q.eccentricity + 11.29786
    if Q.zdr_0 < 0.009233892:
        z += -73.17148 * Q.zdr_0 + 0.6756575
    if Q.sum_z_dr < 0.03360421:
        z += 244.4171 * Q.sum_z_dr - 16.23524
    if 0.03360421 <= Q.sum_z_dr < 0.0717028:
        z += 187.318 * Q.sum_z_dr - 14.31647
    if 0.0717028 <= Q.sum_z_dr < 0.08723651:
        z += 56.98864 * Q.sum_z_dr - 4.971491
    if Q.e2 < 0.032347:
        z += -151.3136 * Q.e2 + 5.656333
    if 0.032347 <= Q.e2 < 0.04110972:
        z += -86.93578 * Q.e2 + 3.573905
    if Q.LHA < 0.1329373:
        z += 27.78529 * Q.LHA - 6.707249
    if 0.1329373 <= Q.LHA < 0.3033137:
        z += 16.51645 * Q.LHA - 5.209199
    if 0.3033137 <= Q.LHA < 0.3127275:
        z += 21.19597 * Q.LHA - 6.628563
    if 0.181053 <= Q.sj3_dr13 < 0.249546:
        z += 12.37191 * Q.sj3_dr13 - 2.239971
    if Q.sj3_dr13 >= 0.249546:
        z += -8.782367 * Q.sj3_dr13 + 3.038993
    if Q.e3 < 7.330536e-06:
        z += 163655.7 * Q.e3 - 1.199684
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1872617:
        z += -364.8135 * (0.2233283 - Q.N2) * (0.1872617 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.lam1 > 0.008375572:
        z += -3706.482 * (0.2233283 - Q.N2) * (Q.lam1 - 0.008375572)
    if Q.N2 < 0.2233283 and Q.LHA > 0.2931906:
        z += 248.9835 * (0.2233283 - Q.N2) * (Q.LHA - 0.2931906)
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1591713:
        z += 243.0416 * (0.2233283 - Q.N2) * (0.1591713 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.LHA > 0.4235925:
        z += -3737.399 * (0.2233283 - Q.N2) * (Q.LHA - 0.4235925)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3255822:
        z += -141.1498 * (0.2233283 - Q.N2) * (Q.LHA - 0.3255822)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3855647:
        z += -505.1907 * (0.2233283 - Q.N2) * (Q.LHA - 0.3855647)
    if Q.N2 < 0.2233283 and Q.z_7 < 0.02320757:
        z += -1566.593 * (0.2233283 - Q.N2) * (0.02320757 - Q.z_7)
    if Q.sum_z_dr2_top3 < 0.002151568 and Q.planar_flow < 0.1950135:
        z += -8779.021 * (0.002151568 - Q.sum_z_dr2_top3) * (0.1950135 - Q.planar_flow)
    if Q.lam2 < 0.003408389 and Q.zdr_1 < 0.01849752:
        z += -11448.11 * (0.003408389 - Q.lam2) * (0.01849752 - Q.zdr_1)
    if Q.eccentricity > 0.9458207 and Q.mean_phi > -0.01753483:
        z += -439.7617 * (Q.eccentricity - 0.9458207) * (Q.mean_phi - -0.01753483)
    if Q.sum_z_dr2_top2 < 0.007639643 and Q.D3 < 0.2213841:
        z += 1483.816 * (0.007639643 - Q.sum_z_dr2_top2) * (0.2213841 - Q.D3)
    if Q.zdr_0 < 0.0211821 and Q.D3 < 0.2213841:
        z += -611.5473 * (0.0211821 - Q.zdr_0) * (0.2213841 - Q.D3)
    if Q.D3 < 0.2213841 and Q.n_dr_0p1_0p2 > 1.0:
        z += -206.7958 * (0.2213841 - Q.D3) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.N2 < 0.2233283 and Q.z_dr_0p2_0p4 > 0.20552:
        z += -219.0044 * (0.2233283 - Q.N2) * (Q.z_dr_0p2_0p4 - 0.20552)
    if Q.eccentricity > 0.9458207 and Q.sum_pt_top5 > 367.5938:
        z += 0.1394804 * (Q.eccentricity - 0.9458207) * (Q.sum_pt_top5 - 367.5938)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.sj3_dr23 > 0.1797097:
        z += -78.47702 * (Q.z_dr_0p05_0p1 - 0.7509095) * (Q.sj3_dr23 - 0.1797097)
    if Q.lam1 < 0.008375572 and Q.sj3_dr23 > 0.1974628:
        z += 10165.57 * (0.008375572 - Q.lam1) * (Q.sj3_dr23 - 0.1974628)
    if Q.lam1 < 0.008375572 and Q.D2 < 0.875672:
        z += -1254.071 * (0.008375572 - Q.lam1) * (0.875672 - Q.D2)
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += 1899.196 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.lam1 < 0.01643375 and Q.D2 < 0.875672:
        z += 65.73891 * (0.01643375 - Q.lam1) * (0.875672 - Q.D2)
    if Q.lam1 < 0.00595415 and Q.sj3_dr23 > 0.1974628:
        z += -12772.05 * (0.00595415 - Q.lam1) * (Q.sj3_dr23 - 0.1974628)
    if Q.lam1 < 0.01643375 and Q.C2_b2 > 0.009032972:
        z += -8608.528 * (0.01643375 - Q.lam1) * (Q.C2_b2 - 0.009032972)
    if Q.lam1 < 0.00733008 and Q.centroid_offset > 0.03776099:
        z += 81486.88 * (0.00733008 - Q.lam1) * (Q.centroid_offset - 0.03776099)
    if Q.lam1 < 0.01643375 and Q.centroid_offset > 0.03776099:
        z += -24162.71 * (0.01643375 - Q.lam1) * (Q.centroid_offset - 0.03776099)
    if Q.tau1 > 0.1027642 and Q.D2_b2 < 0.5327104:
        z += 118.8044 * (Q.tau1 - 0.1027642) * (0.5327104 - Q.D2_b2)
    if Q.tau1 > 0.1416226 and Q.D2_b2 < 0.5327104:
        z += 167.6773 * (Q.tau1 - 0.1416226) * (0.5327104 - Q.D2_b2)
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
