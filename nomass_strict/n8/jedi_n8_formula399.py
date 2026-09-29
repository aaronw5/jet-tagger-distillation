"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network (step 4; no mass observables or exact equivalents), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 89.8% of jets.

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
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_nremoved            number of branches removed by soft drop
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.eta_0                  Δη of particle 0
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
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        pt_2=pt[2],
        pt_3=pt[3],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_nremoved=softdrop("removed"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        eta_0=eta[0],
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
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = 1.63
    if Q.LHA >= 0.3:
        z += -24.5 * Q.LHA + 7.35
    if Q.centroid_offset >= 0.0064:
        z += -81.3 * Q.centroid_offset + 0.52032
    if Q.e3 < 8e-05:
        z += -13500.0 * Q.e3 + 1.08
    if Q.sum_z_dr < 0.08:
        z += 48.9 * Q.sum_z_dr - 3.912
    if Q.lam1 < 0.0043:
        z += 541.0 * Q.lam1 - 0.5784
    if 0.0043 <= Q.lam1 < 0.012:
        z += -227.0 * Q.lam1 + 2.724
    if Q.log_sum_pt >= 6.8:
        z += -26.9 * Q.log_sum_pt + 182.92
    if Q.n_dr_0p2_0p4 < 0.99:
        z += -0.627 * Q.n_dr_0p2_0p4 + 0.62073
    if Q.planar_flow < 0.15:
        z += -5.34 * Q.planar_flow + 0.801
    if Q.pt_6 < 27.0:
        z += 0.0586 * Q.pt_6 - 1.5822
    if 0.14 <= Q.sj3_dr_max < 0.19:
        z += 13.1 * Q.sj3_dr_max - 1.834
    if Q.sj3_dr_max >= 0.19:
        z += -13.1 * Q.sj3_dr_max + 3.144
    if Q.sum_pt < 760.0:
        z += 0.0117 * Q.sum_pt - 8.892
    if Q.sum_pt_top5 >= 790.0:
        z += 0.0165 * Q.sum_pt_top5 - 13.035
    if Q.tau1 < 0.052:
        z += -54.1 * Q.tau1 + 2.8132
    if Q.tau2 < 0.018:
        z += 116.0 * Q.tau2 - 2.088
    if Q.lam1 < 0.0042 and Q.D2 < 0.75:
        z += -5810.0 * (0.0042 - Q.lam1) * (0.75 - Q.D2)
    if Q.lam1 < 0.0059 and Q.n_dr_0p2_0p4 < 0.98:
        z += -306.0 * (0.0059 - Q.lam1) * (0.98 - Q.n_dr_0p2_0p4)
    if Q.lam1 < 0.0044 and Q.sum_pt < 770.0:
        z += 2.11 * (0.0044 - Q.lam1) * (770.0 - Q.sum_pt)
    if Q.log_sum_pt > 6.8 and Q.z_dr_0p1_0p2 < 0.046:
        z += 122.0 * (Q.log_sum_pt - 6.8) * (0.046 - Q.z_dr_0p1_0p2)
    if Q.planar_flow < 0.14 and Q.C3 > 0.05:
        z += 352.0 * (0.14 - Q.planar_flow) * (Q.C3 - 0.05)
    if Q.planar_flow < 0.043 and Q.tau21_b2 < 0.057:
        z += 307.0 * (0.043 - Q.planar_flow) * (0.057 - Q.tau21_b2)
    if Q.sj3_dr_max > 0.12 and Q.D2 > -0.076:
        z += 1.42 * (Q.sj3_dr_max - 0.12) * (Q.D2 - -0.076)
    if Q.sj3_dr_max > 0.15 and Q.dr_7 < 0.067:
        z += 87.6 * (Q.sj3_dr_max - 0.15) * (0.067 - Q.dr_7)
    if Q.sum_pt < 770.0 and Q.centroid_offset > 0.011:
        z += 0.197 * (770.0 - Q.sum_pt) * (Q.centroid_offset - 0.011)
    if Q.sum_pt < 720.0 and Q.z_7 > 0.017:
        z += 0.0611 * (720.0 - Q.sum_pt) * (Q.z_7 - 0.017)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.31
    if Q.sum_z_dr < 0.11:
        z += 38.2 * Q.sum_z_dr - 4.202
    if Q.lam1 < 0.0025:
        z += -530.0 * Q.lam1 + 1.325
    if Q.lam2 < 0.0013:
        z += -646.0 * Q.lam2 + 0.8398
    if 6.4 <= Q.log_sum_pt < 6.6:
        z += 7.24 * Q.log_sum_pt - 46.336
    if Q.log_sum_pt >= 6.6:
        z += 14.1 * Q.log_sum_pt - 91.612
    if 42.0 <= Q.pt_7 < 54.0:
        z += 0.0841 * Q.pt_7 - 3.5322
    if Q.pt_7 >= 54.0:
        z += -0.0579 * Q.pt_7 + 4.1358
    if Q.sj3_dr_max < 0.34:
        z += 3.47 * Q.sj3_dr_max - 1.1798
    if Q.tau1 < 0.1:
        z += -25.8 * Q.tau1 + 2.58
    if Q.z_6 < 0.048:
        z += 107.0 * Q.z_6 - 5.136
    if Q.z_7 < 0.064:
        z += 96.8 * Q.z_7 - 6.1952
    if Q.lam1 < 0.0087 and Q.centroid_offset > 0.021:
        z += 10700.0 * (0.0087 - Q.lam1) * (Q.centroid_offset - 0.021)
    if Q.lam1 < 0.0084 and Q.lam2 < 0.0011:
        z += -338000.0 * (0.0084 - Q.lam1) * (0.0011 - Q.lam2)
    if Q.lam1 < 0.0093 and Q.n_pt_above_50 > 6.1:
        z += -42.7 * (0.0093 - Q.lam1) * (Q.n_pt_above_50 - 6.1)
    if Q.log_sum_pt > 6.6 and Q.dr_max_012 < 0.12:
        z += 33.0 * (Q.log_sum_pt - 6.6) * (0.12 - Q.dr_max_012)
    if Q.log_sum_pt > 6.6 and Q.sum_z_dr2_top3 < 0.0066:
        z += -1280.0 * (Q.log_sum_pt - 6.6) * (0.0066 - Q.sum_z_dr2_top3)
    if Q.pt_7 > 34.0 and Q.n_dr_0_0p05 < 5.9:
        z += 0.00886 * (Q.pt_7 - 34.0) * (5.9 - Q.n_dr_0_0p05)
    if Q.sj2_dr < 0.15 and Q.D2_b2 < 0.73:
        z += 20.4 * (0.15 - Q.sj2_dr) * (0.73 - Q.D2_b2)
    if Q.sum_pt > 990.0 and Q.sum_z_dr2_top2 < 0.0086:
        z += -1.31 * (Q.sum_pt - 990.0) * (0.0086 - Q.sum_z_dr2_top2)
    if Q.z_6 < 0.046 and Q.z_dr_0p2_0p4 < 0.1:
        z += 770.0 * (0.046 - Q.z_6) * (0.1 - Q.z_dr_0p2_0p4)
    if Q.z_7 < 0.058 and Q.D2_b2 < 0.76:
        z += -48.8 * (0.058 - Q.z_7) * (0.76 - Q.D2_b2)
    if Q.z_7 < 0.064 and Q.e3 < 0.0005:
        z += 151000.0 * (0.064 - Q.z_7) * (0.0005 - Q.e3)
    return max(0.0, z)


def neuron_2(Q):
    z = 7.66
    if Q.C2 < 0.04:
        z += 12.9 * Q.C2 - 0.516
    if Q.LHA >= 0.11:
        z += -12.3 * Q.LHA + 1.353
    if Q.e2 < 0.045:
        z += 22.5 * Q.e2 - 1.0125
    if Q.lam1 < 0.012:
        z += -112.0 * Q.lam1 + 1.344
    if Q.log_sum_pt >= 6.1:
        z += -7.12 * Q.log_sum_pt + 43.432
    if Q.mean_phi2 < 8.7e-05:
        z += -44100.0 * Q.mean_phi2 + 3.8367
    if Q.pt_7 < 42.0:
        z += 0.1742 * Q.pt_7 - 7.9548
    if 42.0 <= Q.pt_7 < 54.0:
        z += 0.1642 * Q.pt_7 - 7.5348
    if Q.pt_7 >= 54.0:
        z += 0.111 * Q.pt_7 - 4.662
    if Q.sum_pt < 650.0:
        z += -0.00985 * Q.sum_pt + 6.4025
    if Q.sum_pt_top5 < 320.0:
        z += -0.0266 * Q.sum_pt_top5 + 8.512
    if Q.z_7 < 0.052:
        z += -99.9 * Q.z_7 + 5.2947
    if 0.052 <= Q.z_7 < 0.053:
        z += -190.2 * Q.z_7 + 9.9903
    if Q.z_7 >= 0.053:
        z += -90.3 * Q.z_7 + 4.6956
    if Q.centroid_offset < 0.0055 and Q.dr_7 < 0.037:
        z += -4630.0 * (0.0055 - Q.centroid_offset) * (0.037 - Q.dr_7)
    if Q.lam1 < 0.0072 and Q.max_dr < 0.25:
        z += 1180.0 * (0.0072 - Q.lam1) * (0.25 - Q.max_dr)
    if Q.lam1 < 0.0072 and Q.pt_6 < 62.0:
        z += -3.51 * (0.0072 - Q.lam1) * (62.0 - Q.pt_6)
    if Q.log_sum_pt > 6.9 and Q.D2_b2 < 4.6:
        z += -3.92 * (Q.log_sum_pt - 6.9) * (4.6 - Q.D2_b2)
    if Q.log_sum_pt > 6.9 and Q.absphi_0 < 0.11:
        z += 505.0 * (Q.log_sum_pt - 6.9) * (0.11 - Q.absphi_0)
    if Q.log_sum_pt > 6.1 and Q.mean_phi2 < 8.7e-05:
        z += -73400.0 * (Q.log_sum_pt - 6.1) * (8.7e-05 - Q.mean_phi2)
    if Q.log_sum_pt < 6.7 and Q.mean_phi2 < 8.7e-05:
        z += -69000.0 * (6.7 - Q.log_sum_pt) * (8.7e-05 - Q.mean_phi2)
    if Q.log_sum_pt > 6.9 and Q.n_pt_above_50 > 5.1:
        z += -2.77 * (Q.log_sum_pt - 6.9) * (Q.n_pt_above_50 - 5.1)
    if Q.sum_pt > 860.0 and Q.D2_b2 < 4.8:
        z += 0.00214 * (Q.sum_pt - 860.0) * (4.8 - Q.D2_b2)
    if Q.sum_pt > 990.0 and Q.absphi_0 < 0.1:
        z += -0.3 * (Q.sum_pt - 990.0) * (0.1 - Q.absphi_0)
    if Q.sum_pt > 940.0 and Q.mean_phi2 < 8.3e-05:
        z += 116.0 * (Q.sum_pt - 940.0) * (8.3e-05 - Q.mean_phi2)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.971
    if 0.29 <= Q.LHA < 0.33:
        z += 36.9 * Q.LHA - 10.701
    if Q.LHA >= 0.33:
        z += -10.0 * Q.LHA + 4.776
    if Q.sum_z_dr >= 0.13:
        z += 73.4 * Q.sum_z_dr - 9.542
    if Q.lam1 >= 0.012:
        z += -340.0 * Q.lam1 + 4.08
    if Q.lam2 < 0.0036:
        z += 1120.0 * Q.lam2 - 4.032
    if Q.max_dr >= 0.25:
        z += -10.6 * Q.max_dr + 2.65
    if Q.mean_eta < -0.007:
        z += -37.9 * Q.mean_eta - 0.2653
    if Q.sd_rg >= 0.29:
        z += -20.9 * Q.sd_rg + 6.061
    if Q.sj2_dr >= 0.17:
        z += 22.5 * Q.sj2_dr - 3.825
    if Q.sj3_dr_max >= 0.2:
        z += 11.6 * Q.sj3_dr_max - 2.32
    if Q.sum_z_dr > 0.065 and Q.eccentricity > 0.68:
        z += 124.0 * (Q.sum_z_dr - 0.065) * (Q.eccentricity - 0.68)
    if Q.mean_eta < -0.011 and Q.eta_0 < -0.026:
        z += -447.0 * (-0.011 - Q.mean_eta) * (-0.026 - Q.eta_0)
    if Q.sd_rg > 0.17 and Q.lam2 < 0.00057:
        z += 10900.0 * (Q.sd_rg - 0.17) * (0.00057 - Q.lam2)
    if Q.sj2_dr > 0.16 and Q.C2 < 0.1:
        z += 239.0 * (Q.sj2_dr - 0.16) * (0.1 - Q.C2)
    if Q.sj2_dr > 0.17 and Q.eccentricity > 0.95:
        z += 404.0 * (Q.sj2_dr - 0.17) * (Q.eccentricity - 0.95)
    if Q.sj2_dr > 0.17 and Q.log_sum_pt > 6.0:
        z += -21.7 * (Q.sj2_dr - 0.17) * (Q.log_sum_pt - 6.0)
    if Q.sj2_dr > 0.17 and Q.n_dr_0_0p05 > 2.1:
        z += -4.91 * (Q.sj2_dr - 0.17) * (Q.n_dr_0_0p05 - 2.1)
    if Q.sj3_dr_max > 0.21 and Q.n_dr_0p05_0p1 > 3.9:
        z += 9.73 * (Q.sj3_dr_max - 0.21) * (Q.n_dr_0p05_0p1 - 3.9)
    if Q.sj3_dr_max > 0.27 and Q.n_dr_0p05_0p1 > 3.9:
        z += -21.1 * (Q.sj3_dr_max - 0.27) * (Q.n_dr_0p05_0p1 - 3.9)
    return max(0.0, z)


def neuron_4(Q):
    z = 3.85
    if Q.M2 < 0.043:
        z += 64.8 * Q.M2 - 2.7864
    if Q.N2 < 0.22:
        z += -20.7 * Q.N2 + 4.554
    if Q.e3 < 0.00019:
        z += -22500.0 * Q.e3 + 4.275
    if Q.eccentricity >= 0.71:
        z += -11.1 * Q.eccentricity + 7.881
    if Q.lam1 < 0.0067:
        z += 1816.0 * Q.lam1 - 13.3334
    if 0.0067 <= Q.lam1 < 0.0084:
        z += 686.0 * Q.lam1 - 5.7624
    if Q.lam2 < 0.00064:
        z += 4150.0 * Q.lam2 - 2.656
    if Q.log_sum_pt >= 6.3:
        z += 3.77 * Q.log_sum_pt - 23.751
    if Q.pt_7 < 27.0:
        z += 0.109 * Q.pt_7 - 2.943
    if Q.sj3_dr_max < 0.14:
        z += 26.5 * Q.sj3_dr_max - 3.71
    if Q.sum_pt < 750.0:
        z += 0.00494 * Q.sum_pt - 3.705
    if Q.N2 < 0.24 and Q.n_dr_0p1_0p2 > 1.5:
        z += 1.96 * (0.24 - Q.N2) * (Q.n_dr_0p1_0p2 - 1.5)
    if Q.N2 < 0.21 and Q.sj2_zsoft < 0.29:
        z += -92.7 * (0.21 - Q.N2) * (0.29 - Q.sj2_zsoft)
    if Q.dr_0 < 0.082 and Q.centroid_offset > 0.0083:
        z += 1420.0 * (0.082 - Q.dr_0) * (Q.centroid_offset - 0.0083)
    if Q.lam1 < 0.0068 and Q.eccentricity > 0.73:
        z += 3980.0 * (0.0068 - Q.lam1) * (Q.eccentricity - 0.73)
    if Q.lam1 < 0.0073 and Q.lam2 > 0.00059:
        z += 870000.0 * (0.0073 - Q.lam1) * (Q.lam2 - 0.00059)
    if Q.max_dr > 0.12 and Q.C2_b2 < 0.0089:
        z += 3640.0 * (Q.max_dr - 0.12) * (0.0089 - Q.C2_b2)
    if Q.sj2_dr > 0.23 and Q.D2_b2 < 0.65:
        z += -51.1 * (Q.sj2_dr - 0.23) * (0.65 - Q.D2_b2)
    if Q.sj2_dr > 0.19 and Q.lam2 > 0.0018:
        z += -7380.0 * (Q.sj2_dr - 0.19) * (Q.lam2 - 0.0018)
    if Q.sj3_dr_max > 0.23 and Q.log_sum_pt > 6.3:
        z += -89.8 * (Q.sj3_dr_max - 0.23) * (Q.log_sum_pt - 6.3)
    if Q.sj3_dr_max > 0.2 and Q.sj2_zsoft < 0.06:
        z += 456.0 * (Q.sj3_dr_max - 0.2) * (0.06 - Q.sj2_zsoft)
    if Q.sj3_dr_min < 0.19 and Q.mean_phi < -0.023:
        z += 230.0 * (0.19 - Q.sj3_dr_min) * (-0.023 - Q.mean_phi)
    if Q.sum_pt < 800.0 and Q.M3 < 0.1:
        z += -0.0808 * (800.0 - Q.sum_pt) * (0.1 - Q.M3)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.55
    if Q.LHA < 0.21:
        z += -30.4 * Q.LHA + 6.384
    if Q.centroid_offset < 0.038:
        z += 27.9 * Q.centroid_offset - 1.0602
    if Q.dr_0 < 0.063:
        z += 36.7 * Q.dr_0 - 2.3121
    if Q.lam1 < 7.2e-05:
        z += -28795.0 * Q.lam1 + 3.765
    if 7.2e-05 <= Q.lam1 < 0.0022:
        z += -795.0 * Q.lam1 + 1.749
    if Q.pt_7 >= 39.0:
        z += -0.0636 * Q.pt_7 + 2.4804
    if Q.sum_pt < 740.0:
        z += 0.0151 * Q.sum_pt - 11.174
    if Q.sum_pt_top5 < 580.0:
        z += -0.0173 * Q.sum_pt_top5 + 10.034
    if Q.tau1 < 0.025:
        z += 75.5 * Q.tau1 - 1.8875
    if Q.z_6 < 0.022:
        z += -128.0 * Q.z_6 + 2.816
    if Q.z_7 < 0.033:
        z += -74.5 * Q.z_7 + 2.4585
    if Q.z_top5 < 0.82:
        z += 13.6 * Q.z_top5 - 11.152
    if Q.LHA < 0.22 and Q.log_sum_pt < 6.8:
        z += -110.0 * (0.22 - Q.LHA) * (6.8 - Q.log_sum_pt)
    if Q.LHA < 0.21 and Q.z_6 < 0.096:
        z += -383.0 * (0.21 - Q.LHA) * (0.096 - Q.z_6)
    if Q.e2 < 0.038 and Q.lam2 < 0.00028:
        z += 204000.0 * (0.038 - Q.e2) * (0.00028 - Q.lam2)
    if Q.e2 < 0.038 and Q.z_7 < 0.075:
        z += 967.0 * (0.038 - Q.e2) * (0.075 - Q.z_7)
    if Q.sum_z_dr < 0.022 and Q.sum_pt_top5 > 720.0:
        z += 0.776 * (0.022 - Q.sum_z_dr) * (Q.sum_pt_top5 - 720.0)
    if Q.lam1 < 0.0023 and Q.centroid_offset < 0.021:
        z += 54500.0 * (0.0023 - Q.lam1) * (0.021 - Q.centroid_offset)
    if Q.lam1 < 8.5e-05 and Q.pt_5 < 73.0:
        z += -516.0 * (8.5e-05 - Q.lam1) * (73.0 - Q.pt_5)
    if Q.log_sum_pt > 6.8 and Q.e3 < 6e-05:
        z += -341000.0 * (Q.log_sum_pt - 6.8) * (6e-05 - Q.e3)
    if Q.z_6 < 0.061 and Q.pt1_dr01 < 29.0:
        z += 1.27 * (0.061 - Q.z_6) * (29.0 - Q.pt1_dr01)
    return max(0.0, z)


def neuron_6(Q):
    z = 2.18
    if Q.C2 < 0.036:
        z += -98.0 * Q.C2 + 3.528
    if 0.26 <= Q.LHA < 0.33:
        z += 30.8 * Q.LHA - 8.008
    if Q.LHA >= 0.33:
        z += 76.2 * Q.LHA - 22.99
    if Q.centroid_offset < 0.021:
        z += 108.0 * Q.centroid_offset - 2.268
    if 0.088 <= Q.sum_z_dr < 0.15:
        z += -270.0 * Q.sum_z_dr + 23.76
    if Q.sum_z_dr >= 0.15:
        z += -138.0 * Q.sum_z_dr + 3.96
    if Q.lam2 < 5.8e-05:
        z += 19320.0 * Q.lam2 - 2.9292
    if 5.8e-05 <= Q.lam2 < 0.001:
        z += 1920.0 * Q.lam2 - 1.92
    if Q.max_dr < 0.18:
        z += 31.0 * Q.max_dr - 5.58
    if Q.psi_0p2 >= 0.9:
        z += -7.68 * Q.psi_0p2 + 6.912
    if Q.pt_6 < 20.0:
        z += -0.7548 * Q.pt_6 + 15.6792
    if 20.0 <= Q.pt_6 < 32.0:
        z += -0.0948 * Q.pt_6 + 2.4792
    if 32.0 <= Q.pt_6 < 39.0:
        z += 0.0792 * Q.pt_6 - 3.0888
    if 0.032 <= Q.sj3_dr_max < 0.18:
        z += -17.1 * Q.sj3_dr_max + 0.5472
    if Q.sj3_dr_max >= 0.18:
        z += 10.6 * Q.sj3_dr_max - 4.4388
    if Q.sum_pt >= 850.0:
        z += 0.0108 * Q.sum_pt - 9.18
    if Q.tau1 >= 0.17:
        z += -82.3 * Q.tau1 + 13.991
    if Q.z_6 < 0.022:
        z += 599.0 * Q.z_6 - 13.178
    if Q.centroid_offset > 0.015 and Q.D3 > 0.72:
        z += -10.7 * (Q.centroid_offset - 0.015) * (Q.D3 - 0.72)
    if Q.centroid_offset > 0.0096 and Q.N2 > 0.16:
        z += 210.0 * (Q.centroid_offset - 0.0096) * (Q.N2 - 0.16)
    if Q.centroid_offset > 0.01 and Q.mean_phi2 < 0.015:
        z += 3390.0 * (Q.centroid_offset - 0.01) * (0.015 - Q.mean_phi2)
    if Q.centroid_offset < 0.02 and Q.n_dr_0p05_0p1 > -0.52:
        z += 20.7 * (0.02 - Q.centroid_offset) * (Q.n_dr_0p05_0p1 - -0.52)
    if Q.centroid_offset < 0.019 and Q.planar_flow < 0.13:
        z += -573.0 * (0.019 - Q.centroid_offset) * (0.13 - Q.planar_flow)
    if Q.centroid_offset > 0.027 and Q.pt_2 > 74.0:
        z += 1.7 * (Q.centroid_offset - 0.027) * (Q.pt_2 - 74.0)
    if Q.centroid_offset > 0.0073 and Q.sj3_dr_min < 0.21:
        z += 315.0 * (Q.centroid_offset - 0.0073) * (0.21 - Q.sj3_dr_min)
    if Q.lam2 < 0.0038 and Q.n_dr_0p1_0p2 > 0.55:
        z += 67.9 * (0.0038 - Q.lam2) * (Q.n_dr_0p1_0p2 - 0.55)
    if Q.log_sum_pt < 6.5 and Q.pt_7 < 45.0:
        z += 0.189 * (6.5 - Q.log_sum_pt) * (45.0 - Q.pt_7)
    if Q.pt_6 < 39.0 and Q.D2_b2 < 4.7:
        z += -0.0184 * (39.0 - Q.pt_6) * (4.7 - Q.D2_b2)
    if Q.pt_6 < 39.0 and Q.sum_pt_top3 < 720.0:
        z += 0.000582 * (39.0 - Q.pt_6) * (720.0 - Q.sum_pt_top3)
    if Q.sj2_dr > 0.17 and Q.mean_eta > 0.027:
        z += -494.0 * (Q.sj2_dr - 0.17) * (Q.mean_eta - 0.027)
    if Q.sj2_dr > 0.16 and Q.sj2_zsoft > 0.054:
        z += 119.0 * (Q.sj2_dr - 0.16) * (Q.sj2_zsoft - 0.054)
    if Q.sj3_dr_max > 0.17 and Q.D2_b2 < 1.6:
        z += 12.9 * (Q.sj3_dr_max - 0.17) * (1.6 - Q.D2_b2)
    if Q.sj3_dr_max > 0.043 and Q.dr_min_012 < 0.015:
        z += -179.0 * (Q.sj3_dr_max - 0.043) * (0.015 - Q.dr_min_012)
    if Q.sum_pt > 770.0 and Q.D2_b2 > 0.74:
        z += -0.00061 * (Q.sum_pt - 770.0) * (Q.D2_b2 - 0.74)
    return max(0.0, z)


def neuron_7(Q):
    z = 3.6
    if Q.D2 < 1.5:
        z += 0.987 * Q.D2 - 1.4805
    if Q.LHA < 0.35:
        z += 11.0 * Q.LHA - 4.62
    if 0.35 <= Q.LHA < 0.42:
        z += -361.0 * Q.LHA + 125.58
    if Q.LHA >= 0.42:
        z += -372.0 * Q.LHA + 130.2
    if Q.N2 >= 0.13:
        z += 4.36 * Q.N2 - 0.5668
    if Q.e3 < 1e-05:
        z += -112000.0 * Q.e3 + 1.12
    if Q.sum_z_dr < 0.034:
        z += 74.8 * Q.sum_z_dr - 2.5432
    if 0.087 <= Q.sum_z_dr < 0.1:
        z += -248.0 * Q.sum_z_dr + 21.576
    if Q.sum_z_dr >= 0.1:
        z += 150.0 * Q.sum_z_dr - 18.224
    if Q.lam1 < 0.0035:
        z += 1394.0 * Q.lam1 - 6.0346
    if 0.0035 <= Q.lam1 < 0.0047:
        z += 963.0 * Q.lam1 - 4.5261
    if 0.0073 <= Q.lam1 < 0.012:
        z += -609.0 * Q.lam1 + 4.4457
    if 0.012 <= Q.lam1 < 0.016:
        z += 90.0 * Q.lam1 - 3.9423
    if Q.lam1 >= 0.016:
        z += -1100.0 * Q.lam1 + 15.0977
    if Q.max_dr < 0.046:
        z += 25.0 * Q.max_dr - 1.15
    if Q.sj3_dr_max < 0.073:
        z += -25.6 * Q.sj3_dr_max + 4.096
    if 0.073 <= Q.sj3_dr_max < 0.16:
        z += -16.85 * Q.sj3_dr_max + 3.45725
    if Q.sj3_dr_max >= 0.16:
        z += 8.75 * Q.sj3_dr_max - 0.63875
    if Q.tau1 < 0.062:
        z += -30.4 * Q.tau1 + 2.6448
    if 0.062 <= Q.tau1 < 0.087:
        z += -14.2 * Q.tau1 + 1.6404
    if 0.087 <= Q.tau1 < 0.1:
        z += 16.2 * Q.tau1 - 1.0044
    if Q.tau1 >= 0.1:
        z += 122.2 * Q.tau1 - 11.6044
    if Q.centroid_offset < 0.025 and Q.n_dr_0p2_0p4 < 1.0:
        z += -18.6 * (0.025 - Q.centroid_offset) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.e3 < 7.2e-05 and Q.pt_7 < 45.0:
        z += -352.0 * (7.2e-05 - Q.e3) * (45.0 - Q.pt_7)
    if Q.sum_z_dr > 0.045 and Q.D2 < 2.2:
        z += 35.6 * (Q.sum_z_dr - 0.045) * (2.2 - Q.D2)
    if Q.sum_z_dr > 0.081 and Q.D2 < 2.8:
        z += -38.1 * (Q.sum_z_dr - 0.081) * (2.8 - Q.D2)
    if Q.sum_z_dr > 0.084 and Q.sj2_zsoft < 0.088:
        z += 9370.0 * (Q.sum_z_dr - 0.084) * (0.088 - Q.sj2_zsoft)
    if Q.lam1 > 0.0024 and Q.D2 > 0.26:
        z += -333.0 * (Q.lam1 - 0.0024) * (Q.D2 - 0.26)
    if Q.lam1 > 0.017 and Q.D2 < 3.9:
        z += -243.0 * (Q.lam1 - 0.017) * (3.9 - Q.D2)
    if Q.lam1 > 0.0025 and Q.planar_flow < 0.2:
        z += 2250.0 * (Q.lam1 - 0.0025) * (0.2 - Q.planar_flow)
    if Q.lam1 > 0.0077 and Q.pt_4 < 93.0:
        z += 6.26 * (Q.lam1 - 0.0077) * (93.0 - Q.pt_4)
    if Q.lam1 > 0.0025 and Q.sj3_dr_max < 0.19:
        z += -12500.0 * (Q.lam1 - 0.0025) * (0.19 - Q.sj3_dr_max)
    if Q.planar_flow < 0.18 and Q.lam1 > 0.0083:
        z += -5720.0 * (0.18 - Q.planar_flow) * (Q.lam1 - 0.0083)
    if Q.planar_flow < 0.2 and Q.log_sum_pt > 6.5:
        z += 31.3 * (0.2 - Q.planar_flow) * (Q.log_sum_pt - 6.5)
    if Q.planar_flow < 0.21 and Q.max_dr < 0.24:
        z += -36.8 * (0.21 - Q.planar_flow) * (0.24 - Q.max_dr)
    if Q.planar_flow < 0.24 and Q.pt_6 < 56.0:
        z += -0.0935 * (0.24 - Q.planar_flow) * (56.0 - Q.pt_6)
    if Q.planar_flow < 0.21 and Q.pt_7 < 29.0:
        z += -0.384 * (0.21 - Q.planar_flow) * (29.0 - Q.pt_7)
    if Q.sj3_dr_max < 0.13 and Q.D2 < 1.4:
        z += 19.6 * (0.13 - Q.sj3_dr_max) * (1.4 - Q.D2)
    return max(0.0, z)


def neuron_8(Q):
    z = -1.07
    if Q.centroid_offset >= 0.0074:
        z += -61.8 * Q.centroid_offset + 0.45732
    if Q.lam2 < 0.0002:
        z += -10100.0 * Q.lam2 + 2.02
    if Q.log_sum_pt >= 6.7:
        z += -8.7 * Q.log_sum_pt + 58.29
    if Q.sj3_dr_max < 0.11:
        z += 23.5 * Q.sj3_dr_max - 1.271
    if 0.11 <= Q.sj3_dr_max < 0.2:
        z += -14.6 * Q.sj3_dr_max + 2.92
    if Q.tau1 < 0.027:
        z += 67.6 * Q.tau1 - 1.8252
    if Q.tau2 < 0.017:
        z += 87.5 * Q.tau2 - 1.4875
    if Q.z_6 < 0.044:
        z += -42.9 * Q.z_6 + 1.8876
    if Q.LHA < 0.19 and Q.z_dr_0p2_0p4 < 0.053:
        z += -732.0 * (0.19 - Q.LHA) * (0.053 - Q.z_dr_0p2_0p4)
    if Q.lam1 < 0.0069 and Q.centroid_offset < 0.025:
        z += 9120.0 * (0.0069 - Q.lam1) * (0.025 - Q.centroid_offset)
    if Q.lam1 < 0.0049 and Q.log_sum_pt < 6.5:
        z += -946.0 * (0.0049 - Q.lam1) * (6.5 - Q.log_sum_pt)
    if Q.lam1 < 0.0044 and Q.pt_7 < 39.0:
        z += -13.8 * (0.0044 - Q.lam1) * (39.0 - Q.pt_7)
    if Q.lam1 < 0.0055 and Q.z_dr_0p2_0p4 < 0.058:
        z += 8230.0 * (0.0055 - Q.lam1) * (0.058 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt > 6.7 and Q.sum_z_dr2_top5 < 0.0029:
        z += 2110.0 * (Q.log_sum_pt - 6.7) * (0.0029 - Q.sum_z_dr2_top5)
    if Q.sj3_dr_max < 0.11 and Q.centroid_offset > 0.017:
        z += -3900.0 * (0.11 - Q.sj3_dr_max) * (Q.centroid_offset - 0.017)
    if Q.sj3_dr_max < 0.1 and Q.sum_z_dr2_top2 < 0.0017:
        z += 23800.0 * (0.1 - Q.sj3_dr_max) * (0.0017 - Q.sum_z_dr2_top2)
    if Q.sj3_dr_max < 0.16 and Q.sum_z_dr2_top2 < 0.0016:
        z += -12400.0 * (0.16 - Q.sj3_dr_max) * (0.0016 - Q.sum_z_dr2_top2)
    if Q.tau1 < 0.026 and Q.centroid_offset < 0.032:
        z += 5220.0 * (0.026 - Q.tau1) * (0.032 - Q.centroid_offset)
    return max(0.0, z)


def neuron_9(Q):
    z = -5.29
    if Q.centroid_offset < 0.0024:
        z += 485.0 * Q.centroid_offset + 1.553
    if 0.0024 <= Q.centroid_offset < 0.031:
        z += -95.0 * Q.centroid_offset + 2.945
    if Q.e3 >= 0.00017:
        z += 3110.0 * Q.e3 - 0.5287
    if Q.sum_z_dr < 0.027:
        z += 306.9 * Q.sum_z_dr - 12.1956
    if 0.027 <= Q.sum_z_dr < 0.042:
        z += 199.9 * Q.sum_z_dr - 9.3066
    if 0.042 <= Q.sum_z_dr < 0.054:
        z += 75.9 * Q.sum_z_dr - 4.0986
    if Q.lam1 < 0.00014:
        z += -9009.0 * Q.lam1 + 3.7356
    if 0.00014 <= Q.lam1 < 0.0034:
        z += -759.0 * Q.lam1 + 2.5806
    if Q.lam1 >= 0.014:
        z += 89.4 * Q.lam1 - 1.2516
    if Q.lam2 >= 0.0004:
        z += 392.0 * Q.lam2 - 0.1568
    if Q.sj3_dr_max < 0.14:
        z += 11.3 * Q.sj3_dr_max - 0.462
    if 0.14 <= Q.sj3_dr_max < 0.21:
        z += -16.0 * Q.sj3_dr_max + 3.36
    if Q.sum_pt < 990.0:
        z += -0.00574 * Q.sum_pt + 5.6826
    if Q.centroid_offset < 0.019 and Q.D2_b2 < 0.17:
        z += -563.0 * (0.019 - Q.centroid_offset) * (0.17 - Q.D2_b2)
    if Q.centroid_offset < 0.025 and Q.lam1 < 0.0069:
        z += 15100.0 * (0.025 - Q.centroid_offset) * (0.0069 - Q.lam1)
    if Q.centroid_offset < 0.017 and Q.sj3_dr_min < 0.13:
        z += 1800.0 * (0.017 - Q.centroid_offset) * (0.13 - Q.sj3_dr_min)
    if Q.e3 < 4.6e-06 and Q.n_dr_0p05_0p1 > 2.9:
        z += -429000.0 * (4.6e-06 - Q.e3) * (Q.n_dr_0p05_0p1 - 2.9)
    if Q.sum_z_dr < 0.065 and Q.mean_phi < -9e-05:
        z += -1770.0 * (0.065 - Q.sum_z_dr) * (-9e-05 - Q.mean_phi)
    if Q.lam1 < 0.006 and Q.lam2 < 0.0011:
        z += 1500000.0 * (0.006 - Q.lam1) * (0.0011 - Q.lam2)
    if Q.lam1 < 0.0059 and Q.mean_eta < -0.0066:
        z += -24000.0 * (0.0059 - Q.lam1) * (-0.0066 - Q.mean_eta)
    if Q.lam1 < 0.0059 and Q.mean_eta > 0.0092:
        z += -26400.0 * (0.0059 - Q.lam1) * (Q.mean_eta - 0.0092)
    if Q.lam1 < 0.0061 and Q.mean_phi > 0.013:
        z += -29700.0 * (0.0061 - Q.lam1) * (Q.mean_phi - 0.013)
    if Q.lam1 < 0.0059 and Q.sj3_dr_min < 0.087:
        z += -5040.0 * (0.0059 - Q.lam1) * (0.087 - Q.sj3_dr_min)
    if Q.log_sum_pt < 6.9 and Q.pt_3 < 39.0:
        z += 0.165 * (6.9 - Q.log_sum_pt) * (39.0 - Q.pt_3)
    if Q.sum_pt < 940.0 and Q.centroid_offset < 0.017:
        z += -0.387 * (940.0 - Q.sum_pt) * (0.017 - Q.centroid_offset)
    if Q.sum_pt < 820.0 and Q.e4 < 2.6e-08:
        z += 257000.0 * (820.0 - Q.sum_pt) * (2.6e-08 - Q.e4)
    if Q.sum_pt < 1000.0 and Q.z_5 < 0.057:
        z += 0.258 * (1000.0 - Q.sum_pt) * (0.057 - Q.z_5)
    if Q.tau1 < 0.044 and Q.centroid_offset < 0.032:
        z += 5180.0 * (0.044 - Q.tau1) * (0.032 - Q.centroid_offset)
    return max(0.0, z)


def neuron_10(Q):
    z = 7.84
    if Q.D2_b2 < 0.26:
        z += -2.33 * Q.D2_b2 + 0.6058
    if Q.LHA >= 0.2:
        z += -12.5 * Q.LHA + 2.5
    if Q.centroid_offset >= 0.022:
        z += 18.2 * Q.centroid_offset - 0.4004
    if Q.lam1 < 0.016:
        z += 377.0 * Q.lam1 - 6.032
    if Q.lam1 >= 0.016:
        z += 354.0 * Q.lam1 - 5.664
    if Q.lam2 >= 0.00016:
        z += 532.0 * Q.lam2 - 0.08512
    if Q.log_sum_pt >= 6.7:
        z += -4.68 * Q.log_sum_pt + 31.356
    if Q.n_dr_0p2_0p4 >= 0.99:
        z += 0.679 * Q.n_dr_0p2_0p4 - 0.67221
    if 0.13 <= Q.sj2_dr < 0.16:
        z += 15.7 * Q.sj2_dr - 2.041
    if 0.16 <= Q.sj2_dr < 0.3:
        z += 0.3 * Q.sj2_dr + 0.423
    if Q.sj2_dr >= 0.3:
        z += 9.74 * Q.sj2_dr - 2.409
    if Q.sum_pt >= 990.0:
        z += 0.00706 * Q.sum_pt - 6.9894
    if Q.z_7 < 0.068:
        z += 22.7 * Q.z_7 - 1.5436
    if Q.LHA > 0.32 and Q.planar_flow < 0.82:
        z += -22.5 * (Q.LHA - 0.32) * (0.82 - Q.planar_flow)
    if Q.lam1 < 0.0036 and Q.dr_7 < 0.27:
        z += -2040.0 * (0.0036 - Q.lam1) * (0.27 - Q.dr_7)
    if Q.lam1 > 0.0058 and Q.sj3_pairmin_over_m < 0.51:
        z += -291.0 * (Q.lam1 - 0.0058) * (0.51 - Q.sj3_pairmin_over_m)
    if Q.lam2 > -2.1e-05 and Q.log_sum_pt < 6.5:
        z += -599.0 * (Q.lam2 - -2.1e-05) * (6.5 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.D2_b2 < 4.6:
        z += -0.145 * (Q.n_dr_0p2_0p4 - 1.0) * (4.6 - Q.D2_b2)
    if Q.pt_6 < 48.0 and Q.log_sum_pt < 6.5:
        z += -0.0696 * (48.0 - Q.pt_6) * (6.5 - Q.log_sum_pt)
    if Q.pt_7 > 20.0 and Q.D2_b2 < 0.28:
        z += -0.136 * (Q.pt_7 - 20.0) * (0.28 - Q.D2_b2)
    if Q.sj2_dr > 0.17 and Q.planar_flow < 0.7:
        z += -24.9 * (Q.sj2_dr - 0.17) * (0.7 - Q.planar_flow)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.0851
    if Q.centroid_offset < 0.018:
        z += -75.7 * Q.centroid_offset + 1.3626
    if Q.centroid_offset >= 0.024:
        z += -43.6 * Q.centroid_offset + 1.0464
    if Q.e3 < 7.6e-06:
        z += 141000.0 * Q.e3 - 1.0716
    if Q.sum_z_dr < 0.06:
        z += 102.8 * Q.sum_z_dr - 6.9108
    if 0.06 <= Q.sum_z_dr < 0.072:
        z += 61.9 * Q.sum_z_dr - 4.4568
    if Q.lam1 < 0.012:
        z += -457.0 * Q.lam1 + 5.484
    if Q.mean_phi2 < 0.0021:
        z += 195.0 * Q.mean_phi2 - 0.4095
    if Q.planar_flow < 0.26:
        z += -11.2 * Q.planar_flow + 2.912
    if Q.pt_7 < 46.0:
        z += 0.041 * Q.pt_7 - 1.886
    if 0.12 <= Q.sj2_dr < 0.27:
        z += 6.01 * Q.sj2_dr - 0.7212
    if Q.sj2_dr >= 0.27:
        z += 18.71 * Q.sj2_dr - 4.1502
    if 0.18 <= Q.sj3_dr_max < 0.26:
        z += -23.9 * Q.sj3_dr_max + 4.302
    if Q.sj3_dr_max >= 0.26:
        z += -14.17 * Q.sj3_dr_max + 1.7722
    if Q.tau1 < 0.072:
        z += -36.0 * Q.tau1 + 2.592
    if Q.tau21 < 0.13:
        z += 8.46 * Q.tau21 - 1.0998
    if Q.centroid_offset < 0.018 and Q.sum_pt < 940.0:
        z += -0.159 * (0.018 - Q.centroid_offset) * (940.0 - Q.sum_pt)
    if Q.e3 < 7.6e-06 and Q.D2_b2 < 0.78:
        z += 338000.0 * (7.6e-06 - Q.e3) * (0.78 - Q.D2_b2)
    if Q.sum_z_dr < 0.06 and Q.dr1_7 > 0.2:
        z += 885.0 * (0.06 - Q.sum_z_dr) * (Q.dr1_7 - 0.2)
    if Q.lam1 < 0.0056 and Q.D2_b2 < 0.77:
        z += -409.0 * (0.0056 - Q.lam1) * (0.77 - Q.D2_b2)
    if Q.lam1 < 0.0034 and Q.centroid_offset < 0.027:
        z += -33200.0 * (0.0034 - Q.lam1) * (0.027 - Q.centroid_offset)
    if Q.lam1 < 0.012 and Q.centroid_offset > 0.016:
        z += -14300.0 * (0.012 - Q.lam1) * (Q.centroid_offset - 0.016)
    if Q.lam1 < 0.012 and Q.dr1_7 > 0.24:
        z += -4100.0 * (0.012 - Q.lam1) * (Q.dr1_7 - 0.24)
    if Q.lam1 < 0.016 and Q.lam2 < 0.0011:
        z += 136000.0 * (0.016 - Q.lam1) * (0.0011 - Q.lam2)
    if Q.lam1 < 0.0034 and Q.planar_flow < 0.26:
        z += -2830.0 * (0.0034 - Q.lam1) * (0.26 - Q.planar_flow)
    if Q.lam2 < 0.00055 and Q.C3 > 0.044:
        z += -83100.0 * (0.00055 - Q.lam2) * (Q.C3 - 0.044)
    if Q.planar_flow < 0.26 and Q.D3 < 4.0:
        z += -0.818 * (0.26 - Q.planar_flow) * (4.0 - Q.D3)
    if Q.planar_flow < 0.26 and Q.sum_pt < 840.0:
        z += -0.0152 * (0.26 - Q.planar_flow) * (840.0 - Q.sum_pt)
    if Q.tau2 < 0.011 and Q.pt1_dr01 > 22.0:
        z += -9.59 * (0.011 - Q.tau2) * (Q.pt1_dr01 - 22.0)
    if Q.zdr_0 < 0.0049 and Q.dr0_7 > 0.095:
        z += -4300.0 * (0.0049 - Q.zdr_0) * (Q.dr0_7 - 0.095)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.821
    if Q.centroid_offset >= 0.039:
        z += 21.5 * Q.centroid_offset - 0.8385
    if Q.e2 >= 0.067:
        z += -31.7 * Q.e2 + 2.1239
    if 0.13 <= Q.sum_z_dr < 0.15:
        z += 74.3 * Q.sum_z_dr - 9.659
    if Q.sum_z_dr >= 0.15:
        z += 88.5 * Q.sum_z_dr - 11.789
    if Q.n_dr_0p2_0p4 >= 1.8:
        z += 0.39 * Q.n_dr_0p2_0p4 - 0.702
    if Q.sd_rg >= 0.32:
        z += 13.4 * Q.sd_rg - 4.288
    if Q.sum_z_dr > 0.12 and Q.C2_b2 > 0.0089:
        z += 1040.0 * (Q.sum_z_dr - 0.12) * (Q.C2_b2 - 0.0089)
    if Q.sum_z_dr > 0.1 and Q.log_sum_pt > 6.5:
        z += 129.0 * (Q.sum_z_dr - 0.1) * (Q.log_sum_pt - 6.5)
    if Q.sum_z_dr > 0.12 and Q.pt_6 < 61.0:
        z += -0.999 * (Q.sum_z_dr - 0.12) * (61.0 - Q.pt_6)
    if Q.sum_z_dr > 0.15 and Q.sum_pt < 570.0:
        z += -0.2 * (Q.sum_z_dr - 0.15) * (570.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_13(Q):
    z = -0.0719
    if Q.C2_b2 < 0.0087:
        z += 206.0 * Q.C2_b2 - 1.7922
    if Q.sum_z_dr < 0.15:
        z += -84.3 * Q.sum_z_dr + 12.645
    if Q.lam1 < 0.015:
        z += -223.0 * Q.lam1 + 3.345
    if Q.log_sum_pt < 6.5:
        z += 2.61 * Q.log_sum_pt - 16.965
    if Q.pt_5 < 26.0:
        z += 0.203 * Q.pt_5 - 5.278
    if Q.pt_6 < 50.0:
        z += 0.026 * Q.pt_6 - 1.3
    if Q.pt_7 >= 32.0:
        z += -0.0923 * Q.pt_7 + 2.9536
    if Q.sj3_dr_max >= 0.23:
        z += -11.8 * Q.sj3_dr_max + 2.714
    if Q.tau1 < 0.12:
        z += 27.0 * Q.tau1 - 3.24
    if Q.z_7 >= 0.045:
        z += 61.6 * Q.z_7 - 2.772
    if Q.z_dr_0_0p05 >= 0.61:
        z += -1.4 * Q.z_dr_0_0p05 + 0.854
    if Q.e2 < 0.087 and Q.pt_4 < 60.0:
        z += 0.288 * (0.087 - Q.e2) * (60.0 - Q.pt_4)
    if Q.e3 < 8.5e-05 and Q.centroid_offset < 0.036:
        z += -649000.0 * (8.5e-05 - Q.e3) * (0.036 - Q.centroid_offset)
    if Q.sum_z_dr < 0.16 and Q.lam2 < 0.00025:
        z += -12100.0 * (0.16 - Q.sum_z_dr) * (0.00025 - Q.lam2)
    if Q.sum_z_dr < 0.14 and Q.log_sum_pt < 6.8:
        z += -76.3 * (0.14 - Q.sum_z_dr) * (6.8 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.14 and Q.pt_7 < 43.0:
        z += -0.627 * (0.14 - Q.sum_z_dr) * (43.0 - Q.pt_7)
    if Q.lam1 < 0.02 and Q.pt_7 < 26.0:
        z += -12.5 * (0.02 - Q.lam1) * (26.0 - Q.pt_7)
    if Q.log_sum_pt > 6.7 and Q.C2_b2 > -0.0047:
        z += -485.0 * (Q.log_sum_pt - 6.7) * (Q.C2_b2 - -0.0047)
    if Q.sum_pt_top5 > 690.0 and Q.pt_7 < 40.0:
        z += 0.000752 * (Q.sum_pt_top5 - 690.0) * (40.0 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = -4.73
    if Q.centroid_offset >= 0.05:
        z += -1320.0 * Q.centroid_offset + 66.0
    if Q.e2 < 0.013:
        z += -4.6 * Q.e2 + 1.9582
    if 0.013 <= Q.e2 < 0.041:
        z += -67.8 * Q.e2 + 2.7798
    if Q.e3 < 8.4e-05:
        z += 9840.0 * Q.e3 - 0.82656
    if 0.026 <= Q.sum_z_dr < 0.068:
        z += 86.2 * Q.sum_z_dr - 2.2412
    if 0.068 <= Q.sum_z_dr < 0.087:
        z += 132.5 * Q.sum_z_dr - 5.3896
    if Q.sum_z_dr >= 0.087:
        z += -78.5 * Q.sum_z_dr + 12.9674
    if Q.sum_z_dr2_top3 < 0.0022:
        z += 234.0 * Q.sum_z_dr2_top3 - 0.5148
    if Q.lam1 < 0.0065:
        z += 616.0 * Q.lam1 - 2.915
    if 0.0065 <= Q.lam1 < 0.012:
        z += -198.0 * Q.lam1 + 2.376
    if Q.n_dr_0p05_0p1 < 5.0:
        z += 0.154 * Q.n_dr_0p05_0p1 - 0.77
    if Q.sd_rg < 0.066:
        z += -0.7 * Q.sd_rg + 1.5646
    if 0.066 <= Q.sd_rg < 0.16:
        z += -14.6 * Q.sd_rg + 2.482
    if 0.16 <= Q.sd_rg < 0.17:
        z += 19.6 * Q.sd_rg - 2.99
    if 0.17 <= Q.sd_rg < 0.19:
        z += 34.2 * Q.sd_rg - 5.472
    if Q.sd_rg >= 0.19:
        z += -16.1 * Q.sd_rg + 4.085
    if Q.tau1 < 0.095:
        z += -40.9 * Q.tau1 + 3.8855
    if Q.e2 < 0.043 and Q.mean_eta < -0.027:
        z += 2290.0 * (0.043 - Q.e2) * (-0.027 - Q.mean_eta)
    if Q.lam1 < 0.016 and Q.D2 < 0.42:
        z += 117.0 * (0.016 - Q.lam1) * (0.42 - Q.D2)
    if Q.lam1 < 0.016 and Q.centroid_offset > 0.027:
        z += -12800.0 * (0.016 - Q.lam1) * (Q.centroid_offset - 0.027)
    if Q.lam1 < 0.0065 and Q.sj3_dr23 > 0.2:
        z += 2780.0 * (0.0065 - Q.lam1) * (Q.sj3_dr23 - 0.2)
    if Q.lam2 < 0.0035 and Q.centroid_offset > 0.027:
        z += 9660.0 * (0.0035 - Q.lam2) * (Q.centroid_offset - 0.027)
    if Q.lam2 < 0.0033 and Q.sd_rg > 0.16:
        z += 7020.0 * (0.0033 - Q.lam2) * (Q.sd_rg - 0.16)
    if Q.lam2 < 0.0034 and Q.z_dr_0p2_0p4 > 0.00043:
        z += -1560.0 * (0.0034 - Q.lam2) * (Q.z_dr_0p2_0p4 - 0.00043)
    if Q.planar_flow < 0.11 and Q.centroid_offset < 0.016:
        z += -269.0 * (0.11 - Q.planar_flow) * (0.016 - Q.centroid_offset)
    if Q.planar_flow < 0.11 and Q.lam1 < 0.006:
        z += 3610.0 * (0.11 - Q.planar_flow) * (0.006 - Q.lam1)
    if Q.planar_flow < 0.11 and Q.lam1 < 0.0073:
        z += -7920.0 * (0.11 - Q.planar_flow) * (0.0073 - Q.lam1)
    if Q.planar_flow < 0.11 and Q.lam1 < 0.016:
        z += 1070.0 * (0.11 - Q.planar_flow) * (0.016 - Q.lam1)
    if Q.planar_flow < 0.11 and Q.sd_nremoved < 1.0:
        z += -5.04 * (0.11 - Q.planar_flow) * (1.0 - Q.sd_nremoved)
    if Q.planar_flow < 0.11 and Q.sj2_zsoft < 0.29:
        z += 38.2 * (0.11 - Q.planar_flow) * (0.29 - Q.sj2_zsoft)
    if Q.z_dr_0p05_0p1 > 0.16 and Q.log_sum_pt < 6.7:
        z += -2.88 * (Q.z_dr_0p05_0p1 - 0.16) * (6.7 - Q.log_sum_pt)
    if Q.z_dr_0p05_0p1 > 0.17 and Q.n_dr_0p2_0p4 < 1.0:
        z += 1.95 * (Q.z_dr_0p05_0p1 - 0.17) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p05_0p1 > 0.16 and Q.psi_0p3 < 1.0:
        z += -1210.0 * (Q.z_dr_0p05_0p1 - 0.16) * (1.0 - Q.psi_0p3)
    if Q.z_dr_0p05_0p1 < 0.59 and Q.z_dr_0p1_0p2 < 0.045:
        z += 21.6 * (0.59 - Q.z_dr_0p05_0p1) * (0.045 - Q.z_dr_0p1_0p2)
    if Q.z_dr_0p05_0p1 > 0.16 and Q.z_dr_0p2_0p4 < 0.21:
        z += -9.37 * (Q.z_dr_0p05_0p1 - 0.16) * (0.21 - Q.z_dr_0p2_0p4)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.704
    if Q.e2 < 0.041:
        z += -97.7 * Q.e2 + 4.0057
    if Q.sum_z_dr < 0.033:
        z += 303.5 * Q.sum_z_dr - 20.4276
    if 0.033 <= Q.sum_z_dr < 0.072:
        z += 233.7 * Q.sum_z_dr - 18.1242
    if 0.072 <= Q.sum_z_dr < 0.086:
        z += 92.7 * Q.sum_z_dr - 7.9722
    if Q.sum_z_dr2_top2 < 0.0076:
        z += -227.0 * Q.sum_z_dr2_top2 + 1.7252
    if Q.sum_z_dr2_top3 < 0.0021:
        z += -1370.0 * Q.sum_z_dr2_top3 + 2.877
    if Q.lam1 < 0.0054:
        z += -110.0 * Q.lam1 + 1.9292
    if 0.0054 <= Q.lam1 < 0.0074:
        z += 398.0 * Q.lam1 - 0.814
    if 0.0074 <= Q.lam1 < 0.017:
        z += -222.0 * Q.lam1 + 3.774
    if 0.18 <= Q.sj3_dr13 < 0.25:
        z += 13.2 * Q.sj3_dr13 - 2.376
    if Q.sj3_dr13 >= 0.25:
        z += -7.9 * Q.sj3_dr13 + 2.899
    if Q.tau1 < 0.058:
        z += -54.9 * Q.tau1 + 3.1842
    if 0.096 <= Q.tau1 < 0.1:
        z += -122.0 * Q.tau1 + 11.712
    if 0.1 <= Q.tau1 < 0.13:
        z += 16.0 * Q.tau1 - 2.088
    if Q.tau1 >= 0.13:
        z += -27.0 * Q.tau1 + 3.502
    if Q.D3 < 0.22 and Q.n_dr_0p1_0p2 > 1.0:
        z += -219.0 * (0.22 - Q.D3) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.N2 < 0.19 and Q.LHA > 0.38:
        z += -684.0 * (0.19 - Q.N2) * (Q.LHA - 0.38)
    if Q.N2 < 0.22 and Q.LHA > 0.29:
        z += 268.0 * (0.22 - Q.N2) * (Q.LHA - 0.29)
    if Q.N2 < 0.22 and Q.LHA > 0.44:
        z += -2060.0 * (0.22 - Q.N2) * (Q.LHA - 0.44)
    if Q.N2 < 0.23 and Q.lam1 > 0.0087:
        z += -4110.0 * (0.23 - Q.N2) * (Q.lam1 - 0.0087)
    if Q.N2 < 0.22 and Q.sj2_dr < 0.19:
        z += -347.0 * (0.22 - Q.N2) * (0.19 - Q.sj2_dr)
    if Q.N2 < 0.24 and Q.z_7 < 0.023:
        z += -1420.0 * (0.24 - Q.N2) * (0.023 - Q.z_7)
    if Q.N2 < 0.22 and Q.z_dr_0p2_0p4 > 0.21:
        z += -213.0 * (0.22 - Q.N2) * (Q.z_dr_0p2_0p4 - 0.21)
    if Q.eccentricity > 0.95 and Q.mean_phi > -0.02:
        z += -482.0 * (Q.eccentricity - 0.95) * (Q.mean_phi - -0.02)
    if Q.eccentricity > 0.94 and Q.sum_pt_top5 > 410.0:
        z += 0.122 * (Q.eccentricity - 0.94) * (Q.sum_pt_top5 - 410.0)
    if Q.sum_z_dr2_top3 < 0.0021 and Q.planar_flow < 0.21:
        z += -8790.0 * (0.0021 - Q.sum_z_dr2_top3) * (0.21 - Q.planar_flow)
    if Q.lam1 < 0.017 and Q.C2_b2 > 0.007:
        z += -8520.0 * (0.017 - Q.lam1) * (Q.C2_b2 - 0.007)
    if Q.lam1 < 0.0065 and Q.D2 < 0.89:
        z += 2090.0 * (0.0065 - Q.lam1) * (0.89 - Q.D2)
    if Q.lam1 < 0.0082 and Q.D2 < 0.89:
        z += -1120.0 * (0.0082 - Q.lam1) * (0.89 - Q.D2)
    if Q.lam1 < 0.0078 and Q.centroid_offset > 0.038:
        z += 60200.0 * (0.0078 - Q.lam1) * (Q.centroid_offset - 0.038)
    if Q.lam1 < 0.018 and Q.centroid_offset > 0.038:
        z += -20100.0 * (0.018 - Q.lam1) * (Q.centroid_offset - 0.038)
    if Q.lam1 < 0.006 and Q.sj3_dr23 > 0.2:
        z += -13500.0 * (0.006 - Q.lam1) * (Q.sj3_dr23 - 0.2)
    if Q.lam1 < 0.0084 and Q.sj3_dr23 > 0.2:
        z += 10900.0 * (0.0084 - Q.lam1) * (Q.sj3_dr23 - 0.2)
    if Q.lam2 < 0.0034 and Q.zdr_1 < 0.019:
        z += -11500.0 * (0.0034 - Q.lam2) * (0.019 - Q.zdr_1)
    if Q.tau1 > 0.13 and Q.D2_b2 < 0.54:
        z += 222.0 * (Q.tau1 - 0.13) * (0.54 - Q.D2_b2)
    if Q.z_dr_0p05_0p1 > 0.74 and Q.sj3_dr23 > 0.18:
        z += -67.6 * (Q.z_dr_0p05_0p1 - 0.74) * (Q.sj3_dr23 - 0.18)
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
