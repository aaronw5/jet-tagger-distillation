"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; no mass observables or exact equivalents), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.6% (the network: 65.8%); same class as the network for 87.4% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.ptdr0_7                pT7 · ΔR(0, 7) [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
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
  Q.girth                  pT-weighted mean ΔR
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_4=pt[4],
        pt_5=pt[5],
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        ptdr0_7=pt[7] * math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        z_top3_slots=sum(pt[:3]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        phi_0=phi[0],
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
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
        girth=sum(z[i] * dr[i] for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -3.31
    if Q.C2 < 0.0363:
        z += 27.4 * Q.C2 - 0.99462
    if Q.LHA >= 0.12:
        z += 16.0 * Q.LHA - 1.92
    if 0.00875 <= Q.centroid_offset < 0.0261:
        z += -49.8 * Q.centroid_offset + 0.43575
    if Q.centroid_offset >= 0.0261:
        z += -75.7 * Q.centroid_offset + 1.11174
    if Q.e3 < 0.000501:
        z += -3660.0 * Q.e3 + 1.83366
    if Q.girth >= 0.0816:
        z += -44.6 * Q.girth + 3.63936
    if Q.lam1 < 0.00391:
        z += 19.0 * Q.lam1 + 2.97239
    if 0.00391 <= Q.lam1 < 0.0121:
        z += -372.0 * Q.lam1 + 4.5012
    z += -686.0 * Q.lam2
    if Q.n_dr_0p2_0p4 >= 0.397:
        z += -0.438 * Q.n_dr_0p2_0p4 + 0.173886
    if Q.planar_flow < 0.106:
        z += -7.58 * Q.planar_flow + 0.80348
    if 0.145 <= Q.sj3_dr_max < 0.17:
        z += 27.8 * Q.sj3_dr_max - 4.031
    if Q.sj3_dr_max >= 0.17:
        z += -1.9 * Q.sj3_dr_max + 1.018
    if Q.sum_pt >= 887.0:
        z += -0.00579 * Q.sum_pt + 5.13573
    if Q.zdr_0 >= 0.0148:
        z += 14.4 * Q.zdr_0 - 0.21312
    if Q.girth2_top3 < 0.00764 and Q.phi_0 < 0.0325:
        z += -998.0 * (0.00764 - Q.girth2_top3) * (0.0325 - Q.phi_0)
    if Q.lam1 < 0.00554 and Q.D2 < 1.33:
        z += -577.0 * (0.00554 - Q.lam1) * (1.33 - Q.D2)
    if Q.lam1 < 0.0172 and Q.centroid_offset > 0.0179:
        z += -4930.0 * (0.0172 - Q.lam1) * (Q.centroid_offset - 0.0179)
    if Q.lam1 < 0.00626 and Q.n_dr_0p2_0p4 < 1.47:
        z += -256.0 * (0.00626 - Q.lam1) * (1.47 - Q.n_dr_0p2_0p4)
    if Q.lam1 < 0.00567 and Q.sum_pt < 725.0:
        z += 2.0 * (0.00567 - Q.lam1) * (725.0 - Q.sum_pt)
    if Q.planar_flow < 0.266 and Q.z_7 < 0.0311:
        z += 250.0 * (0.266 - Q.planar_flow) * (0.0311 - Q.z_7)
    if Q.sum_pt < 875.0 and Q.centroid_offset > 0.0158:
        z += 0.198 * (875.0 - Q.sum_pt) * (Q.centroid_offset - 0.0158)
    if Q.sum_pt < 745.0 and Q.e3 < 0.000468:
        z += -19.8 * (745.0 - Q.sum_pt) * (0.000468 - Q.e3)
    return max(0.0, z)


def neuron_1(Q):
    z = 3.72
    if Q.centroid_offset < 0.0196:
        z += 23.4 * Q.centroid_offset - 0.45864
    if Q.girth < 0.0635:
        z += -26.0 * Q.girth + 0.54175
    if 0.0635 <= Q.girth < 0.107:
        z += 25.5 * Q.girth - 2.7285
    if Q.lam1 < 0.000662:
        z += 849.0 * Q.lam1 - 1.73433
    if 0.000662 <= Q.lam1 < 0.00537:
        z += 249.0 * Q.lam1 - 1.33713
    if Q.lam2 < 0.00117:
        z += -414.0 * Q.lam2 + 0.48438
    if Q.log_sum_pt < 6.21:
        z += 3.88 * Q.log_sum_pt - 21.2178
    if 6.21 <= Q.log_sum_pt < 6.56:
        z += -8.22 * Q.log_sum_pt + 53.9232
    if Q.log_sum_pt >= 6.72:
        z += 19.1 * Q.log_sum_pt - 128.352
    if Q.pt_6 >= 37.0:
        z += 0.0304 * Q.pt_6 - 1.1248
    if 33.7 <= Q.pt_7 < 42.0:
        z += 0.0726 * Q.pt_7 - 2.44662
    if 42.0 <= Q.pt_7 < 53.6:
        z += 0.1572 * Q.pt_7 - 5.99982
    if Q.pt_7 >= 53.6:
        z += 0.0482 * Q.pt_7 - 0.15742
    if Q.sj3_dr_max < 0.159:
        z += -5.86 * Q.sj3_dr_max + 0.93174
    if Q.sj3_dr_min >= 0.0229:
        z += -2.2 * Q.sj3_dr_min + 0.05038
    if Q.sum_pt < 499.0:
        z += -0.03 * Q.sum_pt + 14.97
    if Q.sum_pt >= 839.0:
        z += -0.0217 * Q.sum_pt + 18.2063
    if Q.sum_pt_top5 < 556.0:
        z += 0.02276 * Q.sum_pt_top5 - 15.37456
    if 556.0 <= Q.sum_pt_top5 < 726.0:
        z += 0.016 * Q.sum_pt_top5 - 11.616
    if Q.sum_pt_top5 >= 733.0:
        z += 0.0167 * Q.sum_pt_top5 - 12.2411
    if Q.tau1 < 0.1:
        z += -18.5 * Q.tau1 + 1.85
    z += 12.6 * Q.z_5
    if Q.z_6 < 0.0467:
        z += 59.1 * Q.z_6 - 2.75997
    if Q.z_7 < 0.0629:
        z += 71.9 * Q.z_7 - 4.52251
    if Q.lam1 < 0.00913 and Q.centroid_offset > 0.0198:
        z += 9220.0 * (0.00913 - Q.lam1) * (Q.centroid_offset - 0.0198)
    if Q.lam1 < 0.00892 and Q.lam2 < 0.000937:
        z += -381000.0 * (0.00892 - Q.lam1) * (0.000937 - Q.lam2)
    if Q.lam1 < 0.0104 and Q.n_pt_above_50 > 5.76:
        z += -145.0 * (0.0104 - Q.lam1) * (Q.n_pt_above_50 - 5.76)
    if Q.log_sum_pt > 6.29 and Q.dr_2 < 0.104:
        z += -15.9 * (Q.log_sum_pt - 6.29) * (0.104 - Q.dr_2)
    if Q.log_sum_pt > 6.6 and Q.dr_max_012 < 0.123:
        z += 32.7 * (Q.log_sum_pt - 6.6) * (0.123 - Q.dr_max_012)
    if Q.log_sum_pt > 6.32 and Q.girth2_top2 < 0.00352:
        z += -852.0 * (Q.log_sum_pt - 6.32) * (0.00352 - Q.girth2_top2)
    if Q.log_sum_pt > 6.58 and Q.girth2_top3 < 0.00794:
        z += -1380.0 * (Q.log_sum_pt - 6.58) * (0.00794 - Q.girth2_top3)
    if Q.n_pt_above_50 > 5.82 and Q.e3 < 8.49e-05:
        z += 11600.0 * (Q.n_pt_above_50 - 5.82) * (8.49e-05 - Q.e3)
    if Q.pt_7 > 32.6 and Q.e3 < 8.02e-05:
        z += -970.0 * (Q.pt_7 - 32.6) * (8.02e-05 - Q.e3)
    if Q.z_6 < 0.0507 and Q.girth2_top2 < 0.00348:
        z += 12200.0 * (0.0507 - Q.z_6) * (0.00348 - Q.girth2_top2)
    if Q.z_7 < 0.0655 and Q.girth2_top3 < 0.00859:
        z += 6630.0 * (0.0655 - Q.z_7) * (0.00859 - Q.girth2_top3)
    return max(0.0, z)


def neuron_2(Q):
    z = 3.01
    if Q.C2 < 0.0643:
        z += 8.47 * Q.C2 - 0.544621
    if Q.LHA >= 0.12:
        z += -8.11 * Q.LHA + 0.9732
    if Q.lam1 < 0.0121:
        z += -111.0 * Q.lam1 + 1.3431
    if Q.log_sum_pt < 6.72:
        z += -13.5 * Q.log_sum_pt + 90.72
    if Q.mean_eta2 < 3.25e-05:
        z += 24100.0 * Q.mean_eta2 - 0.78325
    if Q.mean_phi2 < 9.69e-05:
        z += -66300.0 * Q.mean_phi2 + 6.42447
    if Q.pt_7 < 19.4:
        z += 0.0712 * Q.pt_7 - 3.82344
    if 19.4 <= Q.pt_7 < 53.7:
        z += 0.1376 * Q.pt_7 - 5.1116
    if Q.pt_7 >= 53.7:
        z += 0.0664 * Q.pt_7 - 1.28816
    if Q.sum_pt < 840.0:
        z += 0.00883 * Q.sum_pt - 7.4172
    if Q.z_7 >= 0.0247:
        z += -62.4 * Q.z_7 + 1.54128
    if Q.girth < 0.116 and Q.centroid_offset > 0.0173:
        z += -803.0 * (0.116 - Q.girth) * (Q.centroid_offset - 0.0173)
    if Q.lam1 < 0.00716 and Q.max_dr < 0.255:
        z += 1340.0 * (0.00716 - Q.lam1) * (0.255 - Q.max_dr)
    if Q.lam1 < 0.00752 and Q.pt_6 < 62.6:
        z += -2.78 * (0.00752 - Q.lam1) * (62.6 - Q.pt_6)
    if Q.log_sum_pt > 6.91 and Q.D2_b2 < 4.72:
        z += -4.56 * (Q.log_sum_pt - 6.91) * (4.72 - Q.D2_b2)
    if Q.log_sum_pt > 6.9 and Q.absphi_0 < 0.107:
        z += 515.0 * (Q.log_sum_pt - 6.9) * (0.107 - Q.absphi_0)
    if Q.log_sum_pt > 6.1 and Q.lam2 < 0.00022:
        z += -4070.0 * (Q.log_sum_pt - 6.1) * (0.00022 - Q.lam2)
    if Q.log_sum_pt > 6.08 and Q.mean_phi2 < 9.53e-05:
        z += -109000.0 * (Q.log_sum_pt - 6.08) * (9.53e-05 - Q.mean_phi2)
    if Q.log_sum_pt < 6.7 and Q.mean_phi2 < 9.61e-05:
        z += -101000.0 * (6.7 - Q.log_sum_pt) * (9.61e-05 - Q.mean_phi2)
    if Q.sum_pt > 867.0 and Q.D2_b2 < 4.31:
        z += 0.00268 * (Q.sum_pt - 867.0) * (4.31 - Q.D2_b2)
    if Q.sum_pt > 988.0 and Q.absphi_0 < 0.104:
        z += -0.332 * (Q.sum_pt - 988.0) * (0.104 - Q.absphi_0)
    if Q.sum_pt > 927.0 and Q.mean_phi2 < 9e-05:
        z += 119.0 * (Q.sum_pt - 927.0) * (9e-05 - Q.mean_phi2)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.838
    if 0.302 <= Q.LHA < 0.324:
        z += 22.5 * Q.LHA - 6.795
    if Q.LHA >= 0.324:
        z += -4.5 * Q.LHA + 1.953
    if Q.eccentricity >= 0.926:
        z += -7.21 * Q.eccentricity + 6.67646
    if Q.girth < 0.0767:
        z += 57.7 * Q.girth - 4.42559
    if Q.girth >= 0.132:
        z += 28.0 * Q.girth - 3.696
    if Q.lam1 < 0.00541:
        z += -92.1 * Q.lam1 + 0.705486
    if 0.00541 <= Q.lam1 < 0.00766:
        z += 262.9 * Q.lam1 - 1.215064
    if 0.00766 <= Q.lam1 < 0.0109:
        z += 355.0 * Q.lam1 - 1.92055
    if Q.lam1 >= 0.0109:
        z += -63.0 * Q.lam1 + 2.63565
    if Q.planar_flow >= 0.0924:
        z += 1.09 * Q.planar_flow - 0.100716
    if 0.181 <= Q.sj2_dr < 0.228:
        z += 17.6 * Q.sj2_dr - 3.1856
    if Q.sj2_dr >= 0.228:
        z += 4.0 * Q.sj2_dr - 0.0848
    if Q.sj3_dr_max < 0.185:
        z += -7.45 * Q.sj3_dr_max + 1.3857
    if 0.185 <= Q.sj3_dr_max < 0.186:
        z += 9.35 * Q.sj3_dr_max - 1.7223
    if 0.186 <= Q.sj3_dr_max < 0.269:
        z += 16.8 * Q.sj3_dr_max - 3.108
    if Q.sj3_dr_max >= 0.269:
        z += -1.0 * Q.sj3_dr_max + 1.6802
    if Q.tau1 < 0.0939:
        z += -8.57 * Q.tau1 + 0.91699
    if 0.0939 <= Q.tau1 < 0.107:
        z += 7.83 * Q.tau1 - 0.62297
    if Q.tau1 >= 0.107:
        z += 16.4 * Q.tau1 - 1.53996
    if Q.LHA > 0.331 and Q.eccentricity > 0.66:
        z += 152.0 * (Q.LHA - 0.331) * (Q.eccentricity - 0.66)
    if Q.N3 < 1.23 and Q.log_sum_pt > 6.1:
        z += -7.49 * (1.23 - Q.N3) * (Q.log_sum_pt - 6.1)
    if Q.N3 < 1.22 and Q.z_7 < 0.0817:
        z += 61.6 * (1.22 - Q.N3) * (0.0817 - Q.z_7)
    if Q.girth > 0.11 and Q.eccentricity > 0.686:
        z += -247.0 * (Q.girth - 0.11) * (Q.eccentricity - 0.686)
    if Q.girth > 0.0444 and Q.n_pt_above_50 > 2.21:
        z += -2.27 * (Q.girth - 0.0444) * (Q.n_pt_above_50 - 2.21)
    if Q.sj2_dr > 0.136 and Q.eccentricity > 0.936:
        z += 147.0 * (Q.sj2_dr - 0.136) * (Q.eccentricity - 0.936)
    if Q.sj2_dr > 0.148 and Q.log_sum_pt > 6.11:
        z += -12.3 * (Q.sj2_dr - 0.148) * (Q.log_sum_pt - 6.11)
    return max(0.0, z)


def neuron_4(Q):
    z = -3.14
    if Q.C2 < 0.0338:
        z += -55.1 * Q.C2 + 1.88993
    if 0.0338 <= Q.C2 < 0.0343:
        z += -114.5 * Q.C2 + 3.89765
    if Q.C2 >= 0.0343:
        z += -59.4 * Q.C2 + 2.00772
    if Q.D2 >= 0.937:
        z += 0.443 * Q.D2 - 0.415091
    if Q.e3 >= 3.15e-05:
        z += -3220.0 * Q.e3 + 0.10143
    if Q.girth < 0.0149:
        z += 436.8 * Q.girth - 1.41648
    if 0.0149 <= Q.girth < 0.0761:
        z += -83.2 * Q.girth + 6.33152
    if Q.lam1 < 0.000141:
        z += -683.0 * Q.lam1 - 12.88384
    if 0.000141 <= Q.lam1 < 0.00378:
        z += 1837.0 * Q.lam1 - 13.23916
    if 0.00378 <= Q.lam1 < 0.00593:
        z += 1622.0 * Q.lam1 - 12.42646
    if 0.00593 <= Q.lam1 < 0.00827:
        z += 1200.0 * Q.lam1 - 9.924
    if Q.lam2 < 0.000416:
        z += 1390.0 * Q.lam2 + 2.91086
    if 0.000416 <= Q.lam2 < 0.00345:
        z += -1150.0 * Q.lam2 + 3.9675
    if Q.log_sum_pt < 6.59:
        z += 7.8 * Q.log_sum_pt - 51.402
    if Q.max_dr < 0.113:
        z += -24.6 * Q.max_dr + 6.0762
    if 0.113 <= Q.max_dr < 0.247:
        z += 26.4 * Q.max_dr + 0.3132
    if Q.max_dr >= 0.247:
        z += 21.5 * Q.max_dr + 1.5235
    if Q.mean_eta < -0.0226:
        z += -47.1 * Q.mean_eta - 1.06446
    z += 0.22 * Q.n_dr_0p1_0p2
    if Q.planar_flow >= 0.0231:
        z += 2.75 * Q.planar_flow - 0.063525
    if Q.pt_7 < 24.3:
        z += 0.102 * Q.pt_7 - 2.4786
    if Q.pt_dispersion < 0.564:
        z += -4.22 * Q.pt_dispersion + 2.38008
    if Q.sj2_dr >= 0.163:
        z += -16.4 * Q.sj2_dr + 2.6732
    if Q.sj3_dr_max < 0.0773:
        z += 5.4 * Q.sj3_dr_max - 2.14491
    if 0.0773 <= Q.sj3_dr_max < 0.142:
        z += 26.7 * Q.sj3_dr_max - 3.7914
    if Q.sj3_dr_min < 0.2:
        z += -36.0 * Q.sj3_dr_min + 7.2
    if Q.sum_pt >= 722.0:
        z += 0.00897 * Q.sum_pt - 6.47634
    if Q.sum_pt_top3 >= 789.0:
        z += -0.00599 * Q.sum_pt_top3 + 4.72611
    if Q.sum_pt_top5 < 365.0:
        z += -0.0143 * Q.sum_pt_top5 + 5.2195
    if Q.z_7 < 0.0662:
        z += 22.3 * Q.z_7 - 1.47626
    if Q.zdr_0 < 0.0161:
        z += 58.4 * Q.zdr_0 - 0.94024
    if Q.N2 < 0.29 and Q.mean_phi < -0.0194:
        z += 280.0 * (0.29 - Q.N2) * (-0.0194 - Q.mean_phi)
    if Q.N2 < 0.259 and Q.sum_pt < 866.0:
        z += -0.021 * (0.259 - Q.N2) * (866.0 - Q.sum_pt)
    if Q.dr_0 < 0.0759 and Q.centroid_offset > 0.0285:
        z += -1200.0 * (0.0759 - Q.dr_0) * (Q.centroid_offset - 0.0285)
    if Q.dr_0 < 0.0818 and Q.centroid_offset > 0.00691:
        z += 1910.0 * (0.0818 - Q.dr_0) * (Q.centroid_offset - 0.00691)
    if Q.eccentricity > 0.72 and Q.n_pt_above_50 > 6.13:
        z += 1.76 * (Q.eccentricity - 0.72) * (Q.n_pt_above_50 - 6.13)
    if Q.lam1 < 0.00822 and Q.lam2 > 0.000406:
        z += 330000.0 * (0.00822 - Q.lam1) * (Q.lam2 - 0.000406)
    if Q.log_sum_pt > 6.3 and Q.n_dr_0p05_0p1 < 6.6:
        z += -0.37 * (Q.log_sum_pt - 6.3) * (6.6 - Q.n_dr_0p05_0p1)
    if Q.max_dr > 0.106 and Q.n_pt_above_50 > 6.19:
        z += -6.37 * (Q.max_dr - 0.106) * (Q.n_pt_above_50 - 6.19)
    if Q.sj2_dr > 0.22 and Q.C2_b2 < 0.00183:
        z += -7670.0 * (Q.sj2_dr - 0.22) * (0.00183 - Q.C2_b2)
    if Q.sj3_dr_max > 0.241 and Q.log_sum_pt > 6.11:
        z += -53.6 * (Q.sj3_dr_max - 0.241) * (Q.log_sum_pt - 6.11)
    if Q.sj3_dr_max > 0.215 and Q.sj2_zsoft < 0.0638:
        z += 543.0 * (Q.sj3_dr_max - 0.215) * (0.0638 - Q.sj2_zsoft)
    if Q.sj3_dr_min < 0.21 and Q.lam2 < 0.00328:
        z += -10000.0 * (0.21 - Q.sj3_dr_min) * (0.00328 - Q.lam2)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.02
    if Q.centroid_offset < 0.0383:
        z += 27.3 * Q.centroid_offset - 1.04559
    if Q.sum_pt < 767.0:
        z += 0.00467 * Q.sum_pt - 3.58189
    if Q.z_7 < 0.0295:
        z += -124.0 * Q.z_7 + 3.658
    if Q.LHA < 0.177 and Q.log_sum_pt < 6.86:
        z += -47.0 * (0.177 - Q.LHA) * (6.86 - Q.log_sum_pt)
    if Q.e2 < 0.0361 and Q.z_7 < 0.0713:
        z += 1550.0 * (0.0361 - Q.e2) * (0.0713 - Q.z_7)
    if Q.girth < 0.0114 and Q.sum_pt_top5 > 598.0:
        z += 0.49 * (0.0114 - Q.girth) * (Q.sum_pt_top5 - 598.0)
    if Q.lam1 < 0.00242 and Q.centroid_offset < 0.0214:
        z += 80600.0 * (0.00242 - Q.lam1) * (0.0214 - Q.centroid_offset)
    if Q.log_sum_pt > 6.83 and Q.e3 < 3.49e-05:
        z += -206000.0 * (Q.log_sum_pt - 6.83) * (3.49e-05 - Q.e3)
    if Q.log_sum_pt > 6.91 and Q.zdr_0 < 0.0118:
        z += -778.0 * (Q.log_sum_pt - 6.91) * (0.0118 - Q.zdr_0)
    if Q.sum_pt < 673.0 and Q.mean_eta2 < 0.00688:
        z += 0.262 * (673.0 - Q.sum_pt) * (0.00688 - Q.mean_eta2)
    if Q.z_6 < 0.0537 and Q.pt1_dr01 < 36.2:
        z += 0.965 * (0.0537 - Q.z_6) * (36.2 - Q.pt1_dr01)
    return max(0.0, z)


def neuron_6(Q):
    z = -0.233
    if Q.LHA < 0.141:
        z += -38.5 * Q.LHA + 12.3585
    if 0.141 <= Q.LHA < 0.257:
        z += -62.8 * Q.LHA + 15.7848
    if 0.257 <= Q.LHA < 0.321:
        z += 11.6 * Q.LHA - 3.336
    if Q.LHA >= 0.321:
        z += 50.1 * Q.LHA - 15.6945
    if Q.centroid_offset >= 0.0271:
        z += 70.9 * Q.centroid_offset - 1.92139
    if Q.eccentricity >= 0.864:
        z += -5.68 * Q.eccentricity + 4.90752
    if Q.girth < 0.0575:
        z += 256.4 * Q.girth - 16.63244
    if 0.0575 <= Q.girth < 0.0816:
        z += 78.4 * Q.girth - 6.39744
    if Q.girth >= 0.0957:
        z += -97.6 * Q.girth + 9.34032
    if Q.lam2 < 0.00088:
        z += -506.0 * Q.lam2 + 2.01388
    if 0.00088 <= Q.lam2 < 0.00398:
        z += -763.0 * Q.lam2 + 2.24004
    if Q.lam2 >= 0.00398:
        z += -257.0 * Q.lam2 + 0.22616
    if Q.log_sum_pt < 6.62:
        z += -6.43 * Q.log_sum_pt + 42.5666
    if Q.max_dr < 0.17:
        z += 19.6 * Q.max_dr - 3.332
    if Q.pt_5 < 26.4:
        z += -0.0713 * Q.pt_5 + 1.88232
    if Q.pt_7 >= 28.5:
        z += 0.0646 * Q.pt_7 - 1.8411
    if Q.sj3_dr_max < 0.176:
        z += -22.5 * Q.sj3_dr_max + 4.1625
    if 0.176 <= Q.sj3_dr_max < 0.185:
        z += -20.62 * Q.sj3_dr_max + 3.83162
    if Q.sj3_dr_max >= 0.185:
        z += 1.88 * Q.sj3_dr_max - 0.33088
    if Q.sum_pt >= 979.0:
        z += 0.00743 * Q.sum_pt - 7.27397
    if Q.sum_pt_top5 >= 765.0:
        z += -0.00741 * Q.sum_pt_top5 + 5.66865
    if Q.tau1 < 0.0605:
        z += -51.5 * Q.tau1 + 3.11575
    if 0.0615 <= Q.tau1 < 0.101:
        z += -27.4 * Q.tau1 + 1.6851
    if Q.tau1 >= 0.101:
        z += 5.3 * Q.tau1 - 1.6176
    if Q.z_7 < 0.0388:
        z += -30.4 * Q.z_7 + 1.20384
    if 0.0388 <= Q.z_7 < 0.0396:
        z += -83.6 * Q.z_7 + 3.268
    if Q.z_7 >= 0.0396:
        z += -53.2 * Q.z_7 + 2.06416
    if Q.centroid_offset > -0.000127 and Q.eccentricity < 0.918:
        z += 52.3 * (Q.centroid_offset - -0.000127) * (0.918 - Q.eccentricity)
    if Q.centroid_offset > 0.0101 and Q.mean_phi2 < 0.0132:
        z += 2350.0 * (Q.centroid_offset - 0.0101) * (0.0132 - Q.mean_phi2)
    if Q.centroid_offset > 0.0249 and Q.n_dr_0_0p05 < 5.27:
        z += -11.7 * (Q.centroid_offset - 0.0249) * (5.27 - Q.n_dr_0_0p05)
    if Q.centroid_offset > 0.0174 and Q.pt_2 > 66.7:
        z += 0.404 * (Q.centroid_offset - 0.0174) * (Q.pt_2 - 66.7)
    if Q.pt_6 < 34.3 and Q.sum_pt_top3 < 728.0:
        z += 0.00035 * (34.3 - Q.pt_6) * (728.0 - Q.sum_pt_top3)
    if Q.sj2_dr > 0.16 and Q.C2 < 0.108:
        z += 151.0 * (Q.sj2_dr - 0.16) * (0.108 - Q.C2)
    if Q.sj2_dr > 0.108 and Q.mean_eta > 0.0256:
        z += -131.0 * (Q.sj2_dr - 0.108) * (Q.mean_eta - 0.0256)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.633
    if Q.C2 >= 0.0303:
        z += 31.1 * Q.C2 - 0.94233
    if Q.LHA >= 0.317:
        z += 120.0 * Q.LHA - 38.04
    if Q.centroid_offset < 0.0511:
        z += -50.4 * Q.centroid_offset + 2.57544
    if Q.girth < 0.0556:
        z += 65.2 * Q.girth - 3.62512
    if 0.0886 <= Q.girth < 0.128:
        z += -213.0 * Q.girth + 18.8718
    if Q.girth >= 0.128:
        z += -8.3922
    if Q.girth2_top3 >= 0.0111:
        z += 128.0 * Q.girth2_top3 - 1.4208
    if Q.lam1 < 0.00461:
        z += 877.0 * Q.lam1 - 4.04297
    if Q.lam1 >= 0.00734:
        z += -1200.0 * Q.lam1 + 8.808
    if Q.lam2 < 0.000344:
        z += 2010.0 * Q.lam2 - 0.69144
    if Q.lam2 >= 0.0011:
        z += -998.0 * Q.lam2 + 1.0978
    if Q.max_dr < 0.0954:
        z += 18.9 * Q.max_dr - 1.80306
    if Q.mean_eta < -0.0118:
        z += 28.4 * Q.mean_eta + 0.33512
    if Q.n_dr_0p2_0p4 < 1.0:
        z += -1.11 * Q.n_dr_0p2_0p4 + 1.11
    if Q.n_dr_0p2_0p4 >= 1.04:
        z += -0.654 * Q.n_dr_0p2_0p4 + 0.68016
    if Q.sj3_dr_max < 0.165:
        z += -22.9 * Q.sj3_dr_max + 3.7785
    if Q.sj3_dr_max >= 0.225:
        z += -6.56 * Q.sj3_dr_max + 1.476
    if Q.tau1 < 0.0573:
        z += -52.6 * Q.tau1 + 4.46048
    if 0.0573 <= Q.tau1 < 0.0848:
        z += 11.1 * Q.tau1 + 0.81047
    if 0.0848 <= Q.tau1 < 0.103:
        z += 63.7 * Q.tau1 - 3.65001
    if Q.tau1 >= 0.103:
        z += 159.8 * Q.tau1 - 13.54831
    if Q.zdr_0 >= 0.0244:
        z += -23.8 * Q.zdr_0 + 0.58072
    if Q.zdr_1 >= 0.0246:
        z += -55.3 * Q.zdr_1 + 1.36038
    if Q.zdr_2 >= 0.0197:
        z += -39.8 * Q.zdr_2 + 0.78406
    if Q.centroid_offset < 0.0315 and Q.n_dr_0p2_0p4 < 1.14:
        z += -75.8 * (0.0315 - Q.centroid_offset) * (1.14 - Q.n_dr_0p2_0p4)
    if Q.e3 < 0.000137 and Q.pt_7 < 37.3:
        z += -401.0 * (0.000137 - Q.e3) * (37.3 - Q.pt_7)
    if Q.girth > 0.0818 and Q.D2 < 2.88:
        z += -113.0 * (Q.girth - 0.0818) * (2.88 - Q.D2)
    if Q.lam1 > 0.00308 and Q.D2 > -0.125:
        z += -237.0 * (Q.lam1 - 0.00308) * (Q.D2 - -0.125)
    if Q.lam1 < 0.00806 and Q.mean_eta < -0.00825:
        z += 6780.0 * (0.00806 - Q.lam1) * (-0.00825 - Q.mean_eta)
    if Q.lam1 > 0.00309 and Q.planar_flow < 0.252:
        z += 2800.0 * (Q.lam1 - 0.00309) * (0.252 - Q.planar_flow)
    if Q.lam1 > 0.00798 and Q.pt_4 < 100.0:
        z += 14.6 * (Q.lam1 - 0.00798) * (100.0 - Q.pt_4)
    if Q.lam1 > 0.00186 and Q.sj3_dr_max < 0.191:
        z += -9760.0 * (Q.lam1 - 0.00186) * (0.191 - Q.sj3_dr_max)
    if Q.planar_flow < 0.262 and Q.lam1 > 0.008:
        z += -2950.0 * (0.262 - Q.planar_flow) * (Q.lam1 - 0.008)
    if Q.planar_flow < 0.197 and Q.log_sum_pt > 6.46:
        z += 19.0 * (0.197 - Q.planar_flow) * (Q.log_sum_pt - 6.46)
    if Q.planar_flow < 0.188 and Q.max_dr < 0.237:
        z += -24.7 * (0.188 - Q.planar_flow) * (0.237 - Q.max_dr)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.72
    if Q.C2_b2 >= 0.00173:
        z += -21.9 * Q.C2_b2 + 0.037887
    if Q.LHA < 0.154:
        z += 15.3 * Q.LHA - 4.7889
    if 0.154 <= Q.LHA < 0.257:
        z += 36.1 * Q.LHA - 7.9921
    if 0.257 <= Q.LHA < 0.313:
        z += 21.8 * Q.LHA - 4.317
    if Q.LHA >= 0.313:
        z += 6.5 * Q.LHA + 0.4719
    if Q.M2 < 0.0945:
        z += -4.51 * Q.M2 + 0.426195
    if Q.N2 >= 0.215:
        z += 1.37 * Q.N2 - 0.29455
    if Q.absphi_0 >= 0.0193:
        z += -5.31 * Q.absphi_0 + 0.102483
    if Q.centroid_offset >= 0.0335:
        z += 6.25 * Q.centroid_offset - 0.209375
    if Q.dr0_7 < 0.16:
        z += 0.903 * Q.dr0_7 - 0.14448
    if Q.dr_7 < 0.107:
        z += -2.03 * Q.dr_7 + 0.21721
    if Q.eccentricity >= 0.811:
        z += -3.15 * Q.eccentricity + 2.55465
    if 0.0218 <= Q.girth < 0.0828:
        z += -103.0 * Q.girth + 2.2454
    if Q.girth >= 0.0828:
        z += -8.6 * Q.girth - 5.57092
    if Q.girth2_top3 < 0.00715:
        z += 64.8 * Q.girth2_top3 - 0.46332
    if Q.girth2_top5 >= 0.00906:
        z += -21.3 * Q.girth2_top5 + 0.192978
    if Q.lam1 < 0.000483:
        z += -1050.0 * Q.lam1 + 1.201747
    if 0.000483 <= Q.lam1 < 0.0045:
        z += -241.0 * Q.lam1 + 0.811
    if 0.0045 <= Q.lam1 < 0.0073:
        z += 321.0 * Q.lam1 - 1.718
    if 0.0073 <= Q.lam1 < 0.00773:
        z += 671.0 * Q.lam1 - 4.273
    if 0.00773 <= Q.lam1 < 0.016:
        z += 74.0 * Q.lam1 + 0.34181
    if Q.lam1 >= 0.016:
        z += -35.0 * Q.lam1 + 2.08581
    if Q.lam2 < 0.000203:
        z += -3140.0 * Q.lam2 + 0.63742
    if Q.log_sum_pt < 6.57:
        z += 2.83 * Q.log_sum_pt - 19.0459
    if 6.57 <= Q.log_sum_pt < 6.73:
        z += 17.63 * Q.log_sum_pt - 116.2819
    if 6.73 <= Q.log_sum_pt < 6.84:
        z += 14.8 * Q.log_sum_pt - 97.236
    if Q.log_sum_pt >= 6.84:
        z += -5.4 * Q.log_sum_pt + 40.932
    if Q.max_dr < 0.0491:
        z += 3.67 * Q.max_dr + 0.39715
    if 0.0491 <= Q.max_dr < 0.116:
        z += -8.63 * Q.max_dr + 1.00108
    if Q.mean_eta < 0.000664:
        z += 3.82 * Q.mean_eta - 0.00253648
    if Q.mean_eta2 < 0.00858:
        z += 73.4 * Q.mean_eta2 - 0.629772
    if Q.mean_eta2 >= 0.00919:
        z += 81.3 * Q.mean_eta2 - 0.747147
    if Q.mean_phi < 0.00363:
        z += 9.82 * Q.mean_phi - 0.0356466
    if Q.mean_phi2 >= 0.00063:
        z += 77.4 * Q.mean_phi2 - 0.048762
    if Q.n_dr_0p1_0p2 < 3.93:
        z += -0.0617 * Q.n_dr_0p1_0p2 + 0.242481
    if Q.n_dr_0p2_0p4 < 1.01:
        z += -0.171 * Q.n_dr_0p2_0p4 + 0.17271
    if Q.phi_0 < -0.0208:
        z += -7.11 * Q.phi_0 - 0.147888
    if Q.phi_0 >= -0.0047:
        z += 4.93 * Q.phi_0 + 0.023171
    if Q.psi_0p2 >= 0.9:
        z += -1.47 * Q.psi_0p2 + 1.323
    if Q.pt_6 >= 32.0:
        z += -0.0271 * Q.pt_6 + 0.8672
    if Q.pt_7 < 41.6:
        z += -0.021 * Q.pt_7 + 0.8736
    if Q.ptdr0_6 < 5.55:
        z += 0.0217 * Q.ptdr0_6 - 0.120869
    if 5.55 <= Q.ptdr0_6 < 5.57:
        z += -0.0034 * Q.ptdr0_6 + 0.018436
    if Q.ptdr0_6 >= 5.57:
        z += -0.0251 * Q.ptdr0_6 + 0.139305
    if Q.ptdr0_7 >= 4.76:
        z += -0.0326 * Q.ptdr0_7 + 0.155176
    if Q.sj2_dr < 0.159:
        z += 1.07 * Q.sj2_dr - 0.17013
    if 0.161 <= Q.sj2_dr < 0.205:
        z += -8.93 * Q.sj2_dr + 1.43773
    if Q.sj2_dr >= 0.205:
        z += -0.58 * Q.sj2_dr - 0.27402
    if Q.sj3_dr_max < 0.0568:
        z += 3.5 * Q.sj3_dr_max - 1.27416
    if 0.0568 <= Q.sj3_dr_max < 0.132:
        z += 14.3 * Q.sj3_dr_max - 1.8876
    if 0.165 <= Q.sj3_dr_max < 0.255:
        z += -6.96 * Q.sj3_dr_max + 1.1484
    if 0.255 <= Q.sj3_dr_max < 0.351:
        z += -0.47 * Q.sj3_dr_max - 0.50655
    if Q.sj3_dr_max >= 0.351:
        z += 2.59 * Q.sj3_dr_max - 1.58061
    if 714.0 <= Q.sum_pt < 937.0:
        z += -0.02 * Q.sum_pt + 14.28
    if Q.sum_pt >= 937.0:
        z += 0.0044 * Q.sum_pt - 8.5828
    if Q.sum_pt_top3 < 436.0:
        z += -0.00422 * Q.sum_pt_top3 + 1.88212
    if 436.0 <= Q.sum_pt_top3 < 446.0:
        z += -0.00736 * Q.sum_pt_top3 + 3.25116
    if Q.sum_pt_top3 >= 446.0:
        z += -0.00314 * Q.sum_pt_top3 + 1.36904
    if Q.tau1 < 0.0259:
        z += 35.0 * Q.tau1 - 0.488
    if 0.0259 <= Q.tau1 < 0.0371:
        z += -21.7 * Q.tau1 + 0.98053
    if 0.0371 <= Q.tau1 < 0.0937:
        z += -3.1 * Q.tau1 + 0.29047
    if Q.tau1 >= 0.154:
        z += 0.988 * Q.tau1 - 0.152152
    if Q.tau2 < 0.0172:
        z += 32.2 * Q.tau2 - 0.55384
    if Q.z_6 >= 0.043:
        z += 21.7 * Q.z_6 - 0.9331
    if Q.z_7 >= 0.0367:
        z += 5.82 * Q.z_7 - 0.213594
    if Q.z_top3_slots < 0.616:
        z += 3.62 * Q.z_top3_slots - 2.22992
    if Q.z_top3_slots >= 0.619:
        z += 3.75 * Q.z_top3_slots - 2.32125
    if Q.zdr_0 < 0.0199:
        z += -17.0 * Q.zdr_0 + 0.3383
    if Q.zdr_1 < 0.014:
        z += -16.2 * Q.zdr_1 + 0.2268
    if Q.zdr_2 < 0.0117:
        z += -17.2 * Q.zdr_2 + 0.20124
    if Q.zdr_5 < 0.00715:
        z += -20.4 * Q.zdr_5 + 0.14586
    if Q.zdr_6 < 0.00632:
        z += -33.4 * Q.zdr_6 + 0.211088
    if Q.girth < 0.0338 and Q.centroid_offset < 0.0314:
        z += -2950.0 * (0.0338 - Q.girth) * (0.0314 - Q.centroid_offset)
    if Q.lam1 < 0.00468 and Q.centroid_offset > 0.0104:
        z += -17500.0 * (0.00468 - Q.lam1) * (Q.centroid_offset - 0.0104)
    if Q.lam1 < 0.00696 and Q.centroid_offset < 0.0237:
        z += 12200.0 * (0.00696 - Q.lam1) * (0.0237 - Q.centroid_offset)
    if Q.lam1 < 0.00575 and Q.log_sum_pt < 6.59:
        z += -302.0 * (0.00575 - Q.lam1) * (6.59 - Q.log_sum_pt)
    if Q.lam1 < 0.00814 and Q.planar_flow < 0.535:
        z += 245.0 * (0.00814 - Q.lam1) * (0.535 - Q.planar_flow)
    if Q.lam1 < 0.00574 and Q.pt_7 < 39.6:
        z += -7.69 * (0.00574 - Q.lam1) * (39.6 - Q.pt_7)
    if Q.log_sum_pt > 6.7 and Q.girth2_top5 < 0.0027:
        z += 1220.0 * (Q.log_sum_pt - 6.7) * (0.0027 - Q.girth2_top5)
    if Q.sj3_dr_max < 0.137 and Q.centroid_offset > 0.0129:
        z += 463.0 * (0.137 - Q.sj3_dr_max) * (Q.centroid_offset - 0.0129)
    if Q.sj3_dr_max < 0.142 and Q.centroid_offset > 0.00405:
        z += -607.0 * (0.142 - Q.sj3_dr_max) * (Q.centroid_offset - 0.00405)
    if Q.sj3_dr_max < 0.16 and Q.girth2_top2 < 0.0018:
        z += 9030.0 * (0.16 - Q.sj3_dr_max) * (0.0018 - Q.girth2_top2)
    if Q.sj3_dr_max < 0.203 and Q.girth2_top2 < 0.00185:
        z += -5470.0 * (0.203 - Q.sj3_dr_max) * (0.00185 - Q.girth2_top2)
    if Q.tau1 < 0.0274 and Q.centroid_offset < 0.0314:
        z += 2350.0 * (0.0274 - Q.tau1) * (0.0314 - Q.centroid_offset)
    if Q.z_6 < 0.0455 and Q.absphi_1 < 0.0703:
        z += -282.0 * (0.0455 - Q.z_6) * (0.0703 - Q.absphi_1)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.23
    if Q.girth < 0.042:
        z += 65.4 * Q.girth - 2.7468
    if Q.girth >= 0.1:
        z += 19.5 * Q.girth - 1.95
    if Q.lam1 < 0.000135:
        z += -11212.0 * Q.lam1 + 2.7484
    if 0.000135 <= Q.lam1 < 0.00724:
        z += -112.0 * Q.lam1 + 1.2499
    if 0.00724 <= Q.lam1 < 0.00805:
        z += -542.0 * Q.lam1 + 4.3631
    if Q.lam2 >= 0.00118:
        z += 358.0 * Q.lam2 - 0.42244
    if Q.sj3_dr_max < 0.163:
        z += 13.8 * Q.sj3_dr_max - 3.0912
    if 0.163 <= Q.sj3_dr_max < 0.224:
        z += -18.8 * Q.sj3_dr_max + 2.2226
    if 0.224 <= Q.sj3_dr_max < 0.242:
        z += -32.6 * Q.sj3_dr_max + 5.3138
    if Q.sj3_dr_max >= 0.242:
        z += 6.5 * Q.sj3_dr_max - 4.1484
    if Q.sum_pt < 963.0:
        z += -0.00562 * Q.sum_pt + 5.41206
    if Q.tau1 < 0.0836:
        z += 34.3 * Q.tau1 - 2.86748
    if Q.centroid_offset < 0.0267 and Q.lam1 < 0.00719:
        z += 48900.0 * (0.0267 - Q.centroid_offset) * (0.00719 - Q.lam1)
    if Q.lam1 < 0.00564 and Q.lam2 < 0.00108:
        z += 1110000.0 * (0.00564 - Q.lam1) * (0.00108 - Q.lam2)
    if Q.sum_pt < 910.0 and Q.centroid_offset < 0.0135:
        z += -0.24 * (910.0 - Q.sum_pt) * (0.0135 - Q.centroid_offset)
    if Q.sum_pt < 994.0 and Q.z_5 < 0.0548:
        z += 0.157 * (994.0 - Q.sum_pt) * (0.0548 - Q.z_5)
    return max(0.0, z)


def neuron_10(Q):
    z = -6.13
    if Q.D2_b2 < 0.307:
        z += -3.56 * Q.D2_b2 + 1.09292
    if Q.lam1 < 0.00279:
        z += 759.0 * Q.lam1 - 2.11761
    if Q.lam1 >= 0.00291:
        z += 245.0 * Q.lam1 - 0.71295
    if 0.000183 <= Q.lam2 < 0.00352:
        z += 988.0 * Q.lam2 - 0.180804
    if Q.lam2 >= 0.00352:
        z += 490.0 * Q.lam2 + 1.572156
    if Q.n_dr_0_0p05 < 4.76:
        z += -1.38 * Q.n_dr_0_0p05 + 6.6654
    if 4.76 <= Q.n_dr_0_0p05 < 4.83:
        z += -2.7 * Q.n_dr_0_0p05 + 12.9486
    if Q.n_dr_0_0p05 >= 4.83:
        z += -1.32 * Q.n_dr_0_0p05 + 6.2832
    if Q.n_dr_0p05_0p1 < 4.0:
        z += -1.48 * Q.n_dr_0p05_0p1 + 6.1716
    if 4.0 <= Q.n_dr_0p05_0p1 < 4.17:
        z += -2.75 * Q.n_dr_0p05_0p1 + 11.2516
    if Q.n_dr_0p05_0p1 >= 4.17:
        z += -1.27 * Q.n_dr_0p05_0p1 + 5.08
    if Q.n_dr_0p1_0p2 < 3.79:
        z += -1.33 * Q.n_dr_0p1_0p2 + 5.2003
    if 3.79 <= Q.n_dr_0p1_0p2 < 3.91:
        z += -2.78 * Q.n_dr_0p1_0p2 + 10.6958
    if Q.n_dr_0p1_0p2 >= 3.91:
        z += -1.45 * Q.n_dr_0p1_0p2 + 5.4955
    if Q.n_dr_0p2_0p4 < 0.986:
        z += -1.51 * Q.n_dr_0p2_0p4 + 1.48886
    if Q.z_7 < 0.0652:
        z += 24.5 * Q.z_7 - 1.5974
    if Q.LHA > 0.311 and Q.planar_flow < 0.791:
        z += -40.0 * (Q.LHA - 0.311) * (0.791 - Q.planar_flow)
    if Q.lam2 > -0.000475 and Q.log_sum_pt < 6.61:
        z += -567.0 * (Q.lam2 - -0.000475) * (6.61 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 > 1.12 and Q.D2_b2 < 4.59:
        z += -0.325 * (Q.n_dr_0p2_0p4 - 1.12) * (4.59 - Q.D2_b2)
    if Q.pt_6 < 44.0 and Q.log_sum_pt < 6.56:
        z += -0.163 * (44.0 - Q.pt_6) * (6.56 - Q.log_sum_pt)
    if Q.pt_7 > 22.6 and Q.D2_b2 < 0.262:
        z += -0.126 * (Q.pt_7 - 22.6) * (0.262 - Q.D2_b2)
    if Q.sj2_dr > 0.161 and Q.D2_b2 < 0.453:
        z += -29.3 * (Q.sj2_dr - 0.161) * (0.453 - Q.D2_b2)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.425
    if Q.centroid_offset < 0.0163:
        z += -45.5 * Q.centroid_offset + 0.74165
    if Q.girth < 0.0745:
        z += 64.7 * Q.girth - 4.82015
    if Q.lam1 < 0.0116:
        z += -826.0 * Q.lam1 + 9.5816
    if Q.lam1 >= 0.0116:
        z += -105.0 * Q.lam1 + 1.218
    if Q.log_sum_pt < 6.78:
        z += 2.72 * Q.log_sum_pt - 18.4416
    if Q.planar_flow < 0.235:
        z += -5.39 * Q.planar_flow + 1.26665
    if 0.122 <= Q.sj2_dr < 0.156:
        z += 30.7 * Q.sj2_dr - 3.7454
    if Q.sj2_dr >= 0.156:
        z += -6.6 * Q.sj2_dr + 2.0734
    if Q.tau1 < 0.138:
        z += 29.1 * Q.tau1 - 4.0158
    if Q.z_7 < 0.0714:
        z += 24.0 * Q.z_7 - 1.7136
    if Q.lam1 < 0.00769 and Q.D2_b2 < 0.378:
        z += -562.0 * (0.00769 - Q.lam1) * (0.378 - Q.D2_b2)
    if Q.lam1 < 0.00434 and Q.centroid_offset < 0.0266:
        z += -35000.0 * (0.00434 - Q.lam1) * (0.0266 - Q.centroid_offset)
    if Q.lam1 < 0.0142 and Q.centroid_offset > 0.0161:
        z += -8590.0 * (0.0142 - Q.lam1) * (Q.centroid_offset - 0.0161)
    if Q.lam1 < 0.0161 and Q.lam2 < 0.00167:
        z += 109000.0 * (0.0161 - Q.lam1) * (0.00167 - Q.lam2)
    if Q.lam1 < 0.00308 and Q.planar_flow < 0.24:
        z += -2240.0 * (0.00308 - Q.lam1) * (0.24 - Q.planar_flow)
    if Q.sum_pt_top5 > 474.0 and Q.D2_b2 < 0.318:
        z += 0.011 * (Q.sum_pt_top5 - 474.0) * (0.318 - Q.D2_b2)
    return max(0.0, z)


def neuron_12(Q):
    z = 4.55
    if Q.C2 < 0.0675:
        z += -16.8 * Q.C2 + 1.134
    if Q.LHA >= 0.105:
        z += -22.8 * Q.LHA + 2.394
    if Q.centroid_offset < 0.0348:
        z += 23.8 * Q.centroid_offset - 0.88298
    if 0.0348 <= Q.centroid_offset < 0.0371:
        z += 69.6 * Q.centroid_offset - 2.47682
    if Q.centroid_offset >= 0.0371:
        z += 45.8 * Q.centroid_offset - 1.59384
    if Q.dr0_7 >= 0.158:
        z += -0.394 * Q.dr0_7 + 0.062252
    if Q.dr_1 >= 0.13:
        z += 1.75 * Q.dr_1 - 0.2275
    if Q.dr_2 >= 0.135:
        z += 2.04 * Q.dr_2 - 0.2754
    if Q.e3 < 0.000194:
        z += -3870.0 * Q.e3 + 0.75078
    if Q.e4 < 3.59e-09:
        z += -1.21e+08 * Q.e4 + 0.43439
    if Q.girth < 0.122:
        z += 72.5 * Q.girth - 8.845
    if 0.123 <= Q.girth < 0.155:
        z += 88.7 * Q.girth - 10.9101
    if Q.girth >= 0.155:
        z += 121.0 * Q.girth - 15.9166
    if Q.girth2_top3 >= 0.00556:
        z += -27.0 * Q.girth2_top3 + 0.15012
    if Q.lam1 < 0.00455:
        z += 847.6 * Q.lam1 - 4.76336
    if 0.00455 <= Q.lam1 < 0.0236:
        z += 47.6 * Q.lam1 - 1.12336
    if Q.lam2 >= 0.000244:
        z += 109.0 * Q.lam2 - 0.026596
    if Q.log_sum_pt >= 6.68:
        z += 8.77 * Q.log_sum_pt - 58.5836
    if Q.max_dr >= 0.0391:
        z += 5.53 * Q.max_dr - 0.216223
    if Q.mean_phi < -0.0185:
        z += 12.5 * Q.mean_phi + 0.23125
    if Q.n_dr_0_0p05 >= 5.69:
        z += 0.182 * Q.n_dr_0_0p05 - 1.03558
    if Q.n_dr_0p2_0p4 >= 0.994:
        z += 0.696 * Q.n_dr_0p2_0p4 - 0.691824
    if Q.psi_0p2 < 0.917:
        z += 1.81 * Q.psi_0p2 - 1.65977
    if Q.pt1_dr01 >= 19.6:
        z += 0.00113 * Q.pt1_dr01 - 0.022148
    if Q.pt_7 < 40.6:
        z += 0.038 * Q.pt_7 - 1.5428
    if Q.ptdr0_2 >= 17.3:
        z += 0.00775 * Q.ptdr0_2 - 0.134075
    if Q.ptdr0_5 >= 13.1:
        z += 0.0116 * Q.ptdr0_5 - 0.15196
    if Q.ptdr0_7 >= 9.41:
        z += 0.0283 * Q.ptdr0_7 - 0.266303
    if Q.sd_rg < 0.077:
        z += -4.57 * Q.sd_rg + 0.23345
    if 0.077 <= Q.sd_rg < 0.161:
        z += 1.41 * Q.sd_rg - 0.22701
    if 0.181 <= Q.sd_rg < 0.316:
        z += -1.25 * Q.sd_rg + 0.22625
    if Q.sd_rg >= 0.316:
        z += 27.25 * Q.sd_rg - 8.77975
    if Q.sj2_dr < 0.124:
        z += -0.98 * Q.sj2_dr - 0.30943
    if 0.124 <= Q.sj2_dr < 0.209:
        z += 5.07 * Q.sj2_dr - 1.05963
    if Q.sj3_dr_max >= 0.256:
        z += 2.21 * Q.sj3_dr_max - 0.56576
    if Q.sj3_dr_min < 0.151:
        z += -7.27 * Q.sj3_dr_min + 1.09777
    if Q.tau21_b2 < 0.0151:
        z += 70.2 * Q.tau21_b2 - 1.8873
    if 0.0151 <= Q.tau21_b2 < 0.0917:
        z += 10.8 * Q.tau21_b2 - 0.99036
    if Q.z_dr_0_0p05 >= 0.782:
        z += -5.82 * Q.z_dr_0_0p05 + 4.55124
    if Q.zdr_0 >= 0.046:
        z += -22.7 * Q.zdr_0 + 1.0442
    if Q.centroid_offset > 0.0246 and Q.sum_pt < 733.0:
        z += -0.0785 * (Q.centroid_offset - 0.0246) * (733.0 - Q.sum_pt)
    if Q.girth > 0.107 and Q.log_sum_pt > 6.52:
        z += 197.0 * (Q.girth - 0.107) * (Q.log_sum_pt - 6.52)
    if Q.girth > 0.144 and Q.log_sum_pt > 6.56:
        z += -169.0 * (Q.girth - 0.144) * (Q.log_sum_pt - 6.56)
    if Q.girth > 0.11 and Q.pt_6 < 65.7:
        z += -0.578 * (Q.girth - 0.11) * (65.7 - Q.pt_6)
    if Q.n_dr_0p2_0p4 > 1.6 and Q.lam2 < 0.00409:
        z += -89.7 * (Q.n_dr_0p2_0p4 - 1.6) * (0.00409 - Q.lam2)
    if Q.n_dr_0p2_0p4 > 1.53 and Q.sum_pt < 635.0:
        z += -0.00172 * (Q.n_dr_0p2_0p4 - 1.53) * (635.0 - Q.sum_pt)
    if Q.sd_rg > 0.321 and Q.log_sum_pt < 6.94:
        z += -79.9 * (Q.sd_rg - 0.321) * (6.94 - Q.log_sum_pt)
    if Q.sd_rg > 0.323 and Q.log_sum_pt < 6.77:
        z += 60.7 * (Q.sd_rg - 0.323) * (6.77 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_13(Q):
    z = 5.59
    if Q.LHA < 0.229:
        z += -14.4 * Q.LHA + 5.4288
    if 0.229 <= Q.LHA < 0.377:
        z += -24.7 * Q.LHA + 7.7875
    if Q.LHA >= 0.377:
        z += -10.3 * Q.LHA + 2.3587
    if Q.lam2 < 0.000711:
        z += 1560.0 * Q.lam2 - 1.10916
    if Q.mean_eta2 < 0.00496:
        z += -317.0 * Q.mean_eta2 + 1.62621
    if 0.00496 <= Q.mean_eta2 < 0.00513:
        z += -656.0 * Q.mean_eta2 + 3.30765
    if Q.mean_eta2 >= 0.00513:
        z += -339.0 * Q.mean_eta2 + 1.68144
    if Q.mean_phi2 < 0.00545:
        z += -343.0 * Q.mean_phi2 + 1.87621
    if 0.00545 <= Q.mean_phi2 < 0.00547:
        z += -669.0 * Q.mean_phi2 + 3.65291
    if Q.mean_phi2 >= 0.00547:
        z += -326.0 * Q.mean_phi2 + 1.7767
    if Q.pt_6 < 25.3:
        z += 0.17 * Q.pt_6 - 4.301
    if Q.pt_7 >= 28.1:
        z += -0.0533 * Q.pt_7 + 1.49773
    if Q.tau1 < 0.126:
        z += 21.6 * Q.tau1 - 2.7216
    if Q.z_7 >= 0.0441:
        z += 53.8 * Q.z_7 - 2.37258
    if Q.e3 < 8.02e-05 and Q.centroid_offset < 0.0338:
        z += -555000.0 * (8.02e-05 - Q.e3) * (0.0338 - Q.centroid_offset)
    if Q.girth < 0.162 and Q.log_sum_pt < 6.86:
        z += -72.7 * (0.162 - Q.girth) * (6.86 - Q.log_sum_pt)
    if Q.girth < 0.153 and Q.pt_7 < 34.9:
        z += -1.45 * (0.153 - Q.girth) * (34.9 - Q.pt_7)
    if Q.sum_pt_top5 > 658.0 and Q.pt_7 < 33.5:
        z += 0.000487 * (Q.sum_pt_top5 - 658.0) * (33.5 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = 1.12
    if Q.LHA < 0.253:
        z += -16.4 * Q.LHA + 4.1492
    if Q.LHA >= 0.318:
        z += 52.9 * Q.LHA - 16.8222
    if Q.e3 >= 3.14e-05:
        z += 3270.0 * Q.e3 - 0.102678
    z += -0.503 * Q.eccentricity
    if Q.girth < 0.0277:
        z += 134.0 * Q.girth - 10.8406
    if 0.0277 <= Q.girth < 0.0557:
        z += 101.4 * Q.girth - 9.93758
    if 0.0557 <= Q.girth < 0.0809:
        z += 81.3 * Q.girth - 8.81801
    if 0.0809 <= Q.girth < 0.0875:
        z += 27.9 * Q.girth - 4.49795
    if Q.girth >= 0.0875:
        z += -157.1 * Q.girth + 11.68955
    if Q.lam1 < 0.00372:
        z += 606.0 * Q.lam1 - 2.03841
    if 0.00372 <= Q.lam1 < 0.00693:
        z += 435.0 * Q.lam1 - 1.40229
    if 0.00693 <= Q.lam1 < 0.0117:
        z += -338.0 * Q.lam1 + 3.9546
    if Q.log_sum_pt >= 6.86:
        z += -5.1 * Q.log_sum_pt + 34.986
    if Q.max_dr >= 0.159:
        z += 8.23 * Q.max_dr - 1.30857
    if Q.mean_phi >= -0.0107:
        z += -6.03 * Q.mean_phi - 0.064521
    if Q.planar_flow < 0.13:
        z += -9.95 * Q.planar_flow + 1.2935
    if Q.pt_7 >= 32.9:
        z += -0.0146 * Q.pt_7 + 0.48034
    if Q.sd_rg < 0.25:
        z += -4.76 * Q.sd_rg + 1.19
    if Q.sj3_dr13 < 0.0963:
        z += -6.54 * Q.sj3_dr13 + 0.629802
    if 0.0963 <= Q.sj3_dr13 < 0.191:
        z += 5.72 * Q.sj3_dr13 - 0.550836
    if Q.sj3_dr13 >= 0.191:
        z += 0.03 * Q.sj3_dr13 + 0.535954
    if Q.sj3_dr23 < 0.112:
        z += -4.72 * Q.sj3_dr23 - 0.04584
    if 0.112 <= Q.sj3_dr23 < 0.198:
        z += 6.68 * Q.sj3_dr23 - 1.32264
    if Q.sj3_dr_max >= 0.23:
        z += -16.1 * Q.sj3_dr_max + 3.703
    if Q.sj3_dr_min < 0.121:
        z += 11.9 * Q.sj3_dr_min - 1.4399
    if Q.tau1 < 0.0529:
        z += -40.392 * Q.tau1 + 3.976377
    if 0.0529 <= Q.tau1 < 0.0981:
        z += -40.7 * Q.tau1 + 3.99267
    if Q.z_7 >= 0.0494:
        z += 6.86 * Q.z_7 - 0.338884
    if Q.lam1 < 0.00782 and Q.D2 < 0.987:
        z += -483.0 * (0.00782 - Q.lam1) * (0.987 - Q.D2)
    if Q.lam1 < 0.00757 and Q.centroid_offset > 0.0197:
        z += 33100.0 * (0.00757 - Q.lam1) * (Q.centroid_offset - 0.0197)
    if Q.lam1 < 0.0153 and Q.centroid_offset > 0.024:
        z += -12400.0 * (0.0153 - Q.lam1) * (Q.centroid_offset - 0.024)
    if Q.lam1 < 0.00753 and Q.sj3_dr23 > 0.186:
        z += 3120.0 * (0.00753 - Q.lam1) * (Q.sj3_dr23 - 0.186)
    if Q.lam2 < 0.00349 and Q.LHA > 0.171:
        z += 4320.0 * (0.00349 - Q.lam2) * (Q.LHA - 0.171)
    if Q.lam2 < 0.0036 and Q.centroid_offset > 0.0196:
        z += -7540.0 * (0.0036 - Q.lam2) * (Q.centroid_offset - 0.0196)
    if Q.lam2 < 0.00338 and Q.sd_rg > 0.191:
        z += -12600.0 * (0.00338 - Q.lam2) * (Q.sd_rg - 0.191)
    if Q.lam2 < 0.00353 and Q.sd_rg > 0.159:
        z += 10900.0 * (0.00353 - Q.lam2) * (Q.sd_rg - 0.159)
    if Q.planar_flow < 0.106 and Q.lam1 < 0.00664:
        z += 6690.0 * (0.106 - Q.planar_flow) * (0.00664 - Q.lam1)
    if Q.planar_flow < 0.113 and Q.lam1 < 0.00746:
        z += -8860.0 * (0.113 - Q.planar_flow) * (0.00746 - Q.lam1)
    if Q.planar_flow < 0.136 and Q.sj2_zsoft < 0.304:
        z += 30.5 * (0.136 - Q.planar_flow) * (0.304 - Q.sj2_zsoft)
    if Q.planar_flow < 0.152 and Q.sum_pt < 810.0:
        z += -0.0209 * (0.152 - Q.planar_flow) * (810.0 - Q.sum_pt)
    if Q.z_dr_0p05_0p1 > -0.0637 and Q.log_sum_pt < 6.7:
        z += -1.32 * (Q.z_dr_0p05_0p1 - -0.0637) * (6.7 - Q.log_sum_pt)
    if Q.z_dr_0p05_0p1 > 0.132 and Q.n_dr_0p2_0p4 < 1.29:
        z += 2.38 * (Q.z_dr_0p05_0p1 - 0.132) * (1.29 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p05_0p1 < 0.698 and Q.z_dr_0p1_0p2 < 0.0428:
        z += 17.2 * (0.698 - Q.z_dr_0p05_0p1) * (0.0428 - Q.z_dr_0p1_0p2)
    if Q.z_dr_0p05_0p1 > 0.166 and Q.z_dr_0p2_0p4 < 0.228:
        z += -13.2 * (Q.z_dr_0p05_0p1 - 0.166) * (0.228 - Q.z_dr_0p2_0p4)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.079
    if Q.LHA < 0.267:
        z += -23.9 * Q.LHA + 6.3813
    if Q.LHA >= 0.314:
        z += 57.3 * Q.LHA - 17.9922
    if Q.centroid_offset < 0.0212:
        z += 13.2 * Q.centroid_offset - 0.27984
    if Q.centroid_offset >= 0.0262:
        z += -27.0 * Q.centroid_offset + 0.7074
    if Q.dr_0 < 0.103:
        z += -8.64 * Q.dr_0 + 0.88992
    if Q.dr_1 < 0.0886:
        z += -6.23 * Q.dr_1 + 0.551978
    if Q.e2 >= 0.0254:
        z += -54.4 * Q.e2 + 1.38176
    if Q.eccentricity >= 0.619:
        z += -4.99 * Q.eccentricity + 3.08881
    if Q.girth < 0.0626:
        z += 163.4 * Q.girth - 12.42068
    if 0.0626 <= Q.girth < 0.0818:
        z += 82.4 * Q.girth - 7.35008
    if 0.0818 <= Q.girth < 0.0892:
        z += -61.6 * Q.girth + 4.42912
    if Q.girth >= 0.0892:
        z += -144.0 * Q.girth + 11.7792
    if Q.lam1 < 0.000228:
        z += 28856.0 * Q.lam1 - 8.23704
    if 0.000228 <= Q.lam1 < 0.000634:
        z += 2156.0 * Q.lam1 - 2.14944
    if 0.000634 <= Q.lam1 < 0.00738:
        z += 116.0 * Q.lam1 - 0.85608
    if Q.lam2 < 0.000866:
        z += -1130.0 * Q.lam2 + 3.9889
    if 0.000866 <= Q.lam2 < 0.00353:
        z += -904.0 * Q.lam2 + 3.793184
    if Q.lam2 >= 0.00353:
        z += 226.0 * Q.lam2 - 0.195716
    if Q.log_sum_pt >= 6.81:
        z += 7.17 * Q.log_sum_pt - 48.8277
    if Q.max_dr < 0.0417:
        z += -13.1 * Q.max_dr + 0.76242
    if 0.0417 <= Q.max_dr < 0.0582:
        z += -10.08 * Q.max_dr + 0.636486
    if Q.max_dr >= 0.0582:
        z += 3.02 * Q.max_dr - 0.125934
    if Q.mean_phi < 0.00818:
        z += -3.9 * Q.mean_phi + 0.031902
    if Q.sj3_dr_max < 0.175:
        z += -7.11 * Q.sj3_dr_max + 1.34379
    if 0.175 <= Q.sj3_dr_max < 0.189:
        z += 1.16 * Q.sj3_dr_max - 0.10346
    if 0.189 <= Q.sj3_dr_max < 0.238:
        z += 8.27 * Q.sj3_dr_max - 1.44725
    if Q.sj3_dr_max >= 0.238:
        z += -4.43 * Q.sj3_dr_max + 1.57535
    if Q.sum_pt < 480.0:
        z += -0.00183 * Q.sum_pt + 0.8784
    if Q.sum_pt_top5 >= 761.0:
        z += -0.00376 * Q.sum_pt_top5 + 2.86136
    if Q.tau1 < 0.0566:
        z += -42.7 * Q.tau1 + 2.41682
    if Q.tau1 >= 0.0979:
        z += 43.3 * Q.tau1 - 4.23907
    if Q.zdr_0 < 0.0192:
        z += 21.0 * Q.zdr_0 - 0.4347
    if 0.0192 <= Q.zdr_0 < 0.0207:
        z += 29.99 * Q.zdr_0 - 0.607308
    if Q.zdr_0 >= 0.0207:
        z += 8.99 * Q.zdr_0 - 0.172608
    if Q.N2 < 0.219 and Q.lam1 > 0.00775:
        z += -1490.0 * (0.219 - Q.N2) * (Q.lam1 - 0.00775)
    if Q.N2 < 0.248 and Q.sj2_dr < 0.162:
        z += 463.0 * (0.248 - Q.N2) * (0.162 - Q.sj2_dr)
    if Q.N2 < 0.25 and Q.sj2_dr < 0.186:
        z += -263.0 * (0.25 - Q.N2) * (0.186 - Q.sj2_dr)
    if Q.N2 < 0.21 and Q.sum_pt_top5 > 428.0:
        z += 0.0111 * (0.21 - Q.N2) * (Q.sum_pt_top5 - 428.0)
    if Q.N2 < 0.236 and Q.z_7 < 0.0306:
        z += -292.0 * (0.236 - Q.N2) * (0.0306 - Q.z_7)
    if Q.eccentricity > 0.894 and Q.sum_pt_top5 > 393.0:
        z += 0.0168 * (Q.eccentricity - 0.894) * (Q.sum_pt_top5 - 393.0)
    if Q.lam1 < 0.00821 and Q.D2 < 1.06:
        z += -894.0 * (0.00821 - Q.lam1) * (1.06 - Q.D2)
    if Q.lam1 < 0.0145 and Q.D2 < 1.07:
        z += 198.0 * (0.0145 - Q.lam1) * (1.07 - Q.D2)
    if Q.lam1 < 0.0142 and Q.centroid_offset > 0.053:
        z += -15200.0 * (0.0142 - Q.lam1) * (Q.centroid_offset - 0.053)
    if Q.lam1 < 0.00977 and Q.sj3_dr23 > 0.193:
        z += 817.0 * (0.00977 - Q.lam1) * (Q.sj3_dr23 - 0.193)
    if Q.lam2 < 0.00328 and Q.zdr_1 < 0.017:
        z += -8330.0 * (0.00328 - Q.lam2) * (0.017 - Q.zdr_1)
    if Q.tau1 > 0.103 and Q.D2_b2 < 0.65:
        z += 30.5 * (Q.tau1 - 0.103) * (0.65 - Q.D2_b2)
    if Q.z_dr_0p05_0p1 > 0.687 and Q.sj3_dr23 > 0.193:
        z += -38.9 * (Q.z_dr_0p05_0p1 - 0.687) * (Q.sj3_dr23 - 0.193)
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
