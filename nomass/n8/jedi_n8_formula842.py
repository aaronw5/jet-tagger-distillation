"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; no mass observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.9% (the network: 65.8%); same class as the network for 88.6% of jets.

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
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z1                 pT share of subjet 1 of 3 (by pT)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.pt1_over_pt0           pT1 / pT0
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.z_3rd                  3rd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_6               |Δη| of particle 6
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_2               |Δφ| of particle 2
  Q.absphi_5               |Δφ| of particle 5
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_5                  ΔR between particle 5 and the hardest particle
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_5                  Δη of particle 5
  Q.phi_0                  Δφ of particle 0
  Q.phi_1                  Δφ of particle 1
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
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
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
        max_dr=max(dr[i] for i in real),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        n_for_90pct=ncum(0.9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        pt_6=pt[6],
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_7=pt[7],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z1=subjets(3)["z"][0],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        z_3rd=zs[2],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        abseta_0=abs(eta[0]),
        abseta_4=abs(eta[4]),
        abseta_6=abs(eta[6]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_2=abs(phi[2]),
        absphi_5=abs(phi[5]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_5=math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_5=eta[5],
        phi_0=phi[0],
        phi_1=phi[1],
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
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = 7.230424
    if Q.planar_flow < 0.1484197:
        z += -6.159525 * Q.planar_flow + 0.9141949
    if Q.sum_z_dr2 < 0.005590289:
        z += 173.4112 * Q.sum_z_dr2 + 3.283982
    if 0.005590289 <= Q.sum_z_dr2 < 0.01323868:
        z += -420.7283 * Q.sum_z_dr2 + 6.605393
    if 0.01323868 <= Q.sum_z_dr2 < 0.01882765:
        z += -185.2768 * Q.sum_z_dr2 + 3.488327
    if Q.sum_pt < 739.5:
        z += 0.01030761 * Q.sum_pt - 7.867423
    if 739.5 <= Q.sum_pt < 788.4484:
        z += 0.005004211 * Q.sum_pt - 3.945562
    if Q.sum_z_dr2_top3 < 0.007929074:
        z += 126.2542 * Q.sum_z_dr2_top3 - 1.001079
    if Q.tau1 < 0.06345984:
        z += -23.17322 * Q.tau1 + 1.470569
    z += -1.167048 * Q.log_sum_pt
    if Q.e3 < 8.147744e-05:
        z += -8195.702 * Q.e3 + 0.6677648
    if Q.tau2 < 0.01357153:
        z += 43.28096 * Q.tau2 - 0.5873889
    if Q.sd_rg < 0.06545715:
        z += 2.528648 * Q.sd_rg - 0.2908623
    if 0.06545715 <= Q.sd_rg < 0.1116471:
        z += -16.62831 * Q.sd_rg + 0.9630976
    if 0.1116471 <= Q.sd_rg < 0.1773029:
        z += 13.60738 * Q.sd_rg - 2.412628
    if Q.lam1_plus_lam2 < 0.003562611:
        z += -226.3275 * Q.lam1_plus_lam2 + 5.033732
    if 0.003562611 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -826.404 * Q.lam1_plus_lam2 + 7.171571
    if Q.centroid_offset < 0.01837778:
        z += 24.4117 * Q.centroid_offset - 0.4486329
    if Q.centroid_offset >= 0.04990367:
        z += -526.0792 * Q.centroid_offset + 26.25328
    if Q.lam1 < 0.00595415:
        z += 510.3649 * Q.lam1 - 4.492353
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += 600.2935 * Q.lam1 - 5.027802
    if Q.eccentricity >= 0.9884745:
        z += -37.14067 * Q.eccentricity + 36.71261
    if Q.sum_pt_top5 < 631.275:
        z += -0.005970174 * Q.sum_pt_top5 + 3.768822
    if Q.n_dr_0_0p05 >= 5.0:
        z += 0.2874263 * Q.n_dr_0_0p05 - 1.437131
    if Q.sum_z_dr < 0.01517359:
        z += 221.7796 * Q.sum_z_dr - 8.221667
    if 0.01517359 <= Q.sum_z_dr < 0.07608178:
        z += 73.54107 * Q.sum_z_dr - 5.972356
    if 0.07608178 <= Q.sum_z_dr < 0.08723651:
        z += 33.81708 * Q.sum_z_dr - 2.950084
    if Q.sum_zz_dr2 < 0.00718279:
        z += 181.013 * Q.sum_zz_dr2 - 1.019773
    if 0.00718279 <= Q.sum_zz_dr2 < 0.01165737:
        z += -62.66615 * Q.sum_zz_dr2 + 0.7305227
    if Q.sum_z_dr2_top2 < 0.004007842:
        z += -101.607 * Q.sum_z_dr2_top2 + 0.4072249
    if Q.e2 < 0.032347:
        z += -41.12973 * Q.e2 + 1.339145
    if 0.032347 <= Q.e2 < 0.06344108:
        z += -0.2804822 * Q.e2 + 0.01779409
    if Q.pt_6 >= 35.28125:
        z += -0.01574449 * Q.pt_6 + 0.5554854
    if Q.z_7 < 0.02807091:
        z += -43.59203 * Q.z_7 + 1.223668
    if Q.planar_flow < 0.1484197 and Q.centroid_offset < 0.04990367:
        z += 118.766 * (0.1484197 - Q.planar_flow) * (0.04990367 - Q.centroid_offset)
    if Q.sum_pt < 788.4484 and Q.M2 > 0.02563286:
        z += 0.01677749 * (788.4484 - Q.sum_pt) * (Q.M2 - 0.02563286)
    if Q.sum_zz_dr2 < 0.006390125 and Q.ptdr0_2 > 3.351396:
        z += -16.98286 * (0.006390125 - Q.sum_zz_dr2) * (Q.ptdr0_2 - 3.351396)
    if Q.sum_z_dr2 < 0.01882765 and Q.M3 < 0.09029177:
        z += 993.6657 * (0.01882765 - Q.sum_z_dr2) * (0.09029177 - Q.M3)
    if Q.planar_flow < 0.1484197 and Q.D2_b2 < 1.129616:
        z += 2.642215 * (0.1484197 - Q.planar_flow) * (1.129616 - Q.D2_b2)
    if Q.e3 < 8.147744e-05 and Q.D2_b2 < 0.2669656:
        z += -50195.74 * (8.147744e-05 - Q.e3) * (0.2669656 - Q.D2_b2)
    if Q.sum_pt < 788.4484 and Q.D2_b2 < 0.380911:
        z += 0.003524606 * (788.4484 - Q.sum_pt) * (0.380911 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.01882765 and Q.centroid_offset > 0.009480685:
        z += -5131.883 * (0.01882765 - Q.sum_z_dr2) * (Q.centroid_offset - 0.009480685)
    if Q.sum_pt < 788.4484 and Q.eccentricity > 0.9598562:
        z += -0.09763388 * (788.4484 - Q.sum_pt) * (Q.eccentricity - 0.9598562)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.eccentricity > 0.9598562:
        z += -4310.249 * (0.004372139 - Q.lam1_plus_lam2) * (Q.eccentricity - 0.9598562)
    if Q.D2_b2 < 0.380911 and Q.pt_5 > 24.57812:
        z += -0.02938546 * (0.380911 - Q.D2_b2) * (Q.pt_5 - 24.57812)
    if Q.lam1_plus_lam2 < 0.003562611 and Q.dr0_7 > 0.1786203:
        z += 7648.349 * (0.003562611 - Q.lam1_plus_lam2) * (Q.dr0_7 - 0.1786203)
    if Q.e3 < 8.147744e-05 and Q.dr0_7 > 0.2405707:
        z += -124153.5 * (8.147744e-05 - Q.e3) * (Q.dr0_7 - 0.2405707)
    if Q.sum_pt < 739.5 and Q.dr0_6 > 0.1979161:
        z += -0.04041368 * (739.5 - Q.sum_pt) * (Q.dr0_6 - 0.1979161)
    if Q.lam1 < 0.008375572 and Q.phi_0 < 0.0402832:
        z += -763.6165 * (0.008375572 - Q.lam1) * (0.0402832 - Q.phi_0)
    if Q.eccentricity > 0.9884745 and Q.C2_b2 < 0.001109927:
        z += 62466.32 * (Q.eccentricity - 0.9884745) * (0.001109927 - Q.C2_b2)
    if Q.sum_pt < 739.5 and Q.D2_b2 < 1.345805:
        z += -0.007641654 * (739.5 - Q.sum_pt) * (1.345805 - Q.D2_b2)
    if Q.sum_pt_top5 < 631.275 and Q.D2_b2 < 1.345805:
        z += 0.004382983 * (631.275 - Q.sum_pt_top5) * (1.345805 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.005590289 and Q.M3 < 0.09310137:
        z += -2817.821 * (0.005590289 - Q.sum_z_dr2) * (0.09310137 - Q.M3)
    if Q.planar_flow < 0.1484197 and Q.M3 < 0.08753082:
        z += -84.48486 * (0.1484197 - Q.planar_flow) * (0.08753082 - Q.M3)
    if Q.lam1_plus_lam2 < 0.003562611 and Q.phi_1 < 0.0425415:
        z += -2452.974 * (0.003562611 - Q.lam1_plus_lam2) * (0.0425415 - Q.phi_1)
    if Q.sum_z_dr2 < 0.005590289 and Q.dr0_5 > 0.171998:
        z += 4008.55 * (0.005590289 - Q.sum_z_dr2) * (Q.dr0_5 - 0.171998)
    if Q.tau1 < 0.06345984 and Q.dr_5 > 0.1631114:
        z += -277.477 * (0.06345984 - Q.tau1) * (Q.dr_5 - 0.1631114)
    if Q.sum_pt < 739.5 and Q.mean_eta > 0.009367547:
        z += 0.06774453 * (739.5 - Q.sum_pt) * (Q.mean_eta - 0.009367547)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.768674
    if Q.lam1 < 0.002464291:
        z += -291.302 * Q.lam1 + 0.9700152
    if 0.002464291 <= Q.lam1 < 0.00595415:
        z += -159.631 * Q.lam1 + 0.6455394
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += 125.9291 * Q.lam1 - 1.054728
    if 6.377723 <= Q.log_sum_pt < 6.572938:
        z += 5.023978 * Q.log_sum_pt - 32.04154
    if 6.572938 <= Q.log_sum_pt < 6.638339:
        z += 10.74598 * Q.log_sum_pt - 69.65191
    if Q.log_sum_pt >= 6.638339:
        z += 11.80986 * Q.log_sum_pt - 76.71432
    if Q.sum_zz_dr2 < 0.005834489:
        z += -200.2995 * Q.sum_zz_dr2 + 1.168646
    if Q.z_7 < 0.05240025:
        z += 80.17553 * Q.z_7 - 4.728063
    if 0.05240025 <= Q.z_7 < 0.06164517:
        z += 56.98753 * Q.z_7 - 3.513006
    if Q.lam1_plus_lam2 < 0.008678045:
        z += 277.9797 * Q.lam1_plus_lam2 - 2.41232
    if 0.0001869378 <= Q.e3 < 0.0005116989:
        z += -1659.691 * Q.e3 + 0.3102591
    if Q.e3 >= 0.0005116989:
        z += -90.2832 * Q.e3 - 0.4928053
    if Q.tau1 < 0.07283629:
        z += -12.95155 * Q.tau1 + 1.251701
    if 0.07283629 <= Q.tau1 < 0.1027642:
        z += -10.30335 * Q.tau1 + 1.058815
    if 38.53125 <= Q.pt_7 < 53.4375:
        z += 0.09337308 * Q.pt_7 - 3.597782
    if Q.pt_7 >= 53.4375:
        z += 0.03670587 * Q.pt_7 - 0.5696272
    if Q.centroid_offset < 0.02076709:
        z += 25.76995 * Q.centroid_offset - 0.5351668
    if Q.zdr_0 < 0.0211821:
        z += 23.97023 * Q.zdr_0 - 0.5077399
    if Q.sum_z_dr < 0.07608178:
        z += -0.2284565 * Q.sum_z_dr - 0.56351
    if 0.07608178 <= Q.sum_z_dr < 0.1019409:
        z += 22.46367 * Q.sum_z_dr - 2.289967
    if Q.sum_z_dr2 < 0.00609665:
        z += 659.431 * Q.sum_z_dr2 - 4.02032
    if Q.sum_pt_top5 >= 531.1875:
        z += -0.01261023 * Q.sum_pt_top5 + 6.698398
    if Q.sj3_dr_max < 0.1594012:
        z += -10.93873 * Q.sj3_dr_max + 1.067412
    if 0.1594012 <= Q.sj3_dr_max < 0.3456459:
        z += 3.630892 * Q.sj3_dr_max - 1.255003
    if Q.max_dr < 0.1326115:
        z += 6.943989 * Q.max_dr - 0.9208531
    if Q.sum_pt < 667.0063:
        z += 0.0009577723 * Q.sum_pt - 0.9466696
    if 667.0063 <= Q.sum_pt < 988.4078:
        z += 0.009582984 * Q.sum_pt - 6.69974
    if Q.sum_pt >= 988.4078:
        z += 0.008625212 * Q.sum_pt - 5.75307
    if Q.pt_6 >= 62.25:
        z += -0.03549641 * Q.pt_6 + 2.209651
    if Q.z_6 >= 0.06727211:
        z += 12.13766 * Q.z_6 - 0.8165261
    if Q.sum_z_dr2_top2 < 0.01403324:
        z += -28.13004 * Q.sum_z_dr2_top2 + 0.3947556
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += 13400.65 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.pt_7 > 34.53125 and Q.e3 < 2.955458e-05:
        z += -1779.78 * (Q.pt_7 - 34.53125) * (2.955458e-05 - Q.e3)
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += 1.978721 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.e3 < 0.0001869378:
        z += 204051.0 * (0.06164517 - Q.z_7) * (0.0001869378 - Q.e3)
    if Q.log_sum_pt > 6.377723 and Q.z_dr_0_0p05 < 0.7681386:
        z += 2.498302 * (Q.log_sum_pt - 6.377723) * (0.7681386 - Q.z_dr_0_0p05)
    if Q.z_7 < 0.06164517 and Q.z_dr_0p05_0p1 > 0.04875823:
        z += -23.51296 * (0.06164517 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.04875823)
    if Q.lam1_plus_lam2 < 0.008678045 and Q.planar_flow < 0.1484197:
        z += -658.5777 * (0.008678045 - Q.lam1_plus_lam2) * (0.1484197 - Q.planar_flow)
    if Q.log_sum_pt > 6.572938 and Q.zdr_6 < 0.008654951:
        z += -137.6491 * (Q.log_sum_pt - 6.572938) * (0.008654951 - Q.zdr_6)
    if Q.tau1 < 0.1027642 and Q.mean_phi > -0.01753483:
        z += -79.7434 * (0.1027642 - Q.tau1) * (Q.mean_phi - -0.01753483)
    if Q.z_7 < 0.05240025 and Q.D2_b2 < 0.1236856:
        z += -246.5031 * (0.05240025 - Q.z_7) * (0.1236856 - Q.D2_b2)
    if Q.log_sum_pt > 6.572938 and Q.abseta_4 < 0.06994629:
        z += -13.79783 * (Q.log_sum_pt - 6.572938) * (0.06994629 - Q.abseta_4)
    if Q.pt_7 > 34.53125 and Q.zdr_1 > 0.005533857:
        z += 1.829887 * (Q.pt_7 - 34.53125) * (Q.zdr_1 - 0.005533857)
    if Q.z_7 < 0.06164517 and Q.sum_z_dr2_top2 < 0.01403324:
        z += 2500.509 * (0.06164517 - Q.z_7) * (0.01403324 - Q.sum_z_dr2_top2)
    if Q.pt_7 > 53.4375 and Q.zdr_1 > 0.001590562:
        z += -4.979669 * (Q.pt_7 - 53.4375) * (Q.zdr_1 - 0.001590562)
    if Q.log_sum_pt > 6.377723 and Q.dr_0 < 0.08082334:
        z += -56.26824 * (Q.log_sum_pt - 6.377723) * (0.08082334 - Q.dr_0)
    if Q.lam1 < 0.008375572 and Q.dr_0 < 0.1442881:
        z += 2511.286 * (0.008375572 - Q.lam1) * (0.1442881 - Q.dr_0)
    if Q.pt_7 > 34.53125 and Q.pt_3 < 77.625:
        z += -0.001301021 * (Q.pt_7 - 34.53125) * (77.625 - Q.pt_3)
    if Q.z_7 < 0.06164517 and Q.zdr_0 < 0.03981924:
        z += 643.6302 * (0.06164517 - Q.z_7) * (0.03981924 - Q.zdr_0)
    if Q.log_sum_pt > 6.572938 and Q.sum_z_dr2_top3 < 0.007929074:
        z += -446.145 * (Q.log_sum_pt - 6.572938) * (0.007929074 - Q.sum_z_dr2_top3)
    if Q.lam1 < 0.008375572 and Q.pt_5 > 24.57812:
        z += -3.739805 * (0.008375572 - Q.lam1) * (Q.pt_5 - 24.57812)
    if Q.log_sum_pt > 6.638339 and Q.tau21_b2 < 0.01817981:
        z += 178.1362 * (Q.log_sum_pt - 6.638339) * (0.01817981 - Q.tau21_b2)
    if Q.lam2 < 0.003408389 and Q.tau21_b2 < 0.2691019:
        z += 490.1337 * (0.003408389 - Q.lam2) * (0.2691019 - Q.tau21_b2)
    if Q.log_sum_pt > 6.638339 and Q.dr01 < 0.1410336:
        z += 14.34667 * (Q.log_sum_pt - 6.638339) * (0.1410336 - Q.dr01)
    if Q.z_7 < 0.06164517 and Q.dr01 < 0.1410336:
        z += -110.4051 * (0.06164517 - Q.z_7) * (0.1410336 - Q.dr01)
    if Q.pt_7 > 38.53125 and Q.zdr_7 < 0.007782684:
        z += -4.604554 * (Q.pt_7 - 38.53125) * (0.007782684 - Q.zdr_7)
    if Q.lam1 < 0.008375572 and Q.n_pt_above_50 > 6.0:
        z += -25.64887 * (0.008375572 - Q.lam1) * (Q.n_pt_above_50 - 6.0)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.8114293
    if Q.pt_7 < 20.125:
        z += 0.08774997 * Q.pt_7 - 2.723764
    if 20.125 <= Q.pt_7 < 53.4375:
        z += 0.1927262 * Q.pt_7 - 4.836412
    if Q.pt_7 >= 53.4375:
        z += 0.1022203 * Q.pt_7
    if Q.log_sum_pt < 6.605974:
        z += -1.357997 * Q.log_sum_pt + 9.160935
    if 6.605974 <= Q.log_sum_pt < 6.701242:
        z += -1.994824 * Q.log_sum_pt + 13.3678
    if Q.log_sum_pt >= 6.896095:
        z += 27.31761 * Q.log_sum_pt - 188.3848
    if 0.02320757 <= Q.z_7 < 0.02807091:
        z += -109.9549 * Q.z_7 + 2.551786
    if 0.02807091 <= Q.z_7 < 0.05557716:
        z += -102.9365 * Q.z_7 + 2.354772
    if Q.z_7 >= 0.05557716:
        z += -87.69182 * Q.z_7 + 1.507518
    if Q.sum_z_dr < 0.007673833:
        z += 241.2054 * Q.sum_z_dr - 1.85097
    if 0.09323897 <= Q.LHA < 0.111565:
        z += 6.979116 * Q.LHA - 0.6507257
    if Q.LHA >= 0.111565:
        z += -6.495378 * Q.LHA + 0.8525569
    if Q.e2 < 0.04447357:
        z += 8.064274 * Q.e2 - 0.358647
    if Q.sum_pt < 615.875:
        z += -0.01899872 * Q.sum_pt + 14.67838
    if 615.875 <= Q.sum_pt < 763.825:
        z += -0.014581 * Q.sum_pt + 11.95761
    if 763.825 <= Q.sum_pt < 840.0195:
        z += -0.0107656 * Q.sum_pt + 9.043315
    if 937.0312 <= Q.sum_pt < 988.4078:
        z += 0.01559928 * Q.sum_pt - 14.61701
    if Q.sum_pt >= 988.4078:
        z += -0.01140957 * Q.sum_pt + 12.07874
    if Q.C2 < 0.0423228:
        z += -40.39576 * Q.C2 + 1.709662
    if Q.sum_zz_dr2 < 0.001101266:
        z += 815.1952 * Q.sum_zz_dr2 - 2.883804
    if 0.001101266 <= Q.sum_zz_dr2 < 0.0030133:
        z += 499.8238 * Q.sum_zz_dr2 - 2.536496
    if 0.0030133 <= Q.sum_zz_dr2 < 0.00718279:
        z += 247.1231 * Q.sum_zz_dr2 - 1.775033
    if Q.lam1 < 0.003377388:
        z += -198.1057 * Q.lam1 + 0.6690797
    if Q.lam1_plus_lam2 < 0.001653836:
        z += -716.2243 * Q.lam1_plus_lam2 + 3.566489
    if 0.001653836 <= Q.lam1_plus_lam2 < 0.007520088:
        z += -285.4519 * Q.lam1_plus_lam2 + 2.854062
    if 0.007520088 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -123.7086 * Q.lam1_plus_lam2 + 1.637737
    if Q.sum_pt_top5 < 367.5938:
        z += -0.004741392 * Q.sum_pt_top5 + 0.613417
    if 367.5938 <= Q.sum_pt_top5 < 716.8828:
        z += 0.00323368 * Q.sum_pt_top5 - 2.318169
    if Q.sum_pt_top5 >= 791.125:
        z += -0.002957557 * Q.sum_pt_top5 + 2.339797
    if Q.sum_z_dr2 < 0.005590289:
        z += -222.2195 * Q.sum_z_dr2 + 1.242271
    if Q.zdr_0 < 0.02383244:
        z += -12.912 * Q.zdr_0 + 0.3077245
    if Q.sj3_dr_min < 0.1278212:
        z += 3.003335 * Q.sj3_dr_min - 0.3838898
    if Q.sj2_dr < 0.1294903:
        z += 3.97759 * Q.sj2_dr - 0.5150594
    if Q.D2_b2 < 0.08499387:
        z += 5.049176 * Q.D2_b2 - 0.429149
    if Q.z_7 > 0.02807091 and Q.centroid_offset < 0.04990367:
        z += 139.8061 * (Q.z_7 - 0.02807091) * (0.04990367 - Q.centroid_offset)
    if Q.lam1 < 0.00733008 and Q.max_dr < 0.2507612:
        z += 1031.892 * (0.00733008 - Q.lam1) * (0.2507612 - Q.max_dr)
    if Q.lam1 < 0.00733008 and Q.pt_6 < 62.25:
        z += -1.912792 * (0.00733008 - Q.lam1) * (62.25 - Q.pt_6)
    if Q.sum_z_dr < 0.007673833 and Q.pt_4 < 75.625:
        z += 6.290644 * (0.007673833 - Q.sum_z_dr) * (75.625 - Q.pt_4)
    if Q.pt_7 < 53.4375 and Q.D2_b2 < 0.08499387:
        z += 0.371846 * (53.4375 - Q.pt_7) * (0.08499387 - Q.D2_b2)
    if Q.sum_pt_top5 > 791.125 and Q.D2_b2 < 1.129616:
        z += 0.007759929 * (Q.sum_pt_top5 - 791.125) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 1.59265:
        z += -7.952034 * (Q.log_sum_pt - 6.896095) * (1.59265 - Q.D2_b2)
    if Q.log_sum_pt < 6.605974 and Q.D2_b2 < 1.129616:
        z += 7.12566 * (6.605974 - Q.log_sum_pt) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt < 6.701242 and Q.D2_b2 < 1.129616:
        z += -5.633862 * (6.701242 - Q.log_sum_pt) * (1.129616 - Q.D2_b2)
    if Q.sum_pt > 988.4078 and Q.ptdr0_2 < 0.9419776:
        z += 0.01212372 * (Q.sum_pt - 988.4078) * (0.9419776 - Q.ptdr0_2)
    if Q.centroid_offset < 0.005576073 and Q.C3 < 0.01685631:
        z += -12062.1 * (0.005576073 - Q.centroid_offset) * (0.01685631 - Q.C3)
    if Q.sum_pt_top5 > 791.125 and Q.abseta_0 < 0.01425934:
        z += -0.6356483 * (Q.sum_pt_top5 - 791.125) * (0.01425934 - Q.abseta_0)
    if Q.sum_pt > 988.4078 and Q.abseta_0 < 0.01425934:
        z += 0.8030839 * (Q.sum_pt - 988.4078) * (0.01425934 - Q.abseta_0)
    if Q.C2 < 0.0423228 and Q.C2_b2 < 0.02415398:
        z += -1946.631 * (0.0423228 - Q.C2) * (0.02415398 - Q.C2_b2)
    if Q.log_sum_pt > 6.896095 and Q.ptdr0_2 < 0.9419776:
        z += -14.18356 * (Q.log_sum_pt - 6.896095) * (0.9419776 - Q.ptdr0_2)
    if Q.z_6 < 0.06081235 and Q.absphi_2 < 0.07720947:
        z += -91.88241 * (0.06081235 - Q.z_6) * (0.07720947 - Q.absphi_2)
    if Q.sum_pt > 937.0312 and Q.n_pt_above_50 > 5.0:
        z += -0.002758055 * (Q.sum_pt - 937.0312) * (Q.n_pt_above_50 - 5.0)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.8060511
    if 0.06663269 <= Q.sum_z_dr < 0.08723651:
        z += 19.37601 * Q.sum_z_dr - 1.291076
    if Q.sum_z_dr >= 0.08723651:
        z += 44.96354 * Q.sum_z_dr - 3.523242
    if 0.1591713 <= Q.sj2_dr < 0.1682655:
        z += -12.01352 * Q.sj2_dr + 1.912208
    if 0.1682655 <= Q.sj2_dr < 0.1778793:
        z += -7.004051 * Q.sj2_dr + 1.069287
    if 0.1778793 <= Q.sj2_dr < 0.2687922:
        z += 11.16685 * Q.sj2_dr - 2.16294
    if 0.2687922 <= Q.sj2_dr < 0.3003793:
        z += -9.762999 * Q.sj2_dr + 3.462838
    if Q.sj2_dr >= 0.3003793:
        z += 0.2073069 * Q.sj2_dr + 0.467965
    if 0.00609665 <= Q.lam1_plus_lam2 < 0.01323868:
        z += 556.1638 * Q.lam1_plus_lam2 - 3.390736
    if 0.01323868 <= Q.lam1_plus_lam2 < 0.01882765:
        z += 315.0478 * Q.lam1_plus_lam2 - 0.1986798
    if Q.lam1_plus_lam2 >= 0.01882765:
        z += 384.7833 * Q.lam1_plus_lam2 - 1.511635
    if 0.006390125 <= Q.sum_zz_dr2 < 0.00718279:
        z += -529.132 * Q.sum_zz_dr2 + 3.381219
    if 0.00718279 <= Q.sum_zz_dr2 < 0.01165737:
        z += 309.6855 * Q.sum_zz_dr2 - 2.64383
    if Q.sum_zz_dr2 >= 0.01165737:
        z += 46.96777 * Q.sum_zz_dr2 + 0.4187687
    if Q.mean_eta < -0.006779839:
        z += -12.20181 * Q.mean_eta - 0.08272629
    if Q.sum_z_dr2 >= 0.008678045:
        z += -911.9655 * Q.sum_z_dr2 + 7.914077
    if 0.0811449 <= Q.tau1 < 0.1136369:
        z += -10.55798 * Q.tau1 + 0.856726
    if Q.tau1 >= 0.1136369:
        z += 10.76054 * Q.tau1 - 1.565844
    if 0.002464291 <= Q.lam1 < 0.008375572:
        z += 71.04636 * Q.lam1 - 0.1750789
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += 526.6447 * Q.lam1 - 3.990975
    if Q.lam1 >= 0.01200373:
        z += 407.3087 * Q.lam1 - 2.558499
    if Q.lam2 >= 0.000537286:
        z += 342.5208 * Q.lam2 - 0.1840316
    if Q.max_dr >= 0.1117619:
        z += 3.662292 * Q.max_dr - 0.4093046
    if Q.sj2_dr > 0.1682655 and Q.sum_pt < 988.4078:
        z += 0.01080791 * (Q.sj2_dr - 0.1682655) * (988.4078 - Q.sum_pt)
    if Q.sj2_dr > 0.1682655 and Q.tau2 < 0.06297984:
        z += 77.26587 * (Q.sj2_dr - 0.1682655) * (0.06297984 - Q.tau2)
    if Q.lam1_plus_lam2 > 0.00609665 and Q.n_pt_above_50 > 3.0:
        z += -19.6967 * (Q.lam1_plus_lam2 - 0.00609665) * (Q.n_pt_above_50 - 3.0)
    if Q.sj2_dr > 0.1682655 and Q.n_dr_0p05_0p1 < 7.0:
        z += -1.320596 * (Q.sj2_dr - 0.1682655) * (7.0 - Q.n_dr_0p05_0p1)
    if Q.lam1_plus_lam2 > 0.00609665 and Q.pt_6 > 31.90625:
        z += -1.189959 * (Q.lam1_plus_lam2 - 0.00609665) * (Q.pt_6 - 31.90625)
    if Q.sum_z_dr > 0.08723651 and Q.lam2 < 0.0003061234:
        z += 131374.0 * (Q.sum_z_dr - 0.08723651) * (0.0003061234 - Q.lam2)
    if Q.lam1_plus_lam2 > 0.00609665 and Q.log_sum_pt > 6.192222:
        z += -210.1728 * (Q.lam1_plus_lam2 - 0.00609665) * (Q.log_sum_pt - 6.192222)
    if Q.mean_eta < -0.006779839 and Q.log_sum_pt < 6.267538:
        z += 54.40585 * (-0.006779839 - Q.mean_eta) * (6.267538 - Q.log_sum_pt)
    if Q.sj2_dr > 0.1591713 and Q.n_dr_0p1_0p2 > 1.0:
        z += 0.6354028 * (Q.sj2_dr - 0.1591713) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.mean_eta < -0.006779839 and Q.eta_0 < -0.03967285:
        z += -182.8953 * (-0.006779839 - Q.mean_eta) * (-0.03967285 - Q.eta_0)
    if Q.sj2_dr > 0.3003793 and Q.z_dr_0p05_0p1 > 0.08878489:
        z += -15.3956 * (Q.sj2_dr - 0.3003793) * (Q.z_dr_0p05_0p1 - 0.08878489)
    if Q.lam1_plus_lam2 > 0.01323868 and Q.eccentricity > 0.9884745:
        z += -12818.36 * (Q.lam1_plus_lam2 - 0.01323868) * (Q.eccentricity - 0.9884745)
    if Q.lam1_plus_lam2 > 0.01882765 and Q.n_pt_above_50 > 3.0:
        z += 26.32486 * (Q.lam1_plus_lam2 - 0.01882765) * (Q.n_pt_above_50 - 3.0)
    if Q.sj2_dr > 0.1778793 and Q.eccentricity > 0.9458207:
        z += 477.5001 * (Q.sj2_dr - 0.1778793) * (Q.eccentricity - 0.9458207)
    if Q.sj2_dr > 0.1682655 and Q.eccentricity > 0.9458207:
        z += -372.3066 * (Q.sj2_dr - 0.1682655) * (Q.eccentricity - 0.9458207)
    if Q.sj2_dr > 0.1778793 and Q.sum_z_dr2_top2 < 0.001864148:
        z += -5039.001 * (Q.sj2_dr - 0.1778793) * (0.001864148 - Q.sum_z_dr2_top2)
    if Q.sj2_dr > 0.2687922 and Q.sum_z_dr2_top2 < 0.007639643:
        z += 1087.305 * (Q.sj2_dr - 0.2687922) * (0.007639643 - Q.sum_z_dr2_top2)
    if Q.sum_zz_dr2 > 0.01165737 and Q.z_7 < 0.07148865:
        z += 1828.829 * (Q.sum_zz_dr2 - 0.01165737) * (0.07148865 - Q.z_7)
    return max(0.0, z)


def neuron_4(Q):
    z = -1.343607
    if Q.N2 < 0.2233283:
        z += -38.93449 * Q.N2 + 8.695171
    if Q.sum_z_dr2 < 0.002635418:
        z += 1517.561 * Q.sum_z_dr2 - 19.11969
    if 0.002635418 <= Q.sum_z_dr2 < 0.006679471:
        z += 2237.226 * Q.sum_z_dr2 - 21.01631
    if 0.006679471 <= Q.sum_z_dr2 < 0.008678045:
        z += 1570.863 * Q.sum_z_dr2 - 16.56535
    if 0.008678045 <= Q.sum_z_dr2 < 0.01323868:
        z += 643.1863 * Q.sum_z_dr2 - 8.514935
    if Q.e3 < 1.340118e-05:
        z += 33148.49 * Q.e3 + 2.8308
    if 1.340118e-05 <= Q.e3 < 0.0001869378:
        z += -18872.26 * Q.e3 + 3.52794
    if 0.2179769 <= Q.sj2_dr < 0.2414351:
        z += -26.33422 * Q.sj2_dr + 5.740251
    if Q.sj2_dr >= 0.2414351:
        z += -15.17199 * Q.sj2_dr + 3.045297
    if Q.sum_z_dr2_top2 < 0.006299534:
        z += -97.97345 * Q.sum_z_dr2_top2 + 0.6171871
    if Q.sj3_dr_min < 0.2089872:
        z += -3.92549 * Q.sj3_dr_min + 0.8203773
    if Q.lam2 < 0.001130645:
        z += 2896.989 * Q.lam2 - 3.275465
    if Q.lam1 < 0.002464291:
        z += -779.6208 * Q.lam1 + 3.694029
    if 0.002464291 <= Q.lam1 < 0.01200373:
        z += -185.8409 * Q.lam1 + 2.230783
    if Q.C2_b2 < 0.009032972:
        z += -314.26 * Q.C2_b2 + 4.853377
    if 0.009032972 <= Q.C2_b2 < 0.02415398:
        z += -133.2368 * Q.C2_b2 + 3.2182
    if Q.max_dr >= 0.0931108:
        z += 16.65764 * Q.max_dr - 1.551006
    if Q.sj3_dr_max < 0.1789613:
        z += -1.526924 * Q.sj3_dr_max + 0.4599121
    if 0.1789613 <= Q.sj3_dr_max < 0.213399:
        z += 9.972003 * Q.sj3_dr_max - 1.597951
    if 0.213399 <= Q.sj3_dr_max < 0.3012016:
        z += -4.655497 * Q.sj3_dr_max + 1.523542
    if 0.3012016 <= Q.sj3_dr_max < 0.3456459:
        z += -3.128572 * Q.sj3_dr_max + 1.06363
    if Q.sj3_dr_max >= 0.3456459:
        z += -29.22373 * Q.sj3_dr_max + 10.08332
    if Q.sum_z_dr2_top5 < 0.007164202:
        z += -182.9102 * Q.sum_z_dr2_top5 + 1.310406
    if Q.sum_z_dr < 0.07608178:
        z += -24.60103 * Q.sum_z_dr + 1.87169
    if Q.tau2 < 0.01357153:
        z += 62.28218 * Q.tau2 - 0.8452645
    if Q.e2 < 0.02045966:
        z += 87.42318 * Q.e2 - 3.888021
    if 0.02045966 <= Q.e2 < 0.04447357:
        z += 203.4523 * Q.e2 - 6.261937
    if Q.e2 >= 0.04447357:
        z += 116.0291 * Q.e2 - 2.373916
    if Q.sum_zz_dr2 < 0.0030133:
        z += -1056.237 * Q.sum_zz_dr2 + 10.94041
    if 0.0030133 <= Q.sum_zz_dr2 < 0.006390125:
        z += -1461.845 * Q.sum_zz_dr2 + 12.16263
    if 0.006390125 <= Q.sum_zz_dr2 < 0.01165737:
        z += -1201.264 * Q.sum_zz_dr2 + 10.49748
    if Q.sum_zz_dr2 >= 0.01165737:
        z += -405.6079 * Q.sum_zz_dr2 + 1.222219
    if 0.02384388 <= Q.C2 < 0.06729223:
        z += -16.0715 * Q.C2 + 0.3832069
    if Q.C2 >= 0.06729223:
        z += -84.70598 * Q.C2 + 5.001774
    if Q.tau21 < 0.33555:
        z += 5.121534 * Q.tau21 - 1.718531
    if Q.log_sum_pt < 6.638339:
        z += 1.289017 * Q.log_sum_pt - 8.556929
    if Q.N2 < 0.2233283 and Q.sum_pt < 868.5094:
        z += 0.07283361 * (0.2233283 - Q.N2) * (868.5094 - Q.sum_pt)
    if Q.N2 < 0.2233283 and Q.centroid_offset > 0.02355416:
        z += -146.5465 * (0.2233283 - Q.N2) * (Q.centroid_offset - 0.02355416)
    if Q.N2 < 0.2233283 and Q.pt_7 < 53.4375:
        z += -0.33351 * (0.2233283 - Q.N2) * (53.4375 - Q.pt_7)
    if Q.N2 < 0.2233283 and Q.planar_flow < 0.4007947:
        z += -23.74717 * (0.2233283 - Q.N2) * (0.4007947 - Q.planar_flow)
    if Q.e3 < 0.0001869378 and Q.sum_pt < 813.4156:
        z += -40.23597 * (0.0001869378 - Q.e3) * (813.4156 - Q.sum_pt)
    if Q.N2 < 0.2233283 and Q.mean_phi < -0.02594505:
        z += 569.7744 * (0.2233283 - Q.N2) * (-0.02594505 - Q.mean_phi)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1626038:
        z += -27.89294 * (0.2233283 - Q.N2) * (0.1626038 - Q.abseta_7)
    if Q.sj3_dr_min < 0.2089872 and Q.mean_eta < -0.01790907:
        z += 309.9714 * (0.2089872 - Q.sj3_dr_min) * (-0.01790907 - Q.mean_eta)
    if Q.N2 < 0.2233283 and Q.pt_6 < 24.42188:
        z += -1.034683 * (0.2233283 - Q.N2) * (24.42188 - Q.pt_6)
    if Q.lam1 < 0.00483998 and Q.pt_entropy > 1.797618:
        z += 1337.074 * (0.00483998 - Q.lam1) * (Q.pt_entropy - 1.797618)
    if Q.sum_z_dr2_top2 < 0.006299534 and Q.centroid_offset > 0.01096064:
        z += 15421.36 * (0.006299534 - Q.sum_z_dr2_top2) * (Q.centroid_offset - 0.01096064)
    if Q.sj2_dr > 0.2414351 and Q.z_dr_0p05_0p1 > 0.5882598:
        z += -30.67498 * (Q.sj2_dr - 0.2414351) * (Q.z_dr_0p05_0p1 - 0.5882598)
    if Q.sj2_dr > 0.2414351 and Q.D2_b2 < 1.129616:
        z += -24.11398 * (Q.sj2_dr - 0.2414351) * (1.129616 - Q.D2_b2)
    if Q.e3 < 0.0001869378 and Q.mean_phi > 0.02612796:
        z += 234839.0 * (0.0001869378 - Q.e3) * (Q.mean_phi - 0.02612796)
    if Q.C2_b2 < 0.02415398 and Q.mean_phi < 0.02612796:
        z += 515.512 * (0.02415398 - Q.C2_b2) * (0.02612796 - Q.mean_phi)
    if Q.sj3_dr_max > 0.3456459 and Q.D2_b2 < 1.129616:
        z += 39.94408 * (Q.sj3_dr_max - 0.3456459) * (1.129616 - Q.D2_b2)
    if Q.lam2 < 0.001130645 and Q.phi_0 > -0.0297699:
        z += 2942.889 * (0.001130645 - Q.lam2) * (Q.phi_0 - -0.0297699)
    if Q.sum_z_dr2_top2 < 0.006299534 and Q.z_4 > 0.06095291:
        z += -1490.438 * (0.006299534 - Q.sum_z_dr2_top2) * (Q.z_4 - 0.06095291)
    if Q.max_dr > 0.0931108 and Q.absphi_0 < 0.04019165:
        z += 121.0713 * (Q.max_dr - 0.0931108) * (0.04019165 - Q.absphi_0)
    if Q.N2 < 0.2233283 and Q.n_dr_0p2_0p4 > 1.0:
        z += -1.632157 * (0.2233283 - Q.N2) * (Q.n_dr_0p2_0p4 - 1.0)
    if Q.e3 < 0.0001869378 and Q.ptdr0_6 > 11.6057:
        z += -1671.659 * (0.0001869378 - Q.e3) * (Q.ptdr0_6 - 11.6057)
    if Q.C2_b2 < 0.009032972 and Q.ptdr0_6 > 5.648151:
        z += 5.588049 * (0.009032972 - Q.C2_b2) * (Q.ptdr0_6 - 5.648151)
    if Q.sum_zz_dr2 < 0.01165737 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -4055.694 * (0.01165737 - Q.sum_zz_dr2) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sum_z_dr2 < 0.01323868 and Q.psi_0p2 > 0.9435576:
        z += 2255.358 * (0.01323868 - Q.sum_z_dr2) * (Q.psi_0p2 - 0.9435576)
    if Q.lam2 < 0.001130645 and Q.phi_1 > -0.0001367927:
        z += 8298.77 * (0.001130645 - Q.lam2) * (Q.phi_1 - -0.0001367927)
    if Q.e3 < 0.0001869378 and Q.phi_1 > 0.009028244:
        z += -52719.41 * (0.0001869378 - Q.e3) * (Q.phi_1 - 0.009028244)
    if Q.sum_z_dr2 < 0.01323868 and Q.dr_7 > 0.1242755:
        z += 349.6196 * (0.01323868 - Q.sum_z_dr2) * (Q.dr_7 - 0.1242755)
    if Q.sj3_dr_max < 0.3012016 and Q.dr_4 > 0.1100973:
        z += 74.06904 * (0.3012016 - Q.sj3_dr_max) * (Q.dr_4 - 0.1100973)
    if Q.sj3_dr_max < 0.3012016 and Q.z_dr_0p2_0p4 < 0.20552:
        z += -11.66003 * (0.3012016 - Q.sj3_dr_max) * (0.20552 - Q.z_dr_0p2_0p4)
    if Q.sj2_dr > 0.2414351 and Q.pt_2 < 154.25:
        z += 0.363342 * (Q.sj2_dr - 0.2414351) * (154.25 - Q.pt_2)
    if Q.sj2_dr > 0.2179769 and Q.z_3rd < 0.1818564:
        z += -223.1203 * (Q.sj2_dr - 0.2179769) * (0.1818564 - Q.z_3rd)
    if Q.sum_z_dr2 < 0.006679471 and Q.log_sum_pt < 6.701242:
        z += 1385.303 * (0.006679471 - Q.sum_z_dr2) * (6.701242 - Q.log_sum_pt)
    if Q.sj3_dr_max < 0.3012016 and Q.log_sum_pt < 6.733425:
        z += -42.03727 * (0.3012016 - Q.sj3_dr_max) * (6.733425 - Q.log_sum_pt)
    if Q.N2 < 0.2233283 and Q.sum_pt < 988.4078:
        z += -0.07788437 * (0.2233283 - Q.N2) * (988.4078 - Q.sum_pt)
    if Q.D2 < 0.7459513 and Q.pt_6 > 19.46875:
        z += 0.04052294 * (0.7459513 - Q.D2) * (Q.pt_6 - 19.46875)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.7171802
    if Q.LHA < 0.1767241:
        z += -24.34949 * Q.LHA + 5.45126
    if 0.1767241 <= Q.LHA < 0.2160559:
        z += -29.19065 * Q.LHA + 6.30681
    if Q.z_7 < 0.02320757:
        z += -203.8693 * Q.z_7 + 5.907879
    if 0.02320757 <= Q.z_7 < 0.02807091:
        z += -101.4834 * Q.z_7 + 3.53175
    if 0.02807091 <= Q.z_7 < 0.03243272:
        z += -80.62491 * Q.z_7 + 2.946233
    if 0.03243272 <= Q.z_7 < 0.04939969:
        z += -19.529 * Q.z_7 + 0.9647266
    if Q.log_sum_pt >= 6.896095:
        z += -5.831755 * Q.log_sum_pt + 40.21634
    if Q.z_6 < 0.02160287:
        z += -142.6075 * Q.z_6 + 4.24156
    if 0.02160287 <= Q.z_6 < 0.02886576:
        z += -72.74447 * Q.z_6 + 2.732317
    if 0.02886576 <= Q.z_6 < 0.06081235:
        z += -19.79843 * Q.z_6 + 1.203989
    if Q.tau1 < 0.03459477:
        z += 9.698329 * Q.tau1 - 0.3355115
    if Q.sum_pt < 715.4688:
        z += 0.009949934 * Q.sum_pt - 7.337992
    if 715.4688 <= Q.sum_pt < 788.4484:
        z += 0.003002543 * Q.sum_pt - 2.367351
    if Q.centroid_offset < 0.03117077:
        z += -20.36581 * Q.centroid_offset + 0.2977617
    if 0.03117077 <= Q.centroid_offset < 0.03776099:
        z += 51.14492 * Q.centroid_offset - 1.931283
    if Q.sum_z_dr < 0.007673833:
        z += -206.3652 * Q.sum_z_dr + 1.583612
    if Q.sum_pt_top5 < 430.75:
        z += 0.003499476 * Q.sum_pt_top5 - 1.507399
    if Q.dr_0 < 0.04649465:
        z += 32.76544 * Q.dr_0 - 1.523418
    if Q.sj3_dr_max < 0.3012016:
        z += 5.461616 * Q.sj3_dr_max - 1.645047
    if Q.max_dr < 0.177305:
        z += -4.353764 * Q.max_dr + 0.7719441
    if Q.e3 < 0.0005116989:
        z += -1150.305 * Q.e3 + 0.5886099
    if Q.pt_7 < 20.125:
        z += 0.1226811 * Q.pt_7 - 2.468957
    if 34.53125 <= Q.pt_7 < 43.5:
        z += -0.02387824 * Q.pt_7 + 0.8245456
    if 43.5 <= Q.pt_7 < 53.4375:
        z += -0.07699646 * Q.pt_7 + 3.135188
    if Q.pt_7 >= 53.4375:
        z += -0.006528152 * Q.pt_7 - 0.6304622
    if Q.sum_z_dr2 < 0.0009641429:
        z += -697.5696 * Q.sum_z_dr2 + 1.184671
    if 0.0009641429 <= Q.sum_z_dr2 < 0.005019719:
        z += -126.2741 * Q.sum_z_dr2 + 0.6338604
    if Q.pt_4 < 31.125:
        z += -0.06403972 * Q.pt_4 + 1.993236
    if Q.z_4 < 0.03747769:
        z += 33.14084 * Q.z_4 - 1.242042
    if Q.LHA < 0.2160559 and Q.z_6 < 0.09543973:
        z += -211.9036 * (0.2160559 - Q.LHA) * (0.09543973 - Q.z_6)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -83.91309 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.e2 < 0.03556091 and Q.z_7 < 0.0753896:
        z += 373.6112 * (0.03556091 - Q.e2) * (0.0753896 - Q.z_7)
    if Q.e2 < 0.03556091 and Q.zdr_0 > 0.004918231:
        z += 1651.391 * (0.03556091 - Q.e2) * (Q.zdr_0 - 0.004918231)
    if Q.e2 < 0.03556091 and Q.log_sum_pt > 6.896095:
        z += -153.0109 * (0.03556091 - Q.e2) * (Q.log_sum_pt - 6.896095)
    if Q.sum_z_dr2 < 0.002635418 and Q.centroid_offset < 0.01627885:
        z += 69379.81 * (0.002635418 - Q.sum_z_dr2) * (0.01627885 - Q.centroid_offset)
    if Q.z_6 < 0.02886576 and Q.zdr_0 > 0.004918231:
        z += -2832.454 * (0.02886576 - Q.z_6) * (Q.zdr_0 - 0.004918231)
    if Q.LHA < 0.1767241 and Q.zdr_0 < 0.006292091:
        z += -1081.278 * (0.1767241 - Q.LHA) * (0.006292091 - Q.zdr_0)
    if Q.sum_pt < 715.4688 and Q.zdr_0 < 0.007798268:
        z += -0.538902 * (715.4688 - Q.sum_pt) * (0.007798268 - Q.zdr_0)
    if Q.sum_z_dr2 < 0.002635418 and Q.pt_6 > 27.57812:
        z += -10.30969 * (0.002635418 - Q.sum_z_dr2) * (Q.pt_6 - 27.57812)
    if Q.sj3_dr_max < 0.3012016 and Q.pt1_dr01 > 12.6865:
        z += -0.2324026 * (0.3012016 - Q.sj3_dr_max) * (Q.pt1_dr01 - 12.6865)
    if Q.sj3_dr_max < 0.3012016 and Q.ptdr0_2 > 11.87288:
        z += -0.318981 * (0.3012016 - Q.sj3_dr_max) * (Q.ptdr0_2 - 11.87288)
    if Q.dr_0 < 0.04649465 and Q.N3 < 2.080881:
        z += 8.061815 * (0.04649465 - Q.dr_0) * (2.080881 - Q.N3)
    if Q.e2 < 0.03556091 and Q.planar_flow < 0.3220738:
        z += 34.92389 * (0.03556091 - Q.e2) * (0.3220738 - Q.planar_flow)
    if Q.z_7 < 0.04939969 and Q.absphi_0 < 0.04678345:
        z += 531.9711 * (0.04939969 - Q.z_7) * (0.04678345 - Q.absphi_0)
    if Q.sum_pt < 715.4688 and Q.dr_0 < 0.0361727:
        z += 0.4752046 * (715.4688 - Q.sum_pt) * (0.0361727 - Q.dr_0)
    if Q.centroid_offset < 0.03117077 and Q.z_5 < 0.09549375:
        z += -791.8351 * (0.03117077 - Q.centroid_offset) * (0.09549375 - Q.z_5)
    if Q.pt_5 < 24.57812 and Q.abseta_6 < 0.1583252:
        z += 0.633805 * (24.57812 - Q.pt_5) * (0.1583252 - Q.abseta_6)
    if Q.e3 < 0.0005116989 and Q.absphi_0 > 0.04678345:
        z += 12231.27 * (0.0005116989 - Q.e3) * (Q.absphi_0 - 0.04678345)
    if Q.sum_z_dr2 < 0.002635418 and Q.n_dr_0p2_0p4 < 2.0:
        z += 325.4051 * (0.002635418 - Q.sum_z_dr2) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.centroid_offset < 0.03117077 and Q.pt_5 > 29.875:
        z += -0.8193371 * (0.03117077 - Q.centroid_offset) * (Q.pt_5 - 29.875)
    if Q.sum_pt < 788.4484 and Q.n_for_90pct > 5.0:
        z += 0.0009608969 * (788.4484 - Q.sum_pt) * (Q.n_for_90pct - 5.0)
    if Q.e3 < 0.0005116989 and Q.n_dr_0p1_0p2 < 3.0:
        z += 167.828 * (0.0005116989 - Q.e3) * (3.0 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.901085
    if Q.lam1_plus_lam2 < 0.01323868:
        z += 185.8624 * Q.lam1_plus_lam2 - 2.460572
    if Q.tau1 < 0.1027642:
        z += -22.30528 * Q.tau1 + 2.134866
    if 0.1027642 <= Q.tau1 < 0.1136369:
        z += 14.46928 * Q.tau1 - 1.644244
    if Q.log_sum_pt < 6.327379:
        z += -6.935365 * Q.log_sum_pt + 44.41586
    if 6.327379 <= Q.log_sum_pt < 6.638339:
        z += -1.714608 * Q.log_sum_pt + 11.38215
    if Q.lam2 < 0.001130645:
        z += 198.0528 * Q.lam2 + 0.5600194
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -344.1768 * Q.lam2 + 1.173089
    if Q.lam1 < 0.008375572:
        z += -443.1025 * Q.lam1 + 3.912527
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -55.48007 * Q.lam1 + 0.6659676
    if Q.z_6 < 0.02160287:
        z += 402.4696 * Q.z_6 - 10.11949
    if 0.02160287 <= Q.z_6 < 0.03448406:
        z += 110.6261 * Q.z_6 - 3.814836
    if Q.sj3_dr_min >= 0.02022982:
        z += -8.152476 * Q.sj3_dr_min + 0.1649231
    if Q.centroid_offset >= 0.01627885:
        z += 74.92302 * Q.centroid_offset - 1.21966
    if Q.eccentricity >= 0.9031255:
        z += -2.109331 * Q.eccentricity + 1.904991
    if Q.e3 < 1.627072e-05:
        z += -24874.69 * Q.e3 + 0.4047292
    if Q.sum_z_dr2 < 0.008678045:
        z += 1173.736 * Q.sum_z_dr2 - 10.18574
    if 0.07264452 <= Q.sj3_dr_max < 0.1789613:
        z += -9.919416 * Q.sj3_dr_max + 0.7205913
    if 0.1789613 <= Q.sj3_dr_max < 0.3012016:
        z += 9.808891 * Q.sj3_dr_max - 2.810013
    if Q.sj3_dr_max >= 0.3012016:
        z += 3.117177 * Q.sj3_dr_max - 0.7944582
    if Q.sum_zz_dr2 < 0.008168571:
        z += -451.6332 * Q.sum_zz_dr2 + 3.689198
    if 0.01165737 <= Q.sum_zz_dr2 < 0.01716248:
        z += -144.4099 * Q.sum_zz_dr2 + 1.68344
    if Q.sum_zz_dr2 >= 0.01716248:
        z += -60.61118 * Q.sum_zz_dr2 + 0.2452464
    if Q.max_dr < 0.1598486:
        z += 14.65428 * Q.max_dr - 2.342466
    if Q.sum_z_dr < 0.02689598:
        z += -68.83509 * Q.sum_z_dr + 1.851387
    if Q.D2 < 1.332146:
        z += -0.4308594 * Q.D2 + 0.5739678
    if Q.pt_6 < 29.90625:
        z += -0.151723 * Q.pt_6 + 4.537465
    if Q.C3 < 0.03853752:
        z += -22.32264 * Q.C3 + 0.8602593
    if Q.LHA >= 0.266913:
        z += 2.943568 * Q.LHA - 0.7856765
    if Q.sum_z_dr2_top2 < 0.0095303:
        z += -43.35875 * Q.sum_z_dr2_top2 + 0.4132219
    if Q.D3 < 1.651088:
        z += 0.3527326 * Q.D3 - 0.5823925
    if Q.dr_1 < 0.1055393:
        z += 2.535476 * Q.dr_1 - 0.2675923
    if Q.centroid_offset > 0.00809236 and Q.sj2_dr < 0.1682655:
        z += 174.4749 * (Q.centroid_offset - 0.00809236) * (0.1682655 - Q.sj2_dr)
    if Q.log_sum_pt < 6.327379 and Q.pt_7 > 25.57812:
        z += -0.3945931 * (6.327379 - Q.log_sum_pt) * (Q.pt_7 - 25.57812)
    if Q.lam1 < 0.01200373 and Q.pt_6 < 19.46875:
        z += 25.4008 * (0.01200373 - Q.lam1) * (19.46875 - Q.pt_6)
    if Q.z_6 < 0.02160287 and Q.log_sum_pt < 6.842717:
        z += 1520.876 * (0.02160287 - Q.z_6) * (6.842717 - Q.log_sum_pt)
    if Q.log_sum_pt < 6.638339 and Q.z_7 < 0.0586137:
        z += 196.9513 * (6.638339 - Q.log_sum_pt) * (0.0586137 - Q.z_7)
    if Q.centroid_offset > 0.00809236 and Q.mean_phi2 < 0.01426135:
        z += 1388.53 * (Q.centroid_offset - 0.00809236) * (0.01426135 - Q.mean_phi2)
    if Q.sj3_dr_min > 0.02022982 and Q.sj3_pairmin_over_m < 0.4246762:
        z += 15.20885 * (Q.sj3_dr_min - 0.02022982) * (0.4246762 - Q.sj3_pairmin_over_m)
    if Q.centroid_offset > 0.01627885 and Q.sum_pt < 988.4078:
        z += -0.2286185 * (Q.centroid_offset - 0.01627885) * (988.4078 - Q.sum_pt)
    if Q.centroid_offset > 0.00809236 and Q.sum_pt < 715.4688:
        z += 0.1494761 * (Q.centroid_offset - 0.00809236) * (715.4688 - Q.sum_pt)
    if Q.log_sum_pt < 6.638339 and Q.n_dr_0p1_0p2 > 3.0:
        z += 0.2146638 * (6.638339 - Q.log_sum_pt) * (Q.n_dr_0p1_0p2 - 3.0)
    if Q.eccentricity > 0.9031255 and Q.D3 < 0.9827275:
        z += -5.078211 * (Q.eccentricity - 0.9031255) * (0.9827275 - Q.D3)
    if Q.sj3_dr_max > 0.1789613 and Q.n_dr_0p05_0p1 < 6.0:
        z += -0.6789034 * (Q.sj3_dr_max - 0.1789613) * (6.0 - Q.n_dr_0p05_0p1)
    if Q.max_dr < 0.1598486 and Q.ptdr0_4 > 8.939062:
        z += 1.610322 * (0.1598486 - Q.max_dr) * (Q.ptdr0_4 - 8.939062)
    if Q.sum_pt > 988.4078 and Q.dr01 > 0.1662967:
        z += -0.7539403 * (Q.sum_pt - 988.4078) * (Q.dr01 - 0.1662967)
    if Q.sum_pt > 988.4078 and Q.dr01 > 0.1410336:
        z += 0.6595205 * (Q.sum_pt - 988.4078) * (Q.dr01 - 0.1410336)
    if Q.sj3_dr_max > 0.1789613 and Q.eccentricity > 0.9031255:
        z += 138.9257 * (Q.sj3_dr_max - 0.1789613) * (Q.eccentricity - 0.9031255)
    if Q.centroid_offset > 0.02076709 and Q.dr_7 < 0.09397911:
        z += 409.3076 * (Q.centroid_offset - 0.02076709) * (0.09397911 - Q.dr_7)
    if Q.LHA > 0.266913 and Q.eccentricity > 0.9031255:
        z += 87.68782 * (Q.LHA - 0.266913) * (Q.eccentricity - 0.9031255)
    if Q.sum_z_dr2_top2 < 0.0095303 and Q.mean_eta < -0.02665591:
        z += 7422.723 * (0.0095303 - Q.sum_z_dr2_top2) * (-0.02665591 - Q.mean_eta)
    if Q.sj3_dr_max > 0.07264452 and Q.eccentricity > 0.9031255:
        z += -73.19749 * (Q.sj3_dr_max - 0.07264452) * (Q.eccentricity - 0.9031255)
    return max(0.0, z)


def neuron_7(Q):
    z = 10.16627
    if Q.sum_z_dr2_top2 < 0.001056655:
        z += -481.4246 * Q.sum_z_dr2_top2 + 0.5086998
    if Q.sum_z_dr2 < 0.0005611231:
        z += 984.2776 * Q.sum_z_dr2 - 0.5523009
    if 0.004372139 <= Q.sum_z_dr2 < 0.007520088:
        z += -531.4383 * Q.sum_z_dr2 + 2.323522
    if 0.007520088 <= Q.sum_z_dr2 < 0.008678045:
        z += -2404.988 * Q.sum_z_dr2 + 16.41278
    if 0.008678045 <= Q.sum_z_dr2 < 0.01323868:
        z += -1339.095 * Q.sum_z_dr2 + 7.162913
    if Q.sum_z_dr2 >= 0.01323868:
        z += 2.326965 * Q.sum_z_dr2 - 10.59573
    if Q.sum_zz_dr2 < 0.001101266:
        z += -702.8451 * Q.sum_zz_dr2 - 0.9282878
    if 0.001101266 <= Q.sum_zz_dr2 < 0.005284669:
        z += -455.8159 * Q.sum_zz_dr2 - 1.200333
    if 0.005284669 <= Q.sum_zz_dr2 < 0.00718279:
        z += 539.6286 * Q.sum_zz_dr2 - 6.460927
    if 0.00718279 <= Q.sum_zz_dr2 < 0.008168571:
        z += 1559.814 * Q.sum_zz_dr2 - 13.7887
    if 0.008168571 <= Q.sum_zz_dr2 < 0.01165737:
        z += 300.1756 * Q.sum_zz_dr2 - 3.499259
    if Q.sum_z_dr < 0.0479157:
        z += 4.918579 * Q.sum_z_dr - 2.094523
    if 0.0479157 <= Q.sum_z_dr < 0.08723651:
        z += 47.27383 * Q.sum_z_dr - 4.124004
    if Q.lam1_plus_lam2 < 0.005590289:
        z += 1330.253 * Q.lam1_plus_lam2 - 7.436498
    if Q.sj2_dr < 0.1294903:
        z += -1.564527 * Q.sj2_dr - 0.2735509
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -6.62727 * Q.sj2_dr + 0.3820253
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 23.95286 * Q.sj2_dr - 4.485454
    if Q.tau1 < 0.09538712:
        z += -6.230075 * Q.tau1 - 0.5639579
    if 0.09538712 <= Q.tau1 < 0.1136369:
        z += 57.06409 * Q.tau1 - 6.601406
    if 0.1136369 <= Q.tau1 < 0.1416226:
        z += 4.174275 * Q.tau1 - 0.5911718
    if Q.centroid_offset >= 0.009480685:
        z += -43.21825 * Q.centroid_offset + 0.4097386
    if Q.LHA < 0.2160559:
        z += 1.902838 * Q.LHA + 0.3092756
    if 0.2160559 <= Q.LHA < 0.2931906:
        z += -9.339432 * Q.LHA + 2.738234
    if Q.lam2 < 0.000537286:
        z += 241.6711 * Q.lam2 + 0.1890812
    if 0.000537286 <= Q.lam2 < 0.001130645:
        z += -537.4955 * Q.lam2 + 0.6077165
    if Q.e3 < 0.0001869378:
        z += 6059.241 * Q.e3 - 1.132701
    if Q.max_dr < 0.121681:
        z += 3.854096 * Q.max_dr + 0.06700527
    if 0.121681 <= Q.max_dr < 0.1979689:
        z += -7.025692 * Q.max_dr + 1.390868
    if Q.sj3_dr_max < 0.1426152:
        z += -11.58549 * Q.sj3_dr_max + 1.927491
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += 4.779633 * Q.sj3_dr_max - 0.4064229
    if 0.1986272 <= Q.sj3_dr_max < 0.213399:
        z += 16.1299 * Q.sj3_dr_max - 2.660895
    if 0.213399 <= Q.sj3_dr_max < 0.3456459:
        z += -5.907203 * Q.sj3_dr_max + 2.0418
    if Q.planar_flow < 0.1950135 and Q.lam1_plus_lam2 > 0.007520088:
        z += -930.2908 * (0.1950135 - Q.planar_flow) * (Q.lam1_plus_lam2 - 0.007520088)
    if Q.planar_flow < 0.1950135 and Q.pt_6 < 33.6875:
        z += -0.3414012 * (0.1950135 - Q.planar_flow) * (33.6875 - Q.pt_6)
    if Q.planar_flow < 0.1950135 and Q.log_sum_pt > 6.423044:
        z += 21.32051 * (0.1950135 - Q.planar_flow) * (Q.log_sum_pt - 6.423044)
    if Q.planar_flow < 0.1950135 and Q.z_7 < 0.05240025:
        z += -53.49857 * (0.1950135 - Q.planar_flow) * (0.05240025 - Q.z_7)
    if Q.lam1_plus_lam2 < 0.005590289 and Q.mean_phi < -0.00469376:
        z += 22980.52 * (0.005590289 - Q.lam1_plus_lam2) * (-0.00469376 - Q.mean_phi)
    if Q.sum_z_dr2 > 0.004372139 and Q.sj3_z1 > 0.8740683:
        z += -9074.597 * (Q.sum_z_dr2 - 0.004372139) * (Q.sj3_z1 - 0.8740683)
    if Q.sum_z_dr2 > 0.008678045 and Q.sj3_z1 > 0.8740683:
        z += -36156.08 * (Q.sum_z_dr2 - 0.008678045) * (Q.sj3_z1 - 0.8740683)
    if Q.tau1 < 0.05356915 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -237.4481 * (0.05356915 - Q.tau1) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.sum_zz_dr2 < 0.01165737 and Q.pt_7 < 48.71875:
        z += -2.313061 * (0.01165737 - Q.sum_zz_dr2) * (48.71875 - Q.pt_7)
    if Q.lam1_plus_lam2 < 0.005590289 and Q.mean_eta < -0.006779839:
        z += 25168.63 * (0.005590289 - Q.lam1_plus_lam2) * (-0.006779839 - Q.mean_eta)
    if Q.lam1_plus_lam2 < 0.005590289 and Q.mean_eta > 0.009367547:
        z += 26369.11 * (0.005590289 - Q.lam1_plus_lam2) * (Q.mean_eta - 0.009367547)
    if Q.lam1_plus_lam2 < 0.005590289 and Q.mean_phi > 0.009050008:
        z += 24453.94 * (0.005590289 - Q.lam1_plus_lam2) * (Q.mean_phi - 0.009050008)
    if Q.sum_z_dr2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 2158.136 * (Q.sum_z_dr2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    if Q.sum_zz_dr2 < 0.01165737 and Q.sj3_dr23 > 0.2989421:
        z += -1513.35 * (0.01165737 - Q.sum_zz_dr2) * (Q.sj3_dr23 - 0.2989421)
    if Q.centroid_offset < 0.02076709 and Q.D2_b2 < 4.721224:
        z += -11.44727 * (0.02076709 - Q.centroid_offset) * (4.721224 - Q.D2_b2)
    if Q.sum_z_dr2_top2 < 0.001056655 and Q.D2_b2 < 0.380911:
        z += -3607.363 * (0.001056655 - Q.sum_z_dr2_top2) * (0.380911 - Q.D2_b2)
    if Q.sum_z_dr2 > 0.01323868 and Q.eccentricity > 0.9458207:
        z += -34288.47 * (Q.sum_z_dr2 - 0.01323868) * (Q.eccentricity - 0.9458207)
    if Q.sum_z_dr2 > 0.008678045 and Q.eccentricity > 0.9458207:
        z += -11484.86 * (Q.sum_z_dr2 - 0.008678045) * (Q.eccentricity - 0.9458207)
    if Q.sum_zz_dr2 < 0.00718279 and Q.sj3_dr13 > 0.1982159:
        z += 1769.325 * (0.00718279 - Q.sum_zz_dr2) * (Q.sj3_dr13 - 0.1982159)
    if Q.sj3_dr_max < 0.213399 and Q.eccentricity > 0.9458207:
        z += -110.8556 * (0.213399 - Q.sj3_dr_max) * (Q.eccentricity - 0.9458207)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.1651751
    if Q.sum_z_dr2 < 0.0003193707:
        z += -2591.1 * Q.sum_z_dr2 + 4.12778
    if 0.0003193707 <= Q.sum_z_dr2 < 0.005019719:
        z += -579.7171 * Q.sum_z_dr2 + 3.485404
    if 0.005019719 <= Q.sum_z_dr2 < 0.006679471:
        z += -346.6702 * Q.sum_z_dr2 + 2.315573
    if Q.tau1 < 0.01357518:
        z += 11.943 * Q.tau1 + 0.472562
    if 0.01357518 <= Q.tau1 < 0.05356915:
        z += -15.86965 * Q.tau1 + 0.8501237
    if Q.LHA < 0.1967397:
        z += -19.74402 * Q.LHA + 3.884433
    if Q.log_sum_pt >= 6.701242:
        z += -8.09832 * Q.log_sum_pt + 54.2688
    if Q.sj3_dr_max < 0.1070199:
        z += 0.4261184 * Q.sj3_dr_max - 0.4961374
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 15.40895 * Q.sj3_dr_max - 2.099599
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += -1.748743 * Q.sj3_dr_max + 0.3473479
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.008931699 * Q.sum_pt_top5 - 6.139985
    if Q.sum_z_dr < 0.05464922:
        z += 55.74287 * Q.sum_z_dr - 3.046305
    if Q.lam1 < 0.001503553:
        z += -344.5599 * Q.lam1 + 0.5180642
    if Q.pt_6 < 24.42188:
        z += 0.074968 * Q.pt_6 - 1.830859
    if Q.n_dr_0_0p05 >= 5.0:
        z += 0.07120038 * Q.n_dr_0_0p05 - 0.3560019
    if Q.tau2 < 0.01713288:
        z += 25.22527 * Q.tau2 - 0.4321816
    if Q.N2 < 0.1531958:
        z += -2.299605 * Q.N2 + 0.3522898
    if Q.sum_z_dr2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 13296.78 * (0.006679471 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.sum_z_dr2 < 0.005019719 and Q.log_sum_pt < 6.502799:
        z += -466.2519 * (0.005019719 - Q.sum_z_dr2) * (6.502799 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.005019719 and Q.pt_7 < 43.5:
        z += -9.530384 * (0.005019719 - Q.sum_z_dr2) * (43.5 - Q.pt_7)
    if Q.sum_z_dr2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -25941.97 * (0.005019719 - Q.sum_z_dr2) * (Q.centroid_offset - 0.006789738)
    if Q.LHA < 0.1967397 and Q.z_dr_0p2_0p4 < 0.20552:
        z += -119.6466 * (0.1967397 - Q.LHA) * (0.20552 - Q.z_dr_0p2_0p4)
    if Q.sum_z_dr2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += 3317.714 * (0.005019719 - Q.sum_z_dr2) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sj3_dr_max < 0.1986272 and Q.lam1_plus_lam2 > 0.002635418:
        z += -1713.301 * (0.1986272 - Q.sj3_dr_max) * (Q.lam1_plus_lam2 - 0.002635418)
    if Q.tau1 < 0.05356915 and Q.sum_z_dr2 < 0.002635418:
        z += 17693.82 * (0.05356915 - Q.tau1) * (0.002635418 - Q.sum_z_dr2)
    if Q.sj3_dr_max < 0.1426152 and Q.lam1_plus_lam2 > 0.005590289:
        z += -67047.91 * (0.1426152 - Q.sj3_dr_max) * (Q.lam1_plus_lam2 - 0.005590289)
    if Q.sum_pt_top5 > 687.4375 and Q.sum_z_dr2_top3 < 0.01002369:
        z += -0.4632994 * (Q.sum_pt_top5 - 687.4375) * (0.01002369 - Q.sum_z_dr2_top3)
    if Q.sum_z_dr < 0.05464922 and Q.lam2 < 0.0001947983:
        z += 173135.3 * (0.05464922 - Q.sum_z_dr) * (0.0001947983 - Q.lam2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2_top3 > 0.001155057:
        z += 1280.023 * (0.1986272 - Q.sj3_dr_max) * (Q.sum_z_dr2_top3 - 0.001155057)
    if Q.LHA < 0.1967397 and Q.lam2 < 0.0001947983:
        z += -39134.04 * (0.1967397 - Q.LHA) * (0.0001947983 - Q.lam2)
    if Q.tau1 < 0.05356915 and Q.pt_7 < 43.5:
        z += 0.5413129 * (0.05356915 - Q.tau1) * (43.5 - Q.pt_7)
    if Q.z_dr_0_0p05 > 0.9008535 and Q.lam2 < 0.0001947983:
        z += -16975.3 * (Q.z_dr_0_0p05 - 0.9008535) * (0.0001947983 - Q.lam2)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi2 < 0.006445344:
        z += -261.7662 * (Q.log_sum_pt - 6.701242) * (0.006445344 - Q.mean_phi2)
    if Q.log_sum_pt > 6.701242 and Q.lam1_plus_lam2 < 0.006679471:
        z += -474.9559 * (Q.log_sum_pt - 6.701242) * (0.006679471 - Q.lam1_plus_lam2)
    if Q.log_sum_pt > 6.701242 and Q.lam1_plus_lam2 < 0.002635418:
        z += 3765.523 * (Q.log_sum_pt - 6.701242) * (0.002635418 - Q.lam1_plus_lam2)
    if Q.sum_pt_top5 > 687.4375 and Q.sum_z_dr2 < 0.0009641429:
        z += -3.812526 * (Q.sum_pt_top5 - 687.4375) * (0.0009641429 - Q.sum_z_dr2)
    if Q.sum_z_dr < 0.05464922 and Q.lam1_plus_lam2 < 0.002635418:
        z += -21768.21 * (0.05464922 - Q.sum_z_dr) * (0.002635418 - Q.lam1_plus_lam2)
    if Q.sum_z_dr2 < 0.006679471 and Q.lam1_plus_lam2 > 0.001653836:
        z += -118927.2 * (0.006679471 - Q.sum_z_dr2) * (Q.lam1_plus_lam2 - 0.001653836)
    if Q.sum_z_dr < 0.05464922 and Q.lam1_plus_lam2 < 0.005590289:
        z += -11032.4 * (0.05464922 - Q.sum_z_dr) * (0.005590289 - Q.lam1_plus_lam2)
    if Q.sj3_dr_max < 0.1986272 and Q.lam2 < 0.0003061234:
        z += 25123.93 * (0.1986272 - Q.sj3_dr_max) * (0.0003061234 - Q.lam2)
    if Q.sj3_dr_max < 0.1070199 and Q.lam2 < 0.0003061234:
        z += -45573.9 * (0.1070199 - Q.sj3_dr_max) * (0.0003061234 - Q.lam2)
    return max(0.0, z)


def neuron_9(Q):
    z = -2.719827
    if Q.sum_z_dr < 0.05464922:
        z += 92.07705 * Q.sum_z_dr - 5.529249
    if 0.05464922 <= Q.sum_z_dr < 0.06663269:
        z += 41.49964 * Q.sum_z_dr - 2.765233
    if Q.sum_z_dr2 < 0.003562611:
        z += -2976.222 * Q.sum_z_dr2 + 12.71581
    if 0.003562611 <= Q.sum_z_dr2 < 0.005019719:
        z += -1449.919 * Q.sum_z_dr2 + 7.278183
    if Q.LHA < 0.2160559:
        z += -2.3766 * Q.LHA + 0.5134783
    if 3.892127e-05 <= Q.e3 < 0.0001869378:
        z += -6435.27 * Q.e3 + 0.2504689
    if Q.e3 >= 0.0001869378:
        z += 870.9731 * Q.e3 - 1.115344
    if Q.centroid_offset < 0.01837778:
        z += 9.418282 * Q.centroid_offset + 0.1733095
    if 0.01837778 <= Q.centroid_offset < 0.02355416:
        z += -66.91869 * Q.centroid_offset + 1.576213
    if Q.log_sum_pt < 6.896095:
        z += -5.276153 * Q.log_sum_pt + 36.38485
    if Q.lam1_plus_lam2 < 0.006679471:
        z += -1114.199 * Q.lam1_plus_lam2 + 7.44226
    if Q.lam1_plus_lam2 >= 0.02530566:
        z += 77.14783 * Q.lam1_plus_lam2 - 1.952277
    if Q.lam2 < 7.300726e-05:
        z += -12590.08 * Q.lam2 + 1.30579
    if 7.300726e-05 <= Q.lam2 < 0.0001947983:
        z += -3174.472 * Q.lam2 + 0.6183818
    if Q.lam2 >= 0.001130645:
        z += 969.1924 * Q.lam2 - 1.095812
    if Q.sj2_dr < 0.1294903:
        z += 13.41327 * Q.sj2_dr - 1.475696
    if 0.1294903 <= Q.sj2_dr < 0.1778793:
        z += -5.397775 * Q.sj2_dr + 0.9601524
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.6916856 * Q.n_dr_0p2_0p4 - 0.6916856
    if Q.sj3_dr_max < 0.1426152:
        z += 16.7629 * Q.sj3_dr_max - 1.627682
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += -13.62138 * Q.sj3_dr_max + 2.705576
    if Q.sum_zz_dr2 < 0.001101266:
        z += 2404.952 * Q.sum_zz_dr2 - 8.770584
    if 0.001101266 <= Q.sum_zz_dr2 < 0.0030133:
        z += 1895.769 * Q.sum_zz_dr2 - 8.209837
    if 0.0030133 <= Q.sum_zz_dr2 < 0.004641801:
        z += 971.9804 * Q.sum_zz_dr2 - 5.426185
    if 0.004641801 <= Q.sum_zz_dr2 < 0.02378091:
        z += 47.7789 * Q.sum_zz_dr2 - 1.136226
    if Q.max_dr < 0.1117619:
        z += -8.864554 * Q.max_dr + 0.9907191
    if Q.zdr_0 < 0.004918231:
        z += 68.57354 * Q.zdr_0 + 0.2829334
    if 0.004918231 <= Q.zdr_0 < 0.01705377:
        z += -51.10559 * Q.zdr_0 + 0.871543
    if Q.sum_pt_top3 < 309.625:
        z += 0.004374932 * Q.sum_pt_top3 - 1.354588
    if Q.planar_flow < 0.04505724:
        z += 14.84664 * Q.planar_flow - 0.6689486
    if Q.tau32 < 0.4309923:
        z += -1.250458 * Q.tau32 + 0.5389377
    if Q.lam1 < 0.00483998:
        z += 137.0748 * Q.lam1 - 0.6634394
    if Q.dr_0 < 0.06413297:
        z += 11.22297 * Q.dr_0 - 0.7197625
    if Q.centroid_offset < 0.01837778 and Q.M2 < 0.04435703:
        z += -3658.869 * (0.01837778 - Q.centroid_offset) * (0.04435703 - Q.M2)
    if Q.centroid_offset < 0.01837778 and Q.tau3 < 0.01364517:
        z += 3730.157 * (0.01837778 - Q.centroid_offset) * (0.01364517 - Q.tau3)
    if Q.centroid_offset < 0.01837778 and Q.C2 < 0.06729223:
        z += 2543.384 * (0.01837778 - Q.centroid_offset) * (0.06729223 - Q.C2)
    if Q.sum_z_dr < 0.06663269 and Q.mean_phi < 0.00161889:
        z += -1355.792 * (0.06663269 - Q.sum_z_dr) * (0.00161889 - Q.mean_phi)
    if Q.centroid_offset < 0.02355416 and Q.N3 > 0.7686247:
        z += 31.93375 * (0.02355416 - Q.centroid_offset) * (Q.N3 - 0.7686247)
    if Q.lam2 > 0.001130645 and Q.planar_flow > 0.2534037:
        z += -820.6745 * (Q.lam2 - 0.001130645) * (Q.planar_flow - 0.2534037)
    if Q.log_sum_pt < 6.896095 and Q.z_5 < 0.05753583:
        z += 115.6797 * (6.896095 - Q.log_sum_pt) * (0.05753583 - Q.z_5)
    if Q.centroid_offset < 0.01837778 and Q.n_dr_0p05_0p1 < 5.0:
        z += 6.866292 * (0.01837778 - Q.centroid_offset) * (5.0 - Q.n_dr_0p05_0p1)
    if Q.centroid_offset < 0.02355416 and Q.tau21_b2 > 0.004811143:
        z += 62.21678 * (0.02355416 - Q.centroid_offset) * (Q.tau21_b2 - 0.004811143)
    if Q.centroid_offset < 0.02355416 and Q.D2_b2 < 0.380911:
        z += -143.9719 * (0.02355416 - Q.centroid_offset) * (0.380911 - Q.D2_b2)
    if Q.log_sum_pt < 6.896095 and Q.planar_flow > 0.1115136:
        z += -1.831149 * (6.896095 - Q.log_sum_pt) * (Q.planar_flow - 0.1115136)
    if Q.lam1_plus_lam2 < 0.006679471 and Q.mean_eta < -0.006779839:
        z += -9282.487 * (0.006679471 - Q.lam1_plus_lam2) * (-0.006779839 - Q.mean_eta)
    if Q.lam1_plus_lam2 < 0.006679471 and Q.tau3 > 0.003447406:
        z += 14953.91 * (0.006679471 - Q.lam1_plus_lam2) * (Q.tau3 - 0.003447406)
    if Q.sum_z_dr < 0.06663269 and Q.D2_b2 < 0.2669656:
        z += 77.07354 * (0.06663269 - Q.sum_z_dr) * (0.2669656 - Q.D2_b2)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 > 93.25:
        z += 0.6678025 * (0.01837778 - Q.centroid_offset) * (Q.pt_1 - 93.25)
    if Q.e3 > 0.0001869378 and Q.pt_7 < 33.21875:
        z += -159.1573 * (Q.e3 - 0.0001869378) * (33.21875 - Q.pt_7)
    if Q.log_sum_pt < 6.896095 and Q.z_dr_0p2_0p4 > 0.0:
        z += -4.768312 * (6.896095 - Q.log_sum_pt) * (Q.z_dr_0p2_0p4 - 0.0)
    if Q.sj2_dr < 0.1294903 and Q.mean_eta < -0.006779839:
        z += 463.5821 * (0.1294903 - Q.sj2_dr) * (-0.006779839 - Q.mean_eta)
    if Q.sj2_dr < 0.1294903 and Q.mean_eta > 0.01271871:
        z += 773.8846 * (0.1294903 - Q.sj2_dr) * (Q.mean_eta - 0.01271871)
    if Q.lam1_plus_lam2 < 0.006679471 and Q.mean_eta > 0.009367547:
        z += -12275.21 * (0.006679471 - Q.lam1_plus_lam2) * (Q.mean_eta - 0.009367547)
    if Q.centroid_offset < 0.002316125 and Q.sj3_z3 < 0.1167998:
        z += -4383.881 * (0.002316125 - Q.centroid_offset) * (0.1167998 - Q.sj3_z3)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.tau32 > 0.2691692:
        z += -0.6366253 * (Q.n_dr_0p2_0p4 - 1.0) * (Q.tau32 - 0.2691692)
    if Q.zdr_0 < 0.004918231 and Q.pt_3 > 71.5625:
        z += 2.636041 * (0.004918231 - Q.zdr_0) * (Q.pt_3 - 71.5625)
    if Q.centroid_offset < 0.02355416 and Q.tau32 < 0.5186963:
        z += -100.5514 * (0.02355416 - Q.centroid_offset) * (0.5186963 - Q.tau32)
    if Q.sum_z_dr2 < 0.003562611 and Q.mean_eta < -0.01284493:
        z += -25749.48 * (0.003562611 - Q.sum_z_dr2) * (-0.01284493 - Q.mean_eta)
    if Q.LHA < 0.2160559 and Q.mean_eta < 0.003218391:
        z += -300.0527 * (0.2160559 - Q.LHA) * (0.003218391 - Q.mean_eta)
    if Q.centroid_offset < 0.01837778 and Q.pt1_over_pt0 > 0.2131291:
        z += -85.32413 * (0.01837778 - Q.centroid_offset) * (Q.pt1_over_pt0 - 0.2131291)
    if Q.sum_z_dr2 < 0.003562611 and Q.mean_eta > 0.01271871:
        z += -32389.79 * (0.003562611 - Q.sum_z_dr2) * (Q.mean_eta - 0.01271871)
    if Q.lam1_plus_lam2 < 0.006679471 and Q.D3 < 0.8272948:
        z += -214.1587 * (0.006679471 - Q.lam1_plus_lam2) * (0.8272948 - Q.D3)
    if Q.sum_z_dr2 < 0.003562611 and Q.D3 < 0.8272948:
        z += 364.1661 * (0.003562611 - Q.sum_z_dr2) * (0.8272948 - Q.D3)
    if Q.sum_zz_dr2 < 0.001101266 and Q.pt_7 > 15.55391:
        z += 21.09511 * (0.001101266 - Q.sum_zz_dr2) * (Q.pt_7 - 15.55391)
    if Q.centroid_offset < 0.01837778 and Q.tau3 < 0.00739495:
        z += 9595.431 * (0.01837778 - Q.centroid_offset) * (0.00739495 - Q.tau3)
    if Q.centroid_offset < 0.01837778 and Q.mratio_min_012 < 0.07735553:
        z += -378.2081 * (0.01837778 - Q.centroid_offset) * (0.07735553 - Q.mratio_min_012)
    if Q.sum_zz_dr2 < 0.001101266 and Q.eccentricity > 0.7117266:
        z += -2089.974 * (0.001101266 - Q.sum_zz_dr2) * (Q.eccentricity - 0.7117266)
    if Q.lam1 < 0.00483998 and Q.dr1_7 < 0.1792439:
        z += -513.9696 * (0.00483998 - Q.lam1) * (0.1792439 - Q.dr1_7)
    if Q.log_sum_pt < 6.896095 and Q.D2_b2 < 0.9206502:
        z += -0.5795773 * (6.896095 - Q.log_sum_pt) * (0.9206502 - Q.D2_b2)
    if Q.centroid_offset < 0.02355416 and Q.dr_7 < 0.0825754:
        z += 295.6378 * (0.02355416 - Q.centroid_offset) * (0.0825754 - Q.dr_7)
    if Q.lam2 < 7.300726e-05 and Q.pt_7 > 27.375:
        z += -161.3415 * (7.300726e-05 - Q.lam2) * (Q.pt_7 - 27.375)
    if Q.log_sum_pt < 6.896095 and Q.dr_2 < 0.01778111:
        z += -74.71733 * (6.896095 - Q.log_sum_pt) * (0.01778111 - Q.dr_2)
    if Q.pt_1 > 159.25 and Q.D2_b2 < 1.129616:
        z += -0.006612202 * (Q.pt_1 - 159.25) * (1.129616 - Q.D2_b2)
    if Q.pt_1 > 159.25 and Q.absphi_5 < 0.1132812:
        z += -0.03579307 * (Q.pt_1 - 159.25) * (0.1132812 - Q.absphi_5)
    if Q.sum_z_dr < 0.06663269 and Q.mean_phi > 0.004406178:
        z += -1075.467 * (0.06663269 - Q.sum_z_dr) * (Q.mean_phi - 0.004406178)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.578524
    z += -9.016036 * Q.e2
    if Q.lam1 < 0.003377388:
        z += 983.1706 * Q.lam1 - 2.376184
    if 0.003377388 <= Q.lam1 < 0.00483998:
        z += 722.0029 * Q.lam1 - 1.494119
    if 0.00483998 <= Q.lam1 < 0.006506576:
        z += 458.0469 * Q.lam1 - 0.2165775
    if Q.lam1 >= 0.006506576:
        z += 228.4147 * Q.lam1 + 1.277542
    if 0.0001947983 <= Q.lam2 < 0.003408389:
        z += 1531.584 * Q.lam2 - 0.2983499
    if Q.lam2 >= 0.003408389:
        z += 1125.035 * Q.lam2 + 1.087325
    if Q.pt_6 < 48.03125:
        z += 0.0165398 * Q.pt_6 - 0.7944272
    if 8.147744e-05 <= Q.e3 < 0.0005116989:
        z += 4584.36 * Q.e3 - 0.3735219
    if Q.e3 >= 0.0005116989:
        z += 2200.285 * Q.e3 + 0.8464064
    if 0.1967397 <= Q.LHA < 0.3127275:
        z += -10.08487 * Q.LHA + 1.984094
    if Q.LHA >= 0.3127275:
        z += -17.23143 * Q.LHA + 4.219019
    if Q.centroid_offset >= 0.02355416:
        z += 14.64249 * Q.centroid_offset - 0.3448915
    if Q.D2 < 1.002471:
        z += -0.5846103 * Q.D2 + 0.5860547
    if Q.n_dr_0p05_0p1 < 5.0:
        z += -0.05066503 * Q.n_dr_0p05_0p1 + 0.2533252
    if Q.M2 < 0.02563286:
        z += 30.54857 * Q.M2 - 0.7830473
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.9571146 * Q.n_dr_0p2_0p4 - 0.9571146
    if Q.z_7 < 0.06810151:
        z += 7.392156 * Q.z_7 - 0.503417
    if Q.tau1 >= 0.04369778:
        z += 12.29691 * Q.tau1 - 0.5373475
    if 0.1294903 <= Q.sj2_dr < 0.1682655:
        z += 8.335772 * Q.sj2_dr - 1.079402
    if 0.1682655 <= Q.sj2_dr < 0.3003793:
        z += 2.893342 * Q.sj2_dr - 0.1636282
    if Q.sj2_dr >= 0.3003793:
        z += 8.261006 * Q.sj2_dr - 1.775963
    if Q.sum_z_dr2_top3 < 0.002915531:
        z += -162.3973 * Q.sum_z_dr2_top3 + 0.4734744
    if Q.lam1_plus_lam2 < 0.0009641429:
        z += 422.473 * Q.lam1_plus_lam2 - 0.4073244
    if Q.sum_pt >= 988.4078:
        z += 0.003775819 * Q.sum_pt - 3.732049
    if Q.M3 < 0.04581318:
        z += -20.1004 * Q.M3 + 0.9208635
    if Q.sj3_dr_max < 0.3012016:
        z += -2.916882 * Q.sj3_dr_max + 0.8785694
    if 0.002635418 <= Q.sum_z_dr2 < 0.007520088:
        z += -358.4292 * Q.sum_z_dr2 + 0.9446106
    if Q.sum_z_dr2 >= 0.007520088:
        z += -72.68738 * Q.sum_z_dr2 - 1.204193
    if Q.sum_z_dr2_top5 < 0.001501708:
        z += -312.5487 * Q.sum_z_dr2_top5 + 0.4693571
    if Q.sum_z_dr2_top5 >= 0.01148293:
        z += -28.95127 * Q.sum_z_dr2_top5 + 0.3324453
    if Q.log_sum_pt >= 6.670067:
        z += -4.49901 * Q.log_sum_pt + 30.0087
    if Q.n_pt_above_50 >= 6.0:
        z += -0.1709555 * Q.n_pt_above_50 + 1.025733
    if Q.sum_z_dr < 0.1245537:
        z += 14.70203 * Q.sum_z_dr - 1.831193
    if Q.lam2 > 0.0001947983 and Q.log_sum_pt < 6.538559:
        z += -521.1292 * (Q.lam2 - 0.0001947983) * (6.538559 - Q.log_sum_pt)
    if Q.lam2 > 0.0001947983 and Q.eccentricity > 0.4829602:
        z += -878.3087 * (Q.lam2 - 0.0001947983) * (Q.eccentricity - 0.4829602)
    if Q.pt_6 < 48.03125 and Q.log_sum_pt < 6.46415:
        z += -0.1173893 * (48.03125 - Q.pt_6) * (6.46415 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.D2_b2 < 4.721224:
        z += -0.197551 * (Q.n_dr_0p2_0p4 - 1.0) * (4.721224 - Q.D2_b2)
    if Q.pt_6 < 48.03125 and Q.D2_b2 < 0.1236856:
        z += 0.2742773 * (48.03125 - Q.pt_6) * (0.1236856 - Q.D2_b2)
    if Q.lam2 > 0.0001947983 and Q.planar_flow > 0.06126205:
        z += -689.1904 * (Q.lam2 - 0.0001947983) * (Q.planar_flow - 0.06126205)
    if Q.sj2_dr > 0.1682655 and Q.planar_flow < 0.6947818:
        z += -14.81323 * (Q.sj2_dr - 0.1682655) * (0.6947818 - Q.planar_flow)
    if Q.lam2 > 0.0001947983 and Q.zdr_7 < 0.01265416:
        z += -12670.29 * (Q.lam2 - 0.0001947983) * (0.01265416 - Q.zdr_7)
    if Q.sj2_dr > 0.1682655 and Q.mean_eta < 0.02644207:
        z += -44.1332 * (Q.sj2_dr - 0.1682655) * (0.02644207 - Q.mean_eta)
    if Q.LHA > 0.3127275 and Q.planar_flow < 0.6947818:
        z += -23.82551 * (Q.LHA - 0.3127275) * (0.6947818 - Q.planar_flow)
    if Q.lam1 < 0.006506576 and Q.pt_3 > 60.59375:
        z += 0.8493293 * (0.006506576 - Q.lam1) * (Q.pt_3 - 60.59375)
    if Q.lam1 < 0.006506576 and Q.absphi_7 < 0.1647949:
        z += -352.1826 * (0.006506576 - Q.lam1) * (0.1647949 - Q.absphi_7)
    if Q.e3 > 8.147744e-05 and Q.pt_7 < 33.21875:
        z += -205.4488 * (Q.e3 - 8.147744e-05) * (33.21875 - Q.pt_7)
    if Q.lam1 < 0.006506576 and Q.n_pt_above_50 > 6.0:
        z += 51.97581 * (0.006506576 - Q.lam1) * (Q.n_pt_above_50 - 6.0)
    if Q.log_sum_pt > 6.670067 and Q.n_dr_0p1_0p2 < 3.0:
        z += 0.5244045 * (Q.log_sum_pt - 6.670067) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.n_dr_0p05_0p1 < 5.0 and Q.sj3_dr23 > 0.3661115:
        z += 0.7823222 * (5.0 - Q.n_dr_0p05_0p1) * (Q.sj3_dr23 - 0.3661115)
    if Q.log_sum_pt > 6.670067 and Q.abseta_0 < 0.04000854:
        z += 61.31572 * (Q.log_sum_pt - 6.670067) * (0.04000854 - Q.abseta_0)
    return max(0.0, z)


def neuron_11(Q):
    z = -2.101345
    if Q.planar_flow < 0.2534037:
        z += -3.818143 * Q.planar_flow + 0.9675315
    if 0.09419022 <= Q.sj2_dr < 0.1778793:
        z += 6.931287 * Q.sj2_dr - 0.6528595
    if 0.1778793 <= Q.sj2_dr < 0.1872617:
        z += -20.28664 * Q.sj2_dr + 4.188647
    if 0.1872617 <= Q.sj2_dr < 0.2687922:
        z += -10.92815 * Q.sj2_dr + 2.436159
    if Q.sj2_dr >= 0.2687922:
        z += -2.154917 * Q.sj2_dr + 0.07798294
    if Q.sum_zz_dr2 < 0.0030133:
        z += 753.654 * Q.sum_zz_dr2 - 7.161438
    if 0.0030133 <= Q.sum_zz_dr2 < 0.006390125:
        z += 1186.671 * Q.sum_zz_dr2 - 8.466248
    if 0.006390125 <= Q.sum_zz_dr2 < 0.008168571:
        z += 496.6547 * Q.sum_zz_dr2 - 4.056959
    if Q.lam1_plus_lam2 < 0.005019719:
        z += -1539.621 * Q.lam1_plus_lam2 + 17.68723
    if 0.005019719 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -1922.249 * Q.lam1_plus_lam2 + 19.60792
    if 0.008678045 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -641.7 * Q.lam1_plus_lam2 + 8.495258
    if Q.tau1 < 0.03459477:
        z += 82.15285 * Q.tau1 - 3.349674
    if 0.03459477 <= Q.tau1 < 0.09538712:
        z += 8.349983 * Q.tau1 - 0.7964809
    if Q.LHA < 0.2809341:
        z += -12.5113 * Q.LHA + 3.300437
    if 0.2809341 <= Q.LHA < 0.3033137:
        z += -4.260738 * Q.LHA + 0.9825716
    if 0.3033137 <= Q.LHA < 0.3127275:
        z += 32.90611 * Q.LHA - 10.29064
    if Q.centroid_offset < 0.01437952:
        z += -28.40506 * Q.centroid_offset + 1.914592
    if 0.01437952 <= Q.centroid_offset < 0.03776099:
        z += -64.41599 * Q.centroid_offset + 2.432412
    if Q.sum_z_dr < 0.02689598:
        z += 198.775 * Q.sum_z_dr - 10.67196
    if 0.02689598 <= Q.sum_z_dr < 0.03360421:
        z += 176.1289 * Q.sum_z_dr - 10.06287
    if 0.03360421 <= Q.sum_z_dr < 0.0717028:
        z += 108.7756 * Q.sum_z_dr - 7.799517
    if 6.572938 <= Q.log_sum_pt < 6.733425:
        z += 5.020469 * Q.log_sum_pt - 32.99923
    if Q.log_sum_pt >= 6.733425:
        z += 0.9642386 * Q.log_sum_pt - 5.686908
    if Q.e2 < 0.02045966:
        z += -33.65388 * Q.e2 - 0.08140426
    if 0.02045966 <= Q.e2 < 0.04447357:
        z += 32.06272 * Q.e2 - 1.425943
    if Q.mean_phi2 < 0.001570267:
        z += 234.746 * Q.mean_phi2 - 0.368614
    if Q.eccentricity >= 0.9884745:
        z += 28.20641 * Q.eccentricity - 27.88131
    if Q.sum_z_dr2 < 0.003562611:
        z += 98.42059 * Q.sum_z_dr2 + 1.254241
    if 0.003562611 <= Q.sum_z_dr2 < 0.006679471:
        z += -514.9012 * Q.sum_z_dr2 + 3.439268
    if Q.lam1 < 0.008375572:
        z += 452.7214 * Q.lam1 - 3.791801
    if Q.pt_6 < 29.90625:
        z += 0.06985637 * Q.pt_6 - 2.439789
    if 29.90625 <= Q.pt_6 < 39.75:
        z += 0.01902512 * Q.pt_6 - 0.9196174
    if 39.75 <= Q.pt_6 < 62.25:
        z += 0.007260832 * Q.pt_6 - 0.4519868
    if Q.pt_7 >= 29.04219:
        z += -0.02176744 * Q.pt_7 + 0.6321742
    if Q.z_dr_0_0p05 >= 0.6080732:
        z += 0.8424305 * Q.z_dr_0_0p05 - 0.5122595
    if Q.lam2 < 0.000537286:
        z += -1036.83 * Q.lam2 + 0.557074
    if Q.z_7 >= 0.01685855:
        z += 16.72824 * Q.z_7 - 0.2820137
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.006544977 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.pt_7 < 37.15625:
        z += -0.09681757 * (0.2534037 - Q.planar_flow) * (37.15625 - Q.pt_7)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.log_sum_pt < 6.502799:
        z += -327.0006 * (0.01323868 - Q.lam1_plus_lam2) * (6.502799 - Q.log_sum_pt)
    if Q.planar_flow < 0.2534037 and Q.e3 < 1.050302e-05:
        z += -299850.3 * (0.2534037 - Q.planar_flow) * (1.050302e-05 - Q.e3)
    if Q.sj2_dr > 0.1778793 and Q.lam2 < 0.001130645:
        z += -3756.81 * (Q.sj2_dr - 0.1778793) * (0.001130645 - Q.lam2)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += -129.4321 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.e2 < 0.04447357 and Q.D2 < 1.002471:
        z += -52.05584 * (0.04447357 - Q.e2) * (1.002471 - Q.D2)
    if Q.centroid_offset < 0.01437952 and Q.M3 < 0.09607851:
        z += 916.4577 * (0.01437952 - Q.centroid_offset) * (0.09607851 - Q.M3)
    if Q.centroid_offset < 0.01437952 and Q.tau21_b2 < 0.05932655:
        z += 1458.013 * (0.01437952 - Q.centroid_offset) * (0.05932655 - Q.tau21_b2)
    if Q.mean_phi2 < 0.001570267 and Q.M2 > 0.01411669:
        z += 2078.155 * (0.001570267 - Q.mean_phi2) * (Q.M2 - 0.01411669)
    if Q.centroid_offset < 0.01437952 and Q.sj3_dr_min < 0.08543881:
        z += -744.053 * (0.01437952 - Q.centroid_offset) * (0.08543881 - Q.sj3_dr_min)
    if Q.tau1 < 0.03459477 and Q.sum_z_dr2_top3 < 0.002915531:
        z += 37188.34 * (0.03459477 - Q.tau1) * (0.002915531 - Q.sum_z_dr2_top3)
    if Q.sum_zz_dr2 < 0.0030133 and Q.sum_z_dr2_top3 > 0.003952582:
        z += -452964.9 * (0.0030133 - Q.sum_zz_dr2) * (Q.sum_z_dr2_top3 - 0.003952582)
    if Q.centroid_offset < 0.03776099 and Q.pt_7 < 53.4375:
        z += -0.8320979 * (0.03776099 - Q.centroid_offset) * (53.4375 - Q.pt_7)
    if Q.log_sum_pt > 6.572938 and Q.D3 < 3.916009:
        z += -1.204253 * (Q.log_sum_pt - 6.572938) * (3.916009 - Q.D3)
    if Q.tau1 < 0.09538712 and Q.pt_6 < 62.25:
        z += 0.2138959 * (0.09538712 - Q.tau1) * (62.25 - Q.pt_6)
    if Q.sum_zz_dr2 < 0.008168571 and Q.dr01 > 0.05595395:
        z += -1076.322 * (0.008168571 - Q.sum_zz_dr2) * (Q.dr01 - 0.05595395)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.pt1_dr01 < 28.39396:
        z += -1.24621 * (0.01323868 - Q.lam1_plus_lam2) * (28.39396 - Q.pt1_dr01)
    if Q.centroid_offset < 0.03776099 and Q.pt_2 < 118.5:
        z += -0.8896294 * (0.03776099 - Q.centroid_offset) * (118.5 - Q.pt_2)
    if Q.sum_zz_dr2 < 0.006390125 and Q.pt_2 < 118.5:
        z += 5.088725 * (0.006390125 - Q.sum_zz_dr2) * (118.5 - Q.pt_2)
    if Q.sum_zz_dr2 < 0.006390125 and Q.z_3rd < 0.1471389:
        z += -3658.219 * (0.006390125 - Q.sum_zz_dr2) * (0.1471389 - Q.z_3rd)
    if Q.centroid_offset < 0.03776099 and Q.z_3rd < 0.1561228:
        z += 510.4086 * (0.03776099 - Q.centroid_offset) * (0.1561228 - Q.z_3rd)
    if Q.log_sum_pt > 6.572938 and Q.z_7 < 0.0586137:
        z += 87.26995 * (Q.log_sum_pt - 6.572938) * (0.0586137 - Q.z_7)
    if Q.centroid_offset < 0.01437952 and Q.tau3 < 0.01364517:
        z += -6620.705 * (0.01437952 - Q.centroid_offset) * (0.01364517 - Q.tau3)
    if Q.log_sum_pt > 6.733425 and Q.n_pt_above_50 > 7.0:
        z += 3.141682 * (Q.log_sum_pt - 6.733425) * (Q.n_pt_above_50 - 7.0)
    if Q.centroid_offset > 0.04990367 and Q.e3 < 0.0005116989:
        z += -218548.1 * (Q.centroid_offset - 0.04990367) * (0.0005116989 - Q.e3)
    if Q.centroid_offset < 0.01437952 and Q.tau21_b2 < 0.2691019:
        z += 115.7425 * (0.01437952 - Q.centroid_offset) * (0.2691019 - Q.tau21_b2)
    if Q.lam1_plus_lam2 < 0.005019719 and Q.sj3_dr23 > 0.1414609:
        z += 2392.007 * (0.005019719 - Q.lam1_plus_lam2) * (Q.sj3_dr23 - 0.1414609)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.3206811
    if 0.01882765 <= Q.sum_z_dr2 < 0.02530566:
        z += 315.7651 * Q.sum_z_dr2 - 5.945116
    if Q.sum_z_dr2 >= 0.02530566:
        z += 662.383 * Q.sum_z_dr2 - 14.71651
    if Q.e2 >= 0.06344108:
        z += -52.49882 * Q.e2 + 3.330582
    if Q.ptdr0_4 >= 15.26525:
        z += 0.2240794 * Q.ptdr0_4 - 3.420627
    if Q.centroid_offset >= 0.04990367:
        z += 16.41805 * Q.centroid_offset - 0.8193209
    if Q.zdr_0 >= 0.03981924:
        z += -29.22568 * Q.zdr_0 + 1.163745
    if Q.sum_z_dr2 > 0.01882765 and Q.pt_7 < 53.4375:
        z += -4.00191 * (Q.sum_z_dr2 - 0.01882765) * (53.4375 - Q.pt_7)
    if Q.sum_z_dr2_top2 > 0.02260535 and Q.sum_pt > 527.1781:
        z += -0.3566689 * (Q.sum_z_dr2_top2 - 0.02260535) * (Q.sum_pt - 527.1781)
    if Q.ptdr0_4 > 15.26525 and Q.sum_pt < 988.4078:
        z += -0.001361443 * (Q.ptdr0_4 - 15.26525) * (988.4078 - Q.sum_pt)
    if Q.ptdr0_4 > 15.26525 and Q.sum_pt < 763.825:
        z += 0.001760143 * (Q.ptdr0_4 - 15.26525) * (763.825 - Q.sum_pt)
    if Q.lam1_plus_lam2 > 0.01323868 and Q.log_sum_pt > 6.327379:
        z += 670.7113 * (Q.lam1_plus_lam2 - 0.01323868) * (Q.log_sum_pt - 6.327379)
    if Q.sum_z_dr2 > 0.02530566 and Q.sum_pt < 868.5094:
        z += -0.5261453 * (Q.sum_z_dr2 - 0.02530566) * (868.5094 - Q.sum_pt)
    if Q.e2 > 0.06344108 and Q.sum_pt > 559.6875:
        z += -0.1995643 * (Q.e2 - 0.06344108) * (Q.sum_pt - 559.6875)
    if Q.sum_z_dr2 > 0.01882765 and Q.planar_flow < 0.7974684:
        z += -156.5142 * (Q.sum_z_dr2 - 0.01882765) * (0.7974684 - Q.planar_flow)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.8229663
    if Q.sum_z_dr < 0.1484084:
        z += -70.23822 * Q.sum_z_dr + 10.42394
    if Q.lam1 < 0.006506576:
        z += -403.5028 * Q.lam1 + 3.357631
    if 0.006506576 <= Q.lam1 < 0.01643375:
        z += -73.75813 * Q.lam1 + 1.212123
    if 482.2812 <= Q.sum_pt_top5 < 687.4375:
        z += -0.009851875 * Q.sum_pt_top5 + 4.751374
    if 687.4375 <= Q.sum_pt_top5 < 839.9547:
        z += -0.008621673 * Q.sum_pt_top5 + 3.905688
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.001996185 * Q.sum_pt_top5 - 1.659422
    if Q.pt_6 < 27.57812:
        z += 0.1024789 * Q.pt_6 - 3.470881
    if 27.57812 <= Q.pt_6 < 50.25:
        z += 0.02843633 * Q.pt_6 - 1.428926
    if Q.sum_pt < 788.4484:
        z += 0.001909349 * Q.sum_pt - 1.505423
    if Q.sum_pt >= 988.4078:
        z += -0.01106503 * Q.sum_pt + 10.93677
    if Q.sj3_dr_max >= 0.233678:
        z += -2.939106 * Q.sj3_dr_max + 0.6868043
    if Q.pt_5 < 24.57812:
        z += 0.2468768 * Q.pt_5 - 6.06777
    if Q.e3 < 8.147744e-05:
        z += -3271.866 * Q.e3 - 0.749316
    if 8.147744e-05 <= Q.e3 < 0.0001869378:
        z += 9632.994 * Q.e3 - 1.800771
    if Q.e2 < 0.08000524:
        z += 22.42074 * Q.e2 - 1.793777
    if Q.z_5 < 0.02818362:
        z += -104.0995 * Q.z_5 + 2.933901
    if Q.sj3_dr_min < 0.1278212:
        z += 2.347844 * Q.sj3_dr_min - 0.3001042
    if Q.log_sum_pt < 6.46415:
        z += 1.776162 * Q.log_sum_pt - 11.48138
    if Q.pt_7 >= 31.85938:
        z += -0.06975239 * Q.pt_7 + 2.222267
    if Q.tau1 < 0.1136369:
        z += 10.26146 * Q.tau1 - 1.16608
    if Q.lam1_plus_lam2 < 0.008678045:
        z += 51.66264 * Q.lam1_plus_lam2 + 0.4108147
    if 0.008678045 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -188.3831 * Q.lam1_plus_lam2 + 2.493942
    if Q.sum_z_dr2 < 0.007520088:
        z += 146.0001 * Q.sum_z_dr2 - 0.02303299
    if 0.007520088 <= Q.sum_z_dr2 < 0.02530566:
        z += -60.43663 * Q.sum_z_dr2 + 1.529389
    if Q.C2_b2 < 0.009032972:
        z += 64.19775 * Q.C2_b2 - 0.5798965
    if Q.z_top5 >= 0.8919245:
        z += -5.485327 * Q.z_top5 + 4.892497
    if Q.z_7 >= 0.04624032:
        z += 46.19034 * Q.z_7 - 2.135856
    if Q.z_dr_0_0p05 >= 0.6080732:
        z += 0.9711636 * Q.z_dr_0_0p05 - 0.5905386
    if Q.sum_z_dr < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -58.32185 * (0.1484084 - Q.sum_z_dr) * (6.804164 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.1484084 and Q.pt_7 < 38.53125:
        z += -0.7992349 * (0.1484084 - Q.sum_z_dr) * (38.53125 - Q.pt_7)
    if Q.sum_z_dr < 0.1484084 and Q.lam2 < 0.0003061234:
        z += -19369.97 * (0.1484084 - Q.sum_z_dr) * (0.0003061234 - Q.lam2)
    if Q.sj3_dr_max > 0.233678 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -30.89711 * (Q.sj3_dr_max - 0.233678) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.lam1 < 0.01643375 and Q.z_dr_0_0p05 > 0.3658817:
        z += -100.3087 * (0.01643375 - Q.lam1) * (Q.z_dr_0_0p05 - 0.3658817)
    if Q.pt_5 < 24.57812 and Q.pt_1 < 169.875:
        z += -0.002987031 * (24.57812 - Q.pt_5) * (169.875 - Q.pt_1)
    if Q.sum_pt > 988.4078 and Q.z_6 > 0.03932388:
        z += 0.2663446 * (Q.sum_pt - 988.4078) * (Q.z_6 - 0.03932388)
    if Q.sum_pt_top5 > 687.4375 and Q.pt_7 < 40.04062:
        z += 0.0004324048 * (Q.sum_pt_top5 - 687.4375) * (40.04062 - Q.pt_7)
    if Q.lam1 < 0.01643375 and Q.pt_7 < 25.57812:
        z += -7.172699 * (0.01643375 - Q.lam1) * (25.57812 - Q.pt_7)
    if Q.sum_pt > 988.4078 and Q.M3 < 0.07092092:
        z += -0.1624786 * (Q.sum_pt - 988.4078) * (0.07092092 - Q.M3)
    if Q.sj3_dr_max > 0.233678 and Q.pt_7 > 37.15625:
        z += -0.356263 * (Q.sj3_dr_max - 0.233678) * (Q.pt_7 - 37.15625)
    if Q.e3 < 8.147744e-05 and Q.centroid_offset < 0.03776099:
        z += -537965.3 * (8.147744e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.log_sum_pt < 6.46415 and Q.pt_4 < 31.125:
        z += -1.621954 * (6.46415 - Q.log_sum_pt) * (31.125 - Q.pt_4)
    if Q.pt_6 < 50.25 and Q.zdr_0 > 0.001113887:
        z += 0.3825005 * (50.25 - Q.pt_6) * (Q.zdr_0 - 0.001113887)
    if Q.e2 < 0.08000524 and Q.pt_4 < 60.03125:
        z += 0.1567399 * (0.08000524 - Q.e2) * (60.03125 - Q.pt_4)
    if Q.sum_pt_top5 > 687.4375 and Q.dr_2 < 0.08525808:
        z += 0.01347228 * (Q.sum_pt_top5 - 687.4375) * (0.08525808 - Q.dr_2)
    if Q.log_sum_pt < 6.46415 and Q.C3 < 0.06809619:
        z += 33.50224 * (6.46415 - Q.log_sum_pt) * (0.06809619 - Q.C3)
    if Q.sum_pt < 788.4484 and Q.D3 < 2.468971:
        z += -0.0006477451 * (788.4484 - Q.sum_pt) * (2.468971 - Q.D3)
    if Q.sum_pt_top5 > 482.2812 and Q.e4 < 3.0374e-08:
        z += 297821.4 * (Q.sum_pt_top5 - 482.2812) * (3.0374e-08 - Q.e4)
    if Q.pt_5 < 24.57812 and Q.z_2nd < 0.1809419:
        z += 3.001513 * (24.57812 - Q.pt_5) * (0.1809419 - Q.z_2nd)
    if Q.pt_7 > 31.85938 and Q.zdr_1 < 0.01391455:
        z += 2.953661 * (Q.pt_7 - 31.85938) * (0.01391455 - Q.zdr_1)
    if Q.sum_pt < 788.4484 and Q.zdr_1 < 0.01391455:
        z += -0.2286942 * (788.4484 - Q.sum_pt) * (0.01391455 - Q.zdr_1)
    if Q.sj3_dr_max > 0.233678 and Q.tau32 < 0.4309923:
        z += -7.829103 * (Q.sj3_dr_max - 0.233678) * (0.4309923 - Q.tau32)
    if Q.sum_pt_top5 > 482.2812 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += 0.001752847 * (Q.sum_pt_top5 - 482.2812) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.log_sum_pt < 6.46415 and Q.zdr_1 < 0.009612129:
        z += 328.7019 * (6.46415 - Q.log_sum_pt) * (0.009612129 - Q.zdr_1)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.1540651
    if Q.planar_flow < 0.1115136:
        z += -30.26742 * Q.planar_flow + 3.375227
    if Q.z_dr_0p05_0p1 < 0.5882598:
        z += -0.8066343 * Q.z_dr_0p05_0p1 + 0.4745105
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += -12.53319 * Q.z_dr_0p05_0p1 + 9.411294
    if 0.8155839 <= Q.psi_0p1 < 0.9761279:
        z += -3.370124 * Q.psi_0p1 + 2.748619
    if Q.psi_0p1 >= 0.9761279:
        z += -15.43316 * Q.psi_0p1 + 14.52368
    if 0.006679471 <= Q.sum_z_dr2 < 0.007520088:
        z += -1195.232 * Q.sum_z_dr2 + 7.983518
    if 0.007520088 <= Q.sum_z_dr2 < 0.008678045:
        z += -2438.812 * Q.sum_z_dr2 + 17.33535
    if Q.sum_z_dr2 >= 0.008678045:
        z += -1903.577 * Q.sum_z_dr2 + 12.69055
    if 0.002074109 <= Q.sum_zz_dr2 < 0.006390125:
        z += -120.0072 * Q.sum_zz_dr2 + 0.2489081
    if 0.006390125 <= Q.sum_zz_dr2 < 0.00718279:
        z += 822.3535 * Q.sum_zz_dr2 - 5.772895
    if 0.00718279 <= Q.sum_zz_dr2 < 0.008168571:
        z += 1456.667 * Q.sum_zz_dr2 - 10.32904
    if 0.008168571 <= Q.sum_zz_dr2 < 0.01165737:
        z += 913.3709 * Q.sum_zz_dr2 - 5.891082
    if 0.01165737 <= Q.sum_zz_dr2 < 0.01716248:
        z += 1224.923 * Q.sum_zz_dr2 - 9.522964
    if Q.sum_zz_dr2 >= 0.01716248:
        z += -299.2674 * Q.sum_zz_dr2 + 16.63593
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += -291.0518 * Q.lam1_plus_lam2 + 3.85314
    if 0.03556091 <= Q.e2 < 0.04110972:
        z += 21.66976 * Q.e2 - 0.7705964
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += 72.28196 * Q.e2 - 2.85125
    if Q.e2 >= 0.05028464:
        z += 82.84384 * Q.e2 - 3.38235
    if 0.02355416 <= Q.centroid_offset < 0.03776099:
        z += -26.54214 * Q.centroid_offset + 0.6251779
    if 0.03776099 <= Q.centroid_offset < 0.04990367:
        z += -308.4338 * Q.centroid_offset + 11.26969
    if Q.centroid_offset >= 0.04990367:
        z += -521.4472 * Q.centroid_offset + 21.89984
    if 0.03459477 <= Q.tau1 < 0.09538712:
        z += -21.97969 * Q.tau1 + 0.7603823
    if Q.tau1 >= 0.09538712:
        z += 19.32347 * Q.tau1 - 3.179407
    if 0.02689598 <= Q.sum_z_dr < 0.04081947:
        z += 34.26288 * Q.sum_z_dr - 0.9215338
    if 0.04081947 <= Q.sum_z_dr < 0.08723651:
        z += 102.7673 * Q.sum_z_dr - 3.717846
    if Q.sum_z_dr >= 0.08723651:
        z += -42.72268 * Q.sum_z_dr + 8.974188
    if Q.n_dr_0p05_0p1 < 5.0:
        z += 0.1974606 * Q.n_dr_0p05_0p1 - 0.987303
    if Q.sj2_dr < 0.1294903:
        z += 2.617146 * Q.sj2_dr + 0.2606106
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -20.19829 * Q.sj2_dr + 3.214989
    if Q.sj3_dr_max < 0.1594012:
        z += 20.41855 * Q.sj3_dr_max - 5.014792
    if 0.1594012 <= Q.sj3_dr_max < 0.213399:
        z += 25.7783 * Q.sj3_dr_max - 5.869143
    if 0.213399 <= Q.sj3_dr_max < 0.233678:
        z += 18.15076 * Q.sj3_dr_max - 4.241432
    if Q.LHA >= 0.3033137:
        z += 16.81254 * Q.LHA - 5.099473
    if Q.eccentricity >= 0.9781608:
        z += -22.525 * Q.eccentricity + 22.03307
    if Q.n_dr_0_0p05 >= 5.0:
        z += 0.243693 * Q.n_dr_0_0p05 - 1.218465
    if Q.sum_z_dr2_top5 < 0.00422683:
        z += -88.14474 * Q.sum_z_dr2_top5 + 0.0899407
    if 0.00422683 <= Q.sum_z_dr2_top5 < 0.005005443:
        z += 362.9943 * Q.sum_z_dr2_top5 - 1.816947
    if 1.0 <= Q.n_dr_0p1_0p2 < 2.0:
        z += 0.2534802 * Q.n_dr_0p1_0p2 - 0.2534802
    if Q.n_dr_0p1_0p2 >= 2.0:
        z += 0.1054977 * Q.n_dr_0p1_0p2 + 0.04248485
    if Q.z_dr_0p1_0p2 < 0.1585582:
        z += -4.617362 * Q.z_dr_0p1_0p2 + 0.7321207
    if Q.C2 < 0.03578649:
        z += 19.00678 * Q.C2 - 0.6801859
    if Q.e3 < 1.960701e-05:
        z += -11930.51 * Q.e3 + 0.2339216
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -2849.31 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += -1947.87 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1_plus_lam2 < 0.00609665:
        z += 1546.873 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.lam1_plus_lam2)
    if Q.planar_flow < 0.1115136 and Q.sum_pt_top5 < 658.125:
        z += -0.01750949 * (0.1115136 - Q.planar_flow) * (658.125 - Q.sum_pt_top5)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -501.0356 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.planar_flow < 0.1115136 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -40.21575 * (0.1115136 - Q.planar_flow) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.sum_zz_dr2 > 0.002074109 and Q.sj2_dr < 0.1872617:
        z += -8027.624 * (Q.sum_zz_dr2 - 0.002074109) * (0.1872617 - Q.sj2_dr)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 1.0:
        z += 14.30345 * (Q.z_dr_0p05_0p1 - 0.7509095) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.psi_0p1 > 0.8155839 and Q.centroid_offset > 0.03776099:
        z += 1288.752 * (Q.psi_0p1 - 0.8155839) * (Q.centroid_offset - 0.03776099)
    if Q.sum_z_dr2 > 0.008678045 and Q.log_sum_pt > 6.267538:
        z += -2660.57 * (Q.sum_z_dr2 - 0.008678045) * (Q.log_sum_pt - 6.267538)
    if Q.sum_zz_dr2 > 0.002074109 and Q.log_sum_pt > 6.267538:
        z += 497.9112 * (Q.sum_zz_dr2 - 0.002074109) * (Q.log_sum_pt - 6.267538)
    if Q.lam1_plus_lam2 > 0.01323868 and Q.sum_pt > 488.9312:
        z += 1.910037 * (Q.lam1_plus_lam2 - 0.01323868) * (Q.sum_pt - 488.9312)
    if Q.LHA > 0.2160559 and Q.sum_pt > 813.4156:
        z += -0.06288978 * (Q.LHA - 0.2160559) * (Q.sum_pt - 813.4156)
    if Q.e2 > 0.03556091 and Q.sum_pt_top5 > 658.125:
        z += 0.3439116 * (Q.e2 - 0.03556091) * (Q.sum_pt_top5 - 658.125)
    if Q.sum_zz_dr2 > 0.008168571 and Q.sum_pt > 488.9312:
        z += 1.412787 * (Q.sum_zz_dr2 - 0.008168571) * (Q.sum_pt - 488.9312)
    if Q.eccentricity > 0.9781608 and Q.D2_b2 < 0.08499387:
        z += 272.9045 * (Q.eccentricity - 0.9781608) * (0.08499387 - Q.D2_b2)
    if Q.psi_0p1 > 0.6332416 and Q.eccentricity > 0.9704496:
        z += 118.6674 * (Q.psi_0p1 - 0.6332416) * (Q.eccentricity - 0.9704496)
    if Q.centroid_offset > 0.02355416 and Q.eccentricity > 0.9841966:
        z += 2067.944 * (Q.centroid_offset - 0.02355416) * (Q.eccentricity - 0.9841966)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.dr0_5 > 0.290089:
        z += -423.5311 * (Q.z_dr_0p05_0p1 - 0.7509095) * (Q.dr0_5 - 0.290089)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.dr0_5 > 0.1925897:
        z += -214.2399 * (Q.z_dr_0p05_0p1 - 0.7509095) * (Q.dr0_5 - 0.1925897)
    if Q.centroid_offset > 0.04990367 and Q.eta_5 > 0.1122437:
        z += -10115.42 * (Q.centroid_offset - 0.04990367) * (Q.eta_5 - 0.1122437)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.dr0_5 > 0.171998:
        z += 73.61012 * (Q.z_dr_0p05_0p1 - 0.7509095) * (Q.dr0_5 - 0.171998)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.165811
    if Q.N2 < 0.2233283:
        z += -3.635394 * Q.N2 + 0.8118861
    if Q.zdr_0 < 0.0211821:
        z += 15.11235 * Q.zdr_0 - 0.3201114
    if Q.lam2 < 0.001130645:
        z += -723.9706 * Q.lam2 + 1.475963
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -288.623 * Q.lam2 + 0.9837395
    if Q.sum_z_dr2 < 0.006679471:
        z += 1231.263 * Q.sum_z_dr2 - 5.290538
    if 0.006679471 <= Q.sum_z_dr2 < 0.007520088:
        z += 714.192 * Q.sum_z_dr2 - 1.836778
    if 0.007520088 <= Q.sum_z_dr2 < 0.01323868:
        z += -617.9864 * Q.sum_z_dr2 + 8.181321
    if Q.sum_zz_dr2 < 0.006390125:
        z += -263.5552 * Q.sum_zz_dr2 - 0.2104191
    if 0.006390125 <= Q.sum_zz_dr2 < 0.01165737:
        z += 443.3829 * Q.sum_zz_dr2 - 4.727841
    if 0.01165737 <= Q.sum_zz_dr2 < 0.01716248:
        z += -80.07809 * Q.sum_zz_dr2 + 1.374339
    if Q.e2 < 0.04110972:
        z += -92.65727 * Q.e2 + 2.987922
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += -7.431931 * Q.e2 - 0.5156674
    if 0.05028464 <= Q.e2 < 0.06344108:
        z += 67.60034 * Q.e2 - 4.288638
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 459.3791 * Q.sum_z_dr2_top3 - 0.9883853
    if Q.lam1 < 0.00595415:
        z += -522.0205 * Q.lam1 + 3.416093
    if 0.00595415 <= Q.lam1 < 0.00733008:
        z += -407.6607 * Q.lam1 + 2.735177
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 241.9993 * Q.lam1 - 2.026883
    if 0.1789613 <= Q.sj3_dr_max < 0.1879486:
        z += 16.69944 * Q.sj3_dr_max - 2.988555
    if 0.1879486 <= Q.sj3_dr_max < 0.233678:
        z += 13.10413 * Q.sj3_dr_max - 2.312821
    if 0.233678 <= Q.sj3_dr_max < 0.2623172:
        z += 4.296096 * Q.sj3_dr_max - 0.2545774
    if Q.sj3_dr_max >= 0.2623172:
        z += -4.413316 * Q.sj3_dr_max + 2.030052
    if Q.dr_0 < 0.08082334:
        z += -4.210543 * Q.dr_0 + 0.3403101
    if Q.z_dr_0_0p05 < 0.1515405:
        z += -3.689929 * Q.z_dr_0_0p05 + 0.5591735
    if Q.lam1_plus_lam2 < 0.002635418:
        z += -437.907 * Q.lam1_plus_lam2 + 1.154068
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += -10.32099 * Q.z_dr_0p05_0p1 + 7.750129
    if 0.121681 <= Q.max_dr < 0.2215867:
        z += -4.049962 * Q.max_dr + 0.4928033
    if Q.max_dr >= 0.2215867:
        z += 0.2336531 * Q.max_dr - 0.456389
    if 0.4008925 <= Q.psi_0p1 < 0.6332416:
        z += -0.5143756 * Q.psi_0p1 + 0.2062093
    if 0.6332416 <= Q.psi_0p1 < 0.9761279:
        z += 1.600971 * Q.psi_0p1 - 1.133316
    if Q.psi_0p1 >= 0.9761279:
        z += -12.86349 * Q.psi_0p1 + 12.98585
    if Q.n_dr_0p1_0p2 < 1.0:
        z += -0.2256762 * Q.n_dr_0p1_0p2 + 0.2256762
    if Q.tau1 < 0.1027642:
        z += 15.99358 * Q.tau1 - 2.097286
    if 0.1027642 <= Q.tau1 < 0.1136369:
        z += 41.73022 * Q.tau1 - 4.742093
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1872617:
        z += -342.9925 * (0.2233283 - Q.N2) * (0.1872617 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.sum_z_dr2 > 0.008678045:
        z += -2547.854 * (0.2233283 - Q.N2) * (Q.sum_z_dr2 - 0.008678045)
    if Q.N2 < 0.2233283 and Q.LHA > 0.2931906:
        z += 202.9236 * (0.2233283 - Q.N2) * (Q.LHA - 0.2931906)
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1591713:
        z += 429.388 * (0.2233283 - Q.N2) * (0.1591713 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3255822:
        z += -152.1322 * (0.2233283 - Q.N2) * (Q.LHA - 0.3255822)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.009070271 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
    if Q.N2 < 0.2233283 and Q.e3 < 8.147744e-05:
        z += -65964.91 * (0.2233283 - Q.N2) * (8.147744e-05 - Q.e3)
    if Q.sum_z_dr2_top3 < 0.002151568 and Q.eccentricity > 0.9598562:
        z += -12659.69 * (0.002151568 - Q.sum_z_dr2_top3) * (Q.eccentricity - 0.9598562)
    if Q.planar_flow < 0.1950135 and Q.mean_phi > -0.01753483:
        z += -39.41226 * (0.1950135 - Q.planar_flow) * (Q.mean_phi - -0.01753483)
    if Q.sum_z_dr2 < 0.006679471 and Q.dr12 > 0.1587481:
        z += -21980.22 * (0.006679471 - Q.sum_z_dr2) * (Q.dr12 - 0.1587481)
    if Q.sum_zz_dr2 < 0.006390125 and Q.dr12 > 0.1587481:
        z += 14912.74 * (0.006390125 - Q.sum_zz_dr2) * (Q.dr12 - 0.1587481)
    if Q.sum_z_dr2 < 0.006679471 and Q.pt_entropy > 1.797618:
        z += -174.3099 * (0.006679471 - Q.sum_z_dr2) * (Q.pt_entropy - 1.797618)
    if Q.N2 < 0.2233283 and Q.pt_7 < 25.57812:
        z += -0.6556836 * (0.2233283 - Q.N2) * (25.57812 - Q.pt_7)
    if Q.planar_flow < 0.1950135 and Q.n_dr_0p05_0p1 < 2.0:
        z += -0.9945458 * (0.1950135 - Q.planar_flow) * (2.0 - Q.n_dr_0p05_0p1)
    if Q.sum_z_dr2 < 0.01323868 and Q.centroid_offset > 0.04990367:
        z += -41217.64 * (0.01323868 - Q.sum_z_dr2) * (Q.centroid_offset - 0.04990367)
    if Q.sj3_dr_max > 0.233678 and Q.pt_1 < 182.125:
        z += -0.05510387 * (Q.sj3_dr_max - 0.233678) * (182.125 - Q.pt_1)
    if Q.planar_flow < 0.1950135 and Q.sum_pt_top5 > 402.625:
        z += 0.01549907 * (0.1950135 - Q.planar_flow) * (Q.sum_pt_top5 - 402.625)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += 156.336 * (Q.z_dr_0p05_0p1 - 0.7509095) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sj3_dr_max > 0.1426152 and Q.pt_1 < 169.875:
        z += -0.0843332 * (Q.sj3_dr_max - 0.1426152) * (169.875 - Q.pt_1)
    if Q.sj3_dr_max > 0.1879486 and Q.pt_1 < 169.875:
        z += 0.1549218 * (Q.sj3_dr_max - 0.1879486) * (169.875 - Q.pt_1)
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
