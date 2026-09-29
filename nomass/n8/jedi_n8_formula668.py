"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; no mass observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.7% (the network: 65.8%); same class as the network for 90.2% of jets.

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
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z1                 pT share of subjet 1 of 3 (by pT)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.pt1_over_pt0           pT1 / pT0
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.z_3rd                  3rd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_6               |Δη| of particle 6
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_5                  ΔR between particle 5 and the hardest particle
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_5                  Δη of particle 5
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
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        n_for_90pct=ncum(0.9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z1=subjets(3)["z"][0],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        zdr_3=z[3] * dr[3],
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        z_3rd=zs[2],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        abseta_0=abs(eta[0]),
        abseta_6=abs(eta[6]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_5=math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_5=eta[5],
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
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = 6.829228
    if Q.planar_flow < 0.1484197:
        z += -10.61742 * Q.planar_flow + 1.575835
    if Q.lam1_plus_lam2 < 0.003562611:
        z += 96.43521 * Q.lam1_plus_lam2 + 4.009648
    if 0.003562611 <= Q.lam1_plus_lam2 < 0.004372139:
        z += -542.3144 * Q.lam1_plus_lam2 + 6.285264
    if 0.004372139 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -909.0283 * Q.lam1_plus_lam2 + 7.888589
    if Q.sum_z_dr2 < 0.005590289:
        z += 285.6243 * Q.sum_z_dr2 + 1.516969
    if 0.005590289 <= Q.sum_z_dr2 < 0.01323868:
        z += -337.971 * Q.sum_z_dr2 + 5.003047
    if 0.01323868 <= Q.sum_z_dr2 < 0.01882765:
        z += -94.60744 * Q.sum_z_dr2 + 1.781236
    if Q.sum_pt < 788.4484:
        z += 0.006051564 * Q.sum_pt - 4.771346
    if Q.sum_z_dr2_top3 < 0.007929074:
        z += 67.89665 * Q.sum_z_dr2_top3 - 0.5383576
    z += -1.199787 * Q.log_sum_pt
    if Q.tau2 < 0.01357153:
        z += 147.9512 * Q.tau2 - 2.007924
    if Q.sd_rg < 0.06545715:
        z += -1.83168 * Q.sd_rg + 0.02267826
    if 0.06545715 <= Q.sd_rg < 0.1116471:
        z += -10.37993 * Q.sd_rg + 0.582222
    if 0.1116471 <= Q.sd_rg < 0.1773029:
        z += 8.783162 * Q.sd_rg - 1.55728
    if Q.tau1 < 0.1027642:
        z += 15.0767 * Q.tau1 - 1.549346
    if Q.centroid_offset >= 0.04990367:
        z += -271.699 * Q.centroid_offset + 13.55878
    if Q.lam1 < 0.00595415:
        z += 300.5349 * Q.lam1 - 2.897763
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += 457.7199 * Q.lam1 - 3.833666
    if Q.sum_pt_top5 < 631.275:
        z += -0.003535965 * Q.sum_pt_top5 + 2.232166
    if Q.n_dr_0_0p05 >= 5.0:
        z += 0.3528285 * Q.n_dr_0_0p05 - 1.764142
    if Q.sum_z_dr < 0.01517359:
        z += 288.5617 * Q.sum_z_dr - 9.86293
    if 0.01517359 <= Q.sum_z_dr < 0.07608178:
        z += 83.66072 * Q.sum_z_dr - 6.753846
    if 0.07608178 <= Q.sum_z_dr < 0.08723651:
        z += 34.85429 * Q.sum_z_dr - 3.040567
    if Q.sum_zz_dr2 < 0.00718279:
        z += 133.338 * Q.sum_zz_dr2 - 0.01137038
    if 0.00718279 <= Q.sum_zz_dr2 < 0.01165737:
        z += -211.4986 * Q.sum_zz_dr2 + 2.465518
    if Q.e2 < 0.032347:
        z += -125.4335 * Q.e2 + 4.702401
    if 0.032347 <= Q.e2 < 0.06344108:
        z += -20.74363 * Q.e2 + 1.315998
    if Q.planar_flow < 0.1484197 and Q.centroid_offset < 0.04990367:
        z += -125.6617 * (0.1484197 - Q.planar_flow) * (0.04990367 - Q.centroid_offset)
    if Q.sum_z_dr2 < 0.01882765 and Q.M3 < 0.09029177:
        z += 698.023 * (0.01882765 - Q.sum_z_dr2) * (0.09029177 - Q.M3)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.dr0_6 > 0.1755206:
        z += 18764.62 * (0.004372139 - Q.lam1_plus_lam2) * (Q.dr0_6 - 0.1755206)
    if Q.planar_flow < 0.1484197 and Q.D2_b2 < 1.129616:
        z += 8.391523 * (0.1484197 - Q.planar_flow) * (1.129616 - Q.D2_b2)
    if Q.e3 < 8.147744e-05 and Q.D2_b2 < 0.2669656:
        z += -22748.67 * (8.147744e-05 - Q.e3) * (0.2669656 - Q.D2_b2)
    if Q.sum_z_dr2_top3 < 0.007929074 and Q.N3 > 0.7686247:
        z += 39.70859 * (0.007929074 - Q.sum_z_dr2_top3) * (Q.N3 - 0.7686247)
    if Q.sum_z_dr2 < 0.01882765 and Q.centroid_offset > 0.009480685:
        z += -4333.511 * (0.01882765 - Q.sum_z_dr2) * (Q.centroid_offset - 0.009480685)
    if Q.sum_pt < 788.4484 and Q.eccentricity > 0.9598562:
        z += -0.06639393 * (788.4484 - Q.sum_pt) * (Q.eccentricity - 0.9598562)
    if Q.D2_b2 < 0.380911 and Q.pt_5 > 24.57812:
        z += -0.0274359 * (0.380911 - Q.D2_b2) * (Q.pt_5 - 24.57812)
    if Q.tau1 < 0.06345984 and Q.dr0_6 > 0.1755206:
        z += -667.0335 * (0.06345984 - Q.tau1) * (Q.dr0_6 - 0.1755206)
    if Q.lam1_plus_lam2 < 0.003562611 and Q.dr0_7 > 0.1786203:
        z += 21132.37 * (0.003562611 - Q.lam1_plus_lam2) * (Q.dr0_7 - 0.1786203)
    if Q.e3 < 8.147744e-05 and Q.dr0_7 > 0.2405707:
        z += -526252.3 * (8.147744e-05 - Q.e3) * (Q.dr0_7 - 0.2405707)
    if Q.sum_pt < 739.5 and Q.D2_b2 < 1.345805:
        z += -0.00431158 * (739.5 - Q.sum_pt) * (1.345805 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.005590289 and Q.M3 < 0.09310137:
        z += -4044.156 * (0.005590289 - Q.sum_z_dr2) * (0.09310137 - Q.M3)
    if Q.sum_z_dr2 < 0.005590289 and Q.dr0_5 > 0.171998:
        z += 11465.98 * (0.005590289 - Q.sum_z_dr2) * (Q.dr0_5 - 0.171998)
    if Q.tau1 < 0.06345984 and Q.dr_5 > 0.1631114:
        z += -954.8474 * (0.06345984 - Q.tau1) * (Q.dr_5 - 0.1631114)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.125976
    if Q.lam1 < 0.002464291:
        z += -672.6807 * Q.lam1 + 2.340792
    if 0.002464291 <= Q.lam1 < 0.00595415:
        z += -243.7085 * Q.lam1 + 1.28368
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += 69.13169 * Q.lam1 - 0.5790175
    if 6.377723 <= Q.log_sum_pt < 6.572938:
        z += 5.049044 * Q.log_sum_pt - 32.2014
    if Q.log_sum_pt >= 6.572938:
        z += 10.61481 * Q.log_sum_pt - 68.78486
    if Q.sum_zz_dr2 < 0.005834489:
        z += -547.3539 * Q.sum_zz_dr2 + 3.19353
    if Q.z_7 < 0.05240025:
        z += 111.5158 * Q.z_7 - 6.492646
    if 0.05240025 <= Q.z_7 < 0.06164517:
        z += 70.22129 * Q.z_7 - 4.328804
    if Q.lam1_plus_lam2 < 0.008678045:
        z += 300.8018 * Q.lam1_plus_lam2 - 2.610372
    if Q.tau1 < 0.07283629:
        z += 5.755498 * Q.tau1 - 0.06881507
    if 0.07283629 <= Q.tau1 < 0.1027642:
        z += -11.70792 * Q.tau1 + 1.203156
    if 38.53125 <= Q.pt_7 < 53.4375:
        z += 0.1129212 * Q.pt_7 - 4.350994
    if Q.pt_7 >= 53.4375:
        z += -0.007636085 * Q.pt_7 + 2.091285
    if Q.zdr_0 < 0.0211821:
        z += 39.7978 * Q.zdr_0 - 0.843001
    if Q.sum_z_dr2 < 0.00609665:
        z += 517.0056 * Q.sum_z_dr2 - 3.152002
    if Q.sum_z_dr < 0.1019409:
        z += 28.67751 * Q.sum_z_dr - 2.923412
    if Q.sum_pt_top5 >= 531.1875:
        z += -0.006136858 * Q.sum_pt_top5 + 3.259822
    if Q.sj3_dr_max < 0.1594012:
        z += -4.440975 * Q.sj3_dr_max - 0.08560653
    if 0.1594012 <= Q.sj3_dr_max < 0.3456459:
        z += 4.26054 * Q.sj3_dr_max - 1.472638
    if Q.max_dr < 0.1326115:
        z += 7.282849 * Q.max_dr - 0.9657899
    if Q.sum_pt < 667.0063:
        z += 0.001911396 * Q.sum_pt - 1.889239
    if 667.0063 <= Q.sum_pt < 988.4078:
        z += 0.009778114 * Q.sum_pt - 7.136389
    if Q.sum_pt >= 988.4078:
        z += 0.007866718 * Q.sum_pt - 5.24715
    if Q.z_6 >= 0.06727211:
        z += 12.48657 * Q.z_6 - 0.8399979
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += 8344.636 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.pt_7 > 34.53125 and Q.e3 < 2.955458e-05:
        z += -2179.863 * (Q.pt_7 - 34.53125) * (2.955458e-05 - Q.e3)
    if Q.z_7 < 0.06164517 and Q.e3 < 0.0001869378:
        z += 237061.7 * (0.06164517 - Q.z_7) * (0.0001869378 - Q.e3)
    if Q.z_7 < 0.05240025 and Q.D2_b2 < 0.1236856:
        z += -376.9237 * (0.05240025 - Q.z_7) * (0.1236856 - Q.D2_b2)
    if Q.z_7 < 0.06164517 and Q.sum_z_dr2_top2 < 0.01403324:
        z += 1135.6 * (0.06164517 - Q.z_7) * (0.01403324 - Q.sum_z_dr2_top2)
    if Q.log_sum_pt > 6.377723 and Q.dr_0 < 0.08082334:
        z += -73.24345 * (Q.log_sum_pt - 6.377723) * (0.08082334 - Q.dr_0)
    if Q.lam1 < 0.008375572 and Q.dr_0 < 0.1442881:
        z += 2507.463 * (0.008375572 - Q.lam1) * (0.1442881 - Q.dr_0)
    if Q.z_7 < 0.06164517 and Q.zdr_0 < 0.03981924:
        z += 772.4849 * (0.06164517 - Q.z_7) * (0.03981924 - Q.zdr_0)
    if Q.log_sum_pt > 6.572938 and Q.sum_z_dr2_top3 < 0.007929074:
        z += -673.3563 * (Q.log_sum_pt - 6.572938) * (0.007929074 - Q.sum_z_dr2_top3)
    if Q.lam1 < 0.008375572 and Q.pt_5 > 24.57812:
        z += -1.329911 * (0.008375572 - Q.lam1) * (Q.pt_5 - 24.57812)
    if Q.lam2 < 0.003408389 and Q.tau21_b2 < 0.2691019:
        z += 448.0979 * (0.003408389 - Q.lam2) * (0.2691019 - Q.tau21_b2)
    if Q.log_sum_pt > 6.638339 and Q.dr01 < 0.1410336:
        z += 23.39682 * (Q.log_sum_pt - 6.638339) * (0.1410336 - Q.dr01)
    if Q.z_7 < 0.06164517 and Q.dr01 < 0.1410336:
        z += -71.70545 * (0.06164517 - Q.z_7) * (0.1410336 - Q.dr01)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.6694068
    if Q.pt_7 < 53.4375:
        z += 0.1388083 * Q.pt_7 - 2.445188
    if Q.pt_7 >= 53.4375:
        z += 0.09305043 * Q.pt_7
    if Q.lam1 < 0.003377388:
        z += -75.55008 * Q.lam1 + 0.05693404
    if 0.003377388 <= Q.lam1 < 0.00733008:
        z += 50.15011 * Q.lam1 - 0.3676043
    if 0.02320757 <= Q.z_7 < 0.02807091:
        z += -97.56583 * Q.z_7 + 2.264266
    if Q.z_7 >= 0.02807091:
        z += -67.49884 * Q.z_7 + 1.420258
    if Q.log_sum_pt < 6.701242:
        z += -1.572963 * Q.log_sum_pt + 10.5408
    if Q.log_sum_pt >= 6.896095:
        z += 20.62273 * Q.log_sum_pt - 142.2163
    if 0.09323897 <= Q.LHA < 0.111565:
        z += 6.444664 * Q.LHA - 0.6008939
    if Q.LHA >= 0.111565:
        z += -9.593 * Q.LHA + 1.188349
    if Q.e2 < 0.04447357:
        z += 9.173991 * Q.e2 - 0.4080001
    if Q.sum_pt < 615.875:
        z += -0.02159628 * Q.sum_pt + 16.05088
    if 615.875 <= Q.sum_pt < 763.825:
        z += -0.01415763 * Q.sum_pt + 11.4696
    if 763.825 <= Q.sum_pt < 840.0195:
        z += -0.008604947 * Q.sum_pt + 7.228324
    if 937.0312 <= Q.sum_pt < 988.4078:
        z += 0.01460584 * Q.sum_pt - 13.68613
    if Q.sum_pt >= 988.4078:
        z += -0.01749948 * Q.sum_pt + 18.04702
    if Q.C2 < 0.0423228:
        z += -36.45481 * Q.C2 + 1.54287
    if Q.sum_zz_dr2 < 0.001101266:
        z += 614.0755 * Q.sum_zz_dr2 - 1.960612
    if 0.001101266 <= Q.sum_zz_dr2 < 0.0030133:
        z += 301.1895 * Q.sum_zz_dr2 - 1.616041
    if 0.0030133 <= Q.sum_zz_dr2 < 0.00718279:
        z += 169.9169 * Q.sum_zz_dr2 - 1.220478
    if Q.lam1_plus_lam2 < 0.001653836:
        z += -558.0646 * Q.lam1_plus_lam2 + 2.856055
    if 0.001653836 <= Q.lam1_plus_lam2 < 0.007520088:
        z += -218.446 * Q.lam1_plus_lam2 + 2.294382
    if 0.007520088 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -113.9527 * Q.lam1_plus_lam2 + 1.508583
    if Q.sum_pt_top5 < 716.8828:
        z += 0.005428661 * Q.sum_pt_top5 - 3.891714
    if Q.zdr_0 < 0.02383244:
        z += -23.65836 * Q.zdr_0 + 0.5638364
    if Q.lam1 < 0.00733008 and Q.max_dr < 0.2507612:
        z += 1241.358 * (0.00733008 - Q.lam1) * (0.2507612 - Q.max_dr)
    if Q.lam1 < 0.00733008 and Q.pt_6 < 62.25:
        z += -3.611451 * (0.00733008 - Q.lam1) * (62.25 - Q.pt_6)
    if Q.log_sum_pt < 6.605974 and Q.D2_b2 < 1.129616:
        z += 4.569606 * (6.605974 - Q.log_sum_pt) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt < 6.701242 and Q.D2_b2 < 1.129616:
        z += -3.474619 * (6.701242 - Q.log_sum_pt) * (1.129616 - Q.D2_b2)
    if Q.centroid_offset < 0.005576073 and Q.C3 < 0.01685631:
        z += -17073.32 * (0.005576073 - Q.centroid_offset) * (0.01685631 - Q.C3)
    if Q.sum_pt_top5 > 791.125 and Q.abseta_0 < 0.01425934:
        z += -0.545935 * (Q.sum_pt_top5 - 791.125) * (0.01425934 - Q.abseta_0)
    if Q.sum_pt > 988.4078 and Q.abseta_0 < 0.01425934:
        z += 0.6782656 * (Q.sum_pt - 988.4078) * (0.01425934 - Q.abseta_0)
    if Q.C2 < 0.0423228 and Q.C2_b2 < 0.02415398:
        z += -1974.901 * (0.0423228 - Q.C2) * (0.02415398 - Q.C2_b2)
    if Q.sum_pt > 937.0312 and Q.n_pt_above_50 > 5.0:
        z += -0.003418746 * (Q.sum_pt - 937.0312) * (Q.n_pt_above_50 - 5.0)
    return max(0.0, z)


def neuron_3(Q):
    z = -2.820237
    if 0.06663269 <= Q.sum_z_dr < 0.08723651:
        z += 154.345 * Q.sum_z_dr - 10.28442
    if Q.sum_z_dr >= 0.08723651:
        z += 137.5965 * Q.sum_z_dr - 8.823339
    if 0.1682655 <= Q.sj2_dr < 0.1778793:
        z += 18.72822 * Q.sj2_dr - 3.151314
    if 0.1778793 <= Q.sj2_dr < 0.2687922:
        z += 49.38243 * Q.sj2_dr - 8.604063
    if Q.sj2_dr >= 0.2687922:
        z += 9.285189 * Q.sj2_dr + 2.17376
    if 0.00609665 <= Q.lam1_plus_lam2 < 0.01323868:
        z += 473.8321 * Q.lam1_plus_lam2 - 2.888788
    if 0.01323868 <= Q.lam1_plus_lam2 < 0.01882765:
        z += 277.3975 * Q.lam1_plus_lam2 - 0.2882552
    if Q.lam1_plus_lam2 >= 0.01882765:
        z += 525.8229 * Q.lam1_plus_lam2 - 4.965522
    if 0.006390125 <= Q.sum_zz_dr2 < 0.00718279:
        z += -418.9337 * Q.sum_zz_dr2 + 2.677039
    if 0.00718279 <= Q.sum_zz_dr2 < 0.01165737:
        z += 311.342 * Q.sum_zz_dr2 - 2.568378
    if Q.sum_zz_dr2 >= 0.01165737:
        z += 351.9385 * Q.sum_zz_dr2 - 3.041626
    if Q.sum_z_dr2 >= 0.008678045:
        z += -1087.476 * Q.sum_z_dr2 + 9.437164
    if 0.0811449 <= Q.tau1 < 0.1136369:
        z += -59.96551 * Q.tau1 + 4.865895
    if Q.tau1 >= 0.1136369:
        z += -32.64759 * Q.tau1 + 1.761572
    if Q.lam1 >= 0.008375572:
        z += 244.7973 * Q.lam1 - 2.050318
    if Q.lam2 >= 0.000537286:
        z += 198.4636 * Q.lam2 - 0.1066317
    if Q.max_dr >= 0.1117619:
        z += 7.398827 * Q.max_dr - 0.8269066
    if Q.sj2_dr > 0.1682655 and Q.sum_pt < 988.4078:
        z += -0.03021284 * (Q.sj2_dr - 0.1682655) * (988.4078 - Q.sum_pt)
    if Q.sj2_dr > 0.1682655 and Q.tau2 < 0.06297984:
        z += 232.6606 * (Q.sj2_dr - 0.1682655) * (0.06297984 - Q.tau2)
    if Q.sj2_dr > 0.1682655 and Q.n_dr_0p05_0p1 < 7.0:
        z += -4.346659 * (Q.sj2_dr - 0.1682655) * (7.0 - Q.n_dr_0p05_0p1)
    if Q.lam1_plus_lam2 > 0.00609665 and Q.pt_6 > 31.90625:
        z += -2.907329 * (Q.lam1_plus_lam2 - 0.00609665) * (Q.pt_6 - 31.90625)
    if Q.lam1_plus_lam2 > 0.00609665 and Q.log_sum_pt > 6.192222:
        z += -377.974 * (Q.lam1_plus_lam2 - 0.00609665) * (Q.log_sum_pt - 6.192222)
    if Q.sj2_dr > 0.1591713 and Q.n_dr_0p1_0p2 > 1.0:
        z += 1.878094 * (Q.sj2_dr - 0.1591713) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.sj2_dr > 0.3003793 and Q.z_dr_0p05_0p1 > 0.08878489:
        z += -55.2368 * (Q.sj2_dr - 0.3003793) * (Q.z_dr_0p05_0p1 - 0.08878489)
    if Q.lam1_plus_lam2 > 0.01323868 and Q.eccentricity > 0.9884745:
        z += -9801.528 * (Q.lam1_plus_lam2 - 0.01323868) * (Q.eccentricity - 0.9884745)
    if Q.sj2_dr > 0.1778793 and Q.eccentricity > 0.9458207:
        z += 781.3742 * (Q.sj2_dr - 0.1778793) * (Q.eccentricity - 0.9458207)
    if Q.sj2_dr > 0.1682655 and Q.eccentricity > 0.9458207:
        z += -526.6101 * (Q.sj2_dr - 0.1682655) * (Q.eccentricity - 0.9458207)
    if Q.sj2_dr > 0.1778793 and Q.sum_z_dr2_top2 < 0.001864148:
        z += -30071.65 * (Q.sj2_dr - 0.1778793) * (0.001864148 - Q.sum_z_dr2_top2)
    if Q.sj2_dr > 0.2687922 and Q.sum_z_dr2_top2 < 0.007639643:
        z += 4042.975 * (Q.sj2_dr - 0.2687922) * (0.007639643 - Q.sum_z_dr2_top2)
    return max(0.0, z)


def neuron_4(Q):
    z = -0.8285995
    if Q.N2 < 0.2233283:
        z += -25.55371 * Q.N2 + 5.706864
    if Q.sum_z_dr2 < 0.002635418:
        z += 426.5068 * Q.sum_z_dr2 - 16.06848
    if 0.002635418 <= Q.sum_z_dr2 < 0.006679471:
        z += 2154.856 * Q.sum_z_dr2 - 20.6234
    if 0.006679471 <= Q.sum_z_dr2 < 0.008678045:
        z += 1609.376 * Q.sum_z_dr2 - 16.97988
    if 0.008678045 <= Q.sum_z_dr2 < 0.01323868:
        z += 660.7965 * Q.sum_z_dr2 - 8.74807
    if Q.e3 < 0.0001869378:
        z += -16048.64 * Q.e3 + 3.000099
    if Q.sj3_dr_min < 0.2089872:
        z += -16.0331 * Q.sj3_dr_min + 3.350713
    if Q.lam2 < 0.001130645:
        z += 2149.562 * Q.lam2 - 2.430391
    if Q.lam1 < 0.002464291:
        z += 398.0957 * Q.lam1 + 0.5214767
    if 0.002464291 <= Q.lam1 < 0.01200373:
        z += -157.5041 * Q.lam1 + 1.890636
    if Q.C2_b2 < 0.009032972:
        z += -129.4216 * Q.C2_b2 + 1.808187
    if 0.009032972 <= Q.C2_b2 < 0.02415398:
        z += -42.26738 * Q.C2_b2 + 1.020926
    if Q.max_dr >= 0.0931108:
        z += 7.511322 * Q.max_dr - 0.6993852
    if Q.sj3_dr_max < 0.1789613:
        z += -3.676741 * Q.sj3_dr_max + 1.10744
    if 0.1789613 <= Q.sj3_dr_max < 0.213399:
        z += 14.13086 * Q.sj3_dr_max - 2.079431
    if 0.213399 <= Q.sj3_dr_max < 0.3012016:
        z += 6.328456 * Q.sj3_dr_max - 0.4144072
    if 0.3012016 <= Q.sj3_dr_max < 0.3456459:
        z += 10.0052 * Q.sj3_dr_max - 1.521847
    if Q.sj3_dr_max >= 0.3456459:
        z += 4.746589 * Q.sj3_dr_max + 0.295769
    if Q.sum_z_dr2_top5 < 0.007164202:
        z += -157.6721 * Q.sum_z_dr2_top5 + 1.129594
    if Q.sum_z_dr < 0.07608178:
        z += -36.70969 * Q.sum_z_dr + 2.792938
    if Q.tau2 < 0.01357153:
        z += 71.56191 * Q.tau2 - 0.9712048
    if Q.e2 < 0.02045966:
        z += 79.37379 * Q.e2 - 3.530036
    if 0.02045966 <= Q.e2 < 0.04447357:
        z += 139.4359 * Q.e2 - 4.758887
    if Q.e2 >= 0.04447357:
        z += 60.06216 * Q.e2 - 1.228851
    if Q.sum_zz_dr2 < 0.0030133:
        z += -1150.657 * Q.sum_zz_dr2 + 12.06882
    if 0.0030133 <= Q.sum_zz_dr2 < 0.006390125:
        z += -1476.093 * Q.sum_zz_dr2 + 13.04946
    if 0.006390125 <= Q.sum_zz_dr2 < 0.01165737:
        z += -1220.776 * Q.sum_zz_dr2 + 11.41795
    if Q.sum_zz_dr2 >= 0.01165737:
        z += -325.4356 * Q.sum_zz_dr2 + 0.9806353
    if Q.sj2_dr >= 0.2179769:
        z += -27.62156 * Q.sj2_dr + 6.020862
    if Q.C2 >= 0.06729223:
        z += -81.08585 * Q.C2 + 5.456447
    if 10.17512 <= Q.ptdr0_3 < 17.94219:
        z += 0.05720771 * Q.ptdr0_3 - 0.582095
    if Q.ptdr0_3 >= 17.94219:
        z += -0.1043533 * Q.ptdr0_3 + 2.316663
    if Q.log_sum_pt < 6.638339:
        z += 1.911045 * Q.log_sum_pt - 12.68616
    if Q.N2 < 0.2233283 and Q.sum_pt < 868.5094:
        z += 0.08120527 * (0.2233283 - Q.N2) * (868.5094 - Q.sum_pt)
    if Q.N2 < 0.2233283 and Q.planar_flow < 0.4007947:
        z += -32.36913 * (0.2233283 - Q.N2) * (0.4007947 - Q.planar_flow)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1626038:
        z += -31.40704 * (0.2233283 - Q.N2) * (0.1626038 - Q.abseta_7)
    if Q.sj2_dr > 0.2414351 and Q.sj3_z3 < 0.1989187:
        z += 81.63368 * (Q.sj2_dr - 0.2414351) * (0.1989187 - Q.sj3_z3)
    if Q.N2 < 0.2233283 and Q.pt_6 < 24.42188:
        z += -1.817879 * (0.2233283 - Q.N2) * (24.42188 - Q.pt_6)
    if Q.sum_z_dr2_top2 < 0.006299534 and Q.centroid_offset > 0.01096064:
        z += 19438.09 * (0.006299534 - Q.sum_z_dr2_top2) * (Q.centroid_offset - 0.01096064)
    if Q.sj2_dr > 0.2414351 and Q.D2_b2 < 1.129616:
        z += -24.39929 * (Q.sj2_dr - 0.2414351) * (1.129616 - Q.D2_b2)
    if Q.sj3_dr_max > 0.3456459 and Q.D2_b2 < 1.129616:
        z += 19.41901 * (Q.sj3_dr_max - 0.3456459) * (1.129616 - Q.D2_b2)
    if Q.max_dr > 0.0931108 and Q.absphi_0 < 0.04019165:
        z += 130.1169 * (Q.max_dr - 0.0931108) * (0.04019165 - Q.absphi_0)
    if Q.sum_zz_dr2 < 0.01165737 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -5279.709 * (0.01165737 - Q.sum_zz_dr2) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.e2 < 0.04447357 and Q.psi_0p2 > 0.9435576:
        z += -605.9128 * (0.04447357 - Q.e2) * (Q.psi_0p2 - 0.9435576)
    if Q.sum_z_dr2 < 0.01323868 and Q.psi_0p2 > 0.9435576:
        z += 5743.282 * (0.01323868 - Q.sum_z_dr2) * (Q.psi_0p2 - 0.9435576)
    if Q.lam2 < 0.001130645 and Q.dr_max_012 < 0.2002199:
        z += -4460.213 * (0.001130645 - Q.lam2) * (0.2002199 - Q.dr_max_012)
    if Q.sj3_dr_max < 0.3012016 and Q.z_dr_0p2_0p4 < 0.20552:
        z += -9.701892 * (0.3012016 - Q.sj3_dr_max) * (0.20552 - Q.z_dr_0p2_0p4)
    if Q.sj2_dr > 0.2414351 and Q.pt_2 < 154.25:
        z += 0.2579101 * (Q.sj2_dr - 0.2414351) * (154.25 - Q.pt_2)
    if Q.sj2_dr > 0.2179769 and Q.z_3rd < 0.1818564:
        z += -144.5569 * (Q.sj2_dr - 0.2179769) * (0.1818564 - Q.z_3rd)
    if Q.sj3_dr_max < 0.3012016 and Q.log_sum_pt < 6.733425:
        z += -53.08542 * (0.3012016 - Q.sj3_dr_max) * (6.733425 - Q.log_sum_pt)
    if Q.N2 < 0.2233283 and Q.sum_pt < 988.4078:
        z += -0.07356311 * (0.2233283 - Q.N2) * (988.4078 - Q.sum_pt)
    if Q.D2 < 0.7459513 and Q.pt_6 > 19.46875:
        z += 0.08965792 * (0.7459513 - Q.D2) * (Q.pt_6 - 19.46875)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.1193543
    if Q.LHA < 0.1767241:
        z += -34.24466 * Q.LHA + 7.602199
    if 0.1767241 <= Q.LHA < 0.2160559:
        z += -39.41706 * Q.LHA + 8.516287
    if Q.log_sum_pt >= 6.896095:
        z += -22.04923 * Q.log_sum_pt + 152.0536
    if Q.tau1 < 0.03459477:
        z += 14.96536 * Q.tau1 - 0.5177234
    if Q.sum_pt < 715.4688:
        z += 0.01433696 * Q.sum_pt - 10.49208
    if 715.4688 <= Q.sum_pt < 788.4484:
        z += 0.003212316 * Q.sum_pt - 2.532746
    if Q.centroid_offset < 0.03117077:
        z += -18.16522 * Q.centroid_offset + 0.4671024
    if 0.03117077 <= Q.centroid_offset < 0.03776099:
        z += 15.0407 * Q.centroid_offset - 0.5679516
    if Q.sum_z_dr < 0.007673833:
        z += -303.2113 * Q.sum_z_dr + 2.326793
    if Q.z_7 < 0.02807091:
        z += -153.7121 * Q.z_7 + 4.480214
    if 0.02807091 <= Q.z_7 < 0.03243272:
        z += -37.91399 * Q.z_7 + 1.229654
    if Q.sum_pt_top5 < 430.75:
        z += 0.03731265 * Q.sum_pt_top5 - 16.07243
    if Q.dr_0 < 0.04649465:
        z += 22.52606 * Q.dr_0 - 1.047341
    if Q.sj3_dr_max < 0.3012016:
        z += 8.644842 * Q.sj3_dr_max - 2.60384
    if Q.max_dr < 0.177305:
        z += -9.575005 * Q.max_dr + 1.697696
    if Q.e3 < 0.0005116989:
        z += -2164.303 * Q.e3 + 1.107472
    if Q.z_6 < 0.02160287:
        z += -172.2652 * Q.z_6 + 4.898406
    if 0.02160287 <= Q.z_6 < 0.06081235:
        z += -30.0178 * Q.z_6 + 1.825453
    if Q.pt_7 < 20.125:
        z += 0.1758003 * Q.pt_7 - 3.537982
    if 34.53125 <= Q.pt_7 < 43.5:
        z += -0.05476759 * Q.pt_7 + 1.891193
    if Q.pt_7 >= 43.5:
        z += -0.1814207 * Q.pt_7 + 7.400605
    if Q.sum_z_dr2 < 0.0009641429:
        z += -1820.832 * Q.sum_z_dr2 + 2.623263
    if 0.0009641429 <= Q.sum_z_dr2 < 0.005019719:
        z += -213.9574 * Q.sum_z_dr2 + 1.074006
    if Q.pt_4 < 31.125:
        z += -0.1328554 * Q.pt_4 + 4.135124
    if Q.z_4 < 0.03747769:
        z += 101.4028 * Q.z_4 - 3.800342
    if Q.LHA < 0.2160559 and Q.z_6 < 0.09543973:
        z += -234.566 * (0.2160559 - Q.LHA) * (0.09543973 - Q.z_6)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -198.2519 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.e2 < 0.03556091 and Q.zdr_0 > 0.004918231:
        z += 2800.847 * (0.03556091 - Q.e2) * (Q.zdr_0 - 0.004918231)
    if Q.sum_z_dr2 < 0.002635418 and Q.centroid_offset < 0.01627885:
        z += 57834.65 * (0.002635418 - Q.sum_z_dr2) * (0.01627885 - Q.centroid_offset)
    if Q.LHA < 0.1767241 and Q.zdr_0 < 0.006292091:
        z += -1979.868 * (0.1767241 - Q.LHA) * (0.006292091 - Q.zdr_0)
    if Q.sum_pt < 715.4688 and Q.zdr_0 < 0.007798268:
        z += 1.101738 * (715.4688 - Q.sum_pt) * (0.007798268 - Q.zdr_0)
    if Q.sj3_dr_max < 0.3012016 and Q.pt1_dr01 > 12.6865:
        z += -0.6523634 * (0.3012016 - Q.sj3_dr_max) * (Q.pt1_dr01 - 12.6865)
    if Q.sj3_dr_max < 0.3012016 and Q.ptdr0_2 > 11.87288:
        z += -0.5075533 * (0.3012016 - Q.sj3_dr_max) * (Q.ptdr0_2 - 11.87288)
    if Q.e2 < 0.03556091 and Q.planar_flow < 0.3220738:
        z += 63.808 * (0.03556091 - Q.e2) * (0.3220738 - Q.planar_flow)
    if Q.z_7 < 0.04939969 and Q.absphi_0 < 0.04678345:
        z += 674.0403 * (0.04939969 - Q.z_7) * (0.04678345 - Q.absphi_0)
    if Q.sum_pt < 715.4688 and Q.dr_0 < 0.0361727:
        z += 0.5922737 * (715.4688 - Q.sum_pt) * (0.0361727 - Q.dr_0)
    if Q.centroid_offset < 0.03117077 and Q.z_5 < 0.09549375:
        z += -822.9736 * (0.03117077 - Q.centroid_offset) * (0.09549375 - Q.z_5)
    if Q.pt_5 < 24.57812 and Q.abseta_6 < 0.1583252:
        z += 0.9047365 * (24.57812 - Q.pt_5) * (0.1583252 - Q.abseta_6)
    if Q.sum_pt < 715.4688 and Q.e4 < 1.50326e-07:
        z += -15081.46 * (715.4688 - Q.sum_pt) * (1.50326e-07 - Q.e4)
    if Q.sum_z_dr2 < 0.002635418 and Q.n_dr_0p2_0p4 < 2.0:
        z += 354.6689 * (0.002635418 - Q.sum_z_dr2) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.centroid_offset < 0.03117077 and Q.pt_5 > 29.875:
        z += -0.8623193 * (0.03117077 - Q.centroid_offset) * (Q.pt_5 - 29.875)
    if Q.sum_pt < 715.4688 and Q.zdr_3 < 0.003299436:
        z += 2.651823 * (715.4688 - Q.sum_pt) * (0.003299436 - Q.zdr_3)
    if Q.sum_pt < 788.4484 and Q.n_for_90pct > 5.0:
        z += 0.002220634 * (788.4484 - Q.sum_pt) * (Q.n_for_90pct - 5.0)
    if Q.pt_7 > 43.5 and Q.dr_6 < 0.06970457:
        z += 1.363665 * (Q.pt_7 - 43.5) * (0.06970457 - Q.dr_6)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.633813
    if 0.00809236 <= Q.centroid_offset < 0.01627885:
        z += 67.82684 * Q.centroid_offset - 0.5488792
    if 0.01627885 <= Q.centroid_offset < 0.02076709:
        z += 196.8645 * Q.centroid_offset - 2.649464
    if Q.centroid_offset >= 0.02076709:
        z += 212.8049 * Q.centroid_offset - 2.980499
    if Q.lam1_plus_lam2 < 0.01323868:
        z += 160.074 * Q.lam1_plus_lam2 - 2.119167
    if Q.tau1 < 0.1027642:
        z += -46.48004 * Q.tau1 + 4.834855
    if 0.1027642 <= Q.tau1 < 0.1136369:
        z += -5.368434 * Q.tau1 + 0.6100522
    if Q.log_sum_pt < 6.327379:
        z += -10.88027 * Q.log_sum_pt + 69.23361
    if 6.327379 <= Q.log_sum_pt < 6.638339:
        z += -1.254291 * Q.log_sum_pt + 8.326409
    if Q.tau2 >= 0.01713288:
        z += -9.564074 * Q.tau2 + 0.1638601
    if Q.lam1 < 0.00733008:
        z += 43.98634 * Q.lam1 + 1.581472
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += -368.8215 * Q.lam1 + 4.607387
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -276.3184 * Q.lam1 + 3.83262
    if 0.01200373 <= Q.lam1 < 0.02188405:
        z += -52.20174 * Q.lam1 + 1.142385
    if Q.z_6 < 0.02160287:
        z += 550.4348 * Q.z_6 - 13.06255
    if 0.02160287 <= Q.z_6 < 0.03448406:
        z += 90.95239 * Q.z_6 - 3.136407
    if Q.sj3_dr_min >= 0.02022982:
        z += -12.50768 * Q.sj3_dr_min + 0.2530281
    if Q.lam2 < 0.001130645:
        z += 663.6091 * Q.lam2 - 0.7503061
    if Q.eccentricity >= 0.9031255:
        z += -6.83088 * Q.eccentricity + 6.169142
    if Q.sum_z_dr2 < 0.008678045:
        z += 1435.449 * Q.sum_z_dr2 - 12.45689
    if 0.07264452 <= Q.sj3_dr_max < 0.1789613:
        z += -10.24404 * Q.sj3_dr_max + 0.7441736
    if Q.sj3_dr_max >= 0.1789613:
        z += 15.99918 * Q.sj3_dr_max - 3.952349
    if Q.sum_zz_dr2 < 0.008168571:
        z += -377.8363 * Q.sum_zz_dr2 + 3.086383
    if 0.01165737 <= Q.sum_zz_dr2 < 0.01716248:
        z += -213.1998 * Q.sum_zz_dr2 + 2.48535
    if Q.sum_zz_dr2 >= 0.01716248:
        z += -89.56686 * Q.sum_zz_dr2 + 0.3635016
    if Q.sum_pt >= 988.4078:
        z += 0.01225527 * Q.sum_pt - 12.1132
    if Q.max_dr < 0.1598486:
        z += 19.05139 * Q.max_dr - 3.045339
    if Q.sum_z_dr < 0.02689598:
        z += -206.4759 * Q.sum_z_dr + 5.553372
    if Q.D2 < 1.332146:
        z += -0.4612243 * Q.D2 + 0.6144184
    if Q.pt_6 < 29.90625:
        z += -0.1692712 * Q.pt_6 + 5.062268
    if Q.n_dr_0_0p05 < 3.0:
        z += -0.2777569 * Q.n_dr_0_0p05 + 0.8332707
    if Q.C3 < 0.03853752:
        z += -41.73781 * Q.C3 + 1.608472
    if Q.sum_z_dr2_top2 < 0.0095303:
        z += -111.276 * Q.sum_z_dr2_top2 + 1.060493
    if Q.D3 < 1.651088:
        z += 0.786912 * Q.D3 - 1.299261
    if Q.centroid_offset > 0.00809236 and Q.sj2_dr < 0.1682655:
        z += 712.6216 * (Q.centroid_offset - 0.00809236) * (0.1682655 - Q.sj2_dr)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.planar_flow < 0.3220738:
        z += -286.7385 * (0.01323868 - Q.lam1_plus_lam2) * (0.3220738 - Q.planar_flow)
    if Q.centroid_offset > 0.00809236 and Q.n_dr_0p05_0p1 < 6.0:
        z += 7.225643 * (Q.centroid_offset - 0.00809236) * (6.0 - Q.n_dr_0p05_0p1)
    if Q.log_sum_pt < 6.327379 and Q.pt_7 > 25.57812:
        z += -0.5327615 * (6.327379 - Q.log_sum_pt) * (Q.pt_7 - 25.57812)
    if Q.lam1 < 0.01200373 and Q.pt_6 < 19.46875:
        z += 28.45337 * (0.01200373 - Q.lam1) * (19.46875 - Q.pt_6)
    if Q.z_6 < 0.02160287 and Q.log_sum_pt < 6.842717:
        z += 4640.379 * (0.02160287 - Q.z_6) * (6.842717 - Q.log_sum_pt)
    if Q.log_sum_pt < 6.638339 and Q.z_7 < 0.0586137:
        z += 414.185 * (6.638339 - Q.log_sum_pt) * (0.0586137 - Q.z_7)
    if Q.centroid_offset > 0.00809236 and Q.mean_phi2 < 0.01426135:
        z += 4163.859 * (Q.centroid_offset - 0.00809236) * (0.01426135 - Q.mean_phi2)
    if Q.centroid_offset > 0.01627885 and Q.sum_pt < 988.4078:
        z += -0.7716144 * (Q.centroid_offset - 0.01627885) * (988.4078 - Q.sum_pt)
    if Q.centroid_offset > 0.00809236 and Q.sum_pt < 715.4688:
        z += 0.4221021 * (Q.centroid_offset - 0.00809236) * (715.4688 - Q.sum_pt)
    if Q.sj3_dr_max > 0.1789613 and Q.n_dr_0p05_0p1 < 6.0:
        z += -0.7647634 * (Q.sj3_dr_max - 0.1789613) * (6.0 - Q.n_dr_0p05_0p1)
    if Q.sj3_dr_max > 0.07264452 and Q.M3 < 0.09310137:
        z += -27.89147 * (Q.sj3_dr_max - 0.07264452) * (0.09310137 - Q.M3)
    if Q.eccentricity > 0.9031255 and Q.mean_eta > 0.01772426:
        z += -361.6469 * (Q.eccentricity - 0.9031255) * (Q.mean_eta - 0.01772426)
    if Q.sum_pt > 988.4078 and Q.dr01 > 0.1662967:
        z += -2.184803 * (Q.sum_pt - 988.4078) * (Q.dr01 - 0.1662967)
    if Q.sum_pt > 988.4078 and Q.dr01 > 0.1410336:
        z += 2.009792 * (Q.sum_pt - 988.4078) * (Q.dr01 - 0.1410336)
    if Q.sj3_dr_max > 0.1789613 and Q.eccentricity > 0.9031255:
        z += 291.033 * (Q.sj3_dr_max - 0.1789613) * (Q.eccentricity - 0.9031255)
    if Q.LHA > 0.266913 and Q.eccentricity > 0.9031255:
        z += 237.0744 * (Q.LHA - 0.266913) * (Q.eccentricity - 0.9031255)
    if Q.sj3_dr_max > 0.07264452 and Q.eccentricity > 0.9031255:
        z += -177.2106 * (Q.sj3_dr_max - 0.07264452) * (Q.eccentricity - 0.9031255)
    return max(0.0, z)


def neuron_7(Q):
    z = 8.380787
    if Q.sum_z_dr2_top2 < 0.001056655:
        z += -637.1128 * Q.sum_z_dr2_top2 + 0.6732086
    if Q.sum_z_dr2 < 0.0005611231:
        z += -1716.693 * Q.sum_z_dr2 + 0.6989592
    if 0.0005611231 <= Q.sum_z_dr2 < 0.0009641429:
        z += 655.8418 * Q.sum_z_dr2 - 0.6323252
    if 0.004372139 <= Q.sum_z_dr2 < 0.007520088:
        z += -490.7346 * Q.sum_z_dr2 + 2.14556
    if 0.007520088 <= Q.sum_z_dr2 < 0.008678045:
        z += -1900.212 * Q.sum_z_dr2 + 12.74496
    if Q.sum_z_dr2 >= 0.008678045:
        z += -1409.108 * Q.sum_z_dr2 + 8.48313
    if Q.sum_zz_dr2 < 0.001101266:
        z += 242.3575 * Q.sum_zz_dr2 - 7.842942
    if 0.001101266 <= Q.sum_zz_dr2 < 0.005284669:
        z += 655.5077 * Q.sum_zz_dr2 - 8.29793
    if 0.005284669 <= Q.sum_zz_dr2 < 0.00718279:
        z += 1158.979 * Q.sum_zz_dr2 - 10.95861
    if 0.00718279 <= Q.sum_zz_dr2 < 0.008168571:
        z += 1786.414 * Q.sum_zz_dr2 - 15.46534
    if 0.008168571 <= Q.sum_zz_dr2 < 0.01165737:
        z += 250.1985 * Q.sum_zz_dr2 - 2.916658
    if Q.tau1 < 0.05356915:
        z += -57.21342 * Q.tau1 + 5.136411
    if 0.05356915 <= Q.tau1 < 0.09538712:
        z += -42.83151 * Q.tau1 + 4.365985
    if 0.09538712 <= Q.tau1 < 0.1136369:
        z += 13.77588 * Q.tau1 - 1.033631
    if 0.1136369 <= Q.tau1 < 0.1416226:
        z += -19.00312 * Q.tau1 + 2.691272
    if Q.sum_z_dr < 0.0479157:
        z += 32.95704 * Q.sum_z_dr - 3.813702
    if 0.0479157 <= Q.sum_z_dr < 0.08723651:
        z += 56.82848 * Q.sum_z_dr - 4.957519
    if Q.centroid_offset < 0.02076709:
        z += -49.67413 * Q.centroid_offset + 1.031587
    if Q.lam1_plus_lam2 < 0.005590289:
        z += 904.7216 * Q.lam1_plus_lam2 - 5.057655
    if Q.sj2_dr < 0.1591713:
        z += -7.473234 * Q.sj2_dr + 0.6413382
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 19.51506 * Q.sj2_dr - 3.654423
    if Q.LHA < 0.2160559:
        z += 20.97576 * Q.LHA - 3.764829
    if 0.2160559 <= Q.LHA < 0.2931906:
        z += -9.945028 * Q.LHA + 2.915789
    if Q.lam1 < 0.004183811:
        z += -111.2892 * Q.lam1 + 0.465613
    if Q.lam2 < 0.000537286:
        z += 916.2033 * Q.lam2 - 0.2500382
    if 0.000537286 <= Q.lam2 < 0.001130645:
        z += -408.227 * Q.lam2 + 0.4615597
    if Q.max_dr < 0.121681:
        z += -3.384502 * Q.max_dr + 1.125715
    if 0.121681 <= Q.max_dr < 0.1979689:
        z += -9.357782 * Q.max_dr + 1.85255
    if Q.sj3_dr_max < 0.1426152:
        z += -10.8661 * Q.sj3_dr_max + 1.712326
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += -4.062953 * Q.sj3_dr_max + 0.7420946
    if 0.1986272 <= Q.sj3_dr_max < 0.213399:
        z += 4.394756 * Q.sj3_dr_max - 0.9378364
    if Q.planar_flow < 0.1950135 and Q.lam1_plus_lam2 > 0.007520088:
        z += 2485.239 * (0.1950135 - Q.planar_flow) * (Q.lam1_plus_lam2 - 0.007520088)
    if Q.planar_flow < 0.1950135 and Q.pt_6 < 33.6875:
        z += -0.3422741 * (0.1950135 - Q.planar_flow) * (33.6875 - Q.pt_6)
    if Q.planar_flow < 0.1950135 and Q.log_sum_pt > 6.423044:
        z += 34.64395 * (0.1950135 - Q.planar_flow) * (Q.log_sum_pt - 6.423044)
    if Q.planar_flow < 0.1950135 and Q.z_7 < 0.05240025:
        z += -155.2063 * (0.1950135 - Q.planar_flow) * (0.05240025 - Q.z_7)
    if Q.lam1_plus_lam2 < 0.005590289 and Q.mean_phi < -0.00469376:
        z += 17883.33 * (0.005590289 - Q.lam1_plus_lam2) * (-0.00469376 - Q.mean_phi)
    if Q.sum_z_dr2 > 0.004372139 and Q.sj3_z1 > 0.8740683:
        z += -10752.13 * (Q.sum_z_dr2 - 0.004372139) * (Q.sj3_z1 - 0.8740683)
    if Q.sum_z_dr2 > 0.008678045 and Q.sj3_z1 > 0.8740683:
        z += 35295.86 * (Q.sum_z_dr2 - 0.008678045) * (Q.sj3_z1 - 0.8740683)
    if Q.sum_zz_dr2 < 0.01165737 and Q.pt_7 < 48.71875:
        z += -1.51928 * (0.01165737 - Q.sum_zz_dr2) * (48.71875 - Q.pt_7)
    if Q.lam1_plus_lam2 < 0.005590289 and Q.mean_eta < -0.006779839:
        z += 19601.86 * (0.005590289 - Q.lam1_plus_lam2) * (-0.006779839 - Q.mean_eta)
    if Q.lam1_plus_lam2 < 0.005590289 and Q.mean_eta > 0.009367547:
        z += 19553.67 * (0.005590289 - Q.lam1_plus_lam2) * (Q.mean_eta - 0.009367547)
    if Q.lam1_plus_lam2 < 0.005590289 and Q.mean_phi > 0.009050008:
        z += 20749.19 * (0.005590289 - Q.lam1_plus_lam2) * (Q.mean_phi - 0.009050008)
    if Q.sum_z_dr2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 1403.981 * (Q.sum_z_dr2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    if Q.centroid_offset < 0.02076709 and Q.D2_b2 < 4.721224:
        z += -18.8138 * (0.02076709 - Q.centroid_offset) * (4.721224 - Q.D2_b2)
    if Q.sum_z_dr2 > 0.01323868 and Q.eccentricity > 0.9458207:
        z += 24423.05 * (Q.sum_z_dr2 - 0.01323868) * (Q.eccentricity - 0.9458207)
    if Q.sum_z_dr2 > 0.008678045 and Q.eccentricity > 0.9458207:
        z += -12288.73 * (Q.sum_z_dr2 - 0.008678045) * (Q.eccentricity - 0.9458207)
    if Q.max_dr < 0.1979689 and Q.n_dr_0p1_0p2 > 4.0:
        z += -6.978719 * (0.1979689 - Q.max_dr) * (Q.n_dr_0p1_0p2 - 4.0)
    if Q.sum_zz_dr2 < 0.00718279 and Q.sj3_dr13 > 0.1982159:
        z += 2777.688 * (0.00718279 - Q.sum_zz_dr2) * (Q.sj3_dr13 - 0.1982159)
    if Q.sj3_dr_max < 0.213399 and Q.eccentricity > 0.9458207:
        z += -165.9268 * (0.213399 - Q.sj3_dr_max) * (Q.eccentricity - 0.9458207)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.5469599
    if Q.sum_z_dr2 < 0.0003193707:
        z += -4391.294 * Q.sum_z_dr2 + 4.893086
    if 0.0003193707 <= Q.sum_z_dr2 < 0.005019719:
        z += -632.0616 * Q.sum_z_dr2 + 3.692497
    if 0.005019719 <= Q.sum_z_dr2 < 0.006679471:
        z += -313.1345 * Q.sum_z_dr2 + 2.091573
    if Q.tau1 < 0.01357518:
        z += 30.45416 * Q.tau1 + 0.2161111
    if 0.01357518 <= Q.tau1 < 0.05356915:
        z += -15.74067 * Q.tau1 + 0.8432142
    if Q.LHA < 0.1967397:
        z += -15.63493 * Q.LHA + 3.076012
    if Q.log_sum_pt >= 6.701242:
        z += -3.326461 * Q.log_sum_pt + 22.29142
    if Q.sj3_dr_max < 0.1070199:
        z += 0.8918839 * Q.sj3_dr_max - 0.7193177
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 17.52674 * Q.sj3_dr_max - 2.499579
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.005989197 * Q.sum_pt_top5 - 4.117198
    if Q.sum_z_dr < 0.05464922:
        z += 53.47845 * Q.sum_z_dr - 2.922556
    if Q.lam1 < 0.001503553:
        z += -345.5057 * Q.lam1 + 0.5194863
    if Q.n_dr_0_0p05 >= 5.0:
        z += 0.1746268 * Q.n_dr_0_0p05 - 0.8731338
    if Q.tau2 < 0.01713288:
        z += 42.72533 * Q.tau2 - 0.7320079
    if Q.sum_z_dr2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 13524.89 * (0.006679471 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.sum_z_dr2 < 0.005019719 and Q.log_sum_pt < 6.502799:
        z += -993.9733 * (0.005019719 - Q.sum_z_dr2) * (6.502799 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.006679471 and Q.planar_flow < 0.4007947:
        z += 396.3815 * (0.006679471 - Q.sum_z_dr2) * (0.4007947 - Q.planar_flow)
    if Q.sum_z_dr2 < 0.005019719 and Q.pt_7 < 43.5:
        z += -10.99583 * (0.005019719 - Q.sum_z_dr2) * (43.5 - Q.pt_7)
    if Q.sum_z_dr2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -39738.48 * (0.005019719 - Q.sum_z_dr2) * (Q.centroid_offset - 0.006789738)
    if Q.LHA < 0.1967397 and Q.z_dr_0p2_0p4 < 0.20552:
        z += -158.5756 * (0.1967397 - Q.LHA) * (0.20552 - Q.z_dr_0p2_0p4)
    if Q.sum_z_dr2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += 4620.287 * (0.005019719 - Q.sum_z_dr2) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt > 6.701242 and Q.psi_0p2 > 0.9435576:
        z += 62.76703 * (Q.log_sum_pt - 6.701242) * (Q.psi_0p2 - 0.9435576)
    if Q.tau1 < 0.05356915 and Q.sum_z_dr2 < 0.002635418:
        z += 30308.73 * (0.05356915 - Q.tau1) * (0.002635418 - Q.sum_z_dr2)
    if Q.sum_pt_top5 > 687.4375 and Q.sum_z_dr2_top3 < 0.01002369:
        z += -0.3430616 * (Q.sum_pt_top5 - 687.4375) * (0.01002369 - Q.sum_z_dr2_top3)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2_top3 < 0.002151568:
        z += 3760.222 * (Q.log_sum_pt - 6.701242) * (0.002151568 - Q.sum_z_dr2_top3)
    if Q.sum_z_dr < 0.05464922 and Q.lam2 < 0.0001947983:
        z += 210819.0 * (0.05464922 - Q.sum_z_dr) * (0.0001947983 - Q.lam2)
    if Q.tau1 < 0.05356915 and Q.lam2 < 0.0003061234:
        z += -45641.91 * (0.05356915 - Q.tau1) * (0.0003061234 - Q.lam2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2_top3 > 0.001155057:
        z += 1382.542 * (0.1986272 - Q.sj3_dr_max) * (Q.sum_z_dr2_top3 - 0.001155057)
    if Q.LHA < 0.1967397 and Q.lam2 < 0.0001947983:
        z += -70684.24 * (0.1967397 - Q.LHA) * (0.0001947983 - Q.lam2)
    if Q.tau1 < 0.05356915 and Q.pt_7 < 43.5:
        z += 0.8486521 * (0.05356915 - Q.tau1) * (43.5 - Q.pt_7)
    if Q.LHA < 0.1967397 and Q.pt_7 > 15.55391:
        z += 0.2357645 * (0.1967397 - Q.LHA) * (Q.pt_7 - 15.55391)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 > 15.55391:
        z += -0.1980941 * (Q.log_sum_pt - 6.701242) * (Q.pt_7 - 15.55391)
    if Q.z_dr_0_0p05 > 0.9008535 and Q.lam2 < 0.0001947983:
        z += -41064.81 * (Q.z_dr_0_0p05 - 0.9008535) * (0.0001947983 - Q.lam2)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi2 < 0.006445344:
        z += -611.7756 * (Q.log_sum_pt - 6.701242) * (0.006445344 - Q.mean_phi2)
    if Q.log_sum_pt > 6.701242 and Q.lam1_plus_lam2 < 0.006679471:
        z += -1043.473 * (Q.log_sum_pt - 6.701242) * (0.006679471 - Q.lam1_plus_lam2)
    if Q.log_sum_pt > 6.701242 and Q.lam1_plus_lam2 < 0.002635418:
        z += 2507.101 * (Q.log_sum_pt - 6.701242) * (0.002635418 - Q.lam1_plus_lam2)
    if Q.sum_pt_top5 > 687.4375 and Q.sum_z_dr2 < 0.0009641429:
        z += -4.950368 * (Q.sum_pt_top5 - 687.4375) * (0.0009641429 - Q.sum_z_dr2)
    if Q.sum_z_dr < 0.05464922 and Q.lam1_plus_lam2 < 0.002635418:
        z += -22646.07 * (0.05464922 - Q.sum_z_dr) * (0.002635418 - Q.lam1_plus_lam2)
    if Q.sum_z_dr2 < 0.006679471 and Q.lam1_plus_lam2 > 0.001653836:
        z += -77565.53 * (0.006679471 - Q.sum_z_dr2) * (Q.lam1_plus_lam2 - 0.001653836)
    if Q.sum_z_dr < 0.05464922 and Q.lam1_plus_lam2 < 0.005590289:
        z += -12568.27 * (0.05464922 - Q.sum_z_dr) * (0.005590289 - Q.lam1_plus_lam2)
    if Q.sj3_dr_max < 0.1986272 and Q.lam2 < 0.0003061234:
        z += 43503.88 * (0.1986272 - Q.sj3_dr_max) * (0.0003061234 - Q.lam2)
    if Q.sj3_dr_max < 0.1070199 and Q.lam2 < 0.0003061234:
        z += -75945.75 * (0.1070199 - Q.sj3_dr_max) * (0.0003061234 - Q.lam2)
    if Q.LHA < 0.1967397 and Q.centroid_offset > 0.01627885:
        z += -10048.96 * (0.1967397 - Q.LHA) * (Q.centroid_offset - 0.01627885)
    return max(0.0, z)


def neuron_9(Q):
    z = -2.865776
    if Q.sum_z_dr < 0.05464922:
        z += 72.51291 * Q.sum_z_dr - 4.27724
    if 0.05464922 <= Q.sum_z_dr < 0.06663269:
        z += 26.24161 * Q.sum_z_dr - 1.748549
    if Q.tau1 < 0.04369778:
        z += -22.3014 * Q.tau1 + 0.9745215
    if Q.sum_z_dr2 < 0.003562611:
        z += -2939.015 * Q.sum_z_dr2 + 12.67215
    if 0.003562611 <= Q.sum_z_dr2 < 0.005019719:
        z += -1510.928 * Q.sum_z_dr2 + 7.584434
    if Q.LHA < 0.2160559:
        z += 8.262134 * Q.LHA - 1.785082
    if 3.892127e-05 <= Q.e3 < 0.0001869378:
        z += -8424.725 * Q.e3 + 0.327901
    if Q.e3 >= 0.0001869378:
        z += 3696.148 * Q.e3 - 1.937949
    if Q.log_sum_pt < 6.896095:
        z += -6.490664 * Q.log_sum_pt + 44.76024
    if Q.lam1_plus_lam2 < 0.006679471:
        z += -1066.58 * Q.lam1_plus_lam2 + 7.124188
    if Q.lam2 < 7.300726e-05:
        z += -11977.2 * Q.lam2 + 1.410326
    if 7.300726e-05 <= Q.lam2 < 0.0001947983:
        z += -4400.19 * Q.lam2 + 0.8571495
    if Q.lam2 >= 0.001130645:
        z += 894.6281 * Q.lam2 - 1.011507
    if Q.sj2_dr < 0.1294903:
        z += 20.31951 * Q.sj2_dr - 2.390966
    if 0.1294903 <= Q.sj2_dr < 0.1778793:
        z += -4.964238 * Q.sj2_dr + 0.8830351
    if Q.centroid_offset < 0.02355416:
        z += -83.94533 * Q.centroid_offset + 1.977262
    if Q.sj3_dr_max < 0.1426152:
        z += 10.09563 * Q.sj3_dr_max - 0.3501418
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += -19.45381 * Q.sj3_dr_max + 3.864057
    if Q.sj3_dr_max >= 0.2623172:
        z += -5.804361 * Q.sj3_dr_max + 1.522584
    if Q.sum_zz_dr2 < 0.001101266:
        z += 2359.804 * Q.sum_zz_dr2 - 8.633758
    if 0.001101266 <= Q.sum_zz_dr2 < 0.0030133:
        z += 1648.884 * Q.sum_zz_dr2 - 7.850845
    if 0.0030133 <= Q.sum_zz_dr2 < 0.004641801:
        z += 738.9198 * Q.sum_zz_dr2 - 5.108849
    if 0.004641801 <= Q.sum_zz_dr2 < 0.02378091:
        z += 87.7225 * Q.sum_zz_dr2 - 2.086121
    if Q.max_dr < 0.1117619:
        z += -16.8014 * Q.max_dr + 1.877756
    if Q.ptdr0_2 >= 21.99783:
        z += -0.1192022 * Q.ptdr0_2 + 2.62219
    if Q.lam1 < 0.00483998:
        z += 150.1445 * Q.lam1 - 0.7266966
    if Q.dr_0 < 0.06413297:
        z += 17.66889 * Q.dr_0 - 1.133158
    if Q.centroid_offset < 0.01837778 and Q.M2 < 0.04435703:
        z += -2481.974 * (0.01837778 - Q.centroid_offset) * (0.04435703 - Q.M2)
    if Q.centroid_offset < 0.01837778 and Q.tau3 < 0.01364517:
        z += 10036.53 * (0.01837778 - Q.centroid_offset) * (0.01364517 - Q.tau3)
    if Q.sum_z_dr < 0.06663269 and Q.mean_phi < 0.00161889:
        z += -1609.445 * (0.06663269 - Q.sum_z_dr) * (0.00161889 - Q.mean_phi)
    if Q.centroid_offset < 0.02355416 and Q.N3 > 0.7686247:
        z += 20.15284 * (0.02355416 - Q.centroid_offset) * (Q.N3 - 0.7686247)
    if Q.lam2 > 0.001130645 and Q.planar_flow > 0.2534037:
        z += -749.0006 * (Q.lam2 - 0.001130645) * (Q.planar_flow - 0.2534037)
    if Q.log_sum_pt < 6.896095 and Q.z_5 < 0.05753583:
        z += 178.3428 * (6.896095 - Q.log_sum_pt) * (0.05753583 - Q.z_5)
    if Q.centroid_offset < 0.01837778 and Q.n_dr_0p05_0p1 < 5.0:
        z += 14.55571 * (0.01837778 - Q.centroid_offset) * (5.0 - Q.n_dr_0p05_0p1)
    if Q.centroid_offset < 0.02355416 and Q.tau21_b2 > 0.004811143:
        z += 68.75381 * (0.02355416 - Q.centroid_offset) * (Q.tau21_b2 - 0.004811143)
    if Q.centroid_offset < 0.02355416 and Q.D2_b2 < 0.380911:
        z += -94.62519 * (0.02355416 - Q.centroid_offset) * (0.380911 - Q.D2_b2)
    if Q.log_sum_pt < 6.896095 and Q.planar_flow > 0.1115136:
        z += -2.396728 * (6.896095 - Q.log_sum_pt) * (Q.planar_flow - 0.1115136)
    if Q.sum_z_dr < 0.06663269 and Q.mean_eta > 0.01271871:
        z += 2541.496 * (0.06663269 - Q.sum_z_dr) * (Q.mean_eta - 0.01271871)
    if Q.lam1_plus_lam2 < 0.006679471 and Q.mean_eta < -0.006779839:
        z += -17411.72 * (0.006679471 - Q.lam1_plus_lam2) * (-0.006779839 - Q.mean_eta)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 > 93.25:
        z += 0.5480812 * (0.01837778 - Q.centroid_offset) * (Q.pt_1 - 93.25)
    if Q.sj2_dr < 0.1294903 and Q.mean_eta < -0.006779839:
        z += 914.9979 * (0.1294903 - Q.sj2_dr) * (-0.006779839 - Q.mean_eta)
    if Q.sj2_dr < 0.1294903 and Q.mean_eta > 0.01271871:
        z += 965.9287 * (0.1294903 - Q.sj2_dr) * (Q.mean_eta - 0.01271871)
    if Q.lam1_plus_lam2 < 0.006679471 and Q.mean_eta > 0.009367547:
        z += -17610.97 * (0.006679471 - Q.lam1_plus_lam2) * (Q.mean_eta - 0.009367547)
    if Q.sum_z_dr2 < 0.003562611 and Q.mean_eta < -0.01284493:
        z += -39102.54 * (0.003562611 - Q.sum_z_dr2) * (-0.01284493 - Q.mean_eta)
    if Q.LHA < 0.2160559 and Q.mean_eta < 0.003218391:
        z += -554.2344 * (0.2160559 - Q.LHA) * (0.003218391 - Q.mean_eta)
    if Q.centroid_offset < 0.01837778 and Q.pt1_over_pt0 > 0.2131291:
        z += -83.51498 * (0.01837778 - Q.centroid_offset) * (Q.pt1_over_pt0 - 0.2131291)
    if Q.sum_z_dr2 < 0.003562611 and Q.mean_eta > 0.01271871:
        z += -89863.06 * (0.003562611 - Q.sum_z_dr2) * (Q.mean_eta - 0.01271871)
    if Q.lam1_plus_lam2 < 0.006679471 and Q.D3 < 0.8272948:
        z += -758.9684 * (0.006679471 - Q.lam1_plus_lam2) * (0.8272948 - Q.D3)
    if Q.sum_z_dr2 < 0.003562611 and Q.D3 < 0.8272948:
        z += 1365.321 * (0.003562611 - Q.sum_z_dr2) * (0.8272948 - Q.D3)
    if Q.centroid_offset < 0.01837778 and Q.mratio_min_012 < 0.07735553:
        z += -832.3661 * (0.01837778 - Q.centroid_offset) * (0.07735553 - Q.mratio_min_012)
    if Q.sum_zz_dr2 < 0.001101266 and Q.eccentricity > 0.7117266:
        z += -2173.11 * (0.001101266 - Q.sum_zz_dr2) * (Q.eccentricity - 0.7117266)
    if Q.lam1 < 0.00483998 and Q.dr1_7 < 0.1792439:
        z += -1443.122 * (0.00483998 - Q.lam1) * (0.1792439 - Q.dr1_7)
    if Q.log_sum_pt < 6.896095 and Q.D2_b2 < 0.9206502:
        z += -2.047601 * (6.896095 - Q.log_sum_pt) * (0.9206502 - Q.D2_b2)
    if Q.centroid_offset < 0.02355416 and Q.dr_7 < 0.0825754:
        z += 645.678 * (0.02355416 - Q.centroid_offset) * (0.0825754 - Q.dr_7)
    if Q.sum_z_dr < 0.06663269 and Q.mean_phi > 0.004406178:
        z += -2298.4 * (0.06663269 - Q.sum_z_dr) * (Q.mean_phi - 0.004406178)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.739968
    if Q.lam1 < 0.003377388:
        z += 973.7634 * Q.lam1 - 2.174836
    if 0.003377388 <= Q.lam1 < 0.00483998:
        z += 740.872 * Q.lam1 - 1.388271
    if 0.00483998 <= Q.lam1 < 0.006506576:
        z += 488.054 * Q.lam1 - 0.164637
    if Q.lam1 >= 0.006506576:
        z += 274.6897 * Q.lam1 + 1.223634
    if 0.0001947983 <= Q.lam2 < 0.003408389:
        z += 1170.903 * Q.lam2 - 0.2280899
    if Q.lam2 >= 0.003408389:
        z += 953.4657 * Q.lam2 + 0.5130212
    if Q.e3 >= 8.147744e-05:
        z += 2856.615 * Q.e3 - 0.2327497
    if 0.1967397 <= Q.LHA < 0.3127275:
        z += -17.59385 * Q.LHA + 3.46141
    if Q.LHA >= 0.3127275:
        z += -29.39852 * Q.LHA + 7.153054
    if Q.centroid_offset >= 0.02355416:
        z += 15.19859 * Q.centroid_offset - 0.35799
    if Q.n_dr_0p05_0p1 < 5.0:
        z += -0.07614098 * Q.n_dr_0p05_0p1 + 0.3807049
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.8450139 * Q.n_dr_0p2_0p4 - 0.8450139
    if Q.z_7 < 0.06810151:
        z += 16.12794 * Q.z_7 - 1.098337
    if Q.tau1 >= 0.04369778:
        z += 16.39942 * Q.tau1 - 0.7166184
    if 0.1294903 <= Q.sj2_dr < 0.1682655:
        z += 8.114553 * Q.sj2_dr - 1.050756
    if 0.1682655 <= Q.sj2_dr < 0.3003793:
        z += 2.115828 * Q.sj2_dr - 0.04137713
    if Q.sj2_dr >= 0.3003793:
        z += 14.42598 * Q.sj2_dr - 3.739093
    if Q.sum_z_dr2_top3 < 0.002915531:
        z += -279.5357 * Q.sum_z_dr2_top3 + 0.814995
    if Q.sum_pt >= 988.4078:
        z += 0.008250392 * Q.sum_pt - 8.154752
    if 0.002635418 <= Q.sum_z_dr2 < 0.007520088:
        z += -307.7115 * Q.sum_z_dr2 + 0.8109484
    if 0.007520088 <= Q.sum_z_dr2 < 0.02530566:
        z += -72.63142 * Q.sum_z_dr2 - 0.9568747
    if Q.sum_z_dr2 >= 0.02530566:
        z += -173.5243 * Q.sum_z_dr2 + 1.596287
    if Q.sj3_dr_max < 0.3012016:
        z += -3.131464 * Q.sj3_dr_max + 0.9432021
    if Q.log_sum_pt >= 6.670067:
        z += -2.935454 * Q.log_sum_pt + 19.57967
    if Q.sum_z_dr < 0.1245537:
        z += 20.79757 * Q.sum_z_dr - 2.590415
    if Q.sum_z_dr2_top5 < 0.001501708:
        z += -207.5166 * Q.sum_z_dr2_top5 + 0.3116295
    if Q.lam2 > 0.0001947983 and Q.log_sum_pt < 6.538559:
        z += -507.6316 * (Q.lam2 - 0.0001947983) * (6.538559 - Q.log_sum_pt)
    if Q.lam2 > 0.0001947983 and Q.eccentricity > 0.4829602:
        z += -819.0071 * (Q.lam2 - 0.0001947983) * (Q.eccentricity - 0.4829602)
    if Q.pt_6 < 48.03125 and Q.log_sum_pt < 6.46415:
        z += -0.09240782 * (48.03125 - Q.pt_6) * (6.46415 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.D2_b2 < 4.721224:
        z += -0.194478 * (Q.n_dr_0p2_0p4 - 1.0) * (4.721224 - Q.D2_b2)
    if Q.pt_6 < 48.03125 and Q.D2_b2 < 0.1236856:
        z += 0.3032796 * (48.03125 - Q.pt_6) * (0.1236856 - Q.D2_b2)
    if Q.lam2 > 0.0001947983 and Q.planar_flow > 0.06126205:
        z += -445.0649 * (Q.lam2 - 0.0001947983) * (Q.planar_flow - 0.06126205)
    if Q.sj2_dr > 0.1682655 and Q.planar_flow < 0.6947818:
        z += -21.48285 * (Q.sj2_dr - 0.1682655) * (0.6947818 - Q.planar_flow)
    if Q.LHA > 0.3127275 and Q.planar_flow < 0.6947818:
        z += -18.44069 * (Q.LHA - 0.3127275) * (0.6947818 - Q.planar_flow)
    if Q.lam1 < 0.006506576 and Q.absphi_7 < 0.1647949:
        z += -611.6588 * (0.006506576 - Q.lam1) * (0.1647949 - Q.absphi_7)
    return max(0.0, z)


def neuron_11(Q):
    z = -2.066514
    if Q.planar_flow < 0.2534037:
        z += -4.723638 * Q.planar_flow + 1.196987
    if 0.09419022 <= Q.sj2_dr < 0.1778793:
        z += 7.209725 * Q.sj2_dr - 0.6790856
    if 0.1778793 <= Q.sj2_dr < 0.1872617:
        z += -22.84533 * Q.sj2_dr + 4.667086
    if 0.1872617 <= Q.sj2_dr < 0.2687922:
        z += -13.03172 * Q.sj2_dr + 2.829374
    if Q.sj2_dr >= 0.2687922:
        z += 3.203352 * Q.sj2_dr - 1.534487
    if Q.lam1_plus_lam2 < 0.005019719:
        z += -1564.998 * Q.lam1_plus_lam2 + 16.44858
    if 0.005019719 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -1779.852 * Q.lam1_plus_lam2 + 17.52709
    if 0.008678045 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -456.3948 * Q.lam1_plus_lam2 + 6.042063
    if Q.LHA < 0.2809341:
        z += -4.284441 * Q.LHA + 0.7831534
    if 0.2809341 <= Q.LHA < 0.3033137:
        z += 2.841805 * Q.LHA - 1.218852
    if 0.3033137 <= Q.LHA < 0.3127275:
        z += 37.91206 * Q.LHA - 11.85614
    if Q.centroid_offset < 0.03776099:
        z += -63.29747 * Q.centroid_offset + 2.390175
    if Q.tau1 < 0.03459477:
        z += 59.31046 * Q.tau1 - 2.051832
    if 6.572938 <= Q.log_sum_pt < 6.733425:
        z += 3.768379 * Q.log_sum_pt - 24.76932
    if Q.log_sum_pt >= 6.733425:
        z += 0.1897824 * Q.log_sum_pt - 0.6731088
    if Q.lam1 < 0.00733008:
        z += 697.4553 * Q.lam1 - 5.660749
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 524.4862 * Q.lam1 - 4.392872
    if Q.sum_zz_dr2 < 0.006390125:
        z += 1035.468 * Q.sum_zz_dr2 - 7.25066
    if 0.006390125 <= Q.sum_zz_dr2 < 0.008168571:
        z += 356.4304 * Q.sum_zz_dr2 - 2.911527
    if Q.mean_phi2 < 0.001570267:
        z += 307.2971 * Q.mean_phi2 - 0.4825386
    if Q.sum_z_dr2 < 0.003562611:
        z += -409.6944 * Q.sum_z_dr2 + 3.204268
    if 0.003562611 <= Q.sum_z_dr2 < 0.006679471:
        z += -559.7576 * Q.sum_z_dr2 + 3.738885
    if Q.sum_z_dr < 0.03360421:
        z += 152.3107 * Q.sum_z_dr - 9.040737
    if 0.03360421 <= Q.sum_z_dr < 0.0717028:
        z += 102.9554 * Q.sum_z_dr - 7.382192
    if Q.pt_7 >= 29.04219:
        z += -0.02261275 * Q.pt_7 + 0.6567238
    if Q.lam2 < 0.000537286:
        z += -1041.738 * Q.lam2 + 0.5597113
    if Q.z_7 >= 0.01685855:
        z += 23.96732 * Q.z_7 - 0.4040542
    if Q.e2 < 0.02045966:
        z += -78.20469 * Q.e2 + 1.600041
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.01546081 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.pt_7 < 37.15625:
        z += -0.2140606 * (0.2534037 - Q.planar_flow) * (37.15625 - Q.pt_7)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.log_sum_pt < 6.502799:
        z += -288.2811 * (0.01323868 - Q.lam1_plus_lam2) * (6.502799 - Q.log_sum_pt)
    if Q.planar_flow < 0.2534037 and Q.e3 < 1.050302e-05:
        z += -238277.7 * (0.2534037 - Q.planar_flow) * (1.050302e-05 - Q.e3)
    if Q.sj2_dr > 0.1778793 and Q.lam2 < 0.001130645:
        z += -3366.595 * (Q.sj2_dr - 0.1778793) * (0.001130645 - Q.lam2)
    if Q.centroid_offset < 0.03776099 and Q.dr_max_012 > 0.1828389:
        z += 189.3073 * (0.03776099 - Q.centroid_offset) * (Q.dr_max_012 - 0.1828389)
    if Q.centroid_offset < 0.01437952 and Q.D2 < 1.332146:
        z += -117.7937 * (0.01437952 - Q.centroid_offset) * (1.332146 - Q.D2)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += -96.41711 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.centroid_offset < 0.01437952 and Q.tau21_b2 < 0.05932655:
        z += 1985.327 * (0.01437952 - Q.centroid_offset) * (0.05932655 - Q.tau21_b2)
    if Q.mean_phi2 < 0.001570267 and Q.M2 > 0.01411669:
        z += 2794.612 * (0.001570267 - Q.mean_phi2) * (Q.M2 - 0.01411669)
    if Q.centroid_offset > 0.04990367 and Q.M3 > 0.03843235:
        z += -14440.34 * (Q.centroid_offset - 0.04990367) * (Q.M3 - 0.03843235)
    if Q.centroid_offset < 0.01437952 and Q.sj3_dr_min < 0.08543881:
        z += -1375.474 * (0.01437952 - Q.centroid_offset) * (0.08543881 - Q.sj3_dr_min)
    if Q.tau1 < 0.03459477 and Q.sum_z_dr2_top3 < 0.002915531:
        z += 19203.08 * (0.03459477 - Q.tau1) * (0.002915531 - Q.sum_z_dr2_top3)
    if Q.sum_zz_dr2 < 0.0030133 and Q.sum_z_dr2_top3 > 0.003952582:
        z += -433222.8 * (0.0030133 - Q.sum_zz_dr2) * (Q.sum_z_dr2_top3 - 0.003952582)
    if Q.centroid_offset < 0.03776099 and Q.pt_7 < 53.4375:
        z += -1.034939 * (0.03776099 - Q.centroid_offset) * (53.4375 - Q.pt_7)
    if Q.log_sum_pt > 6.572938 and Q.D3 < 3.916009:
        z += -0.3719119 * (Q.log_sum_pt - 6.572938) * (3.916009 - Q.D3)
    if Q.tau1 < 0.09538712 and Q.pt_6 < 62.25:
        z += 0.1054701 * (0.09538712 - Q.tau1) * (62.25 - Q.pt_6)
    if Q.centroid_offset < 0.03776099 and Q.pt_2 < 118.5:
        z += -0.8935248 * (0.03776099 - Q.centroid_offset) * (118.5 - Q.pt_2)
    if Q.sum_zz_dr2 < 0.006390125 and Q.pt_2 < 118.5:
        z += 4.577904 * (0.006390125 - Q.sum_zz_dr2) * (118.5 - Q.pt_2)
    if Q.sum_zz_dr2 < 0.006390125 and Q.z_3rd < 0.1471389:
        z += -3490.237 * (0.006390125 - Q.sum_zz_dr2) * (0.1471389 - Q.z_3rd)
    if Q.centroid_offset < 0.03776099 and Q.z_3rd < 0.1561228:
        z += 610.5321 * (0.03776099 - Q.centroid_offset) * (0.1561228 - Q.z_3rd)
    if Q.log_sum_pt > 6.572938 and Q.z_7 < 0.0586137:
        z += 82.44497 * (Q.log_sum_pt - 6.572938) * (0.0586137 - Q.z_7)
    if Q.centroid_offset < 0.01437952 and Q.tau3 < 0.01364517:
        z += -3233.061 * (0.01437952 - Q.centroid_offset) * (0.01364517 - Q.tau3)
    if Q.centroid_offset < 0.01437952 and Q.tau21_b2 < 0.2691019:
        z += 473.981 * (0.01437952 - Q.centroid_offset) * (0.2691019 - Q.tau21_b2)
    if Q.lam1_plus_lam2 < 0.005019719 and Q.sj3_dr23 > 0.1414609:
        z += 2133.572 * (0.005019719 - Q.lam1_plus_lam2) * (Q.sj3_dr23 - 0.1414609)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.5183784
    if 0.01882765 <= Q.sum_z_dr2 < 0.02530566:
        z += 289.5015 * Q.sum_z_dr2 - 5.450634
    if Q.sum_z_dr2 >= 0.02530566:
        z += 446.9489 * Q.sum_z_dr2 - 9.434943
    if Q.e2 >= 0.06344108:
        z += -83.57207 * Q.e2 + 5.301902
    if Q.ptdr0_4 >= 15.26525:
        z += 0.1403386 * Q.ptdr0_4 - 2.142303
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += 91.37525 * Q.lam1_plus_lam2 - 1.209687
    if Q.sum_z_dr2 > 0.01882765 and Q.pt_7 < 53.4375:
        z += -3.46261 * (Q.sum_z_dr2 - 0.01882765) * (53.4375 - Q.pt_7)
    if Q.ptdr0_4 > 15.26525 and Q.sum_pt < 988.4078:
        z += -0.0007580732 * (Q.ptdr0_4 - 15.26525) * (988.4078 - Q.sum_pt)
    if Q.sum_z_dr2 > 0.01882765 and Q.pt_2 < 96.6875:
        z += -1.996819 * (Q.sum_z_dr2 - 0.01882765) * (96.6875 - Q.pt_2)
    if Q.ptdr0_4 > 15.26525 and Q.sum_pt < 763.825:
        z += 0.000889076 * (Q.ptdr0_4 - 15.26525) * (763.825 - Q.sum_pt)
    if Q.lam1_plus_lam2 > 0.01323868 and Q.log_sum_pt > 6.327379:
        z += 272.3864 * (Q.lam1_plus_lam2 - 0.01323868) * (Q.log_sum_pt - 6.327379)
    if Q.sum_z_dr2 > 0.01882765 and Q.planar_flow < 0.7974684:
        z += -124.3112 * (Q.sum_z_dr2 - 0.01882765) * (0.7974684 - Q.planar_flow)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.349293
    if Q.sum_z_dr < 0.1484084:
        z += -71.29043 * Q.sum_z_dr + 10.5801
    if Q.lam1 < 0.01643375:
        z += -31.82002 * Q.lam1 + 0.5229223
    if Q.pt_6 < 27.57812:
        z += 0.1121772 * Q.pt_6 - 3.605315
    if 27.57812 <= Q.pt_6 < 50.25:
        z += 0.02256884 * Q.pt_6 - 1.134084
    if Q.sj3_dr_max >= 0.233678:
        z += -7.760967 * Q.sj3_dr_max + 1.813567
    if Q.pt_5 < 24.57812:
        z += 0.1784474 * Q.pt_5 - 4.385902
    if Q.e3 < 8.147744e-05:
        z += -849.7549 * Q.e3 - 0.6445998
    if 8.147744e-05 <= Q.e3 < 0.0001869378:
        z += 6768.756 * Q.e3 - 1.265337
    if Q.e2 < 0.08000524:
        z += 18.48277 * Q.e2 - 1.478718
    if Q.log_sum_pt < 6.46415:
        z += 2.674333 * Q.log_sum_pt - 17.28729
    if Q.M2 < 0.0294431:
        z += -33.0813 * Q.M2 + 0.9740161
    if Q.pt_7 >= 31.85938:
        z += -0.07042738 * Q.pt_7 + 2.243772
    if Q.tau1 < 0.1136369:
        z += 22.89475 * Q.tau1 - 2.601689
    if Q.lam1_plus_lam2 < 0.008678045:
        z += -170.19 * Q.lam1_plus_lam2 + 2.473706
    if 0.008678045 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -218.5639 * Q.lam1_plus_lam2 + 2.893497
    if Q.sum_pt_top5 >= 482.2812:
        z += -0.01206602 * Q.sum_pt_top5 + 5.819215
    if Q.sum_pt < 788.4484:
        z += 0.001230664 * Q.sum_pt - 0.9703153
    if Q.C2_b2 < 0.009032972:
        z += 202.3162 * Q.C2_b2 - 1.827517
    if Q.z_top5 >= 0.8919245:
        z += -9.225766 * Q.z_top5 + 8.228687
    if Q.LHA < 0.3127275:
        z += -1.739675 * Q.LHA + 0.5440443
    if Q.z_7 >= 0.04624032:
        z += 58.16051 * Q.z_7 - 2.689361
    if Q.sum_z_dr2 < 0.02530566:
        z += -42.82096 * Q.sum_z_dr2 + 1.083613
    if Q.sum_z_dr < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -53.92156 * (0.1484084 - Q.sum_z_dr) * (6.804164 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.1484084 and Q.pt_7 < 38.53125:
        z += -0.6848992 * (0.1484084 - Q.sum_z_dr) * (38.53125 - Q.pt_7)
    if Q.sum_z_dr < 0.1484084 and Q.lam2 < 0.0003061234:
        z += -12994.64 * (0.1484084 - Q.sum_z_dr) * (0.0003061234 - Q.lam2)
    if Q.lam1 < 0.01643375 and Q.z_dr_0_0p05 > 0.3658817:
        z += -32.50034 * (0.01643375 - Q.lam1) * (Q.z_dr_0_0p05 - 0.3658817)
    if Q.pt_5 < 24.57812 and Q.pt_1 < 169.875:
        z += -0.003273427 * (24.57812 - Q.pt_5) * (169.875 - Q.pt_1)
    if Q.sum_pt > 988.4078 and Q.z_6 > 0.03932388:
        z += 0.4108638 * (Q.sum_pt - 988.4078) * (Q.z_6 - 0.03932388)
    if Q.sum_pt_top5 > 687.4375 and Q.pt_7 < 40.04062:
        z += 0.0007129167 * (Q.sum_pt_top5 - 687.4375) * (40.04062 - Q.pt_7)
    if Q.lam1 < 0.01643375 and Q.pt_7 < 25.57812:
        z += -11.01453 * (0.01643375 - Q.lam1) * (25.57812 - Q.pt_7)
    if Q.e3 < 8.147744e-05 and Q.centroid_offset < 0.03776099:
        z += -645554.2 * (8.147744e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.z_7 > 0.0586137 and Q.lam2 > 4.64158e-05:
        z += -6298.816 * (Q.z_7 - 0.0586137) * (Q.lam2 - 4.64158e-05)
    if Q.e2 < 0.08000524 and Q.pt_4 < 60.03125:
        z += 0.3045948 * (0.08000524 - Q.e2) * (60.03125 - Q.pt_4)
    if Q.sum_pt < 788.4484 and Q.D3 < 2.468971:
        z += -0.0006775702 * (788.4484 - Q.sum_pt) * (2.468971 - Q.D3)
    if Q.sum_pt_top5 > 482.2812 and Q.e4 < 3.0374e-08:
        z += 404924.3 * (Q.sum_pt_top5 - 482.2812) * (3.0374e-08 - Q.e4)
    if Q.pt_5 < 24.57812 and Q.z_2nd < 0.1809419:
        z += 2.793294 * (24.57812 - Q.pt_5) * (0.1809419 - Q.z_2nd)
    if Q.sj3_dr_max > 0.233678 and Q.tau32 < 0.4309923:
        z += -20.86012 * (Q.sj3_dr_max - 0.233678) * (0.4309923 - Q.tau32)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.8104655
    if Q.z_dr_0p05_0p1 < 0.5882598:
        z += -1.118872 * Q.z_dr_0p05_0p1 + 0.6581874
    if Q.psi_0p1 >= 0.9761279:
        z += -18.39602 * Q.psi_0p1 + 17.95687
    if 0.006679471 <= Q.sum_z_dr2 < 0.007520088:
        z += -972.125 * Q.sum_z_dr2 + 6.493281
    if Q.sum_z_dr2 >= 0.007520088:
        z += -1658.677 * Q.sum_z_dr2 + 11.65621
    if 0.002074109 <= Q.sum_zz_dr2 < 0.006390125:
        z += 38.39259 * Q.sum_zz_dr2 - 0.07963043
    if 0.006390125 <= Q.sum_zz_dr2 < 0.00718279:
        z += 372.1678 * Q.sum_zz_dr2 - 2.212496
    if 0.00718279 <= Q.sum_zz_dr2 < 0.008168571:
        z += 1013.604 * Q.sum_zz_dr2 - 6.819794
    if 0.008168571 <= Q.sum_zz_dr2 < 0.01165737:
        z += 1527.805 * Q.sum_zz_dr2 - 11.02008
    if Q.sum_zz_dr2 >= 0.01165737:
        z += -634.2071 * Q.sum_zz_dr2 + 14.1833
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.01323868:
        z += 300.005 * Q.lam1_plus_lam2 - 1.677115
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += -633.4741 * Q.lam1_plus_lam2 + 10.68091
    if 0.03556091 <= Q.e2 < 0.05028464:
        z += -23.52793 * Q.e2 + 0.8366748
    if Q.e2 >= 0.05028464:
        z += -3.922585 * Q.e2 - 0.1491732
    if 0.02355416 <= Q.centroid_offset < 0.03776099:
        z += -43.15873 * Q.centroid_offset + 1.016568
    if 0.03776099 <= Q.centroid_offset < 0.04990367:
        z += -285.1587 * Q.centroid_offset + 10.15473
    if Q.centroid_offset >= 0.04990367:
        z += -701.5515 * Q.centroid_offset + 30.93426
    if 0.2160559 <= Q.LHA < 0.3033137:
        z += 7.516748 * Q.LHA - 1.624038
    if Q.LHA >= 0.3033137:
        z += 29.60155 * Q.LHA - 8.322662
    if 0.03459477 <= Q.tau1 < 0.09538712:
        z += -52.55655 * Q.tau1 + 1.818182
    if Q.tau1 >= 0.09538712:
        z += -32.37153 * Q.tau1 - 0.107209
    if 0.02689598 <= Q.sum_z_dr < 0.04081947:
        z += 39.50087 * Q.sum_z_dr - 1.062415
    if 0.04081947 <= Q.sum_z_dr < 0.08723651:
        z += 132.654 * Q.sum_z_dr - 4.864875
    if 0.08723651 <= Q.sum_z_dr < 0.1019409:
        z += -73.24947 * Q.sum_z_dr + 13.09742
    if Q.sum_z_dr >= 0.1019409:
        z += -448.664 * Q.sum_z_dr + 51.36752
    if Q.n_dr_0p05_0p1 < 5.0:
        z += 0.2632565 * Q.n_dr_0p05_0p1 - 1.316282
    if Q.sj2_dr < 0.1294903:
        z += -8.138853 * Q.sj2_dr + 1.426957
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -12.56879 * Q.sj2_dr + 2.00059
    if Q.sj3_dr_max < 0.213399:
        z += 27.80566 * Q.sj3_dr_max - 6.435275
    if 0.213399 <= Q.sj3_dr_max < 0.233678:
        z += 24.73372 * Q.sj3_dr_max - 5.779725
    if Q.n_dr_0_0p05 >= 5.0:
        z += 0.3587289 * Q.n_dr_0_0p05 - 1.793645
    if Q.sum_z_dr2_top5 < 0.00422683:
        z += -9.682358 * Q.sum_z_dr2_top5 - 0.1355579
    if 0.00422683 <= Q.sum_z_dr2_top5 < 0.005005443:
        z += 226.664 * Q.sum_z_dr2_top5 - 1.134554
    if Q.n_dr_0p1_0p2 >= 1.0:
        z += 0.2898654 * Q.n_dr_0p1_0p2 - 0.2898654
    if Q.z_dr_0p1_0p2 < 0.1585582:
        z += -2.299635 * Q.z_dr_0p1_0p2 + 0.364626
    if Q.C2 < 0.03578649:
        z += 15.58898 * Q.C2 - 0.557875
    if Q.e3 < 1.960701e-05:
        z += -25945.98 * Q.e3 + 0.5087231
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -2835.153 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -402.7106 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.planar_flow < 0.1115136 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -32.67734 * (0.1115136 - Q.planar_flow) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.sum_zz_dr2 > 0.002074109 and Q.sj2_dr < 0.1872617:
        z += -5594.457 * (Q.sum_zz_dr2 - 0.002074109) * (0.1872617 - Q.sj2_dr)
    if Q.psi_0p1 > 0.8155839 and Q.centroid_offset > 0.03776099:
        z += 961.6725 * (Q.psi_0p1 - 0.8155839) * (Q.centroid_offset - 0.03776099)
    if Q.sum_z_dr2 > 0.008678045 and Q.log_sum_pt > 6.267538:
        z += -9287.995 * (Q.sum_z_dr2 - 0.008678045) * (Q.log_sum_pt - 6.267538)
    if Q.sum_zz_dr2 > 0.002074109 and Q.log_sum_pt > 6.267538:
        z += 373.1584 * (Q.sum_zz_dr2 - 0.002074109) * (Q.log_sum_pt - 6.267538)
    if Q.lam1_plus_lam2 > 0.01323868 and Q.sum_pt > 488.9312:
        z += -5.339375 * (Q.lam1_plus_lam2 - 0.01323868) * (Q.sum_pt - 488.9312)
    if Q.sum_zz_dr2 > 0.008168571 and Q.sum_pt > 488.9312:
        z += 3.590528 * (Q.sum_zz_dr2 - 0.008168571) * (Q.sum_pt - 488.9312)
    if Q.psi_0p1 > 0.6332416 and Q.eccentricity > 0.9704496:
        z += 150.6218 * (Q.psi_0p1 - 0.6332416) * (Q.eccentricity - 0.9704496)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.dr0_5 > 0.290089:
        z += -11685.06 * (Q.z_dr_0p05_0p1 - 0.7509095) * (Q.dr0_5 - 0.290089)
    if Q.centroid_offset > 0.04990367 and Q.eta_5 > 0.1122437:
        z += -37083.69 * (Q.centroid_offset - 0.04990367) * (Q.eta_5 - 0.1122437)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.503921
    if Q.N2 < 0.2233283:
        z += -18.36563 * Q.N2 + 4.101564
    if Q.sum_z_dr2 < 0.006679471:
        z += 1213.332 * Q.sum_z_dr2 - 5.896631
    if 0.006679471 <= Q.sum_z_dr2 < 0.007520088:
        z += 768.9015 * Q.sum_z_dr2 - 2.928073
    if 0.007520088 <= Q.sum_z_dr2 < 0.01323868:
        z += -499.0977 * Q.sum_z_dr2 + 6.607392
    if Q.sum_zz_dr2 < 0.005834489:
        z += -487.7948 * Q.sum_zz_dr2 + 2.423364
    if 0.005834489 <= Q.sum_zz_dr2 < 0.006390125:
        z += -341.7413 * Q.sum_zz_dr2 + 1.571216
    if 0.006390125 <= Q.sum_zz_dr2 < 0.01165737:
        z += 230.4179 * Q.sum_zz_dr2 - 2.084953
    if 0.01165737 <= Q.sum_zz_dr2 < 0.01716248:
        z += -109.1923 * Q.sum_zz_dr2 + 1.874012
    if Q.e2 < 0.04110972:
        z += -45.04592 * Q.e2 + 1.356793
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += 12.3568 * Q.e2 - 1.003017
    if 0.05028464 <= Q.e2 < 0.06344108:
        z += 29.00935 * Q.e2 - 1.840384
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 529.7184 * Q.sum_z_dr2_top3 - 1.139725
    if Q.lam1 < 0.005433361:
        z += 94.88556 * Q.lam1 - 0.3768759
    if 0.005433361 <= Q.lam1 < 0.00733008:
        z += -217.5578 * Q.lam1 + 1.320742
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 262.0532 * Q.lam1 - 2.194846
    if 0.1789613 <= Q.sj3_dr_max < 0.1879486:
        z += 18.99342 * Q.sj3_dr_max - 3.399087
    if 0.1879486 <= Q.sj3_dr_max < 0.233678:
        z += 25.385 * Q.sj3_dr_max - 4.600377
    if 0.233678 <= Q.sj3_dr_max < 0.2623172:
        z += 37.33146 * Q.sj3_dr_max - 7.392002
    if Q.sj3_dr_max >= 0.2623172:
        z += 7.503261 * Q.sj3_dr_max + 0.43245
    if Q.dr_0 < 0.08082334:
        z += -6.220864 * Q.dr_0 + 0.5027911
    if Q.z_dr_0_0p05 < 0.1515405:
        z += -2.162162 * Q.z_dr_0_0p05 + 0.3276549
    if Q.lam1_plus_lam2 < 0.002635418:
        z += 963.4518 * Q.lam1_plus_lam2 - 2.539098
    if Q.tau1 < 0.0811449:
        z += -53.52975 * Q.tau1 + 4.218538
    if 0.0811449 <= Q.tau1 < 0.1027642:
        z += -10.75867 * Q.tau1 + 0.7478833
    if 0.1027642 <= Q.tau1 < 0.1136369:
        z += 32.90117 * Q.tau1 - 3.738786
    if Q.lam2 < 0.001130645:
        z += 528.2216 * Q.lam2 - 0.597231
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += -17.11812 * Q.z_dr_0p05_0p1 + 12.85416
    if Q.max_dr >= 0.121681:
        z += -14.40882 * Q.max_dr + 1.753279
    if Q.N2 < 0.2233283 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -14.28469 * (0.2233283 - Q.N2) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1872617:
        z += -304.3004 * (0.2233283 - Q.N2) * (0.1872617 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.sum_z_dr2 > 0.008678045:
        z += -6885.998 * (0.2233283 - Q.N2) * (Q.sum_z_dr2 - 0.008678045)
    if Q.N2 < 0.2233283 and Q.LHA > 0.2931906:
        z += 191.7854 * (0.2233283 - Q.N2) * (Q.LHA - 0.2931906)
    if Q.N2 < 0.2233283 and Q.LHA > 0.4235925:
        z += -5363.136 * (0.2233283 - Q.N2) * (Q.LHA - 0.4235925)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3255822:
        z += 325.379 * (0.2233283 - Q.N2) * (Q.LHA - 0.3255822)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3855647:
        z += -341.6866 * (0.2233283 - Q.N2) * (Q.LHA - 0.3855647)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.01954503 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
    if Q.N2 < 0.2233283 and Q.e3 < 8.147744e-05:
        z += -197282.0 * (0.2233283 - Q.N2) * (8.147744e-05 - Q.e3)
    if Q.sum_z_dr2_top3 < 0.002151568 and Q.eccentricity > 0.9598562:
        z += -24987.25 * (0.002151568 - Q.sum_z_dr2_top3) * (Q.eccentricity - 0.9598562)
    if Q.planar_flow < 0.1950135 and Q.mean_phi > -0.01753483:
        z += -110.5901 * (0.1950135 - Q.planar_flow) * (Q.mean_phi - -0.01753483)
    if Q.sum_z_dr2 < 0.006679471 and Q.dr12 > 0.1587481:
        z += -29919.03 * (0.006679471 - Q.sum_z_dr2) * (Q.dr12 - 0.1587481)
    if Q.sum_zz_dr2 < 0.006390125 and Q.dr12 > 0.1587481:
        z += 24954.11 * (0.006390125 - Q.sum_zz_dr2) * (Q.dr12 - 0.1587481)
    if Q.sum_z_dr2 < 0.006679471 and Q.pt_entropy > 1.797618:
        z += -673.2327 * (0.006679471 - Q.sum_z_dr2) * (Q.pt_entropy - 1.797618)
    if Q.N2 < 0.2233283 and Q.pt_7 < 25.57812:
        z += -0.9764468 * (0.2233283 - Q.N2) * (25.57812 - Q.pt_7)
    if Q.planar_flow < 0.1950135 and Q.z_7 < 0.07148865:
        z += -182.9406 * (0.1950135 - Q.planar_flow) * (0.07148865 - Q.z_7)
    if Q.sum_z_dr2 < 0.01323868 and Q.centroid_offset > 0.04990367:
        z += -18927.18 * (0.01323868 - Q.sum_z_dr2) * (Q.centroid_offset - 0.04990367)
    if Q.sj3_dr_max > 0.233678 and Q.pt_1 < 182.125:
        z += -0.03698898 * (Q.sj3_dr_max - 0.233678) * (182.125 - Q.pt_1)
    if Q.planar_flow < 0.1950135 and Q.sum_pt_top5 > 402.625:
        z += 0.04068369 * (0.1950135 - Q.planar_flow) * (Q.sum_pt_top5 - 402.625)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += 210.4448 * (Q.z_dr_0p05_0p1 - 0.7509095) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sj3_dr_max > 0.1426152 and Q.pt_1 < 169.875:
        z += -0.05631112 * (Q.sj3_dr_max - 0.1426152) * (169.875 - Q.pt_1)
    if Q.sj3_dr_max > 0.1879486 and Q.pt_1 < 169.875:
        z += 0.08719927 * (Q.sj3_dr_max - 0.1879486) * (169.875 - Q.pt_1)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.D2_b2 < 4.721224:
        z += 1.211671 * (Q.z_dr_0p05_0p1 - 0.7509095) * (4.721224 - Q.D2_b2)
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
