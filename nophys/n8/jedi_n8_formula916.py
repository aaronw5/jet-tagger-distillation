"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.7% (the network: 65.8%); same class as the network for 90.5% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
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
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.pair_mass_0_7          mass of particles 0 and 7 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.m01                    mass of particles 0 and 1 [GeV]
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.ptdr0_7                pT7 · ΔR(0, 7) [GeV]
  Q.z_2                    pT of particle 2 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
  Q.z_1st                  largest pT share
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_1               |Δη| of particle 1
  Q.abseta_2               |Δη| of particle 2
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_6               |Δη| of particle 6
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_5               |Δφ| of particle 5
  Q.absphi_6               |Δφ| of particle 6
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_3                  ΔR between particle 3 and the hardest particle
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr1_3                  ΔR between particle 3 and the 2nd-hardest particle
  Q.dr1_6                  ΔR between particle 6 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.phi_0                  Δφ of particle 0
  Q.phi_1                  Δφ of particle 1
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
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
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
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
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_6=pair_mass(0, 6),
        pair_mass_0_7=pair_mass(0, 7),
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        m01=pair_mass(0, 1),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        pt_0=pt[0],
        pt_1=pt[1],
        pt_2=pt[2],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        pt_5=pt[5],
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        pt_7=pt[7],
        ptdr0_7=pt[7] * math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        z_2=z[2],
        z_3=z[3],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        z_1st=zs[0],
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_0=abs(eta[0]),
        abseta_1=abs(eta[1]),
        abseta_2=abs(eta[2]),
        abseta_4=abs(eta[4]),
        abseta_6=abs(eta[6]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_5=abs(phi[5]),
        absphi_6=abs(phi[6]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_3=math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr1_3=math.sqrt(dist2(1, 3)) if pt[3] > 0 else 0.0,
        dr1_6=math.sqrt(dist2(1, 6)) if pt[6] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        phi_0=phi[0],
        phi_1=phi[1],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
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
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -1.989818
    if Q.planar_flow < 0.1484197:
        z += -11.91903 * Q.planar_flow + 1.769019
    if Q.width < 0.004372139:
        z += 869.6759 * Q.width - 1.779969
    if 0.004372139 <= Q.width < 0.008678045:
        z += -469.6748 * Q.width + 4.075859
    if Q.girth2 < 0.01323868:
        z += -253.2001 * Q.girth2 + 3.228553
    if 0.01323868 <= Q.girth2 < 0.01882765:
        z += 22.09372 * Q.girth2 - 0.4159728
    if Q.mass < 21.78408:
        z += 0.09584456 * Q.mass - 4.352338
    if 21.78408 <= Q.mass < 29.6447:
        z += 0.1253934 * Q.mass - 4.996034
    if 29.6447 <= Q.mass < 56.92035:
        z += 0.0402492 * Q.mass - 2.471958
    if 56.92035 <= Q.mass < 64.61873:
        z += 0.02350615 * Q.mass - 1.518938
    if Q.girth2_top3 < 0.006756161:
        z += -45.20543 * Q.girth2_top3 + 0.110866
    if 0.006756161 <= Q.girth2_top3 < 0.007929074:
        z += 165.8684 * Q.girth2_top3 - 1.315183
    if Q.lam1 < 0.0002758826:
        z += -14704.29 * Q.lam1 + 2.860942
    if 0.0002758826 <= Q.lam1 < 0.005433361:
        z += 249.4633 * Q.lam1 - 1.264537
    if 0.005433361 <= Q.lam1 < 0.006506576:
        z += -84.68658 * Q.lam1 + 0.5510196
    if Q.sum_pt >= 901.5938:
        z += -0.01588304 * Q.sum_pt + 14.32005
    if Q.C2_b2 < 0.001563465:
        z += 718.7921 * Q.C2_b2 - 1.123807
    if Q.sj3_dr_max < 0.04889979:
        z += 1.764144 * Q.sj3_dr_max + 2.147722
    if 0.04889979 <= Q.sj3_dr_max < 0.1070199:
        z += -8.854428 * Q.sj3_dr_max + 2.666968
    if 0.1070199 <= Q.sj3_dr_max < 0.1789613:
        z += 8.840186 * Q.sj3_dr_max + 0.7732913
    if 0.1789613 <= Q.sj3_dr_max < 0.233678:
        z += -9.127124 * Q.sj3_dr_max + 3.988745
    if 0.233678 <= Q.sj3_dr_max < 0.3012016:
        z += -22.77894 * Q.sj3_dr_max + 7.178874
    if Q.sj3_dr_max >= 0.3012016:
        z += -13.92451 * Q.sj3_dr_max + 4.511906
    if 0.006789738 <= Q.centroid_offset < 0.04990367:
        z += -62.92365 * Q.centroid_offset + 0.4272351
    if Q.centroid_offset >= 0.04990367:
        z += -792.0786 * Q.centroid_offset + 36.81474
    if Q.girth < 0.07608178:
        z += 51.33802 * Q.girth - 4.084532
    if 0.07608178 <= Q.girth < 0.08723651:
        z += 16.01509 * Q.girth - 1.397101
    if Q.e2 < 0.0245477:
        z += -169.4402 * Q.e2 + 5.032127
    if 0.0245477 <= Q.e2 < 0.03556091:
        z += -79.24666 * Q.e2 + 2.818083
    if Q.girth2_top5 < 0.008329695:
        z += 25.61889 * Q.girth2_top5 - 0.2133975
    if Q.log_sum_pt < 6.080494:
        z += 5.784472 * Q.log_sum_pt - 35.17245
    if 6.377723 <= Q.log_sum_pt < 6.670067:
        z += 1.068376 * Q.log_sum_pt - 6.813804
    if Q.log_sum_pt >= 6.670067:
        z += 0.4791648 * Q.log_sum_pt - 2.883728
    if Q.mass_over_sum_pt_sq < 0.003904593:
        z += -462.796 * Q.mass_over_sum_pt_sq + 1.80703
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.00317066 * Q.sum_pt_top5 - 2.179631
    if Q.lam2 < 7.300726e-05:
        z += 15353.73 * Q.lam2 - 1.120934
    if Q.mass_over_sum_pt < 0.08475161:
        z += 33.73927 * Q.mass_over_sum_pt - 2.356948
    if 0.08475161 <= Q.mass_over_sum_pt < 0.1309286:
        z += -10.88223 * Q.mass_over_sum_pt + 1.424795
    if Q.width < 0.004372139 and Q.D2 < 0.7459513:
        z += -2302.51 * (0.004372139 - Q.width) * (0.7459513 - Q.D2)
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += -1119.743 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.girth2 < 0.01323868 and Q.D2 < 1.002471:
        z += 221.2355 * (0.01323868 - Q.girth2) * (1.002471 - Q.D2)
    if Q.planar_flow < 0.1484197 and Q.sum_pt_top2 < 358.375:
        z += -0.03408032 * (0.1484197 - Q.planar_flow) * (358.375 - Q.sum_pt_top2)
    if Q.girth2 < 0.01323868 and Q.mean_phi < -0.003096655:
        z += 883.3702 * (0.01323868 - Q.girth2) * (-0.003096655 - Q.mean_phi)
    if Q.girth2 < 0.01323868 and Q.centroid_offset > 0.01837778:
        z += -1570.959 * (0.01323868 - Q.girth2) * (Q.centroid_offset - 0.01837778)
    if Q.girth2 < 0.01323868 and Q.phi_0 > -0.008995056:
        z += 629.8013 * (0.01323868 - Q.girth2) * (Q.phi_0 - -0.008995056)
    if Q.log_sum_pt > 6.670067 and Q.dr_4 < 0.07232166:
        z += 129.4047 * (Q.log_sum_pt - 6.670067) * (0.07232166 - Q.dr_4)
    if Q.girth2 < 0.01882765 and Q.pt_6 < 46.125:
        z += -0.4357348 * (0.01882765 - Q.girth2) * (46.125 - Q.pt_6)
    if Q.mass < 64.61873 and Q.D2_b2 < 0.1830092:
        z += -0.1266744 * (64.61873 - Q.mass) * (0.1830092 - Q.D2_b2)
    if Q.lam2 < 7.300726e-05 and Q.D2_b2 < 0.2669656:
        z += 79680.71 * (7.300726e-05 - Q.lam2) * (0.2669656 - Q.D2_b2)
    if Q.sj3_dr_max < 0.3012016 and Q.D2_b2 < 0.05744392:
        z += -70.29379 * (0.3012016 - Q.sj3_dr_max) * (0.05744392 - Q.D2_b2)
    if Q.planar_flow < 0.1484197 and Q.dr0_6 > 0.1755206:
        z += -22.85019 * (0.1484197 - Q.planar_flow) * (Q.dr0_6 - 0.1755206)
    if Q.log_sum_pt > 6.377723 and Q.mean_eta > 6.288824e-05:
        z += -41.37036 * (Q.log_sum_pt - 6.377723) * (Q.mean_eta - 6.288824e-05)
    if Q.sum_pt_top5 > 687.4375 and Q.dr_4 < 0.06336451:
        z += -0.08861581 * (Q.sum_pt_top5 - 687.4375) * (0.06336451 - Q.dr_4)
    if Q.lam2 < 7.300726e-05 and Q.dr_2 < 0.08525808:
        z += 98238.32 * (7.300726e-05 - Q.lam2) * (0.08525808 - Q.dr_2)
    if Q.sum_pt > 901.5938 and Q.pt_7 > 33.21875:
        z += -0.0001231385 * (Q.sum_pt - 901.5938) * (Q.pt_7 - 33.21875)
    if Q.log_sum_pt > 6.670067 and Q.z_7 > 0.01685855:
        z += -462.2692 * (Q.log_sum_pt - 6.670067) * (Q.z_7 - 0.01685855)
    if Q.sj3_dr_max < 0.3012016 and Q.z_7 < 0.05557716:
        z += -177.5809 * (0.3012016 - Q.sj3_dr_max) * (0.05557716 - Q.z_7)
    if Q.sj3_dr_max < 0.3012016 and Q.n_dr_0p05_0p1 < 2.0:
        z += -0.5643453 * (0.3012016 - Q.sj3_dr_max) * (2.0 - Q.n_dr_0p05_0p1)
    if Q.log_sum_pt > 6.670067 and Q.m01 < 28.78966:
        z += 0.03251923 * (Q.log_sum_pt - 6.670067) * (28.78966 - Q.m01)
    if Q.planar_flow < 0.1484197 and Q.dr_2 < 0.02270492:
        z += -715.0209 * (0.1484197 - Q.planar_flow) * (0.02270492 - Q.dr_2)
    if Q.girth2_top3 < 0.007929074 and Q.D2_b2 < 0.5327104:
        z += -90.47185 * (0.007929074 - Q.girth2_top3) * (0.5327104 - Q.D2_b2)
    if Q.centroid_offset > 0.02076709 and Q.z_7 < 0.06473447:
        z += -1367.024 * (Q.centroid_offset - 0.02076709) * (0.06473447 - Q.z_7)
    if Q.planar_flow < 0.1484197 and Q.pt_dispersion < 0.4160096:
        z += 59.41533 * (0.1484197 - Q.planar_flow) * (0.4160096 - Q.pt_dispersion)
    if Q.centroid_offset > 0.006789738 and Q.n_pt_above_50 < 7.0:
        z += 4.810093 * (Q.centroid_offset - 0.006789738) * (7.0 - Q.n_pt_above_50)
    if Q.centroid_offset > 0.02076709 and Q.dr_2 < 0.05056028:
        z += -1292.377 * (Q.centroid_offset - 0.02076709) * (0.05056028 - Q.dr_2)
    if Q.width < 0.004372139 and Q.C3 < 0.03532852:
        z += 11497.83 * (0.004372139 - Q.width) * (0.03532852 - Q.C3)
    if Q.centroid_offset > 0.006789738 and Q.pt_1 < 223.375:
        z += 0.1129974 * (Q.centroid_offset - 0.006789738) * (223.375 - Q.pt_1)
    return max(0.0, z)


def neuron_1(Q):
    z = -0.06245375
    if Q.lam1 < 0.0008722282:
        z += 372.2596 * Q.lam1 + 1.645766
    if 0.0008722282 <= Q.lam1 < 0.00595415:
        z += -360.9252 * Q.lam1 + 2.285271
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += -56.27608 * Q.lam1 + 0.4713444
    if Q.pt_7 < 25.57812:
        z += 0.06317856 * Q.pt_7 - 1.615989
    if 34.53125 <= Q.pt_7 < 53.4375:
        z += 0.03527876 * Q.pt_7 - 1.21822
    if Q.pt_7 >= 53.4375:
        z += -0.07220257 * Q.pt_7 + 4.525314
    if 6.377723 <= Q.log_sum_pt < 6.502799:
        z += 3.767792 * Q.log_sum_pt - 24.02993
    if 6.502799 <= Q.log_sum_pt < 6.605974:
        z += 13.25855 * Q.log_sum_pt - 85.74645
    if Q.log_sum_pt >= 6.605974:
        z += 17.50296 * Q.log_sum_pt - 113.7849
    if Q.mass_over_sum_pt_sq < 0.005832932:
        z += -598.8414 * Q.mass_over_sum_pt_sq + 3.493001
    if Q.z_7 < 0.02320757:
        z += 40.80106 * Q.z_7 - 2.515189
    if 0.02320757 <= Q.z_7 < 0.04624032:
        z += 69.40099 * Q.z_7 - 3.178923
    if 0.04624032 <= Q.z_7 < 0.06164517:
        z += 49.80918 * Q.z_7 - 2.272992
    if Q.z_7 >= 0.06164517:
        z += 9.008118 * Q.z_7 + 0.2421967
    if Q.girth2 < 0.005019719:
        z += 557.2545 * Q.girth2 - 4.514977
    if 0.005019719 <= Q.girth2 < 0.008678045:
        z += 469.536 * Q.girth2 - 4.074655
    if Q.mass < 49.6681:
        z += -0.01337204 * Q.mass + 0.6641637
    if Q.sj3_dr_max >= 0.169029:
        z += 3.480095 * Q.sj3_dr_max - 0.5882371
    if Q.e2_sq < 0.008168571:
        z += -199.8503 * Q.e2_sq + 1.632492
    if 367.5938 <= Q.sum_pt_top5 < 531.1875:
        z += 0.001635368 * Q.sum_pt_top5 - 0.6011511
    if Q.sum_pt_top5 >= 531.1875:
        z += -0.01274239 * Q.sum_pt_top5 + 7.036136
    if Q.width < 0.00609665:
        z += 559.4479 * Q.width - 3.410758
    if Q.girth < 0.0717028:
        z += 8.962386 * Q.girth - 1.393288
    if 0.0717028 <= Q.girth < 0.1019409:
        z += 24.82493 * Q.girth - 2.530676
    if Q.tau1 < 0.07283629:
        z += 10.47065 * Q.tau1 - 0.7626435
    if Q.e3 < 0.0005116989:
        z += -898.7269 * Q.e3 + 0.4598776
    if Q.zdr_0 < 0.0211821:
        z += 16.237 * Q.zdr_0 - 0.3439337
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += -14974.89 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.z_7 < 0.06164517 and Q.tau2 < 0.06297984:
        z += 355.8264 * (0.06164517 - Q.z_7) * (0.06297984 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.D2 < 1.679198:
        z += -14.61625 * (0.06164517 - Q.z_7) * (1.679198 - Q.D2)
    if Q.pt_7 > 34.53125 and Q.sj2_dr > 0.1294903:
        z += 0.3359977 * (Q.pt_7 - 34.53125) * (Q.sj2_dr - 0.1294903)
    if Q.log_sum_pt > 6.377723 and Q.z_dr_0p05_0p1 < 0.7509095:
        z += -3.039866 * (Q.log_sum_pt - 6.377723) * (0.7509095 - Q.z_dr_0p05_0p1)
    if Q.z_7 < 0.06164517 and Q.z_dr_0p05_0p1 > 0.04875823:
        z += -35.35703 * (0.06164517 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.04875823)
    if Q.pt_7 > 34.53125 and Q.n_dr_0p1_0p2 > 1.0:
        z += 0.006885915 * (Q.pt_7 - 34.53125) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.pt_7 > 34.53125 and Q.pt_6 < 52.90625:
        z += -0.002863246 * (Q.pt_7 - 34.53125) * (52.90625 - Q.pt_6)
    if Q.log_sum_pt > 6.377723 and Q.centroid_offset > 0.009480685:
        z += 82.98721 * (Q.log_sum_pt - 6.377723) * (Q.centroid_offset - 0.009480685)
    if Q.log_sum_pt > 6.605974 and Q.D2 < 1.432482:
        z += 1.981666 * (Q.log_sum_pt - 6.605974) * (1.432482 - Q.D2)
    if Q.lam1 < 0.008375572 and Q.planar_flow < 0.1484197:
        z += -504.226 * (0.008375572 - Q.lam1) * (0.1484197 - Q.planar_flow)
    if Q.sj3_dr_max > 0.169029 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.08956487 * (Q.sj3_dr_max - 0.169029) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.D2_b2 < 0.03885671:
        z += -25719.21 * (0.005832932 - Q.mass_over_sum_pt_sq) * (0.03885671 - Q.D2_b2)
    if Q.lam1 < 0.008375572 and Q.n_pt_above_50 > 5.0:
        z += -26.2414 * (0.008375572 - Q.lam1) * (Q.n_pt_above_50 - 5.0)
    if Q.girth2 < 0.008678045 and Q.centroid_offset > 0.02076709:
        z += 20891.06 * (0.008678045 - Q.girth2) * (Q.centroid_offset - 0.02076709)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.centroid_offset > 0.00809236:
        z += -5013.681 * (0.005832932 - Q.mass_over_sum_pt_sq) * (Q.centroid_offset - 0.00809236)
    if Q.z_7 < 0.06164517 and Q.mean_phi2 < 0.008921136:
        z += 2087.515 * (0.06164517 - Q.z_7) * (0.008921136 - Q.mean_phi2)
    if Q.z_7 < 0.06164517 and Q.girth2_top2 < 0.01403324:
        z += 1417.438 * (0.06164517 - Q.z_7) * (0.01403324 - Q.girth2_top2)
    if Q.log_sum_pt > 6.502799 and Q.zdr_6 < 0.008654951:
        z += -497.1859 * (Q.log_sum_pt - 6.502799) * (0.008654951 - Q.zdr_6)
    if Q.sum_pt_top5 > 531.1875 and Q.zdr_6 < 0.01392641:
        z += 0.5014613 * (Q.sum_pt_top5 - 531.1875) * (0.01392641 - Q.zdr_6)
    if Q.pt_7 > 34.53125 and Q.pt_3 < 77.625:
        z += -0.001256487 * (Q.pt_7 - 34.53125) * (77.625 - Q.pt_3)
    return max(0.0, z)


def neuron_2(Q):
    z = 3.835844
    if Q.sj3_pair_mass_max < 63.68899:
        z += -0.01883932 * Q.sj3_pair_mass_max + 1.199857
    if Q.log_sum_pt < 6.46415:
        z += -5.533853 * Q.log_sum_pt + 36.29631
    if 6.46415 <= Q.log_sum_pt < 6.605974:
        z += -3.699345 * Q.log_sum_pt + 24.43778
    if 6.842717 <= Q.log_sum_pt < 6.896095:
        z += 13.15428 * Q.log_sum_pt - 90.01104
    if Q.log_sum_pt >= 6.896095:
        z += 19.31269 * Q.log_sum_pt - 132.48
    if Q.z_7 < 0.02320757:
        z += 18.26531 * Q.z_7 - 0.072324
    if 0.02320757 <= Q.z_7 < 0.03243272:
        z += -41.73938 * Q.z_7 + 1.320239
    if 0.03243272 <= Q.z_7 < 0.04939969:
        z += -73.32071 * Q.z_7 + 2.344507
    if 0.04939969 <= Q.z_7 < 0.07148865:
        z += -56.57195 * Q.z_7 + 1.517124
    if Q.z_7 >= 0.07148865:
        z += -43.25593 * Q.z_7 + 0.5651793
    if Q.girth < 0.007673833:
        z += 211.7963 * Q.girth - 2.182468
    if 0.007673833 <= Q.girth < 0.1484084:
        z += 3.959074 * Q.girth - 0.5875599
    if Q.LHA >= 0.111565:
        z += -8.714852 * Q.LHA + 0.9722729
    if Q.sum_pt_top5 < 605.1875:
        z += 0.006911157 * Q.sum_pt_top5 - 4.95449
    if 605.1875 <= Q.sum_pt_top5 < 716.8828:
        z += 0.001874788 * Q.sum_pt_top5 - 1.906542
    if 716.8828 <= Q.sum_pt_top5 < 839.9547:
        z += -0.005036369 * Q.sum_pt_top5 + 3.047948
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.01210695 * Q.sum_pt_top5 + 8.986914
    if Q.pt_6 < 27.57812:
        z += 0.01225192 * Q.pt_6 - 0.505009
    if 27.57812 <= Q.pt_6 < 41.21875:
        z += 0.043513 * Q.pt_6 - 1.367131
    if Q.pt_6 >= 41.21875:
        z += 0.03126108 * Q.pt_6 - 0.8621218
    if Q.z_6 >= 0.02886576:
        z += -11.41097 * Q.z_6 + 0.3293864
    if Q.sum_pt < 559.6875:
        z += -0.007493157 * Q.sum_pt + 6.870644
    if 559.6875 <= Q.sum_pt < 788.4484:
        z += -0.01091435 * Q.sum_pt + 8.785444
    if 788.4484 <= Q.sum_pt < 840.0195:
        z += -0.003491112 * Q.sum_pt + 2.932602
    if Q.girth2 < 0.0003193707:
        z += 9474.929 * Q.girth2 - 2.629354
    if 0.0003193707 <= Q.girth2 < 0.005019719:
        z += -84.3897 * Q.girth2 + 0.4236126
    if Q.pt_7 < 20.125:
        z += 0.02410657 * Q.pt_7 - 1.935986
    if 20.125 <= Q.pt_7 < 43.5:
        z += 0.08972348 * Q.pt_7 - 3.256526
    if 43.5 <= Q.pt_7 < 53.4375:
        z += 0.15491 * Q.pt_7 - 6.092141
    if Q.pt_7 >= 53.4375:
        z += 0.06561691 * Q.pt_7 - 1.32054
    if Q.m012 >= 42.18339:
        z += 0.02207209 * Q.m012 - 0.9310755
    if 15.45403 <= Q.mass < 36.22941:
        z += 0.004540286 * Q.mass - 0.07016573
    if Q.mass >= 36.22941:
        z += -0.01529159 * Q.mass + 0.6483313
    if Q.zdr_0 < 0.0211821:
        z += -19.55162 * Q.zdr_0 + 0.4141445
    if Q.mass_top5 >= 49.18618:
        z += 0.01082055 * Q.mass_top5 - 0.5322218
    if Q.lam2 < 0.0001947983:
        z += 916.5146 * Q.lam2 - 0.1785355
    if Q.dr_5 < 0.02164863:
        z += 15.86548 * Q.dr_5 - 0.343466
    if Q.lam1 < 0.01200373:
        z += -87.54958 * Q.lam1 + 1.050921
    if Q.z_top5 >= 0.8770155:
        z += 8.017109 * Q.z_top5 - 7.031128
    if Q.pt_5 < 33.02656:
        z += 0.03067341 * Q.pt_5 - 1.013037
    if Q.centroid_offset < 0.004430536:
        z += 73.27303 * Q.centroid_offset - 0.3246388
    if Q.z_5 < 0.03672711:
        z += -18.7335 * Q.z_5 + 0.6880272
    if Q.sj3_pair_mass_max < 63.68899 and Q.z_7 < 0.06810151:
        z += -0.5790561 * (63.68899 - Q.sj3_pair_mass_max) * (0.06810151 - Q.z_7)
    if Q.sj3_pair_mass_max < 63.68899 and Q.centroid_offset > 0.01096064:
        z += -0.536414 * (63.68899 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.01096064)
    if Q.lam1 < 0.00595415 and Q.max_dr > 0.0931108:
        z += -2168.511 * (0.00595415 - Q.lam1) * (Q.max_dr - 0.0931108)
    if Q.z_7 > 0.04939969 and Q.sj3_dr_min < 0.0623951:
        z += -195.8653 * (Q.z_7 - 0.04939969) * (0.0623951 - Q.sj3_dr_min)
    if Q.log_sum_pt > 6.842717 and Q.pt_6 > 41.21875:
        z += -0.1435691 * (Q.log_sum_pt - 6.842717) * (Q.pt_6 - 41.21875)
    if Q.girth < 0.007673833 and Q.pt_4 < 71.6875:
        z += 4.824352 * (0.007673833 - Q.girth) * (71.6875 - Q.pt_4)
    if Q.pt_6 > 27.57812 and Q.planar_flow < 0.7974684:
        z += -0.01216847 * (Q.pt_6 - 27.57812) * (0.7974684 - Q.planar_flow)
    if Q.sum_pt < 788.4484 and Q.max_dr < 0.03623337:
        z += 0.04644091 * (788.4484 - Q.sum_pt) * (0.03623337 - Q.max_dr)
    if Q.girth2 < 0.0003193707 and Q.mass_top2 < 36.76827:
        z += 255.4773 * (0.0003193707 - Q.girth2) * (36.76827 - Q.mass_top2)
    if Q.z_7 < 0.07148865 and Q.planar_flow < 0.6947818:
        z += -10.12671 * (0.07148865 - Q.z_7) * (0.6947818 - Q.planar_flow)
    if Q.log_sum_pt > 6.896095 and Q.dr_5 < 0.04416271:
        z += -172.3702 * (Q.log_sum_pt - 6.896095) * (0.04416271 - Q.dr_5)
    if Q.log_sum_pt > 6.842717 and Q.dr_5 < 0.04416271:
        z += 87.91174 * (Q.log_sum_pt - 6.842717) * (0.04416271 - Q.dr_5)
    if Q.sum_pt < 840.0195 and Q.dr_5 < 0.02164863:
        z += 0.7026363 * (840.0195 - Q.sum_pt) * (0.02164863 - Q.dr_5)
    if Q.sum_pt < 788.4484 and Q.dr_5 < 0.02164863:
        z += -0.7880715 * (788.4484 - Q.sum_pt) * (0.02164863 - Q.dr_5)
    if Q.log_sum_pt < 6.605974 and Q.lam2 < 0.0003061234:
        z += 2192.81 * (6.605974 - Q.log_sum_pt) * (0.0003061234 - Q.lam2)
    if Q.sum_pt_top5 > 839.9547 and Q.D2_b2 < 2.326647:
        z += 0.00599087 * (Q.sum_pt_top5 - 839.9547) * (2.326647 - Q.D2_b2)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 3.032555:
        z += -6.298771 * (Q.log_sum_pt - 6.896095) * (3.032555 - Q.D2_b2)
    if Q.sj3_pair_mass_min < 15.95929 and Q.D2 > 0.6203774:
        z += 0.003863336 * (15.95929 - Q.sj3_pair_mass_min) * (Q.D2 - 0.6203774)
    if Q.pt_7 < 43.5 and Q.D2_b2 < 3.032555:
        z += -0.004073673 * (43.5 - Q.pt_7) * (3.032555 - Q.D2_b2)
    if Q.sum_pt < 559.6875 and Q.D2_b2 < 4.721224:
        z += 0.001347548 * (559.6875 - Q.sum_pt) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.896095 and Q.zdr_5 < 0.002460108:
        z += 1934.448 * (Q.log_sum_pt - 6.896095) * (0.002460108 - Q.zdr_5)
    return max(0.0, z)


def neuron_3(Q):
    z = -4.310726
    if Q.mass_over_sum_pt >= 0.0681391:
        z += 69.02568 * Q.mass_over_sum_pt - 4.703348
    if 0.05356915 <= Q.tau1 < 0.1027642:
        z += -53.00389 * Q.tau1 + 2.839374
    if Q.tau1 >= 0.1027642:
        z += -19.04 * Q.tau1 - 0.6508995
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += 545.6614 * Q.lam1 - 4.570226
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 598.641 * Q.lam1 - 5.206179
    if Q.lam1 >= 0.01643375:
        z += 1018.531 * Q.lam1 - 12.10654
    if 0.04081947 <= Q.girth < 0.07608178:
        z += 71.00538 * Q.girth - 2.898402
    if 0.07608178 <= Q.girth < 0.08723651:
        z += 150.2176 * Q.girth - 8.925005
    if Q.girth >= 0.08723651:
        z += 132.0394 * Q.girth - 7.339208
    if 0.007520088 <= Q.width < 0.01323868:
        z += 480.4641 * Q.width - 3.613133
    if Q.width >= 0.01323868:
        z += -146.344 * Q.width + 4.684976
    if Q.e2 >= 0.06344108:
        z += 115.1945 * Q.e2 - 7.308062
    if Q.girth2 < 0.004372139:
        z += 192.385 * Q.girth2 - 0.841134
    if 0.008678045 <= Q.girth2 < 0.01882765:
        z += -1521.225 * Q.girth2 + 13.20126
    if Q.girth2 >= 0.01882765:
        z += -1562.064 * Q.girth2 + 13.97017
    if 0.1492731 <= Q.sj2_dr < 0.1872617:
        z += -2.718736 * Q.sj2_dr + 0.4058342
    if Q.sj2_dr >= 0.1872617:
        z += 24.93019 * Q.sj2_dr - 4.771751
    if Q.mean_eta < -0.004664942:
        z += -27.39743 * Q.mean_eta - 0.1278074
    if Q.mean_eta >= 0.01772426:
        z += 40.18314 * Q.mean_eta - 0.7122162
    if Q.max_dr >= 0.1027585:
        z += 13.27487 * Q.max_dr - 1.364106
    if Q.mass >= 64.61873:
        z += -0.09916532 * Q.mass + 6.407937
    if Q.sd_mass >= 62.73432:
        z += 0.03522671 * Q.sd_mass - 2.209923
    if Q.z_dr_0_0p05 < 0.05226226:
        z += -10.61041 * Q.z_dr_0_0p05 + 0.5545242
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.3315868 * (Q.mass_over_sum_pt - 0.0681391) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.mass_over_sum_pt > 0.0681391 and Q.pt_6 > 31.90625:
        z += -0.684516 * (Q.mass_over_sum_pt - 0.0681391) * (Q.pt_6 - 31.90625)
    if Q.girth > 0.04081947 and Q.log_sum_pt > 6.080494:
        z += 78.55073 * (Q.girth - 0.04081947) * (Q.log_sum_pt - 6.080494)
    if Q.e2 > 0.06344108 and Q.sj2_mass1 > 16.86126:
        z += 1.31868 * (Q.e2 - 0.06344108) * (Q.sj2_mass1 - 16.86126)
    if Q.sj2_dr > 0.1872617 and Q.sj2_mass1 > 2.250113:
        z += -0.4354387 * (Q.sj2_dr - 0.1872617) * (Q.sj2_mass1 - 2.250113)
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj2_dr < 0.2179769:
        z += -1547.797 * (Q.mass_over_sum_pt - 0.0681391) * (0.2179769 - Q.sj2_dr)
    if Q.width > 0.007520088 and Q.sj3_pairmin_over_m > 0.07708997:
        z += 473.1499 * (Q.width - 0.007520088) * (Q.sj3_pairmin_over_m - 0.07708997)
    if Q.centroid_offset > 0.01096064 and Q.abseta_0 < 0.07861328:
        z += 554.0746 * (Q.centroid_offset - 0.01096064) * (0.07861328 - Q.abseta_0)
    if Q.lam2 > 0.001130645 and Q.pt_6 < 56.53125:
        z += 25.55328 * (Q.lam2 - 0.001130645) * (56.53125 - Q.pt_6)
    if Q.e2 > 0.06344108 and Q.z_top5 > 0.7773372:
        z += -570.4618 * (Q.e2 - 0.06344108) * (Q.z_top5 - 0.7773372)
    if Q.sj2_dr > 0.1872617 and Q.dr_3 < 0.05268713:
        z += -358.6294 * (Q.sj2_dr - 0.1872617) * (0.05268713 - Q.dr_3)
    if Q.tau1 > 0.05356915 and Q.pt_5 > 59.125:
        z += 0.3220304 * (Q.tau1 - 0.05356915) * (Q.pt_5 - 59.125)
    if Q.girth > 0.07608178 and Q.eta_0 > 0.07952881:
        z += 111.9878 * (Q.girth - 0.07608178) * (Q.eta_0 - 0.07952881)
    if Q.lam2 > 0.001130645 and Q.z_6 > 0.05096142:
        z += 9333.847 * (Q.lam2 - 0.001130645) * (Q.z_6 - 0.05096142)
    if Q.sj2_dr > 0.2687922 and Q.z_6 < 0.02160287:
        z += -1494.475 * (Q.sj2_dr - 0.2687922) * (0.02160287 - Q.z_6)
    if Q.max_dr > 0.1027585 and Q.eta_1 < -0.05963135:
        z += -35.65911 * (Q.max_dr - 0.1027585) * (-0.05963135 - Q.eta_1)
    if Q.max_dr > 0.1027585 and Q.zdr_1 < 0.01084112:
        z += -670.3936 * (Q.max_dr - 0.1027585) * (0.01084112 - Q.zdr_1)
    if Q.max_dr > 0.1027585 and Q.abseta_0 > 0.1057739:
        z += -59.45479 * (Q.max_dr - 0.1027585) * (Q.abseta_0 - 0.1057739)
    if Q.max_dr > 0.1027585 and Q.D2_b2 < 0.05744392:
        z += -113.3493 * (Q.max_dr - 0.1027585) * (0.05744392 - Q.D2_b2)
    return max(0.0, z)


def neuron_4(Q):
    z = -3.756036
    if Q.N2 < 0.2233283:
        z += -49.18458 * Q.N2 + 10.98431
    if Q.lam2 < 0.000537286:
        z += 4615.588 * Q.lam2 - 3.246155
    if 0.000537286 <= Q.lam2 < 0.001130645:
        z += 1291.402 * Q.lam2 - 1.460117
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -54.23 * Q.mass_over_sum_pt + 4.903141
    if Q.girth2 < 0.002635418:
        z += -1276.461 * Q.girth2 - 6.549603
    if 0.002635418 <= Q.girth2 < 0.003562611:
        z += 1062.494 * Q.girth2 - 12.71373
    if 0.003562611 <= Q.girth2 < 0.008678045:
        z += 1282.567 * Q.girth2 - 13.49776
    if 0.008678045 <= Q.girth2 < 0.01882765:
        z += 564.2589 * Q.girth2 - 7.264251
    if Q.girth2 >= 0.01882765:
        z += 220.0728 * Q.girth2 - 0.7840339
    if 0.009668065 <= Q.e2 < 0.04447357:
        z += 49.06947 * Q.e2 - 0.4744068
    if Q.e2 >= 0.04447357:
        z += 22.82057 * Q.e2 + 0.6929756
    if Q.sum_pt < 739.5:
        z += 0.008139063 * Q.sum_pt - 6.018837
    if 0.169029 <= Q.sj3_dr_max < 0.233678:
        z += 25.45607 * Q.sj3_dr_max - 4.302815
    if 0.233678 <= Q.sj3_dr_max < 0.3456459:
        z += -20.90219 * Q.sj3_dr_max + 6.53009
    if Q.sj3_dr_max >= 0.3456459:
        z += -7.772356 * Q.sj3_dr_max + 1.991817
    if Q.e2_sq < 0.01165737:
        z += -4336.364 * Q.e2_sq + 52.6231
    if 0.01165737 <= Q.e2_sq < 0.01716248:
        z += -376.4643 * Q.e2_sq + 6.461062
    if Q.girth < 0.05464922:
        z += -25.12641 * Q.girth + 3.129588
    if 0.05464922 <= Q.girth < 0.1019409:
        z += -71.80706 * Q.girth + 5.68065
    if 0.1019409 <= Q.girth < 0.1245537:
        z += -46.11241 * Q.girth + 3.061313
    if Q.girth >= 0.1245537:
        z += -20.986 * Q.girth - 0.06827551
    if Q.mass < 69.61135:
        z += -0.02509554 * Q.mass + 2.240991
    if 69.61135 <= Q.mass < 76.6557:
        z += -0.07013521 * Q.mass + 5.376264
    if Q.C2 < 0.05119235:
        z += 27.99898 * Q.C2 - 1.433333
    if Q.girth2_top5 < 0.008329695:
        z += -224.5054 * Q.girth2_top5 + 1.870062
    if 0.1117619 <= Q.max_dr < 0.2215867:
        z += 9.315195 * Q.max_dr - 1.041083
    if Q.max_dr >= 0.2215867:
        z += 21.91183 * Q.max_dr - 3.832331
    if Q.D2 < 0.875672:
        z += -3.833629 * Q.D2 + 3.513055
    if 0.875672 <= Q.D2 < 1.002471:
        z += -1.230715 * Q.D2 + 1.233755
    if Q.width < 0.01323868:
        z += 367.9837 * Q.width - 4.871617
    if Q.C2_b2 < 0.009032972:
        z += -369.7144 * Q.C2_b2 + 3.33962
    if 0.006823012 <= Q.mean_eta < 0.02644207:
        z += 11.54065 * Q.mean_eta - 0.07874199
    if Q.mean_eta >= 0.02644207:
        z += -10.87523 * Q.mean_eta + 0.5139802
    if Q.tau1 >= 0.1136369:
        z += 29.06467 * Q.tau1 - 3.302819
    if 0.1667615 <= Q.sd_rg < 0.2037854:
        z += -5.787182 * Q.sd_rg + 0.9650792
    if 0.2037854 <= Q.sd_rg < 0.324646:
        z += -11.57216 * Q.sd_rg + 2.143972
    if Q.sd_rg >= 0.324646:
        z += 15.79579 * Q.sd_rg - 6.740924
    if Q.e3 < 1.340118e-05:
        z += 65979.57 * Q.e3 - 0.8842044
    if Q.sd_mass < 49.91626:
        z += 0.03061135 * Q.sd_mass - 1.528004
    if Q.lam1 < 0.002464291:
        z += 1000.436 * Q.lam1 - 2.465366
    if Q.mass_over_sum_pt_sq < 0.0116609:
        z += 3192.524 * Q.mass_over_sum_pt_sq - 37.22771
    if Q.mass_top5 < 37.76455:
        z += 0.01194098 * Q.mass_top5 - 0.4509456
    if Q.zdr_0 < 0.03981924:
        z += -27.44472 * Q.zdr_0 + 1.092828
    if Q.N2 < 0.2233283 and Q.mass < 60.63098:
        z += -0.4507397 * (0.2233283 - Q.N2) * (60.63098 - Q.mass)
    if Q.N2 < 0.2233283 and Q.e2_sq > 0.01165737:
        z += -2894.703 * (0.2233283 - Q.N2) * (Q.e2_sq - 0.01165737)
    if Q.N2 < 0.2233283 and Q.mean_phi < -0.009352575:
        z += 91.03735 * (0.2233283 - Q.N2) * (-0.009352575 - Q.mean_phi)
    if Q.N2 < 0.2233283 and Q.eccentricity > 0.7117266:
        z += -74.98021 * (0.2233283 - Q.N2) * (Q.eccentricity - 0.7117266)
    if Q.sum_pt < 739.5 and Q.n_dr_0p05_0p1 > 0.0:
        z += 0.0002971498 * (739.5 - Q.sum_pt) * (Q.n_dr_0p05_0p1 - 0.0)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1626038:
        z += -34.81215 * (0.2233283 - Q.N2) * (0.1626038 - Q.abseta_7)
    if Q.tau1 > 0.07283629 and Q.sj3_dr_min < 0.2089872:
        z += 202.6969 * (Q.tau1 - 0.07283629) * (0.2089872 - Q.sj3_dr_min)
    if Q.C2 < 0.05119235 and Q.mean_eta > 0.02644207:
        z += -1527.886 * (0.05119235 - Q.C2) * (Q.mean_eta - 0.02644207)
    if Q.mass < 76.6557 and Q.D2 < 0.875672:
        z += -0.1824756 * (76.6557 - Q.mass) * (0.875672 - Q.D2)
    if Q.e2_sq < 0.01165737 and Q.D2 < 0.875672:
        z += 429.5417 * (0.01165737 - Q.e2_sq) * (0.875672 - Q.D2)
    if Q.D2 < 1.002471 and Q.pt_7 < 53.4375:
        z += -0.1174116 * (1.002471 - Q.D2) * (53.4375 - Q.pt_7)
    if Q.girth2_top2 < 0.007639643 and Q.mean_eta < -0.01790907:
        z += 6931.434 * (0.007639643 - Q.girth2_top2) * (-0.01790907 - Q.mean_eta)
    if Q.lam2 < 0.000537286 and Q.mean_phi < -0.01753483:
        z += -53600.66 * (0.000537286 - Q.lam2) * (-0.01753483 - Q.mean_phi)
    if Q.sj3_dr_max > 0.233678 and Q.pt_0 < 130.375:
        z += 0.137824 * (Q.sj3_dr_max - 0.233678) * (130.375 - Q.pt_0)
    if Q.sd_mass > 38.43971 and Q.pt_0 < 153.375:
        z += -0.0001982139 * (Q.sd_mass - 38.43971) * (153.375 - Q.pt_0)
    if Q.sj3_dr_max > 0.3456459 and Q.pt_0 < 118.1875:
        z += -0.2221354 * (Q.sj3_dr_max - 0.3456459) * (118.1875 - Q.pt_0)
    if Q.N2 < 0.2233283 and Q.mean_phi < -0.02594505:
        z += 304.1191 * (0.2233283 - Q.N2) * (-0.02594505 - Q.mean_phi)
    if Q.sd_rg > 0.2037854 and Q.pt_0 < 176.25:
        z += 0.152293 * (Q.sd_rg - 0.2037854) * (176.25 - Q.pt_0)
    if Q.sd_rg > 0.1667615 and Q.z_1st < 0.3331409:
        z += -55.67019 * (Q.sd_rg - 0.1667615) * (0.3331409 - Q.z_1st)
    if Q.zdr_0 < 0.03981924 and Q.ptdr0_7 > 10.31097:
        z += -13.46665 * (0.03981924 - Q.zdr_0) * (Q.ptdr0_7 - 10.31097)
    if Q.D2 < 1.002471 and Q.absphi_5 < 0.1132812:
        z += -5.807492 * (1.002471 - Q.D2) * (0.1132812 - Q.absphi_5)
    if Q.N2 < 0.2233283 and Q.abseta_0 > 0.02487183:
        z += 50.53632 * (0.2233283 - Q.N2) * (Q.abseta_0 - 0.02487183)
    if Q.centroid_offset < 0.01837778 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -162.2991 * (0.01837778 - Q.centroid_offset) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.lam2 < 0.000537286 and Q.absphi_6 < 0.1591797:
        z += -6525.839 * (0.000537286 - Q.lam2) * (0.1591797 - Q.absphi_6)
    if Q.girth2 < 0.002635418 and Q.dr1_7 > 0.1343804:
        z += -16779.31 * (0.002635418 - Q.girth2) * (Q.dr1_7 - 0.1343804)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.6810732
    if Q.z_7 < 0.02807091:
        z += -166.0449 * Q.z_7 + 6.001725
    if 0.02807091 <= Q.z_7 < 0.04939969:
        z += -51.72463 * Q.z_7 + 2.792652
    if 0.04939969 <= Q.z_7 < 0.0586137:
        z += -16.13639 * Q.z_7 + 1.034603
    if 0.0586137 <= Q.z_7 < 0.07148865:
        z += -6.896348 * Q.z_7 + 0.4930106
    if 6.267538 <= Q.log_sum_pt < 6.701242:
        z += 2.704687 * Q.log_sum_pt - 16.95173
    if 6.701242 <= Q.log_sum_pt < 6.766778:
        z += -5.708393 * Q.log_sum_pt + 39.42636
    if 6.766778 <= Q.log_sum_pt < 6.896095:
        z += -17.61452 * Q.log_sum_pt + 119.9925
    if Q.log_sum_pt >= 6.896095:
        z += -9.404694 * Q.log_sum_pt + 63.37674
    if Q.e2_sq < 0.005284669:
        z += -30.02537 * Q.e2_sq + 0.1586741
    if Q.z_6 < 0.02160287:
        z += -123.2294 * Q.z_6 + 3.969975
    if 0.02160287 <= Q.z_6 < 0.02886576:
        z += -63.88277 * Q.z_6 + 2.687917
    if 0.02886576 <= Q.z_6 < 0.04737278:
        z += -45.59848 * Q.z_6 + 2.160127
    if Q.girth2 < 9.121392e-05:
        z += -7823.482 * Q.girth2 + 2.787642
    if 9.121392e-05 <= Q.girth2 < 0.001653836:
        z += -957.4017 * Q.girth2 + 2.16136
    if 0.001653836 <= Q.girth2 < 0.002635418:
        z += -588.8195 * Q.girth2 + 1.551785
    if Q.pt_7 < 23.21641:
        z += 0.1032044 * Q.pt_7 - 3.834687
    if 23.21641 <= Q.pt_7 < 37.15625:
        z += 0.04606372 * Q.pt_7 - 2.508087
    if Q.pt_7 >= 37.15625:
        z += -0.05714064 * Q.pt_7 + 1.3266
    if Q.zdr_0 < 0.02383244:
        z += 31.66846 * Q.zdr_0 - 0.7547366
    if Q.sj3_dr_max < 0.1070199:
        z += 7.032963 * Q.sj3_dr_max - 0.6268204
    if 0.1070199 <= Q.sj3_dr_max < 0.1879486:
        z += -3.725623 * Q.sj3_dr_max + 0.5245628
    if 0.1879486 <= Q.sj3_dr_max < 0.3012016:
        z += 1.551064 * Q.sj3_dr_max - 0.4671829
    if Q.mass < 60.63098:
        z += -0.004272356 * Q.mass + 0.2590371
    if Q.girth2_top3 < 0.002915531:
        z += 173.4646 * Q.girth2_top3 - 0.5057413
    if Q.mass_over_sum_pt < 0.09041383:
        z += 20.31761 * Q.mass_over_sum_pt - 1.836993
    if Q.tau1 < 0.09538712:
        z += -20.00017 * Q.tau1 + 1.907759
    if Q.z_dr_0p1_0p2 < 0.07870506:
        z += -3.133275 * Q.z_dr_0p1_0p2 + 0.2466046
    if Q.sum_pt_top5 >= 716.8828:
        z += 0.008852934 * Q.sum_pt_top5 - 6.346516
    if Q.max_pair_mass >= 21.83614:
        z += -0.01673296 * Q.max_pair_mass + 0.3653833
    if Q.C3 < 0.009518026:
        z += 167.2192 * Q.C3 - 1.591597
    if Q.z_7 < 0.04939969 and Q.mass_top5 < 68.43422:
        z += 0.4949835 * (0.04939969 - Q.z_7) * (68.43422 - Q.mass_top5)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -48.72194 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.e2_sq < 0.005284669 and Q.centroid_offset < 0.01437952:
        z += 7066.943 * (0.005284669 - Q.e2_sq) * (0.01437952 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.centroid_offset < 0.03117077:
        z += -1029.147 * (0.07148865 - Q.z_7) * (0.03117077 - Q.centroid_offset)
    if Q.log_sum_pt > 6.896095 and Q.sj3_pair_mass_max > 17.83163:
        z += -0.3378825 * (Q.log_sum_pt - 6.896095) * (Q.sj3_pair_mass_max - 17.83163)
    if Q.z_7 < 0.07148865 and Q.mass_top3 < 42.18339:
        z += 0.1404893 * (0.07148865 - Q.z_7) * (42.18339 - Q.mass_top3)
    if Q.girth2 < 0.001653836 and Q.centroid_offset < 0.02355416:
        z += 82333.38 * (0.001653836 - Q.girth2) * (0.02355416 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.C2_b2 > 0.001109927:
        z += -407.5599 * (0.07148865 - Q.z_7) * (Q.C2_b2 - 0.001109927)
    if Q.mass < 60.63098 and Q.D2 < 3.885568:
        z += 0.001576286 * (60.63098 - Q.mass) * (3.885568 - Q.D2)
    if Q.z_6 < 0.02886576 and Q.n_pt_above_5 < 8.0:
        z += -26.41653 * (0.02886576 - Q.z_6) * (8.0 - Q.n_pt_above_5)
    if Q.sj3_dr_max < 0.3012016 and Q.mass_top2 > 22.84498:
        z += 0.1860015 * (0.3012016 - Q.sj3_dr_max) * (Q.mass_top2 - 22.84498)
    if Q.log_sum_pt > 6.701242 and Q.abseta_4 < 0.01402779:
        z += 212.3535 * (Q.log_sum_pt - 6.701242) * (0.01402779 - Q.abseta_4)
    if Q.log_sum_pt > 6.267538 and Q.abseta_4 < 0.03601074:
        z += -19.61707 * (Q.log_sum_pt - 6.267538) * (0.03601074 - Q.abseta_4)
    if Q.z_7 < 0.02807091 and Q.pt_5 > 24.57812:
        z += 2.112263 * (0.02807091 - Q.z_7) * (Q.pt_5 - 24.57812)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi2 < 0.0002302115:
        z += 29369.2 * (Q.log_sum_pt - 6.701242) * (0.0002302115 - Q.mean_phi2)
    if Q.girth2 < 0.001653836 and Q.absphi_0 < 0.01452637:
        z += -17339.04 * (0.001653836 - Q.girth2) * (0.01452637 - Q.absphi_0)
    if Q.zdr_0 < 0.02383244 and Q.z_5 > 0.02818362:
        z += 72.37946 * (0.02383244 - Q.zdr_0) * (Q.z_5 - 0.02818362)
    if Q.log_sum_pt > 6.896095 and Q.mean_phi2 < 0.0002302115:
        z += -52920.06 * (Q.log_sum_pt - 6.896095) * (0.0002302115 - Q.mean_phi2)
    if Q.tau1 < 0.09538712 and Q.planar_flow < 0.4926918:
        z += 10.78901 * (0.09538712 - Q.tau1) * (0.4926918 - Q.planar_flow)
    if Q.girth2_top3 < 0.002915531 and Q.tau3 < 0.006646257:
        z += -27376.16 * (0.002915531 - Q.girth2_top3) * (0.006646257 - Q.tau3)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta2 < 9.030369e-05:
        z += 191823.0 * (Q.log_sum_pt - 6.701242) * (9.030369e-05 - Q.mean_eta2)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta2 < 0.00424745:
        z += -1990.0 * (Q.log_sum_pt - 6.701242) * (0.00424745 - Q.mean_eta2)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi2 > 0.0007823696:
        z += 784.457 * (Q.log_sum_pt - 6.701242) * (Q.mean_phi2 - 0.0007823696)
    if Q.zdr_0 < 0.02383244 and Q.abseta_0 < 0.02090454:
        z += -990.9667 * (0.02383244 - Q.zdr_0) * (0.02090454 - Q.abseta_0)
    if Q.z_5 < 0.02818362 and Q.abseta_7 < 0.1626038:
        z += 391.4657 * (0.02818362 - Q.z_5) * (0.1626038 - Q.abseta_7)
    if Q.sum_pt_top5 > 716.8828 and Q.mean_eta2 < 9.030369e-05:
        z += -153.5316 * (Q.sum_pt_top5 - 716.8828) * (9.030369e-05 - Q.mean_eta2)
    if Q.pt_7 < 37.15625 and Q.mratio_min_012 < 0.1847499:
        z += 0.136735 * (37.15625 - Q.pt_7) * (0.1847499 - Q.mratio_min_012)
    if Q.z_6 < 0.02160287 and Q.abseta_6 < 0.1583252:
        z += 702.7815 * (0.02160287 - Q.z_6) * (0.1583252 - Q.abseta_6)
    if Q.pt_7 > 23.21641 and Q.absphi_1 < 0.07141113:
        z += -0.1507575 * (Q.pt_7 - 23.21641) * (0.07141113 - Q.absphi_1)
    if Q.max_pair_mass > 21.83614 and Q.dr_max_012 < 0.2366102:
        z += -0.757953 * (Q.max_pair_mass - 21.83614) * (0.2366102 - Q.dr_max_012)
    if Q.mass_over_sum_pt < 0.09041383 and Q.ptdr0_3 > 12.34207:
        z += -4.792266 * (0.09041383 - Q.mass_over_sum_pt) * (Q.ptdr0_3 - 12.34207)
    if Q.log_sum_pt > 6.267538 and Q.ptdr0_5 < 7.737156:
        z += 0.1287581 * (Q.log_sum_pt - 6.267538) * (7.737156 - Q.ptdr0_5)
    if Q.log_sum_pt > 6.701242 and Q.D2_b2 < 4.721224:
        z += -1.064281 * (Q.log_sum_pt - 6.701242) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.896095 and Q.dr_max_012 < 0.03086731:
        z += -216.2808 * (Q.log_sum_pt - 6.896095) * (0.03086731 - Q.dr_max_012)
    if Q.log_sum_pt > 6.701242 and Q.dr_max_012 < 0.06112084:
        z += 116.9721 * (Q.log_sum_pt - 6.701242) * (0.06112084 - Q.dr_max_012)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi > 0.0006973656:
        z += -649.5932 * (Q.log_sum_pt - 6.701242) * (Q.mean_phi - 0.0006973656)
    if Q.log_sum_pt > 6.896095 and Q.mean_phi > 0.01746433:
        z += 2233.456 * (Q.log_sum_pt - 6.896095) * (Q.mean_phi - 0.01746433)
    if Q.girth2_top3 < 0.002915531 and Q.phi_1 < 0.02186584:
        z += -4225.634 * (0.002915531 - Q.girth2_top3) * (0.02186584 - Q.phi_1)
    if Q.log_sum_pt > 6.267538 and Q.C3 < 0.0113471:
        z += 472.9601 * (Q.log_sum_pt - 6.267538) * (0.0113471 - Q.C3)
    if Q.pt_7 < 37.15625 and Q.C3 < 0.02079512:
        z += -2.766457 * (37.15625 - Q.pt_7) * (0.02079512 - Q.C3)
    if Q.log_sum_pt > 6.766778 and Q.C3 < 0.05481757:
        z += -49.60653 * (Q.log_sum_pt - 6.766778) * (0.05481757 - Q.C3)
    if Q.mass_over_sum_pt < 0.09041383 and Q.dr0_3 > 0.06671811:
        z += 98.99336 * (0.09041383 - Q.mass_over_sum_pt) * (Q.dr0_3 - 0.06671811)
    if Q.log_sum_pt > 6.701242 and Q.zdr_3 < 0.001736329:
        z += 4782.636 * (Q.log_sum_pt - 6.701242) * (0.001736329 - Q.zdr_3)
    if Q.log_sum_pt > 6.896095 and Q.zdr_3 < 0.002727284:
        z += -7382.188 * (Q.log_sum_pt - 6.896095) * (0.002727284 - Q.zdr_3)
    if Q.log_sum_pt > 6.267538 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += 0.9132279 * (Q.log_sum_pt - 6.267538) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.z_6 < 0.04737278 and Q.zdr_3 < 0.001736329:
        z += -19008.57 * (0.04737278 - Q.z_6) * (0.001736329 - Q.zdr_3)
    if Q.pt_7 < 37.15625 and Q.zdr_3 < 0.01257746:
        z += 1.5622 * (37.15625 - Q.pt_7) * (0.01257746 - Q.zdr_3)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.4216402
    if 0.00809236 <= Q.centroid_offset < 0.01837778:
        z += 44.92741 * Q.centroid_offset - 0.3635688
    if Q.centroid_offset >= 0.01837778:
        z += -9.09388 * Q.centroid_offset + 0.6292226
    if Q.width < 0.01323868:
        z += 375.3691 * Q.width - 4.969389
    if Q.mass < 45.7571:
        z += -0.03840256 * Q.mass + 2.021499
    if 45.7571 <= Q.mass < 60.63098:
        z += -0.01777002 * Q.mass + 1.077414
    if Q.pt_6 < 19.46875:
        z += 0.005110499 * Q.pt_6 - 4.240247
    if 19.46875 <= Q.pt_6 < 29.90625:
        z += 0.2171659 * Q.pt_6 - 8.368701
    if 29.90625 <= Q.pt_6 < 39.75:
        z += 0.1813875 * Q.pt_6 - 7.298702
    if 39.75 <= Q.pt_6 < 41.21875:
        z += 0.06028873 * Q.pt_6 - 2.485026
    if Q.lam2 < 0.001130645:
        z += 832.7654 * Q.lam2 - 0.1552628
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -345.2095 * Q.lam2 + 1.176608
    if Q.tau1 < 0.1136369:
        z += -24.68984 * Q.tau1 + 2.805677
    if 0.02400746 <= Q.sj3_dr_min < 0.1278212:
        z += -15.72501 * Q.sj3_dr_min + 0.3775174
    if Q.sj3_dr_min >= 0.1278212:
        z += -4.020143 * Q.sj3_dr_min - 1.118612
    if Q.lam1 < 0.00595415:
        z += -440.8562 * Q.lam1 + 5.047475
    if 0.00595415 <= Q.lam1 < 0.00733008:
        z += -500.7831 * Q.lam1 + 5.404288
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += -370.9113 * Q.lam1 + 4.452317
    if Q.sj3_pair_mass_min >= 4.501727:
        z += -0.0283319 * Q.sj3_pair_mass_min + 0.1275425
    if Q.sum_pt < 488.9312:
        z += -0.01008963 * Q.sum_pt + 4.834121
    if 488.9312 <= Q.sum_pt < 615.875:
        z += -0.0006272007 * Q.sum_pt + 0.2076431
    if 615.875 <= Q.sum_pt < 763.825:
        z += 0.001207395 * Q.sum_pt - 0.9222388
    if Q.e2_sq < 0.0030133:
        z += -207.1469 * Q.e2_sq + 0.6241959
    if Q.sj3_dr13 >= 0.181053:
        z += -1.60081 * Q.sj3_dr13 + 0.2898314
    if Q.eccentricity >= 0.927072:
        z += -2.808521 * Q.eccentricity + 2.603701
    if Q.sj3_dr_max < 0.1789613:
        z += -22.82952 * Q.sj3_dr_max + 3.049785
    if 0.1789613 <= Q.sj3_dr_max < 0.1879486:
        z += 8.473617 * Q.sj3_dr_max - 2.552267
    if 0.1879486 <= Q.sj3_dr_max < 0.3012016:
        z += 24.06923 * Q.sj3_dr_max - 5.48344
    if Q.sj3_dr_max >= 0.3012016:
        z += 15.59562 * Q.sj3_dr_max - 2.931174
    if Q.max_dr < 0.1452311:
        z += 29.28081 * Q.max_dr - 4.252485
    if Q.mean_eta >= 0.02644207:
        z += -62.92281 * Q.mean_eta + 1.66381
    if Q.e3 < 0.0005116989:
        z += -2162.871 * Q.e3 + 1.106738
    if Q.girth2 < 0.008678045:
        z += 1056.961 * Q.girth2 - 9.172359
    if Q.C2_b2 < 0.02415398:
        z += -34.62838 * Q.C2_b2 + 0.8364134
    if Q.sj2_dr < 0.1872617:
        z += -4.253476 * Q.sj2_dr + 0.7965133
    if Q.sj3_dr23 >= 0.2207152:
        z += -1.951371 * Q.sj3_dr23 + 0.4306973
    if Q.z_6 < 0.03932388:
        z += -58.62083 * Q.z_6 + 2.305198
    if Q.mass_over_sum_pt < 0.09041383:
        z += -33.73593 * Q.mass_over_sum_pt + 3.050195
    if Q.centroid_offset > 0.00809236 and Q.sj3_pair_mass_min > 6.811308:
        z += 0.7833887 * (Q.centroid_offset - 0.00809236) * (Q.sj3_pair_mass_min - 6.811308)
    if Q.pt_6 < 41.21875 and Q.log_sum_pt < 6.766778:
        z += 0.0454312 * (41.21875 - Q.pt_6) * (6.766778 - Q.log_sum_pt)
    if Q.centroid_offset > 0.00809236 and Q.psi_0p1 > 0.4008925:
        z += 37.95944 * (Q.centroid_offset - 0.00809236) * (Q.psi_0p1 - 0.4008925)
    if Q.centroid_offset > 0.01837778 and Q.C2 < 0.02702951:
        z += 1014.679 * (Q.centroid_offset - 0.01837778) * (0.02702951 - Q.C2)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.008921136:
        z += 8562.303 * (Q.centroid_offset - 0.01837778) * (0.008921136 - Q.mean_phi2)
    if Q.lam1 < 0.01200373 and Q.planar_flow < 0.2534037:
        z += -512.9839 * (0.01200373 - Q.lam1) * (0.2534037 - Q.planar_flow)
    if Q.pt_6 < 41.21875 and Q.z_7 > 0.02320757:
        z += -6.23755 * (41.21875 - Q.pt_6) * (Q.z_7 - 0.02320757)
    if Q.pt_6 < 29.90625 and Q.n_pt_above_10 < 8.0:
        z += 0.01921852 * (29.90625 - Q.pt_6) * (8.0 - Q.n_pt_above_10)
    if Q.mass < 60.63098 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -0.8730217 * (60.63098 - Q.mass) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sum_pt < 615.875 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.00219465 * (615.875 - Q.sum_pt) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.lam2 < 0.003408389 and Q.n_dr_0_0p05 < 3.0:
        z += 83.49632 * (0.003408389 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.max_dr < 0.1452311 and Q.mean_phi > 0.004406178:
        z += -187.7843 * (0.1452311 - Q.max_dr) * (Q.mean_phi - 0.004406178)
    if Q.pt_6 < 41.21875 and Q.dr1_7 < 0.1343804:
        z += 0.1349861 * (41.21875 - Q.pt_6) * (0.1343804 - Q.dr1_7)
    if Q.lam1 < 0.01200373 and Q.mean_eta > 0.02644207:
        z += 6812.44 * (0.01200373 - Q.lam1) * (Q.mean_eta - 0.02644207)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.00433128:
        z += -3123.248 * (Q.centroid_offset - 0.01837778) * (0.00433128 - Q.mean_phi2)
    if Q.sum_pt < 615.875 and Q.z_3 < 0.05101919:
        z += 9.69158 * (615.875 - Q.sum_pt) * (0.05101919 - Q.z_3)
    if Q.pt_6 < 41.21875 and Q.M3 > 0.0782171:
        z += 2.252525 * (41.21875 - Q.pt_6) * (Q.M3 - 0.0782171)
    if Q.lam1 < 0.00733008 and Q.n_dr_0p2_0p4 < 1.0:
        z += 89.47548 * (0.00733008 - Q.lam1) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.centroid_offset > 0.01837778 and Q.eta_0 < 0.02980347:
        z += -79.29404 * (Q.centroid_offset - 0.01837778) * (0.02980347 - Q.eta_0)
    if Q.mean_eta > 0.02644207 and Q.M3 > 0.06688759:
        z += -1408.96 * (Q.mean_eta - 0.02644207) * (Q.M3 - 0.06688759)
    if Q.sj3_pair_mass_min > 4.501727 and Q.n_dr_0p2_0p4 > 1.0:
        z += -0.01511467 * (Q.sj3_pair_mass_min - 4.501727) * (Q.n_dr_0p2_0p4 - 1.0)
    if Q.sj3_dr_max > 0.1879486 and Q.abseta_2 < 0.03250122:
        z += -100.4803 * (Q.sj3_dr_max - 0.1879486) * (0.03250122 - Q.abseta_2)
    if Q.sj3_dr13 > 0.181053 and Q.ptdr0_7 > 10.31097:
        z += -0.4471924 * (Q.sj3_dr13 - 0.181053) * (Q.ptdr0_7 - 10.31097)
    if Q.centroid_offset > 0.01837778 and Q.z_2 < 0.1818564:
        z += -114.2813 * (Q.centroid_offset - 0.01837778) * (0.1818564 - Q.z_2)
    if Q.lam1 < 0.00733008 and Q.mass_top5 > 68.43422:
        z += 121.3379 * (0.00733008 - Q.lam1) * (Q.mass_top5 - 68.43422)
    if Q.centroid_offset > 0.01837778 and Q.sum_pt_top5 > 658.125:
        z += 1.254162 * (Q.centroid_offset - 0.01837778) * (Q.sum_pt_top5 - 658.125)
    if Q.pt_6 < 41.21875 and Q.sum_pt < 988.4078:
        z += 0.0008722185 * (41.21875 - Q.pt_6) * (988.4078 - Q.sum_pt)
    if Q.sum_pt < 763.825 and Q.z_7 < 0.0753896:
        z += 0.1236074 * (763.825 - Q.sum_pt) * (0.0753896 - Q.z_7)
    return max(0.0, z)


def neuron_7(Q):
    z = 9.734622
    if Q.planar_flow < 0.1950135:
        z += 2.324861 * Q.planar_flow - 0.4533793
    if Q.girth2_top2 < 0.001056655:
        z += -580.6614 * Q.girth2_top2 + 0.6135589
    if Q.girth2 < 0.0009641429:
        z += 1031.073 * Q.girth2 - 0.6151086
    if 0.0009641429 <= Q.girth2 < 0.001653836:
        z += -145.8526 * Q.girth2 + 0.5196159
    if 0.001653836 <= Q.girth2 < 0.003562611:
        z += -226.6408 * Q.girth2 + 0.6532265
    if 0.003562611 <= Q.girth2 < 0.004372139:
        z += -80.78828 * Q.girth2 + 0.1336106
    if 0.004372139 <= Q.girth2 < 0.007520088:
        z += -517.2625 * Q.girth2 + 2.041937
    if 0.007520088 <= Q.girth2 < 0.01323868:
        z += -963.2471 * Q.girth2 + 5.39578
    if Q.girth2 >= 0.01323868:
        z += 318.2751 * Q.girth2 - 11.56988
    if 0.01109984 <= Q.mass_over_sum_pt < 0.07269073:
        z += -52.05693 * Q.mass_over_sum_pt + 0.5778237
    if 0.07269073 <= Q.mass_over_sum_pt < 0.07992374:
        z += 41.68348 * Q.mass_over_sum_pt - 6.236236
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 106.1888 * Q.mass_over_sum_pt - 11.39174
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 255.7693 * Q.mass_over_sum_pt - 24.06893
    if 0.09041383 <= Q.mass_over_sum_pt < 0.1079857:
        z += -21.69345 * Q.mass_over_sum_pt + 1.017539
    if Q.mass_over_sum_pt >= 0.1079857:
        z += -254.9658 * Q.mass_over_sum_pt + 26.20761
    if Q.tau1 < 0.05356915:
        z += -39.16024 * Q.tau1 + 2.097781
    if Q.girth < 0.04081947:
        z += 111.6562 * Q.girth - 7.971586
    if 0.04081947 <= Q.girth < 0.08723651:
        z += 73.54708 * Q.girth - 6.415991
    if 36.22941 <= Q.mass < 69.61135:
        z += 0.04343187 * Q.mass - 1.573511
    if 69.61135 <= Q.mass < 76.6557:
        z += 0.008279555 * Q.mass + 0.8734893
    if Q.mass >= 76.6557:
        z += -0.1918378 * Q.mass + 16.21363
    if Q.z_dr_0p1_0p2 < 0.1585582:
        z += -1.444515 * Q.z_dr_0p1_0p2 + 0.2290397
    if Q.width < 0.005590289:
        z += 858.4153 * Q.width - 5.73376
    if 0.005590289 <= Q.width < 0.006679471:
        z += 158.5722 * Q.width - 1.821435
    if 0.006679471 <= Q.width < 0.008678045:
        z += -699.8431 * Q.width + 3.912325
    if Q.width >= 0.008678045:
        z += -386.6017 * Q.width + 1.194002
    if Q.centroid_offset < 0.02076709:
        z += 19.54726 * Q.centroid_offset - 0.4059398
    if Q.centroid_offset >= 0.03776099:
        z += -65.49308 * Q.centroid_offset + 2.473084
    if Q.z_7 >= 0.03243272:
        z += 12.21926 * Q.z_7 - 0.3963038
    if 0.03556091 <= Q.e2 < 0.05028464:
        z += 6.487155 * Q.e2 - 0.2306891
    if Q.e2 >= 0.05028464:
        z += -218.1615 * Q.e2 + 11.06569
    if Q.sj2_dr < 0.1294903:
        z += -8.55711 * Q.sj2_dr + 0.6895024
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -5.653334 * Q.sj2_dr + 0.3134916
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 20.87391 * Q.sj2_dr - 3.908884
    if Q.LHA < 0.3033137:
        z += -8.253802 * Q.LHA + 2.503492
    if Q.pt_7 < 29.04219:
        z += 0.03371876 * Q.pt_7 - 0.9792667
    if 0.2037854 <= Q.sd_rg < 0.2787955:
        z += -21.64186 * Q.sd_rg + 4.410294
    if Q.sd_rg >= 0.2787955:
        z += 20.87423 * Q.sd_rg - 7.442999
    if Q.lam1 < 0.008375572:
        z += 213.8434 * Q.lam1 - 1.791061
    if Q.tau21_b2 < 0.009213932:
        z += 1.57749 * Q.tau21_b2 - 0.8000651
    if 0.009213932 <= Q.tau21_b2 < 0.04019753:
        z += 25.3531 * Q.tau21_b2 - 1.019132
    if Q.mass_top5 >= 53.60766:
        z += 0.07523065 * Q.mass_top5 - 4.032939
    if Q.D2_b2 < 0.05744392:
        z += 22.02159 * Q.D2_b2 - 1.265007
    if 0.04889979 <= Q.sj3_dr_max < 0.1426152:
        z += -6.299038 * Q.sj3_dr_max + 0.3080216
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += 16.54108 * Q.sj3_dr_max - 2.949326
    if 0.213399 <= Q.sj3_dr_max < 0.2623172:
        z += 20.706 * Q.sj3_dr_max - 3.838115
    if Q.sj3_dr_max >= 0.2623172:
        z += -0.8329167 * Q.sj3_dr_max + 1.811914
    if Q.lam2 < 0.0003061234:
        z += 1720.982 * Q.lam2 - 0.5268328
    if Q.max_dr < 0.1117619:
        z += 6.362527 * Q.max_dr - 0.1773502
    if 0.1117619 <= Q.max_dr < 0.177305:
        z += -8.143305 * Q.max_dr + 1.443849
    if Q.planar_flow < 0.1950135 and Q.width > 0.007520088:
        z += 1058.757 * (0.1950135 - Q.planar_flow) * (Q.width - 0.007520088)
    if Q.planar_flow < 0.1950135 and Q.sd_mass > 38.43971:
        z += 0.1320864 * (0.1950135 - Q.planar_flow) * (Q.sd_mass - 38.43971)
    if Q.centroid_offset > 0.03117077 and Q.n_pt_above_50 > 4.0:
        z += -18.22661 * (Q.centroid_offset - 0.03117077) * (Q.n_pt_above_50 - 4.0)
    if Q.centroid_offset < 0.02076709 and Q.sum_pt_top3 > 331.25:
        z += 0.05564829 * (0.02076709 - Q.centroid_offset) * (Q.sum_pt_top3 - 331.25)
    if Q.girth2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 556.9947 * (Q.girth2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    if Q.girth2 > 0.01323868 and Q.eccentricity > 0.9458207:
        z += 4202.502 * (Q.girth2 - 0.01323868) * (Q.eccentricity - 0.9458207)
    if Q.centroid_offset > 0.03776099 and Q.pt_4 > 81.375:
        z += -28.95861 * (Q.centroid_offset - 0.03776099) * (Q.pt_4 - 81.375)
    if Q.centroid_offset < 0.02076709 and Q.C2_b2 < 0.004032342:
        z += -17332.56 * (0.02076709 - Q.centroid_offset) * (0.004032342 - Q.C2_b2)
    if Q.girth < 0.08723651 and Q.C2_b2 < 0.004032342:
        z += 3275.841 * (0.08723651 - Q.girth) * (0.004032342 - Q.C2_b2)
    if Q.centroid_offset < 0.02076709 and Q.tau21_b2 < 0.02656143:
        z += 2480.764 * (0.02076709 - Q.centroid_offset) * (0.02656143 - Q.tau21_b2)
    if Q.girth2_top2 < 0.001056655 and Q.tau21_b2 < 0.02656143:
        z += -103219.0 * (0.001056655 - Q.girth2_top2) * (0.02656143 - Q.tau21_b2)
    if Q.pt_7 < 29.04219 and Q.tau21 < 0.2838437:
        z += -0.1649838 * (29.04219 - Q.pt_7) * (0.2838437 - Q.tau21)
    if Q.lam2 < 0.0003061234 and Q.tau21_b2 < 0.04019753:
        z += 152191.7 * (0.0003061234 - Q.lam2) * (0.04019753 - Q.tau21_b2)
    if Q.mass_over_sum_pt > 0.09041383 and Q.pt_6 < 36.8125:
        z += 17.08295 * (Q.mass_over_sum_pt - 0.09041383) * (36.8125 - Q.pt_6)
    if Q.sj3_dr_max > 0.1426152 and Q.pt_6 < 19.46875:
        z += -0.9337042 * (Q.sj3_dr_max - 0.1426152) * (19.46875 - Q.pt_6)
    if Q.girth2 > 0.01323868 and Q.pt_6 < 38.25:
        z += 70.00181 * (Q.girth2 - 0.01323868) * (38.25 - Q.pt_6)
    if Q.mass_over_sum_pt > 0.01109984 and Q.pt_6 < 35.28125:
        z += -0.7837651 * (Q.mass_over_sum_pt - 0.01109984) * (35.28125 - Q.pt_6)
    if Q.width > 0.008678045 and Q.pt_6 < 38.25:
        z += -116.897 * (Q.width - 0.008678045) * (38.25 - Q.pt_6)
    if Q.mass > 76.6557 and Q.zdr_6 > 0.006366792:
        z += -9.32363 * (Q.mass - 76.6557) * (Q.zdr_6 - 0.006366792)
    if Q.width < 0.006679471 and Q.mean_phi < 0.006452173:
        z += 2711.126 * (0.006679471 - Q.width) * (0.006452173 - Q.mean_phi)
    if Q.sd_rg > 0.2037854 and Q.dr_6 < 0.06970457:
        z += 600.0409 * (Q.sd_rg - 0.2037854) * (0.06970457 - Q.dr_6)
    if Q.girth2 > 0.01323868 and Q.dr_6 < 0.06970457:
        z += -27338.57 * (Q.girth2 - 0.01323868) * (0.06970457 - Q.dr_6)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.3585334
    if Q.girth2 < 0.005019719:
        z += -345.8805 * Q.girth2 + 1.630223
    if 0.005019719 <= Q.girth2 < 0.006679471:
        z += 63.86482 * Q.girth2 - 0.4265833
    if Q.tau1 < 0.05356915:
        z += 14.0657 * Q.tau1 - 0.7534875
    if Q.LHA < 0.07658656:
        z += 1.353844 * Q.LHA - 3.817504
    if 0.07658656 <= Q.LHA < 0.1967397:
        z += 30.90903 * Q.LHA - 6.081034
    if Q.girth < 0.06108601:
        z += 72.95944 * Q.girth - 4.456802
    if Q.mass < 49.6681:
        z += 0.02464482 * Q.mass - 1.224061
    if Q.width < 0.002635418:
        z += -1785.539 * Q.width + 5.823515
    if 0.002635418 <= Q.width < 0.003562611:
        z += -1205.653 * Q.width + 4.295273
    if Q.mass_over_sum_pt < 0.03319429:
        z += 21.25696 * Q.mass_over_sum_pt - 0.7056097
    if Q.sj3_dr_max < 0.1070199:
        z += 17.52203 * Q.sj3_dr_max - 2.868948
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 19.88851 * Q.sj3_dr_max - 3.122209
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += 5.102583 * Q.sj3_dr_max - 1.013512
    if Q.e2 < 0.01289969:
        z += -78.67735 * Q.e2 + 1.014913
    if Q.pt_7 >= 34.53125:
        z += -0.02257386 * Q.pt_7 + 0.7795036
    if Q.sum_pt >= 988.4078:
        z += -0.004667774 * Q.sum_pt + 4.613664
    if Q.dr_0 < 0.008355823:
        z += -54.71944 * Q.dr_0 + 0.457226
    if Q.girth2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 17667.0 * (0.006679471 - Q.girth2) * (0.02355416 - Q.centroid_offset)
    if Q.girth2 < 0.005019719 and Q.pt_7 < 43.5:
        z += -10.3879 * (0.005019719 - Q.girth2) * (43.5 - Q.pt_7)
    if Q.LHA < 0.1967397 and Q.mean_phi > -0.0008556753:
        z += 552.8864 * (0.1967397 - Q.LHA) * (Q.mean_phi - -0.0008556753)
    if Q.girth2 < 0.006679471 and Q.phi_0 > -0.04013062:
        z += 1148.092 * (0.006679471 - Q.girth2) * (Q.phi_0 - -0.04013062)
    if Q.girth2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 6470.469 * (0.005019719 - Q.girth2) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.girth < 0.06108601 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -289.3806 * (0.06108601 - Q.girth) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt > 6.701242 and Q.zdr_0 > 0.007798268:
        z += -133.1803 * (Q.log_sum_pt - 6.701242) * (Q.zdr_0 - 0.007798268)
    if Q.mass < 29.6447 and Q.D2_b2 < 0.716559:
        z += -0.1808751 * (29.6447 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.girth < 0.06108601 and Q.lam2 < 0.0001947983:
        z += 97007.2 * (0.06108601 - Q.girth) * (0.0001947983 - Q.lam2)
    if Q.lam2 < 0.0001330621 and Q.D2_b2 < 4.721224:
        z += 1109.402 * (0.0001330621 - Q.lam2) * (4.721224 - Q.D2_b2)
    if Q.tau1 < 0.05356915 and Q.D2_b2 < 4.721224:
        z += -3.150585 * (0.05356915 - Q.tau1) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.701242 and Q.e3 < 2.955458e-05:
        z += -238149.4 * (Q.log_sum_pt - 6.701242) * (2.955458e-05 - Q.e3)
    if Q.mass < 21.78408 and Q.lam2 < 0.0001947983:
        z += -499.398 * (21.78408 - Q.mass) * (0.0001947983 - Q.lam2)
    if Q.mass < 29.6447 and Q.e3 < 1.762929e-06:
        z += 24912.13 * (29.6447 - Q.mass) * (1.762929e-06 - Q.e3)
    if Q.mass < 49.6681 and Q.lam2 < 0.0001330621:
        z += 73.31912 * (49.6681 - Q.mass) * (0.0001330621 - Q.lam2)
    if Q.LHA < 0.1967397 and Q.psi_0p3 < 1.0:
        z += 1050.171 * (0.1967397 - Q.LHA) * (1.0 - Q.psi_0p3)
    if Q.mass < 21.78408 and Q.pt_7 < 48.71875:
        z += 0.002104314 * (21.78408 - Q.mass) * (48.71875 - Q.pt_7)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 < 48.71875:
        z += 0.1658428 * (Q.log_sum_pt - 6.701242) * (48.71875 - Q.pt_7)
    if Q.mass < 21.78408 and Q.D2_b2 < 0.716559:
        z += 0.1883462 * (21.78408 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.sum_pt_top5 > 687.4375 and Q.n_pt_above_10 < 8.0:
        z += -0.006262548 * (Q.sum_pt_top5 - 687.4375) * (8.0 - Q.n_pt_above_10)
    if Q.girth2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -22266.7 * (0.005019719 - Q.girth2) * (Q.centroid_offset - 0.006789738)
    if Q.LHA < 0.1967397 and Q.pt_7 > 15.55391:
        z += 0.3417011 * (0.1967397 - Q.LHA) * (Q.pt_7 - 15.55391)
    if Q.girth < 0.06108601 and Q.width > 0.0003193707:
        z += 6169.654 * (0.06108601 - Q.girth) * (Q.width - 0.0003193707)
    if Q.log_sum_pt > 6.701242 and Q.width < 0.008678045:
        z += -48.0164 * (Q.log_sum_pt - 6.701242) * (0.008678045 - Q.width)
    if Q.sj3_dr_max < 0.1986272 and Q.girth2 < 0.006679471:
        z += 3102.385 * (0.1986272 - Q.sj3_dr_max) * (0.006679471 - Q.girth2)
    if Q.tau1 < 0.05356915 and Q.width < 0.003562611:
        z += 24763.2 * (0.05356915 - Q.tau1) * (0.003562611 - Q.width)
    if Q.log_sum_pt > 6.701242 and Q.centroid_offset < 0.02355416:
        z += -130.8077 * (Q.log_sum_pt - 6.701242) * (0.02355416 - Q.centroid_offset)
    if Q.log_sum_pt > 6.701242 and Q.girth2 < 0.0005611231:
        z += 10312.63 * (Q.log_sum_pt - 6.701242) * (0.0005611231 - Q.girth2)
    if Q.sj3_dr_max < 0.1986272 and Q.girth2_top5 < 0.005691733:
        z += -1137.73 * (0.1986272 - Q.sj3_dr_max) * (0.005691733 - Q.girth2_top5)
    if Q.sj3_dr_max < 0.1986272 and Q.lam2 < 0.0001947983:
        z += 11473.61 * (0.1986272 - Q.sj3_dr_max) * (0.0001947983 - Q.lam2)
    if Q.pt_7 > 34.53125 and Q.pt_1 < 142.5:
        z += 0.0003640475 * (Q.pt_7 - 34.53125) * (142.5 - Q.pt_1)
    if Q.LHA < 0.1967397 and Q.centroid_offset > 0.01837778:
        z += 11568.51 * (0.1967397 - Q.LHA) * (Q.centroid_offset - 0.01837778)
    if Q.log_sum_pt > 6.701242 and Q.n_dr_0p05_0p1 > 2.0:
        z += -1.146526 * (Q.log_sum_pt - 6.701242) * (Q.n_dr_0p05_0p1 - 2.0)
    if Q.girth < 0.06108601 and Q.centroid_offset > 0.006789738:
        z += -2790.253 * (0.06108601 - Q.girth) * (Q.centroid_offset - 0.006789738)
    if Q.width < 0.003562611 and Q.centroid_offset > 0.006789738:
        z += 25917.33 * (0.003562611 - Q.width) * (Q.centroid_offset - 0.006789738)
    if Q.mass < 29.6447 and Q.centroid_offset < 0.02685622:
        z += -5.810242 * (29.6447 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass_over_sum_pt < 0.03319429 and Q.centroid_offset < 0.02685622:
        z += 5238.598 * (0.03319429 - Q.mass_over_sum_pt) * (0.02685622 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.1986272 and Q.width < 0.003562611:
        z += -3400.825 * (0.1986272 - Q.sj3_dr_max) * (0.003562611 - Q.width)
    if Q.e2 < 0.007078158 and Q.girth2_top5 > 5.672047e-05:
        z += -209497.9 * (0.007078158 - Q.e2) * (Q.girth2_top5 - 5.672047e-05)
    if Q.girth < 0.06108601 and Q.girth2 < 0.003562611:
        z += -29773.77 * (0.06108601 - Q.girth) * (0.003562611 - Q.girth2)
    if Q.sj3_dr_max < 0.1426152 and Q.centroid_offset > 0.01627885:
        z += -2252.815 * (0.1426152 - Q.sj3_dr_max) * (Q.centroid_offset - 0.01627885)
    if Q.e2 < 0.01289969 and Q.psi_0p2 > 0.79448:
        z += -519.402 * (0.01289969 - Q.e2) * (Q.psi_0p2 - 0.79448)
    if Q.sum_pt > 988.4078 and Q.dr12 < 0.05062541:
        z += 0.0793651 * (Q.sum_pt - 988.4078) * (0.05062541 - Q.dr12)
    return max(0.0, z)


def neuron_9(Q):
    z = -2.894874
    if Q.girth < 0.05464922:
        z += 19.9562 * Q.girth - 1.090591
    if Q.girth >= 0.0717028:
        z += -27.21248 * Q.girth + 1.951211
    if Q.tau1 < 0.04369778:
        z += -63.42965 * Q.tau1 + 2.771735
    if Q.mass < 29.6447:
        z += 0.1656506 * Q.mass - 6.677423
    if 29.6447 <= Q.mass < 45.7571:
        z += 0.07458562 * Q.mass - 3.977828
    if 45.7571 <= Q.mass < 53.33237:
        z += 0.04175207 * Q.mass - 2.47546
    if Q.mass >= 53.33237:
        z += -0.03283354 * Q.mass + 1.502368
    if Q.e3 < 2.371297e-05:
        z += 54728.21 * Q.e3 - 1.297769
    if 8.147744e-05 <= Q.e3 < 0.0001869378:
        z += -5277.33 * Q.e3 + 0.4299833
    if Q.e3 >= 0.0001869378:
        z += -741.8335 * Q.e3 - 0.4178725
    if Q.girth2 < 0.003562611:
        z += -3591.238 * Q.girth2 + 16.49868
    if 0.003562611 <= Q.girth2 < 0.005019719:
        z += -1634.565 * Q.girth2 + 9.527815
    if 0.005019719 <= Q.girth2 < 0.00609665:
        z += -1228.265 * Q.girth2 + 7.4883
    if Q.sj3_dr_max < 0.1426152:
        z += 23.03381 * Q.sj3_dr_max - 1.811188
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -20.82089 * Q.sj3_dr_max + 4.443157
    if Q.lam2 < 0.0003061234:
        z += -1962.187 * Q.lam2 + 0.6006711
    if Q.lam2 >= 0.001130645:
        z += 512.5387 * Q.lam2 - 0.5794992
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.5789872 * Q.n_dr_0p2_0p4 - 0.5789872
    if Q.lam1 < 0.003377388:
        z += 584.2372 * Q.lam1 - 3.478636
    if 0.003377388 <= Q.lam1 < 0.005433361:
        z += 346.5589 * Q.lam1 - 2.675904
    if 0.005433361 <= Q.lam1 < 0.00595415:
        z += 778.7711 * Q.lam1 - 5.024268
    if Q.lam1 >= 0.00595415:
        z += 194.5338 * Q.lam1 - 1.545632
    if Q.centroid_offset < 0.002316125:
        z += 112.5424 * Q.centroid_offset + 3.249547
    if 0.002316125 <= Q.centroid_offset < 0.01837778:
        z += -218.5459 * Q.centroid_offset + 4.016389
    if 0.03556091 <= Q.e2 < 0.04447357:
        z += -43.22612 * Q.e2 + 1.53716
    if Q.e2 >= 0.04447357:
        z += 49.87312 * Q.e2 - 2.603295
    if Q.sj3_pair_mass_max < 24.23013:
        z += 0.03903889 * Q.sj3_pair_mass_max - 0.9459174
    if Q.width < 0.0001721983:
        z += -7356.491 * Q.width + 1.266775
    if Q.C3 < 0.0284695:
        z += 21.36064 * Q.C3 - 0.6081268
    if Q.zdr_0 < 0.006292091:
        z += 167.7517 * Q.zdr_0 - 1.055509
    if Q.mass_over_sum_pt < 0.07269073:
        z += 78.62965 * Q.mass_over_sum_pt - 5.715647
    if Q.sum_pt < 988.4078:
        z += -0.005267484 * Q.sum_pt + 5.206422
    z += -1.889101 * Q.z_dr_0p2_0p4
    if Q.log_sum_pt >= 6.896095:
        z += 9.508599 * Q.log_sum_pt - 65.57221
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.005940178 * Q.sum_pt_top5 + 4.989481
    if Q.pt_4 < 31.125:
        z += -0.04275462 * Q.pt_4 + 1.330737
    if Q.n_for_90pct < 7.0:
        z += 0.153264 * Q.n_for_90pct - 1.072848
    if Q.max_dr < 0.1117619:
        z += -28.43249 * Q.max_dr + 3.177668
    if Q.M3 < 0.0782171:
        z += -8.261358 * Q.M3 + 0.6461795
    if Q.absphi_1 < 0.02227783:
        z += 20.01593 * Q.absphi_1 - 0.4459116
    if Q.girth2_top5 >= 0.01148293:
        z += 49.06776 * Q.girth2_top5 - 0.5634415
    if Q.N2 >= 0.2233283:
        z += 1.907428 * Q.N2 - 0.4259826
    if Q.z_5 < 0.06503035:
        z += -13.54388 * Q.z_5 + 0.880763
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -228.7052 * Q.mass_over_sum_pt_sq + 1.642752
    if Q.zdr_2 < 0.002170915:
        z += -146.1438 * Q.zdr_2 + 0.3172658
    if Q.D2 >= 2.055451:
        z += -0.3987595 * Q.D2 + 0.8196307
    if Q.e2_sq < 0.0030133:
        z += 578.1713 * Q.e2_sq - 1.742204
    if Q.mass < 53.33237 and Q.centroid_offset < 0.02685622:
        z += 4.772529 * (53.33237 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass < 53.33237 and Q.log_sum_pt < 6.842717:
        z += 0.1333241 * (53.33237 - Q.mass) * (6.842717 - Q.log_sum_pt)
    if Q.girth2 < 0.00609665 and Q.planar_flow < 0.3220738:
        z += 700.1217 * (0.00609665 - Q.girth2) * (0.3220738 - Q.planar_flow)
    if Q.girth2 < 0.00609665 and Q.mean_phi < 0.002834884:
        z += -8418.642 * (0.00609665 - Q.girth2) * (0.002834884 - Q.mean_phi)
    if Q.mass < 53.33237 and Q.D2 > 2.843757:
        z += 0.009408624 * (53.33237 - Q.mass) * (Q.D2 - 2.843757)
    if Q.lam1 > 0.005433361 and Q.eccentricity > 0.7117266:
        z += -634.9865 * (Q.lam1 - 0.005433361) * (Q.eccentricity - 0.7117266)
    if Q.tau1 < 0.04369778 and Q.eccentricity > 0.9031255:
        z += -196.6898 * (0.04369778 - Q.tau1) * (Q.eccentricity - 0.9031255)
    if Q.sj3_dr_max < 0.1070199 and Q.abseta_1 > 0.03625488:
        z += -736.4925 * (0.1070199 - Q.sj3_dr_max) * (Q.abseta_1 - 0.03625488)
    if Q.tau1 < 0.04369778 and Q.mean_phi < -0.01275329:
        z += 1748.597 * (0.04369778 - Q.tau1) * (-0.01275329 - Q.mean_phi)
    if Q.tau1 < 0.04369778 and Q.mean_phi > 0.02612796:
        z += 5420.702 * (0.04369778 - Q.tau1) * (Q.mean_phi - 0.02612796)
    if Q.centroid_offset < 0.01837778 and Q.tau4 > 0.001224244:
        z += -6834.893 * (0.01837778 - Q.centroid_offset) * (Q.tau4 - 0.001224244)
    if Q.girth2 < 0.00609665 and Q.mean_phi > 0.02612796:
        z += -49387.91 * (0.00609665 - Q.girth2) * (Q.mean_phi - 0.02612796)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.716559:
        z += -35.51538 * (Q.log_sum_pt - 6.896095) * (0.716559 - Q.D2_b2)
    if Q.centroid_offset < 0.01837778 and Q.n_for_90pct > 5.0:
        z += -35.45056 * (0.01837778 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.sj3_dr_max < 0.1426152 and Q.pt_6 > 33.6875:
        z += 0.3340152 * (0.1426152 - Q.sj3_dr_max) * (Q.pt_6 - 33.6875)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 < 159.25:
        z += -1.825005 * (0.01837778 - Q.centroid_offset) * (159.25 - Q.pt_1)
    if Q.centroid_offset < 0.01837778 and Q.z_2nd < 0.2055511:
        z += 961.8382 * (0.01837778 - Q.centroid_offset) * (0.2055511 - Q.z_2nd)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.sj3_dr_min < 0.2089872:
        z += -2.826319 * (Q.n_dr_0p2_0p4 - 1.0) * (0.2089872 - Q.sj3_dr_min)
    if Q.girth < 0.05464922 and Q.tau21_b2 < 0.02656143:
        z += 4196.679 * (0.05464922 - Q.girth) * (0.02656143 - Q.tau21_b2)
    if Q.pt_4 < 31.125 and Q.M3 > 0.05210278:
        z += 7.621526 * (31.125 - Q.pt_4) * (Q.M3 - 0.05210278)
    if Q.tau1 < 0.04369778 and Q.tau21_b2 < 0.02656143:
        z += -4748.4 * (0.04369778 - Q.tau1) * (0.02656143 - Q.tau21_b2)
    if Q.sum_pt_top5 > 839.9547 and Q.tau21_b2 < 0.004811143:
        z += -4.58147 * (Q.sum_pt_top5 - 839.9547) * (0.004811143 - Q.tau21_b2)
    if Q.mass_over_sum_pt < 0.07269073 and Q.mass_top2 > 1.933087:
        z += -1.845825 * (0.07269073 - Q.mass_over_sum_pt) * (Q.mass_top2 - 1.933087)
    if Q.zdr_0 < 0.006292091 and Q.pt_2 > 88.4375:
        z += 1.045519 * (0.006292091 - Q.zdr_0) * (Q.pt_2 - 88.4375)
    if Q.e2 > 0.03556091 and Q.mean_phi > 0.009050008:
        z += -159.2314 * (Q.e2 - 0.03556091) * (Q.mean_phi - 0.009050008)
    if Q.mass_over_sum_pt < 0.07269073 and Q.phi_0 < -0.004917145:
        z += 529.511 * (0.07269073 - Q.mass_over_sum_pt) * (-0.004917145 - Q.phi_0)
    if Q.centroid_offset < 0.01837778 and Q.mean_eta > 6.288824e-05:
        z += 4729.881 * (0.01837778 - Q.centroid_offset) * (Q.mean_eta - 6.288824e-05)
    if Q.sum_pt_top5 > 839.9547 and Q.pt1_dr01 < 1.797343:
        z += -0.002026048 * (Q.sum_pt_top5 - 839.9547) * (1.797343 - Q.pt1_dr01)
    if Q.centroid_offset < 0.01837778 and Q.pt1_dr01 > 17.82896:
        z += -3.059499 * (0.01837778 - Q.centroid_offset) * (Q.pt1_dr01 - 17.82896)
    if Q.girth < 0.05464922 and Q.z_6 > 0.03448406:
        z += -706.9622 * (0.05464922 - Q.girth) * (Q.z_6 - 0.03448406)
    if Q.sum_pt < 988.4078 and Q.dr_2 < 0.01778111:
        z += -0.176864 * (988.4078 - Q.sum_pt) * (0.01778111 - Q.dr_2)
    if Q.e2 > 0.04447357 and Q.sj3_mass3 < 0.2723288:
        z += 40.07455 * (Q.e2 - 0.04447357) * (0.2723288 - Q.sj3_mass3)
    if Q.e3 > 0.0001869378 and Q.pt_2 > 56.5:
        z += 36.74884 * (Q.e3 - 0.0001869378) * (Q.pt_2 - 56.5)
    if Q.tau1 < 0.04369778 and Q.n_dr_0p05_0p1 > 5.0:
        z += -19.12064 * (0.04369778 - Q.tau1) * (Q.n_dr_0p05_0p1 - 5.0)
    if Q.sd_mass > 44.82259 and Q.n_dr_0p05_0p1 > 7.0:
        z += -0.07595047 * (Q.sd_mass - 44.82259) * (Q.n_dr_0p05_0p1 - 7.0)
    if Q.mass < 53.33237 and Q.dr1_7 > 0.2025074:
        z += -0.4962789 * (53.33237 - Q.mass) * (Q.dr1_7 - 0.2025074)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.754032
    z += -7.871966 * Q.e2
    if 11.051 <= Q.sj3_pair_mass_min < 15.95929:
        z += 0.0143966 * Q.sj3_pair_mass_min - 0.1590969
    if Q.sj3_pair_mass_min >= 15.95929:
        z += 0.1221231 * Q.sj3_pair_mass_min - 1.878335
    if Q.lam1 < 0.001503553:
        z += 1068.901 * Q.lam1 - 3.347342
    if 0.001503553 <= Q.lam1 < 0.004183811:
        z += 574.9735 * Q.lam1 - 2.604696
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += 112.473 * Q.lam1 - 0.6696811
    if Q.lam1 >= 0.00733008:
        z += -262.1292 * Q.lam1 + 1.921428
    if 0.0003061234 <= Q.lam2 < 0.003408389:
        z += 959.8844 * Q.lam2 - 0.293843
    if Q.lam2 >= 0.003408389:
        z += 815.6317 * Q.lam2 + 0.1978262
    if Q.pt_7 < 45.75:
        z += 0.01360544 * Q.pt_7 - 0.6224489
    if Q.sj3_dr_min >= 0.1278212:
        z += 13.0736 * Q.sj3_dr_min - 1.671083
    if 22.18342 <= Q.mass_top5 < 45.32077:
        z += 0.01538173 * Q.mass_top5 - 0.3412192
    if Q.mass_top5 >= 45.32077:
        z += -0.005184614 * Q.mass_top5 + 0.5908631
    if Q.zdr_0 < 0.004918231:
        z += 137.1173 * Q.zdr_0 - 1.078607
    if 0.004918231 <= Q.zdr_0 < 0.0211821:
        z += 24.85462 * Q.zdr_0 - 0.5264732
    if Q.M2 < 0.02563286:
        z += 33.35508 * Q.M2 - 0.8549861
    if Q.sum_pt >= 988.4078:
        z += -0.006553122 * Q.sum_pt + 6.477157
    if Q.z_7 < 0.06473447:
        z += 6.001247 * Q.z_7 - 0.3884876
    if Q.LHA >= 0.3033137:
        z += -23.62416 * Q.LHA + 7.165533
    if Q.tau1 >= 0.05356915:
        z += 10.59053 * Q.tau1 - 0.5673258
    if Q.girth2 < 0.002635418:
        z += 314.14 * Q.girth2 - 0.8278902
    if 0.007520088 <= Q.girth2 < 0.02530566:
        z += 417.9058 * Q.girth2 - 3.142688
    if Q.girth2 >= 0.02530566:
        z += 344.7753 * Q.girth2 - 1.292074
    if Q.e3 >= 3.892127e-05:
        z += -449.1781 * Q.e3 + 0.01748258
    if Q.mass < 76.6557:
        z += -0.02435549 * Q.mass + 1.866987
    if Q.girth2_top3 < 0.002151568:
        z += -275.7032 * Q.girth2_top3 + 0.5931942
    if Q.C2_b2 >= 0.009032972:
        z += 28.3135 * Q.C2_b2 - 0.2557551
    if Q.zdr_7 < 0.00325401:
        z += 73.29528 * Q.zdr_7 - 0.2385036
    if Q.centroid_offset < 0.02355416:
        z += -14.10154 * Q.centroid_offset + 0.3321499
    if Q.pt_7 < 45.75 and Q.D2 < 1.002471:
        z += 0.07627818 * (45.75 - Q.pt_7) * (1.002471 - Q.D2)
    if Q.zdr_0 < 0.0211821 and Q.dr_7 > 0.1078355:
        z += 157.4332 * (0.0211821 - Q.zdr_0) * (Q.dr_7 - 0.1078355)
    if Q.zdr_0 < 0.0211821 and Q.centroid_offset > 0.01258764:
        z += 1603.259 * (0.0211821 - Q.zdr_0) * (Q.centroid_offset - 0.01258764)
    if Q.pt_7 < 45.75 and Q.log_sum_pt < 6.572938:
        z += -0.07526681 * (45.75 - Q.pt_7) * (6.572938 - Q.log_sum_pt)
    if Q.lam2 > 0.0003061234 and Q.planar_flow > 0.04505724:
        z += -536.2738 * (Q.lam2 - 0.0003061234) * (Q.planar_flow - 0.04505724)
    if Q.sj3_dr_min > 0.1278212 and Q.z_dr_0_0p05 < 0.3658817:
        z += -23.86072 * (Q.sj3_dr_min - 0.1278212) * (0.3658817 - Q.z_dr_0_0p05)
    if Q.zdr_0 < 0.0211821 and Q.z_dr_0p05_0p1 < 0.2919447:
        z += 145.4518 * (0.0211821 - Q.zdr_0) * (0.2919447 - Q.z_dr_0p05_0p1)
    if Q.e3 < 8.147744e-05 and Q.sj3_dr23 > 0.1797097:
        z += -88271.15 * (8.147744e-05 - Q.e3) * (Q.sj3_dr23 - 0.1797097)
    if Q.lam1 > 0.00733008 and Q.D2_b2 < 0.380911:
        z += -214.8887 * (Q.lam1 - 0.00733008) * (0.380911 - Q.D2_b2)
    if Q.lam1 > 0.00733008 and Q.D2_b2 > 1.129616:
        z += -52.39015 * (Q.lam1 - 0.00733008) * (Q.D2_b2 - 1.129616)
    if Q.sj3_pair_mass_min > 11.051 and Q.sj3_pairmin_over_m > 0.28737:
        z += -0.2546039 * (Q.sj3_pair_mass_min - 11.051) * (Q.sj3_pairmin_over_m - 0.28737)
    if Q.sj3_dr_min > 0.1278212 and Q.sj3_mass2 < 0.5973755:
        z += -7.582802 * (Q.sj3_dr_min - 0.1278212) * (0.5973755 - Q.sj3_mass2)
    if Q.lam1 < 0.00595415 and Q.z_dr_0p05_0p1 > 0.2919447:
        z += -381.6423 * (0.00595415 - Q.lam1) * (Q.z_dr_0p05_0p1 - 0.2919447)
    if Q.lam2 > 0.0003061234 and Q.pt_6 < 31.90625:
        z += -24.53067 * (Q.lam2 - 0.0003061234) * (31.90625 - Q.pt_6)
    if Q.tau1 > 0.05356915 and Q.D2 < 1.002471:
        z += 14.33592 * (Q.tau1 - 0.05356915) * (1.002471 - Q.D2)
    return max(0.0, z)


def neuron_11(Q):
    z = -2.043806
    if Q.planar_flow < 0.2534037:
        z += -6.694983 * Q.planar_flow + 1.696534
    if 0.1778793 <= Q.sj2_dr < 0.2687922:
        z += 4.518102 * Q.sj2_dr - 0.8036768
    if Q.sj2_dr >= 0.2687922:
        z += 17.68419 * Q.sj2_dr - 4.342618
    if Q.girth2 < 0.004372139:
        z += -377.8204 * Q.girth2 + 5.158165
    if 0.004372139 <= Q.girth2 < 0.01323868:
        z += -395.4511 * Q.girth2 + 5.235249
    if Q.tau1 < 0.09538712:
        z += 18.02973 * Q.tau1 - 1.719804
    if Q.mass < 15.45403:
        z += -0.08074558 * Q.mass - 1.26763
    if 15.45403 <= Q.mass < 69.61135:
        z += 0.04644756 * Q.mass - 3.233277
    if Q.centroid_offset < 0.01437952:
        z += -80.64653 * Q.centroid_offset + 2.855728
    if 0.01437952 <= Q.centroid_offset < 0.03776099:
        z += -72.53905 * Q.centroid_offset + 2.739147
    if Q.centroid_offset >= 0.04990367:
        z += 309.0794 * Q.centroid_offset - 15.42419
    if Q.lam1 < 0.0005049491:
        z += 1211.859 * Q.lam1 - 4.960677
    if 0.0005049491 <= Q.lam1 < 0.00733008:
        z += 580.9564 * Q.lam1 - 4.642103
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 366.9529 * Q.lam1 - 3.073441
    if Q.sj3_dr_max < 0.1426152:
        z += 9.641638 * Q.sj3_dr_max - 0.8307603
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 18.13582 * Q.sj3_dr_max - 2.04216
    if 0.169029 <= Q.sj3_dr_max < 0.2623172:
        z += -10.96945 * Q.sj3_dr_max + 2.877477
    if Q.width < 0.006679471:
        z += -1694.786 * Q.width + 13.8704
    if 0.006679471 <= Q.width < 0.008678045:
        z += -1275.974 * Q.width + 11.07296
    if Q.e2_sq < 0.008168571:
        z += 351.5768 * Q.e2_sq - 2.87188
    if Q.girth < 0.02689598:
        z += 93.63735 * Q.girth - 4.991648
    if 0.02689598 <= Q.girth < 0.0717028:
        z += 55.19649 * Q.girth - 3.957743
    if Q.z_7 >= 0.01685855:
        z += 41.7471 * Q.z_7 - 0.7037954
    if Q.pt_6 < 24.42188:
        z += 0.04074677 * Q.pt_6 - 0.7237492
    if 24.42188 <= Q.pt_6 < 41.21875:
        z += -0.01615558 * Q.pt_6 + 0.6659128
    if Q.max_pair_mass >= 33.3761:
        z += -0.02225576 * Q.max_pair_mass + 0.7428103
    if Q.pt_7 >= 29.04219:
        z += -0.04533856 * Q.pt_7 + 1.316731
    if Q.eccentricity >= 0.9884745:
        z += 52.82531 * Q.eccentricity - 52.21647
    if Q.C2 < 0.03578649:
        z += 18.05478 * Q.C2 - 0.6461169
    if Q.max_dr < 0.1452311:
        z += -6.645186 * Q.max_dr + 0.9650878
    if Q.e3 < 1.050302e-05:
        z += -64366.06 * Q.e3 + 0.6760381
    if Q.sum_pt_top5 < 506.875:
        z += 0.003566123 * Q.sum_pt_top5 - 1.807579
    if Q.sum_pt >= 988.4078:
        z += -0.002051231 * Q.sum_pt + 2.027453
    if Q.mass_over_sum_pt < 0.07637363:
        z += 50.10965 * Q.mass_over_sum_pt - 3.827056
    if Q.e2 < 0.01655442:
        z += -62.95713 * Q.e2 + 1.042219
    if Q.n_dr_0p1_0p2 >= 3.0:
        z += -0.3334581 * Q.n_dr_0p1_0p2 + 1.000374
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.01471023 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.pt_7 < 37.15625:
        z += -0.1372828 * (0.2534037 - Q.planar_flow) * (37.15625 - Q.pt_7)
    if Q.tau1 < 0.09538712 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += 4.023256 * (0.09538712 - Q.tau1) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.planar_flow < 0.2534037 and Q.e3 < 1.050302e-05:
        z += -484721.0 * (0.2534037 - Q.planar_flow) * (1.050302e-05 - Q.e3)
    if Q.sj2_dr > 0.1778793 and Q.lam2 < 0.001130645:
        z += -3822.427 * (Q.sj2_dr - 0.1778793) * (0.001130645 - Q.lam2)
    if Q.centroid_offset < 0.03776099 and Q.sum_pt < 901.5938:
        z += -0.1713786 * (0.03776099 - Q.centroid_offset) * (901.5938 - Q.sum_pt)
    if Q.centroid_offset < 0.03776099 and Q.mean_phi < -0.001813533:
        z += -850.8148 * (0.03776099 - Q.centroid_offset) * (-0.001813533 - Q.mean_phi)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += -108.6266 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.centroid_offset < 0.03776099 and Q.absphi_0 < 0.0345459:
        z += -330.802 * (0.03776099 - Q.centroid_offset) * (0.0345459 - Q.absphi_0)
    if Q.centroid_offset < 0.01437952 and Q.abseta_4 < 0.03601074:
        z += -620.1295 * (0.01437952 - Q.centroid_offset) * (0.03601074 - Q.abseta_4)
    if Q.centroid_offset < 0.01437952 and Q.D2_b2 < 0.5327104:
        z += 173.3364 * (0.01437952 - Q.centroid_offset) * (0.5327104 - Q.D2_b2)
    if Q.pt_6 < 41.21875 and Q.D2_b2 < 4.721224:
        z += -0.005601798 * (41.21875 - Q.pt_6) * (4.721224 - Q.D2_b2)
    if Q.e2_sq < 0.008168571 and Q.mass_top2 > 6.779915:
        z += -4.924866 * (0.008168571 - Q.e2_sq) * (Q.mass_top2 - 6.779915)
    if Q.eccentricity > 0.9884745 and Q.sj2_mass2 < 2.771069:
        z += -17.47326 * (Q.eccentricity - 0.9884745) * (2.771069 - Q.sj2_mass2)
    if Q.sum_pt > 988.4078 and Q.phi_0 < -0.07818909:
        z += 0.6040877 * (Q.sum_pt - 988.4078) * (-0.07818909 - Q.phi_0)
    if Q.centroid_offset < 0.01437952 and Q.zdr_7 < 0.00279494:
        z += -6844.488 * (0.01437952 - Q.centroid_offset) * (0.00279494 - Q.zdr_7)
    if Q.sj2_dr > 0.1778793 and Q.dr_7 < 0.04881348:
        z += 184.9741 * (Q.sj2_dr - 0.1778793) * (0.04881348 - Q.dr_7)
    if Q.sj3_dr_max < 0.2623172 and Q.phi_0 < -0.0216713:
        z += 87.69018 * (0.2623172 - Q.sj3_dr_max) * (-0.0216713 - Q.phi_0)
    if Q.centroid_offset < 0.01437952 and Q.sj3_pair_mass_min < 15.95929:
        z += -9.608822 * (0.01437952 - Q.centroid_offset) * (15.95929 - Q.sj3_pair_mass_min)
    if Q.girth < 0.02689598 and Q.sj3_pair_mass_min > 1.590484:
        z += -5.734141 * (0.02689598 - Q.girth) * (Q.sj3_pair_mass_min - 1.590484)
    if Q.planar_flow < 0.2534037 and Q.sj3_pair_mass_min > 1.282345:
        z += -0.2043668 * (0.2534037 - Q.planar_flow) * (Q.sj3_pair_mass_min - 1.282345)
    if Q.sum_pt_top5 < 506.875 and Q.n_dr_0p1_0p2 > 0.0:
        z += 0.0002097257 * (506.875 - Q.sum_pt_top5) * (Q.n_dr_0p1_0p2 - 0.0)
    if Q.centroid_offset > 0.04990367 and Q.tau4 > 0.002207727:
        z += -125360.9 * (Q.centroid_offset - 0.04990367) * (Q.tau4 - 0.002207727)
    if Q.centroid_offset > 0.04990367 and Q.zdr_7 > 0.004322471:
        z += -10662.43 * (Q.centroid_offset - 0.04990367) * (Q.zdr_7 - 0.004322471)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.4358689
    if Q.girth2 >= 0.01882765:
        z += 352.154 * Q.girth2 - 6.630233
    if Q.mass >= 88.15578:
        z += 0.06658865 * Q.mass - 5.870174
    if Q.mass_over_sum_pt >= 0.1309286:
        z += -53.00335 * Q.mass_over_sum_pt + 6.939657
    if Q.zdr_0 >= 0.03981924:
        z += -24.90007 * Q.zdr_0 + 0.9915021
    if Q.girth2_top2 >= 0.01403324:
        z += -29.6847 * Q.girth2_top2 + 0.4165726
    if Q.girth2 > 0.01882765 and Q.lam2 > 0.000537286:
        z += 5342.096 * (Q.girth2 - 0.01882765) * (Q.lam2 - 0.000537286)
    if Q.girth2 > 0.01882765 and Q.abseta_6 > 0.02839661:
        z += 185.4277 * (Q.girth2 - 0.01882765) * (Q.abseta_6 - 0.02839661)
    if Q.girth2 > 0.01882765 and Q.pt_7 < 53.4375:
        z += -2.843648 * (Q.girth2 - 0.01882765) * (53.4375 - Q.pt_7)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.3828516
    if Q.girth < 0.007673833:
        z += 125.3074 * Q.girth + 7.740563
    if 0.007673833 <= Q.girth < 0.1484084:
        z += -61.83377 * Q.girth + 9.176653
    if Q.lam1 < 0.006506576:
        z += -328.8819 * Q.lam1 + 3.607201
    if 0.006506576 <= Q.lam1 < 0.01643375:
        z += -147.8071 * Q.lam1 + 2.429024
    if Q.e3 < 5.334511e-05:
        z += -10525.58 * Q.e3 + 0.5614884
    if Q.pt_6 < 31.90625:
        z += 0.01868046 * Q.pt_6 - 0.5960233
    if Q.sum_pt >= 988.4078:
        z += -0.003796963 * Q.sum_pt + 3.752948
    if Q.e2 < 0.08000524:
        z += 45.73376 * Q.e2 - 3.658941
    if Q.mass >= 49.6681:
        z += -0.01377043 * Q.mass + 0.6839511
    if Q.z_5 < 0.02818362:
        z += 173.1591 * Q.z_5 - 4.880251
    if Q.z_7 < 0.02807091:
        z += 127.4221 * Q.z_7 - 3.576855
    if Q.sum_pt_top5 >= 791.125:
        z += 0.006949916 * Q.sum_pt_top5 - 5.498252
    if Q.sj3_dr23 >= 0.1974628:
        z += -2.459822 * Q.sj3_dr23 + 0.4857233
    if Q.log_sum_pt >= 6.502799:
        z += 1.024125 * Q.log_sum_pt - 6.659676
    if Q.pt_7 >= 48.71875:
        z += -0.03643152 * Q.pt_7 + 1.774898
    if Q.width < 0.007520088:
        z += 156.0814 * Q.width - 0.8595241
    if 0.007520088 <= Q.width < 0.01323868:
        z += -54.94749 * Q.width + 0.727432
    if Q.z_6 < 0.02160287:
        z += 103.7899 * Q.z_6 - 2.24216
    if Q.girth < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -64.15601 * (0.1484084 - Q.girth) * (6.804164 - Q.log_sum_pt)
    if Q.girth < 0.1484084 and Q.pt_7 < 38.53125:
        z += -0.9565173 * (0.1484084 - Q.girth) * (38.53125 - Q.pt_7)
    if Q.sum_pt_top5 > 658.125 and Q.z_7 > 0.02320757:
        z += -0.08297533 * (Q.sum_pt_top5 - 658.125) * (Q.z_7 - 0.02320757)
    if Q.e3 < 5.334511e-05 and Q.centroid_offset < 0.03776099:
        z += -879373.0 * (5.334511e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 40.04062:
        z += 0.0006305688 * (Q.sum_pt_top5 - 658.125) * (40.04062 - Q.pt_7)
    if Q.pt_6 < 31.90625 and Q.z_7 < 0.0586137:
        z += -3.241924 * (31.90625 - Q.pt_6) * (0.0586137 - Q.z_7)
    if Q.girth < 0.1484084 and Q.z_7 > 0.06164517:
        z += 343.9248 * (0.1484084 - Q.girth) * (Q.z_7 - 0.06164517)
    if Q.girth < 0.1484084 and Q.lam2 < 0.000537286:
        z += -7629.709 * (0.1484084 - Q.girth) * (0.000537286 - Q.lam2)
    if Q.girth < 0.1484084 and Q.sj2_mass1 > 31.78116:
        z += -1.498993 * (0.1484084 - Q.girth) * (Q.sj2_mass1 - 31.78116)
    if Q.girth < 0.1484084 and Q.M3 < 0.07474969:
        z += 104.1245 * (0.1484084 - Q.girth) * (0.07474969 - Q.M3)
    if Q.e3 < 5.334511e-05 and Q.D3 > 0.2213841:
        z += 3734.638 * (5.334511e-05 - Q.e3) * (Q.D3 - 0.2213841)
    if Q.girth < 0.1484084 and Q.tau2 > 0.008780509:
        z += 274.7014 * (0.1484084 - Q.girth) * (Q.tau2 - 0.008780509)
    if Q.mass > 49.6681 and Q.z_dr_0p1_0p2 < 0.4684459:
        z += -0.01845044 * (Q.mass - 49.6681) * (0.4684459 - Q.z_dr_0p1_0p2)
    if Q.log_sum_pt > 6.502799 and Q.zdr_7 > 0.0007928864:
        z += -340.8875 * (Q.log_sum_pt - 6.502799) * (Q.zdr_7 - 0.0007928864)
    if Q.log_sum_pt > 6.502799 and Q.pair_mass_0_7 > 10.2219:
        z += 0.08796449 * (Q.log_sum_pt - 6.502799) * (Q.pair_mass_0_7 - 10.2219)
    if Q.sum_pt_top5 > 791.125 and Q.pt_6 < 31.90625:
        z += 0.001174797 * (Q.sum_pt_top5 - 791.125) * (31.90625 - Q.pt_6)
    if Q.sum_pt > 988.4078 and Q.pt_6 < 62.25:
        z += -0.0006184894 * (Q.sum_pt - 988.4078) * (62.25 - Q.pt_6)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.1196781
    if Q.planar_flow < 0.08366273:
        z += 1.499033 * Q.planar_flow + 0.03325868
    if 0.08366273 <= Q.planar_flow < 0.1115136:
        z += -5.697204 * Q.planar_flow + 0.6353155
    if 0.7143804 <= Q.psi_0p1 < 0.9761279:
        z += -0.5882583 * Q.psi_0p1 + 0.4202402
    if Q.psi_0p1 >= 0.9761279:
        z += -15.28506 * Q.psi_0p1 + 14.7662
    if 0.0009641429 <= Q.girth2 < 0.007520088:
        z += -205.0652 * Q.girth2 + 0.1977122
    if 0.007520088 <= Q.girth2 < 0.008678045:
        z += -1131.364 * Q.girth2 + 7.163564
    if Q.girth2 >= 0.008678045:
        z += -1329.309 * Q.girth2 + 8.881334
    if 0.002074109 <= Q.e2_sq < 0.0030133:
        z += 373.6779 * Q.e2_sq - 0.7750489
    if 0.0030133 <= Q.e2_sq < 0.01165737:
        z += 90.05844 * Q.e2_sq + 0.07958189
    if Q.e2_sq >= 0.01165737:
        z += -736.6524 * Q.e2_sq + 9.71686
    if 0.003562611 <= Q.width < 0.005590289:
        z += 428.8335 * Q.width - 1.527767
    if 0.005590289 <= Q.width < 0.006679471:
        z += 559.2512 * Q.width - 2.256839
    if 0.006679471 <= Q.width < 0.01323868:
        z += -799.3016 * Q.width + 6.817575
    if Q.width >= 0.01323868:
        z += -1023.913 * Q.width + 9.791131
    if 0.01655442 <= Q.e2 < 0.03556091:
        z += -113.2769 * Q.e2 + 1.875234
    if 0.03556091 <= Q.e2 < 0.04110972:
        z += -129.8634 * Q.e2 + 2.465063
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += -76.66616 * Q.e2 + 0.2781403
    if Q.e2 >= 0.05028464:
        z += 248.8959 * Q.e2 - 16.09263
    if 0.03776099 <= Q.centroid_offset < 0.04990367:
        z += -72.21106 * Q.centroid_offset + 2.726761
    if Q.centroid_offset >= 0.04990367:
        z += -895.7502 * Q.centroid_offset + 43.82439
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 116.5835 * Q.mass_over_sum_pt - 9.317788
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 282.9877 * Q.mass_over_sum_pt - 23.42081
    if Q.mass_over_sum_pt >= 0.09041383:
        z += 280.9056 * Q.mass_over_sum_pt - 23.23256
    if Q.n_dr_0_0p05 < 5.0:
        z += -0.1810618 * Q.n_dr_0_0p05 + 0.9053092
    if Q.e3 < 5.334511e-05:
        z += 20852.5 * Q.e3 - 1.479911
    if 5.334511e-05 <= Q.e3 < 8.147744e-05:
        z += 13064.4 * Q.e3 - 1.064454
    if Q.sd_mass < 49.91626:
        z += -0.04154737 * Q.sd_mass + 1.774401
    if 49.91626 <= Q.sd_mass < 74.57663:
        z += 0.01214454 * Q.sd_mass - 0.9056988
    if 0.02689598 <= Q.girth < 0.04081947:
        z += 58.76382 * Q.girth - 1.58051
    if 0.04081947 <= Q.girth < 0.08065885:
        z += 107.0133 * Q.girth - 3.550028
    if 0.08065885 <= Q.girth < 0.08723651:
        z += 152.8086 * Q.girth - 7.243826
    if 0.08723651 <= Q.girth < 0.1019409:
        z += -104.1431 * Q.girth + 15.17175
    if Q.girth >= 0.1019409:
        z += -241.6844 * Q.girth + 29.19283
    if Q.mass >= 69.61135:
        z += -0.01552749 * Q.mass + 1.08089
    if Q.mass_top5 >= 49.18618:
        z += -0.01095723 * Q.mass_top5 + 0.5389444
    if 0.06154135 <= Q.sj2_dr < 0.1294903:
        z += -4.708367 * Q.sj2_dr + 0.2897593
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -12.68451 * Q.sj2_dr + 1.322592
    if 0.1591713 <= Q.sj2_dr < 0.1778793:
        z += 22.52479 * Q.sj2_dr - 4.281717
    if 0.1778793 <= Q.sj2_dr < 0.2001708:
        z += 16.15375 * Q.sj2_dr - 3.148442
    if Q.sj2_dr >= 0.2001708:
        z += 6.364017 * Q.sj2_dr - 1.188823
    if Q.lam2 < 0.000537286:
        z += -1034.107 * Q.lam2 + 0.5556112
    if Q.C2_b2 < 0.004032342:
        z += 146.7191 * Q.C2_b2 - 0.5916217
    if Q.LHA >= 0.3033137:
        z += 26.65335 * Q.LHA - 8.084328
    if Q.z_dr_0p05_0p1 < 0.163898:
        z += 2.556852 * Q.z_dr_0p05_0p1 - 0.4190631
    if Q.sum_pt < 715.4688:
        z += 0.004887735 * Q.sum_pt - 3.497022
    if Q.sd_rg < 0.2330919:
        z += 8.562423 * Q.sd_rg - 2.779757
    if 0.2330919 <= Q.sd_rg < 0.2787955:
        z += -9.736121 * Q.sd_rg + 1.485486
    if 0.2787955 <= Q.sd_rg < 0.324646:
        z += 17.47601 * Q.sd_rg - 6.101135
    if Q.sd_rg >= 0.324646:
        z += 8.913589 * Q.sd_rg - 3.321379
    if Q.z_dr_0_0p05 >= 0.1515405:
        z += 1.375004 * Q.z_dr_0_0p05 - 0.2083687
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -3316.504 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += 1124.303 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.width < 0.00609665:
        z += 1010.548 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.width)
    if Q.planar_flow < 0.1115136 and Q.sum_pt_top5 < 658.125:
        z += -0.01656797 * (0.1115136 - Q.planar_flow) * (658.125 - Q.sum_pt_top5)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -296.0416 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.e2_sq > 0.002074109 and Q.sj2_dr < 0.1872617:
        z += -1677.385 * (Q.e2_sq - 0.002074109) * (0.1872617 - Q.sj2_dr)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.D3 < 0.3201347:
        z += 3.547209 * (0.5882598 - Q.z_dr_0p05_0p1) * (0.3201347 - Q.D3)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 1.0:
        z += 6.876778 * (Q.z_dr_0p05_0p1 - 0.7509095) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.e3 < 5.334511e-05 and Q.sj3_dr23 > 0.1629004:
        z += 161109.6 * (5.334511e-05 - Q.e3) * (Q.sj3_dr23 - 0.1629004)
    if Q.sj2_dr > 0.2001708 and Q.z_5 < 0.1058993:
        z += -42.1987 * (Q.sj2_dr - 0.2001708) * (0.1058993 - Q.z_5)
    if Q.sj2_dr > 0.1591713 and Q.D2_b2 < 0.08499387:
        z += 592.4234 * (Q.sj2_dr - 0.1591713) * (0.08499387 - Q.D2_b2)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.D2_b2 < 0.1830092:
        z += -4.779664 * (0.5882598 - Q.z_dr_0p05_0p1) * (0.1830092 - Q.D2_b2)
    if Q.sj2_dr > 0.2001708 and Q.D2_b2 < 0.08499387:
        z += -488.5468 * (Q.sj2_dr - 0.2001708) * (0.08499387 - Q.D2_b2)
    if Q.sj2_dr > 0.1294903 and Q.D2_b2 < 0.08499387:
        z += -221.2058 * (Q.sj2_dr - 0.1294903) * (0.08499387 - Q.D2_b2)
    if Q.psi_0p1 > 0.8155839 and Q.D2_b2 < 0.08499387:
        z += 37.72817 * (Q.psi_0p1 - 0.8155839) * (0.08499387 - Q.D2_b2)
    if Q.planar_flow < 0.1115136 and Q.mean_eta > -0.006779839:
        z += -112.2736 * (0.1115136 - Q.planar_flow) * (Q.mean_eta - -0.006779839)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.sum_pt < 715.4688:
        z += 0.002246607 * (0.5882598 - Q.z_dr_0p05_0p1) * (715.4688 - Q.sum_pt)
    if Q.centroid_offset > 0.04990367 and Q.C2_b2 < 0.0008333816:
        z += 1037492.0 * (Q.centroid_offset - 0.04990367) * (0.0008333816 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.C2_b2 < 0.02415398:
        z += -288.14 * (Q.z_dr_0p05_0p1 - 0.7509095) * (0.02415398 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.zdr_5 > 0.0155217:
        z += -3737.566 * (Q.z_dr_0p05_0p1 - 0.7509095) * (Q.zdr_5 - 0.0155217)
    if Q.sd_rg > 0.2787955 and Q.D2_b2 < 0.2669656:
        z += -76.79819 * (Q.sd_rg - 0.2787955) * (0.2669656 - Q.D2_b2)
    if Q.girth > 0.02689598 and Q.D2_b2 > 4.721224:
        z += -74.36401 * (Q.girth - 0.02689598) * (Q.D2_b2 - 4.721224)
    if Q.sj2_dr > 0.1591713 and Q.dr_2 < 0.03293672:
        z += -503.3022 * (Q.sj2_dr - 0.1591713) * (0.03293672 - Q.dr_2)
    if Q.sj2_dr > 0.2001708 and Q.dr_2 < 0.05056028:
        z += 246.9179 * (Q.sj2_dr - 0.2001708) * (0.05056028 - Q.dr_2)
    if Q.centroid_offset > 0.02685622 and Q.pt_1 < 150.625:
        z += 0.1270221 * (Q.centroid_offset - 0.02685622) * (150.625 - Q.pt_1)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.307148
    if Q.N2 < 0.2233283:
        z += 8.233815 * Q.N2 - 1.838844
    if Q.girth2 < 0.007520088:
        z += 646.8815 * Q.girth2 - 4.864606
    if Q.width < 0.00609665:
        z += 198.4363 * Q.width + 2.394528
    if 0.00609665 <= Q.width < 0.01323868:
        z += -504.6642 * Q.width + 6.681086
    if Q.e2 < 0.03846696:
        z += -72.21304 * Q.e2 + 2.667226
    if 0.03846696 <= Q.e2 < 0.04110972:
        z += -127.8255 * Q.e2 + 4.806468
    if 0.04110972 <= Q.e2 < 0.06344108:
        z += 20.07948 * Q.e2 - 1.273864
    if Q.e2_sq < 0.008168571:
        z += 959.9686 * Q.e2_sq - 8.939327
    if 0.008168571 <= Q.e2_sq < 0.01165737:
        z += 314.6509 * Q.e2_sq - 3.668004
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -479.8345 * Q.mass_over_sum_pt_sq + 3.446573
    if Q.lam1 < 0.00483998:
        z += 296.7742 * Q.lam1 - 1.301512
    if 0.00483998 <= Q.lam1 < 0.005433361:
        z += -227.2889 * Q.lam1 + 1.234943
    if Q.lam2 < 0.0003061234:
        z += 1337.999 * Q.lam2 + 0.1969568
    if 0.0003061234 <= Q.lam2 < 0.001130645:
        z += -735.6382 * Q.lam2 + 0.8317454
    if Q.girth2_top3 < 0.002151568:
        z += -216.061 * Q.girth2_top3 + 1.119579
    if 0.002151568 <= Q.girth2_top3 < 0.006756161:
        z += -142.186 * Q.girth2_top3 + 0.9606315
    if Q.girth < 0.08723651:
        z += 97.00581 * Q.girth - 9.105357
    if 0.08723651 <= Q.girth < 0.1019409:
        z += 43.72212 * Q.girth - 4.457073
    if Q.sj2_dr < 0.1492731:
        z += -4.69138 * Q.sj2_dr + 0.1444173
    if 0.1492731 <= Q.sj2_dr < 0.1591713:
        z += -25.29647 * Q.sj2_dr + 3.220203
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 19.66534 * Q.sj2_dr - 3.936427
    if Q.girth2_top2 < 0.0005124533:
        z += -4388.291 * Q.girth2_top2 + 3.442141
    if 0.0005124533 <= Q.girth2_top2 < 0.007639643:
        z += -167.4358 * Q.girth2_top2 + 1.27915
    if Q.centroid_offset < 0.01837778:
        z += 37.26431 * Q.centroid_offset - 0.6848353
    if Q.LHA < 0.3033137:
        z += -0.9881015 * Q.LHA + 0.2997048
    if Q.tau1 < 0.0262518:
        z += -120.5427 * Q.tau1 + 6.951239
    if 0.0262518 <= Q.tau1 < 0.06345984:
        z += -101.773 * Q.tau1 + 6.458501
    if Q.sj3_pair_mass_max >= 37.79615:
        z += -0.01914714 * Q.sj3_pair_mass_max + 0.7236882
    if 45.7571 <= Q.mass < 76.6557:
        z += 0.05470889 * Q.mass - 2.50332
    if Q.mass >= 76.6557:
        z += 0.004176319 * Q.mass + 1.37029
    if Q.log_sum_pt < 6.080494:
        z += 5.625483 * Q.log_sum_pt - 34.20572
    if 0.6747704 <= Q.z_dr_0p05_0p1 < 0.7509095:
        z += -2.732227 * Q.z_dr_0p05_0p1 + 1.843626
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += -17.84965 * Q.z_dr_0p05_0p1 + 13.19544
    if Q.N2 < 0.2233283 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -6.84907 * (0.2233283 - Q.N2) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.N2 < 0.2233283 and Q.max_dr < 0.121681:
        z += -95.45535 * (0.2233283 - Q.N2) * (0.121681 - Q.max_dr)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 < 716.8828:
        z += 0.03826917 * (0.2233283 - Q.N2) * (716.8828 - Q.sum_pt_top5)
    if Q.girth2 < 0.007520088 and Q.D2 < 0.7459513:
        z += -2298.596 * (0.007520088 - Q.girth2) * (0.7459513 - Q.D2)
    if Q.width < 0.00609665 and Q.D2 < 0.7459513:
        z += 4277.537 * (0.00609665 - Q.width) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.D2 < 0.7459513:
        z += 18.66154 * (0.1309286 - Q.mass_over_sum_pt) * (0.7459513 - Q.D2)
    if Q.N2 < 0.2233283 and Q.n_dr_0p1_0p2 < 4.0:
        z += 1.311734 * (0.2233283 - Q.N2) * (4.0 - Q.n_dr_0p1_0p2)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.04279934 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
    if Q.N2 < 0.2233283 and Q.n_for_90pct < 6.0:
        z += -3.922777 * (0.2233283 - Q.N2) * (6.0 - Q.n_for_90pct)
    if Q.sj3_pair_mass_max > 72.58129 and Q.N3 < 1.666763:
        z += -0.0463082 * (Q.sj3_pair_mass_max - 72.58129) * (1.666763 - Q.N3)
    if Q.lam2 < 0.001130645 and Q.mean_phi > -0.01753483:
        z += -11743.48 * (0.001130645 - Q.lam2) * (Q.mean_phi - -0.01753483)
    if Q.e2 < 0.06344108 and Q.dr12 > 0.119512:
        z += 130.4877 * (0.06344108 - Q.e2) * (Q.dr12 - 0.119512)
    if Q.width < 0.01882765 and Q.ptdr0_2 > 14.78048:
        z += 4.278563 * (0.01882765 - Q.width) * (Q.ptdr0_2 - 14.78048)
    if Q.centroid_offset < 0.01837778 and Q.dr01 < 0.2512159:
        z += -372.4741 * (0.01837778 - Q.centroid_offset) * (0.2512159 - Q.dr01)
    if Q.centroid_offset < 0.01837778 and Q.zdr_0 < 0.01216943:
        z += -16985.78 * (0.01837778 - Q.centroid_offset) * (0.01216943 - Q.zdr_0)
    if Q.lam2 < 0.001130645 and Q.n_for_50pct > 1.0:
        z += -140.0976 * (0.001130645 - Q.lam2) * (Q.n_for_50pct - 1.0)
    if Q.width < 0.00609665 and Q.D2_b2 < 0.08499387:
        z += -7455.062 * (0.00609665 - Q.width) * (0.08499387 - Q.D2_b2)
    if Q.width < 0.01323868 and Q.tau21_b2 < 0.01272888:
        z += 27923.8 * (0.01323868 - Q.width) * (0.01272888 - Q.tau21_b2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.tau21_b2 < 0.01272888:
        z += -3348.137 * (0.1309286 - Q.mass_over_sum_pt) * (0.01272888 - Q.tau21_b2)
    if Q.width < 0.01882765 and Q.lam2 < 0.003408389:
        z += 75528.96 * (0.01882765 - Q.width) * (0.003408389 - Q.lam2)
    if Q.e2_sq < 0.01165737 and Q.lam2 > 6.531723e-06:
        z += 293174.5 * (0.01165737 - Q.e2_sq) * (Q.lam2 - 6.531723e-06)
    if Q.width < 0.01882765 and Q.dr1_3 > 0.0717773:
        z += 161.3364 * (0.01882765 - Q.width) * (Q.dr1_3 - 0.0717773)
    if Q.sj3_pair_mass_max > 37.79615 and Q.n_pt_above_50 > 7.0:
        z += 0.02076158 * (Q.sj3_pair_mass_max - 37.79615) * (Q.n_pt_above_50 - 7.0)
    if Q.girth < 0.1019409 and Q.dr1_3 > 0.1637906:
        z += -129.3787 * (0.1019409 - Q.girth) * (Q.dr1_3 - 0.1637906)
    if Q.tau1 < 0.0262518 and Q.zdr_3 > 0.008191788:
        z += -178024.3 * (0.0262518 - Q.tau1) * (Q.zdr_3 - 0.008191788)
    if Q.width < 0.01882765 and Q.lam2 > 0.001130645:
        z += -75400.17 * (0.01882765 - Q.width) * (Q.lam2 - 0.001130645)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 2.0:
        z += 8.229638 * (Q.z_dr_0p05_0p1 - 0.7509095) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p05_0p1 > 0.6747704 and Q.pt_3 > 68.75:
        z += 0.03347788 * (Q.z_dr_0p05_0p1 - 0.6747704) * (Q.pt_3 - 68.75)
    if Q.z_dr_0p05_0p1 > 0.6747704 and Q.z_2nd < 0.238708:
        z += -12.93692 * (Q.z_dr_0p05_0p1 - 0.6747704) * (0.238708 - Q.z_2nd)
    if Q.mass > 45.7571 and Q.pt_7 < 23.21641:
        z += -0.00411618 * (Q.mass - 45.7571) * (23.21641 - Q.pt_7)
    if Q.z_dr_0p05_0p1 > 0.6747704 and Q.pair_mass_0_6 > 24.79428:
        z += -0.6193861 * (Q.z_dr_0p05_0p1 - 0.6747704) * (Q.pair_mass_0_6 - 24.79428)
    if Q.z_dr_0p05_0p1 > 0.6747704 and Q.dr1_6 > 0.1261715:
        z += 16.78914 * (Q.z_dr_0p05_0p1 - 0.6747704) * (Q.dr1_6 - 0.1261715)
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
