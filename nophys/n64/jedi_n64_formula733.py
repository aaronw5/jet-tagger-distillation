"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.4% (the network: 81.1%); same class as the network for 94.2% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
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
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_12         mass of particles 0 and 12 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.ptdr0_13               pT13 · ΔR(0, 13) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_pt               pT [GeV] of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.z_1                    pT of particle 1 / total pT
  Q.z_11                   pT of particle 11 / total pT
  Q.z_14                   pT of particle 14 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_z                pT share of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.absphi_0               |Δφ| of particle 0
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.soft10_dr              ΔR from the jet axis of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_dr               ΔR from the jet axis of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.eta_0                  Δη of particle 0
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
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
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
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
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
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
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
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_12=pair_mass(0, 12),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_particles=len(real),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_0=pt[0],
        pt_11=pt[11],
        ptdr0_13=pt[13] * math.sqrt(dist2(0, 13)) if pt[13] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft2_pt=softp(2, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        z_1=z[1],
        z_11=z[11],
        z_14=z[14],
        z_7=z[7],
        z_9=z[9],
        soft1_z=softp(1, 'z'),
        soft3_z=softp(3, 'z'),
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft8_z=softp(8, 'z'),
        sj3_z3=subjets(3)["z"][2],
        z_top10_slots=sum(pt[:10]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_mass=softdrop("mass"),
        absphi_0=abs(phi[0]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        soft10_dr=softp(10, 'dr'),
        soft3_dr=softp(3, 'dr'),
        eta_0=eta[0],
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
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
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
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
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
    z = 1.329237
    if Q.mass < 74.25181:
        z += 0.02803251 * Q.mass - 2.832677
    if 74.25181 <= Q.mass < 78.26182:
        z += -0.005814524 * Q.mass - 0.3194735
    if 78.26182 <= Q.mass < 87.36377:
        z += -0.1433175 * Q.mass + 10.44176
    if 87.36377 <= Q.mass < 91.03469:
        z += -0.06385451 * Q.mass + 3.499573
    if 91.03469 <= Q.mass < 92.85979:
        z += -0.09153781 * Q.mass + 6.019714
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.2479179 * Q.mass + 20.54113
    if Q.mass >= 101.0497:
        z += -0.2759504 * Q.mass + 23.37381
    if Q.sum_z_dr2_top20 < 0.005312783:
        z += 56.19551 * Q.sum_z_dr2_top20 - 0.4659162
    if 0.005312783 <= Q.sum_z_dr2_top20 < 0.006374178:
        z += 95.76799 * Q.sum_z_dr2_top20 - 0.6761562
    if 0.006374178 <= Q.sum_z_dr2_top20 < 0.007538019:
        z += 56.46306 * Q.sum_z_dr2_top20 - 0.4256196
    if Q.sum_pt < 972.0419:
        z += 0.009923603 * Q.sum_pt - 9.870999
    if 972.0419 <= Q.sum_pt < 1012.673:
        z += 0.005533717 * Q.sum_pt - 5.603845
    if Q.psi_0p3 >= 0.9956185:
        z += 42.72501 * Q.psi_0p3 - 42.53781
    if Q.log_sum_pt < 6.97212:
        z += -1.574309 * Q.log_sum_pt + 11.00381
    if 6.97212 <= Q.log_sum_pt < 6.98945:
        z += 3.539164 * Q.log_sum_pt - 24.64795
    if 6.98945 <= Q.log_sum_pt < 7.017258:
        z += -3.19576 * Q.log_sum_pt + 22.42547
    if Q.sum_z_dr2_top30 < 0.006363916:
        z += 116.747 * Q.sum_z_dr2_top30 - 0.6955406
    if 0.006363916 <= Q.sum_z_dr2_top30 < 0.007856958:
        z += -31.76569 * Q.sum_z_dr2_top30 + 0.2495817
    if Q.lam1 < 0.005913555:
        z += 75.71738 * Q.lam1 - 0.4477589
    if Q.mass_over_sum_pt_sq < 0.006938798:
        z += -652.7904 * Q.mass_over_sum_pt_sq + 4.788634
    if 0.006938798 <= Q.mass_over_sum_pt_sq < 0.007873266:
        z += -277.2198 * Q.mass_over_sum_pt_sq + 2.182625
    if Q.tau1 < 0.0705748:
        z += 19.43841 * Q.tau1 - 1.371862
    if Q.sum_zz_dr2 < 0.00616708:
        z += 360.8995 * Q.sum_zz_dr2 - 2.225696
    if 71.79516 <= Q.mass_top50 < 82.04491:
        z += 0.01483922 * Q.mass_top50 - 1.065384
    if 82.04491 <= Q.mass_top50 < 92.16545:
        z += 0.01111922 * Q.mass_top50 - 0.7601769
    if Q.mass_top50 >= 92.16545:
        z += -0.01428147 * Q.mass_top50 + 1.580889
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += 2.337345 * Q.z_dr_0p2_0p4 - 0.2132259
    if Q.z_top50_slots < 0.9906378:
        z += 16.72018 * Q.z_top50_slots - 16.56364
    if Q.sum_pt_top40 < 1069.671:
        z += 0.002703918 * Q.sum_pt_top40 - 2.892303
    if Q.mass_top40 < 80.89043:
        z += -0.004273375 * Q.mass_top40 + 0.2957215
    if 80.89043 <= Q.mass_top40 < 83.32554:
        z += 0.02051398 * Q.mass_top40 - 1.709338
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.03282475 * Q.n_dr_0p2_0p4 + 0.4923713
    if Q.sum_pt_top50 < 1156.659:
        z += -0.002503718 * Q.sum_pt_top50 + 2.895949
    if 0.1411617 <= Q.sj2_dr < 0.2232169:
        z += 1.442245 * Q.sj2_dr - 0.2035898
    if Q.sj2_dr >= 0.2232169:
        z += 0.01002395 * Q.sj2_dr + 0.1161062
    if Q.mass_top30 < 76.41544:
        z += -0.01274446 * Q.mass_top30 + 0.9738739
    if Q.sum_pt_top30 < 950.9324:
        z += -0.002518353 * Q.sum_pt_top30 + 2.394784
    if Q.e2 >= 0.05557149:
        z += -42.60326 * Q.e2 + 2.367527
    if Q.log_sum_pt < 6.98945 and Q.sum_pt_top50 > 959.0957:
        z += 0.07212632 * (6.98945 - Q.log_sum_pt) * (Q.sum_pt_top50 - 959.0957)
    if Q.mass > 78.26182 and Q.sj2_zsoft < 0.06909411:
        z += 2.875526 * (Q.mass - 78.26182) * (0.06909411 - Q.sj2_zsoft)
    if Q.mass_top50 > 71.79516 and Q.sj2_zsoft < 0.06909411:
        z += -1.52118 * (Q.mass_top50 - 71.79516) * (0.06909411 - Q.sj2_zsoft)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.476623
    if Q.n_particles >= 38.0:
        z += 0.05347822 * Q.n_particles - 2.032172
    if Q.log_sum_pt < 6.893714:
        z += 6.970881 * Q.log_sum_pt - 49.76718
    if 6.893714 <= Q.log_sum_pt < 6.910131:
        z += 33.75199 * Q.log_sum_pt - 234.3885
    if 6.910131 <= Q.log_sum_pt < 6.959294:
        z += 50.05092 * Q.log_sum_pt - 347.0162
    if 6.959294 <= Q.log_sum_pt < 6.98945:
        z += 28.28766 * Q.log_sum_pt - 195.5593
    if 6.98945 <= Q.log_sum_pt < 7.017258:
        z += 26.40647 * Q.log_sum_pt - 182.4108
    if 7.017258 <= Q.log_sum_pt < 7.139296:
        z += 17.12429 * Q.log_sum_pt - 117.2754
    if Q.log_sum_pt >= 7.139296:
        z += 10.15341 * Q.log_sum_pt - 67.50818
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.008298689 * Q.sum_pt_top50 + 7.959238
    if Q.psi_0p3 >= 0.9980008:
        z += -221.3777 * Q.psi_0p3 + 220.9351
    if Q.sum_pt_top2 < 689.25:
        z += -0.001804995 * Q.sum_pt_top2 + 1.244093
    if 0.9341838 <= Q.z_top30_slots < 0.9564984:
        z += -5.572363 * Q.z_top30_slots + 5.205612
    if Q.z_top30_slots >= 0.9564984:
        z += 1.963109 * Q.z_top30_slots - 2.002056
    if Q.mass_top20 < 47.88842:
        z += 0.0480317 * Q.mass_top20 - 2.300162
    if Q.sj3_mass1 < 32.50209:
        z += 0.006148308 * Q.sj3_mass1 - 0.1998328
    if Q.sum_pt_top40 < 1069.671:
        z += -0.006491763 * Q.sum_pt_top40 + 6.944051
    if Q.sum_z_dr2_top15 < 0.0007894752:
        z += -829.5843 * Q.sum_z_dr2_top15 + 0.9182966
    if 0.0007894752 <= Q.sum_z_dr2_top15 < 0.003270031:
        z += -106.1699 * Q.sum_z_dr2_top15 + 0.3471789
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.08133725 * Q.n_dr_0p2_0p4 - 0.6215606
    if 7.0 <= Q.n_dr_0p2_0p4 < 13.0:
        z += 0.008699975 * Q.n_dr_0p2_0p4 - 0.1130997
    if Q.M3 < 0.03187688:
        z += 9.073163 * Q.M3 - 0.2892242
    if Q.sj2_mass1 < 30.26161:
        z += 0.02352222 * Q.sj2_mass1 - 0.7118202
    if Q.pt_9 < 31.35938:
        z += 0.02021625 * Q.pt_9 - 0.6339689
    if Q.lam1 < 0.004673423:
        z += -67.01314 * Q.lam1 + 0.3131808
    if Q.mass < 89.74183:
        z += 0.007153721 * Q.mass - 1.152668
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.0360502 * Q.mass - 3.74589
    if 101.0497 <= Q.mass < 121.3913:
        z += 0.005064922 * Q.mass - 0.6148374
    if Q.z_dr_0_0p05 >= 0.8459004:
        z += -3.94686 * Q.z_dr_0_0p05 + 3.33865
    if Q.soft1_pt < 1.091797:
        z += -0.03086442 * Q.soft1_pt - 0.3254708
    if 1.091797 <= Q.soft1_pt < 1.521582:
        z += -0.7095375 * Q.soft1_pt + 0.4155023
    if 1.521582 <= Q.soft1_pt < 2.275391:
        z += 0.8810157 * Q.soft1_pt - 2.004655
    if Q.sum_pt < 1017.435:
        z += -0.009411938 * Q.sum_pt + 9.576033
    if Q.sum_pt_top30 >= 1191.938:
        z += 0.00377735 * Q.sum_pt_top30 - 4.502366
    if Q.mass_top30 < 80.24626:
        z += -0.01129191 * Q.mass_top30 + 0.9061336
    if Q.n_dr_0_0p05 < 10.0:
        z += -0.03358736 * Q.n_dr_0_0p05 + 0.3358736
    if Q.tau1 < 0.1751567:
        z += -10.24292 * Q.tau1 + 1.794117
    if Q.mass_top10 >= 31.33272:
        z += 0.001570336 * Q.mass_top10 - 0.0492029
    if Q.zdr_0 >= 0.001901263:
        z += -19.80114 * Q.zdr_0 + 0.03764718
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.05726286 * Q.n_dr_0p1_0p2 - 0.4581029
    if Q.sum_pt_top3 >= 353.0625:
        z += -0.0004443239 * Q.sum_pt_top3 + 0.1568741
    if Q.sum_z_dr2_top5 < 0.00100099:
        z += -685.2379 * Q.sum_z_dr2_top5 + 1.182172
    if 0.00100099 <= Q.sum_z_dr2_top5 < 0.008329695:
        z += -67.7139 * Q.sum_z_dr2_top5 + 0.5640361
    if Q.mass_top5 < 14.54404:
        z += 0.02291262 * Q.mass_top5 - 0.5263667
    if 14.54404 <= Q.mass_top5 < 68.43422:
        z += 0.00358367 * Q.mass_top5 - 0.2452457
    if Q.tau2 < 0.0795038:
        z += -3.61157 * Q.tau2 + 0.2871335
    if Q.sj3_dr_max >= 0.3483732:
        z += 0.8974448 * Q.sj3_dr_max - 0.3126457
    if Q.m012 < 50.3522:
        z += 0.006889051 * Q.m012 - 0.3468789
    if Q.sum_z_dr2_top20 < 0.001823079:
        z += -285.1965 * Q.sum_z_dr2_top20 + 0.5199358
    if Q.z_top50_slots >= 0.9586536:
        z += -19.49572 * Q.z_top50_slots + 18.68965
    if Q.z_7 < 0.03107249:
        z += 20.49297 * Q.z_7 - 0.6367676
    if Q.pt_11 < 33.78125:
        z += 0.008432962 * Q.pt_11 - 0.284876
    if Q.tau3 < 0.03209934:
        z += -11.30124 * Q.tau3 + 0.3627623
    if Q.z_top30_slots > 0.9341838 and Q.max_pair_mass > 13.04793:
        z += 0.4129016 * (Q.z_top30_slots - 0.9341838) * (Q.max_pair_mass - 13.04793)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.1119555:
        z += 0.1055617 * (Q.n_particles - 38.0) * (0.1119555 - Q.dr_0)
    if Q.mass_top20 < 47.88842 and Q.n_real_top40 > 29.0:
        z += 0.003652902 * (47.88842 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 38.0 and Q.soft1_pt < 2.275391:
        z += -0.01307197 * (Q.n_particles - 38.0) * (2.275391 - Q.soft1_pt)
    if Q.z_top30_slots > 0.9341838 and Q.C2 < 0.07279889:
        z += 160.7142 * (Q.z_top30_slots - 0.9341838) * (0.07279889 - Q.C2)
    if Q.sj3_mass1 < 32.50209 and Q.sj3_mass2 < 18.68222:
        z += -0.001557185 * (32.50209 - Q.sj3_mass1) * (18.68222 - Q.sj3_mass2)
    if Q.sum_z_dr2_top15 < 0.003270031 and Q.psi_0p3 > 0.9973959:
        z += 36703.34 * (0.003270031 - Q.sum_z_dr2_top15) * (Q.psi_0p3 - 0.9973959)
    if Q.M3 < 0.03187688 and Q.M2 > 0.05568888:
        z += -402.922 * (0.03187688 - Q.M3) * (Q.M2 - 0.05568888)
    if Q.n_particles > 38.0 and Q.dr_1 < 0.1611545:
        z += 0.1338161 * (Q.n_particles - 38.0) * (0.1611545 - Q.dr_1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 > 7.407874:
        z += 0.7246147 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_3 - 7.407874)
    if Q.pt_9 < 31.35938 and Q.pair_mass_0_12 < 15.47191:
        z += 0.0007512502 * (31.35938 - Q.pt_9) * (15.47191 - Q.pair_mass_0_12)
    if Q.n_particles > 38.0 and Q.zdr_2 < 0.01395978:
        z += 1.296759 * (Q.n_particles - 38.0) * (0.01395978 - Q.zdr_2)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.n_real_top30 < 30.0:
        z += 0.006952531 * (7.0 - Q.n_dr_0p2_0p4) * (30.0 - Q.n_real_top30)
    if Q.z_top30_slots > 0.9564984 and Q.ptdr0_4 > 4.470953:
        z += 0.9337184 * (Q.z_top30_slots - 0.9564984) * (Q.ptdr0_4 - 4.470953)
    if Q.tau1 < 0.1751567 and Q.eta_0 > -0.008410263:
        z += 40.34512 * (0.1751567 - Q.tau1) * (Q.eta_0 - -0.008410263)
    if Q.z_top30_slots > 0.9564984 and Q.sj3_mass3 < 7.880676:
        z += -0.5609 * (Q.z_top30_slots - 0.9564984) * (7.880676 - Q.sj3_mass3)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.04773477
    if Q.log_sum_pt >= 7.062574:
        z += -10.79046 * Q.log_sum_pt + 76.20845
    if Q.sum_pt < 1017.435:
        z += 0.001685314 * Q.sum_pt - 2.124407
    if 1017.435 <= Q.sum_pt < 1052.889:
        z += 0.01266914 * Q.sum_pt - 13.29974
    if 1052.889 <= Q.sum_pt < 1115.723:
        z += 0.005520707 * Q.sum_pt - 5.773223
    if 1115.723 <= Q.sum_pt < 1260.541:
        z += 0.001360316 * Q.sum_pt - 1.13138
    if Q.sum_pt >= 1260.541:
        z += -0.0003249976 * Q.sum_pt + 0.9930267
    if Q.n_particles < 51.0:
        z += 0.008663765 * Q.n_particles - 0.441852
    if Q.sum_pt_top50 >= 1156.659:
        z += 0.009057864 * Q.sum_pt_top50 - 10.47686
    if Q.mass < 78.26182:
        z += 0.01244974 * Q.mass - 0.5955438
    if 78.26182 <= Q.mass < 91.03469:
        z += 0.008338164 * Q.mass - 0.2737642
    if 91.03469 <= Q.mass < 92.85979:
        z += 0.03676566 * Q.mass - 2.861653
    if 92.85979 <= Q.mass < 121.3913:
        z += -0.0153424 * Q.mass + 1.977091
    if 121.3913 <= Q.mass < 143.7876:
        z += -0.00511948 * Q.mass + 0.7361178
    if Q.sum_pt_top40 < 1041.263:
        z += -0.004651235 * Q.sum_pt_top40 + 4.864995
    if 1041.263 <= Q.sum_pt_top40 < 1053.047:
        z += -0.001852994 * Q.sum_pt_top40 + 1.95129
    if Q.sum_pt_top30 < 996.8867:
        z += 0.001750538 * Q.sum_pt_top30 - 1.745088
    if Q.sum_pt_top15 < 1003.329:
        z += 0.0007427486 * Q.sum_pt_top15 - 0.7452213
    if Q.mass_top40 < 83.32554:
        z += -0.003312396 * Q.mass_top40 + 0.1381892
    if 83.32554 <= Q.mass_top40 < 132.4278:
        z += 0.002806751 * Q.mass_top40 - 0.371692
    if Q.lam1 < 0.006716737:
        z += 6.929775 * Q.lam1 - 0.04654547
    if Q.log_sum_pt > 6.903423 and Q.mass_top40 > 77.93668:
        z += -0.01948518 * (Q.log_sum_pt - 6.903423) * (Q.mass_top40 - 77.93668)
    if Q.log_sum_pt > 6.903423 and Q.sum_z_dr2_top15 < 0.02146578:
        z += 658.0181 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.sum_z_dr2_top15)
    if Q.log_sum_pt > 7.062574 and Q.sum_z_dr2_top30 > 0.008376291:
        z += 254.3999 * (Q.log_sum_pt - 7.062574) * (Q.sum_z_dr2_top30 - 0.008376291)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p3 > 0.9299135:
        z += -33.21518 * (Q.log_sum_pt - 6.903423) * (Q.psi_0p3 - 0.9299135)
    if Q.sum_pt_top50 > 988.4554 and Q.sum_z_dr2_top15 < 0.02146578:
        z += -0.3825116 * (Q.sum_pt_top50 - 988.4554) * (0.02146578 - Q.sum_z_dr2_top15)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.04657927
    if Q.lam2 < 0.0006154841:
        z += -761.5657 * Q.lam2 + 0.4687316
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.1059908 * Q.n_dr_0p2_0p4 + 0.5299538
    if Q.n_particles < 46.0:
        z += -0.03508156 * Q.n_particles + 1.613752
    if Q.tau21 < 0.3861957:
        z += 2.491133 * Q.tau21 - 0.962065
    if Q.mass_over_sum_pt < 0.08665515:
        z += -10.67884 * Q.mass_over_sum_pt + 0.9253768
    if Q.sum_z_dr2_top40 < 0.006026828:
        z += -120.5483 * Q.sum_z_dr2_top40 + 0.9681231
    if 0.006026828 <= Q.sum_z_dr2_top40 < 0.006259772:
        z += 6.156456 * Q.sum_z_dr2_top40 + 0.2044954
    if 0.006259772 <= Q.sum_z_dr2_top40 < 0.008840538:
        z += -94.17103 * Q.sum_z_dr2_top40 + 0.8325226
    if Q.mass_top50 < 79.21004:
        z += 0.005353533 * Q.mass_top50 - 0.4240535
    if Q.sum_z_dr2 < 0.009614971:
        z += 168.9805 * Q.sum_z_dr2 - 1.624742
    if Q.mass < 53.87362:
        z += 0.01016405 * Q.mass - 0.3017211
    if 53.87362 <= Q.mass < 92.85979:
        z += -0.006306156 * Q.mass + 0.5855883
    if Q.tau4 < 0.01626937:
        z += -40.26507 * Q.tau4 + 0.6550872
    if Q.max_dr < 0.2738063:
        z += 0.4554439 * Q.max_dr - 0.1247034
    if Q.psi_0p3 >= 0.9995915:
        z += -292.3557 * Q.psi_0p3 + 292.2362
    if Q.sum_z_dr2_top50 < 0.008124776:
        z += -72.53783 * Q.sum_z_dr2_top50 + 0.5893537
    if Q.mass_top40 < 80.89043:
        z += 0.0109921 * Q.mass_top40 - 0.8891559
    if Q.D2_b2 < 1.36316:
        z += -0.1759922 * Q.D2_b2 + 0.2399056
    if Q.tau3 < 0.02197187:
        z += 24.14864 * Q.tau3 - 0.5305907
    if Q.n_dr_0p2_0p4 < 5.0 and Q.sum_pt_top30 < 988.4375:
        z += -0.001180163 * (5.0 - Q.n_dr_0p2_0p4) * (988.4375 - Q.sum_pt_top30)
    if Q.n_particles < 46.0 and Q.mass_top15 > 57.87349:
        z += -0.0009531848 * (46.0 - Q.n_particles) * (Q.mass_top15 - 57.87349)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_dr_0p1_0p2 > 9.0:
        z += -0.01353619 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 9.0)
    if Q.n_particles < 46.0 and Q.sum_pt_top30 > 800.732:
        z += 2.720666e-05 * (46.0 - Q.n_particles) * (Q.sum_pt_top30 - 800.732)
    if Q.D2 < 2.410481 and Q.psi_0p3 > 0.9985421:
        z += 126.8803 * (2.410481 - Q.D2) * (Q.psi_0p3 - 0.9985421)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.psi_0p1 < 0.9538343:
        z += 0.8274907 * (5.0 - Q.n_dr_0p2_0p4) * (0.9538343 - Q.psi_0p1)
    if Q.tau21 < 0.3861957 and Q.lam1 > 0.007671243:
        z += -593.3423 * (0.3861957 - Q.tau21) * (Q.lam1 - 0.007671243)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.z_top50_slots > 0.978741:
        z += 2.163333 * (8.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.978741)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.zdr_4 < 0.006794973:
        z += -12.222 * (5.0 - Q.n_dr_0p2_0p4) * (0.006794973 - Q.zdr_4)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.n_dr_0_0p05 < 18.0:
        z += -0.004801266 * (8.0 - Q.n_dr_0p2_0p4) * (18.0 - Q.n_dr_0_0p05)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.z_dr_0p05_0p1 > 0.5106729:
        z += 0.3607846 * (5.0 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p05_0p1 - 0.5106729)
    if Q.n_dr_0p1_0p2 < 10.0 and Q.psi_0p2 > 0.9087063:
        z += 0.3186415 * (10.0 - Q.n_dr_0p1_0p2) * (Q.psi_0p2 - 0.9087063)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.301282
    if Q.mass_top15 < 69.02716:
        z += -0.007594384 * Q.mass_top15 + 0.5242188
    if Q.mass_top40 < 67.72643:
        z += 0.01964235 * Q.mass_top40 - 1.745987
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += -0.001678921 * Q.mass_top40 - 0.3019741
    if 83.32554 <= Q.mass_top40 < 94.64253:
        z += 0.0390449 * Q.mass_top40 - 3.695308
    if 6.567534e-05 <= Q.e3 < 0.0001841806:
        z += -1902.443 * Q.e3 + 0.1249436
    if Q.e3 >= 0.0001841806:
        z += -532.9241 * Q.e3 - 0.1272952
    if Q.mass < 74.25181:
        z += 0.04894416 * Q.mass - 2.824418
    if 74.25181 <= Q.mass < 79.65241:
        z += -0.01217771 * Q.mass + 1.713991
    if 79.65241 <= Q.mass < 87.36377:
        z += 0.02209692 * Q.mass - 1.016065
    if 87.36377 <= Q.mass < 92.85979:
        z += -0.04357822 * Q.mass + 4.721562
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.05145052 * Q.mass + 5.452583
    if 101.0497 <= Q.mass < 121.3913:
        z += -0.01246326 * Q.mass + 1.512931
    if Q.mass >= 143.7876:
        z += 0.02399023 * Q.mass - 3.449498
    if Q.psi_0p3 >= 0.9973959:
        z += 120.7114 * Q.psi_0p3 - 120.397
    if Q.mass_top30 < 121.737:
        z += 0.008442938 * Q.mass_top30 - 1.027818
    if Q.sj3_pair_mass_min >= 32.51366:
        z += 0.006345295 * Q.sj3_pair_mass_min - 0.2063088
    if Q.n_particles >= 22.0:
        z += -0.02687342 * Q.n_particles + 0.5912151
    if Q.sum_z_dr2_top15 < 0.004855289:
        z += 169.0971 * Q.sum_z_dr2_top15 - 0.8210154
    if Q.sum_z_dr2_top15 >= 0.007887677:
        z += -15.85981 * Q.sum_z_dr2_top15 + 0.1250971
    if Q.n_dr_0_0p05 >= 5.0:
        z += 0.02548545 * Q.n_dr_0_0p05 - 0.1274273
    if Q.sum_zz_dr2 >= 0.01396296:
        z += 93.31137 * Q.sum_zz_dr2 - 1.302903
    if Q.sum_pt < 1042.609:
        z += 0.002744935 * Q.sum_pt - 2.861894
    if Q.sum_z_dr2_top10 < 0.007678544:
        z += 53.51799 * Q.sum_z_dr2_top10 - 0.2827723
    if 0.007678544 <= Q.sum_z_dr2_top10 < 0.008956554:
        z += -100.2871 * Q.sum_z_dr2_top10 + 0.898227
    if Q.n_pt_above_1 < 48.0:
        z += -0.004982498 * Q.n_pt_above_1 + 0.2391599
    if Q.lam2 < 0.0004662705:
        z += 73.67007 * Q.lam2 - 0.1308608
    if 0.0004662705 <= Q.lam2 < 0.001776308:
        z += 127.8288 * Q.lam2 - 0.1561134
    if Q.lam2 >= 0.001776308:
        z += 54.15878 * Q.lam2 - 0.02525264
    if Q.e2 < 0.02515919:
        z += -14.2875 * Q.e2 + 0.3594621
    if Q.e2 >= 0.03680582:
        z += 20.31696 * Q.e2 - 0.7477824
    if Q.mass_top50 >= 138.8977:
        z += -0.0138945 * Q.mass_top50 + 1.929914
    if Q.mass_top20 >= 103.6749:
        z += -0.01557692 * Q.mass_top20 + 1.614936
    if Q.max_dr < 0.2982:
        z += -1.352002 * Q.max_dr + 0.403167
    if Q.D2 < 6.916121:
        z += -0.07951222 * Q.D2 + 0.5499161
    if 0.1512157 <= Q.sj2_dr < 0.1825048:
        z += 1.687988 * Q.sj2_dr - 0.2552502
    if Q.sj2_dr >= 0.1825048:
        z += -1.247716 * Q.sj2_dr + 0.28053
    if Q.n_dr_0p2_0p4 < 26.0:
        z += -0.01012375 * Q.n_dr_0p2_0p4 + 0.2632174
    if Q.LHA >= 0.3719813:
        z += -10.33935 * Q.LHA + 3.846044
    if Q.mass_over_sum_pt_sq < 0.008184368:
        z += -72.57204 * Q.mass_over_sum_pt_sq + 0.5939562
    if Q.sum_z_dr2_top30 < 0.005809485:
        z += 24.63909 * Q.sum_z_dr2_top30 - 0.1431404
    if Q.tau1 < 0.0705748:
        z += 12.26052 * Q.tau1 - 0.8652841
    if Q.sum_z_dr < 0.03577037:
        z += 29.80806 * Q.sum_z_dr - 1.066245
    if Q.mass_top40 < 83.32554 and Q.D2 < 6.916121:
        z += -0.01247615 * (83.32554 - Q.mass_top40) * (6.916121 - Q.D2)
    if Q.n_particles > 22.0 and Q.soft1_pt < 2.275391:
        z += 0.004973791 * (Q.n_particles - 22.0) * (2.275391 - Q.soft1_pt)
    if Q.mass < 87.36377 and Q.zdr_0 > 0.006864207:
        z += -3.497764 * (87.36377 - Q.mass) * (Q.zdr_0 - 0.006864207)
    if Q.mass < 79.65241 and Q.lam2 < 0.003687605:
        z += -25.5111 * (79.65241 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.mass_top30 < 121.737 and Q.sum_pt_top40 < 1053.047:
        z += 4.164693e-05 * (121.737 - Q.mass_top30) * (1053.047 - Q.sum_pt_top40)
    if Q.mass < 101.0497 and Q.lam2 < 0.003687605:
        z += 10.9081 * (101.0497 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.sum_z_dr2_top15 < 0.004855289 and Q.n_dr_0p05_0p1 > 2.0:
        z += 4.434026 * (0.004855289 - Q.sum_z_dr2_top15) * (Q.n_dr_0p05_0p1 - 2.0)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.9383817
    z += -0.02814672 * Q.n_particles + 1.80139
    if 0.08873143 <= Q.mass_over_sum_pt < 0.09046749:
        z += -41.43851 * Q.mass_over_sum_pt + 3.676898
    if 0.09046749 <= Q.mass_over_sum_pt < 0.1708801:
        z += -10.53713 * Q.mass_over_sum_pt + 0.8813277
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 139.6417 * Q.mass_over_sum_pt - 24.78125
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -14.15821 * Q.log_sum_pt + 97.83511
    if 6.920349 <= Q.log_sum_pt < 6.935549:
        z += -26.7888 * Q.log_sum_pt + 185.2431
    if 6.935549 <= Q.log_sum_pt < 6.98945:
        z += -21.89365 * Q.log_sum_pt + 151.2926
    if Q.log_sum_pt >= 6.98945:
        z += -15.75816 * Q.log_sum_pt + 108.4089
    if 907.9372 <= Q.sum_pt < 986.0565:
        z += 0.00472531 * Q.sum_pt - 4.290285
    if Q.sum_pt >= 986.0565:
        z += -0.01098777 * Q.sum_pt + 11.2037
    if Q.psi_0p3 >= 0.9973959:
        z += -146.5518 * Q.psi_0p3 + 146.1701
    if Q.sum_z_dr2_top50 >= 0.01951641:
        z += -30.56955 * Q.sum_z_dr2_top50 + 0.5966078
    if Q.sum_z_dr2 >= 0.004756928:
        z += -39.98833 * Q.sum_z_dr2 + 0.1902216
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += 0.01221411 * Q.sum_pt_top50 - 11.41093
    if Q.sum_pt_top50 >= 959.0957:
        z += 0.02084924 * Q.sum_pt_top50 - 19.69285
    if Q.n_pt_above_1 >= 28.0:
        z += 0.01413809 * Q.n_pt_above_1 - 0.3958664
    if Q.sum_pt_top3 < 787.6281:
        z += 0.0003614652 * Q.sum_pt_top3 - 0.2847002
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.04164889 * Q.n_dr_0p2_0p4 + 0.4581378
    if 994.2695 <= Q.sum_pt_top40 < 1024.942:
        z += 0.003811406 * Q.sum_pt_top40 - 3.789565
    if Q.sum_pt_top40 >= 1024.942:
        z += 0.0009770149 * Q.sum_pt_top40 - 0.8844773
    if Q.sum_pt_top30 >= 933.1875:
        z += 0.001929186 * Q.sum_pt_top30 - 1.800292
    if 0.2404747 <= Q.max_dr < 0.4357228:
        z += -3.633289 * Q.max_dr + 0.8737142
    if Q.max_dr >= 0.4357228:
        z += -0.7552052 * Q.max_dr - 0.3803327
    if Q.z_top30_slots >= 0.9048492:
        z += -6.313685 * Q.z_top30_slots + 5.712933
    if Q.mass_top40 < 111.2487:
        z += 0.007681762 * Q.mass_top40 - 1.152375
    if 111.2487 <= Q.mass_top40 < 150.0144:
        z += 0.003471143 * Q.mass_top40 - 0.6839495
    if Q.mass_top40 >= 150.0144:
        z += -0.004210619 * Q.mass_top40 + 0.4684257
    if Q.mass < 64.48544:
        z += -0.01105248 * Q.mass + 0.7127241
    if 87.36377 <= Q.mass < 91.03469:
        z += 0.03175864 * Q.mass - 2.774555
    if 91.03469 <= Q.mass < 162.8363:
        z += 0.002444189 * Q.mass - 0.1059226
    if 162.8363 <= Q.mass < 172.4888:
        z += -0.3771709 * Q.mass + 61.70921
    if Q.mass >= 172.4888:
        z += -0.4082988 * Q.mass + 67.07842
    if Q.z_dr_0_0p05 >= 0.9084912:
        z += 2.704312 * Q.z_dr_0_0p05 - 2.456844
    if Q.mass_top50 >= 157.5448:
        z += 0.03819692 * Q.mass_top50 - 6.017727
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -418.2579 * Q.mass_over_sum_pt_sq + 12.21314
    if 69.65633 <= Q.sd_mass < 83.30647:
        z += 0.03131479 * Q.sd_mass - 2.181273
    if Q.sd_mass >= 83.30647:
        z += -0.008311983 * Q.sd_mass + 1.119893
    if Q.tau21 < 0.5100475:
        z += 1.068492 * Q.tau21 - 0.5449816
    if 822.975 <= Q.sum_pt_top10 < 943.6922:
        z += 0.002762174 * Q.sum_pt_top10 - 2.2732
    if Q.sum_pt_top10 >= 943.6922:
        z += 0.0004516649 * Q.sum_pt_top10 - 0.09279077
    if Q.z_11 < 0.01375115:
        z += -91.77146 * Q.z_11 + 1.261963
    if Q.pt_11 < 14.14062:
        z += 0.07660055 * Q.pt_11 - 1.08318
    if Q.e2 < 0.04082832:
        z += -15.72194 * Q.e2 + 0.6419002
    if Q.tau1 < 0.1507173:
        z += 3.865019 * Q.tau1 - 0.5825252
    if Q.z_dr_0p2_0p4 < 0.03700182:
        z += 11.04814 * Q.z_dr_0p2_0p4 - 0.4088014
    if Q.n_dr_0p1_0p2 < 21.0:
        z += 0.01823735 * Q.n_dr_0p1_0p2 - 0.3829844
    if Q.mass_top10 >= 71.781:
        z += 0.00837706 * Q.mass_top10 - 0.6013138
    if Q.psi_0p1 < 0.707925:
        z += 0.8772169 * Q.psi_0p1 - 0.6210037
    if Q.sj3_mass1 < 23.45965:
        z += -0.02173805 * Q.sj3_mass1 + 0.5099671
    if Q.D2 < 1.976207:
        z += -0.2677617 * Q.D2 + 0.5291526
    if Q.sum_z_dr2_top30 < 0.007856958:
        z += 37.56042 * Q.sum_z_dr2_top30 - 0.2951107
    if Q.sum_pt_top20 >= 969.7891:
        z += 0.001297273 * Q.sum_pt_top20 - 1.258081
    if Q.z_top10_slots >= 0.7887522:
        z += -2.21493 * Q.z_top10_slots + 1.747031
    if Q.sum_pt_top15 >= 935.1043:
        z += -0.002387199 * Q.sum_pt_top15 + 2.23228
    if Q.soft3_z < 0.0008504382:
        z += -127.7339 * Q.soft3_z + 0.3501993
    if 0.0008504382 <= Q.soft3_z < 0.002741632:
        z += -498.8685 * Q.soft3_z + 0.6658264
    if Q.soft3_z >= 0.002741632:
        z += -371.1346 * Q.soft3_z + 0.3156271
    if Q.soft3_pt >= 0.3395996:
        z += 0.3469341 * Q.soft3_pt - 0.1178187
    if Q.n_real_top50 >= 22.0:
        z += -0.005161574 * Q.n_real_top50 + 0.1135546
    if Q.n_particles < 64.0 and Q.D2 < 2.178951:
        z += -0.01893676 * (64.0 - Q.n_particles) * (2.178951 - Q.D2)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 70828.55 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    if Q.psi_0p3 > 0.9973959 and Q.D2 < 3.345339:
        z += 53.97031 * (Q.psi_0p3 - 0.9973959) * (3.345339 - Q.D2)
    if Q.max_dr > 0.2404747 and Q.sj3_mass2 < 13.23436:
        z += 0.1009376 * (Q.max_dr - 0.2404747) * (13.23436 - Q.sj3_mass2)
    if Q.mass > 172.4888 and Q.soft3_dr < 0.161871:
        z += -2.62699 * (Q.mass - 172.4888) * (0.161871 - Q.soft3_dr)
    if Q.sum_pt_top40 > 994.2695 and Q.e4 < 5.8505e-08:
        z += -123757.8 * (Q.sum_pt_top40 - 994.2695) * (5.8505e-08 - Q.e4)
    return max(0.0, z)


def neuron_6(Q):
    z = -0.4587851
    if Q.mass_top50 < 71.79516:
        z += -0.02586292 * Q.mass_top50 + 1.856833
    if Q.mass < 82.85409:
        z += -0.02294609 * Q.mass + 0.925163
    if 82.85409 <= Q.mass < 89.74183:
        z += -0.000384408 * Q.mass - 0.9441644
    if 89.74183 <= Q.mass < 92.85979:
        z += 0.03763097 * Q.mass - 4.355734
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.1063902 * Q.mass - 10.74071
    if 101.0497 <= Q.mass < 121.3913:
        z += 0.02284063 * Q.mass - 2.298042
    if 121.3913 <= Q.mass < 143.7876:
        z += -0.009288347 * Q.mass + 1.602136
    if 143.7876 <= Q.mass < 172.4888:
        z += -0.04380854 * Q.mass + 6.565713
    if Q.mass >= 172.4888:
        z += -0.0345202 * Q.mass + 4.963577
    if 0.02793599 <= Q.e2 < 0.04755309:
        z += -57.77515 * Q.e2 + 1.614006
    if Q.e2 >= 0.04755309:
        z += 7.613251 * Q.e2 - 1.495414
    if Q.tau1 < 0.05444509:
        z += -8.35299 * Q.tau1 + 0.5271429
    if 0.05444509 <= Q.tau1 < 0.06310829:
        z += 1.107566 * Q.tau1 + 0.01206204
    if Q.tau1 >= 0.06310829:
        z += 9.460556 * Q.tau1 - 0.5150809
    if Q.sj3_mass1 >= 19.30204:
        z += -0.02018118 * Q.sj3_mass1 + 0.3895379
    if Q.lam2 < 0.003687605:
        z += -90.12588 * Q.lam2 + 0.3323486
    if Q.z_dr_0p1_0p2 < 0.2864926:
        z += -1.63149 * Q.z_dr_0p1_0p2 + 0.4674099
    if Q.D2 < 1.409617:
        z += 0.47711 * Q.D2 - 0.8188745
    if 1.409617 <= Q.D2 < 2.178951:
        z += 0.1902063 * Q.D2 - 0.4144501
    if Q.sj3_pair_mass_min < 76.60223:
        z += -0.004325169 * Q.sj3_pair_mass_min + 0.3313176
    if Q.lam1 < 0.007259287:
        z += -102.1767 * Q.lam1 + 0.5235158
    if 0.007259287 <= Q.lam1 < 0.01649354:
        z += 23.63097 * Q.lam1 - 0.3897584
    if Q.mass_over_sum_pt < 0.09795415:
        z += 38.09298 * Q.mass_over_sum_pt - 3.731365
    if Q.sum_zz_dr2 < 0.02580859:
        z += -187.6546 * Q.sum_zz_dr2 + 4.8431
    if Q.sum_z_dr2_top50 < 0.02550569:
        z += 85.24198 * Q.sum_z_dr2_top50 - 2.174156
    if Q.sum_z_dr2_top30 < 0.008376291:
        z += -35.75098 * Q.sum_z_dr2_top30 - 0.9452544
    if 0.008376291 <= Q.sum_z_dr2_top30 < 0.02412652:
        z += 79.02837 * Q.sum_z_dr2_top30 - 1.90668
    if Q.sum_z_dr2_top20 < 0.00287991:
        z += -94.78809 * Q.sum_z_dr2_top20 + 0.2729812
    if Q.sj2_mass1 < 51.2028:
        z += 0.002161336 * Q.sj2_mass1 + 0.009598962
    if 51.2028 <= Q.sj2_mass1 < 65.20727:
        z += -0.008587644 * Q.sj2_mass1 + 0.5599768
    if Q.soft10_dr >= 0.2075213:
        z += -0.1236451 * Q.soft10_dr + 0.025659
    if Q.n_dr_0p1_0p2 >= 17.0:
        z += 0.02500815 * Q.n_dr_0p1_0p2 - 0.4251385
    if Q.psi_0p3 >= 0.9924477:
        z += 23.92995 * Q.psi_0p3 - 23.74923
    if Q.psi_0p1 >= 0.7703036:
        z += -1.237747 * Q.psi_0p1 + 0.9534409
    if Q.mass_top10 >= 99.06678:
        z += 0.03496525 * Q.mass_top10 - 3.463895
    if Q.mass_top40 >= 94.64253:
        z += 0.01286934 * Q.mass_top40 - 1.217987
    if Q.mass < 121.3913 and Q.sum_pt_top50 < 1003.544:
        z += 7.053411e-05 * (121.3913 - Q.mass) * (1003.544 - Q.sum_pt_top50)
    if Q.e2 > 0.04755309 and Q.psi_0p3 > 0.9896594:
        z += -8207.327 * (Q.e2 - 0.04755309) * (Q.psi_0p3 - 0.9896594)
    if Q.tau1 < 0.06310829 and Q.lam2 > 0.0002104765:
        z += 10125.61 * (0.06310829 - Q.tau1) * (Q.lam2 - 0.0002104765)
    if Q.mass < 89.74183 and Q.sum_pt_top20 < 1129.275:
        z += -4.104137e-05 * (89.74183 - Q.mass) * (1129.275 - Q.sum_pt_top20)
    if Q.e2 > 0.04755309 and Q.z_14 > 0.01634243:
        z += -3565.669 * (Q.e2 - 0.04755309) * (Q.z_14 - 0.01634243)
    if Q.n_dr_0p1_0p2 > 33.0 and Q.n_dr_0_0p05 < 9.0:
        z += -0.003918195 * (Q.n_dr_0p1_0p2 - 33.0) * (9.0 - Q.n_dr_0_0p05)
    if Q.e2 > 0.02793599 and Q.ptdr0_13 < 6.06065:
        z += -1.135258 * (Q.e2 - 0.02793599) * (6.06065 - Q.ptdr0_13)
    if Q.e2 > 0.02793599 and Q.psi_0p3 > 0.9896594:
        z += 6137.783 * (Q.e2 - 0.02793599) * (Q.psi_0p3 - 0.9896594)
    if Q.sj2_mass1 < 65.20727 and Q.eccentricity > 0.3346888:
        z += 0.004916278 * (65.20727 - Q.sj2_mass1) * (Q.eccentricity - 0.3346888)
    return max(0.0, z)


def neuron_7(Q):
    z = 8.178256
    if Q.tau21_b2 < 0.2352054:
        z += -6.843447 * Q.tau21_b2 + 1.609616
    if Q.sum_z_dr2 < 0.006403325:
        z += -97.96401 * Q.sum_z_dr2 + 0.8366577
    if 0.006403325 <= Q.sum_z_dr2 < 0.007877041:
        z += -142.0642 * Q.sum_z_dr2 + 1.119046
    if Q.mass_over_sum_pt < 0.1182259:
        z += -13.36894 * Q.mass_over_sum_pt + 1.580555
    if Q.mass < 78.26182:
        z += -0.031021 * Q.mass + 1.727519
    if 78.26182 <= Q.mass < 82.85409:
        z += 0.05355665 * Q.mass - 4.891682
    if 82.85409 <= Q.mass < 91.03469:
        z += 0.06019503 * Q.mass - 5.441699
    if 91.03469 <= Q.mass < 92.85979:
        z += 0.2398358 * Q.mass - 21.79524
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.006587779 * Q.mass - 0.1358777
    if 101.0497 <= Q.mass < 121.3913:
        z += -0.02604594 * Q.mass + 3.16175
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.03500975 * Q.n_dr_0p2_0p4 + 0.2100585
    if Q.psi_0p3 >= 0.9980008:
        z += -343.8972 * Q.psi_0p3 + 343.2097
    z += -1.19585 * Q.log_sum_pt
    if Q.tau1 < 0.07708632:
        z += -11.90151 * Q.tau1 + 0.03350275
    if 0.07708632 <= Q.tau1 < 0.08786745:
        z += 17.68245 * Q.tau1 - 2.247016
    if 0.08786745 <= Q.tau1 < 0.1072713:
        z += 35.73032 * Q.tau1 - 3.832836
    if Q.LHA < 0.2601462:
        z += 0.1347857 * Q.LHA + 0.3948416
    if 0.2601462 <= Q.LHA < 0.2845608:
        z += -5.272233 * Q.LHA + 1.801457
    if 0.2845608 <= Q.LHA < 0.3098384:
        z += -17.82097 * Q.LHA + 5.372334
    if 0.3098384 <= Q.LHA < 0.3332345:
        z += 6.380743 * Q.LHA - 2.126284
    if Q.mass_top20 < 70.42121:
        z += -0.001665559 * Q.mass_top20 + 0.1509218
    if 70.42121 <= Q.mass_top20 < 73.35236:
        z += 0.04797421 * Q.mass_top20 - 3.344771
    if 73.35236 <= Q.mass_top20 < 90.4505:
        z += -0.01019124 * Q.mass_top20 + 0.9218024
    if Q.z_top20_slots >= 0.9102775:
        z += -4.825828 * Q.z_top20_slots + 4.392842
    if Q.sum_z_dr2_top50 < 0.003418057:
        z += 2204.13 * Q.sum_z_dr2_top50 - 7.533841
    if Q.lam1 < 0.006189818:
        z += 67.19871 * Q.lam1 - 0.7522451
    if 0.006189818 <= Q.lam1 < 0.008241985:
        z += 163.8742 * Q.lam1 - 1.350649
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += -275.3065 * Q.mass_over_sum_pt_sq + 2.641569
    if Q.lam1_plus_lam2 < 0.008190222:
        z += 407.309 * Q.lam1_plus_lam2 - 3.335951
    if Q.tau2 < 0.05936026:
        z += 7.536257 * Q.tau2 - 0.4473542
    if Q.sum_pt_top50 < 1245.697:
        z += 0.001592602 * Q.sum_pt_top50 - 1.983899
    if Q.sum_z_dr2_top30 < 0.006929741:
        z += 22.35728 * Q.sum_z_dr2_top30 - 0.2817602
    if 0.006929741 <= Q.sum_z_dr2_top30 < 0.008376291:
        z += 260.2263 * Q.sum_z_dr2_top30 - 1.930131
    if 0.008376291 <= Q.sum_z_dr2_top30 < 0.01215787:
        z += -66.00418 * Q.sum_z_dr2_top30 + 0.8024704
    if Q.n_pt_above_1 < 43.0:
        z += 0.02088029 * Q.n_pt_above_1 - 0.8978526
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1937688:
        z += -27.70637 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1937688)
    if Q.tau21_b2 < 0.2352054 and Q.mass_over_sum_pt_sq < 0.007873266:
        z += -885.8764 * (0.2352054 - Q.tau21_b2) * (0.007873266 - Q.mass_over_sum_pt_sq)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.sj2_dr < 0.1825048:
        z += -0.6429468 * (6.0 - Q.n_dr_0p2_0p4) * (0.1825048 - Q.sj2_dr)
    if Q.mass < 91.03469 and Q.psi_0p3 > 0.9299135:
        z += 0.136493 * (91.03469 - Q.mass) * (Q.psi_0p3 - 0.9299135)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.e3 < 7.876005e-05:
        z += -1521.502 * (6.0 - Q.n_dr_0p2_0p4) * (7.876005e-05 - Q.e3)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9638082:
        z += 3.092179 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.mass < 91.03469 and Q.psi_0p3 > 0.9638082:
        z += -4.352193 * (91.03469 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.tau21_b2 < 0.2352054 and Q.n_real_top50 > 34.0:
        z += -0.1609273 * (0.2352054 - Q.tau21_b2) * (Q.n_real_top50 - 34.0)
    if Q.mass < 91.03469 and Q.n_dr_0_0p05 < 6.0:
        z += -0.009998735 * (91.03469 - Q.mass) * (6.0 - Q.n_dr_0_0p05)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 1260.541:
        z += -0.007773371 * (0.2352054 - Q.tau21_b2) * (1260.541 - Q.sum_pt)
    if Q.mass < 91.03469 and Q.psi_0p3 < 0.9973959:
        z += 1.003168 * (91.03469 - Q.mass) * (0.9973959 - Q.psi_0p3)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9973959:
        z += 15.52142 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9973959)
    if Q.mass_over_sum_pt < 0.1182259 and Q.psi_0p3 > 0.9973959:
        z += 1994.333 * (0.1182259 - Q.mass_over_sum_pt) * (Q.psi_0p3 - 0.9973959)
    if Q.mass < 91.03469 and Q.psi_0p3 > 0.9973959:
        z += -82.72445 * (91.03469 - Q.mass) * (Q.psi_0p3 - 0.9973959)
    if Q.tau21_b2 < 0.2352054 and Q.pt2_over_pt0 < 0.3825328:
        z += -10.21671 * (0.2352054 - Q.tau21_b2) * (0.3825328 - Q.pt2_over_pt0)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9973959:
        z += 54.09666 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9973959)
    if Q.mass < 121.3913 and Q.sd_mass > 76.29481:
        z += 0.001891995 * (121.3913 - Q.mass) * (Q.sd_mass - 76.29481)
    if Q.mass < 91.03469 and Q.sd_mass > 55.96662:
        z += -0.001939301 * (91.03469 - Q.mass) * (Q.sd_mass - 55.96662)
    if Q.tau21_b2 < 0.2352054 and Q.psi_0p3 > 0.9299135:
        z += -19.24058 * (0.2352054 - Q.tau21_b2) * (Q.psi_0p3 - 0.9299135)
    if Q.tau21_b2 < 0.2352054 and Q.sj3_mass3 < 11.65938:
        z += 0.14025 * (0.2352054 - Q.tau21_b2) * (11.65938 - Q.sj3_mass3)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.soft1_pt < 2.275391:
        z += 0.04174408 * (6.0 - Q.n_dr_0p2_0p4) * (2.275391 - Q.soft1_pt)
    if Q.sum_z_dr2 < 0.007877041 and Q.M2 < 0.1134943:
        z += 2281.411 * (0.007877041 - Q.sum_z_dr2) * (0.1134943 - Q.M2)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.pt_4 > 60.03125:
        z += 0.003094764 * (6.0 - Q.n_dr_0p2_0p4) * (Q.pt_4 - 60.03125)
    if Q.mass_top20 < 73.35236 and Q.soft1_z < 0.002181998:
        z += 0.1739103 * (73.35236 - Q.mass_top20) * (0.002181998 - Q.soft1_z)
    if Q.tau21_b2 < 0.2352054 and Q.n_pt_above_50 < 7.0:
        z += 0.4597416 * (0.2352054 - Q.tau21_b2) * (7.0 - Q.n_pt_above_50)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.6369503
    if 0.07696632 <= Q.mass_over_sum_pt < 0.1182259:
        z += 25.26355 * Q.mass_over_sum_pt - 1.944443
    if Q.mass_over_sum_pt >= 0.1182259:
        z += 4.555559 * Q.mass_over_sum_pt + 0.5037794
    if Q.sum_z_dr2_top40 < 0.005196966:
        z += -21.88267 * Q.sum_z_dr2_top40 + 0.1450858
    if 0.005196966 <= Q.sum_z_dr2_top40 < 0.006026828:
        z += 121.7133 * Q.sum_z_dr2_top40 - 0.6011776
    if 0.006026828 <= Q.sum_z_dr2_top40 < 0.006630167:
        z += -139.8891 * Q.sum_z_dr2_top40 + 0.9754551
    if 0.006630167 <= Q.sum_z_dr2_top40 < 0.008031986:
        z += -118.0064 * Q.sum_z_dr2_top40 + 0.8303693
    if Q.sum_z_dr2_top40 >= 0.008031986:
        z += 8.029282 * Q.sum_z_dr2_top40 - 0.1819476
    if Q.n_dr_0p2_0p4 < 8.0:
        z += 0.02568329 * Q.n_dr_0p2_0p4 - 0.3852493
    if 8.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += 0.06652226 * Q.n_dr_0p2_0p4 - 0.7119611
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += 0.04083898 * Q.n_dr_0p2_0p4 - 0.3267118
    if 64.48544 <= Q.mass < 80.78464:
        z += 0.03114791 * Q.mass - 2.008587
    if 80.78464 <= Q.mass < 89.74183:
        z += 0.08608172 * Q.mass - 6.446395
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.09501441 * Q.mass - 7.248031
    if 101.0497 <= Q.mass < 121.3913:
        z += 0.001776577 * Q.mass + 2.173625
    if Q.mass >= 121.3913:
        z += -0.0290177 * Q.mass + 5.911782
    if Q.log_sum_pt < 6.811175:
        z += -3.853788 * Q.log_sum_pt + 27.35168
    if 6.811175 <= Q.log_sum_pt < 6.935549:
        z += -8.867228 * Q.log_sum_pt + 61.49909
    if Q.log_sum_pt >= 6.959294:
        z += 6.271717 * Q.log_sum_pt - 43.64672
    if Q.sum_pt_top40 < 858.8262:
        z += 0.001815432 * Q.sum_pt_top40 - 1.874262
    if 858.8262 <= Q.sum_pt_top40 < 1032.405:
        z += 0.0006900718 * Q.sum_pt_top40 - 0.9077727
    if Q.sum_pt_top40 >= 1032.405:
        z += -0.00112536 * Q.sum_pt_top40 + 0.9664888
    if Q.mass_top20 >= 90.4505:
        z += -0.003553003 * Q.mass_top20 + 0.3213709
    if Q.sum_z_dr2_top15 < 0.001319197:
        z += -157.7909 * Q.sum_z_dr2_top15 + 0.7381153
    if 0.001319197 <= Q.sum_z_dr2_top15 < 0.005383629:
        z += -120.7094 * Q.sum_z_dr2_top15 + 0.6891976
    if 0.005383629 <= Q.sum_z_dr2_top15 < 0.007887677:
        z += -158.6263 * Q.sum_z_dr2_top15 + 0.8933279
    if 0.007887677 <= Q.sum_z_dr2_top15 < 0.01563836:
        z += 114.679 * Q.sum_z_dr2_top15 - 1.262416
    if Q.sum_z_dr2_top15 >= 0.01563836:
        z += 37.08143 * Q.sum_z_dr2_top15 - 0.04891771
    if 43.66601 <= Q.mass_top50 < 80.35535:
        z += -0.0158193 * Q.mass_top50 + 0.6907656
    if 80.35535 <= Q.mass_top50 < 97.93004:
        z += -0.0616784 * Q.mass_top50 + 4.37579
    if Q.mass_top50 >= 97.93004:
        z += -0.01942131 * Q.mass_top50 + 0.2375514
    if 0.007872294 <= Q.sum_zz_dr2 < 0.009606007:
        z += 234.5049 * Q.sum_zz_dr2 - 1.846091
    if Q.sum_zz_dr2 >= 0.009606007:
        z += -190.6511 * Q.sum_zz_dr2 + 2.23796
    if Q.sj2_dr >= 0.2232169:
        z += 3.056049 * Q.sj2_dr - 0.6821617
    if Q.e2 >= 0.04755309:
        z += 35.10745 * Q.e2 - 1.669468
    if Q.sum_pt_top50 < 1003.544:
        z += -0.005840595 * Q.sum_pt_top50 + 5.861295
    if Q.sum_pt_top50 >= 1038.855:
        z += -0.003570282 * Q.sum_pt_top50 + 3.709006
    if Q.n_dr_0p1_0p2 >= 19.0:
        z += 0.01226779 * Q.n_dr_0p1_0p2 - 0.2330879
    if Q.tau1 >= 0.06310829:
        z += -11.20566 * Q.tau1 + 0.7071702
    if Q.n_pt_above_1 >= 54.0:
        z += 0.02766385 * Q.n_pt_above_1 - 1.493848
    if Q.sum_pt < 907.9372:
        z += 0.001659606 * Q.sum_pt - 1.506818
    if Q.z_dr_0p2_0p4 < 0.1291856:
        z += -3.158363 * Q.z_dr_0p2_0p4 + 0.408015
    if Q.LHA >= 0.3332345:
        z += 9.056623 * Q.LHA - 3.017979
    if Q.e3 < 0.0003372339:
        z += -3076.954 * Q.e3 + 1.037653
    if Q.sum_z_dr2_top20 < 0.02737453:
        z += 13.81576 * Q.sum_z_dr2_top20 - 0.3781999
    if Q.C2 >= 0.07996447:
        z += 2.898589 * Q.C2 - 0.2317841
    if Q.sum_z_dr2_top30 < 0.007856958:
        z += -111.8038 * Q.sum_z_dr2_top30 + 0.8784379
    if Q.mass_over_sum_pt_sq >= 0.001785266:
        z += 218.4776 * Q.mass_over_sum_pt_sq - 0.3900407
    if Q.lam1 >= 0.008241985:
        z += 32.51395 * Q.lam1 - 0.2679795
    if Q.mass_top30 < 80.24626:
        z += 0.01799207 * Q.mass_top30 - 1.443797
    if Q.sum_z_dr2_top50 < 0.0289594:
        z += 48.31149 * Q.sum_z_dr2_top50 - 1.399072
    if Q.sum_pt_top15 < 818.9605:
        z += -0.001581168 * Q.sum_pt_top15 + 1.294914
    if Q.log_sum_pt > 6.959294 and Q.sj3_pair_mass_max > 28.35435:
        z += 0.03846112 * (Q.log_sum_pt - 6.959294) * (Q.sj3_pair_mass_max - 28.35435)
    if Q.sum_z_dr2_top15 < 0.007887677 and Q.tau21_b2 < 0.6133424:
        z += -206.3535 * (0.007887677 - Q.sum_z_dr2_top15) * (0.6133424 - Q.tau21_b2)
    if Q.sum_pt_top40 < 1032.405 and Q.psi_0p3 > 0.9924477:
        z += -0.2066913 * (1032.405 - Q.sum_pt_top40) * (Q.psi_0p3 - 0.9924477)
    if Q.sum_pt_top40 < 1032.405 and Q.mass_top3 < 50.3522:
        z += -4.354616e-05 * (1032.405 - Q.sum_pt_top40) * (50.3522 - Q.mass_top3)
    if Q.n_dr_0p2_0p4 > 8.0 and Q.zdr_0 > 0.005911134:
        z += -1.109714 * (Q.n_dr_0p2_0p4 - 8.0) * (Q.zdr_0 - 0.005911134)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.2852587
    if Q.mass_top40 < 80.89043:
        z += -0.01386473 * Q.mass_top40 + 1.119524
    if 80.89043 <= Q.mass_top40 < 89.6788:
        z += -0.00798407 * Q.mass_top40 + 0.6438353
    if 89.6788 <= Q.mass_top40 < 132.4278:
        z += -0.01928828 * Q.mass_top40 + 1.657583
    if 132.4278 <= Q.mass_top40 < 163.2541:
        z += 0.02908953 * Q.mass_top40 - 4.748986
    if Q.sum_pt_top40 < 956.2133:
        z += -0.009361932 * Q.sum_pt_top40 + 8.952004
    if Q.sum_z_dr2_top40 < 0.00217213:
        z += -165.4548 * Q.sum_z_dr2_top40 + 0.7565038
    if 0.00217213 <= Q.sum_z_dr2_top40 < 0.006259772:
        z += -97.15003 * Q.sum_z_dr2_top40 + 0.6081371
    if Q.mass_top30 < 73.33139:
        z += 0.001722244 * Q.mass_top30 - 0.8996009
    if 73.33139 <= Q.mass_top30 < 121.737:
        z += 0.01830205 * Q.mass_top30 - 2.115421
    if 121.737 <= Q.mass_top30 < 152.6883:
        z += -0.003638504 * Q.mass_top30 + 0.5555569
    if Q.mass < 64.48544:
        z += -0.003016252 * Q.mass + 1.580516
    if 64.48544 <= Q.mass < 79.65241:
        z += -0.07254808 * Q.mass + 6.064306
    if 79.65241 <= Q.mass < 87.36377:
        z += -0.03704623 * Q.mass + 3.236498
    if 101.0497 <= Q.mass < 121.3913:
        z += 0.007756284 * Q.mass - 0.7837702
    if 121.3913 <= Q.mass < 143.7876:
        z += 0.003436532 * Q.mass - 0.25939
    if 143.7876 <= Q.mass < 162.8363:
        z += -0.05473627 * Q.mass + 8.105139
    if 162.8363 <= Q.mass < 172.4888:
        z += -0.1267752 * Q.mass + 19.83569
    if Q.mass >= 172.4888:
        z += -0.006081875 * Q.mass - 0.9825543
    if Q.sum_z_dr < 0.05048381:
        z += 19.18749 * Q.sum_z_dr - 0.9686577
    if Q.sum_z_dr >= 0.08589404:
        z += 27.45961 * Q.sum_z_dr - 2.358617
    if Q.psi_0p3 >= 0.9943058:
        z += -55.9995 * Q.psi_0p3 + 55.68063
    if Q.sum_pt_top50 < 959.0957:
        z += -0.00424091 * Q.sum_pt_top50 + 4.067439
    if Q.lam1 >= 0.01174405:
        z += 37.60107 * Q.lam1 - 0.4415888
    if Q.sum_z_dr2_top20 < 0.01083435:
        z += 14.97731 * Q.sum_z_dr2_top20 - 0.1622695
    if Q.sum_z_dr2_top30 < 0.002582316:
        z += 94.76405 * Q.sum_z_dr2_top30 + 0.2034312
    if 0.002582316 <= Q.sum_z_dr2_top30 < 0.01215787:
        z += -46.80062 * Q.sum_z_dr2_top30 + 0.568996
    if Q.sum_z_dr2_top30 >= 0.02412652:
        z += -165.0392 * Q.sum_z_dr2_top30 + 3.981821
    if Q.tau1 < 0.05444509:
        z += 32.2818 * Q.tau1 - 1.757586
    if Q.e3 >= 0.0003372339:
        z += 975.3366 * Q.e3 - 0.3289166
    if Q.LHA < 0.2091025:
        z += -15.06211 * Q.LHA + 3.149525
    if Q.n_dr_0p2_0p4 < 26.0:
        z += -0.007544837 * Q.n_dr_0p2_0p4 + 0.1961658
    if Q.z_dr_0p1_0p2 < 0.2187642:
        z += -1.136891 * Q.z_dr_0p1_0p2 + 0.2487111
    if Q.mass_over_sum_pt_sq < 0.003636595:
        z += -208.1876 * Q.mass_over_sum_pt_sq + 0.7570939
    if Q.sum_pt_top30 < 886.3438:
        z += 0.004762487 * Q.sum_pt_top30 - 4.221201
    if Q.sum_z_dr2_top50 < 0.00634935:
        z += -94.80389 * Q.sum_z_dr2_top50 + 0.601943
    if Q.sum_pt_top40 < 956.2133 and Q.soft4_pt > 1.789258:
        z += 0.004847651 * (956.2133 - Q.sum_pt_top40) * (Q.soft4_pt - 1.789258)
    if Q.mass_top40 < 80.89043 and Q.sum_pt < 1034.834:
        z += -0.0001559175 * (80.89043 - Q.mass_top40) * (1034.834 - Q.sum_pt)
    if Q.mass < 87.36377 and Q.n_particles > 38.0:
        z += 0.0007034334 * (87.36377 - Q.mass) * (Q.n_particles - 38.0)
    if Q.mass_top40 < 80.89043 and Q.sum_pt_top50 < 934.2416:
        z += 0.0002640878 * (80.89043 - Q.mass_top40) * (934.2416 - Q.sum_pt_top50)
    if Q.mass_top30 < 121.737 and Q.n_dr_0p2_0p4 > 10.0:
        z += 0.0007931241 * (121.737 - Q.mass_top30) * (Q.n_dr_0p2_0p4 - 10.0)
    if Q.sum_pt_top40 < 956.2133 and Q.n_pt_above_10 < 20.0:
        z += -0.0006105357 * (956.2133 - Q.sum_pt_top40) * (20.0 - Q.n_pt_above_10)
    if Q.sum_pt_top40 < 956.2133 and Q.soft5_pt > 2.894531:
        z += -0.005543031 * (956.2133 - Q.sum_pt_top40) * (Q.soft5_pt - 2.894531)
    if Q.sum_pt_top50 < 959.0957 and Q.soft4_pt > 1.375:
        z += -0.003330464 * (959.0957 - Q.sum_pt_top50) * (Q.soft4_pt - 1.375)
    if Q.sum_pt_top40 < 956.2133 and Q.soft5_z > 0.0005078411:
        z += -3.20921 * (956.2133 - Q.sum_pt_top40) * (Q.soft5_z - 0.0005078411)
    if Q.mass > 143.7876 and Q.z_dr_0p05_0p1 > 0.4026646:
        z += 0.06802351 * (Q.mass - 143.7876) * (Q.z_dr_0p05_0p1 - 0.4026646)
    if Q.mass_over_sum_pt > 0.1708801 and Q.soft4_pt > 1.789258:
        z += 13.11426 * (Q.mass_over_sum_pt - 0.1708801) * (Q.soft4_pt - 1.789258)
    if Q.sum_pt_top50 < 959.0957 and Q.soft5_z > 0.0009919684:
        z += 4.717852 * (959.0957 - Q.sum_pt_top50) * (Q.soft5_z - 0.0009919684)
    if Q.sum_z_dr > 0.08589404 and Q.soft5_z < 0.002036949:
        z += -23397.91 * (Q.sum_z_dr - 0.08589404) * (0.002036949 - Q.soft5_z)
    if Q.sum_z_dr > 0.09749958 and Q.soft5_z < 0.002036949:
        z += 29158.12 * (Q.sum_z_dr - 0.09749958) * (0.002036949 - Q.soft5_z)
    return max(0.0, z)


def neuron_10(Q):
    z = -2.121778
    if Q.sum_z_dr < 0.03577037:
        z += 25.734 * Q.sum_z_dr - 4.193844
    if 0.03577037 <= Q.sum_z_dr < 0.1207452:
        z += 38.52118 * Q.sum_z_dr - 4.651246
    if Q.mass < 64.48544:
        z += -0.003206267 * Q.mass + 0.2380712
    if 64.48544 <= Q.mass < 74.25181:
        z += -0.05474253 * Q.mass + 3.56141
    if 74.25181 <= Q.mass < 80.78464:
        z += -0.05153627 * Q.mass + 3.323339
    if 80.78464 <= Q.mass < 82.85409:
        z += 0.01296144 * Q.mass - 1.887086
    if 82.85409 <= Q.mass < 89.74183:
        z += -0.0008084988 * Q.mass - 0.7461895
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.06419372 * Q.mass - 6.579608
    if 101.0497 <= Q.mass < 143.7876:
        z += 0.03622916 * Q.mass - 3.753798
    if 143.7876 <= Q.mass < 162.8363:
        z += -0.07114248 * Q.mass + 11.68491
    if 162.8363 <= Q.mass < 172.4888:
        z += -0.1406846 * Q.mass + 23.0089
    if Q.mass >= 172.4888:
        z += -0.1124227 * Q.mass + 18.13403
    if Q.sum_z_dr2_top30 < 0.005402331:
        z += 23.39052 * Q.sum_z_dr2_top30 + 0.8389502
    if 0.005402331 <= Q.sum_z_dr2_top30 < 0.01807679:
        z += -76.16212 * Q.sum_z_dr2_top30 + 1.376767
    if Q.mass_top40 < 67.72643:
        z += -0.01979123 * Q.mass_top40 + 2.839692
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += -0.004978039 * Q.mass_top40 + 1.836447
    if 83.32554 <= Q.mass_top40 < 87.27603:
        z += -0.03618922 * Q.mass_top40 + 4.437136
    if 87.27603 <= Q.mass_top40 < 111.2487:
        z += -0.01266556 * Q.mass_top40 + 2.384084
    if 111.2487 <= Q.mass_top40 < 132.4278:
        z += -0.0007848712 * Q.mass_top40 + 1.062373
    if 132.4278 <= Q.mass_top40 < 163.2541:
        z += 0.01481319 * Q.mass_top40 - 1.003245
    if Q.mass_top40 >= 163.2541:
        z += -0.01694704 * Q.mass_top40 + 4.181745
    if Q.sj2_mass1 < 80.3008:
        z += 0.01996852 * Q.sj2_mass1 - 1.603488
    if Q.D2 < 3.814159:
        z += -0.2415466 * Q.D2 + 0.9212973
    if 0.9956185 <= Q.psi_0p3 < 0.9985421:
        z += -120.2042 * Q.psi_0p3 + 119.6775
    if Q.psi_0p3 >= 0.9985421:
        z += -234.8392 * Q.psi_0p3 + 234.1454
    if Q.mass_top5 >= 14.54404:
        z += 0.009309587 * Q.mass_top5 - 0.135399
    if Q.z_dr_0_0p05 >= 0.7674734:
        z += -0.9086289 * Q.z_dr_0_0p05 + 0.6973486
    if Q.log_sum_pt >= 6.903423:
        z += -6.899079 * Q.log_sum_pt + 47.62726
    z += 0.002297553 * Q.sum_pt
    if Q.mass_top15 < 75.26407:
        z += 0.00749197 * Q.mass_top15 - 0.5638762
    if Q.mass_over_sum_pt < 0.07852883:
        z += -39.33389 * Q.mass_over_sum_pt + 3.088844
    if Q.mass_over_sum_pt >= 0.1708801:
        z += -337.4191 * Q.mass_over_sum_pt + 57.65822
    if Q.sum_pt_top5 < 839.9547:
        z += -0.001520882 * Q.sum_pt_top5 + 1.277472
    if Q.e2 < 0.01541561:
        z += -7.958359 * Q.e2 - 1.747962
    if 0.01541561 <= Q.e2 < 0.04082832:
        z += 49.62188 * Q.e2 - 2.635597
    if 0.04082832 <= Q.e2 < 0.06524004:
        z += 24.9724 * Q.e2 - 1.6292
    if Q.dr_0 < 0.06413297:
        z += -5.16927 * Q.dr_0 + 0.3315206
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.0326028 * Q.n_dr_0p2_0p4 - 0.3586308
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -1.961109 * Q.z_dr_0p2_0p4 + 0.1789035
    if Q.mass_top10 < 71.781:
        z += 0.006558992 * Q.mass_top10 - 0.470811
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += 823.2454 * Q.mass_over_sum_pt_sq - 24.03878
    if Q.dr_1 < 0.06940472:
        z += -4.096335 * Q.dr_1 + 0.284305
    if Q.tau1 < 0.05444509:
        z += 29.57193 * Q.tau1 + 1.205224
    if 0.05444509 <= Q.tau1 < 0.1507173:
        z += -29.24282 * Q.tau1 + 4.407398
    if Q.tau21_b2 < 0.6133424:
        z += -1.29109 * Q.tau21_b2 + 0.7918804
    if Q.pt_entropy >= 2.07371:
        z += 0.4161573 * Q.pt_entropy - 0.8629895
    if Q.M2 >= 0.04577853:
        z += -5.199425 * Q.M2 + 0.238022
    if Q.lam1 >= 0.001260456:
        z += -31.85831 * Q.lam1 + 0.04015598
    if Q.sum_z_dr2_top15 < 0.009962397:
        z += -69.11924 * Q.sum_z_dr2_top15 + 0.6885933
    if Q.mass_top50 >= 138.8977:
        z += 0.04743551 * Q.mass_top50 - 6.588683
    if Q.sum_z_dr2_top20 < 0.005312783:
        z += 87.48795 * Q.sum_z_dr2_top20 - 0.4648045
    if Q.sj2_mass1 < 80.3008 and Q.sum_pt_top2 < 464.75:
        z += 2.067556e-05 * (80.3008 - Q.sj2_mass1) * (464.75 - Q.sum_pt_top2)
    if Q.D2 < 3.814159 and Q.sj2_dr > 0.1937688:
        z += -2.719028 * (3.814159 - Q.D2) * (Q.sj2_dr - 0.1937688)
    if Q.mass_top40 > 163.2541 and Q.soft4_z > 0.001186042:
        z += 30.03248 * (Q.mass_top40 - 163.2541) * (Q.soft4_z - 0.001186042)
    if Q.mass > 64.48544 and Q.soft2_pt < 2.537109:
        z += 0.003673302 * (Q.mass - 64.48544) * (2.537109 - Q.soft2_pt)
    if Q.mass > 82.85409 and Q.psi_0p3 > 0.9853273:
        z += 0.4915612 * (Q.mass - 82.85409) * (Q.psi_0p3 - 0.9853273)
    if Q.log_sum_pt > 6.903423 and Q.dr_4 < 0.0830523:
        z += 21.09224 * (Q.log_sum_pt - 6.903423) * (0.0830523 - Q.dr_4)
    if Q.mass_top40 < 111.2487 and Q.pt1_dr01 > 12.6865:
        z += 0.0008246318 * (111.2487 - Q.mass_top40) * (Q.pt1_dr01 - 12.6865)
    if Q.sj2_mass1 < 80.3008 and Q.sj2_mass2 > 11.91979:
        z += 0.0002440162 * (80.3008 - Q.sj2_mass1) * (Q.sj2_mass2 - 11.91979)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.4472023
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.1481302 * Q.n_dr_0p2_0p4 + 1.224073
    if 6.0 <= Q.n_dr_0p2_0p4 < 10.0:
        z += -0.0838228 * Q.n_dr_0p2_0p4 + 0.838228
    if Q.mass < 64.48544:
        z += -0.01822774 * Q.mass + 2.164784
    if 64.48544 <= Q.mass < 79.65241:
        z += -0.02274996 * Q.mass + 2.456401
    if 79.65241 <= Q.mass < 92.85979:
        z += -0.06085583 * Q.mass + 5.491625
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.01946718 * Q.mass - 1.967152
    if Q.sum_z_dr < 0.05048381:
        z += 8.824139 * Q.sum_z_dr - 1.009929
    if 0.05048381 <= Q.sum_z_dr < 0.07374472:
        z += 24.26616 * Q.sum_z_dr - 1.789501
    if Q.sum_z_dr2_top10 < 0.003213724:
        z += -215.8098 * Q.sum_z_dr2_top10 + 0.7992884
    if 0.003213724 <= Q.sum_z_dr2_top10 < 0.004752876:
        z += -187.991 * Q.sum_z_dr2_top10 + 0.7098866
    if 0.004752876 <= Q.sum_z_dr2_top10 < 0.007678544:
        z += 62.75877 * Q.sum_z_dr2_top10 - 0.481896
    if Q.lam1 < 0.006716737:
        z += 128.7991 * Q.lam1 - 0.8651096
    if Q.z_top30_slots >= 0.9734886:
        z += -8.931293 * Q.z_top30_slots + 8.694513
    if Q.mass_top10 < 38.52415:
        z += 0.004159255 * Q.mass_top10 + 0.283006
    if 38.52415 <= Q.mass_top10 < 60.5496:
        z += -0.02012389 * Q.mass_top10 + 1.218494
    if Q.sum_z_dr2_top30 < 0.006929741:
        z += 126.0624 * Q.sum_z_dr2_top30 - 0.87358
    if Q.sum_zz_dr2 < 0.00818374:
        z += -184.6791 * Q.sum_zz_dr2 + 1.511366
    if Q.LHA < 0.2845608:
        z += -6.175931 * Q.LHA + 1.628155
    if 0.2845608 <= Q.LHA < 0.2941033:
        z += 2.019833 * Q.LHA - 0.7040386
    if 0.2941033 <= Q.LHA < 0.3098384:
        z += 13.93599 * Q.LHA - 4.208619
    if 0.3098384 <= Q.LHA < 0.3203321:
        z += -10.41427 * Q.LHA + 3.336027
    if Q.n_dr_0p1_0p2 < 15.0:
        z += -0.02804206 * Q.n_dr_0p1_0p2 + 0.4206308
    if Q.z_dr_0p1_0p2 < 0.1548383:
        z += -1.10367 * Q.z_dr_0p1_0p2 + 0.1708905
    if Q.mass_top30 < 60.43821:
        z += -0.006524397 * Q.mass_top30 + 0.09969227
    if 60.43821 <= Q.mass_top30 < 86.25221:
        z += 0.0114136 * Q.mass_top30 - 0.9844479
    if Q.mass_top50 < 85.8667:
        z += 0.02142209 * Q.mass_top50 - 1.530809
    if 85.8667 <= Q.mass_top50 < 97.93004:
        z += -0.0255845 * Q.mass_top50 + 2.505491
    if Q.z_top10_slots >= 0.7271951:
        z += -2.119519 * Q.z_top10_slots + 1.541304
    if Q.z_dr_0_0p05 < 0.09573228:
        z += 0.9414234 * Q.z_dr_0_0p05 - 0.09012461
    if Q.e2 < 0.02793599:
        z += -39.327 * Q.e2 + 0.8823464
    if 0.02793599 <= Q.e2 < 0.04082832:
        z += 16.77684 * Q.e2 - 0.6849701
    if Q.tau1 < 0.08786745:
        z += 2.753143 * Q.tau1 - 0.0119384
    if 0.08786745 <= Q.tau1 < 0.1072713:
        z += -11.85197 * Q.tau1 + 1.271375
    if Q.tau21_b2 < 0.144845:
        z += -4.43348 * Q.tau21_b2 + 0.6421673
    if Q.psi_0p3 >= 0.9853273:
        z += 26.79124 * Q.psi_0p3 - 26.39814
    if Q.sum_z_dr2_top2 < 0.007639643:
        z += 42.88538 * Q.sum_z_dr2_top2 - 0.327629
    if Q.e3 < 4.204324e-05:
        z += 8671.716 * Q.e3 - 0.364587
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += 2.090978 * Q.z_dr_0p2_0p4 - 0.1432142
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_particles > 34.0:
        z += -0.002302134 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_particles - 34.0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.sum_z_dr2 < 0.005532208:
        z += -30.9411 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.sum_z_dr2)
    if Q.mass < 101.0497 and Q.mass_top10 > 38.52415:
        z += 0.001401552 * (101.0497 - Q.mass) * (Q.mass_top10 - 38.52415)
    if Q.mass < 79.65241 and Q.D2 < 3.814159:
        z += -0.006311452 * (79.65241 - Q.mass) * (3.814159 - Q.D2)
    if Q.z_top30_slots > 0.9734886 and Q.C2_b2 < 0.05112769:
        z += 225.8933 * (Q.z_top30_slots - 0.9734886) * (0.05112769 - Q.C2_b2)
    if Q.mass < 79.65241 and Q.z_11 > 0.01375115:
        z += -0.4136733 * (79.65241 - Q.mass) * (Q.z_11 - 0.01375115)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.4507959
    if Q.mass < 53.87362:
        z += -0.06365836 * Q.mass + 6.958297
    if 53.87362 <= Q.mass < 80.78464:
        z += -0.123716 * Q.mass + 10.19382
    if 80.78464 <= Q.mass < 82.85409:
        z += -0.05985502 * Q.mass + 5.034833
    if 82.85409 <= Q.mass < 87.36377:
        z += -0.01676382 * Q.mass + 1.46455
    if Q.mass_top40 < 77.93668:
        z += 0.02031367 * Q.mass_top40 - 1.725116
    if 77.93668 <= Q.mass_top40 < 94.64253:
        z += 0.008496176 * Q.mass_top40 - 0.8040996
    if Q.z_dr_0p1_0p2 >= 0.4479367:
        z += 1.677639 * Q.z_dr_0p1_0p2 - 0.751476
    if Q.sum_zz_dr2 < 0.00363788:
        z += 273.3461 * Q.sum_zz_dr2 - 0.9944005
    if Q.sum_z_dr2_top5 < 0.0004005745:
        z += -618.9094 * Q.sum_z_dr2_top5 + 0.2479193
    if Q.sum_pt < 972.0419:
        z += 0.005693591 * Q.sum_pt - 5.534409
    if Q.sum_pt >= 1260.541:
        z += -0.002189499 * Q.sum_pt + 2.759953
    if Q.mass_over_sum_pt < 0.06895248:
        z += 6.749752 * Q.mass_over_sum_pt - 0.4654121
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += -5.150239 * Q.z_dr_0p2_0p4 + 0.3527476
    if Q.mass < 87.36377 and Q.z_dr_0p1_0p2 < 0.06472584:
        z += -0.1805937 * (87.36377 - Q.mass) * (0.06472584 - Q.z_dr_0p1_0p2)
    if Q.mass < 87.36377 and Q.psi_0p3 < 0.9989733:
        z += -3.439254 * (87.36377 - Q.mass) * (0.9989733 - Q.psi_0p3)
    if Q.mass < 87.36377 and Q.z_dr_0p05_0p1 > 0.4026646:
        z += 0.07945523 * (87.36377 - Q.mass) * (Q.z_dr_0p05_0p1 - 0.4026646)
    if Q.sj3_pair_mass_max > 122.7494 and Q.sj2_mass1 < 65.20727:
        z += 0.0003134592 * (Q.sj3_pair_mass_max - 122.7494) * (65.20727 - Q.sj2_mass1)
    if Q.sj3_pair_mass_max > 122.7494 and Q.pt_0 < 435.25:
        z += 3.701571e-05 * (Q.sj3_pair_mass_max - 122.7494) * (435.25 - Q.pt_0)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.planar_flow < 0.3036026:
        z += 28.2128 * (Q.z_dr_0p1_0p2 - 0.4479367) * (0.3036026 - Q.planar_flow)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.sj3_z3 < 0.2502892:
        z += -6.907373 * (Q.z_dr_0p1_0p2 - 0.4479367) * (0.2502892 - Q.sj3_z3)
    if Q.sd_mass > 133.2575 and Q.sj3_mass1 < 32.50209:
        z += 0.001355291 * (Q.sd_mass - 133.2575) * (32.50209 - Q.sj3_mass1)
    if Q.sum_pt < 972.0419 and Q.C2_b2 < 0.05112769:
        z += -0.05206212 * (972.0419 - Q.sum_pt) * (0.05112769 - Q.C2_b2)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.lam2 < 0.001776308:
        z += -4154.213 * (Q.z_dr_0p1_0p2 - 0.4479367) * (0.001776308 - Q.lam2)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.186598
    if Q.sum_pt < 1012.673:
        z += 0.0385028 * Q.sum_pt - 39.54421
    if 1012.673 <= Q.sum_pt < 1085.125:
        z += 0.007639015 * Q.sum_pt - 8.289286
    if Q.sum_pt >= 1260.541:
        z += -0.005966563 * Q.sum_pt + 7.521096
    if 64.48544 <= Q.mass < 78.26182:
        z += 0.01490612 * Q.mass - 0.9612276
    if 78.26182 <= Q.mass < 143.7876:
        z += 0.02861203 * Q.mass - 2.033877
    if 143.7876 <= Q.mass < 172.4888:
        z += -0.1393891 * Q.mass + 22.1226
    if Q.mass >= 172.4888:
        z += -0.1446267 * Q.mass + 23.02602
    if Q.n_for_90pct < 14.0:
        z += 0.07017726 * Q.n_for_90pct - 0.9824817
    if Q.sum_pt_top40 < 1053.047:
        z += -0.005843776 * Q.sum_pt_top40 + 6.153773
    if 0.06030419 <= Q.mass_over_sum_pt < 0.07435617:
        z += -6.584227 * Q.mass_over_sum_pt + 0.3970564
    if 0.07435617 <= Q.mass_over_sum_pt < 0.1708801:
        z += -3.756844 * Q.mass_over_sum_pt + 0.1868231
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 70.55908 * Q.mass_over_sum_pt - 12.51229
    if 60.5496 <= Q.mass_top10 < 71.781:
        z += 0.004651368 * Q.mass_top10 - 0.2816385
    if Q.mass_top10 >= 71.781:
        z += -0.003581659 * Q.mass_top10 + 0.3093364
    if Q.sum_z_dr2_top40 < 0.02497133:
        z += -28.8316 * Q.sum_z_dr2_top40 + 0.7199635
    if Q.log_sum_pt < 6.879399:
        z += 3.273704 * Q.log_sum_pt - 22.52111
    if Q.log_sum_pt >= 7.139296:
        z += 8.692947 * Q.log_sum_pt - 62.06153
    if 97.93004 <= Q.mass_top50 < 138.8977:
        z += -0.03060306 * Q.mass_top50 + 2.996959
    if 138.8977 <= Q.mass_top50 < 157.5448:
        z += 0.05911136 * Q.mass_top50 - 9.464165
    if 157.5448 <= Q.mass_top50 < 168.9698:
        z += -0.03756455 * Q.mass_top50 + 5.766622
    if Q.mass_top50 >= 168.9698:
        z += -0.0585142 * Q.mass_top50 + 9.306481
    if Q.sum_pt_top20 < 889.8383:
        z += 0.003477606 * Q.sum_pt_top20 - 3.094507
    if Q.pt_entropy < 3.654944:
        z += 0.1356354 * Q.pt_entropy - 0.4957396
    if Q.sum_pt_top50 < 959.0957:
        z += -0.01546027 * Q.sum_pt_top50 + 15.35134
    if 959.0957 <= Q.sum_pt_top50 < 1008.935:
        z += -0.01050303 * Q.sum_pt_top50 + 10.59688
    if Q.psi_0p3 >= 0.9777125:
        z += 15.89641 * Q.psi_0p3 - 15.54212
    if Q.mass_top5 >= 37.76455:
        z += 0.008970304 * Q.mass_top5 - 0.3387595
    if Q.mass_top40 >= 150.0144:
        z += 0.07087705 * Q.mass_top40 - 10.63258
    if Q.tau1 < 0.08786745:
        z += -2.25827 * Q.tau1 + 0.1984284
    if Q.lam1 >= 0.01649354:
        z += 25.09048 * Q.lam1 - 0.413831
    if Q.n_for_90pct < 14.0 and Q.D2 < 3.345339:
        z += 0.03253024 * (14.0 - Q.n_for_90pct) * (3.345339 - Q.D2)
    if Q.log_sum_pt > 7.139296 and Q.C3 < 0.00317537:
        z += -1102.136 * (Q.log_sum_pt - 7.139296) * (0.00317537 - Q.C3)
    if Q.sum_pt < 1012.673 and Q.z_9 < 0.01833434:
        z += 1.204898 * (1012.673 - Q.sum_pt) * (0.01833434 - Q.z_9)
    if Q.mass > 172.4888 and Q.pt_9 < 41.4375:
        z += -0.003126461 * (Q.mass - 172.4888) * (41.4375 - Q.pt_9)
    if Q.sum_pt_top40 < 1053.047 and Q.D3 < 0.5260785:
        z += 0.004300216 * (1053.047 - Q.sum_pt_top40) * (0.5260785 - Q.D3)
    if Q.mass_top50 > 97.93004 and Q.soft3_pt < 2.873047:
        z += 0.002804205 * (Q.mass_top50 - 97.93004) * (2.873047 - Q.soft3_pt)
    if Q.log_sum_pt < 6.811175 and Q.eta_1 < -0.0892334:
        z += 55.97407 * (6.811175 - Q.log_sum_pt) * (-0.0892334 - Q.eta_1)
    if Q.mass > 172.4888 and Q.soft6_z < 0.002357824:
        z += 13.43803 * (Q.mass - 172.4888) * (0.002357824 - Q.soft6_z)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.5850222
    if Q.tau21_b2 < 0.342495:
        z += -2.276886 * Q.tau21_b2 + 0.779822
    if Q.mass < 82.85409:
        z += 0.1966472 * Q.mass - 16.74551
    if 82.85409 <= Q.mass < 91.03469:
        z += 0.09139626 * Q.mass - 8.025041
    if 91.03469 <= Q.mass < 121.3913:
        z += -0.009724045 * Q.mass + 1.180414
    if Q.n_dr_0p1_0p2 < 19.0:
        z += 0.02521467 * Q.n_dr_0p1_0p2 - 0.4790787
    if Q.e2 < 0.01256572:
        z += -17.11543 * Q.e2 + 1.154188
    if 0.01256572 <= Q.e2 < 0.02210818:
        z += -45.46576 * Q.e2 + 1.510431
    if 0.02210818 <= Q.e2 < 0.03029714:
        z += -62.0754 * Q.e2 + 1.87764
    if 0.03029714 <= Q.e2 < 0.03680582:
        z += -37.03592 * Q.e2 + 1.119015
    if Q.e2 >= 0.03680582:
        z += -16.60963 * Q.e2 + 0.3672088
    if Q.mass_over_sum_pt < 0.08873143:
        z += 11.29233 * Q.mass_over_sum_pt + 0.06092951
    if 0.08873143 <= Q.mass_over_sum_pt < 0.09046749:
        z += 64.6004 * Q.mass_over_sum_pt - 4.669172
    if 0.09046749 <= Q.mass_over_sum_pt < 0.09795415:
        z += -60.99477 * Q.mass_over_sum_pt + 6.693109
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -35.43929 * Q.mass_over_sum_pt + 4.189843
    if Q.sum_z_dr2_top20 < 0.006374178:
        z += 58.48844 * Q.sum_z_dr2_top20 - 0.8088097
    if 0.006374178 <= Q.sum_z_dr2_top20 < 0.01083435:
        z += 97.75265 * Q.sum_z_dr2_top20 - 1.059087
    if Q.mass_over_sum_pt_sq < 0.01986381:
        z += -147.1162 * Q.mass_over_sum_pt_sq + 2.922288
    if Q.sum_z_dr2_top3 < 0.001592178:
        z += 272.1646 * Q.sum_z_dr2_top3 - 0.4333345
    if Q.n_pt_above_10 >= 11.0:
        z += -0.03142961 * Q.n_pt_above_10 + 0.3457258
    if Q.sum_z_dr2_top40 < 0.01292642:
        z += 195.4661 * Q.sum_z_dr2_top40 - 2.902181
    if 0.01292642 <= Q.sum_z_dr2_top40 < 0.01897915:
        z += 62.0389 * Q.sum_z_dr2_top40 - 1.177445
    if Q.mass_top40 < 67.72643:
        z += -0.032363 * Q.mass_top40 + 2.19183
    if Q.e3 < 7.876005e-05:
        z += 4485.593 * Q.e3 - 0.3532855
    if Q.tau2 < 0.04142826:
        z += 11.54469 * Q.tau2 - 0.4782763
    if Q.tau1 < 0.0705748:
        z += 9.010538 * Q.tau1 - 0.7615609
    if 0.0705748 <= Q.tau1 < 0.1008497:
        z += 4.150105 * Q.tau1 - 0.4185368
    if Q.dr_3 < 0.03948167:
        z += -10.48527 * Q.dr_3 + 0.413976
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.008133935 * Q.sum_pt_top50 + 7.801222
    if Q.log_sum_pt >= 6.935549:
        z += 11.00118 * Q.log_sum_pt - 76.29925
    if Q.mass_top50 >= 157.5448:
        z += -0.0193906 * Q.mass_top50 + 3.054888
    if Q.sum_pt_top40 >= 906.6023:
        z += -0.000178468 * Q.sum_pt_top40 + 0.1617995
    if Q.sum_z_dr2_top10 < 0.007678544:
        z += -18.98864 * Q.sum_z_dr2_top10 + 0.02181197
    if 0.007678544 <= Q.sum_z_dr2_top10 < 0.008956554:
        z += 97.02047 * Q.sum_z_dr2_top10 - 0.868969
    if Q.N2 < 0.4226723 and Q.max_dr > 0.2738063:
        z += -11.23627 * (0.4226723 - Q.N2) * (Q.max_dr - 0.2738063)
    if Q.mass < 82.85409 and Q.n_for_50pct > 5.0:
        z += 0.01516575 * (82.85409 - Q.mass) * (Q.n_for_50pct - 5.0)
    if Q.tau21_b2 < 0.342495 and Q.z_1 < 0.1231747:
        z += -25.12611 * (0.342495 - Q.tau21_b2) * (0.1231747 - Q.z_1)
    if Q.tau21_b2 < 0.342495 and Q.C2_b2 < 0.01691783:
        z += 48.09975 * (0.342495 - Q.tau21_b2) * (0.01691783 - Q.C2_b2)
    if Q.psi_0p3 > 0.9924477 and Q.dr_3 < 0.0458688:
        z += -2235.214 * (Q.psi_0p3 - 0.9924477) * (0.0458688 - Q.dr_3)
    if Q.mass < 121.3913 and Q.zdr_3 > 0.0003281655:
        z += -0.581164 * (121.3913 - Q.mass) * (Q.zdr_3 - 0.0003281655)
    if Q.mass < 91.03469 and Q.soft8_z < 0.002405315:
        z += 4.235345 * (91.03469 - Q.mass) * (0.002405315 - Q.soft8_z)
    if Q.psi_0p3 > 0.9924477 and Q.pt_3 > 60.59375:
        z += 0.9250653 * (Q.psi_0p3 - 0.9924477) * (Q.pt_3 - 60.59375)
    if Q.psi_0p3 > 0.9924477 and Q.soft6_pt < 4.250195:
        z += 4.315505 * (Q.psi_0p3 - 0.9924477) * (4.250195 - Q.soft6_pt)
    if Q.psi_0p3 > 0.9924477 and Q.N3 < 1.059188:
        z += 92.37334 * (Q.psi_0p3 - 0.9924477) * (1.059188 - Q.N3)
    if Q.tau21_b2 < 0.342495 and Q.orientation_deg > -9.840088:
        z += -0.01876074 * (0.342495 - Q.tau21_b2) * (Q.orientation_deg - -9.840088)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.001156876
    if Q.z_dr_0_0p05 < 0.8459004:
        z += 0.5279584 * Q.z_dr_0_0p05 - 0.4466002
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -4.907053 * Q.z_dr_0p1_0p2 + 0.5905328
    if Q.sum_z_dr2_top5 < 0.002270363:
        z += 62.86881 * Q.sum_z_dr2_top5 + 0.4295971
    if 0.002270363 <= Q.sum_z_dr2_top5 < 0.008329695:
        z += -94.45466 * Q.sum_z_dr2_top5 + 0.7867785
    if Q.psi_0p1 >= 0.8976117:
        z += -1.594872 * Q.psi_0p1 + 1.431576
    if Q.sum_pt < 986.0565:
        z += 0.004870913 * Q.sum_pt - 4.997777
    if 986.0565 <= Q.sum_pt < 1002.379:
        z += 0.01193372 * Q.sum_pt - 11.9621
    if Q.log_sum_pt < 6.903423:
        z += 0.4065094 * Q.log_sum_pt - 2.001223
    if 6.903423 <= Q.log_sum_pt < 7.017258:
        z += -7.072381 * Q.log_sum_pt + 49.62872
    if Q.psi_0p3 >= 0.9896594:
        z += -33.46604 * Q.psi_0p3 + 33.11998
    if 0.1666442 <= Q.sj2_dr < 0.2232169:
        z += -0.2515694 * Q.sj2_dr + 0.04192258
    if Q.sj2_dr >= 0.2232169:
        z += 4.348646 * Q.sj2_dr - 0.9849231
    if Q.sum_z_dr2_top10 < 0.007678544:
        z += -53.40618 * Q.sum_z_dr2_top10 + 0.4100817
    if Q.lam1 < 0.004673423:
        z += -15.68062 * Q.lam1 - 0.3142659
    if 0.004673423 <= Q.lam1 < 0.006189818:
        z += 54.81099 * Q.lam1 - 0.6437031
    if 0.006189818 <= Q.lam1 < 0.01174405:
        z += 133.0179 * Q.lam1 - 1.12779
    if 0.01174405 <= Q.lam1 < 0.01649354:
        z += 78.20693 * Q.lam1 - 0.4840867
    if Q.lam1 >= 0.01649354:
        z += -300.0183 * Q.lam1 + 5.754187
    if Q.n_dr_0p1_0p2 < 13.0:
        z += -0.04635469 * Q.n_dr_0p1_0p2 + 0.602611
    if Q.lam2 < 0.001776308:
        z += -152.5099 * Q.lam2 + 0.2709047
    if Q.tau1 < 0.0705748:
        z += 23.55765 * Q.tau1 - 1.662577
    if Q.sum_pt_top20 < 869.693:
        z += 0.0003021413 * Q.sum_pt_top20 - 0.5681011
    if 869.693 <= Q.sum_pt_top20 < 1017.778:
        z += 0.00206186 * Q.sum_pt_top20 - 2.098516
    if Q.n_dr_0p2_0p4 >= 9.0:
        z += -0.02927075 * Q.n_dr_0p2_0p4 + 0.2634367
    if Q.sd_mass < 40.97891:
        z += 0.000998293 * Q.sd_mass + 0.393575
    if 40.97891 <= Q.sd_mass < 79.18312:
        z += -0.01137267 * Q.sd_mass + 0.9005238
    if Q.mass_top50 < 52.15154:
        z += 0.01394881 * Q.mass_top50 - 0.9850617
    if 52.15154 <= Q.mass_top50 < 71.79516:
        z += 0.01311418 * Q.mass_top50 - 0.9415346
    if Q.sum_pt_top30 < 1018.832:
        z += 0.001531233 * Q.sum_pt_top30 - 1.560069
    if Q.sum_z_dr < 0.07031778:
        z += -29.86563 * Q.sum_z_dr + 2.486736
    if 0.07031778 <= Q.sum_z_dr < 0.08068193:
        z += -37.30659 * Q.sum_z_dr + 3.009968
    if Q.mass_top30 < 68.28697:
        z += -0.01676982 * Q.mass_top30 + 1.445113
    if 68.28697 <= Q.mass_top30 < 80.24626:
        z += -0.02508114 * Q.mass_top30 + 2.012668
    if Q.e2 >= 0.03029714:
        z += -38.77775 * Q.e2 + 1.174855
    if Q.e3 < 0.0005178279:
        z += 853.1707 * Q.e3 - 0.4417955
    if Q.mass_over_sum_pt < 0.08329945:
        z += 3.384411 * Q.mass_over_sum_pt + 0.3837836
    if 0.08329945 <= Q.mass_over_sum_pt < 0.140939:
        z += -11.54941 * Q.mass_over_sum_pt + 1.627763
    if Q.LHA < 0.2091025:
        z += 6.595105 * Q.LHA - 2.292432
    if 0.2091025 <= Q.LHA < 0.302389:
        z += 9.791119 * Q.LHA - 2.960727
    if 0.2442262 <= Q.sj3_dr_max < 0.2841485:
        z += 2.747158 * Q.sj3_dr_max - 0.670928
    if 0.2841485 <= Q.sj3_dr_max < 0.3483732:
        z += 0.2648492 * Q.sj3_dr_max + 0.03441621
    if Q.sj3_dr_max >= 0.3483732:
        z += -2.576793 * Q.sj3_dr_max + 1.024368
    if Q.psi_0p2 >= 0.8706159:
        z += -3.623676 * Q.psi_0p2 + 3.15483
    if Q.sum_z_dr2_top30 < 0.01215787:
        z += 37.27765 * Q.sum_z_dr2_top30 - 0.4532169
    if Q.max_dr < 0.2982:
        z += 0.6668381 * Q.max_dr - 0.1988511
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 949.9169:
        z += 0.005220494 * (0.8459004 - Q.z_dr_0_0p05) * (949.9169 - Q.sum_pt)
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 1260.541:
        z += 0.001348044 * (0.8459004 - Q.z_dr_0_0p05) * (1260.541 - Q.sum_pt)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_dr_0p2_0p4 > 5.0:
        z += -0.1849968 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p2_0p4 - 5.0)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.sj3_mass1 > 5.112677:
        z += -1.567016 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.sj3_mass1 - 5.112677)
    if Q.n_dr_0p1_0p2 < 13.0 and Q.absphi_0 < 0.02970886:
        z += -0.7343185 * (13.0 - Q.n_dr_0p1_0p2) * (0.02970886 - Q.absphi_0)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.sj3_pairmin_over_m > 0.09540583:
        z += -94.53172 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.sj3_pairmin_over_m - 0.09540583)
    if Q.n_dr_0p1_0p2 < 13.0 and Q.n_dr_0p05_0p1 > 8.0:
        z += -0.001361928 * (13.0 - Q.n_dr_0p1_0p2) * (Q.n_dr_0p05_0p1 - 8.0)
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
