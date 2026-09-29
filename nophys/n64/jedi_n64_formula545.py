"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.4% (the network: 81.1%); same class as the network for 94.1% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_12         mass of particles 0 and 12 [GeV]
  Q.pair_mass_0_3          mass of particles 0 and 3 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
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
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.ptdr0_13               pT13 · ΔR(0, 13) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
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
  Q.z_6                    pT of particle 6 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_nremoved            number of branches removed by soft drop
  Q.absphi_13              |Δφ| of particle 13
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft10_dr0             ΔR between the hardest and the 10. softest real particle (0 if among the 15 hardest)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.soft10_dr              ΔR from the jet axis of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.eta_0                  Δη of particle 0
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
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
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
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
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_12=pair_mass(0, 12),
        pair_mass_0_3=pair_mass(0, 3),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top2=mass_of(2),
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
        n_particles=len(real),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_0=pt[0],
        pt_11=pt[11],
        ptdr0_13=pt[13] * math.sqrt(dist2(0, 13)) if pt[13] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
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
        z_6=z[6],
        z_9=z[9],
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft7_z=softp(7, 'z'),
        soft8_z=softp(8, 'z'),
        z_top10_slots=sum(pt[:10]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        zdr_5=z[5] * dr[5],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sd_nremoved=softdrop("removed"),
        absphi_13=abs(phi[13]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft10_dr0=softp(10, 'dr0'),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        soft10_dr=softp(10, 'dr'),
        eta_0=eta[0],
        sum_pt_top10=sum(pt[:10]),
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
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
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
    )


def neuron_0(Q):
    z = 1.344566
    if Q.mass < 74.25181:
        z += 0.03045699 * Q.mass - 3.07767
    if 74.25181 <= Q.mass < 78.26182:
        z += -0.003960004 * Q.mass - 0.522146
    if 78.26182 <= Q.mass < 87.36377:
        z += -0.1268472 * Q.mass + 9.09523
    if 87.36377 <= Q.mass < 91.03469:
        z += -0.102688 * Q.mass + 6.984593
    if 91.03469 <= Q.mass < 92.85979:
        z += -0.1498395 * Q.mass + 11.27702
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.2984942 * Q.mass + 25.08105
    if Q.mass >= 101.0497:
        z += -0.3289511 * Q.mass + 28.15872
    if Q.sum_z_dr2_top20 < 0.005312783:
        z += 35.14446 * Q.sum_z_dr2_top20 - 0.4148194
    if 0.005312783 <= Q.sum_z_dr2_top20 < 0.006374178:
        z += 133.7478 * Q.sum_z_dr2_top20 - 0.9386778
    if 0.006374178 <= Q.sum_z_dr2_top20 < 0.007538019:
        z += 74.01811 * Q.sum_z_dr2_top20 - 0.5579499
    if Q.sum_pt < 1012.673:
        z += 0.008056752 * Q.sum_pt - 8.158855
    if Q.psi_0p3 >= 0.9956185:
        z += 68.57764 * Q.psi_0p3 - 68.27717
    if Q.mass_top30 < 80.24626:
        z += -0.01592792 * Q.mass_top30 + 1.278156
    if Q.lam1 < 0.005913555:
        z += 63.18221 * Q.lam1 - 0.3736315
    if Q.mass_over_sum_pt_sq < 0.006938798:
        z += -719.06 * Q.mass_over_sum_pt_sq + 5.315169
    if 0.006938798 <= Q.mass_over_sum_pt_sq < 0.007873266:
        z += -348.6013 * Q.mass_over_sum_pt_sq + 2.744631
    if Q.tau1 < 0.0705748:
        z += 17.57021 * Q.tau1 - 1.240014
    if Q.sum_zz_dr2 < 0.00616708:
        z += 448.8376 * Q.sum_zz_dr2 - 2.768017
    if 71.79516 <= Q.mass_top50 < 82.04491:
        z += 0.01296883 * Q.mass_top50 - 0.9310989
    if Q.mass_top50 >= 82.04491:
        z += 0.02963363 * Q.mass_top50 - 2.298361
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += 3.469101 * Q.z_dr_0p2_0p4 - 0.3164711
    if Q.z_top50_slots < 0.9906378:
        z += 17.27419 * Q.z_top50_slots - 17.11247
    if Q.sum_z_dr2_top30 < 0.006363916:
        z += 167.6191 * Q.sum_z_dr2_top30 - 1.066714
    if Q.log_sum_pt < 7.017258:
        z += -1.676685 * Q.log_sum_pt + 11.76573
    if Q.sum_pt_top40 < 1069.671:
        z += 0.003132379 * Q.sum_pt_top40 - 3.350616
    if Q.sum_pt_top20 < 846.1934:
        z += -0.001705868 * Q.sum_pt_top20 + 1.443494
    if Q.mass_top40 < 80.89043:
        z += -0.006808903 * Q.mass_top40 + 0.5507751
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.03067002 * Q.n_dr_0p2_0p4 + 0.4600503
    if Q.sum_pt_top50 < 1156.659:
        z += -0.002540558 * Q.sum_pt_top50 + 2.938561
    if Q.sum_pt < 1012.673 and Q.M3 < 0.03457336:
        z += -0.1107082 * (1012.673 - Q.sum_pt) * (0.03457336 - Q.M3)
    if Q.log_sum_pt < 6.98945 and Q.sum_pt_top50 > 959.0957:
        z += 0.06134462 * (6.98945 - Q.log_sum_pt) * (Q.sum_pt_top50 - 959.0957)
    if Q.sum_pt < 1012.673 and Q.C3 < 0.0223982:
        z += 0.1153166 * (1012.673 - Q.sum_pt) * (0.0223982 - Q.C3)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.4056649
    if Q.n_particles >= 38.0:
        z += 0.1140403 * Q.n_particles - 4.333531
    if Q.log_sum_pt < 6.893714:
        z += 8.180573 * Q.log_sum_pt - 58.40353
    if 6.893714 <= Q.log_sum_pt < 6.910131:
        z += 37.70123 * Q.log_sum_pt - 261.9105
    if 6.910131 <= Q.log_sum_pt < 6.959294:
        z += 50.98366 * Q.log_sum_pt - 353.6938
    if 6.959294 <= Q.log_sum_pt < 6.98945:
        z += 31.83133 * Q.log_sum_pt - 220.4071
    if 6.98945 <= Q.log_sum_pt < 7.139296:
        z += 18.89994 * Q.log_sum_pt - 130.0238
    if Q.log_sum_pt >= 7.139296:
        z += 10.71937 * Q.log_sum_pt - 71.62029
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.008938069 * Q.sum_pt_top50 + 8.572463
    if Q.psi_0p3 >= 0.9980008:
        z += -163.8747 * Q.psi_0p3 + 163.5471
    if Q.sum_pt_top2 < 689.25:
        z += -0.002081384 * Q.sum_pt_top2 + 1.434594
    if 0.9341838 <= Q.z_top30_slots < 0.9564984:
        z += -5.622614 * Q.z_top30_slots + 5.252555
    if Q.z_top30_slots >= 0.9564984:
        z += 0.6692009 * Q.z_top30_slots - 0.7655557
    if Q.mass_top20 < 47.88842:
        z += 0.04089765 * Q.mass_top20 - 1.958524
    if Q.sum_pt_top40 < 1069.671:
        z += -0.006226552 * Q.sum_pt_top40 + 6.660363
    if Q.sum_z_dr2_top15 < 0.0007894752:
        z += -717.9437 * Q.sum_z_dr2_top15 + 0.8392496
    if 0.0007894752 <= Q.sum_z_dr2_top15 < 0.003270031:
        z += -109.8346 * Q.sum_z_dr2_top15 + 0.3591625
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.08756057 * Q.n_dr_0p2_0p4 - 0.612924
    if Q.sum_z_dr2_top3 < 0.0005522528:
        z += -596.8324 * Q.sum_z_dr2_top3 + 0.3296024
    if Q.M3 < 0.03187688:
        z += 17.21979 * Q.M3 - 0.5489134
    if Q.sj2_mass1 < 30.26161:
        z += 0.02674561 * Q.sj2_mass1 - 0.8093652
    if Q.pt_9 < 31.35938:
        z += 0.03432387 * Q.pt_9 - 1.076375
    if Q.lam1 < 0.004673423:
        z += -107.6739 * Q.lam1 + 0.5032058
    if Q.mass < 89.74183:
        z += 0.008405831 * Q.mass - 1.103671
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.03089143 * Q.mass - 3.12157
    if Q.D3 < 0.08822608:
        z += -1.967991 * Q.D3 + 0.1736282
    if Q.z_dr_0_0p05 >= 0.8459004:
        z += -4.456728 * Q.z_dr_0_0p05 + 3.769947
    if Q.soft1_pt < 1.521582:
        z += -0.4584168 * Q.soft1_pt + 0.6975188
    if Q.sj2_zsoft >= 0.2912328:
        z += 1.377801 * Q.sj2_zsoft - 0.4012609
    if Q.sum_pt < 1017.435:
        z += -0.01136718 * Q.sum_pt + 11.56536
    if Q.sum_pt_top30 >= 1191.938:
        z += 0.003956569 * Q.sum_pt_top30 - 4.715984
    if Q.mass_top30 < 80.24626:
        z += -0.01230519 * Q.mass_top30 + 0.9874454
    if Q.n_dr_0_0p05 < 10.0:
        z += -0.02558681 * Q.n_dr_0_0p05 + 0.2558681
    if Q.tau1 < 0.1751567:
        z += -11.87894 * Q.tau1 + 2.080677
    if Q.mass_top10 >= 31.33272:
        z += 0.004992259 * Q.mass_top10 - 0.156421
    if Q.zdr_0 >= 0.001901263:
        z += -15.39291 * Q.zdr_0 + 0.02926597
    if Q.z_top30_slots > 0.9341838 and Q.max_pair_mass > 13.04793:
        z += 0.4723904 * (Q.z_top30_slots - 0.9341838) * (Q.max_pair_mass - 13.04793)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.1119555:
        z += 0.1764314 * (Q.n_particles - 38.0) * (0.1119555 - Q.dr_0)
    if Q.mass_top20 < 47.88842 and Q.n_real_top40 > 29.0:
        z += 0.004162062 * (47.88842 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 38.0 and Q.soft1_pt < 2.275391:
        z += -0.03961573 * (Q.n_particles - 38.0) * (2.275391 - Q.soft1_pt)
    if Q.z_top30_slots > 0.9341838 and Q.C2 < 0.07279889:
        z += 205.2069 * (Q.z_top30_slots - 0.9341838) * (0.07279889 - Q.C2)
    if Q.sj3_mass1 < 32.50209 and Q.sj3_mass2 < 18.68222:
        z += -0.001529776 * (32.50209 - Q.sj3_mass1) * (18.68222 - Q.sj3_mass2)
    if Q.sum_z_dr2_top15 < 0.003270031 and Q.psi_0p3 > 0.9973959:
        z += 45202.52 * (0.003270031 - Q.sum_z_dr2_top15) * (Q.psi_0p3 - 0.9973959)
    if Q.M3 < 0.03187688 and Q.M2 > 0.05568888:
        z += -344.7084 * (0.03187688 - Q.M3) * (Q.M2 - 0.05568888)
    if Q.n_particles > 38.0 and Q.dr_1 < 0.1611545:
        z += 0.1269012 * (Q.n_particles - 38.0) * (0.1611545 - Q.dr_1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 > 7.407874:
        z += 0.6182076 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_3 - 7.407874)
    if Q.pt_9 < 31.35938 and Q.pair_mass_0_12 < 15.47191:
        z += 0.0009382492 * (31.35938 - Q.pt_9) * (15.47191 - Q.pair_mass_0_12)
    if Q.n_particles > 38.0 and Q.zdr_2 < 0.01395978:
        z += 1.071058 * (Q.n_particles - 38.0) * (0.01395978 - Q.zdr_2)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.n_real_top30 < 30.0:
        z += 0.007336965 * (7.0 - Q.n_dr_0p2_0p4) * (30.0 - Q.n_real_top30)
    if Q.z_top30_slots > 0.9564984 and Q.ptdr0_4 > 4.470953:
        z += 0.8098888 * (Q.z_top30_slots - 0.9564984) * (Q.ptdr0_4 - 4.470953)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.002347209
    if Q.log_sum_pt >= 7.062574:
        z += -9.378196 * Q.log_sum_pt + 66.2342
    if Q.sum_pt < 1017.435:
        z += 0.002069385 * Q.sum_pt - 2.608545
    if 1017.435 <= Q.sum_pt < 1052.889:
        z += 0.01508284 * Q.sum_pt - 15.84889
    if 1052.889 <= Q.sum_pt < 1115.723:
        z += 0.004044013 * Q.sum_pt - 4.22622
    if 1115.723 <= Q.sum_pt < 1260.541:
        z += -0.0008738795 * Q.sum_pt + 1.260784
    if Q.sum_pt >= 1260.541:
        z += -0.002943265 * Q.sum_pt + 3.869329
    if Q.n_particles < 51.0:
        z += 0.008638248 * Q.n_particles - 0.4405506
    if Q.sum_pt_top50 >= 1156.659:
        z += 0.009690578 * Q.sum_pt_top50 - 11.2087
    if Q.mass < 91.03469:
        z += 0.006085937 * Q.mass + 0.06143469
    if 91.03469 <= Q.mass < 92.85979:
        z += 0.03363195 * Q.mass - 2.446208
    if 92.85979 <= Q.mass < 121.3913:
        z += -0.01998243 * Q.mass + 2.532412
    if 121.3913 <= Q.mass < 143.7876:
        z += -0.00476503 * Q.mass + 0.6851523
    if Q.mass_top30 < 82.66587:
        z += 0.005048876 * Q.mass_top30 - 0.4173697
    if Q.sum_pt_top40 < 1041.263:
        z += -0.004640943 * Q.sum_pt_top40 + 4.863952
    if 1041.263 <= Q.sum_pt_top40 < 1053.047:
        z += -0.002673855 * Q.sum_pt_top40 + 2.815696
    if Q.sum_pt_top30 < 996.8867:
        z += 0.001583572 * Q.sum_pt_top30 - 1.578642
    if Q.sum_z_dr2_top30 < 0.006088416:
        z += -126.6648 * Q.sum_z_dr2_top30 + 0.7711881
    if Q.sum_pt_top3 < 787.6281:
        z += 0.0005981218 * Q.sum_pt_top3 - 0.4710976
    if Q.sum_z_dr2_top50 < 0.008124776:
        z += 105.2892 * Q.sum_z_dr2_top50 - 0.8554515
    if Q.log_sum_pt > 6.903423 and Q.sum_z_dr2_top15 < 0.02146578:
        z += 699.1433 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.sum_z_dr2_top15)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p3 > 0.9299135:
        z += -21.88492 * (Q.log_sum_pt - 6.903423) * (Q.psi_0p3 - 0.9299135)
    if Q.sum_pt_top50 > 988.4554 and Q.sum_z_dr2_top15 < 0.02146578:
        z += -0.395954 * (Q.sum_pt_top50 - 988.4554) * (0.02146578 - Q.sum_z_dr2_top15)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.2874307
    if Q.lam2 < 0.0006154841:
        z += -1001.893 * Q.lam2 + 0.6166494
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.2023567 * Q.n_dr_0p2_0p4 + 1.124431
    if 5.0 <= Q.n_dr_0p2_0p4 < 8.0:
        z += -0.03754923 * Q.n_dr_0p2_0p4 + 0.3003938
    if Q.n_particles < 46.0:
        z += -0.02171065 * Q.n_particles + 0.9986899
    if Q.tau21 < 0.3861957:
        z += 1.786492 * Q.tau21 - 0.6899353
    if Q.sum_z_dr2_top40 < 0.006026828:
        z += -89.76953 * Q.sum_z_dr2_top40 + 0.8010953
    if 0.006026828 <= Q.sum_z_dr2_top40 < 0.006259772:
        z += 71.1935 * Q.sum_z_dr2_top40 - 0.1690012
    if 0.006259772 <= Q.sum_z_dr2_top40 < 0.008840538:
        z += -107.1983 * Q.sum_z_dr2_top40 + 0.9476911
    if Q.mass_top50 < 79.21004:
        z += 0.01103303 * Q.mass_top50 - 0.8739271
    if Q.mass < 53.87362:
        z += 0.006911836 * Q.mass - 0.203475
    if 53.87362 <= Q.mass < 87.36377:
        z += -0.005042994 * Q.mass + 0.440575
    if Q.sum_z_dr2 < 0.009614971:
        z += 129.7488 * Q.sum_z_dr2 - 1.247531
    if Q.n_dr_0p1_0p2 < 10.0:
        z += -0.05043822 * Q.n_dr_0p1_0p2 + 0.5043822
    if Q.sum_z_dr2_top50 < 0.008124776:
        z += -111.8487 * Q.sum_z_dr2_top50 + 0.9087454
    if Q.n_particles < 46.0 and Q.mass_top15 > 57.87349:
        z += -0.001259068 * (46.0 - Q.n_particles) * (Q.mass_top15 - 57.87349)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_dr_0p1_0p2 > 9.0:
        z += -0.01758449 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 9.0)
    if Q.n_particles < 46.0 and Q.sum_pt_top30 > 800.732:
        z += 6.457543e-05 * (46.0 - Q.n_particles) * (Q.sum_pt_top30 - 800.732)
    if Q.D2 < 2.410481 and Q.psi_0p3 > 0.9985421:
        z += 154.8658 * (2.410481 - Q.D2) * (Q.psi_0p3 - 0.9985421)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.zdr_5 < 0.007099471:
        z += -12.43015 * (5.0 - Q.n_dr_0p2_0p4) * (0.007099471 - Q.zdr_5)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.psi_0p1 < 0.9538343:
        z += 0.6231591 * (5.0 - Q.n_dr_0p2_0p4) * (0.9538343 - Q.psi_0p1)
    if Q.tau21 < 0.3861957 and Q.lam1 > 0.007671243:
        z += -345.0482 * (0.3861957 - Q.tau21) * (Q.lam1 - 0.007671243)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.zdr_4 < 0.006794973:
        z += -15.76641 * (5.0 - Q.n_dr_0p2_0p4) * (0.006794973 - Q.zdr_4)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.6914295
    if Q.mass_top15 < 69.02716:
        z += -0.01186823 * Q.mass_top15 + 0.8192305
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.02282503 * Q.n_dr_0p2_0p4 + 0.3423755
    if Q.mass_top40 < 67.72643:
        z += -0.0009685885 * Q.mass_top40 - 0.115206
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += -0.01911649 * Q.mass_top40 + 1.113887
    if 83.32554 <= Q.mass_top40 < 94.64253:
        z += 0.04232617 * Q.mass_top40 - 4.005856
    if Q.mass_top30 < 60.43821:
        z += 0.0006107148 * Q.mass_top30 - 0.6133081
    if 60.43821 <= Q.mass_top30 < 121.737:
        z += 0.009403079 * Q.mass_top30 - 1.144703
    if Q.mass < 74.25181:
        z += 0.08232101 * Q.mass - 5.636057
    if 74.25181 <= Q.mass < 79.65241:
        z += 0.01991483 * Q.mass - 1.002286
    if 79.65241 <= Q.mass < 87.36377:
        z += 0.03843821 * Q.mass - 2.477718
    if 87.36377 <= Q.mass < 101.0497:
        z += -0.03180229 * Q.mass + 3.658758
    if 101.0497 <= Q.mass < 121.3913:
        z += -0.02188353 * Q.mass + 2.65647
    if Q.psi_0p3 >= 0.9973959:
        z += 111.2192 * Q.psi_0p3 - 110.9296
    if Q.sj3_pair_mass_min >= 32.51366:
        z += 0.008023934 * Q.sj3_pair_mass_min - 0.2608875
    if Q.n_particles >= 22.0:
        z += -0.01845784 * Q.n_particles + 0.4060725
    if Q.sum_z_dr2_top15 < 0.004855289:
        z += 226.0949 * Q.sum_z_dr2_top15 - 1.097756
    if Q.sum_z_dr2_top15 >= 0.007887677:
        z += -22.59892 * Q.sum_z_dr2_top15 + 0.178253
    if Q.n_dr_0_0p05 >= 5.0:
        z += 0.02503371 * Q.n_dr_0_0p05 - 0.1251685
    if Q.sj2_dr >= 0.2232169:
        z += -2.291058 * Q.sj2_dr + 0.5114029
    if Q.sum_zz_dr2 >= 0.01396296:
        z += 81.86659 * Q.sum_zz_dr2 - 1.1431
    if Q.sum_pt < 1042.609:
        z += 0.003695312 * Q.sum_pt - 3.852766
    if Q.sum_z_dr2_top10 < 0.007678544:
        z += 44.61525 * Q.sum_z_dr2_top10 - 0.3425801
    if Q.lam2 < 0.001776308:
        z += 152.651 * Q.lam2 - 0.2711552
    if Q.D2 < 5.378975:
        z += -0.1559495 * Q.D2 + 0.8388484
    if Q.sd_nremoved < 1.0:
        z += 0.1386088 * Q.sd_nremoved - 0.1386088
    if Q.psi_0p1 >= 0.8747961:
        z += -1.489265 * Q.psi_0p1 + 1.302803
    if Q.mass_top10 >= 76.9886:
        z += -0.009699348 * Q.mass_top10 + 0.7467392
    if Q.n_dr_0p2_0p4 < 15.0 and Q.z_top50_slots < 1.0:
        z += -1.018753 * (15.0 - Q.n_dr_0p2_0p4) * (1.0 - Q.z_top50_slots)
    if Q.mass_top40 < 83.32554 and Q.D2 < 6.916121:
        z += -0.01352989 * (83.32554 - Q.mass_top40) * (6.916121 - Q.D2)
    if Q.mass < 87.36377 and Q.zdr_0 > 0.006864207:
        z += -3.264511 * (87.36377 - Q.mass) * (Q.zdr_0 - 0.006864207)
    if Q.mass < 79.65241 and Q.lam2 < 0.003687605:
        z += -22.2738 * (79.65241 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.mass_top30 < 121.737 and Q.sum_pt_top40 < 1053.047:
        z += 2.637887e-05 * (121.737 - Q.mass_top30) * (1053.047 - Q.sum_pt_top40)
    if Q.mass < 101.0497 and Q.lam2 < 0.003687605:
        z += 13.39343 * (101.0497 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.sum_z_dr2_top15 < 0.004855289 and Q.n_dr_0p05_0p1 > 2.0:
        z += 7.41268 * (0.004855289 - Q.sum_z_dr2_top15) * (Q.n_dr_0p05_0p1 - 2.0)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.365113
    z += -0.03550476 * Q.n_particles + 2.272305
    if 0.09046749 <= Q.mass_over_sum_pt < 0.1708801:
        z += -31.55748 * Q.mass_over_sum_pt + 2.854926
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 162.8298 * Q.mass_over_sum_pt - 30.362
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -20.85056 * Q.log_sum_pt + 144.0801
    if 6.920349 <= Q.log_sum_pt < 6.935549:
        z += -30.59209 * Q.log_sum_pt + 211.4949
    if 6.935549 <= Q.log_sum_pt < 6.98945:
        z += -24.37633 * Q.log_sum_pt + 168.3852
    if Q.log_sum_pt >= 6.98945:
        z += -18.7968 * Q.log_sum_pt + 129.3873
    if 907.9372 <= Q.sum_pt < 986.0565:
        z += 0.007034252 * Q.sum_pt - 6.386659
    if Q.sum_pt >= 986.0565:
        z += -0.006493227 * Q.sum_pt + 6.9522
    if Q.sum_z_dr2_top50 >= 0.01951641:
        z += -38.32114 * Q.sum_z_dr2_top50 + 0.7478911
    if Q.sum_pt_top50 >= 934.2416:
        z += 0.02108265 * Q.sum_pt_top50 - 19.69629
    if Q.n_pt_above_1 >= 28.0:
        z += 0.01997073 * Q.n_pt_above_1 - 0.5591804
    if Q.sum_pt_top3 < 787.6281:
        z += 0.0006994508 * Q.sum_pt_top3 - 0.5509072
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.03204697 * Q.n_dr_0p2_0p4 + 0.3525167
    if Q.sum_pt_top40 >= 1024.942:
        z += -0.005053338 * Q.sum_pt_top40 + 5.17938
    if Q.sum_pt_top30 >= 933.1875:
        z += 0.002719696 * Q.sum_pt_top30 - 2.537987
    if 0.2404747 <= Q.max_dr < 0.4357228:
        z += -2.187253 * Q.max_dr + 0.5259789
    if Q.max_dr >= 0.4357228:
        z += -0.4344527 * Q.max_dr - 0.237756
    if Q.z_top30_slots >= 0.9048492:
        z += -8.345629 * Q.z_top30_slots + 7.551536
    if Q.mass < 64.48544:
        z += -0.01805284 * Q.mass + 1.164145
    if Q.mass >= 172.4888:
        z += 0.01531813 * Q.mass - 2.642207
    if Q.mass_top40 < 150.0144:
        z += 0.01391004 * Q.mass_top40 - 2.086707
    if Q.mass_top50 >= 157.5448:
        z += -0.05923529 * Q.mass_top50 + 9.332213
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -543.0219 * Q.mass_over_sum_pt_sq + 15.85625
    if 69.65633 <= Q.sd_mass < 83.30647:
        z += 0.03605449 * Q.sd_mass - 2.511423
    if Q.sd_mass >= 83.30647:
        z += -0.01049783 * Q.sd_mass + 1.366686
    if Q.tau21 < 0.5100475:
        z += 1.101833 * Q.tau21 - 0.5619873
    if Q.sum_pt_top10 >= 943.6922:
        z += -0.002087266 * Q.sum_pt_top10 + 1.969737
    if Q.z_11 < 0.01375115:
        z += -151.2781 * Q.z_11 + 2.080247
    if Q.pt_11 < 14.14062:
        z += 0.134532 * Q.pt_11 - 1.902366
    if Q.e2 < 0.04082832:
        z += -21.40778 * Q.e2 + 0.8740437
    if Q.tau1 < 0.1507173:
        z += 6.487458 * Q.tau1 - 0.977772
    if Q.z_dr_0p2_0p4 < 0.03700182:
        z += 10.83178 * Q.z_dr_0p2_0p4 - 0.4007956
    if Q.n_dr_0p1_0p2 < 21.0:
        z += 0.01411033 * Q.n_dr_0p1_0p2 - 0.2963168
    if Q.mass_top10 >= 71.781:
        z += 0.01217882 * Q.mass_top10 - 0.8742076
    if Q.psi_0p1 < 0.707925:
        z += 0.6472427 * Q.psi_0p1 - 0.4581993
    if Q.sj3_mass1 < 23.45965:
        z += -0.01920067 * Q.sj3_mass1 + 0.450441
    if Q.n_particles < 64.0 and Q.D2 < 2.178951:
        z += -0.01375075 * (64.0 - Q.n_particles) * (2.178951 - Q.D2)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 9831.014 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    if Q.psi_0p3 > 0.9973959 and Q.D2 < 3.345339:
        z += 34.27206 * (Q.psi_0p3 - 0.9973959) * (3.345339 - Q.D2)
    if Q.log_sum_pt > 6.910131 and Q.absphi_13 < 0.1958008:
        z += -5.676478 * (Q.log_sum_pt - 6.910131) * (0.1958008 - Q.absphi_13)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.3591835
    if Q.mass_top50 < 71.79516:
        z += -0.03309143 * Q.mass_top50 + 2.375804
    if Q.mass < 82.85409:
        z += -0.01229726 * Q.mass - 0.5284734
    if 82.85409 <= Q.mass < 89.74183:
        z += 0.01802821 * Q.mass - 3.041062
    if 89.74183 <= Q.mass < 92.85979:
        z += 0.05115574 * Q.mass - 6.013988
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.1091338 * Q.mass - 11.39782
    if 101.0497 <= Q.mass < 121.3913:
        z += 0.03429446 * Q.mass - 3.835325
    if 121.3913 <= Q.mass < 172.4888:
        z += -0.006413691 * Q.mass + 1.10629
    if Q.e3 < 0.0003372339:
        z += -1131.49 * Q.e3 + 0.3815768
    if Q.e2 >= 0.04755309:
        z += 36.44807 * Q.e2 - 1.733218
    if Q.tau1 < 0.06310829:
        z += -8.79812 * Q.tau1 + 0.5552343
    if Q.sj3_dr_min >= 0.1204829:
        z += -1.995063 * Q.sj3_dr_min + 0.240371
    if Q.n_dr_0p1_0p2 >= 33.0:
        z += 0.05031243 * Q.n_dr_0p1_0p2 - 1.66031
    if Q.sj3_mass1 >= 19.30204:
        z += -0.02226872 * Q.sj3_mass1 + 0.4298317
    if Q.lam2 < 0.003687605:
        z += -81.49459 * Q.lam2 + 0.3005199
    if Q.sum_z_dr2_top2 < 0.001425993:
        z += 124.6715 * Q.sum_z_dr2_top2 - 0.1777808
    if Q.D2 < 2.178951:
        z += 0.3986125 * Q.D2 - 0.868557
    if Q.sj3_pair_mass_min < 76.60223:
        z += -0.006799873 * Q.sj3_pair_mass_min + 0.5208855
    if Q.lam1 < 0.007259287:
        z += -105.1099 * Q.lam1 + 0.3195027
    if 0.007259287 <= Q.lam1 < 0.01649354:
        z += 48.02988 * Q.lam1 - 0.7921829
    if Q.mass_over_sum_pt < 0.09795415:
        z += 41.21214 * Q.mass_over_sum_pt - 4.036899
    if Q.sum_zz_dr2 < 0.02580859:
        z += -203.3618 * Q.sum_zz_dr2 + 5.248481
    if Q.sum_z_dr2_top50 < 0.02550569:
        z += 75.8776 * Q.sum_z_dr2_top50 - 1.935311
    if Q.sum_z_dr2_top30 < 0.008376291:
        z += -40.34685 * Q.sum_z_dr2_top30 - 0.9863121
    if 0.008376291 <= Q.sum_z_dr2_top30 < 0.02412652:
        z += 84.07935 * Q.sum_z_dr2_top30 - 2.028542
    if Q.sum_z_dr2_top20 < 0.00287991:
        z += -80.74909 * Q.sum_z_dr2_top20 + 0.2325501
    if Q.soft10_dr >= 0.2075213:
        z += -1.919322 * Q.soft10_dr + 0.3983003
    if Q.mass < 121.3913 and Q.sum_pt_top50 < 1003.544:
        z += 6.305604e-05 * (121.3913 - Q.mass) * (1003.544 - Q.sum_pt_top50)
    if Q.e2 > 0.04755309 and Q.psi_0p3 > 0.9896594:
        z += 2232.745 * (Q.e2 - 0.04755309) * (Q.psi_0p3 - 0.9896594)
    if Q.mass < 89.74183 and Q.sum_pt_top20 < 1129.275:
        z += -2.192575e-05 * (89.74183 - Q.mass) * (1129.275 - Q.sum_pt_top20)
    if Q.e2 > 0.04755309 and Q.z_14 > 0.01634243:
        z += -4492.619 * (Q.e2 - 0.04755309) * (Q.z_14 - 0.01634243)
    if Q.n_dr_0p1_0p2 > 33.0 and Q.n_dr_0_0p05 < 9.0:
        z += -0.005595608 * (Q.n_dr_0p1_0p2 - 33.0) * (9.0 - Q.n_dr_0_0p05)
    if Q.eta_0 < 0.07952881 and Q.n_dr_0_0p05 > 5.0:
        z += -0.1380028 * (0.07952881 - Q.eta_0) * (Q.n_dr_0_0p05 - 5.0)
    if Q.z_dr_0p1_0p2 < 0.2864926 and Q.soft10_dr0 > 0.1909669:
        z += 6.167434 * (0.2864926 - Q.z_dr_0p1_0p2) * (Q.soft10_dr0 - 0.1909669)
    if Q.e2 > 0.02793599 and Q.ptdr0_13 < 6.06065:
        z += -1.790278 * (Q.e2 - 0.02793599) * (6.06065 - Q.ptdr0_13)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.2224737
    if Q.tau21_b2 < 0.2352054:
        z += -3.491502 * Q.tau21_b2 + 0.8212203
    if Q.sum_z_dr2 < 0.006403325:
        z += -285.3153 * Q.sum_z_dr2 + 2.407596
    if 0.006403325 <= Q.sum_z_dr2 < 0.007877041:
        z += -393.9905 * Q.sum_z_dr2 + 3.103479
    if Q.mass_over_sum_pt < 0.1182259:
        z += -19.03582 * Q.mass_over_sum_pt + 2.250527
    if Q.mass < 78.26182:
        z += 0.1736311 * Q.mass - 14.72887
    if 78.26182 <= Q.mass < 82.85409:
        z += 0.1273144 * Q.mass - 11.10404
    if 82.85409 <= Q.mass < 91.03469:
        z += 0.060156 * Q.mass - 5.539687
    if 91.03469 <= Q.mass < 92.85979:
        z += 0.2440808 * Q.mass - 22.28322
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.01340333 * Q.mass - 0.8625614
    if 101.0497 <= Q.mass < 121.3913:
        z += -0.0241791 * Q.mass + 2.935132
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.1010306 * Q.n_dr_0p2_0p4 + 0.6061837
    if Q.psi_0p3 >= 0.9980008:
        z += -402.5354 * Q.psi_0p3 + 401.7307
    if Q.tau1 < 0.07708632:
        z += -6.933266 * Q.tau1 - 0.4102567
    if 0.07708632 <= Q.tau1 < 0.08786745:
        z += 17.40266 * Q.tau1 - 2.286224
    if 0.08786745 <= Q.tau1 < 0.1072713:
        z += 39.01793 * Q.tau1 - 4.185502
    if Q.LHA < 0.2601462:
        z += 0.898097 * Q.LHA + 0.3402214
    if 0.2601462 <= Q.LHA < 0.2845608:
        z += -8.01789 * Q.LHA + 2.659681
    if 0.2845608 <= Q.LHA < 0.3098384:
        z += -20.54271 * Q.LHA + 6.223753
    if 0.3098384 <= Q.LHA < 0.3332345:
        z += 6.033715 * Q.LHA - 2.010642
    if Q.tau21 < 0.3861957:
        z += 0.8937333 * Q.tau21 - 0.3451559
    if Q.mass_top20 < 70.42121:
        z += -0.005455375 * Q.mass_top20 + 0.1689405
    if 70.42121 <= Q.mass_top20 < 73.35236:
        z += 0.07342954 * Q.mass_top20 - 5.38623
    if Q.z_top20_slots >= 0.9102775:
        z += -6.167508 * Q.z_top20_slots + 5.614143
    if Q.sum_z_dr2_top50 < 0.003418057:
        z += 2854.465 * Q.sum_z_dr2_top50 - 9.756723
    if Q.lam1 < 0.006189818:
        z += 51.23199 * Q.lam1 - 0.8484531
    if 0.006189818 <= Q.lam1 < 0.008241985:
        z += 258.9148 * Q.lam1 - 2.133972
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += -138.5567 * Q.mass_over_sum_pt_sq + 1.329454
    if Q.lam1_plus_lam2 < 0.008190222:
        z += 480.3535 * Q.lam1_plus_lam2 - 3.934202
    if Q.mass < 91.03469 and Q.psi_0p3 > 0.9299135:
        z += 0.02552044 * (91.03469 - Q.mass) * (Q.psi_0p3 - 0.9299135)
    if Q.n_dr_0p2_0p4 < 6.0 and Q.e3 < 7.876005e-05:
        z += -1134.809 * (6.0 - Q.n_dr_0p2_0p4) * (7.876005e-05 - Q.e3)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9638082:
        z += 3.596536 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.mass < 91.03469 and Q.psi_0p3 > 0.9638082:
        z += -4.919033 * (91.03469 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 1260.541:
        z += -0.008735284 * (0.2352054 - Q.tau21_b2) * (1260.541 - Q.sum_pt)
    if Q.mass < 91.03469 and Q.psi_0p3 < 0.9973959:
        z += 1.913659 * (91.03469 - Q.mass) * (0.9973959 - Q.psi_0p3)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9973959:
        z += 11.79884 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9973959)
    if Q.mass_over_sum_pt < 0.1182259 and Q.psi_0p3 > 0.9973959:
        z += 2038.017 * (0.1182259 - Q.mass_over_sum_pt) * (Q.psi_0p3 - 0.9973959)
    if Q.mass < 91.03469 and Q.psi_0p3 > 0.9973959:
        z += -87.32425 * (91.03469 - Q.mass) * (Q.psi_0p3 - 0.9973959)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9973959:
        z += 63.22398 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9973959)
    if Q.psi_0p3 > 0.9980008 and Q.orientation_deg > -9.840088:
        z += -1.860089 * (Q.psi_0p3 - 0.9980008) * (Q.orientation_deg - -9.840088)
    if Q.psi_0p3 > 0.9980008 and Q.zdr_3 > 0.00396157:
        z += 28639.36 * (Q.psi_0p3 - 0.9980008) * (Q.zdr_3 - 0.00396157)
    if Q.mass < 121.3913 and Q.sd_mass > 76.29481:
        z += 0.001908467 * (121.3913 - Q.mass) * (Q.sd_mass - 76.29481)
    if Q.mass < 91.03469 and Q.sd_mass > 55.96662:
        z += -0.002047878 * (91.03469 - Q.mass) * (Q.sd_mass - 55.96662)
    if Q.psi_0p3 > 0.9980008 and Q.pair_mass_0_3 > 6.624378:
        z += -8.349812 * (Q.psi_0p3 - 0.9980008) * (Q.pair_mass_0_3 - 6.624378)
    if Q.tau21_b2 < 0.2352054 and Q.sj3_mass3 < 11.65938:
        z += 0.1147505 * (0.2352054 - Q.tau21_b2) * (11.65938 - Q.sj3_mass3)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.8586652
    if Q.mass_over_sum_pt >= 0.07696632:
        z += 44.22524 * Q.mass_over_sum_pt - 3.403854
    if 0.005196966 <= Q.sum_z_dr2_top40 < 0.006026828:
        z += 74.2517 * Q.sum_z_dr2_top40 - 0.3858836
    if 0.006026828 <= Q.sum_z_dr2_top40 < 0.008031986:
        z += -106.893 * Q.sum_z_dr2_top40 + 0.7058445
    if Q.sum_z_dr2_top40 >= 0.008031986:
        z += 42.39342 * Q.sum_z_dr2_top40 - 0.493222
    if Q.n_dr_0p2_0p4 < 8.0:
        z += 0.03063734 * Q.n_dr_0p2_0p4 - 0.4595601
    if 8.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += 0.07060825 * Q.n_dr_0p2_0p4 - 0.7793274
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += 0.03997092 * Q.n_dr_0p2_0p4 - 0.3197673
    if 64.48544 <= Q.mass < 80.78464:
        z += 0.0376181 * Q.mass - 2.42582
    if 80.78464 <= Q.mass < 89.74183:
        z += 0.08292849 * Q.mass - 6.086203
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.08767869 * Q.mass - 6.512495
    if 101.0497 <= Q.mass < 121.3913:
        z += -0.0119638 * Q.mass + 3.556349
    if 121.3913 <= Q.mass < 143.7876:
        z += -0.03056104 * Q.mass + 5.813893
    if Q.mass >= 143.7876:
        z += -0.04217333 * Q.mass + 7.483596
    if Q.log_sum_pt < 6.811175:
        z += -1.639178 * Q.log_sum_pt + 12.082
    if 6.811175 <= Q.log_sum_pt < 6.935549:
        z += -7.375078 * Q.log_sum_pt + 51.15022
    if Q.log_sum_pt >= 6.959294:
        z += 4.842668 * Q.log_sum_pt - 33.70155
    if Q.sum_pt_top40 < 858.8262:
        z += 0.003470294 * Q.sum_pt_top40 - 3.58275
    if 858.8262 <= Q.sum_pt_top40 < 1032.405:
        z += -0.0002380956 * Q.sum_pt_top40 - 0.3978876
    if Q.sum_pt_top40 >= 1032.405:
        z += -0.00370839 * Q.sum_pt_top40 + 3.184862
    if Q.mass_top20 >= 90.4505:
        z += -0.02112566 * Q.mass_top20 + 1.910827
    if Q.sum_z_dr2_top15 < 0.001319197:
        z += -200.6967 * Q.sum_z_dr2_top15 + 1.158434
    if 0.001319197 <= Q.sum_z_dr2_top15 < 0.005383629:
        z += -120.063 * Q.sum_z_dr2_top15 + 1.052062
    if 0.005383629 <= Q.sum_z_dr2_top15 < 0.007887677:
        z += -230.3037 * Q.sum_z_dr2_top15 + 1.645557
    if 0.007887677 <= Q.sum_z_dr2_top15 < 0.01563836:
        z += 171.0315 * Q.sum_z_dr2_top15 - 1.520045
    if Q.sum_z_dr2_top15 >= 0.01563836:
        z += 80.63364 * Q.sum_z_dr2_top15 - 0.1063717
    if 80.35535 <= Q.mass_top50 < 97.93004:
        z += -0.05127087 * Q.mass_top50 + 4.119889
    if Q.mass_top50 >= 97.93004:
        z += -0.005728699 * Q.mass_top50 - 0.3400581
    if 0.007872294 <= Q.sum_zz_dr2 < 0.009606007:
        z += 216.2895 * Q.sum_zz_dr2 - 1.702695
    if Q.sum_zz_dr2 >= 0.009606007:
        z += -223.8385 * Q.sum_zz_dr2 + 2.525178
    if Q.sj2_dr >= 0.2232169:
        z += 3.355825 * Q.sj2_dr - 0.7490769
    if Q.e2 >= 0.04755309:
        z += 31.16157 * Q.e2 - 1.481829
    if Q.sum_pt_top50 < 1003.544:
        z += -0.007857585 * Q.sum_pt_top50 + 7.885434
    if Q.n_dr_0p1_0p2 >= 19.0:
        z += 0.02347445 * Q.n_dr_0p1_0p2 - 0.4460145
    if Q.tau1 >= 0.06310829:
        z += -10.45823 * Q.tau1 + 0.6600008
    if Q.n_pt_above_1 >= 54.0:
        z += 0.03090803 * Q.n_pt_above_1 - 1.669034
    if Q.z_dr_0p2_0p4 < 0.1291856:
        z += -2.563115 * Q.z_dr_0p2_0p4 + 0.3311175
    if Q.mass_top40 >= 111.2487:
        z += 0.008906562 * Q.mass_top40 - 0.9908432
    if Q.LHA >= 0.3332345:
        z += 5.433696 * Q.LHA - 1.810695
    if Q.e3 < 0.0003372339:
        z += -3178.132 * Q.e3 + 1.071774
    if Q.sum_z_dr2_top20 < 0.02737453:
        z += 38.94893 * Q.sum_z_dr2_top20 - 1.066209
    if Q.C2 >= 0.07996447:
        z += 3.189916 * Q.C2 - 0.2550799
    if Q.sum_z_dr2_top30 < 0.007856958:
        z += 34.11252 * Q.sum_z_dr2_top30 - 0.2680207
    if Q.mass_over_sum_pt_sq >= 0.001785266:
        z += 95.89755 * Q.mass_over_sum_pt_sq - 0.1712027
    if Q.sum_pt_top40 < 1001.523 and Q.psi_0p3 > 0.9924477:
        z += 0.6663901 * (1001.523 - Q.sum_pt_top40) * (Q.psi_0p3 - 0.9924477)
    if Q.log_sum_pt > 6.959294 and Q.sj3_pair_mass_max > 28.35435:
        z += 0.03894753 * (Q.log_sum_pt - 6.959294) * (Q.sj3_pair_mass_max - 28.35435)
    if Q.sum_z_dr2_top15 < 0.007887677 and Q.tau21_b2 < 0.6133424:
        z += -185.7492 * (0.007887677 - Q.sum_z_dr2_top15) * (0.6133424 - Q.tau21_b2)
    if Q.sum_pt_top40 < 1032.405 and Q.psi_0p3 > 0.9924477:
        z += -0.7297459 * (1032.405 - Q.sum_pt_top40) * (Q.psi_0p3 - 0.9924477)
    if Q.sum_pt_top50 < 1003.544 and Q.n_dr_0p05_0p1 < 21.0:
        z += -0.0002108822 * (1003.544 - Q.sum_pt_top50) * (21.0 - Q.n_dr_0p05_0p1)
    if Q.n_dr_0p2_0p4 > 8.0 and Q.sj3_mass1 < 26.76006:
        z += -0.00100911 * (Q.n_dr_0p2_0p4 - 8.0) * (26.76006 - Q.sj3_mass1)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.1731589
    if Q.mass_top40 < 80.89043:
        z += 0.001440831 * Q.mass_top40 - 0.1735981
    if 80.89043 <= Q.mass_top40 < 89.6788:
        z += 0.006491384 * Q.mass_top40 - 0.5821395
    if Q.sum_z_dr2_top40 < 0.006259772:
        z += -263.6798 * Q.sum_z_dr2_top40 + 1.650575
    if Q.mass_top30 < 121.737:
        z += -0.001608495 * Q.mass_top30 - 0.2617823
    if 121.737 <= Q.mass_top30 < 152.6883:
        z += 0.01478441 * Q.mass_top30 - 2.257405
    if Q.mass < 64.48544:
        z += -0.02049817 * Q.mass + 2.988622
    if 64.48544 <= Q.mass < 79.65241:
        z += -0.08812359 * Q.mass + 7.349477
    if 79.65241 <= Q.mass < 87.36377:
        z += -0.0428226 * Q.mass + 3.741144
    if 143.7876 <= Q.mass < 172.4888:
        z += -0.06997643 * Q.mass + 10.06174
    if Q.mass >= 172.4888:
        z += -0.00737147 * Q.mass - 0.7369118
    if Q.sum_z_dr >= 0.08589404:
        z += 35.81701 * Q.sum_z_dr - 3.076467
    if Q.z_top40_slots >= 0.9574183:
        z += -7.363469 * Q.z_top40_slots + 7.04992
    if Q.sum_pt_top50 < 959.0957:
        z += -0.003105244 * Q.sum_pt_top50 + 2.978226
    if Q.lam1 < 0.004673423:
        z += 133.8495 * Q.lam1 - 0.6255354
    if Q.lam1 >= 0.01174405:
        z += 88.79276 * Q.lam1 - 1.042787
    if Q.sum_z_dr2_top20 < 0.01083435:
        z += -36.1164 * Q.sum_z_dr2_top20 + 0.3912978
    if Q.sum_z_dr2_top30 >= 0.02412652:
        z += -186.3027 * Q.sum_z_dr2_top30 + 4.494837
    if Q.tau1 < 0.05444509:
        z += 38.68333 * Q.tau1 - 2.106117
    if Q.LHA < 0.2091025:
        z += -16.74914 * Q.LHA + 3.502288
    if Q.z_dr_0p1_0p2 < 0.2187642:
        z += -1.280691 * Q.z_dr_0p1_0p2 + 0.2801694
    if Q.sum_pt_top40 < 956.2133 and Q.soft4_pt > 1.789258:
        z += 0.003046227 * (956.2133 - Q.sum_pt_top40) * (Q.soft4_pt - 1.789258)
    if Q.mass_top40 < 80.89043 and Q.sum_pt < 1034.834:
        z += -0.0002116931 * (80.89043 - Q.mass_top40) * (1034.834 - Q.sum_pt)
    if Q.mass < 87.36377 and Q.n_particles > 38.0:
        z += 0.0008216668 * (87.36377 - Q.mass) * (Q.n_particles - 38.0)
    if Q.mass_top40 < 80.89043 and Q.sum_pt_top50 < 934.2416:
        z += 0.0008920693 * (80.89043 - Q.mass_top40) * (934.2416 - Q.sum_pt_top50)
    if Q.mass_top30 < 121.737 and Q.n_dr_0p2_0p4 > 10.0:
        z += 0.0009028777 * (121.737 - Q.mass_top30) * (Q.n_dr_0p2_0p4 - 10.0)
    if Q.sum_pt_top40 < 956.2133 and Q.n_pt_above_10 < 20.0:
        z += -0.0009061 * (956.2133 - Q.sum_pt_top40) * (20.0 - Q.n_pt_above_10)
    if Q.sum_pt_top40 < 956.2133 and Q.soft5_pt > 2.894531:
        z += -0.01031488 * (956.2133 - Q.sum_pt_top40) * (Q.soft5_pt - 2.894531)
    if Q.sum_pt_top50 < 959.0957 and Q.soft5_pt > 2.894531:
        z += 0.004939419 * (959.0957 - Q.sum_pt_top50) * (Q.soft5_pt - 2.894531)
    if Q.lam1 > 0.01174405 and Q.soft5_z > 0.002346473:
        z += 17239.04 * (Q.lam1 - 0.01174405) * (Q.soft5_z - 0.002346473)
    return max(0.0, z)


def neuron_10(Q):
    z = -3.948963
    if Q.sum_z_dr < 0.1207452:
        z += 8.881083 * Q.sum_z_dr - 1.072348
    if Q.mass < 64.48544:
        z += -0.02880678 * Q.mass + 2.138956
    if 64.48544 <= Q.mass < 74.25181:
        z += -0.0885814 * Q.mass + 5.993548
    if 74.25181 <= Q.mass < 82.85409:
        z += -0.05977461 * Q.mass + 3.854593
    if 82.85409 <= Q.mass < 143.7876:
        z += 0.02944452 * Q.mass - 3.537577
    if 143.7876 <= Q.mass < 162.8363:
        z += -0.05050252 * Q.mass + 7.957817
    if Q.mass >= 162.8363:
        z += -0.0909373 * Q.mass + 14.54207
    if Q.sum_z_dr2_top30 < 0.005402331:
        z += -18.55639 * Q.sum_z_dr2_top30 + 1.274116
    if 0.005402331 <= Q.sum_z_dr2_top30 < 0.01807679:
        z += -92.61685 * Q.sum_z_dr2_top30 + 1.674215
    if Q.mass_top40 < 67.72643:
        z += -0.01538374 * Q.mass_top40 + 2.843248
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.01243907 * Q.mass_top40 + 0.9589081
    if 83.32554 <= Q.mass_top40 < 87.27603:
        z += -0.04512972 * Q.mass_top40 + 5.755858
    if 87.27603 <= Q.mass_top40 < 132.4278:
        z += -0.0003751814 * Q.mass_top40 + 1.84986
    if 132.4278 <= Q.mass_top40 < 163.2541:
        z += 0.02782281 * Q.mass_top40 - 1.88434
    if Q.mass_top40 >= 163.2541:
        z += -0.005119752 * Q.mass_top40 + 3.49367
    if Q.sj2_mass1 < 80.3008:
        z += 0.02077042 * Q.sj2_mass1 - 1.667881
    if Q.D2 < 3.814159:
        z += -0.232346 * Q.D2 + 0.8862049
    if Q.psi_0p3 >= 0.9985421:
        z += -215.0982 * Q.psi_0p3 + 214.7846
    if Q.mass_top5 >= 14.54404:
        z += 0.008188306 * Q.mass_top5 - 0.119091
    if Q.mass_top50 >= 157.5448:
        z += 0.01090531 * Q.mass_top50 - 1.718075
    if Q.z_dr_0_0p05 >= 0.7674734:
        z += -1.393145 * Q.z_dr_0_0p05 + 1.069202
    if Q.log_sum_pt >= 6.903423:
        z += -7.537004 * Q.log_sum_pt + 52.03113
    z += 0.003656236 * Q.sum_pt
    if Q.mass_top15 < 75.26407:
        z += 0.003736029 * Q.mass_top15 - 0.2811888
    if Q.mass_over_sum_pt >= 0.1708801:
        z += -310.9234 * Q.mass_over_sum_pt + 53.13062
    if Q.sum_pt_top5 < 839.9547:
        z += -0.001829684 * Q.sum_pt_top5 + 1.536852
    if Q.e2 < 0.06524004:
        z += 22.70827 * Q.e2 - 1.481489
    if Q.dr_0 < 0.06413297:
        z += -6.914821 * Q.dr_0 + 0.443468
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.04096867 * Q.n_dr_0p2_0p4 - 0.4506554
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -4.206493 * Q.z_dr_0p2_0p4 + 0.3837402
    if Q.mass_top10 < 71.781:
        z += 0.006647105 * Q.mass_top10 - 0.4771358
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += 727.3351 * Q.mass_over_sum_pt_sq - 21.2382
    if Q.dr_1 < 0.06940472:
        z += -4.029475 * Q.dr_1 + 0.2796646
    if Q.tau1 < 0.05444509:
        z += 40.51704 * Q.tau1 - 2.205954
    if Q.tau21_b2 < 0.6133424:
        z += -1.226455 * Q.tau21_b2 + 0.7522369
    if Q.sj2_mass1 < 80.3008 and Q.sum_pt_top2 < 464.75:
        z += 2.374723e-05 * (80.3008 - Q.sj2_mass1) * (464.75 - Q.sum_pt_top2)
    if Q.D2 < 3.814159 and Q.sj2_dr > 0.1937688:
        z += -2.467481 * (3.814159 - Q.D2) * (Q.sj2_dr - 0.1937688)
    if Q.mass_top40 > 163.2541 and Q.soft4_z > 0.001186042:
        z += 30.05285 * (Q.mass_top40 - 163.2541) * (Q.soft4_z - 0.001186042)
    if Q.mass > 64.48544 and Q.soft2_pt < 2.537109:
        z += 0.004303969 * (Q.mass - 64.48544) * (2.537109 - Q.soft2_pt)
    if Q.log_sum_pt > 6.903423 and Q.dr_4 < 0.0830523:
        z += 19.25813 * (Q.log_sum_pt - 6.903423) * (0.0830523 - Q.dr_4)
    if Q.mass_top40 < 111.2487 and Q.pt1_dr01 > 12.6865:
        z += 0.0008142922 * (111.2487 - Q.mass_top40) * (Q.pt1_dr01 - 12.6865)
    if Q.sj2_mass1 < 80.3008 and Q.sj2_mass2 > 11.91979:
        z += 0.0002261132 * (80.3008 - Q.sj2_mass1) * (Q.sj2_mass2 - 11.91979)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.2780376
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.1601796 * Q.n_dr_0p2_0p4 + 1.30007
    if 6.0 <= Q.n_dr_0p2_0p4 < 10.0:
        z += -0.08474821 * Q.n_dr_0p2_0p4 + 0.8474821
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += -164.7726 * Q.mass_over_sum_pt_sq + 1.580995
    if Q.psi_0p2 >= 0.9935324:
        z += 42.31742 * Q.psi_0p2 - 42.04373
    if Q.mass < 79.65241:
        z += -0.01460457 * Q.mass + 2.037378
    if 79.65241 <= Q.mass < 92.85979:
        z += -0.06618185 * Q.mass + 6.145633
    if Q.sum_z_dr < 0.05048381:
        z += 26.71204 * Q.sum_z_dr - 2.140738
    if 0.05048381 <= Q.sum_z_dr < 0.07374472:
        z += 34.05769 * Q.sum_z_dr - 2.511575
    if Q.sum_z_dr2_top10 < 0.003213724:
        z += -156.1651 * Q.sum_z_dr2_top10 + 0.5018714
    if Q.lam1 < 0.006716737:
        z += 233.8773 * Q.lam1 - 1.570892
    if Q.e2 < 0.02515919:
        z += -46.2301 * Q.e2 + 1.015166
    if 0.02515919 <= Q.e2 < 0.02793599:
        z += -68.12072 * Q.e2 + 1.565916
    if 0.02793599 <= Q.e2 < 0.04082832:
        z += 26.14764 * Q.e2 - 1.067564
    if Q.sum_z_dr2_top30 < 0.006929741:
        z += 282.375 * Q.sum_z_dr2_top30 - 1.956786
    if Q.sum_zz_dr2 < 0.00818374:
        z += -232.8513 * Q.sum_zz_dr2 + 1.905594
    if Q.LHA < 0.2845608:
        z += -10.37528 * Q.LHA + 2.952398
    if Q.n_dr_0p1_0p2 < 15.0:
        z += -0.01854368 * Q.n_dr_0p1_0p2 + 0.2781551
    if Q.z_dr_0p1_0p2 < 0.1548383:
        z += -1.176877 * Q.z_dr_0p1_0p2 + 0.1822256
    if Q.mass_top30 < 60.43821:
        z += -0.02129827 * Q.mass_top30 + 1.287229
    if Q.mass_top50 < 85.8667:
        z += 0.04347799 * Q.mass_top50 - 3.733311
    if Q.z_top10_slots >= 0.7271951:
        z += -2.218355 * Q.z_top10_slots + 1.613177
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_particles > 34.0:
        z += -0.002310951 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_particles - 34.0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.sum_z_dr2 < 0.005532208:
        z += -25.88529 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.sum_z_dr2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.0007130026 * (10.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.mass < 101.0497 and Q.mass_top10 > 38.52415:
        z += 0.0004235186 * (101.0497 - Q.mass) * (Q.mass_top10 - 38.52415)
    if Q.mass < 79.65241 and Q.D2 < 3.814159:
        z += -0.008184396 * (79.65241 - Q.mass) * (3.814159 - Q.D2)
    if Q.z_top30_slots > 0.9734886 and Q.C2_b2 < 0.05112769:
        z += 209.7776 * (Q.z_top30_slots - 0.9734886) * (0.05112769 - Q.C2_b2)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.1994202
    if Q.mass < 53.87362:
        z += -0.08932803 * Q.mass + 8.739333
    if 53.87362 <= Q.mass < 80.78464:
        z += -0.1389522 * Q.mass + 11.41276
    if 80.78464 <= Q.mass < 82.85409:
        z += -0.06397552 * Q.mass + 5.355803
    if 82.85409 <= Q.mass < 87.36377:
        z += -0.01223357 * Q.mass + 1.068771
    if Q.mass_top40 < 77.93668:
        z += 0.03725771 * Q.mass_top40 - 3.467525
    if 77.93668 <= Q.mass_top40 < 94.64253:
        z += 0.03374758 * Q.mass_top40 - 3.193956
    if Q.z_dr_0p1_0p2 >= 0.4479367:
        z += 0.9075831 * Q.z_dr_0p1_0p2 - 0.4065397
    if Q.sum_zz_dr2 < 0.00363788:
        z += 373.0883 * Q.sum_zz_dr2 - 1.357251
    if Q.n_dr_0p2_0p4 < 7.0:
        z += -0.06322242 * Q.n_dr_0p2_0p4 + 0.4425569
    if Q.sum_pt < 972.0419:
        z += 0.003879674 * Q.sum_pt - 3.771206
    if Q.sum_pt >= 1260.541:
        z += -0.001456983 * Q.sum_pt + 1.836587
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += -4.797264 * Q.z_dr_0p2_0p4 + 0.3285718
    if Q.z_dr_0_0p05 >= 0.9084912:
        z += -4.510786 * Q.z_dr_0_0p05 + 4.098009
    if Q.mass < 87.36377 and Q.z_dr_0p1_0p2 < 0.06472584:
        z += -0.09773045 * (87.36377 - Q.mass) * (0.06472584 - Q.z_dr_0p1_0p2)
    if Q.mass < 87.36377 and Q.psi_0p3 < 0.9989733:
        z += -2.682579 * (87.36377 - Q.mass) * (0.9989733 - Q.psi_0p3)
    if Q.mass < 87.36377 and Q.lam2 > 0.0003497174:
        z += -11.1208 * (87.36377 - Q.mass) * (Q.lam2 - 0.0003497174)
    if Q.sj3_pair_mass_max > 122.7494 and Q.pt_0 < 435.25:
        z += 3.315602e-05 * (Q.sj3_pair_mass_max - 122.7494) * (435.25 - Q.pt_0)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.planar_flow < 0.3036026:
        z += 36.39789 * (Q.z_dr_0p1_0p2 - 0.4479367) * (0.3036026 - Q.planar_flow)
    if Q.sd_mass > 133.2575 and Q.sj3_mass1 < 32.50209:
        z += 0.001229078 * (Q.sd_mass - 133.2575) * (32.50209 - Q.sj3_mass1)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.soft1_pt > 0.8144531:
        z += -0.03622926 * (7.0 - Q.n_dr_0p2_0p4) * (Q.soft1_pt - 0.8144531)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.lam2 < 0.001776308:
        z += -5714.218 * (Q.z_dr_0p1_0p2 - 0.4479367) * (0.001776308 - Q.lam2)
    if Q.mass_top40 < 77.93668 and Q.lam2 > 0.000404306:
        z += 13.44873 * (77.93668 - Q.mass_top40) * (Q.lam2 - 0.000404306)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.183915
    if Q.sum_pt < 1012.673:
        z += 0.0426538 * Q.sum_pt - 43.68142
    if 1012.673 <= Q.sum_pt < 1085.125:
        z += 0.006722675 * Q.sum_pt - 7.294942
    if Q.sum_pt >= 1260.541:
        z += -0.002794427 * Q.sum_pt + 3.522489
    if 64.48544 <= Q.mass < 78.26182:
        z += 0.01398955 * Q.mass - 0.902122
    if 78.26182 <= Q.mass < 143.7876:
        z += 0.02944756 * Q.mass - 2.111895
    if 143.7876 <= Q.mass < 172.4888:
        z += -0.1424777 * Q.mass + 22.60883
    if Q.mass >= 172.4888:
        z += -0.06208197 * Q.mass + 8.741463
    if Q.n_for_90pct < 14.0:
        z += 0.08737683 * Q.n_for_90pct - 1.223276
    if Q.log_sum_pt < 6.811175:
        z += 8.670467 * Q.log_sum_pt - 58.76477
    if 6.811175 <= Q.log_sum_pt < 6.879399:
        z += -4.269772 * Q.log_sum_pt + 29.37346
    if Q.log_sum_pt >= 7.139296:
        z += 2.472223 * Q.log_sum_pt - 17.64993
    if Q.sum_pt_top40 < 1053.047:
        z += -0.004774652 * Q.sum_pt_top40 + 5.027935
    if 0.06030419 <= Q.mass_over_sum_pt < 0.1708801:
        z += -7.372589 * Q.mass_over_sum_pt + 0.444598
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 88.76595 * Q.mass_over_sum_pt - 15.98357
    if 60.5496 <= Q.mass_top10 < 71.781:
        z += 0.007236463 * Q.mass_top10 - 0.438165
    if Q.mass_top10 >= 71.781:
        z += -0.008010483 * Q.mass_top10 + 0.6562761
    if Q.sum_z_dr2_top40 < 0.02497133:
        z += -33.77415 * Q.sum_z_dr2_top40 + 0.8433854
    if 97.93004 <= Q.mass_top50 < 138.8977:
        z += -0.02347438 * Q.mass_top50 + 2.298847
    if 138.8977 <= Q.mass_top50 < 157.5448:
        z += 0.06294766 * Q.mass_top50 - 9.704974
    if 157.5448 <= Q.mass_top50 < 168.9698:
        z += -0.06640628 * Q.mass_top50 + 10.67407
    if Q.mass_top50 >= 168.9698:
        z += -0.106288 * Q.mass_top50 + 17.41287
    if Q.n_dr_0p1_0p2 < 19.0:
        z += 0.01185274 * Q.n_dr_0p1_0p2 - 0.225202
    if Q.sum_pt_top20 < 889.8383:
        z += 0.002755389 * Q.sum_pt_top20 - 2.45185
    if Q.sum_pt_top50 < 1008.935:
        z += -0.0142024 * Q.sum_pt_top50 + 14.3293
    if Q.psi_0p3 >= 0.9777125:
        z += 13.02367 * Q.psi_0p3 - 12.73341
    if Q.mass_top5 >= 37.76455:
        z += 0.008051488 * Q.mass_top5 - 0.3040608
    if Q.mass_top40 >= 150.0144:
        z += 0.1144895 * Q.mass_top40 - 17.17508
    if Q.tau1 < 0.08786745:
        z += -3.061151 * Q.tau1 + 0.2689755
    if Q.n_for_90pct < 14.0 and Q.D2 < 3.345339:
        z += 0.02621826 * (14.0 - Q.n_for_90pct) * (3.345339 - Q.D2)
    if Q.sum_pt < 1012.673 and Q.z_9 < 0.01833434:
        z += 1.267996 * (1012.673 - Q.sum_pt) * (0.01833434 - Q.z_9)
    if Q.mass > 172.4888 and Q.pt_9 < 41.4375:
        z += -0.009428357 * (Q.mass - 172.4888) * (41.4375 - Q.pt_9)
    if Q.sum_pt_top40 < 1053.047 and Q.D3 < 0.5260785:
        z += 0.004335024 * (1053.047 - Q.sum_pt_top40) * (0.5260785 - Q.D3)
    if Q.mass_top50 > 97.93004 and Q.soft3_pt < 2.873047:
        z += 0.001822475 * (Q.mass_top50 - 97.93004) * (2.873047 - Q.soft3_pt)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.1502622
    if Q.tau21_b2 < 0.342495:
        z += -2.310509 * Q.tau21_b2 + 0.7913376
    if Q.mass < 79.65241:
        z += 0.3495245 * Q.mass - 29.3537
    if 79.65241 <= Q.mass < 82.85409:
        z += 0.21857 * Q.mass - 18.92285
    if 82.85409 <= Q.mass < 91.03469:
        z += 0.1180323 * Q.mass - 10.5929
    if 91.03469 <= Q.mass < 121.3913:
        z += -0.005011815 * Q.mass + 0.6083906
    if Q.psi_0p3 >= 0.9924477:
        z += -47.79012 * Q.psi_0p3 + 47.42919
    if Q.N2 < 0.4226723:
        z += 1.001672 * Q.N2 - 0.4233789
    if Q.n_dr_0p1_0p2 < 19.0:
        z += 0.02274762 * Q.n_dr_0p1_0p2 - 0.4322048
    if Q.e2 < 0.01256572:
        z += 34.21296 * Q.e2 + 0.6675028
    if 0.01256572 <= Q.e2 < 0.03029714:
        z += -48.75998 * Q.e2 + 1.710118
    if 0.03029714 <= Q.e2 < 0.03680582:
        z += -35.77221 * Q.e2 + 1.316625
    if Q.mass_over_sum_pt < 0.08873143:
        z += 7.346952 * Q.mass_over_sum_pt + 0.4594296
    if 0.08873143 <= Q.mass_over_sum_pt < 0.09046749:
        z += 68.72917 * Q.mass_over_sum_pt - 4.987102
    if 0.09046749 <= Q.mass_over_sum_pt < 0.09795415:
        z += -64.65815 * Q.mass_over_sum_pt + 7.080114
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -36.82854 * Q.mass_over_sum_pt + 4.354089
    if Q.sum_z_dr2_top20 < 0.006374178:
        z += 57.56121 * Q.sum_z_dr2_top20 - 0.9514869
    if 0.006374178 <= Q.sum_z_dr2_top20 < 0.01083435:
        z += 131.0669 * Q.sum_z_dr2_top20 - 1.420025
    if Q.mass_over_sum_pt_sq < 0.01986381:
        z += -154.3518 * Q.mass_over_sum_pt_sq + 3.066015
    if Q.soft7_z < 0.001595561:
        z += 268.3684 * Q.soft7_z - 0.4281982
    if Q.sum_z_dr2_top3 < 0.001592178:
        z += 390.8474 * Q.sum_z_dr2_top3 - 0.6222984
    if Q.n_pt_above_10 >= 11.0:
        z += -0.03778889 * Q.n_pt_above_10 + 0.4156778
    if Q.mass_top40 < 67.72643:
        z += -0.08867661 * Q.mass_top40 + 5.675108
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.02119624 * Q.mass_top40 - 1.766188
    if Q.sum_z_dr2_top40 < 0.01292642:
        z += 174.5665 * Q.sum_z_dr2_top40 - 2.506591
    if 0.01292642 <= Q.sum_z_dr2_top40 < 0.01897915:
        z += 41.31546 * Q.sum_z_dr2_top40 - 0.7841322
    if Q.tau2 < 0.04142826:
        z += 6.23804 * Q.tau2 - 0.2584311
    if Q.tau1 < 0.1008497:
        z += 7.694597 * Q.tau1 - 0.7759979
    if Q.dr_3 < 0.03948167:
        z += -14.44481 * Q.dr_3 + 0.5703054
    if Q.tau21_b2 < 0.342495 and Q.lam1 < 0.02048524:
        z += -46.10193 * (0.342495 - Q.tau21_b2) * (0.02048524 - Q.lam1)
    if Q.N2 < 0.4226723 and Q.max_dr > 0.2738063:
        z += -12.14911 * (0.4226723 - Q.N2) * (Q.max_dr - 0.2738063)
    if Q.mass < 121.3913 and Q.mass_top2 > 36.76827:
        z += -0.001630815 * (121.3913 - Q.mass) * (Q.mass_top2 - 36.76827)
    if Q.mass < 82.85409 and Q.n_for_50pct > 5.0:
        z += 0.01461299 * (82.85409 - Q.mass) * (Q.n_for_50pct - 5.0)
    if Q.tau21_b2 < 0.342495 and Q.z_1 < 0.1231747:
        z += -37.72895 * (0.342495 - Q.tau21_b2) * (0.1231747 - Q.z_1)
    if Q.sum_z_dr2_top20 < 0.01083435 and Q.mass_top3 > 16.89912:
        z += -2.452624 * (0.01083435 - Q.sum_z_dr2_top20) * (Q.mass_top3 - 16.89912)
    if Q.psi_0p3 > 0.9924477 and Q.dr_3 < 0.0458688:
        z += -3185.725 * (Q.psi_0p3 - 0.9924477) * (0.0458688 - Q.dr_3)
    if Q.mass < 121.3913 and Q.zdr_3 > 0.0003281655:
        z += -1.119562 * (121.3913 - Q.mass) * (Q.zdr_3 - 0.0003281655)
    if Q.mass < 91.03469 and Q.soft8_z < 0.002405315:
        z += 32.13681 * (91.03469 - Q.mass) * (0.002405315 - Q.soft8_z)
    if Q.psi_0p3 > 0.9924477 and Q.pt_3 > 60.59375:
        z += 1.221472 * (Q.psi_0p3 - 0.9924477) * (Q.pt_3 - 60.59375)
    if Q.psi_0p3 > 0.9924477 and Q.soft6_pt < 4.250195:
        z += 9.968657 * (Q.psi_0p3 - 0.9924477) * (4.250195 - Q.soft6_pt)
    if Q.tau21_b2 < 0.342495 and Q.z_6 < 0.05799644:
        z += 32.48976 * (0.342495 - Q.tau21_b2) * (0.05799644 - Q.z_6)
    if Q.psi_0p3 > 0.9924477 and Q.N3 < 1.059188:
        z += 124.9068 * (Q.psi_0p3 - 0.9924477) * (1.059188 - Q.N3)
    if Q.tau21_b2 < 0.342495 and Q.sj2_zsoft < 0.34119:
        z += -3.15415 * (0.342495 - Q.tau21_b2) * (0.34119 - Q.sj2_zsoft)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.5934045
    if Q.z_dr_0_0p05 < 0.8459004:
        z += 0.4567752 * Q.z_dr_0_0p05 - 0.3863863
    if Q.sum_z_dr2_top20 < 0.00287991:
        z += -65.89934 * Q.sum_z_dr2_top20 + 0.1897842
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -6.447457 * Q.z_dr_0p1_0p2 + 0.7759106
    if Q.sum_z_dr2_top5 < 0.002270363:
        z += 98.59003 * Q.sum_z_dr2_top5 + 0.4762679
    if 0.002270363 <= Q.sum_z_dr2_top5 < 0.008329695:
        z += -115.5413 * Q.sum_z_dr2_top5 + 0.9624237
    if Q.sum_pt < 986.0565:
        z += 0.007415857 * Q.sum_pt - 7.514654
    if 986.0565 <= Q.sum_pt < 1002.379:
        z += 0.01238817 * Q.sum_pt - 12.41764
    if Q.log_sum_pt < 6.903423:
        z += -1.249676 * Q.log_sum_pt + 9.404904
    if 6.903423 <= Q.log_sum_pt < 7.017258:
        z += -6.833269 * Q.log_sum_pt + 47.95081
    if Q.psi_0p3 >= 0.9896594:
        z += -54.35542 * Q.psi_0p3 + 53.79335
    if Q.sj2_dr >= 0.2232169:
        z += 2.748398 * Q.sj2_dr - 0.6134889
    if Q.sum_z_dr2_top10 < 0.007678544:
        z += -34.36351 * Q.sum_z_dr2_top10 + 0.2638617
    if Q.n_dr_0p1_0p2 < 13.0:
        z += -0.0170685 * Q.n_dr_0p1_0p2 + 0.2218905
    if Q.lam2 < 0.001776308:
        z += -188.7687 * Q.lam2 + 0.3353114
    if Q.tau1 < 0.0705748:
        z += 19.75679 * Q.tau1 - 1.394331
    if Q.sum_pt_top20 < 869.693:
        z += 0.0004156811 * Q.sum_pt_top20 - 0.5335892
    if 869.693 <= Q.sum_pt_top20 < 1017.778:
        z += 0.001161996 * Q.sum_pt_top20 - 1.182654
    if Q.mass_over_sum_pt >= 0.1708801:
        z += -462.0352 * Q.mass_over_sum_pt + 78.95262
    if Q.n_dr_0p2_0p4 >= 9.0:
        z += -0.03860363 * Q.n_dr_0p2_0p4 + 0.3474327
    if Q.sd_mass < 40.97891:
        z += 0.000772709 * Q.sd_mass + 0.7678743
    if 40.97891 <= Q.sd_mass < 79.18312:
        z += -0.02092804 * Q.sd_mass + 1.657147
    if Q.mass_top50 < 52.15154:
        z += -0.007233478 * Q.mass_top50 + 0.1691895
    if 52.15154 <= Q.mass_top50 < 71.79516:
        z += 0.0105911 * Q.mass_top50 - 0.7603895
    if Q.lam1 < 0.004673423:
        z += -69.89853 * Q.lam1 + 0.3266654
    if Q.sum_pt_top30 < 1018.832:
        z += 0.0006493807 * Q.sum_pt_top30 - 0.6616096
    if Q.psi_0p1 >= 0.9371031:
        z += -6.229167 * Q.psi_0p1 + 5.837371
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.sd_mass < 76.29481:
        z += -0.04437039 * (0.1203437 - Q.z_dr_0p1_0p2) * (76.29481 - Q.sd_mass)
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 949.9169:
        z += 0.01022072 * (0.8459004 - Q.z_dr_0_0p05) * (949.9169 - Q.sum_pt)
    if Q.sj2_dr > 0.2232169 and Q.C2_b2 < 0.04008677:
        z += 159.5411 * (Q.sj2_dr - 0.2232169) * (0.04008677 - Q.C2_b2)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.sj3_mass1 > 5.112677:
        z += -1.373813 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.sj3_mass1 - 5.112677)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.sj3_pairmin_over_m > 0.09540583:
        z += -92.18994 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.sj3_pairmin_over_m - 0.09540583)
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
