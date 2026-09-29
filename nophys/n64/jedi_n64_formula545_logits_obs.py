"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
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
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
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
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
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
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
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
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
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
    )


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-1.078125
        + 0.625 * grid(1, max(0.0, 0.4056649
            + 0.1140403 * max(0.0, Q.n_particles - 38.0)
            + 13.28243 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.008938069 * max(0.0, Q.sum_pt_top50 - 959.0957)
            - 12.93139 * max(0.0, Q.log_sum_pt - 6.98945)
            - 163.8747 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 0.002081384 * max(0.0, 689.25 - Q.sum_pt_top2)
            - 5.622614 * max(0.0, Q.z_top30_slots - 0.9341838)
            - 0.04089765 * max(0.0, 47.88842 - Q.mass_top20)
            + 0.4723904 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793)
            + 0.006226552 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.1764314 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0)
            + 0.004162062 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0)
            - 0.03961573 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt)
            + 109.8346 * max(0.0, 0.003270031 - Q.girth2_top15)
            - 0.08756057 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            + 205.2069 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2)
            + 596.8324 * max(0.0, 0.0005522528 - Q.girth2_top3)
            - 0.001529776 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2)
            - 17.21979 * max(0.0, 0.03187688 - Q.M3)
            + 45202.52 * max(0.0, 0.003270031 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 0.02674561 * max(0.0, 30.26161 - Q.sj2_mass1)
            - 344.7084 * max(0.0, 0.03187688 - Q.M3) * max(0.0, Q.M2 - 0.05568888)
            - 0.03432387 * max(0.0, 31.35938 - Q.pt_9)
            + 0.1269012 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1)
            + 0.6182076 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874)
            + 29.52066 * max(0.0, Q.log_sum_pt - 6.893714)
            + 107.6739 * max(0.0, 0.004673423 - Q.lam1)
            - 0.03089143 * max(0.0, 101.0497 - Q.mass)
            + 0.0224856 * max(0.0, 89.74183 - Q.mass)
            + 0.0009382492 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 15.47191 - Q.pair_mass_0_12)
            + 1.071058 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.01395978 - Q.zdr_2)
            + 1.967991 * max(0.0, 0.08822608 - Q.D3)
            - 8.180573 * max(0.0, 7.139296 - Q.log_sum_pt)
            - 4.456728 * max(0.0, Q.z_dr_0_0p05 - 0.8459004)
            + 0.007336965 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, 30.0 - Q.n_real_top30)
            + 0.4584168 * max(0.0, 1.521582 - Q.soft1_pt)
            + 1.377801 * max(0.0, Q.sj2_zsoft - 0.2912328)
            + 608.1091 * max(0.0, 0.0007894752 - Q.girth2_top15)
            - 19.15233 * max(0.0, Q.log_sum_pt - 6.959294)
            + 0.01136718 * max(0.0, 1017.435 - Q.sum_pt)
            + 0.003956569 * max(0.0, Q.sum_pt_top30 - 1191.938)
            + 6.291815 * max(0.0, Q.z_top30_slots - 0.9564984)
            + 0.01230519 * max(0.0, 80.24626 - Q.mass_top30)
            + 0.02558681 * max(0.0, 10.0 - Q.n_dr_0_0p05)
            + 11.87894 * max(0.0, 0.1751567 - Q.tau1)
            + 0.8098888 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, Q.ptdr0_4 - 4.470953)
            + 0.004992259 * max(0.0, Q.mass_top10 - 31.33272)
            - 15.39291 * max(0.0, Q.zdr_0 - 0.001901263)
        ))
        - 0.75 * grid(3, max(0.0, -0.2874307
            + 1001.893 * max(0.0, 0.0006154841 - Q.lam2)
            + 0.1648075 * max(0.0, 5.0 - Q.n_dr_0p2_0p4)
            + 0.02171065 * max(0.0, 46.0 - Q.n_particles)
            - 0.001259068 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349)
            - 0.01758449 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0)
            + 6.457543e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732)
            - 1.786492 * max(0.0, 0.3861957 - Q.tau21)
            + 160.963 * max(0.0, 0.006026828 - Q.girth2_top40)
            + 154.8658 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 12.43015 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5)
            + 0.6231591 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1)
            - 345.0482 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243)
            - 0.01103303 * max(0.0, 79.21004 - Q.mass_top50)
            + 0.005042994 * max(0.0, 87.36377 - Q.mass)
            - 129.7488 * max(0.0, 0.009614971 - Q.girth2)
            + 107.1983 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 0.03754923 * max(0.0, 8.0 - Q.n_dr_0p2_0p4)
            + 0.05043822 * max(0.0, 10.0 - Q.n_dr_0p1_0p2)
            - 0.01195483 * max(0.0, 53.87362 - Q.mass)
            - 178.3918 * max(0.0, 0.006259772 - Q.girth2_top40)
            - 15.76641 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.006794973 - Q.zdr_4)
            + 111.8487 * max(0.0, 0.008124776 - Q.girth2_top50)
        ))
        - 0.875 * grid(4, max(0.0, 0.6914295
            + 0.01186823 * max(0.0, 69.02716 - Q.mass_top15)
            + 0.02282503 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            + 0.06144266 * max(0.0, 83.32554 - Q.mass_top40)
            - 1.018753 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.z_top50_slots)
            + 0.008792364 * max(0.0, 60.43821 - Q.mass_top30)
            + 0.02188353 * max(0.0, 121.3913 - Q.mass)
            + 111.2192 * max(0.0, Q.psi_0p3 - 0.9973959)
            - 0.07024051 * max(0.0, 87.36377 - Q.mass)
            - 0.009403079 * max(0.0, 121.737 - Q.mass_top30)
            + 0.008023934 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.01845784 * max(0.0, Q.n_particles - 22.0)
            - 226.0949 * max(0.0, 0.004855289 - Q.girth2_top15)
            + 0.02503371 * max(0.0, Q.n_dr_0_0p05 - 5.0)
            - 2.291058 * max(0.0, Q.sj2_dr - 0.2232169)
            - 0.01352989 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 6.916121 - Q.D2)
            + 0.009918761 * max(0.0, 101.0497 - Q.mass)
            - 0.0181479 * max(0.0, 67.72643 - Q.mass_top40)
            + 0.01852338 * max(0.0, 79.65241 - Q.mass)
            + 81.86659 * max(0.0, Q.e2_sq - 0.01396296)
            - 3.264511 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.zdr_0 - 0.006864207)
            - 22.59892 * max(0.0, Q.girth2_top15 - 0.007887677)
            - 22.2738 * max(0.0, 79.65241 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 0.06240618 * max(0.0, 74.25181 - Q.mass)
            - 0.003695312 * max(0.0, 1042.609 - Q.sum_pt)
            + 2.637887e-05 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, 1053.047 - Q.sum_pt_top40)
            - 0.04232617 * max(0.0, 94.64253 - Q.mass_top40)
            - 44.61525 * max(0.0, 0.007678544 - Q.girth2_top10)
            + 13.39343 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 152.651 * max(0.0, 0.001776308 - Q.lam2)
            + 0.1559495 * max(0.0, 5.378975 - Q.D2)
            - 0.1386088 * max(0.0, 1.0 - Q.sd_nremoved)
            + 7.41268 * max(0.0, 0.004855289 - Q.girth2_top15) * max(0.0, Q.n_dr_0p05_0p1 - 2.0)
            - 1.489265 * max(0.0, Q.psi_0p1 - 0.8747961)
            - 0.009699348 * max(0.0, Q.mass_top10 - 76.9886)
        ))
        - 0.3125 * grid(5, max(0.0, 1.365113
            + 0.03550476 * max(0.0, 64.0 - Q.n_particles)
            - 31.55748 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 6.215755 * max(0.0, Q.log_sum_pt - 6.935549)
            + 0.007034252 * max(0.0, Q.sum_pt - 907.9372)
            - 9.741529 * max(0.0, Q.log_sum_pt - 6.920349)
            + 194.3873 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 38.32114 * max(0.0, Q.girth2_top50 - 0.01951641)
            - 0.01375075 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.02108265 * max(0.0, Q.sum_pt_top50 - 934.2416)
            + 0.01997073 * max(0.0, Q.n_pt_above_1 - 28.0)
            - 0.0006994508 * max(0.0, 787.6281 - Q.sum_pt_top3)
            + 0.03204697 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 9831.014 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4)
            - 20.85056 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.005053338 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 5.579533 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.002719696 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 2.187253 * max(0.0, Q.max_dr - 0.2404747)
            - 8.345629 * max(0.0, Q.z_top30_slots - 0.9048492)
            + 0.01531813 * max(0.0, Q.mass - 172.4888)
            + 34.27206 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2)
            - 0.01391004 * max(0.0, 150.0144 - Q.mass_top40)
            - 0.05923529 * max(0.0, Q.mass_top50 - 157.5448)
            - 543.0219 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 0.03605449 * max(0.0, Q.sd_mass - 69.65633)
            - 1.101833 * max(0.0, 0.5100475 - Q.tau21)
            - 0.01352748 * max(0.0, Q.sum_pt - 986.0565)
            - 0.002087266 * max(0.0, Q.sum_pt_top10 - 943.6922)
            + 151.2781 * max(0.0, 0.01375115 - Q.z_11)
            - 5.676478 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.1958008 - Q.absphi_13)
            - 0.134532 * max(0.0, 14.14062 - Q.pt_11)
            + 21.40778 * max(0.0, 0.04082832 - Q.e2)
            - 6.487458 * max(0.0, 0.1507173 - Q.tau1)
            - 0.04655232 * max(0.0, Q.sd_mass - 83.30647)
            + 0.01805284 * max(0.0, 64.48544 - Q.mass)
            - 10.83178 * max(0.0, 0.03700182 - Q.z_dr_0p2_0p4)
            - 0.01411033 * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 0.01217882 * max(0.0, Q.mass_top10 - 71.781)
            - 0.6472427 * max(0.0, 0.707925 - Q.psi_0p1)
            + 0.01920067 * max(0.0, 23.45965 - Q.sj3_mass1)
            + 1.7528 * max(0.0, Q.max_dr - 0.4357228)
        ))
        + 0.15625 * grid(6, max(0.0, 0.3591835
            + 0.03309143 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.04070815 * max(0.0, 121.3913 - Q.mass)
            + 1131.49 * max(0.0, 0.0003372339 - Q.e3)
            + 0.03032547 * max(0.0, 82.85409 - Q.mass)
            + 6.305604e-05 * max(0.0, 121.3913 - Q.mass) * max(0.0, 1003.544 - Q.sum_pt_top50)
            + 36.44807 * max(0.0, Q.e2 - 0.04755309)
            + 2232.745 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.psi_0p3 - 0.9896594)
            + 0.006413691 * max(0.0, 172.4888 - Q.mass)
            + 0.03312753 * max(0.0, 89.74183 - Q.mass)
            - 0.07483936 * max(0.0, 101.0497 - Q.mass)
            + 8.79812 * max(0.0, 0.06310829 - Q.tau1)
            - 1.995063 * max(0.0, Q.sj3_dr_min - 0.1204829)
            + 0.05031243 * max(0.0, Q.n_dr_0p1_0p2 - 33.0)
            - 0.02226872 * max(0.0, Q.sj3_mass1 - 19.30204)
            - 2.192575e-05 * max(0.0, 89.74183 - Q.mass) * max(0.0, 1129.275 - Q.sum_pt_top20)
            + 81.49459 * max(0.0, 0.003687605 - Q.lam2)
            + 0.05797808 * max(0.0, 92.85979 - Q.mass)
            - 124.6715 * max(0.0, 0.001425993 - Q.girth2_top2)
            - 0.3986125 * max(0.0, 2.178951 - Q.D2)
            + 0.006799873 * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            - 4492.619 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.z_14 - 0.01634243)
            - 0.005595608 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) * max(0.0, 9.0 - Q.n_dr_0_0p05)
            - 0.1380028 * max(0.0, 0.07952881 - Q.eta_0) * max(0.0, Q.n_dr_0_0p05 - 5.0)
            + 153.1398 * max(0.0, 0.007259287 - Q.lam1)
            - 41.21214 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            + 203.3618 * max(0.0, 0.02580859 - Q.e2_sq)
            - 48.02988 * max(0.0, 0.01649354 - Q.lam1)
            - 75.8776 * max(0.0, 0.02550569 - Q.girth2_top50)
            + 124.4262 * max(0.0, 0.008376291 - Q.girth2_top30)
            - 84.07935 * max(0.0, 0.02412652 - Q.girth2_top30)
            + 6.167434 * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) * max(0.0, Q.soft10_dr0 - 0.1909669)
            + 80.74909 * max(0.0, 0.00287991 - Q.girth2_top20)
            - 1.919322 * max(0.0, Q.soft10_dr - 0.2075213)
            - 1.790278 * max(0.0, Q.e2 - 0.02793599) * max(0.0, 6.06065 - Q.ptdr0_13)
        ))
        + 0.03125 * grid(8, max(0.0, -0.8586652
            + 44.22524 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 74.2517 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.03063734 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.01859725 * max(0.0, Q.mass - 121.3913)
            + 0.004750203 * max(0.0, Q.mass - 89.74183)
            + 4.842668 * max(0.0, Q.log_sum_pt - 6.959294)
            - 0.00370839 * max(0.0, Q.sum_pt_top40 - 858.8262)
            - 0.02112566 * max(0.0, Q.mass_top20 - 90.4505)
            - 0.09964249 * max(0.0, Q.mass - 101.0497)
            + 0.6663901 * max(0.0, 1001.523 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477)
            + 149.2864 * max(0.0, Q.girth2_top40 - 0.008031986)
            + 0.03894753 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435)
            + 0.0376181 * max(0.0, Q.mass - 64.48544)
            + 401.3352 * max(0.0, 0.007887677 - Q.girth2_top15)
            - 0.003470294 * max(0.0, 1032.405 - Q.sum_pt_top40)
            - 0.05127087 * max(0.0, Q.mass_top50 - 80.35535)
            - 440.128 * max(0.0, Q.e2_sq - 0.009606007)
            + 3.355825 * max(0.0, Q.sj2_dr - 0.2232169)
            + 0.03997092 * max(0.0, Q.n_dr_0p2_0p4 - 8.0)
            + 80.63364 * max(0.0, Q.girth2_top15 - 0.001319197)
            + 31.16157 * max(0.0, Q.e2 - 0.04755309)
            + 0.007857585 * max(0.0, 1003.544 - Q.sum_pt_top50)
            - 185.7492 * max(0.0, 0.007887677 - Q.girth2_top15) * max(0.0, 0.6133424 - Q.tau21_b2)
            + 0.02347445 * max(0.0, Q.n_dr_0p1_0p2 - 19.0)
            - 0.7297459 * max(0.0, 1032.405 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477)
            - 10.45823 * max(0.0, Q.tau1 - 0.06310829)
            - 0.0002108822 * max(0.0, 1003.544 - Q.sum_pt_top50) * max(0.0, 21.0 - Q.n_dr_0p05_0p1)
            + 216.2895 * max(0.0, Q.e2_sq - 0.007872294)
            - 181.1447 * max(0.0, Q.girth2_top40 - 0.006026828)
            + 0.03090803 * max(0.0, Q.n_pt_above_1 - 54.0)
            - 5.7359 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 2.563115 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4)
            + 0.008906562 * max(0.0, Q.mass_top40 - 111.2487)
            + 5.433696 * max(0.0, Q.LHA - 0.3332345)
            + 0.04531039 * max(0.0, Q.mass - 80.78464)
            - 0.01161229 * max(0.0, Q.mass - 143.7876)
            + 0.04554217 * max(0.0, Q.mass_top50 - 97.93004)
            + 3178.132 * max(0.0, 0.0003372339 - Q.e3)
            - 38.94893 * max(0.0, 0.02737453 - Q.girth2_top20)
            + 7.375078 * max(0.0, 6.935549 - Q.log_sum_pt)
            + 3.189916 * max(0.0, Q.C2 - 0.07996447)
            - 34.11252 * max(0.0, 0.007856958 - Q.girth2_top30)
            + 95.89755 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266)
            - 90.39785 * max(0.0, 0.01563836 - Q.girth2_top15)
            - 110.2406 * max(0.0, 0.005383629 - Q.girth2_top15)
            - 0.00100911 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) * max(0.0, 26.76006 - Q.sj3_mass1)
        ))
        + 0.515625 * grid(9, max(0.0, -0.1731589
            + 0.005050553 * max(0.0, 80.89043 - Q.mass_top40)
            + 263.6798 * max(0.0, 0.006259772 - Q.girth2_top40)
            + 0.003046227 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258)
            - 0.0002116931 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt)
            + 0.0163929 * max(0.0, 121.737 - Q.mass_top30)
            - 0.06997643 * max(0.0, Q.mass - 143.7876)
            + 0.06260496 * max(0.0, Q.mass - 172.4888)
            + 35.81701 * max(0.0, Q.girth - 0.08589404)
            + 0.0428226 * max(0.0, 87.36377 - Q.mass)
            + 0.0008216668 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.n_particles - 38.0)
            + 0.0008920693 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 934.2416 - Q.sum_pt_top50)
            + 0.0009028777 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, Q.n_dr_0p2_0p4 - 10.0)
            - 0.01478441 * max(0.0, 152.6883 - Q.mass_top30)
            - 0.0009061 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, 20.0 - Q.n_pt_above_10)
            - 7.363469 * max(0.0, Q.z_top40_slots - 0.9574183)
            + 0.003105244 * max(0.0, 959.0957 - Q.sum_pt_top50)
            + 88.79276 * max(0.0, Q.lam1 - 0.01174405)
            - 0.01031488 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.894531)
            + 36.1164 * max(0.0, 0.01083435 - Q.girth2_top20)
            - 0.006491384 * max(0.0, 89.6788 - Q.mass_top40)
            - 186.3027 * max(0.0, Q.girth2_top30 - 0.02412652)
            - 0.06762542 * max(0.0, 64.48544 - Q.mass)
            + 0.04530099 * max(0.0, 79.65241 - Q.mass)
            - 38.68333 * max(0.0, 0.05444509 - Q.tau1)
            - 133.8495 * max(0.0, 0.004673423 - Q.lam1)
            + 16.74914 * max(0.0, 0.2091025 - Q.LHA)
            + 0.004939419 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft5_pt - 2.894531)
            + 1.280691 * max(0.0, 0.2187642 - Q.z_dr_0p1_0p2)
            + 17239.04 * max(0.0, Q.lam1 - 0.01174405) * max(0.0, Q.soft5_z - 0.002346473)
        ))
        - 0.015625 * grid(10, max(0.0, -3.948963
            - 8.881083 * max(0.0, 0.1207452 - Q.girth)
            - 0.04043477 * max(0.0, Q.mass - 162.8363)
            - 74.06046 * max(0.0, 0.005402331 - Q.girth2_top30)
            - 0.03294256 * max(0.0, Q.mass_top40 - 163.2541)
            - 0.02077042 * max(0.0, 80.3008 - Q.sj2_mass1)
            + 0.04475454 * max(0.0, 87.27603 - Q.mass_top40)
            + 0.232346 * max(0.0, 3.814159 - Q.D2)
            - 215.0982 * max(0.0, Q.psi_0p3 - 0.9985421)
            + 2.374723e-05 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, 464.75 - Q.sum_pt_top2)
            + 0.008188306 * max(0.0, Q.mass_top5 - 14.54404)
            - 2.467481 * max(0.0, 3.814159 - Q.D2) * max(0.0, Q.sj2_dr - 0.1937688)
            + 30.05285 * max(0.0, Q.mass_top40 - 163.2541) * max(0.0, Q.soft4_z - 0.001186042)
            + 0.01090531 * max(0.0, Q.mass_top50 - 157.5448)
            - 1.393145 * max(0.0, Q.z_dr_0_0p05 - 0.7674734)
            - 7.537004 * max(0.0, Q.log_sum_pt - 6.903423)
            + 0.08921913 * max(0.0, Q.mass - 82.85409)
            - 0.07994704 * max(0.0, Q.mass - 143.7876)
            - 0.05977461 * max(0.0, Q.mass - 64.48544)
            + 0.004303969 * max(0.0, Q.mass - 64.48544) * max(0.0, 2.537109 - Q.soft2_pt)
            + 0.003656236 * Q.sum_pt
            - 0.003736029 * max(0.0, 75.26407 - Q.mass_top15)
            + 19.25813 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.0830523 - Q.dr_4)
            - 310.9234 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            + 0.001829684 * max(0.0, 839.9547 - Q.sum_pt_top5)
            + 0.0008142922 * max(0.0, 111.2487 - Q.mass_top40) * max(0.0, Q.pt1_dr01 - 12.6865)
            - 22.70827 * max(0.0, 0.06524004 - Q.e2)
            + 92.61685 * max(0.0, 0.01807679 - Q.girth2_top30)
            + 6.914821 * max(0.0, 0.06413297 - Q.dr_0)
            - 0.04096867 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 4.206493 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            + 0.02819799 * max(0.0, 132.4278 - Q.mass_top40)
            - 0.05756879 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.02782281 * max(0.0, Q.mass_top40 - 67.72643)
            - 0.006647105 * max(0.0, 71.781 - Q.mass_top10)
            + 727.3351 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 4.029475 * max(0.0, 0.06940472 - Q.dr_1)
            - 40.51704 * max(0.0, 0.05444509 - Q.tau1)
            + 0.02880678 * max(0.0, 74.25181 - Q.mass)
            + 0.0002261132 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.91979)
            + 1.226455 * max(0.0, 0.6133424 - Q.tau21_b2)
        ))
        + 0.234375 * grid(12, max(0.0, -0.1994202
            + 0.01223357 * max(0.0, 87.36377 - Q.mass)
            - 0.09773045 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            - 0.03374758 * max(0.0, 94.64253 - Q.mass_top40)
            - 2.682579 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3)
            + 0.9075831 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367)
            - 11.1208 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.lam2 - 0.0003497174)
            + 0.05174195 * max(0.0, 82.85409 - Q.mass)
            - 373.0883 * max(0.0, 0.00363788 - Q.e2_sq)
            - 0.003510133 * max(0.0, 77.93668 - Q.mass_top40)
            - 0.04962413 * max(0.0, 53.87362 - Q.mass)
            + 3.315602e-05 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 435.25 - Q.pt_0)
            + 36.39789 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.3036026 - Q.planar_flow)
            + 0.001229078 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1)
            + 0.06322242 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            - 0.001456983 * max(0.0, Q.sum_pt - 1260.541)
            - 0.03622926 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.soft1_pt - 0.8144531)
            - 0.003879674 * max(0.0, 972.0419 - Q.sum_pt)
            + 4.797264 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4)
            - 5714.218 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.001776308 - Q.lam2)
            + 0.07497664 * max(0.0, 80.78464 - Q.mass)
            + 13.44873 * max(0.0, 77.93668 - Q.mass_top40) * max(0.0, Q.lam2 - 0.000404306)
            - 4.510786 * max(0.0, Q.z_dr_0_0p05 - 0.9084912)
        ))
    )


def logit_q(Q):
    return (1.359375
        - 0.1875 * grid(1, max(0.0, 0.4056649
            + 0.1140403 * max(0.0, Q.n_particles - 38.0)
            + 13.28243 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.008938069 * max(0.0, Q.sum_pt_top50 - 959.0957)
            - 12.93139 * max(0.0, Q.log_sum_pt - 6.98945)
            - 163.8747 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 0.002081384 * max(0.0, 689.25 - Q.sum_pt_top2)
            - 5.622614 * max(0.0, Q.z_top30_slots - 0.9341838)
            - 0.04089765 * max(0.0, 47.88842 - Q.mass_top20)
            + 0.4723904 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793)
            + 0.006226552 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.1764314 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0)
            + 0.004162062 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0)
            - 0.03961573 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt)
            + 109.8346 * max(0.0, 0.003270031 - Q.girth2_top15)
            - 0.08756057 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            + 205.2069 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2)
            + 596.8324 * max(0.0, 0.0005522528 - Q.girth2_top3)
            - 0.001529776 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2)
            - 17.21979 * max(0.0, 0.03187688 - Q.M3)
            + 45202.52 * max(0.0, 0.003270031 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 0.02674561 * max(0.0, 30.26161 - Q.sj2_mass1)
            - 344.7084 * max(0.0, 0.03187688 - Q.M3) * max(0.0, Q.M2 - 0.05568888)
            - 0.03432387 * max(0.0, 31.35938 - Q.pt_9)
            + 0.1269012 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1)
            + 0.6182076 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874)
            + 29.52066 * max(0.0, Q.log_sum_pt - 6.893714)
            + 107.6739 * max(0.0, 0.004673423 - Q.lam1)
            - 0.03089143 * max(0.0, 101.0497 - Q.mass)
            + 0.0224856 * max(0.0, 89.74183 - Q.mass)
            + 0.0009382492 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 15.47191 - Q.pair_mass_0_12)
            + 1.071058 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.01395978 - Q.zdr_2)
            + 1.967991 * max(0.0, 0.08822608 - Q.D3)
            - 8.180573 * max(0.0, 7.139296 - Q.log_sum_pt)
            - 4.456728 * max(0.0, Q.z_dr_0_0p05 - 0.8459004)
            + 0.007336965 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, 30.0 - Q.n_real_top30)
            + 0.4584168 * max(0.0, 1.521582 - Q.soft1_pt)
            + 1.377801 * max(0.0, Q.sj2_zsoft - 0.2912328)
            + 608.1091 * max(0.0, 0.0007894752 - Q.girth2_top15)
            - 19.15233 * max(0.0, Q.log_sum_pt - 6.959294)
            + 0.01136718 * max(0.0, 1017.435 - Q.sum_pt)
            + 0.003956569 * max(0.0, Q.sum_pt_top30 - 1191.938)
            + 6.291815 * max(0.0, Q.z_top30_slots - 0.9564984)
            + 0.01230519 * max(0.0, 80.24626 - Q.mass_top30)
            + 0.02558681 * max(0.0, 10.0 - Q.n_dr_0_0p05)
            + 11.87894 * max(0.0, 0.1751567 - Q.tau1)
            + 0.8098888 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, Q.ptdr0_4 - 4.470953)
            + 0.004992259 * max(0.0, Q.mass_top10 - 31.33272)
            - 15.39291 * max(0.0, Q.zdr_0 - 0.001901263)
        ))
        + 0.125 * grid(2, max(0.0, -0.002347209
            - 9.378196 * max(0.0, Q.log_sum_pt - 7.062574)
            + 0.01301346 * max(0.0, Q.sum_pt - 1017.435)
            - 0.008638248 * max(0.0, 51.0 - Q.n_particles)
            + 699.1433 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15)
            - 21.88492 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135)
            - 0.395954 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15)
            + 0.009690578 * max(0.0, Q.sum_pt_top50 - 1156.659)
            + 0.02754601 * max(0.0, 91.03469 - Q.mass)
            - 0.005048876 * max(0.0, 82.66587 - Q.mass_top30)
            + 0.001967088 * max(0.0, 1041.263 - Q.sum_pt_top40)
            - 0.001583572 * max(0.0, 996.8867 - Q.sum_pt_top30)
            - 0.004917892 * max(0.0, Q.sum_pt - 1115.723)
            + 126.6648 * max(0.0, 0.006088416 - Q.girth2_top30)
            - 0.002069385 * max(0.0, 1260.541 - Q.sum_pt)
            - 0.05361438 * max(0.0, 92.85979 - Q.mass)
            + 0.0152174 * max(0.0, 121.3913 - Q.mass)
            - 0.0005981218 * max(0.0, 787.6281 - Q.sum_pt_top3)
            - 0.01103883 * max(0.0, Q.sum_pt - 1052.889)
            - 105.2892 * max(0.0, 0.008124776 - Q.girth2_top50)
            + 0.002673855 * max(0.0, 1053.047 - Q.sum_pt_top40)
            + 0.00476503 * max(0.0, 143.7876 - Q.mass)
        ))
        - 1.0625 * grid(4, max(0.0, 0.6914295
            + 0.01186823 * max(0.0, 69.02716 - Q.mass_top15)
            + 0.02282503 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            + 0.06144266 * max(0.0, 83.32554 - Q.mass_top40)
            - 1.018753 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.z_top50_slots)
            + 0.008792364 * max(0.0, 60.43821 - Q.mass_top30)
            + 0.02188353 * max(0.0, 121.3913 - Q.mass)
            + 111.2192 * max(0.0, Q.psi_0p3 - 0.9973959)
            - 0.07024051 * max(0.0, 87.36377 - Q.mass)
            - 0.009403079 * max(0.0, 121.737 - Q.mass_top30)
            + 0.008023934 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.01845784 * max(0.0, Q.n_particles - 22.0)
            - 226.0949 * max(0.0, 0.004855289 - Q.girth2_top15)
            + 0.02503371 * max(0.0, Q.n_dr_0_0p05 - 5.0)
            - 2.291058 * max(0.0, Q.sj2_dr - 0.2232169)
            - 0.01352989 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 6.916121 - Q.D2)
            + 0.009918761 * max(0.0, 101.0497 - Q.mass)
            - 0.0181479 * max(0.0, 67.72643 - Q.mass_top40)
            + 0.01852338 * max(0.0, 79.65241 - Q.mass)
            + 81.86659 * max(0.0, Q.e2_sq - 0.01396296)
            - 3.264511 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.zdr_0 - 0.006864207)
            - 22.59892 * max(0.0, Q.girth2_top15 - 0.007887677)
            - 22.2738 * max(0.0, 79.65241 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 0.06240618 * max(0.0, 74.25181 - Q.mass)
            - 0.003695312 * max(0.0, 1042.609 - Q.sum_pt)
            + 2.637887e-05 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, 1053.047 - Q.sum_pt_top40)
            - 0.04232617 * max(0.0, 94.64253 - Q.mass_top40)
            - 44.61525 * max(0.0, 0.007678544 - Q.girth2_top10)
            + 13.39343 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 152.651 * max(0.0, 0.001776308 - Q.lam2)
            + 0.1559495 * max(0.0, 5.378975 - Q.D2)
            - 0.1386088 * max(0.0, 1.0 - Q.sd_nremoved)
            + 7.41268 * max(0.0, 0.004855289 - Q.girth2_top15) * max(0.0, Q.n_dr_0p05_0p1 - 2.0)
            - 1.489265 * max(0.0, Q.psi_0p1 - 0.8747961)
            - 0.009699348 * max(0.0, Q.mass_top10 - 76.9886)
        ))
        + 0.21875 * grid(6, max(0.0, 0.3591835
            + 0.03309143 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.04070815 * max(0.0, 121.3913 - Q.mass)
            + 1131.49 * max(0.0, 0.0003372339 - Q.e3)
            + 0.03032547 * max(0.0, 82.85409 - Q.mass)
            + 6.305604e-05 * max(0.0, 121.3913 - Q.mass) * max(0.0, 1003.544 - Q.sum_pt_top50)
            + 36.44807 * max(0.0, Q.e2 - 0.04755309)
            + 2232.745 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.psi_0p3 - 0.9896594)
            + 0.006413691 * max(0.0, 172.4888 - Q.mass)
            + 0.03312753 * max(0.0, 89.74183 - Q.mass)
            - 0.07483936 * max(0.0, 101.0497 - Q.mass)
            + 8.79812 * max(0.0, 0.06310829 - Q.tau1)
            - 1.995063 * max(0.0, Q.sj3_dr_min - 0.1204829)
            + 0.05031243 * max(0.0, Q.n_dr_0p1_0p2 - 33.0)
            - 0.02226872 * max(0.0, Q.sj3_mass1 - 19.30204)
            - 2.192575e-05 * max(0.0, 89.74183 - Q.mass) * max(0.0, 1129.275 - Q.sum_pt_top20)
            + 81.49459 * max(0.0, 0.003687605 - Q.lam2)
            + 0.05797808 * max(0.0, 92.85979 - Q.mass)
            - 124.6715 * max(0.0, 0.001425993 - Q.girth2_top2)
            - 0.3986125 * max(0.0, 2.178951 - Q.D2)
            + 0.006799873 * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            - 4492.619 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.z_14 - 0.01634243)
            - 0.005595608 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) * max(0.0, 9.0 - Q.n_dr_0_0p05)
            - 0.1380028 * max(0.0, 0.07952881 - Q.eta_0) * max(0.0, Q.n_dr_0_0p05 - 5.0)
            + 153.1398 * max(0.0, 0.007259287 - Q.lam1)
            - 41.21214 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            + 203.3618 * max(0.0, 0.02580859 - Q.e2_sq)
            - 48.02988 * max(0.0, 0.01649354 - Q.lam1)
            - 75.8776 * max(0.0, 0.02550569 - Q.girth2_top50)
            + 124.4262 * max(0.0, 0.008376291 - Q.girth2_top30)
            - 84.07935 * max(0.0, 0.02412652 - Q.girth2_top30)
            + 6.167434 * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) * max(0.0, Q.soft10_dr0 - 0.1909669)
            + 80.74909 * max(0.0, 0.00287991 - Q.girth2_top20)
            - 1.919322 * max(0.0, Q.soft10_dr - 0.2075213)
            - 1.790278 * max(0.0, Q.e2 - 0.02793599) * max(0.0, 6.06065 - Q.ptdr0_13)
        ))
        + 0.015625 * grid(8, max(0.0, -0.8586652
            + 44.22524 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 74.2517 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.03063734 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.01859725 * max(0.0, Q.mass - 121.3913)
            + 0.004750203 * max(0.0, Q.mass - 89.74183)
            + 4.842668 * max(0.0, Q.log_sum_pt - 6.959294)
            - 0.00370839 * max(0.0, Q.sum_pt_top40 - 858.8262)
            - 0.02112566 * max(0.0, Q.mass_top20 - 90.4505)
            - 0.09964249 * max(0.0, Q.mass - 101.0497)
            + 0.6663901 * max(0.0, 1001.523 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477)
            + 149.2864 * max(0.0, Q.girth2_top40 - 0.008031986)
            + 0.03894753 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435)
            + 0.0376181 * max(0.0, Q.mass - 64.48544)
            + 401.3352 * max(0.0, 0.007887677 - Q.girth2_top15)
            - 0.003470294 * max(0.0, 1032.405 - Q.sum_pt_top40)
            - 0.05127087 * max(0.0, Q.mass_top50 - 80.35535)
            - 440.128 * max(0.0, Q.e2_sq - 0.009606007)
            + 3.355825 * max(0.0, Q.sj2_dr - 0.2232169)
            + 0.03997092 * max(0.0, Q.n_dr_0p2_0p4 - 8.0)
            + 80.63364 * max(0.0, Q.girth2_top15 - 0.001319197)
            + 31.16157 * max(0.0, Q.e2 - 0.04755309)
            + 0.007857585 * max(0.0, 1003.544 - Q.sum_pt_top50)
            - 185.7492 * max(0.0, 0.007887677 - Q.girth2_top15) * max(0.0, 0.6133424 - Q.tau21_b2)
            + 0.02347445 * max(0.0, Q.n_dr_0p1_0p2 - 19.0)
            - 0.7297459 * max(0.0, 1032.405 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477)
            - 10.45823 * max(0.0, Q.tau1 - 0.06310829)
            - 0.0002108822 * max(0.0, 1003.544 - Q.sum_pt_top50) * max(0.0, 21.0 - Q.n_dr_0p05_0p1)
            + 216.2895 * max(0.0, Q.e2_sq - 0.007872294)
            - 181.1447 * max(0.0, Q.girth2_top40 - 0.006026828)
            + 0.03090803 * max(0.0, Q.n_pt_above_1 - 54.0)
            - 5.7359 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 2.563115 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4)
            + 0.008906562 * max(0.0, Q.mass_top40 - 111.2487)
            + 5.433696 * max(0.0, Q.LHA - 0.3332345)
            + 0.04531039 * max(0.0, Q.mass - 80.78464)
            - 0.01161229 * max(0.0, Q.mass - 143.7876)
            + 0.04554217 * max(0.0, Q.mass_top50 - 97.93004)
            + 3178.132 * max(0.0, 0.0003372339 - Q.e3)
            - 38.94893 * max(0.0, 0.02737453 - Q.girth2_top20)
            + 7.375078 * max(0.0, 6.935549 - Q.log_sum_pt)
            + 3.189916 * max(0.0, Q.C2 - 0.07996447)
            - 34.11252 * max(0.0, 0.007856958 - Q.girth2_top30)
            + 95.89755 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266)
            - 90.39785 * max(0.0, 0.01563836 - Q.girth2_top15)
            - 110.2406 * max(0.0, 0.005383629 - Q.girth2_top15)
            - 0.00100911 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) * max(0.0, 26.76006 - Q.sj3_mass1)
        ))
        + 0.5625 * grid(9, max(0.0, -0.1731589
            + 0.005050553 * max(0.0, 80.89043 - Q.mass_top40)
            + 263.6798 * max(0.0, 0.006259772 - Q.girth2_top40)
            + 0.003046227 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258)
            - 0.0002116931 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt)
            + 0.0163929 * max(0.0, 121.737 - Q.mass_top30)
            - 0.06997643 * max(0.0, Q.mass - 143.7876)
            + 0.06260496 * max(0.0, Q.mass - 172.4888)
            + 35.81701 * max(0.0, Q.girth - 0.08589404)
            + 0.0428226 * max(0.0, 87.36377 - Q.mass)
            + 0.0008216668 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.n_particles - 38.0)
            + 0.0008920693 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 934.2416 - Q.sum_pt_top50)
            + 0.0009028777 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, Q.n_dr_0p2_0p4 - 10.0)
            - 0.01478441 * max(0.0, 152.6883 - Q.mass_top30)
            - 0.0009061 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, 20.0 - Q.n_pt_above_10)
            - 7.363469 * max(0.0, Q.z_top40_slots - 0.9574183)
            + 0.003105244 * max(0.0, 959.0957 - Q.sum_pt_top50)
            + 88.79276 * max(0.0, Q.lam1 - 0.01174405)
            - 0.01031488 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.894531)
            + 36.1164 * max(0.0, 0.01083435 - Q.girth2_top20)
            - 0.006491384 * max(0.0, 89.6788 - Q.mass_top40)
            - 186.3027 * max(0.0, Q.girth2_top30 - 0.02412652)
            - 0.06762542 * max(0.0, 64.48544 - Q.mass)
            + 0.04530099 * max(0.0, 79.65241 - Q.mass)
            - 38.68333 * max(0.0, 0.05444509 - Q.tau1)
            - 133.8495 * max(0.0, 0.004673423 - Q.lam1)
            + 16.74914 * max(0.0, 0.2091025 - Q.LHA)
            + 0.004939419 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft5_pt - 2.894531)
            + 1.280691 * max(0.0, 0.2187642 - Q.z_dr_0p1_0p2)
            + 17239.04 * max(0.0, Q.lam1 - 0.01174405) * max(0.0, Q.soft5_z - 0.002346473)
        ))
        - 0.015625 * grid(10, max(0.0, -3.948963
            - 8.881083 * max(0.0, 0.1207452 - Q.girth)
            - 0.04043477 * max(0.0, Q.mass - 162.8363)
            - 74.06046 * max(0.0, 0.005402331 - Q.girth2_top30)
            - 0.03294256 * max(0.0, Q.mass_top40 - 163.2541)
            - 0.02077042 * max(0.0, 80.3008 - Q.sj2_mass1)
            + 0.04475454 * max(0.0, 87.27603 - Q.mass_top40)
            + 0.232346 * max(0.0, 3.814159 - Q.D2)
            - 215.0982 * max(0.0, Q.psi_0p3 - 0.9985421)
            + 2.374723e-05 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, 464.75 - Q.sum_pt_top2)
            + 0.008188306 * max(0.0, Q.mass_top5 - 14.54404)
            - 2.467481 * max(0.0, 3.814159 - Q.D2) * max(0.0, Q.sj2_dr - 0.1937688)
            + 30.05285 * max(0.0, Q.mass_top40 - 163.2541) * max(0.0, Q.soft4_z - 0.001186042)
            + 0.01090531 * max(0.0, Q.mass_top50 - 157.5448)
            - 1.393145 * max(0.0, Q.z_dr_0_0p05 - 0.7674734)
            - 7.537004 * max(0.0, Q.log_sum_pt - 6.903423)
            + 0.08921913 * max(0.0, Q.mass - 82.85409)
            - 0.07994704 * max(0.0, Q.mass - 143.7876)
            - 0.05977461 * max(0.0, Q.mass - 64.48544)
            + 0.004303969 * max(0.0, Q.mass - 64.48544) * max(0.0, 2.537109 - Q.soft2_pt)
            + 0.003656236 * Q.sum_pt
            - 0.003736029 * max(0.0, 75.26407 - Q.mass_top15)
            + 19.25813 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.0830523 - Q.dr_4)
            - 310.9234 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            + 0.001829684 * max(0.0, 839.9547 - Q.sum_pt_top5)
            + 0.0008142922 * max(0.0, 111.2487 - Q.mass_top40) * max(0.0, Q.pt1_dr01 - 12.6865)
            - 22.70827 * max(0.0, 0.06524004 - Q.e2)
            + 92.61685 * max(0.0, 0.01807679 - Q.girth2_top30)
            + 6.914821 * max(0.0, 0.06413297 - Q.dr_0)
            - 0.04096867 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 4.206493 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            + 0.02819799 * max(0.0, 132.4278 - Q.mass_top40)
            - 0.05756879 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.02782281 * max(0.0, Q.mass_top40 - 67.72643)
            - 0.006647105 * max(0.0, 71.781 - Q.mass_top10)
            + 727.3351 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 4.029475 * max(0.0, 0.06940472 - Q.dr_1)
            - 40.51704 * max(0.0, 0.05444509 - Q.tau1)
            + 0.02880678 * max(0.0, 74.25181 - Q.mass)
            + 0.0002261132 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.91979)
            + 1.226455 * max(0.0, 0.6133424 - Q.tau21_b2)
        ))
        - 0.25 * grid(11, max(0.0, -0.2780376
            + 0.08474821 * max(0.0, 10.0 - Q.n_dr_0p2_0p4)
            - 0.002310951 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0)
            - 25.88529 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2)
            + 164.7726 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq)
            + 0.0007130026 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 42.31742 * max(0.0, Q.psi_0p2 - 0.9935324)
            - 0.05157728 * max(0.0, 79.65241 - Q.mass)
            + 0.0004235186 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.mass_top10 - 38.52415)
            + 0.06618185 * max(0.0, 92.85979 - Q.mass)
            - 0.008184396 * max(0.0, 79.65241 - Q.mass) * max(0.0, 3.814159 - Q.D2)
            + 7.345649 * max(0.0, 0.05048381 - Q.girth)
            - 34.05769 * max(0.0, 0.07374472 - Q.girth)
            + 156.1651 * max(0.0, 0.003213724 - Q.girth2_top10)
            - 233.8773 * max(0.0, 0.006716737 - Q.lam1)
            + 209.7776 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 0.05112769 - Q.C2_b2)
            + 0.07543136 * max(0.0, 6.0 - Q.n_dr_0p2_0p4)
            - 21.89062 * max(0.0, 0.02515919 - Q.e2)
            - 282.375 * max(0.0, 0.006929741 - Q.girth2_top30)
            + 232.8513 * max(0.0, 0.00818374 - Q.e2_sq)
            + 10.37528 * max(0.0, 0.2845608 - Q.LHA)
            + 0.01854368 * max(0.0, 15.0 - Q.n_dr_0p1_0p2)
            + 1.176877 * max(0.0, 0.1548383 - Q.z_dr_0p1_0p2)
            + 0.02129827 * max(0.0, 60.43821 - Q.mass_top30)
            - 0.04347799 * max(0.0, 85.8667 - Q.mass_top50)
            - 2.218355 * max(0.0, Q.z_top10_slots - 0.7271951)
            - 26.14764 * max(0.0, 0.04082832 - Q.e2)
            + 94.26836 * max(0.0, 0.02793599 - Q.e2)
        ))
        + 0.34375 * grid(12, max(0.0, -0.1994202
            + 0.01223357 * max(0.0, 87.36377 - Q.mass)
            - 0.09773045 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            - 0.03374758 * max(0.0, 94.64253 - Q.mass_top40)
            - 2.682579 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3)
            + 0.9075831 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367)
            - 11.1208 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.lam2 - 0.0003497174)
            + 0.05174195 * max(0.0, 82.85409 - Q.mass)
            - 373.0883 * max(0.0, 0.00363788 - Q.e2_sq)
            - 0.003510133 * max(0.0, 77.93668 - Q.mass_top40)
            - 0.04962413 * max(0.0, 53.87362 - Q.mass)
            + 3.315602e-05 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 435.25 - Q.pt_0)
            + 36.39789 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.3036026 - Q.planar_flow)
            + 0.001229078 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1)
            + 0.06322242 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            - 0.001456983 * max(0.0, Q.sum_pt - 1260.541)
            - 0.03622926 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.soft1_pt - 0.8144531)
            - 0.003879674 * max(0.0, 972.0419 - Q.sum_pt)
            + 4.797264 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4)
            - 5714.218 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.001776308 - Q.lam2)
            + 0.07497664 * max(0.0, 80.78464 - Q.mass)
            + 13.44873 * max(0.0, 77.93668 - Q.mass_top40) * max(0.0, Q.lam2 - 0.000404306)
            - 4.510786 * max(0.0, Q.z_dr_0_0p05 - 0.9084912)
        ))
    )


def logit_W(Q):
    return (0.09375
        + 0.75 * grid(0, max(0.0, 1.344566
            - 0.1228872 * max(0.0, Q.mass - 78.26182)
            - 0.1486546 * max(0.0, Q.mass - 92.85979)
            + 98.60338 * max(0.0, 0.005312783 - Q.girth2_top20)
            - 0.008056752 * max(0.0, 1012.673 - Q.sum_pt)
            + 68.57764 * max(0.0, Q.psi_0p3 - 0.9956185)
            - 0.04715151 * max(0.0, Q.mass - 91.03469)
            - 0.034417 * max(0.0, Q.mass - 74.25181)
            - 74.01811 * max(0.0, 0.007538019 - Q.girth2_top20)
            + 0.01592792 * max(0.0, 80.24626 - Q.mass_top30)
            - 0.1107082 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3)
            - 63.18221 * max(0.0, 0.005913555 - Q.lam1)
            + 348.6013 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            - 17.57021 * max(0.0, 0.0705748 - Q.tau1)
            - 448.8376 * max(0.0, 0.00616708 - Q.e2_sq)
            + 0.06134462 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957)
            + 0.0166648 * max(0.0, Q.mass_top50 - 82.04491)
            - 3.469101 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 17.27419 * max(0.0, 0.9906378 - Q.z_top50_slots)
            - 0.03045699 * max(0.0, 101.0497 - Q.mass)
            - 167.6191 * max(0.0, 0.006363916 - Q.girth2_top30)
            + 1.676685 * max(0.0, 7.017258 - Q.log_sum_pt)
            + 370.4587 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq)
            - 0.003132379 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.001705868 * max(0.0, 846.1934 - Q.sum_pt_top20)
            + 0.006808903 * max(0.0, 80.89043 - Q.mass_top40)
            - 59.72972 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.1153166 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.0223982 - Q.C3)
            + 0.01296883 * max(0.0, Q.mass_top50 - 71.79516)
            + 0.02415918 * max(0.0, Q.mass - 87.36377)
            + 0.03067002 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            + 0.002540558 * max(0.0, 1156.659 - Q.sum_pt_top50)
        ))
        + 0.4375 * grid(3, max(0.0, -0.2874307
            + 1001.893 * max(0.0, 0.0006154841 - Q.lam2)
            + 0.1648075 * max(0.0, 5.0 - Q.n_dr_0p2_0p4)
            + 0.02171065 * max(0.0, 46.0 - Q.n_particles)
            - 0.001259068 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349)
            - 0.01758449 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0)
            + 6.457543e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732)
            - 1.786492 * max(0.0, 0.3861957 - Q.tau21)
            + 160.963 * max(0.0, 0.006026828 - Q.girth2_top40)
            + 154.8658 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 12.43015 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5)
            + 0.6231591 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1)
            - 345.0482 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243)
            - 0.01103303 * max(0.0, 79.21004 - Q.mass_top50)
            + 0.005042994 * max(0.0, 87.36377 - Q.mass)
            - 129.7488 * max(0.0, 0.009614971 - Q.girth2)
            + 107.1983 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 0.03754923 * max(0.0, 8.0 - Q.n_dr_0p2_0p4)
            + 0.05043822 * max(0.0, 10.0 - Q.n_dr_0p1_0p2)
            - 0.01195483 * max(0.0, 53.87362 - Q.mass)
            - 178.3918 * max(0.0, 0.006259772 - Q.girth2_top40)
            - 15.76641 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.006794973 - Q.zdr_4)
            + 111.8487 * max(0.0, 0.008124776 - Q.girth2_top50)
        ))
        + 0.34375 * grid(4, max(0.0, 0.6914295
            + 0.01186823 * max(0.0, 69.02716 - Q.mass_top15)
            + 0.02282503 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            + 0.06144266 * max(0.0, 83.32554 - Q.mass_top40)
            - 1.018753 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.z_top50_slots)
            + 0.008792364 * max(0.0, 60.43821 - Q.mass_top30)
            + 0.02188353 * max(0.0, 121.3913 - Q.mass)
            + 111.2192 * max(0.0, Q.psi_0p3 - 0.9973959)
            - 0.07024051 * max(0.0, 87.36377 - Q.mass)
            - 0.009403079 * max(0.0, 121.737 - Q.mass_top30)
            + 0.008023934 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.01845784 * max(0.0, Q.n_particles - 22.0)
            - 226.0949 * max(0.0, 0.004855289 - Q.girth2_top15)
            + 0.02503371 * max(0.0, Q.n_dr_0_0p05 - 5.0)
            - 2.291058 * max(0.0, Q.sj2_dr - 0.2232169)
            - 0.01352989 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 6.916121 - Q.D2)
            + 0.009918761 * max(0.0, 101.0497 - Q.mass)
            - 0.0181479 * max(0.0, 67.72643 - Q.mass_top40)
            + 0.01852338 * max(0.0, 79.65241 - Q.mass)
            + 81.86659 * max(0.0, Q.e2_sq - 0.01396296)
            - 3.264511 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.zdr_0 - 0.006864207)
            - 22.59892 * max(0.0, Q.girth2_top15 - 0.007887677)
            - 22.2738 * max(0.0, 79.65241 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 0.06240618 * max(0.0, 74.25181 - Q.mass)
            - 0.003695312 * max(0.0, 1042.609 - Q.sum_pt)
            + 2.637887e-05 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, 1053.047 - Q.sum_pt_top40)
            - 0.04232617 * max(0.0, 94.64253 - Q.mass_top40)
            - 44.61525 * max(0.0, 0.007678544 - Q.girth2_top10)
            + 13.39343 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 152.651 * max(0.0, 0.001776308 - Q.lam2)
            + 0.1559495 * max(0.0, 5.378975 - Q.D2)
            - 0.1386088 * max(0.0, 1.0 - Q.sd_nremoved)
            + 7.41268 * max(0.0, 0.004855289 - Q.girth2_top15) * max(0.0, Q.n_dr_0p05_0p1 - 2.0)
            - 1.489265 * max(0.0, Q.psi_0p1 - 0.8747961)
            - 0.009699348 * max(0.0, Q.mass_top10 - 76.9886)
        ))
        + 0.578125 * grid(5, max(0.0, 1.365113
            + 0.03550476 * max(0.0, 64.0 - Q.n_particles)
            - 31.55748 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 6.215755 * max(0.0, Q.log_sum_pt - 6.935549)
            + 0.007034252 * max(0.0, Q.sum_pt - 907.9372)
            - 9.741529 * max(0.0, Q.log_sum_pt - 6.920349)
            + 194.3873 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 38.32114 * max(0.0, Q.girth2_top50 - 0.01951641)
            - 0.01375075 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.02108265 * max(0.0, Q.sum_pt_top50 - 934.2416)
            + 0.01997073 * max(0.0, Q.n_pt_above_1 - 28.0)
            - 0.0006994508 * max(0.0, 787.6281 - Q.sum_pt_top3)
            + 0.03204697 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 9831.014 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4)
            - 20.85056 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.005053338 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 5.579533 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.002719696 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 2.187253 * max(0.0, Q.max_dr - 0.2404747)
            - 8.345629 * max(0.0, Q.z_top30_slots - 0.9048492)
            + 0.01531813 * max(0.0, Q.mass - 172.4888)
            + 34.27206 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2)
            - 0.01391004 * max(0.0, 150.0144 - Q.mass_top40)
            - 0.05923529 * max(0.0, Q.mass_top50 - 157.5448)
            - 543.0219 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 0.03605449 * max(0.0, Q.sd_mass - 69.65633)
            - 1.101833 * max(0.0, 0.5100475 - Q.tau21)
            - 0.01352748 * max(0.0, Q.sum_pt - 986.0565)
            - 0.002087266 * max(0.0, Q.sum_pt_top10 - 943.6922)
            + 151.2781 * max(0.0, 0.01375115 - Q.z_11)
            - 5.676478 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.1958008 - Q.absphi_13)
            - 0.134532 * max(0.0, 14.14062 - Q.pt_11)
            + 21.40778 * max(0.0, 0.04082832 - Q.e2)
            - 6.487458 * max(0.0, 0.1507173 - Q.tau1)
            - 0.04655232 * max(0.0, Q.sd_mass - 83.30647)
            + 0.01805284 * max(0.0, 64.48544 - Q.mass)
            - 10.83178 * max(0.0, 0.03700182 - Q.z_dr_0p2_0p4)
            - 0.01411033 * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 0.01217882 * max(0.0, Q.mass_top10 - 71.781)
            - 0.6472427 * max(0.0, 0.707925 - Q.psi_0p1)
            + 0.01920067 * max(0.0, 23.45965 - Q.sj3_mass1)
            + 1.7528 * max(0.0, Q.max_dr - 0.4357228)
        ))
        + 0.0625 * grid(6, max(0.0, 0.3591835
            + 0.03309143 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.04070815 * max(0.0, 121.3913 - Q.mass)
            + 1131.49 * max(0.0, 0.0003372339 - Q.e3)
            + 0.03032547 * max(0.0, 82.85409 - Q.mass)
            + 6.305604e-05 * max(0.0, 121.3913 - Q.mass) * max(0.0, 1003.544 - Q.sum_pt_top50)
            + 36.44807 * max(0.0, Q.e2 - 0.04755309)
            + 2232.745 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.psi_0p3 - 0.9896594)
            + 0.006413691 * max(0.0, 172.4888 - Q.mass)
            + 0.03312753 * max(0.0, 89.74183 - Q.mass)
            - 0.07483936 * max(0.0, 101.0497 - Q.mass)
            + 8.79812 * max(0.0, 0.06310829 - Q.tau1)
            - 1.995063 * max(0.0, Q.sj3_dr_min - 0.1204829)
            + 0.05031243 * max(0.0, Q.n_dr_0p1_0p2 - 33.0)
            - 0.02226872 * max(0.0, Q.sj3_mass1 - 19.30204)
            - 2.192575e-05 * max(0.0, 89.74183 - Q.mass) * max(0.0, 1129.275 - Q.sum_pt_top20)
            + 81.49459 * max(0.0, 0.003687605 - Q.lam2)
            + 0.05797808 * max(0.0, 92.85979 - Q.mass)
            - 124.6715 * max(0.0, 0.001425993 - Q.girth2_top2)
            - 0.3986125 * max(0.0, 2.178951 - Q.D2)
            + 0.006799873 * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            - 4492.619 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.z_14 - 0.01634243)
            - 0.005595608 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) * max(0.0, 9.0 - Q.n_dr_0_0p05)
            - 0.1380028 * max(0.0, 0.07952881 - Q.eta_0) * max(0.0, Q.n_dr_0_0p05 - 5.0)
            + 153.1398 * max(0.0, 0.007259287 - Q.lam1)
            - 41.21214 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            + 203.3618 * max(0.0, 0.02580859 - Q.e2_sq)
            - 48.02988 * max(0.0, 0.01649354 - Q.lam1)
            - 75.8776 * max(0.0, 0.02550569 - Q.girth2_top50)
            + 124.4262 * max(0.0, 0.008376291 - Q.girth2_top30)
            - 84.07935 * max(0.0, 0.02412652 - Q.girth2_top30)
            + 6.167434 * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) * max(0.0, Q.soft10_dr0 - 0.1909669)
            + 80.74909 * max(0.0, 0.00287991 - Q.girth2_top20)
            - 1.919322 * max(0.0, Q.soft10_dr - 0.2075213)
            - 1.790278 * max(0.0, Q.e2 - 0.02793599) * max(0.0, 6.06065 - Q.ptdr0_13)
        ))
        - 0.625 * grid(7, max(0.0, -0.2224737
            + 3.491502 * max(0.0, 0.2352054 - Q.tau21_b2)
            + 393.9905 * max(0.0, 0.007877041 - Q.girth2)
            + 19.03582 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            - 108.6752 * max(0.0, 0.006403325 - Q.girth2)
            + 0.1839247 * max(0.0, 91.03469 - Q.mass)
            - 0.06715843 * max(0.0, 82.85409 - Q.mass)
            - 0.03758243 * max(0.0, 101.0497 - Q.mass)
            + 0.1010306 * max(0.0, 6.0 - Q.n_dr_0p2_0p4)
            - 402.5354 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 0.02552044 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9299135)
            - 1134.809 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3)
            + 3.596536 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 4.919033 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 0.2306774 * max(0.0, 92.85979 - Q.mass)
            - 0.008735284 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt)
            + 1.913659 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.9973959 - Q.psi_0p3)
            + 11.79884 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            + 2038.017 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 87.32425 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            + 21.61526 * max(0.0, 0.08786745 - Q.tau1)
            - 39.01793 * max(0.0, 0.1072713 - Q.tau1)
            - 8.915987 * max(0.0, 0.2601462 - Q.LHA)
            + 26.57642 * max(0.0, 0.3098384 - Q.LHA)
            + 24.33593 * max(0.0, 0.07708632 - Q.tau1)
            + 63.22398 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 1.860089 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.orientation_deg - -9.840088)
            - 0.8937333 * max(0.0, 0.3861957 - Q.tau21)
            - 12.52482 * max(0.0, 0.2845608 - Q.LHA)
            + 0.0241791 * max(0.0, 121.3913 - Q.mass)
            + 28639.36 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.zdr_3 - 0.00396157)
            - 6.033715 * max(0.0, 0.3332345 - Q.LHA)
            - 0.07342954 * max(0.0, 73.35236 - Q.mass_top20)
            + 0.001908467 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.sd_mass - 76.29481)
            - 0.002047878 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.sd_mass - 55.96662)
            + 0.07888491 * max(0.0, 70.42121 - Q.mass_top20)
            - 0.0463167 * max(0.0, 78.26182 - Q.mass)
            - 8.349812 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.pair_mass_0_3 - 6.624378)
            - 6.167508 * max(0.0, Q.z_top20_slots - 0.9102775)
            + 0.1147505 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 11.65938 - Q.sj3_mass3)
            - 2854.465 * max(0.0, 0.003418057 - Q.girth2_top50)
            - 258.9148 * max(0.0, 0.008241985 - Q.lam1)
            + 207.6828 * max(0.0, 0.006189818 - Q.lam1)
            + 138.5567 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq)
            - 480.3535 * max(0.0, 0.008190222 - Q.width)
        ))
        - 0.875 * grid(8, max(0.0, -0.8586652
            + 44.22524 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 74.2517 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.03063734 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.01859725 * max(0.0, Q.mass - 121.3913)
            + 0.004750203 * max(0.0, Q.mass - 89.74183)
            + 4.842668 * max(0.0, Q.log_sum_pt - 6.959294)
            - 0.00370839 * max(0.0, Q.sum_pt_top40 - 858.8262)
            - 0.02112566 * max(0.0, Q.mass_top20 - 90.4505)
            - 0.09964249 * max(0.0, Q.mass - 101.0497)
            + 0.6663901 * max(0.0, 1001.523 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477)
            + 149.2864 * max(0.0, Q.girth2_top40 - 0.008031986)
            + 0.03894753 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435)
            + 0.0376181 * max(0.0, Q.mass - 64.48544)
            + 401.3352 * max(0.0, 0.007887677 - Q.girth2_top15)
            - 0.003470294 * max(0.0, 1032.405 - Q.sum_pt_top40)
            - 0.05127087 * max(0.0, Q.mass_top50 - 80.35535)
            - 440.128 * max(0.0, Q.e2_sq - 0.009606007)
            + 3.355825 * max(0.0, Q.sj2_dr - 0.2232169)
            + 0.03997092 * max(0.0, Q.n_dr_0p2_0p4 - 8.0)
            + 80.63364 * max(0.0, Q.girth2_top15 - 0.001319197)
            + 31.16157 * max(0.0, Q.e2 - 0.04755309)
            + 0.007857585 * max(0.0, 1003.544 - Q.sum_pt_top50)
            - 185.7492 * max(0.0, 0.007887677 - Q.girth2_top15) * max(0.0, 0.6133424 - Q.tau21_b2)
            + 0.02347445 * max(0.0, Q.n_dr_0p1_0p2 - 19.0)
            - 0.7297459 * max(0.0, 1032.405 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477)
            - 10.45823 * max(0.0, Q.tau1 - 0.06310829)
            - 0.0002108822 * max(0.0, 1003.544 - Q.sum_pt_top50) * max(0.0, 21.0 - Q.n_dr_0p05_0p1)
            + 216.2895 * max(0.0, Q.e2_sq - 0.007872294)
            - 181.1447 * max(0.0, Q.girth2_top40 - 0.006026828)
            + 0.03090803 * max(0.0, Q.n_pt_above_1 - 54.0)
            - 5.7359 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 2.563115 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4)
            + 0.008906562 * max(0.0, Q.mass_top40 - 111.2487)
            + 5.433696 * max(0.0, Q.LHA - 0.3332345)
            + 0.04531039 * max(0.0, Q.mass - 80.78464)
            - 0.01161229 * max(0.0, Q.mass - 143.7876)
            + 0.04554217 * max(0.0, Q.mass_top50 - 97.93004)
            + 3178.132 * max(0.0, 0.0003372339 - Q.e3)
            - 38.94893 * max(0.0, 0.02737453 - Q.girth2_top20)
            + 7.375078 * max(0.0, 6.935549 - Q.log_sum_pt)
            + 3.189916 * max(0.0, Q.C2 - 0.07996447)
            - 34.11252 * max(0.0, 0.007856958 - Q.girth2_top30)
            + 95.89755 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266)
            - 90.39785 * max(0.0, 0.01563836 - Q.girth2_top15)
            - 110.2406 * max(0.0, 0.005383629 - Q.girth2_top15)
            - 0.00100911 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) * max(0.0, 26.76006 - Q.sj3_mass1)
        ))
        - 0.21875 * grid(9, max(0.0, -0.1731589
            + 0.005050553 * max(0.0, 80.89043 - Q.mass_top40)
            + 263.6798 * max(0.0, 0.006259772 - Q.girth2_top40)
            + 0.003046227 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258)
            - 0.0002116931 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt)
            + 0.0163929 * max(0.0, 121.737 - Q.mass_top30)
            - 0.06997643 * max(0.0, Q.mass - 143.7876)
            + 0.06260496 * max(0.0, Q.mass - 172.4888)
            + 35.81701 * max(0.0, Q.girth - 0.08589404)
            + 0.0428226 * max(0.0, 87.36377 - Q.mass)
            + 0.0008216668 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.n_particles - 38.0)
            + 0.0008920693 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 934.2416 - Q.sum_pt_top50)
            + 0.0009028777 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, Q.n_dr_0p2_0p4 - 10.0)
            - 0.01478441 * max(0.0, 152.6883 - Q.mass_top30)
            - 0.0009061 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, 20.0 - Q.n_pt_above_10)
            - 7.363469 * max(0.0, Q.z_top40_slots - 0.9574183)
            + 0.003105244 * max(0.0, 959.0957 - Q.sum_pt_top50)
            + 88.79276 * max(0.0, Q.lam1 - 0.01174405)
            - 0.01031488 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.894531)
            + 36.1164 * max(0.0, 0.01083435 - Q.girth2_top20)
            - 0.006491384 * max(0.0, 89.6788 - Q.mass_top40)
            - 186.3027 * max(0.0, Q.girth2_top30 - 0.02412652)
            - 0.06762542 * max(0.0, 64.48544 - Q.mass)
            + 0.04530099 * max(0.0, 79.65241 - Q.mass)
            - 38.68333 * max(0.0, 0.05444509 - Q.tau1)
            - 133.8495 * max(0.0, 0.004673423 - Q.lam1)
            + 16.74914 * max(0.0, 0.2091025 - Q.LHA)
            + 0.004939419 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft5_pt - 2.894531)
            + 1.280691 * max(0.0, 0.2187642 - Q.z_dr_0p1_0p2)
            + 17239.04 * max(0.0, Q.lam1 - 0.01174405) * max(0.0, Q.soft5_z - 0.002346473)
        ))
        + 0.59375 * grid(11, max(0.0, -0.2780376
            + 0.08474821 * max(0.0, 10.0 - Q.n_dr_0p2_0p4)
            - 0.002310951 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0)
            - 25.88529 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2)
            + 164.7726 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq)
            + 0.0007130026 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 42.31742 * max(0.0, Q.psi_0p2 - 0.9935324)
            - 0.05157728 * max(0.0, 79.65241 - Q.mass)
            + 0.0004235186 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.mass_top10 - 38.52415)
            + 0.06618185 * max(0.0, 92.85979 - Q.mass)
            - 0.008184396 * max(0.0, 79.65241 - Q.mass) * max(0.0, 3.814159 - Q.D2)
            + 7.345649 * max(0.0, 0.05048381 - Q.girth)
            - 34.05769 * max(0.0, 0.07374472 - Q.girth)
            + 156.1651 * max(0.0, 0.003213724 - Q.girth2_top10)
            - 233.8773 * max(0.0, 0.006716737 - Q.lam1)
            + 209.7776 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 0.05112769 - Q.C2_b2)
            + 0.07543136 * max(0.0, 6.0 - Q.n_dr_0p2_0p4)
            - 21.89062 * max(0.0, 0.02515919 - Q.e2)
            - 282.375 * max(0.0, 0.006929741 - Q.girth2_top30)
            + 232.8513 * max(0.0, 0.00818374 - Q.e2_sq)
            + 10.37528 * max(0.0, 0.2845608 - Q.LHA)
            + 0.01854368 * max(0.0, 15.0 - Q.n_dr_0p1_0p2)
            + 1.176877 * max(0.0, 0.1548383 - Q.z_dr_0p1_0p2)
            + 0.02129827 * max(0.0, 60.43821 - Q.mass_top30)
            - 0.04347799 * max(0.0, 85.8667 - Q.mass_top50)
            - 2.218355 * max(0.0, Q.z_top10_slots - 0.7271951)
            - 26.14764 * max(0.0, 0.04082832 - Q.e2)
            + 94.26836 * max(0.0, 0.02793599 - Q.e2)
        ))
        - 0.40625 * grid(12, max(0.0, -0.1994202
            + 0.01223357 * max(0.0, 87.36377 - Q.mass)
            - 0.09773045 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            - 0.03374758 * max(0.0, 94.64253 - Q.mass_top40)
            - 2.682579 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3)
            + 0.9075831 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367)
            - 11.1208 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.lam2 - 0.0003497174)
            + 0.05174195 * max(0.0, 82.85409 - Q.mass)
            - 373.0883 * max(0.0, 0.00363788 - Q.e2_sq)
            - 0.003510133 * max(0.0, 77.93668 - Q.mass_top40)
            - 0.04962413 * max(0.0, 53.87362 - Q.mass)
            + 3.315602e-05 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 435.25 - Q.pt_0)
            + 36.39789 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.3036026 - Q.planar_flow)
            + 0.001229078 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1)
            + 0.06322242 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            - 0.001456983 * max(0.0, Q.sum_pt - 1260.541)
            - 0.03622926 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.soft1_pt - 0.8144531)
            - 0.003879674 * max(0.0, 972.0419 - Q.sum_pt)
            + 4.797264 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4)
            - 5714.218 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.001776308 - Q.lam2)
            + 0.07497664 * max(0.0, 80.78464 - Q.mass)
            + 13.44873 * max(0.0, 77.93668 - Q.mass_top40) * max(0.0, Q.lam2 - 0.000404306)
            - 4.510786 * max(0.0, Q.z_dr_0_0p05 - 0.9084912)
        ))
        - 1.375 * grid(14, max(0.0, 0.1502622
            + 2.310509 * max(0.0, 0.342495 - Q.tau21_b2)
            - 46.10193 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.02048524 - Q.lam1)
            - 0.1230442 * max(0.0, 91.03469 - Q.mass)
            + 0.005011815 * max(0.0, 121.3913 - Q.mass)
            - 0.1005377 * max(0.0, 82.85409 - Q.mass)
            - 47.79012 * max(0.0, Q.psi_0p3 - 0.9924477)
            - 1.001672 * max(0.0, 0.4226723 - Q.N2)
            - 12.14911 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2738063)
            - 0.02274762 * max(0.0, 19.0 - Q.n_dr_0p1_0p2)
            - 0.001630815 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.mass_top2 - 36.76827)
            + 12.98777 * max(0.0, 0.03029714 - Q.e2)
            - 133.3873 * max(0.0, 0.09046749 - Q.mass_over_sum_pt)
            + 27.8296 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            + 61.38221 * max(0.0, 0.08873143 - Q.mass_over_sum_pt)
            - 131.0669 * max(0.0, 0.01083435 - Q.girth2_top20)
            + 154.3518 * max(0.0, 0.01986381 - Q.mass_over_sum_pt_sq)
            + 73.50571 * max(0.0, 0.006374178 - Q.girth2_top20)
            - 0.1309545 * max(0.0, 79.65241 - Q.mass)
            - 268.3684 * max(0.0, 0.001595561 - Q.soft7_z)
            + 36.82854 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            - 390.8474 * max(0.0, 0.001592178 - Q.girth2_top3)
            + 0.01461299 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.n_for_50pct - 5.0)
            - 37.72895 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.1231747 - Q.z_1)
            - 2.452624 * max(0.0, 0.01083435 - Q.girth2_top20) * max(0.0, Q.mass_top3 - 16.89912)
            - 0.03778889 * max(0.0, Q.n_pt_above_10 - 11.0)
            - 3185.725 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.0458688 - Q.dr_3)
            - 1.119562 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.zdr_3 - 0.0003281655)
            - 0.02119624 * max(0.0, 83.32554 - Q.mass_top40)
            - 133.251 * max(0.0, 0.01292642 - Q.girth2_top40)
            + 0.1098729 * max(0.0, 67.72643 - Q.mass_top40)
            + 32.13681 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.002405315 - Q.soft8_z)
            + 1.221472 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, Q.pt_3 - 60.59375)
            - 82.97294 * max(0.0, 0.01256572 - Q.e2)
            + 9.968657 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 4.250195 - Q.soft6_pt)
            + 32.48976 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.05799644 - Q.z_6)
            - 6.23804 * max(0.0, 0.04142826 - Q.tau2)
            + 124.9068 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 1.059188 - Q.N3)
            + 35.77221 * max(0.0, 0.03680582 - Q.e2)
            - 7.694597 * max(0.0, 0.1008497 - Q.tau1)
            + 14.44481 * max(0.0, 0.03948167 - Q.dr_3)
            - 3.15415 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.34119 - Q.sj2_zsoft)
            - 41.31546 * max(0.0, 0.01897915 - Q.girth2_top40)
        ))
        - 0.1875 * grid(15, max(0.0, -0.5934045
            - 0.4567752 * max(0.0, 0.8459004 - Q.z_dr_0_0p05)
            + 65.89934 * max(0.0, 0.00287991 - Q.girth2_top20)
            + 6.447457 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2)
            - 0.04437039 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 76.29481 - Q.sd_mass)
            + 115.5413 * max(0.0, 0.008329695 - Q.girth2_top5)
            + 0.004972314 * max(0.0, 986.0565 - Q.sum_pt)
            + 6.833269 * max(0.0, 7.017258 - Q.log_sum_pt)
            + 0.01022072 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt)
            - 54.35542 * max(0.0, Q.psi_0p3 - 0.9896594)
            - 0.01238817 * max(0.0, 1002.379 - Q.sum_pt)
            - 214.1313 * max(0.0, 0.002270363 - Q.girth2_top5)
            + 2.748398 * max(0.0, Q.sj2_dr - 0.2232169)
            + 159.5411 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2)
            + 34.36351 * max(0.0, 0.007678544 - Q.girth2_top10)
            + 0.0170685 * max(0.0, 13.0 - Q.n_dr_0p1_0p2)
            - 1.373813 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677)
            + 188.7687 * max(0.0, 0.001776308 - Q.lam2)
            - 19.75679 * max(0.0, 0.0705748 - Q.tau1)
            - 0.001161996 * max(0.0, 1017.778 - Q.sum_pt_top20)
            - 462.0352 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.03860363 * max(0.0, Q.n_dr_0p2_0p4 - 9.0)
            - 92.18994 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583)
            + 0.02092804 * max(0.0, 79.18312 - Q.sd_mass)
            - 0.02170075 * max(0.0, 40.97891 - Q.sd_mass)
            - 0.0105911 * max(0.0, 71.79516 - Q.mass_top50)
            + 69.89853 * max(0.0, 0.004673423 - Q.lam1)
            - 0.0006493807 * max(0.0, 1018.832 - Q.sum_pt_top30)
            + 0.0007463146 * max(0.0, 869.693 - Q.sum_pt_top20)
            - 5.583593 * max(0.0, 6.903423 - Q.log_sum_pt)
            - 6.229167 * max(0.0, Q.psi_0p1 - 0.9371031)
            + 0.01782458 * max(0.0, 52.15154 - Q.mass_top50)
        ))
    )


def logit_Z(Q):
    return (0.984375
        - 1.375 * grid(0, max(0.0, 1.344566
            - 0.1228872 * max(0.0, Q.mass - 78.26182)
            - 0.1486546 * max(0.0, Q.mass - 92.85979)
            + 98.60338 * max(0.0, 0.005312783 - Q.girth2_top20)
            - 0.008056752 * max(0.0, 1012.673 - Q.sum_pt)
            + 68.57764 * max(0.0, Q.psi_0p3 - 0.9956185)
            - 0.04715151 * max(0.0, Q.mass - 91.03469)
            - 0.034417 * max(0.0, Q.mass - 74.25181)
            - 74.01811 * max(0.0, 0.007538019 - Q.girth2_top20)
            + 0.01592792 * max(0.0, 80.24626 - Q.mass_top30)
            - 0.1107082 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3)
            - 63.18221 * max(0.0, 0.005913555 - Q.lam1)
            + 348.6013 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            - 17.57021 * max(0.0, 0.0705748 - Q.tau1)
            - 448.8376 * max(0.0, 0.00616708 - Q.e2_sq)
            + 0.06134462 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957)
            + 0.0166648 * max(0.0, Q.mass_top50 - 82.04491)
            - 3.469101 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 17.27419 * max(0.0, 0.9906378 - Q.z_top50_slots)
            - 0.03045699 * max(0.0, 101.0497 - Q.mass)
            - 167.6191 * max(0.0, 0.006363916 - Q.girth2_top30)
            + 1.676685 * max(0.0, 7.017258 - Q.log_sum_pt)
            + 370.4587 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq)
            - 0.003132379 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.001705868 * max(0.0, 846.1934 - Q.sum_pt_top20)
            + 0.006808903 * max(0.0, 80.89043 - Q.mass_top40)
            - 59.72972 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.1153166 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.0223982 - Q.C3)
            + 0.01296883 * max(0.0, Q.mass_top50 - 71.79516)
            + 0.02415918 * max(0.0, Q.mass - 87.36377)
            + 0.03067002 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            + 0.002540558 * max(0.0, 1156.659 - Q.sum_pt_top50)
        ))
        - 0.375 * grid(2, max(0.0, -0.002347209
            - 9.378196 * max(0.0, Q.log_sum_pt - 7.062574)
            + 0.01301346 * max(0.0, Q.sum_pt - 1017.435)
            - 0.008638248 * max(0.0, 51.0 - Q.n_particles)
            + 699.1433 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15)
            - 21.88492 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135)
            - 0.395954 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15)
            + 0.009690578 * max(0.0, Q.sum_pt_top50 - 1156.659)
            + 0.02754601 * max(0.0, 91.03469 - Q.mass)
            - 0.005048876 * max(0.0, 82.66587 - Q.mass_top30)
            + 0.001967088 * max(0.0, 1041.263 - Q.sum_pt_top40)
            - 0.001583572 * max(0.0, 996.8867 - Q.sum_pt_top30)
            - 0.004917892 * max(0.0, Q.sum_pt - 1115.723)
            + 126.6648 * max(0.0, 0.006088416 - Q.girth2_top30)
            - 0.002069385 * max(0.0, 1260.541 - Q.sum_pt)
            - 0.05361438 * max(0.0, 92.85979 - Q.mass)
            + 0.0152174 * max(0.0, 121.3913 - Q.mass)
            - 0.0005981218 * max(0.0, 787.6281 - Q.sum_pt_top3)
            - 0.01103883 * max(0.0, Q.sum_pt - 1052.889)
            - 105.2892 * max(0.0, 0.008124776 - Q.girth2_top50)
            + 0.002673855 * max(0.0, 1053.047 - Q.sum_pt_top40)
            + 0.00476503 * max(0.0, 143.7876 - Q.mass)
        ))
        + 0.4375 * grid(3, max(0.0, -0.2874307
            + 1001.893 * max(0.0, 0.0006154841 - Q.lam2)
            + 0.1648075 * max(0.0, 5.0 - Q.n_dr_0p2_0p4)
            + 0.02171065 * max(0.0, 46.0 - Q.n_particles)
            - 0.001259068 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349)
            - 0.01758449 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0)
            + 6.457543e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732)
            - 1.786492 * max(0.0, 0.3861957 - Q.tau21)
            + 160.963 * max(0.0, 0.006026828 - Q.girth2_top40)
            + 154.8658 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 12.43015 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5)
            + 0.6231591 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1)
            - 345.0482 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243)
            - 0.01103303 * max(0.0, 79.21004 - Q.mass_top50)
            + 0.005042994 * max(0.0, 87.36377 - Q.mass)
            - 129.7488 * max(0.0, 0.009614971 - Q.girth2)
            + 107.1983 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 0.03754923 * max(0.0, 8.0 - Q.n_dr_0p2_0p4)
            + 0.05043822 * max(0.0, 10.0 - Q.n_dr_0p1_0p2)
            - 0.01195483 * max(0.0, 53.87362 - Q.mass)
            - 178.3918 * max(0.0, 0.006259772 - Q.girth2_top40)
            - 15.76641 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.006794973 - Q.zdr_4)
            + 111.8487 * max(0.0, 0.008124776 - Q.girth2_top50)
        ))
        + 0.6875 * grid(5, max(0.0, 1.365113
            + 0.03550476 * max(0.0, 64.0 - Q.n_particles)
            - 31.55748 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 6.215755 * max(0.0, Q.log_sum_pt - 6.935549)
            + 0.007034252 * max(0.0, Q.sum_pt - 907.9372)
            - 9.741529 * max(0.0, Q.log_sum_pt - 6.920349)
            + 194.3873 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 38.32114 * max(0.0, Q.girth2_top50 - 0.01951641)
            - 0.01375075 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.02108265 * max(0.0, Q.sum_pt_top50 - 934.2416)
            + 0.01997073 * max(0.0, Q.n_pt_above_1 - 28.0)
            - 0.0006994508 * max(0.0, 787.6281 - Q.sum_pt_top3)
            + 0.03204697 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 9831.014 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4)
            - 20.85056 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.005053338 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 5.579533 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.002719696 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 2.187253 * max(0.0, Q.max_dr - 0.2404747)
            - 8.345629 * max(0.0, Q.z_top30_slots - 0.9048492)
            + 0.01531813 * max(0.0, Q.mass - 172.4888)
            + 34.27206 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2)
            - 0.01391004 * max(0.0, 150.0144 - Q.mass_top40)
            - 0.05923529 * max(0.0, Q.mass_top50 - 157.5448)
            - 543.0219 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 0.03605449 * max(0.0, Q.sd_mass - 69.65633)
            - 1.101833 * max(0.0, 0.5100475 - Q.tau21)
            - 0.01352748 * max(0.0, Q.sum_pt - 986.0565)
            - 0.002087266 * max(0.0, Q.sum_pt_top10 - 943.6922)
            + 151.2781 * max(0.0, 0.01375115 - Q.z_11)
            - 5.676478 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.1958008 - Q.absphi_13)
            - 0.134532 * max(0.0, 14.14062 - Q.pt_11)
            + 21.40778 * max(0.0, 0.04082832 - Q.e2)
            - 6.487458 * max(0.0, 0.1507173 - Q.tau1)
            - 0.04655232 * max(0.0, Q.sd_mass - 83.30647)
            + 0.01805284 * max(0.0, 64.48544 - Q.mass)
            - 10.83178 * max(0.0, 0.03700182 - Q.z_dr_0p2_0p4)
            - 0.01411033 * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 0.01217882 * max(0.0, Q.mass_top10 - 71.781)
            - 0.6472427 * max(0.0, 0.707925 - Q.psi_0p1)
            + 0.01920067 * max(0.0, 23.45965 - Q.sj3_mass1)
            + 1.7528 * max(0.0, Q.max_dr - 0.4357228)
        ))
        - 0.96875 * grid(6, max(0.0, 0.3591835
            + 0.03309143 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.04070815 * max(0.0, 121.3913 - Q.mass)
            + 1131.49 * max(0.0, 0.0003372339 - Q.e3)
            + 0.03032547 * max(0.0, 82.85409 - Q.mass)
            + 6.305604e-05 * max(0.0, 121.3913 - Q.mass) * max(0.0, 1003.544 - Q.sum_pt_top50)
            + 36.44807 * max(0.0, Q.e2 - 0.04755309)
            + 2232.745 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.psi_0p3 - 0.9896594)
            + 0.006413691 * max(0.0, 172.4888 - Q.mass)
            + 0.03312753 * max(0.0, 89.74183 - Q.mass)
            - 0.07483936 * max(0.0, 101.0497 - Q.mass)
            + 8.79812 * max(0.0, 0.06310829 - Q.tau1)
            - 1.995063 * max(0.0, Q.sj3_dr_min - 0.1204829)
            + 0.05031243 * max(0.0, Q.n_dr_0p1_0p2 - 33.0)
            - 0.02226872 * max(0.0, Q.sj3_mass1 - 19.30204)
            - 2.192575e-05 * max(0.0, 89.74183 - Q.mass) * max(0.0, 1129.275 - Q.sum_pt_top20)
            + 81.49459 * max(0.0, 0.003687605 - Q.lam2)
            + 0.05797808 * max(0.0, 92.85979 - Q.mass)
            - 124.6715 * max(0.0, 0.001425993 - Q.girth2_top2)
            - 0.3986125 * max(0.0, 2.178951 - Q.D2)
            + 0.006799873 * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            - 4492.619 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.z_14 - 0.01634243)
            - 0.005595608 * max(0.0, Q.n_dr_0p1_0p2 - 33.0) * max(0.0, 9.0 - Q.n_dr_0_0p05)
            - 0.1380028 * max(0.0, 0.07952881 - Q.eta_0) * max(0.0, Q.n_dr_0_0p05 - 5.0)
            + 153.1398 * max(0.0, 0.007259287 - Q.lam1)
            - 41.21214 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            + 203.3618 * max(0.0, 0.02580859 - Q.e2_sq)
            - 48.02988 * max(0.0, 0.01649354 - Q.lam1)
            - 75.8776 * max(0.0, 0.02550569 - Q.girth2_top50)
            + 124.4262 * max(0.0, 0.008376291 - Q.girth2_top30)
            - 84.07935 * max(0.0, 0.02412652 - Q.girth2_top30)
            + 6.167434 * max(0.0, 0.2864926 - Q.z_dr_0p1_0p2) * max(0.0, Q.soft10_dr0 - 0.1909669)
            + 80.74909 * max(0.0, 0.00287991 - Q.girth2_top20)
            - 1.919322 * max(0.0, Q.soft10_dr - 0.2075213)
            - 1.790278 * max(0.0, Q.e2 - 0.02793599) * max(0.0, 6.06065 - Q.ptdr0_13)
        ))
        + 0.90625 * grid(7, max(0.0, -0.2224737
            + 3.491502 * max(0.0, 0.2352054 - Q.tau21_b2)
            + 393.9905 * max(0.0, 0.007877041 - Q.girth2)
            + 19.03582 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            - 108.6752 * max(0.0, 0.006403325 - Q.girth2)
            + 0.1839247 * max(0.0, 91.03469 - Q.mass)
            - 0.06715843 * max(0.0, 82.85409 - Q.mass)
            - 0.03758243 * max(0.0, 101.0497 - Q.mass)
            + 0.1010306 * max(0.0, 6.0 - Q.n_dr_0p2_0p4)
            - 402.5354 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 0.02552044 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9299135)
            - 1134.809 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3)
            + 3.596536 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 4.919033 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 0.2306774 * max(0.0, 92.85979 - Q.mass)
            - 0.008735284 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt)
            + 1.913659 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.9973959 - Q.psi_0p3)
            + 11.79884 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            + 2038.017 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 87.32425 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            + 21.61526 * max(0.0, 0.08786745 - Q.tau1)
            - 39.01793 * max(0.0, 0.1072713 - Q.tau1)
            - 8.915987 * max(0.0, 0.2601462 - Q.LHA)
            + 26.57642 * max(0.0, 0.3098384 - Q.LHA)
            + 24.33593 * max(0.0, 0.07708632 - Q.tau1)
            + 63.22398 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 1.860089 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.orientation_deg - -9.840088)
            - 0.8937333 * max(0.0, 0.3861957 - Q.tau21)
            - 12.52482 * max(0.0, 0.2845608 - Q.LHA)
            + 0.0241791 * max(0.0, 121.3913 - Q.mass)
            + 28639.36 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.zdr_3 - 0.00396157)
            - 6.033715 * max(0.0, 0.3332345 - Q.LHA)
            - 0.07342954 * max(0.0, 73.35236 - Q.mass_top20)
            + 0.001908467 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.sd_mass - 76.29481)
            - 0.002047878 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.sd_mass - 55.96662)
            + 0.07888491 * max(0.0, 70.42121 - Q.mass_top20)
            - 0.0463167 * max(0.0, 78.26182 - Q.mass)
            - 8.349812 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.pair_mass_0_3 - 6.624378)
            - 6.167508 * max(0.0, Q.z_top20_slots - 0.9102775)
            + 0.1147505 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 11.65938 - Q.sj3_mass3)
            - 2854.465 * max(0.0, 0.003418057 - Q.girth2_top50)
            - 258.9148 * max(0.0, 0.008241985 - Q.lam1)
            + 207.6828 * max(0.0, 0.006189818 - Q.lam1)
            + 138.5567 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq)
            - 480.3535 * max(0.0, 0.008190222 - Q.width)
        ))
        - 0.9375 * grid(8, max(0.0, -0.8586652
            + 44.22524 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 74.2517 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.03063734 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.01859725 * max(0.0, Q.mass - 121.3913)
            + 0.004750203 * max(0.0, Q.mass - 89.74183)
            + 4.842668 * max(0.0, Q.log_sum_pt - 6.959294)
            - 0.00370839 * max(0.0, Q.sum_pt_top40 - 858.8262)
            - 0.02112566 * max(0.0, Q.mass_top20 - 90.4505)
            - 0.09964249 * max(0.0, Q.mass - 101.0497)
            + 0.6663901 * max(0.0, 1001.523 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477)
            + 149.2864 * max(0.0, Q.girth2_top40 - 0.008031986)
            + 0.03894753 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435)
            + 0.0376181 * max(0.0, Q.mass - 64.48544)
            + 401.3352 * max(0.0, 0.007887677 - Q.girth2_top15)
            - 0.003470294 * max(0.0, 1032.405 - Q.sum_pt_top40)
            - 0.05127087 * max(0.0, Q.mass_top50 - 80.35535)
            - 440.128 * max(0.0, Q.e2_sq - 0.009606007)
            + 3.355825 * max(0.0, Q.sj2_dr - 0.2232169)
            + 0.03997092 * max(0.0, Q.n_dr_0p2_0p4 - 8.0)
            + 80.63364 * max(0.0, Q.girth2_top15 - 0.001319197)
            + 31.16157 * max(0.0, Q.e2 - 0.04755309)
            + 0.007857585 * max(0.0, 1003.544 - Q.sum_pt_top50)
            - 185.7492 * max(0.0, 0.007887677 - Q.girth2_top15) * max(0.0, 0.6133424 - Q.tau21_b2)
            + 0.02347445 * max(0.0, Q.n_dr_0p1_0p2 - 19.0)
            - 0.7297459 * max(0.0, 1032.405 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477)
            - 10.45823 * max(0.0, Q.tau1 - 0.06310829)
            - 0.0002108822 * max(0.0, 1003.544 - Q.sum_pt_top50) * max(0.0, 21.0 - Q.n_dr_0p05_0p1)
            + 216.2895 * max(0.0, Q.e2_sq - 0.007872294)
            - 181.1447 * max(0.0, Q.girth2_top40 - 0.006026828)
            + 0.03090803 * max(0.0, Q.n_pt_above_1 - 54.0)
            - 5.7359 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 2.563115 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4)
            + 0.008906562 * max(0.0, Q.mass_top40 - 111.2487)
            + 5.433696 * max(0.0, Q.LHA - 0.3332345)
            + 0.04531039 * max(0.0, Q.mass - 80.78464)
            - 0.01161229 * max(0.0, Q.mass - 143.7876)
            + 0.04554217 * max(0.0, Q.mass_top50 - 97.93004)
            + 3178.132 * max(0.0, 0.0003372339 - Q.e3)
            - 38.94893 * max(0.0, 0.02737453 - Q.girth2_top20)
            + 7.375078 * max(0.0, 6.935549 - Q.log_sum_pt)
            + 3.189916 * max(0.0, Q.C2 - 0.07996447)
            - 34.11252 * max(0.0, 0.007856958 - Q.girth2_top30)
            + 95.89755 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266)
            - 90.39785 * max(0.0, 0.01563836 - Q.girth2_top15)
            - 110.2406 * max(0.0, 0.005383629 - Q.girth2_top15)
            - 0.00100911 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) * max(0.0, 26.76006 - Q.sj3_mass1)
        ))
        + 0.3125 * grid(12, max(0.0, -0.1994202
            + 0.01223357 * max(0.0, 87.36377 - Q.mass)
            - 0.09773045 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            - 0.03374758 * max(0.0, 94.64253 - Q.mass_top40)
            - 2.682579 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3)
            + 0.9075831 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367)
            - 11.1208 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.lam2 - 0.0003497174)
            + 0.05174195 * max(0.0, 82.85409 - Q.mass)
            - 373.0883 * max(0.0, 0.00363788 - Q.e2_sq)
            - 0.003510133 * max(0.0, 77.93668 - Q.mass_top40)
            - 0.04962413 * max(0.0, 53.87362 - Q.mass)
            + 3.315602e-05 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 435.25 - Q.pt_0)
            + 36.39789 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.3036026 - Q.planar_flow)
            + 0.001229078 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1)
            + 0.06322242 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            - 0.001456983 * max(0.0, Q.sum_pt - 1260.541)
            - 0.03622926 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.soft1_pt - 0.8144531)
            - 0.003879674 * max(0.0, 972.0419 - Q.sum_pt)
            + 4.797264 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4)
            - 5714.218 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.001776308 - Q.lam2)
            + 0.07497664 * max(0.0, 80.78464 - Q.mass)
            + 13.44873 * max(0.0, 77.93668 - Q.mass_top40) * max(0.0, Q.lam2 - 0.000404306)
            - 4.510786 * max(0.0, Q.z_dr_0_0p05 - 0.9084912)
        ))
        + 0.5625 * grid(15, max(0.0, -0.5934045
            - 0.4567752 * max(0.0, 0.8459004 - Q.z_dr_0_0p05)
            + 65.89934 * max(0.0, 0.00287991 - Q.girth2_top20)
            + 6.447457 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2)
            - 0.04437039 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 76.29481 - Q.sd_mass)
            + 115.5413 * max(0.0, 0.008329695 - Q.girth2_top5)
            + 0.004972314 * max(0.0, 986.0565 - Q.sum_pt)
            + 6.833269 * max(0.0, 7.017258 - Q.log_sum_pt)
            + 0.01022072 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt)
            - 54.35542 * max(0.0, Q.psi_0p3 - 0.9896594)
            - 0.01238817 * max(0.0, 1002.379 - Q.sum_pt)
            - 214.1313 * max(0.0, 0.002270363 - Q.girth2_top5)
            + 2.748398 * max(0.0, Q.sj2_dr - 0.2232169)
            + 159.5411 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2)
            + 34.36351 * max(0.0, 0.007678544 - Q.girth2_top10)
            + 0.0170685 * max(0.0, 13.0 - Q.n_dr_0p1_0p2)
            - 1.373813 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677)
            + 188.7687 * max(0.0, 0.001776308 - Q.lam2)
            - 19.75679 * max(0.0, 0.0705748 - Q.tau1)
            - 0.001161996 * max(0.0, 1017.778 - Q.sum_pt_top20)
            - 462.0352 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.03860363 * max(0.0, Q.n_dr_0p2_0p4 - 9.0)
            - 92.18994 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583)
            + 0.02092804 * max(0.0, 79.18312 - Q.sd_mass)
            - 0.02170075 * max(0.0, 40.97891 - Q.sd_mass)
            - 0.0105911 * max(0.0, 71.79516 - Q.mass_top50)
            + 69.89853 * max(0.0, 0.004673423 - Q.lam1)
            - 0.0006493807 * max(0.0, 1018.832 - Q.sum_pt_top30)
            + 0.0007463146 * max(0.0, 869.693 - Q.sum_pt_top20)
            - 5.583593 * max(0.0, 6.903423 - Q.log_sum_pt)
            - 6.229167 * max(0.0, Q.psi_0p1 - 0.9371031)
            + 0.01782458 * max(0.0, 52.15154 - Q.mass_top50)
        ))
    )


def logit_t(Q):
    return (0.78125
        + 0.125 * grid(0, max(0.0, 1.344566
            - 0.1228872 * max(0.0, Q.mass - 78.26182)
            - 0.1486546 * max(0.0, Q.mass - 92.85979)
            + 98.60338 * max(0.0, 0.005312783 - Q.girth2_top20)
            - 0.008056752 * max(0.0, 1012.673 - Q.sum_pt)
            + 68.57764 * max(0.0, Q.psi_0p3 - 0.9956185)
            - 0.04715151 * max(0.0, Q.mass - 91.03469)
            - 0.034417 * max(0.0, Q.mass - 74.25181)
            - 74.01811 * max(0.0, 0.007538019 - Q.girth2_top20)
            + 0.01592792 * max(0.0, 80.24626 - Q.mass_top30)
            - 0.1107082 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3)
            - 63.18221 * max(0.0, 0.005913555 - Q.lam1)
            + 348.6013 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            - 17.57021 * max(0.0, 0.0705748 - Q.tau1)
            - 448.8376 * max(0.0, 0.00616708 - Q.e2_sq)
            + 0.06134462 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957)
            + 0.0166648 * max(0.0, Q.mass_top50 - 82.04491)
            - 3.469101 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 17.27419 * max(0.0, 0.9906378 - Q.z_top50_slots)
            - 0.03045699 * max(0.0, 101.0497 - Q.mass)
            - 167.6191 * max(0.0, 0.006363916 - Q.girth2_top30)
            + 1.676685 * max(0.0, 7.017258 - Q.log_sum_pt)
            + 370.4587 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq)
            - 0.003132379 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.001705868 * max(0.0, 846.1934 - Q.sum_pt_top20)
            + 0.006808903 * max(0.0, 80.89043 - Q.mass_top40)
            - 59.72972 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.1153166 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.0223982 - Q.C3)
            + 0.01296883 * max(0.0, Q.mass_top50 - 71.79516)
            + 0.02415918 * max(0.0, Q.mass - 87.36377)
            + 0.03067002 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            + 0.002540558 * max(0.0, 1156.659 - Q.sum_pt_top50)
        ))
        + 0.1875 * grid(4, max(0.0, 0.6914295
            + 0.01186823 * max(0.0, 69.02716 - Q.mass_top15)
            + 0.02282503 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            + 0.06144266 * max(0.0, 83.32554 - Q.mass_top40)
            - 1.018753 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.z_top50_slots)
            + 0.008792364 * max(0.0, 60.43821 - Q.mass_top30)
            + 0.02188353 * max(0.0, 121.3913 - Q.mass)
            + 111.2192 * max(0.0, Q.psi_0p3 - 0.9973959)
            - 0.07024051 * max(0.0, 87.36377 - Q.mass)
            - 0.009403079 * max(0.0, 121.737 - Q.mass_top30)
            + 0.008023934 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.01845784 * max(0.0, Q.n_particles - 22.0)
            - 226.0949 * max(0.0, 0.004855289 - Q.girth2_top15)
            + 0.02503371 * max(0.0, Q.n_dr_0_0p05 - 5.0)
            - 2.291058 * max(0.0, Q.sj2_dr - 0.2232169)
            - 0.01352989 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 6.916121 - Q.D2)
            + 0.009918761 * max(0.0, 101.0497 - Q.mass)
            - 0.0181479 * max(0.0, 67.72643 - Q.mass_top40)
            + 0.01852338 * max(0.0, 79.65241 - Q.mass)
            + 81.86659 * max(0.0, Q.e2_sq - 0.01396296)
            - 3.264511 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.zdr_0 - 0.006864207)
            - 22.59892 * max(0.0, Q.girth2_top15 - 0.007887677)
            - 22.2738 * max(0.0, 79.65241 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 0.06240618 * max(0.0, 74.25181 - Q.mass)
            - 0.003695312 * max(0.0, 1042.609 - Q.sum_pt)
            + 2.637887e-05 * max(0.0, 121.737 - Q.mass_top30) * max(0.0, 1053.047 - Q.sum_pt_top40)
            - 0.04232617 * max(0.0, 94.64253 - Q.mass_top40)
            - 44.61525 * max(0.0, 0.007678544 - Q.girth2_top10)
            + 13.39343 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 152.651 * max(0.0, 0.001776308 - Q.lam2)
            + 0.1559495 * max(0.0, 5.378975 - Q.D2)
            - 0.1386088 * max(0.0, 1.0 - Q.sd_nremoved)
            + 7.41268 * max(0.0, 0.004855289 - Q.girth2_top15) * max(0.0, Q.n_dr_0p05_0p1 - 2.0)
            - 1.489265 * max(0.0, Q.psi_0p1 - 0.8747961)
            - 0.009699348 * max(0.0, Q.mass_top10 - 76.9886)
        ))
        - 0.46875 * grid(5, max(0.0, 1.365113
            + 0.03550476 * max(0.0, 64.0 - Q.n_particles)
            - 31.55748 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 6.215755 * max(0.0, Q.log_sum_pt - 6.935549)
            + 0.007034252 * max(0.0, Q.sum_pt - 907.9372)
            - 9.741529 * max(0.0, Q.log_sum_pt - 6.920349)
            + 194.3873 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 38.32114 * max(0.0, Q.girth2_top50 - 0.01951641)
            - 0.01375075 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.02108265 * max(0.0, Q.sum_pt_top50 - 934.2416)
            + 0.01997073 * max(0.0, Q.n_pt_above_1 - 28.0)
            - 0.0006994508 * max(0.0, 787.6281 - Q.sum_pt_top3)
            + 0.03204697 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 9831.014 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4)
            - 20.85056 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.005053338 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 5.579533 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.002719696 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 2.187253 * max(0.0, Q.max_dr - 0.2404747)
            - 8.345629 * max(0.0, Q.z_top30_slots - 0.9048492)
            + 0.01531813 * max(0.0, Q.mass - 172.4888)
            + 34.27206 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2)
            - 0.01391004 * max(0.0, 150.0144 - Q.mass_top40)
            - 0.05923529 * max(0.0, Q.mass_top50 - 157.5448)
            - 543.0219 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 0.03605449 * max(0.0, Q.sd_mass - 69.65633)
            - 1.101833 * max(0.0, 0.5100475 - Q.tau21)
            - 0.01352748 * max(0.0, Q.sum_pt - 986.0565)
            - 0.002087266 * max(0.0, Q.sum_pt_top10 - 943.6922)
            + 151.2781 * max(0.0, 0.01375115 - Q.z_11)
            - 5.676478 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.1958008 - Q.absphi_13)
            - 0.134532 * max(0.0, 14.14062 - Q.pt_11)
            + 21.40778 * max(0.0, 0.04082832 - Q.e2)
            - 6.487458 * max(0.0, 0.1507173 - Q.tau1)
            - 0.04655232 * max(0.0, Q.sd_mass - 83.30647)
            + 0.01805284 * max(0.0, 64.48544 - Q.mass)
            - 10.83178 * max(0.0, 0.03700182 - Q.z_dr_0p2_0p4)
            - 0.01411033 * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 0.01217882 * max(0.0, Q.mass_top10 - 71.781)
            - 0.6472427 * max(0.0, 0.707925 - Q.psi_0p1)
            + 0.01920067 * max(0.0, 23.45965 - Q.sj3_mass1)
            + 1.7528 * max(0.0, Q.max_dr - 0.4357228)
        ))
        - 0.28125 * grid(7, max(0.0, -0.2224737
            + 3.491502 * max(0.0, 0.2352054 - Q.tau21_b2)
            + 393.9905 * max(0.0, 0.007877041 - Q.girth2)
            + 19.03582 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            - 108.6752 * max(0.0, 0.006403325 - Q.girth2)
            + 0.1839247 * max(0.0, 91.03469 - Q.mass)
            - 0.06715843 * max(0.0, 82.85409 - Q.mass)
            - 0.03758243 * max(0.0, 101.0497 - Q.mass)
            + 0.1010306 * max(0.0, 6.0 - Q.n_dr_0p2_0p4)
            - 402.5354 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 0.02552044 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9299135)
            - 1134.809 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, 7.876005e-05 - Q.e3)
            + 3.596536 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 4.919033 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 0.2306774 * max(0.0, 92.85979 - Q.mass)
            - 0.008735284 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt)
            + 1.913659 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.9973959 - Q.psi_0p3)
            + 11.79884 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            + 2038.017 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 87.32425 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            + 21.61526 * max(0.0, 0.08786745 - Q.tau1)
            - 39.01793 * max(0.0, 0.1072713 - Q.tau1)
            - 8.915987 * max(0.0, 0.2601462 - Q.LHA)
            + 26.57642 * max(0.0, 0.3098384 - Q.LHA)
            + 24.33593 * max(0.0, 0.07708632 - Q.tau1)
            + 63.22398 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959)
            - 1.860089 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.orientation_deg - -9.840088)
            - 0.8937333 * max(0.0, 0.3861957 - Q.tau21)
            - 12.52482 * max(0.0, 0.2845608 - Q.LHA)
            + 0.0241791 * max(0.0, 121.3913 - Q.mass)
            + 28639.36 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.zdr_3 - 0.00396157)
            - 6.033715 * max(0.0, 0.3332345 - Q.LHA)
            - 0.07342954 * max(0.0, 73.35236 - Q.mass_top20)
            + 0.001908467 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.sd_mass - 76.29481)
            - 0.002047878 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.sd_mass - 55.96662)
            + 0.07888491 * max(0.0, 70.42121 - Q.mass_top20)
            - 0.0463167 * max(0.0, 78.26182 - Q.mass)
            - 8.349812 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.pair_mass_0_3 - 6.624378)
            - 6.167508 * max(0.0, Q.z_top20_slots - 0.9102775)
            + 0.1147505 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 11.65938 - Q.sj3_mass3)
            - 2854.465 * max(0.0, 0.003418057 - Q.girth2_top50)
            - 258.9148 * max(0.0, 0.008241985 - Q.lam1)
            + 207.6828 * max(0.0, 0.006189818 - Q.lam1)
            + 138.5567 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq)
            - 480.3535 * max(0.0, 0.008190222 - Q.width)
        ))
        + 0.2109375 * grid(8, max(0.0, -0.8586652
            + 44.22524 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 74.2517 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.03063734 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.01859725 * max(0.0, Q.mass - 121.3913)
            + 0.004750203 * max(0.0, Q.mass - 89.74183)
            + 4.842668 * max(0.0, Q.log_sum_pt - 6.959294)
            - 0.00370839 * max(0.0, Q.sum_pt_top40 - 858.8262)
            - 0.02112566 * max(0.0, Q.mass_top20 - 90.4505)
            - 0.09964249 * max(0.0, Q.mass - 101.0497)
            + 0.6663901 * max(0.0, 1001.523 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477)
            + 149.2864 * max(0.0, Q.girth2_top40 - 0.008031986)
            + 0.03894753 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435)
            + 0.0376181 * max(0.0, Q.mass - 64.48544)
            + 401.3352 * max(0.0, 0.007887677 - Q.girth2_top15)
            - 0.003470294 * max(0.0, 1032.405 - Q.sum_pt_top40)
            - 0.05127087 * max(0.0, Q.mass_top50 - 80.35535)
            - 440.128 * max(0.0, Q.e2_sq - 0.009606007)
            + 3.355825 * max(0.0, Q.sj2_dr - 0.2232169)
            + 0.03997092 * max(0.0, Q.n_dr_0p2_0p4 - 8.0)
            + 80.63364 * max(0.0, Q.girth2_top15 - 0.001319197)
            + 31.16157 * max(0.0, Q.e2 - 0.04755309)
            + 0.007857585 * max(0.0, 1003.544 - Q.sum_pt_top50)
            - 185.7492 * max(0.0, 0.007887677 - Q.girth2_top15) * max(0.0, 0.6133424 - Q.tau21_b2)
            + 0.02347445 * max(0.0, Q.n_dr_0p1_0p2 - 19.0)
            - 0.7297459 * max(0.0, 1032.405 - Q.sum_pt_top40) * max(0.0, Q.psi_0p3 - 0.9924477)
            - 10.45823 * max(0.0, Q.tau1 - 0.06310829)
            - 0.0002108822 * max(0.0, 1003.544 - Q.sum_pt_top50) * max(0.0, 21.0 - Q.n_dr_0p05_0p1)
            + 216.2895 * max(0.0, Q.e2_sq - 0.007872294)
            - 181.1447 * max(0.0, Q.girth2_top40 - 0.006026828)
            + 0.03090803 * max(0.0, Q.n_pt_above_1 - 54.0)
            - 5.7359 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 2.563115 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4)
            + 0.008906562 * max(0.0, Q.mass_top40 - 111.2487)
            + 5.433696 * max(0.0, Q.LHA - 0.3332345)
            + 0.04531039 * max(0.0, Q.mass - 80.78464)
            - 0.01161229 * max(0.0, Q.mass - 143.7876)
            + 0.04554217 * max(0.0, Q.mass_top50 - 97.93004)
            + 3178.132 * max(0.0, 0.0003372339 - Q.e3)
            - 38.94893 * max(0.0, 0.02737453 - Q.girth2_top20)
            + 7.375078 * max(0.0, 6.935549 - Q.log_sum_pt)
            + 3.189916 * max(0.0, Q.C2 - 0.07996447)
            - 34.11252 * max(0.0, 0.007856958 - Q.girth2_top30)
            + 95.89755 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266)
            - 90.39785 * max(0.0, 0.01563836 - Q.girth2_top15)
            - 110.2406 * max(0.0, 0.005383629 - Q.girth2_top15)
            - 0.00100911 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) * max(0.0, 26.76006 - Q.sj3_mass1)
        ))
        + 0.984375 * grid(10, max(0.0, -3.948963
            - 8.881083 * max(0.0, 0.1207452 - Q.girth)
            - 0.04043477 * max(0.0, Q.mass - 162.8363)
            - 74.06046 * max(0.0, 0.005402331 - Q.girth2_top30)
            - 0.03294256 * max(0.0, Q.mass_top40 - 163.2541)
            - 0.02077042 * max(0.0, 80.3008 - Q.sj2_mass1)
            + 0.04475454 * max(0.0, 87.27603 - Q.mass_top40)
            + 0.232346 * max(0.0, 3.814159 - Q.D2)
            - 215.0982 * max(0.0, Q.psi_0p3 - 0.9985421)
            + 2.374723e-05 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, 464.75 - Q.sum_pt_top2)
            + 0.008188306 * max(0.0, Q.mass_top5 - 14.54404)
            - 2.467481 * max(0.0, 3.814159 - Q.D2) * max(0.0, Q.sj2_dr - 0.1937688)
            + 30.05285 * max(0.0, Q.mass_top40 - 163.2541) * max(0.0, Q.soft4_z - 0.001186042)
            + 0.01090531 * max(0.0, Q.mass_top50 - 157.5448)
            - 1.393145 * max(0.0, Q.z_dr_0_0p05 - 0.7674734)
            - 7.537004 * max(0.0, Q.log_sum_pt - 6.903423)
            + 0.08921913 * max(0.0, Q.mass - 82.85409)
            - 0.07994704 * max(0.0, Q.mass - 143.7876)
            - 0.05977461 * max(0.0, Q.mass - 64.48544)
            + 0.004303969 * max(0.0, Q.mass - 64.48544) * max(0.0, 2.537109 - Q.soft2_pt)
            + 0.003656236 * Q.sum_pt
            - 0.003736029 * max(0.0, 75.26407 - Q.mass_top15)
            + 19.25813 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.0830523 - Q.dr_4)
            - 310.9234 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            + 0.001829684 * max(0.0, 839.9547 - Q.sum_pt_top5)
            + 0.0008142922 * max(0.0, 111.2487 - Q.mass_top40) * max(0.0, Q.pt1_dr01 - 12.6865)
            - 22.70827 * max(0.0, 0.06524004 - Q.e2)
            + 92.61685 * max(0.0, 0.01807679 - Q.girth2_top30)
            + 6.914821 * max(0.0, 0.06413297 - Q.dr_0)
            - 0.04096867 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 4.206493 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            + 0.02819799 * max(0.0, 132.4278 - Q.mass_top40)
            - 0.05756879 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.02782281 * max(0.0, Q.mass_top40 - 67.72643)
            - 0.006647105 * max(0.0, 71.781 - Q.mass_top10)
            + 727.3351 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            + 4.029475 * max(0.0, 0.06940472 - Q.dr_1)
            - 40.51704 * max(0.0, 0.05444509 - Q.tau1)
            + 0.02880678 * max(0.0, 74.25181 - Q.mass)
            + 0.0002261132 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.91979)
            + 1.226455 * max(0.0, 0.6133424 - Q.tau21_b2)
        ))
        - 0.375 * grid(12, max(0.0, -0.1994202
            + 0.01223357 * max(0.0, 87.36377 - Q.mass)
            - 0.09773045 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            - 0.03374758 * max(0.0, 94.64253 - Q.mass_top40)
            - 2.682579 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3)
            + 0.9075831 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367)
            - 11.1208 * max(0.0, 87.36377 - Q.mass) * max(0.0, Q.lam2 - 0.0003497174)
            + 0.05174195 * max(0.0, 82.85409 - Q.mass)
            - 373.0883 * max(0.0, 0.00363788 - Q.e2_sq)
            - 0.003510133 * max(0.0, 77.93668 - Q.mass_top40)
            - 0.04962413 * max(0.0, 53.87362 - Q.mass)
            + 3.315602e-05 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 435.25 - Q.pt_0)
            + 36.39789 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.3036026 - Q.planar_flow)
            + 0.001229078 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1)
            + 0.06322242 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            - 0.001456983 * max(0.0, Q.sum_pt - 1260.541)
            - 0.03622926 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.soft1_pt - 0.8144531)
            - 0.003879674 * max(0.0, 972.0419 - Q.sum_pt)
            + 4.797264 * max(0.0, 0.0684915 - Q.z_dr_0p2_0p4)
            - 5714.218 * max(0.0, Q.z_dr_0p1_0p2 - 0.4479367) * max(0.0, 0.001776308 - Q.lam2)
            + 0.07497664 * max(0.0, 80.78464 - Q.mass)
            + 13.44873 * max(0.0, 77.93668 - Q.mass_top40) * max(0.0, Q.lam2 - 0.000404306)
            - 4.510786 * max(0.0, Q.z_dr_0_0p05 - 0.9084912)
        ))
        - 0.90625 * grid(13, max(0.0, 1.183915
            - 0.006722675 * max(0.0, 1085.125 - Q.sum_pt)
            - 0.1719253 * max(0.0, Q.mass - 143.7876)
            - 0.08737683 * max(0.0, 14.0 - Q.n_for_90pct)
            + 0.08039576 * max(0.0, Q.mass - 172.4888)
            + 0.01398955 * max(0.0, Q.mass - 64.48544)
            - 0.03593113 * max(0.0, 1012.673 - Q.sum_pt)
            - 12.94024 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 0.004774652 * max(0.0, 1053.047 - Q.sum_pt_top40)
            + 96.13853 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            + 0.007236463 * max(0.0, Q.mass_top10 - 60.5496)
            + 33.77415 * max(0.0, 0.02497133 - Q.girth2_top40)
            + 2.472223 * max(0.0, Q.log_sum_pt - 7.139296)
            + 0.02621826 * max(0.0, 14.0 - Q.n_for_90pct) * max(0.0, 3.345339 - Q.D2)
            + 0.08642204 * max(0.0, Q.mass_top50 - 138.8977)
            - 0.1293539 * max(0.0, Q.mass_top50 - 157.5448)
            - 0.01185274 * max(0.0, 19.0 - Q.n_dr_0p1_0p2)
            + 4.269772 * max(0.0, 6.879399 - Q.log_sum_pt)
            - 0.002755389 * max(0.0, 889.8383 - Q.sum_pt_top20)
            + 1.267996 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.01833434 - Q.z_9)
            - 0.009428357 * max(0.0, Q.mass - 172.4888) * max(0.0, 41.4375 - Q.pt_9)
            + 0.004335024 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.5260785 - Q.D3)
            - 0.02347438 * max(0.0, Q.mass_top50 - 97.93004)
            - 7.372589 * max(0.0, Q.mass_over_sum_pt - 0.06030419)
            + 0.0142024 * max(0.0, 1008.935 - Q.sum_pt_top50)
            - 0.01524695 * max(0.0, Q.mass_top10 - 71.781)
            + 13.02367 * max(0.0, Q.psi_0p3 - 0.9777125)
            + 0.008051488 * max(0.0, Q.mass_top5 - 37.76455)
            - 0.03988171 * max(0.0, Q.mass_top50 - 168.9698)
            + 0.1144895 * max(0.0, Q.mass_top40 - 150.0144)
            + 3.061151 * max(0.0, 0.08786745 - Q.tau1)
            + 0.001822475 * max(0.0, Q.mass_top50 - 97.93004) * max(0.0, 2.873047 - Q.soft3_pt)
            - 0.002794427 * max(0.0, Q.sum_pt - 1260.541)
            + 0.01545802 * max(0.0, Q.mass - 78.26182)
        ))
        - 0.375 * grid(15, max(0.0, -0.5934045
            - 0.4567752 * max(0.0, 0.8459004 - Q.z_dr_0_0p05)
            + 65.89934 * max(0.0, 0.00287991 - Q.girth2_top20)
            + 6.447457 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2)
            - 0.04437039 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 76.29481 - Q.sd_mass)
            + 115.5413 * max(0.0, 0.008329695 - Q.girth2_top5)
            + 0.004972314 * max(0.0, 986.0565 - Q.sum_pt)
            + 6.833269 * max(0.0, 7.017258 - Q.log_sum_pt)
            + 0.01022072 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt)
            - 54.35542 * max(0.0, Q.psi_0p3 - 0.9896594)
            - 0.01238817 * max(0.0, 1002.379 - Q.sum_pt)
            - 214.1313 * max(0.0, 0.002270363 - Q.girth2_top5)
            + 2.748398 * max(0.0, Q.sj2_dr - 0.2232169)
            + 159.5411 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2)
            + 34.36351 * max(0.0, 0.007678544 - Q.girth2_top10)
            + 0.0170685 * max(0.0, 13.0 - Q.n_dr_0p1_0p2)
            - 1.373813 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677)
            + 188.7687 * max(0.0, 0.001776308 - Q.lam2)
            - 19.75679 * max(0.0, 0.0705748 - Q.tau1)
            - 0.001161996 * max(0.0, 1017.778 - Q.sum_pt_top20)
            - 462.0352 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.03860363 * max(0.0, Q.n_dr_0p2_0p4 - 9.0)
            - 92.18994 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583)
            + 0.02092804 * max(0.0, 79.18312 - Q.sd_mass)
            - 0.02170075 * max(0.0, 40.97891 - Q.sd_mass)
            - 0.0105911 * max(0.0, 71.79516 - Q.mass_top50)
            + 69.89853 * max(0.0, 0.004673423 - Q.lam1)
            - 0.0006493807 * max(0.0, 1018.832 - Q.sum_pt_top30)
            + 0.0007463146 * max(0.0, 869.693 - Q.sum_pt_top20)
            - 5.583593 * max(0.0, 6.903423 - Q.log_sum_pt)
            - 6.229167 * max(0.0, Q.psi_0p1 - 0.9371031)
            + 0.01782458 * max(0.0, 52.15154 - Q.mass_top50)
        ))
    )


def logits(Q):
    return [logit_g(Q), logit_q(Q), logit_W(Q), logit_Z(Q), logit_t(Q)]


def classify(pt, eta, phi):
    s = logits(quantities(pt, eta, phi))
    m = max(s)
    e = [math.exp(x - m) for x in s]
    p = [x / sum(e) for x in e]
    return CLASSES[s.index(m)], s, p


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
