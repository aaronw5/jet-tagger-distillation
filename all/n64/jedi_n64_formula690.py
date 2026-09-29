"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; all observables), as if-statements.

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
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_13         mass of particles 0 and 13 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_11                  pT of particle 11 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.sj3_z2                 pT share of subjet 2 of 3 (by pT)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_11                 pT share × ΔR of particle 11 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.pt2_over_pt0           pT2 / pT0
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft10_dr0             ΔR between the hardest and the 10. softest real particle (0 if among the 15 hardest)
  Q.soft3_dr0              ΔR between the hardest and the 3. softest real particle (0 if among the 15 hardest)
  Q.dr1_12                 ΔR between particle 12 and the 2nd-hardest particle
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_9                   ΔR of particle 9 from the jet axis
  Q.soft3_dr               ΔR from the jet axis of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
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
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
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
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.tau43                  N-subjettiness τ4/τ3
  Q.pt_dispersion          √(Σ pTᵢ²) / Σ pTᵢ
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
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_13=pair_mass(0, 13),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_90pct=ncum(0.9),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_11=pt[11],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_6=pt[6],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        z_11=z[11],
        z_6=z[6],
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft7_z=softp(7, 'z'),
        soft8_z=softp(8, 'z'),
        sj3_z2=subjets(3)["z"][1],
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_11=z[11] * dr[11],
        zdr_4=z[4] * dr[4],
        zdr_5=z[5] * dr[5],
        z_2nd=zs[1],
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft10_dr0=softp(10, 'dr0'),
        soft3_dr0=softp(3, 'dr0'),
        dr1_12=math.sqrt(dist2(1, 12)) if pt[12] > 0 else 0.0,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_9=dr[9] if pt[9] > 0 else 0.0,
        soft3_dr=softp(3, 'dr'),
        eta_0=eta[0],
        eta_1=eta[1],
        phi_0=phi[0],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
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
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
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
        tau3=tau_n(3),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = 1.249546
    if Q.mass < 74.25181:
        z += 0.03268066 * Q.mass - 3.302371
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.002056379 * Q.mass - 1.028463
    if 78.26182 <= Q.mass < 91.19:
        z += -0.1241139 * Q.mass + 8.845852
    if 91.19 <= Q.mass < 92.85979:
        z += -0.07838798 * Q.mass + 4.676106
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.171429 * Q.mass + 13.31588
    if Q.mass >= 101.0497:
        z += -0.2041097 * Q.mass + 16.61825
    if Q.girth2_top20 < 0.005312783:
        z += 46.29106 * Q.girth2_top20 - 0.4259789
    if 0.005312783 <= Q.girth2_top20 < 0.006374178:
        z += 120.3905 * Q.girth2_top20 - 0.8196533
    if 0.006374178 <= Q.girth2_top20 < 0.007538019:
        z += 44.90532 * Q.girth2_top20 - 0.3384972
    if Q.sum_pt < 1012.673:
        z += 0.008844724 * Q.sum_pt - 8.956812
    if Q.psi_0p3 >= 0.9956185:
        z += 48.42516 * Q.psi_0p3 - 48.21298
    if Q.mass_top30 < 80.4:
        z += -0.008969318 * Q.mass_top30 + 0.7211332
    if Q.girth2_top30 < 0.006363916:
        z += 97.64907 * Q.girth2_top30 - 0.5565674
    if 0.006363916 <= Q.girth2_top30 < 0.007856958:
        z += -43.44356 * Q.girth2_top30 + 0.3413342
    if Q.lam1 < 0.005913555:
        z += 93.82037 * Q.lam1 - 0.5548118
    if Q.mass_over_sum_pt_sq < 0.004754444:
        z += -818.4938 * Q.mass_over_sum_pt_sq + 5.812868
    if 0.004754444 <= Q.mass_over_sum_pt_sq < 0.006938798:
        z += -742.5564 * Q.mass_over_sum_pt_sq + 5.451828
    if 0.006938798 <= Q.mass_over_sum_pt_sq < 0.007873266:
        z += -320.3735 * Q.mass_over_sum_pt_sq + 2.522386
    if Q.tau1 < 0.0705748:
        z += 19.42748 * Q.tau1 - 1.371091
    if Q.e2_sq < 0.00616708:
        z += 435.4731 * Q.e2_sq - 2.685597
    if 71.79516 <= Q.mass_top50 < 82.04491:
        z += 0.00673609 * Q.mass_top50 - 0.4836187
    if Q.mass_top50 >= 82.04491:
        z += 0.03093275 * Q.mass_top50 - 2.468831
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += 4.735061 * Q.z_dr_0p2_0p4 - 0.4319591
    if Q.z_top50_slots < 0.9906378:
        z += 24.84985 * Q.z_top50_slots - 24.6172
    if Q.log_sum_pt < 7.017258:
        z += -2.274264 * Q.log_sum_pt + 15.9591
    if Q.sum_pt_top40 < 1069.671:
        z += 0.003169942 * Q.sum_pt_top40 - 3.390796
    if Q.sum_pt_top20 < 846.1934:
        z += -0.001925021 * Q.sum_pt_top20 + 1.62894
    if Q.mass_top40 < 80.89043:
        z += -0.005581344 * Q.mass_top40 + 0.4514774
    if Q.sum_pt_top50 < 1156.659:
        z += -0.003263093 * Q.sum_pt_top50 + 3.774288
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.04623683 * Q.n_dr_0p2_0p4 + 0.6935525
    if Q.log_sum_pt < 6.98945 and Q.sum_pt_top50 > 959.0957:
        z += 0.04865929 * (6.98945 - Q.log_sum_pt) * (Q.sum_pt_top50 - 959.0957)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.355078
    if Q.n_particles >= 38.0:
        z += 0.06012084 * Q.n_particles - 2.284592
    if Q.log_sum_pt < 6.893714:
        z += 5.381246 * Q.log_sum_pt - 38.41831
    if 6.893714 <= Q.log_sum_pt < 6.910131:
        z += 31.31663 * Q.log_sum_pt - 217.2094
    if 6.910131 <= Q.log_sum_pt < 6.959294:
        z += 49.98002 * Q.log_sum_pt - 346.1759
    if 6.959294 <= Q.log_sum_pt < 6.98945:
        z += 30.96222 * Q.log_sum_pt - 213.8254
    if 6.98945 <= Q.log_sum_pt < 7.062574:
        z += 22.63868 * Q.log_sum_pt - 155.6485
    if 7.062574 <= Q.log_sum_pt < 7.139296:
        z += 15.59055 * Q.log_sum_pt - 105.8705
    if Q.log_sum_pt >= 7.139296:
        z += 10.2093 * Q.log_sum_pt - 67.45221
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.009136436 * Q.sum_pt_top50 + 8.762716
    if Q.psi_0p3 >= 0.9980008:
        z += -216.5338 * Q.psi_0p3 + 216.1009
    if Q.sum_pt_top2 < 689.25:
        z += -0.001429287 * Q.sum_pt_top2 + 0.9851364
    if Q.z_top30_slots >= 0.9341838:
        z += -7.679985 * Q.z_top30_slots + 7.174517
    if Q.mass_top20 < 47.88842:
        z += 0.04525841 * Q.mass_top20 - 2.167354
    if Q.sj3_mass1 < 32.50209:
        z += 0.009576886 * Q.sj3_mass1 - 0.3112688
    if Q.sum_pt_top40 < 1069.671:
        z += -0.007369913 * Q.sum_pt_top40 + 7.883384
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.07458565 * Q.n_dr_0p2_0p4 - 0.5220995
    if Q.M3 < 0.03187688:
        z += 16.79871 * Q.M3 - 0.5354905
    if Q.sj2_mass1 < 30.26161:
        z += 0.01641212 * Q.sj2_mass1 - 0.4966572
    if Q.pt_9 < 31.35938:
        z += 0.03741481 * Q.pt_9 - 1.173305
    if Q.lam1 < 0.004673423:
        z += -118.2587 * Q.lam1 + 0.552673
    if Q.mass < 120.6:
        z += 0.01993514 * Q.mass - 2.404178
    if Q.girth2_top30 < 0.02412652:
        z += -8.064263 * Q.girth2_top30 + 0.1945626
    if Q.D3 < 0.1416054:
        z += -1.637488 * Q.D3 + 0.2318772
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -3.961654 * Q.z_dr_0_0p05 + 3.481922
    if Q.n_dr_0_0p05 < 12.0:
        z += -0.01975435 * Q.n_dr_0_0p05 + 0.2370522
    if Q.mass_top10 >= 56.92192:
        z += 0.008245779 * Q.mass_top10 - 0.4693656
    if Q.soft1_pt < 1.521582:
        z += -0.212885 * Q.soft1_pt - 0.4632247
    if 1.521582 <= Q.soft1_pt < 2.275391:
        z += 1.044226 * Q.soft1_pt - 2.376023
    if Q.sum_pt < 1017.435:
        z += -0.007530191 * Q.sum_pt + 7.661477
    if Q.sum_pt_top30 >= 1191.938:
        z += 0.004716503 * Q.sum_pt_top30 - 5.621779
    if Q.z_top20_slots >= 0.8965411:
        z += 5.53653 * Q.z_top20_slots - 4.963727
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.05423471 * Q.n_dr_0p1_0p2 - 0.4338776
    if Q.pt_entropy >= 2.07371:
        z += 1.093313 * Q.pt_entropy - 2.267214
    if Q.girth2_top15 < 0.0007894752:
        z += -492.9975 * Q.girth2_top15 + 0.3892093
    if Q.girth2_top20 < 0.001823079:
        z += -320.3291 * Q.girth2_top20 + 0.8189218
    if 0.001823079 <= Q.girth2_top20 < 0.007538019:
        z += -41.10921 * Q.girth2_top20 + 0.309882
    if Q.psi_0p1 >= 0.8747961:
        z += 1.796618 * Q.psi_0p1 - 1.571674
    if Q.LHA < 0.2601462:
        z += -8.347257 * Q.LHA + 2.171507
    if Q.tau3 < 0.04516808:
        z += -11.78001 * Q.tau3 + 0.5320804
    if Q.e2 < 0.02515919:
        z += 39.28624 * Q.e2 - 0.9884101
    if Q.girth2_top5 < 0.0006570502:
        z += -348.7478 * Q.girth2_top5 + 0.2291448
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += 163.1715 * Q.mass_over_sum_pt_sq - 1.565633
    if Q.girth2_top40 < 0.008031986:
        z += -210.0191 * Q.girth2_top40 + 1.68687
    if Q.z_top50_slots >= 0.9586536:
        z += -23.90451 * Q.z_top50_slots + 22.91615
    if Q.n_for_90pct >= 11.0:
        z += -0.04495908 * Q.n_for_90pct + 0.4945499
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -4.098588 * Q.z_dr_0p2_0p4 + 0.3738965
    if Q.tau1 < 0.1953848:
        z += -12.05597 * Q.tau1 + 2.355555
    if Q.z_top30_slots > 0.9341838 and Q.max_pair_mass > 13.04793:
        z += 0.3796949 * (Q.z_top30_slots - 0.9341838) * (Q.max_pair_mass - 13.04793)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.1119555:
        z += 0.2202526 * (Q.n_particles - 38.0) * (0.1119555 - Q.dr_0)
    if Q.mass_top20 < 47.88842 and Q.n_real_top40 > 29.0:
        z += 0.003248465 * (47.88842 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 38.0 and Q.soft1_pt < 2.275391:
        z += -0.01444483 * (Q.n_particles - 38.0) * (2.275391 - Q.soft1_pt)
    if Q.z_top30_slots > 0.9341838 and Q.C2 < 0.07279889:
        z += 208.8273 * (Q.z_top30_slots - 0.9341838) * (0.07279889 - Q.C2)
    if Q.sj3_mass1 < 32.50209 and Q.sj3_mass2 < 18.68222:
        z += -0.001626019 * (32.50209 - Q.sj3_mass1) * (18.68222 - Q.sj3_mass2)
    if Q.girth2_top15 < 0.003270031 and Q.psi_0p3 > 0.9973959:
        z += 39598.39 * (0.003270031 - Q.girth2_top15) * (Q.psi_0p3 - 0.9973959)
    if Q.n_particles > 38.0 and Q.dr_1 < 0.1611545:
        z += 0.1341038 * (Q.n_particles - 38.0) * (0.1611545 - Q.dr_1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 > 7.407874:
        z += 0.6319509 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_3 - 7.407874)
    if Q.pt_9 < 31.35938 and Q.dr1_12 < 0.3241858:
        z += 0.0448898 * (31.35938 - Q.pt_9) * (0.3241858 - Q.dr1_12)
    if Q.pt_9 < 31.35938 and Q.pair_mass_0_13 < 11.29505:
        z += 0.001732598 * (31.35938 - Q.pt_9) * (11.29505 - Q.pair_mass_0_13)
    if Q.sum_pt_top5 > 430.75 and Q.eta_0 < 0.02980347:
        z += -0.01018586 * (Q.sum_pt_top5 - 430.75) * (0.02980347 - Q.eta_0)
    if Q.z_top20_slots > 0.8965411 and Q.dr_2 < 0.03843804:
        z += -314.8456 * (Q.z_top20_slots - 0.8965411) * (0.03843804 - Q.dr_2)
    if Q.mass < 120.6 and Q.dr_2 < 0.03843804:
        z += 0.1349861 * (120.6 - Q.mass) * (0.03843804 - Q.dr_2)
    if Q.girth2_top20 < 0.007538019 and Q.eta_1 > -0.04302979:
        z += 618.1078 * (0.007538019 - Q.girth2_top20) * (Q.eta_1 - -0.04302979)
    if Q.sj2_mass1 < 30.26161 and Q.soft10_dr0 < 0.2366434:
        z += 0.04707906 * (30.26161 - Q.sj2_mass1) * (0.2366434 - Q.soft10_dr0)
    if Q.pt_entropy > 2.07371 and Q.soft3_dr < 0.3797:
        z += -0.7530735 * (Q.pt_entropy - 2.07371) * (0.3797 - Q.soft3_dr)
    if Q.sum_pt_top2 < 689.25 and Q.soft3_dr0 > 0.02918107:
        z += -0.001620874 * (689.25 - Q.sum_pt_top2) * (Q.soft3_dr0 - 0.02918107)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.2612645
    if 6.959294 <= Q.log_sum_pt < 7.062574:
        z += -12.51421 * Q.log_sum_pt + 87.09003
    if Q.log_sum_pt >= 7.062574:
        z += -18.64909 * Q.log_sum_pt + 130.4181
    if Q.sum_pt < 972.0419:
        z += 0.01425504 * Q.sum_pt - 14.36492
    if 972.0419 <= Q.sum_pt < 1017.435:
        z += 0.001762296 * Q.sum_pt - 2.221446
    if 1017.435 <= Q.sum_pt < 1115.723:
        z += 0.0119162 * Q.sum_pt - 12.55238
    if 1115.723 <= Q.sum_pt < 1260.541:
        z += 0.007188233 * Q.sum_pt - 7.27728
    if Q.sum_pt >= 1260.541:
        z += 0.005425937 * Q.sum_pt - 5.055834
    if Q.sum_pt_top50 < 1048.098:
        z += 0.0001431247 * Q.sum_pt_top50 - 0.2899781
    if 1048.098 <= Q.sum_pt_top50 < 1078.994:
        z += 0.004530346 * Q.sum_pt_top50 - 4.888217
    if Q.sum_pt_top50 >= 1156.659:
        z += 0.003279759 * Q.sum_pt_top50 - 3.793565
    if Q.mass < 91.19:
        z += 0.03808445 * Q.mass - 3.58004
    if 91.19 <= Q.mass < 92.85979:
        z += 0.06415118 * Q.mass - 5.957066
    if Q.mass_over_sum_pt < 0.09795415:
        z += -6.006827 * Q.mass_over_sum_pt + 0.5883936
    if Q.sum_pt_top40 < 984.7009:
        z += -0.006821814 * Q.sum_pt_top40 + 7.103303
    if 984.7009 <= Q.sum_pt_top40 < 1041.263:
        z += -0.008387158 * Q.sum_pt_top40 + 8.644698
    if 1041.263 <= Q.sum_pt_top40 < 1069.671:
        z += -0.001565344 * Q.sum_pt_top40 + 1.541396
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.003763714 * Q.sum_pt_top40 - 4.158944
    if Q.sum_pt_top30 < 966.0633:
        z += 0.004013048 * Q.sum_pt_top30 - 4.078707
    if 966.0633 <= Q.sum_pt_top30 < 996.8867:
        z += 0.006548543 * Q.sum_pt_top30 - 6.528156
    if Q.lam1 < 0.01174405:
        z += -35.78027 * Q.lam1 + 0.4202053
    if Q.mass_top50 < 92.16545:
        z += -0.01952696 * Q.mass_top50 + 1.799711
    if Q.log_sum_pt > 6.903423 and Q.girth2_top15 < 0.02146578:
        z += 688.4915 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.girth2_top15)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p3 > 0.9299135:
        z += -21.1581 * (Q.log_sum_pt - 6.903423) * (Q.psi_0p3 - 0.9299135)
    if Q.sum_pt_top50 > 988.4554 and Q.girth2_top15 < 0.02146578:
        z += -0.3629539 * (Q.sum_pt_top50 - 988.4554) * (0.02146578 - Q.girth2_top15)
    if Q.sum_pt < 972.0419 and Q.e4 < 5.8505e-08:
        z += 238963.8 * (972.0419 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.sum_pt_top20 > 1129.275 and Q.eta_1 > 0.08734131:
        z += -0.6412961 * (Q.sum_pt_top20 - 1129.275) * (Q.eta_1 - 0.08734131)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.09804223
    if Q.lam2 < 0.0006154841:
        z += -843.1049 * Q.lam2 + 0.5189176
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.1034565 * Q.n_dr_0p2_0p4 + 0.5172826
    if Q.n_particles < 46.0:
        z += -0.02552042 * Q.n_particles + 1.173939
    if Q.tau21 < 0.3861957:
        z += 2.250729 * Q.tau21 - 0.8692218
    if Q.D2 < 2.410481:
        z += 0.475418 * Q.D2 - 1.145986
    if Q.mass_over_sum_pt < 0.08665515:
        z += -7.090916 * Q.mass_over_sum_pt + 0.6144644
    if Q.mass_top50 < 79.21004:
        z += 0.01081799 * Q.mass_top50 - 0.8568934
    if Q.mass < 53.87362:
        z += 0.01989457 * Q.mass - 1.004361
    if 53.87362 <= Q.mass < 86.4:
        z += -0.0008419333 * Q.mass + 0.1127896
    if 86.4 <= Q.mass < 92.85979:
        z += -0.006199353 * Q.mass + 0.5756707
    if Q.girth2 < 0.009614971:
        z += 147.1815 * Q.girth2 - 1.415146
    if Q.girth2_top40 < 0.008840538:
        z += -131.7203 * Q.girth2_top40 + 1.164478
    if Q.sj2_mass1 < 27.56535:
        z += -0.01367375 * Q.sj2_mass1 + 0.3769219
    if Q.mass_top40 < 80.4:
        z += 0.0060257 * Q.mass_top40 - 0.4844663
    if Q.tau4 < 0.01626937:
        z += -30.65035 * Q.tau4 + 0.4986618
    if Q.girth2_top50 < 0.008124776:
        z += -61.24456 * Q.girth2_top50 + 0.4975983
    if Q.n_particles < 46.0 and Q.mass_top15 > 57.87349:
        z += -0.0007444088 * (46.0 - Q.n_particles) * (Q.mass_top15 - 57.87349)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_dr_0p1_0p2 > 9.0:
        z += -0.01095947 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 9.0)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.zdr_5 < 0.007099471:
        z += -15.22755 * (5.0 - Q.n_dr_0p2_0p4) * (0.007099471 - Q.zdr_5)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.psi_0p1 < 0.9538343:
        z += 0.533788 * (5.0 - Q.n_dr_0p2_0p4) * (0.9538343 - Q.psi_0p1)
    if Q.tau21 < 0.3861957 and Q.lam1 > 0.007671243:
        z += -282.7285 * (0.3861957 - Q.tau21) * (Q.lam1 - 0.007671243)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.z_top50_slots > 0.978741:
        z += 2.796216 * (8.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.978741)
    if Q.D2 < 2.410481 and Q.D2_b2 < 5.142486:
        z += 0.08884802 * (2.410481 - Q.D2) * (5.142486 - Q.D2_b2)
    if Q.n_dr_0p1_0p2 < 10.0 and Q.psi_0p2 > 0.9313699:
        z += 0.6554723 * (10.0 - Q.n_dr_0p1_0p2) * (Q.psi_0p2 - 0.9313699)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.5943322
    if Q.mass_top40 < 62.55:
        z += -0.04089108 * Q.mass_top40 + 1.627731
    if 62.55 <= Q.mass_top40 < 67.72643:
        z += 0.001852305 * Q.mass_top40 - 1.045867
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.001466346 * Q.mass_top40 - 1.019727
    if 83.32554 <= Q.mass_top40 < 163.2541:
        z += 0.01122931 * Q.mass_top40 - 1.833232
    if Q.e3 >= 0.0001841806:
        z += 946.9125 * Q.e3 - 0.1744029
    if Q.mass_top30 < 60.43821:
        z += 0.000958018 * Q.mass_top30 - 0.6881952
    if 60.43821 <= Q.mass_top30 < 91.69753:
        z += 0.0201634 * Q.mass_top30 - 1.848934
    if Q.mass < 74.25181:
        z += 0.09588532 * Q.mass - 6.702912
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.06872238 * Q.mass - 4.686015
    if 78.26182 <= Q.mass < 80.78464:
        z += 0.03397572 * Q.mass - 1.966678
    if 80.78464 <= Q.mass < 86.4:
        z += 0.05534859 * Q.mass - 3.693278
    if 86.4 <= Q.mass < 92.85979:
        z += -0.02148664 * Q.mass + 2.945286
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.05821677 * Q.mass + 6.356038
    if 101.0497 <= Q.mass < 120.6:
        z += -0.02053976 * Q.mass + 2.548787
    if 120.6 <= Q.mass < 143.7876:
        z += -0.003091837 * Q.mass + 0.4445678
    if Q.psi_0p3 >= 0.9973959:
        z += 204.96 * Q.psi_0p3 - 204.4263
    if Q.n_particles >= 22.0:
        z += -0.01642663 * Q.n_particles + 0.3613858
    if Q.girth2_top15 < 0.004855289:
        z += 175.3547 * Q.girth2_top15 - 0.8513977
    if 0.00727763 <= Q.girth2_top15 < 0.01563836:
        z += -62.8688 * Q.girth2_top15 + 0.4575359
    if Q.girth2_top15 >= 0.01563836:
        z += 4.395287 * Q.girth2_top15 - 0.5943638
    if 0.02793599 <= Q.e2 < 0.04755309:
        z += 11.36042 * Q.e2 - 0.3173648
    if 0.04755309 <= Q.e2 < 0.06524004:
        z += -14.61792 * Q.e2 + 0.9179858
    if Q.e2 >= 0.06524004:
        z += -41.95665 * Q.e2 + 2.701565
    if 0.2232169 <= Q.sj2_dr < 0.2412757:
        z += 4.582783 * Q.sj2_dr - 1.022955
    if Q.sj2_dr >= 0.2412757:
        z += -0.03409815 * Q.sj2_dr + 0.09098652
    if 0.009606007 <= Q.e2_sq < 0.01396296:
        z += 125.1212 * Q.e2_sq - 1.201915
    if Q.e2_sq >= 0.01396296:
        z += 163.3834 * Q.e2_sq - 1.736169
    if Q.mass_top15 < 57.87349:
        z += -0.01230229 * Q.mass_top15 + 0.7290885
    if 57.87349 <= Q.mass_top15 < 91.19:
        z += -0.0005136178 * Q.mass_top15 + 0.0468368
    if Q.n_dr_0p2_0p4 < 26.0:
        z += -0.02978952 * Q.n_dr_0p2_0p4 + 0.7745276
    if Q.log_sum_pt >= 6.97212:
        z += -1.410454 * Q.log_sum_pt + 9.833858
    if Q.girth < 0.03577037:
        z += 42.71303 * Q.girth - 1.527861
    if Q.girth >= 0.1207452:
        z += -20.04686 * Q.girth + 2.420562
    if 0.008241985 <= Q.lam1 < 0.01174405:
        z += -98.20834 * Q.lam1 + 0.8094317
    if Q.lam1 >= 0.01174405:
        z += -56.8112 * Q.lam1 + 0.3232616
    if Q.psi_0p1 < 0.9909875:
        z += -0.5814582 * Q.psi_0p1 + 0.5762178
    if Q.C2 >= 0.06655881:
        z += -5.960823 * Q.C2 + 0.3967452
    if Q.sum_pt >= 995.6769:
        z += 0.00137314 * Q.sum_pt - 1.367204
    if 0.007164202 <= Q.girth2_top5 < 0.0168592:
        z += -18.55187 * Q.girth2_top5 + 0.1329093
    if Q.girth2_top5 >= 0.0168592:
        z += 2.424683 * Q.girth2_top5 - 0.2207385
    if Q.psi_0p3 > 0.9973959 and Q.n_dr_0_0p05 < 13.0:
        z += -12.74451 * (Q.psi_0p3 - 0.9973959) * (13.0 - Q.n_dr_0_0p05)
    if Q.n_particles > 22.0 and Q.n_dr_0_0p05 > 10.0:
        z += 0.0004883492 * (Q.n_particles - 22.0) * (Q.n_dr_0_0p05 - 10.0)
    if Q.n_particles > 22.0 and Q.soft1_pt < 2.275391:
        z += 0.005431767 * (Q.n_particles - 22.0) * (2.275391 - Q.soft1_pt)
    if Q.n_dr_0p2_0p4 < 26.0 and Q.n_dr_0p1_0p2 > 11.0:
        z += -0.001096563 * (26.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 11.0)
    if Q.sj2_dr > 0.2232169 and Q.D2_b2 < 20.40995:
        z += -0.1390807 * (Q.sj2_dr - 0.2232169) * (20.40995 - Q.D2_b2)
    if Q.mass < 80.78464 and Q.D2_b2 < 3.852812:
        z += -0.01614643 * (80.78464 - Q.mass) * (3.852812 - Q.D2_b2)
    if Q.mass < 101.0497 and Q.D2_b2 < 3.852812:
        z += 0.006655309 * (101.0497 - Q.mass) * (3.852812 - Q.D2_b2)
    if Q.mass_top40 < 83.32554 and Q.D2_b2 < 3.014827:
        z += -0.01241625 * (83.32554 - Q.mass_top40) * (3.014827 - Q.D2_b2)
    if Q.n_particles > 22.0 and Q.D2_b2 < 3.852812:
        z += -0.002233423 * (Q.n_particles - 22.0) * (3.852812 - Q.D2_b2)
    if Q.mass < 92.85979 and Q.girth2_top2 < 0.005180665:
        z += 1.928929 * (92.85979 - Q.mass) * (0.005180665 - Q.girth2_top2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.9150036
    z += -0.03731222 * Q.n_particles + 2.387982
    if 0.07696632 <= Q.mass_over_sum_pt < 0.09046749:
        z += -28.16886 * Q.mass_over_sum_pt + 2.168054
    if 0.09046749 <= Q.mass_over_sum_pt < 0.1708801:
        z += -39.00276 * Q.mass_over_sum_pt + 3.148169
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 147.5178 * Q.mass_over_sum_pt - 28.72448
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -16.9276 * Q.log_sum_pt + 116.9719
    if 6.920349 <= Q.log_sum_pt < 6.935549:
        z += -26.24125 * Q.log_sum_pt + 181.4256
    if 6.935549 <= Q.log_sum_pt < 6.949481:
        z += -24.29658 * Q.log_sum_pt + 167.9383
    if 6.949481 <= Q.log_sum_pt < 6.98945:
        z += -22.318 * Q.log_sum_pt + 154.1882
    if Q.log_sum_pt >= 6.98945:
        z += -16.55255 * Q.log_sum_pt + 113.8909
    if 907.9372 <= Q.sum_pt < 986.0565:
        z += 0.006973717 * Q.sum_pt - 6.331697
    if Q.sum_pt >= 986.0565:
        z += -0.007302103 * Q.sum_pt + 7.745069
    if Q.psi_0p3 >= 0.9973959:
        z += -129.6938 * Q.psi_0p3 + 129.3561
    if Q.girth2_top50 >= 0.01951641:
        z += -113.4888 * Q.girth2_top50 + 2.214894
    if Q.girth2 >= 0.004756928:
        z += 49.19172 * Q.girth2 - 0.2340015
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += 0.01235394 * Q.sum_pt_top50 - 11.54156
    if Q.sum_pt_top50 >= 959.0957:
        z += 0.01946879 * Q.sum_pt_top50 - 18.36538
    if Q.sum_pt_top3 < 787.6281:
        z += 0.0005152941 * Q.sum_pt_top3 - 0.4058601
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.02510502 * Q.n_dr_0p2_0p4 + 0.2761552
    if 994.2695 <= Q.sum_pt_top40 < 1024.942:
        z += -0.00314502 * Q.sum_pt_top40 + 3.126998
    if Q.sum_pt_top40 >= 1024.942:
        z += -0.004980058 * Q.sum_pt_top40 + 5.007806
    if Q.sum_pt_top30 >= 933.1875:
        z += 0.002186897 * Q.sum_pt_top30 - 2.040785
    if 0.2404747 <= Q.max_dr < 0.4357228:
        z += -2.938298 * Q.max_dr + 0.7065865
    if Q.max_dr >= 0.4357228:
        z += -0.3444054 * Q.max_dr - 0.4236319
    if Q.z_top30_slots >= 0.9048492:
        z += -4.822889 * Q.z_top30_slots + 4.363987
    if Q.mass < 64.48544:
        z += -0.01149416 * Q.mass + 0.7412062
    if 74.25181 <= Q.mass < 91.19:
        z += 0.02050276 * Q.mass - 1.522367
    if 91.19 <= Q.mass < 172.4888:
        z += 0.006330299 * Q.mass - 0.2299803
    if 172.4888 <= Q.mass < 172.8:
        z += 0.3173037 * Q.mass - 53.86941
    if Q.mass >= 172.8:
        z += -0.2694983 * Q.mass + 47.52998
    if 0.006363916 <= Q.girth2_top30 < 0.007463985:
        z += 51.94471 * Q.girth2_top30 - 0.3305717
    if Q.girth2_top30 >= 0.007463985:
        z += 17.98182 * Q.girth2_top30 - 0.07707327
    if Q.z_dr_0_0p05 >= 0.9084912:
        z += 4.48052 * Q.z_dr_0_0p05 - 4.070513
    if Q.mass_top40 < 150.0144:
        z += 0.004290388 * Q.mass_top40 - 0.6436203
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -504.5666 * Q.mass_over_sum_pt_sq + 14.73335
    if Q.tau21 < 0.5494307:
        z += 1.001578 * Q.tau21 - 0.550298
    if 69.65633 <= Q.sd_mass < 86.4:
        z += 0.03066129 * Q.sd_mass - 2.135753
    if Q.sd_mass >= 86.4:
        z += -0.01600418 * Q.sd_mass + 1.896144
    if Q.mass_top50 >= 157.5448:
        z += -0.263817 * Q.mass_top50 + 41.563
    if Q.z_11 < 0.01375115:
        z += -93.90166 * Q.z_11 + 1.291255
    if Q.e2 < 0.01541561:
        z += 22.37722 * Q.e2 + 0.3472249
    if 0.01541561 <= Q.e2 < 0.03875945:
        z += -29.65165 * Q.e2 + 1.149282
    if Q.n_dr_0p1_0p2 < 21.0:
        z += 0.01672467 * Q.n_dr_0p1_0p2 - 0.351218
    if Q.mass_top10 >= 71.781:
        z += 0.01312086 * Q.mass_top10 - 0.9418286
    if Q.pt_11 < 14.14062:
        z += 0.08539772 * Q.pt_11 - 1.207577
    if Q.D2 < 1.976207:
        z += -0.3205715 * Q.D2 + 0.6335156
    if Q.z_top20_slots >= 0.9102775:
        z += -5.06444 * Q.z_top20_slots + 4.610046
    if Q.z_top50_slots >= 0.9906378:
        z += -37.67212 * Q.z_top50_slots + 37.31943
    if Q.girth2_top15 < 0.005788041:
        z += 61.63405 * Q.girth2_top15 - 0.3567404
    if Q.girth2_top20 < 0.01083435:
        z += 33.37132 * Q.girth2_top20 - 0.3615567
    if Q.n_particles < 64.0 and Q.D2 < 2.178951:
        z += -0.01790453 * (64.0 - Q.n_particles) * (2.178951 - Q.D2)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 28402.86 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    if Q.psi_0p3 > 0.9973959 and Q.D2 < 3.345339:
        z += 58.76815 * (Q.psi_0p3 - 0.9973959) * (3.345339 - Q.D2)
    if Q.n_pt_above_1 > 28.0 and Q.tau43 < 0.942303:
        z += 0.0274454 * (Q.n_pt_above_1 - 28.0) * (0.942303 - Q.tau43)
    return max(0.0, z)


def neuron_6(Q):
    z = -0.1309592
    if Q.mass_top50 < 71.79516:
        z += -0.03556745 * Q.mass_top50 + 3.638858
    if 71.79516 <= Q.mass_top50 < 168.9698:
        z += -0.01007673 * Q.mass_top50 + 1.808748
    if 168.9698 <= Q.mass_top50 < 172.8:
        z += -0.02769687 * Q.mass_top50 + 4.786019
    if Q.mass < 86.4:
        z += -0.01309041 * Q.mass - 0.526307
    if 86.4 <= Q.mass < 92.85979:
        z += 0.02839971 * Q.mass - 4.111053
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.1002286 * Q.mass - 10.78106
    if 101.0497 <= Q.mass < 120.6:
        z += 0.03340097 * Q.mass - 4.028157
    if Q.e3 < 0.0003372339:
        z += 2059.379 * Q.e3 - 0.6944924
    if 29.00832 <= Q.sj3_pair_mass_min < 76.60223:
        z += -0.01378229 * Q.sj3_pair_mass_min + 0.3998011
    if Q.sj3_pair_mass_min >= 76.60223:
        z += -0.1390818 * Q.sj3_pair_mass_min + 9.998027
    if Q.e2 < 0.02793599:
        z += -15.54373 * Q.e2 + 1.014073
    if 0.02793599 <= Q.e2 < 0.04755309:
        z += -65.85042 * Q.e2 + 2.419441
    if 0.04755309 <= Q.e2 < 0.05557149:
        z += -22.93825 * Q.e2 + 0.3788348
    if 0.05557149 <= Q.e2 < 0.06524004:
        z += 36.80986 * Q.e2 - 2.941456
    if Q.e2 >= 0.06524004:
        z += 52.35359 * Q.e2 - 3.95553
    if Q.sj3_mass1 >= 21.11128:
        z += -0.01929173 * Q.sj3_mass1 + 0.4072729
    if Q.log_sum_pt < 6.811175:
        z += 5.082191 * Q.log_sum_pt - 34.6157
    if Q.tau1 < 0.05444509:
        z += -10.70439 * Q.tau1 + 0.675536
    if 0.05444509 <= Q.tau1 < 0.06310829:
        z += -3.445691 * Q.tau1 + 0.2803352
    if Q.tau1 >= 0.06310829:
        z += 7.258703 * Q.tau1 - 0.3952008
    if Q.z_top40_slots < 0.9300465:
        z += -10.7439 * Q.z_top40_slots + 9.992325
    if Q.mass_over_sum_pt < 0.02682209:
        z += 4.980411 * Q.mass_over_sum_pt - 0.4148654
    if 0.02682209 <= Q.mass_over_sum_pt < 0.08329945:
        z += 24.18693 * Q.mass_over_sum_pt - 0.9300243
    if 0.08329945 <= Q.mass_over_sum_pt < 0.09795415:
        z += 19.20652 * Q.mass_over_sum_pt - 0.5151588
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -26.4541 * Q.mass_over_sum_pt + 3.957488
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -56.23402 * Q.mass_over_sum_pt + 7.478247
    if Q.girth2_top30 < 0.00375223:
        z += -20.50994 * Q.girth2_top30 - 0.705028
    if 0.00375223 <= Q.girth2_top30 < 0.008376291:
        z += 54.59059 * Q.girth2_top30 - 0.9868224
    if 0.008376291 <= Q.girth2_top30 < 0.01807679:
        z += 164.9055 * Q.girth2_top30 - 1.910852
    if 0.01807679 <= Q.girth2_top30 < 0.02809026:
        z += 110.3149 * Q.girth2_top30 - 0.92403
    if Q.girth2_top30 >= 0.02809026:
        z += -146.5086 * Q.girth2_top30 + 6.29021
    if Q.girth2_top20 >= 0.008031209:
        z += 105.1103 * Q.girth2_top20 - 0.8441624
    if Q.lam1 < 0.003811746:
        z += -309.6999 * Q.lam1 + 1.536744
    if 0.003811746 <= Q.lam1 < 0.007259287:
        z += -103.3334 * Q.lam1 + 0.7501267
    if Q.girth2_top50 < 0.003418057:
        z += 395.086 * Q.girth2_top50 - 1.656697
    if 0.003418057 <= Q.girth2_top50 < 0.004573744:
        z += 265.0115 * Q.girth2_top50 - 1.212095
    if Q.lam2 < 0.003687605:
        z += -164.9561 * Q.lam2 + 0.6082928
    if Q.girth2_top40 >= 0.008031986:
        z += 44.98612 * Q.girth2_top40 - 0.3613279
    if Q.girth >= 0.02085222:
        z += -1.857785 * Q.girth + 0.03873895
    if Q.sum_pt < 1260.541:
        z += 0.001506028 * Q.sum_pt - 1.89841
    if Q.mass < 120.6 and Q.sum_pt < 1007.788:
        z += 0.0001024942 * (120.6 - Q.mass) * (1007.788 - Q.sum_pt)
    if Q.sj3_pair_mass_min > 29.00832 and Q.psi_0p3 > 0.9896594:
        z += 1.66115 * (Q.sj3_pair_mass_min - 29.00832) * (Q.psi_0p3 - 0.9896594)
    if Q.mass_over_sum_pt > 0.1182259 and Q.C2_b2 > 0.02704832:
        z += 617.8851 * (Q.mass_over_sum_pt - 0.1182259) * (Q.C2_b2 - 0.02704832)
    if Q.girth2_top30 > 0.008376291 and Q.D2_b2 > 1.67722:
        z += -19.22717 * (Q.girth2_top30 - 0.008376291) * (Q.D2_b2 - 1.67722)
    if Q.mass_over_sum_pt > 0.1182259 and Q.D2_b2 < 7.36624:
        z += -3.552588 * (Q.mass_over_sum_pt - 0.1182259) * (7.36624 - Q.D2_b2)
    if Q.girth2_top20 > 0.008031209 and Q.C2_b2 > 0.0008187529:
        z += -2517.034 * (Q.girth2_top20 - 0.008031209) * (Q.C2_b2 - 0.0008187529)
    if Q.mass_over_sum_pt > 0.02682209 and Q.z_dr_0p05_0p1 < 0.7108211:
        z += -5.697276 * (Q.mass_over_sum_pt - 0.02682209) * (0.7108211 - Q.z_dr_0p05_0p1)
    if Q.mass_over_sum_pt > 0.09795415 and Q.soft8_z < 0.001160626:
        z += -17885.54 * (Q.mass_over_sum_pt - 0.09795415) * (0.001160626 - Q.soft8_z)
    if Q.e2 > 0.02793599 and Q.max_dr < 0.4357228:
        z += 95.98814 * (Q.e2 - 0.02793599) * (0.4357228 - Q.max_dr)
    if Q.mass_over_sum_pt > 0.1182259 and Q.soft3_dr0 > 0.1966723:
        z += 42.89248 * (Q.mass_over_sum_pt - 0.1182259) * (Q.soft3_dr0 - 0.1966723)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.148775
    if Q.tau21_b2 < 0.2352054:
        z += -4.517002 * Q.tau21_b2 + 1.062424
    if Q.girth2 < 0.007877041:
        z += -66.16333 * Q.girth2 + 0.9060077
    if 0.007877041 <= Q.girth2 < 0.008190222:
        z += 163.8754 * Q.girth2 - 0.9060166
    if 0.008190222 <= Q.girth2 < 0.009614971:
        z += -306.1303 * Q.girth2 + 2.943434
    if Q.mass_over_sum_pt < 0.1182259:
        z += -7.147416 * Q.mass_over_sum_pt + 0.8450099
    if Q.mass < 78.26182:
        z += 0.02652248 * Q.mass - 2.201011
    if 78.26182 <= Q.mass < 82.85409:
        z += 0.0893913 * Q.mass - 7.12124
    if 82.85409 <= Q.mass < 91.19:
        z += 0.0002106847 * Q.mass + 0.2677389
    if 91.19 <= Q.mass < 92.85979:
        z += 0.0271681 * Q.mass - 2.190508
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.03548395 * Q.mass - 2.962716
    if 101.0497 <= Q.mass < 120.6:
        z += -0.0318628 * Q.mass + 3.842653
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.08087699 * Q.n_dr_0p2_0p4 + 0.6368687
    if 6.0 <= Q.n_dr_0p2_0p4 < 9.0:
        z += -0.05053557 * Q.n_dr_0p2_0p4 + 0.4548202
    if Q.psi_0p3 >= 0.9980008:
        z += -564.3218 * Q.psi_0p3 + 563.1937
    if Q.e2_sq < 0.00363788:
        z += 340.8612 * Q.e2_sq - 1.240012
    if Q.tau1 < 0.07708632:
        z += 10.65481 * Q.tau1 - 1.59992
    if 0.07708632 <= Q.tau1 < 0.08786745:
        z += 22.39038 * Q.tau1 - 2.504572
    if 0.08786745 <= Q.tau1 < 0.09591084:
        z += 14.68413 * Q.tau1 - 1.827443
    if 0.09591084 <= Q.tau1 < 0.1072713:
        z += 36.88916 * Q.tau1 - 3.957146
    if Q.lam1 < 0.006189818:
        z += -22.44823 * Q.lam1 - 0.05513171
    if 0.006189818 <= Q.lam1 < 0.008241985:
        z += 176.7106 * Q.lam1 - 1.287888
    if 0.008241985 <= Q.lam1 < 0.01174405:
        z += -48.13087 * Q.lam1 + 0.5652514
    if Q.sd_mass < 69.65633:
        z += -0.01254189 * Q.sd_mass + 0.4711269
    if 69.65633 <= Q.sd_mass < 86.4:
        z += 0.02403863 * Q.sd_mass - 2.076938
    if Q.sd_mass >= 98.05743:
        z += -0.09302575 * Q.sd_mass + 9.121866
    if 73.35236 <= Q.mass_top20 < 85.79457:
        z += -0.01018549 * Q.mass_top20 + 0.74713
    if Q.mass_top20 >= 85.79457:
        z += 0.007223977 * Q.mass_top20 - 0.7465079
    if Q.psi_0p2 >= 0.9313699:
        z += -5.432158 * Q.psi_0p2 + 5.059348
    if Q.sd_rg < 0.1596365:
        z += 2.437512 * Q.sd_rg - 0.3587833
    if 0.1596365 <= Q.sd_rg < 0.1881908:
        z += 6.145174 * Q.sd_rg - 0.9506614
    if 0.1881908 <= Q.sd_rg < 0.2639816:
        z += -2.715419 * Q.sd_rg + 0.7168206
    if Q.tau21_b2 < 0.2352054 and Q.mass_over_sum_pt_sq < 0.007873266:
        z += -882.7505 * (0.2352054 - Q.tau21_b2) * (0.007873266 - Q.mass_over_sum_pt_sq)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9638082:
        z += 9.832696 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9777125:
        z += 7.870652 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 82.85409 and Q.psi_0p3 < 0.9985421:
        z += 2.125923 * (82.85409 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9777125:
        z += -16.28366 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9777125:
        z += 7.807892 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9980008:
        z += -104.1623 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9980008:
        z += 27.23535 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9980008:
        z += 75.5073 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 1260.541:
        z += -0.01475493 * (0.2352054 - Q.tau21_b2) * (1260.541 - Q.sum_pt)
    if Q.mass < 92.85979 and Q.psi_0p3 > 0.9638082:
        z += -8.249051 * (92.85979 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.mass < 92.85979 and Q.z_dr_0_0p05 < 0.4947602:
        z += -0.09388664 * (92.85979 - Q.mass) * (0.4947602 - Q.z_dr_0_0p05)
    if Q.lam1 < 0.008241985 and Q.zdr_0 > 0.01564747:
        z += -24939.02 * (0.008241985 - Q.lam1) * (Q.zdr_0 - 0.01564747)
    if Q.psi_0p3 > 0.9973959 and Q.z_top50_slots > 0.978741:
        z += 8419.233 * (Q.psi_0p3 - 0.9973959) * (Q.z_top50_slots - 0.978741)
    return max(0.0, z)


def neuron_8(Q):
    z = 1.850461
    if 0.07696632 <= Q.mass_over_sum_pt < 0.1182259:
        z += -20.59952 * Q.mass_over_sum_pt + 1.585469
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -36.93264 * Q.mass_over_sum_pt + 3.516467
    if Q.sum_pt_top40 < 1001.523:
        z += 0.007379816 * Q.sum_pt_top40 - 7.603131
    if 1001.523 <= Q.sum_pt_top40 < 1024.942:
        z += 0.009055537 * Q.sum_pt_top40 - 9.281404
    if Q.girth2_top40 < 0.005196966:
        z += -171.5075 * Q.girth2_top40 + 1.516219
    if 0.005196966 <= Q.girth2_top40 < 0.008840538:
        z += -122.4309 * Q.girth2_top40 + 1.261169
    if Q.girth2_top40 >= 0.008840538:
        z += 49.07666 * Q.girth2_top40 - 0.2550497
    if Q.n_dr_0p2_0p4 < 3.0:
        z += -0.006836131 * Q.n_dr_0p2_0p4 - 0.1732794
    if 3.0 <= Q.n_dr_0p2_0p4 < 9.0:
        z += 0.02613789 * Q.n_dr_0p2_0p4 - 0.2722015
    if 9.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += 0.07210812 * Q.n_dr_0p2_0p4 - 0.6859336
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += 0.03297402 * Q.n_dr_0p2_0p4 - 0.09892206
    if 64.48544 <= Q.mass < 74.25181:
        z += 0.02053202 * Q.mass - 1.324016
    if 74.25181 <= Q.mass < 87.36377:
        z += 0.05793791 * Q.mass - 4.101471
    if 87.36377 <= Q.mass < 101.0497:
        z += 0.1071562 * Q.mass - 8.401367
    if 101.0497 <= Q.mass < 125.1:
        z += 0.0489114 * Q.mass - 2.515746
    if Q.mass >= 125.1:
        z += 0.01957966 * Q.mass + 1.153654
    if Q.sum_pt < 1017.435:
        z += -0.03563302 * Q.sum_pt + 36.43313
    if 1017.435 <= Q.sum_pt < 1028.184:
        z += -0.01663926 * Q.sum_pt + 17.10822
    if 0.006043209 <= Q.girth2_top20 < 0.008031209:
        z += -132.6691 * Q.girth2_top20 + 0.8017469
    if Q.girth2_top20 >= 0.008031209:
        z += 83.94106 * Q.girth2_top20 - 0.9378944
    if Q.log_sum_pt < 6.930088:
        z += 14.69319 * Q.log_sum_pt - 101.8251
    if Q.girth2_top30 < 0.007463985:
        z += -142.932 * Q.girth2_top30 + 1.066843
    if Q.width < 0.009614971:
        z += 388.5343 * Q.width - 3.735746
    if Q.girth2 < 0.007877041:
        z += 2.213408 * Q.girth2 - 0.0174351
    if Q.sj2_dr >= 0.2232169:
        z += 4.696469 * Q.sj2_dr - 1.048331
    if Q.z_dr_0p1_0p2 >= 0.3340477:
        z += 1.026533 * Q.z_dr_0p1_0p2 - 0.3429112
    if Q.n_for_90pct < 7.0:
        z += 0.04154969 * Q.n_for_90pct - 1.620438
    if 7.0 <= Q.n_for_90pct < 39.0:
        z += -0.007160012 * Q.n_for_90pct - 1.27947
    if Q.n_for_90pct >= 39.0:
        z += -0.04870971 * Q.n_for_90pct + 0.3409679
    if Q.e2 >= 0.04755309:
        z += 57.21401 * Q.e2 - 2.720703
    if Q.lam1 < 0.007671243:
        z += -79.32868 * Q.lam1 + 0.6085496
    if Q.C2 >= 0.06655881:
        z += 4.162679 * Q.C2 - 0.2770629
    if 5.13841e-05 <= Q.e3 < 0.0003372339:
        z += -2923.142 * Q.e3 + 0.150203
    if Q.e3 >= 0.0003372339:
        z += -1355.536 * Q.e3 - 0.3784467
    if 82.04491 <= Q.mass_top50 < 117.0487:
        z += -0.04173379 * Q.mass_top50 + 3.424045
    if Q.mass_top50 >= 117.0487:
        z += -0.02088325 * Q.mass_top50 + 0.9835162
    if Q.mass_top20 >= 119.2969:
        z += -0.01506448 * Q.mass_top20 + 1.797145
    if Q.n_dr_0p1_0p2 >= 19.0:
        z += 0.02292944 * Q.n_dr_0p1_0p2 - 0.4356594
    if Q.z_top15_slots < 0.8316924:
        z += -1.19996 * Q.z_top15_slots + 0.9979974
    if Q.psi_0p1 < 0.3628388:
        z += 2.011365 * Q.psi_0p1 - 0.7298014
    if Q.z_top50_slots >= 0.9704436:
        z += -20.86305 * Q.z_top50_slots + 20.24641
    if Q.girth2_top15 < 0.02146578:
        z += 9.138101 * Q.girth2_top15 - 0.1961565
    if Q.sum_pt_top30 >= 978.0762:
        z += 0.001567005 * Q.sum_pt_top30 - 1.532651
    if Q.sum_pt_top50 >= 889.8503:
        z += -0.0005110931 * Q.sum_pt_top50 + 0.4547963
    if Q.mass_over_sum_pt > 0.07696632 and Q.sum_pt < 1115.723:
        z += 0.3071601 * (Q.mass_over_sum_pt - 0.07696632) * (1115.723 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.z_dr_0p1_0p2 > 0.3340477:
        z += 0.1449424 * (15.0 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p1_0p2 - 0.3340477)
    if Q.girth2_top40 > 0.005196966 and Q.log_sum_pt < 7.017258:
        z += -715.0705 * (Q.girth2_top40 - 0.005196966) * (7.017258 - Q.log_sum_pt)
    if Q.mass > 64.48544 and Q.zdr_0 > 0.0008718296:
        z += -0.1396009 * (Q.mass - 64.48544) * (Q.zdr_0 - 0.0008718296)
    if Q.girth2_top30 < 0.007463985 and Q.sj2_dr > 0.1512157:
        z += -578.2375 * (0.007463985 - Q.girth2_top30) * (Q.sj2_dr - 0.1512157)
    if Q.tau2 < 0.0795038 and Q.sj3_mass1 > 13.38202:
        z += 0.3991667 * (0.0795038 - Q.tau2) * (Q.sj3_mass1 - 13.38202)
    if Q.sum_pt < 1017.435 and Q.dr_max_012 > 0.1828389:
        z += -0.2178795 * (1017.435 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    if Q.sum_pt < 1028.184 and Q.dr_max_012 > 0.1828389:
        z += 0.1961136 * (1028.184 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.007784
    if Q.mass_top40 < 80.89043:
        z += -0.001040417 * Q.mass_top40 - 1.873902
    if 80.89043 <= Q.mass_top40 < 125.1:
        z += 0.02154081 * Q.mass_top40 - 3.700507
    if 125.1 <= Q.mass_top40 < 163.2541:
        z += 0.02636024 * Q.mass_top40 - 4.303417
    if Q.sum_pt_top40 < 956.2133:
        z += -0.003532956 * Q.sum_pt_top40 + 3.378259
    if Q.girth2_top40 < 0.00217213:
        z += 62.207 * Q.girth2_top40 + 0.6685449
    if 0.00217213 <= Q.girth2_top40 < 0.006259772:
        z += -196.6088 * Q.girth2_top40 + 1.230726
    if Q.mass < 64.48544:
        z += -0.01894544 * Q.mass + 2.972696
    if 64.48544 <= Q.mass < 82.85409:
        z += -0.07890629 * Q.mass + 6.839298
    if 82.85409 <= Q.mass < 92.85979:
        z += -0.02442601 * Q.mass + 2.325384
    if 92.85979 <= Q.mass < 143.7876:
        z += 0.0408577 * Q.mass - 3.736847
    if 143.7876 <= Q.mass < 160.8:
        z += -0.05413895 * Q.mass + 9.922494
    if 160.8 <= Q.mass < 162.8363:
        z += -0.209507 * Q.mass + 34.90567
    if 162.8363 <= Q.mass < 172.8:
        z += -0.07932044 * Q.mass + 13.70657
    if 0.09749958 <= Q.girth < 0.1402186:
        z += 24.59251 * Q.girth - 2.39776
    if Q.girth >= 0.1402186:
        z += -50.00882 * Q.girth + 8.062734
    if Q.tau1 < 0.05444509:
        z += 5.60108 * Q.tau1 - 0.3049513
    if Q.lam1 < 0.004673423:
        z += -84.06688 * Q.lam1 + 0.3928801
    if Q.lam1 >= 0.01174405:
        z += 55.92495 * Q.lam1 - 0.6567855
    if Q.girth2_top30 >= 0.0008564881:
        z += -24.58855 * Q.girth2_top30 + 0.0210598
    if Q.D2 >= 2.178951:
        z += 0.0258546 * Q.D2 - 0.0563359
    if Q.mass_top50 < 92.16545:
        z += -0.009482011 * Q.mass_top50 + 3.3245
    if 92.16545 <= Q.mass_top50 < 138.8977:
        z += -0.0524389 * Q.mass_top50 + 7.283641
    if Q.mass_top30 < 73.33139:
        z += -0.00549358 * Q.mass_top30 + 0.4028518
    if Q.LHA >= 0.404204:
        z += 39.26635 * Q.LHA - 15.87161
    if Q.z_top40_slots >= 0.9574183:
        z += -8.289421 * Q.z_top40_slots + 7.936444
    if Q.sum_pt < 986.0565:
        z += -0.02162692 * Q.sum_pt + 21.32536
    if Q.log_sum_pt < 6.903423:
        z += 14.34146 * Q.log_sum_pt - 99.00519
    if Q.sum_pt_top40 < 956.2133 and Q.soft4_pt > 1.789258:
        z += 0.002381269 * (956.2133 - Q.sum_pt_top40) * (Q.soft4_pt - 1.789258)
    if Q.mass_top40 < 80.89043 and Q.sum_pt < 1034.834:
        z += -0.000202236 * (80.89043 - Q.mass_top40) * (1034.834 - Q.sum_pt)
    if Q.mass_top40 < 80.89043 and Q.sum_pt_top40 < 906.6023:
        z += 0.0002198253 * (80.89043 - Q.mass_top40) * (906.6023 - Q.sum_pt_top40)
    if Q.sum_pt_top40 < 956.2133 and Q.soft3_pt > 2.873047:
        z += -0.005355553 * (956.2133 - Q.sum_pt_top40) * (Q.soft3_pt - 2.873047)
    if Q.mass_top40 < 125.1 and Q.n_dr_0p2_0p4 > 9.0:
        z += 0.001027823 * (125.1 - Q.mass_top40) * (Q.n_dr_0p2_0p4 - 9.0)
    return max(0.0, z)


def neuron_10(Q):
    z = 0.648597
    if Q.girth < 0.1207452:
        z += 38.20484 * Q.girth - 4.61305
    if Q.mass < 64.48544:
        z += -0.05192362 * Q.mass + 4.185016
    if 64.48544 <= Q.mass < 78.26182:
        z += -0.09424443 * Q.mass + 6.914092
    if 78.26182 <= Q.mass < 86.4:
        z += -0.03330902 * Q.mass + 2.145177
    if 86.4 <= Q.mass < 89.74183:
        z += 0.0009089857 * Q.mass - 0.8112594
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.05048212 * Q.mass - 5.260043
    if 101.0497 <= Q.mass < 143.7876:
        z += 0.0186146 * Q.mass - 2.03984
    if 143.7876 <= Q.mass < 162.8363:
        z += -0.06626794 * Q.mass + 10.16522
    if Q.mass >= 162.8363:
        z += -0.131215 * Q.mass + 20.74096
    if Q.girth2_top30 < 0.005402331:
        z += 120.8329 * Q.girth2_top30 - 0.6527791
    if 136.785 <= Q.mass_top50 < 160.8:
        z += 0.04549703 * Q.mass_top50 - 6.223311
    if 160.8 <= Q.mass_top50 < 172.8:
        z += 0.06589169 * Q.mass_top50 - 9.502773
    if Q.mass_top50 >= 172.8:
        z += 0.04051546 * Q.mass_top50 - 5.11776
    if Q.sj2_mass1 < 65.20727:
        z += 0.02018381 * Q.sj2_mass1 - 1.316131
    if Q.D2 < 2.975532:
        z += -0.4001279 * Q.D2 + 1.190593
    if 0.9980008 <= Q.psi_0p3 < 0.9985421:
        z += -583.0591 * Q.psi_0p3 + 581.8935
    if Q.psi_0p3 >= 0.9985421:
        z += 159.5424 * Q.psi_0p3 - 159.6254
    if Q.sum_pt_top50 < 959.0957:
        z += 0.003044629 * Q.sum_pt_top50 - 2.92009
    if Q.dr_0 < 0.06413297:
        z += -6.96143 * Q.dr_0 + 0.4464572
    if Q.z_dr_0_0p05 >= 0.7674734:
        z += -2.103335 * Q.z_dr_0_0p05 + 1.614254
    if Q.e2 < 0.01256572:
        z += -33.52113 * Q.e2 + 0.06314604
    if 0.01256572 <= Q.e2 < 0.01541561:
        z += 7.971493 * Q.e2 - 0.4582388
    if 0.01541561 <= Q.e2 < 0.03263075:
        z += 57.59461 * Q.e2 - 1.223209
    if 0.03263075 <= Q.e2 < 0.04358622:
        z += 42.19919 * Q.e2 - 0.7208451
    if Q.e2 >= 0.04358622:
        z += 26.0972 * Q.e2 - 0.01902059
    if Q.sum_pt_top10 >= 943.6922:
        z += -0.003775478 * Q.sum_pt_top10 + 3.562889
    if Q.mass_over_sum_pt >= 0.1606361:
        z += -71.1562 * Q.mass_over_sum_pt + 11.43026
    if Q.zdr_1 < 0.008824206:
        z += -34.28765 * Q.zdr_1 + 0.3025613
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.04689392 * Q.n_dr_0p2_0p4 - 0.5158331
    if Q.sum_pt_top15 < 935.1043:
        z += -0.00442156 * Q.sum_pt_top15 + 4.13462
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -4.097284 * Q.z_dr_0p2_0p4 + 0.3737775
    if Q.mass_top15 < 72.18744:
        z += 0.008394985 * Q.mass_top15 - 0.6060125
    if Q.girth2_top20 < 0.005312783:
        z += 61.73893 * Q.girth2_top20 + 0.08171683
    if 0.005312783 <= Q.girth2_top20 < 0.01655983:
        z += -36.42932 * Q.girth2_top20 + 0.6032634
    if Q.mass_top5 >= 22.18342:
        z += 0.01401741 * Q.mass_top5 - 0.310954
    if 78.53034 <= Q.mass_top30 < 89.17293:
        z += -0.02283107 * Q.mass_top30 + 1.792932
    if Q.mass_top30 >= 89.17293:
        z += -0.002804797 * Q.mass_top30 + 0.007130189
    if Q.tau1 < 0.04466492:
        z += 36.54777 * Q.tau1 + 1.590405
    if 0.04466492 <= Q.tau1 < 0.1507173:
        z += -30.38884 * Q.tau1 + 4.580123
    if Q.pt_dispersion >= 0.27462:
        z += -2.679901 * Q.pt_dispersion + 0.7359545
    if Q.tau2 < 0.04828819:
        z += -16.24071 * Q.tau2 + 0.7842347
    if Q.sum_pt < 986.0565:
        z += 0.008131129 * Q.sum_pt - 8.017753
    if Q.girth2_top40 >= 0.02497133:
        z += 119.8379 * Q.girth2_top40 - 2.992511
    if Q.mass_top50 > 160.8 and Q.soft4_z > 0.001721109:
        z += -18.50653 * (Q.mass_top50 - 160.8) * (Q.soft4_z - 0.001721109)
    if Q.D2 < 2.975532 and Q.sj2_dr > 0.2070855:
        z += -3.121215 * (2.975532 - Q.D2) * (Q.sj2_dr - 0.2070855)
    if Q.girth2_top30 < 0.005402331 and Q.psi_0p3 > 0.9985421:
        z += 125630.8 * (0.005402331 - Q.girth2_top30) * (Q.psi_0p3 - 0.9985421)
    if Q.girth2_top10 < 0.01976735 and Q.psi_0p3 > 0.9985421:
        z += -21445.51 * (0.01976735 - Q.girth2_top10) * (Q.psi_0p3 - 0.9985421)
    if Q.girth2_top10 < 0.01976735 and Q.max_dr < 0.4357228:
        z += -105.6553 * (0.01976735 - Q.girth2_top10) * (0.4357228 - Q.max_dr)
    if Q.mass_top5 < 59.40777 and Q.z_dr_0p05_0p1 < 0.8509215:
        z += 0.001537866 * (59.40777 - Q.mass_top5) * (0.8509215 - Q.z_dr_0p05_0p1)
    if Q.mass_top50 > 136.785 and Q.soft5_z > 0.001594761:
        z += -14.94137 * (Q.mass_top50 - 136.785) * (Q.soft5_z - 0.001594761)
    if Q.mass > 162.8363 and Q.soft5_z > 0.001434897:
        z += 17.86242 * (Q.mass - 162.8363) * (Q.soft5_z - 0.001434897)
    if Q.sum_pt_top10 > 943.6922 and Q.eta_0 < -0.07794189:
        z += -0.9066048 * (Q.sum_pt_top10 - 943.6922) * (-0.07794189 - Q.eta_0)
    if Q.sj2_mass1 < 65.20727 and Q.sj2_mass2 > 7.769597:
        z += 0.0003010937 * (65.20727 - Q.sj2_mass1) * (Q.sj2_mass2 - 7.769597)
    if Q.e2 < 0.01541561 and Q.pt2_over_pt0 < 0.7904923:
        z += -86.52511 * (0.01541561 - Q.e2) * (0.7904923 - Q.pt2_over_pt0)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.3849831
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.1555095 * Q.n_dr_0p2_0p4 + 1.180952
    if 6.0 <= Q.n_dr_0p2_0p4 < 10.0:
        z += -0.06197368 * Q.n_dr_0p2_0p4 + 0.6197368
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += -90.35345 * Q.mass_over_sum_pt_sq + 0.8669427
    if Q.mass < 64.48544:
        z += -0.04590649 * Q.mass + 4.712285
    if 64.48544 <= Q.mass < 80.4:
        z += -0.05113527 * Q.mass + 5.049466
    if 80.4 <= Q.mass < 92.85979:
        z += -0.07529737 * Q.mass + 6.992098
    if Q.girth2_top10 < 0.00406126:
        z += -87.02335 * Q.girth2_top10 + 0.06452707
    if 0.00406126 <= Q.girth2_top10 < 0.00625621:
        z += 131.6191 * Q.girth2_top10 - 0.8234366
    if Q.e2 < 0.02515919:
        z += -94.07705 * Q.e2 + 2.566219
    if 0.02515919 <= Q.e2 < 0.03029714:
        z += -48.33225 * Q.e2 + 1.415317
    if 0.03029714 <= Q.e2 < 0.03875945:
        z += 5.791841 * Q.e2 - 0.2244886
    if Q.girth2_top30 < 0.005402331:
        z += 132.903 * Q.girth2_top30 - 0.9699459
    if 0.005402331 <= Q.girth2_top30 < 0.006929741:
        z += 164.9589 * Q.girth2_top30 - 1.143122
    if Q.mass_top30 < 60.43821:
        z += -0.02873823 * Q.mass_top30 + 1.564176
    if 60.43821 <= Q.mass_top30 < 86.4:
        z += 0.006652525 * Q.mass_top30 - 0.5747782
    if 2.178951 <= Q.D2 < 3.814159:
        z += 0.08279106 * Q.D2 - 0.1803976
    if Q.D2 >= 3.814159:
        z += 0.002339825 * Q.D2 + 0.1264562
    if Q.mass_top40 < 83.32554:
        z += 0.01870696 * Q.mass_top40 - 1.558768
    if Q.girth < 0.07374472:
        z += 15.53785 * Q.girth - 1.043701
    if 0.07374472 <= Q.girth < 0.076787:
        z += 20.81182 * Q.girth - 1.432628
    if 0.076787 <= Q.girth < 0.08589404:
        z += -18.16713 * Q.girth + 1.560449
    if Q.e2_sq < 0.00818374:
        z += -174.1925 * Q.e2_sq + 1.425546
    if Q.mass_over_sum_pt < 0.06030419:
        z += 29.45052 * Q.mass_over_sum_pt - 2.194676
    if 0.06030419 <= Q.mass_over_sum_pt < 0.07999061:
        z += 21.26776 * Q.mass_over_sum_pt - 1.701221
    if Q.lam1 < 0.006716737:
        z += 231.4992 * Q.lam1 - 1.554919
    if Q.e3 < 3.793233e-05:
        z += 10418.32 * Q.e3 - 0.3951913
    if Q.psi_0p2 >= 0.9313699:
        z += -5.832568 * Q.psi_0p2 + 5.432278
    if Q.mass_top50 < 80.35535:
        z += 0.02493408 * Q.mass_top50 - 1.419339
    if 80.35535 <= Q.mass_top50 < 97.93004:
        z += -0.03324367 * Q.mass_top50 + 3.255554
    if Q.psi_0p3 >= 0.9896594:
        z += 52.90942 * Q.psi_0p3 - 52.36231
    if Q.n_dr_0p1_0p2 < 15.0:
        z += -0.04634103 * Q.n_dr_0p1_0p2 + 0.6951155
    if Q.z_top5_slots >= 0.534626:
        z += -1.350398 * Q.z_top5_slots + 0.7219578
    if Q.girth2_top20 < 0.008031209:
        z += 47.71101 * Q.girth2_top20 - 0.3831771
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_particles > 34.0:
        z += -0.00312195 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_particles - 34.0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2 < 0.005532208:
        z += -25.91541 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.girth2)
    if Q.mass < 101.0497 and Q.sum_pt_top10 < 891.875:
        z += -4.964193e-05 * (101.0497 - Q.mass) * (891.875 - Q.sum_pt_top10)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.phi_0 < 0.0402832:
        z += 0.3384107 * (10.0 - Q.n_dr_0p2_0p4) * (0.0402832 - Q.phi_0)
    if Q.mass_top50 < 80.35535 and Q.zdr_4 < 0.003932029:
        z += 3.817871 * (80.35535 - Q.mass_top50) * (0.003932029 - Q.zdr_4)
    if Q.mass < 92.85979 and Q.zdr_4 < 0.004495205:
        z += -2.30207 * (92.85979 - Q.mass) * (0.004495205 - Q.zdr_4)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.2314407
    if Q.mass < 53.87362:
        z += -0.02787369 * Q.mass + 4.092193
    if 53.87362 <= Q.mass < 62.55:
        z += -0.06098371 * Q.mass + 5.875949
    if 62.55 <= Q.mass < 74.25181:
        z += -0.1235184 * Q.mass + 9.787493
    if 74.25181 <= Q.mass < 82.85409:
        z += -0.07482799 * Q.mass + 6.172143
    if 82.85409 <= Q.mass < 86.4:
        z += 0.007800973 * Q.mass - 0.6740041
    if Q.sd_mass >= 125.1:
        z += 0.02148033 * Q.sd_mass - 2.68719
    if Q.sj3_pair_mass_max >= 120.6:
        z += -0.01591344 * Q.sj3_pair_mass_max + 1.919161
    if Q.mass_top40 < 74.78616:
        z += 0.01659411 * Q.mass_top40 - 1.24101
    if Q.log_sum_pt < 6.856375:
        z += 3.547257 * Q.log_sum_pt - 24.32132
    if Q.girth2_top15 < 0.006142802:
        z += 120.085 * Q.girth2_top15 - 0.7376584
    if Q.psi_0p1 >= 0.9538343:
        z += 9.434484 * Q.psi_0p1 - 8.998934
    if Q.mass_over_sum_pt < 0.06895248:
        z += 24.9974 * Q.mass_over_sum_pt - 1.723633
    if Q.mass < 86.4 and Q.z_dr_0p1_0p2 < 0.06472584:
        z += -0.2637652 * (86.4 - Q.mass) * (0.06472584 - Q.z_dr_0p1_0p2)
    if Q.mass_top40 < 89.6788 and Q.n_dr_0p05_0p1 > 1.0:
        z += 0.0005106764 * (89.6788 - Q.mass_top40) * (Q.n_dr_0p05_0p1 - 1.0)
    if Q.mass < 86.4 and Q.psi_0p3 < 0.9985421:
        z += -3.808108 * (86.4 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.sj3_pair_mass_max > 120.6 and Q.sj3_pair_mass_min < 76.60223:
        z += 0.0004453983 * (Q.sj3_pair_mass_max - 120.6) * (76.60223 - Q.sj3_pair_mass_min)
    if Q.sj3_pair_mass_max > 120.6 and Q.z_dr_0p1_0p2 > 0.2864926:
        z += 0.03008282 * (Q.sj3_pair_mass_max - 120.6) * (Q.z_dr_0p1_0p2 - 0.2864926)
    if Q.mass < 86.4 and Q.lam2 < 0.003687605:
        z += 10.98316 * (86.4 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.girth2_top10 < 0.0007431905 and Q.sum_pt_top40 < 1069.671:
        z += 4.229545 * (0.0007431905 - Q.girth2_top10) * (1069.671 - Q.sum_pt_top40)
    if Q.mass < 86.4 and Q.z_top50_slots < 0.985099:
        z += -1.387041 * (86.4 - Q.mass) * (0.985099 - Q.z_top50_slots)
    if Q.log_sum_pt < 6.856375 and Q.lam2 < 0.002396991:
        z += -4858.64 * (6.856375 - Q.log_sum_pt) * (0.002396991 - Q.lam2)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.028002
    if Q.sum_pt < 1007.788:
        z += 0.012491 * Q.sum_pt - 12.86906
    if 1007.788 <= Q.sum_pt < 1034.834:
        z += 0.007265337 * Q.sum_pt - 7.602689
    if 1034.834 <= Q.sum_pt < 1085.125:
        z += 0.001675654 * Q.sum_pt - 1.818294
    if 74.25181 <= Q.mass < 101.0497:
        z += 0.01372741 * Q.mass - 1.019285
    if 101.0497 <= Q.mass < 136.785:
        z += -0.005179139 * Q.mass + 0.8912164
    if 136.785 <= Q.mass < 143.7876:
        z += -0.01204584 * Q.mass + 1.830478
    if 143.7876 <= Q.mass < 160.8:
        z += -0.09986849 * Q.mass + 14.45829
    if 160.8 <= Q.mass < 162.8363:
        z += -0.1800721 * Q.mass + 27.35502
    if 162.8363 <= Q.mass < 172.8:
        z += -0.1860316 * Q.mass + 28.32545
    if Q.mass >= 172.8:
        z += -0.2968018 * Q.mass + 47.46654
    if Q.n_for_90pct < 16.0:
        z += 0.07479743 * Q.n_for_90pct - 1.196759
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 289.5766 * Q.mass_over_sum_pt - 49.48288
    if Q.sum_pt_top40 < 1007.44:
        z += -0.009996525 * Q.sum_pt_top40 + 10.25776
    if 1007.44 <= Q.sum_pt_top40 < 1053.047:
        z += -0.004097187 * Q.sum_pt_top40 + 4.314532
    if 27.46086 <= Q.mass_top50 < 92.16545:
        z += 0.005933418 * Q.mass_top50 - 0.1629368
    if 92.16545 <= Q.mass_top50 < 136.785:
        z += 0.0005386458 * Q.mass_top50 + 0.3342748
    if 136.785 <= Q.mass_top50 < 168.9698:
        z += 0.05212922 * Q.mass_top50 - 6.722541
    if Q.mass_top50 >= 168.9698:
        z += 0.1205955 * Q.mass_top50 - 18.29127
    if Q.girth2_top50 < 0.007820315:
        z += -98.13543 * Q.girth2_top50 + 1.406451
    if 0.007820315 <= Q.girth2_top50 < 0.02550569:
        z += -36.1316 * Q.girth2_top50 + 0.9215614
    if Q.mass_over_sum_pt_sq < 0.02580396:
        z += 63.1768 * Q.mass_over_sum_pt_sq - 1.630212
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -664.8962 * Q.mass_over_sum_pt_sq + 19.41498
    z += 0.009600775 * Q.n_particles
    if Q.log_sum_pt < 6.910131:
        z += 20.18632 * Q.log_sum_pt - 139.4901
    if Q.mass_top40 < 160.8:
        z += 0.007350404 * Q.mass_top40 - 1.181945
    if Q.sum_pt_top50 < 997.0189:
        z += -0.005269146 * Q.sum_pt_top50 + 5.253438
    if Q.sum_pt_top30 < 966.0633:
        z += 0.004329449 * Q.sum_pt_top30 - 4.182521
    if Q.mass_top5 >= 33.71058:
        z += 0.01381765 * Q.mass_top5 - 0.4658011
    if Q.sum_pt_top20 >= 956.5062:
        z += 0.001177933 * Q.sum_pt_top20 - 1.1267
    if Q.mass_top10 >= 23.38894:
        z += -0.007796178 * Q.mass_top10 + 0.1823443
    if Q.mass_top30 >= 138.3818:
        z += 0.02730165 * Q.mass_top30 - 3.778053
    if Q.sum_pt < 1085.125 and Q.tau21_b2 < 0.8310045:
        z += -0.007679465 * (1085.125 - Q.sum_pt) * (0.8310045 - Q.tau21_b2)
    if Q.sum_pt_top40 < 1053.047 and Q.sj2_zsoft < 0.2179035:
        z += 0.01667821 * (1053.047 - Q.sum_pt_top40) * (0.2179035 - Q.sj2_zsoft)
    if Q.mass > 74.25181 and Q.sum_pt < 1028.184:
        z += 3.630002e-05 * (Q.mass - 74.25181) * (1028.184 - Q.sum_pt)
    if Q.log_sum_pt > 7.139296 and Q.C3 < 0.00317537:
        z += -1241.082 * (Q.log_sum_pt - 7.139296) * (0.00317537 - Q.C3)
    if Q.log_sum_pt < 6.811175 and Q.dr_13 < 0.1312677:
        z += 98.16882 * (6.811175 - Q.log_sum_pt) * (0.1312677 - Q.dr_13)
    if Q.mass > 172.8 and Q.pt_6 < 56.53125:
        z += 0.002039107 * (Q.mass - 172.8) * (56.53125 - Q.pt_6)
    if Q.mass > 172.8 and Q.z_6 < 0.04568661:
        z += -3.204985 * (Q.mass - 172.8) * (0.04568661 - Q.z_6)
    if Q.log_sum_pt > 7.139296 and Q.sd_zg > 0.4462823:
        z += 1126.014 * (Q.log_sum_pt - 7.139296) * (Q.sd_zg - 0.4462823)
    if Q.log_sum_pt < 6.811175 and Q.zdr_11 < 0.005041702:
        z += -1082.634 * (6.811175 - Q.log_sum_pt) * (0.005041702 - Q.zdr_11)
    if Q.mass > 162.8363 and Q.soft6_z > 0.0005748372:
        z += -34.14614 * (Q.mass - 162.8363) * (Q.soft6_z - 0.0005748372)
    if Q.mass_top50 > 136.785 and Q.soft6_z > 0.0008188601:
        z += -22.64756 * (Q.mass_top50 - 136.785) * (Q.soft6_z - 0.0008188601)
    if Q.mass > 172.8 and Q.soft5_z > 0.0004140594:
        z += -173.3429 * (Q.mass - 172.8) * (Q.soft5_z - 0.0004140594)
    if Q.mass_top50 > 92.16545 and Q.soft7_z < 0.001923089:
        z += -10.34872 * (Q.mass_top50 - 92.16545) * (0.001923089 - Q.soft7_z)
    if Q.mass > 172.8 and Q.soft6_z > 0.0004464147:
        z += 33.79255 * (Q.mass - 172.8) * (Q.soft6_z - 0.0004464147)
    if Q.girth2_top50 < 0.02550569 and Q.psi_0p3 > 0.9638082:
        z += 840.3533 * (0.02550569 - Q.girth2_top50) * (Q.psi_0p3 - 0.9638082)
    if Q.log_sum_pt < 6.811175 and Q.D2 < 2.680253:
        z += 7.910302 * (6.811175 - Q.log_sum_pt) * (2.680253 - Q.D2)
    if Q.mass > 160.8 and Q.D2 > 0.8942376:
        z += 0.01193729 * (Q.mass - 160.8) * (Q.D2 - 0.8942376)
    if Q.mass > 136.785 and Q.D2 > 0.8942376:
        z += -0.01326679 * (Q.mass - 136.785) * (Q.D2 - 0.8942376)
    if Q.mass > 172.8 and Q.D2 > 0.8942376:
        z += 0.06618927 * (Q.mass - 172.8) * (Q.D2 - 0.8942376)
    if Q.n_for_90pct < 16.0 and Q.D2 < 9.676985:
        z += 0.007621655 * (16.0 - Q.n_for_90pct) * (9.676985 - Q.D2)
    if Q.sum_pt < 1034.834 and Q.pt_9 < 16.64062:
        z += 0.0009109533 * (1034.834 - Q.sum_pt) * (16.64062 - Q.pt_9)
    if Q.log_sum_pt > 7.139296 and Q.dr_9 < 0.04047238:
        z += -129.2479 * (Q.log_sum_pt - 7.139296) * (0.04047238 - Q.dr_9)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.2610399
    if Q.tau21_b2 < 0.342495:
        z += -1.287201 * Q.tau21_b2 + 0.44086
    if Q.mass < 74.25181:
        z += 0.2846299 * Q.mass - 23.83726
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.2915113 * Q.mass - 24.34822
    if 78.26182 <= Q.mass < 80.4:
        z += 0.2955015 * Q.mass - 24.66049
    if 80.4 <= Q.mass < 82.85409:
        z += 0.2587614 * Q.mass - 21.70659
    if 82.85409 <= Q.mass < 89.74183:
        z += 0.09891635 * Q.mass - 8.462777
    if 89.74183 <= Q.mass < 91.19:
        z += 0.04915372 * Q.mass - 3.996987
    if 91.19 <= Q.mass < 125.1:
        z += -0.01431261 * Q.mass + 1.790507
    if Q.psi_0p3 >= 0.9924477:
        z += 139.2516 * Q.psi_0p3 - 138.1999
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.01735225 * Q.n_dr_0p2_0p4 + 0.3123404
    if Q.e2 < 0.01256572:
        z += 7.076191 * Q.e2 + 1.503338
    if 0.01256572 <= Q.e2 < 0.02210818:
        z += -93.7278 * Q.e2 + 2.770013
    if 0.02210818 <= Q.e2 < 0.03029714:
        z += -68.45776 * Q.e2 + 2.211338
    if 0.03029714 <= Q.e2 < 0.04358622:
        z += -10.32907 * Q.e2 + 0.450205
    if Q.mass_over_sum_pt < 0.07852883:
        z += -32.62464 * Q.mass_over_sum_pt + 5.872361
    if 0.07852883 <= Q.mass_over_sum_pt < 0.08873143:
        z += -55.36575 * Q.mass_over_sum_pt + 7.658194
    if 0.08873143 <= Q.mass_over_sum_pt < 0.09046749:
        z += -26.88821 * Q.mass_over_sum_pt + 5.131342
    if 0.09046749 <= Q.mass_over_sum_pt < 0.09795415:
        z += -101.0784 * Q.mass_over_sum_pt + 11.84314
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -62.28385 * Q.mass_over_sum_pt + 8.043055
    if 0.1182259 <= Q.mass_over_sum_pt < 0.140939:
        z += -29.91616 * Q.mass_over_sum_pt + 4.216354
    if Q.girth2_top20 < 0.01083435:
        z += 79.87309 * Q.girth2_top20 - 0.8653732
    if Q.girth < 0.076787:
        z += 12.21832 * Q.girth - 0.9382082
    if Q.girth2_top5 < 0.007164202:
        z += -99.3705 * Q.girth2_top5 + 0.7119103
    if Q.sd_rg < 0.1778185:
        z += -1.968113 * Q.sd_rg + 0.08014323
    if 0.1778185 <= Q.sd_rg < 0.3017146:
        z += 2.177822 * Q.sd_rg - 0.6570807
    if Q.girth2_top30 < 0.002582316:
        z += 3129.089 * Q.girth2_top30 - 9.006064
    if 0.002582316 <= Q.girth2_top30 < 0.01215787:
        z += 96.68039 * Q.girth2_top30 - 1.175428
    if Q.tau1 < 0.06310829:
        z += 15.18351 * Q.tau1 - 1.128394
    if 0.06310829 <= Q.tau1 < 0.1072713:
        z += 3.853655 * Q.tau1 - 0.4133864
    if Q.mass_top30 < 82.66587:
        z += -0.007595192 * Q.mass_top30 + 0.6278631
    if Q.sum_pt_top40 < 906.6023:
        z += 0.003952352 * Q.sum_pt_top40 - 3.22503
    if 906.6023 <= Q.sum_pt_top40 < 1024.942:
        z += -0.003026717 * Q.sum_pt_top40 + 3.10221
    if Q.girth2_top3 < 0.001592178:
        z += 218.0002 * Q.girth2_top3 - 0.347095
    if Q.log_sum_pt < 6.941997:
        z += -5.919099 * Q.log_sum_pt + 40.90559
    if 6.941997 <= Q.log_sum_pt < 6.98945:
        z += 3.893865 * Q.log_sum_pt - 27.21598
    if Q.sum_pt_top50 < 976.277:
        z += 0.003633079 * Q.sum_pt_top50 - 3.546892
    if Q.lam1 < 0.007259287:
        z += -6.856445 * Q.lam1 - 0.241339
    if 0.007259287 <= Q.lam1 < 0.007671243:
        z += 118.307 * Q.lam1 - 1.149936
    if 0.007671243 <= Q.lam1 < 0.008241985:
        z += 424.6656 * Q.lam1 - 3.500088
    if Q.psi_0p2 >= 0.9804031:
        z += -11.49168 * Q.psi_0p2 + 11.26648
    if Q.tau2 < 0.0795038:
        z += 5.145148 * Q.tau2 - 0.4090588
    if Q.mass_top50 < 71.79516:
        z += -0.06010793 * Q.mass_top50 + 3.109745
    if 71.79516 <= Q.mass_top50 < 77.37641:
        z += -0.02873889 * Q.mass_top50 + 0.8575999
    if 77.37641 <= Q.mass_top50 < 85.8667:
        z += 0.04437614 * Q.mass_top50 - 4.799778
    if 85.8667 <= Q.mass_top50 < 97.93004:
        z += 0.05663899 * Q.mass_top50 - 5.852749
    if 97.93004 <= Q.mass_top50 < 117.0487:
        z += 0.01601 * Q.mass_top50 - 1.87395
    if Q.girth2_top40 < 0.01897915:
        z += 44.46814 * Q.girth2_top40 - 0.8439673
    if Q.mass_top40 < 89.6788:
        z += -0.03816171 * Q.mass_top40 + 3.422296
    if Q.N2 < 0.4226723 and Q.max_dr > 0.2404747:
        z += -9.822336 * (0.4226723 - Q.N2) * (Q.max_dr - 0.2404747)
    if Q.psi_0p3 > 0.9924477 and Q.sd_rg > 0.2280025:
        z += -1382.842 * (Q.psi_0p3 - 0.9924477) * (Q.sd_rg - 0.2280025)
    if Q.psi_0p3 > 0.9924477 and Q.sd_rg < 0.1881908:
        z += -524.5438 * (Q.psi_0p3 - 0.9924477) * (0.1881908 - Q.sd_rg)
    if Q.tau21_b2 < 0.342495 and Q.n_real_top40 < 32.0:
        z += -0.2202934 * (0.342495 - Q.tau21_b2) * (32.0 - Q.n_real_top40)
    if Q.mass_over_sum_pt < 0.09046749 and Q.psi_0p2 > 0.9087063:
        z += -187.8899 * (0.09046749 - Q.mass_over_sum_pt) * (Q.psi_0p2 - 0.9087063)
    if Q.psi_0p3 > 0.9924477 and Q.z_2nd < 0.1231747:
        z += -1066.083 * (Q.psi_0p3 - 0.9924477) * (0.1231747 - Q.z_2nd)
    if Q.girth2_top5 < 0.007164202 and Q.n_pt_above_10 > 13.0:
        z += -8.927329 * (0.007164202 - Q.girth2_top5) * (Q.n_pt_above_10 - 13.0)
    if Q.girth < 0.076787 and Q.z_dr_0p2_0p4 < 0.05180474:
        z += -141.6383 * (0.076787 - Q.girth) * (0.05180474 - Q.z_dr_0p2_0p4)
    if Q.mass < 91.19 and Q.n_dr_0p05_0p1 > 14.0:
        z += -0.00242825 * (91.19 - Q.mass) * (Q.n_dr_0p05_0p1 - 14.0)
    if Q.psi_0p2 > 0.9804031 and Q.pt2_over_pt0 > 0.1852611:
        z += 35.37608 * (Q.psi_0p2 - 0.9804031) * (Q.pt2_over_pt0 - 0.1852611)
    if Q.tau21_b2 < 0.342495 and Q.orientation_deg > -18.47737:
        z += -0.01655752 * (0.342495 - Q.tau21_b2) * (Q.orientation_deg - -18.47737)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.4606133
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -4.099748 * Q.z_dr_0p1_0p2 + 0.4933787
    if Q.girth2_top5 < 0.002270363:
        z += 111.8754 * Q.girth2_top5 + 0.5187282
    if 0.002270363 <= Q.girth2_top5 < 0.008329695:
        z += -127.5266 * Q.girth2_top5 + 1.062258
    if Q.psi_0p1 >= 0.8976117:
        z += -1.525285 * Q.psi_0p1 + 1.369114
    if Q.sum_pt < 986.0565:
        z += -0.005297559 * Q.sum_pt + 5.183491
    if 986.0565 <= Q.sum_pt < 1002.379:
        z += 0.00246305 * Q.sum_pt - 2.468909
    if Q.log_sum_pt < 6.903423:
        z += -0.6002111 * Q.log_sum_pt + 5.20225
    if 6.903423 <= Q.log_sum_pt < 7.017258:
        z += -9.300657 * Q.log_sum_pt + 65.26511
    if Q.psi_0p3 >= 0.9896594:
        z += -47.36537 * Q.psi_0p3 + 46.87558
    if 0.1666442 <= Q.sj2_dr < 0.2232169:
        z += -2.652673 * Q.sj2_dr + 0.4420524
    if Q.sj2_dr >= 0.2232169:
        z += 2.758782 * Q.sj2_dr - 0.7658757
    if Q.girth2_top10 < 0.007678544:
        z += -54.57441 * Q.girth2_top10 + 0.3195842
    if 0.007678544 <= Q.girth2_top10 < 0.008956554:
        z += 77.83024 * Q.girth2_top10 - 0.6970907
    if Q.lam1 < 0.004673423:
        z += -33.82198 * Q.lam1 - 0.3818832
    if 0.004673423 <= Q.lam1 < 0.01174405:
        z += 76.36488 * Q.lam1 - 0.896833
    if Q.lam1 >= 0.01649354:
        z += -51.93585 * Q.lam1 + 0.8566063
    if Q.n_dr_0p1_0p2 < 13.0:
        z += -0.03505895 * Q.n_dr_0p1_0p2 + 0.4557663
    if Q.D2 < 1.976207:
        z += -0.2147623 * Q.D2 + 0.4244148
    if Q.lam2 < 0.001776308:
        z += -240.4763 * Q.lam2 + 0.4271602
    if Q.tau1 < 0.0705748:
        z += 15.45768 * Q.tau1 - 1.090923
    if Q.sum_pt_top20 < 1017.778:
        z += 0.002825014 * Q.sum_pt_top20 - 2.875238
    if 9.0 <= Q.n_dr_0p2_0p4 < 26.0:
        z += -0.05068911 * Q.n_dr_0p2_0p4 + 0.456202
    if Q.n_dr_0p2_0p4 >= 26.0:
        z += 0.04233631 * Q.n_dr_0p2_0p4 - 1.962459
    if Q.sd_mass < 45.595:
        z += 0.00233018 * Q.sd_mass + 0.5515202
    if 45.595 <= Q.sd_mass < 79.18312:
        z += -0.01958326 * Q.sd_mass + 1.550663
    if Q.mass_top50 < 71.79516:
        z += 0.01677131 * Q.mass_top50 - 1.204099
    if Q.sum_pt_top40 < 935.8189:
        z += 0.01853009 * Q.sum_pt_top40 - 18.17715
    if 935.8189 <= Q.sum_pt_top40 < 1018.698:
        z += 0.01009117 * Q.sum_pt_top40 - 10.27985
    if Q.sum_pt_top30 < 933.1875:
        z += -0.006305724 * Q.sum_pt_top30 + 5.884423
    if Q.sd_rg < 0.1690338:
        z += -0.04488564 * Q.sd_rg - 0.4796639
    if 0.1690338 <= Q.sd_rg < 0.2042612:
        z += 1.729997 * Q.sd_rg - 0.7796792
    if 0.2042612 <= Q.sd_rg < 0.3017146:
        z += 6.790947 * Q.sd_rg - 1.813435
    if Q.sd_rg >= 0.3017146:
        z += 1.774883 * Q.sd_rg - 0.3000153
    if Q.e2 >= 0.03029714:
        z += -60.00874 * Q.e2 + 1.818093
    if Q.mass_top30 < 80.24626:
        z += -0.01777704 * Q.mass_top30 + 1.58523
    if 80.24626 <= Q.mass_top30 < 89.17293:
        z += -0.0094015 * Q.mass_top30 + 0.913125
    if Q.mass_top30 >= 89.17293:
        z += 0.008375535 * Q.mass_top30 - 0.6721053
    if Q.N2 < 0.4226723:
        z += 1.238306 * Q.N2 - 0.5233978
    if Q.e3 >= 0.0001841806:
        z += -2437.581 * Q.e3 + 0.4489552
    if Q.e2_sq < 0.006399858:
        z += 117.3495 * Q.e2_sq - 0.7510202
    if Q.psi_0p2 >= 0.8706159:
        z += -5.655391 * Q.psi_0p2 + 4.923674
    if Q.max_dr < 0.2982:
        z += 4.719428 * Q.max_dr - 1.407333
    if Q.girth < 0.08589404:
        z += -23.36551 * Q.girth + 2.006958
    if Q.LHA < 0.2941033:
        z += 6.273397 * Q.LHA - 1.845027
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 949.9169:
        z += 0.0102639 * (0.8459004 - Q.z_dr_0_0p05) * (949.9169 - Q.sum_pt)
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 1260.541:
        z += 0.001415534 * (0.8459004 - Q.z_dr_0_0p05) * (1260.541 - Q.sum_pt)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_dr_0p2_0p4 > 5.0:
        z += -0.1989695 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p2_0p4 - 5.0)
    if Q.sj2_dr > 0.2232169 and Q.C2_b2 < 0.04008677:
        z += 121.3933 * (Q.sj2_dr - 0.2232169) * (0.04008677 - Q.C2_b2)
    if Q.girth2_top5 < 0.008329695 and Q.sj3_mass1 > 5.112677:
        z += -1.853855 * (0.008329695 - Q.girth2_top5) * (Q.sj3_mass1 - 5.112677)
    if Q.girth2_top5 < 0.008329695 and Q.sj3_pairmin_over_m > 0.09540583:
        z += -89.88402 * (0.008329695 - Q.girth2_top5) * (Q.sj3_pairmin_over_m - 0.09540583)
    if Q.e2 > 0.03029714 and Q.sj3_z2 < 0.3166869:
        z += 146.0272 * (Q.e2 - 0.03029714) * (0.3166869 - Q.sj3_z2)
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
