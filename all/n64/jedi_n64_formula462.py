"""JEDI-linear jet tagger, 64 particles, 3 features: one term per observable per neuron (from the 690), re-tuned on the network's predictions (all observables), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.0% (the network: 81.1%); same class as the network for 93.1% of jets.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its term in ΣzΔR)
  Q.zdr_1                  pT share × ΔR of particle 1 (its term in ΣzΔR)
  Q.zdr_11                 pT share × ΔR of particle 11 (its term in ΣzΔR)
  Q.zdr_4                  pT share × ΔR of particle 4 (its term in ΣzΔR)
  Q.zdr_5                  pT share × ΔR of particle 5 (its term in ΣzΔR)
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
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -0.622534
    if Q.mass >= 87.3546:
        z += -0.2283854 * Q.mass + 19.95052
    if Q.sum_z_dr2_top20 < 0.008023552:
        z += 4.634469 * Q.sum_z_dr2_top20 - 0.0371849
    if Q.sum_pt < 1012.673:
        z += 0.008203341 * Q.sum_pt - 8.307301
    if Q.psi_0p3 >= 0.9956185:
        z += 40.87828 * Q.psi_0p3 - 40.69917
    if Q.mass_top30 < 80.4:
        z += -0.01191534 * Q.mass_top30 + 0.9579935
    if Q.sum_z_dr2_top30 < 0.00581533:
        z += 155.0342 * Q.sum_z_dr2_top30 - 0.9015751
    if Q.lam1 < 0.005913555:
        z += 166.5415 * Q.lam1 - 0.9848523
    if Q.mass_over_sum_pt_sq < 0.007514722:
        z += -997.2062 * Q.mass_over_sum_pt_sq + 7.493727
    if Q.tau1 < 0.0705748:
        z += 24.05408 * Q.tau1 - 1.697612
    if Q.sum_zz_dr2 < 0.00616708:
        z += 657.4348 * Q.sum_zz_dr2 - 4.054453
    if Q.mass_top50 >= 80.38538:
        z += -0.001150877 * Q.mass_top50 + 0.09251368
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += 5.129581 * Q.z_dr_0p2_0p4 - 0.4679495
    if Q.z_top50_slots < 0.9906378:
        z += 28.55546 * Q.z_top50_slots - 28.28812
    if Q.log_sum_pt < 7.017258:
        z += -2.745442 * Q.log_sum_pt + 19.26547
    if Q.sum_pt_top40 < 1069.671:
        z += 0.004955462 * Q.sum_pt_top40 - 5.300715
    if Q.sum_pt_top20 < 846.1934:
        z += -0.002235153 * Q.sum_pt_top20 + 1.891371
    if Q.mass_top40 < 80.89043:
        z += 0.01065555 * Q.mass_top40 - 0.8619318
    if Q.sum_pt_top50 < 1156.659:
        z += -0.006851177 * Q.sum_pt_top50 + 7.924479
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.04941578 * Q.n_dr_0p2_0p4 + 0.7412367
    if Q.log_sum_pt < 6.98945 and Q.sum_pt_top50 > 959.0957:
        z += 0.02408492 * (6.98945 - Q.log_sum_pt) * (Q.sum_pt_top50 - 959.0957)
    return max(0.0, z)


def neuron_1(Q):
    z = -1.340282
    if Q.n_particles >= 38.0:
        z += 0.08464888 * Q.n_particles - 3.216658
    if Q.log_sum_pt >= 6.811325:
        z += 15.50771 * Q.log_sum_pt - 105.6281
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.003982975 * Q.sum_pt_top50 + 3.820054
    if Q.psi_0p3 >= 0.9980008:
        z += -231.5288 * Q.psi_0p3 + 231.0659
    if Q.sum_pt_top2 < 689.25:
        z += -0.0009061766 * Q.sum_pt_top2 + 0.6245822
    if Q.z_top30_slots >= 0.9341838:
        z += -2.879138 * Q.z_top30_slots + 2.689644
    if Q.mass_top20 < 47.88842:
        z += 0.0483101 * Q.mass_top20 - 2.313494
    if Q.sj3_mass1 < 32.50209:
        z += 0.008439912 * Q.sj3_mass1 - 0.2743147
    if Q.sum_pt_top40 < 1069.671:
        z += 0.005147617 * Q.sum_pt_top40 - 5.506258
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.07647049 * Q.n_dr_0p2_0p4 - 0.5352935
    if Q.M3 < 0.03187688:
        z += 17.51152 * Q.M3 - 0.5582126
    if Q.sj2_mass1 < 30.26161:
        z += 0.01743639 * Q.sj2_mass1 - 0.5276533
    if Q.pt_9 < 31.35938:
        z += 0.03091951 * Q.pt_9 - 0.9696166
    if Q.lam1 < 0.004673423:
        z += -7.597458 * Q.lam1 + 0.03550614
    if Q.mass < 120.6:
        z += 0.02162391 * Q.mass - 2.607843
    if Q.sum_z_dr2_top30 < 0.02412652:
        z += -14.86088 * Q.sum_z_dr2_top30 + 0.3585413
    if Q.D3 < 0.1416054:
        z += -1.527829 * Q.D3 + 0.2163489
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -4.026522 * Q.z_dr_0_0p05 + 3.538935
    if Q.n_dr_0_0p05 < 12.0:
        z += -0.03176161 * Q.n_dr_0_0p05 + 0.3811394
    if Q.mass_top10 >= 56.92192:
        z += 0.00757235 * Q.mass_top10 - 0.4310327
    if Q.soft1_pt >= 1.505859:
        z += 0.3681058 * Q.soft1_pt - 0.5543156
    if Q.sum_pt < 1017.435:
        z += -0.01824968 * Q.sum_pt + 18.56785
    if Q.sum_pt_top30 >= 1191.938:
        z += -0.004053815 * Q.sum_pt_top30 + 4.831895
    if Q.z_top20_slots >= 0.8965411:
        z += 9.244056 * Q.z_top20_slots - 8.287676
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.06207465 * Q.n_dr_0p1_0p2 - 0.4965972
    if Q.pt_entropy >= 2.07371:
        z += 1.396681 * Q.pt_entropy - 2.896311
    if Q.sum_z_dr2_top15 < 0.0007894752:
        z += -881.7345 * Q.sum_z_dr2_top15 + 0.6961076
    if Q.sum_z_dr2_top20 < 0.002830721:
        z += -243.7165 * Q.sum_z_dr2_top20 + 0.6898936
    if Q.psi_0p1 >= 0.8747961:
        z += 1.576294 * Q.psi_0p1 - 1.378935
    if Q.LHA < 0.2601462:
        z += -8.739936 * Q.LHA + 2.273661
    if Q.tau3 < 0.04516808:
        z += -8.075248 * Q.tau3 + 0.3647434
    if Q.e2 < 0.02515919:
        z += 37.14192 * Q.e2 - 0.9344608
    if Q.sum_z_dr2_top5 < 0.0006570502:
        z += -301.9284 * Q.sum_z_dr2_top5 + 0.1983821
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += 141.4798 * Q.mass_over_sum_pt_sq - 1.357501
    if Q.sum_z_dr2_top40 < 0.008031986:
        z += -230.518 * Q.sum_z_dr2_top40 + 1.851518
    if Q.z_top50_slots >= 0.9586536:
        z += -21.40151 * Q.z_top50_slots + 20.51664
    if Q.n_for_90pct >= 11.0:
        z += -0.00880371 * Q.n_for_90pct + 0.09684081
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -3.507148 * Q.z_dr_0p2_0p4 + 0.319942
    if Q.tau1 < 0.1953848:
        z += -13.60413 * Q.tau1 + 2.658041
    if Q.z_top30_slots > 0.9341838 and Q.max_pair_mass > 13.04793:
        z += 0.4415797 * (Q.z_top30_slots - 0.9341838) * (Q.max_pair_mass - 13.04793)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.1119555:
        z += 0.2116112 * (Q.n_particles - 38.0) * (0.1119555 - Q.dr_0)
    if Q.mass_top20 < 47.88842 and Q.n_real_top40 > 29.0:
        z += 0.003111176 * (47.88842 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 38.0 and Q.soft1_pt < 2.275391:
        z += -0.01750232 * (Q.n_particles - 38.0) * (2.275391 - Q.soft1_pt)
    if Q.z_top30_slots > 0.9341838 and Q.C2 < 0.07279889:
        z += 271.8355 * (Q.z_top30_slots - 0.9341838) * (0.07279889 - Q.C2)
    if Q.sj3_mass1 < 32.50209 and Q.sj3_mass2 < 18.68222:
        z += -0.001898931 * (32.50209 - Q.sj3_mass1) * (18.68222 - Q.sj3_mass2)
    if Q.sum_z_dr2_top15 < 0.003270031 and Q.psi_0p3 > 0.9973959:
        z += 53737.07 * (0.003270031 - Q.sum_z_dr2_top15) * (Q.psi_0p3 - 0.9973959)
    if Q.n_particles > 38.0 and Q.dr_1 < 0.1611545:
        z += 0.1111805 * (Q.n_particles - 38.0) * (0.1611545 - Q.dr_1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 > 7.407874:
        z += 0.7260578 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_3 - 7.407874)
    if Q.pt_9 < 31.35938 and Q.dr1_12 < 0.3241858:
        z += 0.04554374 * (31.35938 - Q.pt_9) * (0.3241858 - Q.dr1_12)
    if Q.pt_9 < 31.35938 and Q.pair_mass_0_13 < 11.29505:
        z += 0.001429962 * (31.35938 - Q.pt_9) * (11.29505 - Q.pair_mass_0_13)
    if Q.sum_pt_top5 > 430.75 and Q.eta_0 < 0.02980347:
        z += -0.008228654 * (Q.sum_pt_top5 - 430.75) * (0.02980347 - Q.eta_0)
    if Q.z_top20_slots > 0.8965411 and Q.dr_2 < 0.03843804:
        z += -306.0589 * (Q.z_top20_slots - 0.8965411) * (0.03843804 - Q.dr_2)
    if Q.mass < 120.6 and Q.dr_2 < 0.03843804:
        z += 0.1227397 * (120.6 - Q.mass) * (0.03843804 - Q.dr_2)
    if Q.sum_z_dr2_top20 < 0.007538019 and Q.eta_1 > -0.04302979:
        z += 680.3016 * (0.007538019 - Q.sum_z_dr2_top20) * (Q.eta_1 - -0.04302979)
    if Q.sj2_mass1 < 30.26161 and Q.soft10_dr0 < 0.2366434:
        z += 0.0472099 * (30.26161 - Q.sj2_mass1) * (0.2366434 - Q.soft10_dr0)
    if Q.pt_entropy > 2.07371 and Q.soft3_dr < 0.3797:
        z += -0.7594935 * (Q.pt_entropy - 2.07371) * (0.3797 - Q.soft3_dr)
    if Q.sum_pt_top2 < 689.25 and Q.soft3_dr0 > 0.02918107:
        z += -0.001709661 * (689.25 - Q.sum_pt_top2) * (Q.soft3_dr0 - 0.02918107)
    return max(0.0, z)


def neuron_2(Q):
    z = -8.689107
    if Q.log_sum_pt >= 6.971093:
        z += -13.54818 * Q.log_sum_pt + 94.44561
    z += 0.008227157 * Q.sum_pt
    if Q.sum_pt_top50 >= 1060.151:
        z += -0.0008115673 * Q.sum_pt_top50 + 0.8603841
    if Q.mass < 92.85979:
        z += 0.02000225 * Q.mass - 1.857405
    if Q.mass_over_sum_pt < 0.09795415:
        z += 28.00455 * Q.mass_over_sum_pt - 2.743161
    if Q.sum_pt_top40 < 1018.525:
        z += -0.001360213 * Q.sum_pt_top40 + 1.38541
    if Q.sum_pt_top30 < 1011.14:
        z += 0.001570616 * Q.sum_pt_top30 - 1.588113
    if Q.lam1 < 0.01174405:
        z += -42.00214 * Q.lam1 + 0.4932752
    if Q.mass_top50 < 92.16545:
        z += -0.0394228 * Q.mass_top50 + 3.63342
    if Q.log_sum_pt > 6.903423 and Q.sum_z_dr2_top15 < 0.02146578:
        z += 920.5361 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.sum_z_dr2_top15)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p3 > 0.9299135:
        z += -22.82174 * (Q.log_sum_pt - 6.903423) * (Q.psi_0p3 - 0.9299135)
    if Q.sum_pt_top50 > 988.4554 and Q.sum_z_dr2_top15 < 0.02146578:
        z += -0.3810079 * (Q.sum_pt_top50 - 988.4554) * (0.02146578 - Q.sum_z_dr2_top15)
    if Q.sum_pt < 972.0419 and Q.e4 < 5.8505e-08:
        z += 201783.3 * (972.0419 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.sum_pt_top20 > 1129.275 and Q.eta_1 > 0.08734131:
        z += -0.2420955 * (Q.sum_pt_top20 - 1129.275) * (Q.eta_1 - 0.08734131)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.09610848
    if Q.lam2 < 0.0006154841:
        z += -871.425 * Q.lam2 + 0.5363482
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.08360512 * Q.n_dr_0p2_0p4 + 0.4180256
    if Q.n_particles < 46.0:
        z += -0.02059568 * Q.n_particles + 0.9474011
    if Q.tau21 < 0.3861957:
        z += 2.420707 * Q.tau21 - 0.9348664
    if Q.D2 < 2.410481:
        z += 0.5249224 * Q.D2 - 1.265315
    if Q.mass_over_sum_pt < 0.08665515:
        z += -0.3819317 * Q.mass_over_sum_pt + 0.03309635
    if Q.mass_top50 < 79.21004:
        z += 0.01480261 * Q.mass_top50 - 1.172515
    if Q.mass < 53.52607:
        z += -0.001269282 * Q.mass + 0.06793966
    if Q.sum_z_dr2 < 0.009614971:
        z += 53.2613 * Q.sum_z_dr2 - 0.5121059
    if Q.sum_z_dr2_top40 < 0.008840538:
        z += -126.2361 * Q.sum_z_dr2_top40 + 1.115995
    if Q.sj2_mass1 < 27.56535:
        z += -0.01566024 * Q.sj2_mass1 + 0.43168
    if Q.mass_top40 < 80.4:
        z += 0.01150606 * Q.mass_top40 - 0.9250869
    if Q.tau4 < 0.01626937:
        z += -15.31128 * Q.tau4 + 0.2491049
    if Q.sum_z_dr2_top50 < 0.008124776:
        z += -34.16842 * Q.sum_z_dr2_top50 + 0.2776108
    if Q.n_particles < 46.0 and Q.mass_top15 > 57.87349:
        z += -0.0003187616 * (46.0 - Q.n_particles) * (Q.mass_top15 - 57.87349)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_dr_0p1_0p2 > 9.0:
        z += -0.008008199 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 9.0)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.zdr_5 < 0.007099471:
        z += -12.94875 * (5.0 - Q.n_dr_0p2_0p4) * (0.007099471 - Q.zdr_5)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.psi_0p1 < 0.9538343:
        z += 0.5809597 * (5.0 - Q.n_dr_0p2_0p4) * (0.9538343 - Q.psi_0p1)
    if Q.tau21 < 0.3861957 and Q.lam1 > 0.007671243:
        z += -849.5751 * (0.3861957 - Q.tau21) * (Q.lam1 - 0.007671243)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.z_top50_slots > 0.978741:
        z += 3.511995 * (8.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.978741)
    if Q.D2 < 2.410481 and Q.D2_b2 < 5.142486:
        z += 0.09357084 * (2.410481 - Q.D2) * (5.142486 - Q.D2_b2)
    if Q.n_dr_0p1_0p2 < 10.0 and Q.psi_0p2 > 0.9313699:
        z += 0.6331545 * (10.0 - Q.n_dr_0p1_0p2) * (Q.psi_0p2 - 0.9313699)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.3874575
    if Q.mass_top40 < 57.37264:
        z += -0.06244032 * Q.mass_top40 + 3.582366
    if Q.e3 >= 0.0001841806:
        z += 238.9763 * Q.e3 - 0.0440148
    if Q.mass_top30 < 100.7509:
        z += 0.002929752 * Q.mass_top30 - 0.2951751
    if Q.mass < 74.25181:
        z += 0.1533598 * Q.mass - 11.38724
    if Q.psi_0p3 >= 0.9973959:
        z += 318.9384 * Q.psi_0p3 - 318.1079
    if Q.n_particles >= 22.0:
        z += -0.02569093 * Q.n_particles + 0.5652005
    if Q.sum_z_dr2_top15 < 0.004175169:
        z += 245.116 * Q.sum_z_dr2_top15 - 1.023401
    if Q.e2 >= 0.06510219:
        z += -32.18396 * Q.e2 + 2.095247
    if Q.sj2_dr >= 0.1666353:
        z += 0.2872378 * Q.sj2_dr - 0.04786394
    if Q.sum_zz_dr2 >= 0.009606007:
        z += 127.8367 * Q.sum_zz_dr2 - 1.228
    if Q.mass_top15 < 57.91414:
        z += -0.006989202 * Q.mass_top15 + 0.4047736
    if Q.n_dr_0p2_0p4 < 26.0:
        z += -0.02760495 * Q.n_dr_0p2_0p4 + 0.7177286
    if Q.log_sum_pt >= 6.97212:
        z += 0.8709388 * Q.log_sum_pt - 6.072291
    if Q.sum_z_dr < 0.03552114:
        z += 71.84292 * Q.sum_z_dr - 2.551942
    if Q.lam1 >= 0.007259145:
        z += -80.76974 * Q.lam1 + 0.5863193
    if Q.psi_0p1 < 0.9909875:
        z += -0.2891261 * Q.psi_0p1 + 0.2865203
    if Q.C2 >= 0.06655881:
        z += -2.126666 * Q.C2 + 0.1415484
    if Q.sum_pt >= 995.6769:
        z += 0.001596543 * Q.sum_pt - 1.589641
    if Q.sum_z_dr2_top5 < 0.02429622:
        z += -15.14296 * Q.sum_z_dr2_top5 + 0.3679166
    if Q.psi_0p3 > 0.9973959 and Q.n_dr_0_0p05 < 13.0:
        z += -12.02371 * (Q.psi_0p3 - 0.9973959) * (13.0 - Q.n_dr_0_0p05)
    if Q.n_particles > 22.0 and Q.n_dr_0_0p05 > 10.0:
        z += 0.0004135622 * (Q.n_particles - 22.0) * (Q.n_dr_0_0p05 - 10.0)
    if Q.n_particles > 22.0 and Q.soft1_pt < 2.275391:
        z += 0.003910891 * (Q.n_particles - 22.0) * (2.275391 - Q.soft1_pt)
    if Q.n_dr_0p2_0p4 < 26.0 and Q.n_dr_0p1_0p2 > 11.0:
        z += -0.001272899 * (26.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 11.0)
    if Q.sj2_dr > 0.2232169 and Q.D2_b2 < 20.40995:
        z += -0.155522 * (Q.sj2_dr - 0.2232169) * (20.40995 - Q.D2_b2)
    if Q.mass < 80.78464 and Q.D2_b2 < 3.852812:
        z += -0.03836669 * (80.78464 - Q.mass) * (3.852812 - Q.D2_b2)
    if Q.mass_top40 < 83.32554 and Q.D2_b2 < 3.014827:
        z += -0.0002075238 * (83.32554 - Q.mass_top40) * (3.014827 - Q.D2_b2)
    if Q.n_particles > 22.0 and Q.D2_b2 < 3.852812:
        z += -0.0005236706 * (Q.n_particles - 22.0) * (3.852812 - Q.D2_b2)
    if Q.mass < 92.85979 and Q.sum_z_dr2_top2 < 0.005180665:
        z += 1.131772 * (92.85979 - Q.mass) * (0.005180665 - Q.sum_z_dr2_top2)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.101002
    z += -0.03736931 * Q.n_particles + 2.391636
    if Q.mass_over_sum_pt >= 0.07437106:
        z += -23.07984 * Q.mass_over_sum_pt + 1.716472
    if Q.log_sum_pt >= 6.903354:
        z += -24.65407 * Q.log_sum_pt + 170.1958
    if Q.sum_pt >= 1012.538:
        z += -0.01088267 * Q.sum_pt + 11.01911
    if Q.psi_0p3 >= 0.9973959:
        z += -113.7698 * Q.psi_0p3 + 113.4735
    if Q.sum_z_dr2_top50 >= 0.01951641:
        z += -245.2965 * Q.sum_z_dr2_top50 + 4.787307
    if Q.sum_z_dr2 >= 0.004756928:
        z += 58.81691 * Q.sum_z_dr2 - 0.2797878
    if Q.sum_pt_top50 >= 935.0507:
        z += 0.01945395 * Q.sum_pt_top50 - 18.19043
    if Q.sum_pt_top3 < 787.6281:
        z += 0.0004970322 * Q.sum_pt_top3 - 0.3914765
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.01200406 * Q.n_dr_0p2_0p4 + 0.1320447
    if Q.sum_pt_top40 >= 1007.262:
        z += -0.004494195 * Q.sum_pt_top40 + 4.526831
    if Q.sum_pt_top30 >= 933.1875:
        z += 0.003006925 * Q.sum_pt_top30 - 2.806025
    if Q.max_dr < 0.4367319:
        z += -2.581864 * Q.max_dr + 1.127582
    if Q.z_top30_slots >= 0.9048492:
        z += -4.219547 * Q.z_top30_slots + 3.818054
    if Q.mass >= 172.8:
        z += -0.2569092 * Q.mass + 44.39391
    if Q.sum_z_dr2_top30 >= 0.005417092:
        z += 50.10796 * Q.sum_z_dr2_top30 - 0.2714394
    if Q.z_dr_0_0p05 >= 0.9084912:
        z += 4.204195 * Q.z_dr_0_0p05 - 3.819475
    if Q.mass_top40 < 150.0144:
        z += 0.005165109 * Q.mass_top40 - 0.774841
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -504.5666 * Q.mass_over_sum_pt_sq + 14.73335
    if Q.tau21 < 0.5494307:
        z += 0.8523287 * Q.tau21 - 0.4682955
    if Q.sd_mass >= 115.8435:
        z += -0.01356518 * Q.sd_mass + 1.571438
    if Q.mass_top50 >= 157.5448:
        z += -0.1502028 * Q.mass_top50 + 23.66367
    if Q.z_11 < 0.01375115:
        z += -152.3208 * Q.z_11 + 2.094586
    if Q.e2 < 0.04334991:
        z += -10.94103 * Q.e2 + 0.4742928
    if Q.n_dr_0p1_0p2 < 21.0:
        z += 0.02270795 * Q.n_dr_0p1_0p2 - 0.4768669
    if Q.mass_top10 >= 71.781:
        z += 0.01736949 * Q.mass_top10 - 1.246799
    if Q.pt_11 < 14.14062:
        z += 0.1367898 * Q.pt_11 - 1.934293
    if Q.D2 < 1.976207:
        z += -0.4284304 * Q.D2 + 0.8466672
    if Q.z_top20_slots >= 0.9102775:
        z += -5.388477 * Q.z_top20_slots + 4.90501
    if Q.z_top50_slots >= 0.9906378:
        z += -40.65603 * Q.z_top50_slots + 40.2754
    if Q.sum_z_dr2_top15 < 0.005788041:
        z += 15.52728 * Q.sum_z_dr2_top15 - 0.08987256
    if Q.sum_z_dr2_top20 < 0.01083435:
        z += 52.62938 * Q.sum_z_dr2_top20 - 0.5702053
    if Q.n_particles < 64.0 and Q.D2 < 2.178951:
        z += -0.012822 * (64.0 - Q.n_particles) * (2.178951 - Q.D2)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 65130.97 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    if Q.psi_0p3 > 0.9973959 and Q.D2 < 3.345339:
        z += 33.4998 * (Q.psi_0p3 - 0.9973959) * (3.345339 - Q.D2)
    if Q.n_pt_above_1 > 28.0 and Q.tau43 < 0.942303:
        z += 0.02360206 * (Q.n_pt_above_1 - 28.0) * (0.942303 - Q.tau43)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.223857
    if Q.mass_top50 < 115.5062:
        z += 0.0002855914 * Q.mass_top50 - 0.03298758
    if Q.mass >= 86.4:
        z += 0.03549208 * Q.mass - 3.066516
    if Q.e3 < 0.0003372339:
        z += 1616.685 * Q.e3 - 0.5452011
    if Q.sj3_pair_mass_min >= 60.17253:
        z += -0.2271636 * Q.sj3_pair_mass_min + 13.66901
    if Q.e2 < 0.04755309:
        z += -43.76789 * Q.e2 + 2.081298
    if Q.sj3_mass1 >= 21.11128:
        z += -0.01014425 * Q.sj3_mass1 + 0.214158
    if Q.log_sum_pt < 6.811175:
        z += -16.7401 * Q.log_sum_pt + 114.0197
    if Q.tau1 >= 0.0920727:
        z += 12.45479 * Q.tau1 - 1.146747
    if Q.z_top40_slots < 0.9300465:
        z += -8.930605 * Q.z_top40_slots + 8.305878
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -103.3702 * Q.mass_over_sum_pt + 12.22104
    if Q.sum_z_dr2_top30 < 0.02809026:
        z += 86.13323 * Q.sum_z_dr2_top30 - 2.419505
    if Q.sum_z_dr2_top20 >= 0.008031209:
        z += 163.6994 * Q.sum_z_dr2_top20 - 1.314704
    if Q.lam1 < 0.005623108:
        z += -412.8707 * Q.lam1 + 2.321617
    if Q.sum_z_dr2_top50 < 0.004566318:
        z += 187.6623 * Q.sum_z_dr2_top50 - 0.8569259
    if Q.lam2 < 0.003687605:
        z += -87.99085 * Q.lam2 + 0.3244755
    if Q.sum_z_dr2_top40 >= 0.008031986:
        z += 118.6347 * Q.sum_z_dr2_top40 - 0.9528723
    if Q.sum_z_dr >= 0.02085222:
        z += 1.093734 * Q.sum_z_dr - 0.02280678
    if Q.sum_pt < 1260.541:
        z += 0.0005575891 * Q.sum_pt - 0.7028638
    if Q.mass < 120.6 and Q.sum_pt < 1007.788:
        z += -6.943999e-05 * (120.6 - Q.mass) * (1007.788 - Q.sum_pt)
    if Q.sj3_pair_mass_min > 29.00832 and Q.psi_0p3 > 0.9896594:
        z += -0.1862368 * (Q.sj3_pair_mass_min - 29.00832) * (Q.psi_0p3 - 0.9896594)
    if Q.mass_over_sum_pt > 0.1182259 and Q.C2_b2 > 0.02704832:
        z += 735.9931 * (Q.mass_over_sum_pt - 0.1182259) * (Q.C2_b2 - 0.02704832)
    if Q.sum_z_dr2_top30 > 0.008376291 and Q.D2_b2 > 1.67722:
        z += -30.17361 * (Q.sum_z_dr2_top30 - 0.008376291) * (Q.D2_b2 - 1.67722)
    if Q.mass_over_sum_pt > 0.1182259 and Q.D2_b2 < 7.36624:
        z += -12.51585 * (Q.mass_over_sum_pt - 0.1182259) * (7.36624 - Q.D2_b2)
    if Q.sum_z_dr2_top20 > 0.008031209 and Q.C2_b2 > 0.0008187529:
        z += -3733.588 * (Q.sum_z_dr2_top20 - 0.008031209) * (Q.C2_b2 - 0.0008187529)
    if Q.mass_over_sum_pt > 0.02682209 and Q.z_dr_0p05_0p1 < 0.7108211:
        z += -3.069807 * (Q.mass_over_sum_pt - 0.02682209) * (0.7108211 - Q.z_dr_0p05_0p1)
    if Q.mass_over_sum_pt > 0.09795415 and Q.soft8_z < 0.001160626:
        z += -29637.69 * (Q.mass_over_sum_pt - 0.09795415) * (0.001160626 - Q.soft8_z)
    if Q.e2 > 0.02793599 and Q.max_dr < 0.4357228:
        z += 54.64955 * (Q.e2 - 0.02793599) * (0.4357228 - Q.max_dr)
    if Q.mass_over_sum_pt > 0.1182259 and Q.soft3_dr0 > 0.1966723:
        z += 22.09905 * (Q.mass_over_sum_pt - 0.1182259) * (Q.soft3_dr0 - 0.1966723)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.5155219
    if Q.tau21_b2 < 0.2352054:
        z += -13.29427 * Q.tau21_b2 + 3.126885
    if Q.sum_z_dr2 < 0.01372009:
        z += -138.4839 * Q.sum_z_dr2 + 1.900012
    if Q.mass_over_sum_pt < 0.1182259:
        z += -8.93363 * Q.mass_over_sum_pt + 1.056187
    if Q.mass < 87.3546:
        z += 0.08407871 * Q.mass - 7.344662
    if Q.n_dr_0p2_0p4 < 8.0:
        z += -0.09115855 * Q.n_dr_0p2_0p4 + 0.7292684
    if Q.psi_0p3 >= 0.9980008:
        z += -47.9815 * Q.psi_0p3 + 47.88558
    if Q.sum_zz_dr2 < 0.00363788:
        z += 266.6834 * Q.sum_zz_dr2 - 0.9701625
    if Q.tau1 < 0.1199231:
        z += 22.56366 * Q.tau1 - 2.705905
    if Q.lam1 < 0.01174405:
        z += 33.70339 * Q.lam1 - 0.3958143
    if Q.sd_mass >= 97.07106:
        z += -0.2144055 * Q.sd_mass + 20.81257
    if Q.mass_top20 >= 103.0534:
        z += -0.03136905 * Q.mass_top20 + 3.232687
    if Q.psi_0p2 >= 0.9313699:
        z += -3.381283 * Q.psi_0p2 + 3.149225
    if Q.sd_rg < 0.1776078:
        z += -1.2197 * Q.sd_rg + 0.2166284
    if Q.tau21_b2 < 0.2352054 and Q.mass_over_sum_pt_sq < 0.007873266:
        z += -3511.244 * (0.2352054 - Q.tau21_b2) * (0.007873266 - Q.mass_over_sum_pt_sq)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9777125:
        z += -0.3230934 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 1260.541:
        z += -0.0380635 * (0.2352054 - Q.tau21_b2) * (1260.541 - Q.sum_pt)
    if Q.mass < 92.85979 and Q.z_dr_0_0p05 < 0.4947602:
        z += -0.2141436 * (92.85979 - Q.mass) * (0.4947602 - Q.z_dr_0_0p05)
    if Q.lam1 < 0.008241985 and Q.zdr_0 > 0.01564747:
        z += -17259.42 * (0.008241985 - Q.lam1) * (Q.zdr_0 - 0.01564747)
    if Q.psi_0p3 > 0.9973959 and Q.z_top50_slots > 0.978741:
        z += 10475.97 * (Q.psi_0p3 - 0.9973959) * (Q.z_top50_slots - 0.978741)
    return max(0.0, z)


def neuron_8(Q):
    z = 0.8046472
    if Q.mass_over_sum_pt >= 0.08330792:
        z += -25.22054 * Q.mass_over_sum_pt + 2.101071
    if Q.sum_pt_top40 < 1032.05:
        z += 0.01025816 * Q.sum_pt_top40 - 10.58693
    if Q.sum_z_dr2_top40 < 0.00572743:
        z += 137.1225 * Q.sum_z_dr2_top40 - 0.7853598
    if Q.n_dr_0p2_0p4 >= 4.0:
        z += 0.03985967 * Q.n_dr_0p2_0p4 - 0.1594387
    if Q.mass >= 64.06581:
        z += 0.04771258 * Q.mass - 3.056745
    if Q.sum_pt < 1022.377:
        z += -0.03041583 * Q.sum_pt + 31.09644
    if Q.sum_z_dr2_top20 >= 0.01062209:
        z += 3.133055 * Q.sum_z_dr2_top20 - 0.03327957
    if Q.log_sum_pt < 6.930088:
        z += 19.96997 * Q.log_sum_pt - 138.3937
    if Q.sum_z_dr2_top30 < 0.007463985:
        z += -168.6498 * Q.sum_z_dr2_top30 + 1.2588
    if Q.lam1_plus_lam2 < 0.009614971:
        z += 551.2596 * Q.lam1_plus_lam2 - 5.300346
    if Q.sum_z_dr2 < 0.007877041:
        z += 52.53572 * Q.sum_z_dr2 - 0.4138261
    if Q.sj2_dr >= 0.2232169:
        z += 6.693077 * Q.sj2_dr - 1.494008
    if Q.z_dr_0p1_0p2 >= 0.3340477:
        z += -0.2484984 * Q.z_dr_0p1_0p2 + 0.08301034
    if Q.n_for_90pct >= 16.0:
        z += 0.007051714 * Q.n_for_90pct - 0.1128274
    if Q.e2 >= 0.04755309:
        z += 114.9006 * Q.e2 - 5.463879
    if Q.lam1 < 0.007671243:
        z += -69.04252 * Q.lam1 + 0.5296419
    if Q.C2 >= 0.06655881:
        z += 5.029845 * Q.C2 - 0.3347805
    if Q.e3 >= 3.392004e-05:
        z += -3325.33 * Q.e3 + 0.1127953
    if Q.mass_top50 >= 77.33624:
        z += -0.02062476 * Q.mass_top50 + 1.595041
    if Q.mass_top20 >= 119.2969:
        z += -0.06088438 * Q.mass_top20 + 7.263315
    if Q.n_dr_0p1_0p2 >= 19.0:
        z += 0.0356287 * Q.n_dr_0p1_0p2 - 0.6769452
    if Q.z_top15_slots < 0.8316924:
        z += 0.3051501 * Q.z_top15_slots - 0.253791
    if Q.psi_0p1 < 0.3628388:
        z += 1.510988 * Q.psi_0p1 - 0.5482452
    if Q.z_top50_slots >= 0.9704436:
        z += -29.78493 * Q.z_top50_slots + 28.9046
    if Q.sum_z_dr2_top15 < 0.02146578:
        z += 17.23755 * Q.sum_z_dr2_top15 - 0.3700175
    if Q.sum_pt_top30 >= 978.0762:
        z += 0.003947129 * Q.sum_pt_top30 - 3.860593
    if Q.sum_pt_top50 >= 889.8503:
        z += 9.343504e-05 * Q.sum_pt_top50 - 0.0831432
    if Q.mass_over_sum_pt > 0.07696632 and Q.sum_pt < 1115.723:
        z += 0.3909874 * (Q.mass_over_sum_pt - 0.07696632) * (1115.723 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.z_dr_0p1_0p2 > 0.3340477:
        z += 0.2801274 * (15.0 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p1_0p2 - 0.3340477)
    if Q.sum_z_dr2_top40 > 0.005196966 and Q.log_sum_pt < 7.017258:
        z += -556.3926 * (Q.sum_z_dr2_top40 - 0.005196966) * (7.017258 - Q.log_sum_pt)
    if Q.mass > 64.48544 and Q.zdr_0 > 0.0008718296:
        z += -0.08474129 * (Q.mass - 64.48544) * (Q.zdr_0 - 0.0008718296)
    if Q.sum_z_dr2_top30 < 0.007463985 and Q.sj2_dr > 0.1512157:
        z += -755.8765 * (0.007463985 - Q.sum_z_dr2_top30) * (Q.sj2_dr - 0.1512157)
    if Q.tau2 < 0.0795038 and Q.sj3_mass1 > 13.38202:
        z += 0.4102768 * (0.0795038 - Q.tau2) * (Q.sj3_mass1 - 13.38202)
    if Q.sum_pt < 1017.435 and Q.dr_max_012 > 0.1828389:
        z += -0.01316395 * (1017.435 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.6679698
    if Q.mass_top40 >= 80.89043:
        z += 0.03145834 * Q.mass_top40 - 2.544679
    if Q.sum_pt_top40 < 956.2133:
        z += -0.00757525 * Q.sum_pt_top40 + 7.243554
    if Q.sum_z_dr2_top40 < 0.006628677:
        z += -232.9324 * Q.sum_z_dr2_top40 + 1.544034
    if Q.mass < 82.85409:
        z += -0.03307417 * Q.mass + 2.74033
    if Q.sum_z_dr >= 0.1564422:
        z += 10.85975 * Q.sum_z_dr - 1.698923
    if Q.tau1 < 0.05444509:
        z += 12.96909 * Q.tau1 - 0.7061033
    if Q.lam1 >= 0.01174405:
        z += 5.176824 * Q.lam1 - 0.06079688
    if Q.sum_z_dr2_top30 >= 0.0008564881:
        z += -50.05182 * Q.sum_z_dr2_top30 + 0.04286879
    if Q.D2 >= 2.178951:
        z += 0.03096919 * Q.D2 - 0.06748034
    if Q.mass_top50 >= 71.69608:
        z += -0.02667957 * Q.mass_top50 + 1.912821
    if Q.mass_top30 < 73.33139:
        z += 0.007773946 * Q.mass_top30 - 0.5700742
    if Q.LHA >= 0.404204:
        z += 3.603022 * Q.LHA - 1.456356
    if Q.z_top40_slots >= 0.9574183:
        z += -2.796527 * Q.z_top40_slots + 2.677446
    if Q.sum_pt < 986.0565:
        z += -0.02497406 * Q.sum_pt + 24.62583
    if Q.log_sum_pt < 6.903423:
        z += 16.31523 * Q.log_sum_pt - 112.631
    if Q.sum_pt_top40 < 956.2133 and Q.soft4_pt > 1.789258:
        z += 0.004929146 * (956.2133 - Q.sum_pt_top40) * (Q.soft4_pt - 1.789258)
    if Q.mass_top40 < 80.89043 and Q.sum_pt < 1034.834:
        z += 0.0002212187 * (80.89043 - Q.mass_top40) * (1034.834 - Q.sum_pt)
    if Q.mass_top40 < 80.89043 and Q.sum_pt_top40 < 906.6023:
        z += 0.001365454 * (80.89043 - Q.mass_top40) * (906.6023 - Q.sum_pt_top40)
    if Q.sum_pt_top40 < 956.2133 and Q.soft3_pt > 2.873047:
        z += -0.009563416 * (956.2133 - Q.sum_pt_top40) * (Q.soft3_pt - 2.873047)
    if Q.mass_top40 < 125.1 and Q.n_dr_0p2_0p4 > 9.0:
        z += 0.002079943 * (125.1 - Q.mass_top40) * (Q.n_dr_0p2_0p4 - 9.0)
    return max(0.0, z)


def neuron_10(Q):
    z = 5.852727
    if Q.sum_z_dr < 0.1207452:
        z += 38.15018 * Q.sum_z_dr - 4.60645
    z += -0.02893113 * Q.mass
    if Q.sum_z_dr2_top30 < 0.005402331:
        z += 17.94171 * Q.sum_z_dr2_top30 - 0.09692704
    if Q.mass_top50 >= 136.785:
        z += 0.03432946 * Q.mass_top50 - 4.695756
    if Q.sj2_mass1 < 65.20727:
        z += 0.0237279 * Q.sj2_mass1 - 1.547231
    if Q.D2 < 2.975532:
        z += -0.3815812 * Q.D2 + 1.135407
    if Q.psi_0p3 >= 0.995608:
        z += -132.749 * Q.psi_0p3 + 132.166
    if Q.sum_pt_top50 < 959.0957:
        z += 0.008898413 * Q.sum_pt_top50 - 8.53443
    if Q.dr_0 < 0.06413297:
        z += -9.851814 * Q.dr_0 + 0.6318261
    if Q.z_dr_0_0p05 >= 0.7674734:
        z += -1.543786 * Q.z_dr_0_0p05 + 1.184815
    if Q.e2 >= 0.01256762:
        z += 36.69965 * Q.e2 - 0.4612273
    if Q.sum_pt_top10 >= 943.6922:
        z += -0.009446922 * Q.sum_pt_top10 + 8.914986
    if Q.mass_over_sum_pt >= 0.1606361:
        z += -78.80046 * Q.mass_over_sum_pt + 12.6582
    if Q.zdr_1 < 0.008824206:
        z += -25.56947 * Q.zdr_1 + 0.2256303
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.04438826 * Q.n_dr_0p2_0p4 - 0.4882709
    if Q.sum_pt_top15 < 935.1043:
        z += -0.005210601 * Q.sum_pt_top15 + 4.872456
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -2.095497 * Q.z_dr_0p2_0p4 + 0.1911631
    if Q.mass_top15 < 72.18744:
        z += 0.008149676 * Q.mass_top15 - 0.5883042
    if Q.sum_z_dr2_top20 >= 0.007536681:
        z += 18.90485 * Q.sum_z_dr2_top20 - 0.1424798
    if Q.mass_top5 >= 22.18342:
        z += 0.007383845 * Q.mass_top5 - 0.1637989
    if Q.mass_top30 >= 60.13856:
        z += -0.002816316 * Q.mass_top30 + 0.1693692
    if Q.tau1 >= 0.05426326:
        z += -19.3923 * Q.tau1 + 1.052289
    if Q.pt_dispersion >= 0.27462:
        z += -2.416298 * Q.pt_dispersion + 0.6635639
    if Q.tau2 < 0.04828819:
        z += -19.82006 * Q.tau2 + 0.9570746
    if Q.sum_pt < 986.0565:
        z += -0.003804677 * Q.sum_pt + 3.751627
    if Q.sum_z_dr2_top40 >= 0.02497133:
        z += 149.4135 * Q.sum_z_dr2_top40 - 3.731054
    if Q.mass_top50 > 160.8 and Q.soft4_z > 0.001721109:
        z += -30.31637 * (Q.mass_top50 - 160.8) * (Q.soft4_z - 0.001721109)
    if Q.D2 < 2.975532 and Q.sj2_dr > 0.2070855:
        z += -4.242711 * (2.975532 - Q.D2) * (Q.sj2_dr - 0.2070855)
    if Q.sum_z_dr2_top30 < 0.005402331 and Q.psi_0p3 > 0.9985421:
        z += 167274.3 * (0.005402331 - Q.sum_z_dr2_top30) * (Q.psi_0p3 - 0.9985421)
    if Q.sum_z_dr2_top10 < 0.01976735 and Q.psi_0p3 > 0.9985421:
        z += -20694.14 * (0.01976735 - Q.sum_z_dr2_top10) * (Q.psi_0p3 - 0.9985421)
    if Q.sum_z_dr2_top10 < 0.01976735 and Q.max_dr < 0.4357228:
        z += -93.3416 * (0.01976735 - Q.sum_z_dr2_top10) * (0.4357228 - Q.max_dr)
    if Q.mass_top5 < 59.40777 and Q.z_dr_0p05_0p1 < 0.8509215:
        z += 0.0003567504 * (59.40777 - Q.mass_top5) * (0.8509215 - Q.z_dr_0p05_0p1)
    if Q.mass_top50 > 136.785 and Q.soft5_z > 0.001594761:
        z += 4.639392 * (Q.mass_top50 - 136.785) * (Q.soft5_z - 0.001594761)
    if Q.mass > 162.8363 and Q.soft5_z > 0.001434897:
        z += -29.59236 * (Q.mass - 162.8363) * (Q.soft5_z - 0.001434897)
    if Q.sum_pt_top10 > 943.6922 and Q.eta_0 < -0.07794189:
        z += -0.7920926 * (Q.sum_pt_top10 - 943.6922) * (-0.07794189 - Q.eta_0)
    if Q.sj2_mass1 < 65.20727 and Q.sj2_mass2 > 7.769597:
        z += 0.0003346858 * (65.20727 - Q.sj2_mass1) * (Q.sj2_mass2 - 7.769597)
    if Q.e2 < 0.01541561 and Q.pt2_over_pt0 < 0.7904923:
        z += -56.1134 * (0.01541561 - Q.e2) * (0.7904923 - Q.pt2_over_pt0)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.2224498
    if Q.n_dr_0p2_0p4 < 8.0:
        z += -0.1866609 * Q.n_dr_0p2_0p4 + 1.493287
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += -206.034 * Q.mass_over_sum_pt_sq + 1.976899
    if Q.mass < 100.0884:
        z += -0.06524422 * Q.mass + 6.530192
    if Q.sum_z_dr2_top10 < 0.001291487:
        z += -112.7988 * Q.sum_z_dr2_top10 + 0.1456781
    if Q.e2 < 0.02804652:
        z += -116.9578 * Q.e2 + 3.280258
    if Q.sum_z_dr2_top30 < 0.006929741:
        z += 169.5427 * Q.sum_z_dr2_top30 - 1.174887
    if Q.mass_top30 < 60.13856:
        z += -0.04469939 * Q.mass_top30 + 2.688157
    if Q.D2 < 5.435412:
        z += 0.03470187 * Q.D2 - 0.188619
    if Q.mass_top40 < 83.32554:
        z += 0.04189559 * Q.mass_top40 - 3.490972
    if Q.sum_z_dr < 0.07032099:
        z += 21.35909 * Q.sum_z_dr - 1.501992
    if Q.sum_zz_dr2 < 0.00818374:
        z += -282.4322 * Q.sum_zz_dr2 + 2.311351
    if Q.mass_over_sum_pt < 0.07699881:
        z += 31.35044 * Q.mass_over_sum_pt - 2.413947
    if Q.lam1 < 0.006716737:
        z += 323.8591 * Q.lam1 - 2.175276
    if Q.e3 < 3.793233e-05:
        z += 10602.22 * Q.e3 - 0.402167
    if Q.psi_0p2 >= 0.9313699:
        z += -10.52948 * Q.psi_0p2 + 9.806842
    if Q.mass_top50 < 71.69608:
        z += 0.04568582 * Q.mass_top50 - 3.275494
    if Q.psi_0p3 >= 0.9896594:
        z += 49.59144 * Q.psi_0p3 - 49.07864
    if Q.n_dr_0p1_0p2 < 15.0:
        z += -0.04290064 * Q.n_dr_0p1_0p2 + 0.6435096
    if Q.z_top5_slots >= 0.534626:
        z += -1.686013 * Q.z_top5_slots + 0.9013864
    if Q.sum_z_dr2_top20 < 0.008031209:
        z += 79.55608 * Q.sum_z_dr2_top20 - 0.6389315
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_particles > 34.0:
        z += -0.002791928 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_particles - 34.0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.sum_z_dr2 < 0.005532208:
        z += -30.7462 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.sum_z_dr2)
    if Q.mass < 101.0497 and Q.sum_pt_top10 < 891.875:
        z += -8.423846e-05 * (101.0497 - Q.mass) * (891.875 - Q.sum_pt_top10)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.phi_0 < 0.0402832:
        z += 0.30637 * (10.0 - Q.n_dr_0p2_0p4) * (0.0402832 - Q.phi_0)
    if Q.mass_top50 < 80.35535 and Q.zdr_4 < 0.003932029:
        z += 1.189456 * (80.35535 - Q.mass_top50) * (0.003932029 - Q.zdr_4)
    if Q.mass < 92.85979 and Q.zdr_4 < 0.004495205:
        z += -0.1099352 * (92.85979 - Q.mass) * (0.004495205 - Q.zdr_4)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.687824
    if Q.mass < 86.4:
        z += -0.06487528 * Q.mass + 5.605224
    if Q.sd_mass >= 125.1:
        z += 0.03317354 * Q.sd_mass - 4.15001
    if Q.sj3_pair_mass_max >= 120.6:
        z += -0.006274487 * Q.sj3_pair_mass_max + 0.7567032
    if Q.mass_top40 < 74.78616:
        z += 0.01388387 * Q.mass_top40 - 1.038321
    if Q.log_sum_pt < 6.856375:
        z += -1.050452 * Q.log_sum_pt + 7.202296
    if Q.sum_z_dr2_top15 < 0.006142802:
        z += 2.255669 * Q.sum_z_dr2_top15 - 0.01385613
    if Q.psi_0p1 >= 0.9538343:
        z += 3.997568 * Q.psi_0p1 - 3.813018
    if Q.mass_over_sum_pt < 0.06895248:
        z += 27.85309 * Q.mass_over_sum_pt - 1.92054
    if Q.mass < 86.4 and Q.z_dr_0p1_0p2 < 0.06472584:
        z += -0.4164567 * (86.4 - Q.mass) * (0.06472584 - Q.z_dr_0p1_0p2)
    if Q.mass_top40 < 89.6788 and Q.n_dr_0p05_0p1 > 1.0:
        z += 0.0006390558 * (89.6788 - Q.mass_top40) * (Q.n_dr_0p05_0p1 - 1.0)
    if Q.mass < 86.4 and Q.psi_0p3 < 0.9985421:
        z += -2.140137 * (86.4 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.sj3_pair_mass_max > 120.6 and Q.sj3_pair_mass_min < 76.60223:
        z += 0.000399872 * (Q.sj3_pair_mass_max - 120.6) * (76.60223 - Q.sj3_pair_mass_min)
    if Q.sj3_pair_mass_max > 120.6 and Q.z_dr_0p1_0p2 > 0.2864926:
        z += 0.02753936 * (Q.sj3_pair_mass_max - 120.6) * (Q.z_dr_0p1_0p2 - 0.2864926)
    if Q.mass < 86.4 and Q.lam2 < 0.003687605:
        z += 10.81132 * (86.4 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.sum_z_dr2_top10 < 0.0007431905 and Q.sum_pt_top40 < 1069.671:
        z += -0.06098147 * (0.0007431905 - Q.sum_z_dr2_top10) * (1069.671 - Q.sum_pt_top40)
    if Q.mass < 86.4 and Q.z_top50_slots < 0.985099:
        z += -1.757177 * (86.4 - Q.mass) * (0.985099 - Q.z_top50_slots)
    if Q.log_sum_pt < 6.856375 and Q.lam2 < 0.002396991:
        z += -2556.814 * (6.856375 - Q.log_sum_pt) * (0.002396991 - Q.lam2)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.201759
    if Q.sum_pt < 1028.007:
        z += 0.0162933 * Q.sum_pt - 16.74962
    if Q.mass >= 160.8:
        z += -0.3955611 * Q.mass + 63.60622
    if Q.n_for_90pct < 16.0:
        z += 0.3089804 * Q.n_for_90pct - 4.943686
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 231.8463 * Q.mass_over_sum_pt - 39.61793
    if Q.sum_pt_top40 < 1024.662:
        z += -0.00954491 * Q.sum_pt_top40 + 9.780305
    if Q.mass_top50 >= 138.2966:
        z += -0.002319919 * Q.mass_top50 + 0.320837
    if Q.sum_z_dr2_top50 < 0.01327867:
        z += -52.94843 * Q.sum_z_dr2_top50 + 0.7030847
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -697.9658 * Q.mass_over_sum_pt_sq + 20.38061
    z += 0.007045716 * Q.n_particles
    if Q.log_sum_pt < 6.910131:
        z += 16.88337 * Q.log_sum_pt - 116.6663
    if Q.mass_top40 < 160.8:
        z += 0.004157785 * Q.mass_top40 - 0.6685718
    if Q.sum_pt_top50 < 997.0189:
        z += 0.0003549141 * Q.sum_pt_top50 - 0.353856
    if Q.sum_pt_top30 < 966.0633:
        z += 0.002066321 * Q.sum_pt_top30 - 1.996197
    if Q.mass_top5 >= 33.71058:
        z += 0.000308431 * Q.mass_top5 - 0.01039739
    if Q.sum_pt_top20 >= 956.5062:
        z += 0.02173853 * Q.sum_pt_top20 - 20.79304
    if Q.mass_top10 >= 23.38894:
        z += -0.00436336 * Q.mass_top10 + 0.1020544
    if Q.mass_top30 >= 138.3818:
        z += -0.07357329 * Q.mass_top30 + 10.18121
    if Q.sum_pt < 1085.125 and Q.tau21_b2 < 0.8310045:
        z += -0.01180696 * (1085.125 - Q.sum_pt) * (0.8310045 - Q.tau21_b2)
    if Q.sum_pt_top40 < 1053.047 and Q.sj2_zsoft < 0.2179035:
        z += 0.02462104 * (1053.047 - Q.sum_pt_top40) * (0.2179035 - Q.sj2_zsoft)
    if Q.mass > 74.25181 and Q.sum_pt < 1028.184:
        z += 0.0002351627 * (Q.mass - 74.25181) * (1028.184 - Q.sum_pt)
    if Q.log_sum_pt > 7.139296 and Q.C3 < 0.00317537:
        z += -15110.62 * (Q.log_sum_pt - 7.139296) * (0.00317537 - Q.C3)
    if Q.log_sum_pt < 6.811175 and Q.dr_13 < 0.1312677:
        z += 35.35591 * (6.811175 - Q.log_sum_pt) * (0.1312677 - Q.dr_13)
    if Q.mass > 172.8 and Q.pt_6 < 56.53125:
        z += -0.002391099 * (Q.mass - 172.8) * (56.53125 - Q.pt_6)
    if Q.mass > 172.8 and Q.z_6 < 0.04568661:
        z += -27.10021 * (Q.mass - 172.8) * (0.04568661 - Q.z_6)
    if Q.log_sum_pt > 7.139296 and Q.sd_zg > 0.4462823:
        z += -279.7514 * (Q.log_sum_pt - 7.139296) * (Q.sd_zg - 0.4462823)
    if Q.log_sum_pt < 6.811175 and Q.zdr_11 < 0.005041702:
        z += 453.8663 * (6.811175 - Q.log_sum_pt) * (0.005041702 - Q.zdr_11)
    if Q.mass > 162.8363 and Q.soft6_z > 0.0005748372:
        z += -30.68952 * (Q.mass - 162.8363) * (Q.soft6_z - 0.0005748372)
    if Q.mass_top50 > 136.785 and Q.soft6_z > 0.0008188601:
        z += -65.74601 * (Q.mass_top50 - 136.785) * (Q.soft6_z - 0.0008188601)
    if Q.mass > 172.8 and Q.soft5_z > 0.0004140594:
        z += -330.6744 * (Q.mass - 172.8) * (Q.soft5_z - 0.0004140594)
    if Q.mass_top50 > 92.16545 and Q.soft7_z < 0.001923089:
        z += -38.20382 * (Q.mass_top50 - 92.16545) * (0.001923089 - Q.soft7_z)
    if Q.sum_z_dr2_top50 < 0.02550569 and Q.psi_0p3 > 0.9638082:
        z += 1525.787 * (0.02550569 - Q.sum_z_dr2_top50) * (Q.psi_0p3 - 0.9638082)
    if Q.log_sum_pt < 6.811175 and Q.D2 < 2.680253:
        z += 5.341813 * (6.811175 - Q.log_sum_pt) * (2.680253 - Q.D2)
    if Q.mass > 172.8 and Q.D2 > 0.8942376:
        z += 0.02140054 * (Q.mass - 172.8) * (Q.D2 - 0.8942376)
    if Q.n_for_90pct < 16.0 and Q.D2 < 9.676985:
        z += 0.01802415 * (16.0 - Q.n_for_90pct) * (9.676985 - Q.D2)
    if Q.sum_pt < 1034.834 and Q.pt_9 < 16.64062:
        z += 0.001369319 * (1034.834 - Q.sum_pt) * (16.64062 - Q.pt_9)
    if Q.log_sum_pt > 7.139296 and Q.dr_9 < 0.04047238:
        z += -16.11887 * (Q.log_sum_pt - 7.139296) * (0.04047238 - Q.dr_9)
    return max(0.0, z)


def neuron_14(Q):
    z = -1.054178
    if Q.tau21_b2 < 0.342495:
        z += -0.2268461 * Q.tau21_b2 + 0.07769365
    if Q.mass < 82.85409:
        z += 0.3263557 * Q.mass - 27.0399
    if Q.psi_0p3 >= 0.9924477:
        z += 165.736 * Q.psi_0p3 - 164.4843
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.0253835 * Q.n_dr_0p2_0p4 + 0.4569031
    if Q.e2 < 0.03266801:
        z += -93.60449 * Q.e2 + 3.057872
    if Q.mass_over_sum_pt < 0.140939:
        z += -44.53664 * Q.mass_over_sum_pt + 6.276951
    if Q.sum_z_dr2_top20 < 0.01083435:
        z += 113.2049 * Q.sum_z_dr2_top20 - 1.226502
    if Q.sum_z_dr < 0.076787:
        z += 27.89657 * Q.sum_z_dr - 2.142094
    if Q.sum_z_dr2_top5 < 0.007164202:
        z += -37.55152 * Q.sum_z_dr2_top5 + 0.2690267
    if Q.sd_rg < 0.1377378:
        z += -3.520317 * Q.sd_rg + 0.4848808
    if Q.sum_z_dr2_top30 < 0.002582316:
        z += 799.664 * Q.sum_z_dr2_top30 - 2.064985
    if Q.tau1 < 0.07729606:
        z += 7.557624 * Q.tau1 - 0.5841746
    if Q.mass_top30 < 82.66587:
        z += 0.01665717 * Q.mass_top30 - 1.376979
    if Q.sum_pt_top40 < 859.7766:
        z += 0.006107819 * Q.sum_pt_top40 - 5.251359
    if Q.sum_z_dr2_top3 < 0.001592178:
        z += 71.64024 * Q.sum_z_dr2_top3 - 0.114064
    if Q.log_sum_pt < 6.924973:
        z += -1.350718 * Q.log_sum_pt + 9.35369
    if Q.sum_pt_top50 < 976.277:
        z += -0.00406185 * Q.sum_pt_top50 + 3.965491
    if Q.lam1 < 0.02037163:
        z += 4.930291 * Q.lam1 - 0.1004381
    if Q.psi_0p2 >= 0.9804031:
        z += -17.46965 * Q.psi_0p2 + 17.1273
    if Q.tau2 < 0.0795038:
        z += 3.168043 * Q.tau2 - 0.2518715
    if Q.mass_top50 < 61.39113:
        z += -0.06259645 * Q.mass_top50 + 3.842867
    if Q.sum_z_dr2_top40 < 0.01897915:
        z += 40.97645 * Q.sum_z_dr2_top40 - 0.777698
    if Q.mass_top40 < 89.6788:
        z += -0.001670854 * Q.mass_top40 + 0.1498402
    if Q.N2 < 0.4226723 and Q.max_dr > 0.2404747:
        z += -10.062 * (0.4226723 - Q.N2) * (Q.max_dr - 0.2404747)
    if Q.psi_0p3 > 0.9924477 and Q.sd_rg < 0.1881908:
        z += -593.9919 * (Q.psi_0p3 - 0.9924477) * (0.1881908 - Q.sd_rg)
    if Q.tau21_b2 < 0.342495 and Q.n_real_top40 < 32.0:
        z += -0.1648871 * (0.342495 - Q.tau21_b2) * (32.0 - Q.n_real_top40)
    if Q.mass_over_sum_pt < 0.09046749 and Q.psi_0p2 > 0.9087063:
        z += -384.6881 * (0.09046749 - Q.mass_over_sum_pt) * (Q.psi_0p2 - 0.9087063)
    if Q.psi_0p3 > 0.9924477 and Q.z_2nd < 0.1231747:
        z += -662.7838 * (Q.psi_0p3 - 0.9924477) * (0.1231747 - Q.z_2nd)
    if Q.sum_z_dr2_top5 < 0.007164202 and Q.n_pt_above_10 > 13.0:
        z += -3.107422 * (0.007164202 - Q.sum_z_dr2_top5) * (Q.n_pt_above_10 - 13.0)
    if Q.sum_z_dr < 0.076787 and Q.z_dr_0p2_0p4 < 0.05180474:
        z += -374.1672 * (0.076787 - Q.sum_z_dr) * (0.05180474 - Q.z_dr_0p2_0p4)
    if Q.mass < 91.19 and Q.n_dr_0p05_0p1 > 14.0:
        z += -0.005364665 * (91.19 - Q.mass) * (Q.n_dr_0p05_0p1 - 14.0)
    if Q.psi_0p2 > 0.9804031 and Q.pt2_over_pt0 > 0.1852611:
        z += 36.3071 * (Q.psi_0p2 - 0.9804031) * (Q.pt2_over_pt0 - 0.1852611)
    if Q.tau21_b2 < 0.342495 and Q.orientation_deg > -18.47737:
        z += -0.01444672 * (0.342495 - Q.tau21_b2) * (Q.orientation_deg - -18.47737)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.7907442
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -6.156165 * Q.z_dr_0p1_0p2 + 0.7408555
    if Q.sum_z_dr2_top5 < 0.01115288:
        z += -74.7532 * Q.sum_z_dr2_top5 + 0.8337132
    if Q.psi_0p1 >= 0.8976117:
        z += 1.111151 * Q.psi_0p1 - 0.997382
    if Q.sum_pt < 972.3665:
        z += 0.0110151 * Q.sum_pt - 10.71071
    if Q.log_sum_pt < 7.138608:
        z += -4.471037 * Q.log_sum_pt + 31.91698
    if Q.psi_0p3 >= 0.9896594:
        z += -44.55376 * Q.psi_0p3 + 44.09305
    if Q.sj2_dr >= 0.258586:
        z += 2.238813 * Q.sj2_dr - 0.5789258
    if Q.sum_z_dr2_top10 < 0.006265841:
        z += -77.9231 * Q.sum_z_dr2_top10 + 0.4882538
    if Q.lam1 < 0.01174405:
        z += 12.5545 * Q.lam1 - 0.1474407
    if Q.n_dr_0p1_0p2 < 13.0:
        z += -0.03062814 * Q.n_dr_0p1_0p2 + 0.3981659
    if Q.D2 < 1.976207:
        z += -0.0817379 * Q.D2 + 0.161531
    if Q.lam2 < 0.001776308:
        z += -207.4252 * Q.lam2 + 0.3684511
    if Q.tau1 < 0.0705748:
        z += 24.1169 * Q.tau1 - 1.702045
    if Q.sum_pt_top20 < 1017.778:
        z += 0.002107136 * Q.sum_pt_top20 - 2.144597
    if Q.n_dr_0p2_0p4 < 26.0:
        z += -0.03317225 * Q.n_dr_0p2_0p4 + 0.8624785
    if Q.sd_mass < 97.07106:
        z += -0.00341458 * Q.sd_mass + 0.3314569
    if Q.mass_top50 < 71.79516:
        z += 0.006530707 * Q.mass_top50 - 0.4688732
    if Q.sum_pt_top40 < 994.137:
        z += 0.01303126 * Q.sum_pt_top40 - 12.95485
    if Q.sum_pt_top30 < 933.1875:
        z += -0.00790281 * Q.sum_pt_top30 + 7.374804
    if Q.sd_rg >= 0.1776078:
        z += 4.287997 * Q.sd_rg - 0.7615818
    if Q.e2 >= 0.03029714:
        z += -74.71651 * Q.e2 + 2.263697
    if Q.mass_top30 < 73.27438:
        z += -0.006244411 * Q.mass_top30 + 0.4575553
    if Q.N2 < 0.4226723:
        z += 1.700274 * Q.N2 - 0.7186589
    if Q.e3 >= 0.0001841806:
        z += -3530.948 * Q.e3 + 0.6503323
    if Q.sum_zz_dr2 < 0.006399858:
        z += 22.87939 * Q.sum_zz_dr2 - 0.1464249
    if Q.psi_0p2 >= 0.8706159:
        z += -7.761193 * Q.psi_0p2 + 6.757018
    if Q.max_dr < 0.2982:
        z += 2.751393 * Q.max_dr - 0.8204654
    if Q.sum_z_dr < 0.08589404:
        z += -27.53312 * Q.sum_z_dr + 2.364931
    if Q.LHA < 0.2941033:
        z += 10.32548 * Q.LHA - 3.036757
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 949.9169:
        z += 0.01937637 * (0.8459004 - Q.z_dr_0_0p05) * (949.9169 - Q.sum_pt)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_dr_0p2_0p4 > 5.0:
        z += -0.276239 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p2_0p4 - 5.0)
    if Q.sj2_dr > 0.2232169 and Q.C2_b2 < 0.04008677:
        z += 103.6792 * (Q.sj2_dr - 0.2232169) * (0.04008677 - Q.C2_b2)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.sj3_mass1 > 5.112677:
        z += -1.619812 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.sj3_mass1 - 5.112677)
    if Q.sum_z_dr2_top5 < 0.008329695 and Q.sj3_pairmin_over_m > 0.09540583:
        z += -69.81651 * (0.008329695 - Q.sum_z_dr2_top5) * (Q.sj3_pairmin_over_m - 0.09540583)
    if Q.e2 > 0.03029714 and Q.sj3_z2 < 0.3166869:
        z += 287.2252 * (Q.e2 - 0.03029714) * (0.3166869 - Q.sj3_z2)
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
