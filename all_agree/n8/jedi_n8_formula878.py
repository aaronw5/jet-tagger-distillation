"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; all observables, tuned for agreement), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.3% (the network: 65.8%); same class as the network for 87.1% of jets.

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
  Q.sj3_pairmax_over_m     largest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_4          mass of particles 0 and 4 [GeV]
  Q.pair_mass_0_5          mass of particles 0 and 5 [GeV]
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.pair_mass_0_7          mass of particles 0 and 7 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.ptdr0_7                pT7 · ΔR(0, 7) [GeV]
  Q.z_3                    pT of particle 3 / total pT
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_1               |Δη| of particle 1
  Q.abseta_2               |Δη| of particle 2
  Q.abseta_3               |Δη| of particle 3
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_5               |Δη| of particle 5
  Q.abseta_6               |Δη| of particle 6
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_2               |Δφ| of particle 2
  Q.absphi_3               |Δφ| of particle 3
  Q.absphi_5               |Δφ| of particle 5
  Q.absphi_6               |Δφ| of particle 6
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_3                  ΔR between particle 3 and the hardest particle
  Q.dr0_5                  ΔR between particle 5 and the hardest particle
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr1_4                  ΔR between particle 4 and the 2nd-hardest particle
  Q.dr1_5                  ΔR between particle 5 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_2                  Δη of particle 2
  Q.eta_3                  Δη of particle 3
  Q.eta_5                  Δη of particle 5
  Q.eta_6                  Δη of particle 6
  Q.eta_7                  Δη of particle 7
  Q.phi_0                  Δφ of particle 0
  Q.phi_1                  Δφ of particle 1
  Q.phi_2                  Δφ of particle 2
  Q.phi_3                  Δφ of particle 3
  Q.phi_4                  Δφ of particle 4
  Q.phi_5                  Δφ of particle 5
  Q.phi_6                  Δφ of particle 6
  Q.phi_7                  Δφ of particle 7
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau4                   N-subjettiness τ4 (β=1)
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
        sj3_pairmax_over_m=max(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_4=pair_mass(0, 4),
        pair_mass_0_5=pair_mass(0, 5),
        pair_mass_0_6=pair_mass(0, 6),
        pair_mass_0_7=pair_mass(0, 7),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass3=subjets(3)["mass"][2],
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_90pct=ncum(0.9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        pt_2=pt[2],
        pt_3=pt[3],
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        ptdr0_7=pt[7] * math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        z_3=z[3],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        zdr_5=z[5] * dr[5],
        zdr_7=z[7] * dr[7],
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        abseta_0=abs(eta[0]),
        abseta_1=abs(eta[1]),
        abseta_2=abs(eta[2]),
        abseta_3=abs(eta[3]),
        abseta_4=abs(eta[4]),
        abseta_5=abs(eta[5]),
        abseta_6=abs(eta[6]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_2=abs(phi[2]),
        absphi_3=abs(phi[3]),
        absphi_5=abs(phi[5]),
        absphi_6=abs(phi[6]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_3=math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        dr0_5=math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr1_4=math.sqrt(dist2(1, 4)) if pt[4] > 0 else 0.0,
        dr1_5=math.sqrt(dist2(1, 5)) if pt[5] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        eta_2=eta[2],
        eta_3=eta[3],
        eta_5=eta[5],
        eta_6=eta[6],
        eta_7=eta[7],
        phi_0=phi[0],
        phi_1=phi[1],
        phi_2=phi[2],
        phi_3=phi[3],
        phi_4=phi[4],
        phi_5=phi[5],
        phi_6=phi[6],
        phi_7=phi[7],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -1.491521
    if Q.planar_flow < 0.1484197:
        z += -3.405651 * Q.planar_flow + 0.5054657
    if Q.lam1_plus_lam2 < 0.004372139:
        z += 283.4778 * Q.lam1_plus_lam2 + 0.6163135
    if 0.004372139 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -430.9704 * Q.lam1_plus_lam2 + 3.73998
    if Q.sum_z_dr2 < 0.01323868:
        z += -398.3199 * Q.sum_z_dr2 + 5.605113
    if 0.01323868 <= Q.sum_z_dr2 < 0.01882765:
        z += -59.38214 * Q.sum_z_dr2 + 1.118026
    if Q.mass < 21.78408:
        z += 0.02520169 * Q.mass - 2.185881
    if 21.78408 <= Q.mass < 29.6447:
        z += 0.02355211 * Q.mass - 2.149947
    if 29.6447 <= Q.mass < 56.92035:
        z += 0.05076702 * Q.mass - 2.956725
    if 56.92035 <= Q.mass < 64.61873:
        z += 0.008709369 * Q.mass - 0.5627884
    if Q.sum_z_dr2_top3 < 0.007929074:
        z += 40.27119 * Q.sum_z_dr2_top3 - 0.3193132
    if Q.lam1 < 0.0002758826:
        z += 4646.474 * Q.lam1 - 3.732887
    if 0.0002758826 <= Q.lam1 < 0.005433361:
        z += 462.9155 * Q.lam1 - 2.578716
    if 0.005433361 <= Q.lam1 < 0.006506576:
        z += 59.19497 * Q.lam1 - 0.3851565
    if Q.sum_pt >= 901.5938:
        z += -0.008108321 * Q.sum_pt + 7.310411
    if Q.C2_b2 < 0.001563465:
        z += 142.1107 * Q.C2_b2 - 0.2221851
    if Q.sj3_dr_max < 0.1070199:
        z += -6.624108 * Q.sj3_dr_max + 1.995192
    if 0.1070199 <= Q.sj3_dr_max < 0.233678:
        z += 5.874603 * Q.sj3_dr_max + 0.6575806
    if 0.233678 <= Q.sj3_dr_max < 0.3012016:
        z += -13.68988 * Q.sj3_dr_max + 5.229371
    if Q.sj3_dr_max >= 0.3012016:
        z += -7.065776 * Q.sj3_dr_max + 3.234179
    if Q.centroid_offset >= 0.04990367:
        z += 6.940865 * Q.centroid_offset - 0.3463746
    if Q.sum_z_dr < 0.08723651:
        z += 35.47411 * Q.sum_z_dr - 3.094638
    if Q.e2 < 0.0245477:
        z += -66.1447 * Q.e2 + 1.6237
    if Q.sum_z_dr2_top5 < 0.008329695:
        z += 60.00873 * Q.sum_z_dr2_top5 - 0.4998544
    if Q.log_sum_pt < 6.080494:
        z += 7.003249 * Q.log_sum_pt - 42.58322
    if Q.log_sum_pt >= 6.670067:
        z += -4.311552 * Q.log_sum_pt + 28.75834
    if Q.mass_over_sum_pt_sq < 0.003904593:
        z += -318.962 * Q.mass_over_sum_pt_sq + 1.245417
    if Q.n_dr_0_0p05 >= 4.0:
        z += 0.1392757 * Q.n_dr_0_0p05 - 0.557103
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.007349692 * Q.sum_pt_top5 - 5.052454
    if Q.lam2 < 7.300726e-05:
        z += -2624.552 * Q.lam2 + 0.1916114
    if Q.zdr_1 < 0.01587232:
        z += -24.64126 * Q.zdr_1 + 0.391114
    if Q.planar_flow < 0.1484197 and Q.centroid_offset < 0.04990367:
        z += 184.7659 * (0.1484197 - Q.planar_flow) * (0.04990367 - Q.centroid_offset)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.D2 < 0.7459513:
        z += -3170.854 * (0.004372139 - Q.lam1_plus_lam2) * (0.7459513 - Q.D2)
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += -1348.397 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.D2 < 1.002471:
        z += 252.2654 * (0.01323868 - Q.sum_z_dr2) * (1.002471 - Q.D2)
    if Q.planar_flow < 0.1484197 and Q.sum_pt_top2 < 358.375:
        z += 0.009892442 * (0.1484197 - Q.planar_flow) * (358.375 - Q.sum_pt_top2)
    if Q.sum_pt > 901.5938 and Q.dr12 < 0.119512:
        z += -0.02044729 * (Q.sum_pt - 901.5938) * (0.119512 - Q.dr12)
    if Q.sum_z_dr2 < 0.01323868 and Q.mean_phi < -0.003096655:
        z += -1172.632 * (0.01323868 - Q.sum_z_dr2) * (-0.003096655 - Q.mean_phi)
    if Q.sum_z_dr2 < 0.01882765 and Q.mass_top2 > 28.78966:
        z += -4.960794 * (0.01882765 - Q.sum_z_dr2) * (Q.mass_top2 - 28.78966)
    if Q.sum_z_dr2 < 0.01323868 and Q.centroid_offset > 0.01837778:
        z += -11933.8 * (0.01323868 - Q.sum_z_dr2) * (Q.centroid_offset - 0.01837778)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.eccentricity > 0.9598562:
        z += -12417.93 * (0.004372139 - Q.lam1_plus_lam2) * (Q.eccentricity - 0.9598562)
    if Q.mass < 29.6447 and Q.phi_0 < 0.01452637:
        z += -0.4291993 * (29.6447 - Q.mass) * (0.01452637 - Q.phi_0)
    if Q.sum_pt > 901.5938 and Q.absphi_3 < 0.03363037:
        z += -0.06019271 * (Q.sum_pt - 901.5938) * (0.03363037 - Q.absphi_3)
    if Q.sum_z_dr2 < 0.01323868 and Q.phi_0 > -0.008995056:
        z += 830.4574 * (0.01323868 - Q.sum_z_dr2) * (Q.phi_0 - -0.008995056)
    if Q.lam1_plus_lam2 < 0.008678045 and Q.eta_7 > 0.1219513:
        z += -988.4313 * (0.008678045 - Q.lam1_plus_lam2) * (Q.eta_7 - 0.1219513)
    if Q.sum_pt > 901.5938 and Q.eccentricity > 0.9458207:
        z += -0.1711088 * (Q.sum_pt - 901.5938) * (Q.eccentricity - 0.9458207)
    if Q.planar_flow < 0.1484197 and Q.z_7 < 0.02320757:
        z += 4857.609 * (0.1484197 - Q.planar_flow) * (0.02320757 - Q.z_7)
    if Q.lam1 < 0.005433361 and Q.eta_6 > 0.05477905:
        z += 3590.055 * (0.005433361 - Q.lam1) * (Q.eta_6 - 0.05477905)
    if Q.sum_pt > 901.5938 and Q.dr0_6 > 0.2347949:
        z += 0.6416791 * (Q.sum_pt - 901.5938) * (Q.dr0_6 - 0.2347949)
    if Q.log_sum_pt > 6.670067 and Q.dr_4 < 0.07232166:
        z += 74.86577 * (Q.log_sum_pt - 6.670067) * (0.07232166 - Q.dr_4)
    if Q.log_sum_pt > 6.670067 and Q.phi_2 > 0.09649963:
        z += 1.316532 * (Q.log_sum_pt - 6.670067) * (Q.phi_2 - 0.09649963)
    if Q.sum_z_dr2 < 0.01882765 and Q.pt_6 < 46.125:
        z += 0.5270071 * (0.01882765 - Q.sum_z_dr2) * (46.125 - Q.pt_6)
    if Q.sum_z_dr2 < 0.01323868 and Q.sj3_dr12 > 0.2182873:
        z += -3791.535 * (0.01323868 - Q.sum_z_dr2) * (Q.sj3_dr12 - 0.2182873)
    if Q.mass < 64.61873 and Q.D2_b2 < 0.1830092:
        z += -0.02288638 * (64.61873 - Q.mass) * (0.1830092 - Q.D2_b2)
    if Q.lam2 < 7.300726e-05 and Q.D2_b2 < 0.2669656:
        z += 36363.95 * (7.300726e-05 - Q.lam2) * (0.2669656 - Q.D2_b2)
    if Q.sj3_dr_max < 0.3012016 and Q.D2_b2 < 0.05744392:
        z += -33.22003 * (0.3012016 - Q.sj3_dr_max) * (0.05744392 - Q.D2_b2)
    if Q.sj3_dr_max < 0.3012016 and Q.phi_1 > 0.01467133:
        z += 22.92713 * (0.3012016 - Q.sj3_dr_max) * (Q.phi_1 - 0.01467133)
    if Q.n_dr_0_0p05 > 4.0 and Q.pair_mass_0_5 > 16.26599:
        z += -0.01400745 * (Q.n_dr_0_0p05 - 4.0) * (Q.pair_mass_0_5 - 16.26599)
    if Q.lam1 < 0.005433361 and Q.phi_3 > 0.04800415:
        z += -3899.974 * (0.005433361 - Q.lam1) * (Q.phi_3 - 0.04800415)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.pair_mass_0_6 > 20.59519:
        z += 402.613 * (0.004372139 - Q.lam1_plus_lam2) * (Q.pair_mass_0_6 - 20.59519)
    if Q.planar_flow < 0.1484197 and Q.dr0_6 > 0.1755206:
        z += -114.5878 * (0.1484197 - Q.planar_flow) * (Q.dr0_6 - 0.1755206)
    if Q.mass < 56.92035 and Q.dr_max_012 > 0.06112084:
        z += 0.1027296 * (56.92035 - Q.mass) * (Q.dr_max_012 - 0.06112084)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.3116016
    if Q.lam1 < 0.0008722282:
        z += 443.4971 * Q.lam1 + 2.177843
    if 0.0008722282 <= Q.lam1 < 0.00595415:
        z += -481.2742 * Q.lam1 + 2.984454
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += -49.09315 * Q.lam1 + 0.4111833
    if 34.53125 <= Q.pt_7 < 53.4375:
        z += 0.05219307 * Q.pt_7 - 1.802292
    if Q.pt_7 >= 53.4375:
        z += 0.04123725 * Q.pt_7 - 1.21684
    if 6.377723 <= Q.log_sum_pt < 6.502799:
        z += 1.931182 * Q.log_sum_pt - 12.31654
    if 6.502799 <= Q.log_sum_pt < 6.605974:
        z += 4.8647 * Q.log_sum_pt - 31.39262
    if Q.log_sum_pt >= 6.605974:
        z += 9.121457 * Q.log_sum_pt - 59.51265
    if Q.mass_over_sum_pt_sq < 0.005832932:
        z += -379.786 * Q.mass_over_sum_pt_sq + 2.215266
    if Q.tau21_b2 < 0.006726135:
        z += -225.5347 * Q.tau21_b2 + 1.516976
    if Q.z_7 < 0.02320757:
        z += 33.5591 * Q.z_7 - 2.068757
    if 0.02320757 <= Q.z_7 < 0.04624032:
        z += 57.88808 * Q.z_7 - 2.633373
    if 0.04624032 <= Q.z_7 < 0.06164517:
        z += 45.28719 * Q.z_7 - 2.050704
    if 0.06164517 <= Q.z_7 < 0.08670959:
        z += 11.72809 * Q.z_7 + 0.01805284
    if Q.z_7 >= 0.08670959:
        z += -31.46915 * Q.z_7 + 3.763667
    if Q.sum_z_dr2 < 0.008678045:
        z += 539.2242 * Q.sum_z_dr2 - 4.679412
    if Q.mass < 49.6681:
        z += -0.03903446 * Q.mass + 1.938767
    if Q.sj3_dr_min >= 0.03628191:
        z += -4.707469 * Q.sj3_dr_min + 0.170796
    if Q.sj3_dr_max >= 0.169029:
        z += 2.797732 * Q.sj3_dr_max - 0.4728979
    if Q.sum_zz_dr2 < 0.008168571:
        z += -286.5429 * Q.sum_zz_dr2 + 2.340646
    if Q.e3 < 1.627072e-05:
        z += -41225.82 * Q.e3 + 0.6707739
    if Q.n_dr_0_0p05 < 1.0:
        z += 0.3972691 * Q.n_dr_0_0p05 - 0.3972691
    if 367.5938 <= Q.sum_pt_top5 < 531.1875:
        z += 0.0009730814 * Q.sum_pt_top5 - 0.3576986
    if Q.sum_pt_top5 >= 531.1875:
        z += -0.005765837 * Q.sum_pt_top5 + 3.221931
    if Q.lam1_plus_lam2 < 0.00609665:
        z += 458.0762 * Q.lam1_plus_lam2 - 2.79273
    if Q.sum_z_dr < 0.0717028:
        z += -15.27103 * Q.sum_z_dr + 1.094976
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += 5403.224 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.pt_7 > 34.53125 and Q.e3 < 2.955458e-05:
        z += -3.510131 * (Q.pt_7 - 34.53125) * (2.955458e-05 - Q.e3)
    if Q.log_sum_pt > 6.377723 and Q.e3 < 1.627072e-05:
        z += -265657.9 * (Q.log_sum_pt - 6.377723) * (1.627072e-05 - Q.e3)
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += 1.532884 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.tau2 < 0.06297984:
        z += 94.86779 * (0.06164517 - Q.z_7) * (0.06297984 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.D2 < 1.679198:
        z += -29.69715 * (0.06164517 - Q.z_7) * (1.679198 - Q.D2)
    if Q.pt_7 > 34.53125 and Q.sj2_dr > 0.1294903:
        z += 0.5861489 * (Q.pt_7 - 34.53125) * (Q.sj2_dr - 0.1294903)
    if Q.log_sum_pt > 6.377723 and Q.z_dr_0p05_0p1 < 0.7509095:
        z += -5.162676 * (Q.log_sum_pt - 6.377723) * (0.7509095 - Q.z_dr_0p05_0p1)
    if Q.tau21_b2 < 0.006726135 and Q.sj3_pair_mass_min > 6.811308:
        z += 1218.553 * (0.006726135 - Q.tau21_b2) * (Q.sj3_pair_mass_min - 6.811308)
    if Q.log_sum_pt > 6.377723 and Q.sj3_pairmin_over_m < 0.1340569:
        z += 20.17505 * (Q.log_sum_pt - 6.377723) * (0.1340569 - Q.sj3_pairmin_over_m)
    if Q.z_7 < 0.06164517 and Q.z_dr_0p05_0p1 > 0.04875823:
        z += -54.41084 * (0.06164517 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.04875823)
    if Q.pt_7 > 34.53125 and Q.n_dr_0p1_0p2 > 1.0:
        z += -0.0008056515 * (Q.pt_7 - 34.53125) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.pt_7 > 34.53125 and Q.pt_6 < 52.90625:
        z += -0.01161722 * (Q.pt_7 - 34.53125) * (52.90625 - Q.pt_6)
    if Q.log_sum_pt > 6.377723 and Q.centroid_offset > 0.009480685:
        z += 289.3921 * (Q.log_sum_pt - 6.377723) * (Q.centroid_offset - 0.009480685)
    if Q.log_sum_pt > 6.605974 and Q.D2 < 1.432482:
        z += 9.507805 * (Q.log_sum_pt - 6.605974) * (1.432482 - Q.D2)
    if Q.lam1 < 0.008375572 and Q.planar_flow < 0.1484197:
        z += 41.64537 * (0.008375572 - Q.lam1) * (0.1484197 - Q.planar_flow)
    if Q.pt_7 > 34.53125 and Q.z_dr_0p05_0p1 > 0.2919447:
        z += -0.08856392 * (Q.pt_7 - 34.53125) * (Q.z_dr_0p05_0p1 - 0.2919447)
    if Q.z_7 < 0.06164517 and Q.D3 < 3.916009:
        z += -0.8985684 * (0.06164517 - Q.z_7) * (3.916009 - Q.D3)
    if Q.mass < 49.6681 and Q.mean_phi > 0.002834884:
        z += 0.09677754 * (49.6681 - Q.mass) * (Q.mean_phi - 0.002834884)
    if Q.z_7 < 0.06164517 and Q.centroid_offset > 0.02076709:
        z += -229.8987 * (0.06164517 - Q.z_7) * (Q.centroid_offset - 0.02076709)
    if Q.sj3_dr_max > 0.169029 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.1337782 * (Q.sj3_dr_max - 0.169029) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.log_sum_pt > 6.605974 and Q.phi_0 > 0.07977295:
        z += 71.10391 * (Q.log_sum_pt - 6.605974) * (Q.phi_0 - 0.07977295)
    if Q.mass < 49.6681 and Q.mass_top2 > 0.6957161:
        z += 0.001839208 * (49.6681 - Q.mass) * (Q.mass_top2 - 0.6957161)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.D2_b2 < 0.03885671:
        z += -90341.21 * (0.005832932 - Q.mass_over_sum_pt_sq) * (0.03885671 - Q.D2_b2)
    if Q.log_sum_pt > 6.605974 and Q.zdr_4 > 0.002832174:
        z += 980.065 * (Q.log_sum_pt - 6.605974) * (Q.zdr_4 - 0.002832174)
    if Q.z_7 < 0.06164517 and Q.pair_mass_0_4 > 13.64049:
        z += -1.244852 * (0.06164517 - Q.z_7) * (Q.pair_mass_0_4 - 13.64049)
    if Q.pt_7 > 53.4375 and Q.dr01 > 0.1894571:
        z += 0.5175732 * (Q.pt_7 - 53.4375) * (Q.dr01 - 0.1894571)
    if Q.n_dr_0_0p05 < 1.0 and Q.n_pt_above_50 > 5.0:
        z += 0.3762443 * (1.0 - Q.n_dr_0_0p05) * (Q.n_pt_above_50 - 5.0)
    if Q.lam1 < 0.008375572 and Q.n_pt_above_50 > 5.0:
        z += -9.60358 * (0.008375572 - Q.lam1) * (Q.n_pt_above_50 - 5.0)
    if Q.log_sum_pt > 6.605974 and Q.z_4 < 0.09925997:
        z += 47.215 * (Q.log_sum_pt - 6.605974) * (0.09925997 - Q.z_4)
    if Q.sum_z_dr2 < 0.008678045 and Q.centroid_offset > 0.02076709:
        z += 16342.37 * (0.008678045 - Q.sum_z_dr2) * (Q.centroid_offset - 0.02076709)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.centroid_offset > 0.00809236:
        z += -2977.355 * (0.005832932 - Q.mass_over_sum_pt_sq) * (Q.centroid_offset - 0.00809236)
    if Q.n_dr_0_0p05 < 1.0 and Q.zdr_7 > 0.0007928864:
        z += -15.75684 * (1.0 - Q.n_dr_0_0p05) * (Q.zdr_7 - 0.0007928864)
    if Q.z_7 < 0.06164517 and Q.dr1_7 > 0.1343804:
        z += -39.18445 * (0.06164517 - Q.z_7) * (Q.dr1_7 - 0.1343804)
    if Q.z_7 < 0.06164517 and Q.mean_phi2 < 0.008921136:
        z += -441.3224 * (0.06164517 - Q.z_7) * (0.008921136 - Q.mean_phi2)
    return max(0.0, z)


def neuron_2(Q):
    z = 1.726356
    if Q.sj3_pair_mass_max < 62.55:
        z += -0.02425343 * Q.sj3_pair_mass_max + 1.517052
    if Q.log_sum_pt < 6.46415:
        z += -8.273639 * Q.log_sum_pt + 54.1411
    if 6.46415 <= Q.log_sum_pt < 6.605974:
        z += -4.646962 * Q.log_sum_pt + 30.69771
    if Q.log_sum_pt >= 6.842717:
        z += 32.64202 * Q.log_sum_pt - 223.3601
    if Q.lam1 < 0.00595415:
        z += -175.6079 * Q.lam1 + 1.045596
    if Q.z_7 < 0.03243272:
        z += 64.73396 * Q.z_7 - 2.042318
    if 0.03243272 <= Q.z_7 < 0.03629544:
        z += -1.464074 * Q.z_7 + 0.1046647
    if 0.03629544 <= Q.z_7 < 0.04939969:
        z += -35.7337 * Q.z_7 + 1.348496
    if 0.04939969 <= Q.z_7 < 0.07148865:
        z += -22.04946 * Q.z_7 + 0.6724983
    if Q.z_7 >= 0.07148865:
        z += -20.58538 * Q.z_7 + 0.5678336
    if Q.sum_z_dr < 0.007673833:
        z += 744.7125 * Q.sum_z_dr - 5.7148
    if Q.pt_7 < 34.53125:
        z += 0.1111801 * Q.pt_7 - 5.941185
    if 34.53125 <= Q.pt_7 < 43.5:
        z += 0.155514 * Q.pt_7 - 7.472092
    if 43.5 <= Q.pt_7 < 53.4375:
        z += 0.2382489 * Q.pt_7 - 11.07106
    if Q.pt_7 >= 53.4375:
        z += 0.1270688 * Q.pt_7 - 5.129872
    if Q.LHA >= 0.1329373:
        z += -11.2275 * Q.LHA + 1.492554
    if 752.1 <= Q.sum_pt_top5 < 839.9547:
        z += -9.939428e-05 * Q.sum_pt_top5 + 0.07475444
    if Q.sum_pt_top5 >= 839.9547:
        z += 0.003867702 * Q.sum_pt_top5 - 3.257427
    if Q.planar_flow < 0.4926918:
        z += 1.433169 * Q.planar_flow - 0.7061105
    if Q.mass < 36.22941:
        z += 0.04467596 * Q.mass - 1.618584
    if Q.N2 >= 0.1150852:
        z += 2.585113 * Q.N2 - 0.2975082
    if Q.max_dr < 0.2507612:
        z += -4.846927 * Q.max_dr + 1.215421
    if Q.sum_z_dr2 < 0.003562611:
        z += -393.975 * Q.sum_z_dr2 + 1.40358
    if Q.m012 < 3.55957:
        z += -0.1765049 * Q.m012 + 0.6282815
    if Q.m012 >= 40.2:
        z += 0.0835387 * Q.m012 - 3.358256
    if Q.sum_pt < 527.1781:
        z += -0.01059845 * Q.sum_pt + 8.620949
    if 527.1781 <= Q.sum_pt < 813.4156:
        z += -0.008692676 * Q.sum_pt + 7.616264
    if Q.sum_pt >= 813.4156:
        z += 0.001905779 * Q.sum_pt - 1.004685
    if Q.centroid_offset < 0.004430536:
        z += 527.7302 * Q.centroid_offset - 2.338128
    if Q.zdr_0 < 0.0211821:
        z += -47.91207 * Q.zdr_0 + 1.014878
    if Q.sum_z_dr2_top2 < 3.849518e-05:
        z += 27308.19 * Q.sum_z_dr2_top2 - 1.051234
    if Q.mass_over_sum_pt < 0.1079857:
        z += -11.30635 * Q.mass_over_sum_pt + 1.220924
    if Q.sj3_pair_mass_max < 62.55 and Q.z_7 < 0.06810151:
        z += -0.8380464 * (62.55 - Q.sj3_pair_mass_max) * (0.06810151 - Q.z_7)
    if Q.sj3_pair_mass_max < 62.55 and Q.centroid_offset > 0.01096064:
        z += -0.8644899 * (62.55 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.01096064)
    if Q.pt_7 > 34.53125 and Q.D3 > 0.4387703:
        z += -0.001550385 * (Q.pt_7 - 34.53125) * (Q.D3 - 0.4387703)
    if Q.lam1 < 0.00595415 and Q.max_dr > 0.08050702:
        z += -1929.944 * (0.00595415 - Q.lam1) * (Q.max_dr - 0.08050702)
    if Q.z_7 > 0.04939969 and Q.sj3_dr_min < 0.0623951:
        z += 78.77531 * (Q.z_7 - 0.04939969) * (0.0623951 - Q.sj3_dr_min)
    if Q.log_sum_pt > 6.842717 and Q.pt_6 > 41.21875:
        z += 0.5296949 * (Q.log_sum_pt - 6.842717) * (Q.pt_6 - 41.21875)
    if Q.z_7 > 0.04939969 and Q.D2_b2 < 0.9206502:
        z += -48.31535 * (Q.z_7 - 0.04939969) * (0.9206502 - Q.D2_b2)
    if Q.z_7 < 0.03243272 and Q.mass_top3 > 16.89912:
        z += -5.255297 * (0.03243272 - Q.z_7) * (Q.mass_top3 - 16.89912)
    if Q.sum_z_dr < 0.007673833 and Q.D2 < 3.885568:
        z += -269.1369 * (0.007673833 - Q.sum_z_dr) * (3.885568 - Q.D2)
    if Q.lam1 < 0.00595415 and Q.pt_6 > 19.46875:
        z += 6.569538 * (0.00595415 - Q.lam1) * (Q.pt_6 - 19.46875)
    if Q.sum_z_dr < 0.007673833 and Q.pt_4 < 75.625:
        z += -11.2948 * (0.007673833 - Q.sum_z_dr) * (75.625 - Q.pt_4)
    if Q.log_sum_pt < 6.605974 and Q.z_5 < 0.0531335:
        z += 597.0593 * (6.605974 - Q.log_sum_pt) * (0.0531335 - Q.z_5)
    if Q.log_sum_pt < 6.605974 and Q.max_dr < 0.03623337:
        z += 775.3235 * (6.605974 - Q.log_sum_pt) * (0.03623337 - Q.max_dr)
    if Q.mass < 36.22941 and Q.pt_5 < 73.75:
        z += 0.001150231 * (36.22941 - Q.mass) * (73.75 - Q.pt_5)
    if Q.log_sum_pt < 6.46415 and Q.D2_b2 < 1.129616:
        z += 5.108432 * (6.46415 - Q.log_sum_pt) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt > 6.842717 and Q.phi_0 < -0.0216713:
        z += -601.6338 * (Q.log_sum_pt - 6.842717) * (-0.0216713 - Q.phi_0)
    if Q.z_7 < 0.03243272 and Q.D2_b2 < 0.9206502:
        z += -107.3078 * (0.03243272 - Q.z_7) * (0.9206502 - Q.D2_b2)
    if Q.max_dr < 0.2507612 and Q.phi_0 > 0.002076054:
        z += -44.05669 * (0.2507612 - Q.max_dr) * (Q.phi_0 - 0.002076054)
    if Q.pt_7 < 53.4375 and Q.D2_b2 < 1.129616:
        z += -0.03567744 * (53.4375 - Q.pt_7) * (1.129616 - Q.D2_b2)
    if Q.sum_pt > 527.1781 and Q.absphi_3 < 0.05648804:
        z += -0.0471978 * (Q.sum_pt - 527.1781) * (0.05648804 - Q.absphi_3)
    if Q.log_sum_pt > 6.842717 and Q.eta_0 > 0.02130127:
        z += 137.7681 * (Q.log_sum_pt - 6.842717) * (Q.eta_0 - 0.02130127)
    if Q.sum_pt_top5 > 752.1 and Q.eta_0 > 0.009292603:
        z += 0.4256754 * (Q.sum_pt_top5 - 752.1) * (Q.eta_0 - 0.009292603)
    if Q.log_sum_pt > 6.842717 and Q.z_dr_0p05_0p1 < 0.4474937:
        z += 35.50709 * (Q.log_sum_pt - 6.842717) * (0.4474937 - Q.z_dr_0p05_0p1)
    if Q.sum_pt_top5 > 752.1 and Q.D2_b2 < 1.129616:
        z += 0.02123464 * (Q.sum_pt_top5 - 752.1) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt > 6.842717 and Q.D2_b2 < 1.345805:
        z += 3.85012 * (Q.log_sum_pt - 6.842717) * (1.345805 - Q.D2_b2)
    if Q.z_7 < 0.03243272 and Q.n_pt_above_5 < 8.0:
        z += -188.9657 * (0.03243272 - Q.z_7) * (8.0 - Q.n_pt_above_5)
    if Q.zdr_0 < 0.0211821 and Q.mean_eta < -0.01284493:
        z += -820.8311 * (0.0211821 - Q.zdr_0) * (-0.01284493 - Q.mean_eta)
    if Q.sum_pt_top5 > 752.1 and Q.abseta_0 < 0.0174408:
        z += -0.01595129 * (Q.sum_pt_top5 - 752.1) * (0.0174408 - Q.abseta_0)
    if Q.log_sum_pt > 6.842717 and Q.abseta_0 < 0.0174408:
        z += 1455.373 * (Q.log_sum_pt - 6.842717) * (0.0174408 - Q.abseta_0)
    if Q.pt_7 < 53.4375 and Q.abseta_0 > 0.02940369:
        z += -0.2711043 * (53.4375 - Q.pt_7) * (Q.abseta_0 - 0.02940369)
    if Q.sum_pt > 527.1781 and Q.lam2 < 0.0001947983:
        z += -17.93459 * (Q.sum_pt - 527.1781) * (0.0001947983 - Q.lam2)
    return max(0.0, z)


def neuron_3(Q):
    z = -1.353712
    if Q.mass_over_sum_pt >= 0.0681391:
        z += 57.25016 * Q.mass_over_sum_pt - 3.900974
    if Q.centroid_offset >= 0.01096064:
        z += 2.256222 * Q.centroid_offset - 0.02472963
    if 0.05356915 <= Q.tau1 < 0.1027642:
        z += -22.85143 * Q.tau1 + 1.224132
    if Q.tau1 >= 0.1027642:
        z += 2.978848 * Q.tau1 - 1.430297
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += 794.4166 * Q.lam1 - 6.653693
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 582.6773 * Q.lam1 - 4.112033
    if Q.lam1 >= 0.01643375:
        z += 528.0475 * Q.lam1 - 3.21426
    if 0.04081947 <= Q.sum_z_dr < 0.07608178:
        z += 31.42449 * Q.sum_z_dr - 1.282731
    if 0.07608178 <= Q.sum_z_dr < 0.08723651:
        z += 49.8531 * Q.sum_z_dr - 2.684812
    if Q.sum_z_dr >= 0.08723651:
        z += 89.33462 * Q.sum_z_dr - 6.129042
    if 0.007520088 <= Q.lam1_plus_lam2 < 0.01323868:
        z += 833.245 * Q.lam1_plus_lam2 - 6.266076
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += 335.8257 * Q.lam1_plus_lam2 + 0.3190962
    if Q.e2 >= 0.06344108:
        z += -65.31537 * Q.e2 + 4.143677
    if Q.sum_z_dr2 < 0.004372139:
        z += 45.52767 * Q.sum_z_dr2 - 0.1990533
    if 0.008678045 <= Q.sum_z_dr2 < 0.01882765:
        z += -1334.331 * Q.sum_z_dr2 + 11.57938
    if Q.sum_z_dr2 >= 0.01882765:
        z += -1279.348 * Q.sum_z_dr2 + 10.54418
    if 0.1492731 <= Q.sj2_dr < 0.1872617:
        z += -6.42922 * Q.sj2_dr + 0.9597096
    if 0.1872617 <= Q.sj2_dr < 0.2687922:
        z += 20.89613 * Q.sj2_dr - 4.157282
    if Q.sj2_dr >= 0.2687922:
        z += 30.17501 * Q.sj2_dr - 6.651373
    if Q.mean_eta < -0.004664942:
        z += -35.41084 * Q.mean_eta - 0.1651895
    if Q.mean_eta >= 0.01772426:
        z += 65.21388 * Q.mean_eta - 1.155868
    if Q.sum_z_dr2_top5 < 0.007164202:
        z += 242.3194 * Q.sum_z_dr2_top5 - 1.724316
    if 0.007164202 <= Q.sum_z_dr2_top5 < 0.008329695:
        z += -10.04668 * Q.sum_z_dr2_top5 + 0.08368576
    if Q.max_dr >= 0.1027585:
        z += 12.19167 * Q.max_dr - 1.252798
    if Q.lam2 >= 0.001130645:
        z += 281.5112 * Q.lam2 - 0.3182892
    if Q.mass >= 64.61873:
        z += -0.00619041 * Q.mass + 0.4000165
    if Q.sd_mass >= 62.55:
        z += 0.02462767 * Q.sd_mass - 1.540461
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.5456664 * (Q.mass_over_sum_pt - 0.0681391) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.mass_over_sum_pt > 0.0681391 and Q.pt_6 > 31.90625:
        z += -0.07244858 * (Q.mass_over_sum_pt - 0.0681391) * (Q.pt_6 - 31.90625)
    if Q.sum_z_dr > 0.04081947 and Q.log_sum_pt > 6.080494:
        z += 8.00976 * (Q.sum_z_dr - 0.04081947) * (Q.log_sum_pt - 6.080494)
    if Q.e2 > 0.06344108 and Q.sj2_mass1 > 16.86126:
        z += 0.02242201 * (Q.e2 - 0.06344108) * (Q.sj2_mass1 - 16.86126)
    if Q.sj2_dr > 0.1872617 and Q.sj2_mass1 > 2.250113:
        z += -0.08658638 * (Q.sj2_dr - 0.1872617) * (Q.sj2_mass1 - 2.250113)
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj2_dr < 0.2179769:
        z += -1430.569 * (Q.mass_over_sum_pt - 0.0681391) * (0.2179769 - Q.sj2_dr)
    if Q.lam1_plus_lam2 > 0.007520088 and Q.sj3_pairmin_over_m > 0.07708997:
        z += -12.35326 * (Q.lam1_plus_lam2 - 0.007520088) * (Q.sj3_pairmin_over_m - 0.07708997)
    if Q.centroid_offset > 0.01096064 and Q.n_pt_above_50 < 8.0:
        z += 6.285566 * (Q.centroid_offset - 0.01096064) * (8.0 - Q.n_pt_above_50)
    if Q.centroid_offset > 0.01096064 and Q.phi_7 > -0.05731659:
        z += 211.4695 * (Q.centroid_offset - 0.01096064) * (Q.phi_7 - -0.05731659)
    if Q.sj2_dr > 0.1872617 and Q.pt_6 < 27.57812:
        z += -0.5001776 * (Q.sj2_dr - 0.1872617) * (27.57812 - Q.pt_6)
    if Q.centroid_offset > 0.01096064 and Q.abseta_0 < 0.07861328:
        z += 890.145 * (Q.centroid_offset - 0.01096064) * (0.07861328 - Q.abseta_0)
    if Q.mean_eta < -0.004664942 and Q.phi_5 > 0.07385864:
        z += 892.9192 * (-0.004664942 - Q.mean_eta) * (Q.phi_5 - 0.07385864)
    if Q.centroid_offset > 0.01096064 and Q.pair_mass_0_5 > 16.26599:
        z += 5.726107 * (Q.centroid_offset - 0.01096064) * (Q.pair_mass_0_5 - 16.26599)
    if Q.sj2_dr > 0.1872617 and Q.n_dr_0p2_0p4 < 2.0:
        z += 6.143328 * (Q.sj2_dr - 0.1872617) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.tau1 > 0.05356915 and Q.mass_top3 < 50.3522:
        z += -0.1802431 * (Q.tau1 - 0.05356915) * (50.3522 - Q.mass_top3)
    if Q.lam2 > 0.001130645 and Q.pt_6 < 56.53125:
        z += 1.734954 * (Q.lam2 - 0.001130645) * (56.53125 - Q.pt_6)
    if Q.sd_mass > 62.55 and Q.D2_b2 < 0.9206502:
        z += 0.03409939 * (Q.sd_mass - 62.55) * (0.9206502 - Q.D2_b2)
    if Q.e2 > 0.06344108 and Q.z_top5 > 0.7773372:
        z += -586.2537 * (Q.e2 - 0.06344108) * (Q.z_top5 - 0.7773372)
    if Q.mean_eta < -0.004664942 and Q.z_6 > 0.06727211:
        z += 114.6869 * (-0.004664942 - Q.mean_eta) * (Q.z_6 - 0.06727211)
    if Q.sj2_dr > 0.1872617 and Q.dr_3 < 0.05268713:
        z += -343.0044 * (Q.sj2_dr - 0.1872617) * (0.05268713 - Q.dr_3)
    if Q.mass > 64.61873 and Q.phi_6 > 0.1192017:
        z += 0.5830381 * (Q.mass - 64.61873) * (Q.phi_6 - 0.1192017)
    if Q.tau1 > 0.05356915 and Q.pt_5 > 59.125:
        z += 0.8758286 * (Q.tau1 - 0.05356915) * (Q.pt_5 - 59.125)
    if Q.e2 > 0.06344108 and Q.dr0_3 < 0.1332952:
        z += -662.2707 * (Q.e2 - 0.06344108) * (0.1332952 - Q.dr0_3)
    if Q.max_dr > 0.1027585 and Q.phi_4 > 0.1096802:
        z += 31.50304 * (Q.max_dr - 0.1027585) * (Q.phi_4 - 0.1096802)
    if Q.mass > 64.61873 and Q.eta_7 > -0.0803833:
        z += 0.1724377 * (Q.mass - 64.61873) * (Q.eta_7 - -0.0803833)
    if Q.mean_eta > 0.01772426 and Q.mean_phi < 0.02612796:
        z += 395.165 * (Q.mean_eta - 0.01772426) * (0.02612796 - Q.mean_phi)
    if Q.sum_z_dr > 0.07608178 and Q.eta_0 > 0.07952881:
        z += -73.9512 * (Q.sum_z_dr - 0.07608178) * (Q.eta_0 - 0.07952881)
    if Q.lam2 > 0.001130645 and Q.z_6 > 0.05441452:
        z += 351.4824 * (Q.lam2 - 0.001130645) * (Q.z_6 - 0.05441452)
    if Q.lam1 > 0.008375572 and Q.pt_6 < 24.42188:
        z += -51.03083 * (Q.lam1 - 0.008375572) * (24.42188 - Q.pt_6)
    if Q.lam1 > 0.01643375 and Q.n_for_90pct < 6.0:
        z += 1348.678 * (Q.lam1 - 0.01643375) * (6.0 - Q.n_for_90pct)
    if Q.max_dr > 0.1027585 and Q.z_6 < 0.06081235:
        z += 7.537941 * (Q.max_dr - 0.1027585) * (0.06081235 - Q.z_6)
    if Q.max_dr > 0.1027585 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += -13.13876 * (Q.max_dr - 0.1027585) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.e2 > 0.06344108 and Q.zdr_3 < 0.01257746:
        z += 829.2486 * (Q.e2 - 0.06344108) * (0.01257746 - Q.zdr_3)
    return max(0.0, z)


def neuron_4(Q):
    z = -0.9301115
    if Q.N2 < 0.2233283:
        z += -43.28057 * Q.N2 + 9.665774
    if Q.lam2 < 0.000537286:
        z += 2224.421 * Q.lam2 - 1.19515
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -112.0442 * Q.mass_over_sum_pt + 10.13034
    if Q.lam1_plus_lam2 < 0.001653836:
        z += -368.4076 * Q.lam1_plus_lam2 + 0.6092859
    if Q.lam1_plus_lam2 >= 0.003562611:
        z += 320.2636 * Q.lam1_plus_lam2 - 1.140975
    if Q.e2 < 0.04447357:
        z += 83.91647 * Q.e2 - 4.219709
    if 0.04447357 <= Q.e2 < 0.05028464:
        z += 150.7579 * Q.e2 - 7.192387
    if Q.e2 >= 0.05028464:
        z += 66.84145 * Q.e2 - 2.972678
    if Q.sum_pt < 739.5:
        z += -0.001055059 * Q.sum_pt + 0.7802161
    if Q.max_dr < 0.1117619:
        z += -0.2227964 * Q.max_dr - 0.9048023
    if 0.1117619 <= Q.max_dr < 0.1598486:
        z += 14.18459 * Q.max_dr - 2.514999
    if 0.1598486 <= Q.max_dr < 0.177305:
        z += 27.56005 * Q.max_dr - 4.653048
    if Q.max_dr >= 0.177305:
        z += 13.37546 * Q.max_dr - 2.138049
    if Q.C2 < 0.06729223:
        z += -48.06909 * Q.C2 + 3.234676
    if Q.sum_z_dr >= 0.08723651:
        z += -12.63866 * Q.sum_z_dr + 1.102553
    if 44.82259 <= Q.sd_mass < 74.57663:
        z += 0.0945612 * Q.sd_mass - 4.238478
    if Q.sd_mass >= 74.57663:
        z += 0.0337705 * Q.sd_mass + 0.2950871
    if Q.e3 < 0.0001869378:
        z += -6069.409 * Q.e3 + 1.134602
    if Q.sj3_dr_max < 0.213399:
        z += 6.064585 * Q.sj3_dr_max + 0.08437438
    if 0.213399 <= Q.sj3_dr_max < 0.233678:
        z += 14.47027 * Q.sj3_dr_max - 1.709391
    if 0.233678 <= Q.sj3_dr_max < 0.3456459:
        z += -14.93279 * Q.sj3_dr_max + 5.161458
    if Q.sum_z_dr2 < 0.008678045:
        z += 488.0726 * Q.sum_z_dr2 - 4.235516
    if Q.sum_z_dr2_top5 < 0.007164202:
        z += -81.41479 * Q.sum_z_dr2_top5 + 0.583272
    if Q.centroid_offset < 0.04990367:
        z += 41.25755 * Q.centroid_offset - 2.058903
    if Q.sum_z_dr2_top2 < 0.007639643:
        z += -231.4138 * Q.sum_z_dr2_top2 + 1.767919
    if Q.mass_top5 < 37.76455:
        z += 0.001565497 * Q.mass_top5 - 0.05912027
    if Q.mass >= 76.6557:
        z += -0.06886193 * Q.mass + 5.27866
    if Q.sum_zz_dr2 < 0.01165737:
        z += -3137.108 * Q.sum_zz_dr2 + 36.57044
    if Q.tau1 < 0.07283629:
        z += -26.33183 * Q.tau1 + 1.917913
    if Q.mass_over_sum_pt_sq < 0.0116609:
        z += 2744.503 * Q.mass_over_sum_pt_sq - 32.00338
    if Q.LHA < 0.3467135:
        z += -1.897269 * Q.LHA + 0.6578087
    if Q.N2 < 0.2233283 and Q.mass < 62.55:
        z += -0.3780281 * (0.2233283 - Q.N2) * (62.55 - Q.mass)
    if Q.N2 < 0.2233283 and Q.sum_zz_dr2 > 0.01165737:
        z += -809.7366 * (0.2233283 - Q.N2) * (Q.sum_zz_dr2 - 0.01165737)
    if Q.N2 < 0.2233283 and Q.pt_7 < 53.4375:
        z += -0.2346279 * (0.2233283 - Q.N2) * (53.4375 - Q.pt_7)
    if Q.N2 < 0.2233283 and Q.mean_phi < -0.009352575:
        z += 353.8632 * (0.2233283 - Q.N2) * (-0.009352575 - Q.mean_phi)
    if Q.N2 < 0.2233283 and Q.eccentricity > 0.7117266:
        z += -97.3292 * (0.2233283 - Q.N2) * (Q.eccentricity - 0.7117266)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1218872:
        z += -23.95201 * (0.2233283 - Q.N2) * (0.1218872 - Q.abseta_7)
    if Q.e3 < 0.0001869378 and Q.mean_eta < -0.01284493:
        z += 372812.8 * (0.0001869378 - Q.e3) * (-0.01284493 - Q.mean_eta)
    if Q.N2 < 0.2233283 and Q.zdr_0 < 0.004918231:
        z += -4656.235 * (0.2233283 - Q.N2) * (0.004918231 - Q.zdr_0)
    if Q.sd_mass > 44.82259 and Q.centroid_offset > 0.001308549:
        z += -1.317013 * (Q.sd_mass - 44.82259) * (Q.centroid_offset - 0.001308549)
    if Q.sd_mass > 44.82259 and Q.sd_zg < 0.275762:
        z += -0.1031091 * (Q.sd_mass - 44.82259) * (0.275762 - Q.sd_zg)
    if Q.lam1_plus_lam2 > 0.003562611 and Q.sj3_z3 < 0.1057566:
        z += -197.9222 * (Q.lam1_plus_lam2 - 0.003562611) * (0.1057566 - Q.sj3_z3)
    if Q.N2 < 0.2233283 and Q.pt_2 > 69.875:
        z += -0.09018037 * (0.2233283 - Q.N2) * (Q.pt_2 - 69.875)
    if Q.sj3_dr_max < 0.213399 and Q.mean_phi2 > 0.008921136:
        z += 2009.188 * (0.213399 - Q.sj3_dr_max) * (Q.mean_phi2 - 0.008921136)
    if Q.lam2 < 0.000537286 and Q.dr01 < 0.1410336:
        z += -11595.37 * (0.000537286 - Q.lam2) * (0.1410336 - Q.dr01)
    if Q.sj3_dr_max < 0.3456459 and Q.pair_mass_0_6 > 15.55293:
        z += 1.604629 * (0.3456459 - Q.sj3_dr_max) * (Q.pair_mass_0_6 - 15.55293)
    if Q.N2 < 0.2233283 and Q.phi_6 > 0.1192017:
        z += 2.705077 * (0.2233283 - Q.N2) * (Q.phi_6 - 0.1192017)
    if Q.max_dr > 0.1598486 and Q.C2_b2 < 0.0006435798:
        z += -38064.96 * (Q.max_dr - 0.1598486) * (0.0006435798 - Q.C2_b2)
    if Q.sum_z_dr2_top2 < 0.007639643 and Q.C2_b2 > 0.0006435798:
        z += 3442.74 * (0.007639643 - Q.sum_z_dr2_top2) * (Q.C2_b2 - 0.0006435798)
    if Q.sj3_dr_max < 0.3456459 and Q.eta_5 < -0.1135254:
        z += 254.1806 * (0.3456459 - Q.sj3_dr_max) * (-0.1135254 - Q.eta_5)
    if Q.max_dr < 0.177305 and Q.ptdr0_4 > 8.939062:
        z += 3.006578 * (0.177305 - Q.max_dr) * (Q.ptdr0_4 - 8.939062)
    if Q.lam2 < 0.000537286 and Q.phi_0 > 0.004838562:
        z += 5522.972 * (0.000537286 - Q.lam2) * (Q.phi_0 - 0.004838562)
    if Q.lam2 < 0.000537286 and Q.mean_phi < -0.01275329:
        z += -67008.7 * (0.000537286 - Q.lam2) * (-0.01275329 - Q.mean_phi)
    if Q.e3 < 0.0001869378 and Q.mean_phi < -0.009352575:
        z += 368992.7 * (0.0001869378 - Q.e3) * (-0.009352575 - Q.mean_phi)
    if Q.sj3_dr_max < 0.3456459 and Q.absphi_0 > 0.1056549:
        z += 620.3489 * (0.3456459 - Q.sj3_dr_max) * (Q.absphi_0 - 0.1056549)
    if Q.sum_zz_dr2 < 0.01165737 and Q.ptdr0_7 > 2.691363:
        z += 2.255512 * (0.01165737 - Q.sum_zz_dr2) * (Q.ptdr0_7 - 2.691363)
    if Q.sj3_dr_max < 0.3456459 and Q.dr1_5 > 0.1518778:
        z += 154.9992 * (0.3456459 - Q.sj3_dr_max) * (Q.dr1_5 - 0.1518778)
    if Q.N2 < 0.2233283 and Q.pt_1 < 169.875:
        z += 0.0357863 * (0.2233283 - Q.N2) * (169.875 - Q.pt_1)
    if Q.sj3_dr_max < 0.233678 and Q.dr0_3 > 0.1815989:
        z += 2244.012 * (0.233678 - Q.sj3_dr_max) * (Q.dr0_3 - 0.1815989)
    if Q.N2 < 0.2233283 and Q.pt_6 < 24.42188:
        z += 0.7610701 * (0.2233283 - Q.N2) * (24.42188 - Q.pt_6)
    if Q.mass > 76.6557 and Q.sj2_zsoft > 0.4448372:
        z += -0.8768404 * (Q.mass - 76.6557) * (Q.sj2_zsoft - 0.4448372)
    if Q.sd_mass > 44.82259 and Q.zdr_5 < 0.005440034:
        z += 6.935451 * (Q.sd_mass - 44.82259) * (0.005440034 - Q.zdr_5)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.7801241
    if Q.LHA < 0.2160559:
        z += -16.01182 * Q.LHA + 3.459447
    if Q.z_7 < 0.02807091:
        z += -259.1974 * Q.z_7 + 9.303134
    if 0.02807091 <= Q.z_7 < 0.04939969:
        z += -73.66031 * Q.z_7 + 4.094938
    if 0.04939969 <= Q.z_7 < 0.07148865:
        z += -20.65023 * Q.z_7 + 1.476257
    if 6.701242 <= Q.log_sum_pt < 6.896095:
        z += -10.38844 * Q.log_sum_pt + 69.61549
    if Q.log_sum_pt >= 6.896095:
        z += -31.70406 * Q.log_sum_pt + 216.61
    if Q.sum_zz_dr2 < 0.002074109:
        z += -30.3275 * Q.sum_zz_dr2 + 0.3739865
    if 0.002074109 <= Q.sum_zz_dr2 < 0.005284669:
        z += -96.89399 * Q.sum_zz_dr2 + 0.5120526
    if Q.z_6 < 0.02886576:
        z += -188.1401 * Q.z_6 + 5.430806
    if Q.sum_z_dr2 < 0.001653836:
        z += -911.5094 * Q.sum_z_dr2 + 1.507487
    if Q.sum_z_dr < 0.007673833:
        z += -633.4948 * Q.sum_z_dr + 4.861333
    if Q.pt_7 < 37.15625:
        z += 0.006692063 * Q.pt_7 - 0.248652
    if Q.zdr_0 < 0.0211821:
        z += 71.64442 * Q.zdr_0 - 1.517579
    if Q.pair_mass_0_4 >= 23.26771:
        z += -0.1017402 * Q.pair_mass_0_4 + 2.367262
    if 430.75 <= Q.sum_pt_top5 < 687.4375:
        z += 0.002029832 * Q.sum_pt_top5 - 0.8743502
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.005918815 * Q.sum_pt_top5 - 3.547783
    if Q.pt_5 < 24.57812:
        z += -0.4620169 * Q.pt_5 + 11.35551
    if Q.lam1_plus_lam2 < 0.003562611:
        z += -403.1845 * Q.lam1_plus_lam2 + 1.43639
    if Q.centroid_offset < 0.006789738:
        z += -239.877 * Q.centroid_offset + 1.628702
    if Q.pt_3 < 45.05938:
        z += -0.1169149 * Q.pt_3 + 5.268113
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 293.799 * Q.sum_z_dr2_top3 - 0.6321285
    if Q.z_7 < 0.04939969 and Q.mass_top5 < 62.55:
        z += 0.858248 * (0.04939969 - Q.z_7) * (62.55 - Q.mass_top5)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -82.55293 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.sum_zz_dr2 < 0.005284669 and Q.centroid_offset < 0.01437952:
        z += 34148.23 * (0.005284669 - Q.sum_zz_dr2) * (0.01437952 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.centroid_offset < 0.03117077:
        z += 500.9412 * (0.07148865 - Q.z_7) * (0.03117077 - Q.centroid_offset)
    if Q.z_7 < 0.04939969 and Q.sum_pt_top2 < 501.625:
        z += -0.6695192 * (0.04939969 - Q.z_7) * (501.625 - Q.sum_pt_top2)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.05744392:
        z += 1466.639 * (Q.log_sum_pt - 6.896095) * (0.05744392 - Q.D2_b2)
    if Q.z_7 < 0.07148865 and Q.mass_top3 < 40.2:
        z += 0.5220625 * (0.07148865 - Q.z_7) * (40.2 - Q.mass_top3)
    if Q.z_7 < 0.07148865 and Q.C3 < 0.03532852:
        z += 763.3242 * (0.07148865 - Q.z_7) * (0.03532852 - Q.C3)
    if Q.sum_z_dr2 < 0.001653836 and Q.centroid_offset < 0.02355416:
        z += 114227.4 * (0.001653836 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.sum_zz_dr2 < 0.005284669 and Q.n_pt_above_50 > 6.0:
        z += -105.0836 * (0.005284669 - Q.sum_zz_dr2) * (Q.n_pt_above_50 - 6.0)
    if Q.LHA < 0.2160559 and Q.lam1 < 0.001503553:
        z += -4197.684 * (0.2160559 - Q.LHA) * (0.001503553 - Q.lam1)
    if Q.z_7 < 0.07148865 and Q.C2_b2 > 0.002354783:
        z += -733.6553 * (0.07148865 - Q.z_7) * (Q.C2_b2 - 0.002354783)
    if Q.log_sum_pt > 6.701242 and Q.sj3_dr_max > 0.169029:
        z += -8.172404 * (Q.log_sum_pt - 6.701242) * (Q.sj3_dr_max - 0.169029)
    if Q.log_sum_pt > 6.701242 and Q.dr_2 < 0.01341502:
        z += 1165.202 * (Q.log_sum_pt - 6.701242) * (0.01341502 - Q.dr_2)
    if Q.log_sum_pt > 6.896095 and Q.dr_2 < 0.02270492:
        z += -1902.249 * (Q.log_sum_pt - 6.896095) * (0.02270492 - Q.dr_2)
    if Q.LHA < 0.2160559 and Q.n_dr_0p2_0p4 > 0.0:
        z += 1.630102 * (0.2160559 - Q.LHA) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.sum_pt_top5 > 430.75 and Q.centroid_offset > 0.009480685:
        z += 0.1893547 * (Q.sum_pt_top5 - 430.75) * (Q.centroid_offset - 0.009480685)
    if Q.z_7 < 0.04939969 and Q.dr_2 < 0.009661512:
        z += 5849.218 * (0.04939969 - Q.z_7) * (0.009661512 - Q.dr_2)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi > 0.002834884:
        z += -1891.203 * (Q.log_sum_pt - 6.701242) * (Q.mean_phi - 0.002834884)
    if Q.sum_pt_top5 > 430.75 and Q.pt1_dr01 < 28.39396:
        z += 8.69476e-05 * (Q.sum_pt_top5 - 430.75) * (28.39396 - Q.pt1_dr01)
    if Q.sum_pt_top5 > 430.75 and Q.sj2_dr > 0.1682655:
        z += 0.00851293 * (Q.sum_pt_top5 - 430.75) * (Q.sj2_dr - 0.1682655)
    if Q.z_6 < 0.02886576 and Q.mean_phi2 > 0.001125075:
        z += -90459.06 * (0.02886576 - Q.z_6) * (Q.mean_phi2 - 0.001125075)
    if Q.pt_7 < 37.15625 and Q.pt_5 > 24.57812:
        z += 0.000831367 * (37.15625 - Q.pt_7) * (Q.pt_5 - 24.57812)
    if Q.sum_pt_top5 > 430.75 and Q.pt_6 < 46.125:
        z += 0.0001740997 * (Q.sum_pt_top5 - 430.75) * (46.125 - Q.pt_6)
    if Q.LHA < 0.2160559 and Q.z_3 < 0.07235619:
        z += 488.2774 * (0.2160559 - Q.LHA) * (0.07235619 - Q.z_3)
    if Q.pair_mass_0_4 > 23.26771 and Q.eccentricity > 0.9704496:
        z += 3.437301 * (Q.pair_mass_0_4 - 23.26771) * (Q.eccentricity - 0.9704496)
    if Q.zdr_0 < 0.0211821 and Q.phi_0 > -0.0297699:
        z += -201.9133 * (0.0211821 - Q.zdr_0) * (Q.phi_0 - -0.0297699)
    if Q.z_7 < 0.07148865 and Q.mean_phi2 < 0.002127561:
        z += 10576.86 * (0.07148865 - Q.z_7) * (0.002127561 - Q.mean_phi2)
    if Q.pt_5 < 24.57812 and Q.mean_eta2 > 1.797789e-05:
        z += -160.5574 * (24.57812 - Q.pt_5) * (Q.mean_eta2 - 1.797789e-05)
    if Q.pt_5 < 24.57812 and Q.n_pt_above_5 < 8.0:
        z += 0.2521673 * (24.57812 - Q.pt_5) * (8.0 - Q.n_pt_above_5)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta2 < 9.030369e-05:
        z += 73108.59 * (Q.log_sum_pt - 6.701242) * (9.030369e-05 - Q.mean_eta2)
    if Q.pair_mass_0_4 > 23.26771 and Q.mean_eta2 < 0.01423545:
        z += 2.056837 * (Q.pair_mass_0_4 - 23.26771) * (0.01423545 - Q.mean_eta2)
    if Q.lam1_plus_lam2 < 0.003562611 and Q.sj3_dr13 > 0.181053:
        z += 11526.09 * (0.003562611 - Q.lam1_plus_lam2) * (Q.sj3_dr13 - 0.181053)
    if Q.sum_zz_dr2 < 0.005284669 and Q.planar_flow < 0.3220738:
        z += 975.309 * (0.005284669 - Q.sum_zz_dr2) * (0.3220738 - Q.planar_flow)
    if Q.zdr_0 < 0.0211821 and Q.sj3_dr13 > 0.04995258:
        z += -238.1093 * (0.0211821 - Q.zdr_0) * (Q.sj3_dr13 - 0.04995258)
    if Q.centroid_offset < 0.006789738 and Q.pt_5 < 56.4375:
        z += 8.210005 * (0.006789738 - Q.centroid_offset) * (56.4375 - Q.pt_5)
    if Q.sum_zz_dr2 < 0.005284669 and Q.dr0_5 > 0.1122797:
        z += 2616.719 * (0.005284669 - Q.sum_zz_dr2) * (Q.dr0_5 - 0.1122797)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta < -0.003031633:
        z += -1539.295 * (Q.log_sum_pt - 6.701242) * (-0.003031633 - Q.mean_eta)
    if Q.z_7 < 0.04939969 and Q.mean_eta > 0.01271871:
        z += -1072.389 * (0.04939969 - Q.z_7) * (Q.mean_eta - 0.01271871)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.09258154
    if 0.00809236 <= Q.centroid_offset < 0.01837778:
        z += -13.8594 * Q.centroid_offset + 0.1121553
    if Q.centroid_offset >= 0.01837778:
        z += -25.65084 * Q.centroid_offset + 0.3288558
    if Q.lam1_plus_lam2 < 0.01323868:
        z += 465.1867 * Q.lam1_plus_lam2 - 6.158455
    if Q.mass < 45.595:
        z += -0.03338655 * Q.mass + 2.090645
    if 45.595 <= Q.mass < 60.63098:
        z += -0.03780171 * Q.mass + 2.291954
    if Q.pt_6 < 29.90625:
        z += -0.08502725 * Q.pt_6 + 2.735736
    if 29.90625 <= Q.pt_6 < 39.75:
        z += -0.008280598 * Q.pt_6 + 0.4405311
    if 39.75 <= Q.pt_6 < 41.21875:
        z += -0.07583138 * Q.pt_6 + 3.125675
    if Q.lam2 < 0.001130645:
        z += -194.5901 * Q.lam2 + 1.183519
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -423.0092 * Q.lam2 + 1.44178
    if Q.tau1 < 0.1136369:
        z += -20.32078 * Q.tau1 + 2.30919
    if Q.sj3_dr_min >= 0.1278212:
        z += -9.600209 * Q.sj3_dr_min + 1.22711
    if Q.lam1 < 0.00595415:
        z += -1067.497 * Q.lam1 + 8.415046
    if 0.00595415 <= Q.lam1 < 0.00733008:
        z += -856.6702 * Q.lam1 + 7.159751
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += -188.3518 * Q.lam1 + 2.260923
    if Q.sj3_pair_mass_min >= 4.501727:
        z += -0.0708179 * Q.sj3_pair_mass_min + 0.3188028
    if Q.sum_pt < 615.875:
        z += -0.005282152 * Q.sum_pt + 3.253146
    if Q.sum_zz_dr2 < 0.0030133:
        z += -504.5152 * Q.sum_zz_dr2 + 1.520256
    if Q.sj3_dr13 >= 0.181053:
        z += 0.9708341 * Q.sj3_dr13 - 0.1757724
    if Q.eccentricity >= 0.927072:
        z += -0.1516249 * Q.eccentricity + 0.1405672
    if Q.sj3_dr_max < 0.1789613:
        z += -11.23332 * Q.sj3_dr_max + 1.459565
    if 0.1789613 <= Q.sj3_dr_max < 0.1879486:
        z += 4.5056 * Q.sj3_dr_max - 1.357094
    if 0.1879486 <= Q.sj3_dr_max < 0.3012016:
        z += 5.857874 * Q.sj3_dr_max - 1.611252
    if Q.sj3_dr_max >= 0.3012016:
        z += 1.352273 * Q.sj3_dr_max - 0.2541578
    if Q.max_dr < 0.1452311:
        z += 7.1924 * Q.max_dr - 1.04456
    if Q.mean_eta >= 0.02644207:
        z += 33.71855 * Q.mean_eta - 0.8915882
    if Q.e3 < 0.0005116989:
        z += -480.5167 * Q.e3 + 0.2458799
    if Q.sum_z_dr2 < 0.008678045:
        z += 580.8635 * Q.sum_z_dr2 - 5.04076
    if Q.C2_b2 < 0.02415398:
        z += -23.88425 * Q.C2_b2 + 0.5768999
    if Q.sj2_dr < 0.1872617:
        z += -8.63154 * Q.sj2_dr + 1.616357
    if Q.sj2_zsoft < 0.04176067:
        z += -17.30042 * Q.sj2_zsoft + 0.7224774
    if Q.centroid_offset > 0.00809236 and Q.sj3_pair_mass_min > 6.811308:
        z += -1.408232 * (Q.centroid_offset - 0.00809236) * (Q.sj3_pair_mass_min - 6.811308)
    if Q.pt_6 < 41.21875 and Q.log_sum_pt < 6.766778:
        z += 0.6208867 * (41.21875 - Q.pt_6) * (6.766778 - Q.log_sum_pt)
    if Q.centroid_offset > 0.00809236 and Q.psi_0p1 > 0.4008925:
        z += -72.13773 * (Q.centroid_offset - 0.00809236) * (Q.psi_0p1 - 0.4008925)
    if Q.centroid_offset > 0.01837778 and Q.C2 < 0.02702951:
        z += -865.9327 * (Q.centroid_offset - 0.01837778) * (0.02702951 - Q.C2)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.008921136:
        z += -1806.936 * (Q.centroid_offset - 0.01837778) * (0.008921136 - Q.mean_phi2)
    if Q.lam1 < 0.01200373 and Q.planar_flow < 0.2534037:
        z += -191.9176 * (0.01200373 - Q.lam1) * (0.2534037 - Q.planar_flow)
    if Q.pt_6 < 41.21875 and Q.z_7 > 0.02320757:
        z += -2.916301 * (41.21875 - Q.pt_6) * (Q.z_7 - 0.02320757)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.mean_eta < -0.02665591:
        z += 28698.95 * (0.01323868 - Q.lam1_plus_lam2) * (-0.02665591 - Q.mean_eta)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.mean_phi < -0.02594505:
        z += 11295.99 * (0.01323868 - Q.lam1_plus_lam2) * (-0.02594505 - Q.mean_phi)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.mean_phi > 0.02612796:
        z += 10452.2 * (0.01323868 - Q.lam1_plus_lam2) * (Q.mean_phi - 0.02612796)
    if Q.sum_pt < 615.875 and Q.mean_phi > 0.01251955:
        z += 0.3398638 * (615.875 - Q.sum_pt) * (Q.mean_phi - 0.01251955)
    if Q.pt_6 < 29.90625 and Q.n_pt_above_10 < 8.0:
        z += 0.04580266 * (29.90625 - Q.pt_6) * (8.0 - Q.n_pt_above_10)
    if Q.mass < 60.63098 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -0.5552055 * (60.63098 - Q.mass) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sum_pt < 615.875 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.003424373 * (615.875 - Q.sum_pt) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.sj3_dr_max < 0.1789613 and Q.tau21_b2 < 0.1230713:
        z += 271.9391 * (0.1789613 - Q.sj3_dr_max) * (0.1230713 - Q.tau21_b2)
    if Q.lam2 < 0.003408389 and Q.n_dr_0_0p05 < 3.0:
        z += 5.211493 * (0.003408389 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.max_dr < 0.1452311 and Q.mean_phi > 0.004406178:
        z += -1342.01 * (0.1452311 - Q.max_dr) * (Q.mean_phi - 0.004406178)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.eta_7 < -0.0803833:
        z += 126.5595 * (0.01323868 - Q.lam1_plus_lam2) * (-0.0803833 - Q.eta_7)
    if Q.pt_6 < 41.21875 and Q.dr1_7 < 0.1343804:
        z += 0.002899345 * (41.21875 - Q.pt_6) * (0.1343804 - Q.dr1_7)
    if Q.lam1 < 0.01200373 and Q.mean_eta > 0.02644207:
        z += 13049.2 * (0.01200373 - Q.lam1) * (Q.mean_eta - 0.02644207)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.00433128:
        z += -20059.4 * (Q.centroid_offset - 0.01837778) * (0.00433128 - Q.mean_phi2)
    if Q.sum_pt < 615.875 and Q.z_3 < 0.05101919:
        z += 42.23838 * (615.875 - Q.sum_pt) * (0.05101919 - Q.z_3)
    if Q.pt_6 < 41.21875 and Q.M3 > 0.0782171:
        z += 6.240718 * (41.21875 - Q.pt_6) * (Q.M3 - 0.0782171)
    if Q.lam1 < 0.00733008 and Q.n_dr_0p2_0p4 < 1.0:
        z += 250.5237 * (0.00733008 - Q.lam1) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.sj3_dr13 > 0.181053 and Q.abseta_5 < 0.02658081:
        z += -408.5736 * (Q.sj3_dr13 - 0.181053) * (0.02658081 - Q.abseta_5)
    if Q.mass < 45.595 and Q.eta_3 > -0.00592804:
        z += -0.9029297 * (45.595 - Q.mass) * (Q.eta_3 - -0.00592804)
    if Q.centroid_offset > 0.01837778 and Q.eta_0 < 0.02980347:
        z += 27.54928 * (Q.centroid_offset - 0.01837778) * (0.02980347 - Q.eta_0)
    if Q.mean_eta > 0.02644207 and Q.M3 > 0.06688759:
        z += 1033.158 * (Q.mean_eta - 0.02644207) * (Q.M3 - 0.06688759)
    if Q.sj3_pair_mass_min > 4.501727 and Q.n_dr_0p2_0p4 > 1.0:
        z += 0.002885837 * (Q.sj3_pair_mass_min - 4.501727) * (Q.n_dr_0p2_0p4 - 1.0)
    if Q.sj3_dr_max > 0.1879486 and Q.abseta_2 < 0.03250122:
        z += -438.4796 * (Q.sj3_dr_max - 0.1879486) * (0.03250122 - Q.abseta_2)
    return max(0.0, z)


def neuron_7(Q):
    z = 8.509068
    if Q.planar_flow < 0.1950135:
        z += -2.098712 * Q.planar_flow + 0.4092773
    if Q.sum_z_dr2_top2 < 8.10414e-05:
        z += 23779.95 * Q.sum_z_dr2_top2 - 2.711343
    if 8.10414e-05 <= Q.sum_z_dr2_top2 < 0.001056655:
        z += 803.7831 * Q.sum_z_dr2_top2 - 0.8493216
    if Q.sum_z_dr2 < 0.0009641429:
        z += 2764.534 * Q.sum_z_dr2 - 4.32431
    if 0.0009641429 <= Q.sum_z_dr2 < 0.001653836:
        z += 638.4164 * Q.sum_z_dr2 - 2.274429
    if 0.001653836 <= Q.sum_z_dr2 < 0.003562611:
        z += 1028.468 * Q.sum_z_dr2 - 2.919511
    if 0.003562611 <= Q.sum_z_dr2 < 0.004372139:
        z += 390.0518 * Q.sum_z_dr2 - 0.6450818
    if 0.004372139 <= Q.sum_z_dr2 < 0.007520088:
        z += 10.34024 * Q.sum_z_dr2 + 1.01507
    if 0.007520088 <= Q.sum_z_dr2 < 0.01323868:
        z += -1170.225 * Q.sum_z_dr2 + 9.893022
    if Q.sum_z_dr2 >= 0.01323868:
        z += -360.1646 * Q.sum_z_dr2 - 0.8310997
    if 0.01109984 <= Q.mass_over_sum_pt < 0.07269073:
        z += -57.88889 * Q.mass_over_sum_pt + 0.6425575
    if 0.07269073 <= Q.mass_over_sum_pt < 0.08475161:
        z += 99.64731 * Q.mass_over_sum_pt - 10.80886
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 244.4205 * Q.mass_over_sum_pt - 23.07863
    if 0.09041383 <= Q.mass_over_sum_pt < 0.1079857:
        z += -67.68704 * Q.mass_over_sum_pt + 5.140212
    if Q.mass_over_sum_pt >= 0.1079857:
        z += -77.31976 * Q.mass_over_sum_pt + 6.180408
    if Q.tau1 < 0.05356915:
        z += -36.33354 * Q.tau1 + 1.946357
    if Q.sum_z_dr < 0.04081947:
        z += 148.0519 * Q.sum_z_dr - 10.4026
    if 0.04081947 <= Q.sum_z_dr < 0.08723651:
        z += 93.91378 * Q.sum_z_dr - 8.192711
    if 36.22941 <= Q.mass < 76.6557:
        z += 0.03791696 * Q.mass - 1.373709
    if Q.mass >= 76.6557:
        z += -0.05188311 * Q.mass + 5.509978
    if Q.z_dr_0p1_0p2 < 0.1585582:
        z += -1.560636 * Q.z_dr_0p1_0p2 + 0.2474516
    if Q.centroid_offset < 0.02076709:
        z += 7.620223 * Q.centroid_offset - 0.1582499
    if 0.03117077 <= Q.centroid_offset < 0.03776099:
        z += -103.5591 * Q.centroid_offset + 3.228017
    if Q.centroid_offset >= 0.03776099:
        z += -114.0265 * Q.centroid_offset + 3.623277
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -1080.969 * Q.lam1_plus_lam2 + 6.04293
    if Q.lam1_plus_lam2 >= 0.008678045:
        z += 332.4749 * Q.lam1_plus_lam2 - 6.223001
    if Q.z_7 >= 0.03243272:
        z += 26.37185 * Q.z_7 - 0.8553108
    if 0.03556091 <= Q.e2 < 0.05028464:
        z += 94.6769 * Q.e2 - 3.366797
    if Q.e2 >= 0.05028464:
        z += -4.110649 * Q.e2 + 1.6007
    if Q.sj2_dr < 0.1294903:
        z += -17.02031 * Q.sj2_dr + 1.440755
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -6.915142 * Q.sj2_dr + 0.1322332
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 34.47649 * Q.sj2_dr - 6.456126
    if Q.sum_zz_dr2 < 0.001101266:
        z += -1360.765 * Q.sum_zz_dr2 + 1.498565
    if Q.LHA < 0.3033137:
        z += -13.72535 * Q.LHA + 4.163087
    if Q.pt_7 < 29.04219:
        z += 0.07130438 * Q.pt_7 - 2.070835
    if 0.2037854 <= Q.sd_rg < 0.2787955:
        z += -8.314321 * Q.sd_rg + 1.694337
    if Q.sd_rg >= 0.2787955:
        z += -7.053049 * Q.sd_rg + 1.3427
    if Q.lam1 < 0.008375572:
        z += 40.24595 * Q.lam1 - 0.3370828
    if Q.e3 < 8.147744e-05:
        z += 3480.352 * Q.e3 - 0.2835702
    if Q.tau21_b2 < 0.009213932:
        z += -6.874434 * Q.tau21_b2 + 0.06334057
    if Q.planar_flow < 0.1950135 and Q.lam1_plus_lam2 > 0.007520088:
        z += -5294.124 * (0.1950135 - Q.planar_flow) * (Q.lam1_plus_lam2 - 0.007520088)
    if Q.sum_z_dr2_top2 < 0.001056655 and Q.centroid_offset > 0.006789738:
        z += 110546.5 * (0.001056655 - Q.sum_z_dr2_top2) * (Q.centroid_offset - 0.006789738)
    if Q.planar_flow < 0.1950135 and Q.pt_6 < 35.28125:
        z += -0.4651392 * (0.1950135 - Q.planar_flow) * (35.28125 - Q.pt_6)
    if Q.planar_flow < 0.1950135 and Q.sd_mass > 38.43971:
        z += 0.1976117 * (0.1950135 - Q.planar_flow) * (Q.sd_mass - 38.43971)
    if Q.z_dr_0p1_0p2 < 0.1585582 and Q.N3 < 2.080881:
        z += 0.3570794 * (0.1585582 - Q.z_dr_0p1_0p2) * (2.080881 - Q.N3)
    if Q.centroid_offset > 0.03117077 and Q.n_pt_above_50 > 4.0:
        z += -37.97382 * (Q.centroid_offset - 0.03117077) * (Q.n_pt_above_50 - 4.0)
    if Q.tau1 < 0.05356915 and Q.n_dr_0p05_0p1 > 5.0:
        z += -17.25099 * (0.05356915 - Q.tau1) * (Q.n_dr_0p05_0p1 - 5.0)
    if Q.centroid_offset < 0.02076709 and Q.sum_pt_top3 > 331.25:
        z += 0.03920801 * (0.02076709 - Q.centroid_offset) * (Q.sum_pt_top3 - 331.25)
    if Q.sum_z_dr < 0.08723651 and Q.mean_phi < 0.004406178:
        z += 1145.683 * (0.08723651 - Q.sum_z_dr) * (0.004406178 - Q.mean_phi)
    if Q.sum_zz_dr2 < 0.001101266 and Q.phi_1 < -0.04275513:
        z += -551464.3 * (0.001101266 - Q.sum_zz_dr2) * (-0.04275513 - Q.phi_1)
    if Q.sum_z_dr2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 3246.13 * (Q.sum_z_dr2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    if Q.sum_z_dr2 > 0.01323868 and Q.eccentricity > 0.9458207:
        z += 2662.326 * (Q.sum_z_dr2 - 0.01323868) * (Q.eccentricity - 0.9458207)
    if Q.e3 < 8.147744e-05 and Q.sj2_mass1 > 31.78116:
        z += 5003.6 * (8.147744e-05 - Q.e3) * (Q.sj2_mass1 - 31.78116)
    if Q.sum_zz_dr2 < 0.001101266 and Q.eta_2 < -0.06384277:
        z += -257548.5 * (0.001101266 - Q.sum_zz_dr2) * (-0.06384277 - Q.eta_2)
    if Q.centroid_offset > 0.03776099 and Q.pt_4 > 81.375:
        z += -56.13095 * (Q.centroid_offset - 0.03776099) * (Q.pt_4 - 81.375)
    if Q.sum_zz_dr2 < 0.001101266 and Q.phi_0 > 0.0402832:
        z += -477268.8 * (0.001101266 - Q.sum_zz_dr2) * (Q.phi_0 - 0.0402832)
    if Q.planar_flow < 0.1950135 and Q.pt_4 > 71.6875:
        z += 0.2132422 * (0.1950135 - Q.planar_flow) * (Q.pt_4 - 71.6875)
    if Q.sum_z_dr2 < 0.003562611 and Q.mean_eta < -0.006779839:
        z += 69920.16 * (0.003562611 - Q.sum_z_dr2) * (-0.006779839 - Q.mean_eta)
    if Q.centroid_offset < 0.02076709 and Q.C2_b2 < 0.004032342:
        z += -30020.76 * (0.02076709 - Q.centroid_offset) * (0.004032342 - Q.C2_b2)
    if Q.sum_z_dr < 0.08723651 and Q.C2_b2 < 0.004032342:
        z += 1626.976 * (0.08723651 - Q.sum_z_dr) * (0.004032342 - Q.C2_b2)
    if Q.centroid_offset < 0.02076709 and Q.tau21_b2 < 0.02656143:
        z += 4019.7 * (0.02076709 - Q.centroid_offset) * (0.02656143 - Q.tau21_b2)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.3374877
    if Q.sum_z_dr2 < 0.001653836:
        z += -114.3224 * Q.sum_z_dr2 + 0.8735097
    if 0.001653836 <= Q.sum_z_dr2 < 0.005019719:
        z += -231.2641 * Q.sum_z_dr2 + 1.066912
    if 0.005019719 <= Q.sum_z_dr2 < 0.006679471:
        z += 56.61607 * Q.sum_z_dr2 - 0.3781654
    if Q.tau1 < 0.05356915:
        z += 0.4329807 * Q.tau1 - 0.02319441
    if Q.LHA < 0.1967397:
        z += 12.77331 * Q.LHA - 2.513018
    if Q.log_sum_pt >= 6.701242:
        z += -2.995976 * Q.log_sum_pt + 20.07676
    if Q.mass < 21.78408:
        z += 0.03413506 * Q.mass - 0.8523646
    if 21.78408 <= Q.mass < 29.6447:
        z += 0.008433788 * Q.mass - 0.2924861
    if 29.6447 <= Q.mass < 49.6681:
        z += 0.002120967 * Q.mass - 0.1053444
    if Q.sum_z_dr < 0.06108601:
        z += 58.70135 * Q.sum_z_dr - 3.585832
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.0005260242 * Q.sum_pt_top5 - 0.3616088
    if Q.centroid_offset < 0.003343241:
        z += -180.6226 * Q.centroid_offset + 0.603865
    if Q.lam2 < 0.0001330621:
        z += 4479.199 * Q.lam2 - 0.5960115
    if Q.dr_0 < 0.003239423:
        z += 136.0738 * Q.dr_0 - 0.4408008
    if Q.z_dr_0_0p05 >= 0.8477313:
        z += -2.634166 * Q.z_dr_0_0p05 + 2.233065
    if Q.lam1_plus_lam2 < 0.003562611:
        z += -380.0864 * Q.lam1_plus_lam2 + 1.3541
    if Q.z_6 < 0.03448406:
        z += -8.969354 * Q.z_6 + 0.3092997
    if Q.mass_over_sum_pt < 0.03319429:
        z += -14.91812 * Q.mass_over_sum_pt + 0.4951962
    if Q.sj3_dr_max < 0.1426152:
        z += 5.732321 * Q.sj3_dr_max - 0.7311414
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += -1.542071 * Q.sj3_dr_max + 0.3062972
    if Q.sum_z_dr2 < 0.006679471 and Q.D2_b2 < 4.721224:
        z += -10.53718 * (0.006679471 - Q.sum_z_dr2) * (4.721224 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 18204.02 * (0.006679471 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.sum_z_dr2 < 0.005019719 and Q.log_sum_pt < 6.502799:
        z += -652.3951 * (0.005019719 - Q.sum_z_dr2) * (6.502799 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.006679471 and Q.planar_flow < 0.4007947:
        z += 165.9317 * (0.006679471 - Q.sum_z_dr2) * (0.4007947 - Q.planar_flow)
    if Q.sum_z_dr2 < 0.005019719 and Q.pt_7 < 43.5:
        z += -2.488067 * (0.005019719 - Q.sum_z_dr2) * (43.5 - Q.pt_7)
    if Q.sum_z_dr2 < 0.006679471 and Q.n_dr_0p05_0p1 > 0.0:
        z += -7.703774 * (0.006679471 - Q.sum_z_dr2) * (Q.n_dr_0p05_0p1 - 0.0)
    if Q.LHA < 0.1967397 and Q.mean_phi > -0.0008556753:
        z += 320.5241 * (0.1967397 - Q.LHA) * (Q.mean_phi - -0.0008556753)
    if Q.sum_z_dr2 < 0.005019719 and Q.m012 > 28.3451:
        z += -13.72443 * (0.005019719 - Q.sum_z_dr2) * (Q.m012 - 28.3451)
    if Q.sum_z_dr2 < 0.006679471 and Q.m012 > 8.921413:
        z += 2.155826 * (0.006679471 - Q.sum_z_dr2) * (Q.m012 - 8.921413)
    if Q.sum_z_dr2 < 0.006679471 and Q.n_dr_0p1_0p2 > 2.0:
        z += -82.99918 * (0.006679471 - Q.sum_z_dr2) * (Q.n_dr_0p1_0p2 - 2.0)
    if Q.sum_z_dr2 < 0.006679471 and Q.phi_0 > -0.04013062:
        z += 568.6238 * (0.006679471 - Q.sum_z_dr2) * (Q.phi_0 - -0.04013062)
    if Q.sum_z_dr2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 4925.288 * (0.005019719 - Q.sum_z_dr2) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.sum_z_dr < 0.06108601 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += 4.211706 * (0.06108601 - Q.sum_z_dr) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt > 6.701242 and Q.zdr_0 > 0.007798268:
        z += -243.4217 * (Q.log_sum_pt - 6.701242) * (Q.zdr_0 - 0.007798268)
    if Q.sum_pt_top5 > 687.4375 and Q.e3 < 3.10694e-07:
        z += 5718.814 * (Q.sum_pt_top5 - 687.4375) * (3.10694e-07 - Q.e3)
    if Q.mass < 29.6447 and Q.D2_b2 < 0.716559:
        z += -0.03653579 * (29.6447 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.sum_z_dr < 0.06108601 and Q.lam2 < 0.0001947983:
        z += 133315.3 * (0.06108601 - Q.sum_z_dr) * (0.0001947983 - Q.lam2)
    if Q.lam2 < 0.0001330621 and Q.D2_b2 < 4.721224:
        z += 1102.74 * (0.0001330621 - Q.lam2) * (4.721224 - Q.D2_b2)
    if Q.tau1 < 0.05356915 and Q.D2_b2 < 4.721224:
        z += -2.361478 * (0.05356915 - Q.tau1) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.701242 and Q.e3 < 2.955458e-05:
        z += -34096.42 * (Q.log_sum_pt - 6.701242) * (2.955458e-05 - Q.e3)
    if Q.mass < 21.78408 and Q.lam2 < 0.0001947983:
        z += -474.975 * (21.78408 - Q.mass) * (0.0001947983 - Q.lam2)
    if Q.mass < 29.6447 and Q.e3 < 1.762929e-06:
        z += -1003.809 * (29.6447 - Q.mass) * (1.762929e-06 - Q.e3)
    if Q.LHA < 0.1967397 and Q.mean_eta > 0.003218391:
        z += -481.6909 * (0.1967397 - Q.LHA) * (Q.mean_eta - 0.003218391)
    if Q.mass < 49.6681 and Q.lam2 < 0.0001330621:
        z += 226.716 * (49.6681 - Q.mass) * (0.0001330621 - Q.lam2)
    if Q.centroid_offset < 0.003343241 and Q.D2_b2 < 0.5327104:
        z += -224.7253 * (0.003343241 - Q.centroid_offset) * (0.5327104 - Q.D2_b2)
    if Q.LHA < 0.1967397 and Q.psi_0p3 < 1.0:
        z += -836.8204 * (0.1967397 - Q.LHA) * (1.0 - Q.psi_0p3)
    if Q.LHA < 0.1967397 and Q.n_pt_above_50 > 6.0:
        z += -1.026046 * (0.1967397 - Q.LHA) * (Q.n_pt_above_50 - 6.0)
    if Q.mass < 21.78408 and Q.pt_7 < 48.71875:
        z += 3.599221e-05 * (21.78408 - Q.mass) * (48.71875 - Q.pt_7)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 < 48.71875:
        z += 0.09513604 * (Q.log_sum_pt - 6.701242) * (48.71875 - Q.pt_7)
    if Q.mass < 21.78408 and Q.D2_b2 < 0.716559:
        z += 0.02286673 * (21.78408 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.sum_pt_top5 > 687.4375 and Q.n_pt_above_10 < 8.0:
        z += -0.0003235752 * (Q.sum_pt_top5 - 687.4375) * (8.0 - Q.n_pt_above_10)
    if Q.sum_z_dr2 < 0.006679471 and Q.n_pt_above_50 > 3.0:
        z += -20.69724 * (0.006679471 - Q.sum_z_dr2) * (Q.n_pt_above_50 - 3.0)
    if Q.sum_z_dr2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -25196.62 * (0.005019719 - Q.sum_z_dr2) * (Q.centroid_offset - 0.006789738)
    if Q.LHA < 0.1967397 and Q.pt_7 > 15.55391:
        z += 0.1649105 * (0.1967397 - Q.LHA) * (Q.pt_7 - 15.55391)
    if Q.sum_z_dr < 0.06108601 and Q.lam1_plus_lam2 > 0.0003193707:
        z += 5290.333 * (0.06108601 - Q.sum_z_dr) * (Q.lam1_plus_lam2 - 0.0003193707)
    if Q.log_sum_pt > 6.701242 and Q.lam1_plus_lam2 < 0.008678045:
        z += -498.3141 * (Q.log_sum_pt - 6.701242) * (0.008678045 - Q.lam1_plus_lam2)
    if Q.mass < 29.6447 and Q.lam1_plus_lam2 < 0.003562611:
        z += -23.1739 * (29.6447 - Q.mass) * (0.003562611 - Q.lam1_plus_lam2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2 < 0.006679471:
        z += 1031.209 * (0.1986272 - Q.sj3_dr_max) * (0.006679471 - Q.sum_z_dr2)
    if Q.tau1 < 0.05356915 and Q.n_dr_0p2_0p4 > 0.0:
        z += -11.68247 * (0.05356915 - Q.tau1) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.tau1 < 0.05356915 and Q.lam1_plus_lam2 < 0.003562611:
        z += 7166.303 * (0.05356915 - Q.tau1) * (0.003562611 - Q.lam1_plus_lam2)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.288297
    if Q.sum_z_dr < 0.05464922:
        z += 31.50033 * Q.sum_z_dr - 1.721468
    if Q.sum_z_dr >= 0.0717028:
        z += -24.6822 * Q.sum_z_dr + 1.769783
    if Q.tau1 < 0.04369778:
        z += -13.05924 * Q.tau1 + 0.5706598
    if Q.mass < 53.33237:
        z += 0.02354895 * Q.mass - 1.255921
    if Q.e3 < 2.371297e-05:
        z += 35325.04 * Q.e3 - 0.8376619
    if 8.147744e-05 <= Q.e3 < 0.0001869378:
        z += -4473.425 * Q.e3 + 0.3644832
    if Q.e3 >= 0.0001869378:
        z += -472.4072 * Q.e3 - 0.3834583
    if Q.sum_z_dr2 < 0.003562611:
        z += -2833.811 * Q.sum_z_dr2 + 13.87645
    if 0.003562611 <= Q.sum_z_dr2 < 0.00609665:
        z += -1491.96 * Q.sum_z_dr2 + 9.095959
    if Q.sj3_dr_max < 0.1070199:
        z += 13.7537 * Q.sj3_dr_max - 0.666446
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 16.56405 * Q.sj3_dr_max - 0.9672091
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -19.70896 * Q.sj3_dr_max + 4.205872
    if Q.lam2 >= 0.001130645:
        z += 178.4804 * Q.lam2 - 0.2017979
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.09773083 * Q.n_dr_0p2_0p4 - 0.09773083
    if Q.lam1 < 0.003377388:
        z += 482.9771 * Q.lam1 - 2.875718
    if 0.003377388 <= Q.lam1 < 0.005433361:
        z += 184.0052 * Q.lam1 - 1.865974
    if 0.005433361 <= Q.lam1 < 0.00595415:
        z += 613.4496 * Q.lam1 - 4.1993
    if Q.lam1 >= 0.00595415:
        z += 130.4725 * Q.lam1 - 1.323582
    if Q.centroid_offset < 0.002316125:
        z += -1068.86 * Q.centroid_offset + 4.954107
    if 0.002316125 <= Q.centroid_offset < 0.01837778:
        z += -154.3113 * Q.centroid_offset + 2.835899
    if 0.03556091 <= Q.e2 < 0.04447357:
        z += -78.5245 * Q.e2 + 2.792403
    if Q.e2 >= 0.04447357:
        z += -23.68772 * Q.e2 + 0.3536155
    if Q.sj3_pair_mass_max < 24.23013:
        z += 0.0487895 * Q.sj3_pair_mass_max - 1.182176
    if Q.lam1_plus_lam2 < 0.0001721983:
        z += -15818.98 * Q.lam1_plus_lam2 + 2.724002
    if Q.C3 < 0.0284695:
        z += 20.96943 * Q.C3 - 0.5969892
    if Q.zdr_0 < 0.006292091:
        z += -98.24466 * Q.zdr_0 + 0.6181643
    if Q.mass_over_sum_pt < 0.07269073:
        z += 37.67969 * Q.mass_over_sum_pt - 2.738964
    if Q.sum_pt < 559.6875:
        z += -0.004279222 * Q.sum_pt + 2.643854
    if 559.6875 <= Q.sum_pt < 988.4078:
        z += -0.000580395 * Q.sum_pt + 0.573667
    z += -5.888309 * Q.z_dr_0p2_0p4
    if Q.log_sum_pt >= 6.896095:
        z += 9.467519 * Q.log_sum_pt - 65.28891
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.001835306 * Q.sum_pt_top5 + 1.541574
    if Q.pt_4 < 31.125:
        z += -0.2011916 * Q.pt_4 + 6.262088
    if Q.n_for_90pct < 7.0:
        z += -0.08293588 * Q.n_for_90pct + 0.5805512
    if Q.max_dr < 0.1117619:
        z += -10.61775 * Q.max_dr + 1.186659
    if Q.mass < 53.33237 and Q.centroid_offset < 0.02685622:
        z += 7.488679 * (53.33237 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass < 53.33237 and Q.log_sum_pt < 6.842717:
        z += 0.06986586 * (53.33237 - Q.mass) * (6.842717 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.00609665 and Q.planar_flow < 0.3220738:
        z += 1322.818 * (0.00609665 - Q.sum_z_dr2) * (0.3220738 - Q.planar_flow)
    if Q.sum_z_dr2 < 0.00609665 and Q.mean_phi < 0.002834884:
        z += -14862.26 * (0.00609665 - Q.sum_z_dr2) * (0.002834884 - Q.mean_phi)
    if Q.sum_z_dr < 0.05464922 and Q.z_5 > 0.03672711:
        z += 20.26254 * (0.05464922 - Q.sum_z_dr) * (Q.z_5 - 0.03672711)
    if Q.mass < 53.33237 and Q.D2 > 2.843757:
        z += 0.007529857 * (53.33237 - Q.mass) * (Q.D2 - 2.843757)
    if Q.lam1 > 0.005433361 and Q.eccentricity > 0.7117266:
        z += -582.5119 * (Q.lam1 - 0.005433361) * (Q.eccentricity - 0.7117266)
    if Q.sj3_pair_mass_max < 24.23013 and Q.absphi_2 > 0.007518768:
        z += -1.906388 * (24.23013 - Q.sj3_pair_mass_max) * (Q.absphi_2 - 0.007518768)
    if Q.tau1 < 0.04369778 and Q.eccentricity > 0.9031255:
        z += -1073.527 * (0.04369778 - Q.tau1) * (Q.eccentricity - 0.9031255)
    if Q.e3 < 2.371297e-05 and Q.eccentricity > 0.8319502:
        z += -25277.62 * (2.371297e-05 - Q.e3) * (Q.eccentricity - 0.8319502)
    if Q.e2 > 0.03556091 and Q.mean_eta > 0.00189657:
        z += -22.79845 * (Q.e2 - 0.03556091) * (Q.mean_eta - 0.00189657)
    if Q.sj3_dr_max < 0.1070199 and Q.abseta_1 > 0.03625488:
        z += 638.8597 * (0.1070199 - Q.sj3_dr_max) * (Q.abseta_1 - 0.03625488)
    if Q.centroid_offset < 0.01837778 and Q.abseta_1 < 0.07073975:
        z += 1185.314 * (0.01837778 - Q.centroid_offset) * (0.07073975 - Q.abseta_1)
    if Q.tau1 < 0.04369778 and Q.mean_phi < -0.01275329:
        z += -2435.874 * (0.04369778 - Q.tau1) * (-0.01275329 - Q.mean_phi)
    if Q.tau1 < 0.04369778 and Q.mean_phi > 0.02612796:
        z += -1859.065 * (0.04369778 - Q.tau1) * (Q.mean_phi - 0.02612796)
    if Q.centroid_offset < 0.01837778 and Q.tau4 > 0.001224244:
        z += 2200.196 * (0.01837778 - Q.centroid_offset) * (Q.tau4 - 0.001224244)
    if Q.sum_z_dr2 < 0.00609665 and Q.mean_phi > 0.02612796:
        z += -108526.0 * (0.00609665 - Q.sum_z_dr2) * (Q.mean_phi - 0.02612796)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.716559:
        z += -72.46633 * (Q.log_sum_pt - 6.896095) * (0.716559 - Q.D2_b2)
    if Q.centroid_offset < 0.01837778 and Q.n_for_90pct > 5.0:
        z += 11.14006 * (0.01837778 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.sj3_dr_max < 0.1426152 and Q.pt_6 > 33.6875:
        z += 0.2368571 * (0.1426152 - Q.sj3_dr_max) * (Q.pt_6 - 33.6875)
    if Q.pt_4 < 31.125 and Q.dr_min_012 > 0.01488897:
        z += -10.05872 * (31.125 - Q.pt_4) * (Q.dr_min_012 - 0.01488897)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 < 159.25:
        z += -0.5913764 * (0.01837778 - Q.centroid_offset) * (159.25 - Q.pt_1)
    if Q.centroid_offset < 0.01837778 and Q.z_2nd < 0.2055511:
        z += 1704.432 * (0.01837778 - Q.centroid_offset) * (0.2055511 - Q.z_2nd)
    if Q.pt_4 < 31.125 and Q.n_pt_above_1 < 8.0:
        z += -1.259379 * (31.125 - Q.pt_4) * (8.0 - Q.n_pt_above_1)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.sj3_dr_min < 0.2089872:
        z += -8.840094 * (Q.n_dr_0p2_0p4 - 1.0) * (0.2089872 - Q.sj3_dr_min)
    if Q.sum_z_dr < 0.05464922 and Q.tau21_b2 < 0.02656143:
        z += -630.3358 * (0.05464922 - Q.sum_z_dr) * (0.02656143 - Q.tau21_b2)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.17523
    z += 8.827181 * Q.e2
    if Q.sj3_pair_mass_min >= 11.051:
        z += 0.08836044 * Q.sj3_pair_mass_min - 0.9764715
    if Q.lam1 < 0.001503553:
        z += 1289.481 * Q.lam1 - 5.01799
    if 0.001503553 <= Q.lam1 < 0.004183811:
        z += 875.8107 * Q.lam1 - 4.396014
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += 413.36 * Q.lam1 - 2.461207
    if Q.lam1 >= 0.00733008:
        z += -285.8551 * Q.lam1 + 2.095341
    if 0.0003061234 <= Q.lam2 < 0.003408389:
        z += 930.8882 * Q.lam2 - 0.2849666
    if Q.lam2 >= 0.003408389:
        z += 567.4755 * Q.lam2 + 0.9536853
    if Q.pt_7 < 45.75:
        z += -0.0288245 * Q.pt_7 + 1.318721
    if Q.sj3_dr_min >= 0.1278212:
        z += 13.91388 * Q.sj3_dr_min - 1.778488
    if Q.e3 < 3.892127e-05:
        z += -5079.323 * Q.e3 + 0.4138503
    if 3.892127e-05 <= Q.e3 < 8.147744e-05:
        z += -6550.613 * Q.e3 + 0.4711147
    if Q.e3 >= 8.147744e-05:
        z += -1471.29 * Q.e3 + 0.05726446
    if Q.log_sum_pt < 6.080494:
        z += 5.712062 * Q.log_sum_pt - 34.73216
    if Q.mass_top5 >= 45.32077:
        z += -0.02913539 * Q.mass_top5 + 1.320438
    if Q.zdr_0 < 0.0211821:
        z += -12.51653 * Q.zdr_0 + 0.2651265
    if Q.M2 < 0.02563286:
        z += -24.38899 * Q.M2 + 0.6251597
    z += 16.20148 * Q.mean_phi
    if Q.sum_pt >= 988.4078:
        z += 0.008607245 * Q.sum_pt - 8.507468
    if Q.z_7 < 0.06473447:
        z += 37.06745 * Q.z_7 - 2.399542
    if Q.mean_eta >= 0.02644207:
        z += 42.42078 * Q.mean_eta - 1.121693
    if Q.LHA >= 0.3033137:
        z += -13.98098 * Q.LHA + 4.240622
    if Q.tau1 >= 0.05356915:
        z += 19.76692 * Q.tau1 - 1.058897
    if Q.sum_z_dr2 < 0.002635418:
        z += 481.2964 * Q.sum_z_dr2 - 1.268417
    if 0.007520088 <= Q.sum_z_dr2 < 0.02530566:
        z += 545.4446 * Q.sum_z_dr2 - 4.101792
    if Q.sum_z_dr2 >= 0.02530566:
        z += 315.9694 * Q.sum_z_dr2 + 1.705231
    if Q.mass < 76.6557:
        z += -0.01678673 * Q.mass + 1.286799
    if Q.phi_6 >= 0.04013062:
        z += 4.13968 * Q.phi_6 - 0.1661279
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += -554.4825 * Q.sum_z_dr2_top3 + 1.193007
    if Q.sj2_dr >= 0.3003793:
        z += -5.129569 * Q.sj2_dr + 1.540816
    if Q.sj3_dr_max >= 0.1986272:
        z += -2.459767 * Q.sj3_dr_max + 0.4885766
    if Q.C2_b2 >= 0.009032972:
        z += 55.44356 * Q.C2_b2 - 0.5008201
    if Q.lam2 > 0.0003061234 and Q.sj3_pairmax_over_m > 0.6745796:
        z += 2667.51 * (Q.lam2 - 0.0003061234) * (Q.sj3_pairmax_over_m - 0.6745796)
    if Q.e3 < 8.147744e-05 and Q.sj3_dr13 > 0.181053:
        z += 66983.38 * (8.147744e-05 - Q.e3) * (Q.sj3_dr13 - 0.181053)
    if Q.sj3_pair_mass_min > 11.051 and Q.n_dr_0p2_0p4 > 0.0:
        z += 0.01420503 * (Q.sj3_pair_mass_min - 11.051) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.pt_7 < 45.75 and Q.D2 < 1.002471:
        z += 0.1585948 * (45.75 - Q.pt_7) * (1.002471 - Q.D2)
    if Q.zdr_0 < 0.0211821 and Q.dr_7 > 0.1078355:
        z += 777.3304 * (0.0211821 - Q.zdr_0) * (Q.dr_7 - 0.1078355)
    if Q.zdr_0 < 0.0211821 and Q.centroid_offset > 0.01258764:
        z += 5267.474 * (0.0211821 - Q.zdr_0) * (Q.centroid_offset - 0.01258764)
    if Q.pt_7 < 45.75 and Q.log_sum_pt < 6.572938:
        z += -0.01670332 * (45.75 - Q.pt_7) * (6.572938 - Q.log_sum_pt)
    if Q.lam2 > 0.0003061234 and Q.planar_flow > 0.04505724:
        z += -352.9542 * (Q.lam2 - 0.0003061234) * (Q.planar_flow - 0.04505724)
    if Q.sj3_dr_min > 0.1278212 and Q.z_dr_0_0p05 < 0.3658817:
        z += -12.56508 * (Q.sj3_dr_min - 0.1278212) * (0.3658817 - Q.z_dr_0_0p05)
    if Q.sj3_dr_min > 0.1278212 and Q.dr_min_012 < 0.02695109:
        z += 203.9782 * (Q.sj3_dr_min - 0.1278212) * (0.02695109 - Q.dr_min_012)
    if Q.mass_top5 > 45.32077 and Q.n_dr_0p2_0p4 < 2.0:
        z += -0.0005890946 * (Q.mass_top5 - 45.32077) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.zdr_0 < 0.0211821 and Q.z_dr_0p05_0p1 < 0.2919447:
        z += 252.9864 * (0.0211821 - Q.zdr_0) * (0.2919447 - Q.z_dr_0p05_0p1)
    if Q.log_sum_pt < 6.080494 and Q.mean_eta2 > 9.030369e-05:
        z += -392.8565 * (6.080494 - Q.log_sum_pt) * (Q.mean_eta2 - 9.030369e-05)
    if Q.lam1 > 0.00733008 and Q.sj3_mass3 < 0.2723288:
        z += -43.94212 * (Q.lam1 - 0.00733008) * (0.2723288 - Q.sj3_mass3)
    if Q.e3 < 8.147744e-05 and Q.sj3_dr23 > 0.1797097:
        z += 25437.07 * (8.147744e-05 - Q.e3) * (Q.sj3_dr23 - 0.1797097)
    if Q.lam1 > 0.00733008 and Q.D2_b2 < 0.380911:
        z += -252.489 * (Q.lam1 - 0.00733008) * (0.380911 - Q.D2_b2)
    if Q.sj3_dr_min > 0.1278212 and Q.D2_b2 < 1.911889:
        z += 0.9752951 * (Q.sj3_dr_min - 0.1278212) * (1.911889 - Q.D2_b2)
    if Q.mean_eta > 0.02644207 and Q.absphi_5 > 0.04455566:
        z += -22.01822 * (Q.mean_eta - 0.02644207) * (Q.absphi_5 - 0.04455566)
    if Q.z_7 < 0.06473447 and Q.mean_eta < -0.004664942:
        z += -455.6249 * (0.06473447 - Q.z_7) * (-0.004664942 - Q.mean_eta)
    if Q.phi_6 > 0.04013062 and Q.eta_7 < -0.1218262:
        z += 69.9159 * (Q.phi_6 - 0.04013062) * (-0.1218262 - Q.eta_7)
    if Q.sj3_dr_min > 0.1278212 and Q.sj2_mass2 < 0.03573274:
        z += -37.41126 * (Q.sj3_dr_min - 0.1278212) * (0.03573274 - Q.sj2_mass2)
    if Q.LHA > 0.3033137 and Q.sj3_mass1 > 5.294926:
        z += 4.374719 * (Q.LHA - 0.3033137) * (Q.sj3_mass1 - 5.294926)
    if Q.LHA > 0.3033137 and Q.sj3_pairmax_over_m > 0.901705:
        z += 28.23888 * (Q.LHA - 0.3033137) * (Q.sj3_pairmax_over_m - 0.901705)
    if Q.lam1 > 0.00733008 and Q.D2_b2 > 1.129616:
        z += -27.88343 * (Q.lam1 - 0.00733008) * (Q.D2_b2 - 1.129616)
    if Q.sum_z_dr2 > 0.007520088 and Q.dr1_4 < 0.003088708:
        z += 41293.46 * (Q.sum_z_dr2 - 0.007520088) * (0.003088708 - Q.dr1_4)
    if Q.pt_7 < 45.75 and Q.tau4 > 0.007583927:
        z += 0.229557 * (45.75 - Q.pt_7) * (Q.tau4 - 0.007583927)
    if Q.log_sum_pt < 6.080494 and Q.absphi_7 > 0.03546143:
        z += -38.80004 * (6.080494 - Q.log_sum_pt) * (Q.absphi_7 - 0.03546143)
    if Q.tau1 > 0.05356915 and Q.eta_7 > -0.0803833:
        z += 19.9221 * (Q.tau1 - 0.05356915) * (Q.eta_7 - -0.0803833)
    if Q.pt_7 < 45.75 and Q.abseta_3 > 0.0013237:
        z += 0.2035151 * (45.75 - Q.pt_7) * (Q.abseta_3 - 0.0013237)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.8396851
    if Q.planar_flow < 0.2534037:
        z += -5.951642 * Q.planar_flow + 1.508168
    if 0.1778793 <= Q.sj2_dr < 0.2687922:
        z += -6.521752 * Q.sj2_dr + 1.160085
    if Q.sj2_dr >= 0.2687922:
        z += 12.24542 * Q.sj2_dr - 3.884383
    if Q.mass < 15.45403:
        z += -0.06618964 * Q.mass - 0.3280218
    if 15.45403 <= Q.mass < 49.6681:
        z += 0.03135482 * Q.mass - 1.835477
    if 49.6681 <= Q.mass < 69.61135:
        z += 0.01394672 * Q.mass - 0.9708497
    if Q.sum_z_dr2 < 0.004372139:
        z += -117.7325 * Q.sum_z_dr2 + 4.505055
    if 0.004372139 <= Q.sum_z_dr2 < 0.01323868:
        z += -450.0418 * Q.sum_z_dr2 + 5.957958
    if Q.tau1 < 0.09538712:
        z += 24.88784 * Q.tau1 - 2.37398
    if Q.LHA < 0.3127275:
        z += 0.2192106 * Q.LHA - 0.06855317
    if Q.centroid_offset < 0.01437952:
        z += 31.88605 * Q.centroid_offset + 1.318329
    if 0.01437952 <= Q.centroid_offset < 0.03776099:
        z += -75.99332 * Q.centroid_offset + 2.869583
    if Q.centroid_offset >= 0.04990367:
        z += -12.23515 * Q.centroid_offset + 0.6105788
    if Q.lam1 < 0.0005049491:
        z += 3479.665 * Q.lam1 - 5.351063
    if 0.0005049491 <= Q.lam1 < 0.00733008:
        z += 465.5875 * Q.lam1 - 3.829107
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 398.1982 * Q.lam1 - 3.335138
    if Q.sj3_dr_max < 0.1426152:
        z += 7.379875 * Q.sj3_dr_max - 0.7925396
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 28.5263 * Q.sj3_dr_max - 3.808341
    if 0.169029 <= Q.sj3_dr_max < 0.2623172:
        z += -10.86346 * Q.sj3_dr_max + 2.849672
    if Q.lam1_plus_lam2 < 0.008678045:
        z += -1376.023 * Q.lam1_plus_lam2 + 11.94119
    if Q.sum_z_dr < 0.02054282:
        z += 187.4076 * Q.sum_z_dr - 6.891116
    if 0.02054282 <= Q.sum_z_dr < 0.0717028:
        z += 59.44558 * Q.sum_z_dr - 4.262415
    if Q.sum_zz_dr2 < 0.008168571:
        z += 316.0627 * Q.sum_zz_dr2 - 2.581781
    if Q.z_7 >= 0.01685855:
        z += 25.85126 * Q.z_7 - 0.4358147
    if Q.z_dr_0p05_0p1 >= 0.8460335:
        z += 5.609496 * Q.z_dr_0p05_0p1 - 4.745821
    if Q.pt_6 < 41.21875:
        z += -0.005161766 * Q.pt_6 + 0.2127616
    if Q.max_pair_mass >= 33.3761:
        z += -0.05823555 * Q.max_pair_mass + 1.943675
    if Q.pt_7 >= 29.04219:
        z += 0.01042557 * Q.pt_7 - 0.3027813
    if Q.eccentricity >= 0.9884745:
        z += 54.79155 * Q.eccentricity - 54.16005
    if Q.C2 < 0.03578649:
        z += 5.900245 * Q.C2 - 0.211149
    if Q.max_dr < 0.1452311:
        z += -7.773977 * Q.max_dr + 1.129023
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.003406897 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.pt_7 < 37.15625:
        z += -0.1156433 * (0.2534037 - Q.planar_flow) * (37.15625 - Q.pt_7)
    if Q.tau1 < 0.09538712 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += 16.2924 * (0.09538712 - Q.tau1) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.planar_flow < 0.2534037 and Q.e3 < 1.050302e-05:
        z += -796095.9 * (0.2534037 - Q.planar_flow) * (1.050302e-05 - Q.e3)
    if Q.sj2_dr > 0.1778793 and Q.lam2 < 0.001130645:
        z += -10287.49 * (Q.sj2_dr - 0.1778793) * (0.001130645 - Q.lam2)
    if Q.mass < 49.6681 and Q.D2 < 1.332146:
        z += -0.08501926 * (49.6681 - Q.mass) * (1.332146 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.pt_6 > 19.46875:
        z += 1.312609 * (0.01323868 - Q.sum_z_dr2) * (Q.pt_6 - 19.46875)
    if Q.centroid_offset < 0.03776099 and Q.sum_pt < 901.5938:
        z += -0.05431299 * (0.03776099 - Q.centroid_offset) * (901.5938 - Q.sum_pt)
    if Q.centroid_offset < 0.03776099 and Q.mean_phi < -0.001813533:
        z += 1548.743 * (0.03776099 - Q.centroid_offset) * (-0.001813533 - Q.mean_phi)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += -71.98546 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.D2 < 0.415246:
        z += 141.7158 * (0.01323868 - Q.sum_z_dr2) * (0.415246 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.pt_2 < 61.53125:
        z += -1.086499 * (0.01323868 - Q.sum_z_dr2) * (61.53125 - Q.pt_2)
    if Q.centroid_offset < 0.03776099 and Q.absphi_0 < 0.0345459:
        z += -589.2571 * (0.03776099 - Q.centroid_offset) * (0.0345459 - Q.absphi_0)
    if Q.mass < 49.6681 and Q.dr1_7 > 0.1792439:
        z += 0.5379397 * (49.6681 - Q.mass) * (Q.dr1_7 - 0.1792439)
    if Q.mass < 15.45403 and Q.phi_1 < -0.05890198:
        z += -38.704 * (15.45403 - Q.mass) * (-0.05890198 - Q.phi_1)
    if Q.mass < 69.61135 and Q.D2 < 0.415246:
        z += -0.01432663 * (69.61135 - Q.mass) * (0.415246 - Q.D2)
    if Q.sj2_dr > 0.1778793 and Q.n_pt_above_50 > 4.0:
        z += -3.757454 * (Q.sj2_dr - 0.1778793) * (Q.n_pt_above_50 - 4.0)
    if Q.centroid_offset < 0.01437952 and Q.phi_1 > 0.002190304:
        z += 619.9089 * (0.01437952 - Q.centroid_offset) * (Q.phi_1 - 0.002190304)
    if Q.centroid_offset < 0.03776099 and Q.dr_max_012 > 0.1828389:
        z += 17.23304 * (0.03776099 - Q.centroid_offset) * (Q.dr_max_012 - 0.1828389)
    if Q.centroid_offset < 0.01437952 and Q.eta_1 < -0.005273438:
        z += 533.4482 * (0.01437952 - Q.centroid_offset) * (-0.005273438 - Q.eta_1)
    if Q.centroid_offset < 0.01437952 and Q.abseta_4 < 0.03601074:
        z += -4458.73 * (0.01437952 - Q.centroid_offset) * (0.03601074 - Q.abseta_4)
    if Q.planar_flow < 0.2534037 and Q.abseta_4 < 0.04977417:
        z += 73.302 * (0.2534037 - Q.planar_flow) * (0.04977417 - Q.abseta_4)
    if Q.centroid_offset < 0.01437952 and Q.D2_b2 < 0.5327104:
        z += 292.2008 * (0.01437952 - Q.centroid_offset) * (0.5327104 - Q.D2_b2)
    if Q.pt_6 < 41.21875 and Q.D2_b2 < 4.721224:
        z += -0.004067534 * (41.21875 - Q.pt_6) * (4.721224 - Q.D2_b2)
    if Q.centroid_offset > 0.04990367 and Q.tau32 < 0.5502779:
        z += -428.6457 * (Q.centroid_offset - 0.04990367) * (0.5502779 - Q.tau32)
    if Q.centroid_offset < 0.01437952 and Q.tau32 < 0.5186963:
        z += 484.7766 * (0.01437952 - Q.centroid_offset) * (0.5186963 - Q.tau32)
    if Q.sum_zz_dr2 < 0.008168571 and Q.mass_top2 > 6.779915:
        z += 3.909281 * (0.008168571 - Q.sum_zz_dr2) * (Q.mass_top2 - 6.779915)
    if Q.eccentricity > 0.9884745 and Q.sj2_mass2 < 2.771069:
        z += -28.36153 * (Q.eccentricity - 0.9884745) * (2.771069 - Q.sj2_mass2)
    if Q.eccentricity > 0.9884745 and Q.eta_0 < 0.02130127:
        z += -359.4143 * (Q.eccentricity - 0.9884745) * (0.02130127 - Q.eta_0)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.5414684
    if Q.sum_z_dr2 >= 0.01882765:
        z += 296.704 * Q.sum_z_dr2 - 5.58624
    if 69.61135 <= Q.mass < 91.19:
        z += -0.007179293 * Q.mass + 0.4997603
    if Q.mass >= 91.19:
        z += 0.07762021 * Q.mass - 7.233106
    if Q.mass_over_sum_pt >= 0.1309286:
        z += -16.1336 * Q.mass_over_sum_pt + 2.11235
    if Q.zdr_0 >= 0.03981924:
        z += -65.3826 * Q.zdr_0 + 2.603486
    if Q.sum_z_dr2_top2 >= 0.01403324:
        z += -6.431005 * Q.sum_z_dr2_top2 + 0.09024786
    if Q.mean_phi >= 0.02612796:
        z += 10.11424 * Q.mean_phi - 0.2642646
    if Q.e2 >= 0.06344108:
        z += -42.81764 * Q.e2 + 2.716397
    if Q.centroid_offset >= 0.04990367:
        z += 26.89941 * Q.centroid_offset - 1.342379
    if Q.sum_z_dr2 > 0.01882765 and Q.lam2 > 0.000537286:
        z += -8826.943 * (Q.sum_z_dr2 - 0.01882765) * (Q.lam2 - 0.000537286)
    if Q.mass > 91.19 and Q.mass_top3 < 50.3522:
        z += 0.0004649828 * (Q.mass - 91.19) * (50.3522 - Q.mass_top3)
    if Q.mass > 91.19 and Q.centroid_offset > 0.02076709:
        z += -1.62191 * (Q.mass - 91.19) * (Q.centroid_offset - 0.02076709)
    if Q.mass > 91.19 and Q.tau4 < 0.002207727:
        z += 12.3533 * (Q.mass - 91.19) * (0.002207727 - Q.tau4)
    if Q.mass > 91.19 and Q.n_dr_0p2_0p4 > 2.0:
        z += -0.002123757 * (Q.mass - 91.19) * (Q.n_dr_0p2_0p4 - 2.0)
    if Q.sum_z_dr2 > 0.01882765 and Q.pt_7 < 53.4375:
        z += -2.960926 * (Q.sum_z_dr2 - 0.01882765) * (53.4375 - Q.pt_7)
    if Q.mass > 69.61135 and Q.centroid_offset > 0.009480685:
        z += 1.590653 * (Q.mass - 69.61135) * (Q.centroid_offset - 0.009480685)
    if Q.mass > 69.61135 and Q.abseta_5 > 0.1522217:
        z += -0.08385451 * (Q.mass - 69.61135) * (Q.abseta_5 - 0.1522217)
    if Q.mass > 69.61135 and Q.dr_4 > 0.02477348:
        z += 0.01944367 * (Q.mass - 69.61135) * (Q.dr_4 - 0.02477348)
    if Q.mass > 91.19 and Q.eta_5 > 0.007119751:
        z += -0.07871084 * (Q.mass - 91.19) * (Q.eta_5 - 0.007119751)
    if Q.sum_z_dr2 > 0.01882765 and Q.abseta_6 > 0.01612854:
        z += 618.4127 * (Q.sum_z_dr2 - 0.01882765) * (Q.abseta_6 - 0.01612854)
    if Q.mass > 69.61135 and Q.log_sum_pt < 6.423044:
        z += 0.1985295 * (Q.mass - 69.61135) * (6.423044 - Q.log_sum_pt)
    if Q.mass > 91.19 and Q.abseta_1 < 0.117984:
        z += 0.07999748 * (Q.mass - 91.19) * (0.117984 - Q.abseta_1)
    if Q.sum_z_dr2_top2 > 0.01403324 and Q.mean_phi < -0.006701703:
        z += -1219.391 * (Q.sum_z_dr2_top2 - 0.01403324) * (-0.006701703 - Q.mean_phi)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.1803029
    if Q.sum_z_dr < 0.007673833:
        z += 566.8225 * Q.sum_z_dr + 4.232118
    if 0.007673833 <= Q.sum_z_dr < 0.1484084:
        z += -60.97875 * Q.sum_z_dr + 9.04976
    if Q.lam1 < 0.006506576:
        z += -466.8073 * Q.lam1 + 4.233381
    if 0.006506576 <= Q.lam1 < 0.01643375:
        z += -120.4839 * Q.lam1 + 1.980002
    if 658.125 <= Q.sum_pt_top5 < 791.125:
        z += 0.002945915 * Q.sum_pt_top5 - 1.938781
    if Q.sum_pt_top5 >= 791.125:
        z += 0.01538 * Q.sum_pt_top5 - 11.7757
    if Q.e3 < 5.334511e-05:
        z += -11413.35 * Q.e3 + 0.6088466
    if Q.pt_6 < 31.90625:
        z += 0.1047542 * Q.pt_6 - 3.342313
    if Q.sum_pt >= 988.4078:
        z += 0.00241725 * Q.sum_pt - 2.389229
    if Q.e2 < 0.08000524:
        z += 31.25459 * Q.e2 - 2.500531
    if Q.mass >= 49.6681:
        z += -0.03602142 * Q.mass + 1.789115
    if Q.z_5 < 0.02818362:
        z += 122.3621 * Q.z_5 - 3.448607
    if Q.z_7 < 0.02807091:
        z += 61.04217 * Q.z_7 - 1.713509
    if Q.sj3_dr23 >= 0.1974628:
        z += -5.707146 * Q.sj3_dr23 + 1.126949
    if Q.log_sum_pt >= 6.502799:
        z += 3.668802 * Q.log_sum_pt - 23.85749
    if Q.pt_7 >= 48.71875:
        z += -0.08204299 * Q.pt_7 + 3.997032
    if Q.sum_z_dr2_top2 < 0.0005124533:
        z += 2086.847 * Q.sum_z_dr2_top2 - 1.069412
    if Q.lam1_plus_lam2 < 0.007520088:
        z += 186.3161 * Q.lam1_plus_lam2 - 0.3937162
    if 0.007520088 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -176.162 * Q.lam1_plus_lam2 + 2.332151
    if Q.lam2 < 0.0001947983:
        z += -1188.934 * Q.lam2 + 0.2316023
    if Q.z_6 < 0.02160287:
        z += -119.8301 * Q.z_6 + 2.588675
    if Q.sum_z_dr < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -60.85351 * (0.1484084 - Q.sum_z_dr) * (6.804164 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.1484084 and Q.pt_7 < 38.53125:
        z += -1.367473 * (0.1484084 - Q.sum_z_dr) * (38.53125 - Q.pt_7)
    if Q.sum_pt_top5 > 658.125 and Q.z_7 > 0.02320757:
        z += -0.05075859 * (Q.sum_pt_top5 - 658.125) * (Q.z_7 - 0.02320757)
    if Q.e3 < 5.334511e-05 and Q.centroid_offset < 0.03776099:
        z += -728950.0 * (5.334511e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 40.04062:
        z += 0.0006642234 * (Q.sum_pt_top5 - 658.125) * (40.04062 - Q.pt_7)
    if Q.pt_6 < 31.90625 and Q.z_7 < 0.0586137:
        z += -4.219463 * (31.90625 - Q.pt_6) * (0.0586137 - Q.z_7)
    if Q.sum_z_dr < 0.1484084 and Q.z_7 > 0.06164517:
        z += 446.0542 * (0.1484084 - Q.sum_z_dr) * (Q.z_7 - 0.06164517)
    if Q.sum_z_dr < 0.1484084 and Q.lam2 < 0.000537286:
        z += -2846.17 * (0.1484084 - Q.sum_z_dr) * (0.000537286 - Q.lam2)
    if Q.sum_z_dr < 0.1484084 and Q.sj2_mass1 > 31.78116:
        z += -4.235915 * (0.1484084 - Q.sum_z_dr) * (Q.sj2_mass1 - 31.78116)
    if Q.sum_pt > 988.4078 and Q.M3 < 0.08151794:
        z += 0.4574149 * (Q.sum_pt - 988.4078) * (0.08151794 - Q.M3)
    if Q.sum_z_dr < 0.1484084 and Q.M3 < 0.07474969:
        z += 296.6678 * (0.1484084 - Q.sum_z_dr) * (0.07474969 - Q.M3)
    if Q.sum_z_dr < 0.1484084 and Q.mean_phi < -9.311297e-05:
        z += -235.7207 * (0.1484084 - Q.sum_z_dr) * (-9.311297e-05 - Q.mean_phi)
    if Q.pt_7 > 48.71875 and Q.pt_2 > 126.75:
        z += 0.006617122 * (Q.pt_7 - 48.71875) * (Q.pt_2 - 126.75)
    if Q.e3 < 5.334511e-05 and Q.D3 > 0.2213841:
        z += 2988.929 * (5.334511e-05 - Q.e3) * (Q.D3 - 0.2213841)
    if Q.sum_z_dr < 0.1484084 and Q.tau2 > 0.008780509:
        z += 835.4005 * (0.1484084 - Q.sum_z_dr) * (Q.tau2 - 0.008780509)
    if Q.sum_pt_top5 > 791.125 and Q.pt_3 < 74.5625:
        z += 0.0003074168 * (Q.sum_pt_top5 - 791.125) * (74.5625 - Q.pt_3)
    if Q.mass > 49.6681 and Q.z_dr_0p1_0p2 < 0.4684459:
        z += -0.01537123 * (Q.mass - 49.6681) * (0.4684459 - Q.z_dr_0p1_0p2)
    if Q.log_sum_pt > 6.502799 and Q.absphi_6 < 0.04681396:
        z += 47.56636 * (Q.log_sum_pt - 6.502799) * (0.04681396 - Q.absphi_6)
    if Q.log_sum_pt > 6.502799 and Q.zdr_7 > 0.0007928864:
        z += -27.05622 * (Q.log_sum_pt - 6.502799) * (Q.zdr_7 - 0.0007928864)
    if Q.log_sum_pt > 6.502799 and Q.pair_mass_0_7 > 10.2219:
        z += 0.1703409 * (Q.log_sum_pt - 6.502799) * (Q.pair_mass_0_7 - 10.2219)
    if Q.sum_pt_top5 > 791.125 and Q.pt_6 < 31.90625:
        z += 0.001136924 * (Q.sum_pt_top5 - 791.125) * (31.90625 - Q.pt_6)
    if Q.sum_pt > 988.4078 and Q.pt_6 < 62.25:
        z += -0.000343333 * (Q.sum_pt - 988.4078) * (62.25 - Q.pt_6)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.03737792
    if Q.planar_flow < 0.1115136:
        z += -7.670862 * Q.planar_flow + 0.8554051
    if Q.z_dr_0p05_0p1 < 0.5882598:
        z += 0.2384166 * Q.z_dr_0p05_0p1 - 0.1402509
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += -6.452606 * Q.z_dr_0p05_0p1 + 4.845323
    if 0.7143804 <= Q.psi_0p1 < 0.8155839:
        z += 0.5768496 * Q.psi_0p1 - 0.4120901
    if 0.8155839 <= Q.psi_0p1 < 0.9761279:
        z += -0.1148017 * Q.psi_0p1 + 0.1520096
    if Q.psi_0p1 >= 0.9761279:
        z += -26.3403 * Q.psi_0p1 + 25.75145
    if 0.007520088 <= Q.sum_z_dr2 < 0.008678045:
        z += -810.0172 * Q.sum_z_dr2 + 6.091401
    if Q.sum_z_dr2 >= 0.008678045:
        z += 87.87347 * Q.sum_z_dr2 - 1.700534
    if 0.002074109 <= Q.sum_zz_dr2 < 0.0030133:
        z += 355.5214 * Q.sum_zz_dr2 - 0.7373901
    if 0.0030133 <= Q.sum_zz_dr2 < 0.01165737:
        z += 127.7172 * Q.sum_zz_dr2 - 0.05094766
    if Q.sum_zz_dr2 >= 0.01165737:
        z += -8.677078 * Q.sum_zz_dr2 + 1.539051
    if 0.003562611 <= Q.lam1_plus_lam2 < 0.005590289:
        z += 334.3017 * Q.lam1_plus_lam2 - 1.190987
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.006679471:
        z += 506.9085 * Q.lam1_plus_lam2 - 2.155909
    if 0.006679471 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -510.1336 * Q.lam1_plus_lam2 + 4.637395
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += -49.20148 * Q.lam1_plus_lam2 - 1.464736
    if 0.01655442 <= Q.e2 < 0.03556091:
        z += -49.33583 * Q.e2 + 0.8167261
    if 0.03556091 <= Q.e2 < 0.04110972:
        z += -29.4713 * Q.e2 + 0.1103252
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += 92.61893 * Q.e2 - 4.90877
    if Q.e2 >= 0.05028464:
        z += -1.311392 * Q.e2 - 0.185517
    if 0.02685622 <= Q.centroid_offset < 0.04990367:
        z += 7.232913 * Q.centroid_offset - 0.1942487
    if Q.centroid_offset >= 0.04990367:
        z += -86.08156 * Q.centroid_offset + 4.462486
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 99.01451 * Q.mass_over_sum_pt - 7.91361
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 157.704 * Q.mass_over_sum_pt - 12.88764
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -70.63492 * Q.mass_over_sum_pt + 7.757357
    if Q.n_dr_0_0p05 < 5.0:
        z += -0.1524105 * Q.n_dr_0_0p05 + 0.7620525
    if Q.e3 < 5.334511e-05:
        z += 10282.33 * Q.e3 - 0.548512
    if Q.sd_mass < 49.91626:
        z += -0.002748627 * Q.sd_mass - 0.266265
    if 49.91626 <= Q.sd_mass < 74.57663:
        z += 0.01636091 * Q.sd_mass - 1.220142
    if 0.02689598 <= Q.sum_z_dr < 0.04081947:
        z += 27.54533 * Q.sum_z_dr - 0.7408586
    if 0.04081947 <= Q.sum_z_dr < 0.08065885:
        z += 44.00179 * Q.sum_z_dr - 1.412603
    if 0.08065885 <= Q.sum_z_dr < 0.08723651:
        z += 89.89058 * Q.sum_z_dr - 5.11394
    if 0.08723651 <= Q.sum_z_dr < 0.1019409:
        z += -72.38606 * Q.sum_z_dr + 9.042508
    if Q.sum_z_dr >= 0.1019409:
        z += -83.09924 * Q.sum_z_dr + 10.13462
    if Q.mass >= 69.61135:
        z += -0.01163832 * Q.mass + 0.8101589
    if Q.mass_top5 >= 49.18618:
        z += 0.01157187 * Q.mass_top5 - 0.5691761
    if 0.06154135 <= Q.sj2_dr < 0.1294903:
        z += -1.456352 * Q.sj2_dr + 0.08962584
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -16.30907 * Q.sj2_dr + 2.012909
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 20.3721 * Q.sj2_dr - 3.82568
    if Q.sj2_dr >= 0.2001708:
        z += 7.615819 * Q.sj2_dr - 1.272246
    if Q.lam2 < 0.000537286:
        z += -64.80531 * Q.lam2 + 0.03481898
    if Q.C2_b2 < 0.004032342:
        z += 88.94227 * Q.C2_b2 - 0.3586457
    if Q.LHA >= 0.3033137:
        z += 23.45869 * Q.LHA - 7.115343
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -10421.43 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += 1539.714 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1_plus_lam2 < 0.00609665:
        z += 5837.587 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.lam1_plus_lam2)
    if Q.planar_flow < 0.1115136 and Q.sum_pt_top5 < 658.125:
        z += -0.02965034 * (0.1115136 - Q.planar_flow) * (658.125 - Q.sum_pt_top5)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -1106.455 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.planar_flow < 0.1115136 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -1.807303 * (0.1115136 - Q.planar_flow) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.sum_zz_dr2 > 0.002074109 and Q.sj2_dr < 0.1872617:
        z += -12002.16 * (Q.sum_zz_dr2 - 0.002074109) * (0.1872617 - Q.sj2_dr)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.D3 < 0.3201347:
        z += 3.636617 * (0.5882598 - Q.z_dr_0p05_0p1) * (0.3201347 - Q.D3)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 1.0:
        z += 8.245953 * (Q.z_dr_0p05_0p1 - 0.7509095) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.sd_mass < 74.57663 and Q.sj2_mass2 > 3.697444:
        z += 0.003159411 * (74.57663 - Q.sd_mass) * (Q.sj2_mass2 - 3.697444)
    if Q.e3 < 5.334511e-05 and Q.n_dr_0p2_0p4 > 2.0:
        z += -1740.712 * (5.334511e-05 - Q.e3) * (Q.n_dr_0p2_0p4 - 2.0)
    if Q.planar_flow < 0.1115136 and Q.phi_7 < -0.08051147:
        z += -25.83802 * (0.1115136 - Q.planar_flow) * (-0.08051147 - Q.phi_7)
    if Q.planar_flow < 0.1115136 and Q.zdr_7 < 0.001964442:
        z += -3500.066 * (0.1115136 - Q.planar_flow) * (0.001964442 - Q.zdr_7)
    if Q.e3 < 5.334511e-05 and Q.sj3_dr23 > 0.1629004:
        z += 170784.6 * (5.334511e-05 - Q.e3) * (Q.sj3_dr23 - 0.1629004)
    if Q.sj2_dr > 0.2001708 and Q.z_5 < 0.1058993:
        z += -38.25544 * (Q.sj2_dr - 0.2001708) * (0.1058993 - Q.z_5)
    if Q.e3 < 5.334511e-05 and Q.phi_5 < -0.05200653:
        z += -57402.38 * (5.334511e-05 - Q.e3) * (-0.05200653 - Q.phi_5)
    if Q.centroid_offset > 0.02685622 and Q.dr_3 < 0.05268713:
        z += -907.8613 * (Q.centroid_offset - 0.02685622) * (0.05268713 - Q.dr_3)
    if Q.psi_0p1 > 0.8155839 and Q.phi_4 > 0.1096802:
        z += 33.03206 * (Q.psi_0p1 - 0.8155839) * (Q.phi_4 - 0.1096802)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.1863049
    if Q.N2 < 0.2233283:
        z += -0.5080157 * Q.N2 + 0.1134543
    if Q.sum_z_dr2 < 0.007520088:
        z += 865.2327 * Q.sum_z_dr2 - 6.506626
    if Q.mass_over_sum_pt < 0.1309286:
        z += 26.12276 * Q.mass_over_sum_pt - 3.420217
    if Q.lam1_plus_lam2 < 0.00609665:
        z += -451.4189 * Q.lam1_plus_lam2 + 8.460575
    if 0.00609665 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -678.8742 * Q.lam1_plus_lam2 + 9.847291
    if 0.01323868 <= Q.lam1_plus_lam2 < 0.01882765:
        z += -153.8556 * Q.lam1_plus_lam2 + 2.896739
    if Q.e2 < 0.04110972:
        z += -74.47357 * Q.e2 + 3.061587
    if Q.sum_zz_dr2 < 0.008168571:
        z += 953.9244 * Q.sum_zz_dr2 - 8.660713
    if 0.008168571 <= Q.sum_zz_dr2 < 0.01165737:
        z += 248.943 * Q.sum_zz_dr2 - 2.902022
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -966.322 * Q.mass_over_sum_pt_sq + 6.940932
    if Q.lam1 < 0.002464291:
        z += -385.5312 * Q.lam1 + 1.714097
    if 0.002464291 <= Q.lam1 < 0.00483998:
        z += -321.6061 * Q.lam1 + 1.556567
    if Q.lam2 < 0.001130645:
        z += -307.6409 * Q.lam2 + 0.3478325
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 322.0087 * Q.sum_z_dr2_top3 - 0.6928236
    if Q.sum_z_dr < 0.1019409:
        z += 27.03494 * Q.sum_z_dr - 2.755967
    if Q.sj2_dr < 0.1492731:
        z += 7.276348 * Q.sj2_dr - 1.720668
    if 0.1492731 <= Q.sj2_dr < 0.1591713:
        z += -20.22012 * Q.sj2_dr + 2.383815
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 20.35751 * Q.sj2_dr - 4.074979
    if Q.sum_z_dr2_top2 < 0.0005124533:
        z += 1871.303 * Q.sum_z_dr2_top2 - 0.9589557
    if Q.sj3_pair_mass_max >= 80.4:
        z += 0.02072158 * Q.sj3_pair_mass_max - 1.666015
    if Q.sj3_pairmin_over_m < 0.03043859:
        z += -35.27626 * Q.sj3_pairmin_over_m + 1.07376
    if Q.N2 < 0.2233283 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -10.94476 * (0.2233283 - Q.N2) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.N2 < 0.2233283 and Q.max_dr < 0.121681:
        z += -78.20596 * (0.2233283 - Q.N2) * (0.121681 - Q.max_dr)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 < 716.8828:
        z += 0.00477727 * (0.2233283 - Q.N2) * (716.8828 - Q.sum_pt_top5)
    if Q.sum_z_dr2 < 0.007520088 and Q.D2 < 0.7459513:
        z += -3561.767 * (0.007520088 - Q.sum_z_dr2) * (0.7459513 - Q.D2)
    if Q.lam1_plus_lam2 < 0.00609665 and Q.D2 < 0.7459513:
        z += 1475.17 * (0.00609665 - Q.lam1_plus_lam2) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.D2 < 0.7459513:
        z += 52.92858 * (0.1309286 - Q.mass_over_sum_pt) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.z_dr_0p1_0p2 > 0.4684459:
        z += 68.92754 * (0.1309286 - Q.mass_over_sum_pt) * (Q.z_dr_0p1_0p2 - 0.4684459)
    if Q.N2 < 0.2233283 and Q.n_dr_0p1_0p2 < 4.0:
        z += 0.6519716 * (0.2233283 - Q.N2) * (4.0 - Q.n_dr_0p1_0p2)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.02436016 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
    if Q.N2 < 0.2233283 and Q.pt_entropy < 1.54073:
        z += -21.94454 * (0.2233283 - Q.N2) * (1.54073 - Q.pt_entropy)
    if Q.N2 < 0.2233283 and Q.sj3_pairmax_over_m > 0.9367476:
        z += -93.77933 * (0.2233283 - Q.N2) * (Q.sj3_pairmax_over_m - 0.9367476)
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
