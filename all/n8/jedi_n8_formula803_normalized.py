"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; all observables), as if-statements with NORMALIZED weights (how much each one matters).

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.
  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1
             avg_k = the term's average size on the entire training set (term / avg is 1 on average), so share_k is the
             fraction of the neuron's average input that comes from that if-statement (sign: pushes it up / down).
  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1
             h_j = neuron j (after max(0, .) and the network's rounding), avg_j = its average on the training jets.

How much each neuron matters (share of all class scores, averaged over the training jets):
  neuron 13:  14.2%   (on for 88% of jets)
  neuron  9:  11.8%   (on for 70% of jets)
  neuron  7:  10.2%   (on for 61% of jets)
  neuron  3:   8.6%   (on for 29% of jets)
  neuron 10:   8.2%   (on for 80% of jets)
  neuron  2:   7.3%   (on for 88% of jets)
  neuron  5:   7.1%   (on for 64% of jets)
  neuron  6:   7.0%   (on for 49% of jets)
  neuron 11:   6.5%   (on for 71% of jets)
  neuron 14:   4.5%   (on for 30% of jets)
  neuron  0:   4.3%   (on for 43% of jets)
  neuron  1:   3.2%   (on for 56% of jets)
  neuron  4:   3.0%   (on for 44% of jets)
  neuron 15:   2.6%   (on for 33% of jets)
  neuron  8:   1.0%   (on for 34% of jets)
  neuron 12:   0.4%   (on for 8% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.9% (the network: 65.8%); same class as the network for 88.5% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_2          mass of particles 0 and 2 [GeV]
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
  Q.m01                    mass of particles 0 and 1 [GeV]
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
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
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_2               |Δφ| of particle 2
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr1_6                  ΔR between particle 6 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.eta_2                  Δη of particle 2
  Q.phi_0                  Δφ of particle 0
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
  Q.n_pt_above_10          number of particles with pT > 10 GeV
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_2=pair_mass(0, 2),
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
        m01=pair_mass(0, 1),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
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
        abseta_4=abs(eta[4]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_2=abs(phi[2]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr1_6=math.sqrt(dist2(1, 6)) if pt[6] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        eta_2=eta[2],
        phi_0=phi[0],
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
        n_pt_above_10=sum(1 for x in pt if x > 10),
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
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 26.46;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.46116 * (-0.06593593
        + 0.1053629 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +10.5%  sum_z_dr2 < 0.01324
        - 0.08749333 * max(0.0, 21.78408 - Q.mass) / 4.431023   # -8.7%  mass < 21.78
        + 0.08459897 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +8.5%  lam1_plus_lam2 < 0.008678
        + 0.05908642 * max(0.0, Q.sj3_dr_max - 0.1070199) / 0.08518534   # +5.9%  sj3_dr_max > 0.107
        - 0.04975763 * max(0.0, 0.07608178 - Q.sum_z_dr) / 0.02796784   # -5.0%  sum_z_dr < 0.07608
        - 0.04467324 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -4.5%  sum_z_dr < 0.08724
        - 0.04096845 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) / 0.00156571   # -4.1%  lam1_plus_lam2 < 0.004372
        - 0.03726319 * max(0.0, 64.61873 - Q.mass) / 27.57279   # -3.7%  mass < 64.62
        - 0.03640712 * max(0.0, 0.007929074 - Q.sum_z_dr2_top3) / 0.004560262   # -3.6%  sum_z_dr2_top3 < 0.007929
        - 0.02713914 * max(0.0, 0.08475161 - Q.mass_over_sum_pt) / 0.03304118   # -2.7%  mass_over_sum_pt < 0.08475
        - 0.0232984 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # -2.3%  sj3_dr_max > 0.179
        + 0.02048861 * max(0.0, 0.01882765 - Q.sum_z_dr2) / 0.01314296   # +2.0%  sum_z_dr2 < 0.01883
        + 0.02018134 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # +2.0%  mass_over_sum_pt < 0.1309
        - 0.02012769 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -2.0%  lam1 < 0.005433
        + 0.01989276 * max(0.0, 0.03556091 - Q.e2) / 0.01371972   # +2.0%  e2 < 0.03556
        + 0.01950859 * max(0.0, 0.006756161 - Q.sum_z_dr2_top3) / 0.003650633   # +2.0%  sum_z_dr2_top3 < 0.006756
        - 0.01940138 * max(0.0, 0.001563465 - Q.C2_b2) / 0.0008800039   # -1.9%  C2_b2 < 0.001563
        + 0.01891424 * max(0.0, 0.003904593 - Q.mass_over_sum_pt_sq) / 0.001485439   # +1.9%  mass_over_sum_pt_sq < 0.003905
        - 0.01877412 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -1.9%  centroid_offset > 0.00679
        + 0.018021 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +1.8%  planar_flow < 0.1484
        + 0.01647668 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +1.6%  sj3_dr_max < 0.3012
        + 0.01472415 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +1.5%  log_sum_pt > 6.378
        + 0.01353839 * max(0.0, Q.n_dr_0_0p05 - 4.0) / 1.61275   # +1.4%  n_dr_0_0p05 > 4
        - 0.01196628 * max(0.0, 56.92035 - Q.mass) / 21.78475   # -1.2%  mass < 56.92
        + 0.01083606 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) * max(0.0, 0.03532852 - Q.C3) / 3.6978e-05   # +1.1%  lam1_plus_lam2 < 0.004372 and C3 < 0.03533
        - 0.01058619 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -1.1%  centroid_offset > 0.0499
        - 0.0102035 * max(0.0, 7.300726e-05 - Q.lam2) / 2.767163e-05   # -1.0%  lam2 < 7.301e-05
        - 0.009709702 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 2.0 - Q.n_dr_0p05_0p1) / 0.1809599   # -1.0%  sj3_dr_max < 0.3012 and n_dr_0p05_0p1 < 2
        - 0.009134951 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # -0.9%  sum_z_dr2_top5 < 0.00833
        - 0.00825518 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -0.8%  sj3_dr_max > 0.2337
        - 0.007871881 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05557716 - Q.z_7) / 0.001734349   # -0.8%  sj3_dr_max < 0.3012 and z_7 < 0.05558
        + 0.007531577 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.8%  sum_pt_top5 > 687.4
        - 0.007323681 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, Q.z_7 - 0.01685855) / 0.0006290825   # -0.7%  log_sum_pt > 6.67 and z_7 > 0.01686
        - 0.006727746 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -0.7%  log_sum_pt > 6.67
        + 0.006658247 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 0.07232166 - Q.dr_4) / 0.001949106   # +0.7%  log_sum_pt > 6.67 and dr_4 < 0.07232
        - 0.006632256 * max(0.0, 0.0002758826 - Q.lam1) / 3.586469e-05   # -0.7%  lam1 < 0.0002759
        - 0.00609814 * max(0.0, Q.sum_pt - 901.5938) / 12.40362   # -0.6%  sum_pt > 901.6
        + 0.005015332 * max(0.0, Q.centroid_offset - 0.006789738) * max(0.0, 223.375 - Q.pt_1) / 1.309723   # +0.5%  centroid_offset > 0.00679 and pt_1 < 223.4
        + 0.004790813 * max(0.0, 0.0245477 - Q.e2) / 0.007455821   # +0.5%  e2 < 0.02455
        - 0.004651951 * max(0.0, 0.04889979 - Q.sj3_dr_max) / 0.006377687   # -0.5%  sj3_dr_max < 0.0489
        + 0.00458645 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2) / 2.36013e-06   # +0.5%  lam2 < 7.301e-05 and D2_b2 < 0.267
        - 0.004430738 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.06336451 - Q.dr_4) / 1.344914   # -0.4%  sum_pt_top5 > 687.4 and dr_4 < 0.06336
        + 0.00409606 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 28.78966 - Q.m01) / 1.085363   # +0.4%  log_sum_pt > 6.67 and m01 < 28.79
        + 0.003891864 * max(0.0, 0.01587232 - Q.zdr_1) / 0.007691685   # +0.4%  zdr_1 < 0.01587
        - 0.003848229 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 358.375 - Q.sum_pt_top2) / 2.426548   # -0.4%  planar_flow < 0.1484 and sum_pt_top2 < 358.4
        + 0.003761615 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, 1.002471 - Q.D2) / 0.001008123   # +0.4%  sum_z_dr2 < 0.01324 and D2 < 1.002
        + 0.003723077 * max(0.0, Q.centroid_offset - 0.006789738) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.02705531   # +0.4%  centroid_offset > 0.00679 and n_pt_above_50 < 7
        + 0.003075908 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.08525808 - Q.dr_2) / 1.41829e-06   # +0.3%  lam2 < 7.301e-05 and dr_2 < 0.08526
        - 0.00291492 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # -0.3%  lam1 < 0.006507 and D2 < 0.8757
        - 0.002363678 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, -0.003096655 - Q.mean_phi) / 2.294381e-05   # -0.2%  sum_z_dr2 < 0.01324 and mean_phi < -0.003097
        - 0.00208551 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 0.06473447 - Q.z_7) / 3.579716e-05   # -0.2%  centroid_offset > 0.02077 and z_7 < 0.06473
        + 0.001922095 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.4160096 - Q.pt_dispersion) / 0.0006933006   # +0.2%  planar_flow < 0.1484 and pt_dispersion < 0.416
        - 0.001800095 * max(0.0, Q.centroid_offset - 0.02076709) / 0.004751537   # -0.2%  centroid_offset > 0.02077
        - 0.00161683 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.01837778) / 2.056482e-05   # -0.2%  sum_z_dr2 < 0.01324 and centroid_offset > 0.01838
        - 0.001338782 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.mean_eta - 6.288824e-05) / 0.000715688   # -0.1%  log_sum_pt > 6.378 and mean_eta > 6.289e-05
        - 0.001321042 * max(0.0, 56.92035 - Q.mass) * max(0.0, Q.dr_max_012 - 0.06112084) / 0.1887976   # -0.1%  mass < 56.92 and dr_max_012 > 0.06112
        - 0.0009246954 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05744392 - Q.D2_b2) / 0.0005632394   # -0.1%  sj3_dr_max < 0.3012 and D2_b2 < 0.05744
        + 0.0008109172 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.pt_7 - 33.21875) / 68.87156   # +0.1%  sum_pt > 901.6 and pt_7 > 33.22
        + 0.0007958762 * max(0.0, 0.01882765 - Q.sum_z_dr2) * max(0.0, Q.mass_top2 - 28.78966) / 0.006400313   # +0.1%  sum_z_dr2 < 0.01883 and mass_top2 > 28.79
        - 0.0006004082 * max(0.0, 6.080494 - Q.log_sum_pt) / 0.006097657   # -0.1%  log_sum_pt < 6.08
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 18.96;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.95547 * (0.003153758
        - 0.08899136 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -8.9%  sum_z_dr2 < 0.008678
        - 0.08789739 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 0.002544039   # -8.8%  lam1_plus_lam2 < 0.006097
        - 0.07734558 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -7.7%  sum_pt_top5 > 531.2
        + 0.05526076 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +5.5%  log_sum_pt > 6.503
        - 0.05165895 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -5.2%  sum_z_dr < 0.1019
        + 0.04904487 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +4.9%  log_sum_pt > 6.378
        + 0.04578149 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +4.6%  lam1 < 0.005954
        + 0.04542691 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # +4.5%  z_7 > 0.02321
        + 0.0413364 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # +4.1%  sum_z_dr < 0.0717
        + 0.03179762 * max(0.0, 49.6681 - Q.mass) / 17.06893   # +3.2%  mass < 49.67
        + 0.0279848 * max(0.0, Q.sum_pt_top5 - 367.5938) / 230.8403   # +2.8%  sum_pt_top5 > 367.6
        + 0.02712905 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.06297984 - Q.tau2) / 0.0007776805   # +2.7%  z_7 < 0.06165 and tau2 < 0.06298
        - 0.02709072 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -2.7%  z_7 < 0.06165
        - 0.02443516 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, 0.008654951 - Q.zdr_6) / 0.000799435   # -2.4%  log_sum_pt > 6.503 and zdr_6 < 0.008655
        + 0.02099759 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # +2.1%  sum_zz_dr2 < 0.008169
        - 0.02036987 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1) / 0.1196218   # -2.0%  log_sum_pt > 6.378 and z_dr_0p05_0p1 < 0.7509
        + 0.01943103 * max(0.0, Q.sum_pt_top5 - 531.1875) * max(0.0, 0.01392641 - Q.zdr_6) / 1.245695   # +1.9%  sum_pt_top5 > 531.2 and zdr_6 < 0.01393
        - 0.01843265 * max(0.0, Q.z_7 - 0.04624032) / 0.01204993   # -1.8%  z_7 > 0.04624
        + 0.01793782 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) / 0.002578466   # +1.8%  mass_over_sum_pt_sq < 0.005833
        + 0.01753228 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +1.8%  e3 < 0.0005117
        + 0.01608774 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +1.6%  pt_7 > 34.53
        + 0.01347122 * max(0.0, Q.log_sum_pt - 6.605974) / 0.07073019   # +1.3%  log_sum_pt > 6.606
        - 0.01268114 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 1.627072e-05 - Q.e3) / 1.912338e-06   # -1.3%  log_sum_pt > 6.378 and e3 < 1.627e-05
        + 0.01184933 * max(0.0, 1.627072e-05 - Q.e3) / 6.577181e-06   # +1.2%  e3 < 1.627e-05
        - 0.01122337 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # -1.1%  sum_z_dr2 < 0.00502
        - 0.01081526 * max(0.0, 0.0008722282 - Q.lam1) / 0.0001882079   # -1.1%  lam1 < 0.0008722
        + 0.01061353 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +1.1%  tau1 < 0.07284
        - 0.008263338 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.00323438   # -0.8%  lam1 < 0.008376 and n_pt_above_50 > 5
        + 0.007719455 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.01403324 - Q.sum_z_dr2_top2) / 0.0001680113   # +0.8%  z_7 < 0.06165 and sum_z_dr2_top2 < 0.01403
        + 0.007566686 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +0.8%  sj3_dr_max > 0.169
        + 0.007414211 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +0.7%  lam1 < 0.008376
        - 0.007198935 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 1.679198 - Q.D2) / 0.005760154   # -0.7%  z_7 < 0.06165 and D2 < 1.679
        + 0.007003661 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.008921136 - Q.mean_phi2) / 0.0001024498   # +0.7%  z_7 < 0.06165 and mean_phi2 < 0.008921
        - 0.006562534 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -0.7%  centroid_offset < 0.01838
        + 0.005389077 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.1057739 - Q.abseta_0) / 0.001206885   # +0.5%  z_7 < 0.06165 and abseta_0 < 0.1058
        - 0.005261966 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 2.955458e-05 - Q.e3) / 7.137805e-05   # -0.5%  pt_7 > 34.53 and e3 < 2.955e-05
        - 0.005232593 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1484197 - Q.planar_flow) / 0.0001369594   # -0.5%  lam1 < 0.008376 and planar_flow < 0.1484
        - 0.004975776 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -0.5%  zdr_0 < 0.02118
        + 0.004453422 * max(0.0, 0.008678045 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.02076709) / 7.418362e-06   # +0.4%  sum_z_dr2 < 0.008678 and centroid_offset > 0.02077
        - 0.004348092 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 52.90625 - Q.pt_6) / 13.55761   # -0.4%  pt_7 > 34.53 and pt_6 < 52.91
        + 0.004014479 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # +0.4%  pt_7 > 34.53 and tau2 < 0.01713
        + 0.003835935 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 1.432482 - Q.D2) / 0.01970563   # +0.4%  log_sum_pt > 6.606 and D2 < 1.432
        - 0.003727299 * max(0.0, 25.57812 - Q.pt_7) / 1.327389   # -0.4%  pt_7 < 25.58
        - 0.003451146 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.04875823) / 0.002321449   # -0.3%  z_7 < 0.06165 and z_dr_0p05_0p1 > 0.04876
        + 0.003067494 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.centroid_offset - 0.009480685) / 0.0009010889   # +0.3%  log_sum_pt > 6.378 and centroid_offset > 0.009481
        + 0.002933547 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 0.09925997 - Q.z_4) / 0.00251516   # +0.3%  log_sum_pt > 6.606 and z_4 < 0.09926
        - 0.002513735 * max(0.0, Q.sj3_dr_min - 0.03628191) / 0.02376249   # -0.3%  sj3_dr_min > 0.03628
        - 0.002417929 * max(0.0, 0.1019409 - Q.sum_z_dr) * max(0.0, Q.pair_mass_0_6 - 4.037975) / 0.1185817   # -0.2%  sum_z_dr < 0.1019 and pair_mass_0_6 > 4.038
        + 0.002063087 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.sj2_dr - 0.1294903) / 0.1877842   # +0.2%  pt_7 > 34.53 and sj2_dr > 0.1295
        - 0.002045491 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236) / 1.769648e-05   # -0.2%  mass_over_sum_pt_sq < 0.005833 and centroid_offset > 0.008092
        - 0.002016541 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.2%  pt_7 > 53.44
        - 0.001569795 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.5700307   # -0.2%  sj3_dr_max > 0.169 and sj3_pair_mass_min > 5.744
        - 0.001245591 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.centroid_offset - 0.02076709) / 2.871254e-05   # -0.1%  z_7 < 0.06165 and centroid_offset > 0.02077
        + 0.001084286 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 3.175597   # +0.1%  pt_7 > 34.53 and n_dr_0p1_0p2 > 1
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 11.63;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.62794 * (0.1615591
        - 0.1260646 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -12.6%  pt_7 < 53.44
        - 0.08559228 * max(0.0, Q.LHA - 0.1329373) / 0.1175814   # -8.6%  LHA > 0.1329
        + 0.06873095 * max(0.0, 813.4156 - Q.sum_pt) / 129.0758   # +6.9%  sum_pt < 813.4
        - 0.06731483 * max(0.0, 36.22941 - Q.mass) / 10.11393   # -6.7%  mass < 36.23
        + 0.05686607 * max(0.0, Q.sum_pt - 527.1781) / 199.4719   # +5.7%  sum_pt > 527.2
        + 0.05294477 * max(0.0, 0.1079857 - Q.mass_over_sum_pt) / 0.05207765   # +5.3%  mass_over_sum_pt < 0.108
        + 0.05161564 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # +5.2%  log_sum_pt < 6.606
        - 0.03812596 * max(0.0, Q.sum_pt - 527.1781) * max(0.0, 0.0001947983 - Q.lam2) / 0.02863163   # -3.8%  sum_pt > 527.2 and lam2 < 0.0001948
        + 0.03528848 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +3.5%  lam1 < 0.005954
        - 0.03523698 * max(0.0, Q.z_7 - 0.03629544) / 0.01879829   # -3.5%  z_7 > 0.0363
        + 0.02956423 * max(0.0, 0.2507612 - Q.max_dr) / 0.1316542   # +3.0%  max_dr < 0.2508
        + 0.02702482 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +2.7%  z_7 < 0.07149
        - 0.02434066 * max(0.0, 62.55 - Q.sj3_pair_mass_max) * max(0.0, 0.06810151 - Q.z_7) / 0.6094489   # -2.4%  sj3_pair_mass_max < 62.55 and z_7 < 0.0681
        + 0.02238943 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # +2.2%  sum_z_dr2 < 0.003563
        - 0.01916268 * max(0.0, Q.z_7 - 0.03629544) * max(0.0, 0.004032342 - Q.C2_b2) / 4.986777e-05   # -1.9%  z_7 > 0.0363 and C2_b2 < 0.004032
        + 0.01752831 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +1.8%  lam2 < 0.0001948
        + 0.01646759 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.pt_6 - 19.46875) / 0.05232739   # +1.6%  lam1 < 0.005954 and pt_6 > 19.47
        + 0.01606932 * max(0.0, 62.55 - Q.sj3_pair_mass_max) / 30.55764   # +1.6%  sj3_pair_mass_max < 62.55
        - 0.01522082 * max(0.0, Q.sum_pt - 527.1781) * max(0.0, 0.01922607 - Q.absphi_2) / 1.427803   # -1.5%  sum_pt > 527.2 and absphi_2 < 0.01923
        + 0.01235688 * max(0.0, 36.22941 - Q.mass) * max(0.0, 73.75 - Q.pt_5) / 259.9887   # +1.2%  mass < 36.23 and pt_5 < 73.75
        + 0.01217721 * max(0.0, 0.01573181 - Q.absphi_2) / 0.003774997   # +1.2%  absphi_2 < 0.01573
        + 0.01188158 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +1.2%  pt_7 > 34.53
        - 0.01174775 * max(0.0, 53.4375 - Q.pt_7) * max(0.0, 1.129616 - Q.D2_b2) / 9.433694   # -1.2%  pt_7 < 53.44 and D2_b2 < 1.13
        - 0.01157343 * max(0.0, Q.sum_pt_top5 - 752.1) / 21.27415   # -1.2%  sum_pt_top5 > 752.1
        + 0.01121572 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # +1.1%  zdr_0 < 0.02118
        - 0.0102184 * max(0.0, Q.z_7 - 0.04939969) / 0.01023002   # -1.0%  z_7 > 0.0494
        - 0.009889288 * max(0.0, 62.55 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.01096064) / 0.2139385   # -1.0%  sj3_pair_mass_max < 62.55 and centroid_offset > 0.01096
        + 0.009521231 * max(0.0, 0.009668065 - Q.e2) / 0.001482576   # +1.0%  e2 < 0.009668
        - 0.009457545 * max(0.0, 0.4926918 - Q.planar_flow) / 0.2728654   # -0.9%  planar_flow < 0.4927
        - 0.008522976 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 0.0174408 - Q.abseta_0) / 0.216283   # -0.9%  sum_pt_top5 > 752.1 and abseta_0 < 0.01744
        + 0.007860437 * max(0.0, Q.sum_pt - 868.5094) / 18.08463   # +0.8%  sum_pt > 868.5
        + 0.007394231 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # +0.7%  lam1 < 0.001504
        + 0.00677986 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # +0.7%  log_sum_pt > 6.843
        - 0.006471842 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # -0.6%  sum_z_dr < 0.007674
        + 0.006347636 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # +0.6%  log_sum_pt < 6.464
        + 0.006095978 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 1.129616 - Q.D2_b2) / 6.295733   # +0.6%  sum_pt_top5 > 752.1 and D2_b2 < 1.13
        + 0.005704839 * max(0.0, 6.46415 - Q.log_sum_pt) * max(0.0, 1.129616 - Q.D2_b2) / 0.03658616   # +0.6%  log_sum_pt < 6.464 and D2_b2 < 1.13
        - 0.005081428 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.max_dr - 0.08050702) / 4.493327e-05   # -0.5%  lam1 < 0.005954 and max_dr > 0.08051
        + 0.004639368 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.0001947983 - Q.lam2) / 1.313144e-06   # +0.5%  log_sum_pt > 6.843 and lam2 < 0.0001948
        + 0.00376862 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.0174408 - Q.abseta_0) / 8.283244e-05   # +0.4%  log_sum_pt > 6.843 and abseta_0 < 0.01744
        - 0.003351186 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.3%  sum_pt_top5 > 840
        - 0.003196005 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 1.345805 - Q.D2_b2) / 0.002918988   # -0.3%  log_sum_pt > 6.843 and D2_b2 < 1.346
        + 0.002697689 * max(0.0, Q.pt_7 - 43.5) * max(0.0, 0.004032342 - Q.C2_b2) / 0.004969117   # +0.3%  pt_7 > 43.5 and C2_b2 < 0.004032
        + 0.002146423 * max(0.0, 0.007673833 - Q.sum_z_dr) * max(0.0, 75.625 - Q.pt_4) / 0.004511241   # +0.2%  sum_z_dr < 0.007674 and pt_4 < 75.62
        - 0.002097608 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 0.01573181 - Q.absphi_2) / 0.0002701397   # -0.2%  log_sum_pt < 6.606 and absphi_2 < 0.01573
        - 0.001187518 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.pt_6 - 41.21875) / 0.06090836   # -0.1%  log_sum_pt > 6.843 and pt_6 > 41.22
        + 0.001068904 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.4474937 - Q.z_dr_0p05_0p1) / 0.00315435   # +0.1%  log_sum_pt > 6.843 and z_dr_0p05_0p1 < 0.4475
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 20.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.56526 * (-0.0105676
        - 0.1580209 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # -15.8%  sum_z_dr2 > 0.008678
        - 0.1031658 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -10.3%  mass_over_sum_pt_sq < 0.01166
        - 0.08637265 * max(0.0, 0.004372139 - Q.sum_z_dr2) / 0.00156571   # -8.6%  sum_z_dr2 < 0.004372
        + 0.07254216 * max(0.0, Q.lam1_plus_lam2 - 0.007520088) / 0.002470136   # +7.3%  lam1_plus_lam2 > 0.00752
        + 0.06910936 * max(0.0, Q.sum_z_dr - 0.04081947) / 0.0270611   # +6.9%  sum_z_dr > 0.04082
        + 0.06160923 * max(0.0, 0.00817466 - Q.mass_over_sum_pt_sq) / 0.004299658   # +6.2%  mass_over_sum_pt_sq < 0.008175
        + 0.06012886 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +6.0%  lam1 > 0.008376
        + 0.03246776 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # +3.2%  lam1_plus_lam2 > 0.00559
        + 0.03152553 * max(0.0, Q.sj2_dr - 0.1872617) / 0.02732165   # +3.2%  sj2_dr > 0.1873
        - 0.02920456 * max(0.0, Q.sum_z_dr - 0.1019409) / 0.005400003   # -2.9%  sum_z_dr > 0.1019
        + 0.02335678 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # +2.3%  sum_z_dr > 0.08724
        - 0.02201959 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # -2.2%  tau1 > 0.05357
        - 0.02114824 * max(0.0, Q.sj2_dr - 0.1492731) / 0.04449993   # -2.1%  sj2_dr > 0.1493
        + 0.01381784 * max(0.0, Q.max_dr - 0.1027585) / 0.04505724   # +1.4%  max_dr > 0.1028
        - 0.01379353 * max(0.0, Q.sum_z_dr - 0.07608178) / 0.01059033   # -1.4%  sum_z_dr > 0.07608
        + 0.01334593 * max(0.0, Q.LHA - 0.3467135) / 0.008898884   # +1.3%  LHA > 0.3467
        + 0.01277704 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # +1.3%  sj3_dr_max > 0.1986
        - 0.01234755 * max(0.0, Q.centroid_offset - 0.01096064) / 0.008813008   # -1.2%  centroid_offset > 0.01096
        + 0.01207327 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +1.2%  tau1 > 0.1136
        - 0.01175795 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.9641201 - Q.z_dr_0p05_0p1) / 0.01959435   # -1.2%  sj2_dr > 0.1873 and z_dr_0p05_0p1 < 0.9641
        - 0.008642748 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # -0.9%  sj2_dr > 0.2688
        - 0.008306368 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # -0.8%  lam1_plus_lam2 > 0.01324
        - 0.008123386 * max(0.0, Q.mass - 64.61873) / 3.274818   # -0.8%  mass > 64.62
        - 0.007653816 * max(0.0, 0.9008535 - Q.z_dr_0_0p05) / 0.3811445   # -0.8%  z_dr_0_0p05 < 0.9009
        + 0.00711155 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.7%  lam2 > 0.001131
        + 0.007105112 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +0.7%  e2 > 0.04447
        + 0.007081606 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # +0.7%  sum_z_dr2_top5 < 0.00833
        + 0.006757078 * max(0.0, Q.sj2_dr - 0.2687922) * max(0.0, 0.9641201 - Q.z_dr_0p05_0p1) / 0.005973659   # +0.7%  sj2_dr > 0.2688 and z_dr_0p05_0p1 < 0.9641
        + 0.0063242 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.1950293   # +0.6%  mass_over_sum_pt > 0.06814 and sj3_pair_mass_min > 5.744
        - 0.005748872 * max(0.0, Q.sum_z_dr - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494) / 0.008781747   # -0.6%  sum_z_dr > 0.04082 and log_sum_pt > 6.08
        - 0.005742901 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # -0.6%  sj3_pair_mass_min > 11.05
        + 0.005303428 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50) / 0.02994202   # +0.5%  centroid_offset > 0.01096 and n_pt_above_50 < 8
        + 0.005132477 * max(0.0, Q.mass_over_sum_pt - 0.0681391) / 0.01539157   # +0.5%  mass_over_sum_pt > 0.06814
        - 0.004614962 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.02673292   # -0.5%  max_dr > 0.1028 and z_dr_0p05_0p1 < 0.846
        - 0.004448264 * max(0.0, Q.sj3_dr_max - 0.3012016) / 0.01214652   # -0.4%  sj3_dr_max > 0.3012
        - 0.004159804 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # -0.4%  tau1 > 0.1028
        + 0.004103511 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # +0.4%  sum_z_dr2 > 0.01883
        - 0.003536825 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -0.4%  e2 > 0.06344
        + 0.003221592 * max(0.0, Q.sd_mass - 62.55) * max(0.0, 0.9206502 - Q.D2_b2) / 1.823781   # +0.3%  sd_mass > 62.55 and D2_b2 < 0.9207
        + 0.002942634 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 0.07861328 - Q.abseta_0) / 0.0002969212   # +0.3%  centroid_offset > 0.01096 and abseta_0 < 0.07861
        - 0.00291099 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -0.3%  lam1 > 0.012
        + 0.002899337 * max(0.0, 1.0 - Q.n_dr_0_0p05) / 0.2859429   # +0.3%  n_dr_0_0p05 < 1
        - 0.002825992 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.pt_6 - 31.90625) / 0.1358885   # -0.3%  mass_over_sum_pt > 0.06814 and pt_6 > 31.91
        + 0.002580276 * max(0.0, 1.0 - Q.n_dr_0_0p05) * max(0.0, 4.0 - Q.n_dr_0p05_0p1) / 0.3078555   # +0.3%  n_dr_0_0p05 < 1 and n_dr_0p05_0p1 < 4
        + 0.00247297 * max(0.0, 0.03320012 - Q.planar_flow) / 0.004307781   # +0.2%  planar_flow < 0.0332
        + 0.002403195 * max(0.0, -0.004664942 - Q.mean_eta) / 0.003754527   # +0.2%  mean_eta < -0.004665
        + 0.001830549 * max(0.0, Q.sd_mass - 62.55) / 3.348271   # +0.2%  sd_mass > 62.55
        - 0.001672794 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr) / 0.0001196243   # -0.2%  mass_over_sum_pt > 0.06814 and sj2_dr < 0.218
        + 0.001574416 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6) / 0.005634141   # +0.2%  lam2 > 0.001131 and pt_6 < 56.53
        - 0.001214117 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113) / 0.39507   # -0.1%  sj2_dr > 0.1873 and sj2_mass1 > 2.25
        + 0.0006931062 * max(0.0, Q.mean_eta - 0.01772426) / 0.001375178   # +0.1%  mean_eta > 0.01772
        - 0.0002765972 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, Q.eta_2 - -0.04544525) / 0.002690131   # -0.0%  max_dr > 0.1028 and eta_2 > -0.04545
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 78.11;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 78.11022 * (-0.00471066
        + 0.2886833 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # +28.9%  sum_zz_dr2 < 0.01166
        - 0.2417002 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -24.2%  mass_over_sum_pt_sq < 0.01166
        - 0.03927635 * max(0.0, 0.05028464 - Q.e2) / 0.02502152   # -3.9%  e2 < 0.05028
        + 0.03492526 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # +3.5%  sj3_dr_max < 0.3456
        - 0.03353471 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -3.4%  sum_z_dr2 < 0.008678
        - 0.03332487 * max(0.0, 0.233678 - Q.sj3_dr_max) / 0.09057682   # -3.3%  sj3_dr_max < 0.2337
        + 0.02678876 * max(0.0, 0.06729223 - Q.C2) / 0.04162086   # +2.7%  C2 < 0.06729
        - 0.02643774 * max(0.0, 0.04990367 - Q.centroid_offset) / 0.03352573   # -2.6%  centroid_offset < 0.0499
        + 0.02202018 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +2.2%  N2 < 0.2233
        + 0.02194079 * max(0.0, Q.lam1_plus_lam2 - 0.003562611) / 0.004066872   # +2.2%  lam1_plus_lam2 > 0.003563
        + 0.01714409 * max(0.0, 0.3467135 - Q.LHA) / 0.1128474   # +1.7%  LHA < 0.3467
        - 0.01546177 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -1.5%  mass_over_sum_pt > 0.09041
        + 0.01411379 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +1.4%  tau1 < 0.07284
        - 0.01302312 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -1.3%  lam2 < 0.001131
        + 0.01146525 * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) / 0.004543329   # +1.1%  sum_z_dr2_top2 < 0.00764
        - 0.01125005 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -1.1%  mass_over_sum_pt < 0.07269
        + 0.01100598 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) / 0.003638607   # +1.1%  sum_z_dr2_top5 < 0.007164
        - 0.01094134 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # -1.1%  max_dr < 0.1773
        - 0.008759027 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266) / 0.01381411   # -0.9%  N2 < 0.2233 and eccentricity > 0.7117
        - 0.008651582 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -0.9%  sum_pt < 739.5
        + 0.008303309 * max(0.0, Q.sd_mass - 44.82259) / 8.759065   # +0.8%  sd_mass > 44.82
        + 0.007937902 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +0.8%  max_dr < 0.1118
        + 0.007664635 * max(0.0, Q.mass_top5 - 14.54404) / 16.7137   # +0.8%  mass_top5 > 14.54
        + 0.005496854 * max(0.0, 0.001653836 - Q.lam1_plus_lam2) / 0.0004289067   # +0.5%  lam1_plus_lam2 < 0.001654
        - 0.005136432 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -0.5%  lam1 < 0.001504
        + 0.005116519 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 27.42324 - Q.sj3_pair_mass_min) / 138.7479   # +0.5%  sd_mass > 44.82 and sj3_pair_mass_min < 27.42
        - 0.004941301 * max(0.0, Q.mass - 45.7571) / 9.387753   # -0.5%  mass > 45.76
        - 0.004637506 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7) / 0.8992052   # -0.5%  N2 < 0.2233 and pt_7 < 53.44
        - 0.004622355 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.5%  mass > 76.66
        + 0.004568911 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +0.5%  e3 < 0.0001869
        - 0.004568505 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -0.5%  lam2 < 0.0005373
        - 0.004554842 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.1647949 - Q.absphi_7) / 4.92402e-05   # -0.5%  lam2 < 0.0005373 and absphi_7 < 0.1648
        - 0.004335019 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.centroid_offset - 0.001308549) / 0.1760217   # -0.4%  sd_mass > 44.82 and centroid_offset > 0.001309
        + 0.003586877 * max(0.0, Q.max_dr - 0.1598486) / 0.0215652   # +0.4%  max_dr > 0.1598
        + 0.00340887 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +0.3%  e2 > 0.04447
        - 0.003188572 * max(0.0, 0.03874536 - Q.M2) / 0.006229062   # -0.3%  M2 < 0.03875
        - 0.00287452 * max(0.0, Q.LHA - 0.2809341) / 0.02587078   # -0.3%  LHA > 0.2809
        - 0.002370016 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.1410336 - Q.dr01) / 4.252003e-05   # -0.2%  lam2 < 0.0005373 and dr01 < 0.141
        - 0.002082431 * max(0.0, Q.sum_z_dr2_top5 - 0.001501708) / 0.004796547   # -0.2%  sum_z_dr2_top5 > 0.001502
        + 0.002065143 * max(0.0, Q.sd_mass - 74.57663) / 1.60503   # +0.2%  sd_mass > 74.58
        - 0.001859374 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1218872 - Q.abseta_7) / 0.003384012   # -0.2%  N2 < 0.2233 and abseta_7 < 0.1219
        - 0.001707228 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.07796252 - Q.dr1_6) / 1.697653e-05   # -0.2%  lam2 < 0.0005373 and dr1_6 < 0.07796
        - 0.001664886 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.55 - Q.mass) / 0.3959152   # -0.2%  N2 < 0.2233 and mass < 62.55
        + 0.001637897 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +0.2%  sj3_dr_max < 0.2134
        + 0.001588179 * max(0.0, 739.5 - Q.sum_pt) * max(0.0, Q.n_for_50pct - 1.0) / 170.3115   # +0.2%  sum_pt < 739.5 and n_for_50pct > 1
        - 0.001336903 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.275762 - Q.sd_zg) / 0.3498594   # -0.1%  sd_mass > 44.82 and sd_zg < 0.2758
        - 0.001091282 * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) * max(0.0, Q.C2_b2 - 0.0006435798) / 6.028305e-06   # -0.1%  sum_z_dr2_top2 < 0.00764 and C2_b2 > 0.0006436
        + 0.000901108 * max(0.0, Q.mass - 76.6557) * max(0.0, 0.107953 - Q.M3) / 0.09817562   # +0.1%  mass > 76.66 and M3 < 0.108
        + 0.0006305442 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, -0.01284493 - Q.mean_eta) / 2.067889e-07   # +0.1%  e3 < 0.0001869 and mean_eta < -0.01284
        - 0.0006078099 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 7.11886e-05   # -0.1%  N2 < 0.2233 and sum_zz_dr2 > 0.01166
        + 0.0005643433 * max(0.0, 0.2233283 - Q.N2) * max(0.0, -0.009352575 - Q.mean_phi) / 0.0001205535   # +0.1%  N2 < 0.2233 and mean_phi < -0.009353
        - 0.0005636042 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, -0.01275329 - Q.mean_phi) / 4.253124e-07   # -0.1%  lam2 < 0.0005373 and mean_phi < -0.01275
        - 0.0004696857 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.005440034 - Q.zdr_5) / 0.005279271   # -0.0%  sd_mass > 44.82 and zdr_5 < 0.00544
        - 0.0004689654 * max(0.0, Q.lam1_plus_lam2 - 0.003562611) * max(0.0, 0.1057566 - Q.sj3_z3) / 5.535701e-05   # -0.0%  lam1_plus_lam2 > 0.003563 and sj3_z3 < 0.1058
        + 0.0004452978 * max(0.0, Q.mass_top5 - 14.54404) * max(0.0, Q.pt_6 - 42.78125) / 57.98201   # +0.0%  mass_top5 > 14.54 and pt_6 > 42.78
        + 0.0004234171 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, Q.mean_eta - 0.01271871) / 4.463111e-07   # +0.0%  lam2 < 0.0005373 and mean_eta > 0.01272
        + 0.0004031985 * max(0.0, 0.0001869378 - Q.e3) * max(0.0, -0.009352575 - Q.mean_phi) / 2.725001e-07   # +0.0%  e3 < 0.0001869 and mean_phi < -0.009353
        - 0.0003647001 * max(0.0, Q.max_dr - 0.1598486) * max(0.0, 0.0006435798 - Q.C2_b2) / 1.059262e-06   # -0.0%  max_dr > 0.1598 and C2_b2 < 0.0006436
        - 0.0003570532 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # -0.0%  mean_eta > 0.02644
        + 0.0003396713 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, Q.pair_mass_0_6 - 15.55293) / 0.07779524   # +0.0%  sj3_dr_max < 0.3456 and pair_mass_0_6 > 15.55
        + 0.0002735316 * max(0.0, 0.3456459 - Q.sj3_dr_max) * max(0.0, Q.dr_5 - 0.1351015) / 0.0003073221   # +0.0%  sj3_dr_max < 0.3456 and dr_5 > 0.1351
        - 0.0002302479 * max(0.0, Q.sj3_pair_mass_max - 80.4) / 0.3254107   # -0.0%  sj3_pair_mass_max > 80.4
        - 0.0001624312 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 24.42188 - Q.pt_6) / 0.01371831   # -0.0%  N2 < 0.2233 and pt_6 < 24.42
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 7.123;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.122617 * (-0.04849988
        + 0.08233246 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +8.2%  z_7 < 0.07149
        + 0.07536661 * max(0.0, 0.001653836 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 7.0272e-06   # +7.5%  sum_z_dr2 < 0.001654 and centroid_offset < 0.02355
        - 0.06584522 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -6.6%  zdr_0 < 0.02118
        + 0.06342211 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.001184249   # +6.3%  lam1_plus_lam2 < 0.003563
        - 0.05563148 * max(0.0, 37.15625 - Q.pt_7) / 5.803748   # -5.6%  pt_7 < 37.16
        + 0.05500427 * max(0.0, 0.005284669 - Q.sum_zz_dr2) / 0.00223735   # +5.5%  sum_zz_dr2 < 0.005285
        - 0.04456465 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.001503553 - Q.lam1) / 4.027805e-05   # -4.5%  LHA < 0.2161 and lam1 < 0.001504
        - 0.04395456 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -4.4%  log_sum_pt > 6.701
        + 0.03833032 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +3.8%  LHA < 0.2161
        - 0.03741173 * max(0.0, Q.sum_pt_top5 - 430.75) / 176.7518   # -3.7%  sum_pt_top5 > 430.8
        + 0.03616018 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 46.125 - Q.pt_6) / 2161.354   # +3.6%  sum_pt_top5 > 430.8 and pt_6 < 46.12
        + 0.03512605 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +3.5%  sum_pt_top5 > 687.4
        + 0.03230357 * max(0.0, 0.001653836 - Q.sum_z_dr2) / 0.0004289067   # +3.2%  sum_z_dr2 < 0.001654
        - 0.02589114 * max(0.0, 0.002074109 - Q.sum_zz_dr2) / 0.000670556   # -2.6%  sum_zz_dr2 < 0.002074
        + 0.0258514 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 28.39396 - Q.pt1_dr01) / 4201.911   # +2.6%  sum_pt_top5 > 430.8 and pt1_dr01 < 28.39
        - 0.02579935 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -2.6%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.02529782 * max(0.0, 0.005284669 - Q.sum_zz_dr2) * max(0.0, 0.01437952 - Q.centroid_offset) / 1.318015e-05   # +2.5%  sum_zz_dr2 < 0.005285 and centroid_offset < 0.01438
        - 0.02511905 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # -2.5%  sum_z_dr2_top3 < 0.002152
        + 0.0217227 * max(0.0, 0.04939969 - Q.z_7) / 0.007457505   # +2.2%  z_7 < 0.0494
        + 0.01995004 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 40.2 - Q.mass_top3) / 0.60086   # +2.0%  z_7 < 0.07149 and mass_top3 < 40.2
        + 0.01971065 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +2.0%  z_7 < 0.02807
        + 0.01770404 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5) / 0.3059459   # +1.8%  z_7 < 0.0494 and mass_top5 < 62.55
        - 0.01674944 * max(0.0, 0.08051087 - Q.z_6) / 0.02253808   # -1.7%  z_6 < 0.08051
        + 0.01361268 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03532852 - Q.C3) / 0.0003034905   # +1.4%  z_7 < 0.07149 and C3 < 0.03533
        + 0.01360888 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.centroid_offset - 0.009480685) / 0.7981241   # +1.4%  sum_pt_top5 > 430.8 and centroid_offset > 0.009481
        + 0.01263565 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.002127561 - Q.mean_phi2) / 2.516806e-05   # +1.3%  z_7 < 0.07149 and mean_phi2 < 0.002128
        + 0.01260857 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01341502 - Q.dr_2) / 0.0001347035   # +1.3%  log_sum_pt > 6.701 and dr_2 < 0.01342
        + 0.00708527 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.sj2_dr - 0.1682655) / 4.310225   # +0.7%  sum_pt_top5 > 430.8 and sj2_dr > 0.1683
        + 0.00706087 * max(0.0, 37.15625 - Q.pt_7) * max(0.0, Q.pt_5 - 24.57812) / 73.42646   # +0.7%  pt_7 < 37.16 and pt_5 > 24.58
        + 0.005799553 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.sj3_dr13 - 0.04995258) / 0.000417922   # +0.6%  zdr_0 < 0.02118 and sj3_dr13 > 0.04995
        - 0.004949181 * max(0.0, 0.006789738 - Q.centroid_offset) * max(0.0, 56.4375 - Q.pt_5) / 0.01266727   # -0.5%  centroid_offset < 0.00679 and pt_5 < 56.44
        - 0.004900425 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # -0.5%  log_sum_pt > 6.843
        + 0.004394327 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # +0.4%  sum_z_dr < 0.007674
        - 0.004276466 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 0.009661512 - Q.dr_2) / 1.13159e-05   # -0.4%  z_7 < 0.0494 and dr_2 < 0.009662
        + 0.004123249 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # +0.4%  pt_5 < 24.58
        + 0.00382463 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 9.030369e-05 - Q.mean_eta2) / 9.99897e-07   # +0.4%  log_sum_pt > 6.701 and mean_eta2 < 9.03e-05
        - 0.003742611 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.02270492 - Q.dr_2) / 4.344748e-05   # -0.4%  log_sum_pt > 6.896 and dr_2 < 0.0227
        - 0.002179955 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.2%  log_sum_pt > 6.896
        + 0.002037706 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.sj3_dr_max - 0.169029) / 0.0008305926   # +0.2%  log_sum_pt > 6.701 and sj3_dr_max > 0.169
        - 0.001960118 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) * max(0.0, Q.sj3_dr13 - 0.181053) / 2.382944e-06   # -0.2%  lam1_plus_lam2 < 0.003563 and sj3_dr13 > 0.1811
        - 0.001951039 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.mean_phi - 0.002834884) / 4.214377e-05   # -0.2%  log_sum_pt > 6.701 and mean_phi > 0.002835
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 21.61;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.61039 * (0.03118976
        - 0.136716 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -13.7%  sum_z_dr2 < 0.008678
        - 0.1293985 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # -12.9%  lam1_plus_lam2 < 0.01324
        + 0.07924805 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +7.9%  lam1 < 0.012
        + 0.07639343 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +7.6%  lam1 < 0.00733
        + 0.05811806 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +5.8%  lam2 < 0.003408
        + 0.05723285 * max(0.0, 0.1136369 - Q.tau1) * max(0.0, 0.01426135 - Q.mean_phi2) / 0.0007578925   # +5.7%  tau1 < 0.1136 and mean_phi2 < 0.01426
        - 0.04104182 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # -4.1%  max_dr < 0.1452
        + 0.03310279 * max(0.0, 0.1789613 - Q.sj3_dr_max) / 0.05394779   # +3.3%  sj3_dr_max < 0.179
        + 0.0310729 * max(0.0, 0.02415398 - Q.C2_b2) / 0.02132459   # +3.1%  C2_b2 < 0.02415
        - 0.03102111 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -3.1%  lam2 < 0.001131
        - 0.03037931 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 1.341509   # -3.0%  mass < 60.63 and z_dr_0p2_0p4 < 0.05644
        - 0.02515519 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -2.5%  sj3_dr_max < 0.3012
        - 0.022505 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 0.2534037 - Q.planar_flow) / 0.0006566196   # -2.3%  lam1 < 0.012 and planar_flow < 0.2534
        + 0.02170307 * max(0.0, 60.63098 - Q.mass) / 24.47895   # +2.2%  mass < 60.63
        - 0.01614923 * max(0.0, 45.595 - Q.mass) / 14.73532   # -1.6%  mass < 45.59
        + 0.01540331 * max(0.0, 21.78408 - Q.mass) / 4.431023   # +1.5%  mass < 21.78
        - 0.01538109 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -1.5%  pt_6 < 39.75
        + 0.01503256 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +1.5%  pt_6 < 41.22
        + 0.01353641 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # +1.4%  sum_zz_dr2 < 0.003013
        + 0.0129913 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt) / 1.016213   # +1.3%  pt_6 < 41.22 and log_sum_pt < 6.767
        + 0.01262525 * max(0.0, Q.eccentricity - 0.927072) / 0.03232992   # +1.3%  eccentricity > 0.9271
        - 0.01260329 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -1.3%  lam1 < 0.005954
        + 0.01134592 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +1.1%  sj3_dr_max > 0.1879
        + 0.009860461 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +1.0%  centroid_offset > 0.008092
        - 0.008478925 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757) / 0.07098489   # -0.8%  pt_6 < 41.22 and z_7 > 0.02321
        - 0.008330852 * max(0.0, Q.sj3_dr_min - 0.02400746) / 0.02802924   # -0.8%  sj3_dr_min > 0.02401
        - 0.006920293 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.7%  sum_pt_top5 > 840
        + 0.006377947 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # +0.6%  sj2_dr < 0.1873
        + 0.005729598 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.6%  sum_pt > 988.4
        + 0.005486828 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002658741   # +0.5%  lam2 < 0.003408 and n_dr_0_0p05 < 3
        + 0.004296021 * max(0.0, 6.638339 - Q.log_sum_pt) * max(0.0, 0.0753896 - Q.z_7) / 0.001441873   # +0.4%  log_sum_pt < 6.638 and z_7 < 0.07539
        + 0.003772507 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.psi_0p1 - 0.4008925) / 0.003485257   # +0.4%  centroid_offset > 0.008092 and psi_0p1 > 0.4009
        + 0.003249099 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 43.23531   # +0.3%  sum_pt < 615.9 and n_dr_0p2_0p4 < 2
        - 0.002852866 * max(0.0, Q.sj3_pair_mass_min - 4.501727) / 3.480268   # -0.3%  sj3_pair_mass_min > 4.502
        + 0.00263424 * max(0.0, 0.1872617 - Q.sj2_dr) * max(0.0, Q.eccentricity - 0.927072) / 0.0009514438   # +0.3%  sj2_dr < 0.1873 and eccentricity > 0.9271
        + 0.002466399 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2) / 2.179362e-05   # +0.2%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        - 0.002399759 * max(0.0, Q.sj3_dr23 - 0.2207152) / 0.01962555   # -0.2%  sj3_dr23 > 0.2207
        + 0.002242786 * max(0.0, 615.875 - Q.sum_pt) / 30.07766   # +0.2%  sum_pt < 615.9
        + 0.002212283 * max(0.0, 19.46875 - Q.pt_6) / 0.2515892   # +0.2%  pt_6 < 19.47
        + 0.002132549 * max(0.0, 6.267538 - Q.log_sum_pt) / 0.02292097   # +0.2%  log_sum_pt < 6.268
        - 0.002102514 * max(0.0, Q.sj3_dr13 - 0.181053) / 0.02525513   # -0.2%  sj3_dr13 > 0.1811
        + 0.002041773 * max(0.0, 29.90625 - Q.pt_6) / 1.385454   # +0.2%  pt_6 < 29.91
        - 0.00187783 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.2%  z_6 < 0.0216
        - 0.001832059 * max(0.0, Q.tau2 - 0.008780509) / 0.00811667   # -0.2%  tau2 > 0.008781
        + 0.001677916 * max(0.0, 0.1789613 - Q.sj3_dr_max) * max(0.0, 0.1230713 - Q.tau21_b2) / 0.0006424247   # +0.2%  sj3_dr_max < 0.179 and tau21_b2 < 0.1231
        + 0.00162153 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 0.1343804 - Q.dr1_7) / 0.3834372   # +0.2%  pt_6 < 41.22 and dr1_7 < 0.1344
        - 0.001498876 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 3.0374e-08 - Q.e4) / 5.487247e-07   # -0.1%  sum_pt < 615.9 and e4 < 3.037e-08
        - 0.00132 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, Q.sj3_dr12 - 0.1692253) / 0.0003039084   # -0.1%  centroid_offset > 0.01838 and sj3_dr12 > 0.1692
        - 0.001249672 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.002776626 - Q.mean_phi2) / 0.01000982   # -0.1%  sum_pt > 988.4 and mean_phi2 < 0.002777
        + 0.00121754 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +0.1%  sj3_dr_min > 0.1278
        - 0.001149356 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 2.69383   # -0.1%  sj3_pair_mass_min > 4.502 and n_dr_0p2_0p4 > 1
        - 0.0007621744 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # -0.1%  mean_eta > 0.02644
        - 0.0007137098 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.002127561 - Q.mean_phi2) / 0.007417827   # -0.1%  sum_pt > 988.4 and mean_phi2 < 0.002128
        + 0.0006792927 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.mean_eta - 0.02644207) / 1.54475e-06   # +0.1%  lam1 < 0.012 and mean_eta > 0.02644
        + 0.0006416361 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) * max(0.0, -0.02594505 - Q.mean_phi) / 1.701156e-06   # +0.1%  lam1_plus_lam2 < 0.01324 and mean_phi < -0.02595
        + 0.0006413622 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) * max(0.0, -0.02665591 - Q.mean_eta) / 1.690289e-06   # +0.1%  lam1_plus_lam2 < 0.01324 and mean_eta < -0.02666
        + 0.0006060986 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) * max(0.0, Q.mean_phi - 0.02612796) / 1.673749e-06   # +0.1%  lam1_plus_lam2 < 0.01324 and mean_phi > 0.02613
        + 0.0005017487 * max(0.0, 29.90625 - Q.pt_6) * max(0.0, 8.0 - Q.n_pt_above_10) / 0.5895401   # +0.1%  pt_6 < 29.91 and n_pt_above_10 < 8
        - 0.0002215155 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.n_pt_above_50 - 6.0) / 2.33176   # -0.0%  sum_pt > 988.4 and n_pt_above_50 > 6
        - 4.355029e-05 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.mean_phi2 - 0.008921136) / 0.000300354   # -0.0%  sum_pt > 988.4 and mean_phi2 > 0.008921
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 43.02;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 43.01857 * (0.2500928
        - 0.09508825 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -9.5%  sum_z_dr2 > 0.00752
        - 0.0714646 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # -7.1%  lam1_plus_lam2 > 0.00559
        - 0.07092502 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -7.1%  sum_z_dr < 0.08724
        - 0.06207508 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) / 0.00293624   # -6.2%  lam1_plus_lam2 < 0.006679
        - 0.05981447 * max(0.0, Q.mass_over_sum_pt - 0.01109984) / 0.05101851   # -6.0%  mass_over_sum_pt > 0.0111
        + 0.05690107 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +5.7%  mass_over_sum_pt > 0.08475
        + 0.0481511 * max(0.0, Q.mass_over_sum_pt - 0.07269073) / 0.01344138   # +4.8%  mass_over_sum_pt > 0.07269
        - 0.04564185 * max(0.0, Q.sum_z_dr2 - 0.004372139) / 0.003638805   # -4.6%  sum_z_dr2 > 0.004372
        - 0.03555857 * max(0.0, Q.sum_z_dr2 - 0.001653836) / 0.005220304   # -3.6%  sum_z_dr2 > 0.001654
        + 0.0342221 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +3.4%  sj2_dr < 0.1592
        - 0.03354896 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -3.4%  sj2_dr < 0.1873
        + 0.03207924 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +3.2%  LHA < 0.3033
        + 0.0298196 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +3.0%  mass_over_sum_pt > 0.07992
        + 0.02924846 * max(0.0, Q.sj3_dr_max - 0.1426152) / 0.06279029   # +2.9%  sj3_dr_max > 0.1426
        - 0.02525661 * max(0.0, Q.sj3_dr_max - 0.04889979) / 0.1256943   # -2.5%  sj3_dr_max > 0.0489
        - 0.01871981 * max(0.0, Q.lam1_plus_lam2 - 0.008678045) / 0.002214537   # -1.9%  lam1_plus_lam2 > 0.008678
        - 0.01600497 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -1.6%  lam1 < 0.008376
        + 0.0157408 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.6%  tau1 < 0.05357
        + 0.01354845 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +1.4%  e2 > 0.05028
        + 0.01071956 * max(0.0, Q.mass - 36.22941) / 14.20528   # +1.1%  mass > 36.23
        - 0.01065119 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # -1.1%  sum_z_dr2 < 0.003563
        + 0.01055076 * max(0.0, Q.sum_z_dr2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +1.1%  sum_z_dr2 > 0.004372 and planar_flow < 0.195
        - 0.01053411 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.004032342 - Q.C2_b2) / 2.736263e-05   # -1.1%  centroid_offset < 0.02077 and C2_b2 < 0.004032
        + 0.01032715 * max(0.0, 0.08723651 - Q.sum_z_dr) * max(0.0, 0.004032342 - Q.C2_b2) / 0.0001217265   # +1.0%  sum_z_dr < 0.08724 and C2_b2 < 0.004032
        - 0.008364712 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # -0.8%  max_dr < 0.1118
        + 0.008332854 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # +0.8%  max_dr < 0.1773
        - 0.008103549 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # -0.8%  lam2 < 0.0003061
        - 0.007250632 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -0.7%  sj3_dr_max > 0.2623
        - 0.00657118 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -0.7%  centroid_offset < 0.02077
        - 0.006425403 * max(0.0, 0.04019753 - Q.tau21_b2) / 0.01109641   # -0.6%  tau21_b2 < 0.0402
        + 0.006408584 * max(0.0, 0.0003061234 - Q.lam2) * max(0.0, 0.04019753 - Q.tau21_b2) / 2.611351e-06   # +0.6%  lam2 < 0.0003061 and tau21_b2 < 0.0402
        - 0.006124119 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.lam1_plus_lam2 - 0.007520088) / 0.0001455029   # -0.6%  planar_flow < 0.195 and lam1_plus_lam2 > 0.00752
        - 0.005833738 * max(0.0, Q.mass - 76.6557) * max(0.0, Q.zdr_6 - 0.006366792) / 0.006641425   # -0.6%  mass > 76.66 and zdr_6 > 0.006367
        + 0.005703233 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # +0.6%  e2 > 0.03556
        + 0.005580167 * max(0.0, Q.sum_z_dr2 - 0.01323868) * max(0.0, 38.25 - Q.pt_6) / 0.004519922   # +0.6%  sum_z_dr2 > 0.01324 and pt_6 < 38.25
        - 0.005291214 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.5%  centroid_offset > 0.03776
        + 0.005258907 * max(0.0, Q.z_7 - 0.03243272) / 0.02180051   # +0.5%  z_7 > 0.03243
        - 0.005152562 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.5%  mass > 76.66
        - 0.005082579 * max(0.0, 0.0009641429 - Q.sum_z_dr2) / 0.0002052288   # -0.5%  sum_z_dr2 < 0.0009641
        + 0.004461133 * max(0.0, 2.357246 - Q.D2) / 1.097329   # +0.4%  D2 < 2.357
        + 0.004385284 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # +0.4%  sd_rg > 0.2788
        + 0.004377446 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, Q.sum_pt_top3 - 331.25) / 1.710819   # +0.4%  centroid_offset < 0.02077 and sum_pt_top3 > 331.2
        - 0.00409179 * max(0.0, Q.sd_rg - 0.2037854) / 0.01566501   # -0.4%  sd_rg > 0.2038
        - 0.003695497 * max(0.0, 0.04081947 - Q.sum_z_dr) / 0.009176315   # -0.4%  sum_z_dr < 0.04082
        - 0.00365529 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -0.4%  mass_over_sum_pt > 0.09041
        + 0.003421723 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sd_mass - 38.43971) / 1.25962   # +0.3%  planar_flow < 0.195 and sd_mass > 38.44
        - 0.003414179 * max(0.0, Q.sum_z_dr2 - 0.01323868) / 0.001442684   # -0.3%  sum_z_dr2 > 0.01324
        + 0.003312207 * max(0.0, Q.lam1_plus_lam2 - 0.008678045) * max(0.0, 38.25 - Q.pt_6) / 0.006941575   # +0.3%  lam1_plus_lam2 > 0.008678 and pt_6 < 38.25
        + 0.002907569 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) / 0.0003002514   # +0.3%  sum_z_dr2_top2 < 0.001057
        - 0.002484692 * max(0.0, Q.sj3_dr_max - 0.213399) / 0.03061279   # -0.2%  sj3_dr_max > 0.2134
        - 0.001976779 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # -0.2%  sj2_dr < 0.1295
        - 0.001788265 * max(0.0, Q.mass_over_sum_pt - 0.01109984) * max(0.0, 35.28125 - Q.pt_6) / 0.1065511   # -0.2%  mass_over_sum_pt > 0.0111 and pt_6 < 35.28
        - 0.001755241 * max(0.0, 29.04219 - Q.pt_7) / 2.179984   # -0.2%  pt_7 < 29.04
        + 0.001725362 * max(0.0, 0.001101266 - Q.sum_zz_dr2) / 0.000308004   # +0.2%  sum_zz_dr2 < 0.001101
        + 0.001694922 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.02656143 - Q.tau21_b2) / 4.805781e-05   # +0.2%  centroid_offset < 0.02077 and tau21_b2 < 0.02656
        + 0.00160337 * max(0.0, Q.mass_top5 - 53.60766) / 1.978814   # +0.2%  mass_top5 > 53.61
        + 0.001347355 * max(0.0, Q.centroid_offset - 0.03117077) / 0.002505019   # +0.1%  centroid_offset > 0.03117
        - 0.001234047 * max(0.0, 8.10414e-05 - Q.sum_z_dr2_top2) / 7.505045e-06   # -0.1%  sum_z_dr2_top2 < 8.104e-05
        + 0.001192736 * max(0.0, 0.08723651 - Q.sum_z_dr) * max(0.0, 0.004406178 - Q.mean_phi) / 0.0002325096   # +0.1%  sum_z_dr < 0.08724 and mean_phi < 0.004406
        - 0.0009004661 * max(0.0, 0.05744392 - Q.D2_b2) / 0.005938983   # -0.1%  D2_b2 < 0.05744
        - 0.0007632393 * max(0.0, Q.sum_z_dr2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9458207) / 2.125219e-05   # -0.1%  sum_z_dr2 > 0.01324 and eccentricity > 0.9458
        - 0.0005830835 * max(0.0, Q.centroid_offset - 0.03117077) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.00188116   # -0.1%  centroid_offset > 0.03117 and n_pt_above_50 > 4
        - 0.0004974684 * max(0.0, 29.04219 - Q.pt_7) * max(0.0, 0.2838437 - Q.tau21) / 0.1125981   # -0.0%  pt_7 < 29.04 and tau21 < 0.2838
        - 0.0002872244 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) * max(0.0, 0.02656143 - Q.tau21_b2) / 2.239525e-07   # -0.0%  sum_z_dr2_top2 < 0.001057 and tau21_b2 < 0.02656
        - 0.0002588945 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, Q.n_dr_0p05_0p1 - 5.0) / 0.0005328768   # -0.0%  tau1 < 0.05357 and n_dr_0p05_0p1 > 5
        - 0.0001095372 * max(0.0, Q.centroid_offset - 0.03776099) * max(0.0, Q.pt_4 - 81.375) / 0.0001936307   # -0.0%  centroid_offset > 0.03776 and pt_4 > 81.38
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 15;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.99887 * (-0.0160232
        - 0.1085639 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.003562611 - Q.sum_z_dr2) / 5.329563e-05   # -10.9%  sum_z_dr < 0.06109 and sum_z_dr2 < 0.003563
        - 0.08807082 * max(0.0, 0.06108601 - Q.sum_z_dr) / 0.01868685   # -8.8%  sum_z_dr < 0.06109
        + 0.07657074 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.0003703114   # +7.7%  sj3_dr_max < 0.1986 and sum_z_dr2 < 0.006679
        + 0.07310431 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.001184249   # +7.3%  lam1_plus_lam2 < 0.003563
        + 0.06852425 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 4.808955e-05   # +6.9%  tau1 < 0.05357 and lam1_plus_lam2 < 0.003563
        + 0.05759755 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0001901112   # +5.8%  sum_z_dr2 < 0.00502 and z_dr_0p2_0p4 < 0.101
        + 0.0346827 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.0001272072   # +3.5%  mass_over_sum_pt < 0.03319 and centroid_offset < 0.02686
        + 0.03395994 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +3.4%  sum_z_dr2 < 0.006679 and centroid_offset < 0.02355
        + 0.03049886 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +3.0%  sum_z_dr2 < 0.00502
        - 0.03036619 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -3.0%  sj3_dr_max < 0.1426
        - 0.02925407 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.1231549   # -2.9%  mass < 29.64 and centroid_offset < 0.02686
        - 0.02751339 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.0001781324   # -2.8%  sj3_dr_max < 0.1986 and lam1_plus_lam2 < 0.003563
        - 0.02500781 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.005691733 - Q.sum_z_dr2_top5) / 0.0003116052   # -2.5%  sj3_dr_max < 0.1986 and sum_z_dr2_top5 < 0.005692
        - 0.02235345 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.0001947983 - Q.lam2) / 0.0007178522   # -2.2%  mass < 21.78 and lam2 < 0.0001948
        - 0.0206792 * max(0.0, 0.01289969 - Q.e2) * max(0.0, Q.psi_0p2 - 0.79448) / 0.0005184882   # -2.1%  e2 < 0.0129 and psi_0p2 > 0.7945
        + 0.01942783 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.0001947983 - Q.lam2) / 2.869243e-06   # +1.9%  sum_z_dr < 0.06109 and lam2 < 0.0001948
        - 0.01895656 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -1.9%  log_sum_pt > 6.701
        - 0.01854475 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -1.9%  sum_z_dr2 < 0.00502 and centroid_offset > 0.00679
        - 0.01655982 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) / 0.00722828   # -1.7%  mass_over_sum_pt < 0.03319
        + 0.01597162 * max(0.0, 0.0001330621 - Q.lam2) * max(0.0, 4.721224 - Q.D2_b2) / 0.0002340602   # +1.6%  lam2 < 0.0001331 and D2_b2 < 4.721
        - 0.01523914 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # -1.5%  LHA < 0.1967
        - 0.01459165 * max(0.0, 0.0001330621 - Q.lam2) / 6.711676e-05   # -1.5%  lam2 < 0.0001331
        + 0.01252346 * max(0.0, 0.01289969 - Q.e2) / 0.002527207   # +1.3%  e2 < 0.0129
        - 0.0116876 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 4.721224 - Q.D2_b2) / 0.008595177   # -1.2%  sum_z_dr2 < 0.006679 and D2_b2 < 4.721
        - 0.01161995 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 43.5 - Q.pt_7) / 0.02255472   # -1.2%  sum_z_dr2 < 0.00502 and pt_7 < 43.5
        + 0.0107036 * max(0.0, 0.002635418 - Q.lam1_plus_lam2) / 0.0007941347   # +1.1%  lam1_plus_lam2 < 0.002635
        - 0.01049474 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.001011075   # -1.0%  sum_z_dr < 0.06109 and z_dr_0p2_0p4 < 0.05644
        + 0.009287307 * max(0.0, 29.6447 - Q.mass) * max(0.0, 1.762929e-06 - Q.e3) / 9.656157e-06   # +0.9%  mass < 29.64 and e3 < 1.763e-06
        + 0.008957948 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.0001947983 - Q.lam2) / 9.822054e-06   # +0.9%  sj3_dr_max < 0.1986 and lam2 < 0.0001948
        - 0.008354005 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.01627885) / 0.0001130126   # -0.8%  sj3_dr_max < 0.1426 and centroid_offset > 0.01628
        + 0.007721912 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.8%  sum_pt_top5 > 687.4
        + 0.007581515 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, Q.phi_0 - -0.04013062) / 0.0001189753   # +0.8%  sum_z_dr2 < 0.006679 and phi_0 > -0.04013
        - 0.006938558 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # -0.7%  sj3_dr_max < 0.1986
        + 0.006693405 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +0.7%  sj3_dr_max < 0.107
        + 0.006157092 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, Q.lam1_plus_lam2 - 0.0003193707) / 1.048473e-05   # +0.6%  sum_z_dr < 0.06109 and lam1_plus_lam2 > 0.0003194
        + 0.005580406 * max(0.0, 49.6681 - Q.mass) * max(0.0, 0.0001330621 - Q.lam2) / 0.001509268   # +0.6%  mass < 49.67 and lam2 < 0.0001331
        + 0.004894639 * max(0.0, 21.78408 - Q.mass) * max(0.0, 48.71875 - Q.pt_7) / 69.12746   # +0.5%  mass < 21.78 and pt_7 < 48.72
        - 0.004411944 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.02355416 - Q.centroid_offset) / 0.0005980206   # -0.4%  log_sum_pt > 6.701 and centroid_offset < 0.02355
        + 0.003900919 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 2.955458e-05 - Q.e3) / 8.40609e-07   # +0.4%  log_sum_pt > 6.701 and e3 < 2.955e-05
        + 0.003372827 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 48.71875 - Q.pt_7) / 0.776931   # +0.3%  log_sum_pt > 6.701 and pt_7 < 48.72
        + 0.002518279 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.pt_7 - 15.55391) / 0.4153091   # +0.3%  LHA < 0.1967 and pt_7 > 15.55
        - 0.002513366 * max(0.0, 0.03448406 - Q.z_6) / 0.001525592   # -0.3%  z_6 < 0.03448
        + 0.002152592 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0005611231 - Q.sum_z_dr2) / 8.626944e-06   # +0.2%  log_sum_pt > 6.701 and sum_z_dr2 < 0.0005611
        - 0.001552301 * max(0.0, 21.78408 - Q.mass) / 4.431023   # -0.2%  mass < 21.78
        - 0.001526235 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.0002365852   # -0.2%  log_sum_pt > 6.701 and lam1_plus_lam2 < 0.008678
        + 0.001351617 * max(0.0, 29.6447 - Q.mass) / 7.34723   # +0.1%  mass < 29.64
        - 0.000947753 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.2097233   # -0.1%  mass < 29.64 and D2_b2 < 0.7166
        - 0.0005174758 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 8.0 - Q.n_pt_above_10) / 7.731645   # -0.1%  sum_pt_top5 > 687.4 and n_pt_above_10 < 8
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 31.03;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.0281 * (-0.1071774
        + 0.1078855 * max(0.0, 0.00609665 - Q.sum_z_dr2) / 0.002544039   # +10.8%  sum_z_dr2 < 0.006097
        - 0.07645341 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -7.6%  mass_over_sum_pt < 0.07269
        + 0.0697339 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # +7.0%  sum_z_dr2 < 0.003563
        + 0.0535934 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +5.4%  sum_pt < 988.4
        + 0.05244637 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.2973626   # +5.2%  mass < 53.33 and centroid_offset < 0.02686
        - 0.05201732 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -5.2%  sj3_dr_max < 0.1426
        + 0.04753507 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +4.8%  sj3_dr_max < 0.2134
        + 0.04670629 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +4.7%  mass_over_sum_pt_sq < 0.007183
        - 0.04516768 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -4.5%  lam1 < 0.005954
        + 0.03793746 * max(0.0, Q.lam1 - 0.005433361) / 0.00267714   # +3.8%  lam1 > 0.005433
        - 0.0331465 * max(0.0, 53.33237 - Q.mass) / 19.36004   # -3.3%  mass < 53.33
        + 0.03042924 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +3.0%  centroid_offset < 0.01838
        - 0.02592453 * max(0.0, Q.lam1 - 0.003377388) / 0.003672352   # -2.6%  lam1 > 0.003377
        + 0.02567211 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +2.6%  sum_z_dr2 < 0.00502
        - 0.02381108 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # -2.4%  sum_zz_dr2 < 0.003013
        - 0.02346543 * max(0.0, 0.05464922 - Q.sum_z_dr) / 0.01532978   # -2.3%  sum_z_dr < 0.05465
        + 0.02289659 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # +2.3%  tau1 < 0.0437
        - 0.01637781 * max(0.0, 2.371297e-05 - Q.e3) / 1.105328e-05   # -1.6%  e3 < 2.371e-05
        - 0.01506946 * max(0.0, 29.6447 - Q.mass) / 7.34723   # -1.5%  mass < 29.64
        + 0.0125646 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # +1.3%  lam2 < 0.0003061
        + 0.01078148 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +1.1%  max_dr < 0.1118
        - 0.01052453 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1) / 0.1931178   # -1.1%  centroid_offset < 0.01838 and pt_1 < 159.2
        + 0.01022934 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt) / 5.175436   # +1.0%  mass < 53.33 and log_sum_pt < 6.843
        - 0.008083386 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -0.8%  e2 > 0.03556
        - 0.007481906 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266) / 0.0005312449   # -0.7%  lam1 > 0.005433 and eccentricity > 0.7117
        + 0.007450275 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +0.7%  sj3_dr_max < 0.107
        - 0.006766734 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.008712838   # -0.7%  centroid_offset < 0.01838 and n_for_90pct > 5
        - 0.006632848 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, Q.z_6 - 0.03448406) / 0.0003071415   # -0.7%  sum_z_dr < 0.05465 and z_6 > 0.03448
        + 0.00647798 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd) / 0.0002178654   # +0.6%  centroid_offset < 0.01838 and z_2nd < 0.2056
        + 0.006435171 * max(0.0, Q.mass - 45.595) / 9.461082   # +0.6%  mass > 45.59
        + 0.0060104 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +0.6%  e2 > 0.04447
        - 0.005783597 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # -0.6%  e3 > 8.148e-05
        - 0.005621683 * max(0.0, Q.sum_z_dr - 0.0717028) / 0.01201981   # -0.6%  sum_z_dr > 0.0717
        + 0.005348867 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.0002201438   # +0.5%  sum_z_dr2 < 0.006097 and planar_flow < 0.3221
        + 0.004712027 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +0.5%  e3 > 0.0001869
        - 0.004689576 * max(0.0, 0.0284695 - Q.C3) / 0.008714909   # -0.5%  C3 < 0.02847
        + 0.004450236 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.6875) / 0.3914557   # +0.4%  sj3_dr_max < 0.1426 and pt_6 > 33.69
        - 0.003664857 * Q.z_dr_0p2_0p4 / 0.02920532   # -0.4%  z_dr_0p2_0p4
        + 0.003576384 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.4%  n_dr_0p2_0p4 > 1
        + 0.003488973 * max(0.0, Q.N2 - 0.2233283) / 0.04872624   # +0.3%  N2 > 0.2233
        - 0.003483525 * max(0.0, Q.sd_mass - 45.595) / 8.450255   # -0.3%  sd_mass > 45.59
        - 0.003442632 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, 0.002834884 - Q.mean_phi) / 1.344572e-05   # -0.3%  sum_z_dr2 < 0.006097 and mean_phi < 0.002835
        - 0.003261629 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255) / 0.000277001   # -0.3%  tau1 < 0.0437 and eccentricity > 0.9031
        - 0.003209048 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) / 6.087676   # -0.3%  sj3_pair_mass_max < 24.23
        + 0.00316664 * max(0.0, 0.0782171 - Q.M3) / 0.01197321   # +0.3%  M3 < 0.07822
        + 0.003095932 * max(0.0, 0.0001721983 - Q.lam1_plus_lam2) / 1.441896e-05   # +0.3%  lam1_plus_lam2 < 0.0001722
        - 0.002779096 * max(0.0, 0.02227783 - Q.absphi_1) / 0.006894148   # -0.3%  absphi_1 < 0.02228
        - 0.002617 * max(0.0, Q.D2 - 2.055451) / 0.3061148   # -0.3%  D2 > 2.055
        + 0.002615887 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.3%  lam2 > 0.001131
        - 0.002557502 * max(0.0, 7.0 - Q.n_for_90pct) / 0.5603294   # -0.3%  n_for_90pct < 7
        + 0.002518169 * max(0.0, 2.371297e-05 - Q.e3) * max(0.0, Q.eccentricity - 0.8319502) / 8.425435e-07   # +0.3%  e3 < 2.371e-05 and eccentricity > 0.832
        + 0.002294587 * max(0.0, 0.06503035 - Q.z_5) / 0.007566857   # +0.2%  z_5 < 0.06503
        - 0.001819056 * max(0.0, 0.006292091 - Q.zdr_0) / 0.001006321   # -0.2%  zdr_0 < 0.006292
        - 0.001765443 * max(0.0, 988.4078 - Q.sum_pt) * max(0.0, 0.01778111 - Q.dr_2) / 0.3401129   # -0.2%  sum_pt < 988.4 and dr_2 < 0.01778
        - 0.001676333 * max(0.0, Q.sum_z_dr2_top5 - 0.01148293) / 0.00154713   # -0.2%  sum_z_dr2_top5 > 0.01148
        + 0.001649199 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # +0.2%  sum_z_dr2 > 0.01883
        + 0.001623763 * max(0.0, 0.002170915 - Q.zdr_2) / 0.0002841564   # +0.2%  zdr_2 < 0.002171
        - 0.001428766 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.1%  sum_pt_top5 > 840
        + 0.00112923 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) * max(0.0, -0.004917145 - Q.phi_0) / 9.23841e-05   # +0.1%  mass_over_sum_pt < 0.07269 and phi_0 < -0.004917
        - 0.0009201592 * max(0.0, 0.002316125 - Q.centroid_offset) / 9.872932e-05   # -0.1%  centroid_offset < 0.002316
        + 0.000905725 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, 0.02656143 - Q.tau21_b2) / 1.23798e-05   # +0.1%  sum_z_dr < 0.05465 and tau21_b2 < 0.02656
        - 0.0008746011 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.01108383   # -0.1%  n_dr_0p2_0p4 > 1 and sj3_dr_min < 0.209
        + 0.0008213895 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.1%  log_sum_pt > 6.896
        - 0.0007333022 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, 0.02656143 - Q.tau21_b2) / 8.029519e-06   # -0.1%  tau1 < 0.0437 and tau21_b2 < 0.02656
        + 0.0007025521 * max(0.0, 53.33237 - Q.mass) * max(0.0, Q.D2 - 2.843757) / 5.555775   # +0.1%  mass < 53.33 and D2 > 2.844
        - 0.000552022 * max(0.0, Q.sd_mass - 45.595) * max(0.0, Q.n_dr_0p05_0p1 - 7.0) / 0.4890223   # -0.1%  sd_mass > 45.59 and n_dr_0p05_0p1 > 7
        - 0.0004332831 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.pair_mass_0_2 - 17.9037) / 0.01350362   # -0.0%  centroid_offset < 0.01838 and pair_mass_0_2 > 17.9
        - 0.0003184864 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2) / 0.000627491   # -0.0%  log_sum_pt > 6.896 and D2_b2 < 0.7166
        - 0.0002763711 * max(0.0, Q.e3 - 0.0001869378) * max(0.0, 33.6875 - Q.pt_6) / 4.695931e-05   # -0.0%  e3 > 0.0001869 and pt_6 < 33.69
        + 0.0002743969 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.001255404 - Q.zdr_5) / 2.213187e-06   # +0.0%  log_sum_pt > 6.896 and zdr_5 < 0.001255
        - 1.02317e-05 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.mean_phi - 0.02612796) / 2.994783e-06   # -0.0%  tau1 < 0.0437 and mean_phi > 0.02613
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 8.181;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.180894 * (0.2935503
        + 0.1512681 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # +15.1%  sum_z_dr2 > 0.00752
        - 0.1130922 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -11.3%  lam1 < 0.005954
        + 0.07500468 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +7.5%  mass < 76.66
        - 0.06130558 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # -6.1%  lam1 > 0.00733
        + 0.04978096 * max(0.0, Q.lam2 - 0.0003061234) / 0.0004216599   # +5.0%  lam2 > 0.0003061
        - 0.04309589 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -4.3%  LHA > 0.3033
        - 0.0411075 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # -4.1%  sum_z_dr2 < 0.002635
        - 0.0405567 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -4.1%  lam1 < 0.004184
        - 0.03347831 * max(0.0, 0.06473447 - Q.z_7) / 0.01631636   # -3.3%  z_7 < 0.06473
        + 0.03076909 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1) / 0.002082998   # +3.1%  zdr_0 < 0.02118 and z_dr_0p05_0p1 < 0.2919
        + 0.02614847 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # +2.6%  sum_z_dr2_top3 < 0.002152
        - 0.02519826 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 6.572938 - Q.log_sum_pt) / 1.257183   # -2.5%  pt_7 < 45.75 and log_sum_pt < 6.573
        + 0.02492851 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # +2.5%  tau1 > 0.05357
        + 0.02462405 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +2.5%  e3 < 8.148e-05
        - 0.02372551 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -2.4%  lam1 < 0.001504
        - 0.02295087 * Q.e2 / 0.02863215   # -2.3%  e2
        - 0.0216664 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # -2.2%  sj3_dr_max > 0.1986
        - 0.01551264 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.planar_flow - 0.04505724) / 0.0002791469   # -1.6%  lam2 > 0.0003061 and planar_flow > 0.04506
        - 0.01302117 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -1.3%  e3 > 3.892e-05
        + 0.01259107 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +1.3%  sj3_dr_min > 0.1278
        + 0.01229724 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.002471 - Q.D2) / 1.812028   # +1.2%  pt_7 < 45.75 and D2 < 1.002
        + 0.01226981 * max(0.0, Q.sj3_pair_mass_min - 15.95929) / 1.348548   # +1.2%  sj3_pair_mass_min > 15.96
        - 0.01137835 * max(0.0, Q.mass_top5 - 45.32077) / 3.61051   # -1.1%  mass_top5 > 45.32
        - 0.01100474 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2) / 0.0002929492   # -1.1%  lam1 > 0.00733 and D2_b2 < 0.3809
        + 0.01054047 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, 1.002471 - Q.D2) / 0.009558568   # +1.1%  tau1 > 0.05357 and D2 < 1.002
        + 0.009479848 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # +0.9%  sj3_pair_mass_min > 11.05
        - 0.0093281 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -0.9%  lam2 > 0.003408
        + 0.009090521 * max(0.0, Q.mass_top5 - 22.18342) / 12.67741   # +0.9%  mass_top5 > 22.18
        + 0.007642084 * max(0.0, Q.C2_b2 - 0.009032972) / 0.001703043   # +0.8%  C2_b2 > 0.009033
        - 0.007443749 * max(0.0, Q.sj3_pair_mass_min - 11.051) * max(0.0, Q.sj3_pairmin_over_m - 0.28737) / 0.2416857   # -0.7%  sj3_pair_mass_min > 11.05 and sj3_pairmin_over_m > 0.2874
        - 0.006689887 * max(0.0, 0.02563286 - Q.M2) / 0.001872375   # -0.7%  M2 < 0.02563
        - 0.005955613 * max(0.0, 0.004918231 - Q.zdr_0) / 0.0006323791   # -0.6%  zdr_0 < 0.004918
        - 0.005198378 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.3658817 - Q.z_dr_0_0p05) / 0.002463327   # -0.5%  sj3_dr_min > 0.1278 and z_dr_0_0p05 < 0.3659
        + 0.004965806 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0008259449   # +0.5%  lam1 < 0.005954 and n_pt_above_50 > 6
        - 0.004578502 * max(0.0, Q.n_pt_above_50 - 6.0) / 0.2884353   # -0.5%  n_pt_above_50 > 6
        - 0.004225312 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.2723288 - Q.sj3_mass3) / 0.0003425911   # -0.4%  lam1 > 0.00733 and sj3_mass3 < 0.2723
        - 0.003737554 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1797097) / 5.905455e-07   # -0.4%  e3 < 8.148e-05 and sj3_dr23 > 0.1797
        + 0.003591202 * max(0.0, Q.sj3_pair_mass_min - 11.051) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 3.623232   # +0.4%  sj3_pair_mass_min > 11.05 and n_dr_0p2_0p4 > 0
        + 0.00333641 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +0.3%  sj2_dr > 0.3004
        - 0.002679264 * max(0.0, Q.sum_z_dr2 - 0.02530566) / 0.0002851418   # -0.3%  sum_z_dr2 > 0.02531
        + 0.001831756 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.2%  sum_pt > 988.4
        - 0.001512604 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.5973755 - Q.sj3_mass2) / 0.001008755   # -0.2%  sj3_dr_min > 0.1278 and sj3_mass2 < 0.5974
        - 0.001396804 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, 31.90625 - Q.pt_6) / 0.0003664138   # -0.1%  lam2 > 0.0003061 and pt_6 < 31.91
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 57.65;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 57.64568 * (-0.03874086
        - 0.2708234 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -27.1%  sum_zz_dr2 < 0.008169
        + 0.2103733 * max(0.0, 0.00817466 - Q.mass_over_sum_pt_sq) / 0.004299658   # +21.0%  mass_over_sum_pt_sq < 0.008175
        + 0.09394433 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +9.4%  lam1_plus_lam2 < 0.008678
        + 0.07789042 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +7.8%  sum_z_dr2 < 0.01324
        - 0.02609688 * max(0.0, 0.169029 - Q.sj3_dr_max) / 0.04876366   # -2.6%  sj3_dr_max < 0.169
        + 0.02589999 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +2.6%  centroid_offset < 0.03776
        + 0.02335962 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +2.3%  z_7 > 0.01686
        - 0.02321787 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # -2.3%  sum_z_dr < 0.0717
        - 0.02168943 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -2.2%  mass < 69.61
        + 0.0175689 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +1.8%  sj3_dr_max < 0.1426
        - 0.01532324 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -1.5%  lam1 < 0.008376
        - 0.01429706 * max(0.0, 0.004372139 - Q.sum_z_dr2) / 0.00156571   # -1.4%  sum_z_dr2 < 0.004372
        + 0.013495 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +1.3%  lam2 < 0.001131
        - 0.01231624 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # -1.2%  tau1 < 0.09539
        - 0.01107679 * max(0.0, 0.07637363 - Q.mass_over_sum_pt) / 0.02715027   # -1.1%  mass_over_sum_pt < 0.07637
        + 0.01040074 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +1.0%  planar_flow < 0.2534
        + 0.009852852 * max(0.0, 0.2623172 - Q.sj3_dr_max) / 0.1129006   # +1.0%  sj3_dr_max < 0.2623
        - 0.009505348 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt) / 3.513015   # -1.0%  centroid_offset < 0.03776 and sum_pt < 901.6
        + 0.009283875 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # +0.9%  max_dr < 0.1452
        - 0.008610451 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # -0.9%  C2 < 0.03579
        - 0.007485389 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 15.95929 - Q.sj3_pair_mass_min) / 0.05098851   # -0.7%  centroid_offset < 0.01438 and sj3_pair_mass_min < 15.96
        + 0.007189737 * max(0.0, 49.6681 - Q.mass) / 17.06893   # +0.7%  mass < 49.67
        + 0.00572671 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) / 0.00293624   # +0.6%  lam1_plus_lam2 < 0.006679
        + 0.005427947 * max(0.0, 0.09538712 - Q.tau1) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.03209574   # +0.5%  tau1 < 0.09539 and z_dr_0p05_0p1 < 0.846
        + 0.005239125 * max(0.0, 1.050302e-05 - Q.e3) / 3.717236e-06   # +0.5%  e3 < 1.05e-05
        - 0.004922689 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.zdr_6 - 0.007390416) / 3.11545e-06   # -0.5%  centroid_offset > 0.0499 and zdr_6 > 0.00739
        - 0.00450026 * max(0.0, 0.02689598 - Q.sum_z_dr) / 0.004330137   # -0.5%  sum_z_dr < 0.0269
        - 0.004391195 * max(0.0, Q.pt_7 - 29.04219) / 7.785987   # -0.4%  pt_7 > 29.04
        - 0.004064777 * max(0.0, 0.1027585 - Q.max_dr) / 0.02411847   # -0.4%  max_dr < 0.1028
        - 0.003821335 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -0.4%  sj2_dr > 0.1779
        - 0.003012784 * max(0.0, 506.875 - Q.sum_pt_top5) / 34.75344   # -0.3%  sum_pt_top5 < 506.9
        - 0.002727334 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.3%  centroid_offset > 0.0499
        - 0.002411908 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -0.2%  lam1 < 0.00733
        + 0.002199553 * max(0.0, 0.01655442 - Q.e2) / 0.003887347   # +0.2%  e2 < 0.01655
        - 0.002086083 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001130645 - Q.lam2) / 1.843299e-05   # -0.2%  sj2_dr > 0.1779 and lam2 < 0.001131
        + 0.002016303 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0007565864   # +0.2%  centroid_offset < 0.01438 and D2_b2 < 0.5327
        + 0.002000802 * max(0.0, 15.45403 - Q.mass) * max(0.0, 0.003780225 - Q.zdr_7) / 0.006870481   # +0.2%  mass < 15.45 and zdr_7 < 0.00378
        - 0.001968353 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 0.0345459 - Q.absphi_0) / 0.0003803849   # -0.2%  centroid_offset < 0.03776 and absphi_0 < 0.03455
        + 0.001941527 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # +0.2%  centroid_offset > 0.0499 and D2 < 2.844
        + 0.001841341 * max(0.0, Q.eccentricity - 0.9884745) / 0.001961982   # +0.2%  eccentricity > 0.9885
        - 0.00169394 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 4.721224 - Q.D2_b2) / 16.05793   # -0.2%  pt_6 < 41.22 and D2_b2 < 4.721
        - 0.00159746 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, Q.sj3_pair_mass_min - 1.282345) / 0.4025762   # -0.2%  planar_flow < 0.2534 and sj3_pair_mass_min > 1.282
        - 0.001567373 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 1.050302e-05 - Q.e3) / 1.771164e-07   # -0.2%  planar_flow < 0.2534 and e3 < 1.05e-05
        - 0.001479099 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, Q.pt_6 - 19.46875) / 0.1742442   # -0.1%  sum_z_dr2 < 0.01324 and pt_6 > 19.47
        - 0.001460922 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.1%  planar_flow < 0.2534 and sum_pt < 840
        + 0.001276368 * max(0.0, 15.45403 - Q.mass) / 2.385786   # +0.1%  mass < 15.45
        - 0.001121072 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.00279494 - Q.zdr_7) / 5.690817e-06   # -0.1%  centroid_offset < 0.01438 and zdr_7 < 0.002795
        - 0.001118811 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.5502779 - Q.tau32) / 0.0001096908   # -0.1%  centroid_offset > 0.0499 and tau32 < 0.5503
        + 0.001094915 * max(0.0, 0.01437952 - Q.centroid_offset) / 0.004306483   # +0.1%  centroid_offset < 0.01438
        - 0.0009551923 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.03601074 - Q.abseta_4) / 8.179649e-05   # -0.1%  centroid_offset < 0.01438 and abseta_4 < 0.03601
        - 0.0008931241 * max(0.0, Q.eccentricity - 0.9884745) * max(0.0, 2.771069 - Q.sj2_mass2) / 0.00304207   # -0.1%  eccentricity > 0.9885 and sj2_mass2 < 2.771
        + 0.0007675659 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.1%  sj2_dr > 0.2688
        - 0.0006831316 * max(0.0, 24.42188 - Q.pt_6) / 0.6046572   # -0.1%  pt_6 < 24.42
        - 0.0006800392 * max(0.0, 0.02689598 - Q.sum_z_dr) * max(0.0, Q.sj3_pair_mass_min - 1.590484) / 0.005084766   # -0.1%  sum_z_dr < 0.0269 and sj3_pair_mass_min > 1.59
        + 0.0006293635 * max(0.0, 0.003676313 - Q.zdr_0) / 0.0003550098   # +0.1%  zdr_0 < 0.003676
        - 0.0005964556 * max(0.0, Q.n_dr_0p1_0p2 - 3.0) / 0.3318622   # -0.1%  n_dr_0p1_0p2 > 3
        + 0.0005633462 * max(0.0, 506.875 - Q.sum_pt_top5) * max(0.0, Q.z_dr_0p1_0p2 - 0.04510668) / 8.800637   # +0.1%  sum_pt_top5 < 506.9 and z_dr_0p1_0p2 > 0.04511
        - 0.000537561 * max(0.0, 49.6681 - Q.mass) * max(0.0, 1.332146 - Q.D2) / 1.334408   # -0.1%  mass < 49.67 and D2 < 1.332
        + 0.0004867285 * max(0.0, 0.004372139 - Q.sum_z_dr2) * max(0.0, Q.sj3_dr13 - 0.1659434) / 5.846229e-06   # +0.0%  sum_z_dr2 < 0.004372 and sj3_dr13 > 0.1659
        + 0.0003938252 * max(0.0, Q.max_pair_mass - 33.3761) / 0.9279851   # +0.0%  max_pair_mass > 33.38
        + 0.0002725828 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, 0.415246 - Q.D2) / 8.563062e-05   # +0.0%  sum_z_dr2 < 0.01324 and D2 < 0.4152
        + 0.0001301864 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.tau3 - 0.001996306) / 6.194014e-06   # +0.0%  centroid_offset > 0.0499 and tau3 > 0.001996
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.5423;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5422533 * (-0.7179375
        + 0.3399462 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # +34.0%  sum_z_dr2 > 0.01883
        - 0.1276923 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -12.8%  e2 > 0.06344
        - 0.104238 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, 53.4375 - Q.pt_7) / 0.01439677   # -10.4%  sum_z_dr2 > 0.01883 and pt_7 < 53.44
        + 0.09123106 * max(0.0, Q.mass - 91.19) / 0.5839191   # +9.1%  mass > 91.19
        - 0.06695817 * max(0.0, Q.mass - 69.61135) / 2.406702   # -6.7%  mass > 69.61
        - 0.06220864 * max(0.0, Q.sum_z_dr2_top2 - 0.01403324) / 0.001129575   # -6.2%  sum_z_dr2_top2 > 0.01403
        + 0.05644651 * max(0.0, Q.mass - 69.61135) * max(0.0, Q.centroid_offset - 0.009480685) / 0.04081507   # +5.6%  mass > 69.61 and centroid_offset > 0.009481
        + 0.04393464 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, Q.lam2 - 0.000537286) / 2.828867e-06   # +4.4%  sum_z_dr2 > 0.01883 and lam2 > 0.0005373
        + 0.04227095 * max(0.0, Q.mass - 69.61135) * max(0.0, 6.423044 - Q.log_sum_pt) / 0.0982281   # +4.2%  mass > 69.61 and log_sum_pt < 6.423
        + 0.0347235 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +3.5%  centroid_offset > 0.0499
        - 0.03035007 * max(0.0, Q.zdr_0 - 0.03981924) / 0.0005695785   # -3.0%  zdr_0 > 0.03982
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 20.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.22765 * (0.02338177
        + 0.3334913 * max(0.0, 0.1484084 - Q.sum_z_dr) / 0.0905999   # +33.3%  sum_z_dr < 0.1484
        - 0.1472787 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -14.7%  e2 < 0.08001
        - 0.08491339 * max(0.0, 0.007520088 - Q.lam1_plus_lam2) / 0.003544991   # -8.5%  lam1_plus_lam2 < 0.00752
        + 0.0766048 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +7.7%  lam1 < 0.01643
        - 0.06355609 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -6.4%  sum_z_dr < 0.1484 and log_sum_pt < 6.804
        + 0.0525295 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +5.3%  lam1 < 0.006507
        - 0.03758259 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 8.324263e-07   # -3.8%  e3 < 5.335e-05 and centroid_offset < 0.03776
        + 0.03460471 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # +3.5%  e3 < 5.335e-05
        - 0.03360272 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -3.4%  sum_z_dr < 0.1484 and pt_7 < 38.53
        + 0.02320599 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7) / 781.5808   # +2.3%  sum_pt_top5 > 658.1 and pt_7 < 40.04
        + 0.01630185 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +1.6%  lam1_plus_lam2 < 0.01324
        - 0.01541419 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.000537286 - Q.lam2) / 4.021081e-05   # -1.5%  sum_z_dr < 0.1484 and lam2 < 0.0005373
        + 0.01094894 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +1.1%  log_sum_pt > 6.503
        - 0.008910118 * max(0.0, Q.mass - 49.6681) / 7.721594   # -0.9%  mass > 49.67
        - 0.007179376 * max(0.0, 31.90625 - Q.pt_6) / 1.826854   # -0.7%  pt_6 < 31.91
        - 0.006872438 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # -0.7%  z_7 < 0.02807
        - 0.006499763 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # -0.6%  lam2 < 0.0001948
        + 0.00588265 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.z_7 - 0.06164517) / 0.0003465887   # +0.6%  sum_z_dr < 0.1484 and z_7 > 0.06165
        - 0.005030043 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55592   # -0.5%  sum_pt_top5 > 658.1
        + 0.004117888 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 31.90625 - Q.pt_6) / 134.9952   # +0.4%  sum_pt_top5 > 791.1 and pt_6 < 31.91
        - 0.003026842 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.zdr_7 - 0.0007928864) / 0.0001818422   # -0.3%  log_sum_pt > 6.503 and zdr_7 > 0.0007929
        + 0.002988066 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.07474969 - Q.M3) / 0.0006773733   # +0.3%  sum_z_dr < 0.1484 and M3 < 0.07475
        - 0.00284964 * max(0.0, Q.sj3_dr23 - 0.1974628) / 0.02478939   # -0.3%  sj3_dr23 > 0.1975
        + 0.002797993 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.tau2 - 0.008780509) / 0.000269945   # +0.3%  sum_z_dr < 0.1484 and tau2 > 0.008781
        - 0.002464811 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7) / 0.06765006   # -0.2%  pt_6 < 31.91 and z_7 < 0.05861
        + 0.002400186 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +0.2%  sum_pt_top5 > 791.1
        - 0.002211737 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.2%  z_6 < 0.0216
        - 0.001753194 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # -0.2%  z_5 < 0.02818
        - 0.001512439 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6) / 104.494   # -0.2%  sum_pt > 988.4 and pt_6 < 62.25
        + 0.001230035 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.pair_mass_0_7 - 10.2219) / 0.2907061   # +0.1%  log_sum_pt > 6.503 and pair_mass_0_7 > 10.22
        - 0.001133299 * max(0.0, Q.pt_7 - 48.71875) / 0.6759439   # -0.1%  pt_7 > 48.72
        - 0.000682424 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.sj2_mass1 - 31.78116) / 0.01215768   # -0.1%  sum_z_dr < 0.1484 and sj2_mass1 > 31.78
        - 0.0004222495 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.08151794 - Q.M3) / 0.08390625   # -0.0%  sum_pt > 988.4 and M3 < 0.08152
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 38.25;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 38.24941 * (-0.01685157
        - 0.09645868 * max(0.0, Q.lam1_plus_lam2 - 0.006679471) / 0.002702003   # -9.6%  lam1_plus_lam2 > 0.006679
        - 0.07897312 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -7.9%  sum_z_dr2 > 0.00752
        + 0.06091353 * max(0.0, Q.sum_z_dr - 0.02689598) / 0.03613842   # +6.1%  sum_z_dr > 0.0269
        + 0.04997327 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # +5.0%  sj2_dr > 0.1592
        + 0.04589842 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +4.6%  mass_over_sum_pt > 0.08475
        - 0.04572716 * max(0.0, 0.324646 - Q.sd_rg) / 0.2082097   # -4.6%  sd_rg < 0.3246
        + 0.04541495 * max(0.0, Q.lam1_plus_lam2 - 0.003562611) / 0.004066872   # +4.5%  lam1_plus_lam2 > 0.003563
        + 0.04479352 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +4.5%  mass_over_sum_pt > 0.07992
        - 0.0393689 * max(0.0, Q.e2 - 0.01655442) / 0.01596508   # -3.9%  e2 > 0.01655
        + 0.03704173 * max(0.0, Q.sum_zz_dr2 - 0.002074109) / 0.004484017   # +3.7%  sum_zz_dr2 > 0.002074
        - 0.0331413 * max(0.0, Q.sum_zz_dr2 - 0.0030133) / 0.003940214   # -3.3%  sum_zz_dr2 > 0.003013
        + 0.03018894 * max(0.0, Q.sum_z_dr - 0.04081947) / 0.0270611   # +3.0%  sum_z_dr > 0.04082
        + 0.02867829 * max(0.0, 49.91626 - Q.sd_mass) / 22.60916   # +2.9%  sd_mass < 49.92
        - 0.02540564 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # -2.5%  sum_z_dr > 0.08724
        - 0.02357661 * max(0.0, Q.sj2_dr - 0.2001708) / 0.02317349   # -2.4%  sj2_dr > 0.2002
        + 0.02345539 * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 0.001433269   # +2.3%  sum_zz_dr2 > 0.01166
        - 0.01571634 * max(0.0, Q.sj2_dr - 0.06154135) / 0.1002688   # -1.6%  sj2_dr > 0.06154
        - 0.01464696 * max(0.0, Q.sum_z_dr2 - 0.0009641429) / 0.00568632   # -1.5%  sum_z_dr2 > 0.0009641
        - 0.01412097 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -1.4%  sj2_dr > 0.1779
        - 0.01402749 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -1.4%  mass_over_sum_pt > 0.09041
        - 0.01393659 * max(0.0, 0.004032342 - Q.C2_b2) / 0.002883542   # -1.4%  C2_b2 < 0.004032
        + 0.01334432 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # +1.3%  lam2 < 0.0005373
        + 0.013142 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +1.3%  e2 > 0.04111
        - 0.01291671 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -1.3%  e3 < 8.148e-05
        + 0.01013179 * max(0.0, 0.1115136 - Q.planar_flow) / 0.03343187   # +1.0%  planar_flow < 0.1115
        + 0.009342163 * max(0.0, 0.07870506 - Q.z_dr_0p1_0p2) / 0.04317253   # +0.9%  z_dr_0p1_0p2 < 0.07871
        + 0.008950724 * max(0.0, Q.z_dr_0_0p05 - 0.1515405) / 0.4480651   # +0.9%  z_dr_0_0p05 > 0.1515
        - 0.008903103 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.9%  centroid_offset > 0.0499
        - 0.008675713 * max(0.0, 74.57663 - Q.sd_mass) / 42.04056   # -0.9%  sd_mass < 74.58
        - 0.008503074 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.9%  psi_0p1 > 0.9761
        + 0.008491098 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +0.8%  LHA > 0.3033
        - 0.008218648 * max(0.0, 715.4688 - Q.sum_pt) / 69.91196   # -0.8%  sum_pt < 715.5
        - 0.007995217 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -0.8%  planar_flow < 0.1115 and lam1 < 0.00733
        - 0.006053677 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.6%  centroid_offset > 0.03776
        - 0.005464277 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # -0.5%  e3 < 5.335e-05
        + 0.005372197 * max(0.0, 5.0 - Q.n_dr_0_0p05) / 1.94322   # +0.5%  n_dr_0_0p05 < 5
        + 0.00532898 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0006014625   # +0.5%  sj2_dr > 0.1592 and D2_b2 < 0.08499
        - 0.005297293 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 0.02415398 - Q.C2_b2) / 0.0005249308   # -0.5%  z_dr_0p05_0p1 > 0.7509 and C2_b2 < 0.02415
        - 0.005140812 * max(0.0, Q.centroid_offset - 0.02685622) / 0.003254849   # -0.5%  centroid_offset > 0.02686
        + 0.005053743 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # +0.5%  sum_z_dr2 > 0.008678
        + 0.004571979 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067297   # +0.5%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        + 0.004507558 * max(0.0, Q.psi_0p1 - 0.7143804) / 0.1805043   # +0.5%  psi_0p1 > 0.7144
        - 0.004310522 * max(0.0, 0.163898 - Q.z_dr_0p05_0p1) / 0.08682106   # -0.4%  z_dr_0p05_0p1 < 0.1639
        - 0.004285215 * max(0.0, Q.sd_rg - 0.2330919) / 0.01070351   # -0.4%  sd_rg > 0.2331
        + 0.003907058 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 3.351526e-05   # +0.4%  planar_flow < 0.1115 and lam1_plus_lam2 < 0.006097
        + 0.003547545 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # +0.4%  sd_rg > 0.2788
        - 0.003513753 * max(0.0, Q.sj2_dr - 0.1294903) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0009393137   # -0.4%  sj2_dr > 0.1295 and D2_b2 < 0.08499
        - 0.003221117 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # -0.3%  lam1_plus_lam2 > 0.00559
        + 0.002973839 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, 150.625 - Q.pt_1) / 0.1837865   # +0.3%  centroid_offset > 0.02686 and pt_1 < 150.6
        + 0.002765738 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +0.3%  e2 > 0.05028
        - 0.002757797 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # -0.3%  sj2_dr > 0.1295
        - 0.00269243 * max(0.0, Q.mass - 69.61135) / 2.406702   # -0.3%  mass > 69.61
        - 0.002401495 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.2%  planar_flow < 0.1115 and centroid_offset < 0.01838
        + 0.002351568 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.0008333816 - Q.C2_b2) / 1.341084e-07   # +0.2%  centroid_offset > 0.0499 and C2_b2 < 0.0008334
        + 0.001991431 * max(0.0, Q.mass_top5 - 49.18618) / 2.747087   # +0.2%  mass_top5 > 49.19
        - 0.001912485 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0003170016   # -0.2%  sj2_dr > 0.2002 and D2_b2 < 0.08499
        + 0.001912438 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.05056028 - Q.dr_2) / 0.0001260892   # +0.2%  sj2_dr > 0.2002 and dr_2 < 0.05056
        + 0.001693879 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 715.4688 - Q.sum_pt) / 22.02426   # +0.2%  z_dr_0p05_0p1 < 0.5883 and sum_pt < 715.5
        + 0.001607994 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1629004) / 4.043869e-07   # +0.2%  e3 < 5.335e-05 and sj3_dr23 > 0.1629
        - 0.001186671 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 658.125 - Q.sum_pt_top5) / 2.950014   # -0.1%  planar_flow < 0.1115 and sum_pt_top5 < 658.1
        + 0.001152899 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.3724174   # +0.1%  z_dr_0p05_0p1 < 0.5883
        - 0.0009634619 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.03293672 - Q.dr_2) / 6.256053e-05   # -0.1%  sj2_dr > 0.1592 and dr_2 < 0.03294
        + 0.0007998658 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.05268713 - Q.dr_3) / 0.0001291671   # +0.1%  sj2_dr > 0.2002 and dr_3 < 0.05269
        - 0.0004763704 * max(0.0, Q.sd_rg - 0.2787955) * max(0.0, 0.2669656 - Q.D2_b2) / 0.0004100902   # -0.0%  sd_rg > 0.2788 and D2_b2 < 0.267
        - 0.0002843641 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 0.002231562   # -0.0%  planar_flow < 0.1115 and z_dr_0p05_0p1 > 0.6748
        - 0.0002681576 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.02358801 - Q.dr_3) / 2.035984e-05   # -0.0%  sj2_dr > 0.1592 and dr_3 < 0.02359
        - 0.0001590274 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # -0.0%  lam1_plus_lam2 > 0.01324
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 48.97;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 48.9739 * (-0.0144817
        + 0.1719231 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +17.2%  lam1_plus_lam2 < 0.01324
        - 0.1168011 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # -11.7%  mass_over_sum_pt < 0.1309
        + 0.1053602 * max(0.0, 0.01882765 - Q.lam1_plus_lam2) / 0.01314296   # +10.5%  lam1_plus_lam2 < 0.01883
        - 0.09752975 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # -9.8%  sum_zz_dr2 < 0.01166
        - 0.08353865 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -8.4%  sum_z_dr < 0.1019
        - 0.06869784 * max(0.0, 0.007520088 - Q.sum_z_dr2) / 0.00354499   # -6.9%  sum_z_dr2 < 0.00752
        + 0.06632508 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +6.6%  mass_over_sum_pt_sq < 0.007183
        + 0.06438911 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +6.4%  sj2_dr < 0.1592
        + 0.04604581 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +4.6%  e2 < 0.04111
        - 0.03969779 * max(0.0, 0.2001708 - Q.sj2_dr) / 0.07331403   # -4.0%  sj2_dr < 0.2002
        - 0.03498983 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -3.5%  sum_zz_dr2 < 0.008169
        + 0.02351751 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +2.4%  lam1 < 0.00484
        - 0.02103156 * max(0.0, 0.1492731 - Q.sj2_dr) / 0.04374278   # -2.1%  sj2_dr < 0.1493
        - 0.01986608 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 0.002544039   # -2.0%  lam1_plus_lam2 < 0.006097
        + 0.01591673 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +1.6%  lam2 < 0.001131
        - 0.004814742 * max(0.0, 0.007520088 - Q.sum_z_dr2) * max(0.0, 0.7459513 - Q.D2) / 9.127003e-05   # -0.5%  sum_z_dr2 < 0.00752 and D2 < 0.746
        - 0.004580981 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # -0.5%  N2 < 0.2233
        + 0.003928522 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +0.4%  N2 < 0.2233 and sum_pt_top5 > 430.8
        + 0.003398513 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2) / 0.003993721   # +0.3%  mass_over_sum_pt < 0.1309 and D2 < 0.746
        + 0.002308554 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # +0.2%  lam1 < 0.002464
        - 0.002027253 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # -0.2%  sum_z_dr2_top3 < 0.002152
        + 0.001522328 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) * max(0.0, 0.7459513 - Q.D2) / 3.18925e-05   # +0.2%  lam1_plus_lam2 < 0.006097 and D2 < 0.746
        - 0.001215059 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.0114178   # -0.1%  N2 < 0.2233 and z_dr_0p05_0p1 < 0.5883
        + 0.0002950485 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.4684459) / 0.0001955483   # +0.0%  mass_over_sum_pt < 0.1309 and z_dr_0p1_0p2 > 0.4684
        - 0.000278926 * max(0.0, Q.sj3_pair_mass_max - 80.4) / 0.3254107   # -0.0%  sj3_pair_mass_max > 80.4
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0199077731092436, 0.7226136554621849, 2.083326680672269, 0.9447821428571429, 1.1283756302521009, 1.7552281512605041, 0.9350460084033614, 1.8265094537815125, 0.24187836134453783, 2.983231512605042, 2.016080882352941, 2.1297355042016806, 0.09673949579831932, 3.2901508403361346, 0.49691197478991594, 0.3491878151260504]
T = [2.3161915720194326, 1.3513551585477943, 3.3337525899422267, 2.458223703387605, 2.84121498490021]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +39%, n9 +22%, n5 -14%, n1 +12%, n0 -7%, n6 +4% ...
            + 0.3864876 * h[2] / H_AVG[2]
            + 0.2213733 * h[9] / H_AVG[9]
            - 0.142089 * h[5] / H_AVG[5]
            + 0.1218686 * h[1] / H_AVG[1]
            - 0.06880285 * h[0] / H_AVG[0]
            + 0.04415466 * h[6] / H_AVG[6]
            - 0.01522402 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -19%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5605197 * h[9] / H_AVG[9]
            - 0.186487 * h[10] / H_AVG[10]
            + 0.08649151 * h[6] / H_AVG[6]
            - 0.07828084 * h[4] / H_AVG[4]
            + 0.0608843 * h[5] / H_AVG[5]
            + 0.01614989 * h[15] / H_AVG[15]
            + 0.01118684 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +12%, n14 -11%, n0 +11%, n6 -9% ...
            + 0.2395651 * h[11] / H_AVG[11]
            - 0.1416995 * h[3] / H_AVG[3]
            + 0.1198496 * h[7] / H_AVG[7]
            - 0.1117911 * h[14] / H_AVG[14]
            + 0.1051648 * h[0] / H_AVG[0]
            - 0.08764954 * h[6] / H_AVG[6]
            - 0.07201093 * h[15] / H_AVG[15]
            + 0.06939289 * h[13] / H_AVG[13]
            - 0.02796428 * h[9] / H_AVG[9]
            - 0.0181386 * h[8] / H_AVG[8]
            - 0.006773651 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -22%, n6 -14%, n14 +8%, n13 +7%, n9 -4% ...
            + 0.3482906 * h[7] / H_AVG[7]
            - 0.2161886 * h[3] / H_AVG[3]
            - 0.1426405 * h[6] / H_AVG[6]
            + 0.07580351 * h[14] / H_AVG[14]
            + 0.07319518 * h[13] / H_AVG[13]
            - 0.03792413 * h[9] / H_AVG[9]
            + 0.03674471 * h[1] / H_AVG[1]
            + 0.03586099 * h[4] / H_AVG[4]
            - 0.02219513 * h[15] / H_AVG[15]
            + 0.01115661 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +27%, n5 -15%, n4 +5%, n3 +2%, n12 -2% ...
            - 0.4704409 * h[13] / H_AVG[13]
            + 0.266094 * h[10] / H_AVG[10]
            - 0.1544434 * h[5] / H_AVG[5]
            + 0.04964318 * h[4] / H_AVG[4]
            + 0.02078297 * h[3] / H_AVG[3]
            - 0.01702432 * h[12] / H_AVG[12]
            + 0.01596225 * h[8] / H_AVG[8]
            + 0.005608889 * h[0] / H_AVG[0]
        ),
    ]]


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
