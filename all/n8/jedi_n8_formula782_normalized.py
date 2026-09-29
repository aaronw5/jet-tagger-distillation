"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.5%   (on for 90% of jets)
  neuron  9:  11.9%   (on for 61% of jets)
  neuron  7:  10.7%   (on for 61% of jets)
  neuron  3:   9.6%   (on for 25% of jets)
  neuron 10:   8.1%   (on for 84% of jets)
  neuron  2:   7.1%   (on for 88% of jets)
  neuron  5:   7.0%   (on for 62% of jets)
  neuron  6:   7.0%   (on for 33% of jets)
  neuron 11:   6.1%   (on for 73% of jets)
  neuron  0:   3.9%   (on for 40% of jets)
  neuron 14:   3.7%   (on for 26% of jets)
  neuron  1:   3.4%   (on for 64% of jets)
  neuron  4:   3.0%   (on for 52% of jets)
  neuron 15:   2.4%   (on for 28% of jets)
  neuron  8:   1.1%   (on for 34% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.7% (the network: 65.8%); same class as the network for 90.3% of jets.

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
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_4          mass of particles 0 and 4 [GeV]
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.pair_mass_0_7          mass of particles 0 and 7 [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m01                    mass of particles 0 and 1 [GeV]
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_1               |Δη| of particle 1
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr1_6                  ΔR between particle 6 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr02                   ΔR between particles 0 and 2
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
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
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
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
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_4=pair_mass(0, 4),
        pair_mass_0_6=pair_mass(0, 6),
        pair_mass_0_7=pair_mass(0, 7),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        max_dr=max(dr[i] for i in real),
        m01=pair_mass(0, 1),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_90pct=ncum(0.9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        zdr_2=z[2] * dr[2],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        abseta_0=abs(eta[0]),
        abseta_1=abs(eta[1]),
        abseta_4=abs(eta[4]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr1_6=math.sqrt(dist2(1, 6)) if pt[6] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr02=math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
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
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 30.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 30.67257 * (-0.07398584
        + 0.07820913 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +7.8%  sum_z_dr2 < 0.01324
        + 0.0727857 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +7.3%  lam1_plus_lam2 < 0.008678
        - 0.06127323 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) / 0.00156571   # -6.1%  lam1_plus_lam2 < 0.004372
        - 0.05894308 * max(0.0, 0.08475161 - Q.mass_over_sum_pt) / 0.03304118   # -5.9%  mass_over_sum_pt < 0.08475
        + 0.05694131 * max(0.0, Q.sj3_dr_max - 0.1070199) / 0.08518534   # +5.7%  sj3_dr_max > 0.107
        + 0.05063325 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +5.1%  sj3_dr_max < 0.3012
        - 0.04364467 * max(0.0, 0.007929074 - Q.sum_z_dr2_top3) / 0.004560262   # -4.4%  sum_z_dr2_top3 < 0.007929
        + 0.04198336 * max(0.0, 0.006756161 - Q.sum_z_dr2_top3) / 0.003650633   # +4.2%  sum_z_dr2_top3 < 0.006756
        + 0.03859899 * max(0.0, 0.03556091 - Q.e2) / 0.01371972   # +3.9%  e2 < 0.03556
        - 0.03731963 * max(0.0, 0.07608178 - Q.sum_z_dr) / 0.02796784   # -3.7%  sum_z_dr < 0.07608
        - 0.03302615 * max(0.0, 64.61873 - Q.mass) / 27.57279   # -3.3%  mass < 64.62
        - 0.02943821 * max(0.0, Q.centroid_offset - 0.006789738) / 0.01140695   # -2.9%  centroid_offset > 0.00679
        - 0.02617944 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -2.6%  lam1 < 0.005433
        - 0.02617472 * max(0.0, Q.sj3_dr_max - 0.1789613) / 0.04320285   # -2.6%  sj3_dr_max > 0.179
        + 0.02580661 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +2.6%  planar_flow < 0.1484
        - 0.02549566 * max(0.0, 29.6447 - Q.mass) / 7.34723   # -2.5%  mass < 29.64
        - 0.02022687 * max(0.0, 0.001563465 - Q.C2_b2) / 0.0008800039   # -2.0%  C2_b2 < 0.001563
        + 0.0196159 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +2.0%  lam1 < 0.006507
        + 0.01918281 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # +1.9%  mass_over_sum_pt < 0.1309
        + 0.01633565 * max(0.0, 0.0245477 - Q.e2) / 0.007455821   # +1.6%  e2 < 0.02455
        - 0.01484008 * max(0.0, 0.01882765 - Q.sum_z_dr2) / 0.01314296   # -1.5%  sum_z_dr2 < 0.01883
        - 0.01458602 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -1.5%  sum_z_dr < 0.08724
        - 0.01303755 * max(0.0, 0.04889979 - Q.sj3_dr_max) / 0.006377687   # -1.3%  sj3_dr_max < 0.0489
        - 0.0124696 * max(0.0, 7.300726e-05 - Q.lam2) / 2.767163e-05   # -1.2%  lam2 < 7.301e-05
        - 0.01209665 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -1.2%  sj3_dr_max > 0.2337
        - 0.01092129 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -1.1%  centroid_offset > 0.0499
        + 0.01090818 * max(0.0, 0.003904593 - Q.mass_over_sum_pt_sq) / 0.001485439   # +1.1%  mass_over_sum_pt_sq < 0.003905
        - 0.01081599 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, Q.z_7 - 0.01685855) / 0.0006290825   # -1.1%  log_sum_pt > 6.67 and z_7 > 0.01686
        + 0.01080587 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) * max(0.0, 0.03532852 - Q.C3) / 3.6978e-05   # +1.1%  lam1_plus_lam2 < 0.004372 and C3 < 0.03533
        - 0.009579071 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05557716 - Q.z_7) / 0.001734349   # -1.0%  sj3_dr_max < 0.3012 and z_7 < 0.05558
        + 0.008880708 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, 1.002471 - Q.D2) / 0.001008123   # +0.9%  sum_z_dr2 < 0.01324 and D2 < 1.002
        + 0.008362623 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +0.8%  log_sum_pt > 6.378
        - 0.007990005 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.001732318   # -0.8%  planar_flow < 0.1484 and centroid_offset < 0.0499
        + 0.007590186 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2) / 2.36013e-06   # +0.8%  lam2 < 7.301e-05 and D2_b2 < 0.267
        + 0.007192142 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 0.07232166 - Q.dr_4) / 0.001949106   # +0.7%  log_sum_pt > 6.67 and dr_4 < 0.07232
        + 0.006746702 * max(0.0, Q.centroid_offset - 0.006789738) * max(0.0, 223.375 - Q.pt_1) / 1.309723   # +0.7%  centroid_offset > 0.00679 and pt_1 < 223.4
        - 0.006531157 * max(0.0, Q.sum_pt - 901.5938) / 12.40362   # -0.7%  sum_pt > 901.6
        + 0.004972419 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.08525808 - Q.dr_2) / 1.41829e-06   # +0.5%  lam2 < 7.301e-05 and dr_2 < 0.08526
        - 0.004125577 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # -0.4%  lam1 < 0.006507 and D2 < 0.8757
        + 0.003972953 * max(0.0, Q.centroid_offset - 0.006789738) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.02705531   # +0.4%  centroid_offset > 0.00679 and n_pt_above_50 < 7
        - 0.003679567 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.06336451 - Q.dr_4) / 1.344914   # -0.4%  sum_pt_top5 > 687.4 and dr_4 < 0.06336
        + 0.003307076 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, Q.phi_0 - -0.008995056) / 0.0001281354   # +0.3%  sum_z_dr2 < 0.01324 and phi_0 > -0.008995
        + 0.003015845 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 28.78966 - Q.m01) / 1.085363   # +0.3%  log_sum_pt > 6.67 and m01 < 28.79
        + 0.00299488 * max(0.0, 0.0002758826 - Q.lam1) / 3.586469e-05   # +0.3%  lam1 < 0.0002759
        - 0.002988013 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 358.375 - Q.sum_pt_top2) / 2.426548   # -0.3%  planar_flow < 0.1484 and sum_pt_top2 < 358.4
        - 0.002338655 * max(0.0, 0.007929074 - Q.sum_z_dr2_top3) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0005550791   # -0.2%  sum_z_dr2_top3 < 0.007929 and D2_b2 < 0.5327
        - 0.001535215 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 0.06473447 - Q.z_7) / 3.579716e-05   # -0.2%  centroid_offset > 0.02077 and z_7 < 0.06473
        + 0.001508384 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.4160096 - Q.pt_dispersion) / 0.0006933006   # +0.2%  planar_flow < 0.1484 and pt_dispersion < 0.416
        - 0.001435755 * max(0.0, 56.92035 - Q.mass) / 21.78475   # -0.1%  mass < 56.92
        - 0.001414476 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05744392 - Q.D2_b2) / 0.0005632394   # -0.1%  sj3_dr_max < 0.3012 and D2_b2 < 0.05744
        + 0.001384268 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.1%  sum_pt_top5 > 687.4
        - 0.001261944 * max(0.0, 64.61873 - Q.mass) * max(0.0, 0.1830092 - Q.D2_b2) / 0.3382769   # -0.1%  mass < 64.62 and D2_b2 < 0.183
        - 0.001206574 * max(0.0, 6.080494 - Q.log_sum_pt) / 0.006097657   # -0.1%  log_sum_pt < 6.08
        + 0.001122344 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.pt_7 - 33.21875) / 68.87156   # +0.1%  sum_pt > 901.6 and pt_7 > 33.22
        - 0.0008123277 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 0.05056028 - Q.dr_2) / 1.335632e-05   # -0.1%  centroid_offset > 0.02077 and dr_2 < 0.05056
        - 0.0005151037 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.02270492 - Q.dr_2) / 2.517754e-05   # -0.1%  planar_flow < 0.1484 and dr_2 < 0.0227
        - 0.0005102108 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, Q.dr0_6 - 0.1755206) / 0.0008435917   # -0.1%  planar_flow < 0.1484 and dr0_6 > 0.1755
        - 0.0003723373 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -0.0%  log_sum_pt > 6.67
        + 0.0003578914 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.02320757 - Q.z_7) / 2.048782e-05   # +0.0%  planar_flow < 0.1484 and z_7 < 0.02321
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 18.25;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.24516 * (-0.008224408
        - 0.1135057 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -11.4%  sum_z_dr2 < 0.008678
        + 0.09664326 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) / 0.002578466   # +9.7%  mass_over_sum_pt_sq < 0.005833
        - 0.08138611 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -8.1%  sum_pt_top5 > 531.2
        - 0.07082664 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 0.002544039   # -7.1%  lam1_plus_lam2 < 0.006097
        + 0.06500025 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +6.5%  log_sum_pt > 6.503
        - 0.0608363 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -6.1%  sum_z_dr < 0.1019
        + 0.04464908 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +4.5%  log_sum_pt > 6.378
        + 0.04273073 * max(0.0, Q.sum_pt_top5 - 367.5938) / 230.8403   # +4.3%  sum_pt_top5 > 367.6
        + 0.03760253 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +3.8%  e3 < 0.0005117
        + 0.03525711 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +3.5%  lam1 < 0.005954
        - 0.02690565 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -2.7%  z_7 < 0.06165
        + 0.02569312 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # +2.6%  z_7 > 0.02321
        + 0.02455644 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +2.5%  lam1 < 0.008376
        + 0.02411612 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # +2.4%  sum_zz_dr2 < 0.008169
        + 0.02398626 * max(0.0, Q.log_sum_pt - 6.605974) / 0.07073019   # +2.4%  log_sum_pt > 6.606
        + 0.02326699 * max(0.0, Q.sum_pt_top5 - 531.1875) * max(0.0, 0.01392641 - Q.zdr_6) / 1.245695   # +2.3%  sum_pt_top5 > 531.2 and zdr_6 < 0.01393
        - 0.0218373 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, 0.008654951 - Q.zdr_6) / 0.000799435   # -2.2%  log_sum_pt > 6.503 and zdr_6 < 0.008655
        - 0.02076785 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1) / 0.1196218   # -2.1%  log_sum_pt > 6.378 and z_dr_0p05_0p1 < 0.7509
        - 0.01942334 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # -1.9%  tau1 < 0.07284
        + 0.01689651 * max(0.0, 49.6681 - Q.mass) / 17.06893   # +1.7%  mass < 49.67
        - 0.01189102 * max(0.0, 0.0008722282 - Q.lam1) / 0.0001882079   # -1.2%  lam1 < 0.0008722
        + 0.01160846 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.008921136 - Q.mean_phi2) / 0.0001024498   # +1.2%  z_7 < 0.06165 and mean_phi2 < 0.008921
        + 0.00960024 * max(0.0, 0.008678045 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.02076709) / 7.418362e-06   # +1.0%  sum_z_dr2 < 0.008678 and centroid_offset > 0.02077
        - 0.007904057 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -0.8%  zdr_0 < 0.02118
        + 0.007687089 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.06297984 - Q.tau2) / 0.0007776805   # +0.8%  z_7 < 0.06165 and tau2 < 0.06298
        + 0.006839568 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +0.7%  sj3_dr_max > 0.169
        - 0.00677288 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # -0.7%  lam1 < 0.008376 and centroid_offset > 0.02077
        - 0.006462574 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 1.679198 - Q.D2) / 0.005760154   # -0.6%  z_7 < 0.06165 and D2 < 1.679
        + 0.006007487 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.1057739 - Q.abseta_0) / 0.001206885   # +0.6%  z_7 < 0.06165 and abseta_0 < 0.1058
        - 0.005576694 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236) / 1.769648e-05   # -0.6%  mass_over_sum_pt_sq < 0.005833 and centroid_offset > 0.008092
        - 0.004906575 * max(0.0, 25.57812 - Q.pt_7) / 1.327389   # -0.5%  pt_7 < 25.58
        - 0.004728212 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.00323438   # -0.5%  lam1 < 0.008376 and n_pt_above_50 > 5
        - 0.004251821 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.04875823) / 0.002321449   # -0.4%  z_7 < 0.06165 and z_dr_0p05_0p1 > 0.04876
        + 0.003660012 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # +0.4%  sum_z_dr < 0.0717
        + 0.003626997 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.sj2_dr - 0.1294903) / 0.1877842   # +0.4%  pt_7 > 34.53 and sj2_dr > 0.1295
        - 0.00360322 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1484197 - Q.planar_flow) / 0.0001369594   # -0.4%  lam1 < 0.008376 and planar_flow < 0.1484
        + 0.002877729 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 1.432482 - Q.D2) / 0.01970563   # +0.3%  log_sum_pt > 6.606 and D2 < 1.432
        + 0.002763668 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.centroid_offset - 0.009480685) / 0.0009010889   # +0.3%  log_sum_pt > 6.378 and centroid_offset > 0.009481
        - 0.002706745 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 52.90625 - Q.pt_6) / 13.55761   # -0.3%  pt_7 > 34.53 and pt_6 < 52.91
        - 0.002362314 * max(0.0, 0.1019409 - Q.sum_z_dr) * max(0.0, Q.pair_mass_0_6 - 4.037975) / 0.1185817   # -0.2%  sum_z_dr < 0.1019 and pair_mass_0_6 > 4.038
        - 0.002186869 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.5700307   # -0.2%  sj3_dr_max > 0.169 and sj3_pair_mass_min > 5.744
        + 0.002123395 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # +0.2%  pt_7 > 34.53 and tau2 < 0.01713
        - 0.002067784 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.2%  pt_7 > 53.44
        + 0.00126674 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 3.175597   # +0.1%  pt_7 > 34.53 and n_dr_0p1_0p2 > 1
        - 0.0006304985 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, 0.03885671 - Q.D2_b2) / 4.310709e-07   # -0.1%  mass_over_sum_pt_sq < 0.005833 and D2_b2 < 0.03886
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 9.659;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.659184 * (0.2523642
        - 0.1255171 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -12.6%  pt_7 < 53.44
        - 0.1050968 * max(0.0, Q.LHA - 0.1329373) / 0.1175814   # -10.5%  LHA > 0.1329
        + 0.1007062 * max(0.0, 0.1079857 - Q.mass_over_sum_pt) / 0.05207765   # +10.1%  mass_over_sum_pt < 0.108
        + 0.09438489 * max(0.0, 813.4156 - Q.sum_pt) / 129.0758   # +9.4%  sum_pt < 813.4
        + 0.06473542 * max(0.0, 62.55 - Q.sj3_pair_mass_max) / 30.55764   # +6.5%  sj3_pair_mass_max < 62.55
        - 0.05718489 * max(0.0, Q.z_7 - 0.03629544) / 0.01879829   # -5.7%  z_7 > 0.0363
        - 0.04829298 * max(0.0, Q.sum_pt - 527.1781) * max(0.0, 0.0001947983 - Q.lam2) / 0.02863163   # -4.8%  sum_pt > 527.2 and lam2 < 0.0001948
        - 0.03787745 * max(0.0, 36.22941 - Q.mass) / 10.11393   # -3.8%  mass < 36.23
        + 0.03504116 * max(0.0, Q.sum_pt - 527.1781) / 199.4719   # +3.5%  sum_pt > 527.2
        - 0.03301793 * max(0.0, 0.2507612 - Q.max_dr) / 0.1316542   # -3.3%  max_dr < 0.2508
        + 0.02532094 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # +2.5%  log_sum_pt < 6.606
        + 0.02303368 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +2.3%  lam2 < 0.0001948
        - 0.0213511 * max(0.0, 0.4926918 - Q.planar_flow) / 0.2728654   # -2.1%  planar_flow < 0.4927
        + 0.02036231 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # +2.0%  zdr_0 < 0.02118
        + 0.01577709 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # +1.6%  log_sum_pt < 6.464
        - 0.01548415 * max(0.0, 62.55 - Q.sj3_pair_mass_max) * max(0.0, 0.06810151 - Q.z_7) / 0.6094489   # -1.5%  sj3_pair_mass_max < 62.55 and z_7 < 0.0681
        - 0.01487142 * max(0.0, 62.55 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.01096064) / 0.2139385   # -1.5%  sj3_pair_mass_max < 62.55 and centroid_offset > 0.01096
        + 0.01388684 * max(0.0, Q.sum_pt - 868.5094) / 18.08463   # +1.4%  sum_pt > 868.5
        - 0.01215823 * max(0.0, Q.sum_pt_top5 - 752.1) / 21.27415   # -1.2%  sum_pt_top5 > 752.1
        + 0.01202583 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.pt_6 - 19.46875) / 0.05232739   # +1.2%  lam1 < 0.005954 and pt_6 > 19.47
        + 0.01104191 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +1.1%  pt_7 > 34.53
        - 0.01062968 * max(0.0, Q.z_7 - 0.04939969) / 0.01023002   # -1.1%  z_7 > 0.0494
        + 0.009882971 * max(0.0, Q.pt_7 - 43.5) / 1.4368   # +1.0%  pt_7 > 43.5
        + 0.009865352 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # +1.0%  sum_z_dr2 < 0.003563
        - 0.009451428 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.max_dr - 0.08050702) / 4.493327e-05   # -0.9%  lam1 < 0.005954 and max_dr > 0.08051
        - 0.00875089 * max(0.0, Q.z_7 - 0.04939969) * max(0.0, 0.0623951 - Q.sj3_dr_min) / 0.0002978985   # -0.9%  z_7 > 0.0494 and sj3_dr_min < 0.0624
        + 0.008052674 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # +0.8%  lam1 < 0.001504
        - 0.007609918 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 0.0174408 - Q.abseta_0) / 0.216283   # -0.8%  sum_pt_top5 > 752.1 and abseta_0 < 0.01744
        + 0.007269314 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 1.129616 - Q.D2_b2) / 6.295733   # +0.7%  sum_pt_top5 > 752.1 and D2_b2 < 1.13
        + 0.005981435 * max(0.0, 36.22941 - Q.mass) * max(0.0, 73.75 - Q.pt_5) / 259.9887   # +0.6%  mass < 36.23 and pt_5 < 73.75
        - 0.005034878 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # -0.5%  log_sum_pt > 6.843
        - 0.004812476 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # -0.5%  sum_z_dr < 0.007674
        + 0.003754483 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.0001947983 - Q.lam2) / 1.313144e-06   # +0.4%  log_sum_pt > 6.843 and lam2 < 0.0001948
        + 0.00374201 * max(0.0, 6.46415 - Q.log_sum_pt) * max(0.0, 1.129616 - Q.D2_b2) / 0.03658616   # +0.4%  log_sum_pt < 6.464 and D2_b2 < 1.13
        + 0.00329144 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.0174408 - Q.abseta_0) / 8.283244e-05   # +0.3%  log_sum_pt > 6.843 and abseta_0 < 0.01744
        - 0.003274071 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 1.345805 - Q.D2_b2) / 0.002918988   # -0.3%  log_sum_pt > 6.843 and D2_b2 < 1.346
        + 0.003092161 * max(0.0, Q.m012 - 40.2) / 1.307209   # +0.3%  m012 > 40.2
        - 0.002395725 * max(0.0, 0.03243272 - Q.z_7) * max(0.0, Q.mass_top3 - 16.89912) / 0.01064848   # -0.2%  z_7 < 0.03243 and mass_top3 > 16.9
        + 0.002332654 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.4474937 - Q.z_dr_0p05_0p1) / 0.00315435   # +0.2%  log_sum_pt > 6.843 and z_dr_0p05_0p1 < 0.4475
        + 0.002063778 * max(0.0, 0.007673833 - Q.sum_z_dr) * max(0.0, 75.625 - Q.pt_4) / 0.004511241   # +0.2%  sum_z_dr < 0.007674 and pt_4 < 75.62
        - 0.001031643 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.pt_6 - 41.21875) / 0.06090836   # -0.1%  log_sum_pt > 6.843 and pt_6 > 41.22
        + 0.0003023596 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.dr02 - 0.15647) / 2.387299e-05   # +0.0%  log_sum_pt > 6.843 and dr02 > 0.1565
        - 0.0002102752 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.dr02 - 0.2623104) / 3.697963e-06   # -0.0%  log_sum_pt > 6.843 and dr02 > 0.2623
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 34.16;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.1595 * (-0.084606
        - 0.1292757 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -12.9%  mass_over_sum_pt_sq < 0.01166
        - 0.1177123 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # -11.8%  sum_z_dr2 > 0.008678
        + 0.09925851 * max(0.0, Q.sum_z_dr - 0.04081947) / 0.0270611   # +9.9%  sum_z_dr > 0.04082
        + 0.06852874 * max(0.0, 0.00817466 - Q.mass_over_sum_pt_sq) / 0.004299658   # +6.9%  mass_over_sum_pt_sq < 0.008175
        - 0.04687747 * max(0.0, Q.sum_z_dr - 0.1019409) / 0.005400003   # -4.7%  sum_z_dr > 0.1019
        - 0.04418339 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # -4.4%  tau1 > 0.05357
        + 0.03900782 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # +3.9%  sj3_dr_max > 0.1986
        + 0.03359226 * max(0.0, Q.sj2_dr - 0.1872617) / 0.02732165   # +3.4%  sj2_dr > 0.1873
        + 0.02675847 * max(0.0, Q.mass_over_sum_pt - 0.0681391) / 0.01539157   # +2.7%  mass_over_sum_pt > 0.06814
        + 0.02634383 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +2.6%  tau1 > 0.1136
        - 0.02370923 * max(0.0, 0.9008535 - Q.z_dr_0_0p05) / 0.3811445   # -2.4%  z_dr_0_0p05 < 0.9009
        + 0.02360942 * max(0.0, Q.sum_z_dr - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494) / 0.008781747   # +2.4%  sum_z_dr > 0.04082 and log_sum_pt > 6.08
        + 0.02353226 * max(0.0, Q.sum_z_dr - 0.07608178) / 0.01059033   # +2.4%  sum_z_dr > 0.07608
        - 0.02089892 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # -2.1%  tau1 > 0.1028
        + 0.02055084 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +2.1%  lam1 > 0.008376
        + 0.02010497 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # +2.0%  sum_z_dr2_top5 < 0.00833
        - 0.01984213 * max(0.0, 0.004372139 - Q.sum_z_dr2) / 0.00156571   # -2.0%  sum_z_dr2 < 0.004372
        + 0.01901231 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +1.9%  e2 > 0.04447
        - 0.01489404 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.9641201 - Q.z_dr_0p05_0p1) / 0.01959435   # -1.5%  sj2_dr > 0.1873 and z_dr_0p05_0p1 < 0.9641
        - 0.01484576 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # -1.5%  sj2_dr > 0.2688
        + 0.01174001 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # +1.2%  lam1_plus_lam2 > 0.01324
        - 0.01172408 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) / 0.003638607   # -1.2%  sum_z_dr2_top5 < 0.007164
        + 0.01157408 * max(0.0, Q.sj2_dr - 0.2687922) * max(0.0, 0.9641201 - Q.z_dr_0p05_0p1) / 0.005973659   # +1.2%  sj2_dr > 0.2688 and z_dr_0p05_0p1 < 0.9641
        - 0.01039097 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.02673292   # -1.0%  max_dr > 0.1028 and z_dr_0p05_0p1 < 0.846
        + 0.01033992 * max(0.0, Q.LHA - 0.3467135) / 0.008898884   # +1.0%  LHA > 0.3467
        + 0.009054859 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # +0.9%  sum_z_dr2 > 0.01883
        + 0.008985982 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # +0.9%  lam1 > 0.012
        - 0.008552041 * max(0.0, Q.mass - 64.61873) / 3.274818   # -0.9%  mass > 64.62
        + 0.008190875 * max(0.0, Q.centroid_offset - 0.01096064) / 0.008813008   # +0.8%  centroid_offset > 0.01096
        + 0.006819381 * max(0.0, Q.sd_mass - 62.55) / 3.348271   # +0.7%  sd_mass > 62.55
        - 0.006477478 * max(0.0, Q.sj3_dr_max - 0.3012016) / 0.01214652   # -0.6%  sj3_dr_max > 0.3012
        - 0.006071098 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.1950293   # -0.6%  mass_over_sum_pt > 0.06814 and sj3_pair_mass_min > 5.744
        + 0.005705183 * max(0.0, Q.lam1_plus_lam2 - 0.007520088) / 0.002470136   # +0.6%  lam1_plus_lam2 > 0.00752
        + 0.005326857 * max(0.0, 1.0 - Q.n_dr_0_0p05) / 0.2859429   # +0.5%  n_dr_0_0p05 < 1
        + 0.004619336 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # +0.5%  sum_z_dr > 0.08724
        + 0.00359965 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6) / 0.005634141   # +0.4%  lam2 > 0.001131 and pt_6 < 56.53
        - 0.003372027 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113) / 0.39507   # -0.3%  sj2_dr > 0.1873 and sj2_mass1 > 2.25
        + 0.003333435 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 0.07861328 - Q.abseta_0) / 0.0002969212   # +0.3%  centroid_offset > 0.01096 and abseta_0 < 0.07861
        + 0.002530027 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, Q.z_6 - 0.05441452) / 6.527736e-06   # +0.3%  lam2 > 0.001131 and z_6 > 0.05441
        - 0.002492202 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 0.1626038 - Q.abseta_7) / 0.0008475577   # -0.2%  centroid_offset > 0.01096 and abseta_7 < 0.1626
        + 0.002482837 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # +0.2%  lam1 > 0.01643
        + 0.00225586 * max(0.0, 1.0 - Q.n_dr_0_0p05) * max(0.0, 4.0 - Q.n_dr_0p05_0p1) / 0.3078555   # +0.2%  n_dr_0_0p05 < 1 and n_dr_0p05_0p1 < 4
        + 0.002181742 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # +0.2%  lam1_plus_lam2 > 0.00559
        - 0.002017039 * max(0.0, Q.sd_mass - 38.43971) / 11.50403   # -0.2%  sd_mass > 38.44
        - 0.001828786 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.pt_6 - 31.90625) / 0.1358885   # -0.2%  mass_over_sum_pt > 0.06814 and pt_6 > 31.91
        + 0.001828202 * max(0.0, -0.004664942 - Q.mean_eta) / 0.003754527   # +0.2%  mean_eta < -0.004665
        - 0.001813721 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr) / 0.0001196243   # -0.2%  mass_over_sum_pt > 0.06814 and sj2_dr < 0.218
        - 0.001796912 * max(0.0, Q.sd_mass - 86.4) / 0.7553564   # -0.2%  sd_mass > 86.4
        + 0.001616964 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # +0.2%  sj3_pair_mass_min > 11.05
        + 0.001595235 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.06081235 - Q.z_6) / 0.0004655277   # +0.2%  max_dr > 0.1028 and z_6 < 0.06081
        + 0.001333093 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sj2_mass1 - 16.86126) / 0.02534836   # +0.1%  e2 > 0.06344 and sj2_mass1 > 16.86
        + 0.001241552 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # +0.1%  e2 > 0.06344
        - 0.001240828 * max(0.0, Q.sd_mass - 62.55) * max(0.0, 0.9206502 - Q.D2_b2) / 1.823781   # -0.1%  sd_mass > 62.55 and D2_b2 < 0.9207
        - 0.000789667 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.05268713 - Q.dr_3) / 0.000155932   # -0.1%  sj2_dr > 0.1873 and dr_3 < 0.05269
        - 0.0006096771 * max(0.0, Q.sum_z_dr - 0.08723651) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 5.945324e-05   # -0.1%  sum_z_dr > 0.08724 and z_dr_0p05_0p1 > 0.6748
        + 0.000551934 * max(0.0, Q.sum_z_dr - 0.07608178) * max(0.0, Q.eta_0 - 0.07952881) / 0.0001300296   # +0.1%  sum_z_dr > 0.07608 and eta_0 > 0.07953
        - 0.0005023756 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, Q.abseta_0 - 0.1057739) / 0.0002451453   # -0.1%  max_dr > 0.1028 and abseta_0 > 0.1058
        - 0.0004289968 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 27.57812 - Q.pt_6) / 0.028355   # -0.0%  sj2_dr > 0.1873 and pt_6 < 27.58
        - 0.0003208715 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, -0.0892334 - Q.eta_1) / 0.0002499278   # -0.0%  max_dr > 0.1028 and eta_1 < -0.08923
        + 0.0001258037 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 2.662284e-07   # +0.0%  sum_z_dr2 > 0.01883 and z_dr_0p05_0p1 > 0.7509
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 80.17;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 80.17292 * (-0.01701597
        + 0.2890531 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # +28.9%  sum_zz_dr2 < 0.01166
        - 0.227319 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -22.7%  mass_over_sum_pt_sq < 0.01166
        + 0.0490246 * max(0.0, 0.06729223 - Q.C2) / 0.04162086   # +4.9%  C2 < 0.06729
        - 0.04749145 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -4.7%  sum_z_dr2 < 0.008678
        - 0.04096973 * max(0.0, 0.233678 - Q.sj3_dr_max) / 0.09057682   # -4.1%  sj3_dr_max < 0.2337
        + 0.03087144 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +3.1%  N2 < 0.2233
        + 0.02357458 * max(0.0, Q.lam1_plus_lam2 - 0.003562611) / 0.004066872   # +2.4%  lam1_plus_lam2 > 0.003563
        + 0.02250804 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # +2.3%  sj3_dr_max < 0.3456
        - 0.02166319 * max(0.0, 0.05028464 - Q.e2) / 0.02502152   # -2.2%  e2 < 0.05028
        - 0.01670346 * max(0.0, 0.04990367 - Q.centroid_offset) / 0.03352573   # -1.7%  centroid_offset < 0.0499
        - 0.01630885 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -1.6%  lam2 < 0.001131
        - 0.01276552 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -1.3%  lam2 < 0.0005373
        + 0.01215926 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +1.2%  tau1 < 0.07284
        - 0.01204809 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266) / 0.01381411   # -1.2%  N2 < 0.2233 and eccentricity > 0.7117
        + 0.0118275 * max(0.0, 0.3467135 - Q.LHA) / 0.1128474   # +1.2%  LHA < 0.3467
        + 0.01151112 * max(0.0, 0.001653836 - Q.lam1_plus_lam2) / 0.0004289067   # +1.2%  lam1_plus_lam2 < 0.001654
        + 0.01119247 * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) / 0.004543329   # +1.1%  sum_z_dr2_top2 < 0.00764
        - 0.01075375 * max(0.0, Q.mass - 45.7571) / 9.387753   # -1.1%  mass > 45.76
        - 0.01028049 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # -1.0%  max_dr < 0.1773
        + 0.01007259 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) / 0.003638607   # +1.0%  sum_z_dr2_top5 < 0.007164
        - 0.009919084 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -1.0%  mass_over_sum_pt > 0.09041
        - 0.008773031 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.1410336 - Q.dr01) / 4.252003e-05   # -0.9%  lam2 < 0.0005373 and dr01 < 0.141
        - 0.008534558 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -0.9%  sum_pt < 739.5
        + 0.008120642 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +0.8%  sj3_dr_max < 0.2134
        + 0.006871226 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 27.42324 - Q.sj3_pair_mass_min) / 138.7479   # +0.7%  sd_mass > 44.82 and sj3_pair_mass_min < 27.42
        + 0.006472735 * max(0.0, Q.mass_top5 - 14.54404) / 16.7137   # +0.6%  mass_top5 > 14.54
        - 0.006274368 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -0.6%  mass_over_sum_pt < 0.07269
        - 0.005836952 * max(0.0, Q.sum_z_dr2_top5 - 0.001501708) / 0.004796547   # -0.6%  sum_z_dr2_top5 > 0.001502
        - 0.005827511 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -0.6%  lam1 < 0.001504
        + 0.005730382 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +0.6%  e3 < 0.0001869
        - 0.004579832 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7) / 0.8992052   # -0.5%  N2 < 0.2233 and pt_7 < 53.44
        - 0.004460436 * max(0.0, 0.03874536 - Q.M2) / 0.006229062   # -0.4%  M2 < 0.03875
        + 0.004038157 * max(0.0, Q.max_dr - 0.1598486) / 0.0215652   # +0.4%  max_dr > 0.1598
        - 0.003480965 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.55 - Q.mass) / 0.3959152   # -0.3%  N2 < 0.2233 and mass < 62.55
        - 0.002548957 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.centroid_offset - 0.001308549) / 0.1760217   # -0.3%  sd_mass > 44.82 and centroid_offset > 0.001309
        + 0.002518603 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # +0.3%  sum_z_dr2 < 0.002635
        - 0.002366634 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.07796252 - Q.dr1_6) / 1.697653e-05   # -0.2%  lam2 < 0.0005373 and dr1_6 < 0.07796
        + 0.001834227 * max(0.0, Q.mass - 76.6557) * max(0.0, 0.107953 - Q.M3) / 0.09817562   # +0.2%  mass > 76.66 and M3 < 0.108
        + 0.001716953 * max(0.0, Q.sd_mass - 44.82259) / 8.759065   # +0.2%  sd_mass > 44.82
        - 0.001636793 * max(0.0, Q.sd_mass - 74.57663) / 1.60503   # -0.2%  sd_mass > 74.58
        - 0.001573318 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1218872 - Q.abseta_7) / 0.003384012   # -0.2%  N2 < 0.2233 and abseta_7 < 0.1219
        - 0.001569761 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.275762 - Q.sd_zg) / 0.3498594   # -0.2%  sd_mass > 44.82 and sd_zg < 0.2758
        - 0.001423056 * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) * max(0.0, Q.C2_b2 - 0.0006435798) / 6.028305e-06   # -0.1%  sum_z_dr2_top2 < 0.00764 and C2_b2 > 0.0006436
        - 0.001220376 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 7.11886e-05   # -0.1%  N2 < 0.2233 and sum_zz_dr2 > 0.01166
        + 0.001024653 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +0.1%  max_dr < 0.1118
        - 0.000769586 * max(0.0, Q.lam1_plus_lam2 - 0.003562611) * max(0.0, 0.1057566 - Q.sj3_z3) / 5.535701e-05   # -0.1%  lam1_plus_lam2 > 0.003563 and sj3_z3 < 0.1058
        + 0.0007250666 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +0.1%  e2 > 0.04447
        - 0.0006047189 * max(0.0, Q.max_dr - 0.1598486) * max(0.0, 0.0006435798 - Q.C2_b2) / 1.059262e-06   # -0.1%  max_dr > 0.1598 and C2_b2 < 0.0006436
        - 0.0003984709 * max(0.0, 1.340118e-05 - Q.e3) / 5.081834e-06   # -0.0%  e3 < 1.34e-05
        + 0.0003625537 * max(0.0, 0.233678 - Q.sj3_dr_max) * max(0.0, Q.ptdr0_5 - 7.737156) / 0.01098259   # +0.0%  sj3_dr_max < 0.2337 and ptdr0_5 > 7.737
        - 0.000305268 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.0%  mass > 76.66
        + 0.0002840647 * max(0.0, 0.2233283 - Q.N2) * max(0.0, -0.009352575 - Q.mean_phi) / 0.0001205535   # +0.0%  N2 < 0.2233 and mean_phi < -0.009353
        - 9.877669e-05 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 24.42188 - Q.pt_6) / 0.01371831   # -0.0%  N2 < 0.2233 and pt_6 < 24.42
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.33;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.32694 * (-0.03858538
        + 0.1281668 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +12.8%  z_7 < 0.07149
        + 0.0803743 * max(0.0, 0.001653836 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 7.0272e-06   # +8.0%  sum_z_dr2 < 0.001654 and centroid_offset < 0.02355
        - 0.07411794 * max(0.0, 37.15625 - Q.pt_7) / 5.803748   # -7.4%  pt_7 < 37.16
        - 0.05866774 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.001503553 - Q.lam1) / 4.027805e-05   # -5.9%  LHA < 0.2161 and lam1 < 0.001504
        - 0.05405777 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -5.4%  zdr_0 < 0.02118
        + 0.05332903 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.001184249   # +5.3%  lam1_plus_lam2 < 0.003563
        + 0.04376181 * max(0.0, 0.04939969 - Q.z_7) / 0.007457505   # +4.4%  z_7 < 0.0494
        + 0.04184729 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +4.2%  LHA < 0.2161
        + 0.03616494 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 46.125 - Q.pt_6) / 2161.354   # +3.6%  sum_pt_top5 > 430.8 and pt_6 < 46.12
        - 0.03500337 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0004149607   # -3.5%  z_7 < 0.07149 and centroid_offset < 0.03117
        + 0.03383366 * max(0.0, 0.005284669 - Q.sum_zz_dr2) / 0.00223735   # +3.4%  sum_zz_dr2 < 0.005285
        - 0.02976106 * max(0.0, Q.sum_pt_top5 - 430.75) / 176.7518   # -3.0%  sum_pt_top5 > 430.8
        - 0.02887158 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -2.9%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.02578697 * max(0.0, 0.001653836 - Q.sum_z_dr2) / 0.0004289067   # +2.6%  sum_z_dr2 < 0.001654
        - 0.02369769 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -2.4%  log_sum_pt > 6.701
        - 0.02359543 * max(0.0, 0.08051087 - Q.z_6) / 0.02253808   # -2.4%  z_6 < 0.08051
        - 0.02312853 * max(0.0, 0.002074109 - Q.sum_zz_dr2) / 0.000670556   # -2.3%  sum_zz_dr2 < 0.002074
        + 0.01970023 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +2.0%  sum_pt_top5 > 687.4
        + 0.01767913 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 40.2 - Q.mass_top3) / 0.60086   # +1.8%  z_7 < 0.07149 and mass_top3 < 40.2
        - 0.0168608 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # -1.7%  log_sum_pt > 6.843
        + 0.01540959 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01341502 - Q.dr_2) / 0.0001347035   # +1.5%  log_sum_pt > 6.701 and dr_2 < 0.01342
        + 0.01326909 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.phi_0 - -0.0297699) / 0.0002791968   # +1.3%  zdr_0 < 0.02118 and phi_0 > -0.02977
        + 0.01280868 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +1.3%  z_7 < 0.02807
        + 0.009831458 * max(0.0, 37.15625 - Q.pt_7) * max(0.0, Q.pt_5 - 24.57812) / 73.42646   # +1.0%  pt_7 < 37.16 and pt_5 > 24.58
        + 0.009046922 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.sj2_dr - 0.1682655) / 4.310225   # +0.9%  sum_pt_top5 > 430.8 and sj2_dr > 0.1683
        + 0.008813674 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.002127561 - Q.mean_phi2) / 2.516806e-05   # +0.9%  z_7 < 0.07149 and mean_phi2 < 0.002128
        + 0.00830487 * max(0.0, 0.006789738 - Q.centroid_offset) / 0.001012351   # +0.8%  centroid_offset < 0.00679
        - 0.007425101 * max(0.0, Q.pair_mass_0_4 - 23.26771) / 0.6641733   # -0.7%  pair_mass_0_4 > 23.27
        + 0.007321169 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 9.030369e-05 - Q.mean_eta2) / 9.99897e-07   # +0.7%  log_sum_pt > 6.701 and mean_eta2 < 9.03e-05
        - 0.006350477 * max(0.0, 0.006789738 - Q.centroid_offset) * max(0.0, 56.4375 - Q.pt_5) / 0.01266727   # -0.6%  centroid_offset < 0.00679 and pt_5 < 56.44
        - 0.005739552 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 0.009661512 - Q.dr_2) / 1.13159e-05   # -0.6%  z_7 < 0.0494 and dr_2 < 0.009662
        + 0.005359442 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, Q.mean_phi - -0.02594505) / 0.0001045731   # +0.5%  log_sum_pt > 6.896 and mean_phi > -0.02595
        + 0.005326856 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.centroid_offset - 0.009480685) / 0.7981241   # +0.5%  sum_pt_top5 > 430.8 and centroid_offset > 0.009481
        + 0.004807342 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # +0.5%  sum_z_dr < 0.007674
        - 0.004685133 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # -0.5%  sum_z_dr2_top3 < 0.002152
        - 0.004574997 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.5%  log_sum_pt > 6.896
        - 0.004181052 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.02270492 - Q.dr_2) / 4.344748e-05   # -0.4%  log_sum_pt > 6.896 and dr_2 < 0.0227
        - 0.003924472 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.mean_phi - 0.002834884) / 4.214377e-05   # -0.4%  log_sum_pt > 6.701 and mean_phi > 0.002835
        - 0.003756535 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 501.625 - Q.sum_pt_top2) / 0.2299658   # -0.4%  z_7 < 0.0494 and sum_pt_top2 < 501.6
        + 0.003547763 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # +0.4%  pt_5 < 24.58
        + 0.002934314 * max(0.0, Q.pair_mass_0_4 - 23.26771) * max(0.0, Q.eccentricity - 0.9704496) / 0.009376301   # +0.3%  pair_mass_0_4 > 23.27 and eccentricity > 0.9704
        + 0.002840549 * max(0.0, Q.pair_mass_0_4 - 23.26771) * max(0.0, 0.01423545 - Q.mean_eta2) / 0.004884555   # +0.3%  pair_mass_0_4 > 23.27 and mean_eta2 < 0.01424
        - 0.001334834 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 0.001534584   # -0.1%  LHA < 0.2161 and n_dr_0p2_0p4 > 0
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 31.65;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.64886 * (0.02015633
        - 0.1634122 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -16.3%  sum_z_dr2 < 0.008678
        + 0.1115721 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +11.2%  lam1 < 0.012
        - 0.05292585 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # -5.3%  max_dr < 0.1452
        - 0.04943968 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -4.9%  sj3_dr_max < 0.3012
        + 0.04716966 * max(0.0, 0.1789613 - Q.sj3_dr_max) / 0.05394779   # +4.7%  sj3_dr_max < 0.179
        + 0.04454333 * max(0.0, 0.02415398 - Q.C2_b2) / 0.02132459   # +4.5%  C2_b2 < 0.02415
        - 0.0415732 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # -4.2%  lam1_plus_lam2 < 0.01324
        - 0.03938096 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -3.9%  lam2 < 0.001131
        + 0.03861937 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +3.9%  centroid_offset > 0.008092
        - 0.03741236 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 1.341509   # -3.7%  mass < 60.63 and z_dr_0p2_0p4 < 0.05644
        + 0.02728062 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +2.7%  lam2 < 0.003408
        + 0.02508051 * max(0.0, 0.1136369 - Q.tau1) * max(0.0, 0.01426135 - Q.mean_phi2) / 0.0007578925   # +2.5%  tau1 < 0.1136 and mean_phi2 < 0.01426
        - 0.02386655 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 0.2534037 - Q.planar_flow) / 0.0006566196   # -2.4%  lam1 < 0.012 and planar_flow < 0.2534
        - 0.02375808 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -2.4%  pt_6 < 39.75
        + 0.02262371 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +2.3%  pt_6 < 41.22
        + 0.01997319 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +2.0%  tau1 < 0.1136
        + 0.01843508 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt) / 1.016213   # +1.8%  pt_6 < 41.22 and log_sum_pt < 6.767
        - 0.01778802 * max(0.0, Q.centroid_offset - 0.01837778) / 0.005523694   # -1.8%  centroid_offset > 0.01838
        - 0.01680644 * max(0.0, Q.sj3_dr_min - 0.02400746) / 0.02802924   # -1.7%  sj3_dr_min > 0.02401
        + 0.01638365 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # +1.6%  sum_zz_dr2 < 0.003013
        - 0.01520799 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757) / 0.07098489   # -1.5%  pt_6 < 41.22 and z_7 > 0.02321
        + 0.01417845 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +1.4%  sj3_dr_max > 0.1879
        + 0.01183092 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +1.2%  lam1 < 0.00733
        + 0.009987206 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # +1.0%  sj2_dr < 0.1873
        + 0.009700481 * max(0.0, 21.78408 - Q.mass) / 4.431023   # +1.0%  mass < 21.78
        - 0.008457809 * max(0.0, Q.sj3_pair_mass_min - 4.501727) / 3.480268   # -0.8%  sj3_pair_mass_min > 4.502
        + 0.00714242 * max(0.0, 45.595 - Q.mass) / 14.73532   # +0.7%  mass < 45.59
        - 0.006212427 * max(0.0, 6.638339 - Q.log_sum_pt) / 0.1524936   # -0.6%  log_sum_pt < 6.638
        + 0.005657609 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2) / 2.179362e-05   # +0.6%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        + 0.00539023 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002658741   # +0.5%  lam2 < 0.003408 and n_dr_0_0p05 < 3
        + 0.005253539 * max(0.0, 60.63098 - Q.mass) / 24.47895   # +0.5%  mass < 60.63
        + 0.005078466 * max(0.0, 0.1872617 - Q.sj2_dr) * max(0.0, Q.eccentricity - 0.927072) / 0.0009514438   # +0.5%  sj2_dr < 0.1873 and eccentricity > 0.9271
        + 0.004745392 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.psi_0p1 - 0.4008925) / 0.003485257   # +0.5%  centroid_offset > 0.008092 and psi_0p1 > 0.4009
        - 0.004339202 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.002776626 - Q.mean_phi2) / 0.01000982   # -0.4%  sum_pt > 988.4 and mean_phi2 < 0.002777
        - 0.004285714 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.4%  z_6 < 0.0216
        + 0.004274806 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +0.4%  sj3_dr_min > 0.1278
        + 0.003698674 * max(0.0, 19.46875 - Q.pt_6) / 0.2515892   # +0.4%  pt_6 < 19.47
        + 0.003489053 * max(0.0, 6.638339 - Q.log_sum_pt) * max(0.0, 0.0753896 - Q.z_7) / 0.001441873   # +0.3%  log_sum_pt < 6.638 and z_7 < 0.07539
        + 0.003374206 * max(0.0, 615.875 - Q.sum_pt) / 30.07766   # +0.3%  sum_pt < 615.9
        + 0.00323234 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.3%  sum_pt > 988.4
        + 0.00280942 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308) / 0.05438016   # +0.3%  centroid_offset > 0.008092 and sj3_pair_mass_min > 6.811
        - 0.002637547 * max(0.0, Q.tau2 - 0.008780509) / 0.00811667   # -0.3%  tau2 > 0.008781
        + 0.002536279 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.002127561 - Q.mean_phi2) / 0.007417827   # +0.3%  sum_pt > 988.4 and mean_phi2 < 0.002128
        + 0.002530228 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 43.23531   # +0.3%  sum_pt < 615.9 and n_dr_0p2_0p4 < 2
        - 0.002525658 * max(0.0, Q.sj3_dr13 - 0.181053) / 0.02525513   # -0.3%  sj3_dr13 > 0.1811
        + 0.002065321 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 0.1343804 - Q.dr1_7) / 0.3834372   # +0.2%  pt_6 < 41.22 and dr1_7 < 0.1344
        + 0.001771483 * max(0.0, 6.267538 - Q.log_sum_pt) / 0.02292097   # +0.2%  log_sum_pt < 6.268
        - 0.001585961 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, Q.sj3_dr12 - 0.1692253) / 0.0003039084   # -0.2%  centroid_offset > 0.01838 and sj3_dr12 > 0.1692
        - 0.001051055 * max(0.0, Q.sj3_dr23 - 0.2207152) / 0.01962555   # -0.1%  sj3_dr23 > 0.2207
        - 0.0009245011 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.00433128 - Q.mean_phi2) / 7.021694e-06   # -0.1%  centroid_offset > 0.01838 and mean_phi2 < 0.004331
        - 0.0009184026 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.1%  sum_pt_top5 > 840
        - 0.0008239026 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # -0.1%  mean_eta > 0.02644
        - 0.0008007844 * max(0.0, 0.1872617 - Q.sj2_dr) * max(0.0, Q.dr1_6 - 0.07796252) / 0.0003473731   # -0.1%  sj2_dr < 0.1873 and dr1_6 > 0.07796
        - 0.0006791526 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 2.69383   # -0.1%  sj3_pair_mass_min > 4.502 and n_dr_0p2_0p4 > 1
        - 0.0005702992 * max(0.0, Q.sj2_mass1 - 40.2) * max(0.0, 0.4490772 - Q.tau21_b2) / 0.01473   # -0.1%  sj2_mass1 > 40.2 and tau21_b2 < 0.4491
        + 0.000490669 * max(0.0, 29.90625 - Q.pt_6) * max(0.0, 8.0 - Q.n_pt_above_10) / 0.5895401   # +0.0%  pt_6 < 29.91 and n_pt_above_10 < 8
        + 0.0004887444 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 24.57812 - Q.pt_5) / 1.825775   # +0.0%  sum_pt < 615.9 and pt_5 < 24.58
        + 0.0004708088 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, Q.pt_4 - 65.1875) / 0.007630915   # +0.0%  centroid_offset > 0.01838 and pt_4 > 65.19
        - 0.0003600322 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, Q.M3 - 0.06688759) / 9.396674e-06   # -0.0%  mean_eta > 0.02644 and M3 > 0.06689
        + 0.0003286457 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.mean_eta - 0.02644207) / 1.54475e-06   # +0.0%  lam1 < 0.012 and mean_eta > 0.02644
        - 4.954834e-05 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, Q.mean_phi2 - 0.008921136) / 0.000300354   # -0.0%  sum_pt > 988.4 and mean_phi2 > 0.008921
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 35.63;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.63317 * (0.269818
        - 0.08444237 * max(0.0, Q.mass_over_sum_pt - 0.01109984) / 0.05101851   # -8.4%  mass_over_sum_pt > 0.0111
        - 0.07320296 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -7.3%  sum_z_dr < 0.08724
        - 0.06853198 * max(0.0, Q.mass_over_sum_pt - 0.1079857) / 0.005364315   # -6.9%  mass_over_sum_pt > 0.108
        - 0.06046275 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) / 0.00293624   # -6.0%  lam1_plus_lam2 < 0.006679
        + 0.05893035 * max(0.0, Q.sum_z_dr2 - 0.01323868) / 0.001442684   # +5.9%  sum_z_dr2 > 0.01324
        - 0.05395699 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -5.4%  mass_over_sum_pt > 0.09041
        + 0.04591278 * max(0.0, Q.sj3_dr_max - 0.1426152) / 0.06279029   # +4.6%  sj3_dr_max > 0.1426
        - 0.03231143 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -3.2%  sj2_dr < 0.1873
        + 0.03013457 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +3.0%  sj2_dr < 0.1592
        - 0.02886906 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # -2.9%  lam1_plus_lam2 > 0.00559
        - 0.02589661 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -2.6%  lam1 < 0.008376
        + 0.02566495 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +2.6%  mass_over_sum_pt > 0.08475
        - 0.02462894 * max(0.0, Q.sum_z_dr2 - 0.001653836) / 0.005220304   # -2.5%  sum_z_dr2 > 0.001654
        + 0.01931391 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +1.9%  mass_over_sum_pt > 0.07992
        - 0.01883716 * max(0.0, Q.sj3_dr_max - 0.04889979) / 0.1256943   # -1.9%  sj3_dr_max > 0.0489
        - 0.01868396 * max(0.0, Q.sum_z_dr2 - 0.004372139) / 0.003638805   # -1.9%  sum_z_dr2 > 0.004372
        + 0.01759659 * max(0.0, Q.mass - 36.22941) / 14.20528   # +1.8%  mass > 36.23
        - 0.01747805 * max(0.0, Q.lam1_plus_lam2 - 0.008678045) * max(0.0, 38.25 - Q.pt_6) / 0.006941575   # -1.7%  lam1_plus_lam2 > 0.008678 and pt_6 < 38.25
        + 0.01726104 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # +1.7%  max_dr < 0.1773
        + 0.01723856 * max(0.0, Q.mass_over_sum_pt - 0.07269073) / 0.01344138   # +1.7%  mass_over_sum_pt > 0.07269
        + 0.0164181 * max(0.0, 0.08723651 - Q.sum_z_dr) * max(0.0, 0.004032342 - Q.C2_b2) / 0.0001217265   # +1.6%  sum_z_dr < 0.08724 and C2_b2 < 0.004032
        - 0.01593859 * max(0.0, Q.lam1_plus_lam2 - 0.008678045) / 0.002214537   # -1.6%  lam1_plus_lam2 > 0.008678
        - 0.01349312 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # -1.3%  max_dr < 0.1118
        + 0.01243063 * max(0.0, Q.sum_z_dr2 - 0.01323868) * max(0.0, 38.25 - Q.pt_6) / 0.004519922   # +1.2%  sum_z_dr2 > 0.01324 and pt_6 < 38.25
        - 0.01222948 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.004032342 - Q.C2_b2) / 2.736263e-05   # -1.2%  centroid_offset < 0.02077 and C2_b2 < 0.004032
        + 0.01217966 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +1.2%  LHA < 0.3033
        + 0.009705306 * max(0.0, Q.z_7 - 0.03243272) / 0.02180051   # +1.0%  z_7 > 0.03243
        - 0.009647064 * max(0.0, Q.sj3_dr_max - 0.2623172) / 0.01879979   # -1.0%  sj3_dr_max > 0.2623
        + 0.009109868 * max(0.0, 0.0003061234 - Q.lam2) * max(0.0, 0.04019753 - Q.tau21_b2) / 2.611351e-06   # +0.9%  lam2 < 0.0003061 and tau21_b2 < 0.0402
        + 0.008705748 * max(0.0, Q.mass_over_sum_pt - 0.09041383) * max(0.0, 36.8125 - Q.pt_6) / 0.01975833   # +0.9%  mass_over_sum_pt > 0.09041 and pt_6 < 36.81
        - 0.008237412 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # -0.8%  lam2 < 0.0003061
        - 0.007941748 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -0.8%  sum_z_dr2 > 0.00752
        - 0.007899193 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.8%  mass > 76.66
        + 0.007762632 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +0.8%  e2 > 0.05028
        - 0.007460775 * max(0.0, 0.0009641429 - Q.sum_z_dr2) / 0.0002052288   # -0.7%  sum_z_dr2 < 0.0009641
        + 0.007441083 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +0.7%  tau1 < 0.05357
        + 0.006833942 * max(0.0, Q.sum_z_dr2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +0.7%  sum_z_dr2 > 0.004372 and planar_flow < 0.195
        - 0.00666464 * max(0.0, 0.04019753 - Q.tau21_b2) / 0.01109641   # -0.7%  tau21_b2 < 0.0402
        + 0.006314314 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) / 0.09532154   # +0.6%  z_dr_0p1_0p2 < 0.1586
        - 0.005879754 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -0.6%  centroid_offset < 0.02077
        + 0.005762016 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # +0.6%  sd_rg > 0.2788
        - 0.005477695 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.5%  centroid_offset > 0.03776
        - 0.005464867 * max(0.0, 2.357246 - Q.D2) / 1.097329   # -0.5%  D2 < 2.357
        + 0.004279465 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) / 0.0003002514   # +0.4%  sum_z_dr2_top2 < 0.001057
        + 0.003745231 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # +0.4%  sj2_dr < 0.1295
        + 0.003736204 * max(0.0, 0.001101266 - Q.sum_zz_dr2) / 0.000308004   # +0.4%  sum_zz_dr2 < 0.001101
        + 0.003725225 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.02656143 - Q.tau21_b2) / 4.805781e-05   # +0.4%  centroid_offset < 0.02077 and tau21_b2 < 0.02656
        + 0.003686365 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sd_mass - 38.43971) / 1.25962   # +0.4%  planar_flow < 0.195 and sd_mass > 38.44
        + 0.003612856 * max(0.0, Q.mass_top5 - 53.60766) / 1.978814   # +0.4%  mass_top5 > 53.61
        - 0.00322523 * max(0.0, 0.1950135 - Q.planar_flow) / 0.07573236   # -0.3%  planar_flow < 0.195
        - 0.003197341 * max(0.0, Q.sum_z_dr2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9458207) / 2.125219e-05   # -0.3%  sum_z_dr2 > 0.01324 and eccentricity > 0.9458
        + 0.002878323 * max(0.0, Q.centroid_offset - 0.03117077) / 0.002505019   # +0.3%  centroid_offset > 0.03117
        + 0.002877342 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, Q.sum_pt_top3 - 331.25) / 1.710819   # +0.3%  centroid_offset < 0.02077 and sum_pt_top3 > 331.2
        - 0.002800764 * max(0.0, 29.04219 - Q.pt_7) / 2.179984   # -0.3%  pt_7 < 29.04
        - 0.002670228 * max(0.0, 0.04081947 - Q.sum_z_dr) / 0.009176315   # -0.3%  sum_z_dr < 0.04082
        - 0.002008354 * max(0.0, Q.sd_rg - 0.2037854) / 0.01566501   # -0.2%  sd_rg > 0.2038
        - 0.00200253 * max(0.0, 0.05744392 - Q.D2_b2) / 0.005938983   # -0.2%  D2_b2 < 0.05744
        - 0.00195201 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -0.2%  e2 > 0.03556
        - 0.001509597 * max(0.0, Q.mass - 76.6557) * max(0.0, Q.zdr_6 - 0.006366792) / 0.006641425   # -0.2%  mass > 76.66 and zdr_6 > 0.006367
        - 0.001193507 * max(0.0, Q.mass_over_sum_pt - 0.01109984) * max(0.0, 35.28125 - Q.pt_6) / 0.1065511   # -0.1%  mass_over_sum_pt > 0.0111 and pt_6 < 35.28
        - 0.0008351606 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6) / 0.1690249   # -0.1%  planar_flow < 0.195 and pt_6 < 35.28
        - 0.0007852938 * max(0.0, Q.centroid_offset - 0.03117077) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.00188116   # -0.1%  centroid_offset > 0.03117 and n_pt_above_50 > 4
        - 0.0005733952 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) * max(0.0, 0.02656143 - Q.tau21_b2) / 2.239525e-07   # -0.1%  sum_z_dr2_top2 < 0.001057 and tau21_b2 < 0.02656
        - 0.0003529259 * max(0.0, Q.sj3_dr_max - 0.1426152) * max(0.0, 19.46875 - Q.pt_6) / 0.01292439   # -0.0%  sj3_dr_max > 0.1426 and pt_6 < 19.47
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 22.31;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.31089 * (-0.02518236
        + 0.0774112 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 4.808955e-05   # +7.7%  tau1 < 0.05357 and lam1_plus_lam2 < 0.003563
        - 0.07504298 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.003562611 - Q.sum_z_dr2) / 5.329563e-05   # -7.5%  sum_z_dr < 0.06109 and sum_z_dr2 < 0.003563
        - 0.06452181 * max(0.0, 0.06108601 - Q.sum_z_dr) / 0.01868685   # -6.5%  sum_z_dr < 0.06109
        + 0.06405839 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.001184249   # +6.4%  lam1_plus_lam2 < 0.003563
        + 0.06140758 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0001901112   # +6.1%  sum_z_dr2 < 0.00502 and z_dr_0p2_0p4 < 0.101
        + 0.05440243 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.0003703114   # +5.4%  sj3_dr_max < 0.1986 and sum_z_dr2 < 0.006679
        + 0.04728025 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +4.7%  sum_z_dr2 < 0.00502
        - 0.04131355 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.1231549   # -4.1%  mass < 29.64 and centroid_offset < 0.02686
        - 0.03747471 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # -3.7%  LHA < 0.1967
        + 0.03689935 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +3.7%  sum_z_dr2 < 0.006679 and centroid_offset < 0.02355
        - 0.03532886 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.0001781324   # -3.5%  sj3_dr_max < 0.1986 and lam1_plus_lam2 < 0.003563
        + 0.03336575 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.0001272072   # +3.3%  mass_over_sum_pt < 0.03319 and centroid_offset < 0.02686
        - 0.03063782 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -3.1%  sj3_dr_max < 0.1426
        - 0.02987328 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # -3.0%  tau1 < 0.05357
        - 0.02961224 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.005691733 - Q.sum_z_dr2_top5) / 0.0003116052   # -3.0%  sj3_dr_max < 0.1986 and sum_z_dr2_top5 < 0.005692
        - 0.02030071 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -2.0%  log_sum_pt > 6.701
        + 0.01967129 * max(0.0, 0.002635418 - Q.lam1_plus_lam2) / 0.0007941347   # +2.0%  lam1_plus_lam2 < 0.002635
        - 0.01851836 * max(0.0, 49.6681 - Q.mass) / 17.06893   # -1.9%  mass < 49.67
        - 0.01727174 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.001011075   # -1.7%  sum_z_dr < 0.06109 and z_dr_0p2_0p4 < 0.05644
        - 0.01632478 * max(0.0, 0.01289969 - Q.e2) * max(0.0, Q.psi_0p2 - 0.79448) / 0.0005184882   # -1.6%  e2 < 0.0129 and psi_0p2 > 0.7945
        + 0.01424902 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.0001947983 - Q.lam2) / 2.869243e-06   # +1.4%  sum_z_dr < 0.06109 and lam2 < 0.0001948
        - 0.01213106 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 43.5 - Q.pt_7) / 0.02255472   # -1.2%  sum_z_dr2 < 0.00502 and pt_7 < 43.5
        + 0.01191651 * max(0.0, 0.0001330621 - Q.lam2) * max(0.0, 4.721224 - Q.D2_b2) / 0.0002340602   # +1.2%  lam2 < 0.0001331 and D2_b2 < 4.721
        - 0.01142544 * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.00293624   # -1.1%  sum_z_dr2 < 0.006679
        - 0.01093201 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.01627885) / 0.0001130126   # -1.1%  sj3_dr_max < 0.1426 and centroid_offset > 0.01628
        + 0.009766743 * max(0.0, 21.78408 - Q.mass) * max(0.0, 48.71875 - Q.pt_7) / 69.12746   # +1.0%  mass < 21.78 and pt_7 < 48.72
        + 0.009691866 * max(0.0, 0.01289969 - Q.e2) / 0.002527207   # +1.0%  e2 < 0.0129
        + 0.00953063 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, Q.phi_0 - -0.04013062) / 0.0001189753   # +1.0%  sum_z_dr2 < 0.006679 and phi_0 > -0.04013
        + 0.008952641 * max(0.0, 29.6447 - Q.mass) * max(0.0, 1.762929e-06 - Q.e3) / 9.656157e-06   # +0.9%  mass < 29.64 and e3 < 1.763e-06
        + 0.00894101 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 48.71875 - Q.pt_7) / 0.776931   # +0.9%  log_sum_pt > 6.701 and pt_7 < 48.72
        - 0.008836198 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -0.9%  sum_z_dr2 < 0.00502 and centroid_offset > 0.00679
        - 0.008618527 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.0001947983 - Q.lam2) / 0.0007178522   # -0.9%  mass < 21.78 and lam2 < 0.0001948
        - 0.008501951 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 4.721224 - Q.D2_b2) / 0.008595177   # -0.9%  sum_z_dr2 < 0.006679 and D2_b2 < 4.721
        + 0.008100633 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # +0.8%  sj3_dr_max < 0.107
        - 0.008064296 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) / 0.00722828   # -0.8%  mass_over_sum_pt < 0.03319
        + 0.006981425 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.pt_7 - 15.55391) / 0.4153091   # +0.7%  LHA < 0.1967 and pt_7 > 15.55
        + 0.006740958 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.0001947983 - Q.lam2) / 9.822054e-06   # +0.7%  sj3_dr_max < 0.1986 and lam2 < 0.0001948
        - 0.005507611 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, Q.centroid_offset - 0.006789738) / 8.239263e-05   # -0.6%  sum_z_dr < 0.06109 and centroid_offset > 0.00679
        + 0.004360904 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, Q.lam1_plus_lam2 - 0.0003193707) / 1.048473e-05   # +0.4%  sum_z_dr < 0.06109 and lam1_plus_lam2 > 0.0003194
        + 0.003475693 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.3%  sum_pt_top5 > 687.4
        - 0.003154617 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 8.0 - Q.n_pt_above_10) / 7.731645   # -0.3%  sum_pt_top5 > 687.4 and n_pt_above_10 < 8
        + 0.002916038 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0005611231 - Q.sum_z_dr2) / 8.626944e-06   # +0.3%  log_sum_pt > 6.701 and sum_z_dr2 < 0.0005611
        + 0.001531931 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.centroid_offset - 0.01837778) / 3.05705e-06   # +0.2%  LHA < 0.1967 and centroid_offset > 0.01838
        - 0.001156969 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.1%  sum_pt > 988.4
        - 0.00092676 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.2097233   # -0.1%  mass < 29.64 and D2_b2 < 0.7166
        - 0.0008992771 * max(0.0, 0.007078158 - Q.e2) * max(0.0, Q.sum_z_dr2_top5 - 5.672047e-05) / 1.487483e-07   # -0.1%  e2 < 0.007078 and sum_z_dr2_top5 > 5.672e-05
        - 0.0007500841 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 0.01544159   # -0.1%  log_sum_pt > 6.701 and n_dr_0p05_0p1 > 2
        + 0.0006889507 * max(0.0, 0.07658656 - Q.LHA) / 0.0004806283   # +0.1%  LHA < 0.07659
        + 0.0005331696 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.05062541 - Q.dr12) / 0.1693771   # +0.1%  sum_pt > 988.4 and dr12 < 0.05063
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 35.36;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.36309 * (-0.1021888
        + 0.09521437 * max(0.0, 0.00609665 - Q.sum_z_dr2) / 0.002544039   # +9.5%  sum_z_dr2 < 0.006097
        + 0.07477839 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # +7.5%  sum_z_dr2 < 0.003563
        - 0.0564742 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -5.6%  mass_over_sum_pt < 0.07269
        - 0.05231086 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -5.2%  lam1 < 0.005954
        - 0.05141321 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -5.1%  sj3_dr_max < 0.1426
        + 0.04907384 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +4.9%  sum_pt < 988.4
        + 0.04875474 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +4.9%  sj3_dr_max < 0.2134
        + 0.04272948 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +4.3%  centroid_offset < 0.01838
        - 0.04140486 * max(0.0, 53.33237 - Q.mass) / 19.36004   # -4.1%  mass < 53.33
        + 0.03947304 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.2973626   # +3.9%  mass < 53.33 and centroid_offset < 0.02686
        + 0.02999948 * max(0.0, Q.lam1 - 0.005433361) / 0.00267714   # +3.0%  lam1 > 0.005433
        + 0.02694785 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # +2.7%  tau1 < 0.0437
        - 0.02536924 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # -2.5%  sum_zz_dr2 < 0.003013
        + 0.02354712 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +2.4%  sum_z_dr2 < 0.00502
        + 0.02347283 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +2.3%  mass_over_sum_pt_sq < 0.007183
        - 0.0210937 * max(0.0, Q.lam1 - 0.003377388) / 0.003672352   # -2.1%  lam1 > 0.003377
        + 0.02097682 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +2.1%  e2 > 0.04447
        + 0.01995654 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt) / 5.175436   # +2.0%  mass < 53.33 and log_sum_pt < 6.843
        - 0.01930911 * max(0.0, 29.6447 - Q.mass) / 7.34723   # -1.9%  mass < 29.64
        + 0.01921557 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # +1.9%  lam2 < 0.0003061
        + 0.01918017 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +1.9%  max_dr < 0.1118
        - 0.01759951 * max(0.0, 2.371297e-05 - Q.e3) / 1.105328e-05   # -1.8%  e3 < 2.371e-05
        - 0.01522761 * max(0.0, Q.sum_z_dr - 0.0717028) / 0.01201981   # -1.5%  sum_z_dr > 0.0717
        - 0.01240883 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -1.2%  e2 > 0.03556
        - 0.01201767 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266) / 0.0005312449   # -1.2%  lam1 > 0.005433 and eccentricity > 0.7117
        - 0.01103722 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.008712838   # -1.1%  centroid_offset < 0.01838 and n_for_90pct > 5
        - 0.01050281 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1) / 0.1931178   # -1.1%  centroid_offset < 0.01838 and pt_1 < 159.2
        - 0.007395542 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, Q.z_6 - 0.03448406) / 0.0003071415   # -0.7%  sum_z_dr < 0.05465 and z_6 > 0.03448
        - 0.007266183 * max(0.0, Q.mass - 45.595) / 9.461082   # -0.7%  mass > 45.59
        - 0.006811 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # -0.7%  e3 > 8.148e-05
        - 0.006603992 * max(0.0, 0.0284695 - Q.C3) / 0.008714909   # -0.7%  C3 < 0.02847
        + 0.005763424 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd) / 0.0002178654   # +0.6%  centroid_offset < 0.01838 and z_2nd < 0.2056
        + 0.005426547 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.5%  lam2 > 0.001131
        - 0.004994797 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) / 6.087676   # -0.5%  sj3_pair_mass_max < 24.23
        + 0.004866449 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.0002201438   # +0.5%  sum_z_dr2 < 0.006097 and planar_flow < 0.3221
        - 0.004787858 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, 0.002834884 - Q.mean_phi) / 1.344572e-05   # -0.5%  sum_z_dr2 < 0.006097 and mean_phi < 0.002835
        + 0.004299029 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.6875) / 0.3914557   # +0.4%  sj3_dr_max < 0.1426 and pt_6 > 33.69
        - 0.004271992 * max(0.0, Q.D2 - 2.055451) / 0.3061148   # -0.4%  D2 > 2.055
        - 0.00379532 * max(0.0, 0.006292091 - Q.zdr_0) / 0.001006321   # -0.4%  zdr_0 < 0.006292
        - 0.003518259 * max(0.0, 7.0 - Q.n_for_90pct) / 0.5603294   # -0.4%  n_for_90pct < 7
        + 0.003482259 * max(0.0, 0.0001721983 - Q.lam1_plus_lam2) / 1.441896e-05   # +0.3%  lam1_plus_lam2 < 0.0001722
        + 0.003445439 * max(0.0, Q.N2 - 0.2233283) / 0.04872624   # +0.3%  N2 > 0.2233
        - 0.003344009 * max(0.0, 0.02227783 - Q.absphi_1) / 0.006894148   # -0.3%  absphi_1 < 0.02228
        + 0.003109336 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # +0.3%  sum_z_dr2 > 0.01883
        + 0.002583745 * max(0.0, 0.06503035 - Q.z_5) / 0.007566857   # +0.3%  z_5 < 0.06503
        + 0.002505534 * max(0.0, 0.0782171 - Q.M3) / 0.01197321   # +0.3%  M3 < 0.07822
        + 0.002406423 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.2%  n_dr_0p2_0p4 > 1
        - 0.002361877 * max(0.0, 0.05464922 - Q.sum_z_dr) / 0.01532978   # -0.2%  sum_z_dr < 0.05465
        - 0.002197884 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.tau4 - 0.001224244) / 1.099241e-05   # -0.2%  centroid_offset < 0.01838 and tau4 > 0.001224
        - 0.002097481 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255) / 0.000277001   # -0.2%  tau1 < 0.0437 and eccentricity > 0.9031
        - 0.001923817 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.2%  sum_pt_top5 > 840
        - 0.001820606 * Q.z_dr_0p2_0p4 / 0.02920532   # -0.2%  z_dr_0p2_0p4
        + 0.001706105 * max(0.0, 53.33237 - Q.mass) * max(0.0, Q.D2 - 2.843757) / 5.555775   # +0.2%  mass < 53.33 and D2 > 2.844
        + 0.00162695 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.mean_eta - 6.288824e-05) / 1.181456e-05   # +0.2%  centroid_offset < 0.01838 and mean_eta > 6.289e-05
        + 0.001569916 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, 0.02656143 - Q.tau21_b2) / 1.23798e-05   # +0.2%  sum_z_dr < 0.05465 and tau21_b2 < 0.02656
        + 0.00148919 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) * max(0.0, -0.004917145 - Q.phi_0) / 9.23841e-05   # +0.1%  mass_over_sum_pt < 0.07269 and phi_0 < -0.004917
        - 0.001457862 * max(0.0, 988.4078 - Q.sum_pt) * max(0.0, 0.01778111 - Q.dr_2) / 0.3401129   # -0.1%  sum_pt < 988.4 and dr_2 < 0.01778
        - 0.001380778 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.01108383   # -0.1%  n_dr_0p2_0p4 > 1 and sj3_dr_min < 0.209
        - 0.001228767 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, 0.02656143 - Q.tau21_b2) / 8.029519e-06   # -0.1%  tau1 < 0.0437 and tau21_b2 < 0.02656
        + 0.001140098 * max(0.0, 559.6875 - Q.sum_pt) / 16.18216   # +0.1%  sum_pt < 559.7
        + 0.001124255 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +0.1%  e3 > 0.0001869
        - 0.0009656174 * max(0.0, Q.sd_mass - 45.595) * max(0.0, Q.n_dr_0p05_0p1 - 7.0) / 0.4890223   # -0.1%  sd_mass > 45.59 and n_dr_0p05_0p1 > 7
        - 0.0009323106 * max(0.0, 0.002316125 - Q.centroid_offset) / 9.872932e-05   # -0.1%  centroid_offset < 0.002316
        + 0.0008338719 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, -0.01275329 - Q.mean_phi) / 1.158446e-05   # +0.1%  tau1 < 0.0437 and mean_phi < -0.01275
        + 0.0006454338 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.1%  log_sum_pt > 6.896
        + 0.0005427306 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.001255404 - Q.zdr_5) / 2.213187e-06   # +0.1%  log_sum_pt > 6.896 and zdr_5 < 0.001255
        - 0.0005317204 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2) / 0.000627491   # -0.1%  log_sum_pt > 6.896 and D2_b2 < 0.7166
        + 0.0005295553 * max(0.0, 31.125 - Q.pt_4) / 0.3140792   # +0.1%  pt_4 < 31.12
        + 0.0004579165 * max(0.0, Q.e3 - 0.0001869378) * max(0.0, Q.pt_2 - 56.5) / 0.0005941291   # +0.0%  e3 > 0.0001869 and pt_2 > 56.5
        - 0.0004099818 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, Q.mean_phi - 0.02612796) / 3.162575e-07   # -0.0%  sum_z_dr2 < 0.006097 and mean_phi > 0.02613
        + 0.0003466647 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.mean_phi - 0.02612796) / 2.994783e-06   # +0.0%  tau1 < 0.0437 and mean_phi > 0.02613
        - 0.000263758 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.zdr_2 - 0.00759156) / 4.975701e-07   # -0.0%  tau1 < 0.0437 and zdr_2 > 0.007592
        - 0.0002486222 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, Q.abseta_1 - 0.03625488) / 8.135786e-06   # -0.0%  sj3_dr_max < 0.107 and abseta_1 > 0.03625
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 8.427;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.427009 * (0.2228385
        + 0.1285319 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # +12.9%  sum_z_dr2 > 0.00752
        + 0.1156749 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +11.6%  mass < 76.66
        - 0.1027901 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -10.3%  lam1 < 0.005954
        - 0.06151046 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -6.2%  LHA > 0.3033
        + 0.05874522 * max(0.0, Q.lam2 - 0.0003061234) / 0.0004216599   # +5.9%  lam2 > 0.0003061
        - 0.04865375 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -4.9%  lam1 < 0.004184
        + 0.0445192 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # +4.5%  tau1 > 0.05357
        + 0.03958402 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # +4.0%  sum_z_dr2_top3 < 0.002152
        - 0.03870449 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # -3.9%  lam1 > 0.00733
        + 0.03627596 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1) / 0.002082998   # +3.6%  zdr_0 < 0.02118 and z_dr_0p05_0p1 < 0.2919
        - 0.03330553 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -3.3%  lam1 < 0.001504
        - 0.02413016 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.planar_flow - 0.04505724) / 0.0002791469   # -2.4%  lam2 > 0.0003061 and planar_flow > 0.04506
        - 0.02385683 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # -2.4%  sj3_dr_max > 0.1986
        - 0.02289384 * max(0.0, 45.75 - Q.pt_7) / 12.14632   # -2.3%  pt_7 < 45.75
        - 0.01870718 * Q.e2 / 0.02863215   # -1.9%  e2
        + 0.01794797 * max(0.0, Q.sj3_pair_mass_min - 15.95929) / 1.348548   # +1.8%  sj3_pair_mass_min > 15.96
        - 0.01696993 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -1.7%  e3 > 3.892e-05
        + 0.0167979 * max(0.0, Q.mass_top5 - 22.18342) / 12.67741   # +1.7%  mass_top5 > 22.18
        - 0.01476956 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -1.5%  zdr_0 < 0.02118
        + 0.01384796 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.002471 - Q.D2) / 1.812028   # +1.4%  pt_7 < 45.75 and D2 < 1.002
        - 0.0132555 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 6.572938 - Q.log_sum_pt) / 1.257183   # -1.3%  pt_7 < 45.75 and log_sum_pt < 6.573
        - 0.01318521 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # -1.3%  sum_z_dr2 < 0.002635
        - 0.01091506 * max(0.0, 0.00325401 - Q.zdr_7) / 0.001049641   # -1.1%  zdr_7 < 0.003254
        - 0.009445246 * max(0.0, Q.mass_top5 - 45.32077) / 3.61051   # -0.9%  mass_top5 > 45.32
        + 0.008823277 * max(0.0, Q.C2_b2 - 0.009032972) / 0.001703043   # +0.9%  C2_b2 > 0.009033
        - 0.00870857 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2) / 0.0002929492   # -0.9%  lam1 > 0.00733 and D2_b2 < 0.3809
        + 0.008273579 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +0.8%  sj3_dr_min > 0.1278
        - 0.00713812 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.3658817 - Q.z_dr_0_0p05) / 0.002463327   # -0.7%  sj3_dr_min > 0.1278 and z_dr_0_0p05 < 0.3659
        - 0.006932898 * max(0.0, 0.004918231 - Q.zdr_0) / 0.0006323791   # -0.7%  zdr_0 < 0.004918
        + 0.006925699 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.01258764) / 3.400017e-05   # +0.7%  zdr_0 < 0.02118 and centroid_offset > 0.01259
        - 0.006154006 * max(0.0, 0.02563286 - Q.M2) / 0.001872375   # -0.6%  M2 < 0.02563
        - 0.005731538 * max(0.0, Q.sj3_pair_mass_min - 11.051) * max(0.0, Q.sj3_pairmin_over_m - 0.28737) / 0.2416857   # -0.6%  sj3_pair_mass_min > 11.05 and sj3_pairmin_over_m > 0.2874
        + 0.004873014 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, 1.002471 - Q.D2) / 0.009558568   # +0.5%  tau1 > 0.05357 and D2 < 1.002
        - 0.003730601 * max(0.0, Q.sum_z_dr2 - 0.02530566) / 0.0002851418   # -0.4%  sum_z_dr2 > 0.02531
        - 0.003035948 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.3%  sum_pt > 988.4
        + 0.002668595 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +0.3%  sj2_dr > 0.3004
        - 0.001986408 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, Q.D2_b2 - 1.129616) / 0.0002590844   # -0.2%  lam1 > 0.00733 and D2_b2 > 1.13
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 60.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 60.23528 * (-0.04869096
        - 0.2515101 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -25.2%  sum_zz_dr2 < 0.008169
        + 0.2089885 * max(0.0, 0.00817466 - Q.mass_over_sum_pt_sq) / 0.004299658   # +20.9%  mass_over_sum_pt_sq < 0.008175
        + 0.09851811 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +9.9%  lam1_plus_lam2 < 0.008678
        + 0.04774947 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +4.8%  sum_z_dr2 < 0.01324
        + 0.03030189 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +3.0%  centroid_offset < 0.03776
        + 0.02904771 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +2.9%  z_7 > 0.01686
        + 0.02690663 * max(0.0, 0.2623172 - Q.sj3_dr_max) / 0.1129006   # +2.7%  sj3_dr_max < 0.2623
        - 0.02575311 * max(0.0, 0.169029 - Q.sj3_dr_max) / 0.04876366   # -2.6%  sj3_dr_max < 0.169
        + 0.02429378 * max(0.0, 0.006679471 - Q.lam1_plus_lam2) / 0.00293624   # +2.4%  lam1_plus_lam2 < 0.006679
        - 0.02354784 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -2.4%  mass < 69.61
        - 0.02303353 * max(0.0, 0.07637363 - Q.mass_over_sum_pt) / 0.02715027   # -2.3%  mass_over_sum_pt < 0.07637
        - 0.01928334 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -1.9%  lam1 < 0.008376
        - 0.0177894 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # -1.8%  sum_z_dr < 0.0717
        + 0.01287972 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +1.3%  planar_flow < 0.2534
        - 0.01247738 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt) / 3.513015   # -1.2%  centroid_offset < 0.03776 and sum_pt < 901.6
        - 0.01165371 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -1.2%  lam1 < 0.00733
        - 0.01082343 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.tau3 - 0.001996306) / 6.194014e-06   # -1.1%  centroid_offset > 0.0499 and tau3 > 0.001996
        - 0.009199865 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # -0.9%  tau1 < 0.09539
        - 0.008963626 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.9%  centroid_offset > 0.0499
        + 0.007183623 * max(0.0, 49.6681 - Q.mass) / 17.06893   # +0.7%  mass < 49.67
        - 0.00691742 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.zdr_6 - 0.007390416) / 3.11545e-06   # -0.7%  centroid_offset > 0.0499 and zdr_6 > 0.00739
        + 0.006879648 * max(0.0, 15.45403 - Q.mass) / 2.385786   # +0.7%  mass < 15.45
        - 0.006869368 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 15.95929 - Q.sj3_pair_mass_min) / 0.05098851   # -0.7%  centroid_offset < 0.01438 and sj3_pair_mass_min < 15.96
        - 0.006787184 * max(0.0, Q.pt_7 - 29.04219) / 7.785987   # -0.7%  pt_7 > 29.04
        - 0.006581293 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # -0.7%  LHA < 0.3127
        + 0.006234487 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +0.6%  lam2 < 0.001131
        + 0.005699342 * max(0.0, 1.050302e-05 - Q.e3) / 3.717236e-06   # +0.6%  e3 < 1.05e-05
        + 0.005435286 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # +0.5%  max_dr < 0.1452
        - 0.004501308 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # -0.5%  C2 < 0.03579
        + 0.003872857 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # +0.4%  sj2_dr > 0.1779
        + 0.003510495 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.5502779 - Q.tau32) / 0.0001096908   # +0.4%  centroid_offset > 0.0499 and tau32 < 0.5503
        - 0.00345792 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.3%  planar_flow < 0.2534 and sum_pt < 840
        + 0.003020095 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +0.3%  sj3_dr_max < 0.1426
        - 0.002601489 * max(0.0, 506.875 - Q.sum_pt_top5) / 34.75344   # -0.3%  sum_pt_top5 < 506.9
        + 0.002339512 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0007565864   # +0.2%  centroid_offset < 0.01438 and D2_b2 < 0.5327
        - 0.002206457 * max(0.0, 0.02689598 - Q.sum_z_dr) / 0.004330137   # -0.2%  sum_z_dr < 0.0269
        - 0.002142183 * max(0.0, 0.1027585 - Q.max_dr) / 0.02411847   # -0.2%  max_dr < 0.1028
        - 0.002097206 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001130645 - Q.lam2) / 1.843299e-05   # -0.2%  sj2_dr > 0.1779 and lam2 < 0.001131
        - 0.001941615 * max(0.0, Q.n_dr_0p1_0p2 - 3.0) / 0.3318622   # -0.2%  n_dr_0p1_0p2 > 3
        - 0.001884512 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 0.0345459 - Q.absphi_0) / 0.0003803849   # -0.2%  centroid_offset < 0.03776 and absphi_0 < 0.03455
        - 0.001841345 * max(0.0, 0.02054282 - Q.sum_z_dr) / 0.002592388   # -0.2%  sum_z_dr < 0.02054
        + 0.001711719 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +0.2%  pt_6 < 41.22
        - 0.001608538 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 4.721224 - Q.D2_b2) / 16.05793   # -0.2%  pt_6 < 41.22 and D2_b2 < 4.721
        - 0.001573862 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 1.050302e-05 - Q.e3) / 1.771164e-07   # -0.2%  planar_flow < 0.2534 and e3 < 1.05e-05
        - 0.001263621 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 37.15625 - Q.pt_7) / 0.5537514   # -0.1%  planar_flow < 0.2534 and pt_7 < 37.16
        - 0.001040681 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.03601074 - Q.abseta_4) / 8.179649e-05   # -0.1%  centroid_offset < 0.01438 and abseta_4 < 0.03601
        - 0.0009758833 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, Q.sj3_pair_mass_min - 1.282345) / 0.4025762   # -0.1%  planar_flow < 0.2534 and sj3_pair_mass_min > 1.282
        - 0.0009364109 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, -0.001813533 - Q.mean_phi) / 4.860641e-05   # -0.1%  centroid_offset < 0.03776 and mean_phi < -0.001814
        + 0.0008024066 * max(0.0, 506.875 - Q.sum_pt_top5) * max(0.0, Q.z_dr_0p1_0p2 - 0.04510668) / 8.800637   # +0.1%  sum_pt_top5 < 506.9 and z_dr_0p1_0p2 > 0.04511
        - 0.0007463978 * max(0.0, 24.42188 - Q.pt_6) / 0.6046572   # -0.1%  pt_6 < 24.42
        + 0.0005706879 * max(0.0, 0.2623172 - Q.sj3_dr_max) * max(0.0, -0.0216713 - Q.phi_0) / 0.0003847124   # +0.1%  sj3_dr_max < 0.2623 and phi_0 < -0.02167
        - 0.0005554195 * max(0.0, 0.02689598 - Q.sum_z_dr) * max(0.0, Q.sj3_pair_mass_min - 1.590484) / 0.005084766   # -0.1%  sum_z_dr < 0.0269 and sj3_pair_mass_min > 1.59
        + 0.0004547426 * max(0.0, 0.004372139 - Q.sum_z_dr2) * max(0.0, Q.sj3_dr13 - 0.1659434) / 5.846229e-06   # +0.0%  sum_z_dr2 < 0.004372 and sj3_dr13 > 0.1659
        + 0.0004415024 * max(0.0, Q.z_dr_0p05_0p1 - 0.8460335) / 0.01076887   # +0.0%  z_dr_0p05_0p1 > 0.846
        + 0.0003600327 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.0283055   # +0.0%  sj2_dr > 0.1779 and n_pt_above_50 > 4
        - 0.0002073338 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.0%  sum_pt > 988.4
        + 2.494391e-05 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # +0.0%  centroid_offset > 0.0499 and D2 < 2.844
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.4299;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.429902 * (-1.414256
        + 0.4612386 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # +46.1%  sum_z_dr2 > 0.01883
        - 0.1336419 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -13.4%  e2 > 0.06344
        - 0.08537055 * max(0.0, Q.sum_z_dr2_top2 - 0.01403324) / 0.001129575   # -8.5%  sum_z_dr2_top2 > 0.01403
        - 0.07611501 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, 53.4375 - Q.pt_7) / 0.01439677   # -7.6%  sum_z_dr2 > 0.01883 and pt_7 < 53.44
        + 0.07408381 * max(0.0, Q.mass - 91.19) / 0.5839191   # +7.4%  mass > 91.19
        + 0.05604361 * max(0.0, Q.mass - 69.61135) * max(0.0, Q.centroid_offset - 0.009480685) / 0.04081507   # +5.6%  mass > 69.61 and centroid_offset > 0.009481
        + 0.04190066 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +4.2%  centroid_offset > 0.0499
        - 0.03846589 * max(0.0, Q.zdr_0 - 0.03981924) / 0.0005695785   # -3.8%  zdr_0 > 0.03982
        + 0.03314002 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, Q.lam2 - 0.000537286) / 2.828867e-06   # +3.3%  sum_z_dr2 > 0.01883 and lam2 > 0.0005373
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 18.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.06993 * (0.04673154
        + 0.2964793 * max(0.0, 0.1484084 - Q.sum_z_dr) / 0.0905999   # +29.6%  sum_z_dr < 0.1484
        - 0.1508109 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -15.1%  e2 < 0.08001
        + 0.1231303 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +12.3%  lam1 < 0.01643
        - 0.08221854 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -8.2%  sum_z_dr < 0.1484 and log_sum_pt < 6.804
        - 0.05421181 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 8.324263e-07   # -5.4%  e3 < 5.335e-05 and centroid_offset < 0.03776
        - 0.034458 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -3.4%  sum_z_dr < 0.1484 and pt_7 < 38.53
        + 0.03412949 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7) / 781.5808   # +3.4%  sum_pt_top5 > 658.1 and pt_7 < 40.04
        + 0.03333167 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # +3.3%  e3 < 5.335e-05
        - 0.022389 * max(0.0, 0.007520088 - Q.lam1_plus_lam2) / 0.003544991   # -2.2%  lam1_plus_lam2 < 0.00752
        + 0.01925673 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +1.9%  lam1 < 0.006507
        - 0.01921953 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.000537286 - Q.lam2) / 4.021081e-05   # -1.9%  sum_z_dr < 0.1484 and lam2 < 0.0005373
        - 0.01698521 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55592   # -1.7%  sum_pt_top5 > 658.1
        + 0.0159546 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +1.6%  lam1_plus_lam2 < 0.01324
        - 0.01127825 * max(0.0, Q.mass - 49.6681) / 7.721594   # -1.1%  mass > 49.67
        - 0.01120379 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # -1.1%  z_7 < 0.02807
        - 0.008631074 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7) / 0.06765006   # -0.9%  pt_6 < 31.91 and z_7 < 0.05861
        + 0.008044009 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +0.8%  sum_pt_top5 > 791.1
        + 0.007550963 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.z_7 - 0.06164517) / 0.0003465887   # +0.8%  sum_z_dr < 0.1484 and z_7 > 0.06165
        + 0.006940097 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 31.90625 - Q.pt_6) / 134.9952   # +0.7%  sum_pt_top5 > 791.1 and pt_6 < 31.91
        + 0.006639171 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +0.7%  log_sum_pt > 6.503
        + 0.006219009 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.D3 - 0.2213841) / 3.794981e-05   # +0.6%  e3 < 5.335e-05 and D3 > 0.2214
        + 0.0046225 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.07474969 - Q.M3) / 0.0006773733   # +0.5%  sum_z_dr < 0.1484 and M3 < 0.07475
        + 0.003993157 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.tau2 - 0.008780509) / 0.000269945   # +0.4%  sum_z_dr < 0.1484 and tau2 > 0.008781
        - 0.003985393 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # -0.4%  z_5 < 0.02818
        - 0.003516381 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.zdr_7 - 0.0007928864) / 0.0001818422   # -0.4%  log_sum_pt > 6.503 and zdr_7 > 0.0007929
        - 0.003367461 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6) / 104.494   # -0.3%  sum_pt > 988.4 and pt_6 < 62.25
        - 0.002032756 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.2%  z_6 < 0.0216
        - 0.001922773 * max(0.0, Q.pt_7 - 48.71875) / 0.6759439   # -0.2%  pt_7 > 48.72
        - 0.001841518 * max(0.0, Q.sj3_dr23 - 0.1974628) / 0.02478939   # -0.2%  sj3_dr23 > 0.1975
        - 0.00181157 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # -0.2%  sum_z_dr < 0.007674
        + 0.001548576 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.pair_mass_0_7 - 10.2219) / 0.2907061   # +0.2%  log_sum_pt > 6.503 and pair_mass_0_7 > 10.22
        - 0.001202071 * max(0.0, 31.90625 - Q.pt_6) / 1.826854   # -0.1%  pt_6 < 31.91
        - 0.001074402 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.sj2_mass1 - 31.78116) / 0.01215768   # -0.1%  sum_z_dr < 0.1484 and sj2_mass1 > 31.78
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 47.31;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 47.31154 * (-0.002305325
        - 0.06632191 * max(0.0, Q.lam1_plus_lam2 - 0.006679471) / 0.002702003   # -6.6%  lam1_plus_lam2 > 0.006679
        + 0.06601584 * max(0.0, Q.sum_z_dr - 0.02689598) / 0.03613842   # +6.6%  sum_z_dr > 0.0269
        - 0.0544524 * max(0.0, Q.e2 - 0.01655442) / 0.01596508   # -5.4%  e2 > 0.01655
        - 0.05377297 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -5.4%  sum_z_dr2 > 0.00752
        - 0.04845384 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # -4.8%  sum_z_dr2 > 0.008678
        - 0.04482217 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # -4.5%  lam1_plus_lam2 > 0.01324
        - 0.04432132 * max(0.0, 0.324646 - Q.sd_rg) / 0.2082097   # -4.4%  sd_rg < 0.3246
        + 0.04201814 * max(0.0, Q.lam1_plus_lam2 - 0.003562611) / 0.004066872   # +4.2%  lam1_plus_lam2 > 0.003563
        + 0.03787019 * max(0.0, Q.sum_z_dr - 0.04081947) / 0.0270611   # +3.8%  sum_z_dr > 0.04082
        + 0.03451514 * max(0.0, 49.91626 - Q.sd_mass) / 22.60916   # +3.5%  sd_mass < 49.92
        + 0.03343982 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # +3.3%  sj2_dr > 0.1592
        + 0.03257651 * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 0.001433269   # +3.3%  sum_zz_dr2 > 0.01166
        - 0.02927473 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # -2.9%  sum_z_dr > 0.08724
        + 0.02920321 * max(0.0, Q.sum_zz_dr2 - 0.002074109) / 0.004484017   # +2.9%  sum_zz_dr2 > 0.002074
        + 0.02622607 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +2.6%  mass_over_sum_pt > 0.08475
        - 0.02395661 * max(0.0, Q.sum_z_dr2 - 0.0009641429) / 0.00568632   # -2.4%  sum_z_dr2 > 0.0009641
        + 0.02348514 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # +2.3%  mass_over_sum_pt > 0.09041
        - 0.02241052 * max(0.0, Q.sum_zz_dr2 - 0.0030133) / 0.003940214   # -2.2%  sum_zz_dr2 > 0.003013
        - 0.02015225 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -2.0%  e3 < 8.148e-05
        - 0.0195881 * max(0.0, 74.57663 - Q.sd_mass) / 42.04056   # -2.0%  sd_mass < 74.58
        + 0.01902563 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +1.9%  mass_over_sum_pt > 0.07992
        + 0.01723595 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +1.7%  e2 > 0.05028
        - 0.01475134 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -1.5%  centroid_offset > 0.0499
        + 0.01398767 * max(0.0, Q.z_dr_0_0p05 - 0.1515405) / 0.4480651   # +1.4%  z_dr_0_0p05 > 0.1515
        - 0.01165586 * max(0.0, Q.sj2_dr - 0.06154135) / 0.1002688   # -1.2%  sj2_dr > 0.06154
        - 0.01147582 * max(0.0, Q.sum_z_dr - 0.1019409) / 0.005400003   # -1.1%  sum_z_dr > 0.1019
        + 0.011127 * max(0.0, Q.sum_z_dr - 0.08065885) / 0.009330634   # +1.1%  sum_z_dr > 0.08066
        - 0.009327768 * max(0.0, 0.004032342 - Q.C2_b2) / 0.002883542   # -0.9%  C2_b2 < 0.004032
        - 0.008708466 * max(0.0, 715.4688 - Q.sum_pt) / 69.91196   # -0.9%  sum_pt < 715.5
        + 0.00765809 * max(0.0, 5.0 - Q.n_dr_0_0p05) / 1.94322   # +0.8%  n_dr_0_0p05 < 5
        - 0.00753031 * max(0.0, Q.sj2_dr - 0.2001708) / 0.02317349   # -0.8%  sj2_dr > 0.2002
        + 0.00739042 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # +0.7%  lam2 < 0.0005373
        - 0.007353667 * max(0.0, Q.sd_rg - 0.2330919) / 0.01070351   # -0.7%  sd_rg > 0.2331
        - 0.006903914 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # -0.7%  e3 < 5.335e-05
        + 0.006863959 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # +0.7%  lam1_plus_lam2 > 0.00559
        - 0.00551377 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # -0.6%  sj2_dr > 0.1295
        - 0.005450286 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -0.5%  sj2_dr > 0.1779
        + 0.005141709 * max(0.0, 0.1115136 - Q.planar_flow) / 0.03343187   # +0.5%  planar_flow < 0.1115
        + 0.005020997 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +0.5%  e2 > 0.04111
        - 0.004854952 * max(0.0, Q.centroid_offset - 0.02685622) / 0.003254849   # -0.5%  centroid_offset > 0.02686
        - 0.004581299 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.5%  psi_0p1 > 0.9761
        - 0.00445467 * max(0.0, 0.163898 - Q.z_dr_0p05_0p1) / 0.08682106   # -0.4%  z_dr_0p05_0p1 < 0.1639
        + 0.00431309 * max(0.0, 0.07870506 - Q.z_dr_0p1_0p2) / 0.04317253   # +0.4%  z_dr_0p1_0p2 < 0.07871
        - 0.004270478 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -0.4%  planar_flow < 0.1115 and lam1 < 0.00733
        + 0.004168886 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # +0.4%  planar_flow < 0.1115 and lam1 < 0.01643
        + 0.003783264 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0006014625   # +0.4%  sj2_dr > 0.1592 and D2_b2 < 0.08499
        + 0.003248999 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +0.3%  LHA > 0.3033
        + 0.00320209 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.0008333816 - Q.C2_b2) / 1.341084e-07   # +0.3%  centroid_offset > 0.0499 and C2_b2 < 0.0008334
        - 0.003149942 * max(0.0, Q.psi_0p1 - 0.7143804) / 0.1805043   # -0.3%  psi_0p1 > 0.7144
        - 0.002374608 * max(0.0, 0.08366273 - Q.planar_flow) / 0.02146305   # -0.2%  planar_flow < 0.08366
        - 0.002367943 * max(0.0, Q.sj2_dr - 0.1294903) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0009393137   # -0.2%  sj2_dr > 0.1295 and D2_b2 < 0.08499
        + 0.002197307 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, 150.625 - Q.pt_1) / 0.1837865   # +0.2%  centroid_offset > 0.02686 and pt_1 < 150.6
        + 0.002175218 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 3.351526e-05   # +0.2%  planar_flow < 0.1115 and lam1_plus_lam2 < 0.006097
        + 0.001962843 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1629004) / 4.043869e-07   # +0.2%  e3 < 5.335e-05 and sj3_dr23 > 0.1629
        - 0.001729973 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.2%  planar_flow < 0.1115 and centroid_offset < 0.01838
        - 0.001659029 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.2%  centroid_offset > 0.03776
        - 0.001559554 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0003170016   # -0.2%  sj2_dr > 0.2002 and D2_b2 < 0.08499
        - 0.001490285 * max(0.0, Q.mass - 69.61135) / 2.406702   # -0.1%  mass > 69.61
        + 0.001390651 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 715.4688 - Q.sum_pt) / 22.02426   # +0.1%  z_dr_0p05_0p1 < 0.5883 and sum_pt < 715.5
        - 0.001193585 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.1058993 - Q.z_5) / 0.0007616364   # -0.1%  sj2_dr > 0.2002 and z_5 < 0.1059
        - 0.001026833 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 658.125 - Q.sum_pt_top5) / 2.950014   # -0.1%  planar_flow < 0.1115 and sum_pt_top5 < 658.1
        - 0.0007318791 * max(0.0, Q.mass_top5 - 49.18618) / 2.747087   # -0.1%  mass_top5 > 49.19
        + 0.0007069766 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.05056028 - Q.dr_2) / 0.0001260892   # +0.1%  sj2_dr > 0.2002 and dr_2 < 0.05056
        - 0.00069783 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.03293672 - Q.dr_2) / 6.256053e-05   # -0.1%  sj2_dr > 0.1592 and dr_2 < 0.03294
        - 0.0006586906 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, Q.mean_eta - -0.006779839) / 0.000320267   # -0.1%  planar_flow < 0.1115 and mean_eta > -0.00678
        + 0.0004713545 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067297   # +0.0%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        - 0.0004603191 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 0.02415398 - Q.C2_b2) / 0.0005249308   # -0.0%  z_dr_0p05_0p1 > 0.7509 and C2_b2 < 0.02415
        - 0.0001179521 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, Q.zdr_5 - 0.0155217) / 1.782544e-06   # -0.0%  z_dr_0p05_0p1 > 0.7509 and zdr_5 > 0.01552
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 38.98;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 38.9784 * (-0.01378412
        + 0.1591619 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +15.9%  lam1_plus_lam2 < 0.01324
        - 0.11296 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -11.3%  sum_z_dr < 0.1019
        - 0.09430428 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -9.4%  sum_zz_dr2 < 0.008169
        + 0.08906027 * max(0.0, 0.01882765 - Q.lam1_plus_lam2) / 0.01314296   # +8.9%  lam1_plus_lam2 < 0.01883
        + 0.08751333 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +8.8%  e2 < 0.04111
        - 0.06346242 * max(0.0, 0.007520088 - Q.sum_z_dr2) / 0.00354499   # -6.3%  sum_z_dr2 < 0.00752
        - 0.06071858 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # -6.1%  mass_over_sum_pt < 0.1309
        + 0.06016433 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +6.0%  mass_over_sum_pt_sq < 0.007183
        - 0.05793963 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # -5.8%  sum_zz_dr2 < 0.01166
        - 0.05686908 * max(0.0, 0.2001708 - Q.sj2_dr) / 0.07331403   # -5.7%  sj2_dr < 0.2002
        + 0.05568047 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +5.6%  sj2_dr < 0.1592
        - 0.03847015 * max(0.0, 0.0005124533 - Q.sum_z_dr2_top2) / 0.0001104529   # -3.8%  sum_z_dr2_top2 < 0.0005125
        - 0.009744151 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # -1.0%  N2 < 0.2233
        + 0.007161894 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +0.7%  N2 < 0.2233 and sum_pt_top5 > 430.8
        - 0.007004305 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # -0.7%  sum_z_dr2_top3 < 0.002152
        - 0.005386087 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -0.5%  lam2 < 0.001131
        + 0.005113692 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +0.5%  lam1 < 0.00484
        + 0.005094587 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2) / 0.003993721   # +0.5%  mass_over_sum_pt < 0.1309 and D2 < 0.746
        + 0.00460127 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.1094252   # +0.5%  N2 < 0.2233 and n_dr_0p1_0p2 < 4
        - 0.003861419 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 0.002544039   # -0.4%  lam1_plus_lam2 < 0.006097
        - 0.003649568 * max(0.0, 0.007520088 - Q.sum_z_dr2) * max(0.0, 0.7459513 - Q.D2) / 9.127003e-05   # -0.4%  sum_z_dr2 < 0.00752 and D2 < 0.746
        - 0.003643622 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.0114178   # -0.4%  N2 < 0.2233 and z_dr_0p05_0p1 < 0.5883
        + 0.003451893 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 716.8828 - Q.sum_pt_top5) / 7.315623   # +0.3%  N2 < 0.2233 and sum_pt_top5 < 716.9
        - 0.003342124 * max(0.0, 0.1492731 - Q.sj2_dr) / 0.04374278   # -0.3%  sj2_dr < 0.1493
        - 0.0006024351 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.121681 - Q.max_dr) / 0.0004905272   # -0.1%  N2 < 0.2233 and max_dr < 0.1217
        - 0.0003907091 * max(0.0, Q.sj3_pair_mass_max - 80.4) / 0.3254107   # -0.0%  sj3_pair_mass_max > 80.4
        + 0.0003492011 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.4684459) / 0.0001955483   # +0.0%  mass_over_sum_pt < 0.1309 and z_dr_0p1_0p2 > 0.4684
        - 0.0002986199 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 1.54073 - Q.pt_entropy) / 0.000718138   # -0.0%  N2 < 0.2233 and pt_entropy < 1.541
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9345747899159664, 0.7753598739495798, 2.043582668067227, 1.0555871848739495, 1.1338457983193277, 1.7399918067226892, 0.9406823529411765, 1.9231317226890756, 0.2768705882352941, 3.0153542016806725, 1.9976579831932773, 2.0073584033613447, 0.0925640756302521, 3.373484243697479, 0.40177741596638655, 0.3300439075630252]
T = [2.3098364700958514, 1.3587021353072477, 3.269575198595063, 2.5416338579963242, 2.8751005974264707]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +38%, n9 +22%, n5 -14%, n1 +13%, n0 -6%, n6 +4% ...
            + 0.3801576 * h[2] / H_AVG[2]
            + 0.2243726 * h[9] / H_AVG[9]
            - 0.1412431 * h[5] / H_AVG[5]
            + 0.131124 * h[1] / H_AVG[1]
            - 0.06321976 * h[0] / H_AVG[0]
            + 0.04454304 * h[6] / H_AVG[6]
            - 0.01533991 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -18%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5634916 * h[9] / H_AVG[9]
            - 0.1837837 * h[10] / H_AVG[10]
            + 0.08654236 * h[6] / H_AVG[6]
            - 0.07823499 * h[4] / H_AVG[4]
            + 0.06002943 * h[5] / H_AVG[5]
            + 0.01518195 * h[15] / H_AVG[15]
            + 0.01273599 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +23%, n3 -16%, n7 +13%, n0 +10%, n14 -9%, n6 -9% ...
            + 0.2302316 * h[11] / H_AVG[11]
            - 0.1614257 * h[3] / H_AVG[3]
            + 0.1286666 * h[7] / H_AVG[7]
            + 0.09825744 * h[0] / H_AVG[0]
            - 0.09216276 * h[14] / H_AVG[14]
            - 0.08990869 * h[6] / H_AVG[6]
            + 0.07254707 * h[13] / H_AVG[13]
            - 0.06939898 * h[15] / H_AVG[15]
            - 0.0288202 * h[9] / H_AVG[9]
            - 0.02117023 * h[8] / H_AVG[8]
            - 0.007410747 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -23%, n6 -14%, n13 +7%, n14 +6%, n1 +4% ...
            + 0.3546805 * h[7] / H_AVG[7]
            - 0.2336166 * h[3] / H_AVG[3]
            - 0.138791 * h[6] / H_AVG[6]
            + 0.07258615 * h[13] / H_AVG[13]
            + 0.0592794 * h[14] / H_AVG[14]
            + 0.03813295 * h[1] / H_AVG[1]
            - 0.03707451 * h[9] / H_AVG[9]
            + 0.03485227 * h[4] / H_AVG[4]
            - 0.02028985 * h[15] / H_AVG[15]
            + 0.01069681 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -48%, n10 +26%, n5 -15%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4766713 * h[13] / H_AVG[13]
            + 0.260555 * h[10] / H_AVG[10]
            - 0.1512983 * h[5] / H_AVG[5]
            + 0.04929592 * h[4] / H_AVG[4]
            + 0.02294674 * h[3] / H_AVG[3]
            + 0.01805615 * h[8] / H_AVG[8]
            - 0.01609754 * h[12] / H_AVG[12]
            + 0.005079033 * h[0] / H_AVG[0]
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
