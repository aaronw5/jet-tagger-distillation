"""JEDI-linear jet tagger, 8 particles, 3 features: one term per observable per neuron (from the 782), re-tuned on the network's predictions (all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.7%   (on for 93% of jets)
  neuron  9:  12.0%   (on for 64% of jets)
  neuron  3:  10.9%   (on for 24% of jets)
  neuron  7:   9.9%   (on for 62% of jets)
  neuron 10:   8.2%   (on for 97% of jets)
  neuron  6:   7.3%   (on for 32% of jets)
  neuron  5:   7.1%   (on for 71% of jets)
  neuron  2:   6.8%   (on for 90% of jets)
  neuron 11:   5.4%   (on for 70% of jets)
  neuron  0:   3.9%   (on for 42% of jets)
  neuron 14:   3.5%   (on for 28% of jets)
  neuron  4:   3.4%   (on for 64% of jets)
  neuron  1:   3.4%   (on for 64% of jets)
  neuron 15:   2.0%   (on for 26% of jets)
  neuron  8:   1.1%   (on for 28% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 89.5% of jets.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its term in ΣzΔR)
  Q.zdr_2                  pT share × ΔR of particle 2 (its term in ΣzΔR)
  Q.zdr_5                  pT share × ΔR of particle 5 (its term in ΣzΔR)
  Q.zdr_6                  pT share × ΔR of particle 6 (its term in ΣzΔR)
  Q.zdr_7                  pT share × ΔR of particle 7 (its term in ΣzΔR)
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
    # scale S = 22.88;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.87725 * (-0.3313899
        + 0.3229677 * Q.log_sum_pt / 6.542932   # +32.3%  log_sum_pt
        + 0.1318979 * max(0.0, 0.01279734 - Q.sum_z_dr2) / 0.007860418   # +13.2%  sum_z_dr2 < 0.0128
        + 0.08354195 * max(0.0, 0.03227056 - Q.e2) / 0.01167233   # +8.4%  e2 < 0.03227
        - 0.05497704 * max(0.0, 0.07988007 - Q.mass_over_sum_pt) / 0.0295085   # -5.5%  mass_over_sum_pt < 0.07988
        - 0.05048984 * max(0.0, 0.08061324 - Q.sum_z_dr) / 0.03125107   # -5.0%  sum_z_dr < 0.08061
        - 0.04601726 * max(0.0, 0.002631161 - Q.lam1_plus_lam2) / 0.0007924469   # -4.6%  lam1_plus_lam2 < 0.002631
        - 0.02734311 * max(0.0, 7.300726e-05 - Q.lam2) / 2.767163e-05   # -2.7%  lam2 < 7.301e-05
        - 0.02588677 * max(0.0, 0.001563465 - Q.C2_b2) / 0.0008800039   # -2.6%  C2_b2 < 0.001563
        + 0.02281204 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +2.3%  planar_flow < 0.1484
        - 0.02119653 * max(0.0, 0.003904593 - Q.mass_over_sum_pt_sq) / 0.001485439   # -2.1%  mass_over_sum_pt_sq < 0.003905
        - 0.02074146 * max(0.0, 0.00543239 - Q.lam1) / 0.00219356   # -2.1%  lam1 < 0.005432
        - 0.016898 * max(0.0, 41.26805 - Q.mass) / 12.48331   # -1.7%  mass < 41.27
        - 0.01578484 * max(0.0, Q.sj3_dr_max - 0.2321253) / 0.02550155   # -1.6%  sj3_dr_max > 0.2321
        + 0.01550067 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, 1.002471 - Q.D2) / 0.001008123   # +1.6%  sum_z_dr2 < 0.01324 and D2 < 1.002
        - 0.0138531 * max(0.0, Q.centroid_offset - 0.03782766) / 0.001682668   # -1.4%  centroid_offset > 0.03783
        + 0.01342028 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2) / 2.36013e-06   # +1.3%  lam2 < 7.301e-05 and D2_b2 < 0.267
        - 0.01120946 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, Q.z_7 - 0.01685855) / 0.0006290825   # -1.1%  log_sum_pt > 6.67 and z_7 > 0.01686
        + 0.01016012 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 0.07232166 - Q.dr_4) / 0.001949106   # +1.0%  log_sum_pt > 6.67 and dr_4 < 0.07232
        - 0.009464786 * max(0.0, Q.centroid_offset - 0.006789738) * max(0.0, 223.375 - Q.pt_1) / 1.309723   # -0.9%  centroid_offset > 0.00679 and pt_1 < 223.4
        + 0.008820577 * max(0.0, 0.002919842 - Q.sum_z_dr2_top3) / 0.00118117   # +0.9%  sum_z_dr2_top3 < 0.00292
        - 0.008536385 * max(0.0, Q.sum_pt - 901.5938) / 12.40362   # -0.9%  sum_pt > 901.6
        + 0.007784904 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) * max(0.0, 0.03532852 - Q.C3) / 3.6978e-05   # +0.8%  lam1_plus_lam2 < 0.004372 and C3 < 0.03533
        - 0.005455623 * max(0.0, Q.centroid_offset - 0.006789738) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.02705531   # -0.5%  centroid_offset > 0.00679 and n_pt_above_50 < 7
        - 0.005109639 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 358.375 - Q.sum_pt_top2) / 2.426548   # -0.5%  planar_flow < 0.1484 and sum_pt_top2 < 358.4
        + 0.005086844 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.08525808 - Q.dr_2) / 1.41829e-06   # +0.5%  lam2 < 7.301e-05 and dr_2 < 0.08526
        - 0.004943623 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # -0.5%  lam1 < 0.006507 and D2 < 0.8757
        + 0.004740151 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, Q.phi_0 - -0.008995056) / 0.0001281354   # +0.5%  sum_z_dr2 < 0.01324 and phi_0 > -0.008995
        - 0.004389377 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.06336451 - Q.dr_4) / 1.344914   # -0.4%  sum_pt_top5 > 687.4 and dr_4 < 0.06336
        + 0.00381517 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.4%  sum_pt_top5 > 687.4
        - 0.00347044 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 0.06473447 - Q.z_7) / 3.579716e-05   # -0.3%  centroid_offset > 0.02077 and z_7 < 0.06473
        + 0.003464715 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 28.78966 - Q.m01) / 1.085363   # +0.3%  log_sum_pt > 6.67 and m01 < 28.79
        - 0.003352446 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05557716 - Q.z_7) / 0.001734349   # -0.3%  sj3_dr_max < 0.3012 and z_7 < 0.05558
        - 0.003346127 * max(0.0, 64.61873 - Q.mass) * max(0.0, 0.1830092 - Q.D2_b2) / 0.3382769   # -0.3%  mass < 64.62 and D2_b2 < 0.183
        + 0.002690561 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.4160096 - Q.pt_dispersion) / 0.0006933006   # +0.3%  planar_flow < 0.1484 and pt_dispersion < 0.416
        - 0.002567504 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05744392 - Q.D2_b2) / 0.0005632394   # -0.3%  sj3_dr_max < 0.3012 and D2_b2 < 0.05744
        - 0.002452997 * max(0.0, 0.007929074 - Q.sum_z_dr2_top3) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0005550791   # -0.2%  sum_z_dr2_top3 < 0.007929 and D2_b2 < 0.5327
        - 0.001670073 * max(0.0, Q.centroid_offset - 0.02076709) * max(0.0, 0.05056028 - Q.dr_2) / 1.335632e-05   # -0.2%  centroid_offset > 0.02077 and dr_2 < 0.05056
        + 0.001304611 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.001732318   # +0.1%  planar_flow < 0.1484 and centroid_offset < 0.0499
        - 0.001143471 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, Q.dr0_6 - 0.1755206) / 0.0008435917   # -0.1%  planar_flow < 0.1484 and dr0_6 > 0.1755
        + 0.0007200769 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.pt_7 - 33.21875) / 68.87156   # +0.1%  sum_pt > 901.6 and pt_7 > 33.22
        - 0.0005245009 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.02270492 - Q.dr_2) / 2.517754e-05   # -0.1%  planar_flow < 0.1484 and dr_2 < 0.0227
        + 0.0004473521 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.02320757 - Q.z_7) / 2.048782e-05   # +0.0%  planar_flow < 0.1484 and z_7 < 0.02321
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 14.82;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.81529 * (0.06141952
        - 0.1586245 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -15.9%  sum_z_dr2 < 0.008678
        + 0.1472408 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) / 0.002578466   # +14.7%  mass_over_sum_pt_sq < 0.005833
        + 0.1304318 * max(0.0, Q.log_sum_pt - 6.503015) / 0.1246376   # +13.0%  log_sum_pt > 6.503
        - 0.06599483 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -6.6%  sum_z_dr < 0.1019
        + 0.05937339 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +5.9%  e3 < 0.0005117
        - 0.05263279 * max(0.0, Q.sum_pt_top5 - 555.7359) / 92.06948   # -5.3%  sum_pt_top5 > 555.7
        + 0.0515239 * max(0.0, 0.008291375 - Q.lam1) / 0.00423277   # +5.2%  lam1 < 0.008291
        - 0.05140344 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 0.002544039   # -5.1%  lam1_plus_lam2 < 0.006097
        + 0.04198595 * max(0.0, Q.sum_pt_top5 - 531.1875) * max(0.0, 0.01392641 - Q.zdr_6) / 1.245695   # +4.2%  sum_pt_top5 > 531.2 and zdr_6 < 0.01393
        - 0.03644852 * max(0.0, 0.07141996 - Q.z_7) / 0.02134987   # -3.6%  z_7 < 0.07142
        - 0.03607052 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # -3.6%  tau1 < 0.07284
        - 0.033276 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, 0.008654951 - Q.zdr_6) / 0.000799435   # -3.3%  log_sum_pt > 6.503 and zdr_6 < 0.008655
        - 0.01836935 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1) / 0.1196218   # -1.8%  log_sum_pt > 6.378 and z_dr_0p05_0p1 < 0.7509
        + 0.01035705 * max(0.0, 0.008678045 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.02076709) / 7.418362e-06   # +1.0%  sum_z_dr2 < 0.008678 and centroid_offset > 0.02077
        - 0.01013161 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 1.679198 - Q.D2) / 0.005760154   # -1.0%  z_7 < 0.06165 and D2 < 1.679
        + 0.009362677 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +0.9%  sj3_dr_max > 0.169
        - 0.007568968 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -0.8%  zdr_0 < 0.02118
        + 0.007337295 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.008921136 - Q.mean_phi2) / 0.0001024498   # +0.7%  z_7 < 0.06165 and mean_phi2 < 0.008921
        + 0.006814688 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # +0.7%  sum_zz_dr2 < 0.008169
        - 0.006429354 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # -0.6%  lam1 < 0.008376 and centroid_offset > 0.02077
        + 0.006425646 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.sj2_dr - 0.1294903) / 0.1877842   # +0.6%  pt_7 > 34.53 and sj2_dr > 0.1295
        - 0.00595704 * max(0.0, 25.53125 - Q.pt_7) / 1.318103   # -0.6%  pt_7 < 25.53
        - 0.005801807 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236) / 1.769648e-05   # -0.6%  mass_over_sum_pt_sq < 0.005833 and centroid_offset > 0.008092
        + 0.005340528 * max(0.0, 49.6681 - Q.mass) / 17.06893   # +0.5%  mass < 49.67
        + 0.005187061 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.centroid_offset - 0.009480685) / 0.0009010889   # +0.5%  log_sum_pt > 6.378 and centroid_offset > 0.009481
        - 0.004231002 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 52.90625 - Q.pt_6) / 13.55761   # -0.4%  pt_7 > 34.53 and pt_6 < 52.91
        + 0.003934209 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 1.432482 - Q.D2) / 0.01970563   # +0.4%  log_sum_pt > 6.606 and D2 < 1.432
        - 0.003588451 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.5700307   # -0.4%  sj3_dr_max > 0.169 and sj3_pair_mass_min > 5.744
        - 0.003549442 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.04875823) / 0.002321449   # -0.4%  z_7 < 0.06165 and z_dr_0p05_0p1 > 0.04876
        - 0.003412021 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.06297984 - Q.tau2) / 0.0007776805   # -0.3%  z_7 < 0.06165 and tau2 < 0.06298
        - 0.00327539 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1484197 - Q.planar_flow) / 0.0001369594   # -0.3%  lam1 < 0.008376 and planar_flow < 0.1484
        - 0.00230534 * max(0.0, 0.1019409 - Q.sum_z_dr) * max(0.0, Q.pair_mass_0_6 - 4.037975) / 0.1185817   # -0.2%  sum_z_dr < 0.1019 and pair_mass_0_6 > 4.038
        - 0.002208106 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # -0.2%  pt_7 > 34.53 and tau2 < 0.01713
        + 0.001305813 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.1057739 - Q.abseta_0) / 0.001206885   # +0.1%  z_7 < 0.06165 and abseta_0 < 0.1058
        + 0.001180473 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 3.175597   # +0.1%  pt_7 > 34.53 and n_dr_0p1_0p2 > 1
        - 0.0008148924 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, 0.03885671 - Q.D2_b2) / 4.310709e-07   # -0.1%  mass_over_sum_pt_sq < 0.005833 and D2_b2 < 0.03886
        + 0.000105354 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.00323438   # +0.0%  lam1 < 0.008376 and n_pt_above_50 > 5
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 7.638;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.638321 * (0.1069432
        + 0.1467295 * max(0.0, Q.pt_7 - 20.09375) / 15.09823   # +14.7%  pt_7 > 20.09
        - 0.1340463 * max(0.0, Q.LHA - 0.1329373) / 0.1175814   # -13.4%  LHA > 0.1329
        + 0.102101 * max(0.0, 62.55 - Q.sj3_pair_mass_max) / 30.55764   # +10.2%  sj3_pair_mass_max < 62.55
        + 0.09533315 * max(0.0, 0.1079857 - Q.mass_over_sum_pt) / 0.05207765   # +9.5%  mass_over_sum_pt < 0.108
        + 0.08051032 * max(0.0, 6.572982 - Q.log_sum_pt) / 0.1163432   # +8.1%  log_sum_pt < 6.573
        - 0.06309586 * max(0.0, Q.z_7 - 0.03971184) / 0.01631316   # -6.3%  z_7 > 0.03971
        - 0.05331769 * max(0.0, 36.22941 - Q.mass) / 10.11393   # -5.3%  mass < 36.23
        - 0.03723676 * max(0.0, Q.sum_pt - 527.1781) * max(0.0, 0.0001947983 - Q.lam2) / 0.02863163   # -3.7%  sum_pt > 527.2 and lam2 < 0.0001948
        - 0.02892215 * max(0.0, 0.4926918 - Q.planar_flow) / 0.2728654   # -2.9%  planar_flow < 0.4927
        + 0.02749595 * max(0.0, 715.5 - Q.sum_pt) / 69.92768   # +2.7%  sum_pt < 715.5
        + 0.02721073 * max(0.0, 0.0001947983 - Q.lam2) / 0.000112076   # +2.7%  lam2 < 0.0001948
        + 0.02547666 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # +2.5%  zdr_0 < 0.02118
        + 0.02412369 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.pt_6 - 19.46875) / 0.05232739   # +2.4%  lam1 < 0.005954 and pt_6 > 19.47
        - 0.02260864 * max(0.0, 0.2507612 - Q.max_dr) / 0.1316542   # -2.3%  max_dr < 0.2508
        - 0.01898792 * max(0.0, 62.55 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.01096064) / 0.2139385   # -1.9%  sj3_pair_mass_max < 62.55 and centroid_offset > 0.01096
        - 0.01293943 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 0.0174408 - Q.abseta_0) / 0.216283   # -1.3%  sum_pt_top5 > 752.1 and abseta_0 < 0.01744
        - 0.01103794 * max(0.0, 62.55 - Q.sj3_pair_mass_max) * max(0.0, 0.06810151 - Q.z_7) / 0.6094489   # -1.1%  sj3_pair_mass_max < 62.55 and z_7 < 0.0681
        - 0.01081151 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.max_dr - 0.08050702) / 4.493327e-05   # -1.1%  lam1 < 0.005954 and max_dr > 0.08051
        + 0.01046246 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 1.129616 - Q.D2_b2) / 6.295733   # +1.0%  sum_pt_top5 > 752.1 and D2_b2 < 1.13
        + 0.0098083 * max(0.0, 36.22941 - Q.mass) * max(0.0, 73.75 - Q.pt_5) / 259.9887   # +1.0%  mass < 36.23 and pt_5 < 73.75
        - 0.009041947 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # -0.9%  sum_z_dr < 0.007674
        - 0.007844873 * max(0.0, Q.sum_pt_top5 - 752.1) / 21.27415   # -0.8%  sum_pt_top5 > 752.1
        + 0.006254794 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.0001947983 - Q.lam2) / 1.313144e-06   # +0.6%  log_sum_pt > 6.843 and lam2 < 0.0001948
        + 0.00601726 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.0174408 - Q.abseta_0) / 8.283244e-05   # +0.6%  log_sum_pt > 6.843 and abseta_0 < 0.01744
        + 0.005630736 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # +0.6%  lam1 < 0.001504
        - 0.004646543 * max(0.0, Q.z_7 - 0.04939969) * max(0.0, 0.0623951 - Q.sj3_dr_min) / 0.0002978985   # -0.5%  z_7 > 0.0494 and sj3_dr_min < 0.0624
        - 0.004591962 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 1.345805 - Q.D2_b2) / 0.002918988   # -0.5%  log_sum_pt > 6.843 and D2_b2 < 1.346
        + 0.004203185 * max(0.0, Q.m012 - 40.2) / 1.307209   # +0.4%  m012 > 40.2
        + 0.002551612 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.4474937 - Q.z_dr_0p05_0p1) / 0.00315435   # +0.3%  log_sum_pt > 6.843 and z_dr_0p05_0p1 < 0.4475
        + 0.002270858 * max(0.0, 0.007673833 - Q.sum_z_dr) * max(0.0, 75.625 - Q.pt_4) / 0.004511241   # +0.2%  sum_z_dr < 0.007674 and pt_4 < 75.62
        - 0.001909903 * max(0.0, 0.03243272 - Q.z_7) * max(0.0, Q.mass_top3 - 16.89912) / 0.01064848   # -0.2%  z_7 < 0.03243 and mass_top3 > 16.9
        - 0.001560261 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.pt_6 - 41.21875) / 0.06090836   # -0.2%  log_sum_pt > 6.843 and pt_6 > 41.22
        - 0.001207347 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # -0.1%  sum_z_dr2 < 0.003563
        - 1.268717e-05 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.dr02 - 0.2623104) / 3.697963e-06   # -0.0%  log_sum_pt > 6.843 and dr02 > 0.2623
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 22.42;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.42193 * (0.1376651
        - 0.2435686 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -24.4%  sum_z_dr < 0.1019
        - 0.1456854 * max(0.0, Q.sum_z_dr2 - 0.008566102) / 0.00223691   # -14.6%  sum_z_dr2 > 0.008566
        - 0.103521 * max(0.0, 0.01683059 - Q.mass_over_sum_pt_sq) / 0.01173664   # -10.4%  mass_over_sum_pt_sq < 0.01683
        + 0.06296758 * max(0.0, Q.sj3_dr_max - 0.1874999) / 0.03955143   # +6.3%  sj3_dr_max > 0.1875
        + 0.0531511 * max(0.0, Q.mass_over_sum_pt - 0.0681391) / 0.01539157   # +5.3%  mass_over_sum_pt > 0.06814
        + 0.04915974 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +4.9%  e2 > 0.04447
        + 0.04742477 * max(0.0, Q.sj2_dr - 0.1488981) / 0.04470703   # +4.7%  sj2_dr > 0.1489
        - 0.04658529 * max(0.0, Q.tau1 - 0.03461338) / 0.03724626   # -4.7%  tau1 > 0.03461
        + 0.02767163 * max(0.0, Q.lam1 - 0.01167645) / 0.001277708   # +2.8%  lam1 > 0.01168
        + 0.02602784 * max(0.0, Q.centroid_offset - 0.01096064) / 0.008813008   # +2.6%  centroid_offset > 0.01096
        - 0.02281287 * max(0.0, 0.9008535 - Q.z_dr_0_0p05) / 0.3811445   # -2.3%  z_dr_0_0p05 < 0.9009
        + 0.02233968 * max(0.0, Q.sum_z_dr - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494) / 0.008781747   # +2.2%  sum_z_dr > 0.04082 and log_sum_pt > 6.08
        - 0.01936861 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.02673292   # -1.9%  max_dr > 0.1028 and z_dr_0p05_0p1 < 0.846
        - 0.01684483 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.1950293   # -1.7%  mass_over_sum_pt > 0.06814 and sj3_pair_mass_min > 5.744
        - 0.01204607 * max(0.0, Q.mass - 64.61873) / 3.274818   # -1.2%  mass > 64.62
        - 0.01146705 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113) / 0.39507   # -1.1%  sj2_dr > 0.1873 and sj2_mass1 > 2.25
        - 0.01143393 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr) / 0.0001196243   # -1.1%  mass_over_sum_pt > 0.06814 and sj2_dr < 0.218
        + 0.01001328 * max(0.0, Q.lam1_plus_lam2 - 0.01279734) / 0.001508311   # +1.0%  lam1_plus_lam2 > 0.0128
        + 0.009581895 * max(0.0, 1.0 - Q.n_dr_0_0p05) / 0.2859429   # +1.0%  n_dr_0_0p05 < 1
        + 0.006491453 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 0.07861328 - Q.abseta_0) / 0.0002969212   # +0.6%  centroid_offset > 0.01096 and abseta_0 < 0.07861
        + 0.005474236 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # +0.5%  sj3_pair_mass_min > 11.05
        + 0.005288536 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sj2_mass1 - 16.86126) / 0.02534836   # +0.5%  e2 > 0.06344 and sj2_mass1 > 16.86
        + 0.005283162 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6) / 0.005634141   # +0.5%  lam2 > 0.001131 and pt_6 < 56.53
        - 0.005033934 * max(0.0, 0.01115288 - Q.sum_z_dr2_top5) / 0.006888859   # -0.5%  sum_z_dr2_top5 < 0.01115
        + 0.003908012 * max(0.0, Q.sd_mass - 62.55) * max(0.0, 0.9206502 - Q.D2_b2) / 1.823781   # +0.4%  sd_mass > 62.55 and D2_b2 < 0.9207
        + 0.003242925 * max(0.0, -0.004664942 - Q.mean_eta) / 0.003754527   # +0.3%  mean_eta < -0.004665
        - 0.003212385 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 0.1626038 - Q.abseta_7) / 0.0008475577   # -0.3%  centroid_offset > 0.01096 and abseta_7 < 0.1626
        + 0.003131016 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.06081235 - Q.z_6) / 0.0004655277   # +0.3%  max_dr > 0.1028 and z_6 < 0.06081
        + 0.002830485 * max(0.0, 1.0 - Q.n_dr_0_0p05) * max(0.0, 4.0 - Q.n_dr_0p05_0p1) / 0.3078555   # +0.3%  n_dr_0_0p05 < 1 and n_dr_0p05_0p1 < 4
        + 0.002391378 * max(0.0, Q.LHA - 0.3467135) / 0.008898884   # +0.2%  LHA > 0.3467
        + 0.001917073 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, Q.z_6 - 0.05441452) / 6.527736e-06   # +0.2%  lam2 > 0.001131 and z_6 > 0.05441
        - 0.00160754 * max(0.0, Q.sj2_dr - 0.2687922) * max(0.0, 0.9641201 - Q.z_dr_0p05_0p1) / 0.005973659   # -0.2%  sj2_dr > 0.2688 and z_dr_0p05_0p1 < 0.9641
        - 0.001448988 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.05268713 - Q.dr_3) / 0.000155932   # -0.1%  sj2_dr > 0.1873 and dr_3 < 0.05269
        + 0.001302701 * max(0.0, Q.sum_z_dr - 0.07608178) * max(0.0, Q.eta_0 - 0.07952881) / 0.0001300296   # +0.1%  sum_z_dr > 0.07608 and eta_0 > 0.07953
        + 0.001237652 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.pt_6 - 31.90625) / 0.1358885   # +0.1%  mass_over_sum_pt > 0.06814 and pt_6 > 31.91
        - 0.001132665 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 27.57812 - Q.pt_6) / 0.028355   # -0.1%  sj2_dr > 0.1873 and pt_6 < 27.58
        - 0.0009642328 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, Q.abseta_0 - 0.1057739) / 0.0002451453   # -0.1%  max_dr > 0.1028 and abseta_0 > 0.1058
        + 0.0009445877 * max(0.0, Q.sd_mass - 54.15028) / 5.452615   # +0.1%  sd_mass > 54.15
        - 0.000811231 * max(0.0, Q.sum_z_dr - 0.08723651) * max(0.0, Q.z_dr_0p05_0p1 - 0.6747704) / 5.945324e-05   # -0.1%  sum_z_dr > 0.08724 and z_dr_0p05_0p1 > 0.6748
        - 0.0006150456 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, -0.0892334 - Q.eta_1) / 0.0002499278   # -0.1%  max_dr > 0.1028 and eta_1 < -0.08923
        - 5.964227e-05 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 2.662284e-07   # -0.0%  sum_z_dr2 > 0.01883 and z_dr_0p05_0p1 > 0.7509
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 74.2;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 74.19627 * (-0.03878795
        + 0.3084491 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # +30.8%  sum_zz_dr2 < 0.01166
        - 0.2495754 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -25.0%  mass_over_sum_pt_sq < 0.01166
        + 0.06015782 * max(0.0, 0.06729223 - Q.C2) / 0.04162086   # +6.0%  C2 < 0.06729
        - 0.05528059 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -5.5%  sum_z_dr2 < 0.008678
        + 0.03408021 * max(0.0, 0.01662178 - Q.sum_z_dr2_top5) / 0.01167611   # +3.4%  sum_z_dr2_top5 < 0.01662
        + 0.03354608 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +3.4%  N2 < 0.2233
        - 0.03110044 * max(0.0, 0.06278829 - Q.e2) / 0.03598313   # -3.1%  e2 < 0.06279
        - 0.02601287 * max(0.0, 0.001100839 - Q.lam2) / 0.0008868955   # -2.6%  lam2 < 0.001101
        - 0.02267804 * max(0.0, 0.04990367 - Q.centroid_offset) / 0.03352573   # -2.3%  centroid_offset < 0.0499
        - 0.02059006 * max(0.0, 0.2122942 - Q.sj3_dr_max) / 0.0750214   # -2.1%  sj3_dr_max < 0.2123
        + 0.01722927 * max(0.0, 0.3467135 - Q.LHA) / 0.1128474   # +1.7%  LHA < 0.3467
        - 0.01520083 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266) / 0.01381411   # -1.5%  N2 < 0.2233 and eccentricity > 0.7117
        + 0.01341595 * max(0.0, Q.lam1_plus_lam2 - 0.006688642) / 0.002699226   # +1.3%  lam1_plus_lam2 > 0.006689
        + 0.0133237 * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) / 0.004543329   # +1.3%  sum_z_dr2_top2 < 0.00764
        + 0.01331935 * max(0.0, Q.max_dr - 0.06139287) / 0.071506   # +1.3%  max_dr > 0.06139
        + 0.01245904 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +1.2%  tau1 < 0.07284
        - 0.01186512 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -1.2%  mass_over_sum_pt > 0.09041
        - 0.009906364 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.1410336 - Q.dr01) / 4.252003e-05   # -1.0%  lam2 < 0.0005373 and dr01 < 0.141
        + 0.006896212 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +0.7%  e3 < 0.0001869
        + 0.005229383 * max(0.0, Q.mass_top5 - 14.54404) / 16.7137   # +0.5%  mass_top5 > 14.54
        - 0.004938508 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.55 - Q.mass) / 0.3959152   # -0.5%  N2 < 0.2233 and mass < 62.55
        + 0.004885062 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 27.42324 - Q.sj3_pair_mass_min) / 138.7479   # +0.5%  sd_mass > 44.82 and sj3_pair_mass_min < 27.42
        - 0.00429412 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7) / 0.8992052   # -0.4%  N2 < 0.2233 and pt_7 < 53.44
        - 0.003616407 * max(0.0, 0.03874536 - Q.M2) / 0.006229062   # -0.4%  M2 < 0.03875
        - 0.002923884 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.07796252 - Q.dr1_6) / 1.697653e-05   # -0.3%  lam2 < 0.0005373 and dr1_6 < 0.07796
        - 0.00288314 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.centroid_offset - 0.001308549) / 0.1760217   # -0.3%  sd_mass > 44.82 and centroid_offset > 0.001309
        - 0.002751293 * max(0.0, Q.mass - 45.7571) / 9.387753   # -0.3%  mass > 45.76
        - 0.002584219 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -0.3%  sum_pt < 739.5
        - 0.001752863 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1218872 - Q.abseta_7) / 0.003384012   # -0.2%  N2 < 0.2233 and abseta_7 < 0.1219
        - 0.001453044 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.275762 - Q.sd_zg) / 0.3498594   # -0.1%  sd_mass > 44.82 and sd_zg < 0.2758
        - 0.001365281 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 7.11886e-05   # -0.1%  N2 < 0.2233 and sum_zz_dr2 > 0.01166
        + 0.001234637 * max(0.0, Q.mass - 76.6557) * max(0.0, 0.107953 - Q.M3) / 0.09817562   # +0.1%  mass > 76.66 and M3 < 0.108
        - 0.001145585 * max(0.0, Q.sd_mass - 86.29383) / 0.7607031   # -0.1%  sd_mass > 86.29
        + 0.0009270681 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # +0.1%  lam1 < 0.001504
        - 0.0007771997 * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) * max(0.0, Q.C2_b2 - 0.0006435798) / 6.028305e-06   # -0.1%  sum_z_dr2_top2 < 0.00764 and C2_b2 > 0.0006436
        - 0.00075655 * max(0.0, Q.lam1_plus_lam2 - 0.003562611) * max(0.0, 0.1057566 - Q.sj3_z3) / 5.535701e-05   # -0.1%  lam1_plus_lam2 > 0.003563 and sj3_z3 < 0.1058
        - 0.000596921 * max(0.0, Q.max_dr - 0.1598486) * max(0.0, 0.0006435798 - Q.C2_b2) / 1.059262e-06   # -0.1%  max_dr > 0.1598 and C2_b2 < 0.0006436
        + 0.000347264 * max(0.0, 0.233678 - Q.sj3_dr_max) * max(0.0, Q.ptdr0_5 - 7.737156) / 0.01098259   # +0.0%  sj3_dr_max < 0.2337 and ptdr0_5 > 7.737
        + 0.0003446749 * max(0.0, 0.2233283 - Q.N2) * max(0.0, -0.009352575 - Q.mean_phi) / 0.0001205535   # +0.0%  N2 < 0.2233 and mean_phi < -0.009353
        - 0.0001064566 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 24.42188 - Q.pt_6) / 0.01371831   # -0.0%  N2 < 0.2233 and pt_6 < 24.42
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 8.782;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.781686 * (0.0007601412
        + 0.1374947 * max(0.0, 0.0553074 - Q.z_7) / 0.0104063   # +13.7%  z_7 < 0.05531
        + 0.09547999 * max(0.0, 0.001653836 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 7.0272e-06   # +9.5%  sum_z_dr2 < 0.001654 and centroid_offset < 0.02355
        - 0.07928908 * max(0.0, 37.15625 - Q.pt_7) / 5.803748   # -7.9%  pt_7 < 37.16
        - 0.07697162 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.001503553 - Q.lam1) / 4.027805e-05   # -7.7%  LHA < 0.2161 and lam1 < 0.001504
        + 0.05823746 * max(0.0, 0.001653836 - Q.sum_z_dr2) / 0.0004289067   # +5.8%  sum_z_dr2 < 0.001654
        + 0.05717175 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +5.7%  LHA < 0.2161
        - 0.05305068 * max(0.0, 0.08051087 - Q.z_6) / 0.02253808   # -5.3%  z_6 < 0.08051
        + 0.05268226 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 46.125 - Q.pt_6) / 2161.354   # +5.3%  sum_pt_top5 > 430.8 and pt_6 < 46.12
        - 0.04110353 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -4.1%  LHA < 0.2161 and log_sum_pt < 6.804
        - 0.04087695 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -4.1%  zdr_0 < 0.02118
        - 0.03938692 * max(0.0, Q.log_sum_pt - 6.80271) / 0.01279093   # -3.9%  log_sum_pt > 6.803
        + 0.03657152 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 40.2 - Q.mass_top3) / 0.60086   # +3.7%  z_7 < 0.07149 and mass_top3 < 40.2
        + 0.03167603 * max(0.0, 0.01122239 - Q.sum_zz_dr2) / 0.006832884   # +3.2%  sum_zz_dr2 < 0.01122
        - 0.02795047 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # -2.8%  sum_z_dr2_top3 < 0.002152
        + 0.0183065 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.002127561 - Q.mean_phi2) / 2.516806e-05   # +1.8%  z_7 < 0.07149 and mean_phi2 < 0.002128
        + 0.01670988 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.001184249   # +1.7%  lam1_plus_lam2 < 0.003563
        - 0.01547445 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0004149607   # -1.5%  z_7 < 0.07149 and centroid_offset < 0.03117
        + 0.01167909 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01341502 - Q.dr_2) / 0.0001347035   # +1.2%  log_sum_pt > 6.701 and dr_2 < 0.01342
        + 0.01166206 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.sj2_dr - 0.1682655) / 4.310225   # +1.2%  sum_pt_top5 > 430.8 and sj2_dr > 0.1683
        + 0.01131156 * max(0.0, Q.sum_pt_top5 - 752.2188) / 21.2507   # +1.1%  sum_pt_top5 > 752.2
        + 0.009710383 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.phi_0 - -0.0297699) / 0.0002791968   # +1.0%  zdr_0 < 0.02118 and phi_0 > -0.02977
        + 0.009483346 * max(0.0, 37.15625 - Q.pt_7) * max(0.0, Q.pt_5 - 24.57812) / 73.42646   # +0.9%  pt_7 < 37.16 and pt_5 > 24.58
        + 0.009373867 * max(0.0, 0.006789738 - Q.centroid_offset) / 0.001012351   # +0.9%  centroid_offset < 0.00679
        - 0.008497349 * max(0.0, 0.006789738 - Q.centroid_offset) * max(0.0, 56.4375 - Q.pt_5) / 0.01266727   # -0.8%  centroid_offset < 0.00679 and pt_5 < 56.44
        - 0.007454959 * max(0.0, Q.pair_mass_0_4 - 23.26771) / 0.6641733   # -0.7%  pair_mass_0_4 > 23.27
        + 0.007031007 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # +0.7%  pt_5 < 24.58
        + 0.0067927 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 9.030369e-05 - Q.mean_eta2) / 9.99897e-07   # +0.7%  log_sum_pt > 6.701 and mean_eta2 < 9.03e-05
        - 0.006630847 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 501.625 - Q.sum_pt_top2) / 0.2299658   # -0.7%  z_7 < 0.0494 and sum_pt_top2 < 501.6
        - 0.004230289 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 0.009661512 - Q.dr_2) / 1.13159e-05   # -0.4%  z_7 < 0.0494 and dr_2 < 0.009662
        + 0.004117553 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.centroid_offset - 0.009480685) / 0.7981241   # +0.4%  sum_pt_top5 > 430.8 and centroid_offset > 0.009481
        + 0.003754738 * max(0.0, Q.pair_mass_0_4 - 23.26771) * max(0.0, Q.eccentricity - 0.9704496) / 0.009376301   # +0.4%  pair_mass_0_4 > 23.27 and eccentricity > 0.9704
        - 0.003420679 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, Q.mean_phi - -0.02594505) / 0.0001045731   # -0.3%  log_sum_pt > 6.896 and mean_phi > -0.02595
        + 0.002769067 * max(0.0, Q.pair_mass_0_4 - 23.26771) * max(0.0, 0.01423545 - Q.mean_eta2) / 0.004884555   # +0.3%  pair_mass_0_4 > 23.27 and mean_eta2 < 0.01424
        - 0.002071126 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 0.001534584   # -0.2%  LHA < 0.2161 and n_dr_0p2_0p4 > 0
        + 0.001575585 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # +0.2%  sum_z_dr < 0.007674
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 26.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.00674 * (0.005656469
        - 0.1920577 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -19.2%  sum_z_dr2 < 0.008678
        + 0.1287788 * max(0.0, 0.01167645 - Q.lam1) / 0.007037801   # +12.9%  lam1 < 0.01168
        - 0.07454261 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # -7.5%  lam1_plus_lam2 < 0.01324
        - 0.06480363 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # -6.5%  max_dr < 0.1452
        + 0.05652396 * max(0.0, 0.02415398 - Q.C2_b2) / 0.02132459   # +5.7%  C2_b2 < 0.02415
        + 0.05595364 * max(0.0, 0.1278212 - Q.sj3_dr_min) / 0.09151624   # +5.6%  sj3_dr_min < 0.1278
        - 0.04451194 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 1.341509   # -4.5%  mass < 60.63 and z_dr_0p2_0p4 < 0.05644
        - 0.0442535 * max(0.0, 0.02677766 - Q.centroid_offset) / 0.01286388   # -4.4%  centroid_offset < 0.02678
        + 0.04033654 * max(0.0, 0.1136369 - Q.tau1) * max(0.0, 0.01426135 - Q.mean_phi2) / 0.0007578925   # +4.0%  tau1 < 0.1136 and mean_phi2 < 0.01426
        - 0.03948781 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 0.2534037 - Q.planar_flow) / 0.0006566196   # -3.9%  lam1 < 0.012 and planar_flow < 0.2534
        + 0.03535637 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # +3.5%  sj2_dr < 0.1873
        + 0.02370096 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +2.4%  tau1 < 0.1136
        + 0.02321469 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt) / 1.016213   # +2.3%  pt_6 < 41.22 and log_sum_pt < 6.767
        + 0.01902467 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # +1.9%  sum_zz_dr2 < 0.003013
        + 0.01785927 * max(0.0, 29.56813 - Q.mass) / 7.316922   # +1.8%  mass < 29.57
        - 0.01781323 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757) / 0.07098489   # -1.8%  pt_6 < 41.22 and z_7 > 0.02321
        + 0.0158657 * max(0.0, Q.log_sum_pt - 6.270279) / 0.2959784   # +1.6%  log_sum_pt > 6.27
        + 0.01533904 * max(0.0, Q.sj3_dr_max - 0.2321253) / 0.02550155   # +1.5%  sj3_dr_max > 0.2321
        + 0.009021108 * max(0.0, 0.1872617 - Q.sj2_dr) * max(0.0, Q.eccentricity - 0.927072) / 0.0009514438   # +0.9%  sj2_dr < 0.1873 and eccentricity > 0.9271
        - 0.008655727 * max(0.0, 0.0005177258 - Q.lam2) / 0.0003743385   # -0.9%  lam2 < 0.0005177
        - 0.008323129 * max(0.0, Q.sj3_pair_mass_min - 4.501727) / 3.480268   # -0.8%  sj3_pair_mass_min > 4.502
        + 0.007840694 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 43.23531   # +0.8%  sum_pt < 615.9 and n_dr_0p2_0p4 < 2
        - 0.00759813 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.8%  z_6 < 0.0216
        + 0.005613337 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.psi_0p1 - 0.4008925) / 0.003485257   # +0.6%  centroid_offset > 0.008092 and psi_0p1 > 0.4009
        + 0.005262158 * max(0.0, 19.45312 - Q.pt_6) / 0.250827   # +0.5%  pt_6 < 19.45
        + 0.005177549 * max(0.0, 6.638339 - Q.log_sum_pt) * max(0.0, 0.0753896 - Q.z_7) / 0.001441873   # +0.5%  log_sum_pt < 6.638 and z_7 < 0.07539
        + 0.004083413 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # +0.4%  sum_pt > 988.4
        - 0.00391094 * max(0.0, Q.tau2 - 0.008780509) / 0.00811667   # -0.4%  tau2 > 0.008781
        - 0.003727285 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.4%  sum_pt_top5 > 840
        + 0.003620829 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2) / 2.179362e-05   # +0.4%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        + 0.002474386 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308) / 0.05438016   # +0.2%  centroid_offset > 0.008092 and sj3_pair_mass_min > 6.811
        + 0.002358999 * max(0.0, Q.sj3_dr23 - 0.2207152) / 0.01962555   # +0.2%  sj3_dr23 > 0.2207
        - 0.001916994 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 2.69383   # -0.2%  sj3_pair_mass_min > 4.502 and n_dr_0p2_0p4 > 1
        - 0.001591653 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.002776626 - Q.mean_phi2) / 0.01000982   # -0.2%  sum_pt > 988.4 and mean_phi2 < 0.002777
        - 0.001489289 * max(0.0, Q.sj2_mass1 - 40.2) * max(0.0, 0.4490772 - Q.tau21_b2) / 0.01473   # -0.1%  sj2_mass1 > 40.2 and tau21_b2 < 0.4491
        + 0.001342571 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 0.1343804 - Q.dr1_7) / 0.3834372   # +0.1%  pt_6 < 41.22 and dr1_7 < 0.1344
        + 0.001229501 * max(0.0, Q.sj3_dr13 - 0.181053) / 0.02525513   # +0.1%  sj3_dr13 > 0.1811
        + 0.0009751738 * max(0.0, 29.90625 - Q.pt_6) * max(0.0, 8.0 - Q.n_pt_above_10) / 0.5895401   # +0.1%  pt_6 < 29.91 and n_pt_above_10 < 8
        - 0.0008779709 * max(0.0, 0.1872617 - Q.sj2_dr) * max(0.0, Q.dr1_6 - 0.07796252) / 0.0003473731   # -0.1%  sj2_dr < 0.1873 and dr1_6 > 0.07796
        + 0.0008317549 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 24.57812 - Q.pt_5) / 1.825775   # +0.1%  sum_pt < 615.9 and pt_5 < 24.58
        + 0.000664546 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, Q.pt_4 - 65.1875) / 0.007630915   # +0.1%  centroid_offset > 0.01838 and pt_4 > 65.19
        - 0.0006243586 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # -0.1%  mean_eta > 0.02644
        - 0.0005094447 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, Q.M3 - 0.06688759) / 9.396674e-06   # -0.1%  mean_eta > 0.02644 and M3 > 0.06689
        + 0.0003686635 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.mean_eta - 0.02644207) / 1.54475e-06   # +0.0%  lam1 < 0.012 and mean_eta > 0.02644
        - 0.0003514801 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, Q.sj3_dr12 - 0.1692253) / 0.0003039084   # -0.0%  centroid_offset > 0.01838 and sj3_dr12 > 0.1692
        + 0.0001348863 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002658741   # +0.0%  lam2 < 0.003408 and n_dr_0_0p05 < 3
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 17.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.41343 * (0.2604613
        - 0.198293 * max(0.0, Q.mass_over_sum_pt - 0.1060006) / 0.005659886   # -19.8%  mass_over_sum_pt > 0.106
        - 0.1391632 * max(0.0, 0.08699327 - Q.sum_z_dr) / 0.03619126   # -13.9%  sum_z_dr < 0.08699
        - 0.0628857 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -6.3%  lam1 < 0.008376
        - 0.0626462 * max(0.0, Q.lam1_plus_lam2 - 0.01279734) / 0.001508311   # -6.3%  lam1_plus_lam2 > 0.0128
        + 0.05095288 * max(0.0, Q.sum_z_dr2 - 0.01855051) / 0.0007880277   # +5.1%  sum_z_dr2 > 0.01855
        + 0.04788537 * max(0.0, Q.sj3_dr_max - 0.1416042) / 0.06339998   # +4.8%  sj3_dr_max > 0.1416
        - 0.04689657 * max(0.0, Q.lam1_plus_lam2 - 0.008678045) * max(0.0, 38.25 - Q.pt_6) / 0.006941575   # -4.7%  lam1_plus_lam2 > 0.008678 and pt_6 < 38.25
        - 0.04155543 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.004032342 - Q.C2_b2) / 2.736263e-05   # -4.2%  centroid_offset < 0.02077 and C2_b2 < 0.004032
        + 0.03311301 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +3.3%  LHA < 0.3033
        + 0.02618241 * max(0.0, 0.08723651 - Q.sum_z_dr) * max(0.0, 0.004032342 - Q.C2_b2) / 0.0001217265   # +2.6%  sum_z_dr < 0.08724 and C2_b2 < 0.004032
        + 0.02418666 * max(0.0, Q.sum_z_dr2 - 0.01323868) * max(0.0, 38.25 - Q.pt_6) / 0.004519922   # +2.4%  sum_z_dr2 > 0.01324 and pt_6 < 38.25
        + 0.02176226 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +2.2%  tau1 < 0.05357
        - 0.02124154 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # -2.1%  lam2 < 0.0003061
        + 0.01943156 * max(0.0, 0.0003061234 - Q.lam2) * max(0.0, 0.04019753 - Q.tau21_b2) / 2.611351e-06   # +1.9%  lam2 < 0.0003061 and tau21_b2 < 0.0402
        + 0.01770282 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sd_mass - 38.43971) / 1.25962   # +1.8%  planar_flow < 0.195 and sd_mass > 38.44
        + 0.0164769 * max(0.0, 0.09317241 - Q.sj2_dr) / 0.02197556   # +1.6%  sj2_dr < 0.09317
        - 0.01444421 * max(0.0, Q.mass - 87.93725) / 0.7342658   # -1.4%  mass > 87.94
        + 0.01376054 * max(0.0, Q.z_7 - 0.03243272) / 0.02180051   # +1.4%  z_7 > 0.03243
        - 0.01345642 * max(0.0, 0.04019753 - Q.tau21_b2) / 0.01109641   # -1.3%  tau21_b2 < 0.0402
        - 0.01237343 * max(0.0, 0.06139287 - Q.max_dr) / 0.009201578   # -1.2%  max_dr < 0.06139
        - 0.01180811 * max(0.0, Q.mass - 76.6557) * max(0.0, Q.zdr_6 - 0.006366792) / 0.006641425   # -1.2%  mass > 76.66 and zdr_6 > 0.006367
        + 0.01038952 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.02656143 - Q.tau21_b2) / 4.805781e-05   # +1.0%  centroid_offset < 0.02077 and tau21_b2 < 0.02656
        + 0.009845074 * max(0.0, 0.001101266 - Q.sum_zz_dr2) / 0.000308004   # +1.0%  sum_zz_dr2 < 0.001101
        - 0.009813479 * max(0.0, Q.sum_z_dr2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9458207) / 2.125219e-05   # -1.0%  sum_z_dr2 > 0.01324 and eccentricity > 0.9458
        - 0.008826693 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # -0.9%  e2 > 0.05028
        - 0.007823645 * max(0.0, Q.centroid_offset - 0.0500602) / 0.0007985175   # -0.8%  centroid_offset > 0.05006
        - 0.007523151 * max(0.0, 2.357246 - Q.D2) / 1.097329   # -0.8%  D2 < 2.357
        + 0.006873954 * max(0.0, Q.mass_over_sum_pt - 0.09041383) * max(0.0, 36.8125 - Q.pt_6) / 0.01975833   # +0.7%  mass_over_sum_pt > 0.09041 and pt_6 < 36.81
        - 0.006462363 * max(0.0, 29.04219 - Q.pt_7) / 2.179984   # -0.6%  pt_7 < 29.04
        + 0.004780484 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, Q.sum_pt_top3 - 331.25) / 1.710819   # +0.5%  centroid_offset < 0.02077 and sum_pt_top3 > 331.2
        - 0.004519896 * max(0.0, 0.05744392 - Q.D2_b2) / 0.005938983   # -0.5%  D2_b2 < 0.05744
        - 0.004078898 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) / 0.0003002514   # -0.4%  sum_z_dr2_top2 < 0.001057
        + 0.003908219 * max(0.0, Q.mass_top5 - 53.60766) / 1.978814   # +0.4%  mass_top5 > 53.61
        + 0.003868189 * max(0.0, Q.sum_z_dr2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +0.4%  sum_z_dr2 > 0.004372 and planar_flow < 0.195
        + 0.00379092 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) / 0.09532154   # +0.4%  z_dr_0p1_0p2 < 0.1586
        - 0.002776715 * max(0.0, 0.1950135 - Q.planar_flow) / 0.07573236   # -0.3%  planar_flow < 0.195
        - 0.002580364 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # -0.3%  sd_rg > 0.2788
        - 0.00230943 * max(0.0, Q.centroid_offset - 0.03117077) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.00188116   # -0.2%  centroid_offset > 0.03117 and n_pt_above_50 > 4
        - 0.001676627 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6) / 0.1690249   # -0.2%  planar_flow < 0.195 and pt_6 < 35.28
        - 0.0009743809 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) * max(0.0, 0.02656143 - Q.tau21_b2) / 2.239525e-07   # -0.1%  sum_z_dr2_top2 < 0.001057 and tau21_b2 < 0.02656
        - 0.0009598369 * max(0.0, Q.sj3_dr_max - 0.1426152) * max(0.0, 19.46875 - Q.pt_6) / 0.01292439   # -0.1%  sj3_dr_max > 0.1426 and pt_6 < 19.47
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 22.34;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.33832 * (-0.0400731
        + 0.08420343 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 4.808955e-05   # +8.4%  tau1 < 0.05357 and lam1_plus_lam2 < 0.003563
        + 0.07976689 * max(0.0, 0.003559066 - Q.lam1_plus_lam2) / 0.001182667   # +8.0%  lam1_plus_lam2 < 0.003559
        - 0.077008 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.003562611 - Q.sum_z_dr2) / 5.329563e-05   # -7.7%  sum_z_dr < 0.06109 and sum_z_dr2 < 0.003563
        - 0.06642748 * max(0.0, 0.06108601 - Q.sum_z_dr) / 0.01868685   # -6.6%  sum_z_dr < 0.06109
        + 0.06352211 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0001901112   # +6.4%  sum_z_dr2 < 0.00502 and z_dr_0p2_0p4 < 0.101
        + 0.06130448 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.0003703114   # +6.1%  sj3_dr_max < 0.1986 and sum_z_dr2 < 0.006679
        - 0.0450417 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.0001781324   # -4.5%  sj3_dr_max < 0.1986 and lam1_plus_lam2 < 0.003563
        + 0.04033753 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +4.0%  sum_z_dr2 < 0.00502
        - 0.03830976 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.1231549   # -3.8%  mass < 29.64 and centroid_offset < 0.02686
        + 0.03682996 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +3.7%  sum_z_dr2 < 0.006679 and centroid_offset < 0.02355
        - 0.03602263 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # -3.6%  tau1 < 0.05357
        - 0.03285099 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # -3.3%  LHA < 0.1967
        + 0.03274691 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.0001272072   # +3.3%  mass_over_sum_pt < 0.03319 and centroid_offset < 0.02686
        - 0.0289451 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.005691733 - Q.sum_z_dr2_top5) / 0.0003116052   # -2.9%  sj3_dr_max < 0.1986 and sum_z_dr2_top5 < 0.005692
        - 0.02127839 * max(0.0, 49.6681 - Q.mass) / 17.06893   # -2.1%  mass < 49.67
        - 0.02055874 * max(0.0, 0.1591651 - Q.sj3_dr_max) / 0.04411711   # -2.1%  sj3_dr_max < 0.1592
        - 0.02043823 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -2.0%  log_sum_pt > 6.701
        - 0.02001877 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.001011075   # -2.0%  sum_z_dr < 0.06109 and z_dr_0p2_0p4 < 0.05644
        + 0.01946352 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.0001947983 - Q.lam2) / 9.822054e-06   # +1.9%  sj3_dr_max < 0.1986 and lam2 < 0.0001948
        - 0.0159575 * max(0.0, 0.01289969 - Q.e2) * max(0.0, Q.psi_0p2 - 0.79448) / 0.0005184882   # -1.6%  e2 < 0.0129 and psi_0p2 > 0.7945
        + 0.01310154 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.0001947983 - Q.lam2) / 2.869243e-06   # +1.3%  sum_z_dr < 0.06109 and lam2 < 0.0001948
        - 0.01187433 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 43.5 - Q.pt_7) / 0.02255472   # -1.2%  sum_z_dr2 < 0.00502 and pt_7 < 43.5
        - 0.01159538 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, Q.centroid_offset - 0.006789738) / 8.239263e-05   # -1.2%  sum_z_dr < 0.06109 and centroid_offset > 0.00679
        - 0.01102637 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.0001947983 - Q.lam2) / 0.0007178522   # -1.1%  mass < 21.78 and lam2 < 0.0001948
        + 0.01078095 * max(0.0, 0.0001330621 - Q.lam2) * max(0.0, 4.721224 - Q.D2_b2) / 0.0002340602   # +1.1%  lam2 < 0.0001331 and D2_b2 < 4.721
        - 0.01062319 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.01627885) / 0.0001130126   # -1.1%  sj3_dr_max < 0.1426 and centroid_offset > 0.01628
        + 0.01061696 * max(0.0, 0.01289969 - Q.e2) / 0.002527207   # +1.1%  e2 < 0.0129
        + 0.009610758 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, Q.phi_0 - -0.04013062) / 0.0001189753   # +1.0%  sum_z_dr2 < 0.006679 and phi_0 > -0.04013
        + 0.009321884 * max(0.0, 21.78408 - Q.mass) * max(0.0, 48.71875 - Q.pt_7) / 69.12746   # +0.9%  mass < 21.78 and pt_7 < 48.72
        + 0.009199945 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 48.71875 - Q.pt_7) / 0.776931   # +0.9%  log_sum_pt > 6.701 and pt_7 < 48.72
        - 0.009130681 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 4.721224 - Q.D2_b2) / 0.008595177   # -0.9%  sum_z_dr2 < 0.006679 and D2_b2 < 4.721
        - 0.00684287 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) / 0.00722828   # -0.7%  mass_over_sum_pt < 0.03319
        + 0.00679301 * max(0.0, 29.6447 - Q.mass) * max(0.0, 1.762929e-06 - Q.e3) / 9.656157e-06   # +0.7%  mass < 29.64 and e3 < 1.763e-06
        + 0.006097159 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.pt_7 - 15.55391) / 0.4153091   # +0.6%  LHA < 0.1967 and pt_7 > 15.55
        + 0.005423766 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, Q.lam1_plus_lam2 - 0.0003193707) / 1.048473e-05   # +0.5%  sum_z_dr < 0.06109 and lam1_plus_lam2 > 0.0003194
        + 0.00433648 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +0.4%  sum_pt_top5 > 687.4
        - 0.003105902 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 8.0 - Q.n_pt_above_10) / 7.731645   # -0.3%  sum_pt_top5 > 687.4 and n_pt_above_10 < 8
        - 0.002120058 * max(0.0, 0.007078158 - Q.e2) * max(0.0, Q.sum_z_dr2_top5 - 5.672047e-05) / 1.487483e-07   # -0.2%  e2 < 0.007078 and sum_z_dr2_top5 > 5.672e-05
        - 0.001680361 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.2%  sum_pt > 988.4
        + 0.001449377 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.centroid_offset - 0.01837778) / 3.05705e-06   # +0.1%  LHA < 0.1967 and centroid_offset > 0.01838
        + 0.001367047 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0005611231 - Q.sum_z_dr2) / 8.626944e-06   # +0.1%  log_sum_pt > 6.701 and sum_z_dr2 < 0.0005611
        - 0.001145055 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.2097233   # -0.1%  mass < 29.64 and D2_b2 < 0.7166
        - 0.0008867867 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 0.01544159   # -0.1%  log_sum_pt > 6.701 and n_dr_0p05_0p1 > 2
        + 0.0008380239 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 0.05062541 - Q.dr12) / 0.1693771   # +0.1%  sum_pt > 988.4 and dr12 < 0.05063
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 28.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.63772 * (-0.0419622
        + 0.2213425 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +22.1%  sum_z_dr2 < 0.00502
        - 0.1177196 * max(0.0, 0.01167645 - Q.lam1) / 0.007037801   # -11.8%  lam1 < 0.01168
        - 0.07702224 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -7.7%  mass_over_sum_pt < 0.07269
        + 0.06412533 * max(0.0, 988.0299 - Q.sum_pt) / 276.4074   # +6.4%  sum_pt < 988
        - 0.05474904 * max(0.0, 36.17854 - Q.mass) / 10.09124   # -5.5%  mass < 36.18
        + 0.05209657 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.2973626   # +5.2%  mass < 53.33 and centroid_offset < 0.02686
        + 0.0510895 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +5.1%  centroid_offset < 0.01838
        - 0.04360705 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # -4.4%  sum_zz_dr2 < 0.003013
        + 0.03451087 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # +3.5%  tau1 < 0.0437
        + 0.03134268 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +3.1%  mass_over_sum_pt_sq < 0.007183
        + 0.0225068 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +2.3%  max_dr < 0.1118
        - 0.02142206 * max(0.0, Q.sum_z_dr - 0.0760135) / 0.01061092   # -2.1%  sum_z_dr > 0.07601
        + 0.02052248 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt) / 5.175436   # +2.1%  mass < 53.33 and log_sum_pt < 6.843
        - 0.01979165 * max(0.0, 0.1059541 - Q.sj3_dr_max) / 0.02361947   # -2.0%  sj3_dr_max < 0.106
        + 0.01604775 * max(0.0, Q.e2 - 0.04965722) / 0.003462881   # +1.6%  e2 > 0.04966
        - 0.01310316 * max(0.0, 0.0284695 - Q.C3) / 0.008714909   # -1.3%  C3 < 0.02847
        - 0.01116625 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, Q.z_6 - 0.03448406) / 0.0003071415   # -1.1%  sum_z_dr < 0.05465 and z_6 > 0.03448
        + 0.01065652 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.0002201438   # +1.1%  sum_z_dr2 < 0.006097 and planar_flow < 0.3221
        - 0.01052797 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266) / 0.0005312449   # -1.1%  lam1 > 0.005433 and eccentricity > 0.7117
        - 0.01029511 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1) / 0.1931178   # -1.0%  centroid_offset < 0.01838 and pt_1 < 159.2
        - 0.008622751 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.008712838   # -0.9%  centroid_offset < 0.01838 and n_for_90pct > 5
        - 0.006990445 * max(0.0, Q.D2 - 2.055451) / 0.3061148   # -0.7%  D2 > 2.055
        + 0.005841315 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.6875) / 0.3914557   # +0.6%  sj3_dr_max < 0.1426 and pt_6 > 33.69
        - 0.005457152 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, 0.002834884 - Q.mean_phi) / 1.344572e-05   # -0.5%  sum_z_dr2 < 0.006097 and mean_phi < 0.002835
        + 0.005409322 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd) / 0.0002178654   # +0.5%  centroid_offset < 0.01838 and z_2nd < 0.2056
        - 0.004727541 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255) / 0.000277001   # -0.5%  tau1 < 0.0437 and eccentricity > 0.9031
        + 0.004399213 * max(0.0, Q.N2 - 0.2233283) / 0.04872624   # +0.4%  N2 > 0.2233
        - 0.004339027 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # -0.4%  e3 > 0.0001869
        - 0.004069068 * max(0.0, 0.006292091 - Q.zdr_0) / 0.001006321   # -0.4%  zdr_0 < 0.006292
        + 0.003953042 * max(0.0, 0.0001721983 - Q.lam1_plus_lam2) / 1.441896e-05   # +0.4%  lam1_plus_lam2 < 0.0001722
        - 0.003733249 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.tau4 - 0.001224244) / 1.099241e-05   # -0.4%  centroid_offset < 0.01838 and tau4 > 0.001224
        + 0.00372433 * max(0.0, Q.lam2 - 0.003274457) / 0.000163636   # +0.4%  lam2 > 0.003274
        - 0.003535055 * max(0.0, 0.02227783 - Q.absphi_1) / 0.006894148   # -0.4%  absphi_1 < 0.02228
        + 0.003457149 * Q.z_dr_0p2_0p4 / 0.02920532   # +0.3%  z_dr_0p2_0p4
        + 0.003071358 * max(0.0, 53.33237 - Q.mass) * max(0.0, Q.D2 - 2.843757) / 5.555775   # +0.3%  mass < 53.33 and D2 > 2.844
        + 0.002582926 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.mean_eta - 6.288824e-05) / 1.181456e-05   # +0.3%  centroid_offset < 0.01838 and mean_eta > 6.289e-05
        - 0.002399464 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.2%  sum_pt_top5 > 840
        - 0.002309689 * max(0.0, 7.0 - Q.n_for_90pct) / 0.5603294   # -0.2%  n_for_90pct < 7
        + 0.001767512 * max(0.0, 0.06503035 - Q.z_5) / 0.007566857   # +0.2%  z_5 < 0.06503
        + 0.001708505 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) * max(0.0, -0.004917145 - Q.phi_0) / 9.23841e-05   # +0.2%  mass_over_sum_pt < 0.07269 and phi_0 < -0.004917
        + 0.001574065 * max(0.0, 0.0782171 - Q.M3) / 0.01197321   # +0.2%  M3 < 0.07822
        - 0.00152947 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) / 6.087676   # -0.2%  sj3_pair_mass_max < 24.23
        + 0.001525398 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.2%  n_dr_0p2_0p4 > 1
        - 0.001173962 * max(0.0, 988.4078 - Q.sum_pt) * max(0.0, 0.01778111 - Q.dr_2) / 0.3401129   # -0.1%  sum_pt < 988.4 and dr_2 < 0.01778
        - 0.001098867 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, 0.02656143 - Q.tau21_b2) / 8.029519e-06   # -0.1%  tau1 < 0.0437 and tau21_b2 < 0.02656
        - 0.001083783 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.01108383   # -0.1%  n_dr_0p2_0p4 > 1 and sj3_dr_min < 0.209
        + 0.0009787374 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, 0.02656143 - Q.tau21_b2) / 1.23798e-05   # +0.1%  sum_z_dr < 0.05465 and tau21_b2 < 0.02656
        + 0.0008649598 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, -0.01275329 - Q.mean_phi) / 1.158446e-05   # +0.1%  tau1 < 0.0437 and mean_phi < -0.01275
        + 0.000810033 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.1%  log_sum_pt > 6.896
        - 0.0007106243 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2) / 0.000627491   # -0.1%  log_sum_pt > 6.896 and D2_b2 < 0.7166
        + 0.0006759368 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.001255404 - Q.zdr_5) / 2.213187e-06   # +0.1%  log_sum_pt > 6.896 and zdr_5 < 0.001255
        - 0.0006401665 * max(0.0, Q.sd_mass - 45.595) * max(0.0, Q.n_dr_0p05_0p1 - 7.0) / 0.4890223   # -0.1%  sd_mass > 45.59 and n_dr_0p05_0p1 > 7
        + 0.0004752262 * max(0.0, 31.125 - Q.pt_4) / 0.3140792   # +0.0%  pt_4 < 31.12
        - 0.0004442905 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.zdr_2 - 0.00759156) / 4.975701e-07   # -0.0%  tau1 < 0.0437 and zdr_2 > 0.007592
        - 0.0004213719 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, Q.abseta_1 - 0.03625488) / 8.135786e-06   # -0.0%  sj3_dr_max < 0.107 and abseta_1 > 0.03625
        + 0.0002498239 * max(0.0, Q.e3 - 0.0001869378) * max(0.0, Q.pt_2 - 56.5) / 0.0005941291   # +0.0%  e3 > 0.0001869 and pt_2 > 56.5
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 7.088;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.088144 * (0.1739146
        - 0.1598543 * max(0.0, 0.003377149 - Q.lam1) / 0.001133273   # -16.0%  lam1 < 0.003377
        + 0.136394 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +13.6%  mass < 76.66
        + 0.1251129 * max(0.0, Q.sum_z_dr2 - 0.006688641) / 0.002699226   # +12.5%  sum_z_dr2 > 0.006689
        - 0.08071849 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -8.1%  LHA > 0.3033
        + 0.07219078 * max(0.0, Q.lam2 - 0.0003061234) / 0.0004216599   # +7.2%  lam2 > 0.0003061
        + 0.05250952 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # +5.3%  tau1 > 0.05357
        + 0.04246838 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # +4.2%  sum_z_dr2_top3 < 0.002152
        + 0.04205361 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1) / 0.002082998   # +4.2%  zdr_0 < 0.02118 and z_dr_0p05_0p1 < 0.2919
        - 0.03572223 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -3.6%  e3 > 3.892e-05
        + 0.02865584 * Q.e2 / 0.02863215   # +2.9%  e2
        - 0.0233588 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # -2.3%  sj3_dr_max > 0.1986
        - 0.02163921 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.planar_flow - 0.04505724) / 0.0002791469   # -2.2%  lam2 > 0.0003061 and planar_flow > 0.04506
        + 0.02053575 * max(0.0, Q.sj3_pair_mass_min - 15.95929) / 1.348548   # +2.1%  sj3_pair_mass_min > 15.96
        - 0.01916883 * max(0.0, 37.72285 - Q.mass_top5) / 16.68598   # -1.9%  mass_top5 < 37.72
        - 0.01860238 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 6.572938 - Q.log_sum_pt) / 1.257183   # -1.9%  pt_7 < 45.75 and log_sum_pt < 6.573
        + 0.01671345 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.002471 - Q.D2) / 1.812028   # +1.7%  pt_7 < 45.75 and D2 < 1.002
        - 0.01620733 * max(0.0, 0.00325401 - Q.zdr_7) / 0.001049641   # -1.6%  zdr_7 < 0.003254
        - 0.01302855 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2) / 0.0002929492   # -1.3%  lam1 > 0.00733 and D2_b2 < 0.3809
        - 0.01150597 * max(0.0, 0.009238653 - Q.zdr_0) / 0.002028057   # -1.2%  zdr_0 < 0.009239
        - 0.009216539 * max(0.0, 45.75 - Q.pt_7) / 12.14632   # -0.9%  pt_7 < 45.75
        - 0.008486582 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.8%  sum_pt > 988.4
        + 0.008327893 * max(0.0, Q.C2_b2 - 0.009032972) / 0.001703043   # +0.8%  C2_b2 > 0.009033
        - 0.008244301 * max(0.0, 0.02563286 - Q.M2) / 0.001872375   # -0.8%  M2 < 0.02563
        + 0.007852797 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +0.8%  sj3_dr_min > 0.1278
        - 0.007292791 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.3658817 - Q.z_dr_0_0p05) / 0.002463327   # -0.7%  sj3_dr_min > 0.1278 and z_dr_0_0p05 < 0.3659
        - 0.006100988 * max(0.0, Q.sj3_pair_mass_min - 11.051) * max(0.0, Q.sj3_pairmin_over_m - 0.28737) / 0.2416857   # -0.6%  sj3_pair_mass_min > 11.05 and sj3_pairmin_over_m > 0.2874
        + 0.005283728 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.01258764) / 3.400017e-05   # +0.5%  zdr_0 < 0.02118 and centroid_offset > 0.01259
        - 0.002080371 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, 1.002471 - Q.D2) / 0.009558568   # -0.2%  tau1 > 0.05357 and D2 < 1.002
        + 0.0006736245 * max(0.0, Q.sj2_dr - 0.3003793) / 0.004575695   # +0.1%  sj2_dr > 0.3004
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 54.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 54.57316 * (-0.0323767
        - 0.2939328 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -29.4%  sum_zz_dr2 < 0.008169
        + 0.2146895 * max(0.0, 0.00817466 - Q.mass_over_sum_pt_sq) / 0.004299658   # +21.5%  mass_over_sum_pt_sq < 0.008175
        + 0.1536665 * max(0.0, 0.008566101 - Q.lam1_plus_lam2) / 0.004357777   # +15.4%  lam1_plus_lam2 < 0.008566
        + 0.0615406 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +6.2%  sum_z_dr2 < 0.01324
        - 0.03584114 * max(0.0, 0.07637363 - Q.mass_over_sum_pt) / 0.02715027   # -3.6%  mass_over_sum_pt < 0.07637
        - 0.02425569 * max(0.0, 0.008291375 - Q.lam1) / 0.00423277   # -2.4%  lam1 < 0.008291
        + 0.02235573 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +2.2%  planar_flow < 0.2534
        - 0.02232911 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # -2.2%  tau1 < 0.09539
        + 0.01699012 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +1.7%  z_7 > 0.01686
        - 0.01299725 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.tau3 - 0.001996306) / 6.194014e-06   # -1.3%  centroid_offset > 0.0499 and tau3 > 0.001996
        + 0.01299624 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # +1.3%  lam2 < 0.001131
        - 0.01147319 * max(0.0, 0.06101366 - Q.sum_z_dr) / 0.01864723   # -1.1%  sum_z_dr < 0.06101
        + 0.01131162 * max(0.0, 0.1762762 - Q.max_dr) / 0.06965091   # +1.1%  max_dr < 0.1763
        - 0.01081523 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # -1.1%  C2 < 0.03579
        - 0.008603153 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.9%  centroid_offset > 0.0499
        - 0.007805936 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # -0.8%  LHA < 0.3127
        - 0.007593054 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, Q.zdr_6 - 0.007390416) / 3.11545e-06   # -0.8%  centroid_offset > 0.0499 and zdr_6 > 0.00739
        - 0.007106625 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.7%  planar_flow < 0.2534 and sum_pt < 840
        + 0.006995287 * max(0.0, 1.050302e-05 - Q.e3) / 3.717236e-06   # +0.7%  e3 < 1.05e-05
        - 0.006503295 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 15.95929 - Q.sj3_pair_mass_min) / 0.05098851   # -0.7%  centroid_offset < 0.01438 and sj3_pair_mass_min < 15.96
        - 0.006434364 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt) / 3.513015   # -0.6%  centroid_offset < 0.03776 and sum_pt < 901.6
        - 0.006303598 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, 0.001130645 - Q.lam2) / 1.843299e-05   # -0.6%  sj2_dr > 0.1779 and lam2 < 0.001131
        + 0.003914421 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0007565864   # +0.4%  centroid_offset < 0.01438 and D2_b2 < 0.5327
        + 0.003807029 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.5502779 - Q.tau32) / 0.0001096908   # +0.4%  centroid_offset > 0.0499 and tau32 < 0.5503
        + 0.003746334 * max(0.0, 11.07122 - Q.mass) / 1.183571   # +0.4%  mass < 11.07
        - 0.003679684 * max(0.0, Q.pt_7 - 29.04219) / 7.785987   # -0.4%  pt_7 > 29.04
        - 0.003328626 * max(0.0, 506.875 - Q.sum_pt_top5) / 34.75344   # -0.3%  sum_pt_top5 < 506.9
        - 0.00276989 * max(0.0, 0.1059541 - Q.sj3_dr_max) / 0.02361947   # -0.3%  sj3_dr_max < 0.106
        - 0.002277963 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 1.050302e-05 - Q.e3) / 1.771164e-07   # -0.2%  planar_flow < 0.2534 and e3 < 1.05e-05
        - 0.002004017 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 37.15625 - Q.pt_7) / 0.5537514   # -0.2%  planar_flow < 0.2534 and pt_7 < 37.16
        + 0.001869714 * max(0.0, 506.875 - Q.sum_pt_top5) * max(0.0, Q.z_dr_0p1_0p2 - 0.04510668) / 8.800637   # +0.2%  sum_pt_top5 < 506.9 and z_dr_0p1_0p2 > 0.04511
        - 0.001502413 * max(0.0, Q.n_dr_0p1_0p2 - 3.0) / 0.3318622   # -0.2%  n_dr_0p1_0p2 > 3
        - 0.001097337 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, Q.sj3_pair_mass_min - 1.282345) / 0.4025762   # -0.1%  planar_flow < 0.2534 and sj3_pair_mass_min > 1.282
        - 0.0009523448 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, -0.001813533 - Q.mean_phi) / 4.860641e-05   # -0.1%  centroid_offset < 0.03776 and mean_phi < -0.001814
        + 0.0009186077 * max(0.0, 0.004372139 - Q.sum_z_dr2) * max(0.0, Q.sj3_dr13 - 0.1659434) / 5.846229e-06   # +0.1%  sum_z_dr2 < 0.004372 and sj3_dr13 > 0.1659
        - 0.0009077929 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 0.0345459 - Q.absphi_0) / 0.0003803849   # -0.1%  centroid_offset < 0.03776 and absphi_0 < 0.03455
        - 0.0008651708 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 4.721224 - Q.D2_b2) / 16.05793   # -0.1%  pt_6 < 41.22 and D2_b2 < 4.721
        + 0.0008100409 * max(0.0, Q.sj2_dr - 0.1778793) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.0283055   # +0.1%  sj2_dr > 0.1779 and n_pt_above_50 > 4
        + 0.0007067296 * max(0.0, 0.2623172 - Q.sj3_dr_max) * max(0.0, -0.0216713 - Q.phi_0) / 0.0003847124   # +0.1%  sj3_dr_max < 0.2623 and phi_0 < -0.02167
        - 0.0005248364 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # -0.1%  sj2_dr > 0.1779
        - 0.0005209359 * max(0.0, 19.45312 - Q.pt_6) / 0.250827   # -0.1%  pt_6 < 19.45
        - 0.0004763369 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.03601074 - Q.abseta_4) / 8.179649e-05   # -0.0%  centroid_offset < 0.01438 and abseta_4 < 0.03601
        - 0.000374698 * max(0.0, 0.02689598 - Q.sum_z_dr) * max(0.0, Q.sj3_pair_mass_min - 1.590484) / 0.005084766   # -0.0%  sum_z_dr < 0.0269 and sj3_pair_mass_min > 1.59
        - 0.0002106153 * max(0.0, Q.z_dr_0p05_0p1 - 0.8460335) / 0.01076887   # -0.0%  z_dr_0p05_0p1 > 0.846
        - 9.833596e-05 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.0%  sum_pt > 988.4
        - 9.602121e-05 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # -0.0%  centroid_offset > 0.0499 and D2 < 2.844
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.4505;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.4505295 * (-1.820719
        + 0.478197 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # +47.8%  sum_z_dr2 > 0.01883
        - 0.1582349 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -15.8%  e2 > 0.06344
        + 0.0807399 * max(0.0, Q.mass - 91.19) / 0.5839191   # +8.1%  mass > 91.19
        - 0.06379276 * max(0.0, Q.sum_z_dr2_top2 - 0.01403324) / 0.001129575   # -6.4%  sum_z_dr2_top2 > 0.01403
        + 0.05233967 * max(0.0, Q.mass - 69.61135) * max(0.0, Q.centroid_offset - 0.009480685) / 0.04081507   # +5.2%  mass > 69.61 and centroid_offset > 0.009481
        + 0.04716554 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +4.7%  centroid_offset > 0.0499
        - 0.04164377 * max(0.0, Q.zdr_0 - 0.03981924) / 0.0005695785   # -4.2%  zdr_0 > 0.03982
        - 0.04156251 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, 53.4375 - Q.pt_7) / 0.01439677   # -4.2%  sum_z_dr2 > 0.01883 and pt_7 < 53.44
        + 0.03632397 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, Q.lam2 - 0.000537286) / 2.828867e-06   # +3.6%  sum_z_dr2 > 0.01883 and lam2 > 0.0005373
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 17.14;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.13593 * (0.04029081
        + 0.3205368 * max(0.0, 0.1484084 - Q.sum_z_dr) / 0.0905999   # +32.1%  sum_z_dr < 0.1484
        + 0.1591352 * max(0.0, 0.01167645 - Q.lam1) / 0.007037801   # +15.9%  lam1 < 0.01168
        - 0.1262303 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -12.6%  e2 < 0.08001
        - 0.08558376 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -8.6%  sum_z_dr < 0.1484 and log_sum_pt < 6.804
        - 0.06840493 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 8.324263e-07   # -6.8%  e3 < 5.335e-05 and centroid_offset < 0.03776
        + 0.0399092 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # +4.0%  e3 < 5.335e-05
        - 0.03913077 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -3.9%  sum_z_dr < 0.1484 and pt_7 < 38.53
        - 0.02749587 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.000537286 - Q.lam2) / 4.021081e-05   # -2.7%  sum_z_dr < 0.1484 and lam2 < 0.0005373
        + 0.02675932 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7) / 781.5808   # +2.7%  sum_pt_top5 > 658.1 and pt_7 < 40.04
        - 0.01992837 * max(0.0, 0.006102347 - Q.lam1_plus_lam2) / 0.002547723   # -2.0%  lam1_plus_lam2 < 0.006102
        - 0.01727635 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7) / 0.06765006   # -1.7%  pt_6 < 31.91 and z_7 < 0.05861
        + 0.01066765 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 31.90625 - Q.pt_6) / 134.9952   # +1.1%  sum_pt_top5 > 791.1 and pt_6 < 31.91
        + 0.007146275 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.z_7 - 0.06164517) / 0.0003465887   # +0.7%  sum_z_dr < 0.1484 and z_7 > 0.06165
        + 0.005916385 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.07474969 - Q.M3) / 0.0006773733   # +0.6%  sum_z_dr < 0.1484 and M3 < 0.07475
        - 0.00521036 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # -0.5%  z_7 < 0.02807
        - 0.0051781 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # -0.5%  log_sum_pt > 6.503
        - 0.004938728 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # -0.5%  z_5 < 0.02818
        + 0.004612155 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.D3 - 0.2213841) / 3.794981e-05   # +0.5%  e3 < 5.335e-05 and D3 > 0.2214
        + 0.004002317 * max(0.0, 31.90625 - Q.pt_6) / 1.826854   # +0.4%  pt_6 < 31.91
        - 0.003505795 * max(0.0, Q.sj3_dr23 - 0.1974628) / 0.02478939   # -0.4%  sj3_dr23 > 0.1975
        + 0.002906271 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.tau2 - 0.008780509) / 0.000269945   # +0.3%  sum_z_dr < 0.1484 and tau2 > 0.008781
        - 0.00261031 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6) / 104.494   # -0.3%  sum_pt > 988.4 and pt_6 < 62.25
        - 0.002566076 * max(0.0, Q.mass - 49.6681) / 7.721594   # -0.3%  mass > 49.67
        - 0.00253656 * max(0.0, 839.6875 - Q.sum_pt_top5) / 254.87   # -0.3%  sum_pt_top5 < 839.7
        - 0.002486991 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.zdr_7 - 0.0007928864) / 0.0001818422   # -0.2%  log_sum_pt > 6.503 and zdr_7 > 0.0007929
        - 0.001769056 * max(0.0, Q.pt_7 - 48.71875) / 0.6759439   # -0.2%  pt_7 > 48.72
        - 0.001499883 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.1%  z_6 < 0.0216
        - 0.001348047 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.sj2_mass1 - 31.78116) / 0.01215768   # -0.1%  sum_z_dr < 0.1484 and sj2_mass1 > 31.78
        + 0.0007081517 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.pair_mass_0_7 - 10.2219) / 0.2907061   # +0.1%  log_sum_pt > 6.503 and pair_mass_0_7 > 10.22
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 28.28;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.27552 * (0.06147479
        - 0.212595 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -21.3%  sum_z_dr2 > 0.00752
        - 0.1620719 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -16.2%  sum_z_dr < 0.08724
        - 0.1186106 * max(0.0, Q.lam1_plus_lam2 - 0.01279734) / 0.001508311   # -11.9%  lam1_plus_lam2 > 0.0128
        + 0.09804943 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +9.8%  mass_over_sum_pt > 0.08475
        + 0.05877062 * max(0.0, 0.04107712 - Q.e2) / 0.01756014   # +5.9%  e2 < 0.04108
        + 0.04922271 * max(0.0, Q.sum_zz_dr2 - 0.01122239) / 0.001498065   # +4.9%  sum_zz_dr2 > 0.01122
        + 0.03430786 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # +3.4%  sj2_dr > 0.1592
        - 0.03375624 * max(0.0, 7.988762e-05 - Q.e3) / 5.499178e-05   # -3.4%  e3 < 7.989e-05
        - 0.03174122 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -3.2%  centroid_offset > 0.0499
        + 0.02673122 * max(0.0, Q.z_dr_0_0p05 - 0.1515405) / 0.4480651   # +2.7%  z_dr_0_0p05 > 0.1515
        + 0.02653219 * max(0.0, 44.81215 - Q.sd_mass) / 19.43433   # +2.7%  sd_mass < 44.81
        - 0.02346785 * max(0.0, 0.1441415 - Q.sd_rg) / 0.06067958   # -2.3%  sd_rg < 0.1441
        + 0.01760238 * max(0.0, 0.1483346 - Q.planar_flow) / 0.05106082   # +1.8%  planar_flow < 0.1483
        + 0.01388278 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # +1.4%  lam2 < 0.0005373
        - 0.0110656 * max(0.0, 0.004032342 - Q.C2_b2) / 0.002883542   # -1.1%  C2_b2 < 0.004032
        + 0.01095922 * max(0.0, 5.0 - Q.n_dr_0_0p05) / 1.94322   # +1.1%  n_dr_0_0p05 < 5
        - 0.01082895 * max(0.0, 715.4688 - Q.sum_pt) / 69.91196   # -1.1%  sum_pt < 715.5
        - 0.009046719 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -0.9%  planar_flow < 0.1115 and lam1 < 0.00733
        - 0.008097795 * max(0.0, 0.163898 - Q.z_dr_0p05_0p1) / 0.08682106   # -0.8%  z_dr_0p05_0p1 < 0.1639
        - 0.006217457 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.1058993 - Q.z_5) / 0.0007616364   # -0.6%  sj2_dr > 0.2002 and z_5 < 0.1059
        + 0.006061551 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +0.6%  LHA > 0.3033
        + 0.005982841 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.0008333816 - Q.C2_b2) / 1.341084e-07   # +0.6%  centroid_offset > 0.0499 and C2_b2 < 0.0008334
        + 0.003730973 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1629004) / 4.043869e-07   # +0.4%  e3 < 5.335e-05 and sj3_dr23 > 0.1629
        - 0.003367978 * max(0.0, Q.psi_0p1 - 0.9461839) / 0.02458256   # -0.3%  psi_0p1 > 0.9462
        - 0.003171029 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.3%  planar_flow < 0.1115 and centroid_offset < 0.01838
        - 0.002052115 * max(0.0, Q.mass - 69.61135) / 2.406702   # -0.2%  mass > 69.61
        - 0.001854759 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 0.02415398 - Q.C2_b2) / 0.0005249308   # -0.2%  z_dr_0p05_0p1 > 0.7509 and C2_b2 < 0.02415
        - 0.001787239 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 658.125 - Q.sum_pt_top5) / 2.950014   # -0.2%  planar_flow < 0.1115 and sum_pt_top5 < 658.1
        - 0.001589472 * max(0.0, Q.centroid_offset - 0.02685622) * max(0.0, 150.625 - Q.pt_1) / 0.1837865   # -0.2%  centroid_offset > 0.02686 and pt_1 < 150.6
        - 0.001298656 * max(0.0, 0.07870506 - Q.z_dr_0p1_0p2) / 0.04317253   # -0.1%  z_dr_0p1_0p2 < 0.07871
        + 0.001084067 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 3.351526e-05   # +0.1%  planar_flow < 0.1115 and lam1_plus_lam2 < 0.006097
        - 0.001082086 * max(0.0, Q.sj2_dr - 0.1591713) * max(0.0, 0.03293672 - Q.dr_2) / 6.256053e-05   # -0.1%  sj2_dr > 0.1592 and dr_2 < 0.03294
        + 0.0009994646 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067297   # +0.1%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        - 0.0007955666 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, Q.mean_eta - -0.006779839) / 0.000320267   # -0.1%  planar_flow < 0.1115 and mean_eta > -0.00678
        - 0.0005441489 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) * max(0.0, 715.4688 - Q.sum_pt) / 22.02426   # -0.1%  z_dr_0p05_0p1 < 0.5883 and sum_pt < 715.5
        + 0.0004890436 * max(0.0, Q.mass_top5 - 49.18618) / 2.747087   # +0.0%  mass_top5 > 49.19
        - 0.0003953356 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.08499387 - Q.D2_b2) / 0.0003170016   # -0.0%  sj2_dr > 0.2002 and D2_b2 < 0.08499
        - 0.0001559559 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, Q.zdr_5 - 0.0155217) / 1.782544e-06   # -0.0%  z_dr_0p05_0p1 > 0.7509 and zdr_5 > 0.01552
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 34.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.0662 * (-0.007243157
        + 0.2405768 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +24.1%  lam1_plus_lam2 < 0.01324
        - 0.1602015 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -16.0%  sum_zz_dr2 < 0.008169
        - 0.133879 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -13.4%  sum_z_dr < 0.1019
        + 0.1191723 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +11.9%  e2 < 0.04111
        - 0.07644879 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # -7.6%  mass_over_sum_pt < 0.1309
        - 0.07497763 * max(0.0, 0.007520088 - Q.sum_z_dr2) / 0.00354499   # -7.5%  sum_z_dr2 < 0.00752
        - 0.07399491 * max(0.0, 0.0005124533 - Q.sum_z_dr2_top2) / 0.0001104529   # -7.4%  sum_z_dr2_top2 < 0.0005125
        + 0.06134802 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +6.1%  mass_over_sum_pt_sq < 0.007183
        + 0.01051002 * max(0.0, 0.09317241 - Q.sj2_dr) / 0.02197556   # +1.1%  sj2_dr < 0.09317
        + 0.009247305 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +0.9%  lam1 < 0.00484
        + 0.006310362 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +0.6%  N2 < 0.2233 and sum_pt_top5 > 430.8
        - 0.004927356 * max(0.0, 0.007520088 - Q.sum_z_dr2) * max(0.0, 0.7459513 - Q.D2) / 9.127003e-05   # -0.5%  sum_z_dr2 < 0.00752 and D2 < 0.746
        - 0.004648472 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # -0.5%  sum_z_dr2_top3 < 0.002152
        + 0.00443337 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.1094252   # +0.4%  N2 < 0.2233 and n_dr_0p1_0p2 < 4
        - 0.004153667 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -0.4%  lam2 < 0.001131
        + 0.004029947 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2) / 0.003993721   # +0.4%  mass_over_sum_pt < 0.1309 and D2 < 0.746
        - 0.003561716 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.0114178   # -0.4%  N2 < 0.2233 and z_dr_0p05_0p1 < 0.5883
        - 0.003364832 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # -0.3%  N2 < 0.2233
        - 0.002743791 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.121681 - Q.max_dr) / 0.0004905272   # -0.3%  N2 < 0.2233 and max_dr < 0.1217
        + 0.0006994155 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.4684459) / 0.0001955483   # +0.1%  mass_over_sum_pt < 0.1309 and z_dr_0p1_0p2 > 0.4684
        - 0.0004479578 * max(0.0, Q.sj3_pair_mass_max - 80.4) / 0.3254107   # -0.0%  sj3_pair_mass_max > 80.4
        - 0.0003228593 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 1.54073 - Q.pt_entropy) / 0.000718138   # -0.0%  N2 < 0.2233 and pt_entropy < 1.541
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.969380462184874, 0.7984905462184874, 2.0245806722689075, 1.2431867647058823, 1.353516806722689, 1.838035294117647, 1.0262235294117648, 1.849609243697479, 0.2721046218487395, 3.1573050420168065, 2.1085181722689077, 1.8713621848739497, 0.10511470588235294, 3.55569012605042, 0.40531449579831935, 0.29117394957983195]
T = [2.375147094931722, 1.4417572117909663, 3.327697802324055, 2.676011283810399, 3.060314564732143]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +37%, n9 +23%, n5 -15%, n1 +13%, n0 -6%, n6 +5% ...
            + 0.3662666 * h[2] / H_AVG[2]
            + 0.228475 * h[9] / H_AVG[9]
            - 0.1450991 * h[5] / H_AVG[5]
            + 0.1313225 * h[1] / H_AVG[1]
            - 0.06377108 * h[0] / H_AVG[0]
            + 0.04725737 * h[6] / H_AVG[6]
            - 0.01780833 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -18%, n6 +9%, n4 -9%, n5 +6%, n15 +1% ...
            + 0.5560295 * h[9] / H_AVG[9]
            - 0.182808 * h[10] / H_AVG[10]
            + 0.08897333 * h[6] / H_AVG[6]
            - 0.08801218 * h[4] / H_AVG[4]
            + 0.05975896 * h[5] / H_AVG[5]
            + 0.01262236 * h[15] / H_AVG[15]
            + 0.0117957 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +21%, n3 -19%, n7 +12%, n0 +10%, n6 -10%, n14 -9% ...
            + 0.2108848 * h[11] / H_AVG[11]
            - 0.1867938 * h[3] / H_AVG[3]
            + 0.1215862 * h[7] / H_AVG[7]
            + 0.1001367 * h[0] / H_AVG[0]
            - 0.09637139 * h[6] / H_AVG[6]
            - 0.0913502 * h[14] / H_AVG[14]
            + 0.07512986 * h[13] / H_AVG[13]
            - 0.06015633 * h[15] / H_AVG[15]
            - 0.02964986 * h[9] / H_AVG[9]
            - 0.02044241 * h[8] / H_AVG[8]
            - 0.007498526 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +32%, n3 -26%, n6 -14%, n13 +7%, n14 +6%, n4 +4% ...
            + 0.3239913 * h[7] / H_AVG[7]
            - 0.261319 * h[3] / H_AVG[3]
            - 0.1438087 * h[6] / H_AVG[6]
            + 0.07266479 * h[13] / H_AVG[13]
            + 0.05679832 * h[14] / H_AVG[14]
            + 0.03951534 * h[4] / H_AVG[4]
            + 0.03729854 * h[1] / H_AVG[1]
            - 0.03687047 * h[9] / H_AVG[9]
            - 0.0170014 * h[15] / H_AVG[15]
            + 0.01073213 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +26%, n5 -15%, n4 +6%, n3 +3%, n12 -2% ...
            - 0.47201 * h[13] / H_AVG[13]
            + 0.2583703 * h[10] / H_AVG[10]
            - 0.1501508 * h[5] / H_AVG[5]
            + 0.05528504 * h[4] / H_AVG[4]
            + 0.02538928 * h[3] / H_AVG[3]
            - 0.01717384 * h[12] / H_AVG[12]
            + 0.01667136 * h[8] / H_AVG[8]
            + 0.004949351 * h[0] / H_AVG[0]
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
