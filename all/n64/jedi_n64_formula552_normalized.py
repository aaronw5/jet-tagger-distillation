"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; all observables), as if-statements with NORMALIZED weights (how much each one matters).

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.
  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1
             avg_k = the term's average size on the entire training set (term / avg is 1 on average), so share_k is the
             fraction of the neuron's average input that comes from that if-statement (sign: pushes it up / down).
  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1
             h_j = neuron j (after max(0, .) and the network's rounding), avg_j = its average on the training jets.

How much each neuron matters (share of all class scores, averaged over the training jets):
  neuron  8:  13.5%   (on for 52% of jets)
  neuron  1:  12.8%   (on for 95% of jets)
  neuron  5:  11.9%   (on for 79% of jets)
  neuron  4:  10.1%   (on for 65% of jets)
  neuron 13:   7.6%   (on for 89% of jets)
  neuron  0:   7.2%   (on for 61% of jets)
  neuron  9:   7.0%   (on for 57% of jets)
  neuron 10:   6.1%   (on for 75% of jets)
  neuron 12:   4.6%   (on for 46% of jets)
  neuron  6:   3.9%   (on for 51% of jets)
  neuron  3:   3.7%   (on for 61% of jets)
  neuron  7:   3.6%   (on for 28% of jets)
  neuron 11:   3.5%   (on for 70% of jets)
  neuron 15:   2.2%   (on for 57% of jets)
  neuron 14:   1.9%   (on for 36% of jets)
  neuron  2:   0.6%   (on for 48% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.3% (the network: 81.1%); same class as the network for 93.9% of jets.

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
  Q.z_10                   pT of particle 10 / total pT
  Q.z_11                   pT of particle 11 / total pT
  Q.z_2                    pT of particle 2 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_4               |Δη| of particle 4
  Q.absphi_0               |Δφ| of particle 0
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr1_12                 ΔR between particle 12 and the 2nd-hardest particle
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
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
        z_10=z[10],
        z_11=z[11],
        z_2=z[2],
        z_3=z[3],
        z_6=z[6],
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top2_slots=sum(pt[:2]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_5=z[5] * dr[5],
        z_2nd=zs[1],
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_4=abs(eta[4]),
        absphi_0=abs(phi[0]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        dr1_12=math.sqrt(dist2(1, 12)) if pt[12] > 0 else 0.0,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
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
        tau4=tau_n(4),
    )


def neuron_0(Q):
    # scale S = 12.49;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.49278 * (0.09835108
        - 0.1978328 * max(0.0, Q.mass - 78.26182) / 21.33658   # -19.8%  mass > 78.26
        - 0.1502864 * max(0.0, Q.mass - 92.85979) / 14.48273   # -15.0%  mass > 92.86
        + 0.07221495 * max(0.0, Q.mass - 91.19) / 15.01545   # +7.2%  mass > 91.19
        - 0.06293977 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -6.3%  mass < 101
        - 0.0573129 * max(0.0, Q.mass - 74.25181) / 24.07648   # -5.7%  mass > 74.25
        + 0.05700997 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq) / 0.001689525   # +5.7%  mass_over_sum_pt_sq < 0.006939
        + 0.05347702 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 0.002241068   # +5.3%  mass_over_sum_pt_sq < 0.007873
        - 0.04100289 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -4.1%  e2_sq < 0.006167
        + 0.03241539 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +3.2%  sum_pt_top50 < 1157
        + 0.0258901 * max(0.0, Q.mass_top50 - 71.79516) / 24.22501   # +2.6%  mass_top50 > 71.8
        + 0.0242259 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +2.4%  n_dr_0p2_0p4 < 15
        - 0.02388937 * max(0.0, Q.mass - 89.74183) / 15.55572   # -2.4%  mass > 89.74
        - 0.02251384 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -2.3%  girth2_top30 < 0.006364
        - 0.02107987 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.1%  tau1 < 0.07057
        - 0.02069962 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # -2.1%  sum_pt_top40 < 1070
        - 0.02050026 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # -2.1%  girth2_top20 < 0.006374
        - 0.01634267 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # -1.6%  z_dr_0p2_0p4 < 0.09123
        - 0.0146712 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -1.5%  sum_pt < 1013
        + 0.01445005 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +1.4%  log_sum_pt < 7.017
        + 0.01405997 * max(0.0, 80.4 - Q.mass_top30) / 14.65577   # +1.4%  mass_top30 < 80.4
        + 0.01107799 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +1.1%  mass_top40 < 80.89
        + 0.01030321 * max(0.0, 0.005312783 - Q.girth2_top20) / 0.001437214   # +1.0%  girth2_top20 < 0.005313
        - 0.007752653 * max(0.0, 0.9906378 - Q.z_top50_slots) / 0.004783315   # -0.8%  z_top50_slots < 0.9906
        + 0.007728209 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +0.8%  psi_0p3 > 0.9956
        - 0.007458646 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -0.7%  lam1 < 0.005914
        + 0.007198723 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957) / 1.852228   # +0.7%  log_sum_pt < 6.989 and sum_pt_top50 > 959.1
        + 0.003633658 * max(0.0, 846.1934 - Q.sum_pt_top20) / 22.83716   # +0.4%  sum_pt_top20 < 846.2
        + 0.002031974 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.0223982 - Q.C3) / 0.2900897   # +0.2%  sum_pt < 1013 and C3 < 0.0224
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 13.92;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.91693 * (0.02388069
        + 0.1272191 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # +12.7%  log_sum_pt > 6.894
        - 0.1113322 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -11.1%  log_sum_pt < 7.139
        + 0.08911502 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +8.9%  n_particles > 38
        - 0.05781724 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -5.8%  sum_pt_top50 > 959.1
        + 0.05732345 * max(0.0, 0.02412652 - Q.girth2_top30) / 0.01613825   # +5.7%  girth2_top30 < 0.02413
        + 0.05643965 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +5.6%  log_sum_pt > 6.91
        + 0.04989789 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.2734   # +5.0%  sum_pt_top2 < 689.2
        - 0.04174942 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt) / 15.05788   # -4.2%  n_particles > 38 and soft1_pt < 2.275
        + 0.03580432 * max(0.0, 1.521582 - Q.soft1_pt) / 0.9527349   # +3.6%  soft1_pt < 1.522
        - 0.03520959 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -3.5%  log_sum_pt > 6.959
        + 0.02950341 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # +3.0%  sum_pt_top40 < 1070
        - 0.02667287 * max(0.0, 120.6 - Q.mass) / 38.8279   # -2.7%  mass < 120.6
        - 0.02035219 * max(0.0, 31.35938 - Q.pt_9) / 6.538304   # -2.0%  pt_9 < 31.36
        - 0.0202952 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2) / 190.5117   # -2.0%  sj3_mass1 < 32.5 and sj3_mass2 < 18.68
        - 0.0190435 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -1.9%  log_sum_pt > 6.989
        - 0.01723883 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # -1.7%  M3 < 0.03188
        + 0.0158545 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +1.6%  sum_pt < 1017
        + 0.01349603 * max(0.0, Q.pt_entropy - 2.07371) / 0.8049786   # +1.3%  pt_entropy > 2.074
        - 0.01323706 * max(0.0, 47.88842 - Q.mass_top20) / 6.040308   # -1.3%  mass_top20 < 47.89
        + 0.01190457 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0) / 0.570275   # +1.2%  n_particles > 38 and dr_0 < 0.112
        - 0.01185468 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -1.2%  n_dr_0p2_0p4 < 7
        + 0.01082199 * max(0.0, 0.0005522528 - Q.girth2_top3) / 0.000118253   # +1.1%  girth2_top3 < 0.0005523
        + 0.01056223 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 39.23115   # +1.1%  mass_top20 < 47.89 and n_real_top40 > 29
        + 0.01035272 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2) / 0.0008500795   # +1.0%  z_top30_slots > 0.9342 and C2 < 0.0728
        + 0.01033497 * max(0.0, Q.z_top20_slots - 0.8965411) / 0.0341307   # +1.0%  z_top20_slots > 0.8965
        - 0.01022907 * max(0.0, 30.26161 - Q.sj2_mass1) / 7.997784   # -1.0%  sj2_mass1 < 30.26
        - 0.01014605 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # -1.0%  sj3_mass1 < 32.5
        + 0.009715464 * max(0.0, 0.003270031 - Q.girth2_top15) / 0.0007969965   # +1.0%  girth2_top15 < 0.00327
        + 0.009202934 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.9%  lam1 < 0.004673
        + 0.009148151 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1) / 1.005598   # +0.9%  n_particles > 38 and dr_1 < 0.1612
        + 0.008014535 * max(0.0, 0.1416054 - Q.D3) / 0.05511785   # +0.8%  D3 < 0.1416
        + 0.006286262 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793) / 0.1941151   # +0.6%  z_top30_slots > 0.9342 and max_pair_mass > 13.05
        - 0.005010606 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -0.5%  z_dr_0_0p05 > 0.8789
        - 0.004738613 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.5%  psi_0p3 > 0.998
        + 0.004500085 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 11.29505 - Q.pair_mass_0_13) / 38.90723   # +0.5%  pt_9 < 31.36 and pair_mass_0_13 < 11.3
        + 0.004278027 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 0.3241858 - Q.dr1_12) / 1.371943   # +0.4%  pt_9 < 31.36 and dr1_12 < 0.3242
        - 0.004157875 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 0.02980347 - Q.eta_0) / 5.83951   # -0.4%  sum_pt_top5 > 430.8 and eta_0 < 0.0298
        + 0.00385619 * max(0.0, 12.0 - Q.n_dr_0_0p05) / 3.466761   # +0.4%  n_dr_0_0p05 < 12
        + 0.002914857 * max(0.0, 42.41192 - Q.mass_top30) / 2.481216   # +0.3%  mass_top30 < 42.41
        + 0.002777017 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874) / 0.05244257   # +0.3%  z_top30_slots > 0.9342 and ptdr0_3 > 7.408
        + 0.001591693 * max(0.0, Q.sum_pt_top30 - 1191.938) / 6.938323   # +0.2%  sum_pt_top30 > 1192
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 6.895;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.89461 * (0.04799338
        - 0.1491663 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -14.9%  mass < 92.86
        + 0.09202341 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15) / 0.0009227558   # +9.2%  log_sum_pt > 6.903 and girth2_top15 < 0.02147
        - 0.08700148 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # -8.7%  sum_pt < 1261
        + 0.08677588 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # +8.7%  sum_pt > 1017
        - 0.06705797 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15) / 1.045047   # -6.7%  sum_pt_top50 > 988.5 and girth2_top15 < 0.02147
        + 0.05452569 * max(0.0, 91.19 - Q.mass) / 16.46423   # +5.5%  mass < 91.19
        - 0.05184815 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -5.2%  log_sum_pt > 6.959
        + 0.05148311 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +5.1%  lam1 < 0.01174
        + 0.04439424 * max(0.0, 1048.098 - Q.sum_pt_top50) / 44.36839   # +4.4%  sum_pt_top50 < 1048
        + 0.03434598 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +3.4%  mass_over_sum_pt < 0.09795
        + 0.03338016 * max(0.0, 1041.263 - Q.sum_pt_top40) / 49.16087   # +3.3%  sum_pt_top40 < 1041
        + 0.02721915 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # +2.7%  mass_top50 < 92.17
        - 0.02643934 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # -2.6%  sum_pt_top50 < 1079
        - 0.02598766 * max(0.0, 996.8867 - Q.sum_pt_top30) / 42.94346   # -2.6%  sum_pt_top30 < 996.9
        - 0.02189005 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -2.2%  girth2_top30 < 0.006364
        - 0.02159682 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -2.2%  sum_pt < 972
        + 0.01932537 * max(0.0, 0.004763596 - Q.girth2_top30) / 0.0009958273   # +1.9%  girth2_top30 < 0.004764
        + 0.01837098 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4) / 4.657698e-07   # +1.8%  sum_pt < 972 and e4 < 5.851e-08
        - 0.01828864 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # -1.8%  sum_pt > 1116
        + 0.01455482 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +1.5%  sum_pt_top40 > 1070
        - 0.01184657 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003463934   # -1.2%  log_sum_pt > 6.903 and psi_0p3 > 0.9299
        - 0.01087556 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 0.397021 - Q.max_dr) / 11.85447   # -1.1%  sum_pt < 1261 and max_dr < 0.397
        - 0.00991889 * max(0.0, Q.sum_pt_top40 - 984.7009) / 58.19535   # -1.0%  sum_pt_top40 > 984.7
        - 0.007679409 * max(0.0, Q.sum_pt_top50 - 988.4554) / 62.81258   # -0.8%  sum_pt_top50 > 988.5
        + 0.007077494 * max(0.0, Q.sum_pt_top50 - 1156.659) / 13.71464   # +0.7%  sum_pt_top50 > 1157
        - 0.006734941 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.7%  log_sum_pt > 7.063
        - 0.0001919386 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.08734131) / 0.003497383   # -0.0%  sum_pt_top20 > 1129 and eta_1 > 0.08734
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 3.098;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.097512 * (-0.0317981
        - 0.1513911 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # -15.1%  girth2 < 0.009615
        + 0.1452846 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +14.5%  girth2_top40 < 0.008841
        + 0.06585915 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +6.6%  n_dr_0p2_0p4 < 5
        + 0.05928007 * max(0.0, 0.08665515 - Q.mass_over_sum_pt) / 0.01542259   # +5.9%  mass_over_sum_pt < 0.08666
        + 0.05273604 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +5.3%  mass < 92.86
        + 0.05168678 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +5.2%  n_particles < 46
        - 0.04283982 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -4.3%  tau21 < 0.3862
        - 0.0403845 * max(0.0, 80.4 - Q.mass_top40) / 12.19371   # -4.0%  mass_top40 < 80.4
        + 0.03904191 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +3.9%  lam2 < 0.0006155
        - 0.03734896 * max(0.0, 86.4 - Q.mass) / 13.67632   # -3.7%  mass < 86.4
        + 0.03422881 * max(0.0, 27.56535 - Q.sj2_mass1) / 6.309897   # +3.4%  sj2_mass1 < 27.57
        - 0.02635352 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5) / 0.004484684   # -2.6%  n_dr_0p2_0p4 < 5 and zdr_5 < 0.007099
        - 0.02595662 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349) / 64.17204   # -2.6%  n_particles < 46 and mass_top15 > 57.87
        + 0.02373912 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1) / 0.1325294   # +2.4%  n_dr_0p2_0p4 < 5 and psi_0p1 < 0.9538
        + 0.02348154 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732) / 1477.074   # +2.3%  n_particles < 46 and sum_pt_top30 > 800.7
        + 0.02118704 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741) / 0.0491674   # +2.1%  n_dr_0p2_0p4 < 8 and z_top50_slots > 0.9787
        + 0.0209525 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +2.1%  tau4 < 0.01627
        - 0.01985791 * max(0.0, 79.21004 - Q.mass_top50) / 10.65754   # -2.0%  mass_top50 < 79.21
        - 0.01976791 * max(0.0, 2.410481 - Q.D2) / 0.587301   # -2.0%  D2 < 2.41
        - 0.01660191 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03447112 - Q.z_10) / 0.01282533   # -1.7%  n_dr_0p2_0p4 < 5 and z_10 < 0.03447
        - 0.01533735 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243) / 0.0001457263   # -1.5%  tau21 < 0.3862 and lam1 > 0.007671
        - 0.01338063 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0) / 3.186294   # -1.3%  n_dr_0p2_0p4 < 5 and n_dr_0p1_0p2 > 9
        + 0.01303405 * max(0.0, Q.psi_0p2 - 0.9985434) / 0.00014398   # +1.3%  psi_0p2 > 0.9985
        - 0.01297067 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -1.3%  mass < 53.87
        + 0.01274764 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421) / 0.000406438   # +1.3%  D2 < 2.41 and psi_0p3 > 0.9985
        - 0.008258748 * max(0.0, Q.psi_0p2 - 0.9985434) * max(0.0, 0.08575439 - Q.abseta_4) / 6.302321e-06   # -0.8%  psi_0p2 > 0.9985 and abseta_4 < 0.08575
        - 0.006291062 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 988.4375 - Q.sum_pt_top30) / 17.46027   # -0.6%  n_dr_0p2_0p4 < 5 and sum_pt_top30 < 988.4
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 10.83;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.83306 * (0.09225979
        + 0.140464 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +14.0%  mass < 101
        - 0.1066477 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -10.7%  mass_top40 < 163.3
        - 0.1028551 * max(0.0, 86.4 - Q.mass) / 13.67632   # -10.3%  mass < 86.4
        + 0.07615538 * max(0.0, 120.6 - Q.mass) / 38.8279   # +7.6%  mass < 120.6
        - 0.07393265 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -7.4%  mass < 92.86
        + 0.05150721 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +5.2%  n_dr_0p2_0p4 < 26
        - 0.04740525 * max(0.0, Q.n_particles - 22.0) / 23.9717   # -4.7%  n_particles > 22
        + 0.03955214 * max(0.0, Q.e2_sq - 0.009606007) / 0.003182984   # +4.0%  e2_sq > 0.009606
        - 0.0330328 * max(0.0, 78.26182 - Q.mass) / 9.857173   # -3.3%  mass < 78.26
        - 0.02958353 * max(0.0, Q.lam1 - 0.008241985) / 0.002595658   # -3.0%  lam1 > 0.008242
        + 0.02522182 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +2.5%  mass_top30 < 60.44
        - 0.0232875 * max(0.0, 0.004855289 - Q.girth2_top15) / 0.001418414   # -2.3%  girth2_top15 < 0.004855
        - 0.02289678 * max(0.0, 91.69753 - Q.mass_top30) / 22.0127   # -2.3%  mass_top30 < 91.7
        + 0.02227015 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +2.2%  mass_top40 < 67.73
        - 0.02123299 * max(0.0, Q.girth2_top15 - 0.00727763) / 0.002923446   # -2.1%  girth2_top15 > 0.007278
        + 0.02079188 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt) / 36.47388   # +2.1%  n_particles > 22 and soft1_pt < 2.275
        - 0.02042652 * max(0.0, 120.6 - Q.mass_top30) / 45.45988   # -2.0%  mass_top30 < 120.6
        - 0.01704029 * max(0.0, 91.19 - Q.mass_top15) / 34.88803   # -1.7%  mass_top15 < 91.19
        + 0.01558896 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +1.6%  psi_0p3 > 0.9974
        + 0.01544901 * max(0.0, 57.87349 - Q.mass_top15) / 12.46085   # +1.5%  mass_top15 < 57.87
        + 0.01235784 * max(0.0, Q.e2_sq - 0.01396296) / 0.002220692   # +1.2%  e2_sq > 0.01396
        + 0.01152557 * max(0.0, 0.9909875 - Q.psi_0p1) / 0.2210377   # +1.2%  psi_0p1 < 0.991
        + 0.01073778 * max(0.0, 143.7876 - Q.mass) / 57.99337   # +1.1%  mass < 143.8
        + 0.009386738 * max(0.0, Q.girth2_top15 - 0.01563836) / 0.001334556   # +0.9%  girth2_top15 > 0.01564
        - 0.008716547 * max(0.0, 80.78464 - Q.mass) / 10.84952   # -0.9%  mass < 80.78
        - 0.008199356 * max(0.0, 74.25181 - Q.mass) / 8.587064   # -0.8%  mass < 74.25
        - 0.008002203 * max(0.0, Q.girth - 0.1207452) / 0.004285542   # -0.8%  girth > 0.1207
        + 0.005870113 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.n_dr_0_0p05 - 10.0) / 143.1065   # +0.6%  n_particles > 22 and n_dr_0_0p05 > 10
        - 0.005356039 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 56.19962   # -0.5%  n_dr_0p2_0p4 < 26 and n_dr_0p1_0p2 > 11
        - 0.00441124 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 13.0 - Q.n_dr_0_0p05) / 0.003836025   # -0.4%  psi_0p3 > 0.9974 and n_dr_0_0p05 < 13
        - 0.003685044 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -0.4%  mass_top40 < 83.33
        - 0.003300962 * max(0.0, Q.lam2 - 0.001776308) / 0.0005876074   # -0.3%  lam2 > 0.001776
        + 0.003108996 * max(0.0, Q.sj3_pair_mass_min - 32.51366) / 5.88267   # +0.3%  sj3_pair_mass_min > 32.51
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 11.98;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.97545 * (0.05706665
        + 0.1689341 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +16.9%  sum_pt_top50 > 934.2
        - 0.0762085 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -7.6%  log_sum_pt > 6.91
        - 0.06204806 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # -6.2%  sum_pt > 986.1
        + 0.06042922 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +6.0%  sum_pt > 907.9
        + 0.05562656 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +5.6%  n_particles < 64
        - 0.04395376 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -4.4%  log_sum_pt > 6.92
        + 0.0400891 * max(0.0, Q.mass - 74.25181) / 24.07648   # +4.0%  mass > 74.25
        + 0.0361288 * max(0.0, Q.sd_mass - 69.65633) / 11.73272   # +3.6%  sd_mass > 69.66
        - 0.03474331 * max(0.0, Q.z_top30_slots - 0.9048492) / 0.05621641   # -3.5%  z_top30_slots > 0.9048
        - 0.03451555 * max(0.0, Q.mass_over_sum_pt - 0.09046749) / 0.01421039   # -3.5%  mass_over_sum_pt > 0.09047
        - 0.03100794 * max(0.0, 150.0144 - Q.mass_top40) / 66.86198   # -3.1%  mass_top40 < 150
        - 0.03058211 * max(0.0, 0.1937447 - Q.z_dr_0p2_0p4) / 0.146372   # -3.1%  z_dr_0p2_0p4 < 0.1937
        - 0.02714296 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -2.7%  mass_top50 > 157.5
        - 0.02543747 * max(0.0, Q.sd_mass - 86.4) / 6.275027   # -2.5%  sd_mass > 86.4
        - 0.0218654 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -2.2%  max_dr > 0.2405
        - 0.01932914 * max(0.0, 0.5494307 - Q.tau21) / 0.1591168   # -1.9%  tau21 < 0.5494
        - 0.01728843 * max(0.0, Q.mass - 91.19) / 15.01545   # -1.7%  mass > 91.19
        - 0.01656909 * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 9.82015   # -1.7%  n_dr_0p1_0p2 < 21
        + 0.016311 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +1.6%  sum_pt > 907.9 and e4 < 5.851e-08
        + 0.01613634 * max(0.0, Q.log_sum_pt - 6.935549) / 0.0371538   # +1.6%  log_sum_pt > 6.936
        - 0.015616 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2) / 10.81389   # -1.6%  n_particles < 64 and D2 < 2.179
        + 0.01415939 * max(0.0, Q.n_pt_above_1 - 28.0) / 14.80415   # +1.4%  n_pt_above_1 > 28
        + 0.01273381 * max(0.0, 0.01375115 - Q.z_11) / 0.0009450582   # +1.3%  z_11 < 0.01375
        - 0.01201647 * max(0.0, 14.14062 - Q.pt_11) / 0.9740783   # -1.2%  pt_11 < 14.14
        - 0.01190886 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -1.2%  sum_pt_top40 > 1025
        + 0.01075995 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # +1.1%  log_sum_pt > 6.989
        + 0.01075608 * max(0.0, 1.976207 - Q.D2) / 0.367701   # +1.1%  D2 < 1.976
        + 0.0105945 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +1.1%  n_dr_0p2_0p4 < 11
        + 0.01004527 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # +1.0%  e2 < 0.03876
        + 0.009035752 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +0.9%  mass_over_sum_pt > 0.1709
        - 0.008649431 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -0.9%  mass_over_sum_pt_sq > 0.0292
        + 0.007821653 * max(0.0, Q.sum_pt_top30 - 933.1875) / 79.88556   # +0.8%  sum_pt_top30 > 933.2
        + 0.007371213 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # +0.7%  n_dr_0p1_0p2 < 8
        - 0.007126579 * max(0.0, Q.mass_top40 - 120.6) / 5.823744   # -0.7%  mass_top40 > 120.6
        - 0.006023542 * max(0.0, Q.girth2_top50 - 0.01951641) / 0.001208668   # -0.6%  girth2_top50 > 0.01952
        + 0.003659462 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2) / 0.001543279   # +0.4%  psi_0p3 > 0.9974 and D2 < 3.345
        + 0.002289447 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # +0.2%  mass_top10 > 71.78
        - 0.00213336 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.2%  sum_pt_top10 > 943.7
        + 0.001748214 * max(0.0, Q.mass - 172.8) / 0.7291439   # +0.2%  mass > 172.8
        + 0.001204157 * max(0.0, Q.max_dr - 0.4357228) / 0.007711928   # +0.1%  max_dr > 0.4357
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 10.37;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.36639 * (0.1128759
        - 0.1716228 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -17.2%  mass < 101
        - 0.1393077 * max(0.0, 120.6 - Q.mass) / 38.8279   # -13.9%  mass < 120.6
        + 0.1222956 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +12.2%  mass < 92.86
        - 0.04902287 * max(0.0, 0.01807679 - Q.girth2_top30) / 0.01083719   # -4.9%  girth2_top30 < 0.01808
        + 0.04696042 * max(0.0, Q.girth2_top30 - 0.008376291) / 0.003092781   # +4.7%  girth2_top30 > 0.008376
        + 0.04482346 * max(0.0, 86.4 - Q.mass) / 13.67632   # +4.5%  mass < 86.4
        + 0.04439523 * max(0.0, 172.8 - Q.mass_top50) / 85.49388   # +4.4%  mass_top50 < 172.8
        - 0.03750825 * max(0.0, Q.mass_over_sum_pt - 0.09795415) / 0.01222897   # -3.8%  mass_over_sum_pt > 0.09795
        + 0.03674423 * max(0.0, 0.003687605 - Q.lam2) / 0.002601874   # +3.7%  lam2 < 0.003688
        + 0.03273865 * max(0.0, Q.girth2_top20 - 0.008031209) / 0.002906852   # +3.3%  girth2_top20 > 0.008031
        + 0.03106342 * max(0.0, 91.19 - Q.mass_top15) / 34.88803   # +3.1%  mass_top15 < 91.19
        - 0.02889922 * max(0.0, Q.e2 - 0.02793599) / 0.008664879   # -2.9%  e2 > 0.02794
        - 0.02463024 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2) / 0.04240098   # -2.5%  mass_over_sum_pt > 0.1182 and D2_b2 < 7.366
        - 0.02381045 * max(0.0, 0.004573744 - Q.girth2_top50) / 0.0007739713   # -2.4%  girth2_top50 < 0.004574
        - 0.02366881 * max(0.0, Q.girth2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529) / 8.566262e-05   # -2.4%  girth2_top20 > 0.008031 and C2_b2 > 0.0008188
        + 0.02237742 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +2.2%  tau1 < 0.06311
        + 0.0180124 * max(0.0, 0.003811746 - Q.lam1) / 0.0006796118   # +1.8%  lam1 < 0.003812
        + 0.0121788 * max(0.0, 0.00375223 - Q.girth2_top30) / 0.0006714094   # +1.2%  girth2_top30 < 0.003752
        + 0.01145746 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # +1.1%  mass_top50 < 71.8
        + 0.009179362 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832) / 8.191828e-05   # +0.9%  mass_over_sum_pt > 0.1182 and C2_b2 > 0.02705
        + 0.008514626 * max(0.0, Q.e2 - 0.05557149) / 0.001119731   # +0.9%  e2 > 0.05557
        - 0.008286945 * max(0.0, Q.n_dr_0_0p05 - 9.0) / 5.593776   # -0.8%  n_dr_0_0p05 > 9
        - 0.007645471 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722) / 0.001787472   # -0.8%  girth2_top30 > 0.008376 and D2_b2 > 1.677
        + 0.006066022 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt) / 683.4113   # +0.6%  mass < 120.6 and sum_pt < 1008
        - 0.004968848 * max(0.0, 0.002197765 - Q.girth2_top15) / 0.0004509993   # -0.5%  girth2_top15 < 0.002198
        - 0.004906055 * max(0.0, Q.girth2_top30 - 0.02809026) / 0.0001993235   # -0.5%  girth2_top30 > 0.02809
        - 0.004608573 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -0.5%  mass_over_sum_pt > 0.1182
        - 0.004510567 * max(0.0, Q.sj3_pair_mass_min - 29.00832) / 6.839984   # -0.5%  sj3_pair_mass_min > 29.01
        - 0.003907485 * max(0.0, Q.sj3_mass1 - 21.11128) / 1.638992   # -0.4%  sj3_mass1 > 21.11
        - 0.003876418 * max(0.0, Q.sj3_pair_mass_min - 76.60223) / 0.3297656   # -0.4%  sj3_pair_mass_min > 76.6
        + 0.003240447 * max(0.0, 0.9300465 - Q.z_top40_slots) / 0.002595351   # +0.3%  z_top40_slots < 0.93
        + 0.003180871 * max(0.0, Q.sj3_pair_mass_min - 29.00832) * max(0.0, Q.psi_0p3 - 0.9896594) / 0.01664863   # +0.3%  sj3_pair_mass_min > 29.01 and psi_0p3 > 0.9897
        + 0.002590785 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707) / 2.369749e-05   # +0.3%  e2 > 0.05557 and zdr_0 > 0.00137
        - 0.002031006 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.2%  log_sum_pt < 6.811
        - 0.0009691175 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, Q.max_pair_mass - 33.3761) / 3.25809   # -0.1%  n_dr_0p2_0p4 > 15 and max_pair_mass > 33.38
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 40.73;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 40.72794 * (-0.00584671
        + 0.1349642 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5337743   # +13.5%  mass < 91.19 and psi_0p3 > 0.9638
        - 0.1251708 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.3075308   # -12.5%  mass < 91.19 and psi_0p3 > 0.9777
        - 0.1195635 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5700006   # -12.0%  mass < 92.86 and psi_0p3 > 0.9638
        + 0.08938886 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.4385899   # +8.9%  mass < 101 and psi_0p3 > 0.9777
        - 0.05530858 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -5.5%  mass < 101
        - 0.04707524 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -4.7%  mass < 82.85
        + 0.03905568 * max(0.0, 120.6 - Q.mass) / 38.8279   # +3.9%  mass < 120.6
        + 0.03300363 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.01745475   # +3.3%  mass < 101 and psi_0p3 > 0.998
        - 0.02965659 * max(0.0, 0.008190222 - Q.girth2) / 0.002454223   # -3.0%  girth2 < 0.00819
        - 0.02580585 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.01143143   # -2.6%  mass < 91.19 and psi_0p3 > 0.998
        - 0.02522547 * max(0.0, 86.4 - Q.sd_mass) / 35.84633   # -2.5%  sd_mass < 86.4
        + 0.02463063 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.2231625   # +2.5%  mass < 82.85 and psi_0p3 > 0.9777
        - 0.02346912 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -2.3%  tau1 < 0.1073
        + 0.02071226 * max(0.0, 69.65633 - Q.sd_mass) / 24.56034   # +2.1%  sd_mass < 69.66
        - 0.01932374 * max(0.0, 0.00363788 - Q.e2_sq) / 0.0004963098   # -1.9%  e2_sq < 0.003638
        + 0.01806546 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # +1.8%  girth2 < 0.007877
        + 0.01777917 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +1.8%  mass_over_sum_pt < 0.1182
        + 0.01646451 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +1.6%  mass < 78.26
        + 0.0154348 * max(0.0, 91.19 - Q.mass) / 16.46423   # +1.5%  mass < 91.19
        + 0.01478634 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # +1.5%  girth2 < 0.009615
        - 0.01317157 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -1.3%  psi_0p3 > 0.998
        - 0.01303849 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -1.3%  lam1 < 0.008242
        + 0.01218022 * max(0.0, 0.09591084 - Q.tau1) / 0.02629125   # +1.2%  tau1 < 0.09591
        + 0.01046472 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +1.0%  tau21_b2 < 0.2352
        - 0.009938092 * max(0.0, Q.sd_mass - 98.05743) / 4.276978   # -1.0%  sd_mass > 98.06
        + 0.006855094 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.7%  psi_0p3 > 0.9974
        + 0.006245495 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # +0.6%  lam1 < 0.00619
        + 0.006068728 * max(0.0, Q.mass_top20 - 85.79457) / 7.06893   # +0.6%  mass_top20 > 85.79
        - 0.005178922 * max(0.0, Q.mass_top20 - 73.35236) / 11.28399   # -0.5%  mass_top20 > 73.35
        + 0.004084643 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +0.4%  mass < 92.86
        - 0.003878122 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt) / 11.66143   # -0.4%  tau21_b2 < 0.2352 and sum_pt < 1261
        + 0.002751017 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.3%  n_dr_0p2_0p4 < 6
        - 0.002206713 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -0.2%  tau21 < 0.3862
        - 0.001829466 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.4947602 - Q.z_dr_0_0p05) / 0.572942   # -0.2%  mass < 92.86 and z_dr_0_0p05 < 0.4948
        + 0.001622814 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.02947847   # +0.2%  mass < 82.85 and psi_0p3 < 0.9985
        - 0.001465865 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 5.106465e-05   # -0.1%  tau21_b2 < 0.2352 and mass_over_sum_pt_sq < 0.007873
        - 0.001236156 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.1063277 - Q.z_2) / 0.0009585458   # -0.1%  tau21_b2 < 0.2352 and z_2 < 0.1063
        - 0.001042818 * max(0.0, 0.006403325 - Q.girth2) / 0.001406912   # -0.1%  girth2 < 0.006403
        + 0.0007942322 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.007828415   # +0.1%  mass < 82.85 and psi_0p3 > 0.998
        - 0.0007744131 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -9.840088) / 1.453315   # -0.1%  tau21_b2 < 0.2352 and orientation_deg > -9.84
        + 0.0002878978 * max(0.0, 0.000404306 - Q.lam2) / 5.964336e-05   # +0.0%  lam2 < 0.0004043
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 16.13;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.12968 * (0.1281442
        - 0.09402242 * max(0.0, 0.009614971 - Q.width) / 0.003502545   # -9.4%  width < 0.009615
        + 0.06328838 * max(0.0, Q.mass - 74.25181) / 24.07648   # +6.3%  mass > 74.25
        - 0.05789974 * max(0.0, 39.0 - Q.n_for_90pct) / 18.29776   # -5.8%  n_for_90pct < 39
        - 0.0567285 * max(0.0, Q.n_for_90pct - 7.0) / 13.93571   # -5.7%  n_for_90pct > 7
        - 0.04984237 * max(0.0, Q.mass - 101.0497) / 12.31084   # -5.0%  mass > 101
        + 0.04790944 * max(0.0, Q.mass - 64.48544) / 31.19164   # +4.8%  mass > 64.49
        + 0.04295809 * max(0.0, Q.girth2_top20 - 0.008031209) / 0.002906852   # +4.3%  girth2_top20 > 0.008031
        - 0.0415702 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -4.2%  mass_top50 > 82.04
        + 0.0413524 * max(0.0, Q.mass - 87.36377) / 16.57741   # +4.1%  mass > 87.36
        + 0.03764816 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt) / 2.336698   # +3.8%  mass_over_sum_pt > 0.07697 and sum_pt < 1116
        + 0.03729149 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +3.7%  girth2_top40 < 0.008841
        - 0.03241759 * max(0.0, Q.girth2_top20 - 0.006043209) / 0.003593885   # -3.2%  girth2_top20 > 0.006043
        - 0.02841233 * max(0.0, Q.z_top50_slots - 0.9704436) / 0.02331682   # -2.8%  z_top50_slots > 0.9704
        + 0.0281983 * max(0.0, 1028.184 - Q.sum_pt) / 26.95739   # +2.8%  sum_pt < 1028
        + 0.0239874 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +2.4%  sum_pt < 1017
        - 0.02347326 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # -2.3%  sum_pt_top40 < 1025
        - 0.02343103 * max(0.0, 6.930088 - Q.log_sum_pt) / 0.02522479   # -2.3%  log_sum_pt < 6.93
        - 0.02334321 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # -2.3%  mass_over_sum_pt > 0.07697
        + 0.02196914 * max(0.0, 0.007463985 - Q.girth2_top30) / 0.00233708   # +2.2%  girth2_top30 < 0.007464
        - 0.01964762 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt) / 0.0005328922   # -2.0%  girth2_top40 > 0.005197 and log_sum_pt < 7.017
        + 0.01571902 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +1.6%  lam1 < 0.007671
        - 0.01509691 * max(0.0, Q.mass - 125.1) / 7.098976   # -1.5%  mass > 125.1
        + 0.01473055 * max(0.0, Q.girth2_top40 - 0.005196966) / 0.004725809   # +1.5%  girth2_top40 > 0.005197
        - 0.01404276 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -1.4%  n_dr_0p2_0p4 < 15
        + 0.01337405 * max(0.0, Q.mass_top50 - 117.0487) / 7.705579   # +1.3%  mass_top50 > 117
        + 0.01325796 * max(0.0, Q.n_dr_0p2_0p4 - 3.0) / 6.482203   # +1.3%  n_dr_0p2_0p4 > 3
        - 0.01216544 * max(0.0, Q.e3 - 5.13841e-05) / 6.821576e-05   # -1.2%  e3 > 5.138e-05
        - 0.01164524 * max(0.0, 0.02146578 - Q.girth2_top15) / 0.01470441   # -1.2%  girth2_top15 < 0.02147
        - 0.01109357 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -1.1%  mass_over_sum_pt > 0.1182
        + 0.00877844 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.9%  sj2_dr > 0.2232
        + 0.008128852 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.6773544   # +0.8%  sum_pt < 1028 and dr_max_012 > 0.1828
        + 0.008045196 * max(0.0, 9.0 - Q.n_dr_0p2_0p4) / 3.177318   # +0.8%  n_dr_0p2_0p4 < 9
        + 0.007749832 * max(0.0, 1001.523 - Q.sum_pt_top40) / 26.72886   # +0.8%  sum_pt_top40 < 1002
        - 0.007700235 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.5693287   # -0.8%  sum_pt < 1017 and dr_max_012 > 0.1828
        + 0.00696346 * max(0.0, Q.C2 - 0.06655881) / 0.01514001   # +0.7%  C2 > 0.06656
        + 0.006507826 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # +0.7%  e2 > 0.04755
        - 0.004107855 * max(0.0, Q.mass - 64.48544) * max(0.0, Q.zdr_0 - 0.0008718296) / 0.4297651   # -0.4%  mass > 64.49 and zdr_0 > 0.0008718
        + 0.003814005 * max(0.0, 0.8316924 - Q.z_top15_slots) / 0.04702674   # +0.4%  z_top15_slots < 0.8317
        + 0.003492801 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # +0.3%  girth2 < 0.007877
        + 0.00346161 * max(0.0, 0.0795038 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.38202) / 0.1361121   # +0.3%  tau2 < 0.0795 and sj3_mass1 > 13.38
        + 0.003108957 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477) / 0.03720487   # +0.3%  z_dr_0p1_0p2 > 0.334
        - 0.002922836 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, Q.sj2_dr - 0.1512157) / 6.78729e-05   # -0.3%  girth2_top30 < 0.007464 and sj2_dr > 0.1512
        - 0.002777478 * max(0.0, 0.3628388 - Q.psi_0p1) / 0.02326895   # -0.3%  psi_0p1 < 0.3628
        + 0.002222059 * max(0.0, Q.n_dr_0p1_0p2 - 19.0) / 1.732133   # +0.2%  n_dr_0p1_0p2 > 19
        + 0.001876291 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # +0.2%  e3 > 0.0003372
        - 0.001152114 * max(0.0, Q.mass_top20 - 119.2969) / 1.8402   # -0.1%  mass_top20 > 119.3
        - 0.0006735731 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, 13.0 - Q.n_for_90pct) / 15.66746   # -0.1%  sum_pt < 1017 and n_for_90pct < 13
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 43.75;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 43.75377 * (-0.02190886
        - 0.2598641 * max(0.0, 160.8 - Q.mass) / 72.78344   # -26.0%  mass < 160.8
        + 0.2209815 * max(0.0, 162.8363 - Q.mass) / 74.60496   # +22.1%  mass < 162.8
        + 0.1527798 * max(0.0, 172.8 - Q.mass) / 83.78792   # +15.3%  mass < 172.8
        - 0.128222 * max(0.0, 143.7876 - Q.mass) / 57.99337   # -12.8%  mass < 143.8
        + 0.06364249 * max(0.0, 138.8977 - Q.mass_top50) / 55.01559   # +6.4%  mass_top50 < 138.9
        - 0.04387141 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -4.4%  mass_top40 < 163.3
        + 0.02661142 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +2.7%  mass < 92.86
        - 0.01677487 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # -1.7%  mass_top50 < 92.17
        + 0.01483126 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +1.5%  mass < 82.85
        - 0.009053116 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -0.9%  mass < 64.49
        + 0.00843901 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +0.8%  mass_top40 < 80.89
        + 0.006557367 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # +0.7%  girth2_top40 < 0.00626
        + 0.006212501 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +0.6%  sum_pt < 986.1
        - 0.005147448 * max(0.0, Q.z_top40_slots - 0.9574183) / 0.02831517   # -0.5%  z_top40_slots > 0.9574
        - 0.005099769 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -0.5%  log_sum_pt < 6.903
        + 0.004098063 * max(0.0, Q.girth - 0.09749958) / 0.008315822   # +0.4%  girth > 0.0975
        - 0.004030291 * max(0.0, Q.girth2_top30 - 0.0008564881) / 0.007661497   # -0.4%  girth2_top30 > 0.0008565
        - 0.003863245 * max(0.0, Q.girth - 0.1402186) / 0.001871547   # -0.4%  girth > 0.1402
        + 0.003466959 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # +0.3%  psi_0p3 > 0.9943
        + 0.003369154 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +0.3%  LHA > 0.4042
        + 0.00279904 * max(0.0, 73.33139 - Q.mass_top30) / 11.37701   # +0.3%  mass_top30 < 73.33
        + 0.00245639 * max(0.0, Q.lam1 - 0.01174405) / 0.001827524   # +0.2%  lam1 > 0.01174
        - 0.002351781 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt) / 447.38   # -0.2%  mass_top40 < 80.89 and sum_pt < 1035
        - 0.001334782 * max(0.0, 0.00217213 - Q.girth2_top40) / 0.0002075259   # -0.1%  girth2_top40 < 0.002172
        + 0.000992261 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +0.1%  lam1 < 0.004673
        + 0.0009697162 * max(0.0, 956.2133 - Q.sum_pt_top40) / 14.00451   # +0.1%  sum_pt_top40 < 956.2
        + 0.0009559949 * max(0.0, 125.1 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 42.49581   # +0.1%  mass_top40 < 125.1 and n_dr_0p2_0p4 > 9
        + 0.000614276 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40) / 104.3982   # +0.1%  mass_top40 < 80.89 and sum_pt_top40 < 906.6
        + 0.0003548364 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258) / 5.834847   # +0.0%  sum_pt_top40 < 956.2 and soft4_pt > 1.789
        - 0.0002550335 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047) / 1.704178   # -0.0%  sum_pt_top40 < 956.2 and soft3_pt > 2.873
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 11.95;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.95398 * (-0.002239723
        + 0.1108125 * max(0.0, Q.mass - 78.26182) / 21.33658   # +11.1%  mass > 78.26
        - 0.09023315 * max(0.0, Q.mass - 64.48544) / 31.19164   # -9.0%  mass > 64.49
        + 0.08777913 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +8.8%  mass < 89.74
        - 0.0758015 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -7.6%  mass < 101
        + 0.06296715 * max(0.0, Q.e2 - 0.01256572) / 0.01912291   # +6.3%  e2 > 0.01257
        + 0.0615706 * max(0.0, 0.01655983 - Q.girth2_top20) / 0.01003486   # +6.2%  girth2_top20 < 0.01656
        - 0.03945521 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -3.9%  e2 < 0.04359
        + 0.03468022 * max(0.0, 0.5760704 - Q.z_top2_slots) / 0.2302819   # +3.5%  z_top2_slots < 0.5761
        - 0.03375933 * max(0.0, Q.mass - 143.7876) / 3.946979   # -3.4%  mass > 143.8
        + 0.02716794 * max(0.0, 2.975532 - Q.D2) / 0.9290697   # +2.7%  D2 < 2.976
        - 0.02667164 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -2.7%  girth < 0.1207
        - 0.02555358 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # -2.6%  girth2_top30 < 0.005402
        - 0.02380231 * max(0.0, 65.20727 - Q.sj2_mass1) / 36.50254   # -2.4%  sj2_mass1 < 65.21
        + 0.02164753 * max(0.0, 86.4 - Q.mass) / 13.67632   # +2.2%  mass < 86.4
        - 0.02150603 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -2.2%  n_dr_0p2_0p4 < 11
        + 0.02043204 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +2.0%  mass_top50 > 136.8
        + 0.02030299 * max(0.0, 935.1043 - Q.sum_pt_top15) / 95.709   # +2.0%  sum_pt_top15 < 935.1
        + 0.01715517 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +1.7%  z_dr_0p2_0p4 < 0.09123
        - 0.01645626 * max(0.0, Q.mass_top30 - 78.53034) / 14.31024   # -1.6%  mass_top30 > 78.53
        + 0.0153781 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +1.5%  dr_0 < 0.06413
        - 0.01511569 * max(0.0, Q.z_dr_0_0p05 - 0.7674734) / 0.05221088   # -1.5%  z_dr_0_0p05 > 0.7675
        - 0.0147177 * max(0.0, 59.40777 - Q.mass_top5) / 33.74882   # -1.5%  mass_top5 < 59.41
        + 0.01274378 * max(0.0, 59.40777 - Q.mass_top5) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1) / 22.61156   # +1.3%  mass_top5 < 59.41 and z_dr_0p05_0p1 < 0.8509
        - 0.01244894 * max(0.0, 0.002575211 - Q.girth2) / 0.0002582872   # -1.2%  girth2 < 0.002575
        - 0.01211096 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.9985421) / 6.34706e-06   # -1.2%  girth2_top10 < 0.01977 and psi_0p3 > 0.9985
        + 0.01177071 * max(0.0, Q.mass_top5 - 22.18342) / 12.67741   # +1.2%  mass_top5 > 22.18
        - 0.01175744 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, 0.4357228 - Q.max_dr) / 0.001259784   # -1.2%  girth2_top10 < 0.01977 and max_dr < 0.4357
        + 0.01166052 * max(0.0, 0.0001086251 - Q.e3) / 5.924919e-05   # +1.2%  e3 < 0.0001086
        - 0.007806103 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # -0.8%  sum_pt_top50 < 959.1
        + 0.00726617 * max(0.0, 0.008824206 - Q.zdr_1) / 0.00383206   # +0.7%  zdr_1 < 0.008824
        + 0.006614132 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, Q.psi_0p3 - 0.9985421) / 4.280363e-07   # +0.7%  girth2_top30 < 0.005402 and psi_0p3 > 0.9985
        - 0.006529246 * max(0.0, Q.mass - 162.8363) / 1.509832   # -0.7%  mass > 162.8
        - 0.005933613 * max(0.0, Q.mass_over_sum_pt - 0.1606361) / 0.001344467   # -0.6%  mass_over_sum_pt > 0.1606
        - 0.00518638 * max(0.0, Q.e2 - 0.03263075) / 0.006288927   # -0.5%  e2 > 0.03263
        - 0.005085523 * max(0.0, 2.975532 - Q.D2) * max(0.0, Q.sj2_dr - 0.2070855) / 0.0198522   # -0.5%  D2 < 2.976 and sj2_dr > 0.2071
        - 0.005 * max(0.0, 72.18744 - Q.mass_top15) / 20.20734   # -0.5%  mass_top15 < 72.19
        - 0.002951866 * max(0.0, Q.sum_pt_top10 - 943.6922) / 12.05717   # -0.3%  sum_pt_top10 > 943.7
        + 0.002512754 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, 631.275 - Q.sum_pt_top5) / 0.06578377   # +0.3%  girth2_top30 < 0.005402 and sum_pt_top5 < 631.3
        + 0.002347154 * max(0.0, Q.mass_over_sum_pt - 0.1606361) * max(0.0, 0.09696199 - Q.z_3) / 5.287414e-05   # +0.2%  mass_over_sum_pt > 0.1606 and z_3 < 0.09696
        + 0.002177165 * max(0.0, Q.mass_top50 - 160.8) / 1.246461   # +0.2%  mass_top50 > 160.8
        + 0.002008984 * max(0.0, Q.psi_0p3 - 0.9985421) * max(0.0, Q.soft5_z - 0.001434897) / 3.081682e-07   # +0.2%  psi_0p3 > 0.9985 and soft5_z > 0.001435
        - 0.001427691 * max(0.0, Q.mass_top50 - 172.8) / 0.5062389   # -0.1%  mass_top50 > 172.8
        - 0.0009944104 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897) / 0.001013644   # -0.1%  mass > 162.8 and soft5_z > 0.001435
        - 0.0007006869 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109) / 0.0004735247   # -0.1%  mass_top50 > 160.8 and soft4_z > 0.001721
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 9.643;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.642958 * (-0.05040621
        + 0.14223 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +14.2%  mass < 92.86
        - 0.08361836 * max(0.0, 0.076787 - Q.girth) / 0.02105267   # -8.4%  girth < 0.07679
        - 0.07854542 * max(0.0, 80.4 - Q.mass) / 10.6806   # -7.9%  mass < 80.4
        + 0.06044309 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +6.0%  mass < 101
        + 0.05615023 * max(0.0, 0.08589404 - Q.girth) / 0.02745918   # +5.6%  girth < 0.08589
        + 0.05344179 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +5.3%  e2_sq < 0.008184
        + 0.05267993 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +5.3%  e2 < 0.0303
        - 0.04688895 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -4.7%  lam1 < 0.006717
        + 0.03866221 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +3.9%  psi_0p3 > 0.9897
        - 0.03295944 * max(0.0, 0.006929741 - Q.girth2_top30) / 0.002004687   # -3.3%  girth2_top30 < 0.00693
        - 0.0317717 * max(0.0, 0.00625621 - Q.girth2_top10) / 0.002474789   # -3.2%  girth2_top10 < 0.006256
        + 0.03128983 * max(0.0, 0.00406126 - Q.girth2_top10) / 0.001298509   # +3.1%  girth2_top10 < 0.004061
        + 0.03116506 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +3.1%  n_dr_0p2_0p4 < 10
        - 0.02972411 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # -3.0%  psi_0p2 > 0.9314
        - 0.02884119 * max(0.0, 80.35535 - Q.mass_top50) / 11.1399   # -2.9%  mass_top50 < 80.36
        - 0.02846383 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # -2.8%  e2 < 0.03876
        + 0.02729814 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +2.7%  e2 < 0.02516
        - 0.02256591 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -2.3%  mass_top40 < 83.33
        - 0.01806209 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2) / 0.005195774   # -1.8%  n_dr_0p2_0p4 < 10 and girth2 < 0.005532
        - 0.01783451 * max(0.0, 3.793233e-05 - Q.e3) / 1.004564e-05   # -1.8%  e3 < 3.793e-05
        + 0.01449228 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 43.23581   # +1.4%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21
        + 0.01380214 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +1.4%  mass_top30 < 60.44
        + 0.01366339 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # +1.4%  girth < 0.07374
        - 0.007922462 * max(0.0, 101.0497 - Q.mass) * max(0.0, 891.875 - Q.sum_pt_top10) / 2090.674   # -0.8%  mass < 101 and sum_pt_top10 < 891.9
        - 0.007254266 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0) / 30.16986   # -0.7%  n_dr_0p2_0p4 < 10 and n_particles > 34
        + 0.006378041 * max(0.0, Q.psi_0p2 - 0.9935324) / 0.001465572   # +0.6%  psi_0p2 > 0.9935
        + 0.00605951 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0402832 - Q.phi_0) / 0.166757   # +0.6%  n_dr_0p2_0p4 < 10 and phi_0 < 0.04028
        + 0.005236396 * max(0.0, Q.z_top40_slots - 0.996191) * max(0.0, 0.006580753 - Q.C2_b2) / 4.241011e-06   # +0.5%  z_top40_slots > 0.9962 and C2_b2 < 0.006581
        - 0.004611212 * max(0.0, Q.C2_b2 - 0.01330402) / 0.00729146   # -0.5%  C2_b2 > 0.0133
        + 0.004297226 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +0.4%  mass_top40 < 67.73
        - 0.003647261 * max(0.0, 0.06030419 - Q.mass_over_sum_pt) / 0.005383371   # -0.4%  mass_over_sum_pt < 0.0603
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 3.778;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.778475 * (-0.03619592
        + 0.2560895 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +25.6%  mass < 82.85
        + 0.1376084 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.04417088   # +13.8%  mass < 86.4 and lam2 < 0.003688
        + 0.1151248 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +11.5%  mass < 74.25
        - 0.09033958 * max(0.0, 62.55 - Q.mass) / 5.464058   # -9.0%  mass < 62.55
        - 0.06269106 * max(0.0, 0.006142802 - Q.girth2_top15) / 0.002080651   # -6.3%  girth2_top15 < 0.006143
        - 0.04489517 * max(0.0, 74.78616 - Q.mass_top40) / 9.958349   # -4.5%  mass_top40 < 74.79
        - 0.03949631 * max(0.0, 0.06895248 - Q.mass_over_sum_pt) / 0.007729512   # -3.9%  mass_over_sum_pt < 0.06895
        - 0.03653054 * max(0.0, 89.6788 - Q.mass_top40) / 17.47899   # -3.7%  mass_top40 < 89.68
        - 0.03645245 * max(0.0, 86.4 - Q.mass) / 13.67632   # -3.6%  mass < 86.4
        - 0.035177 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.03580539   # -3.5%  mass < 86.4 and psi_0p3 < 0.9985
        - 0.03062259 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.5993597   # -3.1%  mass < 86.4 and z_dr_0p1_0p2 < 0.06473
        - 0.02422325 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -2.4%  mass < 53.87
        + 0.02333961 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, Q.n_dr_0p05_0p1 - 1.0) / 124.4066   # +2.3%  mass_top40 < 89.68 and n_dr_0p05_0p1 > 1
        - 0.0114071 * max(0.0, Q.sj3_pair_mass_max - 120.6) / 2.130243   # -1.1%  sj3_pair_mass_max > 120.6
        - 0.01084085 * max(0.0, 6.856375 - Q.log_sum_pt) * max(0.0, 0.002396991 - Q.lam2) / 7.289393e-06   # -1.1%  log_sum_pt < 6.856 and lam2 < 0.002397
        + 0.01076975 * max(0.0, Q.sd_mass - 125.1) / 1.231011   # +1.1%  sd_mass > 125.1
        - 0.009565091 * max(0.0, 0.002752094 - Q.lam1) / 0.0003912817   # -1.0%  lam1 < 0.002752
        + 0.007514764 * max(0.0, 0.0007431905 - Q.girth2_top10) * max(0.0, 1069.671 - Q.sum_pt_top40) / 0.006568426   # +0.8%  girth2_top10 < 0.0007432 and sum_pt_top40 < 1070
        + 0.005959463 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min) / 54.09437   # +0.6%  sj3_pair_mass_max > 120.6 and sj3_pair_mass_min < 76.6
        + 0.004361039 * max(0.0, 0.05077291 - Q.mass_over_sum_pt) / 0.003255623   # +0.4%  mass_over_sum_pt < 0.05077
        + 0.003629551 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926) / 0.5295314   # +0.4%  sj3_pair_mass_max > 120.6 and z_dr_0p1_0p2 > 0.2865
        - 0.003362113 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.985099 - Q.z_top50_slots) / 0.008644046   # -0.3%  mass < 86.4 and z_top50_slots < 0.9851
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 7.434;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.434208 * (0.2976559
        - 0.128618 * max(0.0, 160.8 - Q.mass_top40) / 76.75884   # -12.9%  mass_top40 < 160.8
        + 0.1062474 * max(0.0, 0.02550569 - Q.girth2_top50) / 0.01683664   # +10.6%  girth2_top50 < 0.02551
        - 0.08147452 * max(0.0, 0.02580396 - Q.mass_over_sum_pt_sq) / 0.01697662   # -8.1%  mass_over_sum_pt_sq < 0.0258
        + 0.06338569 * max(0.0, Q.mass - 74.25181) / 24.07648   # +6.3%  mass > 74.25
        - 0.06245398 * max(0.0, Q.mass - 143.7876) / 3.946979   # -6.2%  mass > 143.8
        - 0.05507886 * max(0.0, Q.mass_top50 - 92.16545) / 13.44564   # -5.5%  mass_top50 > 92.17
        + 0.04935761 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +4.9%  mass_top50 > 136.8
        - 0.03972739 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -4.0%  sum_pt < 1085
        + 0.03930298 * Q.n_particles / 45.81523   # +3.9%  n_particles
        - 0.03338861 * max(0.0, 1007.788 - Q.sum_pt) / 17.7122   # -3.3%  sum_pt < 1008
        - 0.02870822 * max(0.0, 6.910131 - Q.log_sum_pt) / 0.017275   # -2.9%  log_sum_pt < 6.91
        + 0.02645533 * max(0.0, 0.007820315 - Q.girth2_top50) / 0.002264874   # +2.6%  girth2_top50 < 0.00782
        - 0.02486345 * max(0.0, Q.mass - 136.785) / 5.043396   # -2.5%  mass > 136.8
        + 0.0246699 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # +2.5%  sum_pt_top40 < 1007
        - 0.02300007 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.8310045 - Q.tau21_b2) / 33.42141   # -2.3%  sum_pt < 1085 and tau21_b2 < 0.831
        + 0.02237301 * max(0.0, 76.9886 - Q.mass_top10) / 32.58955   # +2.2%  mass_top10 < 76.99
        - 0.01959696 * max(0.0, Q.mass - 160.8) / 1.724663   # -2.0%  mass > 160.8
        + 0.01953223 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +2.0%  mass_over_sum_pt > 0.1709
        + 0.01907175 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +1.9%  sum_pt_top40 < 1053
        - 0.01656319 * max(0.0, 16.0 - Q.n_for_90pct) / 1.776607   # -1.7%  n_for_90pct < 16
        - 0.01602235 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -1.6%  mass_over_sum_pt_sq > 0.0292
        + 0.0143646 * max(0.0, Q.mass_top5 - 33.71058) / 7.394508   # +1.4%  mass_top5 > 33.71
        + 0.01329368 * max(0.0, Q.mass - 162.8363) / 1.509832   # +1.3%  mass > 162.8
        + 0.01286752 * max(0.0, 997.0189 - Q.sum_pt_top50) / 17.90852   # +1.3%  sum_pt_top50 < 997
        - 0.01186129 * max(0.0, 966.0633 - Q.sum_pt_top30) / 29.99923   # -1.2%  sum_pt_top30 < 966.1
        + 0.01130706 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # +1.1%  log_sum_pt < 6.811
        - 0.005426324 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 858.8262 - Q.sum_pt_top40) / 1219.241   # -0.5%  sum_pt < 1085 and sum_pt_top40 < 858.8
        + 0.004860973 * max(0.0, 18.32812 - Q.pt_11) / 2.095384   # +0.5%  pt_11 < 18.33
        + 0.004499177 * max(0.0, Q.sum_pt_top20 - 956.5062) / 37.46564   # +0.4%  sum_pt_top20 > 956.5
        + 0.003995039 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.2179035 - Q.sj2_zsoft) / 2.389974   # +0.4%  sum_pt_top40 < 1053 and sj2_zsoft < 0.2179
        - 0.003819765 * max(0.0, Q.mass - 172.8) * max(0.0, 56.53125 - Q.pt_6) / 9.768822   # -0.4%  mass > 172.8 and pt_6 < 56.53
        + 0.003753976 * max(0.0, Q.mass - 172.8) / 0.7291439   # +0.4%  mass > 172.8
        + 0.003177891 * max(0.0, 16.0 - Q.n_for_90pct) * max(0.0, 3.814159 - Q.D2) / 1.817419   # +0.3%  n_for_90pct < 16 and D2 < 3.814
        + 0.002271704 * max(0.0, Q.mass - 172.8) * max(0.0, 0.04568661 - Q.z_6) / 0.006788705   # +0.2%  mass > 172.8 and z_6 < 0.04569
        + 0.001808474 * max(0.0, 1007.788 - Q.sum_pt) * max(0.0, Q.sum_pt_top3 - 512.4375) / 295.3022   # +0.2%  sum_pt < 1008 and sum_pt_top3 > 512.4
        - 0.001791448 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.00317537 - Q.C3) / 7.466262e-06   # -0.2%  log_sum_pt > 7.139 and C3 < 0.003175
        - 0.001009554 * max(0.0, Q.tau1 - 0.1751567) / 0.002322569   # -0.1%  tau1 > 0.1752
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 15.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.59385 * (-0.01306822
        - 0.1346516 * max(0.0, 91.19 - Q.mass) / 16.46423   # -13.5%  mass < 91.19
        - 0.101664 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) / 0.0178873   # -10.2%  mass_over_sum_pt < 0.09047
        + 0.0879711 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +8.8%  mass_over_sum_pt < 0.1182
        + 0.06060759 * max(0.0, 0.08873143 - Q.mass_over_sum_pt) / 0.01671255   # +6.1%  mass_over_sum_pt < 0.08873
        + 0.05666682 * max(0.0, 0.140939 - Q.mass_over_sum_pt) / 0.05796036   # +5.7%  mass_over_sum_pt < 0.1409
        - 0.05212888 * max(0.0, 80.4 - Q.mass) / 10.6806   # -5.2%  mass < 80.4
        - 0.04631596 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -4.6%  girth2_top30 < 0.01216
        - 0.03830682 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -3.8%  mass < 82.85
        + 0.03549583 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +3.5%  psi_0p3 > 0.9924
        + 0.03125932 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +3.1%  mass_over_sum_pt < 0.09795
        - 0.03058518 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -3.1%  girth2_top20 < 0.01083
        - 0.02979291 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # -3.0%  sd_rg < 0.3017
        + 0.0264496 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # +2.6%  girth2_top5 < 0.007164
        + 0.02360152 * max(0.0, 6.941997 - Q.log_sum_pt) / 0.03180972   # +2.4%  log_sum_pt < 6.942
        - 0.02080311 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # -2.1%  log_sum_pt < 6.989
        + 0.02020668 * max(0.0, 82.66587 - Q.mass_top30) / 15.96638   # +2.0%  mass_top30 < 82.67
        - 0.01960139 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p2 - 0.9087063) / 0.00142529   # -2.0%  mass_over_sum_pt < 0.09047 and psi_0p2 > 0.9087
        - 0.01851303 * max(0.0, 0.076787 - Q.girth) / 0.02105267   # -1.9%  girth < 0.07679
        + 0.0177318 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +1.8%  e2 < 0.0303
        - 0.01538787 * max(0.0, 0.076787 - Q.girth) * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.0007421723   # -1.5%  girth < 0.07679 and z_dr_0p2_0p4 < 0.0518
        + 0.01529378 * max(0.0, 69.65633 - Q.sd_mass) / 24.56034   # +1.5%  sd_mass < 69.66
        + 0.01293942 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # +1.3%  girth2_top20 < 0.006374
        + 0.012776 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +1.3%  n_dr_0p2_0p4 < 18
        - 0.01230699 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1881908 - Q.sd_rg) / 0.0002726472   # -1.2%  psi_0p3 > 0.9924 and sd_rg < 0.1882
        - 0.01177541 * max(0.0, 0.007164202 - Q.girth2_top5) * max(0.0, Q.n_pt_above_10 - 13.0) / 0.02038471   # -1.2%  girth2_top5 < 0.007164 and n_pt_above_10 > 13
        + 0.01050053 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +1.1%  tau21_b2 < 0.3425
        - 0.009385513 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.02048524 - Q.lam1) / 0.001304679   # -0.9%  tau21_b2 < 0.3425 and lam1 < 0.02049
        - 0.008520482 * max(0.0, 0.001592178 - Q.girth2_top3) / 0.000513961   # -0.9%  girth2_top3 < 0.001592
        - 0.007591201 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2404747) / 0.01288721   # -0.8%  N2 < 0.4227 and max_dr > 0.2405
        + 0.007161637 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # +0.7%  sum_pt_top40 < 1025
        + 0.00559262 * max(0.0, 0.1778185 - Q.sd_rg) / 0.06576479   # +0.6%  sd_rg < 0.1778
        - 0.005151447 * max(0.0, 976.277 - Q.sum_pt_top50) / 12.87298   # -0.5%  sum_pt_top50 < 976.3
        - 0.003755566 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1231747 - Q.z_2nd) / 6.106735e-05   # -0.4%  psi_0p3 > 0.9924 and z_2nd < 0.1232
        - 0.002994255 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 32.0 - Q.n_real_top40) / 0.1595466   # -0.3%  tau21_b2 < 0.3425 and n_real_top40 < 32
        - 0.002433617 * max(0.0, 906.6023 - Q.sum_pt_top40) / 6.984569   # -0.2%  sum_pt_top40 < 906.6
        + 0.002117493 * max(0.0, 0.02210818 - Q.e2) / 0.00342102   # +0.2%  e2 < 0.02211
        - 0.001963015 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, Q.sd_rg - 0.2280025) / 1.846945e-05   # -0.2%  psi_0p3 > 0.9924 and sd_rg > 0.228
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 5.263;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.262964 * (-0.1257694
        + 0.1167089 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +11.7%  log_sum_pt < 7.017
        + 0.1165355 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +11.7%  girth2_top5 < 0.00833
        + 0.1045457 * max(0.0, 79.18312 - Q.sd_mass) / 30.46368   # +10.5%  sd_mass < 79.18
        - 0.06465955 * max(0.0, 1018.698 - Q.sum_pt_top40) / 34.89083   # -6.5%  sum_pt_top40 < 1019
        + 0.05997176 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +6.0%  z_dr_0p1_0p2 < 0.1203
        - 0.05714468 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -5.7%  psi_0p3 > 0.9897
        - 0.05570233 * max(0.0, 45.595 - Q.sd_mass) / 13.46843   # -5.6%  sd_mass < 45.59
        - 0.03797897 * max(0.0, 0.002270363 - Q.girth2_top5) / 0.0007634099   # -3.8%  girth2_top5 < 0.00227
        + 0.02958001 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # +3.0%  girth2_top10 < 0.007679
        - 0.02782782 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.8%  tau1 < 0.07057
        + 0.0252609 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +2.5%  sum_pt < 986.1
        - 0.02489768 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -2.5%  psi_0p1 > 0.8976
        + 0.02472849 * max(0.0, 933.1875 - Q.sum_pt_top30) / 20.26547   # +2.5%  sum_pt_top30 < 933.2
        - 0.02319332 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -2.3%  log_sum_pt < 6.903
        - 0.02242466 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 1260.541 - Q.sum_pt) / 87.05552   # -2.2%  z_dr_0_0p05 < 0.8459 and sum_pt < 1261
        + 0.02229656 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +2.2%  lam2 < 0.001776
        - 0.01619225 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583) / 0.0009330603   # -1.6%  girth2_top5 < 0.00833 and sj3_pairmin_over_m > 0.09541
        - 0.01581738 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # -1.6%  mass_over_sum_pt > 0.1709
        + 0.01508414 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +1.5%  sj2_dr > 0.2232
        - 0.0147052 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677) / 0.04548726   # -1.5%  girth2_top5 < 0.00833 and sj3_mass1 > 5.113
        + 0.01467081 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +1.5%  lam1 < 0.004673
        + 0.01366535 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2) / 0.0004272674   # +1.4%  sj2_dr > 0.2232 and C2_b2 < 0.04009
        - 0.01274479 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -1.3%  sum_pt < 1002
        + 0.01265465 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 58.0 - Q.n_particles) / 0.8649544   # +1.3%  z_dr_0p1_0p2 < 0.1203 and n_particles < 58
        - 0.01170432 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # -1.2%  mass_top50 < 71.8
        - 0.01123097 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0) / 0.2039042   # -1.1%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p2_0p4 > 5
        - 0.01102827 * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 3.233647   # -1.1%  n_dr_0p2_0p4 > 9
        - 0.01094574 * max(0.0, 935.8189 - Q.sum_pt_top40) / 10.51928   # -1.1%  sum_pt_top40 < 935.8
        - 0.01057864 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.02970886 - Q.absphi_0) / 0.06277088   # -1.1%  n_dr_0p1_0p2 < 13 and absphi_0 < 0.02971
        + 0.006800864 * max(0.0, Q.lam1 - 0.01649354) / 0.001003295   # +0.7%  lam1 > 0.01649
        + 0.005492048 * max(0.0, 35.30013 - Q.sj3_pair_mass_max) / 1.577545   # +0.5%  sj3_pair_mass_max < 35.3
        + 0.003227806 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt) / 2.722316   # +0.3%  z_dr_0_0p05 < 0.8459 and sum_pt < 949.9
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.5835742647058824, 2.88315131302521, 0.22844747899159665, 0.4214018907563025, 0.7481411764705882, 1.064209243697479, 0.509768487394958, 0.36060871848739495, 1.1968785714285715, 0.9892322478991596, 1.100321848739496, 0.7630335084033614, 0.4999455882352941, 1.532280619747899, 0.25926360294117645, 0.350981512605042]
T = [3.866703842240021, 2.4305102317489498, 4.093818336397058, 4.100463232011555, 3.856815406709559]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +47%, n4 -17%, n9 +13%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.4660221 * h[1] / H_AVG[1]
            - 0.1692976 * h[4] / H_AVG[4]
            + 0.1319141 * h[9] / H_AVG[9]
            - 0.08600746 * h[5] / H_AVG[5]
            - 0.08173665 * h[3] / H_AVG[3]
            + 0.03030352 * h[12] / H_AVG[12]
            + 0.02059928 * h[6] / H_AVG[6]
            + 0.009672956 * h[8] / H_AVG[8]
            - 0.004446301 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +23%, n1 -22%, n11 -8%, n12 +7%, n6 +5% ...
            - 0.3270507 * h[4] / H_AVG[4]
            + 0.2289409 * h[9] / H_AVG[9]
            - 0.2224187 * h[1] / H_AVG[1]
            - 0.07848491 * h[11] / H_AVG[11]
            + 0.07070791 * h[12] / H_AVG[12]
            + 0.04588002 * h[6] / H_AVG[6]
            + 0.01174895 * h[2] / H_AVG[2]
            + 0.007694363 * h[8] / H_AVG[8]
            - 0.00707363 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -26%, n5 +15%, n11 +11%, n0 +11%, n14 -9%, n4 +6% ...
            - 0.2558171 * h[8] / H_AVG[8]
            + 0.1502866 * h[5] / H_AVG[5]
            + 0.1106671 * h[11] / H_AVG[11]
            + 0.1069126 * h[0] / H_AVG[0]
            - 0.08707945 * h[14] / H_AVG[14]
            + 0.06281997 * h[4] / H_AVG[4]
            - 0.05505385 * h[7] / H_AVG[7]
            - 0.05285886 * h[9] / H_AVG[9]
            - 0.04961209 * h[12] / H_AVG[12]
            + 0.04503456 * h[3] / H_AVG[3]
            - 0.01607522 * h[15] / H_AVG[15]
            + 0.007782595 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -27%, n0 -20%, n5 +18%, n6 -12%, n7 +8%, n15 +5% ...
            - 0.2736456 * h[8] / H_AVG[8]
            - 0.1956888 * h[0] / H_AVG[0]
            + 0.1784296 * h[5] / H_AVG[5]
            - 0.1204347 * h[6] / H_AVG[6]
            + 0.07969872 * h[7] / H_AVG[7]
            + 0.04814751 * h[15] / H_AVG[15]
            + 0.04496159 * h[3] / H_AVG[3]
            + 0.0381013 * h[12] / H_AVG[12]
            - 0.02089223 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -36%, n10 +28%, n5 -13%, n8 +7%, n12 -5%, n4 +4% ...
            - 0.3600456 * h[13] / H_AVG[13]
            + 0.2808351 * h[10] / H_AVG[10]
            - 0.129342 * h[5] / H_AVG[5]
            + 0.06545985 * h[8] / H_AVG[8]
            - 0.04860995 * h[12] / H_AVG[12]
            + 0.03637106 * h[4] / H_AVG[4]
            - 0.0341261 * h[15] / H_AVG[15]
            - 0.02629662 * h[7] / H_AVG[7]
            + 0.01891373 * h[0] / H_AVG[0]
        ),
    ]]


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
