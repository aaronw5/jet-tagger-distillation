"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.4%   (on for 88% of jets)
  neuron  9:  12.2%   (on for 66% of jets)
  neuron  7:  10.1%   (on for 60% of jets)
  neuron  3:   8.7%   (on for 24% of jets)
  neuron 10:   8.2%   (on for 81% of jets)
  neuron  2:   7.4%   (on for 92% of jets)
  neuron  6:   7.3%   (on for 36% of jets)
  neuron  5:   7.2%   (on for 60% of jets)
  neuron 11:   6.7%   (on for 83% of jets)
  neuron 14:   3.9%   (on for 28% of jets)
  neuron  0:   3.5%   (on for 39% of jets)
  neuron  4:   3.3%   (on for 65% of jets)
  neuron  1:   3.1%   (on for 61% of jets)
  neuron 15:   2.6%   (on for 28% of jets)
  neuron  8:   1.1%   (on for 37% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 89.9% of jets.

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
  Q.pair_mass_0_7          mass of particles 0 and 7 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_3                    pT of particle 3 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_1               |Δη| of particle 1
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_3               |Δφ| of particle 3
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_3                  Δη of particle 3
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
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
        pair_mass_0_7=pair_mass(0, 7),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_3=z[3],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_7=z[7] * dr[7],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_0=abs(eta[0]),
        abseta_1=abs(eta[1]),
        abseta_4=abs(eta[4]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_3=abs(phi[3]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr23=subjets(3)["dr"][2],
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        eta_3=eta[3],
        phi_0=phi[0],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
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
    )


def neuron_0(Q):
    # scale S = 19.86;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.85813 * (-0.09437805
        + 0.1313374 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +13.1%  sum_z_dr2 < 0.01324
        + 0.1008753 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +10.1%  lam1_plus_lam2 < 0.008678
        - 0.09955276 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) / 0.00156571   # -10.0%  lam1_plus_lam2 < 0.004372
        - 0.06885262 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -6.9%  sum_z_dr < 0.08724
        + 0.04903269 * max(0.0, 0.0245477 - Q.e2) / 0.007455821   # +4.9%  e2 < 0.02455
        + 0.04481602 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +4.5%  sj3_dr_max < 0.3012
        - 0.04413904 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -4.4%  lam1 < 0.005433
        + 0.04049731 * max(0.0, 0.0002758826 - Q.lam1) / 3.586469e-05   # +4.0%  lam1 < 0.0002759
        - 0.04019838 * max(0.0, 21.78408 - Q.mass) / 4.431023   # -4.0%  mass < 21.78
        - 0.03972041 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -4.0%  sj3_dr_max > 0.2337
        + 0.02731033 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +2.7%  planar_flow < 0.1484
        + 0.02591996 * max(0.0, Q.sj3_dr_max - 0.1070199) / 0.08518534   # +2.6%  sj3_dr_max > 0.107
        - 0.02563549 * max(0.0, 29.6447 - Q.mass) / 7.34723   # -2.6%  mass < 29.64
        - 0.02435009 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -2.4%  log_sum_pt > 6.67
        - 0.02318917 * max(0.0, 0.001563465 - Q.C2_b2) / 0.0008800039   # -2.3%  C2_b2 < 0.001563
        + 0.01927186 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +1.9%  sum_pt_top5 > 687.4
        - 0.0187858 * max(0.0, 56.92035 - Q.mass) / 21.78475   # -1.9%  mass < 56.92
        - 0.01765824 * max(0.0, 64.61873 - Q.mass) / 27.57279   # -1.8%  mass < 64.62
        + 0.01504989 * max(0.0, 0.01882765 - Q.sum_z_dr2) / 0.01314296   # +1.5%  sum_z_dr2 < 0.01883
        - 0.01413953 * max(0.0, 7.300726e-05 - Q.lam2) / 2.767163e-05   # -1.4%  lam2 < 7.301e-05
        + 0.01412259 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, 1.002471 - Q.D2) / 0.001008123   # +1.4%  sum_z_dr2 < 0.01324 and D2 < 1.002
        + 0.01206688 * max(0.0, 0.003904593 - Q.mass_over_sum_pt_sq) / 0.001485439   # +1.2%  mass_over_sum_pt_sq < 0.003905
        + 0.01094564 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +1.1%  lam1 < 0.006507
        + 0.01066509 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2) / 2.36013e-06   # +1.1%  lam2 < 7.301e-05 and D2_b2 < 0.267
        - 0.01061986 * max(0.0, 0.007929074 - Q.sum_z_dr2_top3) / 0.004560262   # -1.1%  sum_z_dr2_top3 < 0.007929
        - 0.009957218 * max(0.0, 0.01882765 - Q.sum_z_dr2) * max(0.0, 46.125 - Q.pt_6) / 0.1159788   # -1.0%  sum_z_dr2 < 0.01883 and pt_6 < 46.12
        - 0.00956006 * max(0.0, 0.01323868 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.01837778) / 2.056482e-05   # -1.0%  sum_z_dr2 < 0.01324 and centroid_offset > 0.01838
        - 0.008472975 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.01452637 - Q.phi_0) / 0.1160543   # -0.8%  mass < 29.64 and phi_0 < 0.01453
        + 0.007504315 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 0.07232166 - Q.dr_4) / 0.001949106   # +0.8%  log_sum_pt > 6.67 and dr_4 < 0.07232
        - 0.006877782 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # -0.7%  lam1 < 0.006507 and D2 < 0.8757
        + 0.004352075 * max(0.0, Q.n_dr_0_0p05 - 4.0) / 1.61275   # +0.4%  n_dr_0_0p05 > 4
        - 0.004196586 * max(0.0, Q.sum_pt - 901.5938) / 12.40362   # -0.4%  sum_pt > 901.6
        - 0.004087039 * max(0.0, 6.080494 - Q.log_sum_pt) / 0.006097657   # -0.4%  log_sum_pt < 6.08
        - 0.002509944 * max(0.0, 64.61873 - Q.mass) * max(0.0, 0.1830092 - Q.D2_b2) / 0.3382769   # -0.3%  mass < 64.62 and D2_b2 < 0.183
        - 0.00205255 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 358.375 - Q.sum_pt_top2) / 2.426548   # -0.2%  planar_flow < 0.1484 and sum_pt_top2 < 358.4
        - 0.001848871 * max(0.0, 56.92035 - Q.mass) * max(0.0, Q.dr_max_012 - 0.06112084) / 0.1887976   # -0.2%  mass < 56.92 and dr_max_012 > 0.06112
        - 0.00183193 * max(0.0, 0.3012016 - Q.sj3_dr_max) * max(0.0, 0.05744392 - Q.D2_b2) / 0.0005632394   # -0.2%  sj3_dr_max < 0.3012 and D2_b2 < 0.05744
        - 0.001589955 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.eccentricity - 0.9458207) / 0.1898472   # -0.2%  sum_pt > 901.6 and eccentricity > 0.9458
        - 0.001270141 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, Q.dr0_6 - 0.1755206) / 0.0008435917   # -0.1%  planar_flow < 0.1484 and dr0_6 > 0.1755
        + 0.001112684 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, 0.03363037 - Q.absphi_3) / 0.2707601   # +0.1%  sum_pt > 901.6 and absphi_3 < 0.03363
        - 0.001111202 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, 0.119512 - Q.dr12) / 1.24022   # -0.1%  sum_pt > 901.6 and dr12 < 0.1195
        + 0.001020453 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +0.1%  centroid_offset > 0.0499
        + 0.0008299377 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) * max(0.0, Q.eccentricity - 0.9598562) / 6.567391e-06   # +0.1%  lam1_plus_lam2 < 0.004372 and eccentricity > 0.9599
        + 0.0007088987 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.02320757 - Q.z_7) / 2.048782e-05   # +0.1%  planar_flow < 0.1484 and z_7 < 0.02321
        + 0.0002424763 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.dr0_6 - 0.2347949) / 0.03872508   # +0.0%  sum_pt > 901.6 and dr0_6 > 0.2348
        - 0.0001104566 * max(0.0, 0.004372139 - Q.lam1_plus_lam2) * max(0.0, 0.7459513 - Q.D2) / 7.193909e-06   # -0.0%  lam1_plus_lam2 < 0.004372 and D2 < 0.746
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 14.68;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.68442 * (0.0124167
        - 0.176619 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -17.7%  sum_z_dr2 < 0.008678
        + 0.0943675 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) / 0.002578466   # +9.4%  mass_over_sum_pt_sq < 0.005833
        - 0.08545324 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 0.002544039   # -8.5%  lam1_plus_lam2 < 0.006097
        + 0.07204479 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +7.2%  log_sum_pt > 6.378
        + 0.06395059 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +6.4%  lam1 < 0.005954
        - 0.05900496 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -5.9%  sum_pt_top5 > 531.2
        + 0.0573002 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # +5.7%  sum_zz_dr2 < 0.008169
        + 0.05122687 * max(0.0, Q.sum_pt_top5 - 367.5938) / 230.8403   # +5.1%  sum_pt_top5 > 367.6
        + 0.04913415 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +4.9%  log_sum_pt > 6.503
        - 0.04437027 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -4.4%  z_7 < 0.06165
        - 0.04018798 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # -4.0%  sum_z_dr < 0.0717
        + 0.02925668 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # +2.9%  z_7 > 0.02321
        - 0.02485368 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1) / 0.1196218   # -2.5%  log_sum_pt > 6.378 and z_dr_0p05_0p1 < 0.7509
        + 0.02083764 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +2.1%  lam1 < 0.008376
        + 0.01668591 * max(0.0, Q.log_sum_pt - 6.605974) / 0.07073019   # +1.7%  log_sum_pt > 6.606
        + 0.01083079 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.008921136 - Q.mean_phi2) / 0.0001024498   # +1.1%  z_7 < 0.06165 and mean_phi2 < 0.008921
        + 0.01082562 * max(0.0, 0.008678045 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.02076709) / 7.418362e-06   # +1.1%  sum_z_dr2 < 0.008678 and centroid_offset > 0.02077
        + 0.008529815 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +0.9%  sj3_dr_max > 0.169
        - 0.008114979 * max(0.0, 0.0008722282 - Q.lam1) / 0.0001882079   # -0.8%  lam1 < 0.0008722
        - 0.007771995 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # -0.8%  lam1 < 0.008376 and centroid_offset > 0.02077
        - 0.007635726 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1484197 - Q.planar_flow) / 0.0001369594   # -0.8%  lam1 < 0.008376 and planar_flow < 0.1484
        + 0.006617721 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.centroid_offset - 0.009480685) / 0.0009010889   # +0.7%  log_sum_pt > 6.378 and centroid_offset > 0.009481
        - 0.006256223 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 52.90625 - Q.pt_6) / 13.55761   # -0.6%  pt_7 > 34.53 and pt_6 < 52.91
        + 0.005842108 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 1.627072e-05 - Q.e3) / 1.912338e-06   # +0.6%  log_sum_pt > 6.378 and e3 < 1.627e-05
        - 0.005562249 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236) / 1.769648e-05   # -0.6%  mass_over_sum_pt_sq < 0.005833 and centroid_offset > 0.008092
        - 0.005147283 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.00323438   # -0.5%  lam1 < 0.008376 and n_pt_above_50 > 5
        - 0.004683224 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.04875823) / 0.002321449   # -0.5%  z_7 < 0.06165 and z_dr_0p05_0p1 > 0.04876
        + 0.004593381 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.sj2_dr - 0.1294903) / 0.1877842   # +0.5%  pt_7 > 34.53 and sj2_dr > 0.1295
        - 0.004549593 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 1.679198 - Q.D2) / 0.005760154   # -0.5%  z_7 < 0.06165 and D2 < 1.679
        - 0.003510861 * max(0.0, Q.sj3_dr_min - 0.03628191) / 0.02376249   # -0.4%  sj3_dr_min > 0.03628
        - 0.002953102 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.5700307   # -0.3%  sj3_dr_max > 0.169 and sj3_pair_mass_min > 5.744
        + 0.002910691 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 1.432482 - Q.D2) / 0.01970563   # +0.3%  log_sum_pt > 6.606 and D2 < 1.432
        - 0.002715107 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 2.955458e-05 - Q.e3) / 7.137805e-05   # -0.3%  pt_7 > 34.53 and e3 < 2.955e-05
        + 0.001751445 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 3.175597   # +0.2%  pt_7 > 34.53 and n_dr_0p1_0p2 > 1
        + 0.0015908 * max(0.0, 1.0 - Q.n_dr_0_0p05) * max(0.0, Q.zdr_7 - 0.0007928864) / 0.001861106   # +0.2%  n_dr_0_0p05 < 1 and zdr_7 > 0.0007929
        - 0.001435709 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.1%  pt_7 > 53.44
        - 0.0008781457 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, 0.03885671 - Q.D2_b2) / 4.310709e-07   # -0.1%  mass_over_sum_pt_sq < 0.005833 and D2_b2 < 0.03886
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 10.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.59 * (0.1754639
        - 0.08424404 * max(0.0, Q.LHA - 0.111565) / 0.1352185   # -8.4%  LHA > 0.1116
        - 0.0803284 * max(0.0, 716.8828 - Q.sum_pt_top5) / 152.5573   # -8.0%  sum_pt_top5 < 716.9
        + 0.07241044 * max(0.0, 788.4484 - Q.sum_pt) / 112.1576   # +7.2%  sum_pt < 788.4
        + 0.07019554 * max(0.0, 63.68899 - Q.sj3_pair_mass_max) / 31.57762   # +7.0%  sj3_pair_mass_max < 63.69
        - 0.06810836 * max(0.0, 0.0003193707 - Q.sum_z_dr2) / 4.035777e-05   # -6.8%  sum_z_dr2 < 0.0003194
        + 0.06421125 * max(0.0, 0.0003193707 - Q.sum_z_dr2) * max(0.0, 36.76827 - Q.mass_top2) / 0.001427268   # +6.4%  sum_z_dr2 < 0.0003194 and mass_top2 < 36.77
        - 0.06089928 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -6.1%  pt_7 < 53.44
        + 0.05579566 * max(0.0, Q.pt_6 - 27.57812) / 13.69978   # +5.6%  pt_6 > 27.58
        - 0.04415265 * max(0.0, Q.z_6 - 0.02886576) / 0.03195851   # -4.4%  z_6 > 0.02887
        + 0.04341471 * max(0.0, 840.0195 - Q.sum_pt) / 148.4153   # +4.3%  sum_pt < 840
        + 0.03860944 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +3.9%  z_7 < 0.07149
        + 0.03633608 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +3.6%  lam1 < 0.005954
        - 0.03000739 * max(0.0, 63.68899 - Q.sj3_pair_mass_max) * max(0.0, 0.06810151 - Q.z_7) / 0.6284113   # -3.0%  sj3_pair_mass_max < 63.69 and z_7 < 0.0681
        - 0.02447005 * max(0.0, Q.mass - 36.22941) / 14.20528   # -2.4%  mass > 36.23
        + 0.02370512 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # +2.4%  log_sum_pt < 6.606
        + 0.02052898 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +2.1%  pt_7 > 34.53
        + 0.01886575 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # +1.9%  log_sum_pt < 6.464
        + 0.01790851 * max(0.0, Q.mass - 15.45403) / 27.25251   # +1.8%  mass > 15.45
        + 0.01754497 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # +1.8%  zdr_0 < 0.02118
        - 0.01727989 * max(0.0, Q.z_7 - 0.04939969) / 0.01023002   # -1.7%  z_7 > 0.0494
        + 0.01287936 * max(0.0, 840.0195 - Q.sum_pt) * max(0.0, 0.02164863 - Q.dr_5) / 0.1765901   # +1.3%  sum_pt < 840 and dr_5 < 0.02165
        - 0.01218686 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.6947818 - Q.planar_flow) / 0.009090069   # -1.2%  z_7 < 0.07149 and planar_flow < 0.6948
        + 0.01141729 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # +1.1%  log_sum_pt > 6.843
        - 0.01015043 * max(0.0, 63.68899 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.01096064) / 0.2223898   # -1.0%  sj3_pair_mass_max < 63.69 and centroid_offset > 0.01096
        - 0.009660041 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, 0.02164863 - Q.dr_5) / 0.1197721   # -1.0%  sum_pt < 788.4 and dr_5 < 0.02165
        - 0.009342047 * max(0.0, 0.03243272 - Q.z_7) / 0.002061024   # -0.9%  z_7 < 0.03243
        - 0.00885789 * max(0.0, Q.pt_6 - 27.57812) * max(0.0, 0.7974684 - Q.planar_flow) / 7.391558   # -0.9%  pt_6 > 27.58 and planar_flow < 0.7975
        - 0.008226199 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # -0.8%  sum_z_dr < 0.007674
        - 0.007146924 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.max_dr - 0.0931108) / 3.745922e-05   # -0.7%  lam1 < 0.005954 and max_dr > 0.09311
        - 0.004164289 * max(0.0, Q.z_7 - 0.04939969) * max(0.0, 0.0623951 - Q.sj3_dr_min) / 0.0002978985   # -0.4%  z_7 > 0.0494 and sj3_dr_min < 0.0624
        - 0.002738626 * max(0.0, Q.z_6 - 0.02886576) * max(0.0, 0.02164863 - Q.dr_5) / 6.468094e-05   # -0.3%  z_6 > 0.02887 and dr_5 < 0.02165
        - 0.002608855 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.3%  sum_pt_top5 > 840
        + 0.002608622 * max(0.0, Q.m012 - 42.18339) / 1.09247   # +0.3%  m012 > 42.18
        + 0.002591571 * max(0.0, 6.605974 - Q.log_sum_pt) * max(0.0, 0.0003061234 - Q.lam2) / 1.525985e-05   # +0.3%  log_sum_pt < 6.606 and lam2 < 0.0003061
        + 0.002291394 * max(0.0, 0.007673833 - Q.sum_z_dr) * max(0.0, 71.6875 - Q.pt_4) / 0.003882715   # +0.2%  sum_z_dr < 0.007674 and pt_4 < 71.69
        - 0.002282945 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.2%  log_sum_pt > 6.896
        - 0.001335566 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.pt_6 - 41.21875) / 0.06090836   # -0.1%  log_sum_pt > 6.843 and pt_6 > 41.22
        + 0.0004945596 * max(0.0, 788.4484 - Q.sum_pt) * max(0.0, 0.03623337 - Q.max_dr) / 0.1060521   # +0.0%  sum_pt < 788.4 and max_dr < 0.03623
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 17.21;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.20708 * (-0.2236725
        - 0.1918663 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # -19.2%  sum_z_dr2 > 0.008678
        + 0.08878972 * max(0.0, Q.lam1_plus_lam2 - 0.007520088) / 0.002470136   # +8.9%  lam1_plus_lam2 > 0.00752
        + 0.06883832 * max(0.0, Q.sum_z_dr - 0.04081947) / 0.0270611   # +6.9%  sum_z_dr > 0.04082
        + 0.06462155 * max(0.0, Q.sj2_dr - 0.1872617) / 0.02732165   # +6.5%  sj2_dr > 0.1873
        - 0.06370619 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # -6.4%  tau1 > 0.05357
        + 0.05966105 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +6.0%  lam1 > 0.008376
        + 0.05763409 * max(0.0, Q.mass_over_sum_pt - 0.0681391) / 0.01539157   # +5.8%  mass_over_sum_pt > 0.06814
        - 0.05435749 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # -5.4%  lam1_plus_lam2 > 0.01324
        + 0.04453494 * max(0.0, Q.sum_z_dr - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494) / 0.008781747   # +4.5%  sum_z_dr > 0.04082 and log_sum_pt > 6.08
        + 0.04211786 * max(0.0, Q.sum_z_dr - 0.07608178) / 0.01059033   # +4.2%  sum_z_dr > 0.07608
        + 0.02301526 * max(0.0, Q.max_dr - 0.1027585) / 0.04505724   # +2.3%  max_dr > 0.1028
        - 0.02067177 * max(0.0, Q.mass - 64.61873) / 3.274818   # -2.1%  mass > 64.62
        - 0.02055237 * max(0.0, Q.sj2_dr - 0.1492731) / 0.04449993   # -2.1%  sj2_dr > 0.1493
        + 0.01826051 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # +1.8%  sum_z_dr2_top5 < 0.00833
        - 0.01786891 * max(0.0, 0.004372139 - Q.sum_z_dr2) / 0.00156571   # -1.8%  sum_z_dr2 < 0.004372
        + 0.01609305 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # +1.6%  lam1 > 0.01643
        + 0.01571403 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # +1.6%  tau1 > 0.1028
        - 0.01482969 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) / 0.003638607   # -1.5%  sum_z_dr2_top5 < 0.007164
        + 0.01133674 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # +1.1%  e2 > 0.06344
        - 0.01003637 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr) / 0.0001196243   # -1.0%  mass_over_sum_pt > 0.06814 and sj2_dr < 0.218
        + 0.009311844 * max(0.0, 0.05226226 - Q.z_dr_0_0p05) / 0.01510942   # +0.9%  z_dr_0_0p05 < 0.05226
        - 0.008914548 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113) / 0.39507   # -0.9%  sj2_dr > 0.1873 and sj2_mass1 > 2.25
        + 0.008589275 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 0.07861328 - Q.abseta_0) / 0.0002969212   # +0.9%  centroid_offset > 0.01096 and abseta_0 < 0.07861
        + 0.008372731 * max(0.0, Q.lam1_plus_lam2 - 0.007520088) * max(0.0, Q.sj3_pairmin_over_m - 0.07708997) / 0.0004577253   # +0.8%  lam1_plus_lam2 > 0.00752 and sj3_pairmin_over_m > 0.07709
        + 0.008116771 * max(0.0, Q.centroid_offset - 0.01096064) / 0.008813008   # +0.8%  centroid_offset > 0.01096
        + 0.006458399 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6) / 0.005634141   # +0.6%  lam2 > 0.001131 and pt_6 < 56.53
        - 0.006124661 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.pt_6 - 31.90625) / 0.1358885   # -0.6%  mass_over_sum_pt > 0.06814 and pt_6 > 31.91
        - 0.005951471 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # -0.6%  sj2_dr > 0.2688
        + 0.004929619 * max(0.0, Q.sd_mass - 62.73432) / 3.311364   # +0.5%  sd_mass > 62.73
        + 0.004713512 * max(0.0, -0.004664942 - Q.mean_eta) / 0.003754527   # +0.5%  mean_eta < -0.004665
        - 0.004480723 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.01084112 - Q.zdr_1) / 0.0001011412   # -0.4%  max_dr > 0.1028 and zdr_1 < 0.01084
        + 0.004087056 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, Q.z_6 - 0.05096142) / 7.479719e-06   # +0.4%  lam2 > 0.001131 and z_6 > 0.05096
        - 0.003597655 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -0.4%  lam1 > 0.012
        - 0.002862734 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.05268713 - Q.dr_3) / 0.000155932   # -0.3%  sj2_dr > 0.1873 and dr_3 < 0.05269
        + 0.002697353 * max(0.0, Q.mean_eta - 0.01772426) / 0.001375178   # +0.3%  mean_eta > 0.01772
        + 0.001855995 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sj2_mass1 - 16.86126) / 0.02534836   # +0.2%  e2 > 0.06344 and sj2_mass1 > 16.86
        - 0.001046897 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, -0.05963135 - Q.eta_1) / 0.0004225811   # -0.1%  max_dr > 0.1028 and eta_1 < -0.05963
        - 0.0009579836 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, Q.abseta_0 - 0.1057739) / 0.0002451453   # -0.1%  max_dr > 0.1028 and abseta_0 > 0.1058
        + 0.0009009844 * max(0.0, Q.tau1 - 0.05356915) * max(0.0, Q.pt_5 - 59.125) / 0.0351992   # +0.1%  tau1 > 0.05357 and pt_5 > 59.12
        - 0.0008570619 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.z_top5 - 0.7773372) / 2.699283e-05   # -0.1%  e2 > 0.06344 and z_top5 > 0.7773
        + 0.0006665148 * max(0.0, Q.sum_z_dr - 0.07608178) * max(0.0, Q.eta_0 - 0.07952881) / 0.0001300296   # +0.1%  sum_z_dr > 0.07608 and eta_0 > 0.07953
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 41.95;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 41.95047 * (-0.07380424
        + 0.1152816 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # +11.5%  sum_zz_dr2 < 0.01166
        + 0.1060199 * max(0.0, 0.01716248 - Q.sum_zz_dr2) / 0.0120354   # +10.6%  sum_zz_dr2 < 0.01716
        - 0.08883156 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -8.9%  sum_z_dr2 < 0.008678
        - 0.08036058 * max(0.0, 0.01882765 - Q.sum_z_dr2) / 0.01314296   # -8.0%  sum_z_dr2 < 0.01883
        + 0.06666261 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +6.7%  N2 < 0.2233
        + 0.05899431 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +5.9%  mass < 76.66
        - 0.0527561 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # -5.3%  lam1_plus_lam2 < 0.01324
        - 0.0468086 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -4.7%  lam2 < 0.0005373
        + 0.04597671 * max(0.0, 0.009032972 - Q.C2_b2) / 0.007288418   # +4.6%  C2_b2 < 0.009033
        + 0.03323088 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +3.3%  sj3_dr_max > 0.169
        - 0.02860217 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266) / 0.01381411   # -2.9%  N2 < 0.2233 and eccentricity > 0.7117
        + 0.02593442 * max(0.0, Q.e2 - 0.009668065) / 0.02044666   # +2.6%  e2 > 0.009668
        - 0.02578638 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -2.6%  sj3_dr_max > 0.2337
        - 0.02205732 * max(0.0, Q.sum_z_dr - 0.05464922) / 0.01938482   # -2.2%  sum_z_dr > 0.05465
        - 0.02007476 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -2.0%  sum_pt < 739.5
        - 0.01826276 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -1.8%  mass_over_sum_pt > 0.09041
        - 0.01763866 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -1.8%  mass < 69.61
        + 0.01704916 * max(0.0, Q.sum_z_dr2 - 0.003562611) / 0.004066872   # +1.7%  sum_z_dr2 > 0.003563
        - 0.014085 * max(0.0, 0.05119235 - Q.C2) / 0.02746756   # -1.4%  C2 < 0.05119
        + 0.01260776 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # +1.3%  sum_z_dr2 < 0.002635
        + 0.01166786 * max(0.0, 1.002471 - Q.D2) / 0.1871081   # +1.2%  D2 < 1.002
        - 0.01049945 * max(0.0, 76.6557 - Q.mass) * max(0.0, 0.875672 - Q.D2) / 2.2476   # -1.0%  mass < 76.66 and D2 < 0.8757
        + 0.008891407 * max(0.0, Q.tau1 - 0.07283629) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.001837197   # +0.9%  tau1 > 0.07284 and sj3_dr_min < 0.209
        + 0.007789966 * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) / 0.004543329   # +0.8%  sum_z_dr2_top2 < 0.00764
        - 0.006980721 * max(0.0, 1.002471 - Q.D2) * max(0.0, 53.4375 - Q.pt_7) / 3.084564   # -0.7%  D2 < 1.002 and pt_7 < 53.44
        + 0.006952206 * max(0.0, 0.01165737 - Q.sum_zz_dr2) * max(0.0, 0.875672 - Q.D2) / 0.0005867961   # +0.7%  sum_zz_dr2 < 0.01166 and D2 < 0.8757
        + 0.006711008 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # +0.7%  sum_z_dr2_top5 < 0.00833
        + 0.006410966 * max(0.0, Q.tau1 - 0.1136369) / 0.007045238   # +0.6%  tau1 > 0.1136
        - 0.004963322 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 60.63098 - Q.mass) / 0.3406587   # -0.5%  N2 < 0.2233 and mass < 60.63
        + 0.00374572 * max(0.0, Q.max_dr - 0.2215867) / 0.008157178   # +0.4%  max_dr > 0.2216
        - 0.003741235 * max(0.0, Q.sd_mass - 38.43971) / 11.50403   # -0.4%  sd_mass > 38.44
        - 0.003630616 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1626038 - Q.abseta_7) / 0.005313035   # -0.4%  N2 < 0.2233 and abseta_7 < 0.1626
        + 0.003530458 * max(0.0, Q.tau1 - 0.07283629) / 0.01802787   # +0.4%  tau1 > 0.07284
        - 0.003143706 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # -0.3%  e2 > 0.04447
        - 0.002784498 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 7.11886e-05   # -0.3%  N2 < 0.2233 and sum_zz_dr2 > 0.01166
        + 0.002245065 * max(0.0, Q.max_dr - 0.1117619) / 0.04033009   # +0.2%  max_dr > 0.1118
        + 0.001959116 * max(0.0, Q.sum_z_dr - 0.1019409) / 0.005400003   # +0.2%  sum_z_dr > 0.1019
        - 0.001222098 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.25 - Q.pt_6) / 1.048693   # -0.1%  N2 < 0.2233 and pt_6 < 62.25
        + 0.001106718 * max(0.0, 739.5 - Q.sum_pt) * max(0.0, 1.186019 - Q.sj3_mass2) / 36.67845   # +0.1%  sum_pt < 739.5 and sj3_mass2 < 1.186
        - 0.0009169656 * max(0.0, Q.mass - 88.15578) / 0.7232262   # -0.1%  mass > 88.16
        + 0.0007944779 * max(0.0, 0.2233283 - Q.N2) * max(0.0, -0.009352575 - Q.mean_phi) / 0.0001205535   # +0.1%  N2 < 0.2233 and mean_phi < -0.009353
        + 0.0006790359 * max(0.0, 76.6557 - Q.mass) * max(0.0, Q.mean_eta2 - 0.006395693) / 0.005490776   # +0.1%  mass < 76.66 and mean_eta2 > 0.006396
        - 0.0005837398 * max(0.0, 0.004811143 - Q.tau21_b2) / 0.0001640305   # -0.1%  tau21_b2 < 0.004811
        - 0.0004824205 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, -0.01753483 - Q.mean_phi) / 2.64217e-07   # -0.0%  lam2 < 0.0005373 and mean_phi < -0.01753
        - 0.0004524702 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 25.57812 - Q.pt_7) / 0.037942   # -0.0%  N2 < 0.2233 and pt_7 < 25.58
        - 0.0004330325 * max(0.0, 0.004811143 - Q.tau21_b2) * max(0.0, Q.sj3_mass1 - 0.8392664) / 9.493463e-05   # -0.0%  tau21_b2 < 0.004811 and sj3_mass1 > 0.8393
        - 0.0002940199 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # -0.0%  mean_eta > 0.02644
        - 0.0001868586 * max(0.0, 0.05119235 - Q.C2) * max(0.0, Q.mean_eta - 0.02644207) / 1.096062e-05   # -0.0%  C2 < 0.05119 and mean_eta > 0.02644
        + 0.0001795962 * max(0.0, 0.007639643 - Q.sum_z_dr2_top2) * max(0.0, -0.01790907 - Q.mean_eta) / 2.392698e-06   # +0.0%  sum_z_dr2_top2 < 0.00764 and mean_eta < -0.01791
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 13.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.17821 * (0.003395156
        + 0.1122098 * max(0.0, Q.log_sum_pt - 6.267538) / 0.2983143   # +11.2%  log_sum_pt > 6.268
        + 0.1048608 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # +10.5%  tau1 < 0.09539
        - 0.07178765 * max(0.0, 0.09041383 - Q.mass_over_sum_pt) / 0.03744004   # -7.2%  mass_over_sum_pt < 0.09041
        - 0.06985334 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -7.0%  sj3_dr_max < 0.3012
        - 0.05882281 * max(0.0, 37.15625 - Q.pt_7) / 5.803748   # -5.9%  pt_7 < 37.16
        + 0.05089553 * max(0.0, 0.04939969 - Q.z_7) / 0.007457505   # +5.1%  z_7 < 0.0494
        - 0.05082524 * max(0.0, Q.pt_7 - 23.21641) / 12.35308   # -5.1%  pt_7 > 23.22
        + 0.04693692 * max(0.0, 0.001653836 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 7.0272e-06   # +4.7%  sum_z_dr2 < 0.001654 and centroid_offset < 0.02355
        - 0.04105383 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -4.1%  log_sum_pt > 6.701
        + 0.03285529 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # +3.3%  sum_z_dr2 < 0.002635
        + 0.03249017 * max(0.0, 0.005284669 - Q.sum_zz_dr2) / 0.00223735   # +3.2%  sum_zz_dr2 < 0.005285
        - 0.02861635 * max(0.0, 0.002915531 - Q.sum_z_dr2_top3) / 0.001178811   # -2.9%  sum_z_dr2_top3 < 0.002916
        - 0.02734188 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -2.7%  LHA < 0.2161 and log_sum_pt < 6.804
        - 0.02346039 * max(0.0, 0.02383244 - Q.zdr_0) / 0.01109706   # -2.3%  zdr_0 < 0.02383
        - 0.02313809 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0004149607   # -2.3%  z_7 < 0.07149 and centroid_offset < 0.03117
        + 0.01958188 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 42.18339 - Q.mass_top3) / 0.6375506   # +2.0%  z_7 < 0.07149 and mass_top3 < 42.18
        + 0.01705379 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +1.7%  z_7 < 0.07149
        + 0.01628314 * max(0.0, 60.63098 - Q.mass) / 24.47895   # +1.6%  mass < 60.63
        + 0.01585512 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.0002302115 - Q.mean_phi2) / 3.738179e-06   # +1.6%  log_sum_pt > 6.701 and mean_phi2 < 0.0002302
        + 0.01479626 * max(0.0, 0.07870506 - Q.z_dr_0p1_0p2) / 0.04317253   # +1.5%  z_dr_0p1_0p2 < 0.07871
        + 0.01247292 * max(0.0, 60.63098 - Q.mass) * max(0.0, 3.885568 - Q.D2) / 50.41033   # +1.2%  mass < 60.63 and D2 < 3.886
        + 0.009951173 * max(0.0, 0.001653836 - Q.sum_z_dr2) / 0.0004289067   # +1.0%  sum_z_dr2 < 0.001654
        + 0.009501103 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +1.0%  z_7 < 0.02807
        + 0.0081333 * max(0.0, Q.sum_pt_top5 - 716.8828) / 29.12383   # +0.8%  sum_pt_top5 > 716.9
        - 0.007990702 * max(0.0, Q.log_sum_pt - 6.267538) * max(0.0, 0.03601074 - Q.abseta_4) / 0.004915608   # -0.8%  log_sum_pt > 6.268 and abseta_4 < 0.03601
        + 0.007504873 * max(0.0, 0.02886576 - Q.z_6) / 0.0008381782   # +0.8%  z_6 < 0.02887
        + 0.007500098 * max(0.0, 0.02383244 - Q.zdr_0) * max(0.0, Q.z_5 - 0.02818362) / 0.0004584623   # +0.8%  zdr_0 < 0.02383 and z_5 > 0.02818
        + 0.007460284 * max(0.0, 0.1879486 - Q.sj3_dr_max) / 0.05910331   # +0.7%  sj3_dr_max < 0.1879
        - 0.006438168 * max(0.0, 0.005284669 - Q.sum_zz_dr2) * max(0.0, Q.pt_6 - 27.57812) / 0.03091934   # -0.6%  sum_zz_dr2 < 0.005285 and pt_6 > 27.58
        + 0.005663239 * max(0.0, 0.09538712 - Q.tau1) * max(0.0, 0.4926918 - Q.planar_flow) / 0.008723222   # +0.6%  tau1 < 0.09539 and planar_flow < 0.4927
        + 0.005229656 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 9.030369e-05 - Q.mean_eta2) / 9.99897e-07   # +0.5%  log_sum_pt > 6.701 and mean_eta2 < 9.03e-05
        - 0.004815087 * max(0.0, 0.002915531 - Q.sum_z_dr2_top3) * max(0.0, 0.006646257 - Q.tau3) / 3.390069e-06   # -0.5%  sum_z_dr2_top3 < 0.002916 and tau3 < 0.006646
        + 0.004804924 * max(0.0, 0.005284669 - Q.sum_zz_dr2) * max(0.0, 0.01437952 - Q.centroid_offset) / 1.318015e-05   # +0.5%  sum_zz_dr2 < 0.005285 and centroid_offset < 0.01438
        - 0.004760837 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03532852 - Q.C3) / 0.0003034905   # -0.5%  z_7 < 0.07149 and C3 < 0.03533
        - 0.004134581 * max(0.0, 0.001653836 - Q.sum_z_dr2) * max(0.0, 0.01452637 - Q.absphi_0) / 3.722671e-06   # -0.4%  sum_z_dr2 < 0.001654 and absphi_0 < 0.01453
        - 0.004095671 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.0002302115 - Q.mean_phi2) / 4.941235e-07   # -0.4%  log_sum_pt > 6.896 and mean_phi2 < 0.0002302
        - 0.003803752 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # -0.4%  sj3_dr_max < 0.107
        - 0.003788679 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.00424745 - Q.mean_eta2) / 0.0001193758   # -0.4%  log_sum_pt > 6.701 and mean_eta2 < 0.004247
        + 0.00365088 * max(0.0, 9.121392e-05 - Q.sum_z_dr2) / 4.152703e-06   # +0.4%  sum_z_dr2 < 9.121e-05
        - 0.003187312 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # -0.3%  sum_z_dr < 0.007674
        - 0.002995648 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, Q.C2_b2 - 0.001109927) / 4.204538e-05   # -0.3%  z_7 < 0.07149 and C2_b2 > 0.00111
        - 0.002688609 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 501.625 - Q.sum_pt_top2) / 0.2299658   # -0.3%  z_7 < 0.0494 and sum_pt_top2 < 501.6
        + 0.002681215 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # +0.3%  z_5 < 0.02818
        + 0.002600532 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01402779 - Q.abseta_4) / 0.0001821359   # +0.3%  log_sum_pt > 6.701 and abseta_4 < 0.01403
        - 0.001732186 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.2%  log_sum_pt > 6.896
        + 0.001723735 * max(0.0, 0.02807091 - Q.z_7) * max(0.0, Q.pt_5 - 24.57812) / 0.009960011   # +0.2%  z_7 < 0.02807 and pt_5 > 24.58
        - 0.00083025 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, Q.eccentricity - 0.9781608) / 1.559876e-05   # -0.1%  log_sum_pt > 6.896 and eccentricity > 0.9782
        - 0.0004908251 * max(0.0, 0.02886576 - Q.z_6) * max(0.0, Q.phi_0 - 0.008850098) / 1.965886e-06   # -0.0%  z_6 < 0.02887 and phi_0 > 0.00885
        - 0.000460794 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, Q.sj3_pair_mass_max - 17.83163) / 0.05562116   # -0.0%  log_sum_pt > 6.896 and sj3_pair_mass_max > 17.83
        - 0.0001907044 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 68.43422 - Q.mass_top5) / 0.3465446   # -0.0%  z_7 < 0.0494 and mass_top5 < 68.43
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 27.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.59326 * (-0.003559105
        - 0.1424183 * max(0.0, 0.008678045 - Q.sum_z_dr2) / 0.004447347   # -14.2%  sum_z_dr2 < 0.008678
        - 0.1386678 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # -13.9%  lam1_plus_lam2 < 0.01324
        + 0.07483463 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +7.5%  tau1 < 0.1136
        + 0.0586188 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +5.9%  lam1 < 0.012
        + 0.05695486 * max(0.0, 0.1789613 - Q.sj3_dr_max) / 0.05394779   # +5.7%  sj3_dr_max < 0.179
        - 0.04753175 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # -4.8%  max_dr < 0.1452
        + 0.04701481 * max(0.0, 0.02415398 - Q.C2_b2) / 0.02132459   # +4.7%  C2_b2 < 0.02415
        - 0.0410896 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 1.341509   # -4.1%  mass < 60.63 and z_dr_0p2_0p4 < 0.05644
        + 0.03719285 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +3.7%  lam2 < 0.003408
        - 0.03200693 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -3.2%  lam2 < 0.001131
        + 0.02563193 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +2.6%  lam1 < 0.00733
        + 0.02474204 * max(0.0, 45.7571 - Q.mass) / 14.82409   # +2.5%  mass < 45.76
        + 0.0241884 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt) / 1.016213   # +2.4%  pt_6 < 41.22 and log_sum_pt < 6.767
        + 0.02286366 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +2.3%  centroid_offset > 0.008092
        + 0.02240226 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +2.2%  e3 < 0.0005117
        - 0.02207396 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -2.2%  pt_6 < 39.75
        + 0.01906191 * max(0.0, 0.0030133 - Q.sum_zz_dr2) / 0.001065944   # +1.9%  sum_zz_dr2 < 0.003013
        - 0.01745769 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757) / 0.07098489   # -1.7%  pt_6 < 41.22 and z_7 > 0.02321
        + 0.01710371 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +1.7%  pt_6 < 41.22
        + 0.01611633 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +1.6%  sj3_dr_max > 0.1879
        - 0.01592052 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # -1.6%  sj3_dr_max < 0.3012
        - 0.01525748 * max(0.0, Q.centroid_offset - 0.01837778) / 0.005523694   # -1.5%  centroid_offset > 0.01838
        + 0.01442779 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.003289169   # +1.4%  lam1 < 0.00733 and n_dr_0p2_0p4 < 1
        + 0.008430623 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2) / 2.179362e-05   # +0.8%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        + 0.007470233 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002658741   # +0.7%  lam2 < 0.003408 and n_dr_0_0p05 < 3
        - 0.00597416 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 0.2534037 - Q.planar_flow) / 0.0006566196   # -0.6%  lam1 < 0.012 and planar_flow < 0.2534
        + 0.005601268 * max(0.0, 60.63098 - Q.mass) / 24.47895   # +0.6%  mass < 60.63
        - 0.005524051 * max(0.0, Q.sj3_pair_mass_min - 4.501727) / 3.480268   # -0.6%  sj3_pair_mass_min > 4.502
        + 0.005098094 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 43.23531   # +0.5%  sum_pt < 615.9 and n_dr_0p2_0p4 < 2
        + 0.004652534 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.psi_0p1 - 0.4008925) / 0.003485257   # +0.5%  centroid_offset > 0.008092 and psi_0p1 > 0.4009
        - 0.003391436 * max(0.0, Q.eccentricity - 0.927072) / 0.03232992   # -0.3%  eccentricity > 0.9271
        + 0.002958272 * max(0.0, 29.90625 - Q.pt_6) / 1.385454   # +0.3%  pt_6 < 29.91
        - 0.002268017 * max(0.0, 615.875 - Q.sum_pt) / 30.07766   # -0.2%  sum_pt < 615.9
        - 0.00208405 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # -0.2%  sj3_dr_min > 0.1278
        + 0.002005446 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308) / 0.05438016   # +0.2%  centroid_offset > 0.008092 and sj3_pair_mass_min > 6.811
        - 0.00191514 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # -0.2%  mean_eta > 0.02644
        + 0.001418355 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 0.1343804 - Q.dr1_7) / 0.3834372   # +0.1%  pt_6 < 41.22 and dr1_7 < 0.1344
        + 0.001354636 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.M3 - 0.0782171) / 0.01402652   # +0.1%  pt_6 < 41.22 and M3 > 0.07822
        - 0.001116834 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.00433128 - Q.mean_phi2) / 7.021694e-06   # -0.1%  centroid_offset > 0.01838 and mean_phi2 < 0.004331
        - 0.0009956177 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 2.69383   # -0.1%  sj3_pair_mass_min > 4.502 and n_dr_0p2_0p4 > 1
        + 0.0009599081 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.02702951 - Q.C2) / 2.39871e-05   # +0.1%  centroid_offset > 0.01838 and C2 < 0.02703
        - 0.0009143157 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.02980347 - Q.eta_0) / 0.0002676158   # -0.1%  centroid_offset > 0.01838 and eta_0 < 0.0298
        - 0.0008295062 * max(0.0, 45.7571 - Q.mass) * max(0.0, Q.eta_3 - -0.00592804) / 0.1572446   # -0.1%  mass < 45.76 and eta_3 > -0.005928
        + 0.0005594046 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, Q.mean_eta - 0.02644207) / 1.54475e-06   # +0.1%  lam1 < 0.012 and mean_eta > 0.02644
        - 0.0004471214 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, Q.M3 - 0.06688759) / 9.396674e-06   # -0.0%  mean_eta > 0.02644 and M3 > 0.06689
        - 0.00040597 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, Q.mean_phi - 0.01251955) / 0.1451915   # -0.0%  sum_pt < 615.9 and mean_phi > 0.01252
        + 4.700937e-05 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 0.05101919 - Q.z_3) / 0.0001257894   # +0.0%  sum_pt < 615.9 and z_3 < 0.05102
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 38.48;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 38.48175 * (0.2160934
        - 0.09133745 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # -9.1%  lam1_plus_lam2 > 0.00559
        - 0.07594374 * max(0.0, Q.mass_over_sum_pt - 0.01109984) / 0.05101851   # -7.6%  mass_over_sum_pt > 0.0111
        - 0.07591996 * max(0.0, 0.08723651 - Q.sum_z_dr) / 0.03638569   # -7.6%  sum_z_dr < 0.08724
        - 0.06569759 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -6.6%  sum_z_dr2 > 0.00752
        - 0.06201231 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -6.2%  mass_over_sum_pt > 0.09041
        - 0.05785628 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -5.8%  sj2_dr < 0.1873
        + 0.05412142 * max(0.0, Q.sum_z_dr2 - 0.001653836) / 0.005220304   # +5.4%  sum_z_dr2 > 0.001654
        + 0.04807863 * max(0.0, Q.lam1_plus_lam2 - 0.008678045) / 0.002214537   # +4.8%  lam1_plus_lam2 > 0.008678
        + 0.04773635 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +4.8%  mass_over_sum_pt > 0.08475
        + 0.04655535 * max(0.0, Q.mass_over_sum_pt - 0.07269073) / 0.01344138   # +4.7%  mass_over_sum_pt > 0.07269
        + 0.0441395 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +4.4%  sj2_dr < 0.1592
        - 0.0306777 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -3.1%  e3 < 8.148e-05
        + 0.03051172 * max(0.0, Q.sum_z_dr2 - 0.01323868) / 0.001442684   # +3.1%  sum_z_dr2 > 0.01324
        - 0.02726701 * max(0.0, Q.sum_z_dr2 - 0.004372139) / 0.003638805   # -2.7%  sum_z_dr2 > 0.004372
        - 0.02576044 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -2.6%  lam1 < 0.008376
        + 0.02326334 * max(0.0, Q.mass - 36.22941) / 14.20528   # +2.3%  mass > 36.23
        + 0.02033722 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +2.0%  LHA < 0.3033
        - 0.01745127 * max(0.0, Q.mass_over_sum_pt - 0.1079857) / 0.005364315   # -1.7%  mass_over_sum_pt > 0.108
        - 0.01609215 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.004032342 - Q.C2_b2) / 2.736263e-05   # -1.6%  centroid_offset < 0.02077 and C2_b2 < 0.004032
        + 0.01426437 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.4%  tau1 < 0.05357
        + 0.01130748 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # +1.1%  sj2_dr < 0.1295
        - 0.01043405 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # -1.0%  e2 > 0.05028
        - 0.01001433 * max(0.0, 0.04081947 - Q.sum_z_dr) / 0.009176315   # -1.0%  sum_z_dr < 0.04082
        + 0.008221159 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) / 0.09532154   # +0.8%  z_dr_0p1_0p2 < 0.1586
        - 0.008192149 * max(0.0, 0.0009641429 - Q.sum_z_dr2) / 0.0002052288   # -0.8%  sum_z_dr2 < 0.0009641
        + 0.00819146 * max(0.0, Q.z_7 - 0.03243272) / 0.02180051   # +0.8%  z_7 > 0.03243
        - 0.006840743 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.7%  mass > 76.66
        + 0.005810748 * max(0.0, 0.08723651 - Q.sum_z_dr) * max(0.0, 0.004032342 - Q.C2_b2) / 0.0001217265   # +0.6%  sum_z_dr < 0.08724 and C2_b2 < 0.004032
        - 0.005245301 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -0.5%  centroid_offset < 0.02077
        + 0.004870094 * max(0.0, Q.sum_z_dr2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +0.5%  sum_z_dr2 > 0.004372 and planar_flow < 0.195
        - 0.004742809 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.5%  centroid_offset > 0.03776
        + 0.00419348 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # +0.4%  sd_rg > 0.2788
        + 0.003882988 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.02656143 - Q.tau21_b2) / 4.805781e-05   # +0.4%  centroid_offset < 0.02077 and tau21_b2 < 0.02656
        - 0.003762305 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -0.4%  e2 > 0.03556
        + 0.003704063 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sd_mass - 38.43971) / 1.25962   # +0.4%  planar_flow < 0.195 and sd_mass > 38.44
        - 0.003459672 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # -0.3%  sum_z_dr2 < 0.003563
        + 0.003333094 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, Q.sum_pt_top3 - 331.25) / 1.710819   # +0.3%  centroid_offset < 0.02077 and sum_pt_top3 > 331.2
        - 0.003268913 * max(0.0, Q.sum_z_dr2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9458207) / 2.125219e-05   # -0.3%  sum_z_dr2 > 0.01324 and eccentricity > 0.9458
        - 0.002763945 * max(0.0, 29.04219 - Q.pt_7) / 2.179984   # -0.3%  pt_7 < 29.04
        - 0.002742047 * max(0.0, 0.001101266 - Q.sum_zz_dr2) / 0.000308004   # -0.3%  sum_zz_dr2 < 0.001101
        + 0.002737258 * max(0.0, Q.centroid_offset - 0.03117077) / 0.002505019   # +0.3%  centroid_offset > 0.03117
        + 0.002425032 * max(0.0, 0.001056655 - Q.sum_z_dr2_top2) / 0.0003002514   # +0.2%  sum_z_dr2_top2 < 0.001057
        - 0.002200551 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) * max(0.0, 2.080881 - Q.N3) / 0.06089036   # -0.2%  z_dr_0p1_0p2 < 0.1586 and N3 < 2.081
        - 0.0008848094 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6) / 0.1690249   # -0.1%  planar_flow < 0.195 and pt_6 < 35.28
        - 0.0008209713 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.lam1_plus_lam2 - 0.007520088) / 0.0001455029   # -0.1%  planar_flow < 0.195 and lam1_plus_lam2 > 0.00752
        - 0.0007339344 * max(0.0, Q.centroid_offset - 0.03117077) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.00188116   # -0.1%  centroid_offset > 0.03117 and n_pt_above_50 > 4
        - 0.0001928186 * max(0.0, Q.centroid_offset - 0.03776099) * max(0.0, Q.pt_4 - 81.375) / 0.0001936307   # -0.0%  centroid_offset > 0.03776 and pt_4 > 81.38
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 12.38;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.38303 * (-0.03710792
        - 0.1084135 * max(0.0, 0.06108601 - Q.sum_z_dr) / 0.01868685   # -10.8%  sum_z_dr < 0.06109
        + 0.09577412 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0001901112   # +9.6%  sum_z_dr2 < 0.00502 and z_dr_0p2_0p4 < 0.101
        - 0.07688276 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # -7.7%  LHA < 0.1967
        + 0.06033818 * max(0.0, 0.005019719 - Q.sum_z_dr2) / 0.001903424   # +6.0%  sum_z_dr2 < 0.00502
        + 0.06013158 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 4.808955e-05   # +6.0%  tau1 < 0.05357 and lam1_plus_lam2 < 0.003563
        + 0.05271187 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +5.3%  sum_z_dr2 < 0.006679 and centroid_offset < 0.02355
        + 0.05124831 * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.001184249   # +5.1%  lam1_plus_lam2 < 0.003563
        - 0.04908344 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -4.9%  sj3_dr_max < 0.1426
        - 0.04693526 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.003562611 - Q.lam1_plus_lam2) / 0.02261167   # -4.7%  mass < 29.64 and lam1_plus_lam2 < 0.003563
        + 0.03744723 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.0003703114   # +3.7%  sj3_dr_max < 0.1986 and sum_z_dr2 < 0.006679
        + 0.03253337 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.0001947983 - Q.lam2) / 2.869243e-06   # +3.3%  sum_z_dr < 0.06109 and lam2 < 0.0001948
        + 0.0318483 * max(0.0, 0.0001330621 - Q.lam2) * max(0.0, 4.721224 - Q.D2_b2) / 0.0002340602   # +3.2%  lam2 < 0.0001331 and D2_b2 < 4.721
        - 0.03135234 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -3.1%  log_sum_pt > 6.701
        - 0.02382151 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.0001947983 - Q.lam2) / 0.0007178522   # -2.4%  mass < 21.78 and lam2 < 0.0001948
        - 0.02314773 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -2.3%  sum_z_dr2 < 0.00502 and centroid_offset > 0.00679
        - 0.02085962 * max(0.0, 49.6681 - Q.mass) / 17.06893   # -2.1%  mass < 49.67
        - 0.01653458 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # -1.7%  tau1 < 0.05357
        + 0.01575103 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) / 0.00722828   # +1.6%  mass_over_sum_pt < 0.03319
        + 0.01545829 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +1.5%  sum_pt_top5 > 687.4
        - 0.01541961 * max(0.0, 0.0001330621 - Q.lam2) / 6.711676e-05   # -1.5%  lam2 < 0.0001331
        - 0.0150118 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.001011075   # -1.5%  sum_z_dr < 0.06109 and z_dr_0p2_0p4 < 0.05644
        + 0.01360867 * max(0.0, 29.6447 - Q.mass) * max(0.0, 1.762929e-06 - Q.e3) / 9.656157e-06   # +1.4%  mass < 29.64 and e3 < 1.763e-06
        - 0.01279731 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 4.721224 - Q.D2_b2) / 0.04403777   # -1.3%  tau1 < 0.05357 and D2_b2 < 4.721
        - 0.01234207 * max(0.0, 0.005019719 - Q.sum_z_dr2) * max(0.0, 43.5 - Q.pt_7) / 0.02255472   # -1.2%  sum_z_dr2 < 0.00502 and pt_7 < 43.5
        + 0.01199447 * max(0.0, 49.6681 - Q.mass) * max(0.0, 0.0001330621 - Q.lam2) / 0.001509268   # +1.2%  mass < 49.67 and lam2 < 0.0001331
        - 0.01090335 * max(0.0, 0.006679471 - Q.sum_z_dr2) / 0.00293624   # -1.1%  sum_z_dr2 < 0.006679
        - 0.008870285 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, 4.721224 - Q.D2_b2) / 0.008595177   # -0.9%  sum_z_dr2 < 0.006679 and D2_b2 < 4.721
        + 0.007651956 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 48.71875 - Q.pt_7) / 0.776931   # +0.8%  log_sum_pt > 6.701 and pt_7 < 48.72
        + 0.005696499 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +0.6%  sj3_dr_max < 0.1986
        + 0.005474735 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.pt_7 - 15.55391) / 0.4153091   # +0.5%  LHA < 0.1967 and pt_7 > 15.55
        - 0.005281308 * max(0.0, 0.001653836 - Q.sum_z_dr2) / 0.0004289067   # -0.5%  sum_z_dr2 < 0.001654
        - 0.004350519 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.2097233   # -0.4%  mass < 29.64 and D2_b2 < 0.7166
        + 0.003518484 * max(0.0, 0.06108601 - Q.sum_z_dr) * max(0.0, Q.lam1_plus_lam2 - 0.0003193707) / 1.048473e-05   # +0.4%  sum_z_dr < 0.06109 and lam1_plus_lam2 > 0.0003194
        + 0.003399988 * max(0.0, 0.006679471 - Q.sum_z_dr2) * max(0.0, Q.phi_0 - -0.04013062) / 0.0001189753   # +0.3%  sum_z_dr2 < 0.006679 and phi_0 > -0.04013
        - 0.003143434 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 2.955458e-05 - Q.e3) / 8.40609e-07   # -0.3%  log_sum_pt > 6.701 and e3 < 2.955e-05
        + 0.002420175 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.mean_phi - -0.0008556753) / 5.44837e-05   # +0.2%  LHA < 0.1967 and mean_phi > -0.0008557
        + 0.002389266 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 3.10694e-07 - Q.e3) / 3.648373e-06   # +0.2%  sum_pt_top5 > 687.4 and e3 < 3.107e-07
        + 0.002278389 * max(0.0, 0.03448406 - Q.z_6) / 0.001525592   # +0.2%  z_6 < 0.03448
        + 0.001586058 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.07002129   # +0.2%  mass < 21.78 and D2_b2 < 0.7166
        - 0.001065595 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.zdr_0 - 0.007798268) / 0.0001443534   # -0.1%  log_sum_pt > 6.701 and zdr_0 > 0.007798
        - 0.0005230007 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 8.0 - Q.n_pt_above_10) / 7.731645   # -0.1%  sum_pt_top5 > 687.4 and n_pt_above_10 < 8
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 28.05;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.05176 * (-0.06849083
        + 0.1309897 * max(0.0, 0.00609665 - Q.sum_z_dr2) / 0.002544039   # +13.1%  sum_z_dr2 < 0.006097
        - 0.06751856 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -6.8%  sj3_dr_max < 0.1426
        + 0.06509286 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +6.5%  sj3_dr_max < 0.2134
        + 0.06108134 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.2973626   # +6.1%  mass < 53.33 and centroid_offset < 0.02686
        + 0.05831925 * max(0.0, 0.003562611 - Q.sum_z_dr2) / 0.001184249   # +5.8%  sum_z_dr2 < 0.003563
        - 0.05583238 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -5.6%  lam1 < 0.005954
        + 0.04473783 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +4.5%  centroid_offset < 0.01838
        - 0.04432416 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -4.4%  mass_over_sum_pt < 0.07269
        + 0.04313065 * max(0.0, Q.lam1 - 0.005433361) / 0.00267714   # +4.3%  lam1 > 0.005433
        - 0.04234702 * max(0.0, Q.lam1 - 0.003377388) / 0.003672352   # -4.2%  lam1 > 0.003377
        + 0.0420607 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +4.2%  sum_pt < 988.4
        + 0.03465618 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +3.5%  max_dr < 0.1118
        - 0.03004318 * max(0.0, 53.33237 - Q.mass) / 19.36004   # -3.0%  mass < 53.33
        - 0.02385382 * max(0.0, 2.371297e-05 - Q.e3) / 1.105328e-05   # -2.4%  e3 < 2.371e-05
        + 0.02094499 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +2.1%  e2 > 0.04447
        - 0.01914131 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -1.9%  e2 > 0.03556
        - 0.01794254 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) / 6.087676   # -1.8%  sj3_pair_mass_max < 24.23
        + 0.01713535 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt) / 5.175436   # +1.7%  mass < 53.33 and log_sum_pt < 6.843
        - 0.01702042 * max(0.0, 0.05464922 - Q.sum_z_dr) / 0.01532978   # -1.7%  sum_z_dr < 0.05465
        - 0.0149838 * max(0.0, 0.0284695 - Q.C3) / 0.008714909   # -1.5%  C3 < 0.02847
        - 0.01327984 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1) / 0.1931178   # -1.3%  centroid_offset < 0.01838 and pt_1 < 159.2
        - 0.01317597 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.008712838   # -1.3%  centroid_offset < 0.01838 and n_for_90pct > 5
        - 0.01157062 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266) / 0.0005312449   # -1.2%  lam1 > 0.005433 and eccentricity > 0.7117
        + 0.01077656 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.0002201438   # +1.1%  sum_z_dr2 < 0.006097 and planar_flow < 0.3221
        + 0.008543574 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # +0.9%  tau1 < 0.0437
        - 0.007883332 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # -0.8%  e3 > 8.148e-05
        + 0.007810667 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd) / 0.0002178654   # +0.8%  centroid_offset < 0.01838 and z_2nd < 0.2056
        - 0.007667665 * max(0.0, Q.sum_z_dr - 0.0717028) / 0.01201981   # -0.8%  sum_z_dr > 0.0717
        + 0.006807612 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.07073975 - Q.abseta_1) / 0.0003304963   # +0.7%  centroid_offset < 0.01838 and abseta_1 < 0.07074
        + 0.005668011 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +0.6%  e3 > 0.0001869
        - 0.004919299 * max(0.0, 0.006292091 - Q.zdr_0) / 0.001006321   # -0.5%  zdr_0 < 0.006292
        - 0.004848658 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255) / 0.000277001   # -0.5%  tau1 < 0.0437 and eccentricity > 0.9031
        + 0.004632401 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.5%  lam2 > 0.001131
        - 0.003615652 * max(0.0, 0.1070199 - Q.sj3_dr_max) / 0.02398888   # -0.4%  sj3_dr_max < 0.107
        + 0.003595026 * max(0.0, 0.0001721983 - Q.lam1_plus_lam2) / 1.441896e-05   # +0.4%  lam1_plus_lam2 < 0.0001722
        + 0.003544852 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.4%  n_dr_0p2_0p4 > 1
        - 0.003426005 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.tau4 - 0.001224244) / 1.099241e-05   # -0.3%  centroid_offset < 0.01838 and tau4 > 0.001224
        - 0.003322566 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, 0.002834884 - Q.mean_phi) / 1.344572e-05   # -0.3%  sum_z_dr2 < 0.006097 and mean_phi < 0.002835
        - 0.002989466 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, Q.z_5 - 0.03672711) / 0.0004195788   # -0.3%  sum_z_dr < 0.05465 and z_5 > 0.03673
        - 0.002981562 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.3%  sum_pt_top5 > 840
        + 0.002521699 * max(0.0, 559.6875 - Q.sum_pt) / 16.18216   # +0.3%  sum_pt < 559.7
        + 0.002345784 * max(0.0, 2.371297e-05 - Q.e3) * max(0.0, Q.eccentricity - 0.8319502) / 8.425435e-07   # +0.2%  e3 < 2.371e-05 and eccentricity > 0.832
        + 0.002027072 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.2%  log_sum_pt > 6.896
        + 0.001930057 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.6875) / 0.3914557   # +0.2%  sj3_dr_max < 0.1426 and pt_6 > 33.69
        - 0.001552774 * max(0.0, 0.002316125 - Q.centroid_offset) / 9.872932e-05   # -0.2%  centroid_offset < 0.002316
        - 0.001120371 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2) / 0.000627491   # -0.1%  log_sum_pt > 6.896 and D2_b2 < 0.7166
        - 0.001087547 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 0.2089872 - Q.sj3_dr_min) / 0.01108383   # -0.1%  n_dr_0p2_0p4 > 1 and sj3_dr_min < 0.209
        + 0.00104078 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, -0.01275329 - Q.mean_phi) / 1.158446e-05   # +0.1%  tau1 < 0.0437 and mean_phi < -0.01275
        - 0.0009900571 * Q.z_dr_0p2_0p4 / 0.02920532   # -0.1%  z_dr_0p2_0p4
        + 0.0008524801 * max(0.0, 0.05464922 - Q.sum_z_dr) * max(0.0, 0.02656143 - Q.tau21_b2) / 1.23798e-05   # +0.1%  sum_z_dr < 0.05465 and tau21_b2 < 0.02656
        + 0.0007849694 * max(0.0, 31.125 - Q.pt_4) / 0.3140792   # +0.1%  pt_4 < 31.12
        - 0.0005741805 * max(0.0, 0.00609665 - Q.sum_z_dr2) * max(0.0, Q.mean_phi - 0.02612796) / 3.162575e-07   # -0.1%  sum_z_dr2 < 0.006097 and mean_phi > 0.02613
        + 0.0005192732 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.mean_phi - 0.02612796) / 2.994783e-06   # +0.1%  tau1 < 0.0437 and mean_phi > 0.02613
        - 0.0004375872 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, Q.abseta_1 - 0.03625488) / 8.135786e-06   # -0.0%  sj3_dr_max < 0.107 and abseta_1 > 0.03625
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 7.516;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.515556 * (0.2198727
        + 0.1523994 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # +15.2%  sum_z_dr2 > 0.00752
        - 0.09631945 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # -9.6%  lam1 > 0.00733
        - 0.09056903 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -9.1%  lam1 < 0.004184
        + 0.08018666 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # +8.0%  tau1 > 0.05357
        + 0.0705501 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +7.1%  mass < 76.66
        - 0.0630432 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -6.3%  LHA > 0.3033
        + 0.04466165 * max(0.0, Q.lam2 - 0.0003061234) / 0.0004216599   # +4.5%  lam2 > 0.0003061
        - 0.04075943 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -4.1%  lam1 < 0.001504
        + 0.03996649 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +4.0%  e3 < 8.148e-05
        + 0.03397651 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1) / 0.002082998   # +3.4%  zdr_0 < 0.02118 and z_dr_0p05_0p1 < 0.2919
        - 0.0318272 * max(0.0, 45.75 - Q.pt_7) / 12.14632   # -3.2%  pt_7 < 45.75
        + 0.03023797 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # +3.0%  sum_z_dr2_top3 < 0.002152
        - 0.02889789 * max(0.0, 0.002635418 - Q.sum_z_dr2) / 0.0007941346   # -2.9%  sum_z_dr2 < 0.002635
        - 0.01962731 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.planar_flow - 0.04505724) / 0.0002791469   # -2.0%  lam2 > 0.0003061 and planar_flow > 0.04506
        - 0.01903654 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -1.9%  zdr_0 < 0.02118
        + 0.01882223 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.002471 - Q.D2) / 1.812028   # +1.9%  pt_7 < 45.75 and D2 < 1.002
        - 0.01810451 * max(0.0, 0.06473447 - Q.z_7) / 0.01631636   # -1.8%  z_7 < 0.06473
        - 0.0157113 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -1.6%  lam1 < 0.005954
        + 0.01531676 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # +1.5%  sj3_pair_mass_min > 11.05
        - 0.01079701 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 6.572938 - Q.log_sum_pt) / 1.257183   # -1.1%  pt_7 < 45.75 and log_sum_pt < 6.573
        + 0.008936132 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.01258764) / 3.400017e-05   # +0.9%  zdr_0 < 0.02118 and centroid_offset > 0.01259
        - 0.008049884 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -0.8%  e3 > 3.892e-05
        + 0.007974867 * max(0.0, Q.C2_b2 - 0.009032972) / 0.001703043   # +0.8%  C2_b2 > 0.009033
        - 0.007517542 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1797097) / 5.905455e-07   # -0.8%  e3 < 8.148e-05 and sj3_dr23 > 0.1797
        - 0.007419234 * max(0.0, 0.02563286 - Q.M2) / 0.001872375   # -0.7%  M2 < 0.02563
        + 0.007221587 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +0.7%  sj3_dr_min > 0.1278
        - 0.006726136 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.3658817 - Q.z_dr_0_0p05) / 0.002463327   # -0.7%  sj3_dr_min > 0.1278 and z_dr_0_0p05 < 0.3659
        - 0.006716213 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.7%  sum_pt > 988.4
        - 0.006571804 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2) / 0.0002929492   # -0.7%  lam1 > 0.00733 and D2_b2 < 0.3809
        - 0.003460324 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -0.3%  lam2 > 0.003408
        + 0.003223318 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.sj3_pairmax_over_m - 0.6745796) / 3.382029e-05   # +0.3%  lam2 > 0.0003061 and sj3_pairmax_over_m > 0.6746
        + 0.002246905 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.dr_7 - 0.1078355) / 8.813877e-05   # +0.2%  zdr_0 < 0.02118 and dr_7 > 0.1078
        - 0.001639701 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, Q.D2_b2 - 1.129616) / 0.0002590844   # -0.2%  lam1 > 0.00733 and D2_b2 > 1.13
        + 0.001485671 * max(0.0, Q.sj3_pair_mass_min - 11.051) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 3.623232   # +0.1%  sj3_pair_mass_min > 11.05 and n_dr_0p2_0p4 > 0
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 27.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.23239 * (-0.06274246
        + 0.2211776 * max(0.0, 0.008678045 - Q.lam1_plus_lam2) / 0.004447348   # +22.1%  lam1_plus_lam2 < 0.008678
        + 0.1194904 * max(0.0, 0.01323868 - Q.sum_z_dr2) / 0.008236125   # +11.9%  sum_z_dr2 < 0.01324
        - 0.07393227 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -7.4%  lam1 < 0.008376
        - 0.07141393 * max(0.0, 0.169029 - Q.sj3_dr_max) / 0.04876366   # -7.1%  sj3_dr_max < 0.169
        - 0.06560717 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -6.6%  sum_zz_dr2 < 0.008169
        + 0.0608009 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +6.1%  centroid_offset < 0.03776
        + 0.04685191 * max(0.0, 0.2623172 - Q.sj3_dr_max) / 0.1129006   # +4.7%  sj3_dr_max < 0.2623
        - 0.04183876 * max(0.0, 0.0717028 - Q.sum_z_dr) / 0.02501835   # -4.2%  sum_z_dr < 0.0717
        - 0.0348068 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -3.5%  mass < 69.61
        + 0.03306865 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +3.3%  z_7 > 0.01686
        - 0.03169956 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # -3.2%  tau1 < 0.09539
        + 0.02319639 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +2.3%  sj3_dr_max < 0.1426
        - 0.02185881 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt) / 3.513015   # -2.2%  centroid_offset < 0.03776 and sum_pt < 901.6
        + 0.01955888 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +2.0%  planar_flow < 0.2534
        - 0.01641751 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -1.6%  lam1 < 0.00733
        + 0.0121541 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # +1.2%  max_dr < 0.1452
        + 0.01091515 * max(0.0, 15.45403 - Q.mass) / 2.385786   # +1.1%  mass < 15.45
        - 0.01021237 * max(0.0, 0.02054282 - Q.sum_z_dr) / 0.002592388   # -1.0%  sum_z_dr < 0.02054
        - 0.007786255 * max(0.0, 0.004372139 - Q.sum_z_dr2) / 0.00156571   # -0.8%  sum_z_dr2 < 0.004372
        - 0.007268212 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.7%  planar_flow < 0.2534 and sum_pt < 840
        - 0.007204958 * max(0.0, Q.pt_7 - 29.04219) / 7.785987   # -0.7%  pt_7 > 29.04
        - 0.007106731 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 0.0345459 - Q.absphi_0) / 0.0003803849   # -0.7%  centroid_offset < 0.03776 and absphi_0 < 0.03455
        - 0.006730425 * max(0.0, 0.01437952 - Q.centroid_offset) / 0.004306483   # -0.7%  centroid_offset < 0.01438
        - 0.005970127 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.5502779 - Q.tau32) / 0.0001096908   # -0.6%  centroid_offset > 0.0499 and tau32 < 0.5503
        - 0.005893898 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.6%  centroid_offset > 0.0499
        - 0.005198609 * max(0.0, 0.03578649 - Q.C2) / 0.01504824   # -0.5%  C2 < 0.03579
        + 0.004873364 * max(0.0, Q.sj2_dr - 0.1778793) / 0.03082064   # +0.5%  sj2_dr > 0.1779
        + 0.00446572 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.4%  sj2_dr > 0.2688
        + 0.003964168 * max(0.0, Q.eccentricity - 0.9884745) / 0.001961982   # +0.4%  eccentricity > 0.9885
        - 0.00328124 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 37.15625 - Q.pt_7) / 0.5537514   # -0.3%  planar_flow < 0.2534 and pt_7 < 37.16
        - 0.002588967 * max(0.0, 0.0005049491 - Q.lam1) / 8.739619e-05   # -0.3%  lam1 < 0.0005049
        + 0.002343059 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # +0.2%  LHA < 0.3127
        - 0.001908419 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.03601074 - Q.abseta_4) / 8.179649e-05   # -0.2%  centroid_offset < 0.01438 and abseta_4 < 0.03601
        + 0.001743965 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0007565864   # +0.2%  centroid_offset < 0.01438 and D2_b2 < 0.5327
        - 0.001714109 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 1.050302e-05 - Q.e3) / 1.771164e-07   # -0.2%  planar_flow < 0.2534 and e3 < 1.05e-05
        - 0.001550652 * max(0.0, Q.max_pair_mass - 33.3761) / 0.9279851   # -0.2%  max_pair_mass > 33.38
        - 0.001295263 * max(0.0, Q.eccentricity - 0.9884745) * max(0.0, 2.771069 - Q.sj2_mass2) / 0.00304207   # -0.1%  eccentricity > 0.9885 and sj2_mass2 < 2.771
        - 0.001158565 * max(0.0, Q.eccentricity - 0.9884745) * max(0.0, 0.02130127 - Q.eta_0) / 6.650689e-05   # -0.1%  eccentricity > 0.9885 and eta_0 < 0.0213
        + 0.000615359 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # +0.1%  centroid_offset > 0.0499 and D2 < 2.844
        + 0.0003367673 * max(0.0, 49.6681 - Q.mass) * max(0.0, Q.dr1_7 - 0.1792439) / 0.02583476   # +0.0%  mass < 49.67 and dr1_7 > 0.1792
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.5517;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5516571 * (-1.290992
        + 0.5220396 * max(0.0, Q.sum_z_dr2 - 0.01882765) / 0.0007605369   # +52.2%  sum_z_dr2 > 0.01883
        - 0.2290936 * max(0.0, Q.mass_over_sum_pt - 0.1309286) / 0.002541185   # -22.9%  mass_over_sum_pt > 0.1309
        + 0.08002142 * max(0.0, Q.mass - 88.15578) / 0.7232262   # +8.0%  mass > 88.16
        - 0.06286397 * max(0.0, Q.sum_z_dr2_top2 - 0.01403324) / 0.001129575   # -6.3%  sum_z_dr2_top2 > 0.01403
        - 0.0575892 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, 53.4375 - Q.pt_7) / 0.01439677   # -5.8%  sum_z_dr2 > 0.01883 and pt_7 < 53.44
        - 0.02624855 * max(0.0, Q.zdr_0 - 0.03981924) / 0.0005695785   # -2.6%  zdr_0 > 0.03982
        + 0.02214365 * max(0.0, Q.sum_z_dr2 - 0.01882765) * max(0.0, Q.lam2 - 0.000537286) / 2.828867e-06   # +2.2%  sum_z_dr2 > 0.01883 and lam2 > 0.0005373
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 16.9;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.90463 * (0.01740251
        + 0.3255279 * max(0.0, 0.1484084 - Q.sum_z_dr) / 0.0905999   # +32.6%  sum_z_dr < 0.1484
        - 0.1164682 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -11.6%  e2 < 0.08001
        + 0.08766156 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +8.8%  lam1 < 0.01643
        - 0.08054098 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -8.1%  sum_z_dr < 0.1484 and log_sum_pt < 6.804
        - 0.04901097 * max(0.0, 0.007520088 - Q.lam1_plus_lam2) / 0.003544991   # -4.9%  lam1_plus_lam2 < 0.00752
        + 0.04644693 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +4.6%  lam1_plus_lam2 < 0.01324
        - 0.04265342 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 8.324263e-07   # -4.3%  e3 < 5.335e-05 and centroid_offset < 0.03776
        - 0.03804024 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -3.8%  sum_z_dr < 0.1484 and pt_7 < 38.53
        + 0.03259909 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7) / 781.5808   # +3.3%  sum_pt_top5 > 658.1 and pt_7 < 40.04
        + 0.03230083 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # +3.2%  e3 < 5.335e-05
        + 0.0265694 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +2.7%  lam1 < 0.006507
        - 0.02373642 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.000537286 - Q.lam2) / 4.021081e-05   # -2.4%  sum_z_dr < 0.1484 and lam2 < 0.0005373
        - 0.01327587 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7) / 0.06765006   # -1.3%  pt_6 < 31.91 and z_7 < 0.05861
        - 0.01113709 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # -1.1%  z_7 < 0.02807
        + 0.01061999 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 31.90625 - Q.pt_6) / 134.9952   # +1.1%  sum_pt_top5 > 791.1 and pt_6 < 31.91
        + 0.007713349 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.z_7 - 0.06164517) / 0.0003465887   # +0.8%  sum_z_dr < 0.1484 and z_7 > 0.06165
        + 0.006204202 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, 0.07474969 - Q.M3) / 0.0006773733   # +0.6%  sum_z_dr < 0.1484 and M3 < 0.07475
        + 0.005506062 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.tau2 - 0.008780509) / 0.000269945   # +0.6%  sum_z_dr < 0.1484 and tau2 > 0.008781
        - 0.004776613 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # -0.5%  z_5 < 0.02818
        - 0.004725663 * max(0.0, Q.mass - 49.6681) / 7.721594   # -0.5%  mass > 49.67
        - 0.004533555 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6) / 104.494   # -0.5%  sum_pt > 988.4 and pt_6 < 62.25
        - 0.004059709 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # -0.4%  log_sum_pt > 6.503
        + 0.00387278 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.D3 - 0.2213841) / 3.794981e-05   # +0.4%  e3 < 5.335e-05 and D3 > 0.2214
        - 0.003792289 * max(0.0, Q.sj3_dr23 - 0.1974628) / 0.02478939   # -0.4%  sj3_dr23 > 0.1975
        + 0.003535011 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +0.4%  sum_pt_top5 > 791.1
        - 0.002739433 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.3%  z_6 < 0.0216
        - 0.002615167 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.02320757) / 0.2899084   # -0.3%  sum_pt_top5 > 658.1 and z_7 > 0.02321
        - 0.002126965 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.zdr_7 - 0.0007928864) / 0.0001818422   # -0.2%  log_sum_pt > 6.503 and zdr_7 > 0.0007929
        - 0.002003354 * max(0.0, Q.mass - 49.6681) * max(0.0, 0.4684459 - Q.z_dr_0p1_0p2) / 1.50172   # -0.2%  mass > 49.67 and z_dr_0p1_0p2 < 0.4684
        - 0.001637029 * max(0.0, Q.pt_7 - 48.71875) / 0.6759439   # -0.2%  pt_7 > 48.72
        - 0.001612917 * max(0.0, 0.007673833 - Q.sum_z_dr) / 0.000231517   # -0.2%  sum_z_dr < 0.007674
        - 0.001283252 * max(0.0, 0.1484084 - Q.sum_z_dr) * max(0.0, Q.sj2_mass1 - 31.78116) / 0.01215768   # -0.1%  sum_z_dr < 0.1484 and sj2_mass1 > 31.78
        + 0.0006738039 * max(0.0, Q.log_sum_pt - 6.502799) * max(0.0, Q.pair_mass_0_7 - 10.2219) / 0.2907061   # +0.1%  log_sum_pt > 6.503 and pair_mass_0_7 > 10.22
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 33.04;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.04369 * (-0.02432869
        - 0.1189832 * max(0.0, Q.lam1_plus_lam2 - 0.006679471) / 0.002702003   # -11.9%  lam1_plus_lam2 > 0.006679
        + 0.06060988 * max(0.0, Q.sum_z_dr - 0.02689598) / 0.03613842   # +6.1%  sum_z_dr > 0.0269
        - 0.05839408 * max(0.0, Q.sum_z_dr2 - 0.007520088) / 0.002470136   # -5.8%  sum_z_dr2 > 0.00752
        - 0.05765627 * max(0.0, Q.e2 - 0.01655442) / 0.01596508   # -5.8%  e2 > 0.01655
        - 0.05692846 * max(0.0, Q.sum_z_dr - 0.08723651) / 0.007853436   # -5.7%  sum_z_dr > 0.08724
        + 0.05117983 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +5.1%  mass_over_sum_pt > 0.08475
        + 0.0463376 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # +4.6%  sj2_dr > 0.1592
        + 0.04017389 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +4.0%  mass_over_sum_pt > 0.07992
        + 0.03906665 * max(0.0, Q.lam1_plus_lam2 - 0.003562611) / 0.004066872   # +3.9%  lam1_plus_lam2 > 0.003563
        + 0.03821935 * max(0.0, Q.sum_zz_dr2 - 0.002074109) / 0.004484017   # +3.8%  sum_zz_dr2 > 0.002074
        - 0.03771187 * max(0.0, 74.57663 - Q.sd_mass) / 42.04056   # -3.8%  sd_mass < 74.58
        + 0.03566924 * max(0.0, Q.sum_z_dr - 0.04081947) / 0.0270611   # +3.6%  sum_z_dr > 0.04082
        - 0.03077547 * max(0.0, Q.sum_zz_dr2 - 0.0030133) / 0.003940214   # -3.1%  sum_zz_dr2 > 0.003013
        + 0.02423135 * max(0.0, 49.91626 - Q.sd_mass) / 22.60916   # +2.4%  sd_mass < 49.92
        - 0.02262115 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # -2.3%  sj2_dr > 0.1295
        + 0.0215432 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +2.2%  e2 > 0.05028
        - 0.01859348 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -1.9%  mass_over_sum_pt > 0.09041
        + 0.01841664 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +1.8%  LHA > 0.3033
        - 0.01537785 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # -1.5%  e3 < 5.335e-05
        - 0.01500602 * max(0.0, 0.004032342 - Q.C2_b2) / 0.002883542   # -1.5%  C2_b2 < 0.004032
        + 0.01449517 * max(0.0, Q.sum_z_dr2 - 0.008678045) / 0.002214537   # +1.4%  sum_z_dr2 > 0.008678
        - 0.01432878 * max(0.0, Q.sj2_dr - 0.06154135) / 0.1002688   # -1.4%  sj2_dr > 0.06154
        + 0.01405625 * max(0.0, Q.lam1_plus_lam2 - 0.005590289) / 0.003084256   # +1.4%  lam1_plus_lam2 > 0.00559
        - 0.01257103 * max(0.0, Q.sum_zz_dr2 - 0.01165737) / 0.001433269   # -1.3%  sum_zz_dr2 > 0.01166
        + 0.01047534 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # +1.0%  planar_flow < 0.1115 and lam1 < 0.01643
        - 0.009345612 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # -0.9%  z_dr_0p05_0p1 > 0.7509
        + 0.009321456 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +0.9%  e2 > 0.04111
        - 0.008908256 * max(0.0, Q.sj2_dr - 0.2001708) / 0.02317349   # -0.9%  sj2_dr > 0.2002
        + 0.008832783 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # +0.9%  lam2 < 0.0005373
        - 0.00883038 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -0.9%  planar_flow < 0.1115 and lam1 < 0.00733
        - 0.008735249 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.9%  centroid_offset > 0.0499
        + 0.00851981 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067297   # +0.9%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        - 0.007857987 * max(0.0, Q.psi_0p1 - 0.7143804) / 0.1805043   # -0.8%  psi_0p1 > 0.7144
        + 0.007407323 * max(0.0, 5.0 - Q.n_dr_0_0p05) / 1.94322   # +0.7%  n_dr_0_0p05 < 5
        + 0.007053429 * max(0.0, Q.psi_0p1 - 0.8155839) / 0.1042073   # +0.7%  psi_0p1 > 0.8156
        - 0.005278717 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.5%  psi_0p1 > 0.9761
        + 0.004545491 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.3724174   # +0.5%  z_dr_0p05_0p1 < 0.5883
        + 0.004136089 * max(0.0, Q.sum_z_dr - 0.08065885) / 0.009330634   # +0.4%  sum_z_dr > 0.08066
        - 0.003641529 * max(0.0, Q.centroid_offset - 0.02685622) / 0.003254849   # -0.4%  centroid_offset > 0.02686
        + 0.003318671 * max(0.0, 0.1115136 - Q.planar_flow) / 0.03343187   # +0.3%  planar_flow < 0.1115
        - 0.00321146 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.3%  planar_flow < 0.1115 and centroid_offset < 0.01838
        + 0.003179509 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 3.351526e-05   # +0.3%  planar_flow < 0.1115 and lam1_plus_lam2 < 0.006097
        - 0.002927998 * max(0.0, Q.sum_zz_dr2 - 0.002074109) * max(0.0, 0.1872617 - Q.sj2_dr) / 2.014151e-05   # -0.3%  sum_zz_dr2 > 0.002074 and sj2_dr < 0.1873
        + 0.002586596 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1629004) / 4.043869e-07   # +0.3%  e3 < 5.335e-05 and sj3_dr23 > 0.1629
        - 0.002210677 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.1058993 - Q.z_5) / 0.0007616364   # -0.2%  sj2_dr > 0.2002 and z_5 < 0.1059
        - 0.002058899 * max(0.0, Q.mass - 69.61135) / 2.406702   # -0.2%  mass > 69.61
        - 0.002058007 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 658.125 - Q.sum_pt_top5) / 2.950014   # -0.2%  planar_flow < 0.1115 and sum_pt_top5 < 658.1
        - 0.001679644 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -0.2%  e2 > 0.03556
        + 0.0005695436 * max(0.0, Q.lam1_plus_lam2 - 0.01323868) / 0.001442684   # +0.1%  lam1_plus_lam2 > 0.01324
        + 0.0003628607 * max(0.0, 74.57663 - Q.sd_mass) * max(0.0, Q.sj2_mass2 - 3.697444) / 7.806142   # +0.0%  sd_mass < 74.58 and sj2_mass2 > 3.697
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 36.04;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 36.03662 * (-0.009012786
        + 0.1335107 * max(0.0, 0.01323868 - Q.lam1_plus_lam2) / 0.008236124   # +13.4%  lam1_plus_lam2 < 0.01324
        - 0.08395473 * max(0.0, 0.007520088 - Q.sum_z_dr2) / 0.00354499   # -8.4%  sum_z_dr2 < 0.00752
        + 0.07877442 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +7.9%  e2 < 0.04111
        + 0.07382086 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +7.4%  sj2_dr < 0.1592
        - 0.07331937 * max(0.0, 0.008168571 - Q.sum_zz_dr2) / 0.004294747   # -7.3%  sum_zz_dr2 < 0.008169
        + 0.06417108 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +6.4%  mass_over_sum_pt_sq < 0.007183
        - 0.06053655 * max(0.0, 0.2001708 - Q.sj2_dr) / 0.07331403   # -6.1%  sj2_dr < 0.2002
        + 0.05917126 * max(0.0, 0.01882765 - Q.lam1_plus_lam2) / 0.01314296   # +5.9%  lam1_plus_lam2 < 0.01883
        - 0.05335836 * max(0.0, 0.1019409 - Q.sum_z_dr) / 0.04863667   # -5.3%  sum_z_dr < 0.1019
        - 0.04977249 * max(0.0, 0.01165737 - Q.sum_zz_dr2) / 0.007203073   # -5.0%  sum_zz_dr2 < 0.01166
        - 0.04228338 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) / 0.002544039   # -4.2%  lam1_plus_lam2 < 0.006097
        + 0.02491302 * max(0.0, 0.006756161 - Q.sum_z_dr2_top3) / 0.003650633   # +2.5%  sum_z_dr2_top3 < 0.006756
        - 0.02285796 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -2.3%  lam1 < 0.008376
        - 0.0178722 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # -1.8%  mass_over_sum_pt < 0.1309
        - 0.01592641 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.01216943 - Q.zdr_0) / 3.637364e-05   # -1.6%  centroid_offset < 0.01838 and zdr_0 < 0.01217
        - 0.01444355 * max(0.0, 0.0003061234 - Q.lam2) / 0.0001989094   # -1.4%  lam2 < 0.0003061
        - 0.01421527 * max(0.0, 0.1492731 - Q.sj2_dr) / 0.04374278   # -1.4%  sj2_dr < 0.1493
        - 0.01367819 * max(0.0, 0.06344108 - Q.e2) / 0.03657078   # -1.4%  e2 < 0.06344
        + 0.01358144 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # +1.4%  lam1 < 0.005433
        + 0.0122539 * max(0.0, 0.0005124533 - Q.sum_z_dr2_top2) / 0.0001104529   # +1.2%  sum_z_dr2_top2 < 0.0005125
        - 0.01090293 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2512159 - Q.dr01) / 0.001381924   # -1.1%  centroid_offset < 0.01838 and dr01 < 0.2512
        - 0.01009399 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # -1.0%  centroid_offset < 0.01838
        + 0.009515675 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +1.0%  N2 < 0.2233 and sum_pt_top5 > 430.8
        + 0.007926956 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2) / 0.003993721   # +0.8%  mass_over_sum_pt < 0.1309 and D2 < 0.746
        - 0.005817162 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # -0.6%  N2 < 0.2233
        - 0.005491878 * max(0.0, 0.001130645 - Q.lam2) * max(0.0, Q.mean_phi - -0.01753483) / 1.673325e-05   # -0.5%  lam2 < 0.001131 and mean_phi > -0.01753
        - 0.003769991 * max(0.0, 0.007520088 - Q.sum_z_dr2) * max(0.0, 0.7459513 - Q.D2) / 9.127003e-05   # -0.4%  sum_z_dr2 < 0.00752 and D2 < 0.746
        - 0.003611392 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.0114178   # -0.4%  N2 < 0.2233 and z_dr_0p05_0p1 < 0.5883
        + 0.003577581 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 716.8828 - Q.sum_pt_top5) / 7.315623   # +0.4%  N2 < 0.2233 and sum_pt_top5 < 716.9
        - 0.003427471 * max(0.0, 0.002464291 - Q.lam1) / 0.0007496297   # -0.3%  lam1 < 0.002464
        - 0.002737973 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) * max(0.0, 0.7459513 - Q.D2) / 8.253816e-05   # -0.3%  mass_over_sum_pt_sq < 0.007183 and D2 < 0.746
        + 0.002247134 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +0.2%  LHA < 0.3033
        - 0.001698349 * max(0.0, 0.002151568 - Q.sum_z_dr2_top3) / 0.0007789603   # -0.2%  sum_z_dr2_top3 < 0.002152
        + 0.00163846 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.1094252   # +0.2%  N2 < 0.2233 and n_dr_0p1_0p2 < 4
        + 0.001206924 * max(0.0, 0.06344108 - Q.e2) * max(0.0, Q.dr12 - 0.119512) / 0.0002640953   # +0.1%  e2 < 0.06344 and dr12 > 0.1195
        - 0.0009780089 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 6.0 - Q.n_for_90pct) / 0.005228592   # -0.1%  N2 < 0.2233 and n_for_90pct < 6
        + 0.000874782 * max(0.0, 0.01882765 - Q.lam1_plus_lam2) * max(0.0, Q.ptdr0_2 - 14.78048) / 0.005175612   # +0.1%  lam1_plus_lam2 < 0.01883 and ptdr0_2 > 14.78
        - 0.0007393462 * max(0.0, Q.sj3_pair_mass_max - 72.58129) / 0.608627   # -0.1%  sj3_pair_mass_max > 72.58
        - 0.000626988 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.121681 - Q.max_dr) / 0.0004905272   # -0.1%  N2 < 0.2233 and max_dr < 0.1217
        + 0.0004298037 * max(0.0, Q.sj3_pair_mass_max - 72.58129) * max(0.0, 1.666763 - Q.N3) / 0.3484543   # +0.0%  sj3_pair_mass_max > 72.58 and N3 < 1.667
        - 0.0002720738 * max(0.0, 0.00609665 - Q.lam1_plus_lam2) * max(0.0, 0.08499387 - Q.D2_b2) / 3.927041e-06   # -0.0%  lam1_plus_lam2 < 0.006097 and D2_b2 < 0.08499
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.852643487394958, 0.7095107142857143, 2.140167962184874, 0.9633584033613445, 1.2424617647058824, 1.8029821428571429, 0.99445, 1.8416281512605042, 0.27951785714285715, 3.109162605042017, 2.0466551470588237, 2.2224654411764706, 0.08820189075630253, 3.3909008403361343, 0.42770409663865544, 0.35857457983193275]
T = [2.350022962349002, 1.4104503167673321, 3.316755540966386, 2.491016343881302, 2.921145401129202]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +39%, n9 +23%, n5 -14%, n1 +12%, n0 -6%, n6 +5% ...
            + 0.3913168 * h[2] / H_AVG[2]
            + 0.2273966 * h[9] / H_AVG[9]
            - 0.1438536 * h[5] / H_AVG[5]
            + 0.1179361 * h[1] / H_AVG[1]
            - 0.05669117 * h[0] / H_AVG[0]
            + 0.04628379 * h[6] / H_AVG[6]
            - 0.01652194 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -18%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5597048 * h[9] / H_AVG[9]
            - 0.1813831 * h[10] / H_AVG[10]
            + 0.08813231 * h[6] / H_AVG[6]
            - 0.08258411 * h[4] / H_AVG[4]
            + 0.05992043 * h[5] / H_AVG[5]
            + 0.01588919 * h[15] / H_AVG[15]
            + 0.01238602 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +25%, n3 -15%, n7 +12%, n14 -10%, n6 -9%, n0 +9% ...
            + 0.251277 * h[11] / H_AVG[11]
            - 0.145226 * h[3] / H_AVG[3]
            + 0.1214609 * h[7] / H_AVG[7]
            - 0.09671442 * h[14] / H_AVG[14]
            - 0.09369567 * h[6] / H_AVG[6]
            + 0.08836835 * h[0] / H_AVG[0]
            - 0.07432565 * h[15] / H_AVG[15]
            + 0.07188432 * h[13] / H_AVG[13]
            - 0.02929409 * h[9] / H_AVG[9]
            - 0.02106862 * h[8] / H_AVG[8]
            - 0.006684909 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -22%, n6 -15%, n13 +7%, n14 +6%, n9 -4% ...
            + 0.3465506 * h[7] / H_AVG[7]
            - 0.2175374 * h[3] / H_AVG[3]
            - 0.1497055 * h[6] / H_AVG[6]
            + 0.07444347 * h[13] / H_AVG[13]
            + 0.06438699 * h[14] / H_AVG[14]
            - 0.03900469 * h[9] / H_AVG[9]
            + 0.03896696 * h[4] / H_AVG[4]
            + 0.03560348 * h[1] / H_AVG[1]
            - 0.02249173 * h[15] / H_AVG[15]
            + 0.01130928 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +26%, n5 -15%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4715799 * h[13] / H_AVG[13]
            + 0.2627379 * h[10] / H_AVG[10]
            - 0.1543044 * h[5] / H_AVG[5]
            + 0.05316672 * h[4] / H_AVG[4]
            + 0.02061174 * h[3] / H_AVG[3]
            + 0.01794145 * h[8] / H_AVG[8]
            - 0.01509714 * h[12] / H_AVG[12]
            + 0.00456073 * h[0] / H_AVG[0]
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
