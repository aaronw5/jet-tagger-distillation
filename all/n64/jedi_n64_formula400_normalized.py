"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  14.1%   (on for 61% of jets)
  neuron  5:  11.5%   (on for 78% of jets)
  neuron  1:  11.4%   (on for 93% of jets)
  neuron  4:   9.1%   (on for 70% of jets)
  neuron  9:   7.2%   (on for 69% of jets)
  neuron 10:   7.1%   (on for 78% of jets)
  neuron  0:   7.0%   (on for 55% of jets)
  neuron 13:   6.9%   (on for 85% of jets)
  neuron  7:   5.0%   (on for 40% of jets)
  neuron  6:   4.5%   (on for 59% of jets)
  neuron  3:   4.0%   (on for 75% of jets)
  neuron 12:   3.9%   (on for 37% of jets)
  neuron 11:   3.0%   (on for 68% of jets)
  neuron 14:   2.4%   (on for 38% of jets)
  neuron 15:   2.3%   (on for 58% of jets)
  neuron  2:   0.6%   (on for 55% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.8% (the network: 81.1%); same class as the network for 92.3% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
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
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_11                 pT share × ΔR of particle 11 (its part of the sum_z_dr)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.sum_z_dr2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.sum_z_dr2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
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
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.tau43                  N-subjettiness τ4/τ3
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
        z_top5=sum(zs[:5]),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
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
        pt_2=pt[2],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        z_11=z[11],
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top2_slots=sum(pt[:2]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        zdr_11=z[11] * dr[11],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        eta_1=eta[1],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        sum_z_dr2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        sum_z_dr2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
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
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    # scale S = 15.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.59476 * (0.08645491
        - 0.1991002 * max(0.0, Q.mass - 78.26182) / 21.33658   # -19.9%  mass > 78.26
        - 0.09402685 * max(0.0, Q.mass - 89.74183) / 15.55572   # -9.4%  mass > 89.74
        - 0.07932553 * max(0.0, Q.mass - 92.85979) / 14.48273   # -7.9%  mass > 92.86
        + 0.06360637 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 0.002241068   # +6.4%  mass_over_sum_pt_sq < 0.007873
        - 0.06256694 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -6.3%  mass_top50 > 82.04
        + 0.05769143 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq) / 0.001689525   # +5.8%  mass_over_sum_pt_sq < 0.006939
        + 0.05181636 * max(0.0, Q.mass_top50 - 71.79516) / 24.22501   # +5.2%  mass_top50 > 71.8
        - 0.04852758 * max(0.0, 0.00616708 - Q.sum_zz_dr2) / 0.001295555   # -4.9%  sum_zz_dr2 < 0.006167
        - 0.04439474 * max(0.0, Q.mass - 91.19) / 15.01545   # -4.4%  mass > 91.19
        - 0.03691867 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -3.7%  mass < 101
        + 0.03611093 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +3.6%  sum_pt_top50 < 1157
        - 0.03213421 * max(0.0, 0.006374178 - Q.sum_z_dr2_top20) / 0.001987751   # -3.2%  sum_z_dr2_top20 < 0.006374
        - 0.02991665 * max(0.0, 0.006363916 - Q.sum_z_dr2_top30) / 0.001678624   # -3.0%  sum_z_dr2_top30 < 0.006364
        - 0.02653475 * max(0.0, Q.mass - 74.25181) / 24.07648   # -2.7%  mass > 74.25
        - 0.02351848 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.4%  tau1 < 0.07057
        + 0.02055389 * max(0.0, 0.007856958 - Q.sum_z_dr2_top30) / 0.002601721   # +2.1%  sum_z_dr2_top30 < 0.007857
        - 0.01530144 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # -1.5%  sum_pt_top40 < 1070
        - 0.01422288 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -1.4%  lam1 < 0.005914
        + 0.01220214 * max(0.0, 0.005312783 - Q.sum_z_dr2_top20) / 0.001437214   # +1.2%  sum_z_dr2_top20 < 0.005313
        + 0.01067923 * max(0.0, 0.004754444 - Q.mass_over_sum_pt_sq) / 0.0007997283   # +1.1%  mass_over_sum_pt_sq < 0.004754
        - 0.01040203 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -1.0%  sum_pt < 1013
        - 0.009231241 * max(0.0, 0.9906378 - Q.z_top50_slots) / 0.004783315   # -0.9%  z_top50_slots < 0.9906
        + 0.008569786 * max(0.0, 80.4 - Q.mass_top30) / 14.65577   # +0.9%  mass_top30 < 80.4
        + 0.005433609 * max(0.0, 846.1934 - Q.sum_pt_top20) / 22.83716   # +0.5%  sum_pt_top20 < 846.2
        - 0.004229776 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3) / 0.2515325   # -0.4%  sum_pt < 1013 and M3 < 0.03457
        + 0.002984288 * max(0.0, 858.8262 - Q.sum_pt_top40) / 3.625959   # +0.3%  sum_pt_top40 < 858.8
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 13.16;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.15668 * (0.113033
        + 0.1114178 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # +11.1%  log_sum_pt > 6.894
        - 0.08977086 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -9.0%  log_sum_pt < 7.139
        - 0.08377417 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -8.4%  sum_pt_top50 > 959.1
        + 0.08280222 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +8.3%  log_sum_pt > 6.91
        + 0.07657397 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +7.7%  n_particles > 38
        + 0.06671765 * max(0.0, 0.02412652 - Q.sum_z_dr2_top30) / 0.01613825   # +6.7%  sum_z_dr2_top30 < 0.02413
        + 0.06227825 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.2734   # +6.2%  sum_pt_top2 < 689.2
        - 0.03963327 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2) / 190.5117   # -4.0%  sj3_mass1 < 32.5 and sj3_mass2 < 18.68
        - 0.03639125 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt) / 15.05788   # -3.6%  n_particles > 38 and soft1_pt < 2.275
        - 0.03557508 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # -3.6%  sj3_mass1 < 32.5
        - 0.0333305 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -3.3%  log_sum_pt > 6.989
        - 0.03308891 * max(0.0, 120.6 - Q.mass) / 38.8279   # -3.3%  mass < 120.6
        + 0.03237895 * max(0.0, Q.pt_entropy - 2.07371) / 0.8049786   # +3.2%  pt_entropy > 2.074
        - 0.03111676 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # -3.1%  M3 < 0.03188
        + 0.0276212 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # +2.8%  sum_pt_top40 < 1070
        - 0.0231842 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 0.9624339 - Q.tau43) / 39.08134   # -2.3%  sum_pt_top2 < 689.2 and tau43 < 0.9624
        - 0.01980701 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -2.0%  n_dr_0p2_0p4 < 7
        + 0.0167998 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0) / 0.570275   # +1.7%  n_particles > 38 and dr_0 < 0.112
        + 0.01673838 * max(0.0, 0.0005522528 - Q.sum_z_dr2_top3) / 0.000118253   # +1.7%  sum_z_dr2_top3 < 0.0005523
        + 0.01522807 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +1.5%  sum_pt < 1017
        + 0.01248074 * max(0.0, 0.004673423 - Q.lam1) / 0.0009557143   # +1.2%  lam1 < 0.004673
        + 0.01122065 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 39.23115   # +1.1%  mass_top20 < 47.89 and n_real_top40 > 29
        - 0.01056281 * max(0.0, 31.35938 - Q.pt_9) / 6.538304   # -1.1%  pt_9 < 31.36
        + 0.01038153 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2) / 0.0008500795   # +1.0%  z_top30_slots > 0.9342 and C2 < 0.0728
        + 0.007726749 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793) / 0.1941151   # +0.8%  z_top30_slots > 0.9342 and max_pair_mass > 13.05
        - 0.007467702 * max(0.0, 47.88842 - Q.mass_top20) / 6.040308   # -0.7%  mass_top20 < 47.89
        - 0.005931533 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -0.6%  log_sum_pt > 6.959
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 6.778;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.778125 * (0.04541709
        - 0.1884754 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -18.8%  mass < 92.86
        - 0.153126 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # -15.3%  sum_pt < 1261
        + 0.08810351 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.sum_z_dr2_top15) / 0.0009227558   # +8.8%  log_sum_pt > 6.903 and sum_z_dr2_top15 < 0.02147
        + 0.07790908 * max(0.0, 91.19 - Q.mass) / 16.46423   # +7.8%  mass < 91.19
        - 0.07232609 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.sum_z_dr2_top15) / 1.045047   # -7.2%  sum_pt_top50 > 988.5 and sum_z_dr2_top15 < 0.02147
        + 0.06881916 * max(0.0, 1048.098 - Q.sum_pt_top50) / 44.36839   # +6.9%  sum_pt_top50 < 1048
        + 0.06720214 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # +6.7%  sum_pt > 1017
        + 0.06248382 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +6.2%  lam1 < 0.01174
        + 0.04882341 * max(0.0, Q.z_top20_slots - 0.7516206) / 0.1429779   # +4.9%  z_top20_slots > 0.7516
        - 0.03281944 * max(0.0, Q.sum_pt_top50 - 988.4554) / 62.81258   # -3.3%  sum_pt_top50 > 988.5
        + 0.03121045 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # +3.1%  mass_top50 < 92.17
        - 0.03105283 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -3.1%  log_sum_pt > 6.959
        + 0.02583432 * max(0.0, 0.004763596 - Q.sum_z_dr2_top30) / 0.0009958273   # +2.6%  sum_z_dr2_top30 < 0.004764
        - 0.01654737 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # -1.7%  sum_pt > 1116
        - 0.01490714 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -1.5%  sum_pt < 972
        + 0.01379456 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +1.4%  sum_pt_top40 > 1070
        - 0.006565339 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.08734131) / 0.003497383   # -0.7%  sum_pt_top20 > 1129 and eta_1 > 0.08734
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 3.818;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.818107 * (0.01637393
        - 0.135186 * max(0.0, 0.009614971 - Q.sum_z_dr2) / 0.003502545   # -13.5%  sum_z_dr2 < 0.009615
        + 0.1165985 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +11.7%  tau4 < 0.01627
        - 0.1157156 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -11.6%  tau21 < 0.3862
        + 0.09807283 * max(0.0, 0.08665515 - Q.mass_over_sum_pt) / 0.01542259   # +9.8%  mass_over_sum_pt < 0.08666
        + 0.08082179 * max(0.0, 0.008840538 - Q.sum_z_dr2_top40) / 0.003108622   # +8.1%  sum_z_dr2_top40 < 0.008841
        - 0.07829063 * max(0.0, 80.4 - Q.mass_top40) / 12.19371   # -7.8%  mass_top40 < 80.4
        + 0.07263052 * max(0.0, 27.56535 - Q.sj2_mass1) / 6.309897   # +7.3%  sj2_mass1 < 27.57
        + 0.06797972 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +6.8%  mass < 92.86
        - 0.06593546 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # -6.6%  n_particles < 46
        + 0.05662483 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732) / 1477.074   # +5.7%  n_particles < 46 and sum_pt_top30 > 800.7
        + 0.05116695 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421) / 0.000406438   # +5.1%  D2 < 2.41 and psi_0p3 > 0.9985
        - 0.03809843 * max(0.0, 79.21004 - Q.mass_top50) / 10.65754   # -3.8%  mass_top50 < 79.21
        - 0.01677386 * max(0.0, 0.001260456 - Q.lam1) / 8.753718e-05   # -1.7%  lam1 < 0.00126
        - 0.006104852 * max(0.0, 79.21004 - Q.mass_top50) * max(0.0, Q.mass_top3 - 8.921413) / 4.657205   # -0.6%  mass_top50 < 79.21 and mass_top3 > 8.921
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 9.296;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.295942 * (0.1471144
        + 0.1325698 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +13.3%  mass < 101
        - 0.1215481 * max(0.0, 78.26182 - Q.mass) / 9.857173   # -12.2%  mass < 78.26
        + 0.0840757 * max(0.0, 120.6 - Q.mass) / 38.8279   # +8.4%  mass < 120.6
        + 0.08168577 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +8.2%  mass < 74.25
        - 0.0716117 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -7.2%  mass_top40 < 163.3
        - 0.06375518 * max(0.0, Q.n_particles - 22.0) / 23.9717   # -6.4%  n_particles > 22
        + 0.05719608 * max(0.0, Q.sum_zz_dr2 - 0.009606007) / 0.003182984   # +5.7%  sum_zz_dr2 > 0.009606
        - 0.05676726 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -5.7%  mass < 92.86
        - 0.04315162 * max(0.0, 0.004855289 - Q.sum_z_dr2_top15) / 0.001418414   # -4.3%  sum_z_dr2_top15 < 0.004855
        - 0.04177169 * max(0.0, 86.4 - Q.mass) / 13.67632   # -4.2%  mass < 86.4
        - 0.04171834 * max(0.0, Q.lam1 - 0.008241985) / 0.002595658   # -4.2%  lam1 > 0.008242
        - 0.03830117 * max(0.0, 80.78464 - Q.mass) / 10.84952   # -3.8%  mass < 80.78
        + 0.03716052 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +3.7%  mass_top40 < 67.73
        - 0.03161495 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -3.2%  mass_top40 < 83.33
        + 0.02297039 * max(0.0, 60.43821 - Q.mass_top30) / 6.992028   # +2.3%  mass_top30 < 60.44
        + 0.02290695 * max(0.0, 57.87349 - Q.mass_top15) / 12.46085   # +2.3%  mass_top15 < 57.87
        + 0.01602769 * max(0.0, Q.sj3_pair_mass_min - 32.51366) / 5.88267   # +1.6%  sj3_pair_mass_min > 32.51
        - 0.01199739 * max(0.0, Q.sum_z_dr2_top15 - 0.00727763) / 0.002923446   # -1.2%  sum_z_dr2_top15 > 0.007278
        - 0.009074536 * max(0.0, Q.lam2 - 0.001776308) / 0.0005876074   # -0.9%  lam2 > 0.001776
        + 0.007929575 * max(0.0, Q.sum_z_dr2_top15 - 0.01563836) / 0.001334556   # +0.8%  sum_z_dr2_top15 > 0.01564
        - 0.006165586 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -0.6%  sum_z_dr > 0.1207
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 12.12;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.11805 * (0.0526545
        + 0.1803832 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +18.0%  sum_pt_top50 > 934.2
        + 0.07224319 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +7.2%  sum_pt > 907.9
        - 0.06843017 * max(0.0, Q.z_top30_slots - 0.9048492) / 0.05621641   # -6.8%  z_top30_slots > 0.9048
        - 0.06457416 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -6.5%  log_sum_pt > 6.91
        - 0.05867664 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -5.9%  log_sum_pt > 6.92
        + 0.05723619 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +5.7%  n_particles < 64
        + 0.04850535 * max(0.0, Q.sd_mass - 69.65633) / 11.73272   # +4.9%  sd_mass > 69.66
        + 0.03875722 * max(0.0, Q.sum_pt_top30 - 933.1875) / 79.88556   # +3.9%  sum_pt_top30 > 933.2
        - 0.03864661 * max(0.0, Q.sd_mass - 86.4) / 6.275027   # -3.9%  sd_mass > 86.4
        - 0.03649169 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -3.6%  sum_pt_top40 > 1025
        - 0.03626453 * max(0.0, Q.sum_z_dr2_top30 - 0.006363916) / 0.003802703   # -3.6%  sum_z_dr2_top30 > 0.006364
        + 0.03267733 * max(0.0, Q.mass - 74.25181) / 24.07648   # +3.3%  mass > 74.25
        - 0.03112664 * max(0.0, 0.5494307 - Q.tau21) / 0.1591168   # -3.1%  tau21 < 0.5494
        - 0.02985435 * max(0.0, 150.0144 - Q.mass_top40) / 66.86198   # -3.0%  mass_top40 < 150
        - 0.02734452 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # -2.7%  sum_pt > 986.1
        - 0.0244691 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -2.4%  max_dr > 0.2405
        + 0.01863597 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +1.9%  sum_pt > 907.9 and e4 < 5.851e-08
        - 0.01679816 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -1.7%  mass_top50 > 157.5
        + 0.01633468 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # +1.6%  log_sum_pt > 6.989
        - 0.01469837 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2) / 10.81389   # -1.5%  n_particles < 64 and D2 < 2.179
        - 0.01234868 * max(0.0, Q.mass_top40 - 120.6) / 5.823744   # -1.2%  mass_top40 > 120.6
        + 0.01221799 * max(0.0, 1.976207 - Q.D2) / 0.367701   # +1.2%  D2 < 1.976
        - 0.0119657 * max(0.0, Q.mass_over_sum_pt - 0.09046749) / 0.01421039   # -1.2%  mass_over_sum_pt > 0.09047
        - 0.00856263 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -0.9%  mass_over_sum_pt_sq > 0.0292
        + 0.007642553 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +0.8%  mass_over_sum_pt > 0.1709
        - 0.007022424 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2) / 0.001543279   # -0.7%  psi_0p3 > 0.9974 and D2 < 3.345
        + 0.006605534 * max(0.0, 0.01375115 - Q.z_11) / 0.0009450582   # +0.7%  z_11 < 0.01375
        + 0.00621042 * max(0.0, Q.mass_top10 - 71.781) / 3.677551   # +0.6%  mass_top10 > 71.78
        - 0.005126862 * max(0.0, 14.14062 - Q.pt_11) / 0.9740783   # -0.5%  pt_11 < 14.14
        + 0.004695507 * max(0.0, Q.mass - 172.8) / 0.7291439   # +0.5%  mass > 172.8
        - 0.003007808 * max(0.0, Q.sum_pt_top20 - 1129.275) / 6.499168   # -0.3%  sum_pt_top20 > 1129
        + 0.002445823 * max(0.0, Q.max_dr - 0.4357228) / 0.007711928   # +0.2%  max_dr > 0.4357
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 11.19;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.19134 * (0.1366114
        - 0.1812136 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -18.1%  mass < 101
        - 0.1292214 * max(0.0, 120.6 - Q.mass) / 38.8279   # -12.9%  mass < 120.6
        + 0.1061019 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +10.6%  mass < 92.86
        + 0.0638671 * max(0.0, Q.sum_z_dr2_top30 - 0.008376291) / 0.003092781   # +6.4%  sum_z_dr2_top30 > 0.008376
        - 0.06082149 * max(0.0, Q.sum_z_dr2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529) / 8.566262e-05   # -6.1%  sum_z_dr2_top20 > 0.008031 and C2_b2 > 0.0008188
        + 0.06005523 * max(0.0, 172.8 - Q.mass_top50) / 85.49388   # +6.0%  mass_top50 < 172.8
        + 0.04355733 * max(0.0, 86.4 - Q.mass) / 13.67632   # +4.4%  mass < 86.4
        + 0.04276586 * max(0.0, Q.sum_z_dr2_top20 - 0.008031209) / 0.002906852   # +4.3%  sum_z_dr2_top20 > 0.008031
        + 0.03505324 * max(0.0, 91.19 - Q.mass_top15) / 34.88803   # +3.5%  mass_top15 < 91.19
        - 0.03138107 * max(0.0, Q.e2 - 0.02793599) / 0.008664879   # -3.1%  e2 > 0.02794
        - 0.03071127 * max(0.0, 0.01807679 - Q.sum_z_dr2_top30) / 0.01083719   # -3.1%  sum_z_dr2_top30 < 0.01808
        - 0.02895736 * max(0.0, Q.mass_over_sum_pt - 0.09795415) / 0.01222897   # -2.9%  mass_over_sum_pt > 0.09795
        - 0.02723736 * max(0.0, Q.sj3_dr_min - 0.1204829) / 0.02211875   # -2.7%  sj3_dr_min > 0.1205
        + 0.02391103 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # +2.4%  mass_top50 < 71.8
        - 0.02335395 * max(0.0, 0.004573744 - Q.sum_z_dr2_top50) / 0.0007739713   # -2.3%  sum_z_dr2_top50 < 0.004574
        - 0.02303669 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # -2.3%  LHA > 0.372
        + 0.01903159 * max(0.0, 0.003811746 - Q.lam1) / 0.0006796118   # +1.9%  lam1 < 0.003812
        + 0.01633385 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +1.6%  tau1 < 0.06311
        + 0.01612689 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832) / 8.191828e-05   # +1.6%  mass_over_sum_pt > 0.1182 and C2_b2 > 0.02705
        - 0.006942269 * max(0.0, Q.sum_z_dr2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722) / 0.001787472   # -0.7%  sum_z_dr2_top30 > 0.008376 and D2_b2 > 1.677
        + 0.00640059 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707) / 2.369749e-05   # +0.6%  e2 > 0.05557 and zdr_0 > 0.00137
        + 0.006173286 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt) / 683.4113   # +0.6%  mass < 120.6 and sum_pt < 1008
        - 0.005121315 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2) / 0.04240098   # -0.5%  mass_over_sum_pt > 0.1182 and D2_b2 < 7.366
        + 0.004589107 * max(0.0, Q.sj3_dr_min - 0.1204829) * max(0.0, 7.863702 - Q.sj3_mass2) / 0.02256917   # +0.5%  sj3_dr_min > 0.1205 and sj3_mass2 < 7.864
        - 0.003243519 * max(0.0, Q.sum_z_dr2_top30 - 0.008376291) * max(0.0, Q.orientation_deg - 26.6454) / 0.03454067   # -0.3%  sum_z_dr2_top30 > 0.008376 and orientation_deg > 26.65
        - 0.002708679 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.3%  log_sum_pt < 6.811
        + 0.002082997 * max(0.0, Q.sum_z_dr2_top30 - 0.02809026) / 0.0001993235   # +0.2%  sum_z_dr2_top30 > 0.02809
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 44.3;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 44.30411 * (0.0006370499
        - 0.1333774 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.3075308   # -13.3%  mass < 91.19 and psi_0p3 > 0.9777
        - 0.1147973 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5700006   # -11.5%  mass < 92.86 and psi_0p3 > 0.9638
        + 0.1084927 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.4385899   # +10.8%  mass < 101 and psi_0p3 > 0.9777
        + 0.09406994 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5337743   # +9.4%  mass < 91.19 and psi_0p3 > 0.9638
        - 0.0644465 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -6.4%  mass < 82.85
        + 0.04874097 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.01745475   # +4.9%  mass < 101 and psi_0p3 > 0.998
        + 0.04069143 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +4.1%  mass < 78.26
        + 0.03530655 * max(0.0, 120.6 - Q.mass) / 38.8279   # +3.5%  mass < 120.6
        - 0.0335843 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -3.4%  mass < 101
        - 0.03299564 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -3.3%  tau1 < 0.1073
        + 0.03230186 * max(0.0, 0.009614971 - Q.sum_z_dr2) / 0.003502545   # +3.2%  sum_z_dr2 < 0.009615
        - 0.02837516 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.01143143   # -2.8%  mass < 91.19 and psi_0p3 > 0.998
        - 0.02712884 * max(0.0, 86.4 - Q.sd_mass) / 35.84633   # -2.7%  sd_mass < 86.4
        - 0.02458146 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -2.5%  psi_0p3 > 0.998
        + 0.02359061 * max(0.0, 69.65633 - Q.sd_mass) / 24.56034   # +2.4%  sd_mass < 69.66
        - 0.02349319 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -2.3%  lam1 < 0.008242
        + 0.01864759 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +1.9%  tau21_b2 < 0.2352
        - 0.01758486 * max(0.0, 0.008190222 - Q.sum_z_dr2) / 0.002454223   # -1.8%  sum_z_dr2 < 0.00819
        + 0.01608024 * max(0.0, 0.09591084 - Q.tau1) / 0.02629125   # +1.6%  tau1 < 0.09591
        + 0.01259305 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # +1.3%  lam1 < 0.00619
        + 0.01180929 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +1.2%  psi_0p3 > 0.9974
        - 0.01177956 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008) / 0.007828415   # -1.2%  mass < 82.85 and psi_0p3 > 0.998
        - 0.009877986 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -1.0%  tau21 < 0.3862
        - 0.009157941 * max(0.0, 0.00363788 - Q.sum_zz_dr2) / 0.0004963098   # -0.9%  sum_zz_dr2 < 0.003638
        + 0.00731998 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +0.7%  mass_over_sum_pt < 0.1182
        - 0.005693816 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt) / 11.66143   # -0.6%  tau21_b2 < 0.2352 and sum_pt < 1261
        + 0.003205749 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.02947847   # +0.3%  mass < 82.85 and psi_0p3 < 0.9985
        + 0.003133974 * max(0.0, 0.006403325 - Q.sum_z_dr2) * max(0.0, Q.psi_0p3 - 0.9980008) / 9.11248e-07   # +0.3%  sum_z_dr2 < 0.006403 and psi_0p3 > 0.998
        + 0.002779158 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.54185   # +0.3%  n_dr_0p2_0p4 < 6
        - 0.0025881 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 5.106465e-05   # -0.3%  tau21_b2 < 0.2352 and mass_over_sum_pt_sq < 0.007873
        - 0.001774844 * max(0.0, 0.006403325 - Q.sum_z_dr2) / 0.001406912   # -0.2%  sum_z_dr2 < 0.006403
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 17.88;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.87921 * (0.1185862
        - 0.126974 * max(0.0, 0.009614971 - Q.lam1_plus_lam2) / 0.003502545   # -12.7%  lam1_plus_lam2 < 0.009615
        + 0.06430138 * max(0.0, Q.mass - 74.25181) / 24.07648   # +6.4%  mass > 74.25
        + 0.05372826 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt) / 2.336698   # +5.4%  mass_over_sum_pt > 0.07697 and sum_pt < 1116
        + 0.052831 * max(0.0, Q.mass - 87.36377) / 16.57741   # +5.3%  mass > 87.36
        - 0.05274895 * max(0.0, Q.mass - 101.0497) / 12.31084   # -5.3%  mass > 101
        - 0.04794234 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -4.8%  mass_top50 > 82.04
        - 0.04554585 * max(0.0, Q.n_for_90pct - 7.0) / 13.93571   # -4.6%  n_for_90pct > 7
        - 0.03934985 * max(0.0, Q.sum_z_dr2_top20 - 0.006043209) / 0.003593885   # -3.9%  sum_z_dr2_top20 > 0.006043
        + 0.03889129 * max(0.0, Q.sum_z_dr2_top20 - 0.008031209) / 0.002906852   # +3.9%  sum_z_dr2_top20 > 0.008031
        - 0.03852978 * max(0.0, 39.0 - Q.n_for_90pct) / 18.29776   # -3.9%  n_for_90pct < 39
        + 0.03236308 * max(0.0, 0.0795038 - Q.tau2) / 0.04794178   # +3.2%  tau2 < 0.0795
        + 0.03054281 * max(0.0, Q.mass - 64.48544) / 31.19164   # +3.1%  mass > 64.49
        - 0.03030574 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -3.0%  n_dr_0p2_0p4 < 15
        - 0.02904899 * max(0.0, Q.sum_z_dr2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt) / 0.0005328922   # -2.9%  sum_z_dr2_top40 > 0.005197 and log_sum_pt < 7.017
        + 0.02685724 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +2.7%  lam1 < 0.007671
        + 0.0260375 * max(0.0, 1028.184 - Q.sum_pt) / 26.95739   # +2.6%  sum_pt < 1028
        + 0.02308182 * max(0.0, 0.008840538 - Q.sum_z_dr2_top40) / 0.003108622   # +2.3%  sum_z_dr2_top40 < 0.008841
        + 0.02254755 * max(0.0, Q.mass_top50 - 117.0487) / 7.705579   # +2.3%  mass_top50 > 117
        + 0.02203244 * max(0.0, 0.007877041 - Q.sum_z_dr2) / 0.002242297   # +2.2%  sum_z_dr2 < 0.007877
        - 0.01833849 * max(0.0, 6.930088 - Q.log_sum_pt) / 0.02522479   # -1.8%  log_sum_pt < 6.93
        + 0.01731448 * max(0.0, 0.007463985 - Q.sum_z_dr2_top30) / 0.00233708   # +1.7%  sum_z_dr2_top30 < 0.007464
        - 0.01656854 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -1.7%  psi_0p3 > 0.9943
        - 0.01610336 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # -1.6%  mass_over_sum_pt > 0.07697
        - 0.01478027 * max(0.0, Q.mass - 125.1) / 7.098976   # -1.5%  mass > 125.1
        - 0.0137662 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # -1.4%  sum_pt_top40 < 1025
        + 0.01353094 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.6773544   # +1.4%  sum_pt < 1028 and dr_max_012 > 0.1828
        + 0.01309388 * max(0.0, Q.n_particles - 51.0) / 3.973704   # +1.3%  n_particles > 51
        - 0.0125056 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389) / 0.5693287   # -1.3%  sum_pt < 1017 and dr_max_012 > 0.1828
        + 0.01179878 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +1.2%  sum_pt < 1017
        - 0.01149116 * max(0.0, Q.mass_over_sum_pt - 0.1182259) / 0.007735866   # -1.1%  mass_over_sum_pt > 0.1182
        + 0.01114091 * max(0.0, 0.8316924 - Q.z_top15_slots) / 0.04702674   # +1.1%  z_top15_slots < 0.8317
        + 0.005976079 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # +0.6%  e3 > 0.0003372
        - 0.005064903 * max(0.0, Q.n_particles - 51.0) * max(0.0, 1.091797 - Q.soft1_pt) / 1.364428   # -0.5%  n_particles > 51 and soft1_pt < 1.092
        + 0.004700262 * max(0.0, Q.sum_z_dr2_top40 - 0.005196966) / 0.004725809   # +0.5%  sum_z_dr2_top40 > 0.005197
        - 0.004245083 * max(0.0, Q.e3 - 5.13841e-05) / 6.821576e-05   # -0.4%  e3 > 5.138e-05
        + 0.003296168 * max(0.0, 0.3628388 - Q.psi_0p1) / 0.02326895   # +0.3%  psi_0p1 < 0.3628
        - 0.002625046 * max(0.0, Q.mass_top20 - 119.2969) / 1.8402   # -0.3%  mass_top20 > 119.3
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 44.15;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 44.14627 * (-0.01540521
        - 0.2612797 * max(0.0, 160.8 - Q.mass) / 72.78344   # -26.1%  mass < 160.8
        + 0.2156611 * max(0.0, 162.8363 - Q.mass) / 74.60496   # +21.6%  mass < 162.8
        + 0.1484418 * max(0.0, 172.8 - Q.mass) / 83.78792   # +14.8%  mass < 172.8
        - 0.1271837 * max(0.0, 143.7876 - Q.mass) / 57.99337   # -12.7%  mass < 143.8
        + 0.06233374 * max(0.0, 138.8977 - Q.mass_top50) / 55.01559   # +6.2%  mass_top50 < 138.9
        - 0.04319192 * max(0.0, 163.2541 - Q.mass_top40) / 79.07472   # -4.3%  mass_top40 < 163.3
        + 0.02638921 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +2.6%  mass < 92.86
        - 0.01665358 * max(0.0, 92.16545 - Q.mass_top50) / 17.79873   # -1.7%  mass_top50 < 92.17
        + 0.0147061 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +1.5%  mass < 82.85
        + 0.013053 * max(0.0, Q.sum_z_dr2_top30 - 0.0008564881) / 0.007661497   # +1.3%  sum_z_dr2_top30 > 0.0008565
        - 0.01181843 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -1.2%  mass < 64.49
        + 0.01045606 * max(0.0, 0.006259772 - Q.sum_z_dr2_top40) / 0.001461823   # +1.0%  sum_z_dr2_top40 < 0.00626
        + 0.008937166 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +0.9%  mass_top40 < 80.89
        + 0.006186255 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +0.6%  sum_pt < 986.1
        + 0.004856953 * max(0.0, Q.LHA - 0.404204) / 0.003157714   # +0.5%  LHA > 0.4042
        - 0.004608624 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -0.5%  log_sum_pt < 6.903
        - 0.003857312 * max(0.0, Q.sj3_dr13 - 0.1654269) / 0.05535154   # -0.4%  sj3_dr13 > 0.1654
        + 0.003595018 * max(0.0, 73.33139 - Q.mass_top30) / 11.37701   # +0.4%  mass_top30 < 73.33
        - 0.003565538 * max(0.0, Q.sum_z_dr - 0.1402186) / 0.001871547   # -0.4%  sum_z_dr > 0.1402
        - 0.002590974 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt) / 447.38   # -0.3%  mass_top40 < 80.89 and sum_pt < 1035
        - 0.002339726 * max(0.0, Q.sum_z_dr - 0.09749958) / 0.008315822   # -0.2%  sum_z_dr > 0.0975
        + 0.002143678 * max(0.0, 956.2133 - Q.sum_pt_top40) / 14.00451   # +0.2%  sum_pt_top40 < 956.2
        - 0.001883889 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # -0.2%  e3 > 0.0003372
        + 0.001050306 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) / 0.006411021   # +0.1%  z_dr_0p1_0p2 > 0.6882
        - 0.001033774 * max(0.0, Q.z_dr_0p1_0p2 - 0.6882177) * max(0.0, 154.25 - Q.pt_2) / 0.4838258   # -0.1%  z_dr_0p1_0p2 > 0.6882 and pt_2 < 154.2
        + 0.0009487987 * max(0.0, Q.z_top5 - 0.7963975) / 0.006402524   # +0.1%  z_top5 > 0.7964
        - 0.000430071 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258) / 5.834847   # -0.0%  sum_pt_top40 < 956.2 and soft4_pt > 1.789
        + 0.0004064534 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40) / 104.3982   # +0.0%  mass_top40 < 80.89 and sum_pt_top40 < 906.6
        + 0.0003971716 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047) / 1.704178   # +0.0%  sum_pt_top40 < 956.2 and soft3_pt > 2.873
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 12.42;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.4218 * (0.01220898
        + 0.1090438 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +10.9%  mass < 89.74
        + 0.1074175 * max(0.0, Q.mass - 78.26182) / 21.33658   # +10.7%  mass > 78.26
        - 0.1051315 * max(0.0, Q.mass - 64.48544) / 31.19164   # -10.5%  mass > 64.49
        - 0.08112182 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -8.1%  mass < 101
        + 0.07202315 * max(0.0, 0.01655983 - Q.sum_z_dr2_top20) / 0.01003486   # +7.2%  sum_z_dr2_top20 < 0.01656
        + 0.06828249 * max(0.0, Q.e2 - 0.01256572) / 0.01912291   # +6.8%  e2 > 0.01257
        + 0.06448109 * max(0.0, 0.5760704 - Q.z_top2_slots) / 0.2302819   # +6.4%  z_top2_slots < 0.5761
        - 0.04112112 * max(0.0, 0.1207452 - Q.sum_z_dr) / 0.05578614   # -4.1%  sum_z_dr < 0.1207
        - 0.03786023 * max(0.0, 72.18744 - Q.mass_top15) / 20.20734   # -3.8%  mass_top15 < 72.19
        - 0.03663165 * max(0.0, Q.mass - 143.7876) / 3.946979   # -3.7%  mass > 143.8
        - 0.03457507 * max(0.0, Q.z_dr_0_0p05 - 0.7674734) / 0.05221088   # -3.5%  z_dr_0_0p05 > 0.7675
        + 0.03370748 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +3.4%  z_dr_0p2_0p4 < 0.09123
        - 0.03171805 * max(0.0, 65.20727 - Q.sj2_mass1) / 36.50254   # -3.2%  sj2_mass1 < 65.21
        + 0.03009462 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +3.0%  mass_top50 > 136.8
        + 0.02508862 * max(0.0, 0.0001086251 - Q.e3) / 5.924919e-05   # +2.5%  e3 < 0.0001086
        - 0.02284509 * max(0.0, 59.40777 - Q.mass_top5) / 33.74882   # -2.3%  mass_top5 < 59.41
        + 0.01542958 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +1.5%  dr_0 < 0.06413
        + 0.0147471 * max(0.0, 2.975532 - Q.D2) / 0.9290697   # +1.5%  D2 < 2.976
        - 0.01393842 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # -1.4%  n_dr_0p2_0p4 < 11
        - 0.01245906 * max(0.0, Q.mass_top30 - 78.53034) / 14.31024   # -1.2%  mass_top30 > 78.53
        - 0.01149204 * max(0.0, 0.002575211 - Q.sum_z_dr2) / 0.0002582872   # -1.1%  sum_z_dr2 < 0.002575
        - 0.01138686 * max(0.0, 0.01976735 - Q.sum_z_dr2_top10) * max(0.0, Q.psi_0p3 - 0.9985421) / 6.34706e-06   # -1.1%  sum_z_dr2_top10 < 0.01977 and psi_0p3 > 0.9985
        - 0.00500502 * max(0.0, Q.mass - 162.8363) / 1.509832   # -0.5%  mass > 162.8
        - 0.00453295 * max(0.0, Q.mass_top50 - 172.8) / 0.5062389   # -0.5%  mass_top50 > 172.8
        - 0.004027556 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897) / 0.001013644   # -0.4%  mass > 162.8 and soft5_z > 0.001435
        - 0.002955519 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # -0.3%  sum_pt_top50 < 959.1
        + 0.002882561 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109) / 0.0004735247   # +0.3%  mass_top50 > 160.8 and soft4_z > 0.001721
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 8.924;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.92372 * (-0.07206814
        + 0.1573367 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +15.7%  mass < 92.86
        - 0.094565 * max(0.0, 0.076787 - Q.sum_z_dr) / 0.02105267   # -9.5%  sum_z_dr < 0.07679
        - 0.08646603 * max(0.0, 80.4 - Q.mass) / 10.6806   # -8.6%  mass < 80.4
        + 0.06648121 * max(0.0, 0.08589404 - Q.sum_z_dr) / 0.02745918   # +6.6%  sum_z_dr < 0.08589
        + 0.06101523 * max(0.0, 0.00818374 - Q.sum_zz_dr2) / 0.002451404   # +6.1%  sum_zz_dr2 < 0.008184
        + 0.06066429 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +6.1%  psi_0p3 > 0.9897
        - 0.04426028 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -4.4%  lam1 < 0.006717
        + 0.04304895 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +4.3%  e2 < 0.0303
        + 0.04094108 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +4.1%  e2 < 0.02516
        + 0.03612456 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 43.23581   # +3.6%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21
        - 0.03596812 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # -3.6%  psi_0p2 > 0.9314
        - 0.03489292 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -3.5%  mass_top40 < 83.33
        - 0.03354589 * max(0.0, 0.00625621 - Q.sum_z_dr2_top10) / 0.002474789   # -3.4%  sum_z_dr2_top10 < 0.006256
        - 0.03353652 * max(0.0, 80.35535 - Q.mass_top50) / 11.1399   # -3.4%  mass_top50 < 80.36
        + 0.02831291 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +2.8%  mass < 101
        - 0.02539524 * max(0.0, 3.793233e-05 - Q.e3) / 1.004564e-05   # -2.5%  e3 < 3.793e-05
        - 0.02387189 * max(0.0, 0.006929741 - Q.sum_z_dr2_top30) / 0.002004687   # -2.4%  sum_z_dr2_top30 < 0.00693
        + 0.02349938 * max(0.0, 0.00406126 - Q.sum_z_dr2_top10) / 0.001298509   # +2.3%  sum_z_dr2_top10 < 0.004061
        + 0.02007455 * max(0.0, Q.psi_0p2 - 0.9935324) / 0.001465572   # +2.0%  psi_0p2 > 0.9935
        - 0.01923899 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.sum_z_dr2) / 0.005195774   # -1.9%  n_dr_0p2_0p4 < 10 and sum_z_dr2 < 0.005532
        + 0.01817195 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +1.8%  mass_top40 < 67.73
        + 0.01258826 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.3563114 - Q.planar_flow) / 0.8671033   # +1.3%  mass < 101 and planar_flow < 0.3563
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 3.398;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.398395 * (-0.1673696
        + 0.2739162 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +27.4%  mass < 82.85
        + 0.1702924 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.04417088   # +17.0%  mass < 86.4 and lam2 < 0.003688
        - 0.1352798 * max(0.0, 62.55 - Q.mass) / 5.464058   # -13.5%  mass < 62.55
        + 0.08231763 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +8.2%  mass < 74.25
        - 0.06467385 * max(0.0, 74.78616 - Q.mass_top40) / 9.958349   # -6.5%  mass_top40 < 74.79
        - 0.05641952 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3) / 0.03580539   # -5.6%  mass < 86.4 and psi_0p3 < 0.9985
        - 0.04792721 * max(0.0, 0.05077291 - Q.mass_over_sum_pt) / 0.003255623   # -4.8%  mass_over_sum_pt < 0.05077
        - 0.04308784 * max(0.0, 0.06895248 - Q.mass_over_sum_pt) / 0.007729512   # -4.3%  mass_over_sum_pt < 0.06895
        - 0.03268412 * max(0.0, Q.mass - 160.8) / 1.724663   # -3.3%  mass > 160.8
        - 0.03047173 * max(0.0, Q.sd_mass - 125.1) * max(0.0, Q.lam2 - 0.0001679609) / 0.005601117   # -3.0%  sd_mass > 125.1 and lam2 > 0.000168
        + 0.02171024 * max(0.0, Q.sd_mass - 125.1) * max(0.0, 0.4199841 - Q.sd_zg) / 0.1704664   # +2.2%  sd_mass > 125.1 and sd_zg < 0.42
        + 0.01699195 * max(0.0, 0.0007431905 - Q.sum_z_dr2_top10) / 0.0001243965   # +1.7%  sum_z_dr2_top10 < 0.0007432
        + 0.01693929 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min) / 54.09437   # +1.7%  sj3_pair_mass_max > 120.6 and sj3_pair_mass_min < 76.6
        + 0.007288284 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926) / 0.5295314   # +0.7%  sj3_pair_mass_max > 120.6 and z_dr_0p1_0p2 > 0.2865
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 7.433;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.433403 * (0.3151805
        + 0.1502804 * max(0.0, 0.02550569 - Q.sum_z_dr2_top50) / 0.01683664   # +15.0%  sum_z_dr2_top50 < 0.02551
        - 0.1154735 * max(0.0, 160.8 - Q.mass_top40) / 76.75884   # -11.5%  mass_top40 < 160.8
        - 0.07603536 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -7.6%  sum_pt < 1085
        + 0.07177156 * max(0.0, Q.mass_top50 - 136.785) / 4.250986   # +7.2%  mass_top50 > 136.8
        + 0.05465954 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +5.5%  sum_pt_top40 < 1053
        - 0.05314632 * max(0.0, Q.mass_top50 - 92.16545) / 13.44564   # -5.3%  mass_top50 > 92.17
        - 0.05027766 * max(0.0, Q.mass - 160.8) / 1.724663   # -5.0%  mass > 160.8
        - 0.04847532 * max(0.0, 1007.788 - Q.sum_pt) / 17.7122   # -4.8%  sum_pt < 1008
        - 0.04741843 * max(0.0, 0.02580396 - Q.mass_over_sum_pt_sq) / 0.01697662   # -4.7%  mass_over_sum_pt_sq < 0.0258
        - 0.04000735 * max(0.0, Q.mass - 143.7876) / 3.946979   # -4.0%  mass > 143.8
        + 0.03347398 * max(0.0, Q.mass - 74.25181) / 24.07648   # +3.3%  mass > 74.25
        - 0.02778652 * max(0.0, Q.mass - 136.785) / 5.043396   # -2.8%  mass > 136.8
        + 0.02671659 * max(0.0, 997.0189 - Q.sum_pt_top50) / 17.90852   # +2.7%  sum_pt_top50 < 997
        - 0.0224125 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.8310045 - Q.tau21_b2) / 33.42141   # -2.2%  sum_pt < 1085 and tau21_b2 < 0.831
        - 0.02221693 * max(0.0, 6.910131 - Q.log_sum_pt) / 0.017275   # -2.2%  log_sum_pt < 6.91
        - 0.02179311 * max(0.0, 966.0633 - Q.sum_pt_top30) / 29.99923   # -2.2%  sum_pt_top30 < 966.1
        + 0.02176555 * max(0.0, 0.007820315 - Q.sum_z_dr2_top50) / 0.002264874   # +2.2%  sum_z_dr2_top50 < 0.00782
        + 0.01967808 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # +2.0%  sum_pt_top40 < 1007
        - 0.01721576 * max(0.0, 0.08286256 - Q.tau1) / 0.0188894   # -1.7%  tau1 < 0.08286
        + 0.01710634 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +1.7%  mass_over_sum_pt > 0.1709
        - 0.015956 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -1.6%  mass_over_sum_pt_sq > 0.0292
        + 0.01345695 * max(0.0, Q.mass - 172.8) / 0.7291439   # +1.3%  mass > 172.8
        + 0.008953052 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.005041702 - Q.zdr_11) / 1.572361e-05   # +0.9%  log_sum_pt < 6.811 and zdr_11 < 0.005042
        + 0.007978295 * max(0.0, Q.mass - 162.8363) / 1.509832   # +0.8%  mass > 162.8
        + 0.007506817 * max(0.0, Q.mass_top50 - 168.9698) / 0.6651775   # +0.8%  mass_top50 > 169
        - 0.00567583 * max(0.0, 6.811175 - Q.log_sum_pt) * max(0.0, 0.1312677 - Q.dr_13) / 0.0002562188   # -0.6%  log_sum_pt < 6.811 and dr_13 < 0.1313
        + 0.002762313 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, Q.sd_zg - 0.4462823) / 1.289737e-05   # +0.3%  log_sum_pt > 7.139 and sd_zg > 0.4463
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 16.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.4146 * (-0.04261266
        - 0.1332756 * max(0.0, 91.19 - Q.mass) / 16.46423   # -13.3%  mass < 91.19
        + 0.08914376 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +8.9%  mass_over_sum_pt < 0.1182
        - 0.08793724 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -8.8%  mass < 82.85
        + 0.07595984 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +7.6%  mass_over_sum_pt < 0.09795
        - 0.06889086 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) / 0.0178873   # -6.9%  mass_over_sum_pt < 0.09047
        - 0.0509042 * max(0.0, 0.07852883 - Q.mass_over_sum_pt) / 0.01107516   # -5.1%  mass_over_sum_pt < 0.07853
        + 0.04296008 * max(0.0, 0.140939 - Q.mass_over_sum_pt) / 0.05796036   # +4.3%  mass_over_sum_pt < 0.1409
        + 0.04169274 * max(0.0, 6.941997 - Q.log_sum_pt) / 0.03180972   # +4.2%  log_sum_pt < 6.942
        + 0.04074383 * max(0.0, 0.08873143 - Q.mass_over_sum_pt) / 0.01671255   # +4.1%  mass_over_sum_pt < 0.08873
        + 0.03994508 * max(0.0, Q.psi_0p3 - 0.9924477) / 0.004013302   # +4.0%  psi_0p3 > 0.9924
        - 0.03559171 * max(0.0, 0.3017146 - Q.sd_rg) / 0.1657957   # -3.6%  sd_rg < 0.3017
        + 0.03229822 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +3.2%  tau1 < 0.06311
        + 0.02860642 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) / 0.003638607   # +2.9%  sum_z_dr2_top5 < 0.007164
        - 0.02805891 * max(0.0, 0.01083435 - Q.sum_z_dr2_top20) / 0.00529825   # -2.8%  sum_z_dr2_top20 < 0.01083
        - 0.02725818 * max(0.0, 0.01215787 - Q.sum_z_dr2_top30) / 0.005935412   # -2.7%  sum_z_dr2_top30 < 0.01216
        - 0.02353385 * max(0.0, 976.277 - Q.sum_pt_top50) / 12.87298   # -2.4%  sum_pt_top50 < 976.3
        + 0.02029759 * max(0.0, 0.1778185 - Q.sd_rg) / 0.06576479   # +2.0%  sd_rg < 0.1778
        + 0.01680754 * max(0.0, 0.03029714 - Q.e2) / 0.006822575   # +1.7%  e2 < 0.0303
        - 0.01585535 * max(0.0, 80.4 - Q.mass) / 10.6806   # -1.6%  mass < 80.4
        - 0.01522437 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1881908 - Q.sd_rg) / 0.0002726472   # -1.5%  psi_0p3 > 0.9924 and sd_rg < 0.1882
        - 0.0138213 * max(0.0, 0.076787 - Q.sum_z_dr) / 0.02105267   # -1.4%  sum_z_dr < 0.07679
        + 0.01326416 * max(0.0, 0.006374178 - Q.sum_z_dr2_top20) / 0.001987751   # +1.3%  sum_z_dr2_top20 < 0.006374
        - 0.01210587 * max(0.0, 0.076787 - Q.sum_z_dr) * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4) / 0.0007421723   # -1.2%  sum_z_dr < 0.07679 and z_dr_0p2_0p4 < 0.0518
        - 0.01075468 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.007709916 - Q.sum_z_dr2_top40) / 0.0001062984   # -1.1%  tau21_b2 < 0.3425 and sum_z_dr2_top40 < 0.00771
        + 0.009612927 * max(0.0, 1024.942 - Q.sum_pt_top40) / 38.48716   # +1.0%  sum_pt_top40 < 1025
        - 0.009243484 * max(0.0, 0.007164202 - Q.sum_z_dr2_top5) * max(0.0, Q.n_pt_above_10 - 13.0) / 0.02038471   # -0.9%  sum_z_dr2_top5 < 0.007164 and n_pt_above_10 > 13
        - 0.008226544 * max(0.0, 6.98945 - Q.log_sum_pt) / 0.06598629   # -0.8%  log_sum_pt < 6.989
        - 0.007985708 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2404747) / 0.01288721   # -0.8%  N2 < 0.4227 and max_dr > 0.2405
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 5.47;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.470258 * (-0.08326696
        + 0.1071422 * max(0.0, 79.18312 - Q.sd_mass) / 30.46368   # +10.7%  sd_mass < 79.18
        + 0.1056962 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +10.6%  log_sum_pt < 7.017
        + 0.09291385 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +9.3%  z_dr_0p1_0p2 < 0.1203
        - 0.07956807 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # -8.0%  psi_0p3 > 0.9897
        + 0.07629856 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # +7.6%  sum_z_dr2_top5 < 0.00833
        - 0.06656804 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 1260.541 - Q.sum_pt) / 87.05552   # -6.7%  z_dr_0_0p05 < 0.8459 and sum_pt < 1261
        + 0.06391665 * max(0.0, 0.007678544 - Q.sum_z_dr2_top10) / 0.00347358   # +6.4%  sum_z_dr2_top10 < 0.007679
        - 0.05607251 * max(0.0, 45.595 - Q.sd_mass) / 13.46843   # -5.6%  sd_mass < 45.59
        + 0.04419423 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) / 0.3666529   # +4.4%  z_dr_0_0p05 < 0.8459
        - 0.04020109 * max(0.0, 0.002270363 - Q.sum_z_dr2_top5) / 0.0007634099   # -4.0%  sum_z_dr2_top5 < 0.00227
        - 0.03957924 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -4.0%  tau1 < 0.07057
        - 0.03464412 * max(0.0, 1018.698 - Q.sum_pt_top40) / 34.89083   # -3.5%  sum_pt_top40 < 1019
        - 0.03244599 * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.01931881   # -3.2%  z_dr_0p1_0p2 < 0.06473
        + 0.03053419 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2) / 0.0004272674   # +3.1%  sj2_dr > 0.2232 and C2_b2 < 0.04009
        - 0.02833602 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, Q.sj3_mass1 - 5.112677) / 0.04548726   # -2.8%  sum_z_dr2_top5 < 0.00833 and sj3_mass1 > 5.113
        + 0.02096723 * max(0.0, 933.1875 - Q.sum_pt_top30) / 20.26547   # +2.1%  sum_pt_top30 < 933.2
        + 0.01967964 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +2.0%  sum_pt < 986.1
        - 0.01885801 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -1.9%  sum_pt < 1002
        - 0.01587085 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0) / 0.2039042   # -1.6%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p2_0p4 > 5
        - 0.01426135 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -1.4%  log_sum_pt < 6.903
        - 0.01225196 * max(0.0, 935.8189 - Q.sum_pt_top40) / 10.51928   # -1.2%  sum_pt_top40 < 935.8
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.633110031512605, 2.8405597689075632, 0.2374798319327731, 0.4955794642857143, 0.7452221638655462, 1.1356944327731093, 0.6482349789915967, 0.5585911764705882, 1.3794683823529412, 1.1197564075630253, 1.4056278361344539, 0.7296804621848739, 0.4698174369747899, 1.5473116071428572, 0.3553121323529412, 0.4083267857142857]
T = [4.007854254201681, 2.5131898535976886, 4.635234284729517, 4.761141432510505, 4.314530183987657]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +44%, n4 -16%, n9 +14%, n3 -9%, n5 -9%, n12 +3% ...
            + 0.4429677 * h[1] / H_AVG[1]
            - 0.1626979 * h[4] / H_AVG[4]
            + 0.1440607 * h[9] / H_AVG[9]
            - 0.09273905 * h[3] / H_AVG[3]
            - 0.08855225 * h[5] / H_AVG[5]
            + 0.02747442 * h[12] / H_AVG[12]
            + 0.02527206 * h[6] / H_AVG[6]
            + 0.01075598 * h[8] / H_AVG[8]
            - 0.005479973 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -32%, n9 +25%, n1 -21%, n11 -7%, n12 +6%, n6 +6% ...
            - 0.3150572 * h[4] / H_AVG[4]
            + 0.2506229 * h[9] / H_AVG[9]
            - 0.2119239 * h[1] / H_AVG[1]
            - 0.07258509 * h[11] / H_AVG[11]
            + 0.06426086 * h[12] / H_AVG[12]
            + 0.05642288 * h[6] / H_AVG[6]
            + 0.01181167 * h[2] / H_AVG[2]
            - 0.008739067 * h[10] / H_AVG[10]
            + 0.008576429 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -26%, n5 +14%, n14 -11%, n0 +10%, n11 +9%, n7 -8% ...
            - 0.2604043 * h[8] / H_AVG[8]
            + 0.1416484 * h[5] / H_AVG[5]
            - 0.1054001 * h[14] / H_AVG[14]
            + 0.1024398 * h[0] / H_AVG[0]
            + 0.09346837 * h[11] / H_AVG[11]
            - 0.07531863 * h[7] / H_AVG[7]
            + 0.05526584 * h[4] / H_AVG[4]
            - 0.05284452 * h[9] / H_AVG[9]
            + 0.04677563 * h[3] / H_AVG[3]
            - 0.04117663 * h[12] / H_AVG[12]
            - 0.01651724 * h[15] / H_AVG[15]
            + 0.008740591 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -27%, n0 -18%, n5 +16%, n6 -13%, n7 +11%, n15 +5% ...
            - 0.2716264 * h[8] / H_AVG[8]
            - 0.1828398 * h[0] / H_AVG[0]
            + 0.1639922 * h[5] / H_AVG[5]
            - 0.1318964 * h[6] / H_AVG[6]
            + 0.1063239 * h[7] / H_AVG[7]
            + 0.04824133 * h[15] / H_AVG[15]
            + 0.04553866 * h[3] / H_AVG[3]
            + 0.03083671 * h[12] / H_AVG[12]
            - 0.01870454 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -33%, n10 +32%, n5 -12%, n8 +7%, n12 -4%, n7 -4% ...
            - 0.3250067 * h[13] / H_AVG[13]
            + 0.3206989 * h[10] / H_AVG[10]
            - 0.123387 * h[5] / H_AVG[5]
            + 0.06744225 * h[8] / H_AVG[8]
            - 0.04083447 * h[12] / H_AVG[12]
            - 0.03641272 * h[7] / H_AVG[7]
            - 0.03548997 * h[15] / H_AVG[15]
            + 0.03238572 * h[4] / H_AVG[4]
            + 0.01834238 * h[0] / H_AVG[0]
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
