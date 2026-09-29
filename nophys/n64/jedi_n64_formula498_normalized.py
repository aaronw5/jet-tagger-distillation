"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  12.8%   (on for 53% of jets)
  neuron  1:  11.9%   (on for 95% of jets)
  neuron  5:  11.2%   (on for 79% of jets)
  neuron  4:  10.0%   (on for 68% of jets)
  neuron  0:   7.8%   (on for 61% of jets)
  neuron 13:   7.3%   (on for 89% of jets)
  neuron  9:   6.8%   (on for 65% of jets)
  neuron 10:   6.8%   (on for 79% of jets)
  neuron 12:   4.9%   (on for 53% of jets)
  neuron  7:   4.6%   (on for 38% of jets)
  neuron  6:   4.0%   (on for 62% of jets)
  neuron 11:   3.4%   (on for 69% of jets)
  neuron  3:   3.3%   (on for 51% of jets)
  neuron 15:   2.6%   (on for 67% of jets)
  neuron 14:   2.0%   (on for 36% of jets)
  neuron  2:   0.6%   (on for 39% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.0% (the network: 81.1%); same class as the network for 93.1% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
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
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft2_pt               pT [GeV] of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_12                  ΔR of particle 12 from the jet axis
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
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
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
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
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
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
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_90pct=ncum(0.9),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_3=pt[3],
        pt_9=pt[9],
        soft2_pt=softp(2, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        z_11=z[11],
        z_5=z[5],
        z_6=z[6],
        z_9=z[9],
        soft8_z=softp(8, 'z'),
        z_top10_slots=sum(pt[:10]) / tot,
        z_top2_slots=sum(pt[:2]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_12=dr[12] if pt[12] > 0 else 0.0,
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
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
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
    )


def neuron_0(Q):
    # scale S = 13.9;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.90141 * (-0.1783992
        - 0.1960063 * max(0.0, Q.mass - 79.1) / 20.79972   # -19.6%  mass > 79.1
        + 0.119739 * max(0.0, Q.mass - 64.2) / 31.40643   # +12.0%  mass > 64.2
        + 0.1123718 * max(0.0, 99.7 - Q.mass) / 22.60674   # +11.2%  mass < 99.7
        + 0.1041298 * max(0.0, 0.00926 - Q.mass_over_sum_pt_sq) / 0.00323837   # +10.4%  mass_over_sum_pt_sq < 0.00926
        + 0.09873005 * max(0.0, Q.mass_top50 - 89.2) / 14.47771   # +9.9%  mass_top50 > 89.2
        + 0.04854203 * max(0.0, 7.03 - Q.log_sum_pt) / 0.09997075   # +4.9%  log_sum_pt < 7.03
        + 0.0445867 * max(0.0, Q.mass_over_sum_pt_sq - 0.0102) / 0.003038323   # +4.5%  mass_over_sum_pt_sq > 0.0102
        - 0.03954251 * max(0.0, 81.2 - Q.mass) / 11.03808   # -4.0%  mass < 81.2
        + 0.02988376 * max(0.0, 15.8 - Q.n_dr_0p2_0p4) / 8.129675   # +3.0%  n_dr_0p2_0p4 < 15.8
        - 0.0245914 * max(0.0, Q.mass_over_sum_pt - 0.105) / 0.01055108   # -2.5%  mass_over_sum_pt > 0.105
        - 0.02243978 * max(0.0, Q.girth2_top15 - 0.00637) / 0.003229239   # -2.2%  girth2_top15 > 0.00637
        + 0.02243808 * max(0.0, Q.psi_0p3 - 0.988) / 0.007304939   # +2.2%  psi_0p3 > 0.988
        - 0.02087477 * max(0.0, 0.00609 - Q.e2_sq) / 0.00126169   # -2.1%  e2_sq < 0.00609
        - 0.01997423 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # -2.0%  sum_pt < 1010
        - 0.0198661 * max(0.0, Q.psi_0p2 - 0.923) / 0.04696714   # -2.0%  psi_0p2 > 0.923
        - 0.0196461 * max(0.0, 64.7 - Q.mass) / 5.989221   # -2.0%  mass < 64.7
        - 0.01916216 * max(0.0, 0.00619 - Q.girth2_top30) / 0.001585601   # -1.9%  girth2_top30 < 0.00619
        - 0.01390944 * max(0.0, 0.0703 - Q.tau1) / 0.01333523   # -1.4%  tau1 < 0.0703
        - 0.007938321 * max(0.0, Q.tau1 - 0.0973) / 0.01637297   # -0.8%  tau1 > 0.0973
        + 0.006504781 * max(0.0, 0.0192 - Q.e2) / 0.002477414   # +0.7%  e2 < 0.0192
        - 0.004627853 * max(0.0, 0.991 - Q.z_top50_slots) / 0.004873763   # -0.5%  z_top50_slots < 0.991
        + 0.002341795 * max(0.0, 879.0 - Q.sum_pt_top40) / 4.766361   # +0.2%  sum_pt_top40 < 879
        + 0.002153163 * max(0.0, 2.54 - Q.pt_entropy) / 0.1138099   # +0.2%  pt_entropy < 2.54
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 89.85;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 89.85184 * (-0.1669415
        + 0.5379955 * Q.sum_pt_top40 / 1021.985   # +53.8%  sum_pt_top40
        - 0.3400382 * Q.sum_pt_top50 / 1035.697   # -34.0%  sum_pt_top50
        - 0.02864118 * max(0.0, Q.sum_pt_top50 - 933.0) / 109.5091   # -2.9%  sum_pt_top50 > 933
        + 0.02025196 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # +2.0%  log_sum_pt > 6.91
        + 0.01232449 * max(0.0, Q.pt_entropy - 2.05) / 0.8264018   # +1.2%  pt_entropy > 2.05
        - 0.008693994 * max(0.0, Q.z_top40_slots - 0.969) / 0.01933592   # -0.9%  z_top40_slots > 0.969
        - 0.006244204 * max(0.0, 0.0111 - Q.mass_over_sum_pt_sq) / 0.004636804   # -0.6%  mass_over_sum_pt_sq < 0.0111
        - 0.004457312 * max(0.0, Q.log_sum_pt - 7.0) / 0.01916257   # -0.4%  log_sum_pt > 7
        - 0.004233865 * max(0.0, 0.0401 - Q.M3) / 0.01552737   # -0.4%  M3 < 0.0401
        + 0.00415291 * max(0.0, 0.966 - Q.z_top40_slots) / 0.008366516   # +0.4%  z_top40_slots < 0.966
        + 0.004127571 * max(0.0, 0.00551 - Q.girth2_top30) / 0.001274467   # +0.4%  girth2_top30 < 0.00551
        + 0.003580686 * max(0.0, Q.sj3_mass1 - 5.79) / 9.719976   # +0.4%  sj3_mass1 > 5.79
        + 0.002837641 * max(0.0, Q.z_top30_slots - 0.961) * max(0.0, 0.0852 - Q.C2) / 0.000597113   # +0.3%  z_top30_slots > 0.961 and C2 < 0.0852
        + 0.002815406 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.132 - Q.dr_0) / 0.7528852   # +0.3%  n_particles > 38 and dr_0 < 0.132
        + 0.002443338 * max(0.0, Q.n_particles - 39.2) * max(0.0, 0.176 - Q.dr_1) / 1.065721   # +0.2%  n_particles > 39.2 and dr_1 < 0.176
        - 0.002241994 * max(0.0, 34.4 - Q.sj2_mass1) / 10.8305   # -0.2%  sj2_mass1 < 34.4
        - 0.002168524 * max(0.0, Q.tau3 - 0.0163) / 0.006627409   # -0.2%  tau3 > 0.0163
        - 0.002077674 * max(0.0, 7.25 - Q.n_dr_0p2_0p4) / 2.173258   # -0.2%  n_dr_0p2_0p4 < 7.25
        + 0.002053702 * max(0.0, Q.sj3_mass2 - 3.03) / 5.524818   # +0.2%  sj3_mass2 > 3.03
        + 0.001938493 * max(0.0, 959.0 - Q.sum_pt) / 7.881319   # +0.2%  sum_pt < 959
        + 0.001891062 * max(0.0, 72.1 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 32.4) / 72.92506   # +0.2%  mass_top20 < 72.1 and n_real_top40 > 32.4
        + 0.001553887 * max(0.0, 0.000858 - Q.girth2_top10) / 0.0001535969   # +0.2%  girth2_top10 < 0.000858
        - 0.001314012 * max(0.0, Q.psi_0p3 - 0.997) / 0.001157514   # -0.1%  psi_0p3 > 0.997
        - 0.0009744788 * max(0.0, Q.z_dr_0_0p05 - 0.872) / 0.01758207   # -0.1%  z_dr_0_0p05 > 0.872
        + 0.0009478983 * max(0.0, Q.z_top30_slots - 0.942) * max(0.0, Q.max_pair_mass - 6.05) / 0.2289527   # +0.1%  z_top30_slots > 0.942 and max_pair_mass > 6.05
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 9.978;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.977764 * (0.1793989
        - 0.1124778 * max(0.0, Q.sum_pt_top50 - 930.0) / 112.2277   # -11.2%  sum_pt_top50 > 930
        - 0.091164 * max(0.0, 93.9 - Q.mass) / 18.33897   # -9.1%  mass < 93.9
        - 0.08968804 * max(0.0, Q.mass - 64.5) / 31.1807   # -9.0%  mass > 64.5
        + 0.08490666 * max(0.0, Q.sum_pt - 1020.0) / 46.54828   # +8.5%  sum_pt > 1020
        + 0.0720075 * max(0.0, 0.0251 - Q.girth2_top40) / 0.01666993   # +7.2%  girth2_top40 < 0.0251
        - 0.07012603 * max(0.0, 1280.0 - Q.sum_pt) / 242.9517   # -7.0%  sum_pt < 1280
        + 0.05209048 * max(0.0, Q.e2_sq - 0.00818) / 0.003559908   # +5.2%  e2_sq > 0.00818
        + 0.0519424 * max(0.0, 126.0 - Q.mass) / 43.18909   # +5.2%  mass < 126
        + 0.04245992 * max(0.0, 1160.0 - Q.sum_pt) / 131.1626   # +4.2%  sum_pt < 1160
        - 0.04036158 * max(0.0, 1020.0 - Q.sum_pt) / 22.75245   # -4.0%  sum_pt < 1020
        - 0.03931031 * max(0.0, 7.06 - Q.log_sum_pt) / 0.1265255   # -3.9%  log_sum_pt < 7.06
        - 0.02916379 * max(0.0, Q.girth - 0.0821) / 0.01182884   # -2.9%  girth > 0.0821
        - 0.02520618 * max(0.0, 0.000321 - Q.e3) / 0.0002418282   # -2.5%  e3 < 0.000321
        + 0.01933954 * max(0.0, 64.2 - Q.mass) / 5.86521   # +1.9%  mass < 64.2
        + 0.01849372 * max(0.0, Q.mass_top50 - 81.3) / 18.09078   # +1.8%  mass_top50 > 81.3
        - 0.01841607 * max(0.0, Q.sum_pt - 1050.0) / 34.41033   # -1.8%  sum_pt > 1050
        - 0.01664365 * max(0.0, 53.3 - Q.n_particles) / 10.57748   # -1.7%  n_particles < 53.3
        - 0.01592809 * max(0.0, 1020.0 - Q.sum_pt_top15) / 161.6752   # -1.6%  sum_pt_top15 < 1020
        - 0.01317819 * max(0.0, Q.psi_0p3 - 0.988) / 0.007304939   # -1.3%  psi_0p3 > 0.988
        + 0.01296684 * max(0.0, Q.LHA - 0.309) / 0.01984357   # +1.3%  LHA > 0.309
        + 0.01057697 * max(0.0, 781.0 - Q.sum_pt_top3) / 321.7517   # +1.1%  sum_pt_top3 < 781
        + 0.01039738 * max(0.0, 6.89 - Q.log_sum_pt) / 0.01257487   # +1.0%  log_sum_pt < 6.89
        - 0.009322794 * max(0.0, Q.sum_pt - 1120.0) / 19.83383   # -0.9%  sum_pt > 1120
        - 0.009237341 * max(0.0, Q.girth2_top15 - 0.00603) / 0.003363796   # -0.9%  girth2_top15 > 0.00603
        + 0.006076523 * max(0.0, Q.e2 - 0.0414) / 0.003259684   # +0.6%  e2 > 0.0414
        + 0.004496105 * max(0.0, Q.sum_pt_top30 - 1030.0) / 26.08202   # +0.4%  sum_pt_top30 > 1030
        - 0.004486787 * max(0.0, Q.sd_mass - 84.5) / 6.701812   # -0.4%  sd_mass > 84.5
        + 0.00447316 * max(0.0, 933.0 - Q.sum_pt) / 5.476336   # +0.4%  sum_pt < 933
        + 0.003870011 * max(0.0, Q.lam2 - 7.15e-05) / 0.001340766   # +0.4%  lam2 > 7.15e-05
        + 0.003275362 * max(0.0, 0.338 - Q.N2) / 0.06272705   # +0.3%  N2 < 0.338
        - 0.003150962 * max(0.0, Q.LHA - 0.416) / 0.002110037   # -0.3%  LHA > 0.416
        - 0.002749163 * max(0.0, Q.z_dr_0p1_0p2 - 0.418) / 0.02689264   # -0.3%  z_dr_0p1_0p2 > 0.418
        + 0.002684173 * max(0.0, 0.355 - Q.psi_0p1) / 0.02250592   # +0.3%  psi_0p1 < 0.355
        - 0.002529451 * max(0.0, Q.e3 - 0.000195) / 3.876846e-05   # -0.3%  e3 > 0.000195
        + 0.001695329 * max(0.0, Q.sum_pt_top15 - 1030.0) / 9.666052   # +0.2%  sum_pt_top15 > 1030
        + 0.00164061 * max(0.0, Q.sd_rg - 0.275) / 0.007474713   # +0.2%  sd_rg > 0.275
        - 0.001424286 * max(0.0, Q.C2 - 0.117) / 0.003137129   # -0.1%  C2 > 0.117
        + 0.0009851966 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, Q.mass_top40 - 115.0) / 0.5461144   # +0.1%  log_sum_pt > 6.89 and mass_top40 > 115
        + 0.0008031315 * max(0.0, Q.sj3_dr13 - 0.309) / 0.007284961   # +0.1%  sj3_dr13 > 0.309
        - 0.0002544415 * max(0.0, Q.mass_top20 - 116.0) / 2.188584   # -0.0%  mass_top20 > 116
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 6.165;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.164538 * (-0.1849287
        - 0.1383041 * max(0.0, Q.mass_top50 - 91.6) / 13.6195   # -13.8%  mass_top50 > 91.6
        + 0.120201 * max(0.0, Q.mass - 92.8) / 14.50065   # +12.0%  mass > 92.8
        + 0.1036852 * max(0.0, Q.tau1 - 0.0163) / 0.07054873   # +10.4%  tau1 > 0.0163
        - 0.0912617 * max(0.0, Q.girth2_top50 - 0.00631) / 0.004229972   # -9.1%  girth2_top50 > 0.00631
        + 0.06658877 * max(0.0, Q.mass_over_sum_pt - 0.09) / 0.01435276   # +6.7%  mass_over_sum_pt > 0.09
        + 0.05328526 * max(0.0, 0.0101 - Q.girth2) / 0.00386901   # +5.3%  girth2 < 0.0101
        - 0.04879122 * max(0.0, 0.00803 - Q.girth2_top20) / 0.003097583   # -4.9%  girth2_top20 < 0.00803
        + 0.03826988 * max(0.0, 61.2 - Q.n_particles) / 16.04872   # +3.8%  n_particles < 61.2
        - 0.03462491 * max(0.0, Q.mass - 80.2) / 20.13647   # -3.5%  mass > 80.2
        + 0.02735333 * max(0.0, 0.0295 - Q.e2) / 0.006435902   # +2.7%  e2 < 0.0295
        - 0.02339425 * max(0.0, 73.4 - Q.mass) / 8.336111   # -2.3%  mass < 73.4
        + 0.02332884 * max(0.0, 7.56 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979) / 0.04411397   # +2.3%  n_dr_0p2_0p4 < 7.56 and z_top50_slots > 0.979
        + 0.02324405 * max(0.0, 0.00113 - Q.lam2) / 0.0004637179   # +2.3%  lam2 < 0.00113
        + 0.02020556 * max(0.0, 0.0067 - Q.e2_sq) / 0.001560877   # +2.0%  e2_sq < 0.0067
        - 0.01908916 * max(0.0, 0.466 - Q.tau21) / 0.1089591   # -1.9%  tau21 < 0.466
        - 0.01864779 * max(0.0, Q.sj2_mass1 - 7.0) / 23.70206   # -1.9%  sj2_mass1 > 7
        + 0.0185199 * max(0.0, 66.4 - Q.mass_top5) / 40.19951   # +1.9%  mass_top5 < 66.4
        - 0.01837997 * max(0.0, Q.n_dr_0p1_0p2 - 11.8) / 3.880274   # -1.8%  n_dr_0p1_0p2 > 11.8
        - 0.01719515 * max(0.0, 14.7 - Q.n_dr_0_0p05) / 5.023704   # -1.7%  n_dr_0_0p05 < 14.7
        + 0.01683098 * max(0.0, Q.mass - 144.0) / 3.91529   # +1.7%  mass > 144
        + 0.01266885 * max(0.0, 47.7 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 780.0) / 1828.984   # +1.3%  n_particles < 47.7 and sum_pt_top30 > 780
        + 0.01153104 * max(0.0, 0.56 - Q.tau21) * max(0.0, Q.lam1 - 0.00516) / 0.0006643321   # +1.2%  tau21 < 0.56 and lam1 > 0.00516
        + 0.0112338 * max(0.0, 26.4 - Q.sj3_mass1) / 11.85808   # +1.1%  sj3_mass1 < 26.4
        + 0.009649351 * max(0.0, Q.psi_0p3 - 0.998) / 0.0006668586   # +1.0%  psi_0p3 > 0.998
        + 0.00956907 * max(0.0, 10.9 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.92) / 0.149718   # +1.0%  n_dr_0p1_0p2 < 10.9 and psi_0p2 > 0.92
        + 0.008439592 * max(0.0, 4.44 - Q.n_dr_0p2_0p4) * max(0.0, 1.02 - Q.psi_0p1) / 0.1591015   # +0.8%  n_dr_0p2_0p4 < 4.44 and psi_0p1 < 1.02
        - 0.006890632 * max(0.0, 0.984 - Q.z_top50_slots) / 0.00334469   # -0.7%  z_top50_slots < 0.984
        + 0.0038798 * max(0.0, Q.max_dr - 0.288) / 0.0766576   # +0.4%  max_dr > 0.288
        + 0.002461663 * max(0.0, 0.288 - Q.max_dr) / 0.01158398   # +0.2%  max_dr < 0.288
        + 0.002124466 * max(0.0, 3.31 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p05_0p1 - 0.521) / 0.0614852   # +0.2%  n_dr_0p2_0p4 < 3.31 and z_dr_0p05_0p1 > 0.521
        + 0.0003507482 * max(0.0, Q.n_dr_0p1_0p2 - 23.5) / 1.044541   # +0.0%  n_dr_0p1_0p2 > 23.5
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 8.708;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.70797 * (-0.004157111
        + 0.1942145 * max(0.0, 103.0 - Q.mass) / 25.0922   # +19.4%  mass < 103
        - 0.09674604 * max(0.0, 86.4 - Q.mass) / 13.67632   # -9.7%  mass < 86.4
        + 0.09127924 * max(0.0, Q.mass - 67.7) / 28.79916   # +9.1%  mass > 67.7
        - 0.09031813 * max(0.0, 79.8 - Q.mass) * max(0.0, 0.00376 - Q.lam2) / 0.03480033   # -9.0%  mass < 79.8 and lam2 < 0.00376
        - 0.06144705 * max(0.0, Q.mass_top50 - 97.5) / 12.02425   # -6.1%  mass_top50 > 97.5
        - 0.0536431 * max(0.0, Q.n_particles - 22.4) / 23.59204   # -5.4%  n_particles > 22.4
        + 0.0516434 * max(0.0, Q.sj2_dr - 0.147) / 0.06289639   # +5.2%  sj2_dr > 0.147
        + 0.04682755 * max(0.0, Q.girth2_top50 - 0.00801) / 0.003515284   # +4.7%  girth2_top50 > 0.00801
        - 0.04511854 * max(0.0, Q.sj2_dr - 0.179) / 0.04289202   # -4.5%  sj2_dr > 0.179
        + 0.03695025 * max(0.0, 80.1 - Q.mass_top40) / 12.051   # +3.7%  mass_top40 < 80.1
        + 0.03579664 * max(0.0, 14.9 - Q.n_dr_0p2_0p4) / 7.404182   # +3.6%  n_dr_0p2_0p4 < 14.9
        - 0.02512017 * max(0.0, 0.0691 - Q.tau1) / 0.0128674   # -2.5%  tau1 < 0.0691
        - 0.02437044 * max(0.0, 0.00635 - Q.girth2_top30) / 0.001671001   # -2.4%  girth2_top30 < 0.00635
        - 0.01834643 * max(0.0, Q.psi_0p2 - 0.942) / 0.03273774   # -1.8%  psi_0p2 > 0.942
        + 0.01795376 * max(0.0, 47.9 - Q.n_pt_above_1) / 9.894987   # +1.8%  n_pt_above_1 < 47.9
        - 0.01569465 * max(0.0, 0.00697 - Q.girth2_top10) / 0.002958194   # -1.6%  girth2_top10 < 0.00697
        - 0.01525148 * max(0.0, 1050.0 - Q.sum_pt) / 40.6145   # -1.5%  sum_pt < 1050
        + 0.01359716 * max(0.0, Q.n_dr_0_0p05 - 3.1) * max(0.0, 0.00123 - Q.lam2) / 0.00553288   # +1.4%  n_dr_0_0p05 > 3.1 and lam2 < 0.00123
        - 0.01109547 * max(0.0, 3.06 - Q.pt_entropy) / 0.3390142   # -1.1%  pt_entropy < 3.06
        - 0.0106388 * max(0.0, 0.989 - Q.z_top50_slots) / 0.004390631   # -1.1%  z_top50_slots < 0.989
        + 0.008608036 * max(0.0, Q.sj3_pair_mass_min - 16.1) / 12.4723   # +0.9%  sj3_pair_mass_min > 16.1
        - 0.008579039 * max(0.0, Q.LHA - 0.341) / 0.0123481   # -0.9%  LHA > 0.341
        + 0.007405311 * max(0.0, Q.psi_0p3 - 0.998) / 0.0006668586   # +0.7%  psi_0p3 > 0.998
        + 0.006030422 * max(0.0, Q.mass - 146.0) / 3.621567   # +0.6%  mass > 146
        + 0.004473154 * max(0.0, Q.log_sum_pt - 6.93) / 0.03978763   # +0.4%  log_sum_pt > 6.93
        - 0.004277786 * max(0.0, 14.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.max_dr - 0.214) / 0.7942608   # -0.4%  n_dr_0p2_0p4 < 14 and max_dr > 0.214
        - 0.002296495 * max(0.0, Q.lam1 - 0.0203) / 0.0005235028   # -0.2%  lam1 > 0.0203
        - 0.002276904 * max(0.0, Q.mass_top10 - 85.9) / 1.694633   # -0.2%  mass_top10 > 85.9
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 15.83;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.83239 * (0.2229607
        - 0.2882988 * max(0.0, 1160.0 - Q.sum_pt) / 131.1626   # -28.8%  sum_pt < 1160
        + 0.1222709 * max(0.0, 7.06 - Q.log_sum_pt) / 0.1265255   # +12.2%  log_sum_pt < 7.06
        - 0.1034732 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # -10.3%  log_sum_pt > 6.91
        + 0.08066586 * max(0.0, Q.sum_pt_top50 - 958.0) / 87.47487   # +8.1%  sum_pt_top50 > 958
        - 0.05537213 * max(0.0, 0.0806 - Q.mass_over_sum_pt) / 0.01207538   # -5.5%  mass_over_sum_pt < 0.0806
        + 0.05120283 * max(0.0, Q.mass_top50 - 39.0) / 50.04092   # +5.1%  mass_top50 > 39
        - 0.05083131 * max(0.0, Q.sum_pt - 981.0) / 73.83311   # -5.1%  sum_pt > 981
        + 0.04562122 * max(0.0, 0.00646 - Q.mass_over_sum_pt_sq) / 0.00143597   # +4.6%  mass_over_sum_pt_sq < 0.00646
        + 0.03142541 * max(0.0, 60.2 - Q.n_particles) / 15.3089   # +3.1%  n_particles < 60.2
        - 0.02413935 * max(0.0, Q.mass - 91.1) / 15.0466   # -2.4%  mass > 91.1
        + 0.01961753 * max(0.0, Q.log_sum_pt - 7.06) / 0.01113234   # +2.0%  log_sum_pt > 7.06
        + 0.01733328 * max(0.0, Q.n_pt_above_1 - 30.5) / 12.88391   # +1.7%  n_pt_above_1 > 30.5
        + 0.01715722 * max(0.0, 66.0 - Q.mass) / 6.317202   # +1.7%  mass < 66
        - 0.0155176 * max(0.0, Q.sum_pt_top20 - 842.0) / 108.2293   # -1.6%  sum_pt_top20 > 842
        - 0.01447022 * max(0.0, 769.0 - Q.sum_pt_top3) / 310.4309   # -1.4%  sum_pt_top3 < 769
        - 0.01051652 * max(0.0, 66.6 - Q.n_particles) * max(0.0, 2.47 - Q.D2) / 15.41681   # -1.1%  n_particles < 66.6 and D2 < 2.47
        - 0.008494695 * max(0.0, Q.girth2_top50 - 0.0157) / 0.001829813   # -0.8%  girth2_top50 > 0.0157
        - 0.006851799 * max(0.0, 21.6 - Q.n_dr_0p1_0p2) / 10.33146   # -0.7%  n_dr_0p1_0p2 < 21.6
        + 0.006716016 * max(0.0, Q.psi_0p3 - 0.996) * max(0.0, 3.6 - Q.D2) / 0.003020754   # +0.7%  psi_0p3 > 0.996 and D2 < 3.6
        + 0.005951793 * max(0.0, Q.sd_mass - 66.2) / 13.36611   # +0.6%  sd_mass > 66.2
        + 0.004894941 * max(0.0, 0.00489 - Q.girth2_top10) / 0.001692109   # +0.5%  girth2_top10 < 0.00489
        + 0.004661887 * max(0.0, 0.437 - Q.max_dr) / 0.09157418   # +0.5%  max_dr < 0.437
        - 0.00404802 * max(0.0, Q.mass_top50 - 150.0) / 2.409392   # -0.4%  mass_top50 > 150
        + 0.002844832 * max(0.0, Q.z_dr_0_0p05 - 0.886) / 0.01411927   # +0.3%  z_dr_0_0p05 > 0.886
        - 0.002315971 * max(0.0, 0.000636 - Q.girth2_top10) / 9.856814e-05   # -0.2%  girth2_top10 < 0.000636
        + 0.002181249 * max(0.0, Q.zdr_0 - 0.00541) / 0.005404441   # +0.2%  zdr_0 > 0.00541
        + 0.001740626 * max(0.0, Q.mass - 173.0) / 0.7195367   # +0.2%  mass > 173
        + 0.001384855 * max(0.0, 0.0159 - Q.z_11) / 0.001442471   # +0.1%  z_11 < 0.0159
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 12.52;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.5242 * (0.03465292
        - 0.0978395 * max(0.0, 104.0 - Q.mass) / 25.8515   # -9.8%  mass < 104
        - 0.09419439 * max(0.0, Q.mass_over_sum_pt_sq - 0.00623) / 0.004385535   # -9.4%  mass_over_sum_pt_sq > 0.00623
        + 0.09226288 * max(0.0, Q.mass_over_sum_pt - 0.0808) / 0.01816853   # +9.2%  mass_over_sum_pt > 0.0808
        + 0.0748829 * max(0.0, 82.6 - Q.mass) / 11.70847   # +7.5%  mass < 82.6
        - 0.06309109 * max(0.0, 118.0 - Q.mass) / 36.75187   # -6.3%  mass < 118
        + 0.06203731 * max(0.0, Q.LHA - 0.189) / 0.08849288   # +6.2%  LHA > 0.189
        - 0.05697802 * max(0.0, 7.06 - Q.log_sum_pt) / 0.1265255   # -5.7%  log_sum_pt < 7.06
        + 0.04794435 * max(0.0, 73.6 - Q.sj2_mass1) / 44.1518   # +4.8%  sj2_mass1 < 73.6
        + 0.0408445 * max(0.0, Q.lam1 - 0.00432) / 0.004409866   # +4.1%  lam1 > 0.00432
        - 0.03753716 * max(0.0, Q.e2 - 0.0267) / 0.009364994   # -3.8%  e2 > 0.0267
        + 0.03608556 * max(0.0, 0.00777 - Q.lam1) / 0.002597371   # +3.6%  lam1 < 0.00777
        - 0.03576944 * max(0.0, Q.mass_over_sum_pt - 0.1) / 0.01172731   # -3.6%  mass_over_sum_pt > 0.1
        - 0.03442855 * max(0.0, Q.mass - 65.3) / 30.58084   # -3.4%  mass > 65.3
        - 0.02650545 * max(0.0, Q.sum_pt_top40 - 953.0) / 82.37205   # -2.7%  sum_pt_top40 > 953
        - 0.02400939 * max(0.0, 0.00662 - Q.e2_sq) / 0.001518678   # -2.4%  e2_sq < 0.00662
        + 0.02287403 * max(0.0, Q.mass_top30 - 83.5) / 12.13893   # +2.3%  mass_top30 > 83.5
        + 0.02255391 * max(0.0, 0.0673 - Q.girth) / 0.01560606   # +2.3%  girth < 0.0673
        + 0.01840161 * max(0.0, 0.00585 - Q.girth2_top20) / 0.001694598   # +1.8%  girth2_top20 < 0.00585
        + 0.01567439 * max(0.0, 0.0626 - Q.tau1) / 0.01049781   # +1.6%  tau1 < 0.0626
        + 0.01272981 * max(0.0, Q.e3 - 2.64e-05) / 8.175929e-05   # +1.3%  e3 > 2.64e-05
        + 0.009554726 * max(0.0, 1020.0 - Q.sum_pt_top50) / 27.07359   # +1.0%  sum_pt_top50 < 1020
        + 0.009263242 * max(0.0, Q.tau21_b2 - 0.107) / 0.2283753   # +0.9%  tau21_b2 > 0.107
        - 0.009049609 * max(0.0, Q.sj2_mass2 - 3.52) / 8.272925   # -0.9%  sj2_mass2 > 3.52
        + 0.008777073 * max(0.0, 0.275 - Q.z_dr_0p1_0p2) / 0.1505833   # +0.9%  z_dr_0p1_0p2 < 0.275
        - 0.00793187 * max(0.0, Q.mass - 138.0) / 4.845868   # -0.8%  mass > 138
        + 0.00785436 * max(0.0, Q.e2 - 0.019) * max(0.0, Q.psi_0p3 - 0.99) / 7.180259e-05   # +0.8%  e2 > 0.019 and psi_0p3 > 0.99
        - 0.006937537 * max(0.0, 0.0412 - Q.tau1) / 0.004388236   # -0.7%  tau1 < 0.0412
        + 0.004156533 * max(0.0, Q.sum_pt - 1180.0) / 13.04692   # +0.4%  sum_pt > 1180
        + 0.004042766 * max(0.0, Q.LHA - 0.392) / 0.004480743   # +0.4%  LHA > 0.392
        - 0.004024713 * max(0.0, Q.pt_entropy - 3.0) / 0.150917   # -0.4%  pt_entropy > 3
        + 0.003070214 * max(0.0, 119.0 - Q.mass) * max(0.0, 990.0 - Q.sum_pt_top50) / 526.7391   # +0.3%  mass < 119 and sum_pt_top50 < 990
        - 0.002811296 * max(0.0, 0.0179 - Q.e2) / 0.002095787   # -0.3%  e2 < 0.0179
        + 0.002311188 * max(0.0, Q.mass_top15 - 105.0) / 1.843679   # +0.2%  mass_top15 > 105
        + 0.001847689 * max(0.0, Q.e2 - 0.0462) * max(0.0, Q.zdr_0 - -0.00249) / 5.432118e-05   # +0.2%  e2 > 0.0462 and zdr_0 > -0.00249
        + 0.0009447919 * max(0.0, 0.263 - Q.z_dr_0p1_0p2) * max(0.0, Q.sj3_dr13 - 0.263) / 0.002036619   # +0.1%  z_dr_0p1_0p2 < 0.263 and sj3_dr13 > 0.263
        + 0.0007781595 * max(0.0, Q.n_for_90pct - 36.8) / 0.2989516   # +0.1%  n_for_90pct > 36.8
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 17.43;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.43496 * (0.007628351
        + 0.1424429 * max(0.0, 103.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.963) / 0.827829   # +14.2%  mass < 103 and psi_0p3 > 0.963
        - 0.1355411 * max(0.0, 94.4 - Q.mass) * max(0.0, Q.psi_0p3 - 0.967) / 0.5457629   # -13.6%  mass < 94.4 and psi_0p3 > 0.967
        - 0.0904781 * max(0.0, 90.4 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997) / 0.02019823   # -9.0%  mass < 90.4 and psi_0p3 > 0.997
        - 0.0863921 * max(0.0, 95.5 - Q.mass_top50) / 20.21803   # -8.6%  mass_top50 < 95.5
        + 0.08078762 * max(0.0, 0.00924 - Q.girth2_top50) / 0.003290956   # +8.1%  girth2_top50 < 0.00924
        + 0.07689298 * max(0.0, 100.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997) / 0.03019428   # +7.7%  mass < 100 and psi_0p3 > 0.997
        - 0.06736976 * max(0.0, 0.00818 - Q.width) / 0.002447061   # -6.7%  width < 0.00818
        + 0.02789751 * max(0.0, 84.5 - Q.mass_top50) / 13.21718   # +2.8%  mass_top50 < 84.5
        - 0.02766104 * max(0.0, Q.girth - 0.0882) / 0.01026104   # -2.8%  girth > 0.0882
        + 0.02648864 * max(0.0, 0.0911 - Q.tau1) / 0.02332466   # +2.6%  tau1 < 0.0911
        - 0.02569297 * max(0.0, 0.00841 - Q.girth2_top30) / 0.003006416   # -2.6%  girth2_top30 < 0.00841
        - 0.02554038 * max(0.0, 0.334 - Q.LHA) / 0.08563376   # -2.6%  LHA < 0.334
        + 0.0213756 * max(0.0, 82.4 - Q.mass) / 11.61006   # +2.1%  mass < 82.4
        + 0.01995793 * max(0.0, Q.LHA - 0.318) / 0.01731172   # +2.0%  LHA > 0.318
        + 0.0175184 * max(0.0, 0.00639 - Q.e2_sq) / 0.001401067   # +1.8%  e2_sq < 0.00639
        + 0.01604532 * max(0.0, 87.5 - Q.mass_top30) / 19.03058   # +1.6%  mass_top30 < 87.5
        - 0.01538109 * max(0.0, Q.sd_rg - 0.189) / 0.02460263   # -1.5%  sd_rg > 0.189
        + 0.01345318 * max(0.0, Q.sd_rg - 0.156) / 0.03723107   # +1.3%  sd_rg > 0.156
        + 0.0120774 * max(0.0, 6.65 - Q.n_dr_0p2_0p4) / 1.863443   # +1.2%  n_dr_0p2_0p4 < 6.65
        + 0.009898505 * max(0.0, Q.e2 - 0.0332) / 0.006034267   # +1.0%  e2 > 0.0332
        - 0.00803927 * max(0.0, 91.9 - Q.mass) * max(0.0, Q.sd_mass - 48.4) / 51.53101   # -0.8%  mass < 91.9 and sd_mass > 48.4
        - 0.007637588 * max(0.0, Q.girth2_top15 - 0.00849) / 0.002616131   # -0.8%  girth2_top15 > 0.00849
        + 0.006648862 * max(0.0, 113.0 - Q.mass) * max(0.0, Q.sd_mass - 69.1) / 62.324   # +0.7%  mass < 113 and sd_mass > 69.1
        - 0.006505859 * max(0.0, 0.0813 - Q.mass_over_sum_pt) / 0.01243743   # -0.7%  mass_over_sum_pt < 0.0813
        - 0.005428208 * max(0.0, Q.sd_mass - 69.7) / 11.71295   # -0.5%  sd_mass > 69.7
        + 0.005372178 * max(0.0, 0.467 - Q.planar_flow) / 0.1163524   # +0.5%  planar_flow < 0.467
        + 0.005258811 * max(0.0, Q.sd_rg - 0.279) / 0.006945998   # +0.5%  sd_rg > 0.279
        + 0.003689947 * max(0.0, 49.7 - Q.mass) / 2.785025   # +0.4%  mass < 49.7
        - 0.003187985 * max(0.0, 93.7 - Q.mass) * max(0.0, 11.5 - Q.n_dr_0_0p05) / 15.06298   # -0.3%  mass < 93.7 and n_dr_0_0p05 < 11.5
        - 0.003086547 * max(0.0, Q.zdr_0 - 0.0112) / 0.002599702   # -0.3%  zdr_0 > 0.0112
        + 0.002317033 * max(0.0, Q.sum_pt_top30 - 1090.0) / 14.96199   # +0.2%  sum_pt_top30 > 1090
        - 0.001998117 * max(0.0, Q.sum_pt_top50 - 1180.0) / 11.6512   # -0.2%  sum_pt_top50 > 1180
        - 0.001590035 * max(0.0, Q.psi_0p3 - 0.994) * max(0.0, Q.n_dr_0p1_0p2 - 14.6) / 0.005544439   # -0.2%  psi_0p3 > 0.994 and n_dr_0p1_0p2 > 14.6
        - 0.0003470174 * max(0.0, Q.z_dr_0p1_0p2 - 0.385) / 0.03055674   # -0.0%  z_dr_0p1_0p2 > 0.385
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 14.37;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.37404 * (0.4647267
        - 0.3512555 * max(0.0, 0.0251 - Q.girth2) / 0.01633968   # -35.1%  girth2 < 0.0251
        - 0.1205924 * max(0.0, 7.06 - Q.log_sum_pt) / 0.1265255   # -12.1%  log_sum_pt < 7.06
        + 0.08363346 * max(0.0, 1050.0 - Q.sum_pt_top50) / 45.70915   # +8.4%  sum_pt_top50 < 1050
        + 0.06800819 * max(0.0, Q.mass - 73.3) / 24.74816   # +6.8%  mass > 73.3
        - 0.05343245 * max(0.0, Q.e2_sq - 0.0107) / 0.002920305   # -5.3%  e2_sq > 0.0107
        + 0.0446573 * max(0.0, 0.00796 - Q.girth2_top30) / 0.002674607   # +4.5%  girth2_top30 < 0.00796
        + 0.04408566 * max(0.0, Q.mass_over_sum_pt - 0.0871) / 0.0153808   # +4.4%  mass_over_sum_pt > 0.0871
        - 0.04195021 * max(0.0, Q.sum_pt_top50 - 1050.0) / 31.40593   # -4.2%  sum_pt_top50 > 1050
        - 0.04145932 * max(0.0, Q.mass - 100.0) / 12.57253   # -4.1%  mass > 100
        - 0.04050178 * max(0.0, Q.girth - 0.0102) / 0.05916404   # -4.1%  girth > 0.0102
        - 0.02861415 * max(0.0, 18.4 - Q.n_dr_0p2_0p4) / 10.30829   # -2.9%  n_dr_0p2_0p4 < 18.4
        + 0.02696186 * max(0.0, Q.log_sum_pt - 6.94) / 0.03523189   # +2.7%  log_sum_pt > 6.94
        - 0.01957491 * max(0.0, Q.mass - 126.0) / 6.930307   # -2.0%  mass > 126
        + 0.01123182 * max(0.0, Q.e2_sq - 0.0251) / 0.0005363675   # +1.1%  e2_sq > 0.0251
        + 0.01092686 * max(0.0, Q.sum_pt - 1160.0) / 14.95839   # +1.1%  sum_pt > 1160
        + 0.004478946 * max(0.0, Q.e2 - 0.0447) / 0.002575222   # +0.4%  e2 > 0.0447
        + 0.004057662 * max(0.0, Q.n_dr_0p1_0p2 - 18.8) / 1.772796   # +0.4%  n_dr_0p1_0p2 > 18.8
        - 0.002487114 * max(0.0, 0.969 - Q.z_top50_slots) / 0.001324069   # -0.2%  z_top50_slots < 0.969
        + 0.002090414 * max(0.0, Q.log_sum_pt - 7.03) * max(0.0, Q.sj3_pair_mass_max - 38.9) / 0.664772   # +0.2%  log_sum_pt > 7.03 and sj3_pair_mass_max > 38.9
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 16.68;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.67597 * (0.3058292
        - 0.3055557 * max(0.0, Q.mass_over_sum_pt - 0.0598) / 0.03224961   # -30.6%  mass_over_sum_pt > 0.0598
        - 0.2423822 * max(0.0, 0.00936 - Q.mass_over_sum_pt_sq) / 0.003313081   # -24.2%  mass_over_sum_pt_sq < 0.00936
        + 0.1268392 * max(0.0, Q.mass_over_sum_pt - 0.0971) / 0.01244216   # +12.7%  mass_over_sum_pt > 0.0971
        + 0.0834328 * max(0.0, 0.00908 - Q.girth2_top40) / 0.00328918   # +8.3%  girth2_top40 < 0.00908
        + 0.05973408 * max(0.0, 85.2 - Q.mass) / 13.03827   # +6.0%  mass < 85.2
        + 0.03898975 * max(0.0, 0.00355 - Q.width) / 0.0004745927   # +3.9%  width < 0.00355
        + 0.01919212 * max(0.0, 0.205 - Q.LHA) / 0.019756   # +1.9%  LHA < 0.205
        - 0.01808746 * max(0.0, 0.0364 - Q.girth) / 0.004339943   # -1.8%  girth < 0.0364
        - 0.01666401 * max(0.0, 0.0475 - Q.girth) / 0.007530854   # -1.7%  girth < 0.0475
        - 0.01613649 * max(0.0, 60.3 - Q.mass) / 4.937461   # -1.6%  mass < 60.3
        - 0.01197099 * max(0.0, Q.mass - 147.0) / 3.477839   # -1.2%  mass > 147
        + 0.009340332 * max(0.0, 957.0 - Q.sum_pt_top40) / 14.15992   # +0.9%  sum_pt_top40 < 957
        + 0.009308539 * max(0.0, Q.mass - 99.4) / 12.72368   # +0.9%  mass > 99.4
        + 0.007190191 * max(0.0, Q.psi_0p3 - 0.992) / 0.004328643   # +0.7%  psi_0p3 > 0.992
        - 0.006388551 * max(0.0, Q.mass - 155.0) / 2.40486   # -0.6%  mass > 155
        - 0.006187137 * max(0.0, 85.7 - Q.mass_top40) * max(0.0, 1030.0 - Q.sum_pt) / 484.3968   # -0.6%  mass_top40 < 85.7 and sum_pt < 1030
        + 0.004189069 * max(0.0, Q.n_dr_0p1_0p2 - 15.1) / 2.676506   # +0.4%  n_dr_0p1_0p2 > 15.1
        + 0.003844699 * max(0.0, Q.lam1 - 0.0142) * max(0.0, 3.27 - Q.soft5_pt) / 0.002357136   # +0.4%  lam1 > 0.0142 and soft5_pt < 3.27
        + 0.002684737 * max(0.0, 117.0 - Q.mass_top30) * max(0.0, Q.n_dr_0p2_0p4 - 8.2) / 53.04574   # +0.3%  mass_top30 < 117 and n_dr_0p2_0p4 > 8.2
        + 0.00245676 * max(0.0, Q.mass_top5 - 38.8) / 5.513979   # +0.2%  mass_top5 > 38.8
        + 0.002239912 * max(0.0, 2.34 - Q.pt_entropy) / 0.06841155   # +0.2%  pt_entropy < 2.34
        - 0.001914161 * max(0.0, Q.sum_pt_top15 - 949.0) / 25.13425   # -0.2%  sum_pt_top15 > 949
        - 0.001631634 * max(0.0, Q.mass_over_sum_pt - 0.172) / 0.0005010882   # -0.2%  mass_over_sum_pt > 0.172
        + 0.001469596 * max(0.0, 88.7 - Q.mass_top40) * max(0.0, 907.0 - Q.sum_pt_top50) / 117.2581   # +0.1%  mass_top40 < 88.7 and sum_pt_top50 < 907
        - 0.001401921 * max(0.0, 1000.0 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.64) / 10.0769   # -0.1%  sum_pt_top40 < 1000 and soft4_pt > 1.64
        - 0.0007679399 * max(0.0, 922.0 - Q.sum_pt_top40) * max(0.0, 20.0 - Q.n_pt_above_10) / 20.48983   # -0.1%  sum_pt_top40 < 922 and n_pt_above_10 < 20
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 19.14;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.13691 * (0.2262643
        - 0.2071862 * max(0.0, Q.LHA - 0.109) / 0.1548791   # -20.7%  LHA > 0.109
        + 0.1123207 * max(0.0, 0.0262 - Q.e2_sq) / 0.01733445   # +11.2%  e2_sq < 0.0262
        - 0.08907312 * max(0.0, Q.mass - 64.6) / 31.10555   # -8.9%  mass > 64.6
        - 0.07776947 * max(0.0, 0.067 - Q.girth) / 0.01545449   # -7.8%  girth < 0.067
        + 0.06584724 * max(0.0, Q.girth - 0.0667) / 0.01784862   # +6.6%  girth > 0.0667
        + 0.05203564 * max(0.0, Q.mass - 81.3) / 19.52552   # +5.2%  mass > 81.3
        - 0.04873707 * max(0.0, 80.3 - Q.sj2_mass1) / 50.41496   # -4.9%  sj2_mass1 < 80.3
        - 0.04191386 * max(0.0, 0.0631 - Q.e2) / 0.03273885   # -4.2%  e2 < 0.0631
        + 0.03422777 * max(0.0, 871.0 - Q.sum_pt_top5) / 283.5557   # +3.4%  sum_pt_top5 < 871
        + 0.023986 * max(0.0, Q.mass - 89.6) / 15.61285   # +2.4%  mass > 89.6
        + 0.02127709 * max(0.0, 73.9 - Q.mass) / 8.482868   # +2.1%  mass < 73.9
        - 0.01999081 * max(0.0, Q.mass - 147.0) / 3.477839   # -2.0%  mass > 147
        + 0.01910429 * max(0.0, 0.0977 - Q.z_dr_0p2_0p4) / 0.06347172   # +1.9%  z_dr_0p2_0p4 < 0.0977
        + 0.01762323 * max(0.0, 0.633 - Q.tau21) / 0.2175833   # +1.8%  tau21 < 0.633
        + 0.01724641 * max(0.0, 3.7 - Q.D2) / 1.43497   # +1.7%  D2 < 3.7
        - 0.01645955 * max(0.0, 0.0519 - Q.tau1) / 0.007142517   # -1.6%  tau1 < 0.0519
        - 0.01642268 * max(0.0, 71.5 - Q.mass_top10) / 28.06065   # -1.6%  mass_top10 < 71.5
        + 0.01432799 * max(0.0, Q.mass_top50 - 144.0) / 3.195728   # +1.4%  mass_top50 > 144
        - 0.01420947 * max(0.0, 12.2 - Q.n_dr_0p2_0p4) / 5.342346   # -1.4%  n_dr_0p2_0p4 < 12.2
        + 0.01415819 * max(0.0, Q.n_real_top40 - 33.6) / 4.391314   # +1.4%  n_real_top40 > 33.6
        - 0.01164141 * max(0.0, 3.87 - Q.D2) * max(0.0, Q.n_real_top40 - 32.2) / 8.470742   # -1.2%  D2 < 3.87 and n_real_top40 > 32.2
        - 0.01057462 * max(0.0, Q.mass_top20 - 74.4) / 10.82169   # -1.1%  mass_top20 > 74.4
        + 0.00802688 * max(0.0, Q.mass - 67.9) * max(0.0, 2.92 - Q.soft2_pt) / 52.78683   # +0.8%  mass > 67.9 and soft2_pt < 2.92
        - 0.00717497 * max(0.0, Q.mass - 159.0) / 1.925761   # -0.7%  mass > 159
        + 0.007047626 * max(0.0, Q.mass_top5 - 24.7) / 11.42964   # +0.7%  mass_top5 > 24.7
        - 0.006583001 * max(0.0, 0.445 - Q.max_dr) / 0.09919551   # -0.7%  max_dr < 0.445
        - 0.004903991 * max(0.0, Q.mass_over_sum_pt_sq - 0.0268) / 0.0003709376   # -0.5%  mass_over_sum_pt_sq > 0.0268
        - 0.0044952 * max(0.0, Q.z_top2_slots - 0.326) / 0.07352499   # -0.4%  z_top2_slots > 0.326
        - 0.00340854 * max(0.0, Q.psi_0p3 - 0.998) * max(0.0, Q.n_real_top30 - 22.2) / 0.00437778   # -0.3%  psi_0p3 > 0.998 and n_real_top30 > 22.2
        + 0.003296557 * max(0.0, 87.2 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 6.95) / 300.4091   # +0.3%  sj2_mass1 < 87.2 and sj2_mass2 > 6.95
        + 0.003098466 * max(0.0, Q.e3 - 0.000163) / 4.35993e-05   # +0.3%  e3 > 0.000163
        - 0.00256217 * max(0.0, 978.0 - Q.sum_pt) / 10.52189   # -0.3%  sum_pt < 978
        - 0.00232109 * max(0.0, Q.z_dr_0_0p05 - 0.93) / 0.005384059   # -0.2%  z_dr_0_0p05 > 0.93
        + 0.0009485829 * max(0.0, 121.0 - Q.mass_top40) * max(0.0, Q.pt1_dr01 - 11.9) / 44.38373   # +0.1%  mass_top40 < 121 and pt1_dr01 > 11.9
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 20.66;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.65916 * (0.08180392
        + 0.3041167 * max(0.0, 99.8 - Q.mass) / 22.68156   # +30.4%  mass < 99.8
        - 0.2659639 * max(0.0, 101.0 - Q.mass) / 23.58193   # -26.6%  mass < 101
        - 0.0509541 * max(0.0, Q.mass_over_sum_pt - 0.0798) / 0.01866434   # -5.1%  mass_over_sum_pt > 0.0798
        - 0.04836461 * max(0.0, Q.LHA - 0.179) / 0.09607424   # -4.8%  LHA > 0.179
        + 0.0447769 * max(0.0, Q.mass_over_sum_pt_sq - 0.00635) / 0.004322678   # +4.5%  mass_over_sum_pt_sq > 0.00635
        - 0.03906981 * max(0.0, 0.0764 - Q.girth) / 0.02080282   # -3.9%  girth < 0.0764
        + 0.03316881 * max(0.0, 71.1 - Q.mass_top10) / 27.74249   # +3.3%  mass_top10 < 71.1
        - 0.02417951 * max(0.0, 0.0421 - Q.e2) / 0.01431313   # -2.4%  e2 < 0.0421
        - 0.02316487 * max(0.0, 80.9 - Q.mass) / 10.90129   # -2.3%  mass < 80.9
        - 0.0218374 * max(0.0, 0.00771 - Q.girth2_top10) / 0.003497226   # -2.2%  girth2_top10 < 0.00771
        + 0.01697287 * max(0.0, 0.0267 - Q.e2) / 0.005179398   # +1.7%  e2 < 0.0267
        - 0.01532599 * max(0.0, Q.mass_top40 - 84.1) / 14.86489   # -1.5%  mass_top40 > 84.1
        + 0.01123035 * max(0.0, 9.27 - Q.n_dr_0p2_0p4) * max(0.0, 21.1 - Q.n_dr_0p1_0p2) / 38.3487   # +1.1%  n_dr_0p2_0p4 < 9.27 and n_dr_0p1_0p2 < 21.1
        - 0.01073952 * max(0.0, 40.4 - Q.mass_top10) / 9.60474   # -1.1%  mass_top10 < 40.4
        + 0.009875679 * max(0.0, 0.00342 - Q.girth2_top10) / 0.001025242   # +1.0%  girth2_top10 < 0.00342
        - 0.006612194 * max(0.0, Q.psi_0p3 - 0.988) / 0.007304939   # -0.7%  psi_0p3 > 0.988
        + 0.006319039 * max(0.0, 5.46 - Q.n_dr_0p2_0p4) / 1.30546   # +0.6%  n_dr_0p2_0p4 < 5.46
        + 0.006270533 * max(0.0, Q.mass - 131.0) / 6.025299   # +0.6%  mass > 131
        - 0.006265171 * max(0.0, 8.29 - Q.n_dr_0p2_0p4) * max(0.0, 0.00605 - Q.girth2) / 0.004285866   # -0.6%  n_dr_0p2_0p4 < 8.29 and girth2 < 0.00605
        - 0.006048204 * max(0.0, Q.n_for_90pct - 19.1) / 4.861898   # -0.6%  n_for_90pct > 19.1
        + 0.00603617 * max(0.0, 102.0 - Q.mass) * max(0.0, Q.mass_top10 - 40.4) / 107.5019   # +0.6%  mass < 102 and mass_top10 > 40.4
        + 0.005823467 * max(0.0, 0.0348 - Q.girth) / 0.003944522   # +0.6%  girth < 0.0348
        - 0.005730878 * max(0.0, Q.n_dr_0p1_0p2 - 8.59) / 5.532482   # -0.6%  n_dr_0p1_0p2 > 8.59
        - 0.005670827 * max(0.0, Q.n_dr_0p2_0p4 - 11.3) / 2.456069   # -0.6%  n_dr_0p2_0p4 > 11.3
        + 0.005612956 * max(0.0, Q.girth2_top15 - 0.00805) / 0.002715666   # +0.6%  girth2_top15 > 0.00805
        + 0.00381571 * max(0.0, Q.z_top30_slots - 0.976) * max(0.0, 0.053 - Q.C2_b2) / 0.0003488025   # +0.4%  z_top30_slots > 0.976 and C2_b2 < 0.053
        + 0.003199262 * max(0.0, Q.psi_0p3 - 0.997) / 0.001157514   # +0.3%  psi_0p3 > 0.997
        - 0.003183573 * max(0.0, 0.138 - Q.z_dr_0_0p05) / 0.03461575   # -0.3%  z_dr_0_0p05 < 0.138
        + 0.002700851 * max(0.0, 0.198 - Q.tau21_b2) * max(0.0, Q.z_5 - 0.0251) / 0.0008637353   # +0.3%  tau21_b2 < 0.198 and z_5 > 0.0251
        + 0.00233166 * max(0.0, 0.00902 - Q.mass_over_sum_pt_sq) * max(0.0, Q.z_dr_0p1_0p2 - 0.0762) / 7.988414e-05   # +0.2%  mass_over_sum_pt_sq < 0.00902 and z_dr_0p1_0p2 > 0.0762
        + 0.001808651 * max(0.0, 0.166 - Q.z_dr_0_0p05) * max(0.0, Q.max_dr - 0.235) / 0.00530003   # +0.2%  z_dr_0_0p05 < 0.166 and max_dr > 0.235
        + 0.001487451 * max(0.0, Q.pt1_dr01 - 9.57) / 2.983444   # +0.1%  pt1_dr01 > 9.57
        - 0.001342386 * max(0.0, 80.2 - Q.mass) * max(0.0, 4.32 - Q.D2) / 7.110913   # -0.1%  mass < 80.2 and D2 < 4.32
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 3.072;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.07166 * (0.005176354
        + 0.1704342 * max(0.0, 88.2 - Q.mass) / 14.66431   # +17.0%  mass < 88.2
        + 0.1668418 * max(0.0, 80.7 - Q.mass) / 10.81184   # +16.7%  mass < 80.7
        - 0.1605059 * max(0.0, Q.mass_over_sum_pt - 0.0773) / 0.02004145   # -16.1%  mass_over_sum_pt > 0.0773
        + 0.1013596 * max(0.0, Q.mass - 106.0) / 11.11937   # +10.1%  mass > 106
        - 0.07126908 * max(0.0, 61.7 - Q.mass) / 5.262365   # -7.1%  mass < 61.7
        - 0.06196551 * max(0.0, Q.pt_entropy - 1.99) / 0.8811897   # -6.2%  pt_entropy > 1.99
        + 0.06191411 * max(0.0, Q.LHA - 0.336) / 0.01329924   # +6.2%  LHA > 0.336
        - 0.0560602 * max(0.0, 79.3 - Q.mass) * max(0.0, 0.0962 - Q.z_dr_0p1_0p2) / 0.8009203   # -5.6%  mass < 79.3 and z_dr_0p1_0p2 < 0.0962
        + 0.04701683 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +4.7%  n_dr_0p2_0p4 < 15
        - 0.03304585 * max(0.0, Q.sum_pt - 933.0) / 116.2722   # -3.3%  sum_pt > 933
        - 0.02873439 * max(0.0, Q.lam2 - 0.00046) / 0.001032307   # -2.9%  lam2 > 0.00046
        - 0.02175846 * max(0.0, 90.1 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3) / 0.05967374   # -2.2%  mass < 90.1 and psi_0p3 < 1
        - 0.01733154 * max(0.0, 0.999 - Q.z_top50_slots) / 0.007302687   # -1.7%  z_top50_slots < 0.999
        - 0.001762452 * max(0.0, Q.mass - 187.0) / 0.3682757   # -0.2%  mass > 187
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 25.21;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.21062 * (0.1308972
        - 0.4140459 * max(0.0, 7.06 - Q.log_sum_pt) / 0.1265255   # -41.4%  log_sum_pt < 7.06
        + 0.3631464 * max(0.0, 1160.0 - Q.sum_pt) / 131.1626   # +36.3%  sum_pt < 1160
        + 0.02713684 * max(0.0, Q.mass - 76.3) / 22.65353   # +2.7%  mass > 76.3
        - 0.02614475 * max(0.0, Q.mass - 146.0) / 3.621567   # -2.6%  mass > 146
        - 0.01872381 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # -1.9%  sum_pt < 1010
        + 0.01558313 * max(0.0, Q.mass_top50 - 142.0) / 3.47664   # +1.6%  mass_top50 > 142
        - 0.01326939 * max(0.0, Q.mass - 102.0) / 12.07688   # -1.3%  mass > 102
        + 0.01203525 * max(0.0, 1010.0 - Q.sum_pt_top50) / 22.47526   # +1.2%  sum_pt_top50 < 1010
        - 0.0113088 * max(0.0, Q.sum_pt_top10 - 557.0) / 222.7358   # -1.1%  sum_pt_top10 > 557
        - 0.01083274 * max(0.0, Q.sum_pt_top30 - 943.0) / 72.633   # -1.1%  sum_pt_top30 > 943
        + 0.01061067 * max(0.0, 951.0 - Q.sum_pt) / 7.021037   # +1.1%  sum_pt < 951
        - 0.01033092 * max(0.0, 952.0 - Q.sum_pt_top15) / 107.1806   # -1.0%  sum_pt_top15 < 952
        - 0.009485565 * max(0.0, 6.86 - Q.log_sum_pt) / 0.008420316   # -0.9%  log_sum_pt < 6.86
        - 0.008254653 * max(0.0, Q.girth2_top50 - 0.0079) / 0.003551278   # -0.8%  girth2_top50 > 0.0079
        + 0.007644121 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 0.529 - Q.D3) / 19.80607   # +0.8%  sum_pt_top40 < 1040 and D3 < 0.529
        + 0.006158941 * max(0.0, 0.00841 - Q.girth2_top40) / 0.002787625   # +0.6%  girth2_top40 < 0.00841
        + 0.005860682 * max(0.0, Q.girth - 0.0703) / 0.01613007   # +0.6%  girth > 0.0703
        - 0.004995037 * max(0.0, Q.mass - 157.0) / 2.16   # -0.5%  mass > 157
        - 0.003972475 * max(0.0, 24.4 - Q.n_dr_0p1_0p2) / 12.77405   # -0.4%  n_dr_0p1_0p2 < 24.4
        - 0.003800288 * max(0.0, 13.7 - Q.n_for_90pct) / 1.057479   # -0.4%  n_for_90pct < 13.7
        - 0.002223904 * max(0.0, Q.sj2_zsoft - 0.117) / 0.1419392   # -0.2%  sj2_zsoft > 0.117
        + 0.001990214 * max(0.0, Q.mass - 173.0) * max(0.0, 39.5 - Q.pt_9) / 5.708138   # +0.2%  mass > 173 and pt_9 < 39.5
        + 0.001823062 * max(0.0, 1030.0 - Q.sum_pt) * max(0.0, 0.122 - Q.dr_12) / 1.259192   # +0.2%  sum_pt < 1030 and dr_12 < 0.122
        + 0.001767389 * max(0.0, Q.sum_pt_top20 - 1060.0) / 11.54325   # +0.2%  sum_pt_top20 > 1060
        + 0.001706976 * max(0.0, Q.mass_top5 - 33.9) / 7.318696   # +0.2%  mass_top5 > 33.9
        + 0.001473452 * max(0.0, Q.psi_0p3 - 0.981) * max(0.0, Q.max_dr - 0.224) / 0.001522403   # +0.1%  psi_0p3 > 0.981 and max_dr > 0.224
        + 0.001288482 * max(0.0, Q.mass_over_sum_pt_sq - 0.0281) / 0.000268458   # +0.1%  mass_over_sum_pt_sq > 0.0281
        + 0.001143668 * max(0.0, 13.8 - Q.n_for_90pct) * max(0.0, 3.95 - Q.D2) / 0.9675366   # +0.1%  n_for_90pct < 13.8 and D2 < 3.95
        - 0.0008417138 * max(0.0, Q.mass - 174.0) * max(0.0, 0.0293 - Q.z_9) / 0.002194429   # -0.1%  mass > 174 and z_9 < 0.0293
        - 0.0008040849 * max(0.0, 0.132 - Q.sj2_zsoft) / 0.01859769   # -0.1%  sj2_zsoft < 0.132
        + 0.0007605256 * max(0.0, Q.lam1 - 0.018) / 0.000792286   # +0.1%  lam1 > 0.018
        + 0.0006037707 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.0174 - Q.z_9) / 0.01435984   # +0.1%  sum_pt < 1010 and z_9 < 0.0174
        - 0.0002323765 * max(0.0, Q.log_sum_pt - 7.21) * max(0.0, 0.00379 - Q.C3) / 5.007141e-06   # -0.0%  log_sum_pt > 7.21 and C3 < 0.00379
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 21.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.5726 * (-0.08436629
        + 0.1691168 * max(0.0, Q.girth - 0.0351) / 0.03816202   # +16.9%  girth > 0.0351
        + 0.1125614 * max(0.0, 0.00949 - Q.mass_over_sum_pt_sq) / 0.003410452   # +11.3%  mass_over_sum_pt_sq < 0.00949
        - 0.0855178 * max(0.0, Q.LHA - 0.187) / 0.08999224   # -8.6%  LHA > 0.187
        - 0.06380113 * max(0.0, 90.8 - Q.mass) / 16.2115   # -6.4%  mass < 90.8
        - 0.05435274 * max(0.0, 0.091 - Q.mass_over_sum_pt) / 0.0182637   # -5.4%  mass_over_sum_pt < 0.091
        - 0.03428844 * max(0.0, 1070.0 - Q.sum_pt) / 55.2008   # -3.4%  sum_pt < 1070
        - 0.03420172 * max(0.0, Q.mass_over_sum_pt - 0.0971) / 0.01244216   # -3.4%  mass_over_sum_pt > 0.0971
        + 0.03383249 * max(0.0, 65.4 - Q.sj2_mass1) / 36.67612   # +3.4%  sj2_mass1 < 65.4
        - 0.03253942 * max(0.0, 0.00873 - Q.girth2_top40) / 0.003025688   # -3.3%  girth2_top40 < 0.00873
        + 0.03142354 * max(0.0, Q.tau1 - 0.0813) / 0.0232951   # +3.1%  tau1 > 0.0813
        + 0.02732846 * max(0.0, 1140.0 - Q.sum_pt_top40) / 131.302   # +2.7%  sum_pt_top40 < 1140
        - 0.02670801 * max(0.0, Q.LHA - 0.271) / 0.03578641   # -2.7%  LHA > 0.271
        - 0.02448437 * max(0.0, Q.mass - 96.1) / 13.57818   # -2.4%  mass > 96.1
        - 0.02386158 * max(0.0, 0.0109 - Q.girth2_top20) / 0.005350897   # -2.4%  girth2_top20 < 0.0109
        - 0.0213585 * max(0.0, Q.sum_pt_top40 - 972.0) / 67.46095   # -2.1%  sum_pt_top40 > 972
        + 0.01465356 * max(0.0, Q.mass_top50 - 78.6) / 19.63449   # +1.5%  mass_top50 > 78.6
        + 0.014501 * max(0.0, 1050.0 - Q.sum_pt_top40) / 55.46527   # +1.5%  sum_pt_top40 < 1050
        + 0.0144291 * max(0.0, Q.log_sum_pt - 6.97) / 0.0255142   # +1.4%  log_sum_pt > 6.97
        + 0.01370422 * max(0.0, 81.5 - Q.mass_top40) / 12.74291   # +1.4%  mass_top40 < 81.5
        + 0.01261732 * max(0.0, 0.0304 - Q.e2) / 0.006873443   # +1.3%  e2 < 0.0304
        + 0.01107888 * max(0.0, Q.z_top40_slots - 0.968) / 0.02008405   # +1.1%  z_top40_slots > 0.968
        + 0.01068405 * max(0.0, Q.mass_top30 - 79.4) / 13.8845   # +1.1%  mass_top30 > 79.4
        - 0.01035513 * max(0.0, Q.LHA - 0.322) / 0.01630562   # -1.0%  LHA > 0.322
        + 0.008857339 * max(0.0, 0.00627 - Q.girth2_top20) / 0.001926167   # +0.9%  girth2_top20 < 0.00627
        - 0.008736793 * max(0.0, 0.455 - Q.tau21) / 0.102992   # -0.9%  tau21 < 0.455
        - 0.00791938 * max(0.0, Q.e2 - 0.0302) / 0.007460331   # -0.8%  e2 > 0.0302
        + 0.007391968 * max(0.0, Q.tau21_b2 - 0.275) / 0.1236155   # +0.7%  tau21_b2 > 0.275
        - 0.007326749 * max(0.0, 945.0 - Q.sum_pt_top20) / 59.41993   # -0.7%  sum_pt_top20 < 945
        + 0.006852294 * max(0.0, 1020.0 - Q.sum_pt_top50) / 27.07359   # +0.7%  sum_pt_top50 < 1020
        + 0.00678063 * max(0.0, 94.8 - Q.mass) * max(0.0, 0.00256 - Q.soft8_z) / 0.02098648   # +0.7%  mass < 94.8 and soft8_z < 0.00256
        - 0.006303621 * max(0.0, Q.sj2_mass2 - 6.11) / 6.181158   # -0.6%  sj2_mass2 > 6.11
        - 0.006244092 * max(0.0, Q.log_sum_pt - 7.06) / 0.01113234   # -0.6%  log_sum_pt > 7.06
        + 0.006192043 * max(0.0, Q.sum_pt - 1160.0) / 14.95839   # +0.6%  sum_pt > 1160
        - 0.006147624 * max(0.0, 2.03 - Q.soft6_pt) / 0.6943467   # -0.6%  soft6_pt < 2.03
        + 0.0058007 * max(0.0, 0.00722 - Q.girth2_top5) / 0.003680475   # +0.6%  girth2_top5 < 0.00722
        + 0.005364686 * max(0.0, Q.psi_0p3 - 0.992) * max(0.0, 4.18 - Q.soft6_pt) / 0.01033305   # +0.5%  psi_0p3 > 0.992 and soft6_pt < 4.18
        + 0.005226973 * max(0.0, 79.1 - Q.mass) / 10.1585   # +0.5%  mass < 79.1
        + 0.004871726 * max(0.0, 57.3 - Q.mass) / 4.272186   # +0.5%  mass < 57.3
        + 0.003856082 * max(0.0, Q.psi_0p3 - 0.993) * max(0.0, 1.09 - Q.N3) / 0.00241118   # +0.4%  psi_0p3 > 0.993 and N3 < 1.09
        + 0.003082586 * max(0.0, Q.sj2_dr - 0.143) / 0.06584099   # +0.3%  sj2_dr > 0.143
        + 0.002942217 * max(0.0, 0.371 - Q.tau21_b2) * max(0.0, 0.0574 - Q.z_6) / 0.002258764   # +0.3%  tau21_b2 < 0.371 and z_6 < 0.0574
        - 0.002812511 * max(0.0, 0.97 - Q.z_top40_slots) / 0.009363142   # -0.3%  z_top40_slots < 0.97
        - 0.002487342 * max(0.0, 0.355 - Q.tau21_b2) * max(0.0, 0.0089 - Q.girth2_top40) / 0.0002146337   # -0.2%  tau21_b2 < 0.355 and girth2_top40 < 0.0089
        - 0.002137232 * max(0.0, Q.sum_pt_top40 - 1140.0) / 13.28693   # -0.2%  sum_pt_top40 > 1140
        + 0.001047552 * max(0.0, Q.mass_top10 - 84.9) / 1.793526   # +0.1%  mass_top10 > 84.9
        - 0.0009638169 * max(0.0, Q.z_top10_slots - 0.914) / 0.004442742   # -0.1%  z_top10_slots > 0.914
        + 0.000941301 * max(0.0, Q.psi_0p3 - 0.996) * max(0.0, Q.pt_3 - 62.1) / 0.02770301   # +0.1%  psi_0p3 > 0.996 and pt_3 > 62.1
        - 0.0009350066 * max(0.0, 0.0122 - Q.girth2_top20) * max(0.0, Q.mass_top3 - 26.0) / 0.01061606   # -0.1%  girth2_top20 < 0.0122 and mass_top3 > 26
        + 0.0006314048 * max(0.0, Q.sum_pt_top20 - 1060.0) / 11.54325   # +0.1%  sum_pt_top20 > 1060
        - 0.0005597617 * max(0.0, 0.969 - Q.z_top50_slots) / 0.001324069   # -0.1%  z_top50_slots < 0.969
        - 0.000186876 * max(0.0, Q.mass_top30 - 145.0) / 1.161787   # -0.0%  mass_top30 > 145
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 9.084;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.083633 * (-0.007166736
        - 0.1174274 * max(0.0, 0.303 - Q.LHA) / 0.06274513   # -11.7%  LHA < 0.303
        + 0.1006941 * max(0.0, 0.0795 - Q.girth) / 0.02286671   # +10.1%  girth < 0.0795
        + 0.07453737 * max(0.0, 7.01 - Q.log_sum_pt) / 0.08287272   # +7.5%  log_sum_pt < 7.01
        + 0.0610639 * max(0.0, 0.0553 - Q.e2) / 0.02556139   # +6.1%  e2 < 0.0553
        - 0.05942916 * max(0.0, Q.girth - 0.0399) / 0.03460466   # -5.9%  girth > 0.0399
        - 0.04438948 * max(0.0, Q.sd_mass - 58.2) / 17.53121   # -4.4%  sd_mass > 58.2
        - 0.0360032 * max(0.0, 0.0722 - Q.tau1) / 0.01409655   # -3.6%  tau1 < 0.0722
        + 0.03448575 * max(0.0, 0.0022 - Q.lam2) / 0.001299817   # +3.4%  lam2 < 0.0022
        + 0.03001832 * max(0.0, Q.sd_rg - 0.215) / 0.01793917   # +3.0%  sd_rg > 0.215
        + 0.02952896 * max(0.0, 0.00842 - Q.girth2_top5) / 0.0046167   # +3.0%  girth2_top5 < 0.00842
        - 0.02792444 * max(0.0, 88.5 - Q.mass) / 14.83365   # -2.8%  mass < 88.5
        + 0.02368798 * max(0.0, 0.218 - Q.z_dr_0p1_0p2) / 0.1081271   # +2.4%  z_dr_0p1_0p2 < 0.218
        + 0.02365445 * max(0.0, 0.203 - Q.LHA) / 0.01918468   # +2.4%  LHA < 0.203
        - 0.02230831 * max(0.0, 0.00824 - Q.girth2_top50) / 0.002552148   # -2.2%  girth2_top50 < 0.00824
        - 0.02171305 * max(0.0, Q.psi_0p3 - 0.988) / 0.007304939   # -2.2%  psi_0p3 > 0.988
        + 0.02166744 * max(0.0, Q.e3 - 9.44e-06) / 9.417181e-05   # +2.2%  e3 > 9.44e-06
        - 0.01995375 * max(0.0, Q.psi_0p1 - 0.91) / 0.01996173   # -2.0%  psi_0p1 > 0.91
        + 0.01949809 * max(0.0, Q.mass_top30 - 79.5) / 13.83699   # +1.9%  mass_top30 > 79.5
        + 0.01846182 * max(0.0, 0.076 - Q.z_dr_0p1_0p2) / 0.02451761   # +1.8%  z_dr_0p1_0p2 < 0.076
        - 0.01823044 * max(0.0, 1040.0 - Q.sum_pt_top40) / 48.2795   # -1.8%  sum_pt_top40 < 1040
        + 0.01822348 * max(0.0, Q.sd_mass - 86.2) / 6.318147   # +1.8%  sd_mass > 86.2
        - 0.018152 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # -1.8%  sum_pt < 1000
        - 0.01680709 * max(0.0, Q.e2 - 0.0388) / 0.003955167   # -1.7%  e2 > 0.0388
        + 0.01612293 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # +1.6%  n_dr_0p1_0p2 < 17
        - 0.01508865 * max(0.0, Q.mass_top10 - 24.0) / 26.98027   # -1.5%  mass_top10 > 24
        - 0.01384338 * max(0.0, 0.0025 - Q.girth2_top5) / 0.0008672286   # -1.4%  girth2_top5 < 0.0025
        - 0.0133571 * max(0.0, Q.sd_rg - 0.164) / 0.03360969   # -1.3%  sd_rg > 0.164
        + 0.01225771 * max(0.0, 0.00457 - Q.lam1) / 0.0009202027   # +1.2%  lam1 < 0.00457
        + 0.008555286 * max(0.0, Q.z_top20_slots - 0.9) / 0.03224609   # +0.9%  z_top20_slots > 0.9
        - 0.008184361 * max(0.0, Q.sd_rg - 0.287) / 0.005947499   # -0.8%  sd_rg > 0.287
        - 0.007508416 * max(0.0, 0.12 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 3.57) / 0.2489186   # -0.8%  z_dr_0p1_0p2 < 0.12 and n_dr_0p2_0p4 > 3.57
        - 0.005651236 * max(0.0, 0.915 - Q.z_top20_slots) / 0.04983859   # -0.6%  z_top20_slots < 0.915
        + 0.005565294 * max(0.0, 39.6 - Q.n_particles) / 3.560077   # +0.6%  n_particles < 39.6
        - 0.005229188 * max(0.0, 16.4 - Q.n_dr_0p1_0p2) * max(0.0, 0.0316 - Q.absphi_0) / 0.1034859   # -0.5%  n_dr_0p1_0p2 < 16.4 and absphi_0 < 0.0316
        + 0.005044791 * max(0.0, Q.sj2_dr - 0.231) * max(0.0, 0.0423 - Q.C2_b2) / 0.0004128381   # +0.5%  sj2_dr > 0.231 and C2_b2 < 0.0423
        + 0.004763987 * max(0.0, Q.n_dr_0p1_0p2 - 11.6) / 3.97012   # +0.5%  n_dr_0p1_0p2 > 11.6
        - 0.003917555 * max(0.0, 0.84 - Q.soft2_pt) / 0.2597491   # -0.4%  soft2_pt < 0.84
        - 0.003764648 * max(0.0, 0.715 - Q.psi_0p1) / 0.07565638   # -0.4%  psi_0p1 < 0.715
        + 0.003080747 * max(0.0, 0.00265 - Q.girth2_top20) * max(0.0, 0.428 - Q.max_dr) / 3.454861e-05   # +0.3%  girth2_top20 < 0.00265 and max_dr < 0.428
        + 0.002906664 * max(0.0, 0.0062 - Q.girth2_top5) * max(0.0, 0.0461 - Q.z_6) / 3.135757e-05   # +0.3%  girth2_top5 < 0.0062 and z_6 < 0.0461
        - 0.002812301 * max(0.0, 0.299 - Q.max_dr) / 0.01366092   # -0.3%  max_dr < 0.299
        + 0.002723153 * max(0.0, 1.05 - Q.z_dr_0_0p05) * max(0.0, 968.0 - Q.sum_pt) / 5.319597   # +0.3%  z_dr_0_0p05 < 1.05 and sum_pt < 968
        - 0.001762631 * max(0.0, 0.00822 - Q.girth2_top5) * max(0.0, 0.95 - Q.psi_0p2) / 3.867414e-05   # -0.2%  girth2_top5 < 0.00822 and psi_0p2 < 0.95
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6547550420168067, 2.791848844537815, 0.2155186974789916, 0.3859709558823529, 0.7718283613445378, 1.0429287815126052, 0.5455136554621849, 0.48648487394957984, 1.1705360294117646, 0.9979055672268907, 1.2643392857142857, 0.760580462184874, 0.5527052521008403, 1.536340756302521, 0.27038723739495796, 0.4410086134453782]
T = [3.821305217633928, 2.469314595916492, 4.239449018513655, 4.354492059480042, 4.10870437319459]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +46%, n4 -18%, n9 +13%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.4566255 * h[1] / H_AVG[1]
            - 0.1767328 * h[4] / H_AVG[4]
            + 0.1346517 * h[9] / H_AVG[9]
            - 0.08528899 * h[5] / H_AVG[5]
            - 0.07575375 * h[3] / H_AVG[3]
            + 0.03389949 * h[12] / H_AVG[12]
            + 0.0223056 * h[6] / H_AVG[6]
            + 0.009572449 * h[8] / H_AVG[8]
            - 0.005169778 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +23%, n1 -21%, n11 -8%, n12 +8%, n6 +5% ...
            - 0.3321033 * h[4] / H_AVG[4]
            + 0.2273189 * h[9] / H_AVG[9]
            - 0.2119907 * h[1] / H_AVG[1]
            - 0.0770032 * h[11] / H_AVG[11]
            + 0.07694136 * h[12] / H_AVG[12]
            + 0.0483256 * h[6] / H_AVG[6]
            + 0.01090984 * h[2] / H_AVG[2]
            - 0.008000318 * h[10] / H_AVG[10]
            + 0.007406762 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +14%, n0 +12%, n11 +11%, n14 -9%, n7 -7% ...
            - 0.2415925 * h[8] / H_AVG[8]
            + 0.1422221 * h[5] / H_AVG[5]
            + 0.1158326 * h[0] / H_AVG[0]
            + 0.106522 * h[11] / H_AVG[11]
            - 0.08769594 * h[14] / H_AVG[14]
            - 0.07171994 * h[7] / H_AVG[7]
            + 0.06258266 * h[4] / H_AVG[4]
            - 0.05296361 * h[12] / H_AVG[12]
            - 0.05149062 * h[9] / H_AVG[9]
            + 0.03983119 * h[3] / H_AVG[3]
            - 0.01950468 * h[15] / H_AVG[15]
            + 0.008042225 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -25%, n0 -21%, n5 +16%, n6 -12%, n7 +10%, n15 +6% ...
            - 0.2520105 * h[8] / H_AVG[8]
            - 0.2067493 * h[0] / H_AVG[0]
            + 0.1646607 * h[5] / H_AVG[5]
            - 0.1213612 * h[6] / H_AVG[6]
            + 0.1012465 * h[7] / H_AVG[7]
            + 0.05696815 * h[15] / H_AVG[15]
            + 0.03966488 * h[12] / H_AVG[12]
            + 0.03877887 * h[3] / H_AVG[3]
            - 0.01856003 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +30%, n5 -12%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3388681 * h[13] / H_AVG[13]
            + 0.302914 * h[10] / H_AVG[10]
            - 0.1189847 * h[5] / H_AVG[5]
            + 0.06009436 * h[8] / H_AVG[8]
            - 0.05044521 * h[12] / H_AVG[12]
            - 0.0402507 * h[15] / H_AVG[15]
            + 0.03522225 * h[4] / H_AVG[4]
            - 0.03330098 * h[7] / H_AVG[7]
            + 0.01991975 * h[0] / H_AVG[0]
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
