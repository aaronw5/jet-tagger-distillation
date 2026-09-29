"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  14.1%   (on for 63% of jets)
  neuron  1:  11.4%   (on for 93% of jets)
  neuron  5:  11.1%   (on for 76% of jets)
  neuron  4:   9.7%   (on for 64% of jets)
  neuron  0:   7.3%   (on for 52% of jets)
  neuron  9:   6.6%   (on for 51% of jets)
  neuron 10:   6.5%   (on for 74% of jets)
  neuron 13:   6.5%   (on for 88% of jets)
  neuron  7:   5.4%   (on for 44% of jets)
  neuron  6:   4.7%   (on for 65% of jets)
  neuron  3:   4.6%   (on for 75% of jets)
  neuron 12:   3.8%   (on for 37% of jets)
  neuron 11:   2.8%   (on for 65% of jets)
  neuron 15:   2.6%   (on for 66% of jets)
  neuron 14:   2.4%   (on for 39% of jets)
  neuron  2:   0.6%   (on for 49% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.8% (the network: 81.1%); same class as the network for 92.5% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
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
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_11                  pT of particle 11 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
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
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
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
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
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
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_11=pt[11],
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft5_pt=softp(5, 'pt'),
        z_11=z[11],
        z_top10_slots=sum(pt[:10]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        zdr_2=z[2] * dr[2],
        zdr_5=z[5] * dr[5],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_mass=softdrop("mass"),
        sj2_dr=subjets(2)["dr"][0],
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
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
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
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
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    # scale S = 16.27;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.27152 * (0.07928771
        - 0.2044736 * max(0.0, Q.mass - 78.26182) / 21.33658   # -20.4%  mass > 78.26
        - 0.1758998 * max(0.0, Q.mass - 87.36377) / 16.57741   # -17.6%  mass > 87.36
        + 0.06288064 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq) / 0.002241068   # +6.3%  mass_over_sum_pt_sq < 0.007873
        - 0.05752848 * max(0.0, Q.mass - 92.85979) / 14.48273   # -5.8%  mass > 92.86
        + 0.05363548 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq) / 0.001689525   # +5.4%  mass_over_sum_pt_sq < 0.006939
        - 0.04633908 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -4.6%  e2_sq < 0.006167
        + 0.04364716 * max(0.0, Q.mass_top50 - 71.79516) / 24.22501   # +4.4%  mass_top50 > 71.8
        + 0.04238356 * max(0.0, Q.mass - 91.03469) / 15.0694   # +4.2%  mass > 91.03
        - 0.03850099 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -3.9%  mass < 101
        - 0.03609491 * max(0.0, Q.mass - 74.25181) / 24.07648   # -3.6%  mass > 74.25
        - 0.02912071 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -2.9%  girth2_top30 < 0.006364
        - 0.02765656 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # -2.8%  girth2_top20 < 0.006374
        + 0.02342694 * max(0.0, 0.007856958 - Q.girth2_top30) / 0.002601721   # +2.3%  girth2_top30 < 0.007857
        + 0.02261258 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +2.3%  sum_pt_top50 < 1157
        + 0.02004281 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +2.0%  psi_0p3 > 0.9956
        - 0.01961431 * max(0.0, Q.mass_top50 - 82.04491) / 17.71057   # -2.0%  mass_top50 > 82.04
        - 0.01918439 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # -1.9%  z_dr_0p2_0p4 < 0.09123
        - 0.01454675 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -1.5%  tau1 < 0.07057
        + 0.01315251 * max(0.0, 80.24626 - Q.mass_top30) / 14.57104   # +1.3%  mass_top30 < 80.25
        - 0.01242063 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -1.2%  lam1 < 0.005914
        - 0.01214585 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -1.2%  sum_pt < 1013
        + 0.01195161 * max(0.0, 0.007538019 - Q.girth2_top20) / 0.002740841   # +1.2%  girth2_top20 < 0.007538
        - 0.008509801 * max(0.0, 0.9906378 - Q.z_top50_slots) / 0.004783315   # -0.9%  z_top50_slots < 0.9906
        - 0.004230793 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.03457336 - Q.M3) / 0.2515325   # -0.4%  sum_pt < 1013 and M3 < 0.03457
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 15.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.41292 * (0.1102416
        - 0.1004395 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -10.0%  log_sum_pt < 7.139
        + 0.1001667 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # +10.0%  log_sum_pt > 6.894
        - 0.08590865 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -8.6%  sum_pt_top50 > 959.1
        + 0.07782397 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +7.8%  n_particles > 38
        + 0.07589961 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.2734   # +7.6%  sum_pt_top2 < 689.2
        + 0.07175627 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +7.2%  log_sum_pt > 6.91
        + 0.05615308 * max(0.0, 0.1751567 - Q.tau1) / 0.09101058   # +5.6%  tau1 < 0.1752
        - 0.04935579 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -4.9%  mass < 101
        - 0.03627558 * max(0.0, 32.50209 - Q.sj3_mass1) / 17.52057   # -3.6%  sj3_mass1 < 32.5
        + 0.03388824 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +3.4%  mass < 89.74
        - 0.03367709 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2) / 190.5117   # -3.4%  sj3_mass1 < 32.5 and sj3_mass2 < 18.68
        - 0.03325546 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -3.3%  log_sum_pt > 6.989
        - 0.03086792 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt) / 15.05788   # -3.1%  n_particles > 38 and soft1_pt < 2.275
        - 0.02248493 * max(0.0, Q.zdr_0 - 0.001901263) / 0.007974889   # -2.2%  zdr_0 > 0.001901
        - 0.01901084 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 0.9624339 - Q.tau43) / 39.08134   # -1.9%  sum_pt_top2 < 689.2 and tau43 < 0.9624
        - 0.01883014 * max(0.0, 47.88842 - Q.mass_top20) / 6.040308   # -1.9%  mass_top20 < 47.89
        + 0.01736207 * max(0.0, 1017.435 - Q.sum_pt) / 21.56661   # +1.7%  sum_pt < 1017
        - 0.01730431 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # -1.7%  M3 < 0.03188
        + 0.01548548 * max(0.0, 80.24626 - Q.mass_top30) / 14.57104   # +1.5%  mass_top30 < 80.25
        + 0.01504958 * max(0.0, 1069.671 - Q.sum_pt_top40) / 70.69518   # +1.5%  sum_pt_top40 < 1070
        + 0.01488706 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2) / 0.0008500795   # +1.5%  z_top30_slots > 0.9342 and C2 < 0.0728
        + 0.01402834 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.01395978 - Q.zdr_2) / 0.09391572   # +1.4%  n_particles > 38 and zdr_2 < 0.01396
        + 0.01212982 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 39.23115   # +1.2%  mass_top20 < 47.89 and n_real_top40 > 29
        + 0.0114955 * max(0.0, Q.mass_top10 - 31.33272) / 22.02168   # +1.1%  mass_top10 > 31.33
        - 0.01120427 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -1.1%  n_dr_0p2_0p4 < 7
        + 0.008493663 * max(0.0, 0.0007894752 - Q.girth2_top15) / 9.062109e-05   # +0.8%  girth2_top15 < 0.0007895
        + 0.007337957 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793) / 0.1941151   # +0.7%  z_top30_slots > 0.9342 and max_pair_mass > 13.05
        - 0.006855763 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # -0.7%  log_sum_pt > 6.959
        + 0.002572362 * max(0.0, Q.sum_pt_top30 - 1191.938) / 6.938323   # +0.3%  sum_pt_top30 > 1192
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 6.361;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.360732 * (0.02748872
        - 0.1660355 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -16.6%  mass < 92.86
        - 0.1115794 * max(0.0, 1260.541 - Q.sum_pt) / 224.4005   # -11.2%  sum_pt < 1261
        + 0.09836189 * max(0.0, 91.03469 - Q.mass) / 16.36287   # +9.8%  mass < 91.03
        + 0.09500684 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15) / 0.0009227558   # +9.5%  log_sum_pt > 6.903 and girth2_top15 < 0.02147
        + 0.07303567 * max(0.0, 143.7876 - Q.mass) / 57.99337   # +7.3%  mass < 143.8
        - 0.07143206 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15) / 1.045047   # -7.1%  sum_pt_top50 > 988.5 and girth2_top15 < 0.02147
        - 0.06639622 * max(0.0, 996.8867 - Q.sum_pt_top30) / 42.94346   # -6.6%  sum_pt_top30 < 996.9
        + 0.06171372 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # +6.2%  sum_pt > 1017
        + 0.06013316 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +6.0%  mass < 121.4
        + 0.04847261 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +4.8%  sum_pt_top40 < 1053
        - 0.03606948 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -3.6%  sum_pt > 1053
        + 0.0337635 * max(0.0, 926.0902 - Q.sum_pt_top20) / 50.299   # +3.4%  sum_pt_top20 < 926.1
        - 0.02770926 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.003463934   # -2.8%  log_sum_pt > 6.903 and psi_0p3 > 0.9299
        - 0.02084217 * max(0.0, 1041.263 - Q.sum_pt_top40) * max(0.0, 26.08605 - Q.sj3_pair_mass_min) / 260.2757   # -2.1%  sum_pt_top40 < 1041 and sj3_pair_mass_min < 26.09
        + 0.0117824 * max(0.0, Q.sum_pt_top50 - 1156.659) / 13.71464   # +1.2%  sum_pt_top50 > 1157
        - 0.01073704 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # -1.1%  sum_pt > 1116
        - 0.004935838 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.mass_top40 - 77.93668) / 1.109603   # -0.5%  log_sum_pt > 6.903 and mass_top40 > 77.94
        - 0.001993207 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.dr_3 - 0.1042479) / 6.574879e-05   # -0.2%  log_sum_pt > 7.063 and dr_3 > 0.1042
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 4.131;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.13126 * (0.04388594
        + 0.1320457 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +13.2%  tau4 < 0.01627
        - 0.1131747 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # -11.3%  girth2 < 0.009615
        + 0.1022592 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # +10.2%  girth2_top50 < 0.008125
        - 0.08132597 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # -8.1%  girth2_top40 < 0.00626
        + 0.07784938 * max(0.0, 87.36377 - Q.mass) / 14.19996   # +7.8%  mass < 87.36
        - 0.06926626 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -6.9%  tau21 < 0.3862
        - 0.06706448 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # -6.7%  n_particles < 46
        + 0.06269561 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +6.3%  n_dr_0p2_0p4 < 5
        + 0.04465943 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) / 2.171439   # +4.5%  n_dr_0p1_0p2 < 10
        - 0.04164303 * max(0.0, 79.21004 - Q.mass_top50) / 10.65754   # -4.2%  mass_top50 < 79.21
        - 0.04107852 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243) / 0.0001457263   # -4.1%  tau21 < 0.3862 and lam1 > 0.007671
        + 0.0409501 * max(0.0, 0.08665515 - Q.mass_over_sum_pt) / 0.01542259   # +4.1%  mass_over_sum_pt < 0.08666
        - 0.03974302 * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2) / 0.01931881   # -4.0%  z_dr_0p1_0p2 < 0.06473
        + 0.03024273 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732) / 1477.074   # +3.0%  n_particles < 46 and sum_pt_top30 > 800.7
        - 0.02851497 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5) / 0.004484684   # -2.9%  n_dr_0p2_0p4 < 5 and zdr_5 < 0.007099
        + 0.02748684 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421) / 0.000406438   # +2.7%  D2 < 2.41 and psi_0p3 > 0.9985
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 8.501;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.500673 * (0.155309
        - 0.1372146 * max(0.0, 79.65241 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.03386155   # -13.7%  mass < 79.65 and lam2 < 0.003688
        + 0.1304341 * max(0.0, 101.0497 - Q.mass) * max(0.0, 0.003687605 - Q.lam2) / 0.07479306   # +13.0%  mass < 101 and lam2 < 0.003688
        + 0.08536346 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +8.5%  mass < 121.4
        - 0.0834496 * max(0.0, Q.n_particles - 22.0) / 23.9717   # -8.3%  n_particles > 22
        - 0.0815091 * max(0.0, 121.737 - Q.mass_top30) / 46.42587   # -8.2%  mass_top30 < 121.7
        - 0.08091034 * max(0.0, 94.64253 - Q.mass_top40) / 21.02021   # -8.1%  mass_top40 < 94.64
        + 0.07757048 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +7.8%  mass < 74.25
        - 0.05301009 * max(0.0, 79.65241 - Q.mass) / 10.37103   # -5.3%  mass < 79.65
        - 0.04709287 * max(0.0, 87.36377 - Q.mass) / 14.19996   # -4.7%  mass < 87.36
        + 0.0460651 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # +4.6%  mass_top40 < 83.33
        - 0.04147567 * max(0.0, 0.004855289 - Q.girth2_top15) / 0.001418414   # -4.1%  girth2_top15 < 0.004855
        + 0.03301141 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt) / 36.47388   # +3.3%  n_particles > 22 and soft1_pt < 2.275
        + 0.02867271 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +2.9%  mass_top40 < 67.73
        + 0.02260877 * max(0.0, 57.87349 - Q.mass_top15) / 12.46085   # +2.3%  mass_top15 < 57.87
        - 0.01749402 * max(0.0, 83.32554 - Q.mass_top40) * max(0.0, 6.916121 - Q.D2) / 32.02927   # -1.7%  mass_top40 < 83.33 and D2 < 6.916
        + 0.01405633 * max(0.0, Q.sj3_pair_mass_min - 32.51366) / 5.88267   # +1.4%  sj3_pair_mass_min > 32.51
        - 0.01009896 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.0329485 - Q.C2_b2) / 0.0003026554   # -1.0%  sj2_dr > 0.2232 and C2_b2 < 0.03295
        + 0.00996237 * max(0.0, 0.3036026 - Q.planar_flow) / 0.04665298   # +1.0%  planar_flow < 0.3036
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 12.58;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.58086 * (0.1261739
        + 0.188 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +18.8%  sum_pt_top50 > 934.2
        + 0.09408485 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +9.4%  sum_pt > 907.9
        - 0.0841905 * max(0.0, 150.0144 - Q.mass_top40) / 66.86198   # -8.4%  mass_top40 < 150
        - 0.07004095 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -7.0%  log_sum_pt > 6.91
        - 0.05907162 * max(0.0, Q.z_top30_slots - 0.9048492) / 0.05621641   # -5.9%  z_top30_slots > 0.9048
        - 0.05222295 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -5.2%  log_sum_pt > 6.92
        + 0.0513348 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +5.1%  n_particles < 64
        - 0.03826382 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -3.8%  sum_pt_top40 > 1025
        - 0.03794619 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -3.8%  max_dr > 0.2405
        - 0.03686063 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -3.7%  mass_top50 > 157.5
        - 0.03406336 * max(0.0, Q.sum_pt - 986.0565) / 69.72366   # -3.4%  sum_pt > 986.1
        + 0.03036661 * max(0.0, Q.sum_pt_top30 - 933.1875) / 79.88556   # +3.0%  sum_pt_top30 > 933.2
        + 0.02842064 * max(0.0, Q.sd_mass - 69.65633) / 11.73272   # +2.8%  sd_mass > 69.66
        - 0.02801346 * max(0.0, Q.girth2_top30 - 0.006363916) / 0.003802703   # -2.8%  girth2_top30 > 0.006364
        - 0.02313013 * max(0.0, 0.5100475 - Q.tau21) / 0.1343519   # -2.3%  tau21 < 0.51
        + 0.0209574 * max(0.0, Q.mass - 101.0497) / 12.31084   # +2.1%  mass > 101
        - 0.02053648 * max(0.0, Q.sd_mass - 83.30647) / 6.989667   # -2.1%  sd_mass > 83.31
        - 0.01567174 * max(0.0, Q.mass_top40 - 111.2487) / 7.565029   # -1.6%  mass_top40 > 111.2
        - 0.01338705 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2) / 10.81389   # -1.3%  n_particles < 64 and D2 < 2.179
        + 0.01172964 * max(0.0, 0.01375115 - Q.z_11) / 0.0009450582   # +1.2%  z_11 < 0.01375
        - 0.0116607 * max(0.0, Q.mass_over_sum_pt - 0.09046749) / 0.01421039   # -1.2%  mass_over_sum_pt > 0.09047
        - 0.01090599 * max(0.0, 14.14062 - Q.pt_11) / 0.9740783   # -1.1%  pt_11 < 14.14
        + 0.009976978 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # +1.0%  mass_over_sum_pt > 0.1709
        + 0.009334262 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # +0.9%  log_sum_pt > 6.989
        + 0.008699368 * max(0.0, 64.48544 - Q.mass) / 5.935866   # +0.9%  mass < 64.49
        - 0.007822057 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # -0.8%  mass_over_sum_pt_sq > 0.0292
        + 0.003307808 * max(0.0, Q.max_dr - 0.4357228) / 0.007711928   # +0.3%  max_dr > 0.4357
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 15.98;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.97843 * (0.03570454
        + 0.2134798 * max(0.0, 0.02580859 - Q.e2_sq) / 0.01698103   # +21.3%  e2_sq < 0.02581
        - 0.1248402 * max(0.0, 101.0497 - Q.mass) / 23.61933   # -12.5%  mass < 101
        - 0.09188127 * max(0.0, 121.3913 - Q.mass) / 39.46288   # -9.2%  mass < 121.4
        - 0.08261785 * max(0.0, 0.02550569 - Q.girth2_top50) / 0.01683664   # -8.3%  girth2_top50 < 0.02551
        - 0.08211123 * max(0.0, 0.02412652 - Q.girth2_top30) / 0.01613825   # -8.2%  girth2_top30 < 0.02413
        - 0.06874823 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # -6.9%  mass_over_sum_pt < 0.09795
        + 0.05583999 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +5.6%  mass < 92.86
        + 0.04487144 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +4.5%  e3 < 0.0003372
        + 0.04034985 * max(0.0, 172.4888 - Q.mass) / 83.49222   # +4.0%  mass < 172.5
        - 0.03804327 * max(0.0, 0.01649354 - Q.lam1) / 0.009604223   # -3.8%  lam1 < 0.01649
        + 0.03679958 * max(0.0, 89.74183 - Q.mass) / 15.55633   # +3.7%  mass < 89.74
        + 0.03398 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +3.4%  lam1 < 0.007259
        + 0.02677435 * max(0.0, 71.79516 - Q.mass_top50) / 8.207808   # +2.7%  mass_top50 < 71.8
        + 0.02230964 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +2.2%  mass < 82.85
        + 0.02076776 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # +2.1%  girth2_top30 < 0.008376
        - 0.01350014 * max(0.0, 2.178951 - Q.D2) / 0.4648731   # -1.4%  D2 < 2.179
        + 0.003085356 * max(0.0, 121.3913 - Q.mass) * max(0.0, 1003.544 - Q.sum_pt_top50) / 688.1559   # +0.3%  mass < 121.4 and sum_pt_top50 < 1004
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 38.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 38.22912 * (0.005570529
        - 0.09169058 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.5305504   # -9.2%  mass < 91.03 and psi_0p3 > 0.9638
        - 0.08979102 * max(0.0, 92.85979 - Q.mass) / 17.60131   # -9.0%  mass < 92.86
        + 0.07369077 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.02554674   # +7.4%  mass < 101 and psi_0p3 > 0.9974
        + 0.07181898 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082) / 0.7612129   # +7.2%  mass < 101 and psi_0p3 > 0.9638
        + 0.04876091 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # +4.9%  LHA < 0.3098
        + 0.04714811 * max(0.0, 91.03469 - Q.mass) / 16.36287   # +4.7%  mass < 91.03
        - 0.04674672 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.01682193   # -4.7%  mass < 91.03 and psi_0p3 > 0.9974
        - 0.04414645 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -4.4%  tau1 < 0.1073
        + 0.04063891 * max(0.0, 78.26182 - Q.mass) / 9.857173   # +4.1%  mass < 78.26
        + 0.04060452 * max(0.0, 0.009595015 - Q.mass_over_sum_pt_sq) / 0.003489309   # +4.1%  mass_over_sum_pt_sq < 0.009595
        - 0.03180917 * max(0.0, 73.35236 - Q.mass_top20) / 16.12893   # -3.2%  mass_top20 < 73.35
        - 0.03060435 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9973959) / 0.01175018   # -3.1%  mass < 82.85 and psi_0p3 > 0.9974
        - 0.02985531 * max(0.0, 91.03469 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9299135) / 1.084471   # -3.0%  mass < 91.03 and psi_0p3 > 0.9299
        - 0.02980287 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -3.0%  lam1 < 0.008242
        + 0.02588726 * max(0.0, 70.42121 - Q.mass_top20) / 14.59536   # +2.6%  mass_top20 < 70.42
        - 0.02482329 * max(0.0, 0.008190222 - Q.width) / 0.002454223   # -2.5%  width < 0.00819
        + 0.02351503 * max(0.0, 0.08786745 - Q.tau1) / 0.02149491   # +2.4%  tau1 < 0.08787
        + 0.02099915 * max(0.0, 121.3913 - Q.mass) / 39.46288   # +2.1%  mass < 121.4
        - 0.02052867 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # -2.1%  LHA < 0.3332
        + 0.0196492 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # +2.0%  lam1 < 0.00619
        - 0.01949642 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -1.9%  psi_0p3 > 0.998
        - 0.01574578 * max(0.0, 0.003418057 - Q.girth2_top50) / 0.0004594629   # -1.6%  girth2_top50 < 0.003418
        + 0.0136962 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +1.4%  mass_over_sum_pt < 0.1182
        - 0.01306844 * max(0.0, 0.2845608 - Q.LHA) / 0.05172777   # -1.3%  LHA < 0.2846
        + 0.01273541 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +1.3%  tau21_b2 < 0.2352
        + 0.01088685 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p3 - 0.9973959) / 4.396153e-05   # +1.1%  mass_over_sum_pt < 0.1182 and psi_0p3 > 0.9974
        - 0.009894576 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # -1.0%  girth2 < 0.007877
        - 0.00868963 * max(0.0, 0.3861957 - Q.tau21) / 0.0690327   # -0.9%  tau21 < 0.3862
        + 0.008544652 * max(0.0, 101.0497 - Q.mass) / 23.61933   # +0.9%  mass < 101
        - 0.008494917 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # -0.8%  LHA < 0.2601
        - 0.00824151 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9995915 - Q.psi_0p3) / 0.0378912   # -0.8%  mass < 82.85 and psi_0p3 < 0.9996
        - 0.007221379 * max(0.0, Q.z_top20_slots - 0.9102775) / 0.02689878   # -0.7%  z_top20_slots > 0.9103
        - 0.004084748 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt) / 11.66143   # -0.4%  tau21_b2 < 0.2352 and sum_pt < 1261
        + 0.00358806 * max(0.0, 91.03469 - Q.mass) * max(0.0, 0.9973959 - Q.psi_0p3) / 0.03659629   # +0.4%  mass < 91.03 and psi_0p3 < 0.9974
        + 0.003100179 * max(0.0, 121.3913 - Q.mass) * max(0.0, Q.sd_mass - 76.29481) / 33.79048   # +0.3%  mass < 121.4 and sd_mass > 76.29
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 19.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.40849 * (-0.05788782
        - 0.08536541 * max(0.0, Q.e2_sq - 0.009606007) / 0.003182984   # -8.5%  e2_sq > 0.009606
        - 0.07532586 * max(0.0, Q.mass - 101.0497) / 12.31084   # -7.5%  mass > 101
        + 0.06511453 * max(0.0, Q.mass_over_sum_pt - 0.07696632) / 0.02024106   # +6.5%  mass_over_sum_pt > 0.07697
        + 0.06445605 * max(0.0, Q.e2_sq - 0.007872294) / 0.00365949   # +6.4%  e2_sq > 0.007872
        - 0.05769313 * max(0.0, Q.mass_top50 - 80.35535) / 18.59691   # -5.8%  mass_top50 > 80.36
        + 0.05592817 * max(0.0, Q.mass - 80.78464) / 19.8061   # +5.6%  mass > 80.78
        + 0.05538926 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +5.5%  e3 < 0.0003372
        - 0.04823092 * max(0.0, 0.02737453 - Q.girth2_top20) / 0.01974268   # -4.8%  girth2_top20 < 0.02737
        + 0.04740241 * max(0.0, Q.mass - 64.48544) / 31.19164   # +4.7%  mass > 64.49
        + 0.04043682 * max(0.0, 0.007887677 - Q.girth2_top15) / 0.00326329   # +4.0%  girth2_top15 < 0.007888
        + 0.03920272 * max(0.0, Q.mass_over_sum_pt_sq - 0.001785266) / 0.00762696   # +3.9%  mass_over_sum_pt_sq > 0.001785
        - 0.03610722 * max(0.0, Q.sum_pt_top40 - 858.8262) / 166.7848   # -3.6%  sum_pt_top40 > 858.8
        - 0.03559032 * max(0.0, Q.girth2_top40 - 0.006026828) / 0.00421794   # -3.6%  girth2_top40 > 0.006027
        + 0.03550324 * max(0.0, 0.1291856 - Q.z_dr_0p2_0p4) / 0.08975043   # +3.6%  z_dr_0p2_0p4 < 0.1292
        - 0.03515331 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -3.5%  n_dr_0p2_0p4 < 15
        + 0.02755143 * max(0.0, Q.girth2_top40 - 0.008031986) / 0.003375597   # +2.8%  girth2_top40 > 0.008032
        - 0.02412535 * max(0.0, Q.tau1 - 0.06310829) / 0.03403397   # -2.4%  tau1 > 0.06311
        + 0.02280153 * max(0.0, Q.mass_top50 - 97.93004) / 11.91764   # +2.3%  mass_top50 > 97.93
        + 0.01825818 * max(0.0, Q.mass - 89.74183) / 15.55572   # +1.8%  mass > 89.74
        - 0.0155741 * max(0.0, Q.mass - 121.3913) / 7.81281   # -1.6%  mass > 121.4
        + 0.01449813 * max(0.0, Q.girth2_top15 - 0.001319197) / 0.006270373   # +1.4%  girth2_top15 > 0.001319
        + 0.01411845 * max(0.0, 6.935549 - Q.log_sum_pt) / 0.02809622   # +1.4%  log_sum_pt < 6.936
        + 0.01372039 * max(0.0, Q.mass_top40 - 111.2487) / 7.565029   # +1.4%  mass_top40 > 111.2
        + 0.01198568 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +1.2%  LHA > 0.3332
        - 0.01083574 * max(0.0, 0.005383629 - Q.girth2_top15) / 0.001666876   # -1.1%  girth2_top15 < 0.005384
        + 0.01062262 * max(0.0, Q.girth2_top40 - 0.005196966) / 0.004725809   # +1.1%  girth2_top40 > 0.005197
        - 0.0101866 * max(0.0, 1003.544 - Q.sum_pt_top50) / 20.02299   # -1.0%  sum_pt_top50 < 1004
        + 0.008425321 * max(0.0, Q.n_dr_0p2_0p4 - 8.0) / 3.639534   # +0.8%  n_dr_0p2_0p4 > 8
        + 0.005826996 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # +0.6%  log_sum_pt > 6.959
        - 0.005628943 * max(0.0, Q.mass - 143.7876) / 3.946979   # -0.6%  mass > 143.8
        + 0.004659876 * max(0.0, Q.log_sum_pt - 6.959294) * max(0.0, Q.sj3_pair_mass_max - 28.35435) / 1.470253   # +0.5%  log_sum_pt > 6.959 and sj3_pair_mass_max > 28.35
        + 0.004281312 * max(0.0, 1001.523 - Q.sum_pt_top40) / 26.72886   # +0.4%  sum_pt_top40 < 1002
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 6.702;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.701755 * (0.1224514
        - 0.1333581 * max(0.0, 152.6883 - Q.mass_top30) / 74.18247   # -13.3%  mass_top30 < 152.7
        + 0.1056455 * max(0.0, 87.36377 - Q.mass) / 14.19996   # +10.6%  mass < 87.36
        + 0.09704107 * max(0.0, 79.65241 - Q.mass) / 10.37103   # +9.7%  mass < 79.65
        + 0.08985087 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # +9.0%  girth2_top40 < 0.00626
        - 0.08366264 * max(0.0, Q.z_top40_slots - 0.9574183) / 0.02831517   # -8.4%  z_top40_slots > 0.9574
        - 0.075721 * max(0.0, 64.48544 - Q.mass) / 5.935866   # -7.6%  mass < 64.49
        - 0.07282571 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -7.3%  girth2_top20 < 0.01083
        + 0.06164907 * max(0.0, 0.2091025 - Q.LHA) / 0.02095782   # +6.2%  LHA < 0.2091
        - 0.05975689 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -6.0%  tau1 < 0.05445
        + 0.05379811 * max(0.0, 121.737 - Q.mass_top30) / 46.42587   # +5.4%  mass_top30 < 121.7
        - 0.03321294 * max(0.0, 89.6788 - Q.mass_top40) / 17.47899   # -3.3%  mass_top40 < 89.68
        + 0.03294652 * max(0.0, 80.89043 - Q.mass_top40) / 12.43405   # +3.3%  mass_top40 < 80.89
        + 0.02133133 * max(0.0, Q.mass - 121.3913) / 7.81281   # +2.1%  mass > 121.4
        + 0.01753155 * max(0.0, 959.0957 - Q.sum_pt_top50) / 9.937941   # +1.8%  sum_pt_top50 < 959.1
        - 0.01580758 * max(0.0, Q.e3 - 0.0003372339) / 2.169276e-05   # -1.6%  e3 > 0.0003372
        - 0.01340568 * max(0.0, Q.mass - 143.7876) / 3.946979   # -1.3%  mass > 143.8
        - 0.01131744 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt) / 447.38   # -1.1%  mass_top40 < 80.89 and sum_pt < 1035
        - 0.009481112 * max(0.0, Q.mass - 172.4888) / 0.7446233   # -0.9%  mass > 172.5
        + 0.007290665 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.894531) / 2.293712   # +0.7%  sum_pt_top40 < 956.2 and soft5_pt > 2.895
        - 0.004366282 * max(0.0, 959.0957 - Q.sum_pt_top50) * max(0.0, Q.soft5_pt - 2.894531) / 1.196279   # -0.4%  sum_pt_top50 < 959.1 and soft5_pt > 2.895
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 19.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.28701 * (-0.2036105
        + 0.1756355 * Q.sum_pt / 1043.796   # +17.6%  sum_pt
        - 0.1157719 * max(0.0, 80.3008 - Q.sj2_mass1) / 50.41572   # -11.6%  sj2_mass1 < 80.3
        + 0.1040459 * max(0.0, 132.4278 - Q.mass_top40) / 51.50818   # +10.4%  mass_top40 < 132.4
        - 0.101688 * max(0.0, Q.mass - 64.48544) / 31.19164   # -10.2%  mass > 64.49
        + 0.07939678 * max(0.0, Q.mass - 82.85409) / 18.72167   # +7.9%  mass > 82.85
        - 0.05949886 * max(0.0, 83.32554 - Q.mass_top40) / 13.71649   # -5.9%  mass_top40 < 83.33
        + 0.04816066 * max(0.0, Q.mass_top40 - 67.72643) / 24.79585   # +4.8%  mass_top40 > 67.73
        + 0.04052361 * max(0.0, 839.9547 - Q.sum_pt_top5) / 255.1112   # +4.1%  sum_pt_top5 < 840
        + 0.03751958 * max(0.0, 3.814159 - Q.D2) / 1.519965   # +3.8%  D2 < 3.814
        + 0.03235462 * max(0.0, 87.27603 - Q.mass_top40) / 15.9837   # +3.2%  mass_top40 < 87.28
        + 0.0289512 * max(0.0, 0.6133424 - Q.tau21_b2) / 0.3099701   # +2.9%  tau21_b2 < 0.6133
        + 0.02724641 * max(0.0, 0.01807679 - Q.girth2_top30) / 0.01083719   # +2.7%  girth2_top30 < 0.01808
        - 0.02281095 * max(0.0, Q.z_dr_0_0p05 - 0.7674734) / 0.05221088   # -2.3%  z_dr_0_0p05 > 0.7675
        - 0.01843185 * max(0.0, Q.log_sum_pt - 6.903423) / 0.05662275   # -1.8%  log_sum_pt > 6.903
        + 0.01710233 * max(0.0, 111.2487 - Q.mass_top40) / 33.99436   # +1.7%  mass_top40 < 111.2
        - 0.01519072 * max(0.0, Q.mass - 143.7876) / 3.946979   # -1.5%  mass > 143.8
        + 0.01346334 * max(0.0, 74.25181 - Q.mass) / 8.587064   # +1.3%  mass < 74.25
        - 0.01315634 * max(0.0, 75.26407 - Q.mass_top15) / 22.29039   # -1.3%  mass_top15 < 75.26
        - 0.01035421 * max(0.0, Q.mass_over_sum_pt - 0.1708801) / 0.0005536194   # -1.0%  mass_over_sum_pt > 0.1709
        - 0.009392268 * max(0.0, Q.mass - 172.4888) / 0.7446233   # -0.9%  mass > 172.5
        + 0.008808219 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002) / 0.0002028274   # +0.9%  mass_over_sum_pt_sq > 0.0292
        + 0.006372317 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # +0.6%  mass_top50 > 157.5
        + 0.006285532 * max(0.0, 80.3008 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.91979) / 145.9595   # +0.6%  sj2_mass1 < 80.3 and sj2_mass2 > 11.92
        - 0.005644336 * max(0.0, 3.814159 - Q.D2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.04343614   # -0.6%  D2 < 3.814 and sj2_dr > 0.1938
        + 0.002194594 * max(0.0, Q.mass - 172.4888) * max(0.0, Q.M2 - 0.04260132) / 0.01594348   # +0.2%  mass > 172.5 and M2 > 0.0426
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 7.653;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.652868 * (-0.03983198
        + 0.1772643 * max(0.0, 92.85979 - Q.mass) / 17.60131   # +17.7%  mass < 92.86
        + 0.1039578 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +10.4%  e2_sq < 0.008184
        - 0.102324 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # -10.2%  girth < 0.07374
        - 0.08759922 * max(0.0, 79.65241 - Q.mass) / 10.37103   # -8.8%  mass < 79.65
        - 0.07329344 * max(0.0, 85.8667 - Q.mass_top50) / 13.95835   # -7.3%  mass_top50 < 85.87
        - 0.06308199 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -6.3%  lam1 < 0.006717
        + 0.04493617 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +4.5%  n_dr_0p2_0p4 < 10
        + 0.04474997 * max(0.0, 0.02793599 - Q.e2) / 0.005715277   # +4.5%  e2 < 0.02794
        + 0.04453224 * max(0.0, 15.0 - Q.n_dr_0p1_0p2) / 5.139044   # +4.5%  n_dr_0p1_0p2 < 15
        - 0.04134877 * max(0.0, 79.65241 - Q.mass) * max(0.0, 3.814159 - Q.D2) / 4.810976   # -4.1%  mass < 79.65 and D2 < 3.814
        + 0.03955269 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # +4.0%  tau1 < 0.1073
        - 0.03844255 * max(0.0, 0.05048381 - Q.girth) / 0.008533025   # -3.8%  girth < 0.05048
        - 0.03103428 * max(0.0, 0.006929741 - Q.girth2_top30) / 0.002004687   # -3.1%  girth2_top30 < 0.00693
        + 0.02212342 * max(0.0, 0.2845608 - Q.LHA) / 0.05172777   # +2.2%  LHA < 0.2846
        + 0.01985739 * max(0.0, Q.psi_0p2 - 0.9935324) / 0.001465572   # +2.0%  psi_0p2 > 0.9935
        - 0.01860177 * max(0.0, Q.z_top10_slots - 0.7271951) / 0.0653343   # -1.9%  z_top10_slots > 0.7272
        + 0.01696601 * max(0.0, 0.003213724 - Q.girth2_top10) / 0.0009423838   # +1.7%  girth2_top10 < 0.003214
        + 0.01689432 * max(0.0, 0.1632346 - Q.LHA) / 0.009624216   # +1.7%  LHA < 0.1632
        + 0.01343963 * max(0.0, 64.48544 - Q.mass) / 5.935866   # +1.3%  mass < 64.49
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 3.055;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.054533 * (-0.1167052
        + 0.2619535 * max(0.0, 80.78464 - Q.mass) / 10.84952   # +26.2%  mass < 80.78
        + 0.1966753 * max(0.0, 82.85409 - Q.mass) / 11.83454   # +19.7%  mass < 82.85
        - 0.1166937 * max(0.0, 77.93668 - Q.mass_top40) / 11.12028   # -11.7%  mass_top40 < 77.94
        + 0.09833222 * max(0.0, 87.36377 - Q.mass) / 14.19996   # +9.8%  mass < 87.36
        - 0.09593259 * max(0.0, 0.00363788 - Q.e2_sq) / 0.0004963098   # -9.6%  e2_sq < 0.003638
        - 0.07731469 * max(0.0, 87.36377 - Q.mass) * max(0.0, 0.9989733 - Q.psi_0p3) / 0.0414109   # -7.7%  mass < 87.36 and psi_0p3 < 0.999
        - 0.04861207 * max(0.0, 53.87362 - Q.mass) / 3.56547   # -4.9%  mass < 53.87
        - 0.04106615 * max(0.0, 0.06895248 - Q.mass_over_sum_pt) / 0.007729512   # -4.1%  mass_over_sum_pt < 0.06895
        - 0.01792271 * max(0.0, 972.0419 - Q.sum_pt) / 9.586856   # -1.8%  sum_pt < 972
        + 0.01694752 * max(0.0, Q.sj3_pair_mass_max - 122.7494) * max(0.0, 65.20727 - Q.sj2_mass1) / 34.79473   # +1.7%  sj3_pair_mass_max > 122.7 and sj2_mass1 < 65.21
        - 0.01187292 * max(0.0, Q.sd_mass - 133.2575) / 0.7219495   # -1.2%  sd_mass > 133.3
        + 0.01077962 * max(0.0, Q.sd_mass - 133.2575) * max(0.0, 32.50209 - Q.sj3_mass1) / 9.827377   # +1.1%  sd_mass > 133.3 and sj3_mass1 < 32.5
        - 0.005897073 * max(0.0, Q.sj3_pair_mass_max - 122.7494) / 1.904096   # -0.6%  sj3_pair_mass_max > 122.7
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 5.385;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.385161 * (0.3272377
        + 0.1346477 * max(0.0, Q.mass - 64.48544) / 31.19164   # +13.5%  mass > 64.49
        - 0.1254928 * max(0.0, 1012.673 - Q.sum_pt) / 19.54408   # -12.5%  sum_pt < 1013
        - 0.1192511 * max(0.0, Q.mass - 143.7876) / 3.946979   # -11.9%  mass > 143.8
        - 0.08715145 * max(0.0, Q.mass_top50 - 97.93004) / 11.91764   # -8.7%  mass_top50 > 97.93
        - 0.08101995 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -8.1%  sum_pt < 1085
        + 0.07504671 * max(0.0, Q.mass_top50 - 138.8977) / 3.930268   # +7.5%  mass_top50 > 138.9
        - 0.05524198 * max(0.0, Q.mass_over_sum_pt - 0.07435617) / 0.02188714   # -5.5%  mass_over_sum_pt > 0.07436
        + 0.04792745 * max(0.0, Q.mass - 78.26182) / 21.33658   # +4.8%  mass > 78.26
        + 0.04280355 * max(0.0, 1008.935 - Q.sum_pt_top50) / 22.0436   # +4.3%  sum_pt_top50 < 1009
        + 0.03862398 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +3.9%  sum_pt_top40 < 1053
        - 0.03711658 * max(0.0, Q.mass_over_sum_pt - 0.06030419) / 0.03186977   # -3.7%  mass_over_sum_pt > 0.0603
        - 0.0348044 * max(0.0, Q.mass_top50 - 157.5448) / 1.557119   # -3.5%  mass_top50 > 157.5
        + 0.0332499 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.5260785 - Q.D3) / 23.2296   # +3.3%  sum_pt_top40 < 1053 and D3 < 0.5261
        + 0.03116469 * max(0.0, Q.mass_top40 - 150.0144) / 1.666872   # +3.1%  mass_top40 > 150
        + 0.0189969 * max(0.0, Q.mass_top50 - 97.93004) * max(0.0, 2.873047 - Q.soft3_pt) / 19.23325   # +1.9%  mass_top50 > 97.93 and soft3_pt < 2.873
        + 0.01654275 * max(0.0, Q.mass - 172.4888) / 0.7446233   # +1.7%  mass > 172.5
        + 0.01308444 * max(0.0, 6.879399 - Q.log_sum_pt) / 0.01083309   # +1.3%  log_sum_pt < 6.879
        + 0.007833698 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # +0.8%  log_sum_pt < 6.811
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 14.56;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.56316 * (-0.00628981
        - 0.1387694 * max(0.0, 82.85409 - Q.mass) / 11.83454   # -13.9%  mass < 82.85
        - 0.1295191 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) / 0.0178873   # -13.0%  mass_over_sum_pt < 0.09047
        + 0.1011559 * max(0.0, 0.01986381 - Q.mass_over_sum_pt_sq) / 0.0117775   # +10.1%  mass_over_sum_pt_sq < 0.01986
        - 0.0753595 * max(0.0, 91.03469 - Q.mass) / 16.36287   # -7.5%  mass < 91.03
        + 0.07383209 * max(0.0, 0.09795415 - Q.mass_over_sum_pt) / 0.02339253   # +7.4%  mass_over_sum_pt < 0.09795
        + 0.07214086 * max(0.0, 0.1182259 - Q.mass_over_sum_pt) / 0.03917122   # +7.2%  mass_over_sum_pt < 0.1182
        - 0.06288161 * max(0.0, 0.01292642 - Q.girth2_top40) / 0.006292908   # -6.3%  girth2_top40 < 0.01293
        + 0.04793681 * max(0.0, 0.08873143 - Q.mass_over_sum_pt) / 0.01671255   # +4.8%  mass_over_sum_pt < 0.08873
        - 0.04143459 * max(0.0, 0.01897915 - Q.girth2_top40) / 0.01130444   # -4.1%  girth2_top40 < 0.01898
        + 0.03906496 * max(0.0, 67.72643 - Q.mass_top40) / 7.702939   # +3.9%  mass_top40 < 67.73
        - 0.02915563 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -2.9%  girth2_top20 < 0.01083
        - 0.0271211 * max(0.0, 79.65241 - Q.mass) / 10.37103   # -2.7%  mass < 79.65
        + 0.0253838 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # +2.5%  tau1 < 0.07057
        + 0.02366239 * max(0.0, 0.03680582 - Q.e2) / 0.01052402   # +2.4%  e2 < 0.03681
        + 0.01832276 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 1.059188 - Q.N3) / 0.002548597   # +1.8%  psi_0p3 > 0.9924 and N3 < 1.059
        - 0.01827204 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.007709916 - Q.girth2_top40) / 0.0001062984   # -1.8%  tau21_b2 < 0.3425 and girth2_top40 < 0.00771
        - 0.01566161 * max(0.0, 0.04142826 - Q.tau2) / 0.01568306   # -1.6%  tau2 < 0.04143
        + 0.01400685 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +1.4%  tau21_b2 < 0.3425
        - 0.01361236 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2738063) / 0.009655725   # -1.4%  N2 < 0.4227 and max_dr > 0.2738
        - 0.01239079 * max(0.0, 0.01256572 - Q.e2) / 0.0008030352   # -1.2%  e2 < 0.01257
        + 0.01228484 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.01691783 - Q.C2_b2) / 0.001347445   # +1.2%  tau21_b2 < 0.3425 and C2_b2 < 0.01692
        - 0.008031053 * max(0.0, Q.psi_0p1 - 0.9538343) / 0.006310723   # -0.8%  psi_0p1 > 0.9538
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 4.146;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.145766 * (-0.2307262
        + 0.1752799 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +17.5%  girth2_top5 < 0.00833
        + 0.1600374 * max(0.0, 79.18312 - Q.sd_mass) / 30.46368   # +16.0%  sd_mass < 79.18
        + 0.127372 * max(0.0, 7.017258 - Q.log_sum_pt) / 0.08901073   # +12.7%  log_sum_pt < 7.017
        - 0.09364205 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -9.4%  tau1 < 0.07057
        + 0.08647659 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # +8.6%  girth2_top10 < 0.007679
        - 0.06028062 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.n_real_top40 - 22.0) / 0.06106188   # -6.0%  girth2_top5 < 0.00833 and n_real_top40 > 22
        + 0.05715591 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +5.7%  z_dr_0p1_0p2 < 0.1203
        - 0.05219932 * max(0.0, 40.97891 - Q.sd_mass) / 11.60739   # -5.2%  sd_mass < 40.98
        - 0.04275092 * max(0.0, 1018.832 - Q.sum_pt_top30) / 55.61816   # -4.3%  sum_pt_top30 < 1019
        - 0.03937668 * max(0.0, 0.002270363 - Q.girth2_top5) / 0.0007634099   # -3.9%  girth2_top5 < 0.00227
        - 0.03740655 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -3.7%  log_sum_pt < 6.903
        + 0.02925288 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2) / 0.0004272674   # +2.9%  sj2_dr > 0.2232 and C2_b2 < 0.04009
        - 0.02006221 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 58.0 - Q.n_particles) / 0.8649544   # -2.0%  z_dr_0p1_0p2 < 0.1203 and n_particles < 58
        + 0.01870706 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +1.9%  sum_pt < 986.1
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6497054621848739, 2.790620745798319, 0.22424789915966387, 0.5601856092436974, 0.7852078781512605, 1.0814517857142858, 0.6732211134453782, 0.5971613445378151, 1.3617996848739495, 1.0102555672268907, 1.2744345588235295, 0.6649610294117647, 0.4494199579831933, 1.4250818802521008, 0.35590887605042015, 0.4668238445378151]
T = [3.983193658088235, 2.463011150866597, 4.609661756499475, 4.839097055540966, 4.080169832425158]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +44%, n4 -17%, n9 +13%, n3 -11%, n5 -8%, n12 +3% ...
            + 0.4378743 * h[1] / H_AVG[1]
            - 0.172489 * h[4] / H_AVG[4]
            + 0.1307777 * h[9] / H_AVG[9]
            - 0.105478 * h[3] / H_AVG[3]
            - 0.0848449 * h[5] / H_AVG[5]
            + 0.02644431 * h[12] / H_AVG[12]
            + 0.02640866 * h[6] / H_AVG[6]
            + 0.01068395 * h[8] / H_AVG[8]
            - 0.004999265 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -34%, n9 +23%, n1 -21%, n11 -7%, n12 +6%, n6 +6% ...
            - 0.338725 * h[4] / H_AVG[4]
            + 0.2307211 * h[9] / H_AVG[9]
            - 0.2124397 * h[1] / H_AVG[1]
            - 0.06749472 * h[11] / H_AVG[11]
            + 0.06272327 * h[12] / H_AVG[12]
            + 0.05979149 * h[6] / H_AVG[6]
            + 0.01138078 * h[2] / H_AVG[2]
            + 0.008639068 * h[8] / H_AVG[8]
            - 0.008084835 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -26%, n5 +14%, n14 -11%, n0 +11%, n11 +9%, n7 -8% ...
            - 0.258495 * h[8] / H_AVG[8]
            + 0.1356313 * h[5] / H_AVG[5]
            - 0.1061628 * h[14] / H_AVG[14]
            + 0.1057082 * h[0] / H_AVG[0]
            + 0.08565067 * h[11] / H_AVG[11]
            - 0.08096599 * h[7] / H_AVG[7]
            + 0.05855423 * h[4] / H_AVG[4]
            + 0.05316685 * h[3] / H_AVG[3]
            - 0.04794135 * h[9] / H_AVG[9]
            - 0.03960743 * h[12] / H_AVG[12]
            - 0.01898826 * h[15] / H_AVG[15]
            + 0.009127854 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -18%, n5 +15%, n6 -13%, n7 +11%, n15 +5% ...
            - 0.2638276 * h[8] / H_AVG[8]
            - 0.1846099 * h[0] / H_AVG[0]
            + 0.153644 * h[5] / H_AVG[5]
            - 0.1347737 * h[6] / H_AVG[6]
            + 0.1118344 * h[7] / H_AVG[7]
            + 0.05426393 * h[15] / H_AVG[15]
            + 0.05064606 * h[3] / H_AVG[3]
            + 0.02902272 * h[12] / H_AVG[12]
            - 0.01737782 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -32%, n10 +31%, n5 -12%, n8 +7%, n15 -4%, n12 -4% ...
            - 0.3165262 * h[13] / H_AVG[13]
            + 0.307468 * h[10] / H_AVG[10]
            - 0.1242425 * h[5] / H_AVG[5]
            + 0.07040261 * h[8] / H_AVG[8]
            - 0.04290482 * h[15] / H_AVG[15]
            - 0.04130526 * h[12] / H_AVG[12]
            - 0.0411629 * h[7] / H_AVG[7]
            + 0.03608342 * h[4] / H_AVG[4]
            + 0.01990436 * h[0] / H_AVG[0]
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
