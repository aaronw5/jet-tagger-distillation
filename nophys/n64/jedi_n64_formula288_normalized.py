"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network (step 4; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  1:  12.8%   (on for 96% of jets)
  neuron  8:  12.2%   (on for 55% of jets)
  neuron  5:  12.0%   (on for 79% of jets)
  neuron  4:   9.7%   (on for 66% of jets)
  neuron  9:   7.7%   (on for 75% of jets)
  neuron 13:   7.6%   (on for 87% of jets)
  neuron  0:   7.3%   (on for 64% of jets)
  neuron 10:   7.0%   (on for 77% of jets)
  neuron 12:   4.7%   (on for 45% of jets)
  neuron  6:   4.0%   (on for 61% of jets)
  neuron 11:   3.5%   (on for 73% of jets)
  neuron  7:   3.4%   (on for 29% of jets)
  neuron  3:   3.2%   (on for 57% of jets)
  neuron 15:   2.4%   (on for 57% of jets)
  neuron 14:   1.9%   (on for 36% of jets)
  neuron  2:   0.6%   (on for 55% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.1% (the network: 81.1%); same class as the network for 93.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_pt               pT [GeV] of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.z_1                    pT of particle 1 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft3_z                pT share of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
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
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top10=mass_of(10),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_particles=len(real),
        n_for_50pct=ncum(0.5),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_0=pt[0],
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        soft1_pt=softp(1, 'pt'),
        soft2_pt=softp(2, 'pt'),
        soft3_pt=softp(3, 'pt'),
        z_1=z[1],
        z_7=z[7],
        z_9=z[9],
        soft3_z=softp(3, 'z'),
        soft4_z=softp(4, 'z'),
        z_top10_slots=sum(pt[:10]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_2=z[2] * dr[2],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_mass=softdrop("mass"),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        eta_0=eta[0],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
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
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
    )


def neuron_0(Q):
    # scale S = 12.17;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.17449 * (0.09199563
        - 0.3392436 * max(0.0, Q.mass - 93.0) / 14.44097   # -33.9%  mass > 93
        - 0.2579338 * max(0.0, Q.mass - 78.0) / 21.50831   # -25.8%  mass > 78
        + 0.1129031 * max(0.0, Q.mass - 87.0) / 16.74224   # +11.3%  mass > 87
        + 0.07824219 * max(0.0, 0.0077 - Q.mass_over_sum_pt_sq) / 0.002131004   # +7.8%  mass_over_sum_pt_sq < 0.0077
        - 0.03254839 * max(0.0, 0.0061 - Q.e2_sq) / 0.001266007   # -3.3%  e2_sq < 0.0061
        + 0.02784303 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +2.8%  n_dr_0p2_0p4 < 15
        + 0.02163866 * max(0.0, 1200.0 - Q.sum_pt_top50) / 174.4634   # +2.2%  sum_pt_top50 < 1200
        - 0.02156641 * max(0.0, 100.0 - Q.mass) / 22.83131   # -2.2%  mass < 100
        - 0.02095322 * max(0.0, 0.07 - Q.tau1) / 0.01321735   # -2.1%  tau1 < 0.07
        - 0.02000555 * max(0.0, 0.0073 - Q.girth2_top20) / 0.002577327   # -2.0%  girth2_top20 < 0.0073
        - 0.01836073 * max(0.0, 0.091 - Q.z_dr_0p2_0p4) / 0.05806041   # -1.8%  z_dr_0p2_0p4 < 0.091
        + 0.0162125 * max(0.0, 7.0 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 960.0) / 2.23027   # +1.6%  log_sum_pt < 7 and sum_pt_top50 > 960
        - 0.01103819 * max(0.0, 1100.0 - Q.sum_pt_top40) / 95.98881   # -1.1%  sum_pt_top40 < 1100
        - 0.007315968 * max(0.0, 980.0 - Q.sum_pt) / 10.86197   # -0.7%  sum_pt < 980
        - 0.006499337 * max(0.0, 0.99 - Q.z_top50_slots) / 0.004627259   # -0.6%  z_top50_slots < 0.99
        + 0.005300717 * max(0.0, 950.0 - Q.sum_pt_top30) / 24.82059   # +0.5%  sum_pt_top30 < 950
        + 0.001329407 * max(0.0, Q.mass - 78.0) * max(0.0, 0.07 - Q.sj2_zsoft) / 0.006084531   # +0.1%  mass > 78 and sj2_zsoft < 0.07
        - 0.001065227 * max(0.0, Q.mass_top50 - 72.0) * max(0.0, 0.069 - Q.sj2_zsoft) / 0.007583972   # -0.1%  mass_top50 > 72 and sj2_zsoft < 0.069
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 14.43;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.43121 * (0.1621485
        + 0.1428209 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922642   # +14.3%  log_sum_pt > 6.9
        - 0.111354 * max(0.0, 2.3 - Q.soft1_pt) / 1.675676   # -11.1%  soft1_pt < 2.3
        - 0.06829538 * max(0.0, 7.1 - Q.log_sum_pt) / 0.1631764   # -6.8%  log_sum_pt < 7.1
        + 0.06298846 * max(0.0, 1.6 - Q.soft1_pt) / 1.023649   # +6.3%  soft1_pt < 1.6
        + 0.05222886 * max(0.0, 700.0 - Q.sum_pt_top2) / 333.507   # +5.2%  sum_pt_top2 < 700
        + 0.04742499 * max(0.0, 1100.0 - Q.sum_pt_top40) / 95.98881   # +4.7%  sum_pt_top40 < 1100
        - 0.04714593 * max(0.0, 100.0 - Q.mass) / 22.83131   # -4.7%  mass < 100
        - 0.04252415 * max(0.0, Q.z_top50_slots - 0.96) / 0.03299329   # -4.3%  z_top50_slots > 0.96
        - 0.03940399 * max(0.0, Q.sum_pt_top50 - 960.0) / 85.76884   # -3.9%  sum_pt_top50 > 960
        + 0.03821796 * max(0.0, 88.0 - Q.mass) / 14.55228   # +3.8%  mass < 88
        + 0.03195025 * max(0.0, 0.0087 - Q.girth2_top5) / 0.004843287   # +3.2%  girth2_top5 < 0.0087
        - 0.03107182 * max(0.0, Q.log_sum_pt - 7.0) / 0.01916257   # -3.1%  log_sum_pt > 7
        - 0.02505153 * max(0.0, 51.0 - Q.m012) / 36.51757   # -2.5%  m012 < 51
        + 0.02499415 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +2.5%  n_particles > 38
        - 0.02448681 * max(0.0, 31.0 - Q.sj3_mass1) * max(0.0, 21.0 - Q.sj3_mass2) / 211.6014   # -2.4%  sj3_mass1 < 31 and sj3_mass2 < 21
        + 0.02227428 * max(0.0, 0.078 - Q.tau2) / 0.04658622   # +2.2%  tau2 < 0.078
        - 0.01743909 * max(0.0, Q.zdr_0 - 0.00098) / 0.008799553   # -1.7%  zdr_0 > 0.00098
        - 0.01559631 * max(0.0, 7.4 - Q.n_dr_0p2_0p4) / 2.255247   # -1.6%  n_dr_0p2_0p4 < 7.4
        - 0.01484558 * max(0.0, 32.0 - Q.sj2_mass1) / 9.155545   # -1.5%  sj2_mass1 < 32
        + 0.0131945 * max(0.0, 0.001 - Q.girth2_top5) / 0.000250543   # +1.3%  girth2_top5 < 0.001
        + 0.01243788 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.15 - Q.dr_1) / 0.8974687   # +1.2%  n_particles > 38 and dr_1 < 0.15
        + 0.01172795 * max(0.0, Q.z_top30_slots - 0.93) * max(0.0, 0.075 - Q.C2) / 0.000984003   # +1.2%  z_top30_slots > 0.93 and C2 < 0.075
        - 0.01157383 * max(0.0, 48.0 - Q.mass_top20) / 6.073616   # -1.2%  mass_top20 < 48
        + 0.01056765 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.013 - Q.zdr_2) / 0.08425638   # +1.1%  n_particles > 38 and zdr_2 < 0.013
        - 0.0102806 * max(0.0, 17.0 - Q.mass_top5) / 5.454468   # -1.0%  mass_top5 < 17
        + 0.009623425 * max(0.0, 0.0033 - Q.girth2_top15) / 0.0008074286   # +1.0%  girth2_top15 < 0.0033
        + 0.00907665 * max(0.0, 48.0 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 30.0) / 35.21158   # +0.9%  mass_top20 < 48 and n_real_top40 > 30
        - 0.007161436 * max(0.0, 0.032 - Q.z_7) / 0.003422126   # -0.7%  z_7 < 0.032
        + 0.006622649 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # +0.7%  sum_pt < 1000
        - 0.006104832 * max(0.0, 0.03 - Q.M3) * max(0.0, Q.M2 - 0.046) / 0.0001850843   # -0.6%  M3 < 0.03 and M2 > 0.046
        + 0.005649085 * max(0.0, Q.z_top30_slots - 0.93) * max(0.0, Q.max_pair_mass - 14.0) / 0.1998116   # +0.6%  z_top30_slots > 0.93 and max_pair_mass > 14
        - 0.005027781 * max(0.0, Q.z_dr_0_0p05 - 0.85) / 0.0236342   # -0.5%  z_dr_0_0p05 > 0.85
        + 0.004933946 * max(0.0, 0.00083 - Q.girth2_top15) / 9.875566e-05   # +0.5%  girth2_top15 < 0.00083
        + 0.004892586 * max(0.0, 0.17 - Q.tau1) * max(0.0, Q.eta_0 - -0.016) / 0.001796589   # +0.5%  tau1 < 0.17 and eta_0 > -0.016
        - 0.003972524 * max(0.0, 7.6 - Q.n_dr_0p1_0p2) / 1.182028   # -0.4%  n_dr_0p1_0p2 < 7.6
        + 0.003324958 * max(0.0, Q.z_top30_slots - 0.93) * max(0.0, Q.ptdr0_3 - 6.5) / 0.06484213   # +0.3%  z_top30_slots > 0.93 and ptdr0_3 > 6.5
        + 0.00282319 * max(0.0, Q.z_top30_slots - 0.96) * max(0.0, Q.ptdr0_4 - 3.7) / 0.03182973   # +0.3%  z_top30_slots > 0.96 and ptdr0_4 > 3.7
        + 0.0008900397 * max(0.0, Q.sum_pt_top30 - 1200.0) / 6.553241   # +0.1%  sum_pt_top30 > 1200
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 0.5764;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5763749 * (-0.04875299
        + 0.6459659 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.025 - Q.girth2_top15) / 0.001163495   # +64.6%  log_sum_pt > 6.9 and girth2_top15 < 0.025
        - 0.2696359 * max(0.0, 87.0 - Q.mass) / 14.00102   # -27.0%  mass < 87
        - 0.08439823 * max(0.0, Q.log_sum_pt - 7.1) / 0.007783203   # -8.4%  log_sum_pt > 7.1
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 0.9931;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.9930579 * (-0.0431999
        + 0.2343194 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +23.4%  n_particles < 46
        + 0.1571584 * max(0.0, 0.00061 - Q.lam2) / 0.0001472334   # +15.7%  lam2 < 0.00061
        + 0.1403066 * max(0.0, 6.4 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.97) / 0.04771665   # +14.0%  n_dr_0p2_0p4 < 6.4 and z_top50_slots > 0.97
        - 0.1341123 * max(0.0, 0.38 - Q.tau21) / 0.06625935   # -13.4%  tau21 < 0.38
        + 0.1135976 * max(0.0, 14.0 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.95) / 0.1406596   # +11.4%  n_dr_0p1_0p2 < 14 and psi_0p2 > 0.95
        - 0.1074863 * max(0.0, 0.38 - Q.tau21) * max(0.0, Q.lam1 - 0.0036) / 0.0003421157   # -10.7%  tau21 < 0.38 and lam1 > 0.0036
        - 0.08533549 * max(0.0, 56.0 - Q.mass) / 3.997315   # -8.5%  mass < 56
        - 0.02768387 * max(0.0, 6.6 - Q.n_dr_0p2_0p4) * max(0.0, 1000.0 - Q.sum_pt_top30) / 38.7753   # -2.8%  n_dr_0p2_0p4 < 6.6 and sum_pt_top30 < 1000
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 7.212;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.211931 * (0.001816435
        + 0.1348618 * max(0.0, 100.0 - Q.mass) / 22.83131   # +13.5%  mass < 100
        - 0.1240535 * max(0.0, 87.0 - Q.mass) / 14.00102   # -12.4%  mass < 87
        + 0.1157581 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.004 - Q.lam2) / 0.07950853   # +11.6%  mass < 100 and lam2 < 0.004
        - 0.08884228 * max(0.0, Q.n_particles - 21.0) / 24.93091   # -8.9%  n_particles > 21
        - 0.08259114 * max(0.0, 80.0 - Q.mass) * max(0.0, 0.0037 - Q.lam2) / 0.03443015   # -8.3%  mass < 80 and lam2 < 0.0037
        + 0.07188971 * max(0.0, 6.5 - Q.D2) / 3.756983   # +7.2%  D2 < 6.5
        - 0.06822266 * max(0.0, 74.0 - Q.mass) / 8.512406   # -6.8%  mass < 74
        - 0.06166586 * max(0.0, 89.0 - Q.mass_top40) * max(0.0, 6.8 - Q.D2) / 42.35523   # -6.2%  mass_top40 < 89 and D2 < 6.8
        + 0.05503552 * max(0.0, 25.0 - Q.n_dr_0p2_0p4) / 16.2669   # +5.5%  n_dr_0p2_0p4 < 25
        - 0.03634034 * max(0.0, 0.072 - Q.tau1) / 0.01401519   # -3.6%  tau1 < 0.072
        + 0.02614826 * max(0.0, Q.e2_sq - 0.014) / 0.002213374   # +2.6%  e2_sq > 0.014
        + 0.02563164 * max(0.0, Q.n_dr_0_0p05 - 6.8) / 7.00203   # +2.6%  n_dr_0_0p05 > 6.8
        - 0.02085675 * max(0.0, 0.0054 - Q.girth2_top15) / 0.001675027   # -2.1%  girth2_top15 < 0.0054
        + 0.01523493 * max(0.0, Q.n_particles - 24.0) * max(0.0, 2.1 - Q.soft1_pt) / 29.85686   # +1.5%  n_particles > 24 and soft1_pt < 2.1
        - 0.01168719 * max(0.0, Q.girth2_top15 - 0.0072) / 0.002947106   # -1.2%  girth2_top15 > 0.0072
        - 0.00986173 * max(0.0, Q.mass_top20 - 100.0) / 4.36332   # -1.0%  mass_top20 > 100
        - 0.008961719 * max(0.0, Q.LHA - 0.37) / 0.007446002   # -0.9%  LHA > 0.37
        + 0.00841223 * max(0.0, Q.mass - 140.0) / 4.527494   # +0.8%  mass > 140
        + 0.007778074 * max(0.0, Q.lam2 - 0.00044) / 0.001044598   # +0.8%  lam2 > 0.00044
        + 0.007401267 * max(0.0, Q.e2 - 0.038) / 0.004202947   # +0.7%  e2 > 0.038
        - 0.005906265 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # -0.6%  sum_pt < 1000
        + 0.004817436 * max(0.0, Q.sj3_pair_mass_min - 34.0) / 5.523532   # +0.5%  sj3_pair_mass_min > 34
        + 0.004275681 * max(0.0, 120.0 - Q.mass_top30) * max(0.0, 1000.0 - Q.sum_pt_top40) / 997.9261   # +0.4%  mass_top30 < 120 and sum_pt_top40 < 1000
        - 0.003765917 * max(0.0, 86.0 - Q.mass) * max(0.0, Q.zdr_0 - 0.0069) / 0.007091261   # -0.4%  mass < 86 and zdr_0 > 0.0069
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.88;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.87578 * (-0.01094174
        + 0.1853725 * max(0.0, Q.sum_pt - 920.0) / 128.4122   # +18.5%  sum_pt > 920
        + 0.1649022 * max(0.0, Q.sum_pt_top50 - 950.0) / 94.39159   # +16.5%  sum_pt_top50 > 950
        - 0.1402267 * max(0.0, Q.sum_pt - 990.0) / 66.59719   # -14.0%  sum_pt > 990
        - 0.1056469 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922642   # -10.6%  log_sum_pt > 6.9
        + 0.04120076 * max(0.0, Q.mass_top40 - 59.0) / 31.1174   # +4.1%  mass_top40 > 59
        - 0.03423133 * max(0.0, Q.z_top30_slots - 0.9) / 0.06014418   # -3.4%  z_top30_slots > 0.9
        + 0.03385262 * max(0.0, Q.sd_mass - 70.0) / 11.57779   # +3.4%  sd_mass > 70
        - 0.02436392 * max(0.0, Q.sd_mass - 83.0) / 7.066045   # -2.4%  sd_mass > 83
        - 0.02393506 * max(0.0, Q.mass_over_sum_pt - 0.092) / 0.01377315   # -2.4%  mass_over_sum_pt > 0.092
        + 0.02229279 * max(0.0, Q.soft3_pt - 0.38) / 0.6716111   # +2.2%  soft3_pt > 0.38
        + 0.02080841 * max(0.0, 0.041 - Q.e2) / 0.0134707   # +2.1%  e2 < 0.041
        - 0.0198665 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.2 - Q.D2) / 11.02366   # -2.0%  n_particles < 64 and D2 < 2.2
        - 0.01982212 * max(0.0, 22.0 - Q.n_dr_0p1_0p2) / 10.67233   # -2.0%  n_dr_0p1_0p2 < 22
        + 0.01886801 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +1.9%  n_particles < 46
        - 0.0173726 * max(0.0, 0.036 - Q.z_dr_0p2_0p4) / 0.01765801   # -1.7%  z_dr_0p2_0p4 < 0.036
        - 0.01602415 * max(0.0, Q.soft3_z - 0.00083) / 0.0003556637   # -1.6%  soft3_z > 0.00083
        + 0.01522554 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.487524   # +1.5%  n_dr_0p2_0p4 < 11
        + 0.0152107 * max(0.0, 23.0 - Q.sj3_mass1) / 8.893994   # +1.5%  sj3_mass1 < 23
        - 0.01415154 * max(0.0, Q.mass - 160.0) / 1.812828   # -1.4%  mass > 160
        + 0.01347168 * max(0.0, 2.0 - Q.D2) / 0.378592   # +1.3%  D2 < 2
        - 0.01261458 * max(0.0, 0.49 - Q.tau21) / 0.1224942   # -1.3%  tau21 < 0.49
        + 0.01173772 * max(0.0, 0.42 - Q.max_dr) / 0.0755366   # +1.2%  max_dr < 0.42
        - 0.0116443 * max(0.0, Q.n_particles - 46.0) / 6.207886   # -1.2%  n_particles > 46
        - 0.00862558 * max(0.0, Q.girth2_top50 - 0.016) / 0.001776703   # -0.9%  girth2_top50 > 0.016
        - 0.006223706 * max(0.0, 0.71 - Q.psi_0p1) / 0.07438206   # -0.6%  psi_0p1 < 0.71
        - 0.002308108 * max(0.0, Q.max_dr - 0.33) / 0.04427246   # -0.2%  max_dr > 0.33
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 4.606;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.605518 * (0.2757561
        - 0.4754378 * max(0.0, 120.0 - Q.mass) / 38.34741   # -47.5%  mass < 120
        + 0.2506914 * max(0.0, 84.0 - Q.mass) / 12.41467   # +25.1%  mass < 84
        + 0.0907778 * max(0.0, 76.0 - Q.sj3_pair_mass_min) / 49.89007   # +9.1%  sj3_pair_mass_min < 76
        + 0.05075151 * max(0.0, Q.e2 - 0.022) * max(0.0, Q.psi_0p3 - 0.99) / 5.993256e-05   # +5.1%  e2 > 0.022 and psi_0p3 > 0.99
        - 0.04890108 * max(0.0, Q.e2 - 0.027) / 0.009192441   # -4.9%  e2 > 0.027
        - 0.0383234 * max(0.0, 2.8 - Q.D2) / 0.8171255   # -3.8%  D2 < 2.8
        - 0.02074253 * max(0.0, Q.mass - 140.0) / 4.527494   # -2.1%  mass > 140
        - 0.008755177 * max(0.0, Q.sj3_mass1 - 22.0) / 1.471611   # -0.9%  sj3_mass1 > 22
        + 0.008617053 * max(0.0, Q.mass_top10 - 95.0) / 0.9774876   # +0.9%  mass_top10 > 95
        + 0.007002305 * max(0.0, 150.0 - Q.mass) * max(0.0, 990.0 - Q.sum_pt_top50) / 888.409   # +0.7%  mass < 150 and sum_pt_top50 < 990
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 27.79;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.78871 * (-0.01997214
        - 0.181528 * max(0.0, 93.0 - Q.mass) / 17.69975   # -18.2%  mass < 93
        - 0.1784641 * max(0.0, 91.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.96) / 0.5918003   # -17.8%  mass < 91 and psi_0p3 > 0.96
        + 0.1669487 * max(0.0, 100.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.96) / 0.8225689   # +16.7%  mass < 100 and psi_0p3 > 0.96
        + 0.1546496 * max(0.0, 91.0 - Q.mass) / 16.34035   # +15.5%  mass < 91
        - 0.05434678 * max(0.0, 0.0082 - Q.girth2_top30) / 0.002849485   # -5.4%  girth2_top30 < 0.0082
        - 0.04486383 * max(0.0, 0.0034 - Q.girth2_top50) / 0.000455003   # -4.5%  girth2_top50 < 0.0034
        + 0.04214956 * max(0.0, 81.0 - Q.mass) / 10.94656   # +4.2%  mass < 81
        + 0.04055339 * max(0.0, 0.012 - Q.girth2_top30) / 0.005808899   # +4.1%  girth2_top30 < 0.012
        - 0.03781278 * max(0.0, 0.11 - Q.tau1) / 0.03635877   # -3.8%  tau1 < 0.11
        + 0.03375593 * max(0.0, 0.31 - Q.LHA) / 0.06748444   # +3.4%  LHA < 0.31
        + 0.02219764 * max(0.0, 0.0068 - Q.girth2_top30) / 0.001927637   # +2.2%  girth2_top30 < 0.0068
        - 0.008547639 * max(0.0, Q.z_top20_slots - 0.92) / 0.02219887   # -0.9%  z_top20_slots > 0.92
        + 0.007924434 * max(0.0, 0.0076 - Q.girth2) * max(0.0, 0.12 - Q.M2) / 8.843767e-05   # +0.8%  girth2 < 0.0076 and M2 < 0.12
        + 0.007678233 * max(0.0, 6.3 - Q.n_dr_0p2_0p4) * max(0.0, 2.3 - Q.soft1_pt) / 2.879463   # +0.8%  n_dr_0p2_0p4 < 6.3 and soft1_pt < 2.3
        + 0.006230519 * max(0.0, 0.23 - Q.tau21_b2) / 0.0498957   # +0.6%  tau21_b2 < 0.23
        - 0.003379703 * max(0.0, 6.2 - Q.n_dr_0p2_0p4) * max(0.0, 6.7e-05 - Q.e3) / 5.188818e-05   # -0.3%  n_dr_0p2_0p4 < 6.2 and e3 < 6.7e-05
        - 0.003168675 * max(0.0, 91.0 - Q.mass) * max(0.0, Q.sd_mass - 54.0) / 33.35356   # -0.3%  mass < 91 and sd_mass > 54
        + 0.002881101 * max(0.0, 120.0 - Q.mass) * max(0.0, Q.sd_mass - 76.0) / 33.49878   # +0.3%  mass < 120 and sd_mass > 76
        - 0.001642669 * max(0.0, 0.25 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.19) / 0.0020562   # -0.2%  tau21_b2 < 0.25 and sj2_dr > 0.19
        + 0.001276738 * max(0.0, 6.1 - Q.n_dr_0p2_0p4) * max(0.0, Q.pt_4 - 60.0) / 11.90567   # +0.1%  n_dr_0p2_0p4 < 6.1 and pt_4 > 60
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 12.99;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.99204 * (-0.108528
        + 0.1568602 * max(0.0, Q.mass_over_sum_pt_sq - 0.00055) / 0.008746494   # +15.7%  mass_over_sum_pt_sq > 0.00055
        - 0.1159678 * max(0.0, 0.017 - Q.girth2_top15) / 0.01076185   # -11.6%  girth2_top15 < 0.017
        - 0.1124518 * max(0.0, Q.e2_sq - 0.0098) / 0.003135146   # -11.2%  e2_sq > 0.0098
        + 0.1077173 * max(0.0, Q.e2_sq - 0.008) / 0.003616195   # +10.8%  e2_sq > 0.008
        + 0.1053184 * max(0.0, Q.mass - 81.0) / 19.68778   # +10.5%  mass > 81
        - 0.09143741 * max(0.0, Q.mass_top50 - 81.0) / 18.24821   # -9.1%  mass_top50 > 81
        + 0.07329902 * max(0.0, 0.0082 - Q.girth2_top15) / 0.003501116   # +7.3%  girth2_top15 < 0.0082
        + 0.06166652 * max(0.0, 0.00035 - Q.e3) / 0.0002679511   # +6.2%  e3 < 0.00035
        + 0.0405458 * max(0.0, Q.mass - 65.0) / 30.80541   # +4.1%  mass > 65
        - 0.03638591 * max(0.0, Q.mass - 100.0) / 12.57253   # -3.6%  mass > 100
        - 0.02540143 * max(0.0, Q.mass - 120.0) / 8.088636   # -2.5%  mass > 120
        - 0.01493362 * max(0.0, 16.0 - Q.n_dr_0p2_0p4) / 8.291371   # -1.5%  n_dr_0p2_0p4 < 16
        - 0.01248426 * max(0.0, 0.0094 - Q.girth2_top15) * max(0.0, 0.61 - Q.tau21_b2) / 0.0009950671   # -1.2%  girth2_top15 < 0.0094 and tau21_b2 < 0.61
        + 0.01136525 * max(0.0, 6.9 - Q.log_sum_pt) / 0.01461958   # +1.1%  log_sum_pt < 6.9
        + 0.008975178 * max(0.0, Q.n_dr_0p2_0p4 - 8.1) / 3.598946   # +0.9%  n_dr_0p2_0p4 > 8.1
        + 0.007184287 * max(0.0, Q.sj2_dr - 0.23) / 0.02185914   # +0.7%  sj2_dr > 0.23
        + 0.00669407 * max(0.0, Q.e2 - 0.047) / 0.00219067   # +0.7%  e2 > 0.047
        + 0.004272979 * max(0.0, Q.log_sum_pt - 7.0) * max(0.0, Q.sj3_pair_mass_max - 16.0) / 1.261698   # +0.4%  log_sum_pt > 7 and sj3_pair_mass_max > 16
        + 0.003641561 * max(0.0, Q.n_pt_above_1 - 55.0) / 1.598355   # +0.4%  n_pt_above_1 > 55
        - 0.003397096 * max(0.0, Q.n_dr_0p2_0p4 - 8.9) * max(0.0, Q.zdr_0 - 0.0039) / 0.03221547   # -0.3%  n_dr_0p2_0p4 > 8.9 and zdr_0 > 0.0039
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 7.766;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.76639 * (0.1095747
        - 0.3190547 * max(0.0, 160.0 - Q.mass_top40) / 76.00929   # -31.9%  mass_top40 < 160
        + 0.2056598 * max(0.0, 130.0 - Q.mass_top40) / 49.44998   # +20.6%  mass_top40 < 130
        + 0.1125352 * max(0.0, 84.0 - Q.mass) / 12.41467   # +11.3%  mass < 84
        + 0.05047106 * max(0.0, 0.0061 - Q.girth2_top40) / 0.001385081   # +5.0%  girth2_top40 < 0.0061
        - 0.04810722 * max(0.0, 0.047 - Q.girth) / 0.00736922   # -4.8%  girth < 0.047
        - 0.04021361 * max(0.0, 64.0 - Q.mass) / 5.815914   # -4.0%  mass < 64
        + 0.04015941 * max(0.0, Q.girth - 0.093) / 0.009227622   # +4.0%  girth > 0.093
        + 0.03771634 * max(0.0, 0.21 - Q.LHA) / 0.02122607   # +3.8%  LHA < 0.21
        + 0.02553373 * max(0.0, 0.29 - Q.z_dr_0p1_0p2) / 0.162545   # +2.6%  z_dr_0p1_0p2 < 0.29
        - 0.02273544 * max(0.0, Q.mass - 140.0) / 4.527494   # -2.3%  mass > 140
        - 0.02173138 * max(0.0, Q.mass - 160.0) / 1.812828   # -2.2%  mass > 160
        - 0.01936819 * max(0.0, Q.psi_0p3 - 0.99) / 0.005785419   # -1.9%  psi_0p3 > 0.99
        + 0.01492848 * max(0.0, Q.mass - 170.0) / 0.8918493   # +1.5%  mass > 170
        + 0.01158112 * max(0.0, 960.0 - Q.sum_pt_top50) / 10.07206   # +1.2%  sum_pt_top50 < 960
        - 0.008046743 * max(0.0, Q.girth2_top30 - 0.024) / 0.0005122471   # -0.8%  girth2_top30 > 0.024
        - 0.007102412 * max(0.0, 72.0 - Q.mass_top40) * max(0.0, 1000.0 - Q.sum_pt) / 183.867   # -0.7%  mass_top40 < 72 and sum_pt < 1000
        + 0.005816198 * max(0.0, 88.0 - Q.mass) * max(0.0, Q.n_particles - 40.0) / 64.90066   # +0.6%  mass < 88 and n_particles > 40
        + 0.005512465 * max(0.0, 73.0 - Q.mass_top40) * max(0.0, 940.0 - Q.sum_pt_top50) / 98.87288   # +0.6%  mass_top40 < 73 and sum_pt_top50 < 940
        + 0.002052821 * max(0.0, 110.0 - Q.mass_top30) * max(0.0, Q.n_dr_0p2_0p4 - 11.0) / 21.96006   # +0.2%  mass_top30 < 110 and n_dr_0p2_0p4 > 11
        - 0.001673645 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 19.0 - Q.n_pt_above_10) / 18.17927   # -0.2%  sum_pt_top40 < 930 and n_pt_above_10 < 19
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 10.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.40937 * (0.06436511
        - 0.1615705 * max(0.0, Q.mass - 61.0) / 33.83997   # -16.2%  mass > 61
        + 0.1456341 * max(0.0, Q.mass - 81.0) / 19.68778   # +14.6%  mass > 81
        + 0.129352 * max(0.0, 0.018 - Q.girth2_top30) / 0.01077178   # +12.9%  girth2_top30 < 0.018
        - 0.08254779 * max(0.0, 0.062 - Q.e2) / 0.03170739   # -8.3%  e2 < 0.062
        - 0.0681301 * max(0.0, 78.0 - Q.sj2_mass1) / 48.2443   # -6.8%  sj2_mass1 < 78
        + 0.04315189 * max(0.0, 850.0 - Q.sum_pt_top5) / 264.2258   # +4.3%  sum_pt_top5 < 850
        + 0.04306497 * max(0.0, 3.8 - Q.D2) / 1.509357   # +4.3%  D2 < 3.8
        - 0.03466469 * max(0.0, 0.056 - Q.tau1) / 0.008352719   # -3.5%  tau1 < 0.056
        - 0.03170743 * max(0.0, Q.mass - 140.0) / 4.527494   # -3.2%  mass > 140
        + 0.02989672 * max(0.0, Q.mass - 88.0) / 16.29351   # +3.0%  mass > 88
        + 0.02350638 * max(0.0, 89.0 - Q.sj2_mass1) * max(0.0, 530.0 - Q.sum_pt_top2) / 9671.403   # +2.4%  sj2_mass1 < 89 and sum_pt_top2 < 530
        + 0.02329071 * max(0.0, Q.mass - 59.0) * max(0.0, 2.7 - Q.soft2_pt) / 59.27667   # +2.3%  mass > 59 and soft2_pt < 2.7
        - 0.02224682 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922642   # -2.2%  log_sum_pt > 6.9
        + 0.01937522 * max(0.0, 0.078 - Q.mass_over_sum_pt) / 0.01084321   # +1.9%  mass_over_sum_pt < 0.078
        - 0.01832894 * max(0.0, Q.mass_over_sum_pt - 0.17) / 0.0005999769   # -1.8%  mass_over_sum_pt > 0.17
        - 0.01688912 * max(0.0, 66.0 - Q.mass_top10) / 23.85415   # -1.7%  mass_top10 < 66
        + 0.01529872 * max(0.0, Q.mass_over_sum_pt_sq - 0.029) / 0.0002131861   # +1.5%  mass_over_sum_pt_sq > 0.029
        - 0.01483789 * max(0.0, Q.mass - 160.0) / 1.812828   # -1.5%  mass > 160
        + 0.01473112 * max(0.0, 0.064 - Q.dr_0) / 0.02509683   # +1.5%  dr_0 < 0.064
        + 0.01451054 * max(0.0, Q.mass_top50 - 140.0) / 3.76672   # +1.5%  mass_top50 > 140
        + 0.01380624 * max(0.0, Q.mass_top5 - 15.0) / 16.46211   # +1.4%  mass_top5 > 15
        - 0.01193152 * max(0.0, 3.6 - Q.D2) * max(0.0, Q.sj2_dr - 0.2) / 0.0349858   # -1.2%  D2 < 3.6 and sj2_dr > 0.2
        - 0.008743963 * max(0.0, Q.M2 - 0.055) / 0.01605275   # -0.9%  M2 > 0.055
        - 0.005981591 * max(0.0, 0.005 - Q.girth2_top20) / 0.001302606   # -0.6%  girth2_top20 < 0.005
        - 0.002102945 * max(0.0, 990.0 - Q.sum_pt) / 12.80136   # -0.2%  sum_pt < 990
        + 0.001959497 * max(0.0, 110.0 - Q.mass_top40) * max(0.0, Q.pt1_dr01 - 13.0) / 26.35287   # +0.2%  mass_top40 < 110 and pt1_dr01 > 13
        + 0.001451988 * max(0.0, Q.mass - 90.0) * max(0.0, Q.psi_0p3 - 0.98) / 0.1035224   # +0.1%  mass > 90 and psi_0p3 > 0.98
        + 0.00128655 * max(0.0, Q.mass_top40 - 160.0) * max(0.0, Q.soft4_z - 0.0011) / 0.0004715552   # +0.1%  mass_top40 > 160 and soft4_z > 0.0011
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 4.303;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.302782 * (-0.1280567
        + 0.2603681 * max(0.0, 96.0 - Q.mass) / 19.8636   # +26.0%  mass < 96
        - 0.1437397 * max(0.0, 81.0 - Q.mass) / 10.94656   # -14.4%  mass < 81
        + 0.105596 * max(0.0, 9.0 - Q.n_dr_0p2_0p4) / 3.177318   # +10.6%  n_dr_0p2_0p4 < 9
        + 0.07324037 * max(0.0, 0.0043 - Q.girth2_top10) / 0.001406863   # +7.3%  girth2_top10 < 0.0043
        - 0.06676558 * max(0.0, 0.0065 - Q.lam1) / 0.001784334   # -6.7%  lam1 < 0.0065
        - 0.06262797 * max(0.0, 0.0058 - Q.girth2_top2) / 0.003137072   # -6.3%  girth2_top2 < 0.0058
        + 0.06131268 * max(0.0, Q.psi_0p3 - 0.99) / 0.005785419   # +6.1%  psi_0p3 > 0.99
        + 0.05810861 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # +5.8%  n_dr_0p1_0p2 < 17
        + 0.04397396 * max(0.0, 0.24 - Q.LHA) / 0.03137817   # +4.4%  LHA < 0.24
        - 0.04095244 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0061 - Q.girth2) / 0.006182786   # -4.1%  n_dr_0p2_0p4 < 10 and girth2 < 0.0061
        - 0.01925154 * max(0.0, Q.z_top10_slots - 0.75) / 0.05344205   # -1.9%  z_top10_slots > 0.75
        + 0.01897785 * max(0.0, 0.15 - Q.tau21_b2) / 0.01944228   # +1.9%  tau21_b2 < 0.15
        + 0.01862628 * max(0.0, 95.0 - Q.mass) * max(0.0, Q.mass_top10 - 38.0) / 67.91935   # +1.9%  mass < 95 and mass_top10 > 38
        - 0.016256 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 36.0) / 25.90594   # -1.6%  n_dr_0p2_0p4 < 10 and n_particles > 36
        - 0.01020297 * max(0.0, 84.0 - Q.mass) * max(0.0, 3.6 - Q.D2) / 6.022104   # -1.0%  mass < 84 and D2 < 3.6
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 2.042;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.041675 * (-0.209142
        + 0.6141433 * max(0.0, 84.0 - Q.mass) / 12.41467   # +61.4%  mass < 84
        - 0.1319598 * max(0.0, 56.0 - Q.mass) / 3.997315   # -13.2%  mass < 56
        - 0.07912215 * max(0.0, 90.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3) / 0.05939035   # -7.9%  mass < 90 and psi_0p3 < 1
        - 0.07736092 * max(0.0, 88.0 - Q.mass) * max(0.0, 0.075 - Q.z_dr_0p1_0p2) / 0.755722   # -7.7%  mass < 88 and z_dr_0p1_0p2 < 0.075
        - 0.03094511 * max(0.0, 970.0 - Q.sum_pt) / 9.291157   # -3.1%  sum_pt < 970
        + 0.01977934 * max(0.0, Q.z_dr_0p1_0p2 - 0.4) / 0.02884499   # +2.0%  z_dr_0p1_0p2 > 0.4
        + 0.01361116 * max(0.0, 90.0 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.4) / 0.3540073   # +1.4%  mass < 90 and z_dr_0p05_0p1 > 0.4
        - 0.0135751 * max(0.0, Q.sum_pt - 1200.0) / 11.40574   # -1.4%  sum_pt > 1200
        + 0.009752085 * max(0.0, Q.sj3_pair_mass_max - 130.0) * max(0.0, 530.0 - Q.pt_0) / 442.4576   # +1.0%  sj3_pair_mass_max > 130 and pt_0 < 530
        + 0.009751099 * max(0.0, Q.sd_mass - 130.0) * max(0.0, 37.0 - Q.sj3_mass1) / 16.05531   # +1.0%  sd_mass > 130 and sj3_mass1 < 37
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 4.202;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.202149 * (0.4426307
        - 0.1854002 * max(0.0, 1100.0 - Q.sum_pt) / 79.17474   # -18.5%  sum_pt < 1100
        - 0.1583812 * max(0.0, Q.mass - 140.0) / 4.527494   # -15.8%  mass > 140
        - 0.1140823 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # -11.4%  sum_pt < 1000
        + 0.1029106 * max(0.0, Q.mass - 77.0) / 22.1767   # +10.3%  mass > 77
        + 0.09593972 * max(0.0, 1100.0 - Q.sum_pt_top40) / 95.98881   # +9.6%  sum_pt_top40 < 1100
        + 0.07430986 * max(0.0, Q.mass_top50 - 140.0) / 3.76672   # +7.4%  mass_top50 > 140
        + 0.06229855 * max(0.0, 1000.0 - Q.sum_pt_top50) / 18.83366   # +6.2%  sum_pt_top50 < 1000
        + 0.05706887 * max(0.0, Q.psi_0p3 - 0.98) / 0.01378229   # +5.7%  psi_0p3 > 0.98
        - 0.04625856 * max(0.0, Q.mass_top50 - 97.0) / 12.14909   # -4.6%  mass_top50 > 97
        - 0.04018332 * max(0.0, Q.mass_top50 - 160.0) / 1.31919   # -4.0%  mass_top50 > 160
        + 0.02866494 * max(0.0, Q.mass_top40 - 150.0) / 1.668343   # +2.9%  mass_top40 > 150
        - 0.01662501 * max(0.0, 880.0 - Q.sum_pt_top20) / 32.49339   # -1.7%  sum_pt_top20 < 880
        + 0.01470619 * max(0.0, Q.mass_over_sum_pt - 0.17) / 0.0005999769   # +1.5%  mass_over_sum_pt > 0.17
        + 0.00317054 * max(0.0, 990.0 - Q.sum_pt) * max(0.0, 0.018 - Q.z_9) / 0.01049062   # +0.3%  sum_pt < 990 and z_9 < 0.018
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 6.744;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.744077 * (0.03558678
        - 0.1741315 * max(0.0, 0.091 - Q.mass_over_sum_pt) / 0.0182637   # -17.4%  mass_over_sum_pt < 0.091
        - 0.16597 * max(0.0, 91.0 - Q.mass) / 16.34035   # -16.6%  mass < 91
        + 0.1577197 * max(0.0, 0.12 - Q.mass_over_sum_pt) / 0.04059823   # +15.8%  mass_over_sum_pt < 0.12
        - 0.1440737 * max(0.0, 83.0 - Q.mass) / 11.9074   # -14.4%  mass < 83
        - 0.08887601 * max(0.0, Q.sum_pt_top50 - 950.0) / 94.39159   # -8.9%  sum_pt_top50 > 950
        + 0.07051938 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922642   # +7.1%  log_sum_pt > 6.9
        + 0.0395757 * max(0.0, Q.psi_0p3 - 0.99) * max(0.0, 1.1 - Q.N3) / 0.003896373   # +4.0%  psi_0p3 > 0.99 and N3 < 1.1
        + 0.03318438 * max(0.0, 0.36 - Q.tau21_b2) / 0.1196781   # +3.3%  tau21_b2 < 0.36
        - 0.02864027 * max(0.0, 20.0 - Q.n_dr_0p1_0p2) / 8.983822   # -2.9%  n_dr_0p1_0p2 < 20
        - 0.0191207 * max(0.0, 0.39 - Q.N2) * max(0.0, Q.max_dr - 0.25) / 0.009344307   # -1.9%  N2 < 0.39 and max_dr > 0.25
        + 0.01704208 * max(0.0, 24.0 - Q.n_pt_above_10) / 5.296457   # +1.7%  n_pt_above_10 < 24
        - 0.01689258 * max(0.0, Q.e2 - 0.018) / 0.01500986   # -1.7%  e2 > 0.018
        + 0.01545062 * max(0.0, 83.0 - Q.mass) * max(0.0, Q.n_for_50pct - 4.3) / 7.088445   # +1.5%  mass < 83 and n_for_50pct > 4.3
        - 0.01006519 * max(0.0, Q.mass_top50 - 130.0) / 5.344916   # -1.0%  mass_top50 > 130
        - 0.007674961 * max(0.0, 0.34 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -13.0) / 3.175492   # -0.8%  tau21_b2 < 0.34 and orientation_deg > -13
        + 0.006127624 * max(0.0, Q.psi_0p3 - 0.99) * max(0.0, Q.pt_3 - 63.0) / 0.08906285   # +0.6%  psi_0p3 > 0.99 and pt_3 > 63
        - 0.00493568 * max(0.0, 0.38 - Q.tau21_b2) * max(0.0, 0.12 - Q.z_1) / 0.001742754   # -0.5%  tau21_b2 < 0.38 and z_1 < 0.12
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 4.666;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.666006 * (0.04864974
        + 0.1250368 * max(0.0, 76.0 - Q.sd_mass) / 28.32148   # +12.5%  sd_mass < 76
        + 0.1021805 * max(0.0, 0.009 - Q.girth2_top5) / 0.00508831   # +10.2%  girth2_top5 < 0.009
        + 0.09507201 * max(0.0, 7.0 - Q.log_sum_pt) / 0.07455573   # +9.5%  log_sum_pt < 7
        - 0.08630243 * max(0.0, Q.psi_0p2 - 0.86) / 0.09894046   # -8.6%  psi_0p2 > 0.86
        - 0.07410051 * max(0.0, Q.lam1 - 0.017) / 0.0009294448   # -7.4%  lam1 > 0.017
        - 0.07342971 * max(0.0, 43.0 - Q.sd_mass) / 12.41389   # -7.3%  sd_mass < 43
        - 0.06760845 * max(0.0, 0.97 - Q.z_dr_0_0p05) / 0.466659   # -6.8%  z_dr_0_0p05 < 0.97
        + 0.05976242 * max(0.0, 0.12 - Q.z_dr_0p1_0p2) / 0.04710335   # +6.0%  z_dr_0p1_0p2 < 0.12
        - 0.05773798 * max(0.0, 0.069 - Q.tau1) / 0.01282885   # -5.8%  tau1 < 0.069
        + 0.05685199 * max(0.0, 0.002 - Q.lam2) / 0.00113364   # +5.7%  lam2 < 0.002
        - 0.05505192 * max(0.0, Q.psi_0p3 - 0.99) / 0.005785419   # -5.5%  psi_0p3 > 0.99
        - 0.04417217 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # -4.4%  sum_pt < 1000
        - 0.02896714 * max(0.0, Q.n_dr_0p2_0p4 - 8.6) / 3.396002   # -2.9%  n_dr_0p2_0p4 > 8.6
        - 0.02595396 * max(0.0, 0.0094 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.08) / 0.001175741   # -2.6%  girth2_top5 < 0.0094 and sj3_pairmin_over_m > 0.08
        + 0.02126883 * max(0.0, Q.sj2_dr - 0.23) / 0.02185914   # +2.1%  sj2_dr > 0.23
        - 0.0190207 * max(0.0, 0.0054 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.0) / 0.02528511   # -1.9%  girth2_top5 < 0.0054 and sj3_mass1 > 5
        - 0.007482531 * max(0.0, Q.sj3_dr_max - 0.36) / 0.01163785   # -0.7%  sj3_dr_max > 0.36
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.5939237920168067, 2.8934831407563024, 0.2300733193277311, 0.36241013655462184, 0.7209996848739496, 1.0719641806722688, 0.5195484243697479, 0.34634096638655465, 1.084765231092437, 1.0947188025210084, 1.2694773109243698, 0.7653580882352942, 0.5143857142857143, 1.530103256302521, 0.24929243697478992, 0.38661806722689074]
T = [3.8660355682116596, 2.4717246142988443, 3.987812957917542, 4.01082215730042, 4.012311409368435]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +47%, n4 -16%, n9 +15%, n5 -9%, n3 -7%, n12 +3% ...
            + 0.467773 * h[1] / H_AVG[1]
            - 0.1631839 * h[4] / H_AVG[4]
            + 0.146006 * h[9] / H_AVG[9]
            - 0.08664918 * h[5] / H_AVG[5]
            - 0.07030654 * h[3] / H_AVG[3]
            + 0.03118418 * h[12] / H_AVG[12]
            + 0.02099811 * h[6] / H_AVG[6]
            + 0.008768392 * h[8] / H_AVG[8]
            - 0.00513073 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -31%, n9 +25%, n1 -22%, n11 -8%, n12 +7%, n6 +5% ...
            - 0.3099302 * h[4] / H_AVG[4]
            + 0.2491294 * h[9] / H_AVG[9]
            - 0.2194937 * h[1] / H_AVG[1]
            - 0.07741134 * h[11] / H_AVG[11]
            + 0.07153713 * h[12] / H_AVG[12]
            + 0.04598053 * h[6] / H_AVG[6]
            + 0.01163526 * h[2] / H_AVG[2]
            - 0.008024997 * h[10] / H_AVG[10]
            + 0.00685734 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +16%, n11 +11%, n0 +11%, n14 -9%, n4 +6% ...
            - 0.2380176 * h[8] / H_AVG[8]
            + 0.1554058 * h[5] / H_AVG[5]
            + 0.113955 * h[11] / H_AVG[11]
            + 0.111701 * h[0] / H_AVG[0]
            - 0.08595616 * h[14] / H_AVG[14]
            + 0.06215027 * h[4] / H_AVG[4]
            - 0.06005039 * h[9] / H_AVG[9]
            - 0.05428116 * h[7] / H_AVG[7]
            - 0.05240196 * h[12] / H_AVG[12]
            + 0.03975975 * h[3] / H_AVG[3]
            - 0.01817811 * h[15] / H_AVG[15]
            + 0.008142753 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -25%, n0 -20%, n5 +18%, n6 -13%, n7 +8%, n15 +5% ...
            - 0.2535558 * h[8] / H_AVG[8]
            - 0.2036104 * h[0] / H_AVG[0]
            + 0.1837467 * h[5] / H_AVG[5]
            - 0.1254886 * h[6] / H_AVG[6]
            + 0.07825615 * h[7] / H_AVG[7]
            + 0.05422147 * h[15] / H_AVG[15]
            + 0.04007795 * h[12] / H_AVG[12]
            + 0.03953165 * h[3] / H_AVG[3]
            - 0.02151117 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -35%, n10 +31%, n5 -13%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3456003 * h[13] / H_AVG[13]
            + 0.3114518 * h[10] / H_AVG[10]
            - 0.1252353 * h[5] / H_AVG[5]
            + 0.05702889 * h[8] / H_AVG[8]
            - 0.04807569 * h[12] / H_AVG[12]
            - 0.03613423 * h[15] / H_AVG[15]
            + 0.03369316 * h[4] / H_AVG[4]
            - 0.02427738 * h[7] / H_AVG[7]
            + 0.01850317 * h[0] / H_AVG[0]
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
