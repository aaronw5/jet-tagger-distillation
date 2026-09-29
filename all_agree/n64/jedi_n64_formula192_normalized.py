"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network (step 4; all observables, tuned for agreement), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.2%   (on for 56% of jets)
  neuron  4:  11.8%   (on for 75% of jets)
  neuron  5:  10.5%   (on for 90% of jets)
  neuron  1:   9.9%   (on for 75% of jets)
  neuron  9:   9.7%   (on for 90% of jets)
  neuron  0:   7.6%   (on for 69% of jets)
  neuron 13:   6.2%   (on for 83% of jets)
  neuron 12:   5.6%   (on for 53% of jets)
  neuron  6:   5.0%   (on for 91% of jets)
  neuron  7:   4.8%   (on for 89% of jets)
  neuron 10:   4.7%   (on for 66% of jets)
  neuron  3:   4.3%   (on for 70% of jets)
  neuron 11:   2.6%   (on for 90% of jets)
  neuron 14:   2.5%   (on for 62% of jets)
  neuron 15:   1.3%   (on for 53% of jets)
  neuron  2:   0.2%   (on for 100% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 79.2% (the network: 81.1%); same class as the network for 91.1% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_7                   pT of particle 7 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_z                pT share of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.soft10_abseta          |Δη| of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_9                   ΔR of particle 9 from the jet axis
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
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
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
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
  Q.pt_dispersion          √(Σ pTᵢ²) / Σ pTᵢ
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
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_7=pt[7],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft5_z=softp(5, 'z'),
        soft9_z=softp(9, 'z'),
        z_top2_slots=sum(pt[:2]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        soft10_abseta=softp(10, 'abseta'),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_9=dr[9] if pt[9] > 0 else 0.0,
        eta_1=eta[1],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
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
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
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
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 9.084;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.083645 * (0.1087669
        + 0.2943143 * max(0.0, Q.mass - 90.0) / 15.45345   # +29.4%  mass > 90
        - 0.2733094 * max(0.0, Q.mass - 79.0) / 20.86256   # -27.3%  mass > 79
        + 0.1111818 * max(0.0, 0.0076 - Q.mass_over_sum_pt_sq) / 0.002069541   # +11.1%  mass_over_sum_pt_sq < 0.0076
        - 0.1007337 * max(0.0, Q.mass_top50 - 82.0) / 17.73312   # -10.1%  mass_top50 > 82
        - 0.0506363 * max(0.0, 0.0062 - Q.e2_sq) / 0.001310434   # -5.1%  e2_sq < 0.0062
        + 0.03265486 * max(0.0, 14.0 - Q.n_dr_0p2_0p4) / 6.695829   # +3.3%  n_dr_0p2_0p4 < 14
        + 0.02676232 * max(0.0, 7.0 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 960.0) / 2.23027   # +2.7%  log_sum_pt < 7 and sum_pt_top50 > 960
        - 0.02519601 * max(0.0, 0.084 - Q.z_dr_0p2_0p4) / 0.05249348   # -2.5%  z_dr_0p2_0p4 < 0.084
        - 0.02400867 * max(0.0, 0.07 - Q.tau1) / 0.01321735   # -2.4%  tau1 < 0.07
        - 0.02081133 * max(0.0, 0.0063 - Q.girth2_top30) / 0.00164385   # -2.1%  girth2_top30 < 0.0063
        + 0.01630477 * max(0.0, Q.psi_0p3 - 0.99) / 0.005785419   # +1.6%  psi_0p3 > 0.99
        - 0.008729401 * max(0.0, 1.0 - Q.z_top50_slots) / 0.007698523   # -0.9%  z_top50_slots < 1
        + 0.007907547 * max(0.0, 0.00069 - Q.lam2) / 0.0001885285   # +0.8%  lam2 < 0.00069
        - 0.007449663 * max(0.0, 980.0 - Q.sum_pt) / 10.86197   # -0.7%  sum_pt < 980
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 16.1;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.0977 * (0.3019066
        - 0.1186674 * max(0.0, 2.3 - Q.soft1_pt) / 1.675676   # -11.9%  soft1_pt < 2.3
        - 0.1053477 * max(0.0, Q.z_top50_slots - 0.96) / 0.03299329   # -10.5%  z_top50_slots > 0.96
        - 0.08646544 * max(0.0, 7.1 - Q.log_sum_pt) / 0.1631764   # -8.6%  log_sum_pt < 7.1
        + 0.08388544 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922642   # +8.4%  log_sum_pt > 6.9
        + 0.08311603 * max(0.0, 710.0 - Q.sum_pt_top2) / 343.0709   # +8.3%  sum_pt_top2 < 710
        + 0.05890288 * max(0.0, 0.19 - Q.tau1) / 0.1046579   # +5.9%  tau1 < 0.19
        + 0.057804 * max(0.0, 1.5 - Q.soft1_pt) / 0.9333111   # +5.8%  soft1_pt < 1.5
        - 0.05741024 * max(0.0, 120.0 - Q.mass) / 38.34741   # -5.7%  mass < 120
        - 0.04247221 * max(0.0, 33.0 - Q.sj3_mass1) * max(0.0, 20.0 - Q.sj3_mass2) / 218.436   # -4.2%  sj3_mass1 < 33 and sj3_mass2 < 20
        - 0.02611573 * max(0.0, 7.5 - Q.n_dr_0p2_0p4) / 2.309907   # -2.6%  n_dr_0p2_0p4 < 7.5
        - 0.01990049 * max(0.0, 31.0 - Q.sj3_mass1) / 16.09809   # -2.0%  sj3_mass1 < 31
        - 0.01970372 * max(0.0, 30.0 - Q.pt_9) / 5.623838   # -2.0%  pt_9 < 30
        + 0.01811465 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # +1.8%  sum_pt < 1000
        - 0.01745755 * max(0.0, 30.0 - Q.sj2_mass1) / 7.82803   # -1.7%  sj2_mass1 < 30
        + 0.01671324 * max(0.0, Q.n_particles - 39.0) * max(0.0, 0.16 - Q.dr_1) / 0.9309503   # +1.7%  n_particles > 39 and dr_1 < 0.16
        + 0.01646533 * max(0.0, 0.0044 - Q.lam1) / 0.0008633678   # +1.6%  lam1 < 0.0044
        - 0.01525177 * max(0.0, Q.z_top20_slots - 0.89) * max(0.0, 0.04 - Q.dr_2) / 0.0006137959   # -1.5%  z_top20_slots > 0.89 and dr_2 < 0.04
        + 0.01507587 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.12 - Q.dr_0) / 0.6420285   # +1.5%  n_particles > 38 and dr_0 < 0.12
        + 0.01374005 * max(0.0, Q.z_top30_slots - 0.92) * max(0.0, 0.07 - Q.C2) / 0.001000829   # +1.4%  z_top30_slots > 0.92 and C2 < 0.07
        + 0.01289349 * max(0.0, 0.00065 - Q.girth2_top5) / 0.000137454   # +1.3%  girth2_top5 < 0.00065
        - 0.01233612 * max(0.0, 7.2 - Q.log_sum_pt) * max(0.0, 6.1 - Q.pt1_dr01) / 0.9734464   # -1.2%  log_sum_pt < 7.2 and pt1_dr01 < 6.1
        + 0.01140654 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.038 - Q.dr_2) / 0.6353597   # +1.1%  mass < 120 and dr_2 < 0.038
        - 0.01090399 * max(0.0, Q.log_sum_pt - 7.0) / 0.01916257   # -1.1%  log_sum_pt > 7
        + 0.01050808 * max(0.0, 49.0 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 41.56162   # +1.1%  mass_top20 < 49 and n_real_top40 > 29
        + 0.01021793 * max(0.0, Q.n_particles - 37.0) * max(0.0, 63.0 - Q.mass_top15) / 152.3011   # +1.0%  n_particles > 37 and mass_top15 < 63
        - 0.009450968 * max(0.0, 1.5 - Q.soft1_pt) * max(0.0, 0.13 - Q.soft10_abseta) / 0.05851493   # -0.9%  soft1_pt < 1.5 and soft10_abseta < 0.13
        + 0.009163839 * max(0.0, 0.0074 - Q.girth2_top20) * max(0.0, Q.eta_1 - -0.047) / 0.0001260826   # +0.9%  girth2_top20 < 0.0074 and eta_1 > -0.047
        + 0.008713949 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.17 - Q.dr_9) / 0.9289703   # +0.9%  n_particles > 38 and dr_9 < 0.17
        + 0.007621699 * max(0.0, Q.n_particles - 39.0) * max(0.0, 0.26 - Q.sj3_dr12) / 0.9365786   # +0.8%  n_particles > 39 and sj3_dr12 < 0.26
        + 0.006537149 * max(0.0, 0.12 - Q.D3) / 0.04277766   # +0.7%  D3 < 0.12
        - 0.006103446 * max(0.0, 7.4 - Q.n_dr_0p1_0p2) / 1.1127   # -0.6%  n_dr_0p1_0p2 < 7.4
        + 0.00591515 * max(0.0, 0.00096 - Q.girth2_top15) / 0.0001261196   # +0.6%  girth2_top15 < 0.00096
        + 0.003024806 * max(0.0, 0.34 - Q.psi_0p1) / 0.02107897   # +0.3%  psi_0p1 < 0.34
        + 0.002593175 * max(0.0, Q.sum_pt_top30 - 1200.0) / 6.553241   # +0.3%  sum_pt_top30 > 1200
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 1;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.0 * (0.12
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 1.248;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.247759 * (-0.04063286
        + 0.2896082 * max(0.0, 47.0 - Q.n_particles) / 6.90939   # +29.0%  n_particles < 47
        + 0.163003 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.97) / 0.04237259   # +16.3%  n_dr_0p2_0p4 < 6 and z_top50_slots > 0.97
        - 0.1337876 * max(0.0, 0.34 - Q.tau21) / 0.04953549   # -13.4%  tau21 < 0.34
        + 0.1276773 * max(0.0, 0.009 - Q.girth2) * max(0.0, 30.0 - Q.sj3_mass1) / 0.05122524   # +12.8%  girth2 < 0.009 and sj3_mass1 < 30
        + 0.1077308 * max(0.0, 27.0 - Q.sj2_mass1) / 5.97431   # +10.8%  sj2_mass1 < 27
        + 0.08636841 * max(0.0, 9.4 - Q.n_dr_0p1_0p2) / 1.900651   # +8.6%  n_dr_0p1_0p2 < 9.4
        - 0.05416499 * max(0.0, 0.003 - Q.zdr_0) / 0.0003280817   # -5.4%  zdr_0 < 0.003
        + 0.02453087 * max(0.0, 9.4 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_pt_top40 - 1000.0) / 170.9978   # +2.5%  n_dr_0p2_0p4 < 9.4 and sum_pt_top40 > 1000
        - 0.01312883 * max(0.0, 78.0 - Q.mass_top50) * max(0.0, Q.mass_top3 - 9.1) / 3.731575   # -1.3%  mass_top50 < 78 and mass_top3 > 9.1
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 7.867;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.867319 * (0.4423362
        - 0.3419855 * Q.sum_pt_top30 / 992.8076   # -34.2%  sum_pt_top30
        - 0.1413782 * max(0.0, 85.0 - Q.mass) / 12.93334   # -14.1%  mass < 85
        + 0.1213693 * max(0.0, 120.0 - Q.mass) / 38.34741   # +12.1%  mass < 120
        - 0.06455059 * max(0.0, Q.n_particles - 25.0) / 21.16   # -6.5%  n_particles > 25
        + 0.0543896 * max(0.0, 23.0 - Q.n_dr_0p2_0p4) / 14.40742   # +5.4%  n_dr_0p2_0p4 < 23
        + 0.0523455 * max(0.0, 100.0 - Q.mass) * max(0.0, 4.1 - Q.D2_b2) / 25.90055   # +5.2%  mass < 100 and D2_b2 < 4.1
        + 0.03236193 * max(0.0, Q.n_particles - 23.0) * max(0.0, 2.4 - Q.soft1_pt) / 37.55186   # +3.2%  n_particles > 23 and soft1_pt < 2.4
        + 0.02894326 * max(0.0, 28.0 - Q.n_dr_0p1_0p2) / 16.03563   # +2.9%  n_dr_0p1_0p2 < 28
        - 0.02501789 * max(0.0, 87.0 - Q.mass_top40) * max(0.0, 3.3 - Q.D2_b2) / 7.570142   # -2.5%  mass_top40 < 87 and D2_b2 < 3.3
        - 0.02465619 * max(0.0, 1100.0 - Q.sum_pt) / 79.17474   # -2.5%  sum_pt < 1100
        - 0.02390602 * max(0.0, 0.0055 - Q.girth2_top15) / 0.00172547   # -2.4%  girth2_top15 < 0.0055
        + 0.01456824 * max(0.0, Q.e2_sq - 0.011) / 0.002851068   # +1.5%  e2_sq > 0.011
        + 0.01437217 * max(0.0, Q.n_particles - 14.0) * max(0.0, Q.n_dr_0_0p05 - 9.6) / 190.3542   # +1.4%  n_particles > 14 and n_dr_0_0p05 > 9.6
        + 0.01289107 * max(0.0, Q.sj3_pair_mass_min - 33.0) / 5.762393   # +1.3%  sj3_pair_mass_min > 33
        - 0.01141185 * max(0.0, 80.0 - Q.mass) * max(0.0, 4.1 - Q.D2_b2) / 4.118378   # -1.1%  mass < 80 and D2_b2 < 4.1
        - 0.008867372 * max(0.0, Q.sj2_dr - 0.23) * max(0.0, 20.0 - Q.D2_b2) / 0.3488122   # -0.9%  sj2_dr > 0.23 and D2_b2 < 20
        + 0.008532611 * max(0.0, Q.lam2 - 0.0012) / 0.0007249327   # +0.9%  lam2 > 0.0012
        - 0.008181121 * max(0.0, 20.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.0 - Q.z_top50_slots) / 0.05645919   # -0.8%  n_dr_0p2_0p4 < 20 and z_top50_slots < 1
        - 0.006537899 * max(0.0, Q.n_particles - 13.0) * max(0.0, 34.0 - Q.pt_7) / 102.8715   # -0.7%  n_particles > 13 and pt_7 < 34
        + 0.003733829 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sj3_dr13 - 0.14) / 0.2908438   # +0.4%  n_dr_0p2_0p4 < 15 and sj3_dr13 > 0.14
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 6.489;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.489037 * (0.08722403
        + 0.4047023 * max(0.0, Q.sum_pt_top50 - 930.0) / 112.2277   # +40.5%  sum_pt_top50 > 930
        - 0.2938942 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922642   # -29.4%  log_sum_pt > 6.9
        + 0.08549134 * max(0.0, Q.mass - 56.0) / 37.73854   # +8.5%  mass > 56
        + 0.04790258 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.932092   # +4.8%  n_dr_0p2_0p4 < 13
        - 0.03760996 * max(0.0, Q.mass - 93.0) / 14.44097   # -3.8%  mass > 93
        - 0.03373646 * max(0.0, Q.max_dr - 0.19) / 0.1645994   # -3.4%  max_dr > 0.19
        - 0.03201594 * max(0.0, Q.girth2_top30 - 0.008) / 0.003191285   # -3.2%  girth2_top30 > 0.008
        - 0.02790582 * max(0.0, 75.0 - Q.n_particles) * max(0.0, 2.9 - Q.D2) / 28.20591   # -2.8%  n_particles < 75 and D2 < 2.9
        + 0.02428978 * max(0.0, 2.8 - Q.pt_entropy) / 0.2044323   # +2.4%  pt_entropy < 2.8
        - 0.01245155 * max(0.0, Q.z_top30_slots - 0.9) * max(0.0, Q.sj3_mass1 - 8.0) / 0.3026165   # -1.2%  z_top30_slots > 0.9 and sj3_mass1 > 8
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 6.15;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.149951 * (0.1313832
        - 0.4751919 * max(0.0, 100.0 - Q.mass) / 22.83131   # -47.5%  mass < 100
        + 0.3482417 * max(0.0, 93.0 - Q.mass) / 17.69975   # +34.8%  mass < 93
        + 0.07174729 * max(0.0, 75.0 - Q.mass_top50) / 9.192549   # +7.2%  mass_top50 < 75
        + 0.04736708 * max(0.0, 94.0 - Q.mass_top15) / 37.29901   # +4.7%  mass_top15 < 94
        + 0.04147617 * max(0.0, 0.072 - Q.tau1) / 0.01401519   # +4.1%  tau1 < 0.072
        - 0.007697946 * max(0.0, Q.girth2_top30 - 0.0081) * max(0.0, Q.D2_b2 - 1.3) / 0.002491684   # -0.8%  girth2_top30 > 0.0081 and D2_b2 > 1.3
        + 0.005513945 * max(0.0, Q.girth2_top30 - 0.0085) * max(0.0, 48.0 - Q.n_real_top50) / 0.001784763   # +0.6%  girth2_top30 > 0.0085 and n_real_top50 < 48
        - 0.002763868 * max(0.0, Q.mass_over_sum_pt - 0.15) * max(0.0, 15.0 - Q.D2_b2) / 0.03365872   # -0.3%  mass_over_sum_pt > 0.15 and D2_b2 < 15
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 22.28;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.28332 * (-0.01041138
        - 0.2860332 * max(0.0, 92.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.98) / 0.2795512   # -28.6%  mass < 92 and psi_0p3 > 0.98
        + 0.2012339 * max(0.0, 100.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.98) / 0.37368   # +20.1%  mass < 100 and psi_0p3 > 0.98
        + 0.1073479 * max(0.0, 83.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.98) / 0.1976915   # +10.7%  mass < 83 and psi_0p3 > 0.98
        - 0.07551242 * max(0.0, 100.0 - Q.mass) / 22.83131   # -7.6%  mass < 100
        + 0.07537552 * max(0.0, 120.0 - Q.mass) / 38.34741   # +7.5%  mass < 120
        - 0.06575584 * max(0.0, 0.11 - Q.tau1) / 0.03635877   # -6.6%  tau1 < 0.11
        + 0.06030542 * max(0.0, 0.096 - Q.tau1) / 0.02634912   # +6.0%  tau1 < 0.096
        + 0.04354208 * max(0.0, 0.0099 - Q.girth2) / 0.00371748   # +4.4%  girth2 < 0.0099
        - 0.03579134 * max(0.0, 0.0083 - Q.girth2) / 0.002531904   # -3.6%  girth2 < 0.0083
        + 0.02023157 * max(0.0, Q.mass_top20 - 74.0) / 10.99577   # +2.0%  mass_top20 > 74
        - 0.01329117 * max(0.0, Q.mass_top20 - 86.0) / 7.018281   # -1.3%  mass_top20 > 86
        + 0.01316831 * max(0.0, 6.5 - Q.n_dr_0p2_0p4) / 1.789229   # +1.3%  n_dr_0p2_0p4 < 6.5
        - 0.001818225 * max(0.0, 92.0 - Q.mass) * max(0.0, 0.47 - Q.z_dr_0_0p05) / 0.4733188   # -0.2%  mass < 92 and z_dr_0_0p05 < 0.47
        - 0.0005931277 * max(0.0, 0.0078 - Q.lam1) * max(0.0, Q.zdr_0 - 0.015) / 6.234365e-07   # -0.1%  lam1 < 0.0078 and zdr_0 > 0.015
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 7.121;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.121209 * (0.1881703
        - 0.3456378 * max(0.0, 0.0096 - Q.width) / 0.003491289   # -34.6%  width < 0.0096
        + 0.2038159 * max(0.0, 0.0079 - Q.girth2) / 0.002257256   # +20.4%  girth2 < 0.0079
        + 0.1329456 * max(0.0, Q.mass - 69.0) / 27.84511   # +13.3%  mass > 69
        - 0.06494664 * max(0.0, Q.z_top50_slots - 0.97) / 0.02371788   # -6.5%  z_top50_slots > 0.97
        + 0.05762475 * max(0.0, Q.girth2_top20 - 0.0081) / 0.002889844   # +5.8%  girth2_top20 > 0.0081
        + 0.05552726 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # +5.6%  sum_pt < 1000
        + 0.0541609 * max(0.0, Q.n_dr_0p2_0p4 - 3.0) / 6.482203   # +5.4%  n_dr_0p2_0p4 > 3
        - 0.04486613 * max(0.0, Q.mass - 120.0) / 8.088636   # -4.5%  mass > 120
        - 0.03087199 * max(0.0, Q.psi_0p3 - 0.99) / 0.005785419   # -3.1%  psi_0p3 > 0.99
        + 0.009603003 * max(0.0, 0.084 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.0) / 0.159778   # +1.0%  tau2 < 0.084 and sj3_mass1 > 13
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 14.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.40919 * (-0.1152043
        + 0.4843531 * max(0.0, 180.0 - Q.mass) / 90.75598   # +48.4%  mass < 180
        - 0.3568137 * max(0.0, 150.0 - Q.mass) / 63.31768   # -35.7%  mass < 150
        + 0.08132905 * max(0.0, 87.0 - Q.mass) / 14.00102   # +8.1%  mass < 87
        + 0.02970258 * max(0.0, 0.0061 - Q.girth2_top40) / 0.001385081   # +3.0%  girth2_top40 < 0.0061
        + 0.01837868 * max(0.0, 76.0 - Q.mass_top30) / 12.49159   # +1.8%  mass_top30 < 76
        - 0.01094225 * max(0.0, 65.0 - Q.mass) / 6.064192   # -1.1%  mass < 65
        + 0.01060004 * max(0.0, 950.0 - Q.sum_pt_top40) / 12.83513   # +1.1%  sum_pt_top40 < 950
        + 0.003163644 * max(0.0, 120.0 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 10.0) / 28.67016   # +0.3%  mass_top40 < 120 and n_dr_0p2_0p4 > 10
        - 0.002417524 * max(0.0, 950.0 - Q.sum_pt_top40) * max(0.0, 0.0018 - Q.lam2) / 0.004713742   # -0.2%  sum_pt_top40 < 950 and lam2 < 0.0018
        + 0.002299447 * max(0.0, Q.z_top5 - 0.82) / 0.004342486   # +0.2%  z_top5 > 0.82
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 7.015;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.014566 * (-0.0277993
        + 0.1699082 * max(0.0, 88.0 - Q.mass) / 14.55228   # +17.0%  mass < 88
        + 0.1309192 * max(0.0, Q.e2 - 0.012) / 0.01958084   # +13.1%  e2 > 0.012
        - 0.124335 * max(0.0, 100.0 - Q.mass) / 22.83131   # -12.4%  mass < 100
        + 0.0876773 * max(0.0, 3.4 - Q.D2) / 1.217858   # +8.8%  D2 < 3.4
        - 0.06958883 * max(0.0, 0.019 - Q.girth2_top10) * max(0.0, 0.44 - Q.max_dr) / 0.001248428   # -7.0%  girth2_top10 < 0.019 and max_dr < 0.44
        + 0.06394481 * max(0.0, Q.mass - 79.0) / 20.86256   # +6.4%  mass > 79
        - 0.06319567 * max(0.0, 69.0 - Q.sj2_mass1) / 39.93605   # -6.3%  sj2_mass1 < 69
        - 0.06100419 * max(0.0, 0.049 - Q.tau1) / 0.006339525   # -6.1%  tau1 < 0.049
        + 0.0373441 * max(0.0, 960.0 - Q.sum_pt_top15) / 112.9106   # +3.7%  sum_pt_top15 < 960
        - 0.03308003 * max(0.0, Q.mass - 160.0) / 1.812828   # -3.3%  mass > 160
        + 0.03104547 * max(0.0, 0.58 - Q.z_top2_slots) * max(0.0, 0.0031 - Q.soft9_z) / 0.0002514671   # +3.1%  z_top2_slots < 0.58 and soft9_z < 0.0031
        + 0.02595593 * max(0.0, Q.mass_top5 - 14.0) / 17.01585   # +2.6%  mass_top5 > 14
        - 0.0245075 * max(0.0, Q.pt_dispersion - 0.26) / 0.07409892   # -2.5%  pt_dispersion > 0.26
        - 0.01311967 * max(0.0, 3.3 - Q.D2) * max(0.0, Q.sj2_dr - 0.2) / 0.02849188   # -1.3%  D2 < 3.3 and sj2_dr > 0.2
        - 0.01005567 * max(0.0, Q.sum_pt_top10 - 930.0) / 14.24974   # -1.0%  sum_pt_top10 > 930
        + 0.009800602 * max(0.0, Q.sj3_dr_min - 0.11) / 0.02546184   # +1.0%  sj3_dr_min > 0.11
        - 0.009121799 * max(0.0, Q.mass_top50 - 130.0) * max(0.0, Q.soft5_z - 0.0017) / 0.00187092   # -0.9%  mass_top50 > 130 and soft5_z > 0.0017
        - 0.007347772 * max(0.0, Q.mass_over_sum_pt - 0.16) / 0.001408236   # -0.7%  mass_over_sum_pt > 0.16
        - 0.006758251 * max(0.0, Q.mass - 76.0) * max(0.0, 0.00061 - Q.lam2) / 0.0009169477   # -0.7%  mass > 76 and lam2 < 0.00061
        - 0.006309236 * max(0.0, 950.0 - Q.sum_pt_top50) / 8.694805   # -0.6%  sum_pt_top50 < 950
        + 0.00627861 * max(0.0, Q.mass - 160.0) * max(0.0, Q.soft5_z - 0.0015) / 0.001092847   # +0.6%  mass > 160 and soft5_z > 0.0015
        + 0.005540242 * max(0.0, 100.0 - Q.mass) * max(0.0, Q.pt1_dr01 - 9.9) / 19.92943   # +0.6%  mass < 100 and pt1_dr01 > 9.9
        + 0.003161966 * max(0.0, Q.mass_top50 - 180.0) / 0.3471021   # +0.3%  mass_top50 > 180
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 2.752;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.752284 * (0.03415345
        + 0.3566256 * max(0.0, 95.0 - Q.mass) / 19.13323   # +35.7%  mass < 95
        - 0.2545449 * max(0.0, 81.0 - Q.mass) / 10.94656   # -25.5%  mass < 81
        + 0.1950842 * max(0.0, 9.1 - Q.n_dr_0p2_0p4) * max(0.0, 27.0 - Q.n_dr_0p1_0p2) / 55.58251   # +19.5%  n_dr_0p2_0p4 < 9.1 and n_dr_0p1_0p2 < 27
        - 0.1102329 * max(0.0, 9.2 - Q.n_dr_0p2_0p4) * max(0.0, 0.0064 - Q.girth2) / 0.00589111   # -11.0%  n_dr_0p2_0p4 < 9.2 and girth2 < 0.0064
        + 0.05563553 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.41 - Q.planar_flow) / 2.55634   # +5.6%  mass < 120 and planar_flow < 0.41
        + 0.02787697 * max(0.0, 0.0097 - Q.mass_over_sum_pt_sq) * max(0.0, Q.z_dr_0p1_0p2 - 0.15) / 5.149351e-05   # +2.8%  mass_over_sum_pt_sq < 0.0097 and z_dr_0p1_0p2 > 0.15
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 1.229;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.228828 * (-0.1521776
        + 0.8088586 * max(0.0, 81.0 - Q.mass) / 10.94656   # +80.9%  mass < 81
        - 0.06392889 * max(0.0, 84.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3) / 0.04438281   # -6.4%  mass < 84 and psi_0p3 < 1
        + 0.05285173 * max(0.0, 89.0 - Q.mass) * max(0.0, 0.53 - Q.tau21) / 0.7248399   # +5.3%  mass < 89 and tau21 < 0.53
        + 0.02767027 * max(0.0, Q.sd_mass - 130.0) / 0.9019094   # +2.8%  sd_mass > 130
        + 0.0202477 * max(0.0, Q.sj3_pair_mass_max - 130.0) * max(0.0, 88.0 - Q.sj3_pair_mass_min) / 47.66462   # +2.0%  sj3_pair_mass_max > 130 and sj3_pair_mass_min < 88
        - 0.01370182 * max(0.0, 6.8 - Q.log_sum_pt) * max(0.0, 0.0036 - Q.lam2) / 7.979701e-06   # -1.4%  log_sum_pt < 6.8 and lam2 < 0.0036
        + 0.01274104 * max(0.0, Q.z_dr_0p1_0p2 - 0.6) * max(0.0, Q.n_dr_0p4_up - -0.26) / 0.005493524   # +1.3%  z_dr_0p1_0p2 > 0.6 and n_dr_0p4_up > -0.26
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 3.339;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.338558 * (0.9944414
        - 0.3557288 * max(0.0, 1100.0 - Q.sum_pt) / 79.17474   # -35.6%  sum_pt < 1100
        + 0.199779 * max(0.0, 0.032 - Q.girth2_top50) * max(0.0, Q.psi_0p3 - 0.96) / 0.000779175   # +20.0%  girth2_top50 < 0.032 and psi_0p3 > 0.96
        - 0.1154455 * max(0.0, Q.mass - 150.0) / 3.058902   # -11.5%  mass > 150
        - 0.09677615 * max(0.0, 6.9 - Q.log_sum_pt) / 0.01461958   # -9.7%  log_sum_pt < 6.9
        - 0.08140964 * max(0.0, Q.mass - 140.0) * max(0.0, Q.D2 - -0.44) / 11.76583   # -8.1%  mass > 140 and D2 > -0.44
        + 0.05449399 * max(0.0, Q.n_pt_above_1 - 53.0) / 2.091165   # +5.4%  n_pt_above_1 > 53
        + 0.05340284 * max(0.0, 6.8 - Q.log_sum_pt) / 0.004369815   # +5.3%  log_sum_pt < 6.8
        - 0.02084249 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 3.0 - Q.D2) / 10.67237   # -2.1%  sum_pt < 1000 and D2 < 3
        + 0.01934522 * max(0.0, Q.sum_pt_top30 - 1100.0) / 13.80024   # +1.9%  sum_pt_top30 > 1100
        + 0.001772017 * max(0.0, Q.log_sum_pt - 7.1) * max(0.0, Q.sd_zg - 0.45) / 1.598914e-05   # +0.2%  log_sum_pt > 7.1 and sd_zg > 0.45
        + 0.00100428 * max(0.0, Q.sum_pt_top30 - 1100.0) * max(0.0, 0.012 - Q.mratio_min_012) / 0.001949329   # +0.1%  sum_pt_top30 > 1100 and mratio_min_012 < 0.012
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 1.61;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.609674 * (0.591424
        - 0.491325 * max(0.0, 91.0 - Q.mass) / 16.34035   # -49.1%  mass < 91
        - 0.231952 * max(0.0, 0.089 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p2 - 0.89) / 0.001666818   # -23.2%  mass_over_sum_pt < 0.089 and psi_0p2 > 0.89
        + 0.1407567 * max(0.0, 0.39 - Q.tau21_b2) / 0.1390015   # +14.1%  tau21_b2 < 0.39
        - 0.08853319 * max(0.0, 0.38 - Q.tau21_b2) * max(0.0, 0.008 - Q.girth2_top40) / 0.0001572954   # -8.9%  tau21_b2 < 0.38 and girth2_top40 < 0.008
        - 0.04743305 * max(0.0, Q.max_pair_mass - 19.0) / 3.653194   # -4.7%  max_pair_mass > 19
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 2.564;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.563752 * (-0.02391027
        - 0.1541601 * max(0.0, 0.072 - Q.tau1) / 0.01401519   # -15.4%  tau1 < 0.072
        + 0.1363885 * max(0.0, 7.0 - Q.log_sum_pt) / 0.07455573   # +13.6%  log_sum_pt < 7
        + 0.1358196 * max(0.0, 0.0082 - Q.girth2_top10) / 0.003873279   # +13.6%  girth2_top10 < 0.0082
        + 0.1000391 * max(0.0, 0.26 - Q.z_dr_0p1_0p2) * max(0.0, 63.0 - Q.n_particles) / 2.856073   # +10.0%  z_dr_0p1_0p2 < 0.26 and n_particles < 63
        + 0.09361849 * max(0.0, 0.092 - Q.z_dr_0p1_0p2) / 0.03234698   # +9.4%  z_dr_0p1_0p2 < 0.092
        - 0.09350748 * max(0.0, Q.e2 - 0.03) / 0.00756246   # -9.4%  e2 > 0.03
        - 0.07860645 * max(0.0, 1000.0 - Q.sum_pt) / 15.26723   # -7.9%  sum_pt < 1000
        - 0.0734465 * max(0.0, Q.psi_0p1 - 0.93) / 0.01307629   # -7.3%  psi_0p1 > 0.93
        - 0.06418772 * max(0.0, 2.3 - Q.D2) / 0.5274404   # -6.4%  D2 < 2.3
        + 0.04888647 * max(0.0, Q.sj2_dr - 0.23) * max(0.0, 0.049 - Q.C2_b2) / 0.0005402276   # +4.9%  sj2_dr > 0.23 and C2_b2 < 0.049
        + 0.0213396 * max(0.0, 0.98 - Q.z_dr_0_0p05) * max(0.0, 960.0 - Q.sum_pt) / 4.113492   # +2.1%  z_dr_0_0p05 < 0.98 and sum_pt < 960
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9401834033613445, 3.412781775210084, 0.125, 0.7366463235294117, 1.3271774159663865, 1.4335513655462184, 0.991053781512605, 0.7339859243697479, 1.7776322478991597, 2.086866911764706, 1.2823332983193276, 0.8743115546218487, 0.9309117647058823, 1.9134221113445378, 0.5114116596638656, 0.3276989495798319]
T = [5.819401207983193, 4.042694009322479, 6.506961831998425, 6.414505422794118, 5.088063383830094]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +37%, n4 -20%, n9 +18%, n3 -9%, n5 -8%, n12 +4% ...
            + 0.3665306 * h[1] / H_AVG[1]
            - 0.1995532 * h[4] / H_AVG[4]
            + 0.1849058 * h[9] / H_AVG[9]
            - 0.09493842 * h[3] / H_AVG[3]
            - 0.07698125 * h[5] / H_AVG[5]
            + 0.03749225 * h[12] / H_AVG[12]
            + 0.02660964 * h[6] / H_AVG[6]
            + 0.009545829 * h[8] / H_AVG[8]
            - 0.003443045 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -35%, n9 +29%, n1 -16%, n12 +8%, n11 -5%, n6 +5% ...
            - 0.3488085 * h[4] / H_AVG[4]
            + 0.2903664 * h[9] / H_AVG[9]
            - 0.1582847 * h[1] / H_AVG[1]
            + 0.07915536 * h[12] / H_AVG[12]
            - 0.05406738 * h[11] / H_AVG[11]
            + 0.05362588 * h[6] / H_AVG[6]
            + 0.006870543 * h[8] / H_AVG[8]
            - 0.004956214 * h[10] / H_AVG[10]
            + 0.003864997 * h[2] / H_AVG[2]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +13%, n0 +11%, n14 -11%, n11 +8%, n7 -7% ...
            - 0.2390406 * h[8] / H_AVG[8]
            + 0.127367 * h[5] / H_AVG[5]
            + 0.1083666 * h[0] / H_AVG[0]
            - 0.1080675 * h[14] / H_AVG[14]
            + 0.07977955 * h[11] / H_AVG[11]
            - 0.07050006 * h[7] / H_AVG[7]
            - 0.07015596 * h[9] / H_AVG[9]
            + 0.07011217 * h[4] / H_AVG[4]
            - 0.05811974 * h[12] / H_AVG[12]
            + 0.04952892 * h[3] / H_AVG[3]
            + 0.009519168 * h[6] / H_AVG[6]
            - 0.009442741 * h[15] / H_AVG[15]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -20%, n5 +15%, n6 -15%, n7 +10%, n3 +5% ...
            - 0.2598065 * h[8] / H_AVG[8]
            - 0.2015358 * h[0] / H_AVG[0]
            + 0.1536465 * h[5] / H_AVG[5]
            - 0.1496738 * h[6] / H_AVG[6]
            + 0.1036985 * h[7] / H_AVG[7]
            + 0.05024281 * h[3] / H_AVG[3]
            + 0.04535189 * h[12] / H_AVG[12]
            + 0.02873653 * h[15] / H_AVG[15]
            - 0.007307656 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +25%, n5 -13%, n8 +7%, n12 -7%, n4 +5% ...
            - 0.3408053 * h[13] / H_AVG[13]
            + 0.2480898 * h[10] / H_AVG[10]
            - 0.1320693 * h[5] / H_AVG[5]
            + 0.07369588 * h[8] / H_AVG[8]
            - 0.06860998 * h[12] / H_AVG[12]
            + 0.04890776 * h[4] / H_AVG[4]
            - 0.04057212 * h[7] / H_AVG[7]
            - 0.02415204 * h[15] / H_AVG[15]
            + 0.02309777 * h[0] / H_AVG[0]
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
