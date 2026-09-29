"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.4%   (on for 65% of jets)
  neuron  1:  11.7%   (on for 95% of jets)
  neuron  5:  11.6%   (on for 80% of jets)
  neuron  4:   8.8%   (on for 70% of jets)
  neuron  9:   7.4%   (on for 75% of jets)
  neuron 13:   7.2%   (on for 86% of jets)
  neuron 10:   7.1%   (on for 82% of jets)
  neuron  0:   7.0%   (on for 54% of jets)
  neuron  7:   5.1%   (on for 46% of jets)
  neuron  6:   4.5%   (on for 63% of jets)
  neuron 12:   4.0%   (on for 36% of jets)
  neuron  3:   3.7%   (on for 82% of jets)
  neuron 11:   3.0%   (on for 71% of jets)
  neuron 15:   2.6%   (on for 67% of jets)
  neuron 14:   2.2%   (on for 38% of jets)
  neuron  2:   0.7%   (on for 100% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.6% (the network: 81.1%); same class as the network for 91.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
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
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
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
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        z_top2_slots=sum(pt[:2]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
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
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    # scale S = 8.495;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.495239 * (0.1859865
        - 0.3965805 * max(0.0, Q.mass - 78.9) / 20.92576   # -39.7%  mass > 78.9
        - 0.1928201 * max(0.0, Q.mass - 88.9) / 15.90342   # -19.3%  mass > 88.9
        + 0.1548693 * max(0.0, 0.00768 - Q.mass_over_sum_pt_sq) / 0.002118602   # +15.5%  mass_over_sum_pt_sq < 0.00768
        - 0.1000183 * max(0.0, 0.00614 - Q.sum_zz_dr2) / 0.001283503   # -10.0%  sum_zz_dr2 < 0.00614
        - 0.0479601 * max(0.0, 0.00639 - Q.sum_z_dr2_top20) / 0.001997218   # -4.8%  sum_z_dr2_top20 < 0.00639
        - 0.04016166 * max(0.0, 0.0695 - Q.tau1) / 0.01302225   # -4.0%  tau1 < 0.0695
        + 0.02441897 * max(0.0, 0.00498 - Q.mass_over_sum_pt_sq) / 0.0008679709   # +2.4%  mass_over_sum_pt_sq < 0.00498
        - 0.02200814 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # -2.2%  sum_pt < 1010
        - 0.01430333 * max(0.0, 0.994 - Q.z_top50_slots) / 0.005678046   # -1.4%  z_top50_slots < 0.994
        + 0.006859618 * max(0.0, 852.0 - Q.sum_pt_top40) / 3.311028   # +0.7%  sum_pt_top40 < 852
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 10.39;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.39228 * (0.2963739
        - 0.2306448 * max(0.0, 7.16 - Q.log_sum_pt) / 0.2199014   # -23.1%  log_sum_pt < 7.16
        + 0.14304 * max(0.0, 699.0 - Q.sum_pt_top2) / 332.553   # +14.3%  sum_pt_top2 < 699
        + 0.1189413 * max(0.0, Q.n_particles - 38.5) / 10.47517   # +11.9%  n_particles > 38.5
        + 0.1156913 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922642   # +11.6%  log_sum_pt > 6.9
        - 0.06422089 * max(0.0, Q.n_particles - 37.8) * max(0.0, 2.32 - Q.soft1_pt) / 15.70356   # -6.4%  n_particles > 37.8 and soft1_pt < 2.32
        - 0.05174833 * max(0.0, 32.3 - Q.sj3_mass1) * max(0.0, 18.5 - Q.sj3_mass2) / 185.4424   # -5.2%  sj3_mass1 < 32.3 and sj3_mass2 < 18.5
        - 0.04018003 * max(0.0, 31.5 - Q.sj3_mass1) / 16.56992   # -4.0%  sj3_mass1 < 31.5
        - 0.03858803 * max(0.0, 0.0312 - Q.M3) / 0.00869886   # -3.9%  M3 < 0.0312
        - 0.03724112 * max(0.0, Q.log_sum_pt - 6.99) / 0.0210337   # -3.7%  log_sum_pt > 6.99
        + 0.03384391 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # +3.4%  sum_pt < 1010
        + 0.03382463 * max(0.0, Q.n_particles - 35.1) * max(0.0, 0.127 - Q.dr_0) / 0.8490699   # +3.4%  n_particles > 35.1 and dr_0 < 0.127
        - 0.02875608 * max(0.0, 706.0 - Q.sum_pt_top2) * max(0.0, 0.959 - Q.tau43) / 39.84548   # -2.9%  sum_pt_top2 < 706 and tau43 < 0.959
        + 0.01794482 * max(0.0, 0.000549 - Q.sum_z_dr2_top3) / 0.0001172877   # +1.8%  sum_z_dr2_top3 < 0.000549
        - 0.01752464 * max(0.0, 6.23 - Q.n_dr_0p2_0p4) / 1.655645   # -1.8%  n_dr_0p2_0p4 < 6.23
        + 0.01561087 * max(0.0, 51.1 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 27.7) / 52.84447   # +1.6%  mass_top20 < 51.1 and n_real_top40 > 27.7
        + 0.01219919 * max(0.0, Q.z_top30_slots - 0.947) * max(0.0, Q.max_pair_mass - 2.55) / 0.2356456   # +1.2%  z_top30_slots > 0.947 and max_pair_mass > 2.55
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 0.2;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.1999684 * (0.3825604
        + 0.8770254 * max(0.0, Q.log_sum_pt - 6.93) * max(0.0, 0.0271 - Q.sum_z_dr2_top15) / 0.000876887   # +87.7%  log_sum_pt > 6.93 and sum_z_dr2_top15 < 0.0271
        - 0.1229746 * max(0.0, Q.sum_pt_top20 - 1130.0) * max(0.0, Q.eta_1 - 0.0866) / 0.003543378   # -12.3%  sum_pt_top20 > 1130 and eta_1 > 0.0866
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 2.031;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.031493 * (0.02205275
        + 0.2262301 * max(0.0, 0.0166 - Q.tau4) / 0.003147841   # +22.6%  tau4 < 0.0166
        - 0.1962792 * max(0.0, 0.384 - Q.tau21) / 0.06804432   # -19.6%  tau21 < 0.384
        - 0.1821005 * max(0.0, 51.5 - Q.n_particles) / 9.461273   # -18.2%  n_particles < 51.5
        + 0.1740832 * max(0.0, 53.2 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 762.0) / 2806.737   # +17.4%  n_particles < 53.2 and sum_pt_top30 > 762
        + 0.1087996 * max(0.0, 26.6 - Q.sj2_mass1) / 5.740922   # +10.9%  sj2_mass1 < 26.6
        + 0.07713846 * max(0.0, 2.43 - Q.D2) * max(0.0, Q.psi_0p3 - 0.999) / 0.0002548069   # +7.7%  D2 < 2.43 and psi_0p3 > 0.999
        - 0.03536896 * max(0.0, 0.00144 - Q.lam1) / 0.0001157033   # -3.5%  lam1 < 0.00144
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 6.343;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.343076 * (0.1592287
        + 0.2318986 * max(0.0, 104.0 - Q.mass) / 25.8515   # +23.2%  mass < 104
        - 0.2263917 * max(0.0, 78.2 - Q.mass) / 9.835751   # -22.6%  mass < 78.2
        - 0.1690312 * max(0.0, 87.4 - Q.mass) / 14.21987   # -16.9%  mass < 87.4
        + 0.1633513 * max(0.0, 73.7 - Q.mass) / 8.423982   # +16.3%  mass < 73.7
        - 0.09223958 * max(0.0, Q.n_particles - 22.4) / 23.59204   # -9.2%  n_particles > 22.4
        + 0.05082932 * max(0.0, 59.7 - Q.mass_top30) / 6.773409   # +5.1%  mass_top30 < 59.7
        - 0.04074835 * max(0.0, 0.00506 - Q.sum_z_dr2_top15) / 0.00151152   # -4.1%  sum_z_dr2_top15 < 0.00506
        + 0.02550998 * max(0.0, Q.sj3_pair_mass_min - 33.1) / 5.738005   # +2.6%  sj3_pair_mass_min > 33.1
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 8.517;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.516503 * (0.01066165
        + 0.3659944 * max(0.0, Q.sum_pt_top50 - 934.0) / 108.606   # +36.6%  sum_pt_top50 > 934
        - 0.1582702 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # -15.8%  log_sum_pt > 6.91
        - 0.07040891 * max(0.0, Q.sum_z_dr2_top30 - 0.00755) / 0.00333132   # -7.0%  sum_z_dr2_top30 > 0.00755
        - 0.0655813 * max(0.0, Q.z_top30_slots - 0.898) / 0.06178356   # -6.6%  z_top30_slots > 0.898
        + 0.05957591 * max(0.0, Q.sd_mass - 71.9) / 10.74954   # +6.0%  sd_mass > 71.9
        + 0.05654844 * max(0.0, 57.5 - Q.n_particles) / 13.37764   # +5.7%  n_particles < 57.5
        + 0.05639314 * max(0.0, Q.mass - 70.6) / 26.6818   # +5.6%  mass > 70.6
        - 0.05165462 * max(0.0, Q.sum_pt_top40 - 1020.0) / 37.59972   # -5.2%  sum_pt_top40 > 1020
        - 0.04295623 * max(0.0, 0.524 - Q.tau21) / 0.142905   # -4.3%  tau21 < 0.524
        - 0.04237445 * max(0.0, Q.sd_mass - 88.0) / 5.94534   # -4.2%  sd_mass > 88
        - 0.03024244 * max(0.0, Q.mass_top50 - 154.0) / 1.93654   # -3.0%  mass_top50 > 154
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 8.849;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.84918 * (0.1299555
        - 0.3810767 * max(0.0, 101.0 - Q.mass) / 23.58193   # -38.1%  mass < 101
        + 0.2618144 * max(0.0, 92.6 - Q.mass) / 17.41987   # +26.2%  mass < 92.6
        + 0.09830899 * max(0.0, Q.sum_z_dr2_top20 - 0.00754) / 0.003041797   # +9.8%  sum_z_dr2_top20 > 0.00754
        - 0.06946163 * max(0.0, Q.sum_z_dr2_top20 - 0.00729) * max(0.0, Q.C2_b2 - -0.000102) / 9.299221e-05   # -6.9%  sum_z_dr2_top20 > 0.00729 and C2_b2 > -0.000102
        + 0.0431248 * max(0.0, 92.1 - Q.mass_top15) / 35.66534   # +4.3%  mass_top15 < 92.1
        + 0.03580315 * max(0.0, 72.6 - Q.mass_top50) / 8.44876   # +3.6%  mass_top50 < 72.6
        - 0.03485001 * max(0.0, Q.e2 - 0.0274) / 0.008964943   # -3.5%  e2 > 0.0274
        - 0.0311154 * max(0.0, Q.sj3_dr_min - 0.119) / 0.02256933   # -3.1%  sj3_dr_min > 0.119
        - 0.03079649 * max(0.0, Q.LHA - 0.37) / 0.007446002   # -3.1%  LHA > 0.37
        + 0.007326129 * max(0.0, Q.e2 - 0.0572) * max(0.0, Q.zdr_0 - 0.00437) / 1.785957e-05   # +0.7%  e2 > 0.0572 and zdr_0 > 0.00437
        + 0.006322307 * max(0.0, Q.sj3_dr_min - 0.112) * max(0.0, 7.8 - Q.sj3_mass2) / 0.02531549   # +0.6%  sj3_dr_min > 0.112 and sj3_mass2 < 7.8
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 19.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.77809 * (0.0121852
        + 0.2936932 * max(0.0, 102.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.977) / 0.4684429   # +29.4%  mass < 102 and psi_0p3 > 0.977
        - 0.2632879 * max(0.0, 91.4 - Q.mass) * max(0.0, Q.psi_0p3 - 0.975) / 0.3542402   # -26.3%  mass < 91.4 and psi_0p3 > 0.975
        - 0.1860778 * max(0.0, 93.8 - Q.mass) * max(0.0, Q.psi_0p3 - 0.967) / 0.5333715   # -18.6%  mass < 93.8 and psi_0p3 > 0.967
        - 0.05452178 * max(0.0, 83.1 - Q.mass) * max(0.0, Q.psi_0p3 - 0.998) / 0.007928947   # -5.5%  mass < 83.1 and psi_0p3 > 0.998
        - 0.04693916 * max(0.0, 85.7 - Q.sd_mass) / 35.29913   # -4.7%  sd_mass < 85.7
        + 0.04410046 * max(0.0, 69.8 - Q.sd_mass) / 24.63907   # +4.4%  sd_mass < 69.8
        + 0.03814029 * max(0.0, 98.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997) / 0.02825252   # +3.8%  mass < 98.2 and psi_0p3 > 0.997
        - 0.02932551 * max(0.0, 0.106 - Q.tau1) / 0.03333349   # -2.9%  tau1 < 0.106
        + 0.02618572 * max(0.0, 0.248 - Q.tau21_b2) / 0.05812611   # +2.6%  tau21_b2 < 0.248
        - 0.0177282 * max(0.0, 0.394 - Q.tau21) / 0.07259421   # -1.8%  tau21 < 0.394
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 4.274;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.273736 * (0.1541977
        + 0.3648176 * max(0.0, Q.mass - 73.3) / 24.74816   # +36.5%  mass > 73.3
        - 0.193676 * max(0.0, Q.mass - 104.0) / 11.59272   # -19.4%  mass > 104
        - 0.1474344 * max(0.0, 14.9 - Q.n_dr_0p2_0p4) / 7.404182   # -14.7%  n_dr_0p2_0p4 < 14.9
        - 0.08166979 * max(0.0, Q.psi_0p3 - 0.994) / 0.002957925   # -8.2%  psi_0p3 > 0.994
        + 0.07516694 * max(0.0, Q.mass_over_sum_pt - 0.0758) * max(0.0, 1120.0 - Q.sum_pt) / 2.490261   # +7.5%  mass_over_sum_pt > 0.0758 and sum_pt < 1120
        + 0.07114644 * max(0.0, Q.n_particles - 50.1) / 4.34373   # +7.1%  n_particles > 50.1
        + 0.0433251 * max(0.0, 1030.0 - Q.sum_pt) / 27.96979   # +4.3%  sum_pt < 1030
        - 0.02276361 * max(0.0, Q.n_particles - 50.0) * max(0.0, 1.05 - Q.soft1_pt) / 1.439137   # -2.3%  n_particles > 50 and soft1_pt < 1.05
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 4.546;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.545524 * (0.2815957
        - 0.3484938 * max(0.0, 136.0 - Q.mass) / 51.43139   # -34.8%  mass < 136
        + 0.3285345 * max(0.0, 85.1 - Q.mass) / 12.98575   # +32.9%  mass < 85.1
        - 0.1065945 * max(0.0, 64.7 - Q.mass) / 5.989221   # -10.7%  mass < 64.7
        + 0.1004384 * max(0.0, 0.00598 - Q.sum_z_dr2_top40) / 0.001331035   # +10.0%  sum_z_dr2_top40 < 0.00598
        + 0.04129315 * max(0.0, 984.0 - Q.sum_pt) / 11.58636   # +4.1%  sum_pt < 984
        - 0.04095806 * max(0.0, Q.sj3_dr13 - 0.161) / 0.05781858   # -4.1%  sj3_dr13 > 0.161
        - 0.02275133 * max(0.0, 84.5 - Q.mass_top40) * max(0.0, 1030.0 - Q.sum_pt) / 465.841   # -2.3%  mass_top40 < 84.5 and sum_pt < 1030
        + 0.008396716 * max(0.0, Q.z_top5 - 0.805) / 0.005588211   # +0.8%  z_top5 > 0.805
        + 0.002539584 * max(0.0, 960.0 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.92) / 1.665763   # +0.3%  sum_pt_top40 < 960 and soft3_pt > 2.92
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 9.083;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.082946 * (0.1783562
        + 0.2088001 * max(0.0, 89.3 - Q.mass) / 15.29451   # +20.9%  mass < 89.3
        - 0.1787073 * max(0.0, 102.0 - Q.mass) / 24.33566   # -17.9%  mass < 102
        + 0.09555761 * max(0.0, Q.e2 - 0.0101) / 0.02116938   # +9.6%  e2 > 0.0101
        - 0.06136407 * max(0.0, Q.z_dr_0_0p05 - 0.769) / 0.05160801   # -6.1%  z_dr_0_0p05 > 0.769
        + 0.05908339 * max(0.0, 0.000117 - Q.e3) / 6.600876e-05   # +5.9%  e3 < 0.000117
        + 0.05774471 * max(0.0, 0.0949 - Q.z_dr_0p2_0p4) / 0.06120094   # +5.8%  z_dr_0p2_0p4 < 0.0949
        - 0.05766568 * max(0.0, Q.mass - 148.0) / 3.336142   # -5.8%  mass > 148
        - 0.04965909 * max(0.0, 70.6 - Q.mass_top15) / 19.19365   # -5.0%  mass_top15 < 70.6
        + 0.04699601 * max(0.0, Q.mass_top50 - 138.0) / 4.065355   # +4.7%  mass_top50 > 138
        - 0.04363951 * max(0.0, 67.4 - Q.sj2_mass1) / 38.48304   # -4.4%  sj2_mass1 < 67.4
        - 0.03797788 * max(0.0, 62.6 - Q.mass_top5) / 36.65792   # -3.8%  mass_top5 < 62.6
        - 0.03078873 * max(0.0, Q.z_top2_slots - 0.298) / 0.08849758   # -3.1%  z_top2_slots > 0.298
        - 0.02882821 * max(0.0, 11.9 - Q.n_dr_0p2_0p4) / 5.124171   # -2.9%  n_dr_0p2_0p4 < 11.9
        - 0.01814808 * max(0.0, 0.0201 - Q.sum_z_dr2_top10) * max(0.0, Q.psi_0p3 - 0.998) / 9.990184e-06   # -1.8%  sum_z_dr2_top10 < 0.0201 and psi_0p3 > 0.998
        + 0.01362024 * max(0.0, 0.319 - Q.z_top2_slots) / 0.04096422   # +1.4%  z_top2_slots < 0.319
        - 0.006384281 * max(0.0, Q.mass_top50 - 173.0) / 0.4998972   # -0.6%  mass_top50 > 173
        - 0.005035175 * max(0.0, 967.0 - Q.sum_pt_top50) / 11.18196   # -0.5%  sum_pt_top50 < 967
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 4.426;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.425912 * (-0.136695
        + 0.3646663 * max(0.0, 93.7 - Q.mass) / 18.19595   # +36.5%  mass < 93.7
        - 0.2481498 * max(0.0, 81.2 - Q.mass) / 11.03808   # -24.8%  mass < 81.2
        - 0.1154763 * max(0.0, 0.00632 - Q.lam1) / 0.00168121   # -11.5%  lam1 < 0.00632
        + 0.1049658 * max(0.0, Q.psi_0p3 - 0.99) / 0.005785419   # +10.5%  psi_0p3 > 0.99
        + 0.08148529 * max(0.0, 0.0242 - Q.e2) / 0.004178988   # +8.1%  e2 < 0.0242
        + 0.0441941 * max(0.0, Q.psi_0p2 - 0.994) / 0.001303994   # +4.4%  psi_0p2 > 0.994
        + 0.04106239 * max(0.0, 115.0 - Q.mass) * max(0.0, 0.351 - Q.planar_flow) / 1.580335   # +4.1%  mass < 115 and planar_flow < 0.351
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 2.373;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.37315 * (-0.1693951
        + 0.5602388 * max(0.0, 82.3 - Q.mass) / 11.56114   # +56.0%  mass < 82.3
        - 0.2288458 * max(0.0, 60.9 - Q.mass) / 5.075565   # -22.9%  mass < 60.9
        - 0.08712521 * max(0.0, 92.5 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3) / 0.05398465   # -8.7%  mass < 92.5 and psi_0p3 < 0.999
        - 0.04029246 * max(0.0, Q.mass - 162.0) / 1.596328   # -4.0%  mass > 162
        - 0.0326967 * max(0.0, Q.sd_mass - 128.0) * max(0.0, Q.lam2 - 0.000638) / 0.00419428   # -3.3%  sd_mass > 128 and lam2 > 0.000638
        + 0.0281889 * max(0.0, Q.sj3_pair_mass_max - 122.0) * max(0.0, 80.6 - Q.sj3_pair_mass_min) / 57.66938   # +2.8%  sj3_pair_mass_max > 122 and sj3_pair_mass_min < 80.6
        + 0.02261213 * max(0.0, Q.sd_mass - 128.0) * max(0.0, 0.422 - Q.sd_zg) / 0.1415883   # +2.3%  sd_mass > 128 and sd_zg < 0.422
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 3.136;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.136128 * (0.8226706
        - 0.2308419 * max(0.0, 1090.0 - Q.sum_pt) / 70.97545   # -23.1%  sum_pt < 1090
        + 0.1748423 * max(0.0, Q.mass_top50 - 137.0) / 4.217906   # +17.5%  mass_top50 > 137
        - 0.1613695 * max(0.0, Q.mass - 142.0) / 4.217296   # -16.1%  mass > 142
        - 0.1192327 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # -11.9%  sum_pt < 1010
        + 0.09266727 * max(0.0, 1020.0 - Q.sum_pt_top40) / 35.61475   # +9.3%  sum_pt_top40 < 1020
        - 0.09133139 * max(0.0, Q.mass - 160.0) / 1.812828   # -9.1%  mass > 160
        - 0.07587035 * max(0.0, Q.mass_top50 - 99.2) / 11.60679   # -7.6%  mass_top50 > 99.2
        + 0.04715553 * max(0.0, Q.mass - 172.0) / 0.7702383   # +4.7%  mass > 172
        + 0.006689074 * max(0.0, Q.log_sum_pt - 7.14) * max(0.0, Q.sd_zg - 0.446) / 1.294925e-05   # +0.7%  log_sum_pt > 7.14 and sd_zg > 0.446
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 8.658;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.658175 * (-0.08246541
        - 0.2190699 * max(0.0, 90.8 - Q.mass) / 16.2115   # -21.9%  mass < 90.8
        + 0.1747264 * max(0.0, 0.118 - Q.mass_over_sum_pt) / 0.03898999   # +17.5%  mass_over_sum_pt < 0.118
        - 0.1683941 * max(0.0, 82.7 - Q.mass) / 11.75795   # -16.8%  mass < 82.7
        - 0.08706972 * max(0.0, 0.0775 - Q.mass_over_sum_pt) / 0.01063279   # -8.7%  mass_over_sum_pt < 0.0775
        + 0.08235654 * max(0.0, 0.0657 - Q.tau1) / 0.01159443   # +8.2%  tau1 < 0.0657
        + 0.07296386 * max(0.0, Q.psi_0p3 - 0.993) / 0.003630655   # +7.3%  psi_0p3 > 0.993
        + 0.07074252 * max(0.0, 6.94 - Q.log_sum_pt) / 0.03062506   # +7.1%  log_sum_pt < 6.94
        - 0.03733396 * max(0.0, 973.0 - Q.sum_pt_top50) / 12.24409   # -3.7%  sum_pt_top50 < 973
        - 0.03057017 * max(0.0, Q.psi_0p3 - 0.993) * max(0.0, 0.235 - Q.sd_rg) / 0.0003915412   # -3.1%  psi_0p3 > 0.993 and sd_rg < 0.235
        + 0.02474521 * max(0.0, 0.006 - Q.sum_z_dr2_top5) / 0.002807973   # +2.5%  sum_z_dr2_top5 < 0.006
        - 0.01655842 * max(0.0, 0.331 - Q.tau21_b2) * max(0.0, 0.00769 - Q.sum_z_dr2_top40) / 9.752766e-05   # -1.7%  tau21_b2 < 0.331 and sum_z_dr2_top40 < 0.00769
        - 0.01546927 * max(0.0, 0.455 - Q.N2) * max(0.0, Q.max_dr - 0.244) / 0.01573862   # -1.5%  N2 < 0.455 and max_dr > 0.244
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 2.127;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.126771 * (-0.1260126
        + 0.3891285 * max(0.0, 0.0102 - Q.sum_z_dr2_top10) / 0.005480711   # +38.9%  sum_z_dr2_top10 < 0.0102
        + 0.1819388 * max(0.0, 0.142 - Q.z_dr_0p1_0p2) / 0.05952958   # +18.2%  z_dr_0p1_0p2 < 0.142
        - 0.176774 * max(0.0, Q.psi_0p3 - 0.989) / 0.0065384   # -17.7%  psi_0p3 > 0.989
        - 0.1287208 * max(0.0, 0.0674 - Q.tau1) / 0.01222141   # -12.9%  tau1 < 0.0674
        + 0.07521824 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0397 - Q.C2_b2) / 0.0004080918   # +7.5%  sj2_dr > 0.225 and C2_b2 < 0.0397
        - 0.04821964 * max(0.0, 0.139 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.49) / 0.236841   # -4.8%  z_dr_0p1_0p2 < 0.139 and n_dr_0p2_0p4 > 5.49
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6315200105042017, 2.9156904411764706, 0.2777512605042017, 0.4659682773109244, 0.7274054621848739, 1.1512672268907562, 0.6487329831932773, 0.5671127100840336, 1.3148589285714285, 1.1611518907563025, 1.4237034663865547, 0.7279871848739495, 0.4910439075630252, 1.6095147584033613, 0.33093109243697477, 0.46337058823529415]
T = [4.046540117844012, 2.542920624343487, 4.5662591320903365, 4.757037470456933, 4.409824828486082]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +45%, n4 -16%, n9 +15%, n5 -9%, n3 -9%, n12 +3% ...
            + 0.4503369 * h[1] / H_AVG[1]
            - 0.1572899 * h[4] / H_AVG[4]
            + 0.1479582 * h[9] / H_AVG[9]
            - 0.0889083 * h[5] / H_AVG[5]
            - 0.0863642 * h[3] / H_AVG[3]
            + 0.02844119 * h[12] / H_AVG[12]
            + 0.02504968 * h[6] / H_AVG[6]
            + 0.01015419 * h[8] / H_AVG[8]
            - 0.005497379 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -30%, n9 +26%, n1 -21%, n11 -7%, n12 +7%, n6 +6% ...
            - 0.3039294 * h[4] / H_AVG[4]
            + 0.2568495 * h[9] / H_AVG[9]
            - 0.2149859 * h[1] / H_AVG[1]
            - 0.07156999 * h[11] / H_AVG[11]
            + 0.06637893 * h[12] / H_AVG[12]
            + 0.05580604 * h[6] / H_AVG[6]
            + 0.01365316 * h[2] / H_AVG[2]
            - 0.00874796 * h[10] / H_AVG[10]
            + 0.008079163 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +15%, n0 +10%, n14 -10%, n11 +9%, n7 -8% ...
            - 0.2519571 * h[8] / H_AVG[8]
            + 0.1457597 * h[5] / H_AVG[5]
            + 0.103726 * h[0] / H_AVG[0]
            - 0.09965055 * h[14] / H_AVG[14]
            + 0.09466007 * h[11] / H_AVG[11]
            - 0.07762272 * h[7] / H_AVG[7]
            - 0.05562583 * h[9] / H_AVG[9]
            + 0.0547594 * h[4] / H_AVG[4]
            + 0.04464511 * h[3] / H_AVG[3]
            - 0.04368709 * h[12] / H_AVG[12]
            - 0.01902695 * h[15] / H_AVG[15]
            + 0.008879437 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -18%, n5 +17%, n6 -13%, n7 +11%, n15 +5% ...
            - 0.2591277 * h[8] / H_AVG[8]
            - 0.182538 * h[0] / H_AVG[0]
            + 0.1663843 * h[5] / H_AVG[5]
            - 0.1321117 * h[6] / H_AVG[6]
            + 0.1080391 * h[7] / H_AVG[7]
            + 0.05479165 * h[15] / H_AVG[15]
            + 0.04285464 * h[3] / H_AVG[3]
            + 0.03225773 * h[12] / H_AVG[12]
            - 0.02189529 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -33%, n10 +32%, n5 -12%, n8 +6%, n12 -4%, n15 -4% ...
            - 0.3307666 * h[13] / H_AVG[13]
            + 0.3178036 * h[10] / H_AVG[10]
            - 0.122376 * h[5] / H_AVG[5]
            + 0.06289435 * h[8] / H_AVG[8]
            - 0.04175709 * h[12] / H_AVG[12]
            - 0.03940383 * h[15] / H_AVG[15]
            - 0.03616934 * h[7] / H_AVG[7]
            + 0.03092833 * h[4] / H_AVG[4]
            + 0.01790094 * h[0] / H_AVG[0]
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
