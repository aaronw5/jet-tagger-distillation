"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  12.8%   (on for 44% of jets)
  neuron  1:  12.4%   (on for 94% of jets)
  neuron  5:  11.5%   (on for 83% of jets)
  neuron  4:   9.7%   (on for 65% of jets)
  neuron  9:   7.6%   (on for 69% of jets)
  neuron  0:   7.6%   (on for 60% of jets)
  neuron 13:   7.3%   (on for 84% of jets)
  neuron 10:   6.3%   (on for 76% of jets)
  neuron 12:   5.0%   (on for 38% of jets)
  neuron  7:   4.0%   (on for 32% of jets)
  neuron  3:   3.8%   (on for 60% of jets)
  neuron  6:   3.7%   (on for 49% of jets)
  neuron 11:   3.6%   (on for 74% of jets)
  neuron 15:   2.2%   (on for 57% of jets)
  neuron 14:   1.9%   (on for 25% of jets)
  neuron  2:   0.7%   (on for 47% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.1% (the network: 81.1%); same class as the network for 93.5% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.log_sum_pt             natural log of the total pT
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_14                  pT of particle 14 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_pt               pT [GeV] of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.z_6                    pT of particle 6 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_z                pT share of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.sj3_z2                 pT share of subjet 2 of 3 (by pT)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft2_dr0              ΔR between the hardest and the 2. softest real particle (0 if among the 15 hardest)
  Q.soft5_dr0              ΔR between the hardest and the 5. softest real particle (0 if among the 15 hardest)
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.soft2_dr               ΔR from the jet axis of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_dr               ΔR from the jet axis of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.dr01                   ΔR between particles 0 and 1
  Q.eta_1                  Δη of particle 1
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
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
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
  Q.tau3                   N-subjettiness τ3 (β=1)
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
        z_top5=sum(zs[:5]),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        log_sum_pt=math.log(tot),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_90pct=ncum(0.9),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_11=pt[11],
        pt_14=pt[14],
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        soft1_pt=softp(1, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft7_pt=softp(7, 'pt'),
        z_6=z[6],
        soft1_z=softp(1, 'z'),
        soft3_z=softp(3, 'z'),
        soft7_z=softp(7, 'z'),
        sj3_z2=subjets(3)["z"][1],
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        soft2_dr0=softp(2, 'dr0'),
        soft5_dr0=softp(5, 'dr0'),
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        soft2_dr=softp(2, 'dr'),
        soft5_dr=softp(5, 'dr'),
        dr01=math.sqrt(dist2(0, 1)),
        eta_1=eta[1],
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
        n_pt_above_5=sum(1 for x in pt if x > 5),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
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
        tau3=tau_n(3),
        tau4=tau_n(4),
    )


def neuron_0(Q):
    # scale S = 10.16;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.16394 * (-0.1406241
        + 0.1638443 * max(0.0, 0.007872294 - Q.e2_sq) / 0.002240658   # +16.4%  e2_sq < 0.007872
        + 0.09121852 * max(0.0, Q.sum_pt_top40 - 858.8262) / 166.7848   # +9.1%  sum_pt_top40 > 858.8
        - 0.0726046 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -7.3%  e2_sq < 0.006167
        - 0.07046199 * max(0.0, Q.sum_pt - 1002.379) / 57.37467   # -7.0%  sum_pt > 1002
        + 0.0695906 * max(0.0, 0.006936725 - Q.e2_sq) / 0.001688614   # +7.0%  e2_sq < 0.006937
        - 0.06885677 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -6.9%  e2_sq < 0.009606
        + 0.05500302 * max(0.0, 1191.938 - Q.sum_pt_top30) / 206.0686   # +5.5%  sum_pt_top30 < 1192
        + 0.05074595 * max(0.0, 0.008190222 - Q.girth2) / 0.002454223   # +5.1%  girth2 < 0.00819
        + 0.049261 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # +4.9%  girth2_top50 < 0.008125
        + 0.04084168 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +4.1%  n_dr_0p2_0p4 < 18
        - 0.03967752 * max(0.0, 0.005852839 - Q.girth2_top50) * max(0.0, 1260.541 - Q.sum_pt) / 0.2467156   # -4.0%  girth2_top50 < 0.005853 and sum_pt < 1261
        - 0.03436481 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -3.4%  psi_0p2 > 0.9087
        + 0.03290432 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +3.3%  girth2_top40 < 0.008841
        - 0.02393478 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.4%  tau1 < 0.07057
        + 0.02074471 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # +2.1%  psi_0p3 > 0.9943
        - 0.01734094 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -1.7%  lam1 < 0.005914
        - 0.0162508 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -1.6%  girth2_top30 < 0.006364
        - 0.0158883 * max(0.0, 0.008190222 - Q.girth2) * max(0.0, 0.006142802 - Q.girth2_top15) / 1.08234e-05   # -1.6%  girth2 < 0.00819 and girth2_top15 < 0.006143
        - 0.0141423 * max(0.0, 1073.473 - Q.sum_pt_top30) * max(0.0, 43.66338 - Q.D2_b2) / 3778.139   # -1.4%  sum_pt_top30 < 1073 and D2_b2 < 43.66
        - 0.01377898 * max(0.0, 0.003113918 - Q.girth2_top40) / 0.0004182435   # -1.4%  girth2_top40 < 0.003114
        + 0.01359976 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # +1.4%  sum_pt > 1116
        - 0.007847722 * max(0.0, 0.00375223 - Q.girth2_top30) / 0.0006714094   # -0.8%  girth2_top30 < 0.003752
        - 0.007697678 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.log_sum_pt - 6.915514) / 0.4823065   # -0.8%  n_dr_0p2_0p4 < 18 and log_sum_pt > 6.916
        - 0.005814878 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -0.6%  sum_pt_top40 > 1025
        - 0.003584069 * max(0.0, Q.sum_pt - 1167.447) / 14.21193   # -0.4%  sum_pt > 1167
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 22.39;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.38972 * (0.03917811
        + 0.09165477 * max(0.0, Q.sum_pt - 972.0419) / 81.34075   # +9.2%  sum_pt > 972
        + 0.07642287 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +7.6%  log_sum_pt > 6.91
        - 0.06614211 * max(0.0, 2.275391 - Q.soft1_pt) / 1.652278   # -6.6%  soft1_pt < 2.275
        + 0.06117246 * max(0.0, 0.1953848 - Q.tau1) / 0.1097382   # +6.1%  tau1 < 0.1954
        - 0.06091875 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -6.1%  log_sum_pt < 7.139
        - 0.0506415 * max(0.0, Q.tau1 - 0.02607472) / 0.06189568   # -5.1%  tau1 > 0.02607
        - 0.0493685 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # -4.9%  sum_pt_top50 > 934.2
        + 0.04632306 * max(0.0, 1225.842 - Q.sum_pt_top40) / 211.117   # +4.6%  sum_pt_top40 < 1226
        + 0.04596979 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +4.6%  n_particles > 38
        + 0.04369394 * max(0.0, 0.001452174 - Q.soft1_z) / 0.0009045584   # +4.4%  soft1_z < 0.001452
        + 0.0435768 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # +4.4%  sum_pt_top50 < 1079
        - 0.02894971 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -2.9%  e2_sq < 0.009606
        + 0.0246817 * max(0.0, Q.e2 - 0.01036127) / 0.02094596   # +2.5%  e2 > 0.01036
        - 0.02155069 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -2.2%  log_sum_pt > 6.989
        - 0.02153615 * max(0.0, 1167.447 - Q.sum_pt) / 137.8628   # -2.2%  sum_pt < 1167
        - 0.02077364 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -2.1%  sum_pt > 1053
        - 0.02022449 * max(0.0, Q.sum_pt_top50 - 959.0957) / 86.539   # -2.0%  sum_pt_top50 > 959.1
        + 0.02003275 * max(0.0, 605.875 - Q.sum_pt_top2) / 245.9717   # +2.0%  sum_pt_top2 < 605.9
        + 0.01587205 * max(0.0, Q.z_top30_slots - 0.9564984) / 0.01947087   # +1.6%  z_top30_slots > 0.9565
        - 0.01131844 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.002181998 - Q.soft1_z) / 0.01445558   # -1.1%  n_particles > 38 and soft1_z < 0.002182
        - 0.01092485 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, 10.17512 - Q.ptdr0_3) / 0.1427049   # -1.1%  z_top30_slots > 0.9565 and ptdr0_3 < 10.18
        - 0.01065596 * max(0.0, 0.7058597 - Q.tau21_b2) / 0.3912084   # -1.1%  tau21_b2 < 0.7059
        - 0.01007437 * max(0.0, 0.01397874 - Q.girth2) / 0.006902497   # -1.0%  girth2 < 0.01398
        - 0.009766621 * max(0.0, Q.z_top5 - 0.6551948) * max(0.0, 28.39396 - Q.pt1_dr01) / 0.8332473   # -1.0%  z_top5 > 0.6552 and pt1_dr01 < 28.39
        - 0.009419561 * max(0.0, 0.3845054 - Q.soft2_dr) / 0.1840621   # -0.9%  soft2_dr < 0.3845
        + 0.009351717 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # +0.9%  LHA < 0.2454
        + 0.008796151 * max(0.0, 0.007463985 - Q.girth2_top30) / 0.00233708   # +0.9%  girth2_top30 < 0.007464
        + 0.008129508 * max(0.0, 0.4086381 - Q.soft2_dr0) / 0.1958521   # +0.8%  soft2_dr0 < 0.4086
        + 0.008050179 * max(0.0, 0.5100475 - Q.tau21) / 0.1343519   # +0.8%  tau21 < 0.51
        - 0.007719952 * max(0.0, 0.03457336 - Q.M3) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.0006031662   # -0.8%  M3 < 0.03457 and psi_0p3 > 0.9299
        - 0.007523638 * max(0.0, 29.04688 - Q.pt_11) / 8.607263   # -0.8%  pt_11 < 29.05
        + 0.007360502 * max(0.0, 0.004763596 - Q.girth2_top30) / 0.0009958273   # +0.7%  girth2_top30 < 0.004764
        - 0.006947653 * max(0.0, 0.02017767 - Q.tau4) / 0.005429427   # -0.7%  tau4 < 0.02018
        - 0.006862646 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -0.7%  n_dr_0p2_0p4 < 7
        - 0.006262925 * max(0.0, 0.0007143144 - Q.lam2) / 0.0002017367   # -0.6%  lam2 < 0.0007143
        + 0.006260949 * max(0.0, Q.tau1 - 0.1219132) / 0.01033631   # +0.6%  tau1 > 0.1219
        - 0.004487703 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # -0.4%  psi_0p3 > 0.9966
        + 0.004242709 * max(0.0, 0.002197765 - Q.girth2_top15) / 0.0004509993   # +0.4%  girth2_top15 < 0.002198
        - 0.004069982 * max(0.0, 7.0 - Q.n_dr_0p1_0p2) / 0.9740454   # -0.4%  n_dr_0p1_0p2 < 7
        + 0.004010958 * max(0.0, 0.0006570502 - Q.girth2_top5) / 0.0001395474   # +0.4%  girth2_top5 < 0.0006571
        + 0.003975552 * max(0.0, Q.n_dr_0_0p05 - 15.0) / 2.713461   # +0.4%  n_dr_0_0p05 > 15
        - 0.003827303 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) * max(0.0, 10.0 - Q.n_dr_0p05_0p1) / 0.2065903   # -0.4%  z_dr_0_0p05 > 0.8103 and n_dr_0p05_0p1 < 10
        + 0.003784961 * max(0.0, Q.z_top5 - 0.6551948) / 0.03331427   # +0.4%  z_top5 > 0.6552
        - 0.003289081 * max(0.0, 0.02210818 - Q.e2) / 0.00342102   # -0.3%  e2 < 0.02211
        + 0.00301244 * max(0.0, 0.2140276 - Q.tau21) / 0.01160104   # +0.3%  tau21 < 0.214
        + 0.002810938 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.009970338 - Q.zdr_0) / 0.03317992   # +0.3%  n_particles > 38 and zdr_0 < 0.00997
        + 0.002381111 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, Q.ptdr0_5 - 0.4637912) / 0.04806377   # +0.2%  z_top30_slots > 0.9565 and ptdr0_5 > 0.4638
        + 0.002335169 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.pt1_dr01 - 5.351077) / 0.1255531   # +0.2%  z_top30_slots > 0.9342 and pt1_dr01 > 5.351
        - 0.00191624 * max(0.0, Q.zdr_0 - 0.01240028) / 0.002210186   # -0.2%  zdr_0 > 0.0124
        - 0.0009245005 * max(0.0, Q.n_dr_0_0p05 - 30.0) / 0.212479   # -0.1%  n_dr_0_0p05 > 30
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 7.976;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.975621 * (-0.1037766
        - 0.1957547 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # -19.6%  LHA > 0.372
        - 0.1165067 * max(0.0, 0.006941794 - Q.girth2) / 0.001690424   # -11.7%  girth2 < 0.006942
        + 0.08416308 * max(0.0, 0.00634935 - Q.girth2_top50) / 0.001425305   # +8.4%  girth2_top50 < 0.006349
        + 0.08270786 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +8.3%  lam1 < 0.01174
        + 0.05422152 * max(0.0, Q.sum_pt - 1007.788) / 53.71957   # +5.4%  sum_pt > 1008
        - 0.0508791 * max(0.0, Q.sum_pt_top50 - 1003.544) / 52.17558   # -5.1%  sum_pt_top50 > 1004
        - 0.05084472 * max(0.0, Q.sum_pt_top30 - 800.732) / 195.5077   # -5.1%  sum_pt_top30 > 800.7
        + 0.05066718 * max(0.0, 0.01563836 - Q.girth2_top15) / 0.009593305   # +5.1%  girth2_top15 < 0.01564
        + 0.04699049 * max(0.0, Q.log_sum_pt - 6.930088) / 0.0397436   # +4.7%  log_sum_pt > 6.93
        - 0.04421433 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.girth2_top30 - 0.01215787) / 1.24419e-05   # -4.4%  log_sum_pt > 7.063 and girth2_top30 > 0.01216
        + 0.0356092 * max(0.0, Q.z_top30_slots - 0.9203881) / 0.04411614   # +3.6%  z_top30_slots > 0.9204
        + 0.03515117 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +3.5%  sum_pt_top40 > 1070
        + 0.0268859 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15) / 0.0009227558   # +2.7%  log_sum_pt > 6.903 and girth2_top15 < 0.02147
        - 0.02159785 * max(0.0, Q.sum_pt_top30 - 1011.524) / 32.30985   # -2.2%  sum_pt_top30 > 1012
        - 0.01850238 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # -1.9%  girth2_top40 < 0.008841
        - 0.01815669 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # -1.8%  log_sum_pt > 7.017
        - 0.01513202 * max(0.0, Q.sum_pt - 1085.125) / 25.7173   # -1.5%  sum_pt > 1085
        - 0.01510659 * max(0.0, 0.003638856 - Q.girth2) / 0.0004964601   # -1.5%  girth2 < 0.003639
        + 0.01257917 * max(0.0, Q.sum_pt_top50 - 1003.544) * max(0.0, Q.z_dr_0p1_0p2 - 0.003353111) / 6.148304   # +1.3%  sum_pt_top50 > 1004 and z_dr_0p1_0p2 > 0.003353
        - 0.01058102 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.8747961 - Q.psi_0p1) / 0.005466701   # -1.1%  log_sum_pt > 6.903 and psi_0p1 < 0.8748
        - 0.009857352 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -1.0%  log_sum_pt > 7.063
        + 0.002524468 * max(0.0, Q.sum_pt_top30 - 1111.245) * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.9443349   # +0.3%  sum_pt_top30 > 1111 and z_dr_0p1_0p2 < 0.1203
        - 0.0007579916 * max(0.0, Q.sum_pt_top15 - 1003.329) * max(0.0, Q.lam2 - 0.001163277) / 0.001034672   # -0.1%  sum_pt_top15 > 1003 and lam2 > 0.001163
        - 0.000608511 * max(0.0, Q.sum_pt_top15 - 1003.329) * max(0.0, Q.eta_1 - 0.08734131) / 0.006699938   # -0.1%  sum_pt_top15 > 1003 and eta_1 > 0.08734
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 3.584;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.583959 * (0.07435995
        + 0.1285224 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +12.9%  n_particles < 46
        - 0.103358 * max(0.0, 25.0 - Q.n_dr_0_0p05) / 13.07341   # -10.3%  n_dr_0_0p05 < 25
        + 0.1024579 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +10.2%  girth2_top40 < 0.008841
        - 0.09341571 * max(0.0, 0.3719813 - Q.LHA) / 0.1170757   # -9.3%  LHA < 0.372
        + 0.08432308 * max(0.0, 0.007511864 - Q.e2_sq) / 0.002016671   # +8.4%  e2_sq < 0.007512
        + 0.06904598 * max(0.0, 0.2018786 - Q.tau21_b2) / 0.03799024   # +6.9%  tau21_b2 < 0.2019
        - 0.06640668 * max(0.0, 0.347196 - Q.tau21) / 0.05239131   # -6.6%  tau21 < 0.3472
        + 0.065393 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +6.5%  n_dr_0p2_0p4 < 5
        - 0.0648107 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # -6.5%  girth2 < 0.009615
        - 0.04653206 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.e2 - 0.01036127) / 0.09655312   # -4.7%  n_particles < 46 and e2 > 0.01036
        - 0.03833359 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # -3.8%  girth2_top40 < 0.00626
        - 0.03563332 * max(0.0, 0.02647293 - Q.z_dr_0p2_0p4) / 0.01168948   # -3.6%  z_dr_0p2_0p4 < 0.02647
        + 0.03499106 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.985099) / 0.03403522   # +3.5%  n_dr_0p2_0p4 < 8 and z_top50_slots > 0.9851
        + 0.02294493 * max(0.0, 1.788105 - Q.D2) / 0.2865857   # +2.3%  D2 < 1.788
        - 0.0190834 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03029714 - Q.e2) / 0.006086253   # -1.9%  n_dr_0p2_0p4 < 5 and e2 < 0.0303
        - 0.01722315 * max(0.0, 0.00634935 - Q.girth2_top50) / 0.001425305   # -1.7%  girth2_top50 < 0.006349
        - 0.007525078 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 988.4375 - Q.sum_pt_top30) / 17.46027   # -0.8%  n_dr_0p2_0p4 < 5 and sum_pt_top30 < 988.4
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 11.84;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.83933 * (0.005850994
        + 0.09565597 * max(0.0, 0.009614971 - Q.width) / 0.003502545   # +9.6%  width < 0.009615
        + 0.07395354 * max(0.0, Q.girth2_top50 - 0.004573744) / 0.00533432   # +7.4%  girth2_top50 > 0.004574
        - 0.07326921 * max(0.0, 0.008031986 - Q.girth2_top40) / 0.002514593   # -7.3%  girth2_top40 < 0.008032
        + 0.06712748 * max(0.0, Q.girth2_top15 - 0.002197765) / 0.005632841   # +6.7%  girth2_top15 > 0.002198
        - 0.06108574 * max(0.0, Q.girth2_top50 - 0.00242543) / 0.006946092   # -6.1%  girth2_top50 > 0.002425
        + 0.05401463 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # +5.4%  girth2_top20 < 0.01083
        - 0.04888096 * max(0.0, 1061.184 - Q.sum_pt_top50) / 53.8984   # -4.9%  sum_pt_top50 < 1061
        - 0.04750024 * max(0.0, 0.007538019 - Q.girth2_top20) / 0.002740841   # -4.8%  girth2_top20 < 0.007538
        + 0.0451343 * max(0.0, 0.01563836 - Q.girth2_top15) / 0.009593305   # +4.5%  girth2_top15 < 0.01564
        - 0.04340086 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -4.3%  e2 < 0.04359
        - 0.04182021 * max(0.0, Q.n_particles - 29.0) / 17.61555   # -4.2%  n_particles > 29
        - 0.03798948 * max(0.0, 0.00592208 - Q.e2_sq) / 0.001192334   # -3.8%  e2_sq < 0.005922
        + 0.03412511 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +3.4%  sum_pt_top40 < 1053
        - 0.03105249 * max(0.0, Q.psi_0p1 - 0.3628388) / 0.4305572   # -3.1%  psi_0p1 > 0.3628
        + 0.02547732 * max(0.0, 0.03263075 - Q.e2) / 0.008034085   # +2.5%  e2 < 0.03263
        + 0.02422266 * max(0.0, 0.01397874 - Q.girth2) / 0.006902497   # +2.4%  girth2 < 0.01398
        - 0.02173521 * max(0.0, Q.girth2_top15 - 0.007887677) / 0.002755219   # -2.2%  girth2_top15 > 0.007888
        + 0.02028929 * max(0.0, Q.girth - 0.076787) / 0.01351025   # +2.0%  girth > 0.07679
        + 0.01820293 * max(0.0, Q.n_particles - 29.0) * max(0.0, 2.275391 - Q.soft1_pt) / 26.03979   # +1.8%  n_particles > 29 and soft1_pt < 2.275
        - 0.0178912 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -1.8%  lam1 < 0.006717
        + 0.01763617 * max(0.0, Q.LHA - 0.1870291) / 0.08997036   # +1.8%  LHA > 0.187
        - 0.0166911 * max(0.0, Q.girth2_top15 - 0.004169954) / 0.004340368   # -1.7%  girth2_top15 > 0.00417
        + 0.01666473 * max(0.0, 0.01397874 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.9777125) / 0.0001260069   # +1.7%  girth2 < 0.01398 and psi_0p3 > 0.9777
        - 0.01455223 * max(0.0, Q.LHA - 0.2454112) / 0.05012361   # -1.5%  LHA > 0.2454
        - 0.01227381 * max(0.0, Q.LHA - 0.2845608) / 0.02922319   # -1.2%  LHA > 0.2846
        - 0.009920934 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # -1.0%  LHA > 0.372
        + 0.009781972 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.sum_pt_top40 - 858.8262) / 0.172906   # +1.0%  psi_0p3 > 0.9974 and sum_pt_top40 > 858.8
        + 0.009558752 * max(0.0, Q.n_dr_0_0p05 - 10.0) / 5.012778   # +1.0%  n_dr_0_0p05 > 10
        - 0.003698581 * max(0.0, Q.sj2_dr - 0.2595052) * max(0.0, 0.0329485 - Q.C2_b2) / 0.0001569167   # -0.4%  sj2_dr > 0.2595 and C2_b2 < 0.03295
        - 0.003372393 * max(0.0, Q.lam1 - 0.02048524) / 0.0005048405   # -0.3%  lam1 > 0.02049
        + 0.003020512 * max(0.0, Q.e2_sq - 0.02580859) / 0.0004635665   # +0.3%  e2_sq > 0.02581
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 14.85;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.85135 * (0.06416155
        + 0.1138655 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +11.4%  sum_pt_top50 > 934.2
        - 0.102576 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -10.3%  log_sum_pt > 6.91
        - 0.09021531 * max(0.0, Q.e2_sq - 0.007872294) / 0.00365949   # -9.0%  e2_sq > 0.007872
        + 0.05091425 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +5.1%  sum_pt > 907.9 and e4 < 5.851e-08
        + 0.04817029 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +4.8%  n_particles < 64
        + 0.04419744 * max(0.0, Q.width - 0.006941794) / 0.004053246   # +4.4%  width > 0.006942
        - 0.0405891 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # -4.1%  e3 < 0.0005178
        - 0.03465914 * max(0.0, 902.4062 - Q.sum_pt_top5) / 313.0794   # -3.5%  sum_pt_top5 < 902.4
        - 0.03247863 * max(0.0, Q.e2_sq - 0.009606007) * max(0.0, 0.003627839 - Q.soft7_z) / 5.80783e-06   # -3.2%  e2_sq > 0.009606 and soft7_z < 0.003628
        + 0.03191046 * max(0.0, Q.e2_sq - 0.007872294) * max(0.0, 0.003627839 - Q.soft7_z) / 6.713918e-06   # +3.2%  e2_sq > 0.007872 and soft7_z < 0.003628
        - 0.03018201 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # -3.0%  tau1 < 0.1073
        - 0.02543999 * max(0.0, Q.sum_pt_top50 - 1024.689) * max(0.0, 5.8505e-08 - Q.e4) / 2.280575e-06   # -2.5%  sum_pt_top50 > 1025 and e4 < 5.851e-08
        + 0.02441713 * max(0.0, Q.e2_sq - 0.009606007) / 0.003182984   # +2.4%  e2_sq > 0.009606
        + 0.0215354 * max(0.0, Q.sum_pt_top30 - 911.9328) / 96.43087   # +2.2%  sum_pt_top30 > 911.9
        + 0.02116733 * max(0.0, Q.n_pt_above_1 - 26.0) / 16.44623   # +2.1%  n_pt_above_1 > 26
        + 0.01827544 * max(0.0, Q.log_sum_pt - 6.935549) / 0.0371538   # +1.8%  log_sum_pt > 6.936
        + 0.01699048 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # +1.7%  girth < 0.07374
        - 0.0165575 * max(0.0, 0.01563836 - Q.girth2_top15) / 0.009593305   # -1.7%  girth2_top15 < 0.01564
        + 0.01541551 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # +1.5%  n_dr_0p2_0p4 < 8
        - 0.01458269 * max(0.0, Q.sum_pt_top50 - 1048.098) / 31.96699   # -1.5%  sum_pt_top50 > 1048
        - 0.01346854 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -1.3%  log_sum_pt > 6.92
        + 0.01342104 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +1.3%  sum_pt > 907.9
        + 0.01279486 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # +1.3%  e2 < 0.03876
        - 0.01233607 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -1.2%  girth2_top30 < 0.01216
        + 0.01220641 * max(0.0, Q.sum_pt_top50 - 1024.689) / 40.56456   # +1.2%  sum_pt_top50 > 1025
        - 0.01215742 * max(0.0, 0.008031209 - Q.girth2_top20) / 0.00309849   # -1.2%  girth2_top20 < 0.008031
        - 0.01163465 * max(0.0, Q.girth2 - 0.02928196) * max(0.0, 0.002741632 - Q.soft3_z) / 2.118876e-07   # -1.2%  girth2 > 0.02928 and soft3_z < 0.002742
        - 0.01145006 * max(0.0, 0.3572263 - Q.N2) / 0.07385171   # -1.1%  N2 < 0.3572
        - 0.01133713 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.410481 - Q.D2) / 13.17726   # -1.1%  n_particles < 64 and D2 < 2.41
        - 0.01055882 * max(0.0, Q.girth - 0.1564779) / 0.0006617163   # -1.1%  girth > 0.1565
        + 0.01038016 * max(0.0, Q.n_for_90pct - 11.0) / 10.33888   # +1.0%  n_for_90pct > 11
        - 0.01014838 * max(0.0, Q.z_top30_slots - 0.9203881) / 0.04411614   # -1.0%  z_top30_slots > 0.9204
        + 0.009480199 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # +0.9%  log_sum_pt > 6.959
        + 0.008594881 * max(0.0, Q.sum_pt_top20 - 1017.778) / 17.66072   # +0.9%  sum_pt_top20 > 1018
        - 0.007786063 * max(0.0, Q.n_pt_above_1 - 26.0) * max(0.0, 30.0 - Q.n_dr_0_0p05) / 283.4356   # -0.8%  n_pt_above_1 > 26 and n_dr_0_0p05 < 30
        + 0.007708578 * max(0.0, Q.tau2 - 0.02164722) / 0.0156989   # +0.8%  tau2 > 0.02165
        - 0.007578105 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -0.8%  sum_pt_top5 > 531.2
        + 0.006116826 * max(0.0, 579.875 - Q.sum_pt_top5) / 65.87113   # +0.6%  sum_pt_top5 < 579.9
        - 0.005804382 * max(0.0, Q.sum_pt_top15 - 951.1375) / 24.48977   # -0.6%  sum_pt_top15 > 951.1
        - 0.005060861 * max(0.0, Q.pt_entropy - 2.903111) / 0.1962448   # -0.5%  pt_entropy > 2.903
        + 0.003153499 * max(0.0, Q.girth2 - 0.02928196) / 0.0002029538   # +0.3%  girth2 > 0.02928
        - 0.002683451 * max(0.0, Q.sum_pt_top15 - 967.7705) / 19.94058   # -0.3%  sum_pt_top15 > 967.8
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 16.81;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.81291 * (0.05622506
        + 0.1748836 * max(0.0, 0.02580859 - Q.e2_sq) / 0.01698103   # +17.5%  e2_sq < 0.02581
        - 0.1588294 * max(0.0, 0.009614971 - Q.width) / 0.003502545   # -15.9%  width < 0.009615
        - 0.08133743 * max(0.0, 0.02497133 - Q.girth2_top40) / 0.01655385   # -8.1%  girth2_top40 < 0.02497
        - 0.07070799 * max(0.0, 0.01397874 - Q.girth2) / 0.006902497   # -7.1%  girth2 < 0.01398
        - 0.05993333 * max(0.0, Q.e2 - 0.01541561) / 0.01690634   # -6.0%  e2 > 0.01542
        + 0.03993452 * max(0.0, Q.psi_0p3 - 0.9853273) / 0.009408723   # +4.0%  psi_0p3 > 0.9853
        - 0.0342647 * max(0.0, 0.02265114 - Q.girth2_top20) / 0.01537233   # -3.4%  girth2_top20 < 0.02265
        + 0.03358995 * max(0.0, 0.003811746 - Q.lam1) / 0.0006796118   # +3.4%  lam1 < 0.003812
        + 0.03337796 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +3.3%  girth2_top40 < 0.008841
        - 0.03121775 * max(0.0, 0.01397874 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.9853273) / 7.658135e-05   # -3.1%  girth2 < 0.01398 and psi_0p3 > 0.9853
        - 0.02462449 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -2.5%  girth2_top20 < 0.01083
        + 0.02338488 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +2.3%  lam1 < 0.007671
        + 0.02183712 * max(0.0, Q.LHA - 0.2091025) / 0.07391154   # +2.2%  LHA > 0.2091
        + 0.02143625 * max(0.0, Q.log_sum_pt - 6.811175) / 0.1383141   # +2.1%  log_sum_pt > 6.811
        + 0.01524284 * Q.n_dr_0p1_0p2 / 12.56613   # +1.5%  n_dr_0p1_0p2
        - 0.01494435 * max(0.0, 0.02580859 - Q.e2_sq) * max(0.0, 6.98945 - Q.log_sum_pt) / 0.001023242   # -1.5%  e2_sq < 0.02581 and log_sum_pt < 6.989
        + 0.01460578 * max(0.0, 0.008190222 - Q.width) / 0.002454223   # +1.5%  width < 0.00819
        + 0.01415582 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +1.4%  lam2 < 0.001776
        - 0.01185535 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, 0.3384815 - Q.N2) / 1.669221e-05   # -1.2%  e3 < 0.0003372 and N2 < 0.3385
        + 0.01172611 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # +1.2%  LHA < 0.2601
        + 0.01164212 * max(0.0, Q.sum_pt_top20 - 695.3094) / 236.2671   # +1.2%  sum_pt_top20 > 695.3
        + 0.01120164 * max(0.0, 0.007511864 - Q.e2_sq) / 0.002016671   # +1.1%  e2_sq < 0.007512
        + 0.00999425 * max(0.0, 0.007511864 - Q.e2_sq) * max(0.0, 1007.788 - Q.sum_pt) / 0.02913759   # +1.0%  e2_sq < 0.007512 and sum_pt < 1008
        - 0.00997495 * max(0.0, Q.girth2_top30 - 0.008376291) / 0.003092781   # -1.0%  girth2_top30 > 0.008376
        - 0.009567126 * max(0.0, Q.lam2 - 0.0009731947) / 0.0007968622   # -1.0%  lam2 > 0.0009732
        + 0.008443282 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +0.8%  z_dr_0p1_0p2 < 0.1203
        + 0.007748552 * max(0.0, 0.00287991 - Q.girth2_top20) / 0.000559956   # +0.8%  girth2_top20 < 0.00288
        + 0.007149912 * max(0.0, Q.sum_pt_top20 - 695.3094) * max(0.0, 1.36316 - Q.D2_b2) / 96.35963   # +0.7%  sum_pt_top20 > 695.3 and D2_b2 < 1.363
        + 0.007029513 * max(0.0, Q.e2 - 0.04755309) * max(0.0, 0.3289237 - Q.z_dr_0_0p05) / 0.0005973978   # +0.7%  e2 > 0.04755 and z_dr_0_0p05 < 0.3289
        - 0.005577159 * max(0.0, Q.lam1 - 0.006189818) / 0.003311577   # -0.6%  lam1 > 0.00619
        - 0.004819779 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, 0.001592178 - Q.girth2_top3) / 7.904451e-05   # -0.5%  log_sum_pt > 6.811 and girth2_top3 < 0.001592
        + 0.004453076 * max(0.0, Q.e2 - 0.04755309) * max(0.0, 26.14062 - Q.pt_14) / 0.01767661   # +0.4%  e2 > 0.04755 and pt_14 < 26.14
        - 0.004128938 * max(0.0, 0.003811746 - Q.lam1) * max(0.0, 1008.935 - Q.sum_pt_top50) / 0.0115984   # -0.4%  lam1 < 0.003812 and sum_pt_top50 < 1009
        + 0.002590144 * max(0.0, Q.z_top20_slots - 0.7111557) * max(0.0, Q.girth2_top3 - 0.01002369) / 0.0001325706   # +0.3%  z_top20_slots > 0.7112 and girth2_top3 > 0.01002
        + 0.002030838 * max(0.0, Q.e2 - 0.04755309) * max(0.0, Q.C2_b2 - 0.02704832) / 1.720373e-05   # +0.2%  e2 > 0.04755 and C2_b2 > 0.02705
        - 0.001759164 * max(0.0, Q.z_top20_slots - 0.7111557) * max(0.0, Q.dr01 - 0.1662967) / 0.001099659   # -0.2%  z_top20_slots > 0.7112 and dr01 > 0.1663
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 32.43;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 32.42852 * (-0.02321625
        + 0.1180796 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # +11.8%  e2_sq < 0.009606
        - 0.09168658 * max(0.0, 0.009606007 - Q.e2_sq) * max(0.0, 1167.447 - Q.sum_pt) / 0.4400479   # -9.2%  e2_sq < 0.009606 and sum_pt < 1167
        + 0.0699861 * max(0.0, 0.01397874 - Q.width) / 0.006902497   # +7.0%  width < 0.01398
        - 0.06703143 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # -6.7%  girth2_top50 < 0.008125
        + 0.06598108 * max(0.0, 0.007820315 - Q.girth2_top50) / 0.002264874   # +6.6%  girth2_top50 < 0.00782
        - 0.06483484 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # -6.5%  girth2 < 0.007877
        - 0.05152131 * max(0.0, 0.08589404 - Q.girth) / 0.02745918   # -5.2%  girth < 0.08589
        - 0.04750101 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -4.8%  lam1 < 0.008242
        + 0.03945166 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # +3.9%  girth2_top30 < 0.01216
        - 0.03942888 * max(0.0, 0.00634935 - Q.girth2_top50) / 0.001425305   # -3.9%  girth2_top50 < 0.006349
        + 0.03630692 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +3.6%  lam1 < 0.007671
        + 0.03086866 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # +3.1%  LHA < 0.3024
        - 0.0289635 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # -2.9%  LHA < 0.2454
        + 0.02859016 * max(0.0, 0.05660088 - Q.girth) / 0.01080382   # +2.9%  girth < 0.0566
        - 0.02677573 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # -2.7%  girth2_top30 < 0.008376
        - 0.02303594 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -2.3%  lam1 < 0.005914
        - 0.017924 * max(0.0, 0.006936725 - Q.e2_sq) / 0.001688614   # -1.8%  e2_sq < 0.006937
        + 0.01789517 * max(0.0, 0.005913555 - Q.lam1) * max(0.0, 1167.447 - Q.sum_pt) / 0.1821423   # +1.8%  lam1 < 0.005914 and sum_pt < 1167
        - 0.01272106 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -1.3%  psi_0p2 > 0.9087
        + 0.01195214 * max(0.0, 0.08589404 - Q.girth) * max(0.0, 1156.659 - Q.sum_pt_top50) / 3.29885   # +1.2%  girth < 0.08589 and sum_pt_top50 < 1157
        + 0.01170381 * max(0.0, 0.002396991 - Q.lam2) * max(0.0, 7.139296 - Q.log_sum_pt) / 0.0002814467   # +1.2%  lam2 < 0.002397 and log_sum_pt < 7.139
        - 0.01081054 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -1.1%  max_dr > 0.2405
        + 0.008689549 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # +0.9%  girth2_top30 < 0.006364
        - 0.007309886 * max(0.0, 0.04828819 - Q.tau2) / 0.02103241   # -0.7%  tau2 < 0.04829
        - 0.006315186 * max(0.0, 0.002396991 - Q.lam2) / 0.001466253   # -0.6%  lam2 < 0.002397
        + 0.005564218 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # +0.6%  psi_0p3 > 0.9966
        - 0.005244405 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # -0.5%  LHA < 0.2601
        + 0.005140258 * max(0.0, 0.2352054 - Q.tau21_b2) / 0.05222759   # +0.5%  tau21_b2 < 0.2352
        + 0.005007425 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +0.5%  tau1 < 0.07709
        + 0.004800444 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +0.5%  n_dr_0p2_0p4 < 15
        - 0.004781542 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007872294 - Q.e2_sq) / 5.103177e-05   # -0.5%  tau21_b2 < 0.2352 and e2_sq < 0.007872
        + 0.004630947 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +0.5%  e2 < 0.03481
        + 0.0043192 * max(0.0, 0.05048381 - Q.girth) / 0.008533025   # +0.4%  girth < 0.05048
        - 0.004182736 * max(0.0, 0.00375223 - Q.girth2_top30) / 0.0006714094   # -0.4%  girth2_top30 < 0.003752
        + 0.00399406 * max(0.0, Q.psi_0p3 - 0.9966167) * max(0.0, 2.275391 - Q.soft1_pt) / 0.002319433   # +0.4%  psi_0p3 > 0.9966 and soft1_pt < 2.275
        + 0.003245333 * max(0.0, Q.max_dr - 0.2982) / 0.06837446   # +0.3%  max_dr > 0.2982
        - 0.003163335 * max(0.0, 0.002396991 - Q.lam2) * max(0.0, Q.zdr_0 - 0.002524869) / 9.492656e-06   # -0.3%  lam2 < 0.002397 and zdr_0 > 0.002525
        + 0.002409752 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sum_pt_top30 - 978.0762) / 2.738836   # +0.2%  tau21_b2 < 0.2352 and sum_pt_top30 > 978.1
        + 0.002229014 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.01396296 - Q.e2_sq) / 0.000313768   # +0.2%  tau21_b2 < 0.2352 and e2_sq < 0.01396
        - 0.001889789 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.001696484   # -0.2%  tau21_b2 < 0.2352 and sj2_dr > 0.1938
        - 0.001632029 * max(0.0, 0.0009793444 - Q.mean_eta2) / 0.000113516   # -0.2%  mean_eta2 < 0.0009793
        - 0.001474543 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, 0.0361727 - Q.dr_0) / 4.440919e-06   # -0.1%  psi_0p3 > 0.998 and dr_0 < 0.03617
        + 0.0004881461 * max(0.0, Q.sum_pt_top15 - 1082.548) * max(0.0, Q.mean_phi - -3.355129e-05) / 0.0003894346   # +0.0%  sum_pt_top15 > 1083 and mean_phi > -3.355e-05
        - 0.000438041 * max(0.0, Q.sum_pt_top15 - 1082.548) * max(0.0, Q.mean_phi - 2.298159e-05) / 0.0002044402   # -0.0%  sum_pt_top15 > 1083 and mean_phi > 2.298e-05
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 27.76;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.76248 * (0.0375401
        - 0.1910447 * max(0.0, 0.0258669 - Q.girth2) * max(0.0, 7.139296 - Q.log_sum_pt) / 0.003279142   # -19.1%  girth2 < 0.02587 and log_sum_pt < 7.139
        + 0.1738026 * max(0.0, 0.0258669 - Q.girth2) / 0.01702764   # +17.4%  girth2 < 0.02587
        + 0.08388866 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, 1260.541 - Q.sum_pt) / 0.4807996   # +8.4%  girth2_top30 < 0.007464 and sum_pt < 1261
        - 0.06866165 * max(0.0, 0.01397874 - Q.girth2) / 0.006902497   # -6.9%  girth2 < 0.01398
        - 0.05846667 * max(0.0, 0.007463985 - Q.girth2_top30) / 0.00233708   # -5.8%  girth2_top30 < 0.007464
        - 0.05526158 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -5.5%  e2_sq < 0.009606
        + 0.0521391 * max(0.0, 0.1219132 - Q.tau1) / 0.04578087   # +5.2%  tau1 < 0.1219
        + 0.04797746 * max(0.0, 1245.697 - Q.sum_pt_top50) / 217.4702   # +4.8%  sum_pt_top50 < 1246
        - 0.04210234 * max(0.0, 0.008190222 - Q.width) / 0.002454223   # -4.2%  width < 0.00819
        + 0.03691266 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +3.7%  sum_pt_top50 < 1157
        - 0.02450147 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # -2.5%  n_dr_0p2_0p4 < 18
        - 0.02047806 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, 1095.686 - Q.sum_pt_top40) / 0.172292   # -2.0%  girth2_top30 < 0.007464 and sum_pt_top40 < 1096
        + 0.01985649 * max(0.0, 0.01351431 - Q.girth2_top50) / 0.006620975   # +2.0%  girth2_top50 < 0.01351
        + 0.01921721 * max(0.0, 1028.184 - Q.sum_pt) / 26.95739   # +1.9%  sum_pt < 1028
        - 0.01892782 * max(0.0, Q.girth2_top15 - 0.005788041) / 0.003469237   # -1.9%  girth2_top15 > 0.005788
        + 0.01705711 * max(0.0, Q.girth2_top15 - 0.007887677) / 0.002755219   # +1.7%  girth2_top15 > 0.007888
        + 0.01606375 * max(0.0, 0.003638856 - Q.girth2) / 0.0004964601   # +1.6%  girth2 < 0.003639
        - 0.01468384 * max(0.0, 0.0258669 - Q.girth2) * max(0.0, Q.z_top50_slots - 0.9704436) / 0.000437459   # -1.5%  girth2 < 0.02587 and z_top50_slots > 0.9704
        + 0.009212382 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +0.9%  girth2_top40 < 0.008841
        + 0.006216447 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sj2_zsoft - 0.1181474) / 1.531301   # +0.6%  n_dr_0p2_0p4 < 18 and sj2_zsoft > 0.1181
        - 0.006090797 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # -0.6%  sum_pt_top40 < 1007
        + 0.005710546 * max(0.0, 0.01397874 - Q.girth2) * max(0.0, 986.0565 - Q.sum_pt) / 0.05983431   # +0.6%  girth2 < 0.01398 and sum_pt < 986.1
        - 0.004573028 * max(0.0, 0.002574843 - Q.e2_sq) / 0.0002582574   # -0.5%  e2_sq < 0.002575
        - 0.00446706 * max(0.0, 0.1219132 - Q.tau1) * max(0.0, 0.5364935 - Q.planar_flow) / 0.004630037   # -0.4%  tau1 < 0.1219 and planar_flow < 0.5365
        + 0.001828171 * max(0.0, Q.sum_pt_top20 - 1017.778) / 17.66072   # +0.2%  sum_pt_top20 > 1018
        - 0.0008584563 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.1%  log_sum_pt < 6.811
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 6.439;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.439035 * (-0.1100656
        + 0.1339883 * max(0.0, 0.006941794 - Q.width) / 0.001690424   # +13.4%  width < 0.006942
        + 0.115617 * max(0.0, 1191.938 - Q.sum_pt_top30) / 206.0686   # +11.6%  sum_pt_top30 < 1192
        + 0.1100848 * max(0.0, Q.lam1 - 0.004673423) / 0.004174908   # +11.0%  lam1 > 0.004673
        + 0.09087422 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +9.1%  lam1 < 0.007259
        - 0.07687026 * max(0.0, 0.02675364 - Q.girth2_top15) / 0.01959938   # -7.7%  girth2_top15 < 0.02675
        + 0.0475223 * max(0.0, 0.005712208 - Q.girth2_top40) / 0.001219686   # +4.8%  girth2_top40 < 0.005712
        - 0.04477063 * max(0.0, Q.lam1 - 0.006716737) / 0.003087986   # -4.5%  lam1 > 0.006717
        - 0.04459795 * max(0.0, Q.girth2 - 0.0258669) / 0.0004653552   # -4.5%  girth2 > 0.02587
        - 0.0367463 * max(0.0, 0.007259287 - Q.lam1) * max(0.0, Q.sum_pt - 1085.125) / 0.08770494   # -3.7%  lam1 < 0.007259 and sum_pt > 1085
        - 0.03492654 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -3.5%  girth2_top20 < 0.01083
        - 0.02972706 * max(0.0, Q.e2_sq - 0.01989454) / 0.001200608   # -3.0%  e2_sq > 0.01989
        + 0.02287225 * max(0.0, 0.005712208 - Q.girth2_top40) * max(0.0, Q.log_sum_pt - 7.017258) / 3.653263e-05   # +2.3%  girth2_top40 < 0.005712 and log_sum_pt > 7.017
        + 0.0225408 * max(0.0, 972.4111 - Q.sum_pt_top40) / 17.57725   # +2.3%  sum_pt_top40 < 972.4
        - 0.02157841 * max(0.0, 0.004169954 - Q.girth2_top15) / 0.001130716   # -2.2%  girth2_top15 < 0.00417
        + 0.01814157 * max(0.0, Q.e2_sq - 0.0292152) / 0.0002016952   # +1.8%  e2_sq > 0.02922
        - 0.01776158 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # -1.8%  tau1 < 0.06311
        - 0.01578401 * max(0.0, Q.girth - 0.1207452) / 0.004285542   # -1.6%  girth > 0.1207
        - 0.01392334 * max(0.0, 972.4111 - Q.sum_pt_top40) * max(0.0, 0.0369869 - Q.tau3) / 0.1284345   # -1.4%  sum_pt_top40 < 972.4 and tau3 < 0.03699
        - 0.01295595 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -1.3%  z_dr_0_0p05 > 0.8789
        - 0.01123728 * max(0.0, Q.e2_sq - 0.0292152) * max(0.0, 3.779492 - Q.soft7_pt) / 0.0003078674   # -1.1%  e2_sq > 0.02922 and soft7_pt < 3.779
        + 0.01078256 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # +1.1%  LHA > 0.3332
        + 0.01065806 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +1.1%  sum_pt < 986.1
        + 0.008880198 * max(0.0, 6.903423 - Q.log_sum_pt) * max(0.0, 0.4479367 - Q.z_dr_0p1_0p2) / 0.004178024   # +0.9%  log_sum_pt < 6.903 and z_dr_0p1_0p2 < 0.4479
        + 0.008815205 * max(0.0, Q.LHA - 0.3332345) * max(0.0, 1042.609 - Q.sum_pt) / 0.8026572   # +0.9%  LHA > 0.3332 and sum_pt < 1043
        - 0.008046238 * max(0.0, 0.005712208 - Q.girth2_top40) * max(0.0, 1007.788 - Q.sum_pt) / 0.0192656   # -0.8%  girth2_top40 < 0.005712 and sum_pt < 1008
        + 0.007344032 * max(0.0, Q.e3 - 0.0005178279) / 8.375623e-06   # +0.7%  e3 > 0.0005178
        + 0.006157885 * max(0.0, 0.004169954 - Q.girth2_top15) * max(0.0, Q.n_dr_0p2_0p4 - 8.0) / 0.001650576   # +0.6%  girth2_top15 < 0.00417 and n_dr_0p2_0p4 > 8
        - 0.005707883 * max(0.0, 972.4111 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.106445) / 11.84488   # -0.6%  sum_pt_top40 < 972.4 and soft4_pt > 1.106
        + 0.005604302 * max(0.0, Q.girth - 0.1207452) * max(0.0, 0.4606099 - Q.sj3_pairmin_over_m) / 0.0004267265   # +0.6%  girth > 0.1207 and sj3_pairmin_over_m < 0.4606
        + 0.005483128 * max(0.0, Q.z_dr_0_0p05 - 0.878906) * max(0.0, Q.sum_pt_top40 - 1013.042) / 0.9125551   # +0.5%  z_dr_0_0p05 > 0.8789 and sum_pt_top40 > 1013
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 14.88;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.87786 * (0.1761655
        - 0.1969553 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -19.7%  girth < 0.1207
        + 0.1038444 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # +10.4%  tau1 < 0.1507
        - 0.06820622 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -6.8%  e2_sq < 0.009606
        + 0.06252826 * max(0.0, Q.e2 - 0.006720044) / 0.02422374   # +6.3%  e2 > 0.00672
        + 0.05628026 * max(0.0, 0.3719813 - Q.LHA) / 0.1170757   # +5.6%  LHA < 0.372
        + 0.0502308 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +5.0%  e2_sq < 0.008184
        - 0.04318976 * max(0.0, Q.girth2_top20 - 0.005718442) / 0.003749064   # -4.3%  girth2_top20 > 0.005718
        - 0.03516103 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # -3.5%  e3 < 0.0005178
        + 0.03273179 * max(0.0, Q.girth2_top20 - 0.008031209) / 0.002906852   # +3.3%  girth2_top20 > 0.008031
        + 0.03260738 * max(0.0, 0.006403325 - Q.width) / 0.001406912   # +3.3%  width < 0.006403
        - 0.02759418 * max(0.0, 0.01976735 - Q.girth2_top10) / 0.01371247   # -2.8%  girth2_top10 < 0.01977
        - 0.02682514 * max(0.0, Q.sum_pt_top3 - 309.625) / 165.6552   # -2.7%  sum_pt_top3 > 309.6
        - 0.02453497 * max(0.0, 0.03577037 - Q.girth) / 0.004182456   # -2.5%  girth < 0.03577
        - 0.02327859 * max(0.0, Q.girth2 - 0.01397874) / 0.002228371   # -2.3%  girth2 > 0.01398
        + 0.02242757 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +2.2%  z_dr_0p2_0p4 < 0.09123
        + 0.0179805 * max(0.0, 0.08222447 - Q.M2) / 0.02103373   # +1.8%  M2 < 0.08222
        + 0.01758665 * max(0.0, 0.00592208 - Q.e2_sq) / 0.001192334   # +1.8%  e2_sq < 0.005922
        - 0.01729684 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # -1.7%  girth2_top10 < 0.007679
        - 0.01675531 * max(0.0, 0.0259543 - Q.tau4) / 0.009964748   # -1.7%  tau4 < 0.02595
        - 0.01369558 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -1.4%  n_dr_0p2_0p4 < 15
        - 0.01350708 * max(0.0, 0.3062621 - Q.tau21_b2) / 0.08794306   # -1.4%  tau21_b2 < 0.3063
        + 0.01222557 * max(0.0, 2.410481 - Q.D2) / 0.587301   # +1.2%  D2 < 2.41
        + 0.01077933 * max(0.0, 656.3844 - Q.sum_pt_top3) / 209.0067   # +1.1%  sum_pt_top3 < 656.4
        + 0.009856531 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +1.0%  dr_0 < 0.06413
        - 0.009830961 * max(0.0, Q.e2 - 0.006720044) * max(0.0, Q.log_sum_pt - 6.893714) / 0.001240306   # -1.0%  e2 > 0.00672 and log_sum_pt > 6.894
        - 0.009495577 * max(0.0, Q.e2_sq - 0.02580859) / 0.0004635665   # -0.9%  e2_sq > 0.02581
        - 0.007003756 * max(0.0, 0.08222447 - Q.M2) * max(0.0, 0.4357228 - Q.max_dr) / 0.001990074   # -0.7%  M2 < 0.08222 and max_dr < 0.4357
        - 0.006560506 * max(0.0, Q.z_dr_0_0p05 - 0.7128619) / 0.07526795   # -0.7%  z_dr_0_0p05 > 0.7129
        + 0.006103755 * max(0.0, Q.girth2 - 0.01397874) * max(0.0, 114.75 - Q.pt_3) / 0.1232701   # +0.6%  girth2 > 0.01398 and pt_3 < 114.8
        - 0.005933162 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # -0.6%  girth2_top30 < 0.005402
        - 0.005675774 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, Q.pt_6 - 33.6875) / 0.01241674   # -0.6%  girth2_top30 < 0.005402 and pt_6 > 33.69
        + 0.005277635 * max(0.0, Q.e2 - 0.006720044) * max(0.0, Q.sj3_pairmin_over_m - 0.1765064) / 0.002982204   # +0.5%  e2 > 0.00672 and sj3_pairmin_over_m > 0.1765
        - 0.004313094 * max(0.0, 0.002247756 - Q.girth2_top10) / 0.0005836006   # -0.4%  girth2_top10 < 0.002248
        + 0.003726678 * max(0.0, Q.girth2_top40 - 0.02497133) / 0.0004755075   # +0.4%  girth2_top40 > 0.02497
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 12.9;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.9004 * (0.06123479
        + 0.1361909 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +13.6%  e2_sq < 0.008184
        - 0.08625086 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -8.6%  girth2_top30 < 0.006364
        + 0.08505268 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # +8.5%  girth2_top30 < 0.01216
        - 0.06495182 * max(0.0, 0.1008497 - Q.tau1) / 0.02962944   # -6.5%  tau1 < 0.1008
        + 0.06259755 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +6.3%  LHA < 0.3332
        - 0.05279424 * max(0.0, 0.00818374 - Q.e2_sq) * max(0.0, 1115.723 - Q.sum_pt) / 0.1997244   # -5.3%  e2_sq < 0.008184 and sum_pt < 1116
        - 0.04460192 * max(0.0, 0.3098384 - Q.LHA) / 0.06737135   # -4.5%  LHA < 0.3098
        - 0.03978826 * max(0.0, Q.n_real_top50 - 22.0) / 19.58685   # -4.0%  n_real_top50 > 22
        + 0.03798102 * max(0.0, 0.1072713 - Q.tau1) / 0.03428119   # +3.8%  tau1 < 0.1073
        + 0.03564349 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +3.6%  n_dr_0p2_0p4 < 10
        - 0.03456595 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -3.5%  lam1 < 0.005914
        + 0.03188525 * max(0.0, 0.005913555 - Q.lam1) * max(0.0, 1115.723 - Q.sum_pt) / 0.1207895   # +3.2%  lam1 < 0.005914 and sum_pt < 1116
        + 0.0312944 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +3.1%  e2 < 0.02516
        - 0.02949186 * max(0.0, 0.00634935 - Q.girth2_top50) / 0.001425305   # -2.9%  girth2_top50 < 0.006349
        - 0.02590372 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # -2.6%  e2 < 0.03876
        - 0.02413377 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # -2.4%  girth2_top5 < 0.007164
        - 0.02345799 * max(0.0, Q.psi_0p2 - 0.9313699) / 0.04058308   # -2.3%  psi_0p2 > 0.9314
        - 0.02012639 * max(0.0, 0.09749958 - Q.girth) / 0.03657082   # -2.0%  girth < 0.0975
        + 0.01897296 * max(0.0, 0.005809485 - Q.girth2_top30) / 0.001402136   # +1.9%  girth2_top30 < 0.005809
        + 0.01591351 * max(0.0, 0.004754578 - Q.e2_sq) / 0.0007999118   # +1.6%  e2_sq < 0.004755
        + 0.014876 * max(0.0, 0.00327978 - Q.girth2_top5) / 0.001239029   # +1.5%  girth2_top5 < 0.00328
        - 0.01430165 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 1156.659 - Q.sum_pt_top50) / 469.6702   # -1.4%  n_dr_0p2_0p4 < 10 and sum_pt_top50 < 1157
        - 0.01342895 * max(0.0, 4.450169 - Q.D2) / 2.013555   # -1.3%  D2 < 4.45
        + 0.01135725 * max(0.0, 0.009606007 - Q.e2_sq) * max(0.0, 1115.723 - Q.sum_pt) / 0.2875932   # +1.1%  e2_sq < 0.009606 and sum_pt < 1116
        - 0.01081299 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2) / 0.005195774   # -1.1%  n_dr_0p2_0p4 < 10 and girth2 < 0.005532
        - 0.009734977 * max(0.0, Q.z_top30_slots - 0.9734886) / 0.01006171   # -1.0%  z_top30_slots > 0.9735
        + 0.00880712 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 43.23581   # +0.9%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21
        - 0.008769269 * max(0.0, Q.C2_b2 - 0.002289486) / 0.01267025   # -0.9%  C2_b2 > 0.002289
        + 0.006313228 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 1.08774 - Q.D2_b2) / 0.003455048   # +0.6%  z_top30_slots > 0.9735 and D2_b2 < 1.088
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 8.178;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.177686 * (-0.117075
        + 0.2485241 * max(0.0, 0.006403325 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.811175) / 0.0002246937   # +24.9%  girth2 < 0.006403 and log_sum_pt > 6.811
        - 0.2371403 * max(0.0, 0.007511864 - Q.e2_sq) * max(0.0, Q.log_sum_pt - 6.811175) / 0.0003183593   # -23.7%  e2_sq < 0.007512 and log_sum_pt > 6.811
        + 0.1702551 * max(0.0, 0.007511864 - Q.e2_sq) / 0.002016671   # +17.0%  e2_sq < 0.007512
        + 0.06396812 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +6.4%  psi_0p3 > 0.9897
        - 0.06202174 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -6.2%  psi_0p3 > 0.9943
        - 0.04675551 * max(0.0, 0.002752094 - Q.lam1) / 0.0003912817   # -4.7%  lam1 < 0.002752
        - 0.04374948 * max(0.0, 0.005809485 - Q.girth2_top30) / 0.001402136   # -4.4%  girth2_top30 < 0.005809
        - 0.03449206 * max(0.0, Q.sum_pt - 1085.125) / 25.7173   # -3.4%  sum_pt > 1085
        + 0.03138193 * max(0.0, Q.psi_0p3 - 0.9943058) * max(0.0, 0.00751625 - Q.girth2) / 6.462915e-06   # +3.1%  psi_0p3 > 0.9943 and girth2 < 0.007516
        + 0.02861715 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +2.9%  n_dr_0p2_0p4 < 10
        - 0.01046341 * max(0.0, Q.psi_0p1 - 0.9371031) * max(0.0, 0.05444336 - Q.absphi_0) / 0.0004737047   # -1.0%  psi_0p1 > 0.9371 and absphi_0 < 0.05444
        + 0.006542164 * max(0.0, 6.879399 - Q.log_sum_pt) / 0.01083309   # +0.7%  log_sum_pt < 6.879
        + 0.006156316 * max(0.0, 0.08133662 - Q.psi_0p1) * max(0.0, Q.sum_pt_top50 - 889.8503) / 0.2667141   # +0.6%  psi_0p1 < 0.08134 and sum_pt_top50 > 889.9
        - 0.003401677 * max(0.0, 0.08133662 - Q.psi_0p1) / 0.002484547   # -0.3%  psi_0p1 < 0.08134
        + 0.003287832 * max(0.0, Q.sum_pt - 1085.125) * max(0.0, 0.001868041 - Q.lam1) / 0.006895658   # +0.3%  sum_pt > 1085 and lam1 < 0.001868
        - 0.003243095 * max(0.0, Q.sum_pt - 1085.125) * max(0.0, 0.019523 - Q.z_dr_0p2_0p4) / 0.1925692   # -0.3%  sum_pt > 1085 and z_dr_0p2_0p4 < 0.01952
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 17.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.00727 * (0.04858537
        + 0.2241365 * max(0.0, Q.log_sum_pt - 6.811175) / 0.1383141   # +22.4%  log_sum_pt > 6.811
        - 0.06829952 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # -6.8%  sum_pt > 1017
        - 0.06677312 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -6.7%  sum_pt < 1085
        + 0.06187752 * max(0.0, Q.sum_pt_top50 - 976.277) / 72.29277   # +6.2%  sum_pt_top50 > 976.3
        - 0.05781408 * max(0.0, Q.sum_pt_top50 - 889.8503) / 149.655   # -5.8%  sum_pt_top50 > 889.9
        + 0.04911432 * max(0.0, Q.girth2_top40 - 0.01292642) * max(0.0, 1245.697 - Q.sum_pt_top50) / 0.6025464   # +4.9%  girth2_top40 > 0.01293 and sum_pt_top50 < 1246
        + 0.04901626 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # +4.9%  sum_pt_top50 < 1079
        - 0.04867714 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # -4.9%  log_sum_pt > 6.894
        - 0.04130013 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -4.1%  girth < 0.1207
        - 0.03392894 * max(0.0, Q.sum_pt_top40 - 972.4111) / 67.15108   # -3.4%  sum_pt_top40 > 972.4
        - 0.03221399 * max(0.0, Q.girth2_top40 - 0.008840538) / 0.003161074   # -3.2%  girth2_top40 > 0.008841
        - 0.03217385 * max(0.0, Q.girth2_top50 - 0.01351431) / 0.002240755   # -3.2%  girth2_top50 > 0.01351
        - 0.02564689 * max(0.0, Q.girth2_top40 - 0.01292642) / 0.002259475   # -2.6%  girth2_top40 > 0.01293
        + 0.02499265 * max(0.0, Q.girth2_top50 - 0.005852839) / 0.004485718   # +2.5%  girth2_top50 > 0.005853
        + 0.02408098 * max(0.0, Q.girth2_top40 - 0.02862386) / 0.0001970429   # +2.4%  girth2_top40 > 0.02862
        + 0.02382 * max(0.0, Q.sum_pt_top50 - 1013.916) / 45.94021   # +2.4%  sum_pt_top50 > 1014
        - 0.01784058 * max(0.0, 54.0 - Q.n_particles) / 11.024   # -1.8%  n_particles < 54
        - 0.01612913 * max(0.0, Q.girth2_top40 - 0.02862386) * max(0.0, Q.pt_6 - 19.46875) / 0.003416206   # -1.6%  girth2_top40 > 0.02862 and pt_6 > 19.47
        + 0.01545306 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +1.5%  e2 < 0.03481
        - 0.01210223 * max(0.0, Q.e2_sq - 0.0292152) / 0.0002016952   # -1.2%  e2_sq > 0.02922
        - 0.01195134 * max(0.0, Q.width - 0.01991191) / 0.001207633   # -1.2%  width > 0.01991
        - 0.01000256 * max(0.0, Q.girth2_top40 - 0.02862386) * max(0.0, 62.25 - Q.pt_6) / 0.00503292   # -1.0%  girth2_top40 > 0.02862 and pt_6 < 62.25
        - 0.009402756 * max(0.0, Q.girth2_top40 - 0.02862386) * max(0.0, Q.z_6 - 0.01906139) / 3.587165e-06   # -0.9%  girth2_top40 > 0.02862 and z_6 > 0.01906
        + 0.008746924 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # +0.9%  psi_0p3 > 0.998
        + 0.008746538 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, Q.tau21 - 0.1295048) / 22.44785   # +0.9%  sum_pt < 1085 and tau21 > 0.1295
        - 0.006445372 * max(0.0, 795.6047 - Q.sum_pt_top15) / 30.63834   # -0.6%  sum_pt_top15 < 795.6
        + 0.006364299 * max(0.0, Q.e2 - 0.03680582) / 0.004603798   # +0.6%  e2 > 0.03681
        + 0.005909221 * max(0.0, Q.sum_pt_top10 - 867.9266) / 28.75627   # +0.6%  sum_pt_top10 > 867.9
        - 0.003738416 * max(0.0, 25.0 - Q.n_pt_above_5) / 2.617361   # -0.4%  n_pt_above_5 < 25
        - 0.003301728 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, Q.C2 - 0.06655881) / 0.001737527   # -0.3%  log_sum_pt > 6.811 and C2 > 0.06656
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 26.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.59112 * (-0.0006310009
        - 0.08632745 * max(0.0, 0.001784491 - Q.girth2) / 0.0001206749   # -8.6%  girth2 < 0.001784
        + 0.07844398 * max(0.0, 0.01989454 - Q.e2_sq) / 0.01180402   # +7.8%  e2_sq < 0.01989
        - 0.06416683 * max(0.0, 0.008190222 - Q.girth2) / 0.002454223   # -6.4%  girth2 < 0.00819
        + 0.05979887 * max(0.0, 0.007820315 - Q.girth2_top50) / 0.002264874   # +6.0%  girth2_top50 < 0.00782
        - 0.05627586 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # -5.6%  girth2_top50 < 0.008125
        + 0.05534606 * max(0.0, 0.01396296 - Q.e2_sq) / 0.00689253   # +5.5%  e2_sq < 0.01396
        + 0.04399632 * max(0.0, 0.01215787 - Q.girth2_top30) * max(0.0, Q.sum_pt - 907.9372) / 0.9148907   # +4.4%  girth2_top30 < 0.01216 and sum_pt > 907.9
        - 0.03952073 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -4.0%  girth2_top20 < 0.01083
        - 0.03632341 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -3.6%  girth2_top30 < 0.01216
        - 0.03572644 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -3.6%  e2_sq < 0.006167
        - 0.0349514 * max(0.0, 0.01292642 - Q.girth2_top40) / 0.006292908   # -3.5%  girth2_top40 < 0.01293
        - 0.03161856 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # -3.2%  girth2_top30 < 0.008376
        + 0.03134921 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +3.1%  psi_0p3 > 0.9956
        + 0.0310399 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # +3.1%  tau1 < 0.05445
        - 0.03061999 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, 1225.842 - Q.sum_pt_top40) / 0.3755165   # -3.1%  psi_0p3 > 0.9956 and sum_pt_top40 < 1226
        + 0.03047248 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +3.0%  psi_0p3 > 0.9897
        - 0.0299604 * max(0.0, Q.e2_sq - 0.00363788) / 0.006149555   # -3.0%  e2_sq > 0.003638
        + 0.0262358 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +2.6%  e2 < 0.03481
        - 0.01952807 * max(0.0, 0.03480688 - Q.e2) * max(0.0, Q.sum_pt_top30 - 886.3438) / 1.374127   # -2.0%  e2 < 0.03481 and sum_pt_top30 > 886.3
        - 0.01836081 * max(0.0, 0.008190222 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.9956185) / 5.477284e-06   # -1.8%  girth2 < 0.00819 and psi_0p3 > 0.9956
        - 0.01822948 * max(0.0, 0.1008497 - Q.tau1) / 0.02962944   # -1.8%  tau1 < 0.1008
        - 0.01678409 * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 9.82015   # -1.7%  n_dr_0p1_0p2 < 21
        + 0.01464781 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # +1.5%  tau1 < 0.07057
        - 0.01367705 * max(0.0, 0.2091025 - Q.LHA) / 0.02095782   # -1.4%  LHA < 0.2091
        + 0.01266764 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # +1.3%  e2_sq < 0.009606
        + 0.009002761 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +0.9%  tau21_b2 < 0.3425
        + 0.00894943 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # +0.9%  girth2_top20 < 0.006374
        - 0.007630135 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.02048524 - Q.lam1) / 0.001304679   # -0.8%  tau21_b2 < 0.3425 and lam1 < 0.02049
        + 0.00760822 * max(0.0, 0.1632346 - Q.LHA) / 0.009624216   # +0.8%  LHA < 0.1632
        - 0.007210746 * max(0.0, Q.sum_pt - 1260.541) / 7.655483   # -0.7%  sum_pt > 1261
        + 0.007170963 * max(0.0, Q.sum_pt_top50 - 1245.697) / 7.470315   # +0.7%  sum_pt_top50 > 1246
        - 0.006474007 * max(0.0, 0.1361635 - Q.z_2nd) / 0.02521041   # -0.6%  z_2nd < 0.1362
        - 0.005764121 * max(0.0, 0.3572263 - Q.N2) / 0.07385171   # -0.6%  N2 < 0.3572
        + 0.00555495 * max(0.0, Q.sj2_dr - 0.1666442) / 0.04983259   # +0.6%  sj2_dr > 0.1666
        - 0.004178251 * max(0.0, 28.0 - Q.n_pt_above_5) / 4.053793   # -0.4%  n_pt_above_5 < 28
        + 0.004000694 * max(0.0, Q.LHA - 0.3098384) / 0.0195892   # +0.4%  LHA > 0.3098
        - 0.002910668 * max(0.0, Q.sj2_dr - 0.2412757) / 0.01833576   # -0.3%  sj2_dr > 0.2413
        + 0.002336131 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.007709916 - Q.girth2_top40) / 0.0001062984   # +0.2%  tau21_b2 < 0.3425 and girth2_top40 < 0.00771
        - 0.001791659 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.2%  log_sum_pt > 7.063
        + 0.001314181 * max(0.0, 0.01396296 - Q.e2_sq) * max(0.0, 0.9906378 - Q.z_top50_slots) / 1.282821e-05   # +0.1%  e2_sq < 0.01396 and z_top50_slots < 0.9906
        + 0.001143989 * max(0.0, 0.3572263 - Q.N2) * max(0.0, Q.dr0_7 - 0.1786203) / 0.001295123   # +0.1%  N2 < 0.3572 and dr0_7 > 0.1786
        - 0.000890457 * max(0.0, 0.001595561 - Q.soft7_z) / 0.0003533115   # -0.1%  soft7_z < 0.001596
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 9.954;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.954019 * (-0.04861697
        - 0.1195989 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -12.0%  girth2_top30 < 0.01216
        - 0.0737671 * max(0.0, 0.2941033 - Q.LHA) / 0.05718399   # -7.4%  LHA < 0.2941
        + 0.06684502 * max(0.0, 0.08068193 - Q.girth) / 0.02368455   # +6.7%  girth < 0.08068
        + 0.06383927 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # +6.4%  girth2_top30 < 0.008376
        + 0.05932382 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # +5.9%  e2 < 0.04083
        + 0.0552917 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +5.5%  girth2_top5 < 0.00833
        - 0.0443823 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # -4.4%  girth2_top30 < 0.005402
        + 0.03848775 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # +3.8%  girth < 0.1207
        + 0.03471548 * max(0.0, 6.97212 - Q.log_sum_pt) / 0.05249646   # +3.5%  log_sum_pt < 6.972
        - 0.03290143 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -3.3%  tau1 < 0.07057
        + 0.03100951 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +3.1%  sj2_dr > 0.2232
        + 0.02771299 * max(0.0, Q.sum_pt_top50 - 1156.659) / 13.71464   # +2.8%  sum_pt_top50 > 1157
        + 0.02739984 * max(0.0, 0.006189818 - Q.lam1) / 0.001608778   # +2.7%  lam1 < 0.00619
        - 0.02396515 * max(0.0, Q.sj2_dr - 0.1745007) / 0.04529905   # -2.4%  sj2_dr > 0.1745
        - 0.02333156 * max(0.0, 1002.379 - Q.sum_pt) / 15.95735   # -2.3%  sum_pt < 1002
        + 0.02259413 * max(0.0, 0.1778185 - Q.sd_rg) * max(0.0, Q.soft5_dr0 - 0.03721986) / 0.009661509   # +2.3%  sd_rg < 0.1778 and soft5_dr0 > 0.03722
        + 0.02190352 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +2.2%  z_dr_0p1_0p2 < 0.1203
        - 0.02170807 * max(0.0, 0.1778185 - Q.sd_rg) * max(0.0, Q.soft5_dr - 0.04139378) / 0.009084795   # -2.2%  sd_rg < 0.1778 and soft5_dr > 0.04139
        + 0.02017495 * max(0.0, 24.0 - Q.n_pt_above_10) / 5.296457   # +2.0%  n_pt_above_10 < 24
        - 0.01941694 * max(0.0, Q.sum_pt_top40 - 1095.686) / 18.59291   # -1.9%  sum_pt_top40 > 1096
        + 0.01802753 * max(0.0, 1.976207 - Q.D2) / 0.367701   # +1.8%  D2 < 1.976
        - 0.01789255 * max(0.0, 0.02647293 - Q.z_dr_0p2_0p4) / 0.01168948   # -1.8%  z_dr_0p2_0p4 < 0.02647
        + 0.01640644 * max(0.0, 0.006805717 - Q.girth2_top50) / 0.001665679   # +1.6%  girth2_top50 < 0.006806
        + 0.01521396 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 0.002396991 - Q.lam2) / 8.322782e-05   # +1.5%  z_dr_0p1_0p2 < 0.1203 and lam2 < 0.002397
        - 0.01499369 * max(0.0, Q.sum_pt_top50 - 1003.544) / 52.17558   # -1.5%  sum_pt_top50 > 1004
        - 0.01473685 * max(0.0, 0.9860575 - Q.z_top30_slots) / 0.03906901   # -1.5%  z_top30_slots < 0.9861
        - 0.01370825 * max(0.0, 0.002270363 - Q.girth2_top5) / 0.0007634099   # -1.4%  girth2_top5 < 0.00227
        - 0.01198938 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -1.2%  psi_0p1 > 0.8976
        + 0.01189049 * max(0.0, Q.sum_pt_top50 - 1107.226) / 19.65456   # +1.2%  sum_pt_top50 > 1107
        - 0.009056612 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # -0.9%  log_sum_pt > 7.139
        - 0.008337791 * max(0.0, Q.sum_pt_top30 - 988.4375) / 43.28413   # -0.8%  sum_pt_top30 > 988.4
        - 0.0080254 * max(0.0, 1.976207 - Q.D2) * max(0.0, Q.sj3_z2 - 0.08173536) / 0.08456063   # -0.8%  D2 < 1.976 and sj3_z2 > 0.08174
        - 0.005871846 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 10.0) / 0.006625563   # -0.6%  girth2_top5 < 0.00833 and n_dr_0p2_0p4 > 10
        - 0.005479772 * max(0.0, Q.sj2_dr - 0.2780918) / 0.009227968   # -0.5%  sj2_dr > 0.2781
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6330575105042017, 2.867836344537815, 0.2505174369747899, 0.43909627100840337, 0.7394627100840336, 1.0585211134453782, 0.5012654411764705, 0.4101699579831933, 1.1667653361344539, 1.1024703781512606, 1.1582085084033613, 0.8071648109243698, 0.5580409663865546, 1.5086200105042016, 0.25908476890756305, 0.36367552521008406]
T = [3.9316709279805675, 2.5144500279017854, 4.213204922203257, 4.214351207983193, 3.9283801519826675]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +46%, n4 -16%, n9 +14%, n5 -8%, n3 -8%, n12 +3% ...
            + 0.455887 * h[1] / H_AVG[1]
            - 0.1645687 * h[4] / H_AVG[4]
            + 0.1445852 * h[9] / H_AVG[9]
            - 0.08413416 * h[5] / H_AVG[5]
            - 0.08376139 * h[3] / H_AVG[3]
            + 0.03326597 * h[12] / H_AVG[12]
            + 0.01992098 * h[6] / H_AVG[6]
            + 0.009273771 * h[8] / H_AVG[8]
            - 0.00460288 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -31%, n9 +25%, n1 -21%, n11 -8%, n12 +8%, n6 +4% ...
            - 0.3124656 * h[4] / H_AVG[4]
            + 0.2466303 * h[9] / H_AVG[9]
            - 0.2138517 * h[1] / H_AVG[1]
            - 0.08025262 * h[11] / H_AVG[11]
            + 0.07628968 * h[12] / H_AVG[12]
            + 0.04360867 * h[6] / H_AVG[6]
            + 0.01245389 * h[2] / H_AVG[2]
            + 0.007250376 * h[8] / H_AVG[8]
            - 0.007197203 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +15%, n11 +11%, n0 +11%, n14 -8%, n7 -6% ...
            - 0.2423143 * h[8] / H_AVG[8]
            + 0.1452475 * h[5] / H_AVG[5]
            + 0.1137505 * h[11] / H_AVG[11]
            + 0.1126917 * h[0] / H_AVG[0]
            - 0.08455358 * h[14] / H_AVG[14]
            - 0.06084589 * h[7] / H_AVG[7]
            + 0.06033182 * h[4] / H_AVG[4]
            - 0.05724037 * h[9] / H_AVG[9]
            - 0.053808 * h[12] / H_AVG[12]
            + 0.04559584 * h[3] / H_AVG[3]
            - 0.01618463 * h[15] / H_AVG[15]
            + 0.007435928 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -21%, n5 +17%, n6 -12%, n7 +9%, n15 +5% ...
            - 0.2595518 * h[8] / H_AVG[8]
            - 0.2065452 * h[0] / H_AVG[0]
            + 0.1726798 * h[5] / H_AVG[5]
            - 0.1152255 * h[6] / H_AVG[6]
            + 0.08820255 * h[7] / H_AVG[7]
            + 0.04854068 * h[15] / H_AVG[15]
            + 0.04558344 * h[3] / H_AVG[3]
            + 0.04137951 * h[12] / H_AVG[12]
            - 0.02229146 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -35%, n10 +29%, n5 -13%, n8 +6%, n12 -5%, n4 +4% ...
            - 0.3480282 * h[13] / H_AVG[13]
            + 0.2902243 * h[10] / H_AVG[10]
            - 0.126307 * h[5] / H_AVG[5]
            + 0.06265039 * h[8] / H_AVG[8]
            - 0.05327014 * h[12] / H_AVG[12]
            + 0.03529426 * h[4] / H_AVG[4]
            - 0.03471617 * h[15] / H_AVG[15]
            - 0.02936587 * h[7] / H_AVG[7]
            + 0.02014372 * h[0] / H_AVG[0]
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
