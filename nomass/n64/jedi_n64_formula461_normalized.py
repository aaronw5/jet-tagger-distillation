"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.0%   (on for 44% of jets)
  neuron  1:  12.4%   (on for 95% of jets)
  neuron  5:  11.3%   (on for 84% of jets)
  neuron  4:  10.1%   (on for 66% of jets)
  neuron  9:   7.7%   (on for 72% of jets)
  neuron  0:   7.7%   (on for 61% of jets)
  neuron 13:   7.1%   (on for 84% of jets)
  neuron 10:   6.0%   (on for 76% of jets)
  neuron 12:   5.0%   (on for 39% of jets)
  neuron  7:   3.9%   (on for 33% of jets)
  neuron  3:   3.8%   (on for 64% of jets)
  neuron 11:   3.7%   (on for 78% of jets)
  neuron  6:   3.5%   (on for 48% of jets)
  neuron 15:   2.3%   (on for 68% of jets)
  neuron 14:   2.0%   (on for 26% of jets)
  neuron  2:   0.7%   (on for 49% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.2% (the network: 81.1%); same class as the network for 93.4% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
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
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_14                  pT of particle 14 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_pt              pT [GeV] of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_1                    pT of particle 1 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft10_z               pT share of the 10. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft5_dr0              ΔR between the hardest and the 5. softest real particle (0 if among the 15 hardest)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.soft5_dr               ΔR from the jet axis of the 5. softest real particle (0 if it is among the 15 hardest)
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
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
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
  Q.tau32                  N-subjettiness τ3/τ2
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
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
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_90pct=ncum(0.9),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_11=pt[11],
        pt_14=pt[14],
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_6=pt[6],
        soft1_pt=softp(1, 'pt'),
        soft10_pt=softp(10, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft9_pt=softp(9, 'pt'),
        z_1=z[1],
        z_6=z[6],
        soft1_z=softp(1, 'z'),
        soft10_z=softp(10, 'z'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        soft5_dr0=softp(5, 'dr0'),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        soft5_dr=softp(5, 'dr'),
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
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
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
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
    )


def neuron_0(Q):
    # scale S = 10.21;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.21262 * (-0.1253756
        + 0.1512783 * max(0.0, 0.007872294 - Q.e2_sq) / 0.002240658   # +15.1%  e2_sq < 0.007872
        + 0.08979497 * max(0.0, 0.006936725 - Q.e2_sq) / 0.001688614   # +9.0%  e2_sq < 0.006937
        + 0.08745113 * max(0.0, Q.sum_pt_top40 - 858.8262) / 166.7848   # +8.7%  sum_pt_top40 > 858.8
        - 0.07585761 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -7.6%  e2_sq < 0.009606
        - 0.0693853 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -6.9%  e2_sq < 0.006167
        - 0.06457454 * max(0.0, Q.sum_pt - 1002.379) / 57.37467   # -6.5%  sum_pt > 1002
        + 0.0460927 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # +4.6%  girth2_top50 < 0.008125
        + 0.03939404 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +3.9%  girth2_top40 < 0.008841
        - 0.03789765 * max(0.0, 0.005852839 - Q.girth2_top50) * max(0.0, 1260.541 - Q.sum_pt) / 0.2467156   # -3.8%  girth2_top50 < 0.005853 and sum_pt < 1261
        + 0.03456512 * max(0.0, 0.008190222 - Q.girth2) / 0.002454223   # +3.5%  girth2 < 0.00819
        + 0.03436232 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # +3.4%  n_dr_0p2_0p4 < 18
        + 0.03363544 * max(0.0, 1191.938 - Q.sum_pt_top30) / 206.0686   # +3.4%  sum_pt_top30 < 1192
        - 0.02875385 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -2.9%  girth2_top30 < 0.006364
        - 0.025141 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.5%  tau1 < 0.07057
        - 0.02444202 * max(0.0, Q.psi_0p2 - 0.9087063) / 0.05823225   # -2.4%  psi_0p2 > 0.9087
        - 0.02311164 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -2.3%  lam1 < 0.005914
        + 0.02261548 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # +2.3%  psi_0p3 > 0.9943
        + 0.02147738 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # +2.1%  girth2_top30 < 0.008376
        - 0.02074447 * max(0.0, 0.008190222 - Q.girth2) * max(0.0, 0.006142802 - Q.girth2_top15) / 1.08234e-05   # -2.1%  girth2 < 0.00819 and girth2_top15 < 0.006143
        - 0.01712143 * max(0.0, 0.003113918 - Q.girth2_top40) / 0.0004182435   # -1.7%  girth2_top40 < 0.003114
        - 0.01289261 * max(0.0, 1073.473 - Q.sum_pt_top30) / 97.85231   # -1.3%  sum_pt_top30 < 1073
        + 0.01240598 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # +1.2%  sum_pt > 1116
        - 0.01038818 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -1.0%  sum_pt_top40 > 1025
        - 0.007265718 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.log_sum_pt - 6.915514) / 0.4823065   # -0.7%  n_dr_0p2_0p4 < 18 and log_sum_pt > 6.916
        + 0.006495513 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +0.6%  e2 < 0.01879
        - 0.00285566 * max(0.0, Q.sum_pt - 1167.447) / 14.21193   # -0.3%  sum_pt > 1167
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 14.91;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.9122 * (0.07315077
        + 0.1238774 * max(0.0, Q.sum_pt - 972.0419) / 81.34075   # +12.4%  sum_pt > 972
        - 0.1180991 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -11.8%  log_sum_pt < 7.139
        + 0.1143479 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +11.4%  log_sum_pt > 6.91
        - 0.09642924 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # -9.6%  sum_pt_top50 > 934.2
        + 0.06069777 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # +6.1%  sum_pt_top50 < 1079
        + 0.05427087 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +5.4%  n_particles > 38
        + 0.05238303 * max(0.0, 1225.842 - Q.sum_pt_top40) / 211.117   # +5.2%  sum_pt_top40 < 1226
        - 0.03347453 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -3.3%  log_sum_pt > 6.989
        - 0.02904266 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -2.9%  sum_pt > 1053
        - 0.02578576 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.002181998 - Q.soft1_z) / 0.01445558   # -2.6%  n_particles > 38 and soft1_z < 0.002182
        + 0.02393229 * max(0.0, 605.875 - Q.sum_pt_top2) / 245.9717   # +2.4%  sum_pt_top2 < 605.9
        + 0.02371678 * max(0.0, Q.z_top30_slots - 0.9564984) / 0.01947087   # +2.4%  z_top30_slots > 0.9565
        + 0.02251937 * max(0.0, 0.007463985 - Q.girth2_top30) / 0.00233708   # +2.3%  girth2_top30 < 0.007464
        - 0.02164265 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -2.2%  e2_sq < 0.009606
        + 0.01882766 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # +1.9%  LHA < 0.2454
        - 0.01627716 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, 10.17512 - Q.ptdr0_3) / 0.1427049   # -1.6%  z_top30_slots > 0.9565 and ptdr0_3 < 10.18
        - 0.01322664 * max(0.0, 0.02210818 - Q.e2) / 0.00342102   # -1.3%  e2 < 0.02211
        - 0.01193527 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # -1.2%  psi_0p3 > 0.9966
        - 0.01168067 * max(0.0, 0.0007143144 - Q.lam2) / 0.0002017367   # -1.2%  lam2 < 0.0007143
        - 0.01129851 * max(0.0, 0.03457336 - Q.M3) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.0006031662   # -1.1%  M3 < 0.03457 and psi_0p3 > 0.9299
        - 0.01017788 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -1.0%  n_dr_0p2_0p4 < 7
        - 0.0100614 * max(0.0, 29.04688 - Q.pt_11) / 8.607263   # -1.0%  pt_11 < 29.05
        - 0.009455828 * max(0.0, Q.z_top5 - 0.6551948) * max(0.0, 28.39396 - Q.pt1_dr01) / 0.8332473   # -0.9%  z_top5 > 0.6552 and pt1_dr01 < 28.39
        + 0.008943939 * max(0.0, Q.n_dr_0_0p05 - 15.0) / 2.713461   # +0.9%  n_dr_0_0p05 > 15
        + 0.008101271 * max(0.0, 2.5 - Q.soft9_pt) / 0.7352391   # +0.8%  soft9_pt < 2.5
        + 0.007758657 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.009970338 - Q.zdr_0) / 0.03317992   # +0.8%  n_particles > 38 and zdr_0 < 0.00997
        + 0.00667843 * max(0.0, 0.004763596 - Q.girth2_top30) / 0.0009958273   # +0.7%  girth2_top30 < 0.004764
        + 0.006489457 * max(0.0, 0.0006570502 - Q.girth2_top5) / 0.0001395474   # +0.6%  girth2_top5 < 0.0006571
        + 0.00627608 * max(0.0, 0.002197765 - Q.girth2_top15) / 0.0004509993   # +0.6%  girth2_top15 < 0.002198
        - 0.005800222 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) * max(0.0, 10.0 - Q.n_dr_0p05_0p1) / 0.2065903   # -0.6%  z_dr_0_0p05 > 0.8103 and n_dr_0p05_0p1 < 10
        - 0.005428465 * max(0.0, 7.0 - Q.n_dr_0p1_0p2) / 0.9740454   # -0.5%  n_dr_0p1_0p2 < 7
        + 0.005324727 * max(0.0, Q.n_particles - 38.0) * max(0.0, Q.tau32 - 0.3293142) / 3.693221   # +0.5%  n_particles > 38 and tau32 > 0.3293
        + 0.00464183 * max(0.0, 0.2140276 - Q.tau21) / 0.01160104   # +0.5%  tau21 < 0.214
        - 0.004173544 * max(0.0, Q.tau1 - 0.1219132) / 0.01033631   # -0.4%  tau1 > 0.1219
        + 0.00363881 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.pt1_dr01 - 5.351077) / 0.1255531   # +0.4%  z_top30_slots > 0.9342 and pt1_dr01 > 5.351
        + 0.00351506 * max(0.0, 0.001153063 - Q.girth2_top20) / 0.0001200862   # +0.4%  girth2_top20 < 0.001153
        - 0.003148229 * max(0.0, 0.001784491 - Q.girth2) / 0.0001206749   # -0.3%  girth2 < 0.001784
        - 0.002789769 * max(0.0, Q.zdr_0 - 0.01240028) / 0.002210186   # -0.3%  zdr_0 > 0.0124
        + 0.002764455 * max(0.0, 0.002197765 - Q.girth2_top15) * max(0.0, Q.psi_0p3 - 0.9966167) / 5.451807e-07   # +0.3%  girth2_top15 < 0.002198 and psi_0p3 > 0.9966
        - 0.001366703 * max(0.0, Q.n_dr_0_0p05 - 30.0) / 0.212479   # -0.1%  n_dr_0_0p05 > 30
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 6.432;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.432343 * (-0.09902195
        - 0.1953311 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # -19.5%  LHA > 0.372
        - 0.1249361 * max(0.0, 0.006941794 - Q.girth2) / 0.001690424   # -12.5%  girth2 < 0.006942
        + 0.08772254 * max(0.0, 0.00634935 - Q.girth2_top50) / 0.001425305   # +8.8%  girth2_top50 < 0.006349
        + 0.05947218 * max(0.0, 0.01174405 - Q.lam1) / 0.005678958   # +5.9%  lam1 < 0.01174
        + 0.05655282 * max(0.0, 0.01563836 - Q.girth2_top15) / 0.009593305   # +5.7%  girth2_top15 < 0.01564
        + 0.05540845 * max(0.0, Q.sum_pt - 1007.788) / 53.71957   # +5.5%  sum_pt > 1008
        + 0.05444312 * max(0.0, Q.log_sum_pt - 6.930088) / 0.0397436   # +5.4%  log_sum_pt > 6.93
        - 0.04948646 * max(0.0, Q.sum_pt_top30 - 800.732) / 195.5077   # -4.9%  sum_pt_top30 > 800.7
        - 0.0451763 * max(0.0, Q.sum_pt_top50 - 1003.544) / 52.17558   # -4.5%  sum_pt_top50 > 1004
        - 0.03899125 * max(0.0, Q.log_sum_pt - 7.062574) * max(0.0, Q.girth2_top30 - 0.01215787) / 1.24419e-05   # -3.9%  log_sum_pt > 7.063 and girth2_top30 > 0.01216
        + 0.03526376 * max(0.0, Q.z_top30_slots - 0.9203881) / 0.04411614   # +3.5%  z_top30_slots > 0.9204
        + 0.02996521 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +3.0%  sum_pt_top40 > 1070
        + 0.02815495 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15) / 0.0009227558   # +2.8%  log_sum_pt > 6.903 and girth2_top15 < 0.02147
        + 0.02240065 * max(0.0, Q.sum_pt_top50 - 1003.544) * max(0.0, Q.z_dr_0p1_0p2 - 0.003353111) / 6.148304   # +2.2%  sum_pt_top50 > 1004 and z_dr_0p1_0p2 > 0.003353
        - 0.02187535 * max(0.0, Q.sum_pt_top30 - 1011.524) / 32.30985   # -2.2%  sum_pt_top30 > 1012
        - 0.02174679 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.8747961 - Q.psi_0p1) / 0.005466701   # -2.2%  log_sum_pt > 6.903 and psi_0p1 < 0.8748
        - 0.0191067 * max(0.0, Q.sum_pt - 1085.125) / 25.7173   # -1.9%  sum_pt > 1085
        - 0.01881984 * max(0.0, 0.003638856 - Q.girth2) / 0.0004964601   # -1.9%  girth2 < 0.003639
        - 0.01440343 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # -1.4%  log_sum_pt > 7.017
        - 0.01394513 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -1.4%  log_sum_pt > 7.063
        + 0.004785589 * max(0.0, Q.sum_pt_top30 - 1111.245) * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.9443349   # +0.5%  sum_pt_top30 > 1111 and z_dr_0p1_0p2 < 0.1203
        - 0.001355741 * max(0.0, Q.sum_pt_top15 - 1003.329) * max(0.0, Q.lam2 - 0.001163277) / 0.001034672   # -0.1%  sum_pt_top15 > 1003 and lam2 > 0.001163
        - 0.0006565795 * max(0.0, Q.sum_pt_top15 - 1003.329) * max(0.0, Q.eta_1 - 0.08734131) / 0.006699938   # -0.1%  sum_pt_top15 > 1003 and eta_1 > 0.08734
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 3.778;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.77846 * (0.1281977
        - 0.141132 * max(0.0, 0.3719813 - Q.LHA) / 0.1170757   # -14.1%  LHA < 0.372
        + 0.1062764 * max(0.0, 0.007511864 - Q.e2_sq) / 0.002016671   # +10.6%  e2_sq < 0.007512
        - 0.09174084 * max(0.0, 25.0 - Q.n_dr_0_0p05) / 13.07341   # -9.2%  n_dr_0_0p05 < 25
        + 0.08905623 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +8.9%  n_particles < 46
        + 0.08797834 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +8.8%  girth2_top40 < 0.008841
        - 0.07367129 * max(0.0, 0.347196 - Q.tau21) / 0.05239131   # -7.4%  tau21 < 0.3472
        - 0.06396108 * max(0.0, 0.006259772 - Q.girth2_top40) / 0.001461823   # -6.4%  girth2_top40 < 0.00626
        + 0.06355244 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104091   # +6.4%  n_dr_0p2_0p4 < 5
        + 0.05358544 * max(0.0, 0.2018786 - Q.tau21_b2) / 0.03799024   # +5.4%  tau21_b2 < 0.2019
        - 0.0476597 * max(0.0, 0.02647293 - Q.z_dr_0p2_0p4) / 0.01168948   # -4.8%  z_dr_0p2_0p4 < 0.02647
        + 0.03912251 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.985099) / 0.03403522   # +3.9%  n_dr_0p2_0p4 < 8 and z_top50_slots > 0.9851
        - 0.03794094 * max(0.0, 0.009614971 - Q.girth2) / 0.003502545   # -3.8%  girth2 < 0.009615
        - 0.03697381 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.e2 - 0.01036127) / 0.09655312   # -3.7%  n_particles < 46 and e2 > 0.01036
        + 0.02253429 * max(0.0, 1.788105 - Q.D2) / 0.2865857   # +2.3%  D2 < 1.788
        + 0.02200579 * max(0.0, 40.0 - Q.n_pt_above_1) / 5.337714   # +2.2%  n_pt_above_1 < 40
        - 0.01708931 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03029714 - Q.e2) / 0.006086253   # -1.7%  n_dr_0p2_0p4 < 5 and e2 < 0.0303
        - 0.005719555 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 988.4375 - Q.sum_pt_top30) / 17.46027   # -0.6%  n_dr_0p2_0p4 < 5 and sum_pt_top30 < 988.4
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 11.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.06793 * (0.028139
        + 0.08503256 * max(0.0, 0.009614971 - Q.width) / 0.003502545   # +8.5%  width < 0.009615
        + 0.08334149 * max(0.0, Q.girth2_top50 - 0.004573744) / 0.00533432   # +8.3%  girth2_top50 > 0.004574
        - 0.06753565 * max(0.0, Q.girth2_top50 - 0.00242543) / 0.006946092   # -6.8%  girth2_top50 > 0.002425
        + 0.06370132 * max(0.0, Q.girth2_top15 - 0.002197765) / 0.005632841   # +6.4%  girth2_top15 > 0.002198
        + 0.0580864 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # +5.8%  girth2_top20 < 0.01083
        - 0.05773174 * max(0.0, 0.04358622 - Q.e2) / 0.01548732   # -5.8%  e2 < 0.04359
        - 0.05676258 * max(0.0, 0.008031986 - Q.girth2_top40) / 0.002514593   # -5.7%  girth2_top40 < 0.008032
        - 0.05159513 * max(0.0, 1061.184 - Q.sum_pt_top50) / 53.8984   # -5.2%  sum_pt_top50 < 1061
        - 0.04421397 * max(0.0, 0.007538019 - Q.girth2_top20) / 0.002740841   # -4.4%  girth2_top20 < 0.007538
        - 0.03861408 * max(0.0, 0.00592208 - Q.e2_sq) / 0.001192334   # -3.9%  e2_sq < 0.005922
        + 0.03835129 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +3.8%  sum_pt_top40 < 1053
        + 0.03272082 * max(0.0, 0.03263075 - Q.e2) / 0.008034085   # +3.3%  e2 < 0.03263
        - 0.0298948 * max(0.0, Q.LHA - 0.2454112) / 0.05012361   # -3.0%  LHA > 0.2454
        - 0.02869162 * max(0.0, Q.n_particles - 29.0) / 17.61555   # -2.9%  n_particles > 29
        - 0.02591221 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -2.6%  lam1 < 0.006717
        - 0.02355873 * max(0.0, Q.girth2_top15 - 0.004169954) / 0.004340368   # -2.4%  girth2_top15 > 0.00417
        + 0.0228261 * max(0.0, Q.n_particles - 29.0) * max(0.0, 2.275391 - Q.soft1_pt) / 26.03979   # +2.3%  n_particles > 29 and soft1_pt < 2.275
        + 0.02256384 * max(0.0, 0.01397874 - Q.girth2) / 0.006902497   # +2.3%  girth2 < 0.01398
        + 0.02104557 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +2.1%  n_dr_0p2_0p4 < 26
        + 0.02068738 * max(0.0, Q.LHA - 0.1870291) / 0.08997036   # +2.1%  LHA > 0.187
        - 0.01921813 * max(0.0, Q.girth2_top15 - 0.007887677) / 0.002755219   # -1.9%  girth2_top15 > 0.007888
        - 0.01479857 * max(0.0, Q.soft10_z - 0.0007540821) / 0.001847761   # -1.5%  soft10_z > 0.0007541
        - 0.0137873 * max(0.0, Q.n_particles - 29.0) * max(0.0, Q.eccentricity - 0.3346888) / 7.265231   # -1.4%  n_particles > 29 and eccentricity > 0.3347
        + 0.01361963 * max(0.0, Q.girth - 0.076787) / 0.01351025   # +1.4%  girth > 0.07679
        + 0.01117752 * max(0.0, Q.soft10_pt - 1.603516) / 1.270676   # +1.1%  soft10_pt > 1.604
        - 0.009477634 * max(0.0, Q.LHA - 0.3332345) / 0.01384158   # -0.9%  LHA > 0.3332
        + 0.009436547 * max(0.0, Q.n_dr_0_0p05 - 10.0) / 5.012778   # +0.9%  n_dr_0_0p05 > 10
        + 0.009070065 * max(0.0, Q.psi_0p3 - 0.9973959) / 0.0009545571   # +0.9%  psi_0p3 > 0.9974
        - 0.007678406 * max(0.0, 0.0005358203 - Q.lam2) / 0.0001122488   # -0.8%  lam2 < 0.0005358
        + 0.007072477 * max(0.0, 0.01397874 - Q.girth2) * max(0.0, Q.eccentricity - 0.8903081) / 0.0001316174   # +0.7%  girth2 < 0.01398 and eccentricity > 0.8903
        + 0.006736008 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.sum_pt_top40 - 858.8262) / 0.172906   # +0.7%  psi_0p3 > 0.9974 and sum_pt_top40 > 858.8
        - 0.005060431 * max(0.0, Q.sj2_dr - 0.2595052) * max(0.0, 0.0329485 - Q.C2_b2) / 0.0001569167   # -0.5%  sj2_dr > 0.2595 and C2_b2 < 0.03295
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 11.53;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.52936 * (0.02986427
        + 0.1579767 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +15.8%  sum_pt_top50 > 934.2
        - 0.1273388 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -12.7%  log_sum_pt > 6.91
        - 0.08229087 * max(0.0, Q.e2_sq - 0.007872294) / 0.00365949   # -8.2%  e2_sq > 0.007872
        + 0.07331368 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +7.3%  n_particles < 64
        + 0.05987229 * max(0.0, Q.width - 0.006941794) / 0.004053246   # +6.0%  width > 0.006942
        - 0.059731 * max(0.0, 902.4062 - Q.sum_pt_top5) / 313.0794   # -6.0%  sum_pt_top5 < 902.4
        + 0.05302198 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +5.3%  sum_pt > 907.9
        - 0.04628384 * max(0.0, 0.01563836 - Q.girth2_top15) / 0.009593305   # -4.6%  girth2_top15 < 0.01564
        + 0.03175448 * max(0.0, Q.n_pt_above_1 - 26.0) / 16.44623   # +3.2%  n_pt_above_1 > 26
        + 0.03090199 * max(0.0, Q.sum_pt_top30 - 911.9328) / 96.43087   # +3.1%  sum_pt_top30 > 911.9
        - 0.02806665 * max(0.0, Q.sum_pt_top50 - 1048.098) / 31.96699   # -2.8%  sum_pt_top50 > 1048
        + 0.02751395 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # +2.8%  log_sum_pt > 6.959
        + 0.02472393 * max(0.0, Q.n_for_90pct - 11.0) / 10.33888   # +2.5%  n_for_90pct > 11
        - 0.02342511 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -2.3%  log_sum_pt > 6.92
        - 0.02298588 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -2.3%  girth2_top30 < 0.01216
        - 0.02219976 * max(0.0, Q.z_top30_slots - 0.9203881) / 0.04411614   # -2.2%  z_top30_slots > 0.9204
        - 0.02167797 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.410481 - Q.D2) / 13.17726   # -2.2%  n_particles < 64 and D2 < 2.41
        + 0.01551187 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) / 2.583205   # +1.6%  n_dr_0p2_0p4 < 8
        - 0.01419717 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -1.4%  sum_pt_top5 > 531.2
        + 0.01379933 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4) / 7.592735e-06   # +1.4%  sum_pt > 907.9 and e4 < 5.851e-08
        - 0.01093504 * max(0.0, 0.008031209 - Q.girth2_top20) / 0.00309849   # -1.1%  girth2_top20 < 0.008031
        + 0.01081503 * max(0.0, 0.005383629 - Q.girth2_top15) / 0.001666876   # +1.1%  girth2_top15 < 0.005384
        + 0.01015377 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +1.0%  tau4 < 0.01627
        + 0.008030253 * max(0.0, 579.875 - Q.sum_pt_top5) / 65.87113   # +0.8%  sum_pt_top5 < 579.9
        - 0.007857646 * max(0.0, Q.n_pt_above_1 - 26.0) * max(0.0, 30.0 - Q.n_dr_0_0p05) / 283.4356   # -0.8%  n_pt_above_1 > 26 and n_dr_0_0p05 < 30
        + 0.007739254 * max(0.0, Q.sum_pt_top20 - 1017.778) / 17.66072   # +0.8%  sum_pt_top20 > 1018
        - 0.006075395 * max(0.0, Q.sum_pt_top15 - 967.7705) / 19.94058   # -0.6%  sum_pt_top15 > 967.8
        + 0.001806404 * max(0.0, Q.girth2 - 0.02928196) / 0.0002029538   # +0.2%  girth2 > 0.02928
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 13.76;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.76123 * (0.05139248
        - 0.1692028 * max(0.0, 0.009614971 - Q.width) / 0.003502545   # -16.9%  width < 0.009615
        + 0.1219705 * max(0.0, 0.02580859 - Q.e2_sq) / 0.01698103   # +12.2%  e2_sq < 0.02581
        - 0.08417455 * max(0.0, Q.e2 - 0.01541561) / 0.01690634   # -8.4%  e2 > 0.01542
        - 0.07538302 * max(0.0, 0.01397874 - Q.girth2) / 0.006902497   # -7.5%  girth2 < 0.01398
        - 0.04994006 * max(0.0, 0.02265114 - Q.girth2_top20) / 0.01537233   # -5.0%  girth2_top20 < 0.02265
        + 0.04856872 * max(0.0, Q.psi_0p3 - 0.9853273) / 0.009408723   # +4.9%  psi_0p3 > 0.9853
        + 0.04771798 * max(0.0, 0.003811746 - Q.lam1) / 0.0006796118   # +4.8%  lam1 < 0.003812
        - 0.04385218 * max(0.0, 0.01397874 - Q.girth2) * max(0.0, Q.psi_0p3 - 0.9853273) / 7.658135e-05   # -4.4%  girth2 < 0.01398 and psi_0p3 > 0.9853
        + 0.03544451 * max(0.0, Q.sum_pt_top20 - 695.3094) / 236.2671   # +3.5%  sum_pt_top20 > 695.3
        + 0.03450494 * max(0.0, Q.LHA - 0.2091025) / 0.07391154   # +3.5%  LHA > 0.2091
        - 0.02704731 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -2.7%  girth2_top20 < 0.01083
        + 0.02529957 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +2.5%  lam1 < 0.007671
        + 0.02303602 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +2.3%  girth2_top40 < 0.008841
        + 0.02243283 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +2.2%  e3 < 0.0003372
        - 0.0213351 * max(0.0, 0.02580859 - Q.e2_sq) * max(0.0, 6.98945 - Q.log_sum_pt) / 0.001023242   # -2.1%  e2_sq < 0.02581 and log_sum_pt < 6.989
        + 0.01960068 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +2.0%  lam2 < 0.001776
        + 0.01659862 * max(0.0, 0.008190222 - Q.width) / 0.002454223   # +1.7%  width < 0.00819
        + 0.01584084 * max(0.0, Q.log_sum_pt - 6.811175) / 0.1383141   # +1.6%  log_sum_pt > 6.811
        + 0.01240848 * max(0.0, 0.007511864 - Q.e2_sq) * max(0.0, 1007.788 - Q.sum_pt) / 0.02913759   # +1.2%  e2_sq < 0.007512 and sum_pt < 1008
        - 0.01223501 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, 0.3384815 - Q.N2) / 1.669221e-05   # -1.2%  e3 < 0.0003372 and N2 < 0.3385
        - 0.01128531 * max(0.0, Q.lam1 - 0.006189818) / 0.003311577   # -1.1%  lam1 > 0.00619
        + 0.01059418 * max(0.0, Q.e2 - 0.04755309) * max(0.0, 0.3289237 - Q.z_dr_0_0p05) / 0.0005973978   # +1.1%  e2 > 0.04755 and z_dr_0_0p05 < 0.3289
        - 0.01032177 * max(0.0, Q.lam2 - 0.0009731947) / 0.0007968622   # -1.0%  lam2 > 0.0009732
        + 0.01005756 * max(0.0, 0.00287991 - Q.girth2_top20) / 0.000559956   # +1.0%  girth2_top20 < 0.00288
        + 0.006918064 * max(0.0, 0.002412891 - Q.girth2_top2) / 0.0009487944   # +0.7%  girth2_top2 < 0.002413
        + 0.00660487 * max(0.0, 0.007511864 - Q.e2_sq) / 0.002016671   # +0.7%  e2_sq < 0.007512
        - 0.006530616 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, 0.001592178 - Q.girth2_top3) / 7.904451e-05   # -0.7%  log_sum_pt > 6.811 and girth2_top3 < 0.001592
        + 0.0056187 * max(0.0, Q.sum_pt_top20 - 695.3094) * max(0.0, 1.36316 - Q.D2_b2) / 96.35963   # +0.6%  sum_pt_top20 > 695.3 and D2_b2 < 1.363
        + 0.005506111 * max(0.0, Q.e2 - 0.04755309) * max(0.0, 26.14062 - Q.pt_14) / 0.01767661   # +0.6%  e2 > 0.04755 and pt_14 < 26.14
        - 0.004327848 * max(0.0, 0.003811746 - Q.lam1) * max(0.0, 1008.935 - Q.sum_pt_top50) / 0.0115984   # -0.4%  lam1 < 0.003812 and sum_pt_top50 < 1009
        + 0.004019956 * max(0.0, Q.girth2_top30 - 0.008376291) / 0.003092781   # +0.4%  girth2_top30 > 0.008376
        - 0.003958466 * max(0.0, Q.e2 - 0.04755309) / 0.002107078   # -0.4%  e2 > 0.04755
        + 0.002877275 * max(0.0, Q.psi_0p3 - 0.9853273) * max(0.0, Q.n_dr_0p1_0p2 - 15.0) / 0.01929542   # +0.3%  psi_0p3 > 0.9853 and n_dr_0p1_0p2 > 15
        + 0.002642285 * max(0.0, Q.z_top20_slots - 0.7111557) * max(0.0, Q.girth2_top3 - 0.01002369) / 0.0001325706   # +0.3%  z_top20_slots > 0.7112 and girth2_top3 > 0.01002
        - 0.002143259 * max(0.0, Q.log_sum_pt - 6.811175) * max(0.0, Q.dr_max_012 - 0.1686705) / 0.001990704   # -0.2%  log_sum_pt > 6.811 and dr_max_012 > 0.1687
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 31.21;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.20896 * (-0.01279806
        + 0.124274 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # +12.4%  e2_sq < 0.009606
        - 0.1036581 * max(0.0, 0.008124776 - Q.girth2_top50) / 0.002470567   # -10.4%  girth2_top50 < 0.008125
        - 0.0928503 * max(0.0, 0.009606007 - Q.e2_sq) * max(0.0, 1167.447 - Q.sum_pt) / 0.4400479   # -9.3%  e2_sq < 0.009606 and sum_pt < 1167
        + 0.0839224 * max(0.0, 0.007820315 - Q.girth2_top50) / 0.002264874   # +8.4%  girth2_top50 < 0.00782
        + 0.07961164 * max(0.0, 0.01397874 - Q.width) / 0.006902497   # +8.0%  width < 0.01398
        - 0.05796719 * max(0.0, 0.007877041 - Q.girth2) / 0.002242297   # -5.8%  girth2 < 0.007877
        - 0.05439405 * max(0.0, 0.08589404 - Q.girth) / 0.02745918   # -5.4%  girth < 0.08589
        - 0.0539144 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -5.4%  lam1 < 0.008242
        + 0.04340713 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +4.3%  lam1 < 0.007671
        + 0.03849974 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # +3.8%  LHA < 0.3024
        - 0.02884589 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # -2.9%  LHA < 0.2601
        - 0.02856206 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -2.9%  lam1 < 0.005914
        - 0.02617861 * max(0.0, 0.00634935 - Q.girth2_top50) / 0.001425305   # -2.6%  girth2_top50 < 0.006349
        - 0.02369606 * max(0.0, 0.006936725 - Q.e2_sq) / 0.001688614   # -2.4%  e2_sq < 0.006937
        + 0.01357067 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +1.4%  tau1 < 0.07709
        - 0.01247492 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -1.2%  max_dr > 0.2405
        - 0.01231854 * max(0.0, 0.04828819 - Q.tau2) / 0.02103241   # -1.2%  tau2 < 0.04829
        + 0.01204709 * max(0.0, 0.002396991 - Q.lam2) * max(0.0, 7.139296 - Q.log_sum_pt) / 0.0002814467   # +1.2%  lam2 < 0.002397 and log_sum_pt < 7.139
        + 0.0115097 * max(0.0, 0.005913555 - Q.lam1) * max(0.0, 1167.447 - Q.sum_pt) / 0.1821423   # +1.2%  lam1 < 0.005914 and sum_pt < 1167
        + 0.0112389 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +1.1%  e2 < 0.03481
        + 0.01079911 * max(0.0, 0.08589404 - Q.girth) * max(0.0, 1156.659 - Q.sum_pt_top50) / 3.29885   # +1.1%  girth < 0.08589 and sum_pt_top50 < 1157
        + 0.009824928 * max(0.0, 0.05660088 - Q.girth) / 0.01080382   # +1.0%  girth < 0.0566
        + 0.008110389 * max(0.0, 0.005913555 - Q.lam1) * max(0.0, 1139.734 - Q.sum_pt_top40) / 0.1589332   # +0.8%  lam1 < 0.005914 and sum_pt_top40 < 1140
        + 0.007639508 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # +0.8%  psi_0p3 > 0.9966
        + 0.007149572 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.01396296 - Q.e2_sq) / 0.000313768   # +0.7%  tau21_b2 < 0.2352 and e2_sq < 0.01396
        - 0.005772316 * max(0.0, 0.002396991 - Q.lam2) / 0.001466253   # -0.6%  lam2 < 0.002397
        + 0.005762721 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +0.6%  n_dr_0p2_0p4 < 15
        - 0.005292721 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007872294 - Q.e2_sq) / 5.103177e-05   # -0.5%  tau21_b2 < 0.2352 and e2_sq < 0.007872
        + 0.004145105 * max(0.0, Q.max_dr - 0.2982) / 0.06837446   # +0.4%  max_dr > 0.2982
        - 0.00405797 * max(0.0, 0.007511864 - Q.e2_sq) / 0.002016671   # -0.4%  e2_sq < 0.007512
        - 0.003768872 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 1225.842 - Q.sum_pt_top40) / 1472.634   # -0.4%  n_dr_0p2_0p4 < 15 and sum_pt_top40 < 1226
        - 0.003694759 * max(0.0, 0.002396991 - Q.lam2) * max(0.0, Q.zdr_0 - 0.002524869) / 9.492656e-06   # -0.4%  lam2 < 0.002397 and zdr_0 > 0.002525
        + 0.003396623 * max(0.0, Q.psi_0p3 - 0.9966167) * max(0.0, 2.275391 - Q.soft1_pt) / 0.002319433   # +0.3%  psi_0p3 > 0.9966 and soft1_pt < 2.275
        + 0.002586243 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sum_pt_top30 - 978.0762) / 2.738836   # +0.3%  tau21_b2 < 0.2352 and sum_pt_top30 > 978.1
        - 0.001690256 * max(0.0, 0.0009793444 - Q.mean_eta2) / 0.000113516   # -0.2%  mean_eta2 < 0.0009793
        - 0.00116698 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, Q.n_pt_above_10 - 20.0) / 0.001532328   # -0.1%  psi_0p3 > 0.998 and n_pt_above_10 > 20
        - 0.00115485 * max(0.0, Q.psi_0p3 - 0.9980008) * max(0.0, 0.0361727 - Q.dr_0) / 4.440919e-06   # -0.1%  psi_0p3 > 0.998 and dr_0 < 0.03617
        - 0.001045687 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.1937688) / 0.001696484   # -0.1%  tau21_b2 < 0.2352 and sj2_dr > 0.1938
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 27.1;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.10358 * (0.04475272
        - 0.1853939 * max(0.0, 0.0258669 - Q.girth2) * max(0.0, 7.139296 - Q.log_sum_pt) / 0.003279142   # -18.5%  girth2 < 0.02587 and log_sum_pt < 7.139
        + 0.1604089 * max(0.0, 0.0258669 - Q.girth2) / 0.01702764   # +16.0%  girth2 < 0.02587
        + 0.07780396 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, 1260.541 - Q.sum_pt) / 0.4807996   # +7.8%  girth2_top30 < 0.007464 and sum_pt < 1261
        - 0.07387693 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -7.4%  e2_sq < 0.009606
        - 0.07347127 * max(0.0, 0.01397874 - Q.girth2) / 0.006902497   # -7.3%  girth2 < 0.01398
        + 0.05654526 * max(0.0, 0.1219132 - Q.tau1) / 0.04578087   # +5.7%  tau1 < 0.1219
        - 0.04985973 * max(0.0, 0.007463985 - Q.girth2_top30) / 0.00233708   # -5.0%  girth2_top30 < 0.007464
        + 0.04762263 * max(0.0, 1245.697 - Q.sum_pt_top50) / 217.4702   # +4.8%  sum_pt_top50 < 1246
        - 0.04636132 * max(0.0, 0.008190222 - Q.width) / 0.002454223   # -4.6%  width < 0.00819
        + 0.02613957 * max(0.0, 0.01351431 - Q.girth2_top50) / 0.006620975   # +2.6%  girth2_top50 < 0.01351
        + 0.02421385 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +2.4%  sum_pt_top50 < 1157
        + 0.02083305 * max(0.0, 0.008840538 - Q.girth2_top40) / 0.003108622   # +2.1%  girth2_top40 < 0.008841
        - 0.01792193 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, 1095.686 - Q.sum_pt_top40) / 0.172292   # -1.8%  girth2_top30 < 0.007464 and sum_pt_top40 < 1096
        - 0.01756434 * max(0.0, Q.girth2_top15 - 0.005788041) / 0.003469237   # -1.8%  girth2_top15 > 0.005788
        - 0.01678647 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # -1.7%  n_dr_0p2_0p4 < 18
        + 0.01592048 * max(0.0, Q.girth2_top15 - 0.007887677) / 0.002755219   # +1.6%  girth2_top15 > 0.007888
        + 0.01458724 * max(0.0, 0.003638856 - Q.girth2) / 0.0004964601   # +1.5%  girth2 < 0.003639
        + 0.01400204 * max(0.0, 1028.184 - Q.sum_pt) / 26.95739   # +1.4%  sum_pt < 1028
        - 0.01143515 * max(0.0, 0.0258669 - Q.girth2) * max(0.0, Q.z_top50_slots - 0.9704436) / 0.000437459   # -1.1%  girth2 < 0.02587 and z_top50_slots > 0.9704
        - 0.007514392 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # -0.8%  sum_pt_top40 < 1007
        - 0.006198784 * max(0.0, 0.002574843 - Q.e2_sq) / 0.0002582574   # -0.6%  e2_sq < 0.002575
        + 0.00603733 * max(0.0, 0.01397874 - Q.girth2) * max(0.0, 986.0565 - Q.sum_pt) / 0.05983431   # +0.6%  girth2 < 0.01398 and sum_pt < 986.1
        + 0.005881243 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sj2_zsoft - 0.1181474) / 1.531301   # +0.6%  n_dr_0p2_0p4 < 18 and sj2_zsoft > 0.1181
        + 0.005643583 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # +0.6%  sum_pt < 1085
        + 0.004440289 * max(0.0, 1008.935 - Q.sum_pt_top50) / 22.0436   # +0.4%  sum_pt_top50 < 1009
        - 0.0040364 * max(0.0, 0.1219132 - Q.tau1) * max(0.0, 0.5364935 - Q.planar_flow) / 0.004630037   # -0.4%  tau1 < 0.1219 and planar_flow < 0.5365
        - 0.003390446 * max(0.0, Q.sum_pt_top20 - 926.0902) / 52.62483   # -0.3%  sum_pt_top20 > 926.1
        + 0.00289075 * max(0.0, Q.sum_pt_top20 - 1017.778) / 17.66072   # +0.3%  sum_pt_top20 > 1018
        + 0.002449238 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +0.2%  sj2_dr > 0.2232
        - 0.0007694802 * max(0.0, 6.811175 - Q.log_sum_pt) / 0.004882419   # -0.1%  log_sum_pt < 6.811
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 6.734;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.734327 * (-0.07848734
        + 0.1350806 * max(0.0, 0.006941794 - Q.width) / 0.001690424   # +13.5%  width < 0.006942
        + 0.1246725 * max(0.0, Q.lam1 - 0.004673423) / 0.004174908   # +12.5%  lam1 > 0.004673
        + 0.1210288 * max(0.0, 1191.938 - Q.sum_pt_top30) / 206.0686   # +12.1%  sum_pt_top30 < 1192
        - 0.08825234 * max(0.0, 0.02675364 - Q.girth2_top15) / 0.01959938   # -8.8%  girth2_top15 < 0.02675
        + 0.07843519 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +7.8%  lam1 < 0.007259
        - 0.07458487 * max(0.0, Q.lam1 - 0.006716737) / 0.003087986   # -7.5%  lam1 > 0.006717
        - 0.05215545 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -5.2%  girth2_top20 < 0.01083
        - 0.03617122 * max(0.0, Q.girth2 - 0.0258669) / 0.0004653552   # -3.6%  girth2 > 0.02587
        - 0.02964248 * max(0.0, 0.007259287 - Q.lam1) * max(0.0, Q.sum_pt - 1085.125) / 0.08770494   # -3.0%  lam1 < 0.007259 and sum_pt > 1085
        - 0.02709148 * max(0.0, Q.e2_sq - 0.01989454) / 0.001200608   # -2.7%  e2_sq > 0.01989
        + 0.02467863 * max(0.0, 0.005712208 - Q.girth2_top40) / 0.001219686   # +2.5%  girth2_top40 < 0.005712
        + 0.02280521 * max(0.0, 972.4111 - Q.sum_pt_top40) / 17.57725   # +2.3%  sum_pt_top40 < 972.4
        + 0.02248832 * max(0.0, Q.e2_sq - 0.0292152) / 0.0002016952   # +2.2%  e2_sq > 0.02922
        + 0.01923242 * max(0.0, 0.005712208 - Q.girth2_top40) * max(0.0, Q.log_sum_pt - 7.017258) / 3.653263e-05   # +1.9%  girth2_top40 < 0.005712 and log_sum_pt > 7.017
        - 0.01713464 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # -1.7%  tau1 < 0.06311
        - 0.01375581 * max(0.0, 0.004169954 - Q.girth2_top15) / 0.001130716   # -1.4%  girth2_top15 < 0.00417
        + 0.01339608 * max(0.0, 6.903423 - Q.log_sum_pt) * max(0.0, 0.4479367 - Q.z_dr_0p1_0p2) / 0.004178024   # +1.3%  log_sum_pt < 6.903 and z_dr_0p1_0p2 < 0.4479
        + 0.01107435 * max(0.0, Q.LHA - 0.3332345) * max(0.0, 1042.609 - Q.sum_pt) / 0.8026572   # +1.1%  LHA > 0.3332 and sum_pt < 1043
        + 0.01077571 * max(0.0, 0.007259287 - Q.lam1) * max(0.0, Q.n_particles - 36.0) / 0.01882539   # +1.1%  lam1 < 0.007259 and n_particles > 36
        - 0.01055634 * max(0.0, 972.4111 - Q.sum_pt_top40) * max(0.0, 0.0369869 - Q.tau3) / 0.1284345   # -1.1%  sum_pt_top40 < 972.4 and tau3 < 0.03699
        - 0.009143876 * max(0.0, 0.005712208 - Q.girth2_top40) * max(0.0, 1007.788 - Q.sum_pt) / 0.0192656   # -0.9%  girth2_top40 < 0.005712 and sum_pt < 1008
        - 0.008456039 * max(0.0, Q.girth2_top15 - 0.02146578) / 0.0006182335   # -0.8%  girth2_top15 > 0.02147
        - 0.007115677 * max(0.0, 972.4111 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.106445) / 11.84488   # -0.7%  sum_pt_top40 < 972.4 and soft4_pt > 1.106
        - 0.007065139 * max(0.0, Q.girth - 0.1207452) / 0.004285542   # -0.7%  girth > 0.1207
        - 0.007040332 * max(0.0, Q.e2_sq - 0.0292152) * max(0.0, 1115.723 - Q.sum_pt) / 0.02902816   # -0.7%  e2_sq > 0.02922 and sum_pt < 1116
        + 0.006360557 * max(0.0, Q.girth - 0.1207452) * max(0.0, 0.4606099 - Q.sj3_pairmin_over_m) / 0.0004267265   # +0.6%  girth > 0.1207 and sj3_pairmin_over_m < 0.4606
        + 0.005673895 * max(0.0, Q.e3 - 0.0005178279) / 8.375623e-06   # +0.6%  e3 > 0.0005178
        + 0.005526253 * max(0.0, 26.0 - Q.n_pt_above_1) / 0.8081244   # +0.6%  n_pt_above_1 < 26
        + 0.005212644 * max(0.0, 0.004169954 - Q.girth2_top15) * max(0.0, Q.n_dr_0p2_0p4 - 8.0) / 0.001650576   # +0.5%  girth2_top15 < 0.00417 and n_dr_0p2_0p4 > 8
        + 0.003231598 * max(0.0, Q.z_dr_0_0p05 - 0.878906) * max(0.0, Q.sum_pt_top40 - 1013.042) / 0.9125551   # +0.3%  z_dr_0_0p05 > 0.8789 and sum_pt_top40 > 1013
        + 0.002161534 * max(0.0, 6.903423 - Q.log_sum_pt) * max(0.0, Q.soft4_pt - 1.789258) / 0.002713428   # +0.2%  log_sum_pt < 6.903 and soft4_pt > 1.789
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 14.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.24453 * (0.1944569
        - 0.2462969 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -24.6%  girth < 0.1207
        + 0.08372375 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # +8.4%  tau1 < 0.1507
        + 0.0723857 * max(0.0, 0.3719813 - Q.LHA) / 0.1170757   # +7.2%  LHA < 0.372
        + 0.05441279 * max(0.0, Q.e2 - 0.006720044) / 0.02422374   # +5.4%  e2 > 0.00672
        - 0.05153653 * max(0.0, Q.girth2_top20 - 0.005718442) / 0.003749064   # -5.2%  girth2_top20 > 0.005718
        - 0.04228475 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -4.2%  e2_sq < 0.009606
        + 0.03681235 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +3.7%  e2_sq < 0.008184
        + 0.03613294 * max(0.0, Q.girth2_top20 - 0.008031209) / 0.002906852   # +3.6%  girth2_top20 > 0.008031
        + 0.03046306 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +3.0%  z_dr_0p2_0p4 < 0.09123
        - 0.02962688 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # -3.0%  e3 < 0.0005178
        - 0.02477406 * max(0.0, Q.sum_pt_top3 - 309.625) / 165.6552   # -2.5%  sum_pt_top3 > 309.6
        - 0.02465665 * max(0.0, 0.03577037 - Q.girth) / 0.004182456   # -2.5%  girth < 0.03577
        + 0.02449743 * max(0.0, 0.006403325 - Q.width) / 0.001406912   # +2.4%  width < 0.006403
        - 0.02258438 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -2.3%  n_dr_0p2_0p4 < 15
        - 0.02144723 * max(0.0, Q.girth2 - 0.01397874) / 0.002228371   # -2.1%  girth2 > 0.01398
        - 0.02128892 * max(0.0, 0.01976735 - Q.girth2_top10) / 0.01371247   # -2.1%  girth2_top10 < 0.01977
        + 0.01965392 * max(0.0, 0.00592208 - Q.e2_sq) / 0.001192334   # +2.0%  e2_sq < 0.005922
        - 0.01911062 * max(0.0, 0.007678544 - Q.girth2_top10) / 0.00347358   # -1.9%  girth2_top10 < 0.007679
        + 0.01564301 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +1.6%  dr_0 < 0.06413
        - 0.01410915 * max(0.0, 0.3062621 - Q.tau21_b2) / 0.08794306   # -1.4%  tau21_b2 < 0.3063
        + 0.01267287 * max(0.0, 0.08222447 - Q.M2) / 0.02103373   # +1.3%  M2 < 0.08222
        + 0.01260907 * max(0.0, 2.410481 - Q.D2) / 0.587301   # +1.3%  D2 < 2.41
        + 0.01205635 * max(0.0, 656.3844 - Q.sum_pt_top3) / 209.0067   # +1.2%  sum_pt_top3 < 656.4
        - 0.01131935 * max(0.0, Q.e2 - 0.006720044) * max(0.0, Q.log_sum_pt - 6.893714) / 0.001240306   # -1.1%  e2 > 0.00672 and log_sum_pt > 6.894
        + 0.008759378 * max(0.0, Q.n_dr_0_0p05 - 6.0) / 7.551284   # +0.9%  n_dr_0_0p05 > 6
        - 0.008535502 * max(0.0, 0.005402331 - Q.girth2_top30) / 0.001231293   # -0.9%  girth2_top30 < 0.005402
        - 0.007816228 * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0004356739   # -0.8%  psi_0p3 > 0.9985
        - 0.007450995 * max(0.0, Q.z_dr_0_0p05 - 0.7128619) / 0.07526795   # -0.7%  z_dr_0_0p05 > 0.7129
        - 0.006777055 * max(0.0, Q.e2_sq - 0.02580859) / 0.0004635665   # -0.7%  e2_sq > 0.02581
        + 0.005658405 * max(0.0, Q.girth2 - 0.01397874) * max(0.0, 114.75 - Q.pt_3) / 0.1232701   # +0.6%  girth2 > 0.01398 and pt_3 < 114.8
        - 0.005180396 * max(0.0, 0.002247756 - Q.girth2_top10) / 0.0005836006   # -0.5%  girth2_top10 < 0.002248
        + 0.004823397 * max(0.0, Q.e2 - 0.006720044) * max(0.0, Q.sj3_pairmin_over_m - 0.1765064) / 0.002982204   # +0.5%  e2 > 0.00672 and sj3_pairmin_over_m > 0.1765
        + 0.003879356 * max(0.0, Q.girth2_top40 - 0.02497133) / 0.0004755075   # +0.4%  girth2_top40 > 0.02497
        - 0.001020632 * max(0.0, Q.girth2 - 0.02928196) / 0.0002029538   # -0.1%  girth2 > 0.02928
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 10.33;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.32645 * (0.09688503
        + 0.1678395 * max(0.0, 0.00818374 - Q.e2_sq) / 0.002451404   # +16.8%  e2_sq < 0.008184
        - 0.1058809 * max(0.0, 0.006363916 - Q.girth2_top30) / 0.001678624   # -10.6%  girth2_top30 < 0.006364
        + 0.09649265 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # +9.6%  girth2_top30 < 0.01216
        - 0.06738977 * max(0.0, 0.00818374 - Q.e2_sq) * max(0.0, 1115.723 - Q.sum_pt) / 0.1997244   # -6.7%  e2_sq < 0.008184 and sum_pt < 1116
        - 0.06416239 * max(0.0, Q.n_real_top50 - 22.0) / 19.58685   # -6.4%  n_real_top50 > 22
        - 0.05037645 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # -5.0%  e2_sq < 0.009606
        + 0.04594995 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +4.6%  e2 < 0.02516
        + 0.038942 * max(0.0, 0.005809485 - Q.girth2_top30) / 0.001402136   # +3.9%  girth2_top30 < 0.005809
        - 0.03676705 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -3.7%  lam1 < 0.005914
        - 0.0361538 * max(0.0, 0.00634935 - Q.girth2_top50) / 0.001425305   # -3.6%  girth2_top50 < 0.006349
        + 0.03420075 * max(0.0, 0.005913555 - Q.lam1) * max(0.0, 1115.723 - Q.sum_pt) / 0.1207895   # +3.4%  lam1 < 0.005914 and sum_pt < 1116
        + 0.0338153 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +3.4%  n_dr_0p2_0p4 < 10
        - 0.03351082 * max(0.0, 0.07374472 - Q.girth) / 0.01915593   # -3.4%  girth < 0.07374
        - 0.03278899 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # -3.3%  e2 < 0.03876
        + 0.02482421 * max(0.0, 0.009606007 - Q.e2_sq) * max(0.0, 1115.723 - Q.sum_pt) / 0.2875932   # +2.5%  e2_sq < 0.009606 and sum_pt < 1116
        + 0.02287817 * max(0.0, 0.3332345 - Q.LHA) / 0.08501989   # +2.3%  LHA < 0.3332
        - 0.02215268 * max(0.0, 4.450169 - Q.D2) / 2.013555   # -2.2%  D2 < 4.45
        - 0.02129203 * max(0.0, Q.z_top30_slots - 0.9734886) / 0.01006171   # -2.1%  z_top30_slots > 0.9735
        - 0.01685111 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 1156.659 - Q.sum_pt_top50) / 469.6702   # -1.7%  n_dr_0p2_0p4 < 10 and sum_pt_top50 < 1157
        + 0.01432124 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 43.23581   # +1.4%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21
        - 0.009527266 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2) / 0.005195774   # -1.0%  n_dr_0p2_0p4 < 10 and girth2 < 0.005532
        + 0.008630209 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 1.08774 - Q.D2_b2) / 0.003455048   # +0.9%  z_top30_slots > 0.9735 and D2_b2 < 1.088
        - 0.007854441 * max(0.0, Q.C2_b2 - 0.002289486) / 0.01267025   # -0.8%  C2_b2 > 0.002289
        - 0.007398304 * max(0.0, 0.02210818 - Q.e2) / 0.00342102   # -0.7%  e2 < 0.02211
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 8.225;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.22485 * (-0.1110782
        + 0.2389719 * max(0.0, 0.006403325 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.811175) / 0.0002246937   # +23.9%  girth2 < 0.006403 and log_sum_pt > 6.811
        - 0.219161 * max(0.0, 0.007511864 - Q.e2_sq) * max(0.0, Q.log_sum_pt - 6.811175) / 0.0003183593   # -21.9%  e2_sq < 0.007512 and log_sum_pt > 6.811
        + 0.1848611 * max(0.0, 0.007511864 - Q.e2_sq) / 0.002016671   # +18.5%  e2_sq < 0.007512
        - 0.05991811 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -6.0%  psi_0p3 > 0.9943
        - 0.05411293 * max(0.0, 0.002752094 - Q.lam1) / 0.0003912817   # -5.4%  lam1 < 0.002752
        - 0.04871631 * max(0.0, 0.005809485 - Q.girth2_top30) / 0.001402136   # -4.9%  girth2_top30 < 0.005809
        + 0.04419157 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +4.4%  psi_0p3 > 0.9897
        - 0.03792792 * max(0.0, Q.sum_pt - 1085.125) / 25.7173   # -3.8%  sum_pt > 1085
        + 0.03622842 * max(0.0, Q.psi_0p3 - 0.9943058) * max(0.0, 0.00751625 - Q.girth2) / 6.462915e-06   # +3.6%  psi_0p3 > 0.9943 and girth2 < 0.007516
        + 0.02524679 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +2.5%  n_dr_0p2_0p4 < 10
        - 0.01062277 * max(0.0, 0.03502997 - Q.z_dr_0p1_0p2) / 0.007341504   # -1.1%  z_dr_0p1_0p2 < 0.03503
        + 0.006494684 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +0.6%  LHA > 0.372
        + 0.006272182 * max(0.0, 0.08133662 - Q.psi_0p1) * max(0.0, Q.sum_pt_top50 - 889.8503) / 0.2667141   # +0.6%  psi_0p1 < 0.08134 and sum_pt_top50 > 889.9
        + 0.005709749 * max(0.0, 6.879399 - Q.log_sum_pt) / 0.01083309   # +0.6%  log_sum_pt < 6.879
        - 0.005431105 * max(0.0, Q.psi_0p1 - 0.9371031) * max(0.0, 0.05444336 - Q.absphi_0) / 0.0004737047   # -0.5%  psi_0p1 > 0.9371 and absphi_0 < 0.05444
        - 0.004922512 * max(0.0, Q.sum_pt - 1085.125) * max(0.0, 0.019523 - Q.z_dr_0p2_0p4) / 0.1925692   # -0.5%  sum_pt > 1085 and z_dr_0p2_0p4 < 0.01952
        - 0.004641189 * max(0.0, 0.08133662 - Q.psi_0p1) / 0.002484547   # -0.5%  psi_0p1 < 0.08134
        + 0.003853742 * max(0.0, Q.sum_pt - 1085.125) * max(0.0, 0.001868041 - Q.lam1) / 0.006895658   # +0.4%  sum_pt > 1085 and lam1 < 0.001868
        + 0.002716069 * max(0.0, Q.psi_0p1 - 0.9371031) * max(0.0, Q.dr_max_012 - 0.06112084) / 0.0001515273   # +0.3%  psi_0p1 > 0.9371 and dr_max_012 > 0.06112
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 16.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.29388 * (0.05221523
        + 0.23555 * max(0.0, Q.log_sum_pt - 6.811175) / 0.1383141   # +23.6%  log_sum_pt > 6.811
        - 0.07785867 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -7.8%  sum_pt < 1085
        - 0.06427342 * max(0.0, Q.sum_pt_top50 - 889.8503) / 149.655   # -6.4%  sum_pt_top50 > 889.9
        - 0.05889007 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # -5.9%  sum_pt > 1017
        - 0.0553239 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # -5.5%  log_sum_pt > 6.894
        + 0.05275166 * max(0.0, Q.girth2_top40 - 0.01292642) * max(0.0, 1245.697 - Q.sum_pt_top50) / 0.6025464   # +5.3%  girth2_top40 > 0.01293 and sum_pt_top50 < 1246
        + 0.04996793 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # +5.0%  sum_pt_top50 < 1079
        + 0.04339048 * max(0.0, Q.sum_pt_top50 - 976.277) / 72.29277   # +4.3%  sum_pt_top50 > 976.3
        - 0.03803058 * max(0.0, Q.girth2_top50 - 0.01351431) / 0.002240755   # -3.8%  girth2_top50 > 0.01351
        - 0.03468322 * max(0.0, Q.girth2_top40 - 0.008840538) / 0.003161074   # -3.5%  girth2_top40 > 0.008841
        - 0.0312667 * max(0.0, 0.1207452 - Q.girth) / 0.05578614   # -3.1%  girth < 0.1207
        - 0.02381734 * max(0.0, Q.sum_pt_top40 - 972.4111) / 67.15108   # -2.4%  sum_pt_top40 > 972.4
        - 0.02175569 * max(0.0, Q.e2_sq - 0.0292152) / 0.0002016952   # -2.2%  e2_sq > 0.02922
        + 0.0216641 * max(0.0, Q.girth2_top50 - 0.005852839) / 0.004485718   # +2.2%  girth2_top50 > 0.005853
        - 0.02063195 * max(0.0, Q.girth2_top40 - 0.02862386) * max(0.0, 62.25 - Q.pt_6) / 0.00503292   # -2.1%  girth2_top40 > 0.02862 and pt_6 < 62.25
        - 0.01975453 * max(0.0, Q.girth2_top40 - 0.01292642) / 0.002259475   # -2.0%  girth2_top40 > 0.01293
        + 0.01920834 * max(0.0, Q.sum_pt_top50 - 1013.916) / 45.94021   # +1.9%  sum_pt_top50 > 1014
        - 0.0165302 * max(0.0, 54.0 - Q.n_particles) / 11.024   # -1.7%  n_particles < 54
        + 0.01548518 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +1.5%  e2 < 0.03481
        + 0.01345683 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, Q.tau21 - 0.1295048) / 22.44785   # +1.3%  sum_pt < 1085 and tau21 > 0.1295
        - 0.01321352 * max(0.0, Q.width - 0.01991191) / 0.001207633   # -1.3%  width > 0.01991
        + 0.01274376 * max(0.0, Q.girth2_top40 - 0.02862386) / 0.0001970429   # +1.3%  girth2_top40 > 0.02862
        - 0.01134965 * max(0.0, Q.girth2_top40 - 0.02862386) * max(0.0, Q.z_6 - 0.01906139) / 3.587165e-06   # -1.1%  girth2_top40 > 0.02862 and z_6 > 0.01906
        + 0.009674115 * max(0.0, Q.e2 - 0.03680582) / 0.004603798   # +1.0%  e2 > 0.03681
        - 0.009196234 * max(0.0, Q.girth2_top40 - 0.02862386) * max(0.0, Q.pt_6 - 19.46875) / 0.003416206   # -0.9%  girth2_top40 > 0.02862 and pt_6 > 19.47
        - 0.007671486 * max(0.0, 795.6047 - Q.sum_pt_top15) / 30.63834   # -0.8%  sum_pt_top15 < 795.6
        - 0.007534869 * max(0.0, 25.0 - Q.n_pt_above_5) / 2.617361   # -0.8%  n_pt_above_5 < 25
        + 0.007307632 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # +0.7%  psi_0p3 > 0.998
        + 0.003385431 * max(0.0, Q.sum_pt_top10 - 867.9266) / 28.75627   # +0.3%  sum_pt_top10 > 867.9
        + 0.002195902 * max(0.0, 25.0 - Q.n_pt_above_5) * max(0.0, 5.378975 - Q.D2) / 4.808335   # +0.2%  n_pt_above_5 < 25 and D2 < 5.379
        + 0.001436577 * max(0.0, Q.sum_pt_top20 - 1064.139) / 11.11947   # +0.1%  sum_pt_top20 > 1064
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 20.28;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.28421 * (-0.02824007
        - 0.1300996 * max(0.0, 0.008190222 - Q.girth2) / 0.002454223   # -13.0%  girth2 < 0.00819
        + 0.09026456 * max(0.0, 0.01989454 - Q.e2_sq) / 0.01180402   # +9.0%  e2_sq < 0.01989
        + 0.06564358 * max(0.0, 0.01215787 - Q.girth2_top30) * max(0.0, Q.sum_pt - 907.9372) / 0.9148907   # +6.6%  girth2_top30 < 0.01216 and sum_pt > 907.9
        + 0.06358127 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +6.4%  psi_0p3 > 0.9897
        - 0.0576154 * max(0.0, 0.01083435 - Q.girth2_top20) / 0.00529825   # -5.8%  girth2_top20 < 0.01083
        + 0.05443908 * max(0.0, 0.01396296 - Q.e2_sq) / 0.00689253   # +5.4%  e2_sq < 0.01396
        - 0.05031729 * max(0.0, 0.01951641 - Q.girth2_top50) / 0.01159099   # -5.0%  girth2_top50 < 0.01952
        - 0.04635323 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -4.6%  tau1 < 0.05445
        - 0.04471536 * max(0.0, 0.00616708 - Q.e2_sq) / 0.001295555   # -4.5%  e2_sq < 0.006167
        - 0.04455315 * max(0.0, 0.008376291 - Q.girth2_top30) / 0.002981077   # -4.5%  girth2_top30 < 0.008376
        - 0.04357155 * max(0.0, 0.01215787 - Q.girth2_top30) / 0.005935412   # -4.4%  girth2_top30 < 0.01216
        + 0.03805468 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +3.8%  e2 < 0.03481
        + 0.03277891 * max(0.0, 0.00751625 - Q.width) / 0.002018105   # +3.3%  width < 0.007516
        - 0.03097204 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, 1225.842 - Q.sum_pt_top40) / 0.3755165   # -3.1%  psi_0p3 > 0.9956 and sum_pt_top40 < 1226
        - 0.02713144 * max(0.0, 0.03480688 - Q.e2) * max(0.0, Q.sum_pt_top30 - 886.3438) / 1.374127   # -2.7%  e2 < 0.03481 and sum_pt_top30 > 886.3
        + 0.02110184 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +2.1%  psi_0p3 > 0.9956
        + 0.02108589 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # +2.1%  tau1 < 0.07057
        - 0.0208762 * max(0.0, 0.2091025 - Q.LHA) / 0.02095782   # -2.1%  LHA < 0.2091
        - 0.02001393 * max(0.0, 0.1008497 - Q.tau1) / 0.02962944   # -2.0%  tau1 < 0.1008
        + 0.01667532 * max(0.0, 0.006374178 - Q.girth2_top20) / 0.001987751   # +1.7%  girth2_top20 < 0.006374
        + 0.01564088 * max(0.0, 0.009606007 - Q.e2_sq) / 0.003497866   # +1.6%  e2_sq < 0.009606
        - 0.0130371 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # -1.3%  n_dr_0p1_0p2 < 17
        + 0.01181701 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +1.2%  tau21_b2 < 0.3425
        - 0.008359869 * max(0.0, 0.3572263 - Q.N2) / 0.07385171   # -0.8%  N2 < 0.3572
        + 0.006944615 * max(0.0, 0.01879315 - Q.e2) / 0.002355308   # +0.7%  e2 < 0.01879
        - 0.005813191 * max(0.0, 0.001155057 - Q.girth2_top3) / 0.000329166   # -0.6%  girth2_top3 < 0.001155
        - 0.005360219 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.02048524 - Q.lam1) / 0.001304679   # -0.5%  tau21_b2 < 0.3425 and lam1 < 0.02049
        - 0.003832582 * max(0.0, Q.psi_0p3 - 0.9896594) * max(0.0, 0.1438306 - Q.z_1) / 0.0001672529   # -0.4%  psi_0p3 > 0.9897 and z_1 < 0.1438
        - 0.00365229 * max(0.0, Q.LHA - 0.3098384) / 0.0195892   # -0.4%  LHA > 0.3098
        - 0.00232219 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.2%  log_sum_pt > 7.063
        + 0.001735432 * max(0.0, 0.01396296 - Q.e2_sq) * max(0.0, 0.9906378 - Q.z_top50_slots) / 1.282821e-05   # +0.2%  e2_sq < 0.01396 and z_top50_slots < 0.9906
        - 0.001640268 * max(0.0, Q.n_pt_above_10 - 28.0) / 0.3744807   # -0.2%  n_pt_above_10 > 28
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 4.417;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.417124 * (-0.01685164
        - 0.1161739 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -11.6%  tau1 < 0.07057
        + 0.1124183 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +11.2%  girth2_top5 < 0.00833
        - 0.09440707 * max(0.0, Q.sum_pt_top30 - 988.4375) / 43.28413   # -9.4%  sum_pt_top30 > 988.4
        + 0.09015996 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # +9.0%  e2 < 0.04083
        + 0.06180263 * max(0.0, Q.sum_pt_top20 - 869.693) / 87.9958   # +6.2%  sum_pt_top20 > 869.7
        + 0.06074159 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +6.1%  z_dr_0p1_0p2 < 0.1203
        - 0.04995402 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 184.0097   # -5.0%  n_dr_0p2_0p4 < 26 and n_dr_0p1_0p2 < 21
        - 0.0472937 * max(0.0, 0.9860575 - Q.z_top30_slots) / 0.03906901   # -4.7%  z_top30_slots < 0.9861
        + 0.04694633 * max(0.0, Q.sum_pt_top50 - 1156.659) / 13.71464   # +4.7%  sum_pt_top50 > 1157
        + 0.04458752 * max(0.0, 0.1778185 - Q.sd_rg) * max(0.0, Q.soft5_dr0 - 0.03721986) / 0.009661509   # +4.5%  sd_rg < 0.1778 and soft5_dr0 > 0.03722
        - 0.0438711 * max(0.0, 0.1778185 - Q.sd_rg) * max(0.0, Q.soft5_dr - 0.04139378) / 0.009084795   # -4.4%  sd_rg < 0.1778 and soft5_dr > 0.04139
        + 0.03840941 * max(0.0, 0.08068193 - Q.girth) / 0.02368455   # +3.8%  girth < 0.08068
        - 0.03619003 * max(0.0, 0.002270363 - Q.girth2_top5) / 0.0007634099   # -3.6%  girth2_top5 < 0.00227
        - 0.03512178 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -3.5%  psi_0p1 > 0.8976
        + 0.02485866 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 0.002396991 - Q.lam2) / 8.322782e-05   # +2.5%  z_dr_0p1_0p2 < 0.1203 and lam2 < 0.002397
        + 0.0217435 * max(0.0, Q.sj2_dr - 0.2232169) / 0.02415244   # +2.2%  sj2_dr > 0.2232
        + 0.02070783 * max(0.0, 10.0 - Q.n_dr_0p1_0p2) / 2.171439   # +2.1%  n_dr_0p1_0p2 < 10
        + 0.01665275 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +1.7%  tau1 < 0.06311
        - 0.01595459 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # -1.6%  log_sum_pt > 7.139
        - 0.01317291 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 10.0) / 0.006625563   # -1.3%  girth2_top5 < 0.00833 and n_dr_0p2_0p4 > 10
        - 0.008832399 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 986.0565 - Q.sum_pt) / 0.5137322   # -0.9%  z_dr_0p1_0p2 < 0.1203 and sum_pt < 986.1
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6442422268907563, 2.862663182773109, 0.2551949579831933, 0.43670241596638654, 0.7702443277310924, 1.0382088235294118, 0.4652573529411765, 0.40610210084033616, 1.1791548319327732, 1.1214975840336134, 1.1032322478991596, 0.8270796218487395, 0.5636460084033613, 1.4811573529411766, 0.26707064075630255, 0.382488025210084]
T = [3.9522551240808825, 2.555836270680147, 4.25820376509979, 4.201851680672269, 3.8576490217962185]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +45%, n4 -17%, n9 +15%, n3 -8%, n5 -8%, n12 +3% ...
            + 0.4526946 * h[1] / H_AVG[1]
            - 0.1705264 * h[4] / H_AVG[4]
            + 0.1463145 * h[9] / H_AVG[9]
            - 0.08287087 * h[3] / H_AVG[3]
            - 0.08208991 * h[5] / H_AVG[5]
            + 0.0334251 * h[12] / H_AVG[12]
            + 0.01839367 * h[6] / H_AVG[6]
            + 0.009323434 * h[8] / H_AVG[8]
            - 0.004361562 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -32%, n9 +25%, n1 -21%, n11 -8%, n12 +8%, n6 +4% ...
            - 0.3202023 * h[4] / H_AVG[4]
            + 0.2468243 * h[9] / H_AVG[9]
            - 0.2100093 * h[1] / H_AVG[1]
            - 0.08090108 * h[11] / H_AVG[11]
            + 0.07580819 * h[12] / H_AVG[12]
            + 0.03982064 * h[6] / H_AVG[6]
            + 0.01248099 * h[2] / H_AVG[2]
            + 0.007208715 * h[8] / H_AVG[8]
            - 0.006744565 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +14%, n11 +12%, n0 +11%, n14 -9%, n4 +6% ...
            - 0.2422995 * h[8] / H_AVG[8]
            + 0.1409549 * h[5] / H_AVG[5]
            + 0.1153253 * h[11] / H_AVG[11]
            + 0.1134708 * h[0] / H_AVG[0]
            - 0.08623874 * h[14] / H_AVG[14]
            + 0.06217915 * h[4] / H_AVG[4]
            - 0.05960584 * h[7] / H_AVG[7]
            - 0.05761293 * h[9] / H_AVG[9]
            - 0.05377413 * h[12] / H_AVG[12]
            + 0.04486805 * h[3] / H_AVG[3]
            - 0.01684196 * h[15] / H_AVG[15]
            + 0.006828838 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -21%, n5 +17%, n6 -11%, n7 +9%, n15 +5% ...
            - 0.2630882 * h[8] / H_AVG[8]
            - 0.2108197 * h[0] / H_AVG[0]
            + 0.16987 * h[5] / H_AVG[5]
            - 0.1072665 * h[6] / H_AVG[6]
            + 0.08758758 * h[7] / H_AVG[7]
            + 0.0512035 * h[15] / H_AVG[15]
            + 0.04546979 * h[3] / H_AVG[3]
            + 0.04191947 * h[12] / H_AVG[12]
            - 0.02277522 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -35%, n10 +28%, n5 -13%, n8 +6%, n12 -5%, n4 +4% ...
            - 0.3479577 * h[13] / H_AVG[13]
            + 0.2815171 * h[10] / H_AVG[10]
            - 0.1261547 * h[5] / H_AVG[5]
            + 0.06447657 * h[8] / H_AVG[8]
            - 0.05479173 * h[12] / H_AVG[12]
            + 0.03743752 * h[4] / H_AVG[4]
            - 0.03718146 * h[15] / H_AVG[15]
            - 0.02960773 * h[7] / H_AVG[7]
            + 0.02087548 * h[0] / H_AVG[0]
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
